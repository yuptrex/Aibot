import io
import logging

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes

import cv_filters
import ai_styles
import db

logger = logging.getLogger(__name__)

# Daily per-user cap on AI (Gemini) requests — protects your shared free-tier quota.
# OpenCV styles (cartoon/sketch/oil) are NOT counted against this since they're free.
DAILY_AI_LIMIT_PER_USER = 15

CV_STYLES = {
    "cartoon": "🖍️ Cartoon",
    "sketch": "✏️ Pencil Sketch",
    "sketch_color": "🎨 Color Sketch",
}

STYLE_MENU = [
    [
        InlineKeyboardButton("🎨 Ghibli-inspired", callback_data="ai:ghibli"),
        InlineKeyboardButton("✨ Anime", callback_data="ai:anime"),
    ],
    [
        InlineKeyboardButton("🖌️ Watercolor", callback_data="ai:watercolor"),
        InlineKeyboardButton("💥 Comic Book", callback_data="ai:comic"),
    ],
    [
        InlineKeyboardButton("🖍️ Cartoon (instant)", callback_data="cv:cartoon"),
        InlineKeyboardButton("✏️ Sketch (instant)", callback_data="cv:sketch"),
    ],
    [
        InlineKeyboardButton("🎨 Color Sketch (instant)", callback_data="cv:sketch_color"),
    ],
]


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Send me a photo and I'll turn it into different art styles!\n\n"
        "AI styles (Ghibli-inspired, Anime, Watercolor, Comic) use a limited daily quota. "
        "Instant styles (Cartoon, Sketch) are unlimited and processed locally."
    )


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    photo = update.message.photo[-1]  # highest resolution
    file = await photo.get_file()

    # Save the file_id so the callback handler can fetch it again later
    context.user_data["last_photo_file_id"] = file.file_id

    await update.message.reply_text(
        "Choose a style:", reply_markup=InlineKeyboardMarkup(STYLE_MENU)
    )


async def handle_style_choice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    user = query.from_user
    file_id = context.user_data.get("last_photo_file_id")

    if not file_id:
        await query.edit_message_text(
            "I don't have a photo to work with — please send a new photo first."
        )
        return

    kind, style = query.data.split(":", 1)

    tg_file = await context.bot.get_file(file_id)
    image_buf = io.BytesIO()
    await tg_file.download_to_memory(image_buf)
    image_bytes = image_buf.getvalue()

    if kind == "cv":
        await query.edit_message_text(f"Processing {CV_STYLES[style]}... ⚡")
        try:
            if style == "cartoon":
                result = cv_filters.cartoonify(image_bytes)
            elif style == "sketch":
                result = cv_filters.pencil_sketch(image_bytes, color=False)
            elif style == "sketch_color":
                result = cv_filters.pencil_sketch(image_bytes, color=True)
            else:
                raise ValueError(f"Unknown CV style {style}")

            await context.bot.send_photo(
                chat_id=query.message.chat_id,
                photo=io.BytesIO(result),
                caption=f"{CV_STYLES[style]} ✅",
            )
            db.log_request(user.id, user.username, style, success=True)
        except Exception as e:
            logger.exception("CV style processing failed")
            await query.message.reply_text("Something went wrong processing that image. Try another photo.")
            db.log_request(user.id, user.username, style, success=False, error=str(e))
        return

    # AI (Gemini) style — check per-user daily quota first
    if not db.check_and_increment_usage(user.id, DAILY_AI_LIMIT_PER_USER):
        await query.edit_message_text(
            f"You've hit your daily limit of {DAILY_AI_LIMIT_PER_USER} AI-style images. "
            "Try an instant style (Cartoon/Sketch) or come back tomorrow!"
        )
        return

    label = ai_styles.STYLE_LABELS.get(style, style)
    await query.edit_message_text(f"Generating {label} style with AI... 🪄 (this can take 10-20s)")

    try:
        result = ai_styles.apply_ai_style(image_bytes, style)
        await context.bot.send_photo(
            chat_id=query.message.chat_id,
            photo=io.BytesIO(result),
            caption=f"{label} ✅",
        )
        db.log_request(user.id, user.username, style, success=True)
    except ai_styles.QuotaExceededError as e:
        await query.message.reply_text(
            "⚠️ The AI service is rate-limited right now. Please try again in a few minutes."
        )
        db.log_request(user.id, user.username, style, success=False, error=str(e))
    except Exception as e:
        logger.exception("AI style processing failed")
        await query.message.reply_text(
            "Something went wrong generating that style. This can happen if the photo "
            "triggered a content filter — try a different photo or style."
        )
        db.log_request(user.id, user.username, style, success=False, error=str(e))


async def usage_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    count = db.get_user_usage_today(user.id)
    await update.message.reply_text(
        f"You've used {count}/{DAILY_AI_LIMIT_PER_USER} AI-style generations today.\n"
        "Instant styles (Cartoon/Sketch) don't count toward this limit."
    )
