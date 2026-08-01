import io
import logging

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes

import cv_filters
import db

logger = logging.getLogger(__name__)

# All styles are instant, local OpenCV/PIL processing — no API calls, no
# rate limits, no cost. Nothing here ever calls out to the network.
STYLES = {
    "cartoon":       ("🖍️ Cartoon",        lambda b: cv_filters.cartoonify(b)),
    "sketch":        ("✏️ Pencil Sketch",   lambda b: cv_filters.pencil_sketch(b, color=False)),
    "sketch_color":  ("🎨 Color Sketch",    lambda b: cv_filters.pencil_sketch(b, color=True)),
    "oil":           ("🖼️ Oil Painting",    lambda b: cv_filters.oil_painting(b)),
    "sepia":         ("📜 Sepia",           lambda b: cv_filters.sepia(b)),
    "bw":            ("⚫ Black & White",    lambda b: cv_filters.black_and_white(b)),
    "negative":      ("🌗 Negative",         lambda b: cv_filters.negative(b)),
    "emboss":        ("🪙 Emboss",           lambda b: cv_filters.emboss(b)),
    "hdr":           ("✨ HDR Glow",         lambda b: cv_filters.hdr_glow(b)),
    "vintage":       ("🎞️ Warm Vintage",    lambda b: cv_filters.warm_vintage(b)),
}

STYLE_MENU = [
    [
        InlineKeyboardButton(STYLES["cartoon"][0], callback_data="cv:cartoon"),
        InlineKeyboardButton(STYLES["sketch"][0], callback_data="cv:sketch"),
    ],
    [
        InlineKeyboardButton(STYLES["sketch_color"][0], callback_data="cv:sketch_color"),
        InlineKeyboardButton(STYLES["oil"][0], callback_data="cv:oil"),
    ],
    [
        InlineKeyboardButton(STYLES["sepia"][0], callback_data="cv:sepia"),
        InlineKeyboardButton(STYLES["bw"][0], callback_data="cv:bw"),
    ],
    [
        InlineKeyboardButton(STYLES["negative"][0], callback_data="cv:negative"),
        InlineKeyboardButton(STYLES["emboss"][0], callback_data="cv:emboss"),
    ],
    [
        InlineKeyboardButton(STYLES["hdr"][0], callback_data="cv:hdr"),
        InlineKeyboardButton(STYLES["vintage"][0], callback_data="cv:vintage"),
    ],
]


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Send me a photo and I'll turn it into different art styles!\n\n"
        "All styles are processed instantly, right here — no AI, no waiting, "
        "no daily limits."
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

    _, style = query.data.split(":", 1)

    if style not in STYLES:
        await query.edit_message_text("Unknown style — please send the photo again.")
        return

    label, filter_fn = STYLES[style]

    tg_file = await context.bot.get_file(file_id)
    image_buf = io.BytesIO()
    await tg_file.download_to_memory(image_buf)
    image_bytes = image_buf.getvalue()

    await query.edit_message_text(f"Processing {label}... ⚡")
    try:
        result = filter_fn(image_bytes)
        await context.bot.send_photo(
            chat_id=query.message.chat_id,
            photo=io.BytesIO(result),
            caption=f"{label} ✅\n\nWant to see it in another style?",
            reply_markup=InlineKeyboardMarkup(STYLE_MENU),
        )
        db.log_request(user.id, user.username, style, success=True)
    except Exception as e:
        logger.exception("Style processing failed")
        await query.message.reply_text("Something went wrong processing that image. Try another photo.")
        db.log_request(user.id, user.username, style, success=False, error=str(e))


async def usage_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    count = db.get_user_usage_today(user.id)
    await update.message.reply_text(
        f"You've generated {count} styled image(s) today. All styles are instant "
        "and unlimited — no daily cap."
    )
