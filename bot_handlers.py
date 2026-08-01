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
    "cartoon":       ("🖍️ Cartoon",        lambda b, pct=None: cv_filters.cartoonify(b)),
    "sketch":        ("✏️ Pencil Sketch",   lambda b, pct=None: cv_filters.pencil_sketch(b, color=False)),
    "sketch_color":  ("🎨 Color Sketch",    lambda b, pct=None: cv_filters.pencil_sketch(b, color=True)),
    "oil":           ("🖼️ Oil Painting",    lambda b, pct=None: cv_filters.oil_painting(b)),
    "sepia":         ("📜 Sepia",           lambda b, pct=None: cv_filters.sepia(b)),
    "bw":            ("⚫ Black & White",    lambda b, pct=None: cv_filters.black_and_white(b)),
    "negative":      ("🌗 Negative",         lambda b, pct=None: cv_filters.negative(b)),
    "emboss":        ("🪙 Emboss",           lambda b, pct=None: cv_filters.emboss(b)),
    "hdr":           ("✨ HDR Glow",         lambda b, pct=None: cv_filters.hdr_glow(b)),
    "vintage":       ("🎞️ Warm Vintage",    lambda b, pct=None: cv_filters.warm_vintage(b)),
    "blur":          ("🌫️ Aesthetic Blur",  lambda b, pct=50: cv_filters.aesthetic_blur(b, intensity=pct)),
    "blur_bw":       ("🌫️⚫ Aesthetic Blur B/W", lambda b, pct=50: cv_filters.aesthetic_blur(b, intensity=pct, grayscale=True)),
    "glow":          ("🌅 Aesthetic Glow",  lambda b, pct=100: cv_filters.aesthetic_glow(b, intensity=pct)),
}

# Styles listed here prompt the user for a 1-100 intensity value before running.
NEEDS_PERCENT = {"blur", "blur_bw", "glow"}

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
    [
        InlineKeyboardButton(STYLES["blur"][0], callback_data="cv:blur"),
        InlineKeyboardButton(STYLES["glow"][0], callback_data="cv:glow"),
    ],
    [
        InlineKeyboardButton(STYLES["blur_bw"][0], callback_data="cv:blur_bw"),
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

    # Always store the ORIGINAL uploaded photo's file_id. Every restyle,
    # no matter how many times the user taps another style button, re-fetches
    # from this same original — never from a previously styled result.
    context.user_data["original_photo_file_id"] = file.file_id
    context.user_data.pop("awaiting_percent_for", None)

    await update.message.reply_text(
        "Choose a style:", reply_markup=InlineKeyboardMarkup(STYLE_MENU)
    )


async def _run_style(chat_id, context: ContextTypes.DEFAULT_TYPE, user, file_id, style, pct=None):
    label, filter_fn = STYLES[style]

    tg_file = await context.bot.get_file(file_id)
    image_buf = io.BytesIO()
    await tg_file.download_to_memory(image_buf)
    image_bytes = image_buf.getvalue()

    processing_msg = await context.bot.send_message(chat_id=chat_id, text=f"Processing {label}... ⚡")
    try:
        result = filter_fn(image_bytes, pct) if pct is not None else filter_fn(image_bytes)
        caption = f"{label} ✅" + (f" ({pct}%)" if pct is not None else "")
        await context.bot.send_photo(
            chat_id=chat_id,
            photo=io.BytesIO(result),
            caption=f"{caption}\n\nWant to see it in another style?",
            reply_markup=InlineKeyboardMarkup(STYLE_MENU),
        )
        db.log_request(user.id, user.username, style, success=True)
    except Exception as e:
        logger.exception("Style processing failed")
        await context.bot.send_message(
            chat_id=chat_id, text="Something went wrong processing that image. Try another photo."
        )
        db.log_request(user.id, user.username, style, success=False, error=str(e))
    finally:
        try:
            await processing_msg.delete()
        except Exception:
            pass


async def handle_style_choice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    user = query.from_user
    file_id = context.user_data.get("original_photo_file_id")
    chat_id = query.message.chat_id

    if not file_id:
        # query.message may be a photo (no editable text) or a text message —
        # reply_text always works safely on either, unlike edit_message_text.
        await context.bot.send_message(
            chat_id=chat_id,
            text="I don't have a photo to work with — please send a new photo first.",
        )
        return

    _, style = query.data.split(":", 1)

    if style not in STYLES:
        await context.bot.send_message(chat_id=chat_id, text="Unknown style — please send the photo again.")
        return

    if style in NEEDS_PERCENT:
        context.user_data["awaiting_percent_for"] = style
        await context.bot.send_message(
            chat_id=chat_id,
            text=(
                f"{STYLES[style][0]} selected.\n\n"
                "Send a number between 1-100 for how strong the effect should be "
                "(e.g. 30 for subtle, 80 for heavy)."
            ),
        )
        return

    await _run_style(chat_id, context, user, file_id, style)


async def handle_percent_reply(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Catches the plain-text number reply after an Aesthetic Blur-type style is picked."""
    style = context.user_data.get("awaiting_percent_for")
    if not style:
        return  # not waiting on a percent input, ignore (let other handlers process it)

    text = (update.message.text or "").strip()
    if not text.isdigit() or not (1 <= int(text) <= 100):
        await update.message.reply_text("Please send a whole number between 1 and 100.")
        return

    pct = int(text)
    file_id = context.user_data.get("original_photo_file_id")
    if not file_id:
        await update.message.reply_text("I don't have a photo to work with — please send a new photo first.")
        context.user_data.pop("awaiting_percent_for", None)
        return

    context.user_data.pop("awaiting_percent_for", None)
    await _run_style(update.effective_chat.id, context, update.effective_user, file_id, style, pct=pct)


async def usage_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    count = db.get_user_usage_today(user.id)
    await update.message.reply_text(
        f"You've generated {count} styled image(s) today. All styles are instant "
        "and unlimited — no daily cap."
    )
