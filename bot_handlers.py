import io
import logging
import random

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes

import cv_filters
import db

logger = logging.getLogger(__name__)

# Emoji reactions known to trigger Telegram's big animated "burst" effect
# on the sender's screen when set with is_big=True — same visual you get
# from a long-press reaction in the app. The Bot API only lets us pick the
# emoji; the animation itself is entirely client-side and can't be customized.
START_REACTION_EMOJIS = ["🎉", "🔥", "❤️", "👍"]

# Appended to every message that carries the style menu buttons (initial
# prompt + every styled-photo caption).
CREDIT_LINE = "Powered by @z5met @z5meta @x5meta"

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
    "retro90s":      ("📼 90s Style",       lambda b, pct=100: cv_filters.retro_90s(b, intensity=pct)),
    "heromask":      ("👁️ Cyber Visor B/W", lambda b, pct=None: cv_filters.tech_visor_bw(b)),


    # ---- Page 2: "More styles" ----
    "duotone":       ("🌆 Duotone",          lambda b, pct=None: cv_filters.duotone(b)),
    "halftone":      ("🗞️ Halftone",        lambda b, pct=None: cv_filters.halftone(b)),
    "glitch":        ("📡 Glitch",           lambda b, pct=None: cv_filters.glitch(b)),
    "thermal":       ("🌡️ Thermal",         lambda b, pct=None: cv_filters.thermal(b)),
    "cyberpunk":     ("🌃 Cyberpunk Neon",   lambda b, pct=None: cv_filters.cyberpunk_neon(b)),
    "watercolor":    ("💧 Watercolor",       lambda b, pct=None: cv_filters.watercolor(b)),
    "comic":         ("💥 Comic Book",       lambda b, pct=None: cv_filters.comic_book(b)),
    "xray":          ("☠️ X-Ray",            lambda b, pct=None: cv_filters.xray(b)),
    "infrared":      ("🌿 Infrared",         lambda b, pct=None: cv_filters.infrared(b)),
    "stainedglass":  ("🪟 Stained Glass",    lambda b, pct=None: cv_filters.stained_glass(b)),
    "mosaic":        ("🧩 Mosaic Tile",      lambda b, pct=None: cv_filters.mosaic_tile(b)),
    "doubleexp":     ("👥 Double Exposure",  lambda b, pct=None: cv_filters.double_exposure(b)),
    "pixelart":      ("👾 Pixel Art",        lambda b, pct=None: cv_filters.pixel_art(b)),
    "chalk":         ("🖤 Chalk & Charcoal", lambda b, pct=None: cv_filters.chalk_charcoal(b)),
    "holo":          ("🌈 Holographic",      lambda b, pct=None: cv_filters.holographic(b)),
    "crt":           ("📺 CRT TV",           lambda b, pct=None: cv_filters.crt_tv(b)),
    "frost":         ("❄️ Frost & Ice",      lambda b, pct=None: cv_filters.frost_ice(b)),
    "solarize":      ("🌗 Solarize",         lambda b, pct=None: cv_filters.solarize(b)),
    "copper":        ("🔶 Copper Etch",      lambda b, pct=None: cv_filters.copper_etch(b)),
    "galaxy":        ("🌌 Galaxy",           lambda b, pct=None: cv_filters.galaxy(b)),
}

# Styles listed here prompt the user for a 1-100 intensity value before running.
NEEDS_PERCENT = {"blur", "blur_bw", "glow", "retro90s"}

PAGE_1_KEYS = [
    "cartoon", "sketch", "sketch_color", "oil", "sepia", "bw",
    "negative", "emboss", "hdr", "vintage", "blur", "glow",
    "blur_bw", "retro90s", "heromask",
]

PAGE_2_KEYS = [
    "duotone", "halftone", "glitch", "thermal", "cyberpunk", "watercolor",
    "comic", "xray", "infrared", "stainedglass", "mosaic", "doubleexp",
    "pixelart", "chalk", "holo", "crt", "frost", "solarize", "copper", "galaxy",
]

# Ordered list of mask ids shown in each "Masks" sub-browser (matches the
# cv_filters.MASKS registry keys). Split into the two categories the user
# can pick between: narrow "eye" masks (domino-style) and full "face" masks.
EYE_MASK_KEYS = [
    "cyber_visor", "kitsune_spirit", "void_wraith",
    "glacier_shard", "toxic_viper", "lunar_eclipse",
    "coral_bloom", "amber_talon", "wildwood",
    "crimson_fang", "iron_spike", "boneyard",
    "steel_falcon", "golden_hawk", "brass_goggles",
    "ember_wing", "storm_volt", "camo_ranger",
    "blood_kitsune", "frostbolt", "circuit_white",
    "raven_feather", "scarlet_blade", "weathered_hide",
]

FACE_MASK_KEYS = [
    "onyx_trooper", "crimson_kitsune", "shadow_cowl",
    "frost_sentinel", "inferno_devil", "gilded_warden",
    "toxic_reaper", "amethyst_ghoul", "cracked_marble",
    "midnight_crescent", "scarlet_bolt", "grim_skull",
    "magma_reaper", "verdant_guardian", "diamond_jester",
    "fractured_onyx", "cyber_wraith", "ice_crown",
    "oni_blaze", "neon_phantom", "raven_beak",
    "violet_circuit", "golden_sigil", "fractured_soul",
    "steampunk_gasmask",
]


def _rows_of_two(keys):
    """Pack style keys into a grid of 2-per-row inline buttons."""
    rows = []
    for i in range(0, len(keys), 2):
        pair = keys[i:i + 2]
        rows.append([
            InlineKeyboardButton(STYLES[k][0], callback_data=f"cv:{k}") for k in pair
        ])
    return rows


def build_style_menu(page: int = 1) -> InlineKeyboardMarkup:
    """Page 1 = original styles + '➕ More styles' button.
    Page 2 = the 20 new styles + '⬅️ Back' button."""
    if page == 1:
        rows = _rows_of_two(PAGE_1_KEYS)
        rows.append([InlineKeyboardButton("🎭 Masks", callback_data="nav:masks")])
        rows.append([InlineKeyboardButton("➕ More styles", callback_data="nav:more")])
    else:
        rows = _rows_of_two(PAGE_2_KEYS)
        rows.append([InlineKeyboardButton("⬅️ Back", callback_data="nav:back")])
    return InlineKeyboardMarkup(rows)


def build_mask_category_menu() -> InlineKeyboardMarkup:
    """Top of the mask browser: choose between Eye masks and Face masks."""
    rows = [
        [InlineKeyboardButton(f"👁️ Eye Masks ({len(EYE_MASK_KEYS)})", callback_data="maskcat:eye")],
        [InlineKeyboardButton(f"🎭 Face Masks ({len(FACE_MASK_KEYS)})", callback_data="maskcat:face")],
        [InlineKeyboardButton("⬅️ Back", callback_data="nav:back")],
    ]
    return InlineKeyboardMarkup(rows)


def build_mask_list_menu(category: str) -> InlineKeyboardMarkup:
    """One button per mask in the given category (opens its preview),
    plus a Back button that returns to the category picker."""
    keys = EYE_MASK_KEYS if category == "eye" else FACE_MASK_KEYS
    rows = []
    for key in keys:
        label = cv_filters.MASKS[key]["label"]
        rows.append([InlineKeyboardButton(label, callback_data=f"maskprev:{key}")])
    rows.append([InlineKeyboardButton("⬅️ Back", callback_data="nav:masks")])
    return InlineKeyboardMarkup(rows)


# Page-1 menu, kept under the old name for any direct references.
STYLE_MENU = build_style_menu(1)


async def _clear_previous_menu(context: ContextTypes.DEFAULT_TYPE, chat_id):
    """Strip the caption and buttons off whichever message currently has the
    active style menu, so only the photo itself remains. Keeps the chat from
    filling up with duplicate menus while never touching the photos."""
    msg_id = context.user_data.pop("active_menu_message_id", None)
    if not msg_id:
        return
    try:
        await context.bot.edit_message_caption(
            chat_id=chat_id, message_id=msg_id, caption=None, reply_markup=None
        )
    except Exception:
        # Message may already be edited/deleted/too old — safe to ignore.
        pass


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Big animated reaction burst on the user's /start message — is_big=True
    # is what triggers the fullscreen animation on their screen.
    try:
        emoji = random.choice(START_REACTION_EMOJIS)
        await update.message.set_reaction(reaction=emoji, is_big=True)
    except Exception:
        logger.exception("Failed to set reaction on /start message")

    await update.message.reply_text(
        "👋 Send me a photo and I'll turn it into different art styles!\n\n"
        "All styles are processed instantly, right here — no AI, no waiting, "
        "no daily limits."
    )


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    photo = update.message.photo[-1]  # highest resolution
    file = await photo.get_file()
    chat_id = update.effective_chat.id

    # A new photo means whatever menu was active before is now stale.
    await _clear_previous_menu(context, chat_id)

    # Always store the ORIGINAL uploaded photo's file_id. Every restyle,
    # no matter how many times the user taps another style button, re-fetches
    # from this same original — never from a previously styled result.
    context.user_data["original_photo_file_id"] = file.file_id
    context.user_data.pop("awaiting_percent_for", None)

    # If the user picked a mask from the "🎭 Masks" browser before sending
    # this photo, run that mask immediately instead of showing the style menu.
    mask_id = context.user_data.pop("awaiting_mask_photo", None)
    if mask_id and mask_id in cv_filters.MASKS:
        await _run_mask(chat_id, context, update.effective_user, file.file_id, mask_id)
        return

    prompt_msg = await update.message.reply_text(
        f"Choose a style:\n\n{CREDIT_LINE}", reply_markup=build_style_menu(1)
    )
    context.user_data["active_menu_message_id"] = prompt_msg.message_id


async def _run_style(chat_id, context: ContextTypes.DEFAULT_TYPE, user, file_id, style, pct=None):
    label, filter_fn = STYLES[style]
    # Reopen on whichever page the user picked this style from, so tapping
    # around page 2 doesn't keep bouncing them back to page 1.
    page = 2 if style in PAGE_2_KEYS else 1

    tg_file = await context.bot.get_file(file_id)
    image_buf = io.BytesIO()
    await tg_file.download_to_memory(image_buf)
    image_bytes = image_buf.getvalue()

    processing_msg = await context.bot.send_message(chat_id=chat_id, text=f"Processing {label}... ⚡")
    try:
        result = filter_fn(image_bytes, pct) if pct is not None else filter_fn(image_bytes)
        caption = f"{label} ✅" + (f" ({pct}%)" if pct is not None else "")

        # Strip the caption+buttons off the previous menu message BEFORE
        # sending the new one, so there's never a moment with two active menus.
        await _clear_previous_menu(context, chat_id)

        sent_photo = await context.bot.send_photo(
            chat_id=chat_id,
            photo=io.BytesIO(result),
            caption=f"{caption}\n\nWant to see it in another style?\n\n{CREDIT_LINE}",
            reply_markup=build_style_menu(page),
        )
        context.user_data["active_menu_message_id"] = sent_photo.message_id
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


async def _run_mask(chat_id, context: ContextTypes.DEFAULT_TYPE, user, file_id, mask_id):
    """Fits the chosen mask onto the user's photo and sends the result,
    mirroring _run_style's flow (processing message, menu cleanup, logging)."""
    meta = cv_filters.MASKS[mask_id]
    label = meta["label"]

    tg_file = await context.bot.get_file(file_id)
    image_buf = io.BytesIO()
    await tg_file.download_to_memory(image_buf)
    image_bytes = image_buf.getvalue()

    processing_msg = await context.bot.send_message(chat_id=chat_id, text=f"Fitting {label}... ⚡")
    try:
        result = cv_filters.apply_mask_by_id(image_bytes, mask_id)
        caption = f"{label} ✅"

        await _clear_previous_menu(context, chat_id)

        sent_photo = await context.bot.send_photo(
            chat_id=chat_id,
            photo=io.BytesIO(result),
            caption=f"{caption}\n\nWant to try another mask or style?\n\n{CREDIT_LINE}",
            reply_markup=build_style_menu(1),
        )
        context.user_data["active_menu_message_id"] = sent_photo.message_id
        db.log_request(user.id, user.username, f"mask:{mask_id}", success=True)
    except Exception as e:
        logger.exception("Mask processing failed")
        await context.bot.send_message(
            chat_id=chat_id, text="Something went wrong fitting that mask. Try another photo — a clear, front-facing face works best."
        )
        db.log_request(user.id, user.username, f"mask:{mask_id}", success=False, error=str(e))
    finally:
        try:
            await processing_msg.delete()
        except Exception:
            pass


async def handle_style_choice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    chat_id = query.message.chat_id

    # "➕ More styles" / "⬅️ Back" just swap the button grid on the existing
    # message — no photo needed yet, so this is handled before the file_id check.
    if query.data == "nav:masks":
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"🎭 Pick a mask type:\n\n{CREDIT_LINE}",
            reply_markup=build_mask_category_menu(),
        )
        return

    if query.data.startswith("maskcat:"):
        _, category = query.data.split(":", 1)
        if category not in ("eye", "face"):
            await context.bot.send_message(chat_id=chat_id, text="Unknown mask category — please try again.")
            return
        label = "Eye" if category == "eye" else "Face"
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"👁️ Pick a {label} mask to preview:\n\n{CREDIT_LINE}",
            reply_markup=build_mask_list_menu(category),
        )
        return

    if query.data.startswith("nav:"):
        _, direction = query.data.split(":", 1)
        page = 2 if direction == "more" else 1
        try:
            await query.edit_message_reply_markup(reply_markup=build_style_menu(page))
        except Exception:
            # Message may be too old to edit (e.g. after a long pause) — resend instead.
            await context.bot.send_message(
                chat_id=chat_id, text=f"Choose a style:\n\n{CREDIT_LINE}", reply_markup=build_style_menu(page)
            )
        return

    if query.data.startswith("maskprev:"):
        _, mask_id = query.data.split(":", 1)
        meta = cv_filters.MASKS.get(mask_id)
        if not meta:
            await context.bot.send_message(chat_id=chat_id, text="Unknown mask — please try again.")
            return
        try:
            with open(meta["preview"], "rb") as fp:
                await context.bot.send_photo(
                    chat_id=chat_id,
                    photo=fp,
                    caption=f"{meta['label']}",
                    reply_markup=InlineKeyboardMarkup(
                        [[InlineKeyboardButton("✅ Use this mask", callback_data=f"maskuse:{mask_id}")]]
                    ),
                )
        except FileNotFoundError:
            logger.exception("Missing preview image for mask %s", mask_id)
            await context.bot.send_message(chat_id=chat_id, text="Preview unavailable for that mask right now.")
        return

    if query.data.startswith("maskuse:"):
        _, mask_id = query.data.split(":", 1)
        if mask_id not in cv_filters.MASKS:
            await context.bot.send_message(chat_id=chat_id, text="Unknown mask — please try again.")
            return

        existing_file_id = context.user_data.get("original_photo_file_id")
        if existing_file_id:
            # A photo is already on hand (the one that opened this menu) —
            # fit the mask to it right away instead of asking again.
            context.user_data.pop("awaiting_mask_photo", None)
            await _run_mask(chat_id, context, query.from_user, existing_file_id, mask_id)
            return

        # No photo yet — remember the choice and prompt for one.
        context.user_data["awaiting_mask_photo"] = mask_id
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"{cv_filters.MASKS[mask_id]['label']} selected 🎭\n\nSend me a photo with a clear, front-facing face and I'll fit the mask to it.",
        )
        return

    user = query.from_user
    file_id = context.user_data.get("original_photo_file_id")

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
