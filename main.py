"""
Entry point. Runs a Flask server that receives Telegram webhook updates —
this is the pattern Render (and most PaaS free tiers) work best with, since
long-running polling loops fight with how these platforms manage web services.
"""

import os
import asyncio
import logging

from flask import Flask, request
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters

from bot_handlers import start, handle_photo, handle_style_choice, usage_command

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BOT_TOKEN = os.environ["BOT_TOKEN"]
WEBHOOK_URL = os.environ.get("WEBHOOK_URL", "").rstrip("/")
PORT = int(os.environ.get("PORT", 10000))

flask_app = Flask(__name__)

telegram_app = Application.builder().token(BOT_TOKEN).build()
telegram_app.add_handler(CommandHandler("start", start))
telegram_app.add_handler(CommandHandler("usage", usage_command))
telegram_app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
telegram_app.add_handler(CallbackQueryHandler(handle_style_choice))

# Single event loop reused across requests (Flask is sync, PTB is async)
_loop = asyncio.new_event_loop()
asyncio.set_event_loop(_loop)
_loop.run_until_complete(telegram_app.initialize())

if WEBHOOK_URL:
    _loop.run_until_complete(
        telegram_app.bot.set_webhook(url=f"{WEBHOOK_URL}/webhook/{BOT_TOKEN}")
    )
    logger.info(f"Webhook set to {WEBHOOK_URL}/webhook/{BOT_TOKEN}")
else:
    logger.warning("WEBHOOK_URL not set — webhook was NOT registered with Telegram. "
                    "Set WEBHOOK_URL and redeploy.")


@flask_app.route(f"/webhook/{BOT_TOKEN}", methods=["POST"])
def webhook():
    update = Update.de_json(request.get_json(force=True), telegram_app.bot)
    _loop.run_until_complete(telegram_app.process_update(update))
    return "ok"


@flask_app.route("/", methods=["GET"])
def health_check():
    return "Bot is running."


if __name__ == "__main__":
    flask_app.run(host="0.0.0.0", port=PORT)
