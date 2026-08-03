"""
Entry point. Uses python-telegram-bot's built-in run_webhook(), which binds
$PORT itself - no Flask/gunicorn layer needed. This is the pattern that
deploys cleanly on Render: it fails fast and loud if no webhook URL can be
determined, instead of silently falling into polling mode and never binding
a port (which is what causes Render to time out and kill the deploy).
"""

import os
import asyncio
import logging

import httpx
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters

from bot_handlers import start, handle_photo, handle_style_choice, usage_command, handle_percent_reply

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

BOT_TOKEN = os.environ["BOT_TOKEN"]
WEBHOOK_URL = os.environ.get("WEBHOOK_URL", "").rstrip("/")
PORT = int(os.environ.get("PORT", 10000))

# Render's free tier spins a web service down after 15 minutes with no
# incoming HTTP traffic, which causes a slow "cold start" on the next
# message. Self-pinging our own health endpoint well inside that window
# keeps the service warm. Only makes sense in webhook mode (we need a
# public URL to ping); polling mode has nothing to ping and is skipped.
SELF_PING_INTERVAL_SECONDS = 10 * 60  # 10 minutes


async def self_ping_loop(ping_url: str) -> None:
    logger.info("Self-ping loop started, will ping %s every %s seconds", ping_url, SELF_PING_INTERVAL_SECONDS)
    async with httpx.AsyncClient(timeout=30) as client:
        while True:
            try:
                await asyncio.sleep(SELF_PING_INTERVAL_SECONDS)
                response = await client.get(ping_url)
                logger.info("Self-ping to %s returned status %s", ping_url, response.status_code)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # Never let a failed ping take down the loop - just log and retry next cycle.
                logger.warning("Self-ping to %s failed: %s", ping_url, exc)


def build_app(ping_url: str | None) -> Application:
    application = Application.builder().token(BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("usage", usage_command))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_percent_reply))
    application.add_handler(CallbackQueryHandler(handle_style_choice))

    if ping_url:
        async def _start_self_ping(app: Application) -> None:
            app.bot_data["self_ping_task"] = asyncio.create_task(self_ping_loop(ping_url))

        async def _stop_self_ping(app: Application) -> None:
            task = app.bot_data.get("self_ping_task")
            if task:
                task.cancel()

        application.post_init = _start_self_ping
        application.post_stop = _stop_self_ping

    return application


def main():
    # Python 3.14 removed the implicit "create a loop if none exists"
    # behavior of asyncio.get_event_loop(), which PTB's run_webhook() relies
    # on internally. Creating and setting the loop explicitly here keeps
    # this working regardless of which Python version the host actually uses.
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    render_url = os.environ.get("RENDER_EXTERNAL_URL")  # auto-set by Render
    on_render = os.environ.get("RENDER") == "true"  # set on every Render service
    webhook_target = WEBHOOK_URL or render_url

    if on_render and not webhook_target:
        raise RuntimeError(
            "Running on Render but no webhook URL could be determined. "
            "Set the WEBHOOK_URL env var to this service's public URL "
            "(Render dashboard -> service -> the https://<name>.onrender.com address)."
        )

    ping_url = None
    if webhook_target:
        webhook_base = webhook_target.rstrip("/")
        if not webhook_base.startswith(("http://", "https://")):
            webhook_base = f"https://{webhook_base}"
        full_webhook_url = f"{webhook_base}/{BOT_TOKEN}"
        # Ping our own webhook path. Telegram's webhook server responds to
        # any HTTP request on this route (even a plain GET), which is all
        # we need to prove to Render that the service is alive and reset
        # its idle timer.
        ping_url = full_webhook_url

    application = build_app(ping_url)

    if webhook_target:
        logger.info("Starting in webhook mode on port %s", PORT)
        logger.info("Registering webhook URL: %s", full_webhook_url)
        application.run_webhook(
            listen="0.0.0.0",
            port=PORT,
            url_path=BOT_TOKEN,
            webhook_url=full_webhook_url,
            allowed_updates=Update.ALL_TYPES,
            drop_pending_updates=True,
        )
    else:
        logger.info("Starting in polling mode (no WEBHOOK_URL / RENDER_EXTERNAL_URL set)")
        application.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)


if __name__ == "__main__":
    main()
