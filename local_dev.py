"""
Run this locally for testing WITHOUT deploying to Render or setting up webhooks.
Uses polling instead. Requires a .env file (copy .env.example -> .env and fill it in).

Usage:
    pip install -r requirements.txt
    python local_dev.py
"""

import os
from dotenv import load_dotenv

load_dotenv()

from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters
from bot_handlers import start, handle_photo, handle_style_choice, usage_command, handle_percent_reply

BOT_TOKEN = os.environ["BOT_TOKEN"]

app = Application.builder().token(BOT_TOKEN).build()
app.add_handler(CommandHandler("start", start))
app.add_handler(CommandHandler("usage", usage_command))
app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_percent_reply))
app.add_handler(CallbackQueryHandler(handle_style_choice))

if __name__ == "__main__":
    print("Bot running locally via polling. Press Ctrl+C to stop.")
    app.run_polling()
