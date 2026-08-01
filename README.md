# Telegram Photo Style Bot

Turns a photo into one of 10 instant art styles — Cartoon, Pencil Sketch,
Color Sketch, Oil Painting, Sepia, Black & White, Negative, Emboss, HDR Glow,
and Warm Vintage. Everything runs locally with OpenCV. No AI models, no API
keys, no rate limits, no external network calls of any kind.

## How it works

- All 10 styles are pure OpenCV image processing (`cv_filters.py`) — pixel
  math only, no AI, no third-party service, no cost, no daily cap.
- **MongoDB** logs every request (for your own visibility/usage stats).

## 1. Get your credentials

**Telegram bot token**
1. Message [@BotFather](https://t.me/BotFather) on Telegram
2. `/newbot`, follow the prompts
3. Copy the token it gives you

**MongoDB URL**
1. Create a free cluster at [MongoDB Atlas](https://www.mongodb.com/cloud/atlas/register)
2. Database Access → add a user with a password
3. Network Access → allow access from anywhere (`0.0.0.0/0`) so Render can connect
4. Connect → Drivers → copy the connection string, put your DB name in the path,
   e.g. `mongodb+srv://user:pass@cluster.mongodb.net/ghiblibot?retryWrites=true&w=majority`

## 2. Deploy to Render

1. Push this folder to a GitHub repo
2. On [Render](https://render.com), New → Web Service (or Blueprint) → connect your repo
3. Render should auto-detect `render.yaml`. If not, set manually:
   - Build command: `pip install --no-cache-dir -r requirements.txt`
   - Start command: `python main.py`
4. Add environment variables in Render's dashboard (Settings → Environment):
   - `BOT_TOKEN`
   - `MONGO_URL`
   - `WEBHOOK_URL` — set this to your Render URL, e.g. `https://your-app-name.onrender.com`
     (you'll know this before first deploy if you name the service yourself — Render
     URLs follow `https://<service-name>.onrender.com`)
   - `PYTHON_VERSION` — `3.12.7` (also set as a forced env var to override any
     platform default)
5. Deploy. On startup, the bot automatically registers its webhook with Telegram
   using `WEBHOOK_URL`.

**Note on Render's free tier:** the service sleeps after ~15 minutes of no traffic,
so the first message after idle will be slow (~30-60s cold start) while it spins back up.

## 3. Test it

Message your bot on Telegram, send a photo, tap a style button. Every style
responds instantly — no waiting on an external API.

## Local testing (optional, no Render/webhook needed)

```bash
cp .env.example .env
# fill in .env with your real values
pip install -r requirements.txt
python local_dev.py
```

## Project structure (all files flat, no subfolders — upload straight to GitHub root)

```
main.py               # Webhook server entrypoint (used on Render), via PTB's run_webhook
local_dev.py           # Polling-based runner for local testing
bot_handlers.py        # Telegram command/message/button logic
cv_filters.py           # All 10 instant, local OpenCV styles
db.py                   # MongoDB logging
runtime.txt              # Pins Python version for Render's buildpack
.python-version           # Backup Python version pin
Procfile                   # Backup start command
render.yaml                 # Render service definition
requirements.txt
.env.example
```

## Adjusting things

- **Add a new instant style:** add a function to `cv_filters.py` following the
  pattern of the existing filters (decode → transform → encode), then add an
  entry to the `STYLES` dict and a button in `STYLE_MENU` in `bot_handlers.py`.
- **Change styles shown per row:** edit the `STYLE_MENU` layout in `bot_handlers.py`.
