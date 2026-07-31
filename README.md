# Telegram Photo Style Bot

Turns a photo into Ghibli-inspired, Anime, Watercolor, Comic (AI-powered), or
Cartoon/Sketch (free, instant, local) styles.

## How it works

- **AI styles** (Ghibli-inspired, Anime, Watercolor, Comic Book) use Google's
  Gemini image model, which has a genuinely free tier (no credit card) but
  with rate limits. A per-user daily cap (`DAILY_AI_LIMIT_PER_USER` in
  `handlers/bot_handlers.py`, default 15) protects your shared quota from
  being drained by one person.
- **Instant styles** (Cartoon, Pencil Sketch, Color Sketch) use OpenCV only —
  no API call, no rate limit, runs on your own server for free, forever.
- **MongoDB** logs every request (for your own visibility) and tracks each
  user's daily AI-style usage count.

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

**Gemini API key**
1. Go to [Google AI Studio](https://aistudio.google.com/apikey)
2. Create an API key (no credit card needed for the free tier)
3. Free tier rate limits change over time — check your live limits on the AI Studio dashboard.
   The bot retries automatically on rate-limit errors and tells users to try again later
   if it truly runs out.

## 2. Deploy to Render

1. Push this folder to a GitHub repo
2. On [Render](https://render.com), New → Web Service → connect your repo
3. Render should auto-detect `render.yaml`. If not, set manually:
   - Build command: `pip install -r requirements.txt`
   - Start command: `gunicorn -w 1 --timeout 120 main:flask_app`
4. Add environment variables in Render's dashboard (Settings → Environment):
   - `BOT_TOKEN`
   - `MONGO_URL`
   - `GEMINI_API_KEY`
   - `WEBHOOK_URL` — set this to your Render URL, e.g. `https://your-app-name.onrender.com`
     (you'll know this before first deploy if you name the service yourself — Render
     URLs follow `https://<service-name>.onrender.com`)
5. Deploy. On startup, the bot automatically registers its webhook with Telegram
   using `WEBHOOK_URL`.

**Note on Render's free tier:** the service sleeps after ~15 minutes of no traffic,
so the first message after idle will be slow (~30-60s cold start) while it spins back up.

## 3. Test it

Message your bot on Telegram, send a photo, tap a style button.

## Local testing (optional, no Render/webhook needed)

```bash
cp .env.example .env
# fill in .env with your real values
pip install -r requirements.txt
python local_dev.py
```

## Project structure (all files flat, no subfolders — upload straight to GitHub root)

```
main.py               # Flask webhook server (used on Render)
local_dev.py           # Polling-based runner for local testing
bot_handlers.py        # Telegram command/message/button logic
cv_filters.py           # Free OpenCV styles (cartoon, sketch)
ai_styles.py            # Gemini AI styles (ghibli, anime, watercolor, comic)
db.py                   # MongoDB logging + per-user daily rate limiting
render.yaml              # Render service definition
requirements.txt
.env.example
```

## Adjusting things

- **Change the daily AI limit per user:** edit `DAILY_AI_LIMIT_PER_USER` in
  `bot_handlers.py`.
- **Add a new AI style:** add an entry to `STYLE_PROMPTS` and `STYLE_LABELS` in
  `ai_styles.py`, then add a button for it in `STYLE_MENU` in `bot_handlers.py`.
- **Add a new free/instant style:** add a function to `cv_filters.py` following
  the pattern of `cartoonify`/`pencil_sketch`, then wire it into `handle_style_choice`
  in `bot_handlers.py`.
