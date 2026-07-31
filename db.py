"""
MongoDB connection + simple usage logging / per-user daily rate limiting.

We keep this deliberately simple:
- `requests` collection: one document per style-transfer request (for your own analytics)
- `usage` collection: one document per (user_id, date) to enforce a soft daily cap per user,
  so a single user can't burn through your entire Gemini free-tier quota alone.
"""

import os
from datetime import datetime, timezone
from pymongo import MongoClient

_client = None
_db = None


def get_db():
    global _client, _db
    if _db is None:
        mongo_url = os.environ["MONGO_URL"]
        _client = MongoClient(mongo_url)
        # database name comes from the URL path; falls back to 'ghiblibot' if not specified
        _db = _client.get_default_database(default="ghiblibot")
    return _db


def log_request(user_id: int, username: str, style: str, success: bool, error: str = None):
    db = get_db()
    db.requests.insert_one({
        "user_id": user_id,
        "username": username,
        "style": style,
        "success": success,
        "error": error,
        "timestamp": datetime.now(timezone.utc),
    })


def _today_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def check_and_increment_usage(user_id: int, daily_limit: int) -> bool:
    """
    Returns True if the user is allowed to make another AI (Gemini) request today,
    and atomically increments their counter if so.
    Returns False if they've hit the daily_limit.

    Note: this limits AI-STYLE requests only (Ghibli/Anime/etc).
    OpenCV styles (Cartoon/Sketch) are free and unlimited — don't count them here.
    """
    db = get_db()
    today = _today_str()

    doc = db.usage.find_one_and_update(
        {"user_id": user_id, "date": today, "count": {"$lt": daily_limit}},
        {"$inc": {"count": 1}},
        upsert=False,
        return_document=True,
    )

    if doc is not None:
        return True

    # No matching doc means either it doesn't exist yet (first request today)
    # or it exists but is already at/over the limit.
    existing = db.usage.find_one({"user_id": user_id, "date": today})
    if existing is None:
        db.usage.update_one(
            {"user_id": user_id, "date": today},
            {"$setOnInsert": {"count": 1}},
            upsert=True,
        )
        return True

    return False


def get_user_usage_today(user_id: int) -> int:
    db = get_db()
    today = _today_str()
    doc = db.usage.find_one({"user_id": user_id, "date": today})
    return doc["count"] if doc else 0
