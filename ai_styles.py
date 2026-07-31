"""
AI-powered style transfer using Google's Gemini image model ("Nano Banana").
Free tier, no credit card required — but rate-limited, so we retry with backoff
on 429s and surface a clean error if we truly run out of quota for the day.

Prompts are phrased to describe the AESTHETIC (soft watercolor, hand-painted
anime backgrounds, etc.) rather than naming specific studios/franchises directly,
which tends to avoid content-policy soft-blocks while still getting a close
stylistic result.
"""

import os
import time
import io
from google import genai
from google.genai import types
from google.genai.errors import ClientError

_client = None

STYLE_PROMPTS = {
    "ghibli": (
        "Transform this photo into a hand-painted anime film style: soft watercolor "
        "backgrounds, warm natural lighting, gentle painterly textures, whimsical and "
        "nostalgic atmosphere, in the tradition of classic Japanese animated features. "
        "Keep the subject's likeness and composition recognizable."
    ),
    "anime": (
        "Transform this photo into modern anime art style: clean cel-shaded coloring, "
        "expressive line art, vibrant saturated colors, characteristic anime facial "
        "features and shading. Keep the subject's likeness and composition recognizable."
    ),
    "watercolor": (
        "Transform this photo into a delicate watercolor painting: soft flowing washes "
        "of color, visible paper texture, gentle color bleeding at edges, artistic and "
        "loose brushwork. Keep the subject's likeness and composition recognizable."
    ),
    "comic": (
        "Transform this photo into a Western comic book art style: bold black ink "
        "outlines, dramatic halftone shading, punchy saturated colors, dynamic comic-panel "
        "energy. Keep the subject's likeness and composition recognizable."
    ),
}

STYLE_LABELS = {
    "ghibli": "🎨 Ghibli-inspired",
    "anime": "✨ Anime",
    "watercolor": "🖌️ Watercolor",
    "comic": "💥 Comic Book",
}


def _get_client():
    global _client
    if _client is None:
        api_key = os.environ["GEMINI_API_KEY"]
        _client = genai.Client(api_key=api_key)
    return _client


class QuotaExceededError(Exception):
    pass


def apply_ai_style(image_bytes: bytes, style: str, max_retries: int = 3) -> bytes:
    """
    Sends the image + a style prompt to Gemini's image model and returns the
    resulting image bytes. Retries on transient 429/503 with exponential backoff.
    Raises QuotaExceededError if retries are exhausted due to rate limiting.
    """
    if style not in STYLE_PROMPTS:
        raise ValueError(f"Unknown AI style: {style}")

    client = _get_client()
    prompt = STYLE_PROMPTS[style]

    image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")

    last_error = None
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash-image",
                contents=[prompt, image_part],
            )

            for part in response.candidates[0].content.parts:
                if part.inline_data is not None:
                    return part.inline_data.data

            # No image came back — model may have refused (e.g. safety filter)
            raise RuntimeError(
                "Gemini did not return an image. It may have declined this request "
                "(this can happen with certain photo content or prompts)."
            )

        except ClientError as e:
            last_error = e
            status = getattr(e, "code", None)
            if status == 429:
                wait = 2 ** attempt
                time.sleep(wait)
                continue
            raise

    raise QuotaExceededError(
        "Gemini API rate limit reached. Try again in a bit, or try again tomorrow "
        "once the daily free quota resets."
    ) from last_error
