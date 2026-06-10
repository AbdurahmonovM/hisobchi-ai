"""
ai_client.py
============
Tiny shared wrapper around the Google Gemini client.

Both the speech (audio→text) and NLP (text→transactions) services call Gemini's
`generate_content`. The free tier occasionally returns transient 503/500 errors
("high demand"). We retry those a few times with a short backoff so a temporary
spike doesn't surface to the user as "no transaction found".
"""

from __future__ import annotations

import asyncio
import logging

from google import genai
from google.genai import types

from config import settings

logger = logging.getLogger(__name__)

# Substrings that mark a TRANSIENT, retry-worthy server error.
_TRANSIENT_MARKERS = ("503", "UNAVAILABLE", "500", "INTERNAL", "overloaded")


def get_client() -> genai.Client:
    """Build a Gemini client from current settings (cheap to construct)."""
    return genai.Client(api_key=settings.GEMINI_API_KEY)


async def generate(
    contents,
    *,
    config: types.GenerateContentConfig | None = None,
    retries: int = 3,
):
    """Call Gemini generate_content with retry on transient server errors.

    Raises the last exception if all attempts fail (caller decides what to do).
    """
    client = get_client()
    last_exc: Exception | None = None

    for attempt in range(retries):
        try:
            return await client.aio.models.generate_content(
                model=settings.gemini_model,
                contents=contents,
                config=config,
            )
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            transient = any(m in str(exc) for m in _TRANSIENT_MARKERS)
            if transient and attempt < retries - 1:
                delay = 1.5 * (attempt + 1)
                logger.warning(
                    "Gemini transient error (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1, retries, delay, str(exc)[:120],
                )
                await asyncio.sleep(delay)
                continue
            raise

    assert last_exc is not None
    raise last_exc
