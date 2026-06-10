"""
speech_service.py
=================
Speech-to-text via Google Gemini (multimodal).

Telegram voice messages arrive as OGG/Opus. Gemini accepts the audio bytes
inline (no ffmpeg needed) and transcribes them. We tell Gemini the language is
Uzbek (often mixed with Russian) and ask for a clean transcript only.

The transcript is then passed to `nlp_service.extract_transactions`, which can
pull MULTIPLE income/expense operations out of a single message.
"""

from __future__ import annotations

import logging

from google import genai
from google.genai import types

from config import settings

logger = logging.getLogger(__name__)


def get_gemini_client() -> genai.Client:
    """Create a Gemini client from the current settings (cheap to construct)."""
    return genai.Client(api_key=settings.GEMINI_API_KEY)


# Instruction telling Gemini to return a faithful transcript and nothing else.
_TRANSCRIBE_PROMPT = (
    "Transcribe this voice message into text EXACTLY as spoken. "
    "The language is Uzbek, often mixed with Russian financial slang "
    "(so'm, ming, million, perevod, naqd, zarplata, taksi, obed...). "
    "Return ONLY the transcript text — no translation, no commentary, no quotes."
)


class TranscriptionError(Exception):
    """Raised when audio could not be transcribed into usable text."""


async def transcribe_voice(audio_bytes: bytes, mime_type: str = "audio/ogg") -> str:
    """Transcribe raw Telegram voice bytes (OGG/Opus) to Uzbek text via Gemini.

    Parameters
    ----------
    audio_bytes:
        The downloaded voice message content.
    mime_type:
        MIME type of the audio. Telegram voice notes are "audio/ogg".

    Returns
    -------
    str
        The transcribed text (stripped).

    Raises
    ------
    TranscriptionError
        If Gemini fails or returns empty/blank text.
    """
    if not audio_bytes:
        raise TranscriptionError("Empty audio payload")

    client = get_gemini_client()
    try:
        response = await client.aio.models.generate_content(
            model=settings.gemini_model,
            contents=[
                types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
                _TRANSCRIBE_PROMPT,
            ],
            config=types.GenerateContentConfig(temperature=0),
        )
    except Exception as exc:  # noqa: BLE001 - convert any API error to our type
        logger.exception("Gemini transcription failed")
        raise TranscriptionError("Speech recognition service failed") from exc

    text = (response.text or "").strip()
    if not text:
        raise TranscriptionError("Audio produced no recognisable text")

    logger.info("Transcribed voice -> %r", text)
    return text
