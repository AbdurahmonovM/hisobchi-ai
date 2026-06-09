"""
speech_service.py
=================
Speech-to-text via the OpenAI Whisper API.

Telegram voice messages arrive as OGG/Opus. Whisper accepts that format
directly, so we don't need ffmpeg — we just hand the bytes to the API with a
filename that carries the `.ogg` extension (the API sniffs format from it).

We bias Whisper toward Uzbek with `language="uz"` and a domain `prompt`
containing finance vocabulary, which measurably improves recognition of
numbers and money words.
"""

from __future__ import annotations

import io
import logging
from typing import Optional

from openai import AsyncOpenAI

from config import settings

logger = logging.getLogger(__name__)

def get_ai_client() -> AsyncOpenAI:
    """Lazy-load the OpenAI/Groq client using the latest settings."""
    return AsyncOpenAI(api_key=settings.ai_api_key, base_url=settings.ai_base_url)


# A short priming prompt nudges Whisper toward finance/number vocabulary and
# the Uzbek+Russian code-switching it will hear. It is NOT a transcript — just
# context describing the expected content.
_WHISPER_PROMPT = (
    "Moliyaviy xabar. So'm, dollar, ming, million, perevod, naqd, zarplata, "
    "oylik, taksi, obed, xarajat, kirim, chiqim."
)


class TranscriptionError(Exception):
    """Raised when audio could not be transcribed into usable text."""


async def transcribe_voice(audio_bytes: bytes, filename: str = "voice.ogg") -> str:
    """Transcribe raw voice bytes (OGG/Opus from Telegram) to Uzbek text.

    Parameters
    ----------
    audio_bytes:
        The downloaded voice message content.
    filename:
        Name passed to the API so it can infer the container format. The
        extension matters more than the stem.

    Returns
    -------
    str
        The transcribed text (stripped).

    Raises
    ------
    TranscriptionError
        If the API fails or returns empty/blank text.
    """
    if not audio_bytes:
        raise TranscriptionError("Empty audio payload")

    # Whisper's SDK expects a file-like object with a `.name` attribute.
    buffer = io.BytesIO(audio_bytes)
    buffer.name = filename

    try:
        client = get_ai_client()
        result = await client.audio.transcriptions.create(
            model=settings.whisper_model,
            file=buffer,
            language="uz",            # bias toward Uzbek
            prompt=_WHISPER_PROMPT,   # finance-domain priming
            temperature=0,
        )
    except Exception as exc:  # noqa: BLE001 - convert any API error to our type
        logger.exception("Whisper transcription failed")
        raise TranscriptionError("Speech recognition service failed") from exc

    text: Optional[str] = (result.text or "").strip()
    if not text:
        raise TranscriptionError("Audio produced no recognisable text")

    logger.info("Transcribed voice -> %r", text)
    return text
