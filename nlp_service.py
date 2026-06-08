"""
nlp_service.py
==============
Turns free-form Uzbek / mixed Uzbek-Russian text into a structured financial
transaction using an LLM with strict JSON output.

The hard part of this project is NOT the API call — it is the *prompt*. Uzbek
financial speech is highly colloquial and code-switches with Russian:

    "50 ming so'm perevod qildim"      -> transfer, 50 000
    "naqd 100 dollar keldi"            -> income, 100 (USD)
    "obed uchun 35 ming ketdi"         -> expense, 35 000, category=food
    "taksi 20 ming"                    -> expense, 20 000, category=transport
    "zarplata 5 million tushdi"        -> income, 5 000 000, category=salary

We therefore give the model:
  * A precise role + output contract (JSON only).
  * A normalisation guide for Uzbek/Russian number words ("ming"=1e3,
    "million"/"mln"=1e6, "milliard"=1e9).
  * A fixed category taxonomy so the pie chart stays consistent.
  * Few-shot examples covering the tricky code-switching cases.
  * `response_format={"type": "json_object"}` to force valid JSON.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Optional

from openai import AsyncOpenAI

from config import settings

logger = logging.getLogger(__name__)

# Single shared async client (re-used across calls — cheaper than re-creating).
# Works for both OpenAI and Groq: Groq exposes an OpenAI-compatible API, so we
# just point base_url at it (see config.ai_base_url). FREE default = Groq+Llama.
_client = AsyncOpenAI(api_key=settings.ai_api_key, base_url=settings.ai_base_url)

# Fixed taxonomy. Keeping categories closed makes reports aggregate cleanly.
CATEGORIES = [
    "food",          # ovqat, obed, restoran, magazin oziq-ovqat
    "transport",     # taksi, benzin, avtobus, metro
    "shopping",      # kiyim, texnika, bozor-lik
    "bills",         # kommunal, internet, telefon, ijara/arenda
    "health",        # dori, shifokor, apteka
    "education",     # kurs, o'qish, kitob
    "entertainment", # kino, o'yin, dam olish
    "salary",        # oylik, zarplata, maosh
    "transfer",      # perevod, o'tkazma
    "other",
]


@dataclass(slots=True)
class ParsedTransaction:
    """Validated, normalised result of NLP extraction."""

    amount: Decimal
    type: str  # "income" | "expense" | "transfer"
    category: str
    description: str
    date: dt.date
    confidence: float  # 0..1, how sure the model is that this is a transaction


SYSTEM_PROMPT = """\
You are a financial entity-extraction engine for an Uzbek personal-finance app.
The user speaks Uzbek, often mixed with Russian slang. Your ONLY job is to read
the text and return a single JSON object describing the financial transaction.

Return JSON with EXACTLY these keys:
{
  "amount": <number>,                 // positive number in base units (so'm/dollar as a plain number)
  "type": "income" | "expense" | "transfer",
  "category": one of [food, transport, shopping, bills, health, education,
                      entertainment, salary, transfer, other],
  "description": "<short human description in Uzbek>",
  "date": "YYYY-MM-DD",               // resolve relative dates using TODAY given below
  "confidence": <number 0..1>          // 0 if the text is not about money at all
}

NUMBER NORMALISATION (critical):
- "ming" / "тысяча" = x1000.            e.g. "50 ming" -> 50000
- "million" / "mln" / "млн" = x1000000. e.g. "5 million" -> 5000000
- "milliard" / "mlrd" = x1000000000.
- "yarim" = 0.5 (e.g. "yarim million" -> 500000).
- Plain digits stay as-is. Strip currency words from the number.

TYPE / DIRECTION clues (Uzbek + Russian):
- INCOME  (money in):  keldi, tushdi, oldim, kirdi, zarplata/oylik/maosh,
                       prishlo, poluchil.
- EXPENSE (money out): ketdi, sarfladim, to'ladim, oldim (sotib oldim), xarajat,
                       potratil, zaplatil, obed/taksi/magazin.
- TRANSFER:            perevod, o'tkazdim, o'tkazma, otpravil, kartaga tashladim.
  NOTE: a transfer is moving your own money; if it is clearly a payment for
  goods/services treat it as expense instead.

CATEGORY hints: taksi/benzin->transport, obed/ovqat/restoran->food,
kommunal/internet/ijara->bills, dori/apteka->health, oylik/zarplata->salary,
perevod->transfer, kiyim/bozor->shopping, kino/dam->entertainment.

DATE: "bugun"/today -> TODAY. "kecha"/yesterday -> TODAY-1. "ertaga" -> TODAY+1.
If no date is mentioned, use TODAY.

If the message is NOT a financial transaction, return all zeros/empty with
"confidence": 0. Respond with JSON ONLY — no prose, no markdown fences.
"""

# Few-shot examples steer the model on the code-switching edge cases.
FEW_SHOTS = [
    (
        "50 ming so'm perevod qildim",
        {
            "amount": 50000, "type": "transfer", "category": "transfer",
            "description": "Perevod qilindi", "confidence": 0.95,
        },
    ),
    (
        "naqd 100 dollar keldi",
        {
            "amount": 100, "type": "income", "category": "other",
            "description": "Naqd 100 dollar kirim", "confidence": 0.9,
        },
    ),
    (
        "obed uchun 35 ming ketdi",
        {
            "amount": 35000, "type": "expense", "category": "food",
            "description": "Tushlik (obed)", "confidence": 0.96,
        },
    ),
    (
        "zarplata 5 million tushdi",
        {
            "amount": 5000000, "type": "income", "category": "salary",
            "description": "Oylik tushdi", "confidence": 0.97,
        },
    ),
    (
        "taksiga 20 ming",
        {
            "amount": 20000, "type": "expense", "category": "transport",
            "description": "Taksi", "confidence": 0.93,
        },
    ),
]


def _build_messages(text: str, today: dt.date) -> list[dict]:
    """Assemble the chat messages: system prompt + few-shots + user text."""
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    for example_text, example_json in FEW_SHOTS:
        # Inject TODAY into each example so the model learns date resolution.
        payload = {**example_json, "date": today.isoformat()}
        messages.append({"role": "user", "content": example_text})
        messages.append({"role": "assistant", "content": json.dumps(payload, ensure_ascii=False)})

    messages.append(
        {"role": "user", "content": f"TODAY is {today.isoformat()}.\nText: {text}"}
    )
    return messages


def _coerce(data: dict, today: dt.date) -> Optional[ParsedTransaction]:
    """Validate + normalise the raw LLM dict into a ParsedTransaction.

    Returns None if the payload doesn't describe a usable transaction.
    """
    try:
        amount = Decimal(str(data.get("amount", 0)))
    except (InvalidOperation, TypeError):
        amount = Decimal(0)

    confidence = float(data.get("confidence", 0) or 0)
    if amount <= 0 or confidence <= 0:
        return None  # not a (usable) transaction

    tx_type = str(data.get("type", "expense")).lower()
    if tx_type not in {"income", "expense", "transfer"}:
        tx_type = "expense"

    category = str(data.get("category", "other")).lower()
    if category not in CATEGORIES:
        category = "other"

    # Parse the date defensively, falling back to today.
    raw_date = str(data.get("date", "")) or today.isoformat()
    try:
        parsed_date = dt.date.fromisoformat(raw_date)
    except ValueError:
        parsed_date = today

    return ParsedTransaction(
        amount=amount,
        type=tx_type,
        category=category,
        description=str(data.get("description", "")).strip() or "—",
        date=parsed_date,
        confidence=confidence,
    )


async def extract_transaction(
    text: str, today: Optional[dt.date] = None
) -> Optional[ParsedTransaction]:
    """Extract a structured transaction from Uzbek/mixed text.

    Returns a `ParsedTransaction` on success, or `None` when the text is not a
    financial statement (or the model is not confident).
    """
    text = (text or "").strip()
    if not text:
        return None

    today = today or dt.date.today()

    try:
        response = await _client.chat.completions.create(
            model=settings.nlp_model,
            messages=_build_messages(text, today),
            temperature=0,  # deterministic extraction
            response_format={"type": "json_object"},  # force valid JSON
            max_tokens=300,
        )
        content = response.choices[0].message.content or "{}"
        data = json.loads(content)
    except json.JSONDecodeError:
        logger.warning("NLP returned non-JSON for text=%r", text)
        return None
    except Exception:  # network / API errors — surface as "couldn't parse"
        logger.exception("NLP extraction failed for text=%r", text)
        return None

    return _coerce(data, today)
