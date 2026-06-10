"""
nlp_service.py
==============
Turns free-form Uzbek / mixed Uzbek-Russian text into a LIST of structured
financial transactions using Google Gemini with strict JSON output.

Key capability: a SINGLE message may describe SEVERAL operations, e.g.
    "zarplata 5 million tushdi, obedga 35 ming ketdi, internetga 200 ming"
    -> [income 5 000 000 salary, expense 35 000 food, expense 200 000 bills]

We give the model:
  * A precise role + output contract (JSON object with a "transactions" array).
  * A normalisation guide for Uzbek/Russian number words ("ming"=1e3,
    "million"/"mln"=1e6, "milliard"=1e9).
  * A fixed category taxonomy so the pie chart stays consistent.
  * `response_mime_type="application/json"` to force valid JSON.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Optional

from google.genai import types

import ai_client

logger = logging.getLogger(__name__)


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


# Number-word multipliers (Uzbek + Russian) for the lightweight amount parser.
_AMOUNT_MULTIPLIERS: list[tuple[tuple[str, ...], int]] = [
    (("milliard", "mlrd", "млрд"), 1_000_000_000),
    (("million", "mln", "млн"), 1_000_000),
    (("ming", "тысяч", "тыс"), 1_000),
]


def parse_uzbek_amount(text: str) -> Optional[Decimal]:
    """Parse a money amount from short Uzbek/Russian text WITHOUT calling an LLM.

    Used for onboarding income input where the user just states a number, e.g.
    "5 million", "5000000", "yarim million", "50 ming". Returns None if no
    number can be found (so the bot can re-prompt instead of guessing).
    """
    if not text:
        return None

    t = text.lower().replace(",", ".")
    match = re.search(r"\d+(?:\.\d+)?", t)
    has_yarim = "yarim" in t  # "half"

    multiplier = 1
    for words, factor in _AMOUNT_MULTIPLIERS:
        if any(w in t for w in words):
            multiplier = factor
            break

    if match is None:
        # e.g. bare "yarim million" with no digits.
        return Decimal(str(0.5 * multiplier)) if (has_yarim and multiplier > 1) else None

    number = float(match.group())
    if has_yarim:
        number += 0.5
    value = Decimal(str(number)) * multiplier
    return value if value > 0 else None


@dataclass(slots=True)
class ParsedTransaction:
    """Validated, normalised single transaction extracted from text/voice."""

    amount: Decimal
    type: str  # "income" | "expense" | "transfer"
    category: str
    description: str
    date: dt.date
    confidence: float  # 0..1, how sure the model is that this is a transaction


# The system instruction. Note the emphasis on MULTIPLE transactions per message.
SYSTEM_PROMPT = """\
You are a financial entity-extraction engine for an Uzbek personal-finance app.
The user speaks Uzbek, often mixed with Russian slang. A single message may
contain SEVERAL financial operations. Your job: read the text and return a JSON
object listing EVERY transaction you find.

Return EXACTLY this shape:
{
  "transactions": [
    {
      "amount": <number>,                // positive plain number (no currency words)
      "type": "income" | "expense" | "transfer",
      "category": one of [food, transport, shopping, bills, health, education,
                          entertainment, salary, transfer, other],
      "description": "<short description in Uzbek>",
      "date": "YYYY-MM-DD",              // resolve relative dates using TODAY below
      "confidence": <number 0..1>
    }
    // ... one object PER operation; could be 0, 1, or many
  ]
}

CRITICAL — MULTIPLE OPERATIONS:
The user often lists several things at once, separated by commas, "va", "keyin",
or just spoken in sequence. Split them into SEPARATE transaction objects.
Example: "zarplata 5 million tushdi, obedga 35 ming, internetga 200 ming to'ladim"
-> THREE objects: income 5000000 salary; expense 35000 food; expense 200000 bills.

NUMBER NORMALISATION:
- "ming" / "тысяча" = x1000.            "50 ming" -> 50000
- "million" / "mln" / "млн" = x1000000. "5 million" -> 5000000
- "milliard" / "mlrd" = x1000000000.
- "yarim" = 0.5 ("yarim million" -> 500000).
- Plain digits stay as-is. Strip currency words from the number.

TYPE / DIRECTION (Uzbek + Russian):
- INCOME  (money in):  keldi, tushdi, oldim, kirdi, zarplata/oylik/maosh, prishlo, poluchil.
- EXPENSE (money out): ketdi, sarfladim, to'ladim, sotib oldim, xarajat, potratil, zaplatil.
- TRANSFER:            perevod, o'tkazdim, o'tkazma, otpravil. (Payment for goods = expense.)

CATEGORY hints: taksi/benzin->transport, obed/ovqat/restoran->food,
kommunal/internet/ijara->bills, dori/apteka->health, oylik/zarplata->salary,
perevod->transfer, kiyim/bozor->shopping, kino/dam->entertainment.

DATE: "bugun"->TODAY, "kecha"->TODAY-1, "ertaga"->TODAY+1. No date -> TODAY.

If the message has NO financial operations, return {"transactions": []}.
Respond with JSON ONLY — no prose, no markdown fences.
"""


def _coerce(data: dict, today: dt.date) -> Optional[ParsedTransaction]:
    """Validate + normalise one raw item into a ParsedTransaction, or None."""
    try:
        amount = Decimal(str(data.get("amount", 0)))
    except (InvalidOperation, TypeError):
        amount = Decimal(0)

    confidence = float(data.get("confidence", 0) or 0)
    if amount <= 0 or confidence <= 0:
        return None  # not a usable transaction

    tx_type = str(data.get("type", "expense")).lower()
    if tx_type not in {"income", "expense", "transfer"}:
        tx_type = "expense"

    category = str(data.get("category", "other")).lower()
    if category not in CATEGORIES:
        category = "other"

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


async def extract_transactions(
    text: str, today: Optional[dt.date] = None
) -> list[ParsedTransaction]:
    """Extract ALL transactions from Uzbek/mixed text using Gemini.

    Returns a list (possibly empty). One spoken message can yield many items.
    """
    text = (text or "").strip()
    if not text:
        return []

    today = today or dt.date.today()
    user_content = f"TODAY is {today.isoformat()}.\nText: {text}"

    try:
        response = await ai_client.generate(
            user_content,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0,
                response_mime_type="application/json",
            ),
        )
        data = json.loads(response.text or "{}")
    except json.JSONDecodeError:
        logger.warning("Gemini returned non-JSON for text=%r", text)
        return []
    except Exception:
        logger.exception("Gemini extraction failed for text=%r", text)
        return []

    items = data.get("transactions", []) if isinstance(data, dict) else []
    if not isinstance(items, list):
        return []

    result: list[ParsedTransaction] = []
    for item in items:
        if isinstance(item, dict):
            parsed = _coerce(item, today)
            if parsed is not None:
                result.append(parsed)
    return result
