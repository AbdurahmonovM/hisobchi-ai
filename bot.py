"""
bot.py
======
The aiogram 3.x Telegram bot for Hisobchi AI.

Pipeline for a voice message:
    voice -> download bytes -> Whisper (speech_service) -> text
          -> LLM extraction (nlp_service) -> ParsedTransaction
          -> persist (database) -> confirmation reply with the Web App button.

Commands:
    /start    - register the user and show the "Open Hisobchi" Web App button.
    /balance  - quick text balance without opening the Web App.
    /help     - usage instructions.

Run standalone with:  python bot.py   (long-polling)
In production you would typically run this alongside the FastAPI server; here
they are separate processes that share the same database.
"""

from __future__ import annotations

import asyncio
import logging

from aiogram import Bot, Dispatcher, F, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command, CommandStart
from aiogram.types import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
    WebAppInfo,
)

from config import settings
from database import (
    AsyncSessionLocal,
    TxType,
    add_transaction,
    get_balance,
    get_monthly_totals,
    get_or_create_user,
    init_db,
)
from nlp_service import extract_transaction
from speech_service import TranscriptionError, transcribe_voice

logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("hisobchi.bot")

router = Router(name="hisobchi")

# Emojis per type for friendlier confirmation messages.
_TYPE_EMOJI = {"income": "🟢", "expense": "🔴", "transfer": "🔁"}


def _web_app_keyboard() -> InlineKeyboardMarkup:
    """Inline keyboard with a single button that launches the Web App."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="📊 Hisobotni ochish",
                    web_app=WebAppInfo(url=settings.WEB_APP_URL),
                )
            ]
        ]
    )


def _fmt_money(amount, currency: str = settings.DEFAULT_CURRENCY) -> str:
    """Format a Decimal/number with thousands separators, e.g. '50 000 UZS'."""
    return f"{amount:,.0f}".replace(",", " ") + f" {currency}"


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
@router.message(CommandStart())
async def cmd_start(message: Message) -> None:
    """Register the user and greet them with the Web App button."""
    u = message.from_user
    async with AsyncSessionLocal() as session:
        await get_or_create_user(
            session,
            user_id=u.id,
            username=u.username,
            full_name=u.full_name,
            language_code=u.language_code,
        )

    await message.answer(
        "👋 <b>Hisobchi AI</b>ga xush kelibsiz!\n\n"
        "Menga ovozli xabar yuboring — men daromad va xarajatlaringizni "
        "avtomatik yozib boraman.\n\n"
        "Masalan ayting:\n"
        "• <i>«obed uchun 35 ming ketdi»</i>\n"
        "• <i>«zarplata 5 million tushdi»</i>\n"
        "• <i>«50 ming so'm perevod qildim»</i>\n\n"
        "📊 Hisobotni ko'rish uchun pastdagi tugmani bosing.",
        reply_markup=_web_app_keyboard(),
    )


@router.message(Command("balance"))
async def cmd_balance(message: Message) -> None:
    """Return the user's balance plus this month's income/expense as text."""
    import datetime as dt

    today = dt.date.today()
    async with AsyncSessionLocal() as session:
        balance = await get_balance(session, message.from_user.id)
        income, expense = await get_monthly_totals(
            session, message.from_user.id, today.year, today.month
        )

    await message.answer(
        f"💰 <b>Joriy balans:</b> {_fmt_money(balance)}\n\n"
        f"📅 <b>Bu oy ({today:%Y-%m}):</b>\n"
        f"🟢 Kirim:  <b>{_fmt_money(income)}</b>\n"
        f"🔴 Chiqim: <b>{_fmt_money(expense)}</b>",
        reply_markup=_web_app_keyboard(),
    )


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(
        "ℹ️ <b>Qanday ishlataman?</b>\n\n"
        "Shunchaki ovozli xabar yuboring. Men:\n"
        "1️⃣ Ovozni matnga aylantiraman (Whisper)\n"
        "2️⃣ Summa, tur va kategoriyani ajratib olaman (AI)\n"
        "3️⃣ Tranzaksiyani saqlayman\n\n"
        "Buyruqlar:\n"
        "/balance — joriy balans\n"
        "/start — boshlash va hisobot tugmasi",
    )


# ---------------------------------------------------------------------------
# Voice handler — the core feature
# ---------------------------------------------------------------------------
@router.message(F.voice | F.audio)
async def handle_voice(message: Message, bot: Bot) -> None:
    """Transcribe a voice/audio message, extract a transaction, store it."""
    # Give immediate feedback — STT + NLP can take a couple of seconds.
    status = await message.answer("🎧 Eshityapman...")

    # 1) Download the audio bytes from Telegram.
    voice = message.voice or message.audio
    try:
        file = await bot.get_file(voice.file_id)
        buffer = await bot.download_file(file.file_path)
        audio_bytes = buffer.read()
    except Exception:
        logger.exception("Failed to download voice file")
        await status.edit_text("⚠️ Ovozli xabarni yuklab bo'lmadi. Qayta urinib ko'ring.")
        return

    # 2) Speech-to-text.
    try:
        text = await transcribe_voice(audio_bytes)
    except TranscriptionError:
        await status.edit_text(
            "😕 Ovozni tushunolmadim. Iltimos, aniqroq va sekinroq gapiring."
        )
        return

    await status.edit_text(f"📝 <i>«{text}»</i>\n\n🤖 Tahlil qilyapman...")

    # 3) NLP entity extraction.
    parsed = await extract_transaction(text)
    if parsed is None:
        await status.edit_text(
            f"📝 <i>«{text}»</i>\n\n"
            "❓ Bu yerda moliyaviy ma'lumot topilmadi. "
            "Masalan: «taksiga 20 ming ketdi» deb ayting."
        )
        return

    # 4) Persist.
    async with AsyncSessionLocal() as session:
        await get_or_create_user(
            session,
            user_id=message.from_user.id,
            username=message.from_user.username,
            full_name=message.from_user.full_name,
            language_code=message.from_user.language_code,
        )
        await add_transaction(
            session,
            user_id=message.from_user.id,
            amount=parsed.amount,
            tx_type=parsed.type,
            category=parsed.category,
            description=parsed.description,
            tx_date=parsed.date,
            raw_text=text,
        )
        balance = await get_balance(session, message.from_user.id)

    # 5) Confirm.
    emoji = _TYPE_EMOJI.get(parsed.type, "•")
    await status.edit_text(
        f"{emoji} <b>Saqlandi!</b>\n\n"
        f"Summa: <b>{_fmt_money(parsed.amount)}</b>\n"
        f"Turi: {parsed.type}\n"
        f"Kategoriya: {parsed.category}\n"
        f"Izoh: {parsed.description}\n"
        f"Sana: {parsed.date.isoformat()}\n\n"
        f"💰 Yangi balans: <b>{_fmt_money(balance)}</b>",
        reply_markup=_web_app_keyboard(),
    )


# Fallback for plain text — let users type transactions too.
@router.message(F.text & ~F.text.startswith("/"))
async def handle_text(message: Message) -> None:
    parsed = await extract_transaction(message.text)
    if parsed is None:
        await message.answer(
            "❓ Tushunmadim. Ovozli xabar yuboring yoki «taksiga 20 ming» "
            "kabi yozing."
        )
        return

    async with AsyncSessionLocal() as session:
        await get_or_create_user(
            session,
            user_id=message.from_user.id,
            username=message.from_user.username,
            full_name=message.from_user.full_name,
        )
        await add_transaction(
            session,
            user_id=message.from_user.id,
            amount=parsed.amount,
            tx_type=parsed.type,
            category=parsed.category,
            description=parsed.description,
            tx_date=parsed.date,
            raw_text=message.text,
        )
        balance = await get_balance(session, message.from_user.id)

    emoji = _TYPE_EMOJI.get(parsed.type, "•")
    await message.answer(
        f"{emoji} Saqlandi: <b>{_fmt_money(parsed.amount)}</b> "
        f"({parsed.category})\n💰 Balans: <b>{_fmt_money(balance)}</b>",
        reply_markup=_web_app_keyboard(),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
async def main() -> None:
    await init_db()
    bot = Bot(
        token=settings.BOT_TOKEN,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    dp = Dispatcher()
    dp.include_router(router)

    logger.info("Hisobchi AI bot started (long-polling).")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
