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
from aiogram.filters import Command, CommandStart, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
    WebAppInfo,
)

from config import settings
from database import (
    AsyncSessionLocal,
    add_transaction,
    get_balance,
    get_monthly_totals,
    get_or_create_user,
    init_db,
    set_monthly_income,
    update_user_profile,
)
from nlp_service import extract_transaction, parse_uzbek_amount
from speech_service import TranscriptionError, transcribe_voice

logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("hisobchi.bot")

router = Router(name="hisobchi")

# Emojis per type for friendlier confirmation messages.
_TYPE_EMOJI = {"income": "🟢", "expense": "🔴", "transfer": "🔁"}


class Onboarding(StatesGroup):
    """Three-step intake on first /start: first name -> last name -> income."""

    first_name = State()
    last_name = State()
    income = State()


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
async def cmd_start(message: Message, state: FSMContext) -> None:
    """Register the user. If new, run onboarding; otherwise greet them back."""
    u = message.from_user
    async with AsyncSessionLocal() as session:
        user = await get_or_create_user(
            session,
            user_id=u.id,
            username=u.username,
            full_name=u.full_name,
            language_code=u.language_code,
        )
        already = user.is_onboarded

    # Returning, fully set-up user → just greet + Web App button.
    if already:
        await state.clear()
        await message.answer(
            f"👋 Qaytganingiz bilan, <b>{user.first_name or 'do‘stim'}</b>!\n\n"
            "Xarajat yoki kirimni ovozli xabar bilan yuboring — men hisoblab "
            "boraman.\n\n"
            "📊 Hisobotni ko'rish uchun tugmani bosing.",
            reply_markup=_web_app_keyboard(),
        )
        return

    # New user → start the 3-step onboarding.
    await state.set_state(Onboarding.first_name)
    await message.answer(
        "👋 <b>Hisobchi AI</b>ga xush kelibsiz!\n\n"
        "Boshlashdan oldin bir-ikki savol beraman.\n\n"
        "1️⃣ <b>Ismingizni</b> yozing:"
    )


# ---------------------------------------------------------------------------
# Onboarding steps (FSM). These are registered BEFORE the generic text/voice
# handlers, and those generic handlers use StateFilter(None), so a user being
# onboarded never falls through to transaction parsing.
# ---------------------------------------------------------------------------
@router.message(Onboarding.first_name, F.text)
async def ob_first_name(message: Message, state: FSMContext) -> None:
    await state.update_data(first_name=message.text.strip())
    await state.set_state(Onboarding.last_name)
    await message.answer("2️⃣ <b>Familyangizni</b> yozing:")


@router.message(Onboarding.last_name, F.text)
async def ob_last_name(message: Message, state: FSMContext) -> None:
    await state.update_data(last_name=message.text.strip())
    await state.set_state(Onboarding.income)
    await message.answer(
        "3️⃣ <b>Oylik daromadingizni</b> yozing.\n"
        "Masalan: <i>5 million</i> yoki <i>5000000</i>"
    )


@router.message(Onboarding.income)
async def ob_income(message: Message, state: FSMContext) -> None:
    # Parse the number locally (no LLM) so we never invent an approximate value.
    amount = parse_uzbek_amount(message.text or "")
    if amount is None or amount <= 0:
        await message.answer(
            "🤔 Summani tushunmadim. Faqat raqam yozing.\n"
            "Masalan: <i>5 million</i> yoki <i>5000000</i>"
        )
        return

    import datetime as dt

    data = await state.get_data()
    first = data.get("first_name", "")
    last = data.get("last_name", "")

    async with AsyncSessionLocal() as session:
        await update_user_profile(
            session, user_id=message.from_user.id, first_name=first, last_name=last
        )
        await set_monthly_income(
            session,
            user_id=message.from_user.id,
            amount=amount,
            tx_date=dt.date.today(),
        )

    await state.clear()
    await message.answer(
        f"✅ Tayyor, <b>{first} {last}</b>!\n\n"
        f"💵 Oylik daromad: <b>{_fmt_money(amount)}</b>\n\n"
        "Endi xarajatlaringizni ovozli xabar bilan yuboring — men ularni "
        "daromadingizdan ayirib, balansni hisoblab boraman.\n\n"
        "Masalan ayting:\n"
        "• <i>«obed uchun 35 ming ketdi»</i>\n"
        "• <i>«taksiga 20 ming»</i>",
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


@router.message(Command("summary"))
async def cmd_summary(message: Message) -> None:
    """Detailed text summary for different periods."""
    import datetime as dt
    from database import get_daily_totals, get_weekly_totals

    today = dt.date.today()
    async with AsyncSessionLocal() as session:
        d_inc, d_exp = await get_daily_totals(session, message.from_user.id, today)
        w_inc, w_exp = await get_weekly_totals(session, message.from_user.id, today)
        m_inc, m_exp = await get_monthly_totals(session, message.from_user.id, today.year, today.month)
        balance = await get_balance(session, message.from_user.id)

    await message.answer(
        f"📊 <b>Moliyaviy hisobot</b>\n\n"
        f"🗓 <b>Bugun:</b>\n"
        f"🟢 +{_fmt_money(d_inc)} | 🔴 −{_fmt_money(d_exp)}\n\n"
        f"📅 <b>Shu hafta:</b>\n"
        f"🟢 +{_fmt_money(w_inc)} | 🔴 −{_fmt_money(w_exp)}\n\n"
        f"📆 <b>Shu oy:</b>\n"
        f"🟢 +{_fmt_money(m_inc)} | 🔴 −{_fmt_money(m_exp)}\n\n"
        f"💰 <b>Jami balans:</b> {_fmt_money(balance)}",
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
        "/summary — kunlik/haftalik/oylik hisobot\n"
        "/start — boshlash va hisobot tugmasi",
    )


# ---------------------------------------------------------------------------
# Voice handler — the core feature
# ---------------------------------------------------------------------------
@router.message(StateFilter(None), F.voice | F.audio)
async def handle_voice(message: Message, bot: Bot) -> None:
    """Transcribe a voice/audio message, extract a transaction, store it."""
    # Make sure the user finished onboarding first.
    async with AsyncSessionLocal() as session:
        user = await get_or_create_user(
            session,
            user_id=message.from_user.id,
            username=message.from_user.username,
            full_name=message.from_user.full_name,
            language_code=message.from_user.language_code,
        )
        if not user.is_onboarded:
            await message.answer("Boshlash uchun /start ni bosing 🙏")
            return

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
@router.message(StateFilter(None), F.text & ~F.text.startswith("/"))
async def handle_text(message: Message) -> None:
    async with AsyncSessionLocal() as session:
        user = await get_or_create_user(
            session,
            user_id=message.from_user.id,
            username=message.from_user.username,
            full_name=message.from_user.full_name,
        )
        if not user.is_onboarded:
            await message.answer("Boshlash uchun /start ni bosing 🙏")
            return

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
