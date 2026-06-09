"""
database.py
===========
Async SQLAlchemy 2.0 layer for Hisobchi AI.

Contains:
  * The async engine + session factory (works with SQLite or PostgreSQL).
  * ORM models: `User` and `Transaction`.
  * A small set of repository-style helper coroutines used by the bot and the
    FastAPI server so query logic lives in one place.

The schema is intentionally simple — one user has many transactions.
"""

from __future__ import annotations

import datetime as dt
import enum
from decimal import Decimal
from typing import AsyncGenerator, Optional, Sequence

from sqlalchemy import (
    BigInteger,
    Date,
    DateTime,
    Enum,
    ForeignKey,
    Numeric,
    String,
    func,
    select,
)
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    selectinload,
)

from config import settings

# ---------------------------------------------------------------------------
# Engine & session factory
# ---------------------------------------------------------------------------
# `echo=False` keeps logs clean; flip to True to see emitted SQL while debugging.
engine = create_async_engine(settings.DATABASE_URL, echo=False, future=True)

AsyncSessionLocal: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=engine,
    expire_on_commit=False,  # let us read attributes after commit without re-query
    autoflush=False,
)


class Base(DeclarativeBase):
    """Declarative base class for all ORM models."""


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class TxType(str, enum.Enum):
    """Transaction direction. Mirrors the JSON the NLP service returns."""

    income = "income"
    expense = "expense"
    transfer = "transfer"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class User(Base):
    __tablename__ = "users"

    # We use the Telegram user id directly as the PK — it is globally unique.
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=False)
    username: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    full_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    language_code: Mapped[Optional[str]] = mapped_column(String(8), nullable=True)
    currency: Mapped[str] = mapped_column(String(8), default=settings.DEFAULT_CURRENCY)

    # Collected during onboarding (/start): the user types these themselves —
    # we never guess or pre-fill any amount.
    first_name: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    last_name: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    monthly_income: Mapped[Decimal] = mapped_column(Numeric(18, 2), default=0)
    is_onboarded: Mapped[bool] = mapped_column(default=False)

    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    transactions: Mapped[list["Transaction"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="selectin",
    )


class Transaction(Base):
    __tablename__ = "transactions"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("users.id", ondelete="CASCADE"), index=True
    )

    # Money is stored as NUMERIC to avoid float rounding errors.
    amount: Mapped[Decimal] = mapped_column(Numeric(18, 2))
    type: Mapped[TxType] = mapped_column(Enum(TxType, name="tx_type"))
    category: Mapped[str] = mapped_column(String(64), default="other")
    description: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    tx_date: Mapped[dt.date] = mapped_column(Date)

    # Audit / provenance fields — handy for debugging the STT/NLP pipeline.
    raw_text: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    user: Mapped["User"] = relationship(back_populates="transactions")


# ---------------------------------------------------------------------------
# Lifecycle helpers
# ---------------------------------------------------------------------------
async def init_db() -> None:
    """Create tables if they don't exist. Call once on startup.

    For real production migrations use Alembic; this is fine for a single-file
    schema and for development.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields a session and guarantees cleanup."""
    async with AsyncSessionLocal() as session:
        yield session


# ---------------------------------------------------------------------------
# Repository helpers (shared by bot + API)
# ---------------------------------------------------------------------------
async def get_or_create_user(
    session: AsyncSession,
    *,
    user_id: int,
    username: Optional[str] = None,
    full_name: Optional[str] = None,
    language_code: Optional[str] = None,
) -> User:
    """Fetch a user by Telegram id, creating the row on first contact."""
    user = await session.get(User, user_id)
    if user is None:
        user = User(
            id=user_id,
            username=username,
            full_name=full_name,
            language_code=language_code,
        )
        session.add(user)
        await session.commit()
    return user


async def update_user_profile(
    session: AsyncSession, *, user_id: int, first_name: str, last_name: str
) -> User:
    """Save the name the user typed during onboarding."""
    user = await session.get(User, user_id)
    if user is None:
        user = User(id=user_id)
        session.add(user)
    user.first_name = first_name
    user.last_name = last_name
    user.full_name = f"{first_name} {last_name}".strip()
    await session.commit()
    await session.refresh(user)
    return user


async def set_monthly_income(
    session: AsyncSession, *, user_id: int, amount: Decimal | float, tx_date: dt.date
) -> None:
    """Store the user's stated monthly income and finish onboarding.

    The income is also recorded as a real income transaction (category=salary)
    so the existing balance logic (income − expense) works without any special
    cases — the starting balance simply equals the monthly income, and every
    expense the user reports later is subtracted from it.
    """
    user = await session.get(User, user_id)
    if user is None:
        user = User(id=user_id)
        session.add(user)
    user.monthly_income = Decimal(str(amount))
    user.is_onboarded = True

    session.add(
        Transaction(
            user_id=user_id,
            amount=Decimal(str(amount)),
            type=TxType.income,
            category="salary",
            description="Oylik daromad",
            tx_date=tx_date,
            raw_text="onboarding: monthly income",
        )
    )
    await session.commit()


async def add_transaction(
    session: AsyncSession,
    *,
    user_id: int,
    amount: Decimal | float,
    tx_type: TxType | str,
    category: str,
    description: Optional[str],
    tx_date: dt.date,
    raw_text: Optional[str] = None,
) -> Transaction:
    """Persist a single transaction and return the hydrated row."""
    tx = Transaction(
        user_id=user_id,
        amount=Decimal(str(amount)),
        type=TxType(tx_type) if isinstance(tx_type, str) else tx_type,
        category=category,
        description=description,
        tx_date=tx_date,
        raw_text=raw_text,
    )
    session.add(tx)
    await session.commit()
    await session.refresh(tx)
    return tx


async def get_balance(session: AsyncSession, user_id: int) -> Decimal:
    """Balance = sum(income) - sum(expense). Transfers are net-neutral here."""
    income = await session.scalar(
        select(func.coalesce(func.sum(Transaction.amount), 0)).where(
            Transaction.user_id == user_id, Transaction.type == TxType.income
        )
    )
    expense = await session.scalar(
        select(func.coalesce(func.sum(Transaction.amount), 0)).where(
            Transaction.user_id == user_id, Transaction.type == TxType.expense
        )
    )
    return Decimal(income or 0) - Decimal(expense or 0)


async def get_recent_transactions(
    session: AsyncSession, user_id: int, limit: int = 15
) -> Sequence[Transaction]:
    """Most recent transactions, newest first."""
    result = await session.scalars(
        select(Transaction)
        .where(Transaction.user_id == user_id)
        .order_by(Transaction.created_at.desc())
        .limit(limit)
    )
    return result.all()


async def get_expense_breakdown(
    session: AsyncSession, user_id: int, year: int, month: int
) -> list[tuple[str, Decimal]]:
    """Sum of expenses grouped by category for a given month.

    Returns a list of (category, total) tuples sorted descending — exactly the
    shape the pie chart needs.
    """
    start = dt.date(year, month, 1)
    end = dt.date(year + (month == 12), (month % 12) + 1, 1)  # first day of next month

    rows = await session.execute(
        select(Transaction.category, func.sum(Transaction.amount))
        .where(
            Transaction.user_id == user_id,
            Transaction.type == TxType.expense,
            Transaction.tx_date >= start,
            Transaction.tx_date < end,
        )
        .group_by(Transaction.category)
        .order_by(func.sum(Transaction.amount).desc())
    )
    return [(cat, Decimal(total)) for cat, total in rows.all()]


async def get_monthly_totals(
    session: AsyncSession, user_id: int, year: int, month: int
) -> tuple[Decimal, Decimal]:
    """Return (income, expense) totals for a given month."""
    start = dt.date(year, month, 1)
    end = dt.date(year + (month == 12), (month % 12) + 1, 1)  # first day of next month

    income = await session.scalar(
        select(func.coalesce(func.sum(Transaction.amount), 0)).where(
            Transaction.user_id == user_id,
            Transaction.type == TxType.income,
            Transaction.tx_date >= start,
            Transaction.tx_date < end,
        )
    )
    expense = await session.scalar(
        select(func.coalesce(func.sum(Transaction.amount), 0)).where(
            Transaction.user_id == user_id,
            Transaction.type == TxType.expense,
            Transaction.tx_date >= start,
            Transaction.tx_date < end,
        )
    )
    return Decimal(income or 0), Decimal(expense or 0)


async def get_daily_totals(
    session: AsyncSession, user_id: int, date: dt.date
) -> tuple[Decimal, Decimal]:
    """Return (income, expense) totals for a specific day."""
    income = await session.scalar(
        select(func.coalesce(func.sum(Transaction.amount), 0)).where(
            Transaction.user_id == user_id,
            Transaction.type == TxType.income,
            Transaction.tx_date == date,
        )
    )
    expense = await session.scalar(
        select(func.coalesce(func.sum(Transaction.amount), 0)).where(
            Transaction.user_id == user_id,
            Transaction.type == TxType.expense,
            Transaction.tx_date == date,
        )
    )
    return Decimal(income or 0), Decimal(expense or 0)


async def get_weekly_totals(
    session: AsyncSession, user_id: int, date: dt.date
) -> tuple[Decimal, Decimal]:
    """Return (income, expense) totals for the week containing the given date."""
    # Week starts on Monday
    start = date - dt.timedelta(days=date.weekday())
    end = start + dt.timedelta(days=7)

    income = await session.scalar(
        select(func.coalesce(func.sum(Transaction.amount), 0)).where(
            Transaction.user_id == user_id,
            Transaction.type == TxType.income,
            Transaction.tx_date >= start,
            Transaction.tx_date < end,
        )
    )
    expense = await session.scalar(
        select(func.coalesce(func.sum(Transaction.amount), 0)).where(
            Transaction.user_id == user_id,
            Transaction.type == TxType.expense,
            Transaction.tx_date >= start,
            Transaction.tx_date < end,
        )
    )
    return Decimal(income or 0), Decimal(expense or 0)
