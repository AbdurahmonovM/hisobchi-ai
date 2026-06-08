"""
web_app_server.py
=================
FastAPI server that

  1. serves the Telegram Web App frontend (`web_app/index.html`), and
  2. exposes a JSON API the frontend calls to render balance, the expense
     pie chart, and the recent-transactions list.

SECURITY — Telegram `initData` validation
------------------------------------------
A Web App is just a web page; we must not trust a `user_id` sent from the
browser. Telegram signs the launch parameters (`window.Telegram.WebApp.initData`)
with an HMAC derived from the bot token. We re-compute that HMAC server-side and
reject anything that doesn't match. This both authenticates the user and proves
the request really came from inside Telegram.

The frontend sends the raw `initData` string in the `Authorization` header; the
`verify_init_data` dependency parses it, checks the signature, and returns the
trusted Telegram user id.

Run with:  uvicorn web_app_server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import json
from contextlib import asynccontextmanager
from operator import itemgetter
from urllib.parse import parse_qsl

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from database import (
    User,
    get_balance,
    get_expense_breakdown,
    get_monthly_totals,
    get_recent_transactions,
    get_session,
    init_db,
)

WEB_APP_DIR = "web_app"


# ---------------------------------------------------------------------------
# Telegram initData verification
# ---------------------------------------------------------------------------
def _validate_init_data(init_data: str, bot_token: str, max_age_seconds: int = 86_400) -> dict:
    """Verify Telegram WebApp initData HMAC and return the parsed fields.

    Algorithm (per Telegram docs):
      secret_key = HMAC_SHA256(key="WebAppData", msg=bot_token)
      data_check_string = "\\n".join(sorted "key=value" pairs, excluding `hash`)
      expected_hash = HMAC_SHA256(key=secret_key, msg=data_check_string)
      valid  <=>  expected_hash == provided hash
    """
    parsed = dict(parse_qsl(init_data, strict_parsing=True))
    received_hash = parsed.pop("hash", None)
    if not received_hash:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing hash")

    # Build the data-check-string from the remaining, alphabetically sorted keys.
    data_check_string = "\n".join(
        f"{k}={v}" for k, v in sorted(parsed.items(), key=itemgetter(0))
    )

    secret_key = hmac.new(b"WebAppData", bot_token.encode(), hashlib.sha256).digest()
    expected_hash = hmac.new(
        secret_key, data_check_string.encode(), hashlib.sha256
    ).hexdigest()

    if not hmac.compare_digest(expected_hash, received_hash):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid initData signature")

    # Reject stale launches to limit replay.
    auth_date = int(parsed.get("auth_date", "0"))
    now = int(dt.datetime.now(dt.timezone.utc).timestamp())
    if max_age_seconds and now - auth_date > max_age_seconds:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "initData expired")

    return parsed


async def verify_init_data(
    authorization: str = Header(default=""),
) -> int:
    """FastAPI dependency -> returns the trusted Telegram user id.

    The frontend sends:  Authorization: tma <initData>
    """
    if not authorization:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Authorization header required")

    # Accept both "tma <data>" and a bare initData string.
    init_data = authorization[4:] if authorization.lower().startswith("tma ") else authorization

    parsed = _validate_init_data(init_data, settings.BOT_TOKEN)
    user_json = parsed.get("user")
    if not user_json:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "No user in initData")

    try:
        user = json.loads(user_json)
        return int(user["id"])
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Malformed user") from exc


# ---------------------------------------------------------------------------
# API routes (attached to a router so both the standalone app and the combined
# Railway app in main.py can reuse them).
# ---------------------------------------------------------------------------
web_router = APIRouter()


@web_router.get("/api/summary")
async def api_summary(
    user_id: int = Depends(verify_init_data),
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Everything the dashboard needs in one round-trip.

    Returns balance, current-month expense breakdown (for the pie chart), and
    the recent transactions list.
    """
    today = dt.date.today()

    user = await session.get(User, user_id)
    balance = await get_balance(session, user_id)
    income, expense = await get_monthly_totals(session, user_id, today.year, today.month)
    breakdown = await get_expense_breakdown(session, user_id, today.year, today.month)
    recent = await get_recent_transactions(session, user_id, limit=15)

    full_name = ""
    monthly_income_set = 0.0
    if user is not None:
        full_name = (f"{user.first_name or ''} {user.last_name or ''}").strip()
        monthly_income_set = float(user.monthly_income or 0)

    return {
        "currency": settings.DEFAULT_CURRENCY,
        "name": full_name,
        "balance": float(balance),
        "month": today.strftime("%Y-%m"),
        "monthly_income": float(income),
        "monthly_income_set": monthly_income_set,
        "monthly_expense": float(expense),
        "expenses_by_category": [
            {"category": cat, "amount": float(total)} for cat, total in breakdown
        ],
        "transactions": [
            {
                "id": t.id,
                "amount": float(t.amount),
                "type": t.type.value,
                "category": t.category,
                "description": t.description,
                "date": t.tx_date.isoformat(),
            }
            for t in recent
        ],
    }


@web_router.get("/health")
async def health() -> dict:
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Frontend (static)
# ---------------------------------------------------------------------------
@web_router.get("/")
async def index() -> FileResponse:
    """Serve the Web App entry page."""
    return FileResponse(f"{WEB_APP_DIR}/index.html")


def register_web_routes(app: FastAPI) -> None:
    """Attach the Web App + API routes and static mount onto any FastAPI app.

    Used by this module's standalone `app` and by `main.py`'s combined app.
    """
    app.include_router(web_router)
    # Any extra assets (css/js/images) under web_app/ are served here.
    app.mount("/static", StaticFiles(directory=WEB_APP_DIR), name="static")


# ---------------------------------------------------------------------------
# Standalone app — used in local dev (`uvicorn web_app_server:app`) with the
# bot running separately in polling mode (`python bot.py`).
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(_: FastAPI):
    """Ensure the DB schema exists before serving requests."""
    await init_db()
    yield


app = FastAPI(title="Hisobchi AI", lifespan=lifespan)
register_web_routes(app)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "web_app_server:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
    )
