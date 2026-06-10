"""
main.py — Railway production entrypoint (ROBUST / POLLING mode)
==============================================================
Runs the FastAPI Web App server AND the aiogram bot in ONE process.

Why POLLING instead of webhook on Railway?
  * Webhook needs WEB_APP_URL to be the EXACT public domain, and any error while
    registering it (or a placeholder/wrong domain) makes the bot silently dead
    or crashes startup → failed healthchecks.
  * Polling has none of those dependencies: the bot starts pulling updates from
    Telegram as soon as the process is up — regardless of the domain. The Web
    App URL is then only needed for the "Open report" button, which can be set
    later without breaking the bot.

Design for reliability:
  * The bot runs as a BACKGROUND task, so the web server (and the /health
    endpoint Railway probes) is available immediately.
  * Startup NEVER raises: DB or Telegram errors are logged, not fatal, so the
    container doesn't crash-loop and you can read the logs.

Start command (railway.json / Procfile):
    uvicorn main:app --host 0.0.0.0 --port $PORT
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from fastapi import FastAPI

from bot import router  # all command/voice/text handlers + onboarding FSM
from config import settings
from database import init_db
from web_app_server import register_web_routes

logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("hisobchi.main")

# Bot + dispatcher (MemoryStorage by default → FSM onboarding works).
bot = Bot(
    token=settings.BOT_TOKEN,
    default=DefaultBotProperties(parse_mode=ParseMode.HTML),
)
dp = Dispatcher()
dp.include_router(router)


async def _run_bot() -> None:
    """Background task: drop any stale webhook, then long-poll Telegram."""
    try:
        # A previously-set webhook would block polling with a 409 Conflict, so
        # always clear it first.
        await bot.delete_webhook(drop_pending_updates=True)
        me = await bot.get_me()
        logger.info("✅ Bot connected: @%s (id=%s) — polling started", me.username, me.id)
        await dp.start_polling(bot, handle_signals=False)
    except asyncio.CancelledError:
        raise  # normal shutdown
    except Exception:
        logger.exception("❌ Bot polling crashed")


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Init DB and launch the bot in the background. Never crash on errors."""
    logger.info(
        "Starting Hisobchi AI | provider=%s | model=%s",
        settings.AI_PROVIDER, settings.gemini_model,
    )

    try:
        await init_db()
        logger.info("✅ Database ready (%s)", settings.DATABASE_URL.split("://", 1)[0])
    except Exception:
        logger.exception("❌ Database init failed — bot will error until DB is fixed")

    bot_task = asyncio.create_task(_run_bot())

    try:
        yield
    finally:
        logger.info("Shutting down — stopping bot…")
        bot_task.cancel()
        try:
            await bot_task
        except asyncio.CancelledError:
            pass
        await bot.session.close()
        logger.info("Bot stopped, session closed.")


app = FastAPI(title="Hisobchi AI", lifespan=lifespan)

# Web App + JSON API (/, /api/summary, /health-db, /static).
register_web_routes(app)


@app.get("/health")
async def health() -> dict:
    """Liveness probe for Railway — independent of DB/Telegram so it always
    answers 200 once the web server is up."""
    return {"status": "ok", "app": "hisobchi-ai", "mode": "polling"}
