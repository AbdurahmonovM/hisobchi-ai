"""
main.py — Railway production entrypoint
=======================================
Combines the FastAPI Web App server and the aiogram bot (WEBHOOK mode) into a
single process listening on $PORT.

Why one process on Railway?
  * Railway gives the service ONE public HTTPS domain. That single domain serves
    both the Web App (`/`, `/api/summary`) AND the Telegram webhook
    (`/webhook/<secret>`). No ngrok, no second service.
  * Telegram pushes updates to us instead of us long-polling — cheaper and
    instant.

Startup flow (FastAPI lifespan):
  1. create DB tables
  2. tell Telegram our webhook URL (`bot.set_webhook`)
On shutdown we remove the webhook and close the bot session.

Start command (see railway.json / Procfile):
    uvicorn main:app --host 0.0.0.0 --port $PORT
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.types import Update
from fastapi import FastAPI, Header, Request, Response, status

from bot import router  # the same handlers used in local polling mode
from config import settings
from database import init_db
from web_app_server import register_web_routes

logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("hisobchi.main")

# Build the bot + dispatcher once at import time so the webhook handler can
# feed updates into them.
bot = Bot(
    token=settings.BOT_TOKEN,
    default=DefaultBotProperties(parse_mode=ParseMode.HTML),
)
dp = Dispatcher()
dp.include_router(router)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Set up DB + register the Telegram webhook on boot; clean up on shutdown."""
    await init_db()

    # Register the webhook. `secret_token` is echoed back by Telegram in the
    # X-Telegram-Bot-Api-Secret-Token header so we can reject spoofed requests.
    await bot.set_webhook(
        url=settings.webhook_url,
        secret_token=settings.WEBHOOK_SECRET,
        drop_pending_updates=True,
        allowed_updates=dp.resolve_used_update_types(),
    )
    logger.info("Webhook registered at %s", settings.webhook_url)

    try:
        yield
    finally:
        await bot.delete_webhook()
        await bot.session.close()
        logger.info("Webhook removed, bot session closed.")


app = FastAPI(title="Hisobchi AI", lifespan=lifespan)

# Mount the Web App + JSON API (/, /api/summary, /health, /static).
register_web_routes(app)


@app.post(settings.webhook_path)
async def telegram_webhook(
    request: Request,
    x_telegram_bot_api_secret_token: str = Header(default=""),
) -> Response:
    """Receive a Telegram update and dispatch it to the aiogram handlers."""
    # Verify the shared secret — drop anything that isn't really from Telegram.
    if x_telegram_bot_api_secret_token != settings.WEBHOOK_SECRET:
        logger.warning("Rejected webhook call with bad secret token")
        return Response(status_code=status.HTTP_403_FORBIDDEN)

    data = await request.json()
    update = Update.model_validate(data, context={"bot": bot})
    await dp.feed_update(bot, update)
    # Telegram only needs a 200; returning data here would be ignored.
    return Response(status_code=status.HTTP_200_OK)
