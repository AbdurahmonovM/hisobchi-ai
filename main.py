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
from database import init_db, engine
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
    logger.info("Initializing database...")
    try:
        await init_db()
        # Simple health check query
        from sqlalchemy import text
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        logger.info("✅ Database connection successful.")
    except Exception as e:
        logger.error("❌ Database connection failed: %s", e)
        # We don't raise here to allow the app to start and show errors in health checks

    # Registry the webhook
    webhook_url = settings.webhook_url
    
    # Basic sanity check
    if "localhost" in webhook_url or "0.0.0.0" in webhook_url or "127.0.0.1" in webhook_url:
        logger.error("!!! CRITICAL ERROR: WEB_APP_URL is set to localhost/0.0.0.0. !!!")
        logger.error("Telegram cannot reach your local machine. Set WEB_APP_URL to your Railway public domain.")
        logger.error("Current WEB_APP_URL: %s", settings.WEB_APP_URL)
    else:
        logger.info("Checking current webhook status...")
        try:
            info = await bot.get_webhook_info()
            logger.info("Current Webhook Info: %s", info.model_dump_json(indent=2))
            
            logger.info("Registering webhook at %s", webhook_url)
            success = await bot.set_webhook(
                url=webhook_url,
                secret_token=settings.WEBHOOK_SECRET,
                drop_pending_updates=True,
                allowed_updates=dp.resolve_used_update_types(),
            )
            if success:
                logger.info("✅ Webhook registered successfully.")
                # Verify again
                new_info = await bot.get_webhook_info()
                logger.info("New Webhook Info: %s", new_info.model_dump_json(indent=2))
            else:
                logger.error("❌ Failed to register webhook (Telegram returned False).")
        except Exception as e:
            logger.exception("❌ Error during webhook registration: %s", e)

    try:
        yield
    finally:
        logger.info("Shutting down...")
        try:
            # Optionally keep webhook on shutdown to avoid missing updates during deploy
            # but usually for webhooks it's fine to keep it.
            # await bot.delete_webhook() 
            await bot.session.close()
            logger.info("Bot session closed.")
        except Exception:
            logger.warning("Error during cleanup")


app = FastAPI(title="Hisobchi AI", lifespan=lifespan)

# Mount the Web App + JSON API (/, /api/summary, /health, /static).
register_web_routes(app)


@app.get("/webhook-info")
async def get_webhook_status():
    """Diagnostic endpoint to check the current Telegram webhook configuration."""
    try:
        info = await bot.get_webhook_info()
        return {
            "status": "ok",
            "webhook_info": info.model_dump(),
            "settings_url": settings.webhook_url,
            "bot_id": (await bot.get_me()).id
        }
    except Exception as e:
        logger.exception("Error getting webhook info")
        return {"status": "error", "message": str(e)}


@app.post("/register-webhook-manually")
async def register_webhook():
    """Manual trigger to re-register the webhook if it fails at startup."""
    try:
        success = await bot.set_webhook(
            url=settings.webhook_url,
            secret_token=settings.WEBHOOK_SECRET,
            drop_pending_updates=False, # Keep updates if manual
            allowed_updates=dp.resolve_used_update_types(),
        )
        return {"success": success, "url": settings.webhook_url}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post(settings.webhook_path)
async def telegram_webhook(
    request: Request,
    x_telegram_bot_api_secret_token: str = Header(default=""),
) -> Response:
    """Receive a Telegram update and dispatch it to the aiogram handlers."""
    # 1. Verify the shared secret — drop anything that isn't really from Telegram.
    if x_telegram_bot_api_secret_token != settings.WEBHOOK_SECRET:
        logger.warning(
            "Rejected webhook call with bad secret token. Expected: %s, Got: %s",
            settings.WEBHOOK_SECRET[:4] + "***",
            x_telegram_bot_api_secret_token[:4] + "***"
        )
        return Response(status_code=status.HTTP_403_FORBIDDEN)

    # 2. Process the update
    try:
        data = await request.json()
        logger.debug("Received Telegram update: %s", data)
        
        update = Update.model_validate(data, context={"bot": bot})
        
        # We use feed_update for aiohttp/fastapi style webhooks.
        # It handles the middleware and router dispatching.
        await dp.feed_update(bot, update)
        
    except Exception as e:
        logger.exception("Error processing Telegram update: %s", e)
        # Even on error, we usually return 200 to Telegram so it doesn't 
        # spam the same broken update; we rely on our logs to fix the bug.
    
    return Response(status_code=status.HTTP_200_OK)
