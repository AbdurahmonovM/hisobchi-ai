"""
config.py
=========
Centralised, type-safe application configuration.

All secrets and environment-specific values live in a `.env` file (see
`.env.example`) and are loaded here exactly once via a cached singleton.
Import `settings` anywhere in the project:

    from config import settings
    print(settings.BOT_TOKEN)
"""

from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly-typed application settings, validated at startup."""

    # --- Telegram ---
    BOT_TOKEN: str = Field(..., description="Token from @BotFather")
    # Public HTTPS URL where the Web App is served (required by Telegram).
    # On Railway this is your service domain. We try to auto-detect if not set.
    WEB_APP_URL: str = Field(
        default="", 
        validation_alias=AliasChoices("WEB_APP_URL", "RAILWAY_STATIC_URL", "RAILWAY_PUBLIC_DOMAIN"),
        description="Public HTTPS base URL of the Web App"
    )

    # Secret shared with Telegram so we can verify incoming webhook requests
    # (sent back by Telegram in the X-Telegram-Bot-Api-Secret-Token header).
    WEBHOOK_SECRET: str = Field(
        default="change-me-please", description="Random secret for webhook verification"
    )
    # Set to "webhook" on Railway (single web process) or "polling" for local dev.
    BOT_MODE: Literal["webhook", "polling"] = "polling"

    # --- AI provider (STT + NLP) ---
    # "groq"   -> FREE: Groq Whisper + Llama (one key from console.groq.com)
    # "openai" -> paid: OpenAI Whisper + GPT-4o-mini
    AI_PROVIDER: Literal["groq", "openai"] = "groq"

    # Provide the key for whichever provider you chose. The other can stay empty.
    GROQ_API_KEY: str = Field(default="", description="Free key from console.groq.com")
    OPENAI_API_KEY: str = Field(default="", description="OpenAI API key (paid)")

    # Optional model overrides. If left empty, sensible per-provider defaults are
    # picked by the `whisper_model` / `nlp_model` properties below.
    WHISPER_MODEL: str = ""
    NLP_MODEL: str = ""

    # --- Database ---
    # Dev default: local async SQLite file. Prod example:
    #   postgresql+asyncpg://user:pass@host:5432/hisobchi
    DATABASE_URL: str = "sqlite+aiosqlite:///./hisobchi.db"

    # --- Server ---
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # --- Behaviour ---
    DEFAULT_CURRENCY: str = "UZS"
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    @field_validator("WEB_APP_URL")
    @classmethod
    def _validate_web_app_url(cls, v: str) -> str:
        """Ensure the URL is set and starts with https (required by Telegram)."""
        if not v:
            # On Railway, RAILWAY_STATIC_URL is often just the domain.
            return v
        
        v = v.rstrip("/")
        # If it's a domain only, prepend https://
        if not v.startswith("http"):
            v = f"https://{v}"
        
        # Telegram WebApp URLs MUST be https
        if v.startswith("http://"):
            v = v.replace("http://", "https://")
            
        return v

    @model_validator(mode="after")
    def _check_ai_key(self) -> "Settings":
        """Ensure the chosen provider actually has its API key set."""
        if self.AI_PROVIDER == "groq" and not self.GROQ_API_KEY:
            raise ValueError("AI_PROVIDER=groq but GROQ_API_KEY is empty")
        if self.AI_PROVIDER == "openai" and not self.OPENAI_API_KEY:
            raise ValueError("AI_PROVIDER=openai but OPENAI_API_KEY is empty")
        return self

    # --- AI provider resolution (used by speech_service & nlp_service) ---
    @property
    def ai_api_key(self) -> str:
        """API key for the active provider."""
        return self.GROQ_API_KEY if self.AI_PROVIDER == "groq" else self.OPENAI_API_KEY

    @property
    def ai_base_url(self) -> str | None:
        """Base URL for the active provider (Groq is OpenAI-API-compatible)."""
        # None lets the OpenAI SDK use its own default endpoint.
        return "https://api.groq.com/openai/v1" if self.AI_PROVIDER == "groq" else None

    @property
    def whisper_model(self) -> str:
        """STT model name — explicit override or per-provider default."""
        if self.WHISPER_MODEL:
            return self.WHISPER_MODEL
        return "whisper-large-v3-turbo" if self.AI_PROVIDER == "groq" else "whisper-1"

    @property
    def nlp_model(self) -> str:
        """NLP model name — explicit override or per-provider default."""
        if self.NLP_MODEL:
            return self.NLP_MODEL
        return "llama-3.3-70b-versatile" if self.AI_PROVIDER == "groq" else "gpt-4o-mini"

    @property
    def webhook_path(self) -> str:
        """URL path Telegram will POST updates to (kept secret-ish)."""
        return f"/webhook/{self.WEBHOOK_SECRET}"

    @property
    def webhook_url(self) -> str:
        """Absolute webhook URL registered with Telegram via setWebhook."""
        return f"{self.WEB_APP_URL.rstrip('/')}{self.webhook_path}"


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (parsed only once per process)."""
    return Settings()  # type: ignore[call-arg]


# Convenience module-level singleton.
settings = get_settings()
