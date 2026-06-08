# Hisobchi AI 🎙️💰

Uzbek **voice-driven** personal finance tracker for Telegram. Speak a
transaction in Uzbek (or mixed Uzbek/Russian) → it gets transcribed, parsed by
an LLM into structured data, stored, and visualised in a Telegram Web App.

> _«obed uchun 35 ming ketdi»_ → `expense · 35 000 · food`
> _«zarplata 5 million tushdi»_ → `income · 5 000 000 · salary`
> _«50 ming so'm perevod qildim»_ → `transfer · 50 000`

## 100% free setup 🆓

By default `AI_PROVIDER=groq`, which uses **Groq's free API** for both speech
and NLP — no credit card:

| Need | Free choice | Paid alternative |
|------|-------------|------------------|
| Speech-to-text | Groq `whisper-large-v3-turbo` | OpenAI `whisper-1` |
| NLP extraction | Groq `llama-3.3-70b-versatile` | OpenAI `gpt-4o-mini` |
| Database | SQLite file (local) or Railway Postgres | — |
| Hosting | run locally (`polling`), or Render/Fly.io free tier | Railway |

Get one free key at **https://console.groq.com/keys**, put it in `GROQ_API_KEY`.
To switch back to OpenAI later, set `AI_PROVIDER=openai` and `OPENAI_API_KEY`.

## Architecture

```
            ┌─────────────┐   voice    ┌──────────────────┐
   user ──▶ │  Telegram   │ ─────────▶ │  bot.py (aiogram)│
            └─────────────┘            └────────┬─────────┘
                                                │ ogg bytes
                                  ┌─────────────▼─────────────┐
                                  │ speech_service.py         │  Whisper API
                                  │   → Uzbek text            │  (STT)
                                  └─────────────┬─────────────┘
                                  ┌─────────────▼─────────────┐
                                  │ nlp_service.py            │  GPT-4o-mini
                                  │   → ParsedTransaction(JSON)│ (entity extract)
                                  └─────────────┬─────────────┘
                                  ┌─────────────▼─────────────┐
                                  │ database.py (SQLAlchemy)  │  SQLite / Postgres
                                  └─────────────┬─────────────┘
            ┌─────────────┐  initData  ┌────────▼──────────────┐
   Web App ◀┤  Telegram   │◀──────────▶│ web_app_server.py     │  FastAPI
   (chart)  └─────────────┘   /api      │  + web_app/index.html │  (HMAC-verified)
                                        └───────────────────────┘
```

## Files

| File | Responsibility |
|------|----------------|
| `config.py` | Pydantic settings loaded from `.env` |
| `database.py` | Async SQLAlchemy models + repository helpers |
| `speech_service.py` | Whisper speech-to-text (OGG → Uzbek text) |
| `nlp_service.py` | Prompt + LLM call → validated `ParsedTransaction` |
| `bot.py` | aiogram 3 bot: `/start`, `/balance`, voice & text handlers (router + local polling) |
| `web_app_server.py` | FastAPI: serves frontend + `/api/summary` (signed) |
| `web_app/index.html` | TailwindCSS + Chart.js dashboard |
| `main.py` | **Railway entrypoint** — FastAPI + bot webhook in one process |
| `railway.json` / `Procfile` | Railway build & start config |

## Setup

```bash
python -m venv .venv && . .venv/Scripts/activate   # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env          # then fill in BOT_TOKEN, OPENAI_API_KEY, WEB_APP_URL
```

### Run locally (two processes, polling)

Set `BOT_MODE=polling` in `.env`, then:

```bash
# 1) Web App / API server
uvicorn web_app_server:app --host 0.0.0.0 --port 8000

# 2) The bot (long-polling)
python bot.py
```

> For production (Railway) you instead run the single combined process
> `uvicorn main:app` in **webhook** mode — see **Deploy to Railway** below.

In development expose the server over HTTPS (Telegram requires it for Web Apps):

```bash
ngrok http 8000        # copy the https URL into WEB_APP_URL in .env, restart the bot
```

Then in @BotFather optionally set the Web App via `/setmenubutton`.

## Deploy to Railway 🚂

On Railway the bot runs in **webhook mode** combined with the web server in a
**single process** (`main.py`). Railway's one HTTPS domain serves both the Web
App and the Telegram webhook — no ngrok, no second service.

### 1. Push the code to GitHub
```bash
git init && git add . && git commit -m "Hisobchi AI"
git branch -M main
git remote add origin https://github.com/<you>/hisobchi-ai.git
git push -u origin main
```

### 2. Create the Railway project
1. railway.app → **New Project** → **Deploy from GitHub repo** → pick the repo.
2. Add a database: **New** → **Database** → **Add PostgreSQL**.
3. Railway auto-detects `railway.json` and runs
   `uvicorn main:app --host 0.0.0.0 --port $PORT`.

### 3. Set environment variables (service → **Variables**)
| Variable | Value |
|----------|-------|
| `BOT_TOKEN` | from @BotFather |
| `OPENAI_API_KEY` | your OpenAI key |
| `BOT_MODE` | `webhook` |
| `WEBHOOK_SECRET` | a long random string (`python -c "import secrets;print(secrets.token_urlsafe(32))"`) |
| `DATABASE_URL` | `${{Postgres.DATABASE_URL}}` (references the Postgres plugin) |
| `WEB_APP_URL` | your Railway domain, e.g. `https://hisobchi-production.up.railway.app` |

> **Chicken-and-egg tip:** the domain isn't known until the first deploy.
> Deploy once, then under **Settings → Networking → Generate Domain**, copy the
> URL into `WEB_APP_URL`, and redeploy. On startup the app calls
> `setWebhook(WEB_APP_URL/webhook/<secret>)` automatically.

### 4. Verify
- Visit `https://<domain>/health` → `{"status":"ok"}`.
- Send the bot a voice message — it should reply with the parsed transaction.
- In @BotFather run `/setmenubutton` → set the Web App URL to your domain so the
  dashboard opens from the chat menu.

`config.py` automatically rewrites Railway's `postgres://` URL to the async
`postgresql+asyncpg://` form, so you can paste `${{Postgres.DATABASE_URL}}` as-is.

## Security notes
- The Web App **never trusts a user id from the browser.** The frontend sends
  `Telegram.WebApp.initData`; the server re-computes its HMAC from the bot token
  (`web_app_server._validate_init_data`) and rejects forgeries and stale launches.
- Money is stored as `NUMERIC(18,2)` to avoid float rounding.
- `temperature=0` + `response_format=json_object` make NLP extraction
  deterministic and guaranteed-parseable.

## Notes / extension ideas
- Swap GPT-4o-mini for Claude in `nlp_service.py` (same prompt, Anthropic SDK).
- Add Alembic for real migrations (current `init_db` is create-all).
- Add multi-currency conversion (USD/UZS) — `User.currency` already exists.
