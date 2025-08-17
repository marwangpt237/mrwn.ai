# app.py
import os
import logging
from pathlib import Path
from typing import List

from fastapi import FastAPI, Request
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

from groq import Groq

from rag import ingest_file, ingest_url, search
from memory import add_message, last_dialogue
from models import GROQ_MODELS, DEFAULT_MODEL_ID
from prompt import mainprompt  # we'll create a small helper to read the prompt

# --- logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mr1-groq-bot")

# --- env
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")  # e.g. https://<railway>.up.railway.app/webhook

if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN missing")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY missing")

client = Groq(api_key=GROQ_API_KEY)

# load prompt text
PROMPT_PATH = Path("prompt/mainprompt.md")
MAIN_PROMPT = PROMPT_PATH.read_text(encoding="utf-8").strip()

# in-memory per-chat model selection
CHAT_MODEL = {}

application = Application.builder().token(TELEGRAM_TOKEN).build()

# helper to detect Arabic vs English
def detect_lang(s: str) -> str:
    return "ar" if any("\u0600" <= ch <= "\u06FF" for ch in s) else "en"

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lang = detect_lang(update.message.text or "")
    await update.message.reply_text("🤖 جاهز!" if lang == "ar" else "🤖 Ready!")

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "/models — list models\n"
        "/setmodel <id> — choose model\n"
        "/addurl <url> — ingest URL\n"
        "Send a file (PDF/MD/CSV/TXT) to ingest\n"
    )
    await update.message.reply_text(text)

async def cmd_models(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lines = [f"{m['id']} — {m['label']}" for m in GROQ_MODELS]
    await update.message.reply_text("\n".join(lines))

async def cmd_setmodel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /setmodel <model_id>")
        return
    model_id = context.args[0]
    if model_id not in [m["id"] for m in GROQ_MODELS]:
        await update.message.reply_text("Unknown model id. Use /models")
        return
    CHAT_MODEL[update.effective_chat.id] = model_id
    await update.message.reply_text(f"✅ Model set to {model_id}")

async def cmd_addurl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /addurl <url>")
        return
    url = context.args[0]
    try:
        n = ingest_url(update.effective_user.id, url)
        await update.message.reply_text(f"✅ Added {n} chunks from: {url}")
    except Exception as e:
        await update.message.reply_text(f"Error ingesting URL: {e}")

async def handle_file(update: Update, context: ContextTypes.DEFAULT_TYPE):
    doc = update.message.document
    if not doc:
        return
    file = await context.bot.get_file(doc.file_id)
    udir = Path("data") / str(update.effective_user.id)
    udir.mkdir(parents=True, exist_ok=True)
    local_path = udir / doc.file_name
    await file.download_to_drive(str(local_path))
    try:
        n = ingest_file(update.effective_user.id, local_path)
        await update.message.reply_text(f"✅ Ingested {n} chunks from {doc.file_name}")
    except Exception as e:
        await update.message.reply_text(f"Error ingesting file: {e}")

def build_messages(user_id: int, user_text: str) -> List[dict]:
    # memory
    history = last_dialogue(user_id, limit=10)
    # rag
    docs = search(user_id, user_text, k=4)
    context_text = "\n\n".join([f"[source: {d['source']}] {d['chunk']}" for d in docs])
    lang = "AR" if detect_lang(user_text) == "ar" else "EN"
    system_content = f"{MAIN_PROMPT}\n\nRAG Context:\n{context_text}\n\nLanguage: {lang}."
    msgs = [{"role": "system", "content": system_content}]
    for role, content in history:
        msgs.append({"role": role, "content": content})
    msgs.append({"role": "user", "content": user_text})
    return msgs

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    text = update.message.text.strip()
    add_message(user_id, "user", text)
    model_id = CHAT_MODEL.get(update.effective_chat.id, DEFAULT_MODEL_ID)
    try:
        messages = build_messages(user_id, text)
        completion = client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=0.4,
            max_tokens=700,
        )
        reply = completion.choices[0].message.content
    except Exception as e:
        logger.exception("Groq error")
        reply = f"Error from Groq: {e}"
    add_message(user_id, "assistant", reply)
    await update.message.reply_text(reply)

# FastAPI app + webhook
app = FastAPI()

@app.on_event("startup")
async def on_start():
    # register handlers once
    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CommandHandler("help", cmd_help))
    application.add_handler(CommandHandler("models", cmd_models))
    application.add_handler(CommandHandler("setmodel", cmd_setmodel))
    application.add_handler(CommandHandler("addurl", cmd_addurl))
    application.add_handler(MessageHandler(filters.Document.ALL, handle_file))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    # set webhook if provided
    if WEBHOOK_URL:
        await application.bot.set_webhook(WEBHOOK_URL)
        logger.info(f"Webhook set to {WEBHOOK_URL}")

@app.post("/webhook")
async def telegram_webhook(request: Request):
    data = await request.json()
    update = Update.de_json(data, application.bot)
    await application.process_update(update)
    return {"ok": True}

@app.get("/")
async def health():
    return {"status": "ok", "models": [m["id"] for m in GROQ_MODELS]}

# run with uvicorn (Docker CMD)
