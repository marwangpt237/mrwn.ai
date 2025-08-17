# main.py
# Single-file MR1 Groq Telegram RAG Bot
# - FastAPI webhook
# - Telegram bot (python-telegram-bot v20+)
# - Groq API client usage (groq package)
# - FAISS RAG ingestion (PDF/MD/CSV/TXT + URLs)
# - per-user SQLite memory
# Requirements: see project's requirements.txt

import os
import re
import json
import time
import logging
import sqlite3
import traceback
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import faiss
import numpy as np
import pandas as pd
import httpx
from bs4 import BeautifulSoup
from readability import Document
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from fastapi import FastAPI, Request
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
    ContextTypes,
)

# Groq client (python package)
try:
    from groq import Groq
except Exception:
    Groq = None  # will raise later if missing

# -------------------------
# Config / Environment
# -------------------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")  # e.g. https://<railway>.up.railway.app/webhook
EMBEDDER_NAME = os.getenv("EMBEDDER_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN environment variable is required.")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY environment variable is required.")

# -------------------------
# Logging
# -------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mr1-groq-bot")

# -------------------------
# Model catalog (editable)
# -------------------------
GROQ_MODELS = [
    {"id": "llama-3.1-8b-instant", "label": "Llama 3.1 8B Instant"},
    {"id": "llama-3.1-70b-versatile", "label": "Llama 3.1 70B Versatile"},
    {"id": "gemma-3-270m", "label": "Gemma 3 270M"},
    {"id": "mixtral-8x7b-32768", "label": "Mixtral 8x7B 32K"},
]
DEFAULT_MODEL_ID = GROQ_MODELS[0]["id"]

# -------------------------
# Storage paths
# -------------------------
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
MEMORY_DB = DATA_DIR / "memory.sqlite3"
# -------------------------
# SQLite memory setup
# -------------------------
_conn = sqlite3.connect(str(MEMORY_DB), check_same_thread=False)
_conn.execute(
    """
CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id INTEGER NOT NULL,
  role TEXT NOT NULL,
  content TEXT NOT NULL,
  ts DATETIME DEFAULT CURRENT_TIMESTAMP
)
"""
)
_conn.commit()


def add_message(user_id: int, role: str, content: str) -> None:
    _conn.execute(
        "INSERT INTO messages(user_id, role, content) VALUES(?,?,?)",
        (user_id, role, content),
    )
    _conn.commit()


def last_dialogue(user_id: int, limit: int = 10) -> List[Tuple[str, str]]:
    cur = _conn.execute(
        "SELECT role, content FROM messages WHERE user_id=? ORDER BY id DESC LIMIT ?",
        (user_id, limit),
    )
    rows = cur.fetchall()[::-1]
    return rows


# -------------------------
# RAG (FAISS) utilities
# per-user index at data/<user_id>/faiss.index + meta.jsonl
# -------------------------
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

_embedder: Optional[SentenceTransformer] = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        logger.info(f"Loading embedder: {EMBEDDER_NAME}")
        _embedder = SentenceTransformer(EMBEDDER_NAME)
    return _embedder


def clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = clean_text(text)
    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i : i + size]
        chunks.append(chunk)
        i += size - overlap
    return [c for c in chunks if c.strip()]


def user_paths(user_id: int) -> Tuple[Path, Path, Path]:
    udir = DATA_DIR / str(user_id)
    udir.mkdir(parents=True, exist_ok=True)
    index_path = udir / "faiss.index"
    meta_path = udir / "meta.jsonl"
    return udir, index_path, meta_path


def embed_texts(texts: List[str]) -> np.ndarray:
    emb = get_embedder()
    vecs = emb.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return vecs.astype("float32")


def load_text_from_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = [p.extract_text() or "" for p in reader.pages]
    return "\n".join(pages)


def load_text_from_csv(path: Path, max_rows: int = 2000) -> str:
    df = pd.read_csv(path).head(max_rows)
    return df.to_csv(index=False)


def load_text_from_file(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in [".txt", ".md"]:
        return path.read_text(encoding="utf-8", errors="ignore")
    if ext == ".csv":
        return load_text_from_csv(path)
    if ext == ".pdf":
        return load_text_from_pdf(path)
    raise ValueError("Unsupported file type")


def fetch_url_text(url: str, timeout: int = 15) -> str:
    with httpx.Client(follow_redirects=True, timeout=timeout) as client:
        r = client.get(url)
        r.raise_for_status()
        html = r.text
    doc = Document(html)
    try:
        summary_html = doc.summary(html_partial=True)
    except Exception:
        summary_html = html
    soup = BeautifulSoup(summary_html, "html.parser")
    text = soup.get_text(" ")
    return clean_text(text)


def load_index_if_exists(index_path: Path, dim: int):
    if index_path.exists():
        return faiss.read_index(str(index_path))
    return faiss.IndexFlatIP(dim)


def save_index(index: faiss.Index, index_path: Path):
    faiss.write_index(index, str(index_path))


def add_texts_to_user(user_id: int, texts: List[str], source: str) -> int:
    if not texts:
        return 0
    chunks = []
    for t in texts:
        chunks.extend(chunk_text(t))
    if not chunks:
        return 0

    vecs = embed_texts(chunks)
    dim = vecs.shape[1]
    udir, index_path, meta_path = user_paths(user_id)
    index = load_index_if_exists(index_path, dim)
    start = index.ntotal
    index.add(vecs)
    save_index(index, index_path)

    with open(meta_path, "a", encoding="utf-8") as f:
        for ch in chunks:
            meta = {"source": source, "chunk": ch[:2400], "ts": int(time.time())}
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")
    return index.ntotal - start


def search_user(user_id: int, query: str, k: int = 4) -> List[Dict]:
    udir, index_path, meta_path = user_paths(user_id)
    if not index_path.exists() or not meta_path.exists():
        return []
    index = faiss.read_index(str(index_path))
    qvec = embed_texts([query])
    D, I = index.search(qvec, k)
    metas = []
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_lines = [json.loads(line) for line in f]
    for idx in I[0]:
        if 0 <= idx < len(meta_lines):
            metas.append(meta_lines[idx])
    return metas


def ingest_file_for_user(user_id: int, file_path: Path) -> int:
    txt = load_text_from_file(file_path)
    return add_texts_to_user(user_id, [txt], source=file_path.name)


def ingest_url_for_user(user_id: int, url: str) -> int:
    txt = fetch_url_text(url)
    return add_texts_to_user(user_id, [txt], source=url)


# -------------------------
# Language detection
# -------------------------
def detect_lang(s: str) -> str:
    # crude Arabic detection: presence of Arabic character range
    if any("\u0600" <= ch <= "\u06FF" for ch in s):
        return "ar"
    return "en"


# -------------------------
# Groq client wrapper
# -------------------------
if Groq is None:
    raise RuntimeError("groq package not installed. Add groq==0.9.0 to requirements.")

groq_client = Groq(api_key=GROQ_API_KEY)


def groq_chat_completion(model: str, messages: List[dict], temperature: float = 0.4, max_tokens: int = 700) -> str:
    # messages is list of {"role":.., "content":..}
    resp = groq_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    # The SDK returns choices...structure may vary; adapt if needed.
    try:
        return resp.choices[0].message.content
    except Exception:
        # Fallback: try raw
        return str(resp)


# -------------------------
# Telegram handlers + FastAPI app
# -------------------------
application = Application.builder().token(TELEGRAM_TOKEN).build()

# per-chat model selection (in memory)
CHAT_MODEL: Dict[int, str] = {}


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_lang = detect_lang(update.message.text or "")
    text = "🤖 جاهز! أرسل رسالة، أو استخدم /help" if user_lang == "ar" else "🤖 Ready! Send a message or use /help"
    # show model selection inline
    keyboard = [
        [{"text": m["label"], "callback_data": f"model::{m['id']}"}] for m in GROQ_MODELS
    ]
    # telegram InlineKeyboard needs list[list[InlineKeyboardButton]]
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup

    kb = InlineKeyboardMarkup([[InlineKeyboardButton(m["label"], callback_data=f"model::{m['id']}")] for m in GROQ_MODELS])
    await update.message.reply_text(text, reply_markup=kb)


async def cb_query_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    data = q.data or ""
    if data.startswith("model::"):
        model_id = data.split("::", 1)[1]
        CHAT_MODEL[q.message.chat.id] = model_id
        await q.edit_message_text(text=f"✅ Model set to {model_id}")
        return
    await q.edit_message_text(text="Unknown action")


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "/models — list Groq models\n"
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
        n = ingest_url_for_user(update.effective_user.id, url)
        await update.message.reply_text(f"✅ Added {n} chunks from: {url}")
    except Exception as e:
        logger.exception("ingest_url error")
        await update.message.reply_text(f"Error ingesting URL: {e}")


async def handle_file(update: Update, context: ContextTypes.DEFAULT_TYPE):
    doc = update.message.document
    if not doc:
        return
    file = await context.bot.get_file(doc.file_id)
    udir = DATA_DIR / str(update.effective_user.id)
    udir.mkdir(parents=True, exist_ok=True)
    local_path = udir / doc.file_name
    await file.download_to_drive(str(local_path))
    try:
        n = ingest_file_for_user(update.effective_user.id, local_path)
        await update.message.reply_text(f"✅ Ingested {n} chunks from {doc.file_name}")
    except Exception as e:
        logger.exception("ingest_file error")
        await update.message.reply_text(f"Error ingesting file: {e}")


def build_chat_messages(user_id: int, user_text: str) -> List[dict]:
    # memory
    history = last_dialogue(user_id, limit=10)
    # rag
    docs = search_user(user_id, user_text, k=4)
    context_text = "\n\n".join([f"[source: {d['source']}] {d['chunk']}" for d in docs])
    lang = "AR" if detect_lang(user_text) == "ar" else "EN"

    # load system prompt from prompt/mainprompt.md if exists
    prompt_path = Path("prompt") / "mainprompt.md"
    if prompt_path.exists():
        system_prompt = prompt_path.read_text(encoding="utf-8")
    else:
        system_prompt = (
            "You are MR1 Agent — helpful, safety-first assistant for technical & research queries.\n"
            "- Reply in the same language as the user (Arabic or English).\n"
            "- Use RAG results first (cite source filenames or URLs in brackets).\n"
            "- Keep answers concise, structured, and include code blocks where appropriate.\n"
            "- Maintain per-user memory and include relevant recent context."
        )

    system_content = f"{system_prompt}\n\nRAG Context:\n{context_text}\n\nLanguage: {lang}."
    msgs = [{"role": "system", "content": system_content}]
    for role, content in history:
        msgs.append({"role": role, "content": content})
    msgs.append({"role": "user", "content": user_text})
    return msgs


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    text = (update.message.text or "").strip()
    if not text:
        return
    add_message(user_id, "user", text)

    model_id = CHAT_MODEL.get(update.effective_chat.id, DEFAULT_MODEL_ID)
    messages = build_chat_messages(user_id, text)

    try:
        reply = groq_chat_completion(model=model_id, messages=messages, temperature=0.4, max_tokens=700)
    except Exception as e:
        logger.exception("Groq error")
        reply = f"Error from Groq: {e}"

    add_message(user_id, "assistant", reply)
    await update.message.reply_text(reply)


# Register handlers
application.add_handler(CommandHandler("start", cmd_start))
application.add_handler(CallbackQueryHandler(cb_query_handler))
application.add_handler(CommandHandler("help", cmd_help))
application.add_handler(CommandHandler("models", cmd_models))
application.add_handler(CommandHandler("setmodel", cmd_setmodel))
application.add_handler(CommandHandler("addurl", cmd_addurl))
application.add_handler(MessageHandler(filters.Document.ALL, handle_file))
application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

# -------------------------
# FastAPI app (webhook)
# -------------------------
app = FastAPI()


@app.on_event("startup")
async def on_startup():
    # set webhook if provided
    if WEBHOOK_URL:
        try:
            await application.bot.set_webhook(WEBHOOK_URL)
            logger.info(f"Webhook set to {WEBHOOK_URL}")
        except Exception:
            logger.exception("Failed to set webhook")
    # If no webhook provided, we start polling when running locally.
    # But on Railway we want webhook; polling inside container is not ideal.
    # (We don't call run_polling() here because FastAPI will be used to receive webhooks.)
    logger.info("Bot startup complete")


@app.post("/webhook")
async def telegram_webhook(request: Request):
    try:
        data = await request.json()
        update = Update.de_json(data, application.bot)
        await application.process_update(update)
        return {"ok": True}
    except Exception:
        traceback.print_exc()
        return {"ok": False, "error": "failed to process update"}


@app.get("/")
async def health():
    return {"status": "ok", "models": [m["id"] for m in GROQ_MODELS]}


# If you want to run locally (polling) for tests:
if __name__ == "__main__":
    logger.info("Starting polling (local test mode)")
    application.run_polling()
