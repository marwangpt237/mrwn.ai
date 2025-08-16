import os
import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from rag import RAGStore

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========= Load Main Prompt =========
with open("mainprompt.md", "r") as f:
    MAIN_PROMPT = f.read()

# ========= Load Model =========
MODEL_NAME = "tiiuae/falcon-7b-instruct"  # lightweight enough for Railway with 1-2GB
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")

# ========= RAG =========
rag = RAGStore()

# ========= Telegram Token =========
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
if not BOT_TOKEN:
    raise ValueError("⚠️ TELEGRAM_BOT_TOKEN is missing in environment variables!")

# ========= Generate Function =========
def generate_response(query: str) -> str:
    # 1. Search in RAG
    context_docs = rag.search(query, top_k=3)
    context = "\n".join(context_docs)

    # 2. Build prompt
    full_prompt = f"{MAIN_PROMPT}\n\nContext:\n{context}\n\nUser: {query}\nAI:"

    # 3. Tokenize & generate
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.7)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# ========= Telegram Handlers =========
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🤖 MR1 AI Agent ready! Send me your queries.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.message.text
    response = generate_response(query)

    # Save to RAG memory
    rag.add([query])

    await update.message.reply_text(response)


# ========= Main =========
def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("🚀 Bot is running...")
    app.run_polling()


if __name__ == "__main__":
    main()
