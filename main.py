import os
import torch
from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer
from telegram.ext import Application, CommandHandler, MessageHandler, filters

# تحميل الموديل عند التشغيل
MODEL_NAME = "google/gemma-3-270m-it"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    device_map="cpu"
)

app = FastAPI()

# Telegram Bot
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
application = Application.builder().token(TELEGRAM_TOKEN).build()

async def start(update, context):
    await update.message.reply_text("🤖 أنا جاهز!")

async def chat(update, context):
    text = update.message.text
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=128)
    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
    await update.message.reply_text(reply)

application.add_handler(CommandHandler("start", start))
application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat))

@app.on_event("startup")
async def startup():
    application.run_polling()

@app.get("/")
async def root():
    return {"status": "ok", "model": MODEL_NAME}
