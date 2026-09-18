import asyncio
import logging
import os
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from openai import AsyncOpenAI
from aiohttp import web

# --- Конфиг ---
BOT_TOKEN = os.getenv("BOT_TOKEN")
GROQ_KEY = os.getenv("GROQ_API_KEY")

# Groq через OpenAI-совместимый клиент
client = AsyncOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_KEY,
)

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

# Память диалога (в словаре, сбрасывается при перезапуске)
history = {}

@dp.message(Command("start"))
async def start(msg: types.Message):
    history[msg.from_user.id] = []
    await msg.answer("Привет! Я бот на Llama 3.3. Спрашивай что угодно.")

@dp.message(Command("reset"))
async def reset(msg: types.Message):
    history[msg.from_user.id] = []
    await msg.answer("Контекст очищен.")

@dp.message()
async def chat(msg: types.Message):
    user_id = msg.from_user.id
    if user_id not in history:
        history[user_id] = []

    history[user_id].append({"role": "user", "content": msg.text})
    if len(history[user_id]) > 10:
        history[user_id] = history[user_id][-10:]

    await bot.send_chat_action(msg.chat.id, "typing")

    try:
        response = await client.chat.completions.create(
            model="llama-3.3-70b-versatile",   # актуальная модель на Groq
            messages=[
                {"role": "system", "content": "Ты полезный ассистент. Отвечай кратко."},
                *history[user_id]
            ],
            temperature=0.7,
        )
        answer = response.choices[0].message.content
        history[user_id].append({"role": "assistant", "content": answer})
        await msg.answer(answer)

    except Exception as e:
        logging.error(f"Ошибка: {e}")
        await msg.answer("Упс, ошибка. Попробуй позже.")

# --- Веб-сервер для Render (чтобы не было "No open ports detected") ---
async def handle(request):
    return web.Response(text="Bot is running!")

async def main():
    logging.basicConfig(level=logging.INFO)

    # Запускаем веб-сервер на порту, который даёт Render
    app = web.Application()
    app.router.add_get("/", handle)
    runner = web.AppRunner(app)
    await runner.setup()
    port = int(os.environ.get("PORT", 8080))
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    logging.info(f"Web server on port {port}")

    # Запускаем бота (polling)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
