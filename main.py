import asyncio
import logging
import os
import base64
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from openai import AsyncOpenAI
from aiohttp import web

# --- Конфиг ---
BOT_TOKEN = os.getenv("BOT_TOKEN")
GROQ_KEY = os.getenv("GROQ_API_KEY")

client = AsyncOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_KEY,
)

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

# Память диалога
history = {}

# --- Разбивка длинных сообщений ---
def split_message(text: str, limit: int = 4000) -> list[str]:
    """Разбивает длинный текст на куски по limit символов."""
    if len(text) <= limit:
        return [text]

    parts = []
    while text:
        if len(text) <= limit:
            parts.append(text)
            break

        chunk = text[:limit]
        split_pos = chunk.rfind("\n")
        if split_pos == -1:
            split_pos = chunk.rfind(" ")
        if split_pos == -1:
            split_pos = limit

        parts.append(text[:split_pos])
        text = text[split_pos:].lstrip("\n")

    return parts


@dp.message(Command("start"))
async def start(msg: types.Message):
    history[msg.from_user.id] = []
    await msg.answer("Привет! Я бот на Llama 3.3. Спрашивай что угодно или присылай фото.")


@dp.message(Command("reset"))
async def reset(msg: types.Message):
    history[msg.from_user.id] = []
    await msg.answer("Контекст очищен.")


@dp.message()
async def chat(msg: types.Message):
    user_id = msg.from_user.id
    if user_id not in history:
        history[user_id] = []

    # --- Фото ---
    if msg.photo:
        photo = msg.photo[-1]
        file = await bot.get_file(photo.file_id)
        image_data = await bot.download_file(file.file_path)
        base64_image = base64.b64encode(image_data.read()).decode("utf-8")

        user_content = [
            {"type": "text", "text": msg.caption or "Что на этом изображении?"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            },
        ]
        history[user_id].append({"role": "user", "content": user_content})

    # --- Текст ---
    elif msg.text:
        history[user_id].append({"role": "user", "content": msg.text})

    # --- Всё остальное игнорируем ---
    else:
        return

    if len(history[user_id]) > 10:
        history[user_id] = history[user_id][-10:]

    await bot.send_chat_action(msg.chat.id, "typing")

    try:
        response = await client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=[
                {
                    "role": "system",
                    "content": "Ты полезный ассистент. Отвечай по делу. Никогда не сокращай код и не пиши '...' вместо пропущенного — выдавай всё полностью.",
                },
                *history[user_id],
            ],
            temperature=0.7,
        )
        answer = response.choices[0].message.content
        history[user_id].append({"role": "assistant", "content": answer})

        # --- Отправка с разбивкой ---
        parts = split_message(answer)
        if len(parts) == 1:
            await msg.answer(parts[0])
        else:
            for i, part in enumerate(parts, 1):
                await msg.answer(f"📄 Часть {i}/{len(parts)}\n\n{part}")

    except Exception as e:
        error_text = str(e)
        logging.error(f"Ошибка: {error_text}")
        await msg.answer(f"❌ {error_text[:300]}")


# --- Веб-сервер для Render ---
async def handle(request):
    return web.Response(text="Bot is running!")


async def main():
    logging.basicConfig(level=logging.INFO)

    app = web.Application()
    app.router.add_get("/", handle)
    runner = web.AppRunner(app)
    await runner.setup()
    port = int(os.environ.get("PORT", 8080))
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    logging.info(f"Web server on port {port}")

    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
