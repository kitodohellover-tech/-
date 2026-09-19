import asyncio
import logging
import os
import base64
from io import BytesIO
import asyncpg
import edge_tts
from aiogram import Bot, Dispatcher, types, BaseMiddleware
from aiogram.filters import Command
from openai import AsyncOpenAI
from aiohttp import web

# --- Конфиг ---
BOT_TOKEN = os.getenv("BOT_TOKEN")
GROQ_KEY = os.getenv("GROQ_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

# ТВОЙ ID И ID ДРУЗЕЙ
ALLOWED_IDS = [8834374199]

client = AsyncOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_KEY,
)

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

# --- Инициализация БД ---
db_pool = None

# --- Настройки истории (УМЕНЬШИЛИ, чтобы не упираться в лимиты Groq) ---
HISTORY_LIMIT = 20
MAX_CONTEXT_CHARS = 15000

# --- Разбивка длинных сообщений ---
def split_message(text: str, limit: int = 4000) -> list[str]:
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

def trim_history_by_chars(history: list[dict], max_chars: int = MAX_CONTEXT_CHARS) -> list[dict]:
    total = 0
    trimmed = []
    for msg in reversed(history):
        content = msg["content"]
        length = len(content) if isinstance(content, str) else sum(
            len(p.get("text", "")) + len(p.get("image_url", {}).get("url", ""))
            for p in content
        )
        if total + length > max_chars:
            break
        trimmed.append(msg)
        total += length
    return list(reversed(trimmed))

# --- Транскрипция аудио (Whisper) ---
async def transcribe_audio(file_id: str, file_ext: str = "ogg") -> str:
    file = await bot.get_file(file_id)
    audio_data = await bot.download_file(file.file_path)
    buffer = BytesIO(audio_data.read())
    buffer.name = f"audio.{file_ext}"
    transcription = await client.audio.transcriptions.create(
        model="whisper-large-v3",
        file=buffer,
        language="ru",
    )
    return transcription.text

# --- Генерация голоса (TTS) ---
async def text_to_voice(text: str) -> str:
    # Ограничиваем длину для TTS, чтобы не перегружать
    clean_text = text[:3000].replace("*", "").replace("`", "").strip()
    if not clean_text:
        return None
    
    communicate = edge_tts.Communicate(clean_text, "ru-RU-DmitryNeural")
    output_file = "response_voice.mp3"
    await communicate.save(output_file)
    return output_file

# --- Работа с БД ---
async def init_db():
    async with db_pool.acquire() as conn:
        # Таблица сообщений
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id SERIAL PRIMARY KEY,
                user_id BIGINT,
                role TEXT,
                content TEXT,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)
        # Таблица настроек (режим пользователя)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS user_settings (
                user_id BIGINT PRIMARY KEY,
                mode TEXT DEFAULT 'text'
            )
        """)

async def save_message(user_id: int, role: str, content: str):
    async with db_pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO messages (user_id, role, content) VALUES ($1, $2, $3)",
            user_id, role, content
        )

async def get_history(user_id: int, limit: int = HISTORY_LIMIT):
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT role, content FROM messages WHERE user_id = $1 ORDER BY id DESC LIMIT $2",
            user_id, limit
        )
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

async def clear_history(user_id: int):
    async with db_pool.acquire() as conn:
        await conn.execute("DELETE FROM messages WHERE user_id = $1", user_id)

async def get_user_mode(user_id: int) -> str:
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow("SELECT mode FROM user_settings WHERE user_id = $1", user_id)
        return row['mode'] if row else 'text'

async def set_user_mode(user_id: int, mode: str):
    async with db_pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO user_settings (user_id, mode) VALUES ($1, $2)
            ON CONFLICT (user_id) DO UPDATE SET mode = $2
            """,
            user_id, mode
        )

# --- Middleware: Белый список ---
class AccessMiddleware(BaseMiddleware):
    async def __call__(self, handler, event, data):
        if isinstance(event, types.Message):
            if event.from_user.id not in ALLOWED_IDS:
                logging.info(f"Отказано: {event.from_user.id}")
                return
        return await handler(event, data)

dp.message.middleware(AccessMiddleware())

# --- Системный промпт ---
SYSTEM_PROMPT = (
    "Ты — умный и дружелюбный ассистент с ПОСТОЯННОЙ ПАМЯТЬЮ. "
    "Вся история диалога хранится в базе данных PostgreSQL. "
    "Ты ПОМНИШЬ всё, что обсуждалось раньше: имена, прозвища, договорённости. "
    "Никогда не говори, что 'не помнишь между сессиями'. "
    "Общайся неформально, на 'ты'. Отвечай по делу."
)

# --- Хендлеры ---
@dp.message(Command("start"))
async def start(msg: types.Message):
    await msg.answer(
        "Привет! Я бот с памятью. Спрашивай что угодно, присылай фото, "
        "голосовые или аудиофайлы.\n\n"
        "🎤 **Режимы:**\n"
        "/voice — отвечать голосом\n"
        "/text — отвечать текстом\n"
        "/reset — очистить историю диалога"
    )

@dp.message(Command("reset"))
async def reset(msg: types.Message):
    await clear_history(msg.from_user.id)
    await msg.answer("История диалога очищена.")

@dp.message(Command("voice"))
async def set_voice_mode(msg: types.Message):
    await set_user_mode(msg.from_user.id, "voice")
    await msg.answer("🎤 Режим голосового ответа включён. Теперь я буду озвучивать свои ответы.")

@dp.message(Command("text"))
async def set_text_mode(msg: types.Message):
    await set_user_mode(msg.from_user.id, "text")
    await msg.answer("📝 Режим текстового ответа включён.")

@dp.message()
async def chat(msg: types.Message):
    user_id = msg.from_user.id
    mode = await get_user_mode(user_id)
    
    history = await get_history(user_id)
    user_content = None
    save_text = None

    # --- ФОТО ---
    if msg.photo:
        photo = msg.photo[-1]
        file = await bot.get_file(photo.file_id)
        image_data = await bot.download_file(file.file_path)
        base64_image = base64.b64encode(image_data.read()).decode("utf-8")
        user_content = [
            {"type": "text", "text": msg.caption or "Что на этом изображении?"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
        ]
        save_text = msg.caption or "[Фото]"

    # --- ГОЛОСОВОЕ ---
    elif msg.voice:
        await bot.send_chat_action(msg.chat.id, "typing")
        try:
            text = await transcribe_audio(msg.voice.file_id, "ogg")
        except Exception as e:
            await msg.answer(f"❌ Ошибка распознавания: {str(e)[:200]}")
            return
        user_content = text
        save_text = f"[Голосовое]: {text}"

    # --- АУДИОФАЙЛ ---
    elif msg.audio:
        await bot.send_chat_action(msg.chat.id, "typing")
        ext = "mp3"
        if msg.audio.file_name and "." in msg.audio.file_name:
            ext = msg.audio.file_name.rsplit(".", 1)[-1].lower()
        try:
            text = await transcribe_audio(msg.audio.file_id, ext)
        except Exception as e:
            await msg.answer(f"❌ Ошибка распознавания: {str(e)[:200]}")
            return
        user_content = text
        save_text = f"[Аудио]: {text}"

    # --- ТЕКСТ ---
    elif msg.text:
        user_content = msg.text
        save_text = msg.text
    else:
        return

    await save_message(user_id, "user", save_text)
    history.append({"role": "user", "content": user_content})
    
    # Обрезаем историю, чтобы не упираться в лимиты
    history = trim_history_by_chars(history)

    await bot.send_chat_action(msg.chat.id, "typing")

    try:
        response = await client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                *history
            ],
            temperature=0.7,
        )
        answer = response.choices[0].message.content
        await save_message(user_id, "assistant", answer)

               # --- ОТПРАВКА: гибрид (текст + голос) ---
        # 1. Сначала отправляем текст (с разбивкой, если длинный)
        parts = split_message(answer)
        if len(parts) == 1:
            await msg.answer(parts[0])
        else:
            for i, part in enumerate(parts, 1):
                await msg.answer(f"📄 Часть {i}/{len(parts)}\n\n{part}")

        # 2. Потом озвучиваем (если режим voice или всегда — на твой выбор)
        if mode == "voice":
            await bot.send_chat_action(msg.chat.id, "record_voice")
            try:
                voice_file = await text_to_voice(answer)
                if voice_file:
                    from aiogram.types import FSInputFile
                    await msg.answer_voice(FSInputFile(voice_file))
            except Exception as e:
                logging.error(f"TTS error: {e}")
                # Если TTS упал — текст уже отправлен, ничего не делаем

    except Exception as e:
        error_text = str(e)
        logging.error(f"Ошибка: {error_text}")
        await msg.answer(f"❌ {error_text[:300]}")

# --- Веб-сервер для Render ---
async def handle(request):
    return web.Response(text="Bot is running!")

async def main():
    global db_pool
    logging.basicConfig(level=logging.INFO)
    
    db_pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=5)
    await init_db()
    logging.info("База данных подключена")

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
