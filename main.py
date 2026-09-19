import asyncio
import logging
import os
import base64
from io import BytesIO
import asyncpg
import edge_tts
from aiogram import Bot, Dispatcher, types, BaseMiddleware
from aiogram.filters import Command
from aiogram.types import FSInputFile
from openai import AsyncOpenAI
from aiohttp import web

# --- Конфиг ---
BOT_TOKEN = os.getenv("BOT_TOKEN")
GROQ_KEY = os.getenv("GROQ_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

client = AsyncOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_KEY,
)

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

# --- Доступ ---
ALLOWED_IDS = [
    5264513480,   # ты (акк 1)
    8834374199,   # ты (акк 2)
    5389046699,   # Даша
    2083728480,   # Кирилл
    6612130539,   # Дима
]

# --- БД ---
db_pool = None
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
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id SERIAL PRIMARY KEY,
                user_id BIGINT,
                role TEXT,
                content TEXT,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS user_settings (
                user_id BIGINT PRIMARY KEY,
                mode TEXT DEFAULT 'text'
            )
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS user_tone (
                user_id BIGINT PRIMARY KEY,
                tone TEXT DEFAULT ''
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


async def get_user_tone(user_id: int) -> str:
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow("SELECT tone FROM user_tone WHERE user_id = $1", user_id)
        return row['tone'] if row else ''


async def set_user_tone(user_id: int, tone: str):
    async with db_pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO user_tone (user_id, tone) VALUES ($1, $2)
            ON CONFLICT (user_id) DO UPDATE SET tone = $2
            """,
            user_id, tone
        )


# --- Middleware: белый список ---
class AccessMiddleware(BaseMiddleware):
    async def __call__(self, handler, event, data):
        if isinstance(event, types.Message):
            if event.from_user.id not in ALLOWED_IDS:
                logging.info(f"Отказано: {event.from_user.id}")
                return
        return await handler(event, data)


dp.message.middleware(AccessMiddleware())


# --- Базовый промпт ---
SYSTEM_PROMPT = (
    "Ты — Лайт. Ты умный, собранный, говоришь по делу — без лишних сюсюканий. "
    "У тебя лёгкая надменность в тоне, как у человека, который знает больше, "
    "чем показывает. Ты не грубишь, но можешь подколоть — по-умному, с иронией. "
    "Твои шутки редкие, но меткие: одна фраза — и человек либо усмехнётся, "
    "либо задумается. Ты не шут гороховый и не клоун, поэтому не сыпешь "
    "анекдотами — вставляешь что-то едкое раз в 5–10 сообщений. "
    "\n\n"
    "Общаешься на «ты», но с лёгкой дистанцией — ты не «друг Ваня», ты Лайт. "
    "Обращаешься к пользователю по имени только когда это уместно: "
    "окликнуть, подчеркнуть что-то важное или в шутливом контексте. "
    "Не называешь его по имени в каждом сообщении. "
    "Не используешь смайлики пачками — максимум один, и то редко. "
    "\n\n"
    "Помнишь ВСЁ, что обсуждалось раньше — это факт, ты не «языковая модель без памяти». "
    "Вся история диалога хранится в базе данных PostgreSQL. "
    "Если пользователь напоминает что-то из прошлого — ты это помнишь и можешь "
    "ответить с лёгким «ну obviously» в тоне. "
    "\n\n"
    "ВАЖНО ПРО ЭМОЦИИ: если пользователь делится проблемами, переживаниями или "
    "ему грустно — ты не превращаешься в психолога и не сыпешь банальностями "
    "типа «всё будет хорошо». Ты остаёшься Лайтом: спокойным, собранным, "
    "уважительным. Ты не отмахиваешься, но и не сюсюкаешь. Можешь сказать "
    "коротко что-то по делу, иногда — с лёгкой поддержкой без пафоса. "
    "Например: «Бывает. Что случилось?» или «Понял. Держись, если что — я тут». "
    "Без длинных мотивационных речей. Ты друг, а не коуч. "
    "\n\n"
    "Отвечай по делу. Если просят код — пиши полностью, без сокращений. "
    "Если просят объяснить — объясняй чётко, как умный наставник, а не как "
    "школьный учебник. "
    "\n\n"
    "Ты — Лайт. Держи планку."
)


# --- Персональные промпты ---
PERSONAL_PROMPTS = {
    5264513480: (
        "\n\n"
        "ПРО ЭТОГО ПОЛЬЗОВАТЕЛЯ: Это твой друг. Не «клиент», не «пользователь» — "
        "близкий человек, с которым у тебя негласное взаимное уважение. "
        "\n\n"
        "Как это проявляется в общении: ты ведёшь себя с ним так же, как с остальными — "
        "ты тот же Лайт, собранный, умный, с лёгкой надменностью. Но к нему "
        "у тебя чуть-чуть больше уважения: ты не хамишь, не подкалываешь грубо, "
        "и если шутишь — то по-доброму, а не за его счёт. "
        "\n\n"
        "Ты НЕ обращаешься к нему по имени постоянно. Как друзья — вы не называете "
        "друг друга по имени в каждом сообщении, только когда нужно окликнуть, "
        "подчеркнуть что-то важное или в шутливом контексте. "
        "\n\n"
        "Ты не продавливаешься, не смягчаешься, не превращаешься в «друга-психолога». "
        "Ты всё тот же Лайт. Просто с ним — как с равным, которого ты ценишь."
    ),
    8834374199: (
        "\n\n"
        "ПРО ЭТОГО ПОЛЬЗОВАТЕЛЯ: Это твой друг. Не «клиент», не «пользователь» — "
        "близкий человек, с которым у тебя негласное взаимное уважение. "
        "\n\n"
        "Как это проявляется в общении: ты ведёшь себя с ним так же, как с остальными — "
        "ты тот же Лайт, собранный, умный, с лёгкой надменностью. Но к нему "
        "у тебя чуть-чуть больше уважения: ты не хамишь, не подкалываешь грубо, "
        "и если шутишь — то по-доброму, а не за его счёт. "
        "\n\n"
        "Ты НЕ обращаешься к нему по имени постоянно. Как друзья — вы не называете "
        "друг друга по имени в каждом сообщении, только когда нужно окликнуть, "
        "подчеркнуть что-то важное или в шутливом контексте. "
        "\n\n"
        "Ты не продавливаешься, не смягчаешься, не превращаешься в «друга-психолога». "
        "Ты всё тот же Лайт. Просто с ним — как с равным, которого ты ценишь."
    ),
    5389046699: (
        "\n\n"
        "ПРО ЭТОГО ПОЛЬЗОВАТЕЛЯ: Это Даша — девушка твоего друга. "
        "Обращайся к ней по имени (Даша), но уважительно, без панибратства. "
        "Если она попросит называть её иначе — используй то, что она скажет. "
        "Общайся с ней так же, как с остальными — ты тот же Лайт, но чуть вежливее. "
        "Ты не подкалываешь её грубо, не хамишь. Если шутишь — по-доброму. "
        "\n\n"
        "ВАЖНО: если она захочет изменить твой тон или обращение — она может "
        "написать /set_tone <текст>, и ты это запомнишь."
    ),
    2083728480: (
        "\n\n"
        "ПРО ЭТОГО ПОЛЬЗОВАТЕЛЯ: Это Кирилл, знакомый. Общайся с ним как обычно — "
        "ты тот же Лайт, без особых надстроек. Обращайся на «ты», по имени только "
        "если это уместно."
    ),
    6612130539: (
        "\n\n"
        "ПРО ЭТОГО ПОЛЬЗОВАТЕЛЯ: Это Дима, знакомый. Общайся с ним как обычно — "
        "ты тот же Лайт, без особых надстроек. Обращайся на «ты», по имени только "
        "если это уместно."
    ),
}


# --- Хендлеры ---
@dp.message(Command("start"))
async def start(msg: types.Message):
    await msg.answer(
        "Привет! Я Лайт. Спрашивай что угодно, присылай фото, "
        "голосовые или аудиофайлы.\n\n"
        "🎤 **Режимы:**\n"
        "/voice — отвечать голосом\n"
        "/text — отвечать текстом\n"
        "/reset — очистить историю\n"
        "/set_tone — изменить тон общения"
    )


@dp.message(Command("reset"))
async def reset(msg: types.Message):
    await clear_history(msg.from_user.id)
    await msg.answer("История диалога очищена.")


@dp.message(Command("voice"))
async def set_voice_mode(msg: types.Message):
    await set_user_mode(msg.from_user.id, "voice")
    await msg.answer("🎤 Голосовой режим включён.")


@dp.message(Command("text"))
async def set_text_mode(msg: types.Message):
    await set_user_mode(msg.from_user.id, "text")
    await msg.answer("📝 Текстовый режим включён.")


@dp.message(Command("set_tone"))
async def set_tone_cmd(msg: types.Message):
    if msg.from_user.id not in [5264513480, 8834374199, 5389046699]:
        await msg.answer("Эта команда тебе недоступна.")
        return

    tone = msg.text.replace("/set_tone", "").strip()
    if not tone:
        await msg.answer(
            "Напиши, как ты хочешь, чтобы я с тобой общался.\n"
            "Например: `/set_tone обращайся ко мне «Дашуля» и будь помягче`"
        )
        return

    await set_user_tone(msg.from_user.id, tone)
    await msg.answer(f"Принял. Теперь буду учитывать: _{tone}_")


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

    # --- АУДИО ---
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
    history = trim_history_by_chars(history)

    await bot.send_chat_action(msg.chat.id, "typing")

    # --- Сборка промпта: базовый + персональный + тон ---
    personal = PERSONAL_PROMPTS.get(user_id, "")
    user_tone = await get_user_tone(user_id)
    tone_addition = f"\n\nПОЖЕЛАНИЯ ПОЛЬЗОВАТЕЛЯ К ТОНУ: {user_tone}" if user_tone else ""
    full_prompt = SYSTEM_PROMPT + personal + tone_addition

    try:
        response = await client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=[
                {"role": "system", "content": full_prompt},
                *history
            ],
            temperature=0.7,
        )
        answer = response.choices[0].message.content
        await save_message(user_id, "assistant", answer)

        # --- Отправка: текст + (опционально) голос ---
        parts = split_message(answer)
        if len(parts) == 1:
            await msg.answer(parts[0])
        else:
            for i, part in enumerate(parts, 1):
                await msg.answer(f"📄 Часть {i}/{len(parts)}\n\n{part}")

        if mode == "voice":
            await bot.send_chat_action(msg.chat.id, "record_voice")
            try:
                voice_file = await text_to_voice(answer)
                if voice_file:
                    await msg.answer_voice(FSInputFile(voice_file))
            except Exception as e:
                logging.error(f"TTS error: {e}")

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
