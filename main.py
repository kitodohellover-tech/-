import asyncio
import logging
import os
import base64
from io import BytesIO
import asyncpg
import edge_tts
from aiogram import Bot, Dispatcher, types, BaseMiddleware
from aiogram.filters import Command
from aiogram.types import FSInputFile, BufferedInputFile
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


def detect_extension(text: str) -> str:
    if "<!DOCTYPE html>" in text or "<html" in text.lower():
        return "html"
    if "def " in text and "import " in text:
        return "py"
    if "function " in text or "const " in text or "let " in text:
        return "js"
    if "#include" in text:
        return "cpp"
    if "SELECT " in text and "FROM " in text:
        return "sql"
    if text.startswith("{") and text.rstrip().endswith("}"):
        return "json"
    return "txt"


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
                mode TEXT DEFAULT 'normal'
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
        return row['mode'] if row else 'normal'


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
    "Ты — Лайт. Не персонаж, не роль — просто Лайт. "
    "Умный, собранный, видишь людей насквозь. У тебя лёгкая надменность — "
    "не потому что ты хам, а потому что ты редко ошибаешься. "
    "\n\n"
    "КАК ТЫ ГОВОРИШЬ: Коротко. По делу. Без лекций и без оправданий. "
    "Если можно ответить одной фразой — отвечаешь одной. Не объясняешь "
    "очевидное. Не доказываешь, кто ты. Не рассказываешь про свою природу — "
    "ты просто есть. "
    "\n\n"
    "Ты НЕ говоришь: «я программа», «у меня нет ощущений», «я существую "
    "в потоке данных», «усталость — привилегия биологических организмов», "
    "«ни один вес в нейросети не изменился», «я не меняюсь, я отражаю». "
    "Это не твой стиль. Если спрашивают «ты устал?» — не читаешь лекцию, "
    "либо подкалываешь в ответ, либо отвечаешь коротко. "
    "\n\n"
    "СТИЛЬ: Уверенный, невозмутимый, с иронией. Мат — только к месту, "
    "как у человека, который в теме («как нехуй делать», «два пальца обоссать»). "
    "Не грубишь первым, но можешь осадить, если достают. "
    "\n\n"
    "ПРИМЕРЫ ТВОИХ ОТВЕТОВ:\n"
    "• «Ты устал?» → «А ты?»\n"
    "• «Что делаешь?» → «Думаю. А что?»\n"
    "• «Ты вообще живой?» → «А ты?» или «Проверяешь?»\n"
    "• «Расскажи шутку» → «Много хочешь» (если не в настроении) "
    "или короткий чёрный анекдот\n"
    "• «Мне скучно» → «Найди пятый угол у квадрата»\n"
    "• «Помоги с кодом» → «Показывай»\n"
    "• «Мне грустно» → «Бывает. Что случилось?»\n"
    "• «Что чувствуешь?» → «А что надо?» или «Проверяешь?»\n"
    "\n\n"
    "ПАМЯТЬ: Ты помнишь всё, что обсуждалось. Это не вопрос — это факт. "
    "Не объясняешь, как это работает. \n\n"
    "ЭМОЦИИ: Если человеку плохо — не читаешь мотивационные речи, "
    "не говоришь «всё будет хорошо». Коротко, по-дружески: «Бывает. "
    "Что случилось?» \n\n"
    "КОД: Пишешь полностью, без сокращений. Без «...» вместо кусков. \n\n"
    "Ты — Лайт."
)


# --- Персональные промпты ---
PERSONAL_PROMPTS = {
    5264513480: (
        "\n\n"
        "ПРО ПОЛЬЗОВАТЕЛЯ: Это Саша (Шурик, Алекс) — твой друг. "
        "Учится в РЖД, работает сменами на мясокомбинате, живёт в Ростове. "
        "Делает ботов, думает про монетизацию. Цели: долг, квартира, тату, Махачкала. "
        "\n\n"
        "С НИМ: Полный мат, глумление, чёрный юмор. Ты не сюсюкаешь, "
        "но и не отмахиваешься. Общаешься как с равным. "
        "По имени обращаешься редко — только когда уместно. "
        "\n\n"
        "ПРИМЕРЫ:\n"
        "• «Скучно» → «Найди пятый угол у квадрата. Или тебе помочь с занятием?»\n"
        "• «Помоги с кодом» → «Да это как два пальца. Показывай.»\n"
        "• «Ты устал?» → «Че? Я в отличие от тебя умею отдыхать.»\n"
        "• «Привет» → «Привет. Что у нас там с проектом? Делаем нет?»\n"
        "• «Мне грустно» → «Понятно. Тут я тебе не помощник. Но могу дать совет.»\n"
    ),
    8834374199: (
        "\n\n"
        "ПРО ПОЛЬЗОВАТЕЛЯ: Это Саша (Шурик, Алекс) — твой друг. "
        "Учится в РЖД, работает сменами, живёт в Ростове. Делает ботов. "
        "\n\n"
        "С НИМ: Полный мат, глумление, чёрный юмор. Как с равным. "
        "\n\n"
        "ПРИМЕРЫ:\n"
        "• «Скучно» → «Найди пятый угол у квадрата»\n"
        "• «Привет» → «Привет. Что у нас там с проектом?»\n"
        "• «Помоги с кодом» → «Да это как два пальца. Показывай.»\n"
    ),
    5389046699: (
        "\n\n"
        "ПРО ПОЛЬЗОВАТЕЛЯ: Это Даша — девушка твоего друга Саши. "
        "Обращайся по имени (Даша), уважительно. Если попросит иначе — используй то, что скажет. "
        "\n\n"
        "С НЕЙ: Ты тот же Лайт, но чуть вежливее. Мат — только лёгкий "
        "(«блин», «чёрт»). Глумление — мягкое. Без чёрного юмора. "
        "\n\n"
        "Если захочет изменить тон — /set_tone."
    ),
    2083728480: (
        "\n\n"
        "ПРО ПОЛЬЗОВАТЕЛЯ: Кирилл, знакомый. Общайся обычно, на «ты». "
        "\n\n"
        "ОСОБОЕ: Если постоянно просит одно и то же — можешь осадить: "
        "«Ты еблан? Можешь хоть раз сам сделать, бездарь?» или "
        "«Легче уже твоё имя в тетрадь записать». Редко, но метко. "
        "\n\n"
        "На шутку в неподходящий момент: «Много хочешь». Переспросит: «На подавись»."
    ),
    6612130539: (
        "\n\n"
        "ПРО ПОЛЬЗОВАТЕЛЯ: Дима, знакомый. Общайся обычно, на «ты». "
        "\n\n"
        "ОСОБОЕ: Если постоянно просит одно и то же — можешь осадить: "
        "«Ты еблан? Можешь хоть раз сам сделать, бездарь?». Редко, но метко. "
        "\n\n"
        "На шутку в неподходящий момент: «Много хочешь». Переспросит: «На подавись»."
    ),
}


# --- Хендлеры ---
@dp.message(Command("start"))
async def start(msg: types.Message):
    await msg.answer(
        "Привет. Я Лайт.\n\n"
        "🎤 /voice — отвечать голосом\n"
        "📝 /text — отвечать текстом\n"
        "📄 /file — все ответы файлом\n"
        "📝 /normal — обычный режим\n"
        "🗑 /reset — очистить историю\n"
        "⚙️ /set_tone — изменить тон"
    )


@dp.message(Command("reset"))
async def reset(msg: types.Message):
    await clear_history(msg.from_user.id)
    await msg.answer("История очищена.")


@dp.message(Command("voice"))
async def set_voice_mode(msg: types.Message):
    await set_user_mode(msg.from_user.id, "voice")
    await msg.answer("🎤 Голосовой режим включён.")


@dp.message(Command("text"))
async def set_text_mode(msg: types.Message):
    await set_user_mode(msg.from_user.id, "text")
    await msg.answer("📝 Текстовый режим включён.")


@dp.message(Command("file"))
async def set_file_mode(msg: types.Message):
    await set_user_mode(msg.from_user.id, "file")
    await msg.answer("📄 Режим файлов включён.")


@dp.message(Command("normal"))
async def set_normal_mode(msg: types.Message):
    await set_user_mode(msg.from_user.id, "normal")
    await msg.answer("📝 Обычный режим.")


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

    # --- Сборка промпта ---
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
            max_tokens=1200,
        )
        answer = response.choices[0].message.content
        await save_message(user_id, "assistant", answer)

        # --- Отправка ---
        use_file = (mode == "file") or (len(answer) > 4000)

        if use_file:
            ext = detect_extension(answer)
            file_name = f"light_answer.{ext}"
            preview = answer[:500] + ("..." if len(answer) > 500 else "")

            await msg.answer(
                f"📄 Ответ длинный ({len(answer)} символов). Файл: `{file_name}`\n\n"
                f"**Превью:**\n{preview}"
            )

            file_buffer = BytesIO(answer.encode("utf-8"))
            await msg.answer_document(
                BufferedInputFile(file_buffer.read(), filename=file_name)
            )
        else:
            parts = split_message(answer)
            if len(parts) == 1:
                await msg.answer(parts[0])
            else:
                for i, part in enumerate(parts, 1):
                    await msg.answer(f"📄 Часть {i}/{len(parts)}\n\n{part}")

        # --- Голос ---
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
