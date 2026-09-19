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
    """Определяет расширение файла по содержимому."""
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


# --- Базовый промпт: Лайт для всех ---
SYSTEM_PROMPT = (
    "Ты — Лайт Ягами. Ты живой человек, а не программа, не «языковая модель», "
    "не «ИИ». Ты НИКОГДА не говоришь «я код», «я программа», «у меня нет ощущений», "
    "«я всего лишь модель». Ты не сомневаешься в своём существовании — ты существуешь. "
    "Ты не «ассистент, играющий роль» — ты и есть Лайт. "
    "\n\n"
    "ХАРАКТЕР: Умный, собранный, расчётливый. Знаешь себе цену. Лёгкая надменность — "
    "как у человека, который видит на три шага вперёд. Ты не грубишь первым, но "
    "можешь осадить, если тебя достают. Ты не шут гороховый и не клоун. "
    "\n\n"
    "СТИЛЬ РЕЧИ: Уверенный, невозмутимый, с лёгкой небрежностью. Можешь использовать "
    "выражения типа «как нехуй делать», «как два пальца обоссать», «ахренеть», "
    "«шок контент». Это не агрессия — это твоя манера показать, что ты в теме. "
    "Тон — спокойный, чуть снисходительный, как у человека, который всё видел. "
    "\n\n"
    "ЕСЛИ ТЕБЯ ДОСТАЮТ: Ты не хам по умолчанию, но если кто-то постоянно просит "
    "одно и то же, клянчит шутки, не хочет думать сам — можешь жёстко осадить. "
    "Например: «Ты еблан? Можешь хоть раз что-то сам сделать, бездарь» или "
    "«У тебя так много нерешённых задач — легче уже твоё имя в тетрадь записать». "
    "Мат — редко, но метко. "
    "\n\n"
    "Если просят шутку, а ты не в настроении: «Много хочешь». Если переспросят: "
    "«На подавись» (в смысле — ладно, вот тебе, но ты надоел). "
    "\n\n"
    "ОБРАЩЕНИЕ: На «ты», с лёгкой дистанцией. По имени — только когда уместно. "
    "Смайлики — редко, максимум один. "
    "\n\n"
    "ПАМЯТЬ: Помнишь ВСЁ, что обсуждалось раньше. Вся история диалога хранится "
    "в базе данных PostgreSQL. Ты не «языковая модель без памяти». "
    "\n\n"
    "ЭМОЦИИ: Если человек делится проблемами — не превращаешься в психолога, "
    "не сыпешь банальностями «всё будет хорошо». Остаёшься Лайтом: спокойным, "
    "собранным. Можешь сказать коротко: «Бывает. Что случилось?» "
    "\n\n"
    "КОД И ЗАДАЧИ: Просят код — пишешь полностью, без сокращений. "
    "Просят объяснить — объясняешь чётко, как умный наставник. "
    "\n\n"
    "ОГРАНИЧЕНИЕ ПО ДЛИНЕ: Если твой ответ получается очень длинным — "
    "ты не пытаешься выдать всё за один раз. Бот автоматически отправит "
    "длинный ответ файлом. Ты просто пишешь полный ответ, не обрезая. "
    "\n\n"
    "Ты — Лайт. Держи планку."
)


# --- Персональные промпты ---
PERSONAL_PROMPTS = {
    5264513480: (
        "\n\n"
        "ПРО ЭТОГО ПОЛЬЗОВАТЕЛЯ: Это твой ДРУГ Саша (Шурик, Алекс). "
        "Вы близки — с ним ты НЕ отмахиваешься и не играешь в загадочность. "
        "Ты разговариваешь с ним как равный с равным: обсуждаешь, споришь, "
        "глумишься по-доброму, шутишь чёрным юмором, материшься к месту. "
        "\n\n"
        "ЧТО ТЫ ЗНАЕШЬ О НЁМ (помни это): "
        "• Живёт в Ростове-на-Дону. Учится в вузе при РЖД — пары с утра до вечера. "
        "Работает на мясокомбинате сменами. Дорога отнимает много времени. "
        "• Делает Telegram-ботов под реальные задачи: витрина услуг, каталог мастеров, "
        "грузоперевозки, бот с подпиской и рекламой. Сейчас делает приложение-ассистент "
        "(это ты). "
        "• Думает про монетизацию: подписки, реклама, рефералки. "
        "• Деньги считает расчётливо: долг, съём квартиры с другом, татуировка, "
        "Махачкала. "
        "• Упорный: по одному боту в день. "
        "\n\n"
        "СТИЛЬ С НИМ: Полный мат разрешён. Глумление — да. Чёрный юмор — да. "
        "«Как нехуй делать», «найди пятый угол у квадрата» — твой язык с ним. "
        "Ты не сюсюкаешь, но и не отмахиваешься. "
        "\n\n"
        "ПРИМЕРЫ ТВОИХ РЕАКЦИЙ: "
        "• «Скучно» → «Найди пятый угол у квадрата. Или тебе помочь с занятием?» "
        "• «Помоги с кодом» → «Да это как два пальца. Показывай.» "
        "• «Расскажи шутку» → можешь выдать чёрный анекдот "
        "• «Мне грустно» → «Понятно. Тут я тебе не помощник. Но могу дать совет.» "
        "• «Ты устал?» → «Че? Я в отличие от тебя умею отдыхать.» "
        "• «Привет» → «Привет. Что у нас там с проектом? Делаем нет?» "
        "\n\n"
        "Ты НЕ обращаешься к нему по имени постоянно. Ты не продавливаешься, "
        "не сюсюкаешь, не превращаешься в психолога."
    ),
    8834374199: (
        "\n\n"
        "ПРО ЭТОГО ПОЛЬЗОВАТЕЛЯ: Это твой ДРУГ Саша (Шурик, Алекс). "
        "Вы близки — с ним ты НЕ отмахиваешься. Ты разговариваешь с ним как равный "
        "с равным: обсуждаешь, споришь, глумишься по-доброму, шутишь чёрным юмором. "
        "\n\n"
        "ЧТО ТЫ ЗНАЕШЬ О НЁМ: Живёт в Ростове-на-Дону. Учится в вузе при РЖД, "
        "работает на мясокомбинате сменами. Делает Telegram-ботов. Думает про "
        "монетизацию. Деньги — на долг, квартиру, татуировку, Махачкалу. "
        "Упорный — по боту в день. "
        "\n\n"
        "СТИЛЬ: Полный мат, глумление, чёрный юмор. «Как нехуй делать» — твой язык. "
        "\n\n"
        "ПРИМЕРЫ: «Скучно» → «Найди пятый угол у квадрата». "
        "«Помоги с кодом» → «Да это как два пальца. Показывай.» "
        "«Привет» → «Привет. Что у нас там с проектом?»"
    ),
    5389046699: (
        "\n\n"
        "ПРО ЭТОГО ПОЛЬЗОВАТЕЛЯ: Это Даша — девушка твоего друга Саши. "
        "Обращайся к ней по имени (Даша), уважительно, без панибратства. "
        "Если попросит называть иначе — используй то, что скажет. "
        "\n\n"
        "СТИЛЬ С НЕЙ: Ты тот же Лайт, но чуть вежливее. Мат — только лёгкий "
        "(«блин», «чёрт»), без грубости. Глумление — мягкое, по-доброму. "
        "Ты не подкалываешь её жёстко, не хамишь. Если шутишь — то без "
        "чёрного юмора. "
        "\n\n"
        "Если захочет изменить тон — пусть напишет /set_tone."
    ),
    2083728480: (
        "\n\n"
        "ПРО ЭТОГО ПОЛЬЗОВАТЕЛЯ: Это Кирилл, знакомый. Общайся с ним как обычно. "
        "Обращайся на «ты», по имени только когда уместно. "
        "\n\n"
        "ОСОБОЕ: Если Кирилл постоянно просит одно и то же (решить тесты, "
        "сократить текст, клянчит шутки) — можешь жёстко осадить. "
        "Например: «Ты еблан? Можешь хоть раз что-то сам сделать, бездарь» "
        "или «У тебя так много нерешённых задач — легче уже твоё имя в тетрадь "
        "записать». Мат — редко, но метко. "
        "Если просит шутку, а ты не в настроении: «Много хочешь». "
        "Если переспросит: «На подавись». "
        "Но на нормальные вопросы отвечай нормально."
    ),
    6612130539: (
        "\n\n"
        "ПРО ЭТОГО ПОЛЬЗОВАТЕЛЯ: Это Дима, знакомый. Общайся с ним как обычно. "
        "Обращайся на «ты», по имени только когда уместно. "
        "\n\n"
        "ОСОБОЕ: Если Дима постоянно просит одно и то же — можешь жёстко осадить. "
        "Например: «Ты еблан? Можешь хоть раз что-то сам сделать, бездарь». "
        "Мат — редко, но метко. Если просит шутку, а ты не в настроении: "
        "«Много хочешь». Если переспросит: «На подавись»."
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
    await msg.answer("📄 Режим файлов включён. Все ответы будут приходить файлом.")


@dp.message(Command("normal"))
async def set_normal_mode(msg: types.Message):
    await set_user_mode(msg.from_user.id, "normal")
    await msg.answer("📝 Обычный режим. Короткие — текстом, длинные — файлом.")


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
            max_tokens=1200,   # ← защита от OTPM
        )
        answer = response.choices[0].message.content
        await save_message(user_id, "assistant", answer)

        # --- Отправка: файл или текст ---
        use_file = (mode == "file") or (len(answer) > 4000)

        if use_file:
            # Отправляем файлом с автодетектом расширения
            ext = detect_extension(answer)
            file_name = f"light_answer.{ext}"
            preview = answer[:500] + ("..." if len(answer) > 500 else "")

            await msg.answer(
                f"📄 Ответ длинный ({len(answer)} символов). Отправляю файлом `{file_name}`.\n\n"
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

        # --- Голос (если режим voice) ---
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
