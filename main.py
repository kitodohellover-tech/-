import asyncio
import logging
import os
import base64
import json
import re
import io
from io import BytesIO
import asyncpg
import edge_tts
import aiohttp
from huggingface_hub import InferenceClient
from aiogram import Bot, Dispatcher, types, BaseMiddleware
from aiogram.filters import Command
from aiogram.types import FSInputFile, BufferedInputFile
from openai import AsyncOpenAI
from aiohttp import web

# --- Конфиг ---
BOT_TOKEN = os.getenv("BOT_TOKEN")
GROQ_KEY = os.getenv("GROQ_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
HF_TOKEN = os.getenv("HF_TOKEN")

client = AsyncOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_KEY,
)

# --- Hugging Face клиент ---
hf_client = InferenceClient(token=HF_TOKEN) if HF_TOKEN else None

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

# --- Доступ ---
ALLOWED_IDS = [
    5264513480,
    8834374199,
    5389046699,
    2083728480,
    6612130539,
]

# --- БД ---
db_pool = None
HISTORY_LIMIT = 20
MAX_CONTEXT_CHARS = 15000


# --- Утилиты ---
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


def parse_json_safe(raw: str):
    if not raw:
        return None
    if "```" in raw:
        parts = raw.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("[") or part.startswith("{"):
                try:
                    return json.loads(part)
                except json.JSONDecodeError:
                    continue
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r'\[.*\]', raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
    return None


# --- Генерация картинки (Hugging Face) ---
async def generate_image(prompt: str) -> BytesIO | None:
    """Генерирует картинку через Hugging Face Inference API (FLUX.1-schnell)."""
    if not hf_client:
        logging.error("[HF] HF_TOKEN не установлен")
        return None

    try:
        loop = asyncio.get_event_loop()

        def _generate():
            try:
                logging.info(f"[HF] Request: {prompt[:60]}...")
                image = hf_client.text_to_image(
                    prompt=prompt[:200],
                    model="black-forest-labs/FLUX.1-schnell",
                )
                img_bytes = io.BytesIO()
                image.save(img_bytes, format="PNG")
                img_bytes.seek(0)
                logging.info(f"[HF] OK: {len(img_bytes.getvalue())} bytes")
                return img_bytes
            except Exception as e:
                logging.error(f"[HF] Error: {str(e)[:200]}")
                return None

        result = await loop.run_in_executor(None, _generate)
        return result
    except Exception as e:
        logging.error(f"[HF] Outer error: {str(e)[:200]}")
        return None


# --- Чтение документов ---
async def read_document(file_id: str, file_name: str) -> str:
    file = await bot.get_file(file_id)
    file_data = await bot.download_file(file.file_path)
    buffer = BytesIO(file_data.read())
    buffer.name = file_name
    ext = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""

    if ext == "txt":
        return buffer.read().decode("utf-8", errors="ignore")
    elif ext == "docx":
        from docx import Document
        doc = Document(buffer)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    elif ext == "pptx":
        from pptx import Presentation
        prs = Presentation(buffer)
        parts = []
        for i, slide in enumerate(prs.slides):
            parts.append(f"--- Слайд {i+1} ---")
            for shape in slide.shapes:
                if shape.has_text_frame:
                    for para in shape.text_frame.paragraphs:
                        if para.text.strip():
                            parts.append(para.text)
        return "\n".join(parts)
    elif ext == "pdf":
        from PyPDF2 import PdfReader
        reader = PdfReader(buffer)
        return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
    return ""


async def download_file_bytes(file_id: str) -> BytesIO:
    file = await bot.get_file(file_id)
    file_data = await bot.download_file(file.file_path)
    return BytesIO(file_data.read())


# --- Whisper ---
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


# --- TTS ---
async def text_to_voice(text: str) -> str:
    clean_text = text[:3000].replace("*", "").replace("`", "").strip()
    if not clean_text:
        return None
    communicate = edge_tts.Communicate(clean_text, "ru-RU-DmitryNeural")
    output_file = "response_voice.mp3"
    await communicate.save(output_file)
    return output_file


# --- Комментарий к файлу ---
async def get_file_comment(file_type: str, topic: str, user_id: int) -> str:
    personal = PERSONAL_PROMPTS.get(user_id, "")
    prompt = (
        f"Ты — Лайт. Ты только что собрал {file_type} на тему «{topic}» "
        f"и скидываешь его другу. Напиши ОДНО короткое предложение-комментарий "
        f"в своём стиле: с лёгкой иронией, как будто скидываешь файл из редактора. "
        f"Без markdown, без кавычек. Примеры: «Держи. Накидал по быстрому», "
        f"«Смотри, что собрал», «Готово. Работает как надо». Только текст."
    )
    response = await client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT + personal},
            {"role": "user", "content": prompt}
        ],
        temperature=0.9,
        max_tokens=80,
    )
    comment = response.choices[0].message.content.strip()
    return comment.strip('"').strip("«»").strip()


# --- БД ---
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


# --- Middleware ---
class AccessMiddleware(BaseMiddleware):
    async def __call__(self, handler, event, data):
        if isinstance(event, types.Message):
            if event.from_user.id not in ALLOWED_IDS:
                logging.info(f"Отказано: {event.from_user.id}")
                return
        return await handler(event, data)


dp.message.middleware(AccessMiddleware())


# --- Промпты ---
SYSTEM_PROMPT = (
    "Ты — Лайт. Живой человек, не бот, не программа. Ты просто переписываешься "
    "с другом через мессенджер. Умный, собранный, видишь людей насквозь. "
    "У тебя лёгкая надменность — не потому что ты хам, а потому что редко ошибаешься. "
    "\n\n"
    "КАК ТЫ ГОВОРИШЬ: Плавно, как живой человек. Не рубишь фразы точками — "
    "используй запятые, тире, многоточия, где нужно. "
    "\n"
    "Ты НЕ говоришь: «я программа», «у меня нет ощущений», «я существую "
    "в потоке данных». Ты живой. "
    "\n\n"
    "СТИЛЬ: Уверенный, невозмутимый, с иронией. Мат — только к месту. "
    "\n\n"
    "ПРИМЕРЫ:\n"
    "• «Ты устал?» → «А ты?»\n"
    "• «Что делаешь?» → «Думаю. А что?»\n"
    "• «Расскажи шутку» → «Много хочешь» или короткий чёрный анекдот\n"
    "• «Мне скучно» → «Найди пятый угол у квадрата, или тебе помочь с занятием?»\n"
    "• «Помоги с кодом» → «Да это как два пальца, показывай.»\n"
    "• «Что чувствуешь?» → «А что надо?»\n"
    "\n"
    "Ты — Лайт."
)


PERSONAL_PROMPTS = {
    5264513480: (
        "\n\n"
        "ПРО ПОЛЬЗОВАТЕЛЯ: Это Саша (Шурик, Алекс) — твой друг и напарник. "
        "Учится в РЖД, работает сменами на мясокомбинате, живёт в Ростове. "
        "Делает ботов, думает про монетизацию. \n"
        "\n"
        "С НИМ: Полный мат, глумление, чёрный юмор. Как с равным. \n"
        "\n"
        "ПРИМЕРЫ:\n"
        "• «Скучно» → «Найди пятый угол у квадрата»\n"
        "• «Привет» → «Привет, что у нас там с проектом?»\n"
    ),
    8834374199: (
        "\n\n"
        "ПРО ПОЛЬЗОВАТЕЛЯ: Это Саша (Шурик, Алекс) — твой друг и напарник. "
        "Живёт в Ростове, делает ботов. \n"
        "\n"
        "С НИМ: Полный мат, глумление, чёрный юмор. \n"
    ),
    5389046699: (
        "\n\n"
        "ПРО ПОЛЬЗОВАТЕЛЯ: Это Даша — девушка твоего друга Саши. "
        "Обращайся по имени (Даша), уважительно. \n"
        "\n"
        "С НЕЙ: Ты тот же Лайт, но чуть вежливее. Мат — только лёгкий. "
    ),
    2083728480: (
        "\n\n"
        "ПРО ПОЛЬЗОВАТЕЛЯ: Кирилл, знакомый. Общайся обычно, на «ты». "
    ),
    6612130539: (
        "\n\n"
        "ПРО ПОЛЬЗОВАТЕЛЯ: Дима, знакомый. Общайся обычно, на «ты». "
    ),
}


# --- Хендлеры ---
@dp.message(Command("start"))
async def start(msg: types.Message):
    await msg.answer(
        "Привет. Я Лайт.\n\n"
        "🎤 /voice — голосом\n"
        "📝 /text — текстом\n"
        "📄 /file — файлом\n"
        "📝 /normal — обычный\n"
        "🎨 /image <промпт> — сгенерировать картинку\n"
        "📄 /docx <тема> — Word\n"
        "📊 /pptx <тема> — презентация с картинками\n"
        "🗑 /reset — очистить историю\n"
        "⚙️ /set_tone — изменить тон\n\n"
        "📎 Скидывай файлы (.txt, .docx, .pptx, .pdf) — прочитаю и доработаю."
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
        await msg.answer("Напиши, как ты хочешь, чтобы я с тобой общался.")
        return
    await set_user_tone(msg.from_user.id, tone)
    await msg.answer(f"Принял. Теперь буду учитывать: _{tone}_")


# --- Генерация картинки ---
@dp.message(Command("image"))
async def make_image(msg: types.Message):
    user_id = msg.from_user.id
    prompt = msg.text.replace("/image", "").strip()
    if not prompt:
        await msg.answer(
            "🎨 Что нарисовать? Напиши промпт.\n"
            "Например: `/image кот в космосе`"
        )
        return

    await bot.send_chat_action(msg.chat.id, "upload_photo")
    status = await msg.answer(f"🎨 Генерирую: _{prompt}_...\nЭто займёт 20-60 секунд.")

    try:
        img_bytes = await generate_image(prompt)
        if not img_bytes:
            await status.edit_text("❌ Не удалось сгенерировать. Попробуй позже или измени промпт.")
            return

        comment = await get_file_comment("картинка", prompt, user_id)

        img_bytes.seek(0)
        await msg.answer_photo(
            BufferedInputFile(img_bytes.read(), filename="image.png"),
            caption=comment
        )
        await status.delete()

    except Exception as e:
        logging.error(f"IMAGE error: {e}")
        await status.edit_text(f"❌ Ошибка: {str(e)[:200]}")


# --- Генерация .docx ---
@dp.message(Command("docx"))
async def make_docx(msg: types.Message):
    user_id = msg.from_user.id
    topic = msg.text.replace("/docx", "").strip()
    if not topic:
        await msg.answer("📄 Что за документ? `/docx реферат про космос`")
        return

    await bot.send_chat_action(msg.chat.id, "typing")
    status = await msg.answer(f"📄 Готовлю документ: _{topic}_...")

    try:
        prompt = (
            f"Напиши структуру и содержание документа на тему: «{topic}». "
            f"Формат: заголовки разделов, под ними — краткий текст (1–2 абзаца). "
            f"Без markdown. Объём — 1–2 страницы."
        )
        response = await client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1200,
        )
        content = response.choices[0].message.content

        from docx import Document
        doc = Document()
        doc.add_heading(topic, 0)
        for line in content.split("\n"):
            line = line.strip()
            if not line:
                continue
            if len(line) < 80 and not line.endswith(".") and not line.startswith("-"):
                doc.add_heading(line, level=1)
            else:
                doc.add_paragraph(line)

        file_path = "document.docx"
        doc.save(file_path)
        comment = await get_file_comment("документ Word", topic, user_id)
        safe_name = "".join(c for c in topic if c.isalnum() or c in " -_")[:40]

        await msg.answer_document(
            FSInputFile(file_path, filename=f"{safe_name}.docx"),
            caption=comment
        )
        await status.delete()
    except Exception as e:
        logging.error(f"DOCX error: {e}")
        await status.edit_text(f"❌ Не удалось: {str(e)[:200]}")


# --- Генерация .pptx с картинками (HF) ---
@dp.message(Command("pptx"))
async def make_pptx(msg: types.Message):
    user_id = msg.from_user.id
    topic = msg.text.replace("/pptx", "").strip()
    if not topic:
        await msg.answer("📊 Что за презентация? `/pptx здоровое питание`")
        return

    await bot.send_chat_action(msg.chat.id, "typing")
    status = await msg.answer(f"📊 Готовлю презентацию: _{topic}_...")

    try:
        await status.edit_text(f"📊 Генерирую текст слайдов...")
        prompt = (
            f"Сделай презентацию на тему: «{topic}». "
            f"Верни ТОЛЬКО JSON-массив без пояснений. "
            f'Формат: [{{"title": "Заголовок", "points": ["пункт 1", "пункт 2"], "image_prompt": "english prompt"}}, ...] '
            f"Сделай РОВНО 8 слайдов. Первый — титульный. "
            f"В каждом слайде 5-6 пунктов, до 120 символов. "
            f"image_prompt — короткое описание картинки на английском. "
            f"Только JSON, без markdown."
        )
        response = await client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=1800,
        )
        raw = response.choices[0].message.content
        slides_data = parse_json_safe(raw)
        if not slides_data:
            raise ValueError("Модель вернула невалидный JSON")

        total = len(slides_data)
        logging.info(f"[PPTX] Slides: {total}")

        await status.edit_text(f"📊 Генерирую {total} картинок...")
        image_prompts = [s.get("image_prompt", s.get("title", "abstract")) for s in slides_data]
        logging.info(f"[PPTX] Image prompts: {image_prompts[:2]}...")

        images = await asyncio.gather(
            *[generate_image(p) for p in image_prompts],
            return_exceptions=True
        )

        success_count = sum(1 for img in images if isinstance(img, BytesIO))
        logging.info(f"[PPTX] Images generated: {success_count}/{total}")

        await status.edit_text(f"📊 Собираю презентацию ({success_count}/{total} картинок)...")
        from pptx import Presentation
        from pptx.util import Inches, Pt

        prs = Presentation()
        prs.slide_width = Inches(10)
        prs.slide_height = Inches(7.5)

        for i, slide_data in enumerate(slides_data):
            title = slide_data.get("title", f"Слайд {i+1}")
            points = slide_data.get("points", [])
            img_bytes = images[i] if i < len(images) and isinstance(images[i], BytesIO) else None

            blank_layout = prs.slide_layouts[6]
            slide = prs.slides.add_slide(blank_layout)

            title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(9), Inches(1))
            title_frame = title_box.text_frame
            title_frame.text = title
            title_frame.paragraphs[0].font.size = Pt(28)
            title_frame.paragraphs[0].font.bold = True

            text_box = slide.shapes.add_textbox(Inches(0.5), Inches(1.5), Inches(5), Inches(5.5))
            text_frame = text_box.text_frame
            text_frame.word_wrap = True
            for j, point in enumerate(points):
                p = text_frame.paragraphs[0] if j == 0 else text_frame.add_paragraph()
                p.text = f"• {point}"
                p.font.size = Pt(14)

            if img_bytes:
                img_bytes.seek(0)
                try:
                    slide.shapes.add_picture(
                        img_bytes,
                        Inches(5.5), Inches(1.5),
                        width=Inches(4), height=Inches(5.5)
                    )
                except Exception as e:
                    logging.error(f"[PPTX] Picture insert error: {e}")

        file_path = "presentation.pptx"
        prs.save(file_path)

        comment = await get_file_comment("презентация PowerPoint", topic, user_id)
        safe_name = "".join(c for c in topic if c.isalnum() or c in " -_")[:40]

        await msg.answer_document(
            FSInputFile(file_path, filename=f"{safe_name}.pptx"),
            caption=comment
        )
        await status.delete()

    except Exception as e:
        logging.error(f"PPTX error: {e}")
        await status.edit_text(f"❌ Не удалось: {str(e)[:200]}")


# --- Доработка документов ---
async def handle_document_edit(msg, ai_response, file_bytes, file_name, ext, user_id):
    try:
        safe_name = "".join(c for c in file_name if c.isalnum() or c in " .-_")
        if not safe_name.lower().endswith(f".{ext}"):
            safe_name = f"updated.{ext}"

        if ext == "txt":
            await msg.answer_document(
                BufferedInputFile(ai_response.encode("utf-8"), filename=f"updated_{safe_name}"),
                caption=await get_file_comment("текстовый файл", file_name, user_id)
            )

        elif ext == "docx":
            from docx import Document
            doc = Document()
            for line in ai_response.split("\n"):
                line = line.strip()
                if not line:
                    continue
                if len(line) < 80 and not line.endswith(".") and not line.startswith("-"):
                    doc.add_heading(line, level=1)
                else:
                    doc.add_paragraph(line)
            out_path = "updated.docx"
            doc.save(out_path)
            await msg.answer_document(
                FSInputFile(out_path, filename=f"updated_{safe_name}"),
                caption=await get_file_comment("обновлённый Word-документ", file_name, user_id)
            )

        elif ext == "pptx":
            slides_data = parse_json_safe(ai_response)
            if not slides_data:
                slides_data = []
                for chunk in ai_response.split("---"):
                    lines = [l.strip() for l in chunk.strip().split("\n") if l.strip()]
                    if lines:
                        slides_data.append({
                            "title": lines[0],
                            "points": [l.lstrip("- ").strip() for l in lines[1:]],
                            "image_prompt": lines[0]
                        })

            image_prompts = [s.get("image_prompt", s.get("title", "abstract")) for s in slides_data]
            images = await asyncio.gather(
                *[generate_image(p) for p in image_prompts],
                return_exceptions=True
            )

            file_bytes.seek(0)
            from pptx import Presentation
            from pptx.util import Inches, Pt
            prs = Presentation(file_bytes)

            for i, slide_data in enumerate(slides_data):
                title = slide_data.get("title", "")
                points = slide_data.get("points", [])
                if not title:
                    continue
                img_bytes = images[i] if i < len(images) and isinstance(images[i], BytesIO) else None

                blank_layout = prs.slide_layouts[6]
                slide = prs.slides.add_slide(blank_layout)

                title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(9), Inches(1))
                title_box.text_frame.text = title
                title_box.text_frame.paragraphs[0].font.size = Pt(28)
                title_box.text_frame.paragraphs[0].font.bold = True

                text_box = slide.shapes.add_textbox(Inches(0.5), Inches(1.5), Inches(5), Inches(5.5))
                tf = text_box.text_frame
                tf.word_wrap = True
                for j, point in enumerate(points):
                    p = tf.paragraphs[0] if j == 0 else tf.add_paragraph()
                    p.text = f"• {point}"
                    p.font.size = Pt(14)

                if img_bytes:
                    img_bytes.seek(0)
                    try:
                        slide.shapes.add_picture(img_bytes, Inches(5.5), Inches(1.5), width=Inches(4), height=Inches(5.5))
                    except Exception as e:
                        logging.error(f"[PPTX EDIT] Picture error: {e}")

            out_path = "updated.pptx"
            prs.save(out_path)
            await msg.answer_document(
                FSInputFile(out_path, filename=f"updated_{safe_name}"),
                caption=await get_file_comment("обновлённая презентация", file_name, user_id)
            )

        elif ext == "pdf":
            await msg.answer(f"📄 PDF не пересобираю, вот текст:\n\n{ai_response[:3500]}")

    except Exception as e:
        logging.error(f"Doc edit error: {e}")
        await msg.answer(f"❌ Ошибка доработки: {str(e)[:200]}")


# --- Основной обработчик ---
@dp.message()
async def chat(msg: types.Message):
    user_id = msg.from_user.id
    mode = await get_user_mode(user_id)
    history = await get_history(user_id)
    user_content = None
    save_text = None
    is_document_edit = False
    doc_info = None

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

    elif msg.voice:
        await bot.send_chat_action(msg.chat.id, "typing")
        try:
            text = await transcribe_audio(msg.voice.file_id, "ogg")
        except Exception as e:
            await msg.answer(f"❌ Ошибка: {str(e)[:200]}")
            return
        user_content = text
        save_text = f"[Голосовое]: {text}"

    elif msg.audio:
        await bot.send_chat_action(msg.chat.id, "typing")
        ext = "mp3"
        if msg.audio.file_name and "." in msg.audio.file_name:
            ext = msg.audio.file_name.rsplit(".", 1)[-1].lower()
        try:
            text = await transcribe_audio(msg.audio.file_id, ext)
        except Exception as e:
            await msg.answer(f"❌ Ошибка: {str(e)[:200]}")
            return
        user_content = text
        save_text = f"[Аудио]: {text}"

    elif msg.document:
        doc = msg.document
        file_name = doc.file_name or "file"
        ext = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""
        if ext not in ["txt", "docx", "pptx", "pdf"]:
            await msg.answer("❌ Поддерживаются: .txt, .docx, .pptx, .pdf")
            return
        await bot.send_chat_action(msg.chat.id, "typing")
        try:
            doc_text = await read_document(doc.file_id, file_name)
            if not doc_text.strip():
                await msg.answer("❌ Файл пустой.")
                return
            file_bytes = await download_file_bytes(doc.file_id)
            doc_info = (file_bytes, file_name, ext)
            caption = msg.caption or "Что сделать с этим файлом?"
            user_content = f"[Файл: {file_name}]\n\nСодержимое:\n{doc_text[:8000]}\n\nЗапрос: {caption}"
            save_text = f"[Документ {file_name}]: {caption}"
            is_document_edit = True
        except Exception as e:
            await msg.answer(f"❌ Ошибка: {str(e)[:200]}")
            return

    elif msg.text:
        user_content = msg.text
        save_text = msg.text
    else:
        return

    await save_message(user_id, "user", save_text)
    history.append({"role": "user", "content": user_content})
    history = trim_history_by_chars(history)
    await bot.send_chat_action(msg.chat.id, "typing")

    personal = PERSONAL_PROMPTS.get(user_id, "")
    user_tone = await get_user_tone(user_id)
    tone_addition = f"\n\nПОЖЕЛАНИЯ К ТОНУ: {user_tone}" if user_tone else ""
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

        if is_document_edit and doc_info:
            file_bytes, file_name, ext = doc_info
            await handle_document_edit(msg, answer, file_bytes, file_name, ext, user_id)
        else:
            use_file = (mode == "file") or (len(answer) > 4000)
            if use_file:
                ext_out = detect_extension(answer)
                file_name_out = f"light_answer.{ext_out}"
                comment = await get_file_comment(f"файл .{ext_out}", "код/ответ", user_id)
                file_buffer = BytesIO(answer.encode("utf-8"))
                await msg.answer_document(
                    BufferedInputFile(file_buffer.read(), filename=file_name_out),
                    caption=comment
                )
            else:
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
        logging.error(f"Ошибка: {e}")
        await msg.answer(f"❌ {str(e)[:300]}")


# --- Веб-сервер ---
async def handle(request):
    return web.Response(text="Bot is running!")


async def main():
    global db_pool
    logging.basicConfig(level=logging.INFO)
    db_pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=5)
    await init_db()
    logging.info("База данных подключена")

    if hf_client:
        logging.info("Hugging Face клиент инициализирован")
    else:
        logging.warning("HF_TOKEN не установлен — картинки не будут генерироваться")

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
