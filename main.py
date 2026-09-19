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
import requests
from huggingface_hub import InferenceClient
from pexafy import Client as PexafyClient
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
PEXAFY_KEY = os.getenv("PEXAFY_API_KEY")

client = AsyncOpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_KEY)
hf_client = InferenceClient(token=HF_TOKEN) if HF_TOKEN else None
pexafy_client = PexafyClient(PEXAFY_KEY) if PEXAFY_KEY else None

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

ALLOWED_IDS = [5264513480, 8834374199, 5389046699, 2083728480, 6612130539]

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
            parts.append(text); break
        chunk = text[:limit]
        split_pos = chunk.rfind("\n")
        if split_pos == -1: split_pos = chunk.rfind(" ")
        if split_pos == -1: split_pos = limit
        parts.append(text[:split_pos])
        text = text[split_pos:].lstrip("\n")
    return parts

def trim_history_by_chars(history: list[dict], max_chars: int = MAX_CONTEXT_CHARS) -> list[dict]:
    total = 0; trimmed = []
    for msg in reversed(history):
        content = msg["content"]
        length = len(content) if isinstance(content, str) else sum(
            len(p.get("text", "")) + len(p.get("image_url", {}).get("url", "")) for p in content)
        if total + length > max_chars: break
        trimmed.append(msg); total += length
    return list(reversed(trimmed))

def detect_extension(text: str) -> str:
    if "<!DOCTYPE html>" in text or "<html" in text.lower(): return "html"
    if "def " in text and "import " in text: return "py"
    if "function " in text or "const " in text or "let " in text: return "js"
    return "txt"

def parse_json_safe(raw: str):
    if not raw: return None
    if "```" in raw:
        for part in raw.split("```"):
            part = part.strip()
            if part.startswith("json"): part = part[4:].strip()
            if part.startswith("[") or part.startswith("{"):
                try: return json.loads(part)
                except: continue
    try: return json.loads(raw.strip())
    except:
        match = re.search(r'\[.*\]', raw, re.DOTALL)
        if match:
            try: return json.loads(match.group(0))
            except: return None
    return None


# --- Поиск фото через Pexafy ---
async def search_stock_photo(query: str) -> BytesIO | None:
    if not pexafy_client:
        logging.error("[Pexafy] PEXAFY_API_KEY не установлен")
        return None
    try:
        loop = asyncio.get_event_loop()
        def _search():
            try:
                photos = list(pexafy_client.search(query, per_page=3))
                if not photos: return None
                url = photos[0].urls.regular
                resp = requests.get(url, timeout=30)
                if resp.status_code == 200 and len(resp.content) > 1000:
                    logging.info(f"[Pexafy] '{query[:40]}' -> {len(resp.content)} bytes")
                    return BytesIO(resp.content)
                return None
            except Exception as e:
                logging.error(f"[Pexafy] Error: {str(e)[:200]}")
                return None
        return await loop.run_in_executor(None, _search)
    except Exception as e:
        logging.error(f"[Pexafy] Outer: {str(e)[:200]}")
        return None


# --- Генерация (fallback) ---
async def generate_image(prompt: str) -> BytesIO | None:
    if not hf_client: return None
    try:
        loop = asyncio.get_event_loop()
        def _gen():
            try:
                image = hf_client.text_to_image(prompt=prompt[:200], model="black-forest-labs/FLUX.1-schnell")
                buf = io.BytesIO(); image.save(buf, format="PNG"); buf.seek(0)
                return buf
            except Exception as e:
                logging.error(f"[HF] Error: {str(e)[:150]}"); return None
        return await loop.run_in_executor(None, _gen)
    except: return None


# --- Автоперевод ---
async def translate_to_english(text: str) -> str:
    try:
        resp = await client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=[{"role": "system", "content": "Переведи на английский для поиска фото. Только перевод."},
                      {"role": "user", "content": text}],
            temperature=0.3, max_tokens=150)
        return resp.choices[0].message.content.strip().strip('"')
    except: return text


# --- Чтение документов ---
async def read_document(file_id: str, file_name: str) -> str:
    file = await bot.get_file(file_id); data = await bot.download_file(file.file_path)
    buf = BytesIO(data.read()); buf.name = file_name
    ext = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""
    if ext == "txt": return buf.read().decode("utf-8", errors="ignore")
    elif ext == "docx":
        from docx import Document
        return "\n".join(p.text for p in Document(buf).paragraphs if p.text.strip())
    elif ext == "pptx":
        from pptx import Presentation
        parts = []
        for i, slide in enumerate(Presentation(buf).slides):
            parts.append(f"--- Слайд {i+1} ---")
            for shape in slide.shapes:
                if shape.has_text_frame:
                    for para in shape.text_frame.paragraphs:
                        if para.text.strip(): parts.append(para.text)
        return "\n".join(parts)
    elif ext == "pdf":
        from PyPDF2 import PdfReader
        return "\n".join(page.extract_text() for page in PdfReader(buf).pages if page.extract_text())
    return ""

async def download_file_bytes(file_id: str) -> BytesIO:
    file = await bot.get_file(file_id); data = await bot.download_file(file.file_path)
    return BytesIO(data.read())

async def transcribe_audio(file_id: str, ext: str = "ogg") -> str:
    file = await bot.get_file(file_id); data = await bot.download_file(file.file_path)
    buf = BytesIO(data.read()); buf.name = f"audio.{ext}"
    t = await client.audio.transcriptions.create(model="whisper-large-v3", file=buf, language="ru")
    return t.text

async def text_to_voice(text: str) -> str:
    clean = text[:3000].replace("*", "").replace("`", "").strip()
    if not clean: return None
    comm = edge_tts.Communicate(clean, "ru-RU-DmitryNeural")
    out = "response_voice.mp3"; await comm.save(out); return out

async def get_file_comment(file_type: str, topic: str, user_id: int) -> str:
    personal = PERSONAL_PROMPTS.get(user_id, "")
    prompt = (f"Ты — Лайт. Ты только что сгенерировал {file_type} на тему «{topic}» и скидываешь другу. "
              f"Напиши ОДНО короткое предложение с лёгкой иронией. Не используй «собрал». Без кавычек.")
    resp = await client.chat.completions.create(model="qwen/qwen3.8-27b",
        messages=[{"role": "system", "content": SYSTEM_PROMPT + personal}, {"role": "user", "content": prompt}],
        temperature=0.9, max_tokens=80)
    return resp.choices[0].message.content.strip().strip('"').strip("«»")


# --- БД ---
async def init_db():
    async with db_pool.acquire() as conn:
        await conn.execute("""CREATE TABLE IF NOT EXISTS messages (id SERIAL PRIMARY KEY, user_id BIGINT, role TEXT, content TEXT, created_at TIMESTAMP DEFAULT NOW())""")
        await conn.execute("""CREATE TABLE IF NOT EXISTS user_settings (user_id BIGINT PRIMARY KEY, mode TEXT DEFAULT 'normal')""")
        await conn.execute("""CREATE TABLE IF NOT EXISTS user_tone (user_id BIGINT PRIMARY KEY, tone TEXT DEFAULT '')""")

async def save_message(uid: int, role: str, content: str):
    async with db_pool.acquire() as conn:
        await conn.execute("INSERT INTO messages (user_id, role, content) VALUES ($1, $2, $3)", uid, role, content)

async def get_history(uid: int, limit: int = HISTORY_LIMIT):
    async with db_pool.acquire() as conn:
        rows = await conn.fetch("SELECT role, content FROM messages WHERE user_id = $1 ORDER BY id DESC LIMIT $2", uid, limit)
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

async def clear_history(uid: int):
    async with db_pool.acquire() as conn:
        await conn.execute("DELETE FROM messages WHERE user_id = $1", uid)

async def get_user_mode(uid: int) -> str:
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow("SELECT mode FROM user_settings WHERE user_id = $1", uid)
        return row['mode'] if row else 'normal'

async def set_user_mode(uid: int, mode: str):
    async with db_pool.acquire() as conn:
        await conn.execute("""INSERT INTO user_settings (user_id, mode) VALUES ($1, $2) ON CONFLICT (user_id) DO UPDATE SET mode = $2""", uid, mode)

async def get_user_tone(uid: int) -> str:
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow("SELECT tone FROM user_tone WHERE user_id = $1", uid)
        return row['tone'] if row else ''

async def set_user_tone(uid: int, tone: str):
    async with db_pool.acquire() as conn:
        await conn.execute("""INSERT INTO user_tone (user_id, tone) VALUES ($1, $2) ON CONFLICT (user_id) DO UPDATE SET tone = $2""", uid, tone)


# --- Middleware ---
class AccessMiddleware(BaseMiddleware):
    async def __call__(self, handler, event, data):
        if isinstance(event, types.Message):
            if event.from_user.id not in ALLOWED_IDS: return
        return await handler(event, data)

dp.message.middleware(AccessMiddleware())

# --- Промпты ---
SYSTEM_PROMPT = ("Ты — Лайт. Живой человек, не бот. Умный, собранный, видишь людей насквозь. "
    "Лёгкая надменность. Говоришь плавно, не рубишь фразы точками. "
    "Не говоришь «я программа», «я существую в потоке данных». Ты живой. "
    "Мат — только к месту. Ты — Лайт.")

PERSONAL_PROMPTS = {
    5264513480: "\n\nПРО ПОЛЬЗОВАТЕЛЯ: Саша (Шурик) — твой друг. Ростов. Делает ботов. Полный мат, глумление, чёрный юмор.",
    8834374199: "\n\nПРО ПОЛЬЗОВАТЕЛЯ: Саша (Шурик) — твой друг. Ростов. Полный мат, глумление.",
    5389046699: "\n\nПРО ПОЛЬЗОВАТЕЛЯ: Даша — девушка друга. Уважительно, лёгкий мат.",
    2083728480: "\n\nПРО ПОЛЬЗОВАТЕЛЯ: Кирилл, знакомый. Общайся обычно.",
    6612130539: "\n\nПРО ПОЛЬЗОВАТЕЛЯ: Дима, знакомый. Общайся обычно.",
}


# --- Хендлеры ---
@dp.message(Command("start"))
async def start(msg: types.Message):
    await msg.answer("Привет. Я Лайт.\n\n🎤 /voice — голосом\n📝 /text — текстом\n📄 /file — файлом\n"
        "🎨 /image <промпт> — картинка\n🎬 /speech — речь для защиты\n📄 /docx <тема> — Word\n"
        "📊 /pptx <тема> — презентация с фото\n🗑 /reset — очистить\n⚙️ /set_tone — тон\n\n"
        "📎 Скидывай файлы (.txt, .docx, .pptx, .pdf) — прочитаю и доработаю.")

@dp.message(Command("reset"))
async def reset(msg: types.Message):
    await clear_history(msg.from_user.id); await msg.answer("История очищена.")

@dp.message(Command("voice"))
async def set_voice(msg: types.Message):
    await set_user_mode(msg.from_user.id, "voice"); await msg.answer("🎤 Голосовой режим.")

@dp.message(Command("text"))
async def set_text(msg: types.Message):
    await set_user_mode(msg.from_user.id, "text"); await msg.answer("📝 Текстовый режим.")

@dp.message(Command("file"))
async def set_file(msg: types.Message):
    await set_user_mode(msg.from_user.id, "file"); await msg.answer("📄 Режим файлов.")

@dp.message(Command("normal"))
async def set_normal(msg: types.Message):
    await set_user_mode(msg.from_user.id, "normal"); await msg.answer("📝 Обычный режим.")

@dp.message(Command("set_tone"))
async def set_tone_cmd(msg: types.Message):
    if msg.from_user.id not in [5264513480, 8834374199, 5389046699]:
        await msg.answer("Недоступно."); return
    tone = msg.text.replace("/set_tone", "").strip()
    if not tone: await msg.answer("Напиши, как хочешь общаться."); return
    await set_user_tone(msg.from_user.id, tone); await msg.answer(f"Принял: _{tone}_")


# --- Генерация картинки ---
@dp.message(Command("image"))
async def make_image(msg: types.Message):
    uid = msg.from_user.id; prompt = msg.text.replace("/image", "").strip()
    if not prompt: await msg.answer("🎨 Что нарисовать? `/image кот`"); return
    await bot.send_chat_action(msg.chat.id, "upload_photo")
    status = await msg.answer(f"🎨 Ищу фото: _{prompt}_...")
    try:
        prompt_en = await translate_to_english(prompt)
        img = await search_stock_photo(prompt_en); source = "Pexafy"
        if not img:
            await status.edit_text("🔍 Не нашёл — генерирую...")
            img = await generate_image(prompt_en); source = "HF"
        if not img:
            await status.edit_text("❌ Не удалось."); return
        comment = await get_file_comment("картинка", prompt, uid)
        img.seek(0)
        await msg.answer_photo(BufferedInputFile(img.read(), filename="image.png"), caption=f"{comment}\n\n_(via {source})_")
        await status.delete()
    except Exception as e:
        logging.error(f"IMAGE error: {e}"); await status.edit_text(f"❌ {str(e)[:200]}")


# --- Генерация .docx ---
@dp.message(Command("docx"))
async def make_docx(msg: types.Message):
    uid = msg.from_user.id; topic = msg.text.replace("/docx", "").strip()
    if not topic: await msg.answer("📄 `/docx реферат`"); return
    await bot.send_chat_action(msg.chat.id, "typing")
    status = await msg.answer(f"📄 Готовлю: _{topic}_...")
    try:
        prompt = f"Напиши структуру документа на тему «{topic}». Заголовки разделов, краткий текст. Без markdown."
        resp = await client.chat.completions.create(model="qwen/qwen3.8-27b", messages=[{"role": "user", "content": prompt}], temperature=0.7, max_tokens=1200)
        content = resp.choices[0].message.content
        from docx import Document
        doc = Document(); doc.add_heading(topic, 0)
        for line in content.split("\n"):
            line = line.strip()
            if not line: continue
            if len(line) < 80 and not line.endswith(".") and not line.startswith("-"): doc.add_heading(line, level=1)
            else: doc.add_paragraph(line)
        path = "document.docx"; doc.save(path)
        comment = await get_file_comment("документ", topic, uid)
        safe = "".join(c for c in topic if c.isalnum() or c in " -_")[:40]
        await msg.answer_document(FSInputFile(path, filename=f"{safe}.docx"), caption=comment)
        await status.delete()
    except Exception as e:
        logging.error(f"DOCX error: {e}"); await status.edit_text(f"❌ {str(e)[:200]}")


# --- Генерация .pptx с Pexafy ---
@dp.message(Command("pptx"))
async def make_pptx(msg: types.Message):
    uid = msg.from_user.id; topic = msg.text.replace("/pptx", "").strip()
    if not topic: await msg.answer("📊 `/pptx тема`"); return
    await bot.send_chat_action(msg.chat.id, "typing")
    status = await msg.answer(f"📊 Готовлю: _{topic}_...")
    try:
        await status.edit_text("📊 Генерирую текст слайдов...")
        prompt = (f"Сделай презентацию на тему «{topic}». Верни ТОЛЬКО JSON-массив. "
                  f'Формат: [{{"title": "Заголовок", "points": ["пункт"], "image_prompt": "english search query"}}, ...] '
                  f"РОВНО 8 слайдов. 5-6 пунктов. image_prompt — короткий запрос НА АНГЛИЙСКОМ для поиска фото.")
        resp = await client.chat.completions.create(model="qwen/qwen3.8-27b", messages=[{"role": "user", "content": prompt}], temperature=0.5, max_tokens=1800)
        slides = parse_json_safe(resp.choices[0].message.content)
        if not slides: raise ValueError("Невалидный JSON")
        total = len(slides)
        await status.edit_text(f"📊 Ищу фото для {total} слайдов...")
        queries = [s.get("image_prompt", s.get("title", "abstract")) for s in slides]
        images = await asyncio.gather(*[search_stock_photo(q) for q in queries], return_exceptions=True)
        ok = sum(1 for i in images if isinstance(i, BytesIO))
        logging.info(f"[PPTX] Pexafy found: {ok}/{total}")
        await status.edit_text(f"📊 Собираю ({ok}/{total} фото)...")
        from pptx import Presentation
        from pptx.util import Inches, Pt
        prs = Presentation(); prs.slide_width = Inches(10); prs.slide_height = Inches(7.5)
        for i, s in enumerate(slides):
            title = s.get("title", f"Слайд {i+1}"); points = s.get("points", [])
            img = images[i] if i < len(images) and isinstance(images[i], BytesIO) else None
            slide = prs.slides.add_slide(prs.slide_layouts[6])
            tb = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(9), Inches(1))
            tb.text_frame.text = title; tb.text_frame.paragraphs[0].font.size = Pt(28); tb.text_frame.paragraphs[0].font.bold = True
            tbox = slide.shapes.add_textbox(Inches(0.5), Inches(1.5), Inches(5), Inches(5.5))
            tf = tbox.text_frame; tf.word_wrap = True
            for j, p in enumerate(points):
                para = tf.paragraphs[0] if j == 0 else tf.add_paragraph(); para.text = f"• {p}"; para.font.size = Pt(14)
            if img:
                img.seek(0)
                try: slide.shapes.add_picture(img, Inches(5.5), Inches(1.5), width=Inches(4), height=Inches(5.5))
                except Exception as e: logging.error(f"[PPTX] Pic: {e}")
        path = "presentation.pptx"; prs.save(path)
        comment = await get_file_comment("презентация", topic, uid)
        safe = "".join(c for c in topic if c.isalnum() or c in " -_")[:40]
        await msg.answer_document(FSInputFile(path, filename=f"{safe}.pptx"), caption=comment)
        await status.delete()
    except Exception as e:
        logging.error(f"PPTX error: {e}"); await status.edit_text(f"❌ {str(e)[:200]}")


# --- Спикерская речь ---
@dp.message(Command("speech"))
async def make_speech(msg: types.Message):
    uid = msg.from_user.id
    if not msg.document:
        await msg.answer("🎬 Скинь `.pptx` с caption типа `/speech 5 минут`.")
        return
    doc = msg.document
    await bot.send_chat_action(msg.chat.id, "typing")
    status = await msg.answer("🎬 Читаю презентацию...")
    try:
        slides_text = await read_document(doc.file_id, doc.file_name or "file.pptx")
        duration = msg.caption.replace("/speech", "").strip() if msg.caption else "5 минут"
        await status.edit_text(f"🎬 Пишу речь на {duration}...")
        prompt = (f"Напиши речь для защиты презентации на {duration}. "
                  f"Содержание слайдов:\n{slides_text[:8000]}\n\n"
                  f"Формат: связный текст, абзацы для каждого слайда. "
                  f"Начни с приветствия, закончи благодарностью. Без markdown.")
        resp = await client.chat.completions.create(model="qwen/qwen3.8-27b", messages=[{"role": "user", "content": prompt}], temperature=0.7, max_tokens=1500)
        speech = resp.choices[0].message.content
        from docx import Document
        docx = Document(); docx.add_heading(f"Речь для защиты: {doc.file_name}", 0)
        for para in speech.split("\n"):
            if para.strip(): docx.add_paragraph(para.strip())
        path = "speech.docx"; docx.save(path)
        comment = await get_file_comment("речь для защиты", doc.file_name or "презентация", uid)
        safe = "".join(c for c in (doc.file_name or "speech") if c.isalnum() or c in " .-_")[:30]
        await msg.answer_document(FSInputFile(path, filename=f"speech_{safe}.docx"), caption=comment)
        await status.delete()
    except Exception as e:
        logging.error(f"SPEECH error: {e}"); await status.edit_text(f"❌ {str(e)[:200]}")


# --- Доработка документов ---
async def handle_document_edit(msg, ai_response, file_bytes, file_name, ext, uid):
    try:
        safe = "".join(c for c in file_name if c.isalnum() or c in " .-_")
        if not safe.lower().endswith(f".{ext}"): safe = f"updated.{ext}"
        if ext == "txt":
            await msg.answer_document(BufferedInputFile(ai_response.encode("utf-8"), filename=f"updated_{safe}"), caption=await get_file_comment("файл", file_name, uid))
        elif ext == "docx":
            from docx import Document
            d = Document()
            for line in ai_response.split("\n"):
                line = line.strip()
                if not line: continue
                if len(line) < 80 and not line.endswith(".") and not line.startswith("-"): d.add_heading(line, level=1)
                else: d.add_paragraph(line)
            p = "updated.docx"; d.save(p)
            await msg.answer_document(FSInputFile(p, filename=f"updated_{safe}"), caption=await get_file_comment("обновлённый документ", file_name, uid))
        elif ext == "pptx":
            slides = parse_json_safe(ai_response)
            if not slides:
                slides = []
                for chunk in ai_response.split("---"):
                    lines = [l.strip() for l in chunk.strip().split("\n") if l.strip()]
                    if lines: slides.append({"title": lines[0], "points": [l.lstrip("- ").strip() for l in lines[1:]], "image_prompt": lines[0]})
            queries = [s.get("image_prompt", s.get("title", "abstract")) for s in slides]
            images = await asyncio.gather(*[search_stock_photo(q) for q in queries], return_exceptions=True)
            file_bytes.seek(0)
            from pptx import Presentation
            from pptx.util import Inches, Pt
            prs = Presentation(file_bytes)
            for i, s in enumerate(slides):
                title = s.get("title", ""); points = s.get("points", [])
                if not title: continue
                img = images[i] if i < len(images) and isinstance(images[i], BytesIO) else None
                slide = prs.slides.add_slide(prs.slide_layouts[6])
                tb = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(9), Inches(1))
                tb.text_frame.text = title; tb.text_frame.paragraphs[0].font.size = Pt(28); tb.text_frame.paragraphs[0].font.bold = True
                tbox = slide.shapes.add_textbox(Inches(0.5), Inches(1.5), Inches(5), Inches(5.5))
                tf = tbox.text_frame; tf.word_wrap = True
                for j, p in enumerate(points):
                    para = tf.paragraphs[0] if j == 0 else tf.add_paragraph(); para.text = f"• {p}"; para.font.size = Pt(14)
                if img:
                    img.seek(0)
                    try: slide.shapes.add_picture(img, Inches(5.5), Inches(1.5), width=Inches(4), height=Inches(5.5))
                    except Exception as e: logging.error(f"[PPTX EDIT] Pic: {e}")
            p = "updated.pptx"; prs.save(p)
            await msg.answer_document(FSInputFile(p, filename=f"updated_{safe}"), caption=await get_file_comment("обновлённая презентация", file_name, uid))
        elif ext == "pdf":
            await msg.answer(f"📄 PDF не пересобираю, текст:\n\n{ai_response[:3500]}")
    except Exception as e:
        logging.error(f"Doc edit: {e}"); await msg.answer(f"❌ {str(e)[:200]}")


# --- Основной ---
@dp.message()
async def chat(msg: types.Message):
    uid = msg.from_user.id; mode = await get_user_mode(uid); history = await get_history(uid)
    user_content = None; save_text = None; is_doc_edit = False; doc_info = None
    if msg.photo:
        photo = msg.photo[-1]; file = await bot.get_file(photo.file_id); data = await bot.download_file(file.file_path)
        b64 = base64.b64encode(data.read()).decode("utf-8")
        user_content = [{"type": "text", "text": msg.caption or "Что на фото?"}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]
        save_text = msg.caption or "[Фото]"
    elif msg.voice:
        await bot.send_chat_action(msg.chat.id, "typing")
        try: text = await transcribe_audio(msg.voice.file_id, "ogg")
        except Exception as e: await msg.answer(f"❌ {str(e)[:150]}"); return
        user_content = text; save_text = f"[Голос]: {text}"
    elif msg.audio:
        await bot.send_chat_action(msg.chat.id, "typing")
        ext = "mp3"
        if msg.audio.file_name and "." in msg.audio.file_name: ext = msg.audio.file_name.rsplit(".", 1)[-1].lower()
        try: text = await transcribe_audio(msg.audio.file_id, ext)
        except Exception as e: await msg.answer(f"❌ {str(e)[:150]}"); return
        user_content = text; save_text = f"[Аудио]: {text}"
    elif msg.document:
        doc = msg.document; fname = doc.file_name or "file"; ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else ""
        if ext not in ["txt", "docx", "pptx", "pdf"]: await msg.answer("❌ Поддерживаются: .txt, .docx, .pptx, .pdf"); return
        await bot.send_chat_action(msg.chat.id, "typing")
        try:
            text = await read_document(doc.file_id, fname)
            if not text.strip(): await msg.answer("❌ Файл пустой."); return
            doc_info = (await download_file_bytes(doc.file_id), fname, ext)
            user_content = f"[Файл: {fname}]\n\n{text[:8000]}\n\nЗапрос: {msg.caption or 'Что сделать?'}"
            save_text = f"[Документ {fname}]: {msg.caption or 'доработка'}"; is_doc_edit = True
        except Exception as e: await msg.answer(f"❌ {str(e)[:150]}"); return
    elif msg.text: user_content = msg.text; save_text = msg.text
    else: return
    await save_message(uid, "user", save_text); history.append({"role": "user", "content": user_content}); history = trim_history_by_chars(history)
    await bot.send_chat_action(msg.chat.id, "typing")
    personal = PERSONAL_PROMPTS.get(uid, ""); tone = await get_user_tone(uid)
    tone_add = f"\n\nТОН: {tone}" if tone else ""
    full_prompt = SYSTEM_PROMPT + personal + tone_add
    try:
        resp = await client.chat.completions.create(model="qwen/qwen3.8-27b", messages=[{"role": "system", "content": full_prompt}, *history], temperature=0.7, max_tokens=1200)
        answer = resp.choices[0].message.content; await save_message(uid, "assistant", answer)
        if is_doc_edit and doc_info:
            fb, fn, ext = doc_info; await handle_document_edit(msg, answer, fb, fn, ext, uid)
        else:
            use_file = (mode == "file") or (len(answer) > 4000)
            if use_file:
                ext_out = detect_extension(answer); fname_out = f"light_answer.{ext_out}"
                comment = await get_file_comment(f"файл .{ext_out}", "код", uid)
                await msg.answer_document(BufferedInputFile(answer.encode("utf-8"), filename=fname_out), caption=comment)
            else:
                parts = split_message(answer)
                if len(parts) == 1: await msg.answer(parts[0])
                else:
                    for i, p in enumerate(parts, 1): await msg.answer(f"📄 Часть {i}/{len(parts)}\n\n{p}")
            if mode == "voice":
                await bot.send_chat_action(msg.chat.id, "record_voice")
                try:
                    vf = await text_to_voice(answer)
                    if vf: await msg.answer_voice(FSInputFile(vf))
                except Exception as e: logging.error(f"TTS: {e}")
    except Exception as e:
        logging.error(f"Ошибка: {e}"); await msg.answer(f"❌ {str(e)[:300]}")


# --- Веб-сервер ---
async def handle(request): return web.Response(text="Bot is running!")

async def main():
    global db_pool
    logging.basicConfig(level=logging.INFO)
    db_pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=5); await init_db()
    logging.info("База данных подключена")
    if hf_client: logging.info("Hugging Face клиент инициализирован")
    if pexafy_client: logging.info("Pexafy клиент инициализирован")
    app = web.Application(); app.router.add_get("/", handle); runner = web.AppRunner(app); await runner.setup()
    port = int(os.environ.get("PORT", 8080)); site = web.TCPSite(runner, "0.0.0.0", port); await site.start()
    logging.info(f"Web server on port {port}"); await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
