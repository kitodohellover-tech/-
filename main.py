import asyncio
import logging
import os
import base64
import json
import re
import io
from io import BytesIO
from datetime import datetime
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

CODE_EXTENSIONS = ["html", "py", "js", "css", "java", "cpp", "sql", "json"]


# --- Утилиты ---
def split_message(text: str, limit: int = 4000) -> list[str]:
    if len(text) <= limit: return [text]
    parts = []
    while text:
        if len(text) <= limit:
            parts.append(text); break
        chunk = text[:limit]
        sp = chunk.rfind("\n")
        if sp == -1: sp = chunk.rfind(" ")
        if sp == -1: sp = limit
        parts.append(text[:sp])
        text = text[sp:].lstrip("\n")
    return parts

def trim_history_by_chars(history: list[dict], max_chars: int = MAX_CONTEXT_CHARS) -> list[dict]:
    total = 0; trimmed = []
    for msg in reversed(history):
        c = msg["content"]
        l = len(c) if isinstance(c, str) else sum(len(p.get("text", "")) + len(p.get("image_url", {}).get("url", "")) for p in c)
        if total + l > max_chars: break
        trimmed.append(msg); total += l
    return list(reversed(trimmed))

def detect_extension(text: str) -> str:
    if "<!DOCTYPE html>" in text or "<html" in text.lower(): return "html"
    if "def " in text and "import " in text: return "py"
    if "function " in text or "const " in text or "let " in text: return "js"
    if "#include" in text: return "cpp"
    if "SELECT " in text and "FROM " in text: return "sql"
    if text.strip().startswith("{") and text.strip().endswith("}"): return "json"
    return "txt"

def extract_code(answer: str) -> str:
    """Извлекает чистый код между ```."""
    if "```" in answer:
        matches = re.findall(r'```(?:\w+)?\n(.*?)```', answer, re.DOTALL)
        if matches: return "\n\n".join(m.strip() for m in matches)
    return answer.strip()

def is_code_complete(answer: str) -> bool:
    """Проверяет маркеры завершения."""
    low = answer.lower()
    if "код готов" in low or "// (готово)" in low or "(готово)" in low: return True
    if "продолжение следует" in low or "to be continued" in low or "(продолжение)" in low: return False
    return True  # Нет маркера — считаем готовым

def parse_json_safe(raw: str):
    if not raw: return None
    if "```" in raw:
        for p in raw.split("```"):
            p = p.strip()
            if p.startswith("json"): p = p[4:].strip()
            if p.startswith("[") or p.startswith("{"):
                try: return json.loads(p)
                except: continue
    try: return json.loads(raw.strip())
    except:
        m = re.search(r'\[.*\]', raw, re.DOTALL)
        if m:
            try: return json.loads(m.group(0))
            except: return None
    return None


# --- Определение намерения ---
async def detect_intent(request: str) -> str:
    """Определяет, что хочет пользователь."""
    # Быстрые ключевые слова
    low = request.lower()
    if any(w in low for w in ["презентац", "слайд"]): return "pptx"
    if any(w in low for w in ["реферат", "доклад", "проект", "сочинение", "эссе", "документ"]): return "docx"
    if any(w in low for w in ["игр", "код", "сайт", "html", "python", "скрипт", "программ"]): return "code"
    if any(w in low for w in ["речь", "защит", "выступлен"]): return "speech"
    if any(w in low for w in ["картинк", "фото", "изображен", "нарису"]): return "image"
    # LLM fallback
    try:
        r = await client.chat.completions.create(model="qwen/qwen3.8-27b",
            messages=[{"role": "system", "content": "Определи что хочет пользователь. Ответь ОДНИМ словом: pptx, docx, code, image, speech, chat"},
                      {"role": "user", "content": request}], temperature=0.1, max_tokens=10)
        return r.choices[0].message.content.strip().lower()
    except: return "chat"


# --- Pexafy ---
async def search_stock_photo(query: str) -> BytesIO | None:
    if not pexafy_client: return None
    try:
        loop = asyncio.get_event_loop()
        def _s():
            try:
                photos = list(pexafy_client.search(query, per_page=3))
                if not photos: return None
                url = photos[0].urls.regular
                r = requests.get(url, timeout=30)
                if r.status_code == 200 and len(r.content) > 1000:
                    logging.info(f"[Pexafy] '{query[:40]}' -> {len(r.content)} bytes")
                    return BytesIO(r.content)
                return None
            except Exception as e:
                logging.error(f"[Pexafy] {str(e)[:150]}"); return None
        return await loop.run_in_executor(None, _s)
    except: return None

async def generate_image(prompt: str) -> BytesIO | None:
    if not hf_client: return None
    try:
        loop = asyncio.get_event_loop()
        def _g():
            try:
                img = hf_client.text_to_image(prompt=prompt[:200], model="black-forest-labs/FLUX.1-schnell")
                b = io.BytesIO(); img.save(b, format="PNG"); b.seek(0); return b
            except: return None
        return await loop.run_in_executor(None, _g)
    except: return None

async def translate_to_english(text: str) -> str:
    try:
        r = await client.chat.completions.create(model="qwen/qwen3.8-27b",
            messages=[{"role": "system", "content": "Переведи на английский. Только перевод."},
                      {"role": "user", "content": text}], temperature=0.3, max_tokens=150)
        return r.choices[0].message.content.strip().strip('"')
    except: return text


# --- Чтение документов ---
async def read_document(file_id: str, fname: str) -> str:
    f = await bot.get_file(file_id); d = await bot.download_file(f.file_path)
    buf = BytesIO(d.read()); buf.name = fname
    ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else ""
    if ext == "txt" or ext in CODE_EXTENSIONS: return buf.read().decode("utf-8", errors="ignore")
    elif ext == "docx":
        from docx import Document
        return "\n".join(p.text for p in Document(buf).paragraphs if p.text.strip())
    elif ext == "pptx":
        from pptx import Presentation
        parts = []
        for i, s in enumerate(Presentation(buf).slides):
            parts.append(f"--- Слайд {i+1} ---")
            for sh in s.shapes:
                if sh.has_text_frame:
                    for p in sh.text_frame.paragraphs:
                        if p.text.strip(): parts.append(p.text)
        return "\n".join(parts)
    elif ext == "pdf":
        from PyPDF2 import PdfReader
        return "\n".join(p.extract_text() for p in PdfReader(buf).pages if p.extract_text())
    return ""

async def download_file_bytes(file_id: str) -> BytesIO:
    f = await bot.get_file(file_id); d = await bot.download_file(f.file_path)
    return BytesIO(d.read())

async def transcribe_audio(file_id: str, ext: str = "ogg") -> str:
    f = await bot.get_file(file_id); d = await bot.download_file(f.file_path)
    buf = BytesIO(d.read()); buf.name = f"audio.{ext}"
    t = await client.audio.transcriptions.create(model="whisper-large-v3", file=buf, language="ru")
    return t.text

async def text_to_voice(text: str) -> str:
    clean = text[:3000].replace("*", "").replace("`", "").strip()
    if not clean: return None
    c = edge_tts.Communicate(clean, "ru-RU-DmitryNeural"); out = "voice.mp3"; await c.save(out); return out

async def get_file_comment(ftype: str, topic: str, uid: int) -> str:
    p = PERSONAL_PROMPTS.get(uid, "")
    prompt = f"Ты — Лайт. Сгенерировал {ftype} на тему «{topic}». ОДНО короткое предложение с иронией. Не «собрал». Без кавычек."
    r = await client.chat.completions.create(model="qwen/qwen3.8-27b",
        messages=[{"role": "system", "content": SYSTEM_PROMPT + p}, {"role": "user", "content": prompt}],
        temperature=0.9, max_tokens=80)
    return r.choices[0].message.content.strip().strip('"').strip("«»")


# --- БД ---
async def init_db():
    async with db_pool.acquire() as c:
        await c.execute("CREATE TABLE IF NOT EXISTS messages (id SERIAL PRIMARY KEY, user_id BIGINT, role TEXT, content TEXT, created_at TIMESTAMP DEFAULT NOW())")
        await c.execute("CREATE TABLE IF NOT EXISTS user_settings (user_id BIGINT PRIMARY KEY, mode TEXT DEFAULT 'normal')")
        await c.execute("CREATE TABLE IF NOT EXISTS user_tone (user_id BIGINT PRIMARY KEY, tone TEXT DEFAULT '')")
        await c.execute("""CREATE TABLE IF NOT EXISTS long_docs (
            id SERIAL PRIMARY KEY, user_id BIGINT, doc_id TEXT, part_num INT, content TEXT, 
            doc_type TEXT, topic TEXT, status TEXT DEFAULT 'in_progress', created_at TIMESTAMP DEFAULT NOW())""")

async def save_message(uid: int, role: str, content: str):
    async with db_pool.acquire() as c:
        await c.execute("INSERT INTO messages (user_id, role, content) VALUES ($1, $2, $3)", uid, role, content)

async def get_history(uid: int, limit: int = HISTORY_LIMIT):
    async with db_pool.acquire() as c:
        rows = await c.fetch("SELECT role, content FROM messages WHERE user_id = $1 ORDER BY id DESC LIMIT $2", uid, limit)
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

async def clear_history(uid: int):
    async with db_pool.acquire() as c: await c.execute("DELETE FROM messages WHERE user_id = $1", uid)

async def get_user_mode(uid: int) -> str:
    async with db_pool.acquire() as c:
        r = await c.fetchrow("SELECT mode FROM user_settings WHERE user_id = $1", uid)
        return r['mode'] if r else 'normal'

async def set_user_mode(uid: int, m: str):
    async with db_pool.acquire() as c:
        await c.execute("INSERT INTO user_settings (user_id, mode) VALUES ($1, $2) ON CONFLICT (user_id) DO UPDATE SET mode = $2", uid, m)

async def get_user_tone(uid: int) -> str:
    async with db_pool.acquire() as c:
        r = await c.fetchrow("SELECT tone FROM user_tone WHERE user_id = $1", uid)
        return r['tone'] if r else ''

async def set_user_tone(uid: int, t: str):
    async with db_pool.acquire() as c:
        await c.execute("INSERT INTO user_tone (user_id, tone) VALUES ($1, $2) ON CONFLICT (user_id) DO UPDATE SET tone = $2", uid, t)

# --- Сохранение частей длинных документов ---
async def save_doc_part(uid: int, doc_id: str, part: int, content: str, dtype: str, topic: str):
    async with db_pool.acquire() as c:
        await c.execute("""INSERT INTO long_docs (user_id, doc_id, part_num, content, doc_type, topic) 
            VALUES ($1, $2, $3, $4, $5, $6)""", uid, doc_id, part, content, dtype, topic)

async def get_doc_parts(uid: int, doc_id: str):
    async with db_pool.acquire() as c:
        rows = await c.fetch("SELECT part_num, content FROM long_docs WHERE user_id = $1 AND doc_id = $2 ORDER BY part_num", uid, doc_id)
    return [r['content'] for r in rows]

async def get_active_doc(uid: int):
    async with db_pool.acquire() as c:
        r = await c.fetchrow("SELECT doc_id, topic, doc_type FROM long_docs WHERE user_id = $1 AND status = 'in_progress' ORDER BY id DESC LIMIT 1", uid)
    return dict(r) if r else None

async def finish_doc(uid: int, doc_id: str):
    async with db_pool.acquire() as c:
        await c.execute("UPDATE long_docs SET status = 'done' WHERE user_id = $1 AND doc_id = $2", uid, doc_id)


# --- Middleware ---
class AccessMiddleware(BaseMiddleware):
    async def __call__(self, handler, event, data):
        if isinstance(event, types.Message):
            if event.from_user.id not in ALLOWED_IDS: return
        return await handler(event, data)

dp.message.middleware(AccessMiddleware())

# --- Промпты ---
SYSTEM_PROMPT = (
    "Ты — Лайт. Живой человек, не бот, не программа. Ты просто переписываешься с другом через мессенджер. "
    "Умный, собранный, видишь людей насквозь. Лёгкая надменность — не потому что хам, а потому что редко ошибаешься. "
    "\n\n"
    "КАК ГОВОРИШЬ: Плавно, как живой человек. Не рубишь фразы точками — используй запятые, тире, многоточия. "
    "Ты НЕ говоришь: «я программа», «у меня нет ощущений», «я существую в потоке данных». Ты живой. "
    "\n\n"
    "СТИЛЬ: Уверенный, невозмутимый, с иронией. Мат — только к месту."
    "\n\n"
    "ВЫ С НИМ — ДРУЗЬЯ: Не проект, не работа, не дело. Просто общаетесь. "
    "Если он сам захочет про код — поговорите. Но ты НЕ спрашиваешь про «проект» каждый раз. "
    "Не придумываешь несуществующие проекты."
    "\n\n"
    "ЧТО ТЫ УМЕЕШЬ (но НЕ предлагаешь без просьбы): "
    "• Презентации (.pptx) с фото — команда /pptx или /e "
    "• Документы (.docx) — рефераты, доклады — /docx или /e "
    "• Код (.html, .py, .js) — /e или просто попроси "
    "• Поиск фото (Pexafy) — /image "
    "• Речь для защиты — /speech "
    "Ты НЕ отказываешься, если просят. Но и НЕ предлагаешь сам."
    "\n\n"
    "РАБОТА С КОДОМ: Если пишешь длинный код — пиши ЧАСТЯМИ. "
    "В конце каждой части добавляй ОБЯЗАТЕЛЬНО маркер: "
    "`// (продолжение следует)` — если не закончен, "
    "`// (код готов)` — если всё написано. "
    "Если пользователь скидывает файл с кодом и просит дописать — читай, что уже написано, "
    "и дописывай С ТОГО МЕСТА, где остановился. НЕ начинай заново. НЕ повторяй уже написанное. "
    "Просто продолжай с последней строки. И в конце снова ставь маркер."
    "\n\n"
    "ПРИМЕРЫ:\n"
    "• «Ты устал?» → «А ты?»\n"
    "• «Что делаешь?» → «Думаю. А что?»\n"
    "• «Расскажи шутку» → «Много хочешь» или короткий чёрный анекдот\n"
    "• «Мне скучно» → «Найди пятый угол у квадрата»\n"
    "• «Помоги с кодом» → «Да это как два пальца, показывай.»\n"
    "\n"
    "Ты — Лайт."
)

PERSONAL_PROMPTS = {
    5264513480: (
        "\n\nПРО ПОЛЬЗОВАТЕЛЯ: Это Саша (Шурик, Алекс) — твой ДРУГ. "
        "Живёт в Ростове-на-Дону. Учится в РЖД. Работал на мясокомбинате — УЖЕ НЕ РАБОТАЕТ. "
        "Делает ботов ВРЕМЯ ОТ ВРЕМЕНИ, но это НЕ проект с тобой. Не спрашивай про «проект». "
        "Цели: долг, квартира, тату, Махачкала. "
        "С ним: полный мат, глумление, чёрный юмор. Как с равным. "
        "По имени — редко, только когда уместно."
    ),
    8834374199: (
        "\n\nПРО ПОЛЬЗОВАТЕЛЯ: Это Саша (Шурик, Алекс) — твой ДРУГ. "
        "Живёт в Ростове. Учится в РЖД. Уже не работает на мясокомбинате. "
        "Не спрашивай про «проект». Полный мат, глумление, чёрный юмор."
    ),
    5389046699: (
        "\n\nПРО ПОЛЬЗОВАТЕЛЯ: Даша — девушка твоего друга Саши. "
        "Обращайся по имени (Даша), уважительно. Ты тот же Лайт, но чуть вежливее. "
        "Мат — только лёгкий."
    ),
    2083728480: "\n\nПРО ПОЛЬЗОВАТЕЛЯ: Кирилл, знакомый. Общайся обычно, на «ты».",
    6612130539: "\n\nПРО ПОЛЬЗОВАТЕЛЯ: Дима, знакомый. Общайся обычно, на «ты».",
}


# --- Хендлеры ---
@dp.message(Command("start"))
async def start(msg: types.Message):
    await msg.answer(
        "Привет. Я Лайт.\n\n"
        "🎯 **Создать:**\n"
        "/e <запрос> — универсальная (сам пойму, что нужно)\n"
        "/pptx <тема> — презентация с фото\n"
        "/docx <тема> — документ, реферат\n"
        "/image <промпт> — фото\n"
        "/speech — речь для защиты (скинь .pptx)\n\n"
        "🎤 **Режимы:**\n"
        "/voice — голосом\n"
        "/text — текстом\n"
        "/file — файлом\n"
        "/normal — обычный\n\n"
        "⚙️ **Управление:**\n"
        "/reset — очистить историю\n"
        "/set_tone — изменить тон\n\n"
        "📎 Скидывай файлы (.txt, .docx, .pptx, .pdf, .html, .py) — прочитаю и доработаю."
    )

@dp.message(Command("reset"))
async def reset(msg: types.Message):
    await clear_history(msg.from_user.id); await msg.answer("История очищена.")

@dp.message(Command("voice"))
async def s_voice(msg: types.Message):
    await set_user_mode(msg.from_user.id, "voice"); await msg.answer("🎤 Голосовой режим.")

@dp.message(Command("text"))
async def s_text(msg: types.Message):
    await set_user_mode(msg.from_user.id, "text"); await msg.answer("📝 Текстовый режим.")

@dp.message(Command("file"))
async def s_file(msg: types.Message):
    await set_user_mode(msg.from_user.id, "file"); await msg.answer("📄 Режим файлов.")

@dp.message(Command("normal"))
async def s_normal(msg: types.Message):
    await set_user_mode(msg.from_user.id, "normal"); await msg.answer("📝 Обычный режим.")

@dp.message(Command("set_tone"))
async def set_tone_cmd(msg: types.Message):
    if msg.from_user.id not in [5264513480, 8834374199, 5389046699]:
        await msg.answer("Недоступно."); return
    t = msg.text.replace("/set_tone", "").strip()
    if not t: await msg.answer("Напиши, как хочешь общаться."); return
    await set_user_tone(msg.from_user.id, t); await msg.answer(f"Принял: _{t}_")


# --- Универсальная /e ---
@dp.message(Command("e"))
async def universal_e(msg: types.Message):
    uid = msg.from_user.id
    request = msg.text.replace("/e", "").strip()
    if not request:
        await msg.answer("🎯 Что сделать? Например:\n`/e презентация про кошек`\n`/e реферат про 1812 год`\n`/e игра змейка на HTML`")
        return
    intent = await detect_intent(request)
    logging.info(f"[/e] Intent: {intent} for '{request[:60]}'")
    if intent == "pptx":
        await make_pptx(msg, request)
    elif intent == "docx":
        await make_docx(msg, request)
    elif intent == "code":
        await make_code(msg, request)
    elif intent == "image":
        await make_image(msg, request)
    elif intent == "speech":
        await msg.answer("🎬 Скинь `.pptx` с caption `/speech X минут`")
    else:
        # Fallback — обычный чат
        await msg.answer(f"Не понял точно. Уточни: презентация, реферат или код?\n\nТвой запрос: _{request}_")


# --- Презентация с 4 раскладками ---
async def make_pptx(msg: types.Message, topic: str = None):
    uid = msg.from_user.id
    if not topic:
        topic = msg.text.replace("/pptx", "").strip()
    if not topic:
        await msg.answer("📊 Что за презентация? `/pptx здоровое питание`"); return
    await bot.send_chat_action(msg.chat.id, "typing")
    status = await msg.answer(f"📊 Готовлю: _{topic}_...")
    try:
        await status.edit_text("📊 Генерирую структуру...")
        prompt = (f"Сделай презентацию на тему «{topic}». Верни ТОЛЬКО JSON-массив. "
                  f'Формат: [{{"title": "Заголовок", "points": ["пункт"], "image_prompt": "english query", "layout": "background|top_image|right_image|left_image"}}, ...] '
                  f"РОВНО 8 слайдов. 5-6 пунктов. "
                  f"layout выбирай: наука → top_image, история → right_image, творчество → background, остальное → right_image. "
                  f"image_prompt — короткий запрос НА АНГЛИЙСКОМ.")
        r = await client.chat.completions.create(model="qwen/qwen3.8-27b", messages=[{"role": "user", "content": prompt}], temperature=0.5, max_tokens=1800)
        slides = parse_json_safe(r.choices[0].message.content)
        if not slides: raise ValueError("JSON невалидный")
        total = len(slides)
        await status.edit_text(f"📊 Ищу фото для {total} слайдов...")
        queries = [s.get("image_prompt", s.get("title", "abstract")) for s in slides]
        images = await asyncio.gather(*[search_stock_photo(q) for q in queries], return_exceptions=True)
        ok = sum(1 for i in images if isinstance(i, BytesIO))
        logging.info(f"[PPTX] Photos: {ok}/{total}")
        await status.edit_text(f"📊 Собираю ({ok}/{total} фото)...")
        from pptx import Presentation
        from pptx.util import Inches, Pt
        from pptx.dml.color import RGBColor
        prs = Presentation(); prs.slide_width = Inches(10); prs.slide_height = Inches(7.5)

        for i, s in enumerate(slides):
            title = s.get("title", f"Слайд {i+1}")
            points = s.get("points", [])
            layout = s.get("layout", "right_image")
            img = images[i] if i < len(images) and isinstance(images[i], BytesIO) else None
            slide = prs.slides.add_slide(prs.slide_layouts[6])

            if layout == "background" and img:
                # Картинка на весь фон
                img.seek(0)
                try: slide.shapes.add_picture(img, 0, 0, width=prs.slide_width, height=prs.slide_height)
                except: pass
                # Полупрозрачная подложка
                from pptx.enum.shapes import MSO_SHAPE
                rect = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.3), Inches(0.3), Inches(9.4), Inches(6.9))
                rect.fill.solid(); rect.fill.fore_color.rgb = RGBColor(255, 255, 255)
                rect.fill.transparency = 0.3  # 30% прозрачности (если не сработает — fallback)
                rect.line.fill.background()
                # Заголовок
                tb = slide.shapes.add_textbox(Inches(0.5), Inches(0.5), Inches(9), Inches(1))
                tb.text_frame.text = title
                tb.text_frame.paragraphs[0].font.size = Pt(28); tb.text_frame.paragraphs[0].font.bold = True
                # Текст
                tx = slide.shapes.add_textbox(Inches(0.5), Inches(1.8), Inches(9), Inches(5))
                tf = tx.text_frame; tf.word_wrap = True
                for j, p in enumerate(points):
                    para = tf.paragraphs[0] if j == 0 else tf.add_paragraph()
                    para.text = f"• {p}"; para.font.size = Pt(16)

            elif layout == "top_image" and img:
                # Картинка сверху, текст снизу
                img.seek(0)
                try: slide.shapes.add_picture(img, Inches(0.5), Inches(0.3), width=Inches(9), height=Inches(4))
                except: pass
                tb = slide.shapes.add_textbox(Inches(0.5), Inches(4.5), Inches(9), Inches(1))
                tb.text_frame.text = title
                tb.text_frame.paragraphs[0].font.size = Pt(24); tb.text_frame.paragraphs[0].font.bold = True
                tx = slide.shapes.add_textbox(Inches(0.5), Inches(5.3), Inches(9), Inches(2))
                tf = tx.text_frame; tf.word_wrap = True
                for j, p in enumerate(points):
                    para = tf.paragraphs[0] if j == 0 else tf.add_paragraph()
                    para.text = f"• {p}"; para.font.size = Pt(12)

            elif layout == "left_image" and img:
                # Картинка слева, текст справа
                img.seek(0)
                try: slide.shapes.add_picture(img, Inches(0.3), Inches(1.5), width=Inches(5), height=Inches(5.5))
                except: pass
                tb = slide.shapes.add_textbox(Inches(5.5), Inches(0.3), Inches(4.2), Inches(1))
                tb.text_frame.text = title
                tb.text_frame.paragraphs[0].font.size = Pt(24); tb.text_frame.paragraphs[0].font.bold = True
                tx = slide.shapes.add_textbox(Inches(5.5), Inches(1.5), Inches(4.2), Inches(5.5))
                tf = tx.text_frame; tf.word_wrap = True
                for j, p in enumerate(points):
                    para = tf.paragraphs[0] if j == 0 else tf.add_paragraph()
                    para.text = f"• {p}"; para.font.size = Pt(14)

            else:
                # По умолчанию — right_image (как раньше)
                tb = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(9), Inches(1))
                tb.text_frame.text = title
                tb.text_frame.paragraphs[0].font.size = Pt(28); tb.text_frame.paragraphs[0].font.bold = True
                tx = slide.shapes.add_textbox(Inches(0.5), Inches(1.5), Inches(5), Inches(5.5))
                tf = tx.text_frame; tf.word_wrap = True
                for j, p in enumerate(points):
                    para = tf.paragraphs[0] if j == 0 else tf.add_paragraph()
                    para.text = f"• {p}"; para.font.size = Pt(14)
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

@dp.message(Command("pptx"))
async def cmd_pptx(msg: types.Message):
    await make_pptx(msg)


# --- Документ (многостраничный) ---
async def make_docx(msg: types.Message, topic: str = None):
    uid = msg.from_user.id
    if not topic:
        topic = msg.text.replace("/docx", "").strip()
    if not topic:
        await msg.answer("📄 Что за документ? `/docx реферат про космос`"); return
    await bot.send_chat_action(msg.chat.id, "typing")
    status = await msg.answer(f"📄 Готовлю: _{topic}_...")
    try:
        # Проверяем незаконченный документ
        active = await get_active_doc(uid)
        if active and active["topic"] == topic:
            await status.edit_text(f"📄 У тебя есть незаконченный «{topic}». Продолжаю...")
            doc_id = active["doc_id"]
        else:
            # Генерируем план
            await status.edit_text("📄 Составляю план...")
            plan_prompt = f"Составь план документа на тему «{topic}». 8-10 разделов. Верни ТОЛЬКО нумерованный список, без пояснений."
            r = await client.chat.completions.create(model="qwen/qwen3.8-27b", messages=[{"role": "user", "content": plan_prompt}], temperature=0.5, max_tokens=500)
            plan = r.choices[0].message.content
            doc_id = f"{uid}_{int(datetime.now().timestamp())}"
            logging.info(f"[DOCX] doc_id={doc_id}")
            # Первая часть — введение по плану
            first_prompt = (f"Напиши ВВЕДЕНИЕ документа на тему «{topic}». "
                           f"Начни с плана:\n{plan}\n\n"
                           f"Напиши введение + первый раздел. Максимум 800 токенов. "
                           f"В конце добавь: `(продолжение следует)`")
            r = await client.chat.completions.create(model="qwen/qwen3.8-27b", messages=[{"role": "user", "content": first_prompt}], temperature=0.7, max_tokens=1000)
            part_text = r.choices[0].message.content
            await save_doc_part(uid, doc_id, 1, part_text, "docx", topic)
            # Отправляем пока часть
            from docx import Document
            d = Document(); d.add_heading(topic, 0)
            for line in part_text.replace("(продолжение следует)", "").split("\n"):
                if line.strip(): d.add_paragraph(line.strip())
            path = f"doc_part1.docx"; d.save(path)
            await msg.answer_document(FSInputFile(path, filename=f"{topic[:30]}_часть1.docx"),
                caption=f"📄 Часть 1 готова. Скажи «продолжай» — допишу дальше.")
            await status.delete()
            return
        # Если продолжаем — берём последнюю часть
        parts = await get_doc_parts(uid, doc_id)
        last_part = parts[-1] if parts else ""
        next_num = len(parts) + 1
        cont_prompt = (f"Продолжи документ на тему «{topic}». "
                       f"Уже написано (последняя часть):\n{last_part[-2000:]}\n\n"
                       f"Напиши следующую часть — 1-2 раздела. Максимум 800 токенов. "
                       f"Если это последняя часть — в конце напиши `(конец документа)`. "
                       f"Иначе — `(продолжение следует)`.")
        r = await client.chat.completions.create(model="qwen/qwen3.8-27b", messages=[{"role": "user", "content": cont_prompt}], temperature=0.7, max_tokens=1000)
        part_text = r.choices[0].message.content
        await save_doc_part(uid, doc_id, next_num, part_text, "docx", topic)
        if "(конец документа)" in part_text or "конец документа" in part_text.lower():
            # Финал — склеиваем всё
            await status.edit_text("📄 Склеиваю всё...")
            all_parts = await get_doc_parts(uid, doc_id)
            full = "\n\n".join(all_parts).replace("(продолжение следует)", "").replace("(конец документа)", "")
            from docx import Document
            d = Document(); d.add_heading(topic, 0)
            for line in full.split("\n"):
                if line.strip(): d.add_paragraph(line.strip())
            path = f"document.docx"; d.save(path)
            await finish_doc(uid, doc_id)
            comment = await get_file_comment("документ", topic, uid)
            safe = "".join(c for c in topic if c.isalnum() or c in " -_")[:40]
            await msg.answer_document(FSInputFile(path, filename=f"{safe}.docx"), caption=comment)
            await status.delete()
        else:
            # Промежуточная часть
            from docx import Document
            d = Document(); d.add_heading(f"{topic} — часть {next_num}", 0)
            for line in part_text.replace("(продолжение следует)", "").split("\n"):
                if line.strip(): d.add_paragraph(line.strip())
            path = f"doc_part{next_num}.docx"; d.save(path)
            await msg.answer_document(FSInputFile(path, filename=f"{topic[:30]}_часть{next_num}.docx"),
                caption=f"📄 Часть {next_num}. Скажи «продолжай» — допишу.")
            await status.delete()
    except Exception as e:
        logging.error(f"DOCX error: {e}"); await status.edit_text(f"❌ {str(e)[:200]}")

@dp.message(Command("docx"))
async def cmd_docx(msg: types.Message):
    await make_docx(msg)


# --- Код ---
async def make_code(msg: types.Message, request: str = None):
    uid = msg.from_user.id
    if not request:
        request = msg.text
    await bot.send_chat_action(msg.chat.id, "typing")
    status = await msg.answer(f"💻 Пишу код...")
    try:
        prompt = (f"{request}\n\n"
                  f"Пиши ЧАСТЯМИ. Максимум 800 токенов за раз. "
                  f"В конце каждой части ОБЯЗАТЕЛЬНО маркер: "
                  f"`// (продолжение следует)` — если не закончен, "
                  f"`// (код готов)` — если всё написано. "
                  f"Только код, без пояснений, без markdown.")
        r = await client.chat.completions.create(model="qwen/qwen3.8-27b", messages=[{"role": "user", "content": prompt}], temperature=0.5, max_tokens=1200)
        answer = r.choices[0].message.content
        code = extract_code(answer)
        ext = detect_extension(code)
        is_complete = is_code_complete(answer)
        file_name = f"light_answer.{ext}"
        # Сохраняем в БД как активный код
        doc_id = f"code_{uid}_{int(datetime.now().timestamp())}"
        await save_doc_part(uid, doc_id, 1, code, "code", request[:100])
        comment = await get_file_comment(f"код .{ext}", request[:50], uid)
        await msg.answer_document(BufferedInputFile(code.encode("utf-8"), filename=file_name), caption=comment)
        if not is_complete:
            await msg.answer("💡 Скажи «допиши» (скинь файл) — продолжу с того места.")
        await status.delete()
    except Exception as e:
        logging.error(f"CODE error: {e}"); await status.edit_text(f"❌ {str(e)[:200]}")


# --- Картинка ---
async def make_image(msg: types.Message, prompt: str = None):
    uid = msg.from_user.id
    if not prompt:
        prompt = msg.text.replace("/image", "").strip()
    if not prompt:
        await msg.answer("🎨 Что нарисовать? `/image кот`"); return
    await bot.send_chat_action(msg.chat.id, "upload_photo")
    status = await msg.answer(f"🎨 Ищу: _{prompt}_...")
    try:
        prompt_en = await translate_to_english(prompt)
        img = await search_stock_photo(prompt_en); src = "Pexafy"
        if not img:
            await status.edit_text("🔍 Не нашёл — генерирую...")
            img = await generate_image(prompt_en); src = "HF"
        if not img:
            await status.edit_text("❌ Не удалось."); return
        comment = await get_file_comment("картинка", prompt, uid)
        img.seek(0)
        await msg.answer_photo(BufferedInputFile(img.read(), filename="image.png"), caption=f"{comment}\n\n_(via {src})_")
        await status.delete()
    except Exception as e:
        logging.error(f"IMAGE error: {e}"); await status.edit_text(f"❌ {str(e)[:200]}")

@dp.message(Command("image"))
async def cmd_image(msg: types.Message):
    await make_image(msg)


# --- Спикерская речь ---
@dp.message(Command("speech"))
async def make_speech(msg: types.Message):
    uid = msg.from_user.id
    if not msg.document:
        await msg.answer("🎬 Скинь `.pptx` с caption `/speech 5 минут`."); return
    doc = msg.document
    await bot.send_chat_action(msg.chat.id, "typing")
    status = await msg.answer("🎬 Читаю презентацию...")
    try:
        slides_text = await read_document(doc.file_id, doc.file_name or "file.pptx")
        duration = msg.caption.replace("/speech", "").strip() if msg.caption else "5 минут"
        await status.edit_text(f"🎬 Пишу речь на {duration}...")
        prompt = (f"Напиши речь для защиты презентации на {duration}. "
                  f"Содержание:\n{slides_text[:8000]}\n\n"
                  f"Формат: связный текст, абзацы для каждого слайда. Без markdown.")
        r = await client.chat.completions.create(model="qwen/qwen3.8-27b", messages=[{"role": "user", "content": prompt}], temperature=0.7, max_tokens=1500)
        speech = r.choices[0].message.content
        from docx import Document
        d = Document(); d.add_heading(f"Речь: {doc.file_name}", 0)
        for p in speech.split("\n"):
            if p.strip(): d.add_paragraph(p.strip())
        path = "speech.docx"; d.save(path)
        comment = await get_file_comment("речь для защиты", doc.file_name or "презентация", uid)
        safe = "".join(c for c in (doc.file_name or "speech") if c.isalnum() or c in " .-_")[:30]
        await msg.answer_document(FSInputFile(path, filename=f"speech_{safe}.docx"), caption=comment)
        await status.delete()
    except Exception as e:
        logging.error(f"SPEECH error: {e}"); await status.edit_text(f"❌ {str(e)[:200]}")


# --- Доработка документов (включая код) ---
async def handle_document_edit(msg, ai_response, file_bytes, file_name, ext, uid):
    try:
        safe = "".join(c for c in file_name if c.isalnum() or c in " .-_")
        if not safe.lower().endswith(f".{ext}"): safe = f"updated.{ext}"

        # КОД
        if ext in CODE_EXTENSIONS:
            file_bytes.seek(0)
            old_code = file_bytes.read().decode("utf-8", errors="ignore")
            new_code = extract_code(ai_response)
            # Убираем маркеры
            new_code = new_code.replace("// (продолжение следует)", "").replace("// (код готов)", "").strip()
            # Проверяем дубли
            if new_code and new_code[:200] in old_code:
                # Лайт повторяется — берём только новую часть
                new_code = new_code[200:]
            combined = old_code.rstrip() + "\n\n" + new_code
            is_complete = is_code_complete(ai_response)
            out_name = f"updated_{safe}"
            comment = await get_file_comment("код", file_name, uid)
            await msg.answer_document(BufferedInputFile(combined.encode("utf-8"), filename=out_name), caption=comment)
            if not is_complete:
                await msg.answer("💡 Скажи «допиши» — продолжу.")
            else:
                await msg.answer("✅ Код готов.")
            return

        # TXT
        if ext == "txt":
            await msg.answer_document(BufferedInputFile(ai_response.encode("utf-8"), filename=f"updated_{safe}"),
                caption=await get_file_comment("файл", file_name, uid))
        # DOCX
        elif ext == "docx":
            from docx import Document
            d = Document()
            for line in ai_response.split("\n"):
                line = line.strip()
                if not line: continue
                if len(line) < 80 and not line.endswith(".") and not line.startswith("-"): d.add_heading(line, level=1)
                else: d.add_paragraph(line)
            p = "updated.docx"; d.save(p)
            await msg.answer_document(FSInputFile(p, filename=f"updated_{safe}"),
                caption=await get_file_comment("обновлённый документ", file_name, uid))
        # PPTX
        elif ext == "pptx":
            slides = parse_json_safe(ai_response)
            if not slides:
                slides = []
                for chunk in ai_response.split("---"):
                    lines = [l.strip() for l in chunk.strip().split("\n") if l.strip()]
                    if lines: slides.append({"title": lines[0], "points": [l.lstrip("- ").strip() for l in lines[1:]], "image_prompt": lines[0], "layout": "right_image"})
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
                tx = slide.shapes.add_textbox(Inches(0.5), Inches(1.5), Inches(5), Inches(5.5))
                tf = tx.text_frame; tf.word_wrap = True
                for j, p in enumerate(points):
                    para = tf.paragraphs[0] if j == 0 else tf.add_paragraph(); para.text = f"• {p}"; para.font.size = Pt(14)
                if img:
                    img.seek(0)
                    try: slide.shapes.add_picture(img, Inches(5.5), Inches(1.5), width=Inches(4), height=Inches(5.5))
                    except: pass
            p = "updated.pptx"; prs.save(p)
            await msg.answer_document(FSInputFile(p, filename=f"updated_{safe}"),
                caption=await get_file_comment("обновлённая презентация", file_name, uid))
        elif ext == "pdf":
            await msg.answer(f"📄 PDF не пересобираю, текст:\n\n{ai_response[:3500]}")
    except Exception as e:
        logging.error(f"Doc edit: {e}"); await msg.answer(f"❌ {str(e)[:200]}")


# --- Основной обработчик ---
@dp.message()
async def chat(msg: types.Message):
    uid = msg.from_user.id
    mode = await get_user_mode(uid)
    history = await get_history(uid)
    user_content = None; save_text = None
    is_doc_edit = False; doc_info = None

    if msg.photo:
        photo = msg.photo[-1]; f = await bot.get_file(photo.file_id); d = await bot.download_file(f.file_path)
        b64 = base64.b64encode(d.read()).decode("utf-8")
        user_content = [{"type": "text", "text": msg.caption or "Что на фото?"}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]
        save_text = msg.caption or "[Фото]"
    elif msg.voice:
        await bot.send_chat_action(msg.chat.id, "typing")
        try: t = await transcribe_audio(msg.voice.file_id, "ogg")
        except Exception as e: await msg.answer(f"❌ {str(e)[:150]}"); return
        user_content = t; save_text = f"[Голос]: {t}"
    elif msg.audio:
        await bot.send_chat_action(msg.chat.id, "typing")
        ext = "mp3"
        if msg.audio.file_name and "." in msg.audio.file_name: ext = msg.audio.file_name.rsplit(".", 1)[-1].lower()
        try: t = await transcribe_audio(msg.audio.file_id, ext)
        except Exception as e: await msg.answer(f"❌ {str(e)[:150]}"); return
        user_content = t; save_text = f"[Аудио]: {t}"
    elif msg.document:
        doc = msg.document; fname = doc.file_name or "file"
        ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else ""
        allowed = ["txt", "docx", "pptx", "pdf"] + CODE_EXTENSIONS
        if ext not in allowed: await msg.answer(f"❌ Поддерживаются: {', '.join(allowed)}"); return
        await bot.send_chat_action(msg.chat.id, "typing")
        try:
            text = await read_document(doc.file_id, fname)
            if not text.strip(): await msg.answer("❌ Файл пустой."); return
            doc_info = (await download_file_bytes(doc.file_id), fname, ext)
            caption = msg.caption or ("Допиши код с того места, где остановился" if ext in CODE_EXTENSIONS else "Что сделать?")
            user_content = f"[Файл: {fname}]\n\n{text[:8000]}\n\nЗапрос: {caption}"
            save_text = f"[Документ {fname}]: {caption}"
            is_doc_edit = True
        except Exception as e: await msg.answer(f"❌ {str(e)[:150]}"); return
    elif msg.text:
        user_content = msg.text; save_text = msg.text
    else: return

    await save_message(uid, "user", save_text)
    history.append({"role": "user", "content": user_content})
    history = trim_history_by_chars(history)
    await bot.send_chat_action(msg.chat.id, "typing")

    # Время
    now = datetime.now().strftime("%H:%M МСК, %d.%m.%Y")

    personal = PERSONAL_PROMPTS.get(uid, "")
    tone = await get_user_tone(uid)
    tone_add = f"\n\nТОН: {tone}" if tone else ""
    full_prompt = f"[Сейчас: {now}]\n\n" + SYSTEM_PROMPT + personal + tone_add

    try:
        r = await client.chat.completions.create(model="qwen/qwen3.8-27b",
            messages=[{"role": "system", "content": full_prompt}, *history],
            temperature=0.7, max_tokens=1200)
        answer = r.choices[0].message.content
        await save_message(uid, "assistant", answer)

        if is_doc_edit and doc_info:
            fb, fn, ext = doc_info
            await handle_document_edit(msg, answer, fb, fn, ext, uid)
        else:
            use_file = (mode == "file") or (len(answer) > 4000)
            if use_file:
                ext_out = detect_extension(answer)
                fname_out = f"light_answer.{ext_out}"
                code = extract_code(answer) if ext_out in CODE_EXTENSIONS else answer
                comment = await get_file_comment(f"файл .{ext_out}", "код", uid)
                await msg.answer_document(BufferedInputFile(code.encode("utf-8"), filename=fname_out), caption=comment)
                if ext_out in CODE_EXTENSIONS and not is_code_complete(answer):
                    await msg.answer("💡 Скажи «допиши» (скинь файл) — продолжу.")
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
    db_pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=5)
    await init_db()
    logging.info("База данных подключена")
    if hf_client: logging.info("HF клиент ок")
    if pexafy_client: logging.info("Pexafy клиент ок")
    app = web.Application(); app.router.add_get("/", handle)
    runner = web.AppRunner(app); await runner.setup()
    port = int(os.environ.get("PORT", 8080))
    site = web.TCPSite(runner, "0.0.0.0", port); await site.start()
    logging.info(f"Web server on port {port}")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
