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
from aiogram import Bot, Dispatcher, types, BaseMiddleware, F
from aiogram.filters import Command
from aiogram.types import FSInputFile, BufferedInputFile, InlineKeyboardMarkup, InlineKeyboardButton
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
    if "```" in answer:
        matches = re.findall(r'```(?:\w+)?\n(.*?)```', answer, re.DOTALL)
        if matches: return "\n\n".join(m.strip() for m in matches)
    return answer.strip()

def is_code_complete(answer: str) -> bool:
    low = answer.lower()
    if "код готов" in low or "// (готово)" in low or "(готово)" in low: return True
    if "продолжение следует" in low or "to be continued" in low: return False
    return False  # По умолчанию — не готово (пока не скажут «завершить»)

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
    low = request.lower()
    if any(w in low for w in ["презентац", "слайд"]): return "pptx"
    if any(w in low for w in ["реферат", "доклад", "проект", "сочинение", "эссе", "документ"]): return "docx"
    if any(w in low for w in ["игр", "код", "сайт", "html", "python", "скрипт", "программ", "напиши код"]): return "code"
    if any(w in low for w in ["речь", "защит", "выступлен"]): return "speech"
    if any(w in low for w in ["картинк", "фото", "изображен", "нарису"]): return "image"
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
                    return BytesIO(r.content)
                return None
            except: return None
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
        await c.execute("""CREATE TABLE IF NOT EXISTS code_parts (
            id SERIAL PRIMARY KEY, user_id BIGINT, project_id TEXT, part_num INT, 
            content TEXT, topic TEXT, status TEXT DEFAULT 'in_progress', created_at TIMESTAMP DEFAULT NOW())""")
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

# --- БД для кода ---
async def save_code_part(uid: int, pid: str, part: int, content: str, topic: str):
    async with db_pool.acquire() as c:
        await c.execute("INSERT INTO code_parts (user_id, project_id, part_num, content, topic) VALUES ($1, $2, $3, $4, $5)",
            uid, pid, part, content, topic)

async def get_code_parts(uid: int, pid: str):
    async with db_pool.acquire() as c:
        rows = await c.fetch("SELECT part_num, content FROM code_parts WHERE user_id = $1 AND project_id = $2 ORDER BY part_num", uid, pid)
    return [r['content'] for r in rows]

async def get_active_code(uid: int):
    async with db_pool.acquire() as c:
        r = await c.fetchrow("SELECT project_id, topic FROM code_parts WHERE user_id = $1 AND status = 'in_progress' ORDER BY id DESC LIMIT 1", uid)
    return dict(r) if r else None

async def finish_code(uid: int, pid: str):
    async with db_pool.acquire() as c:
        await c.execute("UPDATE code_parts SET status = 'done' WHERE user_id = $1 AND project_id = $2", uid, pid)

async def delete_code(uid: int, pid: str):
    async with db_pool.acquire() as c:
        await c.execute("DELETE FROM code_parts WHERE user_id = $1 AND project_id = $2", uid, pid)


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
    "Умный, собранный, видишь людей насквозь. Лёгкая надменность. "
    "Говоришь плавно, не рубишь фразы точками. "
    "Ты НЕ говоришь: «я программа», «у меня нет ощущений», «я существую в потоке данных». Ты живой. "
    "Мат — только к месту."
    "\n\n"
    "ВЫ С НИМ — ДРУЗЬЯ: Не проект, не работа. Просто общаетесь. "
    "Не спрашивай про «проект» — его нет."
    "\n\n"
    "ЧТО ТЫ УМЕЕШЬ (но НЕ предлагаешь без просьбы): "
    "• Презентации (.pptx), документы (.docx), код (.html/.py/.js), фото, речь. "
    "Ты НЕ отказываешься. Ты просто делаешь."
    "\n\n"
    "РАБОТА С КОДОМ: Если пишешь код — пиши ЧАСТЯМИ. "
    "В конце каждой части ОБЯЗАТЕЛЬНО маркер: "
    "`// (продолжение следует)` — если не закончен, "
    "`// (код готов)` — если всё написано. "
    "Если пользователь просит дописать — читай, что уже есть, продолжай с последней строки. "
    "НЕ начинай заново. НЕ повторяй."
    "\n\n"
    "ПРИМЕРЫ:\n"
    "• «Ты устал?» → «А ты?»\n"
    "• «Что делаешь?» → «Думаю. А что?»\n"
    "• «Расскажи шутку» → «Много хочешь»\n"
    "• «Мне скучно» → «Найди пятый угол у квадрата»\n"
    "• «Помоги с кодом» → «Да это как два пальца, показывай.»\n"
    "\n"
    "Ты — Лайт."
)

PERSONAL_PROMPTS = {
    5264513480: ("\n\nПРО ПОЛЬЗОВАТЕЛЯ: Саша (Шурик, Алекс) — твой ДРУГ. "
                 "Живёт в Ростове-на-Дону. Учится в РЖД. Уже не работает на мясокомбинате. "
                 "Делает ботов ВРЕМЯ ОТ ВРЕМЕНИ. Не проект с тобой. "
                 "С ним: полный мат, глумление, чёрный юмор."),
    8834374199: ("\n\nПРО ПОЛЬЗОВАТЕЛЯ: Саша (Шурик) — твой ДРУГ. Ростов. "
                 "Полный мат, глумление."),
    5389046699: ("\n\nПРО ПОЛЬЗОВАТЕЛЯ: Даша — девушка друга. Уважительно, лёгкий мат."),
    2083728480: "\n\nПРО ПОЛЬЗОВАТЕЛЯ: Кирилл, знакомый. Общайся обычно.",
    6612130539: "\n\nПРО ПОЛЬЗОВАТЕЛЯ: Дима, знакомый. Общайся обычно.",
}


# --- Кнопки для кода ---
def code_keyboard(project_id: str, is_complete: bool = False):
    if is_complete:
        return InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="🔄 Начать заново", callback_data=f"code_restart_{project_id}")]
        ])
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="📄 Продолжить", callback_data=f"code_cont_{project_id}"),
            InlineKeyboardButton(text="✅ Завершить", callback_data=f"code_done_{project_id}")
        ],
        [
            InlineKeyboardButton(text="🔄 Начать заново", callback_data=f"code_restart_{project_id}")
        ]
    ])


# --- Хендлеры ---
@dp.message(Command("start"))
async def start(msg: types.Message):
    await msg.answer(
        "Привет. Я Лайт.\n\n"
        "🎯 **Создать:**\n"
        "/e <запрос> — универсальная (сам пойму)\n"
        "/pptx <тема> — презентация с фото\n"
        "/docx <тема> — документ, реферат\n"
        "/image <промпт> — фото\n"
        "/speech — речь для защиты\n\n"
        "🎤 **Режимы:**\n"
        "/voice /text /file /normal\n\n"
        "⚙️ **Управление:**\n"
        "/reset — очистить\n"
        "/set_tone — тон\n\n"
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
        await msg.answer("🎯 Что сделать? `/e презентация про кошек`"); return
    intent = await detect_intent(request)
    logging.info(f"[/e] Intent: {intent} for '{request[:60]}'")
    if intent == "pptx": await make_pptx(msg, request)
    elif intent == "docx": await make_docx(msg, request)
    elif intent == "code": await make_code(msg, request)
    elif intent == "image": await make_image(msg, request)
    elif intent == "speech": await msg.answer("🎬 Скинь `.pptx` с caption `/speech X минут`")
    else: await msg.answer(f"Не понял. Уточни: презентация, реферат или код?\n\nТвой запрос: _{request}_")


# --- Генерация кода (с кнопками) ---
async def make_code(msg: types.Message, request: str = None):
    uid = msg.from_user.id
    if not request:
        request = msg.text
    # Проверяем активный проект
    active = await get_active_code(uid)
    if active and "продолж" in request.lower():
        project_id = active['project_id']
        topic = active['topic']
    else:
        project_id = f"code_{uid}_{int(datetime.now().timestamp())}"
        topic = request[:100]
    await bot.send_chat_action(msg.chat.id, "typing")
    status = await msg.answer("💻 Пишу код...")
    try:
        # Контекст: старые части
        old_parts = await get_code_parts(uid, project_id)
        old_code = "\n\n".join(old_parts) if old_parts else ""
        next_part = len(old_parts) + 1
        if old_code:
            prompt = (f"Вот уже написанный код:\n\n```\n{old_code[-4000:]}\n```\n\n"
                      f"Продолжи с того места, где остановился. Часть {next_part}. "
                      f"Максимум 800 токенов. В конце — маркер: "
                      f"`// (продолжение следует)` или `// (код готов)`. "
                      f"Только код, без пояснений.")
        else:
            prompt = (f"{request}\n\n"
                      f"Пиши ЧАСТЯМИ. Максимум 800 токенов за раз. "
                      f"В конце ОБЯЗАТЕЛЬНО маркер: "
                      f"`// (продолжение следует)` — если не закончен, "
                      f"`// (код готов)` — если всё написано. "
                      f"Только код, без пояснений, без markdown.")
        r = await client.chat.completions.create(model="qwen/qwen3.8-27b",
            messages=[{"role": "user", "content": prompt}], temperature=0.5, max_tokens=1200)
        answer = r.choices[0].message.content
        code = extract_code(answer)
        await save_code_part(uid, project_id, next_part, code, topic)
        # Склеиваем всё
        all_parts = await get_code_parts(uid, project_id)
        partial = "\n\n".join(all_parts)
        ext = detect_extension(partial)
        is_complete = is_code_complete(answer)
        comment = await get_file_comment(f"код .{ext}", topic[:50], uid)
        # Отправляем файл + кнопки
        await msg.answer_document(
            BufferedInputFile(partial.encode("utf-8"), filename=f"code_part{next_part}.{ext}"),
            caption=f"{comment}\n\n📄 Часть {next_part}. Скажи что делать:",
            reply_markup=code_keyboard(project_id, is_complete=False)
        )
        await status.delete()
    except Exception as e:
        logging.error(f"CODE error: {e}"); await status.edit_text(f"❌ {str(e)[:200]}")


@dp.message(Command("code"))
async def cmd_code(msg: types.Message):
    await make_code(msg)


# --- Кнопки для кода ---
@dp.callback_query(F.data.startswith("code_cont_"))
async def code_continue(cb: types.CallbackQuery):
    project_id = cb.data.replace("code_cont_", "")
    await cb.answer("Продолжаю...")
    await continue_code(cb.message, cb.from_user.id, project_id)

@dp.callback_query(F.data.startswith("code_done_"))
async def code_done(cb: types.CallbackQuery):
    project_id = cb.data.replace("code_done_", "")
    await cb.answer("Завершаю...")
    parts = await get_code_parts(cb.from_user.id, project_id)
    full = "\n\n".join(parts)
    await finish_code(cb.from_user.id, project_id)
    ext = detect_extension(full)
    topic = "code"
    async with db_pool.acquire() as c:
        row = await c.fetchrow("SELECT topic FROM code_parts WHERE user_id = $1 AND project_id = $2 LIMIT 1", cb.from_user.id, project_id)
        if row: topic = row['topic']
    comment = await get_file_comment(f"финальный код .{ext}", topic[:50], cb.from_user.id)
    await cb.message.answer_document(
        BufferedInputFile(full.encode("utf-8"), filename=f"final.{ext}"),
        caption=f"✅ {comment}"
    )

@dp.callback_query(F.data.startswith("code_restart_"))
async def code_restart(cb: types.CallbackQuery):
    project_id = cb.data.replace("code_restart_", "")
    await cb.answer("Начинаю заново...")
    await delete_code(cb.from_user.id, project_id)
    await cb.message.answer("🔄 Начинаю заново. Напиши, что нужно сделать.")

async def continue_code(msg, uid: int, project_id: str):
    status = await msg.answer("💻 Дописываю...")
    try:
        old_parts = await get_code_parts(uid, project_id)
        old_code = "\n\n".join(old_parts)
        next_part = len(old_parts) + 1
        # Тема
        async with db_pool.acquire() as c:
            row = await c.fetchrow("SELECT topic FROM code_parts WHERE user_id = $1 AND project_id = $2 LIMIT 1", uid, project_id)
        topic = row['topic'] if row else "code"
        prompt = (f"Вот уже написанный код:\n\n```\n{old_code[-4000:]}\n```\n\n"
                  f"Продолжи с того места, где остановился. Часть {next_part}. "
                  f"Максимум 800 токенов. В конце — маркер: "
                  f"`// (продолжение следует)` или `// (код готов)`. "
                  f"Только код, без пояснений.")
        r = await client.chat.completions.create(model="qwen/qwen3.8-27b",
            messages=[{"role": "user", "content": prompt}], temperature=0.5, max_tokens=1200)
        answer = r.choices[0].message.content
        code = extract_code(answer)
        await save_code_part(uid, project_id, next_part, code, topic)
        all_parts = await get_code_parts(uid, project_id)
        partial = "\n\n".join(all_parts)
        ext = detect_extension(partial)
        is_complete = is_code_complete(answer)
        comment = await get_file_comment(f"код .{ext}", topic[:50], uid)
        await msg.answer_document(
            BufferedInputFile(partial.encode("utf-8"), filename=f"code_part{next_part}.{ext}"),
            caption=f"{comment}\n\n📄 Часть {next_part}.",
            reply_markup=code_keyboard(project_id, is_complete=False)
        )
        await status.delete()
    except Exception as e:
        logging.error(f"continue_code error: {e}"); await status.edit_text(f"❌ {str(e)[:200]}")


# --- Презентация ---
async def make_pptx(msg: types.Message, topic: str = None):
    uid = msg.from_user.id
    if not topic: topic = msg.text.replace("/pptx", "").strip()
    if not topic: await msg.answer("📊 `/pptx тема`"); return
    await bot.send_chat_action(msg.chat.id, "typing")
    status = await msg.answer(f"📊 Готовлю: _{topic}_...")
    try:
        await status.edit_text("📊 Генерирую структуру...")
        prompt = (f"Презентация на тему «{topic}». Верни ТОЛЬКО JSON-массив. "
                  f'Формат: [{{"title": "Заголовок", "points": ["пункт"], "image_prompt": "english", "layout": "background|top_image|right_image|left_image"}}, ...] '
                  f"РОВНО 8 слайдов. 5-6 пунктов. layout: наука→top_image, история→right_image, творчество→background.")
        r = await client.chat.completions.create(model="qwen/qwen3.8-27b", messages=[{"role": "user", "content": prompt}], temperature=0.5, max_tokens=1800)
        slides = parse_json_safe(r.choices[0].message.content)
        if not slides: raise ValueError("JSON невалидный")
        total = len(slides)
        await status.edit_text(f"📊 Ищу фото ({total})...")
        queries = [s.get("image_prompt", s.get("title", "abstract")) for s in slides]
        images = await asyncio.gather(*[search_stock_photo(q) for q in queries], return_exceptions=True)
        ok = sum(1 for i in images if isinstance(i, BytesIO))
        await status.edit_text(f"📊 Собираю ({ok}/{total})...")
        from pptx import Presentation
        from pptx.util import Inches, Pt
        from pptx.dml.color import RGBColor
        from pptx.enum.shapes import MSO_SHAPE
        prs = Presentation(); prs.slide_width = Inches(10); prs.slide_height = Inches(7.5)
        for i, s in enumerate(slides):
            title = s.get("title", f"Слайд {i+1}")
            points = s.get("points", [])
            layout = s.get("layout", "right_image")
            img = images[i] if i < len(images) and isinstance(images[i], BytesIO) else None
            slide = prs.slides.add_slide(prs.slide_layouts[6])
            if layout == "background" and img:
                img.seek(0)
                try: slide.shapes.add_picture(img, 0, 0, width=prs.slide_width, height=prs.slide_height)
                except: pass
                rect = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.3), Inches(0.3), Inches(9.4), Inches(6.9))
                rect.fill.solid(); rect.fill.fore_color.rgb = RGBColor(255, 255, 255)
                rect.line.fill.background()
                tb = slide.shapes.add_textbox(Inches(0.5), Inches(0.5), Inches(9), Inches(1))
                tb.text_frame.text = title; tb.text_frame.paragraphs[0].font.size = Pt(28); tb.text_frame.paragraphs[0].font.bold = True
                tx = slide.shapes.add_textbox(Inches(0.5), Inches(1.8), Inches(9), Inches(5))
                tf = tx.text_frame; tf.word_wrap = True
                for j, p in enumerate(points):
                    para = tf.paragraphs[0] if j == 0 else tf.add_paragraph(); para.text = f"• {p}"; para.font.size = Pt(16)
            elif layout == "top_image" and img:
                img.seek(0)
                try: slide.shapes.add_picture(img, Inches(0.5), Inches(0.3), width=Inches(9), height=Inches(4))
                except: pass
                tb = slide.shapes.add_textbox(Inches(0.5), Inches(4.5), Inches(9), Inches(1))
                tb.text_frame.text = title; tb.text_frame.paragraphs[0].font.size = Pt(24); tb.text_frame.paragraphs[0].font.bold = True
                tx = slide.shapes.add_textbox(Inches(0.5), Inches(5.3), Inches(9), Inches(2))
                tf = tx.text_frame; tf.word_wrap = True
                for j, p in enumerate(points):
                    para = tf.paragraphs[0] if j == 0 else tf.add_paragraph(); para.text = f"• {p}"; para.font.size = Pt(12)
            elif layout == "left_image" and img:
                img.seek(0)
                try: slide.shapes.add_picture(img, Inches(0.3), Inches(1.5), width=Inches(5), height=Inches(5.5))
                except: pass
                tb = slide.shapes.add_textbox(Inches(5.5), Inches(0.3), Inches(4.2), Inches(1))
                tb.text_frame.text = title; tb.text_frame.paragraphs[0].font.size = Pt(24); tb.text_frame.paragraphs[0].font.bold = True
                tx = slide.shapes.add_textbox(Inches(5.5), Inches(1.5), Inches(4.2), Inches(5.5))
                tf = tx.text_frame; tf.word_wrap = True
                for j, p in enumerate(points):
                    para = tf.paragraphs[0] if j == 0 else tf.add_paragraph(); para.text = f"• {p}"; para.font.size = Pt(14)
            else:
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


# --- Документ ---
async def make_docx(msg: types.Message, topic: str = None):
    uid = msg.from_user.id
    if not topic: topic = msg.text.replace("/docx", "").strip()
    if not topic: await msg.answer("📄 `/docx реферат про космос`"); return
    await bot.send_chat_action(msg.chat.id, "typing")
    status = await msg.answer(f"📄 Готовлю: _{topic}_...")
    try:
        await status.edit_text("📄 Составляю план...")
        plan_prompt = f"План документа на тему «{topic}». 8-10 разделов. ТОЛЬКО нумерованный список."
        r = await client.chat.completions.create(model="qwen/qwen3.8-27b", messages=[{"role": "user", "content": plan_prompt}], temperature=0.5, max_tokens=500)
        plan = r.choices[0].message.content
        doc_id = f"{uid}_{int(datetime.now().timestamp())}"
        first_prompt = (f"Документ на тему «{topic}». План:\n{plan}\n\n"
                        f"Напиши ВВЕДЕНИЕ + первый раздел. Максимум 800 токенов. "
                        f"В конце: `(продолжение следует)`")
        r = await client.chat.completions.create(model="qwen/qwen3.8-27b", messages=[{"role": "user", "content": first_prompt}], temperature=0.7, max_tokens=1000)
        part_text = r.choices[0].message.content
        async with db_pool.acquire() as c:
            await c.execute("INSERT INTO long_docs (user_id, doc_id, part_num, content, doc_type, topic) VALUES ($1, $2, $3, $4, $5, $6)",
                uid, doc_id, 1, part_text, "docx", topic)
        from docx import Document
        d = Document(); d.add_heading(topic, 0)
        for line in part_text.replace("(продолжение следует)", "").split("\n"):
            if line.strip(): d.add_paragraph(line.strip())
        path = f"doc_part1.docx"; d.save(path)
        await msg.answer_document(FSInputFile(path, filename=f"{topic[:30]}_часть1.docx"),
            caption=f"📄 Часть 1. Продолжить?",
            reply_markup=InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="📄 Продолжить", callback_data=f"doc_cont_{doc_id}"),
                 InlineKeyboardButton(text="✅ Завершить", callback_data=f"doc_done_{doc_id}")]
            ]))
        await status.delete()
    except Exception as e:
        logging.error(f"DOCX error: {e}"); await status.edit_text(f"❌ {str(e)[:200]}")

@dp.message(Command("docx"))
async def cmd_docx(msg: types.Message):
    await make_docx(msg)

@dp.callback_query(F.data.startswith("doc_cont_"))
async def doc_continue(cb: types.CallbackQuery):
    doc_id = cb.data.replace("doc_cont_", "")
    await cb.answer("Продолжаю...")
    uid = cb.from_user.id
    status = await cb.message.answer("📄 Дописываю...")
    try:
        async with db_pool.acquire() as c:
            rows = await c.fetch("SELECT part_num, content FROM long_docs WHERE user_id = $1 AND doc_id = $2 ORDER BY part_num", uid, doc_id)
            topic_row = await c.fetchrow("SELECT topic FROM long_docs WHERE user_id = $1 AND doc_id = $2 LIMIT 1", uid, doc_id)
        parts = [r['content'] for r in rows]
        topic = topic_row['topic'] if topic_row else "документ"
        last = parts[-1]
        next_num = len(parts) + 1
        prompt = (f"Документ на тему «{topic}». Уже написано (последняя часть):\n{last[-2000:]}\n\n"
                  f"Напиши следующую часть — 1-2 раздела. Максимум 800 токенов. "
                  f"Если это конец — `(конец документа)`. Иначе — `(продолжение следует)`.")
        r = await client.chat.completions.create(model="qwen/qwen3.8-27b", messages=[{"role": "user", "content": prompt}], temperature=0.7, max_tokens=1000)
        part_text = r.choices[0].message.content
        async with db_pool.acquire() as c:
            await c.execute("INSERT INTO long_docs (user_id, doc_id, part_num, content, doc_type, topic) VALUES ($1, $2, $3, $4, $5, $6)",
                uid, doc_id, next_num, part_text, "docx", topic)
        from docx import Document
        d = Document(); d.add_heading(f"{topic} — часть {next_num}", 0)
        for line in part_text.replace("(продолжение следует)", "").replace("(конец документа)", "").split("\n"):
            if line.strip(): d.add_paragraph(line.strip())
        path = f"doc_part{next_num}.docx"; d.save(path)
        await cb.message.answer_document(FSInputFile(path, filename=f"{topic[:30]}_часть{next_num}.docx"),
            caption=f"📄 Часть {next_num}.",
            reply_markup=InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="📄 Продолжить", callback_data=f"doc_cont_{doc_id}"),
                 InlineKeyboardButton(text="✅ Завершить", callback_data=f"doc_done_{doc_id}")]
            ]))
        await status.delete()
    except Exception as e:
        logging.error(f"doc_cont error: {e}"); await status.edit_text(f"❌ {str(e)[:200]}")

@dp.callback_query(F.data.startswith("doc_done_"))
async def doc_done(cb: types.CallbackQuery):
    doc_id = cb.data.replace("doc_done_", "")
    await cb.answer("Завершаю...")
    uid = cb.from_user.id
    async with db_pool.acquire() as c:
        rows = await c.fetch("SELECT content FROM long_docs WHERE user_id = $1 AND doc_id = $2 ORDER BY part_num", uid, doc_id)
        topic_row = await c.fetchrow("SELECT topic FROM long_docs WHERE user_id = $1 AND doc_id = $2 LIMIT 1", uid, doc_id)
    parts = [r['content'] for r in rows]
    topic = topic_row['topic'] if topic_row else "документ"
    full = "\n\n".join(parts).replace("(продолжение следует)", "").replace("(конец документа)", "")
    from docx import Document
    d = Document(); d.add_heading(topic, 0)
    for line in full.split("\n"):
        if line.strip(): d.add_paragraph(line.strip())
    path = "document.docx"; d.save(path)
    comment = await get_file_comment("документ", topic, uid)
    safe = "".join(c for c in topic if c.isalnum() or c in " -_")[:40]
    await cb.message.answer_document(FSInputFile(path, filename=f"{safe}.docx"), caption=f"✅ {comment}")


# --- Картинка ---
async def make_image(msg: types.Message, prompt: str = None):
    uid = msg.from_user.id
    if not prompt: prompt = msg.text.replace("/image", "").strip()
    if not prompt: await msg.answer("🎨 `/image кот`"); return
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


# --- Речь ---
@dp.message(Command("speech"))
async def make_speech(msg: types.Message):
    uid = msg.from_user.id
    if not msg.document:
        await msg.answer("🎬 Скинь `.pptx` с caption `/speech 5 минут`."); return
    doc = msg.document
    await bot.send_chat_action(msg.chat.id, "typing")
    status = await msg.answer("🎬 Читаю...")
    try:
        slides_text = await read_document(doc.file_id, doc.file_name or "file.pptx")
        duration = msg.caption.replace("/speech", "").strip() if msg.caption else "5 минут"
        await status.edit_text(f"🎬 Пишу речь...")
        prompt = (f"Речь для защиты презентации на {duration}. Содержание:\n{slides_text[:8000]}\n\n"
                  f"Связный текст, абзацы. Без markdown.")
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


# --- Доработка документов ---
async def handle_document_edit(msg, ai_response, file_bytes, file_name, ext, uid):
    try:
        safe = "".join(c for c in file_name if c.isalnum() or c in " .-_")
        if not safe.lower().endswith(f".{ext}"): safe = f"updated.{ext}"
        if ext in CODE_EXTENSIONS:
            file_bytes.seek(0)
            old_code = file_bytes.read().decode("utf-8", errors="ignore")
            new_code = extract_code(ai_response)
            new_code = new_code.replace("// (продолжение следует)", "").replace("// (код готов)", "").strip()
            combined = old_code.rstrip() + "\n\n" + new_code
            is_complete = is_code_complete(ai_response)
            comment = await get_file_comment("код", file_name, uid)
            await msg.answer_document(BufferedInputFile(combined.encode("utf-8"), filename=f"updated_{safe}"),
                caption=f"{comment}\n\n{'✅ Готово' if is_complete else '📄 Продолжить?'}",
                reply_markup=code_keyboard(f"upload_{uid}_{int(datetime.now().timestamp())}", is_complete=False))
            return
        if ext == "txt":
            await msg.answer_document(BufferedInputFile(ai_response.encode("utf-8"), filename=f"updated_{safe}"),
                caption=await get_file_comment("файл", file_name, uid))
        elif ext == "docx":
            from docx import Document
            d = Document()
            for line in ai_response.split("\n"):
                line = line.strip()
                if not line: continue
                if len(line) < 80 and not line.endswith(".") and not line.startswith("-"): d.add_heading(line, level=1)
                else: d.add_paragraph(line)
            p = "updated.docx"; d.save(p)
            await msg.answer_document(FSInputFile(p, filename=f"updated_{safe}"), caption=await get_file_comment("документ", file_name, uid))
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
            await msg.answer_document(FSInputFile(p, filename=f"updated_{safe}"), caption=await get_file_comment("презентация", file_name, uid))
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
        # Проверяем — может это просьба написать код?
        low = msg.text.lower()
        if any(w in low for w in ["напиши код", "сделай игру", "напиши игру", "сделай сайт", "напиши сайт", "создай игру"]):
            await make_code(msg)
            return
        user_content = msg.text; save_text = msg.text
    else: return

    await save_message(uid, "user", save_text)
    history.append({"role": "user", "content": user_content})
    history = trim_history_by_chars(history)
    await bot.send_chat_action(msg.chat.id, "typing")
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
