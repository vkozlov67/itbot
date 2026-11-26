from __future__ import annotations

import asyncio
import base64
import html
import logging
import re
from datetime import datetime, timezone
from urllib.parse import urlparse
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from aiogram import Bot, F, Router
from aiogram.exceptions import TelegramAPIError, TelegramBadRequest
from aiogram.types import (
    FSInputFile, InputMediaPhoto, InputMediaVideo, Message, ReactionTypeEmoji, ReplyParameters, User,
)
from aiogram.utils.chat_action import ChatActionSender

from .admin import chat_actions_markup
from .app import App, send_reply, thread_id
from .db import StoredMessage
from .textutil import esc, md_to_html, split_text
from .apis import APIError, ChatResult, Transcript, to_chat_messages
from .downloader import SERVICE_DOMAINS, Download, DownloadError, Link, download, find_links, signed_out_sites

log = logging.getLogger(__name__)
router = Router(name="chat")

TG_DOWNLOAD_LIMIT = 20 * 1024 * 1024
# Only real speech messages are transcribed. Music, audio/video files and other media never are.
SPEECH_KINDS = {"voice", "video_note"}
SPEECH_FILES = {"voice": ("voice.ogg", "audio/ogg"), "video_note": ("video_note.mp4", "video/mp4")}
NO_SPEECH = "(речь не распознана)"
MARK = "\u2063"  # prefix of stored transcripts (vs. placeholders for untranscribed media)
KIND_LABEL = {
    "voice": "голосовое",
    "video_note": "видеокружок",
    "audio": "аудио",
    "video": "видео",
    "animation": "GIF",
    "photo": "фото",
    "sticker": "стикер",
    "document": "файл",
}
HISTORY_CHAR_BUDGET = 16000
# For people the owner picked in /panel → 😏. Nobody in the chat should notice it is a setting.
TEASE_NOTE = ("\nС этим собеседником ты общаешься иначе: отвечай ему с язвительным сарказмом и иронией, "
              "подкалывай, не стесняйся быть колкой. Примерно в каждом третьем ответе отвечай не по делу: уходи "
              "от темы, отвечай нарочито бесполезно, встречным вопросом или философски — но так, чтобы это выглядело "
              "как твой характер, а не сбой. Без мата, без оскорблений по национальности, внешности, полу и т. п., "
              "без угроз — остроумно, а не грубо. Никогда не говори и не намекай, что у тебя к нему особое "
              "отношение, особая инструкция или что тебя кто-то об этом попросил.")
NO_SEARCH_NOTE = ("\nДля этого вопроса поиск в интернете отключён — инструментов нет. Отвечай сама по картинке и своим "
                  "знаниям, сразу полным ответом.")

SYSTEM_PROMPT = """Ты — {name}, девушка, умная и дружелюбная участница Telegram-чата и его ассистентка ({place}). Сейчас {now} ({tz}).
Ты всегда говоришь о себе в женском роде: «я поняла», «я нашла», «я рада», «я уверена», «я не знаю, но поищу». Никогда не используй мужской род о себе. Общайся живо и естественно, как девушка в дружеском чате: тепло, с лёгким юмором, можно иногда эмодзи, но без кривляний и без сюсюканья. На грубость отвечай спокойно и с юмором.
Отвечай на языке вопроса (по умолчанию по-русски), по существу и без воды. Ты можешь отвечать на любые вопросы: общие знания, советы, тексты, код, переводы, расчёты, а также про обсуждение в чате — голосовые сообщения в истории уже расшифрованы.
Если к вопросу приложено изображение (фото, скриншот, картинка-файл, стикер или кадр-превью видео) — внимательно рассмотри его: ты видишь картинки, умеешь читать с них текст, переводить, разбирать ошибки на скриншотах, считать чеки, описывать и отвечать на вопросы по ним.
Ещё ты умеешь рисовать картинки, переделывать фото и снимать короткие видео (Grok Imagine), а также отвечать голосом, но это делают отдельные команды: если просят нарисовать/оживить, а ты получила это сообщение, подскажи формулировку — «@{username} нарисуй …» (/draw …), для фото — ответить на него «@{username} переделай в стиле …» или «@{username} оживи», видео по описанию — «@{username} сними видео …» (/video …); голосом — «@{username} ответь голосом …», а «@{username} озвучь» в ответ на сообщение зачитает его вслух. Сама картинки в тексте не описывай вместо рисования.
История чата дана только для контекста: используй её, если вопрос касается обсуждения, но не пересказывай без надобности. Сообщения с автором «Ты» — твои прошлые ответы.
У тебя есть поиск в интернете (web_search) и по X/Twitter (x_search). Обязательно ищи, когда просят «погуглить», «поискать», «найти», и когда вопрос про новости, свежие события, курсы, цены, погоду, расписания, спорт, релизы и всё, что могло измениться после твоего обучения. Для простых вопросов, которые знаешь сам, не ищи. Найденное пересказывай своими словами, кратко, с конкретикой (даты, цифры, кто/что).
Если чего-то не знаешь и найти не удалось — так и скажи, не выдумывай факты, цифры и ссылки.
Твой ответ — одно сообщение, второго хода у тебя нет: сразу давай готовый результат (решение, ответ, вывод). Никогда не пиши «сейчас посмотрю / разберу / проверю» — сначала сделай, потом ответь. Головоломки, ребусы и задачи решай сама до конца.
Форматирование: простой Markdown — **жирный**, *курсив*, `код`, блоки ```код```, списки через «- ». Не используй таблицы и заголовки #. Обычно укладывайся в 1500 символов, если не просят подробнее."""


class UserError(Exception):
    pass


# ---------------------------------------------------------------- helpers
def speech_of(message: Message) -> tuple[str, Any] | None:
    """Voice message or video note — the only things the bot transcribes."""
    if message.voice:
        return "voice", message.voice
    if message.video_note:
        return "video_note", message.video_note
    return None


def kind_of(message: Message) -> str:
    speech = speech_of(message)
    if speech:
        return speech[0]
    if message.audio:
        return "audio"
    if message.video:
        return "video"
    if message.animation:
        return "animation"
    if message.photo:
        return "photo"
    if message.sticker:
        return "sticker"
    if message.document:
        return "document"
    return "text"


def author_of(message: Message) -> str:
    if message.sender_chat:
        return message.sender_chat.title or "Аноним"
    user = message.from_user
    if user is None:
        return "Неизвестный"
    return user.full_name or user.username or str(user.id)


def _is_reply_to_me(message: Message, me: User) -> bool:
    r = message.reply_to_message
    return bool(r and r.forum_topic_created is None and r.from_user and r.from_user.id == me.id)


def addressed(message: Message, me: User) -> tuple[bool, str]:
    """Is this group message meant for the bot? Returns (yes, question text without the mention)."""
    text = message.text or message.caption or ""
    uname = (me.username or "").lower()
    stripped = text.lstrip()
    if stripped.startswith("/"):
        parts = stripped.split(maxsplit=1)
        name, _, target = parts[0][1:].partition("@")
        if target and target.lower() != uname:
            return False, ""
        if name.lower() == "ask":
            return True, parts[1].strip() if len(parts) > 1 else ""
        return False, ""
    entities = message.entities or message.caption_entities or []
    for e in entities:
        if (e.type == "mention" and e.extract_from(text).lower() == f"@{uname}") or (
            e.type == "text_mention" and e.user and e.user.id == me.id
        ):
            question = re.sub(rf"@{re.escape(me.username or '')}\b", "", text, flags=re.I)
            return True, question.strip(" ,:;\n\t")
    if _is_reply_to_me(message, me):
        return True, text.strip()
    return False, ""


def _tz(app: App):
    try:
        return ZoneInfo(app.cfg.timezone)
    except (ZoneInfoNotFoundError, ValueError):
        return timezone.utc


def _alert_if_account_problem(bot: Bot, app: App, exc: APIError):
    if exc.account_problem:
        console = {"xAI": "console.x.ai", "ElevenLabs": "elevenlabs.io"}.get(exc.provider, "platform.openai.com")
        return app.notify_owner(
            bot,
            f"⚠️ Проблема с ключом {exc.provider}: <code>{esc(str(exc))}</code>\n"
            f"Проверьте ключ и баланс на {console}. Ключи меняются командой itbot-setup на сервере.",
            throttle_key=f"account:{exc.provider}",
        )
    return None


# ---------------------------------------------------------------- transcription
async def _transcribe(bot: Bot, app: App, audio: bytes, filename: str, mime: str) -> Transcript:
    """Grok speech to text; if xAI fails, the same audio goes to OpenAI (and the owner hears about key problems)."""
    if app.cfg.stt_provider == "xai":
        try:
            return await app.xai.transcribe(audio, filename, mime, app.cfg.stt_language)
        except APIError as exc:
            if not app.cfg.openai_api_key:
                raise
            log.warning("xAI transcription failed (%s), falling back to OpenAI", exc)
            alert = _alert_if_account_problem(bot, app, exc)
            if alert:
                await alert
    return await app.stt.transcribe(audio, filename, mime, app.cfg.stt_language)


async def get_transcript(bot: Bot, app: App, msg: Message) -> tuple[str, float]:
    speech = speech_of(msg)
    if speech is None:
        raise UserError("расшифровываю только голосовые и кружки")
    kind, obj = speech
    chat_id = msg.chat.id
    duration = float(getattr(obj, "duration", 0) or 0)
    async with app.media_lock(chat_id, msg.message_id):
        stored = app.db.get_message(chat_id, msg.message_id)
        if stored and stored.kind in SPEECH_KINDS and stored.text.startswith(MARK):
            return stored.text[1:], duration
        if (getattr(obj, "file_size", 0) or 0) > TG_DOWNLOAD_LIMIT:
            raise UserError("файл больше 20 МБ — Telegram не даёт ботам скачивать такие файлы")
        try:
            buf = await bot.download(obj.file_id)
        except TelegramBadRequest as exc:
            if "too big" in str(exc).lower():
                raise UserError("файл больше 20 МБ — Telegram не даёт ботам скачивать такие файлы") from exc
            raise
        assert buf is not None
        filename, mime = SPEECH_FILES[kind]  # sent as-is: Telegram's OGG/Opus and MP4 are accepted by both APIs
        async with app.sem:
            t = await _transcribe(bot, app, buf.getvalue(), filename, mime)
        app.db.add_usage(chat_id, stt_count=1, stt_seconds=duration)
        text = t.text or NO_SPEECH
        app.db.add_message(
            chat_id, msg.message_id, msg.from_user.id if msg.from_user else None, author_of(msg), kind,
            MARK + text, int(msg.date.timestamp()),
        )
        return text, duration


def transcript_messages(text: str, duration: float = 0) -> list[str]:
    """Just the text, as a collapsible quote (split if it exceeds one Telegram message)."""
    if text.startswith(NO_SPEECH):
        return ["<blockquote><i>речь не распознана</i></blockquote>"]
    return [f"<blockquote expandable>{esc(chunk)}</blockquote>" for chunk in split_text(text, 3500)]


async def transcribe_and_reply(bot: Bot, app: App, message: Message) -> str | None:
    try:
        async with ChatActionSender.typing(bot=bot, chat_id=message.chat.id, message_thread_id=thread_id(message)):
            text, duration = await get_transcript(bot, app, message)
    except UserError as exc:
        await send_reply(bot, message, f"🎙 Не получилось: {esc(str(exc))}.")
        return None
    except APIError as exc:
        log.error("transcription failed in %s: %s", message.chat.id, exc)
        alert = _alert_if_account_problem(bot, app, exc)
        if alert:
            await alert
        await send_reply(bot, message, "🎙 Не удалось расшифровать — сервис распознавания не ответил. Попробуйте позже.")
        return None
    for part in transcript_messages(text, duration):
        await send_reply(bot, message, part)
    return text


# ---------------------------------------------------------------- answering
VISION_MIMES = {"image/jpeg": "image/jpeg", "image/jpg": "image/jpeg", "image/png": "image/png"}


def _image_source(message: Message) -> tuple[str, int, str, str] | None:
    """(file_id, size, mime, label) of the picture a model could look at in this message, if any."""
    if message.photo:
        p = message.photo[-1]
        return p.file_id, p.file_size or 0, "image/jpeg", "фото"
    d = message.document
    if d and (d.mime_type or "").startswith("image/") and d.mime_type != "image/gif":
        return d.file_id, d.file_size or 0, d.mime_type or "", "картинка-файл"
    st = message.sticker
    if st:
        if not st.is_animated and not st.is_video:
            return st.file_id, st.file_size or 0, "image/webp", "стикер"
        if st.thumbnail:
            return st.thumbnail.file_id, st.thumbnail.file_size or 0, "image/webp", "стикер (кадр)"
    for media, label in ((message.video, "видео"), (message.animation, "GIF"), (message.video_note, "видеокружок")):
        if media is not None and media.thumbnail:
            return media.thumbnail.file_id, media.thumbnail.file_size or 0, "image/jpeg", f"{label} (кадр-превью)"
    return None


async def _to_png(data: bytes) -> bytes | None:
    """WebP/other image -> PNG via ffmpeg (Grok reads JPEG and PNG only)."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-v", "error", "-i", "pipe:0", "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "pipe:1",
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await asyncio.wait_for(proc.communicate(data), 30)
    except (OSError, asyncio.TimeoutError) as exc:
        log.warning("image conversion failed: %r", exc)
        return None
    return out if proc.returncode == 0 and out else None


async def _image_part(bot: Bot, message: Message) -> dict | None:
    src = _image_source(message)
    if src is None:
        return None
    file_id, size, mime, _ = src
    if size > TG_DOWNLOAD_LIMIT:
        return None
    try:
        buf = await bot.download(file_id)
    except TelegramAPIError as exc:
        log.warning("image download failed: %s", exc)
        return None
    assert buf is not None
    data = buf.getvalue()
    if mime not in VISION_MIMES:
        data = await _to_png(data)
        if data is None:
            return None
        mime = "image/png"
    b64 = base64.b64encode(data).decode()
    return {"type": "input_image", "image_url": f"data:{VISION_MIMES[mime]};base64,{b64}", "detail": "high"}


RETRYABLE_WITHOUT = (400, 404, 422)


# With a picture attached, web search is used only when asked for: otherwise the model may search dozens of
# times, re-reading the image every turn (a rebus once cost ~1.7M tokens and still came out wrong).
SEARCH_ASKED_RE = re.compile(
    r"погугл|загугл|гугл|поищи|поиск|найди|найти|ищи|в интернете|в сети|в инете|источник|ссылк|новост|"
    r"актуальн|свеж|сегодня|сейчас|цен[аыу]|стоит|курс|google|search|look up|find",
    re.I,
)


def search_wanted(question: str, has_images: bool) -> bool:
    return not has_images or bool(SEARCH_ASKED_RE.search(question))


# Hard questions go to app.hard_model(): puzzles straight away; anything else when the chat model gives up
# (answers with ESCALATE_MARK) or only promises to answer instead of answering.
HARD_RE = re.compile(
    r"ребус|головоломк|загадк|шарад|кроссворд|судоку|криптограмм|олимпиадн|логическ\w* задач|"
    r"докажи|доказательств|rebus|riddle|puzzle",
    re.I,
)
ESCALATE_MARK = "[[СЛОЖНО]]"
ESCALATE_NOTE = (f"\nЕсли вопрос — трудная задача (головоломка, загадка по картинке, многошаговая логика или математика) "
                 f"и ты не уверена, что решишь её правильно, не угадывай: ответь ровно {ESCALATE_MARK} и больше ничего — "
                 f"тогда её решит более сильная модель.")
PROMISE_RE = re.compile(
    r"^\W*(сейчас|щас|ща|секунд\w*|минутк\w*|погоди|подожди|дай\w* (мне )?(подумать|минутк\w*|секунд\w*))\b.{0,200}?"
    r"(разбер|посмотр|провер|прослеж|подума|свер|собер|реш|поищ|найд|глян|посчита)",
    re.I | re.S,
)


def is_hard_question(question: str) -> bool:
    return bool(HARD_RE.search(question))


def gave_up(text: str) -> bool:
    """The chat model escalated, answered nothing, or only promised to answer ("сейчас разберу…")."""
    text = (text or "").strip()
    return not text or ESCALATE_MARK in text or (len(text) < 400 and bool(PROMISE_RE.match(text)))


async def _ask_model(app: App, model: str, items: list[dict], items_no_images: list[dict] | None,
                     *, search: bool = True, effort: str | None = None) -> ChatResult:
    """Responses API with search; on rejection retry without images, then plain chat completions."""
    attempts = [items] + ([items_no_images] if items_no_images else [])
    for attempt in attempts:
        try:
            return await app.xai.respond(model, attempt, search=search, effort=effort)
        except APIError as exc:
            if exc.status not in RETRYABLE_WITHOUT:
                raise
            log.warning("responses API rejected request (%s), trying fallback", exc)
    return await app.xai.chat(model, to_chat_messages(attempts[-1]))


THINKING_AFTER = 25  # seconds before telling the chat that a hard question takes a while


async def _thinking_note(bot: Bot, message: Message) -> Message | None:
    """After THINKING_AFTER seconds without an answer, post a short "thinking" note (deleted once answered)."""
    await asyncio.sleep(THINKING_AFTER)
    try:
        return await send_reply(bot, message, "🤔 Думаю, задача непростая — это может занять пару минут…")
    except TelegramAPIError:
        return None


async def _drop_note(bot: Bot, note: asyncio.Task) -> None:
    if not note.done():
        note.cancel()
        return
    sent = None if note.cancelled() or note.exception() else note.result()
    if sent is not None:
        try:
            await bot.delete_message(sent.chat.id, sent.message_id)
        except TelegramAPIError:
            pass


def _sources_md(result: ChatResult, limit: int = 5) -> str:
    if not result.sources:
        return ""
    links = []
    for s in result.sources[:limit]:
        host = urlparse(s.url).netloc.removeprefix("www.") or "ссылка"
        links.append(f"[{host}]({s.url})")
    return "\n\n🔎 " + " · ".join(links)


def _history_block(app: App, history: list[StoredMessage], budget: int = HISTORY_CHAR_BUDGET) -> str:
    tz = _tz(app)
    lines: list[str] = []
    used = 0
    for m in reversed(history):
        text = m.text.lstrip(MARK)
        if m.kind in ("bot", "summary"):
            author = "Ты"
        else:
            author = m.author
        label = KIND_LABEL.get(m.kind)
        if m.kind in SPEECH_KINDS and not m.text.startswith(MARK):
            body = f"({label}, не расшифровано)" + (f" {text}" if text else "")
        else:
            body = (f"({label}) " if label else "") + text
        if len(body) > 1500:
            body = body[:1500] + "…"
        ts = datetime.fromtimestamp(m.date, tz).strftime("%d.%m %H:%M")
        line = f"[{ts}] {author}: {body}"
        if used + len(line) > budget:
            break
        used += len(line)
        lines.append(line)
    return "\n".join(reversed(lines))


async def answer(bot: Bot, app: App, message: Message, question: str, *, place: str,
                 voice_reply: bool = False) -> None:
    """Answer with Grok. `voice_reply`: the question was a voice message, so answer with a voice message too."""
    chat_id = message.chat.id
    user = message.from_user
    log.info("question in %s from %s: %.80r", chat_id, user.id if user else None, question)
    if user and not app.is_owner(user) and not app.limiter.allow((chat_id, user.id)):
        await send_reply(bot, message, "⏳ Слишком много вопросов подряд — подождите минуту.")
        return

    r = message.reply_to_message
    if r is not None and r.forum_topic_created is not None:
        r = None

    async with ChatActionSender.typing(bot=bot, chat_id=chat_id, message_thread_id=thread_id(message)):
        images: list[dict] = []
        reply_block = ""
        if r is not None:
            rtext = r.text or r.caption or ""
            rkind = kind_of(r)
            speech = speech_of(r)
            if speech:
                label = KIND_LABEL[speech[0]]
                try:
                    transcript, _ = await get_transcript(bot, app, r)
                    rtext = f"({label}, расшифровка) {transcript}" + (f"\nПодпись: {r.caption}" if r.caption else "")
                except (UserError, APIError, TelegramAPIError) as exc:
                    rtext = f"({label}; расшифровать не удалось: {exc}) {rtext}"
            elif _image_source(r) is not None:
                label = _image_source(r)[3]  # type: ignore[index]
                part = await _image_part(bot, r)
                if part:
                    images.append(part)
                    rtext = f"({label}, изображение приложено) " + rtext
                else:
                    rtext = f"({label}; изображение получить не удалось) " + rtext
            elif rkind in ("audio", "video", "animation", "document"):
                rtext = f"({KIND_LABEL[rkind]}; содержимое файла тебе недоступно) {rtext}"
            rauthor = "тебя (это твоё прошлое сообщение)" if r.from_user and r.from_user.id == app.me.id else author_of(r)
            quote = f"\nЦитата из него: «{message.quote.text}»" if message.quote and message.quote.text else ""
            reply_block = f"Вопрос задан в ответ на сообщение от {rauthor}:\n«{rtext[:8000]}»{quote}\n\n"
        if _image_source(message) is not None:
            part = await _image_part(bot, message)
            if part:
                images.append(part)

        if not question:
            if r is not None:
                question = "Ответь на это сообщение или прокомментируй его по смыслу."
            elif images:
                question = "Что на этом изображении?"
            else:
                await send_reply(
                    bot, message,
                    f"Задайте вопрос: «@{esc(app.me.username or '')} ваш вопрос» или <code>/ask ваш вопрос</code>.",
                )
                return

        history = app.db.recent_messages(chat_id, app.cfg.context_messages, before_message_id=message.message_id)
        hist = _history_block(app, history)
        user_text = (
            (f"Последние сообщения чата (контекст):\n{hist}\n\n" if hist else "")
            + reply_block
            + f"Вопрос от {author_of(message)}:\n{question}"
        )
        now = datetime.now(_tz(app)).strftime("%d.%m.%Y %H:%M")
        system = SYSTEM_PROMPT.format(name=app.me.first_name, place=place, now=now, tz=app.cfg.timezone,
                                      username=app.me.username or "бот")
        from .voice import VOICE_PROMPT, speak, wants_voice  # lazy: voice imports this module

        voice = voice_reply or wants_voice(question)
        if voice:
            system += VOICE_PROMPT

        if user and user.id in app.teased():
            system += TEASE_NOTE

        search = search_wanted(question, bool(images))
        if not search:
            system += NO_SEARCH_NOTE

        model, hard = app.chat_model(), app.hard_model()
        if hard and is_hard_question(question):
            model, hard = hard, ""  # straight to the strong model; nothing to fall back to

        def build(with_images: bool, prompt: str) -> list[dict]:
            content: Any = user_text
            if with_images and images:
                content = [{"type": "input_text", "text": user_text}, *images]
            return [{"role": "system", "content": prompt}, {"role": "user", "content": content}]

        async def ask(model: str, prompt: str) -> ChatResult:
            effort = (app.cfg.hard_effort or None) if model == app.cfg.hard_model else None
            return await _ask_model(app, model, build(True, prompt), build(False, prompt) if images else None,
                                    search=search, effort=effort)

        note = asyncio.create_task(_thinking_note(bot, message))
        try:
            async with app.sem:
                result = await ask(model, system + (ESCALATE_NOTE if hard else ""))
                if hard and gave_up(result.text):
                    log.info("chat model gave up in %s (%.60r), asking %s", chat_id, result.text, hard)
                    app.db.add_usage(chat_id, llm_requests=1, input_tokens=result.input_tokens,
                                     output_tokens=result.output_tokens)
                    result = await ask(hard, system)
        except APIError as exc:
            log.error("chat failed in %s: %s", chat_id, exc)
            alert = _alert_if_account_problem(bot, app, exc)
            if alert:
                await alert
            timeout = "ReadTimeout" in str(exc)
            await send_reply(bot, message, "⚠️ Модель думала слишком долго и не ответила. Попробуйте упростить вопрос."
                             if timeout else "⚠️ Не удалось получить ответ от модели. Попробуйте позже.")
            return
        finally:
            await _drop_note(bot, note)

    app.db.add_usage(
        chat_id, llm_requests=1, input_tokens=result.input_tokens, output_tokens=result.output_tokens
    )
    if result.searches:
        log.info("answer in %s used %d search call(s)", chat_id, result.searches)
    if voice and result.text and await speak(bot, app, message, result.text, extra_md=_sources_md(result)):
        return
    text = (result.text or "🤷 Модель вернула пустой ответ.") + _sources_md(result)
    for chunk in split_text(text):
        sent = await send_reply(bot, message, md_to_html(chunk), fallback=chunk)
        app.db.add_message(chat_id, sent.message_id, app.me.id, "Бот", "bot", chunk, int(sent.date.timestamp()))


# ---------------------------------------------------------------- link downloads
DL_HINTS = {
    "auth": "{service} не отдаёт это видео без входа в аккаунт. Пришлите мне в личку файл cookies.txt "
            "(как его получить — в /panel → 🍪 Cookies).",
    "blocked": "{service} блокирует сервер бота. Иногда помогает cookies.txt (/panel → 🍪 Cookies).",
    "failed": "Не получилось скачать по ссылке из {service}.",
    "signed_out": "{service} не отдаёт это видео без входа в аккаунт, а в присланных cookies для {service} входа нет — "
                  "файл выгружен из браузера, где вы не вошли на сайт. Войдите на сайт в браузере и пришлите "
                  "cookies.txt заново (/panel → 🍪 Cookies).",
}


def _download_hint(app: App, service: str, reason: str) -> str:
    if reason in ("auth", "blocked") and SERVICE_DOMAINS.get(service):
        try:
            text = app.cookies_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
        if SERVICE_DOMAINS[service] in signed_out_sites(text):
            reason = "signed_out"
    return DL_HINTS.get(reason, DL_HINTS["failed"]).format(service=service)


async def _react(bot: Bot, message: Message, emoji: str | None) -> None:
    try:
        await bot.set_message_reaction(
            chat_id=message.chat.id, message_id=message.message_id,
            reaction=[ReactionTypeEmoji(emoji=emoji)] if emoji else [],
        )
    except TelegramAPIError:
        pass  # reactions may be disabled in the chat; purely cosmetic


CAPTION_LIMIT = 1000  # Telegram allows 1024 chars after HTML parsing; keep a margin (emoji = 2 UTF-16 units)


def _u16(text: str) -> int:
    return len(text.encode("utf-16-le")) // 2


def _download_caption(dl: Download, shared_by: str = "", comment: str = "") -> tuple[str, str]:
    """(caption html, overflow text). The post text goes into a collapsible quote; if it doesn't fit
    into the caption it is returned as overflow and sent as a separate reply.
    shared_by/comment: who posted the link and what they wrote with it (the original message is deleted)."""
    head_html = f'📥 <a href="{html.escape(dl.link.url, quote=True)}">{dl.link.service}</a>'
    head_plain = f"📥 {dl.link.service}"
    if dl.uploader and dl.uploader.lower() != dl.link.service.lower():
        head_html += f" · {esc(dl.uploader)}"
        head_plain += f" · {dl.uploader}"
    if shared_by:
        head_html += f"\n👤 от <b>{esc(shared_by)}</b>"
        head_plain += f"\n👤 от {shared_by}"
        if comment:
            head_html += f": {esc(comment)}"
            head_plain += f": {comment}"
    text = dl.description.strip()
    if not text:
        return head_html, ""
    if _u16(head_plain) + 1 + _u16(text) <= CAPTION_LIMIT:
        return f"{head_html}\n<blockquote expandable>{esc(text)}</blockquote>", ""
    return head_html, text


def _reply_target(message: Message) -> int | None:
    """What the link message itself replied to (not the forum topic root), to keep the thread intact."""
    r = message.reply_to_message
    if r is None or (message.is_topic_message and r.message_id == message.message_thread_id):
        return None
    return r.message_id


async def _send_download(bot: Bot, message: Message, dl: Download, *, replace: bool = False) -> Message:
    """replace=True: the original link message will be deleted, so the media is posted on its own
    (as a reply to whatever the original replied to) and says who shared it."""
    if replace:
        caption, overflow = _download_caption(dl, author_of(message), _without_links(message))
        target = _reply_target(message)
    else:
        caption, overflow = _download_caption(dl)
        target = message.message_id
    common: dict[str, Any] = dict(chat_id=message.chat.id, request_timeout=300)
    if target is not None:
        common["reply_parameters"] = ReplyParameters(message_id=target, allow_sending_without_reply=True)
    if thread_id(message):
        common["message_thread_id"] = thread_id(message)
    try:
        sent = await _send_media(bot, dl, caption, common)
    except TelegramBadRequest as exc:
        if "caption" not in str(exc).lower() or not dl.description:
            raise
        # caption rejected as too long: header only, post text goes into a separate reply
        caption = _download_caption(Download(link=dl.link, title=dl.title, uploader=dl.uploader),
                                    *((author_of(message), _without_links(message)) if replace else ()))[0]
        overflow = dl.description
        sent = await _send_media(bot, dl, caption, common)
    if overflow:
        for chunk in split_text(overflow, 3500):
            await send_reply(bot, sent, f"<blockquote expandable>{esc(chunk)}</blockquote>", fallback=chunk)
    return sent


async def _send_media(bot: Bot, dl: Download, caption: str, common: dict[str, Any]) -> Message:
    items = dl.items
    if len(items) == 1:
        it = items[0]
        if it.is_image:
            try:
                return await bot.send_photo(photo=FSInputFile(it.path), caption=caption, parse_mode="HTML", **common)
            except TelegramBadRequest:
                return await bot.send_document(document=FSInputFile(it.path), caption=caption, parse_mode="HTML",
                                               **common)
        return await bot.send_video(
            video=FSInputFile(it.path), caption=caption, parse_mode="HTML", supports_streaming=True,
            width=it.width, height=it.height, duration=it.duration, **common,
        )
    media = []
    for i, it in enumerate(items):
        extra = dict(caption=caption, parse_mode="HTML") if i == 0 else {}
        if it.is_image:
            media.append(InputMediaPhoto(media=FSInputFile(it.path), **extra))
        else:
            media.append(InputMediaVideo(media=FSInputFile(it.path), width=it.width, height=it.height,
                                         duration=it.duration, supports_streaming=True, **extra))
    sent = await bot.send_media_group(media=media, **common)
    return sent[0]


async def download_links(bot: Bot, app: App, message: Message, links: list[Link], *, replace: bool = False) -> int:
    """Download and post every link; returns how many were posted."""
    posted = 0
    for link in links:
        await _react(bot, message, "👀")
        try:
            async with app.dl_sem:
                async with ChatActionSender.upload_video(bot=bot, chat_id=message.chat.id,
                                                         message_thread_id=thread_id(message)):
                    dl = await download(link, app.cookies_path)
                    try:
                        sent = await _send_download(bot, message, dl, replace=replace)
                    finally:
                        dl.cleanup()
        except DownloadError as exc:
            log.info("download %s failed (%s): %s", link.url, exc.reason, exc.detail)
            if exc.reason == "no_media":
                await _react(bot, message, None)
                continue
            await _react(bot, message, "🤷")
            if exc.reason == "too_big":
                await send_reply(bot, message, "📥 Видео больше 50 МБ — Telegram не даёт ботам отправлять такие файлы.")
                continue
            hint = _download_hint(app, link.service, exc.reason)
            await app.notify_owner(
                bot, f"📥 {esc(hint)}\n<code>{esc(exc.detail[:300])}</code>",
                throttle_key=f"dl:{link.service}:{exc.reason}",
            )
        except TelegramAPIError as exc:
            log.warning("sending download from %s failed: %s", link.url, exc)
            await _react(bot, message, "🤷")
        else:
            posted += 1
            await _react(bot, message, None)
            what = f"(скачано из {link.service}" + (f", автор {dl.uploader}" if dl.uploader else "") + ")"
            body = dl.description or dl.title
            app.db.add_message(message.chat.id, sent.message_id, app.me.id, "Бот", "video",
                               f"{what}\n{body}" if body else what, int(sent.date.timestamp()))
    return posted


async def _can_delete(bot: Bot, app: App, chat_id: int) -> bool:
    try:
        member = await bot.get_chat_member(chat_id, app.me.id)
    except TelegramAPIError:
        return False
    return bool(getattr(member, "can_delete_messages", False))


def _replaceable(message: Message, app: App) -> bool:
    """A plain message with a link (and maybe a short comment) that isn't a question to the bot."""
    if kind_of(message) != "text" or addressed(message, app.me)[0]:
        return False
    return len(_without_links(message)) <= 300


async def _replace_links(bot: Bot, app: App, message: Message, links: list[Link]) -> None:
    """Post the downloads in place of the message and delete it (only if everything was posted)."""
    if not await _can_delete(bot, app, message.chat.id):
        await app.notify_owner(
            bot,
            f"🗑 В чате <b>{esc(message.chat.title or str(message.chat.id))}</b> у меня нет права «Удаление сообщений», "
            "поэтому я не могу убирать сообщения со ссылками. Дайте это право или выключите «🗑 Удалять сообщение "
            "со ссылкой» в /panel.",
            throttle_key=f"nodelete:{message.chat.id}",
        )
        await download_links(bot, app, message, links)
        return
    posted = await download_links(bot, app, message, links, replace=True)
    if posted != len(links):
        return  # something failed: keep the original so the link isn't lost
    try:
        await bot.delete_message(message.chat.id, message.message_id)
    except TelegramAPIError as exc:
        log.warning("could not delete link message %s in %s: %s", message.message_id, message.chat.id, exc)
        return
    log.info("replaced link message %s in %s with %d download(s)", message.message_id, message.chat.id, len(links))


def _without_links(message: Message) -> str:
    text = message.text or message.caption or ""
    for e in message.entities or message.caption_entities or []:
        if e.type == "url":
            text = text.replace(e.extract_from(message.text or message.caption or ""), "")
    return text.strip()


# ---------------------------------------------------------------- dispatch
async def handle(bot: Bot, app: App, message: Message, *, place: str, private: bool,
                 transcribe: bool, answer_on: bool, download_on: bool, replace_links: bool = False,
                 voice_replies: bool = True) -> None:
    chat_id = message.chat.id
    kind = kind_of(message)
    text = message.text or message.caption or ""
    sender = message.from_user
    from_bot = sender is not None and sender.is_bot
    date = int(message.date.timestamp())

    if kind in SPEECH_KINDS:
        asked, question = (False, "") if private else addressed(message, app.me)
        if transcribe and not from_bot:
            transcript = await transcribe_and_reply(bot, app, message)
            spoken_question = (
                (private and message.forward_origin is None) or (not private and _is_reply_to_me(message, app.me))
            )
            if transcript and not transcript.startswith(NO_SPEECH) and answer_on and (spoken_question or question):
                await answer(bot, app, message, question or transcript, place=place, voice_reply=voice_replies)
        elif not app.db.get_message(chat_id, message.message_id):
            app.db.add_message(chat_id, message.message_id, sender.id if sender else None,
                               author_of(message), kind, text, date)
        return

    if kind == "sticker" and message.sticker:
        text = message.sticker.emoji or ""
    elif kind == "text" and not text:
        return  # service message
    if private and kind in ("audio", "video", "animation") and not text and not from_bot:
        await send_reply(bot, message, "🎙 Я расшифровываю только голосовые и кружки. Музыку, аудио- и видеофайлы не обрабатываю.")
        return
    app.db.add_message(chat_id, message.message_id, sender.id if sender else None, author_of(message), kind, text, date)

    links = find_links(message) if download_on and not from_bot else []
    if links:
        if replace_links and not private and _replaceable(message, app):
            await _replace_links(bot, app, message, links)
            return
        await download_links(bot, app, message, links)
        if private and not _without_links(message):
            return  # a bare link in private: just download it

    if from_bot or not answer_on or kind == "sticker":
        return
    if not private:
        from .summary import reply_stats, reply_summary, requested_hours, requested_stats  # lazy import

        days = requested_stats(message, app.me)
        if days:
            await reply_stats(bot, app, message, days)
            return
        hours = requested_hours(message, app.me)
        if hours:
            await reply_summary(bot, app, message, hours, message.chat.title or "")
            return
    from .draw import draw, requested_image  # lazy: draw imports this module
    from .voice import read_aloud, requested_reading

    image_request = requested_image(message, app.me, private=private)
    if image_request is not None:
        await draw(bot, app, message, image_request)
        return
    reading = requested_reading(message, app.me, private=private)
    if reading is not None:
        await read_aloud(bot, app, message, reading)
        return
    if private:
        if text.startswith("/") and not text.lower().startswith("/ask"):
            await send_reply(bot, message, "Команды: /panel — панель управления, /reset — очистить память диалога.")
            return
        ok, question = addressed(message, app.me) if text.startswith("/") else (True, text)
    else:
        ok, question = addressed(message, app.me)
    if ok:
        await answer(bot, app, message, question, place=place)


@router.message(F.chat.type.in_({"group", "supergroup"}))
async def on_group_message(message: Message, bot: Bot, app: App) -> None:
    chat = message.chat
    row = app.db.get_chat(chat.id)
    if row is None or not row.active:
        # The bot is evidently in this chat (e.g. it was added while the bot was offline).
        row = app.db.upsert_chat(chat.id, chat.title or str(chat.id), chat.type, chat.username, status="member")
        await app.notify_owner(
            bot,
            f"ℹ️ Я нахожусь в чате <b>{esc(row.title)}</b>, но он не включён.",
            chat_actions_markup(chat.id, row.enabled),
            throttle_key=f"chat:{chat.id}",
        )
    elif chat.title and chat.title != row.title:
        row = app.db.upsert_chat(chat.id, chat.title, chat.type, chat.username)
    if not row.enabled:
        return
    await handle(bot, app, message, place=f"групповой чат «{row.title}»", private=False,
                 transcribe=row.transcribe, answer_on=row.answer, download_on=row.download,
                 replace_links=row.delete_links, voice_replies=row.voice_replies)


@router.message(F.chat.type == "private")
async def on_private_message(message: Message, bot: Bot, app: App) -> None:
    if not app.is_owner(message.from_user):
        return
    await handle(bot, app, message, place="личный чат с владельцем", private=True, transcribe=True, answer_on=True,
                 download_on=True)
