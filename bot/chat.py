from __future__ import annotations

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
from .textutil import esc, fmt_duration, md_to_html, split_text
from .apis import APIError, ChatResult, to_chat_messages
from .downloader import Download, DownloadError, Link, download, find_links

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

SYSTEM_PROMPT = """Ты — {name}, умный ассистент в Telegram ({place}). Сейчас {now} ({tz}).
Отвечай на языке вопроса (по умолчанию по-русски), по существу и без воды. Ты можешь отвечать на любые вопросы: общие знания, советы, тексты, код, переводы, расчёты, а также про обсуждение в чате — голосовые сообщения в истории уже расшифрованы.
История чата дана только для контекста: используй её, если вопрос касается обсуждения, но не пересказывай без надобности. Сообщения с автором «Ты (бот)» — твои прошлые ответы.
У тебя есть поиск в интернете (web_search) и по X/Twitter (x_search). Обязательно ищи, когда просят «погуглить», «поискать», «найти», и когда вопрос про новости, свежие события, курсы, цены, погоду, расписания, спорт, релизы и всё, что могло измениться после твоего обучения. Для простых вопросов, которые знаешь сам, не ищи. Найденное пересказывай своими словами, кратко, с конкретикой (даты, цифры, кто/что).
Если чего-то не знаешь и найти не удалось — так и скажи, не выдумывай факты, цифры и ссылки.
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
        console = "console.x.ai" if exc.provider == "xAI" else "platform.openai.com"
        return app.notify_owner(
            bot,
            f"⚠️ Проблема с ключом {exc.provider}: <code>{esc(str(exc))}</code>\n"
            f"Проверьте ключ и баланс на {console}. Ключи меняются командой itbot-setup на сервере.",
            throttle_key=f"account:{exc.provider}",
        )
    return None


# ---------------------------------------------------------------- transcription
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
        filename, mime = SPEECH_FILES[kind]  # sent as-is: Telegram's OGG/Opus and MP4 are accepted by OpenAI
        async with app.sem:
            t = await app.stt.transcribe(buf.getvalue(), filename, mime, app.cfg.stt_language)
        app.db.add_usage(chat_id, stt_count=1, stt_seconds=duration)
        text = t.text or NO_SPEECH
        app.db.add_message(
            chat_id, msg.message_id, msg.from_user.id if msg.from_user else None, author_of(msg), kind,
            MARK + text, int(msg.date.timestamp()),
        )
        return text, duration


def transcript_messages(text: str, duration: float) -> list[str]:
    if text.startswith(NO_SPEECH):
        return ["🎙 <i>Речь не распознана</i>"]
    if len(text) <= 300:
        return [f"🎙 {esc(text)}"]
    chunks = split_text(text, 3500)
    out = []
    for i, chunk in enumerate(chunks):
        head = "🎙 <b>Расшифровка</b>" + (f" · {fmt_duration(duration)}" if duration and i == 0 else "")
        if len(chunks) > 1:
            head += f" ({i + 1}/{len(chunks)})"
        out.append(f"{head}\n<blockquote expandable>{esc(chunk)}</blockquote>")
    return out


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
async def _photo_part(bot: Bot, message: Message) -> dict | None:
    if not message.photo:
        return None
    photo = message.photo[-1]
    if (photo.file_size or 0) > TG_DOWNLOAD_LIMIT:
        return None
    try:
        buf = await bot.download(photo.file_id)
    except TelegramAPIError as exc:
        log.warning("photo download failed: %s", exc)
        return None
    assert buf is not None
    b64 = base64.b64encode(buf.getvalue()).decode()
    return {"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}", "detail": "high"}


RETRYABLE_WITHOUT = (400, 404, 422)


async def _ask_model(app: App, model: str, items: list[dict], items_no_images: list[dict] | None) -> ChatResult:
    """Responses API with search; on rejection retry without images, then plain chat completions."""
    attempts = [items] + ([items_no_images] if items_no_images else [])
    for attempt in attempts:
        try:
            return await app.xai.respond(model, attempt)
        except APIError as exc:
            if exc.status not in RETRYABLE_WITHOUT:
                raise
            log.warning("responses API rejected request (%s), trying fallback", exc)
    return await app.xai.chat(model, to_chat_messages(attempts[-1]))


def _sources_md(result: ChatResult, limit: int = 5) -> str:
    if not result.sources:
        return ""
    links = []
    for s in result.sources[:limit]:
        host = urlparse(s.url).netloc.removeprefix("www.") or "ссылка"
        links.append(f"[{host}]({s.url})")
    return "\n\n🔎 " + " · ".join(links)


def _history_block(app: App, history: list[StoredMessage]) -> str:
    tz = _tz(app)
    lines: list[str] = []
    used = 0
    for m in reversed(history):
        text = m.text.lstrip(MARK)
        if m.kind == "bot":
            author = "Ты (бот)"
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
        if used + len(line) > HISTORY_CHAR_BUDGET:
            break
        used += len(line)
        lines.append(line)
    return "\n".join(reversed(lines))


async def answer(bot: Bot, app: App, message: Message, question: str, *, place: str) -> None:
    chat_id = message.chat.id
    user = message.from_user
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
            elif r.photo:
                part = await _photo_part(bot, r)
                if part:
                    images.append(part)
                rtext = "(фото, приложено) " + rtext
            elif rkind in ("audio", "video", "animation", "document"):
                rtext = f"({KIND_LABEL[rkind]}; содержимое файла тебе недоступно) {rtext}"
            rauthor = "ты (бот)" if r.from_user and r.from_user.id == app.me.id else author_of(r)
            quote = f"\nЦитата из него: «{message.quote.text}»" if message.quote and message.quote.text else ""
            reply_block = f"Вопрос задан в ответ на сообщение от {rauthor}:\n«{rtext[:8000]}»{quote}\n\n"
        if message.photo:
            part = await _photo_part(bot, message)
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
        system = SYSTEM_PROMPT.format(name=app.me.first_name, place=place, now=now, tz=app.cfg.timezone)

        def build(with_images: bool) -> list[dict]:
            content: Any = user_text
            if with_images and images:
                content = [{"type": "input_text", "text": user_text}, *images]
            return [{"role": "system", "content": system}, {"role": "user", "content": content}]

        try:
            async with app.sem:
                result = await _ask_model(app, app.chat_model(), build(True), build(False) if images else None)
        except APIError as exc:
            log.error("chat failed in %s: %s", chat_id, exc)
            alert = _alert_if_account_problem(bot, app, exc)
            if alert:
                await alert
            await send_reply(bot, message, "⚠️ Не удалось получить ответ от модели. Попробуйте позже.")
            return

    app.db.add_usage(
        chat_id, llm_requests=1, input_tokens=result.input_tokens, output_tokens=result.output_tokens
    )
    if result.searches:
        log.info("answer in %s used %d search call(s)", chat_id, result.searches)
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
}


async def _react(bot: Bot, message: Message, emoji: str | None) -> None:
    try:
        await bot.set_message_reaction(
            chat_id=message.chat.id, message_id=message.message_id,
            reaction=[ReactionTypeEmoji(emoji=emoji)] if emoji else [],
        )
    except TelegramAPIError:
        pass  # reactions may be disabled in the chat; purely cosmetic


async def _send_download(bot: Bot, message: Message, dl: Download) -> Message:
    caption = f'📥 <a href="{html.escape(dl.link.url, quote=True)}">{dl.link.service}</a>'
    if dl.uploader and dl.uploader.lower() != dl.link.service.lower():
        caption += f" · {esc(dl.uploader)}"
    common: dict[str, Any] = dict(
        chat_id=message.chat.id,
        reply_parameters=ReplyParameters(message_id=message.message_id, allow_sending_without_reply=True),
        request_timeout=300,
    )
    if thread_id(message):
        common["message_thread_id"] = thread_id(message)
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


async def download_links(bot: Bot, app: App, message: Message, links: list[Link]) -> None:
    for link in links:
        await _react(bot, message, "👀")
        try:
            async with app.dl_sem:
                async with ChatActionSender.upload_video(bot=bot, chat_id=message.chat.id,
                                                         message_thread_id=thread_id(message)):
                    dl = await download(link, app.cookies_path)
                    try:
                        sent = await _send_download(bot, message, dl)
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
            hint = DL_HINTS.get(exc.reason, DL_HINTS["failed"]).format(service=link.service)
            await app.notify_owner(
                bot, f"📥 {esc(hint)}\n<code>{esc(exc.detail[:300])}</code>",
                throttle_key=f"dl:{link.service}:{exc.reason}",
            )
        except TelegramAPIError as exc:
            log.warning("sending download from %s failed: %s", link.url, exc)
            await _react(bot, message, "🤷")
        else:
            await _react(bot, message, None)
            title = f": {dl.title}" if dl.title else ""
            app.db.add_message(message.chat.id, sent.message_id, app.me.id, "Бот", "video",
                               f"(скачано из {link.service}{title})", int(sent.date.timestamp()))


def _without_links(message: Message) -> str:
    text = message.text or message.caption or ""
    for e in message.entities or message.caption_entities or []:
        if e.type == "url":
            text = text.replace(e.extract_from(message.text or message.caption or ""), "")
    return text.strip()


# ---------------------------------------------------------------- dispatch
async def handle(bot: Bot, app: App, message: Message, *, place: str, private: bool,
                 transcribe: bool, answer_on: bool, download_on: bool) -> None:
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
                await answer(bot, app, message, question or transcript, place=place)
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
        await download_links(bot, app, message, links)
        if private and not _without_links(message):
            return  # a bare link in private: just download it

    if from_bot or not answer_on or kind == "sticker":
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
                 transcribe=row.transcribe, answer_on=row.answer, download_on=row.download)


@router.message(F.chat.type == "private")
async def on_private_message(message: Message, bot: Bot, app: App) -> None:
    if not app.is_owner(message.from_user):
        return
    await handle(bot, app, message, place="личный чат с владельцем", private=True, transcribe=True, answer_on=True,
                 download_on=True)
