"""Grok Imagine: pictures ("@bot нарисуй кота", /draw, "переделай в аниме" on a photo) and videos
("@bot оживи" on a photo, "сними видео про …", /video)."""
from __future__ import annotations

import base64
import logging
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass

from aiogram import Bot
from aiogram.exceptions import TelegramAPIError
from aiogram.types import BufferedInputFile, Message, ReplyParameters, User
from aiogram.utils.chat_action import ChatActionSender

from .apis import APIError
from .app import App, send_reply, thread_id
from .chat import TG_DOWNLOAD_LIMIT, _alert_if_account_problem, addressed, author_of
from .textutil import esc

log = logging.getLogger(__name__)

USER_LIMIT = 5  # pictures per user per USER_WINDOW (the owner is not limited)
USER_WINDOW = 3600
CHAT_DAILY_LIMIT = 40  # pictures per chat per day from non-owners
VIDEO_USER_LIMIT = 2  # videos per user per USER_WINDOW (they cost ~$0.12 each)
VIDEO_CHAT_DAILY_LIMIT = 10
VIDEO_DEFAULT_SECONDS = 6
VIDEO_MAX_SECONDS = 10  # the owner may ask for up to 15

_PREFIX = r"^\W*(?:(?:ну|а|слушай|пожалуйста|плиз|pls|please)\W+)*"
_ASK = r"(?:(?:можешь|сможешь|можете|могла бы|можно)\W+)?"
_NOUN = r"(?:картин\w*|картинк\w*|изображени\w*|рисун\w*|арт\w*|фот\w*|мем\w*|аватар\w*|аву|авку|логотип\w*|лого|обо\w*|стикер\w*|иллюстраци\w*|постер\w*|пикч\w*|image|picture|art)"
DRAW_RE = re.compile(
    _PREFIX + _ASK
    + r"(?:нарису\w*|нарисова\w*|изобраз\w*|отрису\w*|draw\b|imagine\b|"
    + r"(?:сгенер\w*|сделай\w*|сделать|создай\w*|создать|сотвори\w*|нагенер\w*|generate|make|create)\W+(?:\w+\W+){0,2}" + _NOUN + r")",
    re.I,
)
EDIT_RE = re.compile(
    _PREFIX + _ASK
    + r"(?:переделай\w*|перерису\w*|нарису\w*|дорису\w*|измени\w*|отредактиру\w*|отфотошоп\w*|добав\w*|убер\w*|"
    + r"замени\w*|поменя\w*|преврат\w*|стилизу\w*|раскрас\w*|сделай\w*\W+(?:из|в стиле|его|её|ее|это|эту|фото|"
    + r"картинк\w*|так|как|черно|ч/б|аниме|мульт\w*)|edit\b|make it\b|turn (?:it|this|him|her) into)",
    re.I,
)
_VIDEO_NOUN = r"(?:видео\w*|видос\w*|ролик\w*|клип\w*|анимаци\w*|мульт\w*|гиф\w*|gif\w*|video|clip)"
VIDEO_RE = re.compile(
    _PREFIX + _ASK
    + r"(?:сними\w*|снять|сгенер\w*|сделай\w*|сделать|создай\w*|создать|нарису\w*|запили\w*|смонтиру\w*|"
    + r"generate|make|create)\W+(?:\w+\W+){0,3}" + _VIDEO_NOUN,
    re.I,
)
ANIMATE_RE = re.compile(
    _PREFIX + _ASK
    + r"(?:оживи\w*|анимиру\w*|пусть\W+(?:\w+\W+)?(?:двига|шевел|танцу|улыба|моргн|помаш|побеж|полет|запою|заговор)\w*|"
    + r"animate\b|make (?:it|this|him|her) move|bring (?:it|this) to life)",
    re.I,
)
NOT_VIDEO_RE = re.compile(r"саммари|самари|пересказ|перескаж|кратк|о ч[её]м|расшифр|субтитр|скача|найди|посоветуй|подбер", re.I)
DURATION_RE = re.compile(r"(\d{1,2})\s*(?:сек\w*|с\b|s\b|sec\w*)", re.I)
COMMANDS = ("draw", "img", "imagine")
VIDEO_COMMANDS = ("video", "animate")
ASPECTS = [
    (re.compile(r"\b\d{1,2}\s*:\s*\d{1,2}\b"), None),
    (re.compile(r"вертикальн|для сторис|stories|portrait|на телефон|обои на телефон", re.I), "9:16"),
    (re.compile(r"горизонтальн|широкоформат|landscape|обои на (?:комп|рабочий|пк|монитор)|для ютуб", re.I), "16:9"),
    (re.compile(r"квадрат|square|аватар|аву\b|авку\b", re.I), "1:1"),
]
ALLOWED_ASPECTS = {"1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "2:1", "1:2", "21:9", "9:20", "20:9"}
VIDEO_ASPECTS = {"1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "21:9", "5:2"}


@dataclass
class DrawRequest:
    prompt: str
    source: Message | None = None  # photo to edit / animate
    video: bool = False


def _photo_message(message: Message) -> Message | None:
    if message.photo:
        return message
    r = message.reply_to_message
    if r is not None and r.photo and r.forum_topic_created is None:
        return r
    return None


def requested_image(message: Message, me: User, *, private: bool) -> DrawRequest | None:
    """A picture request (new or an edit of a photo), else None."""
    text = (message.text or message.caption or "").strip()
    if text.startswith("/"):
        parts = text.split(maxsplit=1)
        name, _, target = parts[0][1:].partition("@")
        name = name.lower()
        if name not in COMMANDS + VIDEO_COMMANDS or (target and target.lower() != (me.username or "").lower()):
            return None
        prompt = parts[1].strip() if len(parts) > 1 else ""
        photo = _photo_message(message)
        if name in VIDEO_COMMANDS:
            return DrawRequest(prompt or ("Оживи это фото" if photo else ""), photo, video=True)
        return DrawRequest(prompt, photo)
    if private:
        ok, question = message.forward_origin is None, text
    else:
        ok, question = addressed(message, me)
    if not ok or not question:
        return None
    photo = _photo_message(message)
    wants_video = bool(VIDEO_RE.search(question)) and not NOT_VIDEO_RE.search(question)
    animate = bool(ANIMATE_RE.search(question))
    if photo is not None and (wants_video or animate):
        return DrawRequest(question, photo, video=True)
    if wants_video:
        return DrawRequest(question, video=True)
    if animate and len(question) <= 40:
        return DrawRequest("", video=True)  # "оживи" without a photo: explain how
    if photo is not None and EDIT_RE.search(question):
        return DrawRequest(question, photo)
    if DRAW_RE.search(question):
        return DrawRequest(question)
    return None


def aspect_ratio(prompt: str, allowed: set[str] = ALLOWED_ASPECTS) -> str | None:
    for pattern, value in ASPECTS:
        m = pattern.search(prompt)
        if m:
            if value is None:
                ratio = re.sub(r"\s", "", m.group(0))
                if ratio in allowed:
                    return ratio
                continue
            return value
    return None


def video_seconds(prompt: str, owner: bool = False) -> int:
    m = DURATION_RE.search(prompt)
    if not m:
        return VIDEO_DEFAULT_SECONDS
    return max(1, min(15 if owner else VIDEO_MAX_SECONDS, int(m.group(1))))


_user_log: dict[tuple[int, int], deque[float]] = defaultdict(deque)
_video_log: dict[tuple[int, int], deque[float]] = defaultdict(deque)


def _allowed(app: App, message: Message, video: bool = False) -> str | None:
    """None if the user may draw now, else the reason to show."""
    user = message.from_user
    if user is None or app.is_owner(user):
        return None
    if video:
        if app.db.videos_today(message.chat.id) >= VIDEO_CHAT_DAILY_LIMIT:
            return f"🎬 На сегодня лимит видео в этом чате исчерпан ({VIDEO_CHAT_DAILY_LIMIT}). Завтра сниму ещё!"
    elif app.db.images_today(message.chat.id) >= CHAT_DAILY_LIMIT:
        return f"🎨 На сегодня лимит картинок в этом чате исчерпан ({CHAT_DAILY_LIMIT}). Завтра нарисую ещё!"
    q = (_video_log if video else _user_log)[(message.chat.id, user.id)]
    limit = VIDEO_USER_LIMIT if video else USER_LIMIT
    now = time.monotonic()
    while q and now - q[0] > USER_WINDOW:
        q.popleft()
    if len(q) >= limit:
        wait = int((USER_WINDOW - (now - q[0])) // 60) + 1
        if video:
            return f"🎬 Я уже сняла тебе {limit} видео за час — следующее смогу через {wait} мин."
        return f"🎨 Я уже нарисовала тебе {limit} картинок за час — следующую смогу через {wait} мин."
    q.append(now)
    return None


def _context(message: Message, req: DrawRequest) -> str:
    """The request plus what it refers to (the replied-to text), so "нарисуй это" works."""
    prompt = req.prompt
    r = message.reply_to_message
    if req.source is None and r is not None and r.forum_topic_created is None:
        rtext = (r.text or r.caption or "").strip()
        if rtext:
            prompt += f"\n\nО чём речь (сообщение, на которое ответили): {rtext[:1500]}"
    return prompt


def _is_moderation(exc: APIError) -> bool:
    text = str(exc).lower()
    return exc.status in (400, 403, 422) and any(
        w in text for w in ("moderat", "violat", "policy", "safety", "guideline", "inappropriate", "content"))


async def _source_uri(bot: Bot, message: Message, req: DrawRequest) -> str | None:
    """The photo to edit/animate as a data: URI; None after telling the user why it is unavailable."""
    assert req.source is not None and req.source.photo
    photo = req.source.photo[-1]
    if (photo.file_size or 0) > TG_DOWNLOAD_LIMIT:
        await send_reply(bot, message, "🎨 Фото слишком большое, не могу его скачать.")
        return None
    buf = await bot.download(photo.file_id)
    assert buf is not None
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


async def _api_failed(bot: Bot, app: App, message: Message, exc: APIError, what: str) -> None:
    log.error("%s failed in %s: %s", what, message.chat.id, exc)
    alert = _alert_if_account_problem(bot, app, exc)
    if alert:
        await alert
    if _is_moderation(exc):
        await send_reply(bot, message, "🙅‍♀️ Такое я делать не буду — запрос не прошёл фильтр Grok.")
    elif what == "video":
        await send_reply(bot, message, "⚠️ Не получилось снять видео — Grok не справился. Попробуйте ещё раз позже.")
    else:
        await send_reply(bot, message, "⚠️ Не получилось нарисовать — Grok не ответил. Попробуйте ещё раз позже.")


def _reply_params(message: Message) -> dict:
    params: dict = dict(
        chat_id=message.chat.id,
        reply_parameters=ReplyParameters(message_id=message.message_id, allow_sending_without_reply=True),
    )
    if thread_id(message):
        params["message_thread_id"] = thread_id(message)
    return params


async def draw(bot: Bot, app: App, message: Message, req: DrawRequest) -> None:
    if req.video:
        await make_video(bot, app, message, req)
        return
    chat_id = message.chat.id
    if not req.prompt:
        name = f"@{app.me.username}" if app.me.username else "бот"
        await send_reply(bot, message, f"🎨 Что нарисовать? Например: «{esc(name)} нарисуй кота-космонавта» "
                                       "или ответьте на фото: «переделай в стиле аниме».")
        return
    reason = _allowed(app, message)
    if reason:
        await send_reply(bot, message, reason)
        return
    prompt = _context(message, req)
    log.info("image %s in %s from %s: %.80r", "edit" if req.source else "draw", chat_id,
             message.from_user.id if message.from_user else None, req.prompt)
    try:
        async with ChatActionSender.upload_photo(bot=bot, chat_id=chat_id, message_thread_id=thread_id(message)):
            source = None
            if req.source is not None:
                source = await _source_uri(bot, message, req)
                if source is None:
                    return
            async with app.sem:
                image = await app.xai.image(app.cfg.image_model, prompt, source=source,
                                            aspect_ratio=aspect_ratio(req.prompt))
    except APIError as exc:
        await _api_failed(bot, app, message, exc, "image")
        return
    except TelegramAPIError as exc:
        log.warning("photo for edit not downloaded: %s", exc)
        await send_reply(bot, message, "⚠️ Не получилось скачать фото из Telegram.")
        return
    app.db.add_usage(chat_id, images=1)

    caption = f"🎨 <blockquote expandable>{esc(req.prompt[:900])}</blockquote>"
    params = _reply_params(message)
    file = BufferedInputFile(image, filename="grok.jpg")
    try:
        sent = await bot.send_photo(photo=file, caption=caption, parse_mode="HTML", **params)
    except TelegramAPIError as exc:  # unusual size/format for a photo: send as a file
        log.warning("send_photo failed, sending as document: %s", exc)
        sent = await bot.send_document(document=BufferedInputFile(image, filename="grok.jpg"), caption=caption,
                                       parse_mode="HTML", **params)
    what = "отредактировала фото" if req.source else "нарисовала картинку"
    app.db.add_message(chat_id, sent.message_id, app.me.id, "Бот", "photo",
                       f"({what} по просьбе {author_of(message)}: {req.prompt[:300]})", int(sent.date.timestamp()))


async def make_video(bot: Bot, app: App, message: Message, req: DrawRequest) -> None:
    chat_id = message.chat.id
    if not req.prompt:
        name = f"@{app.me.username}" if app.me.username else "бот"
        await send_reply(bot, message, f"🎬 Ответьте на фото: «{esc(name)} оживи» — и я сделаю из него видео. "
                                       f"Или опишите сюжет: «{esc(name)} сними видео, как кот играет на гитаре».")
        return
    reason = _allowed(app, message, video=True)
    if reason:
        await send_reply(bot, message, reason)
        return
    owner = app.is_owner(message.from_user)
    seconds = video_seconds(req.prompt, owner)
    prompt = _context(message, req)
    log.info("video %s in %s from %s (%ds): %.80r", "from photo" if req.source else "from text", chat_id,
             message.from_user.id if message.from_user else None, seconds, req.prompt)
    note = await send_reply(bot, message, "🎬 Снимаю видео — обычно это меньше минуты…")
    try:
        async with ChatActionSender.upload_video(bot=bot, chat_id=chat_id, message_thread_id=thread_id(message)):
            source = None
            if req.source is not None:
                source = await _source_uri(bot, message, req)
                if source is None:
                    return
            async with app.video_sem:
                video = await app.xai.video(app.cfg.video_model, prompt, image=source, duration=seconds,
                                            aspect_ratio=None if source else aspect_ratio(req.prompt, VIDEO_ASPECTS))
    except APIError as exc:
        await _api_failed(bot, app, message, exc, "video")
        return
    except TelegramAPIError as exc:
        log.warning("photo for video not downloaded: %s", exc)
        await send_reply(bot, message, "⚠️ Не получилось скачать фото из Telegram.")
        return
    finally:
        try:
            await bot.delete_message(chat_id, note.message_id)
        except TelegramAPIError:
            pass
    app.db.add_usage(chat_id, videos=1)
    caption = f"🎬 <blockquote expandable>{esc(req.prompt[:900])}</blockquote>"
    params = _reply_params(message)
    try:
        sent = await bot.send_video(video=BufferedInputFile(video, filename="grok.mp4"), caption=caption,
                                    parse_mode="HTML", supports_streaming=True, **params)
    except TelegramAPIError as exc:
        log.warning("send_video failed, sending as document: %s", exc)
        sent = await bot.send_document(document=BufferedInputFile(video, filename="grok.mp4"), caption=caption,
                                       parse_mode="HTML", **params)
    what = "оживила фото в видео" if req.source else "сняла видео"
    app.db.add_message(chat_id, sent.message_id, app.me.id, "Бот", "video",
                       f"({what} по просьбе {author_of(message)}: {req.prompt[:300]})", int(sent.date.timestamp()))
