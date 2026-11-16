"""Spoken replies: "@bot ответь голосом …", voice answers to voice questions, "@bot озвучь" in reply to a message.

ElevenLabs when ELEVENLABS_API_KEY is set (MP3 converted to OGG/Opus with ffmpeg), otherwise — or if it fails —
OpenAI TTS."""
from __future__ import annotations

import asyncio
import logging
import re

from aiogram import Bot
from aiogram.exceptions import TelegramAPIError
from aiogram.types import BufferedInputFile, Message, ReplyParameters, User
from aiogram.utils.chat_action import ChatActionSender

from .apis import APIError
from .app import App, send_reply, thread_id
from .chat import _alert_if_account_problem, _u16, addressed, author_of
from .textutil import esc, md_to_html, md_to_plain, split_text

log = logging.getLogger(__name__)

TTS_LIMIT = 4000  # characters per voice message
CAPTION_LIMIT = 950  # UTF-16 units; longer texts go as a separate message
INSTRUCTIONS = ("Говори на языке текста. Голос молодой девушки: тёплый, живой и естественный, как в дружеском "
                "разговоре; обычный темп, без театральности и без дикторской интонации.")
VOICE_PROMPT = ("\nЭтот ответ будет озвучен и отправлен голосовым сообщением. Пиши как живую устную речь: коротко "
                "(2–6 предложений, до ~700 символов, если не просят подробнее), без Markdown, списков, ссылок, кода "
                "и эмодзи; числа и сокращения пиши так, чтобы их было легко произнести.")

_ADV = r"(?:голосом|голосовым|вслух|войсом|in (?:a )?voice|out loud|aloud)"
_V = r"(?:голосом|голосовым|голосовое|голосовухой|вслух|войсом|войс|аудио|in (?:a )?voice|out loud|aloud)"
VOICE_RE = re.compile(
    r"(?:ответь|отвечай|скажи|расскажи|объясни|пришли|запиши|продиктуй|спой|answer|tell|say|reply)\W+(?:\w+\W+){0,3}" + _V
    + r"|^\W*" + _ADV + r"\b|\b" + _ADV + r"\W*$",
    re.I,
)
READ_RE = re.compile(
    r"^\W*(?:(?:ну|а|пожалуйста|плиз)\W+)*(?:озвучь\w*|озвучи\w*|прочитай\W+(?:\w+\W+)?вслух|прочти\W+(?:\w+\W+)?вслух|"
    r"зачитай\w*|скажи это голосом|read (?:it |this )?(?:out loud|aloud))\W*(?:пожалуйста|плиз)?\W*$",
    re.I,
)


async def to_ogg_opus(audio: bytes) -> bytes | None:
    """Any audio -> OGG/Opus (what Telegram plays as a voice message); None if ffmpeg failed."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-loglevel", "error", "-i", "pipe:0", "-vn", "-ac", "1", "-c:a", "libopus", "-b:a", "48k",
            "-application", "voip", "-f", "ogg", "pipe:1",
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        out, err = await asyncio.wait_for(proc.communicate(audio), timeout=60)
    except (OSError, asyncio.TimeoutError) as exc:
        log.error("ffmpeg opus conversion failed: %r", exc)
        return None
    if proc.returncode != 0 or not out.startswith(b"OggS"):
        log.error("ffmpeg opus conversion failed (%s): %s", proc.returncode, err[:300])
        return None
    return out


async def synthesize(bot: Bot, app: App, text: str) -> bytes:
    """OGG/Opus speech: ElevenLabs first (if configured), OpenAI as the fallback. Raises APIError if both fail."""
    if app.tts is not None:
        try:
            ogg = await to_ogg_opus(await app.tts.speech(text))
            if ogg:
                return ogg
        except APIError as exc:
            log.error("ElevenLabs tts failed, falling back to OpenAI: %s", exc)
            alert = _alert_if_account_problem(bot, app, exc)
            if alert:
                await alert
    return await app.stt.speech(text, model=app.cfg.tts_model, voice=app.cfg.tts_voice, instructions=INSTRUCTIONS)


def wants_voice(question: str) -> bool:
    return bool(VOICE_RE.search(question.strip()))


def requested_reading(message: Message, me: User, *, private: bool) -> str | None:
    """Text to read aloud when the message is "@bot озвучь" in reply to a text message, else None."""
    text = (message.text or message.caption or "").strip()
    if private:
        ok, question = message.forward_origin is None and not text.startswith("/"), text
    else:
        ok, question = addressed(message, me)
    if not ok or not READ_RE.search(question):
        return None
    r = message.reply_to_message
    if r is None or r.forum_topic_created is not None:
        return None
    return (r.text or r.caption or "").strip() or None


async def speak(bot: Bot, app: App, message: Message, text: str, *, extra_md: str = "",
                show_text: bool = True) -> Message | None:
    """Send `text` (Markdown) as a voice reply with the text attached; None if speech synthesis failed."""
    spoken = md_to_plain(text)[:TTS_LIMIT]
    if not spoken:
        return None
    chat_id = message.chat.id
    try:
        async with ChatActionSender.record_voice(bot=bot, chat_id=chat_id, message_thread_id=thread_id(message)):
            audio = await synthesize(bot, app, spoken)
    except APIError as exc:
        log.error("tts failed in %s: %s", chat_id, exc)
        alert = _alert_if_account_problem(bot, app, exc)
        if alert:
            await alert
        return None
    app.db.add_usage(chat_id, tts_count=1)
    shown = (text + extra_md).strip()
    params: dict = dict(
        chat_id=chat_id,
        reply_parameters=ReplyParameters(message_id=message.message_id, allow_sending_without_reply=True),
    )
    if thread_id(message):
        params["message_thread_id"] = thread_id(message)
    caption = None
    long_text = show_text and _u16(shown) > CAPTION_LIMIT
    if show_text and not long_text:
        caption = f"<blockquote expandable>{md_to_html(shown)}</blockquote>"
    voice_file = BufferedInputFile(audio, filename="voice.ogg")
    try:
        sent = await bot.send_voice(voice=voice_file, caption=caption, parse_mode="HTML", **params)
    except TelegramAPIError as exc:
        if not caption:
            raise
        log.warning("voice caption rejected, sending plain: %s", exc)
        sent = await bot.send_voice(voice=BufferedInputFile(audio, filename="voice.ogg"),
                                    caption=f"<blockquote expandable>{esc(shown)}</blockquote>", parse_mode="HTML",
                                    **params)
    app.db.add_message(chat_id, sent.message_id, app.me.id, "Бот", "bot", text, int(sent.date.timestamp()))
    if long_text:
        for chunk in split_text(shown):
            await send_reply(bot, sent, md_to_html(chunk), fallback=chunk)
    return sent


async def read_aloud(bot: Bot, app: App, message: Message, text: str) -> None:
    log.info("read aloud in %s for %s (%d chars)", message.chat.id,
             message.from_user.id if message.from_user else None, len(text))
    user = message.from_user
    if user and not app.is_owner(user) and not app.limiter.allow((message.chat.id, user.id)):
        await send_reply(bot, message, "⏳ Слишком много просьб подряд — подождите минуту.")
        return
    sent = await speak(bot, app, message, text, show_text=False)
    if sent is None:
        await send_reply(bot, message, "⚠️ Не получилось озвучить — сервис синтеза речи не ответил. Попробуйте позже.")
    else:
        app.db.add_message(message.chat.id, sent.message_id, app.me.id, "Бот", "bot",
                           f"(озвучила вслух сообщение по просьбе {author_of(message)})", int(sent.date.timestamp()))
