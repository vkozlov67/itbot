"""Chat summaries: /summary [hours], "@bot что было за день", and an optional daily post."""
from __future__ import annotations

import asyncio
import logging
import re
import time
from datetime import datetime, timedelta

from aiogram import Bot
from aiogram.exceptions import TelegramAPIError
from aiogram.types import Message, User
from aiogram.utils.chat_action import ChatActionSender

from .apis import APIError
from .app import App, send_reply, thread_id
from .chat import _alert_if_account_problem, _history_block, _tz, addressed
from .textutil import md_to_html, split_text

log = logging.getLogger(__name__)

DEFAULT_HOURS = 24
MAX_HOURS = 72  # history is kept for 3 days
MIN_MESSAGES = 5
COOLDOWN = 600  # seconds between on-demand summaries per chat (owner is exempt)
SUMMARY_CHAR_BUDGET = 150_000

TRIGGER = re.compile(
    r"(саммари|самари|summary|сводк|итог|подыт|кратко.{0,20}(чат|переписк)|"
    r"что (тут |здесь |в чате )?(было|обсуждали|произошло|пропустил)|о ч[её]м (тут |здесь )?(говорили|болтали|писали))",
    re.I,
)
HOURS_RE = re.compile(r"(\d{1,2})\s*(?:ч\b|час)", re.I)

PROMPT = """Ты — {name}, девушка, участница Telegram-чата «{title}». О себе говоришь в женском роде.
Сделай саммари переписки за {period}. Пиши по-русски, живо, но информативно и по делу.
Формат:
**🗞 Итоги {period_title}**
- 3–8 главных тем: в каждой коротко суть и кто что сказал (по именам).
- Договорённости, планы, решения — если были.
- Полезные ссылки и находки — если были.
- Одной строкой — самое смешное или яркое, если было.
Опирайся только на переписку, ничего не выдумывай. Не пересказывай каждое сообщение. Голосовые уже расшифрованы.
Не больше ~2500 символов. Простой Markdown: **жирный**, списки через «- », без таблиц и заголовков #."""


def _period(hours: int) -> tuple[str, str]:
    if hours == 24:
        return "последние сутки", "дня"
    if hours % 24 == 0:
        return f"последние {hours // 24} дня", f"за {hours // 24} дня"
    return f"последние {hours} ч", f"за {hours} ч"


def requested_hours(message: Message, me: User) -> int | None:
    """Hours to summarize if this group message asks for a summary, else None."""
    text = (message.text or message.caption or "").strip()
    if text.startswith("/"):
        parts = text.split(maxsplit=1)
        name, _, target = parts[0][1:].partition("@")
        if name.lower() != "summary" or (target and target.lower() != (me.username or "").lower()):
            return None
        arg = parts[1] if len(parts) > 1 else ""
        m = re.search(r"\d+", arg)
        return max(1, min(MAX_HOURS, int(m.group()))) if m else DEFAULT_HOURS
    ok, question = addressed(message, me)
    if not ok or len(question) > 120 or not TRIGGER.search(question):
        return None
    if message.reply_to_message is not None and message.reply_to_message.from_user \
            and message.reply_to_message.from_user.id != me.id:
        return None  # "@bot подведи итог" in reply to a specific message is a normal question
    m = HOURS_RE.search(question)
    if m:
        return max(1, min(MAX_HOURS, int(m.group(1))))
    if re.search(r"недел|3 дня|три дня", question, re.I):
        return MAX_HOURS
    return DEFAULT_HOURS


async def build_summary(app: App, chat_id: int, title: str, hours: int) -> str | None:
    msgs = [m for m in app.db.messages_since(chat_id, int(time.time()) - hours * 3600) if m.kind != "summary"]
    if sum(1 for m in msgs if m.kind != "bot") < MIN_MESSAGES:
        return None
    period, period_title = _period(hours)
    history = _history_block(app, msgs, budget=SUMMARY_CHAR_BUDGET)
    items = [
        {"role": "system", "content": PROMPT.format(name=app.me.first_name, title=title, period=period,
                                                    period_title=period_title)},
        {"role": "user", "content": f"Переписка за {period} (время — {app.cfg.timezone}):\n{history}"},
    ]
    async with app.sem:
        result = await app.xai.respond(app.chat_model(), items, search=False)
    app.db.add_usage(chat_id, llm_requests=1, input_tokens=result.input_tokens, output_tokens=result.output_tokens)
    return result.text or None


async def _send(bot: Bot, app: App, chat_id: int, text: str, reply_to: Message | None) -> None:
    for chunk in split_text(text):
        if reply_to is not None:
            sent = await send_reply(bot, reply_to, md_to_html(chunk), fallback=chunk)
        else:
            try:
                sent = await bot.send_message(chat_id, md_to_html(chunk), parse_mode="HTML")
            except TelegramAPIError:
                sent = await bot.send_message(chat_id, chunk, parse_mode=None)
        app.db.add_message(chat_id, sent.message_id, app.me.id, "Бот", "summary", chunk, int(sent.date.timestamp()))


async def reply_summary(bot: Bot, app: App, message: Message, hours: int, title: str) -> None:
    chat_id = message.chat.id
    now = time.monotonic()
    if not app.is_owner(message.from_user) and now - app.summary_last.get(chat_id, -COOLDOWN) < COOLDOWN:
        await send_reply(bot, message, "⏳ Я недавно уже делала саммари — пролистайте чуть выше или спросите через пару минут.")
        return
    app.summary_last[chat_id] = now
    log.info("summary requested in %s for %dh", chat_id, hours)
    try:
        async with ChatActionSender.typing(bot=bot, chat_id=chat_id, message_thread_id=thread_id(message)):
            text = await build_summary(app, chat_id, title, hours)
    except APIError as exc:
        log.error("summary failed in %s: %s", chat_id, exc)
        alert = _alert_if_account_problem(bot, app, exc)
        if alert:
            await alert
        await send_reply(bot, message, "⚠️ Не получилось сделать саммари — модель не ответила. Попробуйте позже.")
        return
    if text is None:
        await send_reply(bot, message, f"За {_period(hours)[0]} в чате почти ничего не было — нечего подытоживать 🙂")
        return
    await _send(bot, app, chat_id, text, message)


# ---------------------------------------------------------------- daily schedule
def _parse_time(value: str) -> tuple[int, int]:
    try:
        hh, mm = value.split(":")
        return max(0, min(23, int(hh))), max(0, min(59, int(mm)))
    except ValueError:
        return 21, 0


async def daily_tick(bot: Bot, app: App) -> None:
    now = datetime.now(_tz(app))
    hh, mm = _parse_time(app.cfg.summary_time)
    start = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if not (start <= now < start + timedelta(hours=2)):
        return
    today = now.date().isoformat()
    for row in app.db.list_chats():
        if not (row.enabled and row.daily_summary):
            continue
        key = f"daily_summary:{row.chat_id}"
        if app.db.get_setting(key) == today:
            continue
        app.db.set_setting(key, today)  # mark first: never double-post, even if sending fails
        try:
            text = await build_summary(app, row.chat_id, row.title, DEFAULT_HOURS)
            if text:
                await _send(bot, app, row.chat_id, text, None)
                log.info("daily summary posted in %s", row.chat_id)
        except (APIError, TelegramAPIError) as exc:
            log.error("daily summary in %s failed: %s", row.chat_id, exc)


async def scheduler(bot: Bot, app: App) -> None:
    while True:
        try:
            await daily_tick(bot, app)
        except Exception:  # never let the loop die
            log.exception("summary scheduler tick failed")
        await asyncio.sleep(60)
