"""Chat summaries and stats: /summary [hours], "@bot что было за день", /stats, optional daily and weekly posts."""
from __future__ import annotations

import asyncio
import logging
import re
import time
from collections import Counter
from datetime import datetime, timedelta, timezone

from aiogram import Bot
from aiogram.exceptions import TelegramAPIError
from aiogram.types import Message, User
from aiogram.utils.chat_action import ChatActionSender

from .apis import APIError
from .app import App, send_reply, thread_id
from .chat import _alert_if_account_problem, _history_block, _tz, addressed
from .textutil import md_to_html, split_text
from .db import StoredMessage

log = logging.getLogger(__name__)

DEFAULT_HOURS = 24
MAX_HOURS = 168  # history is kept for 8 days
MIN_MESSAGES = 5
COOLDOWN = 600  # seconds between on-demand summaries per chat (owner is exempt)
SUMMARY_CHAR_BUDGET = 150_000
WEEK_CHAR_BUDGET = 400_000
STATS_DAYS = 7
WEEKDAYS = ("понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье")

TRIGGER = re.compile(
    r"(саммари|самари|summary|сводк|итог|подыт|кратко.{0,20}(чат|переписк)|"
    r"что (тут |здесь |в чате )?(было|обсуждали|произошло|пропустил)|о ч[её]м (тут |здесь )?(говорили|болтали|писали))",
    re.I,
)
HOURS_RE = re.compile(r"(\d{1,3})\s*(?:ч\b|час)", re.I)
DAYS_RE = re.compile(r"(\d{1,2})\s*(?:дн|день|дня|сут)", re.I)
STATS_RE = re.compile(
    r"(статистик|\bстат[аыу]\b|\bstats?\b|кто (?:больше|чаще|меньше) (?:всех|всего)|топ (?:болтун|участник|актив)|"
    r"сам(?:ый|ые|ая) (?:активн|болтлив))",
    re.I,
)

PROMPT = """Ты — {name}, девушка, участница Telegram-чата «{title}». О себе говоришь в женском роде.
Сделай саммари переписки за {period}. Пиши по-русски, живо, но информативно и по делу.
Формат:
**🗞 Итоги {period_title}**
- 3–8 главных тем: в каждой коротко суть и кто что сказал (по именам).
- Договорённости, планы, решения — если были.
- Полезные ссылки и находки — если были.
- Одной строкой — самое смешное или яркое, если было.
Опирайся только на переписку, ничего не выдумывай. Не пересказывай каждое сообщение. Голосовые уже расшифрованы.
{extra}Не больше ~{limit} символов. Простой Markdown: **жирный**, списки через «- », без таблиц и заголовков #."""
WEEK_EXTRA = ("- 🏆 Сообщение недели: дословно процитируй одно самое смешное или меткое сообщение участника "
              "и укажи автора.\n")


def _days(n: int) -> str:
    if n % 10 == 1 and n % 100 != 11:
        return f"{n} день"
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return f"{n} дня"
    return f"{n} дней"


def _period(hours: int) -> tuple[str, str]:
    if hours == 24:
        return "последние сутки", "дня"
    if hours == 168:
        return "последнюю неделю", "недели"
    if hours % 24 == 0:
        return f"последние {_days(hours // 24)}", f"за {_days(hours // 24)}"
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
        if re.search(r"недел|week", arg, re.I):
            return 168
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
    m = DAYS_RE.search(question)
    if m:
        return max(1, min(MAX_HOURS, int(m.group(1)) * 24))
    if re.search(r"недел", question, re.I):
        return 168
    if re.search(r"три дня", question, re.I):
        return 72
    return DEFAULT_HOURS


async def build_summary(app: App, chat_id: int, title: str, hours: int) -> str | None:
    msgs = [m for m in app.db.messages_since(chat_id, int(time.time()) - hours * 3600) if m.kind != "summary"]
    if sum(1 for m in msgs if m.kind != "bot") < MIN_MESSAGES:
        return None
    period, period_title = _period(hours)
    week = hours >= 96
    history = _history_block(app, msgs, budget=WEEK_CHAR_BUDGET if week else SUMMARY_CHAR_BUDGET)
    items = [
        {"role": "system", "content": PROMPT.format(name=app.me.first_name, title=title, period=period,
                                                    period_title=period_title, extra=WEEK_EXTRA if week else "",
                                                    limit=3500 if week else 2500)},
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


# ---------------------------------------------------------------- stats
def requested_stats(message: Message, me: User) -> int | None:
    """Days of stats if this group message asks for chat statistics, else None."""
    text = (message.text or message.caption or "").strip()
    if text.startswith("/"):
        parts = text.split(maxsplit=1)
        name, _, target = parts[0][1:].partition("@")
        if name.lower() != "stats" or (target and target.lower() != (me.username or "").lower()):
            return None
        m = re.search(r"\d+", parts[1] if len(parts) > 1 else "")
        return max(1, min(STATS_DAYS, int(m.group()))) if m else STATS_DAYS
    ok, question = addressed(message, me)
    if not ok or len(question) > 120 or not STATS_RE.search(question):
        return None
    m = DAYS_RE.search(question)
    if m:
        return max(1, min(STATS_DAYS, int(m.group(1))))
    if re.search(r"сегодня|за день|сутки", question, re.I):
        return 1
    return STATS_DAYS


def _stats_period(days: int) -> str:
    return "сутки" if days == 1 else "неделю" if days == 7 else _days(days)


def build_stats(app: App, chat_id: int, days: int) -> str | None:
    since = int(time.time()) - days * 86400
    all_msgs = app.db.messages_since(chat_id, since)
    people = [m for m in all_msgs if m.user_id != app.me.id and m.kind not in ("bot", "summary")]
    if not people:
        return None
    tz = _tz(app)

    def key(m: StoredMessage) -> str:
        return str(m.user_id) if m.user_id is not None else m.author

    names = {key(m): m.author for m in people}  # latest name wins
    counts = Counter(key(m) for m in people)
    lines = [f"**📊 Статистика чата за {_stats_period(days)}**",
             f"💬 Сообщений: **{len(people)}** от **{len(counts)}** участников", ""]
    medals = ["🥇", "🥈", "🥉", "4.", "5."]
    lines.append("🏆 Самые активные:")
    for medal, (k, n) in zip(medals, counts.most_common(5)):
        lines.append(f"{medal} {names[k]} — {n}")
    kinds = Counter(m.kind for m in people)
    media = [("🎙 голосовых", "voice"), ("⭕️ кружков", "video_note"), ("📷 фото", "photo"), ("🎞 видео", "video"),
             ("😀 стикеров", "sticker"), ("📎 файлов", "document")]
    parts = [f"{label}: {kinds[k]}" for label, k in media if kinds[k]]
    if parts:
        lines += ["", " · ".join(parts)]
    links = sum(1 for m in all_msgs if m.user_id == app.me.id and m.text.startswith("(скачано из"))
    stamps = [datetime.fromtimestamp(m.date, timezone.utc).astimezone(tz) for m in people]
    extra = []
    if links:
        extra.append(f"🔗 Скачала по ссылкам: {links}")
    if days >= 3:
        day, n = Counter(d.weekday() for d in stamps).most_common(1)[0]
        extra.append(f"📅 Самый активный день: {WEEKDAYS[day]} ({n})")
    hour, _ = Counter(d.hour for d in stamps).most_common(1)[0]
    extra.append(f"⏰ Пик активности: {hour:02d}:00–{(hour + 1) % 24:02d}:00")
    night = Counter(key(m) for m, d in zip(people, stamps) if d.hour < 5)
    if night and night.most_common(1)[0][1] >= 3:
        k, n = night.most_common(1)[0]
        extra.append(f"🦉 Сова: {names[k]} ({n} сообщ. после полуночи)")
    lines += [""] + extra
    since_day = datetime.fromtimestamp(since, timezone.utc).strftime("%Y-%m-%d")
    u = app.db.chat_usage_since(chat_id, since_day)
    bot_parts = [text for text, n in ((f"ответила на {u.llm_requests} вопр.", u.llm_requests),
                                      (f"расшифровала {u.stt_count} голосовых", u.stt_count),
                                      (f"нарисовала {u.images} картинок", u.images),
                                      (f"сняла {u.videos} видео", u.videos),
                                      (f"озвучила {u.tts_count}", u.tts_count)) if n]
    if bot_parts:
        lines.append("🤖 Я: " + " · ".join(bot_parts))
    return "\n".join(lines)


async def reply_stats(bot: Bot, app: App, message: Message, days: int) -> None:
    log.info("stats requested in %s for %dd", message.chat.id, days)
    text = build_stats(app, message.chat.id, days)
    if text is None:
        await send_reply(bot, message, f"За {_stats_period(days)} в чате ещё ничего не было 🙂")
        return
    await _send(bot, app, message.chat.id, text, message)


# ---------------------------------------------------------------- daily schedule
def _parse_time(value: str) -> tuple[int, int]:
    try:
        hh, mm = value.split(":")
        return max(0, min(23, int(hh))), max(0, min(59, int(mm)))
    except ValueError:
        return 21, 0


def _in_window(app: App) -> datetime | None:
    now = datetime.now(_tz(app))
    hh, mm = _parse_time(app.cfg.summary_time)
    start = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    return now if start <= now < start + timedelta(hours=2) else None


async def daily_tick(bot: Bot, app: App) -> None:
    now = _in_window(app)
    if now is None:
        return
    today = now.date().isoformat()
    for row in app.db.list_chats():
        if not (row.enabled and row.daily_summary):
            continue
        if row.weekly_summary and now.weekday() == app.cfg.weekly_day:
            continue  # the weekly post covers this evening
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


async def weekly_tick(bot: Bot, app: App) -> None:
    now = _in_window(app)
    if now is None or now.weekday() != app.cfg.weekly_day:
        return
    year, week, _ = now.isocalendar()
    stamp = f"{year}-W{week:02d}"
    for row in app.db.list_chats():
        if not (row.enabled and row.weekly_summary):
            continue
        key = f"weekly_summary:{row.chat_id}"
        if app.db.get_setting(key) == stamp:
            continue
        app.db.set_setting(key, stamp)  # mark first: never double-post
        try:
            stats = build_stats(app, row.chat_id, 7)
            if stats is None:
                continue
            await _send(bot, app, row.chat_id, stats, None)
            text = await build_summary(app, row.chat_id, row.title, 168)
            if text:
                await _send(bot, app, row.chat_id, text, None)
            log.info("weekly summary posted in %s", row.chat_id)
        except (APIError, TelegramAPIError) as exc:
            log.error("weekly summary in %s failed: %s", row.chat_id, exc)


async def scheduler(bot: Bot, app: App) -> None:
    while True:
        try:
            await daily_tick(bot, app)
            await weekly_tick(bot, app)
        except Exception:  # never let the loop die
            log.exception("summary scheduler tick failed")
        await asyncio.sleep(60)
