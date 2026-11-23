from __future__ import annotations

import logging
import secrets

from aiogram import Bot, F, Router
from aiogram.exceptions import TelegramAPIError, TelegramBadRequest
from aiogram.filters import Command, CommandObject, CommandStart
from aiogram.filters.callback_data import CallbackData
from aiogram.types import (
    BotCommand,
    BotCommandScopeAllGroupChats,
    BotCommandScopeAllPrivateChats,
    BotCommandScopeChat,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from .app import App
from .db import ChatRow
from .textutil import esc, fmt_duration, fmt_tokens
from .apis import APIError
from .downloader import SERVICES, cookie_domains, looks_like_cookies, merge_cookies, signed_out_sites

log = logging.getLogger(__name__)
router = Router(name="admin")

PRIVATE_BOT_TEXT = "🔒 Это приватный бот. Он работает только в чатах, которые выбрал владелец."


class Panel(CallbackData, prefix="p"):
    a: str  # action
    c: int = 0  # chat id
    v: str = ""  # value


FLAG_LABELS = {"enabled": "Бот", "transcribe": "Расшифровка", "answer": "Ответы", "download": "Скачивание",
               "daily_summary": "Итоги дня", "delete_links": "Удаление ссылок", "weekly_summary": "Итоги недели",
               "voice_replies": "Голосом на голосовые"}
WEEKDAYS_ON = ("по понедельникам", "по вторникам", "по средам", "по четвергам", "по пятницам", "по субботам",
               "по воскресеньям")


def btn(text: str, **kw) -> InlineKeyboardButton:
    return InlineKeyboardButton(text=text, callback_data=Panel(**kw).pack())


# ---------------------------------------------------------------- commands menu
async def setup_commands(bot: Bot, app: App) -> None:
    try:
        await bot.set_my_commands(
            [
                BotCommand(command="ask", description="Задать вопрос"),
                BotCommand(command="summary", description="Итоги чата за сутки (/summary 6 — за 6 часов)"),
                BotCommand(command="draw", description="Нарисовать картинку (Grok Imagine)"),
                BotCommand(command="video", description="Снять видео (в ответ на фото — оживить его)"),
                BotCommand(command="stats", description="Статистика чата за неделю (/stats 1 — за сутки)"),
            ],
            scope=BotCommandScopeAllGroupChats(),
        )
        await bot.delete_my_commands(scope=BotCommandScopeAllPrivateChats())
        if app.owner_id:
            await bot.set_my_commands(
                [
                    BotCommand(command="panel", description="Панель управления"),
                    BotCommand(command="reset", description="Очистить память личного диалога"),
                    BotCommand(command="draw", description="Нарисовать картинку"),
                    BotCommand(command="video", description="Снять видео / оживить фото"),
                ],
                scope=BotCommandScopeChat(chat_id=app.owner_id),
            )
    except TelegramAPIError as exc:
        log.warning("set_my_commands failed: %s", exc)


# ---------------------------------------------------------------- views
def _status_label(row: ChatRow) -> str:
    if row.is_admin:
        return "👑 администратор"
    if row.status == "restricted":
        return "⛔️ ограничен в правах"
    return "👤 обычный участник"


async def home_view(app: App) -> tuple[str, InlineKeyboardMarkup]:
    chats = app.db.list_chats()
    today = app.db.usage_totals("today")
    total = app.db.usage_totals()
    lines = [
        "<b>⚙️ Панель управления</b>",
        "",
        f"Ответы (Grok): <code>{esc(app.chat_model())}</code>"
        + (f", сложные задачи — <code>{esc(app.hard_model())}</code>" if app.hard_model() else ""),
        f"Расшифровка (OpenAI): <code>{esc(app.cfg.stt_model)}</code>",
        f"Картинки и видео (Grok Imagine): <code>{esc(app.cfg.image_model)}</code>, "
        f"<code>{esc(app.cfg.video_model)}</code>",
        (f"Голос (ElevenLabs): <code>{esc(app.cfg.elevenlabs_model)}</code>, голос "
         f"<code>{esc(app.tts.voice_name or app.tts.voice_id or 'выберется автоматически')}</code>; "
         f"запасной — OpenAI <code>{esc(app.cfg.tts_voice)}</code>"
         if app.tts else
         f"Голос (OpenAI): <code>{esc(app.cfg.tts_model)}</code>, голос <code>{esc(app.cfg.tts_voice)}</code>"),
        "Режим приватности бота: "
        + ("выключен ✅ (видит все сообщения в группах)" if app.me.can_read_all_group_messages
           else "включён — в группах, где бот не админ, он видит только /ask и ответы на свои сообщения"),
        "Cookies для скачивания: "
        + (", ".join(esc(d) for d in cookie_domains(app.cookies_path)[:6]) or "нет"
           if app.cookies_path.exists() else "нет"),
        "",
        f"Сегодня: 🎙 {today.stt_count} ({fmt_duration(today.stt_seconds)}) · "
        f"💬 {today.llm_requests} ({fmt_tokens(today.input_tokens + today.output_tokens)} ток.) · 🎨 {today.images}"
        f" · 🎬 {today.videos} · 🔊 {today.tts_count}",
        f"Всего: 🎙 {total.stt_count} ({fmt_duration(total.stt_seconds)}) · "
        f"💬 {total.llm_requests} ({fmt_tokens(total.input_tokens + total.output_tokens)} ток.) · 🎨 {total.images}"
        f" · 🎬 {total.videos} · 🔊 {total.tts_count}",
        "",
    ]
    blind = [c for c in chats if c.enabled and not c.is_admin and not app.me.can_read_all_group_messages]
    if blind:
        lines += [
            "⚠️ Бот не админ в: " + ", ".join(f"<b>{esc(c.title)}</b>" for c in blind)
            + ". Там он не видит упоминаний и голосовых. Сделайте его админом "
            "или выключите Group Privacy в @BotFather и добавьте бота заново.",
            "",
        ]
    if chats:
        lines.append("<b>Чаты</b> — ✅ работает, ⏸ выключен, ⚠️ не видит сообщений. Нажмите, чтобы настроить:")
    else:
        lines.append("Бот пока не добавлен ни в один чат. Нажмите «➕ Добавить в группу».")
    def mark(c: ChatRow) -> str:
        if not c.enabled:
            return "⏸"
        return "⚠️" if c in blind else "✅"

    rows = [[btn(f"{mark(c)} {c.title or c.chat_id}"[:60], a="chat", c=c.chat_id)] for c in chats]
    rows.append([InlineKeyboardButton(text="➕ Добавить в группу", url=app.add_to_group_link())])
    rows.append([btn("🤖 Модель", a="models"), btn("🍪 Cookies", a="cookies")])
    rows.append([btn("😏 Сарказм", a="tease"), btn("🔄 Обновить", a="home")])
    return "\n".join(lines), InlineKeyboardMarkup(inline_keyboard=rows)


async def _refresh_chat(bot: Bot, app: App, row: ChatRow) -> ChatRow:
    """Ask Telegram for the bot's real status in the chat and sync the DB."""
    try:
        member = await bot.get_chat_member(row.chat_id, app.me.id)
        status = member.status.value if hasattr(member.status, "value") else str(member.status)
        if status == "restricted" and not getattr(member, "is_member", True):
            status = "left"
        try:
            chat = await bot.get_chat(row.chat_id)
            app.db.upsert_chat(row.chat_id, chat.title or row.title, chat.type, chat.username, status=status)
        except TelegramAPIError:
            app.db.set_status(row.chat_id, status)
    except TelegramBadRequest as exc:
        if "not found" in str(exc).lower() or "kicked" in str(exc).lower():
            app.db.set_status(row.chat_id, "left")
            app.db.set_flag(row.chat_id, "enabled", False)
    except TelegramAPIError as exc:
        log.warning("refresh chat %s failed: %s", row.chat_id, exc)
    return app.db.get_chat(row.chat_id) or row


async def check_chats_on_start(bot: Bot, app: App) -> None:
    """Sync the bot's real status in every chat and warn the owner about chats where it can't see messages."""
    for row in app.db.list_chats():
        await _refresh_chat(bot, app, row)
    if app.me.can_read_all_group_messages:
        return
    blind = [c for c in app.db.list_chats() if c.enabled and not c.is_admin]
    if blind:
        await app.notify_owner(
            bot,
            "⚠️ В этих чатах я не администратор и поэтому не вижу упоминаний, вопросов и голосовых: "
            + ", ".join(f"<b>{esc(c.title)}</b>" for c in blind)
            + "\n\nСделайте меня админом там (права можно не давать) — или в @BotFather: /mybots → бот → "
            "Bot Settings → Group Privacy → Turn off, после чего удалите и снова добавьте меня в чат.",
            throttle_key="blind-chats",
        )


async def _owner_role(bot: Bot, app: App, chat_id: int) -> str:
    if not app.owner_id:
        return "?"
    try:
        m = await bot.get_chat_member(chat_id, app.owner_id)
    except TelegramAPIError:
        return "не удалось проверить"
    status = m.status.value if hasattr(m.status, "value") else str(m.status)
    return {
        "creator": "👑 создатель",
        "administrator": "👑 администратор",
        "member": "👤 участник",
        "restricted": "⛔️ ограничен",
        "left": "не состоите в чате",
        "kicked": "заблокированы в чате",
    }.get(status, status)


async def chat_view(bot: Bot, app: App, chat_id: int) -> tuple[str, InlineKeyboardMarkup] | None:
    row = app.db.get_chat(chat_id)
    if row is None:
        return None
    row = await _refresh_chat(bot, app, row)
    if not row.active:
        return None
    sees_all = row.is_admin or bool(app.me.can_read_all_group_messages)
    lines = [
        f"<b>{esc(row.title or str(row.chat_id))}</b>",
        f"ID: <code>{row.chat_id}</code>" + (f" · @{esc(row.username)}" if row.username else ""),
        "",
        f"Бот в чате: {_status_label(row)}",
        f"Вы в чате: {await _owner_role(bot, app, row.chat_id)}",
        f"Сообщений в памяти: {app.db.count_messages(row.chat_id)}",
        "",
        f"Состояние: {'✅ <b>работает</b>' if row.enabled else '⏸ <b>выключен</b>'}",
    ]
    if not sees_all:
        lines += [
            "",
            "⚠️ Бот не администратор — Telegram не показывает ему голосовые и обычные сообщения. "
            "Назначьте бота администратором чата (достаточно прав без галочек).",
        ]
    def flip(text: str, flag: str) -> InlineKeyboardButton:  # sets the opposite value explicitly (idempotent)
        return btn(text, a="set", c=row.chat_id, v=f"{flag}={0 if getattr(row, flag) else 1}")

    kb = [
        [flip("⏸ Выключить бота" if row.enabled else "▶️ Включить бота", "enabled")],
        [
            flip(f"🎙 Расшифровка: {'вкл' if row.transcribe else 'выкл'}", "transcribe"),
            flip(f"💬 Ответы: {'вкл' if row.answer else 'выкл'}", "answer"),
        ],
        [flip(f"📥 Скачивание по ссылкам: {'вкл' if row.download else 'выкл'}", "download")],
        [flip(f"🗑 Удалять сообщение со ссылкой: {'да' if row.delete_links else 'нет'}", "delete_links")],
        [flip(f"🗞 Итоги дня в {app.cfg.summary_time}: {'вкл' if row.daily_summary else 'выкл'}", "daily_summary")],
        [flip(f"📅 Итоги недели {WEEKDAYS_ON[app.cfg.weekly_day]}: {'вкл' if row.weekly_summary else 'выкл'}",
              "weekly_summary")],
        [flip(f"🗣 Отвечать голосом на голосовые: {'вкл' if row.voice_replies else 'выкл'}", "voice_replies")],
        [btn("🧹 Очистить память", a="clr", c=row.chat_id)],
        [btn("🚪 Выйти из чата", a="leave", c=row.chat_id)],
        [btn("« Назад", a="home")],
    ]
    return "\n".join(lines), InlineKeyboardMarkup(inline_keyboard=kb)


async def models_view(app: App) -> tuple[str, InlineKeyboardMarkup]:
    current = app.chat_model()
    lines = ["<b>🤖 Модель для ответов</b>", "", f"Сейчас: <code>{esc(current)}</code>", ""]
    try:
        models = await app.xai.list_chat_models()
    except APIError as exc:
        models = []
        lines.append(f"Не удалось получить список моделей: {esc(str(exc))}")
    if current not in models:
        models.insert(0, current)
    rows = []
    for m in models:
        cb = Panel(a="setm", v=m).pack()
        if len(cb.encode()) > 64:
            continue
        rows.append([InlineKeyboardButton(text=("✅ " if m == current else "") + m, callback_data=cb)])
    lines.append("Выберите модель:")
    rows.append([btn("« Назад", a="home")])
    return "\n".join(lines), InlineKeyboardMarkup(inline_keyboard=rows)


def tease_view(app: App, query: str = "") -> tuple[str, InlineKeyboardMarkup]:
    people = app.teased()
    lines = [
        "<b>😏 Сарказм</b>",
        "",
        "Этим людям бот отвечает язвительно, а иногда и не по делу — как будто у него такой характер. "
        "В чате это никак не видно: ни команд, ни пометок.",
        "",
        ("Сейчас: " + ", ".join(f"<b>{esc(n)}</b>" for n in people.values())) if people else "Сейчас: никто.",
        "",
        "Добавить: <code>/tease имя</code> (или часть имени, как в чате). Нажмите на человека, чтобы убрать.",
    ]
    rows = [[btn(f"❌ {name}"[:60], a="untease", v=str(uid))] for uid, name in people.items()]
    if query:
        found = [f for f in app.db.find_authors(query) if f[0] not in people]
        lines += ["", f"Найдено по «{esc(query)}»:" if found else f"По «{esc(query)}» никого не нашёл в чатах."]
        rows += [[btn(f"😏 {name} · {n} сообщ."[:60], a="tease", v=str(uid))] for uid, name, n in found]
    rows.append([btn("⬅️ Назад", a="home")])
    return "\n".join(lines), InlineKeyboardMarkup(inline_keyboard=rows)


def cookies_view(app: App) -> tuple[str, InlineKeyboardMarkup]:
    services = ", ".join(name for name, _ in SERVICES)
    domains = cookie_domains(app.cookies_path) if app.cookies_path.exists() else []
    try:
        signed_out = signed_out_sites(app.cookies_path.read_text(encoding="utf-8", errors="replace"))
    except OSError:
        signed_out = []
    lines = [
        "<b>🍪 Cookies для скачивания видео</b>",
        "",
        f"Бот скачивает видео по ссылкам: {services}.",
        "Instagram, Threads, X, VK, Facebook и Pinterest работают без входа. YouTube, TikTok и Reddit с сервера "
        "отдают видео только залогиненным — для них нужны cookies.",
        "Ссылки-«поделиться» вида <code>threads.com/share/…</code> раскрываются только с cookies Threads "
        "(прямые ссылки на пост <code>threads.com/@автор/post/…</code> работают и без них).",
        "",
        "<b>Как добавить:</b>",
        "1. Войдите на нужные сайты в браузере на компьютере (лучше с запасного аккаунта).",
        "2. Поставьте расширение «Get cookies.txt LOCALLY» (Chrome/Firefox) и экспортируйте cookies "
        "в формате Netscape.",
        "3. Пришлите файл <code>cookies.txt</code> сюда, в личку боту. Файлы добавляются: можно присылать "
        "по одному на сайт, новый файл заменяет cookies только тех сайтов, что в нём есть.",
        "",
        "Сейчас: " + (", ".join(esc(d) + (" ⚠️ без входа" if d in signed_out else "") for d in domains[:15])
                      if domains else "cookies не загружены"),
    ]
    if signed_out:
        lines.append("⚠️ «Без входа» — файл выгружен из браузера, где вы не вошли на сайт: такие cookies не помогают. "
                     "Войдите на сайт и выгрузите cookies заново.")
    rows = []
    if app.cookies_path.exists():
        rows.append([btn("🗑 Удалить cookies", a="ckdel")])
    rows.append([btn("« Назад", a="home")])
    return "\n".join(lines), InlineKeyboardMarkup(inline_keyboard=rows)


async def _ack(cb: CallbackQuery, text: str | None = None, show_alert: bool = False) -> None:
    """Answer a button press; a late answer ("query is too old") must not abort the action itself."""
    try:
        await cb.answer(text, show_alert=show_alert)
    except TelegramBadRequest as exc:
        log.info("callback answer skipped: %s", exc)


async def _show(cb: CallbackQuery, text: str, markup: InlineKeyboardMarkup) -> None:
    assert isinstance(cb.message, Message)
    try:
        await cb.message.edit_text(text, parse_mode="HTML", reply_markup=markup)
    except TelegramBadRequest as exc:
        if "not modified" not in str(exc):
            raise


# ---------------------------------------------------------------- private commands
@router.message(CommandStart(), F.chat.type == "private")
async def on_start(message: Message, command: CommandObject, bot: Bot, app: App) -> None:
    user = message.from_user
    if user is None:
        return
    arg = (command.args or "").strip()
    if arg.startswith("claim_") and app.owner_id is None:
        if secrets.compare_digest(arg[len("claim_"):], app.claim_code()):
            app.db.set_setting("owner_id", str(user.id))
            app.db.delete_setting("claim_code")
            await setup_commands(bot, app)
            log.info("owner claimed by %s (%s)", user.id, user.full_name)
            await message.answer("👑 Готово — вы владелец бота.", parse_mode="HTML")
        else:
            await message.answer("Код недействителен.")
            return
    if not app.is_owner(user):
        if app.owner_id is None:
            await message.answer(
                "👑 Владелец ещё не назначен. Откройте ссылку, которую показала команда itbot-setup "
                "на сервере (вида t.me/…?start=claim_…), и нажмите Start."
            )
        else:
            await message.answer(PRIVATE_BOT_TEXT)
        return
    text, markup = await home_view(app)
    await message.answer(
        text
        + "\n\n<i>Как пользоваться: добавьте бота в группу кнопкой ниже, сделайте админом и включите здесь. "
        "В чате бот расшифровывает голосовые и кружки, а на вопросы отвечает, когда его упоминают "
        f"(@{esc(app.me.username or '')}), отвечают на его сообщение или пишут /ask.</i>",
        parse_mode="HTML",
        reply_markup=markup,
    )


@router.message(Command("panel"), F.chat.type == "private")
async def on_panel(message: Message, app: App) -> None:
    if not app.is_owner(message.from_user):
        await message.answer(PRIVATE_BOT_TEXT)
        return
    text, markup = await home_view(app)
    await message.answer(text, parse_mode="HTML", reply_markup=markup)


@router.message(Command("tease"), F.chat.type == "private")
async def on_tease(message: Message, command: CommandObject, app: App) -> None:
    if not app.is_owner(message.from_user):
        return
    text, markup = tease_view(app, (command.args or "").strip())
    await message.answer(text, parse_mode="HTML", reply_markup=markup)


@router.message(Command("reset"), F.chat.type == "private")
async def on_reset(message: Message, app: App) -> None:
    if not app.is_owner(message.from_user):
        return
    app.db.clear_history(message.chat.id)
    await message.answer("🧹 Память личного диалога очищена.")


@router.message(
    F.chat.type == "private",
    lambda m: bool(m.document and (m.document.file_name or "").lower().endswith(".txt")),
)
async def on_cookies_file(message: Message, bot: Bot, app: App) -> None:
    if not app.is_owner(message.from_user):
        return
    doc = message.document
    assert doc is not None
    if (doc.file_size or 0) > 2 * 1024 * 1024:
        await message.answer("Файл слишком большой для cookies.txt.")
        return
    buf = await bot.download(doc.file_id)
    assert buf is not None
    data = buf.getvalue()
    if not looks_like_cookies(data):
        await message.answer("Это не похоже на cookies.txt в формате Netscape. См. /panel → 🍪 Cookies.")
        return
    try:
        old = app.cookies_path.read_text(encoding="utf-8", errors="replace") if app.cookies_path.exists() else ""
    except OSError:
        old = ""
    merged = merge_cookies(old, data.decode("utf-8", errors="replace"))
    tmp = app.cookies_path.with_suffix(".tmp")
    tmp.write_text(merged, encoding="utf-8")
    tmp.chmod(0o600)
    tmp.replace(app.cookies_path)
    domains = cookie_domains(app.cookies_path)
    log.info("cookies updated: %d domains", len(domains))
    signed_out = signed_out_sites(data.decode("utf-8", errors="replace"))
    warning = (
        "\n\n⚠️ Для " + ", ".join(f"<b>{esc(s)}</b>" for s in signed_out) + " в файле нет входа в аккаунт — "
        "похоже, он выгружен из браузера, где вы не вошли на этот сайт. Такие cookies сайт не принимает: "
        "войдите на сайт в браузере и выгрузите cookies заново." if signed_out else ""
    )
    await message.answer(
        "🍪 Cookies сохранены. Сайты: " + (", ".join(esc(d) for d in domains[:20]) or "—") + warning
        + "\n\nСообщение с файлом лучше удалить из чата.",
        parse_mode="HTML",
    )


# ---------------------------------------------------------------- callbacks
@router.callback_query(Panel.filter())
async def on_panel_cb(cb: CallbackQuery, callback_data: Panel, bot: Bot, app: App) -> None:
    if not app.is_owner(cb.from_user):
        await _ack(cb, "Только для владельца", show_alert=True)
        return
    if not isinstance(cb.message, Message):
        await _ack(cb, "Сообщение устарело — отправьте /panel")
        return
    a, chat_id = callback_data.a, callback_data.c

    if a == "home":
        await _show(cb, *(await home_view(app)))
        await _ack(cb)
        return

    if a == "models":
        await _ack(cb, "Загружаю список…")
        await _show(cb, *(await models_view(app)))
        return

    if a == "cookies":
        await _ack(cb)
        await _show(cb, *cookies_view(app))
        return

    if a == "tease" and callback_data.v:
        uid = int(callback_data.v)
        found = app.db.find_authors(str(uid), limit=1)
        app.set_teased(uid, found[0][1] if found else str(uid))
        await _ack(cb, "Добавлен")
        await _show(cb, *tease_view(app))
        return

    if a == "tease":
        await _ack(cb)
        await _show(cb, *tease_view(app))
        return

    if a == "untease":
        app.set_teased(int(callback_data.v), None)
        await _ack(cb, "Убран")
        await _show(cb, *tease_view(app))
        return

    if a == "ckdel":
        app.cookies_path.unlink(missing_ok=True)
        await _ack(cb, "Cookies удалены")
        await _show(cb, *cookies_view(app))
        return

    if a == "setm":
        app.db.set_setting("chat_model", callback_data.v)
        await _ack(cb, f"Модель: {callback_data.v}")
        await _show(cb, *(await models_view(app)))
        return

    row = app.db.get_chat(chat_id)
    if row is None:
        await _ack(cb, "Чат не найден", show_alert=True)
        await _show(cb, *(await home_view(app)))
        return

    if a == "chat":
        view = await chat_view(bot, app, chat_id)
        if view is None:
            await _ack(cb, "Бота уже нет в этом чате", show_alert=True)
            await _show(cb, *(await home_view(app)))
        else:
            await _ack(cb)
            await _show(cb, *view)
        return

    if a == "tog":  # buttons from older versions toggled blindly — just show the current state
        await _ack(cb, "Кнопка устарела — вот актуальные настройки")
        view = await chat_view(bot, app, chat_id)
        await _show(cb, *(view or await home_view(app)))
        return

    if a == "set":
        flag, _, value = callback_data.v.partition("=")
        if flag not in FLAG_LABELS or value not in ("0", "1"):
            await _ack(cb)
            return
        new_value = value == "1"
        changed = bool(getattr(row, flag)) != new_value
        app.db.set_flag(chat_id, flag, new_value)
        await _ack(cb, f"{FLAG_LABELS[flag]}: {'включено' if new_value else 'выключено'}")
        if flag == "enabled" and new_value and changed:
            await announce(bot, app, chat_id)
        view = await chat_view(bot, app, chat_id)
        await _show(cb, *(view or await home_view(app)))
        return

    if a == "clr":
        app.db.clear_history(chat_id)
        await _ack(cb, "Память чата очищена")
        view = await chat_view(bot, app, chat_id)
        await _show(cb, *(view or await home_view(app)))
        return

    if a == "leave":
        await _ack(cb)
        await _show(
            cb,
            f"Выйти из чата <b>{esc(row.title)}</b>? Чтобы вернуть бота, его придётся добавить заново.",
            InlineKeyboardMarkup(
                inline_keyboard=[
                    [btn("🚪 Да, выйти", a="leave_ok", c=chat_id)],
                    [btn("Отмена", a="chat", c=chat_id)],
                ]
            ),
        )
        return

    if a == "leave_ok":
        try:
            await bot.leave_chat(chat_id)
        except TelegramAPIError as exc:
            log.warning("leave_chat %s: %s", chat_id, exc)
        app.db.set_flag(chat_id, "enabled", False)
        app.db.set_status(chat_id, "left")
        await _ack(cb, "Бот вышел из чата")
        await _show(cb, *(await home_view(app)))
        return

    await _ack(cb)


async def announce(bot: Bot, app: App, chat_id: int) -> None:
    row = app.db.get_chat(chat_id)
    if row is None:
        return
    parts = []
    if row.transcribe:
        parts.append("🎙 расшифровываю голосовые и кружки")
    if row.download:
        parts.append("📥 скачиваю видео и фото по ссылкам вместе с текстом поста (Instagram, Threads, X, VK, Facebook, Pinterest…)")
    if row.answer:
        u = esc(app.me.username or "")
        parts.append("🗞 /summary — итоги чата за сутки, /stats — статистика за неделю")
        parts.append(f"🎨 рисую картинки — «@{u} нарисуй …», а в ответ на фото — «переделай в стиле …»")
        parts.append(f"🎬 оживляю фото — ответьте на фото «@{u} оживи», или «@{u} сними видео …»")
        parts.append(f"🔊 отвечаю голосом — «@{u} ответь голосом …», а «@{u} озвучь» в ответ на сообщение зачитает его"
                     + ("; на голосовое в ответ мне — отвечу голосовым" if row.voice_replies else ""))
        parts.append("👀 вижу фото и скриншоты — спросите про них, ответив на картинку")
    if row.answer:
        parts.append(f"💬 отвечаю на вопросы — упомяните @{esc(app.me.username or '')}, ответьте на моё сообщение или напишите /ask")
    if not parts:
        return
    try:
        await bot.send_message(chat_id, "👋 Я на связи:\n" + "\n".join(parts), parse_mode="HTML")
    except TelegramAPIError as exc:
        log.info("announce in %s failed: %s", chat_id, exc)


def chat_actions_markup(chat_id: int, enabled: bool) -> InlineKeyboardMarkup:
    first = btn("⏸ Выключить", a="set", c=chat_id, v="enabled=0") if enabled else btn(
        "✅ Включить", a="set", c=chat_id, v="enabled=1"
    )
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [first, btn("🚪 Выйти", a="leave", c=chat_id)],
            [btn("⚙️ Настройки чата", a="chat", c=chat_id)],
        ]
    )
