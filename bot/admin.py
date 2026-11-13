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
from .downloader import SERVICES, cookie_domains, looks_like_cookies

log = logging.getLogger(__name__)
router = Router(name="admin")

PRIVATE_BOT_TEXT = "🔒 Это приватный бот. Он работает только в чатах, которые выбрал владелец."


class Panel(CallbackData, prefix="p"):
    a: str  # action
    c: int = 0  # chat id
    v: str = ""  # value


def btn(text: str, **kw) -> InlineKeyboardButton:
    return InlineKeyboardButton(text=text, callback_data=Panel(**kw).pack())


# ---------------------------------------------------------------- commands menu
async def setup_commands(bot: Bot, app: App) -> None:
    try:
        await bot.set_my_commands(
            [BotCommand(command="ask", description="Задать вопрос боту")], scope=BotCommandScopeAllGroupChats()
        )
        await bot.delete_my_commands(scope=BotCommandScopeAllPrivateChats())
        if app.owner_id:
            await bot.set_my_commands(
                [
                    BotCommand(command="panel", description="Панель управления"),
                    BotCommand(command="reset", description="Очистить память личного диалога"),
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
        f"Ответы (Grok): <code>{esc(app.chat_model())}</code>",
        f"Расшифровка (OpenAI): <code>{esc(app.cfg.stt_model)}</code>",
        "Режим приватности бота: "
        + ("выключен ✅ (видит все сообщения в группах)" if app.me.can_read_all_group_messages
           else "включён — в группах, где бот не админ, он видит только /ask и ответы на свои сообщения"),
        "Cookies для скачивания: "
        + (", ".join(esc(d) for d in cookie_domains(app.cookies_path)[:6]) or "нет"
           if app.cookies_path.exists() else "нет"),
        "",
        f"Сегодня: 🎙 {today.stt_count} ({fmt_duration(today.stt_seconds)}) · "
        f"💬 {today.llm_requests} ({fmt_tokens(today.input_tokens + today.output_tokens)} ток.)",
        f"Всего: 🎙 {total.stt_count} ({fmt_duration(total.stt_seconds)}) · "
        f"💬 {total.llm_requests} ({fmt_tokens(total.input_tokens + total.output_tokens)} ток.)",
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
    rows.append([btn("🔄 Обновить", a="home")])
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
    kb = [
        [btn("⏸ Выключить бота" if row.enabled else "▶️ Включить бота", a="tog", c=row.chat_id, v="enabled")],
        [
            btn(f"🎙 Расшифровка: {'вкл' if row.transcribe else 'выкл'}", a="tog", c=row.chat_id, v="transcribe"),
            btn(f"💬 Ответы: {'вкл' if row.answer else 'выкл'}", a="tog", c=row.chat_id, v="answer"),
        ],
        [btn(f"📥 Скачивание по ссылкам: {'вкл' if row.download else 'выкл'}", a="tog", c=row.chat_id,
             v="download")],
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


def cookies_view(app: App) -> tuple[str, InlineKeyboardMarkup]:
    services = ", ".join(name for name, _ in SERVICES)
    domains = cookie_domains(app.cookies_path) if app.cookies_path.exists() else []
    lines = [
        "<b>🍪 Cookies для скачивания видео</b>",
        "",
        f"Бот скачивает видео по ссылкам: {services}.",
        "Instagram, X, VK, Facebook и Pinterest работают без входа. YouTube, TikTok и Reddit с сервера "
        "отдают видео только залогиненным — для них нужны cookies.",
        "",
        "<b>Как добавить:</b>",
        "1. Войдите на нужные сайты в браузере на компьютере (лучше с запасного аккаунта).",
        "2. Поставьте расширение «Get cookies.txt LOCALLY» (Chrome/Firefox) и экспортируйте cookies "
        "в формате Netscape.",
        "3. Пришлите файл <code>cookies.txt</code> сюда, в личку боту. Можно один файл с cookies нескольких сайтов.",
        "",
        "Сейчас: " + (", ".join(esc(d) for d in domains[:15]) if domains else "cookies не загружены"),
    ]
    rows = []
    if app.cookies_path.exists():
        rows.append([btn("🗑 Удалить cookies", a="ckdel")])
    rows.append([btn("« Назад", a="home")])
    return "\n".join(lines), InlineKeyboardMarkup(inline_keyboard=rows)


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
    tmp = app.cookies_path.with_suffix(".tmp")
    tmp.write_bytes(data)
    tmp.chmod(0o600)
    tmp.replace(app.cookies_path)
    domains = cookie_domains(app.cookies_path)
    log.info("cookies updated: %d domains", len(domains))
    await message.answer(
        "🍪 Cookies сохранены. Сайты: " + (", ".join(esc(d) for d in domains[:20]) or "—")
        + "\n\nСообщение с файлом лучше удалить из чата.",
        parse_mode="HTML",
    )


# ---------------------------------------------------------------- callbacks
@router.callback_query(Panel.filter())
async def on_panel_cb(cb: CallbackQuery, callback_data: Panel, bot: Bot, app: App) -> None:
    if not app.is_owner(cb.from_user):
        await cb.answer("Только для владельца", show_alert=True)
        return
    if not isinstance(cb.message, Message):
        await cb.answer("Сообщение устарело — отправьте /panel")
        return
    a, chat_id = callback_data.a, callback_data.c

    if a == "home":
        await _show(cb, *(await home_view(app)))
        await cb.answer()
        return

    if a == "models":
        await cb.answer("Загружаю список…")
        await _show(cb, *(await models_view(app)))
        return

    if a == "cookies":
        await cb.answer()
        await _show(cb, *cookies_view(app))
        return

    if a == "ckdel":
        app.cookies_path.unlink(missing_ok=True)
        await cb.answer("Cookies удалены")
        await _show(cb, *cookies_view(app))
        return

    if a == "setm":
        app.db.set_setting("chat_model", callback_data.v)
        await cb.answer(f"Модель: {callback_data.v}")
        await _show(cb, *(await models_view(app)))
        return

    row = app.db.get_chat(chat_id)
    if row is None:
        await cb.answer("Чат не найден", show_alert=True)
        await _show(cb, *(await home_view(app)))
        return

    if a == "chat":
        view = await chat_view(bot, app, chat_id)
        if view is None:
            await cb.answer("Бота уже нет в этом чате", show_alert=True)
            await _show(cb, *(await home_view(app)))
        else:
            await cb.answer()
            await _show(cb, *view)
        return

    if a == "tog":
        flag = callback_data.v
        if flag not in ("enabled", "transcribe", "answer", "download"):
            await cb.answer()
            return
        new_value = not getattr(row, flag)
        app.db.set_flag(chat_id, flag, new_value)
        label = {"enabled": "Бот", "transcribe": "Расшифровка", "answer": "Ответы", "download": "Скачивание"}[flag]
        await cb.answer(f"{label}: {'включено' if new_value else 'выключено'}")
        if flag == "enabled" and new_value:
            await announce(bot, app, chat_id)
        view = await chat_view(bot, app, chat_id)
        await _show(cb, *(view or await home_view(app)))
        return

    if a == "clr":
        app.db.clear_history(chat_id)
        await cb.answer("Память чата очищена")
        view = await chat_view(bot, app, chat_id)
        await _show(cb, *(view or await home_view(app)))
        return

    if a == "leave":
        await cb.answer()
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
        await cb.answer("Бот вышел из чата")
        await _show(cb, *(await home_view(app)))
        return

    await cb.answer()


async def announce(bot: Bot, app: App, chat_id: int) -> None:
    row = app.db.get_chat(chat_id)
    if row is None:
        return
    parts = []
    if row.transcribe:
        parts.append("🎙 расшифровываю голосовые и кружки")
    if row.download:
        parts.append("📥 скачиваю видео по ссылкам (Instagram Reels, X, VK, Facebook, Pinterest…)")
    if row.answer:
        parts.append(f"💬 отвечаю на вопросы — упомяните @{esc(app.me.username or '')}, ответьте на моё сообщение или напишите /ask")
    if not parts:
        return
    try:
        await bot.send_message(chat_id, "👋 Я на связи:\n" + "\n".join(parts), parse_mode="HTML")
    except TelegramAPIError as exc:
        log.info("announce in %s failed: %s", chat_id, exc)


def chat_actions_markup(chat_id: int, enabled: bool) -> InlineKeyboardMarkup:
    first = btn("⏸ Выключить", a="tog", c=chat_id, v="enabled") if enabled else btn(
        "✅ Включить", a="tog", c=chat_id, v="enabled"
    )
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [first, btn("🚪 Выйти", a="leave", c=chat_id)],
            [btn("⚙️ Настройки чата", a="chat", c=chat_id)],
        ]
    )
