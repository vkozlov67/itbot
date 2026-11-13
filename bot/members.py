from __future__ import annotations

import logging

from aiogram import Bot, F, Router
from aiogram.types import ChatMemberUpdated, Message

from .admin import announce, chat_actions_markup
from .app import App
from .textutil import esc

log = logging.getLogger(__name__)
router = Router(name="members")

IN = {"creator", "administrator", "member", "restricted"}


def _status(member) -> str:
    status = member.status.value if hasattr(member.status, "value") else str(member.status)
    if status == "restricted" and not getattr(member, "is_member", True):
        return "left"
    return status


@router.my_chat_member()
async def on_my_chat_member(event: ChatMemberUpdated, bot: Bot, app: App) -> None:
    chat = event.chat
    if chat.type not in ("group", "supergroup"):
        return
    old, new = _status(event.old_chat_member), _status(event.new_chat_member)
    joined, removed = old not in IN and new in IN, old in IN and new not in IN
    actor = event.from_user
    title = esc(chat.title or str(chat.id))
    by = esc(actor.full_name) + (f" (@{esc(actor.username)})" if actor.username else "")

    row = app.db.upsert_chat(
        chat.id, chat.title or str(chat.id), chat.type, chat.username, status=new,
        added_by=actor.id if joined else None,
    )
    log.info("my_chat_member %s: %s -> %s by %s", chat.id, old, new, actor.id)

    if joined:
        by_owner = app.is_owner(actor)
        app.db.set_flag(chat.id, "enabled", by_owner)
        text = f"➕ Меня добавили в чат <b>{title}</b>\nДобавил: {by}\n\n"
        if by_owner:
            text += "✅ Вы добавили меня сами — в этом чате я <b>включён</b>."
        else:
            text += "⏸ В этом чате я <b>выключен</b>, пока вы не включите."
        if new != "administrator" and not app.me.can_read_all_group_messages:
            text += (
                "\n\n⚠️ Я не администратор и не увижу голосовые и обычные сообщения. "
                "Назначьте меня администратором чата."
            )
        await app.notify_owner(bot, text, chat_actions_markup(chat.id, by_owner))
        if by_owner:
            await announce(bot, app, chat.id)
    elif removed:
        app.db.set_flag(chat.id, "enabled", False)
        await app.notify_owner(bot, f"🚪 Меня удалили из чата <b>{title}</b> ({by}).")
    elif old != new:
        if new == "administrator":
            await app.notify_owner(
                bot,
                f"👑 Теперь я администратор в <b>{title}</b>.",
                chat_actions_markup(chat.id, row.enabled) if not row.enabled else None,
            )
        elif old == "administrator" and not app.me.can_read_all_group_messages:
            await app.notify_owner(
                bot,
                f"⚠️ Меня лишили прав администратора в <b>{title}</b> — голосовые там больше не видны.",
            )


@router.message(F.migrate_to_chat_id)
async def on_migrate_to(message: Message, app: App) -> None:
    app.db.migrate_chat(message.chat.id, message.migrate_to_chat_id)


@router.message(F.migrate_from_chat_id)
async def on_migrate_from(message: Message, app: App) -> None:
    app.db.migrate_chat(message.migrate_from_chat_id, message.chat.id)
