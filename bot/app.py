from __future__ import annotations

import asyncio
import logging
import secrets
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field

from aiogram import Bot
from aiogram.exceptions import TelegramAPIError, TelegramBadRequest
from aiogram.types import InlineKeyboardMarkup, Message, ReplyParameters, User

from .config import Config
from .db import Database
from .apis import OpenAITranscriber, XAIClient

log = logging.getLogger(__name__)


class RateLimiter:
    def __init__(self, limit: int, window: float):
        self.limit = limit
        self.window = window
        self._hits: dict[tuple[int, int], deque[float]] = defaultdict(deque)

    def allow(self, key: tuple[int, int]) -> bool:
        now = time.monotonic()
        hits = self._hits[key]
        while hits and now - hits[0] > self.window:
            hits.popleft()
        if len(hits) >= self.limit:
            return False
        hits.append(now)
        return True


@dataclass
class App:
    cfg: Config
    db: Database
    xai: XAIClient
    stt: OpenAITranscriber
    me: User
    sem: asyncio.Semaphore
    limiter: RateLimiter = field(default_factory=lambda: RateLimiter(limit=8, window=60))
    _media_locks: dict[tuple[int, int], asyncio.Lock] = field(default_factory=dict)
    _last_alert: dict[str, float] = field(default_factory=dict)

    # ---------- owner ----------
    @property
    def owner_id(self) -> int | None:
        if self.cfg.owner_id:
            return self.cfg.owner_id
        raw = self.db.get_setting("owner_id")
        return int(raw) if raw else None

    def is_owner(self, user: User | None) -> bool:
        return user is not None and self.owner_id is not None and user.id == self.owner_id

    def claim_code(self) -> str:
        if self.cfg.claim_code:
            return self.cfg.claim_code
        code = self.db.get_setting("claim_code")
        if not code:
            code = secrets.token_urlsafe(12).replace("-", "x").replace("_", "y")
            self.db.set_setting("claim_code", code)
        return code

    def claim_link(self) -> str:
        return f"https://t.me/{self.me.username}?start=claim_{self.claim_code()}"

    def add_to_group_link(self) -> str:
        # Opens Telegram's "add to group" picker and pre-selects admin rights for the bot.
        return f"https://t.me/{self.me.username}?startgroup=itbot&admin=delete_messages+pin_messages"

    def chat_model(self) -> str:
        return self.db.get_setting("chat_model") or self.cfg.chat_model

    def media_lock(self, chat_id: int, message_id: int) -> asyncio.Lock:
        key = (chat_id, message_id)
        lock = self._media_locks.get(key)
        if lock is None:
            if len(self._media_locks) > 500:
                for k in [k for k, v in self._media_locks.items() if not v.locked()]:
                    del self._media_locks[k]
            lock = self._media_locks[key] = asyncio.Lock()
        return lock

    async def notify_owner(
        self, bot: Bot, text: str, markup: InlineKeyboardMarkup | None = None, *, throttle_key: str | None = None
    ) -> None:
        owner = self.owner_id
        if owner is None:
            return
        if throttle_key:
            now = time.monotonic()
            if now - self._last_alert.get(throttle_key, 0) < 3600:
                return
            self._last_alert[throttle_key] = now
        try:
            await bot.send_message(owner, text, parse_mode="HTML", reply_markup=markup)
        except TelegramAPIError as exc:
            log.warning("cannot notify owner: %s", exc)


def thread_id(message: Message) -> int | None:
    return message.message_thread_id if message.is_topic_message else None


async def send_reply(
    bot: Bot, message: Message, text: str, *, html: bool = True, fallback: str | None = None
) -> Message:
    """Reply to a message; survives deleted originals and falls back to plain text on bad HTML."""
    params: dict = dict(
        chat_id=message.chat.id,
        reply_parameters=ReplyParameters(message_id=message.message_id, allow_sending_without_reply=True),
    )
    if thread_id(message):
        params["message_thread_id"] = thread_id(message)
    if html:
        try:
            return await bot.send_message(text=text, parse_mode="HTML", **params)
        except TelegramBadRequest as exc:
            if "parse" not in str(exc).lower() and "entit" not in str(exc).lower():
                raise
            log.warning("HTML rejected by Telegram, sending plain text: %s", exc)
            text = fallback if fallback is not None else text
    return await bot.send_message(text=text, parse_mode=None, **params)
