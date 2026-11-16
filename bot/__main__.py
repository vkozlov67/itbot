from __future__ import annotations

import asyncio
import logging
import socket
import sys

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.methods import GetUpdates

from . import admin, chat, members, summary
from .app import App
from .config import Config
from .db import Database
from .apis import ElevenLabsTTS, OpenAITranscriber, XAIClient

log = logging.getLogger("itbot")


POLL_GRACE = 10  # seconds a long poll may run past its own timeout before we give up on the connection


class TelegramSession(AiohttpSession):
    """IPv4 only, and a hung long poll is dropped after ~20 s instead of the default ~70 s.

    On this host a pooled connection to Telegram sometimes stalls silently; with aiogram's defaults
    the bot then receives nothing for over a minute and gets all messages in one late batch.
    """

    def __init__(self) -> None:
        super().__init__()
        self._connector_init["family"] = socket.AF_INET

    async def make_request(self, bot, method, timeout=None):
        if isinstance(method, GetUpdates):
            timeout = (method.timeout or 0) + POLL_GRACE
        return await super().make_request(bot, method, timeout)


def create_bot(token: str) -> Bot:
    return Bot(token, session=TelegramSession(), default=DefaultBotProperties(link_preview_is_disabled=True))


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("aiogram.event").setLevel(logging.WARNING)

    cfg = Config.from_env()
    db = Database(cfg.data_dir / "itbot.sqlite3")
    xai = XAIClient(cfg.xai_api_key, cfg.xai_base_url)
    stt = OpenAITranscriber(cfg.openai_api_key, cfg.openai_base_url, cfg.stt_model)
    tts = (ElevenLabsTTS(cfg.elevenlabs_api_key, model=cfg.elevenlabs_model, voice_id=cfg.elevenlabs_voice_id)
           if cfg.elevenlabs_api_key else None)
    bot = create_bot(cfg.bot_token)
    try:
        me = await bot.get_me()
        app = App(cfg=cfg, db=db, xai=xai, stt=stt, me=me, sem=asyncio.Semaphore(cfg.max_parallel_requests),
                  tts=tts)
        log.info("started as @%s (id %s), chat model %s, stt model %s, tts %s, group privacy %s",
                 me.username, me.id, app.chat_model(), cfg.stt_model,
                 f"ElevenLabs {cfg.elevenlabs_model}" if tts else f"OpenAI {cfg.tts_model}",
                 "off" if me.can_read_all_group_messages else "ON (non-admin groups see only commands/replies)")
        if app.owner_id is None:
            log.warning("OWNER NOT SET. Open this link in Telegram to become the owner: %s", app.claim_link())

        dp = Dispatcher()
        dp["app"] = app
        dp.include_routers(admin.router, members.router, chat.router)
        await admin.setup_commands(bot, app)
        await admin.check_chats_on_start(bot, app)
        await bot.delete_webhook(drop_pending_updates=False)
        scheduler = asyncio.create_task(summary.scheduler(bot, app))
        try:
            await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())
        finally:
            scheduler.cancel()
    finally:
        await xai.close()
        await stt.close()
        if tts:
            await tts.close()
        await bot.session.close()
        db.close()


if __name__ == "__main__":
    asyncio.run(main())
