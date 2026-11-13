from __future__ import annotations

import asyncio
import logging
import sys

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties

from . import admin, chat, members
from .app import App
from .config import Config
from .db import Database
from .apis import OpenAITranscriber, XAIClient

log = logging.getLogger("itbot")


def create_bot(token: str) -> Bot:
    return Bot(token, default=DefaultBotProperties(link_preview_is_disabled=True))


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
    bot = create_bot(cfg.bot_token)
    try:
        me = await bot.get_me()
        app = App(cfg=cfg, db=db, xai=xai, stt=stt, me=me, sem=asyncio.Semaphore(cfg.max_parallel_requests))
        log.info("started as @%s (id %s), chat model %s, stt model %s, group privacy %s",
                 me.username, me.id, app.chat_model(), cfg.stt_model,
                 "off" if me.can_read_all_group_messages else "ON (non-admin groups see only commands/replies)")
        if app.owner_id is None:
            log.warning("OWNER NOT SET. Open this link in Telegram to become the owner: %s", app.claim_link())

        dp = Dispatcher()
        dp["app"] = app
        dp.include_routers(admin.router, members.router, chat.router)
        await admin.setup_commands(bot, app)
        await admin.check_chats_on_start(bot, app)
        await bot.delete_webhook(drop_pending_updates=False)
        await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())
    finally:
        await xai.close()
        await stt.close()
        await bot.session.close()
        db.close()


if __name__ == "__main__":
    asyncio.run(main())
