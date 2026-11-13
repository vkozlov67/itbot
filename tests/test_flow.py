"""End-to-end handler tests: real Dispatcher + routers, fake Telegram session and fake xAI."""
import asyncio
import datetime as dt
from pathlib import Path

import pytest
from aiogram import Bot, Dispatcher
from aiogram.client.session.base import BaseSession
from aiogram.exceptions import TelegramBadRequest
from aiogram.methods import GetChat, GetChatMember, GetFile, SendMessage, EditMessageText
from aiogram.types import (
    Audio, Document, Video, VideoNote,
    CallbackQuery, Chat, ChatMemberAdministrator, ChatMemberLeft, ChatMemberMember, ChatMemberUpdated,
    File, Message, MessageEntity, Update, User, Voice,
)

from bot import admin, chat, members
from bot.admin import Panel
from bot.app import App
from bot.config import Config
from bot.db import Database
from bot.apis import ChatResult, Transcript

NOW = dt.datetime(2026, 11, 13, 12, 0, tzinfo=dt.timezone.utc)
ME = User(id=999, is_bot=True, first_name="ItBot", username="it_test_bot", can_read_all_group_messages=False)
OWNER = User(id=1, is_bot=False, first_name="Owner")
STRANGER = User(id=2, is_bot=False, first_name="Stranger")
GROUP = Chat(id=-1001, type="supergroup", title="Team")
OTHER = Chat(id=-1002, type="supergroup", title="Random")


class FakeSession(BaseSession):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.n = 1000

    async def close(self):
        pass

    async def make_request(self, bot, method, timeout=None):
        self.calls.append(method)
        if isinstance(method, (SendMessage, EditMessageText)):
            self.n += 1
            chat_id = method.chat_id
            return Message(message_id=self.n, date=NOW, chat=Chat(id=chat_id, type="private" if chat_id > 0 else "supergroup"),
                           from_user=ME, text=method.text)
        if isinstance(method, GetFile):
            return File(file_id=method.file_id, file_unique_id="u", file_path="voice/file.ogg", file_size=10)
        if isinstance(method, GetChatMember):
            return ChatMemberMember(user=User(id=method.user_id, is_bot=False, first_name="x"))
        if isinstance(method, GetChat):
            raise TelegramBadRequest(method=method, message="test: no get_chat")
        return True

    async def stream_content(self, url, headers=None, timeout=30, chunk_size=65536, raise_for_status=True):
        yield b"OggS-fake-audio"

    def sent(self):
        return [c for c in self.calls if isinstance(c, SendMessage)]


class FakeXAI:
    """Plays both roles: Grok chat and OpenAI transcriber."""

    def __init__(self):
        self.transcribed = []
        self.chats = []

    async def transcribe(self, audio, filename, mime, language=""):
        self.transcribed.append((audio, filename, mime))
        return Transcript(text="Встречаемся завтра в 10 утра у офиса", language="ru")

    async def chat(self, model, messages):
        self.chats.append((model, messages))
        return ChatResult(text="**Завтра в 10:00** у офиса.", input_tokens=50, output_tokens=10)

    async def list_chat_models(self):
        return ["grok-4.7", "grok-4.3"]

    async def close(self):
        pass


@pytest.fixture
def env(tmp_path: Path):
    cfg = Config(bot_token="1:x", xai_api_key="k", openai_api_key="o", owner_id=None, claim_code="secret123",
                 chat_model="grok-4.3", stt_model="gpt-transcribe", stt_language="",
                 xai_base_url="https://api.x.ai/v1", openai_base_url="https://api.openai.com/v1",
                 data_dir=tmp_path, timezone="UTC", context_messages=40, max_parallel_requests=2)
    session = FakeSession()
    bot = Bot("42:TEST", session=session)
    xai = FakeXAI()
    db = Database(tmp_path / "db.sqlite3")
    app = App(cfg=cfg, db=db, xai=xai, stt=xai, me=ME, sem=asyncio.Semaphore(2))
    dp = Dispatcher()
    dp["app"] = app
    dp.include_routers(admin.router, members.router, chat.router)
    return bot, dp, app, session, xai


_uid = iter(range(1, 10_000))


def upd(**kw):
    return Update(update_id=next(_uid), **kw)


def private_msg(user, text, mid=1):
    return Message(message_id=mid, date=NOW, chat=Chat(id=user.id, type="private"), from_user=user, text=text,
                   entities=[MessageEntity(type="bot_command", offset=0, length=len(text.split()[0]))]
                   if text.startswith("/") else None)


def admin_member():
    # Fill every required boolean right with False so the test survives new Bot API fields.
    fields = {name: False for name, f in ChatMemberAdministrator.model_fields.items()
              if f.is_required() and f.annotation is bool}
    fields.update(user=ME, can_manage_chat=True, can_delete_messages=True)
    return ChatMemberAdministrator(**fields)


def added(chat, by, new):
    return upd(my_chat_member=ChatMemberUpdated(chat=chat, from_user=by, date=NOW,
                                                old_chat_member=ChatMemberLeft(user=ME), new_chat_member=new))


def run(coro):
    return asyncio.run(coro)


def test_full_flow(env):
    bot, dp, app, session, xai = env

    async def scenario():
        # 1. stranger cannot claim with a wrong code; owner claims with the right one
        await dp.feed_update(bot, upd(message=private_msg(STRANGER, "/start claim_wrong")))
        assert app.owner_id is None
        await dp.feed_update(bot, upd(message=private_msg(OWNER, "/start claim_secret123")))
        assert app.owner_id == OWNER.id
        await dp.feed_update(bot, upd(message=private_msg(STRANGER, "/start claim_secret123")))
        assert app.owner_id == OWNER.id
        texts = [m.text for m in session.sent()]
        assert any("владелец бота" in t for t in texts)
        assert any("Панель управления" in t for t in texts)

        # 2. owner adds bot as admin -> enabled; stranger adds bot to another chat -> disabled
        session.calls.clear()
        await dp.feed_update(bot, added(GROUP, OWNER, admin_member()))
        assert app.db.get_chat(GROUP.id).enabled
        await dp.feed_update(bot, added(OTHER, STRANGER, ChatMemberMember(user=ME)))
        other = app.db.get_chat(OTHER.id)
        assert other and not other.enabled
        to_owner = [m for m in session.sent() if m.chat_id == OWNER.id]
        assert len(to_owner) == 2 and "не администратор" in to_owner[1].text

        # 3. voice in the enabled group -> transcript reply
        session.calls.clear()
        voice = Voice(file_id="VOICE1", file_unique_id="v1", duration=4, mime_type="audio/ogg", file_size=5000)
        vmsg = Message(message_id=50, date=NOW, chat=GROUP, from_user=STRANGER, voice=voice)
        await dp.feed_update(bot, upd(message=vmsg))
        assert xai.transcribed and xai.transcribed[0][1:] == ("voice.ogg", "audio/ogg")
        sent = session.sent()
        assert len(sent) == 1 and sent[0].text.startswith("🎙 Встречаемся завтра")
        assert sent[0].reply_parameters.message_id == 50

        # 4. question with mention -> answer uses the transcript from history, markdown rendered
        session.calls.clear()
        q = "@it_test_bot во сколько встреча?"
        qmsg = Message(message_id=51, date=NOW, chat=GROUP, from_user=OWNER, text=q,
                       entities=[MessageEntity(type="mention", offset=0, length=12)])
        await dp.feed_update(bot, upd(message=qmsg))
        model, messages = xai.chats[-1]
        assert model == "grok-4.3"
        user_text = messages[1]["content"]
        assert "Встречаемся завтра в 10 утра" in user_text and "во сколько встреча?" in user_text
        sent = session.sent()
        assert len(sent) == 1 and sent[0].text == "<b>Завтра в 10:00</b> у офиса." and sent[0].parse_mode == "HTML"

        # 5. reply to a voice with /ask -> cached transcript used, no second STT call
        session.calls.clear()
        ask = Message(message_id=52, date=NOW, chat=GROUP, from_user=STRANGER, text="/ask кратко", reply_to_message=vmsg,
                      entities=[MessageEntity(type="bot_command", offset=0, length=4)])
        await dp.feed_update(bot, upd(message=ask))
        assert len(xai.transcribed) == 1
        assert "расшифровка" in xai.chats[-1][1][1]["content"]

        # 5b. media is never transcribed: mp3 song, video, audio document, even when asked
        session.calls.clear()
        n_stt = len(xai.transcribed)
        song = Audio(file_id="SONG", file_unique_id="s", duration=200, file_name="song.mp3", mime_type="audio/mpeg")
        clip = Video(file_id="CLIP", file_unique_id="c", width=1, height=1, duration=30, mime_type="video/mp4")
        doc = Document(file_id="DOC", file_unique_id="d", file_name="rec.m4a", mime_type="audio/mp4")
        smsg = Message(message_id=60, date=NOW, chat=GROUP, from_user=STRANGER, audio=song)
        await dp.feed_update(bot, upd(message=smsg))
        await dp.feed_update(bot, upd(message=Message(message_id=61, date=NOW, chat=GROUP, from_user=STRANGER, video=clip)))
        await dp.feed_update(bot, upd(message=Message(message_id=62, date=NOW, chat=GROUP, from_user=STRANGER, document=doc)))
        assert len(xai.transcribed) == n_stt and session.sent() == []
        ask_song = Message(message_id=63, date=NOW, chat=GROUP, from_user=OWNER, text="/ask что тут?", reply_to_message=smsg,
                           entities=[MessageEntity(type="bot_command", offset=0, length=4)])
        await dp.feed_update(bot, upd(message=ask_song))
        assert len(xai.transcribed) == n_stt
        assert "содержимое файла тебе недоступно" in xai.chats[-1][1][1]["content"]
        assert not [c for c in session.calls if isinstance(c, GetFile)]

        # 5c. video note (кружок) is transcribed as mp4, unconverted
        note = VideoNote(file_id="NOTE", file_unique_id="n", length=240, duration=5, file_size=9000)
        await dp.feed_update(bot, upd(message=Message(message_id=64, date=NOW, chat=GROUP, from_user=STRANGER, video_note=note)))
        assert xai.transcribed[-1] == (b"OggS-fake-audio", "video_note.mp4", "video/mp4")

        # 6. disabled chat: voice and mention are ignored
        session.calls.clear()
        n_stt, n_chat = len(xai.transcribed), len(xai.chats)
        await dp.feed_update(bot, upd(message=Message(message_id=7, date=NOW, chat=OTHER, from_user=STRANGER, voice=voice)))
        await dp.feed_update(bot, upd(message=Message(message_id=8, date=NOW, chat=OTHER, from_user=STRANGER, text=q,
                                                      entities=[MessageEntity(type="mention", offset=0, length=12)])))
        assert (len(xai.transcribed), len(xai.chats)) == (n_stt, n_chat)
        assert not [m for m in session.sent() if m.chat_id == OTHER.id]

        # 7. stranger cannot press panel buttons; owner can enable the chat
        panel_msg = Message(message_id=900, date=NOW, chat=Chat(id=OWNER.id, type="private"), from_user=ME, text="x")
        data = Panel(a="tog", c=OTHER.id, v="enabled").pack()
        await dp.feed_update(bot, upd(callback_query=CallbackQuery(id="c1", from_user=STRANGER, chat_instance="i",
                                                                   message=panel_msg, data=data)))
        assert not app.db.get_chat(OTHER.id).enabled
        await dp.feed_update(bot, upd(callback_query=CallbackQuery(id="c2", from_user=OWNER, chat_instance="i",
                                                                   message=panel_msg, data=data)))
        assert app.db.get_chat(OTHER.id).enabled

        # 8. models view + switching model
        await dp.feed_update(bot, upd(callback_query=CallbackQuery(
            id="c3", from_user=OWNER, chat_instance="i", message=panel_msg, data=Panel(a="setm", v="grok-4.7").pack())))
        assert app.chat_model() == "grok-4.7"

        # 9. stranger in private gets nothing from the LLM
        n_chat = len(xai.chats)
        await dp.feed_update(bot, upd(message=private_msg(STRANGER, "привет")))
        assert len(xai.chats) == n_chat

        # 10. owner in private: text is answered; a forwarded song is refused without transcription
        await dp.feed_update(bot, upd(message=private_msg(OWNER, "как дела?", mid=5)))
        assert len(xai.chats) == n_chat + 1
        n_stt = len(xai.transcribed)
        session.calls.clear()
        await dp.feed_update(bot, upd(message=Message(message_id=6, date=NOW, chat=Chat(id=OWNER.id, type="private"),
                                                      from_user=OWNER, audio=song)))
        assert len(xai.transcribed) == n_stt
        assert "только голосовые и кружки" in session.sent()[0].text

        # 11. bot removed -> disabled
        await dp.feed_update(bot, upd(my_chat_member=ChatMemberUpdated(
            chat=GROUP, from_user=OWNER, date=NOW, old_chat_member=admin_member(), new_chat_member=ChatMemberLeft(user=ME))))
        g = app.db.get_chat(GROUP.id)
        assert not g.enabled and not g.active

    run(scenario())


def test_create_bot_like_production():
    from bot.__main__ import create_bot

    async def run():
        b = create_bot("42:TEST")
        assert b.default.link_preview_is_disabled is True
        await b.session.close()

    run_ = asyncio.run(run())
