"""End-to-end handler tests: real Dispatcher + routers, fake Telegram session and fake xAI."""
import asyncio
import datetime as dt
from pathlib import Path

import pytest
from aiogram import Bot, Dispatcher
from aiogram.client.session.base import BaseSession
from aiogram.exceptions import TelegramBadRequest
from aiogram.methods import (
    AnswerCallbackQuery, DeleteMessage, EditMessageText, GetChat, GetChatMember, GetFile, SendDocument, SendMessage, SendPhoto, SendVideo, SendVoice,
)
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
from bot.apis import ChatResult, Source, Transcript
from bot.downloader import Download, DownloadError, Item

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
        self.fail_answer = False
        self.member = None  # what GetChatMember returns (default: a plain member)

    async def close(self):
        pass

    async def make_request(self, bot, method, timeout=None):
        self.calls.append(method)
        if self.fail_answer and isinstance(method, AnswerCallbackQuery):
            raise TelegramBadRequest(method=method, message="Bad Request: query is too old")
        if isinstance(method, (SendMessage, EditMessageText, SendVideo, SendPhoto, SendDocument, SendVoice)):
            self.n += 1
            chat_id = method.chat_id
            return Message(message_id=self.n, date=NOW, chat=Chat(id=chat_id, type="private" if chat_id > 0 else "supergroup"),
                           from_user=ME, text=getattr(method, "text", None))
        if isinstance(method, GetFile):
            return File(file_id=method.file_id, file_unique_id="u", file_path="voice/file.ogg", file_size=10)
        if isinstance(method, GetChatMember):
            if self.member is not None:
                return self.member
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
        self.efforts = []

    async def transcribe(self, audio, filename, mime, language=""):
        self.transcribed.append((audio, filename, mime))
        return Transcript(text="Встречаемся завтра в 10 утра у офиса", language="ru")

    last_search = None

    delay = 0.0

    async def respond(self, model, items, search=True, max_turns=5, effort=None):
        if self.delay:
            await asyncio.sleep(self.delay)
        self.chats.append((model, items))
        self.efforts.append(effort)
        self.last_search = search
        if self.queued:
            return self.queued.pop(0)
        return self.next_result

    queued: list = []  # results returned first, one per call (then next_result)

    next_result = ChatResult(text="**Завтра в 10:00** у офиса.", input_tokens=50, output_tokens=10)

    async def chat(self, model, messages):  # fallback path, unused when respond works
        raise AssertionError("chat completions fallback should not be used")

    async def list_chat_models(self):
        return ["grok-4.7", "grok-4.3"]

    images = None
    image_error = None

    async def image(self, model, prompt, source=None, aspect_ratio=None):
        if self.images is None:
            self.images = []
        self.images.append((model, prompt, source, aspect_ratio))
        if self.image_error is not None:
            raise self.image_error
        return b"\xff\xd8fake-jpeg"

    videos = None
    spoken = None
    speech_error = None

    async def video(self, model, prompt, image=None, duration=6, aspect_ratio=None, **kw):
        self.videos = (self.videos or []) + [(model, prompt, image, duration, aspect_ratio)]
        return b"\x00\x00\x00\x18ftypmp42fake"

    async def speech(self, text, *, model, voice, instructions=""):
        self.spoken = (self.spoken or []) + [(text, model, voice)]
        if self.speech_error is not None:
            raise self.speech_error
        return b"OggS-fake-opus"

    async def close(self):
        pass


@pytest.fixture
def env(tmp_path: Path):
    cfg = Config(bot_token="1:x", xai_api_key="k", openai_api_key="o", owner_id=None, claim_code="secret123",
                 chat_model="grok-4.3", stt_model="gpt-transcribe", stt_language="",
                 xai_base_url="https://api.x.ai/v1", openai_base_url="https://api.openai.com/v1",
                 data_dir=tmp_path, timezone="UTC", summary_time="21:00", context_messages=40, max_parallel_requests=2)
    session = FakeSession()
    bot = Bot("42:TEST", session=session)
    xai = FakeXAI()
    db = Database(tmp_path / "db.sqlite3")
    app = App(cfg=cfg, db=db, xai=xai, stt=xai, me=ME, sem=asyncio.Semaphore(2))
    dp = Dispatcher()
    dp["app"] = app
    routers = (admin.router, members.router, chat.router)
    for r in routers:  # module-level routers: detach from a previous test's dispatcher
        r._parent_router = None
    dp.include_routers(*routers)
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
        assert len(sent) == 1 and sent[0].text == (
            "<blockquote expandable>Встречаемся завтра в 10 утра у офиса</blockquote>")
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
        data = Panel(a="set", c=OTHER.id, v="enabled=1").pack()
        await dp.feed_update(bot, upd(callback_query=CallbackQuery(id="c1", from_user=STRANGER, chat_instance="i",
                                                                   message=panel_msg, data=data)))
        assert not app.db.get_chat(OTHER.id).enabled
        # pressing "Включить" several times (delayed updates) keeps it on; a late answer error doesn't abort it
        session.fail_answer = True
        for i in range(4):
            await dp.feed_update(bot, upd(callback_query=CallbackQuery(id=f"c2{i}", from_user=OWNER, chat_instance="i",
                                                                       message=panel_msg, data=data)))
        session.fail_answer = False
        assert app.db.get_chat(OTHER.id).enabled
        # legacy toggle buttons no longer flip anything
        await dp.feed_update(bot, upd(callback_query=CallbackQuery(
            id="c2x", from_user=OWNER, chat_instance="i", message=panel_msg,
            data=Panel(a="tog", c=OTHER.id, v="enabled").pack())))
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
        import socket
        from aiogram.client.session.aiohttp import AiohttpSession
        from aiogram.methods import GetUpdates, SendMessage
        b = create_bot("42:TEST")
        assert b.default.link_preview_is_disabled is True
        assert b.session._connector_init["family"] == socket.AF_INET
        seen = {}

        async def fake(self, bot, method, timeout=None):
            seen[type(method).__name__] = timeout
            return []

        orig = AiohttpSession.make_request
        AiohttpSession.make_request = fake
        try:
            # aiogram asks for session.timeout + polling_timeout (70 s); a long poll is capped at 20 s
            await b.session.make_request(b, GetUpdates(timeout=10), timeout=70)
            await b.session.make_request(b, SendMessage(chat_id=1, text="x"), timeout=None)
        finally:
            AiohttpSession.make_request = orig
        assert seen == {"GetUpdates": 20, "SendMessage": None}
        await b.session.close()

    run_ = asyncio.run(run())


def test_reply_mention_search_and_link_download(env, monkeypatch, tmp_path):
    bot, dp, app, session, xai = env
    from aiogram.methods import SendVideo, SetMessageReaction

    downloaded = []

    fail_next = []

    async def fake_download(link, cookies=None):
        downloaded.append(link)
        if fail_next:
            fail_next.pop()
            raise DownloadError("failed", "boom")
        f = tmp_path / "reel.mp4"
        f.write_bytes(b"mp4")
        return Download(link=link, title="Funny reel", uploader="someone",
                        items=[Item(path=f, is_image=False, width=720, height=1280, duration=5)],
                        description=desc.pop(0) if desc else "")

    desc = ["Подпись <b>под</b> рилсом #тег"]

    monkeypatch.setattr(chat, "download", fake_download)

    async def scenario():
        app.db.set_setting("owner_id", str(OWNER.id))
        await dp.feed_update(bot, added(GROUP, OWNER, admin_member()))

        # 1. tag the bot in a reply to someone else's message -> bot reads that message
        session.calls.clear()
        original = Message(message_id=70, date=NOW, chat=GROUP, from_user=STRANGER,
                           text="Кто знает, во сколько закрывается МФЦ на Тверской?")
        await dp.feed_update(bot, upd(message=original))
        tag = Message(message_id=71, date=NOW, chat=GROUP, from_user=OWNER, text="@it_test_bot",
                      reply_to_message=original, entities=[MessageEntity(type="mention", offset=0, length=12)])
        await dp.feed_update(bot, upd(message=tag))
        user_text = xai.chats[-1][1][1]["content"]
        assert "во сколько закрывается МФЦ" in user_text and "в ответ на сообщение от Stranger" in user_text
        assert session.sent()[-1].reply_parameters.message_id == 71

        # 2. "погугли" -> answer comes with sources line
        xai.next_result = ChatResult(text="Главные новости США: ...", input_tokens=900, output_tokens=60,
                                     sources=(Source("https://www.reuters.com/world/us/"), Source("https://apnews.com/x")),
                                     searches=2)
        session.calls.clear()
        q = "@it_test_bot погугли новости в США"
        await dp.feed_update(bot, upd(message=Message(message_id=72, date=NOW, chat=GROUP, from_user=OWNER, text=q,
                                                      entities=[MessageEntity(type="mention", offset=0, length=12)])))
        out = session.sent()[-1]
        assert "Главные новости США" in out.text
        assert '<a href="https://www.reuters.com/world/us/">reuters.com</a>' in out.text and "apnews.com" in out.text
        system = xai.chats[-1][1][0]["content"]
        assert "web_search" in system  # system prompt tells Grok it can search
        assert "женском роде" in system and "ItBot, девушка" in system

        # 3. a Reels link is downloaded and sent as a video reply; no LLM call for a bare link
        session.calls.clear()
        n_chat = len(xai.chats)
        url = "https://www.instagram.com/reel/Chunk8-jurw/"
        await dp.feed_update(bot, upd(message=Message(message_id=73, date=NOW, chat=GROUP, from_user=STRANGER, text=url,
                                                      entities=[MessageEntity(type="url", offset=0, length=len(url))])))
        assert [l.service for l in downloaded] == ["Instagram"]
        videos = [c for c in session.calls if isinstance(c, SendVideo)]
        assert len(videos) == 1 and videos[0].reply_parameters.message_id == 73 and videos[0].width == 720
        assert "Instagram" in videos[0].caption and "someone" in videos[0].caption
        assert "<blockquote expandable>Подпись &lt;b&gt;под&lt;/b&gt; рилсом #тег</blockquote>" in videos[0].caption
        # the post text is remembered, so the bot can answer questions about the reel
        assert any(m.kind == "video" and "Подпись" in m.text and "someone" in m.text
                   for m in app.db.recent_messages(GROUP.id, 10))
        reactions = [c.reaction for c in session.calls if isinstance(c, SetMessageReaction)]
        assert reactions[0][0].emoji == "👀" and reactions[-1] == []
        assert len(xai.chats) == n_chat

        # 3b. no delete right (above): the original stays and the owner is told once
        assert any("Удаление сообщений" in m.text for m in session.sent() if m.chat_id == OWNER.id)
        assert not [c for c in session.calls if isinstance(c, DeleteMessage)]

        # 3c. with the delete right: media replaces the link message, says who shared it + their comment,
        #     keeps the reply chain, and the original is deleted
        session.member = admin_member()
        session.calls.clear()
        text = f"{url} смотри"
        root = Message(message_id=60, date=NOW, chat=GROUP, from_user=OWNER, text="старое")
        await dp.feed_update(bot, upd(message=Message(
            message_id=75, date=NOW, chat=GROUP, from_user=STRANGER, text=text, reply_to_message=root,
            entities=[MessageEntity(type="url", offset=0, length=len(url))])))
        videos = [c for c in session.calls if isinstance(c, SendVideo)]
        assert len(videos) == 1 and videos[0].reply_parameters.message_id == 60
        assert f"от <b>{STRANGER.full_name}</b>: смотри" in videos[0].caption
        deletes = [c for c in session.calls if isinstance(c, DeleteMessage)]
        assert [(d.chat_id, d.message_id) for d in deletes] == [(GROUP.id, 75)]

        # 3d. a failed download keeps the original; a question to the bot with a link is never deleted
        session.calls.clear()
        fail_next.append(True)
        await dp.feed_update(bot, upd(message=Message(message_id=76, date=NOW, chat=GROUP, from_user=STRANGER, text=url,
                                                      entities=[MessageEntity(type="url", offset=0, length=len(url))])))
        q = f"@it_test_bot что тут {url}"
        await dp.feed_update(bot, upd(message=Message(
            message_id=77, date=NOW, chat=GROUP, from_user=STRANGER, text=q,
            entities=[MessageEntity(type="mention", offset=0, length=12),
                      MessageEntity(type="url", offset=q.index("http"), length=len(url))])))
        assert not [c for c in session.calls if isinstance(c, DeleteMessage)]
        assert [c for c in session.calls if isinstance(c, SendVideo)][-1].reply_parameters.message_id == 77
        session.member = None

        # 4. download toggle off -> link ignored
        app.db.set_flag(GROUP.id, "download", False)
        await dp.feed_update(bot, upd(message=Message(message_id=74, date=NOW, chat=GROUP, from_user=STRANGER, text=url,
                                                      entities=[MessageEntity(type="url", offset=0, length=len(url))])))
        assert len(downloaded) == 4

    run(scenario())


def test_draw_and_edit_pictures(env):
    from aiogram.methods import SendPhoto
    from aiogram.types import PhotoSize
    from bot import draw as draw_mod
    from bot.apis import APIError

    bot, dp, app, session, xai = env

    def mention(mid, text, user=STRANGER, **kw):
        return upd(message=Message(message_id=mid, date=NOW, chat=GROUP, from_user=user, text=text,
                                   entities=[MessageEntity(type="mention", offset=0, length=12)], **kw))

    async def scenario():
        app.db.set_setting("owner_id", str(OWNER.id))
        await dp.feed_update(bot, added(GROUP, OWNER, admin_member()))
        n_chat = len(xai.chats)

        # 1. "@bot нарисуй …" -> picture as a reply, no chat-model call, counted in usage
        session.calls.clear()
        await dp.feed_update(bot, mention(90, "@it_test_bot нарисуй кота-космонавта, вертикально"))
        photos = [c for c in session.calls if isinstance(c, SendPhoto)]
        assert len(photos) == 1 and photos[0].reply_parameters.message_id == 90
        assert "кота-космонавта" in photos[0].caption
        model, prompt, source, ratio = xai.images[-1]
        assert model == app.cfg.image_model and source is None and ratio == "9:16"
        assert len(xai.chats) == n_chat
        assert app.db.images_today(GROUP.id) == 1

        # 2. reply to a photo: "переделай в аниме" -> edit with the photo as a data URI
        pic = Message(message_id=91, date=NOW, chat=GROUP, from_user=OWNER,
                      photo=[PhotoSize(file_id="p1", file_unique_id="u1", width=800, height=600, file_size=10)])
        await dp.feed_update(bot, mention(92, "@it_test_bot переделай в стиле аниме", reply_to_message=pic))
        assert xai.images[-1][2].startswith("data:image/jpeg;base64,")

        # 3. ordinary questions and "как нарисовать" are answered by the chat model, not drawn
        before = len(xai.images)
        await dp.feed_update(bot, mention(93, "@it_test_bot как нарисовать кота?"))
        await dp.feed_update(bot, mention(94, "@it_test_bot сделай саммари"))
        assert len(xai.images) == before

        # 4. /draw command, and a moderation refusal is explained
        xai.image_error = APIError("xAI", 400, "Generated image rejected by content moderation")
        session.calls.clear()
        await dp.feed_update(bot, upd(message=Message(message_id=95, date=NOW, chat=GROUP, from_user=STRANGER,
                                                      text="/draw что-то запрещённое")))
        xai.image_error = None
        assert any("фильтр" in m.text for m in session.sent())

        # 5. per-user limit for non-owners (the owner is not limited)
        session.calls.clear()
        for i in range(draw_mod.USER_LIMIT + 1):
            await dp.feed_update(bot, mention(100 + i, f"@it_test_bot нарисуй собаку {i}"))
        assert any("за час" in m.text for m in session.sent())
        session.calls.clear()
        await dp.feed_update(bot, mention(120, "@it_test_bot нарисуй закат", user=OWNER))
        assert [c for c in session.calls if isinstance(c, SendPhoto)]

    draw_mod._user_log.clear()
    run(scenario())


def test_summary_command_trigger_and_daily(env):
    import dataclasses
    from datetime import datetime, timezone as tz
    from bot import summary

    bot, dp, app, session, xai = env

    async def scenario():
        app.db.set_setting("owner_id", str(OWNER.id))
        await dp.feed_update(bot, added(GROUP, OWNER, admin_member()))
        mid = 100
        for who, text in [(STRANGER, "Пятница — идём в бар?"), (OWNER, "Да, в 19:00"), (STRANGER, "Беру Васю"),
                          (OWNER, "Ок, бронь на 4"), (STRANGER, "Ссылка: https://bar.example")]:
            mid += 1
            await dp.feed_update(bot, upd(message=Message(message_id=mid, date=datetime.now(tz.utc), chat=GROUP,
                                                          from_user=who, text=text)))
        xai.next_result = ChatResult(text="**🗞 Итоги дня**\n- Бар в пятницу в 19:00, бронь на 4", input_tokens=500,
                                     output_tokens=40)

        # /summary -> reply with summary, no web search, history included
        session.calls.clear()
        cmd = Message(message_id=200, date=datetime.now(tz.utc), chat=GROUP, from_user=STRANGER, text="/summary",
                      entities=[MessageEntity(type="bot_command", offset=0, length=8)])
        await dp.feed_update(bot, upd(message=cmd))
        assert xai.last_search is False
        prompt, user = xai.chats[-1][1][0]["content"], xai.chats[-1][1][1]["content"]
        assert "саммари" in prompt and "женском роде" in prompt and "Беру Васю" in user and "бронь на 4" in user
        out = session.sent()[-1]
        assert out.reply_parameters.message_id == 200 and "<b>🗞 Итоги дня</b>" in out.text

        # a second request from a non-owner within the cooldown is refused politely
        n = len(xai.chats)
        await dp.feed_update(bot, upd(message=cmd.model_copy(update={"message_id": 201})))
        assert len(xai.chats) == n and "недавно" in session.sent()[-1].text

        # natural language trigger from the owner, with hours
        q = "@it_test_bot что было в чате за 6 часов?"
        await dp.feed_update(bot, upd(message=Message(message_id=202, date=datetime.now(tz.utc), chat=GROUP,
                                                      from_user=OWNER, text=q,
                                                      entities=[MessageEntity(type="mention", offset=0, length=12)])))
        assert len(xai.chats) == n + 1 and "последние 6 ч" in xai.chats[-1][1][1]["content"]

        # an ordinary question is not a summary
        await dp.feed_update(bot, upd(message=Message(message_id=203, date=datetime.now(tz.utc), chat=GROUP,
                                                      from_user=OWNER, text="@it_test_bot сколько будет 2+2",
                                                      entities=[MessageEntity(type="mention", offset=0, length=12)])))
        assert xai.last_search is True

        # daily summary: off by default; when on, posted once inside the window
        now = datetime.now(tz.utc)
        app.cfg = dataclasses.replace(app.cfg, summary_time=now.strftime("%H:%M"))
        session.calls.clear()
        await summary.daily_tick(bot, app)
        assert session.sent() == []
        app.db.set_flag(GROUP.id, "daily_summary", True)
        await summary.daily_tick(bot, app)
        await summary.daily_tick(bot, app)
        posted = [m for m in session.sent() if m.chat_id == GROUP.id]
        assert len(posted) == 1 and posted[0].reply_parameters is None

    run(scenario())


def test_vision_voice_video_stats_weekly(env):
    import dataclasses
    from datetime import datetime, timezone as tz
    from aiogram.types import PhotoSize, Sticker
    from bot import summary
    from bot import draw as draw_mod
    from bot.apis import APIError

    bot, dp, app, session, xai = env

    def mention(mid, text, user=STRANGER, **kw):
        return upd(message=Message(message_id=mid, date=datetime.now(tz.utc), chat=GROUP, from_user=user, text=text,
                                   entities=[MessageEntity(type="mention", offset=0, length=12)], **kw))

    async def scenario():
        app.db.set_setting("owner_id", str(OWNER.id))
        await dp.feed_update(bot, added(GROUP, OWNER, admin_member()))

        # 1. vision: a screenshot sent as a PNG file is passed to Grok as an image
        doc = Message(message_id=300, date=NOW, chat=GROUP, from_user=OWNER,
                      document=Document(file_id="d1", file_unique_id="ud1", mime_type="image/png", file_size=10))
        await dp.feed_update(bot, mention(301, "@it_test_bot что тут написано?", reply_to_message=doc))
        content = xai.chats[-1][1][1]["content"]
        assert content[1]["image_url"].startswith("data:image/png;base64,")
        assert "картинка-файл, изображение приложено" in content[0]["text"]
        assert xai.last_search is False  # a picture question does not search unless asked to
        assert "поиск в интернете отключён" in xai.chats[-1][1][0]["content"]
        assert "скриншот" in xai.chats[-1][1][0]["content"]  # the system prompt knows it can see pictures
        # a sticker that cannot be converted is described instead of crashing
        st = Message(message_id=302, date=NOW, chat=GROUP, from_user=OWNER,
                     sticker=Sticker(file_id="s1", file_unique_id="us1", type="regular", width=512, height=512,
                                     is_animated=False, is_video=False, file_size=10))
        await dp.feed_update(bot, mention(303, "@it_test_bot что за стикер?", reply_to_message=st))
        assert "стикер" in str(xai.chats[-1][1][1]["content"])

        # 2. "ответь голосом" -> voice reply with the text attached, short spoken style requested
        session.calls.clear()
        xai.next_result = ChatResult(text="Юрий **Гагарин** — первый человек в космосе.", input_tokens=5, output_tokens=5)
        await dp.feed_update(bot, mention(310, "@it_test_bot ответь голосом, кто такой Гагарин"))
        voices = [c for c in session.calls if isinstance(c, SendVoice)]
        assert len(voices) == 1 and voices[0].reply_parameters.message_id == 310
        assert "<b>Гагарин</b>" in voices[0].caption and not session.sent()
        assert xai.spoken[-1][0] == "Юрий Гагарин — первый человек в космосе."
        assert "озвучен" in xai.chats[-1][1][0]["content"]
        # TTS failure -> plain text answer
        xai.speech_error = APIError("OpenAI", 500, "boom")
        session.calls.clear()
        await dp.feed_update(bot, mention(311, "@it_test_bot расскажи анекдот голосом"))
        xai.speech_error = None
        assert not [c for c in session.calls if isinstance(c, SendVoice)] and session.sent()

        # 3. "озвучь" in reply to a message reads it aloud (no chat-model call, no duplicate text)
        n = len(xai.chats)
        src = Message(message_id=320, date=NOW, chat=GROUP, from_user=OWNER, text="Встреча завтра в 10:00")
        session.calls.clear()
        await dp.feed_update(bot, mention(321, "@it_test_bot озвучь", reply_to_message=src))
        voices = [c for c in session.calls if isinstance(c, SendVoice)]
        assert len(voices) == 1 and voices[0].caption is None and len(xai.chats) == n
        assert xai.spoken[-1][0] == "Встреча завтра в 10:00"

        # 4. "оживи" on a photo -> video from the photo, wait note removed, usage counted
        pic = Message(message_id=330, date=NOW, chat=GROUP, from_user=OWNER,
                      photo=[PhotoSize(file_id="p1", file_unique_id="u1", width=800, height=600, file_size=10)])
        session.calls.clear()
        await dp.feed_update(bot, mention(331, "@it_test_bot оживи, пусть улыбнётся", reply_to_message=pic))
        vids = [c for c in session.calls if isinstance(c, SendVideo)]
        assert len(vids) == 1 and vids[0].reply_parameters.message_id == 331 and "улыбнётся" in vids[0].caption
        model, prompt, image, seconds, ratio = xai.videos[-1]
        assert model == app.cfg.video_model and image.startswith("data:image/jpeg") and seconds == 6 and ratio is None
        assert [c for c in session.calls if isinstance(c, DeleteMessage)]  # "снимаю видео…" note
        assert app.db.chat_usage_since(GROUP.id, "2000-01-01").videos == 1
        # text-to-video and the per-user video limit
        session.calls.clear()
        for i in range(draw_mod.VIDEO_USER_LIMIT + 1):
            await dp.feed_update(bot, mention(340 + i, f"@it_test_bot сними видео про море {i}, вертикально"))
        assert xai.videos[-1][2] is None and xai.videos[-1][4] == "9:16"
        assert any("за час" in m.text for m in session.sent())

        # 5. /stats in the chat
        session.calls.clear()
        await dp.feed_update(bot, upd(message=Message(message_id=350, date=datetime.now(tz.utc), chat=GROUP,
                                                      from_user=STRANGER, text="/stats")))
        out = session.sent()[-1]
        assert "Статистика чата за неделю" in out.text and "Stranger" in out.text and "сняла 2 видео" in out.text

        # 6. weekly post: off by default; when on, stats + summary once a week, daily skipped that evening
        for i in range(6):
            await dp.feed_update(bot, upd(message=Message(message_id=360 + i, date=datetime.now(tz.utc), chat=GROUP,
                                                          from_user=STRANGER, text=f"обсуждаем план {i}")))
        now = datetime.now(tz.utc)
        app.cfg = dataclasses.replace(app.cfg, summary_time=now.strftime("%H:%M"), weekly_day=now.weekday())
        xai.next_result = ChatResult(text="**🗞 Итоги недели**\n- план", input_tokens=5, output_tokens=5)
        session.calls.clear()
        await summary.weekly_tick(bot, app)
        assert session.sent() == []
        app.db.set_flag(GROUP.id, "weekly_summary", True)
        app.db.set_flag(GROUP.id, "daily_summary", True)
        await summary.weekly_tick(bot, app)
        await summary.weekly_tick(bot, app)
        await summary.daily_tick(bot, app)
        posted = [m.text for m in session.sent() if m.chat_id == GROUP.id]
        assert len(posted) == 2 and "Статистика" in posted[0] and "Итоги недели" in posted[1]
        assert "Сообщение недели" in xai.chats[-1][1][0]["content"]

    draw_mod._video_log.clear()
    run(scenario())


class FakeEleven:
    voice_id, voice_name = "", ""

    def __init__(self):
        self.spoken = []
        self.error = None

    async def speech(self, text):
        self.spoken.append(text)
        if self.error is not None:
            raise self.error
        return b"ID3fake-mp3"


def test_voice_question_gets_voice_answer(env, monkeypatch):
    from bot import voice as voice_mod
    from bot.apis import APIError

    bot, dp, app, session, xai = env
    eleven = FakeEleven()
    app.tts = eleven
    converted = []

    async def fake_ogg(audio):
        converted.append(audio)
        return b"OggS-from-mp3"

    monkeypatch.setattr(voice_mod, "to_ogg_opus", fake_ogg)
    v = Voice(file_id="VQ", file_unique_id="vq", duration=3, mime_type="audio/ogg", file_size=500)
    mine = Message(message_id=400, date=NOW, chat=GROUP, from_user=ME, text="Чем помочь?")

    def voice_reply(mid):
        return upd(message=Message(message_id=mid, date=NOW, chat=GROUP, from_user=STRANGER, voice=v,
                                   reply_to_message=mine))

    async def scenario():
        app.db.set_setting("owner_id", str(OWNER.id))
        await dp.feed_update(bot, added(GROUP, OWNER, admin_member()))
        assert app.db.get_chat(GROUP.id).voice_replies  # on by default

        # 1. voice in reply to the bot -> transcript, then a voice answer via ElevenLabs (MP3 -> OGG)
        session.calls.clear()
        xai.next_result = ChatResult(text="Да, **в 10** у офиса.", input_tokens=5, output_tokens=5)
        await dp.feed_update(bot, voice_reply(401))
        voices = [c for c in session.calls if isinstance(c, SendVoice)]
        assert len(voices) == 1 and voices[0].reply_parameters.message_id == 401
        assert "<b>в 10</b>" in voices[0].caption
        assert eleven.spoken == ["Да, в 10 у офиса."] and converted == [b"ID3fake-mp3"]
        assert not xai.spoken  # OpenAI TTS not used
        assert "озвучен" in xai.chats[-1][1][0]["content"]
        assert len(session.sent()) == 1  # only the transcript as text

        # 2. ElevenLabs fails -> OpenAI voice instead, owner warned about the key
        eleven.error = APIError("ElevenLabs", 401, "invalid_api_key")
        session.calls.clear()
        await dp.feed_update(bot, voice_reply(402))
        assert len([c for c in session.calls if isinstance(c, SendVoice)]) == 1 and xai.spoken
        assert any("ElevenLabs" in (m.text or "") and m.chat_id == OWNER.id for m in session.sent())
        eleven.error = None

        # 3. the panel switch turns it off -> plain text answer
        app.db.set_flag(GROUP.id, "voice_replies", False)
        session.calls.clear()
        await dp.feed_update(bot, voice_reply(403))
        assert not [c for c in session.calls if isinstance(c, SendVoice)]
        assert any("в 10" in (m.text or "") for m in session.sent())

    run(scenario())


def test_slow_answer_posts_and_removes_thinking_note(env, monkeypatch):
    from datetime import datetime, timezone as tz

    bot, dp, app, session, xai = env
    monkeypatch.setattr(chat, "THINKING_AFTER", 0.01)

    async def scenario():
        app.db.set_setting("owner_id", str(OWNER.id))
        await dp.feed_update(bot, added(GROUP, OWNER, admin_member()))
        session.calls.clear()
        xai.delay = 0.1
        await dp.feed_update(bot, upd(message=Message(
            message_id=500, date=datetime.now(tz.utc), chat=GROUP, from_user=STRANGER, text="@it_test_bot ребус?",
            entities=[MessageEntity(type="mention", offset=0, length=12)])))
        texts = [m.text for m in session.sent()]
        assert any("Думаю" in t for t in texts) and any("Завтра в 10:00" in t for t in texts)
        assert [c for c in session.calls if isinstance(c, DeleteMessage)]  # the note is removed
        # a fast answer never posts the note
        xai.delay = 0
        session.calls.clear()
        await dp.feed_update(bot, upd(message=Message(
            message_id=501, date=datetime.now(tz.utc), chat=GROUP, from_user=STRANGER, text="@it_test_bot привет",
            entities=[MessageEntity(type="mention", offset=0, length=12)])))
        assert not any("Думаю" in m.text for m in session.sent())
        assert not [c for c in session.calls if isinstance(c, DeleteMessage)]

    run(scenario())


def test_hard_questions_go_to_the_strong_model(env):
    from datetime import datetime, timezone as tz

    bot, dp, app, session, xai = env
    assert app.hard_model() == "grok-4.7"

    async def ask(mid, text):
        xai.chats.clear()
        session.calls.clear()
        await dp.feed_update(bot, upd(message=Message(
            message_id=mid, date=datetime.now(tz.utc), chat=GROUP, from_user=STRANGER, text=f"@it_test_bot {text}",
            entities=[MessageEntity(type="mention", offset=0, length=12)])))
        return [m for m, _ in xai.chats], [m.text for m in session.sent()]

    async def scenario():
        app.db.set_setting("owner_id", str(OWNER.id))
        await dp.feed_update(bot, added(GROUP, OWNER, admin_member()))
        # an ordinary question: the chat model only, told it may hand the task over
        models, texts = await ask(600, "как дела?")
        assert models == ["grok-4.3"] and xai.efforts[-1] is None
        assert chat.ESCALATE_MARK in xai.chats[0][1][0]["content"]
        # a puzzle goes straight to the strong model, without the hand-over note
        models, _ = await ask(601, "разгадай ребус")
        assert models == ["grok-4.7"] and xai.efforts[-1] == "medium"  # "high" thinks for 12+ minutes
        assert chat.ESCALATE_MARK not in xai.chats[0][1][0]["content"]
        # the chat model gives up or only promises an answer: the strong model answers
        for gave_up in (chat.ESCALATE_MARK, "Сейчас разберу дорожку почтальона и соберу буквы по порядку."):
            xai.queued = [ChatResult(text=gave_up, input_tokens=5, output_tokens=1)]
            models, texts = await ask(602, "сколько будет, если…")
            assert models == ["grok-4.3", "grok-4.7"]
            assert chat.ESCALATE_MARK not in xai.chats[1][1][0]["content"]
            assert any("Завтра в 10:00" in t for t in texts) and not any(gave_up in t for t in texts)
        # the strong model chosen as the chat model: no fallback
        app.db.set_setting("chat_model", "grok-4.7")
        assert app.hard_model() == ""
        models, _ = await ask(603, "как дела?")
        assert models == ["grok-4.7"] and chat.ESCALATE_MARK not in xai.chats[0][1][0]["content"]

    run(scenario())
