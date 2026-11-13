import asyncio
import datetime as dt
from pathlib import Path

import httpx
import pytest
from aiogram.types import Audio, Chat, Message, MessageEntity, User, Voice

from bot.chat import addressed, kind_of, speech_of, transcript_messages
from bot.db import Database
from bot.textutil import md_to_html, split_text
from bot.apis import APIError, OpenAITranscriber, XAIClient

ME = User(id=999, is_bot=True, first_name="ItBot", username="it_test_bot")
ALICE = User(id=1, is_bot=False, first_name="Alice")
GROUP = Chat(id=-100500, type="supergroup", title="Team")
NOW = dt.datetime(2026, 11, 13, 12, 0, tzinfo=dt.timezone.utc)


def msg(text=None, *, mid=10, entities=None, reply=None, voice=None, caption=None):
    return Message(
        message_id=mid, date=NOW, chat=GROUP, from_user=ALICE, text=text, entities=entities,
        reply_to_message=reply, voice=voice, caption=caption,
    )


# ---------------- formatting
def test_md_to_html_basic():
    out = md_to_html("**Жирный** и *курсив*, `a<b>` и [ссылка](https://x.ai/?a=1&b=2)\n- пункт\n# Заголовок")
    assert "<b>Жирный</b>" in out
    assert "<i>курсив</i>" in out
    assert "<code>a&lt;b&gt;</code>" in out
    assert '<a href="https://x.ai/?a=1&amp;b=2">ссылка</a>' in out
    assert "• пункт" in out
    assert "<b>Заголовок</b>" in out


def test_md_to_html_escapes_and_code_block():
    out = md_to_html("1 < 2 & 3 > 2\n```python\nif a < b:\n    print('**x**')\n```")
    assert "1 &lt; 2 &amp; 3 &gt; 2" in out
    assert "<pre>if a &lt; b:\n    print('**x**')</pre>" in out


def test_md_to_html_no_false_italic():
    assert md_to_html("2 * 3 * 4") == "2 * 3 * 4"
    assert md_to_html("snake_case_name") == "snake_case_name"


def test_split_text():
    text = ("абзац " * 200 + "\n\n") * 5
    parts = split_text(text, 1000)
    assert all(len(p) <= 1000 for p in parts)
    assert "".join(parts).replace(" ", "").replace("\n", "") == text.replace(" ", "").replace("\n", "")
    assert split_text("") == []


def test_transcript_messages():
    assert transcript_messages("привет", 3) == ["🎙 привет"]
    long = transcript_messages("слово " * 1000, 125)
    assert long[0].startswith("🎙 <b>Расшифровка</b> · 2:05 (1/")
    assert all("<blockquote expandable>" in p for p in long)
    assert transcript_messages("(речь не распознана)", 1) == ["🎙 <i>Речь не распознана</i>"]


# ---------------- addressing
def test_addressed_mention():
    text = "@it_test_bot сколько будет 2+2?"
    m = msg(text, entities=[MessageEntity(type="mention", offset=0, length=12)])
    assert addressed(m, ME) == (True, "сколько будет 2+2?")


def test_addressed_other_mention_ignored():
    m = msg("@someone hi", entities=[MessageEntity(type="mention", offset=0, length=8)])
    assert addressed(m, ME) == (False, "")


def test_addressed_commands():
    assert addressed(msg("/ask что такое RAG"), ME) == (True, "что такое RAG")
    assert addressed(msg("/ask@it_test_bot тест"), ME) == (True, "тест")
    assert addressed(msg("/ask@other_bot тест"), ME) == (False, "")
    assert addressed(msg("/start"), ME) == (False, "")


def test_addressed_reply_to_bot():
    bot_msg = Message(message_id=5, date=NOW, chat=GROUP, from_user=ME, text="ответ")
    assert addressed(msg("а почему?", reply=bot_msg), ME) == (True, "а почему?")
    other = Message(message_id=5, date=NOW, chat=GROUP, from_user=ALICE, text="x")
    assert addressed(msg("а почему?", reply=other), ME) == (False, "")


def test_kind_of():
    v = Voice(file_id="f", file_unique_id="u", duration=3)
    assert kind_of(msg(voice=v)) == "voice"
    assert kind_of(msg("hi")) == "text"
    song = Audio(file_id="a", file_unique_id="a", duration=200, file_name="song.mp3", mime_type="audio/mpeg")
    m = Message(message_id=1, date=NOW, chat=GROUP, from_user=ALICE, audio=song)
    assert kind_of(m) == "audio" and speech_of(m) is None
    assert speech_of(msg(voice=v))[0] == "voice"


# ---------------- database
def test_db_chats_and_migration(tmp_path: Path):
    db = Database(tmp_path / "t.sqlite3")
    row = db.upsert_chat(-1, "Old", "group", None, status="administrator", added_by=1)
    assert not row.enabled and row.transcribe and row.answer and row.is_admin
    db.set_flag(-1, "enabled", True)
    db.add_message(-1, 1, 1, "Alice", "text", "hello")
    db.migrate_chat(-1, -1001)
    assert db.get_chat(-1) is None
    new = db.get_chat(-1001)
    assert new and new.enabled and new.title == "Old"
    assert db.recent_messages(-1001, 10)[0].text == "hello"
    db.upsert_chat(-1001, "New", "supergroup", "team", status=None)
    assert db.get_chat(-1001).status == "administrator"
    db.set_status(-1001, "left")
    assert db.list_chats() == []
    assert len(db.list_chats(active_only=False)) == 1


def test_db_history_and_usage(tmp_path: Path):
    db = Database(tmp_path / "t.sqlite3")
    for i in range(1, 400):
        db.add_message(-5, i, 1, "A", "text", f"m{i}")
    db._prune(-5)
    assert db.count_messages(-5) == 300
    recent = db.recent_messages(-5, 3, before_message_id=399)
    assert [m.text for m in recent] == ["m396", "m397", "m398"]
    db.add_usage(-5, stt_count=1, stt_seconds=10.5)
    db.add_usage(-5, llm_requests=1, input_tokens=100, output_tokens=50)
    t = db.usage_totals("today")
    assert (t.stt_count, t.stt_seconds, t.llm_requests, t.input_tokens, t.output_tokens) == (1, 10.5, 1, 100, 50)
    db.set_setting("owner_id", "42")
    assert db.get_setting("owner_id") == "42"


# ---------------- API clients (mocked HTTP)
def _mock(client, handler, base):
    client._http = httpx.AsyncClient(base_url=base, headers={"Authorization": "Bearer k"},
                                     transport=httpx.MockTransport(handler))
    return client


def test_openai_transcribe_multipart():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = request.read()
        seen["path"] = request.url.path
        seen["body"] = body
        seen["auth"] = request.headers["authorization"]
        return httpx.Response(200, json={"text": "привет мир", "languages": [{"code": "ru"}]})

    async def run(lang):
        c = _mock(OpenAITranscriber("k", "https://api.openai.com/v1", "gpt-transcribe"), handler,
                  "https://api.openai.com/v1")
        t = await c.transcribe(b"OggS....", "voice.ogg", "audio/ogg", lang)
        await c.close()
        return t

    t = asyncio.run(run(""))
    assert (t.text, t.language) == ("привет мир", "ru")
    assert seen["path"] == "/v1/audio/transcriptions" and seen["auth"] == "Bearer k"
    assert b'name="model"\r\n\r\ngpt-transcribe' in seen["body"]
    assert b'filename="voice.ogg"' in seen["body"] and b"languages" not in seen["body"]
    asyncio.run(run("ru"))
    assert b'name="languages[]"\r\n\r\nru' in seen["body"]


def test_openai_rejects_unknown_extension():
    async def run():
        c = OpenAITranscriber("k", "https://api.openai.com/v1", "gpt-transcribe")
        try:
            await c.transcribe(b"x", "file.xyz", "application/octet-stream")
        finally:
            await c.close()

    with pytest.raises(APIError):
        asyncio.run(run())


def test_api_error_account_problem():
    assert APIError("OpenAI", 401, "bad key").account_problem
    assert APIError("xAI", 429, "You exceeded your current quota").account_problem
    assert not APIError("xAI", 500, "oops").account_problem


def test_xai_chat_retries_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503, json={"error": "busy"})
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "Ответ"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3},
        })

    async def fast_sleep(_):
        return None

    monkeypatch.setattr(asyncio, "sleep", fast_sleep)

    async def run():
        c = _mock(XAIClient("k", "https://api.x.ai/v1"), handler, "https://api.x.ai/v1")
        r = await c.chat("grok-4.3", [{"role": "user", "content": "hi"}])
        await c.close()
        return r

    r = asyncio.run(run())
    assert (r.text, r.input_tokens, r.output_tokens, calls["n"]) == ("Ответ", 10, 3, 2)
