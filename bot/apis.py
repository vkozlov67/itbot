"""HTTP clients: xAI (Grok) for answers, pictures and videos; OpenAI for speech-to-text and text-to-speech;
ElevenLabs for text-to-speech (optional)."""
from __future__ import annotations

import asyncio
import base64
import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx

log = logging.getLogger(__name__)

RETRY_STATUSES = {408, 409, 429, 500, 502, 503, 504}
RESPONSE_TIMEOUT = 600.0  # seconds a Grok answer may take (hard puzzles with reasoning take minutes)
NON_CHAT_MARKERS = ("imagine", "image", "video", "voice", "tts", "stt", "transcribe", "embed")
# Containers OpenAI's transcription endpoint accepts (by filename extension).
OPENAI_AUDIO_EXTS = {"flac", "mp3", "mp4", "mpeg", "mpga", "m4a", "ogg", "wav", "webm"}


class APIError(Exception):
    def __init__(self, provider: str, status: int | None, message: str):
        super().__init__(f"{provider}: HTTP {status}: {message}" if status else f"{provider}: {message}")
        self.provider = provider
        self.status = status
        self.message = message

    @property
    def account_problem(self) -> bool:
        msg = self.message.lower()
        return self.status in (401, 402, 403) or any(w in msg for w in ("credit", "billing", "quota", "insufficient"))


@dataclass(frozen=True)
class Transcript:
    text: str
    language: str


@dataclass(frozen=True)
class Source:
    url: str
    title: str = ""


@dataclass(frozen=True)
class ChatResult:
    text: str
    input_tokens: int
    output_tokens: int
    sources: tuple[Source, ...] = ()
    searches: int = 0


def to_chat_messages(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Responses-API input -> Chat Completions messages (used as a fallback)."""
    out = []
    for item in items:
        content = item["content"]
        if isinstance(content, list):
            parts = []
            for p in content:
                if p.get("type") == "input_text":
                    parts.append({"type": "text", "text": p["text"]})
                elif p.get("type") == "input_image":
                    parts.append({"type": "image_url", "image_url": {"url": p["image_url"], "detail": p.get("detail", "auto")}})
            content = parts
        out.append({"role": item["role"], "content": content})
    return out


def _error_message(resp: httpx.Response) -> str:
    try:
        data = resp.json()
    except ValueError:
        return resp.text[:300] or resp.reason_phrase
    if isinstance(data, dict):
        detail = data.get("detail")  # ElevenLabs: {"detail": {"status": "quota_exceeded", "message": "..."}}
        if isinstance(detail, dict):
            return " ".join(str(detail.get(k)) for k in ("status", "message") if detail.get(k))[:300] or str(detail)[:300]
        if isinstance(detail, str):
            return detail[:300]
        err = data.get("error")
        if isinstance(err, dict):
            return str(err.get("message") or err)[:300]
        if err:
            return str(err)[:300]
        if data.get("message"):
            return str(data["message"])[:300]
    return str(data)[:300]


class _JsonAPI:
    provider = "api"

    def __init__(self, api_key: str, base_url: str):
        self._http = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers=self._auth(api_key),
            timeout=httpx.Timeout(300.0, connect=20.0),
        )

    @staticmethod
    def _auth(api_key: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {api_key}"}

    async def close(self) -> None:
        await self._http.aclose()

    async def _request(self, method: str, path: str, *, attempts: int = 3, raw: bool = False,
                       retry_timeouts: bool = True, **kwargs: Any) -> Any:
        delay = 2.0
        for attempt in range(1, attempts + 1):
            try:
                resp = await self._http.request(method, path, **kwargs)
            except httpx.HTTPError as exc:
                if attempt == attempts or (not retry_timeouts and isinstance(exc, httpx.ReadTimeout)):
                    raise APIError(self.provider, None, f"сетевая ошибка: {type(exc).__name__}") from exc
                log.warning("%s %s %s network error (attempt %d): %r", self.provider, method, path, attempt, exc)
            else:
                if resp.status_code < 400:
                    if raw:
                        return resp.content
                    try:
                        return resp.json()
                    except ValueError as exc:
                        raise APIError(self.provider, resp.status_code, "ответ не JSON") from exc
                message = _error_message(resp)
                if resp.status_code not in RETRY_STATUSES or attempt == attempts:
                    raise APIError(self.provider, resp.status_code, message)
                log.warning("%s %s %s -> %d (attempt %d): %s",
                            self.provider, method, path, resp.status_code, attempt, message)
                retry_after = resp.headers.get("retry-after", "")
                if retry_after.isdigit():
                    delay = max(delay, min(float(retry_after), 30.0))
            await asyncio.sleep(delay)
            delay *= 2
        raise AssertionError("unreachable")


class XAIClient(_JsonAPI):
    """Grok chat completions (OpenAI-compatible API at api.x.ai)."""

    provider = "xAI"
    reasoning_effort = ""  # "" = the model's default; "low" / "medium" / "high" for reasoning models

    async def chat(self, model: str, messages: list[dict[str, Any]]) -> ChatResult:
        payload = await self._request(
            "POST", "/chat/completions", json={"model": model, "messages": messages}, attempts=2
        )
        choices = payload.get("choices") if isinstance(payload, dict) else None
        if not choices:
            raise APIError(self.provider, None, "пустой ответ модели")
        content = (choices[0].get("message") or {}).get("content")
        if isinstance(content, list):
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        usage = payload.get("usage") or {}
        return ChatResult(
            text=str(content or "").strip(),
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
        )

    async def image(self, model: str, prompt: str, *, source: str | None = None,
                    aspect_ratio: str | None = None) -> bytes:
        """Grok Imagine: a new picture from text, or an edit of `source` (data: URI or URL). Returns image bytes."""
        body: dict[str, Any] = {"model": model, "prompt": prompt, "n": 1, "response_format": "b64_json"}
        if aspect_ratio:
            body["aspect_ratio"] = aspect_ratio
        if source:
            body["image"] = {"url": source, "type": "image_url"}
        payload = await self._request("POST", "/images/edits" if source else "/images/generations",
                                      json=body, attempts=2)
        data = (payload.get("data") or [None])[0] if isinstance(payload, dict) else None
        if not isinstance(data, dict):
            raise APIError(self.provider, None, "модель не вернула картинку")
        if data.get("b64_json"):
            return base64.b64decode(data["b64_json"])
        if data.get("url"):
            content = await _fetch(data["url"])
            if content:
                return content
        raise APIError(self.provider, None, "модель не вернула картинку")

    async def video(self, model: str, prompt: str, *, image: str | None = None, duration: int = 6,
                    aspect_ratio: str | None = None, resolution: str = "720p", timeout: float = 600.0,
                    poll: float = 5.0) -> bytes:
        """Grok Imagine video from text, or from `image` (data: URI or URL). Polls until done; returns MP4 bytes."""
        body: dict[str, Any] = {"model": model, "prompt": prompt, "duration": duration, "resolution": resolution}
        if image:
            body["image"] = {"url": image}  # the output keeps the photo's aspect ratio
        elif aspect_ratio:
            body["aspect_ratio"] = aspect_ratio
        start = await self._request("POST", "/videos/generations", json=body, attempts=2)
        request_id = start.get("request_id") if isinstance(start, dict) else None
        if not request_id:
            raise APIError(self.provider, None, "видео не поставлено в очередь")
        deadline = time.monotonic() + timeout
        while True:
            await asyncio.sleep(poll)
            state = await self._request("GET", f"/videos/{request_id}", attempts=3)
            status = state.get("status") if isinstance(state, dict) else None
            if status == "done":
                video = state.get("video") or {}
                if video.get("respect_moderation") is False:
                    raise APIError(self.provider, 400, "video rejected by content moderation")
                content = await _fetch(video["url"]) if video.get("url") else b""
                if not content:
                    raise APIError(self.provider, None, "модель не вернула видео")
                return content
            if status in ("failed", "expired"):
                err = state.get("error") or {}
                message = err.get("message") if isinstance(err, dict) else str(err)
                code = err.get("code") if isinstance(err, dict) else ""
                status_code = 400 if code in ("invalid_argument", "permission_denied") else None
                raise APIError(self.provider, status_code, f"{code or status}: {message or 'генерация не удалась'}")
            if time.monotonic() > deadline:
                raise APIError(self.provider, None, "видео не готово за 10 минут")

    async def respond(self, model: str, items: list[dict[str, Any]], *, search: bool = True,
                      max_turns: int = 5) -> ChatResult:
        """Responses API with server-side web + X search; Grok decides itself when to search."""
        body: dict[str, Any] = {"model": model, "input": items, "store": False}
        if search:
            body["tools"] = [{"type": "web_search"}, {"type": "x_search"}]
            body["max_turns"] = max_turns
            body["include"] = ["no_inline_citations"]
        if self.reasoning_effort:
            body["reasoning"] = {"effort": self.reasoning_effort}
        # Reasoning models may think for minutes: wait long, and never re-run a request that timed out
        # (it would take as long again and be billed twice).
        payload = await self._request("POST", "/responses", json=body, attempts=2, retry_timeouts=False,
                                      timeout=httpx.Timeout(RESPONSE_TIMEOUT, connect=20.0))
        if not isinstance(payload, dict):
            raise APIError(self.provider, None, "неожиданный ответ")
        texts: list[str] = []
        sources: dict[str, Source] = {}
        searches = 0
        for item in payload.get("output") or []:
            if not isinstance(item, dict):
                continue
            if item.get("type") in ("web_search_call", "custom_tool_call"):
                searches += 1
            if item.get("type") != "message":
                continue
            for part in item.get("content") or []:
                if isinstance(part, dict) and part.get("type") == "output_text":
                    texts.append(str(part.get("text") or ""))
                    for ann in part.get("annotations") or []:
                        url = ann.get("url") if isinstance(ann, dict) else None
                        if url and url not in sources:
                            sources[url] = Source(url=url, title=str(ann.get("title") or ""))
        for url in payload.get("citations") or []:
            if isinstance(url, str) and url.startswith("http") and url not in sources:
                sources[url] = Source(url=url)
        usage = payload.get("usage") or {}
        tool_usage = usage.get("server_side_tool_usage_details") or {}
        searches = max(searches, int(tool_usage.get("web_search_calls") or 0) + int(tool_usage.get("x_search_calls") or 0))
        text = "\n\n".join(t.strip() for t in texts if t.strip())
        if not text and payload.get("status") not in (None, "completed"):
            raise APIError(self.provider, None, f"ответ не завершён: {payload.get('status')}")
        return ChatResult(
            text=text,
            input_tokens=int(usage.get("input_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or 0),
            sources=tuple(sources.values()),
            searches=searches,
        )

    async def list_chat_models(self) -> list[str]:
        payload = await self._request("GET", "/models", attempts=2)
        items = payload.get("data", []) if isinstance(payload, dict) else []
        ids = {str(m.get("id")) for m in items if isinstance(m, dict) and m.get("id")}
        return sorted(
            (i for i in ids if i.startswith("grok") and not any(x in i for x in NON_CHAT_MARKERS)),
            reverse=True,
        )


async def _fetch(url: str) -> bytes:
    """Download a generated file (temporary CDN link; our API key is never sent there)."""
    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=20.0), follow_redirects=True) as client:
        try:
            resp = await client.get(url)
        except httpx.HTTPError as exc:
            raise APIError("xAI", None, f"файл не скачался: {type(exc).__name__}") from exc
    return resp.content if resp.status_code == 200 else b""


class OpenAITranscriber(_JsonAPI):
    """OpenAI /v1/audio/transcriptions and /v1/audio/speech."""

    provider = "OpenAI"

    def __init__(self, api_key: str, base_url: str, model: str):
        super().__init__(api_key, base_url)
        self.model = model

    async def transcribe(self, audio: bytes, filename: str, mime: str, language: str = "") -> Transcript:
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext not in OPENAI_AUDIO_EXTS:
            raise APIError(self.provider, None, f"формат .{ext or '?'} не поддерживается")
        data: dict[str, str] = {"model": self.model, "response_format": "json"}
        if language:
            # gpt-transcribe takes a `languages` list; older models take a single `language`.
            data["languages[]" if self.model.startswith("gpt-transcribe") else "language"] = language
        payload = await self._request(
            "POST", "/audio/transcriptions", data=data, files={"file": (filename, audio, mime)}
        )
        if not isinstance(payload, dict):
            raise APIError(self.provider, None, "неожиданный ответ")
        langs = payload.get("languages") or []
        lang = payload.get("language") or (langs[0].get("code") if langs and isinstance(langs[0], dict) else "")
        return Transcript(text=str(payload.get("text") or "").strip(), language=str(lang or ""))

    async def speech(self, text: str, *, model: str, voice: str, instructions: str = "") -> bytes:
        """Text-to-speech as OGG/Opus, ready for Telegram's send_voice."""
        body: dict[str, Any] = {"model": model, "input": text, "voice": voice, "response_format": "opus"}
        if instructions and not model.startswith("tts-1"):
            body["instructions"] = instructions
        audio = await self._request("POST", "/audio/speech", json=body, raw=True)
        if not audio:
            raise APIError(self.provider, None, "пустой звук")
        return audio

    async def list_models(self) -> list[str]:
        payload = await self._request("GET", "/models", attempts=2)
        items = payload.get("data", []) if isinstance(payload, dict) else []
        return [str(m.get("id")) for m in items if isinstance(m, dict) and m.get("id")]


class ElevenLabsTTS(_JsonAPI):
    """ElevenLabs /v1/text-to-speech. Returns MP3; the caller converts it to OGG/Opus for Telegram."""

    provider = "ElevenLabs"
    VOICE_KEY = "elevenlabs_voice"

    def __init__(self, api_key: str, *, model: str, voice_id: str = "",
                 base_url: str = "https://api.elevenlabs.io"):
        super().__init__(api_key, base_url)
        self.model = model
        self.voice_id = voice_id
        self.voice_name = ""

    @staticmethod
    def _auth(api_key: str) -> dict[str, str]:
        return {"xi-api-key": api_key}

    async def list_voices(self) -> list[dict[str, Any]]:
        payload = await self._request("GET", "/v2/voices", params={"page_size": 100}, attempts=2)
        voices = payload.get("voices", []) if isinstance(payload, dict) else []
        return [v for v in voices if isinstance(v, dict) and v.get("voice_id")]

    async def pick_voice(self) -> str:
        """The configured voice, else the account's first female voice (own/saved voices before premade ones)."""
        if self.voice_id:
            return self.voice_id
        voices = await self.list_voices()
        female = [v for v in voices if str((v.get("labels") or {}).get("gender", "")).lower() == "female"]
        pool = sorted(female or voices, key=lambda v: v.get("category") == "premade")
        if not pool:
            raise APIError(self.provider, None, "в аккаунте нет голосов — задайте ELEVENLABS_VOICE_ID")
        self.voice_id, self.voice_name = str(pool[0]["voice_id"]), str(pool[0].get("name") or "")
        log.info("ElevenLabs voice picked automatically: %s (%s)", self.voice_name, self.voice_id)
        return self.voice_id

    async def speech(self, text: str) -> bytes:
        voice = await self.pick_voice()
        audio = await self._request(
            "POST", f"/v1/text-to-speech/{voice}", params={"output_format": "mp3_44100_128"},
            json={"text": text, "model_id": self.model}, raw=True,
        )
        if not audio:
            raise APIError(self.provider, None, "пустой звук")
        return audio
