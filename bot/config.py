from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / ".env"


def load_env_file(path: Path = ENV_FILE) -> None:
    """Minimal .env loader for running outside systemd. Never overrides real env vars."""
    try:
        content = path.read_text(encoding="utf-8")
    except OSError:
        return
    for raw in content.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


@dataclass(frozen=True)
class Config:
    bot_token: str
    xai_api_key: str
    openai_api_key: str
    owner_id: int | None
    claim_code: str
    chat_model: str
    stt_model: str
    stt_language: str
    xai_base_url: str
    openai_base_url: str
    data_dir: Path
    timezone: str
    summary_time: str
    context_messages: int
    max_parallel_requests: int
    stt_provider: str = "xai"  # "xai" (Grok, OpenAI as fallback) or "openai"
    hard_model: str = "grok-4.7"  # stronger model for hard questions; "" = off
    image_model: str = "grok-imagine-image-2.0"
    video_model: str = "grok-imagine-video-1.5-lite"
    tts_model: str = "gpt-4o-mini-tts"
    tts_voice: str = "marin"
    weekly_day: int = 6  # 0 = Monday … 6 = Sunday
    elevenlabs_api_key: str = ""  # optional: ElevenLabs voice instead of OpenAI TTS
    elevenlabs_voice_id: str = ""  # empty = first female voice of the account
    elevenlabs_model: str = "eleven_multilingual_v2"
    reasoning_effort: str = ""  # Grok thinking depth: "" (model default) / low / medium / high
    hard_effort: str = "medium"  # for the hard model: its default "high" took 12+ minutes on a puzzle

    @classmethod
    def from_env(cls) -> "Config":
        load_env_file()
        bot_token = os.environ.get("BOT_TOKEN", "").strip()
        xai_key = os.environ.get("XAI_API_KEY", "").strip()
        openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
        missing = [
            n for n, v in (("BOT_TOKEN", bot_token), ("XAI_API_KEY", xai_key), ("OPENAI_API_KEY", openai_key))
            if not v
        ]
        if missing:
            raise SystemExit(
                f"Не заданы {', '.join(missing)}. Запустите на сервере: itbot-setup"
            )
        owner_raw = os.environ.get("OWNER_ID", "").strip()
        data_dir = Path(os.environ.get("DATA_DIR", "").strip() or BASE_DIR / "data")
        return cls(
            bot_token=bot_token,
            xai_api_key=xai_key,
            openai_api_key=openai_key,
            owner_id=int(owner_raw) if owner_raw else None,
            claim_code=os.environ.get("CLAIM_CODE", "").strip(),
            chat_model=os.environ.get("CHAT_MODEL", "").strip() or "grok-4.3",
            hard_model=os.environ.get("HARD_MODEL", "grok-4.7").strip(),
            stt_model=os.environ.get("STT_MODEL", "").strip() or "gpt-transcribe",
            stt_language=os.environ.get("STT_LANGUAGE", "").strip(),
            stt_provider="openai" if os.environ.get("STT_PROVIDER", "").strip().lower() == "openai" else "xai",
            xai_base_url=os.environ.get("XAI_BASE_URL", "").strip() or "https://api.x.ai/v1",
            openai_base_url=os.environ.get("OPENAI_BASE_URL", "").strip() or "https://api.openai.com/v1",
            data_dir=data_dir,
            timezone=os.environ.get("TIMEZONE", "").strip() or "UTC",
            summary_time=os.environ.get("SUMMARY_TIME", "").strip() or "21:00",
            image_model=os.environ.get("IMAGE_MODEL", "").strip() or "grok-imagine-image-2.0",
            video_model=os.environ.get("VIDEO_MODEL", "").strip() or "grok-imagine-video-1.5-lite",
            tts_model=os.environ.get("TTS_MODEL", "").strip() or "gpt-4o-mini-tts",
            tts_voice=os.environ.get("TTS_VOICE", "").strip() or "marin",
            weekly_day=max(0, min(6, _int("WEEKLY_DAY", 6))),
            reasoning_effort=os.environ.get("REASONING_EFFORT", "").strip().lower(),
            hard_effort=os.environ.get("HARD_REASONING_EFFORT", "medium").strip().lower(),
            elevenlabs_api_key=os.environ.get("ELEVENLABS_API_KEY", "").strip(),
            elevenlabs_voice_id=os.environ.get("ELEVENLABS_VOICE_ID", "").strip(),
            elevenlabs_model=os.environ.get("ELEVENLABS_MODEL", "").strip() or "eleven_multilingual_v2",
            context_messages=_int("CONTEXT_MESSAGES", 40),
            max_parallel_requests=_int("MAX_PARALLEL_REQUESTS", 4),
        )
