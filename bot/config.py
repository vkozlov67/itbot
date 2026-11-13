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
    context_messages: int
    max_parallel_requests: int

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
            stt_model=os.environ.get("STT_MODEL", "").strip() or "gpt-transcribe",
            stt_language=os.environ.get("STT_LANGUAGE", "").strip(),
            xai_base_url=os.environ.get("XAI_BASE_URL", "").strip() or "https://api.x.ai/v1",
            openai_base_url=os.environ.get("OPENAI_BASE_URL", "").strip() or "https://api.openai.com/v1",
            data_dir=data_dir,
            timezone=os.environ.get("TIMEZONE", "").strip() or "UTC",
            context_messages=_int("CONTEXT_MESSAGES", 40),
            max_parallel_requests=_int("MAX_PARALLEL_REQUESTS", 4),
        )
