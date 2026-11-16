"""Interactive setup: `itbot-setup` on the server. Asks for tokens, validates them, writes .env, restarts the bot."""
from __future__ import annotations

import getpass
import os
import secrets
import shutil
import subprocess
import sys
from pathlib import Path

import httpx

from .config import ENV_FILE

SERVICE = "itbot"
KEYS = (
    "BOT_TOKEN", "XAI_API_KEY", "OPENAI_API_KEY", "OWNER_ID", "CLAIM_CODE",
    "CHAT_MODEL", "STT_MODEL", "STT_LANGUAGE", "TIMEZONE", "ELEVENLABS_API_KEY", "ELEVENLABS_VOICE_ID",
)
ELEVENLABS = "https://api.elevenlabs.io"
# Preferred OpenAI transcription models, best first.
STT_MODELS = ("gpt-transcribe", "gpt-4o-transcribe", "gpt-4o-mini-transcribe", "whisper-1")


def read_env() -> dict[str, str]:
    env: dict[str, str] = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, _, v = line.partition("=")
                env[k.strip()] = v.strip()
    return env


def write_env(env: dict[str, str]) -> None:
    lines = ["# itbot configuration (managed by itbot-setup)"]
    for k in KEYS:
        lines.append(f"{k}={env.get(k, '')}")
    for k, v in env.items():
        if k not in KEYS:
            lines.append(f"{k}={v}")
    tmp = ENV_FILE.with_suffix(".tmp")
    old_umask = os.umask(0o077)
    try:
        tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    finally:
        os.umask(old_umask)
    os.replace(tmp, ENV_FILE)
    if os.geteuid() == 0 and shutil.which("chown"):
        subprocess.run(["chown", "root:itbot", str(ENV_FILE)], check=False)
        os.chmod(ENV_FILE, 0o640)


def ask_secret(prompt: str, current: str) -> str:
    hint = " (Enter — оставить текущий)" if current else ""
    while True:
        value = getpass.getpass(f"{prompt}{hint}: ").strip()
        if value:
            return value
        if current:
            return current


def check_bot(token: str) -> dict | None:
    try:
        r = httpx.get(f"https://api.telegram.org/bot{token}/getMe", timeout=20)
        data = r.json()
    except (httpx.HTTPError, ValueError) as exc:
        print(f"  ✗ Не удалось связаться с Telegram: {exc}")
        return None
    if not data.get("ok"):
        print(f"  ✗ Telegram отклонил токен: {data.get('description')}")
        return None
    return data["result"]


def _check_models(name: str, url: str, key: str) -> list[str] | None:
    try:
        r = httpx.get(url, headers={"Authorization": f"Bearer {key}"}, timeout=20)
    except httpx.HTTPError as exc:
        print(f"  ✗ Не удалось связаться с {name}: {exc}")
        return None
    if r.status_code != 200:
        print(f"  ✗ {name} отклонил ключ: HTTP {r.status_code} {r.text[:200]}")
        return None
    return [m.get("id", "") for m in r.json().get("data", [])]


def ask_optional(prompt: str, current: str) -> str:
    """Hidden input that may stay empty. Enter keeps the current value; "-" clears it."""
    hint = " (Enter — оставить текущий, «-» — убрать)" if current else " (Enter — пропустить)"
    value = getpass.getpass(f"{prompt}{hint}: ").strip()
    if value == "-":
        return ""
    return value or current


def elevenlabs_voices(key: str) -> list[dict] | None:
    """Voices the key can use; None if ElevenLabs rejected the key."""
    try:
        r = httpx.get(f"{ELEVENLABS}/v2/voices", params={"page_size": 100}, headers={"xi-api-key": key}, timeout=20)
    except httpx.HTTPError as exc:
        print(f"  ✗ Не удалось связаться с ElevenLabs: {exc}")
        return None
    if r.status_code != 200:
        print(f"  ✗ ElevenLabs отклонил ключ: HTTP {r.status_code} {r.text[:200]}")
        return None
    return [v for v in r.json().get("voices", []) if v.get("voice_id")]


def setup_elevenlabs(env: dict[str, str], preset: str, interactive: bool) -> None:
    """Optional: ElevenLabs voice for spoken replies (otherwise OpenAI TTS)."""
    key = preset or (ask_optional("4/4 API-ключ ElevenLabs (голос бота; необязательно)",
                                  env.get("ELEVENLABS_API_KEY", "")) if interactive else env.get("ELEVENLABS_API_KEY", ""))
    env["ELEVENLABS_API_KEY"] = key
    if not key:
        print("  · ElevenLabs не подключён — бот говорит голосом OpenAI")
        return
    voices = elevenlabs_voices(key)
    if voices is None:
        if interactive and input("  Сохранить ключ всё равно? [y/N]: ").strip().lower() != "y":
            env["ELEVENLABS_API_KEY"] = ""
        return
    female = [v for v in voices if str((v.get("labels") or {}).get("gender", "")).lower() == "female"]
    shown = (female or voices)[:20]
    current = env.get("ELEVENLABS_VOICE_ID", "")
    print(f"  ✓ Ключ ElevenLabs работает ({len(voices)} голосов)")
    if not interactive or not shown:
        return
    for i, v in enumerate(shown, 1):
        labels = v.get("labels") or {}
        desc = ", ".join(str(labels[k]) for k in ("accent", "age", "descriptive") if labels.get(k))
        mark = " ← сейчас" if v["voice_id"] == current else ""
        print(f"    {i:2}. {v.get('name')} ({desc or v.get('category', '')}){mark}")
    choice = input("  Номер голоса или voice_id из Voice Library (Enter — "
                   + ("оставить текущий" if current else "первый из списка") + "): ").strip()
    if choice.isdigit() and 1 <= int(choice) <= len(shown):
        env["ELEVENLABS_VOICE_ID"] = shown[int(choice) - 1]["voice_id"]
    elif choice:
        env["ELEVENLABS_VOICE_ID"] = choice
    elif not current:
        env["ELEVENLABS_VOICE_ID"] = shown[0]["voice_id"]


def ask_validated(prompt: str, current: str, preset: str, interactive: bool, check) -> tuple[str, object]:
    while True:
        value = preset or ask_secret(prompt, current)
        result = check(value)
        if result is not None:
            return value, result
        if not interactive:
            sys.exit(1)
        preset = ""


def main() -> None:
    env = read_env()
    presets = {k: os.environ.get(f"ITBOT_{k}", "").strip()
               for k in ("BOT_TOKEN", "XAI_API_KEY", "OPENAI_API_KEY", "ELEVENLABS_API_KEY")}
    interactive = sys.stdin.isatty()

    print("=== Настройка itbot ===")
    print("Ввод скрыт — вставьте значение и нажмите Enter.\n")

    # 1. Telegram
    token, me = ask_validated("1/4 Токен бота от @BotFather", env.get("BOT_TOKEN", ""), presets["BOT_TOKEN"],
                              interactive, check_bot)
    print(f"  ✓ Бот @{me['username']}")
    env["BOT_TOKEN"] = token

    # 2. Grok (answers)
    xai_key, xai_models = ask_validated(
        "2/4 API-ключ Grok / xAI (для ответов)", env.get("XAI_API_KEY", ""), presets["XAI_API_KEY"], interactive,
        lambda k: _check_models("xAI", "https://api.x.ai/v1/models", k),
    )
    print(f"  ✓ Ключ Grok работает ({len(xai_models)} моделей)")
    env["XAI_API_KEY"] = xai_key
    chat_model = env.get("CHAT_MODEL") or "grok-4.3"
    if xai_models and chat_model not in xai_models:
        print(f"  ! Модель {chat_model} не найдена у xAI — выберите другую в /panel → Модель.")
    env["CHAT_MODEL"] = chat_model

    # 3. OpenAI (transcription)
    oa_key, oa_models = ask_validated(
        "3/4 API-ключ OpenAI (для расшифровки голосовых)", env.get("OPENAI_API_KEY", ""),
        presets["OPENAI_API_KEY"], interactive,
        lambda k: _check_models("OpenAI", "https://api.openai.com/v1/models", k),
    )
    env["OPENAI_API_KEY"] = oa_key
    stt_model = env.get("STT_MODEL", "")
    if not stt_model or (oa_models and stt_model not in oa_models):
        stt_model = next((m for m in STT_MODELS if m in oa_models), STT_MODELS[0])
    env["STT_MODEL"] = stt_model
    print(f"  ✓ Ключ OpenAI работает, расшифровка: {stt_model}")

    # 4. ElevenLabs (voice, optional)
    setup_elevenlabs(env, presets["ELEVENLABS_API_KEY"], interactive)

    if not env.get("OWNER_ID") and not env.get("CLAIM_CODE"):
        env["CLAIM_CODE"] = secrets.token_urlsafe(12).replace("-", "x").replace("_", "y")
    if not env.get("TIMEZONE"):
        try:
            env["TIMEZONE"] = Path("/etc/timezone").read_text().strip() or "UTC"
        except OSError:
            env["TIMEZONE"] = "UTC"
    write_env(env)
    print(f"  ✓ Настройки сохранены в {ENV_FILE}")

    if shutil.which("systemctl"):
        subprocess.run(["systemctl", "enable", SERVICE], check=False, capture_output=True)
        subprocess.run(["systemctl", "restart", SERVICE], check=True)
        print("  ✓ Бот запущен (статус: systemctl status itbot, логи: journalctl -u itbot -f)")

    username = me["username"]
    print()
    if env.get("OWNER_ID"):
        print(f"Владелец: {env['OWNER_ID']}. Откройте @{username} и нажмите Start.")
    else:
        print("👉 Чтобы стать владельцем, откройте эту ссылку в Telegram и нажмите «Start»:")
        print(f"   https://t.me/{username}?start=claim_{env['CLAIM_CODE']}")
        print("   (если владелец уже назначен — просто откройте бота и отправьте /panel)")
    if not me.get("can_join_groups", True):
        print("\n⚠️ В @BotFather выключено добавление в группы: /mybots → Bot Settings → Allow Groups → Turn on.")


if __name__ == "__main__":
    main()
