from __future__ import annotations

import html
import re

TG_LIMIT = 4096
CHUNK = 3500

_CODE_BLOCK = re.compile(r"```[^\n`]*\n?(.*?)```", re.S)
_INLINE_CODE = re.compile(r"`([^`\n]+)`")
_LINK = re.compile(r"\[([^\]\n]+)\]\((https?://[^\s)]+)\)")
_HEADING = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+(.+?)[ \t]*#*[ \t]*$", re.M)
_BULLET = re.compile(r"^([ \t]*)[-*+][ \t]+", re.M)
_BOLD = re.compile(r"\*\*(?=\S)(.+?)(?<=\S)\*\*", re.S)
_STRIKE = re.compile(r"~~(?=\S)(.+?)(?<=\S)~~", re.S)
_ITALIC = re.compile(r"(?<![\w*])\*(?=[^\s*])([^*\n]+?)(?<=[^\s*])\*(?![\w*])")
_PLACEHOLDER = re.compile(r"\x00(\d+)\x00")


def esc(text: str) -> str:
    return html.escape(text, quote=False)


def md_to_html(text: str) -> str:
    """Convert the Markdown subset LLMs produce into Telegram-safe HTML."""
    text = text.replace("\x00", "")
    stash: list[str] = []

    def keep(fragment: str) -> str:
        stash.append(fragment)
        return f"\x00{len(stash) - 1}\x00"

    text = _CODE_BLOCK.sub(lambda m: keep(f"<pre>{esc(m.group(1).strip(chr(10)))}</pre>"), text)
    text = _INLINE_CODE.sub(lambda m: keep(f"<code>{esc(m.group(1))}</code>"), text)
    text = _LINK.sub(
        lambda m: keep(f'<a href="{html.escape(m.group(2), quote=True)}">{esc(m.group(1))}</a>'), text
    )
    text = esc(text)
    text = _HEADING.sub(lambda m: "<b>" + m.group(1).replace("**", "") + "</b>", text)
    text = _BULLET.sub(r"\1• ", text)
    text = _BOLD.sub(r"<b>\1</b>", text)
    text = _STRIKE.sub(r"<s>\1</s>", text)
    text = _ITALIC.sub(r"<i>\1</i>", text)
    return _PLACEHOLDER.sub(lambda m: stash[int(m.group(1))], text)


def split_text(text: str, limit: int = CHUNK) -> list[str]:
    """Split on paragraph / line / word boundaries so each piece fits in one Telegram message."""
    text = text.strip()
    if not text:
        return []
    parts: list[str] = []
    while len(text) > limit:
        cut = -1
        for sep in ("\n\n", "\n", ". ", " "):
            cut = text.rfind(sep, limit // 2, limit)
            if cut != -1:
                cut += len(sep)
                break
        if cut <= 0:
            cut = limit
        parts.append(text[:cut].rstrip())
        text = text[cut:].lstrip()
    if text:
        parts.append(text)
    return parts


def fmt_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds} с"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{m}:{s:02d}"
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}"


def fmt_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)
