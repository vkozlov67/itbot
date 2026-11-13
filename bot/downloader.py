"""Download videos from social links (Reels, X, VK, ...) with yt-dlp."""
from __future__ import annotations

import asyncio
import logging
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aiogram.types import Message

log = logging.getLogger(__name__)

TG_UPLOAD_LIMIT = 49 * 1024 * 1024  # Bot API: 50 MB per uploaded file
MAX_LINKS_PER_MESSAGE = 3
MAX_ITEMS = 10  # Telegram media group limit (carousels)
DOWNLOAD_TIMEOUT = 240
IMAGE_EXTS = {"jpg", "jpeg", "png", "webp"}

# (service name, URL pattern). Only links to a single post/video are matched, so profile
# or feed links are ignored and the bot never spams errors for them.
SERVICES: list[tuple[str, re.Pattern[str]]] = [
    ("Instagram", re.compile(r"^https?://(?:www\.|m\.)?instagram\.com/(?:[\w.]+/)?(?:reels?|p|tv)/[\w-]+", re.I)),
    ("TikTok", re.compile(
        r"^https?://(?:(?:www|m)\.tiktok\.com/(?:@[\w.-]+/(?:video|photo)/\d+|t/\w+)|(?:vm|vt)\.tiktok\.com/\w+)", re.I)),
    ("YouTube", re.compile(r"^https?://(?:www\.|m\.)?youtube\.com/shorts/[\w-]{11}", re.I)),
    ("X", re.compile(r"^https?://(?:www\.|mobile\.)?(?:x|twitter)\.com/\w+/status/\d+", re.I)),
    ("Facebook", re.compile(
        r"^https?://(?:(?:www|m|web)\.facebook\.com/(?:reel/\d+|watch/?\?v=\d+|[\w.]+/videos/[\w.-]+|share/[rv]/\w+)"
        r"|fb\.watch/\w+)", re.I)),
    ("VK", re.compile(r"^https?://(?:(?:www|m)\.)?(?:vk\.com|vk\.ru|vkvideo\.ru)/(?:clip|video)-?\d+_\d+", re.I)),
    ("Reddit", re.compile(r"^https?://(?:(?:www|old|new)\.reddit\.com/r/\w+/(?:comments|s)/\w+|v\.redd\.it/\w+)", re.I)),
    ("Pinterest", re.compile(r"^https?://(?:(?:\w+\.)?pinterest\.[a-z.]+/pin/[\w-]+|pin\.it/\w+)", re.I)),
    ("Twitch", re.compile(r"^https?://(?:clips\.twitch\.tv/[\w-]+|(?:www\.)?twitch\.tv/\w+/clip/[\w-]+)", re.I)),
]


@dataclass(frozen=True)
class Link:
    url: str
    service: str


@dataclass
class Item:
    path: Path
    is_image: bool
    width: int | None = None
    height: int | None = None
    duration: int | None = None


@dataclass
class Download:
    link: Link
    title: str
    uploader: str
    items: list[Item] = field(default_factory=list)
    tmpdir: Path | None = None

    def cleanup(self) -> None:
        if self.tmpdir:
            shutil.rmtree(self.tmpdir, ignore_errors=True)


class DownloadError(Exception):
    """reason: 'no_media' (nothing to download — stay quiet), 'auth', 'blocked', 'too_big', 'failed'."""

    def __init__(self, reason: str, detail: str):
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


def find_links(message: Message) -> list[Link]:
    text = message.text or message.caption or ""
    entities = message.entities or message.caption_entities or []
    urls: list[str] = []
    for e in entities:
        if e.type == "url":
            urls.append(e.extract_from(text))
        elif e.type == "text_link" and e.url:
            urls.append(e.url)
    found: list[Link] = []
    seen: set[str] = set()
    for raw in urls:
        url = raw.strip()
        if not url.lower().startswith(("http://", "https://")):
            url = "https://" + url
        for service, pattern in SERVICES:
            m = pattern.match(url)
            if m and m.group(0).lower() not in seen:
                seen.add(m.group(0).lower())
                found.append(Link(url=url, service=service))
                break
        if len(found) >= MAX_LINKS_PER_MESSAGE:
            break
    return found


def _classify(err: str) -> str:
    e = err.lower()
    if any(s in e for s in ("no video", "there is no video", "no media", "unsupported url", "not a video",
                            "no formats", "requested format is not available")):
        return "no_media"
    if any(s in e for s in ("login", "sign in", "cookies", "authentication", "private", "age-restricted",
                            "not a bot", "rate-limit", "rate limit")):
        return "auth"
    if "blocked" in e or "not available in your country" in e or "geo" in e:
        return "blocked"
    if "larger than max-filesize" in e or "file is larger" in e:
        return "too_big"
    return "failed"


def _collect(info: dict[str, Any]) -> list[dict[str, Any]]:
    entries = info.get("entries") if info.get("_type") in ("playlist", "multi_video") else None
    if entries is None:
        return [info]
    return [e for e in entries if e]


def _run_ytdlp(url: str, outdir: Path, cookies: Path | None, height: int) -> dict[str, Any]:
    import yt_dlp  # imported lazily: heavy module, only needed when a link shows up

    opts: dict[str, Any] = {
        "outtmpl": str(outdir / "%(autonumber)02d-%(id).40s.%(ext)s"),
        "format": "bv*+ba/b",
        "format_sort": [f"res:{height}", "vcodec:h264", "acodec:aac", "ext:mp4:m4a"],
        "merge_output_format": "mp4",
        "max_filesize": TG_UPLOAD_LIMIT,
        "playlist_items": f"1-{MAX_ITEMS}",
        "noplaylist": True,
        "cachedir": False,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "socket_timeout": 30,
        "retries": 2,
        "restrictfilenames": True,
        "logger": _QuietLogger(),
    }
    if cookies and cookies.exists():
        opts["cookiefile"] = str(cookies)
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return ydl.sanitize_info(info) if info else {}


class _QuietLogger:
    def debug(self, msg: str) -> None:
        pass

    def info(self, msg: str) -> None:
        pass

    def warning(self, msg: str) -> None:
        log.debug("yt-dlp: %s", msg)

    def error(self, msg: str) -> None:
        log.debug("yt-dlp: %s", msg)


def _files(outdir: Path) -> list[Path]:
    return sorted(p for p in outdir.iterdir() if p.is_file() and not p.name.endswith((".part", ".ytdl", ".temp")))


async def download(link: Link, cookies: Path | None = None) -> Download:
    tmpdir = Path(tempfile.mkdtemp(prefix="itbot-dl-"))
    try:
        last_error = DownloadError("failed", "ничего не скачалось")
        for height in (720, 480, 360):
            for old in tmpdir.iterdir():
                old.unlink(missing_ok=True)
            try:
                info = await asyncio.wait_for(
                    asyncio.to_thread(_run_ytdlp, link.url, tmpdir, cookies, height), DOWNLOAD_TIMEOUT
                )
            except asyncio.TimeoutError as exc:
                raise DownloadError("failed", "скачивание заняло слишком много времени") from exc
            except Exception as exc:  # yt-dlp raises DownloadError/ExtractorError with free-form text
                msg = str(exc).replace("ERROR: ", "")
                raise DownloadError(_classify(msg), msg[:300]) from exc

            entries = _collect(info)
            files = [f for f in _files(tmpdir) if f.stat().st_size <= TG_UPLOAD_LIMIT]
            if not files:
                last_error = DownloadError("too_big", f"файл больше {TG_UPLOAD_LIMIT // 1024 // 1024} МБ")
                continue
            meta = {str(e.get("id")): e for e in entries}
            items = []
            for f in files[:MAX_ITEMS]:
                e = next((v for k, v in meta.items() if k and k in f.name), entries[0] if entries else {})
                items.append(Item(
                    path=f,
                    is_image=f.suffix.lstrip(".").lower() in IMAGE_EXTS,
                    width=_int(e.get("width")),
                    height=_int(e.get("height")),
                    duration=_int(e.get("duration")),
                ))
            return Download(
                link=link,
                title=str(info.get("title") or "")[:200],
                uploader=str(info.get("uploader") or info.get("channel") or info.get("uploader_id") or "")[:64],
                items=items,
                tmpdir=tmpdir,
            )
        raise last_error
    except BaseException:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise


def _int(v: Any) -> int | None:
    try:
        return int(round(float(v))) if v is not None else None
    except (TypeError, ValueError):
        return None


def cookie_domains(path: Path) -> list[str]:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    domains = set()
    for line in lines:
        if line.startswith("#HttpOnly_"):
            line = line[len("#HttpOnly_"):]
        elif line.startswith("#") or not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) >= 7:
            domains.add(parts[0].lstrip(".").removeprefix("www."))
    return sorted(domains)


def looks_like_cookies(data: bytes) -> bool:
    head = data[:2000].decode("utf-8", errors="replace")
    if "Netscape HTTP Cookie File" in head or "HTTP Cookie File" in head:
        return True
    return any(len(line.split("\t")) >= 7 for line in head.splitlines() if line and not line.startswith("#"))
