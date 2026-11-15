"""Download videos from social links (Reels, X, VK, ...) with yt-dlp."""
from __future__ import annotations

import asyncio
import json
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
    ("Threads", re.compile(r"^https?://(?:www\.)?threads\.(?:net|com)/@[\w.]+/post/[\w-]+", re.I)),
]

# Services whose post text (caption) is worth sending along with the media.
CAPTION_SERVICES = {"Instagram", "Threads", "X", "TikTok", "Reddit"}
MAX_DESCRIPTION = 4000


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
    description: str = ""  # the post's own text (Instagram caption, tweet, Threads post…)

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


_photo_ie_cls: Any = None


def _instagram_with_photos() -> Any:
    """yt-dlp's Instagram extractor drops photos (it only keeps video formats). This subclass keeps them:
    a photo (or a photo slide of a carousel) gets a single 'image' format with the largest picture."""
    global _photo_ie_cls
    if _photo_ie_cls is not None:
        return _photo_ie_cls
    from yt_dlp.extractor.instagram import InstagramIE as _Base
    from yt_dlp.utils import traverse_obj, url_or_none

    class InstagramIE(_Base):  # same name -> same ie_key "Instagram", replaces the stock extractor
        def _extract_product_media(self, product_media):
            info = super()._extract_product_media(product_media)
            if info.get("formats"):
                return info
            candidates = traverse_obj(product_media, (
                "image_versions2", "candidates", lambda _, v: url_or_none(v["url"]))) or []
            if not candidates:
                return info
            best = max(candidates, key=lambda c: (c.get("width") or 0) * (c.get("height") or 0))
            path = best["url"].split("?", 1)[0].lower()
            ext = "webp" if path.endswith(".webp") else "png" if path.endswith(".png") else "jpg"
            info["formats"] = [{
                "url": best["url"], "format_id": "image", "ext": ext,
                "width": best.get("width"), "height": best.get("height"),
                "vcodec": "none", "acodec": "none",
            }]
            info["duration"] = None
            return info

    _photo_ie_cls = InstagramIE
    return _photo_ie_cls


def _run_ytdlp(url: str, outdir: Path, cookies: Path | None, height: int) -> dict[str, Any]:
    import yt_dlp  # imported lazily: heavy module, only needed when a link shows up

    opts: dict[str, Any] = {
        "outtmpl": str(outdir / "%(autonumber)02d-%(id).40s.%(ext)s"),
        # video+audio, else a muxed file, else a photo (format_id "image" from the Instagram subclass)
        "format": "bv*+ba/b/image",
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
        ydl.add_info_extractor(_instagram_with_photos()())
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


def _description(link: Link, info: dict[str, Any], entries: list[dict[str, Any]]) -> str:
    if link.service not in CAPTION_SERVICES:
        return ""
    text = info.get("description") or next((e.get("description") for e in entries if e.get("description")), "")
    return str(text or "").strip()[:MAX_DESCRIPTION]


async def download(link: Link, cookies: Path | None = None) -> Download:
    if link.service == "Threads":
        return await _download_threads(link)
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
                description=_description(link, info, entries),
            )
        raise last_error
    except BaseException:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise


# ---------------------------------------------------------------- Threads (no yt-dlp extractor)
THREADS_CODE_RE = re.compile(r"threads\.(?:net|com)/@([\w.]+)/post/([\w-]+)", re.I)
THREADS_JSON_RE = re.compile(r'<script type="application/json"[^>]*data-sjs[^>]*>(.*?)</script>', re.S)


def _threads_post(html_text: str, code: str) -> dict[str, Any] | None:
    """The post object (caption, user, media) embedded in the page's JSON blobs."""
    best: dict[str, Any] | None = None

    def walk(o: Any) -> None:
        nonlocal best
        if isinstance(o, dict):
            if o.get("code") == code and ("caption" in o or "image_versions2" in o or "video_versions" in o):
                if best is None or len(o) > len(best):
                    best = o
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    for blob in THREADS_JSON_RE.findall(html_text):
        if code not in blob:
            continue
        try:
            walk(json.loads(blob))
        except ValueError:
            continue
    return best


def _threads_media(post: dict[str, Any]) -> list[dict[str, Any]]:
    """[{url, is_image, width, height}] for every photo/video of the post (carousel or single)."""
    out = []
    for m in (post.get("carousel_media") or [post])[:MAX_ITEMS]:
        if not isinstance(m, dict):
            continue
        videos = [v for v in m.get("video_versions") or [] if isinstance(v, dict) and v.get("url")]
        if videos:
            v = max(videos, key=lambda v: (v.get("width") or 0) * (v.get("height") or 0))
            out.append({"url": v["url"], "is_image": False, "width": v.get("width") or m.get("original_width"),
                        "height": v.get("height") or m.get("original_height")})
            continue
        cands = [c for c in ((m.get("image_versions2") or {}).get("candidates") or [])
                 if isinstance(c, dict) and c.get("url")]
        if cands:
            c = max(cands, key=lambda c: (c.get("width") or 0) * (c.get("height") or 0))
            out.append({"url": c["url"], "is_image": True, "width": c.get("width"), "height": c.get("height")})
    return out


def _threads_fetch(link: Link, outdir: Path) -> tuple[str, str, list[Item]]:
    from curl_cffi import requests as cffi  # installed with yt-dlp[curl-cffi]; looks like a real browser

    m = THREADS_CODE_RE.search(link.url)
    if not m:
        raise DownloadError("no_media", "not a Threads post link")
    user, code = m.groups()
    session = cffi.Session(impersonate="chrome")
    try:
        r = session.get(f"https://www.threads.com/@{user}/post/{code}", timeout=30)
        if r.status_code == 404:
            raise DownloadError("no_media", "post not found")
        if r.status_code != 200:
            raise DownloadError("failed", f"Threads HTTP {r.status_code}")
        post = _threads_post(r.text, code)
        if post is None:
            raise DownloadError("auth" if "login" in r.url else "failed", "post data not found on the page")
        caption = str(((post.get("caption") or {}).get("text")) or "").strip()
        author = str((post.get("user") or {}).get("username") or user)
        media = _threads_media(post)
        if not media:
            raise DownloadError("no_media", "text-only post")
        items = []
        for i, md in enumerate(media, 1):
            path_part = md["url"].split("?", 1)[0].lower()
            ext = "mp4" if not md["is_image"] else ("webp" if path_part.endswith(".webp") else
                                                     "png" if path_part.endswith(".png") else "jpg")
            path = outdir / f"{i:02d}-{code}.{ext}"
            resp = session.get(md["url"], timeout=120, stream=True)
            if resp.status_code != 200:
                raise DownloadError("failed", f"Threads media HTTP {resp.status_code}")
            size, too_big = 0, False
            with path.open("wb") as f:
                for chunk in resp.iter_content():
                    size += len(chunk)
                    if size > TG_UPLOAD_LIMIT:
                        too_big = True
                        break
                    f.write(chunk)
            resp.close()
            if too_big:
                path.unlink(missing_ok=True)
                continue
            items.append(Item(path=path, is_image=md["is_image"], width=_int(md["width"]), height=_int(md["height"])))
        if not items:
            raise DownloadError("too_big", f"файл больше {TG_UPLOAD_LIMIT // 1024 // 1024} МБ")
        return caption, author, items
    finally:
        session.close()


async def _download_threads(link: Link) -> Download:
    tmpdir = Path(tempfile.mkdtemp(prefix="itbot-dl-"))
    try:
        try:
            caption, author, items = await asyncio.wait_for(
                asyncio.to_thread(_threads_fetch, link, tmpdir), DOWNLOAD_TIMEOUT)
        except asyncio.TimeoutError as exc:
            raise DownloadError("failed", "скачивание заняло слишком много времени") from exc
        except DownloadError:
            raise
        except Exception as exc:  # network errors from curl_cffi
            raise DownloadError("failed", str(exc)[:300]) from exc
        return Download(link=link, title=caption[:200], uploader=author[:64], items=items, tmpdir=tmpdir,
                        description=caption[:MAX_DESCRIPTION])
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
