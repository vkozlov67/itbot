#!/usr/bin/env bash
# Social networks break downloaders often; yt-dlp ships fixes quickly. Run daily by itbot-ytdlp-update.timer.
set -euo pipefail
PY=/opt/itbot/venv/bin/python
before=$($PY -m yt_dlp --version 2>/dev/null || echo none)
/opt/itbot/venv/bin/pip install --quiet --upgrade "yt-dlp[default,curl-cffi]"
after=$($PY -m yt_dlp --version)
if [ "$before" != "$after" ]; then
  echo "yt-dlp updated: $before -> $after"
  systemctl try-restart itbot
else
  echo "yt-dlp is up to date ($after)"
fi
