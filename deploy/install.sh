#!/usr/bin/env bash
# Server-side install/update. Run as root from /opt/itbot.
set -euo pipefail
cd /opt/itbot

# System packages: venv support.
NEED=()
python3 -c 'import ensurepip' >/dev/null 2>&1 || NEED+=(python3-venv)
if [ ${#NEED[@]} -gt 0 ]; then
  apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "${NEED[@]}"
fi

id itbot >/dev/null 2>&1 || useradd --system --home-dir /opt/itbot --shell /usr/sbin/nologin itbot
install -d -o itbot -g itbot -m 750 /opt/itbot/data

if [ ! -x venv/bin/python ]; then
  python3 -m venv venv
fi
venv/bin/pip install --quiet --upgrade pip
venv/bin/pip install --quiet -r requirements.txt
venv/bin/pip freeze > requirements.lock

chown -R root:itbot /opt/itbot
chmod -R go-w /opt/itbot
chown -R itbot:itbot /opt/itbot/data
[ -f .env ] && chown root:itbot .env && chmod 640 .env

cat > /usr/local/bin/itbot-setup <<'SH'
#!/usr/bin/env bash
cd /opt/itbot && exec /opt/itbot/venv/bin/python -m bot.setup "$@"
SH
chmod 755 /usr/local/bin/itbot-setup

install -m 644 deploy/itbot.service /etc/systemd/system/itbot.service
systemctl daemon-reload

if [ -f .env ] && grep -q '^BOT_TOKEN=.\+' .env && grep -q '^XAI_API_KEY=.\+' .env \
   && grep -q '^OPENAI_API_KEY=.\+' .env; then
  systemctl enable itbot >/dev/null 2>&1
  systemctl restart itbot
  echo "itbot restarted"
else
  echo "itbot installed. Run: itbot-setup"
fi
