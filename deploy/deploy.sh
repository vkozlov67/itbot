#!/usr/bin/env bash
# Local -> server: copy code and run install. Usage: deploy/deploy.sh [user@host]
set -euo pipefail
HOST="${1:-root@169.58.198.224}"
cd "$(dirname "$0")/.."
rsync -az --delete \
  --exclude '.env' --exclude 'data/' --exclude 'venv/' --exclude '.venv/' \
  --exclude '__pycache__/' --exclude '*.pyc' --exclude 'requirements.lock' --exclude 'tests/' \
  ./ "$HOST:/opt/itbot/"
ssh "$HOST" 'bash /opt/itbot/deploy/install.sh'
