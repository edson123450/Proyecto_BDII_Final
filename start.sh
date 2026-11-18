#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
fi
if ! .venv/bin/python -c 'import fastapi,uvicorn,numpy,pandas,matplotlib' >/dev/null 2>&1; then
  .venv/bin/python -m pip install -r requirements.txt
fi
.venv/bin/python -m backend.prepare
exec .venv/bin/python -m uvicorn backend.app:app --host 127.0.0.1 --port "${PORT:-8014}"
