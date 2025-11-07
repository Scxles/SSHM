#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3 not found. Install from your package manager." >&2
  exit 1
fi
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 app.py
