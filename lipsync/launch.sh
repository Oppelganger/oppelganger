#!/usr/bin/env bash
set -eu

cd /app

# shellcheck source=/dev/null
. venv/bin/activate

uvicorn --host 0.0.0.0 --port 6873 inference:app
