#!/usr/bin/env bash
set -eu

cd /work

# shellcheck source=/dev/null
. venv/bin/activate

export COQUI_TOS_AGREED=1

uvicorn --host 0.0.0.0 --port 6767 main:app
