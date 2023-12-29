#!/usr/bin/env bash
set -eu

cd /work

# shellcheck source=/dev/null
. venv/bin/activate

python main.py
