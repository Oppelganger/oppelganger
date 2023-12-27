#!/usr/bin/env bash
set -eu

cd /app

# shellcheck source=/dev/null
. venv/bin/activate

/main
