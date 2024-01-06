#!/usr/bin/env bash
set -eu

cd /work

export COQUI_TOS_AGREED=1

${PYTHON_EXE} -m personality_engine
