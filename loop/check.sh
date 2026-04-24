#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ENV_BIN="$SCRIPT_DIR"/.venv/bin

if [ ! -d .venv ]; then
  uv venv
fi

source .venv/bin/activate
uv pip install -U pip
VLLM_USE_PRECOMPILED=1 uv pip install --editable ../ --torch-backend=auto

${ENV_BIN}/vllm serve --help=all
