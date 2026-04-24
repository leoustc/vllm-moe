#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

FEATURE_MD="${FEATURE_MD:-$ROOT_DIR/FEATURE.md}"
STATE_DIR="${STATE_DIR:-$ROOT_DIR/.agentwave}"
MAX_ROUNDS="${MAX_ROUNDS:-100}"
SLEEP_BETWEEN_ROUNDS="${SLEEP_BETWEEN_ROUNDS:-5}"
AUTO_COMMIT="${AUTO_COMMIT:-0}"
GIT_REMOTE_BRANCH="${GIT_REMOTE_BRANCH:-agentwave-moe-cpu-offload}"
DONE_GREP="${DONE_GREP:-ALL_ACCEPTANCE_CRITERIA_MET}"

if [[ ! -f "$FEATURE_MD" ]]; then
  echo "ERROR: feature file not found: $FEATURE_MD" >&2
  exit 1
fi

if ! command -v codex >/dev/null 2>&1; then
  echo "ERROR: codex CLI not found in PATH." >&2
  exit 1
fi

if [[ -z "${CODER_CMD_TEMPLATE:-}" ]]; then
  export CODER_CMD_TEMPLATE='codex exec -m gpt-5.4 --dangerously-bypass-approvals-and-sandbox -C "$REPO_DIR" - < "$PROMPT_FILE"'
fi

if [[ -z "${VALIDATE_CMD:-}" ]]; then
  export VALIDATE_CMD="$ROOT_DIR/.venv/bin/python -m pytest tests/v1/engine/test_engine_args.py tests/v1/worker/test_moe_offload_memory.py -q"
fi

export REPO_DIR="$ROOT_DIR"
export FEATURE_MD
export STATE_DIR
export MAX_ROUNDS
export SLEEP_BETWEEN_ROUNDS
export AUTO_COMMIT
export GIT_REMOTE_BRANCH
export DONE_GREP

chmod +x "$ROOT_DIR/loop.sh"

echo "Starting Codex feature loop"
echo "Repo:            $REPO_DIR"
echo "Feature:         $FEATURE_MD"
echo "State dir:       $STATE_DIR"
echo "Max rounds:      $MAX_ROUNDS"
echo "Validate cmd:    $VALIDATE_CMD"
echo "Coder command:   $CODER_CMD_TEMPLATE"

exec "$ROOT_DIR/loop.sh"
