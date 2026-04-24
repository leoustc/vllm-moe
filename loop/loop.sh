#!/usr/bin/env bash
set -euo pipefail

# loop.sh
#
# Single-file long-running coding loop for FEATURE.md-driven development.
#
# What it does:
#   1) Reads FEATURE.md
#   2) Captures repo state
#   3) Builds a fresh prompt for Codex each round
#   4) Runs Codex
#   5) Runs validation/tests
#   6) Logs everything
#   7) Repeats for many rounds
#
# Requirements:
#   - Run inside a git repo (e.g. vLLM repo)
#   - FEATURE.md must exist (default: ./FEATURE.md)
#   - Set CODER_CMD_TEMPLATE to your actual Codex command
#
# Example:
#   chmod +x loop.sh
#   export CODER_CMD_TEMPLATE='codex exec --model gpt-5-codex --cwd "$REPO_DIR" --input "$PROMPT_FILE"'
#   export VALIDATE_CMD='python -m pytest tests/ -q'
#   ./loop.sh
#
# Optional environment variables:
#   REPO_DIR                  default: current dir
#   FEATURE_MD                default: $REPO_DIR/FEATURE.md
#   STATE_DIR                 default: $REPO_DIR/.agentwave
#   VENV_DIR                  default: $REPO_DIR/.venv
#   MAX_ROUNDS                default: 100
#   SLEEP_BETWEEN_ROUNDS      default: 3
#   AUTO_COMMIT               default: 0
#   GIT_REMOTE_BRANCH         default: agentwave-moe-cpu-offload
#   VALIDATE_CMD              default: python -m pytest -q
#   DONE_GREP                 default: ALL_ACCEPTANCE_CRITERIA_MET
#   BOOTSTRAP_VALIDATE_ENV    default: 1
#   UV_PYTHON_VERSION         default: 3.12

ROOT_DIR="$(pwd)"
REPO_DIR="${REPO_DIR:-$ROOT_DIR}"
FEATURE_MD="${FEATURE_MD:-$REPO_DIR/FEATURE.md}"
STATE_DIR="${STATE_DIR:-$REPO_DIR/.agentwave}"
VENV_DIR="${VENV_DIR:-$REPO_DIR/.venv}"
PROMPTS_DIR="$STATE_DIR/prompts"
LOGS_DIR="$STATE_DIR/logs"
REPORTS_DIR="$STATE_DIR/reports"
PATCHES_DIR="$STATE_DIR/patches"

MAX_ROUNDS="${MAX_ROUNDS:-100}"
SLEEP_BETWEEN_ROUNDS="${SLEEP_BETWEEN_ROUNDS:-3}"
AUTO_COMMIT="${AUTO_COMMIT:-0}"
GIT_REMOTE_BRANCH="${GIT_REMOTE_BRANCH:-agentwave-moe-cpu-offload}"
VALIDATE_CMD="${VALIDATE_CMD:-python -m pytest -q}"
CODER_CMD_TEMPLATE="${CODER_CMD_TEMPLATE:-}"
DONE_GREP="${DONE_GREP:-ALL_ACCEPTANCE_CRITERIA_MET}"
BOOTSTRAP_VALIDATE_ENV="${BOOTSTRAP_VALIDATE_ENV:-1}"
UV_PYTHON_VERSION="${UV_PYTHON_VERSION:-3.12}"

mkdir -p "$STATE_DIR" "$PROMPTS_DIR" "$LOGS_DIR" "$REPORTS_DIR" "$PATCHES_DIR"

if [[ ! -f "$FEATURE_MD" ]]; then
  echo "ERROR: FEATURE_MD not found: $FEATURE_MD" >&2
  exit 1
fi

if [[ -z "$CODER_CMD_TEMPLATE" ]]; then
  cat >&2 <<'EOF'
ERROR: CODER_CMD_TEMPLATE is empty.

Set it to your actual Codex CLI command, for example:
  export CODER_CMD_TEMPLATE='codex exec --model gpt-5-codex --cwd "$REPO_DIR" --input "$PROMPT_FILE"'

Then run:
  ./loop.sh
EOF
  exit 1
fi

cd "$REPO_DIR"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "ERROR: REPO_DIR is not a git repository: $REPO_DIR" >&2
  exit 1
fi

ROUND_FILE="$STATE_DIR/round.txt"
if [[ -f "$ROUND_FILE" ]]; then
  START_ROUND="$(cat "$ROUND_FILE")"
else
  START_ROUND=1
fi

echo "Repo dir:          $REPO_DIR"
echo "Feature file:      $FEATURE_MD"
echo "State dir:         $STATE_DIR"
echo "Max rounds:        $MAX_ROUNDS"
echo "Validate command:  $VALIDATE_CMD"

bootstrap_validation_env() {
  if [[ "$BOOTSTRAP_VALIDATE_ENV" != "1" ]]; then
    return 0
  fi

  if [[ -x "$VENV_DIR/bin/python" ]] && "$VENV_DIR/bin/python" -m pytest --version >/dev/null 2>&1; then
    return 0
  fi

  if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: validation bootstrap requires 'uv' in PATH." >&2
    exit 1
  fi

  echo "Bootstrapping validation environment in $VENV_DIR"
  uv venv --python "$UV_PYTHON_VERSION" "$VENV_DIR"
  VLLM_USE_PRECOMPILED=1 uv pip install --python "$VENV_DIR/bin/python" -e . --torch-backend=auto
  uv pip install --python "$VENV_DIR/bin/python" pytest
}

write_repo_snapshot() {
  local out="$1"
  {
    echo "=== REPO STATUS ==="
    git status --short || true
    echo
    echo "=== BRANCH ==="
    git branch --show-current || true
    echo
    echo "=== LAST 10 COMMITS ==="
    git log --oneline -10 || true
    echo
    echo "=== DIFF STAT ==="
    git diff --stat || true
    echo
    echo "=== UNIFIED DIFF (truncated to 2000 lines) ==="
    git diff | sed -n '1,2000p' || true
    echo
    echo "=== CHANGED FILES ==="
    git diff --name-only || true
  } > "$out"
}

write_validation_report() {
  local out="$1"
  {
    echo "=== VALIDATION COMMAND ==="
    echo "$VALIDATE_CMD"
    echo
    echo "=== VALIDATION OUTPUT ==="
    set +e
    bash -lc "$VALIDATE_CMD"
    local rc=$?
    set -e
    echo
    echo "=== VALIDATION EXIT CODE ==="
    echo "$rc"
    return "$rc"
  } > "$out" 2>&1
}

write_prompt() {
  local round="$1"
  local prompt_file="$2"
  local snapshot_file="$3"
  local validation_seed="$4"

  cat > "$prompt_file" <<EOF
You are working inside the vLLM repository.

Primary specification:
- Read and implement the feature from FEATURE.md exactly as written.
- Keep existing vLLM behavior unchanged by default.
- The feature must be fully gated by the new flags, especially --moe-cpu-offload.
- Prioritize correctness, minimal invasive changes, logging, and measurable metrics.

Current round:
$round

Implementation priorities for this round:
1. Read FEATURE.md carefully.
2. Inspect the current repository state and changed files.
3. Continue implementation from the current git working tree state.
4. Make the smallest coherent set of code changes that moves the feature forward.
5. Add or update tests where practical.
6. If validation fails, fix the most blocking issues first.
7. Do not rewrite unrelated subsystems.
8. Preserve stock behavior when new flags are unset.

Important feature requirements:
- stock vLLM path unchanged by default
- add --moe-cpu-offload
- keep router on GPU
- offload expert weights to CPU memory
- fetch only active expert delta
- deepen token pipeline to improve active-expert reuse
- optional same-group fallback only behind explicit flag
- add metrics for offload, reuse, and fallback

Output expectations:
- Modify code directly in the repo.
- At the end, print a concise summary:
  * files changed
  * what was implemented
  * what remains
  * any known test failures
- If the feature is fully complete, print exactly:
  ALL_ACCEPTANCE_CRITERIA_MET

Reference file:
- FEATURE.md at: $FEATURE_MD

Repository snapshot:
$(cat "$snapshot_file")

Latest validation report:
$(cat "$validation_seed")
EOF
}

maybe_commit() {
  local round="$1"
  if [[ "$AUTO_COMMIT" != "1" ]]; then
    return 0
  fi

  if [[ -n "$(git status --porcelain)" ]]; then
    git add -A
    git commit -m "agentwave round ${round}: iterative feature development" || true
  fi
}

ensure_branch() {
  local current
  current="$(git branch --show-current || true)"
  if [[ "$current" != "$GIT_REMOTE_BRANCH" ]]; then
    if git show-ref --verify --quiet "refs/heads/$GIT_REMOTE_BRANCH"; then
      git checkout "$GIT_REMOTE_BRANCH"
    else
      git checkout -b "$GIT_REMOTE_BRANCH"
    fi
  fi
}

ensure_branch

bootstrap_validation_env

BASELINE_VALIDATION="$REPORTS_DIR/validation_round0.txt"
if [[ ! -f "$BASELINE_VALIDATION" ]]; then
  echo "Running initial validation..."
  set +e
  write_validation_report "$BASELINE_VALIDATION"
  BASELINE_RC=$?
  set -e
  echo "Initial validation exit code: $BASELINE_RC"
fi

for (( round=START_ROUND; round<=MAX_ROUNDS; round++ )); do
  echo "$round" > "$ROUND_FILE"

  SNAPSHOT_FILE="$REPORTS_DIR/repo_snapshot_round${round}.txt"
  VALIDATION_FILE="$REPORTS_DIR/validation_round${round}.txt"
  PROMPT_FILE="$PROMPTS_DIR/round_${round}.prompt.txt"
  CODER_LOG="$LOGS_DIR/coder_round${round}.log"
  DIFF_FILE="$PATCHES_DIR/round_${round}.diff"
  VALIDATION_SEED="$REPORTS_DIR/validation_seed_round${round}.txt"

  echo
  echo "================ ROUND $round ================"

  write_repo_snapshot "$SNAPSHOT_FILE"

  if [[ -f "$REPORTS_DIR/validation_round$((round-1)).txt" ]]; then
    cp "$REPORTS_DIR/validation_round$((round-1)).txt" "$VALIDATION_SEED"
  else
    cp "$BASELINE_VALIDATION" "$VALIDATION_SEED"
  fi

  write_prompt "$round" "$PROMPT_FILE" "$SNAPSHOT_FILE" "$VALIDATION_SEED"

  export PROMPT_FILE REPO_DIR FEATURE_MD ROUND="$round"

  echo "Running coder..."
  set +e
  bash -lc "eval \"$CODER_CMD_TEMPLATE\"" > "$CODER_LOG" 2>&1
  CODER_RC=$?
  set -e
  echo "Coder exit code: $CODER_RC"

  git diff > "$DIFF_FILE" || true

  echo "Running validation..."
  set +e
  write_validation_report "$VALIDATION_FILE"
  VALIDATION_RC=$?
  set -e
  echo "Validation exit code: $VALIDATION_RC"

  maybe_commit "$round"

  if grep -q "$DONE_GREP" "$CODER_LOG"; then
    echo
    echo "Completion marker found in coder output."
    echo "Coder log: $CODER_LOG"
    exit 0
  fi

  echo "Round $round complete."
  echo "Prompt:     $PROMPT_FILE"
  echo "Coder log:  $CODER_LOG"
  echo "Validation: $VALIDATION_FILE"
  echo "Diff:       $DIFF_FILE"

  sleep "$SLEEP_BETWEEN_ROUNDS"
done

echo
echo "Reached MAX_ROUNDS=$MAX_ROUNDS without completion marker."
echo "Check logs in: $LOGS_DIR"
exit 2
