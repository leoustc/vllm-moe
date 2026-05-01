#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

PORT="${PORT:-8062}"
BASE_URL="${BASE_URL:-http://127.0.0.1:${PORT}}"
MODEL_PATH="${MODEL_PATH:-/models/gemma-4-26B-A4B-it}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-default}"
VLLM_BIN="${VLLM_BIN:-${HARNESS_DIR}/.venv/bin/vllm}"
PYTHON_BIN="${PYTHON_BIN:-${HARNESS_DIR}/.venv/bin/python}"
STATE_FILE="${STATE_FILE:-${HARNESS_DIR}/logs/${PORT}.env}"

PROMPT_GROUPS="${PROMPT_GROUPS:-100}"
REPEATS_PER_PROMPT="${REPEATS_PER_PROMPT:-32}"
NUM_PROMPTS="${NUM_PROMPTS:-$((PROMPT_GROUPS * REPEATS_PER_PROMPT))}"
OUTPUT_LEN="${OUTPUT_LEN:-64}"
REQUEST_RATE="${REQUEST_RATE:-8}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-16}"
TEMPERATURE="${TEMPERATURE:-0}"
NUM_WARMUPS="${NUM_WARMUPS:-1}"
SEED="${SEED:-0}"
HEALTH_TIMEOUT_SECONDS="${HEALTH_TIMEOUT_SECONDS:-900}"
HEALTH_POLL_SECONDS="${HEALTH_POLL_SECONDS:-5}"

LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs}"
DATA_DIR="${DATA_DIR:-${SCRIPT_DIR}/data}"
RESULT_DIR="${RESULT_DIR:-${LOG_DIR}/bench-results}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(TZ=Asia/Singapore date +%Y%m%dT%H%M%S_SGT)}"
DATASET_FILE="${DATASET_FILE:-${DATA_DIR}/repeated_prompts_${PROMPT_GROUPS}x${REPEATS_PER_PROMPT}.jsonl}"
BENCH_LOG="${LOG_DIR}/repeated_prompts_${PORT}_${RUN_TIMESTAMP}.log"
PAGER_LOG="${LOG_DIR}/repeated_prompts_${PORT}_${RUN_TIMESTAMP}.pager.log"
TAIL_PID=""

mkdir -p "${LOG_DIR}" "${DATA_DIR}" "${RESULT_DIR}"

if [[ -f "${STATE_FILE}" ]]; then
    # shellcheck disable=SC1090
    source "${STATE_FILE}"
fi

cleanup() {
    if [[ -n "${TAIL_PID}" ]] && kill -0 "${TAIL_PID}" 2>/dev/null; then
        kill "${TAIL_PID}" 2>/dev/null || true
        wait "${TAIL_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

wait_for_health() {
    local elapsed=0

    while ! curl -fsS --max-time 2 "${BASE_URL}/health" >/dev/null; do
        if (( elapsed >= HEALTH_TIMEOUT_SECONDS )); then
            echo "ERROR: endpoint ${BASE_URL} did not become healthy within ${HEALTH_TIMEOUT_SECONDS}s" >&2
            return 1
        fi
        echo "Waiting for endpoint ${BASE_URL} health... ${elapsed}s"
        sleep "${HEALTH_POLL_SECONDS}"
        elapsed=$((elapsed + HEALTH_POLL_SECONDS))
    done
}

generate_dataset() {
    "${PYTHON_BIN}" - "${DATASET_FILE}" "${PROMPT_GROUPS}" \
        "${REPEATS_PER_PROMPT}" "${OUTPUT_LEN}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
groups = int(sys.argv[2])
repeats = int(sys.argv[3])
output_len = int(sys.argv[4])

topics = [
    "sparse mixture-of-experts routing",
    "GPU memory pressure during inference",
    "KV cache allocation",
    "expert cache locality",
    "CPU pinned memory transfers",
    "router logits and top-k experts",
    "batch scheduling for LLM serving",
    "token bucket packing",
    "prefetch latency hiding",
    "active expert eviction",
]

with path.open("w", encoding="utf-8") as f:
    for group_id in range(groups):
        topic = topics[group_id % len(topics)]
        prompt = (
            f"Locality group {group_id:03d}: Explain {topic} for a vLLM "
            "MoE inference server in one concise paragraph. Include one "
            "practical observation about performance."
        )
        for repeat_id in range(repeats):
            f.write(
                json.dumps(
                    {
                        "prompt": prompt,
                        "output_tokens": output_len,
                        "group_id": group_id,
                        "repeat_id": repeat_id,
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )
PY
}

if [[ ! -x "${VLLM_BIN}" ]]; then
    echo "ERROR: vLLM binary not found or not executable: ${VLLM_BIN}" >&2
    exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "ERROR: missing Python helper: ${PYTHON_BIN}" >&2
    echo "Run: make base" >&2
    exit 1
fi

generate_dataset
wait_for_health

if [[ -n "${LOG_FILE:-}" && -f "${LOG_FILE}" ]]; then
    (
        tail -n 0 -F "${LOG_FILE}" | while IFS= read -r line; do
            case "${line}" in
                *"MoE expert pager"* | *"MoE GPU prefetch"* | *"[MoE GPU Prefetch]"*)
                    printf '%s\n' "${line}"
                    ;;
            esac
        done
    ) >"${PAGER_LOG}" &
    TAIL_PID="$!"
fi

echo "============================================================"
echo "Repeated-prompt locality benchmark"
echo "============================================================"
echo "BASE_URL           = ${BASE_URL}"
echo "MODEL_PATH         = ${MODEL_PATH}"
echo "SERVED_MODEL_NAME  = ${SERVED_MODEL_NAME}"
echo "PROMPT_GROUPS      = ${PROMPT_GROUPS}"
echo "REPEATS_PER_PROMPT = ${REPEATS_PER_PROMPT}"
echo "NUM_PROMPTS        = ${NUM_PROMPTS}"
echo "OUTPUT_LEN         = ${OUTPUT_LEN}"
echo "REQUEST_RATE       = ${REQUEST_RATE}"
echo "MAX_CONCURRENCY    = ${MAX_CONCURRENCY}"
echo "DATASET_FILE       = ${DATASET_FILE}"
echo "BENCH_LOG          = ${BENCH_LOG}"
echo "PAGER_LOG          = ${PAGER_LOG}"
echo "============================================================"

"${VLLM_BIN}" bench serve \
    --backend openai-chat \
    --base-url "${BASE_URL}" \
    --endpoint /v1/chat/completions \
    --model "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --dataset-name custom \
    --dataset-path "${DATASET_FILE}" \
    --disable-shuffle \
    --num-prompts "${NUM_PROMPTS}" \
    --request-rate "${REQUEST_RATE}" \
    --max-concurrency "${MAX_CONCURRENCY}" \
    --temperature "${TEMPERATURE}" \
    --num-warmups "${NUM_WARMUPS}" \
    --seed "${SEED}" \
    --save-result \
    --save-detailed \
    --result-dir "${RESULT_DIR}" \
    --metadata "mode=locality,groups=${PROMPT_GROUPS},repeats=${REPEATS_PER_PROMPT},output_len=${OUTPUT_LEN},timestamp=${RUN_TIMESTAMP}" \
    2>&1 | tee "${BENCH_LOG}"
