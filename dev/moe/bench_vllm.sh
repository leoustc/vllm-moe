#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

PORT="${PORT:-8072}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs}"
STATE_FILE="${STATE_FILE:-${LOG_DIR}/${PORT}.env}"
HEALTH_TIMEOUT_SECONDS="${HEALTH_TIMEOUT_SECONDS:-900}"
HEALTH_POLL_SECONDS="${HEALTH_POLL_SECONDS:-5}"
UPSTREAM_PORTS="${UPSTREAM_PORTS:-8080 8081 8082 8083 8084 8085 8086 8087}"
BACKEND_PROBE_TIMEOUT_SECONDS="${BACKEND_PROBE_TIMEOUT_SECONDS:-120}"

VLLM_BIN="${VLLM_BIN:-${SCRIPT_DIR}/.venv/bin/vllm}"
MODEL_PATH="${MODEL_PATH:-/models/gemma-4-26B-A4B-it}"
BASE_URL="${BASE_URL:-http://127.0.0.1:${PORT}}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-default}"

# Benchmark workload. Defaults are intentionally short so `make test-case`
# gives a quick single-GPU throughput signal instead of a long soak test.
NUM_PROMPTS="${NUM_PROMPTS:-12}"
REQUEST_RATE="${REQUEST_RATE:-2}"
RANDOM_INPUT_LEN="${RANDOM_INPUT_LEN:-64}"
RANDOM_OUTPUT_LEN="${RANDOM_OUTPUT_LEN:-64}"

# Throughput sweep mode
# Example:
#   SWEEP=1 ./bench_vllm.sh
#   RPS_LIST="0.05 0.1 0.2 0.4 0.8 1.0" SWEEP=1 ./bench_vllm.sh
SWEEP="${SWEEP:-0}"
RPS_LIST="${RPS_LIST:-1 2 4}"

# Optional benchmark controls
TEMPERATURE="${TEMPERATURE:-0}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-8}"
NUM_WARMUPS="${NUM_WARMUPS:-1}"

# Save benchmark result files
SAVE_RESULT="${SAVE_RESULT:-1}"
SAVE_DETAILED="${SAVE_DETAILED:-0}"
RESULT_DIR="${RESULT_DIR:-${LOG_DIR}/bench-results}"

mkdir -p "${LOG_DIR}" "${RESULT_DIR}"

if [[ -f "${STATE_FILE}" ]]; then
    # shellcheck disable=SC1090
    source "${STATE_FILE}"
fi

if [[ ! -x "${VLLM_BIN}" ]]; then
    echo "ERROR: vLLM binary not found or not executable: ${VLLM_BIN}" >&2
    exit 1
fi

wait_for_health() {
    local port="$1"
    local elapsed=0

    while ! curl -fsS --max-time 2 "http://127.0.0.1:${port}/health" >/dev/null; do
        if (( elapsed >= HEALTH_TIMEOUT_SECONDS )); then
            echo "ERROR: endpoint ${port} did not become healthy within ${HEALTH_TIMEOUT_SECONDS}s" >&2
            return 1
        fi
        echo "Waiting for endpoint ${port} health... ${elapsed}s"
        sleep "${HEALTH_POLL_SECONDS}"
        elapsed=$((elapsed + HEALTH_POLL_SECONDS))
    done
}

wait_for_chat_ready() {
    local port="$1"
    local elapsed=0

    while ! curl -fsS --max-time "${BACKEND_PROBE_TIMEOUT_SECONDS}" \
        -H 'Content-Type: application/json' \
        -d '{"model":"default","messages":[{"role":"user","content":"Say ok."}],"max_tokens":1,"temperature":0}' \
        "http://127.0.0.1:${port}/v1/chat/completions" >/dev/null; do
        if (( elapsed >= HEALTH_TIMEOUT_SECONDS )); then
            echo "ERROR: endpoint ${port} did not pass chat probe within ${HEALTH_TIMEOUT_SECONDS}s" >&2
            return 1
        fi
        echo "Waiting for endpoint ${port} chat probe... ${elapsed}s"
        sleep "${HEALTH_POLL_SECONDS}"
        elapsed=$((elapsed + HEALTH_POLL_SECONDS))
    done
}

wait_for_benchmark_backends() {
    if [[ "${PORT}" == "80" ]]; then
        echo "Checking nginx benchmark backends: ${UPSTREAM_PORTS}"
        for upstream_port in ${UPSTREAM_PORTS}; do
            wait_for_health "${upstream_port}"
            wait_for_chat_ready "${upstream_port}"
        done
        wait_for_health "80"
        echo "All nginx benchmark backends are healthy."
        return
    fi

    wait_for_health "${PORT}"
}

wait_for_benchmark_backends

COMMON_ARGS=(
    bench serve
    --backend openai-chat
    --base-url "${BASE_URL}"
    --endpoint /v1/chat/completions
    --model "${MODEL_PATH}"
    --served-model-name "${SERVED_MODEL_NAME}"
    --dataset-name random
    --random-input-len "${RANDOM_INPUT_LEN}"
    --random-output-len "${RANDOM_OUTPUT_LEN}"
    --num-prompts "${NUM_PROMPTS}"
    --temperature "${TEMPERATURE}"
    --num-warmups "${NUM_WARMUPS}"
)

if [[ -n "${MAX_CONCURRENCY}" ]]; then
    COMMON_ARGS+=(--max-concurrency "${MAX_CONCURRENCY}")
fi

if [[ "${SAVE_RESULT}" == "1" ]]; then
    COMMON_ARGS+=(--save-result --result-dir "${RESULT_DIR}")
fi

if [[ "${SAVE_DETAILED}" == "1" ]]; then
    COMMON_ARGS+=(--save-detailed)
fi

run_one() {
    local rps="$1"
    local ts
    ts="$(date +%Y%m%d-%H%M%S)"

    echo
    echo "============================================================"
    echo "vLLM throughput benchmark"
    echo "============================================================"
    echo "BASE_URL              = ${BASE_URL}"
    echo "MODEL_PATH            = ${MODEL_PATH}"
    echo "SERVED_MODEL_NAME     = ${SERVED_MODEL_NAME}"
    echo "NUM_PROMPTS           = ${NUM_PROMPTS}"
    echo "REQUEST_RATE          = ${rps}"
    echo "RANDOM_INPUT_LEN      = ${RANDOM_INPUT_LEN}"
    echo "RANDOM_OUTPUT_LEN     = ${RANDOM_OUTPUT_LEN}"
    echo "TEMPERATURE           = ${TEMPERATURE}"
    echo "MAX_CONCURRENCY       = ${MAX_CONCURRENCY:-unlimited}"
    echo "NUM_WARMUPS           = ${NUM_WARMUPS}"
    echo "RESULT_DIR            = ${RESULT_DIR}"
    echo "TIMESTAMP             = ${ts}"
    echo "============================================================"
    echo

    "${VLLM_BIN}" "${COMMON_ARGS[@]}" \
        --request-rate "${rps}" \
        --metadata "mode=throughput,rps=${rps},input_len=${RANDOM_INPUT_LEN},output_len=${RANDOM_OUTPUT_LEN},num_prompts=${NUM_PROMPTS},timestamp=${ts}"
}

if [[ "${SWEEP}" == "1" ]]; then
    echo "Running RPS sweep: ${RPS_LIST}"
    for rps in ${RPS_LIST}; do
        run_one "${rps}"
    done
else
    run_one "${REQUEST_RATE}"
fi
