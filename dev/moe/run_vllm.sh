#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VLLM_BIN="${VLLM_BIN:-${SCRIPT_DIR}/.venv/bin/vllm}"

MODEL_PATH="${MODEL_PATH:-/models/gemma-4-26B-A4B-it}"
GPU="${GPU:-1}"
PORT="${PORT:-8071}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"

HOST="0.0.0.0"
SERVED_MODEL_NAME="default"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.4}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-12288}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
BASE_URL="http://127.0.0.1:${PORT}"
PID_FILE="${SCRIPT_DIR}/${PORT}.pid"
LOG_DIR="${SCRIPT_DIR}/logs"
RUN_TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_FILE="${LOG_DIR}/vllm_${PORT}_${RUN_TIMESTAMP}.log"
STATE_FILE="${LOG_DIR}/latest.env"
PORT_STATE_FILE="${LOG_DIR}/${PORT}.env"
STARTUP_TIMEOUT_SECONDS="600"
LOG_CHECK_INTERVAL_SECONDS="5"
TAIL_LOG_TO_CONSOLE="${TAIL_LOG_TO_CONSOLE:-0}"
TAIL_PID=""

cleanup_tail() {
    if [[ -n "${TAIL_PID}" ]] && kill -0 "${TAIL_PID}" 2>/dev/null; then
        kill "${TAIL_PID}" 2>/dev/null || true
        wait "${TAIL_PID}" 2>/dev/null || true
    fi
}
trap cleanup_tail EXIT

if ! command -v curl >/dev/null 2>&1; then
    echo "Missing required command: curl" >&2
    exit 1
fi

if [[ ! -x "${VLLM_BIN}" ]]; then
    echo "Missing vLLM executable: ${VLLM_BIN}" >&2
    echo "Run: make compile" >&2
    exit 1
fi

if [[ ! -e "${MODEL_PATH}" ]]; then
    echo "Missing model path: ${MODEL_PATH}" >&2
    exit 1
fi

echo "Launching vLLM:"
echo "  model: ${MODEL_PATH}"
echo "  gpu: ${GPU}"
echo "  port: ${PORT}"
echo "  gpu memory utilization: ${GPU_MEMORY_UTILIZATION}"
echo "  max num batched tokens: ${MAX_NUM_BATCHED_TOKENS}"
echo "  max model len: ${MAX_MODEL_LEN}"
echo "  vllm logging level: ${VLLM_LOGGING_LEVEL}"
echo "  extra vllm args: ${VLLM_EXTRA_ARGS:-<none>}"

mkdir -p "${LOG_DIR}"

if curl -fsS "${BASE_URL}/health" >/dev/null 2>&1; then
    echo "Health endpoint already responds on ${BASE_URL}." >&2
    echo "Refusing to run because the smoke test could hit an existing server." >&2
    exit 1
fi

rm -f "${PID_FILE}"

check_log_failure() {
    if [[ ! -f "${LOG_FILE}" ]]; then
        return 0
    fi

    if grep -Eiq \
        "Traceback|\\bException\\b|\\bERROR\\b|\\bCRITICAL\\b|RuntimeError|CUDA out of memory|Address already in use|No such file or directory" \
        "${LOG_FILE}"; then
        echo "Detected vLLM startup failure in log. Last log lines:" >&2
        tail -n 120 "${LOG_FILE}" >&2 || true
        exit 1
    fi
}

EXTRA_ARGS=()
if [[ -n "${VLLM_EXTRA_ARGS}" ]]; then
    read -r -a EXTRA_ARGS <<<"${VLLM_EXTRA_ARGS}"
fi
env CUDA_VISIBLE_DEVICES="${GPU}" \
    VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL}" \
    "${VLLM_BIN}" serve "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    "${EXTRA_ARGS[@]}" \
    >"${LOG_FILE}" 2>&1 &

echo "$!" >"${PID_FILE}"
echo "  pid: $(cat "${PID_FILE}")"
echo "  log: ${LOG_FILE}"

if [[ "${TAIL_LOG_TO_CONSOLE}" != "0" ]]; then
    tail -n +1 -F "${LOG_FILE}" &
    TAIL_PID="$!"
fi

cat >"${PORT_STATE_FILE}" <<EOF
PID_FILE="${PID_FILE}"
LOG_FILE="${LOG_FILE}"
BASE_URL="${BASE_URL}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME}"
RUN_TIMESTAMP="${RUN_TIMESTAMP}"
EOF
cp "${PORT_STATE_FILE}" "${STATE_FILE}"

for elapsed in $(seq 0 "${LOG_CHECK_INTERVAL_SECONDS}" "${STARTUP_TIMEOUT_SECONDS}"); do
    check_log_failure

    if curl -fsS "${BASE_URL}/health" >/dev/null 2>&1; then
        break
    fi

    if ! kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
        echo "vLLM exited before becoming healthy. Last log lines:" >&2
        tail -n 80 "${LOG_FILE}" >&2 || true
        rm -f "${PID_FILE}"
        exit 1
    fi

    sleep "${LOG_CHECK_INTERVAL_SECONDS}"
done

check_log_failure

if ! curl -fsS "${BASE_URL}/health" >/dev/null; then
    echo "vLLM did not become healthy within ${STARTUP_TIMEOUT_SECONDS}s." >&2
    tail -n 80 "${LOG_FILE}" >&2 || true
    exit 1
fi

echo
echo "vLLM is healthy."
echo "State: ${PORT_STATE_FILE}"
