#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PORT="${PORT:-8071}"
STATE_FILE="${SCRIPT_DIR}/logs/${PORT}.env"
PID_FILE="${SCRIPT_DIR}/${PORT}.pid"
BASE_URL="http://127.0.0.1:${PORT}"
SERVED_MODEL_NAME="default"
MAX_TOKENS="${MAX_TOKENS:-128}"
TAIL_LOG_TO_CONSOLE="${TAIL_LOG_TO_CONSOLE:-0}"
PAGER_LOG_TO_CONSOLE="${PAGER_LOG_TO_CONSOLE:-1}"
TAIL_PID=""
PYTHON_BIN="${SCRIPT_DIR}/.venv/bin/python"

PROMPTS=(
    "What model are you? Answer briefly, then write a two-sentence story about a robot gardener."
    "Explain sparse mixture-of-experts routing in one short paragraph."
    "Write a tiny fable about a fox, a GPU, and a missing expert."
    "Give three practical tips for debugging a slow model server."
    "In one paragraph, compare CPU memory and GPU memory for inference."
    "Tell a short story about two model endpoints sharing one GPU."
    "What is a health check in a server? Answer in simple terms."
    "Write a concise haiku about tokens moving through layers."
    "Explain why logs should include timestamps and ports."
    "Create a five-item checklist for validating an LLM endpoint."
)

json_escape() {
    local value="$1"
    value="${value//\\/\\\\}"
    value="${value//\"/\\\"}"
    value="${value//$'\n'/\\n}"
    printf "%s" "${value}"
}

extract_answer() {
    "${PYTHON_BIN}" - "$1" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as f:
    data = json.load(f)

choice = data.get("choices", [{}])[0]
message = choice.get("message") or {}
answer = message.get("content")
if answer is None:
    answer = choice.get("text", "")
print(answer)
PY
}

extract_metrics() {
    "${PYTHON_BIN}" - "$1" "$2" <<'PY'
import json
import sys

response_file = sys.argv[1]
elapsed_seconds = float(sys.argv[2])

with open(response_file, encoding="utf-8") as f:
    data = json.load(f)

usage = data.get("usage") or {}
prompt_tokens = int(usage.get("prompt_tokens") or 0)
completion_tokens = int(usage.get("completion_tokens") or 0)
total_tokens = int(usage.get("total_tokens") or prompt_tokens + completion_tokens)

completion_tps = completion_tokens / elapsed_seconds if elapsed_seconds > 0 else 0.0
total_tps = total_tokens / elapsed_seconds if elapsed_seconds > 0 else 0.0

print(
    f"{prompt_tokens}\t{completion_tokens}\t{total_tokens}\t"
    f"{elapsed_seconds:.3f}\t{completion_tps:.2f}\t{total_tps:.2f}"
)
PY
}

cleanup_tail() {
    if [[ -n "${TAIL_PID}" ]] && kill -0 "${TAIL_PID}" 2>/dev/null; then
        kill "${TAIL_PID}" 2>/dev/null || true
        wait "${TAIL_PID}" 2>/dev/null || true
    fi
}
trap cleanup_tail EXIT

if [[ -f "${STATE_FILE}" ]]; then
    # shellcheck disable=SC1090
    source "${STATE_FILE}"
fi

if [[ ! -f "${PID_FILE}" ]]; then
    echo "Missing PID file for port ${PORT}: ${PID_FILE}" >&2
    echo "Run: PORT=${PORT} ./run_vllm.sh" >&2
    exit 1
fi

if ! kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
    echo "vLLM PID from ${PID_FILE} is not running." >&2
    rm -f "${PID_FILE}"
    exit 1
fi

if ! curl -fsS "${BASE_URL}/health" >/dev/null; then
    echo "vLLM health check failed on ${BASE_URL}." >&2
    if [[ -n "${LOG_FILE:-}" ]]; then
        tail -n 120 "${LOG_FILE}" >&2 || true
    fi
    exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Missing Python helper: ${PYTHON_BIN}" >&2
    echo "Run: make base" >&2
    exit 1
fi

RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
RESPONSE_FILE="${SCRIPT_DIR}/logs/test_${PORT}_${RUN_TIMESTAMP}.log"
: >"${RESPONSE_FILE}"

if [[ "${TAIL_LOG_TO_CONSOLE}" != "0" && -n "${LOG_FILE:-}" ]]; then
    echo "Tailing vLLM log during curl test: ${LOG_FILE}"
    tail -n 80 -F "${LOG_FILE}" &
    TAIL_PID="$!"
elif [[ "${PAGER_LOG_TO_CONSOLE}" != "0" && -n "${LOG_FILE:-}" ]]; then
    echo "Tailing MoE pager status during curl test: ${LOG_FILE}"
    (
        tail -n 0 -F "${LOG_FILE}" | while IFS= read -r line; do
            case "${line}" in
                *"MoE expert pager"* | *"MoE GPU prefetch"*)
                    printf '%s\n' "${line}"
                    ;;
            esac
        done
    ) &
    TAIL_PID="$!"
fi

echo "Running curl smoke test..."
total_prompt_tokens=0
total_completion_tokens=0
total_tokens=0
total_elapsed_ms=0
for index in "${!PROMPTS[@]}"; do
    prompt="${PROMPTS[${index}]}"
    request_id=$((index + 1))
    tmp_response="$(mktemp)"
    tmp_payload="$(mktemp)"
    tmp_time="$(mktemp)"
    escaped_prompt="$(json_escape "${prompt}")"
    {
        printf "{"
        printf "\"model\":\"%s\"," "${SERVED_MODEL_NAME}"
        printf "\"messages\":[{\"role\":\"user\",\"content\":\"%s\"}]," "${escaped_prompt}"
        printf "\"max_tokens\":%s," "${MAX_TOKENS}"
        printf "\"temperature\":0"
        printf "}"
    } >"${tmp_payload}"
    echo "Curl smoke test ${request_id}/${#PROMPTS[@]}: ${prompt}"
    if ! curl -fsS "${BASE_URL}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        --data-binary @"${tmp_payload}" \
        -w "%{time_total}" \
        -o "${tmp_response}" \
        >"${tmp_time}"; then
        rm -f "${tmp_response}" "${tmp_payload}" "${tmp_time}"
        echo "Curl smoke test ${request_id} failed. Last log lines:" >&2
        if [[ -n "${LOG_FILE:-}" ]]; then
            tail -n 120 "${LOG_FILE}" >&2 || true
        fi
        exit 1
    fi
    answer="$(extract_answer "${tmp_response}")"
    elapsed_seconds="$(cat "${tmp_time}")"
    metrics="$(extract_metrics "${tmp_response}" "${elapsed_seconds}")"
    IFS=$'\t' read -r prompt_tokens completion_tokens request_total_tokens \
        elapsed_display completion_tps total_tps <<<"${metrics}"
    total_prompt_tokens=$((total_prompt_tokens + prompt_tokens))
    total_completion_tokens=$((total_completion_tokens + completion_tokens))
    total_tokens=$((total_tokens + request_total_tokens))
    elapsed_ms="$("${PYTHON_BIN}" - "${elapsed_seconds}" <<'PY'
import sys
print(int(round(float(sys.argv[1]) * 1000)))
PY
)"
    total_elapsed_ms=$((total_elapsed_ms + elapsed_ms))
    echo "  latency=${elapsed_display}s completion_tokens=${completion_tokens} completion_tps=${completion_tps} total_tps=${total_tps}"

    {
        echo "===== endpoint ${BASE_URL} request ${request_id}/${#PROMPTS[@]} ====="
        echo "prompt ${request_id}: ${prompt}"
        echo "metrics ${request_id}: latency=${elapsed_display}s prompt_tokens=${prompt_tokens} completion_tokens=${completion_tokens} total_tokens=${request_total_tokens} completion_tps=${completion_tps} total_tps=${total_tps}"
        echo "answer ${request_id}:"
        echo "${answer}"
        echo
        echo "raw response ${request_id}:"
        cat "${tmp_response}"
        echo
    } >>"${RESPONSE_FILE}"
    rm -f "${tmp_response}" "${tmp_payload}" "${tmp_time}"
done

summary="$("${PYTHON_BIN}" - "${#PROMPTS[@]}" "${total_prompt_tokens}" "${total_completion_tokens}" "${total_tokens}" "${total_elapsed_ms}" <<'PY'
import sys

requests = int(sys.argv[1])
prompt_tokens = int(sys.argv[2])
completion_tokens = int(sys.argv[3])
total_tokens = int(sys.argv[4])
elapsed_seconds = int(sys.argv[5]) / 1000.0
completion_tps = completion_tokens / elapsed_seconds if elapsed_seconds > 0 else 0.0
total_tps = total_tokens / elapsed_seconds if elapsed_seconds > 0 else 0.0
print(
    f"===== endpoint throughput summary =====\n"
    f"requests={requests} elapsed={elapsed_seconds:.3f}s "
    f"prompt_tokens={prompt_tokens} completion_tokens={completion_tokens} "
    f"total_tokens={total_tokens} completion_tps={completion_tps:.2f} "
    f"total_tps={total_tps:.2f}"
)
PY
)"
{
    echo "${summary}"
    echo
} >>"${RESPONSE_FILE}"

cat "${RESPONSE_FILE}"
echo
echo "LLM smoke test passed."
