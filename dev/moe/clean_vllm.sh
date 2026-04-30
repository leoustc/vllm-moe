#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

shopt -s nullglob
PID_FILES=("${SCRIPT_DIR}"/*.pid)

read_pid_file() {
    local pid_file="$1"
    awk '{ for (i = 1; i <= NF; i++) if ($i ~ /^[0-9]+$/) print $i }' "${pid_file}" \
        2>/dev/null || true
}

collect_descendants() {
    local pid="$1"
    local child

    while read -r child; do
        [[ -z "${child}" ]] && continue
        collect_descendants "${child}"
        echo "${child}"
    done < <(pgrep -P "${pid}" 2>/dev/null || true)
}

pid_is_running() {
    local pid="$1"
    kill -0 "${pid}" 2>/dev/null
}

stop_pid() {
    local port="$1"
    local pid="$2"

    if pid_is_running "${pid}"; then
        echo "Stopping vLLM on port ${port}, pid ${pid}"
        kill "${pid}" 2>/dev/null || true
    else
        echo "vLLM pid ${pid} for port ${port} is not running."
        return
    fi

    for _ in $(seq 1 30); do
        if ! pid_is_running "${pid}"; then
            return
        fi
        sleep 1
    done

    if pid_is_running "${pid}"; then
        echo "Force stopping vLLM on port ${port}, pid ${pid}" >&2
        kill -9 "${pid}" 2>/dev/null || true
    fi
}

if (( ${#PID_FILES[@]} == 0 )); then
    echo "No vLLM PID files found."
    exit 0
fi

for pid_file in "${PID_FILES[@]}"; do
    port="$(basename "${pid_file}" .pid)"
    mapfile -t recorded_pids < <(read_pid_file "${pid_file}")

    if (( ${#recorded_pids[@]} == 0 )); then
        echo "Removing empty or invalid PID file: ${pid_file}"
        rm -f "${pid_file}"
        continue
    fi

    pids=()
    for pid in "${recorded_pids[@]}"; do
        while read -r child_pid; do
            [[ -z "${child_pid}" ]] && continue
            pids+=("${child_pid}")
        done < <(collect_descendants "${pid}")
        pids+=("${pid}")
    done

    mapfile -t pids < <(printf "%s\n" "${pids[@]}" | awk '!seen[$0]++')

    for pid in "${pids[@]}"; do
        stop_pid "${port}" "${pid}"
    done

    rm -f "${pid_file}"
done
