#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
DEBUG_DIR="${SCRIPT_DIR}/.debug"
CODING_DIR="${SCRIPT_DIR}/.coding"
RUN_TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
BUG_SESSION_ID="${BUG_SESSION_ID:-$(uuidgen)}"
CODE_SESSION_ID="${CODE_SESSION_ID:-$(uuidgen)}"
BUG_CODEX_SESSION_ID="${BUG_CODEX_SESSION_ID:-}"
CODE_CODEX_SESSION_ID="${CODE_CODEX_SESSION_ID:-}"
BUG_REPORT="${LOG_DIR}/bug_${RUN_TIMESTAMP}_${BUG_SESSION_ID}.md"
CHECK_LOG_FILE="${LOG_DIR}/codex_check_${RUN_TIMESTAMP}_${BUG_SESSION_ID}.log"
CODE_LOG_FILE="${LOG_DIR}/codex_code_${RUN_TIMESTAMP}_${CODE_SESSION_ID}.log"
CHECK_PROMPT_FILE="${LOG_DIR}/codex_check_${RUN_TIMESTAMP}_${BUG_SESSION_ID}.prompt.md"
CODE_PROMPT_FILE="${LOG_DIR}/codex_code_${RUN_TIMESTAMP}_${CODE_SESSION_ID}.prompt.md"

DOCS=(
    "AGENTS.md"
    "DESIGN.md"
    "FEATURE.md"
    "WORK.md"
    "REVIEW.md"
    "HARNESS.md"
    "PROGRESS.md"
)

for doc in "${DOCS[@]}"; do
    if [[ ! -f "${SCRIPT_DIR}/${doc}" ]]; then
        echo "Missing required doc: ${SCRIPT_DIR}/${doc}" >&2
        exit 1
    fi
done

PRIMARY_INSTRUCTION="${1:-Continue the MoE CPU offload implementation from the current progress.}"
TASK="${*:-${PRIMARY_INSTRUCTION} Follow HARNESS.md: diagnose or fix code only, and leave full make test validation to the outer loop.}"
mkdir -p "${LOG_DIR}" "${DEBUG_DIR}" "${CODING_DIR}"

cp "${SCRIPT_DIR}/AGENTS.md" "${DEBUG_DIR}/AGENTS.md"
cp "${SCRIPT_DIR}/HARNESS.md" "${DEBUG_DIR}/HARNESS.md"
cat >>"${DEBUG_DIR}/AGENTS.md" <<EOF

# Debug Session Context

- Repo dir: ${REPO_DIR}
- Workspace dir: ${DEBUG_DIR}
- Common workspace dir: ${SCRIPT_DIR}
- Local harness rules: ${DEBUG_DIR}/HARNESS.md
- Common harness source of truth: ${SCRIPT_DIR}/HARNESS.md
- Inspect logs in: ${LOG_DIR}
- Do not modify code in the repo.
- Write diagnostic bug reports to ${LOG_DIR}.
- Record shared diagnostic progress in ${SCRIPT_DIR}/PROGRESS.md.
- Record shared known bugs in ${SCRIPT_DIR}/BUGS.md.
- Record durable diagnostic memory in ${DEBUG_DIR}/MEMORY.md.
EOF

cp "${SCRIPT_DIR}/AGENTS.md" "${CODING_DIR}/AGENTS.md"
cp "${SCRIPT_DIR}/HARNESS.md" "${CODING_DIR}/HARNESS.md"
cat >>"${CODING_DIR}/AGENTS.md" <<EOF

# Coding Session Context

- Repo dir: ${REPO_DIR}
- Workspace dir: ${CODING_DIR}
- Common workspace dir: ${SCRIPT_DIR}
- Local harness rules: ${CODING_DIR}/HARNESS.md
- Common harness source of truth: ${SCRIPT_DIR}/HARNESS.md
- Make code changes in ${REPO_DIR}, not in this scratch folder.
- Read the latest bug report before coding.
- Read shared progress from ${SCRIPT_DIR}/PROGRESS.md.
- Record shared coding progress in ${SCRIPT_DIR}/PROGRESS.md.
- Record shared known bugs in ${SCRIPT_DIR}/BUGS.md.
- Record durable coding memory in ${CODING_DIR}/MEMORY.md.
EOF

cat >"${CHECK_PROMPT_FILE}" <<EOF
You are running from diagnostic folder:
${DEBUG_DIR}

Directory roles:
- Repo dir:
${REPO_DIR}
- Workspace dir:
${DEBUG_DIR}
- Common workspace dir:
${SCRIPT_DIR}

Read these files:
- ${SCRIPT_DIR}/AGENTS.md
- ${SCRIPT_DIR}/DESIGN.md
- ${SCRIPT_DIR}/FEATURE.md
- ${SCRIPT_DIR}/WORK.md
- ${SCRIPT_DIR}/HARNESS.md
- ${DEBUG_DIR}/HARNESS.md
- ${SCRIPT_DIR}/PROGRESS.md

Record shared diagnostic progress and handoff hints in:
${SCRIPT_DIR}/PROGRESS.md

Record shared known bugs in:
${SCRIPT_DIR}/BUGS.md

Record durable diagnostic memory in:
${DEBUG_DIR}/MEMORY.md

Task:
Inspect the latest debug artifacts in ${SCRIPT_DIR}/logs and identify the current failing behavior.
Do not modify code.
Write a concise bug report to:
${BUG_REPORT}

The bug report must include:
- bug-report session id: ${BUG_SESSION_ID},
- failing command or Make target,
- relevant log file path,
- primary error message,
- likely root cause,
- recommended code area to inspect next.
EOF

cat >"${CODE_PROMPT_FILE}" <<EOF
Highest priority user instruction:
${PRIMARY_INSTRUCTION}

You are running from coding folder:
${CODING_DIR}

Directory roles:
- Repo dir:
${REPO_DIR}
- Workspace dir:
${CODING_DIR}
- Common workspace dir:
${SCRIPT_DIR}

Read and follow these files before coding:
- ${SCRIPT_DIR}/AGENTS.md
- ${SCRIPT_DIR}/DESIGN.md
- ${SCRIPT_DIR}/FEATURE.md
- ${SCRIPT_DIR}/WORK.md
- ${SCRIPT_DIR}/REVIEW.md
- ${SCRIPT_DIR}/HARNESS.md
- ${CODING_DIR}/HARNESS.md
- ${SCRIPT_DIR}/PROGRESS.md

Record shared coding progress and handoff hints in:
${SCRIPT_DIR}/PROGRESS.md

Record shared known bugs in:
${SCRIPT_DIR}/BUGS.md

Record durable coding memory in:
${CODING_DIR}/MEMORY.md

Full user requested coding task from run_codex.sh arguments:
${TASK}

If the highest priority user instruction mentions Case 1, prioritize the Case 1 findings in:
${SCRIPT_DIR}/REVIEW.md

First read this bug report from the diagnostic pass:
${BUG_REPORT}

Coding session id:
${CODE_SESSION_ID}

Important workflow requirements:
- Preserve default vLLM behavior unless --moe-cpu-offload is enabled.
- Only modify or fix implementation code.
- Do not alter ${SCRIPT_DIR}/Makefile unless the user explicitly asks for Makefile changes.
- Do not run the full Makefile sanity workflow inside this Codex routine; the outer make test loop handles sanity and harness validation.
- Record meaningful progress updates in ${SCRIPT_DIR}/PROGRESS.md.
- Record known bugs and bug status updates in ${SCRIPT_DIR}/BUGS.md.
- Do not use bare python3 or bare pip; use uv and .venv/bin/python.
EOF

echo "Launching Codex diagnostic step"
echo "  trace session: ${BUG_SESSION_ID}"
echo "  codex session: ${BUG_CODEX_SESSION_ID:-<last-or-new>}"
echo "  log: ${CHECK_LOG_FILE}"
echo "  prompt: ${CHECK_PROMPT_FILE}"
echo "  bug report: ${BUG_REPORT}"
if [[ -n "${BUG_CODEX_SESSION_ID}" ]]; then
    (cd "${DEBUG_DIR}" && codex exec resume --dangerously-bypass-approvals-and-sandbox "${BUG_CODEX_SESSION_ID}" - <"${CHECK_PROMPT_FILE}") 2>&1 | tee "${CHECK_LOG_FILE}"
else
    (
        cd "${DEBUG_DIR}"
        codex exec resume --last --dangerously-bypass-approvals-and-sandbox - <"${CHECK_PROMPT_FILE}" \
            || codex exec --dangerously-bypass-approvals-and-sandbox - <"${CHECK_PROMPT_FILE}"
    ) 2>&1 | tee "${CHECK_LOG_FILE}"
fi
check_status="${PIPESTATUS[0]}"

if [[ "${check_status}" != "0" ]]; then
    echo "Codex diagnostic step failed with status ${check_status}" >&2
    exit "${check_status}"
fi

if [[ ! -s "${BUG_REPORT}" ]]; then
    echo "Diagnostic step did not write bug report: ${BUG_REPORT}" >&2
    exit 1
fi

echo "Launching Codex coding step"
echo "  trace session: ${CODE_SESSION_ID}"
echo "  codex session: ${CODE_CODEX_SESSION_ID:-<last-or-new>}"
echo "  log: ${CODE_LOG_FILE}"
echo "  prompt: ${CODE_PROMPT_FILE}"
if [[ -n "${CODE_CODEX_SESSION_ID}" ]]; then
    (cd "${CODING_DIR}" && codex exec resume --dangerously-bypass-approvals-and-sandbox "${CODE_CODEX_SESSION_ID}" - <"${CODE_PROMPT_FILE}") 2>&1 | tee "${CODE_LOG_FILE}"
else
    (
        cd "${CODING_DIR}"
        codex exec resume --last --dangerously-bypass-approvals-and-sandbox - <"${CODE_PROMPT_FILE}" \
            || codex exec --dangerously-bypass-approvals-and-sandbox - <"${CODE_PROMPT_FILE}"
    ) 2>&1 | tee "${CODE_LOG_FILE}"
fi
exit "${PIPESTATUS[0]}"
