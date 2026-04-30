# HARNESS: Codex Loop Contract

This file defines how Codex agents interact with the Makefile harness.

## Ownership

- The outer shell/Makefile loop owns validation.
- Codex owns diagnosis and code fixes only.
- `Makefile` is the canonical harness and should be treated as read-only unless
  the user explicitly asks for Makefile changes.

## Codex Workspace Use

`run_codex.sh` copies this file into `.debug/HARNESS.md` and
`.coding/HARNESS.md` for each Codex run. Agents should read the local copy for
workspace rules and use the common file in `dev/moe/HARNESS.md` as the source
of truth when updating the harness contract.

## Outer Loop

The outer loop runs:

```bash
make test
```

`make test` owns:

- compile,
- vLLM launch,
- health checks,
- smoke tests,
- cleanup,
- result logs.

Codex agents should not reproduce this workflow manually.

## Diagnostic Codex Pass

The diagnostic pass should:

- inspect the newest files in `logs/`,
- identify the failing command or Make target,
- summarize the primary error,
- write/update `BUGS.md`,
- write a concise bug report when requested by `run_codex.sh`,
- avoid code changes.

## Coding Codex Pass

The coding pass should:

- read `BUGS.md`, `PROGRESS.md`, `DESIGN.md`, `FEATURE.md`, and `WORK.md`,
- modify only implementation code needed to fix the reported issue,
- not edit `Makefile`,
- not create another harness,
- not run full `make test`,
- optionally run narrow syntax/unit checks only when they directly validate the
  edited code and do not replace the outer harness,
- update `PROGRESS.md` and `BUGS.md` with what changed and what remains.

## Prohibited Inside Codex

- Do not run full sanity/harness targets such as `make test`, `make sanity`,
  `make test-local-base`, `make test-local-moe`, or
  `make test-local-moe-double`.
- Do not modify harness scripts or Makefile unless explicitly asked.
- Do not change ports, model paths, or test topology unless explicitly asked.
- Do not treat Codex-local checks as final validation.

## Logs

- vLLM logs: `logs/vllm_<port>_<timestamp>.log`
- Smoke-test logs: `logs/test_<port>_<timestamp>.log`
- Per-port state: `logs/<port>.env`
- vLLM PID files: `<port>.pid`

Use `./clean_vllm.sh` or `make clean-vllm` only when cleanup is explicitly
needed. The normal validation loop should handle cleanup through the harness.
