# Instructions

This folder tracks MoE CPU offload feature development.

- Read `DESIGN.md` first. Cases are independent; do not mix behavior between
  cases.
- Read `FEATURE.md` for the current feature scope.
- Read `WORK.md` for the development checklist.
- Read `HARNESS.md` for test harness and Codex routine rules.
- Read and update `PROGRESS.md` for current status and handoffs.
- Record known bugs and diagnostic findings in `BUGS.md`.
- Use `Makefile` as the canonical test harness.
- Do not alter `Makefile` unless the user explicitly asks for Makefile changes.
- Codex coding agents should only modify or fix implementation code. Do not run
  the full Makefile sanity workflow inside the Codex agent routine.
- The outer `make test` loop is responsible for sanity and harness validation.
- Case selection precedence:
  dense model uses normal vLLM; MoE with `--moe-gpu-prefetch <num>` uses Case 2;
  MoE with `--moe-cpu-offload` uses Case 1; MoE with no flag uses normal vLLM.
- If both `--moe-gpu-prefetch <num>` and `--moe-cpu-offload` are set, Case 2
  wins and `--moe-cpu-offload` is ignored.
- Do not use `--moe-gpu-limit`; it has been dropped from this feature.
- Do not use `--moe-active-expert-cache`; it is not part of Case 1.
- Case 1 is done/frozen. With only `--moe-cpu-offload`, transfer active experts
  passively after router computation and split transfer into smaller waves when
  GPU RAM is tight. Do not alter Case 1 unless the user explicitly asks for a
  Case 1 fix.
- Case 1 uses a fresh per-layer/per-wave `global_expert_id -> gpu_slot_id` map:
  copy active experts into compact GPU slots, remap routed `topk_ids`, run fused
  expert compute with the staged slot count, then retire the slots. Do not add
  Case 2 active, working, missing, or hot pager-list semantics to Case 1.
- Keep implementation modules separate: Case 1 passive CPU offload lives in
  `moe_offload.py`; Case 2 GPU prefetch lives in `moe_gpu_prefetch.py`.
- Case 2 is the active target. With `--moe-gpu-prefetch <num>`, use a pager
  thread to manage GPU active expert model residency.
- For Case 2 work, prioritize active, working, and missing expert list ownership,
  pager-thread prefetch, cold expert retirement, observability, and focused
  tests.
- Port ladder: `8070` system sanity, `8071` local base, `8072` local MoE, `8073/8074` dual local MoE.
- Debug artifacts: `<port>.pid`, `logs/<port>.env`, `logs/vllm_<port>_<timestamp>.log`, and `logs/test_<port>_<timestamp>.log`.
- Use `./clean_vllm.sh` or `make clean-vllm` to stop all vLLM servers recorded
  by `<port>.pid` files.
- Startup logs should show whether MoE offload flags are ignored or enabled,
  which case is selected, total experts, active experts, and the case-specific
  transfer or prefetch settings.
