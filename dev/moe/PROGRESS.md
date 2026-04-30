# PROGRESS: MoE CPU Offload

## Current Branch

```text
moe
```

This branch implements independent MoE offload cases documented in
`DESIGN.md`.

Current module boundary:

- Case 1 passive CPU offload: `vllm/model_executor/layers/fused_moe/moe_offload.py`
- Case 2 GPU prefetch: `vllm/model_executor/layers/fused_moe/moe_gpu_prefetch.py`
- Shared layer/config code only selects the case and wires the chosen cache.

Read order for agents:

```text
DESIGN.md
FEATURE.md
WORK.md
HARNESS.md
REVIEW.md
BUGS.md
PROGRESS.md
```

## Case Selection

1. Dense model: ignore MoE offload flags and use normal vLLM.
2. MoE model with `--moe-gpu-prefetch <num>`: use Case 2.
3. MoE model with `--moe-cpu-offload`: use Case 1.
4. MoE model with no MoE offload flag: use normal vLLM.

If both `--moe-gpu-prefetch <num>` and `--moe-cpu-offload` are set, Case 2 wins
and `--moe-cpu-offload` is ignored.

## 2026-04-29 Update: Case 2 Focus

Current focus is Case 2: `--moe-gpu-prefetch <num>`.

Case 1 is done for the tested unquantized Gemma MoE path. Keep Case 1 behavior
untouched unless the user explicitly asks for a Case 1 fix.

Case 1 completion notes:

- Compact staged expert execution now passes the staged expert count and no
  expert map to the fused-MoE kernel after top-k ids are remapped. This fixes
  degenerate repetitive completions caused by mixing remapped slot ids with the
  original global expert space.
- Case 1 flow is now documented as a fresh per-layer/per-wave
  `global_expert_id -> gpu_slot_id` map. Case 1 does not maintain Case 2
  active, working, missing, or hot pager lists.
- Passive mode no longer allocates the full expert bank before routing-derived
  wave sizing.
- Per-wave cleanup now covers allocation, expert load, and compute failures.
- Passive split-wave execution rejects `apply_router_weight_on_input=True`
  instead of risking masked slot-0 contributions.
- Unsupported quantized MoE, expert parallelism, and overlapped shared experts
  are rejected earlier during FusedMoE initialization.
- Empty local routed expert counts return zero output instead of falling through
  to the normal kernel with offloaded parameters.
- Chat-based harness smoke testing validates coherent prompt/answer output for
  the Case 1 Gemma MoE endpoint.
- Case 1 logs use passive transfer wording instead of Case 2 pager wording.
- Stale stage terminology was removed from MoE offload code and current docs.

Case 2 active targets:

- Harden active, working, and missing expert list ownership.
- Continue tuning pager-thread behavior for prefetching missing experts and
  retiring cold resident experts.
- Keep Case 2 independent from Case 1 passive transfer semantics.
- Add focused tests for Case 2 list transitions, prefetch sizing, and runtime
  residency behavior.
- Validate Case 2 through the Makefile harness with chat-based endpoint tests.

## Case 1 Status

Flag:

```bash
--moe-cpu-offload
```

Implemented behavior:

- Dense models ignore `--moe-cpu-offload`.
- MoE models keep expert source weights CPU-backed.
- Router path, KV cache, and required runtime structures stay GPU-backed.
- Routing computes active experts per MoE layer.
- Case 1 passively transfers only the routed expert weights needed by each
  layer.
- For each layer/wave, CPU builds a temporary `global_expert_id -> gpu_slot_id`
  map, copies active experts into compact GPU slots, and remaps routed `topk_ids`
  to compact slot ids.
- Fused expert compute uses the staged slot count and no persistent expert map.
- If routed experts do not fit in one GPU staging wave, the runtime splits the
  transfer and compute into smaller waves.
- After each wave computes, the runtime retires the GPU expert copies and clears
  the temporary map.
- Case 1 does not maintain Case 2 active, working, missing, or hot expert lists.
- Case 1 forces eager execution because CPU routing metadata reads and dynamic
  expert transfers are not CUDA-graph-capture safe.
- Status: done/frozen. Do not modify Case 1 while focusing on Case 2 unless the
  user explicitly asks for a Case 1 change.
- Module: `vllm/model_executor/layers/fused_moe/moe_offload.py`.

Latest validation:

```bash
dev/moe/.venv/bin/python -m pytest tests/kernels/moe/test_moe_cpu_offload.py -q
dev/moe/.venv/bin/python -m pytest tests/v1/engine/test_engine_args.py -q
dev/moe/.venv/bin/python -m ruff check vllm/config/moe.py vllm/engine/moe_offload_cli.py vllm/model_executor/layers/fused_moe/layer.py vllm/model_executor/layers/fused_moe/moe_offload.py vllm/model_executor/layers/fused_moe/moe_gpu_prefetch.py vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py tests/kernels/moe/test_moe_cpu_offload.py tests/v1/engine/test_engine_args.py
```

Earlier harness validation passed for:

- dense/system sanity path with MoE flags ignored,
- single Gemma4 MoE endpoint with Case 1 passive offload,
- two Gemma4 MoE endpoints sharing one GPU with Case 1 passive offload.

## Case 2 Status

Flag:

```bash
--moe-gpu-prefetch <num>
```

Implemented behavior:

- CLI/config accepts `--moe-gpu-prefetch <num>`.
- Dense models ignore Case 2 and Case 1 flags.
- MoE models with `--moe-gpu-prefetch <num>` select Case 2.
- If both flags are set, Case 2 takes precedence.
- Effective prefetch size uses the requested value unless it is below the model
  active expert count; then it uses `ceil(active_expert_count * 1.5)`.
- Startup logs report total experts, active experts, requested prefetch, and
  effective prefetch.
- Case 2 implementation lives in
  `vllm/model_executor/layers/fused_moe/moe_gpu_prefetch.py`.
- The runtime tracks active, working, and missing expert lists in the Case 2
  module.
- GPU-side routed metadata builds token counts; CPU receives compact
  expert/count metadata for pager decisions.
- Request execution packages resident waves first and missing waves once, then
  waits for pager-loaded missing waves before fused MoE compute.
- Pager H2D copies use a small copy-stream pool controlled by
  `VLLM_MOE_PREFETCH_COPY_STREAMS`.

Active work:

- Tune active, working, and missing expert list ownership under heavier traffic.
- Improve pager-thread prefetch overlap and residency policy.
- Ensure request execution uses resident experts immediately and minimizes waits
  for missing experts.
- Add observability for pager state without polluting Case 1 passive logs.
- Validate single and dual endpoint Case 2 harness runs.

Recent implementation note:

- Case 1 and Case 2 are now split into independent implementation modules.
  Case 1 remains in `moe_offload.py` as passive CPU offload only. Case 2 lives
  in `moe_gpu_prefetch.py` with pager-thread prefetch, GPU-side routed metadata,
  resident/missing wave packaging, token-pressure load ordering, and copy-stream
  H2D staging.
- `make test-case` is now a short single-GPU throughput comparison. It runs Case
  1 and Case 2 sequentially on `TEST_CASE_GPU`, uses 8 benchmark prompts by
  default, and keeps the benchmark workload at 64 input / 64 output tokens.

## Known Limitations

- Quantized MoE expert offload is currently rejected for Case 1/Case 2.
- Expert parallelism is currently rejected for Case 1/Case 2.
- Overlapped shared experts are currently rejected for Case 1/Case 2.
- Case 2 pager-thread policy and overlap still need throughput tuning.
- Case 2 memory budgeting for low GPU utilization needs continued validation.

## Harness Notes

Use the Makefile as the canonical outer harness.

Do not alter `Makefile` unless explicitly requested.

Debug artifacts:

```text
<port>.pid
logs/<port>.env
logs/vllm_<port>_<timestamp>.log
logs/test_<port>_<timestamp>.log
```

Port ladder:

```text
8070 system sanity
8071 local base
8072 local MoE
8073/8074 dual local MoE
```

Use `./clean_vllm.sh` or `make clean-vllm` to stop servers recorded by port pid
files.
