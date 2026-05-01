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
- `make test-case2` uses a higher-pressure short benchmark by default:
  `TEST_CASE2_BENCH_NUM_PROMPTS=16`, `TEST_CASE2_BENCH_REQUEST_RATE=8`, and
  `TEST_CASE2_BENCH_CONCURRENCY=16`. This improved the observed Case 2 output
  throughput from about `30 tok/s` to about `55 tok/s` on the same single-GPU
  setup while keeping the benchmark short.

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

## 2026-05-01 Handoff: Harness Split and Locality Bench

Repository split:

- Source implementation remains in `leoustc/vllm-moe`.
- Harness/docs/scripts are being split into `leoustc/harness-vllm-moe`.
- The current harness folder is `dev/moe` in the source checkout, but the user
  plans to reorganize the harness folder path and reload Codex later.
- Keep implementation/source changes separate from harness changes unless the
  user explicitly asks to sync them.

Harness repo status:

- Pushed `dev/moe` harness contents to
  `git@github.com:leoustc/harness-vllm-moe.git`.
- The push used HTTPS through `gh` auth because local SSH clone access was
  rejected, but the target repo is still the requested GitHub repo.
- Harness repo commit:
  `4d0a1a3 Add vLLM MoE harness`.
- Published harness files include docs, `Makefile`, `run_vllm.sh`,
  `bench_vllm.sh`, `test_vllm.sh`, `clean_vllm.sh`, `run_codex.sh`,
  `bench_page_size.sh`, and `site.conf`.
- Generated artifacts were intentionally excluded: `.venv/`, `logs/`,
  `.coding/`, `.debug/`, `.review/`, `*.pid`, and bench log outputs.

Recent benchmark PRs in source repo:

- PR #8 merged: `[codex] Add Case 2 benchmark sweep report`.
- PR #9 opened as draft against `main`:
  `https://github.com/leoustc/vllm-moe/pull/9`.
- Latest benchmark docs updated in both `BENCHMARK.md` and
  `dev/moe/CASE2_BENCH_SESSION1.md`.

Case 2 benchmark results so far:

| GPU limit | Prefetch num | Output tok/s | Status |
|---:|---:|---:|---|
| 0.50 | 16 | 20.43 | OK |
| 0.50 | 32 | NA | KV cache startup failure |
| 0.50 | 64 | NA | KV cache startup failure |
| 0.75 | 16 | 20.64 | OK |
| 0.75 | 32 | 34.78 | OK |
| 0.75 | 64 | NA | KV cache startup failure |
| 0.95 | 16 | 21.71 | OK |
| 0.95 | 32 | 33.86 | OK |
| 0.95 | 64 | 56.51 | OK |
| 0.95 | 72 | 60.31 | OK |
| 0.95 | 96 | NA | active expert cache startup failure |

Important benchmark logs:

- `logs/test_case2_grid_20260430T091110Z.log`:
  `0.50` and `0.75` rows.
- `logs/test_case2_gpu095_20260430T213450_SGT.log`:
  `0.95` rows for prefetch `16`, `32`, `64`.
- `logs/test_case2_gpu095_prefetch72_96_20260430T220459_SGT.log`:
  `0.95` rows for prefetch `72`, `96`.
- Prefetch `96` failed at startup because layer `29` could not allocate active
  expert cache memory: required `1141899264` bytes, only `459800576` bytes free
  after retries.

Current locality test plan:

- A new harness-only folder `locality/` has been added under the current
  harness path.
- The goal is to test active-expert locality by sending repeated prompts to the
  same vLLM endpoint and inspecting Case 2 pager behavior.
- Use vLLM's own benchmark command instead of a hand-written curl loop.
- The locality dataset shape is `100` prompt groups, each repeated `32` times,
  for `3200` total requests.
- The dataset is a custom JSONL file for `vllm bench serve` with rows containing
  `prompt`, `output_tokens`, `group_id`, and `repeat_id`.
- The script runs with `--dataset-name custom`, `--dataset-path <jsonl>`, and
  `--disable-shuffle` so each 32-repeat prompt group stays adjacent.

Current locality files:

- `locality/bench_repeated_prompts.sh`
- `locality/README.md`
- `.gitignore` now ignores generated `locality/logs` and `locality/data`.

Default locality command:

```bash
PORT=8062 ./locality/bench_repeated_prompts.sh
```

Default locality settings:

```text
PROMPT_GROUPS=100
REPEATS_PER_PROMPT=32
NUM_PROMPTS=3200
OUTPUT_LEN=64
REQUEST_RATE=8
MAX_CONCURRENCY=16
TEMPERATURE=0
```

Locality outputs:

- Generated dataset:
  `locality/data/repeated_prompts_100x32.jsonl`
- Full benchmark output:
  `locality/logs/repeated_prompts_<port>_<timestamp>.log`
- Captured MoE pager lines:
  `locality/logs/repeated_prompts_<port>_<timestamp>.pager.log`
- vLLM detailed result JSON:
  `locality/logs/bench-results/`

Validation already done for locality harness:

```bash
bash -n locality/bench_repeated_prompts.sh
PROMPT_GROUPS=2 REPEATS_PER_PROMPT=3 OUTPUT_LEN=7 DATA_DIR=<tmp>/data LOG_DIR=<tmp>/logs BASE_URL=http://127.0.0.1:9 HEALTH_TIMEOUT_SECONDS=0 ./locality/bench_repeated_prompts.sh
```

The smoke command intentionally failed health check on port `9`, but it
successfully generated a six-line dataset with three adjacent repeats for each
of two prompt groups.

Next plan after reload/reorg:

1. Place `locality/` in the reorganized harness repo path.
2. Start a Case 2 vLLM server, likely with the current best single-GPU setting:
   `GPU_MEMORY_UTILIZATION=0.95` and `--moe-gpu-prefetch 72`.
3. Run `PORT=<server_port> ./locality/bench_repeated_prompts.sh`.
4. Compare locality benchmark throughput and pager log behavior against the
   random-prompt Case 2 benchmark.
5. Inspect whether repeated prompt groups reduce missing expert pressure and
   improve resident expert hit rate.
