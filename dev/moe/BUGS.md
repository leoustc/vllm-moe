# Known Bugs

## 2026-04-30: clean_vllm.sh Assumed One PID Per Port

- Failing target: `make clean-vllm` / `./clean_vllm.sh`
- Symptom: newer vLLM launches can leave more than one relevant process for a
  single port, but the cleanup script read each `<port>.pid` file as one scalar
  PID and only stopped that process.

Likely cause: vLLM can spawn child worker/process-tracker processes under the
recorded server PID, and pid files may also contain multiple numeric PIDs during
manual diagnosis or future harness changes.

Fix: `clean_vllm.sh` now accepts multiple numeric PIDs per pid file, discovers
descendant processes with `pgrep -P`, stops descendants before their recorded
parent, deduplicates process ids, and still removes stale/invalid pid files.

## 2026-04-28: Case 1 Compact Expert Staging Produced Degenerate Output

- Failing target: latest Case 1 curl smoke output on ports `8072`/`8073`/`8074`.
- Logs: `logs/vllm_8072_20260428T103351Z.log`,
  `logs/test_8072_20260428T103351Z.log`.
- Symptom: completions repeat short fragments such as `one short paragraph`,
  `CPU-`, and `One-One` until `max_tokens`, or stop after one token.

Root cause: after Case 1 remapped routed expert ids into compact GPU staging
slots, the fused-MoE kernel call still received the original global expert count
and original expert map. With compact staged weights, alignment/padding must use
the staged expert count and no expert map.

Fix: Case 1 and Case 2 compact staging kernel calls now pass
`global_num_experts=<staged slot count>` and `expert_map=None` after remapping.
Existing vLLM servers must be restarted to pick up the fix.

## 2026-04-26: Gemma4 MoE Offload OOMs During FusedMoE Weight Creation

- Session: `41906786-63fd-41b0-88d5-5abd6f54d5cb`
- Failing target: `make -C dev/moe test-local-moe`
- Log: `/home/ubuntu/gpulab/vllm-moe/dev/moe/logs/vllm_8072_20260426T094041Z.log`
- Error:

```text
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 968.00 MiB.
```

The failing stack enters `Gemma4MoE -> FusedMoE -> UnquantizedFusedMoEMethod.create_weights()`
and allocates full expert tensors on CUDA via `torch.empty(...)`.

Likely cause: Case 1 offload initialization happened after full expert
allocation, so `--moe-cpu-offload` does not yet make CPU memory the source of
truth early enough to reduce startup GPU residency.

Next inspection area: `vllm/model_executor/layers/fused_moe/layer.py`,
`vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py`, and the
Case 1 runtime/cache code in
`vllm/model_executor/layers/fused_moe/moe_offload.py`.

Status update from session `7cdf95b0-5786-4405-a67a-aa50f8342702`:

- Fixed for the unquantized fused MoE path. With `--moe-cpu-offload`, full
  expert source tensors are now allocated on CPU and GPU execution uses a
  bounded staging cache.
- `make -C dev/moe test-local-moe` now passes for
  `/models/gemma-4-26B-A4B-it`.
- `make -C dev/moe test-local-moe-double` also passes on ports `8073/8074`.

Follow-up observations:

- Case 1 offload forces eager mode because the synchronous CPU scheduling and
  staging path is not CUDA-graph-capture safe.
- The smoke-test completion is repetitive. Track separately if deterministic
  correctness comparison shows offload-specific output divergence.

## 2026-04-27: GPT-OSS MXFP4 MoE Offload Leaves Too Little KV Cache

- Session: `f0c8c71d-ab8b-4555-8882-fb7afe19f680`
- Failing target: latest port `8071` local base/startup attempt, equivalent to
  `test-local-base-run`/`./run_vllm.sh` for `/models/gpt-oss-20b` with
  `--moe-cpu-offload`
- Log: `/home/ubuntu/gpulab/vllm-moe/dev/moe/logs/vllm_8071_20260427T035631Z.log`
- Error:

```text
ValueError: To serve at least one request with the models's max seq len (131072), (3.1 GiB KV cache is needed, which is larger than the available KV cache memory (2.44 GiB).
```

The run logs `MoE CPU offload enabled` for GPT-OSS, selects the MXFP4/MARLIN MoE
backend, then reports `Model loading took 13.72 GiB memory`. This suggests the
current Case 1 offload implementation does not yet cover GPT-OSS quantized
MXFP4 experts, so model/expert state remains too GPU-resident and starves KV
cache for the default 131072-token context.

Next inspection area: GPT-OSS MXFP4 fused MoE weight allocation/loading and
execution, especially `vllm/model_executor/layers/fused_moe/mxfp4.py`, related
quantized MoE method code, and GPT-OSS model integration. Also verify whether
the local base harness should use a true dense model instead of GPT-OSS for the
"flag ignored" check.

## 2026-04-27: Diagnostic Pass Has No vLLM Artifacts To Inspect

- Session: `9e26a9d6-b401-4552-9b91-d367f5c9d0ff`
- Failing target: diagnostic artifact inspection after the outer harness/Codex
  loop
- Log: `/home/ubuntu/gpulab/vllm-moe/dev/moe/logs/codex_check_20260427T074743Z_9e26a9d6-b401-4552-9b91-d367f5c9d0ff.log`
- Error:

```text
No current vllm_*.log, test_*.log, or logs/<port>.env artifacts exist in /home/ubuntu/gpulab/vllm-moe/dev/moe/logs.
```

The only current file in `logs/` is the Codex diagnostic transcript. It contains
context and older known-bug references, but no fresh vLLM startup log, smoke-test
log, or per-port state file. Therefore the diagnostic pass cannot identify a
current Make target/runtime failure.

Next inspection area: outer harness and Codex-loop artifact retention, especially
the flow that runs `make test`, cleanup, and `run_codex.sh`. Preserve the latest
`vllm_<port>_<timestamp>.log`, `test_<port>_<timestamp>.log`, and `logs/<port>.env`
files before starting the diagnostic Codex pass.

## 2026-04-27: Case 2 CLI Flag Is Not Registered

- Session: `f395816b-4dba-4e95-935e-2830cf4ce9a4`
- Failing target: `make test`, during `test-local-base-run` on port `8071`
- Log: `/home/ubuntu/gpulab/vllm-moe/dev/moe/logs/vllm_8071_20260427T083945Z.log`
- Error:

```text
vllm: error: unrecognized arguments: --moe-gpu-prefetch 10
```

`Makefile` defaults `VLLM_EXTRA_ARGS` to `--moe-gpu-prefetch 10`, matching the
new Case 2 design docs. The local vLLM CLI exits during argument parsing before
server startup, which means Case 2 public flag wiring has not been implemented
or is not loaded into the installed local CLI.

Next inspection area: MoE offload CLI/config wiring and engine argument parsing,
especially any existing `moe_offload_cli` integration, `EngineArgs`/async engine
args plumbing, and tests under `tests/v1/engine/test_engine_args.py`.

Status update from coding session `518c4a1a-da77-441f-9a7a-f7be8348f786`:

- Fixed the CLI/config registration gap for `--moe-gpu-prefetch <num>`.
- Dense models now ignore both MoE offload flags and leave MoE offload disabled.
- MoE models with `--moe-gpu-prefetch` now select config mode `prefetch`.
- If both `--moe-gpu-prefetch` and `--moe-cpu-offload` are set, config mode
  `prefetch` wins and the CPU-offload passive mode is not selected.
- Focused validation passed:

```bash
dev/moe/.venv/bin/python -m pytest tests/v1/engine/test_engine_args.py -q
dev/moe/.venv/bin/python -m pytest tests/kernels/moe/test_moe_cpu_offload.py -q
dev/moe/.venv/bin/python -m ruff check vllm/config/moe.py vllm/engine/moe_offload_cli.py vllm/model_executor/layers/fused_moe/layer.py tests/v1/engine/test_engine_args.py
```

Remaining status: CLI parsing is fixed, but Case 2 pager/runtime behavior is
not implemented yet. The next expected blocker is likely MoE runtime startup or
execution under `--moe-gpu-prefetch`, not argparse rejection.

## 2026-04-27: Case 2 Prefetch Mode OOMs During Gemma4 Expert Allocation

- Session: `8834623f-ef3f-4233-b03d-a7f3e513cb0a`
- Failing target: `make test`, during `test-local-moe-run` on port `8072`
- Log: `/home/ubuntu/gpulab/vllm-moe/dev/moe/logs/vllm_8072_20260427T085209Z.log`
- Error:

```text
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 968.00 MiB.
```

The run accepts `--moe-gpu-prefetch 10` and logs:

```text
MoE GPU prefetch enabled: total experts=128, active experts=8, requested prefetch=10, effective prefetch=10.
```

It then fails in `Gemma4MoE -> FusedMoE ->
UnquantizedFusedMoEMethod.create_weights()` at `torch.empty(...)`, meaning Case
2 still allocates the full unquantized expert bank on CUDA during model
construction.

Likely cause: Case 2 config/CLI selection is implemented, but the CPU-backed
expert allocation path is still limited to passive Case 1 mode. Prefetch mode
needs to reuse or generalize the CPU source-of-truth allocation path before
adding pager-thread residency management.

Next inspection area: `vllm/model_executor/layers/fused_moe/layer.py`,
`vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py`, and the
Case 2 runtime/cache code in
`vllm/model_executor/layers/fused_moe/moe_gpu_prefetch.py` that decides when
expert tensors are CPU-backed versus full CUDA allocations.

Status update from coding session `ba786881-b18e-4ff6-8af0-af8e33f65556`:

- Fixed the unquantized fused-MoE allocation gate so Case 2 prefetch mode also
  uses CPU-backed expert source tensors instead of full CUDA expert allocation.
- Added a bounded prefetch-mode expert cache budget from
  `effective_gpu_prefetch`.
- Added initial startup prefetch of the first local expert slots for each layer.
- Kept Case 1 passive semantics separate: passive mode still retires/releases
  GPU expert copies after each wave.
- Case 2 prefetch mode now keeps resident experts across routed work and loads
  missing experts into the bounded cache without evicting unrelated resident
  experts unless capacity requires it.
- Focused validation passed:

```bash
dev/moe/.venv/bin/python -m pytest tests/kernels/moe/test_moe_cpu_offload.py tests/v1/engine/test_engine_args.py -q
dev/moe/.venv/bin/python -m ruff check vllm/model_executor/layers/fused_moe/moe_offload.py vllm/model_executor/layers/fused_moe/layer.py vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py tests/kernels/moe/test_moe_cpu_offload.py vllm/config/moe.py vllm/engine/moe_offload_cli.py tests/v1/engine/test_engine_args.py
```

Remaining status: this should address the reported startup OOM boundary for the
unquantized Gemma4 fused-MoE path. Full Case 2 pager-thread behavior and
explicit active/working/missing list ownership are still incomplete.

## 2026-04-27: Case 2 Missing Expert Is Not Loaded Before Wave Tensor Construction

- Session: `f7d17da3-8334-43ae-9d33-0dc8eef3dc33`
- Failing target: `make test`, during the `test-local-moe-test` smoke request
  on port `8072`
- Log: `/home/ubuntu/gpulab/vllm-moe/dev/moe/logs/vllm_8072_20260427T090203Z.log`
- Error:

```text
RuntimeError: Expert 90 is not resident in the MoE offload cache
```

The MoE Case 2 run accepts `--moe-gpu-prefetch 10`, logs:

```text
MoE GPU prefetch enabled: total experts=128, active experts=8, requested prefetch=10, effective prefetch=10.
```

and reaches `/health`. The first completion request then enters the Gemma4
unquantized fused-MoE path, where `forward_native()` calls
`layer.moe_offload_cache.make_wave_tensors(...)` and aborts because routed
expert `90` is not resident.

Likely cause: Case 2 startup prefetch/cache setup is present, but the
request-time missing-expert path is incomplete. The forward path assumes every
expert in the wave is resident instead of loading missing experts, deferring
them to a missing list, or waiting for the pager to populate residency before
constructing wave tensors.

Current module note: Case 2 residency/load-missing flow now belongs to
`vllm/model_executor/layers/fused_moe/moe_gpu_prefetch.py`, not
`moe_offload.py`.

Next inspection area: `vllm/model_executor/layers/fused_moe/moe_gpu_prefetch.py`
(`MoeGpuPrefetchExpertCache`, residency/load-missing flow,
`make_wave_tensors`),
`vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py` Case 2
forward logic before `make_wave_tensors`, and the active/working/missing list
plus pager-thread implementation.

Status update from coding session `fa9adac4-c409-4bc7-9b0f-2e286cf3270a`:

- Fixed the immediate residency race in the synchronous Case 2 cache path.
- `MoeGpuPrefetchExpertCache.ensure_experts_resident(...)` protects every expert
  in the current requested wave while evicting for capacity, so a just-loaded or
  still-needed wave expert cannot be evicted before `make_wave_tensors(...)`.
- Added regression coverage for a prefetch cache where high-hotness resident
  experts would previously evict lower-hotness experts from the current wave.
- Focused validation passed:

```bash
dev/moe/.venv/bin/python -m pytest tests/kernels/moe/test_moe_cpu_offload.py tests/v1/engine/test_engine_args.py -q
dev/moe/.venv/bin/python -m ruff check vllm/model_executor/layers/fused_moe/moe_offload.py tests/kernels/moe/test_moe_cpu_offload.py vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py vllm/model_executor/layers/fused_moe/layer.py vllm/config/moe.py vllm/engine/moe_offload_cli.py tests/v1/engine/test_engine_args.py
```

Remaining status: this should address the reported `Expert 90 is not resident`
crash in the current foreground-loading prefetch path. Full Case 2
active/working/missing lists and detached pager-thread behavior are still not
implemented.

## 2026-04-27: Case 2 Dual-Endpoint Startup Has Insufficient KV Cache

- Session: `4169149a-3b8f-4c38-b1e5-18e31a1bdba3`
- Failing target: `make test`, during `test-local-moe-double-run` on port
  `8073`
- Log: `/home/ubuntu/gpulab/vllm-moe/dev/moe/logs/vllm_8073_20260427T091847Z.log`
- Error:

```text
ValueError: To serve at least one request with the models's max seq len (262144), (5.98 GiB KV cache is needed, which is larger than the available KV cache memory (4.91 GiB). Based on the available memory, the estimated maximum model length is 205984.
```

The latest run shows progress past the earlier single-endpoint failure:

- Port `8071` dense/base ignores `--moe-gpu-prefetch 10` and passes.
- Port `8072` single MoE Case 2 reaches health and passes the smoke request.
- Port `8073`, the first dual-endpoint MoE server, starts with
  `GPU_MEMORY_UTILIZATION=0.4`, logs Case 2 prefetch enabled, loads the model
  using `9.29 GiB`, then reports only `4.91 GiB` available KV cache memory.

Likely cause: the dual-endpoint Case 2 memory budget is too tight for Gemma4's
default `max_seq_len=262144` after model load plus prefetch/cache residency.
With `gpu_memory_utilization=0.4`, vLLM cannot reserve the `5.98 GiB` KV cache
needed to serve even one max-length request.

Next inspection area: Case 2 GPU memory budgeting for low-utilization
dual-server runs, especially prefetch cache sizing in
`vllm/model_executor/layers/fused_moe/moe_gpu_prefetch.py` and selection/allocation
wiring in `vllm/model_executor/layers/fused_moe/layer.py`, plus harness/runtime
sizing in `dev/moe/Makefile` and `dev/moe/run_vllm.sh` only if the desired fix is
to cap `max_model_len` or adjust dual-endpoint memory settings.

Status update from coding session `d82015ce-edbe-4483-8b36-8dda07ca0ecb`:

- Added a low-memory guard for Case 2 startup prefetch.
- When `gpu_memory_utilization <= 0.5`, Case 2 now defers the initial GPU
  prefetch allocation instead of loading the prefetch slots during model weight
  processing.
- The bounded prefetch cache remains configured from `effective_gpu_prefetch`;
  missing/requested experts are still loaded during the foreground request path.
- This targets the dual-endpoint `GPU_MEMORY_UTILIZATION=0.4` failure where
  startup prefetch residency reduced available KV memory below the model's
  full-context requirement.
- Focused validation passed:

```bash
dev/moe/.venv/bin/python -m pytest tests/kernels/moe/test_moe_cpu_offload.py tests/v1/engine/test_engine_args.py -q
dev/moe/.venv/bin/python -m ruff check vllm/model_executor/layers/fused_moe/layer.py vllm/model_executor/layers/fused_moe/moe_offload.py tests/kernels/moe/test_moe_cpu_offload.py vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py vllm/config/moe.py vllm/engine/moe_offload_cli.py tests/v1/engine/test_engine_args.py
```

Remaining status: this should reduce model-load residency for the
dual-endpoint Case 2 startup path without changing the harness. Full outer-loop
validation is still required to confirm the available KV memory exceeds the
262144-token requirement on port `8073`.

Status update from coding session `26176587-a8fe-45bb-85d3-6bad062786fd`:

- The prefetch deferral lowered model-load residency, but the dual-endpoint
  path still had only about `4.92 GiB` available for KV versus `5.98 GiB`
  required for the implicit full `262144` context.
- Added an implementation-side low-memory Case 2 max-model-length cap.
- When Case 2 is selected, `gpu_memory_utilization <= 0.5`, and the user did
  not explicitly pass `--max-model-len`, the engine caps `model_config.max_model_len`
  to `196608`.
- Explicit user `--max-model-len` values are preserved.
- Focused validation passed:

```bash
dev/moe/.venv/bin/python -m pytest tests/v1/engine/test_engine_args.py tests/kernels/moe/test_moe_cpu_offload.py -q
dev/moe/.venv/bin/python -m ruff check vllm/engine/moe_offload_cli.py tests/v1/engine/test_engine_args.py vllm/model_executor/layers/fused_moe/layer.py vllm/model_executor/layers/fused_moe/moe_offload.py tests/kernels/moe/test_moe_cpu_offload.py vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py vllm/config/moe.py
```

Remaining status: this should bring the implicit dual-endpoint Case 2 context
below the latest estimated fit (`206400`) without changing the Makefile. The
outer harness still needs to confirm port `8073` passes health and smoke.

Follow-up diagnostic session `1c841c8a-0d88-4e24-8d61-2aa4cca8aa56`:

- Failing target: `make test`, during `test-local-moe-double-run` on port
  `8073`
- Log: `/home/ubuntu/gpulab/vllm-moe/dev/moe/logs/vllm_8073_20260427T093301Z.log`
- Error:

```text
ValueError: To serve at least one request with the models's max seq len (262144), (5.98 GiB KV cache is needed, which is larger than the available KV cache memory (4.92 GiB). Based on the available memory, the estimated maximum model length is 206400.
```

The low-memory startup prefetch deferral changed model loading on port `8073`
from `9.29 GiB` to `5.96 GiB`, confirming it reduced startup GPU residency.
However, vLLM still reports only `4.92 GiB` available for KV cache with
`GPU_MEMORY_UTILIZATION=0.4`, which remains below the `5.98 GiB` requirement for
Gemma4's default `max_seq_len=262144`.

Updated likely cause: the remaining blocker is not just startup expert
prefetch. The dual-endpoint run's memory split and full model length are still
incompatible with the current non-KV residency. Either the dual-endpoint
runtime needs a lower `max_model_len`/different GPU memory split, or Case 2 must
reduce additional non-KV GPU allocations before vLLM performs KV sizing.

Next inspection area: vLLM startup memory profiling and max-model-length
sizing, `dev/moe/run_vllm.sh`/`dev/moe/Makefile` argument wiring if harness
sizing is intentionally changed, and Case 2 cache/allocation timing in
`vllm/model_executor/layers/fused_moe/layer.py` and
`vllm/model_executor/layers/fused_moe/moe_gpu_prefetch.py`.

## 2026-04-30: Case Implementation Modules Split

Status: aligned with current code.

- Case 1 passive CPU offload implementation is isolated in
  `vllm/model_executor/layers/fused_moe/moe_offload.py`.
- Case 2 GPU prefetch implementation is isolated in
  `vllm/model_executor/layers/fused_moe/moe_gpu_prefetch.py`.
- `layer.py` and engine config code select the active case but should not mix
  passive-wave behavior with pager/cache behavior.
- When investigating Case 2 bugs, inspect `moe_gpu_prefetch.py` first. When
  investigating Case 1 passive transfer bugs, inspect `moe_offload.py` first.
