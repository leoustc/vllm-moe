# FEATURE: MoE CPU Offload Cases

This folder tracks the independent MoE offload cases in `DESIGN.md`.

Implementation ownership:

- Case 1 is implemented only in `vllm/model_executor/layers/fused_moe/moe_offload.py`.
- Case 2 is implemented only in `vllm/model_executor/layers/fused_moe/moe_gpu_prefetch.py`.
- `layer.py` and engine config code may select between cases, but should not mix
  Case 1 passive transfer behavior with Case 2 pager behavior.

## Flag Precedence

1. Dense model: ignore all MoE offload flags and use normal vLLM workflow.
2. MoE model with `--moe-gpu-prefetch <num>`: use Case 2.
3. MoE model with `--moe-cpu-offload`: use Case 1.
4. MoE model with no MoE offload flag: use normal vLLM workflow.

If both `--moe-gpu-prefetch <num>` and `--moe-cpu-offload` are set, Case 2 wins
and `--moe-cpu-offload` is ignored.

## Case 1: Passive CPU Offload

Status: done/frozen for the tested unquantized Gemma MoE path. Do not change
Case 1 while focusing on Case 2 unless the user explicitly asks for a Case 1
fix.

Module:

```text
vllm/model_executor/layers/fused_moe/moe_offload.py
```

Flag:

```bash
--moe-cpu-offload
```

Behavior:

- Dense models ignore `--moe-cpu-offload` completely and run normally.
- MoE models keep model weights in CPU memory as the source of truth.
- Router weights, KV cache, and essential runtime structures remain on GPU.
- Expert weights are not prefetched.
- After router computation identifies active experts, the runtime passively
  transfers the needed expert weights from CPU to GPU.
- For each layer/wave, CPU builds a temporary `global_expert_id -> gpu_slot_id`
  map, copies active experts into compact GPU slots, and remaps routed `topk_ids`
  to those slot ids before fused expert compute.
- The fused expert kernel runs over the staged slot count with no persistent
  expert map.
- Before each GPU expert transfer, the runtime checks free GPU memory. If memory
  is insufficient, it waits 5 seconds and retries up to 10 times.
- After active expert computation, the runtime retires/frees the GPU expert
  copy and clears the temporary map.
- If all active experts cannot fit at once, the runtime computes smaller passive
  waves to stay GPU-OOM safe.
- Case 1 does not maintain Case 2 active, working, missing, or hot expert lists.

Latest validation:

- Dense sanity path passed with `--moe-cpu-offload` ignored.
- MoE single-endpoint path passed with passive CPU offload enabled.
- MoE dual-endpoint path passed with two `gemma-4-26B-A4B-it` endpoints sharing
  one 40GB GPU.

## Case 2: CPU Offload With GPU Active Model Prefetch

Status: active focus.

Module:

```text
vllm/model_executor/layers/fused_moe/moe_gpu_prefetch.py
```

Flag:

```bash
--moe-gpu-prefetch <num>
```

Behavior:

- Dense models ignore `--moe-gpu-prefetch` and `--moe-cpu-offload`.
- MoE models with `--moe-gpu-prefetch <num>` use Case 2, even if
  `--moe-cpu-offload` is also set.
- Full weights remain CPU-backed.
- Router path, KV cache, and essential runtime structures stay on GPU.
- Runtime maintains:
  - active model list: expert models resident on GPU,
  - working model list: expert models needed by current routed work,
  - missing model list: expert models needed but not resident.
- A dedicated pager thread manages GPU active expert residency like a page
  cache.
- GPU-side routed metadata is used to build expert token counts; CPU only sees
  compact expert/count metadata needed for pager decisions.
- Request execution packages resident waves first, records missing waves once,
  and lets the fused MoE path consume the GPU-side bucket layout.
- Startup prefetches the effective `--moe-gpu-prefetch` number of active expert
  models to GPU.
- Request execution uses resident active experts immediately and places missing
  experts into the missing model list.
- Pager thread prefetches missing experts when there is capacity and retires
  active experts that are not in the working model list.

Prefetch sizing:

- `<num>` is the requested GPU active expert residency target.
- If `<num>` is smaller than the model config active expert count, use
  `ceil(active_expert_count * 1.5)` as the effective prefetch size.

Current focus:

- Harden active, working, and missing expert list ownership.
- Keep pager-thread prefetch and retirement behavior isolated from Case 1
  passive transfer.
- Add focused tests for list transitions, sizing, and runtime residency.
- Validate Case 2 through single and dual endpoint harness runs.

## Observability

Startup logs should show:

- whether MoE offload flags are ignored or enabled,
- which case is selected,
- total experts when known,
- active experts when known,
- Case 1 transfer method: `passive`,
- Case 2 effective prefetch size.
