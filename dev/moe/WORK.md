# WORK: MoE CPU Offload Cases

Follow `DESIGN.md`. Cases are independent and must not be mixed.

Implementation ownership:

- Case 1 passive CPU offload code lives in
  `vllm/model_executor/layers/fused_moe/moe_offload.py`.
- Case 2 GPU prefetch code lives in
  `vllm/model_executor/layers/fused_moe/moe_gpu_prefetch.py`.
- Shared layer/config code may dispatch to the correct module, but do not add
  Case 2 pager state to Case 1 or Case 1 passive-wave behavior to Case 2.

Do not alter `Makefile` unless the user explicitly asks for Makefile changes.
Use it only as the canonical test harness.

Codex coding agents should only modify or fix implementation code. The sanity
and harness validation run outside the Codex agent routine through `make test`.

## Case Selection

1. Dense model: ignore all MoE offload flags and use normal vLLM workflow.
2. MoE model with `--moe-gpu-prefetch <num>`: use Case 2.
3. MoE model with `--moe-cpu-offload`: use Case 1.
4. MoE model with no MoE offload flag: use normal vLLM workflow.

If both flags are present, Case 2 wins and `--moe-cpu-offload` is ignored.

## Case 1: Done/Frozen

Case 1 flag:

```bash
--moe-cpu-offload
```

Completed checklist:

- [x] CLI help exposes `--moe-cpu-offload`.
- [x] Dense model startup logs say the flag was ignored.
- [x] MoE model startup logs say offload is enabled and transfer is `passive`.
- [x] Expert source weights remain CPU-backed.
- [x] Router path and KV cache remain GPU-backed.
- [x] Active experts are copied to GPU only after routing.
- [x] Case 1 builds a fresh per-layer/per-wave `global_expert_id -> gpu_slot_id`
  map.
- [x] Routed `topk_ids` are remapped to compact GPU slot ids before fused expert
  compute.
- [x] Fused expert compute uses the staged slot count and no persistent expert
  map.
- [x] GPU expert copies are freed after computation.
- [x] Active experts can execute in smaller passive waves.
- [x] Focused unit tests pass.
- [x] Full local harness passed for dense sanity, single MoE endpoint, and dual
  MoE endpoints.
- [x] Chat-based smoke tests produce coherent prompt/answer output.

Frozen scope:

- [x] Do not change Case 1 while Case 2 is the active focus unless the user
  explicitly asks for a Case 1 fix.
- [x] Audit stale stage wording in Case 1 code paths and logs.
- [x] Document that Case 1 does not use Case 2 active, working, missing, or hot
  pager lists.
- [x] Keep Case 1 in its own module: `moe_offload.py`.

## Case 2: Active Focus

Case 2 flag:

```bash
--moe-gpu-prefetch <num>
```

Required behavior:

1. Add public `--moe-gpu-prefetch <num>` CLI wiring.
2. If model is dense, ignore `--moe-gpu-prefetch` and `--moe-cpu-offload`.
3. If model is MoE and `--moe-gpu-prefetch <num>` is set, select Case 2 even
   when `--moe-cpu-offload` is also present.
4. Keep full weights in CPU pinned memory.
5. Keep router path, KV cache, and essential runtime structures on GPU.
6. Compute model active expert count from config.
7. If requested `<num>` is smaller than active expert count, set effective
   prefetch size to `ceil(active_expert_count * 1.5)`.
8. Build active model list, working model list, and missing model list.
9. Startup-prefetch the effective number of active expert models into GPU and
   update the active model list.
10. Start and detach a pager thread before API serving.
11. Main thread routes requests, builds GPU expert buckets, syncs bucket metadata
    to CPU, and updates the working model list.
12. Main thread executes buckets whose experts are in the active model list.
13. Main thread inserts non-executed experts into the missing model list.
14. Pager thread watches missing model list pressure.
15. Pager thread prefetches missing expert models into GPU when capacity exists.
16. Pager thread retires active models that are not in the working model list.
17. Case 2 implementation stays in `moe_gpu_prefetch.py`; do not add this pager
    behavior to `moe_offload.py`.

Done when:

- [x] CLI help exposes `--moe-gpu-prefetch <num>`.
- [x] Dense model ignores both MoE offload flags.
- [x] MoE model with `--moe-gpu-prefetch` selects Case 2.
- [x] MoE model with both flags selects Case 2.
- [x] Effective prefetch size applies the `ceil(active_expert_count * 1.5)`
  minimum when requested size is too small.
- [x] Startup logs report Case 2 and effective prefetch size.
- [x] Active, working, and missing model lists are maintained.
- [x] Pager thread prefetches missing experts and can evict cold active experts
  when capacity is needed.
- [x] Focused tests cover precedence, sizing, and list transitions.
- [x] Keep Case 2 in its own module: `moe_gpu_prefetch.py`.
- [ ] Single and dual endpoint Case 2 harness runs pass with chat smoke tests.
