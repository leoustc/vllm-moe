# FEATURE: Agent-Native Utilization-First MoE CPU-Offload Pipeline for vLLM

## Summary

Add a new **optional sparse-MoE CPU-offload execution mode** to vLLM for **agent-oriented workloads** where:

- per-token latency is less important than sustained throughput,
- requests generate many tokens over long horizons,
- GPU memory cannot hold the full expert set,
- expert movement cost is a first-order bottleneck.

This feature must **preserve all existing vLLM behavior unchanged by default**.

When enabled, the new mode will:

1. Keep the **traditional vLLM path unchanged** unless the new flag is set.
2. Offload **MoE expert weights to CPU memory**.
3. Keep only the following always resident on GPU:
   - router / gating model path,
   - currently active or hot experts,
   - optionally a small same-group fallback set.
4. Fetch only the **active expert delta** from CPU to GPU.
5. Increase token pipeline depth to make loaded active experts more reusable.
6. Optionally use an **active expert in the same group** as a fallback when the routed expert is not currently resident.
7. Maximize GPU utilization and tokens-per-loaded-expert instead of minimizing token latency.

This mode is designed for **sparse MoE**, especially memory-constrained single-GPU inference on hardware such as **A100 40GB**.

---

## Hard Requirement

### Existing vLLM behavior must remain unchanged by default

The feature must be fully gated behind new flags.

If the new flags are not set:
- scheduling must remain unchanged,
- weight placement must remain unchanged,
- MoE execution semantics must remain unchanged,
- no new latency tradeoffs should be introduced.

The new path should behave as an **opt-in execution mode**, not as a modification to baseline vLLM serving.

---

## Primary User-Facing Flag

Add a new flag:

```bash
--moe-cpu-offload
```

### Meaning

If `--moe-cpu-offload` is set:

- store MoE expert weights in CPU memory,
- keep the router path on GPU,
- move only active/hot experts to GPU,
- use a utilization-first token pipeline to increase active-expert reuse.

If `--moe-cpu-offload` is **not** set:
- use traditional vLLM behavior unchanged.

---

## Additional Recommended Flags

```bash
--moe-cpu-offload
--moe-gpu-limit 0.5
--moe-active-expert-budget 64
--moe-fallback-expert-budget 16
--moe-wave-min-tokens 2048
--moe-wave-max-wait-ms 200
--moe-transfer-safety-factor 1.5
--moe-group-fallback
--moe-group-fallback-threshold 0.8
--moe-resident-first
--moe-min-residency-steps 3
```

### Suggested meanings

- `--moe-gpu-limit`
  - Fraction of total GPU memory that the offload-aware runtime may consume.
  - Must account for resident expert weights, KV cache growth, token-wave / bucket buffers, transfer staging buffers, and MoE-specific temporary workspace.
  - If projected usage exceeds this cap, retire cold experts first, then shrink or delay the next wave.

- `--moe-active-expert-budget`
  - Number of expert slots reserved for currently active/hot experts on GPU.

- `--moe-fallback-expert-budget`
  - Number of expert slots reserved for same-group fallback / standby experts.

- `--moe-wave-min-tokens`
  - Minimum queued token count before launching a utilization wave.

- `--moe-wave-max-wait-ms`
  - Maximum queue delay before forcing a wave launch.

- `--moe-transfer-safety-factor`
  - Launch when estimated compute time can hide the next expert-delta transfer.

- `--moe-group-fallback`
  - Enable same-group active-expert fallback if the routed expert is not resident.

- `--moe-group-fallback-threshold`
  - Similarity / confidence threshold required before same-group fallback is allowed.

- `--moe-resident-first`
  - Execute resident expert buckets before newly loaded expert buckets.

- `--moe-min-residency-steps`
  - Minimum number of scheduling rounds to retain a loaded expert before eviction.

---

## High-Level Requested Behavior

### Traditional vLLM mode

If the new feature flags are not enabled:

- keep standard vLLM execution and memory placement unchanged.

### New MoE CPU-offload mode

If `--moe-cpu-offload` is enabled:

1. Keep the **router model path** on GPU.
2. Keep a bounded set of **active/hot experts** on GPU.
3. Store the rest of the expert weights in CPU memory.
4. Route tokens as usual using the model’s original router.
5. Build expert-token buckets after routing.
6. Enforce a memory-aware GPU cap that includes resident experts, KV cache, token waves, and staging/workspace buffers.
7. Grow the token pipeline / queue depth to make resident active experts more reusable without violating the GPU cap.
8. Fetch only the expert delta needed for upcoming waves.
9. If a routed expert is not on GPU:
   - if same-group fallback is enabled and valid, use a resident active expert from the same group,
   - otherwise fetch the routed expert from CPU to GPU.

This is explicitly a **utilization-first MoE mode**.

---

## Motivation

vLLM already provides the serving substrate, including:
- paged KV cache,
- scheduler / engine core,
- worker / model-runner abstractions,
- MoE support and fused MoE kernels.

This feature does **not** aim to replace those capabilities.

Instead, it targets a workload regime where:
- MoE models are sparse,
- total expert weights exceed GPU memory,
- agent workloads can tolerate additional latency,
- sustained throughput and GPU utilization matter more than token immediacy.

The optimization goal becomes:

> maximize tokens served per expert load and hide expert transfer behind larger token waves.

---

## Core Design Goals

1. **Keep stock vLLM unchanged by default**
2. **Offload expert weights to CPU only when explicitly requested**
3. **Keep router on GPU**
4. **Fetch only active expert delta**
5. **Increase token pipeline depth to improve active-expert reuse**
6. **Support optional same-group active-expert fallback**
7. **Preserve model correctness as much as possible**
8. **Make fallback opt-in and measurable**
9. **Add clear metrics for offload, reuse, and fallback frequency**

---

## Runtime Model

### GPU-resident components

Always keep on GPU:
- router / gating path,
- non-offloaded core execution path required by model runner,
- currently active/hot expert set,
- optional same-group fallback / standby expert set.

Under offload mode, projected GPU usage must also include:
- resident MoE expert weights,
- paged KV cache usage attributable to queued / active requests,
- token-bucket / wave buffers,
- H2D staging buffers for incoming expert deltas,
- temporary workspace required by MoE execution kernels.

### CPU-resident components

Store in CPU memory:
- offloaded MoE expert weights not currently selected as active/hot/fallback experts.

### Token pipeline

The runtime should intentionally increase the queue depth for MoE work so that:
- more tokens are compacted per active expert,
- active experts process larger token waves,
- the cost of loading an expert is amortized over more token work.

This is a utilization-first policy for agent workloads, but wave growth must remain bounded by the configured GPU memory cap.

### GPU memory limit

Add a global offload-aware limit such as:

```bash
--moe-gpu-limit 0.5
```

Meaning:
- only active when `--moe-cpu-offload` is enabled,
- cap total GPU memory used by the offload-aware runtime to a fraction of device memory,
- include expert residency, KV cache, token buckets / waves, staging buffers, and MoE-specific workspace in the accounting,
- retire cold experts when projected usage would exceed the cap,
- if eviction is insufficient, shrink or delay the next token wave instead of overcommitting GPU memory.

Suggested projected usage rule:

```text
projected_gpu_usage =
    base_model_gpu_bytes
  + kv_cache_bytes
  + resident_expert_bytes
  + token_wave_bytes
  + staging_and_workspace_bytes
```

Constraint:

```text
projected_gpu_usage <= moe_gpu_limit * total_gpu_bytes
```

---

## Active vs Fallback Experts

### Active experts

These are the experts currently expected to process the next MoE wave.

### Fallback experts

These are GPU-resident experts kept as backup representatives for expert groups.

They are used only if:
- routed expert is not resident,
- same-group fallback is enabled,
- fallback confidence / similarity satisfies threshold,
- fetching the exact expert is not preferred by policy.

---

## Same-Group Fallback Semantics

If the token after the router selects expert `e`, and `e` is not resident on GPU:

- find expert group `g = group(e)`,
- search for a resident active/fallback expert `e'` in the same group,
- if `sim(e, e') >= threshold` and `--moe-group-fallback` is enabled:
  - execute `e'` instead of waiting for `e`,
- otherwise:
  - fetch `e` from CPU to GPU.

### Important constraints

Same-group fallback must be:
- optional,
- explicitly enabled,
- measurable,
- easy to disable for correctness parity tests.

Phase 1 should include:
- counters for fallback use,
- per-request / per-wave fallback frequency,
- easy comparison against exact-expert execution.

---

## Expert Grouping

This feature requires an expert grouping interface.

Add a grouping abstraction such as:

```python
class ExpertGrouping:
    expert_to_group: dict[int, int]
    group_to_experts: dict[int, list[int]]
    similarity: dict[tuple[int, int], float]
```

Possible initial sources of grouping:
- offline config file,
- model-specific metadata file,
- static heuristic mapping,
- manually supplied grouping table.

Do not require dynamic online learning in Phase 1.

---

## Execution Flow

### Baseline path (unchanged)
If `--moe-cpu-offload` is not set:
- run standard vLLM behavior unchanged.

### New offload path
If `--moe-cpu-offload` is set:

1. Scheduler selects candidate requests.
2. Router executes on GPU as usual.
3. Build token buckets by expert.
4. Determine:
   - resident active experts,
   - missing routed experts,
   - same-group fallback candidates,
   - next expert delta to fetch,
   - projected KV / token-wave / staging memory for the next launch.
5. Evict cold experts if projected memory exceeds `--moe-gpu-limit`.
6. Grow / maintain a token wave large enough to improve expert reuse while staying within the GPU memory cap.
7. Launch resident-first execution.
8. Overlap active-delta fetch from CPU to GPU.
9. For missing experts:
   - prefer same-group active resident expert if fallback enabled and valid,
   - else fetch exact expert.
10. Update active/hot/fallback sets incrementally.

---

## Incremental Refresh

Do **not** use full-set swapping.

Use:

```text
retain = useful(current resident experts)
delta  = next needed experts not already retained
next_active = retain ∪ delta
```

This must apply separately to:
- active experts,
- fallback / standby experts.

Experts with good recent reuse should remain resident long enough to amortize load cost.

Cold-expert retirement should prefer:
- fallback experts before active experts,
- experts with low recent token reuse,
- experts outside the minimum residency window,
- experts not required by the in-flight wave.

---

## Token Pipeline Behavior

The feature must explicitly deepen the token pipeline under offload mode.

### Why
If active experts are expensive to load, the system should:
- queue more routed tokens,
- compact more tokens onto currently resident active experts,
- trade latency for reuse.

### Scheduler behavior
Under `--moe-cpu-offload`:
- prefer larger MoE waves,
- prioritize resident experts,
- hold requests slightly longer if that improves active-expert reuse,
- respect a maximum wait bound,
- shrink or delay waves when KV growth or token-bucket memory would violate `--moe-gpu-limit`.

---

## Launch Rule

Suggested wave launch condition:

```text
queued_tokens >= wave_min_tokens
AND
est_compute_ms >= alpha * est_transfer_ms
AND
projected_gpu_usage <= moe_gpu_limit * total_gpu_bytes
```

or force launch if:

```text
oldest_request_wait_ms >= wave_max_wait_ms
```

Where:
- `est_transfer_ms` is based on the next active expert delta,
- not the full non-resident expert set.

If `queued_tokens` and transfer-hiding conditions are satisfied but the projected memory cap is violated:
- first retire cold experts,
- then recompute projected usage,
- then shrink the wave if necessary,
- otherwise delay launch until memory becomes available.

---

## Data Structures

### Expert residency state

```python
class ExpertResidencyState:
    active_experts: set[int]
    fallback_experts: set[int]
    resident_experts: set[int]
    cpu_offloaded_experts: set[int]
    resident_expert_bytes: int
    expert_last_used_step: dict[int, int]
    expert_tokens_served: dict[int, int]
    expert_load_count: dict[int, int]
    expert_bytes_loaded: dict[int, int]
```

### Wave plan

```python
class MoEWavePlan:
    request_ids: list[str]
    token_count: int
    expert_buckets: dict[int, list[int]]
    active_experts: set[int]
    missing_experts: set[int]
    fallback_matches: dict[int, int]   # routed_expert -> fallback_expert
    prefetch_delta: list[int]
    est_compute_ms: float
    est_transfer_ms: float
    est_kv_bytes: int
    est_token_wave_bytes: int
    est_staging_bytes: int
    projected_gpu_usage_bytes: int
    launch_reason: str
```

---

## Integration Points in vLLM

This should be implemented as an extension to the existing vLLM architecture.

### A. Config / CLI
Add new flags and plumb them through engine config.

### B. Scheduler / engine core
Likely areas:
- `vllm/v1/engine/core.py`
- scheduler-related modules in `vllm/v1/engine/`

Add:
- utilization-first MoE wave mode,
- deeper queueing for offload mode,
- launch policy for MoE waves,
- memory-aware wave sizing and delay policy under `--moe-gpu-limit`.

### C. GPU worker / model runner
Likely areas:
- `vllm/v1/worker/gpu/model_runner.py`

Add:
- expert demand extraction,
- active/fallback planning,
- CPU-offload fetch orchestration,
- resident-first execution ordering,
- GPU memory accounting for expert residency, KV usage, token buckets, and staging/workspace buffers.

### D. MoE execution path
Likely areas:
- `vllm/model_executor/layers/fused_moe/...`

Add:
- metadata for active expert residency,
- optional same-group fallback mapping,
- hooks for exact-vs-fallback execution accounting.

### E. CPU offload / weight movement helper
Introduce a helper / manager for:
- CPU-resident expert weight storage,
- async H2D expert movement,
- expert-slot allocation on GPU,
- cold-expert retirement under a global GPU memory cap.

---

## Metrics

Add explicit metrics for the new mode.

### Offload metrics
- `moe_cpu_offload_enabled`
- `moe_gpu_limit_ratio`
- `moe_expert_delta_load_count`
- `moe_expert_delta_load_bytes`
- `moe_expert_load_latency_ms`

### Reuse metrics
- `moe_tokens_per_expert_load`
- `moe_bytes_loaded_per_token`
- `moe_expert_reuse_before_eviction`

### Wave metrics
- `moe_wave_count`
- `moe_wave_token_count`
- `moe_wave_queue_wait_ms`
- `moe_wave_compute_estimate_ms`
- `moe_wave_transfer_estimate_ms`
- `moe_wave_projected_gpu_bytes`
- `moe_wave_kv_bytes`
- `moe_wave_token_buffer_bytes`
- `moe_wave_launch_reason`

### Residency metrics
- `moe_active_expert_count`
- `moe_fallback_expert_count`
- `moe_resident_expert_bytes`
- `moe_expert_cache_hit_ratio`
- `moe_cold_expert_eviction_count`

### Fallback metrics
- `moe_group_fallback_count`
- `moe_group_fallback_ratio`
- `moe_group_fallback_exact_miss_count`
- `moe_group_fallback_threshold_reject_count`

These must make it easy to compare:
- exact-only offload mode,
- exact+fallback mode,
- stock vLLM mode.

---

## Correctness / Safety Requirements

### Default safety
- No change to stock vLLM unless flags are set.

### Exact mode parity
When:
- `--moe-cpu-offload` is enabled
- but `--moe-group-fallback` is disabled

Then outputs should remain numerically equivalent to exact routed-expert execution, modulo standard runtime nondeterminism.

### Fallback mode
When `--moe-group-fallback` is enabled:
- outputs may differ from exact execution,
- fallback frequency must be measurable,
- feature must be clearly marked as approximate / optimization mode.

---

## Rollout Plan

### Phase 1
- Add `--moe-cpu-offload`
- Add `--moe-gpu-limit`
- Keep router on GPU
- Offload expert weights to CPU
- Fetch only active expert delta
- Add cold-expert retirement under a memory-aware GPU cap
- Add deeper token wave scheduling
- Account for KV cache, token-wave buffers, and staging/workspace memory in launch planning
- Add metrics
- No fallback yet by default

### Phase 2
- Add `--moe-group-fallback`
- Add static expert grouping input
- Add fallback thresholding
- Add fallback metrics

### Phase 3
- Add learned group hotness prediction
- Add multi-GPU extension
- Add better overlap / scheduling heuristics

---

## Testing Plan

### Unit tests
- active expert delta computation
- expert retention / eviction
- same-group lookup
- thresholded fallback decision
- launch policy
- CPU-offload slot accounting

### Functional tests
- stock path unchanged when flags unset
- exact offload path correctness parity
- fallback path executes only when enabled
- no token loss / no invalid expert mapping

### Performance tests
Compare:
1. stock vLLM
2. `--moe-cpu-offload` exact mode
3. `--moe-cpu-offload --moe-group-fallback`

Measure:
- tokens/sec
- GPU utilization
- bytes loaded per token
- tokens per expert load
- queue wait
- fallback ratio

### Priority target
- single A100 40GB
- agent-style long generation
- sparse MoE models such as Mixtral first, Qwen2-MoE second

---

## Non-Goals

This feature does **not** aim to:
- modify router behavior in Phase 1,
- change default vLLM serving semantics,
- optimize dense models,
- replace paged KV,
- require multi-GPU support initially.

This is an **opt-in sparse-MoE CPU-offload mode** for latency-tolerant, utilization-oriented agent workloads.

---

## Acceptance Criteria

This feature is successful if:

1. Stock vLLM behavior is unchanged when flags are unset.
2. Under `--moe-cpu-offload`, expert weights can be stored in CPU memory and fetched on demand.
3. Router path remains resident on GPU.
4. Active-expert reuse increases via deeper token waves.
5. `tokens_per_expert_load` increases measurably.
6. `bytes_loaded_per_token` decreases measurably.
7. If group fallback is enabled, fallback usage is measurable and bounded.
8. Throughput improves over stock latency-first behavior in long-running agent workloads.

---

## Notes for Codex

Implement in this order:

1. Add config flags
2. Keep default path unchanged
3. Add CPU expert offload manager
4. Keep router on GPU
5. Add active expert residency manager
6. Add deeper token wave scheduling
7. Add resident-first execution
8. Add incremental refresh
9. Add metrics
10. Add optional same-group fallback only after exact offload mode works

Do not modify stock execution behavior unless `--moe-cpu-offload` is explicitly enabled.
