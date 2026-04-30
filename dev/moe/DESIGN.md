# Instruction

Each case is independent. Do not mix and match behavior between cases. Each case
has its own implementation module:

- Case 1 passive CPU offload: `vllm/model_executor/layers/fused_moe/moe_offload.py`
- Case 2 GPU prefetch: `vllm/model_executor/layers/fused_moe/moe_gpu_prefetch.py`

Shared selection/wiring may live in the layer and engine config code, but Case 1
runtime behavior must not be implemented in the Case 2 module and Case 2 pager
behavior must not be implemented in the Case 1 module.

Flag precedence:

1. Dense model: ignore all MoE offload flags and use normal vLLM workflow.
2. MoE model with `--moe-gpu-prefetch <num>`: use Case 2.
3. MoE model with `--moe-cpu-offload`: use Case 1.
4. MoE model with no MoE offload flag: use normal vLLM workflow.

If both `--moe-gpu-prefetch <num>` and `--moe-cpu-offload` are set, Case 2 wins
and `--moe-cpu-offload` is ignored.

## Case 1: CPU offload and passive active expert model transfer

Purpose: use a small GPU to compute a large MoE model and keep the GPU OOM safe.

Implementation file:

```text
vllm/model_executor/layers/fused_moe/moe_offload.py
```

Flag:

```bash
--moe-cpu-offload
```

Flow:

- If the model is dense, ignore this flag completely and use normal vLLM.
- If the model is MoE, offload all weights to CPU.
- Only the router model, KV cache, and essential runtime structures stay on GPU.
- After the router computes the active expert set for a layer, CPU builds a fresh
  wave-local map from `global_expert_id` to compact `gpu_slot_id`.
- CPU passively transfers the needed active expert model weights from CPU to the
  compact GPU slots and updates that map.
- Before fused expert compute, remap routed `topk_ids` from global expert ids to
  compact GPU slot ids.
- Run fused expert compute against the staged GPU slot count with no persistent
  expert map.
- After active expert computation is done, retire/free the active expert model
  GPU memory and clear the wave-local map.
- Case 1 does not maintain persistent active, working, missing, or hot expert
  lists. Those page-cache lists belong to Case 2.

Hints:

1. After the router computes the active expert set, CPU transfers the needed
   active expert model weights to compact GPU slots.
2. Before CPU transfers active expert weights, check free GPU memory.
3. If GPU memory is not enough, wait 5 seconds and retry up to 10 times.
4. With this flag, CPU contains all weights to minimize GPU memory usage and let
   large MoE models work on small GPUs.
5. If the active expert set cannot fit at once, split it into smaller waves and
   build a new `global_expert_id -> gpu_slot_id` map for each wave.

Log Example:
(EngineCore pid=1147671) DEBUG 04-28 08:49:52 [MoE CPU Offload] Passive transfer layer=11 active_experts=[16, 27, 59, 66, 68, 98, 111, 117] step=144

## Case 2: CPU offload with GPU active model prefetch

Purpose: use a dedicated pager thread to manage the active expert model list in
GPU memory, like page cache management in an operating system.

Implementation file:

```text
vllm/model_executor/layers/fused_moe/moe_gpu_prefetch.py
```

Flag:

```bash
--moe-gpu-prefetch <num>
```

Rules:

- If the model is dense, ignore this flag and ignore `--moe-cpu-offload`; use
  normal vLLM workflow.
- If `--moe-gpu-prefetch <num>` is set for a MoE model, use Case 2 even if
  `--moe-cpu-offload` is also set.
- `--moe-gpu-prefetch <num>` is the target number of active expert models to
  keep resident in GPU memory.
- If `<num>` is less than the model config active expert count, set the effective
  prefetch size to `ceil(active_expert_count * 1.5)`. This reserves 50% more
  expert models in GPU than the model active expert count.

Main thread shared data and startup steps:

- Allocate CPU pinned memory for full weights.
- Allocate/read model config in CPU memory.
- Load all weights into CPU memory.
- Build active model list, initially empty.
- Build working model list, initially empty.
- Build missing model list, initially empty.
- Transfer router model to GPU.
- Build KV cache and essential GPU runtime structures.
- Transfer the effective `--moe-gpu-prefetch` number of active expert models to
  GPU and update the active model list.
- Sleep for 5 seconds.
- Detach the pager thread.
- Start web API serving.

Main thread request loop:

- Send request to router model.
- Router model builds expert model buckets and aggregates token buckets in GPU.
- Sync token bucket metadata back to CPU.
- Update the working model list.
- CPU matches the working model list against the active model list.
- Issue GPU compute for buckets whose expert models are in the active model
  list.
- Remove executed models from the working model list.
- Insert non-executed models into the missing model list.
- Case 2 owns GPU-side routed metadata, resident/missing wave packaging, missing
  expert token pressure, pager H2D copy streams, and active/working/missing list
  state.

Pager thread loop:

1. Check the missing model list.
2. If missing model pressure is high, check the GPU active model list and the
   effective `--moe-gpu-prefetch` capacity.
3. If there is room in the active model list, prefetch a missing model into GPU
   memory and update the active model list.
4. Retire active models that are not in the working model list.



## Optimization to GPU
