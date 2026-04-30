# REVIEW: Case 1 Passive MoE CPU Offload

This review covers Case 1: `--moe-cpu-offload`.

Current module boundary:

- Case 1 passive CPU offload is implemented in
  `vllm/model_executor/layers/fused_moe/moe_offload.py`.
- Case 2 GPU prefetch is implemented separately in
  `vllm/model_executor/layers/fused_moe/moe_gpu_prefetch.py`.
- Do not add Case 2 active/working/missing pager state to the Case 1 module.

## Findings

::code-comment{title="[P0][fixed] Passive path allocates all experts on first forward" body="Fixed by removing the early `move_moe_offload_cache_to_device()` call before routing analysis. Passive mode now computes `token_counts`, builds route-derived expert batches, then allocates the GPU staging cache per wave." file="/home/ubuntu/gpulab/vllm-moe/vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py" start=339 end=421 priority=0 confidence=0.92}

::code-comment{title="[P1][fixed] Failed expert load can leave GPU cache allocated" body="Fixed by moving `move_moe_offload_cache_to_device()` and `ensure_experts_resident()` inside the per-wave `try/finally`, so allocation/load/compute failures all retire requested experts and release GPU targets." file="/home/ubuntu/gpulab/vllm-moe/vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py" start=393 end=421 priority=1 confidence=0.83}

::code-comment{title="[P1][fixed] Split waves can be wrong when router weights are applied on input" body="Fixed conservatively by rejecting passive split-wave execution when `apply_router_weight_on_input=True`. Single-wave execution can still proceed, while the unsafe masked-output split path is blocked." file="/home/ubuntu/gpulab/vllm-moe/vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py" start=389 end=395 priority=1 confidence=0.74}

::code-comment{title="[P1][fixed] Unsupported backend coverage is not gated early enough" body="Fixed by adding an early FusedMoE initialization gate. MoE CPU offload now rejects non-unquantized fused-MoE methods, expert parallelism, and overlapped shared experts before weight creation/loading." file="/home/ubuntu/gpulab/vllm-moe/vllm/model_executor/layers/fused_moe/layer.py" start=540 end=559 priority=1 confidence=0.86}

::code-comment{title="[P2][fixed] Empty routed expert counts can fall through to normal kernel" body="Fixed by explicitly returning `torch.zeros_like(x)` when offload routing produces no local expert token counts, avoiding fallback to the normal kernel with offloaded/cache-backed parameters." file="/home/ubuntu/gpulab/vllm-moe/vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py" start=350 end=356 priority=2 confidence=0.61}

## Implementation Improvements

- Separate Case 1 and Case 2 wording in errors and comments.
- Reuse passive staging buffers when possible instead of freeing and reallocating GPU targets for every wave. Keep passive semantics, but avoid allocator churn and repeated `torch.cuda.empty_cache()`.
- Improve memory accounting by reserving headroom for allocator fragmentation and temporary kernel workspace, not only expert tensor bytes.
- Add metrics for expert transfer bytes, transfer latency, wave count, retry count, and per-layer OOM wait count.
- Add a startup log that states the supported runtime path, for example `backend=unquantized`, `expert_parallel=false`, and `shared_experts_overlap=false`.

## Test Gaps

- Add a small end-to-end correctness test that compares Case 1 output with normal all-GPU execution for a tiny MoE model.
- Add more direct forward-path regression coverage for the P0 early allocation bug. Current cache-level coverage verifies route-derived sizing before allocation.
- Add direct tests for failed allocation/load cleanup so GPU targets and layer parameters are restored after exceptions.
- Add direct tests for `apply_router_weight_on_input=True`, biased experts, and split-wave rejection.
- Add direct startup rejection tests for unsupported quantized MoE, expert parallel, and internally overlapped shared experts.
- Add a no-local-expert or empty-token-count test if that route is possible with EP or future routing changes.

## Summary

The highest-risk Case 1 findings from this review are fixed in code. Expert
weights are CPU-backed, routed expert waves are sized before GPU allocation,
loaded to GPU, computed, then retired. The next hardening step is to add more
direct forward-path tests and production metrics around transfer latency, wave
count, and memory retry behavior.
