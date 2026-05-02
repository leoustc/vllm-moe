# Case 2 Benchmark Session

Case 2 benchmark sweep for `--moe-gpu-prefetch` with fixed benchmark load.

## Latest Random-Prompt Benchmark: 2026-05-02

This rerun used the harness at `/home/ubuntu/gpulab/harness-vll-moe` after
updating Case 2 effective prefetch bounds to clamp requested prefetch into the
model range:

```text
active_experts <= effective_gpu_prefetch <= total_experts
```

Command:

```bash
make compile
make test-case2 \
  TEST_CASE_GPU=4 \
  TEST_CASE2_GPU_MEMORY_UTILIZATION_LIST=0.95 \
  TEST_CASE2_PREFETCH_LIST='16 32 64 72'
```

Settings:

- vLLM source path: `/home/ubuntu/gpulab/vllm-moe`
- vLLM source commit: `44043bf58761ae25c5eb1171c68d387f12d16567`
- Harness vLLM package version:
  `0.1.dev16070+g05eed05a4.d20260502.precompiled`
- GPU: `4` (`NVIDIA A100-SXM4-40GB`)
- Model: `/models/gemma-4-26B-A4B-it`
- Port: `8062`
- GPU memory utilization: `0.95`
- Benchmark prompts: `16`
- Request rate: `8`
- Input length: `64`
- Output length: `64`
- Max concurrency: `16`
- Max model length: `4096`

### Latest Random Summary

| Timestamp UTC | GPU limit | Prefetch num | Output tok/s | Req/s | Mean TTFT ms | Mean TPOT ms | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| 20260502-124704 | 0.95 | 16 | 21.13 | 0.33 | 3945.07 | 688.88 | OK |
| 20260502-125137 | 0.95 | 32 | 33.86 | 0.53 | 2987.97 | 415.19 | OK |
| 20260502-125602 | 0.95 | 64 | 52.73 | 0.82 | 2323.98 | 252.98 | OK |
| 20260502-130024 | 0.95 | 72 | 58.98 | 0.92 | 1945.75 | 225.52 | OK |

The trend matches the earlier high-memory random-prompt run: larger prefetch
capacity improves throughput, and prefetch `72` remains the best tested random
point. The latest `72` result is slightly below the previous `60.31 tok/s` row,
but in the same performance band.

### Latest Random Details

| Prefetch | Completed | Failed | Duration s | Input tokens | Output tokens | Output tok/s | Total tok/s | Mean TTFT ms | Median TTFT ms | Mean TPOT ms | Median TPOT ms | Mean ITL ms | Median ITL ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 16 | 0 | 48.46 | 1236 | 1024 | 21.13 | 46.64 | 3945.07 | 4103.67 | 688.88 | 682.67 | 678.11 | 696.62 |
| 32 | 16 | 0 | 30.25 | 1236 | 1024 | 33.86 | 74.72 | 2987.97 | 3075.36 | 415.19 | 410.45 | 408.70 | 426.78 |
| 64 | 16 | 0 | 19.42 | 1236 | 1024 | 52.73 | 116.37 | 2323.98 | 2318.81 | 252.98 | 250.27 | 249.03 | 244.93 |
| 72 | 16 | 0 | 17.36 | 1236 | 1024 | 58.98 | 130.16 | 1945.75 | 1927.35 | 225.52 | 223.75 | 222.00 | 220.36 |

Artifacts:

```text
/home/ubuntu/gpulab/harness-vll-moe/logs/vllm_8062_20260502T124301Z.log
/home/ubuntu/gpulab/harness-vll-moe/logs/vllm_8062_20260502T124753Z.log
/home/ubuntu/gpulab/harness-vll-moe/logs/vllm_8062_20260502T125225Z.log
/home/ubuntu/gpulab/harness-vll-moe/logs/vllm_8062_20260502T125650Z.log
/home/ubuntu/gpulab/harness-vll-moe/logs/bench-results/openai-chat-8.0qps-concurrency16-gemma-4-26B-A4B-it-20260502-124704.json
/home/ubuntu/gpulab/harness-vll-moe/logs/bench-results/openai-chat-8.0qps-concurrency16-gemma-4-26B-A4B-it-20260502-125137.json
/home/ubuntu/gpulab/harness-vll-moe/logs/bench-results/openai-chat-8.0qps-concurrency16-gemma-4-26B-A4B-it-20260502-125602.json
/home/ubuntu/gpulab/harness-vll-moe/logs/bench-results/openai-chat-8.0qps-concurrency16-gemma-4-26B-A4B-it-20260502-130024.json
```

## Latest Locality Benchmark: 20260502T132318Z

Repeated-prompt locality sweep for the best high-memory random-prompt setting:
`GPU_MEMORY_UTILIZATION=0.95` and `--moe-gpu-prefetch 72`.

Command:

```bash
PORT=8062 \
GPU=4 \
GPU_MEMORY_UTILIZATION=0.95 \
MOE_GPU_PREFETCH=72 \
NUM_PROMPTS=100 \
REPEATS_LIST='16 32 64 128' \
REQUEST_RATE=8 \
MAX_CONCURRENCY=16 \
OUTPUT_LEN=64 \
KEEP_VLLM=0 \
./locality/bench_repeated_prompts.sh
```

Settings:

- vLLM source path: `/home/ubuntu/gpulab/vllm-moe`
- vLLM source commit: `44043bf58761ae25c5eb1171c68d387f12d16567`
- Model: `/models/gemma-4-26B-A4B-it`
- GPU: `4` (`NVIDIA A100-SXM4-40GB`)
- Port: `8062`
- GPU memory utilization: `0.95`
- Max model length: `4096`
- Case 2 flag: `--moe-gpu-prefetch 72`
- Requests per repeat block: `100`
- Repeat blocks: `16`, `32`, `64`, `128`
- Request rate: `8`
- Max concurrency: `16`
- Output length: `64`
- Temperature: `0`
- Warmups per repeat block: `1`

### Latest Locality Summary

| GPU util | Prefetch | Repeat block | Completed | Failed | Duration s | Req/s | Output tok/s | Total tok/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.95 | 72 | 16 | 100 | 0 | 84.10 | 1.19 | 76.10 | 148.48 |
| 0.95 | 72 | 32 | 100 | 0 | 78.41 | 1.28 | 81.62 | 159.32 |
| 0.95 | 72 | 64 | 100 | 0 | 80.10 | 1.25 | 79.90 | 157.66 |
| 0.95 | 72 | 128 | 100 | 0 | 77.85 | 1.28 | 82.21 | 163.14 |

Best aggregate row:

- Repeat block `128` had the best output throughput: `82.21 tok/s`.
- Repeat block `128` also had the lowest mean TTFT, lowest P99 TTFT, and best
  total token throughput in this run.
- Repeat block `32` was close on output throughput at `81.62 tok/s`.

### Latest Locality Details

| Repeat block | Input tokens | Output tokens | Mean TTFT ms | Median TTFT ms | P99 TTFT ms | Mean TPOT ms | Median TPOT ms | P99 TPOT ms | Mean ITL ms | Median ITL ms | P99 ITL ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 6088 | 6400 | 626.09 | 585.84 | 984.32 | 183.18 | 183.73 | 192.34 | 180.32 | 179.78 | 399.00 |
| 32 | 6092 | 6400 | 574.47 | 577.58 | 677.20 | 169.82 | 169.30 | 176.83 | 167.17 | 165.41 | 257.14 |
| 64 | 6228 | 6400 | 565.58 | 573.65 | 662.28 | 174.50 | 172.47 | 185.35 | 171.78 | 170.44 | 215.72 |
| 128 | 6300 | 6400 | 560.96 | 568.33 | 640.93 | 169.61 | 169.99 | 177.38 | 166.96 | 166.98 | 212.38 |

Compared with the latest random-prompt prefetch `72` row (`58.98 tok/s`), the
locality rows improved output throughput to `76.10-82.21 tok/s`. This confirms
that repeated prompts with adjacent request ordering reduce effective Case 2
expert-cache pressure for this workload.

Artifacts:

```text
/home/ubuntu/gpulab/harness-vll-moe/logs/vllm_8062_20260502T132318Z.log
/home/ubuntu/gpulab/harness-vll-moe/locality/logs/repeated_prompts_8062_r16_20260502T132318Z.log
/home/ubuntu/gpulab/harness-vll-moe/locality/logs/repeated_prompts_8062_r32_20260502T132318Z.log
/home/ubuntu/gpulab/harness-vll-moe/locality/logs/repeated_prompts_8062_r64_20260502T132318Z.log
/home/ubuntu/gpulab/harness-vll-moe/locality/logs/repeated_prompts_8062_r128_20260502T132318Z.log
/home/ubuntu/gpulab/harness-vll-moe/locality/logs/bench-results/locality_repeats_16_20260502T132318Z.json
/home/ubuntu/gpulab/harness-vll-moe/locality/logs/bench-results/locality_repeats_32_20260502T132318Z.json
/home/ubuntu/gpulab/harness-vll-moe/locality/logs/bench-results/locality_repeats_64_20260502T132318Z.json
/home/ubuntu/gpulab/harness-vll-moe/locality/logs/bench-results/locality_repeats_128_20260502T132318Z.json
```

Pager-filtered locality logs were created but are empty for this run because the
endpoint used the default `VLLM_LOGGING_LEVEL=WARNING`; no matching Case 2 pager
debug lines were emitted during the benchmark.

## Historical Random-Prompt Benchmark

Common settings:

- GPU: `4` (`NVIDIA A100-SXM4-40GB`)
- Model: `/models/gemma-4-26B-A4B-it`
- Port: `8062`
- Benchmark prompts: `16`
- Request rate: `8`
- Input length: `64`
- Output length: `64`
- Max concurrency: `16`
- Max model length: `4096`

## Summary

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

## Bench Details

### GPU limit 0.50, prefetch 16

- Status: OK
- Successful requests: `16`
- Benchmark duration: `50.11 s`
- Total input tokens: `1236`
- Total generated tokens: `1024`
- Request throughput: `0.32 req/s`
- Output token throughput: `20.43 tok/s`
- Mean TTFT: `3940.82 ms`
- Mean TPOT: `715.02 ms`
- Mean ITL: `703.85 ms`
- Log: `logs/test_case2_grid_20260430T091110Z.log`

### GPU limit 0.50, prefetch 32

- Status: NA
- Failure: KV cache startup failure at `max_model_len=4096`
- Required KV cache: `0.86 GiB`
- Available KV cache memory: `0.27 GiB`
- Estimated maximum model length: `1264`
- Log: `logs/test_case2_grid_20260430T091110Z.log`

### GPU limit 0.50, prefetch 64

- Status: NA
- Failure: KV cache startup failure at `max_model_len=4096`
- Log: `logs/test_case2_grid_20260430T091110Z.log`

### GPU limit 0.75, prefetch 16

- Status: OK
- Successful requests: `16`
- Benchmark duration: `49.61 s`
- Total input tokens: `1236`
- Total generated tokens: `1024`
- Request throughput: `0.32 req/s`
- Output token throughput: `20.64 tok/s`
- Mean TTFT: `4007.75 ms`
- Mean TPOT: `705.95 ms`
- Mean ITL: `694.92 ms`
- Log: `logs/test_case2_grid_20260430T091110Z.log`

### GPU limit 0.75, prefetch 32

- Status: OK
- Successful requests: `16`
- Benchmark duration: `29.44 s`
- Total input tokens: `1236`
- Total generated tokens: `1024`
- Request throughput: `0.54 req/s`
- Output token throughput: `34.78 tok/s`
- Mean TTFT: `2878.43 ms`
- Mean TPOT: `404.15 ms`
- Mean ITL: `397.83 ms`
- Log: `logs/test_case2_grid_20260430T091110Z.log`

### GPU limit 0.75, prefetch 64

- Status: NA
- Failure: KV cache startup failure at `max_model_len=4096`
- Log: `logs/test_case2_grid_20260430T091110Z.log`

### GPU limit 0.95, prefetch 16

- Status: OK
- Successful requests: `16`
- Benchmark duration: `47.18 s`
- Total input tokens: `1236`
- Total generated tokens: `1024`
- Request throughput: `0.34 req/s`
- Output token throughput: `21.71 tok/s`
- Mean TTFT: `4037.93 ms`
- Mean TPOT: `667.05 ms`
- Mean ITL: `656.63 ms`
- Log: `logs/test_case2_gpu095_20260430T213450_SGT.log`

### GPU limit 0.95, prefetch 32

- Status: OK
- Successful requests: `16`
- Benchmark duration: `30.24 s`
- Total input tokens: `1236`
- Total generated tokens: `1024`
- Request throughput: `0.53 req/s`
- Output token throughput: `33.86 tok/s`
- Mean TTFT: `2934.17 ms`
- Mean TPOT: `415.90 ms`
- Mean ITL: `409.40 ms`
- Log: `logs/test_case2_gpu095_20260430T213450_SGT.log`

### GPU limit 0.95, prefetch 64

- Status: OK
- Successful requests: `16`
- Benchmark duration: `18.12 s`
- Total input tokens: `1236`
- Total generated tokens: `1024`
- Request throughput: `0.88 req/s`
- Output token throughput: `56.51 tok/s`
- Mean TTFT: `2070.85 ms`
- Mean TPOT: `236.61 ms`
- Mean ITL: `232.92 ms`
- Log: `logs/test_case2_gpu095_20260430T213450_SGT.log`

### GPU limit 0.95, prefetch 72

- Status: OK
- Successful requests: `16`
- Benchmark duration: `16.98 s`
- Total input tokens: `1236`
- Total generated tokens: `1024`
- Request throughput: `0.94 req/s`
- Output token throughput: `60.31 tok/s`
- Mean TTFT: `1914.17 ms`
- Mean TPOT: `219.67 ms`
- Mean ITL: `216.24 ms`
- Log: `logs/test_case2_gpu095_prefetch72_96_20260430T220459_SGT.log`

### GPU limit 0.95, prefetch 96

- Status: NA
- Failure: active expert cache GPU memory startup failure
- Failed layer: `29`
- Required active expert cache memory: `1141899264 bytes`
- Free GPU memory after retries: `459800576 bytes`
- Log: `logs/test_case2_gpu095_prefetch72_96_20260430T220459_SGT.log`
