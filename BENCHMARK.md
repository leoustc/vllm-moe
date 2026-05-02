# Case 2 Benchmark Session

Case 2 benchmark sweep for `--moe-gpu-prefetch` with fixed benchmark load.

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
