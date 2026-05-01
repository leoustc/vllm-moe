# Locality Bench

This folder contains focused Case 2 locality benchmarks. Unlike `bench_vllm.sh`,
it uses a custom vLLM benchmark dataset with repeated prompts so the active
expert set should have higher temporal locality.

## Usage

Start vLLM first, then run:

```bash
PORT=8062 ./locality/bench_repeated_prompts.sh
```

Defaults:

- `PROMPT_GROUPS=100`
- `REPEATS_PER_PROMPT=32`
- `NUM_PROMPTS=3200`
- `OUTPUT_LEN=64`
- `REQUEST_RATE=8`
- `MAX_CONCURRENCY=16`
- `TEMPERATURE=0`
- `SERVED_MODEL_NAME=default`

Useful overrides:

```bash
PORT=8062 \
PROMPT_GROUPS=100 \
REPEATS_PER_PROMPT=32 \
REQUEST_RATE=8 \
MAX_CONCURRENCY=16 \
./locality/bench_repeated_prompts.sh
```

The script generates a custom JSONL dataset at:

```text
locality/data/repeated_prompts_100x32.jsonl
```

Each dataset row has:

- `prompt`: prompt text
- `output_tokens`: requested output length
- `group_id`: prompt group id from `0` to `99`
- `repeat_id`: repeat id from `0` to `31`

The vLLM benchmark is run with `--dataset-name custom`,
`--dataset-path <generated-jsonl>`, and `--disable-shuffle`, so the 32 repeated
requests for each prompt stay adjacent in request order.

Outputs are written under `locality/logs/`:

- `repeated_prompts_<port>_<timestamp>.log`: full `vllm bench serve` output
- `repeated_prompts_<port>_<timestamp>.pager.log`: MoE pager lines captured
  during the run when the vLLM state file exposes `LOG_FILE`
- `bench-results/`: vLLM benchmark result JSON files

This keeps the benchmark path compatible with normal vLLM throughput metrics:
request throughput, output token throughput, TTFT, TPOT, ITL, and detailed
per-request result files.
