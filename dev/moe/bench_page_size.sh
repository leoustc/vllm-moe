#!/usr/bin/env bash
set -euo pipefail

# System test with 8 GPUs
GPU_MEMORY_UTILIZATION=0.95 VLLM_BIN=/home/ubuntu/.venv/bin/vllm MODEL_PATH=/models/gemma-4-26B-A4B-it GPU="0,1,2,3,4,5,6,7" PORT=8090 VLLM_EXTRA_ARGS="--tensor-parallel-size 8" ./run_vllm.sh
REQUEST_RATE=16 MAX_CONCURRENCY=256 PORT=8090 ./bench_vllm.sh
./clean_vllm.sh
sleep 5

#Test use max GPU RAM size but different --moe-gpu-prefetch num

ACTIVE_MODEL_NUM=8
GPU_MEMORY_UTILIZATION=0.95 VLLM_BIN=.venv/bin/vllm MODEL_PATH=/models/gemma-4-26B-A4B-it GPU=0 PORT=8090 VLLM_EXTRA_ARGS="--moe-gpu-prefetch $ACTIVE_MODEL_NUM" ./run_vllm.sh
SWEEP=1 PORT=8090 ./bench_vllm.sh
./clean_vllm.sh
sleep 5

ACTIVE_MODEL_NUM=12
GPU_MEMORY_UTILIZATION=0.95 VLLM_BIN=.venv/bin/vllm MODEL_PATH=/models/gemma-4-26B-A4B-it GPU=0 PORT=8090 VLLM_EXTRA_ARGS="--moe-gpu-prefetch $ACTIVE_MODEL_NUM" ./run_vllm.sh
SWEEP=1 PORT=8090 ./bench_vllm.sh
./clean_vllm.sh
sleep 5

ACTIVE_MODEL_NUM=16
GPU_MEMORY_UTILIZATION=0.95 VLLM_BIN=.venv/bin/vllm MODEL_PATH=/models/gemma-4-26B-A4B-it GPU=0 PORT=8090 VLLM_EXTRA_ARGS="--moe-gpu-prefetch $ACTIVE_MODEL_NUM" ./run_vllm.sh
SWEEP=1 PORT=8090 ./bench_vllm.sh
./clean_vllm.sh
sleep 5

ACTIVE_MODEL_NUM=32
GPU_MEMORY_UTILIZATION=0.95 VLLM_BIN=.venv/bin/vllm MODEL_PATH=/models/gemma-4-26B-A4B-it GPU=0 PORT=8090 VLLM_EXTRA_ARGS="--moe-gpu-prefetch $ACTIVE_MODEL_NUM" ./run_vllm.sh
SWEEP=1 PORT=8090 ./bench_vllm.sh
./clean_vllm.sh
sleep 5

ACTIVE_MODEL_NUM=64
GPU_MEMORY_UTILIZATION=0.95 VLLM_BIN=.venv/bin/vllm MODEL_PATH=/models/gemma-4-26B-A4B-it GPU=0 PORT=8090 VLLM_EXTRA_ARGS="--moe-gpu-prefetch $ACTIVE_MODEL_NUM" ./run_vllm.sh
SWEEP=1 PORT=8090 ./bench_vllm.sh
./clean_vllm.sh
sleep 5
