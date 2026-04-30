
REPO_DIR := $(abspath ../..)
VENV := .venv
PYTHON := $(VENV)/bin/python
VLLM := $(VENV)/bin/vllm
SYSTEM_VLLM := /home/ubuntu/.venv/bin/vllm

MODEL_PATH ?= /models/gemma-4-26B-A4B-it
BASE_MODEL_PATH ?= /models/gemma-4-E4B-it

GPU ?= 1
PORT ?= 8071
TEST_CASE_GPU ?= 0
TEST_CASE1_PORT ?= 8061
TEST_CASE2_PORT ?= 8062
TEST_CASE_BENCH_NUM_PROMPTS ?= 8
TEST_CASE_BENCH_REQUEST_RATE ?= 2
TEST_CASE_BENCH_INPUT_LEN ?= 64
TEST_CASE_BENCH_OUTPUT_LEN ?= 64
TEST_CASE_BENCH_CONCURRENCY ?= 8
TEST_CASE2_PREFETCH ?= 64
TEST_CASE2_PREFETCH_LIST ?= 16 32 64
TEST_CASE2_GPU_MEMORY_UTILIZATION_LIST ?= 0.95
TEST_CASE2_BENCH_NUM_PROMPTS ?= 16
TEST_CASE2_BENCH_REQUEST_RATE ?= 8
TEST_CASE2_BENCH_CONCURRENCY ?= 16
SANITY_PORT ?= 8070
LOCAL_BASE_PORT ?= 8071
LOCAL_MOE_PORT ?= 8072
LOCAL_MOE_DOUBLE_PORT_0 ?= 8073
LOCAL_MOE_DOUBLE_PORT_1 ?= 8074
#VLLM_EXTRA_ARGS ?= --moe-cpu-offload
VLLM_EXTRA_ARGS ?= --moe-gpu-prefetch 2
LOOP_TEST_TARGET ?= test
LOOP_SLEEP_SECONDS ?= 0
LOOP_MAX ?= 20

.PHONY: \
	base compile \
	sanity sanity-run sanity-test \
	test-local-base test-local-base-run test-local-base-test \
	test-local-moe test-local-moe-run test-local-moe-test \
	test-local-moe-double test-local-moe-double-run test-local-moe-double-test \
	clean-vllm test coding loop

# 0001 environment setup
base:
	uv venv --python 3.12 --allow-existing $(VENV)
	uv pip install --python $(PYTHON) -U pip

# 0002 local build install
compile: base
	VLLM_USE_PRECOMPILED=1 uv pip install --python $(PYTHON) $(REPO_DIR) --torch-backend=auto

# 0003 system vLLM baseline sanity
# Sanity test uses the offload flag on a dense model; it should be ignored.
sanity:
	$(MAKE) sanity-run
	$(MAKE) sanity-test

sanity-run:
	VLLM_BIN=$(SYSTEM_VLLM) MODEL_PATH=$(BASE_MODEL_PATH) \
		PORT=$(SANITY_PORT) VLLM_EXTRA_ARGS="$(VLLM_EXTRA_ARGS)" ./run_vllm.sh

sanity-test:
	PORT=$(SANITY_PORT) ./test_vllm.sh

# 0004 local build dense baseline with ignored MoE offload flag
test-local-base:
	$(MAKE) test-local-base-run
	$(MAKE) test-local-base-test

test-local-base-run:
	VLLM_BIN=$(VLLM) MODEL_PATH=$(BASE_MODEL_PATH) GPU=1 PORT=$(LOCAL_BASE_PORT) \
		VLLM_EXTRA_ARGS="$(VLLM_EXTRA_ARGS)" GPU_MEMORY_UTILIZATION=0.9 ./run_vllm.sh

test-local-base-test:
	PORT=$(LOCAL_BASE_PORT) ./test_vllm.sh

# 0005 local build MoE CPU offload feature test
test-local-moe:
	$(MAKE) test-local-moe-run
	$(MAKE) test-local-moe-test

test-local-moe-run:
	VLLM_BIN=$(VLLM) MODEL_PATH=$(MODEL_PATH) GPU=2 PORT=$(LOCAL_MOE_PORT) \
		VLLM_EXTRA_ARGS="$(VLLM_EXTRA_ARGS)" ./run_vllm.sh

test-local-moe-test:
	PORT=$(LOCAL_MOE_PORT) ./test_vllm.sh

# 0006 local build MoE CPU offload dual-endpoint test
test-local-moe-double:
	$(MAKE) test-local-moe-double-run
	$(MAKE) test-local-moe-double-test

test-local-moe-double-run:
	VLLM_BIN=$(VLLM) MODEL_PATH=$(MODEL_PATH) GPU=3 PORT=$(LOCAL_MOE_DOUBLE_PORT_0) \
		VLLM_EXTRA_ARGS="$(VLLM_EXTRA_ARGS)" ./run_vllm.sh
	VLLM_BIN=$(VLLM) MODEL_PATH=$(MODEL_PATH) GPU=3 PORT=$(LOCAL_MOE_DOUBLE_PORT_1) \
		VLLM_EXTRA_ARGS="$(VLLM_EXTRA_ARGS)" ./run_vllm.sh

test-local-moe-double-test:
	PORT=$(LOCAL_MOE_DOUBLE_PORT_0) ./test_vllm.sh
	PORT=$(LOCAL_MOE_DOUBLE_PORT_1) ./test_vllm.sh

clean-vllm:
	./clean_vllm.sh

test-deploy:
#	$(MAKE) test-local-base-run
	$(MAKE) test-local-moe-run
	$(MAKE) test-local-moe-double-run

test-curl:
#	$(MAKE) test-local-base-test
	$(MAKE) test-local-moe-test
	$(MAKE) test-local-moe-double-test

# 0007 full local test suite
test:
	$(MAKE) test-deploy
	$(MAKE) test-curl

bench:
	PORT=8072 ./bench_vllm.sh
	PORT=8073 ./bench_vllm.sh
	PORT=8074 ./bench_vllm.sh
	PORT=80 ./bench_vllm.sh

deploy8: clean-vllm
	VLLM_BIN=$(VLLM) MODEL_PATH=$(MODEL_PATH) GPU=0 PORT=8080 VLLM_EXTRA_ARGS="$(VLLM_EXTRA_ARGS)" ./run_vllm.sh
	VLLM_BIN=$(VLLM) MODEL_PATH=$(MODEL_PATH) GPU=1 PORT=8081 VLLM_EXTRA_ARGS="$(VLLM_EXTRA_ARGS)" ./run_vllm.sh
	VLLM_BIN=$(VLLM) MODEL_PATH=$(MODEL_PATH) GPU=2 PORT=8082 VLLM_EXTRA_ARGS="$(VLLM_EXTRA_ARGS)" ./run_vllm.sh
	VLLM_BIN=$(VLLM) MODEL_PATH=$(MODEL_PATH) GPU=3 PORT=8083 VLLM_EXTRA_ARGS="$(VLLM_EXTRA_ARGS)" ./run_vllm.sh
	VLLM_BIN=$(VLLM) MODEL_PATH=$(MODEL_PATH) GPU=4 PORT=8084 VLLM_EXTRA_ARGS="$(VLLM_EXTRA_ARGS)" ./run_vllm.sh
	VLLM_BIN=$(VLLM) MODEL_PATH=$(MODEL_PATH) GPU=5 PORT=8085 VLLM_EXTRA_ARGS="$(VLLM_EXTRA_ARGS)" ./run_vllm.sh
	VLLM_BIN=$(VLLM) MODEL_PATH=$(MODEL_PATH) GPU=6 PORT=8086 VLLM_EXTRA_ARGS="$(VLLM_EXTRA_ARGS)" ./run_vllm.sh
	VLLM_BIN=$(VLLM) MODEL_PATH=$(MODEL_PATH) GPU=7 PORT=8087 VLLM_EXTRA_ARGS="$(VLLM_EXTRA_ARGS)" ./run_vllm.sh

bench80:
	PORT=80 ./bench_vllm.sh


test-case1: clean-vllm
	GPU_MEMORY_UTILIZATION=0.95 \
	VLLM_BIN=$(VLLM) \
	MODEL_PATH=$(MODEL_PATH) \
	GPU=$(TEST_CASE_GPU) PORT=$(TEST_CASE1_PORT) VLLM_EXTRA_ARGS="--moe-cpu-offload" ./run_vllm.sh
	NUM_PROMPTS=$(TEST_CASE_BENCH_NUM_PROMPTS) \
	REQUEST_RATE=$(TEST_CASE_BENCH_REQUEST_RATE) \
	RANDOM_INPUT_LEN=$(TEST_CASE_BENCH_INPUT_LEN) \
	RANDOM_OUTPUT_LEN=$(TEST_CASE_BENCH_OUTPUT_LEN) \
	MAX_CONCURRENCY=$(TEST_CASE_BENCH_CONCURRENCY) \
	PORT=$(TEST_CASE1_PORT) ./bench_vllm.sh

test-case2: clean-vllm
	@for gpu_limit in $(TEST_CASE2_GPU_MEMORY_UTILIZATION_LIST); do \
		for prefetch in $(TEST_CASE2_PREFETCH_LIST); do \
			echo "============================================================"; \
			echo "Case 2 benchmark: gpu_memory_utilization=$${gpu_limit} prefetch=$${prefetch}"; \
			echo "============================================================"; \
			./clean_vllm.sh; \
			GPU_MEMORY_UTILIZATION=$${gpu_limit} \
			VLLM_BIN=$(VLLM) \
			MODEL_PATH=$(MODEL_PATH) \
			GPU=$(TEST_CASE_GPU) PORT=$(TEST_CASE2_PORT) VLLM_EXTRA_ARGS="--moe-gpu-prefetch $${prefetch}" ./run_vllm.sh || { \
				echo "Case 2 startup failed: gpu_memory_utilization=$${gpu_limit} prefetch=$${prefetch}"; \
				continue; \
			}; \
			NUM_PROMPTS=$(TEST_CASE2_BENCH_NUM_PROMPTS) \
			REQUEST_RATE=$(TEST_CASE2_BENCH_REQUEST_RATE) \
			RANDOM_INPUT_LEN=$(TEST_CASE_BENCH_INPUT_LEN) \
			RANDOM_OUTPUT_LEN=$(TEST_CASE_BENCH_OUTPUT_LEN) \
			MAX_CONCURRENCY=$(TEST_CASE2_BENCH_CONCURRENCY) \
			PORT=$(TEST_CASE2_PORT) ./bench_vllm.sh || { \
				echo "Case 2 benchmark failed: gpu_memory_utilization=$${gpu_limit} prefetch=$${prefetch}"; \
				continue; \
			}; \
		done; \
	done

test-case:
	$(MAKE) test-case1
	$(MAKE) test-case2

clean:
	./clean_vllm.sh


manual-test:
	$(MAKE) clean
	$(MAKE) compile
	$(MAKE) test

# 0008 coding step using Codex
coding:
	./run_codex.sh "Focus on Case 1 and Case 2 in the test-case target. For each case, if the test fails, try to fix it by modifying the run_vllm.sh script or the test_vllm.sh script. If you need to modify the run_vllm.sh script, make sure to keep the same interface and only change the internal implementation. If you need to modify the test_vllm.sh script, make sure to keep the same test cases and only change the expected outputs. After making changes, run the test-case target again to see if it passes. Repeat this process until both cases pass. Remember DO NOT mix case 1 and case 2 code, use seperate file for each case."
	$(MAKE) compile || $(MAKE) coding

loop:
	$(MAKE) clean
	$(MAKE) test-case || { $(MAKE) coding && $(MAKE) loop; }
