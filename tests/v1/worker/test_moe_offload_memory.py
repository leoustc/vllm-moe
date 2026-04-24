# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.config import CacheConfig, MoEOffloadConfig, VllmConfig
from vllm.utils.mem_utils import MemorySnapshot
from vllm.v1.worker.utils import request_memory

GiB = 1 << 30


def make_snapshot(free_gib: float, total_gib: float) -> MemorySnapshot:
    return MemorySnapshot(
        free_memory=int(free_gib * GiB),
        total_memory=int(total_gib * GiB),
        device="cuda:0",
        auto_measure=False,
    )


def test_request_memory_respects_moe_gpu_limit():
    snapshot = make_snapshot(free_gib=12, total_gib=16)
    vllm_config = VllmConfig(
        cache_config=CacheConfig(gpu_memory_utilization=0.9),
        moe_offload_config=MoEOffloadConfig(enabled=True, gpu_limit=0.5),
    )

    requested = request_memory(snapshot, vllm_config)

    assert requested == 8 * GiB


def test_request_memory_uses_cache_limit_without_moe_override():
    snapshot = make_snapshot(free_gib=12, total_gib=16)
    vllm_config = VllmConfig(
        cache_config=CacheConfig(gpu_memory_utilization=0.75),
        moe_offload_config=MoEOffloadConfig(enabled=True, gpu_limit=0.9),
    )

    requested = request_memory(snapshot, vllm_config)

    assert requested == 12 * GiB


def test_request_memory_reports_moe_limit_in_error():
    snapshot = make_snapshot(free_gib=7, total_gib=16)
    vllm_config = VllmConfig(
        cache_config=CacheConfig(gpu_memory_utilization=0.9),
        moe_offload_config=MoEOffloadConfig(enabled=True, gpu_limit=0.5),
    )

    with pytest.raises(ValueError, match="moe_gpu_limit=0.5"):
        request_memory(snapshot, vllm_config)
