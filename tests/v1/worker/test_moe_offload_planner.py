# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.config import MoEOffloadConfig
from vllm.v1.worker.moe_offload import (
    ExpertGrouping,
    ExpertResidencyState,
    MoEOffloadPlanner,
)


def test_compute_active_delta_only_fetches_missing_experts():
    planner = MoEOffloadPlanner(MoEOffloadConfig(enabled=True))

    assert planner.compute_active_delta({1, 2, 4}, {2, 3, 4, 5}) == [3, 5]


def test_should_launch_wave_respects_transfer_hiding_and_wait_bound():
    planner = MoEOffloadPlanner(
        MoEOffloadConfig(
            enabled=True,
            wave_min_tokens=2048,
            wave_max_wait_ms=200,
            transfer_safety_factor=1.5,
            gpu_limit=0.5,
        )
    )

    launch, reason = planner.should_launch_wave(
        queued_tokens=4096,
        oldest_request_wait_ms=100,
        est_compute_ms=10.0,
        est_transfer_ms=8.0,
        projected_gpu_usage_bytes=4,
        total_gpu_bytes=16,
    )
    assert not launch
    assert reason == "transfer_not_hidden"

    launch, reason = planner.should_launch_wave(
        queued_tokens=128,
        oldest_request_wait_ms=250,
        est_compute_ms=1.0,
        est_transfer_ms=100.0,
        projected_gpu_usage_bytes=4,
        total_gpu_bytes=16,
    )
    assert launch
    assert reason == "wait_bound"


def test_should_launch_wave_respects_gpu_limit():
    planner = MoEOffloadPlanner(
        MoEOffloadConfig(enabled=True, wave_min_tokens=512, gpu_limit=0.5)
    )

    launch, reason = planner.should_launch_wave(
        queued_tokens=2048,
        oldest_request_wait_ms=0,
        est_compute_ms=10.0,
        est_transfer_ms=1.0,
        projected_gpu_usage_bytes=10,
        total_gpu_bytes=16,
    )

    assert not launch
    assert reason == "gpu_limit_exceeded"


def test_choose_same_group_fallback_requires_flag_and_threshold():
    grouping = ExpertGrouping.from_mapping(
        {0: 0, 1: 0, 2: 1},
        similarity={(0, 1): 0.9},
    )
    planner = MoEOffloadPlanner(
        MoEOffloadConfig(
            enabled=True,
            group_fallback=True,
            group_fallback_threshold=0.8,
        ),
        grouping=grouping,
    )

    assert planner.choose_fallback_expert(0, {1, 2}) == 1

    strict_planner = MoEOffloadPlanner(
        MoEOffloadConfig(
            enabled=True,
            group_fallback=True,
            group_fallback_threshold=0.95,
        ),
        grouping=grouping,
    )
    assert strict_planner.choose_fallback_expert(0, {1, 2}) is None


def test_build_residency_refresh_plan_prefers_cold_fallback_eviction():
    planner = MoEOffloadPlanner(
        MoEOffloadConfig(
            enabled=True,
            fallback_expert_budget=1,
            min_residency_steps=0,
        )
    )
    state = ExpertResidencyState(
        active_experts={0, 1},
        fallback_experts={2},
        resident_experts={0, 1, 2},
        expert_last_used_step={0: 10, 1: 10, 2: 1},
        expert_tokens_served={0: 100, 1: 80, 2: 1},
    )

    plan = planner.build_residency_refresh_plan(
        state=state,
        next_active={0, 3},
        next_fallback={4},
        step_idx=10,
        max_resident_experts=3,
    )

    assert plan.delta == [3]
    assert plan.evict == [2]
    assert plan.next_active == {0, 3}
    assert plan.next_fallback == set()


def test_residency_state_metrics_capture_reuse_and_bytes_loaded():
    state = ExpertResidencyState(active_experts={1, 2}, resident_experts={1, 2})
    state.record_expert_load(1, 100)
    state.record_expert_load(2, 100)
    state.record_tokens_served({1, 2}, token_count=40, step_idx=1)

    metrics = state.make_metrics()

    assert metrics["expert_delta_load_count"] == 2.0
    assert metrics["expert_delta_load_bytes"] == 200.0
    assert metrics["tokens_per_expert_load"] == 20.0
    assert metrics["bytes_loaded_per_token"] == 5.0
    assert metrics["expert_cache_hit_ratio"] == 1.0
