# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from vllm.config import MoEOffloadConfig


@dataclass
class ExpertGrouping:
    """Static grouping metadata for optional same-group fallback."""

    expert_to_group: dict[int, int]
    group_to_experts: dict[int, list[int]]
    similarity: dict[tuple[int, int], float] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls,
        expert_to_group: dict[int, int],
        similarity: dict[tuple[int, int], float] | None = None,
    ) -> "ExpertGrouping":
        group_to_experts: dict[int, list[int]] = {}
        for expert_id, group_id in expert_to_group.items():
            group_to_experts.setdefault(group_id, []).append(expert_id)
        return cls(
            expert_to_group=expert_to_group,
            group_to_experts={
                group_id: sorted(experts)
                for group_id, experts in group_to_experts.items()
            },
            similarity=similarity or {},
        )

    def get_group(self, expert_id: int) -> int | None:
        return self.expert_to_group.get(expert_id)

    def get_similarity(self, expert_id: int, candidate_id: int) -> float:
        if expert_id == candidate_id:
            return 1.0
        return self.similarity.get(
            (expert_id, candidate_id),
            self.similarity.get((candidate_id, expert_id), 0.0),
        )


@dataclass
class ExpertResidencyState:
    """Mutable residency state for sparse-MoE CPU offload planning."""

    active_experts: set[int] = field(default_factory=set)
    fallback_experts: set[int] = field(default_factory=set)
    resident_experts: set[int] = field(default_factory=set)
    cpu_offloaded_experts: set[int] = field(default_factory=set)
    resident_expert_bytes: int = 0
    expert_last_used_step: dict[int, int] = field(default_factory=dict)
    expert_tokens_served: dict[int, int] = field(default_factory=dict)
    expert_load_count: dict[int, int] = field(default_factory=dict)
    expert_bytes_loaded: dict[int, int] = field(default_factory=dict)

    def record_expert_load(self, expert_id: int, expert_bytes: int) -> None:
        self.resident_experts.add(expert_id)
        self.cpu_offloaded_experts.discard(expert_id)
        self.expert_load_count[expert_id] = self.expert_load_count.get(expert_id, 0) + 1
        self.expert_bytes_loaded[expert_id] = (
            self.expert_bytes_loaded.get(expert_id, 0) + expert_bytes
        )
        self.resident_expert_bytes += expert_bytes

    def record_expert_eviction(self, expert_id: int, expert_bytes: int) -> None:
        self.active_experts.discard(expert_id)
        self.fallback_experts.discard(expert_id)
        if expert_id in self.resident_experts:
            self.resident_experts.remove(expert_id)
            self.cpu_offloaded_experts.add(expert_id)
            self.resident_expert_bytes = max(
                0, self.resident_expert_bytes - expert_bytes
            )

    def record_tokens_served(
        self, expert_ids: set[int], token_count: int, step_idx: int
    ) -> None:
        if not expert_ids or token_count <= 0:
            return
        per_expert_tokens = token_count // len(expert_ids)
        for expert_id in expert_ids:
            self.expert_last_used_step[expert_id] = step_idx
            self.expert_tokens_served[expert_id] = (
                self.expert_tokens_served.get(expert_id, 0) + per_expert_tokens
            )

    def make_metrics(self) -> dict[str, float]:
        total_loads = sum(self.expert_load_count.values())
        total_bytes = sum(self.expert_bytes_loaded.values())
        total_tokens = sum(self.expert_tokens_served.values())
        reuse_samples = [
            self.expert_tokens_served.get(expert_id, 0)
            for expert_id in self.expert_load_count
        ]
        return {
            "active_expert_count": float(len(self.active_experts)),
            "fallback_expert_count": float(len(self.fallback_experts)),
            "resident_expert_bytes": float(self.resident_expert_bytes),
            "expert_delta_load_count": float(total_loads),
            "expert_delta_load_bytes": float(total_bytes),
            "tokens_per_expert_load": (
                float(total_tokens) / total_loads if total_loads else 0.0
            ),
            "bytes_loaded_per_token": (
                float(total_bytes) / total_tokens if total_tokens else 0.0
            ),
            "expert_reuse_before_eviction": (
                float(sum(reuse_samples)) / len(reuse_samples) if reuse_samples else 0.0
            ),
            "expert_cache_hit_ratio": (
                float(len(self.active_experts & self.resident_experts))
                / len(self.active_experts)
                if self.active_experts
                else 1.0
            ),
        }


@dataclass
class MoEWavePlan:
    """A single utilization-first sparse-MoE launch plan."""

    request_ids: list[str]
    token_count: int
    expert_buckets: dict[int, list[int]]
    active_experts: set[int]
    missing_experts: set[int]
    fallback_matches: dict[int, int]
    prefetch_delta: list[int]
    est_compute_ms: float
    est_transfer_ms: float
    est_kv_bytes: int
    est_token_wave_bytes: int
    est_staging_bytes: int
    projected_gpu_usage_bytes: int
    launch_reason: str


@dataclass
class ResidencyRefreshPlan:
    """Resident-set update for the next sparse-MoE wave."""

    retain: set[int]
    delta: list[int]
    evict: list[int]
    next_active: set[int]
    next_fallback: set[int]


class MoEOffloadPlanner:
    """Pure planning helpers for sparse-MoE CPU offload mode."""

    def __init__(
        self,
        config: MoEOffloadConfig,
        grouping: ExpertGrouping | None = None,
    ) -> None:
        self.config = config
        self.grouping = grouping

    def compute_active_delta(
        self, resident_experts: set[int], active_experts: set[int]
    ) -> list[int]:
        return sorted(active_experts - resident_experts)

    def estimate_projected_gpu_usage(
        self,
        *,
        base_model_gpu_bytes: int,
        kv_cache_bytes: int,
        resident_expert_bytes: int,
        token_wave_bytes: int,
        staging_and_workspace_bytes: int,
    ) -> int:
        return (
            base_model_gpu_bytes
            + kv_cache_bytes
            + resident_expert_bytes
            + token_wave_bytes
            + staging_and_workspace_bytes
        )

    def should_launch_wave(
        self,
        *,
        queued_tokens: int,
        oldest_request_wait_ms: int,
        est_compute_ms: float,
        est_transfer_ms: float,
        projected_gpu_usage_bytes: int,
        total_gpu_bytes: int,
    ) -> tuple[bool, Literal["wave_ready", "wait_bound", "insufficient_tokens",
                              "transfer_not_hidden", "gpu_limit_exceeded"]]:
        if not self.config.enabled:
            return True, "wave_ready"

        effective_limit = self.config.effective_gpu_limit(1.0)
        allowed_gpu_bytes = int(total_gpu_bytes * effective_limit)
        if oldest_request_wait_ms >= self.config.wave_max_wait_ms:
            if projected_gpu_usage_bytes <= allowed_gpu_bytes:
                return True, "wait_bound"
            return False, "gpu_limit_exceeded"

        if queued_tokens < self.config.wave_min_tokens:
            return False, "insufficient_tokens"

        if projected_gpu_usage_bytes > allowed_gpu_bytes:
            return False, "gpu_limit_exceeded"

        if est_compute_ms < self.config.transfer_safety_factor * est_transfer_ms:
            return False, "transfer_not_hidden"

        return True, "wave_ready"

    def choose_fallback_expert(
        self,
        routed_expert: int,
        resident_experts: set[int],
    ) -> int | None:
        if (
            not self.config.enabled
            or not self.config.group_fallback
            or self.grouping is None
        ):
            return None

        group_id = self.grouping.get_group(routed_expert)
        if group_id is None:
            return None

        candidates = [
            expert_id
            for expert_id in self.grouping.group_to_experts.get(group_id, [])
            if expert_id in resident_experts and expert_id != routed_expert
        ]
        if not candidates:
            return None

        best_expert = max(
            candidates,
            key=lambda expert_id: self.grouping.get_similarity(
                routed_expert, expert_id
            ),
        )
        similarity = self.grouping.get_similarity(routed_expert, best_expert)
        if similarity < self.config.group_fallback_threshold:
            return None
        return best_expert

    def compute_fallback_matches(
        self, missing_experts: set[int], resident_experts: set[int]
    ) -> dict[int, int]:
        matches: dict[int, int] = {}
        for expert_id in sorted(missing_experts):
            fallback_expert = self.choose_fallback_expert(expert_id, resident_experts)
            if fallback_expert is not None:
                matches[expert_id] = fallback_expert
        return matches

    def build_residency_refresh_plan(
        self,
        *,
        state: ExpertResidencyState,
        next_active: set[int],
        next_fallback: set[int],
        step_idx: int,
        max_resident_experts: int | None = None,
    ) -> ResidencyRefreshPlan:
        protected = set(next_active)
        retain = {
            expert_id
            for expert_id in state.resident_experts
            if expert_id in next_active or expert_id in next_fallback
            or step_idx - state.expert_last_used_step.get(expert_id, step_idx)
            < self.config.min_residency_steps
        }

        evictable = sorted(
            state.resident_experts - retain,
            key=lambda expert_id: (
                expert_id not in state.fallback_experts,
                state.expert_tokens_served.get(expert_id, 0),
                state.expert_last_used_step.get(expert_id, -1),
                expert_id,
            ),
        )

        future_resident = set(state.resident_experts)
        delta: list[int] = []
        evict: list[int] = []

        def ensure_capacity(required_free_slots: int, *, allow_eviction: bool) -> bool:
            if max_resident_experts is None:
                return True
            overflow = len(future_resident) + required_free_slots - max_resident_experts
            if overflow <= 0:
                return True
            if not allow_eviction:
                return False
            candidates = evictable + [
                expert_id
                for expert_id in sorted(future_resident - protected)
                if expert_id not in evict
            ]
            for expert_id in candidates:
                if overflow <= 0 or expert_id in protected or expert_id in evict:
                    continue
                future_resident.discard(expert_id)
                evict.append(expert_id)
                overflow -= 1
            return overflow <= 0

        for expert_id in sorted(next_active - future_resident):
            if not ensure_capacity(1, allow_eviction=True):
                break
            future_resident.add(expert_id)
            delta.append(expert_id)

        next_active_resident = next_active & future_resident
        fallback_budget = self.config.fallback_expert_budget
        next_fallback_candidates = set(next_fallback)
        if fallback_budget is not None:
            next_fallback_candidates = set(sorted(next_fallback_candidates)[:fallback_budget])

        for expert_id in sorted(next_fallback_candidates - future_resident):
            if not ensure_capacity(1, allow_eviction=False):
                break
            future_resident.add(expert_id)
            delta.append(expert_id)

        next_fallback_resident = next_fallback_candidates & future_resident

        return ResidencyRefreshPlan(
            retain=retain,
            delta=delta,
            evict=evict,
            next_active=next_active_resident,
            next_fallback=next_fallback_resident,
        )
