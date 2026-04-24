# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import field

from pydantic import Field, model_validator

from vllm.config.utils import config


@config
class MoEOffloadConfig:
    """Configuration for sparse-MoE CPU offload mode."""

    enabled: bool = False
    """Enable sparse-MoE CPU offload mode."""

    gpu_limit: float | None = Field(default=None, gt=0, le=1)
    """Optional GPU memory limit for offload-aware runtime accounting."""

    active_expert_budget: int | None = Field(default=None, ge=1)
    """Optional cap on active experts retained on GPU."""

    fallback_expert_budget: int | None = Field(default=None, ge=0)
    """Optional cap on fallback experts retained on GPU."""

    wave_min_tokens: int = Field(default=2048, ge=1)
    """Minimum queued tokens before launching a utilization wave."""

    wave_max_wait_ms: int = Field(default=200, ge=0)
    """Maximum time to wait before forcing a wave launch."""

    transfer_safety_factor: float = Field(default=1.5, gt=0)
    """Transfer-hiding factor for deciding when to launch a wave."""

    group_fallback: bool = False
    """Enable same-group fallback when an exact expert is not resident."""

    group_fallback_threshold: float = Field(default=0.8, gt=0, le=1)
    """Similarity threshold for same-group fallback."""

    resident_first: bool = True
    """Execute resident expert buckets before newly loaded buckets."""

    min_residency_steps: int = Field(default=3, ge=0)
    """Minimum number of scheduling steps to retain a loaded expert."""

    metrics_tags: dict[str, str] = field(default_factory=dict, init=False)
    """Derived metrics tags for observability."""

    @model_validator(mode="after")
    def _validate_relationships(self) -> "MoEOffloadConfig":
        if not self.enabled:
            if self.gpu_limit is not None:
                raise ValueError(
                    "moe_gpu_limit requires moe_cpu_offload to be enabled."
                )
            if self.active_expert_budget is not None:
                raise ValueError(
                    "moe_active_expert_budget requires moe_cpu_offload "
                    "to be enabled."
                )
            if self.fallback_expert_budget is not None:
                raise ValueError(
                    "moe_fallback_expert_budget requires moe_cpu_offload "
                    "to be enabled."
                )
            if self.group_fallback:
                raise ValueError(
                    "moe_group_fallback requires moe_cpu_offload to be enabled."
                )
        return self

    def effective_gpu_limit(self, default_limit: float) -> float:
        """Return the stricter of the generic runtime limit and MoE GPU cap."""
        if not self.enabled or self.gpu_limit is None:
            return default_limit
        return min(default_limit, self.gpu_limit)

    def effective_wave_target(self, default_tokens: int) -> int:
        """Return the minimum token wave size to target under offload mode."""
        if not self.enabled:
            return default_tokens
        return max(default_tokens, self.wave_min_tokens)

    def metrics_info(self) -> dict[str, str]:
        return {
            "enabled": str(self.enabled),
            "gpu_limit": str(self.gpu_limit),
            "active_expert_budget": str(self.active_expert_budget),
            "fallback_expert_budget": str(self.fallback_expert_budget),
            "wave_min_tokens": str(self.wave_min_tokens),
            "wave_max_wait_ms": str(self.wave_max_wait_ms),
            "transfer_safety_factor": str(self.transfer_safety_factor),
            "group_fallback": str(self.group_fallback),
            "group_fallback_threshold": str(self.group_fallback_threshold),
            "resident_first": str(self.resident_first),
            "min_residency_steps": str(self.min_residency_steps),
        }
