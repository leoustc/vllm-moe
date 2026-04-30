# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from threading import Condition, Event, RLock, Thread
from typing import Literal

import torch

from vllm.logger import init_logger

ExpertState = Literal["loading", "resident", "executing", "evicting"]
GPU_MEMORY_RETRY_SECONDS = 5
GPU_MEMORY_RETRY_LIMIT = 10
PAGER_LOG_INTERVAL_SECONDS = 5.0
PAGER_POLL_SECONDS = 0.005
PAGER_MAIN_WAIT_SECONDS = 0.002
PAGER_COPY_STREAM_COUNT = int(os.environ.get("VLLM_MOE_PREFETCH_COPY_STREAMS", "2"))
MAX_LOGGED_EXPERT_IDS = 16

logger = init_logger(__name__)


@dataclass
class ActiveExpertEntry:
    layer_id: int
    expert_id: int
    gpu_slot_id: int
    state: ExpertState
    weight_bytes: int
    loaded_step: int
    last_used_step: int
    recent_token_count: int
    ready_event: torch.cuda.Event | None = None


@dataclass
class ExpertCacheMetrics:
    resident_hits: int = 0
    missing_loads: int = 0
    evictions: int = 0
    h2d_bytes: int = 0
    wait_iterations: int = 0
    resident_tokens: int = 0
    missing_tokens: int = 0


@dataclass
class _ExpertTensor:
    name: str
    source: torch.Tensor
    target: torch.Tensor


class MoeGpuPrefetchExpertCache:
    """MoE GPU prefetch cache for Case 2.

    The CPU tensors are the source of truth. The layer parameters remain the
    execution tensors; when an expert is demanded, its slice is copied from CPU
    to the current parameter device before the fused MoE kernel runs.
    """

    def __init__(
        self,
        *,
        layer_id: int,
        active_expert_budget: int | None,
        expert_tensors: list[_ExpertTensor],
        use_identity_slots: bool = True,
    ) -> None:
        if active_expert_budget is not None and active_expert_budget < 1:
            raise ValueError("active_expert_budget must be at least 1")
        if not expert_tensors:
            raise ValueError(
                "MoeGpuPrefetchExpertCache requires at least one expert tensor"
            )

        self.layer_id = layer_id
        self.expert_tensors = expert_tensors
        self.use_identity_slots = use_identity_slots
        self.active_experts: dict[int, ActiveExpertEntry] = {}
        self.step = 0

        self._num_experts = self.expert_tensors[0].source.shape[0]
        self.auto_active_expert_budget = (
            active_expert_budget is None and not self.use_identity_slots
        )
        self.active_expert_budget = active_expert_budget or self._num_experts
        self._slot_count = (
            self.expert_tensors[0].target.shape[0]
            if self.use_identity_slots
            else self.active_expert_budget
        )
        self._free_slots = list(range(self._slot_count))
        self._bytes_by_expert = self._compute_bytes_by_expert()
        self._target_device = self.expert_tensors[0].target.device
        self._last_pager_summary_time = 0.0
        self.working_experts: set[int] = set()
        self.missing_experts: set[int] = set()
        self.missing_expert_counts: dict[int, int] = {}
        self._pager_lock = RLock()
        self._pager_condition = Condition(self._pager_lock)
        self._pager_stop = Event()
        self._pager_thread: Thread | None = None
        self._copy_streams: list[torch.cuda.Stream] = []
        self._copy_stream_device: torch.device | None = None
        self._next_copy_stream = 0
        self._slot_safe_events: dict[int, torch.cuda.Event] = {}
        self._slot_map: torch.Tensor | None = None
        self.metrics = ExpertCacheMetrics()

    @classmethod
    def from_cpu_sources(
        cls,
        *,
        layer_id: int,
        active_expert_budget: int | None,
        sources: dict[str, torch.Tensor],
        device: torch.device,
    ) -> MoeGpuPrefetchExpertCache:
        expert_tensors: list[_ExpertTensor] = []
        for name, source in sources.items():
            source = cls._make_cpu_source(source)
            target = torch.empty((0, *source.shape[1:]), dtype=source.dtype)
            expert_tensors.append(
                _ExpertTensor(name=name, source=source, target=target)
            )

        cache = cls(
            layer_id=layer_id,
            active_expert_budget=active_expert_budget,
            expert_tensors=expert_tensors,
            use_identity_slots=False,
        )
        cache._target_device = device
        return cache

    @staticmethod
    def _make_cpu_source(tensor: torch.Tensor) -> torch.Tensor:
        source = tensor.detach().to(device="cpu", copy=True).contiguous()
        if torch.cuda.is_available():
            source = source.pin_memory()
        return source

    def _compute_bytes_by_expert(self) -> dict[int, int]:
        bytes_by_expert: dict[int, int] = {}
        for expert_id in range(self._num_experts):
            bytes_by_expert[expert_id] = sum(
                int(t.source[expert_id].numel() * t.source.element_size())
                for t in self.expert_tensors
            )
        return bytes_by_expert

    def _required_cache_bytes(self, slot_count: int) -> int:
        if not self._bytes_by_expert:
            return 0
        return max(self._bytes_by_expert.values()) * slot_count

    def _set_slot_count(self, slot_count: int) -> None:
        slot_count = max(1, min(slot_count, self._num_experts))
        if slot_count == self._slot_count:
            return
        self.active_expert_budget = slot_count
        self._slot_count = slot_count
        self.release_targets_to_cpu()

    def _fit_auto_budget_to_available_memory(self, active_expert_count: int) -> None:
        if not self.auto_active_expert_budget:
            return
        desired_slots = max(1, min(active_expert_count, self._num_experts))
        if self._target_device.type == "cuda":
            torch.cuda.empty_cache()
            free_bytes, _ = torch.cuda.mem_get_info(self._target_device)
            max_expert_bytes = max(self._bytes_by_expert.values())
            if max_expert_bytes > 0:
                desired_slots = min(
                    desired_slots,
                    max(1, int(free_bytes // max_expert_bytes)),
                )
        self._set_slot_count(desired_slots)

    def _targets_are_allocated_on(self, device: torch.device) -> bool:
        return all(
            expert_tensor.target.device == device
            and expert_tensor.target.shape[0] == self._slot_count
            for expert_tensor in self.expert_tensors
        )

    def ensure_targets_on_device(self, device: torch.device) -> None:
        if self._targets_are_allocated_on(device):
            return
        if device.type == "cuda":
            required_bytes = self._required_cache_bytes(self._slot_count)
            for attempt in range(GPU_MEMORY_RETRY_LIMIT + 1):
                free_bytes, _ = torch.cuda.mem_get_info(device)
                if free_bytes >= required_bytes:
                    break
                if attempt == GPU_MEMORY_RETRY_LIMIT:
                    raise RuntimeError(
                        "Insufficient free GPU memory to allocate MoE active "
                        f"expert cache for layer {self.layer_id}: required "
                        f"{required_bytes} bytes, free {free_bytes} bytes after "
                        f"{GPU_MEMORY_RETRY_LIMIT} retries."
                    )
                torch.cuda.empty_cache()
                time.sleep(GPU_MEMORY_RETRY_SECONDS)

        for expert_tensor in self.expert_tensors:
            source = expert_tensor.source
            expert_tensor.target = torch.empty(
                (self._slot_count, *source.shape[1:]),
                dtype=source.dtype,
                device=device,
            )
        self._target_device = device
        self._free_slots = list(range(self._slot_count))
        self.active_experts.clear()
        self._slot_safe_events.clear()
        self._slot_map = None

    def release_targets_to_cpu(self) -> None:
        if self.use_identity_slots:
            return
        for expert_tensor in self.expert_tensors:
            source = expert_tensor.source
            expert_tensor.target = torch.empty(
                (0, *source.shape[1:]),
                dtype=source.dtype,
                device="cpu",
            )
        self._free_slots = list(range(self._slot_count))
        self.active_experts.clear()
        self._slot_safe_events.clear()
        self._slot_map = None
        with self._pager_lock:
            self.working_experts.clear()
            self.missing_experts.clear()
            self.missing_expert_counts.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _copy_stream_for(
        self,
        device: torch.device,
    ) -> torch.cuda.Stream | None:
        if device.type != "cuda":
            return None
        if not self._copy_streams or self._copy_stream_device != device:
            with torch.cuda.device(device):
                self._copy_streams = [
                    torch.cuda.Stream(device=device)
                    for _ in range(PAGER_COPY_STREAM_COUNT)
                ]
            self._copy_stream_device = device
            self._next_copy_stream = 0
        copy_stream = self._copy_streams[self._next_copy_stream]
        self._next_copy_stream = (self._next_copy_stream + 1) % len(self._copy_streams)
        return copy_stream

    def _slot_map_for(self, device: torch.device) -> torch.Tensor:
        if (
            self._slot_map is None
            or self._slot_map.device != device
            or self._slot_map.numel() != self._num_experts
        ):
            self._slot_map = torch.full(
                (self._num_experts,),
                -1,
                dtype=torch.long,
                device=device,
            )
            for expert_id, entry in self.active_experts.items():
                self._slot_map[expert_id] = entry.gpu_slot_id
        return self._slot_map

    def _set_slot_map_entry(self, expert_id: int, slot_id: int | None) -> None:
        if self._slot_map is None:
            return
        self._slot_map[expert_id] = -1 if slot_id is None else slot_id

    def _load_expert(
        self,
        expert_id: int,
        slot_id: int,
    ) -> torch.cuda.Event | None:
        self.ensure_targets_on_device(self._target_device)
        device = self.expert_tensors[0].target.device
        copy_stream = self._copy_stream_for(device)
        if copy_stream is None:
            for expert_tensor in self.expert_tensors:
                source = expert_tensor.source[expert_id]
                target = expert_tensor.target[slot_id]
                target.copy_(source, non_blocking=True)
            self.metrics.h2d_bytes += self._bytes_by_expert[expert_id]
            return None

        with torch.cuda.device(device), torch.cuda.stream(copy_stream):
            slot_safe_event = self._slot_safe_events.pop(slot_id, None)
            if slot_safe_event is not None:
                copy_stream.wait_event(slot_safe_event)
            for expert_tensor in self.expert_tensors:
                source = expert_tensor.source[expert_id]
                target = expert_tensor.target[slot_id]
                target.copy_(source, non_blocking=True)
            ready_event = torch.cuda.Event()
            ready_event.record(copy_stream)
        self.metrics.h2d_bytes += self._bytes_by_expert[expert_id]
        return ready_event

    def _wait_for_copy_event(self, event: torch.cuda.Event | None) -> None:
        if event is None:
            return
        event.synchronize()

    def wait_for_experts_ready(self, expert_ids: set[int]) -> None:
        if self._target_device.type != "cuda":
            return
        stream = torch.cuda.current_stream(self._target_device)
        for expert_id in expert_ids:
            entry = self.active_experts.get(expert_id)
            if entry is None:
                raise RuntimeError(
                    f"Expert {expert_id} is not resident in the MoE offload cache"
                )
            if entry.ready_event is not None:
                stream.wait_event(entry.ready_event)
            entry.state = "executing"

    def mark_experts_available(self, expert_ids: set[int]) -> None:
        if not expert_ids:
            return
        use_event: torch.cuda.Event | None = None
        if self._target_device.type == "cuda":
            use_event = torch.cuda.Event()
            use_event.record(torch.cuda.current_stream(self._target_device))
        for expert_id in expert_ids:
            entry = self.active_experts.get(expert_id)
            if entry is None:
                continue
            entry.state = "resident"
            if use_event is not None:
                self._slot_safe_events[entry.gpu_slot_id] = use_event

    def _allocate_slot(self, expert_id: int) -> int:
        if self.use_identity_slots:
            return expert_id
        if not self._free_slots:
            raise RuntimeError("No free MoE offload GPU expert slots are available")
        return self._free_slots.pop(0)

    def _evict_one(self, protected_expert_ids: set[int] | None = None) -> None:
        protected_expert_ids = protected_expert_ids or set()
        candidates = [
            entry
            for entry in self.active_experts.values()
            if (
                entry.state != "executing"
                and entry.expert_id not in protected_expert_ids
            )
        ]
        if not candidates:
            raise RuntimeError("No evictable MoE experts are available")

        victim = min(
            candidates,
            key=lambda entry: (entry.recent_token_count, entry.last_used_step),
        )
        victim.state = "evicting"
        if not self.use_identity_slots:
            self._free_slots.append(victim.gpu_slot_id)
            self._free_slots.sort()
        self._set_slot_map_entry(victim.expert_id, None)
        self.metrics.evictions += 1
        del self.active_experts[victim.expert_id]

    def _evict_expert(self, expert_id: int) -> None:
        entry = self.active_experts[expert_id]
        entry.state = "evicting"
        if not self.use_identity_slots:
            self._free_slots.append(entry.gpu_slot_id)
            self._free_slots.sort()
        self._set_slot_map_entry(expert_id, None)
        self.metrics.evictions += 1
        del self.active_experts[expert_id]

    def _ensure_capacity_for(
        self,
        expert_id: int,
        protected_expert_ids: set[int] | None = None,
    ) -> None:
        if expert_id in self.active_experts:
            return
        while len(self.active_experts) >= self.active_expert_budget:
            self._evict_one(protected_expert_ids)

    def _reserve_pager_slot_locked(
        self,
    ) -> tuple[int, int | None] | None:
        if self._free_slots:
            return self._free_slots.pop(0), None

        candidates = [
            entry
            for entry in self.active_experts.values()
            if (
                entry.state != "executing"
                and entry.expert_id not in self.working_experts
            )
        ]
        if not candidates:
            return None

        victim = min(
            candidates,
            key=lambda entry: (entry.recent_token_count, entry.last_used_step),
        )
        victim.state = "evicting"
        self._set_slot_map_entry(victim.expert_id, None)
        self.metrics.evictions += 1
        del self.active_experts[victim.expert_id]
        return victim.gpu_slot_id, victim.expert_id

    @staticmethod
    def _format_expert_ids(expert_ids: set[int]) -> str:
        sorted_ids = sorted(expert_ids)
        visible_ids = sorted_ids[:MAX_LOGGED_EXPERT_IDS]
        suffix = (
            f", ...(+{len(sorted_ids) - MAX_LOGGED_EXPERT_IDS})"
            if len(sorted_ids) > MAX_LOGGED_EXPERT_IDS
            else ""
        )
        return f"[{', '.join(str(expert_id) for expert_id in visible_ids)}{suffix}]"

    def _log_pager_state(
        self,
        *,
        event: str,
        working_experts: set[int],
        missing_experts: set[int],
        force: bool = False,
    ) -> None:
        if self.use_identity_slots:
            return

        now = time.monotonic()
        if (
            not force
            and now - self._last_pager_summary_time < PAGER_LOG_INTERVAL_SECONDS
        ):
            return
        self._last_pager_summary_time = now

        active_experts = set(self.active_experts)
        logger.debug(
            "MoE expert pager layer=%d event=%s active_model_list=%s "
            "working_model_list=%s missing_model_list=%s resident=%d/%d "
            "free_slots=%d step=%d",
            self.layer_id,
            event,
            self._format_expert_ids(active_experts),
            self._format_expert_ids(working_experts),
            self._format_expert_ids(missing_experts),
            len(active_experts),
            self.active_expert_budget,
            len(self._free_slots),
            self.step,
        )

    def ensure_experts_resident(
        self,
        expert_token_counts: dict[int, int],
        *,
        evict_unrequested: bool = True,
    ) -> None:
        """Load demanded experts synchronously and update LRU/hotness stats."""
        if not expert_token_counts:
            return
        self.ensure_targets_on_device(self._target_device)
        if (
            not self.use_identity_slots
            and len(expert_token_counts) > self.active_expert_budget
        ):
            raise RuntimeError(
                "MoE CPU offload cannot execute a batch that routes to "
                f"{len(expert_token_counts)} local experts with "
                f"active_expert_staging_slots={self.active_expert_budget}. "
                "Use smaller routed waves or reduce batch size."
            )
        required_experts = set(expert_token_counts)
        missing_experts = required_experts - set(self.active_experts)
        if missing_experts:
            self._log_pager_state(
                event="miss",
                working_experts=required_experts,
                missing_experts=missing_experts,
                force=True,
            )
        if not self.use_identity_slots and evict_unrequested:
            for expert_id in list(self.active_experts):
                if expert_id not in required_experts:
                    self._evict_expert(expert_id)

        self.step += 1
        for expert_id, token_count in sorted(
            expert_token_counts.items(), key=lambda item: (-item[1], item[0])
        ):
            self._ensure_capacity_for(expert_id, required_experts)
            entry = self.active_experts.get(expert_id)
            if entry is None:
                slot_id = self._allocate_slot(expert_id)
                ready_event = self._load_expert(expert_id, slot_id)
                self._wait_for_copy_event(ready_event)
                entry = ActiveExpertEntry(
                    layer_id=self.layer_id,
                    expert_id=expert_id,
                    gpu_slot_id=slot_id,
                    state="resident",
                    weight_bytes=self._bytes_by_expert[expert_id],
                    loaded_step=self.step,
                    last_used_step=self.step,
                    recent_token_count=token_count,
                    ready_event=None,
                )
                self.active_experts[expert_id] = entry
                self._set_slot_map_entry(expert_id, slot_id)
            else:
                entry.state = "resident"
                entry.last_used_step = self.step
                entry.recent_token_count = token_count
                self.metrics.resident_hits += 1
        self._log_pager_state(
            event="summary",
            working_experts=required_experts,
            missing_experts=set(),
        )

    def start_prefetch_pager(self) -> None:
        if self.use_identity_slots:
            return
        if self._pager_thread is not None:
            return
        self._pager_stop.clear()
        self._pager_thread = Thread(
            target=self._prefetch_pager_loop,
            name=f"moe-prefetch-pager-layer-{self.layer_id}",
            daemon=True,
        )
        self._pager_thread.start()

    def stop_prefetch_pager(self) -> None:
        self._pager_stop.set()
        with self._pager_condition:
            self._pager_condition.notify_all()

    def _prefetch_pager_loop(self) -> None:
        while not self._pager_stop.is_set():
            loaded = self.pager_step()
            if loaded:
                continue
            with self._pager_condition:
                self._pager_condition.wait(timeout=PAGER_POLL_SECONDS)

    def prepare_prefetch_request(
        self,
        expert_token_counts: dict[int, int],
        *,
        wait_for_resident: bool = False,
    ) -> list[dict[int, int]]:
        """Publish routed demand and return resident expert waves to compute."""
        if self.use_identity_slots:
            return self.expert_batches_for_counts(expert_token_counts)
        if not expert_token_counts:
            return []

        self._fit_auto_budget_to_available_memory(len(expert_token_counts))
        requested_experts = set(expert_token_counts)
        with self._pager_condition:
            self.step += 1
            self.working_experts = set(requested_experts)
            missing_experts = requested_experts - set(self.active_experts)
            self.missing_experts = set(missing_experts)
            self.missing_expert_counts = {
                expert_id: expert_token_counts[expert_id]
                for expert_id in missing_experts
            }
            if missing_experts:
                self._log_pager_state(
                    event="miss",
                    working_experts=self.working_experts,
                    missing_experts=self.missing_experts,
                    force=True,
                )
            self._pager_condition.notify_all()
            while wait_for_resident and missing_experts:
                self._pager_condition.wait(timeout=PAGER_MAIN_WAIT_SECONDS)
                missing_experts = requested_experts - set(self.active_experts)
                self.missing_experts = set(missing_experts)
                self.missing_expert_counts = {
                    expert_id: expert_token_counts[expert_id]
                    for expert_id in missing_experts
                }
            resident_counts = {
                expert_id: token_count
                for expert_id, token_count in expert_token_counts.items()
                if expert_id in self.active_experts
            }
            for expert_id, token_count in resident_counts.items():
                entry = self.active_experts[expert_id]
                entry.last_used_step = self.step
                entry.recent_token_count = token_count
            self.metrics.resident_hits += len(resident_counts)

        sorted_counts = sorted(
            resident_counts.items(), key=lambda item: (-item[1], item[0])
        )
        return [
            dict(sorted_counts[index : index + self.active_expert_budget])
            for index in range(0, len(sorted_counts), self.active_expert_budget)
        ]

    def prepare_prefetch_batch(
        self,
        expert_token_counts: dict[int, int],
    ) -> tuple[list[dict[int, int]], list[dict[int, int]]]:
        """Publish all routed demand and split resident/missing compute waves."""
        if self.use_identity_slots:
            return self.expert_batches_for_counts(expert_token_counts), []
        if not expert_token_counts:
            return [], []

        self._fit_auto_budget_to_available_memory(len(expert_token_counts))
        requested_experts = set(expert_token_counts)
        with self._pager_condition:
            self.step += 1
            self.working_experts = set(requested_experts)
            resident_counts = {
                expert_id: token_count
                for expert_id, token_count in expert_token_counts.items()
                if expert_id in self.active_experts
            }
            missing_counts = {
                expert_id: token_count
                for expert_id, token_count in expert_token_counts.items()
                if expert_id not in self.active_experts
            }
            self.missing_experts = set(missing_counts)
            self.missing_expert_counts = dict(missing_counts)
            self.metrics.resident_hits += len(resident_counts)
            self.metrics.resident_tokens += sum(resident_counts.values())
            self.metrics.missing_tokens += sum(missing_counts.values())
            if missing_counts:
                self._log_pager_state(
                    event="miss",
                    working_experts=self.working_experts,
                    missing_experts=self.missing_experts,
                    force=True,
                )
            self._pager_condition.notify_all()
            for expert_id, token_count in resident_counts.items():
                entry = self.active_experts[expert_id]
                entry.last_used_step = self.step
                entry.recent_token_count = token_count

        return (
            self.expert_batches_for_counts(resident_counts),
            self.expert_batches_for_counts(missing_counts),
        )

    def wait_for_prefetch_experts(
        self,
        expert_ids: set[int],
    ) -> None:
        if self.use_identity_slots or not expert_ids:
            return

        missing_experts = expert_ids - set(self.active_experts)
        with self._pager_condition:
            while missing_experts:
                self._pager_condition.wait(timeout=PAGER_MAIN_WAIT_SECONDS)
                self.metrics.wait_iterations += 1
                missing_experts = expert_ids - set(self.active_experts)
            self.missing_experts.difference_update(expert_ids)
            for expert_id in expert_ids:
                self.missing_expert_counts.pop(expert_id, None)

    def finish_prefetch_request(self, expert_ids: set[int]) -> None:
        if self.use_identity_slots:
            return
        with self._pager_condition:
            self.mark_experts_available(expert_ids)
            self.working_experts.difference_update(expert_ids)
            for expert_id in expert_ids:
                self.missing_expert_counts.pop(expert_id, None)
            self._log_pager_state(
                event="summary",
                working_experts=self.working_experts,
                missing_experts=self.missing_experts,
            )
            self._pager_condition.notify_all()

    def pager_step(self) -> bool:
        """Load all immediately schedulable missing experts into GPU slots."""
        if self.use_identity_slots:
            return False

        self.ensure_targets_on_device(self._target_device)
        loaded_any = False
        while True:
            with self._pager_condition:
                missing_candidates = sorted(
                    (
                        expert_id
                        for expert_id in self.missing_experts
                        if expert_id not in self.active_experts
                    ),
                    key=lambda expert_id: (
                        -self.missing_expert_counts.get(expert_id, 0),
                        expert_id,
                    ),
                )
                if not missing_candidates:
                    return loaded_any
                expert_id = missing_candidates[0]
                token_count = self.missing_expert_counts.get(expert_id, 0)
                reserved = self._reserve_pager_slot_locked()
                if reserved is None:
                    self._log_pager_state(
                        event="wait",
                        working_experts=self.working_experts,
                        missing_experts=self.missing_experts,
                    )
                    return loaded_any
                slot_id, _victim_id = reserved

            try:
                ready_event = self._load_expert(expert_id, slot_id)
            except Exception:
                with self._pager_condition:
                    if slot_id not in self._free_slots:
                        self._free_slots.append(slot_id)
                        self._free_slots.sort()
                    self._pager_condition.notify_all()
                raise

            with self._pager_condition:
                self.active_experts[expert_id] = ActiveExpertEntry(
                    layer_id=self.layer_id,
                    expert_id=expert_id,
                    gpu_slot_id=slot_id,
                    state="resident",
                    weight_bytes=self._bytes_by_expert[expert_id],
                    loaded_step=self.step,
                    last_used_step=self.step,
                    recent_token_count=token_count,
                    ready_event=ready_event,
                )
                self.missing_experts.discard(expert_id)
                self.missing_expert_counts.pop(expert_id, None)
                self._set_slot_map_entry(expert_id, slot_id)
                self.metrics.missing_loads += 1
                self._log_pager_state(
                    event="load",
                    working_experts=self.working_experts,
                    missing_experts=self.missing_experts,
                    force=True,
                )
                self._pager_condition.notify_all()
            loaded_any = True

    def resident_expert_ids(self) -> set[int]:
        return set(self.active_experts)

    def retire_experts(self, expert_ids: set[int] | None = None) -> None:
        """Mark loaded experts non-resident and free their execution slots."""
        if expert_ids is None:
            expert_ids = set(self.active_experts)
        for expert_id in list(expert_ids):
            if expert_id in self.active_experts:
                self._evict_expert(expert_id)

    def target_for(self, name: str) -> torch.Tensor:
        for expert_tensor in self.expert_tensors:
            if expert_tensor.name == name:
                return expert_tensor.target
        raise KeyError(name)

    def move_targets_to(self, device: torch.device) -> None:
        self.ensure_targets_on_device(device)

    def remap_topk_ids(
        self,
        topk_ids: torch.Tensor,
        *,
        expert_map: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.use_identity_slots:
            return topk_ids

        slot_map = torch.full(
            (self._num_experts,),
            -1,
            dtype=torch.long,
            device=topk_ids.device,
        )
        for expert_id, entry in self.active_experts.items():
            slot_map[expert_id] = entry.gpu_slot_id

        local_ids = local_expert_ids_for_topk(
            topk_ids,
            local_num_experts=self._num_experts,
            expert_map=expert_map,
        )
        valid_local_ids = (local_ids >= 0) & (local_ids < self._num_experts)
        safe_local_ids = torch.where(valid_local_ids, local_ids, 0)
        slot_ids = slot_map[safe_local_ids]
        missing = valid_local_ids & (slot_ids < 0)
        if bool(torch.any(missing).item()):
            missing_ids = torch.unique(local_ids[missing], sorted=True)
            missing_id = int(missing_ids[0].to(device="cpu").item())
            raise RuntimeError(
                f"Expert {missing_id} is not resident in the MoE offload cache"
            )
        return torch.where(valid_local_ids, slot_ids, -1).to(dtype=topk_ids.dtype)

    def ensure_experts_resident_and_remap(
        self,
        topk_ids: torch.Tensor,
        *,
        local_num_experts: int,
        expert_map: torch.Tensor | None,
    ) -> torch.Tensor:
        token_counts = local_expert_token_counts(
            topk_ids,
            local_num_experts=local_num_experts,
            expert_map=expert_map,
        )
        self.ensure_experts_resident(token_counts)
        return self.remap_topk_ids(topk_ids, expert_map=expert_map)

    def expert_batches_for_counts(
        self,
        expert_token_counts: dict[int, int],
    ) -> list[dict[int, int]]:
        if not expert_token_counts:
            return []
        self._fit_auto_budget_to_available_memory(len(expert_token_counts))
        sorted_counts = sorted(
            expert_token_counts.items(), key=lambda item: (-item[1], item[0])
        )
        if self.use_identity_slots:
            return [dict(sorted_counts)]
        return [
            dict(sorted_counts[index : index + self.active_expert_budget])
            for index in range(0, len(sorted_counts), self.active_expert_budget)
        ]

    def make_wave_tensors(
        self,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        *,
        local_expert_ids: set[int],
        expert_map: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.use_identity_slots:
            return topk_ids, topk_weights

        slot_map = torch.full(
            (self._num_experts,),
            -1,
            dtype=torch.long,
            device=topk_ids.device,
        )
        for expert_id in local_expert_ids:
            entry = self.active_experts.get(expert_id)
            if entry is None:
                raise RuntimeError(
                    f"Expert {expert_id} is not resident in the MoE offload cache"
                )
            slot_map[expert_id] = entry.gpu_slot_id

        local_ids = local_expert_ids_for_topk(
            topk_ids,
            local_num_experts=self._num_experts,
            expert_map=expert_map,
        )
        valid_local_ids = (local_ids >= 0) & (local_ids < self._num_experts)
        safe_local_ids = torch.where(valid_local_ids, local_ids, 0)
        slot_ids = slot_map[safe_local_ids]
        selected = valid_local_ids & (slot_ids >= 0)
        remapped = torch.where(selected, slot_ids, 0).to(dtype=topk_ids.dtype)
        weights = torch.where(selected, topk_weights, torch.zeros_like(topk_weights))

        return remapped, weights


def local_expert_ids_for_topk(
    topk_ids: torch.Tensor,
    *,
    local_num_experts: int,
    expert_map: torch.Tensor | None,
) -> torch.Tensor:
    """Return routed local expert ids on the same device as ``topk_ids``."""
    ids = topk_ids.to(dtype=torch.long)
    valid_raw_ids = ids >= 0

    if expert_map is None:
        local_ids = ids
    else:
        expert_map = expert_map.to(device=ids.device, dtype=torch.long)
        valid_raw_ids &= ids < expert_map.numel()
        safe_ids = torch.where(valid_raw_ids, ids, 0)
        local_ids = expert_map[safe_ids]

    valid_local_ids = valid_raw_ids & (local_ids >= 0) & (local_ids < local_num_experts)
    return torch.where(valid_local_ids, local_ids, -1)


def local_expert_token_counts(
    topk_ids: torch.Tensor,
    *,
    local_num_experts: int,
    expert_map: torch.Tensor | None,
) -> dict[int, int]:
    """Return token counts keyed by local expert id.

    The routed id filtering and histogram are built with tensor operations on
    the routed tensor device. CPU only receives the compact unique expert ids
    and counts used for pager decisions.
    """
    local_ids = local_expert_ids_for_topk(
        topk_ids,
        local_num_experts=local_num_experts,
        expert_map=expert_map,
    )
    routed_ids = local_ids[local_ids >= 0]
    if routed_ids.numel() == 0:
        return {}

    counts = torch.bincount(routed_ids, minlength=local_num_experts)
    active_ids = torch.nonzero(counts, as_tuple=False).reshape(-1)
    cpu_unique_ids = active_ids.to(device="cpu", dtype=torch.long)
    cpu_counts = counts[active_ids].to(device="cpu", dtype=torch.long)
    return {
        int(expert_id): int(count)
        for expert_id, count in zip(cpu_unique_ids.tolist(), cpu_counts.tolist())
    }
