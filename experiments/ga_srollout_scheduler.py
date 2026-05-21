"""
GA-SRollout Group-Aware Scheduler — standalone module.

This implements the core scheduling algorithm to be patched into OpenRLHF's
rollout dispatch layer. See experiments/patch_openrlhf.md for integration guide.

Usage (standalone):
  from ga_srollout_scheduler import GAGroupScheduler

  scheduler = GAGroupScheduler(num_workers=8, ewma_alpha=0.30, warmup_steps=4)
  assignment = scheduler.assign(groups_metadata)
  # ... run rollout ...
  scheduler.update(groups_metadata, observed_latencies)
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GroupMeta:
    """Metadata for a single GRPO prompt group."""
    group_id: int
    prompt_tokens: int
    source: str = "unknown"
    extra: dict = field(default_factory=dict)


@dataclass
class SchedulerConfig:
    num_workers: int = 8
    ewma_alpha: float = 0.30
    warmup_steps: int = 4


class GAGroupScheduler:
    """
    Group-Aware Scheduler for GRPO rollout workloads.

    Algorithm:
      1. For each prompt group, predict group max-completion latency using
         per-source EWMA estimates (non-reward features only).
      2. Sort groups descending by predicted latency (LPT).
      3. Assign each group to the worker with the lowest accumulated
         predicted load (greedy bin-packing).
      4. After each rollout step, update EWMA with observed group latencies.

    During warm-up (first `warmup_steps` steps), falls back to round-robin.
    """

    def __init__(self, config: SchedulerConfig | None = None, **kwargs):
        if config is None:
            config = SchedulerConfig(**kwargs)
        self.config = config
        self.step_count: int = 0
        # EWMA state: per-source bucket -> estimate
        self._ewma: dict[str, float] = {}
        # Per-source observation counts
        self._counts: dict[str, int] = defaultdict(int)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    # Bucket prompt lengths into bins for finer-grained EWMA
    _LENGTH_BIN_SIZE = 50  # tokens per bin

    def _bin_key(self, prompt_tokens: int) -> str:
        """Bucket key combining source + prompt-length bin for finer-grained EWMA."""
        bin_idx = prompt_tokens // self._LENGTH_BIN_SIZE
        return f"bin_{bin_idx}"

    def _predict(self, group: GroupMeta) -> float:
        """Predict group max-completion latency (seconds)."""
        key = self._bin_key(group.prompt_tokens)
        if key not in self._ewma:
            # Cold-start: linear proxy from prompt token count (1ms/token)
            return group.prompt_tokens * 1e-3
        return self._ewma[key]

    def _update_ewma(self, source: str, observed_latency: float, prompt_tokens: int = 0) -> None:
        alpha = self.config.ewma_alpha
        key = self._bin_key(prompt_tokens) if prompt_tokens > 0 else source
        if key not in self._ewma:
            self._ewma[key] = observed_latency
        else:
            self._ewma[key] = alpha * observed_latency + (1 - alpha) * self._ewma[key]
        self._counts[key] += 1

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def assign(self, groups: list[GroupMeta]) -> list[int]:
        """
        Assign each group to a worker.

        Returns a list of worker IDs (0 .. num_workers-1), one per group.
        """
        nw = self.config.num_workers

        if self.step_count < self.config.warmup_steps:
            # Warmup: round-robin
            return [i % nw for i in range(len(groups))]

        # Sort descending by predicted latency (LPT)
        order = sorted(range(len(groups)), key=lambda i: self._predict(groups[i]), reverse=True)

        worker_loads = [0.0] * nw
        assignment = [0] * len(groups)

        for idx in order:
            g = groups[idx]
            w = min(range(nw), key=lambda i: worker_loads[i])
            worker_loads[w] += self._predict(g)
            assignment[idx] = w

        return assignment

    def update(self, groups: list[GroupMeta], observed_latencies: list[float]) -> None:
        """
        Update EWMA estimates with observed per-group latencies.

        `observed_latencies[i]` is the wall-clock time for group i's
        last completion on its assigned worker.

        Call this once per rollout step, after all completions are collected.
        """
        assert len(groups) == len(observed_latencies), \
            f"groups ({len(groups)}) and latencies ({len(observed_latencies)}) must match"

        for g, lat in zip(groups, observed_latencies):
            self._update_ewma(g.source, lat, prompt_tokens=g.prompt_tokens)

        self.step_count += 1

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "step_count": self.step_count,
            "ewma": dict(self._ewma),
            "counts": dict(self._counts),
            "config": {
                "num_workers": self.config.num_workers,
                "ewma_alpha": self.config.ewma_alpha,
                "warmup_steps": self.config.warmup_steps,
            },
        }

    def load_state_dict(self, state: dict) -> None:
        self.step_count = state["step_count"]
        self._ewma = dict(state["ewma"])
        self._counts = defaultdict(int, state["counts"])

    def summary(self) -> str:
        lines = [f"GAGroupScheduler (step={self.step_count}, nw={self.config.num_workers})"]
        for src, est in sorted(self._ewma.items()):
            lines.append(f"  [{src}] ewma_lat={est*1000:.1f}ms (n={self._counts[src]})")
        return "\n".join(lines)


# ------------------------------------------------------------------
# OpenRLHF integration shim
# ------------------------------------------------------------------

class OpenRLHFSchedulerShim:
    """
    Thin adapter between OpenRLHF's rollout data format and GAGroupScheduler.

    OpenRLHF rollout data is a list of dicts with keys like:
      {"input_ids": [...], "attention_mask": [...], "extra_info": {...}}

    This shim extracts the necessary features and returns group assignments.

    Patch point: in openrlhf/trainer/ppo_trainer.py or vllm_engine.py,
    replace the current `group_assignment = round_robin(groups)` call with:

      shim = OpenRLHFSchedulerShim(scheduler)
      group_assignment = shim.assign_from_rollout_data(rollout_batch, source_tags)
    """

    def __init__(self, scheduler: GAGroupScheduler):
        self.scheduler = scheduler

    def assign_from_rollout_data(
        self,
        rollout_batch: list[dict],
        source_tags: list[str] | None = None,
    ) -> list[int]:
        """
        Build GroupMeta from OpenRLHF rollout batch and return worker assignments.
        """
        if source_tags is None:
            source_tags = ["unknown"] * len(rollout_batch)

        groups = []
        for i, (item, src) in enumerate(zip(rollout_batch, source_tags)):
            input_ids = item.get("input_ids", [])
            prompt_tokens = len(input_ids) if isinstance(input_ids, list) else int(input_ids.shape[-1])
            groups.append(GroupMeta(group_id=i, prompt_tokens=prompt_tokens, source=src))

        return self.scheduler.assign(groups)

    def update_from_results(
        self,
        groups: list[GroupMeta],
        timing_info: list[float],
    ) -> None:
        self.scheduler.update(groups, timing_info)
