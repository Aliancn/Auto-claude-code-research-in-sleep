# OpenRLHF Integration: GA-SRollout Scheduler Patch

This document describes how to patch `openrlhf` to support `--rollout.group_scheduler ewma_lpt`.

## Overview

The patch adds a `GAGroupScheduler` integration layer to OpenRLHF's vLLM rollout
dispatch code. The scheduler runs between prompt batch preparation and vLLM generation.

## Files to Modify

### 1. `openrlhf/trainer/ppo_trainer.py` or `openrlhf/trainer/ray/vllm_worker_wrap.py`

Find the rollout step function (often `_generate_experience` or `_rollout_step`).
The relevant call looks like:

```python
# Before patch: static round-robin or batch-level dispatch
outputs = self.actor_engine.generate(prompts, sampling_params)
```

### 2. Patch Instructions

**Step 1**: Copy `experiments/ga_srollout_scheduler.py` to `openrlhf/ga_srollout_scheduler.py`.

**Step 2**: In the rollout trainer, import and initialize:

```python
from openrlhf.ga_srollout_scheduler import GAGroupScheduler, SchedulerConfig, GroupMeta

# In __init__:
scheduler_type = getattr(self.strategy.args.rollout, 'group_scheduler', 'fifo')
warmup_steps = getattr(self.strategy.args.rollout, 'group_scheduler_warmup_steps', 4)
if scheduler_type == 'ewma_lpt':
    self._ga_scheduler = GAGroupScheduler(SchedulerConfig(
        num_workers=self.strategy.args.vllm.num_engines,
        ewma_alpha=0.30,
        warmup_steps=warmup_steps,
    ))
else:
    self._ga_scheduler = None
```

**Step 3**: In the rollout step, intercept prompt dispatch:

```python
def _dispatch_rollout_batch(self, prompt_batch: list[dict], source_tags: list[str]) -> list[int]:
    """Return worker assignment for each prompt group."""
    if self._ga_scheduler is None:
        # Fallback: round-robin
        nw = self.strategy.args.vllm.num_engines
        return [i % nw for i in range(len(prompt_batch))]

    groups = [
        GroupMeta(
            group_id=i,
            prompt_tokens=len(item.get('input_ids', [])),
            source=source_tags[i] if source_tags else 'unknown',
        )
        for i, item in enumerate(prompt_batch)
    ]
    return self._ga_scheduler.assign(groups)

def _update_scheduler(self, groups: list[GroupMeta], step_timing: dict[int, float]) -> None:
    """Update EWMA predictor with observed per-group latencies."""
    if self._ga_scheduler is None:
        return
    latencies = [step_timing.get(g.group_id, 0.0) for g in groups]
    self._ga_scheduler.update(groups, latencies)
```

**Step 4**: Wrap the vLLM per-worker generate calls to record per-group timing:

```python
import time

# Replace:
#   outputs = engine.generate(worker_prompts[w], sampling_params)
# With:
t0 = time.perf_counter()
outputs = engine.generate(worker_prompts[w], sampling_params)
elapsed = time.perf_counter() - t0
# Record per-group latency: elapsed / num_groups_on_worker
for group_idx in groups_on_worker[w]:
    step_timing[group_idx] = elapsed / len(groups_on_worker[w])
```

**Step 5**: Call `_update_scheduler` after all workers complete:

```python
self._update_scheduler(groups_meta, step_timing)
```

## Adding CLI Args

In `openrlhf/utils/args.py`, add to the rollout argument group:

```python
group.add_argument('--rollout.group_scheduler',
    type=str, default='fifo',
    choices=['fifo', 'rr', 'length_lpt', 'ewma_lpt'],
    help='Group-level rollout scheduler for GRPO fork-join workloads')
group.add_argument('--rollout.group_scheduler_warmup_steps',
    type=int, default=4,
    help='Steps to use FIFO before EWMA predictor is warm')
group.add_argument('--rollout.group_scheduler_trace_path',
    type=str, default=None,
    help='Path to write per-step scheduling events (for analysis)')
```

## Logging

Optionally log per-step scheduler state to the trace path:

```python
if self.strategy.args.rollout.group_scheduler_trace_path:
    with open(self.strategy.args.rollout.group_scheduler_trace_path, 'a') as f:
        import json
        f.write(json.dumps({
            'step': global_step,
            'scheduler_state': self._ga_scheduler.state_dict() if self._ga_scheduler else None,
            'step_timing': step_timing,
        }) + '\n')
```

## Verification

After patching, run the sanity check:

```bash
# Verify scheduler arg is accepted
python3 -m openrlhf.cli.train_ppo_ray --help | grep group_scheduler

# Run 5-step test
python3 -m openrlhf.cli.train_ppo_ray \
  --rollout.group_scheduler ewma_lpt \
  --rollout.group_scheduler_warmup_steps 4 \
  --train.num_episodes 1 \
  --data.max_samples 640 \
  --rollout.batch_size 16 \
  ...  # rest of config
```

Expected: training runs 5 steps without error, `events.jsonl` is written with
non-zero timing values.
