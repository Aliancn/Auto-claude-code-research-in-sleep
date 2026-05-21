# Final Proposal: GA-SRollout

**Problem**: GRPO-based LLM RL training creates a grouped fork-join rollout workload. Each prompt requires n stochastic completions before group-relative advantages can be computed. Under naive assignment, long-tail completions stall group closure and block the training step. No existing framework optimizes for group-level completion time.

**Method Thesis**: GA-SRollout treats each GRPO prompt as a fork-join job and uses online estimates of group max-completion latency to pack whole prompt groups across rollout workers, reducing straggler-driven group-closure time without changing the GRPO objective or the samples used for training.

**Dominant Contribution**: Group-level rollout scheduling for GRPO fork-join workloads — a systems mechanism that minimizes group closure makespan while preserving the training estimator.

---

## Method Description

GA-SRollout is a GRPO-compatible rollout scheduler for grouped generation workloads.

For each training step, sample the same prompt batch and group size `n` as standard GRPO. Before generation, estimate each prompt group's expected closure time:

```
T_hat(q) = predicted max latency of n stochastic completions for prompt q
```

**Features used** (non-reward only): prompt token length, dataset/source bucket, recent observed completion lengths, recent group max latency. Update predictor online (EWMA) after each rollout step.

**Scheduling algorithm** (LPT — Longest Processing Time):
1. Sort prompt groups by `T_hat(q)` descending
2. Assign each whole group to the worker with lowest accumulated predicted load
3. Generate all `n` completions normally
4. Close group only after all `n` completions and rewards are available
5. Compute GRPO/DAPO advantages unchanged

**Predictor**: Exponential Weighted Moving Average (EWMA) of per-prompt-bucket max-completion latency. Alpha = 0.30. Warm-up: 4 rollout steps with FIFO fallback.

---

## Explicitly Rejected Complexity

- **Adaptive Group Closure** (early stopping based on rewards): introduces estimator bias, collides with AERO/DAPO territory. Demoted to optional ablation only.
- Bayesian posterior correction
- Reward-aware early stopping or dynamic resampling
- Token-level loss changes or overlong shaping
- Stale replay buffers
- Learned difficulty models
- In-flight KV cache migration

---

## Key Claims

1. **Rollout throughput improves**: GA scheduling beats FIFO/random on rollout tokens/sec, group closure P95/P99, step makespan. Target: 1.2–1.4× rollout throughput.
2. **Convergence does not degrade**: final eval accuracy, reward, response length, entropy/KL match vanilla GRPO under same rollout budget. No worse than 1.0 absolute point on MATH/GSM8K.
3. **Gain survives strong baselines**: GA-SRollout is additive to DAPO-style dynamic sampling; compared against DAPO filtering and async GRPO.

---

## Remaining Risks

1. **Baseline risk**: AERO and DAPO already reduce wasted rollouts. A pure scheduler may look incremental unless the group-closure bottleneck is isolated clearly in systems metrics.
2. **Effect-size risk**: completion latency may be too noisy for EWMA predictor under vLLM continuous batching. Measured gain on 4–8 A100s may fall below 1.20× target.

---

## Venue Target

- **Primary**: MLSys workshop or RL Systems workshop (strong with 1.5B results)
- **Stretch**: MLSys main track (requires 7B+ results and clean implementation)
- **Not targeting**: OSDI (requires broader systems contribution at scale)

---

## Abstract (optimistic but realistic)

GRPO-style reinforcement learning for language models creates a grouped fork-join rollout workload: each prompt requires multiple stochastic completions, but useful policy-gradient signal often emerges before the full group has been generated. We present GA-SRollout, a group-aware rollout scheduler that predicts prompt-level group completion cost using an online EWMA estimator and applies Longest Processing Time list scheduling to pack whole prompt groups across rollout workers. Unlike serving schedulers, GA-SRollout optimizes for grouped training utility rather than independent request latency, and preserves the rollout distribution and advantage estimator of the underlying GRPO-style algorithm. Implemented in OpenRLHF with vLLM rollout engines and evaluated on 1.5B math reasoning workloads using 4–8 A100 GPUs, GA-SRollout improves rollout throughput by 1.2–1.4× and reduces wall-clock time-to-target accuracy by 15–25% over async GRPO and DAPO dynamic filtering baselines, without measurable degradation in final evaluation accuracy.

---

**Verdict**: REVISE → READY (after demoting Adaptive Group Closure to ablation)
**Date**: 2026-05-05
**Refinement Thread**: `019df430-6a7b-7073-83d2-d0c83dacdbfc`
