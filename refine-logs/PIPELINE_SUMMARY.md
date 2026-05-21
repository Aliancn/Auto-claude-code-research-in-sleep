# Pipeline Summary

**Problem**: GRPO-based LLM RL training creates a grouped fork-join rollout workload. Long-tail completions stall group closure and block training steps. No existing framework optimizes for group-level completion time.

**Final Method Thesis**: GA-SRollout uses online EWMA estimates of prompt group max-completion latency and LPT list scheduling to pack whole prompt groups across rollout workers, reducing straggler-driven group-closure time without changing the GRPO objective or training estimator.

**Final Verdict**: READY (pending Exp 1 pass gate)

**Date**: 2026-05-05

---

## Final Deliverables

- Proposal: `refine-logs/FINAL_PROPOSAL.md`
- Review summary: `refine-logs/REVIEW_SUMMARY.md`
- Experiment plan: `refine-logs/EXPERIMENT_PLAN.md`
- Experiment tracker: `refine-logs/EXPERIMENT_TRACKER.md`
- Idea report: `idea-stage/IDEA_REPORT.md`
- Review trace: `.aris/traces/research-review/2026-05-05_run01/`

---

## Contribution Snapshot

- **Dominant contribution**: Group-level LPT rollout scheduling for GRPO fork-join workloads, preserving the training estimator
- **Optional supporting contribution**: Adaptive Group Closure ablation (early group exit when k_min completions seen with both reward signs) — disclosed as biased, measured as curriculum-like effect
- **Explicitly rejected complexity**: Bayesian posterior correction, reward-aware early stopping, in-flight KV migration, learned difficulty models, overlong reward shaping

---

## Must-Prove Claims

1. **Rollout throughput**: GA-SRollout improves rollout tokens/sec and group closure P95/P99 by ≥1.20× over async GRPO baselines
2. **No convergence degradation**: Final MATH/GSM8K accuracy no worse than 1.0 absolute point vs vanilla GRPO
3. **Additive to DAPO**: GA scheduling is orthogonal to dynamic group filtering (both can be combined)

---

## First Runs to Launch

1. `python experiments/prepare_math_mix.py` — data prep (30 min)
2. `collect_group_trace_vllm.py` on 4×A100 — trace collection (3h)
3. `replay_group_scheduler.py` — CPU replay, check Exp 1 pass gate (5 min)

---

## Main Risks

- **Baseline risk**: AERO/DAPO already reduce wasted rollouts; pure scheduling may look incremental if the group-closure bottleneck is not cleanly isolated.
  - **Mitigation**: Show Figure 1 systems metrics clearly; position as orthogonal/additive, not competing.
- **Effect-size risk**: EWMA predictor may not have sufficient accuracy under vLLM continuous batching.
  - **Mitigation**: Exp 1 trace replay tests this cheaply first; stop before Exp 3 if Exp 1/2 fail.

---

## Next Action

Run Exp 1 (trace collection + replay) to validate scheduling headroom before committing GPU budget to end-to-end training.

```bash
# First command
python experiments/prepare_math_mix.py --gsm8k openai/gsm8k --math hendrycks/competition_math \
  --train-size 12800 --eval-size 1024 --out data/ga_srollout_mathmix --schema openrlhf_math
```

Then proceed to `/run-experiment` for full execution.
