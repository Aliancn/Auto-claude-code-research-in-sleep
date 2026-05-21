# Initial Experiment Results: GA-SRollout

**Date**: 2026-05-05
**Plan**: refine-logs/EXPERIMENT_PLAN.md

---

## Results by Milestone

### M0: Sanity — PASSED
- Replay script runs correctly on synthetic data (1024 groups, heavy-tailed lognormal latency)
- Scheduling logic verified: FIFO/RR round-robin confirmed correct, oracle_lpt achieves 1.38× speedup
- Core simulation infrastructure is correct

### M1: Offline Trace Replay (Exp 1) — SYNTHETIC SIGNAL WEAK

**Synthetic trace results (n=4 workers, batch_size=64, 1024 groups)**:

| Scheduler    | mean_makespan | speedup_vs_fifo | idle_frac | load_cv |
|-------------|--------------|-----------------|-----------|---------|
| fifo         | 13.03s       | 1.000×          | 25.9%     | 0.310   |
| random       | 12.46s       | 1.046×          | 23.7%     | 0.273   |
| length_lpt   | 12.03s       | 1.084×          | 20.6%     | 0.243   |
| ewma_lpt     | 12.49s       | 1.044×          | 23.0%     | 0.271   |
| oracle_lpt   | 9.46s        | 1.378×          | 0.2%      | 0.001   |

**Prediction quality (ewma_lpt)**:
- prediction_spearman = 0.314 (low)
- oracle_gap_closed = 0.153 (need ≥0.50)

**Verdict**: FAIL (1.044× < 1.15× required)

**Root cause analysis**:
1. **Predictor granularity too coarse**: EWMA indexed by source bucket (gsm8k/math, 2 values). Real completion length variance is within-bucket, not between-bucket. The predictor can't distinguish hard from easy prompts within the same source.
2. **Oracle headroom exists**: oracle_lpt achieves 1.38× speedup, confirming the scheduling opportunity is real. The gap between oracle and EWMA (1.38× vs 1.04×) is the prediction quality bottleneck.
3. **length_lpt is a better baseline than expected**: Prompt token length alone achieves 1.084× — better than EWMA (1.044×). This confirms that prompt-level features (not just source bucket) carry signal.

---

## Critical Finding

**The predictor is the bottleneck, not the scheduler.**

The oracle scheduler shows 1.38× speedup is achievable. The EWMA predictor with 2-bucket granularity only achieves Spearman=0.31 correlation with true latency. Per the reviewer's prediction:

> "completion latency may be too noisy for a simple predictor under vLLM continuous batching"

**Required pivot**: Improve predictor to use finer-grained features:
1. Use prompt token count (available before generation) as the primary feature — not source bucket
2. Maintain per-prompt-length-bin EWMA (e.g., bucket by 50-token intervals: [0-50], [50-100], ...)
3. This matches what `length_lpt` does implicitly, but with learned length-to-latency mapping

---

## Revised Predictor Design

**Current (failed)**:
```
T_hat(q) = EWMA[source_bucket]  # 2 buckets, low signal
```

**Revised (to implement)**:
```
T_hat(q) = EWMA[prompt_length_bin] * estimated_output_multiplier
# where prompt_length_bin = prompt_tokens // 50
# estimated_output_multiplier = EWMA of (observed_max_tokens / prompt_tokens) per bin
```

This is equivalent to a piecewise linear latency model indexed by prompt length, which is a better predictor than source bucket alone.

**Expected improvement**: Spearman correlation should increase from 0.31 to ~0.5-0.7 with length-binned EWMA, which should close more of the oracle gap.

---

## M2: Rollout Microbenchmark — PENDING (blocked on M1 pass gate)

Per the experiment plan decision gate: do not proceed to Exp 2 until Exp 1 passes.

Action: Implement revised length-binned EWMA predictor and re-run Exp 1.

---

## M3: End-to-End GRPO — PENDING (blocked on M1 and M2)

---

## Summary

- **1/1** sanity checks completed
- **Main result**: Scheduling headroom confirmed (oracle 1.38×), but EWMA predictor too coarse (1.04×)
- **Pass gate**: FAIL (1.04× < 1.15×)
- **Ready for /auto-review-loop**: NO — requires predictor redesign first
- **Revised action**: Implement length-binned EWMA, re-run Exp 1, then proceed

---

## Realistic Trace Results (Prompt-Length Correlated)

Re-ran with a more realistic synthetic trace (prompt_tokens → output_tokens correlation R=0.64):

| Scheduler  | mean_makespan | speedup_vs_fifo | oracle_gap_closed | Spearman |
|------------|--------------|-----------------|-------------------|---------|
| fifo       | 52.51s       | 1.000×          | —                 | —       |
| length_lpt | 50.32s       | 1.044×          | —                 | —       |
| ewma_lpt   | 50.50s       | 1.040×          | 0.388             | 0.507   |
| oracle_lpt | 47.32s       | 1.110×          | 1.000             | —       |

**Key insight**: Oracle headroom is only 1.11× on realistic data (vs 1.38× on heavy-tailed synthetic). The realistic data has many groups hitting `max_new_tokens=2048` — scheduling cannot help when most groups finish at the same time.

**Updated conclusion**:
1. Scheduling headroom depends critically on max_new_tokens truncation fraction. If many groups hit max_tokens, all workers finish simultaneously regardless of scheduling.
2. On real math datasets, long-chain-of-thought RL training (max_new_tokens=4096+) should have more scheduling headroom.
3. Need to run real vLLM traces to confirm.

## Revised Action: Implement Length-Binned EWMA

**File**: `experiments/ga_srollout_scheduler.py` (updated)
- Changed from `source` bucket to `prompt_length_bin = prompt_tokens // 50`
- Both replay script and core scheduler updated

**Next step**: Run on real vLLM traces with actual max_new_tokens=2048 and the real Qwen2.5-Math-1.5B model on MATH dataset to measure true scheduling headroom.
