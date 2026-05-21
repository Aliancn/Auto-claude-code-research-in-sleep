# Experiment Tracker: GA-SRollout

**Status Legend**: ⬜ Pending | 🔄 Running | ✅ Pass | ❌ Fail | ⏭ Skipped

---

## Exp 1: Offline Trace Replay

| Step | Status | Notes |
|------|--------|-------|
| Data prep (prepare_math_mix.py) | ⬜ | Pending GPU access |
| Trace collection (n=8, max_new=2048) | ⬜ | ~3h on 4xA100 |
| Replay (6 schedulers) | ✅ | Synthetic trace run — see results |
| **PASS GATE**: ewma_lpt ≥1.15× speedup vs fifo | ❌ | Synthetic: 1.044× (heavy-tail) / 1.040× (realistic) |

**Synthetic Trace Results (2026-05-05)**:
- Heavy-tailed trace: ewma speedup = 1.044×, oracle speedup = 1.378×, Spearman = 0.314
- Realistic trace (R=0.64 prompt-output corr): ewma speedup = 1.040×, oracle speedup = 1.110×, Spearman = 0.507
- Decision: Need REAL vLLM traces — synthetic doesn't represent true vLLM scheduling behavior

**Critical risk**: Oracle headroom on realistic data is only 1.11×. If real data shows similar oracle headroom, the method may not be viable at 1.5B scale. Long-CoT settings (max_new_tokens=4096+) should be tested.

---

## Exp 2: Rollout Microbenchmark

| Scheduler | Status | rollout_step_time_s | speedup_vs_fifo | idle_frac |
|-----------|--------|---------------------|-----------------|-----------|
| fifo | ⬜ | — | 1.00× | — |
| rr | ⬜ | — | — | — |
| length_lpt | ⬜ | — | — | — |
| ewma_lpt | ⬜ | — | — | — |

**PASS GATE**: ewma_lpt ≥1.15× over fifo, ewma_lpt beats length_lpt by ≥5%
- Scheduler overhead <2% of step time? TBD
- Decision: PROCEED / STOP

---

## Exp 3A: grpo_fifo (baseline)

| Metric | Value |
|--------|-------|
| Status | ⬜ |
| timing/generation (median) | — |
| timing/step_total (median) | — |
| Final GSM8K Avg@8 | — |
| Final MATH Avg@8 | — |
| KL (end) | — |
| Response length p95 (end) | — |

---

## Exp 3B: grpo_ga (GA-SRollout)

| Metric | Value | vs FIFO |
|--------|-------|---------|
| Status | ⬜ | |
| timing/generation (median) | — | — |
| timing/step_total (median) | — | — |
| Final GSM8K Avg@8 | — | — |
| Final MATH Avg@8 | — | — |
| KL (end) | — | — |
| Response length p95 (end) | — | — |

**PASS GATE**: generation speedup ≥1.20×, step_total speedup ≥1.10×, accuracy drop ≤1.0pp

---

## Optional: dapo_fifo + dapo_ga

| Condition | Status | generation_speedup | accuracy |
|-----------|--------|-------------------|----------|
| dapo_fifo | ⬜ | — | — |
| dapo_ga | ⬜ | — | — |

---

## Final Verdict

- [ ] Exp 1: PASS / FAIL
- [ ] Exp 2: PASS / FAIL
- [ ] Exp 3: PASS / FAIL
- [ ] Paper viable: YES / NO
- Allowed claims: [fill after results]
