# Idea Discovery Report

**Direction**: RL rollout阶段的负载balance，解决rollout阶段的长尾问题，减少RL训练耗时
**Date**: 2026-05-05 (v3 — CARA pivot after 4-round external review)
**Pipeline**: research-lit → idea-creator (Round 1: GA-SRollout) → experiment-bridge (sanity fail) → idea-creator (Round 2: Pairwise Streaming) → research-review (4 rounds, GPT-5.4 xhigh) → CARA

---

## Executive Summary

**Final recommended method: CARA (Cost-Aware Adaptive Rollout Acquisition)**

Three iterations of external review (GPT-5.4 xhigh) produced this convergence:

1. **GA-SRollout** (EWMA predictor + LPT) → DEPRECATED: predictor quality caps the ceiling; not generalizable
2. **Pairwise Streaming GRPO** → REFRAMED: unbiasedness claim was wrong; closest threat is P3O (2310.00212); framing shifts from "new estimator" to "adaptive stopping protocol"
3. **CARA** → RECOMMENDED: cost-aware marginal-utility stopping + balanced curation + prompt-equalized loss; defensible for MLSys/NeurIPS

**CARA's core insight**: Fixed-K grouped RLVR wastes rollout tokens in two ways — (1) dead-zone prompts (all-same reward) burn tokens with zero gradient; (2) prompts with early contrastive signal keep generating beyond their informative threshold. CARA stops each prompt when the expected gradient utility per additional token falls below a cost threshold, then trains on a balanced curated subset with equal prompt weight.

---

## Phase 1: Literature Landscape

### The Problem

GRPO-based LLM RL creates a **grouped fork-join rollout workload**:
- Each prompt needs K stochastic completions before group-relative advantages are computed
- The synchronous barrier means all K completions must finish before the training step
- Output length is heavy-tailed (p95 = 3–10× median)
- Two failure modes: (a) straggler completions block group closure; (b) dead-zone prompts burn K rollout slots for zero gradient

### What Has Been Done

| Paper | arXiv | Approach | Addresses? |
|-------|-------|----------|-----------|
| GRPO (DeepSeekMath) | 2402.03300 | Fixed K=8, group-relative advantages | Baseline |
| DAPO | 2503.14476 | Overlong reward shaping + dynamic sampling | Partially: length penalty, not stopping |
| AERO | 2602.14338 | Bayesian adaptive stopping, dead-zone rescue | Closest: complex posterior-based |
| CPPO | 2503.22342 | Completion pruning by difficulty scores | Partially: pruning, not cost-aware |
| AReaL | 2505.24298 | Fully async rollout + staleness-corrected PPO | System-level barrier; different axis |
| Infinite Sampling | 2506.22950 | Micro-groups + length-aware scheduler | Memory-focused, still group-based |
| AsyncFlow | 2507.01663 | Async pipeline RLHF | Level-1 async only |
| P3O | 2310.00212 | Online pairwise PG from relative feedback | Algorithm predecessor |

### The Gap

AERO is the closest prior work. But AERO is stratified, posterior-guided, and uses per-prompt type heuristics. **CARA's differentiator**: a single unified cost-aware stopping rule (expected gain / expected cost >= threshold) that is simpler and explicitly models rollout token cost. The "maximize informative gradient per rollout token" Pareto framing is new.

---

## Deprecated Methods

### GA-SRollout — ELIMINATED (2026-05-05)
- **Approach**: EWMA length-binned predictor + LPT scheduling
- **Result**: Sanity test — oracle 1.38× speedup, EWMA achieves 1.044× (Spearman=0.31)
- **User veto**: "预测器精度决定了整个工作的效果，不具有普适性"

### Pairwise Streaming GRPO — REFRAMED (2026-05-05)
- **Original claim**: Unbiased streaming pairwise updates eliminate fork-join barrier
- **Review finding**: Unbiasedness claim false (conditioning on discordant pairs introduces bias); P3O already does online pairwise PG; not novel as "new estimator"
- **Reframed as**: AIRS-hard ablation (heuristic baseline for CARA)

---

## Recommended Method: CARA

### One-sentence description
CARA adaptively samples each prompt only while the expected gradient utility per additional rollout token exceeds a threshold, then trains GRPO on a balanced curated subset with equal prompt weight.

### Core hypothesis
Fixed-K grouped RLVR over-spends rollout tokens on both dead-zone prompts and already-informative prompts. A cost-aware stopping rule + balanced curation achieves the same final accuracy as GRPO-K8 with ≥20% fewer rollout tokens and ≥15% less wall-clock.

### Method specification

**Parameters** (with defaults):
```
K_max = 8               # hard stop per prompt
probe_k = 2             # minimum completions before evaluating stop rule
round_k = 1             # completions generated per round
curate_per_sign = 2     # target: 2 pos + 2 neg per prompt (curated group size = 4)
beta_alpha = 1.0        # Beta posterior prior
beta_beta = 1.0
tau_rescue_per_1k = 0.75   # threshold for all-same prompts (rescue mode)
tau_balance_per_1k = 0.35  # threshold for mixed prompts seeking more pairs (balance mode)
min_cost_tokens = 64
drop_dead_prompts = True
equalize_prompt_weight = True
```

**Stopping rule** (cost-aware):
```python
def should_continue(s, cfg):
    if s.n_total >= cfg.K_max:
        return False
    p_pos = (cfg.beta_alpha + s.n_pos) / (cfg.beta_alpha + cfg.beta_beta + s.n_total)
    p_neg = 1.0 - p_pos
    cost_hat = max(cfg.min_cost_tokens, s.mean_new_tokens)

    # rescue mode: all-same, keep going only if likely to find first contrast
    if s.n_pos == 0 or s.n_neg == 0:
        p_gain = p_pos if s.n_neg > 0 else p_neg
        return 1000.0 * p_gain / cost_hat >= cfg.tau_rescue_per_1k

    # balance mode: already mixed, keep going only if likely to grow minority side
    m = min(s.n_pos, s.n_neg)
    if m >= cfg.curate_per_sign:
        return False
    p_gain = p_pos if s.n_pos < s.n_neg else p_neg
    return 1000.0 * p_gain / cost_hat >= cfg.tau_balance_per_1k
```

**Balanced curation**:
```python
def curate_balanced(samples, per_sign=2, rng):
    pos = [s for s in samples if s.reward == 1]
    neg = [s for s in samples if s.reward == 0]
    m = min(len(pos), len(neg), per_sign)
    if m == 0:
        return []  # dead prompt, drop
    return [z for pair in zip(rng.sample(pos,m), rng.sample(neg,m)) for z in pair]
```

**Prompt-equalized GRPO loss**:
```python
prompt_losses = []
for prompt in ready_prompts:
    curated = curate_balanced(prompt.samples, per_sign=cfg.curate_per_sign, rng=rng)
    if not curated: continue
    adv = group_norm([s.reward for s in curated])
    seq_losses = [ppo_loss(sample, a) for sample, a in zip(curated, adv)]
    prompt_losses.append(torch.stack(seq_losses).mean())  # mean over seqs within prompt
loss = torch.stack(prompt_losses).mean()  # mean over prompts (equalization)
```

### Novelty vs AERO
- AERO: stratified (per prompt type), Bayesian posterior, mixed-prompt-aware, complex
- CARA: single unified cost-aware rule (expected gain / expected token cost), balanced curation, simpler
- Framing: "maximize informative gradient per rollout token" Pareto frontier (new)

### Risks
- **MEDIUM**: If beta posterior doesn't track reward distribution well → bad stopping
  - Mitigation: tune tau on pilot trace
- **MEDIUM**: Balanced curation with per_sign=2 creates small group sizes → high variance
  - Mitigation: CARA-curate4 ablation (per_sign=4); check gradient variance
- **HIGH**: If CARA only matches GRPO-K4 (not K8 quality) → no paper
  - Mitigation: if pilot shows <90% mixed-prompt recall vs K4, abort

---

## Baseline Set (Required for Paper)

Main:
- `GRPO-K8`: fixed 8 completions, group-relative advantages
- `GRPO-K4`: fixed 4 completions (cheaper fixed baseline)
- `AIRS-hard`: stop on first (pos, neg) pair, curate 1+1, no cost term
- `CARA-full`: full method

Ablations:
- `CARA-no-cost`: tau_rescue=tau_balance=0, stop only when curate_per_sign reached or K_max
- `CARA-no-equalize`: raw sum loss (not per-prompt mean)
- `CARA-curate1`: per_sign=1 (pairwise)
- `CARA-curate2`: per_sign=2 (CARA default)
- `AERO-lite` (optional): probe 4, if mixed keep 2+2 and stop, if all-same continue by 2 until K_max

---

## Experiment Plan

### Pilot (2h, 1 GPU)
1. Generate K=8 completions for 512 prompts (256 GSM8K + 256 MATH), offline
2. Sort completions by length (proxy for arrival time)
3. Replay 5 schedulers offline: GRPO-K8, GRPO-K4, AIRS-hard, CARA (3 tau settings)
4. **Gate**: CARA ≥25% lower rollout tokens than K8, ≥90% mixed-prompt recall, beats K4 at similar cost

### Full run (~1 day, 4×A100)
1. **2h**: OpenRLHF patch implementation + smoke test (16-32 prompts)
2. **2-3h**: Rollout-only microbench on 1024 prompts — compare all baselines
3. **12-16h**: End-to-end training, 1 seed: GRPO-K8, GRPO-K4, CARA-full + 4 ablations

Config: Qwen2.5-Math-1.5B, max_new_tokens=1024, rollout.batch_size=64, data.max_samples=8192

### Metrics
Primary:
- Pass@1 on GSM8K, MATH
- Wall-clock to 0.9× final K8 accuracy
- Rollout tokens to 0.9× final K8 accuracy

Secondary:
- Rollout tokens per optimizer step
- Rollout tokens per nonzero-gradient prompt
- p95 prompt-close latency
- Fraction of dead prompts
- Mean curated group size
- Trainer idle fraction

### Publishable result threshold
- CARA reaches 0.9× final K8 accuracy with ≥20% fewer rollout tokens AND ≥15% less wall-clock
- Final accuracy within 0.5pp GSM8K, 1.0pp MATH vs K8
- CARA beats K4 on final accuracy at similar rollout-token cost
- Pareto plot (accuracy vs rollout tokens, accuracy vs wall-clock) shows CARA dominates GRPO-K8

---

## Next Steps

- [ ] Run offline pilot (2h, 1 GPU) to validate stopping rule behavior
- [ ] If pilot passes → patch OpenRLHF, run full experiment
- [ ] If pilot fails (only matches K4) → investigate whether tau tuning helps, or pivot to AERO-lite as main method
- [ ] `/research-refine-pipeline` after pilot for method stabilization + paper framing

---

## Review Trace

Full 4-round review saved to: `.aris/traces/research-review/2026-05-05_run01_cara.md`
