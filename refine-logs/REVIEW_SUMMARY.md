# Review Summary

**Topic**: GA-SRollout — Group-Aware Rollout Scheduler for GRPO LLM RL Training
**Review Rounds**: 3 (research-review) + 2 (research-refine)
**Models**: GPT-5.4 xhigh via Codex MCP
**Threads**: `019df425-05d8-7932-86b8-4fd7af2eda8a` (review), `019df430-6a7b-7073-83d2-d0c83dacdbfc` (refine)
**Date**: 2026-05-05

---

## Evolution of the Idea

### Original Idea (SLB-Rollout)
- Three components: length predictor + request router + work-stealing
- Claimed RL-specific constraint: KV cache retention requires pre-start work-stealing only
- Framing: "apply serving scheduling to RL rollout"

### Round 1 Corrections
- **KV-cache claim is factually wrong**: PPO/GRPO stores logprobs, not KV caches. Dropped.
- **"Serving → RL" framing is incremental**: not sufficient for top venues.
- **Baseline too weak**: OpenRLHF already has async/partial rollout; static round-robin is not a real baseline.
- Score: 2/6 Weak Reject

### Round 2 Pivot
- Reframed around GRPO fork-join structure (n completions per prompt group)
- Introduced Adaptive Group Closure (early stopping with k_min, m_min)
- Reviewer: GRPO fork-join framing is novel, but early closure collides with AERO/DAPO
- Score: 3/6 Borderline

### Round 3 Refinement
- **Demoted Adaptive Group Closure** to optional ablation (it changes estimator, enters AERO/DAPO territory)
- **Dominant contribution**: pure group-level LPT scheduling using EWMA predictor on non-reward features
- Method thesis: treat GRPO prompt as fork-join job; minimize straggler-driven group-closure time without changing the estimator
- Score: 3/6 → borderline viable with real experiments

---

## Key Criticisms and Resolutions

| Criticism | Resolution |
|-----------|------------|
| KV cache claim wrong | Dropped entirely |
| "Serving transfer" too incremental | Reframed as GRPO fork-join scheduling (unique to RL) |
| Static round-robin not a real baseline | Baselines now: async GRPO, DAPO filtering, prompt-length heuristic, shared queue, oracle |
| Adaptive closure biased | Demoted to ablation; main method is unbiased scheduler |
| AERO already does adaptive rollout | GA-SRollout is now orthogonal (pure scheduling, no reward-based stopping) |
| No GPU pilot | Experiment 1 (trace replay) addresses this first |

---

## Strongest Baselines Required
1. OpenRLHF async + partial rollout (default)
2. DAPO dynamic group filtering  
3. Prompt-length heuristic (strong simple baseline)
4. Shared central queue without prediction
5. Oracle true-length scheduling (upper bound)
6. AERO-style adaptive sampling (compare orthogonality if available)

---

## Key Papers to Cite and Differentiate From
- AERO (arXiv 2602.14338) — adaptive GRPO rollout sampling
- DAPO — dynamic group filtering in veRL
- DeepSeekMath/GRPO (arXiv 2402.03300)
- OpenRLHF async rollout docs
- Response Length Perception (arXiv 2305.13144)
- TIE (arXiv 2604.00499)
- JITServe (arXiv 2504.20068)
- ORCA / Sarathi-Serve (serving context — contrast, not compare)
