# Experiment Bridge Handoff

**Date**: 2026-05-05
**Status**: Implemented + Sanity-checked. Real GPU run pending.

---

## Implemented Scripts

| File | Purpose | Status |
|------|---------|--------|
| `experiments/prepare_math_mix.py` | Data prep: GSM8K + MATH → JSONL | ✅ Ready |
| `experiments/collect_group_trace_vllm.py` | Exp 1a: vLLM trace collection | ✅ Ready |
| `experiments/replay_group_scheduler.py` | Exp 1b: CPU scheduler replay (6 schedulers) | ✅ Ready (bugs fixed) |
| `experiments/rollout_microbench_vllm.py` | Exp 2: vLLM rollout microbench | ✅ Ready (single-engine) |
| `experiments/eval_math_passk.py` | Exp 3: Pass@1 + Avg@k eval | ✅ Ready (bugs fixed) |
| `experiments/ga_srollout_scheduler.py` | Core scheduler (length-binned EWMA + LPT) | ✅ Ready |
| `experiments/run_experiments.sh` | End-to-end runner (Exp 1→2→3) | ✅ Ready |
| `experiments/patch_openrlhf.md` | OpenRLHF integration guide | ✅ Ready |

## Bugs Fixed (from Codex code review)

| Severity | Bug | Fix Applied |
|----------|-----|-------------|
| CRITICAL | Worker choice leaked true latency during assignment | Fixed: pred_loads for choice, true_loads for makespan |
| CRITICAL | FIFO/RR baselines used LPT, not round-robin | Fixed: strict position % num_workers |
| CRITICAL | Random scheduler shuffled entire trace (cross-batch leak) | Fixed: shuffle within each batch only |
| CRITICAL | Exp 3 ran FIFO + GA concurrently (GPU conflict) | Fixed: sequential runs |
| CRITICAL | OpenRLHF patch not present | Added: patch_openrlhf.md + OpenRLHFSchedulerShim |
| MAJOR | oracle_gap_closed used speedup ratio, not makespan reduction | Fixed: (fifo-ewma)/(fifo-oracle) |
| MAJOR | Pass@1 used stochastic completion, not greedy | Fixed: separate temperature=0 inference |
| MAJOR | Boxed answer parser broke on nested braces | Fixed: balanced-brace extraction |
| MAJOR | EWMA predictor used 2-bucket source, not prompt length | Fixed: length-binned EWMA (50-token bins) |

## Sanity Check Results

Synthetic trace replay completed. Key finding:
- **Oracle headroom**: 1.11–1.38× depending on length distribution
- **EWMA predictor performance**: 1.04× speedup (below 1.15× gate)
- **Root cause**: Low prediction quality (Spearman=0.31–0.51) limits scheduling gain
- **Revised predictor**: Length-binned EWMA (50-token bins) implemented

## Risk Assessment Update

The synthetic experiments reveal that the 1.15× pass gate may require:
1. Real vLLM data (synthetic doesn't capture vLLM batching dynamics)
2. Long-CoT settings (max_new_tokens=4096+) for heavier scheduling headroom
3. Better predictor (length-binned EWMA is a step forward; may need prompt-content features)

## How to Run Real Experiments

**Minimum viable first run** (confirm oracle headroom exists on real data):
```bash
# Step 1: Data prep (~15 min)
python experiments/prepare_math_mix.py --out data/ga_srollout_mathmix

# Step 2: Collect real vLLM traces (~3h on 4×A100)
CUDA_VISIBLE_DEVICES=0,1,2,3 python experiments/collect_group_trace_vllm.py \
    --model Qwen/Qwen2.5-Math-1.5B \
    --dataset data/ga_srollout_mathmix/train.jsonl \
    --num-prompts 4096 --n 8 --max-new-tokens 2048 \
    --out runs/ga_srollout/traces/n8_2048.jsonl

# Step 3: Replay schedulers (~5 min, CPU)
python experiments/replay_group_scheduler.py \
    --trace runs/ga_srollout/traces/n8_2048.jsonl \
    --num-workers 4 8 --rollout-batch-size 128 \
    --out runs/ga_srollout/replay/n8_2048.json

# Check oracle headroom before proceeding
python3 -c "
import json
with open('runs/ga_srollout/replay/n8_2048.json') as f: r = json.load(f)
nw = 'workers_4'
print('Oracle speedup:', r[nw]['oracle_lpt']['speedup_vs_fifo'])
print('ewma_lpt speedup:', r[nw]['ewma_lpt']['speedup_vs_fifo'])
print('Spearman:', r[nw]['ewma_lpt']['prediction_spearman'])
"
```

**Decision gate**: If oracle speedup < 1.15×, the scheduling opportunity is insufficient — reconsider method. If oracle speedup ≥ 1.30×, proceed to Exp 2.

## Next Step

```
→ /run-experiment to deploy Exp 1 on real GPU cluster
→ /auto-review-loop after Exp 1 results are in
```
