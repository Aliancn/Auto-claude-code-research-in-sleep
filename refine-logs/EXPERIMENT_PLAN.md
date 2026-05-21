# Experiment Plan: GA-SRollout

**Proposal**: `refine-logs/FINAL_PROPOSAL.md`
**Date**: 2026-05-05
**Compute Budget**: 4–8 × A100 80GB, 1–2 days (~120–200 GPU-hours minimum)
**Framework**: OpenRLHF 0.10.2 + vLLM
**Model**: Qwen/Qwen2.5-Math-1.5B
**Dataset**: GSM8K + MATH subset (12,800 train / 1,024 eval)

---

## Common Setup

```bash
export GPUS=8  # use 4 if needed
export MODEL=Qwen/Qwen2.5-Math-1.5B
export DATA=data/ga_srollout_mathmix
export OUT=runs/ga_srollout
export N=8
export MAX_NEW=2048
export ROLLOUT_BSZ=128   # prompt groups per rollout step
export TRAIN_BSZ=1024    # ROLLOUT_BSZ * N
export VLLM_USE_V1=1

# Prepare data
python experiments/prepare_math_mix.py \
  --gsm8k openai/gsm8k \
  --math hendrycks/competition_math \
  --train-size 12800 \
  --eval-size 1024 \
  --out $DATA \
  --schema openrlhf_math
```

**Data schema**:
```json
{"prompt":[{"role":"user","content":"..."}],"label":"...","source":"gsm8k|math"}
```

---

## Experiment 1: Offline Trace Replay

**Goal**: Prove the fork-join scheduling opportunity exists before touching training.

### Step 1a: Collect Group Traces

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python experiments/collect_group_trace_vllm.py \
  --model $MODEL \
  --dataset $DATA/train.jsonl \
  --num-prompts 4096 \
  --num-vllm-workers 4 \
  --tensor-parallel-size 1 \
  --n $N \
  --max-new-tokens $MAX_NEW \
  --temperature 1.0 \
  --top-p 1.0 \
  --gpu-memory-utilization 0.70 \
  --enable-prefix-caching \
  --seed 42 \
  --out $OUT/traces/n${N}_${MAX_NEW}.jsonl
```

**Runtime**: 2–4 hours on 4 A100s

### Step 1b: Replay Schedulers

```bash
python experiments/replay_group_scheduler.py \
  --trace $OUT/traces/n${N}_${MAX_NEW}.jsonl \
  --num-workers 4 8 \
  --rollout-batch-size $ROLLOUT_BSZ \
  --schedulers fifo rr random length_lpt ewma_lpt oracle_lpt \
  --ewma-alpha 0.30 \
  --warmup-groups 512 \
  --bootstrap 1000 \
  --out $OUT/replay/n${N}_${MAX_NEW}.json
```

**Runtime**: CPU-only, ~5 minutes

### Metrics
- `step_makespan_s` — primary
- `speedup_vs_fifo`
- `worker_idle_frac`
- `worker_load_cv`
- `group_close_p50/p90/p95/p99`
- `prediction_spearman`
- `prediction_mae_s`
- `oracle_gap_closed = (speedup_ewma_lpt - 1) / (speedup_oracle_lpt - 1)`

### Pass/Fail Criterion
- **Pass**: `ewma_lpt` gives ≥1.15× mean replay makespan speedup vs FIFO AND closes ≥50% of oracle-LPT's available gain
- **Strong pass**: ≥1.25× speedup
- Note: if `prediction_spearman < 0.20` but scheduling still passes, continue — scheduling outcome matters more than predictor quality

### Decision Gate
- **If Exp 1 fails**: run 5-step diagnostic microbenchmark only, then stop or redesign the predictor. Do NOT proceed to Exp 3.
- **If Exp 1 passes**: proceed to Exp 2.

---

## Experiment 2: Rollout Microbenchmark

**Goal**: Show replay gains survive real vLLM continuous batching and worker contention.

```bash
for SCHED in fifo rr length_lpt ewma_lpt; do
  CUDA_VISIBLE_DEVICES=0,1,2,3 python experiments/rollout_microbench_vllm.py \
    --model $MODEL \
    --dataset $DATA/train.jsonl \
    --num-vllm-workers 4 \
    --tensor-parallel-size 1 \
    --scheduler $SCHED \
    --rollout-batch-size $ROLLOUT_BSZ \
    --num-steps 30 \
    --n $N \
    --max-new-tokens $MAX_NEW \
    --temperature 1.0 \
    --top-p 1.0 \
    --gpu-memory-utilization 0.70 \
    --enable-prefix-caching \
    --seed 42 \
    --trace-path $OUT/microbench/${SCHED}.jsonl \
    --out $OUT/microbench/${SCHED}.json
done
```

**Schedulers**: `fifo` | `rr` | `length_lpt` | `ewma_lpt`
**Runtime**: 3–5 hours on 4 A100s

### Metrics
- `rollout_step_time_s` — primary
- `groups_per_second`
- `completions_per_second`
- `output_tokens_per_second`
- `p95_group_close_s`
- `p99_group_close_s`
- `worker_idle_frac`
- `scheduler_overhead_ms` (must be <2% of rollout step time)
- `gpu_util_mean`

### Pass/Fail Criterion
- **Pass**: `ewma_lpt` improves median `rollout_step_time_s` ≥1.15× over `fifo`
- **Strong pass**: ≥1.25×
- **Ablation check**: `ewma_lpt` must beat `length_lpt` by ≥5% — otherwise contribution collapses to "sort by prompt length"
- Scheduler overhead must be <2% of rollout step time

---

## Experiment 3: End-to-End GRPO Training

**Goal**: Show rollout speedup translates to wall-clock training improvement without convergence degradation.

### Required Conditions
1. `grpo_fifo` — vanilla OpenRLHF GRPO (baseline)
2. `grpo_ga` — GRPO + GA-SRollout `ewma_lpt`

### Optional (if time permits)
3. `dapo_fifo` — GRPO + OpenRLHF dynamic filtering
4. `dapo_ga` — dynamic filtering + GA-SRollout

### Start Ray
```bash
ray stop -f
ray start --head --node-ip-address 0.0.0.0 --num-gpus $GPUS
```

### Common OpenRLHF Config
```bash
python3 -m openrlhf.cli.train_ppo_ray \
  --actor.model_name_or_path $MODEL \
  --reward.remote_url examples/python/math_reward_func.py \
  --data.prompt_dataset $DATA/train.jsonl \
  --data.input_key prompt \
  --data.label_key label \
  --data.apply_chat_template \
  --ref.num_nodes 1 --ref.num_gpus_per_node $GPUS \
  --actor.num_nodes 1 --actor.num_gpus_per_node $GPUS \
  --vllm.num_engines $GPUS \
  --vllm.tensor_parallel_size 1 \
  --train.colocate_all \
  --vllm.gpu_memory_utilization 0.55 \
  --vllm.enable_sleep \
  --ds.enable_sleep \
  --vllm.sync_backend nccl \
  --vllm.enforce_eager \
  --vllm.enable_prefix_caching \
  --algo.advantage.estimator group_norm \
  --algo.kl.use_loss \
  --algo.kl.estimator k2 \
  --algo.kl.init_coef 1e-5 \
  --rollout.batch_size $ROLLOUT_BSZ \
  --rollout.n_samples_per_prompt $N \
  --rollout.max_new_tokens $MAX_NEW \
  --rollout.micro_batch_size 8 \
  --train.batch_size $TRAIN_BSZ \
  --train.micro_batch_size 1 \
  --train.dynamic_batch_enable \
  --train.max_tokens_per_gpu 16384 \
  --rollout.max_tokens_per_gpu 32768 \
  --data.max_len 2560 \
  --data.max_samples 12800 \
  --train.num_episodes 2 \
  --train.max_epochs 1 \
  --train.seed 42 \
  --ds.zero_stage 3 \
  --ds.param_dtype bf16 \
  --ds.packing_samples \
  --actor.gradient_checkpointing_enable \
  --actor.adam.lr 5e-7 \
  --ckpt.save_steps -1 \
  --logger.logging_steps 1 \
  --eval.steps -1
```

### Run FIFO
```bash
# append to common config:
  --rollout.group_scheduler fifo \
  --rollout.group_scheduler_trace_path $OUT/e2e/grpo_fifo/events.jsonl \
  --ckpt.output_dir $OUT/e2e/grpo_fifo/final \
  --ckpt.path $OUT/e2e/grpo_fifo/ckpt \
  --logger.tensorboard_dir $OUT/tb/grpo_fifo
```

### Run GA-SRollout
```bash
# append to common config:
  --rollout.group_scheduler ewma_lpt \
  --rollout.group_scheduler_warmup_steps 4 \
  --rollout.group_scheduler_trace_path $OUT/e2e/grpo_ga/events.jsonl \
  --ckpt.output_dir $OUT/e2e/grpo_ga/final \
  --ckpt.path $OUT/e2e/grpo_ga/ckpt \
  --logger.tensorboard_dir $OUT/tb/grpo_ga
```

### DAPO variant (optional, add to either):
```bash
  --algo.dynamic_filtering_enable \
  --algo.dynamic_filtering_range 0.0 1.0
```

### Training Length
200 rollout steps (12800 prompts / 128 batch × 2 episodes)

### Metrics
**Systems**:
- `timing/generation` — primary
- `timing/make_experience`
- `timing/ppo_train`
- `timing/broadcast`
- `timing/step_total`
- rollout output tokens/sec
- groups/sec, worker idle fraction

**Quality**:
- reward mean/std
- KL, entropy
- response length mean/p95
- zero-std reward group fraction
- final GSM8K/MATH accuracy or Avg@8

### Final Eval
```bash
python experiments/eval_math_passk.py \
  --models $OUT/e2e/grpo_fifo/final $OUT/e2e/grpo_ga/final \
  --dataset $DATA/eval.jsonl \
  --n 8 \
  --max-new-tokens 2048 \
  --temperature 0.6 \
  --top-p 0.95 \
  --out $OUT/e2e/final_eval.json
```

### Pass/Fail Criterion
- **Required**: GA improves median `timing/generation` ≥1.20×
- **Required**: GA improves median `timing/step_total` ≥1.10×
- **Required**: Final Avg@8 / Pass@1 no worse than FIFO by more than 1.0 absolute point
- KL, entropy, response length p95 must not show collapse or runaway length growth

---

## Run Order and Decision Gates

```
Exp 1 (trace replay, n=8, max_new=2048)
  ├─ FAIL → 5-step diagnostic microbench only → STOP or redesign predictor
  └─ PASS →
       Exp 2 (rollout microbench, 4 schedulers, 30 steps each)
         ├─ FAIL → do not proceed to Exp 3
         └─ PASS →
              Exp 3A: grpo_fifo (200 steps)
              Exp 3B: grpo_ga  (200 steps, parallel with 3A)
                ├─ PASS → optional: dapo_fifo + dapo_ga
                └─ then: n=16 or max_new=4096 stress traces
```

---

## GPU-Hour Estimate

| Experiment | GPUs | Hours | GPU-Hours |
|-----------|------|-------|-----------|
| Exp 1: Trace collection | 4 | 3 | 12 |
| Exp 2: Microbenchmark | 4 | 4 | 16 |
| Exp 3A: grpo_fifo | 8 | 10 | 80 |
| Exp 3B: grpo_ga | 8 | 10 | 80 |
| Optional DAPO A/B | 8 | 20 | 160 |
| **Minimum (Exp 1-3 A/B)** | | | **~188** |
| **Full plan with DAPO** | | | **~348** |

---

## Viability Metric

**The paper is viable if**:
```
median end-to-end rollout generation speedup >= 1.20x
```
measured as `timing/generation_fifo / timing/generation_ga` during GRPO training, with final math eval no worse than 1.0 absolute point.

---

## Two Key Figures

**Figure 1** — Rollout Systems Figure:
- Closed groups/sec, p95 group completion latency, worker idle fraction
- Across: async GRPO, shared queue, DAPO filtering, GA scheduling, GA+closure (ablation), oracle
- Shows: scheduling headroom, predictor quality, overhead

**Figure 2** — Wall-Clock Training Figure:
- Panel A: Eval accuracy vs wall-clock time
- Panel B: Same metric vs generated tokens
- Separates: systems throughput improvement from sample-efficiency changes
