#!/usr/bin/env bash
# GA-SRollout Experiment Runner
# Usage: bash experiments/run_experiments.sh [exp1|exp2|exp3|all]
# Requires: vllm, transformers, datasets, OpenRLHF, Ray

set -euo pipefail

export GPUS=${GPUS:-8}
export MODEL=${MODEL:-"Qwen/Qwen2.5-Math-1.5B"}
export DATA=${DATA:-"data/ga_srollout_mathmix"}
export OUT=${OUT:-"runs/ga_srollout"}
export N=${N:-8}
export MAX_NEW=${MAX_NEW:-2048}
export ROLLOUT_BSZ=${ROLLOUT_BSZ:-128}
export TRAIN_BSZ=${TRAIN_BSZ:-1024}
export VLLM_USE_V1=1

EXP=${1:-all}

echo "=== GA-SRollout Experiment Runner ==="
echo "GPUS=$GPUS MODEL=$MODEL N=$N MAX_NEW=$MAX_NEW"

mkdir -p "$OUT/traces" "$OUT/replay" "$OUT/microbench" "$OUT/e2e" "$OUT/tb"

# ---------------------------------------------------------------
# Data preparation (always run first)
# ---------------------------------------------------------------
prepare_data() {
    echo ""
    echo "--- Preparing dataset ---"
    python experiments/prepare_math_mix.py \
        --gsm8k openai/gsm8k \
        --math hendrycks/competition_math \
        --train-size 12800 \
        --eval-size 1024 \
        --out "$DATA" \
        --schema openrlhf_math
    echo "Data ready at $DATA"
}

# ---------------------------------------------------------------
# Experiment 1: Offline Trace Replay
# ---------------------------------------------------------------
run_exp1() {
    echo ""
    echo "=== Experiment 1: Offline Trace Replay ==="

    echo "Step 1a: Collecting group traces..."
    CUDA_VISIBLE_DEVICES=0,1,2,3 python experiments/collect_group_trace_vllm.py \
        --model "$MODEL" \
        --dataset "$DATA/train.jsonl" \
        --num-prompts 4096 \
        --num-vllm-workers 4 \
        --tensor-parallel-size 1 \
        --n "$N" \
        --max-new-tokens "$MAX_NEW" \
        --temperature 1.0 \
        --top-p 1.0 \
        --gpu-memory-utilization 0.70 \
        --enable-prefix-caching \
        --seed 42 \
        --out "$OUT/traces/n${N}_${MAX_NEW}.jsonl"

    echo "Step 1b: Replaying schedulers..."
    python experiments/replay_group_scheduler.py \
        --trace "$OUT/traces/n${N}_${MAX_NEW}.jsonl" \
        --num-workers 4 8 \
        --rollout-batch-size "$ROLLOUT_BSZ" \
        --schedulers fifo rr random length_lpt ewma_lpt oracle_lpt \
        --ewma-alpha 0.30 \
        --warmup-groups 512 \
        --bootstrap 1000 \
        --out "$OUT/replay/n${N}_${MAX_NEW}.json"

    echo "Exp 1 complete. Results: $OUT/replay/n${N}_${MAX_NEW}.json"

    # Check pass gate
    python3 -c "
import json, sys
with open('$OUT/replay/n${N}_${MAX_NEW}.json') as f:
    r = json.load(f)
nw_key = list(r.keys())[0]
ewma = r[nw_key].get('ewma_lpt', {})
speedup = ewma.get('speedup_vs_fifo', 0)
gap = ewma.get('oracle_gap_closed', 0)
passed = speedup >= 1.15 and gap >= 0.5
print(f'  ewma_lpt speedup_vs_fifo = {speedup:.3f}x')
print(f'  oracle_gap_closed = {gap:.3f}')
print(f'  PASS GATE: {\"PASS\" if passed else \"FAIL\"} (need >=1.15x AND >=0.50 oracle gap)')
sys.exit(0 if passed else 1)
"
}

# ---------------------------------------------------------------
# Experiment 2: Rollout Microbenchmark
# ---------------------------------------------------------------
run_exp2() {
    echo ""
    echo "=== Experiment 2: Rollout Microbenchmark ==="

    for SCHED in fifo rr length_lpt ewma_lpt; do
        echo "  Running scheduler: $SCHED"
        CUDA_VISIBLE_DEVICES=0,1,2,3 python experiments/rollout_microbench_vllm.py \
            --model "$MODEL" \
            --dataset "$DATA/train.jsonl" \
            --num-vllm-workers 4 \
            --tensor-parallel-size 1 \
            --scheduler "$SCHED" \
            --rollout-batch-size "$ROLLOUT_BSZ" \
            --num-steps 30 \
            --n "$N" \
            --max-new-tokens "$MAX_NEW" \
            --temperature 1.0 \
            --top-p 1.0 \
            --gpu-memory-utilization 0.70 \
            --enable-prefix-caching \
            --seed 42 \
            --trace-path "$OUT/microbench/${SCHED}_trace.jsonl" \
            --out "$OUT/microbench/${SCHED}.json"
    done

    # Pass/fail check
    python3 -c "
import json, statistics
results = {}
for sched in ['fifo', 'rr', 'length_lpt', 'ewma_lpt']:
    with open(f'$OUT/microbench/{sched}.json') as f:
        results[sched] = json.load(f)

fifo_med = results['fifo']['median_step_time_s']
ewma_med = results['ewma_lpt']['median_step_time_s']
length_med = results['length_lpt']['median_step_time_s']
speedup_vs_fifo = fifo_med / ewma_med if ewma_med > 0 else 0
speedup_vs_length = length_med / ewma_med if ewma_med > 0 else 0
overhead_pct = results['ewma_lpt'].get('scheduler_overhead_pct', 0)

passed = speedup_vs_fifo >= 1.15 and speedup_vs_length >= 1.05 and overhead_pct < 2.0
print()
print('Microbenchmark Results:')
print(f'  FIFO median step time:     {fifo_med:.3f}s')
print(f'  ewma_lpt median step time: {ewma_med:.3f}s')
print(f'  Speedup vs FIFO:           {speedup_vs_fifo:.3f}x  (need >=1.15x)')
print(f'  Speedup vs length_lpt:     {speedup_vs_length:.3f}x  (need >=1.05x)')
print(f'  Scheduler overhead:        {overhead_pct:.2f}%   (need <2%)')
print(f'  PASS GATE: {\"PASS\" if passed else \"FAIL\"}')
"

    echo "Exp 2 complete."
}

# ---------------------------------------------------------------
# Experiment 3: End-to-End GRPO Training
# ---------------------------------------------------------------
run_exp3() {
    echo ""
    echo "=== Experiment 3: End-to-End GRPO Training ==="

    # Start Ray
    ray stop -f 2>/dev/null || true
    ray start --head --node-ip-address 0.0.0.0 --num-gpus "$GPUS"
    sleep 5

    COMMON_ARGS=(
        --actor.model_name_or_path "$MODEL"
        --reward.remote_url examples/python/math_reward_func.py
        --data.prompt_dataset "$DATA/train.jsonl"
        --data.input_key prompt
        --data.label_key label
        --data.apply_chat_template
        --ref.num_nodes 1 --ref.num_gpus_per_node "$GPUS"
        --actor.num_nodes 1 --actor.num_gpus_per_node "$GPUS"
        --vllm.num_engines "$GPUS"
        --vllm.tensor_parallel_size 1
        --train.colocate_all
        --vllm.gpu_memory_utilization 0.55
        --vllm.enable_sleep
        --ds.enable_sleep
        --vllm.sync_backend nccl
        --vllm.enforce_eager
        --vllm.enable_prefix_caching
        --algo.advantage.estimator group_norm
        --algo.kl.use_loss
        --algo.kl.estimator k2
        --algo.kl.init_coef 1e-5
        --rollout.batch_size "$ROLLOUT_BSZ"
        --rollout.n_samples_per_prompt "$N"
        --rollout.max_new_tokens "$MAX_NEW"
        --rollout.micro_batch_size 8
        --train.batch_size "$TRAIN_BSZ"
        --train.micro_batch_size 1
        --train.dynamic_batch_enable
        --train.max_tokens_per_gpu 16384
        --rollout.max_tokens_per_gpu 32768
        --data.max_len 2560
        --data.max_samples 12800
        --train.num_episodes 2
        --train.max_epochs 1
        --train.seed 42
        --ds.zero_stage 3
        --ds.param_dtype bf16
        --ds.packing_samples
        --actor.gradient_checkpointing_enable
        --actor.adam.lr 5e-7
        --ckpt.save_steps -1
        --logger.logging_steps 1
        --eval.steps -1
    )

    # Run sequentially (each job claims all GPUs; concurrent runs would conflict).
    # For truly parallel runs, launch on separate machines with disjoint GPU sets.

    echo "Running grpo_fifo baseline (step 1/2)..."
    ray stop -f 2>/dev/null || true
    ray start --head --node-ip-address 0.0.0.0 --num-gpus "$GPUS"
    sleep 5

    python3 -m openrlhf.cli.train_ppo_ray \
        "${COMMON_ARGS[@]}" \
        --rollout.group_scheduler fifo \
        --rollout.group_scheduler_trace_path "$OUT/e2e/grpo_fifo/events.jsonl" \
        --ckpt.output_dir "$OUT/e2e/grpo_fifo/final" \
        --ckpt.path "$OUT/e2e/grpo_fifo/ckpt" \
        --logger.tensorboard_dir "$OUT/tb/grpo_fifo"

    echo "Running grpo_ga (GA-SRollout) (step 2/2)..."
    ray stop -f 2>/dev/null || true
    ray start --head --node-ip-address 0.0.0.0 --num-gpus "$GPUS"
    sleep 5

    python3 -m openrlhf.cli.train_ppo_ray \
        "${COMMON_ARGS[@]}" \
        --rollout.group_scheduler ewma_lpt \
        --rollout.group_scheduler_warmup_steps 4 \
        --rollout.group_scheduler_trace_path "$OUT/e2e/grpo_ga/events.jsonl" \
        --ckpt.output_dir "$OUT/e2e/grpo_ga/final" \
        --ckpt.path "$OUT/e2e/grpo_ga/ckpt" \
        --logger.tensorboard_dir "$OUT/tb/grpo_ga"

    echo ""
    echo "Running final evaluation..."
    python experiments/eval_math_passk.py \
        --models "$OUT/e2e/grpo_fifo/final" "$OUT/e2e/grpo_ga/final" \
        --dataset "$DATA/eval.jsonl" \
        --n 8 \
        --max-new-tokens 2048 \
        --temperature 0.6 \
        --top-p 0.95 \
        --out "$OUT/e2e/final_eval.json"

    echo "Exp 3 complete. Results: $OUT/e2e/final_eval.json"
}

# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
case "$EXP" in
    data)   prepare_data ;;
    exp1)   prepare_data && run_exp1 ;;
    exp2)   run_exp2 ;;
    exp3)   run_exp3 ;;
    all)
        prepare_data
        run_exp1 && echo "Exp 1 PASSED — proceeding to Exp 2"
        run_exp2 && echo "Exp 2 PASSED — proceeding to Exp 3"
        run_exp3
        ;;
    *)
        echo "Usage: $0 [data|exp1|exp2|exp3|all]"
        exit 1
        ;;
esac
