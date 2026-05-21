"""
Collect per-group generation traces for offline scheduling replay (Experiment 1a).

Each output record:
  {
    "prompt_id": int,
    "prompt_tokens": int,
    "source": "gsm8k|math",
    "completions": [
      {"tokens": int, "finish_reason": "stop|length", "latency_s": float},
      ...  # n completions
    ],
    "group_max_latency_s": float,
    "group_total_tokens": int,
  }
"""

import argparse
import json
import time
from pathlib import Path


def build_sampling_params(n: int, max_new_tokens: int, temperature: float, top_p: float):
    from vllm import SamplingParams
    return SamplingParams(
        n=n,
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=["</s>", "<|im_end|>", "<|endoftext|>"],
    )


def load_prompts(dataset_path: str, num_prompts: int) -> list[dict]:
    prompts = []
    with open(dataset_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            prompts.append(json.loads(line))
            if len(prompts) >= num_prompts:
                break
    return prompts


def format_prompt(item: dict, tokenizer) -> str:
    """Apply chat template to prompt messages."""
    messages = item["prompt"]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Collect group generation traces via vLLM")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--num-prompts", type=int, default=4096)
    parser.add_argument("--num-vllm-workers", type=int, default=4,
                        help="Number of independent vLLM engines (simulating N rollout workers)")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--n", type=int, default=8, help="Completions per prompt (GRPO group size)")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.70)
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    try:
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer
    except ImportError:
        raise ImportError("pip install vllm transformers")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer from {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print(f"Loading {args.num_prompts} prompts from {args.dataset}")
    raw_prompts = load_prompts(args.dataset, args.num_prompts)

    print(f"Initializing vLLM engine (tensor_parallel={args.tensor_parallel_size})")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=args.enable_prefix_caching,
        seed=args.seed,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        n=args.n,
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    # Format all prompts
    formatted = [format_prompt(p, tokenizer) for p in raw_prompts]
    prompt_token_lens = [len(tokenizer.encode(f)) for f in formatted]

    print(f"Generating {args.n} completions for each of {len(formatted)} prompts...")
    t0 = time.perf_counter()
    outputs = llm.generate(formatted, sampling_params)
    total_time = time.perf_counter() - t0
    print(f"Generation done in {total_time:.1f}s")

    records = []
    for i, (output, raw) in enumerate(zip(outputs, raw_prompts)):
        completions = []
        for req_output in output.outputs:
            completions.append({
                "tokens": len(req_output.token_ids),
                "finish_reason": req_output.finish_reason,
                # Latency per completion not directly available from vLLM batch;
                # approximate as proportional to token count scaled by batch throughput
                "latency_s": None,  # filled in by replay simulator
            })

        group_max_tokens = max(c["tokens"] for c in completions)
        group_total_tokens = sum(c["tokens"] for c in completions)

        records.append({
            "prompt_id": i,
            "prompt_tokens": prompt_token_lens[i],
            "source": raw.get("source", "unknown"),
            "completions": completions,
            "group_max_tokens": group_max_tokens,
            "group_total_tokens": group_total_tokens,
            # Approximation: latency ~ max_tokens / throughput_rate
            # Will be calibrated in replay using actual measured token rate
            "group_max_latency_s_approx": None,
        })

    # Estimate latency from throughput: total_tokens / total_time ~ tokens/sec
    total_output_tokens = sum(
        sum(c["tokens"] for c in r["completions"]) for r in records
    )
    tokens_per_sec = total_output_tokens / total_time if total_time > 0 else 1.0
    print(f"Throughput: {tokens_per_sec:.0f} output tokens/sec")

    for r in records:
        for c in r["completions"]:
            c["latency_s"] = c["tokens"] / tokens_per_sec
        r["group_max_latency_s_approx"] = r["group_max_tokens"] / tokens_per_sec
        r["tokens_per_sec_global"] = tokens_per_sec

    with open(args.out, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"Wrote {len(records)} trace records to {args.out}")
    stats = {
        "num_prompts": len(records),
        "n_completions": args.n,
        "total_output_tokens": total_output_tokens,
        "tokens_per_sec": tokens_per_sec,
        "total_time_s": total_time,
        "group_max_tokens_p50": sorted(r["group_max_tokens"] for r in records)[len(records)//2],
        "group_max_tokens_p95": sorted(r["group_max_tokens"] for r in records)[int(len(records)*0.95)],
        "group_max_tokens_p99": sorted(r["group_max_tokens"] for r in records)[int(len(records)*0.99)],
    }
    print("Stats:", json.dumps(stats, indent=2))

    summary_path = Path(args.out).with_suffix(".stats.json")
    summary_path.write_text(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
