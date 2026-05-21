"""
Evaluate checkpointed models on math datasets (GSM8K + MATH subset).

Computes:
  - Pass@1 (greedy / temperature=0)
  - Avg@k (average accuracy over k samples at temperature 0.6)
  - pass_any@k (1 if any of k completions is correct)

Ground truth comes from the dataset's label field — NOT from any model output.

Output JSON per model:
  {
    "model_path": str,
    "dataset": str,
    "n": int,
    "pass_at_1": float,
    "avg_at_k": float,
    "pass_any_at_k": float,
    "by_source": {"gsm8k": {...}, "math": {...}},
    "num_examples": int,
  }
"""

import argparse
import json
import re
import statistics
from pathlib import Path


def load_eval_set(dataset_path: str) -> list[dict]:
    items = []
    with open(dataset_path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def normalize_answer(answer: str) -> str:
    """Normalize math answer string for comparison."""
    answer = answer.strip()
    # Remove LaTeX formatting
    answer = re.sub(r"\\boxed\{([^}]*)\}", r"\1", answer)
    answer = re.sub(r"\$", "", answer)
    answer = re.sub(r"\s+", " ", answer).strip()
    # Normalize fractions: \frac{a}{b} -> a/b
    answer = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"\1/\2", answer)
    # Remove commas in numbers
    answer = re.sub(r"(\d),(\d)", r"\1\2", answer)
    return answer.lower()


def _extract_last_boxed(text: str) -> str | None:
    """Extract last \\boxed{...} with balanced brace handling."""
    idx = text.rfind(r"\boxed{")
    if idx == -1:
        return None
    depth = 0
    start = idx + len(r"\boxed{") - 1  # position of '{'
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1: i]
    return None


def extract_boxed_answer(text: str) -> str | None:
    """Extract the last \\boxed{} expression from model output."""
    raw = _extract_last_boxed(text)
    if raw is not None:
        return normalize_answer(raw)
    # Fallback: look for "the answer is X" patterns
    match = re.search(r"(?:the answer is|answer:\s*)\**\s*([\-\d\./]+)", text, re.IGNORECASE)
    if match:
        return normalize_answer(match.group(1))
    return None


def is_correct(prediction: str | None, label: str) -> bool:
    if prediction is None:
        return False
    norm_pred = normalize_answer(prediction)
    norm_label = normalize_answer(label)
    return norm_pred == norm_label


def run_inference_vllm(
    model_path: str,
    items: list[dict],
    n: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> tuple[list[str], list[list[str]]]:
    """
    Run vLLM inference.

    Returns:
      greedy_completions: list[str]  — one greedy (temp=0) completion per item (for Pass@1)
      sample_completions: list[list[str]]  — n sampled completions per item (for Avg@k)
    """
    try:
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer
    except ImportError:
        raise ImportError("pip install vllm transformers")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    llm = LLM(model=model_path, trust_remote_code=True, gpu_memory_utilization=0.70)

    prompts = [
        tokenizer.apply_chat_template(
            item["prompt"], tokenize=False, add_generation_prompt=True
        )
        for item in items
    ]

    # Greedy pass (temperature=0) for Pass@1
    greedy_params = SamplingParams(n=1, max_tokens=max_new_tokens, temperature=0.0)
    greedy_outputs = llm.generate(prompts, greedy_params)
    greedy_completions = [out.outputs[0].text for out in greedy_outputs]

    # Sampled pass for Avg@k / PassAny@k
    sample_params = SamplingParams(n=n, max_tokens=max_new_tokens, temperature=temperature, top_p=top_p)
    sample_outputs = llm.generate(prompts, sample_params)
    sample_completions = [[req.text for req in out.outputs] for out in sample_outputs]

    return greedy_completions, sample_completions


def compute_metrics(
    items: list[dict],
    greedy_completions: list[str],
    sample_completions: list[list[str]],
    n: int,
) -> dict:
    """Compute pass@1, avg@k, pass_any@k. Ground truth ALWAYS from dataset labels."""
    pass1_scores = []
    avgk_scores = []
    pass_any_scores = []
    by_source: dict[str, dict] = {}

    for item, greedy_c, item_completions in zip(items, greedy_completions, sample_completions):
        label = item["label"]   # ground truth from dataset — NEVER from model
        source = item.get("source", "unknown")

        # Pass@1: greedy (temperature=0) completion
        pred_1 = extract_boxed_answer(greedy_c)
        p1 = is_correct(pred_1, label)
        pass1_scores.append(float(p1))

        # Avg@k: fraction of k completions that are correct
        correct_count = sum(
            is_correct(extract_boxed_answer(c), label) for c in item_completions[:n]
        )
        avg_k = correct_count / len(item_completions[:n])
        avgk_scores.append(avg_k)

        # Pass_any@k: 1 if any completion is correct
        any_correct = any(is_correct(extract_boxed_answer(c), label) for c in item_completions[:n])
        pass_any_scores.append(float(any_correct))

        if source not in by_source:
            by_source[source] = {"pass1": [], "avgk": [], "pass_any": []}
        by_source[source]["pass1"].append(float(p1))
        by_source[source]["avgk"].append(avg_k)
        by_source[source]["pass_any"].append(float(any_correct))

    return {
        "pass_at_1": statistics.mean(pass1_scores) * 100,
        "avg_at_k": statistics.mean(avgk_scores) * 100,
        "pass_any_at_k": statistics.mean(pass_any_scores) * 100,
        "by_source": {
            src: {
                "pass_at_1": statistics.mean(v["pass1"]) * 100,
                "avg_at_k": statistics.mean(v["avgk"]) * 100,
                "pass_any_at_k": statistics.mean(v["pass_any"]) * 100,
                "n": len(v["pass1"]),
            }
            for src, v in by_source.items()
        },
        "num_examples": len(items),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate models on math datasets")
    parser.add_argument("--models", nargs="+", required=True,
                        help="Paths to model checkpoints to evaluate")
    parser.add_argument("--dataset", required=True, help="Path to eval.jsonl")
    parser.add_argument("--n", type=int, default=8, help="Samples per prompt")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    items = load_eval_set(args.dataset)
    print(f"Loaded {len(items)} eval examples from {args.dataset}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for model_path in args.models:
        print(f"\nEvaluating: {model_path}")
        greedy_completions, sample_completions = run_inference_vllm(
            model_path, items, args.n, args.max_new_tokens, args.temperature, args.top_p
        )
        metrics = compute_metrics(items, greedy_completions, sample_completions, args.n)
        metrics["model_path"] = model_path
        metrics["dataset"] = args.dataset
        metrics["n"] = args.n
        all_results[model_path] = metrics

        print(f"  Pass@1:      {metrics['pass_at_1']:.1f}%")
        print(f"  Avg@{args.n}:      {metrics['avg_at_k']:.1f}%")
        print(f"  PassAny@{args.n}: {metrics['pass_any_at_k']:.1f}%")
        for src, sv in metrics["by_source"].items():
            print(f"  [{src}] Pass@1={sv['pass_at_1']:.1f}% Avg@k={sv['avg_at_k']:.1f}% (n={sv['n']})")

    Path(args.out).write_text(json.dumps(all_results, indent=2))
    print(f"\nAll results saved to {args.out}")

    # Print comparison table if multiple models
    if len(args.models) > 1:
        print("\n--- Comparison ---")
        print(f"{'Model':<50} {'Pass@1':>8} {'Avg@k':>8} {'PassAny@k':>10}")
        for m, r in all_results.items():
            name = Path(m).name
            print(f"{name:<50} {r['pass_at_1']:>7.1f}% {r['avg_at_k']:>7.1f}% {r['pass_any_at_k']:>9.1f}%")


if __name__ == "__main__":
    main()
