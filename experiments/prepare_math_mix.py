"""
Prepare GSM8K + MATH mixed dataset for GA-SRollout experiments.

Output schema per line (jsonl):
  {"prompt": [{"role": "user", "content": "..."}], "label": "...", "source": "gsm8k|math"}
"""

import argparse
import json
import random
import re
from pathlib import Path


SYSTEM_PROMPT = (
    "You are a helpful math assistant. Solve the following problem step by step. "
    "Put your final answer in \\boxed{...}."
)


def extract_gsm8k_answer(answer_str: str) -> str:
    """Extract numeric answer from GSM8K '#### X' format."""
    match = re.search(r"####\s*([\-\d,\.]+)", answer_str)
    if match:
        return match.group(1).replace(",", "").strip()
    return answer_str.strip()


def extract_math_answer(solution: str) -> str:
    """Extract boxed answer from MATH solution string (brace-balanced)."""
    idx = solution.rfind(r"\boxed{")
    if idx == -1:
        return solution.strip()
    depth = 0
    start = idx + len(r"\boxed{") - 1
    for i in range(start, len(solution)):
        if solution[i] == "{":
            depth += 1
        elif solution[i] == "}":
            depth -= 1
            if depth == 0:
                return solution[start + 1: i].strip()
    return solution.strip()


def load_gsm8k(split: str, max_samples: int) -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets")

    ds = load_dataset("openai/gsm8k", "main", split=split, trust_remote_code=True)
    items = []
    for ex in ds:
        question = ex["question"].strip()
        answer = extract_gsm8k_answer(ex["answer"])
        items.append({
            "prompt": [{"role": "user", "content": SYSTEM_PROMPT + "\n\n" + question}],
            "label": answer,
            "source": "gsm8k",
        })
        if len(items) >= max_samples:
            break
    return items


def load_math(split: str, max_samples: int, levels: list[str] | None = None) -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets")

    ds = load_dataset("hendrycks/competition_math", split=split, trust_remote_code=True)
    items = []
    for ex in ds:
        if levels and ex.get("level") not in levels:
            continue
        problem = ex["problem"].strip()
        answer = extract_math_answer(ex["solution"])
        items.append({
            "prompt": [{"role": "user", "content": SYSTEM_PROMPT + "\n\n" + problem}],
            "label": answer,
            "source": "math",
        })
        if len(items) >= max_samples:
            break
    return items


def write_jsonl(items: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Wrote {len(items)} items to {path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare GSM8K+MATH mix for GA-SRollout")
    parser.add_argument("--gsm8k", default="openai/gsm8k")
    parser.add_argument("--math", default="hendrycks/competition_math")
    parser.add_argument("--train-size", type=int, default=12800)
    parser.add_argument("--eval-size", type=int, default=1024)
    parser.add_argument("--out", type=str, default="data/ga_srollout_mathmix")
    parser.add_argument("--schema", default="openrlhf_math")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gsm8k-train-frac", type=float, default=0.5,
                        help="Fraction of train set from GSM8K (rest from MATH)")
    args = parser.parse_args()

    random.seed(args.seed)
    out = Path(args.out)

    # --- Train split ---
    gsm8k_n = int(args.train_size * args.gsm8k_train_frac)
    math_n = args.train_size - gsm8k_n

    print(f"Loading GSM8K train ({gsm8k_n} samples)...")
    gsm8k_train = load_gsm8k("train", gsm8k_n)
    print(f"Loading MATH train ({math_n} samples)...")
    math_train = load_math("train", math_n)

    train_items = gsm8k_train + math_train
    random.shuffle(train_items)
    train_items = train_items[: args.train_size]
    write_jsonl(train_items, out / "train.jsonl")

    # --- Eval split ---
    gsm8k_eval_n = args.eval_size // 2
    math_eval_n = args.eval_size - gsm8k_eval_n

    print(f"Loading GSM8K test ({gsm8k_eval_n} samples)...")
    gsm8k_eval = load_gsm8k("test", gsm8k_eval_n)
    print(f"Loading MATH test ({math_eval_n} samples)...")
    math_eval = load_math("test", math_eval_n)

    eval_items = gsm8k_eval + math_eval
    random.shuffle(eval_items)
    eval_items = eval_items[: args.eval_size]
    write_jsonl(eval_items, out / "eval.jsonl")

    # Write metadata
    meta = {
        "train_size": len(train_items),
        "eval_size": len(eval_items),
        "gsm8k_train_frac": args.gsm8k_train_frac,
        "schema": args.schema,
        "seed": args.seed,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print("Done.", json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
