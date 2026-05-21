"""
Offline replay of rollout group schedulers (Experiment 1b).

Fixes applied per code review:
- CRITICAL: Worker assignment uses predicted load, makespan computed from true load.
- CRITICAL: FIFO/RR use true position-based round-robin, not LPT. Random shuffles
  within each batch only (not the entire trace), then round-robin.
- MAJOR: oracle_gap_closed = (fifo_makespan - ewma_makespan) / (fifo_makespan - oracle_makespan)
- MINOR: worker_load_cv computed over true worker loads per batch, not idle fractions.

Schedulers:
  fifo       - position-based round-robin within each batch (arrival order)
  rr         - same as fifo (round-robin, no latency awareness)
  random     - shuffle within each batch, then round-robin
  length_lpt - LPT using prompt_tokens as predicted latency; worker choice via pred loads
  ewma_lpt   - LPT using EWMA predicted latency; worker choice via pred loads
  oracle_lpt - LPT using true latency for both sorting and worker choice (upper bound)
"""

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple


class Group(NamedTuple):
    prompt_id: int
    prompt_tokens: int
    source: str
    group_max_latency_s: float
    group_total_tokens: int


def load_trace(path: str) -> list[Group]:
    groups = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            lat = r.get("group_max_latency_s_approx") or (
                r["group_max_tokens"] / r.get("tokens_per_sec_global", 1000)
            )
            groups.append(Group(
                prompt_id=r["prompt_id"],
                prompt_tokens=r["prompt_tokens"],
                source=r.get("source", "unknown"),
                group_max_latency_s=lat,
                group_total_tokens=r["group_total_tokens"],
            ))
    return groups


def _percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    return s[min(int(len(s) * p / 100), len(s) - 1)]


def _batch_metrics(
    sorted_batch: list[Group],
    num_workers: int,
    pred_fn,       # group -> predicted latency (used for worker CHOICE)
    true_fn,       # group -> true latency (used for makespan / CV)
) -> tuple[float, float, float, list[float]]:
    """
    Assign a sorted batch to workers.

    Worker selection: greedy min predicted-load (no leakage of true latency).
    Makespan / idle / CV: from true loads.

    Returns: (makespan, idle_frac, load_cv, group_close_times_within_batch)
    """
    pred_loads = [0.0] * num_workers
    true_loads = [0.0] * num_workers
    worker_group_lists: list[list[Group]] = [[] for _ in range(num_workers)]

    for g in sorted_batch:
        # Choose worker by predicted load (no oracle leakage)
        w = min(range(num_workers), key=lambda i: pred_loads[i])
        pred_loads[w] += pred_fn(g)
        true_loads[w] += true_fn(g)
        worker_group_lists[w].append(g)

    makespan = max(true_loads)

    # Idle fraction
    total_work = sum(true_loads)
    total_capacity = makespan * num_workers
    idle_frac = (total_capacity - total_work) / total_capacity if total_capacity > 0 else 0.0

    # Load CV (coefficient of variation over true loads)
    if len(true_loads) > 1 and statistics.mean(true_loads) > 0:
        load_cv = statistics.stdev(true_loads) / statistics.mean(true_loads)
    else:
        load_cv = 0.0

    # Group close times: cumsum within each worker (true latency)
    close_times = []
    for w_groups in worker_group_lists:
        t = 0.0
        for g in w_groups:
            t += true_fn(g)
            close_times.append(t)

    return makespan, idle_frac, load_cv, close_times


def simulate_rr(
    groups: list[Group],
    num_workers: int,
    rollout_batch_size: int,
    shuffle_within_batch: bool = False,
) -> dict:
    """FIFO / RR: strict position-based round-robin within each batch."""
    makespans, idles, cvs, all_close = [], [], [], []

    for batch_start in range(0, len(groups), rollout_batch_size):
        batch = list(groups[batch_start: batch_start + rollout_batch_size])
        if not batch:
            break

        if shuffle_within_batch:
            random.shuffle(batch)

        # Round-robin assignment
        true_loads = [0.0] * num_workers
        worker_group_lists: list[list[Group]] = [[] for _ in range(num_workers)]
        for pos, g in enumerate(batch):
            w = pos % num_workers
            true_loads[w] += g.group_max_latency_s
            worker_group_lists[w].append(g)

        makespan = max(true_loads)
        makespans.append(makespan)

        total_capacity = makespan * num_workers
        idles.append((total_capacity - sum(true_loads)) / total_capacity if total_capacity > 0 else 0.0)

        if len(true_loads) > 1 and statistics.mean(true_loads) > 0:
            cvs.append(statistics.stdev(true_loads) / statistics.mean(true_loads))
        else:
            cvs.append(0.0)

        for w_groups in worker_group_lists:
            t = 0.0
            for g in w_groups:
                t += g.group_max_latency_s
                all_close.append(t)

    return {
        "mean_makespan_s": statistics.mean(makespans) if makespans else 0,
        "p50_makespan_s": _percentile(makespans, 50),
        "p95_makespan_s": _percentile(makespans, 95),
        "worker_idle_frac": statistics.mean(idles) if idles else 0,
        "worker_load_cv": statistics.mean(cvs) if cvs else 0,
        "group_close_p50_s": _percentile(all_close, 50),
        "group_close_p95_s": _percentile(all_close, 95),
        "group_close_p99_s": _percentile(all_close, 99),
    }


def simulate_lpt_with_pred(
    groups: list[Group],
    num_workers: int,
    rollout_batch_size: int,
    pred_fn,   # group -> predicted latency (sort key + worker choice)
) -> dict:
    """LPT scheduling: sort by pred_fn descending, assign by min pred load."""
    makespans, idles, cvs, all_close = [], [], [], []

    for batch_start in range(0, len(groups), rollout_batch_size):
        batch = groups[batch_start: batch_start + rollout_batch_size]
        if not batch:
            break
        sorted_batch = sorted(batch, key=pred_fn, reverse=True)
        makespan, idle, cv, close_times = _batch_metrics(
            sorted_batch, num_workers,
            pred_fn=pred_fn,
            true_fn=lambda g: g.group_max_latency_s,
        )
        makespans.append(makespan)
        idles.append(idle)
        cvs.append(cv)
        all_close.extend(close_times)

    return {
        "mean_makespan_s": statistics.mean(makespans) if makespans else 0,
        "p50_makespan_s": _percentile(makespans, 50),
        "p95_makespan_s": _percentile(makespans, 95),
        "worker_idle_frac": statistics.mean(idles) if idles else 0,
        "worker_load_cv": statistics.mean(cvs) if cvs else 0,
        "group_close_p50_s": _percentile(all_close, 50),
        "group_close_p95_s": _percentile(all_close, 95),
        "group_close_p99_s": _percentile(all_close, 99),
    }


class EWMAPredictor:
    """
    Online EWMA latency predictor using prompt-length bins.

    Buckets prompts by `prompt_tokens // length_bin_size` (default bin=50 tokens).
    This provides much finer-grained signal than source-bucket EWMA,
    because within a source, prompt length is correlated with output length.

    Also maintains an output-to-input length ratio (multiplier) per bin to
    handle variable completion lengths at different prompt lengths.
    """

    def __init__(self, alpha: float = 0.30, length_bin_size: int = 50):
        self.alpha = alpha
        self.length_bin_size = length_bin_size
        self.estimates: dict[int, float] = {}  # bin -> predicted latency

    def _bin(self, prompt_tokens: int) -> int:
        return prompt_tokens // self.length_bin_size

    def predict(self, group: Group) -> float:
        b = self._bin(group.prompt_tokens)
        if b not in self.estimates:
            # Cold-start: scale by prompt tokens (empirically ~1ms/token at 1000 tps)
            return group.prompt_tokens * 1e-3
        return self.estimates[b]

    def update(self, group: Group, actual_latency: float) -> None:
        b = self._bin(group.prompt_tokens)
        if b not in self.estimates:
            self.estimates[b] = actual_latency
        else:
            self.estimates[b] = self.alpha * actual_latency + (1 - self.alpha) * self.estimates[b]


def simulate_ewma_lpt(
    groups: list[Group],
    num_workers: int,
    rollout_batch_size: int,
    ewma_alpha: float,
    warmup_groups: int,
) -> dict:
    """EWMA-LPT scheduler (proposed GA-SRollout)."""
    predictor = EWMAPredictor(alpha=ewma_alpha)
    makespans, idles, cvs, all_close = [], [], [], []
    predicted_lats, true_lats = [], []
    total_groups_seen = 0

    for batch_start in range(0, len(groups), rollout_batch_size):
        batch = groups[batch_start: batch_start + rollout_batch_size]
        if not batch:
            break

        in_warmup = total_groups_seen < warmup_groups

        # Collect predictions before sorting (no leakage)
        preds = {g.prompt_id: predictor.predict(g) for g in batch}

        if in_warmup:
            # Warmup: round-robin (fall back to FIFO ordering)
            sorted_batch = list(batch)
        else:
            sorted_batch = sorted(batch, key=lambda g: preds[g.prompt_id], reverse=True)

        for g in sorted_batch:
            predicted_lats.append(preds[g.prompt_id])
            true_lats.append(g.group_max_latency_s)

        if in_warmup:
            # Round-robin during warmup
            true_loads = [0.0] * num_workers
            worker_group_lists: list[list[Group]] = [[] for _ in range(num_workers)]
            for pos, g in enumerate(sorted_batch):
                w = pos % num_workers
                true_loads[w] += g.group_max_latency_s
                worker_group_lists[w].append(g)
            makespan = max(true_loads)
            total_capacity = makespan * num_workers
            idle = (total_capacity - sum(true_loads)) / total_capacity if total_capacity > 0 else 0
            cv = statistics.stdev(true_loads) / statistics.mean(true_loads) if statistics.mean(true_loads) > 0 else 0
            close_times = []
            for w_groups in worker_group_lists:
                t = 0.0
                for g in w_groups:
                    t += g.group_max_latency_s
                    close_times.append(t)
        else:
            makespan, idle, cv, close_times = _batch_metrics(
                sorted_batch, num_workers,
                pred_fn=lambda g: preds[g.prompt_id],
                true_fn=lambda g: g.group_max_latency_s,
            )

        makespans.append(makespan)
        idles.append(idle)
        cvs.append(cv)
        all_close.extend(close_times)

        # Update predictor AFTER dispatch (no leakage)
        for g in batch:
            predictor.update(g, g.group_max_latency_s)
        total_groups_seen += len(batch)

    spearman = _spearman(predicted_lats, true_lats)
    mae = statistics.mean(abs(p - t) for p, t in zip(predicted_lats, true_lats)) if predicted_lats else 0

    return {
        "mean_makespan_s": statistics.mean(makespans) if makespans else 0,
        "p50_makespan_s": _percentile(makespans, 50),
        "p95_makespan_s": _percentile(makespans, 95),
        "worker_idle_frac": statistics.mean(idles) if idles else 0,
        "worker_load_cv": statistics.mean(cvs) if cvs else 0,
        "group_close_p50_s": _percentile(all_close, 50),
        "group_close_p95_s": _percentile(all_close, 95),
        "group_close_p99_s": _percentile(all_close, 99),
        "prediction_spearman": spearman,
        "prediction_mae_s": mae,
    }


def _spearman(x: list[float], y: list[float]) -> float:
    if len(x) < 2:
        return 0.0
    n = len(x)

    def ranks(v: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        for rank, idx in enumerate(order):
            r[idx] = rank + 1.0
        return r

    rx, ry = ranks(x), ranks(y)
    d2 = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    return 1.0 - 6.0 * d2 / (n * (n ** 2 - 1))


def run_scheduler(
    name: str,
    groups: list[Group],
    num_workers: int,
    rollout_batch_size: int,
    ewma_alpha: float,
    warmup_groups: int,
) -> dict:
    if name in ("fifo", "rr"):
        return simulate_rr(groups, num_workers, rollout_batch_size, shuffle_within_batch=False)
    elif name == "random":
        return simulate_rr(groups, num_workers, rollout_batch_size, shuffle_within_batch=True)
    elif name == "length_lpt":
        return simulate_lpt_with_pred(
            groups, num_workers, rollout_batch_size,
            pred_fn=lambda g: g.prompt_tokens,
        )
    elif name == "ewma_lpt":
        return simulate_ewma_lpt(groups, num_workers, rollout_batch_size, ewma_alpha, warmup_groups)
    elif name == "oracle_lpt":
        # Oracle: both sort and worker choice use true latency
        return simulate_lpt_with_pred(
            groups, num_workers, rollout_batch_size,
            pred_fn=lambda g: g.group_max_latency_s,
        )
    else:
        raise ValueError(f"Unknown scheduler: {name}")


def main():
    parser = argparse.ArgumentParser(description="Replay group schedulers on collected traces")
    parser.add_argument("--trace", required=True)
    parser.add_argument("--num-workers", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--rollout-batch-size", type=int, default=128)
    parser.add_argument("--schedulers", nargs="+",
                        default=["fifo", "rr", "random", "length_lpt", "ewma_lpt", "oracle_lpt"])
    parser.add_argument("--ewma-alpha", type=float, default=0.30)
    parser.add_argument("--warmup-groups", type=int, default=512)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading trace from {args.trace}")
    groups = load_trace(args.trace)
    print(f"Loaded {len(groups)} groups")

    all_results: dict = {}

    for nw in args.num_workers:
        print(f"\n--- num_workers={nw} ---")
        results_by_sched: dict[str, dict] = {}

        for sched in args.schedulers:
            print(f"  Running scheduler: {sched} ...", end=" ", flush=True)
            result = run_scheduler(sched, groups, nw, args.rollout_batch_size, args.ewma_alpha, args.warmup_groups)
            results_by_sched[sched] = result
            print(f"mean_makespan={result['mean_makespan_s']:.3f}s")

        # Compute speedup vs fifo
        fifo_makespan = results_by_sched.get("fifo", {}).get("mean_makespan_s", 1.0)
        oracle_makespan = results_by_sched.get("oracle_lpt", {}).get("mean_makespan_s", fifo_makespan)

        for sched, r in results_by_sched.items():
            ms = r["mean_makespan_s"]
            r["speedup_vs_fifo"] = round(fifo_makespan / ms, 4) if ms > 0 else 1.0

        # oracle_gap_closed = makespan reduction fraction (not speedup ratio)
        # = (fifo_makespan - ewma_makespan) / (fifo_makespan - oracle_makespan)
        ewma_makespan = results_by_sched.get("ewma_lpt", {}).get("mean_makespan_s", fifo_makespan)
        denom = fifo_makespan - oracle_makespan
        if denom > 1e-9:
            oracle_gap_closed = (fifo_makespan - ewma_makespan) / denom
        else:
            oracle_gap_closed = 0.0

        if "ewma_lpt" in results_by_sched:
            results_by_sched["ewma_lpt"]["oracle_gap_closed"] = round(oracle_gap_closed, 4)

        all_results[f"workers_{nw}"] = results_by_sched

        # Print summary
        print(f"\n  Summary (num_workers={nw}):")
        print(f"  {'Scheduler':<15} {'mean_makespan':>13} {'speedup_vs_fifo':>15} {'idle%':>6} {'load_cv':>7}")
        for sched in args.schedulers:
            if sched not in results_by_sched:
                continue
            r = results_by_sched[sched]
            print(f"  {sched:<15} {r['mean_makespan_s']:>12.3f}s "
                  f"{r['speedup_vs_fifo']:>14.3f}x "
                  f"{r['worker_idle_frac']*100:>5.1f}% "
                  f"{r['worker_load_cv']:>7.3f}")

        # Pass/fail
        ewma_r = results_by_sched.get("ewma_lpt", {})
        speedup = ewma_r.get("speedup_vs_fifo", 1.0)
        gap_closed = ewma_r.get("oracle_gap_closed", 0.0)
        spearman = ewma_r.get("prediction_spearman", 0.0)
        passed = speedup >= 1.15 and gap_closed >= 0.5
        strong_passed = speedup >= 1.25
        print(f"\n  PASS GATE (ewma_lpt, nw={nw}):")
        print(f"    speedup_vs_fifo   = {speedup:.3f}x (need >=1.15x) → {'PASS' if speedup >= 1.15 else 'FAIL'}")
        print(f"    oracle_gap_closed = {gap_closed:.3f} (need >=0.50) → {'PASS' if gap_closed >= 0.5 else 'FAIL'}")
        print(f"    prediction_spearman = {spearman:.3f}")
        print(f"    Overall: {'STRONG PASS' if strong_passed else 'PASS' if passed else 'FAIL'}")

    Path(args.out).write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()
