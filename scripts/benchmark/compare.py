"""Aggregate benchmark results and produce cost-normalized quality comparison.

Usage:
    python -m scripts.benchmark.compare results/gsm8k_*.json
    python -m scripts.benchmark.compare results/gsm8k_*.json --csv

Reads one or more result JSON files produced by run_gsm8k.py / run_humaneval.py /
run_truthfulqa.py, then prints:
- Accuracy (or pass@1) per baseline ± σ across runs
- Mean cost per task ($USD)
- Quality per $0.01 token cost
- Paired t-test p-value for H2 vs each other baseline (if ≥2 runs)
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Paired t-test (no scipy dependency)
# ---------------------------------------------------------------------------

def _t_statistic(diffs: list[float]) -> float:
    n = len(diffs)
    if n < 2:
        return float("nan")
    mean_d = statistics.mean(diffs)
    std_d = statistics.stdev(diffs)
    if std_d == 0.0:
        return float("nan")
    return mean_d / (std_d / math.sqrt(n))


def _p_value_two_tailed(t: float, df: int) -> float:
    """Rough p-value via incomplete beta (Abramowitz & Stegun 26.7.8)."""
    if math.isnan(t) or df <= 0:
        return float("nan")
    # Use normal approximation for df > 30; exact Cornish-Fisher otherwise
    x = df / (df + t * t)

    def _ibeta_half(a: float, b: float, xx: float) -> float:
        """Regularised incomplete beta I_x(a,b) via continued fraction."""
        if xx <= 0.0:
            return 0.0
        if xx >= 1.0:
            return 1.0
        lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
        front = math.exp(math.log(xx) * a + math.log(1 - xx) * b - lbeta) / a
        # Lentz continued fraction
        qab = a + b
        qap = a + 1
        qam = a - 1
        c, d = 1.0, 1.0 - qab * xx / qap
        if abs(d) < 1e-30:
            d = 1e-30
        d = 1.0 / d
        h = d
        for m in range(1, 200):
            m2 = 2 * m
            aa = m * (b - m) * xx / ((qam + m2) * (a + m2))
            d = 1.0 + aa * d
            if abs(d) < 1e-30:
                d = 1e-30
            c = 1.0 + aa / c
            if abs(c) < 1e-30:
                c = 1e-30
            d = 1.0 / d
            h *= d * c
            aa = -(a + m) * (qab + m) * xx / ((a + m2) * (qap + m2))
            d = 1.0 + aa * d
            if abs(d) < 1e-30:
                d = 1e-30
            c = 1.0 + aa / c
            if abs(c) < 1e-30:
                c = 1e-30
            d = 1.0 / d
            delta = d * c
            h *= delta
            if abs(delta - 1.0) < 1e-10:
                break
        return front * h

    p_one_tail = _ibeta_half(df / 2.0, 0.5, x)
    return min(1.0, 2.0 * p_one_tail)


# ---------------------------------------------------------------------------
# Result loading
# ---------------------------------------------------------------------------

def _metric_key(benchmark: str) -> str:
    return "pass_at_1" if benchmark == "humaneval" else "accuracy"


def _load_results(paths: list[Path]) -> dict[str, list[dict]]:
    """Load result files, keyed by benchmark name."""
    by_benchmark: dict[str, list[dict]] = {}
    for p in paths:
        with open(p) as f:
            data = json.load(f)
        bm = data.get("benchmark", p.stem.split("_")[0])
        by_benchmark.setdefault(bm, []).extend(data["results"])
    return by_benchmark


# ---------------------------------------------------------------------------
# Aggregate per (baseline, run_index)
# ---------------------------------------------------------------------------

def _aggregate(results: list[dict], metric: str) -> dict[str, Any]:
    """Return summary table: baseline → {accuracy_runs, cost_runs, tasks_per_task}."""
    by_baseline: dict[str, dict[str, list]] = {}
    for r in results:
        bl = r["baseline"]
        if bl not in by_baseline:
            by_baseline[bl] = {"metric": [], "cost": [], "task_correct": []}
        by_baseline[bl]["metric"].append(r[metric])
        by_baseline[bl]["cost"].append(r["mean_cost_usd"])
        # Per-task correctness for paired t-test
        task_key = "passed" if metric == "pass_at_1" else "correct"
        by_baseline[bl]["task_correct"].append(
            [t[task_key] for t in r.get("task_results", [])]
        )
    return by_baseline


def _print_table(by_baseline: dict[str, dict], metric_name: str) -> None:
    print(f"\n{'Baseline':<12} {'Metric':>8} {'±σ':>6} {'Cost/task':>10} {'Q/$0.01':>10}")
    print("-" * 52)
    for bl, d in sorted(by_baseline.items()):
        runs = d["metric"]
        costs = d["cost"]
        mean_m = statistics.mean(runs)
        std_m = statistics.stdev(runs) if len(runs) > 1 else 0.0
        mean_c = statistics.mean(costs)
        quality_per_cent = mean_m / (mean_c * 100) if mean_c > 0 else float("inf")
        print(
            f"{bl:<12} {mean_m:>8.4f} {std_m:>6.4f} "
            f"${mean_c:>9.5f} {quality_per_cent:>10.3f}"
        )


def _paired_t_test(
    by_baseline: dict[str, dict],
    task_key: str,
    reference: str = "h2",
) -> None:
    if reference not in by_baseline:
        return
    ref_runs = by_baseline[reference]["task_correct"]
    if not ref_runs or len(ref_runs[0]) == 0:
        return

    print(f"\nPaired t-test vs '{reference}':")
    for bl, d in sorted(by_baseline.items()):
        if bl == reference:
            continue
        other_runs = d["task_correct"]
        n_runs = min(len(ref_runs), len(other_runs))
        if n_runs == 0:
            continue
        # Flatten diffs across runs
        diffs: list[float] = []
        for run_idx in range(n_runs):
            r_tasks = ref_runs[run_idx]
            o_tasks = other_runs[run_idx]
            n = min(len(r_tasks), len(o_tasks))
            diffs.extend(
                float(r_tasks[i]) - float(o_tasks[i]) for i in range(n)
            )
        t = _t_statistic(diffs)
        df = len(diffs) - 1
        p = _p_value_two_tailed(t, df)
        sig = " *" if p < 0.05 else ("  (n.s.)" if not math.isnan(p) else "  (n/a)")
        p_str = f"{p:.4f}" if not math.isnan(p) else "  n/a"
        print(f"  {reference} vs {bl}: t={t:.3f}, df={df}, p={p_str}{sig}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare benchmark results")
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--csv", action="store_true", help="Also emit CSV")
    parser.add_argument("--reference", default="h2", help="Reference baseline for t-test")
    args = parser.parse_args()

    by_benchmark = _load_results(args.files)
    for bm, results in by_benchmark.items():
        metric = _metric_key(bm)
        print(f"\n{'=' * 52}")
        print(f"Benchmark: {bm.upper()}  (metric: {metric})")
        by_baseline = _aggregate(results, metric)
        _print_table(by_baseline, metric)
        _paired_t_test(by_baseline, metric, reference=args.reference)

    if args.csv:
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["benchmark", "baseline", "metric_mean", "metric_std",
                         "cost_per_task_usd", "quality_per_cent"])
        for bm, results in by_benchmark.items():
            metric = _metric_key(bm)
            by_baseline = _aggregate(results, metric)
            for bl, d in by_baseline.items():
                runs = d["metric"]
                costs = d["cost"]
                mean_m = statistics.mean(runs)
                std_m = statistics.stdev(runs) if len(runs) > 1 else 0.0
                mean_c = statistics.mean(costs)
                qpc = mean_m / (mean_c * 100) if mean_c > 0 else 0.0
                writer.writerow([bm, bl, f"{mean_m:.4f}", f"{std_m:.4f}",
                                  f"{mean_c:.6f}", f"{qpc:.4f}"])
        print("\nCSV:\n" + buf.getvalue())


if __name__ == "__main__":
    main()
