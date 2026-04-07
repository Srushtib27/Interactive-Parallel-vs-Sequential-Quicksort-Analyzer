"""
analytics.py - Performance Metrics, Insights, and Recommendations Engine
"""
import math
import statistics
from dataclasses import dataclass, field, asdict
from typing import Optional
import json
import csv
import os
from datetime import datetime


@dataclass
class BenchmarkResult:
    timestamp: str
    n: int
    processes: int
    dataset_type: str
    seq_time: float
    par_time: float
    speedup: float
    efficiency: float
    throughput_seq: float   # elements/sec
    throughput_par: float
    is_parallel_better: bool

    def to_dict(self):
        return asdict(self)


def compute_metrics(n: int, processes: int, seq_time: float, par_time: float, dataset_type: str) -> BenchmarkResult:
    speedup = seq_time / par_time if par_time > 0 else 0.0
    efficiency = (speedup / processes) * 100 if processes > 0 else 0.0
    throughput_seq = n / seq_time if seq_time > 0 else 0.0
    throughput_par = n / par_time if par_time > 0 else 0.0
    is_parallel_better = par_time < seq_time

    return BenchmarkResult(
        timestamp=datetime.now().isoformat(),
        n=n,
        processes=processes,
        dataset_type=dataset_type,
        seq_time=round(seq_time, 6),
        par_time=round(par_time, 6),
        speedup=round(speedup, 4),
        efficiency=round(efficiency, 2),
        throughput_seq=round(throughput_seq, 2),
        throughput_par=round(throughput_par, 2),
        is_parallel_better=is_parallel_better,
    )


def generate_insights(result: BenchmarkResult, all_results: list[BenchmarkResult] = None) -> list[str]:
    """Generate AI-style performance insights from a benchmark result."""
    insights = []
    n = result.n
    p = result.processes
    speedup = result.speedup
    efficiency = result.efficiency

    # --- Size-based insights ---
    if n < 5000:
        insights.append(
            f"⚠️ Small input (n={n:,}): Parallelization overhead dominates. "
            "Sequential sorting is almost always faster at this scale."
        )
    elif n < 20000:
        insights.append(
            f"📊 Medium input (n={n:,}): Parallel gains start appearing, "
            "but communication latency still has a visible impact."
        )
    else:
        insights.append(
            f"🚀 Large input (n={n:,}): This is where parallelism shines — "
            "data volume justifies inter-process coordination cost."
        )

    # --- Speedup analysis ---
    if speedup < 1.0:
        insights.append(
            f"🐢 Speedup is {speedup:.2f}x (< 1) — parallel is actually slower. "
            "MPI overhead exceeds any gains from splitting the work."
        )
    elif speedup < 1.5:
        insights.append(
            f"📈 Modest speedup of {speedup:.2f}x with {p} processes. "
            "Amdahl's Law in action: serial portions cap your gains."
        )
    elif speedup >= 1.5 and speedup < p:
        insights.append(
            f"✅ Good speedup of {speedup:.2f}x with {p} processes. "
            "Parallelism is paying off, though not perfectly linear."
        )
    elif speedup >= p:
        insights.append(
            f"🌟 Super-linear speedup of {speedup:.2f}x! "
            "Possible cache effects or favorable data distribution."
        )

    # --- Efficiency analysis ---
    if efficiency >= 80:
        insights.append(f"💚 Parallel efficiency: {efficiency:.1f}% — excellent resource utilization.")
    elif efficiency >= 50:
        insights.append(
            f"🟡 Parallel efficiency: {efficiency:.1f}% — moderate. "
            f"~{100 - efficiency:.0f}% of compute time is spent in coordination."
        )
    else:
        insights.append(
            f"🔴 Parallel efficiency: {efficiency:.1f}% — poor. "
            "Too many processes for this input size. Reduce process count."
        )

    # --- Process count recommendations ---
    if p > 4 and efficiency < 50:
        insights.append(
            f"💡 Recommendation: Try fewer processes (2–4) for n={n:,}. "
            "Diminishing returns set in quickly beyond this point."
        )

    if p == 1:
        insights.append(
            "ℹ️ With 1 process, parallel mode mimics sequential — "
            "no real MPI parallelism is applied."
        )

    # --- Dataset-type insights ---
    if result.dataset_type == "sorted":
        insights.append(
            "🔍 Sorted input: Quicksort's worst-case (O(n²)) pivot behavior "
            "is avoided here due to middle-pivot selection."
        )
    elif result.dataset_type == "reverse":
        insights.append(
            "🔄 Reverse-sorted input: Similar to sorted — "
            "middle pivot keeps performance near O(n log n)."
        )
    elif result.dataset_type == "nearly_sorted":
        insights.append(
            "📐 Nearly-sorted input: Very cache-friendly for sequential. "
            "Parallel may see unbalanced chunk work distribution."
        )

    # --- Historical trend insights ---
    if all_results and len(all_results) >= 3:
        same_n = [r for r in all_results if r.n == n]
        if len(same_n) >= 2:
            best = max(same_n, key=lambda r: r.speedup)
            insights.append(
                f"📚 Across {len(same_n)} runs at n={n:,}, "
                f"best speedup was {best.speedup:.2f}x using {best.processes} processes."
            )

    return insights


def recommend_config(results: list[BenchmarkResult]) -> dict:
    """Return recommended configuration based on accumulated results."""
    if not results:
        return {}

    # Find config with best speedup where efficiency >= 60%
    good = [r for r in results if r.efficiency >= 60]
    if good:
        best = max(good, key=lambda r: r.speedup)
    else:
        best = max(results, key=lambda r: r.speedup)

    avg_seq = statistics.mean(r.seq_time for r in results)
    avg_par = statistics.mean(r.par_time for r in results)

    return {
        "recommended_processes": best.processes,
        "recommended_n": best.n,
        "use_parallel": best.is_parallel_better,
        "best_speedup": best.speedup,
        "best_efficiency": best.efficiency,
        "avg_seq_time": round(avg_seq, 6),
        "avg_par_time": round(avg_par, 6),
        "note": (
            f"Use {best.processes} processes for input size ~{best.n:,}. "
            f"Achieved {best.speedup:.2f}x speedup at {best.efficiency:.1f}% efficiency."
        )
    }


# ── Persistence ──────────────────────────────────────────────────────────────

RESULTS_FILE = "benchmark_results.json"


def save_result(result: BenchmarkResult):
    results = load_results()
    results.append(result.to_dict())
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)


def load_results() -> list[BenchmarkResult]:
    if not os.path.exists(RESULTS_FILE):
        return []
    try:
        with open(RESULTS_FILE) as f:
            raw = json.load(f)
        return [BenchmarkResult(**r) for r in raw]
    except Exception:
        return []


def export_csv(results: list[BenchmarkResult], path: str = "benchmark_results.csv"):
    if not results:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].to_dict().keys())
        writer.writeheader()
        for r in results:
            writer.writerow(r.to_dict())
