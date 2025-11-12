"""Visualization utilities for analysis outputs.

Graceful fallback: if matplotlib (or related plotting stack) is not available,
the public functions print a concise message and return without raising.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, cast

try:
    import matplotlib.pyplot as plt  # type: ignore
    import numpy as np  # type: ignore
    _PLOTS_AVAILABLE = True
except Exception:
    plt = cast(Any, None)  # type: ignore
    np = cast(Any, None)  # type: ignore
    _PLOTS_AVAILABLE = False

from .stats import ExperimentResults
def _bt_canonical_entry(results: ExperimentResults, N: int) -> Dict[str, Any]:
    entry = results["BT"][N]
    if isinstance(entry.get("solution_found") if isinstance(entry, dict) else None, bool):
        return entry  # legacy single-solver structure
    # New per-solver dict: pick preferred solver if available
    priority = ["mcv_hybrid", "mcv", "lcv", "first"]
    for key in priority:
        if key in entry:
            return entry[key]
    # Fallback: take arbitrary first
    first_key = next(iter(entry))
    return entry[first_key]

try:
    import seaborn as sns  # noqa: F401
except Exception:
    sns = None  # type: ignore


def plot_comprehensive_analysis(
    results: ExperimentResults,
    N_values: List[int],
    fitness_mode: str,
    out_dir: str,
    raw_runs: Optional[Dict[str, Any]] = None,
    tuning_data: Optional[Dict[str, Any]] = None,
) -> None:
    if not _PLOTS_AVAILABLE:
        print("Plotting skipped: matplotlib not installed.")
        return
    os.makedirs(out_dir, exist_ok=True)

    bt_sr = [1.0 if _bt_canonical_entry(results, N)["solution_found"] else 0.0 for N in N_values]
    sa_sr = [cast(float, results["SA"][N].get("success_rate", 0.0) or 0.0) for N in N_values]
    ga_sr = [cast(float, results["GA"][N].get("success_rate", 0.0) or 0.0) for N in N_values]

    bt_timeout = [0.0 for _ in N_values]
    sa_timeout = [results["SA"][N].get("timeout_rate", 0.0) for N in N_values]
    ga_timeout = [results["GA"][N].get("timeout_rate", 0.0) for N in N_values]

    plt.figure(figsize=(12, 8))
    plt.plot(N_values, bt_sr, marker="o", linewidth=2, markersize=8, label="Backtracking")
    plt.plot(N_values, sa_sr, marker="s", linewidth=2, markersize=8, label="Simulated Annealing")
    plt.plot(N_values, ga_sr, marker="^", linewidth=2, markersize=8, label=f"Genetic Algorithm (F{fitness_mode})")
    plt.xlabel("N (board size)", fontsize=12)
    plt.ylabel("Success rate", fontsize=12)
    plt.title("Success Rate vs Problem Size\n(Algorithm reliability as N grows)", fontsize=14)
    plt.ylim(-0.05, 1.05)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.7)
    plt.xticks(N_values)

    for i, n in enumerate(N_values):
        plt.annotate(f"{bt_sr[i]:.1f}", (n, bt_sr[i]), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=9, color="blue")
        plt.annotate(f"{sa_sr[i]:.2f}", (n, sa_sr[i]), textcoords="offset points", xytext=(0, -15), ha="center", fontsize=9, color="orange")
        plt.annotate(f"{ga_sr[i]:.2f}", (n, ga_sr[i]), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=9, color="green")

    fname = os.path.join(out_dir, f"01_success_rate_vs_N_F{fitness_mode}.png")
    plt.savefig(fname, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved success-rate chart: {fname}")

    bt_time = [_bt_canonical_entry(results, N)["time"] if _bt_canonical_entry(results, N)["solution_found"] else 0 for N in N_values]
    sa_time = [cast(float, results["SA"][N].get("success_time", {}).get("mean", 0.0) or 0.0) for N in N_values]
    ga_time = [cast(float, results["GA"][N].get("success_time", {}).get("mean", 0.0) or 0.0) for N in N_values]

    plt.figure(figsize=(12, 8))
    bt_time_plot = [max(t, 1e-6) for t in bt_time]
    sa_time_plot = [max(t, 1e-6) for t in sa_time]
    ga_time_plot = [max(t, 1e-6) for t in ga_time]

    plt.semilogy(N_values, bt_time_plot, marker="o", linewidth=2, markersize=8, label="Backtracking")
    plt.semilogy(N_values, sa_time_plot, marker="s", linewidth=2, markersize=8, label="Simulated Annealing")
    plt.semilogy(N_values, ga_time_plot, marker="^", linewidth=2, markersize=8, label=f"Genetic Algorithm (F{fitness_mode})")
    plt.xlabel("N (board size)", fontsize=12)
    plt.ylabel("Average time [s] (log scale)", fontsize=12)
    plt.title("Execution Time vs Problem Size\n(Successful runs only — highlights BT growth)", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.7)
    plt.xticks(N_values)

    fname = os.path.join(out_dir, f"02_time_vs_N_log_scale_F{fitness_mode}.png")
    plt.savefig(fname, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved execution-time chart (log scale): {fname}")

    bt_nodes = [_bt_canonical_entry(results, N)["nodes"] for N in N_values]
    sa_steps = [cast(float, results["SA"][N].get("success_steps", {}).get("mean", 0.0) or 0.0) for N in N_values]
    ga_gen = [cast(float, results["GA"][N].get("success_gen", {}).get("mean", 0.0) or 0.0) for N in N_values]

    plt.figure(figsize=(12, 8))
    plt.semilogy(N_values, [max(n, 1) for n in bt_nodes], marker="o", linewidth=2, markersize=8, label="BT: Explored nodes")
    plt.semilogy(N_values, [max(s, 1) for s in sa_steps], marker="s", linewidth=2, markersize=8, label="SA: Average iterations")
    plt.semilogy(N_values, [max(g, 1) for g in ga_gen], marker="^", linewidth=2, markersize=8, label="GA: Average generations")
    plt.xlabel("N (board size)", fontsize=12)
    plt.ylabel("Logical cost (log scale)", fontsize=12)
    plt.title("Theoretical Computational Cost vs Problem Size\n(Hardware-independent scalability)", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.7)
    plt.xticks(N_values)

    fname = os.path.join(out_dir, f"03_logical_cost_vs_N_F{fitness_mode}.png")
    plt.savefig(fname, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved logical-cost chart: {fname}")

    sa_evals = [cast(float, results["SA"][N].get("success_evals", {}).get("mean", 0.0) or 0.0) for N in N_values]
    ga_evals = [cast(float, results["GA"][N].get("success_evals", {}).get("mean", 0.0) or 0.0) for N in N_values]

    plt.figure(figsize=(12, 8))
    plt.semilogy(N_values, [max(e, 1) for e in sa_evals], marker="s", linewidth=2, markersize=8, label="SA: Conflict evaluations")
    plt.semilogy(N_values, [max(e, 1) for e in ga_evals], marker="^", linewidth=2, markersize=8, label=f"GA-F{fitness_mode}: Fitness evaluations")
    plt.xlabel("N (board size)", fontsize=12)
    plt.ylabel("Objective evaluations (log scale)", fontsize=12)
    plt.title("Pure Objective Evaluation Cost\n(Computational burden of evaluations)", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.7)
    plt.xticks(N_values)

    fname = os.path.join(out_dir, f"04_fitness_evaluations_vs_N_F{fitness_mode}.png")
    plt.savefig(fname, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved fitness-evaluation chart: {fname}")

    plt.figure(figsize=(12, 8))
    plt.plot(N_values, [results["SA"][N].get("timeout_rate", 0.0) for N in N_values], marker="s", linewidth=2, markersize=8, label="SA: Timeout rate")
    plt.plot(N_values, [results["GA"][N].get("timeout_rate", 0.0) for N in N_values], marker="^", linewidth=2, markersize=8, label=f"GA-F{fitness_mode}: Timeout rate")
    plt.xlabel("N (board size)", fontsize=12)
    plt.ylabel("Timeout rate", fontsize=12)
    plt.title("Timeout Rate vs Problem Size\n(Where algorithms start exceeding the time limit)", fontsize=14)
    plt.ylim(-0.05, 1.05)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.7)
    plt.xticks(N_values)

    fname = os.path.join(out_dir, f"05_timeout_rate_vs_N_F{fitness_mode}.png")
    plt.savefig(fname, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved timeout-rate chart: {fname}")

    sa_fail_quality = [
        float(results["SA"][N].get("failure_best_conflicts", {}).get("mean", N) if results["SA"][N].get("failure_best_conflicts") else N)
        for N in N_values
    ]
    ga_fail_quality = [
        float(results["GA"][N].get("failure_best_conflicts", {}).get("mean", N) if results["GA"][N].get("failure_best_conflicts") else N)
        for N in N_values
    ]

    plt.figure(figsize=(12, 8))
    plt.plot(N_values, sa_fail_quality, marker="s", linewidth=2, markersize=8, label="SA: Average conflicts (failures)")
    plt.plot(N_values, ga_fail_quality, marker="^", linewidth=2, markersize=8, label=f"GA-F{fitness_mode}: Average conflicts (failures)")
    plt.plot(N_values, [0] * len(N_values), "k--", alpha=0.5, label="Optimal solution (0 conflicts)")
    plt.xlabel("N (board size)", fontsize=12)
    plt.ylabel("Average conflicts in failures", fontsize=12)
    plt.title("Solution Quality in Failed Runs\n(How close to optimal despite failure)", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.7)
    plt.xticks(N_values)

    fname = os.path.join(out_dir, f"06_failure_quality_vs_N_F{fitness_mode}.png")
    plt.savefig(fname, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved failure-quality chart: {fname}")

    if any(sa_steps) and any(sa_time):
        plt.figure(figsize=(12, 8))
        valid_sa = [(s, t, n) for s, t, n in zip(sa_steps, sa_time, N_values) if s > 0 and t > 0]
        if valid_sa:
            steps_valid, time_valid, n_valid = zip(*valid_sa)
            plt.scatter(steps_valid, time_valid, c=n_valid, cmap="viridis", s=100, alpha=0.8)
            plt.colorbar(label="N (size)")
            if len(valid_sa) > 2:
                z = np.polyfit(steps_valid, time_valid, 1)
                p = np.poly1d(z)
                x_trend = np.linspace(min(steps_valid), max(steps_valid), 100)
                plt.plot(x_trend, p(x_trend), "r--", alpha=0.8, label=f"Trend: y={z[0]:.2e}x+{z[1]:.2e}")
                plt.legend()

        plt.xlabel("SA iterations (logical cost)", fontsize=12)
        plt.ylabel("Time [s] (practical cost)", fontsize=12)
        plt.title(
            "Simulated Annealing: Theoretical vs Practical Cost Correlation\n(Linearity confirms evaluation-cost dominance)",
            fontsize=14,
        )
        plt.grid(True, alpha=0.7)

        fname = os.path.join(out_dir, f"07_SA_theoretical_vs_practical_F{fitness_mode}.png")
        plt.savefig(fname, bbox_inches="tight", dpi=300)
        plt.close()
    print(f"Saved SA theoretical-vs-practical chart: {fname}")

    if any(ga_evals) and any(ga_time):
        plt.figure(figsize=(12, 8))
        valid_ga = [(e, t, n) for e, t, n in zip(ga_evals, ga_time, N_values) if e > 0 and t > 0]
        if valid_ga:
            evals_valid, time_valid, n_valid = zip(*valid_ga)
            plt.scatter(evals_valid, time_valid, c=n_valid, cmap="plasma", s=100, alpha=0.8)
            plt.colorbar(label="N (size)")
            if len(valid_ga) > 2:
                z = np.polyfit(evals_valid, time_valid, 1)
                p = np.poly1d(z)
                x_trend = np.linspace(min(evals_valid), max(evals_valid), 100)
                plt.plot(x_trend, p(x_trend), "r--", alpha=0.8, label=f"Trend: y={z[0]:.2e}x+{z[1]:.2e}")
                plt.legend()

        plt.xlabel("GA fitness evaluations (logical cost)", fontsize=12)
        plt.ylabel("Time [s] (practical cost)", fontsize=12)
        plt.title(
            f"GA-F{fitness_mode}: Theoretical vs Practical Cost Correlation\n(Linearity confirms evaluation-cost dominance)",
            fontsize=14,
        )
        plt.grid(True, alpha=0.7)

        fname = os.path.join(out_dir, f"08_GA_theoretical_vs_practical_F{fitness_mode}.png")
        plt.savefig(fname, bbox_inches="tight", dpi=300)
        plt.close()
    print(f"Saved GA theoretical-vs-practical chart: {fname}")

    if any(bt_nodes) and any(bt_time):
        plt.figure(figsize=(12, 8))
        valid_bt = [(n, t, nval) for n, t, nval in zip(bt_nodes, bt_time, N_values) if n > 0 and t > 0]
        if valid_bt:
            nodes_valid, time_valid, n_valid = zip(*valid_bt)
            plt.scatter(nodes_valid, time_valid, c=n_valid, cmap="coolwarm", s=100, alpha=0.8)
            plt.colorbar(label="N (size)")
            if len(valid_bt) > 2:
                z = np.polyfit(nodes_valid, time_valid, 1)
                p = np.poly1d(z)
                x_trend = np.linspace(min(nodes_valid), max(nodes_valid), 100)
                plt.plot(x_trend, p(x_trend), "r--", alpha=0.8, label=f"Trend: y={z[0]:.2e}x+{z[1]:.2e}")
                plt.legend()

        plt.xlabel("BT explored nodes (logical cost)", fontsize=12)
        plt.ylabel("Time [s] (practical cost)", fontsize=12)
        plt.title("Backtracking: Theoretical vs Practical Cost Correlation\n(Each node has near-constant evaluation time)", fontsize=14)
        plt.grid(True, alpha=0.7)

        fname = os.path.join(out_dir, f"09_BT_theoretical_vs_practical_F{fitness_mode}.png")
        plt.savefig(fname, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"Saved BT theoretical-vs-practical chart: {fname}")

    print(f"\nComplete analysis generated in: {out_dir}")
    print(f"Generated {9} base charts for fitness F{fitness_mode}")


def plot_fitness_comparison(
    all_results: Dict[str, ExperimentResults], N_values: List[int], out_dir: str, raw_runs: Optional[Dict[str, Any]] = None
) -> None:
    if not _PLOTS_AVAILABLE:
        print("Plotting skipped: matplotlib not installed.")
        return
    os.makedirs(out_dir, exist_ok=True)
    fitness_modes = list(all_results.keys())

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    fitness_colors = {f: colors[i % len(colors)] for i, f in enumerate(fitness_modes)}

    analysis_N = [n for n in [16, 24, 40] if n in N_values]

    for N in analysis_N:
        plt.figure(figsize=(12, 8))
        success_rates = [cast(float, all_results[f]["GA"][N].get("success_rate", 0.0) or 0.0) for f in fitness_modes]

        bars = plt.bar(
            fitness_modes, success_rates, color=[fitness_colors[f] for f in fitness_modes], alpha=0.8
        )
        plt.xlabel("Fitness Function", fontsize=12)
        plt.ylabel("Success rate", fontsize=12)
        plt.title(
            f"Success Rate Comparison across Fitness Functions (N={N})\n(Which fitness converges better at the same size)",
            fontsize=14,
        )
        plt.ylim(0, 1.05)
        plt.grid(True, alpha=0.3, axis="y")

        for bar, sr in zip(bars, success_rates):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{sr:.3f}", ha="center", va="bottom", fontweight="bold")

        fname = os.path.join(out_dir, f"fitness_success_rate_N{N}.png")
        plt.savefig(fname, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"Saved success-rate comparison for N={N}: {fname}")

        plt.figure(figsize=(12, 8))
        gen_means = []
        gen_stds = []

        for f in fitness_modes:
            gen_stats = all_results[f]["GA"][N]["success_gen"]
            gen_means.append(gen_stats.get("mean", 0))
            gen_stds.append(gen_stats.get("std", 0))

        bars = plt.bar(
            fitness_modes, gen_means, yerr=gen_stds, color=[fitness_colors[f] for f in fitness_modes], alpha=0.8, capsize=5
        )
        plt.xlabel("Fitness Function", fontsize=12)
        plt.ylabel("Average generations +/- std", fontsize=12)
        plt.title(f"Convergence Speed Comparison (N={N})\n(Generations to reach a solution)", fontsize=14)
        plt.grid(True, alpha=0.3, axis="y")

        for bar, mean, std in zip(bars, gen_means, gen_stds):
            if mean > 0:
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.5, f"{mean:.1f}+/-{std:.1f}", ha="center", va="bottom", fontsize=10)

        fname = os.path.join(out_dir, f"fitness_generations_N{N}.png")
        plt.savefig(fname, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"Saved generation comparison for N={N}: {fname}")

        plt.figure(figsize=(12, 8))
        time_means = []
        time_stds = []

        for f in fitness_modes:
            time_stats = all_results[f]["GA"][N]["success_time"]
            time_means.append(time_stats.get("mean", 0))
            time_stds.append(time_stats.get("std", 0))

        bars = plt.bar(
            fitness_modes, time_means, yerr=time_stds, color=[fitness_colors[f] for f in fitness_modes], alpha=0.8, capsize=5
        )
        plt.xlabel("Fitness Function", fontsize=12)
        plt.ylabel("Average time [s] +/- std", fontsize=12)
        plt.title("Temporal Efficiency Comparison (N={N})\n(Trade-off between success and speed)", fontsize=14)
        plt.grid(True, alpha=0.3, axis="y")

        for bar, mean, std in zip(bars, time_means, time_stds):
            if mean > 0:
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + std + 0.001,
                    f"{mean:.3f}+/-{std:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    rotation=0,
                )

        fname = os.path.join(out_dir, f"fitness_time_N{N}.png")
        plt.savefig(fname, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"Saved time comparison for N={N}: {fname}")

    plt.figure(figsize=(15, 10))
    for f in fitness_modes:
        ga_sr_all_N = [cast(float, all_results[f]["GA"][N].get("success_rate", 0.0) or 0.0) for N in N_values]
        plt.plot(N_values, ga_sr_all_N, marker="o", linewidth=2, markersize=8, label=f"GA-F{f}", color=fitness_colors[f])

    plt.xlabel("N (board size)", fontsize=12)
    plt.ylabel("Success rate", fontsize=12)
    plt.title("Success Rate Evolution: All Fitness Functions\n(Reliability evolution as size increases)", fontsize=14)
    plt.ylim(-0.05, 1.05)
    plt.legend(fontsize=11, ncol=2)
    plt.grid(True, alpha=0.7)
    plt.xticks(N_values)

    fname = os.path.join(out_dir, f"fitness_evolution_all.png")
    plt.savefig(fname, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved fitness evolution overview: {fname}")

    print(f"\nFitness function comparison analysis completed")
    print(f"Generated comparison charts for F1-F6")


def plot_statistical_analysis(
    all_results: Dict[str, ExperimentResults], N_values: List[int], out_dir: str, raw_runs: Optional[Dict[int, Dict[str, Any]]] = None
) -> None:
    if not _PLOTS_AVAILABLE:
        print("Plotting skipped: matplotlib not installed.")
        return
    os.makedirs(out_dir, exist_ok=True)

    if not raw_runs:
        print("Raw runs not available for detailed statistical analysis")
        return

    analysis_N = [n for n in [16, 24, 40] if n in N_values and n in raw_runs]

    for N in analysis_N:
        if N not in raw_runs:
            continue

        print(f"Statistical analysis for N={N}...")

        import matplotlib.pyplot as plt  # local import to avoid heavy import if unused

        plt.figure(figsize=(14, 8))
        time_data = []
        labels = []

        if "SA" in raw_runs[N]:
            sa_times = [run["time"] for run in raw_runs[N]["SA"] if run["success"]]
            if sa_times:
                time_data.append(sa_times)
                labels.append("SA")

        for fitness in sorted(all_results.keys()):
            if fitness in raw_runs[N]:
                ga_times = [run["time"] for run in raw_runs[N][fitness] if run["success"]]
                if ga_times:
                    time_data.append(ga_times)
                    labels.append(f"GA-F{fitness}")

        if time_data:
            box_plot = plt.boxplot(time_data, labels=labels, patch_artist=True)
            colors = ["#ff7f0e", "#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
            for patch, color in zip(box_plot["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            plt.ylabel("Execution time [s]", fontsize=12)
            plt.title(
                f"Execution Time Distribution (N={N}, successes only)\n(Boxplot shows median, quartiles, outliers)",
                fontsize=14,
            )
            plt.yscale("log")
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)

            fname = os.path.join(out_dir, f"boxplot_times_N{N}.png")
            plt.savefig(fname, bbox_inches="tight", dpi=300)
            plt.close()
            print(f"Time boxplot N={N}: {fname}")

        plt.figure(figsize=(14, 8))
        iter_data = []
        iter_labels = []

        if "SA" in raw_runs[N]:
            sa_steps = [run["steps"] for run in raw_runs[N]["SA"] if run["success"]]
            if sa_steps:
                iter_data.append(sa_steps)
                iter_labels.append("SA (steps)")

        for fitness in sorted(all_results.keys()):
            if fitness in raw_runs[N]:
                ga_gens = [run["gen"] for run in raw_runs[N][fitness] if run["success"]]
                if ga_gens:
                    iter_data.append(ga_gens)
                    iter_labels.append(f"GA-F{fitness} (gen)")

        if iter_data:
            box_plot = plt.boxplot(iter_data, labels=iter_labels, patch_artist=True)
            colors = ["#ff7f0e", "#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
            for patch, color in zip(box_plot["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            plt.ylabel("Iterations/Generations", fontsize=12)
            plt.title(
                f"Logical Cost Distribution (N={N}, successes only)\n(Algorithm variability in terms of effort)",
                fontsize=14,
            )
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)

            fname = os.path.join(out_dir, f"boxplot_iterations_N{N}.png")
            plt.savefig(fname, bbox_inches="tight", dpi=300)
            plt.close()
            print(f"Iterations boxplot N={N}: {fname}")

        if "SA" in raw_runs[N]:
            sa_times = [run["time"] for run in raw_runs[N]["SA"] if run["success"]]
            if len(sa_times) > 5:
                plt.figure(figsize=(12, 6))
                plt.hist(sa_times, bins=min(20, len(sa_times) // 2), alpha=0.7, color="orange", edgecolor="black")
                plt.xlabel("Time [s]", fontsize=12)
                plt.ylabel("Frequency", fontsize=12)
                plt.title(
                    f"SA Time Distribution (N={N})\n(Distribution shape indicates algorithm stability)",
                    fontsize=14,
                )
                plt.grid(True, alpha=0.3)
                mean_time = float(np.mean(sa_times))
                std_time = float(np.std(sa_times))
                plt.axvline(float(mean_time), color="red", linestyle="--", label=f"Mean: {mean_time:.3f}s")
                plt.axvline(float(mean_time + std_time), color="red", linestyle=":", alpha=0.7, label=f"+/-1 sigma: {std_time:.3f}s")
                plt.axvline(float(mean_time - std_time), color="red", linestyle=":", alpha=0.7)
                plt.legend()
                fname = os.path.join(out_dir, f"histogram_SA_times_N{N}.png")
                plt.savefig(fname, bbox_inches="tight", dpi=300)
                plt.close()
                print(f"SA time histogram N={N}: {fname}")

        best_fitness = min(all_results.keys(), key=lambda f: -all_results[f]["GA"][N]["success_rate"])
        if best_fitness in raw_runs[N]:
            ga_times = [run["time"] for run in raw_runs[N][best_fitness] if run["success"]]
            if len(ga_times) > 5:
                plt.figure(figsize=(12, 6))
                plt.hist(ga_times, bins=min(20, len(ga_times) // 2), alpha=0.7, color="green", edgecolor="black")
                plt.xlabel("Time [s]", fontsize=12)
                plt.ylabel("Frequency", fontsize=12)
                plt.title(
                    f"GA-F{best_fitness} Time Distribution (N={N})\n(More stable algorithm = tighter distribution)",
                    fontsize=14,
                )
                plt.grid(True, alpha=0.3)
                mean_time = float(np.mean(ga_times))
                std_time = float(np.std(ga_times))
                plt.axvline(float(mean_time), color="red", linestyle="--", label=f"Mean: {mean_time:.3f}s")
                plt.axvline(float(mean_time + std_time), color="red", linestyle=":", alpha=0.7, label=f"+/-1 sigma: {std_time:.3f}s")
                plt.axvline(float(mean_time - std_time), color="red", linestyle=":", alpha=0.7)
                plt.legend()
                fname = os.path.join(out_dir, f"histogram_GA_F{best_fitness}_times_N{N}.png")
                plt.savefig(fname, bbox_inches="tight", dpi=300)
                plt.close()
                print(f"GA-F{best_fitness} time histogram N={N}: {fname}")

    print("Statistical analysis completed")


def plot_and_save(results: ExperimentResults, N_values: List[int], fitness_mode: str, out_dir: str) -> None:
    if not _PLOTS_AVAILABLE:
        print("Plotting skipped: matplotlib not installed.")
        return
    plot_comprehensive_analysis(results, N_values, fitness_mode, out_dir)
