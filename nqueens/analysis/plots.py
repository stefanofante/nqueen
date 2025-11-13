"""Visualization utilities for analysis outputs.

Overview
--------
This module contains plotting helpers that generate PNG charts from the
aggregated experiment results produced by the analysis pipeline. It is designed
to degrade gracefully: if the plotting stack (matplotlib/numpy, optionally
seaborn) is unavailable, public functions will emit a short message and return
without raising so upstream pipelines can continue.

Inputs and data contract
------------------------
- The primary input is an ``ExperimentResults`` mapping with three top-level
    keys, ``"BT"``, ``"SA"``, and ``"GA"``, each containing per-N aggregates.
- Functions also accept an ordered list of ``N_values`` that determines the
    x-axis for most charts.

Outputs and naming
------------------
- Charts are written as PNG files into ``out_dir``. Filenames are prefixed by a
    two-digit index (01..09) where applicable for stable ordering and include an
    optional suffix reflecting algorithm presence, run tag, and date, according to
    ``nqueens.analysis.settings``.

Notes
-----
- The module uses only side effects (file creation, stdout prints); return
    values are ``None``.
- Some scatter plots that correlate logical vs practical cost are only emitted
    in sequential mode to avoid wall-clock noise from parallel execution.

Chart map (filenames → contenuto e significato)
----------------------------------------------
Combined (tutti gli algoritmi insieme):
- 01_success_rate_vs_N.png — Success rate vs N
    - What: Reliability as problem size grows.
    - X: N (board size). Y: Success rate in [0, 1].
    - Metric: successes / total_runs per algorithm; BT is 1.0 when the canonical solver found a solution, else 0.0.
- 02_time_vs_N_log_scale.png — Avg time (success only, log scale) vs N
    - What: Practical runtime growth for successful runs only (hardware dependent).
    - X: N. Y: Average wall-clock time [s] on a log scale.
    - Metric: mean(success_time) for SA/GA; for BT, time of the successful run or 0 when no success.
- 03_logical_cost_vs_N.png — Logical cost (BT nodes / SA steps / GA generations) vs N
    - What: Hardware-independent effort proxy per algorithm.
    - X: N. Y: Logical cost (log scale): BT explored nodes; SA iterations (steps); GA generations.
- 04_fitness_evaluations_vs_N.png — Objective evaluations vs N (SA/GA)
    - What: Pure evaluation burden, often the dominant cost.
    - X: N. Y: Average number of objective/fitness evaluations (log scale).
    - Note: Only SA/GA; BT omitted.
- 05_timeout_rate_vs_N.png — Timeout rate vs N (SA/GA)
    - What: Where algorithms start exceeding the time budget.
    - X: N. Y: Timeout fraction in [0, 1]. Only SA/GA.
- 06_failure_quality_vs_N.png — Average conflicts in failed runs vs N (SA/GA)
    - What: Solution quality when failing (proximity to optimal, 0 conflicts is optimal).
    - X: N. Y: Average best-conflicts among failed runs. Only SA/GA.
- 07_all_theoretical_vs_practical.png — Correlazione costo teorico vs pratico (tutti) (solo in sequential)
    - What: Overlay of theoretical cost (x) vs practical time (y) for BT/SA/GA.
    - X: BT nodes / SA steps / GA fitness evaluations. Y: Time [s].
    - Note: Only meaningful in sequential mode to reduce wall-clock noise.
- 07_SA_theoretical_vs_practical.png — SA: steps vs time (solo in sequential)
    - What: Linearity indicates evaluation-cost dominance for SA.
    - X: SA iterations (steps). Y: Time [s].
- 08_GA_theoretical_vs_practical.png — GA: evaluations vs time (solo in sequential)
    - What: Linearity indicates evaluation-cost dominance for GA.
    - X: GA fitness evaluations. Y: Time [s].
- 09_BT_theoretical_vs_practical.png — BT: nodes vs time (solo in sequential)
    - What: Correlates explored nodes with time; near-linearity expected.
    - X: BT explored nodes. Y: Time [s] (wall-clock).
- 09b_BT_time_per_node_vs_N.png — BT: time per node vs N (diagnostico, sequential)
    - What: Diagnostic of proportionality: time-per-node should be roughly flat vs N.
    - X: N. Y: time/nodes [s/node].

Per algoritmo (solo BT/SA/GA):
- 01_success_rate_vs_N_[BT|SA|GA].png — Success rate vs N
    - Same as combined 01, filtered per algorithm.
- 02_time_vs_N_log_scale_[BT|SA|GA].png — Avg time (success only, log scale) vs N
    - Same as combined 02, filtered per algorithm.
- 03_logical_cost_vs_N_[BT|SA|GA].png — Logical cost vs N
    - Same as combined 03, filtered per algorithm.
- 04_fitness_evaluations_vs_N_[SA|GA].png — Objective/Fitness evaluations vs N
    - Same as combined 04, per SA or GA only.
- 05_timeout_rate_vs_N_[SA|GA].png — Timeout rate vs N
    - Same as combined 05, per SA or GA only.
- 06_failure_quality_vs_N_[SA|GA].png — Failure quality vs N
    - Same as combined 06, per SA or GA only.

Confronto fra fitness (GA):
- fitness_success_rate_N{N}.png — Success rate per fitness (bar)
    - What: Which GA fitness function is more reliable at fixed N.
    - X: Fitness id (F1..F6). Y: Success rate.
- fitness_generations_N{N}.png — Generazioni: media ± std per fitness (bar)
    - What: Convergence speed of GA by fitness at fixed N.
    - X: Fitness id. Y: Mean ± std of generations (success-only).
- fitness_time_N{N}.png — Tempo: media ± std per fitness (bar)
    - What: Practical efficiency of GA by fitness at fixed N.
    - X: Fitness id. Y: Mean ± std of time [s] (success-only).
- fitness_evolution_all.png — Success rate vs N per fitness (linee)
    - What: Reliability evolution across N for each fitness function.
    - X: N. Y: Success rate.

Analisi statistica (raw runs):
- boxplot_times_N{N}.png — Distribuzione tempi (successi)
    - What: Spread and outliers of execution time for all algorithms (success-only).
    - X: Algorithm/Fitness label. Y: Time [s] (log scale).
- boxplot_iterations_N{N}.png — Distribuzione steps/gen (successi)
    - What: Spread of logical cost (SA steps, GA generations) across runs.
    - X: Algorithm/Fitness label. Y: Steps/Generations.
- histogram_SA_times_N{N}.png — Istogramma tempi SA (se campioni sufficienti)
    - What: Distribution shape for SA time; mean and ±1σ annotated.
- histogram_GA_F{F}_times_N{N}.png — Istogramma tempi GA fitness migliore (se campioni sufficienti)
    - What: Distribution shape for GA time using the best fitness at N; mean and ±1σ annotated.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, cast
from . import settings

try:
    import matplotlib.pyplot as plt  # type: ignore
    import numpy as np  # type: ignore
    _PLOTS_AVAILABLE = True
except Exception:
    plt = cast(Any, None)  # type: ignore
    np = cast(Any, None)  # type: ignore
    _PLOTS_AVAILABLE = False

from .stats import ExperimentResults


def _detect_presence(results: ExperimentResults, N_values: List[int]) -> tuple[bool, bool, bool]:
    """Return booleans indicating whether BT/SA/GA data are present.

    Parameters
    ----------
    results : ExperimentResults
        Aggregated results mapping containing optional "BT", "SA", "GA" keys.
    N_values : List[int]
        N sizes to probe for presence.

    Returns
    -------
    tuple[bool, bool, bool]
        A triple ``(has_bt, has_sa, has_ga)``.

    Raises
    ------
    None
    """
    has_bt = any(bool(results.get("BT", {}).get(N)) for N in N_values)
    def _has_runs(sub: str) -> bool:
        for N in N_values:
            d = results.get(sub, {}).get(N, {})
            if d.get("total_runs", 0) > 0 or len(d.get("raw_runs", [])) > 0:
                return True
        return False
    has_sa = _has_runs("SA")
    has_ga = _has_runs("GA")
    return has_bt, has_sa, has_ga


def _collect_bt_solver_labels(results: ExperimentResults, N_values: List[int]) -> List[str]:
    """Collect distinct Backtracking solver labels observed across N.

    Parameters
    ----------
    results : ExperimentResults
        Aggregated results containing the "BT" section.
    N_values : List[int]
        Problem sizes to scan for solver labels.

    Returns
    -------
    List[str]
        Distinct solver labels in a stable discovery order.

    Raises
    ------
    None
    """
    labels: List[str] = []
    seen = set()
    for N in N_values:
        bt_entry = results.get("BT", {}).get(N, {})
        if isinstance(bt_entry.get("solution_found") if isinstance(bt_entry, dict) else None, bool):
            for l in ["mcv_hybrid"]:
                if l not in seen:
                    seen.add(l)
                    labels.append(l)
        else:
            for l in bt_entry.keys():
                if l not in seen:
                    seen.add(l)
                    labels.append(l)
    return labels


def _build_suffix(results: ExperimentResults, N_values: List[int]) -> str:
    """Build a filename suffix based on executed algorithms and settings.

    Parameters
    ----------
    results : ExperimentResults
        Aggregated results used to infer which subsystems ran.
    N_values : List[int]
        N sizes considered; used to detect presence across entries.

    Returns
    -------
    str
        A leading ``_`` followed by components, or an empty string when no
        suffixing is configured.

    Raises
    ------
    None
    """
    parts: List[str] = []
    if getattr(settings, "ALG_IN_FILENAMES", False):
        has_bt, has_sa, has_ga = _detect_presence(results, N_values)
        alg_parts: List[str] = []
        if has_bt:
            bt_labels = _collect_bt_solver_labels(results, N_values)
            alg_parts.append("BT-" + "+".join(bt_labels) if bt_labels else "BT")
        if has_sa:
            alg_parts.append("SA")
        if has_ga:
            alg_parts.append("GA")
        if alg_parts:
            parts.append("-".join(alg_parts))
    # Optional run tag
    run_tag = getattr(settings, "RUN_TAG", None)
    if run_tag:
        parts.append(str(run_tag))
    # Optional datestamp
    if getattr(settings, "DATE_IN_FILENAMES", False):
        run_id = getattr(settings, "RUN_ID", None)
        if run_id:
            parts.append(str(run_id))
    return ("_" + "_".join(parts)) if parts else ""

def _date_suffix() -> str:
    """Return a simple date/time suffix based on settings (or empty).

    Returns
    -------
    str
        A leading ``_`` followed by ``RUN_TAG`` and/or ``RUN_ID`` if enabled;
        otherwise an empty string.

    Raises
    ------
    None
    """
    parts: List[str] = []
    run_tag = getattr(settings, "RUN_TAG", None)
    if run_tag:
        parts.append(str(run_tag))
    if getattr(settings, "DATE_IN_FILENAMES", False):
        run_id = getattr(settings, "RUN_ID", None)
        if run_id:
            parts.append(str(run_id))
    return ("_" + "_".join(parts)) if parts else ""
def _bt_canonical_entry(results: ExperimentResults, N: int) -> Dict[str, Any]:
    """Return a canonical BT entry for a given N supporting legacy/new shapes.

    Parameters
    ----------
    results : ExperimentResults
        Aggregated results including the "BT" section.
    N : int
        Problem size key.

    Returns
    -------
    Dict[str, Any]
        A single solver entry with at least ``solution_found``, ``nodes``, ``time``.

    Notes
    -----
    - Legacy shape: a flat dict at ``results["BT"][N]`` is returned directly.
    - New shape: a per-solver dict; the function selects a preferred solver
      (mcv_hybrid > mcv > lcv > first) or falls back to the first available.

    Raises
    ------
    None
    """
    entry = results["BT"][N]
    if isinstance(entry.get("solution_found") if isinstance(entry, dict) else None, bool):
        return cast(Dict[str, Any], entry)  # legacy single-solver structure
    # New per-solver dict: pick preferred solver if available
    priority = ["mcv_hybrid", "mcv", "lcv", "first"]
    for key in priority:
        if key in entry:
            return entry[key]
    # Fallback: take arbitrary first
    first_key = next(iter(entry))
    return entry[first_key]

try:
    import seaborn as sns  # type: ignore  # noqa: F401
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
    """Generate a full set of per-algorithm and combined charts.

    Parameters
    ----------
    results : ExperimentResults
        Aggregated per-N summaries for BT/SA/GA produced by the analysis
        pipeline.
    N_values : List[int]
        Ordered list of N values to display on the x-axis.
    fitness_mode : str
        Fitness identifier used for GA plots (e.g., "1".."6"); used in labels
        and filenames when GA data is present.
    out_dir : str
        Destination directory; will be created if missing.
    raw_runs : Optional[Dict[str, Any]], optional
        Optional raw run records (not strictly required by this function);
        included for symmetry with other plotting helpers.
    tuning_data : Optional[Dict[str, Any]], optional
        Optional tuning metadata; currently unused but accepted for forward
        compatibility.

    Returns
    -------
    None
        Images are written to ``out_dir``; nothing is returned.

    Raises
    ------
    None
    """
    if not _PLOTS_AVAILABLE:
        print("Plotting skipped: matplotlib not installed.")
        return
    os.makedirs(out_dir, exist_ok=True)

    # Detect presence of algorithms and build filename suffix
    has_bt, has_sa, has_ga = _detect_presence(results, N_values)
    suffix = (f"_F{fitness_mode}" if has_ga else "") + _build_suffix(results, N_values)

    bt_sr = [1.0 if has_bt and _bt_canonical_entry(results, N)["solution_found"] else 0.0 for N in N_values]
    sa_sr = [cast(float, results["SA"][N].get("success_rate", 0.0) or 0.0) for N in N_values] if has_sa else [0.0 for _ in N_values]
    ga_sr = [cast(float, results["GA"][N].get("success_rate", 0.0) or 0.0) for N in N_values] if has_ga else [0.0 for _ in N_values]

    bt_timeout = [0.0 for _ in N_values]
    sa_timeout = [results["SA"][N].get("timeout_rate", 0.0) for N in N_values] if has_sa else [0.0 for _ in N_values]
    ga_timeout = [results["GA"][N].get("timeout_rate", 0.0) for N in N_values] if has_ga else [0.0 for _ in N_values]

    plt.figure(figsize=(12, 8))
    if has_bt:
        plt.plot(N_values, bt_sr, marker="o", linewidth=2, markersize=8, label="Backtracking")
    if has_sa:
        plt.plot(N_values, sa_sr, marker="s", linewidth=2, markersize=8, label="Simulated Annealing")
    if has_ga:
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
        if has_ga:
            plt.annotate(f"{ga_sr[i]:.2f}", (n, ga_sr[i]), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=9, color="green")

    fname = os.path.join(out_dir, f"01_success_rate_vs_N{suffix}.png")
    plt.savefig(fname, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved success-rate chart: {fname}")

    bt_time = [_bt_canonical_entry(results, N)["time"] if (has_bt and _bt_canonical_entry(results, N)["solution_found"]) else 0 for N in N_values]
    sa_time = [cast(float, results["SA"][N].get("success_time", {}).get("mean", 0.0) or 0.0) for N in N_values] if has_sa else [0.0 for _ in N_values]
    ga_time = [cast(float, results["GA"][N].get("success_time", {}).get("mean", 0.0) or 0.0) for N in N_values] if has_ga else [0.0 for _ in N_values]

    plt.figure(figsize=(12, 8))
    bt_time_plot = [max(t, 1e-6) for t in bt_time]
    sa_time_plot = [max(t, 1e-6) for t in sa_time]
    ga_time_plot = [max(t, 1e-6) for t in ga_time]

    if has_bt:
        plt.semilogy(N_values, bt_time_plot, marker="o", linewidth=2, markersize=8, label="Backtracking")
    if has_sa:
        plt.semilogy(N_values, sa_time_plot, marker="s", linewidth=2, markersize=8, label="Simulated Annealing")
    if has_ga:
        plt.semilogy(N_values, ga_time_plot, marker="^", linewidth=2, markersize=8, label=f"Genetic Algorithm (F{fitness_mode})")
    plt.xlabel("N (board size)", fontsize=12)
    plt.ylabel("Average time [s] (log scale)", fontsize=12)
    plt.title("Execution Time vs Problem Size\n(Successful runs only — highlights BT growth)", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.7)
    plt.xticks(N_values)

    fname = os.path.join(out_dir, f"02_time_vs_N_log_scale{suffix}.png")
    plt.savefig(fname, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved execution-time chart (log scale): {fname}")

    bt_nodes = [_bt_canonical_entry(results, N)["nodes"] for N in N_values] if has_bt else [0 for _ in N_values]
    sa_steps = [cast(float, results["SA"][N].get("success_steps", {}).get("mean", 0.0) or 0.0) for N in N_values] if has_sa else [0.0 for _ in N_values]
    ga_gen = [cast(float, results["GA"][N].get("success_gen", {}).get("mean", 0.0) or 0.0) for N in N_values] if has_ga else [0.0 for _ in N_values]

    plt.figure(figsize=(12, 8))
    if has_bt:
        plt.semilogy(N_values, [max(n, 1) for n in bt_nodes], marker="o", linewidth=2, markersize=8, label="BT: Explored nodes")
    if has_sa:
        plt.semilogy(N_values, [max(s, 1) for s in sa_steps], marker="s", linewidth=2, markersize=8, label="SA: Average iterations")
    if has_ga:
        plt.semilogy(N_values, [max(g, 1) for g in ga_gen], marker="^", linewidth=2, markersize=8, label="GA: Average generations")
    plt.xlabel("N (board size)", fontsize=12)
    plt.ylabel("Logical cost (log scale)", fontsize=12)
    plt.title("Theoretical Computational Cost vs Problem Size\n(Hardware-independent scalability)", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.7)
    plt.xticks(N_values)

    fname = os.path.join(out_dir, f"03_logical_cost_vs_N{suffix}.png")
    plt.savefig(fname, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved logical-cost chart: {fname}")

    sa_evals = [cast(float, results["SA"][N].get("success_evals", {}).get("mean", 0.0) or 0.0) for N in N_values] if has_sa else [0.0 for _ in N_values]
    ga_evals = [cast(float, results["GA"][N].get("success_evals", {}).get("mean", 0.0) or 0.0) for N in N_values] if has_ga else [0.0 for _ in N_values]

    if has_sa or has_ga:
        plt.figure(figsize=(12, 8))
        if has_sa:
            plt.semilogy(N_values, [max(e, 1) for e in sa_evals], marker="s", linewidth=2, markersize=8, label="SA: Conflict evaluations")
        if has_ga:
            plt.semilogy(N_values, [max(e, 1) for e in ga_evals], marker="^", linewidth=2, markersize=8, label=f"GA-F{fitness_mode}: Fitness evaluations")
        plt.xlabel("N (board size)", fontsize=12)
        plt.ylabel("Objective evaluations (log scale)", fontsize=12)
        plt.title("Pure Objective Evaluation Cost\n(Computational burden of evaluations)", fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.7)
        plt.xticks(N_values)

        fname = os.path.join(out_dir, f"04_fitness_evaluations_vs_N{suffix}.png")
        plt.savefig(fname, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"Saved fitness-evaluation chart: {fname}")

    if has_sa or has_ga:
        plt.figure(figsize=(12, 8))
        if has_sa:
            plt.plot(N_values, [results["SA"][N].get("timeout_rate", 0.0) for N in N_values], marker="s", linewidth=2, markersize=8, label="SA: Timeout rate")
        if has_ga:
            plt.plot(N_values, [results["GA"][N].get("timeout_rate", 0.0) for N in N_values], marker="^", linewidth=2, markersize=8, label=f"GA-F{fitness_mode}: Timeout rate")
        plt.xlabel("N (board size)", fontsize=12)
        plt.ylabel("Timeout rate", fontsize=12)
        plt.title("Timeout Rate vs Problem Size\n(Where algorithms start exceeding the time limit)", fontsize=14)
        plt.ylim(-0.05, 1.05)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.7)
        plt.xticks(N_values)

        fname = os.path.join(out_dir, f"05_timeout_rate_vs_N{suffix}.png")
        plt.savefig(fname, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"Saved timeout-rate chart: {fname}")

    sa_fail_quality = [
        (lambda v: float(v) if isinstance(v, (int, float)) else float(N))(results["SA"][N].get("failure_best_conflicts", {}).get("mean"))
        if has_sa and results["SA"][N].get("failure_best_conflicts") else float(N)
        for N in N_values
    ] if has_sa else [0.0 for _ in N_values]
    ga_fail_quality = [
        (lambda v: float(v) if isinstance(v, (int, float)) else float(N))(results["GA"][N].get("failure_best_conflicts", {}).get("mean"))
        if has_ga and results["GA"][N].get("failure_best_conflicts") else float(N)
        for N in N_values
    ] if has_ga else [0 for _ in N_values]

    if has_sa or has_ga:
        plt.figure(figsize=(12, 8))
        if has_sa:
            plt.plot(N_values, sa_fail_quality, marker="s", linewidth=2, markersize=8, label="SA: Average conflicts (failures)")
        if has_ga:
            plt.plot(N_values, ga_fail_quality, marker="^", linewidth=2, markersize=8, label=f"GA-F{fitness_mode}: Average conflicts (failures)")
        plt.plot(N_values, [0] * len(N_values), "k--", alpha=0.5, label="Optimal solution (0 conflicts)")
        plt.xlabel("N (board size)", fontsize=12)
        plt.ylabel("Average conflicts in failures", fontsize=12)
        plt.title("Solution Quality in Failed Runs\n(How close to optimal despite failure)", fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.7)
        plt.xticks(N_values)

        fname = os.path.join(out_dir, f"06_failure_quality_vs_N{suffix}.png")
        plt.savefig(fname, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"Saved failure-quality chart: {fname}")

    # Combined theoretical vs practical correlation (all algorithms)
    # Meaningful only in sequential mode (to avoid wall-clock noise)
    if settings.CURRENT_PIPELINE_MODE == 'sequential':
        combined_points = 0
        plt.figure(figsize=(12, 8))

        # SA points
        if has_sa and any(sa_steps) and any(sa_time):
            sa_valid = [(s, t) for s, t in zip(sa_steps, sa_time) if s > 0 and t > 0]
            if sa_valid:
                xs, ys = zip(*sa_valid)
                plt.scatter(xs, ys, c='orange', marker='s', s=90, alpha=0.8, label='SA (steps)')
                combined_points += len(sa_valid)

        # GA points
        if has_ga and any(ga_evals) and any(ga_time):
            ga_valid = [(e, t) for e, t in zip(ga_evals, ga_time) if e > 0 and t > 0]
            if ga_valid:
                xs, ys = zip(*ga_valid)
                plt.scatter(xs, ys, c='green', marker='^', s=90, alpha=0.8, label=f'GA-F{fitness_mode} (evals)')
                combined_points += len(ga_valid)

        # BT points
        if has_bt and any(bt_nodes) and any(bt_time):
            bt_valid = [(n, t) for n, t in zip(bt_nodes, bt_time) if n > 0 and t > 0]
            if bt_valid:
                xs, ys = zip(*bt_valid)
                plt.scatter(xs, ys, c='blue', marker='o', s=90, alpha=0.8, label='BT (nodes)')
                combined_points += len(bt_valid)

        if combined_points > 0:
            plt.xlabel('Theoretical cost (steps/evals/nodes)', fontsize=12)
            plt.ylabel('Time [s] (practical cost)', fontsize=12)
            plt.title('Theoretical vs Practical Cost — All Algorithms\n(Sequential mode for clearer proportionality)', fontsize=14)
            plt.grid(True, alpha=0.7)
            plt.legend(fontsize=11)

            fname = os.path.join(out_dir, f"07_all_theoretical_vs_practical{suffix}.png")
            plt.savefig(fname, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"Saved ALL-algorithms theoretical-vs-practical chart: {fname}")
        else:
            plt.close()

    # SA correlation meaningful in sequential mode only
    if settings.CURRENT_PIPELINE_MODE == 'sequential' and has_sa and any(sa_steps) and any(sa_time):
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

        fname = os.path.join(out_dir, f"07_SA_theoretical_vs_practical{suffix}.png")
        plt.savefig(fname, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"Saved SA theoretical-vs-practical chart: {fname}")
    elif has_sa:
        print("Skipping SA theoretical-vs-practical correlation in parallel/concurrent mode.")

    # GA correlation meaningful in sequential mode only
    if settings.CURRENT_PIPELINE_MODE == 'sequential' and has_ga and any(ga_evals) and any(ga_time):
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

        fname = os.path.join(out_dir, f"08_GA_theoretical_vs_practical{suffix}.png")
        plt.savefig(fname, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"Saved GA theoretical-vs-practical chart: {fname}")
    elif has_ga:
        print("Skipping GA theoretical-vs-practical correlation in parallel/concurrent mode.")

    # Skip BT scatter in parallel/concurrent modes as wall-clock noise breaks proportionality
    if settings.CURRENT_PIPELINE_MODE == 'sequential' and has_bt and any(bt_nodes) and any(bt_time):
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
        plt.ylabel("Time [s] (wall-clock)", fontsize=12)
        plt.title("Backtracking: Nodes vs Time (wall-clock)\n(For clearer linearity, run sequential mode)", fontsize=14)
        plt.grid(True, alpha=0.7)

        fname = os.path.join(out_dir, f"09_BT_theoretical_vs_practical{suffix}.png")
        plt.savefig(fname, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"Saved BT theoretical-vs-practical chart: {fname}")

        # Additional diagnostic: time-per-node across N (should be ~constant)
        tpn = [ (t / n) if (n > 0 and t > 0) else 0.0 for n, t in zip(bt_nodes, bt_time) ]
        if any(tpn):
            plt.figure(figsize=(12, 6))
            plt.plot(N_values, tpn, marker="o", linewidth=2)
            plt.xlabel("N (board size)", fontsize=12)
            plt.ylabel("Time per node [s/node]", fontsize=12)
            plt.title("Backtracking: Time per Node vs N\n(Flat line indicates proportionality)", fontsize=14)
            plt.grid(True, alpha=0.7)
            fname2 = os.path.join(out_dir, f"09b_BT_time_per_node_vs_N{suffix}.png")
            plt.savefig(fname2, bbox_inches="tight", dpi=300)
            plt.close()
            print(f"Saved BT time-per-node chart: {fname2}")
    elif has_bt:
        print("Skipping BT nodes vs time scatter (Plot 9) in parallel/concurrent mode.")

    # Per-algorithm focused charts
    # 1) Success rate per algorithm
    if has_bt:
        plt.figure(figsize=(10, 6))
        plt.plot(N_values, bt_sr, marker="o", linewidth=2, markersize=7, label="BT")
        plt.xlabel("N (board size)")
        plt.ylabel("Success rate")
        plt.title("BT — Success Rate vs N")
        plt.ylim(-0.05, 1.05)
        plt.grid(True, alpha=0.6)
        fname_bt = os.path.join(out_dir, f"01_success_rate_vs_N_BT{suffix}.png")
        plt.savefig(fname_bt, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"Saved BT-only success-rate chart: {fname_bt}")

    if has_sa:
        plt.figure(figsize=(10, 6))
        plt.plot(N_values, sa_sr, marker="s", linewidth=2, markersize=7, label="SA", color="orange")
        plt.xlabel("N (board size)")
        plt.ylabel("Success rate")
        plt.title("SA — Success Rate vs N")
        plt.ylim(-0.05, 1.05)
        plt.grid(True, alpha=0.6)
        fname_sa = os.path.join(out_dir, f"01_success_rate_vs_N_SA{suffix}.png")
        plt.savefig(fname_sa, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"Saved SA-only success-rate chart: {fname_sa}")

    if has_ga:
        plt.figure(figsize=(10, 6))
        plt.plot(N_values, ga_sr, marker="^", linewidth=2, markersize=7, label=f"GA-F{fitness_mode}", color="green")
        plt.xlabel("N (board size)")
        plt.ylabel("Success rate")
        plt.title(f"GA-F{fitness_mode} — Success Rate vs N")
        plt.ylim(-0.05, 1.05)
        plt.grid(True, alpha=0.6)
        fname_ga = os.path.join(out_dir, f"01_success_rate_vs_N_GA{suffix}.png")
        plt.savefig(fname_ga, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"Saved GA-only success-rate chart: {fname_ga}")

    # 2) Time vs N per algorithm (log scale)
    if has_bt:
        plt.figure(figsize=(10, 6))
        plt.semilogy(N_values, [max(t, 1e-6) for t in bt_time], marker="o", linewidth=2, markersize=7, label="BT")
        plt.xlabel("N (board size)")
        plt.ylabel("Average time [s] (log scale)")
        plt.title("BT — Execution Time vs N (success only)")
        plt.grid(True, alpha=0.6)
        fname_bt_t = os.path.join(out_dir, f"02_time_vs_N_log_scale_BT{suffix}.png")
        plt.savefig(fname_bt_t, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"Saved BT-only time chart: {fname_bt_t}")

    if has_sa:
        plt.figure(figsize=(10, 6))
        plt.semilogy(N_values, [max(t, 1e-6) for t in sa_time], marker="s", linewidth=2, markersize=7, label="SA", color="orange")
        plt.xlabel("N (board size)")
        plt.ylabel("Average time [s] (log scale)")
        plt.title("SA — Execution Time vs N (success only)")
        plt.grid(True, alpha=0.6)
        fname_sa_t = os.path.join(out_dir, f"02_time_vs_N_log_scale_SA{suffix}.png")
        plt.savefig(fname_sa_t, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"Saved SA-only time chart: {fname_sa_t}")

    if has_ga:
        plt.figure(figsize=(10, 6))
        plt.semilogy(N_values, [max(t, 1e-6) for t in ga_time], marker="^", linewidth=2, markersize=7, label=f"GA-F{fitness_mode}", color="green")
        plt.xlabel("N (board size)")
        plt.ylabel("Average time [s] (log scale)")
        plt.title(f"GA-F{fitness_mode} — Execution Time vs N (success only)")
        plt.grid(True, alpha=0.6)
        fname_ga_t = os.path.join(out_dir, f"02_time_vs_N_log_scale_GA{suffix}.png")
        plt.savefig(fname_ga_t, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"Saved GA-only time chart: {fname_ga_t}")

    # 3) Logical cost vs N per algorithm
    if has_bt:
        plt.figure(figsize=(10, 6))
        plt.semilogy(N_values, [max(n, 1) for n in bt_nodes], marker="o", linewidth=2, markersize=7, label="BT: nodes")
        plt.xlabel("N (board size)")
        plt.ylabel("Logical cost (log scale)")
        plt.title("BT — Logical Cost vs N (explored nodes)")
        plt.grid(True, alpha=0.6)
        fname_bt_l = os.path.join(out_dir, f"03_logical_cost_vs_N_BT{suffix}.png")
        plt.savefig(fname_bt_l, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"Saved BT-only logical-cost chart: {fname_bt_l}")

    if has_sa:
        plt.figure(figsize=(10, 6))
        plt.semilogy(N_values, [max(s, 1) for s in sa_steps], marker="s", linewidth=2, markersize=7, label="SA: iterations", color="orange")
        plt.xlabel("N (board size)")
        plt.ylabel("Logical cost (log scale)")
        plt.title("SA — Logical Cost vs N (iterations)")
        plt.grid(True, alpha=0.6)
        fname_sa_l = os.path.join(out_dir, f"03_logical_cost_vs_N_SA{suffix}.png")
        plt.savefig(fname_sa_l, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"Saved SA-only logical-cost chart: {fname_sa_l}")

    if has_ga:
        plt.figure(figsize=(10, 6))
        plt.semilogy(N_values, [max(g, 1) for g in ga_gen], marker="^", linewidth=2, markersize=7, label="GA: generations", color="green")
        plt.xlabel("N (board size)")
        plt.ylabel("Logical cost (log scale)")
        plt.title(f"GA-F{fitness_mode} — Logical Cost vs N (generations)")
        plt.grid(True, alpha=0.6)
        fname_ga_l = os.path.join(out_dir, f"03_logical_cost_vs_N_GA{suffix}.png")
        plt.savefig(fname_ga_l, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"Saved GA-only logical-cost chart: {fname_ga_l}")

    # 4/5/6 specific to SA/GA already handled above in combined charts; also emit SA/GA only versions
    if has_sa and any(sa_evals):
        plt.figure(figsize=(10, 6))
        plt.semilogy(N_values, [max(e, 1) for e in sa_evals], marker="s", linewidth=2, markersize=7, color="orange")
        plt.xlabel("N (board size)")
        plt.ylabel("Objective evaluations (log scale)")
        plt.title("SA — Conflict Evaluations vs N")
        plt.grid(True, alpha=0.6)
        fname_sa_ev = os.path.join(out_dir, f"04_fitness_evaluations_vs_N_SA{suffix}.png")
        plt.savefig(fname_sa_ev, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"Saved SA-only evaluation chart: {fname_sa_ev}")

    if has_ga and any(ga_evals):
        plt.figure(figsize=(10, 6))
        plt.semilogy(N_values, [max(e, 1) for e in ga_evals], marker="^", linewidth=2, markersize=7, color="green")
        plt.xlabel("N (board size)")
        plt.ylabel("Objective evaluations (log scale)")
        plt.title(f"GA-F{fitness_mode} — Fitness Evaluations vs N")
        plt.grid(True, alpha=0.6)
        fname_ga_ev = os.path.join(out_dir, f"04_fitness_evaluations_vs_N_GA{suffix}.png")
        plt.savefig(fname_ga_ev, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"Saved GA-only evaluation chart: {fname_ga_ev}")

    if has_sa:
        plt.figure(figsize=(10, 6))
        plt.plot(N_values, sa_timeout, marker="s", linewidth=2, markersize=7, color="orange")
        plt.xlabel("N (board size)")
        plt.ylabel("Timeout rate")
        plt.title("SA — Timeout Rate vs N")
        plt.ylim(-0.05, 1.05)
        plt.grid(True, alpha=0.6)
        fname_sa_to = os.path.join(out_dir, f"05_timeout_rate_vs_N_SA{suffix}.png")
        plt.savefig(fname_sa_to, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"Saved SA-only timeout chart: {fname_sa_to}")

    if has_ga:
        plt.figure(figsize=(10, 6))
        plt.plot(N_values, ga_timeout, marker="^", linewidth=2, markersize=7, color="green")
        plt.xlabel("N (board size)")
        plt.ylabel("Timeout rate")
        plt.title(f"GA-F{fitness_mode} — Timeout Rate vs N")
        plt.ylim(-0.05, 1.05)
        plt.grid(True, alpha=0.6)
        fname_ga_to = os.path.join(out_dir, f"05_timeout_rate_vs_N_GA{suffix}.png")
        plt.savefig(fname_ga_to, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"Saved GA-only timeout chart: {fname_ga_to}")

    if has_sa and any(sa_fail_quality):
        plt.figure(figsize=(10, 6))
        plt.plot(N_values, sa_fail_quality, marker="s", linewidth=2, markersize=7, color="orange")
        plt.xlabel("N (board size)")
        plt.ylabel("Average conflicts in failures")
        plt.title("SA — Failure Quality vs N")
        plt.grid(True, alpha=0.6)
        fname_sa_fq = os.path.join(out_dir, f"06_failure_quality_vs_N_SA{suffix}.png")
        plt.savefig(fname_sa_fq, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"Saved SA-only failure-quality chart: {fname_sa_fq}")

    if has_ga:
        plt.figure(figsize=(10, 6))
        plt.plot(N_values, ga_fail_quality, marker="^", linewidth=2, markersize=7, color="green")
        plt.xlabel("N (board size)")
        plt.ylabel("Average conflicts in failures")
        plt.title(f"GA-F{fitness_mode} — Failure Quality vs N")
        plt.grid(True, alpha=0.6)
        fname_ga_fq = os.path.join(out_dir, f"06_failure_quality_vs_N_GA{suffix}.png")
        plt.savefig(fname_ga_fq, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"Saved GA-only failure-quality chart: {fname_ga_fq}")

    print(f"\nComplete analysis generated in: {out_dir}")
    print("Generated per-algorithm and merged charts.")


def plot_fitness_comparison(
    all_results: Dict[str, ExperimentResults], N_values: List[int], out_dir: str, raw_runs: Optional[Dict[str, Any]] = None
) -> None:
    """Compare GA performance across fitness functions.

    Parameters
    ----------
    all_results : Dict[str, ExperimentResults]
        Mapping ``fitness_id -> ExperimentResults`` from separate GA runs.
    N_values : List[int]
        N sizes to include on plots. A subset may be selected internally for
        clarity on some charts.
    out_dir : str
        Destination directory; will be created if missing.
    raw_runs : Optional[Dict[str, Any]], optional
        Optional raw run records for advanced charts; not strictly required.

    Returns
    -------
    None
        Images are written to ``out_dir``; nothing is returned.

    Raises
    ------
    None
    """
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

        fname = os.path.join(out_dir, f"fitness_success_rate_N{N}{_date_suffix()}.png")
        plt.savefig(fname, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"Saved success-rate comparison for N={N}: {fname}")

        plt.figure(figsize=(12, 8))
        gen_means = []
        gen_stds = []

        for f in fitness_modes:
            gen_stats = cast(Dict[str, Any], all_results[f]["GA"][N].get("success_gen", {}))
            gen_means.append(cast(float, gen_stats.get("mean", 0) or 0))
            gen_stds.append(cast(float, gen_stats.get("std", 0) or 0))

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

        fname = os.path.join(out_dir, f"fitness_generations_N{N}{_date_suffix()}.png")
        plt.savefig(fname, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"Saved generation comparison for N={N}: {fname}")

        plt.figure(figsize=(12, 8))
        time_means = []
        time_stds = []

        for f in fitness_modes:
            time_stats = cast(Dict[str, Any], all_results[f]["GA"][N].get("success_time", {}))
            time_means.append(cast(float, time_stats.get("mean", 0) or 0))
            time_stds.append(cast(float, time_stats.get("std", 0) or 0))

        bars = plt.bar(
            fitness_modes, time_means, yerr=time_stds, color=[fitness_colors[f] for f in fitness_modes], alpha=0.8, capsize=5
        )
        plt.xlabel("Fitness Function", fontsize=12)
        plt.ylabel("Average time [s] +/- std", fontsize=12)
        plt.title(f"Temporal Efficiency Comparison (N={N})\n(Trade-off between success and speed)", fontsize=14)
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

        fname = os.path.join(out_dir, f"fitness_time_N{N}{_date_suffix()}.png")
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

    fname = os.path.join(out_dir, f"fitness_evolution_all{_date_suffix()}.png")
    plt.savefig(fname, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved fitness evolution overview: {fname}")

    print(f"\nFitness function comparison analysis completed")
    print(f"Generated comparison charts for F1-F6")


def plot_statistical_analysis(
    all_results: Dict[str, ExperimentResults], N_values: List[int], out_dir: str, raw_runs: Optional[Dict[int, Dict[str, Any]]] = None
) -> None:
    """Produce boxplots and histograms leveraging raw run distributions.

    Parameters
    ----------
    all_results : Dict[str, ExperimentResults]
        Aggregated summaries per fitness function, used to select the best
        fitness for additional histograms.
    N_values : List[int]
        Set of N values considered; a subset is selected for readability.
    out_dir : str
        Destination directory; will be created if missing.
    raw_runs : Optional[Dict[int, Dict[str, Any]]], optional
        Raw run data indexed by N (then by "SA" and fitness ids). Required to
        build distributions; if absent, the function returns early.

    Returns
    -------
    None
        Images are written to ``out_dir``; nothing is returned.

    Raises
    ------
    None
    """
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
            _plt = cast(Any, plt)
            box_plot = _plt.boxplot(time_data, labels=labels, patch_artist=True)
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

            fname = os.path.join(out_dir, f"boxplot_times_N{N}{_date_suffix()}.png")
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
            _plt = cast(Any, plt)
            box_plot = _plt.boxplot(iter_data, labels=iter_labels, patch_artist=True)
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

            fname = os.path.join(out_dir, f"boxplot_iterations_N{N}{_date_suffix()}.png")
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
                fname = os.path.join(out_dir, f"histogram_SA_times_N{N}{_date_suffix()}.png")
                plt.savefig(fname, bbox_inches="tight", dpi=300)
                plt.close()
                print(f"SA time histogram N={N}: {fname}")

        best_fitness = min(all_results.keys(), key=lambda f: -cast(float, all_results[f]["GA"][N].get("success_rate", 0.0) or 0.0))
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
                fname = os.path.join(out_dir, f"histogram_GA_F{best_fitness}_times_N{N}{_date_suffix()}.png")
                plt.savefig(fname, bbox_inches="tight", dpi=300)
                plt.close()
                print(f"GA-F{best_fitness} time histogram N={N}: {fname}")

    print("Statistical analysis completed")


def plot_and_save(results: ExperimentResults, N_values: List[int], fitness_mode: str, out_dir: str) -> None:
    """Thin wrapper to generate the comprehensive analysis figures.

    Parameters
    ----------
    results : ExperimentResults
        Aggregated per-N summaries for BT/SA/GA.
    N_values : List[int]
        Ordered list of N values to plot.
    fitness_mode : str
        Fitness identifier used for GA labeling.
    out_dir : str
        Destination directory for the saved figures.

    Returns
    -------
    None
        Images are written to ``out_dir``; nothing is returned.

    Raises
    ------
    None
    """
    if not _PLOTS_AVAILABLE:
        print("Plotting skipped: matplotlib not installed.")
        return
    plot_comprehensive_analysis(results, N_values, fitness_mode, out_dir)
