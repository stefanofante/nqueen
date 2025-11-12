"""CSV export utilities for experiment outputs (aggregates and raw runs).

These helpers materialize concise CSV summaries as well as full per-run raw
data for downstream analysis or spreadsheet inspection. Filenames include the
GA fitness label to disambiguate multi-fitness experiments.
"""
from __future__ import annotations

import csv
import os
from typing import Any, Dict, List

from .stats import ExperimentResults


def _detect_presence(results: ExperimentResults, N_values: List[int]) -> tuple[bool, bool, bool]:
    """Infer which subsystems have actual runs across provided N values.

    Returns (has_bt, has_sa, has_ga).
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


def save_results_to_csv(results: ExperimentResults, N_values: List[int], fitness_mode: str, out_dir: str) -> None:
    """Write compact per-N aggregate metrics for BT/SA/GA to CSV.

    Column names follow lowercase snake_case with subsystem prefixes:
    - bt_* for Backtracking, sa_* for Simulated Annealing, ga_* for Genetic Algorithm.
    """
    os.makedirs(out_dir, exist_ok=True)
    has_bt, has_sa, has_ga = _detect_presence(results, N_values)
    if not has_ga:
        if has_bt and has_sa:
            filename = os.path.join(out_dir, "results_BT_SA.csv")
        elif has_sa:
            filename = os.path.join(out_dir, "results_SA.csv")
        else:
            filename = os.path.join(out_dir, "results_BT.csv")
    else:
        filename = os.path.join(out_dir, f"results_GA_{fitness_mode}_tuned.csv")

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "n",
            # Backtracking (per-solver and backward-compatible aggregate)
            "bt_solution_found",
            "bt_nodes_explored",
            "bt_time_seconds",
            "bt_first_solution_found",
            "bt_first_nodes_explored",
            "bt_first_time_seconds",
            "bt_mcv_solution_found",
            "bt_mcv_nodes_explored",
            "bt_mcv_time_seconds",
            "bt_lcv_solution_found",
            "bt_lcv_nodes_explored",
            "bt_lcv_time_seconds",
            "bt_mcv_hybrid_solution_found",
            "bt_mcv_hybrid_nodes_explored",
            "bt_mcv_hybrid_time_seconds",
            "sa_success_rate",
            "sa_timeout_rate",
            "sa_failure_rate",
            "sa_total_runs",
            "sa_successes",
            "sa_failures",
            "sa_timeouts",
            "sa_success_steps_mean",
            "sa_success_steps_median",
            "sa_success_evals_mean",
            "sa_success_evals_median",
            "sa_timeout_steps_mean",
            "sa_timeout_steps_median",
            "sa_timeout_evals_mean",
            "sa_timeout_evals_median",
            "sa_success_time_mean",
            "sa_success_time_median",
            "ga_success_rate",
            "ga_timeout_rate",
            "ga_failure_rate",
            "ga_total_runs",
            "ga_successes",
            "ga_failures",
            "ga_timeouts",
            "ga_success_gen_mean",
            "ga_success_gen_median",
            "ga_success_evals_mean",
            "ga_success_evals_median",
            "ga_timeout_gen_mean",
            "ga_timeout_gen_median",
            "ga_timeout_evals_mean",
            "ga_timeout_evals_median",
            "ga_success_time_mean",
            "ga_success_time_median",
            "ga_pop_size",
            "ga_max_gen",
            "ga_pm",
            "ga_pc",
            "ga_tournament_size",
        ])

        for N in N_values:
            bt = results["BT"][N]
            # Support both legacy (single dict) and new per-solver dict
            if isinstance(bt.get("solution_found") if isinstance(bt, dict) else None, bool):
                bt_first = bt_mcv = bt_lcv = bt_h = bt  # type: ignore
            else:
                bt_first = bt.get("first", {"solution_found": False, "nodes": 0, "time": 0.0})
                bt_mcv = bt.get("mcv", {"solution_found": False, "nodes": 0, "time": 0.0})
                bt_lcv = bt.get("lcv", {"solution_found": False, "nodes": 0, "time": 0.0})
                bt_h = bt.get("mcv_hybrid", {"solution_found": False, "nodes": 0, "time": 0.0})
            sa = results["SA"][N]
            ga = results["GA"][N]

            sa_steps_success = sa.get("success_steps", {})
            sa_evals_success = sa.get("success_evals", {})
            sa_time_success = sa.get("success_time", {})
            sa_steps_timeout = sa.get("timeout_steps", {})
            sa_evals_timeout = sa.get("timeout_evals", {})

            ga_gen_success = ga.get("success_gen", {})
            ga_evals_success = ga.get("success_evals", {})
            ga_time_success = ga.get("success_time", {})
            ga_gen_timeout = ga.get("timeout_gen", {})
            ga_evals_timeout = ga.get("timeout_evals", {})

            writer.writerow([
                N,
                # Backward-compatible aggregate: use hybrid as canonical
                int(bt_h["solution_found"]),
                bt_h["nodes"],
                bt_h["time"],
                int(bt_first["solution_found"]),
                bt_first["nodes"],
                bt_first["time"],
                int(bt_mcv["solution_found"]),
                bt_mcv["nodes"],
                bt_mcv["time"],
                int(bt_lcv["solution_found"]),
                bt_lcv["nodes"],
                bt_lcv["time"],
                int(bt_h["solution_found"]),
                bt_h["nodes"],
                bt_h["time"],
                sa.get("success_rate", 0.0),
                sa.get("timeout_rate", 0),
                sa.get("failure_rate", 0),
                sa.get("total_runs", 0),
                sa.get("successes", 0),
                sa.get("failures", 0),
                sa.get("timeouts", 0),
                sa_steps_success.get("mean", ""),
                sa_steps_success.get("median", ""),
                sa_evals_success.get("mean", ""),
                sa_evals_success.get("median", ""),
                sa_steps_timeout.get("mean", ""),
                sa_steps_timeout.get("median", ""),
                sa_evals_timeout.get("mean", ""),
                sa_evals_timeout.get("median", ""),
                sa_time_success.get("mean", ""),
                sa_time_success.get("median", ""),
                ga.get("success_rate", 0.0),
                ga.get("timeout_rate", 0),
                ga.get("failure_rate", 0),
                ga.get("total_runs", 0),
                ga.get("successes", 0),
                ga.get("failures", 0),
                ga.get("timeouts", 0),
                ga_gen_success.get("mean", ""),
                ga_gen_success.get("median", ""),
                ga_evals_success.get("mean", ""),
                ga_evals_success.get("median", ""),
                ga_gen_timeout.get("mean", ""),
                ga_gen_timeout.get("median", ""),
                ga_evals_timeout.get("mean", ""),
                ga_evals_timeout.get("median", ""),
                ga_time_success.get("mean", ""),
                ga_time_success.get("median", ""),
                ga.get("pop_size", 0),
                ga.get("max_gen", 0),
                ga.get("pm", 0.0),
                ga.get("pc", 0.0),
                ga.get("tournament_size", 0),
            ])

    print(f"CSV saved: {filename}")


def save_bt_solvers_summary(results: ExperimentResults, N_values: List[int], fitness_mode: str, out_dir: str) -> None:
    """Write a long-format CSV with per-solver BT metrics for each N.

    Columns:
    - n: board size
    - bt_solver: solver label (suffix of function name without 'bt_nqueens_')
    - solution_found, nodes_explored, time_seconds
    """
    os.makedirs(out_dir, exist_ok=True)
    has_bt, has_sa, has_ga = _detect_presence(results, N_values)
    if not has_bt:
        # No BT runs present: skip creating summary entirely
        return
    if not has_ga:
        filename = os.path.join(out_dir, "bt_solvers_summary.csv")
    else:
        filename = os.path.join(out_dir, f"bt_solvers_summary_{fitness_mode}.csv")

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n", "bt_solver", "solution_found", "nodes_explored", "time_seconds"])
        for N in N_values:
            bt_entry = results["BT"][N]
            # Legacy single-entry support: emit one row labelled 'mcv_hybrid'
            if isinstance(bt_entry.get("solution_found") if isinstance(bt_entry, dict) else None, bool):
                writer.writerow([N, "mcv_hybrid", bt_entry["solution_found"], bt_entry["nodes"], bt_entry["time"]])
            else:
                for label, row in bt_entry.items():
                    writer.writerow([N, label, row.get("solution_found", False), row.get("nodes", 0), row.get("time", 0.0)])

    print(f"BT solvers summary saved: {filename}")


def save_raw_data_to_csv(results: ExperimentResults, N_values: List[int], fitness_mode: str, out_dir: str) -> None:
    """Write full per-run raw data for SA, GA, and BT to CSV files.

    Column names are standardized to lowercase snake_case.
    """
    os.makedirs(out_dir, exist_ok=True)
    has_bt, has_sa, has_ga = _detect_presence(results, N_values)

    sa_filename = None
    if has_sa:
        sa_filename = os.path.join(out_dir, (f"raw_data_SA_{fitness_mode}.csv" if has_ga else "raw_data_SA.csv"))
        with open(sa_filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "n",
                "run_id",
                "algorithm",
                "success",
                "timeout",
                "steps",
                "time_seconds",
                "evals",
                "best_conflicts",
            ])
            for N in N_values:
                sa_data = results["SA"][N]
                if "raw_runs" in sa_data:
                    for i, run in enumerate(sa_data["raw_runs"]):
                        writer.writerow([
                            N,
                            i + 1,
                            "SA",
                            run["success"],
                            run["timeout"],
                            run["steps"],
                            run["time"],
                            run["evals"],
                            run["best_conflicts"],
                        ])

    ga_filename = None
    if has_ga:
        ga_filename = os.path.join(out_dir, f"raw_data_GA_{fitness_mode}.csv")
        with open(ga_filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "n",
                "run_id",
                "algorithm",
                "success",
                "timeout",
                "gen",
                "time_seconds",
                "evals",
                "best_fitness",
                "best_conflicts",
                "pop_size",
                "max_gen",
                "pm",
                "pc",
                "tournament_size",
            ])
            for N in N_values:
                ga_data = results["GA"][N]
                if "raw_runs" in ga_data:
                    for i, run in enumerate(ga_data["raw_runs"]):
                        writer.writerow([
                            N,
                            i + 1,
                            "GA",
                            run["success"],
                            run["timeout"],
                            run["gen"],
                            run["time"],
                            run["evals"],
                            run.get("best_fitness", ""),
                            run.get("best_conflicts", ""),
                            ga_data.get("pop_size", 0),
                            ga_data.get("max_gen", 0),
                            ga_data.get("pm", 0.0),
                            ga_data.get("pc", 0.0),
                            ga_data.get("tournament_size", 0),
                        ])

    bt_filename = None
    if has_bt:
        bt_filename = os.path.join(out_dir, (f"raw_data_BT_{fitness_mode}.csv" if has_ga else "raw_data_BT.csv"))
        with open(bt_filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["n", "algorithm", "solution_found", "nodes_explored", "time_seconds"])
            for N in N_values:
                bt_data = results["BT"][N]
                if isinstance(bt_data.get("solution_found") if isinstance(bt_data, dict) else None, bool):
                    rows = [("BT", bt_data)]
                else:
                    rows = [
                        ("BT_FIRST", bt_data.get("first", {"solution_found": False, "nodes": 0, "time": 0.0})),
                        ("BT_MCV", bt_data.get("mcv", {"solution_found": False, "nodes": 0, "time": 0.0})),
                        ("BT_LCV", bt_data.get("lcv", {"solution_found": False, "nodes": 0, "time": 0.0})),
                        ("BT_MCV_HYBRID", bt_data.get("mcv_hybrid", {"solution_found": False, "nodes": 0, "time": 0.0})),
                    ]
                for alg_label, row in rows:
                    writer.writerow([N, alg_label, row["solution_found"], row["nodes"], row["time"]])
    print("Raw data saved:")
    if sa_filename:
        print(f"  SA: {sa_filename}")
    if ga_filename:
        print(f"  GA: {ga_filename}")
    if bt_filename:
        print(f"  BT: {bt_filename}")


def save_logical_cost_analysis(results: ExperimentResults, N_values: List[int], fitness_mode: str, out_dir: str) -> None:
    """Write a CSV focused on hardware-independent 'logical cost' metrics.

    Column names follow lowercase snake_case with bt_/sa_/ga_ prefixes.
    """
    os.makedirs(out_dir, exist_ok=True)
    has_bt, has_sa, has_ga = _detect_presence(results, N_values)
    filename = os.path.join(out_dir, (f"logical_costs_{fitness_mode}.csv" if has_ga else "logical_costs.csv"))

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "n",
            "bt_solution_found",
            "bt_nodes_explored",
            "sa_success_rate",
            "sa_steps_mean_all",
            "sa_steps_median_all",
            "sa_evals_mean_all",
            "sa_evals_median_all",
            "sa_steps_mean_success",
            "sa_evals_mean_success",
            "ga_success_rate",
            "ga_gen_mean_all",
            "ga_gen_median_all",
            "ga_evals_mean_all",
            "ga_evals_median_all",
            "ga_gen_mean_success",
            "ga_evals_mean_success",
            "bt_time_seconds",
            "sa_time_mean_success",
            "ga_time_mean_success",
        ])

        for N in N_values:
            bt = results["BT"][N]
            if isinstance(bt.get("solution_found") if isinstance(bt, dict) else None, bool):
                bt_h = bt
            else:
                bt_h = bt.get("mcv_hybrid", {"solution_found": False, "nodes": 0, "time": 0.0})
            sa = results["SA"][N]
            ga = results["GA"][N]

            sa_all_steps = sa.get("all_steps", {})
            sa_all_evals = sa.get("all_evals", {})
            sa_success_steps = sa.get("success_steps", {})
            sa_success_evals = sa.get("success_evals", {})
            sa_success_time = sa.get("success_time", {})

            ga_all_gen = ga.get("all_gen", {})
            ga_all_evals = ga.get("all_evals", {})
            ga_success_gen = ga.get("success_gen", {})
            ga_success_evals = ga.get("success_evals", {})
            ga_success_time = ga.get("success_time", {})

            writer.writerow([
                N,
                int(bt_h["solution_found"]),
                bt_h["nodes"],
                sa.get("success_rate", 0.0),
                sa_all_steps.get("mean", ""),
                sa_all_steps.get("median", ""),
                sa_all_evals.get("mean", ""),
                sa_all_evals.get("median", ""),
                sa_success_steps.get("mean", ""),
                sa_success_evals.get("mean", ""),
                ga.get("success_rate", 0.0),
                ga_all_gen.get("mean", ""),
                ga_all_gen.get("median", ""),
                ga_all_evals.get("mean", ""),
                ga_all_evals.get("median", ""),
                ga_success_gen.get("mean", ""),
                ga_success_evals.get("mean", ""),
                bt_h["time"],
                sa_success_time.get("mean", ""),
                ga_success_time.get("mean", ""),
            ])

    print(f"Logical cost analysis saved: {filename}")
