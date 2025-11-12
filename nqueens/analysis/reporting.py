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


def save_results_to_csv(results: ExperimentResults, N_values: List[int], fitness_mode: str, out_dir: str) -> None:
    """Write compact per-N aggregate metrics for BT/SA/GA to CSV."""
    os.makedirs(out_dir, exist_ok=True)
    filename = os.path.join(out_dir, f"results_GA_{fitness_mode}_tuned.csv")

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "N",
            "BT_solution_found",
            "BT_nodes_explored",
            "BT_time_seconds",
            "SA_success_rate",
            "SA_timeout_rate",
            "SA_failure_rate",
            "SA_total_runs",
            "SA_successes",
            "SA_failures",
            "SA_timeouts",
            "SA_success_steps_mean",
            "SA_success_steps_median",
            "SA_success_evals_mean",
            "SA_success_evals_median",
            "SA_timeout_steps_mean",
            "SA_timeout_steps_median",
            "SA_timeout_evals_mean",
            "SA_timeout_evals_median",
            "SA_success_time_mean",
            "SA_success_time_median",
            "GA_success_rate",
            "GA_timeout_rate",
            "GA_failure_rate",
            "GA_total_runs",
            "GA_successes",
            "GA_failures",
            "GA_timeouts",
            "GA_success_gen_mean",
            "GA_success_gen_median",
            "GA_success_evals_mean",
            "GA_success_evals_median",
            "GA_timeout_gen_mean",
            "GA_timeout_gen_median",
            "GA_timeout_evals_mean",
            "GA_timeout_evals_median",
            "GA_success_time_mean",
            "GA_success_time_median",
            "GA_pop_size",
            "GA_max_gen",
            "GA_pm",
            "GA_pc",
            "GA_tournament_size",
        ])

        for N in N_values:
            bt = results["BT"][N]
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
                int(bt["solution_found"]),
                bt["nodes"],
                bt["time"],
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


def save_raw_data_to_csv(results: ExperimentResults, N_values: List[int], fitness_mode: str, out_dir: str) -> None:
    """Write full per-run raw data for SA, GA, and BT to CSV files."""
    os.makedirs(out_dir, exist_ok=True)

    sa_filename = os.path.join(out_dir, f"raw_data_SA_{fitness_mode}.csv")
    with open(sa_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "N",
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

    ga_filename = os.path.join(out_dir, f"raw_data_GA_{fitness_mode}.csv")
    with open(ga_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "N",
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

    bt_filename = os.path.join(out_dir, f"raw_data_BT_{fitness_mode}.csv")
    with open(bt_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["N", "algorithm", "solution_found", "nodes_explored", "time_seconds"])
        for N in N_values:
            bt_data = results["BT"][N]
            writer.writerow([N, "BT", bt_data["solution_found"], bt_data["nodes"], bt_data["time"]])

    print("Raw data saved:")
    print(f"  SA: {sa_filename}")
    print(f"  GA: {ga_filename}")
    print(f"  BT: {bt_filename}")


def save_logical_cost_analysis(results: ExperimentResults, N_values: List[int], fitness_mode: str, out_dir: str) -> None:
    """Write a CSV focused on hardware-independent 'logical cost' metrics."""
    os.makedirs(out_dir, exist_ok=True)
    filename = os.path.join(out_dir, f"logical_costs_{fitness_mode}.csv")

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "N",
            "BT_solution_found",
            "BT_nodes_explored",
            "SA_success_rate",
            "SA_steps_mean_all",
            "SA_steps_median_all",
            "SA_evals_mean_all",
            "SA_evals_median_all",
            "SA_steps_mean_success",
            "SA_evals_mean_success",
            "GA_success_rate",
            "GA_gen_mean_all",
            "GA_gen_median_all",
            "GA_evals_mean_all",
            "GA_evals_median_all",
            "GA_gen_mean_success",
            "GA_evals_mean_success",
            "BT_time_seconds",
            "SA_time_mean_success",
            "GA_time_mean_success",
        ])

        for N in N_values:
            bt = results["BT"][N]
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
                int(bt["solution_found"]),
                bt["nodes"],
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
                bt["time"],
                sa_success_time.get("mean", ""),
                ga_success_time.get("mean", ""),
            ])

    print(f"Logical cost analysis saved: {filename}")
