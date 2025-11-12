"""Final experiments runners for BT/SA/GA (sequential and parallel)."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Tuple, cast

from .settings import (
    SA_TIME_LIMIT,
    GA_TIME_LIMIT,
    BT_TIME_LIMIT,
    NUM_PROCESSES,
)
from .stats import (
    ExperimentResults,
    compute_grouped_statistics,
    ProgressPrinter,
)
from nqueens.backtracking import bt_nqueens_first
from nqueens.simulated_annealing import sa_nqueens
from nqueens.genetic import ga_nqueens


# Reusable workers -----------------------------------------------------------

def run_single_sa_experiment(params: Tuple[int, int, float, float]):
    N, max_iter, T0, alpha = params
    return sa_nqueens(N, max_iter=max_iter, T0=T0, alpha=alpha, time_limit=SA_TIME_LIMIT)


def run_single_ga_experiment(params: Tuple[int, int, int, float, float, int, str]):
    N, pop_size, max_gen, pc, pm, tournament_size, fitness_mode = params
    return ga_nqueens(
        N,
        pop_size=pop_size,
        max_gen=max_gen,
        pc=pc,
        pm=pm,
        tournament_size=tournament_size,
        fitness_mode=fitness_mode,
        time_limit=GA_TIME_LIMIT,
    )


# Sequential runner ----------------------------------------------------------

def run_experiments_with_best_ga(
    N_values: List[int],
    runs_sa: int,
    runs_ga: int,
    bt_time_limit: Optional[float],
    fitness_mode: str,
    best_ga_params_for_N: Dict[int, Dict[str, Any]],
    progress_label: Optional[str] = None,
) -> ExperimentResults:
    results = cast(ExperimentResults, {"BT": {}, "SA": {}, "GA": {}})
    progress = ProgressPrinter(len(N_values), progress_label) if progress_label else None

    for index, N in enumerate(N_values, start=1):
        if progress:
            progress.update(index, f"N={N}")
        print(f"=== (Final) N = {N}, GA fitness {fitness_mode} ===")

        sol, nodes, t = bt_nqueens_first(N, time_limit=bt_time_limit)
        results["BT"][N] = {"solution_found": sol is not None, "nodes": nodes, "time": t}

        sa_runs: List[Dict[str, Any]] = []
        max_iter_sa = 2000 + 200 * N
        for _ in range(runs_sa):
            s, steps, tt, bestc, evals, timeout = sa_nqueens(
                N, max_iter=max_iter_sa, T0=1.0, alpha=0.995, time_limit=SA_TIME_LIMIT
            )
            sa_runs.append(
                {
                    "success": s,
                    "steps": steps,
                    "time": tt,
                    "best_conflicts": bestc,
                    "evals": evals,
                    "timeout": timeout,
                }
            )

        sa_stats = compute_grouped_statistics(sa_runs, "success")
        results["SA"][N] = {
            "success_rate": sa_stats["success_rate"],
            "timeout_rate": sa_stats["timeout_rate"],
            "failure_rate": sa_stats["failure_rate"],
            "total_runs": sa_stats["total_runs"],
            "successes": sa_stats["successes"],
            "failures": sa_stats["failures"],
            "timeouts": sa_stats["timeouts"],
            "success_steps": sa_stats.get("success_steps", {}),
            "success_time": sa_stats.get("success_time", {}),
            "success_evals": sa_stats.get("success_evals", {}),
            "success_best_conflicts": sa_stats.get("success_best_conflicts", {}),
            "timeout_steps": sa_stats.get("timeout_steps", {}),
            "timeout_time": sa_stats.get("timeout_time", {}),
            "timeout_evals": sa_stats.get("timeout_evals", {}),
            "timeout_best_conflicts": sa_stats.get("timeout_best_conflicts", {}),
            "failure_steps": sa_stats.get("failure_steps", {}),
            "failure_time": sa_stats.get("failure_time", {}),
            "failure_evals": sa_stats.get("failure_evals", {}),
            "failure_best_conflicts": sa_stats.get("failure_best_conflicts", {}),
            "all_steps": sa_stats.get("all_steps", {}),
            "all_time": sa_stats.get("all_time", {}),
            "all_evals": sa_stats.get("all_evals", {}),
            "all_best_conflicts": sa_stats.get("all_best_conflicts", {}),
            "raw_runs": sa_runs.copy(),
        }

        params = best_ga_params_for_N[N]
        pop_size = params["pop_size"]
        max_gen = params["max_gen"]
        pm = params["pm"]
        pc = params["pc"]
        tsize = params["tournament_size"]

        ga_runs: List[Dict[str, Any]] = []
        for _ in range(runs_ga):
            s, gen, tt, bestc, evals, timeout = ga_nqueens(
                N,
                pop_size=pop_size,
                max_gen=max_gen,
                pc=pc,
                pm=pm,
                tournament_size=tsize,
                fitness_mode=fitness_mode,
                time_limit=GA_TIME_LIMIT,
            )
            ga_runs.append(
                {
                    "success": s,
                    "gen": gen,
                    "time": tt,
                    "best_conflicts": bestc,
                    "evals": evals,
                    "timeout": timeout,
                }
            )

        ga_stats = compute_grouped_statistics(ga_runs, "success")
        results["GA"][N] = {
            "success_rate": ga_stats["success_rate"],
            "timeout_rate": ga_stats["timeout_rate"],
            "failure_rate": ga_stats["failure_rate"],
            "total_runs": ga_stats["total_runs"],
            "successes": ga_stats["successes"],
            "failures": ga_stats["failures"],
            "timeouts": ga_stats["timeouts"],
            "success_gen": ga_stats.get("success_gen", {}),
            "success_time": ga_stats.get("success_time", {}),
            "success_evals": ga_stats.get("success_evals", {}),
            "success_best_conflicts": ga_stats.get("success_best_conflicts", {}),
            "timeout_gen": ga_stats.get("timeout_gen", {}),
            "timeout_time": ga_stats.get("timeout_time", {}),
            "timeout_evals": ga_stats.get("timeout_evals", {}),
            "timeout_best_conflicts": ga_stats.get("timeout_best_conflicts", {}),
            "failure_gen": ga_stats.get("failure_gen", {}),
            "failure_time": ga_stats.get("failure_time", {}),
            "failure_evals": ga_stats.get("failure_evals", {}),
            "failure_best_conflicts": ga_stats.get("failure_best_conflicts", {}),
            "all_gen": ga_stats.get("all_gen", {}),
            "all_time": ga_stats.get("all_time", {}),
            "all_evals": ga_stats.get("all_evals", {}),
            "all_best_conflicts": ga_stats.get("all_best_conflicts", {}),
            "pop_size": pop_size,
            "max_gen": max_gen,
            "pm": pm,
            "pc": pc,
            "tournament_size": tsize,
            "raw_runs": ga_runs.copy(),
        }

    return results


# Parallel runner ------------------------------------------------------------

def run_experiments_with_best_ga_parallel(
    N_values: List[int],
    runs_sa: int,
    runs_ga: int,
    bt_time_limit: Optional[float],
    fitness_mode: str,
    best_ga_params_for_N: Dict[int, Dict[str, Any]],
    progress_label: Optional[str] = None,
) -> ExperimentResults:
    results = cast(ExperimentResults, {"BT": {}, "SA": {}, "GA": {}})
    progress = ProgressPrinter(len(N_values), progress_label) if progress_label else None

    for index, N in enumerate(N_values, start=1):
        if progress:
            progress.update(index, f"N={N}")
        print(f"=== (Final Parallel) N = {N}, GA fitness {fitness_mode} ===")

        sol, nodes, t = bt_nqueens_first(N, time_limit=bt_time_limit)
        results["BT"][N] = {"solution_found": sol is not None, "nodes": nodes, "time": t}

        print(f"  Running {runs_sa} SA runs in parallel...")
        max_iter_sa = 2000 + 200 * N
        sa_params = [(N, max_iter_sa, 1.0, 0.995) for _ in range(runs_sa)]

        with ProcessPoolExecutor(max_workers=NUM_PROCESSES) as executor:
            sa_raw_results = list(executor.map(run_single_sa_experiment, sa_params))

        sa_runs: List[Dict[str, Any]] = []
        for s, steps, tt, bestc, evals, timeout in sa_raw_results:
            sa_runs.append(
                {
                    "success": s,
                    "steps": steps,
                    "time": tt,
                    "best_conflicts": bestc,
                    "evals": evals,
                    "timeout": timeout,
                }
            )

        sa_stats = compute_grouped_statistics(sa_runs, "success")
        results["SA"][N] = {
            "success_rate": sa_stats["success_rate"],
            "timeout_rate": sa_stats["timeout_rate"],
            "failure_rate": sa_stats["failure_rate"],
            "total_runs": sa_stats["total_runs"],
            "successes": sa_stats["successes"],
            "failures": sa_stats["failures"],
            "timeouts": sa_stats["timeouts"],
            "success_steps": sa_stats.get("success_steps", {}),
            "success_time": sa_stats.get("success_time", {}),
            "success_evals": sa_stats.get("success_evals", {}),
            "success_best_conflicts": sa_stats.get("success_best_conflicts", {}),
            "timeout_steps": sa_stats.get("timeout_steps", {}),
            "timeout_time": sa_stats.get("timeout_time", {}),
            "timeout_evals": sa_stats.get("timeout_evals", {}),
            "timeout_best_conflicts": sa_stats.get("timeout_best_conflicts", {}),
            "failure_steps": sa_stats.get("failure_steps", {}),
            "failure_time": sa_stats.get("failure_time", {}),
            "failure_evals": sa_stats.get("failure_evals", {}),
            "failure_best_conflicts": sa_stats.get("failure_best_conflicts", {}),
            "all_steps": sa_stats.get("all_steps", {}),
            "all_time": sa_stats.get("all_time", {}),
            "all_evals": sa_stats.get("all_evals", {}),
            "all_best_conflicts": sa_stats.get("all_best_conflicts", {}),
            "raw_runs": sa_runs.copy(),
        }

        print(f"  Running {runs_ga} GA runs in parallel...")
        params = best_ga_params_for_N[N]
        pop_size = params["pop_size"]
        max_gen = params["max_gen"]
        pm = params["pm"]
        pc = params["pc"]
        tsize = params["tournament_size"]

        ga_params = [(N, pop_size, max_gen, pc, pm, tsize, fitness_mode) for _ in range(runs_ga)]

        with ProcessPoolExecutor(max_workers=NUM_PROCESSES) as executor:
            ga_raw_results = list(executor.map(run_single_ga_experiment, ga_params))

        ga_runs: List[Dict[str, Any]] = []
        for s, gen, tt, bestc, evals, timeout in ga_raw_results:
            ga_runs.append(
                {
                    "success": s,
                    "gen": gen,
                    "time": tt,
                    "best_conflicts": bestc,
                    "evals": evals,
                    "timeout": timeout,
                }
            )

        ga_stats = compute_grouped_statistics(ga_runs, "success")
        results["GA"][N] = {
            "success_rate": ga_stats["success_rate"],
            "timeout_rate": ga_stats["timeout_rate"],
            "failure_rate": ga_stats["failure_rate"],
            "total_runs": ga_stats["total_runs"],
            "successes": ga_stats["successes"],
            "failures": ga_stats["failures"],
            "timeouts": ga_stats["timeouts"],
            "success_gen": ga_stats.get("success_gen", {}),
            "success_time": ga_stats.get("success_time", {}),
            "success_evals": ga_stats.get("success_evals", {}),
            "success_best_conflicts": ga_stats.get("success_best_conflicts", {}),
            "timeout_gen": ga_stats.get("timeout_gen", {}),
            "timeout_time": ga_stats.get("timeout_time", {}),
            "timeout_evals": ga_stats.get("timeout_evals", {}),
            "timeout_best_conflicts": ga_stats.get("timeout_best_conflicts", {}),
            "failure_gen": ga_stats.get("failure_gen", {}),
            "failure_time": ga_stats.get("failure_time", {}),
            "failure_evals": ga_stats.get("failure_evals", {}),
            "failure_best_conflicts": ga_stats.get("failure_best_conflicts", {}),
            "all_gen": ga_stats.get("all_gen", {}),
            "all_time": ga_stats.get("all_time", {}),
            "all_evals": ga_stats.get("all_evals", {}),
            "all_best_conflicts": ga_stats.get("all_best_conflicts", {}),
            "pop_size": pop_size,
            "max_gen": max_gen,
            "pm": pm,
            "pc": pc,
            "tournament_size": tsize,
            "raw_runs": ga_runs.copy(),
        }

    return results


# Alias for backward compatibility ------------------------------------------

def run_experiments_parallel(
    N_values: List[int],
    runs_bt: int,
    runs_sa: int,
    runs_ga: int,
    bt_time_limit: Optional[float],
    fitness_mode: str,
    best_ga_params_for_N: Dict[int, Dict[str, Any]],
    progress_label: Optional[str] = None,
) -> ExperimentResults:
    del runs_bt  # BT is always deterministic here
    return run_experiments_with_best_ga_parallel(
        N_values=N_values,
        runs_sa=runs_sa,
        runs_ga=runs_ga,
        bt_time_limit=bt_time_limit,
        fitness_mode=fitness_mode,
        best_ga_params_for_N=best_ga_params_for_N,
        progress_label=progress_label,
    )
