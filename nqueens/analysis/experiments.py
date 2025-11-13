"""Final experiment runners for BT/SA/GA (sequential and parallel).

These routines execute repeatable batches of runs for Backtracking (BT),
Simulated Annealing (SA), and Genetic Algorithm (GA) given a set of board
sizes and, for GA, a mapping of tuned hyperparameters per N.

Outputs are structured dictionaries suitable for CSV export and plotting.
Validation hooks optionally check solution correctness and consistency of
reported metrics.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Tuple, cast
import inspect

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
from nqueens.backtracking import (
    bt_nqueens_first,
    bt_nqueens_mcv,
    bt_nqueens_lcv,
    bt_nqueens_mcv_hybrid,
)
from nqueens.simulated_annealing import sa_nqueens
from nqueens.genetic import ga_nqueens
from nqueens.utils import is_valid_solution


# Reusable workers -----------------------------------------------------------

def run_single_sa_experiment(params: Tuple[int, int, float, float]):
    """Worker wrapper to invoke a single SA run (for parallel mapping)."""
    N, max_iter, T0, alpha = params
    return sa_nqueens(N, max_iter=max_iter, T0=T0, alpha=alpha, time_limit=SA_TIME_LIMIT)


def run_single_ga_experiment(params: Tuple[int, int, int, float, float, int, str]):
    """Worker wrapper to invoke a single GA run (for parallel mapping)."""
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

def _discover_bt_solvers() -> List[Tuple[str, Any]]:
    import nqueens.backtracking as bt_mod
    solvers = [(name, fn) for name, fn in inspect.getmembers(bt_mod, inspect.isfunction) if name.startswith("bt_nqueens_")]
    # Map to short labels by removing prefix
    labelled = []
    for name, fn in solvers:
        label = name[len("bt_nqueens_"):]
        labelled.append((label, fn))
    # Preferred order if present
    priority = {"mcv_hybrid": 0, "mcv": 1, "lcv": 2, "first": 3}
    labelled.sort(key=lambda x: (priority.get(x[0], 100), x[0]))
    return labelled


def run_experiments_with_best_ga(
    N_values: List[int],
    runs_sa: int,
    runs_ga: int,
    bt_time_limit: Optional[float],
    fitness_mode: str,
    best_ga_params_for_N: Dict[int, Dict[str, Any]],
    progress_label: Optional[str] = None,
    validate: bool = False,
    include_bt: bool = True,
    include_sa: bool = True,
    include_ga: bool = True,
    bt_solvers: Optional[List[str]] = None,
) -> ExperimentResults:
    """Run sequential final experiments with tuned GA hyperparameters.

    For each N in ``N_values``, this function performs:
    - One deterministic BT run (timeout-aware).
    - ``runs_sa`` SA runs with a size-dependent iteration cap.
    - ``runs_ga`` GA runs using the tuned parameters for that N.
    """
    results: Any = {"BT": {}, "SA": {}, "GA": {}}
    progress = ProgressPrinter(len(N_values), progress_label) if progress_label else None

    interrupted = False
    for index, N in enumerate(N_values, start=1):
        try:
            if progress:
                progress.update(index, f"N={N}")
            if include_ga:
                print(f"=== (Final) N = {N}, GA fitness {fitness_mode} ===")
            else:
                enabled = "+".join([name for name, flag in (("BT", include_bt), ("SA", include_sa)) if flag]) or "NONE"
                print(f"=== (Final) N = {N}, {enabled} ===")

        if include_bt:
            discovered = _discover_bt_solvers()
            if bt_solvers:
                wanted = {s.strip() for s in bt_solvers}
                available = {label for label, _ in discovered}
                unknown = wanted.difference(available)
                if unknown:
                    raise ValueError("Unknown BT solver(s): " + ", ".join(sorted(unknown)) + ". Available: " + ", ".join(sorted(available)))
                selected = [(label, fn) for label, fn in discovered if label in wanted]
            else:
                selected = discovered

            if selected:
                print("  Running BT solvers: " + ", ".join(label for label, _ in selected) + "...")

            bt_results: Dict[str, Dict[str, Any]] = {}
            for label, fn in selected:
                sol, nodes, t = fn(N, time_limit=bt_time_limit)
                if validate and sol is not None and not is_valid_solution(sol):
                    raise AssertionError(f"Invalid BT solution produced for N={N} by {label}: {sol}")
                bt_results[label] = {"solution_found": sol is not None, "nodes": nodes, "time": t}
            results["BT"][N] = bt_results
        else:
            results["BT"][N] = {}

        if include_sa:
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

            if validate:
                for idx, run in enumerate(sa_runs):
                    if run["success"]:
                        if run["best_conflicts"] != 0 or run["timeout"]:
                            raise AssertionError(
                                f"SA validation failed for N={N}, run {idx}: success but best_conflicts={run['best_conflicts']}, timeout={run['timeout']}"
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
        else:
            results["SA"][N] = {
                "success_rate": 0.0,
                "timeout_rate": 0.0,
                "failure_rate": 0.0,
                "total_runs": 0,
                "successes": 0,
                "failures": 0,
                "timeouts": 0,
                "success_steps": {},
                "success_time": {},
                "success_evals": {},
                "success_best_conflicts": {},
                "timeout_steps": {},
                "timeout_time": {},
                "timeout_evals": {},
                "timeout_best_conflicts": {},
                "failure_steps": {},
                "failure_time": {},
                "failure_evals": {},
                "failure_best_conflicts": {},
                "all_steps": {},
                "all_time": {},
                "all_evals": {},
                "all_best_conflicts": {},
                "raw_runs": [],
            }

        if include_ga:
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

            if validate:
                for idx, run in enumerate(ga_runs):
                    if run["success"]:
                        if run["best_conflicts"] != 0 or run["timeout"]:
                            raise AssertionError(
                                f"GA validation failed for N={N}, run {idx}: success but best_conflicts={run['best_conflicts']}, timeout={run['timeout']}"
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
        else:
            results["GA"][N] = {
                "success_rate": 0.0,
                "timeout_rate": 0.0,
                "failure_rate": 0.0,
                "total_runs": 0,
                "successes": 0,
                "failures": 0,
                "timeouts": 0,
                "success_gen": {},
                "success_time": {},
                "success_evals": {},
                "success_best_conflicts": {},
                "timeout_gen": {},
                "timeout_time": {},
                "timeout_evals": {},
                "timeout_best_conflicts": {},
                "failure_gen": {},
                "failure_time": {},
                "failure_evals": {},
                "failure_best_conflicts": {},
                "all_gen": {},
                "all_time": {},
                "all_evals": {},
                "all_best_conflicts": {},
                "pop_size": 0,
                "max_gen": 0,
                "pm": 0.0,
                "pc": 0.0,
                "tournament_size": 0,
                "raw_runs": [],
            }
        except KeyboardInterrupt:
            print("\nInterrupted by user (sequential). Returning partial results...")
            interrupted = True
            break

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
    validate: bool = False,
    include_bt: bool = True,
    include_sa: bool = True,
    include_ga: bool = True,
    bt_solvers: Optional[List[str]] = None,
) -> ExperimentResults:
    """Parallel version of final experiments using process pools.

    SA and GA runs are distributed across processes according to
    ``NUM_PROCESSES`` to accelerate experimentation on multi-core systems.
    """
    results: Any = {"BT": {}, "SA": {}, "GA": {}}
    progress = ProgressPrinter(len(N_values), progress_label) if progress_label else None

    interrupted = False
    for index, N in enumerate(N_values, start=1):
        if progress:
            progress.update(index, f"N={N}")
        try:
            if include_ga:
                print(f"=== (Final Parallel) N = {N}, GA fitness {fitness_mode} ===")
            else:
                enabled = "+".join([name for name, flag in (("BT", include_bt), ("SA", include_sa)) if flag]) or "NONE"
                print(f"=== (Final Parallel) N = {N}, {enabled} ===")

        if include_bt:
            discovered = _discover_bt_solvers()
            if bt_solvers:
                wanted = {s.strip() for s in bt_solvers}
                available = {label for label, _ in discovered}
                unknown = wanted.difference(available)
                if unknown:
                    raise ValueError("Unknown BT solver(s): " + ", ".join(sorted(unknown)) + ". Available: " + ", ".join(sorted(available)))
                selected = [(label, fn) for label, fn in discovered if label in wanted]
            else:
                selected = discovered

            if selected:
                print("  Running BT solvers: " + ", ".join(label for label, _ in selected) + "...")

            bt_results: Dict[str, Dict[str, Any]] = {}
            for label, fn in selected:
                sol, nodes, t = fn(N, time_limit=bt_time_limit)
                if validate and sol is not None and not is_valid_solution(sol):
                    raise AssertionError(f"Invalid BT solution produced (parallel) for N={N} by {label}: {sol}")
                bt_results[label] = {"solution_found": sol is not None, "nodes": nodes, "time": t}
            results["BT"][N] = bt_results
        else:
            results["BT"][N] = {}

        if include_sa:
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

            if validate:
                for idx, run in enumerate(sa_runs):
                    if run["success"]:
                        if run["best_conflicts"] != 0 or run["timeout"]:
                            raise AssertionError(
                                f"SA validation failed (parallel) for N={N}, run {idx}: success but best_conflicts={run['best_conflicts']}, timeout={run['timeout']}"
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
        else:
            results["SA"][N] = {
                "success_rate": 0.0,
                "timeout_rate": 0.0,
                "failure_rate": 0.0,
                "total_runs": 0,
                "successes": 0,
                "failures": 0,
                "timeouts": 0,
                "success_steps": {},
                "success_time": {},
                "success_evals": {},
                "success_best_conflicts": {},
                "timeout_steps": {},
                "timeout_time": {},
                "timeout_evals": {},
                "timeout_best_conflicts": {},
                "failure_steps": {},
                "failure_time": {},
                "failure_evals": {},
                "failure_best_conflicts": {},
                "all_steps": {},
                "all_time": {},
                "all_evals": {},
                "all_best_conflicts": {},
                "raw_runs": [],
            }

        if include_ga:
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

            if validate:
                for idx, run in enumerate(ga_runs):
                    if run["success"]:
                        if run["best_conflicts"] != 0 or run["timeout"]:
                            raise AssertionError(
                                f"GA validation failed (parallel) for N={N}, run {idx}: success but best_conflicts={run['best_conflicts']}, timeout={run['timeout']}"
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
        else:
            results["GA"][N] = {
                "success_rate": 0.0,
                "timeout_rate": 0.0,
                "failure_rate": 0.0,
                "total_runs": 0,
                "successes": 0,
                "failures": 0,
                "timeouts": 0,
                "success_gen": {},
                "success_time": {},
                "success_evals": {},
                "success_best_conflicts": {},
                "timeout_gen": {},
                "timeout_time": {},
                "timeout_evals": {},
                "timeout_best_conflicts": {},
                "failure_gen": {},
                "failure_time": {},
                "failure_evals": {},
                "failure_best_conflicts": {},
                "all_gen": {},
                "all_time": {},
                "all_evals": {},
                "all_best_conflicts": {},
                "pop_size": 0,
                "max_gen": 0,
                "pm": 0.0,
                "pc": 0.0,
                "tournament_size": 0,
                "raw_runs": [],
            }
        except KeyboardInterrupt:
            print("\nInterrupted by user (parallel). Returning partial results...")
            interrupted = True
            break

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
    """Backward-compatible alias retaining the old public name."""
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
