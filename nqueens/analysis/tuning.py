"""GA hyperparameter tuning (sequential and parallel).

Provides exhaustive grid-search routines and parallelized evaluators to select
robust GA hyperparameters for each board size N and fitness function.
"""
from __future__ import annotations

import statistics
from concurrent.futures import ProcessPoolExecutor
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple

from .settings import (
    GA_TIME_LIMIT,
    NUM_PROCESSES,
)
from .stats import ProgressPrinter
from nqueens.genetic import ga_nqueens


# -------- Worker wrappers (must be top-level callables for ProcessPoolExecutor)

def run_single_ga_experiment(params: Tuple[int, int, int, float, float, int, str]):
    """Execute a single GA run with provided parameters (for executors)."""
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


def test_parameter_combination_parallel(params: Tuple[int, str, int, int, float, float, int, int]) -> Dict[str, Any]:
    """Evaluate one GA parameter combination using multiple parallel runs.

    Returns aggregate success rate and average generations to success.
    """
    N, fitness_mode, pop_size, max_gen, pc, pm, tournament_size, runs_tuning = params

    run_params = [
        (N, pop_size, max_gen, pc, pm, tournament_size, fitness_mode) for _ in range(runs_tuning)
    ]

    with ProcessPoolExecutor(max_workers=min(NUM_PROCESSES, runs_tuning)) as executor:
        results = list(executor.map(run_single_ga_experiment, run_params))

    successes = 0
    gen_success: List[int] = []
    for s, gen, _, bestc, _, _ in results:
        if s:
            successes += 1
            gen_success.append(gen)

    success_rate = successes / runs_tuning
    avg_gen = statistics.mean(gen_success) if gen_success else None

    return {
        "N": N,
        "fitness_mode": fitness_mode,
        "pop_size": pop_size,
        "max_gen": max_gen,
        "pc": pc,
        "pm": pm,
        "tournament_size": tournament_size,
        "success_rate": success_rate,
        "avg_gen_success": avg_gen,
    }


# -------- Sequential tuning

def tune_ga_for_N(
    N: int,
    fitness_mode: str,
    pop_multipliers: List[int],
    gen_multipliers: List[int],
    pm_values: List[float],
    pc: float,
    tournament_size: int,
    runs_tuning: int = 10,
) -> Dict[str, Any]:
    """Exhaustive grid search to identify robust GA hyperparameters.

    Iterates over population-size and generation multipliers and mutation rates,
    evaluating each combination ``runs_tuning`` times. Best candidate is chosen
    by success-rate, with average successful generations as a tiebreaker.
    """
    best: Optional[Dict[str, Any]] = None

    for k in pop_multipliers:
        pop_size = max(50, int(k * N))
        for m in gen_multipliers:
            max_gen = int(m * N)
            for pm in pm_values:
                successes = 0
                gen_success: List[int] = []

                for _ in range(runs_tuning):
                    s, gen, _, bestc, _, timeout = ga_nqueens(
                        N,
                        pop_size=pop_size,
                        max_gen=max_gen,
                        pc=pc,
                        pm=pm,
                        tournament_size=tournament_size,
                        fitness_mode=fitness_mode,
                        time_limit=GA_TIME_LIMIT,
                    )
                    if s:
                        successes += 1
                        gen_success.append(gen)

                success_rate = successes / runs_tuning
                avg_gen = statistics.mean(gen_success) if gen_success else None

                candidate = {
                    "N": N,
                    "fitness_mode": fitness_mode,
                    "pop_size": pop_size,
                    "max_gen": max_gen,
                    "pc": pc,
                    "pm": pm,
                    "tournament_size": tournament_size,
                    "success_rate": success_rate,
                    "avg_gen_success": avg_gen,
                }

                if best is None:
                    best = candidate
                else:
                    if candidate["success_rate"] > best["success_rate"]:
                        best = candidate
                    elif candidate["success_rate"] == best["success_rate"]:
                        if candidate["avg_gen_success"] is not None and best["avg_gen_success"] is not None:
                            if candidate["avg_gen_success"] < best["avg_gen_success"]:
                                best = candidate

    if best is None:
        raise RuntimeError("No GA parameter candidate evaluated; check tuning grid.")
    return best


# -------- Parallel tuning

def tune_ga_for_N_parallel(
    N: int,
    fitness_mode: str,
    pop_multipliers: List[int],
    gen_multipliers: List[int],
    pm_values: List[float],
    pc: float,
    tournament_size: int,
    runs_tuning: int = 10,
) -> Dict[str, Any]:
    """Parallel version of GA tuning for a single ``(N, fitness_mode)`` pair."""
    print(
        f"  Preparazione {len(pop_multipliers) * len(gen_multipliers) * len(pm_values)} combinazioni di parametri..."
    )

    param_combinations: List[Tuple[int, str, int, int, float, float, int, int]] = []
    for k in pop_multipliers:
        pop_size = max(50, int(k * N))
        for m in gen_multipliers:
            max_gen = int(m * N)
            for pm in pm_values:
                param_combinations.append(
                    (N, fitness_mode, pop_size, max_gen, pc, pm, tournament_size, runs_tuning)
                )

    print(f"  Esecuzione parallela con {NUM_PROCESSES} processi...")
    with ProcessPoolExecutor(max_workers=NUM_PROCESSES) as executor:
        candidates = list(executor.map(test_parameter_combination_parallel, param_combinations))

    best: Optional[Dict[str, Any]] = None
    for candidate in candidates:
        if best is None:
            best = candidate
        else:
            if candidate["success_rate"] > best["success_rate"]:
                best = candidate
            elif candidate["success_rate"] == best["success_rate"]:
                if candidate["avg_gen_success"] is not None and best["avg_gen_success"] is not None:
                    if candidate["avg_gen_success"] < best["avg_gen_success"]:
                        best = candidate

    if best is None:
        raise RuntimeError("No GA parameter candidate evaluated in parallel; check tuning grid.")
    print(
        f"  Migliore combinazione: pop_size={best['pop_size']}, max_gen={best['max_gen']}, pm={best['pm']}, success_rate={best['success_rate']:.3f}"
    )
    return best


def tune_single_fitness(
    params: Tuple[int, str, List[int], List[int], List[float], float, int, int]
) -> Tuple[str, Dict[str, Any]]:
    """Wrapper to tune GA parameters for a single fitness function."""
    N, fitness_mode, pop_multipliers, gen_multipliers, pm_values, pc, tournament_size, runs_tuning = params
    return (
        fitness_mode,
        tune_ga_for_N_parallel(
            N, fitness_mode, pop_multipliers, gen_multipliers, pm_values, pc, tournament_size, runs_tuning
        ),
    )


def tune_all_fitness_parallel(
    N: int,
    fitness_modes: List[str],
    pop_multipliers: List[int],
    gen_multipliers: List[int],
    pm_values: List[float],
    pc: float,
    tournament_size: int,
    runs_tuning: int = 10,
) -> Dict[str, Dict[str, Any]]:
    """Tune GA parameters for all fitness functions concurrently for a given N."""
    print(f"Tuning contemporaneo di {len(fitness_modes)} fitness per N={N}")

    tuning_params = [
        (N, fitness_mode, pop_multipliers, gen_multipliers, pm_values, pc, tournament_size, runs_tuning)
        for fitness_mode in fitness_modes
    ]

    print(
        f"  Utilizzando {min(NUM_PROCESSES, len(fitness_modes))} processi per {len(fitness_modes)} fitness..."
    )
    start_time = perf_counter()

    with ProcessPoolExecutor(max_workers=min(NUM_PROCESSES, len(fitness_modes))) as executor:
        results = list(executor.map(tune_single_fitness, tuning_params))

    elapsed_time = perf_counter() - start_time

    best_params_per_fitness: Dict[str, Dict[str, Any]] = {}
    for fitness_mode, best_params in results:
        best_params_per_fitness[fitness_mode] = best_params
        print(
            f"  Completato {fitness_mode}: success_rate={best_params['success_rate']:.3f}, pop_size={best_params['pop_size']}, pm={best_params['pm']}"
        )

    print(f"Tuning contemporaneo completato in {elapsed_time:.1f}s per N={N}")
    return best_params_per_fitness
