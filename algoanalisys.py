"""
N-Queens Orchestrator and Analysis Utilities
===========================================

This module provides the end-to-end orchestration, tuning, execution, and
analysis toolkit for solving the N-Queens problem with multiple algorithms:

- Backtracking (deterministic, exact) implementations under `nqueens.backtracking`
- Simulated Annealing (stochastic local search)
- Genetic Algorithm (stochastic evolutionary search) with parameter tuning

It includes:
- Configuration loading and application (timeouts, grids, fitness modes)
- Grid-search and parallel tuning for GA hyperparameters
- Final experiment runners (sequential and parallel) that collect rich metrics
- Statistical summarization utilities and comprehensive plotting routines
- A quick regression runner that smoke-tests all available algorithms

All comments and docstrings are written to serve as a concise developer guide
and a reliable reference for future extensions and maintenance.
"""

import argparse
import csv
import multiprocessing
import os
import random
import statistics
import tempfile
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    import seaborn as sns
except Exception:  # seaborn is optional for core logic/tests
    sns = None  # type: ignore
from concurrent.futures import ProcessPoolExecutor

from config_manager import ConfigManager
from nqueens.backtracking import bt_nqueens_first
from nqueens.genetic import ga_nqueens
from nqueens.simulated_annealing import sa_nqueens
import importlib
import inspect
import pkgutil
import types
from typing import Any, Dict, List, Optional, Tuple, TypedDict, cast

# ======================================================
# TYPE DEFINITIONS (for static analysis and readability)
# ======================================================

class StatsSummary(TypedDict, total=False):
    count: int
    mean: Optional[float]
    median: Optional[float]
    std: Optional[float]
    min: Optional[float]
    max: Optional[float]
    q25: Optional[float]
    q75: Optional[float]
    range: Optional[float]

class BTEntry(TypedDict):
    solution_found: bool
    nodes: int
    time: float

class SARecord(TypedDict):
    success: bool
    steps: int
    time: float
    best_conflicts: int
    evals: int
    timeout: bool

class GARecord(TypedDict):
    success: bool
    gen: int
    time: float
    best_conflicts: int
    evals: int
    timeout: bool

class SAResultEntry(TypedDict, total=False):
    success_rate: float
    timeout_rate: float
    failure_rate: float
    total_runs: int
    successes: int
    failures: int
    timeouts: int
    success_steps: StatsSummary
    success_time: StatsSummary
    success_evals: StatsSummary
    success_best_conflicts: StatsSummary
    timeout_steps: StatsSummary
    timeout_time: StatsSummary
    timeout_evals: StatsSummary
    timeout_best_conflicts: StatsSummary
    failure_steps: StatsSummary
    failure_time: StatsSummary
    failure_evals: StatsSummary
    failure_best_conflicts: StatsSummary
    all_steps: StatsSummary
    all_time: StatsSummary
    all_evals: StatsSummary
    all_best_conflicts: StatsSummary
    raw_runs: List[SARecord]

class GAResultEntry(TypedDict, total=False):
    success_rate: float
    timeout_rate: float
    failure_rate: float
    total_runs: int
    successes: int
    failures: int
    timeouts: int
    success_gen: StatsSummary
    success_time: StatsSummary
    success_evals: StatsSummary
    success_best_conflicts: StatsSummary
    timeout_gen: StatsSummary
    timeout_time: StatsSummary
    timeout_evals: StatsSummary
    timeout_best_conflicts: StatsSummary
    failure_gen: StatsSummary
    failure_time: StatsSummary
    failure_evals: StatsSummary
    failure_best_conflicts: StatsSummary
    all_gen: StatsSummary
    all_time: StatsSummary
    all_evals: StatsSummary
    all_best_conflicts: StatsSummary
    pop_size: int
    max_gen: int
    pm: float
    pc: float
    tournament_size: int
    raw_runs: List[GARecord]

class ExperimentResults(TypedDict):
    BT: Dict[int, BTEntry]
    SA: Dict[int, SAResultEntry]
    GA: Dict[int, GAResultEntry]

# ======================================================
# STATISTICS UTILITIES
# ======================================================

def compute_detailed_statistics(values: List[float], label: str = "") -> StatsSummary:
    """Compute detailed summary statistics for a sequence of numbers.

    Handles empty inputs gracefully by returning a structure with None fields
    rather than raising. Uses population standard deviation for stability on
    small samples.

    Returns a dictionary with: count, mean, median, std, min, max, q25, q75, range.
    The optional 'label' is only kept for external tracing and is not used here.
    """
    if not values:
        return {
            'count': 0,
            'mean': None,
            'median': None,
            'std': None,
            'min': None,
            'max': None,
            'q25': None,
            'q75': None,
            'range': None,
        }
    
    sorted_vals = sorted(values)
    n = len(values)
    
    # Basic statistics
    mean_val = statistics.mean(values)
    median_val = statistics.median(values)
    min_val = min(values)
    max_val = max(values)
    range_val = max_val - min_val
    
    # Standard deviation (population)
    std_val = statistics.pstdev(values) if n > 1 else 0
    
    # Quartiles (simple index-based selection for small samples)
    q25 = sorted_vals[n // 4] if n >= 4 else min_val
    q75 = sorted_vals[3 * n // 4] if n >= 4 else max_val
    
    return {
        'count': n,
        'mean': mean_val,
        'median': median_val,
        'std': std_val,
        'min': min_val,
        'max': max_val,
        'q25': q25,
        'q75': q75,
        'range': range_val,
    }

def compute_grouped_statistics(results_list: List[Dict[str, Any]], success_key: str = 'success') -> Dict[str, Any]:
    """Aggregate metrics by outcome groups (success, failure, timeout).

    Expects a list of dictionaries coming from multiple independent runs, where
    each dictionary may contain keys among: success, timeout, time, steps, nodes,
    gen, evals, best_conflicts. The 'success_key' parameter identifies the field
    used to mark a successful run.

    Returns a dictionary with overall counts and rates, plus detailed statistics
    (mean, median, std, percentiles) for each metric within: all runs, successes,
    timeouts, and failures (timeouts excluded).
    """
    successes = [r for r in results_list if r.get(success_key, False)]
    timeouts = [r for r in results_list if r.get('timeout', False)]
    failures = [r for r in results_list if not r.get(success_key, False) and not r.get('timeout', False)]
    
    stats = {
        'total_runs': len(results_list),
        'successes': len(successes),
        'failures': len(failures),
        'timeouts': len(timeouts),
        'success_rate': len(successes) / len(results_list) if results_list else 0,
        'timeout_rate': len(timeouts) / len(results_list) if results_list else 0,
        'failure_rate': len(failures) / len(results_list) if results_list else 0,
    }
    
    # Statistiche per tutti i run
    for metric in ['time', 'steps', 'nodes', 'gen', 'evals', 'best_conflicts']:
        if any(metric in r for r in results_list):
            values = [r[metric] for r in results_list if metric in r]
            stats[f'all_{metric}'] = compute_detailed_statistics(values, f"all_{metric}")
    
    # Statistiche per successi
    for metric in ['time', 'steps', 'nodes', 'gen', 'evals', 'best_conflicts']:
        if any(metric in r for r in successes):
            values = [r[metric] for r in successes if metric in r]
            stats[f'success_{metric}'] = compute_detailed_statistics(values, f"success_{metric}")
    
    # Statistiche per timeout
    for metric in ['time', 'steps', 'nodes', 'gen', 'evals', 'best_conflicts']:
        if any(metric in r for r in timeouts):
            values = [r[metric] for r in timeouts if metric in r]
            stats[f'timeout_{metric}'] = compute_detailed_statistics(values, f"timeout_{metric}")
    
    # Statistiche per fallimenti (esclusi timeout)
    for metric in ['time', 'steps', 'nodes', 'gen', 'evals', 'best_conflicts']:
        if any(metric in r for r in failures):
            values = [r[metric] for r in failures if metric in r]
            stats[f'failure_{metric}'] = compute_detailed_statistics(values, f"failure_{metric}")
    
    return stats


class ProgressPrinter:
    """Minimal, stdout-only progress reporter for long-running loops.

    Prints a single line per update in the form:
        [<label>] <index>/<total> (<percent>%) - <detail>
    """

    def __init__(self, total, label):
        self.total = max(1, total)
        self.label = label

    def update(self, index, detail=""):
        percent = (index / self.total) * 100
        suffix = f" - {detail}" if detail else ""
        print(f"[{self.label}] {index}/{self.total} ({percent:.0f}%)" + suffix)


def parse_fitness_filters(fitness_args: Optional[List[str]]):
    """Normalize CLI fitness arguments into a de-duplicated uppercase list.

    Accepts multiple flags and comma-separated entries; returns None when no
    filters are provided.
    """
    if not fitness_args:
        return None

    selected = []
    for entry in fitness_args:
        for token in entry.split(','):
            token = token.strip().upper()
            if token:
                selected.append(token)

    return selected or None


def normalize_optimal_parameters(raw_params: Optional[Dict[Any, Any]]) -> Dict[Any, Any]:
    """Coerce JSON-loaded optimal parameters to a dict keyed by integer N.

    Some configurations store keys as strings (e.g. "8"). This function ensures
    consumers can safely access by integer keys while preserving any uncastable
    entries.
    """
    normalized = {}
    if not raw_params:
        return normalized

    for key, value in raw_params.items():
        try:
            normalized[int(key)] = value
        except (TypeError, ValueError):
            normalized[key] = value

    return normalized


def ensure_parameters_for_all_n(params: Dict[int, Dict[str, Any]], n_values: List[int], fitness_mode: str) -> None:
    """Validate that optimal GA parameters are present for every requested N.

    Raises ValueError if any N is missing, with a clear remediation hint.
    """
    missing = [n for n in n_values if n not in params]
    if missing:
        missing_str = ', '.join(str(n) for n in missing)
        raise ValueError(
            f"Missing GA parameters for fitness {fitness_mode} and N values: {missing_str}. "
            "Run tuning or update config.json."
        )


def apply_configuration(config_path: str, fitness_filter: Optional[List[str]] = None) -> Tuple[ConfigManager, List[str]]:
    """Load configuration from JSON and apply overrides to module globals.

    Returns a tuple (ConfigManager, selected_fitness_modes). The configuration
    may override experiment sizes, run counts, timeout settings, and GA tuning
    grids. When fitness_filter is provided, only a validated subset of modes
    will be selected; otherwise defaults from the file are used.
    """

    config_mgr = ConfigManager(config_path)

    experiment_settings = config_mgr.get_experiment_settings()
    if experiment_settings:
        global N_VALUES, RUNS_SA_FINAL, RUNS_GA_FINAL, RUNS_BT_FINAL, RUNS_GA_TUNING, OUT_DIR

        N_values_cfg = experiment_settings.get("N_values", N_VALUES)
        N_VALUES = [int(n) for n in N_values_cfg]
        RUNS_SA_FINAL = int(experiment_settings.get("runs_sa_final", RUNS_SA_FINAL))
        RUNS_GA_FINAL = int(experiment_settings.get("runs_ga_final", RUNS_GA_FINAL))
        RUNS_BT_FINAL = int(experiment_settings.get("runs_bt_final", RUNS_BT_FINAL))
        RUNS_GA_TUNING = int(experiment_settings.get("runs_ga_tuning", RUNS_GA_TUNING))
        OUT_DIR = experiment_settings.get("output_dir", OUT_DIR)

    timeout_settings = config_mgr.get_timeout_settings()
    if timeout_settings:
        set_timeouts(
            bt_timeout=timeout_settings.get("bt_time_limit", BT_TIME_LIMIT),
            sa_timeout=timeout_settings.get("sa_time_limit", SA_TIME_LIMIT),
            ga_timeout=timeout_settings.get("ga_time_limit", GA_TIME_LIMIT),
            experiment_timeout=timeout_settings.get("experiment_timeout", EXPERIMENT_TIMEOUT),
        )

    tuning_grid = config_mgr.get_tuning_grid()
    if tuning_grid:
        global POP_MULTIPLIERS, GEN_MULTIPLIERS, PM_VALUES, PC_FIXED, TOURNAMENT_SIZE_FIXED

        POP_MULTIPLIERS = [int(v) for v in tuning_grid.get("pop_multipliers", POP_MULTIPLIERS)]
        GEN_MULTIPLIERS = [int(v) for v in tuning_grid.get("gen_multipliers", GEN_MULTIPLIERS)]
        PM_VALUES = [float(v) for v in tuning_grid.get("pm_values", PM_VALUES)]
        PC_FIXED = float(tuning_grid.get("pc_fixed", PC_FIXED))
        TOURNAMENT_SIZE_FIXED = int(tuning_grid.get("tournament_size_fixed", TOURNAMENT_SIZE_FIXED))

    fitness_modes_cfg = [mode.upper() for mode in config_mgr.get_fitness_modes()]
    if not fitness_modes_cfg:
        fitness_modes_cfg = ["F1"]

    if fitness_filter:
        requested = {mode.upper() for mode in fitness_filter}
        unknown = requested.difference(set(fitness_modes_cfg))
        if unknown:
            raise ValueError(
                "Unknown fitness modes requested: " + ', '.join(sorted(unknown))
            )
        selected_modes = [mode for mode in fitness_modes_cfg if mode in requested]
    else:
        selected_modes = fitness_modes_cfg

    if not selected_modes:
        raise ValueError("No fitness modes selected after applying filters.")

    global FITNESS_MODES
    FITNESS_MODES = selected_modes

    return config_mgr, selected_modes


def load_optimal_parameters(fitness_mode: str, config_mgr: ConfigManager, n_values: List[int]) -> Dict[int, Dict[str, Any]]:
    """Fetch and validate optimal GA parameters for a given fitness mode.

    Ensures parameters exist for all requested N. Intended for use when the
    tuning phase is skipped and stored parameters are reused.
    """
    if config_mgr is None:
        raise ValueError("Config manager is required when skip-tuning is enabled.")

    params = normalize_optimal_parameters(config_mgr.get_optimal_parameters(fitness_mode))
    ensure_parameters_for_all_n(params, n_values, fitness_mode)
    return params

# ======================================================
# GLOBAL CONFIGURATION
# ======================================================

# Board sizes to evaluate (in ascending order) for scalability analysis
N_VALUES = [8, 16, 24, 40, 80, 120]

# Number of independent runs in final experiments (higher = more robust stats)
RUNS_SA_FINAL = 40
RUNS_GA_FINAL = 40
RUNS_BT_FINAL = 1   # Backtracking is deterministic; one run per N is sufficient

# Number of runs per GA tuning combination (kept lower to accelerate grid search)
RUNS_GA_TUNING = 5

# Backtracking time limit in seconds (None = no limit)
# Helps avoid pathological runtimes on large N when needed
BT_TIME_LIMIT = 60*5.0  # e.g., 5 minutes

# SA and GA time limits in seconds (None = no limit)
SA_TIME_LIMIT = 120.0  # Simulated Annealing
GA_TIME_LIMIT = 240.0  # Genetic Algorithm 

# Global timeout per experiment bundle (None = no limit)
EXPERIMENT_TIMEOUT = 120.0  # ~2 minutes per full experiment

def set_timeouts(bt_timeout: Optional[float] = None, sa_timeout: Optional[float] = 30.0, ga_timeout: Optional[float] = 60.0, experiment_timeout: Optional[float] = 120.0) -> None:
    """Configure timeouts for all algorithms and the experiment wrapper.

    Args:
        bt_timeout: Backtracking time limit in seconds (None = unlimited)
        sa_timeout: Simulated Annealing time limit in seconds (None = unlimited)
        ga_timeout: Genetic Algorithm time limit in seconds (None = unlimited)
        experiment_timeout: Global time limit in seconds for an experiment (None = unlimited)
    """
    global BT_TIME_LIMIT, SA_TIME_LIMIT, GA_TIME_LIMIT, EXPERIMENT_TIMEOUT
    BT_TIME_LIMIT = bt_timeout
    SA_TIME_LIMIT = sa_timeout
    GA_TIME_LIMIT = ga_timeout
    EXPERIMENT_TIMEOUT = experiment_timeout
    
    print("Timeout settings configured:")
    print(f"   - BT: {BT_TIME_LIMIT}s" if BT_TIME_LIMIT else "   - BT: unlimited")
    print(f"   - SA: {SA_TIME_LIMIT}s" if SA_TIME_LIMIT else "   - SA: unlimited")
    print(f"   - GA: {GA_TIME_LIMIT}s" if GA_TIME_LIMIT else "   - GA: unlimited")
    print(f"   - Experiment: {EXPERIMENT_TIMEOUT}s" if EXPERIMENT_TIMEOUT else "   - Experiment: unlimited")

# Output directory for CSV and charts
OUT_DIR = "results_nqueens_tuning"

# GA tuning grid (defines the parameter search space)
POP_MULTIPLIERS = [4, 8, 16]       # pop_size ≈ {k}*N; population scales with N
GEN_MULTIPLIERS = [30, 50, 80]     # max_gen ≈ {m}*N; generations scale with N
PM_VALUES = [0.05, 0.1, 0.15]      # mutation rate candidates
PC_FIXED = 0.8                     # fixed crossover probability
TOURNAMENT_SIZE_FIXED = 3          # tournament size for selection

# Fitness functions to evaluate (F1–F6)
FITNESS_MODES = ["F1", "F2", "F3", "F4", "F5", "F6"]

# Number of worker processes to use (leave one core for the OS)
NUM_PROCESSES = max(1, multiprocessing.cpu_count() - 1)


# ======================================================
# GA: Parameter tuning for a single (N, fitness_mode)
# ======================================================

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

    Selection criterion:
    1) maximize success_rate; 2) among ties, minimize avg_gen_success.

    Args:
        N: problem size (number of queens)
        fitness_mode: fitness function label (e.g., "F1", "F2")
        pop_multipliers: list of multipliers for population size (k*N)
        gen_multipliers: list of multipliers for max generations (m*N)
        pm_values: list of mutation rates to try
        pc: fixed crossover probability
        tournament_size: selection tournament size
        runs_tuning: independent runs per parameter combination

    Returns:
        dict describing the best configuration with aggregated stats:
        {
            "N": int,
            "fitness_mode": str,
            "pop_size": int,
            "max_gen": int,
            "pm": float,
            "pc": float,
            "tournament_size": int,
            "success_rate": float,
            "avg_gen_success": Optional[float],
        }
    """
    best: Optional[Dict[str, Any]] = None

    # Evaluate all parameter combinations
    for k in pop_multipliers:
        pop_size = max(50, int(k * N))  # enforce a minimum population size
        for m in gen_multipliers:
            max_gen = int(m * N)  # scale generations with problem size
            for pm in pm_values:
                # Test this combination across multiple runs
                successes = 0
                gen_success = []

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
                    if s:  # solution found
                        successes += 1
                        gen_success.append(gen)

                # Compute aggregated stats for this combination
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

                # Update the current best according to the selection criterion
                if best is None:
                    best = candidate
                else:
                    # Criterio di selezione: prima success_rate, poi avg_gen_success
                    if candidate["success_rate"] > best["success_rate"]:
                        best = candidate
                    elif candidate["success_rate"] == best["success_rate"]:
                        # A parità di success rate, minimizza generazioni medie
                        if candidate["avg_gen_success"] is not None and best["avg_gen_success"] is not None:
                            if candidate["avg_gen_success"] < best["avg_gen_success"]:
                                best = candidate

    if best is None:
        raise RuntimeError("No GA parameter candidate evaluated; check tuning grid.")
    return best


# ======================================================
# GA: Helper functions for parallel execution
# ======================================================

def run_single_ga_experiment(params: Tuple[int, int, int, float, float, int, str]):
    """Execute a single GA run in a worker process.

    Required because ProcessPoolExecutor needs top-level callables (it cannot
    serialize lambdas or bound methods).

    Args:
        params: tuple (N, pop_size, max_gen, pc, pm, tournament_size, fitness_mode)

    Returns:
        tuple: result of ga_nqueens()
    """
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


def run_single_sa_experiment(params: Tuple[int, int, float, float]):
    """Execute a single SA run in a worker process.

    Args:
        params: tuple (N, max_iter, T0, alpha)

    Returns:
        tuple: result of sa_nqueens()
    """
    N, max_iter, T0, alpha = params
    return sa_nqueens(N, max_iter=max_iter, T0=T0, alpha=alpha, time_limit=SA_TIME_LIMIT)


def run_with_timeout(func, args, timeout: Optional[float]) -> Tuple[bool, Any]:
    """Run a function with a timeout using a dedicated worker process.

    Args:
        func: callable to execute
        args: arguments passed to the callable
        timeout: timeout in seconds

    Returns:
        tuple: (success, result) with success=True if the function completes in time,
        and result set to the return value or None on timeout/error.
    """
    try:
        with ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func, args)
            try:
                result = future.result(timeout=timeout)
            except KeyboardInterrupt:
                executor.shutdown(cancel_futures=True)
                raise
            return True, result
    except Exception as e:
        print(f"WARNING: Timeout or error during execution: {e}")
        return False, None


def test_parameter_combination_parallel(params: Tuple[int, str, int, int, float, float, int, int]) -> Dict[str, Any]:
    """Evaluate one GA parameter combination with multiple parallel runs.

    Args:
        params: tuple (N, fitness_mode, pop_size, max_gen, pc, pm, tournament_size, runs_tuning)

    Returns:
        dict: aggregated statistics for this parameter combination
    """
    N, fitness_mode, pop_size, max_gen, pc, pm, tournament_size, runs_tuning = params
    
    # Prepara parametri per tutti i run di questa combinazione
    run_params = [(N, pop_size, max_gen, pc, pm, tournament_size, fitness_mode) 
                  for _ in range(runs_tuning)]
    
    # Esegui i run in parallelo (limitato da NUM_PROCESSES)
    with ProcessPoolExecutor(max_workers=min(NUM_PROCESSES, runs_tuning)) as executor:
        try:
            results = list(executor.map(run_single_ga_experiment, run_params))
        except KeyboardInterrupt:
            executor.shutdown(cancel_futures=True)
            raise
    
    # Calcola statistiche aggregate
    successes = 0
    gen_success = []
    for s, gen, _, bestc, _, _ in results:  # Aggiunto _ per timeout
        if s:  # se ha trovato soluzione
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
    """Parallel version of GA tuning for a single (N, fitness_mode).

    Two levels of parallelism:
    1) different parameter combinations are evaluated in parallel;
    2) multiple independent runs per combination are also executed in parallel.

    Args:
        N: problem size
        fitness_mode: fitness function to use
        pop_multipliers, gen_multipliers, pm_values: parameter search space
        pc, tournament_size: fixed parameters
        runs_tuning: independent runs per combination

    Returns:
        dict describing the best parameters found
    """
    print(f"  Preparazione {len(pop_multipliers) * len(gen_multipliers) * len(pm_values)} combinazioni di parametri...")
    
    # Genera tutte le combinazioni di parametri da testare
    param_combinations = []
    for k in pop_multipliers:
        pop_size = max(50, int(k * N))
        for m in gen_multipliers:
            max_gen = int(m * N)
            for pm in pm_values:
                param_combinations.append(
                    (N, fitness_mode, pop_size, max_gen, pc, pm, tournament_size, runs_tuning)
                )
    
    # Testa tutte le combinazioni in parallelo usando ProcessPoolExecutor
    print(f"  Esecuzione parallela con {NUM_PROCESSES} processi...")
    with ProcessPoolExecutor(max_workers=NUM_PROCESSES) as executor:
        try:
            candidates = list(executor.map(test_parameter_combination_parallel, param_combinations))
        except KeyboardInterrupt:
            executor.shutdown(cancel_futures=True)
            raise
    
    # Seleziona la migliore combinazione usando stesso criterio della versione sequenziale
    best: Optional[Dict[str, Any]] = None
    for candidate in candidates:
        if best is None:
            best = candidate
        else:
            # Criterio: prima success_rate, poi avg_gen_success
            if candidate["success_rate"] > best["success_rate"]:
                best = candidate
            elif candidate["success_rate"] == best["success_rate"]:
                # A parità di success rate, minimizza generazioni medie
                if candidate["avg_gen_success"] is not None and best["avg_gen_success"] is not None:
                    if candidate["avg_gen_success"] < best["avg_gen_success"]:
                        best = candidate
    
    if best is None:
        raise RuntimeError("No GA parameter candidate evaluated in parallel; check tuning grid.")
    print(f"  Migliore combinazione: pop_size={best['pop_size']}, max_gen={best['max_gen']}, pm={best['pm']}, success_rate={best['success_rate']:.3f}")
    return best


def tune_single_fitness(params: Tuple[int, str, List[int], List[int], List[float], float, int, int]) -> Tuple[str, Dict[str, Any]]:
    """Wrapper to tune GA parameters for a single fitness function.

    Runs the tuning in a separate process to improve throughput.

    Args:
        params: tuple (N, fitness_mode, pop_multipliers, gen_multipliers,
                      pm_values, pc, tournament_size, runs_tuning)

    Returns:
        tuple: (fitness_mode, best_params)
    """
    N, fitness_mode, pop_multipliers, gen_multipliers, pm_values, pc, tournament_size, runs_tuning = params
    return fitness_mode, tune_ga_for_N_parallel(
        N, fitness_mode, pop_multipliers, gen_multipliers, pm_values, pc, tournament_size, runs_tuning
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
    """Tune GA parameters for ALL fitness functions concurrently for a given N.

    Each fitness is tuned on a separate worker process, enabling near-linear
    speedup up to the number of available CPU cores.

    Args:
        N: problem size
        fitness_modes: list of fitness labels ["F1", "F2", ...]
        pop_multipliers, gen_multipliers, pm_values: parameter search space
        pc, tournament_size: fixed parameters
        runs_tuning: independent runs per parameter combination

    Returns:
        dict mapping fitness_mode -> best_params
    """
    print(f"Tuning contemporaneo di {len(fitness_modes)} fitness per N={N}")
    
    # Prepara parametri per tutte le fitness
    tuning_params = []
    for fitness_mode in fitness_modes:
        tuning_params.append((
            N, fitness_mode, pop_multipliers, gen_multipliers, pm_values, 
            pc, tournament_size, runs_tuning
        ))
    
    # Esegui il tuning di tutte le fitness in parallelo
    # Ogni fitness viene processata su un core diverso
    print(f"  Utilizzando {min(NUM_PROCESSES, len(fitness_modes))} processi per {len(fitness_modes)} fitness...")
    start_time = perf_counter()
    
    with ProcessPoolExecutor(max_workers=min(NUM_PROCESSES, len(fitness_modes))) as executor:
        try:
            results = list(executor.map(tune_single_fitness, tuning_params))
        except KeyboardInterrupt:
            executor.shutdown(cancel_futures=True)
            raise
    
    elapsed_time = perf_counter() - start_time
    
    # Organizza risultati per fitness
    best_params_per_fitness = {}
    for fitness_mode, best_params in results:
        best_params_per_fitness[fitness_mode] = best_params
        print(f"  Completato {fitness_mode}: success_rate={best_params['success_rate']:.3f}, "
              f"pop_size={best_params['pop_size']}, pm={best_params['pm']}")
    
    print(f"Tuning contemporaneo completato in {elapsed_time:.1f}s per N={N}")
    return best_params_per_fitness


# ======================================================
# Esperimenti finali con parametri GA ottimizzati
# ======================================================

def run_experiments_with_best_ga(
    N_values: List[int],
    runs_sa: int,
    runs_ga: int,
    bt_time_limit: Optional[float],
    fitness_mode: str,
    best_ga_params_for_N: Dict[int, Dict[str, Any]],
    progress_label: Optional[str] = None,
) -> ExperimentResults:
    """Run final sequential experiments using optimal GA parameters.

    Returns a structured dictionary with aggregated statistics for BT, SA, and GA.
    Expected shape used elsewhere in the module:
    results = {
        "BT": {N: {"solution_found": bool, "nodes": int, "time": float}},
        "SA": {N: {...}},
        "GA": {N: {...}},
    }
    """
    results = cast(ExperimentResults, {"BT": {}, "SA": {}, "GA": {}})

    progress = ProgressPrinter(len(N_values), progress_label) if progress_label else None

    for index, N in enumerate(N_values, start=1):
        if progress:
            progress.update(index, f"N={N}")
        print(f"=== (Final) N = {N}, GA fitness {fitness_mode} ===")

        # ----- BACKTRACKING (deterministico) -----
        sol, nodes, t = bt_nqueens_first(N, time_limit=bt_time_limit)
        results["BT"][N] = {
            "solution_found": sol is not None,
            "nodes": nodes,
            "time": t,
        }

        # ----- SIMULATED ANNEALING (stocastico) -----
        sa_runs = []
        max_iter_sa = 2000 + 200 * N
        for _ in range(runs_sa):
            s, steps, tt, bestc, evals, timeout = sa_nqueens(
                N, max_iter=max_iter_sa, T0=1.0, alpha=0.995, time_limit=SA_TIME_LIMIT
            )
            sa_runs.append({
                "success": s,
                "steps": steps,
                "time": tt,
                "best_conflicts": bestc,
                "evals": evals,
                "timeout": timeout,
            })

        sa_stats = compute_grouped_statistics(sa_runs, 'success')

        results["SA"][N] = {
            "success_rate": sa_stats['success_rate'],
            "timeout_rate": sa_stats['timeout_rate'],
            "failure_rate": sa_stats['failure_rate'],
            "total_runs": sa_stats['total_runs'],
            "successes": sa_stats['successes'],
            "failures": sa_stats['failures'],
            "timeouts": sa_stats['timeouts'],
            # Statistiche per successi
            "success_steps": sa_stats.get('success_steps', {}),
            "success_time": sa_stats.get('success_time', {}),
            "success_evals": sa_stats.get('success_evals', {}),
            "success_best_conflicts": sa_stats.get('success_best_conflicts', {}),
            # Statistiche per timeout
            "timeout_steps": sa_stats.get('timeout_steps', {}),
            "timeout_time": sa_stats.get('timeout_time', {}),
            "timeout_evals": sa_stats.get('timeout_evals', {}),
            "timeout_best_conflicts": sa_stats.get('timeout_best_conflicts', {}),
            # Statistiche per fallimenti
            "failure_steps": sa_stats.get('failure_steps', {}),
            "failure_time": sa_stats.get('failure_time', {}),
            "failure_evals": sa_stats.get('failure_evals', {}),
            "failure_best_conflicts": sa_stats.get('failure_best_conflicts', {}),
            # Statistiche complessive
            "all_steps": sa_stats.get('all_steps', {}),
            "all_time": sa_stats.get('all_time', {}),
            "all_evals": sa_stats.get('all_evals', {}),
            "all_best_conflicts": sa_stats.get('all_best_conflicts', {}),
            # Raw
            "raw_runs": sa_runs.copy(),
        }

        # ----- ALGORITMO GENETICO con parametri ottimali -----
        params = best_ga_params_for_N[N]
        pop_size = params["pop_size"]
        max_gen = params["max_gen"]
        pm = params["pm"]
        pc = params["pc"]
        tsize = params["tournament_size"]

        ga_runs = []
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
            ga_runs.append({
                "success": s,
                "gen": gen,
                "time": tt,
                "best_conflicts": bestc,
                "evals": evals,
                "timeout": timeout,
            })

        ga_stats = compute_grouped_statistics(ga_runs, 'success')
        results["GA"][N] = {
            "success_rate": ga_stats['success_rate'],
            "timeout_rate": ga_stats['timeout_rate'],
            "failure_rate": ga_stats['failure_rate'],
            "total_runs": ga_stats['total_runs'],
            "successes": ga_stats['successes'],
            "failures": ga_stats['failures'],
            "timeouts": ga_stats['timeouts'],
            # Statistiche per successi
            "success_gen": ga_stats.get('success_gen', {}),
            "success_time": ga_stats.get('success_time', {}),
            "success_evals": ga_stats.get('success_evals', {}),
            "success_best_conflicts": ga_stats.get('success_best_conflicts', {}),
            # Statistiche per timeout
            "timeout_gen": ga_stats.get('timeout_gen', {}),
            "timeout_time": ga_stats.get('timeout_time', {}),
            "timeout_evals": ga_stats.get('timeout_evals', {}),
            "timeout_best_conflicts": ga_stats.get('timeout_best_conflicts', {}),
            # Statistiche per fallimenti
            "failure_gen": ga_stats.get('failure_gen', {}),
            "failure_time": ga_stats.get('failure_time', {}),
            "failure_evals": ga_stats.get('failure_evals', {}),
            "failure_best_conflicts": ga_stats.get('failure_best_conflicts', {}),
            # Statistiche complessive
            "all_gen": ga_stats.get('all_gen', {}),
            "all_time": ga_stats.get('all_time', {}),
            "all_evals": ga_stats.get('all_evals', {}),
            "all_best_conflicts": ga_stats.get('all_best_conflicts', {}),
            # Parametri GA
            "pop_size": pop_size,
            "max_gen": max_gen,
            "pm": pm,
            "pc": pc,
            "tournament_size": tsize,
            # Raw
            "raw_runs": ga_runs.copy(),
        }

    return results


def run_experiments_with_best_ga_parallel(
    N_values: List[int],
    runs_sa: int,
    runs_ga: int,
    bt_time_limit: Optional[float],
    fitness_mode: str,
    best_ga_params_for_N: Dict[int, Dict[str, Any]],
    progress_label: Optional[str] = None,
) -> ExperimentResults:
    """Parallel version of the final experiments runner.

    Executes SA and GA across multiple processes to achieve significant speedup.
    Backtracking remains serial (deterministic and fast).

    Args:
        N_values: list of board sizes to test
        runs_sa, runs_ga: number of independent runs per algorithm
        bt_time_limit: time limit for Backtracking
        fitness_mode: GA fitness function label
        best_ga_params_for_N: per-N GA parameters obtained from tuning

    Returns:
        dict with the same structure as the sequential runner
    """
    results = cast(ExperimentResults, {"BT": {}, "SA": {}, "GA": {}})

    progress = ProgressPrinter(len(N_values), progress_label) if progress_label else None

    for index, N in enumerate(N_values, start=1):
        if progress:
            progress.update(index, f"N={N}")
        print(f"=== (Final Parallel) N = {N}, GA fitness {fitness_mode} ===")

        # ----- BACKTRACKING (sempre seriale, è veloce) -----
        sol, nodes, t = bt_nqueens_first(N, time_limit=bt_time_limit)
        results["BT"][N] = {
            "solution_found": sol is not None,
            "nodes": nodes,
            "time": t,
        }

        # ----- SIMULATED ANNEALING (parallel) -----
        print(f"  Running {runs_sa} SA runs in parallel...")
        max_iter_sa = 2000 + 200 * N
        
        # Prepara parametri per tutti i run SA
        sa_params = [(N, max_iter_sa, 1.0, 0.995) for _ in range(runs_sa)]
        
        # Esegui tutti i run SA in parallelo
        with ProcessPoolExecutor(max_workers=NUM_PROCESSES) as executor:
            try:
                sa_raw_results = list(executor.map(run_single_sa_experiment, sa_params))
            except KeyboardInterrupt:
                executor.shutdown(cancel_futures=True)
                raise
        
        # Converte risultati in formato strutturato
        sa_runs = []
        for s, steps, tt, bestc, evals, timeout in sa_raw_results:
            sa_runs.append({
                "success": s,
                "steps": steps,
                "time": tt,
                "best_conflicts": bestc,
                "evals": evals,
                "timeout": timeout,
            })

        # Calcola statistiche aggregate SA con funzione avanzata
        sa_stats = compute_grouped_statistics(sa_runs, 'success')

        results["SA"][N] = {
            "success_rate": sa_stats['success_rate'],
            "timeout_rate": sa_stats['timeout_rate'],
            "failure_rate": sa_stats['failure_rate'],
            "total_runs": sa_stats['total_runs'],
            "successes": sa_stats['successes'],
            "failures": sa_stats['failures'],
            "timeouts": sa_stats['timeouts'],
            
            # Statistiche complete per successi
            "success_steps": sa_stats.get('success_steps', {}),
            "success_time": sa_stats.get('success_time', {}),
            "success_evals": sa_stats.get('success_evals', {}),
            "success_best_conflicts": sa_stats.get('success_best_conflicts', {}),
            
            # Statistiche complete per timeout
            "timeout_steps": sa_stats.get('timeout_steps', {}),
            "timeout_time": sa_stats.get('timeout_time', {}),
            "timeout_evals": sa_stats.get('timeout_evals', {}),
            "timeout_best_conflicts": sa_stats.get('timeout_best_conflicts', {}),
            
            # Statistiche complete per fallimenti  
            "failure_steps": sa_stats.get('failure_steps', {}),
            "failure_time": sa_stats.get('failure_time', {}),
            "failure_evals": sa_stats.get('failure_evals', {}),
            "failure_best_conflicts": sa_stats.get('failure_best_conflicts', {}),
            
            # Statistiche per tutti i run
            "all_steps": sa_stats.get('all_steps', {}),
            "all_time": sa_stats.get('all_time', {}),
            "all_evals": sa_stats.get('all_evals', {}),
            "all_best_conflicts": sa_stats.get('all_best_conflicts', {}),
            
            # Dati grezzi per analisi dettagliate
            "raw_runs": sa_runs.copy(),
        }

        # ----- GENETIC ALGORITHM (parallel) with optimal parameters -----
        print(f"  Running {runs_ga} GA runs in parallel...")
        params = best_ga_params_for_N[N]
        pop_size = params["pop_size"]
        max_gen = params["max_gen"]
        pm = params["pm"]
        pc = params["pc"]
        tsize = params["tournament_size"]

        # Prepara parametri per tutti i run GA
        ga_params = [(N, pop_size, max_gen, pc, pm, tsize, fitness_mode) 
                     for _ in range(runs_ga)]
        
        # Esegui tutti i run GA in parallelo
        with ProcessPoolExecutor(max_workers=NUM_PROCESSES) as executor:
            try:
                ga_raw_results = list(executor.map(run_single_ga_experiment, ga_params))
            except KeyboardInterrupt:
                executor.shutdown(cancel_futures=True)
                raise
        
        # Converte risultati in formato strutturato
        ga_runs = []
        for s, gen, tt, bestc, evals, timeout in ga_raw_results:
            ga_runs.append({
                "success": s,
                "gen": gen,
                "time": tt,
                "best_conflicts": bestc,
                "evals": evals,
                "timeout": timeout,
            })

        # Calcola statistiche aggregate GA con funzione avanzata
        ga_stats = compute_grouped_statistics(ga_runs, 'success')

        results["GA"][N] = {
            "success_rate": ga_stats['success_rate'],
            "timeout_rate": ga_stats['timeout_rate'],
            "failure_rate": ga_stats['failure_rate'],
            "total_runs": ga_stats['total_runs'],
            "successes": ga_stats['successes'],
            "failures": ga_stats['failures'],
            "timeouts": ga_stats['timeouts'],
            
            # Statistiche complete per successi
            "success_gen": ga_stats.get('success_gen', {}),
            "success_time": ga_stats.get('success_time', {}),
            "success_evals": ga_stats.get('success_evals', {}),
            "success_best_conflicts": ga_stats.get('success_best_conflicts', {}),
            
            # Statistiche complete per timeout
            "timeout_gen": ga_stats.get('timeout_gen', {}),
            "timeout_time": ga_stats.get('timeout_time', {}),
            "timeout_evals": ga_stats.get('timeout_evals', {}),
            "timeout_best_conflicts": ga_stats.get('timeout_best_conflicts', {}),
            
            # Statistiche complete per fallimenti  
            "failure_gen": ga_stats.get('failure_gen', {}),
            "failure_time": ga_stats.get('failure_time', {}),
            "failure_evals": ga_stats.get('failure_evals', {}),
            "failure_best_conflicts": ga_stats.get('failure_best_conflicts', {}),
            
            # Statistiche per tutti i run
            "all_gen": ga_stats.get('all_gen', {}),
            "all_time": ga_stats.get('all_time', {}),
            "all_evals": ga_stats.get('all_evals', {}),
            "all_best_conflicts": ga_stats.get('all_best_conflicts', {}),
            
            # Salva anche i parametri GA utilizzati
            "pop_size": pop_size,
            "max_gen": max_gen,
            "pm": pm,
            "pc": pc,
            "tournament_size": tsize,
            
            # Dati grezzi per analisi dettagliate
            "raw_runs": ga_runs.copy(),
        }

    return results


# ======================================================
# CSV export and charting
# ======================================================

def save_results_to_csv(results: ExperimentResults, N_values: List[int], fitness_mode: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    filename = os.path.join(out_dir, f"results_GA_{fitness_mode}_tuned.csv")

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "N",
            # BT metriche logiche e temporali
            "BT_solution_found",
            "BT_nodes_explored",
            "BT_time_seconds",
            
            # SA metriche statistiche complete
            "SA_success_rate",
            "SA_timeout_rate", 
            "SA_failure_rate",
            "SA_total_runs",
            "SA_successes",
            "SA_failures",
            "SA_timeouts",
            
            # SA metriche logiche per successi
            "SA_success_steps_mean",
            "SA_success_steps_median", 
            "SA_success_evals_mean",
            "SA_success_evals_median",
            
            # SA metriche logiche per timeout
            "SA_timeout_steps_mean",
            "SA_timeout_steps_median",
            "SA_timeout_evals_mean", 
            "SA_timeout_evals_median",
            
            # SA metriche temporali per successi
            "SA_success_time_mean",
            "SA_success_time_median",
            
            # GA metriche statistiche complete  
            "GA_success_rate",
            "GA_timeout_rate",
            "GA_failure_rate", 
            "GA_total_runs",
            "GA_successes", 
            "GA_failures",
            "GA_timeouts",
            
            # GA metriche logiche per successi
            "GA_success_gen_mean",
            "GA_success_gen_median",
            "GA_success_evals_mean", 
            "GA_success_evals_median",
            
            # GA metriche logiche per timeout
            "GA_timeout_gen_mean",
            "GA_timeout_gen_median",
            "GA_timeout_evals_mean",
            "GA_timeout_evals_median",
            
            # GA metriche temporali per successi
            "GA_success_time_mean",
            "GA_success_time_median",
            
            # GA parametri utilizzati
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
            
            # Estrae statistiche SA
            sa_steps_success = sa.get("success_steps", {})
            sa_evals_success = sa.get("success_evals", {})
            sa_time_success = sa.get("success_time", {})
            sa_steps_timeout = sa.get("timeout_steps", {})
            sa_evals_timeout = sa.get("timeout_evals", {})
            
            # Estrae statistiche GA
            ga_gen_success = ga.get("success_gen", {})
            ga_evals_success = ga.get("success_evals", {})
            ga_time_success = ga.get("success_time", {})
            ga_gen_timeout = ga.get("timeout_gen", {})
            ga_evals_timeout = ga.get("timeout_evals", {})
            
            writer.writerow([
                N,
                # BT
                int(bt["solution_found"]),
                bt["nodes"],
                bt["time"],
                
                # SA statistiche generali
                sa.get("success_rate", 0.0),
                sa.get("timeout_rate", 0),
                sa.get("failure_rate", 0),
                sa.get("total_runs", 0),
                sa.get("successes", 0), 
                sa.get("failures", 0),
                sa.get("timeouts", 0),
                
                # SA metriche logiche successi
                sa_steps_success.get("mean", ""),
                sa_steps_success.get("median", ""),
                sa_evals_success.get("mean", ""),
                sa_evals_success.get("median", ""),
                
                # SA metriche logiche timeout
                sa_steps_timeout.get("mean", ""),
                sa_steps_timeout.get("median", ""),
                sa_evals_timeout.get("mean", ""),
                sa_evals_timeout.get("median", ""),
                
                # SA metriche temporali successi
                sa_time_success.get("mean", ""),
                sa_time_success.get("median", ""),
                
                # GA statistiche generali
                ga.get("success_rate", 0.0),
                ga.get("timeout_rate", 0),
                ga.get("failure_rate", 0),
                ga.get("total_runs", 0),
                ga.get("successes", 0),
                ga.get("failures", 0),
                ga.get("timeouts", 0),
                
                # GA metriche logiche successi
                ga_gen_success.get("mean", ""),
                ga_gen_success.get("median", ""),
                ga_evals_success.get("mean", ""),
                ga_evals_success.get("median", ""),
                
                # GA metriche logiche timeout
                ga_gen_timeout.get("mean", ""),
                ga_gen_timeout.get("median", ""),
                ga_evals_timeout.get("mean", ""),
                ga_evals_timeout.get("median", ""),
                
                # GA metriche temporali successi
                ga_time_success.get("mean", ""),
                ga_time_success.get("median", ""),
                
                # GA parametri
                ga.get("pop_size", 0),
                ga.get("max_gen", 0),
                ga.get("pm", 0.0),
                ga.get("pc", 0.0),
                ga.get("tournament_size", 0),
            ])

    print(f"CSV saved: {filename}")


def save_raw_data_to_csv(results: ExperimentResults, N_values: List[int], fitness_mode: str, out_dir: str) -> None:
    """Save raw per-run data for detailed external analysis.

    Writes separate CSVs for SA, GA, and BT capturing individual run outcomes
    (success, timeout) and metrics (steps/generations, evaluations, time, etc.).
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # File per dati grezzi SA
    sa_filename = os.path.join(out_dir, f"raw_data_SA_{fitness_mode}.csv")
    with open(sa_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "N", "run_id", "algorithm", "success", "timeout",
            "steps", "time_seconds", "evals", "best_conflicts"
        ])
        
        for N in N_values:
            sa_data = results["SA"][N]
            if "raw_runs" in sa_data:
                for i, run in enumerate(sa_data["raw_runs"]):
                    writer.writerow([
                        N, i+1, "SA", run["success"], run["timeout"],
                        run["steps"], run["time"], run["evals"], run["best_conflicts"]
                    ])
    
    # File per dati grezzi GA  
    ga_filename = os.path.join(out_dir, f"raw_data_GA_{fitness_mode}.csv")
    with open(ga_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "N", "run_id", "algorithm", "success", "timeout",
            "gen", "time_seconds", "evals", "best_fitness", "best_conflicts",
            "pop_size", "max_gen", "pm", "pc", "tournament_size"
        ])
        
        for N in N_values:
            ga_data = results["GA"][N]
            if "raw_runs" in ga_data:
                for i, run in enumerate(ga_data["raw_runs"]):
                    writer.writerow([
                        N, i+1, "GA", run["success"], run["timeout"],
                        run["gen"], run["time"], run["evals"], 
                        run.get("best_fitness", ""), run.get("best_conflicts", ""),
                        ga_data.get("pop_size", 0), ga_data.get("max_gen", 0), 
                        ga_data.get("pm", 0.0), ga_data.get("pc", 0.0), ga_data.get("tournament_size", 0)
                    ])
    
    # File per dati grezzi BT (uno per N)
    bt_filename = os.path.join(out_dir, f"raw_data_BT_{fitness_mode}.csv") 
    with open(bt_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "N", "algorithm", "solution_found", "nodes_explored", "time_seconds"
        ])
        
        for N in N_values:
            bt_data = results["BT"][N]
            writer.writerow([
                N, "BT", bt_data["solution_found"], bt_data["nodes"], bt_data["time"]
            ])
    
    print("Raw data saved:")
    print(f"  SA: {sa_filename}")
    print(f"  GA: {ga_filename}")
    print(f"  BT: {bt_filename}")


def save_logical_cost_analysis(results: ExperimentResults, N_values: List[int], fitness_mode: str, out_dir: str) -> None:
    """Save analysis focused on machine-independent logical costs.

    Includes nodes explored (BT), SA iterations, GA generations, and related
    success rates, plus timing for reference.
    """
    os.makedirs(out_dir, exist_ok=True)
    filename = os.path.join(out_dir, f"logical_costs_{fitness_mode}.csv")
    
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "N",
            # BT - costo logico
            "BT_solution_found",
            "BT_nodes_explored",  # costo logico primario
            
            # SA - costi logici  
            "SA_success_rate",
            "SA_steps_mean_all",        # costo logico primario
            "SA_steps_median_all",
            "SA_evals_mean_all",        # costo logico secondario
            "SA_evals_median_all",
            "SA_steps_mean_success",    # per successi
            "SA_evals_mean_success", 
            
            # GA - costi logici
            "GA_success_rate", 
            "GA_gen_mean_all",          # costo logico primario
            "GA_gen_median_all",
            "GA_evals_mean_all",        # costo logico secondario
            "GA_evals_median_all",
            "GA_gen_mean_success",      # per successi
            "GA_evals_mean_success",
            
            # Tempi come riferimento sperimentale
            "BT_time_seconds",
            "SA_time_mean_success",
            "GA_time_mean_success",
        ])
        
        for N in N_values:
            bt = results["BT"][N]
            sa = results["SA"][N]
            ga = results["GA"][N]
            
            # Estrae metriche logiche SA
            sa_all_steps = sa.get("all_steps", {})
            sa_all_evals = sa.get("all_evals", {})
            sa_success_steps = sa.get("success_steps", {})
            sa_success_evals = sa.get("success_evals", {})
            sa_success_time = sa.get("success_time", {})
            
            # Estrae metriche logiche GA  
            ga_all_gen = ga.get("all_gen", {})
            ga_all_evals = ga.get("all_evals", {})
            ga_success_gen = ga.get("success_gen", {})
            ga_success_evals = ga.get("success_evals", {})
            ga_success_time = ga.get("success_time", {})
            
            writer.writerow([
                N,
                # BT
                int(bt["solution_found"]),
                bt["nodes"],
                
                # SA costi logici
                sa.get("success_rate", 0.0),
                sa_all_steps.get("mean", ""),
                sa_all_steps.get("median", ""),
                sa_all_evals.get("mean", ""), 
                sa_all_evals.get("median", ""),
                sa_success_steps.get("mean", ""),
                sa_success_evals.get("mean", ""),
                
                # GA costi logici
                ga.get("success_rate", 0.0),
                ga_all_gen.get("mean", ""),
                ga_all_gen.get("median", ""),
                ga_all_evals.get("mean", ""),
                ga_all_evals.get("median", ""),
                ga_success_gen.get("mean", ""),
                ga_success_evals.get("mean", ""),
                
                # Tempi sperimentali
                bt["time"],
                sa_success_time.get("mean", ""),
                ga_success_time.get("mean", ""),
            ])
    
    print(f"Logical cost analysis saved: {filename}")


def plot_comprehensive_analysis(results: ExperimentResults, N_values: List[int], fitness_mode: str, out_dir: str, raw_runs: Optional[Dict[str, Any]] = None, tuning_data: Optional[Dict[str, Any]] = None) -> None:
    """Generate the full set of charts for a comprehensive analysis.

    Produces success rates, time scaling (log), logical costs, timeout rates,
    failure quality, and theoretical vs practical correlations for all algorithms.
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # ===========================================
    # 1. BASE CHARTS 
    # ===========================================
    
    # Estrai dati base per tutti gli algoritmi
    bt_sr = [1.0 if results["BT"][N]["solution_found"] else 0.0 for N in N_values]
    sa_sr = [cast(float, results["SA"][N].get("success_rate", 0.0) or 0.0) for N in N_values]
    ga_sr = [cast(float, results["GA"][N].get("success_rate", 0.0) or 0.0) for N in N_values]
    
    bt_timeout = [0.0 for N in N_values]  # BT non ha timeout nei dati attuali
    sa_timeout = [results["SA"][N].get("timeout_rate", 0.0) for N in N_values]
    ga_timeout = [results["GA"][N].get("timeout_rate", 0.0) for N in N_values]
    
    # 1.1 Tasso di successo vs N
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
    # Aggiungi annotazioni
    for i, n in enumerate(N_values):
        plt.annotate(f'{bt_sr[i]:.1f}', (n, bt_sr[i]), textcoords="offset points", xytext=(0,5), ha='center', fontsize=9, color='blue')
        plt.annotate(f'{sa_sr[i]:.2f}', (n, sa_sr[i]), textcoords="offset points", xytext=(0,-15), ha='center', fontsize=9, color='orange')
        plt.annotate(f'{ga_sr[i]:.2f}', (n, ga_sr[i]), textcoords="offset points", xytext=(0,5), ha='center', fontsize=9, color='green')
    
    fname = os.path.join(out_dir, f"01_success_rate_vs_N_F{fitness_mode}.png")
    plt.savefig(fname, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved success-rate chart: {fname}")
    
    # 1.2 Tempo medio vs N (scala logaritmica, solo successi)
    bt_time = [results["BT"][N]["time"] if results["BT"][N]["solution_found"] else 0 for N in N_values]
    sa_time = [cast(float, results["SA"][N].get("success_time", {}).get("mean", 0.0) or 0.0) for N in N_values]
    ga_time = [cast(float, results["GA"][N].get("success_time", {}).get("mean", 0.0) or 0.0) for N in N_values]
    
    plt.figure(figsize=(12, 8))
    # Filtra valori zero per la scala log
    bt_time_plot = [max(t, 1e-6) for t in bt_time]
    sa_time_plot = [max(t, 1e-6) for t in sa_time]  
    ga_time_plot = [max(t, 1e-6) for t in ga_time]
    
    plt.semilogy(N_values, bt_time_plot, marker="o", linewidth=2, markersize=8, label="Backtracking")
    plt.semilogy(N_values, sa_time_plot, marker="s", linewidth=2, markersize=8, label="Simulated Annealing")
    plt.semilogy(N_values, ga_time_plot, marker="^", linewidth=2, markersize=8, label=f"Genetic Algorithm (F{fitness_mode})")
    plt.xlabel("N (Dimensione scacchiera)", fontsize=12)
    plt.ylabel("Average time [s] (log scale)", fontsize=12)
    plt.title("Execution Time vs Problem Size\n(Successful runs only — highlights BT growth)", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.7)
    plt.xticks(N_values)
    
    fname = os.path.join(out_dir, f"02_time_vs_N_log_scale_F{fitness_mode}.png")
    plt.savefig(fname, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved execution-time chart (log scale): {fname}")
    
    # 1.3 Costo logico vs N (indipendente dalla macchina)
    bt_nodes = [results["BT"][N]["nodes"] for N in N_values]
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
    
    # 1.4 Valutazioni di fitness vs N (SA vs GA)
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
    
    # ===========================================
    # 2. ANALISI TIMEOUT E FALLIMENTI
    # ===========================================
    
    # 2.1 Percentuale di timeout vs N
    plt.figure(figsize=(12, 8))
    plt.plot(N_values, sa_timeout, marker="s", linewidth=2, markersize=8, label="SA: Timeout rate")
    plt.plot(N_values, ga_timeout, marker="^", linewidth=2, markersize=8, label=f"GA-F{fitness_mode}: Timeout rate")
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
    
    # 2.2 Qualità nei fallimenti (best_conflicts nei run falliti)
    sa_fail_quality = [float(results["SA"][N].get("failure_best_conflicts", {}).get("mean", N) if results["SA"][N].get("failure_best_conflicts") else N) for N in N_values]
    ga_fail_quality = [float(results["GA"][N].get("failure_best_conflicts", {}).get("mean", N) if results["GA"][N].get("failure_best_conflicts") else N) for N in N_values]
    
    plt.figure(figsize=(12, 8))
    plt.plot(N_values, sa_fail_quality, marker="s", linewidth=2, markersize=8, label="SA: Conflitti medi (fallimenti)")
    plt.plot(N_values, ga_fail_quality, marker="^", linewidth=2, markersize=8, label=f"GA-F{fitness_mode}: Conflitti medi (fallimenti)")
    plt.plot(N_values, [0]*len(N_values), 'k--', alpha=0.5, label="Soluzione ottima (0 conflitti)")
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
    
    # ===========================================
    # 3. CONFRONTO TEORICO VS PRATICO
    # ===========================================
    
    # 3.1 Tempo vs Costo logico (SA)
    if any(sa_steps) and any(sa_time):
        plt.figure(figsize=(12, 8))
        # Filtra punti con dati validi
        valid_sa = [(s, t, n) for s, t, n in zip(sa_steps, sa_time, N_values) if s > 0 and t > 0]
        if valid_sa:
            steps_valid, time_valid, n_valid = zip(*valid_sa)
            plt.scatter(steps_valid, time_valid, c=n_valid, cmap='viridis', s=100, alpha=0.8)
            plt.colorbar(label='N (dimensione)')
            
            # Aggiungi linea di trend se possibile
            if len(valid_sa) > 2:
                z = np.polyfit(steps_valid, time_valid, 1)
                p = np.poly1d(z)
                x_trend = np.linspace(min(steps_valid), max(steps_valid), 100)
                plt.plot(x_trend, p(x_trend), "r--", alpha=0.8, label=f'Trend: y={z[0]:.2e}x+{z[1]:.2e}')
                plt.legend()
        
        plt.xlabel("Iterazioni SA (costo logico)", fontsize=12)
        plt.ylabel("Tempo [s] (costo pratico)", fontsize=12)
        plt.title("Simulated Annealing: Correlazione Costo Teorico vs Pratico\n(Linearità conferma dominio del costo di valutazione)", fontsize=14)
        plt.grid(True, alpha=0.7)
        
        fname = os.path.join(out_dir, f"07_SA_theoretical_vs_practical_F{fitness_mode}.png")
        plt.savefig(fname, bbox_inches="tight", dpi=300)
        plt.close()
    print(f"Saved SA theoretical-vs-practical chart: {fname}")
    
    # 3.2 Tempo vs Valutazioni fitness (GA)
    if any(ga_evals) and any(ga_time):
        plt.figure(figsize=(12, 8))
        valid_ga = [(e, t, n) for e, t, n in zip(ga_evals, ga_time, N_values) if e > 0 and t > 0]
        if valid_ga:
            evals_valid, time_valid, n_valid = zip(*valid_ga)
            plt.scatter(evals_valid, time_valid, c=n_valid, cmap='plasma', s=100, alpha=0.8)
            plt.colorbar(label='N (dimensione)')
            
            # Trend line
            if len(valid_ga) > 2:
                z = np.polyfit(evals_valid, time_valid, 1)
                p = np.poly1d(z)
                x_trend = np.linspace(min(evals_valid), max(evals_valid), 100)
                plt.plot(x_trend, p(x_trend), "r--", alpha=0.8, label=f'Trend: y={z[0]:.2e}x+{z[1]:.2e}')
                plt.legend()
        
        plt.xlabel("Valutazioni Fitness GA (costo logico)", fontsize=12)
        plt.ylabel("Tempo [s] (costo pratico)", fontsize=12)
        plt.title(f"GA-F{fitness_mode}: Correlazione Costo Teorico vs Pratico\n(Linearità conferma dominio del costo di valutazione)", fontsize=14)
        plt.grid(True, alpha=0.7)
        
        fname = os.path.join(out_dir, f"08_GA_theoretical_vs_practical_F{fitness_mode}.png")
        plt.savefig(fname, bbox_inches="tight", dpi=300)
        plt.close()
    print(f"Saved GA theoretical-vs-practical chart: {fname}")
    
    # 3.3 Tempo vs Nodi BT (conferma linearità)
    if any(bt_nodes) and any(bt_time):
        plt.figure(figsize=(12, 8))
        valid_bt = [(n, t, nval) for n, t, nval in zip(bt_nodes, bt_time, N_values) if n > 0 and t > 0]
        if valid_bt:
            nodes_valid, time_valid, n_valid = zip(*valid_bt)
            plt.scatter(nodes_valid, time_valid, c=n_valid, cmap='coolwarm', s=100, alpha=0.8)
            plt.colorbar(label='N (dimensione)')
            
            # Trend line
            if len(valid_bt) > 2:
                z = np.polyfit(nodes_valid, time_valid, 1)
                p = np.poly1d(z)
                x_trend = np.linspace(min(nodes_valid), max(nodes_valid), 100)
                plt.plot(x_trend, p(x_trend), "r--", alpha=0.8, label=f'Trend: y={z[0]:.2e}x+{z[1]:.2e}')
                plt.legend()
        
        plt.xlabel("Nodi Esplorati BT (costo logico)", fontsize=12)
        plt.ylabel("Tempo [s] (costo pratico)", fontsize=12)
        plt.title("Backtracking: Correlazione Costo Teorico vs Pratico\n(Ogni nodo costa tempo quasi costante)", fontsize=14)
        plt.grid(True, alpha=0.7)
        
        fname = os.path.join(out_dir, f"09_BT_theoretical_vs_practical_F{fitness_mode}.png")
        plt.savefig(fname, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"Saved BT theoretical-vs-practical chart: {fname}")
    
    print(f"\nComplete analysis generated in: {out_dir}")
    print(f"Generated {9} base charts for fitness F{fitness_mode}")


def plot_fitness_comparison(all_results: Dict[str, ExperimentResults], N_values: List[int], out_dir: str, raw_runs: Optional[Dict[str, Any]] = None) -> None:
    """Compare GA fitness functions (F1–F6) across selected N values.

    Generates success-rate, generation, and time comparisons per fitness,
    plus trade-off scatter plots and evolution across N.
    """
    os.makedirs(out_dir, exist_ok=True)
    fitness_modes = list(all_results.keys())
    
    # Colori per le diverse fitness
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    fitness_colors = {f: colors[i % len(colors)] for i, f in enumerate(fitness_modes)}
    
    # ===========================================
    # 1. SUCCESS RATE PER FITNESS (fixed N)
    # ===========================================
    
    # Scegli N rappresentativi per l'analisi
    analysis_N = [n for n in [16, 24, 40] if n in N_values]  # N piccoli, medi, grandi
    
    for N in analysis_N:
        # Bar chart: success rate per fitness
        plt.figure(figsize=(12, 8))
        success_rates = [cast(float, all_results[f]["GA"][N].get("success_rate", 0.0) or 0.0) for f in fitness_modes]
        
        bars = plt.bar(fitness_modes, success_rates, color=[fitness_colors[f] for f in fitness_modes], alpha=0.8)
        plt.xlabel("Fitness Function", fontsize=12)
        plt.ylabel("Success rate", fontsize=12)
        plt.title(f"Success Rate Comparison across Fitness Functions (N={N})\n(Which fitness converges better at the same size)", fontsize=14)
        plt.ylim(0, 1.05)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Aggiungi valori sui bar
        for bar, sr in zip(bars, success_rates):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{sr:.3f}', ha='center', va='bottom', fontweight='bold')
        
        fname = os.path.join(out_dir, f"fitness_success_rate_N{N}.png")
        plt.savefig(fname, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"Saved success-rate comparison for N={N}: {fname}")
        
        # ===========================================
    # 2. AVERAGE GENERATIONS PER FITNESS
        # ===========================================
        
        plt.figure(figsize=(12, 8))
        gen_means = []
        gen_stds = []
        
        for f in fitness_modes:
            gen_stats = all_results[f]["GA"][N]["success_gen"]
            gen_means.append(gen_stats.get("mean", 0))
            gen_stds.append(gen_stats.get("std", 0))
        
        bars = plt.bar(fitness_modes, gen_means, yerr=gen_stds, 
                      color=[fitness_colors[f] for f in fitness_modes], 
                      alpha=0.8, capsize=5)
        plt.xlabel("Fitness Function", fontsize=12)
        plt.ylabel("Average generations +/- std", fontsize=12)
        plt.title(f"Convergence Speed Comparison (N={N})\n(Generations to reach a solution)", fontsize=14)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Valori sui bar
        for bar, mean, std in zip(bars, gen_means, gen_stds):
            if mean > 0:
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.5,
                        f'{mean:.1f}+/-{std:.1f}', ha='center', va='bottom', fontsize=10)
        
        fname = os.path.join(out_dir, f"fitness_generations_N{N}.png")
        plt.savefig(fname, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"Saved generation comparison for N={N}: {fname}")
        
        # ===========================================
    # 3. AVERAGE TIME PER FITNESS
        # ===========================================
        
        plt.figure(figsize=(12, 8))
        time_means = []
        time_stds = []
        
        for f in fitness_modes:
            time_stats = all_results[f]["GA"][N]["success_time"]
            time_means.append(time_stats.get("mean", 0))
            time_stds.append(time_stats.get("std", 0))
        
        bars = plt.bar(fitness_modes, time_means, yerr=time_stds,
                      color=[fitness_colors[f] for f in fitness_modes], 
                      alpha=0.8, capsize=5)
        plt.xlabel("Fitness Function", fontsize=12)
        plt.ylabel("Average time [s] +/- std", fontsize=12)
        plt.title(f"Temporal Efficiency Comparison (N={N})\n(Trade-off between success and speed)", fontsize=14)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Valori sui bar
        for bar, mean, std in zip(bars, time_means, time_stds):
            if mean > 0:
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.001,
                        f'{mean:.3f}+/-{std:.3f}', ha='center', va='bottom', fontsize=10, rotation=0)
        
        fname = os.path.join(out_dir, f"fitness_time_N{N}.png")
        plt.savefig(fname, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"Saved time comparison for N={N}: {fname}")
    
    # ===========================================
    # 4. TRADE-OFF SUCCESS RATE VS COSTO (scatter)
    # ===========================================
    
    for N in analysis_N:
        plt.figure(figsize=(12, 8))
        
        success_rates = []
        gen_means = []
        eval_means = []
        
        for f in fitness_modes:
            ga_data = all_results[f]["GA"][N]
            success_rates.append(ga_data["success_rate"])
            gen_means.append(ga_data["success_gen"].get("mean", 0))
            eval_means.append(ga_data["success_evals"].get("mean", 0))
        
        # Scatter: success rate vs average generations
        scatter = plt.scatter(gen_means, success_rates, 
                            c=[fitness_colors[f] for f in fitness_modes], 
                            s=150, alpha=0.8, edgecolors='black', linewidth=2)
        
        # Etichette per ogni punto
        for f, x, y in zip(fitness_modes, gen_means, success_rates):
            plt.annotate(f'F{f}', (x, y), xytext=(5, 5), 
                        textcoords='offset points', fontweight='bold', fontsize=12)
        
        plt.xlabel("Average generations to success", fontsize=12)
        plt.ylabel("Success rate", fontsize=12)
        plt.title(f"Quality vs Cost Trade-off (N={N})\n(Pareto front: high success, low cost)", fontsize=14)
        plt.grid(True, alpha=0.7)
        
        # Evidenzia area Pareto-ottima (alto successo, basso costo)
        max_sr = max(success_rates)
        min_gen = min([g for g in gen_means if g > 0])
        plt.axhline(y=max_sr*0.9, color='green', linestyle='--', alpha=0.5, label='Alto successo (>90% max)')
        plt.axvline(x=min_gen*1.5, color='red', linestyle='--', alpha=0.5, label='Basso costo (<150% min)')
        plt.legend()
        
        fname = os.path.join(out_dir, f"fitness_tradeoff_N{N}.png")
        plt.savefig(fname, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"Saved fitness trade-off for N={N}: {fname}")
    
    # ===========================================
    # 5. EVOLUZIONE SUCCESS RATE TUTTE LE FITNESS
    # ===========================================
    
    plt.figure(figsize=(15, 10))
    for f in fitness_modes:
        ga_sr_all_N = [cast(float, all_results[f]["GA"][N].get("success_rate", 0.0) or 0.0) for N in N_values]
        plt.plot(N_values, ga_sr_all_N, marker='o', linewidth=2, markersize=8, 
                label=f'GA-F{f}', color=fitness_colors[f])
    
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


def plot_and_save(results: ExperimentResults, N_values: List[int], fitness_mode: str, out_dir: str) -> None:
    """Compatibility wrapper that calls the comprehensive plotting routine."""
    plot_comprehensive_analysis(results, N_values, fitness_mode, out_dir)


# ======================================================
# 9. Main: versione parallela del tuning + esperimenti finali
# ======================================================

def main_sequential(fitness_modes: Optional[List[str]] = None, skip_tuning: bool = False, config_mgr: Optional[ConfigManager] = None) -> None:
    """Run the sequential pipeline for the selected fitness modes."""

    os.makedirs(OUT_DIR, exist_ok=True)
    selected_fitness = fitness_modes or FITNESS_MODES

    for fitness_mode in selected_fitness:
        print("\n============================================")
        print(f"SEQUENTIAL PIPELINE FOR GA FITNESS {fitness_mode}")
        print("============================================")

        if skip_tuning:
            print("Skipping GA tuning and reusing parameters from configuration.")
            best_ga_params_for_N = load_optimal_parameters(fitness_mode, config_mgr, N_VALUES)
        else:
            print("Starting GA tuning (sequential search).")
            best_ga_params_for_N = {}
            tuning_csv = os.path.join(OUT_DIR, f"tuning_GA_{fitness_mode}_seq.csv")
            progress = ProgressPrinter(len(N_VALUES), f"Tuning GA-{fitness_mode}")

            with open(tuning_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "N",
                    "pop_size",
                    "max_gen",
                    "pm",
                    "pc",
                    "tournament_size",
                    "success_rate_tuning",
                    "avg_gen_success_tuning",
                ])

                for index, N in enumerate(N_VALUES, start=1):
                    progress.update(index, f"N={N}")
                    print(f"Tuning GA: N = {N}, fitness = {fitness_mode}")
                    best = tune_ga_for_N(
                        N,
                        fitness_mode,
                        POP_MULTIPLIERS,
                        GEN_MULTIPLIERS,
                        PM_VALUES,
                        PC_FIXED,
                        TOURNAMENT_SIZE_FIXED,
                        runs_tuning=RUNS_GA_TUNING,
                    )
                    best_ga_params_for_N[N] = best
                    print("  Best parameters:", best)

                    writer.writerow([
                        N,
                        best["pop_size"],
                        best["max_gen"],
                        best["pm"],
                        best["pc"],
                        best["tournament_size"],
                        best["success_rate"],
                        best["avg_gen_success"],
                    ])

            print(f"Tuning export CSV: {tuning_csv}")

            if config_mgr:
                config_mgr.save_optimal_parameters(fitness_mode, best_ga_params_for_N)

        print(f"\nRunning final experiments for GA fitness {fitness_mode}")
        results = run_experiments_with_best_ga(
            N_VALUES,
            runs_sa=RUNS_SA_FINAL,
            runs_ga=RUNS_GA_FINAL,
            bt_time_limit=BT_TIME_LIMIT,
            fitness_mode=fitness_mode,
            best_ga_params_for_N=best_ga_params_for_N,
            progress_label=f"Experiments GA-{fitness_mode}",
        )

        save_results_to_csv(results, N_VALUES, fitness_mode, OUT_DIR)
        save_raw_data_to_csv(results, N_VALUES, fitness_mode, OUT_DIR)
        save_logical_cost_analysis(results, N_VALUES, fitness_mode, OUT_DIR)
        plot_and_save(results, N_VALUES, fitness_mode, OUT_DIR)

    print("\nSequential pipeline completed.")


def main_parallel(fitness_modes: Optional[List[str]] = None, skip_tuning: bool = False, config_mgr: Optional[ConfigManager] = None) -> None:
    """Run the parallel pipeline for the selected fitness modes."""

    os.makedirs(OUT_DIR, exist_ok=True)
    selected_fitness = fitness_modes or FITNESS_MODES

    print(f"\nStarting parallel pipeline with {NUM_PROCESSES} worker processes")
    print(f"Available CPU cores: {multiprocessing.cpu_count()}")
    print("Configured timeouts:")
    print(f"   - BT: {BT_TIME_LIMIT}s" if BT_TIME_LIMIT else "   - BT: unlimited")
    print(f"   - SA: {SA_TIME_LIMIT}s" if SA_TIME_LIMIT else "   - SA: unlimited")
    print(f"   - GA: {GA_TIME_LIMIT}s" if GA_TIME_LIMIT else "   - GA: unlimited")
    print(f"   - Experiment: {EXPERIMENT_TIMEOUT}s" if EXPERIMENT_TIMEOUT else "   - Experiment: unlimited")

    start_total = perf_counter()
    all_best_params = {}

    if skip_tuning:
        print("\nSkipping GA tuning phase and loading parameters from configuration.")
        for fitness_mode in selected_fitness:
            all_best_params[fitness_mode] = load_optimal_parameters(fitness_mode, config_mgr, N_VALUES)
    else:
        print("\n" + "=" * 60)
        print("PHASE 1: PARALLEL GA TUNING")
        print("=" * 60)

        for fitness_mode in selected_fitness:
            print(f"\nTuning fitness {fitness_mode}...")
            fitness_start = perf_counter()

            best_ga_params_for_N = {}
            tuning_csv = os.path.join(OUT_DIR, f"tuning_GA_{fitness_mode}.csv")
            progress = ProgressPrinter(len(N_VALUES), f"Tuning GA-{fitness_mode}")

            with open(tuning_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "N",
                    "pop_size",
                    "max_gen",
                    "pm",
                    "pc",
                    "tournament_size",
                    "success_rate_tuning",
                    "avg_gen_success_tuning",
                ])

                for index, N in enumerate(N_VALUES, start=1):
                    progress.update(index, f"N={N}")
                    tuning_start = perf_counter()

                    best = tune_ga_for_N_parallel(
                        N,
                        fitness_mode,
                        POP_MULTIPLIERS,
                        GEN_MULTIPLIERS,
                        PM_VALUES,
                        PC_FIXED,
                        TOURNAMENT_SIZE_FIXED,
                        runs_tuning=RUNS_GA_TUNING,
                    )

                    tuning_time = perf_counter() - tuning_start
                    best_ga_params_for_N[N] = best
                    print(
                        f"     Completed in {tuning_time:.1f}s - Success rate: {best['success_rate']:.3f}"
                    )

                    writer.writerow([
                        N,
                        best["pop_size"],
                        best["max_gen"],
                        best["pm"],
                        best["pc"],
                        best["tournament_size"],
                        best["success_rate"],
                        best["avg_gen_success"],
                    ])

            all_best_params[fitness_mode] = best_ga_params_for_N

            fitness_time = perf_counter() - fitness_start
            print(f"Tuning {fitness_mode} completed in {fitness_time:.1f}s - CSV: {tuning_csv}")

            if config_mgr:
                config_mgr.save_optimal_parameters(fitness_mode, best_ga_params_for_N)

    print("\n" + "=" * 60)
    print("PHASE 2: PARALLEL FINAL EXPERIMENTS")
    print("=" * 60)

    for fitness_mode in selected_fitness:
        print(f"\nFinal experiments for {fitness_mode}...")
        experiments_start = perf_counter()

        results = run_experiments_with_best_ga_parallel(
            N_VALUES,
            runs_sa=RUNS_SA_FINAL,
            runs_ga=RUNS_GA_FINAL,
            bt_time_limit=BT_TIME_LIMIT,
            fitness_mode=fitness_mode,
            best_ga_params_for_N=all_best_params[fitness_mode],
            progress_label=f"Experiments GA-{fitness_mode}",
        )

        experiments_time = perf_counter() - experiments_start
        print(f"  Experiments completed in {experiments_time:.1f}s")

        print("Generating charts and CSV reports...")
        save_results_to_csv(results, N_VALUES, fitness_mode, OUT_DIR)
        save_raw_data_to_csv(results, N_VALUES, fitness_mode, OUT_DIR)
        save_logical_cost_analysis(results, N_VALUES, fitness_mode, OUT_DIR)
        plot_and_save(results, N_VALUES, fitness_mode, OUT_DIR)
        print(f"  Results saved for {fitness_mode}")

    total_time = perf_counter() - start_total
    print("\nParallel pipeline completed!")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Fitness processed: {len(selected_fitness)}")
    print(f"Worker processes used: {NUM_PROCESSES}")


def main_concurrent_tuning(fitness_modes: Optional[List[str]] = None, skip_tuning: bool = False, config_mgr: Optional[ConfigManager] = None) -> None:
    """Run concurrent tuning and experiments across the selected fitness modes."""

    os.makedirs(OUT_DIR, exist_ok=True)
    selected_fitness = fitness_modes or FITNESS_MODES

    print("\nCONCURRENT TUNING FOR SELECTED FITNESS FUNCTIONS")
    print(f"Fitness modes: {selected_fitness}")
    print(f"Processes: {NUM_PROCESSES}")
    print(f"Available CPU cores: {multiprocessing.cpu_count()}")
    print("Configured timeouts:")
    print(f"   - BT: {BT_TIME_LIMIT}s" if BT_TIME_LIMIT else "   - BT: unlimited")
    print(f"   - SA: {SA_TIME_LIMIT}s" if SA_TIME_LIMIT else "   - SA: unlimited")
    print(f"   - GA: {GA_TIME_LIMIT}s" if GA_TIME_LIMIT else "   - GA: unlimited")
    print(f"   - Experiment: {EXPERIMENT_TIMEOUT}s" if EXPERIMENT_TIMEOUT else "   - Experiment: unlimited")

    start_total = perf_counter()
    all_best_params = {fitness_mode: {} for fitness_mode in selected_fitness}

    if skip_tuning:
        print("\nSkipping GA tuning phase and loading parameters from configuration.")
        for fitness_mode in selected_fitness:
            all_best_params[fitness_mode] = load_optimal_parameters(fitness_mode, config_mgr, N_VALUES)
    else:
        print("\n" + "=" * 70)
        print("PHASE 1: PARALLEL TUNING FOR ALL FITNESS FUNCTIONS")
        print("=" * 70)

        progress = ProgressPrinter(len(N_VALUES), "Concurrent GA tuning")

        for index, N in enumerate(N_VALUES, start=1):
            progress.update(index, f"N={N}")
            print(f"\nParallel tuning for N = {N}")
            print("-" * 50)

            fitness_results = tune_all_fitness_parallel(
                N,
                selected_fitness,
                POP_MULTIPLIERS,
                GEN_MULTIPLIERS,
                PM_VALUES,
                PC_FIXED,
                TOURNAMENT_SIZE_FIXED,
                runs_tuning=RUNS_GA_TUNING,
            )

            for fitness_mode, best_params in fitness_results.items():
                all_best_params[fitness_mode][N] = best_params

        for fitness_mode in selected_fitness:
            save_tuning_results(all_best_params[fitness_mode], fitness_mode, OUT_DIR)
            if config_mgr:
                config_mgr.save_optimal_parameters(fitness_mode, all_best_params[fitness_mode])

    print("\n" + "=" * 70)
    print("PHASE 2: FINAL EXPERIMENTS WITH OPTIMAL PARAMETERS")
    print("=" * 70)

    all_results = {}

    for fitness_mode in selected_fitness:
        print(f"\nFinal experiments GA-{fitness_mode}")

        results = run_experiments_with_best_ga_parallel(
            N_VALUES,
            runs_sa=RUNS_SA_FINAL,
            runs_ga=RUNS_GA_FINAL,
            bt_time_limit=BT_TIME_LIMIT,
            fitness_mode=fitness_mode,
            best_ga_params_for_N=all_best_params[fitness_mode],
            progress_label=f"Experiments GA-{fitness_mode}",
        )

        all_results[fitness_mode] = results

        save_results_to_csv(results, N_VALUES, fitness_mode, OUT_DIR)
        save_raw_data_to_csv(results, N_VALUES, fitness_mode, OUT_DIR)
        save_logical_cost_analysis(results, N_VALUES, fitness_mode, OUT_DIR)
        plot_and_save(results, N_VALUES, fitness_mode, OUT_DIR)

    print("\n" + "=" * 70)
    print("PHASE 3: COMPARATIVE ANALYSIS AND ADVANCED CHARTS")
    print("=" * 70)

    for fitness in selected_fitness:
        print(f"  Comprehensive analysis for GA-F{fitness}...")
        plot_comprehensive_analysis(
            all_results[fitness],
            N_VALUES,
            fitness,
            os.path.join(OUT_DIR, f"analysis_F{fitness}"),
            raw_runs=None,
        )

    print("  Comparing fitness functions...")
    plot_fitness_comparison(
        all_results,
        N_VALUES,
        os.path.join(OUT_DIR, "fitness_comparison"),
    )

    print("  Statistical analysis...")
    plot_statistical_analysis(
        all_results,
        N_VALUES,
        os.path.join(OUT_DIR, "statistical_analysis"),
        raw_runs=None,
    )

    total_time = perf_counter() - start_total
    print("\nConcurrent pipeline completed!")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Fitness processed: {len(selected_fitness)}")

def plot_statistical_analysis(all_results: Dict[str, ExperimentResults], N_values: List[int], out_dir: str, raw_runs: Optional[Dict[int, Dict[str, Any]]] = None) -> None:
    """Statistical charts: boxplots and variability analysis.

    Requires raw per-run data for meaningful distribution plots.
    """
    os.makedirs(out_dir, exist_ok=True)
    
    if not raw_runs:
        print("Raw runs non disponibili per analisi statistica dettagliata")
        return
    
    # Analizza un subset rappresentativo di N 
    analysis_N = [n for n in [16, 24, 40] if n in N_values and n in raw_runs]
    
    for N in analysis_N:
        if N not in raw_runs:
            continue

        print(f"Analisi statistica per N={N}...")
        
        # ===========================================
        # 1. BOXPLOT DEI TEMPI (solo successi)
        # ===========================================
        
        plt.figure(figsize=(14, 8))
        
        # Raccogli dati temporali per tutti gli algoritmi
        time_data = []
        labels = []
        
        # SA tempi (solo successi)
        if "SA" in raw_runs[N]:
            sa_times = [run["time"] for run in raw_runs[N]["SA"] if run["success"]]
            if sa_times:
                time_data.append(sa_times)
                labels.append("SA")
        
        # GA tempi per ogni fitness (solo successi)
        for fitness in sorted(all_results.keys()):
            if fitness in raw_runs[N]:
                ga_times = [run["time"] for run in raw_runs[N][fitness] if run["success"]]
                if ga_times:
                    time_data.append(ga_times)
                    labels.append(f"GA-F{fitness}")
        
        if time_data:
            box_plot = plt.boxplot(time_data, labels=labels, patch_artist=True)
            
            # Colora i boxplot
            colors = ['#ff7f0e', '#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
            for patch, color in zip(box_plot['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            plt.ylabel("Tempo di Esecuzione [s]", fontsize=12)
            plt.title(f"Distribuzione Tempi di Esecuzione (N={N}, solo successi)\n(Boxplot mostra mediana, quartili, outliers)", fontsize=14)
            plt.yscale('log')
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            
            fname = os.path.join(out_dir, f"boxplot_times_N{N}.png")
            plt.savefig(fname, bbox_inches="tight", dpi=300)
            plt.close()
            print(f"Boxplot tempi N={N}: {fname}")
        
        # ===========================================
        # 2. BOXPLOT DELLE ITERAZIONI/GENERAZIONI
        # ===========================================
        
        plt.figure(figsize=(14, 8))
        
        iter_data = []
        iter_labels = []
        
        # SA iterazioni (solo successi)
        if "SA" in raw_runs[N]:
            sa_steps = [run["steps"] for run in raw_runs[N]["SA"] if run["success"]]
            if sa_steps:
                iter_data.append(sa_steps)
                iter_labels.append("SA (steps)")
        
        # GA generazioni per ogni fitness (solo successi)
        for fitness in sorted(all_results.keys()):
            if fitness in raw_runs[N]:
                ga_gens = [run["gen"] for run in raw_runs[N][fitness] if run["success"]]
                if ga_gens:
                    iter_data.append(ga_gens)
                    iter_labels.append(f"GA-F{fitness} (gen)")
        
        if iter_data:
            box_plot = plt.boxplot(iter_data, labels=iter_labels, patch_artist=True)
            
            # Colora i boxplot
            for patch, color in zip(box_plot['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            plt.ylabel("Iterazioni/Generazioni", fontsize=12)
            plt.title(f"Distribuzione Costi Logici (N={N}, solo successi)\n(Variabilità dell'algoritmo in termini di sforzo)", fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            
            fname = os.path.join(out_dir, f"boxplot_iterations_N{N}.png")
            plt.savefig(fname, bbox_inches="tight", dpi=300)
            plt.close()
            print(f"Boxplot iterazioni N={N}: {fname}")
        
        # ===========================================
        # 3. ISTOGRAMMI DISTRIBUZIONI
        # ===========================================
        
        # SA histogram
        if "SA" in raw_runs[N]:
            sa_times = [run["time"] for run in raw_runs[N]["SA"] if run["success"]]
            if len(sa_times) > 5:  # Abbastanza dati per istogramma
                plt.figure(figsize=(12, 6))
                plt.hist(sa_times, bins=min(20, len(sa_times)//2), alpha=0.7, color='orange', edgecolor='black')
                plt.xlabel("Tempo [s]", fontsize=12)
                plt.ylabel("Frequenza", fontsize=12)
                plt.title(f"Distribuzione Tempi SA (N={N})\n(Forma della distribuzione indica stabilità algoritmo)", fontsize=14)
                plt.grid(True, alpha=0.3)
                
                # Aggiungi statistiche
                mean_time = np.mean(sa_times)
                std_time = np.std(sa_times)
                plt.axvline(mean_time, color='red', linestyle='--', label=f'Media: {mean_time:.3f}s')
                plt.axvline(mean_time + std_time, color='red', linestyle=':', alpha=0.7, label=f'+/-1 sigma: {std_time:.3f}s')
                plt.axvline(mean_time - std_time, color='red', linestyle=':', alpha=0.7)
                plt.legend()
                
                fname = os.path.join(out_dir, f"histogram_SA_times_N{N}.png")
                plt.savefig(fname, bbox_inches="tight", dpi=300)
                plt.close()
                print(f"Istogramma SA tempi N={N}: {fname}")
        
        # GA histogram per la migliore fitness
        best_fitness = min(all_results.keys(), key=lambda f: -all_results[f]["GA"][N]["success_rate"])
        if best_fitness in raw_runs[N]:
            ga_times = [run["time"] for run in raw_runs[N][best_fitness] if run["success"]]
            if len(ga_times) > 5:
                plt.figure(figsize=(12, 6))
                plt.hist(ga_times, bins=min(20, len(ga_times)//2), alpha=0.7, color='green', edgecolor='black')
                plt.xlabel("Tempo [s]", fontsize=12)
                plt.ylabel("Frequenza", fontsize=12)
                plt.title(f"Distribuzione Tempi GA-F{best_fitness} (N={N})\n(Algoritmo più stabile = distribuzione stretta)", fontsize=14)
                plt.grid(True, alpha=0.3)
                
                # Statistiche
                mean_time = np.mean(ga_times)
                std_time = np.std(ga_times)
                plt.axvline(mean_time, color='red', linestyle='--', label=f'Media: {mean_time:.3f}s')
                plt.axvline(mean_time + std_time, color='red', linestyle=':', alpha=0.7, label=f'+/-1 sigma: {std_time:.3f}s')
                plt.axvline(mean_time - std_time, color='red', linestyle=':', alpha=0.7)
                plt.legend()
                
                fname = os.path.join(out_dir, f"histogram_GA_F{best_fitness}_times_N{N}.png")
                plt.savefig(fname, bbox_inches="tight", dpi=300)
                plt.close()
                print(f"Istogramma GA-F{best_fitness} tempi N={N}: {fname}")
    
    print("Analisi statistica completata")


def plot_tuning_analysis(tuning_data: Dict[str, Dict[int, List[Dict[str, Any]]]], fitness_modes: List[str], N_values: List[int], out_dir: str) -> None:
    """Visual analysis of GA tuning data (heatmaps and scatter plots)."""
    if not tuning_data:
        print("Dati tuning non disponibili")
        return
        
    os.makedirs(out_dir, exist_ok=True)
    
    # Per ogni fitness
    for fitness in fitness_modes:
        if fitness not in tuning_data:
            continue

        print(f"Analisi tuning GA-F{fitness}...")
        
        # Scegli N rappresentativo per analisi dettagliata
        analysis_N = [n for n in [24, 40] if n in N_values and n in tuning_data[fitness]]
        
        for N in analysis_N:
            if N not in tuning_data[fitness]:
                continue
                
            tuning_runs = tuning_data[fitness][N]
            if not tuning_runs:
                continue
            
            # ===========================================
            # 1. HEATMAP SUCCESS RATE vs POP_SIZE, MAX_GEN
            # ===========================================
            
            # Raccogli dati in matrice
            pop_sizes = sorted(set(run['pop_size'] for run in tuning_runs))
            max_gens = sorted(set(run['max_gen'] for run in tuning_runs))
            
            if len(pop_sizes) > 1 and len(max_gens) > 1:
                # Crea matrice success_rate
                sr_matrix = np.zeros((len(max_gens), len(pop_sizes)))
                
                for i, mg in enumerate(max_gens):
                    for j, ps in enumerate(pop_sizes):
                        # Trova run con questi parametri
                        matching_runs = [r for r in tuning_runs 
                                       if r['pop_size'] == ps and r['max_gen'] == mg]
                        if matching_runs:
                            sr_matrix[i, j] = matching_runs[0]['success_rate']
                
                # Plot heatmap
                plt.figure(figsize=(12, 8))
                im = plt.imshow(sr_matrix, cmap='RdYlGn', aspect='auto', interpolation='nearest')
                plt.colorbar(im, label='Success Rate')
                
                # Labels
                plt.xticks(range(len(pop_sizes)), [f'{ps}' for ps in pop_sizes])
                plt.yticks(range(len(max_gens)), [f'{mg}' for mg in max_gens])
                plt.xlabel("Population Size", fontsize=12)
                plt.ylabel("Max Generations", fontsize=12)
                plt.title(f"GA-F{fitness} Tuning Heatmap (N={N})\n(Zona verde = parametri ottimali)", fontsize=14)
                
                # Aggiungi valori nelle celle
                for i in range(len(max_gens)):
                    for j in range(len(pop_sizes)):
                        if sr_matrix[i, j] > 0:
                            plt.text(j, i, f'{sr_matrix[i, j]:.2f}', 
                                   ha="center", va="center", fontweight='bold',
                                   color='white' if sr_matrix[i, j] < 0.5 else 'black')
                
                fname = os.path.join(out_dir, f"heatmap_tuning_GA_F{fitness}_N{N}.png")
                plt.savefig(fname, bbox_inches="tight", dpi=300)
                plt.close()
                print(f"Heatmap tuning GA-F{fitness} N={N}: {fname}")
            
            # ===========================================
            # 2. SCATTER: COSTO vs QUALITÀ
            # ===========================================
            
            plt.figure(figsize=(12, 8))
            
            costs = [run['pop_size'] * run['max_gen'] for run in tuning_runs]
            success_rates = [run['success_rate'] for run in tuning_runs]
            
            # Scatter colorato per mutation rate se disponibile
            if 'pm' in tuning_runs[0]:
                pms = [run['pm'] for run in tuning_runs]
                scatter = plt.scatter(costs, success_rates, c=pms, cmap='viridis', 
                                    s=100, alpha=0.7, edgecolors='black')
                plt.colorbar(scatter, label='Mutation Rate (pm)')
            else:
                plt.scatter(costs, success_rates, s=100, alpha=0.7, edgecolors='black')
            
            plt.xlabel("Costo Computazionale (pop_size x max_gen)", fontsize=12)
            plt.ylabel("Tasso di Successo", fontsize=12)
            plt.title(f"GA-F{fitness}: Trade-off Costo vs Qualità (N={N})\n(Mostra se vale la pena aumentare parametri)", fontsize=14)
            plt.grid(True, alpha=0.3)
            
            # Evidenzia configurazioni Pareto-ottimali
            max_sr = max(success_rates)
            pareto_points = [(c, sr) for c, sr in zip(costs, success_rates) 
                           if sr >= max_sr * 0.95]  # Entro 95% del massimo
            
            if pareto_points:
                min_cost_pareto = min(p[0] for p in pareto_points)
                plt.axhline(y=max_sr*0.95, color='red', linestyle='--', alpha=0.5, 
                          label=f'95% max success ({max_sr*0.95:.2f})')
                plt.axvline(x=min_cost_pareto*1.1, color='green', linestyle='--', alpha=0.5,
                          label=f'Costo efficiente (<{min_cost_pareto*1.1:.0f})')
                plt.legend()
            
            fname = os.path.join(out_dir, f"scatter_cost_quality_GA_F{fitness}_N{N}.png")
            plt.savefig(fname, bbox_inches="tight", dpi=300)
            plt.close()
            print(f"Scatter costo-qualità GA-F{fitness} N={N}: {fname}")
            
            # ===========================================
            # 3. LINE PLOT per MUTATION RATE
            # ===========================================
            
            if 'pm' in tuning_runs[0]:
                pm_values = sorted(set(run['pm'] for run in tuning_runs))
                
                if len(pm_values) > 1:
                    plt.figure(figsize=(12, 8))
                    
                    # Per ogni pm, calcola success rate medio
                    pm_sr_means = []
                    pm_sr_stds = []
                    
                    for pm in pm_values:
                        pm_runs = [run for run in tuning_runs if run['pm'] == pm]
                        srs = [run['success_rate'] for run in pm_runs]
                        pm_sr_means.append(np.mean(srs))
                        pm_sr_stds.append(np.std(srs) if len(srs) > 1 else 0)
                    
                    plt.errorbar(pm_values, pm_sr_means, yerr=pm_sr_stds, 
                               marker='o', linewidth=2, markersize=8, capsize=5)
                    plt.xlabel("Mutation Rate (pm)", fontsize=12)
                    plt.ylabel("Success Rate Medio +/- Std", fontsize=12)
                    plt.title(f"GA-F{fitness}: Effetto Mutation Rate (N={N})\n(Mostra sensibilità alla mutazione)", fontsize=14)
                    plt.grid(True, alpha=0.3)
                    plt.ylim(0, 1.05)
                    
                    fname = os.path.join(out_dir, f"lineplot_mutation_GA_F{fitness}_N{N}.png")
                    plt.savefig(fname, bbox_inches="tight", dpi=300)
                    plt.close()
                    print(f"Line plot mutation GA-F{fitness} N={N}: {fname}")
    
    print("Analisi tuning completata")


def save_tuning_results(best_params_for_N: Dict[int, Dict[str, Any]], fitness_mode: str, out_dir: str) -> None:
    """Save tuning results for one fitness mode to a CSV file."""
    filename = os.path.join(out_dir, f"tuning_GA_F{fitness_mode}.csv")
    
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "N", "pop_size", "max_gen", "pm", "pc", "tournament_size", 
            "success_rate_tuning", "avg_gen_success_tuning"
        ])
        
        for N in sorted(best_params_for_N.keys()):
            params = best_params_for_N[N]
            writer.writerow([
                N,
                params.get("pop_size", ""),
                params.get("max_gen", ""),
                params.get("pm", ""),
                params.get("pc", ""),
                params.get("tournament_size", ""),
                params.get("success_rate", ""),
                params.get("avg_gen_success", "")
            ])
    
    print(f"Risultati tuning GA-F{fitness_mode} salvati: {filename}")


def run_quick_regression_tests() -> None:
    """Run lightweight deterministic checks for all BT solvers, SA, GA, and CSV generation.

    - Discovers all functions in `nqueens.backtracking` named `bt_nqueens_*` and tests them on N=8.
    - Runs a short SA and GA check with fixed seeds.
    - Generates a minimal CSV via `run_experiments_with_best_ga` to verify I/O path.
    """

    print("Running quick regression tests (N=8) across all algorithms...")

    # Backtracking: discover and test all current and future solvers
    import nqueens.backtracking as bt_mod
    bt_solvers = [(name, fn) for name, fn in inspect.getmembers(bt_mod, inspect.isfunction) if name.startswith("bt_nqueens_")]
    if not bt_solvers:
        raise AssertionError("No backtracking solvers discovered (expected functions named 'bt_nqueens_*').")

    # Sort for stable output
    bt_solvers.sort(key=lambda x: x[0])

    for name, solver in bt_solvers:
        random.seed(42)
        try:
            solution, nodes, elapsed = solver(8, time_limit=5.0)
        except TypeError:
            solution, nodes, elapsed = solver(8, 5.0)
        if solution is None:
            raise AssertionError(f"{name} failed to find a solution for N=8.")
        if not isinstance(nodes, int) or nodes <= 0:
            raise AssertionError(f"{name} returned invalid nodes count: {nodes}.")
        print(f"  [BT] {name}: solution found, nodes={nodes}, time={elapsed:.4f}s")

    random.seed(42)
    sa_success, _, sa_time, _, _, sa_timeout = sa_nqueens(
        8, max_iter=5000, T0=1.0, alpha=0.995, time_limit=5.0
    )
    if not sa_success or sa_timeout:
        raise AssertionError("Simulated Annealing did not succeed for N=8 with deterministic seed.")
    print(f"  Simulated Annealing: success in {sa_time:.4f}s")

    random.seed(42)
    ga_success, _, ga_time, _, _, ga_timeout = ga_nqueens(
        8,
        pop_size=60,
        max_gen=200,
        pc=0.8,
        pm=0.1,
        tournament_size=3,
        fitness_mode="F1",
        time_limit=5.0,
    )
    if not ga_success or ga_timeout:
        raise AssertionError("Genetic Algorithm did not succeed for N=8 with deterministic seed.")
    print(f"  Genetic Algorithm: success in {ga_time:.4f}s")

    best_ga_params_for_N = {
        8: {
            "pop_size": 60,
            "max_gen": 200,
            "pm": 0.1,
            "pc": 0.8,
            "tournament_size": 3,
        }
    }

    results = run_experiments_with_best_ga(
        [8],
        runs_sa=3,
        runs_ga=3,
        bt_time_limit=5.0,
        fitness_mode="F1",
        best_ga_params_for_N=best_ga_params_for_N,
        progress_label="Quick regression experiments",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        save_results_to_csv(results, [8], "F1", tmpdir)
        csv_path = Path(tmpdir) / "results_GA_F1_tuned.csv"
        if not csv_path.exists() or csv_path.stat().st_size == 0:
            raise AssertionError("Results CSV was not generated successfully during quick tests.")

    print("Quick regression tests passed.")


# Alias per compatibilità con il main
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
    """
    Wrapper per run_experiments_with_best_ga_parallel per compatibilità
    """
    return run_experiments_with_best_ga_parallel(
        N_values=N_values,
        runs_sa=runs_sa,
        runs_ga=runs_ga,
        bt_time_limit=bt_time_limit,
        fitness_mode=fitness_mode,
        best_ga_params_for_N=best_ga_params_for_N,
        progress_label=progress_label,
    )


def build_arg_parser():
    """Create the CLI argument parser for the orchestrator."""

    parser = argparse.ArgumentParser(
        description="Run N-Queens tuning and experiment pipelines."
    )
    parser.add_argument(
        "--mode",
        choices=["sequential", "parallel", "concurrent"],
        default="concurrent",
        help="Execution mode: sequential tuning, parallel tuning, or concurrent tuning (default).",
    )
    parser.add_argument(
        "--fitness",
        "-f",
        action="append",
        help="Filter fitness modes (accepts comma-separated values or multiple flags).",
    )
    parser.add_argument(
        "--skip-tuning",
        action="store_true",
        help="Reuse stored GA parameters from config.json instead of running tuning.",
    )
    parser.add_argument(
        "--config",
        default="config.json",
        help="Path to configuration file (default: config.json).",
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run quick regression tests (N=8) and exit.",
    )
    return parser


def main():
    """Entry point for CLI execution."""

    parser = build_arg_parser()
    args = parser.parse_args()
    fitness_filter = parse_fitness_filters(args.fitness)

    if args.quick_test:
        run_quick_regression_tests()
        return

    try:
        config_mgr, selected_fitness = apply_configuration(args.config, fitness_filter)
    except FileNotFoundError as exc:
        print(f"Configuration file not found: {exc}")
        raise SystemExit(1) from exc
    except ValueError as exc:
        print(f"Configuration error: {exc}")
        raise SystemExit(1) from exc

    print(f"Selected fitness modes: {selected_fitness}")

    try:
        if args.mode == "sequential":
            main_sequential(selected_fitness, skip_tuning=args.skip_tuning, config_mgr=config_mgr)
        elif args.mode == "parallel":
            main_parallel(selected_fitness, skip_tuning=args.skip_tuning, config_mgr=config_mgr)
        else:
            main_concurrent_tuning(selected_fitness, skip_tuning=args.skip_tuning, config_mgr=config_mgr)
    except KeyboardInterrupt:
        print("\nExecution interrupted by user. Cleaning up workers...")
        raise SystemExit(130) from None
    except ValueError as exc:
        print(f"Execution error: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
