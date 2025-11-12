"""Command-line interface and high-level pipelines for N-Queens experiments.

This module wires together configuration loading, optional GA tuning, and
execution of final experiment suites (sequential, parallel, or concurrent
across multiple fitness functions). It intentionally isolates I/O, argument
parsing, and progress reporting from the core algorithmic modules so that the
rest of the codebase remains easy to test programmatically.
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple

from . import settings
from .experiments import (
    run_experiments_with_best_ga,
    run_experiments_with_best_ga_parallel,
)
try:
    from .plots import (
        plot_comprehensive_analysis,
        plot_fitness_comparison,
        plot_statistical_analysis,
        plot_and_save,
    )
except Exception:
    # Provide stubs when matplotlib or plotting stack isn't available
    def _missing_plots(*args, **kwargs):  # type: ignore
        raise RuntimeError(
            "Plotting utilities require matplotlib; install dependencies (matplotlib, seaborn) to enable plotting."
        )

    plot_comprehensive_analysis = _missing_plots  # type: ignore
    plot_fitness_comparison = _missing_plots  # type: ignore
    plot_statistical_analysis = _missing_plots  # type: ignore
    plot_and_save = _missing_plots  # type: ignore
from .reporting import (
    save_logical_cost_analysis,
    save_raw_data_to_csv,
    save_results_to_csv,
)
from .stats import ProgressPrinter
from .tuning import (
    tune_all_fitness_parallel,
    tune_ga_for_N,
    tune_ga_for_N_parallel,
)
from config_manager import ConfigManager
from nqueens.backtracking import bt_nqueens_first
from nqueens.genetic import ga_nqueens
from nqueens.simulated_annealing import sa_nqueens
from nqueens.utils import is_valid_solution


# ------------- Utils --------------------------------------------------------

def parse_fitness_filters(fitness_args: Optional[List[str]]):
    """Normalize fitness filter CLI inputs into a flat list of labels.

    Accepts repeated flags (e.g., ``-f F1 -f F2``) and comma-separated lists
    (e.g., ``-f F1,F3``). Returns ``None`` when no filter is provided so that
    callers can fall back to the configured default set.
    """
    if not fitness_args:
        return None
    selected: List[str] = []
    for entry in fitness_args:
        for token in entry.split(","):
            token = token.strip().upper()
            if token:
                selected.append(token)
    return selected or None


def parse_algorithm_filters(alg_args: Optional[List[str]]):
    """Normalize algorithm filter CLI inputs into a set of labels.

    Accepts repeated flags and comma-separated lists. Valid values: BT, SA, GA.
    Returns None when no filter is provided (meaning all are enabled).
    """
    if not alg_args:
        return None
    selected: List[str] = []
    valid = {"BT", "SA", "GA"}
    for entry in alg_args:
        for token in entry.split(","):
            token = token.strip().upper()
            if token:
                if token not in valid:
                    raise ValueError(f"Unknown algorithm '{token}'. Allowed: BT, SA, GA")
                selected.append(token)
    unique = list(dict.fromkeys(selected))  # preserve order, remove dups
    return unique or None


def normalize_optimal_parameters(raw_params: Optional[Dict[Any, Any]]) -> Dict[Any, Any]:
    """Normalize keys of persisted GA parameters to integer N values.

    Config files may store N as strings; this helper converts them to ints
    where possible while preserving any non-integer keys verbatim.
    """
    normalized: Dict[Any, Any] = {}
    if not raw_params:
        return normalized
    for key, value in raw_params.items():
        try:
            normalized[int(key)] = value
        except (TypeError, ValueError):
            normalized[key] = value
    return normalized


def ensure_parameters_for_all_n(params: Dict[int, Dict[str, Any]], n_values: List[int], fitness_mode: str) -> None:
    """Validate that tuned GA parameters exist for every required N.

    Raises a ``ValueError`` with a precise message if some sizes are missing.
    """
    missing = [n for n in n_values if n not in params]
    if missing:
        missing_str = ", ".join(str(n) for n in missing)
        raise ValueError(
            f"Missing GA parameters for fitness {fitness_mode} and N values: {missing_str}. "
            "Run tuning or update config.json."
        )


def apply_configuration(config_path: str, fitness_filter: Optional[List[str]] = None) -> Tuple[ConfigManager, List[str]]:
    """Load configuration and apply optional fitness filtering.

    This function updates the global ``settings`` module in-place to reflect
    values from ``config.json`` (or a user-specified path). It returns the
    ``ConfigManager`` used and the list of selected fitness labels.
    """
    config_mgr = ConfigManager(config_path)

    experiment_settings = config_mgr.get_experiment_settings()
    if experiment_settings:
        settings.N_VALUES = [int(n) for n in experiment_settings.get("N_values", settings.N_VALUES)]
        settings.RUNS_SA_FINAL = int(experiment_settings.get("runs_sa_final", settings.RUNS_SA_FINAL))
        settings.RUNS_GA_FINAL = int(experiment_settings.get("runs_ga_final", settings.RUNS_GA_FINAL))
        settings.RUNS_BT_FINAL = int(experiment_settings.get("runs_bt_final", settings.RUNS_BT_FINAL))
        settings.RUNS_GA_TUNING = int(experiment_settings.get("runs_ga_tuning", settings.RUNS_GA_TUNING))
        settings.OUT_DIR = experiment_settings.get("output_dir", settings.OUT_DIR)

    timeout_settings = config_mgr.get_timeout_settings()
    if timeout_settings:
        settings.set_timeouts(
            bt_timeout=timeout_settings.get("bt_time_limit", settings.BT_TIME_LIMIT),
            sa_timeout=timeout_settings.get("sa_time_limit", settings.SA_TIME_LIMIT),
            ga_timeout=timeout_settings.get("ga_time_limit", settings.GA_TIME_LIMIT),
            experiment_timeout=timeout_settings.get("experiment_timeout", settings.EXPERIMENT_TIMEOUT),
        )

    tuning_grid = config_mgr.get_tuning_grid()
    if tuning_grid:
        settings.POP_MULTIPLIERS = [int(v) for v in tuning_grid.get("pop_multipliers", settings.POP_MULTIPLIERS)]
        settings.GEN_MULTIPLIERS = [int(v) for v in tuning_grid.get("gen_multipliers", settings.GEN_MULTIPLIERS)]
        settings.PM_VALUES = [float(v) for v in tuning_grid.get("pm_values", settings.PM_VALUES)]
        settings.PC_FIXED = float(tuning_grid.get("pc_fixed", settings.PC_FIXED))
        settings.TOURNAMENT_SIZE_FIXED = int(tuning_grid.get("tournament_size_fixed", settings.TOURNAMENT_SIZE_FIXED))

    fitness_modes_cfg = [mode.upper() for mode in config_mgr.get_fitness_modes()] or ["F1"]

    if fitness_filter:
        requested = {mode.upper() for mode in fitness_filter}
        unknown = requested.difference(set(fitness_modes_cfg))
        if unknown:
            raise ValueError("Unknown fitness modes requested: " + ", ".join(sorted(unknown)))
        selected_modes = [mode for mode in fitness_modes_cfg if mode in requested]
    else:
        selected_modes = fitness_modes_cfg

    if not selected_modes:
        raise ValueError("No fitness modes selected after applying filters.")

    settings.FITNESS_MODES = selected_modes
    return config_mgr, selected_modes


def load_optimal_parameters(
    fitness_mode: str,
    config_mgr: Optional[ConfigManager],
    n_values: List[int],
) -> Dict[int, Dict[str, Any]]:
    """Load tuned GA parameters for a fitness label and verify coverage."""
    if config_mgr is None:
        raise ValueError("Config manager is required when tuning is skipped (default).")
    params = normalize_optimal_parameters(config_mgr.get_optimal_parameters(fitness_mode))
    ensure_parameters_for_all_n(params, n_values, fitness_mode)
    return params


# ------------- Pipeline: sequential ----------------------------------------

def main_sequential(
    fitness_modes: Optional[List[str]] = None,
    skip_tuning: bool = False,
    config_mgr: Optional[ConfigManager] = None,
    validate: bool = False,
    algorithms: Optional[List[str]] = None,
) -> None:
    """Run sequential GA tuning (optional) and final experiments per fitness.

    Suitable when parallel resources are limited or when deterministic ordering
    is preferred for debugging and reproducibility of I/O.
    """
    os.makedirs(settings.OUT_DIR, exist_ok=True)
    selected_fitness = fitness_modes or settings.FITNESS_MODES

    include_bt = (algorithms is None) or ("BT" in algorithms)
    include_sa = (algorithms is None) or ("SA" in algorithms)
    include_ga = (algorithms is None) or ("GA" in algorithms)

    for fitness_mode in selected_fitness:
        print("\n============================================")
        print(f"SEQUENTIAL PIPELINE FOR GA FITNESS {fitness_mode}")
        print("============================================")

        if not include_ga:
            best_ga_params_for_N: Dict[int, Dict[str, Any]] = {}
        elif skip_tuning:
            print("Skipping GA tuning and reusing parameters from configuration.")
            try:
                best_ga_params_for_N = load_optimal_parameters(fitness_mode, config_mgr, settings.N_VALUES)
            except ValueError as exc:
                print(f"  No optimal GA parameters available ({exc}). Auto-running tuning now...")
                best_ga_params_for_N = {}
                tuning_csv = os.path.join(settings.OUT_DIR, f"tuning_GA_{fitness_mode}_seq.csv")
                progress = ProgressPrinter(len(settings.N_VALUES), f"Tuning GA-{fitness_mode}")

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

                    for index, N in enumerate(settings.N_VALUES, start=1):
                        progress.update(index, f"N={N}")
                        print(f"Tuning GA: N = {N}, fitness = {fitness_mode}")
                        best = tune_ga_for_N(
                            N,
                            fitness_mode,
                            settings.POP_MULTIPLIERS,
                            settings.GEN_MULTIPLIERS,
                            settings.PM_VALUES,
                            settings.PC_FIXED,
                            settings.TOURNAMENT_SIZE_FIXED,
                            runs_tuning=settings.RUNS_GA_TUNING,
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

                if config_mgr:
                    config_mgr.save_optimal_parameters(fitness_mode, best_ga_params_for_N)
        else:
            print("Starting GA tuning (sequential search).")
            best_ga_params_for_N: Dict[int, Dict[str, Any]] = {}
            tuning_csv = os.path.join(settings.OUT_DIR, f"tuning_GA_{fitness_mode}_seq.csv")
            progress = ProgressPrinter(len(settings.N_VALUES), f"Tuning GA-{fitness_mode}")

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

                for index, N in enumerate(settings.N_VALUES, start=1):
                    progress.update(index, f"N={N}")
                    print(f"Tuning GA: N = {N}, fitness = {fitness_mode}")
                    best = tune_ga_for_N(
                        N,
                        fitness_mode,
                        settings.POP_MULTIPLIERS,
                        settings.GEN_MULTIPLIERS,
                        settings.PM_VALUES,
                        settings.PC_FIXED,
                        settings.TOURNAMENT_SIZE_FIXED,
                        runs_tuning=settings.RUNS_GA_TUNING,
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

            if config_mgr:
                config_mgr.save_optimal_parameters(fitness_mode, best_ga_params_for_N)

        print(f"\nRunning final experiments for GA fitness {fitness_mode}")
        results = run_experiments_with_best_ga(
            settings.N_VALUES,
            runs_sa=settings.RUNS_SA_FINAL,
            runs_ga=settings.RUNS_GA_FINAL,
            bt_time_limit=settings.BT_TIME_LIMIT,
            fitness_mode=fitness_mode,
            best_ga_params_for_N=best_ga_params_for_N,
            progress_label=f"Experiments GA-{fitness_mode}",
            validate=validate,
            include_bt=include_bt,
            include_sa=include_sa,
            include_ga=include_ga,
        )

        save_results_to_csv(results, settings.N_VALUES, fitness_mode, settings.OUT_DIR)
        save_raw_data_to_csv(results, settings.N_VALUES, fitness_mode, settings.OUT_DIR)
        save_logical_cost_analysis(results, settings.N_VALUES, fitness_mode, settings.OUT_DIR)
        plot_and_save(results, settings.N_VALUES, fitness_mode, settings.OUT_DIR)

    print("\nSequential pipeline completed.")


# ------------- Pipeline: parallel -----------------------------------------

def main_parallel(
    fitness_modes: Optional[List[str]] = None,
    skip_tuning: bool = False,
    config_mgr: Optional[ConfigManager] = None,
    validate: bool = False,
    algorithms: Optional[List[str]] = None,
) -> None:
    """Run GA tuning and final experiments leveraging process-level parallelism."""
    os.makedirs(settings.OUT_DIR, exist_ok=True)
    selected_fitness = fitness_modes or settings.FITNESS_MODES

    print(f"\nStarting parallel pipeline with {settings.NUM_PROCESSES} worker processes")
    print(f"Available CPU cores: {os.cpu_count()}")
    print("Configured timeouts:")
    print(f"   - BT: {settings.BT_TIME_LIMIT}s" if settings.BT_TIME_LIMIT else "   - BT: unlimited")
    print(f"   - SA: {settings.SA_TIME_LIMIT}s" if settings.SA_TIME_LIMIT else "   - SA: unlimited")
    print(f"   - GA: {settings.GA_TIME_LIMIT}s" if settings.GA_TIME_LIMIT else "   - GA: unlimited")
    print(
        f"   - Experiment: {settings.EXPERIMENT_TIMEOUT}s"
        if settings.EXPERIMENT_TIMEOUT
        else "   - Experiment: unlimited"
    )

    start_total = perf_counter()
    all_best_params: Dict[str, Dict[int, Dict[str, Any]]] = {}

    include_bt = (algorithms is None) or ("BT" in algorithms)
    include_sa = (algorithms is None) or ("SA" in algorithms)
    include_ga = (algorithms is None) or ("GA" in algorithms)

    if skip_tuning or not include_ga:
        print("\nSkipping GA tuning phase and loading parameters from configuration.")
        for fitness_mode in selected_fitness:
            if not include_ga:
                all_best_params[fitness_mode] = {}
                continue
            try:
                all_best_params[fitness_mode] = load_optimal_parameters(
                    fitness_mode, config_mgr, settings.N_VALUES
                )
            except ValueError as exc:
                print(f"  No optimal GA parameters for {fitness_mode} ({exc}). Auto-running parallel tuning...")
                best_ga_params_for_N: Dict[int, Dict[str, Any]] = {}
                tuning_csv = os.path.join(settings.OUT_DIR, f"tuning_GA_{fitness_mode}.csv")
                progress = ProgressPrinter(len(settings.N_VALUES), f"Tuning GA-{fitness_mode}")

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

                    for index, N in enumerate(settings.N_VALUES, start=1):
                        progress.update(index, f"N={N}")
                        tuning_start = perf_counter()

                        best = tune_ga_for_N_parallel(
                            N,
                            fitness_mode,
                            settings.POP_MULTIPLIERS,
                            settings.GEN_MULTIPLIERS,
                            settings.PM_VALUES,
                            settings.PC_FIXED,
                            settings.TOURNAMENT_SIZE_FIXED,
                            runs_tuning=settings.RUNS_GA_TUNING,
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
                if config_mgr:
                    config_mgr.save_optimal_parameters(fitness_mode, best_ga_params_for_N)
    else:
        print("\n" + "=" * 60)
        print("PHASE 1: PARALLEL GA TUNING")
        print("=" * 60)

        for fitness_mode in selected_fitness:
            print(f"\nTuning fitness {fitness_mode}...")
            fitness_start = perf_counter()

            best_ga_params_for_N: Dict[int, Dict[str, Any]] = {}
            tuning_csv = os.path.join(settings.OUT_DIR, f"tuning_GA_{fitness_mode}.csv")
            progress = ProgressPrinter(len(settings.N_VALUES), f"Tuning GA-{fitness_mode}")

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

                for index, N in enumerate(settings.N_VALUES, start=1):
                    progress.update(index, f"N={N}")
                    tuning_start = perf_counter()

                    best = tune_ga_for_N_parallel(
                        N,
                        fitness_mode,
                        settings.POP_MULTIPLIERS,
                        settings.GEN_MULTIPLIERS,
                        settings.PM_VALUES,
                        settings.PC_FIXED,
                        settings.TOURNAMENT_SIZE_FIXED,
                        runs_tuning=settings.RUNS_GA_TUNING,
                    )

                    tuning_time = perf_counter() - tuning_start
                    best_ga_params_for_N[N] = best
                    print(f"     Completed in {tuning_time:.1f}s - Success rate: {best['success_rate']:.3f}")

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
            settings.N_VALUES,
            runs_sa=settings.RUNS_SA_FINAL,
            runs_ga=settings.RUNS_GA_FINAL,
            bt_time_limit=settings.BT_TIME_LIMIT,
            fitness_mode=fitness_mode,
            best_ga_params_for_N=all_best_params[fitness_mode],
            progress_label=f"Experiments GA-{fitness_mode}",
            validate=validate,
            include_bt=include_bt,
            include_sa=include_sa,
            include_ga=include_ga,
        )

        experiments_time = perf_counter() - experiments_start
        print(f"  Experiments completed in {experiments_time:.1f}s")

        print("Generating charts and CSV reports...")
        save_results_to_csv(results, settings.N_VALUES, fitness_mode, settings.OUT_DIR)
        save_raw_data_to_csv(results, settings.N_VALUES, fitness_mode, settings.OUT_DIR)
        save_logical_cost_analysis(results, settings.N_VALUES, fitness_mode, settings.OUT_DIR)
        plot_and_save(results, settings.N_VALUES, fitness_mode, settings.OUT_DIR)
        print(f"  Results saved for {fitness_mode}")

    total_time = perf_counter() - start_total
    print("\nParallel pipeline completed!")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Fitness processed: {len(selected_fitness)}")
    print(f"Worker processes used: {settings.NUM_PROCESSES}")


# ------------- Pipeline: concurrent tuning across fitness ------------------

def save_tuning_results(best_params_for_N: Dict[int, Dict[str, Any]], fitness_mode: str, out_dir: str) -> None:
    filename = os.path.join(out_dir, f"tuning_GA_F{fitness_mode}.csv")
    with open(filename, "w", newline="") as f:
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
                params.get("avg_gen_success", ""),
            ])
            print(f"Saved GA-F{fitness_mode} tuning results: {filename}")


def main_concurrent_tuning(
    fitness_modes: Optional[List[str]] = None,
    skip_tuning: bool = False,
    config_mgr: Optional[ConfigManager] = None,
    validate: bool = False,
    algorithms: Optional[List[str]] = None,
) -> None:
    """Tune GA parameters for all selected fitness functions concurrently.

    After tuning, it executes final experiments and generates comparative plots
    across fitness functions. Intended as the default, comprehensive pipeline.
    """
    os.makedirs(settings.OUT_DIR, exist_ok=True)
    selected_fitness = fitness_modes or settings.FITNESS_MODES

    print("\nCONCURRENT TUNING FOR SELECTED FITNESS FUNCTIONS")
    print(f"Fitness modes: {selected_fitness}")
    print(f"Processes: {settings.NUM_PROCESSES}")
    print(f"Available CPU cores: {os.cpu_count()}")
    print("Configured timeouts:")
    print(f"   - BT: {settings.BT_TIME_LIMIT}s" if settings.BT_TIME_LIMIT else "   - BT: unlimited")
    print(f"   - SA: {settings.SA_TIME_LIMIT}s" if settings.SA_TIME_LIMIT else "   - SA: unlimited")
    print(f"   - GA: {settings.GA_TIME_LIMIT}s" if settings.GA_TIME_LIMIT else "   - GA: unlimited")
    print(
        f"   - Experiment: {settings.EXPERIMENT_TIMEOUT}s"
        if settings.EXPERIMENT_TIMEOUT
        else "   - Experiment: unlimited"
    )

    start_total = perf_counter()
    all_best_params: Dict[str, Dict[int, Dict[str, Any]]] = {fitness_mode: {} for fitness_mode in selected_fitness}

    include_bt = (algorithms is None) or ("BT" in algorithms)
    include_sa = (algorithms is None) or ("SA" in algorithms)
    include_ga = (algorithms is None) or ("GA" in algorithms)

    if skip_tuning or not include_ga:
        print("\nSkipping GA tuning phase and loading parameters from configuration.")
        for fitness_mode in selected_fitness:
            if not include_ga:
                all_best_params[fitness_mode] = {}
                continue
            try:
                all_best_params[fitness_mode] = load_optimal_parameters(
                    fitness_mode, config_mgr, settings.N_VALUES
                )
            except ValueError as exc:
                print(f"  No optimal GA parameters for {fitness_mode} ({exc}). Auto-running parallel tuning...")
                # Fallback: tune this single fitness across all N using parallel tuner
                best_ga_params_for_N: Dict[int, Dict[str, Any]] = {}
                progress = ProgressPrinter(len(settings.N_VALUES), f"Tuning GA-{fitness_mode}")
                for index, N in enumerate(settings.N_VALUES, start=1):
                    progress.update(index, f"N={N}")
                    best = tune_ga_for_N_parallel(
                        N,
                        fitness_mode,
                        settings.POP_MULTIPLIERS,
                        settings.GEN_MULTIPLIERS,
                        settings.PM_VALUES,
                        settings.PC_FIXED,
                        settings.TOURNAMENT_SIZE_FIXED,
                        runs_tuning=settings.RUNS_GA_TUNING,
                    )
                    best_ga_params_for_N[N] = best
                save_tuning_results(best_ga_params_for_N, fitness_mode, settings.OUT_DIR)
                all_best_params[fitness_mode] = best_ga_params_for_N
                if config_mgr:
                    config_mgr.save_optimal_parameters(fitness_mode, best_ga_params_for_N)
    else:
        print("\n" + "=" * 70)
        print("PHASE 1: PARALLEL TUNING FOR ALL FITNESS FUNCTIONS")
        print("=" * 70)

        progress = ProgressPrinter(len(settings.N_VALUES), "Concurrent GA tuning")

        for index, N in enumerate(settings.N_VALUES, start=1):
            progress.update(index, f"N={N}")
            print(f"\nParallel tuning for N = {N}")
            print("-" * 50)

            fitness_results = tune_all_fitness_parallel(
                N,
                selected_fitness,
                settings.POP_MULTIPLIERS,
                settings.GEN_MULTIPLIERS,
                settings.PM_VALUES,
                settings.PC_FIXED,
                settings.TOURNAMENT_SIZE_FIXED,
                runs_tuning=settings.RUNS_GA_TUNING,
            )

            for fitness_mode, best_params in fitness_results.items():
                all_best_params[fitness_mode][N] = best_params

        for fitness_mode in selected_fitness:
            save_tuning_results(all_best_params[fitness_mode], fitness_mode, settings.OUT_DIR)
            if config_mgr:
                config_mgr.save_optimal_parameters(fitness_mode, all_best_params[fitness_mode])

    print("\n" + "=" * 70)
    print("PHASE 2: FINAL EXPERIMENTS WITH OPTIMAL PARAMETERS")
    print("=" * 70)

    all_results: Dict[str, Any] = {}

    for fitness_mode in selected_fitness:
        print(f"\nFinal experiments GA-{fitness_mode}")

        results = run_experiments_with_best_ga_parallel(
            settings.N_VALUES,
            runs_sa=settings.RUNS_SA_FINAL,
            runs_ga=settings.RUNS_GA_FINAL,
            bt_time_limit=settings.BT_TIME_LIMIT,
            fitness_mode=fitness_mode,
            best_ga_params_for_N=all_best_params[fitness_mode],
            progress_label=f"Experiments GA-{fitness_mode}",
            validate=validate,
            include_bt=include_bt,
            include_sa=include_sa,
            include_ga=include_ga,
        )

        all_results[fitness_mode] = results

        save_results_to_csv(results, settings.N_VALUES, fitness_mode, settings.OUT_DIR)
        save_raw_data_to_csv(results, settings.N_VALUES, fitness_mode, settings.OUT_DIR)
        save_logical_cost_analysis(results, settings.N_VALUES, fitness_mode, settings.OUT_DIR)
        plot_and_save(results, settings.N_VALUES, fitness_mode, settings.OUT_DIR)

    print("\n" + "=" * 70)
    print("PHASE 3: COMPARATIVE ANALYSIS AND ADVANCED CHARTS")
    print("=" * 70)

    for fitness in selected_fitness:
        print(f"  Comprehensive analysis for GA-F{fitness}...")
        plot_comprehensive_analysis(
            all_results[fitness], settings.N_VALUES, fitness, os.path.join(settings.OUT_DIR, f"analysis_F{fitness}"), raw_runs=None
        )

    print("  Comparing fitness functions...")
    plot_fitness_comparison(all_results, settings.N_VALUES, os.path.join(settings.OUT_DIR, "fitness_comparison"))

    print("  Statistical analysis...")
    plot_statistical_analysis(
        all_results, settings.N_VALUES, os.path.join(settings.OUT_DIR, "statistical_analysis"), raw_runs=None
    )

    total_time = perf_counter() - start_total
    print("\nConcurrent pipeline completed!")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Fitness processed: {len(selected_fitness)}")


# ------------- Quick regression -------------------------------------------

def run_quick_regression_tests() -> None:
    """Execute a fast, deterministic smoke test for BT/SA/GA at N=8.

    Verifies that:
    - Each backtracking variant finds a valid solution and returns reasonable
      node counts and times.
    - SA and GA succeed under fixed seeds and common hyperparameters.
    - The experiment pipeline produces a non-empty CSV in a temporary folder.
    """
    print("Running quick regression tests (N=8) across all algorithms...")

    import inspect
    import importlib

    import nqueens.backtracking as bt_mod
    bt_solvers = [(name, fn) for name, fn in inspect.getmembers(bt_mod, inspect.isfunction) if name.startswith("bt_nqueens_")]
    if not bt_solvers:
        raise AssertionError("No backtracking solvers discovered (expected functions named 'bt_nqueens_*').")

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
        if not is_valid_solution(solution):
            raise AssertionError(f"{name} returned an invalid solution for N=8: {solution}.")
        print(f"  [BT] {name}: solution found, nodes={nodes}, time={elapsed:.4f}s")

    random.seed(42)
    sa_success, _, sa_time, _, _, sa_timeout = sa_nqueens(8, max_iter=5000, T0=1.0, alpha=0.995, time_limit=5.0)
    if not sa_success or sa_timeout:
        raise AssertionError("Simulated Annealing did not succeed for N=8 with deterministic seed.")
    print(f"  Simulated Annealing: success in {sa_time:.4f}s")

    random.seed(42)
    ga_success, _, ga_time, _, _, ga_timeout = ga_nqueens(
        8, pop_size=60, max_gen=200, pc=0.8, pm=0.1, tournament_size=3, fitness_mode="F1", time_limit=5.0
    )
    if not ga_success or ga_timeout:
        raise AssertionError("Genetic Algorithm did not succeed for N=8 with deterministic seed.")
    print(f"  Genetic Algorithm: success in {ga_time:.4f}s")

    best_ga_params_for_N = {8: {"pop_size": 60, "max_gen": 200, "pm": 0.1, "pc": 0.8, "tournament_size": 3}}

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


# ------------- CLI wiring --------------------------------------------------

def build_arg_parser():
    """Construct the argument parser for the CLI entry point."""
    parser = argparse.ArgumentParser(description="Run N-Queens tuning and experiment pipelines.")
    parser.add_argument(
        "--mode",
        choices=["sequential", "parallel", "concurrent"],
        default="parallel",
        help="Execution mode: sequential tuning, parallel tuning (default), or concurrent tuning.",
    )
    parser.add_argument(
        "--fitness",
        "-f",
        action="append",
        help="Filter fitness modes (accepts comma-separated values or multiple flags).",
    )
    parser.add_argument(
        "--alg",
        "-a",
        action="append",
        help="Filter algorithms to execute: BT, SA, GA (comma-separated or multiple flags). Default: all.",
    )
    tune_group = parser.add_mutually_exclusive_group()
    tune_group.add_argument("--tune", action="store_true", help="Run GA tuning before experiments (default is to reuse stored parameters).")
    parser.add_argument("--config", default="config.json", help="Path to configuration file (default: config.json).")
    parser.add_argument("--quick-test", action="store_true", help="Run quick regression tests (N=8) and exit.")
    parser.add_argument("--validate", action="store_true", help="Validate solutions and run consistency checks on results (extra assertions).")
    return parser


def main() -> None:
    """CLI entry point: parse arguments and dispatch to the chosen pipeline."""
    parser = build_arg_parser()
    args = parser.parse_args()
    fitness_filter = parse_fitness_filters(args.fitness)
    alg_filter = parse_algorithm_filters(args.alg)

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

    # Tuning policy: only on explicit request (--tune). Default: reuse parameters.
    skip_tuning_effective = not getattr(args, "tune", False)

    try:
        if args.mode == "sequential":
            main_sequential(selected_fitness, skip_tuning=skip_tuning_effective, config_mgr=config_mgr, validate=args.validate, algorithms=alg_filter)
        elif args.mode == "parallel":
            main_parallel(selected_fitness, skip_tuning=skip_tuning_effective, config_mgr=config_mgr, validate=args.validate, algorithms=alg_filter)
        else:
            main_concurrent_tuning(selected_fitness, skip_tuning=skip_tuning_effective, config_mgr=config_mgr, validate=args.validate, algorithms=alg_filter)
    except KeyboardInterrupt:
        print("\nExecution interrupted by user. Cleaning up workers...")
        raise SystemExit(130) from None
    except ValueError as exc:
        print(f"Execution error: {exc}")
        raise SystemExit(1) from exc
