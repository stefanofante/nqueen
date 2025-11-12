"""
N-Queens Orchestrator and Analysis Utilities (modular layout)
============================================================

This module is a thin façade that re-exports the primary public APIs from the
modular implementation under `nqueens.analysis`. It also acts as the executable
entry point of the project when invoked as a script.

- Command-line interface and orchestration live in `nqueens.analysis.cli`.
- Experiment runners are in `nqueens.analysis.experiments`.
- GA parameter tuning lives in `nqueens.analysis.tuning`.
- Typed statistics and helpers are in `nqueens.analysis.stats`.
- CSV export utilities are in `nqueens.analysis.reporting`.
- Plotting/visualization utilities are in `nqueens.analysis.plots`.
- Global configuration resides in `nqueens.analysis.settings`.

Rationale: separate concerns for maintainability, enable focused testing of
subsystems, and keep a stable import surface for dependents. Backward
compatibility is preserved (tests and external users may `import algoanalisys`).
"""

# Re-export types and helpers
from nqueens.analysis.stats import (
    StatsSummary,
    BTEntry,
    SARecord,
    GARecord,
    SAResultEntry,
    GAResultEntry,
    ExperimentResults,
    compute_detailed_statistics,
    compute_grouped_statistics,
    ProgressPrinter,
)

# Re-export settings and timeouts
from nqueens.analysis.settings import (
    N_VALUES,
    RUNS_SA_FINAL,
    RUNS_GA_FINAL,
    RUNS_BT_FINAL,
    RUNS_GA_TUNING,
    BT_TIME_LIMIT,
    SA_TIME_LIMIT,
    GA_TIME_LIMIT,
    EXPERIMENT_TIMEOUT,
    OUT_DIR,
    POP_MULTIPLIERS,
    GEN_MULTIPLIERS,
    PM_VALUES,
    PC_FIXED,
    TOURNAMENT_SIZE_FIXED,
    FITNESS_MODES,
    NUM_PROCESSES,
    set_timeouts,
)

# Re-export tuning utilities
from nqueens.analysis.tuning import (
    tune_ga_for_N,
    tune_ga_for_N_parallel,
    tune_all_fitness_parallel,
)

# Re-export experiment runners
from nqueens.analysis.experiments import (
    run_experiments_with_best_ga,
    run_experiments_with_best_ga_parallel,
    run_experiments_parallel,  # backward-compat alias
)

# Re-export CSV/report utilities
from nqueens.analysis.reporting import (
    save_results_to_csv,
    save_raw_data_to_csv,
    save_logical_cost_analysis,
)

# Re-export plotting utilities
# Re-export plotting utilities (optional dependency)
try:
    from nqueens.analysis.plots import (
        plot_comprehensive_analysis,
        plot_fitness_comparison,
        plot_statistical_analysis,
        plot_and_save,
    )
    _PLOTS_AVAILABLE = True
except Exception:  # matplotlib may be missing in minimal environments
    _PLOTS_AVAILABLE = False
    # Define lightweight stubs that raise a clear error if called
    def _missing_plots(*args, **kwargs):  # type: ignore
        raise RuntimeError("Plotting utilities require matplotlib; install dependencies to use plotting features.")
    plot_comprehensive_analysis = _missing_plots  # type: ignore
    plot_fitness_comparison = _missing_plots  # type: ignore
    plot_statistical_analysis = _missing_plots  # type: ignore
    plot_and_save = _missing_plots  # type: ignore

# Re-export CLI and high-level pipelines
from nqueens.analysis.cli import (
    parse_fitness_filters,
    normalize_optimal_parameters,
    ensure_parameters_for_all_n,
    apply_configuration,
    load_optimal_parameters,
    main_sequential,
    main_parallel,
    main_concurrent_tuning,
    run_quick_regression_tests,
    build_arg_parser,
    main,
)

__all__ = [
    # types
    "StatsSummary",
    "BTEntry",
    "SARecord",
    "GARecord",
    "SAResultEntry",
    "GAResultEntry",
    "ExperimentResults",
    # stats helpers
    "compute_detailed_statistics",
    "compute_grouped_statistics",
    "ProgressPrinter",
    # settings
    "N_VALUES",
    "RUNS_SA_FINAL",
    "RUNS_GA_FINAL",
    "RUNS_BT_FINAL",
    "RUNS_GA_TUNING",
    "BT_TIME_LIMIT",
    "SA_TIME_LIMIT",
    "GA_TIME_LIMIT",
    "EXPERIMENT_TIMEOUT",
    "OUT_DIR",
    "POP_MULTIPLIERS",
    "GEN_MULTIPLIERS",
    "PM_VALUES",
    "PC_FIXED",
    "TOURNAMENT_SIZE_FIXED",
    "FITNESS_MODES",
    "NUM_PROCESSES",
    "set_timeouts",
    # tuning
    "tune_ga_for_N",
    "tune_ga_for_N_parallel",
    "tune_all_fitness_parallel",
    # experiments
    "run_experiments_with_best_ga",
    "run_experiments_with_best_ga_parallel",
    "run_experiments_parallel",
    # reporting
    "save_results_to_csv",
    "save_raw_data_to_csv",
    "save_logical_cost_analysis",
    # plotting (included even if stubs when matplotlib missing)
    "plot_comprehensive_analysis",
    "plot_fitness_comparison",
    "plot_statistical_analysis",
    "plot_and_save",
    # cli
    "parse_fitness_filters",
    "normalize_optimal_parameters",
    "ensure_parameters_for_all_n",
    "apply_configuration",
    "load_optimal_parameters",
    "main_sequential",
    "main_parallel",
    "main_concurrent_tuning",
    "run_quick_regression_tests",
    "build_arg_parser",
    "main",
]

def _run_as_script() -> None:
    """Entrypoint quando eseguito direttamente come script.

    Delega al `main()` definito in `nqueens.analysis.cli` che gestisce
    l'argument parsing e l'orchestrazione delle pipeline.
    """
    # `main` is imported from the CLI section above
    main()


if __name__ == "__main__":
    _run_as_script()
