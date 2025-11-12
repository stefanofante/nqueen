"""
N-Queens Orchestrator and Analysis Utilities (modularized)
=========================================================

Questo file è ora un sottile "facciata" che re-esporta le API principali
dall'implementazione modulare nel pacchetto `nqueens.analysis`.

- Le funzioni di orchestrazione e CLI vivono in `nqueens.analysis.cli`
- I runner degli esperimenti in `nqueens.analysis.experiments`
- Le utility di tuning in `nqueens.analysis.tuning`
- Le statistiche tipate in `nqueens.analysis.stats`
- Le esportazioni CSV in `nqueens.analysis.reporting`
- I grafici in `nqueens.analysis.plots`
- Le impostazioni globali in `nqueens.analysis.settings`

L'obiettivo è separare responsabilità, facilitare la manutenzione e rendere il
codice testabile e riusabile. Questo file mantiene i nomi pubblici originari
per compatibilità all'indietro (es. i test importano da `algoanalisys`).
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
