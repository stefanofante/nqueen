"""
Analysis and orchestration package for N-Queens experiments.

This package contains:
- settings: global knobs and timeouts
- stats: typed summaries and aggregation helpers
- tuning: GA hyperparameter search (sequential and parallel)
- experiments: runners for BT/SA/GA with result shaping
- reporting: CSV exports and raw-data writers
- plots: all visualization utilities
- cli: top-level pipeline entry points and argument parser
"""

from . import settings as settings  # re-export for convenience
from .stats import (
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

__all__ = [
    # types
    "StatsSummary",
    "BTEntry",
    "SARecord",
    "GARecord",
    "SAResultEntry",
    "GAResultEntry",
    "ExperimentResults",
    # utils
    "compute_detailed_statistics",
    "compute_grouped_statistics",
    "ProgressPrinter",
    # settings module
    "settings",
]
