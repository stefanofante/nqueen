"""Typed result shapes and statistics helpers for the analysis pipeline.

Defines ``TypedDict`` structures for experiment outputs and provides utilities
to compute robust aggregate statistics across heterogeneous result records.
"""
from __future__ import annotations

import statistics
from typing import Any, Dict, List, Optional, Tuple, TypedDict, cast


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


class ProgressPrinter:
    """Minimal, stdout-only progress reporter for long-running loops.

    Parameters
    ----------
    total : int
        Total number of steps/items expected. Values <= 0 are coerced to 1 to
        avoid division by zero when reporting percentages.
    label : str
        Short label printed in front of the progress counters to provide
        context (e.g., the current phase or algorithm name).
    """

    def __init__(self, total: int, label: str):
        self.total = max(1, total)
        self.label = label

    def update(self, index: int, detail: str = "") -> None:
        """Print a single-line progress update to stdout.

        Parameters
        ----------
        index : int
            The current 1-based or 0-based index of progress. Values greater
            than ``total`` are allowed and will print >100%.
        detail : str, optional
            Free-form suffix to provide additional context (e.g., current N,
            fitness, or solver). If empty, no suffix is appended.

        Notes
        -----
        - This method is intentionally side-effect only (prints to stdout) and
          does not persist state.
        - The percentage is computed as ``index / total * 100`` using the
          ``total`` value passed to the constructor (coerced to at least 1).
        """
        percent = (index / self.total) * 100
        suffix = f" - {detail}" if detail else ""
        print(f"[{self.label}] {index}/{self.total} ({percent:.0f}%)" + suffix)


def compute_detailed_statistics(values: List[float], label: str = "") -> StatsSummary:
    """Compute summary statistics for a numeric sequence.

    Parameters
    ----------
    values : List[float]
        A list of numeric values to summarize. Non-finite values should be
        filtered by the caller; this function assumes finite floats.
    label : str, optional
        Optional label carried through to help downstream debugging or logging
        contexts. The value is not used in calculations.

    Returns
    -------
    StatsSummary
        A dictionary-like structure containing count, mean, median, std, min,
        max, 25th and 75th percentiles (q25, q75), and range. When ``values``
        is empty, all numeric fields are ``None`` and ``count`` is 0 to keep
        CSV/plot generation consistent.

    Raises
    ------
    None
        This function does not raise; it returns a ``None``-filled structure on
        empty input and uses population standard deviation for stability.
    """
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
            "q25": None,
            "q75": None,
            "range": None,
        }

    sorted_vals = sorted(values)
    n = len(values)

    mean_val = statistics.mean(values)
    median_val = statistics.median(values)
    min_val = min(values)
    max_val = max(values)
    range_val = max_val - min_val
    std_val = statistics.pstdev(values) if n > 1 else 0

    q25 = sorted_vals[n // 4] if n >= 4 else min_val
    q75 = sorted_vals[3 * n // 4] if n >= 4 else max_val

    return {
        "count": n,
        "mean": mean_val,
        "median": median_val,
        "std": std_val,
        "min": min_val,
        "max": max_val,
        "q25": q25,
        "q75": q75,
        "range": range_val,
    }


def compute_grouped_statistics(
    results_list: List[Dict[str, Any]], success_key: str = "success"
) -> Dict[str, Any]:
    """Aggregate metrics by outcome groups (success, failure, timeout).

    Parameters
    ----------
    results_list : List[Dict[str, Any]]
        A list of per-run result dictionaries produced by the algorithm
        runners. Dictionaries may be heterogeneous but commonly contain keys
        like ``time``, ``steps`` (SA), ``nodes`` (BT), ``gen`` (GA),
        ``evals``, ``best_conflicts``, and boolean flags such as
        ``success`` and ``timeout``.
    success_key : str, optional
        The key to interpret as the success flag (default: ``"success"``).

    Returns
    -------
    Dict[str, Any]
        A dictionary containing rates (``success_rate``, ``timeout_rate``,
        ``failure_rate``), counters (``total_runs``, ``successes``,
        ``failures``, ``timeouts``), and detailed statistics for each metric
        across all runs (``all_<metric>``) and per outcome group
        (``success_<metric>``, ``timeout_<metric>``, ``failure_<metric>``),
        where ``<metric>`` spans the keys present among
        ``["time", "steps", "nodes", "gen", "evals", "best_conflicts"]``.

    Raises
    ------
    None
        This function does not raise; missing metrics are simply omitted from
        the corresponding summary sections.
    """
    successes = [r for r in results_list if r.get(success_key, False)]
    timeouts = [r for r in results_list if r.get("timeout", False)]
    failures = [r for r in results_list if not r.get(success_key, False) and not r.get("timeout", False)]

    stats: Dict[str, Any] = {
        "total_runs": len(results_list),
        "successes": len(successes),
        "failures": len(failures),
        "timeouts": len(timeouts),
        "success_rate": len(successes) / len(results_list) if results_list else 0,
        "timeout_rate": len(timeouts) / len(results_list) if results_list else 0,
        "failure_rate": len(failures) / len(results_list) if results_list else 0,
    }

    for metric in ["time", "steps", "nodes", "gen", "evals", "best_conflicts"]:
        if any(metric in r for r in results_list):
            values = [r[metric] for r in results_list if metric in r]
            stats[f"all_{metric}"] = compute_detailed_statistics(values, f"all_{metric}")

    for metric in ["time", "steps", "nodes", "gen", "evals", "best_conflicts"]:
        if any(metric in r for r in successes):
            values = [r[metric] for r in successes if metric in r]
            stats[f"success_{metric}"] = compute_detailed_statistics(values, f"success_{metric}")

    for metric in ["time", "steps", "nodes", "gen", "evals", "best_conflicts"]:
        if any(metric in r for r in timeouts):
            values = [r[metric] for r in timeouts if metric in r]
            stats[f"timeout_{metric}"] = compute_detailed_statistics(values, f"timeout_{metric}")

    for metric in ["time", "steps", "nodes", "gen", "evals", "best_conflicts"]:
        if any(metric in r for r in failures):
            values = [r[metric] for r in failures if metric in r]
            stats[f"failure_{metric}"] = compute_detailed_statistics(values, f"failure_{metric}")

    return stats
