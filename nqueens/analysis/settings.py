"""Global settings and timeouts for the N-Queens analysis pipeline."""
from __future__ import annotations

import multiprocessing
from typing import List, Optional

# Board sizes to evaluate (in ascending order) for scalability analysis
N_VALUES: List[int] = [8, 16, 24, 40, 80, 120]

# Number of independent runs in final experiments (higher = more robust stats)
RUNS_SA_FINAL: int = 40
RUNS_GA_FINAL: int = 40
RUNS_BT_FINAL: int = 1  # Backtracking is deterministic; one run per N is sufficient

# Number of runs per GA tuning combination (kept lower to accelerate grid search)
RUNS_GA_TUNING: int = 5

# Backtracking time limit in seconds (None = no limit)
BT_TIME_LIMIT: Optional[float] = 60 * 5.0  # e.g., 5 minutes

# SA and GA time limits in seconds (None = no limit)
SA_TIME_LIMIT: Optional[float] = 120.0  # Simulated Annealing
GA_TIME_LIMIT: Optional[float] = 240.0  # Genetic Algorithm

# Global timeout per experiment bundle (None = no limit)
EXPERIMENT_TIMEOUT: Optional[float] = 120.0  # ~2 minutes per full experiment

# Output directory for CSV and charts
OUT_DIR: str = "results_nqueens_tuning"

# GA tuning grid (defines the parameter search space)
POP_MULTIPLIERS: List[int] = [4, 8, 16]       # pop_size ≈ {k}*N
GEN_MULTIPLIERS: List[int] = [30, 50, 80]     # max_gen ≈ {m}*N
PM_VALUES: List[float] = [0.05, 0.1, 0.15]    # mutation rate candidates
PC_FIXED: float = 0.8                          # fixed crossover probability
TOURNAMENT_SIZE_FIXED: int = 3                 # tournament size for selection

# Fitness functions to evaluate (F1–F6)
FITNESS_MODES: List[str] = ["F1", "F2", "F3", "F4", "F5", "F6"]

# Number of worker processes to use (leave one core for the OS)
NUM_PROCESSES: int = max(1, multiprocessing.cpu_count() - 1)


def set_timeouts(
    bt_timeout: Optional[float] = None,
    sa_timeout: Optional[float] = 30.0,
    ga_timeout: Optional[float] = 60.0,
    experiment_timeout: Optional[float] = 120.0,
) -> None:
    """Configure timeouts for all algorithms and the experiment wrapper."""
    global BT_TIME_LIMIT, SA_TIME_LIMIT, GA_TIME_LIMIT, EXPERIMENT_TIMEOUT
    BT_TIME_LIMIT = bt_timeout
    SA_TIME_LIMIT = sa_timeout
    GA_TIME_LIMIT = ga_timeout
    EXPERIMENT_TIMEOUT = experiment_timeout

    print("Timeout settings configured:")
    print(f"   - BT: {BT_TIME_LIMIT}s" if BT_TIME_LIMIT else "   - BT: unlimited")
    print(f"   - SA: {SA_TIME_LIMIT}s" if SA_TIME_LIMIT else "   - SA: unlimited")
    print(f"   - GA: {GA_TIME_LIMIT}s" if GA_TIME_LIMIT else "   - GA: unlimited")
    print(
        f"   - Experiment: {EXPERIMENT_TIMEOUT}s"
        if EXPERIMENT_TIMEOUT
        else "   - Experiment: unlimited"
    )
