"""Global settings and timeouts for the N-Queens analysis pipeline.

This module centralizes tunable constants used across the orchestration code.
Values can be overridden at runtime via the configuration loader in
`nqueens.analysis.cli.apply_configuration`.
"""
from __future__ import annotations

import multiprocessing
from typing import List, Optional
from datetime import datetime

# Board sizes to evaluate (in ascending order) for scalability analysis
N_VALUES: List[int] = [8, 16, 24, 40, 80, 120]

# Number of independent runs in final experiments (higher = more robust stats)
RUNS_SA_FINAL: int = 40
RUNS_GA_FINAL: int = 40
RUNS_BT_FINAL: int = 1  # Backtracking is deterministic; one run per N is sufficient

# Number of runs per GA tuning combination (kept lower to accelerate grid search)
RUNS_GA_TUNING: int = 5

# Backtracking time limit in seconds (None = no limit)
BT_TIME_LIMIT: Optional[float] = 60 * 2.0  # e.g., 2 minutes

# SA and GA time limits in seconds (None = no limit)
SA_TIME_LIMIT: Optional[float] = 120.0  # Simulated Annealing
GA_TIME_LIMIT: Optional[float] = 120.0  # Genetic Algorithm

# Global timeout per experiment bundle (None = no limit)
EXPERIMENT_TIMEOUT: Optional[float] = 120.0  # ~2 minutes per full experiment

# Output directory for CSV and charts
OUT_DIR: str = "results_nqueens_tuning"

# GA tuning grid (defines the parameter search space)
POP_MULTIPLIERS: List[int] = [4, 8, 16]       # pop_size ≈ {k}*N
GEN_MULTIPLIERS: List[int] = [30, 50, 80]     # max_gen ≈ {m}*N
PM_VALUES: List[float] = [0.05, 0.1, 0.2]    # mutation rate candidates
PC_FIXED: float = 0.8                          # fixed crossover probability
TOURNAMENT_SIZE_FIXED: int = 3                 # tournament size for selection

# Fitness functions to evaluate (F1–F6)
FITNESS_MODES: List[str] = ["F1", "F2", "F3", "F4", "F5", "F6"]

# Number of worker processes to use (leave one core for the OS)
NUM_PROCESSES: int = max(1, multiprocessing.cpu_count() - 1)

# Output naming policy --------------------------------------------------------

# When True, results and plots will include a datestamp suffix (e.g., _20251113-142530)
# applied consistently across all artifacts produced within the same run.
DATE_IN_FILENAMES: bool = True

# Unique run identifier used for filename stamping; set once at import time.
RUN_ID: str = datetime.now().strftime("%Y%m%d-%H%M%S")

# When True, append which algorithms/solvers contributed to the results
# (e.g., _BT-mcv_hybrid+mcv+first_SA_GA). Default disabled to keep filenames shorter.
ALG_IN_FILENAMES: bool = False

# Optional run labeling to avoid overwriting outputs
# Filenames will include algorithm selection and/or a run tag by default
RUN_TAG: Optional[str] = None

# Current pipeline mode for plotting/gating decisions: 'sequential' | 'parallel' | 'concurrent'
CURRENT_PIPELINE_MODE: str = 'parallel'


def set_timeouts(
        bt_timeout: Optional[float] = None,
        sa_timeout: Optional[float] = 30.0,
        ga_timeout: Optional[float] = 60.0,
        experiment_timeout: Optional[float] = 120.0,
) -> None:
        """Configure timeouts for all algorithms and the experiment wrapper.

        Parameters
        - bt_timeout: Backtracking limit in seconds (None disables the limit).
        - sa_timeout: Simulated Annealing limit in seconds (None disables).
        - ga_timeout: Genetic Algorithm limit in seconds (None disables).
        - experiment_timeout: Hard cap for a whole experiment bundle in seconds
            (None disables). When reached, outer loops should cease scheduling new
            work for the current N.

        Side effects
        - Updates module-level globals and prints a concise summary to stdout to
            make the active limits explicit at run start.
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
        print(
                f"   - Experiment: {EXPERIMENT_TIMEOUT}s"
                if EXPERIMENT_TIMEOUT
                else "   - Experiment: unlimited"
        )
