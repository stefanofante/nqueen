"""Simulated Annealing solver for the N-Queens problem.

This module implements a standard Simulated Annealing (SA) approach to search
for a conflict-free placement of N queens. The configuration is a length-N list
``board[col] = row``. At each step, a random column is selected and its row is
proposed to mutate. The move is accepted if it improves the objective or with
Metropolis probability otherwise.

Contract (public API)
---------------------
- Input: problem size ``size >= 1`` and SA hyperparameters ``max_iter``,
  initial temperature ``T0``, and cooling factor ``alpha``.
- Output: a 6-tuple ``SAResult`` summarizing the run:
    (success, iterations, elapsed_seconds, best_conflicts, evaluations, timeout)

Where:
- success: True when a conflict-free board was found, False otherwise.
- iterations: number of iterations executed when the run ended.
- elapsed_seconds: wall time measured via ``perf_counter()``.
- best_conflicts: best (lowest) conflict count observed.
- evaluations: number of conflict evaluations performed.
- timeout: True when ended due to ``time_limit``.

Determinism
-----------
SA is stochastic. For reproducibility, set the Python ``random`` seed before
invocation.
"""

from __future__ import annotations

import math
import random
from time import perf_counter
from typing import Optional, Sequence, Tuple

from .utils import conflicts


SAResult = Tuple[bool, int, float, int, int, bool]


def sa_nqueens(
    size: int,
    max_iter: int = 20000,
    T0: float = 1.0,
    alpha: float = 0.995,
    time_limit: Optional[float] = None,
) -> SAResult:
    """Run Simulated Annealing to minimize conflicts in the N-Queens problem.

    Parameters
    ----------
    size : int
        Board dimension N.
    max_iter : int, default 20000
        Maximum number of iterations.
    T0 : float, default 1.0
        Initial temperature.
    alpha : float, default 0.995
        Geometric cooling factor; temperature is updated as ``T *= alpha``.
    time_limit : float | None
        Optional wall-clock time limit in seconds.

    Returns
    -------
    SAResult
        Tuple (success, iterations, elapsed, best_conflicts, evaluations, timeout).

    Notes
    -----
    - Objective: minimize the number of pairwise conflicts; zero indicates a valid solution.
    - Move generation: pick a column uniformly and propose a new random row.
    - Acceptance: accept improvements (``delta <= 0``) or with probability
      ``exp(-delta / T)`` otherwise.
    - Cooling: temperature decays multiplicatively by ``alpha`` after each iteration.

    Raises
    ------
    None
    """
    board = [random.randrange(size) for _ in range(size)]
    current_cost = conflicts(board)
    best_cost = current_cost
    evaluations = 1
    start = perf_counter()

    if current_cost == 0:
        return True, 0, perf_counter() - start, 0, evaluations, False

    temperature = T0
    for iteration in range(1, max_iter + 1):
        if time_limit is not None and (perf_counter() - start) > time_limit:
            return False, iteration, perf_counter() - start, best_cost, evaluations, True

        # Propose a random single-column change
        column = random.randrange(size)
        old_row = board[column]
        new_row = random.randrange(size)
        while new_row == old_row:
            new_row = random.randrange(size)
        board[column] = new_row

        candidate_cost = conflicts(board)
        evaluations += 1
        delta = candidate_cost - current_cost

        if delta <= 0 or random.random() < math.exp(-delta / temperature):
            current_cost = candidate_cost
            best_cost = min(best_cost, current_cost)
        else:
            board[column] = old_row

        if current_cost == 0:
            return True, iteration, perf_counter() - start, 0, evaluations, False

        # Geometric cooling schedule
        temperature *= alpha

    return False, max_iter, perf_counter() - start, best_cost, evaluations, False
