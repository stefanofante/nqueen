"""Simulated Annealing solver for the N-Queens problem."""

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
    """Run simulated annealing and return the execution summary."""
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

        temperature *= alpha

    return False, max_iter, perf_counter() - start, best_cost, evaluations, False
