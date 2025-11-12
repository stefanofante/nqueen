"""Fitness functions for Genetic Algorithm variants on N-Queens.

This module centralizes a collection of fitness formulations (F1–F6) used by
the Genetic Algorithm. All functions accept a board encoding ``board[col] = row``
and return a scalar fitness value (higher is better unless otherwise stated).

Guidance
--------
- F1/F2 are directly derived from pairwise conflicts.
- F3/F5 penalize diagonal clusters with different severity (linear vs quadratic).
- F4 mixes global fitness with a per-queen worst-case penalty.
- F6 applies an exponential transform to the conflict count.

Note: The project commonly tracks progress using the number of conflicts in
addition to any chosen fitness. This helps compare runs across modes.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Callable, Sequence

from .utils import conflicts


def fitness_f1(board: Sequence[int]) -> float:
    """Return the classic negative-conflicts fitness.

    Parameters
    ----------
    board : Sequence[int]
        Configuration encoding ``board[col] = row]``.

    Returns
    -------
    float
        ``-conflicts(board)`` so that higher fitness corresponds to fewer conflicts.
    """
    return -conflicts(board)


def fitness_f2(board: Sequence[int]) -> float:
    """Return the number of non-conflicting pairs of queens.

    The maximum value is ``N*(N-1)/2`` for N queens.
    """
    n = len(board)
    max_pairs = n * (n - 1) // 2
    return max_pairs - conflicts(board)


def fitness_f3(board: Sequence[int]) -> float:
    """Apply a linear penalty to diagonal clusters (mild discouragement).

    Counts the number of pairs on the same diagonals using a linear term
    for clusters of size > 1.
    """
    diag1: Counter[int] = Counter()
    diag2: Counter[int] = Counter()
    for column, row in enumerate(board):
        diag1[row - column] += 1
        diag2[row + column] += 1

    penalty = 0
    for count in diag1.values():
        if count > 1:
            penalty += count * (count - 1) // 2
    for count in diag2.values():
        if count > 1:
            penalty += count * (count - 1) // 2

    n = len(board)
    max_pairs = n * (n - 1) // 2
    return max_pairs - penalty


def fitness_f4(board: Sequence[int]) -> float:
    """Penalize the queen with the largest number of conflicts.

    Starts from the F2 perspective (non-conflicting pairs) and subtracts the
    worst per-queen conflict count to reduce concentration of conflicts.
    """
    n = len(board)
    max_pairs = n * (n - 1) // 2
    total_conflicts = conflicts(board)
    base_fitness = max_pairs - total_conflicts

    worst_queen_conflicts = 0
    for column in range(n):
        queen_conflicts = 0
        for other_column in range(n):
            if other_column == column:
                continue
            if board[other_column] == board[column] or abs(board[other_column] - board[column]) == abs(other_column - column):
                queen_conflicts += 1
        worst_queen_conflicts = max(worst_queen_conflicts, queen_conflicts)

    return base_fitness - worst_queen_conflicts


def fitness_f5(board: Sequence[int]) -> float:
    """Apply a quadratic penalty to diagonal clusters (strong discouragement).

    Clusters on the same diagonal incur a squared penalty to more strongly
    penalize larger clusters.
    """
    diag1: Counter[int] = Counter()
    diag2: Counter[int] = Counter()
    for column, row in enumerate(board):
        diag1[row - column] += 1
        diag2[row + column] += 1

    penalty = 0
    for count in diag1.values():
        if count > 1:
            penalty += count ** 2
    for count in diag2.values():
        if count > 1:
            penalty += count ** 2

    n = len(board)
    max_pairs = n * (n - 1) // 2
    return max_pairs - penalty


def fitness_f6(board: Sequence[int], lam: float = 0.3) -> float:
    """Return the exponential fitness ``exp(-lam * conflicts(board))``.

    Parameters
    ----------
    lam : float
        Controls the steepness of the decay with conflicts.
    """
    return math.exp(-lam * conflicts(board))


def get_fitness_function(mode: str) -> Callable[[Sequence[int]], float]:
    """Return a fitness function by label.

    Parameters
    ----------
    mode : str
        One of "F1".."F6".

    Returns
    -------
    Callable[[Sequence[int]], float]
        The corresponding fitness function.
    """
    mapping = {
        "F1": fitness_f1,
        "F2": fitness_f2,
        "F3": fitness_f3,
        "F4": fitness_f4,
        "F5": fitness_f5,
        "F6": lambda board: fitness_f6(board, lam=0.3),
    }
    try:
        return mapping[mode]
    except KeyError as exc:  # pragma: no cover - defensive programming
        raise ValueError(f"Unknown fitness mode: {mode}") from exc
