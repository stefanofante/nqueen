"""Backtracking solver for the N-Queens problem."""

from __future__ import annotations

from time import perf_counter
from typing import List, Optional, Tuple


def bt_nqueens_first(size: int, time_limit: Optional[float] = None) -> Tuple[Optional[List[int]], int, float]:
    """Return the first solution found via iterative backtracking."""
    positions = [-1] * size
    row_used = [False] * size
    diag1_used = [False] * (2 * size - 1)
    diag2_used = [False] * (2 * size - 1)

    column = 0
    row = 0
    explored = 0
    start = perf_counter()

    while 0 <= column < size:
        if time_limit is not None and (perf_counter() - start) > time_limit:
            return None, explored, perf_counter() - start

        placed = False
        while row < size and not placed:
            explored += 1
            if not row_used[row]:
                diag1_index = row - column + (size - 1)
                diag2_index = row + column
                if not diag1_used[diag1_index] and not diag2_used[diag2_index]:
                    positions[column] = row
                    row_used[row] = True
                    diag1_used[diag1_index] = True
                    diag2_used[diag2_index] = True
                    placed = True
                    if column == size - 1:
                        return positions.copy(), explored, perf_counter() - start
                    column += 1
                    row = 0
                else:
                    row += 1
            else:
                row += 1

        if not placed:
            column -= 1
            if column >= 0:
                previous_row = positions[column]
                positions[column] = -1
                row_used[previous_row] = False
                diag1_index = previous_row - column + (size - 1)
                diag2_index = previous_row + column
                diag1_used[diag1_index] = False
                diag2_used[diag2_index] = False
                row = previous_row + 1

    return None, explored, perf_counter() - start
