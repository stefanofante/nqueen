"""Backtracking solvers for the N-Queens problem.

This module implements iterative (non-recursive) backtracking search for the
N-Queens problem and provides three entry points:

- bt_nqueens_first(size, time_limit=None): a plain left-to-right iterative
    backtracking search that returns the first solution found.
- bt_nqueens_mcv(size, time_limit=None): backtracking with the Most Constrained
    Variable heuristic (choose the column with fewest legal rows).
- bt_nqueens_lcv(size, time_limit=None): backtracking with the Least Constraining
    Value heuristic (for the chosen column, prefer rows that leave the most
    legal choices for remaining columns).

All functions are non-recursive and return a tuple:
        (solution: Optional[List[int]], nodes_explored: int, elapsed_seconds: float)

`solution` is a list of length `size` where `solution[col] = row` places a
queen at (row, col). If no solution is found within the optional
`time_limit` (seconds) the functions return (None, explored, elapsed).

Implementation overview
-----------------------
- State representation: a solution is a size-length list `positions` where
    `positions[c] = r` means a queen on row r, column c; `-1` means unassigned.
- Constraint tracking: three boolean arrays ensure O(1) checks for row and
    diagonal availability: `row_used[r]`, `diag1_used[r-c+offset]`,
    `diag2_used[r+c]`, where `offset = size - 1` maps negative indices to [0..].
- Search strategy: depth-first search implemented iteratively with an explicit
    stack of decision frames, avoiding Python recursion overhead.
- Heuristic injection: a shared iterative engine delegates variable/value order
    to a `select_column(...) -> (column, candidates)` callback.

Contract (public API)
---------------------
- Input: `size >= 1` and optional `time_limit` (seconds, float|None).
- Output: `(solution, nodes_explored, elapsed_seconds)` where
    - `solution` is a list[int] of length `size` if found, else `None`.
    - `nodes_explored` counts candidate value attempts (see notes below).
    - `elapsed_seconds` is wall-clock time measured via `perf_counter()`.
- Determinism: results are deterministic for equal inputs; tie-breakers are
    made explicit (see individual docstrings).
- Timeouts: if `time_limit` is not None and exceeded, the solver returns
    `(None, explored, elapsed)` immediately.
- Nodes explored semantics: incremented every time the search attempts to
    evaluate/advance a candidate assignment for a column (even if quickly
    rejected by current constraints). This provides a consistent proxy of logical
    search effort across heuristics.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable, List, Optional, Tuple


@dataclass
class _Frame:
    """Mutable stack frame capturing the state at a decision level."""

    column: int
    candidates: List[int]
    next_index: int = 0


def bt_nqueens_first(size: int, time_limit: Optional[float] = None) -> Tuple[Optional[List[int]], int, float]:
    """Find the first solution via plain iterative backtracking.

    Parameters
    ----------
    size : int
        Board dimension N (N >= 1).
    time_limit : float | None
        Optional wall-clock time limit in seconds.

    Returns
    -------
    (solution, nodes_explored, elapsed_seconds)
        - solution: list[int] of length N, or None on timeout/failure.
        - nodes_explored: int, number of candidate placements considered.
        - elapsed_seconds: float, total wall time.

    Determinism and ordering
    ------------------------
    - Columns are assigned in natural order 0..N-1.
    - Within a column, rows are tried top-to-bottom 0..N-1.
    - As a result, this returns the lexicographically first solution.

    Complexity
    ----------
    Exponential in the worst case; small instances finish quickly due to early
    pruning using row/diagonal constraints.
    """

    # Track queen positions by column; -1 means the column is still unassigned.
    positions = [-1] * size
    row_used = [False] * size
    diag1_used = [False] * (2 * size - 1)
    diag2_used = [False] * (2 * size - 1)

    column = 0
    row = 0
    explored = 0
    start = perf_counter()

    # Advance column by column, backtracking when no safe row remains.
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
                    # Place the queen and mark the corresponding constraints.
                    positions[column] = row
                    row_used[row] = True
                    diag1_used[diag1_index] = True
                    diag2_used[diag2_index] = True
                    placed = True
                    if column == size - 1:
                        return positions.copy(), explored, perf_counter() - start
                    # Move forward to the next column and reset row scanning.
                    column += 1
                    row = 0
                else:
                    row += 1
            else:
                row += 1

        if not placed:
            # Exhausted all rows in this column; undo the previous decision.
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


def _available_rows(
    size: int,
    column: int,
    row_used: List[bool],
    diag1_used: List[bool],
    diag2_used: List[bool],
) -> List[int]:
    """Compute all rows that are currently safe for a given column.

    Parameters
    ----------
    size : int
        Board size (N).
    column : int
        Column index for which to compute safe rows.
    row_used, diag1_used, diag2_used : list[bool]
        Current constraint masks (True means the row/diagonal is occupied).

    Returns
    -------
    list[int]
        Rows r such that placing a queen at (r, column) violates no constraint.
    """
    # Shift index so the range [-size+1, size-1] maps to [0, 2*size-2].
    offset = size - 1
    candidates: List[int] = []
    for row in range(size):
        if row_used[row]:
            continue
        diag1_index = row - column + offset
        diag2_index = row + column
        if not diag1_used[diag1_index] and not diag2_used[diag2_index]:
            # Row is safe w.r.t. both diagonals, keep it as a candidate.
            candidates.append(row)
    return candidates


def _iterative_backtracking(
    size: int,
    select_column: Callable[
        [List[int], List[bool], List[bool], List[bool]], Tuple[Optional[int], List[int]]
    ],
    time_limit: Optional[float] = None,

) -> Tuple[Optional[List[int]], int, float]:
    """Generic non-recursive backtracking with pluggable ordering.

    Parameters
    ----------
    size : int
        Problem size N.
    select_column : callable
        Strategy hook that, given the current set of unassigned columns and
        constraint masks, returns a tuple `(column, candidates)` where:
          - column: the next column to assign (int) or None if solved.
          - candidates: list[int] of feasible rows for that column, ordered
            according to the heuristic (may be empty to trigger backtrack).
    time_limit : float | None
        Optional wall-clock time limit in seconds.

    Returns
    -------
    (solution, nodes_explored, elapsed_seconds)
        See module-level contract. Deterministic for a deterministic selector.

        Notes
        -----
        - Uses a small `_Frame` dataclass to store the column, its ordered
            candidates, and the next candidate index to try. This enables true
            non-recursive DFS with explicit state.
        - Tie-breaking: delegate entirely to `select_column`; this routine preserves
            given order and updates it only by advancing `next_index`.
        """

    positions = [-1] * size
    row_used = [False] * size
    diag1_used = [False] * (2 * size - 1)
    diag2_used = [False] * (2 * size - 1)
    unassigned = list(range(size))
    stack: List[_Frame] = []
    explored = 0
    start = perf_counter()
    # Reuse the offset when translating diagonal indices.
    offset = size - 1

    while True:
        if time_limit is not None and (perf_counter() - start) > time_limit:
            return None, explored, perf_counter() - start

        if not stack:
            if not unassigned:
                return positions.copy(), explored, perf_counter() - start

            column, candidates = select_column(unassigned, row_used, diag1_used, diag2_used)
            if column is None or not candidates:
                return None, explored, perf_counter() - start

            # Create a new decision frame for the selected column.
            stack.append(_Frame(column, candidates))
            unassigned.remove(column)
            continue

        frame = stack[-1]
        column = frame.column
        candidates = frame.candidates
        index = frame.next_index

        if positions[column] != -1:
            # Remove the previous assignment for this column before trying a new row.
            row = positions[column]
            positions[column] = -1
            row_used[row] = False
            diag1_index = row - column + offset
            diag2_index = row + column
            diag1_used[diag1_index] = False
            diag2_used[diag2_index] = False

        if index >= len(candidates):
            # All candidate rows failed; backtrack to the previous column.
            stack.pop()
            unassigned.append(column)
            continue

        row = candidates[index]
        frame.next_index = index + 1
        explored += 1

        diag1_index = row - column + offset
        diag2_index = row + column
        if row_used[row] or diag1_used[diag1_index] or diag2_used[diag2_index]:
            # Constraint now violated because of downstream assignments; skip value.
            continue

        positions[column] = row
        row_used[row] = True
        diag1_used[diag1_index] = True
        diag2_used[diag2_index] = True

        if not unassigned:
            return positions.copy(), explored, perf_counter() - start

        next_column, next_candidates = select_column(unassigned, row_used, diag1_used, diag2_used)
        if next_column is None:
            return positions.copy(), explored, perf_counter() - start
        if not next_candidates:
            continue

        # Descend one level deeper in the search tree with the chosen variable order.
        stack.append(_Frame(next_column, next_candidates))
        unassigned.remove(next_column)


def bt_nqueens_mcv(size: int, time_limit: Optional[float] = None) -> Tuple[Optional[List[int]], int, float]:
    """Backtracking using the Most Constrained Variable (MCV) heuristic.

        Heuristic
        ---------
        - Select next column among the unassigned ones as that with the fewest
            currently legal rows (minimum domain size).
        - Tie-break deterministically by the smallest column index for reproducible
            behavior.

        Determinism
        -----------
        - For a fixed `size`, the variable order is deterministic. Value order is
            the natural ascending row order returned by `_available_rows`.

    Complexity
    ----------
    Worst-case exponential; MCV typically lowers branching by exposing dead-ends
    earlier in CSPs like N-Queens.
    """

    def select_column(
        unassigned: List[int],
        row_used: List[bool],
        diag1_used: List[bool],
        diag2_used: List[bool],
    ) -> Tuple[Optional[int], List[int]]:
        best_column: Optional[int] = None
        best_candidates: List[int] = []
        min_candidates = size + 1

        for column in unassigned:
            candidates = _available_rows(size, column, row_used, diag1_used, diag2_used)
            cand_len = len(candidates)
            if cand_len == 0:
                # Immediate dead-end for this column; force a quick backtrack.
                return column, []
            if cand_len < min_candidates or (cand_len == min_candidates and (best_column is None or column < best_column)):
                # Prefer the column with the fewest legal rows; break ties deterministically.
                best_column = column
                best_candidates = candidates
                min_candidates = cand_len
                if min_candidates == 1:
                    break

        return best_column, best_candidates

    return _iterative_backtracking(size, select_column, time_limit)


def bt_nqueens_lcv(size: int, time_limit: Optional[float] = None) -> Tuple[Optional[List[int]], int, float]:
    """Backtracking using the Least Constraining Value (LCV) heuristic.

        Heuristic
        ---------
        - Select next column deterministically as the smallest index in the set of
            unassigned columns (to focus only on value ordering, not variable order).
        - Score each candidate row by the total number of options it leaves across
            the remaining unassigned columns when tentatively applied.
            Higher score is preferred (least constraining).
        - If a candidate would immediately make some other column impossible
            (no legal rows), assign it score -1 so it is tried last.
        - Final value order: sort by descending score, then by row index (stable,
            deterministic ordering).

        Determinism
        -----------
        - For a fixed `size`, both column and row orders are deterministic given
            the scoring rule above; results are reproducible.

        Complexity
        ----------
        - Additional overhead per decision due to scoring (it queries the domains of
            remaining columns), but often a significant reduction in backtracking.
        - Worst-case remains exponential.
        """

    offset = size - 1

    def select_column(
        unassigned: List[int],
        row_used: List[bool],
        diag1_used: List[bool],
        diag2_used: List[bool],
    ) -> Tuple[Optional[int], List[int]]:
        if not unassigned:
            return None, []

        column = min(unassigned)
        candidates = _available_rows(size, column, row_used, diag1_used, diag2_used)
        if not candidates:
            # No feasible value for this column; propagate the failure upward.
            return column, []

        scores = {}
        for row in candidates:
            diag1_index = row - column + offset
            diag2_index = row + column
            # Temporarily commit to (column, row) to measure its downstream impact.
            row_used[row] = True
            diag1_used[diag1_index] = True
            diag2_used[diag2_index] = True

            total_options = 0
            feasible = True
            for other_column in unassigned:
                if other_column == column:
                    continue
                options = _available_rows(size, other_column, row_used, diag1_used, diag2_used)
                if not options:
                    # A neighboring column would become impossible; penalise this value.
                    feasible = False
                    break
                total_options += len(options)

            row_used[row] = False
            diag1_used[diag1_index] = False
            diag2_used[diag2_index] = False

            # Higher scores are better: they leave more slack for the remaining columns.
            scores[row] = total_options if feasible else -1

        # Choose values that constrain the future search the least (higher score first).
        candidates.sort(key=lambda r: (-scores[r], r))
        return column, candidates

    return _iterative_backtracking(size, select_column, time_limit)
