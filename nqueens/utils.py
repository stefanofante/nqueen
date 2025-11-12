"""Utility helpers for the N-Queens project.

This module exposes low-level primitives that are shared across the
individual algorithm implementations, such as conflict counters.
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence


def conflicts(board: Sequence[int]) -> int:
    """Compute the number of conflicting queen pairs in *O(N).*"""
    n = len(board)
    row_count: Counter[int] = Counter()
    diag1: Counter[int] = Counter()
    diag2: Counter[int] = Counter()

    for column, row in enumerate(board):
        row_count[row] += 1
        diag1[row - column] += 1
        diag2[row + column] += 1

    def _pairs(counter: Counter[int]) -> int:
        total = 0
        for count in counter.values():
            if count > 1:
                total += count * (count - 1) // 2
        return total

    return _pairs(row_count) + _pairs(diag1) + _pairs(diag2)


def conflicts_on2(board: Sequence[int]) -> int:
    """Compute the number of conflicting queen pairs in *O(N^2).*"""
    n = len(board)
    conflicts_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if board[i] == board[j] or abs(board[i] - board[j]) == abs(i - j):
                conflicts_count += 1
    return conflicts_count
