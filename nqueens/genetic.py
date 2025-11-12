"""Genetic Algorithm solver for the N-Queens problem.

This module implements a simple, effective Genetic Algorithm (GA) to search
for valid N-Queens configurations. The representation is a length-N list where
``board[col] = row``. A configuration is a solution when no pairs of queens
attack each other.

Contract (public API)
---------------------
- Input: problem size ``size >= 1`` and typical GA hyperparameters.
- Output: a 6-tuple ``GAResult`` summarizing the run:
    (success, iterations, elapsed_seconds, best_conflicts, evaluations, timeout)

Where:
- success: True when a conflict-free board was found, False otherwise.
- iterations: number of generations executed when the run ended.
- elapsed_seconds: wall time measured with ``perf_counter()``.
- best_conflicts: the lowest conflict count observed (0 on success).
- evaluations: number of objective evaluations (fitness/conflicts) performed.
- timeout: True when terminated due to ``time_limit``.

Determinism
-----------
The algorithm is stochastic. For reproducible experiments, set the Python
``random`` seed externally before invoking the solver.
"""

from __future__ import annotations

import random
from time import perf_counter
from typing import Callable, List, Optional, Sequence, Tuple

from .fitness import get_fitness_function
from .utils import conflicts

GAResult = Tuple[bool, int, float, int, int, bool]


def ga_nqueens(
    size: int,
    pop_size: int = 100,
    max_gen: int = 1000,
    pc: float = 0.8,
    pm: float = 0.1,
    tournament_size: int = 3,
    fitness_mode: str = "F1",
    time_limit: Optional[float] = None,
) -> GAResult:
    """Run a Genetic Algorithm search for an N-Queens solution.

    Parameters
    ----------
    size : int
        Board dimension N.
    pop_size : int, default 100
        Number of individuals in the population. A minimum of ~50 is often
        advisable for diversity on medium-sized boards.
    max_gen : int, default 1000
        Maximum number of generations.
    pc : float, default 0.8
        Crossover probability. Single-point crossover is used.
    pm : float, default 0.1
        Mutation probability. A mutation replaces the row of a random column
        with a new random row.
    tournament_size : int, default 3
        Tournament selection size.
    fitness_mode : str, default "F1"
        Label selecting the fitness function (see ``nqueens.fitness``).
    time_limit : float | None
        Optional wall-clock time limit in seconds.

    Returns
    -------
    GAResult
        Tuple (success, generations, elapsed, best_conflicts, evaluations, timeout).

    Notes
    -----
    - Representation: ``board[col] = row``. Multiple queens may share rows;
      the fitness discourages conflicts and directs the search.
        - Evaluations count includes fitness evaluations only; additional
            conflict checks used to track/report the best individual are not
            included in this counter.
    - If a perfect (0-conflict) solution is found, the run terminates early.
    """
    fitness_function = get_fitness_function(fitness_mode)

    population: List[List[int]] = [
        [random.randrange(size) for _ in range(size)] for _ in range(pop_size)
    ]
    # Initial fitness evaluation for the entire population
    fitness_values = [fitness_function(individual) for individual in population]
    evaluations = pop_size

    best_index = max(range(pop_size), key=lambda i: fitness_values[i])
    best_individual = population[best_index][:]
    # Track best quality using number of conflicts for clarity in reports
    best_conflicts = conflicts(best_individual)
    start = perf_counter()

    if best_conflicts == 0:
        return True, 0, perf_counter() - start, 0, evaluations, False

    def tournament() -> int:
        """Return the index of the winner of a tournament selection."""
        winner = None
        for _ in range(tournament_size):
            candidate = random.randrange(pop_size)
            if winner is None or fitness_values[candidate] > fitness_values[winner]:
                winner = candidate
        assert winner is not None
        return winner

    generation = 0
    while generation < max_gen:
        if time_limit is not None and (perf_counter() - start) > time_limit:
            return False, generation, perf_counter() - start, best_conflicts, evaluations, True

        generation += 1
        # Elitism: keep a copy of the current best individual
        new_population: List[List[int]] = [best_individual[:]]

        while len(new_population) < pop_size:
            parent1 = population[tournament()]
            parent2 = population[tournament()]

            if random.random() < pc:
                # Single-point crossover
                cut = random.randrange(1, size)
                child1 = parent1[:cut] + parent2[cut:]
                child2 = parent2[:cut] + parent1[cut:]
            else:
                child1 = parent1[:]
                child2 = parent2[:]

            def mutate(individual: List[int]) -> None:
                """Apply point mutation with probability pm to a random column."""
                if random.random() < pm:
                    column = random.randrange(size)
                    individual[column] = random.randrange(size)

            mutate(child1)
            mutate(child2)

            new_population.append(child1)
            if len(new_population) < pop_size:
                new_population.append(child2)

        population = new_population
        # Evaluate the whole population's fitness
        fitness_values = [fitness_function(individual) for individual in population]
        evaluations += pop_size

        # Track best individual by actual conflicts for clear reporting
        for individual in population:
            conflicts_count = conflicts(individual)
            if conflicts_count < best_conflicts:
                best_conflicts = conflicts_count
                best_individual = individual[:]

        if best_conflicts == 0:
            return True, generation, perf_counter() - start, 0, evaluations, False

    return False, max_gen, perf_counter() - start, best_conflicts, evaluations, False
