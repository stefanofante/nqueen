"""Genetic Algorithm solver for the N-Queens problem."""

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
    """Execute the GA and return the execution summary."""
    fitness_function = get_fitness_function(fitness_mode)

    population: List[List[int]] = [
        [random.randrange(size) for _ in range(size)] for _ in range(pop_size)
    ]
    fitness_values = [fitness_function(individual) for individual in population]
    evaluations = pop_size

    best_index = max(range(pop_size), key=lambda i: fitness_values[i])
    best_individual = population[best_index][:]
    best_conflicts = conflicts(best_individual)
    start = perf_counter()

    if best_conflicts == 0:
        return True, 0, perf_counter() - start, 0, evaluations, False

    def tournament() -> int:
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
        new_population: List[List[int]] = [best_individual[:]]  # elitism

        while len(new_population) < pop_size:
            parent1 = population[tournament()]
            parent2 = population[tournament()]

            if random.random() < pc:
                cut = random.randrange(1, size)
                child1 = parent1[:cut] + parent2[cut:]
                child2 = parent2[:cut] + parent1[cut:]
            else:
                child1 = parent1[:]
                child2 = parent2[:]

            def mutate(individual: List[int]) -> None:
                if random.random() < pm:
                    column = random.randrange(size)
                    individual[column] = random.randrange(size)

            mutate(child1)
            mutate(child2)

            new_population.append(child1)
            if len(new_population) < pop_size:
                new_population.append(child2)

        population = new_population
        fitness_values = [fitness_function(individual) for individual in population]
        evaluations += pop_size

        for individual in population:
            conflicts_count = conflicts(individual)
            if conflicts_count < best_conflicts:
                best_conflicts = conflicts_count
                best_individual = individual[:]

        if best_conflicts == 0:
            return True, generation, perf_counter() - start, 0, evaluations, False

    return False, max_gen, perf_counter() - start, best_conflicts, evaluations, False
