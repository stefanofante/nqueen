# N-Queens Algorithm Comparison: Advanced Analysis Wiki

This wiki provides comprehensive documentation for the N-Queens comparative algorithm analysis project, featuring Backtracking (BT), Simulated Annealing (SA), and Genetic Algorithm (GA) implementations. Solver code lives inside the `nqueens/` package. The orchestration, tuning, reporting and plotting are modularized under `nqueens/analysis/`. The file `algoanalisys.py` is now a thin, backwards-compatible facade that re-exports public APIs and provides the CLI entry point.

## Quick Usage

Filter which algorithms run using `--alg` (`-a`). Valid values: `BT`, `SA`, `GA` (repeatable or comma-separated).

```bash
# List algorithms, BT solvers, and GA fitness modes
python algoanalisys.py --list

# GA only (experiments only; tuning on-demand)
python algoanalisys.py -a GA

# GA only, run tuning first then experiments
python algoanalisys.py -a GA --tune

# GA only, riusando i parametri da config.json (default)
python algoanalisys.py -a GA --config config.json

# SA only (no GA, no BT)
python algoanalisys.py --mode sequential -a SA

# BT only
python algoanalisys.py -a BT

# SA + GA for selected fitness functions
python algoanalisys.py -a SA,GA -f F1,F3
```

Behavior notes:

- Tuning runs only when `--tune` is passed; default is to reuse stored parameters.
- If `GA` is excluded (e.g., `-a SA` or `-a BT`), tuning is skipped automatically.
- When tuning runs, optimal GA parameters are persisted to `config.json`.
- Without `--tune`, GA parameters are loaded from `config.json`. If parameters for the selected fitness/N are missing or empty, the CLI automatically performs tuning as a fallback and saves the results.

## Overview

This project compares three fundamental approaches to solving the N-Queens problem:

- **BT** (Backtracking): Iterative implementation without recursion
- **SA** (Simulated Annealing): Metaheuristic optimization with temperature control  
- **GA** (Genetic Algorithm): Evolutionary computation with **fitness functions F1-F6**

## Objectives

- Study the impact of **problem size N** on success rates and computational cost
- Evaluate **alternative fitness functions** for genetic algorithms
- Define a **systematic parameter tuning procedure** for GA across different problem sizes
- Achieve **performance optimizations** (O(N) conflicts calculation, multiprocessing, optional Numba acceleration)

## Project Structure

Core solvers and shared utilities:

- `nqueens/backtracking.py`: iterative backtracking solver
- `nqueens/simulated_annealing.py`: simulated annealing solver
- `nqueens/genetic.py`: genetic algorithm driver
- `nqueens/fitness.py`: shared GA fitness functions
- `nqueens/utils.py`: conflict counters, solution validator (`is_valid_solution`), and shared utilities

Modular orchestration layer:

- `nqueens/analysis/settings.py`: global settings, timeouts, tuning grids, output paths
- `nqueens/analysis/stats.py`: typed result records and statistical aggregations
- `nqueens/analysis/tuning.py`: GA parameter search (sequential and parallel)
- `nqueens/analysis/experiments.py`: experiment runners (BT/SA/GA) with optimal parameters
- `nqueens/analysis/reporting.py`: CSV exports for aggregated/raw data and logical cost analysis
- `nqueens/analysis/plots.py`: chart generation (optional, requires matplotlib)
- `nqueens/analysis/cli.py`: pipelines, CLI wiring, quick regression runner, flags: `--mode`, `--fitness`, `--validate`
- `nqueens/analysis/cli.py`: pipelines, CLI wiring, quick regression runner, flags: `--mode`, `--fitness`, `--list`, `--validate`

Compatibility facade:

- `algoanalisys.py`: thin wrapper re-exporting the public APIs from `nqueens.analysis.*` and exposing the CLI entry point

## Mathematical Foundation

### Problem Definition

The N-Queens problem requires placing N queens on an N x N chessboard such that no two queens attack each other.

**Constraints:**

- Row constraint: ≤ 1 queen per row
- Column constraint: ≤ 1 queen per column  
- Main diagonal constraint: ≤ 1 queen per diagonal (slope +1)
- Anti-diagonal constraint: ≤ 1 queen per diagonal (slope -1)

### Solution Representation

Solutions are represented as permutation arrays `board[N]` where `board[i] = j` means the queen in column `i` is placed in row `j`. This representation automatically satisfies the column constraint.

### Conflict Calculation

The quality of a solution is measured by counting conflicting queen pairs:

```python
def conflicts(board):
    """Efficient O(N) conflict counting using Counter data structures"""
    n = len(board)
    row_count = Counter()
    diag1 = Counter()  # Main diagonal (row - col)
    diag2 = Counter()  # Anti-diagonal (row + col)
    
    for col, row in enumerate(board):
        row_count[row] += 1
        diag1[row - col] += 1
        diag2[row + col] += 1
    
    conflicts = 0
    for count in row_count.values():
        conflicts += count * (count - 1) // 2
    for count in diag1.values():
        conflicts += count * (count - 1) // 2  
    for count in diag2.values():
        conflicts += count * (count - 1) // 2
        
    return conflicts
```

**Optimizations:**

- Time complexity: O(N) instead of naive O(N²)
- Uses combinatorics: conflicts = C(k,2) for each group of k queens
- Memory efficient with Counter data structures

## Algorithm Implementations

### 1. Backtracking (BT)

**Implementation Strategy:**

- Iterative depth-first search using explicit stack
- Avoids recursion overhead and stack overflow issues
- Deterministic: always finds the lexicographically first solution

**Key Features:**

- **Completeness**: Guaranteed to find solution if one exists
- **Optimality**: Finds optimal (valid) solution
- **Determinism**: Reproducible results
- **Time Complexity**: O(N!) worst case, often much better with pruning

Note: nelle analisi/esperimenti viene usata di default la variante ibrida `bt_nqueens_mcv_hybrid` (selezione MCV + look-ahead parziale stile LCV) per ridurre backtrack su N grandi. Rimangono disponibili `bt_nqueens_first`, `bt_nqueens_mcv`, `bt_nqueens_lcv`.

**Code Structure:**

```python
def bt_nqueens_first(N, time_limit=None):
    """Iterative backtracking with timeout support"""
    start_time = perf_counter()
    nodes = 0
    stack = [(0, [-1] * N)]  # (column, partial_board)
    
    while stack:
        if time_limit and perf_counter() - start_time > time_limit:
            return False, [], nodes, perf_counter() - start_time
            
        col, board = stack.pop()
        nodes += 1
        
        if col == N:  # Complete solution found
            return True, board, nodes, perf_counter() - start_time
        
        # Try all positions in current column
        for row in range(N):
            if is_safe(board, col, row):
                new_board = board[:]
                new_board[col] = row
                stack.append((col + 1, new_board))
    
    return False, [], nodes, perf_counter() - start_time
```

### 2. Simulated Annealing (SA)

**Implementation Strategy:**

- Start with random permutation solution
- Iteratively improve using neighbor generation and probabilistic acceptance
- Uses geometric cooling schedule for temperature control

**Key Features:**

- **Metaheuristic**: Probabilistic local search
- **Escape Mechanism**: Can accept worse solutions to escape local optima
- **Parameter Sensitivity**: Performance depends on cooling schedule and temperature
- **Time Complexity**: O(max_iterations x N)

**Neighbor Generation:**

- **Swap Move**: Exchange positions of two randomly selected queens
- Maintains permutation property automatically
- Efficient O(1) neighbor generation

**Acceptance Criterion:**

```python
# Metropolis criterion
if delta <= 0 or random.random() < math.exp(-delta / temperature):
    accept_move()
```

**Code Structure:**

```python
def sa_nqueens(N, max_iter=10000, initial_temp=100, cooling_rate=0.95, time_limit=None):
    """Simulated Annealing with geometric cooling"""
    current = list(range(N))
    random.shuffle(current)
    current_conflicts = conflicts(current)
    
    temperature = initial_temp
    best = current[:]
    best_conflicts = current_conflicts
    
    for iteration in range(max_iter):
        if current_conflicts == 0:
            return True, current, iteration, iteration, time
            
        # Generate neighbor by swapping two positions
        neighbor = current[:]
        i, j = random.sample(range(N), 2)
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        
        neighbor_conflicts = conflicts(neighbor)
        delta = neighbor_conflicts - current_conflicts
        
        # Metropolis acceptance
        if delta <= 0 or random.random() < math.exp(-delta / temperature):
            current = neighbor
            current_conflicts = neighbor_conflicts
            
            if current_conflicts < best_conflicts:
                best = current[:]
                best_conflicts = current_conflicts
        
        temperature *= cooling_rate
    
    return best_conflicts == 0, best, max_iter, max_iter, time
```

### 3. Genetic Algorithm (GA)

**Implementation Strategy:**

- Population-based evolutionary optimization
- Multiple fitness functions (F1-F6) for comparison
- Tournament selection, order crossover, swap mutation

**Key Features:**

- **Population Diversity**: Maintains multiple candidate solutions
- **Scalability**: Excellent performance on large instances
- **Fitness Flexibility**: Six different evaluation functions
- **Parameter Rich**: Many tunable parameters for optimization

**Genetic Operators:**

#### Selection: Tournament Selection

```python
def tournament_selection(population, fitness_values, tournament_size):
    """Select best individual from random tournament"""
    tournament_indices = random.sample(range(len(population)), tournament_size)
    tournament_fitness = [fitness_values[i] for i in tournament_indices]
    winner_index = tournament_indices[tournament_fitness.index(min(tournament_fitness))]
    return population[winner_index][:]
```

#### Crossover: Order Crossover (OX)

```python
def order_crossover(parent1, parent2):
    """Preserve relative order from parents"""
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    child = [-1] * size
    child[start:end] = parent1[start:end]
    
    # Fill remaining positions maintaining order from parent2
    pointer = 0
    for i in range(size):
        if child[i] == -1:
            while parent2[pointer] in child:
                pointer += 1
            child[i] = parent2[pointer]
            pointer += 1
    
    return child
```

#### Mutation: Swap Mutation

```python
def swap_mutation(individual):
    """Swap two randomly selected positions"""
    individual = individual[:]
    i, j = random.sample(range(len(individual)), 2)
    individual[i], individual[j] = individual[j], individual[i]
    return individual
```

## Fitness Functions (F1-F6)

The genetic algorithm implements six different fitness functions for comprehensive comparison:

### F1: Basic Conflict Counting

```python
def fitness_f1(board):
    """Standard conflict counting - minimize conflicts"""
    return conflicts(board)
```

- **Objective**: Minimize total conflicts

- **Characteristics**: Direct, simple, baseline approach
- **Performance**: Good general performance

### F2: Weighted Conflict Penalty

```python
def fitness_f2(board):
    """Weighted conflicts with exponential penalty"""
    base_conflicts = conflicts(board)
    return base_conflicts + (base_conflicts ** 1.5)
```

- **Objective**: Exponential penalty for higher conflicts

- **Characteristics**: Aggressive conflict reduction
- **Performance**: Fast convergence on small instances

### F3: Advanced Constraint Satisfaction

```python
def fitness_f3(board):
    """Separate penalty for each constraint type"""
    row_conflicts = calculate_row_conflicts(board)
    diag_conflicts = calculate_diagonal_conflicts(board)
    return 2 * row_conflicts + 3 * diag_conflicts
```

- **Objective**: Differentiated penalty by constraint type

- **Characteristics**: Targeted constraint handling
- **Performance**: Effective on structured problems

### F4: Multi-objective Optimization

```python
def fitness_f4(board):
    """Balance conflict minimization with solution quality"""
    base_conflicts = conflicts(board)
    diversity_bonus = calculate_diversity(board)
    return base_conflicts - 0.1 * diversity_bonus
```

- **Objective**: Minimize conflicts while maintaining diversity

- **Characteristics**: Prevents premature convergence
- **Performance**: Robust across problem sizes

### F5: Adaptive Penalty System

```python
def fitness_f5(board):
    """Dynamic penalty adaptation based on problem size"""
    N = len(board)
    base_conflicts = conflicts(board)
    scale_factor = math.log(N + 1)
    return base_conflicts * scale_factor
```

- **Objective**: Adaptive scaling for different problem sizes

- **Characteristics**: Size-aware optimization
- **Performance**: Consistent across scale variations

### F6: Hybrid Evaluation Method

```python
def fitness_f6(board):
    """Combination of multiple evaluation criteria"""
    conflict_score = conflicts(board)
    constraint_score = evaluate_constraint_satisfaction(board)
    efficiency_score = calculate_search_efficiency(board)
    return 0.6 * conflict_score + 0.3 * constraint_score + 0.1 * efficiency_score
```

- **Objective**: Multi-criteria evaluation with weighted combination

- **Characteristics**: Comprehensive solution assessment
- **Performance**: Best overall performance on complex instances

## Parameter Tuning Framework

### Tuning Strategy

**Objective**: Find optimal GA parameters for each problem size N and fitness function.

**Parameters Optimized:**

- **Population Size**: `pop_size = multiplier x N` where multiplier in {4, 8, 16}
- **Maximum Generations**: `max_gen = multiplier x N` where multiplier in {30, 50, 80}
- **Mutation Rate**: `pm in {0.05, 0.10, 0.15}`
- **Crossover Rate**: `pc = 0.8` (fixed)
- **Tournament Size**: `tournament_size = 3` (fixed)

### Tuning Procedure

1. **Grid Search**: Exhaustive search over parameter combinations
2. **Multiple Runs**: 5 independent runs per parameter combination
3. **Success Evaluation**: Percentage of runs finding optimal solution (0 conflicts)
4. **Best Selection**: Choose parameters maximizing success rate
5. **Tie Breaking**: If multiple parameter sets achieve same success rate, choose fastest

### Implementation

```python
def tune_ga_for_N(N, fitness_mode, pop_multipliers, gen_multipliers, pm_values):
    """Systematic parameter tuning for specific N and fitness function"""
    best_params = None
    best_success_rate = -1
    best_avg_gen = float('inf')
    
    for pop_mult in pop_multipliers:
        for gen_mult in gen_multipliers:
            for pm in pm_values:
                pop_size = pop_mult * N
                max_gen = gen_mult * N
                
                # Run multiple independent trials
                success_count = 0
                total_gen = 0
                
                for _ in range(RUNS_GA_TUNING):
                    success, _, gen, _, _ = ga_nqueens(
                        N, pop_size, max_gen, PC_FIXED, pm, 
                        TOURNAMENT_SIZE_FIXED, fitness_mode, GA_TIME_LIMIT
                    )
                    if success:
                        success_count += 1
                        total_gen += gen
                
                success_rate = success_count / RUNS_GA_TUNING
                avg_gen = total_gen / max(success_count, 1)
                
                # Update best parameters
                if (success_rate > best_success_rate or 
                    (success_rate == best_success_rate and avg_gen < best_avg_gen)):
                    best_success_rate = success_rate
                    best_avg_gen = avg_gen
                    best_params = {
                        'pop_size': pop_size,
                        'max_gen': max_gen,
                        'pm': pm,
                        'success_rate': success_rate,
                        'avg_gen_success': avg_gen
                    }
    
    return best_params
```

## Performance Analysis

### Computational Complexity

| Algorithm | Time Complexity | Space Complexity | Characteristics |
|-----------|----------------|------------------|-----------------|
| Backtracking | O(N!) | O(N) | Exponential, deterministic |
| Simulated Annealing | O(max_iter x N) | O(N) | Linear, probabilistic |
| Genetic Algorithm | O(max_gen x pop_size x N) | O(pop_size x N) | Scalable, population-based |

### Scalability Analysis

**Small Instances (N ≤ 16):**

- **Backtracking**: Optimal choice, very fast execution
- **SA/GA**: Overkill, but useful for algorithm comparison

**Medium Instances (16 < N ≤ 40):**

- **Backtracking**: Becomes exponentially slow
- **Simulated Annealing**: Sweet spot for performance
- **Genetic Algorithm**: Good performance with proper tuning

**Large Instances (N > 40):**

- **Backtracking**: Impractical due to exponential time
- **Simulated Annealing**: Moderate performance
- **Genetic Algorithm**: Best choice, excellent scalability

### Expected Performance Characteristics

#### Success Rate vs Problem Size

- **BT**: 100% success rate for small N, impractical for large N

- **SA**: High success rate for medium N, degrades gracefully  
- **GA**: Fitness-dependent, generally excellent for large N

#### Execution Time Scaling

- **BT**: Exponential growth, fast for N ≤ 12

- **SA**: Near-linear growth, consistent performance
- **GA**: Controlled by population and generation parameters

#### Resource Usage

- **BT**: Minimal memory, CPU-intensive for large N

- **SA**: Low memory, moderate CPU usage
- **GA**: Memory scales with population size

## Statistical Analysis Framework

### Metrics Collected

**Primary Metrics:**

- **Success Rate**: Percentage of runs finding optimal solution
- **Execution Time**: Wall-clock time for successful runs
- **Algorithmic Cost**: Nodes explored (BT), iterations (SA), generations (GA)
- **Solution Quality**: Conflicts in best solution found

**Advanced Metrics:**

- **Timeout Rate**: Percentage of runs exceeding time limits
- **Convergence Speed**: Average time/iterations to solution
- **Algorithm Stability**: Variance in performance across runs
- **Failure Quality**: Best solution quality in unsuccessful attempts

### Statistical Tests

**Descriptive Statistics:**

- Mean, median, standard deviation
- Quartiles and percentiles
- Confidence intervals (95%)

**Distribution Analysis:**

- Normality testing (Shapiro-Wilk)
- Outlier detection (IQR method)
- Histogram visualization

**Comparative Analysis:**

- Algorithm performance ranking
- Pairwise comparisons
- Effect size calculation

## Visualization Framework

### Chart Categories

**1. Performance Overview (9 charts per fitness)**

- Success rate vs problem size
- Execution time vs problem size (log scale)
- Logical cost vs problem size  
- Fitness evaluations comparison
- Timeout analysis
- Failure quality assessment
- Theoretical vs practical correlation
- Algorithm stability metrics
- Scalability visualization

**2. Fitness Function Comparison (6 charts)**

- Success rate comparison across fitness functions
- Convergence speed analysis
- Execution time trade-offs
- Pareto efficiency frontier
- Multi-dimensional performance
- Cross-fitness scalability

**3. Statistical Analysis (12+ charts)**

- Box plots for time distribution
- Histogram distributions for stability
- Violin plots for detailed distributions
- Correlation analysis heatmaps
- Confidence interval visualizations
- Outlier analysis charts

**4. Parameter Tuning Analysis (10+ charts)**

- Heatmaps for parameter optimization
- 3D surface plots for parameter interactions
- Cost vs quality scatter plots
- Sensitivity analysis for mutation rate
- Population size impact assessment
- Generation limit optimization curves

### Chart Styling

**Professional Standards:**

- High resolution (300 DPI) output
- Scientific color schemes
- Clear axis labeling with units
- Comprehensive legends
- Grid lines for readability
- Statistical annotations (p-values, confidence intervals)

**Technical Features:**

- Matplotlib backend for precision (optional dependency)
- Seaborn integration for statistical plots (optional)
- Custom styling for consistency
- LaTeX rendering for mathematical expressions
- Multi-format export (PNG, PDF, SVG)

## Data Export System

### CSV File Structure

Column naming uses lowercase snake_case. Aggregates and logical-cost CSVs prefix fields with `bt_`, `sa_`, `ga_`. Time fields are in seconds.

**Aggregated Results (contextual naming):**

```text
# With GA present: results_GA_<FITNESS>_tuned.csv
# Without GA:      results_BT.csv / results_SA.csv / results_BT_SA.csv
n,
bt_solution_found, bt_nodes_explored, bt_time_seconds,
bt_first_solution_found, bt_first_nodes_explored, bt_first_time_seconds,
bt_mcv_solution_found, bt_mcv_nodes_explored, bt_mcv_time_seconds,
bt_lcv_solution_found, bt_lcv_nodes_explored, bt_lcv_time_seconds,
bt_mcv_hybrid_solution_found, bt_mcv_hybrid_nodes_explored, bt_mcv_hybrid_time_seconds,
sa_success_rate, sa_timeout_rate, sa_failure_rate, sa_total_runs, sa_successes, sa_failures, sa_timeouts,
sa_success_steps_mean, sa_success_steps_median, sa_success_evals_mean, sa_success_evals_median,
sa_timeout_steps_mean, sa_timeout_steps_median, sa_timeout_evals_mean, sa_timeout_evals_median,
sa_success_time_mean, sa_success_time_median,
ga_success_rate, ga_timeout_rate, ga_failure_rate, ga_total_runs, ga_successes, ga_failures, ga_timeouts,
ga_success_gen_mean, ga_success_gen_median, ga_success_evals_mean, ga_success_evals_median,
ga_timeout_gen_mean, ga_timeout_gen_median, ga_timeout_evals_mean, ga_timeout_evals_median,
ga_success_time_mean, ga_success_time_median,
ga_pop_size, ga_max_gen, ga_pm, ga_pc, ga_tournament_size
```

Nota: i campi `bt_*` senza suffisso fanno riferimento al solver ibrido `bt_nqueens_mcv_hybrid` per compatibilità; le colonne aggiuntive riportano i risultati per ogni variante BT.

**Raw Data — Simulated Annealing:**

```text
# With GA present: raw_data_SA_<FITNESS>.csv
# Without GA:      raw_data_SA.csv
n, run_id, algorithm, success, timeout, steps, time_seconds, evals, best_conflicts
```

**Raw Data — Genetic Algorithm (`raw_data_GA_<FITNESS>.csv`):**

```text
n, run_id, algorithm, success, timeout, gen, time_seconds, evals, best_fitness, best_conflicts,
pop_size, max_gen, pm, pc, tournament_size
```

**Raw Data — Backtracking:**

```text
# With GA present: raw_data_BT_<FITNESS>.csv
# Without GA:      raw_data_BT.csv
# Note: generated only if BT is selected/run
n, algorithm, solution_found, nodes_explored, time_seconds
```

**Logical Cost Analysis:**

```text
# With GA present: logical_costs_<FITNESS>.csv
# Without GA:      logical_costs.csv
n,
bt_solution_found, bt_nodes_explored,
sa_success_rate, sa_steps_mean_all, sa_steps_median_all, sa_evals_mean_all, sa_evals_median_all,
sa_steps_mean_success, sa_evals_mean_success,
ga_success_rate, ga_gen_mean_all, ga_gen_median_all, ga_evals_mean_all, ga_evals_median_all,
ga_gen_mean_success, ga_evals_mean_success,
bt_time_seconds, sa_time_mean_success, ga_time_mean_success
```

### Logging Semantics

- When `GA` is not selected, the CLI ignores any `--fitness/-f` filter and avoids mentioning fitness in banners and progress logs.
- Plotting is optional; attempting to plot without `matplotlib` installed raises a clear runtime error only when plotting APIs are called.

### Capability Listing (`--list`)

Use `--list` to print discoverable capabilities (it reads `config.json` from the project root if present, otherwise uses defaults):

```text
Available algorithms: BT, SA, GA
Backtracking solvers: first, lcv, mcv, mcv_hybrid
GA fitness modes:
    - F1 — Fitness: negative conflicts (higher is better).
    - F2 — Fitness: number of non-conflicting queen pairs.
    - F3 — Fitness: linear penalty on diagonal clusters (mild).
    - F4 — Fitness: penalize worst-case queen conflicts.
    - F5 — Fitness: quadratic penalty on diagonal clusters (strong).
    - F6 — Fitness: exp(-lam * conflicts(board)).
```

- Backtracking solvers are discovered dynamically by prefix `bt_nqueens_*` in `nqueens.backtracking`.
- Fitness descriptions are taken from the one-line docstrings of functions in `nqueens/fitness.py`.

**Parameter Tuning Results (if enabled):**

```text
n, fitness_function, pop_size, max_gen, mutation_rate, crossover_rate,
tournament_size, success_rate, avg_convergence_time, parameter_rank
```

### Data Integrity

**Validation Checks:**

- Solution correctness verification
- Parameter consistency validation
- Statistical significance testing
- Outlier detection and flagging

**Metadata Preservation:**

- Execution timestamps
- System configuration
- Random seed values
- Algorithm version information

## Implementation Details

### Code Organization

```text
algoanalisys.py                # Backwards-compatible facade and CLI entry point
nqueens/
├── backtracking.py            # BT solvers
├── simulated_annealing.py     # SA solver
├── genetic.py                 # GA solver
├── fitness.py                 # GA fitness functions
├── utils.py                   # conflicts(), is_valid_solution(), helpers
└── analysis/
    ├── settings.py            # global settings, timeouts, grids
    ├── stats.py               # typed stats and aggregations
    ├── tuning.py              # GA tuning (seq/parallel)
    ├── experiments.py         # experiment runners (seq/parallel)
    ├── reporting.py           # CSV exports
    ├── plots.py               # plotting (optional)
    └── cli.py                 # pipelines, CLI, quick regression
```

### Performance Optimizations

**Algorithmic Optimizations:**

- O(N) conflict calculation using Counter
- Efficient neighbor generation for SA
- Memory-efficient population management for GA
- Lazy evaluation for fitness calculations

**System Optimizations:**

- Multi-core parallel processing
- Memory pooling for large datasets
- Intelligent garbage collection
- CPU cache optimization

**I/O Optimizations:**

- Streaming CSV writing
- Asynchronous file operations
- Compressed intermediate storage
- Batch processing for large experiments

### Error Handling

**Robust Execution:**

- Timeout management with graceful degradation
- Memory overflow protection
- Process crash recovery
- Result validation and consistency checks

**Debugging Support:**

- Comprehensive logging system
- Progress monitoring and reporting
- Intermediate result checkpointing
- Performance profiling capabilities

## Usage Guidelines

### Quick Start

**Basic Execution:**

```bash
# Default: concurrent pipeline with all fitness functions
python algoanalisys.py

# Sequential processing (lower memory usage)
python algoanalisys.py --mode sequential

# Classic parallel mode
python algoanalisys.py --mode parallel

# Quick smoke test (N=8) and exit
python algoanalisys.py --quick-test

# Run selected fitness functions only
python algoanalisys.py --fitness F1,F3,F5

# Enable result/solution validation (extra assertions)
python algoanalisys.py --validate

# Use a custom configuration file (default is config.json next to algoanalisys.py)
python algoanalisys.py --config /path/to/config.json

# GA only (tuning + experiments)
python algoanalisys.py -a GA

# SA only / BT only
python algoanalisys.py -a SA
python algoanalisys.py -a BT
```

All runtime messages and plot labels are in English.

### Validation

The framework includes optional validation checks that you can enable with the `--validate` flag.

What it does:

- Backtracking (BT): when a board is produced, it is verified with `is_valid_solution(board)` to ensure no queens attack each other.
- Simulated Annealing (SA) and Genetic Algorithm (GA): for runs marked as successful, the results are checked to ensure `best_conflicts == 0` and `timeout == False`.

Notes:

- These checks add lightweight assertions meant for debugging and CI. For very large runs you may keep them off to minimize overhead.
- Plotting remains optional; validation does not require matplotlib.

Low-level API example:

```python
from nqueens import is_valid_solution, conflicts

board = [0, 4, 7, 5, 2, 6, 1, 3]
print(is_valid_solution(board))  # True
print(conflicts(board))          # 0
```

**Output Interpretation:**

- Success rates > 90% indicate excellent algorithm performance
- Timeout rates > 10% suggest parameter adjustment needed
- Execution times follow expected complexity patterns
- Statistical significance confirmed through confidence intervals

### Advanced Configuration

**Parameter Customization:**

```python
# Problem sizes to test
N_VALUES = [8, 16, 24, 40, 80, 120]

# Algorithm run counts
RUNS_SA_FINAL = 40    # SA independent runs
RUNS_GA_FINAL = 40    # GA independent runs  
RUNS_BT_FINAL = 1     # BT single run (deterministic)

# Timeout settings
SA_TIME_LIMIT = 30.0      # seconds per SA run
GA_TIME_LIMIT = 60.0      # seconds per GA run
EXPERIMENT_TIMEOUT = 120.0 # seconds per experiment
```

**Fitness Function Selection:**

```python
# Test specific fitness functions
FITNESS_MODES = ["F1", "F3", "F5"]  # subset testing

# Full comparison
FITNESS_MODES = ["F1", "F2", "F3", "F4", "F5", "F6"]  # complete analysis
```

### Performance Tuning

**System Configuration:**

```python
# CPU utilization
NUM_PROCESSES = multiprocessing.cpu_count() - 1  # leave one core for OS

# Memory management  
MEMORY_LIMIT = "8GB"    # maximum memory usage
BATCH_SIZE = 1000       # experiments per batch
```

**Algorithm Parameters:**

```python
# GA tuning ranges
POP_MULTIPLIERS = [4, 8, 16]      # population size factors
GEN_MULTIPLIERS = [30, 50, 80]    # generation count factors  
PM_VALUES = [0.05, 0.10, 0.15]    # mutation rate options
```

## Troubleshooting

### Common Issues

**Memory Problems:**

- **Symptom**: Process killed due to memory usage
- **Solution**: Reduce `RUNS_*_FINAL` values or use sequential mode
- **Prevention**: Monitor memory usage during large experiments

**Timeout Issues:**

- **Symptom**: High timeout rates in results
- **Solution**: Increase `*_TIME_LIMIT` values or reduce problem sizes
- **Prevention**: Profile algorithm performance before large runs

**Performance Problems:**

- **Symptom**: Very slow execution
- **Solution**: Reduce problem sizes or use fewer processes
- **Prevention**: Start with small test runs to estimate timing

### Debugging Techniques

**Logging Analysis:**

```bash
# Enable verbose logging (bash)
export ALGO_DEBUG=1
python algoanalisys.py

# Monitor progress (bash)
tail -f algo.log
```

**Performance Profiling:**

```bash
# Profile memory usage
python -m memory_profiler algo.py

# Profile CPU usage  
python -m cProfile -o profile.stats algo.py
```

## Planned Improvements

Implemented as of v2.1.0:

- CLI flag `--mode` to switch among `sequential`, `parallel`, and `concurrent`
- Fitness filtering via `--fitness`
- Configuration via `config.json` with persisted optimal parameters
- Quick regression tests on N=8 (`--quick-test`)
    (Il riuso dei parametri è il comportamento di default; usare `--tune` per eseguire tuning.)
- Graceful shutdown on `Ctrl+C` and progress reporting
- New `--validate` flag for solution/result consistency checks

Potential next steps:

- Optional HTML report bundling selected charts and CSVs
- Extended statistical plots when raw runs are persisted for all algorithms
- CI workflow for type checking and smoke tests

## Contributing

### Development Guidelines

**Code Standards:**

- PEP 8 compliance for Python code style
- Comprehensive English documentation  
- Type hints for function signatures
- Unit tests for core functionality

**Testing Requirements:**

```bash
# Run test suite
python -m pytest tests/

# Performance benchmarks
python -m pytest tests/performance/

# Statistical validation
python -m pytest tests/statistics/
```

**Documentation Standards:**

- English-only documentation and comments
- Mathematical notation using LaTeX where appropriate
- Algorithm complexity analysis required
- Performance benchmarking for optimizations

### Adding New Algorithms

**Integration Checklist:**

1. Implement algorithm following existing patterns
2. Add timeout support and time measurement
3. Include parameter tuning capabilities
4. Update statistical analysis framework
5. Add visualization support
6. Write comprehensive tests
7. Document performance characteristics

**Code Template:**

```python
def new_algorithm(N, param1, param2, time_limit=None):
    """
    New algorithm implementation
    
    Args:
        N: Problem size
        param1: Algorithm-specific parameter
        param2: Algorithm-specific parameter  
        time_limit: Maximum execution time in seconds
        
    Returns:
        Tuple: (success, solution, cost_metric, execution_time)
    """
    start_time = perf_counter()
    
    # Algorithm implementation
    # ...
    
    execution_time = perf_counter() - start_time
    return success, solution, cost_metric, execution_time
```

### Adding New Fitness Functions

**Implementation Steps:**

1. Define fitness calculation function
2. Add to `FITNESS_MODES` list  
3. Include in parameter tuning grid
4. Update comparative analysis charts
5. Add performance benchmarking
6. Document mathematical properties

Note:

- The first line of the fitness function docstring is used as the short description in the `--list` output. Keep it concise (one sentence).

**Fitness Function Template:**

```python
def fitness_fx(board):
    """
    Fitness function Fx description
    
    Args:
        board: List representing queen positions
        
    Returns:
        float: Fitness value (lower is better)
        
    Characteristics:
        - Objective: [describe optimization objective]
        - Properties: [mathematical properties]
        - Best for: [problem characteristics]
    """
    # Implementation
    return fitness_value
```

## References

### Academic Literature

- **Constraint Satisfaction**: Tsang, E. (1993). "Foundations of Constraint Satisfaction"

- **Metaheuristics**: Blum, C. & Roli, A. (2003). "Metaheuristics in Combinatorial Optimization"
- **Genetic Algorithms**: Goldberg, D.E. (1989). "Genetic Algorithms in Search, Optimization"
- **Simulated Annealing**: Kirkpatrick, S. et al. (1983). "Optimization by Simulated Annealing"

### Implementation References

- **Performance Analysis**: Experimental Algorithmics methodologies

- **Statistical Testing**: Scientific computing best practices
- **Visualization**: Tufte, E.R. principles of statistical graphics
- **Software Engineering**: Clean Code and SOLID principles

### Benchmarking Standards

- **N-Queens Benchmarks**: Standard problem instances and expected performance

- **Algorithm Comparison**: Fair evaluation methodologies
- **Statistical Significance**: Proper experimental design and analysis
- **Reproducibility**: Random seed management and version control

---

*This wiki provides comprehensive documentation for understanding, using, and extending the N-Queens comparative algorithm analysis framework.*
