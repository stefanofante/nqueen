# N-Queens Problem: Comprehensive Algorithm Analysis

This project provides a comprehensive comparative analysis of three fundamental algorithms for solving the N-Queens problem: **Backtracking**, **Simulated Annealing**, and **Genetic Algorithm**. The framework includes advanced statistical analysis, automated parameter tuning, and extensive visualization capabilities. The solver implementations live inside the `nqueens/` package. The orchestration, tuning, reporting, and plots have been modularized under `nqueens/analysis/`, while `algoanalisys.py` now acts as a thin backwards-compatible facade re-exporting the public APIs.

## Quick Usage

Run only specific algorithms via the `--alg` (`-a`) filter. Accepted values: `BT`, `SA`, `GA` (repeatable or comma-separated). Examples:

```bash
# GA only (tuning + experiments) [default: parallel mode]
python algoanalisys.py -a GA

# GA only, skip tuning and reuse parameters from config.json
python algoanalisys.py -a GA --skip-tuning --config config.json

# SA only (no GA, no BT)
python algoanalisys.py --mode sequential -a SA

# BT only
python algoanalisys.py -a BT

# SA + GA for selected fitness functions
python algoanalisys.py -a SA,GA -f F1,F3
```

Behavior notes:

- If `GA` is included and `--skip-tuning` is NOT set, the pipeline performs GA tuning first (according to the selected `--mode`) and then runs final experiments.
- If `GA` is excluded (e.g., `-a SA` or `-a BT`), tuning is automatically skipped.
- When tuning runs, optimal GA parameters are saved back to `config.json` (via `ConfigManager`).
- With `--skip-tuning`, GA parameters are loaded from `config.json`; missing sizes cause an explicit error suggesting to run tuning first.

All runtime messages and plot labels are in English.

## Overview

The N-Queens problem consists of placing N queens on an N x N chessboard such that no two queens can attack each other. Two queens attack each other if they are in the same row, column, or diagonal.

This is a classic constraint satisfaction problem used as a benchmark for:

- **Exact algorithms** (exhaustive search and backtracking)
- **Metaheuristic algorithms** (simulated annealing, genetic algorithms)  
- **Scalability analysis** of computational cost as N varies

## Key Features

### Algorithms Implemented

- **Backtracking (BT)**: Systematic depth-first search with constraint propagation
- **Simulated Annealing (SA)**: Metaheuristic optimization with temperature scheduling
- **Genetic Algorithm (GA)**: Population-based evolutionary computation with 6 fitness functions

### Advanced Capabilities

- **Parallel Processing**: Multi-core execution using `ProcessPoolExecutor`
- **Automated Parameter Tuning**: Systematic optimization of algorithm parameters
- **Advanced Statistical Analysis**: Success/failure/timeout categorization
- **Professional Visualization**: 60+ high-quality scientific charts
- **Timeout Management**: Configurable execution limits for scalability testing
- **Comprehensive Data Export**: CSV files with detailed results and raw data

### Genetic Algorithm Fitness Functions

- **F1**: Basic conflict counting
- **F2**: Weighted conflict penalty
- **F3**: Advanced constraint satisfaction  
- **F4**: Multi-objective optimization
- **F5**: Adaptive penalty system
- **F6**: Hybrid evaluation method

## Mathematical Definition

Given an integer N >= 4, the problem consists of finding a placement of N queens on an N x N chessboard such that:

1. **Row constraint**: at most one queen per row
2. **Column constraint**: at most one queen per column
3. **Main diagonal constraint**: at most one queen per diagonal with slope +1
4. **Anti-diagonal constraint**: at most one queen per diagonal with slope -1

### Solution Representation

The problem is represented using an array `board[N]` where `board[i]` indicates the row where the queen of column `i` is placed. This representation automatically guarantees the column constraint.

### Conflict Function

Solution quality is measured through the number of conflicting queen pairs:

```python
def conflicts(board):
    """Count conflicting queen pairs using counters for rows and diagonals"""
    n = len(board)
    row_count = Counter()
    diag1 = Counter()  # Diagonal r-c
    diag2 = Counter()  # Diagonal r+c
    
    # Count queens per row and diagonal
    for col, row in enumerate(board):
        row_count[row] += 1
        diag1[row - col] += 1
        diag2[row + col] += 1
    
    # Calculate conflicts as combinations C(k,2) for each group    
    return (sum(count * (count - 1) // 2 for count in row_count.values()) +
            sum(count * (count - 1) // 2 for count in diag1.values()) +
            sum(count * (count - 1) // 2 for count in diag2.values()))
```

A solution is valid when `conflicts(board) = 0`.

## Algorithm Implementations

### 1. Iterative Backtracking

```python
def bt_nqueens_first(N, time_limit=None):
    """
    Iterative backtracking implementation
    Returns: (solution_found, board, nodes_explored, execution_time)
    """
    start_time = perf_counter()
    nodes = 0
    stack = [(0, [-1] * N)]  # (column, partial_board)
    
    while stack:
        if time_limit and perf_counter() - start_time > time_limit:
            return False, [], nodes, perf_counter() - start_time
            
        col, board = stack.pop()
        nodes += 1
        
        if col == N:
            return True, board, nodes, perf_counter() - start_time
        
        for row in range(N):
            if is_safe(board, col, row):
                new_board = board[:]
                new_board[col] = row
                stack.append((col + 1, new_board))
    
    return False, [], nodes, perf_counter() - start_time
```

**Characteristics:**

- **Time Complexity**: O(N!) in worst case
- **Space Complexity**: O(N) for the stack
- **Deterministic**: Always finds the same solution
- **Complete**: Guaranteed to find a solution if one exists

### 2. Simulated Annealing

```python
def sa_nqueens(N, max_iter=10000, initial_temp=100, cooling_rate=0.95, time_limit=None):
    """
    Simulated Annealing implementation with geometric cooling
    Returns: (success, final_board, iterations, evaluations, execution_time)
    """
    start_time = perf_counter()
    
    # Random initial solution
    current = list(range(N))
    random.shuffle(current)
    current_conflicts = conflicts(current)
    
    temp = initial_temp
    best = current[:]
    best_conflicts = current_conflicts
    
    for iteration in range(max_iter):
        if time_limit and perf_counter() - start_time > time_limit:
            break
            
        if current_conflicts == 0:
            return True, current, iteration, iteration, perf_counter() - start_time
        
        # Generate neighbor by swapping two random queens
        neighbor = current[:]
        i, j = random.sample(range(N), 2)
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        
        neighbor_conflicts = conflicts(neighbor)
        delta = neighbor_conflicts - current_conflicts
        
        # Accept or reject move
        if delta <= 0 or random.random() < math.exp(-delta / temp):
            current = neighbor
            current_conflicts = neighbor_conflicts
            
            if current_conflicts < best_conflicts:
                best = current[:]
                best_conflicts = current_conflicts
        
        # Cool down temperature
        temp *= cooling_rate
    
    return best_conflicts == 0, best, max_iter, max_iter, perf_counter() - start_time
```

**Characteristics:**

- **Time Complexity**: O(max_iter x N)
- **Space Complexity**: O(N)
- **Probabilistic**: May find different solutions
- **Incomplete**: Not guaranteed to find optimal solution

### 3. Genetic Algorithm

```python
def ga_nqueens(N, pop_size=100, max_gen=500, pc=0.8, pm=0.1, 
               tournament_size=3, fitness_mode="F1", time_limit=None):
    """
    Genetic Algorithm with configurable fitness functions
    Returns: (success, best_individual, generations, evaluations, execution_time)
    """
    start_time = perf_counter()
    
    # Initialize random population
    population = [list(range(N)) for _ in range(pop_size)]
    for individual in population:
        random.shuffle(individual)
    
    evaluations = 0
    
    for generation in range(max_gen):
        if time_limit and perf_counter() - start_time > time_limit:
            break
        
        # Evaluate population
        fitness_values = []
        for individual in population:
            fitness = evaluate_fitness(individual, fitness_mode)
            fitness_values.append(fitness)
            evaluations += 1
            
            if fitness == 0:  # Found solution
                return True, individual, generation, evaluations, perf_counter() - start_time
        
        # Selection, crossover, mutation
        new_population = []
        
        for _ in range(pop_size):
            # Tournament selection
            parent1 = tournament_selection(population, fitness_values, tournament_size)
            parent2 = tournament_selection(population, fitness_values, tournament_size)
            
            # Crossover
            if random.random() < pc:
                child = order_crossover(parent1, parent2)
            else:
                child = parent1[:]
            
            # Mutation
            if random.random() < pm:
                child = swap_mutation(child)
            
            new_population.append(child)
        
        population = new_population
    
    # Return best individual found
    best_individual = min(population, key=lambda x: evaluate_fitness(x, fitness_mode))
    return False, best_individual, max_gen, evaluations, perf_counter() - start_time
```

**Characteristics:**

- **Time Complexity**: O(max_gen x pop_size x N)
- **Space Complexity**: O(pop_size x N)
- **Stochastic**: Uses randomization extensively
- **Scalable**: Performs well on large instances

## Installation

### Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`

### Setup

```bash
git clone https://github.com/stefanofante/nqueen.git
cd nqueen
pip install -r requirements.txt
```

### Dependencies

- Core: `numpy>=1.21.0`, `pandas>=1.3.0`
- Optional (for charts): `matplotlib>=3.5.0`

Note: Plotting is optional at runtime. If `matplotlib` is not installed, the CLI and APIs will still run; plotting functions are stubbed and will raise a clear runtime error only when called.

## Usage

### Quick Start

```bash
# Run complete analysis (parallel mode - default)
python algoanalisys.py

# Sequential pipeline
python algoanalisys.py --mode sequential

# Concurrent pipeline (across fitness functions)
python algoanalisys.py --mode concurrent

# Run quick smoke test (N=8) and exit
python algoanalisys.py --quick-test

# Run only selected fitness functions (comma-separated or multiple flags)
python algoanalisys.py --fitness F1,F3,F5
```

CLI flags overview:

- --mode {sequential|parallel|concurrent}
- --fitness, -f: filter fitness functions
- --skip-tuning: reuse parameters from config.json
- --config: configuration file path (default: config.json)
- --quick-test: run quick regression and exit
- --validate: validate solutions and assert consistency (extra checks)

Note: All runtime messages and plot labels are in English.

### Python API

You can also call the high-level APIs directly. For example, a quick smoke test:

```python
from algoanalisys import run_quick_regression_tests

run_quick_regression_tests()
```

The module `algoanalisys.py` re-exports the public functions from `nqueens.analysis.*` to preserve backwards compatibility.

Additionally, the core package exposes low-level helpers:

```python
from nqueens import is_valid_solution, conflicts

board = [0, 4, 7, 5, 2, 6, 1, 3]
assert is_valid_solution(board)  # True if no queens attack each other
print(conflicts(board))          # 0 for a valid solution
```

### Configuration

#### Problem Sizes

Default test dimensions: `[8, 16, 24, 40, 80, 120]`

#### Algorithm Parameters

- **SA Runs**: 40 independent executions
- **GA Runs**: 40 independent executions
- **BT Runs**: 1 (deterministic algorithm)
- **Tuning Runs**: 5 per parameter combination

#### Timeout Settings

```python
BT_TIME_LIMIT = None        # No limit (very fast)
SA_TIME_LIMIT = 30.0        # 30 seconds per run
GA_TIME_LIMIT = 60.0        # 60 seconds per run
EXPERIMENT_TIMEOUT = 120.0  # 2 minutes total per experiment
```

## Public API contracts (solvers)

The orchestrator relies on the following stable contracts:

- Backtracking solvers: functions named `bt_nqueens_*` in `nqueens.backtracking`.
  - Signature: `(N: int, time_limit: Optional[float]) -> Tuple[Optional[List[int]], int, float]`
  - Returns: (solution or None, explored_nodes, elapsed_seconds)

- Simulated Annealing: `sa_nqueens` in `nqueens.simulated_annealing`.
  - Signature: `(N: int, max_iter: int, T0: float, alpha: float, time_limit: Optional[float]) -> Tuple[bool, int, float, int, int, bool]`
  - Returns: (success, steps_or_gen, elapsed_seconds, best_conflicts, evaluations, timeout)

- Genetic Algorithm: `ga_nqueens` in `nqueens.genetic`.
  - Signature: `(N: int, pop_size: int, max_gen: int, pc: float, pm: float, tournament_size: int, fitness_mode: str, time_limit: Optional[float]) -> Tuple[bool, int, float, int, int, bool]`
  - Returns: (success, steps_or_gen, elapsed_seconds, best_conflicts, evaluations, timeout)

The orchestrator discovers backtracking solvers dynamically by the `bt_nqueens_*` prefix and validates everything via the quick regression runner.

## Output Structure

### Generated Files

#### CSV Data Files

```text
results_nqueens_tuning/
├── results_GA_F1_tuned.csv      # GA F1 aggregated results
├── results_GA_F2_tuned.csv      # GA F2 aggregated results
├── ...                          # F3-F6 results
├── tuning_GA_F1.csv             # GA F1 parameter tuning data
├── tuning_GA_F2.csv             # GA F2 parameter tuning data
└── ...                          # F3-F6 tuning data
```

### CSV Schema

The CSV exports use lowercase snake_case column names. Aggregated metrics use subsystem prefixes: `bt_` (Backtracking), `sa_` (Simulated Annealing), `ga_` (Genetic Algorithm). Time fields are in seconds.

#### Aggregated Results (`results_GA_<FITNESS>_tuned.csv`)

```text
n,
bt_solution_found, bt_nodes_explored, bt_time_seconds,
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

#### Raw Data — Simulated Annealing (`raw_data_SA_<FITNESS>.csv`)

```text
n, run_id, algorithm, success, timeout, steps, time_seconds, evals, best_conflicts
```

#### Raw Data — Genetic Algorithm (`raw_data_GA_<FITNESS>.csv`)

```text
n, run_id, algorithm, success, timeout, gen, time_seconds, evals, best_fitness, best_conflicts,
pop_size, max_gen, pm, pc, tournament_size
```

#### Raw Data — Backtracking (`raw_data_BT_<FITNESS>.csv`)

```text
n, algorithm, solution_found, nodes_explored, time_seconds
```

#### Logical Cost Analysis (`logical_costs_<FITNESS>.csv`)

```text
n,
bt_solution_found, bt_nodes_explored,
sa_success_rate, sa_steps_mean_all, sa_steps_median_all, sa_evals_mean_all, sa_evals_median_all,
sa_steps_mean_success, sa_evals_mean_success,
ga_success_rate, ga_gen_mean_all, ga_gen_median_all, ga_evals_mean_all, ga_evals_median_all,
ga_gen_mean_success, ga_evals_mean_success,
bt_time_seconds, sa_time_mean_success, ga_time_mean_success
```

Backward compatibility: if you were consuming previous CSVs with different headers, update your schemas to the names above.

#### Visualization Output

```text
results_nqueens_tuning/
├── analysis_F1/                 # 9 detailed charts for F1
├── analysis_F2/                 # 9 detailed charts for F2  
├── ...                          # F3-F6 analysis
├── fitness_comparison/          # comparative charts
├── statistical_analysis/        # optional statistical charts
└── tuning_GA_F*.csv             # tuning exports per fitness
```

### Chart Categories

#### 1. Base Analysis (9 charts per fitness)

- Success rate vs problem size
- Execution time vs problem size (log scale)
- Logical cost vs problem size
- Fitness evaluations vs problem size
- Timeout rate analysis
- Failure quality assessment
- Theoretical vs practical correlation
- Algorithm stability analysis
- Performance scalability

#### 2. Fitness Function Comparison (6 charts)

- Success rate comparison (bar charts)
- Convergence speed analysis  
- Execution time trade-offs
- Pareto efficiency analysis
- Cross-fitness performance evolution
- Multi-dimensional quality assessment

#### 3. Statistical Analysis (12+ charts)

- Box plots for execution time distribution
- Box plots for iteration/generation counts
- Histogram distributions for algorithm stability
- Variance analysis across problem sizes
- Outlier detection and analysis
- Confidence interval visualization

#### 4. Parameter Tuning Analysis (10+ charts)

- Heatmaps for parameter optimization
- Cost vs quality scatter plots
- Mutation rate sensitivity analysis
- Population size impact assessment
- Generation limit optimization
- Parameter interaction analysis

## Performance Analysis

### Expected Results

#### Backtracking

- **Strengths**: Guaranteed optimal solution, deterministic behavior
- **Limitations**: Exponential time complexity, poor scalability
- **Best Performance**: Small problems (N ≤ 16)
- **Typical Behavior**: Very fast for N ≤ 12, exponential slowdown after N = 16

#### Simulated Annealing  

- **Strengths**: Good balance of quality and speed, robust performance
- **Limitations**: Parameter sensitive, probabilistic results
- **Best Performance**: Medium problems (N ≤ 40)
- **Typical Behavior**: Consistent performance, moderate resource usage

#### Genetic Algorithm

- **Strengths**: Highly scalable, fitness function flexibility, population diversity
- **Limitations**: Parameter intensive, population overhead, convergence uncertainty
- **Best Performance**: Large problems (N >= 24)
- **Typical Behavior**: Excellent scalability, fitness function dependent

### Performance Metrics

#### Success Indicators

- **Success Rate**: Percentage of runs finding optimal solution (0 conflicts)
- **Convergence Speed**: Average iterations/generations to solution
- **Execution Time**: Wall-clock time for successful runs
- **Resource Usage**: Logical cost (nodes/iterations/evaluations)

#### Quality Indicators

- **Timeout Rate**: Percentage of runs exceeding time limits
- **Failure Quality**: Best solution quality in unsuccessful attempts
- **Algorithm Stability**: Variance in performance metrics across runs
- **Scalability**: Performance degradation with increasing problem size

## Technical Architecture

### Modular Architecture

The orchestration layer is organized as a modular package under `nqueens/analysis/`:

- `settings.py`: global settings, timeouts, tuning grids, output paths
- `stats.py`: typed result records and statistical aggregations
- `tuning.py`: GA parameter search (sequential and parallel)
- `experiments.py`: experiment runners for BT/SA/GA with optimal parameters
- `reporting.py`: CSV exports for aggregated and raw data, logical cost analysis
- `plots.py`: chart generation (optional, requires matplotlib)
- `cli.py`: pipelines, CLI wiring, and a quick regression runner

The legacy `algoanalisys.py` file is now a thin facade that imports and re-exports these public APIs to avoid breaking existing imports and tests.

### Core Components

#### Solver Package (`nqueens/`)

- Dedicated modules for backtracking, simulated annealing, and genetic algorithm
- Shared fitness utilities and conflict counters centralized in the package
- Keeps solver logic independent from orchestration, plotting, and export layers

#### Statistical Engine

- Advanced descriptive statistics calculation
- Success/failure/timeout categorization
- Confidence interval computation
- Distribution analysis and outlier detection

#### Parallel Processing Framework

- Multi-core algorithm execution using `ProcessPoolExecutor`
- Concurrent parameter tuning across fitness functions
- Intelligent process pool management
- Memory-efficient resource allocation

#### Visualization Engine

- Matplotlib-based scientific chart generation
- Seaborn statistical plots and distributions
- High-resolution output (300 DPI)
- Professional styling with comprehensive labeling

#### Data Export System

- Structured CSV format for external analysis
- Raw experimental data preservation
- Aggregated statistics with confidence intervals
- Parameter optimization results and recommendations

## Contributing

### Development Setup

```bash
git clone https://github.com/stefanofante/nqueen.git
cd nqueen
python -m venv .venv
# On Windows (PowerShell)
.venv\Scripts\Activate.ps1
# On Linux/macOS
source .venv/bin/activate
pip install -r requirements.txt
```

### Code Style

- PEP 8 compliance
- Comprehensive English documentation
- Type hints where applicable
- Unit tests for core functions

### Adding New Algorithms

1. Implement algorithm function following existing patterns
2. Add parameter tuning capabilities  
3. Include timeout support
4. Update statistical analysis framework
5. Add visualization support

### Adding New Fitness Functions

1. Define fitness calculation function
2. Add to `FITNESS_MODES` list
3. Include in parameter tuning grid
4. Update comparative analysis charts

### Planned Improvements

- Add optional static typing enforcement via mypy (mypy.ini included)
- Additional statistical plots (violin plots, CI bands) when raw runs are persisted
- Lightweight HTML report bundling selected charts and CSVs

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{nqueens_analysis,
  author = {Stefano Fante},
  title = {N-Queens Problem: Comprehensive Algorithm Analysis},
  url = {https://github.com/stefanofante/nqueen},
  year = {2025}
}
```

## Acknowledgments

- Algorithm implementations based on classical optimization literature
- Statistical analysis inspired by experimental algorithmics best practices
- Visualization design following scientific plotting guidelines  
- Performance benchmarking using industry-standard methodologies

## Support

For questions, issues, or contributions:

- **Issues**: [GitHub Issues](https://github.com/stefanofante/nqueen/issues)
- **Discussions**: [GitHub Discussions](https://github.com/stefanofante/nqueen/discussions)
- **Email**: <stefano.fante@example.com>

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for detailed version history and updates.
