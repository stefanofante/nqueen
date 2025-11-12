# N-Queens Algorithm Comparison: Advanced Analysis Wiki

This wiki provides comprehensive documentation for the N-Queens comparative algorithm analysis project, featuring Backtracking (BT), Simulated Annealing (SA), and Genetic Algorithm (GA) implementations. Solver code now lives inside the `nqueens/` package, while `algo.py` acts as the orchestrator for tuning, experimentation, and reporting.

## Overview

This project compares three fundamental approaches to solving the N-Queens problem:

* **BT** (Backtracking): Iterative implementation without recursion
* **SA** (Simulated Annealing): Metaheuristic optimization with temperature control  
* **GA** (Genetic Algorithm): Evolutionary computation with **fitness functions F1-F6**

## Objectives

* Study the impact of **problem size N** on success rates and computational cost
* Evaluate **alternative fitness functions** for genetic algorithms
* Define a **systematic parameter tuning procedure** for GA across different problem sizes
* Achieve **performance optimizations** (O(N) conflicts calculation, multiprocessing, optional Numba acceleration)

## Project Structure

* `algo.py`: orchestration layer handling tuning pipelines, experiment scheduling, chart generation, and CSV export
* `nqueens/backtracking.py`: iterative backtracking solver
* `nqueens/simulated_annealing.py`: simulated annealing solver
* `nqueens/genetic.py`: genetic algorithm driver
* `nqueens/fitness.py`: shared GA fitness functions
* `nqueens/utils.py`: conflict counters and shared utilities

## Mathematical Foundation

### Problem Definition

The N-Queens problem requires placing N queens on an N x N chessboard such that no two queens attack each other.

**Constraints:**

* Row constraint: ≤ 1 queen per row
* Column constraint: ≤ 1 queen per column  
* Main diagonal constraint: ≤ 1 queen per diagonal (slope +1)
* Anti-diagonal constraint: ≤ 1 queen per diagonal (slope -1)

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

* Time complexity: O(N) instead of naive O(N²)
* Uses combinatorics: conflicts = C(k,2) for each group of k queens
* Memory efficient with Counter data structures

## Algorithm Implementations

### 1. Backtracking (BT)

**Implementation Strategy:**

* Iterative depth-first search using explicit stack
* Avoids recursion overhead and stack overflow issues
* Deterministic: always finds the lexicographically first solution

**Key Features:**

* **Completeness**: Guaranteed to find solution if one exists
* **Optimality**: Finds optimal (valid) solution
* **Determinism**: Reproducible results
* **Time Complexity**: O(N!) worst case, often much better with pruning

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

* Start with random permutation solution
* Iteratively improve using neighbor generation and probabilistic acceptance
* Uses geometric cooling schedule for temperature control

**Key Features:**

* **Metaheuristic**: Probabilistic local search
* **Escape Mechanism**: Can accept worse solutions to escape local optima
* **Parameter Sensitivity**: Performance depends on cooling schedule and temperature
* **Time Complexity**: O(max_iterations x N)

**Neighbor Generation:**

* **Swap Move**: Exchange positions of two randomly selected queens
* Maintains permutation property automatically
* Efficient O(1) neighbor generation

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

* Population-based evolutionary optimization
* Multiple fitness functions (F1-F6) for comparison
* Tournament selection, order crossover, swap mutation

**Key Features:**

* **Population Diversity**: Maintains multiple candidate solutions
* **Scalability**: Excellent performance on large instances
* **Fitness Flexibility**: Six different evaluation functions
* **Parameter Rich**: Many tunable parameters for optimization

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

* **Objective**: Minimize total conflicts

* **Characteristics**: Direct, simple, baseline approach
* **Performance**: Good general performance

### F2: Weighted Conflict Penalty

```python
def fitness_f2(board):
    """Weighted conflicts with exponential penalty"""
    base_conflicts = conflicts(board)
    return base_conflicts + (base_conflicts ** 1.5)
```

* **Objective**: Exponential penalty for higher conflicts

* **Characteristics**: Aggressive conflict reduction
* **Performance**: Fast convergence on small instances

### F3: Advanced Constraint Satisfaction

```python
def fitness_f3(board):
    """Separate penalty for each constraint type"""
    row_conflicts = calculate_row_conflicts(board)
    diag_conflicts = calculate_diagonal_conflicts(board)
    return 2 * row_conflicts + 3 * diag_conflicts
```

* **Objective**: Differentiated penalty by constraint type

* **Characteristics**: Targeted constraint handling
* **Performance**: Effective on structured problems

### F4: Multi-objective Optimization

```python
def fitness_f4(board):
    """Balance conflict minimization with solution quality"""
    base_conflicts = conflicts(board)
    diversity_bonus = calculate_diversity(board)
    return base_conflicts - 0.1 * diversity_bonus
```

* **Objective**: Minimize conflicts while maintaining diversity

* **Characteristics**: Prevents premature convergence
* **Performance**: Robust across problem sizes

### F5: Adaptive Penalty System

```python
def fitness_f5(board):
    """Dynamic penalty adaptation based on problem size"""
    N = len(board)
    base_conflicts = conflicts(board)
    scale_factor = math.log(N + 1)
    return base_conflicts * scale_factor
```

* **Objective**: Adaptive scaling for different problem sizes

* **Characteristics**: Size-aware optimization
* **Performance**: Consistent across scale variations

### F6: Hybrid Evaluation Method

```python
def fitness_f6(board):
    """Combination of multiple evaluation criteria"""
    conflict_score = conflicts(board)
    constraint_score = evaluate_constraint_satisfaction(board)
    efficiency_score = calculate_search_efficiency(board)
    return 0.6 * conflict_score + 0.3 * constraint_score + 0.1 * efficiency_score
```

* **Objective**: Multi-criteria evaluation with weighted combination

* **Characteristics**: Comprehensive solution assessment
* **Performance**: Best overall performance on complex instances

## Parameter Tuning Framework

### Tuning Strategy

**Objective**: Find optimal GA parameters for each problem size N and fitness function.

**Parameters Optimized:**

* **Population Size**: `pop_size = multiplier x N` where multiplier in {4, 8, 16}
* **Maximum Generations**: `max_gen = multiplier x N` where multiplier in {30, 50, 80}
* **Mutation Rate**: `pm in {0.05, 0.10, 0.15}`
* **Crossover Rate**: `pc = 0.8` (fixed)
* **Tournament Size**: `tournament_size = 3` (fixed)

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

* **Backtracking**: Optimal choice, very fast execution
* **SA/GA**: Overkill, but useful for algorithm comparison

**Medium Instances (16 < N ≤ 40):**

* **Backtracking**: Becomes exponentially slow
* **Simulated Annealing**: Sweet spot for performance
* **Genetic Algorithm**: Good performance with proper tuning

**Large Instances (N > 40):**

* **Backtracking**: Impractical due to exponential time
* **Simulated Annealing**: Moderate performance
* **Genetic Algorithm**: Best choice, excellent scalability

### Expected Performance Characteristics

#### Success Rate vs Problem Size

* **BT**: 100% success rate for small N, impractical for large N

* **SA**: High success rate for medium N, degrades gracefully  
* **GA**: Fitness-dependent, generally excellent for large N

#### Execution Time Scaling

* **BT**: Exponential growth, fast for N ≤ 12

* **SA**: Near-linear growth, consistent performance
* **GA**: Controlled by population and generation parameters

#### Resource Usage

* **BT**: Minimal memory, CPU-intensive for large N

* **SA**: Low memory, moderate CPU usage
* **GA**: Memory scales with population size

## Statistical Analysis Framework

### Metrics Collected

**Primary Metrics:**

* **Success Rate**: Percentage of runs finding optimal solution
* **Execution Time**: Wall-clock time for successful runs
* **Algorithmic Cost**: Nodes explored (BT), iterations (SA), generations (GA)
* **Solution Quality**: Conflicts in best solution found

**Advanced Metrics:**

* **Timeout Rate**: Percentage of runs exceeding time limits
* **Convergence Speed**: Average time/iterations to solution
* **Algorithm Stability**: Variance in performance across runs
* **Failure Quality**: Best solution quality in unsuccessful attempts

### Statistical Tests

**Descriptive Statistics:**

* Mean, median, standard deviation
* Quartiles and percentiles
* Confidence intervals (95%)

**Distribution Analysis:**

* Normality testing (Shapiro-Wilk)
* Outlier detection (IQR method)
* Histogram visualization

**Comparative Analysis:**

* Algorithm performance ranking
* Pairwise comparisons
* Effect size calculation

## Visualization Framework

### Chart Categories

**1. Performance Overview (9 charts per fitness)**

* Success rate vs problem size
* Execution time vs problem size (log scale)
* Logical cost vs problem size  
* Fitness evaluations comparison
* Timeout analysis
* Failure quality assessment
* Theoretical vs practical correlation
* Algorithm stability metrics
* Scalability visualization

**2. Fitness Function Comparison (6 charts)**

* Success rate comparison across fitness functions
* Convergence speed analysis
* Execution time trade-offs
* Pareto efficiency frontier
* Multi-dimensional performance
* Cross-fitness scalability

**3. Statistical Analysis (12+ charts)**

* Box plots for time distribution
* Histogram distributions for stability
* Violin plots for detailed distributions
* Correlation analysis heatmaps
* Confidence interval visualizations
* Outlier analysis charts

**4. Parameter Tuning Analysis (10+ charts)**

* Heatmaps for parameter optimization
* 3D surface plots for parameter interactions
* Cost vs quality scatter plots
* Sensitivity analysis for mutation rate
* Population size impact assessment
* Generation limit optimization curves

### Chart Styling

**Professional Standards:**

* High resolution (300 DPI) output
* Scientific color schemes
* Clear axis labeling with units
* Comprehensive legends
* Grid lines for readability
* Statistical annotations (p-values, confidence intervals)

**Technical Features:**

* Matplotlib backend for precision
* Seaborn integration for statistical plots
* Custom styling for consistency
* LaTeX rendering for mathematical expressions
* Multi-format export (PNG, PDF, SVG)

## Data Export System

### CSV File Structure

**Aggregated Results:**

```
N, algorithm, success_rate, avg_time, std_time, avg_cost, std_cost, 
timeout_rate, best_quality_failures, confidence_interval_low, confidence_interval_high
```

**Raw Experimental Data:**

```
N, algorithm, run_id, success, time, cost, solution_quality, 
timeout_occurred, fitness_evaluations, convergence_generation
```

**Parameter Tuning Results:**

```
N, fitness_function, pop_size, max_gen, mutation_rate, crossover_rate,
tournament_size, success_rate, avg_convergence_time, parameter_rank
```

**Logical Cost Analysis:**

```
N, algorithm, theoretical_complexity, measured_operations, efficiency_ratio,
scaling_factor, projected_performance, resource_utilization
```

### Data Integrity

**Validation Checks:**

* Solution correctness verification
* Parameter consistency validation
* Statistical significance testing
* Outlier detection and flagging

**Metadata Preservation:**

* Execution timestamps
* System configuration
* Random seed values
* Algorithm version information

## Implementation Details

### Code Organization

```
algo.py
├── Statistical Functions (lines 16-120)
├── Global Parameters (lines 121-190)
├── Utility Functions (lines 191-255)
├── Algorithm Implementations (lines 256-720)
├── Tuning Framework (lines 721-1400)
├── Experiment Management (lines 1401-2000)
├── Visualization Engine (lines 2001-2800)
└── Main Execution (lines 2801-3000)
```

### Performance Optimizations

**Algorithmic Optimizations:**

* O(N) conflict calculation using Counter
* Efficient neighbor generation for SA
* Memory-efficient population management for GA
* Lazy evaluation for fitness calculations

**System Optimizations:**

* Multi-core parallel processing
* Memory pooling for large datasets
* Intelligent garbage collection
* CPU cache optimization

**I/O Optimizations:**

* Streaming CSV writing
* Asynchronous file operations
* Compressed intermediate storage
* Batch processing for large experiments

### Error Handling

**Robust Execution:**

* Timeout management with graceful degradation
* Memory overflow protection
* Process crash recovery
* Result validation and consistency checks

**Debugging Support:**

* Comprehensive logging system
* Progress monitoring and reporting
* Intermediate result checkpointing
* Performance profiling capabilities

## Usage Guidelines

### Quick Start

**Basic Execution:**

```bash
# Default mode: concurrent tuning with all fitness functions
python algo.py

# Sequential processing (lower memory usage)  
python algo.py --sequential

# Classic parallel mode (legacy compatibility)
python algo.py --parallel
```

**Output Interpretation:**

* Success rates > 90% indicate excellent algorithm performance
* Timeout rates > 10% suggest parameter adjustment needed
* Execution times follow expected complexity patterns
* Statistical significance confirmed through confidence intervals

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

* **Symptom**: Process killed due to memory usage
* **Solution**: Reduce `RUNS_*_FINAL` values or use sequential mode
* **Prevention**: Monitor memory usage during large experiments

**Timeout Issues:**

* **Symptom**: High timeout rates in results
* **Solution**: Increase `*_TIME_LIMIT` values or reduce problem sizes
* **Prevention**: Profile algorithm performance before large runs

**Performance Problems:**

* **Symptom**: Very slow execution
* **Solution**: Reduce problem sizes or use fewer processes
* **Prevention**: Start with small test runs to estimate timing

### Debugging Techniques

**Logging Analysis:**

```bash
# Enable verbose logging
export ALGO_DEBUG=1
python algo.py

# Monitor progress
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

* Add a CLI flag that switches among `sequential`, `parallel`, and `concurrent` modes and selects which fitness functions run
* Load tuning parameters and board sizes from a JSON configuration file, refreshing it after successful tuning runs
* Provide quick regression tests on a small instance (for example N = 8) to validate the three solvers and the CSV export pipeline
* Expose a flag to skip tuning when trusted parameters already exist
* Guarantee clean shutdown on `Ctrl+C` by terminating all child processes
* Display progress information so long experiments share their current stage

## Contributing

### Development Guidelines

**Code Standards:**

* PEP 8 compliance for Python code style
* Comprehensive English documentation  
* Type hints for function signatures
* Unit tests for core functionality

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

* English-only documentation and comments
* Mathematical notation using LaTeX where appropriate
* Algorithm complexity analysis required
* Performance benchmarking for optimizations

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

* **Constraint Satisfaction**: Tsang, E. (1993). "Foundations of Constraint Satisfaction"

* **Metaheuristics**: Blum, C. & Roli, A. (2003). "Metaheuristics in Combinatorial Optimization"
* **Genetic Algorithms**: Goldberg, D.E. (1989). "Genetic Algorithms in Search, Optimization"
* **Simulated Annealing**: Kirkpatrick, S. et al. (1983). "Optimization by Simulated Annealing"

### Implementation References

* **Performance Analysis**: Experimental Algorithmics methodologies

* **Statistical Testing**: Scientific computing best practices
* **Visualization**: Tufte, E.R. principles of statistical graphics
* **Software Engineering**: Clean Code and SOLID principles

### Benchmarking Standards

* **N-Queens Benchmarks**: Standard problem instances and expected performance

* **Algorithm Comparison**: Fair evaluation methodologies
* **Statistical Significance**: Proper experimental design and analysis
* **Reproducibility**: Random seed management and version control

---

*This wiki provides comprehensive documentation for understanding, using, and extending the N-Queens comparative algorithm analysis framework.*
