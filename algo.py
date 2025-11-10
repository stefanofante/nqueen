import os
import csv
import random
import math
import time
import statistics
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import matplotlib.pyplot as plt

# ======================================================
# PARAMETRI GLOBALI
# ======================================================

# Dimensioni della scacchiera da testare
N_VALUES = [8, 16, 24]

# Numero di run indipendenti per SA e GA (esperimenti finali)
RUNS_SA_FINAL = 20
RUNS_GA_FINAL = 20

# Numero di run per la fase di tuning GA (per combinazione di parametri)
RUNS_GA_TUNING = 5

# Limite di tempo per BT (None = nessun limite)
BT_TIME_LIMIT = None  # es. 5.0 secondi

# Directory di output per CSV e grafici
OUT_DIR = "results_nqueens_tuning"

# Griglia di tuning per il GA
POP_MULTIPLIERS = [4, 8, 16]       # pop_size ≈ 4N, 8N, 16N
GEN_MULTIPLIERS = [30, 50, 80]     # max_gen ≈ 30N, 50N, 80N
RUNS_GA_TUNING = 5                 # per non morire di tempi
PM_VALUES = [0.05, 0.1, 0.15]        # probabilità mutazione
PC_FIXED = 0.8
TOURNAMENT_SIZE_FIXED = 3

FITNESS_MODES = ["F1",  "F3", "F4", "F5", "F6"]

# Numero di processi per il parallelismo
NUM_PROCESSES = multiprocessing.cpu_count() - 1  # Lascia un core libero


# ======================================================
# 1. Utility comuni
# ======================================================
from collections import Counter

def conflicts(board):
    """
    Versione O(N): conta le coppie di regine in conflitto usando
    righe e diagonali.
    board[col] = row.
    """
    n = len(board)
    row_count = Counter()
    diag1 = Counter()
    diag2 = Counter()

    for c, r in enumerate(board):
        row_count[r] += 1
        diag1[r - c] += 1
        diag2[r + c] += 1

    def pairs(counter):
        tot = 0
        for cnt in counter.values():
            if cnt > 1:
                tot += cnt * (cnt - 1) // 2
        return tot

    row_conf = pairs(row_count)
    d1_conf  = pairs(diag1)
    d2_conf  = pairs(diag2)

    return row_conf + d1_conf + d2_conf

def conflicts_on2(board):
    """
    Conta il numero di coppie di regine in conflitto (stessa riga o diagonale).
    board[i] = riga della regina nella colonna i.
    """
    n = len(board)
    c = 0
    for i in range(n):
        for j in range(i + 1, n):
            if board[i] == board[j] or abs(board[i] - board[j]) == abs(i - j):
                c += 1
    return c


# ======================================================
# 2. Funzioni di fitness F1 ... F6 per il GA
# ======================================================

def fitness_f1(ind):
    """F1: fitness = -conflitti."""
    return -conflicts(ind)


def fitness_f2(ind):
    """F2: numero di coppie NON in conflitto."""
    n = len(ind)
    max_pairs = n * (n - 1) // 2
    c = conflicts(ind)
    return max_pairs - c


def fitness_f3(ind):
    """
    F3: penalità su cluster di regine sulle stesse diagonali
    (penalità lineare C(cnt,2)).
    """
    n = len(ind)
    diag1 = Counter()
    diag2 = Counter()
    for c, r in enumerate(ind):
        diag1[r - c] += 1
        diag2[r + c] += 1

    penalty = 0
    for cnt in diag1.values():
        if cnt > 1:
            penalty += cnt * (cnt - 1) // 2
    for cnt in diag2.values():
        if cnt > 1:
            penalty += cnt * (cnt - 1) // 2

    max_pairs = n * (n - 1) // 2
    return max_pairs - penalty


def fitness_f4(ind):
    """
    F4: F2 - conflitti della regina peggiore.
    """
    n = len(ind)
    max_pairs = n * (n - 1) // 2
    total_conf = conflicts(ind)
    base = max_pairs - total_conf

    max_conf_for_queen = 0
    for c in range(n):
        conf_q = 0
        for j in range(n):
            if j == c:
                continue
            if ind[j] == ind[c] or abs(ind[j] - ind[c]) == abs(j - c):
                conf_q += 1
        if conf_q > max_conf_for_queen:
            max_conf_for_queen = conf_q

    return base - max_conf_for_queen


def fitness_f5(ind):
    """
    F5: penalità QUADRATICA per cluster sulle diagonali.
    """
    n = len(ind)
    diag1 = Counter()
    diag2 = Counter()
    for c, r in enumerate(ind):
        diag1[r - c] += 1
        diag2[r + c] += 1

    penalty = 0
    for cnt in diag1.values():
        if cnt > 1:
            penalty += cnt ** 2
    for cnt in diag2.values():
        if cnt > 1:
            penalty += cnt ** 2

    max_pairs = n * (n - 1) // 2
    return max_pairs - penalty


def fitness_f6(ind, lam=0.3):
    """
    F6: trasformazione esponenziale dei conflitti.
    fitness = exp(-lam * conflicts)
    """
    c = conflicts(ind)
    return math.exp(-lam * c)


def get_fitness_function(mode):
    """Ritorna la funzione di fitness corrispondente a mode."""
    if mode == "F1":
        return fitness_f1
    elif mode == "F2":
        return fitness_f2
    elif mode == "F3":
        return fitness_f3
    elif mode == "F4":
        return fitness_f4
    elif mode == "F5":
        return fitness_f5
    elif mode == "F6":
        return lambda ind: fitness_f6(ind, lam=0.3)
    else:
        raise ValueError(f"fitness_mode sconosciuto: {mode}")


# ======================================================
# 3. Backtracking iterativo (prima soluzione)
# ======================================================

def bt_nqueens_first(N, time_limit=None):
    """
    Backtracking iterativo:
    - trova UNA sola soluzione
    - conta i tentativi di posizionamento (nodes)
    """
    pos = [-1] * N
    row_used = [False] * N
    diag1_used = [False] * (2 * N - 1)
    diag2_used = [False] * (2 * N - 1)

    col = 0
    row = 0
    nodes = 0
    start = time.time()

    while col >= 0 and col < N:
        if time_limit is not None and (time.time() - start) > time_limit:
            return None, nodes, time.time() - start

        placed = False
        while row < N and not placed:
            nodes += 1
            if not row_used[row]:
                d1 = row - col + (N - 1)
                d2 = row + col
                if not diag1_used[d1] and not diag2_used[d2]:
                    pos[col] = row
                    row_used[row] = True
                    diag1_used[d1] = True
                    diag2_used[d2] = True
                    placed = True

                    if col == N - 1:
                        return pos.copy(), nodes, time.time() - start
                    else:
                        col += 1
                        row = 0
                else:
                    row += 1
            else:
                row += 1

        if not placed:
            col -= 1
            if col >= 0:
                prev_row = pos[col]
                pos[col] = -1
                row_used[prev_row] = False
                d1 = prev_row - col + (N - 1)
                d2 = prev_row + col
                diag1_used[d1] = False
                diag2_used[d2] = False
                row = prev_row + 1

    return None, nodes, time.time() - start


# ======================================================
# 4. Simulated Annealing (SA)
# ======================================================

def sa_nqueens(N, max_iter=20000, T0=1.0, alpha=0.995):
    """
    Simulated Annealing per N-Queens.
    Restituisce:
      success (bool),
      iterazioni,
      tempo,
      best_conflicts,
      fitness_evals
    """
    board = [random.randrange(N) for _ in range(N)]
    cur_cost = conflicts(board)
    best_cost = cur_cost
    fitness_evals = 1
    start = time.time()

    if cur_cost == 0:
        return True, 0, time.time() - start, 0, fitness_evals

    T = T0
    for it in range(1, max_iter + 1):
        c = random.randrange(N)
        old_row = board[c]
        new_row = random.randrange(N)
        while new_row == old_row:
            new_row = random.randrange(N)
        board[c] = new_row

        new_cost = conflicts(board)
        fitness_evals += 1
        delta = new_cost - cur_cost

        if delta <= 0 or random.random() < math.exp(-delta / T):
            cur_cost = new_cost
            if cur_cost < best_cost:
                best_cost = cur_cost
        else:
            board[c] = old_row

        if cur_cost == 0:
            return True, it, time.time() - start, 0, fitness_evals

        T *= alpha

    return False, max_iter, time.time() - start, best_cost, fitness_evals


# ======================================================
# 5. Algoritmo Genetico (GA) base
# ======================================================

def ga_nqueens(
    N,
    pop_size=100,
    max_gen=1000,
    pc=0.8,
    pm=0.1,
    tournament_size=3,
    fitness_mode="F1",
):
    """
    Algoritmo genetico per N-Queens.

    Restituisce:
      success (bool),
      generazioni,
      tempo,
      best_conflicts,
      fitness_evals
    """
    fit_fn = get_fitness_function(fitness_mode)

    # popolazione iniziale
    pop = [[random.randrange(N) for _ in range(N)] for _ in range(pop_size)]
    fitness = [fit_fn(ind) for ind in pop]
    fitness_evals = pop_size

    best_idx = max(range(pop_size), key=lambda i: fitness[i])
    best_ind = pop[best_idx][:]
    best_conf = conflicts(best_ind)
    start = time.time()

    if best_conf == 0:
        return True, 0, time.time() - start, 0, fitness_evals

    def tournament():
        best_i = None
        for _ in range(tournament_size):
            i = random.randrange(pop_size)
            if best_i is None or fitness[i] > fitness[best_i]:
                best_i = i
        return best_i

    gen = 0
    while gen < max_gen:
        gen += 1
        new_pop = []

        # elitismo: tieni il migliore
        new_pop.append(best_ind[:])

        while len(new_pop) < pop_size:
            # selezione
            p1 = pop[tournament()]
            p2 = pop[tournament()]

            # crossover monofrontiera
            if random.random() < pc:
                cut = random.randrange(1, N)
                child1 = p1[:cut] + p2[cut:]
                child2 = p2[:cut] + p1[cut:]
            else:
                child1 = p1[:]
                child2 = p2[:]

            # mutazione
            def mutate(ind):
                if random.random() < pm:
                    c = random.randrange(N)
                    ind[c] = random.randrange(N)

            mutate(child1)
            mutate(child2)

            new_pop.append(child1)
            if len(new_pop) < pop_size:
                new_pop.append(child2)

        pop = new_pop
        fitness = [fit_fn(ind) for ind in pop]
        fitness_evals += pop_size

        # aggiorna best sulla base dei conflitti reali
        for ind in pop:
            c = conflicts(ind)
            if c < best_conf:
                best_conf = c
                best_ind = ind[:]

        if best_conf == 0:
            return True, gen, time.time() - start, 0, fitness_evals

    return False, max_gen, time.time() - start, best_conf, fitness_evals


# ======================================================
# 6. Tuning dei parametri GA per un singolo (N, fitness_mode)
# ======================================================

def tune_ga_for_N(
    N,
    fitness_mode,
    pop_multipliers,
    gen_multipliers,
    pm_values,
    pc,
    tournament_size,
    runs_tuning=10,
):
    """
    Fa grid-search sui parametri GA per un certo N e fitness_mode.
    Ritorna il miglior set di parametri trovato (più alcune metriche).
    Criterio:
      1) massimizza il success_rate
      2) a parità di success_rate, minimizza avg_gen_success
    """
    best = None

    for k in pop_multipliers:
        pop_size = max(50, int(k * N))
        for m in gen_multipliers:
            max_gen = int(m * N)
            for pm in pm_values:
                # esegui RUNS_GA_TUNING run con questi parametri
                successes = 0
                gen_success = []

                for _ in range(runs_tuning):
                    s, gen, _, bestc, _ = ga_nqueens(
                        N,
                        pop_size=pop_size,
                        max_gen=max_gen,
                        pc=pc,
                        pm=pm,
                        tournament_size=tournament_size,
                        fitness_mode=fitness_mode,
                    )
                    if s:
                        successes += 1
                        gen_success.append(gen)

                success_rate = successes / runs_tuning
                avg_gen = statistics.mean(gen_success) if gen_success else None

                candidate = {
                    "N": N,
                    "fitness_mode": fitness_mode,
                    "pop_size": pop_size,
                    "max_gen": max_gen,
                    "pc": pc,
                    "pm": pm,
                    "tournament_size": tournament_size,
                    "success_rate": success_rate,
                    "avg_gen_success": avg_gen,
                }

                if best is None:
                    best = candidate
                else:
                    # confronto: prima success_rate, poi avg_gen_success
                    if candidate["success_rate"] > best["success_rate"]:
                        best = candidate
                    elif candidate["success_rate"] == best["success_rate"]:
                        # se entrambi hanno successi, minimizza avg_gen_success
                        if candidate["avg_gen_success"] is not None and best["avg_gen_success"] is not None:
                            if candidate["avg_gen_success"] < best["avg_gen_success"]:
                                best = candidate

    return best


# ======================================================
# 6.5. Funzioni per parallelizzazione
# ======================================================

def run_single_ga_experiment(params):
    """
    Funzione wrapper per eseguire un singolo esperimento GA.
    Necessaria per il multiprocessing.
    """
    N, pop_size, max_gen, pc, pm, tournament_size, fitness_mode = params
    return ga_nqueens(
        N,
        pop_size=pop_size,
        max_gen=max_gen,
        pc=pc,
        pm=pm,
        tournament_size=tournament_size,
        fitness_mode=fitness_mode,
    )


def run_single_sa_experiment(params):
    """
    Funzione wrapper per eseguire un singolo esperimento SA.
    Necessaria per il multiprocessing.
    """
    N, max_iter, T0, alpha = params
    return sa_nqueens(N, max_iter=max_iter, T0=T0, alpha=alpha)


def test_parameter_combination_parallel(params):
    """
    Testa una singola combinazione di parametri GA con run paralleli.
    """
    N, fitness_mode, pop_size, max_gen, pc, pm, tournament_size, runs_tuning = params
    
    # Prepara i parametri per tutti i run
    run_params = [(N, pop_size, max_gen, pc, pm, tournament_size, fitness_mode) 
                  for _ in range(runs_tuning)]
    
    # Esegui i run in parallelo
    with ProcessPoolExecutor(max_workers=min(NUM_PROCESSES, runs_tuning)) as executor:
        results = list(executor.map(run_single_ga_experiment, run_params))
    
    # Calcola statistiche
    successes = 0
    gen_success = []
    for s, gen, _, bestc, _ in results:
        if s:
            successes += 1
            gen_success.append(gen)
    
    success_rate = successes / runs_tuning
    avg_gen = statistics.mean(gen_success) if gen_success else None
    
    return {
        "N": N,
        "fitness_mode": fitness_mode,
        "pop_size": pop_size,
        "max_gen": max_gen,
        "pc": pc,
        "pm": pm,
        "tournament_size": tournament_size,
        "success_rate": success_rate,
        "avg_gen_success": avg_gen,
    }


def tune_ga_for_N_parallel(
    N,
    fitness_mode,
    pop_multipliers,
    gen_multipliers,
    pm_values,
    pc,
    tournament_size,
    runs_tuning=10,
):
    """
    Versione parallela di tune_ga_for_N.
    Parallelizza sia le combinazioni di parametri che i run per ogni combinazione.
    """
    print(f"  Preparazione {len(pop_multipliers) * len(gen_multipliers) * len(pm_values)} combinazioni di parametri...")
    
    # Prepara tutte le combinazioni di parametri
    param_combinations = []
    for k in pop_multipliers:
        pop_size = max(50, int(k * N))
        for m in gen_multipliers:
            max_gen = int(m * N)
            for pm in pm_values:
                param_combinations.append(
                    (N, fitness_mode, pop_size, max_gen, pc, pm, tournament_size, runs_tuning)
                )
    
    # Testa tutte le combinazioni in parallelo
    print(f"  Esecuzione parallela con {NUM_PROCESSES} processi...")
    with ProcessPoolExecutor(max_workers=NUM_PROCESSES) as executor:
        candidates = list(executor.map(test_parameter_combination_parallel, param_combinations))
    
    # Trova la migliore combinazione
    best = None
    for candidate in candidates:
        if best is None:
            best = candidate
        else:
            # confronto: prima success_rate, poi avg_gen_success
            if candidate["success_rate"] > best["success_rate"]:
                best = candidate
            elif candidate["success_rate"] == best["success_rate"]:
                # se entrambi hanno successi, minimizza avg_gen_success
                if candidate["avg_gen_success"] is not None and best["avg_gen_success"] is not None:
                    if candidate["avg_gen_success"] < best["avg_gen_success"]:
                        best = candidate
    
    print(f"  Migliore combinazione: pop_size={best['pop_size']}, max_gen={best['max_gen']}, pm={best['pm']}, success_rate={best['success_rate']:.3f}")
    return best


def tune_single_fitness(params):
    """
    Funzione wrapper per il tuning di una singola fitness.
    Necessaria per il multiprocessing.
    """
    N, fitness_mode, pop_multipliers, gen_multipliers, pm_values, pc, tournament_size, runs_tuning = params
    return fitness_mode, tune_ga_for_N_parallel(
        N, fitness_mode, pop_multipliers, gen_multipliers, pm_values, pc, tournament_size, runs_tuning
    )


def tune_all_fitness_parallel(
    N,
    fitness_modes,
    pop_multipliers,
    gen_multipliers,
    pm_values,
    pc,
    tournament_size,
    runs_tuning=10,
):
    """
    Fa il tuning di tutte le fitness contemporaneamente per un dato N.
    Parallelizza il tuning di F1, F2, F3, F4, F5, F6 simultaneamente.
    """
    print(f"🚀 Tuning contemporaneo di {len(fitness_modes)} fitness per N={N}")
    
    # Prepara i parametri per tutte le fitness
    tuning_params = []
    for fitness_mode in fitness_modes:
        tuning_params.append((
            N, fitness_mode, pop_multipliers, gen_multipliers, pm_values, 
            pc, tournament_size, runs_tuning
        ))
    
    # Esegui il tuning di tutte le fitness in parallelo
    print(f"  Utilizzando {min(NUM_PROCESSES, len(fitness_modes))} processi per {len(fitness_modes)} fitness...")
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=min(NUM_PROCESSES, len(fitness_modes))) as executor:
        results = list(executor.map(tune_single_fitness, tuning_params))
    
    elapsed_time = time.time() - start_time
    
    # Organizza i risultati per fitness
    best_params_per_fitness = {}
    for fitness_mode, best_params in results:
        best_params_per_fitness[fitness_mode] = best_params
        print(f"  ✅ {fitness_mode}: success_rate={best_params['success_rate']:.3f}, "
              f"pop_size={best_params['pop_size']}, pm={best_params['pm']}")
    
    print(f"🏁 Tuning contemporaneo completato in {elapsed_time:.1f}s per N={N}")
    return best_params_per_fitness


# ======================================================
# 7. Esperimenti finali con parametri ottimali GA
# ======================================================

def run_experiments_with_best_ga(
    N_values,
    runs_sa,
    runs_ga,
    bt_time_limit,
    fitness_mode,
    best_ga_params_for_N,
):
    """
    Esegue BT, SA, GA con i parametri GA ottimali già trovati
    (best_ga_params_for_N[N]) per ciascun N.
    """
    results = {"BT": {}, "SA": {}, "GA": {}}

    for N in N_values:
        print(f"=== (Final) N = {N}, GA fitness {fitness_mode} ===")

        # ----- BT -----
        sol, nodes, t = bt_nqueens_first(N, time_limit=bt_time_limit)
        results["BT"][N] = {
            "solution_found": sol is not None,
            "nodes": nodes,
            "time": t,
        }

        # ----- SA -----
        sa_runs = []
        max_iter_sa = 2000 + 200 * N
        for _ in range(runs_sa):
            s, steps, tt, bestc, evals = sa_nqueens(
                N, max_iter=max_iter_sa, T0=1.0, alpha=0.995
            )
            sa_runs.append(
                {
                    "success": s,
                    "steps": steps,
                    "time": tt,
                    "best_conflicts": bestc,
                    "evals": evals,
                }
            )

        sa_successes = [r for r in sa_runs if r["success"]]
        sa_success_rate = len(sa_successes) / runs_sa
        sa_avg_steps = (
            statistics.mean(r["steps"] for r in sa_successes)
            if sa_successes
            else None
        )
        sa_avg_time = (
            statistics.mean(r["time"] for r in sa_successes)
            if sa_successes
            else None
        )

        results["SA"][N] = {
            "success_rate": sa_success_rate,
            "avg_steps_success": sa_avg_steps,
            "avg_time_success": sa_avg_time,
        }

        # ----- GA con parametri ottimali -----
        params = best_ga_params_for_N[N]
        pop_size = params["pop_size"]
        max_gen = params["max_gen"]
        pm = params["pm"]
        pc = params["pc"]
        tsize = params["tournament_size"]

        ga_runs = []
        for _ in range(runs_ga):
            s, gen, tt, bestc, evals = ga_nqueens(
                N,
                pop_size=pop_size,
                max_gen=max_gen,
                pc=pc,
                pm=pm,
                tournament_size=tsize,
                fitness_mode=fitness_mode,
            )
            ga_runs.append(
                {
                    "success": s,
                    "gen": gen,
                    "time": tt,
                    "best_conflicts": bestc,
                    "evals": evals,
                }
            )

        ga_successes = [r for r in ga_runs if r["success"]]
        ga_success_rate = len(ga_successes) / runs_ga
        ga_avg_gen = (
            statistics.mean(r["gen"] for r in ga_successes)
            if ga_successes
            else None
        )
        ga_avg_time = (
            statistics.mean(r["time"] for r in ga_successes)
            if ga_successes
            else None
        )

        results["GA"][N] = {
            "success_rate": ga_success_rate,
            "avg_gen_success": ga_avg_gen,
            "avg_time_success": ga_avg_time,
            # Salviamo anche i parametri GA usati
            "pop_size": pop_size,
            "max_gen": max_gen,
            "pm": pm,
            "pc": pc,
            "tournament_size": tsize,
        }

    return results


def run_experiments_with_best_ga_parallel(
    N_values,
    runs_sa,
    runs_ga,
    bt_time_limit,
    fitness_mode,
    best_ga_params_for_N,
):
    """
    Versione parallela di run_experiments_with_best_ga.
    Parallelizza i run SA e GA per ogni N.
    """
    results = {"BT": {}, "SA": {}, "GA": {}}

    for N in N_values:
        print(f"=== (Final Parallel) N = {N}, GA fitness {fitness_mode} ===")

        # ----- BT (sempre seriale, è veloce) -----
        sol, nodes, t = bt_nqueens_first(N, time_limit=bt_time_limit)
        results["BT"][N] = {
            "solution_found": sol is not None,
            "nodes": nodes,
            "time": t,
        }

        # ----- SA Parallelo -----
        print(f"  Eseguendo {runs_sa} run SA in parallelo...")
        max_iter_sa = 2000 + 200 * N
        sa_params = [(N, max_iter_sa, 1.0, 0.995) for _ in range(runs_sa)]
        
        with ProcessPoolExecutor(max_workers=NUM_PROCESSES) as executor:
            sa_raw_results = list(executor.map(run_single_sa_experiment, sa_params))
        
        sa_runs = []
        for s, steps, tt, bestc, evals in sa_raw_results:
            sa_runs.append({
                "success": s,
                "steps": steps,
                "time": tt,
                "best_conflicts": bestc,
                "evals": evals,
            })

        sa_successes = [r for r in sa_runs if r["success"]]
        sa_success_rate = len(sa_successes) / runs_sa
        sa_avg_steps = (
            statistics.mean(r["steps"] for r in sa_successes)
            if sa_successes
            else None
        )
        sa_avg_time = (
            statistics.mean(r["time"] for r in sa_successes)
            if sa_successes
            else None
        )

        results["SA"][N] = {
            "success_rate": sa_success_rate,
            "avg_steps_success": sa_avg_steps,
            "avg_time_success": sa_avg_time,
        }

        # ----- GA Parallelo con parametri ottimali -----
        print(f"  Eseguendo {runs_ga} run GA in parallelo...")
        params = best_ga_params_for_N[N]
        pop_size = params["pop_size"]
        max_gen = params["max_gen"]
        pm = params["pm"]
        pc = params["pc"]
        tsize = params["tournament_size"]

        ga_params = [(N, pop_size, max_gen, pc, pm, tsize, fitness_mode) 
                     for _ in range(runs_ga)]
        
        with ProcessPoolExecutor(max_workers=NUM_PROCESSES) as executor:
            ga_raw_results = list(executor.map(run_single_ga_experiment, ga_params))
        
        ga_runs = []
        for s, gen, tt, bestc, evals in ga_raw_results:
            ga_runs.append({
                "success": s,
                "gen": gen,
                "time": tt,
                "best_conflicts": bestc,
                "evals": evals,
            })

        ga_successes = [r for r in ga_runs if r["success"]]
        ga_success_rate = len(ga_successes) / runs_ga
        ga_avg_gen = (
            statistics.mean(r["gen"] for r in ga_successes)
            if ga_successes
            else None
        )
        ga_avg_time = (
            statistics.mean(r["time"] for r in ga_successes)
            if ga_successes
            else None
        )

        results["GA"][N] = {
            "success_rate": ga_success_rate,
            "avg_gen_success": ga_avg_gen,
            "avg_time_success": ga_avg_time,
            # Salviamo anche i parametri GA usati
            "pop_size": pop_size,
            "max_gen": max_gen,
            "pm": pm,
            "pc": pc,
            "tournament_size": tsize,
        }

    return results


# ======================================================
# 8. Salvataggio CSV e grafici
# ======================================================

def save_results_to_csv(results, N_values, fitness_mode, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    filename = os.path.join(out_dir, f"results_GA_{fitness_mode}_tuned.csv")

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "N",
            "BT_solution_found",
            "BT_nodes",
            "BT_time",
            "SA_success_rate",
            "SA_avg_steps_success",
            "SA_avg_time_success",
            "GA_success_rate",
            "GA_avg_gen_success",
            "GA_avg_time_success",
            "GA_pop_size",
            "GA_max_gen",
            "GA_pm",
            "GA_pc",
            "GA_tournament_size",
        ])
        for N in N_values:
            bt = results["BT"][N]
            sa = results["SA"][N]
            ga = results["GA"][N]
            writer.writerow([
                N,
                int(bt["solution_found"]),
                bt["nodes"],
                bt["time"],
                sa["success_rate"],
                sa["avg_steps_success"] if sa["avg_steps_success"] is not None else "",
                sa["avg_time_success"] if sa["avg_time_success"] is not None else "",
                ga["success_rate"],
                ga["avg_gen_success"] if ga["avg_gen_success"] is not None else "",
                ga["avg_time_success"] if ga["avg_time_success"] is not None else "",
                ga["pop_size"],
                ga["max_gen"],
                ga["pm"],
                ga["pc"],
                ga["tournament_size"],
            ])

    print(f"CSV salvato: {filename}")


def plot_and_save(results, N_values, fitness_mode, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    bt_sr = [1.0 if results["BT"][N]["solution_found"] else 0.0 for N in N_values]
    sa_sr = [results["SA"][N]["success_rate"] for N in N_values]
    ga_sr = [results["GA"][N]["success_rate"] for N in N_values]

    # Success rate vs N
    plt.figure()
    plt.plot(N_values, bt_sr, marker="o", label="BT (successo entro limite)")
    plt.plot(N_values, sa_sr, marker="o", label="SA (tasso di successo)")
    plt.plot(N_values, ga_sr, marker="o", label=f"GA-{fitness_mode} (tuned)")
    plt.xlabel("N")
    plt.ylabel("Tasso di successo")
    plt.title(f"Tasso di successo vs N (GA {fitness_mode} con tuning)")
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.grid(True)
    fname_success = os.path.join(out_dir, f"success_vs_N_GA_{fitness_mode}_tuned.png")
    plt.savefig(fname_success, bbox_inches="tight")
    plt.close()
    print(f"Grafico salvato: {fname_success}")

    # Tempo vs N
    bt_time = [results["BT"][N]["time"] for N in N_values]
    sa_time = [
        results["SA"][N]["avg_time_success"] if results["SA"][N]["avg_time_success"] is not None else 0.0
        for N in N_values
    ]
    ga_time = [
        results["GA"][N]["avg_time_success"] if results["GA"][N]["avg_time_success"] is not None else 0.0
        for N in N_values
    ]

    plt.figure()
    plt.plot(N_values, bt_time, marker="o", label="BT (tempo 1ª soluzione)")
    plt.plot(N_values, sa_time, marker="o", label="SA (tempo medio, successi)")
    plt.plot(N_values, ga_time, marker="o", label=f"GA-{fitness_mode} (tempo medio, tuned)")
    plt.xlabel("N")
    plt.ylabel("Tempo [s]")
    plt.title(f"Tempo vs N (GA {fitness_mode} con tuning)")
    plt.legend()
    plt.grid(True)
    fname_time = os.path.join(out_dir, f"time_vs_N_GA_{fitness_mode}_tuned.png")
    plt.savefig(fname_time, bbox_inches="tight")
    plt.close()
    print(f"Grafico salvato: {fname_time}")


# ======================================================
# 9. Main: versione parallela del tuning + esperimenti finali
# ======================================================

def main_sequential():
    """Main originale sequenziale (mantenuto per confronto)"""
    os.makedirs(OUT_DIR, exist_ok=True)

    for fitness_mode in FITNESS_MODES:
        print("\n============================================")
        print(f"TUNING GA SEQUENZIALE per fitness mode: {fitness_mode}")
        print("============================================")

        # 1) Tuning per ogni N
        best_ga_params_for_N = {}
        tuning_csv = os.path.join(OUT_DIR, f"tuning_GA_{fitness_mode}_seq.csv")
        with open(tuning_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "N",
                "pop_size",
                "max_gen",
                "pm",
                "pc",
                "tournament_size",
                "success_rate_tuning",
                "avg_gen_success_tuning",
            ])

            for N in N_VALUES:
                print(f"Tuning GA: N = {N}, fitness = {fitness_mode}")
                best = tune_ga_for_N(
                    N,
                    fitness_mode,
                    POP_MULTIPLIERS,
                    GEN_MULTIPLIERS,
                    PM_VALUES,
                    PC_FIXED,
                    TOURNAMENT_SIZE_FIXED,
                    runs_tuning=RUNS_GA_TUNING,
                )
                best_ga_params_for_N[N] = best
                print("  Migliori parametri trovati:", best)

                writer.writerow([
                    N,
                    best["pop_size"],
                    best["max_gen"],
                    best["pm"],
                    best["pc"],
                    best["tournament_size"],
                    best["success_rate"],
                    best["avg_gen_success"],
                ])

        print(f"Tuning export CSV: {tuning_csv}")

        # 2) Esperimenti finali con quei parametri ottimali
        print(f"\nESPERIMENTI FINALI per GA fitness {fitness_mode} (parametri ottimali)")
        results = run_experiments_with_best_ga(
            N_VALUES,
            runs_sa=RUNS_SA_FINAL,
            runs_ga=RUNS_GA_FINAL,
            bt_time_limit=BT_TIME_LIMIT,
            fitness_mode=fitness_mode,
            best_ga_params_for_N=best_ga_params_for_N,
        )

        # 3) Salva CSV riassuntivo e grafici
        save_results_to_csv(results, N_VALUES, fitness_mode, OUT_DIR)
        plot_and_save(results, N_VALUES, fitness_mode, OUT_DIR)

    print("\nTutti i tuning e gli esperimenti finali sequenziali sono completati.")


def main_parallel():
    """Main parallelo ottimizzato"""
    os.makedirs(OUT_DIR, exist_ok=True)
    
    print(f"\n🚀 AVVIO VERSIONE PARALLELA (utilizzando {NUM_PROCESSES} processi)")
    print(f"CPU disponibili: {multiprocessing.cpu_count()}")
    
    start_total = time.time()

    # ======================================================
    # FASE 1: TUNING PARALLELO PER TUTTE LE FITNESS
    # ======================================================
    print("\n" + "="*60)
    print("FASE 1: TUNING GA PARALLELO PER TUTTE LE FITNESS")
    print("="*60)
    
    # Dizionario per salvare i parametri ottimali di ogni fitness
    all_best_params = {}
    
    for fitness_mode in FITNESS_MODES:
        print(f"\n🔧 Tuning per fitness {fitness_mode}...")
        fitness_start = time.time()

        # Tuning parallelo per ogni N di questa fitness
        best_ga_params_for_N = {}
        tuning_csv = os.path.join(OUT_DIR, f"tuning_GA_{fitness_mode}.csv")
        with open(tuning_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "N",
                "pop_size",
                "max_gen",
                "pm",
                "pc",
                "tournament_size",
                "success_rate_tuning",
                "avg_gen_success_tuning",
            ])

            for N in N_VALUES:
                print(f"  🔧 Tuning N = {N}...")
                tuning_start = time.time()
                
                best = tune_ga_for_N_parallel(
                    N,
                    fitness_mode,
                    POP_MULTIPLIERS,
                    GEN_MULTIPLIERS,
                    PM_VALUES,
                    PC_FIXED,
                    TOURNAMENT_SIZE_FIXED,
                    runs_tuning=RUNS_GA_TUNING,
                )
                
                tuning_time = time.time() - tuning_start
                best_ga_params_for_N[N] = best
                print(f"     ✅ Completato in {tuning_time:.1f}s - Success rate: {best['success_rate']:.3f}")

                writer.writerow([
                    N,
                    best["pop_size"],
                    best["max_gen"],
                    best["pm"],
                    best["pc"],
                    best["tournament_size"],
                    best["success_rate"],
                    best["avg_gen_success"],
                ])

        # Salva i parametri per questa fitness
        all_best_params[fitness_mode] = best_ga_params_for_N
        
        fitness_time = time.time() - fitness_start
        print(f"📄 Tuning {fitness_mode} completato in {fitness_time:.1f}s - CSV: {tuning_csv}")

    # ======================================================
    # FASE 2: ESPERIMENTI FINALI PARALLELI PER TUTTE LE FITNESS
    # ======================================================
    print(f"\n" + "="*60)
    print("FASE 2: ESPERIMENTI FINALI PARALLELI")
    print("="*60)
    
    for fitness_mode in FITNESS_MODES:
        print(f"\n🧪 Esperimenti finali per {fitness_mode}...")
        experiments_start = time.time()
        
        results = run_experiments_with_best_ga_parallel(
            N_VALUES,
            runs_sa=RUNS_SA_FINAL,
            runs_ga=RUNS_GA_FINAL,
            bt_time_limit=BT_TIME_LIMIT,
            fitness_mode=fitness_mode,
            best_ga_params_for_N=all_best_params[fitness_mode],
        )
        
        experiments_time = time.time() - experiments_start
        print(f"  ✅ Esperimenti completati in {experiments_time:.1f}s")

        # Salva risultati finali
        print(f"📊 Generazione grafici e CSV finali...")
        save_results_to_csv(results, N_VALUES, fitness_mode, OUT_DIR)
        plot_and_save(results, N_VALUES, fitness_mode, OUT_DIR)
        print(f"  ✅ Risultati salvati per {fitness_mode}")

    total_time = time.time() - start_total
    print(f"\n🏁 PIPELINE PARALLELA COMPLETATA!")
    print(f"⏱️  Tempo totale: {total_time:.1f}s ({total_time/60:.1f} minuti)")
    print(f"📊 Fitness processate: {len(FITNESS_MODES)}")
    print(f"🖥️  Processi utilizzati: {NUM_PROCESSES}")


def main_concurrent_tuning():
    """Main con tuning contemporaneo di tutte le fitness"""
    os.makedirs(OUT_DIR, exist_ok=True)
    
    print(f"\n🚀 TUNING CONTEMPORANEO DI TUTTE LE FITNESS")
    print(f"Fitness: {FITNESS_MODES}")
    print(f"Processi: {NUM_PROCESSES}")
    print(f"CPU disponibili: {multiprocessing.cpu_count()}")
    
    start_total = time.time()

    # ======================================================
    # FASE 1: TUNING CONTEMPORANEO PER TUTTI N
    # ======================================================
    print("\n" + "="*70)
    print("FASE 1: TUNING CONTEMPORANEO PER TUTTE LE FITNESS")
    print("="*70)
    
    # Dizionario per salvare i parametri ottimali: all_best_params[fitness_mode][N] = params
    all_best_params = {fitness_mode: {} for fitness_mode in FITNESS_MODES}
    
    for N in N_VALUES:
        print(f"\n🎯 Tuning contemporaneo per N = {N}")
        print("-" * 50)
        
        # Tuning contemporaneo di tutte le fitness per questo N
        fitness_results = tune_all_fitness_parallel(
            N,
            FITNESS_MODES,
            POP_MULTIPLIERS,
            GEN_MULTIPLIERS,
            PM_VALUES,
            PC_FIXED,
            TOURNAMENT_SIZE_FIXED,
            runs_tuning=RUNS_GA_TUNING,
        )
        
        # Salva i risultati per ogni fitness
        for fitness_mode, best_params in fitness_results.items():
            all_best_params[fitness_mode][N] = best_params
    
    # Salva i file CSV di tuning per ogni fitness
    print(f"\n💾 Salvando file CSV di tuning...")
    for fitness_mode in FITNESS_MODES:
        tuning_csv = os.path.join(OUT_DIR, f"tuning_GA_{fitness_mode}.csv")
        with open(tuning_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "N",
                "pop_size",
                "max_gen",
                "pm",
                "pc",
                "tournament_size",
                "success_rate_tuning",
                "avg_gen_success_tuning",
            ])
            
            for N in N_VALUES:
                best = all_best_params[fitness_mode][N]
                writer.writerow([
                    N,
                    best["pop_size"],
                    best["max_gen"],
                    best["pm"],
                    best["pc"],
                    best["tournament_size"],
                    best["success_rate"],
                    best["avg_gen_success"],
                ])
        print(f"  ✅ {tuning_csv}")

    # ======================================================
    # FASE 2: ESPERIMENTI FINALI PER TUTTE LE FITNESS
    # ======================================================
    print(f"\n" + "="*70)
    print("FASE 2: ESPERIMENTI FINALI PER TUTTE LE FITNESS")
    print("="*70)
    
    for fitness_mode in FITNESS_MODES:
        print(f"\n🧪 Esperimenti finali per {fitness_mode}...")
        experiments_start = time.time()
        
        results = run_experiments_with_best_ga_parallel(
            N_VALUES,
            runs_sa=RUNS_SA_FINAL,
            runs_ga=RUNS_GA_FINAL,
            bt_time_limit=BT_TIME_LIMIT,
            fitness_mode=fitness_mode,
            best_ga_params_for_N=all_best_params[fitness_mode],
        )
        
        experiments_time = time.time() - experiments_start
        print(f"  ✅ Esperimenti completati in {experiments_time:.1f}s")

        # Salva risultati finali
        print(f"📊 Generazione grafici e CSV finali...")
        save_results_to_csv(results, N_VALUES, fitness_mode, OUT_DIR)
        plot_and_save(results, N_VALUES, fitness_mode, OUT_DIR)
        print(f"  ✅ Risultati salvati per {fitness_mode}")

    total_time = time.time() - start_total
    print(f"\n🏆 TUNING CONTEMPORANEO COMPLETATO!")
    print(f"⏱️  Tempo totale: {total_time:.1f}s ({total_time/60:.1f} minuti)")
    print(f"📊 Fitness processate contemporaneamente: {len(FITNESS_MODES)}")
    print(f"🖥️  Processi utilizzati: {NUM_PROCESSES}")


if __name__ == "__main__":
    # Scelta tra le diverse modalità
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--sequential":
        print("🔄 Esecuzione in modalità SEQUENZIALE")
        main_sequential()
    elif len(sys.argv) > 1 and sys.argv[1] == "--parallel":
        print("🚀 Esecuzione in modalità PARALLELA (vecchia)")
        main_parallel()
    else:
        print("🚀 Esecuzione in modalità TUNING CONTEMPORANEO (default)")
        print("   Usa --sequential per la versione sequenziale")
        print("   Usa --parallel per la versione parallela classica")
        main_concurrent_tuning()
