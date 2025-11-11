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

# Dimensioni della scacchiera da testare - valori crescenti per analisi scalabilità
N_VALUES = [8, 16, 24]

# Numero di run indipendenti per SA e GA negli esperimenti finali
# Più run = maggiore affidabilità statistica, ma tempi più lunghi
RUNS_SA_FINAL = 20
RUNS_GA_FINAL = 20

# Numero di run per la fase di tuning GA (per combinazione di parametri)
# Meno run nel tuning per velocizzare la ricerca parametri
RUNS_GA_TUNING = 5

# Limite di tempo per BT in secondi (None = nessun limite)
# Utile per evitare che BT rimanga bloccato su istanze difficili
BT_TIME_LIMIT = None  # es. 5.0 secondi

# Directory di output per CSV e grafici
OUT_DIR = "results_nqueens_tuning"

# Griglia di tuning per il GA - definisce lo spazio di ricerca parametri
POP_MULTIPLIERS = [4, 8, 16]       # pop_size ≈ 4N, 8N, 16N - popolazione scala con N
GEN_MULTIPLIERS = [30, 50, 80]     # max_gen ≈ 30N, 50N, 80N - generazioni scala con N
RUNS_GA_TUNING = 5                 # per non morire di tempi
PM_VALUES = [0.05, 0.1, 0.15]        # probabilità mutazione - range tipico per GA
PC_FIXED = 0.8                     # probabilità crossover fissa (valore standard)
TOURNAMENT_SIZE_FIXED = 3          # dimensione torneo per selezione

# Tutte le funzioni di fitness da testare (F1-F6)
FITNESS_MODES = ["F1", "F2", "F3", "F4", "F5", "F6"]

# Numero di processi per il parallelismo
# Lascia un core libero per il sistema operativo
NUM_PROCESSES = multiprocessing.cpu_count() - 1


# ======================================================
# 1. Utility comuni
# ======================================================

def conflicts(board):
    """
    Versione O(N): conta le coppie di regine in conflitto usando
    contatori per righe e diagonali.
    
    Args:
        board: lista dove board[col] = row (posizione regine)
    
    Returns:
        int: numero totale di coppie di regine in conflitto
        
    Note:
        - Usa Counter per efficienza O(N) invece di O(N²)
        - Calcola conflitti come combinazioni C(k,2) per ogni gruppo
    """
    n = len(board)
    row_count = Counter()    # conta regine per riga
    diag1 = Counter()        # diagonale principale (r-c costante)
    diag2 = Counter()        # diagonale secondaria (r+c costante)

    # Conta occupazioni per ogni riga e diagonale
    for c, r in enumerate(board):
        row_count[r] += 1
        diag1[r - c] += 1    # diagonale ↘ (top-left to bottom-right)
        diag2[r + c] += 1    # diagonale ↙ (top-right to bottom-left)

    def pairs(counter):
        """Calcola numero di coppie in conflitto per un contatore"""
        tot = 0
        for cnt in counter.values():
            if cnt > 1:
                tot += cnt * (cnt - 1) // 2  # combinazioni C(cnt,2)
        return tot

    # Somma conflitti da righe e entrambe le diagonali
    row_conf = pairs(row_count)
    d1_conf  = pairs(diag1)
    d2_conf  = pairs(diag2)

    return row_conf + d1_conf + d2_conf

def conflicts_on2(board):
    """
    Versione O(N²): conta il numero di coppie di regine in conflitto.
    Mantenuta per confronto/debugging con la versione ottimizzata.
    
    Args:
        board: lista dove board[i] = riga della regina nella colonna i
        
    Returns:
        int: numero di coppie di regine in conflitto
    """
    n = len(board)
    c = 0
    # Confronta ogni coppia di regine
    for i in range(n):
        for j in range(i + 1, n):
            # Stesso riga o stessa diagonale
            if board[i] == board[j] or abs(board[i] - board[j]) == abs(i - j):
                c += 1
    return c


# ======================================================
# 2. Funzioni di fitness F1 ... F6 per il GA
# ======================================================

def fitness_f1(ind):
    """
    F1: fitness = -conflitti.
    
    Approccio diretto: minimizza i conflitti trasformandoli in fitness negativa.
    Più conflitti = fitness peggiore (più negativa).
    """
    return -conflicts(ind)


def fitness_f2(ind):
    """
    F2: numero di coppie NON in conflitto.
    
    Approccio positivo: conta le coppie di regine che NON si attaccano.
    Soluzione ottima ha tutte le C(N,2) coppie non in conflitto.
    """
    n = len(ind)
    max_pairs = n * (n - 1) // 2  # numero massimo di coppie possibili
    c = conflicts(ind)
    return max_pairs - c


def fitness_f3(ind):
    """
    F3: penalità lineare su cluster di regine sulle stesse diagonali.
    
    Diverso da F1/F2: penalizza specificamente i cluster sulle diagonali
    con penalità lineare C(cnt,2). Incentiva distribuzione uniforme.
    """
    n = len(ind)
    diag1 = Counter()  # diagonale principale
    diag2 = Counter()  # diagonale secondaria
    
    # Conta regine per diagonale
    for c, r in enumerate(ind):
        diag1[r - c] += 1
        diag2[r + c] += 1

    penalty = 0
    # Penalità lineare per cluster sulle diagonali
    for cnt in diag1.values():
        if cnt > 1:
            penalty += cnt * (cnt - 1) // 2  # C(cnt,2)
    for cnt in diag2.values():
        if cnt > 1:
            penalty += cnt * (cnt - 1) // 2

    max_pairs = n * (n - 1) // 2
    return max_pairs - penalty


def fitness_f4(ind):
    """
    F4: F2 meno i conflitti della regina con più conflitti.
    
    Penalizza soluzioni sbilanciate dove una regina ha troppi conflitti.
    Incentiva soluzioni più equilibrate.
    """
    n = len(ind)
    max_pairs = n * (n - 1) // 2
    total_conf = conflicts(ind)
    base = max_pairs - total_conf  # base F2

    # Trova la regina con più conflitti
    max_conf_for_queen = 0
    for c in range(n):
        conf_q = 0
        # Conta conflitti per regina in colonna c
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
    
    Simile a F3 ma con penalizzazione più severa (k² invece di C(k,2)).
    Scoraggia fortemente la formazione di grandi cluster.
    """
    n = len(ind)
    diag1 = Counter()
    diag2 = Counter()
    
    for c, r in enumerate(ind):
        diag1[r - c] += 1
        diag2[r + c] += 1

    penalty = 0
    # Penalità quadratica per cluster
    for cnt in diag1.values():
        if cnt > 1:
            penalty += cnt ** 2  # penalità quadratica
    for cnt in diag2.values():
        if cnt > 1:
            penalty += cnt ** 2

    max_pairs = n * (n - 1) // 2
    return max_pairs - penalty


def fitness_f6(ind, lam=0.3):
    """
    F6: trasformazione esponenziale dei conflitti.
    
    fitness = exp(-lambda * conflicts)
    Trasformazione non-lineare che amplifica le differenze tra soluzioni
    con pochi conflitti. Lambda controlla la "pendenza".
    """
    c = conflicts(ind)
    return math.exp(-lam * c)


def get_fitness_function(mode):
    """
    Factory function: ritorna la funzione di fitness corrispondente al mode.
    
    Args:
        mode: stringa che identifica la fitness ("F1", "F2", etc.)
        
    Returns:
        function: funzione di fitness corrispondente
        
    Raises:
        ValueError: se il mode non è riconosciuto
    """
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
        return lambda ind: fitness_f6(ind, lam=0.3)  # lambda fissato
    else:
        raise ValueError(f"fitness_mode sconosciuto: {mode}")


# ======================================================
# 3. Backtracking iterativo (prima soluzione)
# ======================================================

def bt_nqueens_first(N, time_limit=None):
    """
    Backtracking iterativo per trovare UNA sola soluzione delle N-Regine.
    
    Implementazione ottimizzata:
    - Iterativa (no ricorsione) per evitare stack overflow
    - Tracking veloce dei conflitti con array booleani
    - Conta i nodi esplorati per analisi di complessità
    - Time limit opzionale per evitare esecuzioni infinite
    
    Args:
        N: dimensione scacchiera (N×N)
        time_limit: limite tempo in secondi (None = illimitato)
        
    Returns:
        tuple: (soluzione, nodi_esplorati, tempo_esecuzione)
        - soluzione: lista [row0, row1, ..., rowN-1] o None se non trovata
        - nodi_esplorati: numero di posizionamenti tentati
        - tempo_esecuzione: secondi di esecuzione
    """
    # Stato della ricerca
    pos = [-1] * N                           # posizione regine: pos[col] = row
    row_used = [False] * N                   # righe occupate
    diag1_used = [False] * (2 * N - 1)      # diagonali principali occupate
    diag2_used = [False] * (2 * N - 1)      # diagonali secondarie occupate

    col = 0          # colonna corrente
    row = 0          # riga corrente da tentare
    nodes = 0        # contatore nodi esplorati
    start = time.time()

    # Ciclo principale del backtracking
    while col >= 0 and col < N:
        # Controlla time limit
        if time_limit is not None and (time.time() - start) > time_limit:
            return None, nodes, time.time() - start

        placed = False
        
        # Prova tutte le righe disponibili per la colonna corrente
        while row < N and not placed:
            nodes += 1  # conta ogni tentativo di posizionamento
            
            # Controlla se la posizione (row, col) è valida
            if not row_used[row]:
                # Calcola indici diagonali
                d1 = row - col + (N - 1)  # diagonale principale ↘
                d2 = row + col            # diagonale secondaria ↙
                
                # Se entrambe le diagonali sono libere
                if not diag1_used[d1] and not diag2_used[d2]:
                    # Posiziona la regina
                    pos[col] = row
                    row_used[row] = True
                    diag1_used[d1] = True
                    diag2_used[d2] = True
                    placed = True

                    # Se abbiamo posizionato tutte le regine
                    if col == N - 1:
                        return pos.copy(), nodes, time.time() - start
                    else:
                        # Passa alla colonna successiva
                        col += 1
                        row = 0
                else:
                    row += 1  # prova riga successiva
            else:
                row += 1  # prova riga successiva

        # Se non è stato possibile posizionare in questa colonna
        if not placed:
            # Backtrack: rimuovi regina dalla colonna precedente
            col -= 1
            if col >= 0:
                # Ripristina stato precedente
                prev_row = pos[col]
                pos[col] = -1
                row_used[prev_row] = False
                d1 = prev_row - col + (N - 1)
                d2 = prev_row + col
                diag1_used[d1] = False
                diag2_used[d2] = False
                row = prev_row + 1  # riprendi dalla riga successiva

    # Nessuna soluzione trovata
    return None, nodes, time.time() - start


# ======================================================
# 4. Simulated Annealing (SA)
# ======================================================

def sa_nqueens(N, max_iter=20000, T0=1.0, alpha=0.995):
    """
    Simulated Annealing per il problema delle N-Regine.
    
    Algoritmo:
    1. Parte da configurazione casuale
    2. Ad ogni iterazione sposta una regina casuale
    3. Accetta la mossa secondo criterio di Metropolis
    4. Diminuisce temperatura geometricamente
    
    Args:
        N: dimensione scacchiera
        max_iter: numero massimo di iterazioni
        T0: temperatura iniziale (alta = più esplorazione)
        alpha: fattore raffreddamento (0 < alpha < 1, tipicamente ~0.995)
        
    Returns:
        tuple: (successo, iterazioni, tempo, migliori_conflitti, valutazioni_fitness)
        - successo: True se trovata soluzione (0 conflitti)
        - iterazioni: numero iterazioni eseguite
        - tempo: secondi di esecuzione
        - migliori_conflitti: minor numero conflitti raggiunto
        - valutazioni_fitness: numero chiamate funzione conflicts()
    """
    # Inizializzazione casuale: una regina per colonna
    board = [random.randrange(N) for _ in range(N)]
    cur_cost = conflicts(board)     # conflitti configurazione corrente
    best_cost = cur_cost           # miglior configurazione vista
    fitness_evals = 1              # contatore valutazioni fitness
    start = time.time()

    # Controlla se già abbiamo la soluzione
    if cur_cost == 0:
        return True, 0, time.time() - start, 0, fitness_evals

    T = T0  # temperatura corrente
    
    # Ciclo principale SA
    for it in range(1, max_iter + 1):
        # Genera vicino: sposta una regina casuale
        c = random.randrange(N)        # colonna casuale
        old_row = board[c]             # riga precedente
        new_row = random.randrange(N)  # nuova riga casuale
        
        # Assicurati che sia diversa (vero move)
        while new_row == old_row:
            new_row = random.randrange(N)
            
        board[c] = new_row

        # Valuta nuova configurazione
        new_cost = conflicts(board)
        fitness_evals += 1
        delta = new_cost - cur_cost  # differenza costi

        # Criterio di accettazione Metropolis
        if delta <= 0 or random.random() < math.exp(-delta / T):
            # Accetta la mossa
            cur_cost = new_cost
            if cur_cost < best_cost:
                best_cost = cur_cost
        else:
            # Rifiuta la mossa: ripristina stato precedente
            board[c] = old_row

        # Controlla se soluzione trovata
        if cur_cost == 0:
            return True, it, time.time() - start, 0, fitness_evals

        # Raffreddamento geometrico
        T *= alpha

    # Max iterazioni raggiunte senza trovare soluzione
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
    Algoritmo Genetico per il problema delle N-Regine.
    
    Implementazione standard con:
    - Rappresentazione: lista di N interi (pos. regine)
    - Selezione: tournament selection
    - Crossover: single-point crossover
    - Mutazione: flip random
    - Elitismo: preserva sempre il migliore
    
    Args:
        N: dimensione scacchiera
        pop_size: dimensione popolazione
        max_gen: numero massimo generazioni
        pc: probabilità crossover (0.0-1.0)
        pm: probabilità mutazione (0.0-1.0)
        tournament_size: dimensione torneo per selezione
        fitness_mode: stringa funzione fitness ("F1", "F2", etc.)
        
    Returns:
        tuple: (successo, generazioni, tempo, migliori_conflitti, valutazioni_fitness)
        - successo: True se trovata soluzione (0 conflitti)
        - generazioni: numero generazioni eseguite
        - tempo: secondi di esecuzione
        - migliori_conflitti: minor numero conflitti nel migliore individuo
        - valutazioni_fitness: numero totale chiamate fitness
    """
    fit_fn = get_fitness_function(fitness_mode)

    # Inizializzazione popolazione casuale
    pop = [[random.randrange(N) for _ in range(N)] for _ in range(pop_size)]
    fitness = [fit_fn(ind) for ind in pop]
    fitness_evals = pop_size

    # Trova migliore iniziale
    best_idx = max(range(pop_size), key=lambda i: fitness[i])
    best_ind = pop[best_idx][:]
    best_conf = conflicts(best_ind)  # numero conflitti reale (non fitness)
    start = time.time()

    # Controlla se già risolto
    if best_conf == 0:
        return True, 0, time.time() - start, 0, fitness_evals

    def tournament():
        """Selezione a torneo: sceglie il migliore tra tournament_size individui"""
        best_i = None
        for _ in range(tournament_size):
            i = random.randrange(pop_size)
            if best_i is None or fitness[i] > fitness[best_i]:
                best_i = i
        return best_i

    gen = 0
    
    # Ciclo evolutivo principale
    while gen < max_gen:
        gen += 1
        new_pop = []

        # Elitismo: mantieni sempre il migliore
        new_pop.append(best_ind[:])

        # Genera nuova popolazione
        while len(new_pop) < pop_size:
            # Selezione genitori
            p1 = pop[tournament()]
            p2 = pop[tournament()]

            # Crossover single-point
            if random.random() < pc:
                cut = random.randrange(1, N)  # punto taglio [1, N-1]
                child1 = p1[:cut] + p2[cut:]
                child2 = p2[:cut] + p1[cut:]
            else:
                # Nessun crossover: copia genitori
                child1 = p1[:]
                child2 = p2[:]

            # Mutazione
            def mutate(ind):
                """Mutazione: flip random di una posizione"""
                if random.random() < pm:
                    c = random.randrange(N)
                    ind[c] = random.randrange(N)

            mutate(child1)
            mutate(child2)

            # Aggiungi figli alla nuova popolazione
            new_pop.append(child1)
            if len(new_pop) < pop_size:
                new_pop.append(child2)

        # Sostituisci popolazione
        pop = new_pop
        fitness = [fit_fn(ind) for ind in pop]
        fitness_evals += pop_size

        # Aggiorna migliore basandoti sui conflitti reali
        for ind in pop:
            c = conflicts(ind)
            if c < best_conf:
                best_conf = c
                best_ind = ind[:]

        # Controlla se soluzione trovata
        if best_conf == 0:
            return True, gen, time.time() - start, 0, fitness_evals

    # Max generazioni raggiunte
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
    Grid search esaustiva per ottimizzare parametri GA.
    
    Testa tutte le combinazioni di parametri e seleziona la migliore
    secondo il criterio: 1) massimo success rate, 2) minime generazioni.
    
    Args:
        N: dimensione problema
        fitness_mode: funzione fitness da usare ("F1", "F2", etc.)
        pop_multipliers: moltiplicatori per dimensione popolazione [k1, k2, ...]
        gen_multipliers: moltiplicatori per numero generazioni [m1, m2, ...]
        pm_values: valori probabilità mutazione da testare [p1, p2, ...]
        pc: probabilità crossover fissa
        tournament_size: dimensione torneo fissa
        runs_tuning: numero run indipendenti per combinazione
        
    Returns:
        dict: migliori parametri con statistiche associate
        {
            "N": N,
            "fitness_mode": fitness_mode,
            "pop_size": migliore_pop_size,
            "max_gen": migliore_max_gen,
            "pm": migliore_pm,
            "pc": pc,
            "tournament_size": tournament_size,
            "success_rate": tasso_successo_miglior_configurazione,
            "avg_gen_success": generazioni_medie_successi_migliore
        }
    """
    best = None

    # Prova tutte le combinazioni di parametri
    for k in pop_multipliers:
        pop_size = max(50, int(k * N))  # popolazione minima 50
        for m in gen_multipliers:
            max_gen = int(m * N)  # generazioni scalano con N
            for pm in pm_values:
                # Testa questa combinazione con runs_tuning esperimenti
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
                    if s:  # se ha trovato soluzione
                        successes += 1
                        gen_success.append(gen)

                # Calcola statistiche per questa combinazione
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

                # Confronta con miglior candidato attuale
                if best is None:
                    best = candidate
                else:
                    # Criterio di selezione: prima success_rate, poi avg_gen_success
                    if candidate["success_rate"] > best["success_rate"]:
                        best = candidate
                    elif candidate["success_rate"] == best["success_rate"]:
                        # A parità di success rate, minimizza generazioni medie
                        if candidate["avg_gen_success"] is not None and best["avg_gen_success"] is not None:
                            if candidate["avg_gen_success"] < best["avg_gen_success"]:
                                best = candidate

    return best


# ======================================================
# 6.5. Funzioni per parallelizzazione
# ======================================================

def run_single_ga_experiment(params):
    """
    Wrapper per eseguire un singolo esperimento GA in un processo separato.
    
    Necessaria perché ProcessPoolExecutor richiede funzioni top-level
    (non può serializzare lambda o metodi di classe).
    
    Args:
        params: tupla (N, pop_size, max_gen, pc, pm, tournament_size, fitness_mode)
        
    Returns:
        tuple: risultato di ga_nqueens()
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
    Wrapper per eseguire un singolo esperimento SA in un processo separato.
    
    Args:
        params: tupla (N, max_iter, T0, alpha)
        
    Returns:
        tuple: risultato di sa_nqueens()
    """
    N, max_iter, T0, alpha = params
    return sa_nqueens(N, max_iter=max_iter, T0=T0, alpha=alpha)


def test_parameter_combination_parallel(params):
    """
    Testa una singola combinazione di parametri GA eseguendo
    multiple run in parallelo e calcolando le statistiche.
    
    Args:
        params: tupla (N, fitness_mode, pop_size, max_gen, pc, pm, tournament_size, runs_tuning)
        
    Returns:
        dict: statistiche per questa combinazione di parametri
    """
    N, fitness_mode, pop_size, max_gen, pc, pm, tournament_size, runs_tuning = params
    
    # Prepara parametri per tutti i run di questa combinazione
    run_params = [(N, pop_size, max_gen, pc, pm, tournament_size, fitness_mode) 
                  for _ in range(runs_tuning)]
    
    # Esegui i run in parallelo (limitato da NUM_PROCESSES)
    with ProcessPoolExecutor(max_workers=min(NUM_PROCESSES, runs_tuning)) as executor:
        results = list(executor.map(run_single_ga_experiment, run_params))
    
    # Calcola statistiche aggregate
    successes = 0
    gen_success = []
    for s, gen, _, bestc, _ in results:
        if s:  # se ha trovato soluzione
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
    
    Parallelizza su due livelli:
    1. Combinazioni di parametri diverse vengono testate in parallelo
    2. I run multipli per ogni combinazione vengono eseguiti in parallelo
    
    Questo porta a un speedup significativo rispetto alla versione sequenziale,
    specialmente quando ci sono molte combinazioni da testare.
    
    Args:
        N: dimensione problema
        fitness_mode: funzione fitness da usare
        pop_multipliers, gen_multipliers, pm_values: spazio parametri
        pc, tournament_size: parametri fissi
        runs_tuning: run per combinazione
        
    Returns:
        dict: migliori parametri trovati
    """
    print(f"  Preparazione {len(pop_multipliers) * len(gen_multipliers) * len(pm_values)} combinazioni di parametri...")
    
    # Genera tutte le combinazioni di parametri da testare
    param_combinations = []
    for k in pop_multipliers:
        pop_size = max(50, int(k * N))
        for m in gen_multipliers:
            max_gen = int(m * N)
            for pm in pm_values:
                param_combinations.append(
                    (N, fitness_mode, pop_size, max_gen, pc, pm, tournament_size, runs_tuning)
                )
    
    # Testa tutte le combinazioni in parallelo usando ProcessPoolExecutor
    print(f"  Esecuzione parallela con {NUM_PROCESSES} processi...")
    with ProcessPoolExecutor(max_workers=NUM_PROCESSES) as executor:
        candidates = list(executor.map(test_parameter_combination_parallel, param_combinations))
    
    # Seleziona la migliore combinazione usando stesso criterio della versione sequenziale
    best = None
    for candidate in candidates:
        if best is None:
            best = candidate
        else:
            # Criterio: prima success_rate, poi avg_gen_success
            if candidate["success_rate"] > best["success_rate"]:
                best = candidate
            elif candidate["success_rate"] == best["success_rate"]:
                # A parità di success rate, minimizza generazioni medie
                if candidate["avg_gen_success"] is not None and best["avg_gen_success"] is not None:
                    if candidate["avg_gen_success"] < best["avg_gen_success"]:
                        best = candidate
    
    print(f"  Migliore combinazione: pop_size={best['pop_size']}, max_gen={best['max_gen']}, pm={best['pm']}, success_rate={best['success_rate']:.3f}")
    return best


def tune_single_fitness(params):
    """
    Wrapper per il tuning di una singola fitness function.
    Esegue il tuning GA per una specifica fitness in un processo separato.
    
    Args:
        params: tupla (N, fitness_mode, pop_multipliers, gen_multipliers, 
                      pm_values, pc, tournament_size, runs_tuning)
        
    Returns:
        tuple: (fitness_mode, migliori_parametri)
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
    Esegue il tuning di TUTTE le fitness functions contemporaneamente per un dato N.
    
    Questa è la funzione chiave per il parallelismo avanzato:
    invece di fare il tuning di F1, poi F2, poi F3, etc. in sequenza,
    esegue il tuning di F1, F2, F3, F4, F5, F6 simultaneamente su core diversi.
    
    Vantaggi:
    - Speedup lineare con numero di fitness (fino a limite di core disponibili)
    - Utilizzo ottimale delle risorse multi-core
    - Tempo totale = max(tempo_singola_fitness) invece di sum(tempi_fitness)
    
    Args:
        N: dimensione problema
        fitness_modes: lista funzioni fitness da testare ["F1", "F2", ...]
        pop_multipliers, gen_multipliers, pm_values: spazio parametri
        pc, tournament_size: parametri fissi
        runs_tuning: run per combinazione parametri
        
    Returns:
        dict: {fitness_mode: migliori_parametri} per ogni fitness
    """
    print(f"Tuning contemporaneo di {len(fitness_modes)} fitness per N={N}")
    
    # Prepara parametri per tutte le fitness
    tuning_params = []
    for fitness_mode in fitness_modes:
        tuning_params.append((
            N, fitness_mode, pop_multipliers, gen_multipliers, pm_values, 
            pc, tournament_size, runs_tuning
        ))
    
    # Esegui il tuning di tutte le fitness in parallelo
    # Ogni fitness viene processata su un core diverso
    print(f"  Utilizzando {min(NUM_PROCESSES, len(fitness_modes))} processi per {len(fitness_modes)} fitness...")
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=min(NUM_PROCESSES, len(fitness_modes))) as executor:
        results = list(executor.map(tune_single_fitness, tuning_params))
    
    elapsed_time = time.time() - start_time
    
    # Organizza risultati per fitness
    best_params_per_fitness = {}
    for fitness_mode, best_params in results:
        best_params_per_fitness[fitness_mode] = best_params
        print(f"  Completato {fitness_mode}: success_rate={best_params['success_rate']:.3f}, "
              f"pop_size={best_params['pop_size']}, pm={best_params['pm']}")
    
    print(f"Tuning contemporaneo completato in {elapsed_time:.1f}s per N={N}")
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
    Esegue esperimenti finali con parametri GA ottimali (versione sequenziale).
    
    Per ogni N:
    1. Esegue Backtracking (1 volta, deterministico)
    2. Esegue SA (runs_sa volte, stocastico)
    3. Esegue GA (runs_ga volte, stocastico) con parametri già ottimizzati
    
    Args:
        N_values: liste dimensioni da testare [8, 16, 24, ...]
        runs_sa: numero run indipendenti per SA
        runs_ga: numero run indipendenti per GA
        bt_time_limit: limite tempo per BT (None = illimitato)
        fitness_mode: fitness function per GA ("F1", "F2", ...)
        best_ga_params_for_N: dict {N: parametri_ottimali} dal tuning
        
    Returns:
        dict: risultati strutturati
        {
            "BT": {N: {"solution_found": bool, "nodes": int, "time": float}},
            "SA": {N: {"success_rate": float, "avg_steps_success": float, "avg_time_success": float}},
            "GA": {N: {"success_rate": float, "avg_gen_success": float, "avg_time_success": float, ...}}
        }
    """
    results = {"BT": {}, "SA": {}, "GA": {}}

    for N in N_values:
        print(f"=== (Final) N = {N}, GA fitness {fitness_mode} ===")

        # ----- BACKTRACKING (deterministico) -----
        sol, nodes, t = bt_nqueens_first(N, time_limit=bt_time_limit)
        results["BT"][N] = {
            "solution_found": sol is not None,
            "nodes": nodes,
            "time": t,
        }

        # ----- SIMULATED ANNEALING (stocastico) -----
        sa_runs = []
        max_iter_sa = 2000 + 200 * N  # iterazioni scalabili con N
        
        for _ in range(runs_sa):
            s, steps, tt, bestc, evals = sa_nqueens(
                N, max_iter=max_iter_sa, T0=1.0, alpha=0.995
            )
            sa_runs.append({
                "success": s,
                "steps": steps,
                "time": tt,
                "best_conflicts": bestc,
                "evals": evals,
            })

        # Calcola statistiche aggregate SA
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

        # ----- ALGORITMO GENETICO con parametri ottimali -----
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
            ga_runs.append({
                "success": s,
                "gen": gen,
                "time": tt,
                "best_conflicts": bestc,
                "evals": evals,
            })

        # Calcola statistiche aggregate GA
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
            # Salva anche i parametri GA utilizzati per questo N
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
    
    Parallelizza i run multipli di SA e GA per ottenere speedup significativo.
    BT rimane seriale perché è deterministico (1 sola esecuzione) e già veloce.
    
    Vantaggi parallelizzazione:
    - SA: runs_sa esperimenti indipendenti → speedup ~cores utilizzati
    - GA: runs_ga esperimenti indipendenti → speedup ~cores utilizzati
    - Tempo totale ≈ tempo_bt + max(tempo_sa, tempo_ga) / cores
    
    Args:
        N_values: dimensioni da testare
        runs_sa, runs_ga: numero run per algoritmo
        bt_time_limit: limite tempo BT
        fitness_mode: fitness GA
        best_ga_params_for_N: parametri ottimali dal tuning
        
    Returns:
        dict: stessa struttura della versione sequenziale
    """
    results = {"BT": {}, "SA": {}, "GA": {}}

    for N in N_values:
        print(f"=== (Final Parallel) N = {N}, GA fitness {fitness_mode} ===")

        # ----- BACKTRACKING (sempre seriale, è veloce) -----
        sol, nodes, t = bt_nqueens_first(N, time_limit=bt_time_limit)
        results["BT"][N] = {
            "solution_found": sol is not None,
            "nodes": nodes,
            "time": t,
        }

        # ----- SIMULATED ANNEALING Parallelo -----
        print(f"  Eseguendo {runs_sa} run SA in parallelo...")
        max_iter_sa = 2000 + 200 * N
        
        # Prepara parametri per tutti i run SA
        sa_params = [(N, max_iter_sa, 1.0, 0.995) for _ in range(runs_sa)]
        
        # Esegui tutti i run SA in parallelo
        with ProcessPoolExecutor(max_workers=NUM_PROCESSES) as executor:
            sa_raw_results = list(executor.map(run_single_sa_experiment, sa_params))
        
        # Converte risultati in formato strutturato
        sa_runs = []
        for s, steps, tt, bestc, evals in sa_raw_results:
            sa_runs.append({
                "success": s,
                "steps": steps,
                "time": tt,
                "best_conflicts": bestc,
                "evals": evals,
            })

        # Calcola statistiche aggregate SA
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

        # ----- ALGORITMO GENETICO Parallelo con parametri ottimali -----
        print(f"  Eseguendo {runs_ga} run GA in parallelo...")
        params = best_ga_params_for_N[N]
        pop_size = params["pop_size"]
        max_gen = params["max_gen"]
        pm = params["pm"]
        pc = params["pc"]
        tsize = params["tournament_size"]

        # Prepara parametri per tutti i run GA
        ga_params = [(N, pop_size, max_gen, pc, pm, tsize, fitness_mode) 
                     for _ in range(runs_ga)]
        
        # Esegui tutti i run GA in parallelo
        with ProcessPoolExecutor(max_workers=NUM_PROCESSES) as executor:
            ga_raw_results = list(executor.map(run_single_ga_experiment, ga_params))
        
        # Converte risultati in formato strutturato
        ga_runs = []
        for s, gen, tt, bestc, evals in ga_raw_results:
            ga_runs.append({
                "success": s,
                "gen": gen,
                "time": tt,
                "best_conflicts": bestc,
                "evals": evals,
            })

        # Calcola statistiche aggregate GA
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
            # Salva anche i parametri GA utilizzati
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


def save_bt_sa_results(all_results, N_values, out_dir):
    """
    Salva risultati specifici per BT e SA (indipendenti dalle fitness GA)
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # Raccogli risultati BT e SA da tutti i fitness (sono identici)
    first_fitness = list(all_results.keys())[0]
    bt_results = all_results[first_fitness]["BT"]
    sa_results = all_results[first_fitness]["SA"]
    
    # CSV risultati BT
    bt_csv = os.path.join(out_dir, "results_BT.csv")
    with open(bt_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["N", "solution_found", "nodes", "time_seconds"])
        for N in N_values:
            bt = bt_results[N]
            writer.writerow([
                N,
                int(bt["solution_found"]),
                bt["nodes"],
                bt["time"]
            ])
    print(f"CSV BT salvato: {bt_csv}")
    
    # CSV risultati SA
    sa_csv = os.path.join(out_dir, "results_SA.csv")
    with open(sa_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["N", "success_rate", "avg_steps_success", "avg_time_success"])
        for N in N_values:
            sa = sa_results[N]
            writer.writerow([
                N,
                sa["success_rate"],
                sa["avg_steps_success"] if sa["avg_steps_success"] is not None else "",
                sa["avg_time_success"] if sa["avg_time_success"] is not None else ""
            ])
    print(f"CSV SA salvato: {sa_csv}")


def plot_bt_sa_analysis(all_results, N_values, out_dir):
    """
    Genera grafici di analisi specifica per BT e SA
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # Raccogli dati BT e SA
    first_fitness = list(all_results.keys())[0]
    bt_results = all_results[first_fitness]["BT"]
    sa_results = all_results[first_fitness]["SA"]
    
    # Dati BT
    bt_nodes = [bt_results[N]["nodes"] for N in N_values]
    bt_time = [bt_results[N]["time"] for N in N_values]
    bt_success = [1.0 if bt_results[N]["solution_found"] else 0.0 for N in N_values]
    
    # Dati SA
    sa_success_rate = [sa_results[N]["success_rate"] for N in N_values]
    sa_avg_steps = [sa_results[N]["avg_steps_success"] if sa_results[N]["avg_steps_success"] is not None else 0 for N in N_values]
    sa_avg_time = [sa_results[N]["avg_time_success"] if sa_results[N]["avg_time_success"] is not None else 0 for N in N_values]
    
    # Grafico 1: BT - Nodi esplorati vs N (scala log)
    plt.figure(figsize=(10, 6))
    plt.semilogy(N_values, bt_nodes, marker="o", linewidth=2, markersize=8, color="blue")
    plt.xlabel("N (dimensione scacchiera)")
    plt.ylabel("Nodi esplorati (scala log)")
    plt.title("Backtracking: Complessità vs Dimensione Problema")
    plt.grid(True, alpha=0.7)
    plt.xticks(N_values)
    
    # Aggiungi annotazioni
    for i, (n, nodes) in enumerate(zip(N_values, bt_nodes)):
        plt.annotate(f'{nodes:,}', (n, nodes), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=9)
    
    fname_bt_nodes = os.path.join(out_dir, "BT_nodes_vs_N.png")
    plt.savefig(fname_bt_nodes, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Grafico BT nodi salvato: {fname_bt_nodes}")
    
    # Grafico 2: BT - Tempo vs N
    plt.figure(figsize=(10, 6))
    plt.semilogy(N_values, [t*1000 for t in bt_time], marker="o", linewidth=2, markersize=8, color="blue")
    plt.xlabel("N (dimensione scacchiera)")
    plt.ylabel("Tempo (ms, scala log)")
    plt.title("Backtracking: Tempo di Esecuzione vs Dimensione")
    plt.grid(True, alpha=0.7)
    plt.xticks(N_values)
    
    # Annotazioni tempo
    for i, (n, t) in enumerate(zip(N_values, bt_time)):
        if t < 0.001:
            plt.annotate(f'{t*1000:.2f}ms', (n, t*1000), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=9)
        else:
            plt.annotate(f'{t:.3f}s', (n, t*1000), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=9)
    
    fname_bt_time = os.path.join(out_dir, "BT_time_vs_N.png")
    plt.savefig(fname_bt_time, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Grafico BT tempo salvato: {fname_bt_time}")
    
    # Grafico 3: SA - Success rate e iterazioni
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Success rate SA
    ax1.plot(N_values, sa_success_rate, marker="o", linewidth=2, markersize=8, color="red")
    ax1.set_xlabel("N (dimensione scacchiera)")
    ax1.set_ylabel("Tasso di Successo")
    ax1.set_title("Simulated Annealing: Tasso di Successo")
    ax1.grid(True, alpha=0.7)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_xticks(N_values)
    
    # Annotazioni success rate
    for n, sr in zip(N_values, sa_success_rate):
        ax1.annotate(f'{sr:.2f}', (n, sr), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=9)
    
    # Iterazioni medie SA
    ax2.plot(N_values, sa_avg_steps, marker="o", linewidth=2, markersize=8, color="red")
    ax2.set_xlabel("N (dimensione scacchiera)")
    ax2.set_ylabel("Iterazioni Medie (successi)")
    ax2.set_title("Simulated Annealing: Iterazioni per Successo")
    ax2.grid(True, alpha=0.7)
    ax2.set_xticks(N_values)
    
    # Annotazioni iterazioni
    for n, steps in zip(N_values, sa_avg_steps):
        if steps > 0:
            ax2.annotate(f'{steps:.0f}', (n, steps), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=9)
    
    fname_sa_analysis = os.path.join(out_dir, "SA_analysis.png")
    plt.savefig(fname_sa_analysis, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Grafico SA analisi salvato: {fname_sa_analysis}")
    
    # Grafico 4: Confronto diretto BT vs SA (tempo)
    plt.figure(figsize=(10, 6))
    plt.semilogy(N_values, [t*1000 for t in bt_time], marker="o", linewidth=2, label="Backtracking", color="blue")
    plt.semilogy(N_values, [t*1000 for t in sa_avg_time], marker="s", linewidth=2, label="Simulated Annealing", color="red")
    plt.xlabel("N (dimensione scacchiera)")
    plt.ylabel("Tempo (ms, scala log)")
    plt.title("Confronto Tempi: Backtracking vs Simulated Annealing")
    plt.legend()
    plt.grid(True, alpha=0.7)
    plt.xticks(N_values)
    
    fname_comparison = os.path.join(out_dir, "BT_vs_SA_time_comparison.png")
    plt.savefig(fname_comparison, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Grafico confronto salvato: {fname_comparison}")


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
    
    print(f"\nTUNING CONTEMPORANEO DI TUTTE LE FITNESS")
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
        print(f"\nTuning contemporaneo per N = {N}")
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
    print(f"\nSalvando file CSV di tuning...")
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
        print(f"  Completato: {tuning_csv}")

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

    # ======================================================
    # FASE 3: ANALISI SPECIFICA BT e SA
    # ======================================================
    print(f"\n" + "="*70)
    print("FASE 3: ANALISI DEDICATA BACKTRACKING e SIMULATED ANNEALING")
    print("="*70)
    
    print(f"Generazione analisi specifiche BT e SA...")
    
    # Raccogli tutti i risultati da una qualsiasi fitness (BT/SA sono identici)
    first_fitness = FITNESS_MODES[0]
    first_results = run_experiments_with_best_ga_parallel(
        N_VALUES,
        runs_sa=RUNS_SA_FINAL,
        runs_ga=RUNS_GA_FINAL,
        bt_time_limit=BT_TIME_LIMIT,
        fitness_mode=first_fitness,
        best_ga_params_for_N=all_best_params[first_fitness],
    )
    
    # Salva risultati e grafici specifici per BT e SA
    save_bt_sa_results({first_fitness: first_results}, N_VALUES, OUT_DIR)
    plot_bt_sa_analysis({first_fitness: first_results}, N_VALUES, OUT_DIR)
    print(f"  Analisi BT/SA completata")

    total_time = time.time() - start_total
    print(f"\nTUNING CONTEMPORANEO COMPLETATO!")
    print(f"Tempo totale: {total_time:.1f}s ({total_time/60:.1f} minuti)")
    print(f"Fitness processate contemporaneamente: {len(FITNESS_MODES)}")
    print(f"CSV generati: 6 GA + 6 tuning + 1 BT + 1 SA = 14 files")
    print(f"Grafici generati: 12 GA (2 per fitness) + 4 BT/SA = 16 files")
    print(f"Processi utilizzati: {NUM_PROCESSES}")


if __name__ == "__main__":
    # Scelta tra le diverse modalità
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--sequential":
        print("Esecuzione in modalità SEQUENZIALE")
        main_sequential()
    elif len(sys.argv) > 1 and sys.argv[1] == "--parallel":
        print("Esecuzione in modalità PARALLELA (vecchia)")
        main_parallel()
    else:
        print("Esecuzione in modalità TUNING CONTEMPORANEO (default)")
        print("   Usa --sequential per la versione sequenziale")
        print("   Usa --parallel per la versione parallela classica")
        main_concurrent_tuning()
