import argparse
import csv
import multiprocessing
import os
import random
import statistics
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config_manager import ConfigManager
from nqueens.backtracking import bt_nqueens_first
from nqueens.genetic import ga_nqueens
from nqueens.simulated_annealing import sa_nqueens

# ======================================================
# FUNZIONI STATISTICHE AVANZATE
# ======================================================

def compute_detailed_statistics(values, label=""):
    """
    Calcola statistiche dettagliate per una lista di valori.
    Gestisce separatamente valori vuoti e restituisce un dizionario completo.
    """
    if not values:
        return {
            'count': 0,
            'mean': None,
            'median': None,
            'std': None,
            'min': None,
            'max': None,
            'q25': None,
            'q75': None,
            'range': None,
        }
    
    sorted_vals = sorted(values)
    n = len(values)
    
    # Statistiche base
    mean_val = statistics.mean(values)
    median_val = statistics.median(values)
    min_val = min(values)
    max_val = max(values)
    range_val = max_val - min_val
    
    # Deviazione standard (usando population std)
    std_val = statistics.pstdev(values) if n > 1 else 0
    
    # Quartili
    q25 = sorted_vals[n // 4] if n >= 4 else min_val
    q75 = sorted_vals[3 * n // 4] if n >= 4 else max_val
    
    return {
        'count': n,
        'mean': mean_val,
        'median': median_val,
        'std': std_val,
        'min': min_val,
        'max': max_val,
        'q25': q25,
        'q75': q75,
        'range': range_val,
    }

def compute_grouped_statistics(results_list, success_key='success'):
    """
    Calcola statistiche separate per successi, fallimenti e timeout.
    
    Args:
        results_list: Lista di dizionari con risultati degli esperimenti
        success_key: Chiave che indica il successo (default: 'success')
    
    Returns:
        Dizionario con statistiche separate per successi, fallimenti e timeout
    """
    successes = [r for r in results_list if r.get(success_key, False)]
    timeouts = [r for r in results_list if r.get('timeout', False)]
    failures = [r for r in results_list if not r.get(success_key, False) and not r.get('timeout', False)]
    
    stats = {
        'total_runs': len(results_list),
        'successes': len(successes),
        'failures': len(failures),
        'timeouts': len(timeouts),
        'success_rate': len(successes) / len(results_list) if results_list else 0,
        'timeout_rate': len(timeouts) / len(results_list) if results_list else 0,
        'failure_rate': len(failures) / len(results_list) if results_list else 0,
    }
    
    # Statistiche per tutti i run
    for metric in ['time', 'steps', 'nodes', 'gen', 'evals', 'best_conflicts']:
        if any(metric in r for r in results_list):
            values = [r[metric] for r in results_list if metric in r]
            stats[f'all_{metric}'] = compute_detailed_statistics(values, f"all_{metric}")
    
    # Statistiche per successi
    for metric in ['time', 'steps', 'nodes', 'gen', 'evals', 'best_conflicts']:
        if any(metric in r for r in successes):
            values = [r[metric] for r in successes if metric in r]
            stats[f'success_{metric}'] = compute_detailed_statistics(values, f"success_{metric}")
    
    # Statistiche per timeout
    for metric in ['time', 'steps', 'nodes', 'gen', 'evals', 'best_conflicts']:
        if any(metric in r for r in timeouts):
            values = [r[metric] for r in timeouts if metric in r]
            stats[f'timeout_{metric}'] = compute_detailed_statistics(values, f"timeout_{metric}")
    
    # Statistiche per fallimenti (esclusi timeout)
    for metric in ['time', 'steps', 'nodes', 'gen', 'evals', 'best_conflicts']:
        if any(metric in r for r in failures):
            values = [r[metric] for r in failures if metric in r]
            stats[f'failure_{metric}'] = compute_detailed_statistics(values, f"failure_{metric}")
    
    return stats


class ProgressPrinter:
    """Lightweight textual progress reporter for long-running loops."""

    def __init__(self, total, label):
        self.total = max(1, total)
        self.label = label

    def update(self, index, detail=""):
        percent = (index / self.total) * 100
        suffix = f" - {detail}" if detail else ""
        print(f"[{self.label}] {index}/{self.total} ({percent:.0f}%)" + suffix)


def parse_fitness_filters(fitness_args):
    """Normalize fitness CLI arguments into an uppercase list."""
    if not fitness_args:
        return None

    selected = []
    for entry in fitness_args:
        for token in entry.split(','):
            token = token.strip().upper()
            if token:
                selected.append(token)

    return selected or None


def normalize_optimal_parameters(raw_params):
    """Convert JSON-loaded optimal parameters into an int-keyed mapping."""
    normalized = {}
    if not raw_params:
        return normalized

    for key, value in raw_params.items():
        try:
            normalized[int(key)] = value
        except (TypeError, ValueError):
            normalized[key] = value

    return normalized


def ensure_parameters_for_all_n(params, n_values, fitness_mode):
    """Ensure optimal GA parameters are available for every requested N."""
    missing = [n for n in n_values if n not in params]
    if missing:
        missing_str = ', '.join(str(n) for n in missing)
        raise ValueError(
            f"Missing GA parameters for fitness {fitness_mode} and N values: {missing_str}. "
            "Run tuning or update config.json."
        )


def apply_configuration(config_path, fitness_filter=None):
    """Load configuration from JSON and apply overrides to module globals."""

    config_mgr = ConfigManager(config_path)

    experiment_settings = config_mgr.get_experiment_settings()
    if experiment_settings:
        global N_VALUES, RUNS_SA_FINAL, RUNS_GA_FINAL, RUNS_BT_FINAL, RUNS_GA_TUNING, OUT_DIR

        N_values_cfg = experiment_settings.get("N_values", N_VALUES)
        N_VALUES = [int(n) for n in N_values_cfg]
        RUNS_SA_FINAL = int(experiment_settings.get("runs_sa_final", RUNS_SA_FINAL))
        RUNS_GA_FINAL = int(experiment_settings.get("runs_ga_final", RUNS_GA_FINAL))
        RUNS_BT_FINAL = int(experiment_settings.get("runs_bt_final", RUNS_BT_FINAL))
        RUNS_GA_TUNING = int(experiment_settings.get("runs_ga_tuning", RUNS_GA_TUNING))
        OUT_DIR = experiment_settings.get("output_dir", OUT_DIR)

    timeout_settings = config_mgr.get_timeout_settings()
    if timeout_settings:
        set_timeouts(
            bt_timeout=timeout_settings.get("bt_time_limit", BT_TIME_LIMIT),
            sa_timeout=timeout_settings.get("sa_time_limit", SA_TIME_LIMIT),
            ga_timeout=timeout_settings.get("ga_time_limit", GA_TIME_LIMIT),
            experiment_timeout=timeout_settings.get("experiment_timeout", EXPERIMENT_TIMEOUT),
        )

    tuning_grid = config_mgr.get_tuning_grid()
    if tuning_grid:
        global POP_MULTIPLIERS, GEN_MULTIPLIERS, PM_VALUES, PC_FIXED, TOURNAMENT_SIZE_FIXED

        POP_MULTIPLIERS = [int(v) for v in tuning_grid.get("pop_multipliers", POP_MULTIPLIERS)]
        GEN_MULTIPLIERS = [int(v) for v in tuning_grid.get("gen_multipliers", GEN_MULTIPLIERS)]
        PM_VALUES = [float(v) for v in tuning_grid.get("pm_values", PM_VALUES)]
        PC_FIXED = float(tuning_grid.get("pc_fixed", PC_FIXED))
        TOURNAMENT_SIZE_FIXED = int(tuning_grid.get("tournament_size_fixed", TOURNAMENT_SIZE_FIXED))

    fitness_modes_cfg = [mode.upper() for mode in config_mgr.get_fitness_modes()]
    if not fitness_modes_cfg:
        fitness_modes_cfg = ["F1"]

    if fitness_filter:
        requested = {mode.upper() for mode in fitness_filter}
        unknown = requested.difference(set(fitness_modes_cfg))
        if unknown:
            raise ValueError(
                "Unknown fitness modes requested: " + ', '.join(sorted(unknown))
            )
        selected_modes = [mode for mode in fitness_modes_cfg if mode in requested]
    else:
        selected_modes = fitness_modes_cfg

    if not selected_modes:
        raise ValueError("No fitness modes selected after applying filters.")

    global FITNESS_MODES
    FITNESS_MODES = selected_modes

    return config_mgr, selected_modes


def load_optimal_parameters(fitness_mode, config_mgr, n_values):
    """Fetch and validate optimal GA parameters from the configuration file."""
    if config_mgr is None:
        raise ValueError("Config manager is required when skip-tuning is enabled.")

    params = normalize_optimal_parameters(config_mgr.get_optimal_parameters(fitness_mode))
    ensure_parameters_for_all_n(params, n_values, fitness_mode)
    return params

# ======================================================
# PARAMETRI GLOBALI
# ======================================================

# Dimensioni della scacchiera da testare - valori crescenti per analisi scalabilità
N_VALUES = [8, 16, 24, 40, 80, 120]

# Numero di run indipendenti per SA e GA negli esperimenti finali
# Più run = maggiore affidabilità statistica, ma tempi più lunghi
RUNS_SA_FINAL = 40
RUNS_GA_FINAL = 40
RUNS_BT_FINAL = 1   # BT è deterministico, basta 1 run per N

# Numero di run per la fase di tuning GA (per combinazione di parametri)
# Meno run nel tuning per velocizzare la ricerca parametri
RUNS_GA_TUNING = 5

# Limite di tempo per BT in secondi (None = nessun limite)
# Utile per evitare che BT rimanga bloccato su istanze difficili
BT_TIME_LIMIT = 60*5.0  # es. 5.0 minuti

# Limiti di tempo per SA e GA in secondi (None = nessun limite)
SA_TIME_LIMIT = 120.0  # 120 secondi per SA
GA_TIME_LIMIT = 240.0  # 240 secondi per GA 

# Timeout globale per singolo esperimento (None = nessun limite)  
EXPERIMENT_TIMEOUT = 120.0  # 2 minuti per esperimento completo

# Funzione per configurare facilmente i timeout
def set_timeouts(bt_timeout=None, sa_timeout=30.0, ga_timeout=60.0, experiment_timeout=120.0):
    """
    Configura i timeout per tutti gli algoritmi.
    
    Args:
        bt_timeout: timeout BT in secondi (None = illimitato)
        sa_timeout: timeout SA in secondi (None = illimitato)  
        ga_timeout: timeout GA in secondi (None = illimitato)
        experiment_timeout: timeout esperimento in secondi (None = illimitato)
    """
    global BT_TIME_LIMIT, SA_TIME_LIMIT, GA_TIME_LIMIT, EXPERIMENT_TIMEOUT
    BT_TIME_LIMIT = bt_timeout
    SA_TIME_LIMIT = sa_timeout
    GA_TIME_LIMIT = ga_timeout
    EXPERIMENT_TIMEOUT = experiment_timeout
    
    print("Timeout settings configured:")
    print(f"   - BT: {BT_TIME_LIMIT}s" if BT_TIME_LIMIT else "   - BT: unlimited")
    print(f"   - SA: {SA_TIME_LIMIT}s" if SA_TIME_LIMIT else "   - SA: unlimited")
    print(f"   - GA: {GA_TIME_LIMIT}s" if GA_TIME_LIMIT else "   - GA: unlimited")
    print(f"   - Experiment: {EXPERIMENT_TIMEOUT}s" if EXPERIMENT_TIMEOUT else "   - Experiment: unlimited")

# Directory di output per CSV e grafici
OUT_DIR = "results_nqueens_tuning"

# Griglia di tuning per il GA - definisce lo spazio di ricerca parametri
POP_MULTIPLIERS = [4, 8, 16]       # pop_size ≈ 4N, 8N, 16N - popolazione scala con N
GEN_MULTIPLIERS = [30, 50, 80]     # max_gen ≈ 30N, 50N, 80N - generazioni scala con N
PM_VALUES = [0.05, 0.1, 0.15]        # probabilità mutazione - range tipico per GA
PC_FIXED = 0.8                     # probabilità crossover fissa (valore standard)
TOURNAMENT_SIZE_FIXED = 3          # dimensione torneo per selezione

# Tutte le funzioni di fitness da testare (F1-F6)
FITNESS_MODES = ["F1", "F2", "F3", "F4", "F5", "F6"]

# Numero di processi per il parallelismo
# Lascia un core libero per il sistema operativo
NUM_PROCESSES = max(1, multiprocessing.cpu_count() - 1)


# ======================================================
# GA: Tuning dei parametri per un singolo (N, fitness_mode)
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
                    s, gen, _, bestc, _, timeout = ga_nqueens(
                        N,
                        pop_size=pop_size,
                        max_gen=max_gen,
                        pc=pc,
                        pm=pm,
                        tournament_size=tournament_size,
                        fitness_mode=fitness_mode,
                        time_limit=GA_TIME_LIMIT,
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
# GA: Funzioni di supporto per la parallelizzazione
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
        time_limit=GA_TIME_LIMIT,
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
    return sa_nqueens(N, max_iter=max_iter, T0=T0, alpha=alpha, time_limit=SA_TIME_LIMIT)


def run_with_timeout(func, args, timeout):
    """
    Esegue una funzione con timeout usando ProcessPoolExecutor.
    
    Args:
        func: funzione da eseguire
        args: argomenti della funzione
        timeout: timeout in secondi
        
    Returns:
        tuple: (success, result) - success=True se completato, result=valore o None
    """
    try:
        with ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func, args)
            try:
                result = future.result(timeout=timeout)
            except KeyboardInterrupt:
                executor.shutdown(cancel_futures=True)
                raise
            return True, result
    except Exception as e:
        print(f"WARNING: Timeout or error during execution: {e}")
        return False, None


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
        try:
            results = list(executor.map(run_single_ga_experiment, run_params))
        except KeyboardInterrupt:
            executor.shutdown(cancel_futures=True)
            raise
    
    # Calcola statistiche aggregate
    successes = 0
    gen_success = []
    for s, gen, _, bestc, _, _ in results:  # Aggiunto _ per timeout
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
        try:
            candidates = list(executor.map(test_parameter_combination_parallel, param_combinations))
        except KeyboardInterrupt:
            executor.shutdown(cancel_futures=True)
            raise
    
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
    start_time = perf_counter()
    
    with ProcessPoolExecutor(max_workers=min(NUM_PROCESSES, len(fitness_modes))) as executor:
        try:
            results = list(executor.map(tune_single_fitness, tuning_params))
        except KeyboardInterrupt:
            executor.shutdown(cancel_futures=True)
            raise
    
    elapsed_time = perf_counter() - start_time
    
    # Organizza risultati per fitness
    best_params_per_fitness = {}
    for fitness_mode, best_params in results:
        best_params_per_fitness[fitness_mode] = best_params
        print(f"  Completato {fitness_mode}: success_rate={best_params['success_rate']:.3f}, "
              f"pop_size={best_params['pop_size']}, pm={best_params['pm']}")
    
    print(f"Tuning contemporaneo completato in {elapsed_time:.1f}s per N={N}")
    return best_params_per_fitness


# ======================================================
# Esperimenti finali con parametri GA ottimizzati
# ======================================================

def run_experiments_with_best_ga(
    N_values,
    runs_sa,
    runs_ga,
    bt_time_limit,
    fitness_mode,
    best_ga_params_for_N,
    progress_label=None,
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

    progress = ProgressPrinter(len(N_values), progress_label) if progress_label else None

    for index, N in enumerate(N_values, start=1):
        if progress:
            progress.update(index, f"N={N}")
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
            s, steps, tt, bestc, evals, timeout = sa_nqueens(
                N, max_iter=max_iter_sa, T0=1.0, alpha=0.995, time_limit=SA_TIME_LIMIT
            )
            sa_runs.append({
                "success": s,
                "steps": steps,
                "time": tt,
                "best_conflicts": bestc,
                "evals": evals,
                "timeout": timeout,
            })

        # Calcola statistiche aggregate SA con funzione avanzata
        sa_stats = compute_grouped_statistics(sa_runs, 'success')

        results["SA"][N] = {
            "success_rate": sa_stats['success_rate'],
            "timeout_rate": sa_stats['timeout_rate'],
            "failure_rate": sa_stats['failure_rate'],
            "total_runs": sa_stats['total_runs'],
            "successes": sa_stats['successes'],
            "failures": sa_stats['failures'],
            "timeouts": sa_stats['timeouts'],
            
            # Statistiche complete per successi
            "success_steps": sa_stats.get('success_steps', {}),
            "success_time": sa_stats.get('success_time', {}),
            "success_evals": sa_stats.get('success_evals', {}),
            "success_best_conflicts": sa_stats.get('success_best_conflicts', {}),
            
            # Statistiche complete per timeout
            "timeout_steps": sa_stats.get('timeout_steps', {}),
            "timeout_time": sa_stats.get('timeout_time', {}),
            "timeout_evals": sa_stats.get('timeout_evals', {}),
            "timeout_best_conflicts": sa_stats.get('timeout_best_conflicts', {}),
            
            # Statistiche complete per fallimenti  
            "failure_steps": sa_stats.get('failure_steps', {}),
            "failure_time": sa_stats.get('failure_time', {}),
            "failure_evals": sa_stats.get('failure_evals', {}),
            "failure_best_conflicts": sa_stats.get('failure_best_conflicts', {}),
            
            # Statistiche per tutti i run
            "all_steps": sa_stats.get('all_steps', {}),
            "all_time": sa_stats.get('all_time', {}),
            "all_evals": sa_stats.get('all_evals', {}),
            "all_best_conflicts": sa_stats.get('all_best_conflicts', {}),
            
            # Dati grezzi per analisi dettagliate
            "raw_runs": sa_runs.copy(),
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
            s, gen, tt, bestc, evals, timeout = ga_nqueens(
                N,
                pop_size=pop_size,
                max_gen=max_gen,
                pc=pc,
                pm=pm,
                tournament_size=tsize,
                fitness_mode=fitness_mode,
                time_limit=GA_TIME_LIMIT,
            )
            ga_runs.append({
                "success": s,
                "gen": gen,
                "time": tt,
                "best_conflicts": bestc,
                "evals": evals,
                "timeout": timeout,
            })

        # Calcola statistiche aggregate GA con funzione avanzata
        ga_stats = compute_grouped_statistics(ga_runs, 'success')

        results["GA"][N] = {
            "success_rate": ga_stats['success_rate'],
            "timeout_rate": ga_stats['timeout_rate'],
            "failure_rate": ga_stats['failure_rate'],
            "total_runs": ga_stats['total_runs'],
            "successes": ga_stats['successes'],
            "failures": ga_stats['failures'],
            "timeouts": ga_stats['timeouts'],
            
            # Statistiche complete per successi
            "success_gen": ga_stats.get('success_gen', {}),
            "success_time": ga_stats.get('success_time', {}),
            "success_evals": ga_stats.get('success_evals', {}),
            "success_best_conflicts": ga_stats.get('success_best_conflicts', {}),
            
            # Statistiche complete per timeout
            "timeout_gen": ga_stats.get('timeout_gen', {}),
            "timeout_time": ga_stats.get('timeout_time', {}),
            "timeout_evals": ga_stats.get('timeout_evals', {}),
            "timeout_best_conflicts": ga_stats.get('timeout_best_conflicts', {}),
            
            # Statistiche complete per fallimenti  
            "failure_gen": ga_stats.get('failure_gen', {}),
            "failure_time": ga_stats.get('failure_time', {}),
            "failure_evals": ga_stats.get('failure_evals', {}),
            "failure_best_conflicts": ga_stats.get('failure_best_conflicts', {}),
            
            # Statistiche per tutti i run
            "all_gen": ga_stats.get('all_gen', {}),
            "all_time": ga_stats.get('all_time', {}),
            "all_evals": ga_stats.get('all_evals', {}),
            "all_best_conflicts": ga_stats.get('all_best_conflicts', {}),
            
            # Salva anche i parametri GA utilizzati per questo N
            "pop_size": pop_size,
            "max_gen": max_gen,
            "pm": pm,
            "pc": pc,
            "tournament_size": tsize,
            
            # Dati grezzi per analisi dettagliate
            "raw_runs": ga_runs.copy(),
        }

    return results


def run_experiments_with_best_ga_parallel(
    N_values,
    runs_sa,
    runs_ga,
    bt_time_limit,
    fitness_mode,
    best_ga_params_for_N,
    progress_label=None,
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

    progress = ProgressPrinter(len(N_values), progress_label) if progress_label else None

    for index, N in enumerate(N_values, start=1):
        if progress:
            progress.update(index, f"N={N}")
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
            try:
                sa_raw_results = list(executor.map(run_single_sa_experiment, sa_params))
            except KeyboardInterrupt:
                executor.shutdown(cancel_futures=True)
                raise
        
        # Converte risultati in formato strutturato
        sa_runs = []
        for s, steps, tt, bestc, evals, timeout in sa_raw_results:
            sa_runs.append({
                "success": s,
                "steps": steps,
                "time": tt,
                "best_conflicts": bestc,
                "evals": evals,
                "timeout": timeout,
            })

        # Calcola statistiche aggregate SA con funzione avanzata
        sa_stats = compute_grouped_statistics(sa_runs, 'success')

        results["SA"][N] = {
            "success_rate": sa_stats['success_rate'],
            "timeout_rate": sa_stats['timeout_rate'],
            "failure_rate": sa_stats['failure_rate'],
            "total_runs": sa_stats['total_runs'],
            "successes": sa_stats['successes'],
            "failures": sa_stats['failures'],
            "timeouts": sa_stats['timeouts'],
            
            # Statistiche complete per successi
            "success_steps": sa_stats.get('success_steps', {}),
            "success_time": sa_stats.get('success_time', {}),
            "success_evals": sa_stats.get('success_evals', {}),
            "success_best_conflicts": sa_stats.get('success_best_conflicts', {}),
            
            # Statistiche complete per timeout
            "timeout_steps": sa_stats.get('timeout_steps', {}),
            "timeout_time": sa_stats.get('timeout_time', {}),
            "timeout_evals": sa_stats.get('timeout_evals', {}),
            "timeout_best_conflicts": sa_stats.get('timeout_best_conflicts', {}),
            
            # Statistiche complete per fallimenti  
            "failure_steps": sa_stats.get('failure_steps', {}),
            "failure_time": sa_stats.get('failure_time', {}),
            "failure_evals": sa_stats.get('failure_evals', {}),
            "failure_best_conflicts": sa_stats.get('failure_best_conflicts', {}),
            
            # Statistiche per tutti i run
            "all_steps": sa_stats.get('all_steps', {}),
            "all_time": sa_stats.get('all_time', {}),
            "all_evals": sa_stats.get('all_evals', {}),
            "all_best_conflicts": sa_stats.get('all_best_conflicts', {}),
            
            # Dati grezzi per analisi dettagliate
            "raw_runs": sa_runs.copy(),
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
            try:
                ga_raw_results = list(executor.map(run_single_ga_experiment, ga_params))
            except KeyboardInterrupt:
                executor.shutdown(cancel_futures=True)
                raise
        
        # Converte risultati in formato strutturato
        ga_runs = []
        for s, gen, tt, bestc, evals, timeout in ga_raw_results:
            ga_runs.append({
                "success": s,
                "gen": gen,
                "time": tt,
                "best_conflicts": bestc,
                "evals": evals,
                "timeout": timeout,
            })

        # Calcola statistiche aggregate GA con funzione avanzata
        ga_stats = compute_grouped_statistics(ga_runs, 'success')

        results["GA"][N] = {
            "success_rate": ga_stats['success_rate'],
            "timeout_rate": ga_stats['timeout_rate'],
            "failure_rate": ga_stats['failure_rate'],
            "total_runs": ga_stats['total_runs'],
            "successes": ga_stats['successes'],
            "failures": ga_stats['failures'],
            "timeouts": ga_stats['timeouts'],
            
            # Statistiche complete per successi
            "success_gen": ga_stats.get('success_gen', {}),
            "success_time": ga_stats.get('success_time', {}),
            "success_evals": ga_stats.get('success_evals', {}),
            "success_best_conflicts": ga_stats.get('success_best_conflicts', {}),
            
            # Statistiche complete per timeout
            "timeout_gen": ga_stats.get('timeout_gen', {}),
            "timeout_time": ga_stats.get('timeout_time', {}),
            "timeout_evals": ga_stats.get('timeout_evals', {}),
            "timeout_best_conflicts": ga_stats.get('timeout_best_conflicts', {}),
            
            # Statistiche complete per fallimenti  
            "failure_gen": ga_stats.get('failure_gen', {}),
            "failure_time": ga_stats.get('failure_time', {}),
            "failure_evals": ga_stats.get('failure_evals', {}),
            "failure_best_conflicts": ga_stats.get('failure_best_conflicts', {}),
            
            # Statistiche per tutti i run
            "all_gen": ga_stats.get('all_gen', {}),
            "all_time": ga_stats.get('all_time', {}),
            "all_evals": ga_stats.get('all_evals', {}),
            "all_best_conflicts": ga_stats.get('all_best_conflicts', {}),
            
            # Salva anche i parametri GA utilizzati
            "pop_size": pop_size,
            "max_gen": max_gen,
            "pm": pm,
            "pc": pc,
            "tournament_size": tsize,
            
            # Dati grezzi per analisi dettagliate
            "raw_runs": ga_runs.copy(),
        }

    return results


# ======================================================
# Salvataggio CSV e grafici
# ======================================================

def save_results_to_csv(results, N_values, fitness_mode, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    filename = os.path.join(out_dir, f"results_GA_{fitness_mode}_tuned.csv")

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "N",
            # BT metriche logiche e temporali
            "BT_solution_found",
            "BT_nodes_explored",
            "BT_time_seconds",
            
            # SA metriche statistiche complete
            "SA_success_rate",
            "SA_timeout_rate", 
            "SA_failure_rate",
            "SA_total_runs",
            "SA_successes",
            "SA_failures",
            "SA_timeouts",
            
            # SA metriche logiche per successi
            "SA_success_steps_mean",
            "SA_success_steps_median", 
            "SA_success_evals_mean",
            "SA_success_evals_median",
            
            # SA metriche logiche per timeout
            "SA_timeout_steps_mean",
            "SA_timeout_steps_median",
            "SA_timeout_evals_mean", 
            "SA_timeout_evals_median",
            
            # SA metriche temporali per successi
            "SA_success_time_mean",
            "SA_success_time_median",
            
            # GA metriche statistiche complete  
            "GA_success_rate",
            "GA_timeout_rate",
            "GA_failure_rate", 
            "GA_total_runs",
            "GA_successes", 
            "GA_failures",
            "GA_timeouts",
            
            # GA metriche logiche per successi
            "GA_success_gen_mean",
            "GA_success_gen_median",
            "GA_success_evals_mean", 
            "GA_success_evals_median",
            
            # GA metriche logiche per timeout
            "GA_timeout_gen_mean",
            "GA_timeout_gen_median",
            "GA_timeout_evals_mean",
            "GA_timeout_evals_median",
            
            # GA metriche temporali per successi
            "GA_success_time_mean",
            "GA_success_time_median",
            
            # GA parametri utilizzati
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
            
            # Estrae statistiche SA
            sa_steps_success = sa.get("success_steps", {})
            sa_evals_success = sa.get("success_evals", {})
            sa_time_success = sa.get("success_time", {})
            sa_steps_timeout = sa.get("timeout_steps", {})
            sa_evals_timeout = sa.get("timeout_evals", {})
            
            # Estrae statistiche GA
            ga_gen_success = ga.get("success_gen", {})
            ga_evals_success = ga.get("success_evals", {})
            ga_time_success = ga.get("success_time", {})
            ga_gen_timeout = ga.get("timeout_gen", {})
            ga_evals_timeout = ga.get("timeout_evals", {})
            
            writer.writerow([
                N,
                # BT
                int(bt["solution_found"]),
                bt["nodes"],
                bt["time"],
                
                # SA statistiche generali
                sa["success_rate"],
                sa.get("timeout_rate", 0),
                sa.get("failure_rate", 0),
                sa["total_runs"],
                sa["successes"], 
                sa["failures"],
                sa.get("timeouts", 0),
                
                # SA metriche logiche successi
                sa_steps_success.get("mean", ""),
                sa_steps_success.get("median", ""),
                sa_evals_success.get("mean", ""),
                sa_evals_success.get("median", ""),
                
                # SA metriche logiche timeout
                sa_steps_timeout.get("mean", ""),
                sa_steps_timeout.get("median", ""),
                sa_evals_timeout.get("mean", ""),
                sa_evals_timeout.get("median", ""),
                
                # SA metriche temporali successi
                sa_time_success.get("mean", ""),
                sa_time_success.get("median", ""),
                
                # GA statistiche generali
                ga["success_rate"],
                ga.get("timeout_rate", 0),
                ga.get("failure_rate", 0),
                ga["total_runs"],
                ga["successes"],
                ga["failures"],
                ga.get("timeouts", 0),
                
                # GA metriche logiche successi
                ga_gen_success.get("mean", ""),
                ga_gen_success.get("median", ""),
                ga_evals_success.get("mean", ""),
                ga_evals_success.get("median", ""),
                
                # GA metriche logiche timeout
                ga_gen_timeout.get("mean", ""),
                ga_gen_timeout.get("median", ""),
                ga_evals_timeout.get("mean", ""),
                ga_evals_timeout.get("median", ""),
                
                # GA metriche temporali successi
                ga_time_success.get("mean", ""),
                ga_time_success.get("median", ""),
                
                # GA parametri
                ga["pop_size"],
                ga["max_gen"],
                ga["pm"],
                ga["pc"],
                ga["tournament_size"],
            ])

    print(f"CSV salvato: {filename}")


def save_raw_data_to_csv(results, N_values, fitness_mode, out_dir):
    """Salva i dati grezzi di tutti i run individuali per analisi dettagliate"""
    os.makedirs(out_dir, exist_ok=True)
    
    # File per dati grezzi SA
    sa_filename = os.path.join(out_dir, f"raw_data_SA_{fitness_mode}.csv")
    with open(sa_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "N", "run_id", "algorithm", "success", "timeout",
            "steps", "time_seconds", "evals", "best_conflicts"
        ])
        
        for N in N_values:
            sa_data = results["SA"][N]
            if "raw_runs" in sa_data:
                for i, run in enumerate(sa_data["raw_runs"]):
                    writer.writerow([
                        N, i+1, "SA", run["success"], run["timeout"],
                        run["steps"], run["time"], run["evals"], run["best_conflicts"]
                    ])
    
    # File per dati grezzi GA  
    ga_filename = os.path.join(out_dir, f"raw_data_GA_{fitness_mode}.csv")
    with open(ga_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "N", "run_id", "algorithm", "success", "timeout",
            "gen", "time_seconds", "evals", "best_fitness", "best_conflicts",
            "pop_size", "max_gen", "pm", "pc", "tournament_size"
        ])
        
        for N in N_values:
            ga_data = results["GA"][N]
            if "raw_runs" in ga_data:
                for i, run in enumerate(ga_data["raw_runs"]):
                    writer.writerow([
                        N, i+1, "GA", run["success"], run["timeout"],
                        run["gen"], run["time"], run["evals"], 
                        run.get("best_fitness", ""), run.get("best_conflicts", ""),
                        ga_data["pop_size"], ga_data["max_gen"], 
                        ga_data["pm"], ga_data["pc"], ga_data["tournament_size"]
                    ])
    
    # File per dati grezzi BT (uno per N)
    bt_filename = os.path.join(out_dir, f"raw_data_BT_{fitness_mode}.csv") 
    with open(bt_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "N", "algorithm", "solution_found", "nodes_explored", "time_seconds"
        ])
        
        for N in N_values:
            bt_data = results["BT"][N]
            writer.writerow([
                N, "BT", bt_data["solution_found"], bt_data["nodes"], bt_data["time"]
            ])
    
    print(f"Dati grezzi salvati:")
    print(f"  SA: {sa_filename}")
    print(f"  GA: {ga_filename}")  
    print(f"  BT: {bt_filename}")


def save_logical_cost_analysis(results, N_values, fitness_mode, out_dir):
    """Salva analisi focalizzata sui costi logici indipendenti dalla macchina"""
    os.makedirs(out_dir, exist_ok=True)
    filename = os.path.join(out_dir, f"logical_costs_{fitness_mode}.csv")
    
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "N",
            # BT - costo logico
            "BT_solution_found",
            "BT_nodes_explored",  # costo logico primario
            
            # SA - costi logici  
            "SA_success_rate",
            "SA_steps_mean_all",        # costo logico primario
            "SA_steps_median_all",
            "SA_evals_mean_all",        # costo logico secondario
            "SA_evals_median_all",
            "SA_steps_mean_success",    # per successi
            "SA_evals_mean_success", 
            
            # GA - costi logici
            "GA_success_rate", 
            "GA_gen_mean_all",          # costo logico primario
            "GA_gen_median_all",
            "GA_evals_mean_all",        # costo logico secondario
            "GA_evals_median_all",
            "GA_gen_mean_success",      # per successi
            "GA_evals_mean_success",
            
            # Tempi come riferimento sperimentale
            "BT_time_seconds",
            "SA_time_mean_success",
            "GA_time_mean_success",
        ])
        
        for N in N_values:
            bt = results["BT"][N]
            sa = results["SA"][N]
            ga = results["GA"][N]
            
            # Estrae metriche logiche SA
            sa_all_steps = sa.get("all_steps", {})
            sa_all_evals = sa.get("all_evals", {})
            sa_success_steps = sa.get("success_steps", {})
            sa_success_evals = sa.get("success_evals", {})
            sa_success_time = sa.get("success_time", {})
            
            # Estrae metriche logiche GA  
            ga_all_gen = ga.get("all_gen", {})
            ga_all_evals = ga.get("all_evals", {})
            ga_success_gen = ga.get("success_gen", {})
            ga_success_evals = ga.get("success_evals", {})
            ga_success_time = ga.get("success_time", {})
            
            writer.writerow([
                N,
                # BT
                int(bt["solution_found"]),
                bt["nodes"],
                
                # SA costi logici
                sa["success_rate"],
                sa_all_steps.get("mean", ""),
                sa_all_steps.get("median", ""),
                sa_all_evals.get("mean", ""), 
                sa_all_evals.get("median", ""),
                sa_success_steps.get("mean", ""),
                sa_success_evals.get("mean", ""),
                
                # GA costi logici
                ga["success_rate"],
                ga_all_gen.get("mean", ""),
                ga_all_gen.get("median", ""),
                ga_all_evals.get("mean", ""),
                ga_all_evals.get("median", ""),
                ga_success_gen.get("mean", ""),
                ga_success_evals.get("mean", ""),
                
                # Tempi sperimentali
                bt["time"],
                sa_success_time.get("mean", ""),
                ga_success_time.get("mean", ""),
            ])
    
    print(f"Analisi costi logici salvata: {filename}")


def plot_comprehensive_analysis(results, N_values, fitness_mode, out_dir, raw_runs=None, tuning_data=None):
    """
    Genera tutti i grafici richiesti per l'analisi completa degli algoritmi N-Queens
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # ===========================================
    # 1. GRAFICI BASE 
    # ===========================================
    
    # Estrai dati base per tutti gli algoritmi
    bt_sr = [1.0 if results["BT"][N]["solution_found"] else 0.0 for N in N_values]
    sa_sr = [results["SA"][N]["success_rate"] for N in N_values]
    ga_sr = [results["GA"][N]["success_rate"] for N in N_values]
    
    bt_timeout = [0.0 for N in N_values]  # BT non ha timeout nei dati attuali
    sa_timeout = [results["SA"][N].get("timeout_rate", 0.0) for N in N_values]
    ga_timeout = [results["GA"][N].get("timeout_rate", 0.0) for N in N_values]
    
    # 1.1 Tasso di successo vs N
    plt.figure(figsize=(12, 8))
    plt.plot(N_values, bt_sr, marker="o", linewidth=2, markersize=8, label="Backtracking")
    plt.plot(N_values, sa_sr, marker="s", linewidth=2, markersize=8, label="Simulated Annealing") 
    plt.plot(N_values, ga_sr, marker="^", linewidth=2, markersize=8, label=f"Genetic Algorithm (F{fitness_mode})")
    plt.xlabel("N (Dimensione scacchiera)", fontsize=12)
    plt.ylabel("Tasso di Successo", fontsize=12)
    plt.title("Tasso di Successo vs Dimensione Problema\n(Mostra affidabilità algoritmi al crescere di N)", fontsize=14)
    plt.ylim(-0.05, 1.05)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.7)
    plt.xticks(N_values)
    # Aggiungi annotazioni
    for i, n in enumerate(N_values):
        plt.annotate(f'{bt_sr[i]:.1f}', (n, bt_sr[i]), textcoords="offset points", xytext=(0,5), ha='center', fontsize=9, color='blue')
        plt.annotate(f'{sa_sr[i]:.2f}', (n, sa_sr[i]), textcoords="offset points", xytext=(0,-15), ha='center', fontsize=9, color='orange')
        plt.annotate(f'{ga_sr[i]:.2f}', (n, ga_sr[i]), textcoords="offset points", xytext=(0,5), ha='center', fontsize=9, color='green')
    
    fname = os.path.join(out_dir, f"01_success_rate_vs_N_F{fitness_mode}.png")
    plt.savefig(fname, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved success-rate chart: {fname}")
    
    # 1.2 Tempo medio vs N (scala logaritmica, solo successi)
    bt_time = [results["BT"][N]["time"] if results["BT"][N]["solution_found"] else 0 for N in N_values]
    sa_time = [results["SA"][N]["success_time"].get("mean", 0.0) if results["SA"][N]["success_time"] else 0.0 for N in N_values]
    ga_time = [results["GA"][N]["success_time"].get("mean", 0.0) if results["GA"][N]["success_time"] else 0.0 for N in N_values]
    
    plt.figure(figsize=(12, 8))
    # Filtra valori zero per la scala log
    bt_time_plot = [max(t, 1e-6) for t in bt_time]
    sa_time_plot = [max(t, 1e-6) for t in sa_time]  
    ga_time_plot = [max(t, 1e-6) for t in ga_time]
    
    plt.semilogy(N_values, bt_time_plot, marker="o", linewidth=2, markersize=8, label="Backtracking")
    plt.semilogy(N_values, sa_time_plot, marker="s", linewidth=2, markersize=8, label="Simulated Annealing")
    plt.semilogy(N_values, ga_time_plot, marker="^", linewidth=2, markersize=8, label=f"Genetic Algorithm (F{fitness_mode})")
    plt.xlabel("N (Dimensione scacchiera)", fontsize=12)
    plt.ylabel("Tempo Medio [s] (scala log)", fontsize=12)
    plt.title("Tempo di Esecuzione vs Dimensione Problema\n(Solo run di successo - evidenzia esplosione computazionale BT)", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.7)
    plt.xticks(N_values)
    
    fname = os.path.join(out_dir, f"02_time_vs_N_log_scale_F{fitness_mode}.png")
    plt.savefig(fname, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved execution-time chart (log scale): {fname}")
    
    # 1.3 Costo logico vs N (indipendente dalla macchina)
    bt_nodes = [results["BT"][N]["nodes"] for N in N_values]
    sa_steps = [results["SA"][N]["success_steps"].get("mean", 0.0) if results["SA"][N]["success_steps"] else 0.0 for N in N_values]
    ga_gen = [results["GA"][N]["success_gen"].get("mean", 0.0) if results["GA"][N]["success_gen"] else 0.0 for N in N_values]
    
    plt.figure(figsize=(12, 8))
    plt.semilogy(N_values, [max(n, 1) for n in bt_nodes], marker="o", linewidth=2, markersize=8, label="BT: Nodi esplorati")
    plt.semilogy(N_values, [max(s, 1) for s in sa_steps], marker="s", linewidth=2, markersize=8, label="SA: Iterazioni medie")
    plt.semilogy(N_values, [max(g, 1) for g in ga_gen], marker="^", linewidth=2, markersize=8, label="GA: Generazioni medie")
    plt.xlabel("N (Dimensione scacchiera)", fontsize=12)
    plt.ylabel("Costo Logico (scala log)", fontsize=12)
    plt.title("Costo Computazionale Teorico vs Dimensione\n(Scalabilità indipendente dall'hardware)", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.7)
    plt.xticks(N_values)
    
    fname = os.path.join(out_dir, f"03_logical_cost_vs_N_F{fitness_mode}.png")
    plt.savefig(fname, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved logical-cost chart: {fname}")
    
    # 1.4 Valutazioni di fitness vs N (SA vs GA)
    sa_evals = [results["SA"][N]["success_evals"].get("mean", 0.0) if results["SA"][N]["success_evals"] else 0.0 for N in N_values]
    ga_evals = [results["GA"][N]["success_evals"].get("mean", 0.0) if results["GA"][N]["success_evals"] else 0.0 for N in N_values]
    
    plt.figure(figsize=(12, 8))
    plt.semilogy(N_values, [max(e, 1) for e in sa_evals], marker="s", linewidth=2, markersize=8, label="SA: Valutazioni conflitti")
    plt.semilogy(N_values, [max(e, 1) for e in ga_evals], marker="^", linewidth=2, markersize=8, label=f"GA-F{fitness_mode}: Valutazioni fitness")
    plt.xlabel("N (Dimensione scacchiera)", fontsize=12)
    plt.ylabel("Valutazioni Funzione Obiettivo (scala log)", fontsize=12)
    plt.title("Costo Puro in Chiamate alla Funzione Obiettivo\n(Misura il carico computazionale di valutazione)", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.7)
    plt.xticks(N_values)
    
    fname = os.path.join(out_dir, f"04_fitness_evaluations_vs_N_F{fitness_mode}.png")
    plt.savefig(fname, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved fitness-evaluation chart: {fname}")
    
    # ===========================================
    # 2. ANALISI TIMEOUT E FALLIMENTI
    # ===========================================
    
    # 2.1 Percentuale di timeout vs N
    plt.figure(figsize=(12, 8))
    plt.plot(N_values, sa_timeout, marker="s", linewidth=2, markersize=8, label="SA: Timeout rate")
    plt.plot(N_values, ga_timeout, marker="^", linewidth=2, markersize=8, label=f"GA-F{fitness_mode}: Timeout rate")
    plt.xlabel("N (Dimensione scacchiera)", fontsize=12)
    plt.ylabel("Tasso di Timeout", fontsize=12)
    plt.title("Timeout Rate vs Dimensione Problema\n(Mostra fino a che N l'algoritmo regge entro tempo massimo)", fontsize=14)
    plt.ylim(-0.05, 1.05)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.7)
    plt.xticks(N_values)
    
    fname = os.path.join(out_dir, f"05_timeout_rate_vs_N_F{fitness_mode}.png")
    plt.savefig(fname, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved timeout-rate chart: {fname}")
    
    # 2.2 Qualità nei fallimenti (best_conflicts nei run falliti)
    sa_fail_quality = [results["SA"][N]["failure_best_conflicts"].get("mean", N) if results["SA"][N].get("failure_best_conflicts") else N for N in N_values]
    ga_fail_quality = [results["GA"][N]["failure_best_conflicts"].get("mean", N) if results["GA"][N].get("failure_best_conflicts") else N for N in N_values]
    
    plt.figure(figsize=(12, 8))
    plt.plot(N_values, sa_fail_quality, marker="s", linewidth=2, markersize=8, label="SA: Conflitti medi (fallimenti)")
    plt.plot(N_values, ga_fail_quality, marker="^", linewidth=2, markersize=8, label=f"GA-F{fitness_mode}: Conflitti medi (fallimenti)")
    plt.plot(N_values, [0]*len(N_values), 'k--', alpha=0.5, label="Soluzione ottima (0 conflitti)")
    plt.xlabel("N (Dimensione scacchiera)", fontsize=12)
    plt.ylabel("Conflitti Medi nei Fallimenti", fontsize=12)
    plt.title("Qualità delle Soluzioni nei Run Falliti\n(Anche senza successo, quanto ci si avvicina all'ottimo)", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.7)
    plt.xticks(N_values)
    
    fname = os.path.join(out_dir, f"06_failure_quality_vs_N_F{fitness_mode}.png")
    plt.savefig(fname, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved failure-quality chart: {fname}")
    
    # ===========================================
    # 3. CONFRONTO TEORICO VS PRATICO
    # ===========================================
    
    # 3.1 Tempo vs Costo logico (SA)
    if any(sa_steps) and any(sa_time):
        plt.figure(figsize=(12, 8))
        # Filtra punti con dati validi
        valid_sa = [(s, t, n) for s, t, n in zip(sa_steps, sa_time, N_values) if s > 0 and t > 0]
        if valid_sa:
            steps_valid, time_valid, n_valid = zip(*valid_sa)
            plt.scatter(steps_valid, time_valid, c=n_valid, cmap='viridis', s=100, alpha=0.8)
            plt.colorbar(label='N (dimensione)')
            
            # Aggiungi linea di trend se possibile
            if len(valid_sa) > 2:
                z = np.polyfit(steps_valid, time_valid, 1)
                p = np.poly1d(z)
                x_trend = np.linspace(min(steps_valid), max(steps_valid), 100)
                plt.plot(x_trend, p(x_trend), "r--", alpha=0.8, label=f'Trend: y={z[0]:.2e}x+{z[1]:.2e}')
                plt.legend()
        
        plt.xlabel("Iterazioni SA (costo logico)", fontsize=12)
        plt.ylabel("Tempo [s] (costo pratico)", fontsize=12)
        plt.title("Simulated Annealing: Correlazione Costo Teorico vs Pratico\n(Linearità conferma dominio del costo di valutazione)", fontsize=14)
        plt.grid(True, alpha=0.7)
        
        fname = os.path.join(out_dir, f"07_SA_theoretical_vs_practical_F{fitness_mode}.png")
        plt.savefig(fname, bbox_inches="tight", dpi=300)
        plt.close()
    print(f"Saved SA theoretical-vs-practical chart: {fname}")
    
    # 3.2 Tempo vs Valutazioni fitness (GA)
    if any(ga_evals) and any(ga_time):
        plt.figure(figsize=(12, 8))
        valid_ga = [(e, t, n) for e, t, n in zip(ga_evals, ga_time, N_values) if e > 0 and t > 0]
        if valid_ga:
            evals_valid, time_valid, n_valid = zip(*valid_ga)
            plt.scatter(evals_valid, time_valid, c=n_valid, cmap='plasma', s=100, alpha=0.8)
            plt.colorbar(label='N (dimensione)')
            
            # Trend line
            if len(valid_ga) > 2:
                z = np.polyfit(evals_valid, time_valid, 1)
                p = np.poly1d(z)
                x_trend = np.linspace(min(evals_valid), max(evals_valid), 100)
                plt.plot(x_trend, p(x_trend), "r--", alpha=0.8, label=f'Trend: y={z[0]:.2e}x+{z[1]:.2e}')
                plt.legend()
        
        plt.xlabel("Valutazioni Fitness GA (costo logico)", fontsize=12)
        plt.ylabel("Tempo [s] (costo pratico)", fontsize=12)
        plt.title(f"GA-F{fitness_mode}: Correlazione Costo Teorico vs Pratico\n(Linearità conferma dominio del costo di valutazione)", fontsize=14)
        plt.grid(True, alpha=0.7)
        
        fname = os.path.join(out_dir, f"08_GA_theoretical_vs_practical_F{fitness_mode}.png")
        plt.savefig(fname, bbox_inches="tight", dpi=300)
        plt.close()
    print(f"Saved GA theoretical-vs-practical chart: {fname}")
    
    # 3.3 Tempo vs Nodi BT (conferma linearità)
    if any(bt_nodes) and any(bt_time):
        plt.figure(figsize=(12, 8))
        valid_bt = [(n, t, nval) for n, t, nval in zip(bt_nodes, bt_time, N_values) if n > 0 and t > 0]
        if valid_bt:
            nodes_valid, time_valid, n_valid = zip(*valid_bt)
            plt.scatter(nodes_valid, time_valid, c=n_valid, cmap='coolwarm', s=100, alpha=0.8)
            plt.colorbar(label='N (dimensione)')
            
            # Trend line
            if len(valid_bt) > 2:
                z = np.polyfit(nodes_valid, time_valid, 1)
                p = np.poly1d(z)
                x_trend = np.linspace(min(nodes_valid), max(nodes_valid), 100)
                plt.plot(x_trend, p(x_trend), "r--", alpha=0.8, label=f'Trend: y={z[0]:.2e}x+{z[1]:.2e}')
                plt.legend()
        
        plt.xlabel("Nodi Esplorati BT (costo logico)", fontsize=12)
        plt.ylabel("Tempo [s] (costo pratico)", fontsize=12)
        plt.title("Backtracking: Correlazione Costo Teorico vs Pratico\n(Ogni nodo costa tempo quasi costante)", fontsize=14)
        plt.grid(True, alpha=0.7)
        
        fname = os.path.join(out_dir, f"09_BT_theoretical_vs_practical_F{fitness_mode}.png")
        plt.savefig(fname, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"Saved BT theoretical-vs-practical chart: {fname}")
    
    print(f"\nComplete analysis generated in: {out_dir}")
    print(f"Generated {9} base charts for fitness F{fitness_mode}")


def plot_fitness_comparison(all_results, N_values, out_dir, raw_runs=None):
    """
    Confronto dettagliato tra le diverse fitness functions F1-F6 del GA
    """
    os.makedirs(out_dir, exist_ok=True)
    fitness_modes = list(all_results.keys())
    
    # Colori per le diverse fitness
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    fitness_colors = {f: colors[i % len(colors)] for i, f in enumerate(fitness_modes)}
    
    # ===========================================
    # 1. SUCCESS RATE PER FITNESS (a N fisso)
    # ===========================================
    
    # Scegli N rappresentativi per l'analisi
    analysis_N = [n for n in [16, 24, 40] if n in N_values]  # N piccoli, medi, grandi
    
    for N in analysis_N:
        # Bar chart: success rate per fitness
        plt.figure(figsize=(12, 8))
        success_rates = [all_results[f]["GA"][N]["success_rate"] for f in fitness_modes]
        
        bars = plt.bar(fitness_modes, success_rates, color=[fitness_colors[f] for f in fitness_modes], alpha=0.8)
        plt.xlabel("Fitness Function", fontsize=12)
        plt.ylabel("Tasso di Successo", fontsize=12) 
        plt.title(f"Confronto Success Rate tra Fitness Functions (N={N})\n(Quale fitness converge meglio a parità di dimensione)", fontsize=14)
        plt.ylim(0, 1.05)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Aggiungi valori sui bar
        for bar, sr in zip(bars, success_rates):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{sr:.3f}', ha='center', va='bottom', fontweight='bold')
        
        fname = os.path.join(out_dir, f"fitness_success_rate_N{N}.png")
        plt.savefig(fname, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"Saved success-rate comparison for N={N}: {fname}")
        
        # ===========================================
        # 2. GENERAZIONI MEDIE PER FITNESS
        # ===========================================
        
        plt.figure(figsize=(12, 8))
        gen_means = []
        gen_stds = []
        
        for f in fitness_modes:
            gen_stats = all_results[f]["GA"][N]["success_gen"]
            gen_means.append(gen_stats.get("mean", 0))
            gen_stds.append(gen_stats.get("std", 0))
        
        bars = plt.bar(fitness_modes, gen_means, yerr=gen_stds, 
                      color=[fitness_colors[f] for f in fitness_modes], 
                      alpha=0.8, capsize=5)
        plt.xlabel("Fitness Function", fontsize=12)
        plt.ylabel("Generazioni Medie +/- Std", fontsize=12)
        plt.title(f"Confronto Velocità di Convergenza (N={N})\n(Generazioni necessarie per trovare soluzione)", fontsize=14)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Valori sui bar
        for bar, mean, std in zip(bars, gen_means, gen_stds):
            if mean > 0:
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.5,
                        f'{mean:.1f}+/-{std:.1f}', ha='center', va='bottom', fontsize=10)
        
        fname = os.path.join(out_dir, f"fitness_generations_N{N}.png")
        plt.savefig(fname, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"Saved generation comparison for N={N}: {fname}")
        
        # ===========================================
        # 3. TEMPO MEDIO PER FITNESS
        # ===========================================
        
        plt.figure(figsize=(12, 8))
        time_means = []
        time_stds = []
        
        for f in fitness_modes:
            time_stats = all_results[f]["GA"][N]["success_time"]
            time_means.append(time_stats.get("mean", 0))
            time_stds.append(time_stats.get("std", 0))
        
        bars = plt.bar(fitness_modes, time_means, yerr=time_stds,
                      color=[fitness_colors[f] for f in fitness_modes], 
                      alpha=0.8, capsize=5)
        plt.xlabel("Fitness Function", fontsize=12)
        plt.ylabel("Tempo Medio [s] +/- Std", fontsize=12)
        plt.title(f"Confronto Efficienza Temporale (N={N})\n(Trade-off tra successo e velocità)", fontsize=14)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Valori sui bar
        for bar, mean, std in zip(bars, time_means, time_stds):
            if mean > 0:
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.001,
                        f'{mean:.3f}+/-{std:.3f}', ha='center', va='bottom', fontsize=10, rotation=0)
        
        fname = os.path.join(out_dir, f"fitness_time_N{N}.png")
        plt.savefig(fname, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"Saved time comparison for N={N}: {fname}")
    
    # ===========================================
    # 4. TRADE-OFF SUCCESS RATE VS COSTO (scatter)
    # ===========================================
    
    for N in analysis_N:
        plt.figure(figsize=(12, 8))
        
        success_rates = []
        gen_means = []
        eval_means = []
        
        for f in fitness_modes:
            ga_data = all_results[f]["GA"][N]
            success_rates.append(ga_data["success_rate"])
            gen_means.append(ga_data["success_gen"].get("mean", 0))
            eval_means.append(ga_data["success_evals"].get("mean", 0))
        
        # Scatter: success rate vs generazioni medie
        scatter = plt.scatter(gen_means, success_rates, 
                            c=[fitness_colors[f] for f in fitness_modes], 
                            s=150, alpha=0.8, edgecolors='black', linewidth=2)
        
        # Etichette per ogni punto
        for f, x, y in zip(fitness_modes, gen_means, success_rates):
            plt.annotate(f'F{f}', (x, y), xytext=(5, 5), 
                        textcoords='offset points', fontweight='bold', fontsize=12)
        
        plt.xlabel("Generazioni Medie per Successo", fontsize=12)
        plt.ylabel("Tasso di Successo", fontsize=12)
        plt.title(f"Trade-off Qualità vs Costo (N={N})\n(Pareto front: alto successo, basso costo)", fontsize=14)
        plt.grid(True, alpha=0.7)
        
        # Evidenzia area Pareto-ottima (alto successo, basso costo)
        max_sr = max(success_rates)
        min_gen = min([g for g in gen_means if g > 0])
        plt.axhline(y=max_sr*0.9, color='green', linestyle='--', alpha=0.5, label='Alto successo (>90% max)')
        plt.axvline(x=min_gen*1.5, color='red', linestyle='--', alpha=0.5, label='Basso costo (<150% min)')
        plt.legend()
        
        fname = os.path.join(out_dir, f"fitness_tradeoff_N{N}.png")
        plt.savefig(fname, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"Saved fitness trade-off for N={N}: {fname}")
    
    # ===========================================
    # 5. EVOLUZIONE SUCCESS RATE TUTTE LE FITNESS
    # ===========================================
    
    plt.figure(figsize=(15, 10))
    for f in fitness_modes:
        ga_sr_all_N = [all_results[f]["GA"][N]["success_rate"] for N in N_values]
        plt.plot(N_values, ga_sr_all_N, marker='o', linewidth=2, markersize=8, 
                label=f'GA-F{f}', color=fitness_colors[f])
    
    plt.xlabel("N (Dimensione scacchiera)", fontsize=12)
    plt.ylabel("Tasso di Successo", fontsize=12)
    plt.title("Confronto Success Rate: Tutte le Fitness Functions\n(Evoluzione affidabilità al crescere della dimensione)", fontsize=14)
    plt.ylim(-0.05, 1.05)
    plt.legend(fontsize=11, ncol=2)
    plt.grid(True, alpha=0.7)
    plt.xticks(N_values)
    
    fname = os.path.join(out_dir, f"fitness_evolution_all.png")
    plt.savefig(fname, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved fitness evolution overview: {fname}")
    
    print(f"\nFitness function comparison analysis completed")
    print(f"Generated comparison charts for F1-F6")


def plot_and_save(results, N_values, fitness_mode, out_dir):
    """
    Wrapper di compatibilità - chiama la nuova funzione completa
    """
    plot_comprehensive_analysis(results, N_values, fitness_mode, out_dir)


# ======================================================
# 9. Main: versione parallela del tuning + esperimenti finali
# ======================================================

def main_sequential(fitness_modes=None, skip_tuning=False, config_mgr=None):
    """Run the sequential pipeline for the selected fitness modes."""

    os.makedirs(OUT_DIR, exist_ok=True)
    selected_fitness = fitness_modes or FITNESS_MODES

    for fitness_mode in selected_fitness:
        print("\n============================================")
        print(f"SEQUENTIAL PIPELINE FOR GA FITNESS {fitness_mode}")
        print("============================================")

        if skip_tuning:
            print("Skipping GA tuning and reusing parameters from configuration.")
            best_ga_params_for_N = load_optimal_parameters(fitness_mode, config_mgr, N_VALUES)
        else:
            print("Starting GA tuning (sequential search).")
            best_ga_params_for_N = {}
            tuning_csv = os.path.join(OUT_DIR, f"tuning_GA_{fitness_mode}_seq.csv")
            progress = ProgressPrinter(len(N_VALUES), f"Tuning GA-{fitness_mode}")

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

                for index, N in enumerate(N_VALUES, start=1):
                    progress.update(index, f"N={N}")
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
                    print("  Best parameters:", best)

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

            if config_mgr:
                config_mgr.save_optimal_parameters(fitness_mode, best_ga_params_for_N)

        print(f"\nRunning final experiments for GA fitness {fitness_mode}")
        results = run_experiments_with_best_ga(
            N_VALUES,
            runs_sa=RUNS_SA_FINAL,
            runs_ga=RUNS_GA_FINAL,
            bt_time_limit=BT_TIME_LIMIT,
            fitness_mode=fitness_mode,
            best_ga_params_for_N=best_ga_params_for_N,
            progress_label=f"Experiments GA-{fitness_mode}",
        )

        save_results_to_csv(results, N_VALUES, fitness_mode, OUT_DIR)
        save_raw_data_to_csv(results, N_VALUES, fitness_mode, OUT_DIR)
        save_logical_cost_analysis(results, N_VALUES, fitness_mode, OUT_DIR)
        plot_and_save(results, N_VALUES, fitness_mode, OUT_DIR)

    print("\nSequential pipeline completed.")


def main_parallel(fitness_modes=None, skip_tuning=False, config_mgr=None):
    """Run the parallel pipeline for the selected fitness modes."""

    os.makedirs(OUT_DIR, exist_ok=True)
    selected_fitness = fitness_modes or FITNESS_MODES

    print(f"\nStarting parallel pipeline with {NUM_PROCESSES} worker processes")
    print(f"Available CPU cores: {multiprocessing.cpu_count()}")
    print("Configured timeouts:")
    print(f"   - BT: {BT_TIME_LIMIT}s" if BT_TIME_LIMIT else "   - BT: unlimited")
    print(f"   - SA: {SA_TIME_LIMIT}s" if SA_TIME_LIMIT else "   - SA: unlimited")
    print(f"   - GA: {GA_TIME_LIMIT}s" if GA_TIME_LIMIT else "   - GA: unlimited")
    print(f"   - Experiment: {EXPERIMENT_TIMEOUT}s" if EXPERIMENT_TIMEOUT else "   - Experiment: unlimited")

    start_total = perf_counter()
    all_best_params = {}

    if skip_tuning:
        print("\nSkipping GA tuning phase and loading parameters from configuration.")
        for fitness_mode in selected_fitness:
            all_best_params[fitness_mode] = load_optimal_parameters(fitness_mode, config_mgr, N_VALUES)
    else:
        print("\n" + "=" * 60)
        print("PHASE 1: PARALLEL GA TUNING")
        print("=" * 60)

        for fitness_mode in selected_fitness:
            print(f"\nTuning fitness {fitness_mode}...")
            fitness_start = perf_counter()

            best_ga_params_for_N = {}
            tuning_csv = os.path.join(OUT_DIR, f"tuning_GA_{fitness_mode}.csv")
            progress = ProgressPrinter(len(N_VALUES), f"Tuning GA-{fitness_mode}")

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

                for index, N in enumerate(N_VALUES, start=1):
                    progress.update(index, f"N={N}")
                    tuning_start = perf_counter()

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

                    tuning_time = perf_counter() - tuning_start
                    best_ga_params_for_N[N] = best
                    print(
                        f"     Completed in {tuning_time:.1f}s - Success rate: {best['success_rate']:.3f}"
                    )

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

            all_best_params[fitness_mode] = best_ga_params_for_N

            fitness_time = perf_counter() - fitness_start
            print(f"Tuning {fitness_mode} completed in {fitness_time:.1f}s - CSV: {tuning_csv}")

            if config_mgr:
                config_mgr.save_optimal_parameters(fitness_mode, best_ga_params_for_N)

    print("\n" + "=" * 60)
    print("PHASE 2: PARALLEL FINAL EXPERIMENTS")
    print("=" * 60)

    for fitness_mode in selected_fitness:
        print(f"\nFinal experiments for {fitness_mode}...")
        experiments_start = perf_counter()

        results = run_experiments_with_best_ga_parallel(
            N_VALUES,
            runs_sa=RUNS_SA_FINAL,
            runs_ga=RUNS_GA_FINAL,
            bt_time_limit=BT_TIME_LIMIT,
            fitness_mode=fitness_mode,
            best_ga_params_for_N=all_best_params[fitness_mode],
            progress_label=f"Experiments GA-{fitness_mode}",
        )

        experiments_time = perf_counter() - experiments_start
        print(f"  Experiments completed in {experiments_time:.1f}s")

        print("Generating charts and CSV reports...")
        save_results_to_csv(results, N_VALUES, fitness_mode, OUT_DIR)
        save_raw_data_to_csv(results, N_VALUES, fitness_mode, OUT_DIR)
        save_logical_cost_analysis(results, N_VALUES, fitness_mode, OUT_DIR)
        plot_and_save(results, N_VALUES, fitness_mode, OUT_DIR)
        print(f"  Results saved for {fitness_mode}")

    total_time = perf_counter() - start_total
    print("\nParallel pipeline completed!")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Fitness processed: {len(selected_fitness)}")
    print(f"Worker processes used: {NUM_PROCESSES}")


def main_concurrent_tuning(fitness_modes=None, skip_tuning=False, config_mgr=None):
    """Run concurrent tuning and experiments across the selected fitness modes."""

    os.makedirs(OUT_DIR, exist_ok=True)
    selected_fitness = fitness_modes or FITNESS_MODES

    print("\nCONCURRENT TUNING FOR SELECTED FITNESS FUNCTIONS")
    print(f"Fitness modes: {selected_fitness}")
    print(f"Processes: {NUM_PROCESSES}")
    print(f"Available CPU cores: {multiprocessing.cpu_count()}")
    print("Configured timeouts:")
    print(f"   - BT: {BT_TIME_LIMIT}s" if BT_TIME_LIMIT else "   - BT: unlimited")
    print(f"   - SA: {SA_TIME_LIMIT}s" if SA_TIME_LIMIT else "   - SA: unlimited")
    print(f"   - GA: {GA_TIME_LIMIT}s" if GA_TIME_LIMIT else "   - GA: unlimited")
    print(f"   - Experiment: {EXPERIMENT_TIMEOUT}s" if EXPERIMENT_TIMEOUT else "   - Experiment: unlimited")

    start_total = perf_counter()
    all_best_params = {fitness_mode: {} for fitness_mode in selected_fitness}

    if skip_tuning:
        print("\nSkipping GA tuning phase and loading parameters from configuration.")
        for fitness_mode in selected_fitness:
            all_best_params[fitness_mode] = load_optimal_parameters(fitness_mode, config_mgr, N_VALUES)
    else:
        print("\n" + "=" * 70)
        print("PHASE 1: PARALLEL TUNING FOR ALL FITNESS FUNCTIONS")
        print("=" * 70)

        progress = ProgressPrinter(len(N_VALUES), "Concurrent GA tuning")

        for index, N in enumerate(N_VALUES, start=1):
            progress.update(index, f"N={N}")
            print(f"\nParallel tuning for N = {N}")
            print("-" * 50)

            fitness_results = tune_all_fitness_parallel(
                N,
                selected_fitness,
                POP_MULTIPLIERS,
                GEN_MULTIPLIERS,
                PM_VALUES,
                PC_FIXED,
                TOURNAMENT_SIZE_FIXED,
                runs_tuning=RUNS_GA_TUNING,
            )

            for fitness_mode, best_params in fitness_results.items():
                all_best_params[fitness_mode][N] = best_params

        for fitness_mode in selected_fitness:
            save_tuning_results(all_best_params[fitness_mode], fitness_mode, OUT_DIR)
            if config_mgr:
                config_mgr.save_optimal_parameters(fitness_mode, all_best_params[fitness_mode])

    print("\n" + "=" * 70)
    print("PHASE 2: FINAL EXPERIMENTS WITH OPTIMAL PARAMETERS")
    print("=" * 70)

    all_results = {}

    for fitness_mode in selected_fitness:
        print(f"\nFinal experiments GA-{fitness_mode}")

        results = run_experiments_with_best_ga_parallel(
            N_VALUES,
            runs_sa=RUNS_SA_FINAL,
            runs_ga=RUNS_GA_FINAL,
            bt_time_limit=BT_TIME_LIMIT,
            fitness_mode=fitness_mode,
            best_ga_params_for_N=all_best_params[fitness_mode],
            progress_label=f"Experiments GA-{fitness_mode}",
        )

        all_results[fitness_mode] = results

        save_results_to_csv(results, N_VALUES, fitness_mode, OUT_DIR)
        save_raw_data_to_csv(results, N_VALUES, fitness_mode, OUT_DIR)
        save_logical_cost_analysis(results, N_VALUES, fitness_mode, OUT_DIR)
        plot_and_save(results, N_VALUES, fitness_mode, OUT_DIR)

    print("\n" + "=" * 70)
    print("PHASE 3: COMPARATIVE ANALYSIS AND ADVANCED CHARTS")
    print("=" * 70)

    for fitness in selected_fitness:
        print(f"  Comprehensive analysis for GA-F{fitness}...")
        plot_comprehensive_analysis(
            all_results[fitness],
            N_VALUES,
            fitness,
            os.path.join(OUT_DIR, f"analysis_F{fitness}"),
            raw_runs=None,
        )

    print("  Comparing fitness functions...")
    plot_fitness_comparison(
        all_results,
        N_VALUES,
        os.path.join(OUT_DIR, "fitness_comparison"),
    )

    print("  Statistical analysis...")
    plot_statistical_analysis(
        all_results,
        N_VALUES,
        os.path.join(OUT_DIR, "statistical_analysis"),
        raw_runs=None,
    )

    total_time = perf_counter() - start_total
    print("\nConcurrent pipeline completed!")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Fitness processed: {len(selected_fitness)}")

def plot_statistical_analysis(all_results, N_values, out_dir, raw_runs=None):
    """
    Grafici statistici con boxplot e analisi della variabilità
    """
    os.makedirs(out_dir, exist_ok=True)
    
    if not raw_runs:
        print("Raw runs non disponibili per analisi statistica dettagliata")
        return
    
    # Analizza un subset rappresentativo di N 
    analysis_N = [n for n in [16, 24, 40] if n in N_values and n in raw_runs]
    
    for N in analysis_N:
        if N not in raw_runs:
            continue

        print(f"Analisi statistica per N={N}...")
        
        # ===========================================
        # 1. BOXPLOT DEI TEMPI (solo successi)
        # ===========================================
        
        plt.figure(figsize=(14, 8))
        
        # Raccogli dati temporali per tutti gli algoritmi
        time_data = []
        labels = []
        
        # SA tempi (solo successi)
        if "SA" in raw_runs[N]:
            sa_times = [run["time"] for run in raw_runs[N]["SA"] if run["success"]]
            if sa_times:
                time_data.append(sa_times)
                labels.append("SA")
        
        # GA tempi per ogni fitness (solo successi)
        for fitness in sorted(all_results.keys()):
            if fitness in raw_runs[N]:
                ga_times = [run["time"] for run in raw_runs[N][fitness] if run["success"]]
                if ga_times:
                    time_data.append(ga_times)
                    labels.append(f"GA-F{fitness}")
        
        if time_data:
            box_plot = plt.boxplot(time_data, labels=labels, patch_artist=True)
            
            # Colora i boxplot
            colors = ['#ff7f0e', '#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
            for patch, color in zip(box_plot['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            plt.ylabel("Tempo di Esecuzione [s]", fontsize=12)
            plt.title(f"Distribuzione Tempi di Esecuzione (N={N}, solo successi)\n(Boxplot mostra mediana, quartili, outliers)", fontsize=14)
            plt.yscale('log')
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            
            fname = os.path.join(out_dir, f"boxplot_times_N{N}.png")
            plt.savefig(fname, bbox_inches="tight", dpi=300)
            plt.close()
            print(f"Boxplot tempi N={N}: {fname}")
        
        # ===========================================
        # 2. BOXPLOT DELLE ITERAZIONI/GENERAZIONI
        # ===========================================
        
        plt.figure(figsize=(14, 8))
        
        iter_data = []
        iter_labels = []
        
        # SA iterazioni (solo successi)
        if "SA" in raw_runs[N]:
            sa_steps = [run["steps"] for run in raw_runs[N]["SA"] if run["success"]]
            if sa_steps:
                iter_data.append(sa_steps)
                iter_labels.append("SA (steps)")
        
        # GA generazioni per ogni fitness (solo successi)
        for fitness in sorted(all_results.keys()):
            if fitness in raw_runs[N]:
                ga_gens = [run["gen"] for run in raw_runs[N][fitness] if run["success"]]
                if ga_gens:
                    iter_data.append(ga_gens)
                    iter_labels.append(f"GA-F{fitness} (gen)")
        
        if iter_data:
            box_plot = plt.boxplot(iter_data, labels=iter_labels, patch_artist=True)
            
            # Colora i boxplot
            for patch, color in zip(box_plot['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            plt.ylabel("Iterazioni/Generazioni", fontsize=12)
            plt.title(f"Distribuzione Costi Logici (N={N}, solo successi)\n(Variabilità dell'algoritmo in termini di sforzo)", fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            
            fname = os.path.join(out_dir, f"boxplot_iterations_N{N}.png")
            plt.savefig(fname, bbox_inches="tight", dpi=300)
            plt.close()
            print(f"Boxplot iterazioni N={N}: {fname}")
        
        # ===========================================
        # 3. ISTOGRAMMI DISTRIBUZIONI
        # ===========================================
        
        # SA histogram
        if "SA" in raw_runs[N]:
            sa_times = [run["time"] for run in raw_runs[N]["SA"] if run["success"]]
            if len(sa_times) > 5:  # Abbastanza dati per istogramma
                plt.figure(figsize=(12, 6))
                plt.hist(sa_times, bins=min(20, len(sa_times)//2), alpha=0.7, color='orange', edgecolor='black')
                plt.xlabel("Tempo [s]", fontsize=12)
                plt.ylabel("Frequenza", fontsize=12)
                plt.title(f"Distribuzione Tempi SA (N={N})\n(Forma della distribuzione indica stabilità algoritmo)", fontsize=14)
                plt.grid(True, alpha=0.3)
                
                # Aggiungi statistiche
                mean_time = np.mean(sa_times)
                std_time = np.std(sa_times)
                plt.axvline(mean_time, color='red', linestyle='--', label=f'Media: {mean_time:.3f}s')
                plt.axvline(mean_time + std_time, color='red', linestyle=':', alpha=0.7, label=f'+/-1 sigma: {std_time:.3f}s')
                plt.axvline(mean_time - std_time, color='red', linestyle=':', alpha=0.7)
                plt.legend()
                
                fname = os.path.join(out_dir, f"histogram_SA_times_N{N}.png")
                plt.savefig(fname, bbox_inches="tight", dpi=300)
                plt.close()
                print(f"Istogramma SA tempi N={N}: {fname}")
        
        # GA histogram per la migliore fitness
        best_fitness = min(all_results.keys(), key=lambda f: -all_results[f]["GA"][N]["success_rate"])
        if best_fitness in raw_runs[N]:
            ga_times = [run["time"] for run in raw_runs[N][best_fitness] if run["success"]]
            if len(ga_times) > 5:
                plt.figure(figsize=(12, 6))
                plt.hist(ga_times, bins=min(20, len(ga_times)//2), alpha=0.7, color='green', edgecolor='black')
                plt.xlabel("Tempo [s]", fontsize=12)
                plt.ylabel("Frequenza", fontsize=12)
                plt.title(f"Distribuzione Tempi GA-F{best_fitness} (N={N})\n(Algoritmo più stabile = distribuzione stretta)", fontsize=14)
                plt.grid(True, alpha=0.3)
                
                # Statistiche
                mean_time = np.mean(ga_times)
                std_time = np.std(ga_times)
                plt.axvline(mean_time, color='red', linestyle='--', label=f'Media: {mean_time:.3f}s')
                plt.axvline(mean_time + std_time, color='red', linestyle=':', alpha=0.7, label=f'+/-1 sigma: {std_time:.3f}s')
                plt.axvline(mean_time - std_time, color='red', linestyle=':', alpha=0.7)
                plt.legend()
                
                fname = os.path.join(out_dir, f"histogram_GA_F{best_fitness}_times_N{N}.png")
                plt.savefig(fname, bbox_inches="tight", dpi=300)
                plt.close()
                print(f"Istogramma GA-F{best_fitness} tempi N={N}: {fname}")
    
    print("Analisi statistica completata")


def plot_tuning_analysis(tuning_data, fitness_modes, N_values, out_dir):
    """
    Analisi dei dati di tuning GA con heatmap e scatter plots
    """
    if not tuning_data:
        print("Dati tuning non disponibili")
        return
        
    os.makedirs(out_dir, exist_ok=True)
    
    # Per ogni fitness
    for fitness in fitness_modes:
        if fitness not in tuning_data:
            continue

        print(f"Analisi tuning GA-F{fitness}...")
        
        # Scegli N rappresentativo per analisi dettagliata
        analysis_N = [n for n in [24, 40] if n in N_values and n in tuning_data[fitness]]
        
        for N in analysis_N:
            if N not in tuning_data[fitness]:
                continue
                
            tuning_runs = tuning_data[fitness][N]
            if not tuning_runs:
                continue
            
            # ===========================================
            # 1. HEATMAP SUCCESS RATE vs POP_SIZE, MAX_GEN
            # ===========================================
            
            # Raccogli dati in matrice
            pop_sizes = sorted(set(run['pop_size'] for run in tuning_runs))
            max_gens = sorted(set(run['max_gen'] for run in tuning_runs))
            
            if len(pop_sizes) > 1 and len(max_gens) > 1:
                # Crea matrice success_rate
                sr_matrix = np.zeros((len(max_gens), len(pop_sizes)))
                
                for i, mg in enumerate(max_gens):
                    for j, ps in enumerate(pop_sizes):
                        # Trova run con questi parametri
                        matching_runs = [r for r in tuning_runs 
                                       if r['pop_size'] == ps and r['max_gen'] == mg]
                        if matching_runs:
                            sr_matrix[i, j] = matching_runs[0]['success_rate']
                
                # Plot heatmap
                plt.figure(figsize=(12, 8))
                im = plt.imshow(sr_matrix, cmap='RdYlGn', aspect='auto', interpolation='nearest')
                plt.colorbar(im, label='Success Rate')
                
                # Labels
                plt.xticks(range(len(pop_sizes)), [f'{ps}' for ps in pop_sizes])
                plt.yticks(range(len(max_gens)), [f'{mg}' for mg in max_gens])
                plt.xlabel("Population Size", fontsize=12)
                plt.ylabel("Max Generations", fontsize=12)
                plt.title(f"GA-F{fitness} Tuning Heatmap (N={N})\n(Zona verde = parametri ottimali)", fontsize=14)
                
                # Aggiungi valori nelle celle
                for i in range(len(max_gens)):
                    for j in range(len(pop_sizes)):
                        if sr_matrix[i, j] > 0:
                            plt.text(j, i, f'{sr_matrix[i, j]:.2f}', 
                                   ha="center", va="center", fontweight='bold',
                                   color='white' if sr_matrix[i, j] < 0.5 else 'black')
                
                fname = os.path.join(out_dir, f"heatmap_tuning_GA_F{fitness}_N{N}.png")
                plt.savefig(fname, bbox_inches="tight", dpi=300)
                plt.close()
                print(f"Heatmap tuning GA-F{fitness} N={N}: {fname}")
            
            # ===========================================
            # 2. SCATTER: COSTO vs QUALITÀ
            # ===========================================
            
            plt.figure(figsize=(12, 8))
            
            costs = [run['pop_size'] * run['max_gen'] for run in tuning_runs]
            success_rates = [run['success_rate'] for run in tuning_runs]
            
            # Scatter colorato per mutation rate se disponibile
            if 'pm' in tuning_runs[0]:
                pms = [run['pm'] for run in tuning_runs]
                scatter = plt.scatter(costs, success_rates, c=pms, cmap='viridis', 
                                    s=100, alpha=0.7, edgecolors='black')
                plt.colorbar(scatter, label='Mutation Rate (pm)')
            else:
                plt.scatter(costs, success_rates, s=100, alpha=0.7, edgecolors='black')
            
            plt.xlabel("Costo Computazionale (pop_size x max_gen)", fontsize=12)
            plt.ylabel("Tasso di Successo", fontsize=12)
            plt.title(f"GA-F{fitness}: Trade-off Costo vs Qualità (N={N})\n(Mostra se vale la pena aumentare parametri)", fontsize=14)
            plt.grid(True, alpha=0.3)
            
            # Evidenzia configurazioni Pareto-ottimali
            max_sr = max(success_rates)
            pareto_points = [(c, sr) for c, sr in zip(costs, success_rates) 
                           if sr >= max_sr * 0.95]  # Entro 95% del massimo
            
            if pareto_points:
                min_cost_pareto = min(p[0] for p in pareto_points)
                plt.axhline(y=max_sr*0.95, color='red', linestyle='--', alpha=0.5, 
                          label=f'95% max success ({max_sr*0.95:.2f})')
                plt.axvline(x=min_cost_pareto*1.1, color='green', linestyle='--', alpha=0.5,
                          label=f'Costo efficiente (<{min_cost_pareto*1.1:.0f})')
                plt.legend()
            
            fname = os.path.join(out_dir, f"scatter_cost_quality_GA_F{fitness}_N{N}.png")
            plt.savefig(fname, bbox_inches="tight", dpi=300)
            plt.close()
            print(f"Scatter costo-qualità GA-F{fitness} N={N}: {fname}")
            
            # ===========================================
            # 3. LINE PLOT per MUTATION RATE
            # ===========================================
            
            if 'pm' in tuning_runs[0]:
                pm_values = sorted(set(run['pm'] for run in tuning_runs))
                
                if len(pm_values) > 1:
                    plt.figure(figsize=(12, 8))
                    
                    # Per ogni pm, calcola success rate medio
                    pm_sr_means = []
                    pm_sr_stds = []
                    
                    for pm in pm_values:
                        pm_runs = [run for run in tuning_runs if run['pm'] == pm]
                        srs = [run['success_rate'] for run in pm_runs]
                        pm_sr_means.append(np.mean(srs))
                        pm_sr_stds.append(np.std(srs) if len(srs) > 1 else 0)
                    
                    plt.errorbar(pm_values, pm_sr_means, yerr=pm_sr_stds, 
                               marker='o', linewidth=2, markersize=8, capsize=5)
                    plt.xlabel("Mutation Rate (pm)", fontsize=12)
                    plt.ylabel("Success Rate Medio +/- Std", fontsize=12)
                    plt.title(f"GA-F{fitness}: Effetto Mutation Rate (N={N})\n(Mostra sensibilità alla mutazione)", fontsize=14)
                    plt.grid(True, alpha=0.3)
                    plt.ylim(0, 1.05)
                    
                    fname = os.path.join(out_dir, f"lineplot_mutation_GA_F{fitness}_N{N}.png")
                    plt.savefig(fname, bbox_inches="tight", dpi=300)
                    plt.close()
                    print(f"Line plot mutation GA-F{fitness} N={N}: {fname}")
    
    print("Analisi tuning completata")


def save_tuning_results(best_params_for_N, fitness_mode, out_dir):
    """
    Salva i risultati del tuning in un CSV
    """
    filename = os.path.join(out_dir, f"tuning_GA_F{fitness_mode}.csv")
    
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "N", "pop_size", "max_gen", "pm", "pc", "tournament_size", 
            "success_rate_tuning", "avg_gen_success_tuning"
        ])
        
        for N in sorted(best_params_for_N.keys()):
            params = best_params_for_N[N]
            writer.writerow([
                N,
                params.get("pop_size", ""),
                params.get("max_gen", ""),
                params.get("pm", ""),
                params.get("pc", ""),
                params.get("tournament_size", ""),
                params.get("success_rate", ""),
                params.get("avg_gen_success", "")
            ])
    
    print(f"Risultati tuning GA-F{fitness_mode} salvati: {filename}")


def run_quick_regression_tests():
    """Run lightweight deterministic checks for BT, SA, GA, and CSV generation."""

    print("Running quick regression tests (N=8)...")

    random.seed(42)
    solution, nodes, elapsed = bt_nqueens_first(8, time_limit=5.0)
    if solution is None:
        raise AssertionError("Backtracking failed to find a solution for N=8.")
    print(f"  Backtracking: solution found with {nodes} nodes in {elapsed:.4f}s")

    random.seed(42)
    sa_success, _, sa_time, _, _, sa_timeout = sa_nqueens(
        8, max_iter=5000, T0=1.0, alpha=0.995, time_limit=5.0
    )
    if not sa_success or sa_timeout:
        raise AssertionError("Simulated Annealing did not succeed for N=8 with deterministic seed.")
    print(f"  Simulated Annealing: success in {sa_time:.4f}s")

    random.seed(42)
    ga_success, _, ga_time, _, _, ga_timeout = ga_nqueens(
        8,
        pop_size=60,
        max_gen=200,
        pc=0.8,
        pm=0.1,
        tournament_size=3,
        fitness_mode="F1",
        time_limit=5.0,
    )
    if not ga_success or ga_timeout:
        raise AssertionError("Genetic Algorithm did not succeed for N=8 with deterministic seed.")
    print(f"  Genetic Algorithm: success in {ga_time:.4f}s")

    best_ga_params_for_N = {
        8: {
            "pop_size": 60,
            "max_gen": 200,
            "pm": 0.1,
            "pc": 0.8,
            "tournament_size": 3,
        }
    }

    results = run_experiments_with_best_ga(
        [8],
        runs_sa=3,
        runs_ga=3,
        bt_time_limit=5.0,
        fitness_mode="F1",
        best_ga_params_for_N=best_ga_params_for_N,
        progress_label="Quick regression experiments",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        save_results_to_csv(results, [8], "F1", tmpdir)
        csv_path = Path(tmpdir) / "results_GA_F1_tuned.csv"
        if not csv_path.exists() or csv_path.stat().st_size == 0:
            raise AssertionError("Results CSV was not generated successfully during quick tests.")

    print("Quick regression tests passed.")


# Alias per compatibilità con il main
def run_experiments_parallel(
    N_values,
    runs_bt,
    runs_sa,
    runs_ga,
    bt_time_limit,
    fitness_mode,
    best_ga_params_for_N,
    progress_label=None,
):
    """
    Wrapper per run_experiments_with_best_ga_parallel per compatibilità
    """
    return run_experiments_with_best_ga_parallel(
        N_values=N_values,
        runs_sa=runs_sa,
        runs_ga=runs_ga,
        bt_time_limit=bt_time_limit,
        fitness_mode=fitness_mode,
        best_ga_params_for_N=best_ga_params_for_N,
        progress_label=progress_label,
    )


def build_arg_parser():
    """Create the CLI argument parser for the orchestrator."""

    parser = argparse.ArgumentParser(
        description="Run N-Queens tuning and experiment pipelines."
    )
    parser.add_argument(
        "--mode",
        choices=["sequential", "parallel", "concurrent"],
        default="concurrent",
        help="Execution mode: sequential tuning, parallel tuning, or concurrent tuning (default).",
    )
    parser.add_argument(
        "--fitness",
        "-f",
        action="append",
        help="Filter fitness modes (accepts comma-separated values or multiple flags).",
    )
    parser.add_argument(
        "--skip-tuning",
        action="store_true",
        help="Reuse stored GA parameters from config.json instead of running tuning.",
    )
    parser.add_argument(
        "--config",
        default="config.json",
        help="Path to configuration file (default: config.json).",
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run quick regression tests (N=8) and exit.",
    )
    return parser


def main():
    """Entry point for CLI execution."""

    parser = build_arg_parser()
    args = parser.parse_args()
    fitness_filter = parse_fitness_filters(args.fitness)

    if args.quick_test:
        run_quick_regression_tests()
        return

    try:
        config_mgr, selected_fitness = apply_configuration(args.config, fitness_filter)
    except FileNotFoundError as exc:
        print(f"Configuration file not found: {exc}")
        raise SystemExit(1) from exc
    except ValueError as exc:
        print(f"Configuration error: {exc}")
        raise SystemExit(1) from exc

    print(f"Selected fitness modes: {selected_fitness}")

    try:
        if args.mode == "sequential":
            main_sequential(selected_fitness, skip_tuning=args.skip_tuning, config_mgr=config_mgr)
        elif args.mode == "parallel":
            main_parallel(selected_fitness, skip_tuning=args.skip_tuning, config_mgr=config_mgr)
        else:
            main_concurrent_tuning(selected_fitness, skip_tuning=args.skip_tuning, config_mgr=config_mgr)
    except KeyboardInterrupt:
        print("\nExecution interrupted by user. Cleaning up workers...")
        raise SystemExit(130) from None
    except ValueError as exc:
        print(f"Execution error: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
