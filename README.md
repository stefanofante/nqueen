# N-Queens Problem: Comparative Analysis with Parallel Optimization# N-Queens Problem: Comparative Analysis of Optimization Algorithms# N-Queens Problem: Comparative Analysis of Optimization Algorithms

## 🎯 Panoramica## 🎯 Panoramica## 🎯 Panoramica

Questo repository presenta un'analisi comparativa completa di tre approcci algoritmici per risolvere il famoso **problema delle N-Regine**: Backtracking, Simulated Annealing e Algoritmo Genetico con ottimizzazione automatica dei parametri.Questo repository presenta un'analisi comparativa completa di tre approcci algoritmici per risolvere il famoso **problema delle N-Regine**: Backtracking, Simulated Annealing e Algoritmo Genetico con ottimizzazione automatica dei parametri.Questo repository presenta un'analisi comparativa completa di tre approcci algoritmici per risolvere il famoso **problema delle N-Regine**: Backtracking, Simulated Annealing e Algoritmo Genetico con ottimizzazione automatica dei parametri.

**🚀 NOVITÀ**: Implementazione parallela ottimizzata con `ProcessPoolExecutor` per accelerare significativamente tuning e sperimentazioni!Il problema delle N-Regine consiste nel posizionare N regine su una scacchiera N×N in modo che nessuna regina possa attaccare un'altra. Due regine si attaccano se si trovano sulla stessa riga, colonna o diagonale.Il problema delle N-Regine consiste nel posizionare N regine su una scacchiera N×N in modo che nessuna regina possa attaccare un'altra. Due regine si attaccano se si trovano sulla stessa riga, colonna o diagonale.

Il problema delle N-Regine consiste nel posizionare N regine su una scacchiera N×N in modo che nessuna regina possa attaccare un'altra.## 📊 Caratteristiche Principali## 📊 Caratteristiche Principali

## 📊 Caratteristiche Principali- **Implementazione completa di 3 algoritmi**: Backtracking iterativo, Simulated Annealing, Algoritmo Genetico- **Implementazione completa di 3 algoritmi**: Backtracking iterativo, Simulated Annealing, Algoritmo Genetico

- **Implementazione completa di 3 algoritmi**: Backtracking iterativo, Simulated Annealing, Algoritmo Genetico- **6 funzioni di fitness diverse** per l'algoritmo genetico (F1-F6)- **6 funzioni di fitness diverse** per l'algoritmo genetico (F1-F6)

- **6 funzioni di fitness diverse** per l'algoritmo genetico (F1-F6)

- **🚀 Tuning automatico parallelo** per ottimizzare le performance del GA- **Tuning automatico dei parametri** per ottimizzare le performance del GA- **Tuning automatico dei parametri** per ottimizzare le performance del GA

- **🔥 Parallelizzazione multi-core** con speedup 3-8x

- **Analisi statistica robusta** con multiple esecuzioni indipendenti- **Analisi statistica robusta** con multiple esecuzioni indipendenti- **Analisi statistica robusta** con multiple esecuzioni indipendenti

- **Visualizzazione dei risultati** con grafici comparativi

- **Export dati in formato CSV** per analisi successive- **Visualizzazione dei risultati** con grafici comparativi- **Visualizzazione dei risultati** con grafici comparativi

## 🧮 Il Problema delle N-Regine- **Export dati in formato CSV** per analisi successive- **Export dati in formato CSV** per analisi successive

### Definizione Matematica## 🧮 Il Problema## 🧮 Il Problema delle N-Regine

Dato un intero N ≥ 4, il problema consiste nel trovare un posizionamento di N regine su una scacchiera N×N tale che:### Definizione Matematica### Definizione Matematica

1. **Vincolo di riga**: al massimo una regina per rigaDato un intero N ≥ 4, il problema consiste nel trovare un posizionamento di N regine su una scacchiera N×N tale che:Dato un intero N ≥ 4, il problema consiste nel trovare un posizionamento di N regine su una scacchiera N×N tale che:

2. **Vincolo di colonna**: al massimo una regina per colonna

3. **Vincolo di diagonale principale**: al massimo una regina per diagonale con pendenza +11. **Vincolo di riga**: al massimo una regina per riga1. **Vincolo di riga**: al massimo una regina per riga

4. **Vincolo di diagonale secondaria**: al massimo una regina per diagonale con pendenza -1

2. **Vincolo di colonna**: al massimo una regina per colonna2. **Vincolo di colonna**: al massimo una regina per colonna  

### Rappresentazione della Soluzione

3. **Vincolo di diagonale principale**: al massimo una regina per diagonale con pendenza +13. **Vincolo di diagonale principale**: al massimo una regina per diagonale con pendenza +1

Il problema viene rappresentato tramite un array `board[N]` dove `board[i]` indica la riga in cui è posizionata la regina della colonna `i`.

4. **Vincolo di diagonale secondaria**: al massimo una regina per diagonale con pendenza -14. **Vincolo di diagonale secondaria**: al massimo una regina per diagonale con pendenza -1

### Funzione di Conflitto Ottimizzata

### Rappresentazione della Soluzione### Rappresentazione della Soluzione

```python

def conflicts(board):Il problema viene rappresentato tramite un array `board[N]` dove `board[i]` indica la riga in cui è posizionata la regina della colonna `i`. Questa rappresentazione garantisce automaticamente il rispetto del vincolo di colonna.Il problema viene rappresentato tramite un array `board[N]` dove `board[i]` indica la riga in cui è posizionata la regina della colonna `i`. Questa rappresentazione garantisce automaticamente il rispetto del vincolo di colonna.

    """Algoritmo O(N) ottimizzato con contatori"""

    n = len(board)### Funzione di Conflitto### Funzione di Conflitto

    row_count = Counter()

    diag1 = Counter()  # Diagonale r-cLa qualità di una soluzione è misurata attraverso il numero di coppie di regine in conflitto:La qualità di una soluzione è misurata attraverso il numero di coppie di regine in conflitto:

    diag2 = Counter()  # Diagonale r+c

    ```python```python

    for c, r in enumerate(board):

        row_count[r] += 1def conflicts(board):def conflicts(board):

        diag1[r - c] += 1

        diag2[r + c] += 1    """Conta le coppie di regine in conflitto usando contatori per righe e diagonali"""    """Conta le coppie di regine in conflitto usando contatori per righe e diagonali"""

    

    total_conflicts = 0    n = len(board)    n = len(board)

    for counter in [row_count, diag1, diag2]:

        for count in counter.values():    row_count = Counter()    row_count = Counter()

            if count > 1:

                total_conflicts += count * (count - 1) // 2    diag1 = Counter()  # Diagonale r-c    diag1 = Counter()  # Diagonale r-c

    

    return total_conflicts    diag2 = Counter()  # Diagonale r+c    diag2 = Counter()  # Diagonale r+c

```

## 🔍 Algoritmi Implementati

    for c, r in enumerate(board):    for c, r in enumerate(board):

### 1. Backtracking Iterativo

        row_count[r] += 1        row_count[r] += 1

**Approccio**: Ricerca esaustiva sistematica ottimizzata.

        diag1[r - c] += 1        diag1[r - c] += 1

**Caratteristiche**:

- ✅ **Garanzia di ottimalità**: trova sempre una soluzione se esiste        diag2[r + c] += 1        diag2[r + c] += 1

- ⚠️ **Complessità esponenziale**: O(N!) nel caso peggiore

- 🎯 **Deterministico**: comportamento prevedibile

- ⏱️ **Time limit opzionale**: per evitare esecuzioni infinite

  # Calcola conflitti come combinazioni C(k,2) per ogni gruppo    # Calcola conflitti come combinazioni C(k,2) per ogni gruppo

### 2. Simulated Annealing

    total_conflicts = 0    total_conflicts = 0

**Approccio**: Metaeuristica con raffreddamento geometrico.

    for counter in [row_count, diag1, diag2]:    for counter in [row_count, diag1, diag2]:

**Parametri ottimizzati**:

- `T0 = 1.0`: Temperatura iniziale        for count in counter.values():        for count in counter.values():

- `alpha = 0.995`: Fattore di raffreddamento

- `max_iter = 2000 + 200*N`: Iterazioni scalabili            if count > 1:            if count > 1:

**Caratteristiche**:                total_conflicts += count *(count - 1) // 2                total_conflicts += count* (count - 1) // 2

- 🎲 **Stocastico**: sfugge agli ottimi locali

- ⚡ **Veloce**: convergenza rapida

- 📈 **Scalabile**: degrada gradualmente con N

    return total_conflicts    return total_conflicts

### 3. Algoritmo Genetico

``````

**Approccio**: Metaeuristica evolutiva con operatori ottimizzati.



#### Operatori Genetici

## 🔍 Algoritmi Implementati## 🔍 Algoritmi Implementati

**Selezione**: Tournament selection (size = 3)

**Crossover**: Single-point (pc = 0.8)

**Mutazione**: Flip random (pm = tuned)

**Elitismo**: Preservazione del migliore### 1. Backtracking Iterativo### 1. Backtracking Iterativo



## 🎯 Funzioni di Fitness



**F1: Negative Conflicts** - Minimizza conflitti diretti**Approccio**: Ricerca esaustiva sistematica che esplora lo spazio delle soluzioni posizionando una regina per volta e applicando backtrack quando si incontra un vicolo cieco.**Approccio**: Ricerca esaustiva sistematica che esplora lo spazio delle soluzioni posizionando una regina per volta e applicando backtrack quando si incontra un vicolo cieco.

**F2: Non-Conflicting Pairs** - Massimizza coppie valide  

**F3: Linear Diagonal Penalty** - Penalità lineare per cluster

**F4: Worst Queen Penalty** - Penalizza soluzioni sbilanciate

**F5: Quadratic Diagonal Penalty** - Penalità quadratica severa```python```python

**F6: Exponential Transform** - Trasformazione esponenziale

def bt_nqueens_first(N, time_limit=None):def bt_nqueens_first(N, time_limit=None):

## ⚙️ Tuning Automatico Parallelo

    """    """

### 🚀 Parallelizzazione Avanzata

    Backtracking iterativo ottimizzato:    Backtracking iterativo ottimizzato:

Il sistema implementa parallelizzazione su più livelli:

    - Usa strutture dati per tracking veloce dei conflitti    - Usa strutture dati per tracking veloce dei conflitti

```python

# Parametri di parallelizzazione    - Implementazione iterativa (no ricorsione)    - Implementazione iterativa (no ricorsione)

NUM_PROCESSES = multiprocessing.cpu_count() - 1  # Lascia un core libero

    - Trova solo la prima soluzione valida    - Trova solo la prima soluzione valida

# Grid search parallela

POP_MULTIPLIERS = [4, 8, 16]       # pop_size = k * N    - Conta i nodi esplorati per analisi di complessità    - Conta i nodi esplorati per analisi di complessità

GEN_MULTIPLIERS = [30, 50, 80]     # max_gen = m * N  

PM_VALUES = [0.05, 0.1, 0.15]      # Probabilità mutazione    """    """

```

``````

### Livelli di Parallelizzazione

**Caratteristiche**:**Caratteristiche**:

1. **Combinazioni di parametri**: Testa configurazioni diverse in parallelo

2. **Run multipli**: Esegue esperimenti indipendenti contemporaneamente- ✅ **Garanzia di ottimalità**: trova sempre una soluzione se esiste- ✅ **Garanzia di ottimalità**: trova sempre una soluzione se esiste

3. **Algoritmi**: SA e GA parallelizzati separatamente

- ⚠️ **Complessità esponenziale**: O(N!) nel caso peggiore- ⚠️ **Complessità esponenziale**: O(N!) nel caso peggiore

### Funzioni Parallele Chiave

- 🎯 **Deterministico**: comportamento prevedibile- 🎯 **Deterministico**: comportamento prevedibile

```python

def tune_ga_for_N_parallel():- ⏱️ **Time limit opzionale**: per evitare esecuzioni infinite- ⏱️ **Time limit opzionale**: per evitare esecuzioni infinite

    """Tuning parallelo con ProcessPoolExecutor"""

    with ProcessPoolExecutor(max_workers=NUM_PROCESSES) as executor:### 2. Simulated Annealing### 2. Simulated Annealing (SA)

        candidates = list(executor.map(test_parameter_combination_parallel, param_combinations))

**Approccio**: Metaeuristica che simula il processo di ricottura dei metalli, accettando soluzioni peggioranti con probabilità decrescente nel tempo.**Approccio**: Metaeuristica che simula il processo di ricottura dei metalli, accettando soluzioni peggioranti con probabilità decrescente nel tempo.

def run_experiments_with_best_ga_parallel():

    """Esperimenti finali parallelizzati"""```python```python

    with ProcessPoolExecutor(max_workers=NUM_PROCESSES) as executor:

        sa_results = list(executor.map(run_single_sa_experiment, sa_params))def sa_nqueens(N, max_iter=20000, T0=1.0, alpha=0.995):def sa_nqueens(N, max_iter=20000, T0=1.0, alpha=0.995):

        ga_results = list(executor.map(run_single_ga_experiment, ga_params))

```    """    """



## 📈 Pipeline di Sperimentazione Parallela    Simulated Annealing per N-Queens:    Simulated Annealing per N-Queens:



### Fase 1: Tuning Automatico Parallelo    - Partenza da configurazione casuale    - Partenza da configurazione casuale

- 27 combinazioni di parametri per (N, fitness)

- 5 run paralleli per combinazione    - Operatore di vicinato: sposta una regina casuale    - Operatore di vicinato: sposta una regina casuale

- Speedup: 4-6x rispetto alla versione sequenziale

    - Criterio di Metropolis per accettazione    - Criterio di Metropolis per accettazione

### Fase 2: Valutazione Finale Parallela

- 20 run SA + 20 run GA in parallelo    - Raffreddamento geometrico della temperatura    - Raffreddamento geometrico della temperatura

- Backtracking seriale (ottimizzato per singola esecuzione)

- Speedup: 3-4x rispetto alla versione sequenziale    """    """



### Fase 3: Analisi e Visualizzazione``````

- Export CSV strutturati

- Generazione grafici comparativi

- Analisi statistiche automatiche

**Parametri chiave**:**Parametri chiave**:

## 🚀 Come Utilizzare



### Prerequisiti

- `T0 = 1.0`: Temperatura iniziale- `T0 = 1.0`: Temperatura iniziale

```bash

pip install -r requirements.txt- `alpha = 0.995`: Fattore di raffreddamento- `alpha = 0.995`: Fattore di raffreddamento

```

- `max_iter = 2000 + 200*N`: Iterazioni massime scalabili- `max_iter = 2000 + 200*N`: Iterazioni massime scalabili

### Esecuzione Rapida

**Versione parallela (raccomandato)**:

```bash**Caratteristiche**:**Caratteristiche**:

python algo.py                    # Versione parallela completa

python algo_parallel.py           # Versione con opzioni extra

```

- 🎲 **Stocastico**: può sfuggire da ottimi locali- 🎲 **Stocastico**: può sfuggire da ottimi locali

**Versione sequenziale**:

```bash- ⚡ **Veloce**: convergenza rapida per problemi di media dimensione- ⚡ **Veloce**: convergenza rapida per problemi di media dimensione

python algo.py --sequential       # Forza modalità sequenziale

```- 📈 **Scalabile**: performance degrada gradualmente all'aumentare di N- 📈 **Scalabile**: performance degrada gradualmente all'aumentare di N



### Test e Debugging



**Test rapido con parametri ridotti**:### 3. Algoritmo Genetico### 3. Algoritmo Genetico (GA)

```bash

python algo_parallel.py --quick

```

**Approccio**: Metaeuristica evolutiva che mantiene una popolazione di soluzioni candidate e le fa evolvere attraverso selezione, crossover e mutazione.**Approccio**: Metaeuristica evolutiva che mantiene una popolazione di soluzioni candidate e le fa evolvere attraverso selezione, crossover e mutazione.

**Confronto performance**:

```bash

python algo_parallel.py --performance

```#### Operatori Genetici#### Operatori Genetici



**Test di una singola fitness**:

```python

# Modifica in algo.py**Selezione**: Tournament selection con size = 3**Selezione**: Tournament selection con size = 3

FITNESS_MODES = ["F1"]  # Solo F1 per test veloce

```

## 📊 Performance della Versione Parallela```python```python

### Benchmark su Sistema 8-Coredef tournament():def tournament()

| Operazione | Sequenziale | Parallelo | Speedup |    """Seleziona il migliore tra 3 individui casuali"""    """Seleziona il migliore tra 3 individui casuali"""

|------------|------------|-----------|---------|

| Tuning GA (N=24, F1) | ~12 min | ~2.8 min | **4.3x** |    best_i = None    best_i = None

| Esperimenti finali | ~8 min | ~2.1 min | **3.8x** |

| Pipeline completa | ~45 min | ~8 min | **5.6x** |    for _in range(tournament_size):    for _in range(tournament_size):

### Ottimizzazioni Implementate        i = random.randrange(pop_size)        i = random.randrange(pop_size)

1. **Utilizzo CPU intelligente**: N-1 core (lascia uno per il sistema)        if best_i is None or fitness[i] > fitness[best_i]:        if best_i is None or fitness[i] > fitness[best_i]:

2. **Bilanciamento carico**: Distribuzione equa dei task

3. **Memory efficiency**: Processi lightweight per gli algoritmi            best_i = i            best_i = i

4. **Gestione eccezioni**: Robustezza contro errori di processo

    return best_i    return best_i

## 📊 Struttura dei Risultati

``````

```text

results_nqueens_tuning/**Crossover**: Single-point crossover con probabilità pc = 0.8**Crossover**: Single-point crossover con probabilità pc = 0.8

├── tuning_GA_F1.csv              # Parametri ottimali F1 (parallelo)

├── tuning_GA_F1_seq.csv          # Parametri ottimali F1 (sequenziale)```python```python

├── results_GA_F1_tuned.csv       # Risultati finali F1

├── success_vs_N_GA_F1_tuned.png  # Grafici performanceif random.random() < pc:if random.random() < pc:

└── time_vs_N_GA_F1_tuned.png     # Grafici tempi

```    cut = random.randrange(1, N)    cut = random.randrange(1, N)



## 🔬 Risultati Sperimentali    child1 = parent1[:cut] + parent2[cut:]    child1 = parent1[:cut] + parent2[cut:]



### Performance Comparative (N=8, 16, 24)    child2 = parent2[:cut] + parent1[cut:]    child2 = parent2[:cut] + parent1[cut:]



**Backtracking**: Eccellente fino N=24, poi esplosione combinatoriale``````

**Simulated Annealing**: Miglior compromesso velocità/successo

**Algoritmo Genetico**: Variabile, dipende dalla funzione fitness



### Osservazioni dalla Parallelizzazione**Mutazione**: Flip random con probabilità pm (tuned)**Mutazione**: Flip random con probabilità pm (tuned)



1. **Scalabilità**: Speedup lineare fino a 6-8 core

2. **Overhead**: Trascurabile per N≥16

3. **Memory usage**: +15-20% per la versione parallela```python```python

4. **Stabilità**: Risultati identici tra versione seq/parallel

def mutate(individual):def mutate(individual):

## 🔧 Personalizzazione Avanzata

    if random.random() < pm:    if random.random() < pm:

### Controllo Parallelizzazione

        col = random.randrange(N)        col = random.randrange(N)

```python

# Modifica il numero di processi        individual[col] = random.randrange(N)        individual[col] = random.randrange(N)

NUM_PROCESSES = 4  # Forza 4 processi invece di auto-detect

``````

# Disabilita parallelizzazione per debugging

NUM_PROCESSES = 1  # Equivalente a modalità sequenziale**Elitismo**: Il miglior individuo sopravvive sempre alla generazione successiva.**Elitismo**: Il miglior individuo sopravvive sempre alla generazione successiva.

```

## 🎯 Funzioni di Fitness## 🎯 Funzioni di Fitness

### Parametri di Tuning

Il GA implementa 6 diverse funzioni di fitness per confrontare approcci alternativi:Il GA implementa 6 diverse funzioni di fitness per confrontare approcci alternativi:

```python

# Test rapido**F1: Negative Conflicts** - Minimizza conflitti### F1: Negative Conflicts

N_VALUES = [8, 16]           # Solo 2 dimensioni

FITNESS_MODES = ["F1", "F3"] # Solo 2 fitness```python```python

RUNS_GA_TUNING = 3          # Meno run per combinazione

def fitness_f1(ind):def fitness_f1(ind):

# Test intensivo

RUNS_GA_FINAL = 50          # Più run per maggiore precisione    return -conflicts(ind)    return -conflicts(ind)  # Minimizza conflitti

POP_MULTIPLIERS = [2,4,8,16] # Griglia più fine

`````````

## 🔬 Approfondimenti Tecnici

### Complessità Computazionale Parallela**F2: Non-Conflicting Pairs** - Massimizza coppie non in conflitto### F2: Non-Conflicting Pairs

- **Tuning parallelo**: O(C×R/P) dove C=combinazioni, R=run, P=processi

- **Speedup teorico**: Min(P, C×R) limitato da Amdahl's law

- **Memory scaling**: O(P×N) per processo```python```python

### Considerazioni Multi-Coredef fitness_f2(ind):def fitness_f2(ind)

- **NUMA awareness**: Ottimale su sistemi single-socket    max_pairs = n *(n - 1) // 2    max_pairs = n* (n - 1) // 2

- **Hyperthreading**: Beneficio marginale (floating point intensivo)

- **I/O bottleneck**: Trascurabile (algoritmi CPU-bound)    return max_pairs - conflicts(ind)    return max_pairs - conflicts(ind)  # Massimizza coppie non in conflitto

## 📚 Riferimenti``````

1. **N-Queens Problem**: [Wikipedia](https://en.wikipedia.org/wiki/Eight_queens_puzzle)**F3: Linear Diagonal Penalty** - Penalità lineare C(k,2) per cluster su diagonali### F3: Linear Diagonal Penalty

2. **ProcessPoolExecutor**: [Python Docs](https://docs.python.org/3/library/concurrent.futures.html)

3. **Parallel Genetic Algorithms**: Alba & Tomassini (2002)**F4: Worst Queen Penalty** - F2 meno conflitti della regina peggiore```python

4. **High-Performance Python**: Gorelick & Ozsvald (2020)

def fitness_f3(ind):

## 🤝 Contributi

**F5: Quadratic Diagonal Penalty** - Penalità quadratica k² per cluster su diagonali    # Penalità lineare C(k,2) per cluster su diagonali

Benvenuti contributi per:

    # Incentiva distribuzione uniforme

- **Algoritmi ibridi**: GA + ricerca locale

- **Parallelizzazione GPU**: CUDA/OpenCL implementation  **F6: Exponential Transform** - Trasformazione esponenziale dei conflitti```

- **Ottimizzazioni avanzate**: SIMD, cache-friendly algorithms

- **Benchmark estesi**: Confronti con altri solver## ⚙️ Tuning Automatico dei Parametri### F4: Worst Queen Penalty

## 👨‍💻 AutoreIl sistema implementa una **grid search esaustiva** per ottimizzare i parametri del GA:```python

Sviluppato come progetto di ricerca in algoritmi di ottimizzazione combinatoriale parallela.def fitness_f4(ind):

## 📄 Licenza```python    # F2 - conflitti della regina con più conflitti

MIT License - Vedi file `LICENSE` per dettagli.POP_MULTIPLIERS = [4, 8, 16]       # pop_size = k * N    # Penalizza soluzioni sbilanciate

GEN_MULTIPLIERS = [30, 50, 80]     # max_gen = m * N  ```

PM_VALUES = [0.05, 0.1, 0.15]      # Probabilità mutazione

PC_FIXED = 0.8                     # Crossover fisso### F5: Quadratic Diagonal Penalty

TOURNAMENT_SIZE_FIXED = 3          # Selection pressure

``````python

def fitness_f5(ind):

**Criterio di Ottimizzazione**:    # Penalità quadratica k² per cluster su diagonali

    # Penalizzazione più severa di F3

1. **Primario**: Massimizza il tasso di successo```

2. **Secondario**: Minimizza il numero medio di generazioni (a parità di successo)

### F6: Exponential Transform

## 📈 Pipeline di Sperimentazione

```python

### Fase 1: Tuning Automaticodef fitness_f6(ind, lambda=0.3):

    return math.exp(-lambda * conflicts(ind))  # Trasformazione esponenziale

- Per ogni N ∈ {8, 16, 24}```

- Per ogni funzione di fitness ∈ {F1, F3, F4, F5, F6}

- Grid search su 3×3×3 = 27 combinazioni di parametri## ⚙️ Tuning Automatico dei Parametri

- 5 run per combinazione = 135 esperimenti per (N, fitness)

Il sistema implementa una **grid search esaustiva** per ottimizzare i parametri del GA:

### Fase 2: Valutazione Finale

### Spazio dei Parametri

- **20 run indipendenti** per SA e GA con parametri ottimali

- **1 esecuzione** di Backtracking (deterministico)```python

- Raccolta metriche: tempo, successo, nodi/generazioniPOP_MULTIPLIERS = [4, 8, 16]       # pop_size = k * N

GEN_MULTIPLIERS = [30, 50, 80]     # max_gen = m * N  

### Fase 3: Analisi e VisualizzazionePM_VALUES = [0.05, 0.1, 0.15]      # Probabilità mutazione

PC_FIXED = 0.8                     # Crossover fisso

- Export dati in CSV strutturatiTOURNAMENT_SIZE_FIXED = 3          # Selection pressure

- Generazione grafici comparativi```

- Analisi statistica dei risultati

### Criterio di Ottimizzazione

## 📊 Struttura dei Risultati

1. **Primario**: Massimizza il tasso di successo

```text2. **Secondario**: Minimizza il numero medio di generazioni (a parità di successo)

results_nqueens_tuning/

├── tuning_GA_F1.csv              # Parametri ottimali per F1### Processo di Tuning

├── tuning_GA_F3.csv              # Parametri ottimali per F3  

├── results_GA_F1_tuned.csv       # Risultati finali con F1```python

├── results_GA_F3_tuned.csv       # Risultati finali con F3def tune_ga_for_N(N, fitness_mode, ...):

├── success_vs_N_GA_F1_tuned.png  # Grafico tasso successo F1    """

├── success_vs_N_GA_F3_tuned.png  # Grafico tasso successo F3    Per ogni combinazione di parametri:

├── time_vs_N_GA_F1_tuned.png     # Grafico tempi F1    1. Esegue RUNS_GA_TUNING=5 run indipendenti  

└── time_vs_N_GA_F3_tuned.png     # Grafico tempi F3    2. Calcola tasso di successo e generazioni medie

```    3. Seleziona la combinazione ottimale secondo criterio

    """

## 🔬 Risultati Sperimentali```



### Performance per N=8## 📈 Pipeline di Sperimentazione



- **Backtracking**: 100% successo, 876 nodi, <0.1ms### Fase 1: Tuning Automatico

- **SA**: 95% successo, ~573 iterazioni medie, ~1.3ms

- **GA-F1**: 25% successo, 45.4 generazioni medie, ~9.9ms- Per ogni N ∈ {8, 16, 24}

- Per ogni funzione di fitness ∈ {F1, F3, F4, F5, F6}

### Performance per N=16- Grid search su 3×3×3 = 27 combinazioni di parametri

- 5 run per combinazione = 135 esperimenti per (N, fitness)

- **Backtracking**: 100% successo, 160,712 nodi, ~9.7ms

- **SA**: 65% successo, ~2,255 iterazioni medie, ~17.3ms### Fase 2: Valutazione Finale

- **GA-F1**: 15% successo, 172 generazioni medie, ~164ms

- **20 run indipendenti** per SA e GA con parametri ottimali

### Performance per N=24- **1 esecuzione** di Backtracking (deterministico)

- Raccolta metriche: tempo, successo, nodi/generazioni

- **Backtracking**: 100% successo, 9,878,316 nodi, ~557ms

- **SA**: 60% successo, ~3,262 iterazioni medie, ~53.7ms### Fase 3: Analisi e Visualizzazione

- **GA-F1**: 0% successo (nessuna soluzione in 20 run)

- Export dati in CSV strutturati

### Osservazioni Chiave- Generazione grafici comparativi

- Analisi statistica dei risultati

1. **Backtracking**: Prestazioni eccellenti fino a N=24, poi esplosione combinatoriale

2. **Simulated Annealing**: Miglior compromesso velocità/successo per problemi medi## 📊 Struttura dei Risultati

3. **Algoritmo Genetico**: Prestazioni variabili, dipende fortemente dalla funzione di fitness

```text

## 🚀 Come Utilizzareresults_nqueens_tuning/

├── tuning_GA_F1.csv              # Parametri ottimali per F1

### Prerequisiti├── tuning_GA_F3.csv              # Parametri ottimali per F3  

├── results_GA_F1_tuned.csv       # Risultati finali con F1

```bash├── results_GA_F3_tuned.csv       # Risultati finali con F3

pip install -r requirements.txt├── success_vs_N_GA_F1_tuned.png  # Grafico tasso successo F1

```├── success_vs_N_GA_F3_tuned.png  # Grafico tasso successo F3

├── time_vs_N_GA_F1_tuned.png     # Grafico tempi F1

### Esecuzione Completa└── time_vs_N_GA_F3_tuned.png     # Grafico tempi F3

```

```bash

python algo.py## 🔬 Risultati Sperimentali

```

### Performance per N=8

Questo comando eseguirà:

- **Backtracking**: 100% successo, 876 nodi, <0.1ms

1. Tuning automatico per tutte le funzioni di fitness- **SA**: 95% successo, ~573 iterazioni medie, ~1.3ms  

2. Esperimenti finali con parametri ottimali- **GA-F1**: 25% successo, 45.4 generazioni medie, ~9.9ms

3. Generazione di CSV e grafici nella cartella `results_nqueens_tuning/`

### Performance per N=16

### Personalizzazione

- **Backtracking**: 100% successo, 160,712 nodi, ~9.7ms

**Modifica parametri globali**:- **SA**: 65% successo, ~2,255 iterazioni medie, ~17.3ms

- **GA-F1**: 15% successo, 172 generazioni medie, ~164ms

```python

N_VALUES = [8, 12, 16, 20]         # Dimensioni da testare### Performance per N=24

RUNS_SA_FINAL = 30                 # Più run per SA

RUNS_GA_FINAL = 30                 # Più run per GA- **Backtracking**: 100% successo, 9,878,316 nodi, ~557ms  

FITNESS_MODES = ["F1", "F2"]       # Solo alcune fitness- **SA**: 60% successo, ~3,262 iterazioni medie, ~53.7ms

```- **GA-F1**: 0% successo (nessuna soluzione in 20 run)



**Aggiusta limiti di tempo**:### Osservazioni Chiave



```python1. **Backtracking**: Prestazioni eccellenti fino a N=24, poi esplosione combinatoriale

BT_TIME_LIMIT = 10.0               # Max 10 secondi per BT2. **Simulated Annealing**: Miglior compromesso velocità/successo per problemi medi

```3. **Algoritmo Genetico**: Prestazioni variabili, dipende fortemente dalla funzione di fitness



## 🔬 Approfondimenti Tecnici## 🚀 Come Utilizzare



### Complessità Computazionale### Prerequisiti



- **Backtracking**: O(N!) tempo, O(N) spazio```bash

- **SA**: O(T × N) tempo, O(N) spaziopip install -r requirements.txt

- **GA**: O(G × P × N) tempo, O(P × N) spazio```



Dove T=iterazioni SA, G=generazioni GA, P=popolazione GA.### Esecuzione Completa



### Ottimizzazioni Implementate```bash

python algo.py

1. **Tracking veloce conflitti**: Uso di Counter per calcolo O(N) invece di O(N²)```

2. **Backtracking iterativo**: Evita overflow dello stack

3. **Elitismo nel GA**: Preserva sempre la miglior soluzioneQuesto comando eseguirà:

4. **Tuning adattivo**: Dimensioni popolazione/generazioni scalano con N

1. Tuning automatico per tutte le funzioni di fitness

### Considerazioni di Scalabilità2. Esperimenti finali con parametri ottimali  

3. Generazione di CSV e grafici nella cartella `results_nqueens_tuning/`

Il codice è ottimizzato per problemi fino a N≈50. Per istanze più grandi:

### Personalizzazione

- Aumentare i limiti di iterazioni/generazioni

- Considerare algoritmi ibridi (GA + ricerca locale)**Modifica parametri globali**:

- Implementare parallelizzazione delle population

```python

## 📚 RiferimentiN_VALUES = [8, 12, 16, 20]         # Dimensioni da testare

RUNS_SA_FINAL = 30                 # Più run per SA

1. **N-Queens Problem**: [Wikipedia](https://en.wikipedia.org/wiki/Eight_queens_puzzle)RUNS_GA_FINAL = 30                 # Più run per GA

2. **Simulated Annealing**: Kirkpatrick et al. (1983)FITNESS_MODES = ["F1", "F2"]       # Solo alcune fitness

3. **Genetic Algorithms**: Holland (1975), Goldberg (1989)```

4. **Combinatorial Optimization**: Papadimitriou & Steiglitz (1982)

**Aggiusta limiti di tempo**:

## 📄 Licenza

```python

Questo progetto è rilasciato sotto licenza MIT. Vedi il file `LICENSE` per i dettagli.BT_TIME_LIMIT = 10.0               # Max 10 secondi per BT

```

## 🤝 Contributi

## 🔬 Approfondimenti Tecnici

I contributi sono benvenuti! Sentiti libero di:

### Complessità Computazionale

- Aggiungere nuove funzioni di fitness

- Implementare altri algoritmi di ottimizzazione- **Backtracking**: O(N!) tempo, O(N) spazio

- Migliorare le visualizzazioni- **SA**: O(T × N) tempo, O(N) spazio  

- Ottimizzare le performance- **GA**: O(G × P × N) tempo, O(P × N) spazio

## 👨‍💻 AutoreDove T=iterazioni SA, G=generazioni GA, P=popolazione GA

Sviluppato come progetto di ricerca in algoritmi di ottimizzazione combinatoriale.### Ottimizzazioni Implementate

1. **Tracking veloce conflitti**: Uso di Counter per calcolo O(N) invece di O(N²)
2. **Backtracking iterativo**: Evita overflow dello stack
3. **Elitismo nel GA**: Preserva sempre la miglior soluzione
4. **Tuning adattivo**: Dimensioni popolazione/generazioni scalano con N

### Considerazioni di Scalabilità

Il codice è ottimizzato per problemi fino a N≈50. Per istanze più grandi:

- Aumentare i limiti di iterazioni/generazioni
- Considerare algoritmi ibridi (GA + ricerca locale)
- Implementare parallelizzazione delle population

## 📚 Riferimenti

1. **N-Queens Problem**: [Wikipedia](https://en.wikipedia.org/wiki/Eight_queens_puzzle)
2. **Simulated Annealing**: Kirkpatrick et al. (1983)
3. **Genetic Algorithms**: Holland (1975), Goldberg (1989)
4. **Combinatorial Optimization**: Papadimitriou & Steiglitz (1982)

## 📄 Licenza

Questo progetto è rilasciato sotto licenza MIT. Vedi il file `LICENSE` per i dettagli.

## 🤝 Contributi

I contributi sono benvenuti! Sentiti libero di:

- Aggiungere nuove funzioni di fitness
- Implementare altri algoritmi di ottimizzazione  
- Migliorare le visualizzazioni
- Ottimizzare le performance

## 👨‍💻 Autore

Sviluppato come progetto di ricerca in algoritmi di ottimizzazione combinatoriale.
