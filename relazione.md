---
title: "Risoluzione del problema N-Queens con Backtracking, Simulated Annealing e Algoritmo Genetico"
author: "Tuo Nome"
date: "YYYY-MM-DD"
---

# 1. Introduzione

Il problema delle N regine (N-Queens) consiste nel posizionare N regine su una scacchiera N x N in modo che nessuna minacci le altre. In termini formali, nessuna coppia di regine deve condividere la stessa riga, colonna o diagonale.

Questo problema è un classico esempio di problema di soddisfacimento di vincoli e viene spesso utilizzato come benchmark per:

- algoritmi **esatti** (ricerca esaustiva e backtracking)
- algoritmi **metaeuristici** (simulated annealing, algoritmi genetici)
- studio dello **scaling** del costo computazionale al variare della dimensione N

In questo lavoro vengono confrontati tre approcci:

1. **Backtracking (BT)** iterativo, senza ricorsione
2. **Simulated Annealing (SA)**
3. **Algoritmo Genetico (GA)** con diverse funzioni di fitness (F1...F6) e con una procedura di tuning automatico dei parametri

Gli obiettivi principali sono:

- analizzare il tasso di successo dei diversi algoritmi al variare di N
- confrontare il costo computazionale (numero di passi / generazioni e tempo di esecuzione)
- studiare l’effetto di diverse funzioni di fitness sul comportamento del GA
- valutare l’impatto della dimensione della popolazione e del numero massimo di generazioni

---

# 2. Formulazione del problema e rappresentazione

Il problema N-Queens richiede di posizionare N regine su una scacchiera N x N in modo che:

- non ce ne siano due sulla stessa riga
- non ce ne siano due sulla stessa colonna
- non ce ne siano due sulla stessa diagonale (principale o secondaria)

Nel progetto viene adottata la seguente rappresentazione:

- una configurazione è un vettore di interi:  
  `board[col] = row`

dove:

- `col` è l’indice di colonna (0...N-1)
- `row` è la riga della regina in quella colonna

Con questa codifica è garantito che ci sia al massimo una regina per colonna, mentre possono esistere conflitti sulle righe e sulle diagonali.

Due regine sono in conflitto se:

- hanno la stessa riga: `row[i] == row[j]`
- sono sulla stessa diagonale principale: `row[i] - i == row[j] - j`
- sono sulla stessa diagonale secondaria: `row[i] + i == row[j] + j`

La funzione chiave è:

`conflicts(board) → numero di coppie di regine in conflitto`

Per efficienza è stata implementata in O(N), usando contatori per:

- righe (`row_count[row]`)
- diagonali principali (`diag1[row - col]`)
- diagonali secondarie (`diag2[row + col]`)

e sommando, per ogni contatore `cnt > 1`, le combinazioni `cnt * (cnt - 1) / 2`.

---

# 3. Algoritmi considerati

## 3.1 Backtracking (BT) iterativo

L’algoritmo di backtracking esplora lo spazio delle configurazioni colonna per colonna, posizionando una regina alla volta e facendo backtrack quando non sono possibili posizionamenti validi.

Caratteristiche principali:

- implementazione **iterativa**, senza ricorsione, tramite:
  - vettore `pos[col]` per la riga scelta in ogni colonna
  - array booleani:
    - `row_used[row]` (righe occupate)
    - `diag1_used[row - col + (N-1)]`
    - `diag2_used[row + col]`
- viene cercata **una sola soluzione** (la prima trovata)
- viene mantenuto un contatore dei **nodi esplorati** (tentativi di posizionamento)

Per ogni N si misurano:

- `solution_found` (true/false)
- `nodes` (nodi esplorati)
- `time` (tempo per trovare la prima soluzione)

BT risulta molto efficiente per N piccoli, ma rapidamente più costoso al crescere di N.

## 3.2 Simulated Annealing (SA)

Simulated Annealing (SA) è una metaeuristica ispirata ai processi di ricottura in metallurgia. Parte da una soluzione casuale e applica modifiche locali, accettando anche mosse peggiorative in funzione della temperatura.

Elementi principali:

- stato: configurazione `board[col] = row` generata casualmente
- costo: `conflicts(board)` (numero di coppie di regine in conflitto)
- mossa: scelta di una colonna casuale e variazione della riga della relativa regina
- accettazione:
  - se il costo diminuisce, la mossa è sempre accettata
  - se il costo aumenta, la mossa viene accettata con probabilità `exp(-Delta / T)`
  - Delta = aumento dei conflitti, T = temperatura corrente
- raffreddamento:
  - temperatura iniziale `T0`
  - aggiornata con `T = alpha * T` ad ogni iterazione

Parametri tipici nel codice:

- `max_iter = 2000 + 200 * N`
- `T0 = 1.0`
- `alpha = 0.995`

Output per ogni run:

- `success` (true se viene raggiunta una configurazione con 0 conflitti)
- `steps` (numero di iterazioni)
- `time` (tempo di esecuzione)
- `best_conflicts` (numero di conflitti del miglior stato trovato)
- `evals` (numero di valutazioni della funzione `conflicts`)

## 3.3 Algoritmo Genetico (GA)

L’algoritmo genetico (GA) mantiene una popolazione di individui (configurazioni) e applica iterativamente:

- selezione
- crossover
- mutazione

### Rappresentazione

- individuo: lista di N interi `ind[col] = row`
- popolazione iniziale: individui con righe casuali per ogni colonna

### Operatori

- **Selezione**: torneo di dimensione 3 (si estraggono 3 individui a caso e si seleziona il migliore secondo la fitness)
- **Crossover**: one-point (monofrontiera): indice di taglio `k` e combinazione delle porzioni dei genitori
- **Mutazione**: con probabilità `p_m`, per ogni figlio si sceglie una colonna casuale e si assegna una nuova riga casuale alla regina in quella colonna

### Parametri principali

- `pop_size`: dimensione della popolazione
- `max_gen`: numero massimo di generazioni
- `p_c`: probabilità di crossover (tipicamente 0.8)
- `p_m`: probabilità di mutazione (tra 0.05 e 0.15)
- `tournament_size`: dimensione del torneo (3)

### Criterio di arresto

Un singolo run del GA termina quando:

- viene trovato un individuo con 0 conflitti (`success = true`)
- oppure viene raggiunta la generazione `max_gen` senza soluzione (`success = false`)

Output:

- `success` (true/false)
- `gen` (numero di generazioni effettivamente utilizzate)
- `time` (tempo di esecuzione)
- `best_conflicts` (conflitti residui del miglior individuo)
- `evals` (numero di valutazioni di fitness)

---

# 4. Funzioni di fitness per il GA (F1...F6)

Le funzioni di fitness definiscono come il GA valuta gli individui. L’obiettivo è assegnare valori più alti alle configurazioni “migliori” (meno conflitti, struttura desiderabile).

## F1 – fitness ingenua

`F1(ind) = -conflicts(ind)`

Maggiore numero di conflitti porta a fitness più bassa; scelta semplice ma con segnale debole per N grandi.

## F2 – coppie non in conflitto

```
max_pairs = N * (N - 1) / 2
F2(ind) = max_pairs - conflicts(ind)
```

Di fatto trasla F1; ranking identico.

## F3 – penalità sui cluster diagonali (lineare)

Penalizza le diagonali con più regine:

- somma le penalità delle diagonali principali e secondarie per cui `cnt > 1`
- fitness: `F3(ind) = max_pairs - penalty_diag`

## F4 – F2 meno la “regina peggiore”

`F4(ind) = F2(ind) - worst`

Penalizza le configurazioni in cui una regina ha il massimo numero di conflitti.

## F5 – penalità diagonali quadratica

Penalità più aggressiva su diagonali affollate (quadratica):

`F5(ind) = max_pairs - penalty_quad`

## F6 – trasformazione esponenziale

`F6(ind) = exp(-λ * conflicts(ind))` con `λ > 0` (tipicamente 0.3)

---

# 5. Metriche di valutazione

Per confrontare SA e GA vengono eseguiti per ogni combinazione (N, fitness, parametri):

- **BT**: singola esecuzione (deterministica)
- **SA**: più run indipendenti (es. 20)
- **GA**: più run indipendenti (es. 20)

Metriche principali:

- **Tasso di successo (success_rate)**:  
  `success_rate = (# run con success = true) / (# run totali)`
- **Generazioni medie sui run riusciti (avg_gen_success, solo GA)**  
  `avg_gen_success = media dei valori di gen sui run con success = true`
- **Tempo medio sui run riusciti (avg_time_success)**:  
  tempo medio in secondi sui soli run che terminano con successo

---

# 6. Tuning dei parametri GA

I parametri del GA influenzano fortemente il tasso di successo e il tempo di esecuzione. È stata implementata una procedura di tuning automatico.

## 6.1 Griglia di ricerca

Per ogni coppia (N, fitness Fₖ) si esplora una griglia di parametri:

- `pop_size in {max(50, 3N), max(50, 6N), max(50, 10N)}`
- `max_gen in {20N, 40N, 60N}`
- `p_m in {0.05, 0.10, 0.15}`
- `p_c = 0.8`
- `tournament_size = 3`

Per ogni combinazione, vengono eseguiti vari run e si calcolano `success_rate` e `avg_gen_success`.  
I risultati sono salvati in file CSV (es. `tuning_GA_F1.csv`, ecc.).

## 6.2 Criterio di scelta dei parametri ottimali

Segue questo criterio:

1. Massimizzare il tasso di successo (`success_rate`)
2. A parità di success rate, minimizzare `avg_gen_success`

## 6.3 Osservazioni

- Parametri troppo piccoli rendono il GA veloce ma con success rate basso, soprattutto per fitness deboli (F1/F2)
- Parametri generosi aumentano il success rate, ma allungano i tempi: è un trade-off naturale

---

# 7. Risultati sperimentali (qualitativi)

Esempio di tabella tipo (da riempire con i dati reali):

```
Esempio – Confronto BT / SA / GA-F3 (parametri tuning)

| N  | BT succ. | BT nodi  | BT s   | SA succ. | SA iter medie | SA s    | GA-F3 succ. | GA gen medie | GA s    |
|----|----------|----------|--------|----------|---------------|---------|-------------|--------------|---------|
|  8 | 1.00     | ...      | ...    | 1.00     | ...           | ...     | ...         | ...          | ...     |
| 16 | 1.00     | ...      | ...    | ...      | ...           | ...     | ...         | ...          | ...     |
| 24 | 1.00     | ...      | ...    | ...      | ...           | ...     | ...         | ...          | ...     |
| 32 | 1.00     | ...      | ...    | ...      | ...           | ...     | ...         | ...          | ...     |
```

Grafici tipo (da inserire collegando file PNG):

```
Successo vs N (GA-F3 tuned)
results_nqueens_tuning/success_vs_N_GA_F3_tuned.png

Tempo vs N (GA-F3 tuned)
results_nqueens_tuning/time_vs_N_GA_F3_tuned.png
```

## 7.1 Comportamento di F1 e F2

Con F1 e F2 anche dopo tuning il tasso di successo resta limitato (circa 50–60% per N = 8), scende ulteriormente per N maggiori. L’aumento di `pop_size` e `max_gen` migliora solo parzialmente, ma cresce il tempo di esecuzione.

Quindi:

- una fitness basata solo sul numero totale di conflitti fornisce un segnale troppo debole per guidare efficacemente l’evoluzione, specialmente per N grande.

## 7.2 Comportamento di F3, F5 e F6

F3 penalizza i cluster di regine sulle stesse diagonali, migliorando nettamente le prestazioni:

- per N piccoli, successo vicino al 100%
- per N medi e grandi, tassi nettamente superiori rispetto a F1/F2

F5 (penalità quadratica) e F6 (trasformazione esponenziale) danno risultati analoghi, con differenze di dettaglio nei tempi e parametri.

## 7.3 Confronto complessivo BT – SA – GA

- **BT**: per N piccoli è veloce e deterministico, ma scala male per N grandi
- **SA**: offre un buon compromesso tra semplicità, costo e successo; tuning moderato
- **GA/F3+varianti**: parametri adeguati consentono successo anche per N grandi, costo pop_size x max_gen elevato, flessibilità e robustezza superiori

---

# 8. Ottimizzazioni di performance

## 8.1 Funzione conflicts in O(N)

Versione O(N) basata su conteggi di righe e diagonali:  
per ogni contatore `cnt > 1`, aggiunge `cnt * (cnt - 1) / 2` ai conflitti.

## 8.2 Tuning “controllato”

Per N piccoli si può usare una griglia di tuning più ricca; per N grandi, griglia e numero run ridotti. Possibile anche early-stop.

## 8.3 Esecuzione parallela (multiprocessing)

Run SA e GA sono indipendenti e si possono parallelizzare usando ad esempio `ProcessPoolExecutor`.

## 8.4 Compilazione JIT (Numba, opzionale)

Annotando le parti numeriche con Numba (`@njit`) si ottengono notevoli speed-up.

---

# 9. Conclusioni

Sono stati confrontati tre algoritmi per la risoluzione del problema N-Queens:

- backtracking iterativo (BT)
- simulated annealing (SA)
- algoritmo genetico (GA) con fitness e tuning

Conclusioni:

- **BT** efficace per N piccoli, ma scala male
- **SA** buon compromesso, complessità contenuta
- **GA** con funzioni di fitness adeguate (F3/F5/F6) e tuning può essere molto efficace anche per N grandi

Le funzioni F1/F2 basate solo sui conflitti sono insufficienti; serve una fitness più informativa.  
Sviluppi futuri: metodi ibridi, JIT, nuove fitness, operatori crossover/mutazione più sofisticati.
