Ecco una **pagina unica** in Markdown pronta per il Wiki GitHub (puoi incollarla come `Home` o `README` del wiki).

---

# N-Queens: BT, SA, GA con Tuning, Fitness F1…F6 e Ottimizzazioni

Questo progetto confronta tre approcci all’N-Queens:

* **BT** (Backtracking) iterativo senza ricorsione
* **SA** (Simulated Annealing)
* **GA** (Algoritmo Genetico) con funzioni di **fitness F1…F6**

Obiettivi:

* Studiare l’impatto di **N** su tasso di successo e costo computazionale
* Valutare **fitness** alternative per GA
* Definire una **procedura di tuning** dei parametri GA al variare di N
* Migliorare le **prestazioni** (O(N) per `conflicts`, multiprocessing, opzionale Numba)

---

## 1) Algoritmi

### Backtracking (BT) iterativo

* Rappresentazione: `board[col] = row`
* Strutture: `row_used`, `diag1_used (r-c)`, `diag2_used (r+c)`
* Restituisce: `solution_found`, `nodes` (tentativi), `time`
* Pro: molto efficace per N piccoli
* Contro: cresce velocemente con N

### Simulated Annealing (SA)

* Stato: configurazione casuale `board[col] = row`
* Costo: `conflicts(board)` = coppie di regine in conflitto
* Mossa: cambia riga di una regina in una colonna
* Accettazione: migliore ⇒ sempre; peggiore ⇒ `exp(-Δ/T)`
* Tipico: `max_iter = 2000 + 200*N`, `T0 = 1.0`, `alpha = 0.995`
* Output run: `success`, `steps`, `time`, `best_conflicts`, `evals`

### Algoritmo Genetico (GA)

* Individuo: lista `ind[col] = row`; popolazione casuale
* Selezione: **torneo (k=3)**
* Crossover: **one-point**
* Mutazione: con `p_m` cambia la riga in una colonna
* Parametri: `pop_size`, `max_gen`, `p_c (0.8)`, `p_m (0.05–0.15)`, `tournament_size (3)`
* Arresto: successo se compare individuo con **0 conflitti**; fallimento a `max_gen`
* Output run: `success`, `gen`, `time`, `best_conflicts`, `evals`

---

## 2) Funzioni di fitness (F1…F6)

> Tutte basate su `conflicts(board)` (righe + diagonali).

* **F1 – ingenua**
  `F1 = -conflicts(ind)`
  Segnale debole: molte soluzioni hanno fitness simili.

* **F2 – coppie non in conflitto**
  `max_pairs = N*(N-1)/2;  F2 = max_pairs - conflicts`
  Per N fissato è **solo F1 traslata** ⇒ *equivalente a F1* per la selezione.

* **F3 – penalità diagonali (lineare)**
  Penalizza cluster su stesse diagonali con `C(cnt,2)`;
  `F3 = max_pairs - penalty_diag`.

* **F4 – F2 meno “regina peggiore”**
  `F4 = F2 - max_conflitti_di_una_regina`
  Penalizza outlier molto conflittuali.

* **F5 – penalità diagonali (quadratica)**
  Come F3, ma penalità ~ `cnt²` per diagonale affollata;
  `F5 = max_pairs - penalty_quad`.

* **F6 – esponenziale sui conflitti**
  `F6 = exp(-λ * conflicts)`, tipicamente `λ=0.3`
  Esalta differenze tra soluzioni quasi buone e cattive.

**Indicazioni pratiche:**

* Usa **F1/F2** come baseline (spesso **deludenti**).
* Metti il focus su **F3** (e se serve **F5/F6**).

---

## 3) Metriche

Per SA/GA su (`N`, `fitness`), eseguire più run indipendenti (es. 20):

* **success_rate** = frazione di run che trovano la soluzione
* **avg_gen_success (GA)** = *media delle generazioni solo sui run riusciti*

  > *“Quando ce la fa, quante generazioni impiega in media?”*
* **avg_time_success** = tempo medio (s) solo sui run riusciti

Distinguono **quanto spesso** si vince da **quanto costa** vincere.

---

## 4) Tuning dei parametri GA (per N e fitness)

Griglia consigliata (più “seria” per F3…F6):

* `pop_size ∈ {max(50, 3N), max(50, 6N), max(50, 10N)}`
* `max_gen ∈ {20N, 40N, 60N}`
* `p_m ∈ {0.05, 0.10, 0.15}`
* `p_c = 0.8`, `tournament_size = 3`
* `RUNS_GA_TUNING = 5` (per non esplodere coi tempi)

**Criterio di selezione**:

1. massimizza `success_rate`
2. a parità, minimizza `avg_gen_success`

**Suggerimenti**:

* Per N ≥ 24, puoi ridurre griglia o run tuning.
* Se trovi `success_rate = 1.0` con poche generazioni ⇒ **early stop**.
* Se il tuo obiettivo è *solo* massimizzare il successo, puoi ignorare il punto (2) e privilegiare pop/max_gen più grandi.

---

## 5) Prestazioni & Speed-up

### 5.1 `conflicts` O(N) (fondamentale)

Sostituisci la versione O(N²) con conteggio di righe e diagonali e combinazioni `C(cnt,2)`.
È il **boost** principale (chiamata migliaia di volte).

### 5.2 Multiprocessing (tuning/esperimenti)

I run SA/GA sono **indipendenti** ⇒ lancia in parallelo con `ProcessPoolExecutor`:

```python
from concurrent.futures import ProcessPoolExecutor, as_completed

with ProcessPoolExecutor() as ex:
    futures = [ex.submit(ga_nqueens, N, pop, gen, pc, pm, tsize, fitness) 
               for _ in range(runs)]
    results = [f.result() for f in as_completed(futures)]
```

Riduce drasticamente **il tempo totale** (non il tempo di un singolo run).

### 5.3 Numba (opzionale)

Compila JIT la parte calcolo-intensiva (es. `conflicts` riscritta con array NumPy e loop semplici).
Guadagni tipicamente **5–20×** a seconda di N.

### 5.4 Altri

* PyPy (JIT interprete alternativo)
* Cython (più lavoro, ma velocissimo)

---

## 6) Interpretare i log (esempi reali F1/F3)

Esempio osservato (tuning su 10 run/comb.):

* **F1/F2**

  * N=8: `success_rate ≈ 0.5–0.6`
  * N=16: `≈ 0.2–0.4`
  * N=24: `≈ 0.2`
    ⇒ Fitness **povera**, segnale debole anche aumentando poco pop/gen.

* **F3** (con griglia iniziale “piccola”)

  * N=8: `success_rate ≈ 0.5` con pop=50, gen=160
  * N=16: `≈ 0.1` con pop=64, gen=320
    ⇒ **Sotto-dimensionato**. Aumentare pop/max_gen (es. 6–10N e 40–60N) migliora.

**Messaggio chiave per il report:**
popolazioni **scarse** e `max_gen` **basso** portano a `success_rate` insoddisfacenti;
con budget più alto (pop grandi + più generazioni) **il successo cresce**, ma **è normale sia più lento**.

---

## 7) Come eseguire

### Requisiti

* Python 3.x
* `matplotlib` (grafici)
* (opz.) `numpy`, `numba`

```bash
pip install matplotlib
# opzionale
pip install numpy numba
```

### Script principale

`nqueens_tuning_all_fitness.py`

* Tuning GA per F1…F6 su `N_VALUES`
* Esperimenti finali BT/SA/GA con **parametri ottimi**
* Salva CSV e grafici in `results_nqueens_tuning/`

Esegui:

```bash
python nqueens_tuning_all_fitness.py
```

### Output

* `tuning_GA_Fk.csv` → parametri GA ottimi per Fk (per N)
* `results_GA_Fk_tuned.csv` → confronto BT/SA/GA-Fk (tuned)
* `success_vs_N_GA_Fk_tuned.png`, `time_vs_N_GA_Fk_tuned.png`

---

## 8) Tabelle/Grafici per il Wiki (template)

**Tuning GA (esempio F3):**

```markdown
| N  | Popolazione | MaxGen | p_m  | p_c | Torneo | Successo tuning |
|----|-------------|--------|------|-----|--------|-----------------|
|  8 | …           | …      | …    | 0.8 | 3      | …               |
| 16 | …           | …      | …    | 0.8 | 3      | …               |
| 24 | …           | …      | …    | 0.8 | 3      | …               |
| 32 | …           | …      | …    | 0.8 | 3      | …               |
```

**Confronto finale (es. GA-F3 tuned):**

```markdown
| N  | BT succ. | BT nodi | BT s | SA succ. | SA iter medie | SA s | GA succ. | GA gen medie | GA s |
|----|----------|---------|------|----------|---------------|------|----------|--------------|------|
|  8 | …        | …       | …    | …        | …             | …    | …        | …            | …    |
| 16 | …        | …       | …    | …        | …             | …    | …        | …            | …    |
| 24 | …        | …       | …    | …        | …             | …    | …        | …            | …    |
| 32 | …        | …       | …    | …        | …             | …    | …        | …            | …    |
```

**Inserisci grafici PNG:**

```markdown
![Successo vs N (GA-F3 tuned)](results_nqueens_tuning/success_vs_N_GA_F3_tuned.png)

![Tempo vs N (GA-F3 tuned)](results_nqueens_tuning/time_vs_N_GA_F3_tuned.png)
```

---

## 9) FAQ rapide

* **Perché F2 non migliora F1?**
  È F1 traslata: stesso ordinamento di selezione ⇒ comportamento analogo.

* **Che cos’è `avg_gen_success`?**
  Media delle generazioni **solo sui run riusciti**.
  *“Quando il GA ce la fa, quante generazioni usa in media?”*

* **È normale che sia più lento con pop/max_gen grandi?**
  Sì. Costo ~ `pop_size × max_gen × costo_fitness`.
  Più budget ⇒ più successo, ma più tempo.

* **Come ridurre i tempi complessivi?**
  `conflicts` O(N), multiprocessing per tuning/esperimenti, ridurre griglia/run, opzionale Numba.

---

## 10) Raccomandazioni rapide

* Per il **confronto finale**: usa BT, SA “standard” e GA con **fitness F3** (o F5/F6) e **parametri tuning** per ogni N.
* Se il tuning è pesante:

  * Per N piccoli: griglia più ampia.
  * Per N ≥ 24: griglia ridotta (es. pop ∈ {6N, 10N}, gen ∈ {40N, 60N}).
  * `RUNS_GA_TUNING = 5` e multiprocessing.
* Per il **report**: evidenzia che F1/F2 sono baseline deboli; F3+ (con budget adeguato) aumenta il success rate e la robustezza.

---
