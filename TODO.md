# TODO

Aggiornato il: 2025-11-12

## Completato

- Modularizzazione orchestrazione sotto `nqueens/analysis/*` (settings, stats, tuning, experiments, reporting, plots, cli)
- `algoanalisys.py` convertito in facciata compatibile all’indietro (re-export API + CLI)
- Plotting opzionale: stub chiari se manca `matplotlib`
- Aggiunta `is_valid_solution(board)` in `nqueens/utils.py` ed esportata dal package
- Quick regression: validazione soluzioni BT con `is_valid_solution`
- Flag CLI `--validate` (+ propagazione ai runner) per check aggiuntivi su BT/SA/GA
- Aggiornata documentazione: README, CHANGELOG (con Migration Notes 2.1.0), wiki (struttura modulare + sezione Validation)

## Prossimi passi (opzionali)

- Estendere SA/GA per restituire la board nelle run di successo (deep validation opt-in; breaking o feature flag?)
- Generare un report HTML leggero che aggreghi alcuni grafici e CSV principali
- Ampliare i grafici statistici quando si persiste l’insieme completo dei "raw runs"
- Aggiungere CI con type-check (mypy) e smoke test (quick regression) su ogni push
- Rifiniture wiki/Markdown per aderire pienamente a tutte le regole di stile

## Promemoria d’uso

- CLI: `python algoanalisys.py [--mode {sequential|parallel|concurrent}] [--fitness ...] [--config config.json] [--validate]`
- Test rapidi: `python algoanalisys.py --quick-test`
- API utility: `from nqueens import is_valid_solution, conflicts`
