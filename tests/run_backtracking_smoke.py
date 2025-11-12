import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
module_path = ROOT / "nqueens" / "backtracking.py"

spec = importlib.util.spec_from_file_location("nqueens.backtracking", str(module_path))
back = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = back
spec.loader.exec_module(back)

print("Module loaded:", back)

for fn_name in ("bt_nqueens_first", "bt_nqueens_mcv", "bt_nqueens_lcv"):
    fn = getattr(back, fn_name)
    print(f"Running {fn_name}() for N=8")
    sol, nodes, elapsed = fn(8, time_limit=2.0)
    print(f"  -> sol is None? {sol is None}, nodes={nodes}, elapsed={elapsed:.4f}s")
    if sol is not None:
        assert len(sol) == 8
        print("  -> sample solution:", sol)

print("Smoke test finished.")
