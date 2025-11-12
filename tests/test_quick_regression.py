"""Quick regression tests for the N-Queens experiment orchestrator."""

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import algo


class QuickRegressionTests(unittest.TestCase):
    """Verify that the lightweight regression checks pass."""

    def test_algorithms_and_csv_generation(self):
        """Ensure BT, SA, GA, and CSV export succeed for N=8."""
        algo.run_quick_regression_tests()


if __name__ == "__main__":
    unittest.main()
