"""Lightweight tests that validate the three BT solvers for N=8.

This test intentionally avoids importing the experiment orchestrator to keep
dependencies minimal and independent from CLI/reporting layers.
"""

import unittest

from algoanalisys import run_quick_regression_tests


class QuickRegressionTests(unittest.TestCase):
    """Use the project’s quick regression runner to validate all algorithms on N=8."""

    def test_quick_regression_runner(self):
        # Expect no exceptions; output is printed for manual inspection if needed
        run_quick_regression_tests()


if __name__ == "__main__":
    unittest.main()
