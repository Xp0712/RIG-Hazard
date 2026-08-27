from __future__ import annotations

import unittest

from scripts.run_dynamic_hard_budget_experiments import (
    STANDARD_ALGORITHM_METHODS,
    _baseline_taxonomy,
)


class HardBudgetTheorySupplementTests(unittest.TestCase):
    def test_original_six_and_added_standard_families_are_explicit(self) -> None:
        taxonomy = _baseline_taxonomy()
        original = taxonomy.loc[taxonomy["counted_in_original_six"].eq(1)]
        added = taxonomy.loc[taxonomy["coverage_status"].eq("standard_added")]
        self.assertEqual(original.shape[0], 6)
        self.assertEqual(set(added["method"]), set(STANDARD_ALGORITHM_METHODS))
        self.assertIn("online primal-dual", set(added["algorithm_family"]))
        self.assertIn("online stochastic knapsack", set(added["algorithm_family"]))

    def test_partial_existing_algorithms_are_not_mislabelled_as_canonical(self) -> None:
        taxonomy = _baseline_taxonomy().set_index("method")
        self.assertEqual(
            taxonomy.loc["primal_dual_hard_guard", "coverage_status"], "partial"
        )
        self.assertEqual(
            taxonomy.loc["dual_mirror_descent_hard_guard", "coverage_status"],
            "standard_added",
        )


if __name__ == "__main__":
    unittest.main()
