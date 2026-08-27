import unittest

import numpy as np
from scipy import sparse

from rig_hazard.graph_models import StructuredCloglogHazard


class StructuredCloglogHazardTests(unittest.TestCase):
    def make_data(self) -> tuple[sparse.csr_matrix, np.ndarray]:
        rng = np.random.default_rng(41)
        rows = 5000
        local = rng.normal(size=rows)
        graph = rng.binomial(1, 0.18, size=rows) * rng.uniform(0.5, 2.0, size=rows)
        nuisance = rng.normal(size=rows)
        eta = -5.2 + 0.7 * local + 1.15 * graph
        probability = -np.expm1(-np.exp(eta))
        y = rng.binomial(1, probability).astype(np.int8)
        return sparse.csr_matrix(np.column_stack([local, graph, nuisance])), y

    def fit_model(self, graph_l1: float) -> StructuredCloglogHazard:
        x, y = self.make_data()
        model = StructuredCloglogHazard(
            feature_names=["local::eta", "graph::edge", "graph::nuisance"],
            l2_penalties=np.array([1e-4, 1e-4, 1e-4]),
            l1_penalties=np.array([0.0, graph_l1, graph_l1]),
            coefficient_bounds=[(0.0, 3.0), (0.0, 3.0), (0.0, 3.0)],
            max_iter=250,
        )
        return model.fit(x, y)

    def test_nonnegative_sparse_graph_recovers_signal(self) -> None:
        model = self.fit_model(2e-4)
        self.assertTrue(model.converged)
        self.assertGreater(model.coefficients[1], 0.25)
        self.assertGreaterEqual(model.coefficients[2], 0.0)
        self.assertGreater(model.coefficients[1], model.coefficients[2])

    def test_stronger_l1_shrinks_graph_coefficients(self) -> None:
        weak = self.fit_model(0.0)
        strong = self.fit_model(2e-3)
        self.assertLessEqual(strong.coefficients[1], weak.coefficients[1] + 1e-8)
        self.assertLessEqual(strong.coefficients[2], weak.coefficients[2] + 1e-8)

    def test_round_trip_preserves_predictions(self) -> None:
        x, _ = self.make_data()
        model = self.fit_model(2e-4)
        restored = StructuredCloglogHazard.from_dict(model.to_dict())
        np.testing.assert_allclose(model.decision_function(x[:30]), restored.decision_function(x[:30]))


if __name__ == "__main__":
    unittest.main()
