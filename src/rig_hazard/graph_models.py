from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from scipy import sparse
from scipy.optimize import minimize

from .baseline_models import EPS, cloglog_probability


@dataclass
class StructuredCloglogHazard:
    """Penalized complementary-log-log hazard with feature-specific structure.

    Non-zero L1 penalties are intended for coefficients constrained to be
    non-negative. This turns the graph penalty into a differentiable linear
    term while L-BFGS-B handles the exact zero boundary.
    """

    feature_names: list[str]
    l2_penalties: np.ndarray
    l1_penalties: np.ndarray
    coefficient_bounds: list[tuple[float | None, float | None]]
    max_iter: int = 300
    tolerance: float = 1e-8
    coefficients: np.ndarray | None = None
    intercept: float = 0.0
    converged: bool = False
    iterations: int = 0
    objective: float | None = None
    message: str = ""

    def __post_init__(self) -> None:
        feature_count = len(self.feature_names)
        self.l2_penalties = np.asarray(self.l2_penalties, dtype=np.float64)
        self.l1_penalties = np.asarray(self.l1_penalties, dtype=np.float64)
        if self.l2_penalties.shape != (feature_count,):
            raise ValueError("l2_penalties must match feature_names")
        if self.l1_penalties.shape != (feature_count,):
            raise ValueError("l1_penalties must match feature_names")
        if len(self.coefficient_bounds) != feature_count:
            raise ValueError("coefficient_bounds must match feature_names")
        if np.any(self.l1_penalties < 0) or np.any(self.l2_penalties < 0):
            raise ValueError("penalties must be non-negative")
        for penalty, bounds in zip(self.l1_penalties, self.coefficient_bounds, strict=True):
            if penalty > 0 and (bounds[0] is None or bounds[0] < 0):
                raise ValueError("L1-penalized coefficients must have a non-negative lower bound")

    def fit(
        self,
        x: sparse.spmatrix | np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
        initial_intercept: float | None = None,
        initial_coefficients: np.ndarray | None = None,
    ) -> "StructuredCloglogHazard":
        matrix = sparse.csr_matrix(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if matrix.ndim != 2 or y.ndim != 1 or matrix.shape[0] != y.size:
            raise ValueError("x and y shapes are inconsistent")
        if matrix.shape[1] != len(self.feature_names):
            raise ValueError("x columns do not match feature_names")

        weight = np.ones(y.size, dtype=np.float64) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
        if weight.shape != y.shape or np.any(weight < 0) or not np.isfinite(weight).all():
            raise ValueError("sample_weight must be finite, non-negative, and match y")
        weight = weight / max(float(weight.mean()), EPS)
        weight_sum = float(weight.sum())

        prevalence = float(np.clip(np.dot(weight, y) / weight_sum, 1e-8, 1.0 - 1e-8))
        parameters = np.zeros(matrix.shape[1] + 1, dtype=np.float64)
        parameters[0] = float(np.log(-np.log1p(-prevalence))) if initial_intercept is None else float(initial_intercept)
        if initial_coefficients is not None:
            initial_coefficients = np.asarray(initial_coefficients, dtype=np.float64)
            if initial_coefficients.shape != (matrix.shape[1],):
                raise ValueError("initial_coefficients has the wrong shape")
            parameters[1:] = initial_coefficients

        def objective(values: np.ndarray) -> tuple[float, np.ndarray]:
            intercept = values[0]
            coefficients = values[1:]
            eta = intercept + matrix @ coefficients
            clipped_eta = np.clip(eta, -30.0, 15.0)
            rate = np.exp(clipped_eta)
            positive_probability = -np.expm1(-rate)
            losses = np.where(y > 0.5, -np.log(np.clip(positive_probability, EPS, 1.0)), rate)

            positive_gradient = np.zeros_like(rate)
            stable = rate < 50.0
            positive_gradient[stable] = -rate[stable] / np.expm1(rate[stable])
            gradient_eta = np.where(y > 0.5, positive_gradient, rate)
            gradient_eta[(eta <= -30.0) | (eta >= 15.0)] = 0.0
            weighted_gradient = weight * gradient_eta

            penalty = 0.5 * np.dot(self.l2_penalties, coefficients * coefficients)
            penalty += np.dot(self.l1_penalties, coefficients)
            loss = float(np.dot(weight, losses) / weight_sum + penalty)
            gradient = np.empty_like(values)
            gradient[0] = weighted_gradient.sum() / weight_sum
            gradient[1:] = np.asarray(matrix.T @ weighted_gradient).ravel() / weight_sum
            gradient[1:] += self.l2_penalties * coefficients + self.l1_penalties
            return loss, gradient

        result = minimize(
            objective,
            parameters,
            method="L-BFGS-B",
            jac=True,
            bounds=[(None, None), *self.coefficient_bounds],
            options={"maxiter": int(self.max_iter), "ftol": float(self.tolerance), "gtol": 1e-9, "maxls": 50},
        )
        self.intercept = float(result.x[0])
        self.coefficients = np.asarray(result.x[1:], dtype=np.float64)
        self.converged = bool(result.success)
        self.iterations = int(result.nit)
        self.objective = float(result.fun)
        self.message = str(result.message)
        return self

    def decision_function(self, x: sparse.spmatrix | np.ndarray) -> np.ndarray:
        if self.coefficients is None:
            raise RuntimeError("Model is not fitted")
        matrix = sparse.csr_matrix(x, dtype=np.float64)
        return self.intercept + np.asarray(matrix @ self.coefficients).ravel()

    def predict_probability(self, x: sparse.spmatrix | np.ndarray, steps: float = 1.0) -> np.ndarray:
        return cloglog_probability(self.decision_function(x), steps=steps)

    def to_dict(self) -> dict[str, Any]:
        if self.coefficients is None:
            raise RuntimeError("Model is not fitted")
        return {
            "feature_names": self.feature_names,
            "l2_penalties": self.l2_penalties.tolist(),
            "l1_penalties": self.l1_penalties.tolist(),
            "coefficient_bounds": [list(value) for value in self.coefficient_bounds],
            "max_iter": self.max_iter,
            "tolerance": self.tolerance,
            "intercept": self.intercept,
            "coefficients": self.coefficients.tolist(),
            "converged": self.converged,
            "iterations": self.iterations,
            "objective": self.objective,
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "StructuredCloglogHazard":
        model = cls(
            feature_names=list(value["feature_names"]),
            l2_penalties=np.asarray(value["l2_penalties"], dtype=np.float64),
            l1_penalties=np.asarray(value["l1_penalties"], dtype=np.float64),
            coefficient_bounds=[tuple(item) for item in value["coefficient_bounds"]],
            max_iter=int(value.get("max_iter", 300)),
            tolerance=float(value.get("tolerance", 1e-8)),
        )
        model.intercept = float(value["intercept"])
        model.coefficients = np.asarray(value["coefficients"], dtype=np.float64)
        model.converged = bool(value.get("converged", True))
        model.iterations = int(value.get("iterations", 0))
        model.objective = value.get("objective")
        model.message = str(value.get("message", ""))
        return model


def penalty_vector(feature_names: Sequence[str], prefix_values: dict[str, float], default: float = 0.0) -> np.ndarray:
    values = []
    for name in feature_names:
        values.append(next((float(value) for prefix, value in prefix_values.items() if name.startswith(prefix)), float(default)))
    return np.asarray(values, dtype=np.float64)
