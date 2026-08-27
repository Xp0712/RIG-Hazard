from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


EPS = 1e-12


def cloglog_probability(eta: np.ndarray, steps: float = 1.0) -> np.ndarray:
    eta = np.asarray(eta, dtype=np.float64)
    rate = np.exp(np.clip(eta, -30.0, 15.0))
    return -np.expm1(-steps * rate)


def logit_probability(eta: np.ndarray) -> np.ndarray:
    return expit(np.asarray(eta, dtype=np.float64))


def binary_nll(y: np.ndarray, probability: np.ndarray, weight: np.ndarray | None = None) -> float:
    y = np.asarray(y, dtype=np.float64)
    probability = np.clip(np.asarray(probability, dtype=np.float64), EPS, 1.0 - EPS)
    losses = -(y * np.log(probability) + (1.0 - y) * np.log1p(-probability))
    if weight is None:
        return float(losses.mean())
    weight = np.asarray(weight, dtype=np.float64)
    return float(np.dot(weight, losses) / weight.sum())


@dataclass
class WeightedBinaryGLM:
    link: str
    l2: float = 1e-4
    max_iter: int = 250
    tolerance: float = 1e-7
    coefficients: np.ndarray | None = None
    intercept: float = 0.0
    converged: bool = False
    iterations: int = 0
    objective: float | None = None
    message: str = ""
    penalty_weights: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "WeightedBinaryGLM":
        x = np.asarray(x, dtype=np.float64, order="C")
        y = np.asarray(y, dtype=np.float64)
        if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.size:
            raise ValueError("x and y shapes are inconsistent")
        if self.link not in {"logit", "cloglog"}:
            raise ValueError(f"Unsupported link: {self.link}")
        weight = np.ones(y.size, dtype=np.float64) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
        weight = weight / max(float(weight.mean()), EPS)
        prevalence = float(np.dot(weight, y) / weight.sum())
        prevalence = float(np.clip(prevalence, 1e-8, 1.0 - 1e-8))
        if self.link == "logit":
            initial_intercept = float(np.log(prevalence / (1.0 - prevalence)))
        else:
            initial_intercept = float(np.log(-np.log1p(-prevalence)))
        initial = np.zeros(x.shape[1] + 1, dtype=np.float64)
        initial[0] = initial_intercept
        weight_sum = float(weight.sum())
        if self.penalty_weights is None:
            penalty_weights = np.ones(x.shape[1], dtype=np.float64)
        else:
            penalty_weights = np.asarray(self.penalty_weights, dtype=np.float64)
            if penalty_weights.shape != (x.shape[1],) or np.any(penalty_weights < 0):
                raise ValueError("penalty_weights must be a non-negative vector matching the feature count")

        def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
            intercept = parameters[0]
            coefficients = parameters[1:]
            eta = intercept + x @ coefficients
            if self.link == "logit":
                probability = expit(eta)
                losses = np.logaddexp(0.0, eta) - y * eta
                gradient_eta = probability - y
            else:
                clipped_eta = np.clip(eta, -30.0, 15.0)
                rate = np.exp(clipped_eta)
                positive_probability = -np.expm1(-rate)
                losses = np.where(y > 0.5, -np.log(np.clip(positive_probability, EPS, 1.0)), rate)
                positive_gradient = np.zeros_like(rate)
                stable = rate < 50.0
                positive_gradient[stable] = -rate[stable] / np.expm1(rate[stable])
                gradient_eta = np.where(y > 0.5, positive_gradient, rate)
            weighted_gradient = weight * gradient_eta
            loss = float(
                np.dot(weight, losses) / weight_sum
                + 0.5 * self.l2 * np.dot(penalty_weights, np.square(coefficients))
            )
            gradient = np.empty_like(parameters)
            gradient[0] = weighted_gradient.sum() / weight_sum
            gradient[1:] = x.T @ weighted_gradient / weight_sum + self.l2 * penalty_weights * coefficients
            return loss, gradient

        result = minimize(
            objective,
            initial,
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": int(self.max_iter), "ftol": float(self.tolerance), "gtol": 1e-6, "maxls": 40},
        )
        self.intercept = float(result.x[0])
        self.coefficients = np.asarray(result.x[1:], dtype=np.float64)
        self.converged = bool(result.success)
        self.iterations = int(result.nit)
        self.objective = float(result.fun)
        self.message = str(result.message)
        return self

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        if self.coefficients is None:
            raise RuntimeError("Model is not fitted")
        return self.intercept + np.asarray(x, dtype=np.float64) @ self.coefficients

    def predict_probability(self, x: np.ndarray, steps: float = 1.0) -> np.ndarray:
        eta = self.decision_function(x)
        if self.link == "logit":
            if steps != 1.0:
                raise ValueError("steps is only defined for the cloglog hazard model")
            return logit_probability(eta)
        return cloglog_probability(eta, steps=steps)

    def to_dict(self) -> dict[str, Any]:
        if self.coefficients is None:
            raise RuntimeError("Model is not fitted")
        return {
            "link": self.link,
            "l2": self.l2,
            "max_iter": self.max_iter,
            "tolerance": self.tolerance,
            "intercept": self.intercept,
            "coefficients": self.coefficients.tolist(),
            "converged": self.converged,
            "iterations": self.iterations,
            "objective": self.objective,
            "message": self.message,
            "penalty_weights": None if self.penalty_weights is None else self.penalty_weights.tolist(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "WeightedBinaryGLM":
        penalty = value.get("penalty_weights")
        model = cls(
            link=value["link"],
            l2=float(value.get("l2", 0.0)),
            max_iter=int(value.get("max_iter", 250)),
            penalty_weights=None if penalty is None else np.asarray(penalty, dtype=np.float64),
        )
        model.intercept = float(value["intercept"])
        model.coefficients = np.asarray(value["coefficients"], dtype=np.float64)
        model.converged = bool(value.get("converged", True))
        model.iterations = int(value.get("iterations", 0))
        model.objective = value.get("objective")
        model.message = str(value.get("message", ""))
        return model


@dataclass
class HazardRateCalibrator:
    log_rate_shift: float = 0.0
    slope: float = 1.0
    objective: float | None = None
    converged: bool = False

    def fit(
        self,
        eta: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "HazardRateCalibrator":
        eta = np.asarray(eta, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        weight = np.ones(y.size, dtype=np.float64) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
        weight = weight / max(float(weight.mean()), EPS)
        denominator = float(weight.sum())

        def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
            shift, slope = parameters
            calibrated_eta = shift + slope * eta
            clipped_eta = np.clip(calibrated_eta, -30.0, 15.0)
            rate = np.exp(clipped_eta)
            probability = -np.expm1(-rate)
            losses = np.where(y > 0.5, -np.log(np.clip(probability, EPS, 1.0)), rate)
            positive_gradient = np.zeros_like(rate)
            stable = rate < 50.0
            positive_gradient[stable] = -rate[stable] / np.expm1(rate[stable])
            gradient_eta = np.where(y > 0.5, positive_gradient, rate)
            weighted_gradient = weight * gradient_eta
            gradient = np.array(
                [weighted_gradient.sum() / denominator, np.dot(weighted_gradient, eta) / denominator],
                dtype=np.float64,
            )
            return float(np.dot(weight, losses) / denominator), gradient

        result = minimize(
            objective,
            np.array([0.0, 1.0], dtype=np.float64),
            method="L-BFGS-B",
            jac=True,
            bounds=[(-20.0, 20.0), (0.0, 10.0)],
            options={"maxiter": 200, "ftol": 1e-10, "maxls": 40},
        )
        self.log_rate_shift = float(result.x[0])
        self.slope = float(result.x[1])
        self.objective = float(result.fun)
        self.converged = bool(result.success)
        return self

    def predict(self, eta: np.ndarray, steps: float = 1.0) -> np.ndarray:
        calibrated_eta = self.log_rate_shift + self.slope * np.asarray(eta, dtype=np.float64)
        return cloglog_probability(calibrated_eta, steps=steps)

    def to_dict(self) -> dict[str, float | bool | None]:
        return {
            "log_rate_shift": self.log_rate_shift,
            "slope": self.slope,
            "objective": self.objective,
            "converged": self.converged,
        }


@dataclass
class PlattCalibrator:
    intercept: float = 0.0
    slope: float = 1.0
    objective: float | None = None
    converged: bool = False

    def fit(self, score: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        score = np.asarray(score, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
            intercept, slope = parameters
            eta = intercept + slope * score
            probability = expit(eta)
            loss = float(np.mean(np.logaddexp(0.0, eta) - y * eta))
            residual = probability - y
            gradient = np.array([residual.mean(), np.mean(residual * score)], dtype=np.float64)
            return loss, gradient

        result = minimize(
            objective,
            np.array([0.0, 1.0], dtype=np.float64),
            method="L-BFGS-B",
            jac=True,
            bounds=[(-30.0, 30.0), (0.0, 10.0)],
            options={"maxiter": 200, "ftol": 1e-10, "maxls": 40},
        )
        self.intercept = float(result.x[0])
        self.slope = float(result.x[1])
        self.objective = float(result.fun)
        self.converged = bool(result.success)
        return self

    def predict(self, score: np.ndarray) -> np.ndarray:
        return expit(self.intercept + self.slope * np.asarray(score, dtype=np.float64))

    def to_dict(self) -> dict[str, float | bool | None]:
        return {"intercept": self.intercept, "slope": self.slope, "objective": self.objective, "converged": self.converged}
