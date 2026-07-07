from __future__ import annotations

import numpy as np


def normalized_l2_residual(predicted: np.ndarray, observed: np.ndarray, eps: float = 1.0e-8) -> float:
    residual = np.asarray(predicted, dtype=float) - np.asarray(observed, dtype=float)
    return float(np.linalg.norm(residual) / (np.linalg.norm(observed) + eps))


def nrms_residual(predicted: np.ndarray, observed: np.ndarray, eps: float = 1.0e-8) -> float:
    residual = np.asarray(predicted, dtype=float) - np.asarray(observed, dtype=float)
    denom = np.sqrt(np.mean(np.asarray(observed, dtype=float) ** 2)) + eps
    return float(np.sqrt(np.mean(residual**2)) / denom)


def trace_correlation(predicted: np.ndarray, observed: np.ndarray, eps: float = 1.0e-8) -> float:
    pred = np.asarray(predicted, dtype=float).ravel()
    obs = np.asarray(observed, dtype=float).ravel()
    pred = pred - float(np.mean(pred))
    obs = obs - float(np.mean(obs))
    return float(np.dot(pred, obs) / ((np.linalg.norm(pred) * np.linalg.norm(obs)) + eps))


def envelope_error(predicted: np.ndarray, observed: np.ndarray, eps: float = 1.0e-8) -> float:
    return normalized_l2_residual(np.abs(predicted), np.abs(observed), eps=eps)


def audit_record_pair(predicted: np.ndarray, observed: np.ndarray) -> dict[str, float | None]:
    return {
        "normalized_l2_residual": normalized_l2_residual(predicted, observed),
        "nrms_residual": nrms_residual(predicted, observed),
        "trace_correlation": trace_correlation(predicted, observed),
        "envelope_error": envelope_error(predicted, observed),
        "phase_error_if_available": None,
    }

