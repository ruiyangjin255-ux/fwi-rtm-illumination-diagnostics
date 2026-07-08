from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter

from rtm_acoustic.diagnostics.matched_budget import BudgetMatchError, scale_to_budget


@dataclass(frozen=True)
class ReliabilityConfig:
    eps: float = 1.0e-8
    robust_percentiles: tuple[float, float] = (1.0, 99.0)
    coverage: float = 0.3635
    sigma_x: float = 4.0
    sigma_z: float = 4.0
    alpha_max: float = 0.3


def config_hash(config: dict[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def array_hash(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(array)
    return hashlib.sha256(arr.view(np.uint8)).hexdigest()[:16]


def compute_gradient_consensus(group_gradients: np.ndarray, eps: float = 1.0e-8) -> np.ndarray:
    groups = np.asarray(group_gradients, dtype=float)
    if groups.ndim < 3:
        raise ValueError("group_gradients must have shape (groups, nz, nx)")
    numerator = np.abs(np.sum(groups, axis=0))
    denominator = np.sum(np.abs(groups), axis=0) + float(eps)
    return np.clip(numerator / denominator, 0.0, 1.0).astype(np.float32)


def compute_illumination_score(
    hessian_proxy: np.ndarray,
    robust_percentiles: tuple[float, float] = (1.0, 99.0),
    eps: float = 1.0e-8,
) -> np.ndarray:
    proxy = np.asarray(hessian_proxy, dtype=float)
    if proxy.ndim == 3:
        proxy = np.sum(proxy, axis=0)
    log_proxy = np.log(np.maximum(proxy, 0.0) + float(eps))
    low, high = np.percentile(log_proxy, robust_percentiles)
    if high <= low:
        return np.zeros_like(log_proxy, dtype=np.float32)
    return np.clip((log_proxy - low) / (high - low), 0.0, 1.0).astype(np.float32)


def compute_descent_alignment(
    delta_model: np.ndarray,
    aggregate_gradient: np.ndarray,
    group_gradients: np.ndarray,
    eps: float = 1.0e-8,
) -> np.ndarray:
    delta = np.asarray(delta_model, dtype=float)
    agg = np.asarray(aggregate_gradient, dtype=float)
    groups = np.asarray(group_gradients, dtype=float)
    denominator = np.abs(delta) * np.sum(np.abs(groups), axis=0) + float(eps)
    aligned = np.maximum(-delta * agg, 0.0) / denominator
    return np.clip(aligned, 0.0, 1.0).astype(np.float32)


def compute_reliability_score(
    illumination: np.ndarray,
    consensus: np.ndarray,
    descent: np.ndarray,
    eps: float = 1.0e-8,
) -> np.ndarray:
    product = np.maximum(illumination, 0.0) * np.maximum(consensus, 0.0) * np.maximum(descent, 0.0)
    return np.power(product + float(eps), 1.0 / 3.0).astype(np.float32)


def top_coverage_support(score: np.ndarray, coverage: float) -> np.ndarray:
    if coverage <= 0.0 or coverage > 1.0:
        raise ValueError("coverage must be in (0, 1]")
    values = np.asarray(score, dtype=float)
    flat = values.ravel()
    count = max(1, int(np.ceil(float(coverage) * flat.size)))
    selected = np.argpartition(flat, flat.size - count)[flat.size - count :]
    support = np.zeros(flat.size, dtype=bool)
    support[selected] = True
    support = support.reshape(values.shape)
    if not np.any(support):
        raise ValueError("coverage support is empty")
    return support


def build_soft_gate(
    score: np.ndarray,
    *,
    coverage: float,
    sigma_x: float,
    sigma_z: float,
    alpha_max: float,
    target_update_l2: float,
    delta_model: np.ndarray,
) -> np.ndarray:
    support = top_coverage_support(score, coverage)
    raw = np.asarray(score, dtype=float) * support.astype(float)
    soft = gaussian_filter(raw, sigma=(float(sigma_z), float(sigma_x)))
    if np.max(soft) <= 0.0:
        raise BudgetMatchError("BUDGET_MATCH_FAILED: smoothed gate is empty")
    soft = soft / float(np.max(soft))
    return scale_to_budget(soft, delta_model, target_update_l2=target_update_l2, alpha_max=alpha_max)


def build_reliability_components(
    *,
    delta_model: np.ndarray,
    group_gradients: np.ndarray,
    hessian_proxy: np.ndarray,
    config: ReliabilityConfig = ReliabilityConfig(),
) -> dict[str, np.ndarray]:
    aggregate = np.sum(np.asarray(group_gradients, dtype=float), axis=0)
    consensus = compute_gradient_consensus(group_gradients, eps=config.eps)
    illumination = compute_illumination_score(hessian_proxy, config.robust_percentiles, eps=config.eps)
    descent = compute_descent_alignment(delta_model, aggregate, group_gradients, eps=config.eps)
    reliability = compute_reliability_score(illumination, consensus, descent, eps=config.eps)
    return {
        "aggregate_gradient": aggregate.astype(np.float32),
        "consensus": consensus,
        "illumination": illumination,
        "descent": descent,
        "reliability": reliability,
    }


def write_manifest(path: Path, *, config: dict[str, Any], arrays: dict[str, np.ndarray], extra: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config_hash": config_hash(config),
        "array_hashes": {name: array_hash(array) for name, array in arrays.items()},
        **extra,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
