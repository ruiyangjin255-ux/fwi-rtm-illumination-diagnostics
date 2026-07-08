from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

from rtm_acoustic.diagnostics.matched_budget import scale_to_budget
from rtm_acoustic.diagnostics.update_reliability import build_soft_gate


@dataclass(frozen=True)
class GateAblationConfig:
    coverage: float = 0.3635
    alpha_max: float = 0.3
    sigma_x: float = 4.0
    sigma_z: float = 4.0
    random_seeds: tuple[int, ...] = (0, 1, 2, 3, 4)


def _top_support(score: np.ndarray, coverage: float) -> np.ndarray:
    if coverage <= 0.0 or coverage > 1.0:
        raise ValueError("coverage must be in (0, 1]")
    values = np.asarray(score, dtype=float)
    flat = values.ravel()
    count = max(1, int(np.ceil(float(coverage) * flat.size)))
    selected = np.argpartition(flat, flat.size - count)[flat.size - count :]
    support = np.zeros(flat.size, dtype=bool)
    support[selected] = True
    return support.reshape(values.shape)


def _smooth_support(support: np.ndarray, sigma_z: float, sigma_x: float) -> np.ndarray:
    smoothed = gaussian_filter(support.astype(float), sigma=(float(sigma_z), float(sigma_x)))
    if np.max(smoothed) <= 0.0:
        raise ValueError("support is empty")
    return smoothed / float(np.max(smoothed))


def depth_matched_random_support(reference_support: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    ref = np.asarray(reference_support, dtype=bool)
    out = np.zeros_like(ref, dtype=bool)
    for z in range(ref.shape[0]):
        count = int(np.sum(ref[z]))
        if count <= 0:
            continue
        indices = rng.choice(ref.shape[1], size=min(count, ref.shape[1]), replace=False)
        out[z, indices] = True
    return out


def depth_score(shape: tuple[int, int]) -> np.ndarray:
    nz, nx = shape
    z = np.linspace(1.0, 0.0, nz, dtype=np.float32)[:, None]
    return np.repeat(z, nx, axis=1)


def build_matched_gate_suite(
    *,
    delta_model: np.ndarray,
    illumination: np.ndarray,
    consensus: np.ndarray,
    reliability: np.ndarray,
    target_update_l2: float,
    config: GateAblationConfig = GateAblationConfig(),
) -> dict[str, np.ndarray]:
    gates: dict[str, np.ndarray] = {}
    gates["global_matched"] = scale_to_budget(
        np.ones_like(delta_model, dtype=float),
        delta_model,
        target_update_l2=target_update_l2,
        alpha_max=config.alpha_max,
    )
    gates["illumination_only_matched"] = build_soft_gate(
        illumination,
        coverage=config.coverage,
        sigma_x=config.sigma_x,
        sigma_z=config.sigma_z,
        alpha_max=config.alpha_max,
        target_update_l2=target_update_l2,
        delta_model=delta_model,
    )
    gates["gradient_consensus_only_matched"] = build_soft_gate(
        consensus,
        coverage=config.coverage,
        sigma_x=config.sigma_x,
        sigma_z=config.sigma_z,
        alpha_max=config.alpha_max,
        target_update_l2=target_update_l2,
        delta_model=delta_model,
    )
    gates["ecg_reliability_gate"] = build_soft_gate(
        reliability,
        coverage=config.coverage,
        sigma_x=config.sigma_x,
        sigma_z=config.sigma_z,
        alpha_max=config.alpha_max,
        target_update_l2=target_update_l2,
        delta_model=delta_model,
    )
    inverse_illumination = 1.0 - np.asarray(illumination, dtype=float)
    gates["inverse_illumination_negative_control"] = build_soft_gate(
        inverse_illumination,
        coverage=config.coverage,
        sigma_x=config.sigma_x,
        sigma_z=config.sigma_z,
        alpha_max=config.alpha_max,
        target_update_l2=target_update_l2,
        delta_model=delta_model,
    )
    depth_gate = _smooth_support(_top_support(depth_score(delta_model.shape), config.coverage), config.sigma_z, config.sigma_x)
    gates["depth_matched"] = scale_to_budget(
        depth_gate,
        delta_model,
        target_update_l2=target_update_l2,
        alpha_max=config.alpha_max,
    )
    reference_support = _top_support(reliability, config.coverage)
    for seed in config.random_seeds:
        support = depth_matched_random_support(reference_support, seed)
        raw = _smooth_support(support, config.sigma_z, config.sigma_x)
        gates[f"random_matched_seed_{seed}"] = scale_to_budget(
            raw,
            delta_model,
            target_update_l2=target_update_l2,
            alpha_max=config.alpha_max,
        )
    return gates
