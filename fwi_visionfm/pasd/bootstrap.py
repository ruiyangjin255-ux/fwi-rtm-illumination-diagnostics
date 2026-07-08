"""Aligned paired-bootstrap inference for PASD prediction archives."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from .metrics import per_sample_metrics


MetricName = Literal["mae", "rmse", "ssim", "edge_mae", "gradient_error"]


def _load_archive(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        required = {"sample_id", "prediction", "target"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"Prediction archive {path} is missing keys: {sorted(missing)}")
        return {name: np.asarray(archive[name]) for name in required}


def aligned_metric_difference(
    baseline_archive: str | Path,
    candidate_archive: str | Path,
    metric: MetricName,
) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned sample IDs and candidate-minus-baseline metric difference."""

    base = _load_archive(baseline_archive)
    candidate = _load_archive(candidate_archive)
    base_id_to_index = {int(sample_id): index for index, sample_id in enumerate(base["sample_id"].tolist())}
    candidate_id_to_index = {int(sample_id): index for index, sample_id in enumerate(candidate["sample_id"].tolist())}
    shared_ids = np.asarray(sorted(set(base_id_to_index).intersection(candidate_id_to_index)), dtype=np.int64)
    if len(shared_ids) == 0:
        raise ValueError("The two archives have no aligned sample_id values.")
    if len(shared_ids) != len(base_id_to_index) or len(shared_ids) != len(candidate_id_to_index):
        raise ValueError("Paired bootstrap requires identical sample_id sets; archive alignment is incomplete.")
    base_indices = np.asarray([base_id_to_index[int(sample_id)] for sample_id in shared_ids])
    candidate_indices = np.asarray([candidate_id_to_index[int(sample_id)] for sample_id in shared_ids])

    base_metrics = per_sample_metrics(torch.from_numpy(base["prediction"][base_indices]), torch.from_numpy(base["target"][base_indices]))[metric]
    candidate_metrics = per_sample_metrics(
        torch.from_numpy(candidate["prediction"][candidate_indices]), torch.from_numpy(candidate["target"][candidate_indices])
    )[metric]
    return shared_ids, (candidate_metrics - base_metrics).cpu().numpy().astype(np.float64)


def paired_bootstrap(
    baseline_archive: str | Path,
    candidate_archive: str | Path,
    metric: MetricName,
    n_resamples: int = 2000,
    seed: int = 0,
    lower_is_better: bool | None = None,
) -> dict[str, object]:
    """Estimate mean paired difference and 95% CI without rerunning inference."""

    ids, difference = aligned_metric_difference(baseline_archive, candidate_archive, metric)
    if n_resamples < 100:
        raise ValueError("Use at least 100 bootstrap resamples for a meaningful uncertainty estimate.")
    if lower_is_better is None:
        lower_is_better = metric != "ssim"
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(difference), size=(n_resamples, len(difference)))
    boot = difference[indices].mean(axis=1)
    mean_difference = float(difference.mean())
    ci_low, ci_high = np.quantile(boot, [0.025, 0.975]).tolist()
    improvement = -difference if lower_is_better else difference
    return {
        "metric": metric,
        "n_samples": int(len(ids)),
        "n_resamples": int(n_resamples),
        "baseline_minus_candidate_interpretation": "positive improvement" if lower_is_better else "positive improvement",
        "candidate_minus_baseline_mean": mean_difference,
        "candidate_minus_baseline_ci95": [float(ci_low), float(ci_high)],
        "improvement_probability": float((improvement > 0.0).mean()),
        "sample_id_min": int(ids.min()),
        "sample_id_max": int(ids.max()),
    }


def save_paired_bootstrap(
    baseline_archive: str | Path,
    candidate_archive: str | Path,
    output: str | Path,
    metric: MetricName,
    n_resamples: int = 2000,
    seed: int = 0,
) -> dict[str, object]:
    result = paired_bootstrap(baseline_archive, candidate_archive, metric, n_resamples=n_resamples, seed=seed)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    return result
