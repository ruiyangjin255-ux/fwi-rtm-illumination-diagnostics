from __future__ import annotations

import numpy as np


ERROR_METRICS = ("normalized_l2_residual", "nrms_residual", "envelope_error", "phase_error")
CORRELATION_METRICS = ("trace_correlation",)


def _finite_pair(predicted: np.ndarray, observed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(predicted, dtype=np.float64)
    obs = np.asarray(observed, dtype=np.float64)
    if pred.shape != obs.shape:
        raise ValueError(f"record shapes differ: {pred.shape} vs {obs.shape}")
    if pred.ndim != 2:
        raise ValueError("records must be 2-D arrays with shape (nt, n_receivers)")
    if not np.isfinite(pred).all() or not np.isfinite(obs).all():
        raise ValueError("records contain NaN or Inf")
    return pred, obs


def normalized_l2_residual(predicted: np.ndarray, observed: np.ndarray, eps: float = 1.0e-8) -> float:
    pred, obs = _finite_pair(predicted, observed)
    residual = pred - obs
    return float(np.linalg.norm(residual) / (np.linalg.norm(obs) + eps))


def nrms_residual(predicted: np.ndarray, observed: np.ndarray, eps: float = 1.0e-8) -> float:
    pred, obs = _finite_pair(predicted, observed)
    residual = pred - obs
    denom = np.sqrt(np.mean(pred**2)) + np.sqrt(np.mean(obs**2)) + eps
    return float(2.0 * np.sqrt(np.mean(residual**2)) / denom)


def trace_correlation(predicted: np.ndarray, observed: np.ndarray, eps: float = 1.0e-8) -> float:
    pred, obs = _finite_pair(predicted, observed)
    pred = pred - np.mean(pred, axis=0, keepdims=True)
    obs = obs - np.mean(obs, axis=0, keepdims=True)
    numer = np.sum(pred * obs, axis=0)
    denom = np.linalg.norm(pred, axis=0) * np.linalg.norm(obs, axis=0)
    valid = denom > eps
    if not np.any(valid):
        return float("nan")
    return float(np.mean(numer[valid] / (denom[valid] + eps)))


def _analytic_signal(record: np.ndarray) -> np.ndarray:
    n = record.shape[0]
    spectrum = np.fft.fft(record, axis=0)
    multiplier = np.zeros(n, dtype=np.float64)
    if n % 2 == 0:
        multiplier[0] = 1.0
        multiplier[n // 2] = 1.0
        multiplier[1 : n // 2] = 2.0
    else:
        multiplier[0] = 1.0
        multiplier[1 : (n + 1) // 2] = 2.0
    return np.fft.ifft(spectrum * multiplier[:, None], axis=0)


def envelope_error(predicted: np.ndarray, observed: np.ndarray, eps: float = 1.0e-8) -> float:
    pred, obs = _finite_pair(predicted, observed)
    return normalized_l2_residual(np.abs(_analytic_signal(pred)), np.abs(_analytic_signal(obs)), eps=eps)


def phase_error(predicted: np.ndarray, observed: np.ndarray, eps: float = 1.0e-8) -> float:
    pred, obs = _finite_pair(predicted, observed)
    pred_analytic = _analytic_signal(pred)
    obs_analytic = _analytic_signal(obs)
    amp = np.abs(obs_analytic)
    threshold = max(float(np.percentile(amp, 60.0)), eps)
    valid = amp >= threshold
    if not np.any(valid):
        return float("nan")
    phase_delta = np.angle(pred_analytic[valid] * np.conj(obs_analytic[valid]))
    return float(np.sqrt(np.mean(phase_delta**2)))


def audit_record_pair(predicted: np.ndarray, observed: np.ndarray) -> dict[str, float]:
    return {
        "normalized_l2_residual": normalized_l2_residual(predicted, observed),
        "nrms_residual": nrms_residual(predicted, observed),
        "trace_correlation": trace_correlation(predicted, observed),
        "envelope_error": envelope_error(predicted, observed),
        "phase_error": phase_error(predicted, observed),
    }


def summarize_metric_rows(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    methods = sorted({str(row["method"]) for row in rows})
    summaries: list[dict[str, float | int | str]] = []
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        item: dict[str, float | int | str] = {"method": method, "shot_count": len(method_rows)}
        for metric in (*ERROR_METRICS, *CORRELATION_METRICS):
            values = np.asarray([float(row[metric]) for row in method_rows], dtype=np.float64)
            item[f"{metric}_mean"] = float(np.mean(values))
            item[f"{metric}_std"] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        summaries.append(item)
    return summaries


def paired_bootstrap(
    rows: list[dict[str, float | int | str]],
    *,
    reference_method: str,
    comparator_methods: list[str],
    metrics: tuple[str, ...] = (*ERROR_METRICS, *CORRELATION_METRICS),
    samples: int = 2000,
    seed: int = 20260707,
) -> dict[str, dict[str, dict[str, float | int | str]]]:
    by_method = {method: {int(row["shot_index"]): row for row in rows if row["method"] == method} for method in {str(row["method"]) for row in rows}}
    rng = np.random.default_rng(seed)
    output: dict[str, dict[str, dict[str, float | int | str]]] = {}
    for comparator in comparator_methods:
        common = sorted(set(by_method.get(reference_method, {})).intersection(by_method.get(comparator, {})))
        if not common:
            raise ValueError(f"no common shots for {reference_method} vs {comparator}")
        output[comparator] = {}
        draw_index = rng.integers(0, len(common), size=(samples, len(common)))
        for metric in metrics:
            ref_values = np.asarray([float(by_method[reference_method][shot][metric]) for shot in common], dtype=np.float64)
            cmp_values = np.asarray([float(by_method[comparator][shot][metric]) for shot in common], dtype=np.float64)
            diffs = ref_values - cmp_values
            boot = np.mean(diffs[draw_index], axis=1)
            better = boot > 0.0 if metric in CORRELATION_METRICS else boot < 0.0
            output[comparator][metric] = {
                "n_shots": len(common),
                "mean_delta": float(np.mean(diffs)),
                "ci95_low": float(np.percentile(boot, 2.5)),
                "ci95_high": float(np.percentile(boot, 97.5)),
                "probability_ecg_better": float(np.mean(better)),
                "direction": "higher_is_better" if metric in CORRELATION_METRICS else "lower_is_better",
            }
    return output
