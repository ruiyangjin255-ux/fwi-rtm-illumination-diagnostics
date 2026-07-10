from __future__ import annotations

import numpy as np
from scipy.ndimage import sobel


def image_correlation(a: np.ndarray, b: np.ndarray, eps: float = 1.0e-8) -> float:
    aa = np.asarray(a, dtype=float).ravel()
    bb = np.asarray(b, dtype=float).ravel()
    aa = aa - float(np.mean(aa))
    bb = bb - float(np.mean(bb))
    return float(np.dot(aa, bb) / ((np.linalg.norm(aa) * np.linalg.norm(bb)) + eps))


def simple_ssim(a: np.ndarray, b: np.ndarray, eps: float = 1.0e-8) -> float:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    mu_a = float(np.mean(aa))
    mu_b = float(np.mean(bb))
    var_a = float(np.var(aa))
    var_b = float(np.var(bb))
    cov = float(np.mean((aa - mu_a) * (bb - mu_b)))
    c1 = 0.01**2
    c2 = 0.03**2
    return float(((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2) + eps))


def local_structure_tensor_coherence(image: np.ndarray, eps: float = 1.0e-8) -> float:
    arr = np.asarray(image, dtype=float)
    gx = sobel(arr, axis=1)
    gz = sobel(arr, axis=0)
    jxx = np.mean(gx * gx)
    jzz = np.mean(gz * gz)
    jxz = np.mean(gx * gz)
    trace = jxx + jzz
    det_term = np.sqrt((jxx - jzz) ** 2 + 4.0 * jxz * jxz)
    return float(det_term / (trace + eps))


def split_consistency(a: np.ndarray, b: np.ndarray, laplacian_a: np.ndarray | None = None, laplacian_b: np.ndarray | None = None) -> dict[str, float | None]:
    return {
        "rtm_split_correlation": image_correlation(a, b),
        "rtm_split_ssim": simple_ssim(a, b),
        "local_structure_tensor_coherence": local_structure_tensor_coherence(0.5 * (np.asarray(a) + np.asarray(b))),
        "laplacian_rtm_split_correlation": image_correlation(laplacian_a, laplacian_b) if laplacian_a is not None and laplacian_b is not None else None,
    }

