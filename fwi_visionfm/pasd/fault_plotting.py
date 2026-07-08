"""Phase-2 FlatFault-A figure generation."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .diagnostics import gradient_magnitude_np


def save_both(fig: plt.Figure, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def comparison_figure(target: np.ndarray, predictions: dict[str, np.ndarray], metrics: dict[str, dict[str, float]], output: str | Path, title: str) -> None:
    names = list(predictions)
    vals = [target] + [predictions[name] for name in names]
    vmin, vmax = min(float(v.min()) for v in vals), max(float(v.max()) for v in vals)
    emax = max(float(np.abs(predictions[name] - target).max()) for name in names)
    fig, axes = plt.subplots(2, len(names) + 1, figsize=(3.2 * (len(names) + 1), 6.4), constrained_layout=True)
    axes[0, 0].imshow(target, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
    axes[0, 0].set_title("Ground Truth")
    axes[1, 0].axis("off")
    for col, name in enumerate(names, start=1):
        m = metrics.get(name, {})
        axes[0, col].imshow(predictions[name], cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
        axes[0, col].set_title(f"{name}\nMAE={m.get('MAE', 0):.1f}, SSIM={m.get('SSIM', 0):.3f}")
        axes[1, col].imshow(np.abs(predictions[name] - target), cmap="magma", vmin=0, vmax=emax, aspect="auto")
        axes[1, col].set_title(f"|error|\nedge={m.get('edge_MAE', 0):.1f}, g_edge={m.get('gradient_l1_edge', 0):.1f}")
    fig.suptitle(title)
    save_both(fig, output)


def gradient_figure(target: np.ndarray, predictions: dict[str, np.ndarray], edge_masks: dict[str, np.ndarray], output: str | Path, title: str) -> None:
    names = list(predictions)
    grads = {"truth": gradient_magnitude_np(target), **{name: gradient_magnitude_np(predictions[name]) for name in names}}
    gmax = max(float(g.max()) for g in grads.values())
    fig, axes = plt.subplots(3, len(names) + 1, figsize=(3.2 * (len(names) + 1), 9.2), constrained_layout=True)
    axes[0, 0].imshow(grads["truth"], cmap="viridis", vmin=0, vmax=gmax, aspect="auto")
    axes[0, 0].set_title("true gradient magnitude")
    axes[1, 0].imshow(edge_masks["truth"], cmap="gray", aspect="auto")
    axes[1, 0].set_title("true edge mask")
    axes[2, 0].axis("off")
    for col, name in enumerate(names, start=1):
        axes[0, col].imshow(grads[name], cmap="viridis", vmin=0, vmax=gmax, aspect="auto")
        axes[0, col].set_title(f"{name} gradient")
        axes[1, col].imshow(edge_masks[name], cmap="gray", aspect="auto")
        axes[1, col].set_title(f"{name} predicted edge")
        axes[2, col].imshow(np.abs(grads[name] - grads["truth"]), cmap="magma", aspect="auto")
        axes[2, col].set_title(f"{name} gradient error")
    fig.suptitle(title)
    save_both(fig, output)


def profiles_figure(target: np.ndarray, predictions: dict[str, np.ndarray], output: str | Path, title: str) -> None:
    g = gradient_magnitude_np(target)
    row, col = np.unravel_index(int(np.argmax(g)), g.shape)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for arr, label in [(target, "Ground truth"), *[(predictions[name], name) for name in predictions]]:
        axes[0].plot(arr[row], label=label)
        axes[1].plot(arr[:, col], label=label)
    axes[0].set_title(f"Horizontal profile depth={row}")
    axes[1].set_title(f"Vertical profile distance={col}")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(title)
    save_both(fig, output)
