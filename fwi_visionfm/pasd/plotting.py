"""Publication-oriented deterministic plotting utilities for PASD-FWI outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_bridge_attributes(attributes: np.ndarray, output: str | Path, shot_index: int = 0) -> None:
    """Render raw, envelope, and band-energy channels for a fixed shot index."""

    path = _ensure_parent(output)
    if attributes.ndim == 5:
        attributes = attributes[0]
    shot_index = min(shot_index, attributes.shape[0] - 1)
    names = ["Raw gather", "Hilbert envelope", "Band-energy attribute"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    for index, ax in enumerate(axes):
        image = attributes[shot_index, index]
        image_map = "seismic" if index == 0 else "viridis"
        im = ax.imshow(image, aspect="auto", cmap=image_map)
        ax.set_title(names[index])
        ax.set_xlabel("Receiver index")
        ax.set_ylabel("Time sample")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_bridge_attribute_files(attributes: np.ndarray, output_dir: str | Path, shot_index: int = 0) -> None:
    """Write the fixed per-channel bridge evidence files required by the Phase-1 audit."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if attributes.ndim == 5:
        attributes = attributes[0]
    shot_index = min(shot_index, attributes.shape[0] - 1)
    names = [("raw.png", "seismic"), ("envelope.png", "viridis"), ("band_energy.png", "magma")]
    for channel, (filename, cmap) in enumerate(names):
        image = attributes[shot_index, min(channel, attributes.shape[1] - 1)]
        fig, ax = plt.subplots(figsize=(4.8, 3.4), constrained_layout=True)
        im = ax.imshow(image, aspect="auto", cmap=cmap)
        ax.set_xlabel("Receiver index")
        ax.set_ylabel("Time sample")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.savefig(out / filename, dpi=220)
        plt.close(fig)
    geometry = np.tile(np.linspace(-1.0, 1.0, attributes.shape[-1], dtype=np.float32), (attributes.shape[-2], 1))
    fig, ax = plt.subplots(figsize=(4.8, 3.4), constrained_layout=True)
    im = ax.imshow(geometry, aspect="auto", cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_xlabel("Receiver index")
    ax.set_ylabel("Time sample")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(out / "geometry_map.png", dpi=220)
    plt.close(fig)
    plot_bridge_attributes(attributes, out / "bridge_panel.png", shot_index=shot_index)


def plot_velocity_comparison(
    truth: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    output: str | Path,
    title: str | None = None,
) -> None:
    """Render one truth row and a shared-error-scale prediction/error comparison row."""

    path = _ensure_parent(output)
    truth = np.asarray(truth).squeeze()
    ordered = list(predictions.items())
    values = [truth] + [np.asarray(value).squeeze() for _, value in ordered]
    vmin, vmax = min(float(v.min()) for v in values), max(float(v.max()) for v in values)
    errors = [np.abs(value - truth) for value in values[1:]]
    emax = max(float(error.max()) for error in errors) if errors else 1.0
    columns = len(values)
    fig, axes = plt.subplots(2, columns, figsize=(3.1 * columns, 6.0), constrained_layout=True, squeeze=False)
    im_velocity = axes[0, 0].imshow(truth, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
    axes[0, 0].set_title("Ground truth")
    axes[1, 0].axis("off")
    im_error = None
    for col, (name, prediction) in enumerate(ordered, start=1):
        prediction = np.asarray(prediction).squeeze()
        axes[0, col].imshow(prediction, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
        axes[0, col].set_title(name)
        im_error = axes[1, col].imshow(np.abs(prediction - truth), cmap="magma", vmin=0.0, vmax=emax, aspect="auto")
        axes[1, col].set_title(f"|Error|: {name}")
    for ax in axes.flat:
        if ax.axison:
            ax.set_xlabel("Distance index")
            ax.set_ylabel("Depth index")
    fig.colorbar(im_velocity, ax=axes[0, :].tolist(), shrink=0.72, label="Velocity")
    if im_error is not None:
        fig.colorbar(im_error, ax=axes[1, 1:].tolist(), shrink=0.72, label="Absolute error")
    if title:
        fig.suptitle(title)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_profiles(
    truth: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    output: str | Path,
    row: int | None = None,
    column: int | None = None,
) -> None:
    """Plot fixed horizontal/vertical profiles; caller should keep row/column constant across variants."""

    path = _ensure_parent(output)
    truth = np.asarray(truth).squeeze()
    row = truth.shape[0] // 2 if row is None else int(row)
    column = truth.shape[1] // 2 if column is None else int(column)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    axes[0].plot(truth[row], linewidth=2.2, label="Ground truth")
    axes[1].plot(truth[:, column], linewidth=2.2, label="Ground truth")
    for name, value in predictions.items():
        value = np.asarray(value).squeeze()
        axes[0].plot(value[row], label=name)
        axes[1].plot(value[:, column], label=name)
    axes[0].set_title(f"Horizontal profile, depth index={row}")
    axes[1].set_title(f"Vertical profile, distance index={column}")
    for ax in axes:
        ax.set_xlabel("Index")
        ax.set_ylabel("Velocity")
        ax.legend(frameon=False)
        ax.grid(alpha=0.25)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _gradient(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array).squeeze()
    gz, gx = np.gradient(array)
    return np.sqrt(gx**2 + gz**2)


def plot_gradient_comparison(truth: np.ndarray, predictions: Mapping[str, np.ndarray], output: str | Path) -> None:
    """Show structural evidence using common gradient and gradient-error scales."""

    path = _ensure_parent(output)
    truth = np.asarray(truth).squeeze()
    truth_gradient = _gradient(truth)
    ordered = [(name, np.asarray(value).squeeze()) for name, value in predictions.items()]
    gradients = [truth_gradient] + [_gradient(value) for _, value in ordered]
    gmax = max(float(item.max()) for item in gradients)
    errors = [np.abs(_gradient(value) - truth_gradient) for _, value in ordered]
    emax = max((float(item.max()) for item in errors), default=1.0)
    columns = len(ordered) + 1
    fig, axes = plt.subplots(2, columns, figsize=(3.1 * columns, 6.0), constrained_layout=True, squeeze=False)
    truth_im = axes[0, 0].imshow(truth_gradient, cmap="viridis", vmin=0.0, vmax=gmax, aspect="auto")
    axes[0, 0].set_title("True gradient")
    axes[1, 0].axis("off")
    error_im = None
    for column, (name, prediction) in enumerate(ordered, start=1):
        axes[0, column].imshow(_gradient(prediction), cmap="viridis", vmin=0.0, vmax=gmax, aspect="auto")
        axes[0, column].set_title(f"Gradient: {name}")
        error_im = axes[1, column].imshow(np.abs(_gradient(prediction) - truth_gradient), cmap="magma", vmin=0.0, vmax=emax, aspect="auto")
        axes[1, column].set_title(f"Gradient error: {name}")
    for ax in axes.flat:
        if ax.axison:
            ax.set_xlabel("Distance index")
            ax.set_ylabel("Depth index")
    fig.colorbar(truth_im, ax=axes[0, :].tolist(), shrink=0.72, label="Gradient magnitude")
    if error_im is not None:
        fig.colorbar(error_im, ax=axes[1, 1:].tolist(), shrink=0.72, label="Gradient absolute error")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_training_history(history: Sequence[Mapping[str, float]], output: str | Path, title: str | None = None) -> None:
    """Render loss curves from standardized history rows."""

    if not history:
        return
    path = _ensure_parent(output)
    epochs = [row["epoch"] for row in history]
    fig, ax = plt.subplots(figsize=(6.5, 4.0), constrained_layout=True)
    if "train_loss" in history[0]:
        ax.plot(epochs, [row["train_loss"] for row in history], marker="o", label="Train loss")
    if "val_loss" in history[0]:
        ax.plot(epochs, [row["val_loss"] for row in history], marker="o", label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    if title:
        ax.set_title(title)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_protocol_metric_summary(
    summary_rows: Sequence[Mapping[str, object]],
    output: str | Path,
    metric: str,
    split: str,
) -> None:
    """Plot mean ± standard deviation for one metric across protocol variants."""

    selected = [row for row in summary_rows if row.get("split") == split and row.get("metric") == metric]
    if not selected:
        return
    path = _ensure_parent(output)
    labels = [str(row["variant"]) for row in selected]
    means = [float(row["mean"]) for row in selected]
    stds = [float(row["std"]) for row in selected]
    fig, ax = plt.subplots(figsize=(max(6.0, len(labels) * 1.6), 4.2), constrained_layout=True)
    positions = np.arange(len(labels))
    ax.bar(positions, means, yerr=stds, capsize=4)
    ax.set_xticks(positions, labels, rotation=20, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(f"{split}: {metric} across seeds")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(path, dpi=220)
    plt.close(fig)
