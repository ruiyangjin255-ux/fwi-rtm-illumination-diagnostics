"""Small plotting helpers for PASD Phase-1b diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_both(fig: plt.Figure, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def barplot(rows: Sequence[Mapping[str, object]], *, x: str, y: str, group: str | None, output: str | Path, title: str) -> None:
    labels = sorted({str(row[x]) for row in rows})
    groups = [""] if group is None else sorted({str(row[group]) for row in rows})
    fig, ax = plt.subplots(figsize=(max(7.0, len(labels) * 1.3), 4.2), constrained_layout=True)
    width = 0.8 / max(1, len(groups))
    positions = np.arange(len(labels))
    for gi, g in enumerate(groups):
        vals = []
        for label in labels:
            selected = [float(row[y]) for row in rows if str(row[x]) == label and (group is None or str(row[group]) == g)]
            vals.append(float(np.mean(selected)) if selected else np.nan)
        ax.bar(positions - 0.4 + width / 2 + gi * width, vals, width=width, label=g if g else None)
    ax.set_xticks(positions, labels, rotation=20, ha="right")
    ax.set_ylabel(y)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    if group is not None:
        ax.legend(frameon=False)
    save_both(fig, output)


def boxplot(groups: Mapping[str, Sequence[float]], output: str | Path, title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(max(7.0, len(groups) * 1.2), 4.2), constrained_layout=True)
    labels = list(groups)
    ax.boxplot([list(groups[label]) for label in labels], labels=labels, showfliers=False)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    save_both(fig, output)
