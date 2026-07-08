from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any

from fwi_visionfm.scripts.summarize_local_mae_ablation import write_local_mae_ablation_summary


METRICS = ["MAE", "RMSE", "SSIM", "gradient_error", "edge_MAE"]
LOWER_IS_BETTER = {"MAE", "RMSE", "gradient_error", "edge_MAE"}


def _load_matplotlib():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("需要安装 matplotlib 才能生成 Protocol V5 seed stability 图") from exc
    return plt


def _f(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "") or 0.0)
    except ValueError:
        return 0.0


def _read_summary(root: Path) -> list[dict[str, str]]:
    summary_path = root / "local_mae_ablation_summary.csv"
    if not summary_path.exists():
        summary_path = write_local_mae_ablation_summary(root)
    with summary_path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _trace_dropout_pairs(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    success = [row for row in rows if row.get("status") == "SUCCESS"]
    pretrained = {
        row["seed"]: row
        for row in success
        if row.get("model_type") == "pretrained_local_mae"
        and row.get("bridge") == "raw_envelope_spectrum3"
        and row.get("mask_type") == "trace_dropout"
        and row.get("loss_name") == "default_l1"
    }
    random_rows = {
        row["seed"]: row
        for row in success
        if row.get("model_type") == "random_mae_encoder"
        and row.get("bridge") == "raw_envelope_spectrum3"
        and row.get("mask_type") == "trace_dropout"
        and row.get("loss_name") == "default_l1"
    }
    seeds = sorted(set(pretrained) & set(random_rows), key=int)
    paired: list[dict[str, Any]] = []
    for seed in seeds:
        pre = pretrained[seed]
        rand = random_rows[seed]
        row: dict[str, Any] = {"seed": seed}
        for metric in METRICS:
            row[f"pretrained_{metric}"] = _f(pre, metric)
            row[f"random_{metric}"] = _f(rand, metric)
            if metric in LOWER_IS_BETTER:
                row[f"pretrained_win_{metric}"] = int(row[f"pretrained_{metric}"] < row[f"random_{metric}"])
            else:
                row[f"pretrained_win_{metric}"] = int(row[f"pretrained_{metric}"] > row[f"random_{metric}"])
        paired.append(row)
    return paired


def _write_seed_stability_csv(output_dir: Path, paired_rows: list[dict[str, Any]]) -> Path:
    path = output_dir / "trace_dropout_seed_stability_metrics.csv"
    fields = ["seed"]
    for metric in METRICS:
        fields.extend(
            [
                f"pretrained_{metric}",
                f"random_{metric}",
                f"pretrained_win_{metric}",
            ]
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(paired_rows)
        writer.writerow({})
        writer.writerow({"seed": "metric", "pretrained_MAE": "win_count"})
        for metric in METRICS:
            writer.writerow({"seed": metric, "pretrained_MAE": str(sum(int(row[f"pretrained_win_{metric}"]) for row in paired_rows))})
    return path


def _plot_pretrained_vs_random(output_dir: Path, paired_rows: list[dict[str, Any]]) -> Path:
    plt = _load_matplotlib()
    fig, axes = plt.subplots(len(METRICS), 1, figsize=(10, 3.2 * len(METRICS)), constrained_layout=True)
    if len(METRICS) == 1:
        axes = [axes]
    seeds = [str(row["seed"]) for row in paired_rows]
    x = list(range(len(seeds)))
    width = 0.36
    for ax, metric in zip(axes, METRICS):
        pre_vals = [row[f"pretrained_{metric}"] for row in paired_rows]
        rand_vals = [row[f"random_{metric}"] for row in paired_rows]
        ax.bar([i - width / 2 for i in x], pre_vals, width=width, label="pretrained")
        ax.bar([i + width / 2 for i in x], rand_vals, width=width, label="random")
        ax.set_title(metric)
        ax.set_xticks(x, seeds)
        ax.set_xlabel("seed")
        ax.set_ylabel(metric)
        ax.legend()
    path = output_dir / "trace_dropout_pretrained_vs_random_bar.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_win_counts(output_dir: Path, paired_rows: list[dict[str, Any]]) -> Path:
    plt = _load_matplotlib()
    metrics = METRICS
    counts = [sum(int(row[f"pretrained_win_{metric}"]) for row in paired_rows) for metric in metrics]
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    ax.bar(metrics, counts)
    ax.set_ylim(0, max(len(paired_rows), 1))
    ax.set_ylabel("pretrained win count")
    ax.set_title("Trace-dropout pretrained wins out of available seeds")
    path = output_dir / "trace_dropout_win_count_bar.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_protocol_v5_seed_stability_artifacts(root: str | Path, output_dir: str | Path) -> dict[str, Path]:
    root_path = Path(root)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paired_rows = _trace_dropout_pairs(_read_summary(root_path))
    csv_path = _write_seed_stability_csv(output_path, paired_rows)
    comparison_png = _plot_pretrained_vs_random(output_path, paired_rows)
    win_count_png = _plot_win_counts(output_path, paired_rows)
    return {
        "csv_path": csv_path,
        "comparison_png": comparison_png,
        "win_count_png": win_count_png,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Protocol V5 trace-dropout seed stability.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    result = write_protocol_v5_seed_stability_artifacts(**vars(parse_args()))
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
