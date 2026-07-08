from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


def _load_matplotlib():
    import os

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError as exc:
        raise RuntimeError("需要安装 matplotlib 才能绘图: python -m pip install matplotlib") from exc


def _read_eval_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _parse_protocol_summary(path: Path) -> list[dict[str, str]]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    header = "| experiment | source_families | target_family | model | final_val_mae | final_val_rmse | test_mae | test_rmse | trainable_ratio | peft_type | injected_lora_modules |"
    if header not in lines:
        return []
    start = lines.index(header) + 2
    rows: list[dict[str, str]] = []
    for line in lines[start:]:
        if not line.startswith("|"):
            break
        parts = [part.strip() for part in line.strip("|").split("|")]
        if len(parts) != 11:
            continue
        rows.append(
            {
                "experiment": parts[0],
                "source_families": parts[1],
                "target_family": parts[2],
                "model": parts[3],
                "final_val_mae": parts[4],
                "final_val_rmse": parts[5],
                "test_mae": parts[6],
                "test_rmse": parts[7],
                "trainable_ratio": parts[8],
                "peft_type": parts[9],
                "injected_lora_modules": parts[10],
            }
        )
    return rows


def _write_bar_chart(path: Path, labels: list[str], values: list[float], *, title: str, ylabel: str) -> None:
    plt = _load_matplotlib()
    fig, ax = plt.subplots(figsize=(max(8.0, 1.2 * len(labels)), 4.6), constrained_layout=True)
    ax.bar(range(len(labels)), values)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_protocol_comparison(*, summary_path: str | Path, eval_csv: str | Path, output_dir: str | Path) -> dict[str, str]:
    summary_rows = _parse_protocol_summary(Path(summary_path))
    eval_rows = _read_eval_csv(Path(eval_csv))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    complete_rows = [row for row in eval_rows if row.get("status") == "complete"]
    labels = [f"{row['experiment']}:{row['model_name']}" for row in complete_rows]
    maes = [float(row["mae"]) for row in complete_rows]
    rmses = [float(row["rmse"]) for row in complete_rows]
    bar_test_mae = output_dir / "bar_test_mae_by_experiment.png"
    bar_test_rmse = output_dir / "bar_test_rmse_by_experiment.png"
    _write_bar_chart(bar_test_mae, labels, maes, title="Protocol v1 subset500 CPU small-scale test MAE", ylabel="test_mae")
    _write_bar_chart(bar_test_rmse, labels, rmses, title="Protocol v1 subset500 CPU small-scale test RMSE", ylabel="test_rmse")

    grouped: dict[tuple[str, str], float] = {}
    trainable_rows: list[dict[str, str]] = []
    for row in summary_rows:
        grouped[(row["experiment"], row["model"])] = float(row["test_mae"])
        trainable_rows.append({"experiment": row["experiment"], "model": row["model"], "trainable_ratio": row["trainable_ratio"]})
    delta_labels = []
    delta_mae_values = []
    delta_rmse_values = []
    for row in summary_rows:
        if row["model"] == "torch_cnn_baseline":
            continue
        cnn_key = (row["experiment"], "torch_cnn_baseline")
        if cnn_key not in grouped:
            continue
        delta_labels.append(f"{row['experiment']}:{row['model']}")
        delta_mae_values.append(float(row["test_mae"]) - grouped[cnn_key])
        delta_rmse_values.append(float(row["test_rmse"]) - next(float(item["test_rmse"]) for item in summary_rows if item["experiment"] == row["experiment"] and item["model"] == "torch_cnn_baseline"))
    delta_vs_cnn_mae = output_dir / "delta_vs_cnn_mae.png"
    delta_vs_cnn_rmse = output_dir / "delta_vs_cnn_rmse.png"
    _write_bar_chart(delta_vs_cnn_mae, delta_labels, delta_mae_values, title="Protocol v1 subset500 CPU small-scale delta vs CNN MAE", ylabel="delta_test_mae")
    _write_bar_chart(delta_vs_cnn_rmse, delta_labels, delta_rmse_values, title="Protocol v1 subset500 CPU small-scale delta vs CNN RMSE", ylabel="delta_test_rmse")

    in_domain_labels = []
    in_domain_mae_values = []
    in_domain_rmse_values = []
    for row in summary_rows:
        if row["model"] != "torch_cnn_baseline":
            continue
        label = row["experiment"]
        in_domain_labels.append(label)
        in_domain_mae_values.append(float(row["test_mae"]))
        in_domain_rmse_values.append(float(row["test_rmse"]))
    indomain_vs_crossfamily_mae = output_dir / "indomain_vs_crossfamily_mae.png"
    indomain_vs_crossfamily_rmse = output_dir / "indomain_vs_crossfamily_rmse.png"
    _write_bar_chart(indomain_vs_crossfamily_mae, in_domain_labels, in_domain_mae_values, title="Protocol v1 subset500 CPU small-scale in-domain vs cross-family MAE", ylabel="test_mae")
    _write_bar_chart(indomain_vs_crossfamily_rmse, in_domain_labels, in_domain_rmse_values, title="Protocol v1 subset500 CPU small-scale in-domain vs cross-family RMSE", ylabel="test_rmse")

    trainable_ratio_csv = output_dir / "trainable_ratio.csv"
    with trainable_ratio_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["experiment", "model", "trainable_ratio"])
        writer.writeheader()
        writer.writerows(trainable_rows)

    ranking_md = output_dir / "protocol_v1_model_ranking.md"
    ranking_rows = sorted(complete_rows, key=lambda row: float(row["mae"]))
    lines = [
        "# Protocol v1 Model Ranking",
        "",
        "dummy_dinov2 is not real pretrained DINOv2。",
        "",
        "| rank | experiment | model_name | test_mae | test_rmse |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for index, row in enumerate(ranking_rows, start=1):
        lines.append(f"| {index} | {row['experiment']} | {row['model_name']} | {row['mae']} | {row['rmse']} |")
    ranking_md.write_text("\n".join(lines), encoding="utf-8")
    return {
        "bar_test_mae": str(bar_test_mae),
        "bar_test_rmse": str(bar_test_rmse),
        "delta_vs_cnn_mae": str(delta_vs_cnn_mae),
        "delta_vs_cnn_rmse": str(delta_vs_cnn_rmse),
        "indomain_vs_crossfamily_mae": str(indomain_vs_crossfamily_mae),
        "indomain_vs_crossfamily_rmse": str(indomain_vs_crossfamily_rmse),
        "trainable_ratio_csv": str(trainable_ratio_csv),
        "ranking_md": str(ranking_md),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="根据 Protocol v1 summary 和 eval csv 生成模型对比图。")
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--eval-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = plot_protocol_comparison(summary_path=args.summary, eval_csv=args.eval_csv, output_dir=args.output_dir)
    print(f"写出图件目录: {args.output_dir}")
    print(f"ranking: {result['ranking_md']}")


if __name__ == "__main__":
    main()
