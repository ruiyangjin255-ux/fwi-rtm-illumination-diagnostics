from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from fwi_visionfm.scripts.run_protocol_v7_boundary_auxiliary_seed_stability import compute_seed_stability_win_counts


SUMMARY_FIELDS = [
    "seed",
    "baseline_MAE",
    "boundary_MAE",
    "baseline_RMSE",
    "boundary_RMSE",
    "baseline_SSIM",
    "boundary_SSIM",
    "baseline_gradient_error",
    "boundary_gradient_error",
    "baseline_edge_MAE",
    "boundary_edge_MAE",
    "baseline_status",
    "boundary_status",
]


def _load_matplotlib():
    import os

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _to_float(value: str) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _read_rows(root: Path) -> list[dict[str, str]]:
    summary_path = root / "protocol_v7_boundary_auxiliary_seed_stability_summary.csv"
    with summary_path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _pair_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["seed"], {})[row["model_type"]] = row
    paired = []
    for seed in sorted(grouped.keys(), key=int):
        baseline = grouped[seed].get("baseline", {})
        boundary = grouped[seed].get("boundary_aux", {})
        paired.append(
            {
                "seed": seed,
                "baseline_MAE": baseline.get("cross_family_MAE", ""),
                "boundary_MAE": boundary.get("cross_family_MAE", ""),
                "baseline_RMSE": baseline.get("cross_family_RMSE", ""),
                "boundary_RMSE": boundary.get("cross_family_RMSE", ""),
                "baseline_SSIM": baseline.get("cross_family_SSIM", ""),
                "boundary_SSIM": boundary.get("cross_family_SSIM", ""),
                "baseline_gradient_error": baseline.get("cross_family_gradient_error", ""),
                "boundary_gradient_error": boundary.get("cross_family_gradient_error", ""),
                "baseline_edge_MAE": baseline.get("cross_family_edge_MAE", ""),
                "boundary_edge_MAE": boundary.get("cross_family_edge_MAE", ""),
                "baseline_status": baseline.get("status", "SKIPPED"),
                "boundary_status": boundary.get("status", "SKIPPED"),
            }
        )
    return paired


def _write_summary_csv(path: Path, paired_rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(paired_rows)
    return path


def _draw_metrics_bar(path: Path, paired_rows: list[dict[str, Any]]) -> Path:
    plt = _load_matplotlib()
    seeds = [row["seed"] for row in paired_rows]
    metrics = [
        ("MAE", "baseline_MAE", "boundary_MAE"),
        ("RMSE", "baseline_RMSE", "boundary_RMSE"),
        ("SSIM", "baseline_SSIM", "boundary_SSIM"),
        ("gradient_error", "baseline_gradient_error", "boundary_gradient_error"),
        ("edge_MAE", "baseline_edge_MAE", "boundary_edge_MAE"),
    ]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 14), constrained_layout=True)
    for ax, (label, baseline_key, boundary_key) in zip(axes, metrics):
        baseline_vals = [_to_float(row[baseline_key]) or 0.0 for row in paired_rows]
        boundary_vals = [_to_float(row[boundary_key]) or 0.0 for row in paired_rows]
        x = list(range(len(seeds)))
        ax.bar([i - 0.18 for i in x], baseline_vals, width=0.36, label="baseline", color="#4c78a8")
        ax.bar([i + 0.18 for i in x], boundary_vals, width=0.36, label="boundary_aux", color="#f58518")
        ax.set_xticks(x)
        ax.set_xticklabels([f"seed {seed}" for seed in seeds])
        ax.set_title(label)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(loc="upper right")
    fig.suptitle("Boundary Auxiliary vs Baseline Metrics", fontsize=13)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def _draw_win_count_bar(path: Path, wins: dict[str, int]) -> Path:
    plt = _load_matplotlib()
    labels = ["MAE", "RMSE", "SSIM", "gradient_error", "edge_MAE"]
    values = [
        wins["MAE_lower"],
        wins["RMSE_lower"],
        wins["SSIM_higher"],
        wins["gradient_error_lower"],
        wins["edge_MAE_lower"],
    ]
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    bars = ax.bar(labels, values, color=["#4c78a8", "#4c78a8", "#72b7b2", "#54a24b", "#54a24b"])
    ax.set_ylim(0, 3.4)
    ax.set_ylabel("boundary_aux win count")
    ax.set_title("Boundary Auxiliary Win Counts Across Seeds")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, value + 0.05, f"{value}/3", ha="center", va="bottom")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def _tile_images(cells: list[list[tuple[str, Path | None]]], output_path: Path, *, title: str) -> Path:
    font_fill = (0, 0, 0)
    label_height = 26
    padding = 8
    rendered_rows: list[Image.Image] = []
    for row in cells:
        rendered_cells: list[Image.Image] = []
        max_height = 0
        for label, path in row:
            if path is not None and path.exists():
                image = Image.open(path).convert("RGB")
            else:
                image = Image.new("RGB", (320, 180), (245, 245, 245))
                draw = ImageDraw.Draw(image)
                draw.text((12, 80), "missing", fill=font_fill)
            canvas = Image.new("RGB", (image.width, image.height + label_height), "white")
            canvas.paste(image, (0, label_height))
            draw = ImageDraw.Draw(canvas)
            draw.text((6, 4), label, fill=font_fill)
            rendered_cells.append(canvas)
            max_height = max(max_height, canvas.height)
        row_width = sum(cell.width for cell in rendered_cells) + padding * (len(rendered_cells) - 1)
        row_image = Image.new("RGB", (row_width, max_height), "white")
        x = 0
        for cell in rendered_cells:
            row_image.paste(cell, (x, 0))
            x += cell.width + padding
        rendered_rows.append(row_image)
    width = max(row.width for row in rendered_rows)
    body_height = sum(row.height for row in rendered_rows) + padding * (len(rendered_rows) - 1)
    title_height = 32
    final = Image.new("RGB", (width, body_height + title_height), "white")
    draw = ImageDraw.Draw(final)
    draw.text((8, 6), title, fill=font_fill)
    y = title_height
    for row in rendered_rows:
        final.paste(row, (0, y))
        y += row.height + padding
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final.save(output_path)
    return output_path


def _prediction_run_dir(root: Path, seed: str, model_type: str) -> Path:
    return root / f"seed_{seed}_{model_type}"


def _write_comparison_grids(root: Path, output_dir: Path) -> dict[str, Path]:
    prediction_rows = []
    gradient_rows = []
    boundary_rows = []
    for seed in ("0", "1", "2"):
        prediction_rows.append(
            [
                (f"seed {seed} baseline prediction", _prediction_run_dir(root, seed, "baseline") / "prediction_grid.png"),
                (f"seed {seed} boundary_aux prediction", _prediction_run_dir(root, seed, "boundary_aux") / "prediction_grid.png"),
            ]
        )
        gradient_rows.append(
            [
                (f"seed {seed} baseline gradient", _prediction_run_dir(root, seed, "baseline") / "gradient_grid.png"),
                (f"seed {seed} boundary_aux gradient", _prediction_run_dir(root, seed, "boundary_aux") / "gradient_grid.png"),
            ]
        )
        boundary_rows.append(
            [
                (f"seed {seed} boundary prediction", _prediction_run_dir(root, seed, "boundary_aux") / "boundary_prediction_grid.png"),
                (f"seed {seed} boundary target", _prediction_run_dir(root, seed, "boundary_aux") / "boundary_target_grid.png"),
            ]
        )
    return {
        "prediction_grid_path": _tile_images(prediction_rows, output_dir / "selected_seed_prediction_grid_comparison.png", title="Selected seed prediction grid comparison"),
        "gradient_grid_path": _tile_images(gradient_rows, output_dir / "selected_seed_gradient_grid_comparison.png", title="Selected seed gradient grid comparison"),
        "boundary_grid_path": _tile_images(boundary_rows, output_dir / "selected_seed_boundary_grid_comparison.png", title="Selected seed boundary grid comparison"),
    }


def _write_claims(path: Path) -> Path:
    lines = [
        "# Protocol V7 Claims And Limitations",
        "",
        "## 可以写",
        "- Boundary auxiliary head can enter the real OpenFWI compact NPZ training pipeline.",
        "- In the selected matched setting, boundary_aux lowers gradient_error and edge_MAE in 3/3 seeds.",
        "- MAE/RMSE do not show obvious trade-off.",
        "- Results suggest boundary auxiliary is a promising direction for mitigating over-smoothed velocity boundaries.",
        "",
        "## 不能写",
        "- boundary auxiliary improves FWI generalization.",
        "- boundary auxiliary solves structural recovery.",
        "- boundary auxiliary is benchmark-proven.",
        "- current velocity predictions are application-level.",
        "- SAM/NCS results are supported by this run.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_report(
    *,
    root: Path,
    output_dir: Path,
    paired_rows: list[dict[str, Any]],
    wins: dict[str, int],
) -> Path:
    prior_report_path = root / "protocol_v7_boundary_auxiliary_seed_stability_report.md"
    prior_excerpt = prior_report_path.read_text(encoding="utf-8") if prior_report_path.exists() else ""
    lines = [
        "# Protocol V7 Selected Multi-seed Boundary Auxiliary Report",
        "",
        "## 1. Goal",
        "本轮只整理 selected boundary auxiliary 的 seed=0/1/2 稳定性证据，不新增训练，不做 benchmark claim。",
        "",
        "## 2. Matched Comparison",
        "- baseline 与 boundary_aux 仅 decoder/loss 不同，其余设置保持一致。",
        "- source / target split 与 V7 selected seed stability 一致。",
        "- bridge = raw_envelope_spectrum3",
        "- geometry disabled",
        "- aggregator = mean",
        "- backbone = cnn_baseline",
        "- train_size / val_size / test_size = 100 / 50 / 50",
        "- epochs = 2",
        "- metric_space = physical_velocity",
        "",
        "## 3. Metrics Table",
        "| seed | baseline_MAE | boundary_MAE | baseline_RMSE | boundary_RMSE | baseline_SSIM | boundary_SSIM | baseline_gradient_error | boundary_gradient_error | baseline_edge_MAE | boundary_edge_MAE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in paired_rows:
        lines.append(
            f"| {row['seed']} | {row['baseline_MAE']} | {row['boundary_MAE']} | {row['baseline_RMSE']} | {row['boundary_RMSE']} | "
            f"{row['baseline_SSIM']} | {row['boundary_SSIM']} | {row['baseline_gradient_error']} | {row['boundary_gradient_error']} | "
            f"{row['baseline_edge_MAE']} | {row['boundary_edge_MAE']} |"
        )
    lines.extend(
        [
            "",
            "## 4. Win Counts",
            f"- gradient_error: {wins['gradient_error_lower']}/3",
            f"- edge_MAE: {wins['edge_MAE_lower']}/3",
            f"- MAE: {wins['MAE_lower']}/3",
            f"- RMSE: {wins['RMSE_lower']}/3",
            f"- SSIM: {wins['SSIM_higher']}/3",
            "",
            "## 5. Scientific Interpretation",
            "- selected boundary auxiliary shows stable structural benefit in this CPU small-sample matched setting.",
            "- numerical trade-off is not obvious.",
            "- SSIM improvement is not stable.",
            "- boundary auxiliary is a promising candidate for mitigating over-smoothed velocity boundaries.",
            "- 本报告不写泛化提升结论，也不写 application-level performance。",
            "",
            "## 6. Visual Evidence",
            "- prediction_grid / gradient_grid / boundary_grid 已整理为 selected seed 对比图。",
            "- 新图只是更接近速度模型的可视化画法；可视化改善不是模型性能改善。",
            "- 模型预测仍偏平滑，复杂构造恢复仍不足。",
            "",
            "## 7. Limitations",
            "- CPU small-sample",
            "- only seed=0/1/2",
            "- selected setting only",
            "- train_size=100 / val_size=50 / test_size=50",
            "- 2 epochs only",
            "- boundary target is gradient-derived, not manually interpreted geology",
            "- no DINOv2/SAM/NCS in this run",
            "- not benchmark-level proof",
            "",
            "## 8. Next Step",
            "- 小范围调参 lambda_boundary=0.03/0.05/0.10；",
            "- 对比 gradient_magnitude / sobel / thresholded_gradient；",
            "- 若结构收益继续稳定，再扩大 train_size 或 seed；",
            "- 不要立即扩大自然图像 backbone。",
        ]
    )
    if prior_excerpt:
        lines.extend(["", "## Appendix", "- 已复用既有 seed stability report 作为前序结论输入。"])
    path = output_dir / "protocol_v7_selected_multiseed_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_protocol_v7_selected_multiseed_report(root: str | Path, output_dir: str | Path) -> dict[str, Path]:
    root_path = Path(root)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rows = _read_rows(root_path)
    paired_rows = _pair_rows(rows)
    wins = compute_seed_stability_win_counts(rows)

    summary_path = _write_summary_csv(output_path / "protocol_v7_selected_multiseed_summary.csv", paired_rows)
    metrics_bar_path = _draw_metrics_bar(output_path / "boundary_aux_vs_baseline_metrics_bar.png", paired_rows)
    win_count_bar_path = _draw_win_count_bar(output_path / "boundary_aux_win_count_bar.png", wins)
    grid_paths = _write_comparison_grids(root_path, output_path)
    claims_path = _write_claims(output_path / "protocol_v7_claims_and_limitations.md")
    report_path = _write_report(root=root_path, output_dir=output_path, paired_rows=paired_rows, wins=wins)
    return {
        "report_path": report_path,
        "summary_path": summary_path,
        "metrics_bar_path": metrics_bar_path,
        "win_count_bar_path": win_count_bar_path,
        "prediction_grid_path": grid_paths["prediction_grid_path"],
        "gradient_grid_path": grid_paths["gradient_grid_path"],
        "boundary_grid_path": grid_paths["boundary_grid_path"],
        "claims_path": claims_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write Protocol V7 selected multi-seed report.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    result = write_protocol_v7_selected_multiseed_report(**vars(parse_args()))
    printable = {key: str(value) for key, value in result.items()}
    print(json.dumps(printable, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
