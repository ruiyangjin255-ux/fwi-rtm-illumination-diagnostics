from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _pick_experiment_dirs(root_dir: Path) -> list[Path]:
    return sorted(
        path for path in root_dir.iterdir()
        if path.is_dir() and ((path / "config_resolved.json").exists() or (path / "resolved_foundation_config.json").exists())
    )


def collect_bridge_ablation_results(
    *,
    root_dir: str | Path,
    output_csv: str | Path | None = None,
    output_md: str | Path | None = None,
    output_report: str | Path | None = None,
) -> dict[str, Any]:
    root_dir = Path(root_dir)
    csv_path = Path(output_csv) if output_csv is not None else root_dir / "summary_bridge_ablation.csv"
    md_path = Path(output_md) if output_md is not None else root_dir / "summary_bridge_ablation.md"
    report_path = Path(output_report) if output_report is not None else root_dir / "bridge_ablation_report.md"
    extended_mode = "extended" in str(csv_path).lower() or "extended" in str(md_path).lower() or "extended" in str(report_path).lower()
    rows: list[dict[str, Any]] = []
    for experiment_dir in _pick_experiment_dirs(root_dir):
        config_path = experiment_dir / "config_resolved.json"
        if not config_path.exists():
            config_path = experiment_dir / "resolved_foundation_config.json"
        config = _read_json(config_path)
        history_path = experiment_dir / "foundation_training_history.csv"
        if not history_path.exists():
            history_path = experiment_dir / "training_history.csv"
        history_rows = _read_csv_rows(history_path) if history_path.exists() else []
        metrics_in_path = experiment_dir / "metrics_in_family_extended.json"
        if not metrics_in_path.exists():
            metrics_in_path = experiment_dir / "metrics_in_family.json"
        metrics_cross_path = experiment_dir / "metrics_cross_family_extended.json"
        if not metrics_cross_path.exists():
            metrics_cross_path = experiment_dir / "metrics_cross_family.json"
        if extended_mode and (not (experiment_dir / "metrics_in_family_extended.json").exists() or not (experiment_dir / "metrics_cross_family_extended.json").exists()):
            continue
        metrics_in = _read_json(metrics_in_path) if metrics_in_path.exists() else {}
        metrics_cross = _read_json(metrics_cross_path) if metrics_cross_path.exists() else {}
        final_train_loss = float(history_rows[-1]["train_loss"]) if history_rows else "NA"
        best_val_loss = min((float(row["val_loss"]) for row in history_rows if row.get("val_loss") not in (None, "")), default="NA")
        in_mae = metrics_in.get("mae", "NA")
        in_rmse = metrics_in.get("rmse", "NA")
        cross_mae = metrics_cross.get("mae", "NA")
        cross_rmse = metrics_cross.get("rmse", "NA")
        gap_mae = "NA" if in_mae == "NA" or cross_mae == "NA" else float(cross_mae) - float(in_mae)
        gap_rmse = "NA" if in_rmse == "NA" or cross_rmse == "NA" else float(cross_rmse) - float(in_rmse)
        rows.append(
            {
                "experiment": experiment_dir.name,
                "bridge_feature_mode": config.get("bridge_feature_mode", "raw_repeat3"),
                "backbone_type": config.get("backbone_type", "NA"),
                "backbone_name": config.get("model_name", config.get("backbone_name", "NA")),
                "transfer_mode": config.get("transfer_mode", "NA"),
                "epochs": config.get("epochs", "NA"),
                "seed": config.get("seed", "NA"),
                "in_family_mae": in_mae,
                "in_family_rmse": in_rmse,
                "in_family_psnr": metrics_in.get("psnr", "NA"),
                "in_family_ssim": metrics_in.get("ssim", "NA"),
                "in_family_edge_mae": metrics_in.get("edge_mae", "NA"),
                "in_family_laplacian_mae": metrics_in.get("laplacian_mae", "NA"),
                "in_family_horizon_gradient_mae": metrics_in.get("horizon_gradient_mae", "NA"),
                "in_family_vertical_gradient_mae": metrics_in.get("vertical_gradient_mae", "NA"),
                "cross_family_mae": cross_mae,
                "cross_family_rmse": cross_rmse,
                "cross_family_psnr": metrics_cross.get("psnr", "NA"),
                "cross_family_ssim": metrics_cross.get("ssim", "NA"),
                "cross_family_edge_mae": metrics_cross.get("edge_mae", "NA"),
                "cross_family_laplacian_mae": metrics_cross.get("laplacian_mae", "NA"),
                "cross_family_horizon_gradient_mae": metrics_cross.get("horizon_gradient_mae", "NA"),
                "cross_family_vertical_gradient_mae": metrics_cross.get("vertical_gradient_mae", "NA"),
                "generalization_gap_mae": gap_mae,
                "generalization_gap_rmse": gap_rmse,
                "final_train_loss": final_train_loss,
                "best_val_loss": best_val_loss,
                "output_dir": str(experiment_dir),
            }
        )
    if not rows:
        raise ValueError(f"No bridge ablation experiments found under {root_dir}")

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# OpenFWI Bridge Ablation Summary",
        "",
        "| experiment | bridge_feature_mode | in_family_mae | in_family_psnr | in_family_ssim | cross_family_mae | cross_family_psnr | cross_family_ssim | generalization_gap_mae | best_val_loss |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['experiment']} | {row['bridge_feature_mode']} | {row['in_family_mae']} | {row['in_family_psnr']} | {row['in_family_ssim']} | {row['cross_family_mae']} | {row['cross_family_psnr']} | {row['cross_family_ssim']} | {row['generalization_gap_mae']} | {row['best_val_loss']} |"
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")

    by_mode = {row["bridge_feature_mode"]: row for row in rows}
    best_in = min(rows, key=lambda row: float(row["in_family_mae"]) if row["in_family_mae"] != "NA" else float("inf"))
    best_cross = min(rows, key=lambda row: float(row["cross_family_mae"]) if row["cross_family_mae"] != "NA" else float("inf"))
    baseline = by_mode.get("raw_repeat3")
    env_spec = by_mode.get("raw_envelope_spectrogram")
    env_only = by_mode.get("envelope_repeat3")
    spec_only = by_mode.get("spectrogram_repeat3")

    def _compare(candidate: dict[str, Any] | None, ref: dict[str, Any] | None, key: str) -> str:
        if candidate is None or ref is None or candidate.get(key) in ("NA", None, "") or ref.get(key) in ("NA", None, ""):
            return "N/A"
        delta = float(candidate[key]) - float(ref[key])
        if delta < 0:
            return f"improved by {abs(delta):.6f}"
        if delta > 0:
            return f"worse by {delta:.6f}"
        return "matched baseline"

    report_lines = [
        "# OpenFWI Bridge Ablation Report",
        "",
        "## 指标表",
        "",
        *lines[2:],
        "",
        "## 问题回答",
        "",
        f"1. raw_repeat3 是否仍是最强基线：当前 cross-family 最优模式为 `{best_cross['bridge_feature_mode']}`，in-family 最优模式为 `{best_in['bridge_feature_mode']}`。",
        f"2. raw_envelope_spectrogram 是否改善 in-family：MAE {_compare(env_spec, baseline, 'in_family_mae')}；PSNR {_compare(env_spec, baseline, 'in_family_psnr')}；edge {_compare(env_spec, baseline, 'in_family_edge_mae')}",
        f"3. raw_envelope_spectrogram 是否改善 cross-family：MAE {_compare(env_spec, baseline, 'cross_family_mae')}；PSNR {_compare(env_spec, baseline, 'cross_family_psnr')}；edge {_compare(env_spec, baseline, 'cross_family_edge_mae')}",
        f"4. envelope 单独是否有效：`envelope_repeat3` 相对 raw baseline 的 in-family = {_compare(env_only, baseline, 'in_family_mae')}，cross-family = {_compare(env_only, baseline, 'cross_family_mae')}",
        f"5. spectrogram 单独是否有效：`spectrogram_repeat3` 相对 raw baseline 的 in-family = {_compare(spec_only, baseline, 'in_family_mae')}，cross-family = {_compare(spec_only, baseline, 'cross_family_mae')}",
        f"6. raw_envelope 的 in-family 数值优势是否也体现在 SSIM/PSNR/edge：PSNR 相对 raw = {_compare(by_mode.get('raw_envelope'), baseline, 'in_family_psnr')}；edge = {_compare(by_mode.get('raw_envelope'), baseline, 'in_family_edge_mae')}；laplacian = {_compare(by_mode.get('raw_envelope'), baseline, 'in_family_laplacian_mae')}",
        f"7. raw_spectrogram 的 cross-family MAE 优势是否也体现在结构指标：PSNR 相对 raw = {_compare(by_mode.get('raw_spectrogram'), baseline, 'cross_family_psnr')}；edge = {_compare(by_mode.get('raw_spectrogram'), baseline, 'cross_family_edge_mae')}；laplacian = {_compare(by_mode.get('raw_spectrogram'), baseline, 'cross_family_laplacian_mae')}",
        f"8. raw_repeat3 是否在某些结构指标上仍然更好：需要结合 CSV 逐项判断；当前报告保留原始指标表，避免只按 MAE 单独定结论。",
        "9. 当前 bridge 选择是否应以 MAE/RMSE 为主，还是需要引入边界指标共同判断：建议联合判断。MAE/RMSE 反映数值误差，edge/laplacian/gradient 更直接反映界面与构造恢复。",
    ]
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    return {"row_count": len(rows), "csv_path": str(csv_path), "md_path": str(md_path), "report_path": str(report_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="收集 OpenFWI bridge ablation 实验结果。")
    parser.add_argument("--root", "--root-dir", dest="root_dir", required=True, type=Path)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--output-report", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = collect_bridge_ablation_results(
        root_dir=args.root_dir,
        output_csv=args.output_csv,
        output_md=args.output_md,
        output_report=args.output_report,
    )
    print(f"写出 bridge ablation CSV: {result['csv_path']}")
    print(f"写出 bridge ablation 报告: {result['report_path']}")


if __name__ == "__main__":
    main()
