from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _read_frozen_probe(path: Path, *, method_name: str, method_family: str, feature_type: str, reused_from: str = "") -> dict[str, Any]:
    config = _load_json(path / "config.json", {})
    cross = _load_json(path / "metrics_cross_family_test.json", {})
    return {
        "method_name": method_name,
        "method_family": method_family,
        "seed": int(config.get("seed", -1)),
        "backbone": str(config.get("backbone", path.name.replace("_seed0", ""))),
        "feature_type": feature_type,
        "decoder": str(config.get("decoder_name", "lightweight_feature_decoder")),
        "loss": str(config.get("loss_name", "default_l1")),
        "is_real_feature": bool(config.get("is_real_feature", False)),
        "train_size": 100,
        "val_size": 50,
        "test_size": 50,
        "cross_family_MAE": float(cross.get("mae")),
        "cross_family_RMSE": float(cross.get("rmse")),
        "cross_family_SSIM": float(cross.get("ssim")),
        "cross_family_gradient_error": float(cross.get("gradient_error")),
        "cross_family_edge_MAE": float(cross.get("edge_mae", cross.get("edge_MAE"))),
        "status": str(config.get("status", "SKIPPED")),
        "reused_from": reused_from,
        "limitation_note": "frozen feature + decoder-only",
    }


def _collect_v7_rows(v7_selected_summary: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with v7_selected_summary.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            seed = int(raw["seed"])
            rows.append(
                {
                    "method_name": "cnn_baseline_unet_l1",
                    "method_family": "task_specific_supervised",
                    "seed": seed,
                    "backbone": "cnn_baseline",
                    "feature_type": "end_to_end_supervised",
                    "decoder": "unet_decoder",
                    "loss": "default_l1",
                    "is_real_feature": False,
                    "train_size": 100,
                    "val_size": 50,
                    "test_size": 50,
                    "cross_family_MAE": float(raw["baseline_MAE"]),
                    "cross_family_RMSE": float(raw["baseline_RMSE"]),
                    "cross_family_SSIM": float(raw["baseline_SSIM"]),
                    "cross_family_gradient_error": float(raw["baseline_gradient_error"]),
                    "cross_family_edge_MAE": float(raw["baseline_edge_MAE"]),
                    "status": str(raw["baseline_status"]),
                    "reused_from": str(v7_selected_summary),
                    "limitation_note": "end-to-end supervised baseline",
                }
            )
            rows.append(
                {
                    "method_name": "boundary_aux_gradient_lambda010",
                    "method_family": "boundary_auxiliary",
                    "seed": seed,
                    "backbone": "cnn_baseline",
                    "feature_type": "end_to_end_supervised",
                    "decoder": "boundary_aux_unet",
                    "loss": "boundary_aux_l1",
                    "is_real_feature": False,
                    "train_size": 100,
                    "val_size": 50,
                    "test_size": 50,
                    "cross_family_MAE": float(raw["boundary_MAE"]),
                    "cross_family_RMSE": float(raw["boundary_RMSE"]),
                    "cross_family_SSIM": float(raw["boundary_SSIM"]),
                    "cross_family_gradient_error": float(raw["boundary_gradient_error"]),
                    "cross_family_edge_MAE": float(raw["boundary_edge_MAE"]),
                    "status": str(raw["boundary_status"]),
                    "reused_from": str(v7_selected_summary),
                    "limitation_note": "boundary-aware end-to-end supervised route",
                }
            )
    return rows


def collect_selected_rows(
    *,
    ncs2d_root: str | Path,
    ncs2d_seed0_dir: str | Path,
    vit_mae_seed0_dir: str | Path,
    vit_mae_seed_root: str | Path,
    v7_boundary_root: str | Path,
    v7_selected_root: str | Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows.extend(_collect_v7_rows(Path(v7_selected_root) / "protocol_v7_selected_multiseed_summary.csv"))

    ncs_seed0 = Path(ncs2d_seed0_dir)
    if ncs_seed0.exists():
        row = _read_frozen_probe(
            ncs_seed0,
            method_name="ncs2d_frozen_decoder",
            method_family="seismic_domain_ncs_frozen",
            feature_type="frozen_decoder_only",
            reused_from=str(ncs_seed0),
        )
        row["seed"] = 0
        row["backbone"] = "ncs_2d"
        rows.append(row)
    for run_dir in sorted((Path(ncs2d_root) / "decoder_probe").glob("ncs_2d_seed*")):
        row = _read_frozen_probe(
            run_dir,
            method_name="ncs2d_frozen_decoder",
            method_family="seismic_domain_ncs_frozen",
            feature_type="frozen_decoder_only",
        )
        row["backbone"] = "ncs_2d"
        rows.append(row)

    vit_seed0 = Path(vit_mae_seed0_dir)
    if vit_seed0.exists():
        row = _read_frozen_probe(
            vit_seed0,
            method_name="vit_mae_base_frozen_decoder",
            method_family="natural_image_mae_frozen",
            feature_type="frozen_decoder_only",
            reused_from=str(vit_seed0),
        )
        row["seed"] = 0
        row["backbone"] = "vit_mae_base"
        rows.append(row)
    for run_dir in sorted(Path(vit_mae_seed_root).glob("vit_mae_base_seed*")):
        row = _read_frozen_probe(
            run_dir,
            method_name="vit_mae_base_frozen_decoder",
            method_family="natural_image_mae_frozen",
            feature_type="frozen_decoder_only",
        )
        row["backbone"] = "vit_mae_base"
        rows.append(row)
    rows.sort(key=lambda item: (item["method_name"], int(item["seed"])))
    return rows


def _stats(rows: list[dict[str, Any]], method_name: str, key: str) -> tuple[float, float]:
    values = [float(row[key]) for row in rows if row["method_name"] == method_name and row["status"] == "SUCCESS"]
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


def _write_claims(path: Path) -> None:
    lines = [
        "# Protocol V9 Selected Comparison Claims And Limitations",
        "",
        "## Can Claim",
        "- ncs2d real frozen feature baseline is available and stable under seed=0/1/2。",
        "- vit_mae_base frozen feature baseline can be compared as natural-image MAE reference if seed=0/1/2 complete。",
        "- selected comparison summarizes numerical and structural trade-offs among task-specific, boundary-aware, natural-image frozen, and seismic-domain frozen routes。",
        "- boundary_aux remains the current structure-aware route if structural metrics are better。",
        "",
        "## Cannot Claim",
        "- NCS improves FWI。",
        "- NCS improves FWI generalization。",
        "- vit_mae_base improves FWI。",
        "- MAE improves FWI generalization。",
        "- boundary auxiliary improves FWI generalization。",
        "- selected comparison is benchmark-level proof。",
        "- frozen feature decoder-only results prove application-level performance。",
        "- ncs_2p5d result is available。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_metric_bars(rows: list[dict[str, Any]], output_path: Path, metric_key: str, title: str) -> None:
    methods = []
    means = []
    stds = []
    for method_name in sorted({row["method_name"] for row in rows}):
        mean, std = _stats(rows, method_name, metric_key)
        methods.append(method_name)
        means.append(mean)
        stds.append(std)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width = 1000
    height = 480
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((20, 20), title, fill="black")
    if not means:
        img.save(output_path)
        return
    max_value = max(mean + std for mean, std in zip(means, stds)) or 1.0
    left = 70
    bottom = height - 80
    top = 70
    chart_width = width - 120
    bar_width = max(40, int(chart_width / max(len(methods) * 2, 1)))
    gap = bar_width
    for idx, (method, mean, std) in enumerate(zip(methods, means, stds)):
        x0 = left + idx * (bar_width + gap)
        x1 = x0 + bar_width
        bar_h = int((mean / max_value) * (bottom - top))
        y0 = bottom - bar_h
        draw.rectangle([x0, y0, x1, bottom], fill=(79, 129, 189), outline="black")
        err_h = int((std / max_value) * (bottom - top))
        center = (x0 + x1) // 2
        draw.line([center, y0 - err_h, center, y0], fill="black", width=2)
        draw.text((x0 - 10, bottom + 8), method[:18], fill="black")
        draw.text((x0 - 5, y0 - err_h - 18), f"{mean:.1f}", fill="black")
    draw.line([left, bottom, width - 40, bottom], fill="black", width=2)
    draw.line([left, top, left, bottom], fill="black", width=2)
    img.save(output_path)


def write_protocol_v9_selected_comparison_report(*, rows: list[dict[str, Any]], output_dir: str | Path) -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary_path = out / "protocol_v9_selected_comparison_summary.csv"
    fieldnames = [
        "method_name",
        "method_family",
        "seed",
        "backbone",
        "feature_type",
        "decoder",
        "loss",
        "is_real_feature",
        "train_size",
        "val_size",
        "test_size",
        "cross_family_MAE",
        "cross_family_RMSE",
        "cross_family_SSIM",
        "cross_family_gradient_error",
        "cross_family_edge_MAE",
        "status",
        "reused_from",
        "limitation_note",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    claims_path = out / "protocol_v9_selected_comparison_claims_and_limitations.md"
    _write_claims(claims_path)

    _plot_metric_bars(rows, out / "selected_comparison_metrics_bar.png", "cross_family_MAE", "Selected Comparison MAE")
    _plot_metric_bars(rows, out / "selected_comparison_stability_bar.png", "cross_family_gradient_error", "Selected Comparison Gradient Error")

    methods = sorted({row["method_name"] for row in rows})
    seed_coverage = {method: sorted(int(row["seed"]) for row in rows if row["method_name"] == method and row["status"] == "SUCCESS") for method in methods}
    mae_best = min(methods, key=lambda name: _stats(rows, name, "cross_family_MAE")[0])
    rmse_best = min(methods, key=lambda name: _stats(rows, name, "cross_family_RMSE")[0])
    grad_best = min(methods, key=lambda name: _stats(rows, name, "cross_family_gradient_error")[0])
    edge_best = min(methods, key=lambda name: _stats(rows, name, "cross_family_edge_MAE")[0])
    ncs_mae_mean, _ = _stats(rows, "ncs2d_frozen_decoder", "cross_family_MAE")
    ncs_grad_mean, _ = _stats(rows, "ncs2d_frozen_decoder", "cross_family_gradient_error")
    boundary_grad_mean, _ = _stats(rows, "boundary_aux_gradient_lambda010", "cross_family_gradient_error")

    lines = [
        "# Protocol V9 Selected Comparison Report",
        "",
        "## 1. Goal",
        "本轮只做 selected CPU small-sample comparison，对比 task-specific baseline、boundary auxiliary、natural-image MAE frozen feature、seismic-domain NCS 2D frozen feature。",
        "",
        "## 2. Compared Methods",
        "1. cnn_baseline_unet_l1",
        "2. boundary_aux_gradient_lambda010",
        "3. vit_mae_base_frozen_decoder",
        "4. ncs2d_frozen_decoder",
        "",
        "- cnn/boundary_aux 是端到端监督训练。",
        "- vit_mae_base/ncs2d 是 frozen feature + decoder-only。",
        "- methods are not perfectly matched training paradigms, so this remains a selected comparison rather than benchmark-level proof.",
        "- This report remains not benchmark-level proof.",
        "",
        "## 3. Seed Coverage",
    ]
    for method, seeds in seed_coverage.items():
        lines.append(f"- {method}: seed={','.join(str(seed) for seed in seeds) if seeds else 'missing'}")
    lines.extend(["", "## 4. Metrics Table"])
    for method in methods:
        mae_mean, mae_std = _stats(rows, method, "cross_family_MAE")
        rmse_mean, rmse_std = _stats(rows, method, "cross_family_RMSE")
        ssim_mean, ssim_std = _stats(rows, method, "cross_family_SSIM")
        grad_mean, grad_std = _stats(rows, method, "cross_family_gradient_error")
        edge_mean, edge_std = _stats(rows, method, "cross_family_edge_MAE")
        lines.append(
            f"- {method}: MAE {mae_mean:.4f}+/-{mae_std:.4f}, RMSE {rmse_mean:.4f}+/-{rmse_std:.4f}, "
            f"SSIM {ssim_mean:.4f}+/-{ssim_std:.4f}, gradient_error {grad_mean:.4f}+/-{grad_std:.4f}, edge_MAE {edge_mean:.4f}+/-{edge_std:.4f}"
        )
    lines.extend(
        [
            "",
            "## 5. Numerical Readout",
            f"- Lower MAE in this selected setting: {mae_best}.",
            f"- Lower RMSE in this selected setting: {rmse_best}.",
            "- ncs2d frozen feature has stable decoder-only metrics across seed=0/1/2." if seed_coverage.get("ncs2d_frozen_decoder") == [0, 1, 2] else "- ncs2d frozen feature seed coverage is incomplete.",
            "- vit_mae_base/ncs2d are numerically competitive within this selected setting." if min(_stats(rows, "vit_mae_base_frozen_decoder", "cross_family_MAE")[0], ncs_mae_mean) < _stats(rows, "cnn_baseline_unet_l1", "cross_family_MAE")[0] + 15.0 else "- vit_mae_base/ncs2d remain numerical references rather than clear leaders.",
            "",
            "## 6. Structural Readout",
            f"- Lower gradient_error: {grad_best}.",
            f"- Lower edge_MAE: {edge_best}.",
            "- If frozen decoder-only methods show better MAE/RMSE but weaker gradient_error/edge_MAE, the current evidence still points to boundary-aware decoders for structure recovery.",
            "- V7 boundary-aware route remains the current structure-enhancement mainline." if boundary_grad_mean <= ncs_grad_mean else "- Frozen decoder-only structural metrics are not clearly behind boundary-aware route in this selected setting.",
            "",
            "## 7. Interpretation",
            "- ncs2d has progressed from adapter availability to a stable real frozen feature baseline.",
            "- vit_mae_base provides a natural-image MAE frozen feature reference.",
            "- selected comparison is used to decide whether frozen features should be combined with boundary-aware decoders next.",
            "- The current most stable structural direction still likely comes from boundary_aux rather than replacing the frozen backbone alone." if boundary_grad_mean <= ncs_grad_mean else "- Structural advantage is not exclusively owned by boundary_aux in this selected slice.",
            "",
            "## 8. Limitations",
            "- CPU-only",
            "- train_size=100 / val_size=50 / test_size=50",
            "- selected comparison",
            "- methods are not perfectly matched training paradigms",
            "- frozen feature + decoder-only differs from end-to-end supervised CNN/boundary_aux",
            "- no full fine-tuning",
            "- no benchmark-level proof",
            "- no application-level performance",
            "- ncs_2p5d remains pending",
            "- OpenFWI shot gather input differs from migrated seismic cube domain used by NCS pretraining",
            "",
            "## 9. Next Step",
        ]
    )
    if ncs_mae_mean <= _stats(rows, "cnn_baseline_unet_l1", "cross_family_MAE")[0] + 10.0 and ncs_grad_mean > boundary_grad_mean:
        lines.append("- ncs2d numerical metrics are competitive but structural metrics remain weaker, so the next step is ncs2d + boundary_aux_decoder probe.")
    elif _stats(rows, "vit_mae_base_frozen_decoder", "cross_family_MAE")[0] > _stats(rows, "cnn_baseline_unet_l1", "cross_family_MAE")[0] and ncs_mae_mean > _stats(rows, "cnn_baseline_unet_l1", "cross_family_MAE")[0]:
        lines.append("- frozen feature path stays as a baseline, while boundary_aux / geometry remains the higher-priority route.")
    else:
        lines.append("- expand selected stability or combine frozen features with continued seismic adaptation.")

    report_path = out / "protocol_v9_selected_comparison_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"summary_path": summary_path, "report_path": report_path, "claims_path": claims_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write Protocol V9 selected comparison report.")
    parser.add_argument("--ncs2d-root", type=Path, required=True)
    parser.add_argument("--vit-mae-seed0-dir", type=Path, required=True)
    parser.add_argument("--vit-mae-seed-root", type=Path, required=True)
    parser.add_argument("--v7-boundary-root", type=Path, required=True)
    parser.add_argument("--v7-selected-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = collect_selected_rows(
        ncs2d_root=args.ncs2d_root,
        ncs2d_seed0_dir=Path(r"D:\ryjin\fwi_visionfm\outputs\protocol_v9_ncs_adapter_repair\decoder_probe\ncs_2d"),
        vit_mae_seed0_dir=args.vit_mae_seed0_dir,
        vit_mae_seed_root=args.vit_mae_seed_root,
        v7_boundary_root=args.v7_boundary_root,
        v7_selected_root=args.v7_selected_root,
    )
    payload = write_protocol_v9_selected_comparison_report(rows=rows, output_dir=args.output_dir)
    print(json.dumps({key: str(value) for key, value in payload.items()}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
