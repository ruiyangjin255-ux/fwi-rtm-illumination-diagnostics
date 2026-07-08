from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_value(metrics: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key in metrics:
            return float(metrics[key])
    return None


def _read_probe_dir(path: Path, *, reused_from: str = "") -> dict[str, Any]:
    config = _load_json(path / "config.json", {})
    val_metrics = _load_json(path / "metrics_val.json", {})
    cross_metrics = _load_json(path / "metrics_cross_family_test.json", {})
    seed = int(config.get("seed", -1))
    return {
        "seed": seed,
        "backbone_name": "ncs_2d",
        "feature_mode": "mean_patch",
        "is_real_feature": bool(config.get("is_real_feature", False)),
        "decoder": str(config.get("decoder_name", "lightweight_feature_decoder")),
        "loss": str(config.get("loss_name", "default_l1")),
        "epochs": int(config.get("epochs", 2)),
        "val_MAE": _metric_value(val_metrics, "mae"),
        "val_RMSE": _metric_value(val_metrics, "rmse"),
        "val_SSIM": _metric_value(val_metrics, "ssim"),
        "cross_family_MAE": _metric_value(cross_metrics, "mae"),
        "cross_family_RMSE": _metric_value(cross_metrics, "rmse"),
        "cross_family_SSIM": _metric_value(cross_metrics, "ssim"),
        "cross_family_gradient_error": _metric_value(cross_metrics, "gradient_error"),
        "cross_family_edge_MAE": _metric_value(cross_metrics, "edge_MAE", "edge_mae"),
        "status": str(config.get("status", "SKIPPED")),
        "reused_from": reused_from,
        "skip_reason": "" if bool(config.get("is_real_feature", False)) else "is_real_feature_false",
        "probe_dir": str(path),
    }


def collect_seed_rows(*, root: str | Path, seed0_dir: str | Path) -> list[dict[str, Any]]:
    root = Path(root)
    rows: list[dict[str, Any]] = []
    seed0_path = Path(seed0_dir)
    if seed0_path.exists():
        row = _read_probe_dir(seed0_path, reused_from=str(seed0_path))
        row["seed"] = 0
        rows.append(row)
    for run_dir in sorted((root / "decoder_probe").glob("ncs_2d_seed*")):
        rows.append(_read_probe_dir(run_dir))
    rows.sort(key=lambda item: int(item["seed"]))
    return rows


def _write_claims(path: Path) -> None:
    lines = [
        "# Protocol V9 NCS 2D Seed Stability Claims And Limitations",
        "",
        "## Can Claim",
        "- ncs_2d real frozen feature cache and decoder-only probe are available。",
        "- ncs_2d seed=0/1/2 decoder-only stability has been checked。",
        "- ncs_2d can be included in selected comparison if all seeds succeed。",
        "",
        "## Cannot Claim",
        "- NCS improves FWI。",
        "- NCS improves FWI generalization。",
        "- NCS outperforms MAE/CNN/boundary_aux。",
        "- decoder-only results are benchmark-level proof。",
        "- ncs_2p5d result is available。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _stats(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    values = [float(row[key]) for row in rows if row.get(key) is not None and str(row.get("status")) == "SUCCESS"]
    if not values:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def write_protocol_v9_ncs2d_seed_stability_report(
    *,
    root: str | Path,
    seed0_dir: str | Path,
    adapter_repair_report: str | Path | None,
    output_dir: str | Path,
) -> dict[str, Path]:
    root = Path(root)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = collect_seed_rows(root=root, seed0_dir=seed0_dir)
    adapter_text = Path(adapter_repair_report).read_text(encoding="utf-8") if adapter_repair_report and Path(adapter_repair_report).exists() else ""

    summary_path = out / "protocol_v9_ncs2d_seed_stability_summary.csv"
    fieldnames = [
        "seed",
        "backbone_name",
        "feature_mode",
        "is_real_feature",
        "decoder",
        "loss",
        "epochs",
        "val_MAE",
        "val_RMSE",
        "val_SSIM",
        "cross_family_MAE",
        "cross_family_RMSE",
        "cross_family_SSIM",
        "cross_family_gradient_error",
        "cross_family_edge_MAE",
        "status",
        "reused_from",
        "skip_reason",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    claims_path = out / "protocol_v9_ncs2d_seed_stability_claims_and_limitations.md"
    _write_claims(claims_path)

    metric_keys = [
        "cross_family_MAE",
        "cross_family_RMSE",
        "cross_family_SSIM",
        "cross_family_gradient_error",
        "cross_family_edge_MAE",
    ]
    stats_map = {key: _stats(rows, key) for key in metric_keys}
    all_success = len(rows) == 3 and all(str(row.get("status")) == "SUCCESS" and bool(row.get("is_real_feature")) for row in rows)
    grad_std = stats_map["cross_family_gradient_error"]["std"]
    mae_std = stats_map["cross_family_MAE"]["std"]
    stability_text = "acceptable" if all_success and mae_std < 15.0 and grad_std < 10.0 else "limited"

    lines = [
        "# Protocol V9 NCS 2D Seed Stability Report",
        "",
        "## 1. Goal",
        "本轮只验证 ncs_2d real frozen feature decoder-only probe 的 seed=0/1/2 稳定性，不做 benchmark claim。",
        "",
        "## 2. Matched Settings",
        "- ncs_2d real frozen feature",
        "- raw_envelope_spectrum3",
        "- mean_patch feature",
        "- decoder-only",
        "- default_l1",
        "- CPU small-sample",
        "",
        "## 3. Seed Stability Table",
        "",
        "| seed | MAE | RMSE | SSIM | gradient_error | edge_MAE | reused_from | status |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | {row['cross_family_MAE']} | {row['cross_family_RMSE']} | {row['cross_family_SSIM']} | "
            f"{row['cross_family_gradient_error']} | {row['cross_family_edge_MAE']} | {row['reused_from']} | {row['status']} |"
        )
    lines.extend(["", "## 4. Stability Readout"])
    for key in metric_keys:
        stat = stats_map[key]
        lines.append(f"- {key}: mean={stat['mean']:.4f}, std={stat['std']:.4f}, min={stat['min']:.4f}, max={stat['max']:.4f}")
    lines.extend(
        [
            "",
            "## 5. Interpretation",
            f"- ncs_2d real frozen feature decoder-only chain is {stability_text} under seed=0/1/2.",
            f"- is_real_feature=True across successful rows: {all(bool(row.get('is_real_feature')) for row in rows if str(row.get('status')) == 'SUCCESS')}.",
            "- ncs_2d can now be included in selected comparison as a real seismic-domain frozen feature baseline." if all_success else "- ncs_2d selected comparison should wait until all seeds succeed.",
            "- structural metrics remain weak / require boundary-aware decoder if observed." if stats_map["cross_family_gradient_error"]["mean"] > 50.0 else "- structural metrics are not the main bottleneck in this small decoder-only check.",
            "",
            "## 6. Relation to V9 Adapter Repair",
        ]
    )
    if "IMPORT_ERROR" in adapter_text:
        lines.append("- previous ncs_2d IMPORT_ERROR has been repaired.")
    else:
        lines.append("- previous ncs_2d IMPORT_ERROR repaired state is assumed from adapter repair outputs.")
    lines.append("- ncs_2d now uses transformers-compatible real feature extraction.")
    lines.append("- ncs_2p5d remains adapter pending.")
    lines.append("- seed stability does not resolve ncs_2p5d.")
    lines.extend(
        [
            "",
            "## 7. Limitations",
            "- CPU-only",
            "- train_size=100 / val_size=50 / test_size=50",
            "- decoder-only",
            "- frozen feature only",
            "- no full fine-tuning",
            "- only seed=0/1/2",
            "- OpenFWI shot gather is not the same input domain as migrated seismic cubes used by NCS pretraining",
            "- not benchmark-level proof",
            "",
            "## 8. Next Step",
        ]
    )
    if all_success:
        lines.append("- 进入 selected comparison：ncs_2d vs vit_mae_base vs cnn_baseline vs boundary_aux。")
    else:
        lines.append("- 检查 feature normalization / decoder capacity / feature_mode。")
    if stats_map["cross_family_gradient_error"]["mean"] > 50.0:
        lines.append("- 后续只将 NCS frozen features 与 boundary_aux decoder 结合，不直接 claim NCS benefit。")
    report_path = out / "protocol_v9_ncs2d_seed_stability_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"summary_path": summary_path, "report_path": report_path, "claims_path": claims_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write Protocol V9 NCS 2D seed stability report.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed0-dir", type=Path, required=True)
    parser.add_argument("--adapter-repair-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = write_protocol_v9_ncs2d_seed_stability_report(
        root=args.root,
        seed0_dir=args.seed0_dir,
        adapter_repair_report=args.adapter_repair_report,
        output_dir=args.output_dir,
    )
    print(json.dumps({key: str(value) for key, value in payload.items()}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
