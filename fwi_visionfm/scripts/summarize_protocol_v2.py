from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

METRICS = ["MAE", "RMSE", "SSIM", "PSNR", "gradient_error"]
FIELDNAMES = [
    "source_family",
    "target_family",
    "model_name",
    "bridge",
    "seed",
    "metric_space",
    "val_MAE",
    "val_RMSE",
    "val_SSIM",
    "val_PSNR",
    "val_gradient_error",
    "in_family_MAE",
    "in_family_RMSE",
    "in_family_SSIM",
    "in_family_PSNR",
    "in_family_gradient_error",
    "cross_family_MAE",
    "cross_family_RMSE",
    "cross_family_SSIM",
    "cross_family_PSNR",
    "cross_family_gradient_error",
    "cross_minus_in_MAE",
    "cross_minus_in_RMSE",
    "cross_minus_in_gradient_error",
    "status",
    "skip_reason",
    "runtime_seconds",
]
BRIDGE_COMPARISON_FIELDS = [
    "source_family",
    "target_family",
    "model_name",
    "seed",
    "raw_bridge",
    "spectrogram_bridge",
    "raw_cross_MAE",
    "spectrogram_cross_MAE",
    "delta_cross_MAE",
    "raw_cross_RMSE",
    "spectrogram_cross_RMSE",
    "delta_cross_RMSE",
    "raw_cross_SSIM",
    "spectrogram_cross_SSIM",
    "delta_cross_SSIM",
    "raw_cross_gradient_error",
    "spectrogram_cross_gradient_error",
    "delta_cross_gradient_error",
    "numerical_gain",
    "structural_gradient_gain",
    "numerical_gain_without_structural_gain",
]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(metrics: dict[str, Any], name: str) -> Any:
    keys = {
        "MAE": "mae",
        "RMSE": "rmse",
        "SSIM": "ssim",
        "PSNR": "psnr",
        "gradient_error": "gradient_error",
    }
    return metrics.get(keys[name], "")


def collect_rows(root: str | Path) -> list[dict[str, Any]]:
    rows = []
    for config_path in sorted(Path(root).glob("*_to_*/*/*/seed_*/config.json")):
        run_dir = config_path.parent
        config = _read_json(config_path)
        val = _read_json(run_dir / "metrics_val.json")
        in_metrics = _read_json(run_dir / "metrics_in_family_test.json")
        cross = _read_json(run_dir / "metrics_cross_family_test.json")
        row = {
            "source_family": config.get("source_family", ""),
            "target_family": config.get("target_family", ""),
            "model_name": config.get("model_name", ""),
            "bridge": config.get("bridge", ""),
            "seed": config.get("seed", ""),
            "metric_space": cross.get("metric_space") or in_metrics.get("metric_space") or val.get("metric_space") or config.get("metric_space", ""),
            "status": config.get("status", "UNKNOWN"),
            "skip_reason": config.get("skip_reason", ""),
            "runtime_seconds": config.get("runtime_seconds", ""),
        }
        for prefix, payload in (("val", val), ("in_family", in_metrics), ("cross_family", cross)):
            for metric in METRICS:
                row[f"{prefix}_{metric}"] = _metric(payload, metric)
        row["cross_minus_in_MAE"] = _delta(row.get("cross_family_MAE"), row.get("in_family_MAE"))
        row["cross_minus_in_RMSE"] = _delta(row.get("cross_family_RMSE"), row.get("in_family_RMSE"))
        row["cross_minus_in_gradient_error"] = _delta(row.get("cross_family_gradient_error"), row.get("in_family_gradient_error"))
        rows.append(row)
    return rows


def _delta(left: Any, right: Any) -> Any:
    try:
        return float(left) - float(right)
    except (TypeError, ValueError):
        return ""


def _build_bridge_comparison(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row.get("status") != "SUCCESS":
            continue
        key = (row["source_family"], row["target_family"], row["model_name"], str(row["seed"]))
        grouped[key][row["bridge"]] = row
    comparisons: list[dict[str, Any]] = []
    for (source_family, target_family, model_name, seed), bridges in sorted(grouped.items()):
        raw_row = bridges.get("raw_repeat3")
        spec_row = bridges.get("raw_spectrogram")
        if raw_row is None or spec_row is None:
            continue
        delta_mae = _delta(spec_row.get("cross_family_MAE"), raw_row.get("cross_family_MAE"))
        delta_rmse = _delta(spec_row.get("cross_family_RMSE"), raw_row.get("cross_family_RMSE"))
        delta_ssim = _delta(spec_row.get("cross_family_SSIM"), raw_row.get("cross_family_SSIM"))
        delta_grad = _delta(spec_row.get("cross_family_gradient_error"), raw_row.get("cross_family_gradient_error"))
        numerical_gain = _bool_pair(
            spec_row.get("cross_family_MAE"),
            raw_row.get("cross_family_MAE"),
            spec_row.get("cross_family_RMSE"),
            raw_row.get("cross_family_RMSE"),
        )
        structural_gradient_gain = _compare_less(spec_row.get("cross_family_gradient_error"), raw_row.get("cross_family_gradient_error"))
        structural_degradation = _compare_greater(spec_row.get("cross_family_gradient_error"), raw_row.get("cross_family_gradient_error"))
        comparisons.append(
            {
                "source_family": source_family,
                "target_family": target_family,
                "model_name": model_name,
                "seed": seed,
                "raw_bridge": "raw_repeat3",
                "spectrogram_bridge": "raw_spectrogram",
                "raw_cross_MAE": raw_row.get("cross_family_MAE", ""),
                "spectrogram_cross_MAE": spec_row.get("cross_family_MAE", ""),
                "delta_cross_MAE": delta_mae,
                "raw_cross_RMSE": raw_row.get("cross_family_RMSE", ""),
                "spectrogram_cross_RMSE": spec_row.get("cross_family_RMSE", ""),
                "delta_cross_RMSE": delta_rmse,
                "raw_cross_SSIM": raw_row.get("cross_family_SSIM", ""),
                "spectrogram_cross_SSIM": spec_row.get("cross_family_SSIM", ""),
                "delta_cross_SSIM": delta_ssim,
                "raw_cross_gradient_error": raw_row.get("cross_family_gradient_error", ""),
                "spectrogram_cross_gradient_error": spec_row.get("cross_family_gradient_error", ""),
                "delta_cross_gradient_error": delta_grad,
                "numerical_gain": numerical_gain,
                "structural_gradient_gain": structural_gradient_gain,
                "numerical_gain_without_structural_gain": bool(numerical_gain and structural_degradation),
            }
        )
    return comparisons


def _compare_less(left: Any, right: Any) -> bool:
    try:
        return float(left) < float(right)
    except (TypeError, ValueError):
        return False


def _compare_greater(left: Any, right: Any) -> bool:
    try:
        return float(left) > float(right)
    except (TypeError, ValueError):
        return False


def _bool_pair(spec_mae: Any, raw_mae: Any, spec_rmse: Any, raw_rmse: Any) -> bool:
    try:
        return float(spec_mae) < float(raw_mae) and float(spec_rmse) < float(raw_rmse)
    except (TypeError, ValueError):
        return False


def write_summary(root: str | Path) -> dict[str, Path]:
    output_root = Path(root)
    rows = collect_rows(output_root)
    summary_path = output_root / "protocol_v2_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    bridge_path = output_root / "protocol_v2_bridge_comparison.csv"
    bridge_rows = _build_bridge_comparison(rows)
    with bridge_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=BRIDGE_COMPARISON_FIELDS)
        writer.writeheader()
        writer.writerows(bridge_rows)
    group_path = output_root / "protocol_v2_summary_by_group.csv"
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["source_family"], row["target_family"], row["model_name"], row["bridge"])].append(row)
    group_fields = ["source_family", "target_family", "model_name", "bridge", "successful_seeds", "metric_space"]
    value_fields = [field for field in FIELDNAMES if field not in {"source_family", "target_family", "model_name", "bridge", "seed", "status", "metric_space"}]
    group_fields.extend([f"{field}_mean" for field in value_fields])
    group_fields.extend([f"{field}_std" for field in value_fields])
    with group_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=group_fields)
        writer.writeheader()
        for key, group_rows in sorted(grouped.items()):
            successful = [row for row in group_rows if row.get("status") == "SUCCESS"]
            out = {
                "source_family": key[0],
                "target_family": key[1],
                "model_name": key[2],
                "bridge": key[3],
                "successful_seeds": len(successful),
                "metric_space": successful[0]["metric_space"] if successful else "",
            }
            for field in value_fields:
                values = [_float(row.get(field)) for row in successful]
                values = [value for value in values if value is not None]
                out[f"{field}_mean"] = statistics.mean(values) if values else ""
                out[f"{field}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0 if values else ""
            writer.writerow(out)
    return {"summary": summary_path, "group_summary": group_path, "bridge_comparison": bridge_path}


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize protocol v2 results.")
    parser.add_argument("--root", type=Path, default=Path("outputs/protocol_v2_small"))
    return parser.parse_args()


def main() -> None:
    paths = write_summary(parse_args().root)
    print(f"Wrote {paths['summary']}")
    print(f"Wrote {paths['group_summary']}")
    print(f"Wrote {paths['bridge_comparison']}")


if __name__ == "__main__":
    main()
