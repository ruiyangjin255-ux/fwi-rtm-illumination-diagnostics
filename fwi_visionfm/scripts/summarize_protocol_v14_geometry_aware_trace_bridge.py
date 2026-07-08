# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


COMMON_REQUIRED_FILES = [
    "config.json",
    "config_hash.txt",
    "model_card.json",
    "train_history.csv",
    "metrics_val.json",
    "metrics_in_family_test.json",
    "metrics_cross_family_test.json",
    "predictions_in_family_test.npz",
    "predictions_cross_family_test.npz",
    "prediction_grid.png",
    "gradient_grid.png",
    "run_log.txt",
]

TRAINED_RUN_EXTRA_FILES = ["geometry_metadata.json"]


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _run_dir(protocol_root: Path, row: dict[str, Any]) -> Path:
    return protocol_root / "runs" / str(row["transfer_id"]) / str(row["method_key"]) / f"seed_{row['seed']}" / str(row["bridge_id"])


def _required_files(row: dict[str, Any]) -> list[str]:
    if str(row.get("bridge_id", "")) == "B0":
        return list(COMMON_REQUIRED_FILES)
    return list(COMMON_REQUIRED_FILES) + list(TRAINED_RUN_EXTRA_FILES)


def _safe_float(value: Any) -> float | str:
    if value in (None, "", "nan"):
        return ""
    return float(value)


def _collect_run_record(protocol_root: Path, row: dict[str, Any]) -> dict[str, Any]:
    run_dir = _run_dir(protocol_root, row)
    config = _read_json(run_dir / "config.json")
    val_metrics = _read_json(run_dir / "metrics_val.json")
    in_family_metrics = _read_json(run_dir / "metrics_in_family_test.json")
    cross_metrics = _read_json(run_dir / "metrics_cross_family_test.json")
    actual_status = str(config.get("status", row.get("status", "MISSING_CONFIG")))
    missing_files = [name for name in _required_files(row) if not (run_dir / name).exists()]
    metric_space = (
        cross_metrics.get("metric_space")
        or val_metrics.get("metric_space")
        or in_family_metrics.get("metric_space")
        or config.get("metric_space")
        or row.get("metric_space", "")
    )
    is_real_feature = config.get("is_real_feature", "")
    if is_real_feature == "" and (run_dir / "predictions_cross_family_test.npz").exists():
        try:
            import numpy as np

            with np.load(run_dir / "predictions_cross_family_test.npz", allow_pickle=True) as payload:
                if "is_real_feature" in payload:
                    is_real_feature = bool(payload["is_real_feature"].item())
        except Exception:
            is_real_feature = ""
    return {
        "run_id": row["run_id"],
        "transfer_id": row["transfer_id"],
        "source_family": row["source_family"],
        "target_family": row["target_family"],
        "method_id": row["method_id"],
        "method_key": row["method_key"],
        "method_name": row.get("method_name", ""),
        "bridge_id": row["bridge_id"],
        "bridge_name": row["bridge_name"],
        "seed": row["seed"],
        "status": actual_status,
        "matrix_status": row.get("status", ""),
        "run_dir": str(run_dir),
        "reused_from": config.get("reused_from", row.get("reused_from", "")),
        "output_contract_ok": "True" if not missing_files else "False",
        "missing_required_files": ";".join(missing_files),
        "metric_space": metric_space,
        "is_real_feature": is_real_feature,
        "val_MAE": _safe_float(val_metrics.get("mae")),
        "val_RMSE": _safe_float(val_metrics.get("rmse")),
        "val_SSIM": _safe_float(val_metrics.get("ssim")),
        "in_family_MAE": _safe_float(in_family_metrics.get("mae")),
        "in_family_RMSE": _safe_float(in_family_metrics.get("rmse")),
        "cross_family_MAE": _safe_float(cross_metrics.get("mae")),
        "cross_family_RMSE": _safe_float(cross_metrics.get("rmse")),
        "cross_family_SSIM": _safe_float(cross_metrics.get("ssim")),
        "cross_family_gradient_error": _safe_float(cross_metrics.get("gradient_error")),
        "cross_family_edge_MAE": _safe_float(cross_metrics.get("edge_mae")),
        "skip_reason": config.get("skip_reason", row.get("skip_reason", "")),
    }


def _mean(values: list[float]) -> float | str:
    if not values:
        return ""
    return float(sum(values) / len(values))


def _as_float(value: Any) -> float | None:
    if value in (None, "", "nan"):
        return None
    return float(value)


def summarize_protocol_v14_geometry_aware_trace_bridge(*, root: str | Path) -> dict[str, Any]:
    protocol_root = Path(root)
    matrix_path = protocol_root / "protocol_v14_run_matrix.csv"
    rows = _read_csv(matrix_path)
    records = [_collect_run_record(protocol_root, row) for row in rows]

    per_run_path = protocol_root / "protocol_v14_per_run_metrics.csv"
    per_run_fields = [
        "run_id",
        "transfer_id",
        "source_family",
        "target_family",
        "method_id",
        "method_key",
        "method_name",
        "bridge_id",
        "bridge_name",
        "seed",
        "status",
        "matrix_status",
        "run_dir",
        "reused_from",
        "output_contract_ok",
        "missing_required_files",
        "metric_space",
        "is_real_feature",
        "val_MAE",
        "val_RMSE",
        "val_SSIM",
        "in_family_MAE",
        "in_family_RMSE",
        "cross_family_MAE",
        "cross_family_RMSE",
        "cross_family_SSIM",
        "cross_family_gradient_error",
        "cross_family_edge_MAE",
        "skip_reason",
    ]
    with per_run_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=per_run_fields)
        writer.writeheader()
        writer.writerows(records)

    aggregate_path = protocol_root / "protocol_v14_aggregate_metrics.csv"
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        if record["status"] != "SUCCESS":
            continue
        key = (str(record["method_id"]), str(record["bridge_id"]))
        grouped.setdefault(key, []).append(record)
    with aggregate_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method_id",
                "bridge_id",
                "success_run_count",
                "contract_ok_run_count",
                "cross_family_MAE",
                "cross_family_RMSE",
                "cross_family_SSIM",
                "cross_family_gradient_error",
                "cross_family_edge_MAE",
            ],
        )
        writer.writeheader()
        for (method_id, bridge_id), group in sorted(grouped.items()):
            writer.writerow(
                {
                    "method_id": method_id,
                    "bridge_id": bridge_id,
                    "success_run_count": len(group),
                    "contract_ok_run_count": sum(row["output_contract_ok"] == "True" for row in group),
                    "cross_family_MAE": _mean([v for row in group if (v := _as_float(row["cross_family_MAE"])) is not None]),
                    "cross_family_RMSE": _mean([v for row in group if (v := _as_float(row["cross_family_RMSE"])) is not None]),
                    "cross_family_SSIM": _mean([v for row in group if (v := _as_float(row["cross_family_SSIM"])) is not None]),
                    "cross_family_gradient_error": _mean([v for row in group if (v := _as_float(row["cross_family_gradient_error"])) is not None]),
                    "cross_family_edge_MAE": _mean([v for row in group if (v := _as_float(row["cross_family_edge_MAE"])) is not None]),
                }
            )

    unsuccessful = [row for row in records if row["status"] != "SUCCESS"]
    unsuccessful_path = protocol_root / "protocol_v14_unsuccessful_runs.csv"
    with unsuccessful_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=per_run_fields)
        writer.writeheader()
        writer.writerows(unsuccessful)

    incomplete = [row for row in records if row["output_contract_ok"] != "True"]
    incomplete_path = protocol_root / "protocol_v14_incomplete_outputs.csv"
    with incomplete_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=per_run_fields)
        writer.writeheader()
        writer.writerows(incomplete)

    summary = {
        "run_count": len(records),
        "success": sum(row["status"] == "SUCCESS" for row in records),
        "failed": sum(row["status"] == "FAILED" for row in records),
        "skipped": sum(str(row["status"]).startswith("SKIPPED") for row in records),
        "pending": sum(row["status"] == "PENDING" for row in records),
        "unsuccessful_count": len(unsuccessful),
        "incomplete_output_count": len(incomplete),
    }
    (protocol_root / "protocol_v14_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (protocol_root / "protocol_v14_generalization_evidence.md").write_text(
        "\n".join(
            [
                "# Protocol V14 Generalization Evidence",
                "",
                f"- 当前总 run 数：{summary['run_count']}",
                f"- 成功 run 数：{summary['success']}",
                f"- 未成功 run 数：{summary['unsuccessful_count']}",
                f"- 输出不完整 run 数：{summary['incomplete_output_count']}",
                "",
                "当前文件基于实际 run 目录自动汇总，仅用于方向性审计；不构成标准基准级结论。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (protocol_root / "protocol_v14_protocol_integrity_report.md").write_text(
        "\n".join(
            [
                "# Protocol V14 Protocol Integrity Report",
                "",
                f"- 成功 run：{summary['success']} / {summary['run_count']}",
                f"- 未成功 run：{summary['unsuccessful_count']}",
                f"- 输出不完整 run：{summary['incomplete_output_count']}",
                f"- 未成功结果表：`{unsuccessful_path.name}`",
                f"- 不完整输出表：`{incomplete_path.name}`",
                "",
                "CPU 小样本统一协议；结果用于检验方向性证据，不构成标准基准级结论。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    print(json.dumps(summarize_protocol_v14_geometry_aware_trace_bridge(root=args.root), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
