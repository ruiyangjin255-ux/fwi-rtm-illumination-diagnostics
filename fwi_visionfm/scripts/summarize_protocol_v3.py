from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


METRICS = ["MAE", "RMSE", "SSIM", "PSNR", "gradient_error", "edge_MAE", "boundary_F1", "edge_overlap"]
FIELDNAMES = [
    "source_family",
    "target_family",
    "model_name",
    "bridge",
    "decoder_name",
    "loss_name",
    "seed",
    "metric_space",
    "val_MAE",
    "val_RMSE",
    "val_SSIM",
    "val_PSNR",
    "val_gradient_error",
    "val_edge_MAE",
    "val_boundary_F1",
    "cross_family_MAE",
    "cross_family_RMSE",
    "cross_family_SSIM",
    "cross_family_PSNR",
    "cross_family_gradient_error",
    "cross_family_edge_MAE",
    "cross_family_boundary_F1",
    "status",
    "skip_reason",
    "runtime_seconds",
]
DECODER_COMPARISON_FIELDS = [
    "source_family",
    "target_family",
    "model_name",
    "bridge",
    "loss_name",
    "seed",
    "simple_MAE",
    "unet_MAE",
    "delta_MAE",
    "simple_RMSE",
    "unet_RMSE",
    "delta_RMSE",
    "simple_SSIM",
    "unet_SSIM",
    "delta_SSIM",
    "simple_gradient_error",
    "unet_gradient_error",
    "delta_gradient_error",
    "simple_edge_MAE",
    "unet_edge_MAE",
    "delta_edge_MAE",
    "numerical_winner",
    "structural_winner",
    "tradeoff_type",
]
LOSS_COMPARISON_FIELDS = [
    "source_family",
    "target_family",
    "model_name",
    "bridge",
    "decoder_name",
    "seed",
    "default_MAE",
    "gradient_l1_MAE",
    "structure_loss_MAE",
    "default_RMSE",
    "gradient_l1_RMSE",
    "structure_loss_RMSE",
    "default_gradient_error",
    "gradient_l1_gradient_error",
    "structure_loss_gradient_error",
    "default_edge_MAE",
    "gradient_l1_edge_MAE",
    "structure_loss_edge_MAE",
    "best_loss_by_MAE",
    "best_loss_by_gradient_error",
    "best_loss_by_edge_MAE",
    "loss_tradeoff_summary",
]
TOP_CONFIG_FIELDS = [
    "rank_type",
    "source_family",
    "target_family",
    "model_name",
    "bridge",
    "decoder_name",
    "loss_name",
    "seed",
    "MAE",
    "RMSE",
    "SSIM",
    "gradient_error",
    "edge_MAE",
    "note",
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
        "edge_MAE": "edge_mae",
        "boundary_F1": "boundary_f1",
        "edge_overlap": "edge_overlap",
    }
    return metrics.get(keys[name], "")


def collect_rows(root: str | Path) -> list[dict[str, Any]]:
    rows = []
    output_root = Path(root)
    config_paths = [
        path
        for path in sorted(output_root.rglob("config.json"))
        if "manifests" not in path.parts
    ]
    if not config_paths and (output_root / "protocol_v3_summary.csv").exists():
        with (output_root / "protocol_v3_summary.csv").open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    for config_path in config_paths:
        if "manifests" in config_path.parts:
            continue
        run_dir = config_path.parent
        config = _read_json(config_path)
        if "decoder_name" not in config or "loss_name" not in config:
            continue
        val = _read_json(run_dir / "metrics_val.json")
        cross = _read_json(run_dir / "metrics_cross_family_test.json")
        row = {
            "source_family": config.get("source_family", ""),
            "target_family": config.get("target_family", ""),
            "model_name": config.get("model_name", ""),
            "bridge": config.get("bridge", ""),
            "decoder_name": config.get("decoder_name", ""),
            "loss_name": config.get("loss_name", ""),
            "seed": config.get("seed", ""),
            "metric_space": cross.get("metric_space") or config.get("metric_space", ""),
            "status": config.get("status", "UNKNOWN"),
            "skip_reason": config.get("skip_reason", ""),
            "runtime_seconds": config.get("runtime_seconds", ""),
        }
        for prefix, payload in (("val", val), ("cross_family", cross)):
            for metric in METRICS:
                if metric == "edge_overlap":
                    continue
                row[f"{prefix}_{metric}"] = _metric(payload, metric)
        rows.append(row)
    return rows


def _float(value: Any) -> float | None:
    try:
        if value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _delta(left: Any, right: Any) -> Any:
    left_value = _float(left)
    right_value = _float(right)
    if left_value is None or right_value is None:
        return ""
    return left_value - right_value


def _winner_lower(left_name: str, left: Any, right_name: str, right: Any) -> str:
    left_value = _float(left)
    right_value = _float(right)
    if left_value is None or right_value is None:
        return "mixed"
    if left_value < right_value:
        return left_name
    if right_value < left_value:
        return right_name
    return "mixed"


def build_decoder_comparison(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str, str], dict[str, dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "SUCCESS":
            continue
        key = (
            row.get("source_family", ""),
            row.get("target_family", ""),
            row.get("model_name", ""),
            row.get("bridge", ""),
            row.get("loss_name", ""),
            str(row.get("seed", "")),
        )
        grouped.setdefault(key, {})[row.get("decoder_name", "")] = row
    out: list[dict[str, Any]] = []
    for (source, target, model, bridge, loss, seed), decoders in sorted(grouped.items()):
        simple = decoders.get("simple_bounded_decoder")
        unet = decoders.get("unet_decoder")
        if simple is None or unet is None:
            continue
        delta_mae = _delta(unet.get("cross_family_MAE"), simple.get("cross_family_MAE"))
        delta_rmse = _delta(unet.get("cross_family_RMSE"), simple.get("cross_family_RMSE"))
        delta_ssim = _delta(unet.get("cross_family_SSIM"), simple.get("cross_family_SSIM"))
        delta_grad = _delta(unet.get("cross_family_gradient_error"), simple.get("cross_family_gradient_error"))
        delta_edge = _delta(unet.get("cross_family_edge_MAE"), simple.get("cross_family_edge_MAE"))
        if _float(delta_mae) is not None and _float(delta_rmse) is not None and float(delta_mae) < 0 and float(delta_rmse) < 0:
            numerical_winner = "unet_decoder"
        elif _float(delta_mae) is not None and _float(delta_rmse) is not None and float(delta_mae) > 0 and float(delta_rmse) > 0:
            numerical_winner = "simple_bounded_decoder"
        else:
            numerical_winner = "mixed"
        if _float(delta_grad) is not None and _float(delta_edge) is not None and float(delta_grad) < 0 and float(delta_edge) < 0:
            structural_winner = "unet_decoder"
        elif _float(delta_grad) is not None and _float(delta_edge) is not None and float(delta_grad) > 0 and float(delta_edge) > 0:
            structural_winner = "simple_bounded_decoder"
        else:
            structural_winner = "mixed"
        if numerical_winner == "simple_bounded_decoder" and structural_winner == "unet_decoder":
            tradeoff = "simple numerical advantage vs unet structural advantage"
        elif numerical_winner == "unet_decoder" and structural_winner == "unet_decoder":
            tradeoff = "unet dominates"
        elif numerical_winner == "simple_bounded_decoder" and structural_winner == "simple_bounded_decoder":
            tradeoff = "simple dominates"
        else:
            tradeoff = "mixed"
        out.append(
            {
                "source_family": source,
                "target_family": target,
                "model_name": model,
                "bridge": bridge,
                "loss_name": loss,
                "seed": seed,
                "simple_MAE": simple.get("cross_family_MAE", ""),
                "unet_MAE": unet.get("cross_family_MAE", ""),
                "delta_MAE": delta_mae,
                "simple_RMSE": simple.get("cross_family_RMSE", ""),
                "unet_RMSE": unet.get("cross_family_RMSE", ""),
                "delta_RMSE": delta_rmse,
                "simple_SSIM": simple.get("cross_family_SSIM", ""),
                "unet_SSIM": unet.get("cross_family_SSIM", ""),
                "delta_SSIM": delta_ssim,
                "simple_gradient_error": simple.get("cross_family_gradient_error", ""),
                "unet_gradient_error": unet.get("cross_family_gradient_error", ""),
                "delta_gradient_error": delta_grad,
                "simple_edge_MAE": simple.get("cross_family_edge_MAE", ""),
                "unet_edge_MAE": unet.get("cross_family_edge_MAE", ""),
                "delta_edge_MAE": delta_edge,
                "numerical_winner": numerical_winner,
                "structural_winner": structural_winner,
                "tradeoff_type": tradeoff,
            }
        )
    return out


def build_loss_comparison(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str, str], dict[str, dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "SUCCESS":
            continue
        key = (
            row.get("source_family", ""),
            row.get("target_family", ""),
            row.get("model_name", ""),
            row.get("bridge", ""),
            row.get("decoder_name", ""),
            str(row.get("seed", "")),
        )
        grouped.setdefault(key, {})[row.get("loss_name", "")] = row
    out: list[dict[str, Any]] = []
    for (source, target, model, bridge, decoder, seed), losses in sorted(grouped.items()):
        if not {"default_l1", "gradient_l1", "structure_loss"}.issubset(losses):
            continue
        default = losses["default_l1"]
        gradient = losses["gradient_l1"]
        structure = losses["structure_loss"]
        candidates = {
            "default_l1": default,
            "gradient_l1": gradient,
            "structure_loss": structure,
        }
        best_mae = min(candidates, key=lambda name: _float(candidates[name].get("cross_family_MAE")) or float("inf"))
        best_grad = min(candidates, key=lambda name: _float(candidates[name].get("cross_family_gradient_error")) or float("inf"))
        best_edge = min(candidates, key=lambda name: _float(candidates[name].get("cross_family_edge_MAE")) or float("inf"))
        structure_mae = _float(structure.get("cross_family_MAE"))
        structure_rmse = _float(structure.get("cross_family_RMSE"))
        structure_grad = _float(structure.get("cross_family_gradient_error"))
        default_mae = _float(default.get("cross_family_MAE"))
        default_rmse = _float(default.get("cross_family_RMSE"))
        default_grad = _float(default.get("cross_family_gradient_error"))
        if (
            structure_grad is not None
            and default_grad is not None
            and structure_grad < default_grad
            and structure_mae is not None
            and default_mae is not None
            and structure_rmse is not None
            and default_rmse is not None
            and structure_mae > default_mae
            and structure_rmse > default_rmse
        ):
            summary = "structure loss improves gradient metrics with numerical tradeoff"
        elif (
            structure_grad is not None
            and default_grad is not None
            and structure_grad < default_grad
            and structure_mae is not None
            and default_mae is not None
            and structure_rmse is not None
            and default_rmse is not None
            and structure_mae < default_mae
            and structure_rmse < default_rmse
        ):
            summary = "structure loss improves both numerical and structural metrics"
        else:
            summary = "no clear loss advantage"
        out.append(
            {
                "source_family": source,
                "target_family": target,
                "model_name": model,
                "bridge": bridge,
                "decoder_name": decoder,
                "seed": seed,
                "default_MAE": default.get("cross_family_MAE", ""),
                "gradient_l1_MAE": gradient.get("cross_family_MAE", ""),
                "structure_loss_MAE": structure.get("cross_family_MAE", ""),
                "default_RMSE": default.get("cross_family_RMSE", ""),
                "gradient_l1_RMSE": gradient.get("cross_family_RMSE", ""),
                "structure_loss_RMSE": structure.get("cross_family_RMSE", ""),
                "default_gradient_error": default.get("cross_family_gradient_error", ""),
                "gradient_l1_gradient_error": gradient.get("cross_family_gradient_error", ""),
                "structure_loss_gradient_error": structure.get("cross_family_gradient_error", ""),
                "default_edge_MAE": default.get("cross_family_edge_MAE", ""),
                "gradient_l1_edge_MAE": gradient.get("cross_family_edge_MAE", ""),
                "structure_loss_edge_MAE": structure.get("cross_family_edge_MAE", ""),
                "best_loss_by_MAE": best_mae,
                "best_loss_by_gradient_error": best_grad,
                "best_loss_by_edge_MAE": best_edge,
                "loss_tradeoff_summary": summary,
            }
        )
    return out


def _top_row(rank_type: str, row: dict[str, Any], note: str) -> dict[str, Any]:
    return {
        "rank_type": rank_type,
        "source_family": row.get("source_family", ""),
        "target_family": row.get("target_family", ""),
        "model_name": row.get("model_name", ""),
        "bridge": row.get("bridge", ""),
        "decoder_name": row.get("decoder_name", ""),
        "loss_name": row.get("loss_name", ""),
        "seed": row.get("seed", ""),
        "MAE": row.get("cross_family_MAE", ""),
        "RMSE": row.get("cross_family_RMSE", ""),
        "SSIM": row.get("cross_family_SSIM", ""),
        "gradient_error": row.get("cross_family_gradient_error", ""),
        "edge_MAE": row.get("cross_family_edge_MAE", ""),
        "note": note,
    }


def _dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    metrics = ["cross_family_MAE", "cross_family_RMSE", "cross_family_gradient_error", "cross_family_edge_MAE"]
    left_values = [_float(left.get(metric)) for metric in metrics]
    right_values = [_float(right.get(metric)) for metric in metrics]
    if any(value is None for value in left_values + right_values):
        return False
    no_worse = all(float(lv) <= float(rv) for lv, rv in zip(left_values, right_values))
    strictly_better = any(float(lv) < float(rv) for lv, rv in zip(left_values, right_values))
    return bool(no_worse and strictly_better)


def build_top_configs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    successful = [row for row in rows if row.get("status") == "SUCCESS"]
    out: list[dict[str, Any]] = []
    sort_map = {
        "top_3_by_MAE": "cross_family_MAE",
        "top_3_by_RMSE": "cross_family_RMSE",
        "top_3_by_gradient_error": "cross_family_gradient_error",
        "top_3_by_edge_MAE": "cross_family_edge_MAE",
    }
    for rank_type, field in sort_map.items():
        ranked = sorted(successful, key=lambda row: _float(row.get(field)) if _float(row.get(field)) is not None else float("inf"))
        for index, row in enumerate(ranked[:3], start=1):
            out.append(_top_row(rank_type, row, f"rank {index} by {field.replace('cross_family_', '')}"))
    for row in successful:
        if not any(other is not row and _dominates(other, row) for other in successful):
            out.append(_top_row("pareto_candidates", row, "pareto candidate"))
    return out


def write_summary(root: str | Path) -> dict[str, Path]:
    output_root = Path(root)
    rows = collect_rows(output_root)
    summary_path = output_root / "protocol_v3_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    decoder_path = output_root / "protocol_v3_decoder_comparison.csv"
    with decoder_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DECODER_COMPARISON_FIELDS)
        writer.writeheader()
        writer.writerows(build_decoder_comparison(rows))
    loss_path = output_root / "protocol_v3_loss_comparison.csv"
    with loss_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOSS_COMPARISON_FIELDS)
        writer.writeheader()
        writer.writerows(build_loss_comparison(rows))
    top_path = output_root / "protocol_v3_top_configs.csv"
    with top_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TOP_CONFIG_FIELDS)
        writer.writeheader()
        writer.writerows(build_top_configs(rows))
    return {"summary": summary_path, "decoder_comparison": decoder_path, "loss_comparison": loss_path, "top_configs": top_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Protocol V3 results.")
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    paths = write_summary(parse_args().root)
    print(f"Wrote {paths['summary']}")
    print(f"Wrote {paths['decoder_comparison']}")
    print(f"Wrote {paths['loss_comparison']}")
    print(f"Wrote {paths['top_configs']}")


if __name__ == "__main__":
    main()
