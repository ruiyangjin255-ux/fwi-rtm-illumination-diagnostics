from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from fwi_visionfm.scripts.summarize_protocol_v3 import collect_rows, write_summary


VISUAL_FIELDS = [
    "source_family",
    "target_family",
    "model_name",
    "bridge",
    "decoder_name",
    "loss_name",
    "seed",
    "metric_space",
    "MAE",
    "RMSE",
    "SSIM",
    "gradient_error",
    "edge_MAE",
    "visual_score",
    "status",
    "skip_reason",
]
BEST_FIELDS = ["selection_type", *VISUAL_FIELDS]
ERROR_METRICS = ["MAE", "RMSE", "gradient_error", "edge_MAE"]


def _float(value: Any) -> float | None:
    try:
        if value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _metric(row: dict[str, Any], name: str) -> float | None:
    return _float(row.get(name) or row.get(f"cross_family_{name}"))


def _normalize(values: list[float], value: float, *, higher_is_better: bool) -> float:
    lo = min(values)
    hi = max(values)
    if hi == lo:
        return 1.0
    scaled = (value - lo) / (hi - lo)
    return scaled if higher_is_better else 1.0 - scaled


def _visual_score(row: dict[str, Any], successful: list[dict[str, Any]]) -> float:
    components: list[float] = []
    for metric in ERROR_METRICS:
        values = [_metric(candidate, metric) for candidate in successful]
        usable = [float(value) for value in values if value is not None]
        value = _metric(row, metric)
        if usable and value is not None:
            components.append(_normalize(usable, float(value), higher_is_better=False))
    ssim_values = [_metric(candidate, "SSIM") for candidate in successful]
    usable_ssim = [float(value) for value in ssim_values if value is not None]
    ssim = _metric(row, "SSIM")
    if usable_ssim and ssim is not None:
        components.append(_normalize(usable_ssim, float(ssim), higher_is_better=True))
    if not components:
        return 0.0
    return float(sum(components) / len(components))


def _visual_row(row: dict[str, Any], successful: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "source_family": row.get("source_family", ""),
        "target_family": row.get("target_family", ""),
        "model_name": row.get("model_name", ""),
        "bridge": row.get("bridge", ""),
        "decoder_name": row.get("decoder_name", ""),
        "loss_name": row.get("loss_name", ""),
        "seed": row.get("seed", ""),
        "metric_space": row.get("metric_space", ""),
        "MAE": row.get("cross_family_MAE", ""),
        "RMSE": row.get("cross_family_RMSE", ""),
        "SSIM": row.get("cross_family_SSIM", ""),
        "gradient_error": row.get("cross_family_gradient_error", ""),
        "edge_MAE": row.get("cross_family_edge_MAE", ""),
        "visual_score": f"{_visual_score(row, successful):.6f}",
        "status": row.get("status", ""),
        "skip_reason": row.get("skip_reason", ""),
    }


def build_visual_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    successful = [row for row in rows if row.get("status") == "SUCCESS"]
    return [_visual_row(row, successful) for row in successful]


def _best(rows: list[dict[str, Any]], field: str, *, higher_is_better: bool = False) -> dict[str, Any]:
    return sorted(
        rows,
        key=lambda row: _float(row.get(field)) if _float(row.get(field)) is not None else (-float("inf") if higher_is_better else float("inf")),
        reverse=higher_is_better,
    )[0]


def _dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    checks = []
    for metric in ERROR_METRICS:
        left_value = _float(left.get(metric))
        right_value = _float(right.get(metric))
        if left_value is None or right_value is None:
            return False
        checks.append((left_value, right_value, left_value <= right_value, left_value < right_value))
    left_ssim = _float(left.get("SSIM"))
    right_ssim = _float(right.get("SSIM"))
    if left_ssim is None or right_ssim is None:
        return False
    checks.append((left_ssim, right_ssim, left_ssim >= right_ssim, left_ssim > right_ssim))
    return all(check[2] for check in checks) and any(check[3] for check in checks)


def build_best_models(visual_rows: list[dict[str, Any]]) -> dict[str, Any]:
    best = {
        "best_by_MAE": _best(visual_rows, "MAE"),
        "best_by_RMSE": _best(visual_rows, "RMSE"),
        "best_by_SSIM": _best(visual_rows, "SSIM", higher_is_better=True),
        "best_by_gradient_error": _best(visual_rows, "gradient_error"),
        "best_by_edge_MAE": _best(visual_rows, "edge_MAE"),
        "best_by_visual_score": _best(visual_rows, "visual_score", higher_is_better=True),
    }
    best["pareto_candidates"] = [
        row for row in visual_rows if not any(other is not row and _dominates(other, row) for other in visual_rows)
    ]
    return best


def _selection_rows(best: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, value in best.items():
        if key == "pareto_candidates":
            for row in value:
                rows.append({"selection_type": key, **row})
        else:
            rows.append({"selection_type": key, **value})
    return rows


def _config_label(row: dict[str, Any]) -> str:
    return (
        f"{row.get('model_name')} / {row.get('bridge')} / {row.get('decoder_name')} / "
        f"{row.get('loss_name')} / seed {row.get('seed')}"
    )


def _has_tradeoff(rows: list[dict[str, Any]]) -> bool:
    grouped: dict[tuple[str, str, str, str, str, str], dict[str, dict[str, Any]]] = {}
    for row in rows:
        if "dinov2" in row.get("model_name", "").lower():
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
    for decoders in grouped.values():
        simple = decoders.get("simple_bounded_decoder")
        unet = decoders.get("unet_decoder")
        if simple is None or unet is None:
            continue
        simple_mae = _float(simple.get("MAE"))
        simple_rmse = _float(simple.get("RMSE"))
        simple_grad = _float(simple.get("gradient_error"))
        simple_edge = _float(simple.get("edge_MAE"))
        unet_mae = _float(unet.get("MAE"))
        unet_rmse = _float(unet.get("RMSE"))
        unet_grad = _float(unet.get("gradient_error"))
        unet_edge = _float(unet.get("edge_MAE"))
        if None in {simple_mae, simple_rmse, simple_grad, simple_edge, unet_mae, unet_rmse, unet_grad, unet_edge}:
            continue
        if simple_mae < unet_mae and simple_rmse < unet_rmse and unet_grad < simple_grad and unet_edge < simple_edge:
            return True
    return False


def write_visual_report(root: Path, best: dict[str, Any], visual_rows: list[dict[str, Any]]) -> Path:
    dino_rows = [row for row in visual_rows if "dinov2" in row.get("model_name", "").lower()]
    report_path = root / "protocol_v4_visual_report.md"
    lines = [
        "# Protocol V4 Visual-quality-driven Model Selection",
        "",
        "Protocol V4 shifts the objective from metric-only comparison to visual-quality-driven model selection. Under CPU-limited small-sample settings, it identifies the best available backbone-bridge-decoder-loss combinations for final velocity-map quality, while keeping benchmark-level VisionFM claims conservative.",
        "",
        "## Selection Summary",
        f"- 数值 MAE 最好: {_config_label(best['best_by_MAE'])}, MAE={best['best_by_MAE'].get('MAE')}",
        f"- 数值 RMSE 最好: {_config_label(best['best_by_RMSE'])}, RMSE={best['best_by_RMSE'].get('RMSE')}",
        f"- SSIM 最好: {_config_label(best['best_by_SSIM'])}, SSIM={best['best_by_SSIM'].get('SSIM')}",
        f"- 结构 gradient_error 最好: {_config_label(best['best_by_gradient_error'])}, gradient_error={best['best_by_gradient_error'].get('gradient_error')}",
        f"- 结构 edge_MAE 最好: {_config_label(best['best_by_edge_MAE'])}, edge_MAE={best['best_by_edge_MAE'].get('edge_MAE')}",
        f"- 综合 visual_score 最好: {_config_label(best['best_by_visual_score'])}, visual_score={best['best_by_visual_score'].get('visual_score')}",
        "",
        "## Decoder Trade-off",
        "simple decoder 与 U-Net decoder 的 trade-off 仍存在。" if _has_tradeoff(visual_rows) else "simple decoder 与 U-Net decoder 的 trade-off 在当前可用行中不明显。",
        "",
        "## DINOv2-LoRA Probe",
        f"DINOv2-LoRA remains a limited-seed probe: available rows={len(dino_rows)}. It is not benchmark evidence.",
        "",
        "## Visual Quality Limitation",
        "当前成图仍处于 CPU-limited small-sample validation 阶段，不能视为已经达到复杂 FWI 应用级别。",
        "",
        "## Pareto Candidates",
    ]
    for row in best["pareto_candidates"]:
        lines.append(f"- {_config_label(row)}: MAE={row.get('MAE')}, RMSE={row.get('RMSE')}, SSIM={row.get('SSIM')}, gradient_error={row.get('gradient_error')}, edge_MAE={row.get('edge_MAE')}, visual_score={row.get('visual_score')}")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def run_visual_selection(root: str | Path) -> dict[str, Path]:
    output_root = Path(root)
    if not (output_root / "protocol_v3_summary.csv").exists():
        write_summary(output_root)
    rows = collect_rows(output_root)
    visual_rows = build_visual_summary(rows)
    best = build_best_models(visual_rows)

    visual_path = output_root / "protocol_v4_visual_summary.csv"
    with visual_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=VISUAL_FIELDS)
        writer.writeheader()
        writer.writerows(visual_rows)

    best_csv_path = output_root / "protocol_v4_best_models.csv"
    with best_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=BEST_FIELDS)
        writer.writeheader()
        writer.writerows(_selection_rows(best))

    best_json_path = output_root / "protocol_v4_best_models.json"
    best_json_path.write_text(json.dumps(best, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path = write_visual_report(output_root, best, visual_rows)
    return {
        "visual_summary": visual_path,
        "best_models_csv": best_csv_path,
        "best_models_json": best_json_path,
        "report": report_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select best Protocol V4 visual-quality model from Protocol V3 outputs.")
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    paths = run_visual_selection(parse_args().root)
    for path in paths.values():
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
