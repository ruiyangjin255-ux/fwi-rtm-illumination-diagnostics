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


def _load_npz_meta(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=True) as payload:
        metadata = json.loads(str(payload["metadata_json"].item())) if "metadata_json" in payload else {}
        metadata["sample_id_count"] = int(len(payload["sample_id"])) if "sample_id" in payload else int(metadata.get("sample_id_count", 0))
        metadata["feature_shape"] = payload["feature_shape"].tolist() if "feature_shape" in payload else metadata.get("feature_shape", [])
        metadata["status"] = str(payload["status"].item()) if "status" in payload else metadata.get("status", "")
        metadata["is_real_feature"] = bool(payload["is_real_feature"].item()) if "is_real_feature" in payload else bool(metadata.get("is_real_feature", False))
        return metadata


def _write_claims(path: Path) -> None:
    lines = [
        "# Protocol V9 NCS Adapter Claims And Limitations",
        "",
        "## Can Claim",
        "- NCS 2D adapter has been repaired if load/forward/cache succeed。",
        "- Real NCS 2D frozen feature cache can be generated if is_real_feature=True。",
        "- Decoder-only probe can run on real NCS 2D frozen features if probe succeeds。",
        "",
        "## Cannot Claim",
        "- NCS improves FWI。",
        "- NCS outperforms MAE / CNN / Local MAE。",
        "- NCS improves FWI generalization。",
        "- NCS 2.5D result is available if adapter remains pending。",
        "- decoder-only result is benchmark-level proof。",
        "- fallback feature is a real NCS result。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_protocol_v9_ncs_adapter_repair_report(
    *,
    root: str | Path,
    previous_v9_report: str | Path | None,
    output_dir: str | Path,
) -> dict[str, Path]:
    root = Path(root)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    availability = _load_json(root / "availability_report.json", {"models": []})
    previous_text = Path(previous_v9_report).read_text(encoding="utf-8") if previous_v9_report and Path(previous_v9_report).exists() else ""
    cache_dir = root / "feature_cache" / "ncs_2d"
    cache_config = _load_json(cache_dir / "cache_config.json", {})
    train_meta = _load_npz_meta(cache_dir / "train_features.npz")
    val_meta = _load_npz_meta(cache_dir / "val_features.npz")
    cross_meta = _load_npz_meta(cache_dir / "cross_family_test_features.npz")
    adapter_status = _load_json(root / "adapter_status_report.json", {})

    probe_dir = root / "decoder_probe" / "ncs_2d"
    probe_config = _load_json(probe_dir / "config.json", {})
    val_metrics = _load_json(probe_dir / "metrics_val.json", {})
    cross_metrics = _load_json(probe_dir / "metrics_cross_family_test.json", {})

    summary_path = out / "protocol_v9_ncs_adapter_repair_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["section", "name", "status", "detail"])
        writer.writeheader()
        for model in availability.get("models", []):
            writer.writerow({"section": "availability", "name": model.get("name", ""), "status": model.get("status", ""), "detail": model.get("message", "")})
        writer.writerow({"section": "feature_cache", "name": "ncs_2d", "status": cache_config.get("status", ""), "detail": f"is_real_feature={cache_config.get('is_real_feature')} feature_shape={cache_config.get('feature_shape')}"})
        writer.writerow({"section": "decoder_probe", "name": "ncs_2d", "status": probe_config.get("status", "SKIPPED"), "detail": f"metric_space={cross_metrics.get('metric_space', '')}"})

    claims_path = out / "protocol_v9_ncs_adapter_claims_and_limitations.md"
    _write_claims(claims_path)

    lines = [
        "# Protocol V9 NCS Adapter Repair and Real Probe Report",
        "",
        "## 1. Goal",
        "本轮目标是修复 NCS adapter，并验证真实 NCS frozen feature 是否能进入 decoder-only 链路。",
        "",
        "## 2. Previous V9 Status",
        "- ncs_2d previously IMPORT_ERROR。",
        "- ncs_2p5d previously WEIGHTS_PRESENT_ADAPTER_PENDING。",
        "- vit_mae_base previously AVAILABLE and probed。",
        "- previous vit_mae_base probe is not benchmark-level proof。",
    ]
    if previous_text:
        if "IMPORT_ERROR" in previous_text:
            lines.append("- previous report confirmed IMPORT_ERROR before adapter repair.")
        if "not benchmark-level proof" in previous_text:
            lines.append("- previous report already marked the probe as not benchmark-level proof.")
    lines.extend(
        [
            "",
            "## 3. Adapter Repair",
        ]
    )
    for model in availability.get("models", []):
        lines.append(
            f"- {model.get('name')}: status={model.get('status')}, load_status={model.get('load_status', '')}, "
            f"forward_status={model.get('forward_status', '')}, adapter_status={model.get('adapter_status', '')}, "
            f"message={model.get('message', '')}"
        )
    if adapter_status:
        lines.append(f"- ncs_2p5d pending reason: {adapter_status.get('pending_reason', '')}")
    lines.extend(
        [
            "",
            "## 4. Feature Cache",
            f"- backbone_name: {cache_config.get('backbone_name', 'ncs_2d')}",
            f"- is_real_feature: {cache_config.get('is_real_feature')}",
            f"- feature_shape: {cache_config.get('feature_shape', train_meta.get('feature_shape', []))}",
            f"- sample_id_count(train/val/cross): {train_meta.get('sample_id_count', 0)}/{val_meta.get('sample_id_count', 0)}/{cross_meta.get('sample_id_count', 0)}",
            f"- status: {cache_config.get('status', '')}",
            "",
            "## 5. Decoder-only Probe",
        ]
    )
    if probe_config:
        lines.append(f"- status: {probe_config.get('status', 'SKIPPED')}")
        if val_metrics:
            lines.append(
                f"- val metrics: MAE={val_metrics.get('mae', '')}, RMSE={val_metrics.get('rmse', '')}, SSIM={val_metrics.get('ssim', '')}, "
                f"gradient_error={val_metrics.get('gradient_error', '')}, edge_MAE={val_metrics.get('edge_mae', '')}, metric_space={val_metrics.get('metric_space', '')}"
            )
        if cross_metrics:
            lines.append(
                f"- cross-family metrics: MAE={cross_metrics.get('mae', '')}, RMSE={cross_metrics.get('rmse', '')}, SSIM={cross_metrics.get('ssim', '')}, "
                f"gradient_error={cross_metrics.get('gradient_error', '')}, edge_MAE={cross_metrics.get('edge_mae', '')}, metric_space={cross_metrics.get('metric_space', '')}"
            )
        lines.append("- This remains only decoder-only frozen feature probe evidence, not benchmark-level proof.")
    else:
        lines.append("- FAILED or SKIPPED: no completed decoder-only probe output was found.")
    lines.extend(
        [
            "",
            "## 6. Comparison Note",
            "- vit_mae_base previous decoder-only metrics may be used only as context.",
            "- cross-model comparison is not benchmark-level proof.",
            "- training remains small-sample, seed=0, decoder-only.",
            "- no conclusion on NCS superiority.",
            "",
            "## 7. Limitations",
            "- CPU-only",
            "- train_size=100 / val_size=50 / test_size=50",
            "- decoder-only",
            "- frozen feature only",
            "- no full fine-tuning",
            "- no benchmark-level proof",
            "- NCS 2.5D may still be adapter pending",
            "- OpenFWI shot gather is not the same input domain as migrated seismic cubes used by NCS pretraining。",
            "",
            "## 8. Next Step",
        ]
    )
    if cache_config.get("is_real_feature") and probe_config.get("status") == "SUCCESS":
        lines.append("- 进入 seed=0/1/2 decoder-only stability，再与 vit_mae_base、cnn_baseline、boundary_aux 做 selected comparison。")
    else:
        lines.append("- 保留 adapter failure report，并优先解决 NCS repo API mapping。")
    if not cache_config.get("is_real_feature"):
        lines.append("- 不把 fallback feature 当作真实 NCS feature。")
    if adapter_status.get("status") == "WEIGHTS_PRESENT_ADAPTER_PENDING":
        lines.append("- 单独做 2.5D pseudo-view adapter，不混入当前结论。")
    report_path = out / "protocol_v9_ncs_adapter_repair_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"report_path": report_path, "summary_path": summary_path, "claims_path": claims_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write Protocol V9 NCS adapter repair report.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--previous-v9-report", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = write_protocol_v9_ncs_adapter_repair_report(
        root=args.root,
        previous_v9_report=args.previous_v9_report,
        output_dir=args.output_dir,
    )
    print(json.dumps({key: str(value) for key, value in payload.items()}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
