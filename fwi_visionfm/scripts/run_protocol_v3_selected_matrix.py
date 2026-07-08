from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Any

from fwi_visionfm.scripts.build_protocol_v2_splits import build_protocol_v2_splits
from fwi_visionfm.scripts.run_protocol_v3_structure_matrix import (
    LOSS_PRESETS,
    RealDINOv2Skipped,
    _load_manifest,
    _manifest_for,
    _run_single,
)


def selected_entries() -> list[dict[str, Any]]:
    return [
        {
            "model_name": "cnn_baseline",
            "bridge": "raw_repeat3",
            "decoder_name": "simple_bounded_decoder",
            "loss_name": "default_l1",
        },
        {
            "model_name": "vit_tiny_scratch",
            "bridge": "raw_repeat3",
            "decoder_name": "simple_bounded_decoder",
            "loss_name": "default_l1",
        },
        {
            "model_name": "cnn_baseline",
            "bridge": "raw_repeat3",
            "decoder_name": "unet_decoder",
            "loss_name": "default_l1",
        },
        {
            "model_name": "vit_tiny_scratch",
            "bridge": "raw_spectrogram",
            "decoder_name": "unet_decoder",
            "loss_name": "default_l1",
        },
        {
            "model_name": "vit_tiny_scratch",
            "bridge": "raw_spectrogram",
            "decoder_name": "unet_decoder",
            "loss_name": "gradient_l1",
        },
        {
            "model_name": "vit_tiny_scratch",
            "bridge": "raw_spectrogram",
            "decoder_name": "unet_decoder",
            "loss_name": "structure_loss",
        },
        {
            "model_name": "dinov2_lora_smoke",
            "bridge": "raw_spectrogram",
            "decoder_name": "simple_bounded_decoder",
            "loss_name": "default_l1",
            "is_probe": True,
        },
        {
            "model_name": "dinov2_lora_smoke",
            "bridge": "raw_spectrogram",
            "decoder_name": "unet_decoder",
            "loss_name": "default_l1",
            "is_probe": True,
        },
    ]


def _run_dir(root: Path, source: str, target: str, entry: dict[str, Any], seed: int) -> Path:
    return (
        root
        / f"{source}_to_{target}"
        / entry["model_name"]
        / entry["bridge"]
        / entry["decoder_name"]
        / entry["loss_name"]
        / f"seed_{seed}"
    )


def _write_run_log(run_dir: Path, lines: list[str]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_log.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_protocol_v3_selected_matrix(
    *,
    data_root: str | Path,
    output_root: str | Path,
    source: str,
    target: str,
    seeds: list[int],
    train_size: int,
    val_size: int,
    test_size: int,
    epochs: int,
    device: str,
    dino_seeds: list[int] | None = None,
) -> dict[str, Any]:
    root = Path(output_root)
    build_protocol_v2_splits(
        data_root=data_root,
        output_root=root,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        seeds=seeds,
    )
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        manifest = _load_manifest(_manifest_for(root, source=source, target=target, seed=int(seed)))
        for entry in selected_entries():
            if entry["model_name"].startswith("dinov2") and dino_seeds is not None and int(seed) not in set(dino_seeds):
                continue
            run_dir = _run_dir(root, source, target, entry, int(seed))
            run_dir.mkdir(parents=True, exist_ok=True)
            start = time.perf_counter()
            status = "SUCCESS"
            skip_reason = ""
            extra: dict[str, Any] = {}
            log_lines = [
                "Protocol V3 selected multi-seed run",
                f"source={source}",
                f"target={target}",
                f"seed={seed}",
                f"model_name={entry['model_name']}",
                f"bridge={entry['bridge']}",
                f"decoder_name={entry['decoder_name']}",
                f"loss_name={entry['loss_name']}",
            ]
            try:
                extra = _run_single(
                    run_dir=run_dir,
                    manifest=manifest,
                    model_name=entry["model_name"],
                    bridge=entry["bridge"],
                    decoder_name=entry["decoder_name"],
                    loss_name=entry["loss_name"],
                    loss_weights=LOSS_PRESETS[entry["loss_name"]],
                    epochs=1 if entry["model_name"].startswith("dinov2") else epochs,
                    batch_size=4,
                    device=device,
                )
                status = str(extra.get("status", "SUCCESS"))
                log_lines.append(f"status={status}")
            except RealDINOv2Skipped as exc:
                status = "SKIPPED_REAL_DINOV2"
                skip_reason = str(exc)
                log_lines.extend([f"status={status}", f"skip_reason={skip_reason}"])
            except Exception as exc:
                status = "FAILED"
                skip_reason = f"{type(exc).__name__}: {exc}"
                log_lines.extend([f"status={status}", f"exception={skip_reason}", traceback.format_exc()])
            config = {
                "protocol": "protocol_v3_selected_multiseed_validation",
                "source_family": source,
                "target_family": target,
                "model_name": entry["model_name"],
                "bridge": entry["bridge"],
                "decoder_name": entry["decoder_name"],
                "loss_name": entry["loss_name"],
                "loss_weights": LOSS_PRESETS[entry["loss_name"]],
                "seed": int(seed),
                "epochs": int(1 if entry["model_name"].startswith("dinov2") else epochs),
                "batch_size": 4,
                "device": device,
                "metric_space": extra.get("metric_space", "physical_velocity"),
                "status": status,
                "skip_reason": skip_reason,
                "runtime_seconds": time.perf_counter() - start,
                "is_probe": bool(entry.get("is_probe") or extra.get("is_probe", False)),
            }
            (run_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
            _write_run_log(run_dir, log_lines)
            rows.append(config)
    summary = {"run_count": len(rows), "runs": rows, "output_root": str(root)}
    (root / "matrix_run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run selected Protocol V3 multi-seed validation matrix.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--train-size", type=int, default=300)
    parser.add_argument("--val-size", type=int, default=100)
    parser.add_argument("--test-size", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dino-seeds", type=int, nargs="*", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_protocol_v3_selected_matrix(
        data_root=args.data_root,
        output_root=args.output_root,
        source=args.source,
        target=args.target,
        seeds=[int(seed) for seed in args.seeds],
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        epochs=args.epochs,
        device=args.device,
        dino_seeds=args.dino_seeds,
    )
    print(f"Wrote matrix summary: {Path(args.output_root) / 'matrix_run_summary.json'}")
    print(f"run_count={summary['run_count']}")


if __name__ == "__main__":
    main()
