from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from fwi_visionfm.scripts.build_protocol_v2_splits import build_protocol_v2_splits
from fwi_visionfm.scripts.run_protocol_v3_structure_matrix import LOSS_PRESETS, _load_manifest, _manifest_for, _run_single


DEFAULT_BRIDGES = [
    "raw_repeat3",
    "envelope_repeat3",
    "raw_plus_envelope",
    "raw_spectrogram",
    "spectrogram_multiband",
    "raw_envelope_spectrum3",
]


def _write_prediction_grid(predictions_path: Path, output_path: Path) -> None:
    from PIL import Image, ImageDraw

    with np.load(predictions_path) as payload:
        pred = np.asarray(payload["prediction"])
        target = np.asarray(payload["target"])
    count = min(3, pred.shape[0])
    rows = []
    for index in range(count):
        error = pred[index] - target[index]
        tiles = []
        for title, image in (("prediction", pred[index]), ("target", target[index]), ("error", error)):
            image = np.asarray(image, dtype=np.float32)
            lo = float(np.nanmin(image))
            hi = float(np.nanmax(image))
            scaled = np.zeros_like(image, dtype=np.uint8) if hi == lo else np.clip((image - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
            tile = Image.fromarray(scaled, mode="L").convert("RGB")
            canvas = Image.new("RGB", (tile.width, tile.height + 18), "white")
            canvas.paste(tile, (0, 18))
            ImageDraw.Draw(canvas).text((2, 2), title, fill=(0, 0, 0))
            tiles.append(canvas)
        row = Image.new("RGB", (sum(tile.width for tile in tiles), max(tile.height for tile in tiles)), "white")
        x = 0
        for tile in tiles:
            row.paste(tile, (x, 0))
            x += tile.width
        rows.append(row)
    grid = Image.new("RGB", (max(row.width for row in rows), sum(row.height for row in rows)), "white")
    y = 0
    for row in rows:
        grid.paste(row, (0, y))
        y += row.height
    grid.save(output_path)


def run_bridge_selection(
    *,
    data_root: str | Path,
    output_root: str | Path,
    source: str,
    target: str,
    train_size: int,
    val_size: int,
    test_size: int,
    seed: int,
    epochs: int,
    device: str,
    bridge_names: list[str] | None = None,
) -> dict[str, Any]:
    root = Path(output_root)
    names = bridge_names or DEFAULT_BRIDGES
    build_protocol_v2_splits(
        data_root=data_root,
        output_root=root,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        seeds=[int(seed)],
    )
    manifest = _load_manifest(_manifest_for(root, source=source, target=target, seed=int(seed)))
    rows = []
    for bridge_name in names:
        run_dir = root / bridge_name / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        status = "SUCCESS"
        skip_reason = ""
        extra: dict[str, Any] = {}
        log_lines = [f"bridge_name={bridge_name}", f"seed={seed}"]
        try:
            extra = _run_single(
                run_dir=run_dir,
                manifest=manifest,
                model_name="cnn_baseline",
                bridge=bridge_name,
                decoder_name="simple_bounded_decoder",
                loss_name="default_l1",
                loss_weights=LOSS_PRESETS["default_l1"],
                epochs=epochs,
                batch_size=4,
                device=device,
            )
            _write_prediction_grid(run_dir / "predictions_cross_family_test.npz", run_dir / "prediction_grid_cross_family_test.png")
            log_lines.append("status=SUCCESS")
        except Exception as exc:
            status = "FAILED"
            skip_reason = f"{type(exc).__name__}: {exc}"
            log_lines.extend([f"status=FAILED", skip_reason, traceback.format_exc()])
        config = {
            "protocol": "protocol_v4_bridge_selection",
            "source_family": source,
            "target_family": target,
            "bridge_name": bridge_name,
            "bridge": bridge_name,
            "bridge_metadata": extra.get("bridge_metadata", {}),
            "model_name": "cnn_baseline",
            "decoder_name": "simple_bounded_decoder",
            "loss_name": "default_l1",
            "seed": int(seed),
            "epochs": int(epochs),
            "device": device,
            "status": status,
            "skip_reason": skip_reason,
            "runtime_seconds": time.perf_counter() - start,
        }
        (run_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
        (run_dir / "run_log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        rows.append(config)
    summary = {"run_count": len(rows), "runs": rows}
    (root / "bridge_selection_run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CPU bridge auto-selection smoke.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source", default="flatvel_a_subset2k")
    parser.add_argument("--target", default="curvevel_a_subset500")
    parser.add_argument("--train-size", type=int, default=100)
    parser.add_argument("--val-size", type=int, default=50)
    parser.add_argument("--test-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bridges", nargs="*", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_bridge_selection(
        data_root=args.data_root,
        output_root=args.output_root,
        source=args.source,
        target=args.target,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
        epochs=args.epochs,
        device=args.device,
        bridge_names=args.bridges,
    )
    print(f"Wrote {Path(args.output_root) / 'bridge_selection_run_summary.json'}")
    print(f"run_count={summary['run_count']}")


if __name__ == "__main__":
    main()
