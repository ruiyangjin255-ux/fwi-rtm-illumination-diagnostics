from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from fwi_visionfm.scripts.build_protocol_v2_splits import build_protocol_v2_splits
from fwi_visionfm.scripts.run_protocol_v3_structure_matrix import LOSS_PRESETS, RealDINOv2Skipped, _load_manifest, _manifest_for, _run_single


def integrated_entries() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for bridge in ("raw_repeat3", "raw_spectrogram", "spectrogram_multiband", "raw_envelope_spectrum3"):
        entries.append({"model_name": "cnn_baseline", "bridge": bridge, "decoder_name": "unet_decoder", "loss_name": "default_l1"})
    for bridge in ("raw_repeat3", "raw_spectrogram", "spectrogram_multiband", "raw_envelope_spectrum3"):
        entries.append({"model_name": "vit_tiny_scratch", "bridge": bridge, "decoder_name": "unet_decoder", "loss_name": "gradient_l1"})
    for bridge in ("raw_spectrogram", "spectrogram_multiband", "raw_envelope_spectrum3"):
        entries.append({"model_name": "vit_tiny_scratch", "bridge": bridge, "decoder_name": "unet_decoder", "loss_name": "structure_loss"})
    for bridge in ("raw_spectrogram", "spectrogram_multiband", "raw_envelope_spectrum3"):
        entries.append({"model_name": "dinov2_lora_smoke", "bridge": bridge, "decoder_name": "unet_decoder", "loss_name": "default_l1", "is_probe": True})
    return entries


def _run_dir(root: Path, source: str, target: str, entry: dict[str, Any], seed: int) -> Path:
    return root / f"{source}_to_{target}" / entry["model_name"] / entry["bridge"] / entry["decoder_name"] / entry["loss_name"] / f"seed_{seed}"


def _load_matplotlib():
    import os

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _gradient_mag(image: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(np.asarray(image, dtype=np.float32))
    return np.sqrt(gx * gx + gy * gy)


def _write_triplet_grid(npz_path: Path, output_path: Path, *, gradient: bool) -> None:
    with np.load(npz_path) as payload:
        pred = np.asarray(payload["velocity_pred_physical"] if "velocity_pred_physical" in payload else payload["prediction"], dtype=np.float32)
        target = np.asarray(payload["velocity_true_physical"] if "velocity_true_physical" in payload else payload["target"], dtype=np.float32)
    if pred.ndim == 4 and pred.shape[1] == 1:
        pred = pred[:, 0]
    if target.ndim == 4 and target.shape[1] == 1:
        target = target[:, 0]
    errors = np.mean(np.abs(pred - target), axis=(1, 2))
    order = np.argsort(errors)
    picks = [int(order[0]), int(order[len(order) // 2]), int(order[-1])]
    labels = ["best", "median", "worst"]
    plt = _load_matplotlib()
    fig, axes = plt.subplots(len(picks), 3, figsize=(11.5, 3.4 * len(picks)), constrained_layout=True)
    axes = np.atleast_2d(axes)
    for row_idx, (label, index) in enumerate(zip(labels, picks)):
        p = _gradient_mag(pred[index]) if gradient else pred[index]
        t = _gradient_mag(target[index]) if gradient else target[index]
        e = np.abs(p - t).astype(np.float32)
        value_min = 0.0 if gradient else float(min(np.min(p), np.min(t)))
        value_max = float(max(np.max(p), np.max(t)))
        error_max = float(np.max(e)) if float(np.max(e)) > 0.0 else 1.0
        panels = [
            (p, f"{label} pred", "viridis", value_min, value_max),
            (t, f"{label} target", "viridis", value_min, value_max),
            (e, f"{label} abs error", "magma", 0.0, error_max),
        ]
        for col_idx, (array, title, cmap, vmin, vmax) in enumerate(panels):
            ax = axes[row_idx, col_idx]
            im = ax.imshow(array, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(title, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle("gradient diagnostics" if gradient else "velocity prediction diagnostics", fontsize=12)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_required_grids(run_dir: Path) -> None:
    npz_path = run_dir / "predictions_cross_family_test.npz"
    if not npz_path.exists():
        return
    _write_triplet_grid(npz_path, run_dir / "best_prediction_grid.png", gradient=False)
    _write_triplet_grid(npz_path, run_dir / "median_prediction_grid.png", gradient=False)
    _write_triplet_grid(npz_path, run_dir / "worst_prediction_grid.png", gradient=False)
    _write_triplet_grid(npz_path, run_dir / "best_gradient_diagnostic_grid.png", gradient=True)
    _write_triplet_grid(npz_path, run_dir / "median_gradient_diagnostic_grid.png", gradient=True)
    _write_triplet_grid(npz_path, run_dir / "worst_gradient_diagnostic_grid.png", gradient=True)


def _is_complete(run_dir: Path) -> bool:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return False
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    required = ["metrics_cross_family_test.json", "predictions_cross_family_test.npz", "train_history.csv"]
    return config.get("status") == "SUCCESS" and all((run_dir / name).exists() for name in required)


def run_integrated_visual_search(
    *,
    data_root: str | Path,
    output_root: str | Path,
    source: str,
    target: str,
    train_size: int,
    val_size: int,
    test_size: int,
    seeds: list[int],
    dinov2_seeds: list[int],
    epochs: int,
    device: str,
) -> dict[str, Any]:
    root = Path(output_root)
    all_seeds = sorted(set([int(seed) for seed in seeds] + [int(seed) for seed in dinov2_seeds]))
    build_protocol_v2_splits(
        data_root=data_root,
        output_root=root,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        seeds=all_seeds,
    )
    rows: list[dict[str, Any]] = []
    for entry in integrated_entries():
        run_seeds = dinov2_seeds if entry["model_name"].startswith("dinov2") else seeds
        for seed in run_seeds:
            run_dir = _run_dir(root, source, target, entry, int(seed))
            start = time.perf_counter()
            status = "SUCCESS"
            skip_reason = ""
            extra: dict[str, Any] = {}
            if _is_complete(run_dir):
                extra = {"metric_space": "physical_velocity"}
                status = "SUCCESS"
            else:
                run_dir.mkdir(parents=True, exist_ok=True)
                try:
                    manifest = _load_manifest(_manifest_for(root, source=source, target=target, seed=int(seed)))
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
                    _write_required_grids(run_dir)
                except RealDINOv2Skipped as exc:
                    status = "SKIPPED_REAL_DINOV2"
                    skip_reason = str(exc)
                except Exception as exc:
                    status = "FAILED"
                    skip_reason = f"{type(exc).__name__}: {exc}"
                    (run_dir / "run_log.txt").write_text(traceback.format_exc(), encoding="utf-8")
            config = {
                "protocol": "protocol_v4_integrated_bridge_visual_search",
                "source_family": source,
                "target_family": target,
                "model_name": entry["model_name"],
                "bridge": entry["bridge"],
                "decoder_name": entry["decoder_name"],
                "loss_name": entry["loss_name"],
                "loss_weights": LOSS_PRESETS[entry["loss_name"]],
                "seed": int(seed),
                "epochs": int(1 if entry["model_name"].startswith("dinov2") else epochs),
                "device": device,
                "metric_space": extra.get("metric_space", "physical_velocity"),
                "status": status,
                "skip_reason": skip_reason,
                "runtime_seconds": time.perf_counter() - start,
                "is_probe": bool(entry.get("is_probe", False)),
            }
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
            if not (run_dir / "run_log.txt").exists():
                (run_dir / "run_log.txt").write_text(f"status={status}\n", encoding="utf-8")
            rows.append(config)
    summary = {"run_count": len(rows), "runs": rows, "output_root": str(root)}
    (root / "protocol_v4_integrated_run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run integrated Protocol V4 bridge visual search.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--train-size", type=int, default=300)
    parser.add_argument("--val-size", type=int, default=100)
    parser.add_argument("--test-size", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--dinov2-seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_integrated_visual_search(
        data_root=args.data_root,
        output_root=args.output_root,
        source=args.source,
        target=args.target,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        seeds=[int(seed) for seed in args.seeds],
        dinov2_seeds=[int(seed) for seed in args.dinov2_seeds],
        epochs=args.epochs,
        device=args.device,
    )
    print(f"Wrote {Path(args.output_root) / 'protocol_v4_integrated_run_summary.json'}")
    print(f"run_count={summary['run_count']}")


if __name__ == "__main__":
    main()
