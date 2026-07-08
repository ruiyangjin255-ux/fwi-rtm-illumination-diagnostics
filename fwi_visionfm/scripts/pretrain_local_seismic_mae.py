from __future__ import annotations

import argparse
import json
from pathlib import Path

from fwi_visionfm.scripts.build_protocol_v2_splits import build_protocol_v2_splits
from fwi_visionfm.training.mae_pretrain import run_local_mae_pretrain


def _load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest_path(root: Path, source: str, target: str, seed: int) -> Path:
    return root / "manifests" / f"{source}_to_{target}_seed{seed}_manifest.json"


def pretrain_local_mae_matrix(
    *,
    data_root: str | Path,
    output_root: str | Path,
    source: str,
    bridges: list[str],
    train_size: int,
    val_size: int,
    seed: int,
    epochs: int,
    batch_size: int,
    mask_ratio: float,
    device: str,
    mask_types: list[str] | None = None,
) -> dict:
    root = Path(output_root)
    build_protocol_v2_splits(data_root=data_root, output_root=root, train_size=train_size, val_size=val_size, test_size=val_size, seeds=[seed])
    manifest = _load_manifest(_manifest_path(root, source, "curvevel_a_subset500", int(seed)))
    rows = []
    selected_masks = list(mask_types or ["random_patch"])
    for bridge in bridges:
        for mask_type in selected_masks:
            out = root / "pretrain" / bridge / mask_type / f"seed_{seed}"
            result = run_local_mae_pretrain(
                train_paths=[row["path"] for row in manifest["train_samples"]],
                val_paths=[row["path"] for row in manifest["val_samples"]],
                output_dir=out,
                bridge=bridge,
                seed=seed,
                epochs=epochs,
                batch_size=batch_size,
                device=device,
                mask_ratio=mask_ratio,
                mask_type=mask_type,
            )
            rows.append({"bridge": bridge, "mask_type": mask_type, **result})
    summary = {"status": "SUCCESS", "rows": rows}
    (root / "pretrain_run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain local seismic MAE on OpenFWI source-family samples.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source", default="flatvel_a_subset2k")
    parser.add_argument("--bridges", nargs="+", default=["raw_envelope_spectrum3", "spectrogram_multiband", "raw_spectrogram"])
    parser.add_argument("--train-size", type=int, default=300)
    parser.add_argument("--val-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--mask-ratio", type=float, default=0.75)
    parser.add_argument("--mask-types", nargs="+", default=["random_patch"])
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = pretrain_local_mae_matrix(
        data_root=args.data_root,
        output_root=args.output_root,
        source=args.source,
        bridges=args.bridges,
        train_size=args.train_size,
        val_size=args.val_size,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        mask_ratio=args.mask_ratio,
        device=args.device,
        mask_types=args.mask_types,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
