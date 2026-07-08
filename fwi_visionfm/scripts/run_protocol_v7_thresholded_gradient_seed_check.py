from __future__ import annotations

import argparse
import csv
import json
import traceback
from pathlib import Path
from typing import Any

from fwi_visionfm.scripts.build_protocol_v2_splits import build_protocol_v2_splits
from fwi_visionfm.scripts.run_protocol_v7_boundary_auxiliary_tuning import (
    _evaluate_tuned,
    _run_epoch_tuned,
)
from fwi_visionfm.scripts.run_protocol_v7_boundary_auxiliary_smoke import (
    ProtocolV7SmokeModel,
    _first_velocity_shape,
    _is_complete,
    _json,
    _manifest_for,
    _plot_boundary_grid,
    _split_paths,
    _write_history,
    _write_prediction_npz,
)
from fwi_visionfm.scripts.run_protocol_v4_integrated_visual_search import _write_triplet_grid
from fwi_visionfm.torch_backend import require_torch_backend
from fwi_visionfm.torch_backend.data import build_torch_dataloader


SUMMARY_FIELDS = [
    "seed",
    "boundary_method",
    "lambda_boundary",
    "threshold",
    "MAE",
    "RMSE",
    "SSIM",
    "gradient_error",
    "edge_MAE",
    "status",
    "reused_from",
    "skip_reason",
]


def _load_tuning_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _find_tuning_row(rows: list[dict[str, str]], run_id: str) -> dict[str, str]:
    for row in rows:
        if row["run_id"] == run_id:
            return row
    return {}


def _summary_row(
    *,
    seed: int,
    boundary_method: str,
    lambda_boundary: float,
    threshold: float | None,
    mae: str,
    rmse: str,
    ssim: str,
    gradient_error: str,
    edge_mae: str,
    status: str,
    reused_from: str,
    skip_reason: str,
) -> dict[str, str]:
    return {
        "seed": str(seed),
        "boundary_method": boundary_method,
        "lambda_boundary": str(lambda_boundary),
        "threshold": "" if threshold is None else str(threshold),
        "MAE": mae,
        "RMSE": rmse,
        "SSIM": ssim,
        "gradient_error": gradient_error,
        "edge_MAE": edge_mae,
        "status": status,
        "reused_from": reused_from,
        "skip_reason": skip_reason,
    }


def _row_from_tuning(row: dict[str, str], *, alias: str) -> dict[str, str]:
    if not row:
        seed = 0
        lambda_boundary = 0.05 if "lambda005" in alias or "thresholded" in alias else 0.10
        method = "thresholded_gradient" if "thresholded" in alias else (
            "gradient_magnitude_lambda010" if "lambda010" in alias else "gradient_magnitude_lambda005"
        )
        threshold = 0.2 if "thresholded" in alias else None
        return _summary_row(
            seed=seed,
            boundary_method=method,
            lambda_boundary=lambda_boundary,
            threshold=threshold,
            mae="",
            rmse="",
            ssim="",
            gradient_error="",
            edge_mae="",
            status="SKIPPED",
            reused_from=f"tuning:{alias}",
            skip_reason="missing reused tuning row",
        )
    method = row["boundary_method"]
    if alias.startswith("lambda005"):
        method = "gradient_magnitude_lambda005"
    elif alias.startswith("lambda010"):
        method = "gradient_magnitude_lambda010"
    return _summary_row(
        seed=int(row["seed"]),
        boundary_method=method,
        lambda_boundary=float(row["lambda_boundary"]),
        threshold=None if row["threshold"] == "" else float(row["threshold"]),
        mae=row["cross_family_MAE"],
        rmse=row["cross_family_RMSE"],
        ssim=row["cross_family_SSIM"],
        gradient_error=row["cross_family_gradient_error"],
        edge_mae=row["cross_family_edge_MAE"],
        status=row["status"],
        reused_from=f"tuning:{alias}",
        skip_reason=row["skip_reason"],
    )


def build_thresholded_seed_check_summary(*, root: str | Path, reuse_tuning_root: str | Path) -> Path:
    root_path = Path(root)
    tuning_rows = _load_tuning_rows(Path(reuse_tuning_root) / "protocol_v7_boundary_auxiliary_tuning_summary.csv")
    rows: list[dict[str, str]] = []
    for run_id in ("thresholded_seed0", "lambda005_seed0", "lambda005_seed1", "lambda005_seed2", "lambda010_seed0", "lambda010_seed1", "lambda010_seed2"):
        rows.append(_row_from_tuning(_find_tuning_row(tuning_rows, run_id), alias=run_id))
    for seed in (1, 2):
        run_dir = root_path / f"thresholded_seed{seed}"
        config_path = run_dir / "config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            metrics = json.loads((run_dir / "metrics_cross_family_test.json").read_text(encoding="utf-8"))
            rows.append(
                _summary_row(
                    seed=int(config["seed"]),
                    boundary_method="thresholded_gradient",
                    lambda_boundary=float(config["lambda_boundary"]),
                    threshold=config.get("threshold"),
                    mae=str(metrics["mae"]),
                    rmse=str(metrics["rmse"]),
                    ssim=str(metrics["ssim"]),
                    gradient_error=str(metrics["gradient_error"]),
                    edge_mae=str(metrics["edge_mae"]),
                    status=config["status"],
                    reused_from=config.get("reused_from", ""),
                    skip_reason=config.get("skip_reason", ""),
                )
            )
        else:
            rows.append(
                _summary_row(
                    seed=seed,
                    boundary_method="thresholded_gradient",
                    lambda_boundary=0.05,
                    threshold=0.2,
                    mae="",
                    rmse="",
                    ssim="",
                    gradient_error="",
                    edge_mae="",
                    status="SKIPPED",
                    reused_from="",
                    skip_reason="missing run directory",
                )
            )
    rows.sort(key=lambda item: (item["boundary_method"], int(item["seed"])))
    summary_path = root_path / "protocol_v7_thresholded_gradient_seed_check_summary.csv"
    root_path.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return summary_path


def _train_thresholded_run(
    *,
    run_dir: Path,
    seed: int,
    splits: dict[str, list[Path]],
    depth: int,
    width: int,
    epochs: int,
    device: str,
) -> None:
    torch = require_torch_backend()
    model = ProtocolV7SmokeModel(decoder_name="boundary_aux_unet", depth=depth, width=width).to(device)
    optimizer = torch.optim.Adam(list(model.parameters()), lr=1.0e-3)
    train_loader = build_torch_dataloader(splits["train"], batch_size=4, shuffle=True, seed=seed)
    val_loader = build_torch_dataloader(splits["val"], batch_size=4, shuffle=False, seed=seed)
    cross_loader = build_torch_dataloader(splits["cross_family_test"], batch_size=4, shuffle=False, seed=seed)
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        train_metrics = _run_epoch_tuned(
            model,
            train_loader,
            optimizer,
            device=device,
            lambda_boundary=0.05,
            boundary_method="thresholded_gradient",
            threshold=0.2,
        )
        val_metrics, _, _, _, _, _ = _evaluate_tuned(
            model,
            val_loader,
            device=device,
            boundary_method="thresholded_gradient",
            threshold=0.2,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_mae": train_metrics["mae"],
                "train_rmse": train_metrics["rmse"],
                "val_loss": val_metrics["loss"],
                "val_mae": val_metrics["mae"],
                "val_rmse": val_metrics["rmse"],
            }
        )
    val_metrics, val_pred, val_target, val_ids, val_boundary_pred, val_boundary_target = _evaluate_tuned(
        model,
        val_loader,
        device=device,
        boundary_method="thresholded_gradient",
        threshold=0.2,
    )
    cross_metrics, cross_pred, cross_target, cross_ids, cross_boundary_pred, cross_boundary_target = _evaluate_tuned(
        model,
        cross_loader,
        device=device,
        boundary_method="thresholded_gradient",
        threshold=0.2,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_history(run_dir / "train_history.csv", history)
    _json(run_dir / "metrics_val.json", {**val_metrics, "metric_space": "physical_velocity"})
    _json(run_dir / "metrics_cross_family_test.json", {**cross_metrics, "metric_space": "physical_velocity"})
    _write_prediction_npz(run_dir / "predictions_val.npz", prediction=val_pred, target=val_target, sample_ids=val_ids, boundary_pred=val_boundary_pred, boundary_target=val_boundary_target)
    _write_prediction_npz(run_dir / "predictions_cross_family_test.npz", prediction=cross_pred, target=cross_target, sample_ids=cross_ids, boundary_pred=cross_boundary_pred, boundary_target=cross_boundary_target)
    _write_triplet_grid(run_dir / "predictions_cross_family_test.npz", run_dir / "prediction_grid.png", gradient=False)
    _write_triplet_grid(run_dir / "predictions_cross_family_test.npz", run_dir / "gradient_grid.png", gradient=True)
    _plot_boundary_grid(run_dir / "predictions_cross_family_test.npz", run_dir / "boundary_prediction_grid.png", key="boundary")
    _plot_boundary_grid(run_dir / "predictions_cross_family_test.npz", run_dir / "boundary_target_grid.png", key="boundary")


def run_protocol_v7_thresholded_gradient_seed_check(
    *,
    data_root: str | Path,
    source: str,
    target: str,
    output_root: str | Path,
    reuse_tuning_root: str | Path,
    reuse_seed_stability_root: str | Path,
    reuse_smoke_root: str | Path,
    train_size: int,
    val_size: int,
    test_size: int,
    epochs: int,
    seeds: list[int],
    device: str,
) -> dict[str, Any]:
    del reuse_seed_stability_root, reuse_smoke_root
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    seeds = [int(seed) for seed in seeds]
    build_protocol_v2_splits(
        data_root=data_root,
        output_root=root,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        seeds=seeds,
    )
    results: list[dict[str, Any]] = []
    for seed in seeds:
        run_dir = root / f"thresholded_seed{seed}"
        status = "SUCCESS"
        skip_reason = ""
        if not _is_complete(run_dir, has_boundary=True):
            try:
                manifest = _manifest_for(root, source=source, target=target, seed=seed)
                splits = _split_paths(manifest)
                depth, width = _first_velocity_shape(splits["train"])
                _train_thresholded_run(
                    run_dir=run_dir,
                    seed=seed,
                    splits=splits,
                    depth=depth,
                    width=width,
                    epochs=epochs,
                    device=device,
                )
            except Exception as exc:
                status = "FAILED"
                skip_reason = f"{type(exc).__name__}: {exc}"
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        config = {
            "protocol": "protocol_v7_thresholded_gradient_seed_check",
            "source_family": source,
            "target_family": target,
            "run_id": f"thresholded_seed{seed}",
            "seed": seed,
            "bridge": "raw_envelope_spectrum3",
            "geometry_enabled": False,
            "aggregator": "mean",
            "backbone": "cnn_baseline",
            "decoder": "boundary_aux_unet",
            "loss": "boundary_aux_l1",
            "boundary_method": "thresholded_gradient",
            "lambda_boundary": 0.05,
            "threshold": 0.2,
            "train_size": int(train_size),
            "val_size": int(val_size),
            "test_size": int(test_size),
            "epochs": int(epochs),
            "device": device,
            "metric_space": "physical_velocity",
            "status": status,
            "reused_from": "",
            "skip_reason": skip_reason,
        }
        _json(run_dir / "config.json", config)
        (run_dir / "run_log.txt").write_text(f"status={status}\nskip_reason={skip_reason}\n", encoding="utf-8")
        results.append(config)
    summary_path = build_thresholded_seed_check_summary(root=root, reuse_tuning_root=reuse_tuning_root)
    return {"root": str(root), "summary_path": str(summary_path), "runs": results}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Protocol V7 thresholded-gradient seed check.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--reuse-tuning-root", type=Path, required=True)
    parser.add_argument("--reuse-seed-stability-root", type=Path, required=True)
    parser.add_argument("--reuse-smoke-root", type=Path, required=True)
    parser.add_argument("--train-size", type=int, default=100)
    parser.add_argument("--val-size", type=int, default=50)
    parser.add_argument("--test-size", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main() -> None:
    result = run_protocol_v7_thresholded_gradient_seed_check(**vars(parse_args()))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
