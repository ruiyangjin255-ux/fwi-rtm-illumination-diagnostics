from __future__ import annotations

import argparse
import csv
import json
import traceback
from pathlib import Path
from typing import Any

from fwi_visionfm.data.boundary_targets import build_velocity_boundary_target
from fwi_visionfm.scripts.build_protocol_v2_splits import build_protocol_v2_splits
from fwi_visionfm.scripts.run_protocol_v4_integrated_visual_search import _write_triplet_grid
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
from fwi_visionfm.torch_backend import require_torch_backend
from fwi_visionfm.torch_backend.data import build_torch_dataloader
from fwi_visionfm.training.losses import compute_loss_components


SUMMARY_FIELDS = [
    "run_id",
    "seed",
    "lambda_boundary",
    "boundary_method",
    "threshold",
    "val_MAE",
    "val_RMSE",
    "val_SSIM",
    "cross_family_MAE",
    "cross_family_RMSE",
    "cross_family_SSIM",
    "cross_family_gradient_error",
    "cross_family_edge_MAE",
    "boundary_val_L1",
    "status",
    "reused_from",
    "skip_reason",
]


TUNING_RUNS = [
    {"run_id": "lambda003_seed0", "seed": 0, "lambda_boundary": 0.03, "boundary_method": "gradient_magnitude", "threshold": None, "reuse": None},
    {"run_id": "lambda003_seed1", "seed": 1, "lambda_boundary": 0.03, "boundary_method": "gradient_magnitude", "threshold": None, "reuse": None},
    {"run_id": "lambda003_seed2", "seed": 2, "lambda_boundary": 0.03, "boundary_method": "gradient_magnitude", "threshold": None, "reuse": None},
    {"run_id": "lambda005_seed0", "seed": 0, "lambda_boundary": 0.05, "boundary_method": "gradient_magnitude", "threshold": None, "reuse": "smoke:run_2"},
    {"run_id": "lambda005_seed1", "seed": 1, "lambda_boundary": 0.05, "boundary_method": "gradient_magnitude", "threshold": None, "reuse": None},
    {"run_id": "lambda005_seed2", "seed": 2, "lambda_boundary": 0.05, "boundary_method": "gradient_magnitude", "threshold": None, "reuse": None},
    {"run_id": "lambda010_seed0", "seed": 0, "lambda_boundary": 0.10, "boundary_method": "gradient_magnitude", "threshold": None, "reuse": "seed_stability:boundary_aux_seed0"},
    {"run_id": "lambda010_seed1", "seed": 1, "lambda_boundary": 0.10, "boundary_method": "gradient_magnitude", "threshold": None, "reuse": "seed_stability:boundary_aux_seed1"},
    {"run_id": "lambda010_seed2", "seed": 2, "lambda_boundary": 0.10, "boundary_method": "gradient_magnitude", "threshold": None, "reuse": "seed_stability:boundary_aux_seed2"},
    {"run_id": "sobel_seed0", "seed": 0, "lambda_boundary": 0.05, "boundary_method": "sobel", "threshold": None, "reuse": "smoke:run_4"},
    {"run_id": "thresholded_seed0", "seed": 0, "lambda_boundary": 0.05, "boundary_method": "thresholded_gradient", "threshold": 0.2, "reuse": None},
    {"run_id": "baseline_seed0", "seed": 0, "lambda_boundary": None, "boundary_method": None, "threshold": None, "reuse": "seed_stability:baseline_seed0"},
    {"run_id": "baseline_seed1", "seed": 1, "lambda_boundary": None, "boundary_method": None, "threshold": None, "reuse": "seed_stability:baseline_seed1"},
    {"run_id": "baseline_seed2", "seed": 2, "lambda_boundary": None, "boundary_method": None, "threshold": None, "reuse": "seed_stability:baseline_seed2"},
]


def _run_dir(root: Path, run_id: str) -> Path:
    return root / run_id


def _sample_ids(batch: dict[str, Any]) -> Any:
    import numpy as np

    return np.asarray([str(item) for item in batch["path"]], dtype=object)


def _compute_boundary_val_l1(boundary_pred: Any, boundary_target: Any) -> float | None:
    import numpy as np

    if boundary_pred is None or boundary_target is None:
        return None
    return float(np.mean(np.abs(boundary_pred - boundary_target)))


def _run_epoch_tuned(
    model: ProtocolV7SmokeModel,
    loader: Any,
    optimizer: Any,
    *,
    device: str,
    lambda_boundary: float,
    boundary_method: str,
    threshold: float | None,
) -> dict[str, float]:
    import numpy as np

    losses: list[float] = []
    preds: list[Any] = []
    trues: list[Any] = []
    model.train()
    for batch in loader:
        records = batch["records"].to(device)
        velocity = batch["velocity"].to(device)
        source_positions = batch["source_positions"].to(device)
        optimizer.zero_grad()
        output = model(records, source_positions)
        components = compute_loss_components(
            output,
            velocity,
            weights={"boundary_aux_l1": 1.0},
            component_kwargs={
                "boundary_aux_l1": {
                    "lambda_boundary": float(lambda_boundary),
                    "boundary_method": str(boundary_method),
                    "threshold": float(0.2 if threshold is None else threshold),
                }
            },
        )
        loss = components["total_loss"]
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        preds.append(output["velocity"].detach().cpu().numpy()[:, 0])
        trues.append(velocity.detach().cpu().numpy())
    prediction_np = np.concatenate(preds, axis=0)
    target_np = np.concatenate(trues, axis=0)
    from fwi_visionfm.evaluation.metrics import compute_velocity_metrics

    metrics = compute_velocity_metrics(prediction_np, target_np)
    metrics["loss"] = float(np.mean(losses))
    return metrics


def _evaluate_tuned(
    model: ProtocolV7SmokeModel,
    loader: Any,
    *,
    device: str,
    boundary_method: str,
    threshold: float | None,
) -> tuple[dict[str, float], Any, Any, Any, Any | None, Any | None]:
    import numpy as np

    torch = require_torch_backend()
    criterion = torch.nn.L1Loss()
    preds: list[Any] = []
    trues: list[Any] = []
    sample_ids: list[Any] = []
    boundary_preds: list[Any] = []
    boundary_targets: list[Any] = []
    losses: list[float] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            records = batch["records"].to(device)
            velocity = batch["velocity"].to(device)
            source_positions = batch["source_positions"].to(device)
            output = model(records, source_positions)
            prediction = output["velocity"][:, 0]
            losses.append(float(criterion(prediction, velocity).detach().cpu()))
            preds.append(prediction.detach().cpu().numpy())
            trues.append(velocity.detach().cpu().numpy())
            sample_ids.append(_sample_ids(batch))
            boundary_target = build_velocity_boundary_target(
                velocity.unsqueeze(1),
                method=str(boundary_method),
                threshold=float(0.2 if threshold is None else threshold),
            )
            boundary_preds.append(output["boundary"].detach().cpu().numpy())
            boundary_targets.append(boundary_target.detach().cpu().numpy())
    prediction_np = np.concatenate(preds, axis=0)
    target_np = np.concatenate(trues, axis=0)
    from fwi_visionfm.evaluation.metrics import compute_velocity_metrics

    metrics = compute_velocity_metrics(prediction_np, target_np)
    metrics["loss"] = float(np.mean(losses))
    boundary_pred_np = np.concatenate(boundary_preds, axis=0)
    boundary_target_np = np.concatenate(boundary_targets, axis=0)
    metrics["boundary_val_l1"] = _compute_boundary_val_l1(boundary_pred_np, boundary_target_np)
    return metrics, prediction_np, target_np, np.concatenate(sample_ids, axis=0), boundary_pred_np, boundary_target_np


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _summary_row(
    *,
    run_id: str,
    seed: int,
    lambda_boundary: float | None,
    boundary_method: str | None,
    threshold: float | None,
    val: dict[str, Any],
    cross: dict[str, Any],
    status: str,
    reused_from: str,
    skip_reason: str,
) -> dict[str, str]:
    return {
        "run_id": run_id,
        "seed": str(seed),
        "lambda_boundary": "" if lambda_boundary is None else str(lambda_boundary),
        "boundary_method": "" if boundary_method is None else str(boundary_method),
        "threshold": "" if threshold is None else str(threshold),
        "val_MAE": "" if "mae" not in val else str(val["mae"]),
        "val_RMSE": "" if "rmse" not in val else str(val["rmse"]),
        "val_SSIM": "" if "ssim" not in val else str(val["ssim"]),
        "cross_family_MAE": "" if "mae" not in cross else str(cross["mae"]),
        "cross_family_RMSE": "" if "rmse" not in cross else str(cross["rmse"]),
        "cross_family_SSIM": "" if "ssim" not in cross else str(cross["ssim"]),
        "cross_family_gradient_error": "" if "gradient_error" not in cross else str(cross["gradient_error"]),
        "cross_family_edge_MAE": "" if "edge_mae" not in cross else str(cross["edge_mae"]),
        "boundary_val_L1": "" if "boundary_val_l1" not in val else str(val["boundary_val_l1"]),
        "status": status,
        "reused_from": reused_from,
        "skip_reason": skip_reason,
    }


def _seed_stability_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _reused_seed_row(rows: list[dict[str, str]], *, seed: int, model_type: str) -> dict[str, str]:
    for row in rows:
        if int(row["seed"]) == int(seed) and row["model_type"] == model_type:
            return row
    return {}


def _row_from_seed_stability(rows: list[dict[str, str]], *, seed: int, model_type: str, run_id: str, reused_from: str) -> dict[str, str]:
    row = _reused_seed_row(rows, seed=seed, model_type=model_type)
    if not row:
        return _summary_row(
            run_id=run_id,
            seed=seed,
            lambda_boundary=None if model_type == "baseline" else 0.10,
            boundary_method=None if model_type == "baseline" else "gradient_magnitude",
            threshold=None,
            val={},
            cross={},
            status="SKIPPED",
            reused_from=reused_from,
            skip_reason="missing reused seed-stability row",
        )
    return {
        "run_id": run_id,
        "seed": row["seed"],
        "lambda_boundary": row["lambda_boundary"],
        "boundary_method": row["boundary_method"],
        "threshold": "",
        "val_MAE": row["val_MAE"],
        "val_RMSE": row["val_RMSE"],
        "val_SSIM": row["val_SSIM"],
        "cross_family_MAE": row["cross_family_MAE"],
        "cross_family_RMSE": row["cross_family_RMSE"],
        "cross_family_SSIM": row["cross_family_SSIM"],
        "cross_family_gradient_error": row["cross_family_gradient_error"],
        "cross_family_edge_MAE": row["cross_family_edge_MAE"],
        "boundary_val_L1": row["boundary_val_L1"],
        "status": row["status"],
        "reused_from": reused_from,
        "skip_reason": row["skip_reason"],
    }


def _row_from_run_dir(run_dir: Path, *, run_id: str, reused_from: str = "") -> dict[str, str]:
    config = _load_json(run_dir / "config.json")
    val = _load_json(run_dir / "metrics_val.json")
    cross = _load_json(run_dir / "metrics_cross_family_test.json")
    return _summary_row(
        run_id=run_id,
        seed=int(config.get("seed", 0)),
        lambda_boundary=config.get("lambda_boundary"),
        boundary_method=config.get("boundary_method"),
        threshold=config.get("threshold"),
        val=val,
        cross=cross,
        status=config.get("status", ""),
        reused_from=reused_from or config.get("reused_from", ""),
        skip_reason=config.get("skip_reason", ""),
    )


def build_boundary_tuning_summary(*, root: str | Path, reuse_seed_stability_root: str | Path, reuse_smoke_root: str | Path) -> Path:
    root_path = Path(root)
    reuse_seed_root = Path(reuse_seed_stability_root)
    reuse_smoke = Path(reuse_smoke_root)
    rows: list[dict[str, str]] = []

    seed_rows = _seed_stability_rows(reuse_seed_root / "protocol_v7_boundary_auxiliary_seed_stability_summary.csv")
    for seed in (0, 1, 2):
        rows.append(_row_from_seed_stability(seed_rows, seed=seed, model_type="baseline", run_id=f"baseline_seed{seed}", reused_from=f"seed_stability:baseline_seed{seed}"))
        rows.append(_row_from_seed_stability(seed_rows, seed=seed, model_type="boundary_aux", run_id=f"lambda010_seed{seed}", reused_from=f"seed_stability:boundary_aux_seed{seed}"))

    reused_smoke_map = {
        "lambda005_seed0": reuse_smoke / "run_2",
        "sobel_seed0": reuse_smoke / "run_4",
    }
    for run_id, run_dir in reused_smoke_map.items():
        rows.append(_row_from_run_dir(run_dir, run_id=run_id, reused_from=f"smoke:{run_dir.name}"))

    for config_path in sorted(root_path.glob("*\\config.json")):
        run_id = config_path.parent.name
        if run_id in {"lambda005_seed0", "sobel_seed0"}:
            continue
        rows.append(_row_from_run_dir(config_path.parent, run_id=run_id))

    existing_ids = {row["run_id"] for row in rows}
    for entry in TUNING_RUNS:
        if entry["run_id"] not in existing_ids:
            rows.append(
                _summary_row(
                    run_id=entry["run_id"],
                    seed=int(entry["seed"]),
                    lambda_boundary=entry["lambda_boundary"],
                    boundary_method=entry["boundary_method"],
                    threshold=entry["threshold"],
                    val={},
                    cross={},
                    status="SKIPPED",
                    reused_from=str(entry["reuse"] or ""),
                    skip_reason="missing run directory",
                )
            )
    rows.sort(key=lambda item: item["run_id"])
    summary_path = root_path / "protocol_v7_boundary_auxiliary_tuning_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return summary_path


def _train_boundary_run(
    *,
    run_dir: Path,
    seed: int,
    lambda_boundary: float,
    boundary_method: str,
    threshold: float | None,
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
    in_loader = build_torch_dataloader(splits["in_family_test"], batch_size=4, shuffle=False, seed=seed)
    cross_loader = build_torch_dataloader(splits["cross_family_test"], batch_size=4, shuffle=False, seed=seed)
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        train_metrics = _run_epoch_tuned(
            model,
            train_loader,
            optimizer,
            device=device,
            lambda_boundary=lambda_boundary,
            boundary_method=boundary_method,
            threshold=threshold,
        )
        val_metrics, _, _, _, _, _ = _evaluate_tuned(
            model,
            val_loader,
            device=device,
            boundary_method=boundary_method,
            threshold=threshold,
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
        boundary_method=boundary_method,
        threshold=threshold,
    )
    in_metrics, in_pred, in_target, in_ids, in_boundary_pred, in_boundary_target = _evaluate_tuned(
        model,
        in_loader,
        device=device,
        boundary_method=boundary_method,
        threshold=threshold,
    )
    cross_metrics, cross_pred, cross_target, cross_ids, cross_boundary_pred, cross_boundary_target = _evaluate_tuned(
        model,
        cross_loader,
        device=device,
        boundary_method=boundary_method,
        threshold=threshold,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_history(run_dir / "train_history.csv", history)
    _json(run_dir / "metrics_val.json", {**val_metrics, "metric_space": "physical_velocity"})
    _json(run_dir / "metrics_in_family_test.json", {**in_metrics, "metric_space": "physical_velocity"})
    _json(run_dir / "metrics_cross_family_test.json", {**cross_metrics, "metric_space": "physical_velocity"})
    _write_prediction_npz(run_dir / "predictions_val.npz", prediction=val_pred, target=val_target, sample_ids=val_ids, boundary_pred=val_boundary_pred, boundary_target=val_boundary_target)
    _write_prediction_npz(run_dir / "predictions_in_family_test.npz", prediction=in_pred, target=in_target, sample_ids=in_ids, boundary_pred=in_boundary_pred, boundary_target=in_boundary_target)
    _write_prediction_npz(run_dir / "predictions_cross_family_test.npz", prediction=cross_pred, target=cross_target, sample_ids=cross_ids, boundary_pred=cross_boundary_pred, boundary_target=cross_boundary_target)
    _write_triplet_grid(run_dir / "predictions_cross_family_test.npz", run_dir / "prediction_grid.png", gradient=False)
    _write_triplet_grid(run_dir / "predictions_cross_family_test.npz", run_dir / "gradient_grid.png", gradient=True)
    _plot_boundary_grid(run_dir / "predictions_cross_family_test.npz", run_dir / "boundary_prediction_grid.png", key="boundary")
    _plot_boundary_grid(run_dir / "predictions_cross_family_test.npz", run_dir / "boundary_target_grid.png", key="boundary")


def run_protocol_v7_boundary_auxiliary_tuning(
    *,
    data_root: str | Path,
    source: str,
    target: str,
    output_root: str | Path,
    reuse_seed_stability_root: str | Path,
    reuse_smoke_root: str | Path,
    train_size: int,
    val_size: int,
    test_size: int,
    epochs: int,
    device: str,
) -> dict[str, Any]:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    build_protocol_v2_splits(
        data_root=data_root,
        output_root=root,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        seeds=[0, 1, 2],
    )
    results: list[dict[str, Any]] = []
    for entry in TUNING_RUNS:
        if entry["reuse"] is not None:
            results.append({"run_id": entry["run_id"], "status": "REUSED"})
            continue
        run_dir = _run_dir(root, entry["run_id"])
        status = "SUCCESS"
        skip_reason = ""
        if not _is_complete(run_dir, has_boundary=True):
            try:
                manifest = _manifest_for(root, source=source, target=target, seed=int(entry["seed"]))
                splits = _split_paths(manifest)
                depth, width = _first_velocity_shape(splits["train"])
                _train_boundary_run(
                    run_dir=run_dir,
                    seed=int(entry["seed"]),
                    lambda_boundary=float(entry["lambda_boundary"]),
                    boundary_method=str(entry["boundary_method"]),
                    threshold=entry["threshold"],
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
            "protocol": "protocol_v7_boundary_auxiliary_tuning",
            "source_family": source,
            "target_family": target,
            "run_id": entry["run_id"],
            "seed": int(entry["seed"]),
            "bridge": "raw_envelope_spectrum3",
            "geometry_enabled": False,
            "aggregator": "mean",
            "backbone": "cnn_baseline",
            "decoder": "boundary_aux_unet",
            "loss": "boundary_aux_l1",
            "lambda_boundary": entry["lambda_boundary"],
            "boundary_method": entry["boundary_method"],
            "threshold": entry["threshold"],
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
    summary_path = build_boundary_tuning_summary(root=root, reuse_seed_stability_root=reuse_seed_stability_root, reuse_smoke_root=reuse_smoke_root)
    return {"root": str(root), "summary_path": str(summary_path), "runs": results}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Protocol V7 boundary auxiliary tuning.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--reuse-seed-stability-root", type=Path, required=True)
    parser.add_argument("--reuse-smoke-root", type=Path, required=True)
    parser.add_argument("--train-size", type=int, default=100)
    parser.add_argument("--val-size", type=int, default=50)
    parser.add_argument("--test-size", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main() -> None:
    result = run_protocol_v7_boundary_auxiliary_tuning(**vars(parse_args()))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
