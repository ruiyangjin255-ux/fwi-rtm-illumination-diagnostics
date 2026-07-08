from __future__ import annotations

import argparse
import csv
import json
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from fwi_visionfm.scripts.build_protocol_v2_splits import build_protocol_v2_splits
from fwi_visionfm.scripts.run_protocol_v7_boundary_auxiliary_smoke import (
    ProtocolV7SmokeModel,
    _evaluate,
    _first_velocity_shape,
    _is_complete,
    _json,
    _manifest_for,
    _plot_boundary_grid,
    _run_epoch,
    _split_paths,
    _write_history,
    _write_prediction_npz,
)
from fwi_visionfm.scripts.run_protocol_v4_integrated_visual_search import _write_triplet_grid
from fwi_visionfm.torch_backend import require_torch_backend
from fwi_visionfm.torch_backend.data import build_torch_dataloader


SUMMARY_FIELDS = [
    "seed",
    "model_type",
    "decoder",
    "loss",
    "lambda_boundary",
    "boundary_method",
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
    "skip_reason",
]


SELECTED_RUNS = [
    {
        "model_type": "baseline",
        "run_name": "baseline",
        "decoder": "unet_decoder",
        "loss": "default_l1",
        "lambda_boundary": None,
        "boundary_method": None,
    },
    {
        "model_type": "boundary_aux",
        "run_name": "boundary_aux",
        "decoder": "boundary_aux_unet",
        "loss": "boundary_aux_l1",
        "lambda_boundary": 0.10,
        "boundary_method": "gradient_magnitude",
    },
]


def _run_dir(root: Path, seed: int, model_type: str) -> Path:
    return root / f"seed_{int(seed)}_{model_type}"


def _entry_to_row(entry: dict[str, Any], val: dict[str, Any] | None, cross: dict[str, Any] | None, *, status: str, skip_reason: str) -> dict[str, str]:
    val = val or {}
    cross = cross or {}
    return {
        "seed": str(entry["seed"]),
        "model_type": str(entry["model_type"]),
        "decoder": str(entry["decoder"]),
        "loss": str(entry["loss"]),
        "lambda_boundary": "" if entry.get("lambda_boundary") is None else str(entry["lambda_boundary"]),
        "boundary_method": str(entry.get("boundary_method") or ""),
        "val_MAE": "" if "mae" not in val else str(val["mae"]),
        "val_RMSE": "" if "rmse" not in val else str(val["rmse"]),
        "val_SSIM": "" if "ssim" not in val else str(val["ssim"]),
        "cross_family_MAE": "" if "mae" not in cross else str(cross["mae"]),
        "cross_family_RMSE": "" if "rmse" not in cross else str(cross["rmse"]),
        "cross_family_SSIM": "" if "ssim" not in cross else str(cross["ssim"]),
        "cross_family_gradient_error": "" if "gradient_error" not in cross else str(cross["gradient_error"]),
        "cross_family_edge_MAE": "" if "edge_mae" not in cross else str(cross["edge_mae"]),
        "boundary_val_L1": "" if "boundary_val_l1" not in val else str(val["boundary_val_l1"]),
        "status": str(status),
        "skip_reason": str(skip_reason),
    }


def _load_metrics(run_dir: Path, name: str) -> dict[str, Any]:
    path = run_dir / name
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _row_from_existing_run(run_dir: Path) -> dict[str, str]:
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    val = _load_metrics(run_dir, "metrics_val.json")
    cross = _load_metrics(run_dir, "metrics_cross_family_test.json")
    return _entry_to_row(
        {
            "seed": config.get("seed", ""),
            "model_type": config.get("model_type", "baseline"),
            "decoder": config.get("decoder", ""),
            "loss": config.get("loss", ""),
            "lambda_boundary": config.get("lambda_boundary"),
            "boundary_method": config.get("boundary_method"),
        },
        val,
        cross,
        status=config.get("status", ""),
        skip_reason=config.get("skip_reason", ""),
    )


def _expected_row(seed: int, entry: dict[str, Any], *, status: str, skip_reason: str) -> dict[str, str]:
    return _entry_to_row(
        {
            "seed": int(seed),
            "model_type": entry["model_type"],
            "decoder": entry["decoder"],
            "loss": entry["loss"],
            "lambda_boundary": entry["lambda_boundary"],
            "boundary_method": entry["boundary_method"],
        },
        {},
        {},
        status=status,
        skip_reason=skip_reason,
    )


def build_seed_stability_summary(*, root: str | Path, reuse_seed0_root: str | Path | None = None) -> Path:
    output_root = Path(root)
    rows: list[dict[str, str]] = []
    for run_dir in sorted(output_root.glob("seed_*_*")):
        config_path = run_dir / "config.json"
        if config_path.exists():
            rows.append(_row_from_existing_run(run_dir))
    if reuse_seed0_root is not None:
        seed0_root = Path(reuse_seed0_root)
        smoke_mapping = {"baseline": seed0_root / "run_1", "boundary_aux": seed0_root / "run_3"}
        for model_type, run_dir in smoke_mapping.items():
            if (run_dir / "config.json").exists():
                row = _row_from_existing_run(run_dir)
                row["seed"] = "0"
                row["model_type"] = model_type
                if model_type == "baseline":
                    row["decoder"] = "unet_decoder"
                    row["loss"] = "default_l1"
                    row["lambda_boundary"] = ""
                    row["boundary_method"] = ""
                else:
                    row["decoder"] = "boundary_aux_unet"
                    row["loss"] = "boundary_aux_l1"
                    row["lambda_boundary"] = "0.1"
                    row["boundary_method"] = "gradient_magnitude"
                rows.append(row)
    seen = {(row["seed"], row["model_type"]) for row in rows}
    for seed in (0, 1, 2):
        for entry in SELECTED_RUNS:
            key = (str(seed), entry["model_type"])
            if key not in seen:
                rows.append(_expected_row(seed, entry, status="SKIPPED", skip_reason="missing run directory"))
    rows.sort(key=lambda item: (int(item["seed"]), item["model_type"]))
    summary_path = output_root / "protocol_v7_boundary_auxiliary_seed_stability_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return summary_path


def _to_float(value: str) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def compute_seed_stability_win_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    grouped: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["seed"], {})[row["model_type"]] = row
    wins = {
        "MAE_lower": 0,
        "RMSE_lower": 0,
        "SSIM_higher": 0,
        "gradient_error_lower": 0,
        "edge_MAE_lower": 0,
    }
    for seed_rows in grouped.values():
        baseline = seed_rows.get("baseline")
        boundary = seed_rows.get("boundary_aux")
        if not baseline or not boundary:
            continue
        if baseline.get("status") != "SUCCESS" or boundary.get("status") != "SUCCESS":
            continue
        comparisons = [
            ("MAE_lower", "cross_family_MAE", lambda a, b: a < b),
            ("RMSE_lower", "cross_family_RMSE", lambda a, b: a < b),
            ("SSIM_higher", "cross_family_SSIM", lambda a, b: a > b),
            ("gradient_error_lower", "cross_family_gradient_error", lambda a, b: a < b),
            ("edge_MAE_lower", "cross_family_edge_MAE", lambda a, b: a < b),
        ]
        for win_key, field, compare in comparisons:
            boundary_value = _to_float(boundary.get(field, ""))
            baseline_value = _to_float(baseline.get(field, ""))
            if boundary_value is None or baseline_value is None:
                continue
            if compare(boundary_value, baseline_value):
                wins[win_key] += 1
    return wins


def _run_single_selected(
    *,
    run_dir: Path,
    seed: int,
    entry: dict[str, Any],
    splits: dict[str, list[Path]],
    depth: int,
    width: int,
    epochs: int,
    device: str,
) -> dict[str, Any]:
    torch = require_torch_backend()
    model = ProtocolV7SmokeModel(decoder_name=entry["decoder"], depth=depth, width=width).to(device)
    optimizer = torch.optim.Adam(list(model.parameters()), lr=1.0e-3)
    train_loader = build_torch_dataloader(splits["train"], batch_size=4, shuffle=True, seed=seed)
    val_loader = build_torch_dataloader(splits["val"], batch_size=4, shuffle=False, seed=seed)
    in_loader = build_torch_dataloader(splits["in_family_test"], batch_size=4, shuffle=False, seed=seed)
    cross_loader = build_torch_dataloader(splits["cross_family_test"], batch_size=4, shuffle=False, seed=seed)
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        train_metrics = _run_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            loss_name=entry["loss"],
            lambda_boundary=entry["lambda_boundary"],
            boundary_method=entry["boundary_method"],
        )
        val_metrics, _, _, _, _, _ = _evaluate(model, val_loader, device=device, boundary_method=entry["boundary_method"])
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
    val_metrics, val_pred, val_target, val_ids, val_boundary_pred, val_boundary_target = _evaluate(model, val_loader, device=device, boundary_method=entry["boundary_method"])
    in_metrics, in_pred, in_target, in_ids, in_boundary_pred, in_boundary_target = _evaluate(model, in_loader, device=device, boundary_method=entry["boundary_method"])
    cross_metrics, cross_pred, cross_target, cross_ids, cross_boundary_pred, cross_boundary_target = _evaluate(model, cross_loader, device=device, boundary_method=entry["boundary_method"])
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_history(run_dir / "train_history.csv", history)
    _json(run_dir / "metrics_val.json", {**val_metrics, "metric_space": "physical_velocity"})
    _json(run_dir / "metrics_in_family_test.json", {**in_metrics, "metric_space": "physical_velocity"})
    _json(run_dir / "metrics_cross_family_test.json", {**cross_metrics, "metric_space": "physical_velocity"})
    _write_prediction_npz(
        run_dir / "predictions_val.npz",
        prediction=val_pred,
        target=val_target,
        sample_ids=val_ids,
        boundary_pred=val_boundary_pred,
        boundary_target=val_boundary_target,
    )
    _write_prediction_npz(
        run_dir / "predictions_in_family_test.npz",
        prediction=in_pred,
        target=in_target,
        sample_ids=in_ids,
        boundary_pred=in_boundary_pred,
        boundary_target=in_boundary_target,
    )
    _write_prediction_npz(
        run_dir / "predictions_cross_family_test.npz",
        prediction=cross_pred,
        target=cross_target,
        sample_ids=cross_ids,
        boundary_pred=cross_boundary_pred,
        boundary_target=cross_boundary_target,
    )
    _write_triplet_grid(run_dir / "predictions_cross_family_test.npz", run_dir / "prediction_grid.png", gradient=False)
    _write_triplet_grid(run_dir / "predictions_cross_family_test.npz", run_dir / "gradient_grid.png", gradient=True)
    if cross_boundary_pred is not None and cross_boundary_target is not None:
        _plot_boundary_grid(run_dir / "predictions_cross_family_test.npz", run_dir / "boundary_prediction_grid.png", key="boundary")
        _plot_boundary_grid(run_dir / "predictions_cross_family_test.npz", run_dir / "boundary_target_grid.png", key="boundary")
    return {
        "val_metrics": val_metrics,
        "cross_metrics": cross_metrics,
    }


def run_protocol_v7_boundary_auxiliary_seed_stability(
    *,
    data_root: str | Path,
    source: str,
    target: str,
    output_root: str | Path,
    reuse_seed0_root: str | Path,
    train_size: int,
    val_size: int,
    test_size: int,
    epochs: int,
    seeds: list[int],
    device: str,
) -> dict[str, Any]:
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
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        manifest = _manifest_for(root, source=source, target=target, seed=int(seed))
        splits = _split_paths(manifest)
        depth, width = _first_velocity_shape(splits["train"])
        for entry in SELECTED_RUNS:
            run_dir = _run_dir(root, seed, entry["model_type"])
            status = "SUCCESS"
            skip_reason = ""
            has_boundary = entry["decoder"] == "boundary_aux_unet"
            if not _is_complete(run_dir, has_boundary=has_boundary):
                try:
                    _run_single_selected(
                        run_dir=run_dir,
                        seed=int(seed),
                        entry=entry,
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
                "protocol": "protocol_v7_boundary_auxiliary_seed_stability",
                "source_family": source,
                "target_family": target,
                "seed": int(seed),
                "model_type": entry["model_type"],
                "bridge": "raw_envelope_spectrum3",
                "geometry_enabled": False,
                "aggregator": "mean",
                "backbone": "cnn_baseline",
                "decoder": entry["decoder"],
                "loss": entry["loss"],
                "lambda_boundary": entry["lambda_boundary"],
                "boundary_method": entry["boundary_method"],
                "train_size": int(train_size),
                "val_size": int(val_size),
                "test_size": int(test_size),
                "epochs": int(epochs),
                "device": device,
                "metric_space": "physical_velocity",
                "status": status,
                "skip_reason": skip_reason,
            }
            _json(run_dir / "config.json", config)
            (run_dir / "run_log.txt").write_text(f"status={status}\nskip_reason={skip_reason}\n", encoding="utf-8")
            rows.append(config)
    summary_path = build_seed_stability_summary(root=root, reuse_seed0_root=reuse_seed0_root)
    return {"root": str(root), "summary_path": str(summary_path), "runs": rows}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Protocol V7 boundary auxiliary selected seed stability.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--reuse-seed0-root", type=Path, required=True)
    parser.add_argument("--train-size", type=int, default=100)
    parser.add_argument("--val-size", type=int, default=50)
    parser.add_argument("--test-size", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main() -> None:
    result = run_protocol_v7_boundary_auxiliary_seed_stability(**vars(parse_args()))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
