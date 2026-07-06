from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from rtm_acoustic.acoustic_rtm import (
    RTMConfig,
    crop_padded_model,
    crop_padded_record,
    multishot_reverse_time_migrate,
    mute_direct_arrivals,
    pad_rtm_config,
    pad_velocity_model,
    preprocess_migration_section,
    preprocess_stacked_record,
    read_binary_model,
    shot_positions_from_spacing,
)
from rtm_acoustic.plot_paper_style import save_migration_figure, save_record_and_migration_figure
from rtm_acoustic.run_multishot_rtm import limited_shots


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRUE_MODEL = ROOT / "fd2d_pml" / "vel" / "seg676x230.bin"
DEFAULT_FWI_DIR = ROOT / "rtm_acoustic" / "outputs" / "FWI" / "full_salt_fwi_cg_allshots_v2"
DEFAULT_OUTPUT = ROOT / "rtm_acoustic" / "outputs" / "RTM" / "before_after_fwi_smoke"


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    av = np.asarray(a, dtype=np.float64).ravel()
    bv = np.asarray(b, dtype=np.float64).ravel()
    if av.size != bv.size or av.size == 0 or np.std(av) == 0.0 or np.std(bv) == 0.0:
        return float("nan")
    return float(np.corrcoef(av, bv)[0, 1])


def _image_metrics(image: np.ndarray, reference: np.ndarray | None = None) -> dict[str, Any]:
    image64 = np.asarray(image, dtype=np.float64)
    metrics: dict[str, Any] = {
        "abs_mean": float(np.mean(np.abs(image64))),
        "abs_p95": float(np.percentile(np.abs(image64), 95)),
        "abs_p99": float(np.percentile(np.abs(image64), 99)),
        "rms": float(np.sqrt(np.mean(image64 * image64))),
        "nonzero_fraction": float(np.mean(np.abs(image64) > 0.0)),
    }
    if reference is not None:
        ref = np.asarray(reference, dtype=np.float64)
        diff = image64 - ref
        metrics["reference_corr"] = _safe_corr(image64, ref)
        metrics["reference_rmse"] = float(np.sqrt(np.mean(diff * diff)))
        metrics["reference_mae"] = float(np.mean(np.abs(diff)))
    return metrics


def _run_case(
    *,
    true_velocity: np.ndarray,
    migration_velocity: np.ndarray,
    cfg: RTMConfig,
    shots: list[int],
    output_dir: Path,
    laplacian_power: int,
    min_illumination_fraction: float,
    direct_mute: bool,
    subtract_direct_wave: bool,
    direct_velocity: float,
    direct_mute_padding: float,
    direct_mute_taper: float,
) -> dict[str, np.ndarray]:
    output_dir.mkdir(parents=True, exist_ok=True)

    def record_provider(source_x: int, record: np.ndarray) -> np.ndarray:
        if not direct_mute:
            return record
        return mute_direct_arrivals(
            record,
            cfg,
            source_x=source_x,
            direct_velocity=direct_velocity,
            padding_time=direct_mute_padding,
            taper_time=direct_mute_taper,
        )

    result = multishot_reverse_time_migrate(
        true_velocity,
        cfg,
        shot_positions=shots,
        wavefield_path=output_dir / "source_wavefield_float32.dat",
        laplacian_power=laplacian_power,
        migration_velocity=migration_velocity,
        subtract_direct_wave=subtract_direct_wave,
        min_illumination_fraction=min_illumination_fraction,
        record_provider=record_provider,
    )
    arrays = {
        "stacked_record_raw": result.stacked_record.astype(np.float32),
        "rtm_raw": result.image.astype(np.float32),
        "rtm_source_normalized": result.normalized_image.astype(np.float32),
        "rtm_source_receiver_normalized": result.source_receiver_normalized_image.astype(np.float32),
        "rtm_laplacian_filtered": result.filtered_image.astype(np.float32),
        "illumination": result.illumination.astype(np.float32),
        "receiver_illumination": result.receiver_illumination.astype(np.float32),
    }
    for name, array in arrays.items():
        np.save(output_dir / f"{name}.npy", array)
    return arrays


def run_before_after_rtm(
    *,
    true_model_path: Path,
    initial_model_path: Path,
    inverted_model_path: Path,
    output_dir: Path,
    nx: int = 676,
    nz: int = 230,
    dx: float = 10.0,
    dz: float = 10.0,
    dt: float = 0.001,
    nt: int = 600,
    f0: float = 15.0,
    source_z: int = 4,
    receiver_z: int = 4,
    absorb_cells: int = 40,
    fd_order: int = 8,
    shot_spacing: float = 90.0,
    shot_margin_cells: int = 4,
    max_shots: int = 12,
    pad_x: int = 0,
    pad_top: int = 0,
    pad_bottom: int = 0,
    laplacian_power: int = 1,
    min_illumination_fraction: float = 0.01,
    direct_mute: bool = True,
    direct_velocity: float = 2000.0,
    direct_mute_padding: float = 0.03,
    direct_mute_taper: float = 0.02,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    true_physical = read_binary_model(true_model_path, nx=nx, nz=nz)
    initial_physical = np.load(initial_model_path).astype(np.float32)
    inverted_physical = np.load(inverted_model_path).astype(np.float32)
    if true_physical.shape != initial_physical.shape or true_physical.shape != inverted_physical.shape:
        raise ValueError("true, initial and inverted models must have identical physical shapes")
    true_velocity = pad_velocity_model(true_physical, pad_x=pad_x, pad_top=pad_top, pad_bottom=pad_bottom)
    initial_velocity = pad_velocity_model(initial_physical, pad_x=pad_x, pad_top=pad_top, pad_bottom=pad_bottom)
    inverted_velocity = pad_velocity_model(inverted_physical, pad_x=pad_x, pad_top=pad_top, pad_bottom=pad_bottom)

    base_cfg = RTMConfig(
        nx=nx,
        nz=nz,
        dx=dx,
        dz=dz,
        dt=dt,
        nt=nt,
        f0=f0,
        source_x=nx // 2,
        source_z=source_z,
        receiver_z=receiver_z,
        absorb_cells=absorb_cells,
        fd_order=fd_order,
    )
    cfg = pad_rtm_config(base_cfg, pad_x=pad_x, pad_top=pad_top, pad_bottom=pad_bottom)
    physical_shots = limited_shots(
        shot_positions_from_spacing(nx=nx, dx=dx, spacing_m=shot_spacing, margin_cells=shot_margin_cells),
        max_shots=max_shots,
    )
    shots = [shot + pad_x for shot in physical_shots]
    if not shots:
        raise ValueError("no shots selected")

    reference = _run_case(
        true_velocity=true_velocity,
        migration_velocity=true_velocity,
        cfg=cfg,
        shots=shots,
        output_dir=output_dir / "reference_true_velocity",
        laplacian_power=laplacian_power,
        min_illumination_fraction=min_illumination_fraction,
        direct_mute=direct_mute,
        subtract_direct_wave=False,
        direct_velocity=direct_velocity,
        direct_mute_padding=direct_mute_padding,
        direct_mute_taper=direct_mute_taper,
    )
    before = _run_case(
        true_velocity=true_velocity,
        migration_velocity=initial_velocity,
        cfg=cfg,
        shots=shots,
        output_dir=output_dir / "before_initial_velocity",
        laplacian_power=laplacian_power,
        min_illumination_fraction=min_illumination_fraction,
        direct_mute=direct_mute,
        subtract_direct_wave=False,
        direct_velocity=direct_velocity,
        direct_mute_padding=direct_mute_padding,
        direct_mute_taper=direct_mute_taper,
    )
    after = _run_case(
        true_velocity=true_velocity,
        migration_velocity=inverted_velocity,
        cfg=cfg,
        shots=shots,
        output_dir=output_dir / "after_fwi_velocity",
        laplacian_power=laplacian_power,
        min_illumination_fraction=min_illumination_fraction,
        direct_mute=direct_mute,
        subtract_direct_wave=False,
        direct_velocity=direct_velocity,
        direct_mute_padding=direct_mute_padding,
        direct_mute_taper=direct_mute_taper,
    )

    original_shape = true_physical.shape
    metrics: dict[str, Any] = {
        "config": asdict(base_cfg),
        "padded_config": asdict(cfg),
        "padding": {"pad_x": pad_x, "pad_top": pad_top, "pad_bottom": pad_bottom},
        "physical_shot_positions": physical_shots,
        "padded_shot_positions": shots,
        "shot_count": len(shots),
        "cases": {},
    }
    for label, arrays in (
        ("reference_true_velocity", reference),
        ("before_initial_velocity", before),
        ("after_fwi_velocity", after),
    ):
        case_dir = output_dir / label
        cropped_filtered = crop_padded_model(
            arrays["rtm_laplacian_filtered"],
            original_shape=original_shape,
            pad_x=pad_x,
            pad_top=pad_top,
        )
        cropped_norm = crop_padded_model(
            arrays["rtm_source_normalized"],
            original_shape=original_shape,
            pad_x=pad_x,
            pad_top=pad_top,
        )
        cropped_record = crop_padded_record(arrays["stacked_record_raw"], original_nx=nx, pad_x=pad_x)
        display_record = preprocess_stacked_record(cropped_record, dt=dt, mute_time=0.0, time_power=0.2)
        display_migration = preprocess_migration_section(
            cropped_filtered,
            depth_power=0.15,
            clip_percentile=99.5,
            trace_balance=0.25,
            output_clip=0.80,
        )
        np.save(case_dir / "rtm_laplacian_filtered_physical.npy", cropped_filtered.astype(np.float32))
        np.save(case_dir / "rtm_source_normalized_physical.npy", cropped_norm.astype(np.float32))
        np.save(case_dir / "stacked_record_physical.npy", cropped_record.astype(np.float32))
        np.save(case_dir / "rtm_display.npy", display_migration.astype(np.float32))
        save_migration_figure(
            case_dir / "rtm_display.png",
            display_migration,
            dx=dx,
            dz=dz,
            title=label.replace("_", " "),
        )
        save_record_and_migration_figure(
            case_dir / "record_and_rtm.png",
            display_record,
            display_migration,
            dx=dx,
            dz=dz,
            dt=dt,
        )
        metrics["cases"][label] = {
            "filtered": _image_metrics(cropped_filtered, None if label == "reference_true_velocity" else crop_padded_model(reference["rtm_laplacian_filtered"], original_shape, pad_x, pad_top)),
            "source_normalized": _image_metrics(cropped_norm, None if label == "reference_true_velocity" else crop_padded_model(reference["rtm_source_normalized"], original_shape, pad_x, pad_top)),
        }

    before_rmse = metrics["cases"]["before_initial_velocity"]["filtered"]["reference_rmse"]
    after_rmse = metrics["cases"]["after_fwi_velocity"]["filtered"]["reference_rmse"]
    metrics["filtered_reference_rmse_improvement_fraction"] = float((before_rmse - after_rmse) / max(before_rmse, 1.0e-20))
    metrics["verdict"] = "after_fwi_closer_to_reference" if after_rmse < before_rmse else "after_fwi_not_closer_to_reference"
    (output_dir / "rtm_before_after_summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    lines = ["# RTM before/after FWI summary", ""]
    lines.append(f"- `verdict`: {metrics['verdict']}")
    lines.append(f"- `filtered_reference_rmse_improvement_fraction`: {metrics['filtered_reference_rmse_improvement_fraction']}")
    lines.append(f"- `shot_count`: {metrics['shot_count']}")
    (output_dir / "rtm_before_after_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare RTM images before and after FWI velocity updating.")
    parser.add_argument("--true-model", type=Path, default=DEFAULT_TRUE_MODEL)
    parser.add_argument("--fwi-dir", type=Path, default=DEFAULT_FWI_DIR)
    parser.add_argument("--initial-model", type=Path, default=None)
    parser.add_argument("--inverted-model", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--nx", type=int, default=676)
    parser.add_argument("--nz", type=int, default=230)
    parser.add_argument("--dx", type=float, default=10.0)
    parser.add_argument("--dz", type=float, default=10.0)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--nt", type=int, default=600)
    parser.add_argument("--f0", type=float, default=15.0)
    parser.add_argument("--source-z", type=int, default=4)
    parser.add_argument("--receiver-z", type=int, default=4)
    parser.add_argument("--absorb-cells", type=int, default=40)
    parser.add_argument("--fd-order", type=int, default=8)
    parser.add_argument("--shot-spacing", type=float, default=90.0)
    parser.add_argument("--shot-margin-cells", type=int, default=4)
    parser.add_argument("--max-shots", type=int, default=12)
    parser.add_argument("--pad-x", type=int, default=0)
    parser.add_argument("--pad-top", type=int, default=0)
    parser.add_argument("--pad-bottom", type=int, default=0)
    parser.add_argument("--laplacian-power", type=int, default=1)
    parser.add_argument("--min-illumination-fraction", type=float, default=0.01)
    parser.add_argument("--no-direct-mute", action="store_true")
    parser.add_argument("--direct-velocity", type=float, default=2000.0)
    parser.add_argument("--direct-mute-padding", type=float, default=0.03)
    parser.add_argument("--direct-mute-taper", type=float, default=0.02)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    initial_model = args.initial_model or args.fwi_dir / "full_salt_initial_model.npy"
    inverted_model = args.inverted_model or args.fwi_dir / "full_salt_inverted_model.npy"
    metrics = run_before_after_rtm(
        true_model_path=args.true_model,
        initial_model_path=initial_model,
        inverted_model_path=inverted_model,
        output_dir=args.output_dir,
        nx=args.nx,
        nz=args.nz,
        dx=args.dx,
        dz=args.dz,
        dt=args.dt,
        nt=args.nt,
        f0=args.f0,
        source_z=args.source_z,
        receiver_z=args.receiver_z,
        absorb_cells=args.absorb_cells,
        fd_order=args.fd_order,
        shot_spacing=args.shot_spacing,
        shot_margin_cells=args.shot_margin_cells,
        max_shots=args.max_shots,
        pad_x=args.pad_x,
        pad_top=args.pad_top,
        pad_bottom=args.pad_bottom,
        laplacian_power=args.laplacian_power,
        min_illumination_fraction=args.min_illumination_fraction,
        direct_mute=not args.no_direct_mute,
        direct_velocity=args.direct_velocity,
        direct_mute_padding=args.direct_mute_padding,
        direct_mute_taper=args.direct_mute_taper,
    )
    print(f"verdict: {metrics['verdict']}")
    print(f"summary: {args.output_dir / 'rtm_before_after_summary.json'}")


if __name__ == "__main__":
    main()
