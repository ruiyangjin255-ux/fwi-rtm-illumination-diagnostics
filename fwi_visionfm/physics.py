from __future__ import annotations

import numpy as np

Array = np.ndarray


def acoustic_data_misfit(predicted_record: Array, observed_record: Array) -> float:
    predicted = np.asarray(predicted_record, dtype=np.float64)
    observed = np.asarray(observed_record, dtype=np.float64)
    if predicted.shape != observed.shape:
        raise ValueError(f"record shapes must match, got {predicted.shape} and {observed.shape}")
    residual = predicted - observed
    return float(0.5 * np.mean(residual * residual))


def physics_consistency_loss(
    *,
    predicted_velocity: Array,
    observed_velocity: Array,
    nx: int,
    nz: int,
    nt: int,
    source_x: int,
    source_z: int,
    receiver_z: int,
    dx: float = 10.0,
    dz: float = 10.0,
    dt: float = 0.001,
    f0: float = 15.0,
    absorb_cells: int = 6,
    fd_order: int = 4,
) -> dict[str, float]:
    from rtm_acoustic.acoustic_rtm import RTMConfig, forward_model

    cfg = RTMConfig(
        nx=nx,
        nz=nz,
        dx=dx,
        dz=dz,
        dt=dt,
        nt=nt,
        f0=f0,
        source_x=source_x,
        source_z=source_z,
        receiver_z=receiver_z,
        absorb_cells=absorb_cells,
        fd_order=fd_order,
    )
    predicted_record = forward_model(np.asarray(predicted_velocity, dtype=np.float32), cfg)
    observed_record = forward_model(np.asarray(observed_velocity, dtype=np.float32), cfg)
    return {"data_misfit": acoustic_data_misfit(predicted_record, observed_record), "nt": float(nt), "nx": float(nx), "nz": float(nz)}
