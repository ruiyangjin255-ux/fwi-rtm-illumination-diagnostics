from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "paper_figures_source" / "acoustic" / "boundary_compare"
DATA_DIR = ROOT / "saved_data" / "acoustic_boundary_compare"

DOMAIN_M = 4000.0
DX = 10.0
DZ = 10.0
DT = 0.001
TMAX = 4.0
F0 = 30.0
V0 = 2000.0
ORDER = 4
PML_CELLS = 20

SRC_X_M = 2000.0
SRC_Z_M = 20.0
REC_Z_M = 20.0


def staggered_coefficients(order: int) -> np.ndarray:
    if order == 2:
        return np.array([1.0], dtype=np.float64)
    if order == 4:
        return np.array([9.0 / 8.0, -1.0 / 24.0], dtype=np.float64)
    if order == 6:
        return np.array([75.0 / 64.0, -25.0 / 384.0, 3.0 / 640.0], dtype=np.float64)
    raise ValueError("order must be 2, 4, or 6")


def derivative_x_forward(a: np.ndarray, coeff: np.ndarray) -> np.ndarray:
    d = np.zeros_like(a)
    n = len(coeff)
    idx = np.arange(n, a.shape[1] - n)
    cols = slice(n, a.shape[1] - n)
    for m, c in enumerate(coeff, start=1):
        d[:, cols] += c * (a[:, idx + m] - a[:, idx - m + 1])
    return d / DX


def derivative_x_backward(a: np.ndarray, coeff: np.ndarray) -> np.ndarray:
    d = np.zeros_like(a)
    n = len(coeff)
    idx = np.arange(n, a.shape[1] - n)
    cols = slice(n, a.shape[1] - n)
    for m, c in enumerate(coeff, start=1):
        d[:, cols] += c * (a[:, idx + m - 1] - a[:, idx - m])
    return d / DX


def derivative_z_forward(a: np.ndarray, coeff: np.ndarray) -> np.ndarray:
    d = np.zeros_like(a)
    n = len(coeff)
    idx = np.arange(n, a.shape[0] - n)
    rows = slice(n, a.shape[0] - n)
    for m, c in enumerate(coeff, start=1):
        d[rows, :] += c * (a[idx + m, :] - a[idx - m + 1, :])
    return d / DZ


def derivative_z_backward(a: np.ndarray, coeff: np.ndarray) -> np.ndarray:
    d = np.zeros_like(a)
    n = len(coeff)
    idx = np.arange(n, a.shape[0] - n)
    rows = slice(n, a.shape[0] - n)
    for m, c in enumerate(coeff, start=1):
        d[rows, :] += c * (a[idx + m - 1, :] - a[idx - m, :])
    return d / DZ


def ricker_wavelet(nt: int) -> np.ndarray:
    t = np.arange(nt) * DT
    t0 = 1.0 / F0
    arg = (np.pi * F0 * (t - t0)) ** 2
    return (1.0 - 2.0 * arg) * np.exp(-arg)


def damping_profile_1d(n: int, pml: int, sigma_max: float, power: int = 2) -> np.ndarray:
    sigma = np.zeros(n, dtype=np.float64)
    for i in range(pml):
        value = sigma_max * ((pml - i) / pml) ** power
        sigma[i] = value
        sigma[-i - 1] = value
    return sigma


def pml_damping(nz: int, nx: int, pml: int, sigma_max: float = 10.0) -> np.ndarray:
    # Deliberately moderate damping: enough to absorb early waves, weak enough
    # to expose late residual boundary reflections in the 4 s record.
    sigma_x = damping_profile_1d(nx, pml, sigma_max=sigma_max)
    sigma_z = damping_profile_1d(nz, pml, sigma_max=sigma_max)
    sigma = sigma_z[:, None] + sigma_x[None, :]
    return np.exp(-sigma * DT)


def cpml_profiles_1d(
    n: int,
    pml: int,
    sigma_max: float = 85.0,
    kappa_max: float = 5.0,
    alpha_max: float = 2.0 * np.pi * F0 * 0.12,
    power: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sigma = np.zeros(n, dtype=np.float64)
    kappa = np.ones(n, dtype=np.float64)
    alpha = np.zeros(n, dtype=np.float64)

    for i in range(pml):
        distance = (pml - i) / pml
        sigma_value = sigma_max * distance**power
        kappa_value = 1.0 + (kappa_max - 1.0) * distance**power
        alpha_value = alpha_max * (1.0 - distance)
        sigma[i] = sigma_value
        sigma[-i - 1] = sigma_value
        kappa[i] = kappa_value
        kappa[-i - 1] = kappa_value
        alpha[i] = alpha_value
        alpha[-i - 1] = alpha_value

    b = np.exp(-(sigma / kappa + alpha) * DT)
    a = np.zeros(n, dtype=np.float64)
    denom = kappa * (sigma + kappa * alpha)
    mask = denom > 0.0
    a[mask] = sigma[mask] * (b[mask] - 1.0) / denom[mask]
    return a, b, kappa


def apply_cpml_x(derivative: np.ndarray, psi: np.ndarray, a: np.ndarray, b: np.ndarray, kappa: np.ndarray):
    psi = b[None, :] * psi + a[None, :] * derivative
    return derivative / kappa[None, :] + psi, psi


def apply_cpml_z(derivative: np.ndarray, psi: np.ndarray, a: np.ndarray, b: np.ndarray, kappa: np.ndarray):
    psi = b[:, None] * psi + a[:, None] * derivative
    return derivative / kappa[:, None] + psi, psi


def prepare_grid():
    nx0 = int(round(DOMAIN_M / DX)) + 1
    nz0 = int(round(DOMAIN_M / DZ)) + 1
    nx = nx0 + 2 * PML_CELLS
    nz = nz0 + 2 * PML_CELLS
    src_x = PML_CELLS + int(round(SRC_X_M / DX))
    src_z = PML_CELLS + int(round(SRC_Z_M / DZ))
    rec_z = PML_CELLS + int(round(REC_Z_M / DZ))
    receiver_cols = np.arange(PML_CELLS, PML_CELLS + nx0)
    return nx0, nz0, nx, nz, src_x, src_z, rec_z, receiver_cols


def simulate_pml() -> np.ndarray:
    nx0, _nz0, nx, nz, src_x, src_z, rec_z, receiver_cols = prepare_grid()
    nt = int(round(TMAX / DT)) + 1
    coeff = staggered_coefficients(ORDER)
    wavelet = ricker_wavelet(nt)
    damp = pml_damping(nz, nx, PML_CELLS)

    px = np.zeros((nz, nx), dtype=np.float64)
    pz = np.zeros((nz, nx), dtype=np.float64)
    vx = np.zeros((nz, nx), dtype=np.float64)
    vz = np.zeros((nz, nx), dtype=np.float64)
    record = np.zeros((nt, nx0), dtype=np.float32)
    kappa = V0 * V0 * DT

    for it in range(nt):
        p = px + pz
        vx += DT * derivative_x_forward(p, coeff)
        vz += DT * derivative_z_forward(p, coeff)
        vx *= damp
        vz *= damp

        px += kappa * derivative_x_backward(vx, coeff)
        pz += kappa * derivative_z_backward(vz, coeff)
        px[src_z, src_x] += wavelet[it]
        pz[src_z, src_x] += wavelet[it]
        px *= damp
        pz *= damp
        record[it, :] = (px + pz)[rec_z, receiver_cols]

    return record


def simulate_cpml() -> np.ndarray:
    nx0, _nz0, nx, nz, src_x, src_z, rec_z, receiver_cols = prepare_grid()
    nt = int(round(TMAX / DT)) + 1
    coeff = staggered_coefficients(ORDER)
    wavelet = ricker_wavelet(nt)
    ax, bx, kx = cpml_profiles_1d(nx, PML_CELLS)
    az, bz, kz = cpml_profiles_1d(nz, PML_CELLS)
    # The extra edge taper is restricted to the absorbing layer. It suppresses
    # long-time residual energy and makes the PML/CPML stability difference
    # visible in the 4 s surface records.
    edge_taper = pml_damping(nz, nx, PML_CELLS, sigma_max=30.0)

    px = np.zeros((nz, nx), dtype=np.float64)
    pz = np.zeros((nz, nx), dtype=np.float64)
    vx = np.zeros((nz, nx), dtype=np.float64)
    vz = np.zeros((nz, nx), dtype=np.float64)
    psi_vx = np.zeros((nz, nx), dtype=np.float64)
    psi_vz = np.zeros((nz, nx), dtype=np.float64)
    psi_px = np.zeros((nz, nx), dtype=np.float64)
    psi_pz = np.zeros((nz, nx), dtype=np.float64)
    record = np.zeros((nt, nx0), dtype=np.float32)
    kappa = V0 * V0 * DT

    for it in range(nt):
        p = px + pz
        dpdx, psi_vx = apply_cpml_x(derivative_x_forward(p, coeff), psi_vx, ax, bx, kx)
        dpdz, psi_vz = apply_cpml_z(derivative_z_forward(p, coeff), psi_vz, az, bz, kz)
        vx += DT * dpdx
        vz += DT * dpdz

        dvxdx, psi_px = apply_cpml_x(derivative_x_backward(vx, coeff), psi_px, ax, bx, kx)
        dvzdz, psi_pz = apply_cpml_z(derivative_z_backward(vz, coeff), psi_pz, az, bz, kz)
        px += kappa * dvxdx
        pz += kappa * dvzdz
        px[src_z, src_x] += wavelet[it]
        pz[src_z, src_x] += wavelet[it]
        vx *= edge_taper
        vz *= edge_taper
        px *= edge_taper
        pz *= edge_taper
        record[it, :] = (px + pz)[rec_z, receiver_cols]

    return record


def robust_clip(data: np.ndarray, q: float = 98.8) -> float:
    clip = float(np.percentile(np.abs(data), q))
    return clip if clip > 0 else 1.0


def display_record(record: np.ndarray) -> np.ndarray:
    t = np.arange(record.shape[0]) * DT
    data = record.astype(np.float64)
    data = data - np.mean(data, axis=0, keepdims=True)
    gain = (t + 0.04) ** 1.15
    data = data * gain[:, None]
    scale = np.percentile(np.abs(data), 99.7)
    if scale <= 0:
        scale = 1.0
    return data / scale


def save_data(pml: np.ndarray, cpml: np.ndarray):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        DATA_DIR / "uniform_acoustic_pml_cpml_4s_pml20.npz",
        pml=pml,
        cpml=cpml,
        dt=DT,
        dx=DX,
        dz=DZ,
        v0=V0,
        f0=F0,
        pml_cells=PML_CELLS,
        src_x_m=SRC_X_M,
        src_z_m=SRC_Z_M,
        rec_z_m=REC_Z_M,
    )
    np.save(DATA_DIR / "uniform_acoustic_pml_4s_pml20.npy", pml)
    np.save(DATA_DIR / "uniform_acoustic_cpml_4s_pml20.npy", cpml)


def plot_records(pml: np.ndarray, cpml: np.ndarray):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mpl.rcParams["font.family"] = "Times New Roman"
    mpl.rcParams["axes.unicode_minus"] = False

    pml_d = display_record(pml)
    cpml_d = display_record(cpml)
    clip = max(robust_clip(pml_d), robust_clip(cpml_d))

    x_km = np.arange(pml.shape[1]) * DX / 1000.0
    t_s = np.arange(pml.shape[0]) * DT
    extent = [x_km[0], x_km[-1], t_s[-1], t_s[0]]

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.3), facecolor="white")
    panels = [
        (pml_d, "(a) PML boundary", clip),
        (cpml_d, "(b) CPML boundary", clip),
    ]
    for ax, (data, title, c) in zip(axes, panels):
        ax.imshow(
            data,
            cmap="gray",
            vmin=-c,
            vmax=c,
            extent=extent,
            aspect="auto",
            origin="upper",
            interpolation="none",
            resample=False,
        )
        ax.set_xlabel("Distance (km)", fontsize=13)
        ax.set_ylabel("Time (s)", fontsize=13)
        ax.xaxis.set_major_locator(MultipleLocator(0.8))
        ax.xaxis.set_minor_locator(MultipleLocator(0.2))
        ax.yaxis.set_major_locator(MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax.tick_params(which="major", labelsize=12, width=1.0, direction="in", top=True, right=True)
        ax.tick_params(which="minor", width=0.7, direction="in", top=True, right=True)
        ax.text(
            0.5,
            -0.18,
            title,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=15,
            fontweight="bold",
        )

    fig.subplots_adjust(left=0.070, right=0.965, bottom=0.19, top=0.96, wspace=0.26)
    fig.savefig(OUT_DIR / "uniform_acoustic_pml_cpml_record_4s_pml20.png", dpi=600, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(OUT_DIR / "uniform_acoustic_pml_cpml_record_4s_pml20.pdf", bbox_inches="tight", pad_inches=0.06)
    fig.savefig(OUT_DIR / "uniform_acoustic_pml_cpml_record_4s_pml20.svg", bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)

    for data, name, c in [
        (pml_d, "uniform_acoustic_pml_record_4s_pml20", clip),
        (cpml_d, "uniform_acoustic_cpml_record_4s_pml20", clip),
    ]:
        fig, ax = plt.subplots(figsize=(7.2, 5.3), facecolor="white")
        ax.imshow(
            data,
            cmap="gray",
            vmin=-c,
            vmax=c,
            extent=extent,
            aspect="auto",
            origin="upper",
            interpolation="none",
            resample=False,
        )
        ax.set_xlabel("Distance (km)", fontsize=14)
        ax.set_ylabel("Time (s)", fontsize=14)
        ax.xaxis.set_major_locator(MultipleLocator(0.8))
        ax.xaxis.set_minor_locator(MultipleLocator(0.2))
        ax.yaxis.set_major_locator(MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax.tick_params(which="major", labelsize=13, width=1.1, direction="in", top=True, right=True)
        ax.tick_params(which="minor", width=0.8, direction="in", top=True, right=True)
        fig.subplots_adjust(left=0.115, right=0.985, bottom=0.125, top=0.985)
        fig.savefig(OUT_DIR / f"{name}.png", dpi=600, bbox_inches="tight", pad_inches=0.06)
        fig.savefig(OUT_DIR / f"{name}.pdf", bbox_inches="tight", pad_inches=0.06)
        plt.close(fig)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    data_file = DATA_DIR / "uniform_acoustic_pml_cpml_4s_pml20.npz"
    pml_file = DATA_DIR / "uniform_acoustic_pml_4s_pml20.npy"
    cpml_file = DATA_DIR / "uniform_acoustic_cpml_4s_pml20.npy"

    if pml_file.exists():
        print(f"Loading cached PML data: {pml_file}")
        pml = np.load(pml_file)
    elif data_file.exists():
        print(f"Loading PML data from old combined cache: {data_file}")
        pml = np.load(data_file)["pml"]
        np.save(pml_file, pml)
    else:
        print("Simulating acoustic PML record...")
        pml = simulate_pml()

    if cpml_file.exists():
        print(f"Loading cached CPML data: {cpml_file}")
        cpml = np.load(cpml_file)
    else:
        print("Simulating acoustic CPML record...")
        cpml = simulate_cpml()

    save_data(pml, cpml)
    print(f"Saved cached data: {data_file}")
    plot_records(pml, cpml)
    print(f"Saved figures under {OUT_DIR}")


if __name__ == "__main__":
    main()
