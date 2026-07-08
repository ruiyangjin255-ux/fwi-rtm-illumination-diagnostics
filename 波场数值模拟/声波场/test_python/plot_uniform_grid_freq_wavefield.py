from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "figure"

DOMAIN_M = 4000.0
DT = 0.001
T_SNAP = 0.8
V0 = 2000.0
ORDER = 4


def staggered_coefficients(order: int) -> np.ndarray:
    if order == 2:
        return np.array([1.0], dtype=np.float64)
    if order == 4:
        return np.array([9.0 / 8.0, -1.0 / 24.0], dtype=np.float64)
    if order == 6:
        return np.array([75.0 / 64.0, -25.0 / 384.0, 3.0 / 640.0], dtype=np.float64)
    raise ValueError("order must be 2, 4, or 6")


def derivative_x_forward(a: np.ndarray, coeff: np.ndarray, dx: float) -> np.ndarray:
    d = np.zeros_like(a)
    n = len(coeff)
    idx = np.arange(n, a.shape[1] - n)
    cols = slice(n, a.shape[1] - n)
    for m, c in enumerate(coeff, start=1):
        d[:, cols] += c * (a[:, idx + m] - a[:, idx - m + 1])
    return d / dx


def derivative_x_backward(a: np.ndarray, coeff: np.ndarray, dx: float) -> np.ndarray:
    d = np.zeros_like(a)
    n = len(coeff)
    idx = np.arange(n, a.shape[1] - n)
    cols = slice(n, a.shape[1] - n)
    for m, c in enumerate(coeff, start=1):
        d[:, cols] += c * (a[:, idx + m - 1] - a[:, idx - m])
    return d / dx


def derivative_z_forward(a: np.ndarray, coeff: np.ndarray, dz: float) -> np.ndarray:
    d = np.zeros_like(a)
    n = len(coeff)
    idx = np.arange(n, a.shape[0] - n)
    rows = slice(n, a.shape[0] - n)
    for m, c in enumerate(coeff, start=1):
        d[rows, :] += c * (a[idx + m, :] - a[idx - m + 1, :])
    return d / dz


def derivative_z_backward(a: np.ndarray, coeff: np.ndarray, dz: float) -> np.ndarray:
    d = np.zeros_like(a)
    n = len(coeff)
    idx = np.arange(n, a.shape[0] - n)
    rows = slice(n, a.shape[0] - n)
    for m, c in enumerate(coeff, start=1):
        d[rows, :] += c * (a[idx + m - 1, :] - a[idx - m, :])
    return d / dz


def ricker_wavelet(nt: int, f0: float) -> np.ndarray:
    t = np.arange(nt) * DT
    t0 = 1.0 / f0
    arg = (np.pi * f0 * (t - t0)) ** 2
    return (1.0 - 2.0 * arg) * np.exp(-arg)


def cosine_taper(nz: int, nx: int, nb: int) -> np.ndarray:
    nb = max(8, min(nb, nx // 6, nz // 6))
    damp = np.ones((nz, nx), dtype=np.float64)
    edge = np.sin(np.linspace(0.0, np.pi / 2.0, nb)) ** 2
    for i, factor in enumerate(edge):
        damp[i, :] *= factor
        damp[-i - 1, :] *= factor
        damp[:, i] *= factor
        damp[:, -i - 1] *= factor
    return damp


def simulate(dx: float, f0: float, order: int = ORDER) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dz = dx
    nx = int(round(DOMAIN_M / dx)) + 1
    nz = int(round(DOMAIN_M / dz)) + 1
    nt = int(round(T_SNAP / DT))

    cfl = V0 * DT * np.sqrt(1.0 / dx**2 + 1.0 / dz**2)
    if cfl >= 0.70:
        raise ValueError(f"unstable CFL={cfl:.3f}; increase dx or reduce DT")

    coeff = staggered_coefficients(order)
    wavelet = ricker_wavelet(nt, f0)
    damp = cosine_taper(nz, nx, nb=round(300.0 / dx))

    px = np.zeros((nz, nx), dtype=np.float64)
    pz = np.zeros((nz, nx), dtype=np.float64)
    vx = np.zeros((nz, nx), dtype=np.float64)
    vz = np.zeros((nz, nx), dtype=np.float64)
    kappa = V0 * V0 * DT
    src_z = nz // 2
    src_x = nx // 2

    for it in range(nt):
        p = px + pz
        vx += DT * derivative_x_forward(p, coeff, dx)
        vz += DT * derivative_z_forward(p, coeff, dz)

        px += kappa * derivative_x_backward(vx, coeff, dx)
        pz += kappa * derivative_z_backward(vz, coeff, dz)

        px[src_z, src_x] += wavelet[it]
        pz[src_z, src_x] += wavelet[it]

        px *= damp
        pz *= damp
        vx *= damp
        vz *= damp

    field = px + pz
    field /= np.max(np.abs(field))
    x_km = np.arange(nx) * dx / 1000.0
    z_km = np.arange(nz) * dz / 1000.0
    return field, x_km, z_km


def add_right_cax(ax):
    divider = make_axes_locatable(ax)
    return divider.append_axes("right", size="3%", pad=0.04)


def setup_axis(ax):
    ax.set_xlabel("Distance (km)", fontsize=10, labelpad=5)
    ax.set_ylabel("Depth (km)", fontsize=10, labelpad=3)
    ax.xaxis.set_major_locator(MultipleLocator(0.8))
    ax.xaxis.set_minor_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(which="major", labelsize=9, width=1.1)
    ax.tick_params(which="minor", width=0.8)


def plot_group(cases, out_name: str):
    fields = []
    axes_xy = []
    for case in cases:
        print(f"Simulating {case['label']}: dx={case['dx']} m, f0={case['f0']} Hz")
        field, x_km, z_km = simulate(case["dx"], case["f0"])
        fields.append(field)
        axes_xy.append((x_km, z_km))

    mpl.rcParams["font.family"] = "Arial"
    fig, axs = plt.subplots(2, 2, figsize=(10.6, 8.2), facecolor="white")
    axs = axs.ravel()

    for ax, field, (x_km, z_km), case in zip(axs, fields, axes_xy, cases):
        extent = [x_km[0], x_km[-1], z_km[-1], z_km[0]]
        im = ax.imshow(
            field,
            cmap="seismic",
            vmin=-0.50,
            vmax=0.50,
            extent=extent,
            origin="upper",
            aspect="equal",
        )
        setup_axis(ax)
        ax.text(
            0.5,
            -0.31,
            case["label"],
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=13,
            fontweight="bold",
        )
        cbar = fig.colorbar(im, cax=add_right_cax(ax))
        cbar.set_label("Amplitude", fontsize=10)
        cbar.ax.tick_params(labelsize=9)

    fig.subplots_adjust(wspace=0.34, hspace=0.52)
    png = OUT_DIR / f"{out_name}.png"
    pdf = OUT_DIR / f"{out_name}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.03)
    print(f"Saved {png}")
    print(f"Saved {pdf}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    grid_cases = [
        {"dx": 5.0, "f0": 20.0, "label": "a. Grid spacing = 5 m"},
        {"dx": 8.0, "f0": 20.0, "label": "b. Grid spacing = 8 m"},
        {"dx": 10.0, "f0": 20.0, "label": "c. Grid spacing = 10 m"},
        {"dx": 15.0, "f0": 20.0, "label": "d. Grid spacing = 15 m"},
    ]
    plot_group(grid_cases, "uniform_grid_spacing_wavefield_0p8s")

    frequency_cases = [
        {"dx": 10.0, "f0": 10.0, "label": "a. f = 10 Hz"},
        {"dx": 10.0, "f0": 20.0, "label": "b. f = 20 Hz"},
        {"dx": 10.0, "f0": 30.0, "label": "c. f = 30 Hz"},
        {"dx": 10.0, "f0": 50.0, "label": "d. f = 50 Hz"},
    ]
    plot_group(frequency_cases, "uniform_source_frequency_wavefield_0p8s")


if __name__ == "__main__":
    main()
