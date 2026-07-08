from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "figure"

NX = 401
NZ = 401
DX = 10.0
DZ = 10.0
DT = 0.001
T_SNAP = 0.8
F0 = 20.0
V0 = 2000.0


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
    cols = slice(n, a.shape[1] - n)
    idx = np.arange(n, a.shape[1] - n)
    for m, c in enumerate(coeff, start=1):
        d[:, cols] += c * (a[:, idx + m] - a[:, idx - m + 1])
    return d / DX


def derivative_x_backward(a: np.ndarray, coeff: np.ndarray) -> np.ndarray:
    d = np.zeros_like(a)
    n = len(coeff)
    cols = slice(n, a.shape[1] - n)
    idx = np.arange(n, a.shape[1] - n)
    for m, c in enumerate(coeff, start=1):
        d[:, cols] += c * (a[:, idx + m - 1] - a[:, idx - m])
    return d / DX


def derivative_z_forward(a: np.ndarray, coeff: np.ndarray) -> np.ndarray:
    d = np.zeros_like(a)
    n = len(coeff)
    rows = slice(n, a.shape[0] - n)
    idx = np.arange(n, a.shape[0] - n)
    for m, c in enumerate(coeff, start=1):
        d[rows, :] += c * (a[idx + m, :] - a[idx - m + 1, :])
    return d / DZ


def derivative_z_backward(a: np.ndarray, coeff: np.ndarray) -> np.ndarray:
    d = np.zeros_like(a)
    n = len(coeff)
    rows = slice(n, a.shape[0] - n)
    idx = np.arange(n, a.shape[0] - n)
    for m, c in enumerate(coeff, start=1):
        d[rows, :] += c * (a[idx + m - 1, :] - a[idx - m, :])
    return d / DZ


def ricker_wavelet(nt: int) -> np.ndarray:
    t = np.arange(nt) * DT
    t0 = 1.0 / F0
    arg = (np.pi * F0 * (t - t0)) ** 2
    return (1.0 - 2.0 * arg) * np.exp(-arg)


def cosine_taper(nz: int, nx: int, nb: int = 30) -> np.ndarray:
    damp = np.ones((nz, nx), dtype=np.float64)
    edge = np.sin(np.linspace(0.0, np.pi / 2.0, nb)) ** 2
    for i in range(nb):
        factor = edge[i]
        damp[i, :] *= factor
        damp[-i - 1, :] *= factor
        damp[:, i] *= factor
        damp[:, -i - 1] *= factor
    return damp


def simulate(order: int) -> np.ndarray:
    nt = int(round(T_SNAP / DT))
    coeff = staggered_coefficients(order)
    wavelet = ricker_wavelet(nt)
    damp = cosine_taper(NZ, NX)

    px = np.zeros((NZ, NX), dtype=np.float64)
    pz = np.zeros((NZ, NX), dtype=np.float64)
    vx = np.zeros((NZ, NX), dtype=np.float64)
    vz = np.zeros((NZ, NX), dtype=np.float64)

    src_z = NZ // 2
    src_x = NX // 2
    kappa = V0 * V0 * DT

    for it in range(nt):
        p = px + pz
        vx += DT * derivative_x_forward(p, coeff)
        vz += DT * derivative_z_forward(p, coeff)

        px += kappa * derivative_x_backward(vx, coeff)
        pz += kappa * derivative_z_backward(vz, coeff)

        px[src_z, src_x] += wavelet[it]
        pz[src_z, src_x] += wavelet[it]

        px *= damp
        pz *= damp
        vx *= damp
        vz *= damp

    field = px + pz
    field /= np.max(np.abs(field))
    return field


def add_right_cax(ax):
    divider = make_axes_locatable(ax)
    return divider.append_axes("right", size="3%", pad=0.04)


def plot_compare():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    orders = [2, 4, 6]
    labels = ["a. 2nd-order FD", "b. 4th-order FD", "c. 6th-order FD"]
    fields = [simulate(order) for order in orders]

    x_max = (NX - 1) * DX / 1000.0
    z_max = (NZ - 1) * DZ / 1000.0
    extent = [0.0, x_max, z_max, 0.0]

    mpl.rcParams["font.family"] = "Arial"
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.6), facecolor="white")

    for ax, field, label in zip(axes, fields, labels):
        im = ax.imshow(
            field,
            cmap="seismic",
            vmin=-0.50,
            vmax=0.50,
            extent=extent,
            origin="upper",
            aspect="equal",
        )
        ax.set_xlabel("Distance (km)", fontsize=12)
        ax.set_ylabel("Depth (km)", fontsize=12)
        ax.xaxis.set_major_locator(MultipleLocator(0.8))
        ax.xaxis.set_minor_locator(MultipleLocator(0.2))
        ax.yaxis.set_major_locator(MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax.tick_params(which="major", labelsize=10, width=1.2)
        ax.tick_params(which="minor", width=0.8)
        cbar = fig.colorbar(im, cax=add_right_cax(ax))
        cbar.set_label("Amplitude", fontsize=11)
        cbar.ax.tick_params(labelsize=10)

    fig.subplots_adjust(left=0.055, right=0.985, top=0.98, bottom=0.23, wspace=0.34)
    for ax, label in zip(axes, labels):
        ax.text(
            0.5,
            -0.22,
            label,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=16,
            fontweight="bold",
        )

    png = OUT_DIR / "uniform_fd246_wavefield_0p8s.png"
    pdf = OUT_DIR / "uniform_fd246_wavefield_0p8s.pdf"
    svg = OUT_DIR / "uniform_fd246_wavefield_0p8s.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(svg, bbox_inches="tight", pad_inches=0.03)
    print(f"Saved {png}")
    print(f"Saved {pdf}")
    print(f"Saved {svg}")


if __name__ == "__main__":
    plot_compare()
