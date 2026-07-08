from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "figure"

DOMAIN_M = 4000.0
DX = 10.0
DZ = 10.0
DT = 0.001
T_SNAP = 1.20
F0 = 20.0
V0 = 2000.0
ORDER = 4
EXTEND_CELLS = 180
CJK_FONT = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf")


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


def simulate(nx: int, nz: int, src_x: int, src_z: int) -> np.ndarray:
    nt = int(round(T_SNAP / DT))
    coeff = staggered_coefficients(ORDER)
    wavelet = ricker_wavelet(nt)

    px = np.zeros((nz, nx), dtype=np.float64)
    pz = np.zeros((nz, nx), dtype=np.float64)
    vx = np.zeros((nz, nx), dtype=np.float64)
    vz = np.zeros((nz, nx), dtype=np.float64)
    kappa = V0 * V0 * DT

    for it in range(nt):
        p = px + pz
        vx += DT * derivative_x_forward(p, coeff)
        vz += DT * derivative_z_forward(p, coeff)

        px += kappa * derivative_x_backward(vx, coeff)
        pz += kappa * derivative_z_backward(vz, coeff)

        px[src_z, src_x] += wavelet[it]
        pz[src_z, src_x] += wavelet[it]

    return px + pz


def simulate_no_boundary() -> np.ndarray:
    nx = int(round(DOMAIN_M / DX)) + 1
    nz = int(round(DOMAIN_M / DZ)) + 1
    return simulate(nx, nz, nx // 2, nz // 2)


def simulate_with_boundary() -> np.ndarray:
    nx0 = int(round(DOMAIN_M / DX)) + 1
    nz0 = int(round(DOMAIN_M / DZ)) + 1
    nx = nx0 + 2 * EXTEND_CELLS
    nz = nz0 + 2 * EXTEND_CELLS
    field = simulate(nx, nz, nx // 2, nz // 2)
    z0 = EXTEND_CELLS
    x0 = EXTEND_CELLS
    return field[z0:z0 + nz0, x0:x0 + nx0]


def add_right_cax(ax):
    divider = make_axes_locatable(ax)
    return divider.append_axes("right", size="3%", pad=0.04)


def setup_axis(ax):
    ax.set_xlabel("Distance (km)", fontsize=11, labelpad=4)
    ax.set_ylabel("Depth (km)", fontsize=11, labelpad=4)
    ax.xaxis.set_major_locator(MultipleLocator(0.8))
    ax.xaxis.set_minor_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(which="major", labelsize=10, width=1.1)
    ax.tick_params(which="minor", width=0.8)


def plot_compare():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Simulating without boundary condition...")
    no_boundary = simulate_no_boundary()
    print("Simulating with boundary condition...")
    boundary = simulate_with_boundary()

    scale = max(np.max(np.abs(no_boundary)), np.max(np.abs(boundary)))
    no_boundary = no_boundary / scale
    boundary = boundary / scale

    x_max = DOMAIN_M / 1000.0
    z_max = DOMAIN_M / 1000.0
    extent = [0.0, x_max, z_max, 0.0]

    mpl.rcParams["font.family"] = "Times New Roman"
    mpl.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.25), facecolor="white")
    cases = [
        (no_boundary, "a.未加入边界条件"),
        (boundary, "b.加入边界条件"),
    ]

    for ax, (field, label) in zip(axes, cases):
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
            -0.18,
            label,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=10.5,
            fontweight="bold",
            fontproperties=CJK_FONT,
        )
        cbar = fig.colorbar(im, cax=add_right_cax(ax))
        cbar.set_label("Amplitude", fontsize=11)
        cbar.ax.tick_params(labelsize=10)

    fig.subplots_adjust(wspace=0.30, bottom=0.16)
    png = OUT_DIR / "uniform_boundary_condition_compare_1p2s.png"
    pdf = OUT_DIR / "uniform_boundary_condition_compare_1p2s.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.03)
    print(f"Saved {png}")
    print(f"Saved {pdf}")


if __name__ == "__main__":
    plot_compare()
