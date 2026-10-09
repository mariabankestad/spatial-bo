"""Animated figure for the README: one BO run per rule on the 2-D controlled
landscape, one frame per evaluation. Writes docs/bo_rounds.gif.

    python make_readme_gif.py
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_ROOT in _sys.path or _sys.path.insert(0, _ROOT)
import math
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image

from spatialbo import FUNCS, load_field, reference_lengthscale, standardized
from spatialbo.acquisition import Surrogate, optimize
from spatialbo.functions import MB4_POS

torch.set_default_dtype(torch.float64)

FUNC, SEED, LAM, NOISE = "mb4_2d_interior", 2, 2.0, 0.0025
N0, N_TOTAL = 4, 15  # initial design and total number of evaluations shown
RULES = [("fit_fit", None, "GP-LCB", "#2a78d6"),
         ("fit_gfix", "geo", r"$g_{\mathrm{geo}}$", "#eb6834"),
         ("fit_gufix", "uniform", r"$g_{\mathrm{uniform}}$", "#eda100"),
         ("fit_gefffix", "eff", r"$g_{\mathrm{eff}}$", "#1baf7a")]
ATTR = {"geo": "g_ref", "uniform": "g_u", "eff": "g_eff"}
OUT = os.path.join(_ROOT, "docs", "bo_rounds.gif")


def run(rule, kind):
    h, D = standardized(FUNC)
    ell_e = reference_lengthscale(D)
    n0 = N0
    g = load_field(kind, D) if kind else None
    Xo = torch.quasirandom.SobolEngine(D, scramble=True, seed=100 + SEED).draw(n0).clamp(1e-3, 1 - 1e-3)
    yo = h(Xo) + math.sqrt(NOISE) * torch.randn(n0, generator=torch.Generator().manual_seed(1000 + SEED))
    noise_gen = torch.Generator().manual_seed(8000 + SEED)
    for t in range(N_TOTAL - n0):
        N = len(Xo)
        sur = Surrogate(Xo, yo, D, NOISE, ell_e)
        if g is not None:
            c = float(g(Xo, N, ell_e).mean())
            setattr(sur, ATTR[kind], lambda U, N=N, c=c: g(U, N, ell_e) / c)
        raw = torch.quasirandom.SobolEngine(D, scramble=True, seed=10000 * (SEED + 1) + t).draw(512).clamp(1e-4, 1 - 1e-4)
        xn, _ = optimize(rule, sur, LAM, D, raw, torch.Generator().manual_seed(500000 + 1000 * SEED + t))
        yn = h(xn) + math.sqrt(NOISE) * torch.randn(1, generator=noise_gen)
        Xo, yo = torch.cat([Xo, xn]), torch.cat([yo, yn])
    return Xo.numpy(), h(Xo).numpy()


def main():
    runs = {r: run(r, k) for r, k, _, _ in RULES}
    f, D = FUNCS[FUNC]
    n0 = N0
    n = 201
    grid = torch.linspace(0, 1, n)
    U = torch.stack(torch.meshgrid(grid, grid, indexing="xy"), -1).reshape(-1, 2)
    Z = f(U).reshape(n, n).numpy()
    rounds = N_TOTAL - n0
    h, _ = standardized(FUNC)
    h_star = float(h(torch.quasirandom.SobolEngine(D, scramble=True, seed=3).draw(2 ** 17)).max())

    frames = []
    for t in range(rounds + 1):
        k = n0 + t
        fig, axes = plt.subplots(1, 4, figsize=(11, 3.7), dpi=110)
        fig.subplots_adjust(left=0.012, right=0.988, top=0.80, bottom=0.085, wspace=0.07)
        for ax, (rule, kind, label, col) in zip(axes, RULES):
            X, hx = runs[rule]
            ax.imshow(Z, extent=[0, 1, 0, 1], origin="lower", cmap="Greys_r", vmin=Z.min() - 0.9,
                      vmax=Z.max() + 0.25, interpolation="bilinear")
            ax.contour(grid, grid, Z, levels=7, colors="w", linewidths=0.5, alpha=0.55)
            ax.plot(*MB4_POS[D]["interior"], "o", ms=16, mfc="none", mec="k", mew=1.0, alpha=0.6)
            if rule == "fit_fit":
                ax.annotate("global minimum", MB4_POS[D]["interior"], (0.62, 0.78), color="k", fontsize=8,
                            ha="center", arrowprops=dict(arrowstyle="-", color="k", lw=0.8))
            ax.scatter(*X[:n0].T, marker="x", s=48, c="k", lw=1.6, zorder=3, clip_on=False)
            if t > 0:
                ax.scatter(*X[n0:k].T, s=70, c=col, edgecolors="k", linewidths=0.9, zorder=4, clip_on=False)
                ax.plot(*X[k - 1], "o", ms=17, mfc="none", mec="k", mew=2.0, zorder=5, clip_on=False)
            best = hx[:k].max()
            wall = int((np.minimum(X[n0:k], 1 - X[n0:k]).min(1) < 0.05).sum())
            ax.set_title(label, color=col, fontsize=14, pad=7)
            ax.text(0.035, 0.965, f"regret  {max(h_star - best, 0):4.2f}\nat wall {wall:4d}", transform=ax.transAxes,
                    va="top", ha="left", fontsize=8.5, family="monospace", color="k",
                    bbox=dict(boxstyle="round,pad=0.35", fc="w", ec="#cccccc", lw=0.6, alpha=0.9))
            ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.03, 1.03); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor("#c8c8c8")
        fig.text(0.012, 0.955, "Where the exploration term sends BO", fontsize=13, ha="left", va="center")
        fig.text(0.988, 0.955, f"evaluation {k:2d} / {N_TOTAL}", fontsize=11, ha="right", va="center",
                 family="monospace", color="#555555")
        fig.legend(handles=[
            Line2D([], [], ls="", marker="x", mec="k", mew=1.6, ms=7, label="initial design (Sobol)"),
            Line2D([], [], ls="", marker="o", mfc="#999999", mec="k", mew=0.9, ms=8, label="evaluation"),
            Line2D([], [], ls="", marker="o", mfc="none", mec="k", mew=2.0, ms=11, label="newest evaluation"),
            Line2D([], [], ls="", marker="o", mfc="none", mec="k", mew=1.0, ms=11, alpha=0.6, label="global minimum")],
            loc="lower center", bbox_to_anchor=(0.5, -0.005), ncol=4, frameon=False, fontsize=9,
            handletextpad=0.4, columnspacing=2.5, labelcolor="#555555")
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3])
                      .quantize(colors=255, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE))
        plt.close(fig)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    durations = [2000] + [900] * (rounds - 1) + [5000]
    frames[0].save(OUT, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)
    print("wrote", OUT, f"({os.path.getsize(OUT) / 1e6:.1f} MB, {len(frames)} frames)")


if __name__ == "__main__":
    main()
