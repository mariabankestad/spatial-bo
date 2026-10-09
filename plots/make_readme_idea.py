"""Static figure for the README: the boundary inflation of the posterior
standard deviation, the weight field, and their product. Writes
docs/field_idea.png.

    python make_readme_idea.py
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_ROOT in _sys.path or _sys.path.insert(0, _ROOT)
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spatialbo import load_field, reference_lengthscale
from spatialbo.acquisition import Surrogate

torch.set_default_dtype(torch.float64)

D, N, N_DESIGNS, GRID = 2, 12, 1000, 101
KIND = "geo"
OUT = os.path.join(_ROOT, "docs", "field_idea.png")


def symmetrize(Z):
    """Average over the eight symmetries of the square (the expectation over designs is invariant)."""
    Zs = [Z, Z[::-1], Z[:, ::-1], Z[::-1, ::-1]]
    return sum(Zs + [z.T for z in Zs]) / 8


def main():
    ell = reference_lengthscale(D)
    g = load_field(KIND, D)
    grid = torch.linspace(0, 1, GRID)
    U = torch.stack(torch.meshgrid(grid, grid, indexing="xy"), -1).reshape(-1, 2)
    sig = torch.zeros(U.shape[0])
    for s in range(N_DESIGNS):
        X = torch.quasirandom.SobolEngine(D, scramble=True, seed=s).draw(N)
        sur = Surrogate(X, torch.zeros(N), D, 1e-4, ell, need_fit=False, device="cpu")
        sig += sur.sigma(U, "geom")
    sig = symmetrize((sig / N_DESIGNS).reshape(GRID, GRID).numpy())
    G = g(U, N, ell).reshape(GRID, GRID).numpy()
    prod = G * sig

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), dpi=130)
    fig.subplots_adjust(left=0.02, right=0.95, top=0.78, bottom=0.04, wspace=0.28)
    lo, hi = min(sig.min(), prod.min()), max(sig.max(), prod.max())
    panels = [(sig, r"$\sigma(x)$,  averaged over designs", "viridis", lo, hi),
              (G, rf"weight field  $g_{{\mathrm{{{KIND}}}}}(x;\,N,\ell)$", "magma", None, None),
              (prod, r"$g(x)\,\sigma(x)$", "viridis", lo, hi)]
    cbs = []
    for ax, (Z, title, cmap, vmin, vmax) in zip(axes, panels):
        im = ax.imshow(Z, extent=[0, 1, 0, 1], origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                       interpolation="bilinear")
        if Z is not prod:
            ax.contour(grid, grid, Z, levels=6, colors="w", linewidths=0.5, alpha=0.5)
        ax.set_title(title, fontsize=12, pad=8)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#c8c8c8")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=8)
        cb.outline.set_edgecolor("#c8c8c8")
        cbs.append(cb)
    for cb, ax, s in zip(cbs, axes[1:], ("×", "=")):
        x = (cb.ax.get_position().x1 + ax.get_position().x0) / 2 + 0.012
        fig.text(x, ax.get_position().y0 + ax.get_position().height / 2, s,
                 fontsize=22, ha="center", va="center", color="#555555")
    fig.text(0.5, 0.95, f"Why a weight field: the exploration term inherits a preference for the boundary "
             rf"($D$ = {D}, $N$ = {N}, $\ell$ = {ell:.2f})", fontsize=12, ha="center", va="center")
    fig.savefig(OUT)
    print("wrote", OUT, f"sigma range {sig.min():.3f}-{sig.max():.3f}, product range {prod.min():.3f}-{prod.max():.3f}")


if __name__ == "__main__":
    main()
