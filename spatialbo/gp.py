"""Gaussian-process pieces: Matern-5/2 kernel, the reference lengthscale, and the
MAP fit of the single lengthscale under the prior of Hvarfner et al. (2024).

Random streams are drawn on the CPU, so runs are identical with and without a
GPU; BO_DEVICE=cpu forces the CPU for the linear algebra.
"""
import math
import os

import numpy as np
import torch
from scipy.optimize import minimize

DT = torch.float64


def _pick_device():
    env = os.environ.get("BO_DEVICE", "").strip().lower()
    if env:
        return torch.device(env)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


DEVICE = _pick_device()


def matern52(A, B, ell):
    """Isotropic Matern-5/2 with unit output scale; `ell` may be a 1-D tensor of R
    lengthscales, giving an (R, |A|, |B|) result."""
    d = torch.cdist(A, B)
    if torch.is_tensor(ell) and ell.ndim == 1:
        d = d.unsqueeze(0)
        ell = ell.view(-1, 1, 1)
    z = math.sqrt(5.0) * d / ell
    return (1.0 + z + z * z / 3.0) * torch.exp(-z)


# --------------------------------------------------------------------------
# the reference (exploration) lengthscale
# --------------------------------------------------------------------------
# the reference lengthscale, fixed for a run; ELL_BASE is the D = 2 value
ELL_BASE = math.sqrt(0.12 * 0.6)   # 0.26833, the warp paper's D=2 value


def spacing_lengthscale(n, D, volume=1.0, mult=1.0):
    """(V/n)^(1/D), the mean spacing of n points in volume V."""
    return mult * (volume / max(n, 1)) ** (1.0 / D)


def protocol_lengthscale(D, mult=1.0):
    """The reference lengthscale of the paper, 0.268 sqrt(D/2)."""
    return mult * ELL_BASE * math.sqrt(D / 2.0)


def exploration_lengthscale(n0, D, rule="paper", mult=1.0):
    """Reference lengthscale for a run: "paper" (fixed) or "spacing_n0" (from the initial design)."""
    if rule == "paper":
        return protocol_lengthscale(D, mult)
    if rule == "spacing_n0":
        return spacing_lengthscale(n0, D, mult=mult)
    raise ValueError(f"unknown lengthscale rule {rule!r}")


# --------------------------------------------------------------------------
# fitting (lengthscale only)
# --------------------------------------------------------------------------
def hvarfner_prior(D):
    """log(ell) ~ N(sqrt(2) + log(D)/2, sqrt(3)), Hvarfner et al. (2024)."""
    return math.sqrt(2.0) + 0.5 * math.log(D), math.sqrt(3.0)


def fit_lengthscale(X, y, noise, D=None, bounds=(0.02, 3.0), n_restart=6,
                    use_map=True, device=None):
    """MAP fit of the single lengthscale: bounded L-BFGS-B on log(ell) from
    n_restart log-spaced starts. Output scale, prior mean and noise are fixed."""
    # CPU: ~130 small sequential solves, latency-bound on a GPU
    dev = device if device is not None else torch.device("cpu")
    lo, hi = bounds
    n = X.shape[0]
    D = D if D is not None else X.shape[1]
    mu0, sd0 = hvarfner_prior(D)
    Xd, yd = X.to(dev), y.to(dev)
    I = noise * torch.eye(n, dtype=DT, device=dev)

    def nmap(z):
        ell = float(np.clip(math.exp(z[0]), lo, hi))
        K = matern52(Xd, Xd, ell) + I
        L = torch.linalg.cholesky(K)
        a = torch.cholesky_solve(yd.unsqueeze(-1), L).squeeze(-1)
        v = 0.5 * float(yd @ a) + float(torch.log(torch.diagonal(L)).sum())
        if use_map:            # -log p(ell) for LogNormal, constants dropped
            v += math.log(ell) + (math.log(ell) - mu0) ** 2 / (2 * sd0 ** 2)
        return v

    best = None
    for k in range(n_restart):
        z0 = math.log(lo * (hi / lo) ** ((k + 0.5) / n_restart))
        r = minimize(nmap, [z0], method="L-BFGS-B",
                     bounds=[(math.log(lo), math.log(hi))],
                     options={"maxiter": 80, "ftol": 1e-14})
        if best is None or r.fun < best.fun:
            best = r
    return float(np.clip(math.exp(best.x[0]), lo, hi))
