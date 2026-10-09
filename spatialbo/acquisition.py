"""The weighted GP-LCB acquisition and its optimizer, for maximization of the
standardized objective: alpha(x) = mu(x) + lam * b(x) with

    fit_fit      sigma_fit(x)                 standard GP-LCB (sigma at the fitted lengthscale)
    fit_fix      sigma(x)                     g = 1
    fit_gfix     g_geo(x)     * sigma(x)      geometric equalization
    fit_gefffix  g_eff(x)     * sigma(x)      efficiency-oriented exploration
    fit_gufix    g_uniform(x) * sigma(x)      uniform allocation
    random       uniform draw from the candidate pool (handled by the runner)

sigma is the posterior standard deviation at the reference lengthscale with
jitter in place of the noise. The runner attaches the normalized fields to
the Surrogate as g_ref, g_eff and g_u. Optimization: 512 Sobol candidates,
the 32 best plus 10 Boltzmann-sampled starts, one stacked L-BFGS-B with
gradients through g, the best candidate as fallback. lam multiplies the
exploration term (BoTorch's beta = lam^2).
"""
import math

import numpy as np
import torch
from scipy.optimize import minimize

from spatialbo.equalizer import JITTER
from spatialbo.gp import DEVICE, DT, fit_lengthscale, matern52

ARMS = ("fit_fit", "fit_fix", "fit_gfix", "fit_gefffix", "fit_gufix", "random")
SIG_FLOOR = JITTER          # variance floor on sigma outputs (sigma >= 1e-4)
N_TOP = 32                  # deterministic starts: the N_TOP best candidates


class Surrogate:
    """The GP of one round: the fitted factorization for the mean, the
    reference-lengthscale factorization for the exploration term."""

    def __init__(self, X, y, D, noise, ell_ref, need_fit=True, device=None):
        self.X, self.y, self.D, self.noise = X, y, D, noise
        self.device = device if device is not None else DEVICE
        self.ell_ref = ell_ref
        self.ell_fit = (fit_lengthscale(X, y, noise, D=D, device=self.device)
                        if need_fit else float("nan"))
        self._chol, self._alpha = {}, None

    def _factor(self, which):
        """"fit": fitted lengthscale with the noise; "geom": reference lengthscale with jitter."""
        if which not in self._chol:
            ell, reg = {"fit": (self.ell_fit, self.noise),
                        "geom": (self.ell_ref, JITTER)}[which]
            n = self.X.shape[0]
            Xd = self.X.to(self.device)
            K = matern52(Xd, Xd, ell) + reg * torch.eye(n, dtype=DT, device=self.device)
            L = torch.linalg.cholesky(K)
            # smallest resolved eigenvalue of the round, logged as a conditioning monitor
            self.cond_floor = min(getattr(self, "cond_floor", float("inf")),
                                  float(torch.diagonal(L).min()) ** 2)
            self._chol[which] = (L, Xd, ell)
        return self._chol[which]

    def mean(self, U):
        L, Xd, ell = self._factor("fit")
        if self._alpha is None:
            self._alpha = torch.cholesky_solve(self.y.to(self.device).unsqueeze(-1), L).squeeze(-1)
        out = matern52(Xd, U.to(self.device), ell).transpose(-2, -1) @ self._alpha
        return out.to(U.device)

    def sigma(self, U, which):
        """Posterior standard deviation at U under the factorization `which`."""
        L, Xd, ell = self._factor(which)
        K = matern52(Xd, U.to(self.device), ell)
        v = 1.0 - (K * torch.cholesky_solve(K, L)).sum(0)
        return v.clamp_min(SIG_FLOOR).sqrt().to(U.device)


def alpha(arm, sur, U, lam):
    """Acquisition value of rule `arm` at the locations U."""
    mu = sur.mean(U)
    if arm == "fit_fit":
        return mu + lam * sur.sigma(U, "fit")
    if arm == "fit_fix":
        return mu + lam * sur.sigma(U, "geom")
    if arm == "fit_gfix":
        return mu + lam * sur.g_ref(U) * sur.sigma(U, "geom")
    if arm == "fit_gefffix":
        return mu + lam * sur.g_eff(U) * sur.sigma(U, "geom")
    if arm == "fit_gufix":
        return mu + lam * sur.g_u(U) * sur.sigma(U, "geom")
    raise ValueError(f"unknown arm {arm!r}")


def _boltzmann_starts(raw, a, n_start, gen, eta=1.0, n_top=None):
    """The n_top best candidates plus n_start draws with probability
    proportional to exp(eta * z), z the standardized acquisition value."""
    n_top = N_TOP if n_top is None else n_top
    z = (a - a.mean()) / a.std().clamp_min(1e-12)
    w = torch.softmax(eta * z, dim=0)
    k_top = min(n_top, raw.shape[0])
    top_idx = torch.topk(a, k_top).indices
    w = w.clone(); w[top_idx] = 0.0
    k = min(n_start, int((w > 0).sum()))
    idx = torch.multinomial(w, k, replacement=False, generator=gen) if k > 0 else top_idx[:0]
    return raw[torch.cat([top_idx, idx])]


def optimize(arm, sur, lam, D, raw, gen, n_start=10, polish=True, maxiter=200):
    """Candidates -> starts -> stacked L-BFGS-B -> best candidate as fallback.
    Returns the chosen point, shape (1, D), and a diagnostics dict."""
    with torch.no_grad():
        a = alpha(arm, sur, raw, lam)
        mu_raw = sur.mean(raw)
    i_a, i_mu = int(a.argmax()), int(mu_raw.argmax())
    best_raw = raw[i_a][None, :]
    diag = {"pick_is_mu_argmax": i_a == i_mu, "lam_eff": lam}
    if not polish:
        return best_raw, diag

    starts = _boltzmann_starts(raw, a, n_start, gen)
    k = starts.shape[0]

    def negf(z):
        u = torch.as_tensor(z, dtype=DT).reshape(k, D).requires_grad_(True)
        v = alpha(arm, sur, u, lam).sum()
        (grad,) = torch.autograd.grad(v, u)
        return -float(v.detach()), -grad.numpy().ravel()

    try:
        r = minimize(negf, starts.numpy().ravel(), jac=True, method="L-BFGS-B",
                     bounds=[(0.0, 1.0)] * (k * D), options={"maxiter": maxiter})
        cand = torch.as_tensor(r.x, dtype=DT).reshape(k, D).clamp(0.0, 1.0)
    except Exception:
        return best_raw, diag

    with torch.no_grad():
        ac = alpha(arm, sur, cand, lam)
    if float(ac.max()) > float(a.max()):
        return cand[int(ac.argmax())][None, :], diag
    return best_raw, diag
