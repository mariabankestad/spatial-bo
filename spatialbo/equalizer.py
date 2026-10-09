"""The geometric equalizer computed by Monte Carlo.

    sigma(x | X) = sqrt(k(x, x) - k(x, X) (K + eps I)^-1 k(X, x))
    m(x) = E_X[ sigma(x | X) ]     over scrambled Sobol designs of size n, symmetrized
    g(x) = min(1, m(center) / m(x))

eps is a small jitter, not the observation noise: the exploration term
measures coverage of the input space. The neural surrogates in fields/ are
trained on this object.
"""
import torch

from spatialbo.gp import DEVICE, DT, matern52

# jitter of the exploration factorization
JITTER = 1e-8


class Equalizer:
    """g with the reference designs factorized once; differentiable in x."""

    def __init__(self, n, D, ell, reg=JITTER, reps=256, sym=16, seed=0,
                 device=None):
        self.n, self.D, self.ell, self.reps, self.sym = n, D, ell, reps, sym
        self.reg = reg
        self.device = device if device is not None else DEVICE
        # RNG on CPU so the design/symmetry streams are hardware-independent
        g = torch.Generator().manual_seed(seed)
        designs = []
        for _ in range(reps):
            s = int(torch.randint(0, 2 ** 31 - 1, (1,), generator=g))
            designs.append(torch.quasirandom.SobolEngine(
                D, scramble=True, seed=s).draw(n).to(DT))
        perms = [torch.randperm(D, generator=g) for _ in range(sym)]
        flips = [torch.randint(0, 2, (D,), generator=g).to(DT)
                 for _ in range(sym)]
        dev = self.device
        self.designs = torch.stack(designs).to(dev)              # (S, n, D)
        I = reg * torch.eye(n, dtype=DT, device=dev)
        self.chols = torch.linalg.cholesky(
            matern52(self.designs, self.designs, ell) + I)       # (S, n, n)
        self.perms = [p.to(dev) for p in perms]
        self.flips = [f.to(dev) for f in flips]
        centre = torch.full((1, D), 0.5, dtype=DT)
        ms, mv = self.profiles(centre)
        self.m_int, self.v_int = float(ms), float(mv)

    def profiles(self, U):
        """(E[sigma](U), E[sigma^2](U)) over the symmetrized ensemble."""
        dev = self.device
        Uq = U.to(dev)
        M, S = Uq.shape[0], self.sym
        Q = torch.cat([f + (1.0 - 2.0 * f) * Uq[:, p]
                       for p, f in zip(self.perms, self.flips)], dim=0)
        acc_s = torch.zeros(M * S, dtype=DT, device=dev)
        acc_v = torch.zeros(M * S, dtype=DT, device=dev)
        if dev.type == "cpu":
            chunk = 1
        else:
            chunk = max(1, min(self.reps,
                               int(512e6 // max(self.n * Q.shape[0] * 8, 1))))
        for i in range(0, self.reps, chunk):
            Xd, L = self.designs[i:i + chunk], self.chols[i:i + chunk]
            KxQ = matern52(Xd, Q.expand(Xd.shape[0], *Q.shape), self.ell)
            v = (1.0 - (KxQ * torch.cholesky_solve(KxQ, L)).sum(1)
                 ).clamp_min(self.reg)
            acc_v = acc_v + v.sum(0)
            acc_s = acc_s + v.sqrt().sum(0)
        ms = (acc_s / self.reps).reshape(S, M).mean(0)
        mv = (acc_v / self.reps).reshape(S, M).mean(0)
        return ms.to(U.device), mv.to(U.device)

    def __call__(self, U, kind="sigma"):
        """g at U from E[sigma] (kind="sigma") or from E[sigma^2] (kind="var")."""
        ms, mv = self.profiles(U)
        if kind == "sigma":
            return torch.clamp(self.m_int / ms, max=1.0)
        if kind == "var":
            return torch.clamp(self.v_int / mv, max=1.0)
        raise ValueError(f"unknown kind {kind!r}")
