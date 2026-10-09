"""Probe locations for training the surrogates (`sample_points`), and an earlier
network form kept for reference; the deployed networks are in spatialbo/nets.py.
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, "experiments"), _os.path.join(_ROOT, "training"), _os.path.join(_ROOT, "plots")):
    _p in _sys.path or _sys.path.insert(0, _p)
import numpy as np
import torch
import torch.nn as nn

from spatialbo.equalizer import Equalizer
from spatialbo.gp import DT


def _mlp(sizes):
    layers = []
    for a, b in zip(sizes[:-2], sizes[1:-1]):
        layers += [nn.Linear(a, b), nn.SiLU()]
    layers.append(nn.Linear(sizes[-2], sizes[-1]))
    return nn.Sequential(*layers)


class FaceSetField(nn.Module):
    """Earlier network form: DeepSets over the face distances."""

    def __init__(self, emb=32, use_n1d=True, use_d2=False, norm=False):
        super().__init__()
        self.use_n1d, self.use_d2, self.norm = use_n1d, use_d2, norm
        ng = 3 + int(use_n1d)
        self.phi = _mlp([2, 32, 32, emb])
        self.rho = _mlp([emb + ng + 2 + 2 * int(use_d2), 64, 64, 32, 1])
        self.double()

    def r(self, X, n, ell):
        """X (N,D) on any device; n and ell are Python scalars."""
        N, D = X.shape
        dev = X.device
        d = torch.minimum(X, 1.0 - X)
        u = torch.stack([2.0 * d, torch.log1p(d / ell)], dim=-1)
        h = self.phi(u).sum(1)
        ds = torch.sort(d, dim=1).values
        ln, le = np.log(n), np.log(ell)
        if self.norm:      # training-range standardization
            ln = (ln - 4.0) / 1.1
            le = (le - (-1.2)) / 0.5
        gvals = [ln] + ([n ** (1.0 / D)] if self.use_n1d else []) + [le, float(D)]
        gl = torch.tensor(gvals, dtype=X.dtype, device=dev).expand(N, len(gvals))
        cols = [h, gl, 2.0 * ds[:, :1], torch.log1p(ds[:, :1] / ell)]
        if self.use_d2:
            cols += [2.0 * ds[:, 1:2], torch.log1p(ds[:, 1:2] / ell)]
        return self.rho(torch.cat(cols, dim=-1)).squeeze(-1)

    def g(self, X, n, ell):
        return torch.exp(-self.r(X, n, ell)).clamp(max=1.0)


class FieldEnsemble:
    def __init__(self, nets):
        self.nets = nets

    def g(self, X, n, ell):
        return torch.stack([f.g(X, n, ell) for f in self.nets]).mean(0)

    def spread(self, X, n, ell):
        """Disagreement = extrapolation flag (ell >~ 1, small-n corner)."""
        return torch.stack([f.g(X, n, ell) for f in self.nets]).std(0)


# ---------------------------------------------------------------- targets --
def sample_points(M, D, gen):
    """Probe points stratified by codimension: k in {0..D} coordinates log-uniform
    near a face, the rest uniform."""
    ks = torch.randint(0, D + 1, (M,), generator=gen)
    x = torch.rand(M, D, generator=gen, dtype=DT)
    depth = 10 ** (-4 + 3.7 * torch.rand(M, D, generator=gen, dtype=DT))
    mask = torch.arange(D).expand(M, D) < ks[:, None]
    x = torch.where(mask, depth, x)
    perm = torch.argsort(torch.rand(M, D, generator=gen), dim=1)
    x = torch.gather(x, 1, perm)
    flip = torch.randint(0, 2, (M, D), generator=gen, dtype=torch.bool)
    return torch.where(flip, 1.0 - x, x).clamp(1e-4, 1 - 1e-4)


def make_node(D, n, ell, M, gen, ens_seeds=4, orbit_reps=2, reps=256, sym=64):
    """Targets at one (n, ell) node, averaged over reference ensembles and orbit representatives."""
    X = sample_points(M, D, gen)
    m = torch.zeros(M, dtype=DT)
    mc = 0.0
    for s in range(ens_seeds):
        eq = Equalizer(int(n), D, float(ell), reps=reps, sym=sym, seed=s)
        mc += eq.m_int
        for _ in range(orbit_reps):
            perm = torch.randperm(D, generator=gen)
            flip = torch.randint(0, 2, (D,), generator=gen, dtype=torch.bool)
            m += eq.profiles(torch.where(flip, 1.0 - X[:, perm],
                                         X[:, perm]))[0]
    m /= ens_seeds * orbit_reps
    mc /= ens_seeds
    return {"X": X, "n": int(n), "ell": float(ell), "m": m, "m_c": mc,
            "r": torch.log(m / mc)}


def make_dataset(D, nodes, M=1700, seed=0, **kw):
    gen = torch.Generator().manual_seed(seed)
    rows = []
    for n, ell in nodes:
        rows.append(make_node(D, n, ell, M, gen, **kw))
        print(f"  node D={D} n={n} ell={ell:.4f}: m_c={rows[-1]['m_c']:.4f}",
              flush=True)
    return rows


# ------------------------------------------------------------------ train --
def train_field(rows, seed=0, epochs=300, lr=3e-3, holdout=0.15,
                eta_g=0.0, **net_kw):
    """Fit on the r-space loss plus eta_g times the g-space loss."""
    torch.manual_seed(seed)
    net = FaceSetField(**net_kw)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    split = []
    for row in rows:
        p = torch.randperm(len(row["X"]))
        k = int(len(p) * (1 - holdout))
        split.append((row, p[:k], p[k:]))
    best, best_state = np.inf, None
    for ep in range(epochs):
        opt.zero_grad()
        loss = 0.0
        for row, tr, _ in split:
            rh = net.r(row["X"][tr], row["n"], row["ell"])
            loss = loss + ((rh - row["r"][tr]) ** 2).mean()
            if eta_g > 0:
                gh = torch.exp(-rh).clamp(max=1.0)
                gt = torch.exp(-row["r"][tr]).clamp(max=1.0)
                loss = loss + eta_g * ((gh - gt) ** 2).mean()
        loss.backward(); opt.step(); sched.step()
        if ep % 10 == 0 or ep == epochs - 1:
            with torch.no_grad():
                v = sum(((net.r(row["X"][va], row["n"], row["ell"])
                          - row["r"][va]) ** 2).mean()
                        for row, _, va in split) / len(split)
            if float(v) < best:
                best, best_state = float(v), {k2: t.clone() for k2, t in
                                              net.state_dict().items()}
    net.load_state_dict(best_state)
    return net


def report(ens, row, label):
    """Validation on g (what BO uses), with conditional slices."""
    exact = torch.minimum(torch.ones_like(row["r"]), torch.exp(-row["r"]))
    with torch.no_grad():
        approx = ens.g(row["X"], row["n"], row["ell"])
    e = (exact - approx).abs()
    def cond(th):
        msk = exact < th
        return float(e[msk].mean()) if msk.any() else float("nan")
    print(f"   {label:32s}: MAE_g {float(e.mean()):.4f}  p99 "
          f"{float(e.quantile(0.99)):.4f}  MAE|g<0.95 {cond(0.95):.4f}  "
          f"MAE|g<0.8 {cond(0.8):.4f}")
