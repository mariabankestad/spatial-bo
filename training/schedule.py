"""Training schedule of the surrogates: learning-rate factor, floor anneal,
Polyak averaging over the final 40% of the steps, and a drift monitor.
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, "experiments"), _os.path.join(_ROOT, "training"), _os.path.join(_ROOT, "plots")):
    _p in _sys.path or _sys.path.insert(0, _p)
import copy
import math

import torch


def lr_factor(step, steps, schedule, peak, eta_min_legacy=1e-5, eta_min_v3=1e-6,
              tail_frac=0.15):
    """Multiplier of the peak lr at `step` (for LambdaLR)."""
    if schedule == "final":
        T1 = 0.55 * steps
        if step < T1:
            c = 0.5 * (1.0 + math.cos(math.pi * step / T1))
            return (1e-5 + (peak - 1e-5) * c) / peak
        c = 0.5 * (1.0 + math.cos(math.pi * (step - T1) / (steps - T1)))
        return (1e-6 + (1e-5 - 1e-6) * c) / peak
    if schedule == "legacy":
        c = 0.5 * (1.0 + math.cos(math.pi * step / steps))
        return (eta_min_legacy + (peak - eta_min_legacy) * c) / peak
    T = (1.0 - tail_frac) * steps
    if step >= T:
        return eta_min_v3 / peak
    c = 0.5 * (1.0 + math.cos(math.pi * step / T))
    return (eta_min_v3 + (peak - eta_min_v3) * c) / peak


def floor_at(step, steps, floors, schedule, finish_frac=0.5):
    """Relative-loss floor at `step`."""
    if schedule == "final":
        finish_frac = 0.30
    span = steps if schedule == "legacy" else finish_frac * steps
    stage = min(int(step / max(1.0, span / len(floors))), len(floors) - 1)
    return floors[stage]


class PolyakAverage:
    """Uniform average of the parameters over steps >= start (state_dict-
    serialisable for checkpoint/resume)."""
    def __init__(self, net, steps, start_frac=0.8):
        self.start = int(start_frac * steps); self.k = 0
        self.avg = {n: p.detach().clone() for n, p in net.named_parameters()}

    def update(self, net, step):
        if step < self.start:
            return
        self.k += 1
        with torch.no_grad():
            for n, p in net.named_parameters():
                self.avg[n] += (p.detach() - self.avg[n]) / self.k

    def state_dict(self):
        return {"k": self.k, "start": self.start, "avg": self.avg}

    def load_state_dict(self, d):
        self.k = d["k"]; self.start = d["start"]; self.avg = d["avg"]

    def apply(self, net):
        """Return a copy of `net` carrying the averaged weights."""
        out = copy.deepcopy(net)
        with torch.no_grad():
            for n, p in out.named_parameters():
                p.copy_(self.avg[n])
        return out


def field_drift(net_a, net_b, D, probes):
    """max and mean |g_a - g_b| over a fixed probe set [(X, ln, le), ...]."""
    diffs = []
    with torch.no_grad():
        for X, ln, le in probes:
            diffs.append((net_a.g(X, ln, le) - net_b.g(X, ln, le)).abs())
    d = torch.cat(diffs)
    return float(d.max()), float(d.mean())


def drift_probes(D, n_lo, n_hi, ell_lo, ell_hi, device, seed=99, M=400):
    """Fixed probe set for drift: 10 log-spaced (n, ell) nodes x M points,
    half codimension-stratified, half wall-layer."""
    from gen_field_corpus import sample_points
    from gen_field_v2 import wall_points
    gen = torch.Generator().manual_seed(seed)
    q = torch.quasirandom.SobolEngine(2, scramble=True, seed=seed).draw(10)
    out = []
    for k in range(10):
        n = int(round(n_lo * (n_hi / n_lo) ** float(q[k, 0]))); ell = ell_lo * (ell_hi / ell_lo) ** float(q[k, 1])
        X = torch.cat([sample_points(M // 2, D, gen), wall_points(M - M // 2, D, ell, gen)]).to(device)
        out.append((X, torch.full((M,), math.log(n), dtype=X.dtype, device=device),
                    torch.full((M,), math.log(ell), dtype=X.dtype, device=device)))
    return out
