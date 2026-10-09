"""The three weight fields as neural surrogates.

    g = load_field("geo", D)      # also "eff" and "uniform"; D = 2..6
    w = g(X, N, ell)              # X: (M, D) in [0,1]^D; N observations; exploration lengthscale

g(center) = 1; differentiable in X; (N, ell) outside the trained range are
clamped. One file per field and dimension, fields/<kind>/g_<kind>_D<D>.pt,
holding the member networks (two seeds for geo and eff, one for uniform);
the field is their geometric mean. training/pack_field.py writes the files.
"""
import math
import os

import numpy as np
import torch

from spatialbo.nets import DT, FieldGeff, FieldV2

FIELD_DIR = os.environ.get(
    "BO_FIELD_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fields"))
KINDS = {"geo": FieldV2, "eff": FieldGeff, "uniform": FieldV2}


def field_path(kind, D):
    return os.path.join(FIELD_DIR, kind, f"g_{kind}_D{D}.pt")


def reference_lengthscale(D):
    """The exploration lengthscale used in the paper, ell_e = 0.268 * sqrt(D/2)."""
    return math.sqrt(0.12 * 0.6) * math.sqrt(D / 2.0)


class Field:
    """A weight field: the geometric mean of its member networks."""

    def __init__(self, kind, D, path=None):
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {sorted(KINDS)}, got {kind!r}")
        if not 2 <= int(D) <= 6:
            raise ValueError(f"fields are available for D = 2..6, got D={D}")
        self.kind, self.D = kind, int(D)
        z = torch.load(path or field_path(kind, D), map_location="cpu", weights_only=False)
        self.nets = []
        for m in z["members"]:
            net = KINDS[kind](self.D, *m["stats"])
            net.load_state_dict(m["state"])
            net.eval()
            for p in net.parameters():
                p.requires_grad_(False)
            self.nets.append(net)
        self.net = self.nets[0]
        self.meta = [m.get("meta", {}) for m in z["members"]]
        env = z["envelope"]
        self.n_lo, self.n_hi = int(env["n_lo"]), int(env["n_hi"])
        self.ell_lo, self.ell_hi = float(env["ell_lo"]), float(env["ell_hi"])
        self.clamp_events = 0

    def in_envelope(self, n, ell):
        return (self.n_lo <= int(n) <= self.n_hi
                and self.ell_lo <= float(ell) <= self.ell_hi)

    def __call__(self, U, n, ell):
        n_c = min(max(int(n), self.n_lo), self.n_hi)
        e_c = min(max(float(ell), self.ell_lo), self.ell_hi)
        if n_c != n or e_c != float(ell):
            self.clamp_events += 1
        assert U.ndim == 2 and U.shape[1] == self.D, f"expected (M, {self.D}), got {tuple(U.shape)}"
        assert torch.isfinite(U).all(), "non-finite input to the field"
        dev = next(self.net.parameters()).device
        U = U.to(device=dev, dtype=DT)
        ln = torch.full((U.shape[0],), np.log(n_c), dtype=DT, device=dev)
        le = torch.full((U.shape[0],), np.log(e_c), dtype=DT, device=dev)
        if len(self.nets) == 1:
            return self.net.g(U, ln, le)
        return torch.exp(sum(torch.log(m.g(U, ln, le)) for m in self.nets) / len(self.nets))


_CACHE = {}


def load_field(kind, D):
    """The field `kind` in {"geo", "eff", "uniform"} for dimension D (cached)."""
    path = field_path(kind, D)
    key = (path, os.path.getmtime(path))
    if key not in _CACHE:
        _CACHE[key] = Field(kind, D, path)
    return _CACHE[key]


def clear_cache():
    _CACHE.clear()
