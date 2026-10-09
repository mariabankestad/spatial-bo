"""Training corpus of the g_eff surrogate: paired single-realization labels of

  V(x) = E[ v_X(x) ]                                        local posterior variance
  G(x) = E[ (1/|Omega|) int c_X(u, x)^2 / (v_X(x) + eps) du ]   total variance reduction

with g_eff = sqrt(G/V) normalized to 1 at the center. The domain integral is
importance-sampled around the design points where a grid cannot resolve it.

    python gen_field_geff.py --D 3 --shard 0
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, "experiments"), _os.path.join(_ROOT, "training"), _os.path.join(_ROOT, "plots")):
    _p in _sys.path or _sys.path.insert(0, _p)
import argparse
import os

import math

import numpy as np
import torch

from spatialbo.gp import matern52
from gen_field_corpus import sample_points, N_HI, ELL_LO, ELL_HI
from gen_field_v2 import draw_design, EPS

DT = torch.float64
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_REAL = int(os.environ.get("N_REAL", 16000))
XPER = int(os.environ.get("XPER", 200))
NU = int(os.environ.get("NU", 1024))      # uniform quadrature nodes
N_LOC = int(os.environ.get("N_LOC", 32))  # local nodes per query point
OUTDIR = os.environ.get(
    "FIELD3_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "results", "field3_data"))


_QBASE = {}


def quad_grid(D, seed=None):
    """Quadrature nodes for the domain integral: the base grid (seed=None) or a randomly shifted copy."""
    if D not in _QBASE:
        _QBASE[D] = torch.quasirandom.SobolEngine(
            D, scramble=True, seed=4242).draw(NU).to(DT)
    Z = _QBASE[D]
    if seed is None:
        return Z
    g = torch.Generator().manual_seed(int(seed))
    return (Z + torch.rand(1, D, generator=g, dtype=DT)) % 1.0


def _normal_box_mass(centres, sd, D):
    """P(N(c, sd^2 I) in [0,1]^D) = prod_j [Phi((1-c_j)/sd) - Phi(-c_j/sd)].
    Exact because the proposal is isotropic and the domain is a box."""
    n = torch.distributions.Normal(0.0, 1.0)
    hi = n.cdf((1.0 - centres) / sd)
    lo = n.cdf((0.0 - centres) / sd)
    return (hi - lo).clamp_min(1e-300).prod(dim=-1)


def uniform_is_enough(ell, D, n_uni, thresh=30.0):
    """Expected number of uniform nodes within 2 ell of a query point; above ~30 the
    grid resolves the integrand and importance sampling is not used."""
    r = 2.0 * float(ell)
    vol = math.pi ** (D / 2) / math.gamma(D / 2 + 1) * r ** D
    return n_uni * min(vol, 1.0) >= thresh


def mixture_nodes(P, ell, D, seed, n_loc=8, n_uni=512):
    """Importance-sampling nodes for the domain integral: an equal mixture of
    uniform points and Gaussians of width 2 ell around each query point,
    truncated to the cube, with their exact mixture density."""
    M = len(P)
    g = torch.Generator().manual_seed(int(seed))
    sd = 2.0 * float(ell)
    U_uni = torch.rand(n_uni, D, generator=g, dtype=DT)
    # n_loc candidates per query point from N(x_i, sd^2), kept if inside the cube; the pooled density
    # q(u) = [n_uni + n_loc sum_i N(u; x_i)] / [n_uni + n_loc sum_i Z_i] uses the actual counts
    cand = (P[:, None, :] + sd * torch.randn(M, n_loc, D, generator=g,
                                             dtype=DT)).reshape(-1, D)
    cand = cand[((cand >= 0) & (cand <= 1)).all(dim=1)]
    U = torch.cat([U_uni, cand])
    Z = _normal_box_mass(P, sd, D)                       # (M,)
    d2 = torch.cdist(U, P) ** 2                          # (NU, M)
    phi = torch.exp(-0.5 * d2 / sd ** 2) / (sd * math.sqrt(2 * math.pi)) ** D
    q = (n_uni + n_loc * phi.sum(dim=1)) / (n_uni + n_loc * float(Z.sum()))
    return U, q


def labels(Xd, P, U, ell, q=None):
    """(y_V, y_G) at points P for one design Xd; q = None for a grid, else importance weights."""
    Xg, Pg, Ug = Xd.to(DEV), P.to(DEV), U.to(DEV)
    K = matern52(Xg, Xg, ell) + EPS * torch.eye(len(Xd), dtype=DT, device=DEV)
    L = torch.linalg.cholesky(K)
    zP = torch.linalg.solve_triangular(L, matern52(Xg, Pg, ell), upper=False)
    zU = torch.linalg.solve_triangular(L, matern52(Xg, Ug, ell), upper=False)
    vP = (1.0 - (zP * zP).sum(0)).clamp_min(1e-12)          # (M,)
    cpost = matern52(Ug, Pg, ell) - zU.transpose(0, 1) @ zP  # (NU, M)
    integ = (cpost ** 2) if q is None else (cpost ** 2) / q.to(DEV)[:, None]
    yG = integ.mean(0) / (vP + EPS)
    return vP.cpu(), yG.cpu()


def gen_corpus(D, shard, seed0, supplement=False, interior=False):
    """One shard; supplement=True draws (n, ell) from the sparse regions with half
    the probes in the wall layer, interior=True from the long-ell/h band only."""
    from gen_field_v2 import supplement_node, wall_points, long_band_node
    gen = torch.Generator().manual_seed(1000 + seed0)
    qm = torch.quasirandom.SobolEngine(3, scramble=True, seed=90 + shard
                                       ).draw(N_REAL)
    n_lo = 2 * D + 2
    xc = torch.full((1, D), 0.5, dtype=DT)
    rows = {"X": [], "ln": [], "le": [], "yV": [], "yG": [],
            "yV_c": [], "yG_c": []}
    for i in range(N_REAL):
        if interior:
            n, ell = long_band_node(D, n_lo, qm[i], ELL_LO, ELL_HI)
            P = sample_points(XPER, D, gen)
        elif supplement:
            n, ell = supplement_node(D, n_lo, qm[i], ELL_LO, ELL_HI)
            P = torch.cat([sample_points(XPER // 2, D, gen),
                           wall_points(XPER - XPER // 2, D, ell, gen)])
        else:
            n = int(round(float(n_lo * (N_HI / n_lo) ** qm[i, 0])))
            ell = float(ELL_LO * (ELL_HI / ELL_LO) ** qm[i, 1])
            P = sample_points(XPER, D, gen)
        Xd = draw_design(n, D, seed0 * 100000 + i)
        Q = torch.cat([P, xc])
        sd = 7_000_000 + seed0 * 100000 + i
        if uniform_is_enough(ell, D, NU):
            vV, vG = labels(Xd, Q, quad_grid(D, seed=sd), ell)
        else:
            U, qd = mixture_nodes(Q, ell, D, seed=sd, n_loc=N_LOC, n_uni=NU)
            vV, vG = labels(Xd, Q, U, ell, q=qd)
        rows["X"].append(P.float())
        rows["yV"].append(vV[:-1].float()); rows["yG"].append(vG[:-1].float())
        rows["yV_c"].append(np.float32(vV[-1]))
        rows["yG_c"].append(np.float32(vG[-1]))
        rows["ln"].append(np.float32(np.log(n)))
        rows["le"].append(np.float32(np.log(ell)))
        if (i + 1) % 500 == 0:
            print(f"  shard {shard}: {i+1}/{N_REAL}", flush=True)
    return rows


def hf_at(P, n, ell, D, seed0, S=2048, chunk=32):
    """High-fidelity (V, G) at P by averaging S reference realizations."""
    totV = torch.zeros(len(P), dtype=DT)
    totG = torch.zeros(len(P), dtype=DT)
    for c in range(S // chunk):
        for j in range(chunk):
            sd = seed0 + c * chunk + j
            Xd = draw_design(n, D, sd)
            if uniform_is_enough(ell, D, NU):
                vV, vG = labels(Xd, P, quad_grid(D, seed=8_000_000 + sd), ell)
            else:
                U, qd = mixture_nodes(P, ell, D, seed=8_000_000 + sd,
                                      n_loc=N_LOC, n_uni=NU)
                vV, vG = labels(Xd, P, U, ell, q=qd)
            totV += vV; totG += vG
    return totV / S, totG / S


def gen_valgold(D, n_nodes=10, M=200, S=int(os.environ.get("VALGOLD_S", 2048))):
    """Validation nodes: 10 Sobol nodes in (log n, log ell), S=256 realizations, stored as g_eff."""
    from gen_field_v2 import wall_points
    # envelope of the ONLINE trainers (env FIELD_N_LO_MULT, FIELD_N_HI;
    # defaults 2D and 150) so the validation gold covers what they train
    n_lo = int(os.environ.get("FIELD_N_LO_MULT", 2)) * D
    N_HI_ = int(os.environ.get("FIELD_N_HI", 150))
    q = torch.quasirandom.SobolEngine(2, scramble=True, seed=4243).draw(n_nodes)
    gen = torch.Generator().manual_seed(4244)
    xc = torch.full((1, D), 0.5, dtype=DT)
    nodes = []
    for k in range(n_nodes):
        n = int(round(float(n_lo * (N_HI_ / n_lo) ** q[k, 0])))
        ell = float(ELL_LO * (ELL_HI / ELL_LO) ** q[k, 1])
        U = torch.cat([sample_points(M // 2, D, gen), wall_points(M - M // 2, D, ell, gen)])
        V, G = hf_at(torch.cat([U, xc]), n, ell, D, 7_800_000 + 1000 * n, S=S)
        g = torch.sqrt((G[:-1] / G[-1]) / (V[:-1] / V[-1]))
        nodes.append({"n": n, "ell": ell, "X": U.numpy(), "g": g.numpy()})
        print(f"  valgold node {k}: n={n} ell={ell:.3f}", flush=True)
    return nodes


def gen_gold(D):
    n_lo = 2 * D + 2
    n_mid = int(np.sqrt(n_lo * N_HI)); l_mid = float(np.sqrt(ELL_LO * ELL_HI))
    picks = [(n_lo, ELL_LO), (n_lo, ELL_HI), (N_HI, ELL_LO), (N_HI, ELL_HI),
             (n_mid, l_mid), (n_lo, l_mid), (N_HI, l_mid),
             (n_mid, ELL_LO), (n_mid, ELL_HI), (int(0.8 * N_HI), 1.3)]
    gen = torch.Generator().manual_seed(4242)
    xc = torch.full((1, D), 0.5, dtype=DT)
    gold = []
    for n, ell in picks:
        P = sample_points(300, D, gen)
        V, G = hf_at(torch.cat([P, xc]), n, ell, D, 9_000_000 + n * 1000)
        gold.append({"n": n, "ell": ell, "X": P.numpy(),
                     "V": V[:-1].numpy(), "G": G[:-1].numpy(),
                     "V_c": float(V[-1]), "G_c": float(G[-1])})
        print(f"  gold: (n={n}, ell={ell:.3f})", flush=True)
    return gold


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--D", type=int, required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--gold", action="store_true")
    ap.add_argument("--valgold", action="store_true")
    ap.add_argument("--supplement", action="store_true")
    ap.add_argument("--interior", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    if a.gold:
        np.savez(os.path.join(OUTDIR, f"gold_geff_D{a.D}.npz"),
                 gold=np.array(gen_gold(a.D), dtype=object),
                 meta=np.array({"eps": EPS, "NU": NU}, dtype=object))
        print(f"saved gold_geff_D{a.D}.npz")
    elif a.valgold:
        np.savez(os.path.join(OUTDIR, f"valgold_geff_D{a.D}.npz"),
                 nodes=np.array(gen_valgold(a.D), dtype=object),
                 meta=np.array({"eps": EPS, "S": int(os.environ.get("VALGOLD_S", 2048)),
                                "seed_base": 7_800_000}, dtype=object))
        print(f"saved valgold_geff_D{a.D}.npz")
    else:
        seed0 = a.shard + 17 * a.D + (50 if a.supplement else 0)
        r = gen_corpus(a.D, a.shard, seed0=seed0, supplement=a.supplement,
                       interior=a.interior)
        tag = f"sup{a.shard}" if a.supplement else f"s{a.shard}"
        np.savez(os.path.join(OUTDIR, f"corpus_geff_D{a.D}_{tag}.npz"),
                 X=np.stack(r["X"]), yV=np.stack(r["yV"]),
                 yG=np.stack(r["yG"]), yV_c=np.array(r["yV_c"]),
                 yG_c=np.array(r["yG_c"]), ln=np.array(r["ln"]),
                 le=np.array(r["le"]),
                 meta=np.array({"D": a.D, "shard": a.shard, "eps": EPS,
                                "NU": NU, "QN": "random-shifted Sobol",
                                "n_hi": N_HI, "ell": [ELL_LO, ELL_HI],
                                "envelope": {"n_lo": 2 * a.D + 2, "n_hi": N_HI,
                                             "ell_lo": ELL_LO, "ell_hi": ELL_HI},
                                "supplement": bool(a.supplement),
                                "spec": "g_eff field "},
                               dtype=object))
        print(f"saved corpus_geff_D{a.D}_{tag}.npz")
