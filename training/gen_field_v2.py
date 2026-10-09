"""Reference designs (random-shifted scrambled Sobol), wall-layer probes and the
converged Monte Carlo gold nodes used to train and validate the g_geo surrogate.

    python gen_field_v2.py --D 3 --gold
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, "experiments"), _os.path.join(_ROOT, "training"), _os.path.join(_ROOT, "plots")):
    _p in _sys.path or _sys.path.insert(0, _p)
import argparse
import os

import numpy as np
import torch

from spatialbo.gp import matern52
from gen_field_corpus import sample_points, N_HI, ELL_HI
from gen_field_corpus import ELL_LO as _ELL_LO_DEFAULT

DT = torch.float64
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-4                    # deployment epsilon, frozen
# the lower lengthscale edge is overridable (0.02 in D = 2); the range is written into the metadata
ELL_LO = float(os.environ.get("F2_ELL_LO", _ELL_LO_DEFAULT))
N_REAL = int(os.environ.get("N_REAL", 16000))
XPER = int(os.environ.get("XPER", 200))
OUTDIR = os.environ.get(
    "FIELD2_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "results", "field2_data"))


def draw_design(n, D, seed):
    # scramble and phase shift are independent randomizations
    Z = torch.quasirandom.SobolEngine(D, scramble=True, seed=seed
                                      ).draw(n).to(DT)
    g = torch.Generator().manual_seed(seed + 1_000_000_007)
    u = torch.rand(1, D, generator=g, dtype=DT)
    return (Z + u) % 1.0                      # toroidal phase average


def sigma_of(Xd, U, ell):
    Xg, Ug = Xd.to(DEV), U.to(DEV)
    K = matern52(Xg, Xg, ell) + EPS * torch.eye(len(Xd), dtype=DT, device=DEV)
    L = torch.linalg.cholesky(K)
    z = torch.linalg.solve_triangular(L, matern52(Xg, Ug, ell), upper=False)
    return (1.0 - (z * z).sum(0)).clamp_min(0).sqrt().cpu()


# ---------------------------------------------------------------------
# supplement sampling: (n, ell) from the sparse regions, half the probes in the wall layer
#   0.40 long ell/h band, 0.20 low-n edge, 0.20 short-ell edge, 0.20 long-ell edge
def supplement_node(D, n_lo, u, ell_lo=None, ell_hi=None):
    """(n, ell) from the sparse regions; u = three uniforms. The envelope
    is passed explicitly so the g_eff generator uses ITS envelope."""
    ell_lo = ELL_LO if ell_lo is None else ell_lo
    ell_hi = ELL_HI if ell_hi is None else ell_hi
    h_lo = N_HI ** (-1.0 / D)              # smallest spacing (largest n)
    r_max = ell_hi / h_lo                  # largest reachable ell/h
    if u[0] < 0.40:
        n = int(round(float(n_lo * (N_HI / n_lo) ** u[1])))
        r = 2.0 * (r_max / 2.0) ** float(u[2])
        ell = float(min(max(r * n ** (-1.0 / D), ell_lo), ell_hi))
    elif u[0] < 0.60:
        n = int(round(float(n_lo * 2.0 ** u[1])))
        ell = float(ell_lo * (ell_hi / ell_lo) ** u[2])
    elif u[0] < 0.80:
        n = int(round(float(n_lo * (N_HI / n_lo) ** u[1])))
        ell = float(ell_lo * 2.0 ** u[2])
    else:
        n = int(round(float(n_lo * (N_HI / n_lo) ** u[1])))
        ell = float((ell_hi / 2.0) * 2.0 ** u[2])
    return n, ell


def wall_points(M, D, ell, gen):
    """Probe points in the wall layer: depth log-uniform in [ell/100, min(3 ell, 0.5)]
    on one random face, other coordinates uniform."""
    lo, hi = ell / 100.0, min(3.0 * ell, 0.5)
    depth = lo * (hi / lo) ** torch.rand(M, generator=gen, dtype=DT)
    x = torch.rand(M, D, generator=gen, dtype=DT)
    j = torch.randint(0, D, (M,), generator=gen)
    x[torch.arange(M), j] = depth
    flip = torch.randint(0, 2, (M, D), generator=gen, dtype=torch.bool)
    return torch.where(flip, 1.0 - x, x).clamp(1e-4, 1 - 1e-4)


def long_band_node(D, n_lo, u, ell_lo=None, ell_hi=None):
    """(n, ell) with ell/h log-uniform in [2, max reachable]."""
    ell_lo = ELL_LO if ell_lo is None else ell_lo
    ell_hi = ELL_HI if ell_hi is None else ell_hi
    r_max = ell_hi / N_HI ** (-1.0 / D)
    n = int(round(float(n_lo * (N_HI / n_lo) ** u[1])))
    r = 2.0 * (r_max / 2.0) ** float(u[2])
    return n, float(min(max(r * n ** (-1.0 / D), ell_lo), ell_hi))


def gen_corpus(D, shard, seed0, supplement=False, interior=False):
    """One shard; supplement=True draws (n, ell) from the sparse regions with half
    the probes in the wall layer, interior=True from the long-ell/h band only."""
    gen = torch.Generator().manual_seed(1000 + seed0)
    qm = torch.quasirandom.SobolEngine(3, scramble=True, seed=90 + shard
                                       ).draw(N_REAL)
    n_lo = 2 * D + 2
    rows = {"X": [], "ln": [], "le": [], "y": [], "y_c": []}
    xc = torch.full((1, D), 0.5, dtype=DT)
    for i in range(N_REAL):
        if interior:
            n, ell = long_band_node(D, n_lo, qm[i])
        elif supplement:
            n, ell = supplement_node(D, n_lo, qm[i])
        else:
            n = int(round(float(n_lo * (N_HI / n_lo) ** qm[i, 0])))
            ell = float(ELL_LO * (ELL_HI / ELL_LO) ** qm[i, 1])
        Xd = draw_design(n, D, seed0 * 100000 + i)
        if supplement and not interior:
            U = torch.cat([sample_points(XPER // 2, D, gen),
                           wall_points(XPER - XPER // 2, D, ell, gen)])
        else:
            U = sample_points(XPER, D, gen)
        sig = sigma_of(Xd, torch.cat([U, xc]), ell)
        rows["X"].append(U.float())
        rows["y"].append(sig[:-1].float())
        rows["y_c"].append(np.float32(sig[-1]))
        rows["ln"].append(np.float32(np.log(n)))
        rows["le"].append(np.float32(np.log(ell)))
        if (i + 1) % 1000 == 0:
            print(f"  shard {shard}: {i+1}/{N_REAL}", flush=True)
    return rows


def mbar_at(points, n, ell, D, seed0, rel_se, s_chunk=128, s_max=32768):
    """Converged E[sigma] at `points` at the deployed eps."""
    P = points.to(DEV)
    tot = torch.zeros(len(points), dtype=DT, device=DEV)
    tot2 = torch.zeros(len(points), dtype=DT, device=DEV)
    S = 0
    while S < s_max:
        Xb = torch.stack([draw_design(n, D, seed0 + S + j)
                          for j in range(s_chunk)]).to(DEV)
        Kb = matern52(Xb, Xb, ell) + EPS * torch.eye(n, dtype=DT, device=DEV)
        Lb = torch.linalg.cholesky(Kb)
        kb = matern52(Xb, P.expand(s_chunk, *P.shape), ell)
        z = torch.linalg.solve_triangular(Lb, kb, upper=False)
        sig = (1.0 - (z * z).sum(1)).clamp_min(0).sqrt()
        tot += sig.sum(0); tot2 += (sig * sig).sum(0); S += s_chunk
        m = tot / S
        se = ((tot2 / S - m ** 2).clamp_min(0) / S).sqrt()
        if float((se / m.clamp_min(1e-6)).max()) < rel_se:
            break
    return (tot / S).cpu(), S


def gen_valgold(D, n_nodes=10, M=200, S=int(os.environ.get("VALGOLD_S", 4096))):
    """Validation nodes for the plateau rule: a Sobol draw in (log n, log ell) with
    independent realization seeds, stored as g. Never used for selection."""
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
        P = torch.cat([U, xc]).to(DEV)
        tot = torch.zeros(len(P), dtype=DT, device=DEV)
        for c in range(S // 128):
            Xb = torch.stack([draw_design(n, D, 7_700_000 + 1000 * n + c * 128 + j)
                              for j in range(128)]).to(DEV)
            Kb = matern52(Xb, Xb, ell) + EPS * torch.eye(n, dtype=DT, device=DEV)
            Lb = torch.linalg.cholesky(Kb)
            z = torch.linalg.solve_triangular(Lb, matern52(Xb, P.expand(128, *P.shape), ell), upper=False)
            tot += (1.0 - (z * z).sum(1)).clamp_min(0).sqrt().sum(0)
        m = (tot / S).cpu()
        nodes.append({"n": n, "ell": ell, "X": U.numpy(), "g": (m[-1] / m[:-1]).numpy()})
        print(f"  valgold node {k}: n={n} ell={ell:.3f}", flush=True)
    return nodes


def gen_gold(D):
    """Gold nodes at the deployed eps on the designed (n, ell) set."""
    n_lo = 2 * D + 2
    n_mid = int(np.sqrt(n_lo * N_HI)); l_mid = float(np.sqrt(ELL_LO * ELL_HI))
    picks = [(n_lo, ELL_LO), (n_lo, ELL_HI), (N_HI, ELL_LO), (N_HI, ELL_HI),
             (n_mid, l_mid), (n_lo, l_mid), (N_HI, l_mid),
             (n_mid, ELL_LO), (n_mid, ELL_HI), (int(0.8 * N_HI), 1.3)]
    gen = torch.Generator().manual_seed(4242)
    gold = []
    xc = torch.full((1, D), 0.5, dtype=DT)
    for n, ell in picks:
        U = sample_points(400, D, gen)
        # paired gold: m(x) and m_c from the same realizations
        both, S = mbar_at(torch.cat([U, xc]), n, ell, D,
                          9_000_000 + n * 1000, 0.003)
        gold.append({"n": n, "ell": ell, "X": U.numpy(),
                     "m": both[:-1].numpy(), "m_c": float(both[-1]), "S": S})
        print(f"  gold: (n={n}, ell={ell:.3f}) S={S}", flush=True)
    return gold


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--D", type=int, required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--gold", action="store_true")
    ap.add_argument("--valgold", action="store_true")
    ap.add_argument("--supplement", action="store_true",
                    help="sparse-region shard (file corpus_v2_D<D>_sup<shard>.npz)")
    ap.add_argument("--interior", action="store_true",
                    help="with --supplement: long-band (n, ell), bulk-style probes")
    a = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    if a.gold:
        gold = gen_gold(a.D)
        np.savez(os.path.join(OUTDIR, f"gold_v2_D{a.D}.npz"),
                 gold=np.array(gold, dtype=object),
                 meta=np.array({"eps": EPS}, dtype=object))
        print(f"saved gold_v2_D{a.D}.npz")
    elif a.valgold:
        nodes = gen_valgold(a.D)
        np.savez(os.path.join(OUTDIR, f"valgold_v2_D{a.D}.npz"),
                 nodes=np.array(nodes, dtype=object),
                 meta=np.array({"eps": EPS, "S": int(os.environ.get("VALGOLD_S", 4096)),
                                "seed_base": 7_700_000}, dtype=object))
        print(f"saved valgold_v2_D{a.D}.npz")
    else:
        # supplement shards use a distinct seed stream (offset 50) so their
        # designs never coincide with the bulk shards'
        seed0 = a.shard + 17 * a.D + (50 if a.supplement else 0)
        rows = gen_corpus(a.D, a.shard, seed0=seed0, supplement=a.supplement,
                          interior=a.interior)
        tag = f"sup{a.shard}" if a.supplement else f"s{a.shard}"
        np.savez(os.path.join(OUTDIR, f"corpus_v2_D{a.D}_{tag}.npz"),
                 X=np.stack(rows["X"]), y=np.stack(rows["y"]),
                 y_c=np.array(rows["y_c"]),
                 ln=np.array(rows["ln"]), le=np.array(rows["le"]),
                 meta=np.array({"D": a.D, "shard": a.shard, "eps": EPS,
                                "QN": "random-shifted scrambled Sobol (phase-averaged)",
                                "n_hi": N_HI, "ell": [ELL_LO, ELL_HI],
                                "envelope": {"n_lo": 2 * a.D + 2, "n_hi": N_HI,
                                             "ell_lo": ELL_LO, "ell_hi": ELL_HI},
                                "supplement": bool(a.supplement),
                                "spec": "field-v2 option-B "},
                               dtype=object))
        print(f"saved corpus_v2_D{a.D}_{tag}.npz")
