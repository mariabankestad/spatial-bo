"""Training corpus of the g_geo surrogate: sigma(x | X_n) at probe locations for
quasi-random (n, ell) and one reference design each; shard 0 adds the
converged Monte Carlo anchors and validation nodes.

    python gen_field_corpus.py --D 3 --shard 0
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, "experiments"), _os.path.join(_ROOT, "training"), _os.path.join(_ROOT, "plots")):
    _p in _sys.path or _sys.path.insert(0, _p)
import argparse
import os
import time

import numpy as np
import torch

from equalizer_field import sample_points
from spatialbo.gp import DT, matern52

OUTDIR = os.environ.get("FIELD_DATA_DIR", "field_data")
# independent realizations average away realization error; prefer more realizations with fewer x each
N_REAL = 16000           # realizations per shard (4 shards -> 64k per D)
XPER = 200               # stratified x per realization
ELL_LO, ELL_HI = 0.05, 3.0
N_HI = 220
# 32 x 32 anchor grid, chosen from the curvature map of log m_c
ANCHOR_GRID = 32         # anchor nodes per axis
ANCHOR_RELSE = 0.003     # stop when SE(m_c)/m_c below this
GOLD_X = 600
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def draw_design(n, D, seed):
    return torch.quasirandom.SobolEngine(D, scramble=True, seed=seed
                                         ).draw(n).to(DT)


def sigma_of(Xd, U, ell):
    Xg, Ug = Xd.to(DEV), U.to(DEV)
    K = matern52(Xg, Xg, ell) + 1e-8 * torch.eye(len(Xd), dtype=DT, device=DEV)
    L = torch.linalg.cholesky(K)
    z = torch.linalg.solve_triangular(L, matern52(Xg, Ug, ell), upper=False)
    return (1.0 - (z * z).sum(0)).clamp_min(0).sqrt().cpu()


def gen_corpus(D, shard, seed0):
    gen = torch.Generator().manual_seed(1000 + seed0)
    qm = torch.quasirandom.SobolEngine(2, scramble=True, seed=90 + shard
                                       ).draw(N_REAL)
    n_lo = 2 * D + 2
    rows = {"X": [], "ln": [], "le": [], "y": []}
    for i in range(N_REAL):
        n = int(round(float(n_lo * (N_HI / n_lo) ** qm[i, 0])))
        ell = float(ELL_LO * (ELL_HI / ELL_LO) ** qm[i, 1])
        Xd = draw_design(n, D, seed0 * 100000 + i)
        U = sample_points(XPER, D, gen)
        y = sigma_of(Xd, U, ell)
        rows["X"].append(U.float()); rows["y"].append(y.float())
        rows["ln"].append(np.float32(np.log(n)))
        rows["le"].append(np.float32(np.log(ell)))
        if (i + 1) % 1000 == 0:
            print(f"  shard {shard}: {i+1}/{N_REAL}", flush=True)
    return rows


def mbar_at(points, n, ell, D, seed0, rel_se, s_chunk=128, s_max=32768):
    """E[sigma] at `points`, averaging designs in batched chunks until the relative
    standard error is below rel_se."""
    P = points.to(DEV)
    tot = torch.zeros(len(points), dtype=DT, device=DEV)
    tot2 = torch.zeros(len(points), dtype=DT, device=DEV)
    S = 0
    while S < s_max:
        Xb = torch.stack([draw_design(n, D, seed0 + S + j)
                          for j in range(s_chunk)]).to(DEV)      # (B,n,D)
        Kb = matern52(Xb, Xb, ell) + 1e-8 * torch.eye(n, dtype=DT, device=DEV)
        Lb = torch.linalg.cholesky(Kb)
        kb = matern52(Xb, P.expand(s_chunk, *P.shape), ell)      # (B,n,M)
        z = torch.linalg.solve_triangular(Lb, kb, upper=False)
        sig = (1.0 - (z * z).sum(1)).clamp_min(0).sqrt()         # (B,M)
        tot += sig.sum(0); tot2 += (sig * sig).sum(0); S += s_chunk
        m = tot / S
        se = ((tot2 / S - m ** 2).clamp_min(0) / S).sqrt()
        if float((se / m.clamp_min(1e-6)).max()) < rel_se:
            break
    return (tot / S).cpu(), S


def gen_anchors_and_gold(D):
    n_lo = 2 * D + 2
    ns = np.unique(np.round(np.geomspace(n_lo, N_HI, ANCHOR_GRID)).astype(int))
    ells = np.geomspace(ELL_LO, ELL_HI, ANCHOR_GRID)
    xc = torch.full((1, D), 0.5, dtype=DT)
    mc = np.zeros((len(ns), len(ells)))
    for i, n in enumerate(ns):
        for j, ell in enumerate(ells):
            m, S = mbar_at(xc, int(n), float(ell), D, 7_000_000 + 1000 * (i * 40 + j),
                           ANCHOR_RELSE)
            mc[i, j] = float(m)
        print(f"  anchors: n={n} done", flush=True)
    # gold nodes: the (n, ell) corners, center, edge midpoints, the dense-long regime and off-grid gaps
    gold = []
    gen = torch.Generator().manual_seed(4242)
    n_mid = int(np.sqrt(n_lo * N_HI)); l_mid = float(np.sqrt(ELL_LO * ELL_HI))
    def _gap(k, j):
        k = min(k, len(ns) - 2); j = min(j, len(ells) - 2)
        return (int(np.sqrt(ns[k] * ns[k + 1])),
                float(np.sqrt(ells[j] * ells[j + 1])))
    gaps = [_gap(3, 10), _gap(12, 18), _gap(7, 4)]
    picks = [(n_lo, ELL_LO), (n_lo, ELL_HI), (N_HI, ELL_LO), (N_HI, ELL_HI),
             (n_mid, l_mid), (n_lo, l_mid), (N_HI, l_mid), (n_mid, ELL_LO),
             (n_mid, ELL_HI), (int(0.8 * N_HI), 1.3)] + gaps
    for n, ell in picks:
        U = sample_points(GOLD_X, D, gen)
        m, S = mbar_at(U, n, ell, D, 9_000_000 + n * 1000, ANCHOR_RELSE)
        mcv, _ = mbar_at(xc, n, ell, D, 9_500_000 + n * 1000, ANCHOR_RELSE)
        gold.append({"n": n, "ell": ell, "X": U.numpy(), "m": m.numpy(),
                     "m_c": float(mcv), "S": S})
        print(f"  gold: (n={n}, ell={ell:.3f}) S={S}", flush=True)
    # holdout at off-grid (n, ell), never used to build the table
    hold_n, hold_le, hold_mc = [], [], []
    # half the holdout points sit in the measured high-curvature region
    # (small n, ell in [0.1, 0.5]) -- the gate tests where failure is likeliest
    kj_hard = [(1, 8), (2, 11), (4, 9), (3, 13)]
    kj_broad = [(10, 2), (15, 25), (20, 15), (26, 20)]
    for k, j in kj_hard + kj_broad:
        k = min(k, len(ns) - 2); j = min(j, len(ells) - 2)
        n_h = int(np.sqrt(ns[k] * ns[k + 1]))
        l_h = float(np.sqrt(ells[j] * ells[j + 1]))
        m, _ = mbar_at(xc, n_h, l_h, D, 11_000_000 + 1000 * (k * 30 + j),
                       ANCHOR_RELSE)
        hold_n.append(n_h); hold_le.append(l_h); hold_mc.append(float(m))
    return {"ns": ns, "ells": ells, "mc": mc,
            "hold_n": np.array(hold_n), "hold_ell": np.array(hold_le),
            "hold_mc": np.array(hold_mc)}, gold


def recompute_rows(D, n_max_row, rel_se=0.001):
    """Relabel the small-n anchor rows and in-range holdout points at tighter precision."""
    path = os.path.join(OUTDIR, f"anchors_D{D}.npz")
    z = {k: np.array(v) for k, v in np.load(path).items()}
    xc = torch.full((1, D), 0.5, dtype=DT)
    for i, n in enumerate(z["ns"]):
        if n > n_max_row:
            break
        for j, ell in enumerate(z["ells"]):
            m, S = mbar_at(xc, int(n), float(ell), D,
                           17_000_000 + 1000 * (i * 200 + j), rel_se,
                           s_chunk=256, s_max=262144)
            z["mc"][i, j] = float(m)
        # save per row (atomic): an interruption loses at most one row
        np.savez_compressed(path + ".tmp.npz", **z)
        os.replace(path + ".tmp.npz", path)
        print(f"  recompute: n={n} done (saved)", flush=True)
    for h, n in enumerate(z["hold_n"]):
        if n <= n_max_row:
            m, S = mbar_at(xc, int(n), float(z["hold_ell"][h]), D,
                           18_000_000 + 1000 * h, rel_se,
                           s_chunk=256, s_max=262144)
            z["hold_mc"][h] = float(m)
            print(f"  recompute holdout: (n={n}, ell={z['hold_ell'][h]:.3f})",
                  flush=True)
    np.savez_compressed(path + ".tmp.npz", **z)
    os.replace(path + ".tmp.npz", path)
    print(f"recomputed rows n<={n_max_row} of anchors_D{D}", flush=True)


def refine_anchors(D, add_n=(), add_ell=()):
    """Insert n rows or ell columns into an existing anchor table, computing only the new cells."""
    path = os.path.join(OUTDIR, f"anchors_D{D}.npz")
    z = dict(np.load(path))
    ns = list(z["ns"]); ells = list(z["ells"])
    xc = torch.full((1, D), 0.5, dtype=DT)
    new_ns = sorted(set(int(n) for n in add_n) - set(ns))
    new_ells = sorted(set(float(l) for l in add_ell) - set(ells))
    ns2 = np.array(sorted(ns + new_ns)); ells2 = np.array(sorted(ells + new_ells))
    mc2 = np.full((len(ns2), len(ells2)), np.nan)
    old_i = {n: i for i, n in enumerate(ns)}
    old_j = {l: j for j, l in enumerate(ells)}
    for i, n in enumerate(ns2):
        for j, l in enumerate(ells2):
            if int(n) in old_i and float(l) in old_j:
                mc2[i, j] = z["mc"][old_i[int(n)], old_j[float(l)]]
            else:
                m, _ = mbar_at(xc, int(n), float(l), D,
                               13_000_000 + 1000 * (i * 200 + j), ANCHOR_RELSE)
                mc2[i, j] = float(m)
        print(f"  refine: n={n} done", flush=True)
    z.update({"ns": ns2, "ells": ells2, "mc": mc2})
    # atomic replace: an interrupted refinement must never corrupt the table
    np.savez_compressed(path + ".tmp.npz", **z)
    os.replace(path + ".tmp.npz", path)
    print(f"refined anchors_D{D}: {len(ns)}x{len(ells)} -> "
          f"{len(ns2)}x{len(ells2)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--D", type=int, required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--refine", action="store_true",
                    help="local anchor refinement instead of generation")
    ap.add_argument("--recompute_rows", type=int, default=0,
                    help="re-label anchor rows with n <= this at rel SE 0.001")
    ap.add_argument("--ultra_gold", action="store_true",
                    help="generate the very-high-fidelity gold set (rel SE "
                         "0.001) used by the accuracy-cost scaling study")
    ap.add_argument("--add_n", default="", help="comma ints, e.g. 12,16")
    ap.add_argument("--add_ell", default="", help="comma floats, e.g. 0.15,0.3")
    a = ap.parse_args()
    if a.ultra_gold:
        # reference nodes at relative SE 0.001
        D = a.D
        n_lo = 2 * D + 2
        nodes = [(n_lo, 0.15), (n_lo, 1.0), (36, 0.3), (36, 1.0),
                 (110, 0.3), (110, 0.8), (60, 0.15)]
        gen = torch.Generator().manual_seed(9090)
        xc = torch.full((1, D), 0.5, dtype=DT)
        gold = []
        for n, ell in nodes:
            U = sample_points(GOLD_X, D, gen)
            m, S = mbar_at(U, n, float(ell), D, 21_000_000 + n * 1000, 0.001,
                           s_chunk=256, s_max=1_048_576)
            mcv, _ = mbar_at(xc, n, float(ell), D, 21_500_000 + n * 1000,
                             0.0007, s_chunk=256, s_max=1_048_576)
            gold.append({"n": n, "ell": float(ell), "X": U.numpy(),
                         "m": m.numpy(), "m_c": float(mcv), "S": S})
            print(f"  ultra gold: (n={n}, ell={ell}) S={S}", flush=True)
        os.makedirs(OUTDIR, exist_ok=True)
        np.savez_compressed(os.path.join(OUTDIR, f"gold_ultra_D{D}.npz"),
                            gold=np.array(gold, dtype=object))
        raise SystemExit
    if a.recompute_rows:
        recompute_rows(a.D, a.recompute_rows)
        raise SystemExit
    if a.refine:
        refine_anchors(a.D,
                       [int(v) for v in a.add_n.split(",") if v],
                       [float(v) for v in a.add_ell.split(",") if v])
        raise SystemExit
    os.makedirs(OUTDIR, exist_ok=True)
    t0 = time.time()
    seed0 = a.D * 10 + a.shard
    rows = gen_corpus(a.D, a.shard, seed0)
    out = {"D": a.D, "shard": a.shard,
           "X": np.concatenate([x.numpy() for x in rows["X"]]),
           "y": np.concatenate([y.numpy() for y in rows["y"]]),
           "ln": np.repeat(np.array(rows["ln"]), XPER),
           "le": np.repeat(np.array(rows["le"]), XPER)}
    if a.shard == 0:
        anchors, gold = gen_anchors_and_gold(a.D)
        np.savez_compressed(os.path.join(OUTDIR, f"anchors_D{a.D}.npz"),
                            **anchors)
        np.savez_compressed(os.path.join(OUTDIR, f"gold_D{a.D}.npz"),
                            gold=np.array(gold, dtype=object))
    np.savez_compressed(os.path.join(OUTDIR, f"corpus_D{a.D}_s{a.shard}.npz"),
                        **out)
    print(f"done D={a.D} shard={a.shard} ({time.time()-t0:.0f}s)", flush=True)
