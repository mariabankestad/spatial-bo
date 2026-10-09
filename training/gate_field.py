"""Certification of a trained g_geo or g_eff surrogate, without any objective
function: accuracy against an independent Monte Carlo estimate, smoothness in
x and in (N, ell), cube symmetry, selection regret on synthetic GP states,
and agreement between the two members of the ensemble. Writes JSON to
results/.

    python gate_field.py --field v2 --D 3      # v2 = g_geo, geff = g_eff
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, "experiments"), _os.path.join(_ROOT, "training"), _os.path.join(_ROOT, "plots")):
    _p in _sys.path or _sys.path.insert(0, _p)
import argparse
import itertools
import json
import math
import os
import time

import numpy as np
import torch

from spatialbo.gp import matern52, DT
from gen_field_v2 import draw_design, EPS
from equalizer_field import sample_points

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fixed = {
    "acc_mae_abs": 0.005, "acc_mae_floor": 2.0,
    "acc_q95_abs": 0.015, "acc_q95_floor": 3.0,
    "acc_max_abs": 0.03, "acc_max_floor": 5.0,
    "prominence": 1e-3, "grad_jump_ratio": 0.10,
    "k_extrema": {"v2": 0, "geff": 1},
    "n_step_jump": 0.01,
    "symmetry": 1e-12,
    "regret_abs": 0.01,          # absolute alpha units; see gate_policy
    "regret_max": 5e-3, "regret_median": 1e-3,   # relative form, reported only
}


# ---------------------------------------------------------------- fields
def load_field(kind, D):
    from spatialbo.fields import load_field as _load
    return _load({"v2": "geo", "geff": "eff"}[kind], D)


# ------------------------------------------------------------------ gold
def gold_v2(P, n, ell, D, S, seed0, chunk=128):
    """E[sigma](x_c) / E[sigma](P) over S shifted-Sobol designs."""
    Q = torch.cat([torch.full((1, D), 0.5, dtype=DT), P]).to(DEV)
    tot = torch.zeros(len(Q), dtype=DT, device=DEV)
    for c in range(S // chunk):
        Xb = torch.stack([draw_design(n, D, seed0 + c * chunk + j)
                          for j in range(chunk)]).to(DEV)
        K = matern52(Xb, Xb, ell) + EPS * torch.eye(n, dtype=DT, device=DEV)
        L = torch.linalg.cholesky(K)
        z = torch.linalg.solve_triangular(
            L, matern52(Xb, Q.expand(chunk, *Q.shape), ell), upper=False)
        tot += (1.0 - (z * z).sum(1)).clamp_min(0).sqrt().sum(0)
    m = tot / S
    return (m[0] / m[1:]).cpu()


def gold_geff(P, n, ell, D, S, seed0, chunk=16):
    """Monte Carlo g_eff = sqrt((G/G_c)/(V/V_c)) with importance-sampled labels,
    batched over `chunk` designs that share one node draw."""
    from gen_field_geff import (mixture_nodes, quad_grid, uniform_is_enough,
                                NU, N_LOC)
    Q = torch.cat([torch.full((1, D), 0.5, dtype=DT), P])
    Qg = Q.to(DEV); M = len(Q)
    totV = torch.zeros(M, dtype=DT, device=DEV); totG = torch.zeros_like(totV)
    for c in range(S // chunk):
        sd = seed0 + c * chunk
        if uniform_is_enough(ell, D, NU):
            U, q = quad_grid(D, seed=8_500_000 + sd), None
        else:
            U, q = mixture_nodes(Q, ell, D, seed=8_500_000 + sd,
                                 n_loc=N_LOC, n_uni=NU)
        Ug = U.to(DEV)
        Xb = torch.stack([draw_design(n, D, sd + j) for j in range(chunk)]).to(DEV)
        K = matern52(Xb, Xb, ell) + EPS * torch.eye(n, dtype=DT, device=DEV)
        L = torch.linalg.cholesky(K)
        zP = torch.linalg.solve_triangular(
            L, matern52(Xb, Qg.expand(chunk, M, D), ell), upper=False)
        zU = torch.linalg.solve_triangular(
            L, matern52(Xb, Ug.expand(chunk, len(U), D), ell), upper=False)
        vP = (1.0 - (zP * zP).sum(1)).clamp_min(1e-12)                  # (chunk, M)
        cpost = matern52(Ug, Qg, ell)[None] - zU.transpose(1, 2) @ zP     # (chunk, NU, M)
        integ = cpost ** 2 if q is None else cpost ** 2 / q.to(DEV)[None, :, None]
        yG = integ.mean(1) / (vP + EPS)
        totV += vP.sum(0); totG += yG.sum(0)
    V = (totV / S).cpu(); G = (totG / S).cpu()
    return torch.sqrt((G[1:] / G[0]) / (V[1:] / V[0]))


GOLD = {"v2": gold_v2, "geff": gold_geff}


# --------------------------------------------------------------- helpers
def nodes(fg, k_n, k_ell):
    """Log-spaced (n, ell) grid over the artifact's own envelope."""
    ns = np.unique(np.round(np.exp(np.linspace(
        np.log(fg.n_lo), np.log(fg.n_hi), k_n))).astype(int))
    ells = np.exp(np.linspace(np.log(fg.ell_lo), np.log(fg.ell_hi), k_ell))
    return [(int(n), float(e)) for n in ns for e in ells]


def n_extrema(y, prom):
    """Interior local extrema of a 1-D series with two-sided prominence > prom."""
    y = np.asarray(y, float)
    cnt = 0
    anchor = y[0]          # last confirmed extremum (or the start)
    ext = y[0]; ext_i = 0; direction = 0
    for i, v in enumerate(y[1:], 1):
        if direction >= 0 and v >= ext:
            ext = v; ext_i = i; direction = 1
        elif direction <= 0 and v <= ext:
            ext = v; ext_i = i; direction = -1
        elif (direction == 1 and v < ext - prom) or (direction == -1 and v > ext + prom):
            # reversal confirmed; the candidate is an extremum only if the
            # move into it also exceeded prom
            if abs(ext - anchor) > prom:
                cnt += 1; anchor = ext
            ext = v; ext_i = i; direction = -direction
    return cnt


def field_eval(fg, U, n, ell):
    with torch.no_grad():
        return fg(U, n, ell)


def grad_and_curv(fg, U, n, ell, v):
    """grad_x g (norm) and directional curvature v^T H v along unit v."""
    U = U.clone().requires_grad_(True)
    g = fg(U, n, ell)
    (gr,) = torch.autograd.grad(g.sum(), U, create_graph=True)
    dv = (gr * v).sum(1)
    (hv,) = torch.autograd.grad(dv.sum(), U)
    return gr.norm(dim=1).detach(), (hv * v).sum(1).detach()


def summarize(e):
    e = np.abs(np.asarray(e, float))
    return {"mae": float(e.mean()), "rmse": float(np.sqrt((e ** 2).mean())),
            "q95": float(np.quantile(e, 0.95)), "max": float(e.max()),
            "n": int(e.size)}


# ---------------------------------------------------------------- gate 1
def gate_accuracy(fg, kind, D, S, M, k_n, k_ell):
    gold = GOLD[kind]
    gen = torch.Generator().manual_seed(2026)
    rows = []           # (d/h, ell/h, err, half-diff)
    for i, (n, ell) in enumerate(nodes(fg, k_n, k_ell)):
        P = sample_points(M, D, gen)
        h = n ** (-1.0 / D)
        g1 = gold(P, n, ell, D, S // 2, 3_000_000 + 10_000 * i)
        g2 = gold(P, n, ell, D, S // 2, 4_000_000 + 10_000 * i)
        gg = 0.5 * (g1 + g2)
        gf = field_eval(fg, P, n, ell)
        d = torch.minimum(P, 1 - P).min(1).values
        for a, b, c, dd in zip((gf - gg).tolist(), (0.5 * (g1 - g2)).tolist(),
                               d.tolist(), [ell / h] * M):
            rows.append((c / h, dd, a, b, 0.0, n, ell))
        print(f"  acc node {i+1}: n={n} ell={ell:.3f} ell/h={ell/h:.2f} "
              f"MAE {float((gf-gg).abs().mean()):.4f} floor "
              f"{float((0.5*(g1-g2)).pow(2).mean().sqrt()):.4f}", flush=True)
    R = np.array(rows)
    strata = {}
    dbins = [(0, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, np.inf)]
    lbins = [(0, 0.5), (0.5, 2.0), (2.0, np.inf)]
    ok = True
    # the criteria are also reported on the operational region (n <= 65, ell <= 2) and on the rest
    regions = {}
    for name, m in (("operational", (R[:, 4 + 1] <= 65) & (R[:, 4 + 2] <= 2.0)),
                    ("stress", ~((R[:, 4 + 1] <= 65) & (R[:, 4 + 2] <= 2.0)))):
        if m.sum() < 20:
            continue
        sm = summarize(R[m, 2]); fl_ = float(np.sqrt((R[m, 3] ** 2).mean()))
        F = fixed
        regions[name] = {**sm, "floor": fl_,
                         "pass": bool(sm["mae"] <= max(F["acc_mae_abs"], F["acc_mae_floor"] * fl_)
                                      and sm["q95"] <= max(F["acc_q95_abs"], F["acc_q95_floor"] * fl_)
                                      and sm["max"] <= max(F["acc_max_abs"], F["acc_max_floor"] * fl_))}
    for (d0, d1) in dbins:
        for (l0, l1) in lbins:
            m = (R[:, 0] >= d0) & (R[:, 0] < d1) & (R[:, 1] >= l0) & (R[:, 1] < l1)
            if m.sum() < 20:
                continue
            s = summarize(R[m, 2]); floor = float(np.sqrt((R[m, 3] ** 2).mean()))
            F = fixed
            passed = (s["mae"] <= max(F["acc_mae_abs"], F["acc_mae_floor"] * floor)
                      and s["q95"] <= max(F["acc_q95_abs"], F["acc_q95_floor"] * floor)
                      and s["max"] <= max(F["acc_max_abs"], F["acc_max_floor"] * floor))
            ok &= passed
            strata[f"d/h[{d0},{d1}) ell/h[{l0},{l1})"] = {**s, "floor": floor,
                                                          "pass": bool(passed)}
    return {"pass": bool(ok), "strata": strata, "regions": regions,
            "overall": {**summarize(R[:, 2]),
                        "floor": float(np.sqrt((R[:, 3] ** 2).mean()))}}


# ---------------------------------------------------------------- gate 2
def gate_spatial(fg, kind, D, k_n, k_ell, prom, n_rand=3, M=400):
    """Extrema and gradient jumps of g along lines from the center to the boundary;
    prom = max(fixed prominence, 3 x the gold noise floor)."""
    F = fixed; worst = {"extrema": 0, "extrema_1e-3": 0, "jump_ratio": 0.0,
                         "jump_ratio_wall": 0.0, "max_curv": 0.0,
                         "prominence_used": prom, "worst_node": None}
    gen = torch.Generator().manual_seed(7)
    for (n, ell) in nodes(fg, k_n, k_ell):
        dirs = [torch.eye(D, dtype=DT)[0], torch.ones(D, dtype=DT) / math.sqrt(D)]
        for _ in range(n_rand):
            v = torch.randn(D, generator=gen, dtype=DT); dirs.append(v / v.norm())
        for v in dirs:
            # center -> boundary along v in uniform steps of max(ell/50, 1/2000), ending 1e-3 short of the face
            tmax = float((0.5 - 1e-3) / v.abs().max())
            step = max(ell / 50.0, 1.0 / 2000.0)
            Mline = int(min(max(int(tmax / step) + 1, 100), 4000))
            s = torch.linspace(0.0, tmax, Mline, dtype=DT)
            U = 0.5 + s[:, None] * v[None, :]
            M = Mline
            g = field_eval(fg, U, n, ell).numpy()
            gr, cv = grad_and_curv(fg, U, n, ell, v[None, :].expand(M, D))
            gr = gr.numpy()
            ne = n_extrema(g, prom)
            if ne > worst["extrema"]:
                worst["worst_node"] = (n, ell)
            worst["extrema"] = max(worst["extrema"], ne)
            worst["extrema_1e-3"] = max(worst["extrema_1e-3"], n_extrema(g, 1e-3))
            # gradient-jump ratio on d >= 0.01; the last 1% before the face is reported separately
            d = torch.minimum(U, 1 - U).min(1).values.numpy()
            j = np.abs(np.diff(gr)); inner = d[1:] >= 0.01
            # gradients below 1 (g per unit length) count as flat: the jump is judged against 1
            scale = max(gr.max(), 1.0)
            worst["jump_ratio"] = max(worst["jump_ratio"], float(j[inner].max() / scale))
            if (~inner).any():
                worst["jump_ratio_wall"] = max(worst["jump_ratio_wall"], float(j[~inner].max() / scale))
            worst["max_curv"] = max(worst["max_curv"], float(cv.abs().max()))
    passed = (worst["extrema"] <= F["k_extrema"][kind]
              and worst["jump_ratio"] <= F["grad_jump_ratio"])
    return {"pass": bool(passed), **worst}


# ---------------------------------------------------------------- gate 3
def gate_parameter(fg, kind, D, prom, S, M=50, k_ell=400):
    """Smoothness in ell (extrema and jumps along a dense log sweep) and in N
    (the field's step between consecutive N against the gold's step)."""
    F = fixed; gold = GOLD[kind]
    gen = torch.Generator().manual_seed(11)
    P = sample_points(M, D, gen)
    ells = np.exp(np.linspace(np.log(fg.ell_lo), np.log(fg.ell_hi), k_ell))
    worst = {"extrema_ell": 0, "extrema_ell_1e-3": 0, "jump_ratio_ell": 0.0,
             "n_step": [], "prominence_used": prom}
    for n in np.unique(np.round(np.exp(np.linspace(
            np.log(fg.n_lo), np.log(fg.n_hi), 4))).astype(int)):
        G = np.stack([field_eval(fg, P, int(n), float(e)).numpy() for e in ells])
        for j in range(M):
            worst["extrema_ell"] = max(worst["extrema_ell"], n_extrema(G[:, j], prom))
            worst["extrema_ell_1e-3"] = max(worst["extrema_ell_1e-3"],
                                            n_extrema(G[:, j], 1e-3))
            dg = np.diff(G[:, j]) / np.diff(np.log(ells))
            worst["jump_ratio_ell"] = max(
                worst["jump_ratio_ell"],
                float(np.abs(np.diff(dg)).max() / max(np.abs(dg).max(), 1e-9)))
    # ell-extrema are judged RELATIVE TO GOLD at the field's worst point:
    # the estimand itself is not unimodal in ell, so an absolute count was wrong.
    n_mid = int(math.sqrt(fg.n_lo * fg.n_hi))
    Gm = np.stack([field_eval(fg, P, n_mid, float(e)).numpy() for e in ells])
    jw = max(range(M), key=lambda j: n_extrema(Gm[:, j], prom))
    ell_g = np.exp(np.linspace(np.log(fg.ell_lo), np.log(fg.ell_hi), 40))
    gg = np.array([float(gold(P[jw:jw + 1], n_mid, float(e), D, max(S // 4, 64),
                              9_500_000 + k)) for k, e in enumerate(ell_g)])
    gf = np.array([float(field_eval(fg, P[jw:jw + 1], n_mid, float(e))) for e in ell_g])
    worst["ell_extrema_field_vs_gold"] = (n_extrema(gf, prom), n_extrema(gg, prom))
    ell_ok = n_extrema(gf, prom) <= max(n_extrema(gg, prom), 1)
    n_ok = True
    for ell in np.exp(np.linspace(np.log(fg.ell_lo), np.log(fg.ell_hi), 5)):
        prev = None; best = (0.0, None, None)
        for n in range(fg.n_lo, min(fg.n_hi, fg.n_lo + 60) + 1):
            g = field_eval(fg, P, n, float(ell)).numpy()
            if prev is not None:
                j = int(np.abs(g - prev).argmax()); jmp = float(abs(g[j] - prev[j]))
                if jmp > best[0]:
                    best = (jmp, n, j)
            prev = g
        jmp, n, j = best
        # gold N-step from two independent batches; pass if field step <= max(0.005, 2 |gold step| + 2 floor)
        gA = [gold(P[j:j + 1], m, float(ell), D, S // 2, sd)
              for m in (n - 1, n) for sd in (9_100_000 + m, 9_600_000 + m)]
        gs1 = float(gA[2] - gA[0]); gs2 = float(gA[3] - gA[1])
        gs = 0.5 * (gs1 + gs2); gfloor = 0.5 * abs(gs1 - gs2)
        ok = jmp <= max(0.005, 2.0 * abs(gs) + 2.0 * gfloor)
        n_ok &= ok
        worst["n_step"].append({"ell": float(ell), "n": n, "field": jmp,
                                "gold": gs, "gold_floor": gfloor, "pass": bool(ok)})
    passed = (ell_ok and worst["jump_ratio_ell"] <= F["grad_jump_ratio"] and n_ok)
    return {"pass": bool(passed), **worst}


# ---------------------------------------------------------------- gate 6
def gate_replicate(fg, kind, D, k_n, k_ell, M=200, pool_n=512):
    """Agreement between the two members of the ensemble: max |g_a - g_b| <= 0.01
    and pure-exploration regret between them below the thresholds."""
    if len(fg.nets) < 2:
        return {"pass": None, "skipped": "single-member field"}
    net = fg.nets[1]
    gen = torch.Generator().manual_seed(23)
    worst = 0.0; regs = []
    for i, (n, ell) in enumerate(nodes(fg, k_n, k_ell)):
        P = sample_points(M, D, gen)
        ln = torch.full((M,), math.log(n), dtype=DT); le = torch.full((M,), math.log(ell), dtype=DT)
        with torch.no_grad():
            ga = fg(P, n, ell); gb = net.g(P, ln, le)
        worst = max(worst, float((ga - gb).abs().max()))
        X = draw_design(n, D, 700_000 + i)
        pool = torch.quasirandom.SobolEngine(D, scramble=True, seed=700_007 + i
                                             ).draw(pool_n).to(DT).clamp(1e-4, 1 - 1e-4)
        K = matern52(X, X, ell) + EPS * torch.eye(n, dtype=DT); L = torch.linalg.cholesky(K)
        Kp = matern52(X, pool, ell)
        sig = (1.0 - (Kp * torch.cholesky_solve(Kp, L)).sum(0)).clamp_min(1e-16).sqrt()
        lnp = torch.full((pool_n,), math.log(n), dtype=DT); lep = torch.full((pool_n,), math.log(ell), dtype=DT)
        with torch.no_grad():
            sa = fg(pool, n, ell) * sig; sb = net.g(pool, lnp, lep) * sig
        regs.append(float(sa[int(sa.argmax())] - sa[int(sb.argmax())]))   # absolute, sigma-units
    r = np.array(regs)
    ok = worst <= 0.01 and r.max() <= fixed["regret_abs"] and np.median(r) <= 1e-4
    return {"pass": bool(ok), "max_abs_diff": worst, "regret_abs_max": float(r.max()),
            "regret_abs_median": float(np.median(r))}


# ---------------------------------------------------------------- gate 4
def gate_symmetry(fg, D, M=2000):
    gen = torch.Generator().manual_seed(5)
    U = torch.rand(M, D, generator=gen, dtype=DT)
    worst = 0.0
    for (n, ell) in ((fg.n_lo, fg.ell_lo), (fg.n_hi, fg.ell_hi),
                     (int(math.sqrt(fg.n_lo * fg.n_hi)), math.sqrt(fg.ell_lo * fg.ell_hi))):
        base = field_eval(fg, U, n, ell)
        perms = (itertools.permutations(range(D)) if D <= 4
                 else [tuple(range(D))[::-1], tuple(range(1, D)) + (0,)])
        for p in perms:
            worst = max(worst, float((field_eval(fg, U[:, list(p)], n, ell) - base).abs().max()))
        for mask in itertools.product([0, 1], repeat=D):
            m = torch.tensor(mask, dtype=DT)
            worst = max(worst, float((field_eval(fg, torch.where(m > 0, 1 - U, U), n, ell) - base).abs().max()))
    return {"pass": bool(worst <= fixed["symmetry"]), "max_dev": worst}


# ---------------------------------------------------------------- gate 5
def stress_design(n, D, seed):
    """Sobol initial design (2D+2) + the rest in a few tight clusters:
    an off-reference geometry with holes, NOT a model of BO states."""
    g = torch.Generator().manual_seed(seed)
    n0 = min(2 * D + 2, n)
    X = torch.quasirandom.SobolEngine(D, scramble=True, seed=seed).draw(n0).to(DT)
    k = max(1, (n - n0) // 6)
    centres = torch.rand(k, D, generator=g, dtype=DT)
    idx = torch.randint(k, (n - n0,), generator=g)
    extra = centres[idx] + 0.04 * torch.randn(n - n0, D, generator=g, dtype=DT)
    return torch.cat([X, extra]).clamp(1e-3, 1 - 1e-3)


def gate_policy(fg, kind, D, S, k_n, k_ell, n_states, lam=2.0, pool_n=512):
    gold = GOLD[kind]
    out = {}
    for cls in ("reference", "stress"):
        reg = {"explore": [], "posterior": []}; flips = {"explore": 0, "posterior": 0}
        floors = {"explore": [], "posterior": []}; tot = 0
        reg_abs = {"explore": [], "posterior": []}; floors_abs = {"explore": [], "posterior": []}
        for i, (n, ell) in enumerate(nodes(fg, k_n, k_ell)):
            for s in range(n_states):
                seed = 600_000 + 1000 * i + s
                X = (draw_design(n, D, seed) if cls == "reference"
                     else stress_design(n, D, seed))
                pool = torch.quasirandom.SobolEngine(D, scramble=True, seed=seed + 7
                                                     ).draw(pool_n).to(DT).clamp(1e-4, 1 - 1e-4)
                K = matern52(X, X, ell) + EPS * torch.eye(n, dtype=DT)
                L = torch.linalg.cholesky(K)
                Kp = matern52(X, pool, ell)
                sig = (1.0 - (Kp * torch.cholesky_solve(Kp, L)).sum(0)).clamp_min(1e-16).sqrt()
                # synthetic posterior: y = a GP-prior sample path at X
                y = L @ torch.randn(n, generator=torch.Generator().manual_seed(seed + 3), dtype=DT)
                mu = Kp.T @ torch.cholesky_solve(y[:, None], L)[:, 0]
                g1 = gold(pool, n, ell, D, S // 2, 5_000_000 + 10_000 * i + 100 * s)
                g2 = gold(pool, n, ell, D, S // 2, 6_000_000 + 10_000 * i + 100 * s)
                gg = 0.5 * (g1 + g2)
                gf = field_eval(fg, pool, n, ell)
                for name, sc in (("explore", lambda g: g * sig),
                                 ("posterior", lambda g: mu + lam * g * sig)):
                    sg, sf = sc(gg), sc(gf)
                    rng = float(sg.max() - sg.min())
                    i_g, i_f = int(sg.argmax()), int(sf.argmax())
                    ab = float(sg[i_g] - sg[i_f])
                    reg_abs[name].append(ab)
                    reg[name].append(ab / max(rng, 1e-12))
                    flips[name] += int(i_g != i_f)
                    # gold-vs-gold regret on the same state = noise floor
                    s1, s2 = sc(g1), sc(g2)
                    fab = float(s1[int(s1.argmax())] - s1[int(s2.argmax())])
                    floors_abs[name].append(fab)
                    floors[name].append(fab / max(rng, 1e-12))
                tot += 1
        res = {}
        for name in ("explore", "posterior"):
            r = np.array(reg[name]); fl = np.array(floors[name])
            ra = np.array(reg_abs[name]); fla = np.array(floors_abs[name])
            # absolute criterion: max regret <= max(REGRET_ABS, 2 x gold-vs-gold max), median <= 2 x gold-vs-gold median
            ok = (ra.max() <= max(fixed["regret_abs"], 2.0 * fla.max())
                  and np.median(ra) <= max(1e-4, 2.0 * np.median(fla)))
            res[name] = {"abs_max": float(ra.max()), "abs_median": float(np.median(ra)),
                         "abs_q95": float(np.quantile(ra, 0.95)),
                         "abs_gold_floor_max": float(fla.max()),
                         "abs_gold_floor_median": float(np.median(fla)),
                         "rel_max": float(r.max()), "rel_median": float(np.median(r)),
                         "rel_gold_floor_max": float(fl.max()),
                         "flip_rate": flips[name] / tot, "n_states": tot,
                         "pass": bool(ok)}
        out[cls] = res
    passed = all(out[c][k]["pass"] for c in out for k in ("explore", "posterior"))
    return {"pass": bool(passed), **out}


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", choices=("v2", "geff"), required=True)
    ap.add_argument("--D", type=int, required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--gates", default="1,2,3,4,5,6")
    a = ap.parse_args()
    fg = load_field(a.field, a.D)
    env = (fg.n_lo, fg.n_hi, fg.ell_lo, fg.ell_hi)
    if a.smoke:
        S, M, k_n, k_ell, n_states = 256, 60, 3, 3, 1
    else:
        # S = 8192 designs; GATE_S overrides
        S, M, k_n, k_ell, n_states = int(os.environ.get("GATE_S", 8192)), 200, 5, 6, 3
    print(f"FIELD GATE  field={a.field} D={a.D} envelope n[{env[0]},{env[1]}] "
          f"ell[{env[2]},{env[3]}]  S={S} M={M} nodes={k_n}x{k_ell} "
          f"{'SMOKE' if a.smoke else 'FULL'}", flush=True)
    gates = set(int(x) for x in a.gates.split(","))
    rep = {"field": a.field, "D": a.D, "envelope": env, "smoke": a.smoke,
           "frozen": fixed, "meta": fg.meta}
    t0 = time.time()
    if 4 in gates:
        rep["symmetry"] = gate_symmetry(fg, a.D)
        print(f"[4 symmetry]  {rep['symmetry']}  ({time.time()-t0:.0f}s)", flush=True)
    # gate 1 runs first: its gold noise floor sets the prominence that
    # gates 2 and 3 can legitimately resolve
    prom = fixed["prominence"]
    if 1 in gates:
        rep["accuracy"] = gate_accuracy(fg, a.field, a.D, S, M, k_n, k_ell)
        print(f"[1 accuracy]  pass={rep['accuracy']['pass']} overall={rep['accuracy']['overall']}")
        for k, v in rep["accuracy"]["strata"].items():
            print(f"     {k:32s} MAE {v['mae']:.4f} q95 {v['q95']:.4f} max {v['max']:.4f} "
                  f"floor {v['floor']:.4f} n={v['n']} {'ok' if v['pass'] else 'FAIL'}")
        for k, v in rep["accuracy"].get("regions", {}).items():
            print(f"     REGION {k:12s} MAE {v['mae']:.4f} q95 {v['q95']:.4f} max {v['max']:.4f} "
                  f"floor {v['floor']:.4f} n={v['n']} {'ok' if v['pass'] else 'FAIL'}")
        print(f"  ({time.time()-t0:.0f}s)", flush=True)
        prom = max(prom, 3.0 * rep["accuracy"]["overall"]["floor"])
    if 2 in gates:
        rep["spatial"] = gate_spatial(fg, a.field, a.D, k_n, k_ell, prom)
        print(f"[2 spatial]   {rep['spatial']}  ({time.time()-t0:.0f}s)", flush=True)
    if 3 in gates:
        rep["parameter"] = gate_parameter(fg, a.field, a.D, prom, S)
        print(f"[3 parameter] {rep['parameter']}  ({time.time()-t0:.0f}s)", flush=True)
    if 6 in gates:
        rep["replicate"] = gate_replicate(fg, a.field, a.D, k_n, k_ell)
        print(f"[6 replicate] {rep['replicate']}  ({time.time()-t0:.0f}s)", flush=True)
    if 5 in gates:
        rep["policy"] = gate_policy(fg, a.field, a.D, S, k_n, k_ell, n_states)
        for cls in ("reference", "stress"):
            for nm in ("explore", "posterior"):
                v = rep["policy"][cls][nm]
                print(f"[5 policy]    {cls:9s} {nm:9s} ABS regret max {v['abs_max']:.2e} "
                      f"median {v['abs_median']:.2e} q95 {v['abs_q95']:.2e} "
                      f"(gold floor max {v['abs_gold_floor_max']:.2e}) | rel max {v['rel_max']:.2e} "
                      f"flip {v['flip_rate']:.2f} n={v['n_states']} {'ok' if v['pass'] else 'FAIL'}")
        print(f"  ({time.time()-t0:.0f}s)", flush=True)
    verdict = all(rep[k]["pass"] for k in ("symmetry", "spatial", "parameter",
                                           "accuracy", "policy", "replicate")
                  if k in rep and rep[k]["pass"] is not None)
    if rep.get("replicate", {}).get("pass") is None:
        print("  replicate gate skipped (single-member field)")
    rep["pass"] = bool(verdict)
    print(f"\nVERDICT field={a.field} D={a.D}: {'PASS' if verdict else 'FAIL'}"
          f"{' (smoke - not a certification)' if a.smoke else ''}")
    outdir = os.environ.get("GATE_OUT", os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"))
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"gate_field_{a.field}_D{a.D}{'_smoke' if a.smoke else ''}.json")
    with open(path, "w") as f:
        json.dump(rep, f, indent=1, default=str)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
