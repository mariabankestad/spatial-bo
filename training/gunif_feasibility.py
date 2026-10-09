"""Calibration of g_uniform at one (N, ell) node: a damped multiplicative
correction of log g by the ratio of the selection density (kernel estimate in
volume-PIT coordinates) to the uniform density, until argmax g * sigma is
uniform in volume over the reference designs. Writes the node field.

    python gunif_feasibility.py --D 3 --N 40 --ell 0.3
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, "experiments"), _os.path.join(_ROOT, "training"), _os.path.join(_ROOT, "plots")):
    _p in _sys.path or _sys.path.insert(0, _p)
import argparse
import json
import math
import os
import time
import multiprocessing as mp

import numpy as np
import torch
from scipy.optimize import minimize

from spatialbo.acquisition import N_TOP, _boltzmann_starts
from gen_field_v2 import EPS, draw_design, wall_points
from spatialbo.fields import field_path
from spatialbo.gp import DT, matern52
from train_field_v2 import FieldV2
from uniformity import coord_report, energy_distance, radial_ks

torch.set_default_dtype(DT)
OUTDIR = os.environ.get("GUNIF_OUT", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "gunif_feas"))
POOL_N, N_START, MAXITER = 512, 10, 200      # deployed optimizer constants


# ------------------------------------------------------- estimator (smooth)
def volume_pit(X, k=2):
    """Probability-integral transform of the k smallest face distances under the
    volume measure: (v_1..v_k) are iid U(0, 1) for uniform X, so uniformity in
    volume is flatness of their density."""
    D = X.shape[1]; k = min(k, D)
    d = torch.sort(torch.minimum(X, 1.0 - X), dim=1).values.clamp(0, 0.5 - 1e-12)
    vs = [1.0 - (1.0 - 2.0 * d[:, 0]) ** D]
    for j in range(1, k):
        vs.append(1.0 - ((0.5 - d[:, j]) / (0.5 - d[:, j - 1]).clamp_min(1e-12)).clamp(0, 1) ** (D - j))
    return torch.stack(vs, 1)


def _refl_int(v, h):
    """int_0^1 [k(v-s) + k(v+s) + k(v-2+s)] ds for k(z) = exp(-z^2/2h^2): the
    reflected-kernel density of the uniform law, in closed form."""
    Phi = lambda z: 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    return h * math.sqrt(2 * math.pi) * (Phi((1 - v) / h) - Phi(-v / h) + Phi((1 + v) / h) - Phi(v / h)
                                        + Phi((v - 1) / h) - Phi((v - 2) / h))


def kde_ratio(V_eval, V_samp, h=0.12, chunk=1024):
    """Reflected Gaussian KDE of V_samp at V_eval divided by the same estimator
    applied to the uniform law: the selection density relative to uniform."""
    def k1(e, s):
        return (torch.exp(-0.5 * ((e - s) / h) ** 2) + torch.exp(-0.5 * ((e + s) / h) ** 2)
                + torch.exp(-0.5 * ((e - 2.0 + s) / h) ** 2))
    out = []
    for a in range(0, V_eval.shape[0], chunk):
        e = V_eval[a:a + chunk]
        k = torch.ones(e.shape[0], V_samp.shape[0], dtype=DT); norm = torch.ones(e.shape[0], dtype=DT)
        for j in range(V_eval.shape[1]):
            k = k * k1(e[:, j][:, None], V_samp[:, j][None, :]); norm = norm * _refl_int(e[:, j], h)
        out.append(k.mean(1) / norm)
    return torch.cat(out)


def drive_stat(X, k=2):
    """Driving statistic of the calibration: max over the k volume-PIT coordinates
    of the KS distance to U(0, 1)."""
    V = volume_pit(torch.as_tensor(np.asarray(X), dtype=DT), k).numpy()
    n = V.shape[0]; grid = np.arange(1, n + 1) / n
    return float(max(np.abs(np.sort(V[:, j]) - grid).max() for j in range(V.shape[1])))


def drive_null(R, D, rng, reps=20, k=2):
    v = np.array([drive_stat(rng.random((R, D)), k) for _ in range(reps)])
    return float(np.quantile(v, 0.95)), float(v.std())


# ------------------------------------------------------------------ bins
class FaceBins:
    """Equal-volume quantile bins over (d_(1), d_(2)), the two smallest face
    distances. u_i (volume fractions) measured by Monte Carlo, not assumed."""
    def __init__(self, D, nb1=6, nb2=4, M=400_000, seed=0):
        g = torch.Generator().manual_seed(seed)
        X = torch.rand(M, D, generator=g, dtype=DT)
        d = self.coords(X)
        q1 = torch.quantile(d[:, 0], torch.linspace(0, 1, nb1 + 1, dtype=DT))
        q1[0], q1[-1] = -1.0, 2.0
        self.q1 = q1
        self.q2 = []
        for i in range(nb1):
            sel = (d[:, 0] > q1[i]) & (d[:, 0] <= q1[i + 1])
            q2 = torch.quantile(d[sel, 1], torch.linspace(0, 1, nb2 + 1, dtype=DT))
            q2[0], q2[-1] = -1.0, 2.0
            self.q2.append(q2)
        self.nb1, self.nb2, self.K = nb1, nb2, nb1 * nb2
        self.u = torch.bincount(self.index(X), minlength=self.K).to(DT) / M

    @staticmethod
    def coords(X):
        d = torch.minimum(X, 1.0 - X)
        s = torch.sort(d, dim=1).values
        return s[:, :2] if X.shape[1] >= 2 else torch.cat([s, s], 1)

    def index(self, X):
        d = self.coords(X)
        i1 = torch.bucketize(d[:, 0].contiguous(), self.q1[1:-1], right=False).clamp(0, self.nb1 - 1)
        i2 = torch.zeros_like(i1)
        for i in range(self.nb1):
            sel = i1 == i
            if sel.any():
                i2[sel] = torch.bucketize(d[sel, 1].contiguous(), self.q2[i][1:-1], right=False).clamp(0, self.nb2 - 1)
        return i1 * self.nb2 + i2


# ------------------------------------------------------------ sigma, argmax
def sigma_factory(Xd, ell):
    K = matern52(Xd, Xd, ell) + EPS * torch.eye(len(Xd), dtype=DT)
    L = torch.linalg.cholesky(K)

    def sigma(U):
        z = torch.linalg.solve_triangular(L, matern52(Xd, U, ell), upper=False)
        return (1.0 - (z * z).sum(0)).clamp_min(1e-16).sqrt()
    return sigma


_NET = None       # per-worker field, set by the pool initializer (spawn context:
                  # a fork after torch has used OpenMP deadlocks, measured )
_NODE = None


def set_net(state, D, stats, ln, le):
    global _NET, _NODE
    torch.set_num_threads(1)
    _NET = FieldV2(D, *stats); _NET.load_state_dict(state); _NET.eval()
    for p in _NET.parameters():
        p.requires_grad_(False)
    _NODE = (ln, le)


def gfield(U):
    ln, le = _NODE
    M = U.shape[0]
    return _NET.g(U, torch.full((M,), ln, dtype=DT), torch.full((M,), le, dtype=DT))


def argmax_one(args):
    """The deployed optimizer on alpha = g * sigma for one design (CRN: the
    design seed fixes the design, the raw pool and the Boltzmann stream)."""
    N, D, ell, seed = args
    Xd = draw_design(N, D, seed)
    sigma = sigma_factory(Xd, ell)
    raw = torch.quasirandom.SobolEngine(D, scramble=True, seed=seed + 7
                                        ).draw(POOL_N).to(DT).clamp(1e-4, 1 - 1e-4)
    gen = torch.Generator().manual_seed(seed + 11)
    with torch.no_grad():
        a = gfield(raw) * sigma(raw)
    i_a = int(a.argmax()); best_raw = raw[i_a][None, :]
    starts = _boltzmann_starts(raw, a, N_START, gen)
    k = starts.shape[0]

    def negf(z):
        u = torch.as_tensor(z, dtype=DT).reshape(k, D).requires_grad_(True)
        v = (gfield(u) * sigma(u)).sum()
        (grad,) = torch.autograd.grad(v, u)
        return -float(v.detach()), -grad.numpy().ravel()
    try:
        r = minimize(negf, starts.numpy().ravel(), jac=True, method="L-BFGS-B",
                     bounds=[(0.0, 1.0)] * (k * D), options={"maxiter": MAXITER})
        cand = torch.as_tensor(r.x, dtype=DT).reshape(k, D).clamp(0.0, 1.0)
    except Exception:
        return best_raw[0].numpy(), True
    with torch.no_grad():
        ac = gfield(cand) * sigma(cand)
    if float(ac.max()) > float(a.max()):
        return cand[int(ac.argmax())].numpy(), False
    return best_raw[0].numpy(), True


def run_ensembles(workers, state, D, stats, ln, le, N, ell, seed_sets):
    """One spawn pool per iteration, every worker initialised with the
    current field; returns (X*, fallback fraction) per seed set."""
    res = []
    if workers <= 0:                      # serial, in-process (debug / tiny runs)
        set_net(state, D, stats, ln, le)
        for seeds in seed_sets:
            out = [argmax_one((N, D, ell, int(s))) for s in seeds]
            res.append((np.stack([o[0] for o in out]), float(np.mean([o[1] for o in out]))))
        return res
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=set_net, initargs=(state, D, stats, ln, le)) as pool:
        for seeds in seed_sets:
            out = pool.map(argmax_one, [(N, D, ell, int(s)) for s in seeds], chunksize=16)
            res.append((np.stack([o[0] for o in out]), float(np.mean([o[1] for o in out]))))
    return res


# ------------------------------------------------------------------ refit
def refit(net, ln, le, probes, target, steps=300, lr=1e-3):
    """Absorb the multiplier into the network by warm-started Adam on log g;
    returns the rms residual."""
    for p in net.parameters():
        p.requires_grad_(True)
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    M = probes.shape[0]
    lnv = torch.full((M,), ln, dtype=DT); lev = torch.full((M,), le, dtype=DT)
    for _ in range(steps):
        loss = ((-net.delta(probes, lnv, lev) - target) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        res = ((-net.delta(probes, lnv, lev) - target) ** 2).mean().sqrt()
    return float(res)


def profile(net, ln, le, D, M=101):
    """g along the face normal x = (t, .5, ..., .5) and the diagonal
    x = (t, ..., t), t in [0, 0.5]: the plotting record."""
    t = torch.linspace(1e-4, 0.5, M, dtype=DT)
    Xn = torch.full((M, D), 0.5, dtype=DT); Xn[:, 0] = t
    Xd = t[:, None].repeat(1, D)
    lnv = torch.full((M,), ln, dtype=DT); lev = torch.full((M,), le, dtype=DT)
    with torch.no_grad():
        return t.numpy(), net.g(Xn, lnv, lev).numpy(), net.g(Xd, lnv, lev).numpy()


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--D", type=int, required=True)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--rho", type=float, default=None, help="ell / N^(-1/D)")
    ap.add_argument("--ell", type=float, default=None, help="explicit ell (node grid); overrides --rho")
    ap.add_argument("--init", choices=["info", "one"], default="info")
    ap.add_argument("--R", type=int, default=2000, help="calibration designs (CRN)")
    ap.add_argument("--R_ho", type=int, default=1000, help="validation designs (inspected per iteration)")
    ap.add_argument("--R_cert", type=int, default=4000, help="fresh certification designs, evaluated once at the end")
    ap.add_argument("--pit_dims", type=int, default=2, help="k smallest face distances in the calibration PIT")
    ap.add_argument("--patience", type=int, default=4, help="stop when E_val has not improved for this many iterations")
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--eta", type=float, default=0.1, help="max damping; adapted by backtracking (0.3 and 0.15 overshot 10x in the smoke test; 0.075 accepted)")
    ap.add_argument("--h", type=float, default=0.12, help="KDE bandwidth in PIT coordinates")
    ap.add_argument("--max_bt", type=int, default=4, help="backtracking halvings of eta per iteration")
    ap.add_argument("--clip", type=float, default=0.5, help="clip of the log density ratio per step")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    D, N = a.D, a.N
    assert (a.rho is None) != (a.ell is None), "give exactly one of --rho / --ell"
    ell = float(a.ell) if a.ell is not None else a.rho * N ** (-1.0 / D)
    if a.rho is None:
        a.rho = ell * N ** (1.0 / D)
    ln, le = math.log(N), math.log(ell)
    os.makedirs(OUTDIR, exist_ok=True)
    name = (f"gunif_D{D}_N{N}_ell{ell:.4f}_{a.init}{a.tag}" if a.ell is not None
            else f"gunif_D{D}_N{N}_rho{a.rho:g}_{a.init}{a.tag}")
    print(f"gunif_feasibility {name}: ell={ell:.4f}  R={a.R} R_ho={a.R_ho} R_cert={a.R_cert} eta={a.eta}", flush=True)

    # --- field: the g_geo network at a fixed node, as initialization
    z = torch.load(field_path("geo", D), map_location="cpu", weights_only=False)["members"][0]
    stats = z["stats"]
    net = FieldV2(D, *stats)
    if a.init == "info":
        net.load_state_dict(z["state"])
    else:
        torch.manual_seed(1)
        net = FieldV2(D, *stats)
        with torch.no_grad():                      # g == 1 exactly: zero the last rho layer
            net.rho[-1].weight.zero_(); net.rho[-1].bias.zero_()
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)

    # --- CRN design seeds: calibration and held-out disjoint; the two
    #     inits use different calibration sets
    base = 10_000_000 * (1 + (a.init == "one")) + 100_000 * D + 1000 * N
    seeds_cal = base + np.arange(a.R)
    seeds_ho = 50_000_000 + 100_000 * D + 1000 * N + np.arange(a.R_ho)     # validation (inspected)
    seeds_cert = 70_000_000 + 100_000 * D + 1000 * N + np.arange(a.R_cert)  # certification, once at the end

    bins = FaceBins(D)
    u = bins.u.numpy()
    gen = torch.Generator().manual_seed(3)
    probes = torch.cat([torch.rand(6144, D, generator=gen, dtype=DT).clamp(1e-4, 1 - 1e-4),
                        wall_points(2048, D, ell, gen)])
    rng = np.random.default_rng(0)
    rng_cal = rng.random((a.R, D))                 # fixed uniform reference for E_cal (CRN)
    dq_cal, _ = drive_null(a.R, D, rng, k=a.pit_dims); dq_val, _ = drive_null(a.R_ho, D, rng, k=a.pit_dims)
    print(f"  driving statistic null q95: cal {dq_cal:.4f} (R={a.R})  val {dq_val:.4f} (R_ho={a.R_ho})", flush=True)
    eta = a.eta
    floors = np.array([energy_distance(rng.random((a.R_ho, D)), rng.random((a.R_ho, D)), n_perm=0)[0]
                       for _ in range(10)])
    floor_mu, floor_sd = float(floors.mean()), float(floors.std())
    print(f"  uniform-vs-uniform floor at R_ho={a.R_ho}: {floor_mu:.5f} +- {floor_sd:.5f}", flush=True)

    hist = []; prof = []
    if True:
        for it in range(a.iters + 1):
            t0 = time.time()
            state = {k: v.clone() for k, v in net.state_dict().items()}
            (Xc, fb_c), (Xh, fb_h) = run_ensembles(a.workers, state, D, stats, ln, le, N, ell,
                                                   (seeds_cal, seeds_ho))
            cnt = np.bincount(bins.index(torch.as_tensor(Xc)).numpy(), minlength=bins.K).astype(float)
            phat = (cnt + 0.5) / (a.R + 0.5 * bins.K)              # diagnostic bins only
            ue = (u * a.R + 0.5) / (a.R + 0.5 * bins.K)
            e_cal = energy_distance(Xc, rng_cal, n_perm=0)[0]
            e_ho = energy_distance(Xh, rng.random((a.R_ho, D)), n_perm=0)[0]
            d_cal = drive_stat(Xc, a.pit_dims); d_val = drive_stat(Xh, a.pit_dims)
            cr = coord_report(Xh)
            rec = {"iter": it, "e_cal": float(e_cal), "e_val": float(e_ho), "d_cal": d_cal, "d_val": d_val,
                   "ratio_max": float((phat / ue).max()), "ratio_min": float((phat / ue).min()),
                   "radial_ks_val": radial_ks(Xh, D),
                   "coord_ks_max": max(k for _, k in cr),
                   "coord_mean_rng": [min(m for m, _ in cr), max(m for m, _ in cr)],
                   "fallback_frac": float(0.5 * (fb_c + fb_h)),
                   "phat": phat.tolist()}
            tt, gn, gd = profile(net, ln, le, D); prof.append(np.stack([gn, gd]))
            print(f"  it {it:2d}  drive cal {d_cal:.4f} val {d_val:.4f} (null q95 {dq_cal:.4f}/{dq_val:.4f})  "
                  f"E_cal {e_cal:.5f}  E_val {e_ho:.5f} (floor {floor_mu:.5f}+-{floor_sd:.5f})  "
                  f"bin ratio [{rec['ratio_min']:.2f},{rec['ratio_max']:.2f}]  radKS {rec['radial_ks_val']:.3f}  "
                  f"coordKS {rec['coord_ks_max']:.3f}  fallback {rec['fallback_frac']:.3f}  "
                  f"g_face(0)={gn[0]:.3f} g_diag(0)={gd[0]:.3f}  [{time.time()-t0:.0f}s]", flush=True)
            d_val_hist = [h["d_val"] for h in hist] + [d_val]
            converged = (len(d_val_hist) >= 2 and all(v <= dq_val for v in d_val_hist[-2:])
                         and d_cal <= dq_cal and float(e_ho) <= floor_mu + 2 * floor_sd)
            n_rej = 0
            for h in reversed(hist):
                if h.get("accepted"): break
                n_rej += 1
            stalled = n_rej >= a.patience
            if it == a.iters or converged or stalled:
                rec["stop"] = "max_iters" if it == a.iters else ("converged" if converged else "STALL above floor")
                hist.append(rec); print(f"  stop: {rec['stop']}", flush=True); break
            # damped multiplicative update on the smooth density ratio; a step is kept only if it lowers E_cal
            V_c = volume_pit(torch.as_tensor(Xc), a.pit_dims)
            with torch.no_grad():
                ratio = kde_ratio(volume_pit(probes, a.pit_dims), V_c, h=a.h)
            logr = torch.log(ratio.clamp_min(1e-3)).clamp(-a.clip, a.clip)   # bounded step per region
            lnv = torch.full((probes.shape[0],), ln, dtype=DT); lev = torch.full_like(lnv, le)
            with torch.no_grad():
                base_logg = -net.delta(probes, lnv, lev)
            state0 = {k: v.clone() for k, v in net.state_dict().items()}
            accepted = False
            for bt in range(a.max_bt):
                net.load_state_dict(state0)
                target = base_logg - eta * logr
                rres = refit(net, ln, le, probes, target)
                st = {k: v.clone() for k, v in net.state_dict().items()}
                (Xc_try, _), = run_ensembles(a.workers, st, D, stats, ln, le, N, ell, (seeds_cal,))
                d_try = drive_stat(Xc_try, a.pit_dims)
                print(f"      try eta={eta:.4f}: drive {d_try:.4f} vs {d_cal:.4f} (null q95 {dq_cal:.4f})  refit_rms {rres:.4f}", flush=True)
                if d_try < max(d_cal, dq_cal):
                    accepted = True; break
                eta *= 0.5
            rec["eta"] = eta; rec["accepted"] = accepted; rec["refit_rms"] = rres
            rec["mult_max_abs"] = float((eta * logr).abs().max())
            if not accepted:
                net.load_state_dict(state0)
                print("      no eta accepted; field unchanged", flush=True)
            else:
                eta = min(a.eta, eta * 1.25)
            hist.append(rec)
            np.savez(os.path.join(OUTDIR, name + ".npz"), t=tt, profiles=np.stack(prof),
                     Xho_last=Xh, Xcal_last=Xc, u=u, ell=ell, N=N, D=D)
            with open(os.path.join(OUTDIR, name + ".json"), "w") as f:
                json.dump({"node": {"D": D, "N": N, "ell": ell, "rho": a.rho}, "init": a.init,
                           "floor_mu": floor_mu, "floor_sd": floor_sd, "hist": hist,
                           "args": vars(a)}, f, indent=1)
    # certification on the fresh ensemble: energy distance (primary), radial and coordinate KS, bin ratios
    state = {k: v.clone() for k, v in net.state_dict().items()}
    (Xcert, fb_cert), = run_ensembles(a.workers, state, D, stats, ln, le, N, ell, (seeds_cert,))
    rng_c = np.random.default_rng(12345)
    cf = np.array([energy_distance(rng_c.random((a.R_cert, D)), rng_c.random((a.R_cert, D)), n_perm=0)[0] for _ in range(10)])
    e_cert = float(energy_distance(Xcert, rng_c.random((a.R_cert, D)), n_perm=0)[0])
    nulls = {"radial_ks": [], "coord_ks": [], "bin_ratio_max": []}
    for _ in range(20):
        Un = rng_c.random((a.R_cert, D))
        nulls["radial_ks"].append(radial_ks(Un, D)); nulls["coord_ks"].append(max(k for _, k in coord_report(Un)))
        cn = np.bincount(bins.index(torch.as_tensor(Un)).numpy(), minlength=bins.K) / a.R_cert
        nulls["bin_ratio_max"].append(float((cn / u).max()))
    q95 = {k: float(np.quantile(v, 0.95)) for k, v in nulls.items()}
    cnt_c = np.bincount(bins.index(torch.as_tensor(Xcert)).numpy(), minlength=bins.K) / a.R_cert
    diag = {"radial_ks": radial_ks(Xcert, D), "coord_ks": max(k for _, k in coord_report(Xcert)),
            "bin_ratio_max": float((cnt_c / u).max()), "bin_ratio_min": float((cnt_c / u).min()),
            "bin_profile": (cnt_c / u).tolist(), "fallback_frac": fb_cert}
    flags = {k: bool(diag[k] > q95[k]) for k in q95}
    ok_primary = e_cert <= float(cf.mean() + 2 * cf.std())
    # verdict = primary certificate AND no diagnostic above its uniform null
    # (tiny-R smoke, D=6: energy distance "passed" while radial KS was 1.0)
    ok = ok_primary and not any(flags.values())
    final = hist[-1]
    cert = {"e_cert": e_cert, "floor_mu": float(cf.mean()), "floor_sd": float(cf.std()),
            "pass_primary": ok_primary, "pass": ok,
            "diag": diag, "null_q95": q95, "flags": flags, "R_cert": a.R_cert}
    print(f"  CERT {name}: E_cert {e_cert:.5f} vs floor {cf.mean():.5f}+2*{cf.std():.5f} -> "
          f"{'pass' if ok_primary else 'fail'} (primary); verdict {'PASS' if ok else 'FAIL'};  diagnostics over null q95: "
          f"{[k for k, v in flags.items() if v] or 'none'}  (radKS {diag['radial_ks']:.3f}/{q95['radial_ks']:.3f}, "
          f"coordKS {diag['coord_ks']:.3f}/{q95['coord_ks']:.3f}, binmax {diag['bin_ratio_max']:.2f}/{q95['bin_ratio_max']:.2f})", flush=True)
    torch.save({"state": net.state_dict(), "stats": stats, "node": {"D": D, "N": N, "ell": ell},
                "init": a.init, "pass": ok, "cert": cert}, os.path.join(OUTDIR, name + ".pt"))
    np.savez(os.path.join(OUTDIR, name + ".npz"), t=tt, profiles=np.stack(prof),
             Xho_last=Xh, Xcal_last=Xc, Xcert=Xcert, u=u, ell=ell, N=N, D=D)
    with open(os.path.join(OUTDIR, name + ".json"), "w") as f:
        json.dump({"node": {"D": D, "N": N, "ell": ell, "rho": a.rho}, "init": a.init,
                   "floor_mu": floor_mu, "floor_sd": floor_sd, "hist": hist, "pass": ok, "cert": cert,
                   "args": vars(a)}, f, indent=1)


if __name__ == "__main__":
    main()
