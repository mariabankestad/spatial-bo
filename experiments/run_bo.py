"""Run the exploration rules on one function and one seed.

    python run_bo.py --func hartmann_3d --seed 0 --lam 2 --g_norm design

The objective is maximized in standardized form, h = -(f - m)/s. Within a
seed all rules share the initial design, the candidate pools and the
optimizer's random stream. Options:
  --ell_rule   "paper" = 0.268 sqrt(D/2); "mult:<f>" = f times that; "spacing_n" = (1/n)^(1/D)
  --g_ell      lengthscale at which the fields are evaluated: "ref" (paper) or "fit"
  --g_norm     normalization of g: "design" (mean over the evaluated points), "mean" (over the domain), "none"
  --noise_y / --noise_model   noise variance added to the data / assumed by the GP
One .npz per (function, seed) goes to $BO_RESULTS_DIR (default ../results).
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, "experiments")):
    _p in _sys.path or _sys.path.insert(0, _p)
import argparse
import math
import os
import time

import numpy as np
import torch

import spatialbo.acquisition as A
import functions as F
from spatialbo.fields import load_field
from spatialbo.gp import DEVICE, DT, protocol_lengthscale, spacing_lengthscale

NOISE = 0.0025                # default observation variance (sigma_n = 0.05)
NOISE_Y = NOISE               # variance of the noise added to observations
NOISE_M = NOISE               # variance assumed by the GP
RAW_SAMPLES = 512             # candidate pool per round
NORM_SAMPLES = 1024           # fixed Sobol set for the domain average of g
N_RESTART = 10                # Boltzmann-sampled starts for the optimizer
LAM = 2.0
BUDGET = {2: 25, 3: 40, 4: 55, 5: 70, 6: 85}   # total evaluations per dimension

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.environ.get("BO_RESULTS_DIR", os.path.join(os.path.dirname(HERE), "results"))


def ell_of(rule, n, D):
    if rule == "paper":
        return protocol_lengthscale(D)
    if rule.startswith("mult:"):
        return float(rule[5:]) * protocol_lengthscale(D)
    if rule == "spacing_n":
        return spacing_lengthscale(n, D)
    raise ValueError(f"unknown ell rule {rule!r}")


def run(func, seed, rounds, n0, lam, methods, tag, ell_rule, polish, gell_rule="ref", g_norm="design"):
    h, D = F.standardized(func)
    n0 = n0 if n0 > 0 else 2 * D + 2
    rounds = rounds if rounds > 0 else BUDGET[D] - n0
    X0 = torch.quasirandom.SobolEngine(D, scramble=True, seed=100 + seed
                                       ).draw(n0).to(DT).clamp(1e-3, 1 - 1e-3)
    y0 = h(X0) + math.sqrt(NOISE_Y) * torch.randn(
        n0, generator=torch.Generator().manual_seed(1000 + seed), dtype=DT)

    fg = load_field("geo", D)
    f3 = load_field("eff", D)
    fgu = load_field("uniform", D)
    norm_pool = torch.quasirandom.SobolEngine(D, scramble=True, seed=777
                                              ).draw(NORM_SAMPLES).to(DT).clamp(1e-4, 1 - 1e-4)
    results = {}
    print(f"{func} D={D} n0={n0} rounds={rounds} ell_rule={ell_rule} lam={lam} "
          f"noise_y={NOISE_Y} noise_model={NOISE_M} g_ell={gell_rule} g_norm={g_norm} device={DEVICE}", flush=True)
    for arm in methods:
        t0 = time.time()
        X, y = X0.clone(), y0.clone()
        noise_gen = torch.Generator().manual_seed(8000 + seed)
        log = {k: [] for k in ("picks", "yobs", "htrue", "ell_fit", "ell_ref",
                               "d_face", "g_pick", "g_norm_c", "mu_pick", "bonus_pick",
                               "mu_argmax", "cond_floor", "clamps", "lam_eff",
                               "mu_gap", "ell_oob")}
        for t in range(rounds):
            n = X.shape[0]
            ell_ref = ell_of(ell_rule, n, D)
            raw = torch.quasirandom.SobolEngine(
                D, scramble=True, seed=10000 * (seed + 1) + t
                ).draw(RAW_SAMPLES).to(DT).clamp(1e-4, 1 - 1e-4)
            opt_gen = torch.Generator().manual_seed(500000 + 1000 * seed + t)

            if arm == "random":
                xn = raw[int(torch.randint(RAW_SAMPLES, (1,), generator=opt_gen))][None, :]
                sur, dg, gk = None, {"pick_is_mu_argmax": False}, float("nan")
            else:
                sur = A.Surrogate(X, y, D, NOISE_M, ell_ref, need_fit=True)
                g_ell = (lambda s_: s_.ell_fit) if gell_rule == "fit" else (lambda s_: ell_ref)
                sur.g_norm_cs = {}

                def _norm(f, ell_fn, name, s_=sur):
                    """The field at the round's state divided by its average (a float, not differentiated)."""
                    if g_norm == "none":
                        return lambda U: f(U, n, ell_fn(s_))
                    with torch.no_grad():
                        c = float(f(X if g_norm == "design" else norm_pool, n, ell_fn(s_)).mean())
                    s_.g_norm_cs[name] = c
                    return lambda U: f(U, n, ell_fn(s_)) / c

                sur.g_ref = _norm(fg, lambda s_: ell_ref, "g_ref")
                if arm == "fit_gefffix":
                    sur.g_eff = _norm(f3, g_ell, "g_eff")
                if arm == "fit_gufix":
                    sur.g_u = _norm(fgu, g_ell, "g_u")
                xn, dg = A.optimize(arm, sur, lam, D, raw, opt_gen, n_start=N_RESTART, polish=polish)
                with torch.no_grad():
                    gk = (float(sur.g_eff(xn)) if arm == "fit_gefffix"
                          else float(sur.g_ref(xn)) if arm == "fit_gfix"
                          else float(sur.g_u(xn)) if arm == "fit_gufix"
                          else float("nan"))
            hn = h(xn)
            yn = hn + math.sqrt(NOISE_Y) * torch.randn(1, generator=noise_gen, dtype=DT)
            with torch.no_grad():
                mu_p = float(sur.mean(xn)) if sur is not None else float("nan")
                lam_used = dg.get("lam_eff", lam)
                mu_gap = float(sur.mean(raw).max()) - mu_p if sur is not None else float("nan")
                bonus_p = (float(A.alpha(arm, sur, xn, lam_used)) - mu_p
                           if sur is not None else float("nan"))
            X = torch.cat([X, xn]); y = torch.cat([y, yn])
            gname = {"fit_gfix": "g_ref", "fit_gefffix": "g_eff", "fit_gufix": "g_u"}.get(arm)
            log["picks"].append(xn.numpy().ravel())
            log["yobs"].append(float(yn)); log["htrue"].append(float(hn))
            log["ell_fit"].append(sur.ell_fit if sur is not None else float("nan"))
            log["ell_ref"].append(ell_ref)
            log["d_face"].append(float(torch.minimum(xn, 1 - xn).min()))
            log["g_pick"].append(gk)
            log["g_norm_c"].append(getattr(sur, "g_norm_cs", {}).get(gname, float("nan"))
                                   if (sur is not None and gname) else float("nan"))
            log["mu_pick"].append(mu_p); log["bonus_pick"].append(bonus_p)
            log["mu_argmax"].append(bool(dg["pick_is_mu_argmax"]))
            log["cond_floor"].append(getattr(sur, "cond_floor", float("nan")) if sur is not None else float("nan"))
            log["clamps"].append(fg.clamp_events)
            log["ell_oob"].append(bool(sur is not None and not fg.in_envelope(n, sur.ell_fit)))
            log["lam_eff"].append(float(dg.get("lam_eff", float("nan"))))
            log["mu_gap"].append(mu_gap)
            if (t + 1) % 25 == 0:
                print(f"    [{arm}] round {t+1}/{rounds} best h={max(log['htrue']):+.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
        results[arm] = {k: np.array(v) for k, v in log.items()}
        print(f"[{func} s{seed} {arm:>11s}] best h={max(log['htrue']):+.4f} ({time.time() - t0:.0f}s)", flush=True)

    os.makedirs(OUTDIR, exist_ok=True)
    out = os.path.join(OUTDIR, f"{func}_s{seed}_{tag}.npz")
    np.savez(out,
             meta=np.array({"func": func, "seed": seed, "rounds": rounds, "n0": n0, "D": D, "lam": lam,
                            "noise_var": NOISE, "noise_y": NOISE_Y, "noise_model": NOISE_M,
                            "ell_rule": ell_rule, "g_ell": gell_rule, "g_norm": g_norm,
                            "raw_samples": RAW_SAMPLES, "norm_samples": NORM_SAMPLES,
                            "n_restart": N_RESTART, "n_top": A.N_TOP, "g_grad": True,
                            "polish": polish, "tag": tag, "device": str(DEVICE),
                            "X0": X0.numpy(), "y0": y0.numpy(), "h0": h(X0).numpy()},
                           dtype=object),
             results=np.array(results, dtype=object))
    print(f"saved: {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--func", required=True, choices=sorted(F.FUNCS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=0, help="0 = budget of the dimension minus the initial design")
    ap.add_argument("--n0", type=int, default=0, help="0 = 2D+2")
    ap.add_argument("--lam", type=float, default=LAM)
    ap.add_argument("--methods", default=",".join(A.ARMS))
    ap.add_argument("--tag", default="run")
    ap.add_argument("--ell_rule", default="paper")
    ap.add_argument("--g_norm", default="design", choices=("none", "mean", "design"))
    ap.add_argument("--g_ell", default="ref", choices=("fit", "ref"))
    ap.add_argument("--no_polish", action="store_true", help="candidate argmax only, no L-BFGS-B")
    ap.add_argument("--noise_y", type=float, default=-1.0, help="observation-noise variance; -1 = %(default)s -> 0.0025")
    ap.add_argument("--noise_model", type=float, default=-1.0, help="GP noise variance; -1 = 0.0025")
    a = ap.parse_args()
    if a.noise_y >= 0:
        NOISE_Y = a.noise_y
    if a.noise_model >= 0:
        NOISE_M = a.noise_model
    run(a.func, a.seed, a.rounds, a.n0, a.lam, a.methods.split(","), a.tag,
        a.ell_rule, not a.no_polish, a.g_ell, a.g_norm)
