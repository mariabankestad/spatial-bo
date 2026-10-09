"""Score runs of run_bo.py: simple regret against the polished optimum
(results/hstar.json), per function, paired across seeds with bootstrap intervals.

    python eval_bo.py --tag <tag>
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, "experiments"), _os.path.join(_ROOT, "training"), _os.path.join(_ROOT, "plots")):
    _p in _sys.path or _sys.path.insert(0, _p)
import argparse
import json
import os
import re

import numpy as np
import torch
from scipy.optimize import minimize

import spatialbo.acquisition as A
import functions as F
from spatialbo.gp import DT
from run_bo import OUTDIR
from spatialbo.functions import standardized

HSTAR_CACHE = os.path.join(OUTDIR, "hstar.json")
CONTRASTS = [("fit_fix", "fit_fit", "geometric vs fitted exploration (H4)"),
             ("fit_gfix", "fit_fix", "what the equalizer adds"),
             ("fit_gvar", "fit_gfix", "stronger correction vs derived"),
             ("fit_gfix", "fit_fit", "method vs matched fitted baseline"),
             ("fit_fit", "random", "baseline vs random floor")]


def hstar(func):
    cache = json.load(open(HSTAR_CACHE)) if os.path.exists(HSTAR_CACHE) else {}
    if func in cache:
        return cache[func]
    h, D = standardized(func)
    P = torch.quasirandom.SobolEngine(D, scramble=True, seed=1).draw(2 ** 17).to(DT)
    x0s = P[h(P).argsort(descending=True)[:16]].numpy()
    obj = lambda x: -float(h(torch.tensor(x, dtype=DT).unsqueeze(0)))
    best = max(-minimize(obj, x0, method="L-BFGS-B",
                         bounds=[(0, 1)] * D).fun for x0 in x0s)
    cache[func] = best
    os.makedirs(os.path.dirname(HSTAR_CACHE), exist_ok=True)
    json.dump(cache, open(HSTAR_CACHE, "w"), indent=1)
    return best


def curves(func, tag, arm, max_seed=64):
    """Per-seed regret curves and the seed ids, so arms can be paired."""
    hs = hstar(func)
    out, seeds = [], []
    for s in range(max_seed):
        p = os.path.join(OUTDIR, f"{func}_s{s}_{tag}.npz")
        if not os.path.exists(p):
            continue
        z = np.load(p, allow_pickle=True)
        r = z["results"].item()
        if arm not in r:
            continue
        h0 = float(np.max(z["meta"].item()["h0"]))
        inc = np.maximum.accumulate(np.concatenate([[h0], r[arm]["htrue"]]))[1:]
        out.append(hs - inc)
        seeds.append(s)
    return (np.array(out), seeds) if out else (None, [])


def boot_ci(d, n=20000, seed=0):
    """Percentile bootstrap over the paired differences."""
    if len(d) < 2:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    m = rng.choice(d, size=(n, len(d)), replace=True).mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def checkpoints(rounds):
    """Standard BO reporting points: quarter, half, three-quarter, full budget."""
    return [max(1, int(rounds * f)) for f in (0.25, 0.5, 0.75, 1.0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="x2")
    ap.add_argument("--baseline", default="fit_fit")
    a = ap.parse_args()
    files = ([f for f in sorted(os.listdir(OUTDIR)) if f.endswith(f"_{a.tag}.npz")]
             if os.path.isdir(OUTDIR) else [])
    if not files:
        print(f"no results with tag {a.tag!r} in {OUTDIR}")
        return
    pat = re.compile(r"^(?P<f>.+)_s\d+_" + re.escape(a.tag) + r"\.npz$")
    funcs = sorted({m.group("f") for m in map(pat.match, files) if m},
                   key=lambda k: (F.DIM[k], k))

    store = {(fn, arm): curves(fn, a.tag, arm) for fn in funcs for arm in A.ARMS}

    # ---- 1. simple regret at the budget, the headline table -----------------
    print(f"Simple regret at the full budget, mean +- 95% CI over seeds. "
          f"tag={a.tag}\n")
    print(f"{'function':20s} {'n':>3s}  " + "  ".join(f"{x:>17s}" for x in A.ARMS))
    for fn in funcs:
        cells, ns = [], []
        for arm in A.ARMS:
            cv, sd = store[(fn, arm)]
            if cv is None:
                cells.append(f"{'--':>17s}"); continue
            v = cv[:, -1]; ns.append(len(sd))
            lo, hi = boot_ci(v)
            cells.append(f"{v.mean():8.4f}+-{(hi-lo)/2:<8.4f}")
        print(f"{fn:20s} {max(ns):3d}  " + "  ".join(cells))

    # ---- 2. regret trajectory at checkpoints --------------------------------
    print("\nSimple regret at 25 / 50 / 75 / 100% of the budget (mean over seeds)\n")
    for fn in funcs:
        cv0, _ = store[(fn, A.ARMS[0])]
        if cv0 is None:
            continue
        cks = checkpoints(cv0.shape[1])
        print(f"  {fn} (rounds {', '.join(map(str, cks))})")
        for arm in A.ARMS:
            cv, _ = store[(fn, arm)]
            if cv is None:
                continue
            print(f"    {arm:9s} " + "  ".join(f"{cv[:, c-1].mean():9.4f}" for c in cks))

    # ---- 3. paired comparison against the baseline, at the budget -----------
    print(f"\nPaired comparison against {a.baseline}, final simple regret.")
    print("d > 0 means the arm is better. CI is a 20k bootstrap over paired "
          "seeds; wins counts seeds.\n")
    for arm in A.ARMS:
        if arm == a.baseline:
            continue
        print(f"-- {arm} vs {a.baseline}")
        for fn in funcs:
            ca, sa = store[(fn, arm)]; cb, sb = store[(fn, a.baseline)]
            if ca is None or cb is None:
                continue
            keep = [x for x in sa if x in sb]
            if len(keep) < 2:
                continue
            d = (cb[[sb.index(x) for x in keep], -1]
                 - ca[[sa.index(x) for x in keep], -1])
            lo, hi = boot_ci(d)
            v = arm if lo > 0 else (a.baseline if hi < 0 else "tie")
            print(f"   {fn:20s} {d.mean():+9.4f} [{lo:+8.4f},{hi:+8.4f}] "
                  f"{int((d > 0).sum()):3d}/{len(d):<3d}  {v}")
        print()

    # ---- 4. mean rank across functions, the standard aggregate --------------
    print("Mean rank across functions at the full budget (1 = best of six)\n")
    ranks = {arm: [] for arm in A.ARMS}
    for fn in funcs:
        vals = {arm: store[(fn, arm)][0] for arm in A.ARMS}
        if any(v is None for v in vals.values()):
            continue
        seeds = set.intersection(*[set(store[(fn, arm)][1]) for arm in A.ARMS])
        for s in sorted(seeds):
            row = [vals[arm][store[(fn, arm)][1].index(s), -1] for arm in A.ARMS]
            order = np.argsort(np.argsort(row)) + 1
            for arm, r in zip(A.ARMS, order):
                ranks[arm].append(r)
    for arm in sorted(A.ARMS, key=lambda k: np.mean(ranks[k]) if ranks[k] else 9):
        if ranks[arm]:
            print(f"   {arm:9s} {np.mean(ranks[arm]):.2f}")


if __name__ == "__main__":
    main()
