"""Early-phase readout of the controlled synthetic experiment: share of the first
D+2 evaluations per spatial class, fraction of greedy rounds, probability of
having entered the global basin by round t, and the objective-free allocation
at the same scale.

    python mb4_early.py --dir ../results/bo_mb4_norm
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, "experiments"), _os.path.join(_ROOT, "training"), _os.path.join(_ROOT, "plots")):
    _p in _sys.path or _sys.path.insert(0, _p)
import argparse, glob, json, os
import numpy as np
from mb4_read import CFG, POS, ARMS, CTRL, codim, load

BOOT = np.random.default_rng(0)
# radius of "inside the basin" in units of s; 1.5 s contains the exact corner in every D
RADX = 1.5


def ci(d, B=3000):
    m = BOOT.choice(d, (B, len(d))).mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="../results/bo_mb4_norm")
    ap.add_argument("--ctrl_dir", default="../results/bo_mb4_ref")
    ap.add_argument("--baseline", default="../results/mb4_baseline_matched.json")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--dims", default="2,3,4")
    ap.add_argument("--lams", default="1,2,4,8")
    ap.add_argument("--seeds", type=int, default=25)
    a = ap.parse_args()
    base = json.load(open(a.baseline))["data"]
    mult = {"_s05": "x0.5", "": "x1", "_s15": "x1.5"}[a.suffix]
    lams = [int(x) for x in a.lams.split(",")]
    rng = np.random.default_rng(0)
    for D in [int(x) for x in a.dims.split(",")]:
        B, s_, variants = CFG[D]; h = B ** (-1 / D); rad = RADX * s_; W = D + 2
        Uv = rng.random((300000, D)); pu = np.bincount(codim(Uv, h), minlength=D + 1) / len(Uv)
        print(f"\n================ scale {mult}  D={D}  h={h:.3f}  early window = first {W} picks; checkpoints t={W}, {2*W}, last")
        col = base[str(D)]["cols"][f"matched {mult}"]
        print("  D. objective-free baseline E_k at this scale (pure exploration, N=B): "
              + " | ".join(f"{m}: " + " ".join(f"{e:4.1f}" for e in col[m]["E"]) for m in col))
        for lam in lams:
            tag = f"mb4_l{lam}{a.suffix}"
            print(f"\n  lambda={lam}")
            print(f"     {'arm':>7s} | A early E_k " + " ".join(f"{'k'+str(k):>5s}" for k in range(D + 1))
                  + f" | B greedy early/all | C hit by t={W:2d} {2*W:2d} last  " + "  ".join(f"[{v[:4]} @{2*W}]" for v in variants))
            base_hit = {}
            for name, arm in ARMS + (CTRL if lam == lams[0] else ()):
                isctrl = arm in dict(CTRL).values()
                cnt = np.zeros(D + 1); g_e = []; g_a = []
                hit = {v: {W: [], 2 * W: [], "last": []} for v in variants}; nfiles = 0
                for v in variants:
                    ctr = np.array(POS[D][v])
                    for s in range(a.seeds):
                        r = load(a.ctrl_dir if isctrl else a.dir, D, v, s, "mb4_ctrl" if isctrl else tag)
                        if r is None or arm not in r[1]:
                            continue
                        nfiles += 1
                        P = np.asarray(r[1][arm]["picks"]); X0 = np.asarray(r[0]["X0"])
                        cnt += np.bincount(codim(P[:W], h), minlength=D + 1)
                        if "mu_argmax" in r[1][arm] and not isctrl:
                            ma = np.asarray(r[1][arm]["mu_argmax"], float); g_e.append(ma[:W].mean()); g_a.append(ma.mean())
                        din = np.linalg.norm(np.vstack([X0, P]) - ctr, axis=1) <= rad
                        n0 = len(X0)
                        for t in (W, 2 * W):
                            hit[v][t].append(float(din[:n0 + t].any()))
                        hit[v]["last"].append(float(din.any()))
                if nfiles == 0:
                    print(f"     {name:>7s} | no files"); continue
                E = cnt / cnt.sum() / pu
                H = {t: np.mean([np.mean(hit[v][t]) for v in variants]) for t in (W, 2 * W, "last")}
                cells = []
                for v in variants:
                    d = np.array(hit[v][2 * W])
                    if name == "raw":
                        base_hit[v] = d; cells.append(f"{d.mean():.2f}     ")
                    elif v in base_hit and len(base_hit[v]) == len(d):
                        lo, hi = ci(d - base_hit[v]); sig = "*" if (lo > 0 or hi < 0) else " "
                        cells.append(f"{d.mean():.2f}{d.mean()-base_hit[v].mean():+.2f}{sig}")
                    else:
                        cells.append(f"{d.mean():.2f}     ")
                gtxt = f"{np.mean(g_e):.2f}/{np.mean(g_a):.2f}" if g_e else "  -  /  -  "
                print(f"     {name:>7s} | " + " ".join(f"{e:5.2f}" for e in E)
                      + f"     |    {gtxt}     |   {H[W]:.2f} {H[2*W]:.2f} {H['last']:.2f}   " + "  ".join(f"{c:>11s}" for c in cells)
                      + (f"   (n={nfiles})" if nfiles < len(variants) * a.seeds else ""))


if __name__ == "__main__":
    main()
