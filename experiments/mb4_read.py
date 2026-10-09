"""Where each rule evaluates in the controlled synthetic experiment: share of
evaluations per spatial class (volume-corrected), and the objective-free
allocation against the rate of reaching each basin. `load` and the constants
are shared with the other mb4 scripts.

    python mb4_read.py --dir ../results/bo_mb4_ref
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, "experiments"), _os.path.join(_ROOT, "training"), _os.path.join(_ROOT, "plots")):
    _p in _sys.path or _sys.path.insert(0, _p)
import argparse, glob, json, os
import numpy as np

CFG = {2: (25, 0.100, ("corner", "face", "interior")),
       3: (40, 0.146, ("corner", "edge", "face", "interior")),
       4: (55, 0.184, ("corner", "edge", "face", "interior"))}
POS = {2: {"corner": (0.07, 0.07), "face": (0.16, 0.93), "interior": (0.80, 0.42)},
       3: {"corner": (0.102, 0.898, 0.898), "edge": (0.634, 0.898, 0.102),
           "face": (0.102, 0.384, 0.234), "interior": (0.692, 0.292, 0.692)},
       4: {"corner": (0.129, 0.129, 0.871, 0.871), "edge": (0.544, 0.129, 0.344, 0.871),
           "face": (0.871, 0.694, 0.294, 0.694), "interior": (0.367, 0.517, 0.617, 0.367)}}
ARMS = (("raw", "fit_fix"), ("g_info", "fit_gfix"), ("g_eff", "fit_gefffix"),
        ("g_unif", "fit_gufix"), ("std_bo", "fit_fit"))
CTRL = (("sobol", "sobol"), ("fps", "fps"), ("random", "random"))


def codim(X, h):
    return (np.minimum(X, 1 - X) < h / 2).sum(1)


def load(d, D, v, s, tag):
    p = os.path.join(d, f"mb4_{D}d_{v}_s{s}_{tag}.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=True)
    return z["meta"].item(), z["results"].item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="../results/bo_mb4_ref")
    ap.add_argument("--baseline", default="../results/mb4_baseline.json")
    ap.add_argument("--seeds", type=int, default=25)
    a = ap.parse_args()
    base = json.load(open(a.baseline))["data"]
    rng = np.random.default_rng(0)
    for D, (B, s_, variants) in CFG.items():
        h = B ** (-1 / D); rad = 1.177 * s_
        Uv = rng.random((400000, D)); pu = np.bincount(codim(Uv, h), minlength=D + 1) / len(Uv)
        print(f"\n================ D={D}  h={h:.3f}  uniform class shares " +
              " ".join(f"k{k}={pu[k]:.3f}" for k in range(D + 1)))
        # ---- A: realised allocation, all picks and early window
        for lam in (1, 2, 4):
            print(f"\n  A. enrichment E_k of picks, lambda={lam}   (all rounds | first {D+2} picks)")
            print(f"     {'arm':>8s} | " + "  ".join(f"{'k'+str(k):>5s}" for k in range(D + 1))
                  + " | " + "  ".join(f"{'k'+str(k):>5s}" for k in range(D + 1)))
            for name, arm in ARMS + (CTRL if lam == 2 else ()):
                tag = "mb4_ctrl" if arm in dict(CTRL).values() else f"mb4_l{lam}"
                cnt = np.zeros(D + 1); cnt_e = np.zeros(D + 1)
                for v in variants:
                    for s in range(a.seeds):
                        r = load(a.dir, D, v, s, tag)
                        if r is None or arm not in r[1]:
                            continue
                        P = np.asarray(r[1][arm]["picks"])
                        cnt += np.bincount(codim(P, h), minlength=D + 1)
                        cnt_e += np.bincount(codim(P[:D + 2], h), minlength=D + 1)
                if cnt.sum() == 0:
                    continue
                E = cnt / cnt.sum() / pu; Ee = cnt_e / cnt_e.sum() / pu
                print(f"     {name:>8s} | " + "  ".join(f"{e:5.2f}" for e in E)
                      + " | " + "  ".join(f"{e:5.2f}" for e in Ee))
        # ---- B: q_{m,A} (baseline, ell_ref column) vs measured allocation
        col = base[str(D)]["cols"]["ell_ref"]
        bname = {"raw": "raw sigma", "g_info": "g_info", "g_eff": "g_eff", "g_unif": "g_unif"}
        print(f"\n  B. objective-free q_(m,A) -> measured P(any pick in well A)   [q ever = 1-(1-q)^R]")
        for lam in (2, 4):
            print(f"     lambda={lam}:  {'arm':>7s} | " + "  ".join(f"{v[:8]:>14s}" for v in variants) + "   corr(ever, hit)")
            for name, arm in ARMS[:4]:
                ev = col[bname[name]]["ever"]; hit = []
                for v in variants:
                    ctr = np.array(POS[D][v]); n = 0; k = 0
                    for s in range(a.seeds):
                        r = load(a.dir, D, v, s, f"mb4_l{lam}")
                        if r is None:
                            continue
                        X = np.vstack([np.asarray(r[0]["X0"]), np.asarray(r[1][arm]["picks"])])
                        n += 1; k += int((np.linalg.norm(X - ctr, axis=1) <= rad).any())
                    hit.append(k / max(n, 1))
                c = np.corrcoef(ev, hit)[0, 1] if len(variants) > 2 else float("nan")
                print(f"             {name:>7s} | " + "  ".join(f"{e:5.2f} -> {hh:4.2f}  " for e, hh in zip(ev, hit))
                      + f"   r={c:+.2f}")


if __name__ == "__main__":
    main()
