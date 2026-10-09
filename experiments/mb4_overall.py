"""Overall score of the controlled synthetic experiment: probability that the
final best evaluation lies in the global basin, averaged over placements and
paired per seed against the unweighted rule.

    python mb4_overall.py
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, "experiments"), _os.path.join(_ROOT, "training"), _os.path.join(_ROOT, "plots")):
    _p in _sys.path or _sys.path.insert(0, _p)
import glob, os, sys
import numpy as np
from mb4_read import CFG, POS, ARMS, CTRL, load

rng = np.random.default_rng(1)


def boot(d, B=4000):
    m = rng.choice(d, (B, len(d))).mean(1)
    return d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def main(d="../results/bo_mb4_ref", S=25):
    for D, (Bn, s_, variants) in CFG.items():
        rad = 1.177 * s_
        print(f"\n================ D={D}   OVERALL = mean over {len(variants)} addresses, paired on seed")
        for lam in (1, 2, 4):
            print(f"\n  lambda={lam}   {'arm':>7s} |  hit  [95% CI of hit - raw]   | d(best) | pick=argmax(mu)")
            H = {}; R = {}; M = {}
            for name, arm in ARMS + CTRL:
                tag = "mb4_ctrl" if arm in dict(CTRL).values() else f"mb4_l{lam}"
                if arm in dict(CTRL).values() and lam != 2:
                    continue
                hs = np.zeros(S); rs = np.zeros(S); mm = []
                for v in variants:
                    ctr = np.array(POS[D][v])
                    for s in range(S):
                        r = load(d, D, v, s, tag); a = r[1][arm]
                        X = np.vstack([np.asarray(r[0]["X0"]), np.asarray(a["picks"])])
                        hall = np.concatenate([np.asarray(r[0]["h0"]), np.asarray(a["htrue"])])
                        hs[s] += float(np.linalg.norm(X[np.argmax(hall)] - ctr) <= rad) / len(variants)
                        rs[s] += (hall.max()) / len(variants)   # mean best value found (hstar differs per variant; contrast only)
                        if "mu_argmax" in a:
                            mm.append(np.mean(a["mu_argmax"]))
                H[name] = hs; R[name] = rs; M[name] = np.mean(mm) if mm else np.nan
            for name in H:
                m, lo, hi = boot(H[name] - H["raw"]) if name != "raw" else (0, 0, 0)
                sig = "*" if (lo > 0 or hi < 0) else " "
                rm = (R[name] - R["raw"]).mean()
                print(f"           {name:>7s} |  {H[name].mean():.2f}  [{m:+.2f} ({lo:+.2f},{hi:+.2f})]{sig}   | {rm:+.3f}  |  {M[name]:.2f}"
                      if name != "raw" else
                      f"           {name:>7s} |  {H[name].mean():.2f}  [   raw reference   ]    | {'0':>6s}  |  {M[name]:.2f}")


if __name__ == "__main__":
    main()
