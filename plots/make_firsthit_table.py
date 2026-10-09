"""Table of the median number of rounds until the first evaluation in the global
basin, given that the initial design missed it ('>R': more than half of the
runs never entered within R rounds).
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, "experiments"), _os.path.join(_ROOT, "training"), _os.path.join(_ROOT, "plots")):
    _p in _sys.path or _sys.path.insert(0, _p)
import argparse, os, sys
import numpy as np
from mb4_read import CFG, POS, load
RADX = 1.5
RULES = ((r"$g=1$", "fit_fix"), (r"$g_{\mathrm{geo}}$", "fit_gfix"), (r"$g_{\mathrm{eff}}$", "fit_gefffix"), (r"$g_{\mathrm{uniform}}$", "fit_gufix"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "bo_mb4_norm"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
    ap.add_argument("--lam", type=int, default=4); ap.add_argument("--dims", default="2,3,4"); ap.add_argument("--seeds", type=int, default=25)
    a = ap.parse_args()
    tex = [r"\begin{tabular}{llrrrr}", r"\toprule", r"$D$ & optimum at & " + " & ".join(n for n, _ in RULES) + r" \\", r"\midrule"]
    md = ["| D | optimum at | " + " | ".join(n for n, _ in RULES) + " |", "|---|---|---|---|---|---|"]
    ftex = list(tex); fmd = list(md)          # fraction of conditioned runs entering within R
    for D in [int(x) for x in a.dims.split(",")]:
        B, s_, variants = CFG[D]; rad = RADX * s_; R = B - (2 * D + 2)
        for v in variants:
            ctr = np.array(POS[D][v]); cells = []; fcells = []
            for _, arm in RULES:
                T = []
                for s in range(a.seeds):
                    r = load(a.dir, D, v, s, f"mb4_l{a.lam}")
                    if r is None: continue
                    X0 = np.asarray(r[0]["X0"]); P = np.asarray(r[1][arm]["picks"]); n0 = len(X0)
                    din = np.linalg.norm(np.vstack([X0, P]) - ctr, axis=1) <= rad
                    if din[:n0].any(): continue                      # initial design hit: excluded
                    idx = np.where(din[n0:])[0]; T.append(int(idx[0]) + 1 if len(idx) else R + 1)
                m = np.median(T); cells.append(f"$>{R}$" if m > R else f"{m:.0f}")
                fcells.append(f"{np.mean(np.array(T) <= R):.2f}")
            tex.append(f"{D} & {v} & " + " & ".join(cells) + r" \\"); md.append(f"| {D} | {v} | " + " | ".join(cells) + " |")
            ftex.append(f"{D} & {v} & " + " & ".join(fcells) + r" \\"); fmd.append(f"| {D} | {v} | " + " | ".join(fcells) + " |")
        if D != [int(x) for x in a.dims.split(",")][-1]: tex.append(r"\addlinespace"); ftex.append(r"\addlinespace")
    tex += [r"\bottomrule", r"\end{tabular}"]; ftex += [r"\bottomrule", r"\end{tabular}"]
    sfx = "" if a.lam == 4 else f"_l{a.lam}"
    open(os.path.join(a.out, f"tab_mb4_firsthit{sfx}.tex"), "w").write("\n".join(tex) + "\n")
    open(os.path.join(a.out, f"tab_mb4_firsthit{sfx}.md"), "w").write("\n".join(md) + "\n")
    open(os.path.join(a.out, f"tab_mb4_hitfrac{sfx}.tex"), "w").write("\n".join(ftex) + "\n")
    open(os.path.join(a.out, f"tab_mb4_hitfrac{sfx}.md"), "w").write("\n".join(fmd) + "\n"); print("\n".join(fmd))
    print("\n".join(md))

if __name__ == "__main__":
    main()
