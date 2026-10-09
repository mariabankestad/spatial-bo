"""Task table of the benchmark experiment: location of the optimum, its distance
to the nearest wall, that distance relative to the budget's spacing, and the
resulting class. Writes results/benchmark_tasks.{json,md}.

    python bench_tasks.py
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, "experiments"), _os.path.join(_ROOT, "training"), _os.path.join(_ROOT, "plots")):
    _p in _sys.path or _sys.path.insert(0, _p)
import argparse, json, os
import numpy as np, torch
from scipy.optimize import minimize
from run_bo import DT
from spatialbo.functions import standardized

SUITE = ["beale", "levy2d", "levy_3d", "hartmann_3d", "styblinski_tang_3d", "hartmann_4d", "levy_4d", "michalewicz_5d"]
STRESS = ["hartmann_6d"]
DEV = ["branin", "camel6", "rastrigin_3d", "rosenbrock_4d", "zakharov_5d"]
BUDGET = {2: 25, 3: 40, 4: 55, 5: 70, 6: 85}
CLS = {0: "interior", 1: "face", 2: "edge", 3: "3-wall"}

def optimum(func, n_top=32):
    h, D = standardized(func)
    P = torch.quasirandom.SobolEngine(D, scramble=True, seed=1).draw(2 ** 17).to(DT)
    vals = h(P); idx = vals.argsort(descending=True)[:n_top]
    obj = lambda x: -float(h(torch.tensor(x, dtype=DT).unsqueeze(0)))
    best = None
    for x0 in P[idx].numpy():
        r = minimize(obj, x0, method="L-BFGS-B", bounds=[(0, 1)] * D)
        if best is None or r.fun < best.fun: best = r
    return best.x, -best.fun, D

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "benchmark_tasks"))
    a = ap.parse_args(); rows = []
    for group, names in (("suite", SUITE), ("stress", STRESS), ("dev", DEV)):
        for f in names:
            x, hs, D = optimum(f); B = BUDGET[D]; h = B ** (-1 / D)
            db = float(np.min(np.minimum(x, 1 - x))); nwall = int(np.sum(np.minimum(x, 1 - x) < h / 2))
            cls = "corner" if nwall == D else CLS.get(nwall, f"{nwall}-wall")
            rows.append({"task": f, "group": group, "D": D, "B": B, "n0": 2 * D + 2, "rounds": B - 2 * D - 2, "t_early": 2 * (D + 2),
                         "x_opt": [round(float(v), 4) for v in x], "h_opt_std": round(hs, 4), "d_boundary": round(db, 4),
                         "d_over_h": round(db / h, 3), "n_wall": nwall, "class": cls})
            print(f"{group:6s} {f:20s} D={D} B={B:2d} x*={np.round(x,3)} d={db:.3f} d/h={db/h:.2f} class={cls}")
    json.dump(rows, open(a.out + ".json", "w"), indent=1)
    md = ["| group | task | D | B | rounds | t_early | optimum (normalised) | d_boundary | d/h | class |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows: md.append(f"| {r['group']} | {r['task']} | {r['D']} | {r['B']} | {r['rounds']} | {r['t_early']} | {r['x_opt']} | {r['d_boundary']:.3f} | {r['d_over_h']:.2f} | {r['class']} |")
    open(a.out + ".md", "w").write("\n".join(md) + "\n"); print("wrote", a.out + ".{json,md}")

if __name__ == "__main__":
    main()
