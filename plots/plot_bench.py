"""Benchmark figure: paired regret difference against g = 1 per task, after
2(D+2) evaluations and at the end of the budget; tasks ordered by the
distance of the optimum to the boundary. Reads results/bench_score_<tag>.json.

    --mode delta | relative | init     absolute, relative to r_{g=1}, or relative to the initial regret
    --all                               every task
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, "experiments"), _os.path.join(_ROOT, "training"), _os.path.join(_ROOT, "plots")):
    _p in _sys.path or _sys.path.insert(0, _p)
import argparse, json, os
import numpy as np
import matplotlib.pyplot as plt
from paperstyle import COL, LAB, TEXTWIDTH, GREY

# main figure: the controlled arms (only g changes, sigma at ell_ref in all).
# --all adds standard GP-LCB (sigma at the fitted ell) and the dev tasks.
ARMS_MAIN = (("fit_gfix", "g_info"), ("fit_gefffix", "g_eff"), ("fit_gufix", "g_unif"))
ARMS_ALL = (("fit_fit", "std_bo"),) + ARMS_MAIN
SKIP = {"beale", "zakharov_5d"}       # solved to zero regret by every rule: no information
SHORT = {"hartmann_3d": "hartmann", "hartmann_4d": "hartmann", "hartmann_6d": "hartmann",
         "styblinski_tang_3d": "styblinski", "michalewicz_5d": "michalewicz", "levy2d": "levy",
         "levy_3d": "levy", "levy_4d": "levy", "branin": "branin", "camel6": "camel",
         "rastrigin_3d": "rastrigin", "rosenbrock_4d": "rosenbrock", "beale": "beale", "zakharov_5d": "zakharov"}


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--score", default=os.path.join(here, "..", "results", "bench_score_bench_l2_design.json"))
    ap.add_argument("--suffix", default="", help="output-name suffix, e.g. _poolmean for the appendix version")
    ap.add_argument("--out", default=os.path.join(here, "out"))
    ap.add_argument("--mode", choices=("delta", "relative", "abs", "init"), default="relative")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--final", action="store_true", help="add a fourth panel at the budget")
    a = ap.parse_args()
    d = json.load(open(a.score))
    groups = ("suite", "stress", "dev") if a.all else ("suite", "stress")
    tasks = [(f, r) for f, r in d["tasks"].items() if r["group"] in groups and f not in SKIP]
    tasks.sort(key=lambda kv: (groups.index(kv[1]["group"]), kv[1]["d_boundary"]))
    n = len(tasks)
    ARMS = ARMS_ALL if a.all else ARMS_MAIN
    ARMS = tuple((arm, key) for arm, key in ARMS if all(arm in r["delta"] for _, r in tasks))   # skip arms this score lacks
    CKS = (("t_early", r"after $2(D+2)$ picks"), ("t3", r"after $3(D+2)$ picks"), ("t4", r"after $4(D+2)$ picks"))
    if a.final:
        CKS = CKS + (("t_final", "at the budget"),)
    fig, axes = plt.subplots(1, len(CKS), figsize=(TEXTWIDTH, 2.45 if not a.all else 2.7), sharey=(a.mode in ("relative", "init")))
    w = 0.8 / (len(ARMS) + 1)
    for ax, (ck, title) in zip(axes, CKS):
        for j, (arm, key) in enumerate(ARMS):
            x = np.arange(n) + (j - (len(ARMS) - 1) / 2) * w
            m = np.array([r["delta"][arm][ck]["mean"] for _, r in tasks])
            lo = np.array([r["delta"][arm][ck]["lo"] for _, r in tasks])
            hi = np.array([r["delta"][arm][ck]["hi"] for _, r in tasks])
            if a.mode == "relative":
                base = np.array([r["abs"]["fit_fix"][ck]["mean"] for _, r in tasks])
                m, lo, hi = 100 * m / base, 100 * lo / base, 100 * hi / base
            elif a.mode == "init":
                base = np.array([r["r0_mean"] for _, r in tasks])   # regret after the initial design
                m, lo, hi = 100 * m / base, 100 * lo / base, 100 * hi / base
            if a.mode == "abs":
                # linear axis clipped at YCLIP; off-scale means are drawn at the clip with the value written
                YCLIP = -0.25
                off = m < YCLIP
                mc = np.where(off, YCLIP, m); loc = np.maximum(lo, YCLIP); hic = np.where(off, YCLIP, hi)
                ax.errorbar(x[~off], mc[~off], yerr=[mc[~off] - loc[~off], hic[~off] - mc[~off]], fmt="o", color=COL[key],
                            ms=3, lw=0.8, capsize=1.5, label=LAB[key] if ck == "t_early" else None)
                for xi, mi in zip(x[off], m[off]):
                    ax.plot(xi, YCLIP, marker="v", color=COL[key], ms=4, clip_on=False)
                    ax.text(xi, YCLIP - 0.012, f"{mi:+.2f}", ha="center", va="top", fontsize=5.2, color=COL[key], rotation=90)
                continue
            ax.errorbar(x, m, yerr=[m - lo, hi - m], fmt="o", color=COL[key], ms=3, lw=0.8, capsize=1.5,
                        label=LAB[key] if ck == "t_early" else None)
        ax.axhline(0, color="k", lw=0.6)
        # divider between the fixed in advance suite and the stress / dev tasks
        cuts = [i for i in range(1, n) if tasks[i][1]["group"] != tasks[i - 1][1]["group"]]
        for c in cuts:
            ax.axvline(c - 0.5, color=GREY, lw=0.6, ls=":")
        ax.set_xticks(range(n))
        ax.set_xticklabels([f"{SHORT[f]} ({r['D']}D)" for f, r in tasks], rotation=55, ha="right", fontsize=6.3)
        ax.set_xlim(-0.6, n - 0.4)
        ax.set_title(title, pad=4)
        if a.mode == "delta":
            ax.set_yscale("symlog", linthresh=0.05, linscale=0.8)
            ax.set_ylabel(r"$\Delta r$ vs.\ $g=1$ (regret units)" if ck == "t_early" else "")
        elif a.mode == "abs":
            ax.set_ylabel(r"$\Delta r$ vs.\ $g=1$ (regret units)" if ck == "t_early" else "")
            ax.set_ylim(-0.25, 0.05)
        elif a.mode == "init":
            ax.set_ylabel(r"$\Delta r$ vs.\ $g=1$, \% of initial regret" if ck == "t_early" else "")
        else:
            ax.set_ylabel(r"regret change vs.\ $g=1$ (\%)" if ck == "t_early" else "")
    if a.mode == "relative":
        axes[0].set_ylim(-100, 125)
        for ax in axes:
            ax.set_yticks([-100, -50, 0, 50, 100])
    for ax in axes:
        ax.set_axisbelow(True)
    fig.legend(loc="lower center", ncol=len(ARMS), bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(left=0.085, right=0.99, top=0.92, bottom=0.40 if not a.all else 0.42, wspace=0.10)
    os.makedirs(a.out, exist_ok=True)
    stem = f"fig_bench_{a.mode}" + ("_all" if a.all else "") + a.suffix
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(a.out, f"{stem}.{ext}"))
    print("saved", os.path.join(a.out, stem + ".pdf"))


if __name__ == "__main__":
    main()
