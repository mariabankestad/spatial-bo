"""Regret per round on benchmark tasks: mean over the paired seeds with a 90%
band, the four controlled rules and uniform random.

    python plot_bench_traj.py          # main text
    python plot_bench_traj.py --all    # every task (appendix)
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, "experiments"), _os.path.join(_ROOT, "training"), _os.path.join(_ROOT, "plots")):
    _p in _sys.path or _sys.path.insert(0, _p)
import argparse, json, os
import numpy as np
import matplotlib.pyplot as plt
from paperstyle import COL, LAB, GREY, TEXTWIDTH, COLWIDTH

ARMS = (("fit_fix", "raw"), ("fit_gfix", "g_info"), ("fit_gefffix", "g_eff"), ("fit_gufix", "g_unif"))
TITLE = {"hartmann_3d": "Hartmann 3D\noptimum on a face", "hartmann_4d": "Hartmann 4D", "hartmann_6d": "Hartmann 6D\noptimum on an edge",
         "styblinski_tang_3d": "Styblinski-Tang 3D\ninterior optimum", "michalewicz_5d": "Michalewicz 5D", "levy2d": "Levy 2D",
         "levy_3d": "Levy 3D", "levy_4d": "Levy 4D\ninterior optimum", "beale": "Beale 2D", "branin": "Branin 2D (face, dev)",
         "camel6": "Six-hump camel 2D (dev)", "rastrigin_3d": "Rastrigin 3D (dev)", "rosenbrock_4d": "Rosenbrock 4D (dev)",
         "zakharov_5d": "Zakharov 5D (dev)"}


def curves(rdir, func, tag, hs):
    out = {}
    for fn in sorted(os.listdir(rdir)):
        if not (fn.startswith(func + "_s") and fn.endswith(f"_{tag}.npz")):
            continue
        z = np.load(os.path.join(rdir, fn), allow_pickle=True)
        meta = z["meta"].item(); res = z["results"].item()
        h0 = float(np.max(meta["h0"]))
        for arm, r in res.items():
            inc = np.maximum.accumulate(np.concatenate([[h0], r["htrue"]]))
            out.setdefault(arm, []).append(hs - inc)     # index 0 = after the initial design
    return {k: np.array(v) for k, v in out.items()}, int(meta["D"])


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--dir", default=os.path.join(here, "..", "results", "bo_bench"))
    ap.add_argument("--score", default=os.path.join(here, "..", "results", "bench_score_bench_l2_design.json"))
    ap.add_argument("--tag", default="bench_l2_design", help="tag of the equalizer runs")
    ap.add_argument("--base_tag", default="bench_l2", help="tag holding g=1 (fit_fix) and random for the same seeds")
    ap.add_argument("--tasks", default="hartmann_3d,styblinski_tang_3d,hartmann_6d,levy_4d")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", default=os.path.join(here, "out"))
    a = ap.parse_args()
    hs_all = json.load(open(os.path.join(here, "..", "results", "hstar.json")))
    sc = json.load(open(a.score))["tasks"]
    if a.all:
        tasks = sorted(sc, key=lambda f: (("suite", "stress", "dev").index(sc[f]["group"]), sc[f]["D"], f))
        ncol = 4; nrow = int(np.ceil(len(tasks) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(TEXTWIDTH, 1.55 * nrow + 0.4))
        axes = axes.ravel()
    else:
        tasks = a.tasks.split(",")
        if len(tasks) > 2:                       # 2 x 2 grid at column width
            ncol = 2; nrow = int(np.ceil(len(tasks) / ncol))
            fig, axes = plt.subplots(nrow, ncol, figsize=(COLWIDTH, 1.5 * nrow + 0.8))
            axes = axes.ravel()
        else:
            fig, axes = plt.subplots(1, len(tasks), figsize=(COLWIDTH, 2.15))
            axes = np.atleast_1d(axes)
    for ax, f in zip(axes, tasks):
        cv, D = curves(a.dir, f, a.tag, hs_all[f])
        if a.base_tag != a.tag:
            cb, _ = curves(a.dir, f, a.base_tag, hs_all[f])
            for arm in ("fit_fix", "random", "fit_fit"):
                if arm in cb and arm not in cv:
                    cv[arm] = cb[arm]
        t = np.arange(cv["fit_fix"].shape[1])
        for arm, key in ARMS:
            m = cv[arm].mean(0); se = cv[arm].std(0, ddof=1) / np.sqrt(len(cv[arm]))
            ax.plot(t, m, color=COL[key], label=LAB[key])
            ax.fill_between(t, m - 1.645 * se, m + 1.645 * se, color=COL[key], alpha=0.18, lw=0)
        if "random" in cv:
            m = cv["random"].mean(0)
            ax.plot(t, m, color=GREY, ls="--", lw=1.0, label="uniform random")
        ax.axvline(2 * (D + 2), color=GREY, lw=0.6, ls=":")
        ax.text(2 * (D + 2), 1.01, r"$2(D{+}2)$", transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=6, color=GREY)
        ax.set_yscale("log")
        lo_, hi_ = ax.get_ylim()
        cand = [m * 10.0 ** k for k in range(-6, 2) for m in (1, 2, 3, 5, 7)]
        ticks = [v for v in cand if lo_ <= v <= hi_]
        if len(ticks) > 6:
            ticks = [v for v in ticks if round(v / 10 ** np.floor(np.log10(v)), 6) in (1, 2, 5)]
        ax.set_yticks(ticks); ax.set_yticklabels([f"{v:g}" for v in ticks]); ax.yaxis.set_minor_formatter(plt.NullFormatter())
        ttl = TITLE.get(f, f)
        if a.all and "\n" in ttl:
            ttl = ttl.replace("\n", " (") + ")"
        ax.set_title(ttl, fontsize=7.5 if not a.all else 8, pad=9)
        if a.all:
            ax.set_xlabel("BO round $t$")
        ax.set_xlim(0, t[-1])
    for ax in axes[len(tasks):]:
        ax.axis("off")
    axes[0].set_ylabel("simple regret")
    if a.all or len(tasks) > 2:
        for i in range(0, len(tasks), ncol):
            axes[i].set_ylabel("simple regret")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3 if not a.all else 5, bbox_to_anchor=(0.5, -0.01),
               columnspacing=1.0, handlelength=1.5, fontsize=6.5 if not a.all else 7)
    if a.all:
        fig.subplots_adjust(left=0.085, right=0.99, top=0.96, bottom=0.10, wspace=0.34, hspace=0.62)
    elif len(tasks) > 2:
        fig.supxlabel("BO round $t$", fontsize=8, y=0.105)
        fig.subplots_adjust(left=0.14, right=0.98, top=0.89, bottom=0.19, wspace=0.42, hspace=0.70)
    else:
        fig.supxlabel("BO round $t$", fontsize=8, y=0.215)
        fig.subplots_adjust(left=0.13, right=0.98, top=0.84, bottom=0.33, wspace=0.42)
    os.makedirs(a.out, exist_ok=True)
    stem = "fig_bench_traj" + ("_all" if a.all else "")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(a.out, f"{stem}.{ext}"))
    print("saved", os.path.join(a.out, stem + ".pdf"))


if __name__ == "__main__":
    main()
