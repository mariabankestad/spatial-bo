"""Figure of the controlled synthetic experiment, one row per dimension: share
of the first D+2 evaluations per spatial class, then regret per round for
each placement of the global minimum. lambda = 4, 25 seeds.
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, "experiments"), _os.path.join(_ROOT, "training"), _os.path.join(_ROOT, "plots")):
    _p in _sys.path or _sys.path.insert(0, _p)
import argparse, os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from paperstyle import COL, LAB, ORDER, GREY, TEXTWIDTH
from mb4_read import CFG, POS, ARMS, load, codim
import eval_bo as EV
ARM = dict(ARMS)
CLS = {2: ["interior", "face", "corner"], 3: ["interior", "face", "edge", "corner"], 4: ["interior", "face", "edge", "3-wall", "corner"]}
ADDR = ("corner", "edge", "face", "interior")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "bo_mb4_norm"))
    ap.add_argument("--ctrl_dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "bo_mb4_ref"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
    ap.add_argument("--lam", type=int, default=4); ap.add_argument("--seeds", type=int, default=25); ap.add_argument("--dims", default="2,3,4")
    a = ap.parse_args(); dims = [int(x) for x in a.dims.split(",")]
    ncol = 2 + len(ADDR)          # bar column, a narrow spacer, then the regret columns
    fig = plt.figure(figsize=(TEXTWIDTH, 1.42 * len(dims) + 0.5))
    gs = gridspec.GridSpec(len(dims), ncol, figure=fig, width_ratios=[1.95, 0.2] + [1] * len(ADDR), wspace=0.11, hspace=0.34,
                           left=0.095, right=0.99, top=0.95, bottom=0.145)
    rs = np.random.default_rng(1)
    for i, D in enumerate(dims):
        B, s_, variants = CFG[D]; h = B ** (-1 / D); rad = 1.5 * s_; W = D + 2
        rng = np.random.default_rng(0); Uv = rng.random((300000, D)); pu = np.bincount(codim(Uv, h), minlength=D + 1) / len(Uv)
        # ---- data
        per_seed = {n: np.zeros((a.seeds, D + 1)) for n in ORDER}; rnd = np.zeros((a.seeds, D + 1))
        reg = {n: {v: [] for v in variants} for n in ORDER}
        for v in variants:
            hs = EV.hstar(f"mb4_{D}d_{v}")
            for s in range(a.seeds):
                r = load(a.dir, D, v, s, f"mb4_l{a.lam}")
                if r is None: continue
                n0 = len(r[0]["X0"])
                for n in ORDER:
                    P = np.asarray(r[1][ARM[n]]["picks"]); per_seed[n][s] += np.bincount(codim(P[:W], h), minlength=D + 1)
                    hall = np.concatenate([np.asarray(r[0]["h0"]), np.asarray(r[1][ARM[n]]["htrue"])])
                    reg[n][v].append([hs - hall[:n0 + t + 1].max() for t in range(len(hall) - n0)])
                rc = load(a.ctrl_dir, D, v, s, "mb4_ctrl")
                if rc is not None and "random" in rc[1]: rnd[s] += np.bincount(codim(np.asarray(rc[1]["random"]["picks"])[:W], h), minlength=D + 1)
        # ---- column 0: share of the first evaluations by location, one bar per rule, grey bar = share of the domain
        ax = fig.add_subplot(gs[i, 0]); k = np.arange(D + 1); groups = ["volume"] + list(ORDER); w = 0.8 / len(groups)
        FLOOR = 0.004   # shares of exactly zero are drawn at the floor (stated in the caption)
        comp = {n: per_seed[n].sum(0) / per_seed[n].sum() for n in ORDER}; comp["volume"] = pu
        for j, n in enumerate(groups):
            x = k + (j - (len(groups) - 1) / 2) * w
            y = np.maximum(comp[n], FLOOR)
            if n == "volume":
                ax.bar(x, y - FLOOR, w * 0.9, bottom=FLOOR, color="#c9c9c9", lw=0, label="share of the domain" if i == 0 else None)
            else:
                ax.bar(x, y - FLOOR, w * 0.9, bottom=FLOOR, color=COL[n], lw=0)
                boots = np.array([(lambda b: b / max(b.sum(), 1))(per_seed[n][rs.integers(0, a.seeds, a.seeds)].sum(0)) for _ in range(1500)])
                lo, hi = np.maximum(np.percentile(boots, 5, 0), FLOOR), np.maximum(np.percentile(boots, 95, 0), FLOOR)
                ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="none", ecolor="#333", elinewidth=0.45, capsize=1.0)
        cs = comp["raw"][-1]; xr = k[-1] + (groups.index("raw") - (len(groups) - 1) / 2) * w
        ax.annotate(f"{100 * cs:.0f}\\%", (xr, cs), xytext=(5, 6), textcoords="offset points", ha="left", va="bottom", fontsize=6.5, color="#333")
        ax.set_yscale("log"); ax.set_ylim(FLOOR, 1.35)
        ax.set_yticks([0.01, 0.1, 1.0]); ax.set_yticklabels([r"1\%", r"10\%", r"100\%"])
        ax.set_xticks(k); ax.set_xticklabels(CLS[D], fontsize=6.2, rotation=20, ha="right", rotation_mode="anchor")
        ax.grid(axis="x", visible=False); ax.tick_params(labelsize=6.5)
        ax.set_ylabel("share of first evaluations", fontsize=6.8, labelpad=1); ax.grid(axis="y", which="minor", visible=False)
        pos = ax.get_position(); fig.text(0.006, (pos.y0 + pos.y1) / 2, f"$D = {D}$", rotation=90, va="center", ha="center", fontsize=9)
        if i == 0: ax.set_title("where the first evaluations go", fontsize=8)
        # ---- regret columns
        ymax = 0; ref = None
        for c, v in enumerate(ADDR):
            ax = fig.add_subplot(gs[i, 2 + c], sharey=ref) if ref is not None else fig.add_subplot(gs[i, 2 + c])
            if v not in variants:
                ax.axis("off"); ax.text(0.5, 0.5, f"no {v} well\nin $D={D}$", ha="center", va="center", fontsize=6.5, color="#888", transform=ax.transAxes)
                if i == 0: ax.set_title(f"optimum: {v}", fontsize=8)
                if ref is None: ref = ax
                continue
            for n in ORDER:
                Rm = np.array(reg[n][v]); m = Rm.mean(0); sem = Rm.std(0, ddof=1) / np.sqrt(Rm.shape[0]); t_ = np.arange(len(m))
                ax.fill_between(t_, m - 1.645 * sem, m + 1.645 * sem, color=COL[n], alpha=0.13, lw=0)
                ax.plot(t_, m, color=COL[n], lw=1.25, label=LAB[n] if (i == 0 and c == 0) else None); ymax = max(ymax, (m + 1.645 * sem).max())
            for t in (W, 2 * W): ax.axvline(t, color="#bbb", lw=0.6, ls=":")
            ax.tick_params(labelsize=6.5); ax.set_ylim(0, None)
            if ref is None: ref = ax
            else: ax.tick_params(labelleft=False)
            if i == 0: ax.set_title(f"optimum: {v}", fontsize=8)
            if i == len(dims) - 1: ax.set_xlabel("BO round $t$", fontsize=7.5)
            if c == 0: ax.set_ylabel("simple regret", fontsize=7.5)
    reg_ax = [a_ for a_ in fig.axes if a_.get_legend_handles_labels()[1] and "$g" in "".join(a_.get_legend_handles_labels()[1])][0]
    hv, lv = fig.axes[0].get_legend_handles_labels()
    h_, l_ = reg_ax.get_legend_handles_labels(); h_ = hv + h_; l_ = lv + l_
    fig.legend(h_, l_, loc="lower center", ncol=len(l_), bbox_to_anchor=(0.5, 0.005), columnspacing=1.4, handlelength=1.4, fontsize=7)
    fig.savefig(os.path.join(a.out, "fig_mb4_combined.pdf")); fig.savefig(os.path.join(a.out, "fig_mb4_combined.png"), dpi=170)
    print("wrote fig_mb4_combined")

if __name__ == "__main__":
    main()
