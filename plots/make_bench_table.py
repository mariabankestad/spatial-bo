"""Appendix tables of the benchmark experiment: regret at the end of the budget
for all rules and tasks, and Delta r against g = 1 for the weighted rules.
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, "experiments"), _os.path.join(_ROOT, "training"), _os.path.join(_ROOT, "plots")):
    _p in _sys.path or _sys.path.insert(0, _p)
import argparse, json, os

ARMS_ALL = (("fit_fix", r"$g=1$"), ("fit_fit", "std.\\ UCB"), ("fit_gfix", r"$g_{\mathrm{geo}}$"),
            ("fit_gefffix", r"$g_{\mathrm{eff}}$"), ("fit_gufix", r"$g_{\mathrm{uniform}}$"), ("random", "random"))
EQ = (("fit_gfix", r"$g_{\mathrm{geo}}$"), ("fit_gefffix", r"$g_{\mathrm{eff}}$"), ("fit_gufix", r"$g_{\mathrm{uniform}}$"))
NAME = {"hartmann_3d": "Hartmann 3D", "hartmann_4d": "Hartmann 4D", "hartmann_6d": "Hartmann 6D",
        "styblinski_tang_3d": "Styblinski-Tang 3D", "michalewicz_5d": "Michalewicz 5D", "levy2d": "Levy 2D",
        "levy_3d": "Levy 3D", "levy_4d": "Levy 4D", "beale": "Beale 2D", "branin": "Branin 2D",
        "camel6": "Six-hump camel 2D", "rastrigin_3d": "Rastrigin 3D", "rosenbrock_4d": "Rosenbrock 4D",
        "zakharov_5d": "Zakharov 5D"}
GROUP = {"suite": "benchmark suite", "stress": "stress test", "dev": "used during development"}


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--design", default=os.path.join(here, "..", "results", "bench_score_bench_l2_design.json"))
    ap.add_argument("--pool", default=os.path.join(here, "..", "results", "bench_score_bench_l2.json"))
    ap.add_argument("--out", default=os.path.join(here, "out"))
    a = ap.parse_args()
    d = json.load(open(a.design))["tasks"]; p = json.load(open(a.pool))["tasks"]
    order = sorted(d, key=lambda f: (("suite", "stress", "dev").index(d[f]["group"]), d[f]["d_boundary"]))

    def cell(c, prec=3):
        hw = (c["hi"] - c["lo"]) / 2
        return f"{c['mean']:.{prec}f} $\\pm$ {hw:.{prec}f}"

    # ---- table 1: regret at the budget, all arms
    rows_tex, rows_md = [], []
    last = None
    for f in order:
        r = d[f]
        if r["group"] != last:
            rows_tex.append(f"\\midrule\n\\multicolumn{{{4 + len(ARMS_ALL)}}}{{l}}{{\\emph{{{GROUP[r['group']]}}}}}\\\\")
            rows_md.append(f"| *{GROUP[r['group']]}* |" + " |" * (3 + len(ARMS_ALL)))
            last = r["group"]
        cells = []
        for arm, _ in ARMS_ALL:
            src = d if arm in r["abs"] else p
            cells.append(cell(src[f]["abs"][arm]["t_final"]))
        rows_tex.append(f"{NAME[f]} & {r['D']} & {r['d_boundary']:.2f} & {r['class']} & " + " & ".join(cells) + r" \\")
        rows_md.append(f"| {NAME[f]} | {r['D']} | {r['d_boundary']:.2f} | {r['class']} | " + " | ".join(c.replace('$\\pm$', '±') for c in cells) + " |")
    head = " & ".join(n for _, n in ARMS_ALL)
    tex = ("\\begin{tabular}{lrrl" + "r" * len(ARMS_ALL) + "}\n\\toprule\n"
           f"task & $D$ & $d_\\partial$ & class & {head} \\\\\n" + "\n".join(rows_tex) + "\n\\bottomrule\n\\end{tabular}\n")
    open(os.path.join(a.out, "tab_bench_final.tex"), "w").write(tex)
    md = ("| task | D | d_bnd | class | " + " | ".join(n.replace('$', '').replace('\\mathrm', '').replace('\\ ', ' ') for _, n in ARMS_ALL) + " |\n"
          "|---|---|---|---|" + "---|" * len(ARMS_ALL) + "\n" + "\n".join(rows_md) + "\n")
    open(os.path.join(a.out, "tab_bench_final.md"), "w").write(md)

    # ---- table 2: Delta r vs g=1, equalizers, early and budget
    rows_tex, rows_md = [], []
    last = None
    for f in order:
        r = d[f]
        if r["group"] != last:
            rows_tex.append(f"\\midrule\n\\multicolumn{{{2 + 2 * len(EQ)}}}{{l}}{{\\emph{{{GROUP[r['group']]}}}}}\\\\")
            rows_md.append(f"| *{GROUP[r['group']]}* |" + " |" * (1 + 2 * len(EQ)))
            last = r["group"]
        cells = []
        for ck in ("t_early", "t_final"):
            for arm, _ in EQ:
                c = r["delta"][arm][ck]
                cells.append(f"{c['mean']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}]")
        rows_tex.append(f"{NAME[f]} & {r['D']} & " + " & ".join(cells) + r" \\")
        rows_md.append(f"| {NAME[f]} | {r['D']} | " + " | ".join(cells) + " |")
    head = " & ".join(n for _, n in EQ)
    tex = ("\\begin{tabular}{lr" + "r" * (2 * len(EQ)) + "}\n\\toprule\n"
           f" & & \\multicolumn{{{len(EQ)}}}{{c}}{{after $2(D+2)$ picks}} & \\multicolumn{{{len(EQ)}}}{{c}}{{at the budget}} \\\\\n"
           f"task & $D$ & {head} & {head} \\\\\n" + "\n".join(rows_tex) + "\n\\bottomrule\n\\end{tabular}\n")
    open(os.path.join(a.out, "tab_bench_delta.tex"), "w").write(tex)
    md = ("| task | D | " + " | ".join(f"early {n}" for _, n in EQ) + " | " + " | ".join(f"budget {n}" for _, n in EQ) + " |\n"
          "|---|---|" + "---|" * (2 * len(EQ)) + "\n" + "\n".join(rows_md) + "\n").replace('$', '').replace('\\mathrm', '')
    open(os.path.join(a.out, "tab_bench_delta.md"), "w").write(md)
    print("saved tab_bench_final.{tex,md}, tab_bench_delta.{tex,md} in", a.out)


if __name__ == "__main__":
    main()
