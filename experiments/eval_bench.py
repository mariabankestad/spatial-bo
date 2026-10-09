"""Score the benchmark experiment: simple regret per task and rule, paired over
seeds with the g = 1 rule at D+2, 2(D+2), (B-n0)/2 and B-n0 rounds, with 90%
percentile-bootstrap intervals. Negative Delta r = lower regret than g = 1.

    python eval_bench.py --tag bench_l2_design --g_norm design --baseline_tag bench_l2
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, "experiments"), _os.path.join(_ROOT, "training"), _os.path.join(_ROOT, "plots")):
    _p in _sys.path or _sys.path.insert(0, _p)
import argparse
import json
import os

import numpy as np

CI = 0.90
N_BOOT = 20000
ARMS_MAIN = ("fit_fit", "fit_gfix", "fit_gufix", "fit_gefffix")
ARMS_ALL = ("fit_fix",) + ARMS_MAIN + ("random",)
NAMES = {"fit_fix": "g=1", "fit_fit": "std GP-LCB", "fit_gfix": "g_info",
         "fit_gufix": "g_unif", "fit_gefffix": "g_eff", "random": "random"}


def boot(d, seed=0):
    d = np.asarray(d, float)
    if len(d) < 2:
        return float(d.mean()) if len(d) else float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    m = rng.choice(d, size=(N_BOOT, len(d)), replace=True).mean(axis=1)
    a = (1 - CI) / 2 * 100
    return float(d.mean()), float(np.percentile(m, a)), float(np.percentile(m, 100 - a))


def load_task(outdir, func, tag, hs, lam, g_norm="mean"):
    """Per-arm regret curves (seeds x rounds), the seed ids, and the protocol."""
    curves, seeds, proto = {}, [], None
    for s in range(1000):
        p = os.path.join(outdir, f"{func}_s{s}_{tag}.npz")
        if not os.path.exists(p):
            continue
        z = np.load(p, allow_pickle=True)
        meta = z["meta"].item(); res = z["results"].item()
        pr = {"D": int(meta["D"]), "n0": int(meta["n0"]), "rounds": int(meta["rounds"]),
              "lam": float(meta["lam"]), "g_norm": meta.get("g_norm"),
              "g_ell": meta.get("g_ell"), "ell_rule": meta.get("ell_rule"),
              "noise_model": float(meta.get("noise_model", -1))}
        want = {"n0": 2 * pr["D"] + 2, "lam": float(lam), "g_norm": g_norm,
                "g_ell": "ref", "ell_rule": "paper"}
        bad = {k: (pr[k], v) for k, v in want.items() if pr[k] != v}
        if bad:
            raise SystemExit(f"PROTOCOL MISMATCH in {p}: {bad}")
        if proto is None:
            proto = pr
        elif pr != proto:
            raise SystemExit(f"PROTOCOL DIFFERS BETWEEN SEEDS in {p}: {pr} vs {proto}")
        h0 = float(np.max(meta["h0"]))
        curves.setdefault("_r0", {})[s] = hs - h0      # regret after the initial design, shared by all arms
        for arm, r in res.items():
            inc = np.maximum.accumulate(np.concatenate([[h0], r["htrue"]]))[1:]
            curves.setdefault(arm, {})[s] = hs - inc
        seeds.append(s)
    return curves, sorted(seeds), proto


def score(outdir, tag, tasks_json, lam, baseline="fit_fix", g_norm="mean", baseline_tag=None):
    cands = [os.path.join(outdir, "hstar.json"),
             os.path.join(os.path.dirname(outdir.rstrip("/")), "hstar.json")]
    hs_path = next((c for c in cands if os.path.exists(c)), None)
    if hs_path is None:
        raise SystemExit(f"no hstar.json in {cands}")
    hs_all = json.load(open(hs_path))
    rows = json.load(open(tasks_json))
    out = {"tag": tag, "baseline": baseline, "baseline_tag": baseline_tag or tag, "g_norm": g_norm, "ci": CI, "n_boot": N_BOOT,
           "delta_sign": "r_arm - r_baseline (negative = arm better)",
           "tasks": {}}
    for row in rows:
        func = row["task"]
        if func not in hs_all:
            print(f"  {func}: no h* in cache, skipped"); continue
        curves, seeds, proto = load_task(outdir, func, tag, hs_all[func], lam, g_norm)
        if not seeds:
            continue
        if baseline_tag and baseline_tag != tag:
            # the baseline arm from another campaign with the same seeds
            # (e.g. g=1 from bench_l2 for the design-normalised equalizers)
            bc, bs, bp = load_task(outdir, func, baseline_tag, hs_all[func], lam, "mean")
            bc.pop("_r0", None)
            if baseline in bc:
                curves[baseline] = bc[baseline]
                seeds = sorted(set(seeds) & set(bs))
        D, R = proto["D"], proto["rounds"]
        cks = {"t1": D + 2, "t_early": 2 * (D + 2), "t3": min(3 * (D + 2), R), "t4": min(4 * (D + 2), R),
               "t_mid": R // 2, "t_final": R}
        r0 = curves.pop("_r0", {})
        rec = {"group": row["group"], "D": D, "B": row["B"], "n0": proto["n0"],
               "r0_mean": float(np.mean([r0[s] for s in seeds if s in r0])) if r0 else float("nan"),
               "rounds": R, "d_boundary": row["d_boundary"], "d_over_h": row["d_over_h"],
               "class": row["class"], "n_seeds": len(seeds), "seeds": seeds,
               "checkpoints": cks, "protocol": proto, "abs": {}, "delta": {}}
        for arm, cv in curves.items():
            rec["abs"][arm] = {}
            for name, t in cks.items():
                v = [cv[s][t - 1] for s in seeds if s in cv]
                m, lo, hi = boot(v)
                rec["abs"][arm][name] = {"mean": m, "lo": lo, "hi": hi, "n": len(v)}
        base = curves.get(baseline, {})
        for arm, cv in curves.items():
            if arm == baseline:
                continue
            rec["delta"][arm] = {}
            for name, t in cks.items():
                common = [s for s in seeds if s in cv and s in base]
                d = np.array([cv[s][t - 1] - base[s][t - 1] for s in common])
                m, lo, hi = boot(d)
                rec["delta"][arm][name] = {
                    "mean": m, "lo": lo, "hi": hi, "median": float(np.median(d)) if len(d) else float("nan"),
                    "wins": int((d < 0).sum()), "n": len(d)}
        out["tasks"][func] = rec
    return out


def fmt(c):
    return f"{c['mean']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}]"


def report(out):
    tag, base = out["tag"], out["baseline"]
    lines = [f"# Benchmark score, tag {tag}", "",
             f"Delta r = r_arm - r_{NAMES.get(base, base)}; negative = arm better than g=1. "
             f"{int(out['ci']*100)}% paired percentile bootstrap ({out['n_boot']} resamples). "
             "Regret in standardised units. Tasks ordered by d_boundary.", ""]
    tasks = sorted(out["tasks"].items(), key=lambda kv: (kv[1]["group"] != "suite", kv[1]["d_boundary"]))
    for ck, title in (("t_early", "t_early = 2(D+2) picks"), ("t3", "3(D+2) picks"), ("t4", "4(D+2) picks"),
                      ("t_mid", "half the rounds"), ("t_final", "final budget"), ("t1", "t1 = D+2 picks (appendix)")):
        lines += [f"## Delta r at {title}", "",
                  "| task | grp | D | d_bnd | class | n | " + " | ".join(NAMES[a] for a in ARMS_MAIN) + " |",
                  "|---|---|---|---|---|---|" + "---|" * len(ARMS_MAIN)]
        for f, r in tasks:
            cells = [fmt(r["delta"][a][ck]) + f" ({r['delta'][a][ck]['wins']}/{r['delta'][a][ck]['n']})"
                     if a in r["delta"] else "--" for a in ARMS_MAIN]
            lines.append(f"| {f} | {r['group']} | {r['D']} | {r['d_boundary']:.3f} | {r['class']} | "
                         f"{r['n_seeds']} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## Absolute regret (mean) at t_early / final", "",
              "| task | " + " | ".join(NAMES.get(a, a) for a in ARMS_ALL) + " |",
              "|---|" + "---|" * len(ARMS_ALL)]
    for f, r in tasks:
        cells = []
        for a in ARMS_ALL:
            if a in r["abs"]:
                cells.append(f"{r['abs'][a]['t_early']['mean']:.3f} / {r['abs'][a]['t_final']['mean']:.3f}")
            else:
                cells.append("--")
        lines.append(f"| {f} | " + " | ".join(cells) + " |")
    lines += ["", "## Protocol as recorded in the run files", "",
              "| task | D | n0 | rounds | lam | g_norm | g_ell | ell_rule | noise_model | seeds |", "|---|---|---|---|---|---|---|---|---|---|"]
    for f, r in tasks:
        p = r["protocol"]
        lines.append(f"| {f} | {p['D']} | {p['n0']} | {p['rounds']} | {p['lam']} | {p['g_norm']} | "
                     f"{p['g_ell']} | {p['ell_rule']} | {p['noise_model']} | {r['n_seeds']} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="bench_l2")
    ap.add_argument("--lam", type=float, default=None, help="expected lambda (default: parsed from tag bench_l<lam>)")
    ap.add_argument("--tasks", default=None, help="benchmark_tasks.json (default: <results>/benchmark_tasks.json)")
    ap.add_argument("--out", default=None, help="output stem; writes <out>.json and <out>.md")
    ap.add_argument("--baseline", default="fit_fix")
    ap.add_argument("--baseline_tag", default=None, help="take the baseline arm from this tag's files (same seeds)")
    ap.add_argument("--g_norm", default="mean", help="normalisation the run files must record")
    ap.add_argument("--load", default=None, help="reprint an existing score json, no npz reads")
    a = ap.parse_args()
    if a.load:
        out = json.load(open(a.load))
        print(report(out)); return
    outdir = os.environ.get("BO_RESULTS_DIR",
                            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"))
    results_root = os.path.dirname(outdir.rstrip("/"))
    tasks = a.tasks or os.path.join(results_root, "benchmark_tasks.json")
    lam = a.lam if a.lam is not None else float(a.tag.split("_l")[1].split("_")[0])
    out = score(outdir, a.tag, tasks, lam, a.baseline, a.g_norm, a.baseline_tag)
    stem = a.out or os.path.join(results_root, f"bench_score_{a.tag}")
    json.dump(out, open(stem + ".json", "w"), indent=1)
    md = report(out)
    open(stem + ".md", "w").write(md)
    print(md)
    print(f"saved: {stem}.json, {stem}.md")


if __name__ == "__main__":
    main()
