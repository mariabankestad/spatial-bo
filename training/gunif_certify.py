"""Certification of the g_uniform surrogate at held-out (n, ell) nodes: the
argmax of g * sigma under the deployed optimizer is tested for uniformity in
volume (energy distance, radial and coordinate KS, bin ratios, symmetry,
smoothness). Writes cert_gu_D<D>.json beside the field file.

    python gunif_certify.py --D 3
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, "experiments"), _os.path.join(_ROOT, "training"), _os.path.join(_ROOT, "plots")):
    _p in _sys.path or _sys.path.insert(0, _p)
import argparse
import json
import math
import os
import time

import numpy as np
import torch

import gunif_feasibility as GF
from spatialbo.gp import DT
from train_field_v2 import FieldV2
from uniformity import coord_report, energy_distance, radial_ks

GU_DIR = os.environ.get("BO_FIELDGU_DIR", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "field_gu"))


def heldout_nodes(D, k=4):
    n_lo, n_hi = 2 * D, 150; e_lo, e_hi = (0.02 if D == 2 else 0.05), 3.0
    out = []
    for i in range(k):
        for j in range(k):
            fn, fe = (i + 0.5) / k, (j + 0.5) / k
            out.append((int(round(n_lo * (n_hi / n_lo) ** fn)), e_lo * (e_hi / e_lo) ** fe))
    return out


def structural(net, D, nodes, gen):
    """Exact symmetry and ell-scaled smoothness of the continuous field."""
    sym, smooth = 0.0, 0.0
    with torch.no_grad():
        for N, ell in nodes:
            X = torch.rand(2000, D, generator=gen, dtype=DT).clamp(1e-4, 1 - 1e-4)
            ln = torch.full((2000,), math.log(N), dtype=DT); le = torch.full((2000,), math.log(ell), dtype=DT)
            g = net.g(X, ln, le)
            perm = torch.randperm(D, generator=gen); flip = torch.randint(0, 2, (D,), generator=gen).to(DT)
            Xs = torch.where(flip > 0, 1.0 - X[:, perm], X[:, perm])
            sym = max(sym, float((net.g(Xs, ln, le) - g).abs().max()))
            step = max(ell / 50.0, 1.0 / 2000.0)
            d = torch.randn(2000, D, generator=gen, dtype=DT); d = d / d.norm(dim=1, keepdim=True)
            X2 = (X + step * d).clamp(1e-4, 1 - 1e-4)
            smooth = max(smooth, float(((net.g(X2, ln, le) - g).abs() / g).max()))
    return sym, smooth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--D", type=int, required=True)
    ap.add_argument("--R", type=int, default=3000, help="fresh designs per held-out node")
    ap.add_argument("--k", type=int, default=4, help="held-out grid k x k (log-midpoints)")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    a = ap.parse_args()
    D = a.D
    z = torch.load(os.path.join(GU_DIR, f"field_gu_D{D}.pt"), map_location="cpu", weights_only=False)
    net = FieldV2(D, *z["stats"]); net.load_state_dict(z["state"]); net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    state = {k: v.clone() for k, v in net.state_dict().items()}
    nodes = heldout_nodes(D, a.k)
    gen = torch.Generator().manual_seed(7)
    sym, smooth = structural(net, D, nodes, gen)
    print(f"D={D}: symmetry max |g(x) - g(sym x)| = {sym:.2e} ({'pass' if sym <= 1e-10 else 'FAIL'}); "
          f"smoothness max |dg|/g over ell/50 = {smooth:.4f} ({'pass' if smooth <= 0.10 else 'FAIL'})", flush=True)
    bins = {}
    rng = np.random.default_rng(2026)
    rows = []
    for i, (N, ell) in enumerate(nodes):
        t0 = time.time()
        seeds = 90_000_000 + 100_000 * D + 10_000 * i + np.arange(a.R)
        (X, fb), = GF.run_ensembles(a.workers, state, D, z["stats"], math.log(N), math.log(ell), N, ell, (seeds,))
        cf = np.array([energy_distance(rng.random((a.R, D)), rng.random((a.R, D)), n_perm=0)[0] for _ in range(10)])
        e = float(energy_distance(X, rng.random((a.R, D)), n_perm=0)[0])
        if D not in bins:
            bins[D] = GF.FaceBins(D)
        b = bins[D]; u = b.u.numpy()
        nulls = {"radial_ks": [], "coord_ks": [], "bin_ratio_max": []}
        for _ in range(20):
            Un = rng.random((a.R, D))
            nulls["radial_ks"].append(radial_ks(Un, D)); nulls["coord_ks"].append(max(k for _, k in coord_report(Un)))
            cn = np.bincount(b.index(torch.as_tensor(Un)).numpy(), minlength=b.K) / a.R
            nulls["bin_ratio_max"].append(float((cn / u).max()))
        q95 = {k: float(np.quantile(v, 0.95)) for k, v in nulls.items()}
        cnt = np.bincount(b.index(torch.as_tensor(X)).numpy(), minlength=b.K) / a.R
        diag = {"radial_ks": radial_ks(X, D), "coord_ks": max(k for _, k in coord_report(X)),
                "bin_ratio_max": float((cnt / u).max()), "bin_ratio_min": float((cnt / u).min())}
        flags = {k: bool(diag[k] > q95[k]) for k in q95}
        ok_p = e <= float(cf.mean() + 2 * cf.std()); ok = ok_p and not any(flags.values())
        rows.append({"N": N, "ell": ell, "e_cert": e, "floor_mu": float(cf.mean()), "floor_sd": float(cf.std()),
                     "pass_primary": ok_p, "pass": ok, "diag": diag, "null_q95": q95, "flags": flags,
                     "fallback_frac": fb, "bin_profile": (cnt / u).tolist()})
        print(f"  held-out N={N:4d} ell={ell:.4f}: E {e:.5f} vs floor {cf.mean():.5f}+2*{cf.std():.5f} "
              f"{'pass' if ok_p else 'FAIL'}; flags {[k for k, v in flags.items() if v] or 'none'} "
              f"(radKS {diag['radial_ks']:.3f}/{q95['radial_ks']:.3f} binmax {diag['bin_ratio_max']:.2f}/{q95['bin_ratio_max']:.2f})"
              f"  -> {'PASS' if ok else 'FAIL'}  [{time.time() - t0:.0f}s]", flush=True)
    n_pass = sum(r["pass"] for r in rows)
    overall = n_pass == len(rows) and sym <= 1e-10 and smooth <= 0.10
    out = {"D": D, "R": a.R, "symmetry_max": sym, "smooth_max_rel_step": smooth,
           "nodes": rows, "n_pass": n_pass, "n_nodes": len(rows), "pass": overall,
           "field_meta": z.get("meta", {})}
    with open(os.path.join(GU_DIR, f"cert_gu_D{D}.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"CERTIFICATE g_uniform D={D}: {n_pass}/{len(rows)} held-out nodes pass; "
          f"symmetry {'pass' if sym <= 1e-10 else 'FAIL'}; smoothness {'pass' if smooth <= 0.10 else 'FAIL'} "
          f"-> {'PASS' if overall else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
