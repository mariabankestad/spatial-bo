"""Continuous g_uniform surrogate: one network fitted by least squares on log g
to the calibrated node fields of gunif_feasibility.py. Writes field_gu_D<D>.pt.

    python gunif_fit.py --D 3
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, "experiments"), _os.path.join(_ROOT, "training"), _os.path.join(_ROOT, "plots")):
    _p in _sys.path or _sys.path.insert(0, _p)
import argparse
import glob
import json
import math
import os
import time

import numpy as np
import torch

from gen_field_v2 import wall_points
from spatialbo.fields import field_path
from spatialbo.gp import DT
from train_field_v2 import FieldV2

NODES_DIR = os.environ.get("GUNIF_OUT", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "gunif_nodes"))
OUT_DIR = os.environ.get("BO_FIELDGU_DIR", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "field_gu"))
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_nodes(D, pattern):
    nodes = []
    for f in sorted(glob.glob(os.path.join(NODES_DIR, pattern.format(D=D)))):
        z = torch.load(f, map_location="cpu", weights_only=False)
        j = json.load(open(f[:-3] + ".json")) if os.path.exists(f[:-3] + ".json") else {}
        nodes.append({"file": f, "N": int(z["node"]["N"]), "ell": float(z["node"]["ell"]),
                      "state": z["state"], "stats": z["stats"], "pass": bool(z.get("pass", False)),
                      "stop": (j.get("hist") or [{}])[-1].get("stop", "?"),
                      "e_cert": (z.get("cert") or {}).get("e_cert")})
    return nodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--D", type=int, required=True)
    ap.add_argument("--pattern", default="gunif_D{D}_N*_ell*_info.pt")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--per_node", type=int, default=1024, help="probe points per node per step")
    ap.add_argument("--M", type=int, default=8192, help="probe points per node (3/4 uniform, 1/4 wall)")
    ap.add_argument("--min_nodes", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    D = a.D
    torch.manual_seed(a.seed)
    nodes = load_nodes(D, a.pattern)
    assert len(nodes) >= a.min_nodes, f"only {len(nodes)} node fields for D={D}"
    print(f"D={D}: {len(nodes)} node fields; certified at node: {sum(n['pass'] for n in nodes)}; "
          f"stalled: {sum(n['stop'].startswith('STALL') for n in nodes)}", flush=True)
    z0 = torch.load(field_path("geo", D), map_location="cpu", weights_only=False)["members"][0]
    stats = z0["stats"]
    # --- targets: log g of each node field on its own probe set
    gen = torch.Generator().manual_seed(a.seed + 1)
    Xs, LN, LE, T = [], [], [], []
    for nd in nodes:
        net_n = FieldV2(D, *nd["stats"]); net_n.load_state_dict(nd["state"]); net_n.eval()
        M = a.M; Mu = 3 * M // 4
        X = torch.cat([torch.rand(Mu, D, generator=gen, dtype=DT).clamp(1e-4, 1 - 1e-4),
                       wall_points(M - Mu, D, nd["ell"], gen)])
        ln = torch.full((M,), math.log(nd["N"]), dtype=DT); le = torch.full((M,), math.log(nd["ell"]), dtype=DT)
        with torch.no_grad():
            t = -net_n.delta(X, ln, le)
        Xs.append(X); LN.append(ln); LE.append(le); T.append(t)
    Xs = torch.stack(Xs).to(DEV); LN = torch.stack(LN).to(DEV); LE = torch.stack(LE).to(DEV); T = torch.stack(T).to(DEV)
    K, M = Xs.shape[0], Xs.shape[1]
    # --- continuous net, init at g_info (numerical start only)
    net = FieldV2(D, *stats); net.load_state_dict(z0["state"]); net.to(DEV); net.train()
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (1e-5 + (a.lr - 1e-5) * 0.5 * (1 + math.cos(math.pi * s / a.steps))) / a.lr)
    t0 = time.time()
    for step in range(a.steps):
        idx = torch.randint(0, M, (K, a.per_node), device=DEV)
        Xb = torch.gather(Xs, 1, idx[..., None].expand(-1, -1, D)).reshape(-1, D)
        lnb = torch.gather(LN, 1, idx).reshape(-1); leb = torch.gather(LE, 1, idx).reshape(-1)
        tb = torch.gather(T, 1, idx).reshape(-1)
        loss = ((-net.delta(Xb, lnb, leb) - tb) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if step % max(1, a.steps // 20) == 0 or step == a.steps - 1:
            print(f"  step {step:6d}  rms(log g) {float(loss.sqrt()):.5f}  [{time.time() - t0:.0f}s]", flush=True)
    net.eval()
    res = []
    with torch.no_grad():
        for k in range(K):
            r = ((-net.delta(Xs[k], LN[k], LE[k]) - T[k]) ** 2).mean().sqrt()
            res.append(float(r))
    for nd, r in zip(nodes, res):
        print(f"  node N={nd['N']:4d} ell={nd['ell']:.4f}  fit rms(log g) {r:.4f}  node-cert {'pass' if nd['pass'] else 'FAIL'}  stop {nd['stop']}", flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    meta = {"kind": "g_uniform", "D": D,
            "envelope": {"n_lo": 2 * D, "n_hi": 150, "ell_lo": 0.02 if D == 2 else 0.05, "ell_hi": 3.0},
            "nodes": [{"N": nd["N"], "ell": nd["ell"], "pass": nd["pass"], "stop": nd["stop"],
                       "e_cert": nd["e_cert"], "fit_rms": r} for nd, r in zip(nodes, res)],
            "fit": {"steps": a.steps, "lr": a.lr, "M": a.M, "init": "g_info (numerical start only)"},
            "definition": "p(argmax g*sigma_X) = Uniform(volume) under the reference ensemble; "
                          "certified by gunif_certify.py at held-out nodes"}
    net.cpu()
    torch.save({"state": net.state_dict(), "stats": stats, "meta": meta}, os.path.join(OUT_DIR, f"field_gu_D{D}.pt"))
    with open(os.path.join(OUT_DIR, f"field_gu_D{D}_fit.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"saved {os.path.join(OUT_DIR, f'field_gu_D{D}.pt')}  max node fit rms {max(res):.4f}", flush=True)


if __name__ == "__main__":
    main()
