"""Trainer of the g_eff surrogate on a stored corpus. Two heads, one per
expectation; g_eff(x) = exp(([rG(x) - rG(c)] - [rV(x) - rV(c)]) / 2), so
g_eff(c) = 1 at the center c. The released fields come from
train_field_geff_online.py.

    python train_field_geff.py --D 3
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, "experiments"), _os.path.join(_ROOT, "training"), _os.path.join(_ROOT, "plots")):
    _p in _sys.path or _sys.path.insert(0, _p)
import argparse
import glob
import os
import time

import numpy as np
import torch
import torch.nn as nn

from spatialbo.nets import FieldGeff, DT   # noqa: E402
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LAM_C = 1.0
OUTDIR = os.environ.get(
    "FIELD3_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "results", "field3_data"))


def corpus_files(D, exclude=None):
    files = sorted(glob.glob(os.path.join(OUTDIR, f"corpus_geff_D{D}_s*.npz")))
    files = [f for f in files if "_sup" not in os.path.basename(f)]
    files += sorted(glob.glob(os.path.join(OUTDIR, f"corpus_geff_D{D}_sup*.npz")))
    if exclude is not None:      # held-out validation shard (plateau report)
        files = [f for f in files if not f.endswith(f"_s{exclude}.npz")]
    return files


def load_corpus(D, exclude=None):
    files = corpus_files(D, exclude)
    assert files, f"no g_eff corpus for D={D} in {OUTDIR}"
    k = {n: [] for n in ("X", "yV", "yG", "yV_c", "yG_c", "ln", "le")}
    for f in files:
        z = np.load(f, allow_pickle=True)
        for n in k:
            k[n].append(z[n])
    return {n: torch.tensor(np.concatenate(v), dtype=DT) for n, v in k.items()}


def gold_error(net, D):
    """MAE of g_eff against the gold nodes, or None if absent; reported during training only."""
    path = os.path.join(OUTDIR, f"gold_geff_D{D}.npz")
    if not os.path.exists(path):
        return None
    gold = np.load(path, allow_pickle=True)["gold"]
    errs = []
    for node in gold:
        gt = np.sqrt(node["G"] / np.maximum(node["V"], 1e-12))
        gt = gt / np.sqrt(node["G_c"] / max(node["V_c"], 1e-12))
        X = torch.tensor(node["X"], dtype=DT).to(DEV)
        ln = torch.full((len(X),), np.log(float(node["n"])), dtype=DT).to(DEV)
        le = torch.full((len(X),), np.log(float(node["ell"])), dtype=DT).to(DEV)
        with torch.no_grad():
            errs.append(np.abs(net.g(X, ln, le).cpu().numpy() - gt))
    return float(np.concatenate(errs).mean())


def validation_error(net, D, shard):
    """Raw relative MSE of both heads on the held-out shard; reported
    for the plateau judgement, never used to select a checkpoint."""
    z = np.load(os.path.join(OUTDIR, f"corpus_geff_D{D}_s{shard}.npz"), allow_pickle=True)
    Xv = torch.tensor(z["X"][:2000], dtype=DT); Rv, Pv, _ = Xv.shape
    lnv = torch.tensor(z["ln"][:2000], dtype=DT).repeat_interleave(Pv).to(DEV)
    lev = torch.tensor(z["le"][:2000], dtype=DT).repeat_interleave(Pv).to(DEV)
    out = {}
    with torch.no_grad():
        for which, key in (("V", "yV"), ("G", "yG")):
            yh = net.pred(Xv.reshape(-1, D).to(DEV), lnv, lev, which).cpu()
            yr = torch.tensor(z[key][:2000], dtype=DT).reshape(-1)
            out[which] = float((((yh - yr) / yr.abs().clamp_min(1e-12)) ** 2).mean())
    return out


def valgold_mae(net, D):
    """MAE of g_eff against valgold_geff_D<D>.npz (10 held-out nodes,
    S=256, independent seeds); the plateau rule reads this curve."""
    path = os.path.join(OUTDIR, f"valgold_geff_D{D}.npz")
    if not os.path.exists(path):
        return None
    errs = []
    for node in np.load(path, allow_pickle=True)["nodes"]:
        X = torch.tensor(node["X"], dtype=DT).to(DEV)
        ln = torch.full((len(X),), np.log(float(node["n"])), dtype=DT, device=DEV)
        le = torch.full((len(X),), np.log(float(node["ell"])), dtype=DT, device=DEV)
        with torch.no_grad():
            errs.append(np.abs(net.g(X, ln, le).cpu().numpy() - node["g"]))
    return float(np.concatenate(errs).mean())


def train(D, steps, batch=256, seed=0,
          floors=(0.0, 1.0, 0.3, 0.1, 0.03, 0.01), val_shard=None,
          init=None, lr=1e-3):
    """init: warm start from an artifact (stats reused, final floor
    stage, lower lr) - see train_field_v2.train for the rationale."""
    from train_field_v2 import plateau_verdict
    C = load_corpus(D, exclude=val_shard)
    R, P, _ = C["X"].shape
    torch.manual_seed(seed)
    if init is not None:
        z = torch.load(init, map_location="cpu", weights_only=False)
        net = FieldGeff(D, *z["stats"]).to(DEV); net.load_state_dict(z["state"])
        print(f"  warm start from {init} (stats reused)", flush=True)
    else:
        net = FieldGeff(D, float(C["ln"].mean()), float(C["ln"].std()),
                        float(C["le"].mean()), float(C["le"].std())).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    # cosine decay to 1e-5 over the run
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps,
                                                     eta_min=1e-5)
    curve = []
    C = {n: v.to(DEV) for n, v in C.items()}
    g = torch.Generator().manual_seed(seed)
    n_stage = max(1, steps // len(floors))
    for step in range(steps):
        floor = floors[min(step // n_stage, len(floors) - 1)]
        idx = torch.randint(R, (batch,), generator=g).to(DEV)
        Xb = C["X"][idx].reshape(batch * P, D)
        lnb = C["ln"][idx].repeat_interleave(P)
        leb = C["le"][idx].repeat_interleave(P)
        loss = 0.0
        for which, ykey, ckey in (("V", "yV", "yV_c"), ("G", "yG", "yG_c")):
            yh = net.pred(Xb, lnb, leb, which)
            yc = net.pred_c(C["ln"][idx], C["le"][idx], which)
            yt = C[ykey][idx].reshape(-1); ytc = C[ckey][idx]
            # per-row scale from the row's own detached center prediction
            row = yc.detach().abs().clamp_min(1e-12)
            sc = row.repeat_interleave(P)
            if floor > 0:
                # floor relative to the row scale, annealed
                sc = torch.maximum(sc * floor, yh.detach().abs())
                scc = torch.maximum(row * floor, yc.detach().abs())
            else:
                scc = row
            loss = loss + (((yh - yt) / sc) ** 2).mean() \
                + LAM_C * (((yc - ytc) / scc) ** 2).mean()
        opt.zero_grad(); loss.backward()
        # clip: the per-row relative loss has heavy tails in high D
        # (observed loss 13-16 at D=6 vs 1-2 at D=2)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
        opt.step(); sch.step()
        if (step + 1) % max(1, steps // 40) == 0:
            # gold is printed for information only, never used to select a checkpoint
            ge = gold_error(net, D)
            ve = validation_error(net, D, val_shard) if val_shard is not None else None
            vg = valgold_mae(net, D); curve.append(vg)
            print(f"  step {step+1}/{steps} floor={floor} "
                  f"lr={sch.get_last_lr()[0]:.2e} "
                  f"loss={float(loss.detach()):.5f}"
                  + (f"  val_relMSE V {ve['V']:.4f} G {ve['G']:.4f}" if ve else "")
                  + (f"  VALGOLD MAE {vg:.4f}" if vg is not None else "  (no valgold)")
                  + (f"  GOLD(info only) {ge:.4f}" if ge is not None else ""),
                  flush=True)
    ok, why = plateau_verdict(curve)
    print(f"TRAINING PLATEAU: {'PASS' if ok else 'FAIL' if ok is not None else 'N/A'} - {why}",
          flush=True)
    net.plateau = (ok, why, curve)
    return net


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--D", type=int, required=True)
    ap.add_argument("--steps", type=int,
                    default=int(os.environ.get("STEPS", 160000)))
    ap.add_argument("--seed", type=int, default=0,
                    help="replicate seed; >0 writes field_geff_D<D>_seed<k>.pt")
    ap.add_argument("--val_shard", type=int, default=None)
    ap.add_argument("--init", default=None, help="warm-start artifact (.pt)")
    ap.add_argument("--lr", type=float, default=None)
    a = ap.parse_args()
    floors = (0.01,) if a.init else (0.0, 1.0, 0.3, 0.1, 0.03, 0.01)
    lr = a.lr if a.lr is not None else (3e-4 if a.init else 1e-3)
    net = train(a.D, a.steps, seed=a.seed, val_shard=a.val_shard, floors=floors,
                init=a.init, lr=lr)
    default_name = (f"field_geff_D{a.D}.pt" if a.seed == 0
                    else f"field_geff_D{a.D}_seed{a.seed}.pt")
    out = os.path.join(OUTDIR, os.environ.get("F3_OUT", default_name))
    from train_field_v2 import corpus_envelope
    files = corpus_files(a.D)
    torch.save({"state": net.state_dict(), "D": a.D, "stats": net.stats,
                "meta": {"arch": "two-head centred-log ratio",
                         "eps": 1e-4, "envelope": corpus_envelope(a.D, files),
                         "seed": a.seed, "val_shard": a.val_shard, "steps": a.steps,
                         "init": a.init, "lr": lr,
                         "shards": [os.path.basename(f) for f in files],
                         "plateau": {"pass": net.plateau[0], "why": net.plateau[1],
                                     "curve": net.plateau[2]},
                         "spec": "g_eff field "}}, out)
    print(f"saved {out}")
