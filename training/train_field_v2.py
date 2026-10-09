"""Trainer of the g_geo surrogate on a stored corpus, and pieces shared with the
online trainer. Model: log m(x) = b(n, ell) + r(x) - r(c), g(x) = exp(-(r(x) - r(c))).

    python train_field_v2.py --D 2
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, "experiments"), _os.path.join(_ROOT, "training"), _os.path.join(_ROOT, "plots")):
    _p in _sys.path or _sys.path.insert(0, _p)
import argparse
import glob
import math
import os

import numpy as np
import torch
import torch.nn as nn

from spatialbo.nets import FieldV2, _mlp, DT   # noqa: E402
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LAM_C = 1.0
OUTDIR = os.environ.get(
    "FIELD2_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "results", "field2_data"))


def corpus_envelope(D, files=None, pattern="corpus_v2_D{D}_s*.npz"):
    """The (n, ell) range of the corpus from its shard metadata; written into the artifact."""
    files = files or sorted(glob.glob(os.path.join(OUTDIR, pattern.format(D=D))))
    envs = []
    for f in files:
        m = np.load(f, allow_pickle=True)["meta"].item()
        if "envelope" in m:
            envs.append(m["envelope"])
        else:                       # pre-shard: rebuild from fields
            envs.append({"n_lo": 2 * D + 2, "n_hi": int(m["n_hi"]),
                         "ell_lo": float(m["ell"][0]),
                         "ell_hi": float(m["ell"][1])})
    assert all(e == envs[0] for e in envs), f"shards disagree on envelope: {envs}"
    return envs[0]


def corpus_files(D, exclude=None):
    """Bulk shards (_s*) plus supplement shards (_sup*, sparse-region
    rows, ); exclude = bulk shard held out as validation."""
    files = sorted(glob.glob(os.path.join(OUTDIR, f"corpus_v2_D{D}_s*.npz")))
    files = [f for f in files if "_sup" not in os.path.basename(f)]
    files += sorted(glob.glob(os.path.join(OUTDIR, f"corpus_v2_D{D}_sup*.npz")))
    if exclude is not None:
        files = [f for f in files if not f.endswith(f"_s{exclude}.npz")]
    return files


def load_corpus(D, exclude=None):
    """exclude = shard index held out as a validation set
    (shards are iid draws of the same corpus distribution)."""
    files = corpus_files(D, exclude)
    assert files, f"no corpus for D={D} in {OUTDIR}"
    X, y, yc, ln, le = [], [], [], [], []
    for f in files:
        z = np.load(f, allow_pickle=True)
        X.append(z["X"]); y.append(z["y"]); yc.append(z["y_c"])
        ln.append(z["ln"]); le.append(z["le"])
    return (torch.tensor(np.concatenate(X), dtype=DT),
            torch.tensor(np.concatenate(y), dtype=DT),
            torch.tensor(np.concatenate(yc), dtype=DT),
            torch.tensor(np.concatenate(ln), dtype=DT),
            torch.tensor(np.concatenate(le), dtype=DT))


def validation_error(net, D, shard):
    """Relative MSE against the held-out shard's labels; dominated by label
    variance, so the plateau rule uses valgold_mae instead."""
    z = np.load(os.path.join(OUTDIR, f"corpus_v2_D{D}_s{shard}.npz"), allow_pickle=True)
    Xv = torch.tensor(z["X"][:2000], dtype=DT); yv = torch.tensor(z["y"][:2000], dtype=DT)
    lnv = torch.tensor(z["ln"][:2000], dtype=DT); lev = torch.tensor(z["le"][:2000], dtype=DT)
    Rv, Pv, _ = Xv.shape
    with torch.no_grad():
        mh = net.m(Xv.reshape(-1, D).to(DEV), lnv.repeat_interleave(Pv).to(DEV),
                   lev.repeat_interleave(Pv).to(DEV)).cpu()
        yr = yv.reshape(-1)
        return float((((mh - yr) / yr.clamp_min(1e-3)) ** 2).mean())


def valgold_mae(net, D):
    """MAE of g against the stored validation nodes, or None if absent."""
    path = os.path.join(OUTDIR, f"valgold_v2_D{D}.npz")
    if not os.path.exists(path):
        return None
    errs = []
    for node in np.load(path, allow_pickle=True)["nodes"]:
        X = torch.tensor(node["X"], dtype=DT).to(DEV)
        ln = torch.full((len(X),), math.log(float(node["n"])), dtype=DT, device=DEV)
        le = torch.full((len(X),), math.log(float(node["ell"])), dtype=DT, device=DEV)
        with torch.no_grad():
            errs.append(np.abs(net.g(X, ln, le).cpu().numpy() - node["g"]))
    return float(np.concatenate(errs).mean())


def plateau_verdict(curve, frac=0.2, tol=0.10):
    """Plateau rule: the last `frac` of validation values lie within `tol` of the
    minimum and do not trend upward."""
    c = np.asarray([v for v in curve if v is not None], float)
    if len(c) < 5:
        return None, "too few evaluations"
    k = max(2, int(round(frac * len(c)))); tail = c[-k:]; mn = c.min()
    within = bool((tail <= (1 + tol) * mn).all())
    half = k // 2
    no_rise = bool(tail[-half:].mean() <= tail[:half].mean() * (1 + 0.02))
    return bool(within and no_rise), (f"tail max/min {tail.max()/mn:.3f} (tol {1+tol:.2f}), "
                                      f"tail trend {'flat/down' if no_rise else 'UP'}")


def train(D, steps, emb=32, batch=256, seed=0, floors=(0.0,), val_shard=None,
          init=None, lr=1e-3):
    """Train one network on the corpus shards. init: warm start from an artifact
    (its normalization statistics are reused). Environment flags F2_PIN,
    F2_WD and F2_DECAY add a center pin, weight decay and cosine decay."""
    X, y, yc, ln, le = load_corpus(D, exclude=val_shard)
    R, P, _ = X.shape
    torch.manual_seed(seed)
    if init is not None:
        z = torch.load(init, map_location="cpu", weights_only=False)
        net = FieldV2(D, *z["stats"], emb=emb).to(DEV)
        net.load_state_dict(z["state"])
        print(f"  warm start from {init} (stats reused)", flush=True)
    else:
        net = FieldV2(D, float(ln.mean()), float(ln.std()),
                      float(le.mean()), float(le.std()), emb=emb).to(DEV)
    pin = float(os.environ.get("F2_PIN", 0.0))
    wd = float(os.environ.get("F2_WD", 0.0))
    decay = os.environ.get("F2_DECAY", "0") == "1" or init is not None
    opt = torch.optim.Adam(net.parameters(),
                           lr=(2e-3 if decay and init is None else lr), weight_decay=wd)
    sch = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps,
                                                      eta_min=1e-5 if init is not None else 5e-5)
           if decay else None)
    curve = []
    X, y, yc, ln, le = (t.to(DEV) for t in (X, y, yc, ln, le))
    g = torch.Generator().manual_seed(seed)
    n_stage = max(1, steps // len(floors))
    for step in range(steps):
        floor = floors[min(step // n_stage, len(floors) - 1)]
        idx = torch.randint(R, (batch,), generator=g).to(DEV)
        Xb = X[idx].reshape(batch * P, D)
        lnb = ln[idx].repeat_interleave(P); leb = le[idx].repeat_interleave(P)
        mhat = net.m(Xb, lnb, leb)
        mcb = net.m_c(ln[idx], le[idx])
        if floor > 0:
            # scale = detached prediction, never the label (label-dependent weights bias the minimizer)
            w = 1.0 / mhat.detach().clamp_min(floor) ** 2
            wc = 1.0 / mcb.detach().clamp_min(floor) ** 2
            loss = (w * (mhat - y[idx].reshape(-1)) ** 2).mean() \
                 + LAM_C * (wc * (mcb - yc[idx]) ** 2).mean()
        else:
            loss = ((mhat - y[idx].reshape(-1)) ** 2).mean() \
                 + LAM_C * ((mcb - yc[idx]) ** 2).mean()
        if pin > 0:
            xc = torch.full((batch, net.D), 0.5, dtype=DT, device=DEV)
            loss = loss + pin * (net._r(xc, ln[idx], le[idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if sch is not None:
            sch.step()
        if (step + 1) % max(1, steps // 20) == 0:
            ve = (validation_error(net, D, val_shard) if val_shard is not None
                  else float("nan"))
            vg = valgold_mae(net, D); curve.append(vg)
            print(f"  step {step+1}/{steps} floor={floor} "
                  f"loss={float(loss.detach()):.5f}  val_relMSE={ve:.5f}"
                  + (f"  VALGOLD MAE {vg:.4f}" if vg is not None else "  (no valgold)"),
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
                    default=int(os.environ.get("STEPS", 20000)))
    ap.add_argument("--seed", type=int, default=0,
                    help="replicate seed; >0 writes field_v2_D<D>_seed<k>.pt")
    ap.add_argument("--val_shard", type=int, default=None,
                    help="shard held out for the plateau report (not trained on)")
    ap.add_argument("--init", default=None, help="warm-start artifact (.pt)")
    ap.add_argument("--lr", type=float, default=None,
                    help="peak lr (default 1e-3 from scratch, 3e-4 warm)")
    a = ap.parse_args()
    # deeper final stage: the dense-long (small-m) regime
    # held the p99 tails at floor 0.01; 0.005/0.003 sharpen exactly there
    floors = (0.003,) if a.init else (0.0, 0.1, 0.03, 0.01, 0.005, 0.003)
    lr = a.lr if a.lr is not None else (3e-4 if a.init else 1e-3)
    net = train(a.D, a.steps, floors=floors, seed=a.seed, val_shard=a.val_shard,
                init=a.init, lr=lr)
    default_name = (f"field_v2_D{a.D}.pt" if a.seed == 0
                    else f"field_v2_D{a.D}_seed{a.seed}.pt")
    out = os.path.join(OUTDIR, os.environ.get("F2_OUT", default_name))
    files = corpus_files(a.D)
    torch.save({"state": net.state_dict(), "D": a.D, "stats": net.stats,
                "meta": {"arch": "centred-log-relative", "eps": 1e-4,
                         "QN": "random-shifted scrambled Sobol",
                         "envelope": corpus_envelope(a.D, files),
                         "seed": a.seed, "val_shard": a.val_shard,
                         "steps": a.steps, "init": a.init, "lr": lr,
                         "shards": [os.path.basename(f) for f in files],
                         "plateau": {"pass": net.plateau[0], "why": net.plateau[1],
                                     "curve": net.plateau[2]},
                         "spec": "field-v2 final "}}, out)
    print(f"saved {out}")
