"""Trainer of the g_eff surrogate on fresh realizations (used for the released
fields): the model of train_field_geff.py with labels drawn from new reference
designs at every step. Checkpoints allow resuming.

    python train_field_geff_online.py --D 3 --seed 0 --steps 300000 --K 32 --KV 64 --schedule final --sampler final
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, "experiments"), _os.path.join(_ROOT, "training"), _os.path.join(_ROOT, "plots")):
    _p in _sys.path or _sys.path.insert(0, _p)
import argparse
import math
import os
import time

import numpy as np
import torch

from spatialbo.gp import matern52
from gen_field_v2 import EPS, wall_points
from gen_field_corpus import ELL_LO, ELL_HI, sample_points
from train_field_online import N_LO_MULT, N_HI      # the same envelope as g_info
from gen_field_geff import mixture_nodes, uniform_is_enough
from train_field_geff import FieldGeff, DT, DEV, OUTDIR, LAM_C, valgold_mae
from train_field_v2 import plateau_verdict
from train_field_online import OnlineSampler

# defaults: full quadrature; the G head gets K designs, the V head K_V >= K
NU = int(os.environ.get("NU", 1024))
N_LOC = int(os.environ.get("N_LOC", 32))
_GRID = {}


def shifted_grid(D, seed):
    """Randomly shifted scrambled-Sobol grid of NU nodes (the online
    trainer's own NU, independent of gen_field_geff's module default)."""
    if D not in _GRID:
        _GRID[D] = torch.quasirandom.SobolEngine(D, scramble=True, seed=4242).draw(NU).to(DT)
    g = torch.Generator().manual_seed(int(seed))
    return (_GRID[D] + torch.rand(1, D, generator=g, dtype=DT)) % 1.0


class GeffSampler(OnlineSampler):
    """K designs per row for the G head; K_V >= K for the V head, the first K shared."""
    def __init__(self, D, seed, B=32, K=16, K_V=32, P=100, version="v3"):
        super().__init__(D, seed, B=B, K=K_V, P=P, version=version)
        self.K_G = K

    def labels_group(self, Xd, Q, U, q, mask, e3, n):
        """Batched (R, ...) labels for one regime group. U: (R, NUmax, D),
        q: (R, NUmax) density (1 on pads), mask: (R, NUmax)."""
        Km = matern52(Xd, Xd, e3) + EPS * torch.eye(n, dtype=DT, device=DEV)
        L = torch.linalg.cholesky(Km)
        zP = torch.linalg.solve_triangular(L, matern52(Xd, Q, e3), upper=False)   # (R,n,P+1)
        zU = torch.linalg.solve_triangular(L, matern52(Xd, U, e3), upper=False)   # (R,n,NU)
        vP = (1.0 - (zP * zP).sum(1)).clamp_min(1e-12)                            # (R,P+1)
        cpost = matern52(U, Q, e3) - zU.transpose(1, 2) @ zP                       # (R,NU,P+1)
        integ = (cpost ** 2) / q[:, :, None] * mask[:, :, None]
        yG = integ.sum(1) / mask.sum(1, keepdim=True) / (vP + EPS)
        return vP, yG

    def v_labels(self, Xd, Q, e3, n):
        """V head only: (R, P+1) posterior variances - no quadrature nodes."""
        Km = matern52(Xd, Xd, e3) + EPS * torch.eye(n, dtype=DT, device=DEV)
        L = torch.linalg.cholesky(Km)
        zP = torch.linalg.solve_triangular(L, matern52(Xd, Q, e3), upper=False)
        return (1.0 - (zP * zP).sum(1)).clamp_min(1e-12)

    def batch(self):
        B, KV, K, D = self.B, self.K, self.K_G, self.D
        n = self.draw_n(); ell = self.draw_ells(n)
        U = self.draw_points(ell)                                            # (B,P,D)
        xc = torch.full((B, 1, D), 0.5, dtype=DT)
        Q = torch.cat([U, xc], 1)                                            # (B,P+1,D)
        Xd_all = self.draw_designs(n)                                        # (B*KV,n,D)
        # V from ALL K_V designs (cheap); G from the first K of each row
        e_all = ell.repeat_interleave(KV).to(DEV).view(-1, 1, 1)
        yV = self.v_labels(Xd_all.to(DEV), Q.repeat_interleave(KV, 0).to(DEV), e_all, n
                           ).reshape(B, KV, -1).mean(1)
        Xd = Xd_all.reshape(B, KV, n, D)[:, :K].reshape(B * K, n, D)          # (B*K,n,D)
        is_rows = [i for i in range(B) if not uniform_is_enough(float(ell[i]), D, NU)]
        gr_rows = [i for i in range(B) if i not in set(is_rows)]
        yG = torch.zeros(B, Q.shape[1], dtype=DT, device=DEV)
        for rows, is_group in ((is_rows, True), (gr_rows, False)):
            if not rows:
                continue
            nodes, dens = [], []
            for i in rows:
                if is_group:
                    Ui, qi = mixture_nodes(Q[i], float(ell[i]), D, seed=self.counter,
                                           n_loc=N_LOC, n_uni=NU)
                else:
                    Ui, qi = shifted_grid(D, self.counter), torch.ones(NU, dtype=DT)
                self.counter += 1
                nodes.append(Ui); dens.append(qi)
            m = max(len(u) for u in nodes)
            Ug = torch.zeros(len(rows), m, D, dtype=DT); qg = torch.ones(len(rows), m, dtype=DT)
            mask = torch.zeros(len(rows), m, dtype=DT)
            for j, (u, qq) in enumerate(zip(nodes, dens)):
                Ug[j, :len(u)] = u; qg[j, :len(u)] = qq; mask[j, :len(u)] = 1.0
            idx = torch.tensor(rows)
            sel = (idx[:, None] * K + torch.arange(K)[None, :]).reshape(-1)
            R = len(rows) * K
            vP, yg = self.labels_group(
                Xd[sel].to(DEV), Q[idx].repeat_interleave(K, 0).to(DEV),
                Ug.repeat_interleave(K, 0).to(DEV), qg.repeat_interleave(K, 0).to(DEV),
                mask.repeat_interleave(K, 0).to(DEV),
                ell[idx].repeat_interleave(K).to(DEV).view(R, 1, 1), n)
            yG[idx] = yg.reshape(len(rows), K, -1).mean(1)
        ln = torch.full((B,), math.log(n), dtype=DT, device=DEV)
        le = torch.log(ell).to(DEV)
        return U.to(DEV), ln, le, yV[:, :-1], yV[:, -1], yG[:, :-1], yG[:, -1]


def train(D, steps, seed, B=32, K=16, K_V=32, floors=(0.0, 1.0, 0.3, 0.1, 0.03, 0.01), lr=1e-3,
          schedule="v3", sampler="v3", ckpt=None):
    """schedule: 'legacy' or 'v3' (schedule.py); v3 returns the Polyak
    average over the last 20% and stores iterate drift in net.drift."""
    import copy
    from schedule import lr_factor, floor_at, PolyakAverage, field_drift, drift_probes
    torch.manual_seed(seed)
    n_lo = N_LO_MULT * D
    ln_mu = 0.5 * (math.log(n_lo) + math.log(N_HI)); ln_sd = (math.log(N_HI) - math.log(n_lo)) / math.sqrt(12)
    le_mu = 0.5 * (math.log(ELL_LO) + math.log(ELL_HI)); le_sd = (math.log(ELL_HI) - math.log(ELL_LO)) / math.sqrt(12)
    net = FieldGeff(D, ln_mu, ln_sd, le_mu, le_sd).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, steps, schedule, lr))
    avg = (PolyakAverage(net, steps, start_frac=0.6 if schedule == "final" else 0.8)
           if schedule in ("v3", "final") else None)
    probes = drift_probes(D, n_lo, N_HI, ELL_LO, ELL_HI, DEV); snap = None
    sampler = GeffSampler(D, seed, B=B, K=K, K_V=K_V, version=sampler)
    curve = []; t0 = time.time(); start = 0
    if ckpt and os.path.exists(ckpt):
        z = torch.load(ckpt, map_location=DEV, weights_only=False)
        net.load_state_dict(z["net"]); opt.load_state_dict(z["opt"])
        if avg is not None: avg.load_state_dict(z["avg"])
        sampler.load_state_dict(z["sampler"]); curve = z["curve"]; start = z["step"]
        snap = z.get("snap")
        for _ in range(start): sch.step()
        print(f"  RESUMED from {ckpt} at step {start}", flush=True)
    for step in range(start, steps):
        floor = floor_at(step, steps, floors, schedule)
        U, ln, le, yV, yVc, yG, yGc = sampler.batch()
        P = U.shape[1]
        Xb = U.reshape(-1, D); lnb = ln.repeat_interleave(P); leb = le.repeat_interleave(P)
        loss = 0.0
        for which, yt, ytc in (("V", yV, yVc), ("G", yG, yGc)):
            yh = net.pred(Xb, lnb, leb, which); yc = net.pred_c(ln, le, which)
            row = yc.detach().abs().clamp_min(1e-12); sc = row.repeat_interleave(P)
            if floor > 0:
                sc = torch.maximum(sc * floor, yh.detach().abs()); scc = torch.maximum(row * floor, yc.detach().abs())
            else:
                scc = row
            loss = loss + (((yh - yt.reshape(-1)) / sc) ** 2).mean() + LAM_C * (((yc - ytc) / scc) ** 2).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
        opt.step(); sch.step()
        if avg is not None:
            avg.update(net, step)
        if step == int(0.98 * steps) - 1:
            snap = copy.deepcopy(net)
        if (step + 1) % max(1, steps // 50) == 0:
            vg = valgold_mae(net, D); curve.append(vg)
            print(f"  step {step+1}/{steps} floor={floor} lr={sch.get_last_lr()[0]:.2e} "
                  f"loss={float(loss.detach()):.5f}"
                  + (f"  VALGOLD MAE {vg:.4f}" if vg is not None else "  (no valgold)")
                  + f"  ({(time.time()-t0)/60:.0f} min)", flush=True)
            if ckpt:
                torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "step": step + 1,
                            "avg": avg.state_dict() if avg is not None else None,
                            "sampler": sampler.state_dict(), "curve": curve, "snap": snap},
                           ckpt + ".tmp")
                os.replace(ckpt + ".tmp", ckpt)
    ok, why = plateau_verdict(curve)
    print(f"TRAINING PLATEAU: {'PASS' if ok else 'FAIL' if ok is not None else 'N/A'} - {why}", flush=True)
    drift = {}
    if snap is not None:
        drift["last_vs_2pct_earlier"] = field_drift(net, snap, D, probes)
    out_net = net
    if avg is not None:
        out_net = avg.apply(net)
        drift["last_vs_average"] = field_drift(net, out_net, D, probes)
        vga = valgold_mae(out_net, D)
        print(f"  Polyak average over the last 20%: VALGOLD MAE {vga if vga is None else round(vga, 4)}", flush=True)
    print(f"ITERATE DRIFT (max, mean |dg| on fixed probes): {drift}", flush=True)
    out_net.plateau = (ok, why, curve); out_net.drift = drift; out_net.schedule = schedule
    return out_net


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--D", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=int(os.environ.get("STEPS", 200000)))
    ap.add_argument("--B", type=int, default=32)
    ap.add_argument("--K", type=int, default=16, help="designs per row for the G head")
    ap.add_argument("--KV", type=int, default=32, help="designs per row for the V head (>= K)")
    ap.add_argument("--schedule", choices=("legacy", "v3", "final"), default="v3")
    ap.add_argument("--sampler", choices=("v3", "v5", "v6", "final"), default="v3")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    name = f"field_geff_D{a.D}.pt" if a.seed == 0 else f"field_geff_D{a.D}_seed{a.seed}.pt"
    out = os.path.join(OUTDIR, os.environ.get("F3_OUT", name))
    os.makedirs(OUTDIR, exist_ok=True)
    ckpt = out + ".ckpt" if a.resume else None
    net = train(a.D, a.steps, a.seed, B=a.B, K=a.K, K_V=a.KV, schedule=a.schedule,
                sampler=a.sampler, ckpt=ckpt)
    torch.save({"state": net.state_dict(), "D": a.D, "stats": net.stats,
                "meta": {"arch": "two-head centred-log ratio", "eps": EPS,
                         "envelope": {"n_lo": N_LO_MULT * a.D, "n_hi": N_HI,
                                      "ell_lo": ELL_LO, "ell_hi": ELL_HI},
                         "online": True, "B": a.B, "K_G": a.K, "K_V": a.KV, "steps": a.steps,
                         "NU": NU, "N_LOC": N_LOC, "seed": a.seed,
                         "schedule": a.schedule, "sampler": a.sampler, "drift": net.drift,
                         "plateau": {"pass": net.plateau[0], "why": net.plateau[1],
                                     "curve": net.plateau[2]},
                         "spec": "g_eff field online "}}, out)
    print(f"saved {out}")
    if ckpt and os.path.exists(ckpt):
        os.remove(ckpt)
