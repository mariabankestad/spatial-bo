"""Trainer of the g_geo surrogate on fresh realizations (used for the released
fields): g(x; N, ell) = E[sigma(c | X_N)] / E[sigma(x | X_N)] learned from new
designs at every step. Validation against the gold nodes every 5% of the
steps; checkpoints allow resuming.

    python train_field_online.py --D 3 --seed 0 --steps 600000 --K 8 --schedule final --loss rowrel --sampler final
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, "experiments"), _os.path.join(_ROOT, "training"), _os.path.join(_ROOT, "plots")):
    _p in _sys.path or _sys.path.insert(0, _p)
import argparse
import copy
import math
import os
import time

import numpy as np
import torch

from spatialbo.gp import matern52
from gen_field_v2 import EPS, ELL_LO, ELL_HI, wall_points
from gen_field_corpus import sample_points
# trained range in n: [2D, 150]; overridable, written to the metadata
N_LO_MULT = int(os.environ.get("FIELD_N_LO_MULT", 2))
N_HI = int(os.environ.get("FIELD_N_HI", 150))
from train_field_v2 import (FieldV2, DT, DEV, OUTDIR, LAM_C, valgold_mae,
                            plateau_verdict)


class OnlineSampler:
    """Sampler of (N, ell, probes, designs) per step. 'v5' draws rho = ell/h over the
    range reachable at the step's N and gives each row K independent designs;
    'v3' is the earlier version."""
    def __init__(self, D, seed, B=48, K=6, P=200, n_scr=4, version="v3"):
        self.D, self.B, self.K, self.P, self.version = D, B, K, P, version
        self.n_scr = K if version in ("v5", "v6", "final") else n_scr
        self.gap_step = False
        self.n_lo = N_LO_MULT * D
        self.gen = torch.Generator().manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.counter = 10_000_000 * (seed + 1)      # design seeds never reused
        h_lo, h_hi = N_HI ** (-1.0 / D), self.n_lo ** (-1.0 / D)
        self.rho_lo, self.rho_hi = ELL_LO / h_hi, ELL_HI / h_lo

    def state_dict(self):
        return {"gen": self.gen.get_state(), "rng": self.rng.bit_generator.state,
                "counter": self.counter}

    def load_state_dict(self, d):
        self.gen.set_state(d["gen"].cpu()); self.rng.bit_generator.state = d["rng"]
        self.counter = d["counter"]

    def draw_n(self):
        """N per step: 70% log-uniform on [n_lo, N_HI], 30% on [n_lo, 2 n_lo];
    'v6' adds 15% on [N_HI/1.5, N_HI]."""
        u = self.rng.random()
        self.gap_step = False
        if self.version == "final" and u < 0.10:
            # 10% of steps draw n in [n_lo, 4 n_lo] with ell in [0.5, 2]
            self.gap_step = True
            return int(round(self.n_lo * 4.0 ** self.rng.random()))
        if u < 0.30:
            return int(round(self.n_lo * 2.0 ** self.rng.random()))
        if self.version == "v6" and u < 0.45:
            return int(round((N_HI / 1.5) * 1.5 ** self.rng.random()))
        return int(round(self.n_lo * (N_HI / self.n_lo) ** self.rng.random()))

    def draw_ells(self, n):
        h = n ** (-1.0 / self.D)
        u = self.rng.random(self.B); v = self.rng.random(self.B)
        if self.gap_step:
            return torch.tensor(0.5 * (2.0 / 0.5) ** v, dtype=DT)
        if self.version in ("v5", "v6", "final"):
            rlo, rhi = ELL_LO / h, ELL_HI / h                 # reachable rho range for THIS n
            rho_branch = rlo * (rhi / rlo) ** v * h            # == ell in [ELL_LO, ELL_HI], no clip
        else:
            rho_branch = np.clip(self.rho_lo * (self.rho_hi / self.rho_lo) ** v * h, ELL_LO, ELL_HI)
        ell = np.where(u < 0.50, ELL_LO * (ELL_HI / ELL_LO) ** v,
              np.where(u < 0.80, rho_branch,
              np.where(u < 0.90, ELL_LO * 2.0 ** v, (ELL_HI / 2.0) * 2.0 ** v)))
        return torch.tensor(ell, dtype=DT)

    def draw_points(self, ell):
        half = self.P // 2
        return torch.stack([torch.cat([sample_points(half, self.D, self.gen),
                                       wall_points(self.P - half, self.D, float(e), self.gen)])
                            for e in ell])                                  # (B, P, D)

    def draw_designs(self, n):
        """(B*K, n, D) designs, row-major; 'v5' gives row i K independent
    (scramble, shift) realizations. Scramble seeds advance every step."""
        R = self.B * self.K
        if self.version in ("v5", "v6", "final"):
            blocks = []
            for k in range(self.K):
                Z = torch.quasirandom.SobolEngine(self.D, scramble=True, seed=self.counter
                                                  ).draw(n).to(DT)
                self.counter += 1
                sh = torch.rand(self.B, 1, self.D, generator=self.gen, dtype=DT)
                blocks.append((Z[None] + sh) % 1.0)                 # (B, n, D)
            return torch.stack(blocks, 1).reshape(R, n, self.D)       # (B, K, n, D) -> row-major
        per = R // self.n_scr
        out = []
        for s in range(self.n_scr):
            Z = torch.quasirandom.SobolEngine(self.D, scramble=True, seed=self.counter
                                              ).draw(n).to(DT)
            self.counter += 1
            sh = torch.rand(per, 1, self.D, generator=self.gen, dtype=DT)
            out.append((Z[None] + sh) % 1.0)
        return torch.cat(out)[:R]

    def batch(self):
        n = self.draw_n(); ell = self.draw_ells(n)
        U = self.draw_points(ell)                                            # (B,P,D)
        xc = torch.full((self.B, 1, self.D), 0.5, dtype=DT)
        Q = torch.cat([U, xc], 1).repeat_interleave(self.K, 0).to(DEV)       # (B*K,P+1,D)
        Xd = self.draw_designs(n).to(DEV)
        e3 = ell.repeat_interleave(self.K).to(DEV).view(-1, 1, 1)
        Kmat = matern52(Xd, Xd, e3) + EPS * torch.eye(n, dtype=DT, device=DEV)
        L = torch.linalg.cholesky(Kmat)
        z = torch.linalg.solve_triangular(L, matern52(Xd, Q, e3), upper=False)
        sig = (1.0 - (z * z).sum(1)).clamp_min(0).sqrt().reshape(self.B, self.K, -1).mean(1)
        y, yc = sig[:, :-1], sig[:, -1]
        ln = torch.full((self.B,), math.log(n), dtype=DT, device=DEV)
        le = torch.log(ell).to(DEV)
        return U.to(DEV), ln, le, y, yc


def train(D, steps, seed, B=48, K=6, floors=(0.0, 0.1, 0.03, 0.01, 0.005, 0.003),
          lr=1e-3, schedule="v3", loss_mode="global", sampler="v3", ckpt=None):
    """Train one network. schedule: see schedule.py ('final' returns the Polyak
    average). loss_mode 'rowrel': relative weights with the floor relative to
    the row's own center prediction; 'global': floor on the absolute scale."""
    from schedule import lr_factor, floor_at, PolyakAverage, field_drift, drift_probes
    torch.manual_seed(seed)
    n_lo = N_LO_MULT * D
    # feature normalisation from the sampling ranges (no corpus to take it from)
    ln_mu = 0.5 * (math.log(n_lo) + math.log(N_HI)); ln_sd = (math.log(N_HI) - math.log(n_lo)) / math.sqrt(12)
    le_mu = 0.5 * (math.log(ELL_LO) + math.log(ELL_HI)); le_sd = (math.log(ELL_HI) - math.log(ELL_LO)) / math.sqrt(12)
    net = FieldV2(D, ln_mu, ln_sd, le_mu, le_sd).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, steps, schedule, lr))
    avg = (PolyakAverage(net, steps, start_frac=0.6 if schedule == "final" else 0.8)
           if schedule in ("v3", "final") else None)
    probes = drift_probes(D, n_lo, N_HI, ELL_LO, ELL_HI, DEV)
    snap = None
    sampler = OnlineSampler(D, seed, B=B, K=K, version=sampler)
    curve = []; t0 = time.time(); start = 0
    # checkpoint every 2% of the steps: net, optimizer, Polyak state, validation curve, RNG states
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
        U, ln, le, y, yc = sampler.batch()
        P = U.shape[1]
        Xb = U.reshape(-1, D); lnb = ln.repeat_interleave(P); leb = le.repeat_interleave(P)
        mhat = net.m(Xb, lnb, leb); mcb = net.m_c(ln, le)
        if floor > 0 and loss_mode == "rowrel":
            row = mcb.detach().clamp_min(1e-12)                  # row scale = centre level
            fl = (row * floor).repeat_interleave(P)               # floor relative to the row
            w = 1.0 / torch.maximum(mhat.detach(), fl) ** 2
            wc = 1.0 / torch.maximum(mcb.detach(), row * floor) ** 2
            loss = (w * (mhat - y.reshape(-1)) ** 2).mean() + LAM_C * (wc * (mcb - yc) ** 2).mean()
        elif floor > 0:
            w = 1.0 / mhat.detach().clamp_min(floor) ** 2
            wc = 1.0 / mcb.detach().clamp_min(floor) ** 2
            loss = (w * (mhat - y.reshape(-1)) ** 2).mean() + LAM_C * (wc * (mcb - yc) ** 2).mean()
        else:
            loss = ((mhat - y.reshape(-1)) ** 2).mean() + LAM_C * ((mcb - yc) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if avg is not None:
            avg.update(net, step)
        if step == int(0.98 * steps) - 1:
            snap = copy.deepcopy(net)                      # iterate 2% before the end
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
    ap.add_argument("--steps", type=int, default=int(os.environ.get("STEPS", 320000)))
    ap.add_argument("--B", type=int, default=48)
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--schedule", choices=("legacy", "v3", "final"), default="v3")
    ap.add_argument("--loss", choices=("global", "rowrel"), default="global")
    ap.add_argument("--sampler", choices=("v3", "v5", "v6", "final"), default="v3")
    ap.add_argument("--resume", action="store_true",
                    help="continue from <artifact>.ckpt if it exists (final runs span allocations)")
    a = ap.parse_args()
    name = f"field_v2_D{a.D}.pt" if a.seed == 0 else f"field_v2_D{a.D}_seed{a.seed}.pt"
    out = os.path.join(OUTDIR, os.environ.get("F2_OUT", name))
    os.makedirs(OUTDIR, exist_ok=True)
    ckpt = out + ".ckpt" if a.resume else None
    net = train(a.D, a.steps, a.seed, B=a.B, K=a.K, schedule=a.schedule, loss_mode=a.loss,
                sampler=a.sampler, ckpt=ckpt)
    torch.save({"state": net.state_dict(), "D": a.D, "stats": net.stats,
                "meta": {"arch": "centred-log-relative", "eps": EPS,
                         "QN": "random-shifted scrambled Sobol (online, fresh per step)",
                         "envelope": {"n_lo": N_LO_MULT * a.D, "n_hi": N_HI,
                                      "ell_lo": ELL_LO, "ell_hi": ELL_HI},
                         "online": True, "B": a.B, "K": a.K, "steps": a.steps,
                         "seed": a.seed, "schedule": a.schedule, "loss": a.loss, "sampler": a.sampler,
                         "drift": net.drift,
                         "sampling": "n: 70% loguniform / 30% low-N floor; ell: 50% loguniform, "
                                     "30% rho-loguniform, 10%+10% edges; x: 50% codim-stratified, "
                                     "50% wall layer [ell/100, 3 ell]",
                         "plateau": {"pass": net.plateau[0], "why": net.plateau[1],
                                     "curve": net.plateau[2]},
                         "spec": "field-v2 online "}}, out)
    print(f"saved {out}")
    if ckpt and os.path.exists(ckpt):
        os.remove(ckpt)
