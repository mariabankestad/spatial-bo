"""Pack trained artifacts into one release file per field and dimension.

    python pack_field.py --kind geo --D 2 field_v2_D2.pt field_v2_D2_seed1.pt
    python pack_field.py --kind uniform --D 2 field_gu_D2.pt

Writes fields/<kind>/g_<kind>_D<D>.pt with the inputs as members; the field
is their geometric mean.
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(_os.path.abspath(__file__))))
_ROOT in _sys.path or _sys.path.insert(0, _ROOT)
import argparse
import os

import torch

from spatialbo.fields import FIELD_DIR, KINDS


def pack(kind, D, paths, out=None):
    members = []
    for p in paths:
        z = torch.load(p, map_location="cpu", weights_only=False)
        members.append({"state": z["state"], "stats": tuple(z["stats"]), "meta": z.get("meta", {})})
    env = members[0]["meta"]["envelope"]
    out = out or os.path.join(FIELD_DIR, kind, f"g_{kind}_D{D}.pt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save({"kind": kind, "D": int(D), "members": members,
                "envelope": {k: env[k] for k in ("n_lo", "n_hi", "ell_lo", "ell_hi")}}, out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--kind", required=True, choices=sorted(KINDS))
    ap.add_argument("--D", type=int, required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("artifacts", nargs="+")
    a = ap.parse_args()
    print("wrote", pack(a.kind, a.D, a.artifacts, a.out))
