"""Uniformity statistics of a selection distribution: energy distance to the
uniform law with its null, and Kolmogorov-Smirnov distances of the radial and
coordinate marginals.
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, "experiments"), _os.path.join(_ROOT, "training"), _os.path.join(_ROOT, "plots")):
    _p in _sys.path or _sys.path.insert(0, _p)
import numpy as np


def _pdist2(A, B):
    return ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)


def energy_distance(X, Y, rng=None, n_perm=200):
    """E-distance between samples, with a permutation p-value."""
    X = np.asarray(X, float); Y = np.asarray(Y, float)
    def stat(A, B):
        dAB = np.sqrt(_pdist2(A, B)).mean()
        dAA = np.sqrt(_pdist2(A, A)).mean()
        dBB = np.sqrt(_pdist2(B, B)).mean()
        return 2 * dAB - dAA - dBB
    e = stat(X, Y)
    if n_perm <= 0:
        return e, float("nan")
    rng = rng or np.random.default_rng(0)
    Z = np.vstack([X, Y]); n = len(X)
    cnt = 0
    for _ in range(n_perm):
        idx = rng.permutation(len(Z))
        if stat(Z[idx[:n]], Z[idx[n:]]) >= e:
            cnt += 1
    return e, (cnt + 1) / (n_perm + 1)


def mmd(X, Y):
    """Gaussian-kernel MMD^2 with the median heuristic."""
    X = np.asarray(X, float); Y = np.asarray(Y, float)
    Z = np.vstack([X, Y])
    med = np.median(np.sqrt(_pdist2(Z, Z))[np.triu_indices(len(Z), 1)])
    s2 = max(med, 1e-9) ** 2
    k = lambda A, B: np.exp(-_pdist2(A, B) / (2 * s2))
    return float(k(X, X).mean() + k(Y, Y).mean() - 2 * k(X, Y).mean())


def radial_ks(X, D):
    """The 1-D projection: KS of face distance vs its uniform law.
    Diagnostic only; passing it does not imply uniformity."""
    X = np.asarray(X, float)
    r = np.minimum(X, 1 - X).min(axis=1)
    pit = np.sort(1 - (1 - 2 * np.minimum(r, 0.5)) ** D)
    n = len(pit)
    return float(np.max(np.abs(pit - np.arange(1, n + 1) / n)))


def coord_report(X):
    """Per-coordinate mean and KS vs U(0,1): catches folding/asymmetry."""
    X = np.asarray(X, float)
    out = []
    for j in range(X.shape[1]):
        v = np.sort(X[:, j]); n = len(v)
        ks = float(np.max(np.abs(v - np.arange(1, n + 1) / n)))
        out.append((float(v.mean()), ks))
    return out


def certify(X, D, rng=None, n_perm=200):
    """Full report for a set of selected points."""
    X = np.asarray(X, float)
    rng = rng or np.random.default_rng(0)
    Y = rng.random((len(X), D))
    e, p = energy_distance(X, Y, rng=rng, n_perm=n_perm)
    # scale: the same statistic between two independent uniform samples
    e0, _ = energy_distance(rng.random((len(X), D)),
                            rng.random((len(X), D)), n_perm=0)
    return {"energy": e, "energy_null": e0, "p_value": p,
            "mmd": mmd(X, Y), "radial_ks": radial_ks(X, D),
            "coords": coord_report(X)}


def format_report(rep):
    c = rep["coords"]
    return (f"energy {rep['energy']:+.4f} (null {rep['energy_null']:+.4f}, "
            f"p={rep['p_value']:.3f})  mmd {rep['mmd']:.4f}  "
            f"radial-KS {rep['radial_ks']:.3f}  "
            f"coord-mean [{min(m for m, _ in c):.2f},{max(m for m, _ in c):.2f}]  "
            f"coord-KS max {max(k for _, k in c):.3f}")
