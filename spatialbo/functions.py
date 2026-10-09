"""The test functions of the paper, on the unit cube and in minimization form.

    f, D = FUNCS["hartmann_3d"]          # f: (M, D) tensor in [0,1]^D -> (M,)
    h, D = standardized("hartmann_3d")   # maximized and standardized, as the experiments see it

The benchmark roster of Section 4.2 (domains as in the Simon Fraser test
suite) and the controlled landscape of Section 4.1, mb4_{D}d_{variant}.
"""
import math

import torch


def _to_domain(U, lo, hi):
    lo = torch.as_tensor(lo, dtype=U.dtype, device=U.device)
    hi = torch.as_tensor(hi, dtype=U.dtype, device=U.device)
    return lo + (hi - lo) * U


def _cube(U, D):
    U = U.reshape(1, -1) if U.ndim == 1 else U
    assert U.shape[-1] == D, f"expected D={D}, got {U.shape[-1]}"
    return U


# --------------------------------------------------------------------------
# 2-D
# --------------------------------------------------------------------------
def branin(U):
    """Branin-Hoo. f* = 0.397887 at three global minima, all interior."""
    X = _to_domain(_cube(U, 2), [-5.0, 0.0], [10.0, 15.0])
    x1, x2 = X[:, 0], X[:, 1]
    b, c, t = 5.1 / (4.0 * math.pi ** 2), 5.0 / math.pi, 1.0 / (8.0 * math.pi)
    return ((x2 - b * x1 ** 2 + c * x1 - 6.0) ** 2
            + 10.0 * (1.0 - t) * torch.cos(x1) + 10.0)


def camel6(U):
    """Six-hump camel. f* = -1.031628 at (+-0.0898, -+0.7126)."""
    X = _to_domain(_cube(U, 2), [-3.0, -2.0], [3.0, 2.0])
    x1, x2 = X[:, 0], X[:, 1]
    return ((4.0 - 2.1 * x1 ** 2 + x1 ** 4 / 3.0) * x1 ** 2
            + x1 * x2 + (-4.0 + 4.0 * x2 ** 2) * x2 ** 2)


def beale(U):
    """Beale. f* = 0 at (3, 0.5), near a corner of [-4.5, 4.5]^2."""
    X = _to_domain(_cube(U, 2), [-4.5, -4.5], [4.5, 4.5])
    x, y = X[:, 0], X[:, 1]
    return ((1.5 - x + x * y) ** 2 + (2.25 - x + x * y ** 2) ** 2
            + (2.625 - x + x * y ** 3) ** 2)


def zakharov_5d(U):
    """Zakharov, D=5. f* = 0 at the origin (u_i = 1/3); one broad basin."""
    X = _to_domain(_cube(U, 5), [-5.0] * 5, [10.0] * 5)
    i = torch.arange(1, 6, dtype=U.dtype, device=U.device)
    s2 = (0.5 * i * X).sum(-1)
    return (X ** 2).sum(-1) + s2 ** 2 + s2 ** 4


def rastrigin_3d(U):
    """Rastrigin, D=3. f* = 0 at the center; dense periodic local minima."""
    X = _to_domain(_cube(U, 3), [-5.12] * 3, [5.12] * 3)
    return 10.0 * 3 + (X ** 2 - 10.0 * torch.cos(2.0 * math.pi * X)).sum(-1)


# --------------------------------------------------------------------------
# families
# --------------------------------------------------------------------------
def _levy(X):
    w = 1.0 + (X - 1.0) / 4.0
    t1 = torch.sin(math.pi * w[:, 0]) ** 2
    wi = w[:, :-1]
    t2 = ((wi - 1.0) ** 2
          * (1.0 + 10.0 * torch.sin(math.pi * wi + 1.0) ** 2)).sum(dim=1)
    t3 = (w[:, -1] - 1.0) ** 2 * (1.0 + torch.sin(2.0 * math.pi * w[:, -1]) ** 2)
    return t1 + t2 + t3


def _rosenbrock(X):
    return (100.0 * (X[:, 1:] - X[:, :-1] ** 2) ** 2
            + (1.0 - X[:, :-1]) ** 2).sum(dim=1)


def _hartmann(X, alpha, A, P):
    dt, dv = X.dtype, X.device
    alpha = torch.tensor(alpha, dtype=dt, device=dv)
    A = torch.tensor(A, dtype=dt, device=dv)
    P = torch.tensor(P, dtype=dt, device=dv)
    inner = (A * (X.unsqueeze(1) - P) ** 2).sum(dim=2)
    return -(alpha * torch.exp(-inner)).sum(dim=1)


_H_ALPHA = [1.0, 1.2, 3.0, 3.2]
_H3_A = [[3.0, 10.0, 30.0], [0.1, 10.0, 35.0],
         [3.0, 10.0, 30.0], [0.1, 10.0, 35.0]]
_H3_P = [[0.3689, 0.1170, 0.2673], [0.4699, 0.4387, 0.7470],
         [0.1091, 0.8732, 0.5547], [0.0381, 0.5743, 0.8828]]
_H4_A = [[10.0, 3.0, 17.0, 3.5], [0.05, 10.0, 17.0, 0.1],
         [3.0, 3.5, 1.7, 10.0], [17.0, 8.0, 0.05, 10.0]]
_H4_P = [[0.1312, 0.1696, 0.5569, 0.0124], [0.2329, 0.4135, 0.8307, 0.3736],
         [0.2348, 0.1451, 0.3522, 0.2883], [0.4047, 0.8828, 0.8732, 0.5743]]
_H6_A = [[10.0, 3.0, 17.0, 3.5, 1.7, 8.0], [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
         [3.0, 3.5, 1.7, 10.0, 17.0, 8.0], [17.0, 8.0, 0.05, 10.0, 0.1, 14.0]]
_H6_P = [[0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
         [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
         [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
         [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]]


def levy2d(U):  return _levy(_to_domain(_cube(U, 2), [-10.0] * 2, [10.0] * 2))
def levy_3d(U): return _levy(_to_domain(_cube(U, 3), [-10.0] * 3, [10.0] * 3))
def levy_4d(U): return _levy(_to_domain(_cube(U, 4), [-10.0] * 4, [10.0] * 4))


def rosenbrock_4d(U):
    return _rosenbrock(_to_domain(_cube(U, 4), [-2.048] * 4, [2.048] * 4))


def hartmann_3d(U):
    """f* = -3.86278 at (0.1146, 0.5556, 0.8525). Native unit cube."""
    return _hartmann(_cube(U, 3), _H_ALPHA, _H3_A, _H3_P)


def hartmann_4d(U):
    """Picheny et al. (2012) rescaling; f* = -3.134494 at (0.1874, 0.1942, 0.5579, 0.2648)."""
    outer = -_hartmann(_cube(U, 4), _H_ALPHA, _H4_A, _H4_P)
    return (1.1 - outer) / 0.839


def hartmann_6d(U):
    """f* = -3.32237 at (0.2017,0.1500,0.4769,0.2753,0.3117,0.6573)."""
    return _hartmann(_cube(U, 6), _H_ALPHA, _H6_A, _H6_P)


def styblinski_tang_3d(U):
    """f* = -39.16599 d at x_i = -2.903534 (u_i = 0.2096), near a corner."""
    X = _to_domain(_cube(U, 3), [-5.0] * 3, [5.0] * 3)
    return 0.5 * (X ** 4 - 16.0 * X ** 2 + 5.0 * X).sum(dim=1)


def michalewicz_5d(U):
    """Michalewicz, m=10, D=5. f* = -4.687658 on [0, pi]^5."""
    X = _to_domain(_cube(U, 5), [0.0] * 5, [math.pi] * 5)
    i = torch.arange(1, 6, dtype=X.dtype, device=X.device)
    return -(torch.sin(X) * torch.sin(i * X ** 2 / math.pi) ** 20).sum(dim=1)


# --------------------------------------------------------------------------
# Controlled synthetic landscape (Section 4.1): a weak periodic background
# with Gaussian basins of identical shape at a corner, a face, the interior
# and (D >= 3) an edge; the variant names the basin that is deepened.
# --------------------------------------------------------------------------
_MB_RIPPLE = 0.06                  # amplitude of the periodic background
_MB_A_LOC, _MB_A_GLOB = 1.0, 1.25  # depth of a local basin and of the global one

_MB4_CFG = {
    # D: ({variant: centre}, s, budget)
    3: ({"corner": (0.102, 0.898, 0.898),
         "edge": (0.634, 0.898, 0.102),
         "face": (0.102, 0.384, 0.234),
         "interior": (0.692, 0.292, 0.692)}, 0.146, 40),
    4: ({"corner": (0.129, 0.129, 0.871, 0.871),
         "edge": (0.544, 0.129, 0.344, 0.871),
         "face": (0.871, 0.694, 0.294, 0.694),
         "interior": (0.367, 0.517, 0.617, 0.367)}, 0.184, 55),
}
# 2D first (three addresses: codim 2, 1, 0), then the two carrying four.
_MB4_CFG = {2: ({"corner": (0.07, 0.07),
                 "face": (0.16, 0.93),
                 "interior": (0.80, 0.42)}, 0.100, 25), **_MB4_CFG}
MB4_VARIANTS = {D: tuple(_MB4_CFG[D][0]) for D in _MB4_CFG}
MB4_CODIM = {D: {v: sum(1 for x in _MB4_CFG[D][0][v]
                        if min(x, 1 - x) < 0.5 * _MB4_CFG[D][2] ** (-1.0 / D))
                 for v in _MB4_CFG[D][0]} for D in _MB4_CFG}


def _multibasin4(U, D, gv):
    P, s, _ = _MB4_CFG[D]
    U = _cube(U, D)
    out = _MB_RIPPLE * torch.cos(2.0 * math.pi * U).mean(-1)
    for v in P:
        p = P[v]
        a = _MB_A_GLOB if v == gv else _MB_A_LOC
        q = sum((U[:, j] - p[j]) ** 2 for j in range(D))
        out = out - a * torch.exp(-q / (2.0 * s ** 2))
    return out


FUNCS = {
    "branin": (branin, 2), "camel6": (camel6, 2), "beale": (beale, 2),
    "levy2d": (levy2d, 2),
    "hartmann_3d": (hartmann_3d, 3), "levy_3d": (levy_3d, 3),
    "styblinski_tang_3d": (styblinski_tang_3d, 3),
    "hartmann_4d": (hartmann_4d, 4), "levy_4d": (levy_4d, 4),
    "rosenbrock_4d": (rosenbrock_4d, 4),
    "michalewicz_5d": (michalewicz_5d, 5), "hartmann_6d": (hartmann_6d, 6),
    "zakharov_5d": (zakharov_5d, 5), "rastrigin_3d": (rastrigin_3d, 3),
}

for _D in _MB4_CFG:
    for _v in MB4_VARIANTS[_D]:
        FUNCS[f"mb4_{_D}d_{_v}"] = (
            (lambda U, d=_D, v=_v: _multibasin4(U, d, v)), _D)
MB4_POS = {d: _MB4_CFG[d][0] for d in _MB4_CFG}
MB4_S = {d: _MB4_CFG[d][1] for d in _MB4_CFG}
MB4_B = {d: _MB4_CFG[d][2] for d in _MB4_CFG}

DIM = {k: d for k, (_, d) in FUNCS.items()}


def standardized(name, n_ref=8192, seed=999):
    """The function `name` maximized and standardized: h = -(f - m)/s with m, s
    over a fixed Sobol set of n_ref points. Returns (h, D)."""
    f, D = FUNCS[name]
    big = torch.quasirandom.SobolEngine(D, scramble=True, seed=seed).draw(n_ref).to(torch.float64)
    fb = f(big).reshape(-1)
    m, s = fb.mean(), fb.std().clamp_min(1e-8)
    return (lambda U: -(f(U).reshape(-1) - m) / s), D
