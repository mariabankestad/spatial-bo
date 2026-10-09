"""The weight-field surrogates: one permutation- and reflection-invariant
architecture, with one set of weights per principle and dimension.
FieldV2 represents a single log-score (g_geo, and the g_uniform fit);
FieldGeff has two heads, for V and G, and returns g_eff = sqrt(G/V)."""
import torch
import torch.nn as nn

DT = torch.float64


def _mlp(sizes, final_linear=True):
    layers = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), nn.SiLU()]
    if final_linear:
        layers = layers[:-1]
    return nn.Sequential(*layers)


class FieldV2(nn.Module):
    def __init__(self, D, ln_mu, ln_sd, le_mu, le_sd, emb=32):
        super().__init__()
        self.D = D
        self.stats = (ln_mu, ln_sd, le_mu, le_sd)
        self.phi = _mlp([2, 32, 32, emb])
        self.rho = _mlp([emb + 8, 64, 64, 32, 1])
        self.bnet = _mlp([2, 32, 32, 1])
        self.double()

    def _r(self, X, ln, le):
        """v1 smooth_all features, verbatim; linear output (log scale)."""
        ln_mu, ln_sd, le_mu, le_sd = self.stats
        ell = torch.exp(le)
        dlt = 2.0 * X * (1.0 - X)
        u = torch.stack([2 * dlt, torch.log1p(dlt / ell[:, None])], dim=-1)
        h = self.phi(u).sum(1)
        P = (2.0 * dlt).prod(1)
        ssum = torch.exp(-dlt / ell[:, None]).sum(1)
        gl = torch.stack([(ln - ln_mu) / ln_sd, (le - le_mu) / le_sd,
                          torch.exp(ln) ** (1.0 / self.D) / 5.0,
                          torch.full_like(ln, float(self.D)),
                          ssum / self.D, P,
                          torch.log1p(P / ell),
                          torch.log1p(ssum)], dim=-1)
        return self.rho(torch.cat([h, gl], dim=-1)).squeeze(-1)

    def _b(self, ln, le):
        ln_mu, ln_sd, le_mu, le_sd = self.stats
        z = torch.stack([(ln - ln_mu) / ln_sd, (le - le_mu) / le_sd], dim=-1)
        return self.bnet(z).squeeze(-1)

    def delta(self, X, ln, le):
        xc = torch.full_like(X, 0.5)
        return self._r(X, ln, le) - self._r(xc, ln, le)

    def m(self, X, ln, le):
        return torch.exp(self._b(ln, le) + self.delta(X, ln, le))

    def m_c(self, ln, le):
        return torch.exp(self._b(ln, le))

    def g(self, X, ln, le):
        return torch.exp(-self.delta(X, ln, le))


class FieldGeff(nn.Module):
    def __init__(self, D, ln_mu, ln_sd, le_mu, le_sd, emb=32):
        super().__init__()
        self.D = D
        self.stats = (ln_mu, ln_sd, le_mu, le_sd)
        self.phiV = _mlp([2, 32, 32, emb]); self.rhoV = _mlp([emb + 8, 64, 64, 32, 1])
        self.phiG = _mlp([2, 32, 32, emb]); self.rhoG = _mlp([emb + 8, 64, 64, 32, 1])
        self.bV = _mlp([2, 32, 32, 1]); self.bG = _mlp([2, 32, 32, 1])
        self.double()

    def _feat(self, X, ln, le):
        ln_mu, ln_sd, le_mu, le_sd = self.stats
        ell = torch.exp(le)
        dlt = 2.0 * X * (1.0 - X)
        u = torch.stack([2 * dlt, torch.log1p(dlt / ell[:, None])], dim=-1)
        P = (2.0 * dlt).prod(1)
        ssum = torch.exp(-dlt / ell[:, None]).sum(1)
        gl = torch.stack([(ln - ln_mu) / ln_sd, (le - le_mu) / le_sd,
                          torch.exp(ln) ** (1.0 / self.D) / 5.0,
                          torch.full_like(ln, float(self.D)),
                          ssum / self.D, P,
                          torch.log1p(P / ell), torch.log1p(ssum)], dim=-1)
        return u, gl

    def _r(self, X, ln, le, which):
        u, gl = self._feat(X, ln, le)
        phi, rho = ((self.phiV, self.rhoV) if which == "V"
                    else (self.phiG, self.rhoG))
        return rho(torch.cat([phi(u).sum(1), gl], dim=-1)).squeeze(-1)

    def _b(self, ln, le, which):
        ln_mu, ln_sd, le_mu, le_sd = self.stats
        z = torch.stack([(ln - ln_mu) / ln_sd, (le - le_mu) / le_sd], dim=-1)
        return (self.bV if which == "V" else self.bG)(z).squeeze(-1)

    def pred(self, X, ln, le, which):
        return torch.exp(self._b(ln, le, which) + self._r(X, ln, le, which))

    def pred_c(self, ln, le, which):
        xc = torch.full((len(ln), self.D), 0.5, dtype=DT, device=ln.device)
        return torch.exp(self._b(ln, le, which) + self._r(xc, ln, le, which))

    def g(self, X, ln, le):
        xc = torch.full_like(X, 0.5)
        dG = self._r(X, ln, le, "G") - self._r(xc, ln, le, "G")
        dV = self._r(X, ln, le, "V") - self._r(xc, ln, le, "V")
        return torch.exp(0.5 * (dG - dV))
