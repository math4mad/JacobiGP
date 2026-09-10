"""Reference baselines: a classical (spatial) RBF-GP and finite-difference differentiation.

Used by the experiments to show what the Jacobi basis buys:
* an RBF GP on a bounded interval suffers the well-known variance collapse at the
  edges (it 'thinks' the function keeps living outside [-1,1]);
* differentiating an RBF GP posterior by finite differences is noisy and needs the
  full grid, while the Jacobi GP has the derivative in closed form.
"""

from __future__ import annotations

import math

import torch

from .model import GPParams, _cholesky_spd


class RBFGP(torch.nn.Module):
    def __init__(self, lengthscale=0.3, sigma2=1.0, noise=1e-2, learn=("lengthscale", "sigma2", "noise"),
                 ard=False, dim=1, dtype=torch.float64):
        super().__init__()
        self.dtype = dtype
        self.params = GPParams(0.0, 0.0, lengthscale, sigma2, noise, learn=learn)
        self.ard = ard
        self.dim = dim
        if ard:
            self.ls = torch.nn.Parameter(
                torch.full((dim,), float(lengthscale), dtype=dtype), requires_grad="lengthscale" in learn
            )

    def _ls(self):
        return self.ls if self.ard else self.params.lengthscale.reshape(1).expand(self.dim)

    def kernel(self, x1, x2=None):
        x1 = torch.as_tensor(x1, dtype=self.dtype).reshape(len(x1), -1)
        x2 = x1 if x2 is None else torch.as_tensor(x2, dtype=self.dtype).reshape(len(x2), -1)
        ls = self._ls()
        d = (x1[:, None, :] - x2[None, :, :]) / ls
        r2 = (d**2).sum(-1)
        if self.ard:
            amp = torch.exp(self.params.raw_sigma2) / ls.sqrt().prod()  # ARD normalisation
        else:
            amp = self.params.sigma2
        return amp * torch.exp(-0.5 * r2)

    def log_marginal_likelihood(self, X, y):
        y = torch.as_tensor(y, dtype=self.dtype).reshape(-1)
        K = self.kernel(X) + self.params.noise * torch.eye(len(y), dtype=self.dtype)
        L = _cholesky_spd(K, jitter0=0.0)
        alpha_ = torch.cholesky_solve(y.reshape(-1, 1), L).reshape(-1)
        return -(0.5 * (y @ alpha_) + torch.log(torch.diagonal(L)).sum()
                 + 0.5 * len(y) * math.log(2 * math.pi))

    def fit(self, X, y, steps=200, lr=0.05, method="adam", verbose=False):
        trainable = [p for p in self.parameters() if p.requires_grad]
        if method == "adam":
            opt = torch.optim.Adam(trainable, lr=lr)
        else:
            opt = torch.optim.LBFGS(trainable, max_iter=steps, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = -self.log_marginal_likelihood(X, y)
            loss.backward()
            return loss

        history = []
        for it in range(1 if method != "adam" else steps):
            history.append(-float(opt.step(closure)))
            if verbose and it % max(1, steps // 10) == 0:
                print(f"  [{it:4d}] lml={history[-1]: .4f}  {self.params.as_dict()}")
        return history

    def predict(self, Xs, X, y, noise=False, deriv=0, fd_eps=None):
        """Latent mean / variance.

        deriv = 1, 2 uses the ANALYTIC derivative of the RBF kernel (the reference
        implementation);  deriv = -1, -2 instead finite-differences the posterior mean,
        which is what one is forced to do with a kernel that has no derivative formula.
        """
        Xs = torch.as_tensor(Xs, dtype=self.dtype).reshape(-1)
        X = torch.as_tensor(X, dtype=self.dtype).reshape(-1)
        y = torch.as_tensor(y, dtype=self.dtype).reshape(-1)
        K = self.kernel(X) + self.params.noise * torch.eye(len(y), dtype=self.dtype)
        L = _cholesky_spd(K, jitter0=0.0)
        a = torch.cholesky_solve(y.reshape(-1, 1), L).reshape(-1)

        if deriv in (0,):
            Ks = self.kernel(Xs, X)
            mean = Ks @ a
            v = torch.linalg.solve_triangular(L, Ks.T, upper=False)
            var = self.params.sigma2 - (v**2).sum(0)
            if noise:
                var = var + self.params.noise
            return mean, var

        if deriv < 0:                                  # finite differences of the mean
            h = fd_eps if fd_eps is not None else 1e-2
            order = -deriv
            if order == 1:
                return (self.predict(Xs + h, X, y)[0] - self.predict(Xs - h, X, y)[0]) / (2 * h), None
            m_p, _ = self.predict(Xs + h, X, y)
            m_0, _ = self.predict(Xs, X, y)
            m_m, _ = self.predict(Xs - h, X, y)
            return (m_p - 2 * m_0 + m_m) / h**2, None

        # analytic derivatives of the squared-exponential kernel
        ls = self.params.lengthscale
        d = Xs.reshape(-1, 1) - X.reshape(1, -1)          # (M*, M)
        K = self.params.sigma2 * torch.exp(-0.5 * (d / ls) ** 2)
        if deriv == 1:
            Ks = -K * d / ls**2
            var = None
        else:
            Ks = K * (d**2 / ls**4 - 1.0 / ls**2)
            var = None
        return Ks @ a, var

