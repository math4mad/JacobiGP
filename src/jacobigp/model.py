"""
JacobiGP : a truncated Gaussian process written in the orthonormal Jacobi basis.

    f_N(x) = sum_{n<N} w_n  \\hat P_n^{(alpha,beta)}(x),      w ~ N(0, Lambda)
    y_i    = f_N(x_i) + eps_i,                               eps ~ N(0, noise)

Inference is closed-form Bayesian linear regression, O(M N^2 + N^3) - independent of
the number of test points after one factorisation, which is the practical advantage
over an O(M^3) kernel GP when M >> N.

Numerics
--------
Everything is done in the *whitened* form.  With U = Phi Lam^{1/2} and
G = I + U^T U / noise (eigenvalues >= 1, always well conditioned):

    Sigma_w = Lam^{1/2} G^{-1} Lam^{1/2}
    mu      = Lam^{1/2} z,          z = G^{-1} (U^T y / noise)
    -2 log p(y) = y^T y / noise - u^T z + log|G| + M log(2 pi noise),  u = U^T y / noise

so the reciprocal eigenvalues 1/lambda_n never appear and the model survives eigenvalues
that underflow to zero (very smooth priors), where the naive
(A = Lam^{-1} + Phi^T Phi / noise) form produces inf/NaN.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .basis import jacobi_deriv_orthonormal, jacobi_orthonormal
from .spectra import log_eigenvalues


# --------------------------------------------------------------------------- #
#  constrained <-> unconstrained reparametrisation
# --------------------------------------------------------------------------- #
def _inv_softplus(y) -> torch.Tensor:
    y = torch.as_tensor(float(y), dtype=torch.float64)
    return torch.log(torch.expm1(y))


class GPParams(torch.nn.Module):
    """Constrained hyper-parameters, stored unconstrained.

    alpha, beta : (-1, inf) via -1 + softplus(raw)   <- the admissible half-plane
    lengthscale : ( 0, inf) via  softplus(raw)
    sigma2, noise: (0, inf) via  exp(raw)
    """

    def __init__(self, alpha=0.0, beta=0.0, lengthscale=0.5, sigma2=1.0, noise=1e-2, learn=()):
        super().__init__()
        learn = set(learn)
        self._raw = {}
        for name, val in dict(
            alpha=alpha, beta=beta, lengthscale=lengthscale, sigma2=sigma2, noise=noise
        ).items():
            if name in ("alpha", "beta"):
                raw = _inv_softplus(val + 1.0)
            elif name == "lengthscale":
                raw = _inv_softplus(val)
            else:
                raw = torch.log(torch.as_tensor(float(val), dtype=torch.float64))
            p = torch.nn.Parameter(raw.to(torch.float64))
            p.requires_grad_(name in learn)
            self.register_parameter(f"raw_{name}", p)
            self._raw[name] = p

    @property
    def alpha(self):
        return -1.0 + torch.nn.functional.softplus(self.raw_alpha)

    @property
    def beta(self):
        return -1.0 + torch.nn.functional.softplus(self.raw_beta)

    @property
    def lengthscale(self):
        return torch.nn.functional.softplus(self.raw_lengthscale)

    @property
    def sigma2(self):
        return torch.exp(self.raw_sigma2)

    @property
    def noise(self):
        return torch.exp(self.raw_noise)

    def as_dict(self) -> dict:
        return {
            "alpha": float(self.alpha.detach()),
            "beta": float(self.beta.detach()),
            "lengthscale": float(self.lengthscale.detach()),
            "sigma2": float(self.sigma2.detach()),
            "noise": float(self.noise.detach()),
        }

    def trainable_names(self):
        return [n for n, p in self._raw.items() if p.requires_grad]


@dataclass
class Posterior:
    """Whitened posterior over the basis weights."""

    mu: torch.Tensor        # (N,) posterior mean of w
    chol: torch.Tensor      # (N,N) lower Cholesky of G = I + Lam^{1/2}Phi^TPhi Lam^{1/2}/noise
    sqrt_lam: torch.Tensor  # (N,)
    lam: torch.Tensor       # (N,)


# --------------------------------------------------------------------------- #
class JacobiGP(torch.nn.Module):
    def __init__(
        self,
        N: int = 40,
        spectrum: str = "se",
        alpha: float = 0.0,
        beta: float = 0.0,
        lengthscale: float = 0.5,
        sigma2: float = 1.0,
        noise: float = 1e-2,
        nu: float = 2.5,
        learn=("lengthscale", "sigma2", "noise"),
        dtype=torch.float64,
    ):
        super().__init__()
        self.N = int(N)
        self.spectrum = spectrum
        self.nu = float(nu)
        self.dtype = dtype
        self.params = GPParams(alpha, beta, lengthscale, sigma2, noise, learn=learn)

    # -- introspection -------------------------------------------------------
    @property
    def alpha(self):
        return self.params.alpha

    @property
    def beta(self):
        return self.params.beta

    def hyperparameters(self) -> dict:
        d = self.params.as_dict()
        d.update(N=self.N, spectrum=self.spectrum, nu=self.nu)
        return d

    # -- building blocks -----------------------------------------------------
    def basis(self, x: torch.Tensor, deriv: int = 0) -> torch.Tensor:
        """Design matrix (M, N); deriv = k gives the analytic d^k/dx^k (shifted Jacobi family)."""
        x = torch.as_tensor(x, dtype=self.dtype).reshape(-1)
        if deriv == 0:
            return jacobi_orthonormal(x, self.N, self.alpha, self.beta)
        return jacobi_deriv_orthonormal(x, self.N, self.alpha, self.beta, order=deriv)

    def log_eigenvalues(self) -> torch.Tensor:
        p = self.params
        return log_eigenvalues(
            self.spectrum, self.N, p.lengthscale, p.sigma2, self.alpha, self.beta, nu=self.nu
        )

    def sqrt_eigenvalues(self) -> torch.Tensor:
        """Lam^{1/2} = exp(log lam / 2).

        Deliberately *not* eigenvalues().sqrt(): Autograd would need d sqrt(x) = 1/2sqrt(x)
        at x = 0, which is inf, and 0 * inf = NaN kills the Adam update as soon as some
        lambda_n underflows (a very smooth prior).
        """
        return torch.exp(0.5 * self.log_eigenvalues())

    def eigenvalues(self) -> torch.Tensor:
        return self.sqrt_eigenvalues() ** 2

    def kernel(self, x1, x2=None, deriv1: int = 0, deriv2: int = 0) -> torch.Tensor:
        """k(x,x') = sum_n lam_n \\hat P_n(x) \\hat P_n(x') (optionally differentiated)."""
        Phi1 = self.basis(x1, deriv=deriv1)
        Phi2 = self.basis(x2, deriv=deriv2) if x2 is not None else Phi1
        return (Phi1 * self.eigenvalues()) @ Phi2.T

    # -- inference -----------------------------------------------------------
    def _whitened(self, Psi, y, noise=None):
        """Whitened solve for observations y = Psi w + eps (Psi: (K,N) design matrix).

        `noise` may be omitted (uses params.noise), be a scalar variance, or be a (K,) **row
        vector of variances** - which is what a physics-informed solve needs, since PDE-
        residual rows and boundary rows have completely different scales.

        Returns (lam, sqrt_lam, noise, L, u, z).
        """
        sqrt_lam = self.sqrt_eigenvalues()
        lam = sqrt_lam**2
        noise = self.params.noise if noise is None else noise
        if torch.is_tensor(noise) and noise.dim() > 0:
            noise = noise.reshape(-1, 1)
        prec = 1.0 / noise                                   # 1/sigma^2, per row or scalar
        U = (Psi * sqrt_lam) * prec.sqrt()                   # (K,N) whitened design
        G = torch.eye(self.N, dtype=self.dtype) + (U.T @ U)
        L = _cholesky_spd(G)
        u = U.T @ (y.reshape(-1) * prec.reshape(-1).sqrt())  # (N,)
        z = _chol_solve(L, u)
        return lam, sqrt_lam, noise, L, u, z

    def posterior(self, X, y, noise=None) -> Posterior:
        """Posterior from point observations y_i = f(x_i) + eps_i."""
        X = torch.as_tensor(X, dtype=self.dtype).reshape(-1)
        y = torch.as_tensor(y, dtype=self.dtype).reshape(-1)
        return self.linear_posterior(self.basis(X), y, noise=noise)

    def linear_posterior(self, Psi: torch.Tensor, y: torch.Tensor, noise=None) -> Posterior:
        r"""Posterior from *arbitrary linear functionals* of the weights,

            y = Psi w + eps,     Psi[k, n] = <observation k, \hat P_n>.

        Point values (Psi = Phi), derivative/collocation rows (Psi = -Phi''(nodes)) and
        boundary rows (Psi = Phi(+-1)) are all the same object here - which is what makes
        a Bayesian physics-informed solve (Experiment 3b) a one-liner.  Pass `noise` as a
        row vector to weight the different kinds of rows differently.
        """
        y = torch.as_tensor(y, dtype=self.dtype).reshape(-1)
        lam, sqrt_lam, _noise, L, u, z = self._whitened(Psi, y, noise)
        return Posterior(mu=sqrt_lam * z, chol=L, sqrt_lam=sqrt_lam, lam=lam)

    def log_marginal_likelihood(self, X, y) -> torch.Tensor:
        X = torch.as_tensor(X, dtype=self.dtype).reshape(-1)
        y = torch.as_tensor(y, dtype=self.dtype).reshape(-1)
        return self.log_marginal_likelihood_operator(self.basis(X), y)

    def log_marginal_likelihood_operator(self, Psi, y, noise=None) -> torch.Tensor:
        y = torch.as_tensor(y, dtype=self.dtype).reshape(-1)
        _lam, _sq, noise_, L, u, z = self._whitened(Psi, y, noise)
        M = y.numel()
        logdet_G = 2.0 * torch.log(torch.diagonal(L)).sum()
        log_det_noise = (noise_.log().sum() if torch.is_tensor(noise_) and noise_.dim() > 0
                         else M * torch.log(noise_))
        # -2 log p = y^T y / noise - u^T z + log|G| + log|noise| + M log(2 pi)   (Woodbury)
        prec = 1.0 / noise_
        nllml = 0.5 * (((y.reshape(-1) ** 2) * prec.reshape(-1)).sum()
                       - (u @ z) + logdet_G + log_det_noise + M * math.log(2.0 * math.pi))
        return -nllml

    # -- exact leave-one-out (a *predictive* model-selection criterion) ------
    def loo(self, X, y):
        """Exact LOO-CV of the induced GP: mean, variance, total log predictive score.

        With A = (K + noise I)^{-1} the leave-one-out predictive law is
        p(y_i | y_{-i}) = N(y_i - [A y]_i / A_ii, 1 / A_ii)   (R&W ch.5.2, eq. 5.12-5.13).
        Cost O(M^3) once, and it is differentiable in (alpha, beta), so it
        can drive the same optimiser as the evidence.  Unlike the evidence it is a
        *predictive* score: a prior that shrinks the amplitude to fit the training points
        cannot inflate it.
        """
        X = torch.as_tensor(X, dtype=self.dtype).reshape(-1)
        y = torch.as_tensor(y, dtype=self.dtype).reshape(-1)
        M = y.numel()
        K = self.kernel(X) + self.params.noise * torch.eye(M, dtype=self.dtype)
        L = _cholesky_spd(K, jitter0=0.0)
        A = torch.cholesky_inverse(L)
        d = torch.diagonal(A)
        mu = y - (A @ y) / d          # note the y_i - ... form, not [Ay]_i/A_ii
        var = 1.0 / d
        logscore = -0.5 * (torch.log(2.0 * math.pi * var) + (y - mu) ** 2 / var).sum()
        return mu, var, logscore

    def fit(self, X, y, steps=200, lr=0.05, method="adam", verbose=False, log_every=None,
            objective: str = "lml", penalty=None):
        """Maximise the log marginal likelihood (objective='loo' for LOO-CV) of the trainable hp.

        `penalty(params) -> tensor` turns the criterion into a MAP objective; needed for
        (alpha, beta), whose admissible set {alpha, beta > -1} is not compact and whose
        evidence is *increasing* towards the singular corner (see Experiment 4).
        """
        trainable = [p for p in self.parameters() if p.requires_grad]
        if not trainable:
            return []
        if method == "adam":
            opt = torch.optim.Adam(trainable, lr=lr)
        elif method == "lbfgs":
            opt = torch.optim.LBFGS(trainable, lr=lr, max_iter=steps, line_search_fn="strong_wolfe")
        else:
            raise ValueError(method)

        history = []

        def obj():
            base = (self.log_marginal_likelihood(X, y) if objective == "lml"
                    else self.loo(X, y)[2] if objective == "loo" else None)
            if base is None:
                raise ValueError(objective)
            return base if penalty is None else base - penalty(self.params)

        def closure():
            opt.zero_grad()
            loss = -obj()
            loss.backward()
            return loss

        n_iter = 1 if method == "lbfgs" else steps
        for it in range(n_iter):
            history.append(-float(opt.step(closure).detach()))
            if verbose and (log_every or max(1, n_iter // 10)) and it % (log_every or max(1, n_iter // 10)) == 0:
                print(f"  [{it:4d}] lml={history[-1]: .4f}  {self.params.as_dict()}")
        return history

    # -- prediction ----------------------------------------------------------
    def predict(self, Xs, post: Posterior = None, deriv: int = 0, noise: bool = False,
                full_cov: bool = False):
        """(mean, marginal variance[, covariance]) of the latent function (or its d^deriv)."""
        Xs = torch.as_tensor(Xs, dtype=self.dtype).reshape(-1)
        return self.predict_operator(self.basis(Xs, deriv=deriv), post, noise, full_cov)

    def predict_operator(self, Psi: torch.Tensor, post: Posterior = None, noise: bool = False,
                         full_cov: bool = False):
        """Predict any linear functional y* = Psi* w (point values, derivatives, integrals)."""
        lam = self.eigenvalues()
        if post is None:                                       # prior
            mean = torch.zeros(Psi.shape[0], dtype=self.dtype)
            cov = (Psi * lam) @ Psi.T
            var = (Psi**2 * lam).sum(-1)
        else:                                                  # posterior
            U = Psi * post.sqrt_lam
            mean = Psi @ post.mu
            Z = torch.linalg.solve_triangular(post.chol, U.T, upper=False)   # (N, K*)
            cov = Z.T @ Z
            var = (Z**2).sum(0)

        if noise and post is not None:
            var = var + self.params.noise
            if full_cov:
                cov = cov + self.params.noise * torch.eye(cov.shape[0], dtype=self.dtype)
        return mean, var, (cov if full_cov else None)

    def sample(self, Xs, num: int = 1, post: Posterior = None, deriv: int = 0, generator=None):
        """Prior (post=None) or posterior function samples, shape (num, len(Xs))."""
        Xs = torch.as_tensor(Xs, dtype=self.dtype).reshape(-1)
        Phi = self.basis(Xs, deriv=deriv)
        eps = torch.randn(num, self.N, dtype=self.dtype, generator=generator)
        if post is None:
            return (eps * self.sqrt_eigenvalues()) @ Phi.T
        # w = mu + Lam^{1/2} L^{-T} eps
        winv = torch.linalg.solve_triangular(post.chol, eps.T, upper=False, transpose=True).T * post.sqrt_lam
        return (post.mu + winv) @ Phi.T


# --------------------------------------------------------------------------- #
def _chol_solve(L: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Solve (L L^T) x = b."""
    # L is *lower* triangular; the transpose solve must use transpose=..., not L.T
    squeeze = b.dim() == 1
    if squeeze:
        b = b.unsqueeze(-1)
    out = torch.cholesky_solve(b, L)          # solves (L L^T) x = b
    return out.squeeze(-1) if squeeze else out


def _cholesky_spd(A: torch.Tensor, jitter0: float = 1e-14):
    """Cholesky with escalating jitter (basis matrices get ill-conditioned for large N).

    The failure mode to expect here is *not* a bug in the recurrence: as (alpha, beta)
    approach the singular corner alpha + beta = -2 of the admissible half-plane the design
    matrix loses rank, so no amount of jitter rescues it.  The message says so, because a
    bare "not positive definite" is otherwise a very misleading error during a
    hyper-parameter optimisation (see Experiment 4, part 2).
    """
    eye = torch.eye(A.shape[0], dtype=A.dtype, device=A.device)
    jitter = 0.0
    last = None
    for _ in range(14):
        try:
            return torch.linalg.cholesky(A + jitter * eye)
        except RuntimeError as exc:          # torch._C._LinAlgError subclasses RuntimeError
            last = exc
            jitter = max(jitter0, jitter * 10.0)
    try:
        cond = f"{float(torch.linalg.cond(A)):.1e}"
    except Exception:
        cond = "unresolved (the matrix is numerically rank-deficient)"
    raise RuntimeError(
        f"precision matrix is not positive definite (cond={cond}).  Two ways to get here: "
        f"(i) a hyper-parameter search drove (alpha, beta) onto the singular boundary of the "
        f"admissible half-plane (alpha, beta > -1, and the recurrence needs alpha+beta > -2) "
        f"- put a prior on them (MAP) or bound them away from -1; (ii) the row variances of a "
        f"linear-functional (e.g. PDE collocation) system span more than ~1e14, which double "
        f"precision cannot separate."
    ) from last
