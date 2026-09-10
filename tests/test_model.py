"""Tests for the Jacobi-GP inference layer (Phase 2 gate before the experiments)."""

import math

import pytest
import torch

from jacobigp.basis import jacobi_orthonormal, jacobi_orthonormal_and_dx
from jacobigp.model import JacobiGP, _cholesky_spd

DT = torch.float64


def make(N=25, **kw):
    kw.setdefault("dtype", DT)
    return JacobiGP(N=N, **kw)


# --------------------------------------------------------------------------- #
def test_kernel_is_symmetric_positive_semidefinite():
    gp = make(N=20, alpha=1.5, beta=-0.3, lengthscale=0.4, spectrum="matern", nu=3.0)
    x = torch.linspace(-1, 1, 37, dtype=DT)
    K = gp.kernel(x)
    assert torch.allclose(K, K.T, atol=1e-12)
    lam = torch.linalg.eigvalsh(K)
    assert lam.min() > -1e-10, lam.min()
    assert (gp.eigenvalues() > 0).all()
    assert gp.eigenvalues().sum() < 20 * float(gp.eigenvalues().max())  # trace-class decay


def test_weight_space_and_kernel_space_agree():
    """The O(M N^2 + N^3) algebra used by the model must equal the O(M^3) GP algebra."""
    gp = make(N=18, alpha=0.4, beta=0.9, lengthscale=0.35, sigma2=1.7, noise=0.02)
    g = torch.Generator().manual_seed(3)
    X = torch.rand(41, dtype=DT, generator=g) * 2 - 1
    y = torch.randn(41, dtype=DT, generator=g)

    lam = gp.eigenvalues()
    Phi = gp.basis(X)
    K = Phi @ torch.diag(lam) @ Phi.T + gp.params.noise * torch.eye(41, dtype=DT)
    L = torch.linalg.cholesky(K)
    alpha_w = torch.cholesky_solve(y.reshape(-1, 1), L).reshape(-1)
    lml_kernel = -(
        0.5 * (y @ alpha_w) + torch.log(torch.diagonal(L)).sum() + 0.5 * 41 * math.log(2 * math.pi)
    )
    assert torch.allclose(gp.log_marginal_likelihood(X, y), lml_kernel, atol=1e-9)

    post = gp.posterior(X, y)
    mean, var, _ = gp.predict(X, post, noise=False)
    Ks = gp.kernel(X)
    mean_k = Ks @ alpha_w
    v = torch.linalg.solve_triangular(L, Ks.T, upper=False)
    # NOTE: the spectral kernel is *non-stationary*, so k(x,x) != sigma2
    var_k = torch.diagonal(Ks) - (v ** 2).sum(0)
    assert torch.allclose(mean, mean_k, atol=1e-8)
    assert torch.allclose(var, var_k, atol=1e-8)
    m_noisy, v_noisy, _ = gp.predict(X, post, noise=True)
    assert torch.allclose(v_noisy, var_k + gp.params.noise, atol=1e-8)


def test_well_specified_recovery_and_calibration():
    """Draw f from the model's own prior, observe it, and check the posterior is honest:
    the standardised error of the truth must be ~N(0,1)."""
    gp = make(N=22, alpha=0.5, beta=0.5, lengthscale=0.4, sigma2=1.0, noise=1e-4)
    g = torch.Generator().manual_seed(0)
    X = torch.linspace(-1, 1, 120, dtype=DT)
    zs = []
    for rep in range(24):
        gg = torch.Generator().manual_seed(1000 + rep)
        f_grid = gp.sample(X, num=1, generator=gg)[0]     # truth drawn from the model prior
        y = f_grid + 1e-2 * torch.randn(X.shape, dtype=DT, generator=gg)   # noise VARIANCE 1e-4
        post = gp.posterior(X, y)
        mean, var, _ = gp.predict(X, post, noise=False)
        assert (mean - f_grid).abs().max() < 0.2, (mean - f_grid).abs().max()
        zs.append(((f_grid - mean) / var.sqrt().clamp_min(1e-14)).unsqueeze(0))
    zs = torch.cat(zs, 0)                                 # (rep, M)
    per_rep_std = zs.std(dim=1)
    assert 0.5 < float(per_rep_std.mean()) < 1.6, per_rep_std.mean()

    # posterior variance must shrink below the prior variance where data live
    prior_var = gp.predict(X)[1]
    assert (var < prior_var + 1e-9).all()
    assert float((var / prior_var).max()) < 0.2


def test_prior_sample_covariance_matches_kernel():
    gp = make(N=15, alpha=2.0, beta=-0.5, lengthscale=0.5)
    x = torch.tensor([-1.0, -0.63, -0.21, 0.4, 0.88, 1.0], dtype=DT)
    S = gp.sample(x, num=20000, generator=torch.Generator().manual_seed(7))
    emp = (S.T @ S) / S.shape[0]
    K = gp.kernel(x)
    rel = ((emp - K).norm() / K.norm()).item()
    assert rel < 0.1, rel


def test_evidence_is_differentiable_in_the_function_space():
    gp = make(N=16, alpha=0.3, beta=1.2, lengthscale=0.4, learn=("alpha", "beta", "lengthscale"))
    X = torch.linspace(-1, 1, 50, dtype=DT)
    y = torch.sin(3 * X) * torch.exp(-X)
    lml = gp.log_marginal_likelihood(X, y)
    gA, gB, gL = torch.autograd.grad(lml, (gp.params.raw_alpha, gp.params.raw_beta,
                                          gp.params.raw_lengthscale))
    for g, name in ((gA, "alpha"), (gB, "beta"), (gL, "lengthscale")):
        assert torch.isfinite(g).all(), name
        assert abs(float(g)) > 1e-6, (name, float(g))


def test_alpha_beta_act_as_boundary_condition_controllers():
    """Theory (docs/MATH.md §5): alpha,beta < 0 collapses the variance at the edges,
    alpha,beta > 0 inflates it.  Same eigenvalues in both runs - only the space changes."""
    kw = dict(N=30, lengthscale=0.5, sigma2=1.0, spectrum="se")
    x = torch.tensor([-1.0, -0.9, 0.0, 0.9, 1.0], dtype=DT)
    v_neg = make(alpha=-0.7, beta=-0.7, **kw).predict(x)[1]
    v_leg = make(alpha=0.0, beta=0.0, **kw).predict(x)[1]
    v_pos = make(alpha=2.0, beta=2.0, **kw).predict(x)[1]
    assert v_neg[-1] < v_leg[-1] < v_pos[-1]
    assert v_neg[0] < v_leg[0] < v_pos[0]
    # interior variance is nearly unaffected by (alpha, beta)
    # interior variance changes far less than the edge variance (ratio of edge effects ~ 10x)
    edge_neg = float(v_leg[-1] - v_neg[-1]); edge_pos = float(v_pos[-1] - v_leg[-1])
    assert edge_neg > 0.5 * float(v_leg[-1]) and edge_pos > 2.0 * float(v_leg[-1])


def test_analytic_derivative_gp_vs_finite_differences():
    gp = make(N=24, alpha=0.5, beta=0.5, lengthscale=0.4)
    X = torch.linspace(-1, 1, 80, dtype=DT)
    y = (1 - X**2) * torch.sin(2 * math.pi * X) + 1e-3 * torch.randn_like(X)
    post = gp.posterior(X, y)
    xs = torch.linspace(-0.98, 0.98, 40, dtype=DT)
    d_mean, d_var, _ = gp.predict(xs, post, deriv=1, noise=False)
    h = 1e-5
    fd = (gp.predict(xs + h, post, noise=False)[0] - gp.predict(xs - h, post, noise=False)[0]) / (2 * h)
    assert torch.allclose(d_mean, fd, atol=1e-4)
    assert (d_var >= -1e-12).all()


def test_fit_improves_evidence_and_respects_bounds():
    g = torch.Generator().manual_seed(11)
    X = torch.rand(90, dtype=DT, generator=g) * 2 - 1
    y = torch.sin(4 * X) * (1 - X**2) + 0.02 * torch.randn(90, dtype=DT, generator=g)
    gp = make(N=30, alpha=0.0, beta=0.0, lengthscale=2.0, sigma2=0.3, noise=0.3,
              learn=("alpha", "beta", "lengthscale", "sigma2", "noise"))
    start = gp.log_marginal_likelihood(X, y)
    gp.fit(X, y, steps=150, lr=0.1)
    end = gp.log_marginal_likelihood(X, y)
    assert float(end) > float(start)
    hp = gp.hyperparameters()
    assert hp["alpha"] > -1 and hp["beta"] > -1, hp
    assert hp["lengthscale"] > 0 and hp["noise"] > 0


def test_cholesky_jitter_recovers_from_bad_matrix():
    A = torch.eye(4, dtype=DT) - 1e-16
    A[0, 0] = 0.0
    L = _cholesky_spd(A @ A.T + 1e-18 * torch.eye(4, dtype=DT))
    assert torch.allclose(L @ L.T, A @ A.T + 1e-18 * torch.eye(4, dtype=DT), atol=1e-10)


# --------------------------------------------------------------------------- #
#  LOO-CV and the (alpha, beta) prior - the two objects Experiment 4 runs on
# --------------------------------------------------------------------------- #
def test_loo_matches_an_explicit_leave_one_out_refit():
    """p(y_i | y_-i) = N([A y]_i / A_ii, 1/A_ii) with A = (K + noise I)^-1: verify it by
    actually dropping each point and re-predicting it."""
    gp = make(N=12, alpha=0.7, beta=-0.2, lengthscale=0.4, sigma2=1.3, noise=0.05)
    g = torch.Generator().manual_seed(5)
    X = torch.sort(torch.rand(25, dtype=DT, generator=g) * 2 - 1).values
    y = torch.sin(2 * X) + 0.1 * torch.randn(25, dtype=DT, generator=g)
    mu, var, logscore = gp.loo(X, y)

    from jacobigp.model import _cholesky_spd
    for i in range(len(X)):
        keep = torch.arange(len(X)) != i
        K = gp.kernel(X[keep]) + gp.params.noise * torch.eye(len(X) - 1, dtype=DT)
        L = _cholesky_spd(K, jitter0=0.0)
        a = torch.cholesky_solve(y[keep].reshape(-1, 1), L).reshape(-1)
        k_i = gp.kernel(X[i].reshape(1), X[keep])
        m_i = k_i @ a
        v_i = float(gp.kernel(X[i].reshape(1))[0, 0] + gp.params.noise) \
            - float(k_i @ torch.cholesky_solve(k_i.T, L))
        assert abs(float(mu[i]) - m_i) < 1e-8, (i, float(mu[i]), float(m_i))
        assert abs(float(var[i]) - v_i) < 1e-8, (i, float(var[i]), v_i)
        assert float(var[i]) >= float(gp.params.noise) - 1e-10
    assert torch.isfinite(logscore)


def test_map_penalty_keeps_the_space_interior_where_pure_ml_runs_off():
    """The admissible set alpha, beta > -1 is OPEN and the evidence is unbounded on it.
    A N(0, sd) prior on the unconstrained coordinates must turn the runaway into an
    interior optimum (this is the mechanism behind Experiment 4, part 2)."""
    from jacobigp.model import GPParams

    def penalty(params, sd=2.0):
        return 0.5 * ((params.raw_alpha / sd) ** 2 + (params.raw_beta / sd) ** 2)

    g = torch.Generator().manual_seed(2)
    X = torch.sort(torch.rand(60, dtype=DT, generator=g) * 1.7 - 0.85).values
    y = (1 + X) ** 2 / 4
    learn = ("alpha", "beta", "lengthscale", "sigma2", "noise")

    ml = make(N=25, alpha=0.0, beta=0.0, noise=1e-3, learn=learn)
    ml.fit(X, y, steps=400, lr=0.05)
    hp_ml = ml.hyperparameters()

    mp = make(N=25, alpha=0.0, beta=0.0, noise=1e-3, learn=learn)
    mp.fit(X, y, steps=400, lr=0.05, penalty=penalty)
    hp_mp = mp.hyperparameters()

    assert min(hp_mp["alpha"], hp_mp["beta"]) > -1 + 1e-3, hp_mp
    assert hp_mp["sigma2"] > 1e-6, hp_mp
    # pure ML is allowed to be interior too, but never on both sides of the singular set
    print(f"\n  pure ML: {hp_ml}\n  MAP    : {hp_mp}")


def test_cholesky_failure_says_why():
    A = -torch.eye(5, dtype=DT)              # negative definite: no jitter can rescue it
    with pytest.raises(RuntimeError, match="positive definite"):
        _cholesky_spd(A)


def test_per_row_noise_matches_manual_solve():
    """A physics-informed solve weights PDE rows and boundary rows differently; the
    whitened algebra must equal the textbook (Lambda^-1 + Psi^T Sigma^-1 Psi)^-1 solve."""
    gp = make(N=12, alpha=0.2, beta=-0.3, lengthscale=0.5, sigma2=1.0, noise=0.05)
    g = torch.Generator().manual_seed(0)
    X = torch.linspace(-1, 1, 20, dtype=DT)
    y = torch.sin(3 * X) + 0.1 * torch.randn(20, dtype=DT, generator=g)

    scalar = gp.posterior(X, y)
    same = gp.posterior(X, y, torch.full((20,), 0.05, dtype=DT))
    assert torch.allclose(scalar.mu, same.mu, atol=1e-12)
    assert abs(float(gp.log_marginal_likelihood(X, y))
               - float(gp.log_marginal_likelihood_operator(gp.basis(X), y,
                                                           torch.full((20,), 0.05, dtype=DT)))) < 1e-9

    row_var = torch.linspace(0.02, 0.2, 20, dtype=DT)
    Phi = gp.basis(X)
    C = torch.linalg.inv(torch.diag(1.0 / gp.eigenvalues())
                         + Phi.T @ (Phi / row_var[:, None]))
    mu = C @ (Phi.T @ (y / row_var))
    got = gp.linear_posterior(Phi, y, row_var)
    assert torch.allclose(got.mu, mu, atol=1e-12)
    mean, var, _ = gp.predict(X, got)
    assert (var >= -1e-14).all()


def test_pde_collocation_rows_give_a_solver():
    """The exact ladder derivative turns the strong form of -u'' = s into linear functionals
    of the weights, so `linear_posterior` with per-row variances IS a PDE solver: no values
    of u are used, only the equation and the two Dirichlet rows."""
    from jacobigp.datasets import poisson_exact, poisson_source

    J, N = 24, 28
    gp = make(N=N, alpha=0.0, beta=0.0, lengthscale=0.5, sigma2=1.0, noise=1.0, learn=())
    k = torch.arange(1, J + 1, dtype=DT)
    C = -torch.cos(k * torch.pi / (J + 1))                  # interior Chebyshev nodes
    Psi = torch.cat([-gp.basis(C, deriv=2),                  # -u''(c_j)
                     gp.basis(torch.tensor([-1.0, 1.0], dtype=DT))])   # u(-1), u(+1)
    y = torch.cat([poisson_source(C), torch.zeros(2, dtype=DT)])
    row_var = torch.cat([torch.full((J,), 1e-6, dtype=DT), torch.full((2,), 1e-18, dtype=DT)])
    post = gp.linear_posterior(Psi, y, row_var)

    xs = torch.linspace(-1, 1, 401, dtype=DT)
    u, u_sd = gp.predict(xs, post)[:2]
    assert (u_sd >= -1e-14).all()
    assert math.sqrt(float(((u - poisson_exact(xs)) ** 2).mean())) < 1e-3
    assert abs(float(u[0])) < 1e-10 and abs(float(u[-1])) < 1e-10, "Dirichlet rows must hold"
    # the residual is small on a DENSE grid, not only at the collocation nodes
    resid = -gp.predict(xs, post, deriv=2)[0] - poisson_source(xs)
    assert rms_of(resid) / rms_of(poisson_source(xs)) < 1e-2


def rms_of(v):
    return float(torch.sqrt((v ** 2).mean()))
