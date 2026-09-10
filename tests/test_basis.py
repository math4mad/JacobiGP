"""Unit tests for the orthonormal Jacobi machinery (Phase 2 gate: `Code Weaver` -> `Test Master`)."""

import math

import pytest
import torch
from scipy.special import eval_jacobi, gammaln, roots_jacobi

from jacobigp.basis import (
    jacobi_deriv_orthonormal,
    jacobi_log_norm,
    jacobi_log_norm_classic,
    jacobi_mass,
    jacobi_orthonormal,
    jacobi_orthonormal_and_dx,
    jacobi_weight,
    recurrence_coeffs,
)

DT = torch.float64
PAIRS = [(0.0, 0.0), (-0.5, -0.5), (0.5, 0.5), (2.0, 2.0), (5.0, 0.0), (-0.9, 1.3), (3.0, -0.7)]
PARAMS = [pytest.param(p, id=f"a{p[0]}_b{p[1]}") for p in PAIRS]


def ab(a, b):
    return torch.tensor(a, dtype=DT), torch.tensor(b, dtype=DT)


def classical_reference(x, N, a, b):
    """scipy eval_jacobi / sqrt(h_n) with the textbook Gamma formula."""
    n = torch.arange(N, dtype=DT)
    logh = (
        (a + b + 1) * math.log(2)
        + torch.as_tensor(gammaln(n.numpy() + a + 1))
        + torch.as_tensor(gammaln(n.numpy() + b + 1))
        - torch.as_tensor(gammaln(n.numpy() + 1))
        - torch.as_tensor(gammaln(n.numpy() + a + b + 1))
        - torch.log(2 * n + a + b + 1)
    )
    P = torch.stack([torch.as_tensor(eval_jacobi(k, a, b, x.numpy()), dtype=DT) for k in range(N)], -1)
    return P * torch.exp(-0.5 * logh)


# --------------------------------------------------------------------------- #
def test_matches_scipy_and_is_orthonormal():
    for (a, b) in PAIRS:
        A, B = ab(a, b)
        x = torch.linspace(-1, 1, 11, dtype=DT)
        got = jacobi_orthonormal(x, 8, A, B)
        if a + b > -1:  # textbook h_n is regular here
            ref = classical_reference(x, 8, a, b)
            assert torch.allclose(got, ref, atol=1e-9), (a, b, (got - ref).abs().max())

        nodes, wts, mu0 = roots_jacobi(80, a, b, mu=True)
        G = jacobi_orthonormal(torch.as_tensor(nodes, dtype=DT), 8, A, B)
        gram = G.T @ (G * torch.as_tensor(wts, dtype=DT)[:, None])
        assert torch.allclose(gram, torch.eye(8, dtype=DT), atol=1e-6)


def test_chebyshev_line_is_pole_free():
    """alpha+beta+1 = 0 is where the textbook h_n formula breaks; ours must stay finite."""
    for (a, b) in [(-0.5, -0.5), (-0.8, -0.2), (-0.99, -0.01)]:
        A, B = ab(a, b)
        h0 = torch.exp(jacobi_log_norm(torch.tensor(0), A, B))
        assert math.isclose(float(h0), float(jacobi_mass(A, B)), rel_tol=1e-12)
        x = torch.linspace(-1, 1, 9, dtype=DT)
        assert torch.isfinite(jacobi_orthonormal(x, 10, A, B)).all()
        assert torch.isfinite(jacobi_orthonormal_and_dx(x, 10, A, B, 2)).all()
        # reference value: h_0 = int (1-x)^a (1+x)^b = pi for a=b=-1/2
        if (a, b) == (-0.5, -0.5):
            assert math.isclose(float(h0), math.pi, rel_tol=1e-10)


def test_recurrence_is_satisfied():
    A, B = ab(1.3, -0.6)
    N = 9
    x = torch.linspace(-0.99, 0.99, 25, dtype=DT)
    P = jacobi_orthonormal(x, N + 1, A, B)
    a, b = recurrence_coeffs(N, A, B)
    for n in range(N):
        lhs = x * P[:, n]
        rhs = a[n] * P[:, n + 1] + b[n] * P[:, n] + (a[n - 1] * P[:, n - 1] if n else 0.0)
        assert torch.allclose(lhs, rhs, atol=1e-10)


def test_derivative_ladder_matches_autograd():
    """d^k/dx^k of the basis == the (a+k,b+k) family (the Sobolev-ladder claim)."""
    for (a, b) in PAIRS:
        A, B = ab(a, b)
        x = (torch.rand(7, dtype=DT) * 1.9 - 0.95)
        for order in (1, 2, 3):
            fd_step = 1e-6
            low = (jacobi_orthonormal_and_dx(x + fd_step, 10, A, B, order - 1)
                   - jacobi_orthonormal_and_dx(x - fd_step, 10, A, B, order - 1)) / (2 * fd_step)
            an = jacobi_orthonormal_and_dx(x, 10, A, B, order)
            scale = an.abs().max()
            assert torch.allclose(low[:, order:], an[:, order:], atol=1e-4 * scale)
            assert torch.allclose(an[:, :order], torch.zeros_like(an[:, :order]))
            # autograd through the recurrence must agree with the closed form
            ag = torch.autograd.functional.jacobian(
                lambda t: jacobi_orthonormal(t, 10, A, B), x
            )  # (7,10,7)
            if order == 1:
                d = ag.diagonal(dim1=0, dim2=2).T
                assert torch.allclose(d, an, atol=1e-8 * scale)


def test_gradient_wrt_alpha_beta_is_finite_everywhere():
    """Experiment 4 needs d(Phi)/d(alpha); NaNs on the Chebyshev line would kill it."""
    for (a, b) in PAIRS:
        A, B = ab(a, b)
        A.requires_grad_(True)
        B.requires_grad_(True)
        x = torch.linspace(-1, 1, 13, dtype=DT)
        loss = jacobi_orthonormal(x, 12, A, B).square().sum()
        gA, gB = torch.autograd.grad(loss, (A, B))
        assert torch.isfinite(gA).all() and torch.isfinite(gB).all()
        # the analytic gradient must agree with a central difference in alpha
        h = 1e-6
        with torch.no_grad():
            l_plus = jacobi_orthonormal(x, 12, A + h, B).square().sum()
            l_minus = jacobi_orthonormal(x, 12, A - h, B).square().sum()
        fd = (l_plus - l_minus) / (2 * h)
        assert abs(float(fd)) > 0            # the loss really does depend on alpha
        assert abs(float(gA) - float(fd)) < 1e-5 * max(abs(float(fd)), 1.0)


def test_weight_and_domain_map():
    A, B = ab(0.5, -0.25)
    x = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=DT)
    w = jacobi_weight(x, A, B)
    assert torch.isfinite(w).all()
    assert float(w[0]) > 0 and float(w[-1]) == pytest.approx(0.0, abs=1e-6)


def test_log_norm_vs_classic_agree_when_regular():
    A, B = ab(0.7, 1.1)
    n = torch.arange(1, 12, dtype=DT)
    assert torch.allclose(jacobi_log_norm(n.long(), A, B), jacobi_log_norm_classic(n, A, B),
                          rtol=1e-10, atol=1e-10)
