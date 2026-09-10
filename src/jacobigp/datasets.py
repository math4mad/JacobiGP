"""Synthetic targets used by Experiments 1-4 (all defined on the reference interval)."""

from __future__ import annotations

import math

import torch


def _as_x(x):
    return torch.as_tensor(x, dtype=torch.float64) if not hasattr(x, "float") else x.double()


# --------------------------------------------------------------------------- #
#  targets
# --------------------------------------------------------------------------- #
def f_vanishing(x):
    """Expt 1: zero at both boundaries, strongly oscillatory in the interior."""
    x = _as_x(x)
    return (1.0 - x**2) * torch.sin(4.0 * torch.pi * x)


def f_boundary_layer(x):
    """Expt 1 control: non-zero at BOTH boundaries -> the Dirichlet prior must fail."""
    x = _as_x(x)
    return torch.cos(2.0 * torch.pi * x) + 0.3 * x


def f_interior_oscillation(x):
    x = _as_x(x)
    return torch.exp(-3.0 * x**2) * torch.sin(6.0 * torch.pi * x)


def f_steep_right(x):
    """Expt 4: flat (and ~zero) at x=-1, steep gradient / jump-like near x=+1."""
    x = _as_x(x)
    return ((1.0 + x) / 2.0) ** 2 * 0.5 * (1.0 + torch.tanh(9.0 * (x - 0.55)))


def f_ramp_right(x):
    """Expt 4, *mild* asymmetric case: ((1+x)/2)^2.

    Flat and ~0 at x = -1 (value and slope both vanish), growing with slope 1 at x = +1.
    Unlike f_steep_right there is no jump, so the test error is a meaningful measure of the
    space: the ML pair should satisfy alpha > beta AND improve the extrapolated edges.
    """
    x = _as_x(x)
    return ((1.0 + x) / 2.0) ** 2


def f_ramp_left(x):
    """Exact mirror image of f_ramp_right: f_ramp_left(x) == f_ramp_right(-x)."""
    x = _as_x(x)
    return ((1.0 - x) / 2.0) ** 2


def f_steep_left(x):
    """Expt 4 mirror control: the MIRROR image of f_steep_right.

    Steep / jump-like at x = -1, flat at x = +1.  If the evidence really selects the
    function space, the ML (alpha, beta) of this target must be the mirror image of the
    ML pair of f_steep_right, i.e. beta >> alpha.  This is the falsification test of the
    'alpha controls the RIGHT edge, beta the LEFT edge' reading of w = (1-x)^a (1+x)^b.
    """
    x = _as_x(x)
    return ((1.0 - x) / 2.0) ** 2 * 0.5 * (1.0 + torch.tanh(-9.0 * (x + 0.55)))


def f_abs(x):
    """Mildly singular function (kink at 0): good probe of basis conditioning."""
    return _as_x(x).abs()


# --------------------------------------------------------------------------- #
#  1-D Poisson benchmark:  -u'' = s(x),  u(-1) = u(1) = 0
# --------------------------------------------------------------------------- #
def poisson_exact(x):
    """u(x) = (1-x^2) sin(2 pi x):  u(+-1) = 0."""
    x = _as_x(x)
    return (1.0 - x**2) * torch.sin(2.0 * torch.pi * x)


def poisson_d1(x):
    x = _as_x(x)
    return (-2.0 * x * torch.sin(2.0 * torch.pi * x)
            + 2.0 * torch.pi * (1.0 - x**2) * torch.cos(2.0 * torch.pi * x))


def poisson_source(x):
    """s(x) = -u''(x) for the manufactured solution above (exact)."""
    x = _as_x(x)
    return (2.0 * torch.sin(2.0 * torch.pi * x)
            + 8.0 * torch.pi * x * torch.cos(2.0 * torch.pi * x)
            + 4.0 * torch.pi**2 * (1.0 - x**2) * torch.sin(2.0 * torch.pi * x))


# --------------------------------------------------------------------------- #
#  sampling
# --------------------------------------------------------------------------- #
def make_data(fn, M: int = 60, noise: float = 1e-4, seed: int = 0, mode: str = "uniform",
              device=None, domain=(-1.0, 1.0)):
    """Return (X, y) with X in `domain` (a sub-interval of [-1,1]).

    `noise` is a VARIANCE - the same convention as in the model.
    mode: uniform | cheb | grid | edge
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    if mode == "uniform":
        u = torch.rand(M, generator=g, dtype=torch.float64)
    elif mode == "cheb":
        k = torch.arange(M, dtype=torch.float64)
        u = 0.5 * (1 - torch.cos(torch.pi * (k + 0.5) / M))
    elif mode == "grid":
        u = torch.linspace(0, 1, M, dtype=torch.float64)
    elif mode == "edge":  # Chebyshev-like clustering at both edges
        k = torch.arange(M, dtype=torch.float64)
        u = 0.5 * (1 + torch.sin(torch.pi * (k + 0.5) / M - torch.pi / 2))
    else:
        raise ValueError(mode)
    X = domain[0] + (domain[1] - domain[0]) * u
    eps = torch.randn(M, generator=g, dtype=torch.float64) * math.sqrt(noise)
    y = fn(X) + eps
    if device is not None:
        X, y = X.to(device), y.to(device)
    return X, y


def grid(n: int = 401):
    return torch.linspace(-1.0, 1.0, n, dtype=torch.float64)


def f_edge_support(x):
    """Expt 4: symmetric, SMALL in the middle and NON-ZERO (increasing) at both edges.

    cosh(3x)/cosh(3): f(+-1) = 1, f(0) = 0.10.  The data stop at |x| = 0.85 where f = 0.64
    and still grows, so a prior that collapses at the edge is wrong on both sides.  Along
    the ML diagonal alpha = beta this target must sit ABOVE `vanishing` in alpha + beta:
    the learned space opens up exactly where the data demand amplitude.
    """
    x = _as_x(x)
    return torch.cosh(3.0 * x) / math.cosh(3.0)
