"""Eigenvalue (spectral density) schedules  lambda_n.

`lengthscale` is used as  n_scale = 1 / lengthscale, i.e. small lengthscale ->
slow decay -> many active polynomial modes.  All schedules are multiplied by
the amplitude sigma2 and are trace class (sum_n lambda_n < infinity).
"""

from __future__ import annotations

import math

import torch

SPECTRA = ("se", "exp", "matern", "sll")


def log_eigenvalues(
    kind: str,
    N: int,
    lengthscale: torch.Tensor,
    sigma2: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    nu: float = 2.5,
) -> torch.Tensor:
    """log lambda_n, n = 0..N-1.  Kept separate so the model can stay stable even when
    lambda_n underflows (extremely smooth priors)."""
    if kind not in SPECTRA:
        raise ValueError(f"unknown spectrum {kind!r}, expected one of {SPECTRA}")
    n = torch.arange(N, dtype=lengthscale.dtype, device=lengthscale.device)
    nscale = 1.0 / lengthscale.clamp_min(1e-6)
    logs = torch.log(sigma2.clamp_min(1e-300))

    if kind == "se":
        return logs - 0.5 * (n / nscale) ** 2
    if kind == "exp":
        return logs - n / nscale
    if kind == "matern":
        return logs - nu * torch.log1p((n / nscale) ** 2)
    mu = n * (n + alpha + beta + 1.0)                       # Jacobi Sturm-Liouville eigenvalues
    return logs - nu * torch.log1p(mu * lengthscale**2 / max(1.0, math.pi**2))


def eigenvalues(
    kind: str,
    N: int,
    lengthscale: torch.Tensor,
    sigma2: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    nu: float = 2.5,
) -> torch.Tensor:
    """lambda_n for n = 0..N-1 (shape (N,))."""
    if kind not in SPECTRA:
        raise ValueError(f"unknown spectrum {kind!r}, expected one of {SPECTRA}")
    n = torch.arange(N, dtype=lengthscale.dtype, device=lengthscale.device)
    nscale = 1.0 / lengthscale.clamp_min(1e-6)

    if kind == "se":
        shape = torch.exp(-0.5 * (n / nscale) ** 2)
    elif kind == "exp":
        shape = torch.exp(-n / nscale)
    elif kind == "matern":
        shape = (1.0 + (n / nscale) ** 2) ** (-nu)
    elif kind == "sll":
        # Sturm-Liouville eigenvalues of the Jacobi operator:
        #   -d/dx[ (1-x^2) w u' ] = mu_n w u,   mu_n = n (n + alpha + beta + 1)
        mu = n * (n + alpha + beta + 1.0)
        shape = (1.0 + (mu * lengthscale**2) / max(1.0, math.pi**2)) ** (-nu)
    return sigma2 * shape
