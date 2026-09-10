"""
Orthonormal Jacobi polynomials  \\hat P_n^{(a,b)}  on [-1, 1].

Everything the GP needs is computed from the *symmetric three-term recurrence* in
double precision, built out of torch primitives only, so that Autograd differentiates
through it with respect to the parameters alpha, beta (and the input x).  This is what
makes "learning the function space from data" (Experiment 4) possible.

Conventions
-----------
weight         w(x) = (1-x)**alpha (1+x)**beta,  alpha, beta > -1
orthonormal    int  hat P_n hat P_m w dx = delta_nm
normalisation  hat P_n = P_n^{(a,b)} / sqrt(h_n),  P_n = scipy.special.eval_jacobi
               (Szego normalisation, P_n^{(a,b)}(1) = binom(n+a, n)),
               h_n = 2^{a+b+1} G(n+a+1)G(n+b+1) / [ n! (2n+a+b+1) G(n+a+b+1) ]

Recurrence (Gautschi, *Orthogonal Polynomials*, 3.2.4-3.2.7):

    hat P_0   = 1/sqrt(mu0),   mu0 = int w
    x hat P_n = a_n hat P_{n+1} + b_n hat P_n + a_{n-1} hat P_{n-1}
    a_n = sqrt(lam_{n+1}),
    lam_n = 4 n (n+A)(n+B)(n+A+B) / ((2n+A+B)^2 (2n+A+B+1)(2n+A+B-1))   (n >= 2)
    lam_1 = 4 (1+A)(1+B) / ((2+A+B)^2 (3+A+B))                            (cancelled form)
    b_0 = (B - A)/(A + B + 2),   b_n = (B^2 - A^2)/((2n+A+B)(2n+A+B+2))

Every quantity (including the norms h_n, obtained by cumulating the recurrence ratios
h_n/h_{n-1} = lam_n) is pole-free on the whole admissible half-plane, in particular at
the Chebyshev line alpha + beta + 1 = 0 where the textbook Gamma ratio breaks down.
"""

from __future__ import annotations

import math

import torch


# --------------------------------------------------------------------------- #
# recurrence coefficients
# --------------------------------------------------------------------------- #
def recurrence_coeffs(N: int, alpha: torch.Tensor, beta: torch.Tensor):
    """a_n = sqrt(lam_{n+1}) (n=0..N-1) and b_n (n=0..N-1) of the orthonormal recurrence."""
    dtype, device = alpha.dtype, alpha.device
    ab = alpha + beta
    if ab <= -2 + 1e-12:  # outside the admissible range
        raise ValueError(f"need alpha+beta > -2 (got {float(ab)})")

    # lam_n = h_n/h_{n-1} (monic) : generic formula valid for n >= 2.  n = 1 is stored
    # separately (the factor (1+a+b) cancels) because the generic branch is 0/0 there when
    # alpha+beta+1 = 0.  It must not even be *evaluated* at n = 1: Autograd would propagate
    # NaN through the discarded entry and poison d/dalpha, d/dbeta.
    lam1 = 4.0 * (1 + alpha) * (1 + beta) / ((2 + ab) ** 2 * (3 + ab))
    if N == 1:
        lam = lam1.reshape(1)
    else:
        n = torch.arange(2, N + 1, dtype=dtype, device=device)
        num = 4.0 * n * (n + alpha) * (n + beta) * (n + ab)
        den = (2 * n + ab) ** 2 * (2 * n + ab + 1) * (2 * n + ab - 1)
        lam = torch.cat([lam1.reshape(1), num / den])

    a = lam.sqrt()[:N]                      # a_n = sqrt(lam_{n+1})  -> a[n]
    n_ge1 = torch.arange(1, N, dtype=dtype, device=device)
    b = torch.cat([
        ((beta - alpha) / (ab + 2)).reshape(1),
        (beta**2 - alpha**2) / ((2 * n_ge1 + ab) * (2 * n_ge1 + ab + 2)),
    ])[:N]
    return a, b


# --------------------------------------------------------------------------- #
# normalisation constants  h_n   (pole-free: built from the recurrence)
# --------------------------------------------------------------------------- #
def jacobi_mass(alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """mu0 = int_{-1}^{1} (1-x)^a (1+x)^b dx."""
    ab = alpha + beta
    if ab <= -2 + 1e-12:
        raise ValueError("need alpha+beta > -2")
    return torch.exp(
        (ab + 1) * math.log(2.0)
        + torch.lgamma(alpha + 1)
        + torch.lgamma(beta + 1)
        - torch.lgamma(ab + 2)
    )


def jacobi_log_norm_table(N: int, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """[log h_0, ..., log h_N] of the classical Jacobi norms, pole-free on alpha,beta > -1.

    h_n = 2^{a+b+1} G(n+a+1) G(n+b+1) / [ (2n+a+b+1) n! G(n+a+b+1) ]  (DLMF 18.3.7),
    with h_0 merged into mu0 = 2^{a+b+1} G(a+1)G(b+1)/G(a+b+2) so that the whole line
    alpha + beta + 1 = 0 (Chebyshev of the first kind and its neighbours) stays finite.
    """
    dtype, device = alpha.dtype, alpha.device
    n = torch.arange(1, N + 1, dtype=dtype, device=device)
    ab = alpha + beta
    log_h = (
        (ab + 1) * math.log(2.0)
        + torch.lgamma(n + alpha + 1)
        + torch.lgamma(n + beta + 1)
        - torch.lgamma(n + 1)
        - torch.log(2 * n + ab + 1)
        - torch.lgamma(n + ab + 1)
    )
    return torch.cat([jacobi_mass(alpha, beta).log().reshape(1), log_h])


def jacobi_log_norm(n, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """log h_n^{(alpha,beta)} of the classical Jacobi polynomials (n integer index)."""
    idx = torch.as_tensor(n)
    flat = idx.reshape(-1).long()
    table = jacobi_log_norm_table(int(flat.max()), alpha, beta)
    return table[flat].reshape(idx.shape if idx.dim() else ())


def jacobi_norm(n, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    return torch.exp(jacobi_log_norm(n, alpha, beta))


def jacobi_log_norm_classic(n, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """Textbook Gamma-ratio expression (hits a pole at alpha+beta+1 = 0; used in tests)."""
    n = torch.as_tensor(n, dtype=alpha.dtype, device=alpha.device)
    return (
        (alpha + beta + 1) * math.log(2.0)
        + torch.lgamma(n + alpha + 1)
        + torch.lgamma(n + beta + 1)
        - torch.lgamma(n + 1)
        - torch.lgamma(n + alpha + beta + 1)
        - torch.log(2 * n + alpha + beta + 1)
    )
# --------------------------------------------------------------------------- #
# basis evaluation
# --------------------------------------------------------------------------- #
def _edge_clamp(x: torch.Tensor) -> torch.Tensor:
    """Polynomials are continuous, so clamping |x| to 1-1e-15 is numerically exact."""
    return x.clamp(-1.0 + 1e-15, 1.0 - 1e-15)


def jacobi_orthonormal(x: torch.Tensor, N: int, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """\\hat P_n^{(alpha,beta)}(x), n = 0..N-1;  x of shape (M,) -> output (M, N).

    Differentiable in x, alpha and beta (pure arithmetic + sqrt/lgamma).
    """
    x = _edge_clamp(torch.as_tensor(x, dtype=alpha.dtype)).reshape(-1)
    a, b = recurrence_coeffs(max(N, 1), alpha, beta)

    out = torch.empty(x.numel(), N, dtype=alpha.dtype, device=alpha.device)
    p_prev = torch.zeros_like(x)
    p = torch.ones_like(x) / torch.sqrt(jacobi_mass(alpha, beta))
    for n in range(N):
        out[:, n] = p
        rest = a[n - 1] * p_prev if n else torch.zeros_like(p)
        p_next = ((x - b[n]) * p - rest) / a[n]
        p_prev, p = p, p_next
    return out


def jacobi_basis(x: torch.Tensor, N: int, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """Convenience: (M, N) design matrix for a flat vector of M points."""
    return jacobi_orthonormal(x, N, alpha, beta)


def jacobi_orthonormal_and_dx(
    x: torch.Tensor, N: int, alpha: torch.Tensor, beta: torch.Tensor, order: int = 1
) -> torch.Tensor:
    """d^order/dx^order of the orthonormal basis, shape (M, N).

    Ladder identity:  d^k P_n^{(a,b)} / dx^k = (n+a+b+1)^{rising k} / 2^k * P_{n-k}^{(a+k,b+k)},
    i.e. the derivative basis *is* the (a+k, b+k) orthonormal Jacobi family, rescaled -
    the derivative of the GP lives in the shifted weighted Sobolev space.
    """
    if order == 0:
        return jacobi_orthonormal(x, N, alpha, beta)

    ak, bk = alpha + order, beta + order
    lower = jacobi_orthonormal(x, max(N - order, 0), ak, bk)  # \\hat P^{(a+k,b+k)}_{0..N-k-1}

    n = torch.arange(order, N, dtype=alpha.dtype, device=alpha.device)
    pre = torch.ones_like(n)
    for j in range(order):                       # rising factorial, sign-safe (no logs)
        pre = pre * (n + alpha + beta + 1 + j) / 2.0
    ratio = torch.exp(
        0.5 * (jacobi_log_norm(n - order, ak, bk) - jacobi_log_norm(n, alpha, beta))
    )
    coef = pre * ratio

    deriv = torch.zeros(x.numel(), N, dtype=alpha.dtype, device=alpha.device)
    if N > order:
        deriv[:, order:] = lower * coef
    return deriv


def jacobi_deriv_orthonormal(x, N, alpha, beta, order: int = 1):
    """Alias of :func:`jacobi_orthonormal_and_dx` with an explicit `order` argument."""
    return jacobi_orthonormal_and_dx(x, N, alpha, beta, order=order)


# --------------------------------------------------------------------------- #
# weighted utilities
# --------------------------------------------------------------------------- #
def jacobi_weight(x: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """w(x) = (1-x)^alpha (1+x)^beta, evaluated safely at the edges."""
    xc = x.clamp(-1.0 + 1e-15, 1.0 - 1e-15)
    return torch.exp(alpha * torch.log1p(-xc) + beta * torch.log1p(xc))


def map_to_reference(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """Affine map [lo, hi] -> [-1, 1]: how a real bounded domain enters the basis."""
    return 2.0 * (x - lo) / (hi - lo) - 1.0
