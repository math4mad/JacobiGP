"""Jacobi-basis Gaussian Process (spectral GP) on [-1, 1]."""

from .basis import (
    jacobi_orthonormal,
    jacobi_orthonormal_and_dx,
    jacobi_deriv_orthonormal,
    jacobi_log_norm,
    jacobi_norm,
    recurrence_coeffs,
)
from .spectra import eigenvalues, SPECTRA
from .model import JacobiGP, GPParams, Posterior

__all__ = [
    "jacobi_orthonormal",
    "jacobi_orthonormal_and_dx",
    "jacobi_deriv_orthonormal",
    "jacobi_log_norm",
    "jacobi_norm",
    "recurrence_coeffs",
    "eigenvalues",
    "SPECTRA",
    "JacobiGP",
    "GPParams",
    "Posterior",
]
