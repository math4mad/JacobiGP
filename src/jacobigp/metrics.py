"""Evaluation metrics: RMSE split into interior / boundary, plus evidence."""

from __future__ import annotations

import torch


def rmse(pred, truth, mask=None):
    pred, truth = torch.as_tensor(pred).double(), torch.as_tensor(truth).double()
    if mask is not None:
        pred, truth = pred[mask], truth[mask]
    return float(torch.sqrt(torch.mean((pred - truth) ** 2)))


def boundary_mask(x, tol: float = 1e-2, edge: float = 1.0):
    x = torch.as_tensor(x).double()
    return (x.abs() >= edge - tol) if edge is not None else (x.abs() >= 1.0 - tol)


def interior_mask(x, tol: float = 1e-2):
    return ~boundary_mask(x, tol)


def left_mask(x, tol: float = 1e-2):
    return torch.as_tensor(x).double() <= -1.0 + tol


def right_mask(x, tol: float = 1e-2):
    return torch.as_tensor(x).double() >= 1.0 - tol


def report(x, pred, truth, tol: float = 1e-2) -> dict:
    x = torch.as_tensor(x).double()
    return {
        "rmse": rmse(pred, truth),
        "boundary_rmse": rmse(pred, truth, boundary_mask(x, tol)),
        "interior_rmse": rmse(pred, truth, interior_mask(x, tol)),
        "left_rmse": rmse(pred, truth, left_mask(x, tol)),
        "right_rmse": rmse(pred, truth, right_mask(x, tol)),
    }
