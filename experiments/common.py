"""Shared helpers for the experiment scripts (Phase 3: `Test Master`)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FIG = ROOT / "figures"
RES = ROOT / "results"
FIG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)

DT = torch.float64

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "axes.grid": True,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 7.5,
        "font.size": 8.5,
        "lines.linewidth": 1.4,
        "axes.prop_cycle": plt.cycler(
            color=["#2b6cb0", "#c05621", "#2f855a", "#805ad5", "#b83280", "#4a5563"]
        ),
    }
)


def fit_gp(target, alpha, beta, M=60, noise=1e-4, seed=0, N=40, spectrum="se",
           steps=400, lr=0.05, learn=("lengthscale", "sigma2", "noise"), lengthscale=0.4,
           mode="uniform", nu=2.5, init_noise=None, domain=(-1.0, 1.0)):
    """Fit a Jacobi GP with (alpha, beta) held fixed (or learnable) and return (gp, X, y)."""
    from jacobigp import JacobiGP
    from jacobigp.datasets import make_data

    X, y = make_data(target, M=M, noise=noise, seed=seed, mode=mode, domain=domain)
    gp = JacobiGP(
        N=N, spectrum=spectrum, alpha=alpha, beta=beta, lengthscale=lengthscale,
        sigma2=1.0, noise=init_noise or max(noise, 1e-6), nu=nu, learn=learn, dtype=DT,
    )
    gp.fit(X, y, steps=steps, lr=lr)
    return gp, X, y


def curve_report(gp, target, X, y, n: int = 801, tol: float = 1e-2):
    """Metrics + the pieces needed to plot one fit."""
    from jacobigp.metrics import report

    xs = torch.linspace(-1, 1, n, dtype=DT)
    mean, var, _ = gp.predict(xs, gp.posterior(X, y), noise=False)
    band = 2.0 * var.sqrt()
    truth = target(xs)
    rep = report(xs, mean, truth, tol)
    rep["nlml"] = float(gp.log_marginal_likelihood(X, y))
    rep["edge_mean"] = float(mean[-1] - mean[0])
    rep["sd_at_pm1"] = float(var.sqrt()[[0, -1]].mean())
    rep["hyper"] = gp.hyperparameters()
    return dict(xs=xs, mean=mean, lo=mean - band, hi=mean + band, truth=truth, metrics=rep)


def save_json(name, payload):
    def default(o):
        if isinstance(o, torch.Tensor):
            return float(o) if o.numel() == 1 else o.tolist()
        if isinstance(o, np.generic):
            return o.item()
        raise TypeError(type(o))

    path = RES / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=default))
    print(f"[saved] {path.relative_to(ROOT)}")
    return path


def save_csv(name, rows, header):
    path = RES / f"{name}.csv"
    with open(path, "w") as fh:
        fh.write(",".join(header) + "\n")
        for r in rows:
            fh.write(",".join(str(r.get(h, "")) for h in header) + "\n")
    print(f"[saved] {path.relative_to(ROOT)}")
    return path


def det(obj):
    """Detach everything that goes into a plot / a saved metric."""
    if torch.is_tensor(obj):
        return obj.detach()
    if isinstance(obj, dict):
        return {k: det(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(det(v) for v in obj)
    return obj


def savefig(fig, name):
    path = FIG / name
    fig.savefig(path)
    plt.close(fig)
    print(f"[saved] {path.relative_to(ROOT)}")
    return path


def banner(text):
    print("\n" + "=" * 78 + f"\n{text}\n" + "=" * 78)


def timer():
    class T:
        def __enter__(self):
            self.t0 = time.time()
            return self

        def __exit__(self, *a):
            print(f"[time] {time.time() - self.t0:.1f}s")

    return T()


def table(rows, cols, labels=None):
    labels = labels or cols
    print("  " + " | ".join(f"{l:>13}" for l in labels))
    for r in rows:
        vals = []
        for c in cols:
            v = r.get(c, "")
            vals.append(f"{v:>13.4g}" if isinstance(v, (int, float)) else f"{str(v):>13}")
        print("  " + " | ".join(vals))
