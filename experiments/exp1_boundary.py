"""
Experiment 1 - Boundary behaviour validation
============================================
Hypothesis (AGENTS.md): altering (alpha, beta) forces the GP to respect boundary
conditions that are nowhere coded in the model.

Design note (this is the part that mattered): boundary behaviour can only be measured
where the data cannot speak.  Fitting M points that *include* x = +-1 makes every basis
interpolate them, so Runs A-D are indistinguishable (verified: boundary RMSE equal to
4 digits).  The experiment therefore leaves the two boundary layers |x| > 0.85
**unsampled** and scores the model in the extrapolated edges - which is exactly the
situation in PDE / surrogate work, where one cares about the value at the boundary.

Run A  Legendre        (0,   0)   : no boundary preference
Run B  Chebyshev-II    (0.5, 0.5) : w -> 0 at the edges  -> boundary freedom
Run C  Chebyshev-I    (-0.5,-0.5) : w -> inf at the edges -> pulled to 0
Run D  strong Dirichlet (-0.9,-0.9)
Run E  asymmetric     (-0.8, 0.8) : clamped left, free right
+ an RBF-GP reference, which has no boundary model at all.

Targets
  vanishing : (1-x^2) sin(4 pi x)  - zero at both edges (hypothesis-friendly)
  layer     : cos(2 pi x) + 0.3 x  - NON-zero at both edges (falsification test)

Expected: edge error orders as D < C < A < B (vanishing) and *reverses* (layer).
"""

from common import DT, banner, fit_gp, save_csv, savefig, save_json, table

import matplotlib.pyplot as plt
import torch

from jacobigp.baselines import RBFGP
from jacobigp.datasets import f_boundary_layer, f_vanishing, make_data

EDGE = 0.85          # data live in [-EDGE, EDGE]; the layers |x| > EDGE are extrapolated
RUNS = [
    ("A", "Legendre", 0.0, 0.0),
    ("B", "Chebyshev-II", 0.5, 0.5),
    ("C", "Chebyshev-I", -0.5, -0.5),
    ("D", "Dirichlet", -0.9, -0.9),
    ("E", "asymmetric", -0.8, 0.8),
]
TARGETS = {"vanishing": f_vanishing, "layer": f_boundary_layer}


def _masks(xs):
    return dict(edge=xs.abs() > EDGE, interior=xs.abs() <= EDGE - 0.05,
                left=xs < -0.97, right=xs > 0.97)


def _rms(v):
    return float(torch.sqrt((v**2).mean())) if v.numel() else float("nan")


def score(xs, mean, truth, post_sd=None, prior_sd=None):
    """RMSE of the fit, split into the extrapolated edges and the sampled interior."""
    mk = _masks(xs)
    e = mean - truth
    out = {
        "rmse": _rms(e),
        "edge_rmse": _rms(e[mk["edge"]]),
        "interior_rmse": _rms(e[mk["interior"]]),
        "left_edge_rmse": _rms(e[xs < -0.97]),
        "right_edge_rmse": _rms(e[xs > 0.97]),
    }
    if post_sd is not None:
        out["edge_sd"] = float(post_sd[mk["edge"]].mean())
    if prior_sd is not None:
        at_edge = (xs < -0.97) | (xs > 0.97)
        out["prior_sd_x1"] = float(prior_sd[at_edge].mean())
        out["prior_sd_0"] = float(prior_sd[xs.abs() < 0.02].mean())
    return out


def main():
    banner("Experiment 1 - boundary behaviour of (alpha, beta)")
    xs = torch.linspace(-1, 1, 801, dtype=DT)
    fits, rows = {}, []
    for tname, fn in TARGETS.items():
        fits[tname] = {}
        truth = fn(xs)
        for label, family, a, b in RUNS:
            gp, X, y = fit_gp(fn, a, b, M=40, noise=1e-3, N=30, seed=1, steps=400,
                              mode="uniform", lengthscale=0.5, domain=(-EDGE, EDGE))
            mean, var, _ = gp.predict(xs, gp.posterior(X, y), noise=False)
            pmean, pvar, _ = gp.predict(xs)                    # prior (mechanism)
            m = score(xs, mean, truth, var.sqrt(), pvar.sqrt())
            m["prior_edge_gain"] = m["prior_sd_x1"] / max(m["prior_sd_0"], 1e-12)
            m.update(target=tname, run=label, family=family, alpha=a, beta=b,
                     lml=float(gp.log_marginal_likelihood(X, y)), model="jacobi")
            rows.append(m)
            fits[tname][label] = dict(xs=xs, mean=mean, sd=var.sqrt(), truth=truth,
                                      prior_sd=pvar.sqrt(), X=X, y=y, metrics=m,
                                      hyper=gp.hyperparameters())

        # RBF reference: the same data, a kernel with no notion of a boundary
        Xr, yr = make_data(fn, M=40, noise=1e-3, seed=1, mode="uniform", domain=(-EDGE, EDGE))
        rbf = RBFGP(lengthscale=0.3, sigma2=1.0, noise=1e-3, learn=("lengthscale", "sigma2", "noise"),
                    dtype=DT)
        rbf.fit(Xr, yr, steps=400, lr=0.05)
        mr, vr = rbf.predict(xs, Xr, yr)
        m = score(xs, mr, truth)
        m.update(target=tname, run="R", family="RBF-GP", alpha="-", beta="-",
                 lml=float(rbf.log_marginal_likelihood(Xr, yr)), model="rbf")
        rows.append(m)
        fits[tname]["R"] = dict(xs=xs, mean=mr, sd=vr.sqrt(), truth=truth, X=Xr, y=yr,
                                metrics=m, hyper=rbf.params.as_dict())

    cols = ["run", "family", "edge_rmse", "interior_rmse", "left_edge_rmse",
            "right_edge_rmse", "edge_sd", "prior_sd_x1", "lml"]
    for tname in TARGETS:
        print(f"\n target = {tname}   (data only in [{-EDGE:+g}, {EDGE:+g}], "
              f"N=30 basis, alpha/beta FIXED, other params ML-fitted)")
        table([r for r in rows if r["target"] == tname], cols,
              labels=["run", "family", "EDGE RMSE", "interior RMSE", "left RMSE",
                      "right RMSE", "edge sd", "prior sd@±1", "log ML"])

    V = {r["run"]: r for r in rows if r["target"] == "vanishing"}
    L = {r["run"]: r for r in rows if r["target"] == "layer"}
    order_V = [V[k]["edge_rmse"] for k in ("D", "C", "A", "B")]
    order_L = [L[k]["edge_rmse"] for k in ("B", "A", "C", "D")]
    checks = {
        "edge RMSE decreases monotonically with alpha+beta (D<C<A<B on 'vanishing')":
            all(x < y for x, y in zip(order_V, order_V[1:])),
        "the ORDERING REVERSES on the non-vanishing target (B<A<C<D)":
            all(x < y for x, y in zip(order_L, order_L[1:])),
        "prior sd at x=+-1 is monotone in alpha+beta  (the mechanism)":
            all(V[k]["prior_sd_x1"] < V[j]["prior_sd_x1"] for k, j in (("D", "C"), ("C", "A"), ("A", "B"))),
        "interior fit is almost untouched (effect is a boundary phenomenon)":
            (max(V[k]["interior_rmse"] for k in "ABCD") - min(V[k]["interior_rmse"] for k in "ABCD"))
            < 0.25 * (max(V[k]["edge_rmse"] for k in "ABCD")
                      - min(V[k]["edge_rmse"] for k in "ABCD")),
        # NOTE on conventions: w(x) = (1-x)^alpha (1+x)^beta, so ALPHA acts on the RIGHT
        # edge (x=+1) and BETA on the LEFT edge (x=-1).  AGENTS.md tabulates it the other
        # way round; the numerics follow the weight, and so does this check.
        "asymmetric run E (alpha=-0.8, beta=+0.8): error at x=+1 << error at x=-1":
            V["E"]["right_edge_rmse"] < V["E"]["left_edge_rmse"],
        "RBF reference has a larger edge error than the best Jacobi run":
            V["R"]["edge_rmse"] > min(V[k]["edge_rmse"] for k in "ABCD"),
    }
    print("\n hypothesis checks")
    for k, ok in checks.items():
        print(f"   [{'PASS' if ok else 'FAIL'}] {k}")

    save_json("exp1_boundary", dict(
        metrics={f"{r['target']}/{r['run']}": {k: v for k, v in r.items()
                                               if k not in ("target", "run")} for r in rows},
        checks=checks, edge=EDGE,
        runs=[dict(zip(("label", "family", "alpha", "beta"), r)) for r in RUNS]))
    save_csv("exp1_boundary", rows, ["target", "run", "model", "family", "alpha", "beta"] + cols[2:])
    plot(fits)


def plot(fits):
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 6.8))
    for col, tname in enumerate(TARGETS):
        ax = axes[0][col]
        for lab, *_ in [(r[0],) for r in RUNS] + [("R",)]:
            f = fits[tname][lab]
            x = f["xs"].numpy()
            ax.plot(x, f["mean"].detach().numpy(),
                    label=lab if lab != "R" else "R (RBF)", lw=1.6 if lab == "R" else 1.2,
                    ls="--" if lab == "R" else "-")
            ax.fill_between(x, (f["mean"] - 2 * f["sd"]).detach().numpy(),
                            (f["mean"] + 2 * f["sd"]).detach().numpy(), alpha=0.10, lw=0)
        ax.plot(x, fits[tname]["A"]["truth"].numpy(), "k--", lw=1.6, label="truth")
        ax.axvspan(-1, -EDGE, color="orange", alpha=0.13, lw=0)
        ax.axvspan(EDGE, 1, color="orange", alpha=0.13, lw=0)
        ax.plot(fits[tname]["A"]["X"].numpy(), fits[tname]["A"]["y"].numpy(), "k.", ms=3)
        ax.set_title(f"target '{tname}'" + ("  - clamped runs win" if col == 0
                                            else "  - clamped runs lose"))
        ax.set_xlabel("x"); ax.set_ylabel("posterior mean")
        ax.legend(fontsize=6.4, loc="upper left", ncol=3)

        ax = axes[1][col]
        for lab, family, a, b in RUNS:
            f = fits[tname][lab]
            ax.plot(x, f["prior_sd"].detach().numpy(), label=f"{lab}: α={a:g}, β={b:g}")
        ax.axvspan(-1, -EDGE, color="orange", alpha=0.13, lw=0)
        ax.axvspan(EDGE, 1, color="orange", alpha=0.13, lw=0)
        ax.set_title(f"prior sd, same eigenvalues - only the SPACE changed ('{tname}')")
        ax.set_xlabel("x"); ax.set_ylabel("prior sd"); ax.legend(fontsize=7)
        if col == 1:
            ax.set_yscale("log")
    fig.tight_layout()
    savefig(fig, "exp1_boundary.png")


if __name__ == "__main__":
    main()
