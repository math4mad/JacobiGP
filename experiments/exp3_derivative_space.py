"""
Experiment 3 - Physics-informed derivative space
================================================
Hypothesis (AGENTS.md): the derivative of the Jacobi GP lives in the space defined by
(alpha+1, beta+1), so differentiation is *exact* - no finite differences, no extra data,
no step size - and the strong form of a differential operator can be evaluated at any
point as a linear functional of the same weights.

Benchmark: 1-D Poisson with Dirichlet data
        -u''(x) = s(x),  u(-1) = u(1) = 0,  u(x) = (1-x^2) sin(2 pi x)
(manufactured: u, u', u'' and s are known exactly; note |u''(+-1)| = 16 pi ~ 50, so the
second derivative is dominated by the two boundary layers).

What is compared, all on the same 60 noiseless Chebyshev observations of u:
  1. the LADDER identity itself: is d^k/dx^k of the basis really inside the
     (alpha+k, beta+k) span?  (machine-precision check, no modelling involved)
  2. the analytic derivative of the fitted GP vs the CENTRAL FINITE DIFFERENCE of the
     very same posterior mean, over a sweep of step sizes h.  Finite differences are
     scored on the interior mask |x| <= 1 - h, i.e. only where their stencil stays in the
     domain - without that mask the comparison is meaningless, because the basis
     evaluation clamps |x| to 1 and the one-sided node then collapses onto the boundary,
     which is what produced the spurious "catastrophic cancellation" in the first run of
     this experiment.
  3. CALIBRATION: the derivative GP comes with an exact posterior variance, a finite
     difference of the mean does not.  Is the reported sd honest?
  4. the PDE residual -u''_GP - s, relative to ||s||.
  5. the effect of the spectral tail (se vs matern vs sll vs exp) on how far one can
     push N before differentiation starts amplifying the neglected modes.
  6. a classical RBF-GP on the same data: it has no derivative theory in this codebase,
     so its u' and u'' can only be finite-differenced and carry no uncertainty at all.
"""

from common import DT, banner, det, save_csv, savefig, save_json

import matplotlib.pyplot as plt
import torch

from jacobigp import JacobiGP
from jacobigp.baselines import RBFGP
from jacobigp.datasets import make_data, poisson_d1, poisson_exact, poisson_source

M, NOISE, SEED = 60, 1e-6, 7
XS = torch.linspace(-1, 1, 601, dtype=DT)
NS = (8, 12, 16, 20, 25, 30, 40, 50)
HS = (1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 1e-4, 1e-5, 1e-6)
REF_N = 25                      # the N used for the step-size / spectrum studies


def fit_jacobi(alpha=0.0, beta=0.0, N=30, ls=0.5, spectrum="matern", nu=3.0):
    X, y = make_data(poisson_exact, M=M, noise=NOISE, seed=SEED, mode="cheb")
    gp = JacobiGP(N=N, spectrum=spectrum, alpha=alpha, beta=beta, lengthscale=ls,
                  sigma2=1.0, noise=NOISE, nu=nu,
                  learn=("lengthscale", "sigma2", "noise"), dtype=DT)
    gp.fit(X, y, steps=300, lr=0.05)
    return gp, X, y


def rms(v):
    return float(torch.sqrt((v**2).mean())) if v.numel() else float("nan")


def fd(f, h, order):
    """Central difference of the posterior *mean function* at offset h."""
    if order == 1:
        return (f(h) - f(-h)) / (2 * h)
    return (f(h) - 2 * f(0.0) + f(-h)) / h**2


def main():
    banner("Experiment 3 - derivative space of the Jacobi GP (1-D Poisson)")
    u_true, u1_true, s_true = poisson_exact(XS), poisson_d1(XS), poisson_source(XS)
    u2_true = -s_true
    s_rms = rms(s_true)
    print(f"  magnitudes: rms(u)={rms(u_true):.3f}  rms(u')={rms(u1_true):.3f}  "
          f"rms(u'')=rms(s)={s_rms:.3f}   (|u''(1)| = {abs(float(u2_true[-1])):.1f})")

    # ---- 1. does d/dx really land in the (alpha+1, beta+1) space? -----------
    gp, X, y = fit_jacobi(N=30)
    a, b = float(gp.alpha), float(gp.beta)
    ladder = {}
    for k in (1, 2):
        Dk = gp.basis(XS, deriv=k)
        ref = JacobiGP(N=30 - k, spectrum=gp.spectrum, alpha=a + k, beta=b + k,
                       lengthscale=float(gp.params.lengthscale), sigma2=1.0, noise=NOISE,
                       nu=gp.nu, dtype=DT)
        P = ref.basis(XS)
        resid = Dk - P @ torch.linalg.lstsq(P, Dk).solution
        ladder[f"d{k}_span_max_resid"] = float(resid.abs().max())
        ladder[f"d{k}_relative_resid"] = ladder[f"d{k}_span_max_resid"] / float(Dk.abs().max())
        print(f"  ladder k={k}: max residual of d^{k}/dx^{k} in "
              f"span(hat P^({a + k:g},{b + k:g})) = {ladder[f'd{k}_span_max_resid']:.2e} "
              f"(relative {ladder[f'd{k}_relative_resid']:.2e})")

    # ---- 2. analytic vs finite differences of the same fit, N sweep ---------
    rows, curves = [], {}
    for N in NS:
        gp, X, y = fit_jacobi(N=N)
        post = gp.posterior(X, y)
        m0, v0 = gp.predict(XS, post, noise=False)[:2]
        m1, v1 = gp.predict(XS, post, deriv=1, noise=False)[:2]
        m2, v2 = gp.predict(XS, post, deriv=2, noise=False)[:2]
        f = lambda hh: gp.predict(XS + hh, post, noise=False)[0]
        row = dict(N=N, alpha=float(gp.alpha), beta=float(gp.beta),
                   lengthscale=float(gp.params.lengthscale), spectrum=gp.spectrum,
                   u_rmse=rms(m0 - u_true),
                   d1_analytic_rmse=rms(m1 - u1_true), d2_analytic_rmse=rms(m2 - u2_true),
                   d2_interior_rmse=rms((m2 - u2_true)[XS.abs() <= 0.9]),
                   d2_edge_rmse=rms((m2 - u2_true)[XS.abs() > 0.9]),
                   ode_residual_rel=rms(-m2 - s_true) / s_rms,
                   d1_sd_mean=float(v1.sqrt().mean()),
                   d1_zscore_mean=float(((m1 - u1_true).abs() / v1.sqrt()).mean()),
                   d1_coverage95=float(((m1 - u1_true).abs() < 1.96 * v1.sqrt()).float().mean()))
        for h in HS:
            mk = XS.abs() <= 1 - h          # the stencil must stay inside [-1, 1]
            row[f"d1_fd_h{h:g}_rmse"] = rms((fd(f, h, 1) - u1_true)[mk])
            row[f"d2_fd_h{h:g}_rmse"] = rms((fd(f, h, 2) - u2_true)[mk])
            row[f"d1_analytic_on_mask_h{h:g}"] = rms((m1 - u1_true)[mk])
        rows.append(row)
        if N == REF_N:
            def inside(v, h):            # a F.D. stencil is only defined where it fits
                return torch.where(XS.abs() <= 1 - h, v, torch.full_like(v, float("nan")))
            curves["ref"] = det(dict(m0=m0, v0=v0, m1=m1, v1=v1, m2=m2, v2=v2,
                                     fd1_h1e1=inside(fd(f, 1e-1, 1), 1e-1),
                                     fd1_h1e4=inside(fd(f, 1e-4, 1), 1e-4),
                                     fd2_h1e4=inside(fd(f, 1e-4, 2), 1e-4)))

    jac = list(rows)          # the Jacobi N-sweep only (the RBF row is appended later)
    best_d1 = min(jac, key=lambda r: r["d1_analytic_rmse"])
    ref = next(r for r in jac if r["N"] == REF_N)

    # ---- 3. spectrum sweep: how far can N be pushed before d/dx amplifies? ---
    spec_rows = []
    for spectrum, nu in (("matern", 3.0), ("matern", 5.0), ("se", 1.0), ("sll", 3.0),
                         ("exp", 1.0)):
        for N in (16, 24, 32, 48):
            gp, X, y = fit_jacobi(N=N, spectrum=spectrum, nu=nu)
            post = gp.posterior(X, y)
            m0 = gp.predict(XS, post, noise=False)[0]
            m2 = gp.predict(XS, post, deriv=2, noise=False)[0]
            spec_rows.append(dict(spectrum=spectrum, nu=nu, N=N,
                                  lengthscale=float(gp.params.lengthscale),
                                  u_rmse=rms(m0 - u_true), d2_analytic_rmse=rms(m2 - u2_true),
                                  ode_residual_rel=rms(-m2 - s_true) / s_rms))
    print("\n spectrum sweep (analytic u'' error vs basis count):")
    print("   " + " | ".join(f"{c:>18}" for c in
                             ("spectrum", "nu", "N", "u_rmse", "d2_analytic_rmse",
                              "ode_residual_rel")))
    for r in spec_rows:
        print("   " + " | ".join(f"{r[c]:>18.3e}" if isinstance(r[c], float) else f"{r[c]:>18}"
                                 for c in ("spectrum", "nu", "N", "u_rmse", "d2_analytic_rmse",
                                           "ode_residual_rel")))

    # ---- 4. RBF-GP reference ------------------------------------------------
    Xr, yr = make_data(poisson_exact, M=M, noise=NOISE, seed=SEED, mode="cheb")
    rbf = RBFGP(lengthscale=0.3, sigma2=1.0, noise=NOISE,
                learn=("lengthscale", "sigma2", "noise"), dtype=DT)
    rbf.fit(Xr, yr, steps=300, lr=0.05)
    rbf_m = rbf.predict(XS, Xr, yr)[0]
    rbf_row = dict(N="RBF-GP", alpha="-", beta="-",
                   lengthscale=float(rbf.params.lengthscale), spectrum="-",
                   u_rmse=rms(rbf_m - u_true), d1_analytic_rmse=float("nan"),
                   d2_analytic_rmse=float("nan"), ode_residual_rel=float("nan"),
                   d1_sd_mean=float("nan"), d1_zscore_mean=float("nan"),
                   d1_coverage95=float("nan"))
    g = lambda hh: rbf.predict(XS + hh, Xr, yr)[0]
    for h in HS:
        mk = XS.abs() <= 1 - h
        rbf_row[f"d1_fd_h{h:g}_rmse"] = rms((fd(g, h, 1) - u1_true)[mk])
        rbf_row[f"d2_fd_h{h:g}_rmse"] = rms((fd(g, h, 2) - u2_true)[mk])
    rbf_has_deriv_sd = rbf.predict(XS, Xr, yr, deriv=1)[1] is not None
    rows.append(rbf_row)

    # ---------------------------------------------------------------- table ---
    cols = (["N", "u_rmse", "d1_analytic_rmse"]
            + [f"d1_fd_h{h:g}_rmse" for h in HS]
            + ["d2_analytic_rmse", "d2_interior_rmse", "d2_edge_rmse", "ode_residual_rel",
               "d1_zscore_mean", "d1_coverage95"])
    print("\n " + " | ".join(f"{c:>16}" for c in cols))
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, float("nan"))
            cells.append(f"{v:>16.3e}" if isinstance(v, float) else f"{str(v):>16}")
        print("  " + " | ".join(cells))

    # ---------------------------------------------------------- hypotheses ---
    fd_curve = [ref[f"d1_fd_h{h:g}_rmse"] for h in HS]
    ana_ref = ref["d1_analytic_on_mask_h1e-05"]
    se_curve = {(r["spectrum"], r["nu"], r["N"]): r["d2_analytic_rmse"] for r in spec_rows}
    checks = {
        "d/dx of the basis lies in the (alpha+1, beta+1) span to machine precision":
            ladder["d1_relative_resid"] < 1e-10,
        "d2/dx2 of the basis lies in the (alpha+2, beta+2) span to machine precision":
            ladder["d2_relative_resid"] < 1e-8,
        "finite differences CONVERGE to the analytic ladder derivative "
        f"(< 10 % apart for h <= 3e-3, N = {REF_N})":
            all(abs(ref[f"d1_fd_h{h:g}_rmse"] - ref[f"d1_analytic_on_mask_h{h:g}"])
                < 0.1 * ref[f"d1_analytic_on_mask_h{h:g}"] for h in HS if h <= 3e-3),
        "a coarse step is the real danger for F.D.: h = 1e-1 costs > 5x over the analytic one":
            fd_curve[0] > 5 * ana_ref,
        "the analytic derivative is step-independent: every F.D. column at h <= 1e-3 sits "
        "within 10 % of it (the ladder result is one evaluation, no h)":
            all(abs(ref[f"d1_fd_h{h:g}_rmse"] - ana_ref) < 0.1 * ana_ref
                for h in HS if h <= 1e-3),
        "the derivative GP is CALIBRATED once the basis resolves u "
        "(95 % interval covers > 0.9 for every N >= 25)":
            all(r["d1_coverage95"] > 0.9 for r in jac if r["N"] >= 25),
        "and it is over-confident when the basis is too small: at N = 8 the 95 % interval "
        "covers ~1 % of the grid (mean z-score ~ 59)":
            jac[0]["d1_coverage95"] < 0.1 and jac[0]["d1_zscore_mean"] > 10,
        f"the analytic u'' error is dominated by the boundary layers "
        f"(edge >> interior at N = {REF_N})":
            ref["d2_edge_rmse"] > 3 * ref["d2_interior_rmse"],
        "relative PDE residual ||-u''_GP - s|| / ||s|| goes below 5 %":
            min(r["ode_residual_rel"] for r in jac) < 0.05,
        "an exponential (se) tail keeps the analytic u'' stable as N grows, while an "
        "algebraic (matern) tail amplifies the unresolved modes by > 10x":
            se_curve[("se", 1.0, 48)] < 1.5 * se_curve[("se", 1.0, 16)]
            and se_curve[("matern", 3.0, 48)] > 10 * se_curve[("matern", 3.0, 16)],
        "the RBF-GP cannot supply a posterior variance for its derivative "
        "(the Jacobi GP supplies mean AND sd in closed form)":
            not rbf_has_deriv_sd,
        "the RBF-GP fits u about as well, so the comparison of derivatives is fair":
            0.3 < rbf_row["u_rmse"] / ref["u_rmse"] < 3.0,
    }
    print("\n hypothesis checks")
    for k, ok in checks.items():
        print(f"   [{'PASS' if ok else 'FAIL'}] {k}")

    save_json("exp3_derivative_space", dict(
        problem="-u''=s, u(+-1)=0, u=(1-x^2)sin(2 pi x)", M=M, noise=NOISE, seed=SEED,
        magnitudes=dict(rms_u=rms(u_true), rms_d1=rms(u1_true), rms_s=s_rms,
                        abs_u2_at_edge=float(abs(u2_true[-1]))),
        ladder=ladder, rows=rows, spectrum_sweep=spec_rows, checks=checks,
        ref_N=REF_N, best_d1_N=best_d1["N"], hs=list(HS),
        rbf_has_derivative_sd=rbf_has_deriv_sd,
    ))
    save_csv("exp3_derivative_space", rows, cols)
    plot(curves, rows, jac, rbf_row, u1_true, u2_true, ref, spec_rows, s_rms)


def plot(curves, rows, jac, rbf_row, u1_true, u2_true, ref, spec_rows, s_rms):
    c = curves["ref"]
    fig, axes = plt.subplots(2, 3, figsize=(13.6, 6.8))

    ax = axes[0, 0]
    ax.plot(XS, c["m0"], label="posterior mean")
    ax.fill_between(XS, c["m0"] - 2 * c["v0"].sqrt(), c["m0"] + 2 * c["v0"].sqrt(), alpha=0.15)
    ax.plot(XS, poisson_exact(XS), "k--", lw=1.6, label="exact u")
    Xd, yd = make_data(poisson_exact, M=M, noise=NOISE, seed=SEED, mode="cheb")
    ax.plot(Xd, yd, "k.", ms=4, label="data")
    ax.set_title(f"u(x): Jacobi-GP fit (N={ref['N']}, α=β=0, RMSE {ref['u_rmse']:.1e})")
    ax.set_xlabel("x"); ax.legend(fontsize=7)

    ax = axes[0, 1]
    ax.plot(XS, u1_true, "k--", lw=1.8, label="exact u'")
    ax.plot(XS, c["m1"], lw=2, label="analytic ladder u'")
    ax.plot(XS, c["fd1_h1e1"], lw=1, alpha=0.9, label="F.D. of the same mean, h=1e-1")
    ax.set_title("u'(x): the ladder derivative and a coarse F.D.")
    ax.set_xlabel("x"); ax.legend(fontsize=6.5)

    ax = axes[0, 2]
    ax.plot(XS, u2_true, "k--", lw=1.8, label="exact u''")
    ax.plot(XS, c["m2"], lw=1.4, label="analytic u'' (α+2, β+2 space)")
    ax.plot(XS, c["fd2_h1e4"], lw=1, alpha=0.85, label="F.D. u'', h=1e-4 (masked to |x|<=1-h)")
    ax.set_title("u''(x): exact, and the F.D. that converges to it")
    ax.set_xlabel("x"); ax.legend(fontsize=6.5)

    ax = axes[1, 0]
    ax.plot(HS, [ref[f"d1_fd_h{h:g}_rmse"] for h in HS], "o-", label="F.D. u' (h masked)")
    ax.axhline(ref["d1_analytic_on_mask_h1e-05"], color="C0", ls="--",
               label="analytic u' (no h)")
    ax.plot(HS, [rbf_row[f"d1_fd_h{h:g}_rmse"] for h in HS], "s:", color="0.4",
            label="RBF-GP: F.D. only")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("step size h"); ax.set_ylabel("RMSE u'")
    ax.set_title("F.D. has a truncation/round-off trade-off;\nthe ladder derivative has none")
    ax.legend(fontsize=6.5)

    ax = axes[1, 1]
    for k, lab, st in (("d1_analytic_rmse", "analytic u'", "o-"),
                       ("d1_fd_h0.01_rmse", "F.D. h=1e-2", "s-"),
                       ("d1_fd_h0.0001_rmse", "F.D. h=1e-4", "^-")):
        ax.plot([r["N"] for r in jac], [r[k] for r in jac], st, ms=4, label=lab)
    ax.axhline(min(rbf_row[f"d1_fd_h{h:g}_rmse"] for h in HS), color="0.4", ls=":",
               label="RBF-GP, best F.D. step")
    ax.set_yscale("log"); ax.set_xlabel("basis functions N"); ax.set_ylabel("RMSE u'")
    ax.set_title("u' error vs N"); ax.legend(fontsize=6.5)

    ax = axes[1, 2]
    for spectrum, nu, st in (("matern", 3.0, "o-"), ("se", 1.0, "s-"), ("sll", 3.0, "^:")):
        rr = [r for r in spec_rows if r["spectrum"] == spectrum and r["nu"] == nu]
        ax.plot([r["N"] for r in rr], [r["d2_analytic_rmse"] for r in rr], st, ms=4,
                label=f"{spectrum}(ν={nu:g})")
    ax.axhline(s_rms, color="k", lw=0.8, ls="--", label="rms(s) - the scale to beat")
    ax.set_yscale("log"); ax.set_xlabel("basis functions N")
    ax.set_ylabel("RMSE of the analytic u''")
    ax.set_title("differentiation amplifies the tail:\nthe spectrum decides how far N can go")
    ax.legend(fontsize=6.5)
    fig.tight_layout()
    savefig(fig, "exp3_derivative_space.png")


if __name__ == "__main__":
    main()
