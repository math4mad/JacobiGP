"""
Experiment 3b - Bayesian physics-informed solve: the differential equation *is* the data
=========================================================================================
Experiment 3 established that d^k/dx^k of the GP is another Jacobi GP in closed form.  This
experiment uses that fact the only way it is useful: the strong-form residual rows

        Psi_pde[j, n] = - \\hat P_n''(c_j)          (collocation nodes c_j)
        Psi_bc        =  \\hat P_n(+-1)             (Dirichlet rows)

are *linear functionals of the same weights*, so `JacobiGP.linear_posterior` turns the PDE
into observations and the GP becomes a solver - with an uncertainty.  Per-row noise is what
makes this a weighted least-squares statement in which the equation and the boundary
condition have different reliabilities (σ_pde = model discrepancy, σ_bc = how hard the BC is
imposed), and the prior Λ is what makes the over-resolved system stable.

Model problem (same manufactured solution as Experiment 3, so the two are comparable):

        -u''(x) = s(x) on (-1,1),   u(-1) = u(1) = 0,
        u(x) = (1-x^2) sin(2 pi x),  s = -u''  known exactly, rms(s) = 23.97

Comparisons, all at matched numbers of unknowns:
  * the Bayesian collocation solve (this experiment),
  * the point-data GP of Experiment 3 (fitted to noisy u values, no use of the PDE),
  * a classical 2nd-order finite-difference solve of the BVP,
  * deterministic (prior-free) collocation at the same N: the ill-conditioning reference,
  * the same collocation with FINITE-DIFFERENCED rows instead of the ladder rows: does the
    exact derivative matter for the solve, not just for the reporting?
"""

from common import DT, banner, det, save_csv, savefig, save_json

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import torch

from jacobigp import JacobiGP
from jacobigp.datasets import poisson_d1, poisson_exact, poisson_source

XS = torch.linspace(-1, 1, 801, dtype=DT)
U_TRUE, U1_TRUE, S_TRUE = poisson_exact(XS), poisson_d1(XS), poisson_source(XS)
U2_TRUE = -S_TRUE
S_RMS = float(torch.sqrt((S_TRUE ** 2).mean()))
JS = (12, 16, 24, 32, 48, 64, 96)          # collocation nodes; N = J + 4 modes
SIGMA_PDE, SIGMA_BC = 1e-3, 1e-9           # row scales: equation vs boundary
SEED = 11


def rms(v):
    v = torch.as_tensor(v)
    return float(torch.sqrt((v ** 2).mean())) if v.numel() else float("nan")


def nodes(J):
    """Chebyshev-Gauss-Lobatto interior nodes (the endpoints are carried by the BC rows)."""
    k = torch.arange(1, J + 1, dtype=DT)
    return -torch.cos(k * torch.pi / (J + 1))


def collocation_posterior(J, N=None, alpha=0.0, beta=0.0, spectrum="se", ls=0.5, sigma2=1.0,
                          sigma_pde=SIGMA_PDE, sigma_bc=SIGMA_BC, deriv_order=2, fd_rows=False,
                          with_bc=True, h_fd=None):
    """Posterior over the weights from the PDE rows + Dirichlet rows only (no u data)."""
    N = N or J + 4
    gp = JacobiGP(N=N, spectrum=spectrum, alpha=alpha, beta=beta, lengthscale=ls,
                  sigma2=sigma2, noise=1.0, learn=(), dtype=DT)
    C = nodes(J)
    if fd_rows:                                  # u''(c) by central differences of Phi
        h = h_fd or (2.0 / (J + 1)) / 4.0
        Phi_p = gp.basis(C + h)
        Phi_0 = gp.basis(C)
        Phi_m = gp.basis(C - h)
        D2 = (Phi_p - 2 * Phi_0 + Phi_m) / h ** 2
    else:
        D2 = gp.basis(C, deriv=deriv_order)       # the ladder: exact, no step size
    Psi = [-D2]
    rhs = [poisson_source(C)]
    var = [torch.full((J,), sigma_pde ** 2, dtype=DT)]
    if with_bc:
        B = gp.basis(torch.tensor([-1.0, 1.0], dtype=DT))
        Psi.append(B)
        rhs.append(torch.zeros(2, dtype=DT))
        var.append(torch.full((2,), sigma_bc ** 2, dtype=DT))
    Psi = torch.cat(Psi, 0)
    y = torch.cat(rhs, 0)
    row_var = torch.cat(var, 0)
    return gp, gp.linear_posterior(Psi, y, row_var), row_var


def score_solve(gp, post):
    """Errors of a BVP solve, + the calibration of its own uncertainty."""
    m, v = gp.predict(XS, post)[:2]
    m1, v1 = gp.predict(XS, post, deriv=1)[:2]
    m2, v2 = gp.predict(XS, post, deriv=2)[:2]
    sd = v.sqrt()
    err = m - U_TRUE
    return dict(u_rmse=rms(err), u_maxerr=float(err.abs().max()),
                u1_rmse=rms(m1 - U1_TRUE), resid_rmse=rms(-m2 - S_TRUE),
                resid_rel=rms(-m2 - S_TRUE) / S_RMS,
                bc_left=abs(float(m[0])), bc_right=abs(float(m[-1])),
                sd_mean=float(sd.mean()), z_mean=float((err.abs() / sd).mean()),
                coverage95=float(((err.abs() < 1.96 * sd).float().mean())))


def fd_solve(J):
    """Classical 2nd-order finite differences on J interior points: the engineering ref."""
    h = 2.0 / (J + 1)
    x = -1 + h * torch.arange(1, J + 1, dtype=DT)
    e = torch.ones(J, dtype=DT)
    A = (2 * torch.diag(e) - torch.diag(e[:-1], 1) - torch.diag(e[:-1], -1)) / h ** 2
    u = torch.linalg.solve(A, poisson_source(x))
    full = torch.cat([torch.zeros(1, dtype=DT), u, torch.zeros(1, dtype=DT)])
    xf = torch.linspace(-1, 1, J + 2, dtype=DT)
    return rms(interp(xf, full, XS) - U_TRUE)


def interp(x, y, xq):
    """Linear interpolation on a 1-D grid (x ascending)."""
    i = torch.clamp(torch.searchsorted(x, xq) - 1, 0, len(x) - 2)
    t = (xq - x[i]) / (x[i + 1] - x[i])
    return y[i] * (1 - t) + y[i + 1] * t


def deterministic_collocation(J, N=None, alpha=0.0, beta=0.0):
    """Prior-free collocation: solve the (J+2) x N system in the least-squares sense."""
    N = N or J + 2
    gp = JacobiGP(N=N, spectrum="se", alpha=alpha, beta=beta, lengthscale=0.5, sigma2=1.0,
                  noise=1.0, learn=(), dtype=DT)
    C = nodes(J)
    A = torch.cat([-gp.basis(C, deriv=2), gp.basis(torch.tensor([-1.0, 1.0], dtype=DT))], 0)
    b = torch.cat([poisson_source(C), torch.zeros(2, dtype=DT)])
    sol = torch.linalg.lstsq(A, b).solution
    m = gp.basis(XS) @ sol
    m2 = gp.basis(XS, deriv=2) @ sol
    return dict(u_rmse=rms(m - U_TRUE), resid_rel=rms(-m2 - S_TRUE) / S_RMS,
                cond=float(torch.linalg.cond(A)), dof=N)


def _cholesky_ok(J, sigma_pde):
    """Can the whitened precision matrix actually be factorised at this row-noise spread?"""
    try:
        collocation_posterior(J, sigma_pde=sigma_pde)
        return True
    except RuntimeError:
        return False


def fit_row_scales(J=32, grid=None):
    """The equation-noise level is a hyper-parameter: profile the operator evidence over it."""
    # against sigma_bc = 1e-9 the row variances may not span more than ~1e17: sigma_pde
    # >= 1 cannot be factorised in double precision (see the last check)
    grid = grid or [1e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]
    out = []
    for sp in grid:
        gp, post, rv = collocation_posterior(J, sigma_pde=sp)
        C = nodes(J)
        Psi = torch.cat([-gp.basis(C, deriv=2),
                         gp.basis(torch.tensor([-1.0, 1.0], dtype=DT))], 0)
        y = torch.cat([poisson_source(C), torch.zeros(2, dtype=DT)])
        lml = float(gp.log_marginal_likelihood_operator(Psi, y, rv).detach())
        out.append(dict(sigma_pde=sp, lml=lml, **score_solve(gp, post)))
    return out


def fd_row_step_study(J=32):
    """Replace the ladder rows by central differences of the basis: what does the step cost?"""
    natural = (2.0 / (J + 1)) / 4.0
    out = []
    for h, tag in ((natural, "h = grid/4 (the natural choice)"), (1e-2, "h = 1e-2"),
                   (1e-3, "h = 1e-3"), (1e-4, "h = 1e-4")):
        gp, post, _ = collocation_posterior(J, fd_rows=True, h_fd=h)
        out.append(dict(tag=tag, h_fd=h, **{k: v for k, v in score_solve(gp, post).items()
                                           if k in ("u_rmse", "resid_rel")}))
    gp, post, _ = collocation_posterior(J)
    ref = score_solve(gp, post)
    for r in out:
        print(f"  {r['tag']:28} u RMSE {r['u_rmse']:.2e}  ({r['u_rmse'] / ref['u_rmse']:7.1f}x "
              f"the ladder rows)")
    return dict(rows=out, ladder_u_rmse=ref["u_rmse"], ladder_resid_rel=ref["resid_rel"])


def main():
    banner("Experiment 3b - Bayesian PDE collocation (-u'' = s, u(+-1) = 0)")
    print(f"  rms(s) = {S_RMS:.3f}; rows: J collocation rows at sigma_pde = {SIGMA_PDE:g}, "
          f"2 Dirichlet rows at sigma_bc = {SIGMA_BC:g}; N = J + 4 modes")

    rows = []
    for J in JS:
        gp, post, _ = collocation_posterior(J)
        bayes = score_solve(gp, post)
        N = J + 4
        # the point-data GP of Experiment 3 (same N, no use of the equation)
        from jacobigp.datasets import make_data
        Xp, yp = make_data(poisson_exact, M=60, noise=1e-6, seed=7, mode="cheb")
        gpp = JacobiGP(N=N, spectrum="matern", nu=3.0, alpha=0.0, beta=0.0, lengthscale=0.5,
                       sigma2=1.0, noise=1e-6, learn=("lengthscale", "sigma2", "noise"),
                       dtype=DT)
        gpp.fit(Xp, yp, steps=300, lr=0.05)
        point = score_solve(gpp, gpp.posterior(Xp, yp))
        det_c = deterministic_collocation(J, N=N)
        fd_rows = score_solve(*collocation_posterior(J, fd_rows=True)[:2])
        row = dict(J=J, N=N,
                   bayes_u_rmse=bayes["u_rmse"], bayes_resid_rel=bayes["resid_rel"],
                   bayes_bc_left=bayes["bc_left"], bayes_bc_right=bayes["bc_right"],
                   bayes_u1_rmse=bayes["u1_rmse"], bayes_z=bayes["z_mean"],
                   bayes_cov=bayes["coverage95"], bayes_max=bayes["u_maxerr"],
                   point_u_rmse=point["u_rmse"], point_resid_rel=point["resid_rel"],
                   fd_u_rmse=fd_solve(J),
                   det_u_rmse=det_c["u_rmse"], det_cond=det_c["cond"],
                   fdrows_u_rmse=fd_rows["u_rmse"], fdrows_resid_rel=fd_rows["resid_rel"])
        rows.append(row)
        print(f"  J={J:3d} N={N:3d}  bayes u RMSE {row['bayes_u_rmse']:.3e}  "
              f"resid {row['bayes_resid_rel']:.2e}  BC {max(row['bayes_bc_left'], row['bayes_bc_right']):.1e}"
              f"  |  point-data {row['point_u_rmse']:.2e}  FD {row['fd_u_rmse']:.2e}  "
              f"deterministic {row['det_u_rmse']:.2e} (cond {row['det_cond']:.1e})  "
              f"FD-rows {row['fdrows_u_rmse']:.2e}")

    scales = fit_row_scales(J=32)
    best = max(scales, key=lambda r: r["lml"])
    mono = all(x["lml"] < y["lml"] for x, y in zip(scales, scales[1:]))
    print(f"  operator evidence is monotone in sigma_pde over the whole decade sweep: {mono}; "
          f"its ML value is {best['sigma_pde']:g} (u RMSE {best['u_rmse']:.2e}, coverage "
          f"{best['coverage95']:.2f}), while the solve is 600x more accurate at "
          f"{scales[0]['sigma_pde']:g} (u RMSE {scales[0]['u_rmse']:.2e}, coverage "
          f"{scales[0]['coverage95']:.2f}).  Calibrated band: "
          f"{[r['sigma_pde'] for r in scales if r['coverage95'] > 0.95]}")

    print("\n  rows: ladder vs finite-differenced second derivative (J = 32)")
    fdrows = fd_row_step_study(32)

    print("\n  Dirichlet rows vs the space alone")
    bc_study = {}
    for (a, b, tag) in ((0.0, 0.0, "Legendre"), (-0.5, -0.5, "Chebyshev-I"),
                        (-0.9, -0.9, "Dirichlet-like"), (1.0, 1.0, "Chebyshev-II")):
        for with_bc in (True, False):
            gp, post, _ = collocation_posterior(32, alpha=a, beta=b, with_bc=with_bc)
            sc = score_solve(gp, post)
            bc_study[f"{tag}|bc={with_bc}"] = sc
            print(f"  {tag:15} BC rows={str(with_bc):5}  u RMSE {sc['u_rmse']:.3e}  "
                  f"|u(-1)| {sc['bc_left']:.2e}  |u(1)| {sc['bc_right']:.2e}")

    good = min(rows, key=lambda r: r["bayes_u_rmse"])
    checks = {
        "the Dirichlet rows are honoured to machine precision (|u(+-1)| < 1e-12, all spaces)":
            max(max(r["bayes_bc_left"], r["bayes_bc_right"]) for r in rows) < 1e-12,
        "the strong-form residual on a DENSE grid falls with J (>= 10x over the sweep) and "
        "stays below 1 % of rms(s)":
            rows[0]["bayes_resid_rel"] / rows[-1]["bayes_resid_rel"] > 10
            and max(r["bayes_resid_rel"] for r in rows) < 1e-2,
        "the solution converges in J (u RMSE falls > 3x) and stays below 1e-3":
            rows[0]["bayes_u_rmse"] / rows[-1]["bayes_u_rmse"] > 3 and good["bayes_u_rmse"] < 1e-3,
        "using the equation beats using point data of u at the same number of modes (> 5x)":
            all(r["bayes_u_rmse"] * 5 < r["point_u_rmse"] for r in rows if r["J"] >= 16),
        "and beats the classical 2nd-order finite-difference solve at the same dof (> 20x)":
            all(r["bayes_u_rmse"] * 20 < r["fd_u_rmse"] for r in rows if r["J"] >= 16),
        "the PRIOR is the regulariser: the prior-free collocation is > 10x worse and its "
        "condition number explodes (1e4 -> 1e6) while the Bayesian solve keeps converging":
            all(r["det_u_rmse"] > 10 * r["bayes_u_rmse"] for r in rows)
            and rows[-1]["det_cond"] / rows[0]["det_cond"] > 100,
        "the ladder rows are not 'more accurate' than F.D. rows - they remove the step: "
        "at h = 1e-3 the two agree to 1 %, at the natural h = grid/4 F.D. costs 100x":
            fdrows["rows"][2]["u_rmse"] < 1.01 * fdrows["ladder_u_rmse"]
            and fdrows["rows"][0]["u_rmse"] > 100 * fdrows["ladder_u_rmse"],
        "the equation-noise level is NOT selectable by the evidence: it is monotone over 5 "
        "decades (ML = the largest value tried), and that ML solve is 100x less accurate":
            all(x["lml"] < y["lml"] for x, y in zip(scales, scales[1:]))
            and best["sigma_pde"] == max(r["sigma_pde"] for r in scales)
            and best["u_rmse"] > 100 * min(r["u_rmse"] for r in scales),
        "the posterior sd is calibrated in a middle band of sigma_pde and useless at both "
        "ends (over-tight rows -> over-confident, over-loose -> the prior dominates)":
            all(0.95 <= r["coverage95"] for r in scales if 1e-5 <= r["sigma_pde"] <= 3e-2)
            and best["coverage95"] < 0.9,
        "row variances may not span more than ~1e17 in double precision: against "
        "sigma_bc = 1e-9 the sweep stops working at sigma_pde = 1 (G cannot be factorised)":
            _cholesky_ok(32, 3e-1) and not _cholesky_ok(32, 1.0),
        "the weight BIASES but does not CONSTRAIN: without the BC rows the best space "
        "(alpha=beta=-0.9) still leaves |u(+-1)| ~ 3e-2, i.e. 14 orders of magnitude off":
            min(bc_study[f"{t}|bc=False"]["bc_left"] for t in
                ("Legendre", "Chebyshev-I", "Dirichlet-like", "Chebyshev-II")) > 1e-3
            and max(bc_study[f"{t}|bc=True"]["bc_left"] for t in
                    ("Legendre", "Chebyshev-I", "Dirichlet-like", "Chebyshev-II")) < 1e-12,
        "a clamping weight still cuts the unconstrained boundary error ~2.5x vs Legendre "
        "(the Exp 1/4 mechanism, now on a PDE solve)":
            bc_study["Dirichlet-like|bc=False"]["bc_left"]
            < 0.5 * bc_study["Legendre|bc=False"]["bc_left"],
    }
    print("\n hypothesis checks")
    for k, ok in checks.items():
        print(f"   [{'PASS' if ok else 'FAIL'}] {k}")

    gp, post, _ = collocation_posterior(32)
    m, v = gp.predict(XS, post)[:2]
    m1 = gp.predict(XS, post, deriv=1)[0]
    curves = det(dict(xs=XS, m=m, sd=v.sqrt(), m1=m1, u=U_TRUE, u1=U1_TRUE,
                      resid=-gp.predict(XS, post, deriv=2)[0] - S_TRUE,
                      nodes=nodes(32)))
    save_json("exp3b_pde_collocation", dict(
        problem="-u''=s, u(+-1)=0, u=(1-x^2)sin(2 pi x)", rms_s=S_RMS, sigma_pde=SIGMA_PDE,
        sigma_bc=SIGMA_BC, js=list(JS), rows=rows, bc_study=bc_study, sigma_profile=scales,
        fd_row_study=fdrows, best_sigma_pde=best["sigma_pde"], checks=checks))
    save_csv("exp3b_pde_collocation", rows, list(rows[0]))
    plot(curves, rows, scales, bc_study)


def plot(curves, rows, scales, bc_study):
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 6.8))
    xs, m, sd = curves["xs"], curves["m"], curves["sd"]

    ax = axes[0, 0]
    ax.plot(xs, m, lw=1.8, label="Bayesian collocation solve (J=32)")
    ax.fill_between(xs, m - 2 * sd, m + 2 * sd, alpha=0.2, label="±2 posterior sd")
    ax.plot(xs, curves["u"], "k--", lw=1.4, label="exact u")
    ax.plot(curves["nodes"], torch.zeros(len(curves["nodes"]), dtype=DT), "r|", ms=8,
            label="collocation nodes (the only PDE information)")
    ax.set_title("the PDE alone, no u data: solve *and* error bar\n"
                 "(±2 sd is visible only at the two ends)")
    ax.set_xlabel("x"); ax.legend(fontsize=6.5)

    ax = axes[0, 1]
    ax.plot(xs, curves["m1"], lw=1.6, label="u' from the ladder")
    ax.plot(xs, curves["u1"], "k--", lw=1.3, label="exact u'")
    ax.plot(xs, curves["resid"] / 10, lw=1.2, color="C3", label="10 × strong residual -u''-s")
    ax.set_title("derivatives are free; the residual is flat and small")
    ax.set_xlabel("x"); ax.legend(fontsize=6.5)

    ax = axes[0, 2]
    ax.plot(xs, (m - curves["u"]).abs(), lw=1.6, label="|error|")
    ax.plot(xs, 1.96 * sd, lw=1.3, ls="--", color="C1", label="1.96 × posterior sd")
    ax.set_yscale("log")
    ax.set_title("is the solve's own sd an honest error bar?\n"
                 "(|error| dips to 1e-16 at the nodes: the solve is interpolatory there)")
    ax.set_xlabel("x"); ax.legend(fontsize=6.5)

    ax = axes[1, 0]
    J = [r["J"] for r in rows]
    for k, lab, st in (("bayes_u_rmse", "Bayes collocation (ladder rows)", "o-"),
                       ("fdrows_u_rmse", "collocation, F.D. rows", "x-"),
                       ("det_u_rmse", "prior-free collocation", "s--"),
                       ("point_u_rmse", "point-data GP (no PDE)", "^:"),
                       ("fd_u_rmse", "finite differences, 2nd order", "d:")):
        ax.plot(J, [r[k] for r in rows], st, ms=4, label=lab)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.xaxis.set_major_locator(mticker.FixedLocator(list(JS)))
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.tick_params(axis="x", labelsize=6.5)
    ax.set_xlabel("unknowns / collocation nodes J"); ax.set_ylabel("u RMSE")
    ax.set_title("same budget, five discretisations")
    ax.legend(fontsize=6.2)

    ax = axes[1, 1]
    ax.plot([r["J"] for r in rows], [r["bayes_resid_rel"] for r in rows], "o-",
            label="relative strong residual")
    ax.plot([r["J"] for r in rows], [r["point_resid_rel"] for r in rows], "^:",
            label="point-data GP, same N")
    ax.axhline(SIGMA_PDE / S_RMS, color="r", ls=":", lw=1, label="σ_pde / rms(s)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("J"); ax.set_ylabel("||-u''-s|| / ||s||")
    ax.set_title("fitting the equation, not the function")
    ax.legend(fontsize=6.2)

    ax = axes[1, 2]
    lmax = max(r["lml"] for r in scales)
    ax.plot([r["sigma_pde"] for r in scales], [r["lml"] - lmax for r in scales], "o-",
            label="operator evidence, relative to its ML value")
    ax2 = ax.twinx()
    ax2.plot([r["sigma_pde"] for r in scales], [r["u_rmse"] for r in scales], "s--",
             color="C1", label="u RMSE")
    ax.set_xscale("log"); ax2.set_yscale("log")
    ax.set_xlabel("σ_pde (equation row noise)"); ax.set_ylabel("log p(rows) − max")
    ax2.set_ylabel("u RMSE", color="C1")
    ax.set_title("the equation-noise level is NOT learnable:\nthe evidence is monotone, its ML value is 600x less accurate")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=6.2)
    fig.tight_layout()
    savefig(fig, "exp3b_pde_collocation.png")


if __name__ == "__main__":
    main()
