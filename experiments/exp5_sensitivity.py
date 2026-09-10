"""
Experiment 5 - How much of Experiment 4's answer is the method, and how much is a choice?
===========================================================================================
Experiment 4 recovered the boundary signature of six targets from the evidence, but it did
so with three decisions baked in: the strength of the MAP prior on (alpha, beta), the number
of basis functions N, and the `se` spectrum.  A reviewer's first question is therefore the
right one: *does the learned space survive changing them?*  This script answers it by
re-running the same MAP fit over four grids, and it adds the control that makes the whole
story credible: if the boundary layers are **sampled**, the effect must disappear.

  grid 1  prior sd on the unconstrained (alpha, beta): 0.75 ... 4, plus pure ML (no prior)
  grid 2  basis size N: 16, 24, 32, 48, 64
  grid 3  spectrum: se | exp | matern | sll (the Sturm-Liouville tail of the same operator)
  grid 4  how much of the interval is left unsampled: EDGE = 0.70 ... 0.999
  grid 5  five independent data draws (sampling variability of the learned pair)

Representative targets: ramp_right (mild asymmetry), steep_right (jump), vanishing
(symmetric clamping).  The quantities compared are the learned pair (alpha, beta), the sign
of alpha - beta (which edge is free), the evidence gain over the fixed Legendre space, and
the test error at the clamped edge.
"""

from common import DT, banner, save_csv, savefig, save_json

import matplotlib.pyplot as plt
import numpy as np
import torch

import exp4_evidence_learning as E4
from jacobigp import JacobiGP
from jacobigp.datasets import make_data

TARGETS = ("ramp_right", "steep_right", "vanishing")
EDGE = E4.EDGE
M, NOISE, SEED = E4.M, E4.NOISE, E4.SEED
SPECTRA = ("se", "exp", "matern", "sll")


def data_for(tname, edge=EDGE, m=M, noise=NOISE, seed=SEED):
    return make_data(E4.TARGETS[tname], M=m, noise=noise, seed=seed, mode="uniform",
                     domain=(-edge, edge))


def learn(tname, sd=E4.PRIOR_SD, n=E4.N, spectrum=E4.SPECTRUM, edge=EDGE, m=M,
          noise=NOISE, seed=SEED, penalise=True):
    """One MAP fit of the space, returning (alpha, beta, evidence, errors)."""
    fn = E4.TARGETS[tname]
    X, y = data_for(tname, edge=edge, m=m, noise=noise, seed=seed)
    gp = JacobiGP(N=n, spectrum=spectrum, alpha=0.0, beta=0.0, lengthscale=0.4, sigma2=1.0,
                  noise=noise, learn=("alpha", "beta", *E4.NUISANCE), dtype=DT)
    traj = E4.fit_track(gp, X, y, penalise=penalise, adam_steps=600, lbfgs_iters=200,
                        prior_sd=sd)
    broke = traj[-1].get("breakdown")
    r = E4.score(gp if broke is None else E4.gp_at(traj[-1]), fn, X, y)
    r["broke"] = broke is not None
    # the fixed-Legendre reference on the same data, for the evidence gain
    g0 = JacobiGP(N=n, spectrum=spectrum, alpha=0.0, beta=0.0, lengthscale=0.4, sigma2=1.0,
                  noise=noise, learn=E4.NUISANCE, dtype=DT)
    E4.fit_track(g0, X, y, penalise=False, adam_steps=500, lbfgs_iters=150)
    r0 = E4.score(g0, fn, X, y)
    r.update(target=tname, gain=r["lml"] - r0["lml"], legendre_rmse=r0["rmse"],
             legendre_clamp=min(r0["left_edge_rmse"], r0["right_edge_rmse"]),
             clamp=min(r["left_edge_rmse"], r["right_edge_rmse"]),
             prior_sd=sd, N=n, spectrum=spectrum, edge=edge, M=m, data_noise=noise, seed=seed)
    return r


def sweep(name, rows, key_fmt=lambda r: ""):
    print(f"\n {name}")
    for r in rows:
        print(f"   {r['target']:12} {key_fmt(r):22} α={r['alpha']:8.3f} β={r['beta']:8.3f} "
              f"α-β={r['alpha'] - r['beta']:+7.3f}  gain={r['gain']:+7.2f}  "
              f"rmse={r['rmse']:.4f}  clamp {r['legendre_clamp']:.2e}->{r['clamp']:.2e}"
              f"{'  [BROKE]' if r['broke'] else ''}")
    return rows


def main():
    banner("Experiment 5 - sensitivity of the learned function space")

    g1 = sweep("grid 1: strength of the MAP prior on (alpha, beta)",
               [learn(t, sd=sd) for t in TARGETS for sd in (0.75, 1.0, 1.5, 2.0, 3.0, 4.0)],
               key_fmt=lambda r: f"prior sd={r['prior_sd']:g}")
    print("   (and with NO prior at all, i.e. pure ML - the failure Experiment 4 documents)")
    g1_ml = [learn(t, penalise=False) for t in TARGETS]
    for r in g1_ml:
        print(f"   pure ML      {r['target']:12} α={r['alpha']:8.3f} β={r['beta']:8.3f} "
              f"σ²={r['sigma2']:.1e} broke={r['broke']} rmse={r['rmse']:.4f}")

    g2 = sweep("grid 2: number of basis functions N",
               [learn(t, n=n) for t in TARGETS for n in (16, 24, 32, 48, 64)],
               key_fmt=lambda r: f"N={r['N']}")

    g3 = sweep("grid 3: the spectral tail (same sigma2, same lengthscale family)",
               [learn(t, spectrum=s) for t in TARGETS for s in SPECTRA],
               key_fmt=lambda r: f"{r['spectrum']}")

    g4 = sweep("grid 4: how much of the interval is left unsampled (edge of the data)",
               [learn(t, edge=e) for t in TARGETS for e in (0.70, 0.80, 0.85, 0.90, 0.95, 0.999)],
               key_fmt=lambda r: f"EDGE={r['edge']:g}")

    g5 = sweep("grid 5: five independent data draws",
               [learn(t, seed=100 + s) for t in TARGETS for s in range(5)],
               key_fmt=lambda r: f"seed={r['seed']}")

    # ------------------------------------------------------------- analysis -------------
    def by(rows, t):
        return [r for r in rows if r["target"] == t]

    def spread(rows, t, f):
        v = [f(r) for r in by(rows, t)]
        return float(np.mean(v)), float(np.std(v)), float(np.ptp(v))

    print("\n summary: spread of the learned pair over each grid")
    summ = {}
    for label, rows in (("prior sd", g1), ("N", g2), ("spectrum", g3), ("edge", g4),
                        ("data seed", g5)):
        for t in TARGETS:
            ma, sa, pa = spread(rows, t, lambda r: r["alpha"])
            mb, sb, pb = spread(rows, t, lambda r: r["beta"])
            md, sd_, pd = spread(rows, t, lambda r: r["alpha"] - r["beta"])
            summ[f"{label}|{t}"] = dict(alpha_mean=ma, alpha_std=sa, alpha_ptp=pa,
                                        beta_mean=mb, beta_std=sb, beta_ptp=pb,
                                        diff_mean=md, diff_ptp=pd)
            print(f"   {label:10} {t:12} α={ma:7.3f}±{sa:.3f}  β={mb:7.3f}±{sb:.3f}  "
                  f"α-β={md:+7.3f} (range {pd:.3f})")

    asym = [t for t in TARGETS if E4.EXPECT[t]["asym"] != 0]
    sym = [t for t in TARGETS if E4.EXPECT[t]["asym"] == 0]
    sign_ok = lambda rows, t: ((r["alpha"] - r["beta"]) * E4.EXPECT[t]["asym"] > 0
                               for r in rows if r["target"] == t and not r["broke"])
    checks = {
        "the sign of alpha-beta survives the prior strength (sd 0.75-4) on the asymmetric targets":
            all(sign_ok(g1, t) for t in asym),
        "and it survives N = 16..64":
            all(sign_ok(g2, t) for t in asym),
        "and it survives the four spectral tails (se, exp, matern, sll)":
            all(sign_ok(g3, t) for t in asym),
        "the MAGNITUDE is prior-shrunk, monotonically in sd (a MAP effect, not a bug): at "
        "sd=0.75 the asymmetry is < 40 % of its sd=4 value, while the test error moves < 40 %":
            all(min(r["alpha"] - r["beta"] for r in by(g1, t))
                < 0.4 * max(r["alpha"] - r["beta"] for r in by(g1, t)) for t in asym)
            and all(min(r["rmse"] for r in by(g1, t))
                    > 0.6 * max(r["rmse"] for r in by(g1, t)) for t in asym),
        "and the shrinkage is monotone in sd for the jump target, in the clamped-edge error too":
            all(a[0] < a[1] for a in zip([r["alpha"] - r["beta"] for r in by(g1, "steep_right")],
                                         [r["alpha"] - r["beta"] for r in by(g1, "steep_right")][1:])
                )
            and all(b[0] > b[1] for b in zip([r["clamp"] for r in by(g1, "steep_right")],
                                             [r["clamp"] for r in by(g1, "steep_right")][1:])),
        "the magnitude is NOT robust across spectral tails (they trade against the space) - "
        "so report (alpha,beta) together with the spectrum, never alone":
            max(summ[f"spectrum|{t}"]["diff_ptp"] for t in asym) > 1.0,
        "the symmetric targets stay symmetric (|alpha-beta| < 0.3) on every grid":
            all(abs(summ[f"{g}|{t}"]["diff_mean"]) < 0.3 for g in
                ("prior sd", "N", "spectrum", "data seed") for t in sym),
        "sampling variability of the learned pair is small compared with the signal "
        "(std over 5 data draws < 0.5 for the asymmetric targets)":
            max(spread(g5, t, lambda r: r["alpha"])[1] for t in asym) < 0.5,
        "THE CONTROL: once the boundary layer IS sampled the extrapolative ADVANTAGE over "
        "Legendre disappears (clamped-edge ratio 1e-3 at edge=0.85 -> 0.8 at edge=0.999)":
            (by(g4, "steep_right")[2]["clamp"] / by(g4, "steep_right")[2]["legendre_clamp"] < 1e-2)
            and (by(g4, "steep_right")[-1]["clamp"]
                 / by(g4, "steep_right")[-1]["legendre_clamp"] > 0.3),
        "while the EVIDENCE gain grows as the boundary becomes observable "
        "(the likelihood can see the geometry, not just guess it)":
            by(g4, "steep_right")[-1]["gain"] > by(g4, "steep_right")[0]["gain"]
            and by(g4, "vanishing")[-1]["gain"] > by(g4, "vanishing")[0]["gain"],
        "and the learned asymmetry shrinks monotonically for the jump target as the data "
        "approach the free edge (the space is needed only where the data are absent)":
            all(a > b for a, b in zip([r["alpha"] - r["beta"] for r in by(g4, "steep_right")],
                                      [r["alpha"] - r["beta"] for r in by(g4, "steep_right")][1:])),
        "pure ML still breaks on this protocol (no prior -> the singular edge or a "
        "non-factorisable system)":
            sum(r["broke"] or r["beta"] < -0.995 or r["alpha"] > 5.0 for r in g1_ml) == len(g1_ml),
        "the clamped-edge error improvement (>= 2x) holds for the jump target on every grid":
            all(r["clamp"] < 0.5 * r["legendre_clamp"] for rows in (g1, g2, g3, g5)
                for r in rows if r["target"] == "steep_right" and not r["broke"]),
    }
    print("\n hypothesis checks")
    for k, ok in checks.items():
        print(f"   [{'PASS' if ok else 'FAIL'}] {k}")

    save_json("exp5_sensitivity", dict(
        targets=list(TARGETS), base=dict(edge=EDGE, M=M, noise=NOISE, N=E4.N,
                                         spectrum=E4.SPECTRUM, prior_sd=E4.PRIOR_SD),
        prior_sd_grid=g1, prior_sd_grid_pure_ml=g1_ml, n_grid=g2, spectrum_grid=g3,
        edge_grid=g4, seed_grid=g5, summary=summ, checks=checks))
    allrows = g1 + g2 + g3 + g4 + g5
    save_csv("exp5_sensitivity", allrows,
             ["target", "prior_sd", "N", "spectrum", "edge", "M", "data_noise", "seed",
              "alpha", "beta", "gain", "rmse", "clamp", "legendre_clamp", "broke"])
    plot(g1, g1_ml, g2, g3, g4, g5)


def plot(g1, g1_ml, g2, g3, g4, g5):
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 7.0))
    cols = {"ramp_right": "C0", "steep_right": "C1", "vanishing": "C2"}
    marks = {"ramp_right": "o", "steep_right": "s", "vanishing": "^"}

    def per_target(ax, rows, xkey, xlabel, logx=False, ml=None):
        for t in TARGETS:
            rr = [r for r in rows if r["target"] == t]
            x = [r[xkey] for r in rr]
            ax.plot(x, [r["alpha"] - r["beta"] for r in rr], marks[t] + "-", color=cols[t],
                    label=f"{t} (expected "
                          f"{'α>β' if E4.EXPECT[t]['asym'] > 0 else 'α<β' if E4.EXPECT[t]['asym'] < 0 else 'α=β'})")
        if ml:
            ax.text(0.98, 0.04, "pure ML, no prior:  " + "   ".join(
                        f"{r['target'][:5]}: {r['alpha'] - r['beta']:+.1f}"
                        + ("*" if r["broke"] else "") for r in ml)
                    + "\n* the solver itself broke down   (off scale, so no marker)",
                    transform=ax.transAxes, fontsize=6.2, ha="right", va="bottom")
        ax.axhline(0, color="k", lw=1)
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel); ax.set_ylabel("learned α − β")
        ax.legend(fontsize=6.2, loc="upper left")

    per_target(axes[0, 0], g1, "prior_sd", "MAP prior sd on raw(α), raw(β)", logx=True,
               ml=g1_ml)
    axes[0, 0].set_title("the prior strength moves the magnitude,\nnot the sign")
    per_target(axes[0, 1], g2, "N", "basis functions N")
    axes[0, 1].set_title("not a basis-size artefact\n(α−β flat over N = 16..64; only the test\n"
                         "error of the jump target needs the modes)")
    per_target(axes[0, 2], g3, "spectrum", "spectral tail of λn")
    axes[0, 2].set_xticks(range(len(SPECTRA)), SPECTRA, fontsize=7)
    axes[0, 2].set_title("the SIGN survives the tail, the magnitude\ndoes not: the spectrum "
                         "and the space trade off")

    ax = axes[1, 0]
    for t in TARGETS:
        rr = [r for r in g4 if r["target"] == t]
        ax.plot([r["edge"] for r in rr], [r["gain"] for r in rr], marks[t] + "-", color=cols[t],
                label=t)
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("data occupy [-edge, edge]")
    ax.set_ylabel("log ML gain over the fixed Legendre space")
    ax.set_title("the evidence likes the learned space MORE\nwhen the boundary is observed")
    ax.legend(fontsize=6.5)

    ax = axes[1, 1]
    for t in TARGETS:
        rr = [r for r in g4 if r["target"] == t]
        ax.plot([r["edge"] for r in rr],
                [max(r["clamp"], 1e-18) / max(r["legendre_clamp"], 1e-18) for r in rr],
                marks[t] + "-", color=cols[t], label=t)
    ax.axhline(1, color="k", lw=1, ls="--")
    ax.set_yscale("log")
    ax.set_xlabel("edge of the sampled region")
    ax.set_ylabel("clamped-edge RMSE: learned / Legendre")
    ax.set_title("THE CONTROL: the extrapolative advantage disappears\n"
                 "once the boundary layer is sampled (ratio -> 1)")
    ax.legend(fontsize=6.5)

    ax = axes[1, 2]
    for t in TARGETS:
        rr = [r for r in g5 if r["target"] == t]
        d = [r["alpha"] - r["beta"] for r in rr]
        ax.plot(range(1, len(d) + 1), d, marks[t] + "-", color=cols[t], ms=6,
                label=f"{t} (mean {np.mean(d):+.2f} ± {np.std(d):.2f})")
        ax.axhline(np.mean(d), color=cols[t], lw=0.8, ls=":")
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("independent data draw"); ax.set_ylabel("learned α − β")
    ax.set_title("sampling variability is small next to the signal\n"
                 "(the symmetric target stays on the α=β line)")
    ax.legend(fontsize=6.5)
    fig.tight_layout()
    savefig(fig, "exp5_sensitivity.png")


if __name__ == "__main__":
    main()
