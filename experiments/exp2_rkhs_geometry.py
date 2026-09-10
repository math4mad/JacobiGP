"""
Experiment 2 - RKHS geometry & prior sampling
=============================================
Hypothesis (AGENTS.md): with the eigenvalues lambda_n *fixed*, changing (alpha, beta)
changes the *shape* of the function space, i.e. the RKHS itself - not merely its
smoothness.

Protocol
  Set 1  (0, 0)  Legendre       Set 2  (2, 2)  symmetric boundary freedom
  Set 3  (5, 0)  asymmetric     Set 4  (-0.5, -0.5)  implicit Dirichlet
For each set: 50 prior samples, the marginal sd k(x,x)^(1/2), the weight w(x), and the
spectral decay (identical by construction).  A numerical RKHS fingerprint is added:
the energy of a random draw near the edges vs the interior, and the "effective
dimension" (participation ratio of lambda_n * the basis norms).
"""

from common import DT, banner, save_csv, savefig, save_json

import matplotlib.pyplot as plt
import torch

from jacobigp import JacobiGP
from jacobigp.basis import jacobi_orthonormal, jacobi_weight

SETS = [
    ("1  Legendre   (0,0)", 0.0, 0.0),
    ("2  symmetric  (2,2)", 2.0, 2.0),
    ("3  asymmetric (5,0)", 5.0, 0.0),
    ("4  Dirichlet (-0.5,-0.5)", -0.5, -0.5),
]
SPECTRUM = "se"
LS, SIGMA2, N = 0.5, 1.0, 30


def main():
    banner("Experiment 2 - prior geometry of the space, lambda_n held fixed")
    xs = torch.linspace(-1, 1, 401, dtype=DT)
    # the SAME standard normal coefficients for every set: the only thing that changes between
    # the panels is the space, so a difference in the pictures is a difference in the RKHS
    rows, store = [], {}

    for name, a, b in SETS:
        gp = JacobiGP(N=N, spectrum=SPECTRUM, alpha=a, beta=b, lengthscale=LS,
                      sigma2=SIGMA2, noise=1e-6, dtype=DT)
        with torch.no_grad():
            S = gp.sample(xs, num=50,                     # = (eps * sqrt(lam)) @ Phi.T
                          generator=torch.Generator().manual_seed(42))
            mean, var, _ = gp.predict(xs)
            sd = var.sqrt()
            w = jacobi_weight(xs, gp.alpha, gp.beta)
        lam = gp.eigenvalues()
        Phi = jacobi_orthonormal(xs, N, gp.alpha, gp.beta)
        edge = xs.abs() > 0.9
        r = {
            "set": name, "alpha": a, "beta": b,
            "sample_min": float(S.min()), "sample_max": float(S.max()),
            "edge_sd": float(sd[edge].mean()), "centre_sd": float(sd[~edge].mean()),
            "edge_over_centre": float(sd[edge].mean() / sd[~edge].mean()),
            "edge_crossing_frac": float((S[:, edge].abs().min(dim=0).values
                                         > 0.5 * sd[edge].mean()).float().mean()),
            "trace_K": float(lam.sum()),
            "participation_dim": float(lam.sum() ** 2 / (lam**2).sum()),
            "mean|f|_edge": float(S[:, edge].abs().mean()),
            "mean|f|_centre": float(S[:, ~edge].abs().mean()),
        }
        # ensemble sd of the samples at the right edge vs the analytic sd: ~1 if the
        # Monte-Carlo draw reproduces the prescribed geometry of the space
        near = xs > 0.9
        r["edge_consistency"] = float((S[:, near].std() / sd[near].mean()) ** 2)
        rows.append(r)
        store[name] = dict(xs=xs, S=S, sd=sd, w=w, lam=lam.detach(), Phi=Phi.detach())
        print(f"  {name:28s} sd(±1)/sd(0) = {r['edge_over_centre']:6.2f}   "
              f"sample range [{r['sample_min']:6.2f},{r['sample_max']:6.2f}]   "
              f"edge|f|/centre|f| = {r['mean|f|_edge']/r['mean|f|_centre']:.2f}")

    checks = {
        "identical spectra => trace(K) and participation dimension are the same for all sets":
            max(r["trace_K"] for r in rows) - min(r["trace_K"] for r in rows) < 1e-9
            and abs(max(r["participation_dim"] for r in rows)
                    - min(r["participation_dim"] for r in rows)) < 1e-9,
        "edge/centre sd is monotone in alpha+beta (spaces are geometrically different)":
            all(rows[i]["edge_over_centre"] < rows[j]["edge_over_centre"]
                for i, j in ((3, 0), (0, 1), (1, 2))),
        "Set 3 (5,0) is asymmetric: sd at +1 >> sd at -1":
            bool(store[SETS[2][0]]["sd"][-1] > 3 * store[SETS[2][0]]["sd"][0]),
        "typical prior sample of Set 4 is small at the edge, Set 2 is large":
            rows[3]["mean|f|_edge"] / rows[3]["mean|f|_centre"]
            < rows[1]["mean|f|_edge"] / rows[1]["mean|f|_centre"],
    }
    print("\n hypothesis checks")
    for k, ok in checks.items():
        print(f"   [{'PASS' if ok else 'FAIL'}] {k}")

    save_json("exp2_rkhs_geometry", dict(sets=[dict(zip(("label", "alpha", "beta"), s))
                                               for s in SETS],
                                         spectrum=SPECTRUM, lengthscale=LS, sigma2=SIGMA2, N=N,
                                         metrics=rows, checks=checks))
    save_csv("exp2_rkhs_geometry", rows, list(rows[0].keys()))
    plot(store, rows)


def plot(store, rows):
    names = [s[0] for s in SETS]
    xs = store[names[0]]["xs"]
    fig, axes = plt.subplots(3, 4, figsize=(12.6, 8.6))

    # row 0: 50 prior samples per space (identical underlying weights -> only geometry differs)
    for i, name in enumerate(names):
        ax = axes[0, i]
        d = store[name]
        for s in d["S"]:
            ax.plot(xs, s, lw=0.6, alpha=0.75)
        ax.plot(xs, d["sd"], "k--", lw=1.0)
        ax.plot(xs, -d["sd"], "k--", lw=1.0)
        ax.set_title(name, fontsize=8.5)
        ax.set_ylim(-5, 5)
        ax.set_xlabel("x")
        ax.set_ylabel("prior samples" if i == 0 else "")
        if i:
            ax.set_yticklabels([])

    # row 1: marginal sd and the weight function (shared panels)
    for i, name in enumerate(names):
        d = store[name]
        axes[1, 0].plot(xs, d["sd"], label=name.split()[0], lw=1.3)
        axes[1, 1].plot(xs, d["w"], label=name.split()[0], lw=1.3)
    axes[1, 0].set_yscale("log"); axes[1, 0].axhline(1.0, color="0.6", lw=0.7)
    axes[1, 0].set_title("marginal sd  k(x,x)^1/2   (identical lambda_n)")
    axes[1, 0].set_xlabel("x"); axes[1, 0].set_ylabel("sd  (log)")
    axes[1, 0].legend(fontsize=6.5, ncol=2, title="set")
    axes[1, 1].set_yscale("log"); axes[1, 1].set_ylim(1e-3, 1e3)
    axes[1, 1].set_title("weight w(x) = (1-x)^alpha (1+x)^beta")
    axes[1, 1].set_xlabel("x"); axes[1, 1].set_ylabel("w  (log)")
    axes[1, 1].legend(fontsize=6.5, ncol=2, title="set")
    axes[1, 2].bar([n.split()[0] for n in names],
                   [r["edge_over_centre"] for r in rows], edgecolor="k", lw=0.4)
    axes[1, 2].set_title("edge/centre sd ratio  = geometry fingerprint")
    axes[1, 2].set_ylabel("sd(|x|>0.9)/sd(int)"); axes[1, 2].set_yscale("log")
    axes[1, 3].bar([n.split()[0] for n in names],
                   [r["mean|f|_edge"] / r["mean|f|_centre"] for r in rows],
                   edgecolor="k", lw=0.4)
    axes[1, 3].set_title("typical sample: mean|f| at edge / in centre")
    axes[1, 3].axhline(1.0, color="0.6", lw=0.7); axes[1, 3].set_ylabel("ratio")

    # row 2: basis functions and spectral decay (identical by construction)
    for i, name in enumerate(names[:2]):
        axes[2, i].plot(xs, store[name]["Phi"], lw=0.7, alpha=0.85)
        axes[2, i].set_yscale("log"); axes[2, i].set_ylim(1e-3, 1e2)
        axes[2, i].set_title(f"|hat P_n(x)|, set {name.split()[0]}")
        axes[2, i].set_xlabel("x"); axes[2, i].set_ylabel("log|hat P_n|")
    axes[2, 2].plot(torch.arange(N), store[names[0]]["lam"], "o-", label="all four sets")
    axes[2, 2].set_yscale("log")
    axes[2, 2].set_title("lambda_n : identical for every set")
    axes[2, 2].set_xlabel("degree n"); axes[2, 2].set_ylabel("lambda_n"); axes[2, 2].legend()
    axes[2, 3].axis("off")
    axes[2, 3].text(0.02, 0.72,
                    "Same random weights, same spectrum.\n"
                    "Only (alpha, beta) changed.\n\n"
                    "-> the plotted objects are not the\n"
                    "   same probability measure.\n"
                    "   alpha,beta > 0 : boundary freedom\n"
                    "   alpha,beta < 0 : implicit Dirichlet",
                    fontsize=8.5, va="top", family="monospace")
    fig.tight_layout()
    savefig(fig, "exp2_rkhs_geometry.png")


if __name__ == "__main__":
    main()
