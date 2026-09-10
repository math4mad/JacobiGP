# JacobiGP — a Gaussian Process whose *function space* is a hyper-parameter

A truncated Gaussian process written in the orthonormal Jacobi basis

$$f_N(x)=\sum_{n<N} w_n\,\widehat P_n^{(\alpha,\beta)}(x),\qquad w\sim\mathcal N(0,\Lambda),$$

so that the two weight exponents $(\alpha,\beta)$ of
$w(x)=(1-x)^\alpha(1+x)^\beta$ are ordinary, learnable hyper-parameters: changing them
changes the RKHS (which functions have finite norm, and how much variance the prior puts at
the two ends of the interval) without touching the spectrum $\Lambda$.

This work based on dialog with ima.copilot and qwen on multi-round.

ref : [Jacobi 多项式：换一组权重，就是换一个空间](https://math4mad.github.io/talk-with-agents/talking/mathematics/jacobi-orthogonal-polynomials.html)

Finally I decide to put it into practice.

Honestly. LLM Agents totally change world. make some daydreamer to a practicer

That is really good !

---

## What was claimed, what was verified

| | claim | result |
|---|---|---|
| Exp 1 | (α, β) impose boundary conditions that are nowhere coded | ✅ edge RMSE monotone in α+β over 5.8×, interior unchanged; the ordering **reverses** when the target does not vanish |
| Exp 2 | the space changes, not the spectrum | ✅ identical tr K and participation dimension, edge/centre sd 1.22 → 5.20 |
| Exp 3 | d/dx of the GP is another Jacobi GP, in the (α+1, β+1) space | ✅ ladder identity to 1.5e-15 relative; finite differences **converge** to it; the derivative carries an exact (and calibrated) sd |
| Exp 3b | so the PDE itself can be the data: a Bayesian collocation solve | ✅ u error 4.3e-5 at 36 modes (18× better than fitting u data, 130× better than finite differences at the same dof), Dirichlet data honoured to 1e-16, and the prior shown to be the regulariser |
| Exp 4 | the best space can be learned from the marginal likelihood | ✅ 6 targets, 6 boundary signatures, all recovered; 6 multi-starts agree to 1e-5 — **but only as MAP**: pure ML is degenerate on the open admissible set |
| Exp 5 | is that answer the method or the prior? | ✅ the **sign** of α−β survives prior sd 0.75–4, N 16–64, four spectral tails and five data draws; the **magnitude** is MAP-shrunk and spectrum-dependent; and the extrapolative advantage vanishes exactly when the boundary layer is sampled |

Read [`docs/RESULTS.md`](docs/RESULTS.md) for the numbers, the figures, the sensitivity
study and the negative results; [`docs/MATH.md`](docs/MATH.md) for the mathematical specification (basis,
eigenvalue schedules, inference, the ladder rule, edge asymptotics, and the addendum on why
a prior on (α, β) is part of the method).

Convention note: with $w=(1-x)^{\alpha}(1+x)^{\beta}$, **α controls the edge x = +1** and
**β controls x = -1** — the table in `AGENTS.md` has the two the other way round.

## Layout

```
src/jacobigp/
  basis.py       orthonormal Jacobi polynomials + derivatives, via the symmetric
                 three-term recurrence (pole-free, Autograd-differentiable in x, α, β)
  spectra.py     λ_n: se | exp | matern | sll (Sturm–Liouville of the Jacobi operator)
  model.py       whitened Bayesian linear regression: posterior, predictions of any linear
                 functional (per-row noise!), log marginal likelihood, exact LOO-CV,
                 MAP/ML fitting
  datasets.py    the synthetic targets and the manufactured Poisson problem
  metrics.py     RMSE split into interior / left edge / right edge
  baselines.py   an RBF-GP reference (and its finite-difference-only derivatives)
experiments/     exp1_boundary.py        exp2_rkhs_geometry.py   exp3_derivative_space.py
                 exp3b_pde_collocation.py exp4_evidence_learning.py
                 exp5_sensitivity.py     (shared helpers in common.py, and exp4's MAP fitter
                 is imported by exp5; each writes results/*.json + results/*.csv + figures/*.png)
tests/           21 tests: recurrence vs scipy, norms vs the Γ formula, kernel ≡ weight
                 space, prior-sample covariance, ladder derivatives, exact LOO vs an
                 explicit refit, per-row noise, the (α, β) prior, a PDE collocation solve
results/ figures/  one {json, csv, png} per experiment
```

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt           # numpy scipy torch matplotlib pytest
python -m pytest                          # 20 passed
python experiments/exp1_boundary.py        # ~40 s
python experiments/exp2_rkhs_geometry.py   # ~20 s
python experiments/exp3_derivative_space.py  # ~2 min
python experiments/exp3b_pde_collocation.py  # ~1 min
python experiments/exp4_evidence_learning.py # ~7 min
python experiments/exp5_sensitivity.py      # ~4 min
```

## Two things to know before using it

`α, β > -1` is enforced structurally (`α = -1 + softplus(u)`), the admissible set is
nevertheless **open**, and the evidence is unbounded on it: with no prior, the optimiser
walks to β = -1 (or to α = β = -1, where the recurrence stops existing) while σ² absorbs
the basis normalisation — up to α ≈ 224, σ² ≈ 1e62, then the design matrix loses rank.
Maximise the *penalised* evidence:

```python
penalty = lambda p: 0.5 * ((p.raw_alpha / 2.0) ** 2 + (p.raw_beta / 2.0) ** 2)
gp.fit(X, y, steps=400, lr=0.05, penalty=penalty)     # interior, unique, reproducible
```

**The exact derivative makes the PDE itself usable as data.** `linear_posterior` takes any
matrix of linear functionals of the weights, with a *per-row* variance — so a strong-form
collocation solve of `-u'' = s`, `u(±1) = 0` is four lines and no values of `u` are needed:

```python
C = nodes(J)                                    # Chebyshev collocation nodes
Psi = torch.cat([-gp.basis(C, deriv=2),         # exact, via the ladder identity
                  gp.basis(torch.tensor([-1., 1.]))])
y  = torch.cat([source(C), torch.zeros(2)])
row_var = torch.cat([(1e-3)**2 * torch.ones(J), (1e-9)**2 * torch.ones(2)])
post = gp.linear_posterior(Psi, y, row_var)     # u, u', u'' and their sds: gp.predict(..., deriv=k)
```

See `experiments/exp3b_pde_collocation.py`; the same call also accepts integral rows, which is
what a weak-form/Galerkin discretisation needs.
