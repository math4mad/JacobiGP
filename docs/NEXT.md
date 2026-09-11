# ONE SPACE — draft: merging JacobiGP and the SVD/LoRA project

> **Status: proposal, not result.** Nothing on this page has been measured yet.
> The numbers that *do* exist live in [RESULTS](report.html); the machinery this
> proposal reuses is specified in [MATH](math.html). Drafted 2026-09-11,
> the morning the 花絮 conversations happened.

---

## 1. The one idea

Both projects are instances of a single object:

$$
\Delta \;=\; \sum_{n=0}^{N-1} c_n\, \phi_n,
\qquad c_n \sim \mathcal{N}(0, \lambda_n),
\qquad \{\phi_n\} \text{ orthogonal under } w_{\alpha,\beta}(x) = (1-x)^\alpha (1+x)^\beta
$$

Three knobs, three distinct jobs — and the discipline of the merged project is
to **never let the jobs blur together**:

| knob | JacobiGP role | LoRA / SVD counterpart | what it cuts |
|---|---|---|---|
| $N$ | basis truncation | rank $r$ | **size** — how many directions may change |
| $\lambda_n$ | spectral decay | init scale / weight decay on $B$ | **smoothness** — which frequencies are cheap |
| $(\alpha,\beta)$ | boundary weight | *(no standard equivalent — this is the contribution)* | **shape** — *where* change is allowed |

LoRA says $\Delta W = AB$: an $r$-dimensional update, **isotropic** inside that
subspace — every direction costs the same. The SVD project chooses the
subspace from the weight matrix's own spectrum ($UV^\top$ from top singular
directions). Neither one says anything about *where in input space* the update
should be stiff and where supple. That is exactly the degree of freedom
$(\alpha,\beta)$ supplies, and exactly the one the morning conversation was
reaching for:

> Rank chooses *how much* you're allowed to change; $(\alpha,\beta)$ chooses
> *where you're allowed to change* — and the evidence chooses the boundaries,
> which is the only way choosing them isn't arrogance.

Hallucination, in this frame, is the **Runge phenomenon at the boundary of the
data manifold**: ringing where nothing constrains you. Ford's line and the
Westworld poster are the same sentence with the knobs swapped —
*live without limits* is what an isotropic, untruncated, unweighted space
actually does, and it fits noise perfectly and means nothing.

## 2. Why one project, not two

The shared object is not a metaphor; it is the same code path:

1. **An orthonormal basis, chosen or measured.** JacobiGP evaluates
   $\Phi_{mn} = P_n^{(\alpha,\beta)}(x_m)$ analytically; the SVD project
   computes bases numerically from weight/data spectra. Both then do
   *truncated expansion with a prior on coefficients*.
2. **Diagonal prior $\Rightarrow$ everything is $\Phi \Lambda \Phi^\top$.**
   Posterior, marginals, and gradients w.r.t. the shape of the space are the
   3-line Bayesian-linear-regression kernel already implemented here; a
   spectral LoRA (coefficients on measured singular vectors with a prior on
   the coefficients) is the same kernel with a different $\Phi$.
3. **Evidence learning over the space itself.** $\log p(X \mid \theta, \alpha, \beta)$
   optimized here w.r.t. $(\alpha,\beta)$ (Exp 4) is the template for learning
   *any* shape parameter of a function/weight space — including the anisotropic
   penalty on LoRA coefficient directions proposed in §4.

So the merge is: **`spaces/` — one library of bases (Jacobi, Fourier, SVD-from-data),
one library of spectra ($\lambda_n$), one evidence optimizer, one experiment
harness with the check-pill discipline of Exps 1–5.** A GP on a function space
and a fine-tune inside a weight space become two front ends of the same object.

```
one-space/
├── spaces/          # bases: jacobi.py, svd.py, fourier.py; spectra: decay.py
│                    # w_{α,β}-orthogonality, numerically stable eval (from src/jacobigp)
├── infer/           # shared ΦΛΦᵀ posterior + profiled evidence (Exp 4 machinery)
│                    # — backend: torch, MPS-friendly (M1 Pro is the bench)
├── models/          # checkpoints + the SVD project's extracted spectra/subspaces,
│                    # published alongside so experiments are reproducible, not hand-tuned
├── experiments/     # exp1..5 (this repo) + exp6 (diagnostic) + exp7 (weighted LoRA)
└── results/         # *.json with "checks": one pill per hypothesis — the site reads them
```

Shared model artifacts are load-bearing, not courtesy: the SVD project
provides the *measured* $\Phi$ (real LLM weight spectra) that lets the Jacobi
half stop being an analogy and become a claim about actual models.

## 3. Experiment 6 (next, cheap, falsifiable): learned exponents as a rank gauge

*The pilot that connects the two halves without touching a big GPU.*

1. Train the same small model at ranks $r \in \{2, 8, 32, 64\}$ on a
   two-domain mixture (e.g. logic puzzles + code).
2. Take each validation-loss-vs-epoch curve, map epoch $\mapsto [-1,1]$, and
   fit `JacobiGP` with learnable $(\alpha,\beta)$ via the Exp-4 evidence path.
3. **Hypothesis (H6a):** the learned $(\alpha,\beta)$ of an *under-ranked* run
   (curve forced to wiggle late) separates from an *over-ranked* run (rising
   tail = overfit) **before** the val curve itself shows it — i.e. the
   exponents are an early-warning gauge of rank mis-set.
   **H6b:** a threshold rule on $(\alpha,\beta)$, calibrated on these curves,
   selects $r$ within ±1 octave of the oracle-best $r$ on held-out model/data
   pairs — beating grid search at equal compute.
4. Negative result is first-class: if the exponents track nothing beyond what
   the tail of the curve already says, we report that (Exp 4's pure-ML ridge
   set the precedent).

## 4. Experiment 7 (the flagship): boundary-weighted spectral LoRA

The actual transplant. Give the LoRA coefficient prior the Jacobi weight:

$$
\Delta W = \sum_{n} c_n\, U_n V_n^\top, \qquad
c_n \sim \mathcal{N}\!\big(0, \lambda_n(\alpha,\beta)\big), \qquad
\lambda_n \propto \int_{-1}^{1} P_n^{(\alpha,\beta)}(u)^2\, (1-u)^\alpha (1+u)^\beta\, du
$$

where $u$ is a *semantic coordinate* for update directions — e.g. normalized
singular index (which layer of the spectrum), or alignment of $V_n$ with the
in-/out-of-distribution split measured on probe data. High $\alpha$: stiff at
the "familiar" end (protect known behaviour — the anti-catastrophic-forgetting
dial); low $\beta$: supple at the "edge" (allow adaptation where the data
manifold ends). $(\alpha,\beta)$, $\ell$, and $r$ are then learned jointly by
evidence (or its cheap variational proxy on probe batches) — the Exp-4 loop,
now over the shape of a *weight* space.

**Predictions, stated to be killed if wrong:**

- **P1 (shape ≠ size):** at fixed $r$, learned $(\alpha,\beta)$ improves
  two-domain retention vs. plain LoRA and vs. isotropic-spectral LoRA
  (the correct ablation: same $\lambda$, no boundary weight). If only $r$
  matters, this project's thesis dies and we say so.
- **P2 (boundary dial):** forced-negative test — $\alpha \to$ high should
  measurably reduce forgetting on domain A while fine-tuning on domain B;
  the $(\alpha,\beta)$-as-boundary-condition controller result of Exp 1,
  restated in weight space.
- **P3 (cheap gauge):** H6a's exponents predict the P1 optimum, making the
  full search a warm start.

## 5. Naming and papers

Working name **ONE SPACE** (one space, three knobs). Tagline already earned,
from the 花絮, and it is a *scientific* sentence, not decoration:

> *JacobiGP: to see the world — in Westworld there are boundaries; in the
> functional world there are no limits. We choose the boundaries to see the
> beauty.* — and choosing is done by the evidence, never by hand.

Publication split: JacobiGP (this site) = the auto-constructed function space,
negative results included; SVD/LoRA = measured weight-space spectra; the joint
contribution is §4: **inductive bias as a learnable measure on update
directions.**

## 6. Sequence

1. Merge: extract `spaces/` + `infer/` from this repo and the SVD repo; keep
   exp1–5 green (pills must not dim) while both repos import the shared core.
2. Publish model artifacts (extracted spectra, probe data) under `models/`.
3. Exp 6 — one afternoon on the M1 Pro bench; it either funds §4's confidence
   or spends itself honestly on a null.
4. Exp 7 — flagship; small models first, P1/P2/P3 in order.

*Reality is all the $(\alpha_i, \beta_i)$ superposed; a project is the act of
picking two of them and being answerable for the choice. This page is that act,
written down before the data gets a vote.*
