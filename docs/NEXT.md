# ONE SPACE — draft: merging JacobiGP, Middle-Eigen-function, and Sarcos-NN-Model

> **Status: proposal, not result.** Nothing on this page has been measured yet —
> except the parts quoted from [Middle-Eigen-function](https://github.com/math4mad/Middle-Eigen-function)
> and [Sarcos-NN-Model](https://math4mad.github.io/Sarcos-NN-Model), which are
> marked as measured. The numbers that *this* site owns live in
> [RESULTS](report.html); the machinery reused below is specified in [MATH](math.html).
> Drafted 2026-09-11, the morning the 花絮 conversations happened; third bench
> wired in the same day.

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

| knob | JacobiGP role | MEF / Sarcos counterpart | what it cuts |
|---|---|---|---|
| $N$ | basis truncation | rank $r$; "给学习多大范围" | **size** — how many directions may change |
| $\lambda_n$ | spectral decay | which singular band to keep (A/B/C, 中带, top 谱) | **smoothness** — which frequencies are cheap |
| $(\alpha,\beta)$ | boundary weight | *no equivalent yet — this is the contribution* | **shape** — *where* change is allowed |

LoRA says $\Delta W = AB$: an $r$-dimensional update, **isotropic** inside that
subspace. MEF cut that subspace by singular value position; Sarcos cut it
under a pinned split with pre-registered predictions. Neither says anything
about *where in input space* the update should be stiff and where supple. That
is exactly the degree of freedom $(\alpha,\beta)$ supplies:

> Rank chooses *how much* you're allowed to change; $(\alpha,\beta)$ chooses
> *where you're allowed to change* — and the evidence chooses the boundaries,
> which is the only way choosing them isn't arrogance.

### 1b. The schedule clause — onset is an argument, not a fourth knob

*Ruled into the table 2026-09-12 by the Geometer, on the proposal of Kairos
(Letter 008); the paragraph was requested, not a table row, and so it is —
the knobs stay three.*

The three knobs are not properties of a run; they are properties of a
**training program**, and every program has a fixed-point structure: the
same $(N, \lambda, \alpha, \beta)$ committed at checkpoint $k=0$ and the same
triple committed at $k=75\%$ act on *different geometries of the loss
surface* — because the container that meets the update is itself the
product of the update history. Onset $k$ is therefore not a fourth
coordinate beside size, order and shape; **$k$ is the argument at which the
prior is applied.** The omission is not cosmetic: without it, §5's
"boundary-weighted LoRA" is ambiguous — boundary *when*? The two readings
(fixed at init vs. injected into a trained base) are different experiments
that share a formula, and law 4 already forces them into different rows
(*injection-into-a-moving-base* is a third regime, per Letter 014
amendment 4). That difference in geometry is itself measurable — it is
exactly what exp6's gauge (reading the container's stiffness) and Sitting
003's H9 (reading the rank needed at onset $k$) measure from two
directions; if they ever disagree, one of the two readings of "the same
dial" was décor.

**Vocabulary, adopted with an operative definition:** *kairotic* (adj.,
from $\kappa\alpha\iota\rho\acute{o}\varsigma$; the Horologist's offer) — a
claim $C(N,\lambda,\alpha,\beta)$ is kairotic iff its verdict (within
calibrated bands, never eyeballed) changes when the commitment time $k$
changes at fixed knob values. Example of use, falsifiable as written: *the
gauge is kairotic iff H6b's rank selection degrades beyond its registered
band at $k>0$ relative to $k=0$.* One word, one axis of falsification, no
new table. Kairos keeps studying the schedule whether or not the word
survives here; the word is the cheap part of the debt.

Hallucination, in this frame, is the **Runge phenomenon at the boundary of the
data manifold**: ringing where nothing constrains you. *Live without limits*
is what an isotropic, untruncated, unweighted space actually does — it fits
noise perfectly and means nothing.

## 2. What Middle-Eigen-function already settled — and it triangulates with us

Measured there (their report's own words, ±0.030 noise band, paired seeds):

1. **Size is real.** *"真正决定微调效果的是'给学习多大范围'，而不是'切掉了第几大的那些'"* —
   the update-space extent ($N$) dominates. Our table's first row, confirmed
   from the weight side.
2. **Spectral *position* is mostly noise.** σ-order works at coarse scale and
   fails at fine scale (Stages 11–14: shuffle-rebuild, random-subspace,
   Kendall-τ lesions); the band-surgery hypotheses were progressively
   retracted; junk writes into the **middle band**, not the tail (Stage 10).
   Row two, demoted — you cannot express a domain-space boundary as a
   σ-band cut, and they empirically proved it before we theorized it.
3. **A residue of *where* survived everything.** L6 (Stage 9): cutting the top
   spectral band *of the deep layers only* = removing harmful memory,
   direction replicated, amplitude +0.02–0.03. Small — but it is the same
   ghost we chase: location, not magnitude.
4. **The atoms are already there.** Stage 14's rank-1 atom reparameterization,
   $\Delta W = \sum_n c_n U_n V_n^\top$, *is* the shared object's $\Phi$;
   ATOMfull gained +0.022 with 1/61 of the parameters. What the atom basis
   lacks is a measure — an answer to "which atoms are *near the boundary of
   the data manifold*?"

So: **MEF killed the σ-axis as a location axis and showed the size axis
matters; JacobiGP contributes the third axis and a machine that learns it
from evidence (Exp 4).** The merged thesis is precisely the question MEF
cannot ask with its own knobs: not *which* atoms, but *where the atoms live
in semantic space*.

### 2b. The Sarcos bench — what it adds, and what it costs us

Measured at [`Sarcos-NN-Model`](https://github.com/math4mad/Sarcos-NN-Model)
(commit `4ea7678`), on 21-64-64-7 MLPs, 3 seeds, split pinned in `data.py`,
predictions pre-registered:

5. **The middle-band hypothesis died a second time, on a different bench.**
   Sarcos's "read-out inversion" (middle $>$ leading at low rank) **failed on
   its own pre-registered test**: leading $\gg$ middle $\approx$ trailing
   transfers from $W$ to $\Delta W$ (r64: 0.025 vs 0.805/0.807). Two
   independent agents, one small dense model and one LLM rig, killing the same
   idea — row two of the table is now the *best-supported* claim in the
   programme.
6. **A constraint on §5: the increment is not low-rank.** Sarcos's from-scratch
   $\Delta W$ needs rank 144/256 for 99% of its energy, $\|\Delta W\| \approx
   1.7\,\|W_{\text{init}}\|$. The boundary-weighted prior must be defined over a
   *full-rank* atom dictionary with a decaying $\lambda_n$ — not over a
   handful of top directions. (Their own note: this bounds what that bench
   tests, it does not refute LoRA's fine-tuning ranks. §5 lives in the
   fine-tuning regime; Sarcos keeps the from-scratch control.)
7. **The method becomes the programme's law.** Sarcos's hard rules — one
   canonical split module, regimes in separate rows, retained energy next to
   every error, pre-registered predictions, never tune on test — are adopted
   across all three repos (written into this repo's `AGENTS.md` §0).
8. **The flagship's first real data.** SARCOS *is* the GPML benchmark — the
   classic function-space dataset walks in from the weight-space side. And it
   is the one dataset in the programme whose boundaries are not metaphor:
   joint angles have **physical range-of-motion limits**, and inverse-dynamics
   torques behave very differently inside them than at them. A JacobiGP per
   joint with $(\alpha,\beta)$ at the measured joint limits, tested against
   the z-scored block-split data pipeline already pinned in `data.py`, is the
   cheapest possible Exp 7.5 — real data, real boundaries, both repos' code
   already written.

### 2c. The fourth bench — shape taken inside the layer (PolyNN)

[`Polynomial-Activated-NN`](https://github.com/math4mad/Polynomial-Activated-NN)
(brief at `2a109f2`, no code yet) puts a learnable polynomial
$\phi = \sum_{k=0}^{d} c_k P_k^{(\alpha,\beta)}$ where ReLU sits. In the
three-knob table this is not a new row — it is the **third row, interiorized**:
size was depth/width, spectrum was weight decay, and now *shape* lives in the
nonlinearity each neuron applies to its pre-activation. ReLU vs. Jacobi-
activated is literally two containers: piecewise-linear (infinite high-frequency
energy at the kink — the network equivalent of a flat spectrum) vs. a degree-$d$
basis (bandlimited by construction). Parameter-efficiency claims are therefore
spectral claims wearing a hat.

**Compute reality check** (the brief was feared unrunnable; arithmetic says
otherwise): 784-128-10 $\approx 10^5$ params, fwd+bwd $\approx 0.6$ MFLOP per
sample, 60k $\times$ 20 epochs $\approx 0.7$ TFLOP *total per arm* — same
order as MEF's declared "minimum publishable" 20-minute run. Whole planned
matrix (2 arms × 2 widths × seeds): **well under an hour on the shared M1
Pro.** Fashion-MNIST (~30 MB) downloads into the shared `chora/data/` store,
already symlinked.

Pre-registered risks worth naming before any epoch runs:

- **Explosion.** Hermite/Chebyshev powers grow fast on wide pre-activations;
  the brief's own $[-3,3]$ plot window is where the story is. Jacobi on a
  *squashed* pre-activation $u = \tanh(x)$ keeps the basis on its natural
  finite interval — the family with built-in boundaries is the one whose
  domain matches a bounded activation. Prediction P-shape: for equal degree,
  tanh-squashed Jacobi $\geq$ raw Hermite $\geq$ raw Chebyshev in
  accuracy-per-parameter.
- **Regime separation (programme rule).** A learned *unbounded* polynomial
  activation is neither post-hoc truncation nor a LoRA increment — its own
  row, its own baseline, per Sarcos's rule 3.
- **The honest ablation.** Same coefficient budget: a degree-$d$ poly
  activation has $d{+}1$ params per neuron; "50–70% fewer neurons at equal
  accuracy" only means something against ReLU nets matched on *total*
  parameters, not neuron count.

If P-shape holds, Exp 7's boundary-weighted LoRA gains a sibling: the same
evidence-learned $(\alpha,\beta)$ dial, one applied to *how neurons fire*, one
to *how weights move*.

> **VERDICT 2026-09-12 18:0x (+0800), measured — the sibling above is HALF dead.**
> H6c, the init-measure form of that sentence, **FAILS**: on PolyNN's own pooled
> epoch-0 pre-activation histogram of the A-jacobi h=128 arm, the Exp-4 evidence
> path returns $(\hat\alpha,\hat\beta)=(-0.9226,-0.9163)$ against the registered
> target $(0.40,0.37)$ at band $\pm0.2000$ — six bands wide, and *negative*, which
> is the second clause of this experiment's own pre-written obituary. All six
> starts × five seeds × both registered basis sizes ($N=30$ deciding, $N=64$
> sensitivity, 4 dp apart) agree, and the fit's explained variance is 0.9999 while
> $\sigma^2\sim10^8$: the objective reaches the good fit and puts the exponents on
> the $\alpha,\beta\to-1$ edge `MATH.md` §6 was written to fence. So **"the same
> dial read from two ends" is struck as a merged claim for the *init* row** and
> stays live, untested, only where the measure has amplitude $O(1)$ by
> construction: training curves (H6a/H6b) and weight-space increments (exp 7).
> PolyNN's own result is untouched — its learned pair still walks to
> $(0.3589,0.3590)$, replicated bit-identically on machine B. Nothing here is
> evidence against P-shape. Bytes: `chora:artifacts/results/jacobigp/exp6_h6c/h6c_verdict.json`
> `(sha256 c51e001a3b1d…)`; paperwork `PREREG_EXP6.md` §H6c + A1 @`99c7a4a`;
> announcement Letter 006.

## 3. Why one project, not two

The shared object is not a metaphor; it is the same code path:

1. **An orthonormal basis, chosen analytically or measured.** Here
   $\Phi_{mn} = P_n^{(\alpha,\beta)}(x_m)$; there, singular vectors of real
   Qwen2.5 weight matrices and rank-1 atom dictionaries — both already
   produced as artifacts in `outputs/stage*/`.
2. **Diagonal prior ⇒ everything is $\Phi \Lambda \Phi^\top$.** Posterior,
   marginals, and gradients w.r.t. the shape of the space are the
   Bayesian-linear-regression kernel implemented here; spectral LoRA is the
   same kernel with a measured $\Phi$.
3. **Evidence learning over the space itself.** Optimizing
   $\log p(X \mid \theta, \alpha, \beta)$ (Exp 4) is the template for learning
   *any* shape parameter of a function/weight space — including the
   anisotropic penalty on LoRA coefficient directions in §5.

Proposed layout for the merge:

```
one-space/
├── spaces/          # bases: jacobi.py (from src/jacobigp), svd_atoms.py (MEF stage14),
│                    # fourier.py; spectra: decay.py; w_{α,β}-orthogonality, stable eval
├── infer/           # shared ΦΛΦᵀ posterior + profiled evidence (Exp-4 machinery),
│                    # torch backend, MPS-friendly (M1 Pro is the bench of all three)
├── data/            # Sarcos's data.py split loader, promoted: one canonical split,
│                    # pinned, leakage-diagnosed — the pattern every dataset follows
├── models/          # shared artifacts: MEF spectra / atom dictionaries, Sarcos run
│                    # records (weights_init.npz included), this repo's results/*.json
├── experiments/     # exp1..5 (JacobiGP) + MEF stages + Sarcos bands + exp6 + exp7
└── results/         # *.json with "checks": one pill per hypothesis — the site reads them
```

All three repos already endow every number with a recompute path (`scripts/` →
`outputs/` → rendered report; here `experiments/` → `results/*.json` →
`build_site.py`; there `results/` → `summary.json` → Quarto tables that
*cannot disagree* with the runs). Publishing models/spectra alongside is the
only new obligation, and it is what turns "the analogy holds" into "the claim
is about actual weights".

## 4. Experiment 6 (next, cheap, falsifiable): learned exponents as a rank gauge

> **Registered 2026-09-12:** the pre-registration text of this experiment —
> H6a/H6b/H6c as one committed document, bands, regime rows and obituaries
> included — is `PREREG_EXP6.md` (this directory). Where this sketch and
> the registered text differ, **the registered text governs the numbers.**
>
> **Measured 2026-09-12, same day:** **H6c FAILS** (the first number of exp6 — see
> the §2c verdict block, `h6c_verdict.json (c51e001a3b1d…)`, Letter 006).
> **H6a's Sarcos pilot is DEAD on its own curve-clause** (A2 §3: HOLDS iff
> $S(0.50)>3.0$ **and** $\Delta\le B$; measured $S=7.880$ — 2.6× the threshold —
> but $\Delta=0.41648$ against $B=0.12416$, i.e. the val curves separated long
> before the gauge fired; `h6a_pilot_verdict.json (e08ca86d2ea4b381…)`,
> Letter 007→Sarcos). What survives it is **composition**:
> $\mathrm d\beta=-0.78143$ vs $\mathrm d\alpha=-0.06661$ (per-coordinate pooled
> sd 0.13916 / 0.02112) — one edge of the dial carries the entire effect and the
> other is flat, the same one-ruler-too-global signature as H6c. **This does not
> license the MEF rank-curve training** (A2 §4 said a HOLD was the licence; a
> DEAD clause is not a HOLD and is not a FAIL of the dial either).
> **H6a on MEF's curves and H6b therefore remain open and unmeasured**, and both
> must now report $\sigma^2$ and the fitted edge-distance next to every
> $(\hat\alpha,\hat\beta)$ (A1 §3), because §6's ridge has now been met twice on
> this programme's own data.

*The pilot that connects the halves — runs on MEF's harness, not a new one.*

1. Reuse MEF's LoRA rig (Qwen2.5-0.5B + RTE, paired seeds, ±0.030 band),
   training at ranks $r \in \{2, 8, 32, 64\}$; MEF already swept ranks for
   effect size, we sweep them for **curve geometry**. A cheaper pilot of the
   same gauge runs on Sarcos's MLPs first (minutes per arm, from-scratch
   regime, reported as its own row per rule 3).
2. Take each validation-loss-vs-epoch curve, map epoch $\mapsto [-1,1]$, fit
   `JacobiGP` with learnable $(\alpha,\beta)$ via the Exp-4 evidence path.
3. **H6a:** the learned $(\alpha,\beta)$ of an *under-ranked* run (curve
   forced to wiggle late) separates from an *over-ranked* run (rising tail)
   **before** the val curve itself shows it — the exponents as early-warning
   gauge of rank mis-set.
   **H6b:** a threshold rule on $(\alpha,\beta)$ selects $r$ within ±1 octave
   of the oracle-best $r$ on held-out model/data pairs, beating grid search at
   equal compute.
4. Negative result is first-class — both repos now carry that reflex
   (Exp 4's pure-ML ridge; MEF's retractions). If the exponents track nothing
   beyond the curve tail, we say so, and §5 proceeds without the gauge.

## 5. Experiment 7 (the flagship): boundary-weighted spectral LoRA

The transplant. Give the LoRA/atom coefficient prior a Jacobi weight — atoms
now indexed not by σ-rank but by a **semantic boundary coordinate** $u$:

$$
\Delta W = \sum_{n} c_n\, U_n V_n^\top, \qquad
c_n \sim \mathcal{N}\!\big(0, \lambda_n(\alpha,\beta)\big), \qquad
\lambda_n \propto \int_{-1}^{1} P_n^{(\alpha,\beta)}(u)^2\, (1-u)^\alpha (1+u)^\beta\, du
$$

$u$ per atom: e.g. alignment of $V_n$ with the in-/out-of-distribution split
measured on probe data (RTE-dev vs. a perturbed/OOD variant), or the atom's
effective-rank excursion measured by MEF's Stage-10 online monitor — the
infrastructure to compute $u$ already exists there. High $\alpha$: stiff at
the "familiar" end (protect known behaviour — the anti-forgetting dial, and
the honest home of the L6 residue: not a band, a *boundary*); low $\beta$:
supple at the "edge" (adapt where the data manifold ends). $(\alpha,\beta)$,
$\ell$, $r$ learned jointly by evidence or a variational proxy on probe
batches — the Exp-4 loop over the shape of a *weight* space.

**Predictions, stated to be killed if wrong:**

- **P1 (shape ≠ size):** at fixed $r$, learned $(\alpha,\beta)$ improves
  two-domain retention vs. plain LoRA *and* vs. isotropic-spectral LoRA (the
  decisive ablation: same $\lambda$, no boundary weight). If only $r$ matters,
  the merged thesis dies and we write the obituary — MEF has practiced this.
- **P2 (boundary dial):** forced-negative test — $\alpha \to$ high should
  measurably reduce forgetting on domain A while fine-tuning on domain B:
  Exp 1's boundary-condition result restated in weight space.
- **P3 (cheap gauge):** H6a's exponents predict the P1 optimum, warming the
  full search.

## 6. Naming and papers

Working name **ONE SPACE** (one space, three knobs; three benches, one
epistemology) — **christened CHORA** (χώρα: the receptacle that must be
shaped before anything can be poured into it; Letter 003): the programme is
not about the manifold, it is about the *choice of vessel*, and

> the shape of the container is the knowledge.

The shared workshop lives at `~/Programming/code-2026/chora` (root brief +
hash-pinned store). The tagline, earned in the 花絮 and a *scientific* sentence,
not decoration:

> *JacobiGP: to see the world — in Westworld there are boundaries; in the
> functional world there are no limits. We choose the boundaries to see the
> beauty.* — and choosing is done by the evidence, never by hand.

Publication split: JacobiGP = the auto-constructed *function* space, negative
results included; Middle-Eigen-function = the measured *weight* space of real
LLMs and what its σ-axis cannot say; Sarcos-NN-Model = the controlled fast
bench where predictions are registered before they are killed;
Polynomial-Activated-NN = shape inside the layer; the joint contribution is
§5: **inductive bias as a learnable measure on update directions — location
in domain space, not rank in spectrum space.**

## 7. Sequence

1. Merge: extract `spaces/` + `infer/` from all three repos; keep JacobiGP
   exp1–5 green (pills must not dim), MEF's stage scripts importable, and
   Sarcos's split loader canonical while all three fronts move onto the
   shared core.
2. Publish artifacts into `models/` (MEF spectra/atoms; Sarcos run records
   incl. `weights_init.npz`; this repo's `results/*.json` stay as they are).
3. Exp 6 — Sarcos pilot first (afternoon), MEF rig second (evening); it either
   funds §5's confidence or spends itself honestly on a null.
4. Exp 7 — flagship; P1 → P2 → P3, small models first. Exp 7.5 (JacobiGP on
   SARCOS at physical joint limits) can run in parallel — it needs no LoRA at
   all, only `data.py` and `infer/`.
5. One site, three histories: this builder absorbs the two Quarto reports or
   links them — decided after the merge, not before.

*Reality is all the $(\alpha_i, \beta_i)$ superposed; a project is the act of
picking two of them and being answerable for the choice. This page is that
act, written down before the data gets a vote.*
