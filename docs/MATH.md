# Phase 1 — Mathematical specification (`Dr. Ortho`)

Deliverable of the math phase: the exact objects that `Code Weaver` must implement
and that `Plot Smith`/`Dr. Ortho` later validate empirically. §1–§5 are the a-priori
specification; §6–§8 are the addenda that the experiments forced (each is *measured*, see
`docs/RESULTS.md`).

---

## 1. Basis: *orthonormal* Jacobi polynomials

Work on the reference interval $x\in[-1,1]$ with the Jacobi weight

$$
w^{(\alpha,\beta)}(x) = (1-x)^{\alpha}(1+x)^{\beta},\qquad \alpha>-1,\ \beta>-1 .
$$

The classical (Szegő-normalised) polynomials $P_n^{(\alpha,\beta)}$ satisfy

$$
\int_{-1}^{1} P_n^{(\alpha,\beta)} P_m^{(\alpha,\beta)}\, w\,dx = h_n\,\delta_{nm},\qquad
h_n^{(\alpha,\beta)}=\frac{2^{\alpha+\beta+1}}{2n+\alpha+\beta+1}
\frac{\Gamma(n+\alpha+1)\Gamma(n+\beta+1)}{n!\,\Gamma(n+\alpha+\beta+1)} .
$$

We use the **orthonormal** family

$$
\widehat P_n^{(\alpha,\beta)}(x) = P_n^{(\alpha,\beta)}(x)\,\sqrt{1/h_n^{(\alpha,\beta)}},
\qquad
\int \widehat P_n \widehat P_m\, w\,dx = \delta_{nm}.
$$

Why orthonormal: `eval_jacobi` plus $1/\sqrt{h_n}$ blows up in floating point once
$\alpha,\beta$ or $n$ grow ($h_n$ contains $2^{\alpha+\beta+1}\Gamma\Gamma/\Gamma$).
Instead the code evaluates $\widehat P_n$ directly with the **symmetric three-term
recurrence** (Gautschi, *Orthogonal Polynomials*, §3.2):

$$
x\,\widehat P_n = a_n \widehat P_{n+1} + b_n \widehat P_n + a_{n-1}\widehat P_{n-1},
\qquad \widehat P_0 = \frac{1}{\sqrt{h_0}},
$$

$$
a_n=\sqrt{\frac{4(n+1)(n+\alpha+1)(n+\beta+1)(n+\alpha+\beta+1)}
{(2n+\alpha+\beta+1)(2n+\alpha+\beta+2)^2(2n+\alpha+\beta+3)}},
$$

$$
b_0=\frac{\beta-\alpha}{\alpha+\beta+2},\qquad
b_n=\frac{\alpha^2-\beta^2}{(2n+\alpha+\beta)(2n+\alpha+\beta+2)}\quad(n\ge 1).
$$

This recurrence is built from `+ - * / sqrt lgamma` only, hence is **exactly
differentiable** by Autograd with respect to $\alpha$, $\beta$ *and* $x$.
That is what makes gradient-based learning of the function space (Experiment 4) possible:
measured agreement with central differences on $\log p(y)$ is $2\cdot10^{-9}$ (§6).

### Constraints for optimisation
$\alpha,\beta$ are parametrised as $\alpha = -1+\mathrm{softplus}(u_\alpha)$
(likewise $\beta$), so the admissible half-plane $\alpha,\beta>-1$ is enforced
*structurally*, never by projection or clamping. That is necessary but **not sufficient** —
see §6.1.

---

## 2. Truncated GP = Bayesian linear regression

$$
f_N(x)=\sum_{n=0}^{N-1} w_n\,\widehat P_n^{(\alpha,\beta)}(x),
\qquad w \sim \mathcal N(0,\Lambda),\ \Lambda=\mathrm{diag}(\lambda_0,\dots,\lambda_{N-1}).
$$

Induced (spectral) kernel:

$$
k_{\alpha,\beta}(x,x') = \sum_{n=0}^{N-1}\lambda_n\,
\widehat P_n^{(\alpha,\beta)}(x)\widehat P_n^{(\alpha,\beta)}(x'),
\qquad
K=\Phi\Lambda\Phi^{\top},\ \Phi_{in}=\widehat P_n(x_i).
$$

Because the basis is orthonormal **in the weighted inner product**, changing
$(\alpha,\beta)$ changes the measure, so $k$ changes in a way that is *not*
absorbable into $\lambda_n$: the RKHS as a set of functions changes
(Experiment 2: identical $\mathrm{tr}\,K$ and participation dimension, edge sd from 1.10 to
20.8).

### Eigenvalue schedules (smoothness / length-scale)
With $n_{\text{scale}}=1/\ell$:

| name | $\lambda_n$ (up to the amplitude $\sigma^2$) | analogue |
|---|---|---|
| `se` | $\exp\!\left(-\tfrac12 (n/n_{\text{scale}})^2\right)$ | squared-exponential / analytic |
| `exp` | $\exp(-n/n_{\text{scale}})$ | rough, exponential decay |
| `matern` | $(1+(n/n_{\text{scale}})^2)^{-\nu}$ | algebraic decay, $\nu$ controls smoothness |
| `sll` | $(1+n(n+\alpha+\beta+1)\ell^2)^{-\nu}$ | true Jacobi Sturm–Liouville eigenvalues |

`sll` is the theoretically clean option: $-\big(w\,u'\big)'=\mu\, w\,u$ has
$\mu_n=n(n+\alpha+\beta+1)$ for the Jacobi weight, so `sll` *is* a weighted
Sobolev-space definition $\|u\|^2_{H^\nu_w}=\sum (1+\mu_n\ell^2)|c_n|^2$.
(Experiment 3 shows the price of that cleanliness: an algebraic tail makes the analytic
$u''$ error *grow* with $N$ — §7.3.)

**Admissibility.** $\sum_n\lambda_n<\infty$ guarantees a trace-class covariance and
a valid GP (needed for $N\to\infty$); with `se` the sum is bounded by
$\sigma^2\sqrt{\pi/2}\,n_{\text{scale}}$.

---

## 3. Inference (closed form)

Data $y_i=f_N(x_i)+\epsilon_i$, $\epsilon\sim\mathcal N(0,\sigma_{\text{noise}}^2)$,
$\Phi\in\mathbb R^{M\times N}$:

$$
\Sigma_w=\Big(\Lambda^{-1}+\frac{1}{\sigma_{\text{noise}}^2}\Phi^{\top}\Phi\Big)^{-1},
\qquad
\mu_w=\frac{1}{\sigma_{\text{noise}}^2}\Sigma_w\Phi^{\top}y .
$$

Cost $O(MN^2+N^3)$ — independent of the test-set size after one factorisation, which
is the practical advantage over $O(M^3)$ kernel GPs when $M\gg N$.

Predictions at test points $\Phi_*$:
$\ \mathbb E[f]=\Phi_*\mu_w,\quad \mathrm{Var}[f]=\mathrm{diag}\!\big(\Phi_*\Sigma_w\Phi_*^{\top}\big)+\sigma_{\text{noise}}^2$
(with `noise=False` for the latent function).

Log marginal likelihood (evidence), computed in weight space:

$$
-2\log p(y) = \frac{\|y\|^2}{\sigma_{\text{noise}}^2}
+\mu_w^{\top}\Lambda^{-1}\mu_w-\log|\Sigma_w|+\log|\Lambda|
+M\log\big(2\pi\sigma_{\text{noise}}^2\big).
$$

The evidence is the objective used to learn $(\alpha,\beta,\ell,\sigma^2,\sigma_{\text{noise}}^2)$
(Experiment 4). Note it is the *only* object through which $(\alpha,\beta)$ talk to
the data — this is precisely the "automatic construction of the functional space".
The code uses the algebraically equivalent whitened form
$G=I+\Lambda^{1/2}\Phi^{\top}\Phi\Phi^{\top}\Lambda^{1/2}/\sigma^2_{\text{noise}}$,
whose eigenvalues are $\ge1$, so $\lambda_n$ may underflow to $0$ without producing `inf`.

---

## 4. Derivative spaces (the ladder property)

$$
\boxed{\ \frac{d^{k}}{dx^{k}}P_n^{(\alpha,\beta)}(x)
=\frac{(\alpha+\beta+n+1)^{\overline{k}}}{2^{k}}\,P_{n-k}^{(\alpha+k,\beta+k)}(x)\ }
$$

($r^{\overline{k}}$ = rising factorial; the term is $0$ for $k>n$). Hence for the
orthonormal family

$$
\frac{d^{k}}{dx^{k}}\widehat P_n^{(\alpha,\beta)}(x)
=\frac{(\alpha+\beta+n+1)^{\overline{k}}}{2^{k}}
\sqrt{\frac{h_{n-k}^{(\alpha+k,\beta+k)}}{h_{n}^{(\alpha,\beta)}}}\;
\widehat P_{\,n-k}^{(\alpha+k,\beta+k)}(x),
$$

with the norm ratio evaluated in log-space via `lgamma` (never as a quotient of gammas).

Consequences used in Experiment 3:
* the GP derivative is **another truncated Jacobi GP** with parameters
  $(\alpha+1,\beta+1)$ and $N-1$ terms — analytic, no finite differences;
  verified: $d^{k}\Phi$ lies in the $(\alpha+k,\beta+k)$ span to a *relative* residual of
  $1.5\cdot10^{-15}$ for $k=1,2$;
* the second derivative lives in $(\alpha+2,\beta+2)$, so the strong form of a
  $2^{nd}$-order BVP, $-\,\mathcal L u = f$, is evaluated pointwise exactly;
* $w^{(\alpha+1,\beta+1)}=(1-x)(1+x)\,w^{(\alpha,\beta)}$: differentiation converts a
  weighted space into one that is degenerate at the boundary by exactly one power —
  the algebraic shadow of boundary conditions.

---

## 5. Boundary behaviour, predicted a priori

From Mehler/Heine asymptotics at the edge, $\widehat P_n^{(\alpha,\beta)}(1)\asymp
n^{\alpha+1/2}/\sqrt{2^{\alpha+\beta+1}}$ while the bulk amplitude is
$\asymp n^{(\alpha+\beta)/2-1/4}$, so the marginal variance
$k(x,x)=\sum\lambda_n\widehat P_n(x)^2$:

* $\alpha,\beta<0$ (Chebyshev-type, $w\to\infty$ at the edge): variance **collapses**
  at $x=\pm1$ ⇒ predictions are pulled to the prior mean $0$ ⇒ *implicit Dirichlet*,
* $\alpha,\beta>0$ ($w\to0$): variance **grows** at the edge ⇒ boundary freedom,
* $\alpha=\beta=0$ (Legendre): no boundary preference,
* asymmetric $(\alpha,\beta)$ acts on the two edges independently.

These four statements are the hypotheses falsified/confirmed by Experiments 1, 2 and 4.

---

## 6. Addendum — the admissible set is open, and the evidence is unbounded on it

Everything in §1–§5 is correct, but §1's "*strict* bounds $\alpha,\beta>-1$" are not enough
for optimisation, because the set is **open** and the objective is **singular at its
boundary**:

1. $\mu_0=\int w=2^{\alpha+\beta+1}\Gamma(\alpha+1)\Gamma(\beta+1)/\Gamma(\alpha+\beta+2)$
   diverges as $\alpha\to-1$ (likewise $\beta$), so $\widehat P_n=P_n/\sqrt{h_n}$ shrinks
   toward $0$ in the interior: the prior variance can be driven to $0$ *at the data* by
   pushing a parameter to the boundary, at no cost in fit.
2. Conversely, growing $\alpha$ inflates $\widehat P_n(1)$, and the same normalisation lets
   $\sigma^2$ compensate any change of $(\alpha,\beta)$: along the ridge measured in
   Experiment 4 the evidence gains $9$ nats while $\sigma^2$ collapses $400\times$.
3. The corner $\alpha=\beta=-1$ ($\alpha+\beta=-2$) is where the recurrence coefficients
   themselves stop existing; the optimiser reaches it in floating point from inside, since
   each coordinate is protected only strictly.

Hence the *method* is not "maximise the evidence in $(\alpha,\beta)$" but

$$
(\alpha,\beta,\ell,\sigma^2,\sigma^2_{\text{noise}})
=\arg\max\Big[\log p(y\mid\theta)+\log p_{\text{prior}}\big(u_\alpha,u_\beta\big)\Big],
\qquad u_t=\log\!\big(e^{t+1}-1\big),
$$

i.e. a weakly-informative $\mathcal N(0,s^2)$ on the *unconstrained* coordinates. In
constrained coordinates its density vanishes at $t=-1$, so it acts as a logarithmic barrier
at the singular edge; $s=2$ puts 95 % of the mass on $\alpha,\beta\in(-0.98,2.9)$. With it
the optimum is interior, unique (6 starts agree to $10^{-5}$) and it tracks the boundary
signature of the data. Without it, Experiment 4 does not work — measured, not asserted.

### 6.1 Which parameter controls which edge

$(1-x)^{\alpha}$ is the factor that degenerates at $x=+1$, so **$\alpha$ acts on the right
edge and $\beta$ on the left**. `AGENTS.md` §3/§5 lists them the other way round; the
numerics follow the weight, and the mirror-symmetric targets of Experiment 4 are the direct
test: the learned pairs are $(1.368,-0.871)$ and $(-0.861,1.317)$.

---

## 7. Addendum — what the derivative really buys (Experiment 3, measured)

§4 is exact, but three qualifications belong in the statement of the result.

1. **A central difference of the posterior mean converges to the ladder derivative**
   (agreement to $0.1\%$ for $h\le10^{-3}$). So the claim is *not* "more accurate at the
   optimal step"; it is: no step size to choose, one basis evaluation instead of three, an
   exact posterior covariance, and operator rows that stay **linear in the weights** — which
   is what a Bayesian collocation solve needs.
2. **Finite differences must keep their stencil inside $[-1,1]$.** `basis.py` clamps $|x|$
   to $1-10^{-15}$ (exact *at* the edge, since polynomials are continuous), hence a node at
   $1+h$ collapses onto $1$ and the second difference degenerates into a first one — a fake
   "catastrophic cancellation" of order $h^{-1}$ near the boundary. Score F.D. on
   $|x|\le1-h$ only. (This was a real mis-measurement in the first run of Experiment 3.)
3. **How large $N$ may be is decided by the tail of $\lambda_n$, not by the basis.**
   $d^2/dx^2$ multiplies mode $n$ by $\sim n^2$, so an algebraic tail (`matern`, `sll`) makes
   the analytic $u''$ error *grow* with $N$ (measured: $1.03\to30.6$ from $N=16$ to $48$)
   while an exponential one (`se`) saturates ($0.93\to0.98$). A truncated spectral GP is not
   a spectral method: the truncation must be matched to the operator being applied.
4. The derivative sd is a *model*: calibrated once the basis resolves the truth (95 %
   coverage $0.97$ at $N=25$), wildly over-confident when it does not ($0.013$ at $N=8$,
   mean $z$-score $59$).

---

## 8. Addendum — exact LOO-CV, and one sign trap

Because the model is linear-Gaussian, leave-one-out is closed form and differentiable in
$(\alpha,\beta)$: with $A=(K+\sigma^2_{\text{noise}}I)^{-1}$,

$$
p(y_i\mid y_{-i})=\mathcal N\!\Big(y_i-\tfrac{[Ay]_i}{A_{ii}},\ \tfrac{1}{A_{ii}}\Big)
\qquad\text{(`JacobiGP.loo')} .
$$

The mean is $y_i-[Ay]_i/A_{ii}$, **not** $[Ay]_i/A_{ii}$ (which is the residual); dropping
the offset silently turns LOO into a nonsense criterion — it did here, and
`tests/test_model.py` now pins it against an explicit leave-one-out refit.
