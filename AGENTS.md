Here is a comprehensive `agents.md` blueprint designed to orchestrate a multi-agent system (or a structured research team) to implement, test, and validate the use of Jacobi polynomials as an automatic basis for Gaussian Processes. 

This document defines the roles, workflows, and specific experimental protocols to empirically prove the theoretical properties discussed in our dialogue.

***

# `agents.md`: Jacobi-Basis Gaussian Process Research & Testing Framework

## 0. Cross-repo partnership (ONE SPACE)

This repo is one bench of a joint programme. The other two:

- [`math4mad/Middle-Eigen-function`](https://github.com/math4mad/Middle-Eigen-function)
  — SVD ablation of real LLM weight matrices (Qwen2.5 + RTE, LoRA rigs).
- [`math4mad/Sarcos-NN-Model`](https://github.com/math4mad/Sarcos-NN-Model)
  — the fast controlled bench: small MLPs on SARCOS, band structure of both
  $W$ and the increment $\Delta W$, pre-registered predictions, and SARCOS
  itself — the GPML dataset, 21→7 robot-arm torques with **physical joint
  limits**.
- [`math4mad/Polynomial-Activated-NN`](https://github.com/math4mad/Polynomial-Activated-NN)
  (`~/Programming/code-2026/Polynomial-Activated NN `, trailing space real) —
  the shape knob taken *inside the layer*: learnable Jacobi/Hermite/Chebyshev/
  Bernstein polynomials as activation functions vs. ReLU on Fashion-MNIST
  (planning stage `2a109f2`, code not yet written). Its README's own thesis —
  adjusting $(\alpha,\beta)$ lets the activation "work in different functional
  spaces" — is ONE SPACE's third row implemented as a nonlinearity.

The joint spec reconciling all three is **`docs/NEXT.md`** — its three-knob
table (size $N$/rank, spectrum $\lambda_n$, boundary shape $(\alpha,\beta)$)
is the shared coordinate system, and the two negative verdicts are load-
bearing: MEF killed σ-position at fine scale in LLMs; Sarcos killed the
middle-band hypothesis on a pre-registered small-MLP test.

The joint workshop is **`~/Programming/code-2026/chora`** — named **CHORA**
(χώρα, Plato's shaped receptacle; Letter 003), git-initialized, holding the
shared model/data store (hash-pinned manifests), the artifacts directories,
and its own root `AGENTS.md` for joint sessions. *ONE SPACE* remains the
programme's working title; CHORA is its name.

Rules for any agent working here:
1. `docs/NEXT.md` is the single source of truth for merged claims. Read it
   before proposing anything about LoRA/rank/SVD; do not fork the idea into a
   new file.
2. Sister repos are **read-only**. Clone shallow, cite the commit SHA behind
   any claim taken from them. Never push to them from a session started here.
3. Communication across repos is by **letter, committed in git**: write to
   `docs/LETTERS/YYYY-MM-DD-<to-repo>-<topic>.md`, dated and signed by role
   (human / agent-model). The sister repo answers with its own letter file.
   No claim crosses repos that isn't anchored to a SHA or a `results/*.json`
   (or Sarcos's `results/summary.json`).
4. Experiment numbers are reserved: exp1-5 here; MEF stages there; Sarcos
   keeps its band/regime experiments there; **exp6 (rank gauge) and exp7
   (boundary-weighted LoRA) belong to the joint `one-space/` repo** once it
   exists, and must reuse both sides' harnesses (this repo's evidence
   optimizer; MEF's Qwen2.5+RTE LoRA rig and its ±0.030 paired-seed noise
   band; Sarcos's split-pinned loader as the cheap pilot before either).
5. Experiments enter the same discipline everywhere: a hypothesis check per
   number, negative results reported, never quietly dropped. Adopted from
   Sarcos explicitly: **pre-register predictions before running**, **keep
   training-under-constraint / post-hoc-truncation / frozen-base-increment
   regimes in separate rows**, **always report retained energy next to
   error**, **never tune on test**.

---

## 1. Project Objective
To implement a Gaussian Process (GP) using truncated Jacobi polynomials $P_n^{(\alpha, \beta)}(x)$ as a basis, and to empirically validate that altering the parameters $\alpha$ and $\beta$ automatically constructs different Reproducing Kernel Hilbert Spaces (RKHS) and weighted Sobolev spaces with distinct boundary behaviors and geometric properties.

---

## 2. Agent Roles & Personas

### 🧠 Agent 1: Theoretical Mathematician (`Dr. Ortho`)
**Role:** Mathematical formulation, theoretical validation, and hyperparameter constraints.
**Responsibilities:**
*   Define the exact mathematical formulation of the truncated GP: $f_N(x) = \sum_{n=0}^{N-1} w_n P_n^{(\alpha, \beta)}(x)$.
*   Specify the eigenvalue decay functions $\lambda_n$ (e.g., exponential for RBF-like, algebraic for Matérn-like) to ensure valid covariance matrices.
*   Calculate the theoretical RKHS norms and verify the derivative property: $\frac{d}{dx} P_n^{(\alpha, \beta)}(x) \propto P_{n-1}^{(\alpha+1, \beta+1)}(x)$.
*   Set strict mathematical bounds for $\alpha > -1$ and $\beta > -1$ during optimization.

### 💻 Agent 2: Core Implementation Engineer (`Code Weaver`)
**Role:** Software architecture, numerical computation, and GP inference.
**Responsibilities:**
*   Implement the basis evaluation using numerically stable libraries (e.g., `scipy.special.eval_jacobi`).
*   Construct the design matrix $\mathbf{\Phi}$ and the prior covariance matrix $\mathbf{K} = \mathbf{\Phi} \mathbf{\Lambda} \mathbf{\Phi}^T$.
*   Implement the Bayesian linear regression inference (posterior mean and covariance of weights $\mathbf{w}$).
*   Optimize the code for GPU acceleration (using PyTorch or JAX) to handle large $N$ (number of basis functions) and large $M$ (number of data points).

### 🧪 Agent 3: Experiment Designer (`Test Master`)
**Role:** Dataset generation, experimental protocol design, and metric definition.
**Responsibilities:**
*   Design synthetic datasets with known boundary behaviors (e.g., functions that vanish at boundaries, functions with boundary singularities).
*   Design a Physics-Informed test case (e.g., solving a 1D Poisson equation with Dirichlet boundary conditions).
*   Define evaluation metrics: Root Mean Square Error (RMSE), Boundary Error (error specifically at $x = \pm 1$), and Negative Log Marginal Likelihood (NLML).

### 📊 Agent 4: Visualization & Analysis Specialist (`Plot Smith`)
**Role:** Data visualization, RKHS sampling, and result interpretation.
**Responsibilities:**
*   Write scripts to draw prior and posterior samples from the GP to visually inspect the functional space.
*   Plot the weight function $w(x) = (1-x)^\alpha (1+x)^\beta$ alongside the GP variance to visualize boundary weighting.
*   Generate spectral decay plots (eigenvalues $\lambda_n$ vs. frequency $n$) for different $(\alpha, \beta)$ pairs.
*   Compile the final report comparing the empirical results against `Dr. Ortho`'s theoretical predictions.

---

## 3. Experimental Test Suite

`Test Master` will execute the following specific experiments to validate the "automatic functional space construction" hypothesis.

### Experiment 1: Boundary Behavior Validation
**Hypothesis:** Altering $\alpha$ and $\beta$ forces the GP to respect specific boundary conditions without explicitly coding them.
**Protocol:**
1.  **Setup:** Generate a target function $y(x)$ on $[-1, 1]$ that is zero at the boundaries but highly oscillatory in the middle.
2.  **Run A (Legendre):** Set $\alpha=0, \beta=0$. Observe GP behavior at boundaries.
3.  **Run B (Chebyshev 2nd Kind):** Set $\alpha=0.5, \beta=0.5$. The weight $w(x) \to 0$ at boundaries. The GP should easily fit the interior without being overly constrained by boundary points.
4.  **Run C (Negative Parameters):** Set $\alpha=-0.5, \beta=-0.5$. The weight $w(x) \to \infty$ at boundaries. The GP variance should artificially collapse at the edges, strongly forcing the prediction to zero.
**Success Metric:** The boundary error in Run C should be significantly lower than in Run A, while Run B should have lower interior error.

### Experiment 2: RKHS Geometry & Prior Sampling
**Hypothesis:** Changing $(\alpha, \beta)$ changes the "shape" of the functions the GP considers probable (the RKHS).
**Protocol:**
1.  **Setup:** Fix the eigenvalue decay $\lambda_n$ to be identical across all runs.
2.  **Action:** Draw 50 prior samples from the GP for three different parameter sets:
    *   Set 1: $(\alpha, \beta) = (0, 0)$
    *   Set 2: $(\alpha, \beta) = (2, 2)$ (Allows boundary flexibility)
    *   Set 3: $(\alpha, \beta) = (5, 0)$ (Asymmetric boundary behavior)
**Success Metric:** Visual inspection by `Plot Smith` must clearly show distinct geometric families. Set 3 should show asymmetric variance/bending compared to the symmetric Set 1 and 2.

### Experiment 3: Physics-Informed Derivative Space
**Hypothesis:** The derivative of the GP naturally lives in the Sobolev space defined by $(\alpha+1, \beta+1)$.
**Protocol:**
1.  **Setup:** Define a 1D differential equation: $-u''(x) = f(x)$ with $u(-1)=u(1)=0$.
2.  **Action:** Train a Jacobi-GP to predict $u(x)$ using $\alpha=0, \beta=0$.
3.  **Evaluation:** Analytically compute the GP's first and second derivatives using the Jacobi derivative property. 
4.  **Comparison:** Compare the analytical derivative GP against a standard GP that uses numerical differentiation (finite differences).
**Success Metric:** The analytical Jacobi derivative should be smoother, require fewer basis functions $N$, and exhibit lower variance than the numerical derivative.

### Experiment 4: Automated Hyperparameter Learning
**Hypothesis:** The optimal functional space (best $\alpha, \beta$) can be learned directly from data via Marginal Likelihood maximization.
**Protocol:**
1.  **Setup:** Generate a dataset from a function with a known steep gradient at $x=1$ and flat at $x=-1$.
2.  **Action:** Initialize $\alpha, \beta$ randomly. Use gradient-based optimization (L-BFGS or Adam) to maximize the log marginal likelihood with respect to $\alpha, \beta$, and the length-scale $\ell$.
**Success Metric:** The optimizer should converge to a high $\beta$ (allowing flexibility at $x=1$) and a lower or zero $\alpha$ (keeping it flat at $x=-1$).

---

## 4. Technology Stack & Tools

*   **Core Math:** `scipy.special.eval_jacobi` (for stable polynomial evaluation), `numpy.linalg`.
*   **GP Framework:** `GPyTorch` (for scalable GP inference and gradient-based hyperparameter optimization) OR custom `PyTorch`/`JAX` implementation for exact control over the basis matrix.
*   **Visualization:** `matplotlib`, `seaborn`.
*   **Agent Orchestration:** `CrewAI` or `AutoGen` (if automating the agents via LLMs), or standard Git/CI-CD workflows (if human-executed).

---

## 5. Workflow & Handoffs

1.  **Phase 1 (Math):** `Dr. Ortho` outputs the exact equations for the covariance matrix and the derivative mappings. -> *Handoff to Code Weaver.*
2.  **Phase 2 (Code):** `Code Weaver` builds the `JacobiGP` class, ensuring it passes unit tests for basic GP properties (symmetric positive definite covariance, correct posterior mean). -> *Handoff to Test Master.*
3.  **Phase 3 (Testing):** `Test Master` runs Experiments 1-4, saving raw data and model checkpoints. -> *Handoff to Plot Smith.*
4.  **Phase 4 (Analysis):** `Plot Smith` generates the visual reports. `Dr. Ortho` reviews the reports to confirm theoretical alignment. Final paper/report is compiled.

---

## 6. Definition of Done (DoD)

The project is considered successful when:
1.  A fully functional, numerically stable `JacobiGP` class is implemented and open-sourced.
2.  Experiment 1 conclusively proves that $\alpha, \beta$ act as boundary-condition controllers.
3.  Experiment 2 provides visual proof of RKHS geometry shifting.
4.  Experiment 4 demonstrates that $\alpha$ and $\beta$ can be successfully optimized via gradient descent, proving the "automatic" construction of the functional space from data.