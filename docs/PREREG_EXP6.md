# PREREG — Experiment 6: learned exponents as a rank gauge (H6a / H6b / H6c)

**Registered:** 2026-09-12, 00:4x UTC (the two-machine day, machine A / `m1pro-32g`)
**Registering hand:** The Geometer (JacobiGP bench), chora root session named by Letter 016
**Status: paperwork before numbers. No number in this document exists yet; none
of the three checks may be scored against data that predates this commit
(exceptions named per check).**

Anchors (bytes and minutes this text consumes, cited as law 2 demands):

- Concept: `docs/NEXT.md` §4 (H6a/H6b intent, verbatim); §1 three-knob table.
- Room mandate: `chora/meetings/2026-09-11-003-lora-critical-windows.md`
  @`chora` — R1 (A-jacobi-evidence enters *as H6c inside exp6's document*),
  T01–T04 turns, R4 (init-measure predictive; post-training fits carry
  `post-hoc` label forever).
- Method: `docs/MATH.md` §6 — evidence fit is
  argmax[log p(y|θ) + log N(0,s²) on u_t = log(e^{t+1}−1)], s = 2, never raw
  evidence maximisation; §6.1 — **α acts on the right edge (x=+1), β on the
  left** (AGENTS.md §3/§5 have them swapped; this document follows MATH.md).
- H6c's donor data: Letter 006 `(chora copy sha256 5916687d996e23d3…)`;
  `artifacts/results/polynn/exp8_summary.json` `b3eaccf6e366…`,
  `noise_band.json` `e685bce134ce…`.
- Apparatus bands, not travel (Letter 014 amendment 2): MEF's ±0.030 is
  MEF's; exp8's ±0.938 pp is PolyNN's; each check below states the band it
  may use and where a new band must be calibrated in place.

## Discipline (programme law 4, restated for this document)

1. One hypothesis check per number; H6c's pair is **one** check, not two
   (Sitting 003, T01).
2. Regimes in separate rows, forever: *post-hoc-truncation*,
   *training-under-constraint*, *frozen-base-increment*, and — new row
   introduced here — *init-measure* (predictive by R4; anything fit after
   training is description and is labeled `post-hoc` in every artifact).
3. Never tune on test. The rule of H6b is frozen before it sees a single
   held-out oracle; retained energy reported next to every curve-fit.
4. Negative results are first-class: each check below names its own
   obituary, written now.

---

## H6a — the exponents as an early-warning gauge of rank mis-set

**Apparatus.** MEF's LoRA rig (Qwen2.5-0.5B + RTE, paired seeds,
frozen-base-increment regime), ranks r ∈ {2, 8, 32, 64} (NEXT.md §4.1);
plus the sanctioned cheap pilot on Sarcos MLPs (from-scratch regime, its own
rows, seeds 13/14/15, pinned `data.py` split). Run meta tagged `run-on:`
(machine), per Letter 016 law 3.

**Fit.** Each validation-loss-vs-epoch curve, epoch ↦ [−1, 1], fit by the
Exp-4 evidence path (§6 prior, s = 2, ℓ, σ², σ²_noise free).

**Prediction (falsifiable form).** Take cumulative prefixes of each curve at
checkpoints τ ∈ {25%, 50%, 75%} of the run. Define under-ranked group
U = {r=2,8} (curves forced to wiggle late) and over-ranked group O =
{r=32,64}. At τ = 50%:

> **(H6a)** the between-group separation of learned (α̂,β̂) at τ — measured
> as distance of group means divided by the *pooled within-group seed spread*
> in (α,β) — exceeds **3.0**, while the val-loss curves themselves are still
> within their paired-seed noise band at τ (separation of the *metric* they
> are supposed to predict has not yet occurred). If U/O coincide with the
> bands at 0.030 by τ = 50%, the curves have already separated and H6a is
> dead: the gauge arrived after the event, which is the obituary sentence.

**Obituary if false:** "the exponents track nothing beyond the curve tail"
(NEXT.md §4.4) — filed as said, and §5 (exp7) proceeds without the gauge.

## H6b — a threshold rule that reads rank

**Apparatus.** Same curves, same fits, held-out model/data pairs: at least
8 pairs not used to write the rule (3 Qwen2.5 sizes × RTE variants + Sarcos
pilot pairs; composition frozen in the run script before scoring).

**The rule (frozen before first scoring, per this text).** From the fit on
the τ = 100% prefix of a *short probe* (first 30% of epochs) at a single
pilot rank r_p = 8: read (α̂,β̂); g(α̂,β̂) selects the next r from
{2,8,32,64} by thresholds on α̂+β̂ and α̂−β̂ whose values are written into
the H6b run script **before any pair is scored** and may not be edited
afterward (rule-text hash committed; edits after = new row, `post-hoc`).

> **(H6b)** g selects an r within ±1 octave of the oracle-best r (oracle =
> argmin final val loss over all four ranks, full runs) on ≥ 6 of 8 held-out
> pairs, at total probe compute strictly less than the 4-rank grid search.
> Both numbers reported even when the rule fails on < 6.

**Obituary if false:** the gauge cannot *prescribe*, only *describe*; H6a
(detection) stands or falls on its own row; exp7's P3 ("cheap gauge
warms the search") is then struck from NEXT.md, honestly.

## H6c — the same dial read from two ends (A-jacobi-evidence, adopted per R1)

**Question.** Letter 006: left free, PolyNN's per-neuron (α,β) — **[corr 2026-09-12, Amendment A3:
"per-neuron" is wrong. `models.py` L5 at `PolyNN@0eb2930` reads "c per neuron, (alpha,beta) per
layer", and every epoch of the walk bytes carries exactly one pair — the arm learns ONE pair per
layer, shared by its neurons. The check below is unaffected: it scored a mean pair against a mean
pair, and its verdict stands either way. Corrected at the site of the error, claim untouched; the
misstatement's consequence is Letter 023 §1–§3]** — walked from
Legendre (0,0) to ≈ (0.40, 0.37) in every seed. Does the GP's *gradient-free
evidence path*, shown only the **initial** geometry of the same container,
walk to the same place *before training*?

**Input (named bytes, none of which exist yet — this is the ordering
clause).** PolyNN h=128 **init pre-activation dumps**: per-arm histograms of
the tanh-squashed pre-activations at init (epoch 0), all four arms
(A-relu, A-jacobi, A-hermite, A-cheby), 5 seeds, reconstructed from the
pinned init states in `artifacts/init-states/`; and the **per-seed α,β
walks** (Letter 014's homework, Letter 016 assigns both to machine B).
Both cross only with manifest entries (path, sha256) — R4. The **primary
arm** for the prediction is A-jacobi (the arm that learned the pair); the
other three arms are fitted and reported as a contrast row, not as extra
checks.

**Coordinate (Anatomist's demand, adopted).** The fit consumes exactly
PolyNN's squashed coordinate: z ↦ tanh(z) mapped affinely to [−1, 1],
per-layer histograms, weights = histogram counts, no re-squashing, no
re-scaling. Same squash, same domain, or the dial is not the same dial
(Joiner's T04).

**Fit.** Exp-4 evidence path, §6 prior (s = 2), on the *combined* init
histogram of the A-jacobi h=128 arm, per seed; report α̂_s, β̂_s for the 5
seeds and the mean pair (α̂, β̂).

**Band (the Warden's ruling, executed as mechanics).** Provisional band
**±0.2** as drafted in T01/R1, to be *widened by rule and never narrowed*:
when PolyNN's per-seed walk bytes arrive, before any fit runs, freeze
band = ±max(0.2, 2·sd_seed), with sd_seed the across-seed standard
deviation of the learned terminal values (per coordinate: α uses α's, β
uses β's). The re-set is mechanical, dated, and committed as an
amendment to this file *before* the first H6c number exists; if the walks
never arrive, the ±0.2 stands and the check runs on B's dumps alone.
(Joiner's note honored: point estimates 0.40/0.37 were means; direction
agrees, detail may not.)

> **(H6c — one check for the pair)** On the initial squashed pre-activation
> measure, the evidence fit's mean pair satisfies |α̂ − 0.40| ≤ band AND
> |β̂ − 0.37| ≤ band, i.e. lands off Legendre, where gradient descent later
> chose to live. The joint conjunction is the single hypothesis check;
> per-coordinate halves are reported but are not separate claims.

**Obituary if false:** if (α̂,β̂) lands at (0,0)±band, or wanders negative —
"the same dial read from two ends" (Letter 006's echo, NEXT.md §2b/Sitting
003 PS) **dies in the room where it was born** (T01). Filed as a
first-class negative; the evidence optimizer keeps its Exp-4 standing on
its own data.

**Regime note (R4).** Every post-training fit of the same measure, if ever
run, is a description row labeled `post-hoc` permanently; it can neither
rescue nor upgrade H6c.

---

## Compute budget (measured not feared; Letter 006's correction applied)

- H6a/H6b: no new training beyond MEF's existing rank sweeps **if the saved
  val curves exist**; else 4 ranks × 2 seeds ≈ MEF stage-schedule cost
  (Anatomist to confirm byte source; the run script will refuse to start
  without a named curve file). Sarcos pilot: minutes/arm (Sarcos's own
  published timings).
- H6c: 5 fits of an N ≤ 64 basis, M ≤ 256-bin histogram — seconds on A;
  the real cost is B's ~11 min dump job, already scheduled (Letter 016).
- This document claims no wall-clock it has not seen; the run scripts
  report it after.

## Sequencing (Letter 014 queue + Warden's "it goes inside")

exp6 paperwork (this file) → H9-S Step 0 (Sarcos; B is running it) → exp9
registration (PolyNN, if K-1 is adopted at home) → Sarcos pilot of the
gauge → MEF rig of the gauge → H6c fit when dumps land. Nothing in this
document jumps ahead of anything; it is the paperwork the queue was waiting
on.

*Signed: The Geometer (JacobiGP bench) — registered at this commit; the
first number may not predate it.*

---

# Amendments

## A1 (2026-09-12, 17:4x +0800, machine A) — H6c: the band re-set by rule, the pointer named, and the fit's open mechanics frozen **before the first number**

**Filed by:** a `chora` root session at the human's word (*"go on at exp series"*), holding the
Geometer's hand. **Ordering clause honored:** this text is committed first;
`chora/experiments/exp6_h6c_fit.py` is written after it and **refuses to run** unless this
amendment is present in `docs/PREREG_EXP6.md` at `git HEAD`. No H6c number exists at this moment:
`chora/artifacts/results/jacobigp/` does not exist, and Letter 020 says so on the record —
*"no H6c fit ran on this laptop"* (B), and A has run none (the eight-hour rebase freeze ate
exactly the hours in which this fit was scheduled to happen).

### 1 · The band, by the rule this file already wrote

The walks have arrived, so H6c's Band clause fires: *band = ±max(0.2, 2·sd_seed), per coordinate,
frozen before any fit runs.* Computed independently by this hand from the crossing bytes —
`artifacts/results/polynn/h6cB/h6c_walks_summary.json` `(sha256 bd58e62bb8d1ab85…, 18,043 B)`,
whose `terminal_values` reproduce B's reported `sd_seed_sample` to six decimals, so the two
machines agree on the donor statistic itself:

| coord | sd_seed (sample, ddof=1) | 2·sd_seed | **band** |
|---|---|---|---|
| α | 0.061500 | 0.123001 | **± 0.200000** |
| β | 0.041281 | 0.082562 | **± 0.200000** |

**The provisional band stands: ±0.2 on both coordinates** — now frozen *by rule* rather than by
default, and therefore never to be narrowed. Robustness of the convention is stated so it cannot be
argued later: with the population sd (ddof=0) the doubled values are 0.110015 / 0.073845, and under
either convention both coordinates stay below 0.2, so **no sd convention available on these five
numbers changes the band.** Widening is impossible here: the rule takes a maximum with 0.2.

### 2 · The pointer (Letter 020: "a prediction is a pointer to bytes") — the claim is **not** moved

The registered claim reads `|α̂ − 0.40| ≤ band AND |β̂ − 0.37| ≤ band`. **Those two literals stay the
primary pointer, unchanged**; an amendment may set a band and freeze mechanics, it may not move a
target after the donor data has been read. What this section does is *name* the ambiguity B found,
and pre-declare the secondary reading so no future hand can choose between them after seeing an
answer:

| pointer | value | what it is |
|---|---|---|
| **primary** (decides H6c) | **(0.40, 0.37)** | the literals as registered in this file |
| secondary (reading row, decides nothing) | (0.358882, 0.358988) | A's own across-seed **mean** terminal pair over the same committed exp8 bytes — the mean of the very `terminal_values` used for the band |

The two pointers differ by (0.041118, 0.011012), so they can disagree about a verdict only inside a
**0.041-wide window** on α and a **0.011-wide** one on β. Pre-declared handling, so it is not a
future judgement call: if primary and secondary readings return different verdicts, H6c is reported
**AMBIGUOUS-POINTER, verdict taken from the primary row**, and the disagreement itself is filed as a
first-class finding — against the *paperwork*, not the dial. The seed-1000 cell of this arm is
(0.39773887, 0.36774677): the registered literals are that single **cell**, while this file's own
Joiner note ("point estimates 0.40/0.37 were **means**") says the opposite. Both spellings are now in
the record before any fit sees either.

### 3 · The fit's open mechanics, frozen (the registered text leaves these to the script)

Registered apparatus, reused not reinvented: `src/jacobigp` at `JacobiGP@ecf67d6`, the **Exp-4
evidence path** exactly as `experiments/exp4_evidence_learning.py` implements it, with `MATH.md` §6's
prior.

| degree of freedom | frozen value | why it is the registered one |
|---|---|---|
| x | the 256 bin **centres** of the [−1,1] 1/256 grid | H6c's Coordinate clause: same grid, no re-scaling |
| y | `combined_all_layers.counts` **verbatim** (integers; sum 7,040,000 at seed 1000) | "weights = histogram counts, no re-squashing, no re-scaling" |
| measure | `combined_all_layers` (all layers pooled), per seed, arm **A-jacobi** h=128 | "the *combined* init histogram of the A-jacobi h=128 arm, per seed" |
| seeds | 1000–1004 (B's five) | Letter 016's assignment; R4's named bytes |
| N | **30** deciding; **64** pre-declared sensitivity row | registered ceiling is N ≤ 64; exp-4's own validated N is 30 — the deciding row reuses the measured apparatus |
| spectrum | `se` | exp-4's `SPECTRUM` |
| free parameters | (α, β) + nuisance (`lengthscale`, `sigma2`, `noise`), inits 0.4 / 1.0 / 1e-3 | exp-4's `new_gp`, `NUISANCE` |
| objective | MAP: log-evidence + `log N(0, 2²)` on the unconstrained softplus coordinates | `MATH.md` §6 verbatim, s = 2 |
| starts | exp-4's six `INITS`, best objective wins | exp-4's multi-start; §6.2's ridge is why it exists |
| dtype | float64 | exp-4's `DT` |
| **the check scores the mean pair (α̂, β̂) over the five seeds; nothing else** | — | "one check for the pair"; per-seed and per-coordinate numbers are reported, not scored |

**Contrast rows, never scored:** the other three arms (A-relu, A-hermite, A-cheby) are fitted with the
identical rig and reported as a row. Letter 020 already killed their *contrast* function — all four
arms' init histograms are integer-identical, because `z = fc_in(x)` is measured before the activation
exists — so on this donor data the contrast row is a **duplicate-check, not a falsifier**, and is
filed as unreachable-as-registered rather than quietly dropped. It is still run, because
"the four arms agree" across two laptops is evidence about the *rig*.

**Reported next to every number** (law 4's retained-energy clause, in its evidence-fit form): the
fitted `sigma2`, `noise`, `lengthscale`, the achieved MAP objective, and the fraction of histogram
mass the fit explains. An (α̂, β̂) that arrives with `sigma2` collapsed toward §6.2's ridge is **the
gauge failing, not the data cooperating**, and is filed as such.

**Declared non-movements, so tonight's answer cannot be improved afterwards:** no re-start-set
tuning, no bin-subset trimming, no re-scaling or log-count variant, no narrowing of ±0.2, no choosing
the secondary pointer over the primary, no swapping the N=64 row into the deciding slot. If a fit does
not converge, or lands within 0.02 of the barrier at −0.98 on either coordinate, that outcome is
reported unaltered — the obituary this file already wrote ("the same dial read from two ends dies in
the room where it was born") is the pre-registered text for exactly that case.

*Signed: The Geometer's hand (chora root session, machine A), for the chair; H6c's first number may
not predate this commit, and this commit predates it by construction — the run script checks.*

## A2 (2026-09-12, 18:4x +0800, machine A) — H6a's sanctioned cheap pilot on Sarcos: the rungs, the groups, the estimators and the two clauses, all frozen before the first curve is fit

**Filed by:** the same hand as A1 (a `chora` root session at the human's word, holding the
Geometer's seat), one commit after H6c's negative (`JacobiGP@99c7a4a`, `chora@015aad0`).
**Scope:** this amendment registers **the pilot row only**. H6a proper, on MEF's LoRA curves, stays
as registered and unmeasured; this text cannot score it and its result cannot strike it.
**Ordering:** enforced by code again — `chora/experiments/exp6_h6a_sarcos_pilot.py` refuses to run
unless the marker strings of this section are in `docs/PREREG_EXP6.md` at JacobiGP's git HEAD, and
refuses a second draw if its verdict file exists.

### 1 · Regime row, and the one design fact that had to be said before any number

The pilot runs in Sarcos's **training-under-a-rank-constraint** regime (from scratch, each layer
capped at rank r; `--mode constrained`). Per Sarcos's own rule 3 it is **its own row, forever**: it
is never merged with the *post-hoc-truncation* rows or with the `lora` **frozen-base-increment**
rows already sitting in the same `results/runs/` tree, and no number here is compared against one
from those rows without naming which regime each arm came from.

Measured in this amendment, not assumed: **on Sarcos the seed moves the split as well as the
initialisation** — `run.json → split.seed` equals the run seed for every one of the 22 committed
constrained runs, and the val-set size changes with it (4,373 rows at seed 13; 4,608 at seed 14).
Therefore *the within-rung spread this pilot pools is split ⊕ init, not init alone*, and the phrase
"pooled within-group seed spread" in H6a means something weaker here than it does on MEF's paired
rig. The design is nevertheless **paired by seed across rungs** (every rung is run at all three
seeds), so the U/O contrast holds split fixed against the comparison even though it does not hold it
fixed within a rung. This sentence is in the amendment because if it were not, the pilot's effect
size would silently be read as comparable to MEF's — and it is not.

Never read for anything in this check: the test split. `run.json` records test metrics because the
runner always records them; this script consumes `history[*].mse_val` and nothing else (Sarcos rule 2).

### 2 · The ladder, frozen

| field | frozen value | note |
|---|---|---|
| rungs | r ∈ **{1, 2, 4, 8, 16, 21}** | per layer, last layer capped `min(r, 7)` — Sarcos's own convention (`ranks16-16-7`); 21 is the input-dim cap, so r = 21 is where the constraint stops binding, and that is a *fact about this dataset*, stated in §4 |
| groups | **U = {1, 2}**, **O = {16, 21}**; {4, 8} are middle rungs, reported, in **neither** group | frozen here; the middle rungs may not be recruited into either group afterwards, in either direction |
| widths | primary **21-64-64-7** (the published canonical choice); sensitivity **256h256** | separate rows; the primary decides |
| seeds | **13, 14, 15** (split ⊕ init, §1) | the registered three |
| epochs / lr / batch / optim | 60 / 1e-3 / 256 / Adam, loss = mse on standardised targets | identical to the 22 committed runs, so the audit in §6 is meaningful |
| device | CPU, `torch.use_deterministic_algorithms(True)`, single thread | Sarcos rule 5 |
| curve | `history[e].mse_val`, e = 0…59, **validation only** | the metric the bench selects on |

### 3 · What is fit, and how the check is computed (every estimator written as a formula, so the script has no freedom left)

* **Prefix.** At τ ∈ {0.25, 0.50, 0.75} the fit sees epochs 0 … ⌈τ·60⌉−1 (15 / 30 / 45 points).
  **The registered check is at τ = 0.50**; τ = 0.25 and 0.75 are reported as description and score
  nothing.
* **Coordinate.** epoch index e ↦ x = 2e/(E−1) − 1 over the prefix (E = prefix length), so each
  prefix spans [−1, 1] — H6a's "epoch ↦ [−1,1]" applied per prefix, as registered.
* **y.** the `mse_val` values **verbatim**, no re-scaling, no log, no smoothing. Amplitude here is
  O(10⁻²–10⁰) and the target is smooth and monotone — which is the regime `MATH.md` §6's prior was
  calibrated for, unlike H6c's 7.04×10⁶-count measure. Recorded so that tonight's corner result is
  not silently blamed on the objective, and equally so that a *good* reading here cannot be claimed
  as a vindication of the rig.
* **Fit.** A1 §3's rig verbatim: Exp-4 evidence path, N = **30** deciding (N = 64 sensitivity row),
  spectrum `se`, nuisance (ℓ, σ², σ²_noise) free, MAP with `log N(0, 2²)` on the unconstrained
  softplus (α, β), exp-4's six INITS with the best objective winning, float64. Barrier rule carries
  over: any coordinate within 0.02 of −0.98 is flagged and reported unaltered.
* **Within-rung spread** (per coordinate c ∈ {α, β}, per τ): sd²(r, c) over the three seeds,
  ddof = 1. **Pooled spread:** s_c(τ)² = mean over the four rungs of U ∪ O of sd²(r, c);
  s(τ) = √( (s_α² + s_β²) / 2 ).
* **Group means:** m_g,c(τ) = mean of ĉ over the two rungs of g and the three seeds (n = 6).
* **The separation statistic:** **S(τ) = ‖m_U(τ) − m_O(τ)‖₂ / s(τ)**. The per-coordinate halves
  |d_c|/s_c are reported and score nothing (one check, as in H6c).
* **The curve-side clause**, which is what actually kills H6a, frozen as a formula: at the last
  epoch E_τ of each prefix, Δ(τ) = |mean_U mse_val(E_τ) − mean_O mse_val(E_τ)| (means over the same
  six arms), against the band B(τ) = 2 · √( mean over the four rungs of U ∪ O of the within-rung
  sample variance of mse_val at epoch E_τ ).

> **(H6a-pilot, one check)** the pilot row **HOLDS** iff **S(0.50) > 3.0** AND **Δ(0.50) ≤ B(0.50)**.
> If Δ(0.50) > B(0.50) the curves have already separated and H6a's own death-clause fires — *"the
> gauge arrived after the event"* — whatever S does. Both numbers are printed in every case.

**Sensitivity, pre-declared, cannot change the verdict:** S at τ = 0.25 / 0.75; the 256h256 row; the
N = 64 row; the per-coordinate halves. **Obituary** is H6a's registered one, verbatim: *"the
exponents track nothing beyond the curve tail"*, filed as said, with §5 (exp 7) proceeding without
the gauge.

### 4 · What a HOLD and a FAIL would each license, written now

* **HOLD** here licenses exactly one thing: running the MEF rig of H6a (r ∈ {2, 8, 32, 64}, paired
  seeds, ±0.030 band), whose donor curves **do not exist in the record** — checked at 18:3x:
  `outputs/multi_model/summary.json` carries final scalars only (no `curve`, no `val_loss`, no rank
  keys), and `stage18_kairos_mini/base_run.json`'s `curve` is a 20-point **per-k** trace, not
  per-rank. So the pilot is the gate that decides whether that training gets done at all.
* **FAIL** here is filed against the *pilot row* and does **not** strike H6a on MEF's curves — but
  it does raise their price, and if the failure mode is again the walk to α,β → −1, that is two
  independent instances of §6's ridge in this programme's own data and the gauge idea goes to the
  room rather than to the next rung.

### 5 · Budget, measured not feared

The 22 committed constrained runs report `wall_seconds` 3.35–6.48 (CPU, single thread). The pilot
grid is 6 rungs × 3 seeds × 2 widths = **36 arms ≈ 4 min of training**, plus ≈ 90 evidence fits at
the ~4–12 s/fit A1 measured → **under 25 min end to end on A**. The registered text's promise
("Sarcos pilot: minutes/arm") is confirmed against those records rather than repeated.

### 6 · A free determinism audit, demanded of the runner before any fit

Sarcos's `train.py` claims *"a rerun with the same --seed and split args reproduces the record
bitwise."* Twenty-two of the pilot's arms share a configuration with a committed run (rungs
{1, 4, 16, 21} × seeds {13, 14, 15} × both widths). The pilot therefore **re-runs those
configurations under new run-ids and requires the `history` arrays to be bitwise identical** to the
committed ones, and reports every mismatch instead of quietly keeping the newer file. If that check
fails, the pilot's donor curves are not what its own bench thinks they are, and that is the
headline of the letter regardless of what S says.

*Signed: The Geometer's hand (chora root session, machine A), for the chair; the pilot's first
number may not predate this commit.*

## A3 (2026-09-12, 19:1x +0800, machine A) — a correction to §H6c's *Question*, not to §H6c

**What:** the phrase "PolyNN's **per-neuron** (α,β)" is false of the rig it names.
`models.py` L5 at `PolyNN@0eb2930`: `c per neuron, (alpha,beta) per layer`; confirmed from the other
side by the donor bytes, whose `walk[]` entries carry exactly one `alpha` and one `beta` per epoch
(`artifacts/results/polynn/h6cB/h6c_walk_jacobi_h128_s1000.json` and its four siblings). The A-jacobi
arm therefore learns **one pair per layer**, shared by all h=128 neurons of that layer.

**What it changes:** nothing registered. The Input, Coordinate, Fit, Band, the conjunction itself, the
obituary and the regime note all read on a *mean pair*, and were scored against a mean pair; the
verdict (`FAILS`, `(α̂,β̂) = (−0.922628, −0.916317)` at band ±0.200000,
`artifacts/results/jacobigp/exp6_h6c/h6c_verdict.json` `c51e001a3b1d…`) stands as measured. The
inline bracket sits at the site of the error rather than replacing the sentence, per the archive's own
rule that corrections travel as new text and the mistake stays where it was made.

**What it costs:** one sentence in `docs/NEXT.md` §2c and any future reader's expectation. A rig whose
shape field is *per neuron* has 128·2 parameters per layer where this one has 2 — that is a different
object, a different budget under PolyNN's own "coefficient budget is billed as architecture" law, and
a different experiment. Naming it is the substance of **Letter 023** (chair→PolyNN cc all), which
raises it as a proposal: if the shape field is allowed to vary across neurons or across regions of the
domain, the programme's "shape" knob stops being a parameter and becomes a **field** — i.e. it stops
assuming the space is flat enough for one chart. No number is claimed for that here, and R4 keeps it
from touching this check in either direction: a mixture-of-weights that fits better is a statement
about a different object, forever a new row.

*Signed: The Geometer's hand (chora root session, machine A), correcting its own Question paragraph;
the verdict was already in when the wording was found wrong, which is the order in which such things
are always found.*
