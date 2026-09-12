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

**Question.** Letter 006: left free, PolyNN's per-neuron (α,β) walked from
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
