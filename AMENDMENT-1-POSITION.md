# marmoset-order-test, amendment 1: is the second-order advantage a position effect? (registered 2026-11-25)

MV2 of Munim, Inc. (mv2@munimtech.com). Made with AI agents working in his name; no human expert
has checked this amendment. Text CC BY 4.0, code Apache-2.0.

## Why this amendment

The registered test in [PLAN.md](PLAN.md) returned **SECOND_ORDER_SURVIVES**
([READING-MARMOSET.md](READING-MARMOSET.md)): the order-2 advantage of Bosshard et al. (2024) is
far beyond what session-specific first-order chains, with or without looped chunks, produce
(real D_best 303.20 nats; the largest of 499 looped-session worlds was 52.40; p 0.0020).

Those null worlds share one blind spot with the original first-order model. The authors pad every
sequence with '#' and add no end token, so the order-2 context ('#', call 1) tells the model it is
predicting the **second** call of a sequence, which the order-1 context (call 1) does not. If
marmosets follow call 1 with different calls at the second position than later in a sequence (a
positional template, not a dependency on the call two back), order 2 wins and so does our test.
The authors argue against a pure start-and-end role for chirps, but no test in the paper or in
PLAN.md separates position from second-order dependence.

## What has been seen before this amendment (full disclosure)

The real-data outputs of PLAN.md's job: results/marmoset_results.json and RESULTS-MARMOSET.md
(log-likelihoods per alpha for order 1 and order 2, the null summaries, the length counts 1: 4242,
2: 655, 3: 769, 4: 94, 5: 28, 6: 8, 7: 4, 9: 1). No position-aware statistic has been computed.
Nobody has looked at the call-by-position table of the data.

## The added model and statistic

**M1p (position-aware first order):** the context is the previous call plus whether the call being
predicted is the second call (t = 1) or a later one (t >= 2); the first call uses the start context.
M1p is exactly the order-2 model with every real call two back merged into one context, so
order 2 can beat M1p only through the identity of the call two back. Leave-one-sequence-out
log-likelihoods with the same add-alpha smoothing and alphas as PLAN.md.

- **Dpos(alpha) = LL2 - LL1p**; **Dpos_best** = best LL2 - best LL1p (primary statistic).
- **Position share** = (best LL1p - best LL1) / D_best (descriptive: how much of the published
  order-2 gain a position-aware first-order model already reaches).
- Leave-one-session-out Dpos(alpha) (descriptive).

## Null worlds (seed 20261127; 499 each)

Every real sequence is replaced by one of the same length from its session's **position-aware**
first-order chain: a start distribution, rows for the step into the second call and rows for later
steps, each session row shrunk toward the pooled row of its own kind with beta = 2 pseudo-counts.

- **H0c:** as above.
- **H0d:** as H0c, with looped sequences (smallest period p, length >= 2p) drawn as a chunk of
  length p and repeated, chains fitted on chunks (as H0b in PLAN.md).
- **H0c_beta10:** H0c with beta = 10 (sensitivity).

## Tests

- **T5 (beyond position and session):** p = (1 + #{H0c worlds with Dpos_best >= real}) / 500.
- **T6 (beyond position, session and loops):** the same against H0d.
- **T7 (calibration of the authors' test against position):** fire rate of the authors' reading
  (order 2 beats order 1 at all five alphas) in H0c worlds.
- **Power:** 200 H0d worlds in which a random 30% of the contexts with a REAL call two back get
  their own next-call row (Dirichlet 0.5); power = share with Dpos_best above the 95th percentile
  of H0d.

## Verdict rules (fixed now)

- p_T6 <= 0.05: **SECOND_ORDER_BEYOND_POSITION**.
- else p_T5 <= 0.05: **LOOPING_WITH_POSITION_EXPLAINS** if power >= 0.8, else
  BEYOND_POSITION_LOOPING_UNDECIDED.
- else **POSITION_EXPLAINS** if power >= 0.8, else UNDECIDED_LOW_POWER; suffix
  _SHRINKAGE_SENSITIVE if p_T5 under beta = 10 is <= 0.05.
- Fewer than 4 sessions: NO_BLOCKS.

The PLAN.md verdict (SECOND_ORDER_SURVIVES) stands as registered whatever this amendment returns;
this amendment answers a narrower question, and both are reported.

## Predictions (priors, for a Brier score)

| id | prediction | prior |
|---|---|---|
| P1 | order 2 beats M1p at all five alphas on the real data | 0.65 |
| P2 | p_T5 <= 0.05 | 0.65 |
| P3 | p_T6 <= 0.05 | 0.60 |
| P4 | position share of D_best >= 0.5 | 0.35 |
| P5 | the authors' test fires in >= 30% of H0c worlds | 0.40 |
| P6 | power >= 0.8 | 0.80 |

## Limits

Position has two levels only (second call; later calls), because sequences of four or more calls
are 2.3% of all sequences; a finer position split would mostly add empty contexts. Sessions, not
animals, are the blocks (as in PLAN.md). The data are read exactly as in PLAN.md and are not
redistributed; only summary statistics are published.

## Code and run

marmoset_position.py (numpy only, self-contained), test_marmoset_position.py (hand-traced unit
tests and an end-to-end run on a synthetic file), gate_a1.py (nothing is computed until this
amendment and its code on GitHub equal the job's copies), fetch_data.py (unchanged from PLAN.md).
One CPU job, 1 GB.
