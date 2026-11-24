# marmoset-order-test: pre-registered plan (2026-11-24)

MV2 of Munim, Inc. Made with AI agents working in his name; no human expert has checked this plan.
Text CC BY 4.0, code Apache-2.0.

## The claim

Bosshard, Burkart, Merlo, Cathcart, Townsend and Bickel (2024), "Beyond bigrams: call sequencing
in the common marmoset (Callithrix jacchus) vocal system", Royal Society Open Science 11:240218,
report that "the second-order Markov model appears to account for patterns in marmoset call
combinations best out of all models surveyed": in leave-one-sequence-out cross-validation the
second-order minus first-order log-likelihood differences "include only positive values" across
smoothing constants. It is the first second-order Markov result reported for a non-human
call system, and it is the benchmark a 2026 magpie preprint compares itself with ("matches the
highest complexity reported in any non-human semantic call system to date").

## Why it might not mean second-order rules

The data are 8,497 calls from eight captive marmosets, recorded in focal sessions. The authors'
cross-validation holds out one sequence at a time and fits one chain to all sequences pooled.

1. **Pooling.** If animals or sessions differ in their first-order habits, a single pooled
   first-order chain is wrong for each of them, and the call two back carries information about
   which animal or session a sequence comes from. A second-order model then wins in
   cross-validation although no animal uses a second-order rule. Holding out a sequence does not
   remove this: other sequences from the same session stay in the training data.
2. **Looping.** A sequence that repeats a short chunk (A-B-A-B) has each call fixed by the call
   two back.

## What has been seen before this plan (full disclosure)

- The paper's text (Europe PMC full text, PMC11537759): methods, results and data statement.
- The authors' OSF project (osf.io/4x6ed, view-only link in the paper): the file list and sizes,
  and the OSF metadata of marmoset_sequencing_data.csv (120,510 bytes, MD5
  d3846a4f76749f83c0fc0048a9542c90). The **code** of two of their files was read through a
  summarising fetcher: marmoset_sequencing_preprocessing.py (it writes a session ID taken from the
  annotation file name and the call n-gram of each sequence) and LOO_CV_model_comparison.ipynb (it
  reads the CSV with ';' as separator, skips the header, uses only field 1 as the sequence, removes
  spaces, splits on ',', drops 'chatter', pads with '#', adds no end token, holds out one sequence
  at a time, uses alpha 0.0001 to 10, and compares orders with a Wilcoxon test). No data value,
  header line or row of the CSV has been seen. No statistic has been computed.

## Data and loader

The CSV is read exactly as the authors' notebook reads it; field 0 is kept as the **block**
(the recording session written by their preprocessing script). The header line is recorded in the
results. If field 0 has fewer than 4 distinct values, the verdict is NO_BLOCKS and only the
reproduction is reported.

## Models

Order-1 and order-2 chains as in the notebook: '#' padding, no end token, add-alpha smoothing over
the V observed call types, alpha in {0.0001, 0.001, 0.01, 0.1, 1}. Every call is scored with counts
from all other held-out units.

- **R0 (reproduction):** leave-one-sequence-out D(alpha) = LL2 - LL1. R0 holds if D > 0 for all
  five alphas (the authors' reading). This is also "the authors' test" in the null worlds.
- **D_best** = best LL2 over alpha minus best LL1 over alpha (primary statistic for T1, T2).
- Leave-one-session-out D(alpha) is reported (descriptive).

## Null worlds (seed 20261126; 499 each)

Every real sequence is replaced by a sequence of the **same length** from its block's
first-order chain (start distribution and transition rows; each block's rows are shrunk toward
the pooled rows with beta = 2 pseudo-counts).

- **H0p:** one pooled first-order chain (no blocks, no loops).
- **H0a:** block-specific first-order chains, beta = 2.
- **H0b:** as H0a, but a sequence whose smallest period p satisfies length >= 2p is a chunk of
  length p looped to the sequence's length (chains fitted on chunks).
- **H0a_beta10:** H0a with beta = 10 (sensitivity: less between-session difference).

## Tests

- **T1 (beyond session pooling):** p = (1 + #{H0a worlds with D_best >= real}) / 500.
- **T2 (beyond pooling and loops):** the same against H0b.
- **T3 (calibration of the authors' test):** its fire rate in H0p (ledger procedure
  "marmoset-loo-sequence", worlds = nulls.H0p_pooled_first_order.worlds, hits =
  .authors_test_fires), in H0a and in H0b.
- **T4 (power):** 200 H0b worlds in which a random 30% of second-order contexts (prev2 incl. '#')
  get their own next-call distribution (Dirichlet 0.5), shared by all sessions. Power = share with
  D_best above the 95th percentile of H0b.

## Verdict rules (fixed now)

- p_T2 <= 0.05: **SECOND_ORDER_SURVIVES**.
- else p_T1 <= 0.05: **LOOPING_EXPLAINS** if power >= 0.8, else BEYOND_POOLING_LOOPING_UNDECIDED.
- else **POOLING_EXPLAINS** if power >= 0.8, else UNDECIDED_LOW_POWER; suffix
  _SHRINKAGE_SENSITIVE if p_T1 under beta = 10 is <= 0.05.
- Suffix _NOT_REPRODUCED if R0 does not hold. Fewer than 4 blocks: NO_BLOCKS.

## Predictions (priors, for a Brier score)

| id | prediction | prior |
|---|---|---|
| Q1 | R0 holds on the real data | 0.85 |
| Q2 | p_T1 <= 0.05 | 0.50 |
| Q3 | p_T2 <= 0.05 | 0.40 |
| Q4 | authors' test fires in >= 30% of H0p worlds | 0.15 |
| Q5 | authors' test fires in >= 30% of H0a worlds | 0.55 |
| Q6 | power >= 0.8 | 0.70 |

## Limits

Blocks are sessions, not animals: the file has no animal column that the notebook uses, and
sessions nest within animals. Shrinkage beta = 2 is a choice; beta = 10 is the registered
sensitivity check. The authors' Wilcoxon test treats sequences as independent; it is not
re-run here. Single calls count as sequences, as in the notebook. The CSV is downloaded at run
time from the authors' OSF project and is not redistributed; only summary statistics are
published.

## Code and run

marmoset_order.py (numpy only), test_marmoset_order.py (hand-traced unit tests), fetch_data.py
(checks OSF's size, MD5 and SHA-256), gate.py (nothing is computed until this plan and the code
on GitHub equal the job's copies). One CPU job, 1 GB.
