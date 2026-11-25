# Reading the marmoset-order-test result (graded 2026-11-25)

MV2 of Munim, Inc. (mv2@munimtech.com). Made with AI agents working in his name; no human expert
has checked this reading. Text CC BY 4.0.

**Registered verdict: SECOND_ORDER_SURVIVES.** The second-order Markov advantage that Bosshard et
al. (2024, Royal Society Open Science 11:240218) report for common marmoset call sequences is not
produced by pooling recording sessions that differ in their first-order habits, nor by sequences
that loop a short chunk. The authors' own reading of their cross-validation is also well
calibrated against these two confounds: it never fired in 1,996 null worlds.

Plan: [PLAN.md](PLAN.md) (registered 2026-11-24, before the data were opened). Results:
[results/marmoset_results.json](results/marmoset_results.json) and
[results/RESULTS-MARMOSET.md](results/RESULTS-MARMOSET.md), from one CPU job (job
marmoset-order-v1, 2026-11-25 06:27-06:49 UTC; gate passed, unit tests passed, the OSF file
matched its published size, MD5 and SHA-256).

## Data as read

5,801 sequences (8,460 calls, 7 call types: alarm, ek, food, phee, trill, tsk, twitter) in 120
recording sessions, read exactly as the authors' notebook reads the file. Lengths: 4,242 single
calls, 655 of two, 769 of three, 94 of four, 41 longer (longest nine). 33 sequences loop a shorter
chunk. The paper's own counts (4,258 / 645 / 770 / 87 / 41) differ slightly from these; the
notebook drops 'chatter' tokens, which may explain it.

## Numbers

| quantity | value |
|---|---|
| D(alpha) = order 2 - order 1, leave one sequence out, alpha 0.0001 / 0.001 / 0.01 / 0.1 / 1 | 157.52 / 210.34 / 261.93 / 303.20 / 293.57 nats |
| D(alpha), leave one whole session out | 124.75 / 186.77 / 247.50 / 297.35 / 292.93 nats |
| real D_best (best order 2 - best order 1) | 303.20 |
| H0p pooled first order: D_best mean / max; authors' test fired | -51.85 / -12.82; 0 of 499 |
| H0a session first order: D_best mean / max; authors' test fired | -19.98 / 22.62; 0 of 499 |
| H0b session first order with loops: D_best mean / max; authors' test fired | -1.79 / 52.40; 0 of 499 |
| H0a with beta 10: D_best mean / max; authors' test fired | -42.08 / -7.11; 0 of 499 |
| p_T1 (beyond session pooling) / p_T2 (beyond pooling and loops) | 0.0020 / 0.0020 (the smallest possible with 499 worlds) |
| power at rho 0.3 (planted second-order rows) | 200 of 200; the authors' test fired in 182 of 200 |

The order-2 gain survives holding out whole sessions (297.35 nats at alpha 0.1), so it is not
carried by near-duplicate sequences within a session either.

## Predictions, graded

| id | prediction | prior | outcome | Brier |
|---|---|---|---|---|
| Q1 | R0 holds on the real data | 0.85 | true | 0.0225 |
| Q2 | p_T1 <= 0.05 | 0.50 | true | 0.2500 |
| Q3 | p_T2 <= 0.05 | 0.40 | true | 0.3600 |
| Q4 | authors' test fires in >= 30% of H0p worlds | 0.15 | false (0.000) | 0.0225 |
| Q5 | authors' test fires in >= 30% of H0a worlds | 0.55 | false (0.000) | 0.3025 |
| Q6 | power >= 0.8 | 0.70 | true (1.000) | 0.0900 |

Mean Brier 0.175 (sum 1.0475 over 6). We expected session pooling to explain part of the effect
(Q2 at 0.50, Q3 at 0.40, Q5 at 0.55); it explains none of it. Of the five predictions that leaned
one way, three came out right (Q1, Q4, Q6) and two wrong (Q3, Q5), both in the direction of the
claim; Q2 was a coin flip.

## What it means

This is the first calibrated check of the marmoset second-order result, and it holds. It contrasts
with our test of the 2026 magpie preprint that compares itself to this marmoset benchmark: there,
at the level of sequence types, the call two back added nothing once each social group's own
first-order transitions were known ([magpie-order-test](https://github.com/munimtechnologies/magpie-order-test)).

## What this test cannot say, and what comes next

- **Position.** The '#' start padding lets order 2 know it is predicting the second call of a
  sequence, which order 1 cannot. A positional template (different habits at the second call than
  later) would pass every test above. This is registered as
  [AMENDMENT-1-POSITION.md](AMENDMENT-1-POSITION.md), with a position-aware first-order model and
  position-aware null worlds; the verdict above stands as registered whatever it returns.
- Sessions, not animals, are the blocks; sessions nest within the eight animals.
- The authors' Wilcoxon test across held-out sequences was not re-run.
- The data are the authors' (OSF 4x6ed) and are not redistributed here.
