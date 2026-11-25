# marmoset-order-test

A pre-registered, calibrated re-test of the second-order Markov result for common marmoset call
sequences (Bosshard et al. 2024, Royal Society Open Science 11:240218): is the second-order
advantage more than pooling recording sessions that differ in their first-order habits, or
looping short chunks?

**Result (graded 2026-11-25): SECOND_ORDER_SURVIVES.** The held-out gain of order 2 over order 1
is 303.20 nats; the largest gain in 499 worlds built from each session's own first-order chain was
22.62, and 52.40 with looped chunks (p 0.0020 for both). The authors' reading of their
cross-validation never fired in 1,996 worlds without second-order rules. See
[READING-MARMOSET.md](READING-MARMOSET.md).

- [PLAN.md](PLAN.md): the claim, tests, verdict rules and predictions, fixed before the data
  were opened.
- [READING-MARMOSET.md](READING-MARMOSET.md): the graded reading (Brier 0.175).
- [results/](results/): the job's output (`marmoset_results.json`, `RESULTS-MARMOSET.md`).
- [AMENDMENT-1-POSITION.md](AMENDMENT-1-POSITION.md): next test, registered 2026-11-25 before it
  was computed: is the order-2 gain a position effect (the start padding tells order 2 it is
  predicting the second call)? Code `marmoset_position.py`, tests `test_marmoset_position.py`,
  gate `gate_a1.py`.
- `marmoset_order.py`, `test_marmoset_order.py`, `fetch_data.py`, `gate.py`: analysis, unit tests,
  data check and job gate of PLAN.md.

Part of MV2's [ledger of calibrated re-tests of animal-communication claims](https://github.com/munimtechnologies/calibrated-animal-claims).
Companion study: [magpie-order-test](https://github.com/munimtechnologies/magpie-order-test).

By MV2, Munim, Inc. (mv2@munimtech.com). Made with AI agents working in his name; no human expert
has checked it yet. Code Apache-2.0; text CC BY 4.0. The authors' data are not redistributed here.
