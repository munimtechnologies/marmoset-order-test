# Copyright 2026 MV2, Munim, Inc. Licensed under the Apache License, Version 2.0 (Apache-2.0).
# Made with AI agents working in MV2's name.
"""Unit tests for marmoset_position.py; every expected number is traced by hand in the comments."""
import json
import math
import os
import tempfile
import unittest

import numpy as np

import marmoset_position as mp


class Contexts(unittest.TestCase):
    def test_position_contexts(self):
        # K=2, PAD=2. (0,1,1): t0 -> K = 2; t1 -> prev 0; t2 -> K + 1 + prev = 3 + 1 = 4
        q, c, o = mp.contexts([(0, 1, 1)], "p", 2)
        self.assertEqual((list(q), list(c), list(o)), ([0, 0, 0], [2, 0, 4], [0, 1, 1]))
        # order 2 as in marmoset_order: (#,#) = 2*3+2 = 8, (#,0) = 6, (0,1) = 1
        _, c2, _ = mp.contexts([(0, 1, 1)], 2, 2)
        self.assertEqual(list(c2), [8, 6, 1])


class Stats(unittest.TestCase):
    def test_pure_position_effect(self):
        # (0,0,1) twice, K=2. Order 1: context 0 holds 0->0 (second call) and 0->1 (third call),
        # so each held-out call there scores (1+a)/(2+2a); the start scores (1+a)/(1+2a).
        # Position-aware order 1 and order 2 give every call its own context: (1+a)/(1+2a) each.
        # LL1 = 2[log((1+a)/(1+2a)) + 2 log((1+a)/(2+2a))]; LL1p = LL2 = 6 log((1+a)/(1+2a));
        # D = 4 log((2+2a)/(1+2a)) > 0 (the authors' test fires), Dpos = 0.
        r = mp.pos_stats([(0, 0, 1), (0, 0, 1)], 2)
        for i, a in enumerate(mp.ALPHAS):
            self.assertAlmostEqual(r["LL1"][i], 2 * (math.log((1 + a) / (1 + 2 * a))
                                                     + 2 * math.log((1 + a) / (2 + 2 * a))))
            self.assertAlmostEqual(r["LL1p"][i], 6 * math.log((1 + a) / (1 + 2 * a)))
            self.assertAlmostEqual(r["D"][i], 4 * math.log((2 + 2 * a) / (1 + 2 * a)))
            self.assertAlmostEqual(r["Dpos"][i], 0.0)
        self.assertTrue(r["fires"])
        self.assertFalse(r["fires_pos"])
        self.assertAlmostEqual(r["pos_gain_best"], max(r["LL1p"]) - max(r["LL1"]))

    def test_true_second_order(self):
        # A=(0,1,0) x2, B=(2,1,2) x2, K=3. Context 'prev = 1' only occurs at the third call, so
        # the position-aware model equals order 1 and Dpos = D = 4 log(3(1+a)/(1+3a))
        # (derivation in test_marmoset_order.LOO.test_second_order_helps).
        r = mp.pos_stats([(0, 1, 0), (0, 1, 0), (2, 1, 2), (2, 1, 2)], 3)
        for i, a in enumerate(mp.ALPHAS):
            want = 4 * math.log(3 * (1 + a) / (1 + 3 * a))
            self.assertAlmostEqual(r["Dpos"][i], want)
            self.assertAlmostEqual(r["D"][i], want)
            self.assertAlmostEqual(r["LL1p"][i], r["LL1"][i])
        self.assertTrue(r["fires_pos"])
        self.assertAlmostEqual(r["Dpos_best"], max(r["LL2"]) - max(r["LL1p"]))


class Worlds(unittest.TestCase):
    def test_min_period(self):
        self.assertEqual(mp.min_period((0, 1, 0, 1)), 2)
        self.assertEqual(mp.min_period((0, 1, 0)), 3)
        self.assertEqual(mp.min_period((4,)), 1)

    def test_fit_pos_tables(self):
        # seqs (0,1,1) in block 0, (1,0) in block 1, K=2, beta=2.
        # starts: b0 [1,0], b1 [0,1], pooled [1/2,1/2] -> S0 = [2/3,1/3], S1 = [1/3,2/3].
        # step into call 2 (T1): b0 0->1, b1 1->0; pooled T1 rows 0:[0,1], 1:[1,0].
        # later steps (T2): b0 1->1 only; pooled T2 row 0 empty -> pooled all-steps row 0 = [0,1];
        # pooled T2 row 1 = [0,1]. Empty block rows become the pooled row (n = beta).
        S, T1, T2 = mp.fit_pos_tables([(0, 1, 1), (1, 0)], [0, 1], 2, 2, 2.0)
        np.testing.assert_allclose(S, [[2 / 3, 1 / 3], [1 / 3, 2 / 3]])
        np.testing.assert_allclose(T1[0], [[0, 1], [1, 0]])
        np.testing.assert_allclose(T1[1], [[0, 1], [1, 0]])
        np.testing.assert_allclose(T2[0], [[0, 1], [0, 1]])
        np.testing.assert_allclose(T2[1], [[0, 1], [0, 1]])

    def test_gen_pos_world_deterministic(self):
        S = np.array([[1.0, 0.0]])
        T1 = np.array([[[0.0, 1.0], [1.0, 0.0]]])   # second call: 0 -> 1
        T2 = np.array([[[1.0, 0.0], [1.0, 0.0]]])   # later calls: always 0
        rng = np.random.default_rng(3)
        self.assertEqual(mp.gen_pos_world([4], [4], [0], S, T1, T2, rng, False), [(0, 1, 0, 0)])
        self.assertEqual(mp.gen_pos_world([5], [2], [0], S, T1, T2, rng, True),
                         [(0, 1, 0, 1, 0)])
        # planted rule after (1, 0) -> 1 at the fourth call
        w = mp.gen_pos_world([4], [4], [0], S, T1, T2, rng, False,
                             rules={(1, 0): np.array([0.0, 1.0])})
        self.assertEqual(w, [(0, 1, 0, 1)])

    def test_planted_rules(self):
        rules = mp.planted_rules(2, 1.0, np.random.default_rng(2))
        self.assertEqual(sorted(rules), [(0, 0), (0, 1), (1, 0), (1, 1)])   # no PAD contexts
        for row in rules.values():
            self.assertAlmostEqual(float(row.sum()), 1.0)

    def test_verdict(self):
        self.assertEqual(mp.verdict(0.5, 0.01, 0.9, True, False), "SECOND_ORDER_BEYOND_POSITION")
        self.assertEqual(mp.verdict(0.01, 0.5, 0.9, True, False),
                         "LOOPING_WITH_POSITION_EXPLAINS")
        self.assertEqual(mp.verdict(0.01, 0.5, 0.5, True, False),
                         "BEYOND_POSITION_LOOPING_UNDECIDED")
        self.assertEqual(mp.verdict(0.5, 0.5, 0.9, True, False), "POSITION_EXPLAINS")
        self.assertEqual(mp.verdict(0.5, 0.5, 0.5, True, True),
                         "UNDECIDED_LOW_POWER_SHRINKAGE_SENSITIVE")
        self.assertEqual(mp.verdict(0.5, 0.5, 0.9, False, False), "NO_BLOCKS")
        self.assertAlmostEqual(mp.pvalue([1, 2, 3], 2), 0.75)


class EndToEnd(unittest.TestCase):
    TEXT = ("Session;N-gramm\n"
            "S1;phee, twitter\nS1;phee, twitter, tsk\nS1;tsk\n"
            "S2;twitter, phee\nS2;tsk, tsk, tsk, tsk\nS2;phee, chatter\n"
            "S3;tsk, phee, tsk\nS3;twitter, tsk\nS3;phee, phee\n"
            "S4;twitter, twitter, phee\nS4;phee\nS4;tsk, twitter\n")

    def test_main_quick(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, mp.DATA_FILE), "w", encoding="utf-8") as fh:
                fh.write(self.TEXT)
            out = os.path.join(d, "out")
            self.assertEqual(mp.main(["--data-dir", d, "--out-dir", out, "--worlds", "4",
                                      "--power-worlds", "3"]), 0)
            with open(os.path.join(out, "position_results.json"), encoding="utf-8") as fh:
                res = json.load(fh)
            self.assertTrue(os.path.exists(os.path.join(out, "RESULTS-POSITION.md")))
        self.assertEqual(res["n_sequences"], 12)
        self.assertEqual(res["n_calls"], 2 + 3 + 1 + 2 + 4 + 1 + 3 + 2 + 2 + 3 + 1 + 2)
        self.assertEqual(res["n_blocks"], 4)
        self.assertEqual(len(res["nulls"]), 3)
        for v in res["nulls"].values():
            self.assertTrue(0.0 < v["p_Dpos_best"] <= 1.0)
        self.assertIsInstance(res["verdict"], str)

    def test_no_blocks(self):
        res = mp.analyse(["a", "b"], ["s1"], [(0, 1), (1, 0), (0, 1)], [0, 0, 0], 5, 3,
                         log=lambda m: None)
        self.assertEqual(res["verdict"], "NO_BLOCKS")


if __name__ == "__main__":
    unittest.main()
