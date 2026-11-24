# Copyright 2026 MV2, Munim, Inc. Licensed under the Apache License, Version 2.0 (Apache-2.0).
# Made with AI agents working in MV2's name.
"""Unit tests for marmoset_order.py; every expected number is traced by hand in the comments."""
import json
import math
import os
import tempfile
import unittest

import numpy as np

import marmoset_order as mm


class Load(unittest.TestCase):
    def test_load_and_encode(self):
        text = ("Session;N-gramm\n"
                "S1;phee, twitter\n"
                "S1;chatter\n"                    # only chatter -> empty -> dropped
                'S2;"tsik,ek,chatter,ek"\n'       # quotes and chatter removed
                "bad line without separator\n"
                "\n")
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.csv")
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(text)
            header, rows = mm.load(p)
        self.assertEqual(header, "Session;N-gramm")
        self.assertEqual(rows, [("S1", ["phee", "twitter"]), ("S2", ["tsik", "ek", "ek"])])
        units, blocks, seqs, bid = mm.encode_rows(rows)
        self.assertEqual(units, ["ek", "phee", "tsik", "twitter"])
        self.assertEqual(blocks, ["S1", "S2"])
        self.assertEqual(seqs, [(1, 3), (2, 0, 0)])
        self.assertEqual(bid, [0, 1])


class Contexts(unittest.TestCase):
    def test_contexts(self):
        # K=2, PAD=2. order 1: (#->0) c=2, (0->1) c=0. order 2: (#,#) = 2*3+2 = 8, (#,0) = 6
        q, c, o = mm.contexts([(0, 1)], 1, 2)
        self.assertEqual((list(q), list(c), list(o)), ([0, 0], [2, 0], [0, 1]))
        q, c, o = mm.contexts([(0, 1)], 2, 2)
        self.assertEqual(list(c), [8, 6])


class LOO(unittest.TestCase):
    def test_identical_sequences(self):
        # (0,1) twice, K=V=2: every held-out call has num 1, den 1 -> (1+a)/(1+2a), both orders
        r = mm.order_stats([(0, 1), (0, 1)], 2)
        for i, a in enumerate(mm.ALPHAS):
            self.assertAlmostEqual(r["LL1"][i], 4 * math.log((1 + a) / (1 + 2 * a)))
            self.assertAlmostEqual(r["D"][i], 0.0)
        self.assertFalse(r["fires"])

    def test_second_order_helps(self):
        # A=(0,1,0) x2, B=(2,1,2) x2, K=V=3. Order 1 scores 1->0 in A at 1/3 (context 1 has
        # 4 counts, 1 own); order 2 scores (0,1)->0 at (1+a)/(1+3a). The other calls are scored
        # the same by both orders: (#->0) 1/3 and (0->1) (1+a)/(1+3a). So
        # D(a) = 4 log(3(1+a)/(1+3a)); LL1(a) = 4 [2 log(1/3) + log((1+a)/(1+3a))].
        seqs = [(0, 1, 0), (0, 1, 0), (2, 1, 2), (2, 1, 2)]
        r = mm.order_stats(seqs, 3)
        for i, a in enumerate(mm.ALPHAS):
            self.assertAlmostEqual(r["D"][i], 4 * math.log(3 * (1 + a) / (1 + 3 * a)))
            self.assertAlmostEqual(r["LL1"][i],
                                   4 * (2 * math.log(1 / 3) + math.log((1 + a) / (1 + 3 * a))))
        self.assertTrue(r["fires"])
        self.assertAlmostEqual(r["D_best"], max(r["LL2"]) - max(r["LL1"]))

    def test_leave_one_session_out(self):
        # (0,1),(0,1) in session 0 and (1,0) in session 1, K=2, alpha=1, order 1:
        # session 0: (#->0) num 0 den 1 -> 1/3 (x2); (0->1) num 0 den 0 -> 1/2 (x2)
        # session 1: (#->1) num 0 den 2 -> 1/4; (1->0) num 0 den 0 -> 1/2
        r = mm.order_stats([(0, 1), (0, 1), (1, 0)], 2, units_q=[0, 0, 1])
        want = 2 * math.log(1 / 3) + 2 * math.log(0.5) + math.log(0.25) + math.log(0.5)
        self.assertAlmostEqual(r["LL1"][4], want)


class Worlds(unittest.TestCase):
    def test_min_period(self):
        self.assertEqual(mm.min_period((0, 1, 0, 1)), 2)
        self.assertEqual(mm.min_period((0, 0, 0)), 1)
        self.assertEqual(mm.min_period((0, 1, 0)), 3)
        self.assertEqual(mm.min_period((4,)), 1)

    def test_fit_tables(self):
        # st: b0 [2,0], b1 [0,1]; pooled start [2/3,1/3]; b0 rows: 0->1 x2; b1 rows: 1->0 x1;
        # pooled rows: 0 -> [0,1], 1 -> [1,0]. beta = 2:
        # S0 = ([2,0] + [4/3,2/3]) / 4 = [5/6,1/6]; S1 = ([0,1] + [4/3,2/3]) / 3 = [4/9,5/9]
        S, T = mm.fit_tables([(0, 1), (0, 1), (1, 0)], [0, 0, 1], 2, 2, 2.0)
        np.testing.assert_allclose(S, [[5 / 6, 1 / 6], [4 / 9, 5 / 9]])
        np.testing.assert_allclose(T[0], [[0, 1], [1, 0]])
        np.testing.assert_allclose(T[1], [[0, 1], [1, 0]])

    def test_gen_world_deterministic(self):
        S = np.array([[1.0, 0.0]])
        T = np.array([[[0.0, 1.0], [1.0, 0.0]]])
        rng = np.random.default_rng(1)
        w = mm.gen_world([4, 3], [2, 3], [0, 0], S, T, rng, True)
        self.assertEqual(w, [(0, 1, 0, 1), (0, 1, 0)])
        # planted rule: after (PAD, 0) comes 1 although T says 0 -> 0
        T2 = np.array([[[1.0, 0.0], [1.0, 0.0]]])
        w = mm.gen_world([3], [3], [0], S, T2, rng, False, rules={(2, 0): np.array([0.0, 1.0])})
        self.assertEqual(w, [(0, 1, 0)])

    def test_planted_rules(self):
        rules = mm.planted_rules(2, 1.0, np.random.default_rng(2))
        self.assertEqual(len(rules), 6)          # (K+1) * K contexts
        for row in rules.values():
            self.assertAlmostEqual(float(row.sum()), 1.0)

    def test_verdict(self):
        self.assertEqual(mm.verdict(True, 0.5, 0.01, 0.9, True, False), "SECOND_ORDER_SURVIVES")
        self.assertEqual(mm.verdict(True, 0.01, 0.5, 0.9, True, False), "LOOPING_EXPLAINS")
        self.assertEqual(mm.verdict(True, 0.5, 0.5, 0.9, True, False), "POOLING_EXPLAINS")
        self.assertEqual(mm.verdict(True, 0.5, 0.5, 0.9, True, True),
                         "POOLING_EXPLAINS_SHRINKAGE_SENSITIVE")
        self.assertEqual(mm.verdict(False, 0.5, 0.5, 0.5, True, False),
                         "UNDECIDED_LOW_POWER_NOT_REPRODUCED")
        self.assertEqual(mm.verdict(True, 0.5, 0.5, 0.9, False, False), "NO_BLOCKS")
        self.assertAlmostEqual(mm.pvalue([1, 2, 3], 2), 0.75)


class Smoke(unittest.TestCase):
    def test_analyse_smoke(self):
        seqs = [(0, 1), (0, 1, 2), (2, 1), (1, 2, 1, 2), (0, 2), (1, 0, 1), (2, 0), (0,)] * 2
        bid = [0, 0, 1, 1, 2, 2, 3, 3] * 2
        res = mm.analyse(["a", "b", "c"], ["s1", "s2", "s3", "s4"], seqs, bid, 5, 3, seed=7,
                         log=lambda m: None)
        self.assertEqual(res["n_sequences"], 16)
        self.assertEqual(res["n_calls"], 2 * 19)   # lengths 2+3+2+4+2+3+2+1, twice
        self.assertEqual(len(res["nulls"]), 4)
        for v in res["nulls"].values():
            self.assertTrue(0.0 < v["p_D_best"] <= 1.0)
        self.assertIsInstance(res["verdict"], str)
        json.dumps(res)

    def test_no_blocks(self):
        res = mm.analyse(["a", "b"], ["s1"], [(0, 1), (1, 0), (0, 1)], [0, 0, 0], 5, 3,
                         log=lambda m: None)
        self.assertEqual(res["verdict"], "NO_BLOCKS")


if __name__ == "__main__":
    unittest.main()
