# Copyright 2026 MV2, Munim, Inc. Licensed under the Apache License, Version 2.0 (Apache-2.0).
# Made with AI agents working in MV2's name.
"""marmoset-order-test: is the second-order Markov advantage reported for common marmoset call
sequences (Bosshard et al. 2024, R Soc Open Sci 11:240218) more than pooling recording sessions
with different first-order habits, or looping short chunks? Pre-registered in PLAN.md. numpy only.

Usage: python marmoset_order.py --data-dir DIR --out-dir DIR [--worlds 499] [--power-worlds 200]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ALPHAS = (1e-4, 1e-3, 1e-2, 0.1, 1.0)
BETA = 2.0
BETA_SENS = 10.0
SEED = 20261126
RHO = 0.3
DATA_FILE = "marmoset_sequencing_data.csv"


# ----------------------------------------------------------------------------- data

def load(path):
    """The authors' reading (LOO_CV_model_comparison.ipynb): skip the header, split each line on
    ';', take field 1, remove spaces, split on ',', drop 'chatter'. Field 0 is kept as the block
    (recording session). Empty tokens and surrounding quotes are dropped."""
    with open(path, encoding="utf-8-sig", errors="replace") as fh:
        lines = fh.read().splitlines()
    header = lines[0] if lines else ""
    rows = []
    for line in lines[1:]:
        f = line.split(";")
        if len(f) < 2:
            continue
        field = f[1].replace(" ", "").replace('"', "")
        calls = [x for x in field.split(",") if x and x != "chatter"]
        if calls:
            rows.append((f[0].strip().strip('"'), calls))
    return header, rows


def encode_rows(rows):
    units = sorted({c for _, s in rows for c in s})
    idx = {u: i for i, u in enumerate(units)}
    blocks = sorted({b for b, _ in rows})
    bidx = {b: i for i, b in enumerate(blocks)}
    seqs = [tuple(idx[c] for c in s) for _, s in rows]
    bid = [bidx[b] for b, _ in rows]
    return units, blocks, seqs, bid


# ----------------------------------------------------------------------------- models

def contexts(seqs, order, K):
    """Arrays (q, c, o): sequence index, context code and outcome for every call. PAD = K is the
    authors' '#' start padding; there is no end token (as in the authors' notebook).
    order 1: c = previous call or PAD; order 2: c = prev2 * (K + 1) + prev1."""
    q, c, o = [], [], []
    for i, s in enumerate(seqs):
        for t, x in enumerate(s):
            p1 = s[t - 1] if t >= 1 else K
            if order == 1:
                cc = p1
            else:
                p2 = s[t - 2] if t >= 2 else K
                cc = p2 * (K + 1) + p1
            q.append(i)
            c.append(cc)
            o.append(x)
    return (np.array(q, dtype=np.int64), np.array(c, dtype=np.int64),
            np.array(o, dtype=np.int64))


def loo_loglik(q, c, o, K, alphas=ALPHAS):
    """Held-out log-likelihood with add-alpha smoothing over V = K outcomes, every call scored
    with the counts of all OTHER units q (q = sequence index: the authors' leave-one-sequence-out;
    q = block index: leave-one-session-out). Returns one value per alpha."""
    V = K
    nctx = (K + 1) ** 2
    code = c * V + o
    tot_co = np.bincount(code, minlength=nctx * V)
    tot_c = np.bincount(c, minlength=nctx)
    _, inv, cnt = np.unique(q * (nctx * V) + code, return_inverse=True, return_counts=True)
    own_co = cnt[np.ravel(inv)]
    _, inv2, cnt2 = np.unique(q * nctx + c, return_inverse=True, return_counts=True)
    own_c = cnt2[np.ravel(inv2)]
    num = (tot_co[code] - own_co).astype(float)
    den = (tot_c[c] - own_c).astype(float)
    return [float(np.sum(np.log((num + a) / (den + a * V)))) for a in alphas]


def order_stats(seqs, K, units_q=None):
    """D(alpha) = order-2 minus order-1 held-out log-likelihood; D_best = best order 2 minus
    best order 1 over alpha; fires = D(alpha) > 0 for every alpha (the authors' reading)."""
    out = {}
    for order in (1, 2):
        q, c, o = contexts(seqs, order, K)
        if units_q is not None:
            q = np.asarray(units_q, dtype=np.int64)[q]
        out[order] = loo_loglik(q, c, o, K)
    D = [b - a for a, b in zip(out[1], out[2])]
    return {"LL1": out[1], "LL2": out[2], "D": D, "D_best": max(out[2]) - max(out[1]),
            "fires": bool(all(d > 0 for d in D))}


# ----------------------------------------------------------------------------- worlds

def min_period(s):
    L = len(s)
    for p in range(1, L // 2 + 1):
        if all(s[i] == s[i - p] for i in range(p, L)):
            return p
    return L


def fit_tables(seqs, bid, nb, K, beta):
    """Per block: start distribution and first-order rows, each shrunk toward the pooled
    maximum-likelihood row with beta pseudo-counts (pooled row uniform if it has no counts)."""
    st = np.zeros((nb, K))
    tr = np.zeros((nb, K, K))
    for s, b in zip(seqs, bid):
        st[b, s[0]] += 1
        for x, y in zip(s[:-1], s[1:]):
            tr[b, x, y] += 1
    pst = st.sum(axis=0)
    pst = pst / pst.sum() if pst.sum() > 0 else np.full(K, 1.0 / K)
    ptr = tr.sum(axis=0)
    prow = np.zeros((K, K))
    for x in range(K):
        prow[x] = ptr[x] / ptr[x].sum() if ptr[x].sum() > 0 else np.full(K, 1.0 / K)
    S = np.zeros((nb, K))
    T = np.zeros((nb, K, K))
    for b in range(nb):
        n = st[b].sum() + beta
        S[b] = (st[b] + beta * pst) / n if n > 0 else pst
        for x in range(K):
            n = tr[b, x].sum() + beta
            T[b, x] = (tr[b, x] + beta * prow[x]) / n if n > 0 else prow[x]
    return S, T


def _draw(cum, rng):
    return int(min(np.searchsorted(cum, rng.random(), side="right"), cum.size - 1))


def gen_world(lengths, periods, bid, S, T, rng, loop, rules=None):
    """Every real sequence replaced by one of the same length from its block's chain; with
    loop=True a chunk of the sequence's own period is drawn and repeated. rules: optional
    {(prev2, prev1): row} planted second-order rows (prev2 = K for none)."""
    K = S.shape[1]
    cS = np.cumsum(S, axis=1)
    cT = np.cumsum(T, axis=2)
    cR = {k: np.cumsum(v) for k, v in (rules or {}).items()}
    out = []
    for L, p, b in zip(lengths, periods, bid):
        n = p if loop else L
        x = [_draw(cS[b], rng)]
        while len(x) < n:
            key = (x[-2] if len(x) >= 2 else K, x[-1])
            x.append(_draw(cR[key], rng) if key in cR else _draw(cT[b, x[-1]], rng))
        out.append(tuple(x[i % n] for i in range(L)))
    return out


def planted_rules(K, rho, rng):
    ctx = [(a, b) for a in range(K + 1) for b in range(K)]
    m = int(round(rho * len(ctx)))
    pick = rng.choice(len(ctx), size=m, replace=False)
    return {ctx[int(i)]: rng.dirichlet(np.full(K, 0.5)) for i in pick}


def pvalue(null, obs):
    null = np.asarray(null, dtype=float)
    return float((1 + np.sum(null >= obs - 1e-9)) / (1 + null.size))


def verdict(reproduced, p1, p2, power, blocks_ok, sens_flag):
    if not blocks_ok:
        return "NO_BLOCKS"
    if p2 <= 0.05:
        v = "SECOND_ORDER_SURVIVES"
    elif p1 <= 0.05:
        v = "LOOPING_EXPLAINS" if power >= 0.8 else "BEYOND_POOLING_LOOPING_UNDECIDED"
    else:
        v = "POOLING_EXPLAINS" if power >= 0.8 else "UNDECIDED_LOW_POWER"
        if sens_flag:
            v += "_SHRINKAGE_SENSITIVE"
    return v if reproduced else v + "_NOT_REPRODUCED"


def _summ(v):
    v = np.asarray(v, dtype=float)
    return {"n": int(v.size), "mean": float(v.mean()), "q50": float(np.quantile(v, 0.5)),
            "q95": float(np.quantile(v, 0.95)), "max": float(v.max())}


def analyse(units, blocks, seqs, bid, worlds, power_worlds, seed=SEED, log=print):
    rng = np.random.default_rng(seed)
    K = len(units)
    nb = len(blocks)
    lengths = [len(s) for s in seqs]
    periods = [min_period(s) for s in seqs]
    res = {"units": units, "n_units": K, "n_blocks": nb, "n_sequences": len(seqs),
           "n_calls": int(sum(lengths)),
           "length_counts": {str(L): int(n) for L, n in
                             zip(*np.unique(np.asarray(lengths), return_counts=True))},
           "looped_sequences": int(sum(1 for s, p in zip(seqs, periods) if p < len(s))),
           "alphas": list(ALPHAS), "beta": BETA}
    real = order_stats(seqs, K)
    res["R0_loo_sequence"] = real
    res["R0_loso_session"] = order_stats(seqs, K, units_q=bid) if nb >= 2 else None
    blocks_ok = nb >= 4
    log("real: D %s fires %s; blocks %d" % (["%.2f" % d for d in real["D"]], real["fires"], nb))
    if not blocks_ok:
        res["verdict"] = verdict(real["fires"], 1.0, 1.0, 0.0, False, False)
        return res

    one = [0] * len(seqs)
    chunks = [s[:p] for s, p in zip(seqs, periods)]
    defs = [
        ("H0p_pooled_first_order", fit_tables(seqs, one, 1, K, 0.0), one, False),
        ("H0a_session_first_order", fit_tables(seqs, bid, nb, K, BETA), bid, False),
        ("H0b_session_first_order_loops", fit_tables(chunks, bid, nb, K, BETA), bid, True),
        ("H0a_beta10", fit_tables(seqs, bid, nb, K, BETA_SENS), bid, False),
    ]
    nulls, keep = {}, {}
    for name, (S, T), b_ids, loop in defs:
        t = time.time()
        rows = []
        for w in range(worlds):
            ws = gen_world(lengths, periods, b_ids, S, T, rng, loop)
            rows.append(order_stats(ws, K))
            if (w + 1) % 100 == 0:
                log("%s: %d worlds, %.0f s" % (name, w + 1, time.time() - t))
        db = [r["D_best"] for r in rows]
        fires = int(sum(r["fires"] for r in rows))
        keep[name] = db
        nulls[name] = {"worlds": worlds, "authors_test_fires": fires,
                       "authors_test_fire_rate": fires / worlds, "D_best": _summ(db),
                       "p_D_best": pvalue(db, real["D_best"])}
    res["nulls"] = nulls

    S, T = fit_tables(chunks, bid, nb, K, BETA)
    q95 = float(np.quantile(keep["H0b_session_first_order_loops"], 0.95))
    hits = fires = 0
    for _ in range(power_worlds):
        rules = planted_rules(K, RHO, rng)
        st = order_stats(gen_world(lengths, periods, bid, S, T, rng, True, rules), K)
        hits += int(st["D_best"] > q95)
        fires += int(st["fires"])
    res["power"] = {"rho": RHO, "worlds": power_worlds, "threshold_q95_H0b": q95, "hits": hits,
                    "rate": hits / power_worlds if power_worlds else float("nan"),
                    "authors_test_fires": fires}
    log("power: %d/%d" % (hits, power_worlds))

    p1 = nulls["H0a_session_first_order"]["p_D_best"]
    p2 = nulls["H0b_session_first_order_loops"]["p_D_best"]
    p1s = nulls["H0a_beta10"]["p_D_best"]
    pw = res["power"]["rate"]
    sens = (p1 > 0.05) and (p1s <= 0.05)
    res["tests"] = {"R0_reproduced": real["fires"], "p_T1_beyond_session_pooling": p1,
                    "p_T2_beyond_pooling_and_loops": p2, "p_T1_beta10": p1s,
                    "power_rho_0.3": pw, "shrinkage_sensitive": bool(sens)}
    res["verdict"] = verdict(real["fires"], p1, p2, pw, True, sens)
    h0p = nulls["H0p_pooled_first_order"]
    res["ledger"] = {"procedure": "marmoset-loo-sequence", "worlds_H0p": h0p["worlds"],
                     "fires_H0p": h0p["authors_test_fires"],
                     "worlds_H0a": worlds,
                     "fires_H0a": nulls["H0a_session_first_order"]["authors_test_fires"]}
    return res


def write_md(res, path):
    lines = ["# marmoset-order-test results", "",
             "Made with AI agents working in MV2's name (MV2, Munim, Inc.). CC BY 4.0.", "",
             "Verdict: **%s**" % res["verdict"], "",
             "| item | value |", "|---|---|",
             "| sequences / calls / call types / sessions | %d / %d / %d / %d |" % (
                 res["n_sequences"], res["n_calls"], res["n_units"], res["n_blocks"]),
             "| D(alpha), leave-one-sequence-out | %s |" % ", ".join(
                 "%.2f" % d for d in res["R0_loo_sequence"]["D"])]
    if res.get("R0_loso_session"):
        lines.append("| D(alpha), leave-one-session-out | %s |" % ", ".join(
            "%.2f" % d for d in res["R0_loso_session"]["D"]))
    for name, v in res.get("nulls", {}).items():
        lines.append("| %s: authors' test fire rate; p of real D_best | %.3f; %.4f |" % (
            name, v["authors_test_fire_rate"], v["p_D_best"]))
    if "power" in res:
        lines.append("| power rho 0.3 | %.3f |" % res["power"]["rate"])
    lines.append("")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--worlds", type=int, default=499)
    ap.add_argument("--power-worlds", type=int, default=200)
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    header, rows = load(os.path.join(a.data_dir, DATA_FILE))
    units, blocks, seqs, bid = encode_rows(rows)
    t = time.time()
    res = analyse(units, blocks, seqs, bid, a.worlds, a.power_worlds,
                  log=lambda m: print(m, flush=True))
    res["header_line"] = header
    res["seconds"] = time.time() - t
    with open(os.path.join(a.out_dir, "marmoset_results.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    write_md(res, os.path.join(a.out_dir, "RESULTS-MARMOSET.md"))
    print("verdict", res["verdict"], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
