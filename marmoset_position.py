# Copyright 2026 MV2, Munim, Inc. Licensed under the Apache License, Version 2.0 (Apache-2.0).
# Made with AI agents working in MV2's name.
"""marmoset-order-test, amendment 1: is the second-order advantage in common marmoset call
sequences (Bosshard et al. 2024, R Soc Open Sci 11:240218) more than a POSITION effect?

The '#' start padding lets an order-2 model know when it is predicting the second call of a
sequence, which an order-1 model cannot. If marmosets simply use different first-order habits at
the second call than later, order 2 wins without any call depending on the call two back. This
script adds a position-aware first-order model (M1p: the previous call, plus whether it is the
first call of the sequence), which is exactly the order-2 model with every real call two back
merged into one, and compares order 2 with it. Pre-registered in AMENDMENT-1-POSITION.md.
numpy only; self-contained (reads the data exactly as marmoset_order.py does).

Usage: python marmoset_position.py --data-dir DIR --out-dir DIR [--worlds 499] [--power-worlds 200]
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
SEED = 20261127
RHO = 0.3
DATA_FILE = "marmoset_sequencing_data.csv"


# ----------------------------------------------------------------------------- data

def load(path):
    """As marmoset_order.load (the authors' notebook reading): skip the header, split on ';',
    field 1 is the sequence (spaces and quotes removed, split on ',', 'chatter' dropped), field 0
    is the block (recording session)."""
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
    return units, blocks, [tuple(idx[c] for c in s) for _, s in rows], [bidx[b] for b, _ in rows]


# ----------------------------------------------------------------------------- models

def contexts(seqs, model, K):
    """(q, c, o) for every call: sequence index, context code, outcome. PAD = K ('#'), no end token.
    model 1: c = previous call or PAD.
    model 'p' (position-aware first order): t = 0 -> K; t = 1 -> previous call (0..K-1);
      t >= 2 -> K + 1 + previous call.
    model 2: c = prev2 * (K + 1) + prev1 (prev2 = PAD when t < 2)."""
    q, c, o = [], [], []
    for i, s in enumerate(seqs):
        for t, x in enumerate(s):
            p1 = s[t - 1] if t >= 1 else K
            if model == 1:
                cc = p1
            elif model == "p":
                cc = K if t == 0 else (p1 if t == 1 else K + 1 + p1)
            else:
                p2 = s[t - 2] if t >= 2 else K
                cc = p2 * (K + 1) + p1
            q.append(i)
            c.append(cc)
            o.append(x)
    return (np.array(q, dtype=np.int64), np.array(c, dtype=np.int64),
            np.array(o, dtype=np.int64))


def loo_loglik(q, c, o, K, alphas=ALPHAS):
    """Held-out log-likelihood, add-alpha smoothing over V = K outcomes, each call scored with the
    counts of all OTHER units q. One value per alpha. (Same as marmoset_order.loo_loglik.)"""
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


def pos_stats(seqs, K, units_q=None):
    """LL1, LL1p, LL2 per alpha; D = LL2 - LL1 (the authors' comparison), Dpos = LL2 - LL1p;
    *_best use the best alpha of each model; fires = D > 0 at every alpha (the authors' reading);
    fires_pos = Dpos > 0 at every alpha; pos_gain_best = best LL1p - best LL1."""
    ll = {}
    for m in (1, "p", 2):
        q, c, o = contexts(seqs, m, K)
        if units_q is not None:
            q = np.asarray(units_q, dtype=np.int64)[q]
        ll[m] = loo_loglik(q, c, o, K)
    D = [b - a for a, b in zip(ll[1], ll[2])]
    Dp = [b - a for a, b in zip(ll["p"], ll[2])]
    return {"LL1": ll[1], "LL1p": ll["p"], "LL2": ll[2], "D": D, "Dpos": Dp,
            "D_best": max(ll[2]) - max(ll[1]), "Dpos_best": max(ll[2]) - max(ll["p"]),
            "pos_gain_best": max(ll["p"]) - max(ll[1]),
            "fires": bool(all(d > 0 for d in D)), "fires_pos": bool(all(d > 0 for d in Dp))}


# ----------------------------------------------------------------------------- worlds

def min_period(s):
    L = len(s)
    for p in range(1, L // 2 + 1):
        if all(s[i] == s[i - p] for i in range(p, L)):
            return p
    return L


def _norm_rows(counts, fallback):
    out = np.zeros_like(counts, dtype=float)
    for x in range(counts.shape[0]):
        n = counts[x].sum()
        out[x] = counts[x] / n if n > 0 else fallback[x]
    return out


def fit_pos_tables(seqs, bid, nb, K, beta):
    """Per block: start distribution S, rows T1 for the step into the second call (t = 1) and
    rows T2 for later steps (t >= 2). Each block row is shrunk toward the pooled row of its own
    kind with beta pseudo-counts. An empty pooled T1 or T2 row falls back to the pooled row over
    all steps, and an empty one of those to uniform; an empty pooled start to uniform."""
    st = np.zeros((nb, K))
    c1 = np.zeros((nb, K, K))
    c2 = np.zeros((nb, K, K))
    for s, b in zip(seqs, bid):
        st[b, s[0]] += 1
        for t in range(1, len(s)):
            (c1 if t == 1 else c2)[b, s[t - 1], s[t]] += 1
    uni = np.full((K, K), 1.0 / K)
    pall = _norm_rows(c1.sum(axis=0) + c2.sum(axis=0), uni)
    p1 = _norm_rows(c1.sum(axis=0), pall)
    p2 = _norm_rows(c2.sum(axis=0), pall)
    pst = st.sum(axis=0)
    pst = pst / pst.sum() if pst.sum() > 0 else np.full(K, 1.0 / K)
    S = np.zeros((nb, K))
    T1 = np.zeros((nb, K, K))
    T2 = np.zeros((nb, K, K))
    for b in range(nb):
        n = st[b].sum() + beta
        S[b] = (st[b] + beta * pst) / n if n > 0 else pst
        for x in range(K):
            for cnt, pool, T in ((c1, p1, T1), (c2, p2, T2)):
                n = cnt[b, x].sum() + beta
                T[b, x] = (cnt[b, x] + beta * pool[x]) / n if n > 0 else pool[x]
    return S, T1, T2


def _draw(cum, rng):
    return int(min(np.searchsorted(cum, rng.random(), side="right"), cum.size - 1))


def gen_pos_world(lengths, periods, bid, S, T1, T2, rng, loop, rules=None):
    """Every real sequence replaced by one of the same length from its block's position-aware
    chain; with loop=True a chunk of the sequence's own period is drawn and repeated. rules:
    optional {(prev2, prev1): row} planted second-order rows (prev2 a real call)."""
    K = S.shape[1]
    cS = np.cumsum(S, axis=1)
    c1 = np.cumsum(T1, axis=2)
    c2 = np.cumsum(T2, axis=2)
    cR = {k: np.cumsum(v) for k, v in (rules or {}).items()}
    out = []
    for L, p, b in zip(lengths, periods, bid):
        n = p if loop else L
        x = [_draw(cS[b], rng)]
        while len(x) < n:
            t = len(x)
            key = (x[-2] if t >= 2 else K, x[-1])
            if key in cR:
                x.append(_draw(cR[key], rng))
            else:
                x.append(_draw((c1 if t == 1 else c2)[b, x[-1]], rng))
        out.append(tuple(x[i % n] for i in range(L)))
    return out


def planted_rules(K, rho, rng):
    """A random share rho of the K * K contexts with a REAL call two back get their own next-call
    row (Dirichlet 0.5): pure second-order structure that position cannot mimic."""
    ctx = [(a, b) for a in range(K) for b in range(K)]
    m = int(round(rho * len(ctx)))
    pick = rng.choice(len(ctx), size=m, replace=False)
    return {ctx[int(i)]: rng.dirichlet(np.full(K, 0.5)) for i in pick}


def pvalue(null, obs):
    null = np.asarray(null, dtype=float)
    return float((1 + np.sum(null >= obs - 1e-9)) / (1 + null.size))


def _summ(v):
    v = np.asarray(v, dtype=float)
    return {"n": int(v.size), "mean": float(v.mean()), "q50": float(np.quantile(v, 0.5)),
            "q95": float(np.quantile(v, 0.95)), "max": float(v.max())}


def verdict(p5, p6, power, blocks_ok, sens_flag):
    if not blocks_ok:
        return "NO_BLOCKS"
    if p6 <= 0.05:
        return "SECOND_ORDER_BEYOND_POSITION"
    if p5 <= 0.05:
        return ("LOOPING_WITH_POSITION_EXPLAINS" if power >= 0.8
                else "BEYOND_POSITION_LOOPING_UNDECIDED")
    v = "POSITION_EXPLAINS" if power >= 0.8 else "UNDECIDED_LOW_POWER"
    return v + "_SHRINKAGE_SENSITIVE" if sens_flag else v


def analyse(units, blocks, seqs, bid, worlds, power_worlds, seed=SEED, log=print):
    rng = np.random.default_rng(seed)
    K = len(units)
    nb = len(blocks)
    lengths = [len(s) for s in seqs]
    periods = [min_period(s) for s in seqs]
    res = {"units": units, "n_units": K, "n_blocks": nb, "n_sequences": len(seqs),
           "n_calls": int(sum(lengths)), "alphas": list(ALPHAS), "beta": BETA}
    real = pos_stats(seqs, K)
    res["real_loo_sequence"] = real
    res["real_loso_session"] = pos_stats(seqs, K, units_q=bid) if nb >= 2 else None
    span = real["D_best"]
    res["position_share_of_D_best"] = (real["pos_gain_best"] / span) if span > 0 else None
    log("real: D_best %.2f Dpos_best %.2f pos_gain %.2f; blocks %d" % (
        real["D_best"], real["Dpos_best"], real["pos_gain_best"], nb))
    blocks_ok = nb >= 4
    if not blocks_ok:
        res["verdict"] = "NO_BLOCKS"
        return res

    chunks = [s[:p] for s, p in zip(seqs, periods)]
    defs = [
        ("H0c_session_position", fit_pos_tables(seqs, bid, nb, K, BETA), False),
        ("H0d_session_position_loops", fit_pos_tables(chunks, bid, nb, K, BETA), True),
        ("H0c_beta10", fit_pos_tables(seqs, bid, nb, K, BETA_SENS), False),
    ]
    nulls, keep = {}, {}
    for name, (S, T1, T2), loop in defs:
        t = time.time()
        rows = []
        for w in range(worlds):
            rows.append(pos_stats(gen_pos_world(lengths, periods, bid, S, T1, T2, rng, loop), K))
            if (w + 1) % 100 == 0:
                log("%s: %d worlds, %.0f s" % (name, w + 1, time.time() - t))
        dp = [r["Dpos_best"] for r in rows]
        keep[name] = dp
        fires = int(sum(r["fires"] for r in rows))
        fires_pos = int(sum(r["fires_pos"] for r in rows))
        nulls[name] = {"worlds": worlds,
                       "authors_test_fires": fires, "authors_test_fire_rate": fires / worlds,
                       "pos_test_fires": fires_pos, "pos_test_fire_rate": fires_pos / worlds,
                       "Dpos_best": _summ(dp), "D_best": _summ([r["D_best"] for r in rows]),
                       "p_Dpos_best": pvalue(dp, real["Dpos_best"]),
                       "p_D_best": pvalue([r["D_best"] for r in rows], real["D_best"])}
    res["nulls"] = nulls

    S, T1, T2 = fit_pos_tables(chunks, bid, nb, K, BETA)
    q95 = float(np.quantile(keep["H0d_session_position_loops"], 0.95))
    hits = 0
    for _ in range(power_worlds):
        rules = planted_rules(K, RHO, rng)
        st = pos_stats(gen_pos_world(lengths, periods, bid, S, T1, T2, rng, True, rules), K)
        hits += int(st["Dpos_best"] > q95)
    res["power"] = {"rho": RHO, "worlds": power_worlds, "threshold_q95_H0d": q95, "hits": hits,
                    "rate": hits / power_worlds if power_worlds else float("nan")}
    log("power: %d/%d" % (hits, power_worlds))

    p5 = nulls["H0c_session_position"]["p_Dpos_best"]
    p6 = nulls["H0d_session_position_loops"]["p_Dpos_best"]
    p5s = nulls["H0c_beta10"]["p_Dpos_best"]
    pw = res["power"]["rate"]
    sens = (p5 > 0.05) and (p5s <= 0.05)
    res["tests"] = {"R1_order2_beats_position_every_alpha": real["fires_pos"],
                    "p_T5_beyond_position": p5, "p_T6_beyond_position_and_loops": p6,
                    "p_T5_beta10": p5s, "power_rho_0.3": pw, "shrinkage_sensitive": bool(sens),
                    "authors_test_fire_rate_H0c":
                        nulls["H0c_session_position"]["authors_test_fire_rate"]}
    res["verdict"] = verdict(p5, p6, pw, True, sens)
    return res


def write_md(res, path):
    r = res["real_loo_sequence"]
    lines = ["# marmoset-order-test amendment 1 results (position)", "",
             "Made with AI agents working in MV2's name (MV2, Munim, Inc.). CC BY 4.0.", "",
             "Verdict: **%s**" % res["verdict"], "",
             "| item | value |", "|---|---|",
             "| sequences / calls / call types / sessions | %d / %d / %d / %d |" % (
                 res["n_sequences"], res["n_calls"], res["n_units"], res["n_blocks"]),
             "| D(alpha) = order 2 - order 1, leave-one-sequence-out | %s |" % ", ".join(
                 "%.2f" % d for d in r["D"]),
             "| Dpos(alpha) = order 2 - position-aware order 1 | %s |" % ", ".join(
                 "%.2f" % d for d in r["Dpos"]),
             "| D_best; Dpos_best; position gain (best M1p - best M1) | %.2f; %.2f; %.2f |" % (
                 r["D_best"], r["Dpos_best"], r["pos_gain_best"])]
    if res.get("position_share_of_D_best") is not None:
        lines.append("| position share of D_best | %.3f |" % res["position_share_of_D_best"])
    if res.get("real_loso_session"):
        lines.append("| Dpos(alpha), leave-one-session-out | %s |" % ", ".join(
            "%.2f" % d for d in res["real_loso_session"]["Dpos"]))
    for name, v in res.get("nulls", {}).items():
        lines.append("| %s: p of real Dpos_best; authors' test fire rate | %.4f; %.3f |" % (
            name, v["p_Dpos_best"], v["authors_test_fire_rate"]))
    if "power" in res:
        lines.append("| power rho 0.3 (planted rules with a real call two back) | %.3f |" %
                     res["power"]["rate"])
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
    with open(os.path.join(a.out_dir, "position_results.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    write_md(res, os.path.join(a.out_dir, "RESULTS-POSITION.md"))
    print("verdict", res["verdict"], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
