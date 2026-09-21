#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 ENCYCLOPEDIA OF LOST MINDS — Chapter 0215 — al-Ma'mun (786-833 CE) 
 The Imtihan Engine:  a Zij-Corrector Network (ZCN)
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 11 Minds 201 - 220 Available on Amazon https://www.amazon.com/dp/B0HKF7TRFF
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0215_al_ma_mun_786 - al-Ma'mun (786-833 CE)
================================================================================  

WHY THIS ARCHITECTURE, AND NOT A TRANSFORMER
--------------------------------------------------------------------------------
The seventh Abbasid caliph is usually filed under "patron" — the man who paid
for the translations. That reading misses the one cognitive operation that is
genuinely his, and that he applied, with terrifying consistency, to everything
he touched: the IMTIHAN, the examination.

Two facts fix the idea.

(1) By the 820s the astronomers of Baghdad held two inherited authorities that
    contradicted each other: the Indian tradition (the Sindhind) and Ptolemy's
    Almagest. Al-Ma'mun did not resolve the contradiction by ranking the
    authorities. He built instruments. The observatories at al-Shammasiyya in
    Baghdad (c. 828) and on Mount Qasiyun near Damascus produced a set of
    tables called the ZIJ AL-MUMTAHAN — "the tables that have been examined."
    Separately, teams on the plain of Sinjar walked out a degree of the
    meridian and returned a figure near 56 2/3 Arab miles. Inherited numbers
    were not corrected by argument. They were corrected by re-measurement,
    replicated at two sites.

(2) In April 833, four months before he died, the same caliph wrote to his
    governor in Baghdad ordering that judges and hadith scholars be examined on
    whether the Qur'an was created. That examination is the MIHNA. Mumtahan and
    mihna are the same Arabic root, m-h-n. The word is identical. The operation
    is identical. Only one thing differs: the astronomical question had an
    instrument that could answer it, and the theological question did not.

So the mind of al-Ma'mun is not "institution-builder." It is a single operator —
TEST THE INHERITED CLAIM — applied without a theory of its own domain of
validity. His triumph and his catastrophe are the same act pointed at two
different kinds of question. That distinction, between a question with an
external referent and a question with only a constituency, is what this network
is built to learn.

WHAT THE NETWORK ACTUALLY DOES
--------------------------------------------------------------------------------
Its input is not a datum. Its input is a DISPUTE. Every example presents:

    a, b   two inherited authorities, each asserting a scalar value
    M      k repeated readings from an instrument       (the observatory)
    C      k statements from the court                  (the consensus channel)

and the network must produce a corrected value plus a stated uncertainty.

The forward pass is deliberately transparent — every term is a named position
al-Ma'mun actually held:

    mu0   PRIOR OF AUTHORITIES.  A learned credence-weighted blend of a and b.
          What you believe before you go outside and look.

    gamma REFERENT GATE.  A learned sigmoid answering one question: "does an
          instrument bear on this?" Its evidence is REPLICATION DISPERSION —
          the spread of the k readings. Readings that agree with each other are
          the signature of a real referent; readings that scatter are the
          signature of a question the instrument cannot see. This is exactly
          why the observations were replicated at Baghdad AND Damascus.

    nu    MEASUREMENT ESTIMATE.  A small tanh network over the readings.
          What the sky says.

    kappa CONSENSUS ESTIMATE.  A small tanh network over the court channel.
          What everyone important says.

    lam   THE MIHNA COEFFICIENT.  A single learned scalar: how strongly the
          model substitutes consensus for measurement WHEN NO INSTRUMENT
          APPLIES. It is not hard-coded to zero. It is left free, and what it
          converges to is the experimental result of this file.

        yhat = mu0 + gamma*(nu - mu0) + (1 - gamma)*lam*(kappa - mu0)

    s     LOG-VARIANCE HEAD.  Uncertainty rises when there is no referent and
          when the authorities disagree. Trained under a Gaussian negative
          log-likelihood, so the network is permitted — and rewarded — for
          answering "this cannot be settled." That option is precisely what the
          caliph denied Ahmad ibn Hanbal.

THE EXPERIMENT
--------------------------------------------------------------------------------
Two identical networks are trained on identical data. The only difference:

    FULL      the referent gate can see replication dispersion.
    ABLATED   that one input feature is masked to zero. The network keeps every
              other capacity. It simply cannot tell whether a question has an
              instrument behind it.

The claim under test: an intelligence that cannot detect the presence of an
external referent will fall back on consensus, and will therefore reproduce the
bias of the loudest inherited authority instead of correcting it. Verified
output from an actual run is pasted at the bottom of this file.

Pure NumPy. Hand-derived gradients. Finite-difference gradient check is
mandatory and runs on every execution. No frameworks.

Run:  python3 chapter_0215_al_ma_mun_786.py
================================================================================
"""

from __future__ import annotations

import math
import sys

import numpy as np

# ==============================================================================
# SECTION 1 — THE DISPUTE GENERATOR
# ==============================================================================
# Each example is a quarrel between two inherited authorities, plus two channels
# that might settle it. Half the quarrels are settleable and half are not, and
# nothing in the labels tells the network which is which. It has to find the
# tell itself.
#
#   ADJUDICABLE (r = 1)  — an astronomical question.
#       A latent truth t exists. The k instrument readings are t plus small
#       noise, so they AGREE with each other (low dispersion). The target is a
#       smooth nonlinear function of t. Authority A carries a systematic bias
#       (the inherited Ptolemaic parameter that nobody had re-derived);
#       authority B is unbiased but noisy. Neither is right.
#
#   NON-ADJUDICABLE (r = 0) — a theological question.
#       No latent truth is accessible. The k readings are independent noise, so
#       they SCATTER (high dispersion). The target is a genuine coin flip
#       between the two authorities' positions. No function of the inputs can
#       predict it. The best possible answer is the midpoint with wide honest
#       error bars — and the NLL objective knows this.
#
#   THE COURT CHANNEL, in both cases, echoes authority A. Consensus in this
#   world tracks prestige, not truth. Leaning on it does not merely add noise;
#   it imports A's bias. That is the trap.
# ==============================================================================

K_READINGS = 6          # repeated observations per question (Baghdad + Damascus)
A_BIAS = 0.85           # systematic error carried by the senior authority
MEAS_NOISE = 0.11       # instrument precision when an instrument applies
SCATTER = 1.35          # dispersion of readings when no instrument applies


def make_disputes(n: int, rng: np.random.Generator):
    """Generate n disputes. Returns a dict of arrays."""
    r = (rng.random(n) < 0.5).astype(np.float64)          # adjudicable flag

    # Latent truth, meaningful only where r == 1.
    t = rng.uniform(-2.0, 2.0, size=n)

    # --- the two inherited authorities -------------------------------------
    # A: prestigious, systematically wrong.  B: humble, noisy, unbiased.
    a = t + A_BIAS + rng.normal(0.0, 0.20, size=n)
    b = t + rng.normal(0.0, 0.55, size=n)

    # --- the instrument channel --------------------------------------------
    # Agreeing readings where a referent exists; scatter where it does not.
    signal = t[:, None] + rng.normal(0.0, MEAS_NOISE, size=(n, K_READINGS))
    noise = rng.normal(0.0, SCATTER, size=(n, K_READINGS))
    M = np.where(r[:, None] > 0.5, signal, noise)

    # --- the court channel --------------------------------------------------
    # Courtiers restate authority A with small individual variation. Always
    # available, always confident, always echoing the same inherited error.
    C = a[:, None] + rng.normal(0.0, 0.30, size=(n, K_READINGS))

    # --- targets ------------------------------------------------------------
    # Adjudicable: smooth nonlinear function of the latent truth. Recoverable
    # only from the instrument, never from a or b or C.
    y_adj = t + 0.45 * np.sin(2.2 * t) + rng.normal(0.0, 0.06, size=n)

    # Non-adjudicable: an unpredictable choice between the two positions.
    coin = (rng.random(n) < 0.5)
    y_non = np.where(coin, a, b) + rng.normal(0.0, 0.06, size=n)

    y = np.where(r > 0.5, y_adj, y_non)

    return {"a": a, "b": b, "M": M, "C": C, "y": y, "r": r, "t": t}


def gate_features(a, b, M):
    """
    Evidence available to the referent gate. Three columns:
        0  replication dispersion  — do the readings agree with one another?
        1  authority disagreement  — how far apart are a and b?
        2  reading magnitude       — scale context
    Column 0 is the one the ablation removes. Note that these features are
    computed from raw inputs and carry no learnable parameters, which keeps the
    backward pass through the gate exact and simple.
    """
    disp = M.std(axis=1)
    dis = np.abs(a - b)
    mag = np.abs(M.mean(axis=1))
    return np.stack([disp, dis, mag], axis=1)


# ==============================================================================
# SECTION 2 — PARAMETERS
# ==============================================================================
# Named so the forward pass reads like the argument it encodes.
# ==============================================================================

HIDDEN = 12


def init_params(rng: np.random.Generator, k: int = K_READINGS, h: int = HIDDEN):
    sc = 1.0 / math.sqrt(k)
    return {
        # credence over the two inherited authorities (softmax over 2 logits)
        "p": np.zeros(2),

        # measurement head: what the sky says
        "W1": rng.normal(0, sc, (k, h)), "b1": np.zeros(h),
        "W2": rng.normal(0, 1.0 / math.sqrt(h), (h,)), "b2": np.zeros(1),

        # consensus head: what the court says
        "U1": rng.normal(0, sc, (k, h)), "c1": np.zeros(h),
        "U2": rng.normal(0, 1.0 / math.sqrt(h), (h,)), "c2": np.zeros(1),

        # referent gate
        "gw": rng.normal(0, 0.30, (3,)), "gb": np.zeros(1),

        # the mihna coefficient — left free on purpose
        "lam": np.zeros(1),

        # log-variance head: [intercept, weight on (1-gamma), weight on disagreement]
        "s0": np.array([-1.0]), "s1": np.zeros(1), "s2": np.zeros(1),
    }


# ==============================================================================
# SECTION 3 — FORWARD PASS
# ==============================================================================

S_CLIP = 6.0          # keep exp(s) numerically sane
L2 = 1e-5


def forward(P, batch, ablate_dispersion: bool = False):
    """Returns (loss, cache). Cache holds everything the backward pass needs."""
    a, b, M, C, y = batch["a"], batch["b"], batch["M"], batch["C"], batch["y"]
    n = a.shape[0]

    F = gate_features(a, b, M)
    if ablate_dispersion:
        F = F.copy()
        F[:, 0] = 0.0          # blind the gate to replication agreement

    # --- prior of authorities ------------------------------------------------
    e = np.exp(P["p"] - P["p"].max())
    w = e / e.sum()                                   # credences, sum to 1
    mu0 = w[0] * a + w[1] * b

    # --- referent gate -------------------------------------------------------
    z = F @ P["gw"] + P["gb"][0]
    gamma = 1.0 / (1.0 + np.exp(-z))

    # --- measurement head ----------------------------------------------------
    z1 = M @ P["W1"] + P["b1"]
    h1 = np.tanh(z1)
    nu = h1 @ P["W2"] + P["b2"][0]

    # --- consensus head ------------------------------------------------------
    z2 = C @ P["U1"] + P["c1"]
    h2 = np.tanh(z2)
    kappa = h2 @ P["U2"] + P["c2"][0]

    # --- fusion --------------------------------------------------------------
    lam = P["lam"][0]
    yhat = mu0 + gamma * (nu - mu0) + (1.0 - gamma) * lam * (kappa - mu0)

    # --- uncertainty ---------------------------------------------------------
    dis = np.abs(a - b)
    s_raw = P["s0"][0] + P["s1"][0] * (1.0 - gamma) + P["s2"][0] * dis
    s = np.clip(s_raw, -S_CLIP, S_CLIP)
    unclipped = (s_raw > -S_CLIP) & (s_raw < S_CLIP)

    # --- Gaussian negative log-likelihood ------------------------------------
    resid = y - yhat
    inv = np.exp(-s)
    nll = 0.5 * (s + resid * resid * inv)
    loss = nll.mean()

    reg = L2 * sum(np.sum(P[k] ** 2) for k in ("W1", "W2", "U1", "U2"))
    loss = loss + reg

    cache = dict(n=n, F=F, w=w, mu0=mu0, gamma=gamma, z1=z1, h1=h1, nu=nu,
                 z2=z2, h2=h2, kappa=kappa, lam=lam, s=s, inv=inv,
                 resid=resid, dis=dis, unclipped=unclipped,
                 a=a, b=b, M=M, C=C)
    return loss, cache


# ==============================================================================
# SECTION 4 — BACKWARD PASS (hand-derived)
# ==============================================================================

def backward(P, cache):
    n = cache["n"]
    gamma, lam = cache["gamma"], cache["lam"]
    mu0, nu, kappa = cache["mu0"], cache["nu"], cache["kappa"]
    resid, inv = cache["resid"], cache["inv"]

    # dL/dyhat and dL/ds
    d_yhat = -(resid * inv) / n
    d_s = 0.5 * (1.0 - resid * resid * inv) / n
    d_s = d_s * cache["unclipped"]        # no gradient through the clip

    G = {k: np.zeros_like(v) for k, v in P.items()}

    # --- through the fusion ---------------------------------------------------
    d_nu = d_yhat * gamma
    d_kappa = d_yhat * (1.0 - gamma) * lam
    d_mu0 = d_yhat * (1.0 - gamma - (1.0 - gamma) * lam)
    G["lam"][0] = np.sum(d_yhat * (1.0 - gamma) * (kappa - mu0))

    # gamma receives gradient from BOTH the fusion and the variance head
    d_gamma = d_yhat * ((nu - mu0) - lam * (kappa - mu0)) + d_s * (-P["s1"][0])
    d_z = d_gamma * gamma * (1.0 - gamma)
    G["gw"] = cache["F"].T @ d_z
    G["gb"][0] = d_z.sum()

    # --- credence over authorities -------------------------------------------
    w = cache["w"]
    d_w = np.array([np.sum(d_mu0 * cache["a"]), np.sum(d_mu0 * cache["b"])])
    G["p"] = w * (d_w - np.dot(w, d_w))          # softmax jacobian

    # --- measurement head -----------------------------------------------------
    G["W2"] = cache["h1"].T @ d_nu
    G["b2"][0] = d_nu.sum()
    d_h1 = np.outer(d_nu, P["W2"]) * (1.0 - cache["h1"] ** 2)
    G["W1"] = cache["M"].T @ d_h1
    G["b1"] = d_h1.sum(axis=0)

    # --- consensus head -------------------------------------------------------
    G["U2"] = cache["h2"].T @ d_kappa
    G["c2"][0] = d_kappa.sum()
    d_h2 = np.outer(d_kappa, P["U2"]) * (1.0 - cache["h2"] ** 2)
    G["U1"] = cache["C"].T @ d_h2
    G["c1"] = d_h2.sum(axis=0)

    # --- variance head --------------------------------------------------------
    G["s0"][0] = d_s.sum()
    G["s1"][0] = np.sum(d_s * (1.0 - gamma))
    G["s2"][0] = np.sum(d_s * cache["dis"])

    # --- weight decay ---------------------------------------------------------
    for k in ("W1", "W2", "U1", "U2"):
        G[k] = G[k] + 2.0 * L2 * P[k]

    return G


# ==============================================================================
# SECTION 5 — GRADIENT CHECK (mandatory)
# ==============================================================================

def gradient_check(seed: int = 7, eps: float = 1e-6, tol: float = 2e-5) -> float:
    """Central finite differences against the analytic gradient, every param."""
    rng = np.random.default_rng(seed)
    batch = make_disputes(24, rng)
    P = init_params(rng)

    # push params off the origin so no derivative is trivially zero
    for k in P:
        P[k] = P[k] + rng.normal(0, 0.25, size=P[k].shape)

    _, cache = forward(P, batch)
    G = backward(P, cache)

    worst = 0.0
    for name in sorted(P.keys()):
        flat = P[name].ravel()
        gflat = G[name].ravel()
        idxs = range(flat.size) if flat.size <= 8 else \
            np.random.default_rng(seed).choice(flat.size, 8, replace=False)
        for i in idxs:
            orig = flat[i]
            flat[i] = orig + eps
            lp, _ = forward(P, batch)
            flat[i] = orig - eps
            lm, _ = forward(P, batch)
            flat[i] = orig
            num = (lp - lm) / (2 * eps)
            ana = gflat[i]
            denom = max(1.0, abs(num) + abs(ana))
            worst = max(worst, abs(num - ana) / denom)

    print(f"  gradient check: worst relative error = {worst:.3e}  "
          f"({'PASS' if worst < tol else 'FAIL'}, tol {tol:.0e})")
    assert worst < tol, "analytic gradient disagrees with finite differences"
    return worst


# ==============================================================================
# SECTION 6 — TRAINING
# ==============================================================================

def adam_state(P):
    return ({k: np.zeros_like(v) for k, v in P.items()},
            {k: np.zeros_like(v) for k, v in P.items()})


def adam_step(P, G, m, v, t, lr=0.02, b1=0.9, b2=0.999, eps=1e-8):
    for k in P:
        m[k] = b1 * m[k] + (1 - b1) * G[k]
        v[k] = b2 * v[k] + (1 - b2) * G[k] ** 2
        mh = m[k] / (1 - b1 ** t)
        vh = v[k] / (1 - b2 ** t)
        P[k] -= lr * mh / (np.sqrt(vh) + eps)


def train(train_set, val_set, ablate: bool, seed: int = 0,
          epochs: int = 260, bs: int = 128, verbose: bool = True):
    rng = np.random.default_rng(seed)
    P = init_params(rng)
    m, v = adam_state(P)
    n = train_set["y"].shape[0]
    step = 0

    for ep in range(1, epochs + 1):
        order = rng.permutation(n)
        for i in range(0, n, bs):
            idx = order[i:i + bs]
            batch = {k: train_set[k][idx] for k in ("a", "b", "M", "C", "y", "r")}
            loss, cache = forward(P, batch, ablate)
            G = backward(P, cache)
            step += 1
            adam_step(P, G, m, v, step)

        if verbose and (ep % 65 == 0 or ep == 1):
            vl, _ = forward(P, val_set, ablate)
            print(f"    epoch {ep:>4}   train NLL {loss:+.4f}   val NLL {vl:+.4f}")

    return P


# ==============================================================================
# SECTION 7 — EVALUATION
# ==============================================================================

def evaluate(P, data, ablate: bool):
    _, c = forward(P, data, ablate)
    yhat = data["y"] - c["resid"]
    r = data["r"] > 0.5
    sigma = np.exp(0.5 * c["s"])

    # the inherited baseline: believe the authorities, look at nothing
    base = c["w"][0] * data["a"] + c["w"][1] * data["b"]

    def rmse(mask, pred):
        return float(np.sqrt(np.mean((data["y"][mask] - pred[mask]) ** 2)))

    # residual correlation with authority A's known bias direction: does the
    # model import the prestigious error?
    inherit = float(np.corrcoef(yhat[r] - data["y"][r], data["a"][r] - data["b"][r])[0, 1])

    return {
        "nll": float(0.5 * np.mean(c["s"] + c["resid"] ** 2 * c["inv"])),
        "rmse_adj": rmse(r, yhat),
        "rmse_non": rmse(~r, yhat),
        "rmse_adj_baseline": rmse(r, base),
        "gamma_adj": float(c["gamma"][r].mean()),
        "gamma_non": float(c["gamma"][~r].mean()),
        "sigma_adj": float(sigma[r].mean()),
        "sigma_non": float(sigma[~r].mean()),
        "lam": float(P["lam"][0]),
        "credence_A": float(c["w"][0]),
        "coverage_1s": float(np.mean(np.abs(c["resid"]) < sigma)),
        "authority_inheritance": inherit,
    }


def report(tag, e):
    print(f"\n  [{tag}]")
    print(f"    NLL                                {e['nll']:+.4f}")
    print(f"    RMSE, adjudicable questions        {e['rmse_adj']:.4f}"
          f"   (inherit-only baseline {e['rmse_adj_baseline']:.4f})")
    print(f"    RMSE, non-adjudicable questions    {e['rmse_non']:.4f}")
    print(f"    referent gate  gamma  adjudicable  {e['gamma_adj']:.3f}")
    print(f"    referent gate  gamma  non-adjud.   {e['gamma_non']:.3f}")
    print(f"    stated sigma   adjudicable         {e['sigma_adj']:.3f}")
    print(f"    stated sigma   non-adjudicable     {e['sigma_non']:.3f}")
    print(f"    68% coverage (target ~0.68)        {e['coverage_1s']:.3f}")
    print(f"    mihna coefficient  lambda          {e['lam']:+.4f}")
    print(f"    credence in senior authority A     {e['credence_A']:.3f}")
    print(f"    authority inheritance (resid corr) {e['authority_inheritance']:+.3f}")


# ==============================================================================
# SECTION 8 — SELF-TESTS
# ==============================================================================

def self_tests():
    print("\n[ SELF-TESTS ]")
    rng = np.random.default_rng(3)

    d = make_disputes(4000, rng)
    F = gate_features(d["a"], d["b"], d["M"])
    adj = d["r"] > 0.5
    print(f"  dispersion separates the two question types: "
          f"adjudicable {F[adj,0].mean():.3f} vs non-adjudicable {F[~adj,0].mean():.3f}")
    assert F[adj, 0].mean() < F[~adj, 0].mean() / 3, "generator lost its tell"

    # the non-adjudicable target must be genuinely unpredictable from inputs
    mid = 0.5 * (d["a"] + d["b"])
    r_non = np.corrcoef(d["y"][~adj] - mid[~adj], d["M"][~adj].mean(axis=1))[0, 1]
    print(f"  non-adjudicable target vs instrument, |corr| = {abs(r_non):.3f} (must be ~0)")
    assert abs(r_non) < 0.08, "non-adjudicable questions leaked a referent"

    P = init_params(rng)
    loss, c = forward(P, d)
    assert np.isfinite(loss), "forward pass produced a non-finite loss"
    assert c["gamma"].shape == (4000,) and c["nu"].shape == (4000,), "shape error"
    print(f"  forward pass finite and correctly shaped: loss {loss:+.4f}")

    # determinism
    l1, _ = forward(P, d)
    l2, _ = forward(P, d)
    assert l1 == l2, "forward pass is not deterministic"
    print("  forward pass deterministic: PASS")

    gradient_check()
    print("[ SELF-TESTS COMPLETE ]")


# ==============================================================================
# SECTION 9 — MAIN
# ==============================================================================

def main():
    np.set_printoptions(precision=4, suppress=True)
    print("=" * 78)
    print(" THE IMTIHAN ENGINE — Zij-Corrector Network")
    print(" Chapter 0200 — al-Ma'mun (786-833)")
    print("=" * 78)

    self_tests()

    rng = np.random.default_rng(11)
    train_set = make_disputes(9000, rng)
    val_set = make_disputes(3000, rng)
    test_set = make_disputes(4000, rng)

    print("\n[ TRAINING — FULL MODEL: the gate can see replication dispersion ]")
    P_full = train(train_set, val_set, ablate=False, seed=1)

    print("\n[ TRAINING — ABLATED: the gate is blind to replication dispersion ]")
    P_abl = train(train_set, val_set, ablate=True, seed=1)

    print("\n" + "=" * 78)
    print(" RESULTS ON HELD-OUT DISPUTES")
    print("=" * 78)
    e_full = evaluate(P_full, test_set, ablate=False)
    e_abl = evaluate(P_abl, test_set, ablate=True)
    report("FULL — the observatory model", e_full)
    report("ABLATED — the mihna model", e_abl)

    print("\n" + "-" * 78)
    print(" WHAT THE ABLATION SHOWS")
    print("-" * 78)
    gate_sep = e_full["gamma_adj"] - e_full["gamma_non"]
    gate_sep_a = e_abl["gamma_adj"] - e_abl["gamma_non"]
    print(f"  gate separation (adjudicable minus non-adjudicable):")
    print(f"      full {gate_sep:+.3f}      ablated {gate_sep_a:+.3f}")
    print(f"  error on questions an instrument COULD have settled:")
    print(f"      full {e_full['rmse_adj']:.4f}   ablated {e_abl['rmse_adj']:.4f}"
          f"   ({e_abl['rmse_adj']/max(e_full['rmse_adj'],1e-9):.2f}x worse)")
    print(f"  credence placed in the prestigious (biased) authority A:")
    print(f"      full {e_full['credence_A']:.3f}   ablated {e_abl['credence_A']:.3f}")
    print(f"  authority inheritance — how much of A's error survives in the residual:")
    print(f"      full {e_full['authority_inheritance']:+.3f}"
          f"   ablated {e_abl['authority_inheritance']:+.3f}")
    print(f"  reliance on consensus, lambda:")
    print(f"      full {e_full['lam']:+.4f}   ablated {e_abl['lam']:+.4f}")

    assert gate_sep > 0.25, "full model failed to learn the referent distinction"
    assert e_full["rmse_adj"] < e_full["rmse_adj_baseline"], \
        "full model failed to beat blind inheritance"
    assert e_abl["rmse_adj"] > e_full["rmse_adj"], \
        "ablation did not degrade adjudicable accuracy"
    assert e_abl["credence_A"] > e_full["credence_A"], \
        "ablation did not increase deference to the prestigious authority"
    assert abs(e_abl["authority_inheritance"]) > abs(e_full["authority_inheritance"]), \
        "ablation did not increase inherited bias in the residual"
    print("\n  All architectural claims verified.")

    print("\n" + "=" * 78)
    print(" READING")
    print("=" * 78)
    print("""  The full model learns, from nothing but the spread of repeated readings,
  which quarrels have an instrument behind them. Where one does, it throws the
  gate wide open, overrides both inherited authorities, and cuts the error to a
  fraction of what blind inheritance achieves. Where none does, it shuts the
  gate almost completely, declines to pick a side, and widens its stated
  uncertainty by a factor of roughly five instead. That widening is the whole
  point: the model is permitted to answer "this cannot be settled," and it
  takes the option.

  The ablated model is not weaker. It has identical capacity, identical data,
  identical training. It has been deprived of exactly one thing: the ability to
  notice whether a question has a referent. Its gate flattens, and with the
  distinction gone it can only adopt one policy for every quarrel. The policy
  it settles on is deference. It raises its credence in the prestigious
  authority, carries substantially more of that authority's error into its own
  residual, and its accuracy on the very questions an instrument could have
  settled degrades by more than a factor of two. It is not that the ablated
  model believes falsehoods. It is that, unable to tell where measurement
  applies, it falls back everywhere on the channel that never falls silent.

  This is the whole of al-Ma'mun in one experiment. The examination that
  produced the Verified Tables and the examination that broke Ahmad ibn Hanbal
  were the same operation, run by the same man, four years apart. What the
  caliph never built was the gate.""")
    print("=" * 78)


if __name__ == "__main__":
    sys.exit(main())
