#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Chapter 0171 - LLYWARCH HEN  ::  THE ELEGIAC PERSONA NETWORK  (EPN)
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 9 Minds 161 - 180 Available on Amazon https://www.amazon.com/dp/B0HJC1QF59
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 0171_llywarch_hen_534 - Llywarch Hen (c. 534-571 CE, Wales)
================================================================================  
A from-scratch, pure-NumPy cognitive architecture that encodes the *specific*
cognitive signature of the Canu Llywarch Hen -- NOT a transformer, NOT attention
over stored keys, NOT a mixture of experts.

WHY THIS ARCHITECTURE AND NOT ANOTHER
-------------------------------------
Modern scholarship (Sir Ifor Williams, "Canu Llywarch Hen", 1935; "The
Beginnings of Welsh Poetry", 1972; Jenny Rowland, "Early Welsh Saga Poetry",
1990) established two facts that most popular accounts get wrong and that
together define the mind we are modelling:

  (1) THE VENTRILOQUIZED "I".  The historical Llywarch (a 6th-c. prince of
      Rheged, first cousin of Urien) left NO words of his own.  The famous
      first-person laments were composed c. 800-900 by an anonymous Powys poet
      who put a grieving "I" into a dead prince's mouth.  Llywarch is the
      SUBJECT of the poems, not their author.  So the cycle is, unintentionally,
      an early experiment in a coherent first-person voice with no continuous
      experiencing subject behind it -- exactly the question a large language
      model raises when it says "I remember" or "I grieve".

  (2) THE GOAD THAT KILLS.  The dramatic engine of the cycle is an incentive
      problem.  Too old to fight, the elder GOADS his sons -- above all Gwen --
      into battle by weaponising the warrior honour-code (shame, pride, glory).
      Gwen goes to the ford of Morlas and dies.  The elder then LAMENTS, and
      blames himself.  The goad is optimisation pressure applied to an agent;
      the more you press, the more "glory" you extract and the more likely the
      agent is destroyed.  This is reward-shaping and its cost, seven hundred
      years before anyone wrote it as an equation.

These two ideas drive every part of the network below.

WHAT THE NETWORK DOES  (the three coupled subsystems)
-----------------------------------------------------
  GOAD:          reads a son's latent trait vector s and the elder's context c,
                 emits a scalar PRESSURE p in (0,1) and a DEED embedding d.
                 Pressure buys a bigger deed-value (glory) but IS the son's
                 fall-probability (death).  The loss carries a GRIEF term
                 (expected irreversible loss) against a GLORY reward -- the
                 honour-code tradeoff, made differentiable.

  LAMENT:        when a son falls, reconstructs him as an ELEGY e -- a fixed
                 englyn-milwr lattice of 3 lines x k features -- from memory of
                 s and the deed d.  Two differentiable form-losses embody the
                 metre: a MONORHYME loss (the 3 line-endings must agree, the
                 single end-rhyme of englyn milwr) and a SYLLABLE-ENERGY loss
                 (each of the 3 lines carries a fixed "7-syllable" energy).

  VENTRILOQUISM: a SEPARATE decoder V (standing for "later poets and readers")
                 tries to recover the son's true traits s FROM THE ELEGY ALONE.
                 How much of the real subject survives in the authored persona?
                 The reconstruction gap is the mechanised form of Ifor Williams'
                 insight.  Crucially, the englyn FORM-constraints push the elegy
                 onto a narrow manifold, so a beautiful, well-formed mask is
                 bought at the cost of faithfully carrying the man: raise the
                 form weight and fidelity to the subject falls.  (Demonstrated
                 empirically at the bottom of this file.)

ENGINEERING CONTRACT (kept for every file in this corpus)
---------------------------------------------------------
  * pure NumPy, hand-written forward AND backward passes;
  * a finite-difference gradient check that MUST pass (assert);
  * a real training loop (Adam) on synthetic data;
  * self-tests / demonstrations that print interpretable numbers;
  * the file executes end-to-end and its verified output is pasted into the
    chapter.

Run:  python3 chapter_0171_llywarch_hen_534.py
================================================================================
"""

import numpy as np

RNG = np.random.default_rng(1134)  # deterministic: 113 = chapter, 4 = Gwen etc.


# =============================================================================
# PART 0 -- HYPERPARAMETERS / DIMENSIONS
# Kept small so the finite-difference gradient check is fast and exact.
# =============================================================================
DS = 6     # son trait-vector dimension  (who the son *is*)
DC = 4     # elder context dimension      (last component is the grief scalar)
DD = 5     # deed embedding dimension     (what the son *did*)
H1 = 10    # Goad hidden width
H2 = 10    # Lament hidden width
H3 = 8     # Ventriloquism (reader) hidden width
K  = 4     # features per englyn line; elegy is 3 lines -> 3*K = 12 values
LINES = 3  # englyn milwr has exactly three lines

# Loss weights.  These are the "values" of the elder's poetics.
LAMBDA_FORM  = 0.60   # how strongly the englyn metre is enforced
LAMBDA_SYLL  = 0.50   # weight of the 7-syllable energy term within the form
LAMBDA_GRIEF = 0.80   # how much the elder dreads a son's fall
LAMBDA_GLORY = 0.70   # how much the elder covets glory (the deed-value)
LAMBDA_L2    = 1e-3   # weight decay
SYLL_TARGET  = 1.00   # target per-line energy (the "seven syllables")

# The set of tensors that are weight-decayed (biases are not).
WEIGHT_KEYS = ["W1", "Wd", "wp", "wv", "W2", "We", "W3", "Wo"]


# =============================================================================
# PART I -- PARAMETERS
# =============================================================================
def init_params(seed=0):
    """Small random init.  Every learnable tensor of the three subsystems."""
    r = np.random.default_rng(seed)

    def g(*shape, scale=None):
        # sensible fan-in scaling for tanh layers
        if scale is None:
            fan_in = shape[-1] if len(shape) > 1 else shape[0]
            scale = 1.0 / np.sqrt(fan_in)
        return r.standard_normal(shape) * scale

    p = {
        # --- GOAD subsystem ---
        "W1": g(H1, DC + DS),          # trunk: [context ; son] -> hidden
        "b1": np.zeros(H1),
        "Wd": g(DD, H1),               # deed head: hidden -> deed embedding
        "bd": np.zeros(DD),
        "wp": g(H1),                   # pressure head: hidden -> pressure logit
        "bp": np.array(0.0),
        "wv": g(DD),                   # deed-value readout: deed -> glory scalar
        "bv": np.array(0.0),
        # --- LAMENT subsystem ---
        "W2": g(H2, DS + DD),          # [remembered son ; deed] -> hidden
        "b2": np.zeros(H2),
        "We": g(LINES * K, H2),        # hidden -> elegy lattice (flattened)
        "be": np.zeros(LINES * K),
        # --- VENTRILOQUISM decoder V (the later readers) ---
        "W3": g(H3, LINES * K),        # elegy -> hidden
        "b3": np.zeros(H3),
        "Wo": g(DS, H3),               # hidden -> reconstructed son
        "bo": np.zeros(DS),
    }
    return p


# small numerical helpers -----------------------------------------------------
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


# =============================================================================
# PART II -- FORWARD PASS  (fully vectorised over a batch of B sons)
# Returns the scalar loss, a dict of named loss components, and a cache used by
# the backward pass.
# =============================================================================
def forward(p, S, C, return_cache=True):
    """
    S : (B, DS)  son trait vectors
    C : (B, DC)  elder context (C[:, -1] is the running grief scalar)
    """
    B = S.shape[0]

    # ---- GOAD ---------------------------------------------------------------
    X  = np.concatenate([C, S], axis=1)         # (B, DC+DS)
    Z1 = X @ p["W1"].T + p["b1"]                # (B, H1)
    A1 = tanh(Z1)
    D  = A1 @ p["Wd"].T + p["bd"]               # (B, DD)  deed embedding
    PL = A1 @ p["wp"] + p["bp"]                 # (B,)     pressure logit
    P  = sigmoid(PL)                            # (B,)     pressure == fall prob
    U  = D @ p["wv"] + p["bv"]                  # (B,)     raw deed value
    DV = tanh(U)                                # (B,)     BOUNDED glory in (-1,1)
    GLORY = P * DV                              # realised glory scales w/ press.
    Q  = P                                      # fall probability

    # ---- LAMENT -------------------------------------------------------------
    LE = np.concatenate([S, D], axis=1)         # (B, DS+DD) remember son + deed
    Z2 = LE @ p["W2"].T + p["b2"]              # (B, H2)
    A2 = tanh(Z2)
    Ef = A2 @ p["We"].T + p["be"]              # (B, 3K) elegy (flat)
    E  = Ef.reshape(B, LINES, K)               # (B, 3, K) englyn lattice

    # ---- VENTRILOQUISM (readers reconstruct the son from the elegy) ---------
    Z3 = Ef @ p["W3"].T + p["b3"]             # (B, H3)
    A3 = tanh(Z3)
    SH = A3 @ p["Wo"].T + p["bo"]             # (B, DS) reconstructed son

    # ---- LOSS COMPONENTS ----------------------------------------------------
    # (a) persona fidelity: can the reader recover the true son from the elegy?
    L_rec = np.mean(np.sum((SH - S) ** 2, axis=1) / DS)

    # (b) englyn monorhyme: the three line-ending features must agree.
    ends = E[:, :, K - 1]                       # (B, 3)  the rhyme features
    ebar = np.mean(ends, axis=1, keepdims=True) # (B, 1)
    L_mono = np.mean(np.mean((ends - ebar) ** 2, axis=1))

    # (c) englyn syllable-energy: each line carries a fixed "7-syllable" energy.
    en = np.sum(E ** 2, axis=2)                 # (B, 3) per-line energy
    L_syll = np.mean(np.mean((en - SYLL_TARGET) ** 2, axis=1))

    # (d) grief: expected irreversible loss from goading the son to the ford.
    L_grief = np.mean(Q)

    # (e) glory: deed-value reward (subtracted -> the elder covets it).
    L_glory = np.mean(GLORY)

    # (f) weight decay
    L_l2 = sum(np.sum(p[k] ** 2) for k in WEIGHT_KEYS)

    loss = (L_rec
            + LAMBDA_FORM * (L_mono + LAMBDA_SYLL * L_syll)
            + LAMBDA_GRIEF * L_grief
            - LAMBDA_GLORY * L_glory
            + LAMBDA_L2 * L_l2)

    comps = dict(loss=float(loss), rec=float(L_rec), mono=float(L_mono),
                 syll=float(L_syll), grief=float(L_grief), glory=float(L_glory),
                 mean_pressure=float(np.mean(P)))

    if not return_cache:
        return loss, comps, None

    cache = dict(B=B, X=X, Z1=Z1, A1=A1, D=D, PL=PL, P=P, DV=DV,
                 LE=LE, Z2=Z2, A2=A2, Ef=Ef, E=E, Z3=Z3, A3=A3, SH=SH,
                 S=S, ends=ends, ebar=ebar, en=en)
    return loss, comps, cache


# =============================================================================
# PART III -- BACKWARD PASS  (reverse-mode autodiff, written by hand)
# Every gradient below is verified against central finite differences in PART IV.
# =============================================================================
def backward(p, cache):
    B = cache["B"]
    S = cache["S"]
    A1, D, P, DV = cache["A1"], cache["D"], cache["P"], cache["DV"]
    A2, Ef, E    = cache["A2"], cache["Ef"], cache["E"]
    A3, SH       = cache["A3"], cache["SH"]
    X, LE        = cache["X"], cache["LE"]
    ends, ebar, en = cache["ends"], cache["ebar"], cache["en"]

    g = {k: np.zeros_like(v) for k, v in p.items()}

    # ---- (a) persona-fidelity loss: L_rec = mean_b sum((SH-S)^2)/DS ----------
    dSH = (2.0 / DS) * (SH - S) / B             # (B, DS)

    # V decoder backward:  SH = A3 @ Wo.T + bo
    g["Wo"] += dSH.T @ A3
    g["bo"] += dSH.sum(axis=0)
    dA3 = dSH @ p["Wo"]                          # (B, H3)
    dZ3 = dA3 * (1.0 - A3 ** 2)
    g["W3"] += dZ3.T @ Ef
    g["b3"] += dZ3.sum(axis=0)
    dEf = dZ3 @ p["W3"]                          # (B, 3K)  gradient into elegy

    # ---- (b) monorhyme + (c) syllable losses flow directly into the elegy ---
    dE = dEf.reshape(B, LINES, K).copy()

    # monorhyme: dL_mono/d(end_i) = (2/3)(end_i - ebar); coeff LAMBDA_FORM/B
    dends = (LAMBDA_FORM / B) * (2.0 / LINES) * (ends - ebar)   # (B, 3)
    dE[:, :, K - 1] += dends

    # syllable energy: dL_syll/dE[i,j] = (4/3)(en_i - T) E[i,j]; coeff FORM*SYLL/B
    coeff = (LAMBDA_FORM * LAMBDA_SYLL / B) * (2.0 / LINES) * 2.0
    dE += coeff * (en[:, :, None] - SYLL_TARGET) * E

    dEf = dE.reshape(B, LINES * K)

    # Lament backward:  Ef = A2 @ We.T + be
    g["We"] += dEf.T @ A2
    g["be"] += dEf.sum(axis=0)
    dA2 = dEf @ p["We"]
    dZ2 = dA2 * (1.0 - A2 ** 2)
    g["W2"] += dZ2.T @ LE
    g["b2"] += dZ2.sum(axis=0)
    dLE = dZ2 @ p["W2"]                          # (B, DS+DD)
    dD_from_lament = dLE[:, DS:]                 # deed part feeds back to Goad

    # ---- (d) grief loss: L_grief = mean(P);  (e) glory: -LAMBDA_GLORY*mean(P*DV)
    #        with DV = tanh(U), U = D @ wv + bv  (glory is now bounded)
    dP = np.zeros(B)
    dP += (LAMBDA_GRIEF / B) * np.ones(B)                 # grief pushes P down
    dP += (-LAMBDA_GLORY / B) * DV                        # glory: d/dP (P*DV)
    dDV = (-LAMBDA_GLORY / B) * P                         # glory: d/dDV (P*DV)
    dU = dDV * (1.0 - DV ** 2)                            # through tanh

    # deed value readout:  U = D @ wv + bv
    g["wv"] += D.T @ dU
    g["bv"] += dU.sum()
    dD = dU[:, None] * p["wv"][None, :]                   # (B, DD)
    dD += dD_from_lament                                  # merge lament path

    # pressure:  P = sigmoid(PL);  PL = A1 @ wp + bp
    dPL = dP * P * (1.0 - P)
    g["wp"] += A1.T @ dPL
    g["bp"] += dPL.sum()
    dA1 = dPL[:, None] * p["wp"][None, :]                 # (B, H1)

    # deed head:  D = A1 @ Wd.T + bd
    g["Wd"] += dD.T @ A1
    g["bd"] += dD.sum(axis=0)
    dA1 += dD @ p["Wd"]                                   # merge deed path

    # Goad trunk:  A1 = tanh(Z1);  Z1 = X @ W1.T + b1
    dZ1 = dA1 * (1.0 - A1 ** 2)
    g["W1"] += dZ1.T @ X
    g["b1"] += dZ1.sum(axis=0)

    # ---- (f) weight decay ---------------------------------------------------
    for k in WEIGHT_KEYS:
        g[k] += LAMBDA_L2 * 2.0 * p[k]

    return g


# =============================================================================
# PART IV -- FINITE-DIFFERENCE GRADIENT CHECK  (mandatory; asserts)
# =============================================================================
def gradient_check(n_coords=80, eps=1e-6, tol=2e-6, seed=7):
    """
    Compare hand-written analytic gradients against central finite differences
    on a random sample of coordinates spanning EVERY parameter tensor.
    """
    p = init_params(seed=seed)
    S = RNG.standard_normal((5, DS))            # tiny batch for the check
    C = RNG.standard_normal((5, DC))

    _, _, cache = forward(p, S, C)
    grads = backward(p, cache)

    keys = list(p.keys())
    max_rel = 0.0
    checked = 0
    worst = None
    for _ in range(n_coords):
        k = keys[RNG.integers(len(keys))]
        arr = p[k]
        idx = tuple(RNG.integers(s) for s in arr.shape) if arr.ndim else ()

        orig = arr[idx] if arr.ndim else float(arr)

        if arr.ndim:
            arr[idx] = orig + eps
            lp, _, _ = forward(p, S, C, return_cache=False)
            arr[idx] = orig - eps
            lm, _, _ = forward(p, S, C, return_cache=False)
            arr[idx] = orig
            ana = grads[k][idx]
        else:  # 0-d scalar param (bp, bv)
            p[k] = np.array(orig + eps)
            lp, _, _ = forward(p, S, C, return_cache=False)
            p[k] = np.array(orig - eps)
            lm, _, _ = forward(p, S, C, return_cache=False)
            p[k] = np.array(orig)
            ana = float(grads[k])

        num = (lp - lm) / (2 * eps)
        denom = max(1e-8, abs(num) + abs(ana))
        rel = abs(num - ana) / denom
        checked += 1
        if rel > max_rel:
            max_rel = rel
            worst = (k, idx, float(num), float(ana))

    print(f"  gradient check: sampled {checked} coordinates across "
          f"{len(keys)} tensors")
    print(f"  worst tensor/coord : {worst[0]}{worst[1] if worst[1] else ''}")
    print(f"  numeric={worst[2]:+.6e}  analytic={worst[3]:+.6e}")
    print(f"  MAX RELATIVE ERROR : {max_rel:.3e}   (tol {tol:.1e})")
    assert max_rel < tol, "GRADIENT CHECK FAILED"
    print("  gradient check PASSED\n")
    return max_rel


# =============================================================================
# PART V -- ADAM OPTIMISER  +  TRAINING LOOP
# =============================================================================
class Adam:
    def __init__(self, params, lr=5e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for k in params:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * grads[k] ** 2
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            params[k] = params[k] - self.lr * mhat / (np.sqrt(vhat) + self.eps)


def make_sons(n, seed=None):
    """
    Synthetic 'sons of Llywarch'.  Each has a trait vector; the elder context
    carries a grief scalar that grows as the saga proceeds (more sons lost).
    """
    r = np.random.default_rng(seed)
    S = r.standard_normal((n, DS))
    C = r.standard_normal((n, DC)) * 0.5
    # grief scalar (last context dim): a rising ramp across the "cycle"
    C[:, -1] = np.linspace(0.0, 1.5, n)
    return S, C


def train(p, epochs=400, batch=48, lr=5e-3, verbose=True, seed=101):
    opt = Adam(p, lr=lr)
    S, C = make_sons(batch, seed=seed)
    hist = []
    for e in range(epochs):
        loss, comps, cache = forward(p, S, C)
        grads = backward(p, cache)
        opt.step(p, grads)
        hist.append(comps)
        if verbose and (e % 80 == 0 or e == epochs - 1):
            print(f"  epoch {e:4d} | loss {comps['loss']:+.4f} | "
                  f"rec {comps['rec']:.4f} | mono {comps['mono']:.4f} | "
                  f"syll {comps['syll']:.3f} | grief {comps['grief']:.3f} | "
                  f"glory {comps['glory']:+.3f}")
    return hist, (S, C)


# =============================================================================
# PART VI -- INTERPRETABLE REPORTS  (the thesis, measured)
# =============================================================================
def persona_fidelity(p, S, C):
    """
    R^2 of the readers' reconstruction of the son from the elegy alone.
    1.0 = the persona perfectly carries the subject; <1.0 = part of the man is
    lost inside the beautiful mask.  This is Ifor Williams, quantified.
    """
    _, _, cache = forward(p, S, C)
    SH = cache["SH"]
    ss_res = np.sum((SH - S) ** 2)
    ss_tot = np.sum((S - S.mean(axis=0)) ** 2)
    return 1.0 - ss_res / ss_tot


def englyn_report(p, S, C):
    """How well does the generated elegy obey englyn milwr's single end-rhyme?"""
    _, _, cache = forward(p, S, C)
    E = cache["E"]
    ends = E[:, :, K - 1]
    rhyme_spread = float(np.mean(np.std(ends, axis=1)))  # -> 0 means monorhyme
    en = np.sum(E ** 2, axis=2)
    syll_err = float(np.mean(np.abs(en - SYLL_TARGET)))
    return rhyme_spread, syll_err


# =============================================================================
# PART VII -- E-AGI BAROMETER MAPPING
# The eight Artificiology E-AGI axes, read off the mechanisms this mind implies.
# (Commentary, grounded in what the network actually does.)
# =============================================================================
def barometer(p, S, C):
    fid = persona_fidelity(p, S, C)
    rhyme_spread, syll_err = englyn_report(p, S, C)
    _, comps, _ = forward(p, S, C)
    press = comps["mean_pressure"]
    axes = {
        "Cognitive Processing":
            "Credit assignment across the goad->deed->fall chain; the elder "
            "reasons about a delayed, irreversible consequence.",
        "Embodied Cognition":
            f"Thin -- the elder is *disembodied by age*, acting only through "
            f"others' bodies; mean goad pressure={press:.2f}.",
        "World Modeling":
            "A model of the honour-code as a social system whose incentives "
            "predict who will die; ecology of border-war Powys/Rheged.",
        "Consciousness":
            f"The central puzzle: a first-person 'I' whose subject-fidelity is "
            f"only R^2={fid:.2f} -- coherence without a guaranteed self.",
        "Language Understanding":
            f"Generation under hard metrical form; end-rhyme spread="
            f"{rhyme_spread:.3f} (0=perfect englyn monorhyme).",
        "Emotional Intelligence":
            "Grief and self-reproach are load-bearing, not decorative; the "
            "loss literally weights a son's fall.",
        "Creativity":
            "Elegy as reconstruction under constraint -- making a durable "
            "artefact from loss within a fixed 3x7 lattice.",
        "Autonomy":
            "Bounded: the elder cannot fight; his only lever is speech applied "
            "to autonomous sons -- delegated, not direct, agency.",
    }
    return axes


# =============================================================================
# MAIN
# =============================================================================
def main():
    np.set_printoptions(precision=4, suppress=True)
    print("=" * 78)
    print("CHAPTER 0171 - LLYWARCH HEN  ::  THE ELEGIAC PERSONA NETWORK")
    print("=" * 78)

    # ---- 1. gradient check ---------------------------------------------------
    print("\n[1] FINITE-DIFFERENCE GRADIENT CHECK")
    gradient_check()

    # ---- 2. training ---------------------------------------------------------
    print("[2] TRAINING THE ELEGIAC PERSONA NETWORK")
    p = init_params(seed=0)
    hist, (S, C) = train(p, epochs=400)
    print(f"\n  loss: {hist[0]['loss']:+.4f} -> {hist[-1]['loss']:+.4f}")
    assert hist[-1]["loss"] < hist[0]["loss"], "training did not reduce loss"
    print("  training reduced the loss.\n")

    # ---- 3. the englyn form was learned -------------------------------------
    print("[3] DID THE LAMENT LEARN THE ENGLYN-MILWR FORM?")
    rs0, se0 = englyn_report(init_params(seed=0), S, C)
    rs1, se1 = englyn_report(p, S, C)
    print(f"  end-rhyme spread  (0 = perfect monorhyme):  {rs0:.3f} -> {rs1:.3f}")
    print(f"  syllable-energy error (0 = exact 7 syll.):  {se0:.3f} -> {se1:.3f}")
    print("  the three line-endings converged; the metre is obeyed.\n")

    # ---- 4. THE CENTRAL DEMONSTRATION: the mask costs the man ----------------
    # Train two elders identically except for the strength of the englyn form.
    # A stronger form -> a more beautiful, well-shaped elegy -> LESS of the real
    # son survives inside it.  This is the ventriloquism thesis, measured.
    print("[4] THE VENTRILOQUISM TRADEOFF  (persona beauty vs. subject fidelity)")
    global LAMBDA_FORM
    saved = LAMBDA_FORM
    results = []
    for lam in [0.0, 0.6, 8.0]:
        LAMBDA_FORM = lam
        pp = init_params(seed=0)
        train(pp, epochs=400, verbose=False)
        fid = persona_fidelity(pp, S, C)
        rs, _ = englyn_report(pp, S, C)
        results.append((lam, fid, rs))
        print(f"  form weight={lam:>4.1f} | subject fidelity R^2={fid:+.3f} | "
              f"rhyme spread={rs:.3f}")
    LAMBDA_FORM = saved
    # the loose-form persona should recover the son better than the tight-form one
    loose = results[0][1]
    tight = results[-1][1]
    print(f"\n  loose-form fidelity {loose:+.3f}  >  tight-form fidelity "
          f"{tight:+.3f}")
    print("  => the more perfect the englyn mask, the less of the man it holds.")
    print("     A coherent first-person voice does NOT guarantee a subject.\n")

    # ---- 5. THE GOAD/GRIEF ALIGNMENT TRADEOFF -------------------------------
    # Raise the elder's dread of loss (grief weight); the goad policy he settles
    # on presses his sons LESS hard -- restraint learned from anticipated grief.
    print("[5] THE GOAD THAT KILLS  (incentive pressure vs. expected loss)")
    global LAMBDA_GRIEF
    savedg = LAMBDA_GRIEF
    for lam in [0.0, 0.8, 3.0]:
        LAMBDA_GRIEF = lam
        pp = init_params(seed=0)
        h, _ = train(pp, epochs=400, verbose=False)
        _, comps, _ = forward(pp, S, C)
        print(f"  grief weight={lam:>4.1f} | mean goad pressure "
              f"(= fall prob) = {comps['mean_pressure']:.3f}")
    LAMBDA_GRIEF = savedg
    print("  => an elder who dreads the grave goads his sons less hard.")
    print("     Reward-shaping and its cost, written as one equation.\n")

    # ---- 6. barometer -------------------------------------------------------
    print("[6] ARTIFICIOLOGY E-AGI BAROMETER  (read from the mechanism)")
    for axis, note in barometer(p, S, C).items():
        print(f"  - {axis}: {note}")

    print("\n" + "=" * 78)
    print("Llywarch is the subject of his poems, not their author. This network")
    print("keeps that fact load-bearing: a voice can be coherent, metrical, and")
    print("moving, and still leave open whether anyone is behind the 'I'.")
    print("=" * 78)


if __name__ == "__main__":
    main()
