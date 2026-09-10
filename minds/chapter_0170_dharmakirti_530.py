#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 chapter_0170_dharmakirti_530 - Dharmakirti (c. 550-610 CE, Nalanda)
 THE APOHA ENGINE  —  a trainable cognitive architecture built from
 Dharmakirti's epistemology (Dharmakirti, c. 550-610 CE, Nalanda)
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 9 Minds 161 - 180 Available on Amazon https://www.amazon.com/dp/B0HJC1QF59
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 0170_dharmakirti_530 - Dharmakirti (c. 550-610 CE, Nalanda)
================================================================================  

WHAT THIS FILE IS
-----------------
This is not a demo and not a metaphor. It is a working, from-scratch (pure
NumPy) neural architecture whose *mechanism* is Dharmakirti's theory of mind,
with hand-derived gradients, a finite-difference gradient check, a real
training loop, and a suite of self-tests that each probe one doctrine.

THE ONE IDEA THE WHOLE MACHINE IS BUILT ON
------------------------------------------
Dharmakirti holds that the world contains no kinds. It contains only
particulars (svalaksana), each momentary, each doing exactly what it does.
"Cow", "fire", "blue" are not out there to be found; they are fictions the
mind manufactures. And it manufactures them *by exclusion*: the concept "cow"
is not a positive template of cowness, it is anyapoha, "the exclusion of what
is other" -- non-non-cow. A concept has no content of its own. It is nothing
but a system of differences.

So the central layer of this machine has NO positive parameters.
Concepts here are stored as an ANTISYMMETRIC PAIRWISE EXCLUSION FIELD:

        excl(k, j)(a) =  w[k,j] . a + b[k,j]        with  w[j,k] = -w[k,j]

        belong(k)(a)  =  softmin over all j != k of excl(k, j)(a)

To belong to concept k is nothing but to be excluded from every non-k. There
is no prototype for k anywhere in memory. Delete the differences and nothing
remains -- which is exactly Dharmakirti's claim, and which SELF-TEST 4
verifies numerically by zeroing the field and watching every concept in the
machine evaporate at once.

The exclusion field is trained by a PURELY NEGATIVE LOSS. The supervision
never says "this is fire". It only ever says "not water, not grain, not bell".
Membership is the residue.

WHAT LICENSES THE FICTIONS, THEN?
---------------------------------
Nothing about truth. Only *arthakriya* -- causal efficacy. A cognition is a
pramana (a source of knowledge) if and only if it is AVISAMVADIN: non-belying.
It does not deceive you when you act on it. So the top-level objective of this
machine is not label accuracy. It is: predict what the world will actually do
when I act. Concepts survive only if acting under them does not get you burned.

THE FIREWALL
------------
Dharmakirti will not accept a reason merely because it correlates. A good
reason must have svabhavapratibandha -- a nexus grounded in identity
(svabhava-hetu) or causation (karya-hetu), or it must be a non-apprehension of
the perceptible (anupalabdhi-hetu). There is no fourth kind. Mere co-occurrence,
however perfect, is anaikantika -- inconclusive.

Implemented, this becomes an inference gate that refuses to license a
correlation unless its co-absence (vyatireka) survives INTERVENTION. That is
Dharmakirti's anvaya/vyatireka examination read as a do-operator. SELF-TEST 6
plants a shortcut feature that correlates perfectly with one class and shows
the gate rejecting it while accepting the genuine causal reason -- and
SELF-TEST 7 shows that training under the same discipline is what makes the
network itself robust when the shortcut is broken.

THE PARTS
---------
  KsanikaWorld        the world: momentary particulars, causal powers, occlusion
  Pratyaksa           non-conceptual perception -> the bare image (akara)
  ApohaField          concepts as pure difference (the antisymmetric field)
  ArthakriyaHead      predict the effect of acting: the only warrant there is
  Svasamvedana        reflexive awareness: the cognition reports on itself
  Anupalabdhi         three-valued absence: PRESENT / ABSENT / UNDETERMINED
  Trairupya           the three marks + the admissibility gate on reason-kinds
  Vadanyaya           counterexample search, retraction, points of defeat
  Santanantara        inference to another mindstream

Every one of these is exercised by a self-test at the bottom of the file.

RUN:  python3 chapter_0170_dharmakirti_530.py
================================================================================
"""

import numpy as np

RNG = np.random.default_rng(530)          # seed: his (contested) floruit

# ==============================================================================
# 0.  THE WORLD  (Ksanika-loka: the world of the momentary)
# ==============================================================================
# Dharmakirti's world has three properties this generator takes literally.
#
#  (1) NO NATURAL KINDS.  In Pramanavarttika III.73-74 he points out that many
#      different plants all reduce fever, yet share no common intrinsic feature
#      that explains it. They are grouped ONLY by having the same causal power.
#      So: each class here is generated by TWO DISJOINT feature mechanisms. A
#      class is a union of two far-apart lumps in feature space. Any learner
#      that looks for a shared positive template will fail. (SELF-TEST 3.)
#
#  (2) THINGS PERISH.  Each particular is a fresh draw; nothing persists.
#
#  (3) THINGS CAN BE OUT OF REACH.  Dharmakirti's three remotenesses
#      (viprakarsa) -- of place, of time, of nature -- mean some things simply
#      cannot be perceived. Here that is an occlusion mask. The correct answer
#      about an occluded thing is not "absent". It is "I cannot say".
#
#  Plus one modern addition, in his spirit: a SHORTCUT channel that correlates
#  perfectly with class 0 in observation but is causally inert. He would call
#  a reason built on it anaikantika. We will watch his gate throw it out.

N_CLASS   = 4                              # FIRE, WATER, GRAIN, BELL
N_MODE    = 2                              # two mechanisms per kind -- ANTIPODAL
D_SENSE   = 16                             # genuine sense channels
IDX_SHORT = D_SENSE                        # the shortcut channel
D_IN      = D_SENSE + 1

CLASS_NAMES  = ["FIRE", "WATER", "GRAIN", "BELL"]
ACTION_NAMES = ["STRIKE", "POUR", "TASTE"]
EFFECT_NAMES = ["BURN", "DOUSE", "NOURISH", "RING", "NOTHING"]
N_ACT, N_EFF = len(ACTION_NAMES), len(EFFECT_NAMES)

# The causal powers (sakti) of the world. effect_of[class][action] -> effect.
# This table is the world's ground truth; the machine never sees it.
NOTHING = 4
EFFECT_OF = np.array([
    # STRIKE   POUR      TASTE
    [   0,        NOTHING,  0      ],   # FIRE  : burns you either way
    [   NOTHING,  1,        NOTHING],   # WATER : douses when poured
    [   NOTHING,  NOTHING,  2      ],   # GRAIN : nourishes when tasted
    [   3,        NOTHING,  NOTHING],   # BELL  : rings when struck
], dtype=np.int64)

# The mechanisms. Each kind has exactly two, and they are OPPOSITE:
# mode 0 is +v_k, mode 1 is -v_k. Two particulars of the same kind may
# therefore share NOTHING WHATEVER as a feature -- they are as far apart as it
# is possible to be -- and yet they have the same causal power. This is
# Pramanavarttika III.73-74 built into the physics: the various febrifuge
# plants cure fever, and there is no common property that explains it.
#
# The consequence is exact and testable: the CENTROID of every kind is the
# ORIGIN. Any learner that represents a kind by a positive shared template --
# a real universal, a class prototype, a single weight vector -- is looking for
# something that is not there, and will find nothing.
_V = RNG.standard_normal((N_CLASS, D_SENSE))
_V /= np.linalg.norm(_V, axis=1, keepdims=True)


def make_world(n, rng, p_occlude=0.0, intervene_shortcut=False):
    """Emit n momentary particulars.

    Returns x (n,D_IN), y (n,), u (n,), e (n,), occluded (n,).

    intervene_shortcut=True performs do(mark := random): it severs the
    shortcut's correlation with FIRE while leaving every real causal power of
    the world untouched. This is the interventional half of Dharmakirti's
    anvaya/vyatireka examination -- the only thing that tells a reason with a
    real nexus apart from a reason that merely rides along.
    """
    y = rng.integers(0, N_CLASS, size=n)
    sign = rng.choice([-1.0, 1.0], size=n)                 # which of the two mechanisms
    amp = 1.0 + 0.20 * rng.standard_normal(n)

    x = np.zeros((n, D_IN))
    x[:, :D_SENSE] = (sign * amp)[:, None] * _V[y]         # +v_k or -v_k
    x[:, :D_SENSE] += 0.15 * rng.standard_normal((n, D_SENSE))

    # The shortcut: rides perfectly on FIRE in observation, and causes nothing.
    if intervene_shortcut:
        x[:, IDX_SHORT] = rng.integers(0, 2, size=n).astype(float)     # do(.)
    else:
        x[:, IDX_SHORT] = (y == 0).astype(float)

    # Occlusion (viprakarsa: remoteness of place, time, or nature).
    occluded = rng.random(n) < p_occlude
    if occluded.any():
        idx = np.where(occluded)[0]
        x[idx, :D_SENSE] = 0.04 * rng.standard_normal((len(idx), D_SENSE))
        x[idx, IDX_SHORT] = 0.0          # the mark is gone too: nothing is given

    u = rng.integers(0, N_ACT, size=n)
    e = EFFECT_OF[y, u]
    return x, y, u, e, occluded


def is_perceptible(x, theta=0.50):
    """drsya: is this thing even in range of the senses?

    Dharmakirti's anupalabdhi-hetu only licenses an assertion of absence when
    the thing, IF present, WOULD have been perceived. Here that condition is
    read straight off the input: if no sense channel carries appreciable
    energy, the senses are simply not in contact, and no absence may be
    asserted. This single check is the machine's whole defence against
    confabulating a negative.
    """
    return np.linalg.norm(x[:, :D_SENSE], axis=1) > theta


# ==============================================================================
# 1.  PARAMETERS
# ==============================================================================
# Wp, bp : Pratyaksa   -- perception, produces the bare image (akara)
# Wraw,braw: ApohaField -- the exclusion field (antisymmetrised on use)
# Ceff   : Arthakriya  -- what happens if I act thus upon a thing of this kind
# gam, c : Svasamvedana -- the two scalars of reflexive awareness

H_DIM = 48          # dimensionality of the akara
BETA  = 4.0         # sharpness of the softmin ("how hard is the exclusion?")
BETA_P = 3.0        # sharpness of the same-judgment (ekapratyavamarsa)
MARGIN = 1.0        # how far a thing must be pushed out of what it is not


def init_params(rng):
    return {
        "Wp":   0.30 * rng.standard_normal((D_IN, H_DIM)),
        "bp":   np.zeros(H_DIM),
        "Wraw": 0.30 * rng.standard_normal((N_CLASS, N_CLASS, H_DIM)),
        "braw": np.zeros((N_CLASS, N_CLASS)),
        "Ceff": 0.30 * rng.standard_normal((N_CLASS, N_ACT, N_EFF)),
        "gam":  np.array([1.0]),
        "c":    np.array([0.0]),
    }


def antisymmetrise(Wraw, braw):
    """w[k,j] = -w[j,k], and w[k,k] = 0, by construction.

    This is not decoration. It is the formal content of anyapoha. Because the
    field is antisymmetric, "a is excluded from j" and "a is included in k" are
    literally the same number with opposite sign. There is no separate fact of
    belonging. Belonging is exclusion, read backwards.
    """
    W = Wraw - np.transpose(Wraw, (1, 0, 2))
    B = braw - braw.T
    return W, B


# ==============================================================================
# 2.  FORWARD PASS
# ==============================================================================

def logsumexp(z, axis):
    m = np.max(z, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(z - m), axis=axis, keepdims=True))).squeeze(axis)


def softmax(z, axis=-1):
    z = z - np.max(z, axis=axis, keepdims=True)
    ez = np.exp(z)
    return ez / np.sum(ez, axis=axis, keepdims=True)


DIAG = np.eye(N_CLASS, dtype=bool)


def forward(P, x, u, need_cache=True):
    """One momentary cognition, start to finish."""
    # --- Pratyaksa: perception. Non-conceptual, non-linguistic. No labels here.
    h = x @ P["Wp"] + P["bp"]
    a = np.tanh(h)                                        # the akara (image)

    # --- Apoha: the concept layer, which contains no concepts.
    W, B = antisymmetrise(P["Wraw"], P["braw"])
    Es = np.einsum("nh,kjh->nkj", a, W) + B[None, :, :]   # excl scores (n,K,K)

    t = -BETA * Es
    t = np.where(DIAG[None, :, :], -1e30, t)              # a thing is not other than itself
    lse = logsumexp(t, axis=2)                            # (n,K)
    belong = -lse / BETA                                  # softmin_j excl(k,j)
    sm = np.exp(t - lse[:, :, None])                      # softmin weights (n,K,K)

    # --- Ekapratyavamarsa: "the same judgment". Which fiction do I settle on?
    p = softmax(BETA_P * belong, axis=1)                  # (n,K)

    # --- Arthakriya: if I do u to this, what will the world DO?
    #     This -- not any resemblance to a label -- is what warrants the concept.
    Cu = P["Ceff"][:, u, :]                               # (K,n,E)
    logits = np.einsum("nk,kne->ne", p, Cu)               # (n,E)

    cache = None
    if need_cache:
        cache = dict(x=x, u=u, h=h, a=a, W=W, Es=Es, sm=sm,
                     belong=belong, p=p, logits=logits)
    return logits, belong, p, cache


# ------------------------------------------------------------------ Svasamvedana
def svasamvedana(belong, P):
    """Reflexive awareness (svasamvedana): every cognition is self-illuminating.

    Note carefully what this is and is not. It has NO features of its own and
    NO separate critic network. Confidence is read off the cognition's own
    internal structure -- the gap between the concept it settled on and its
    nearest rival, i.e. how decisively the exclusions carved. The cognition
    lights itself up in the act of occurring.

    And note the limit, which Dharmakirti's readers have always insisted on:
    reflexive awareness tells you THAT a cognition is occurring and how firmly.
    It does not tell you that the cognition is JUSTIFIED. It illuminates
    delusions exactly as brightly as it illuminates knowledge. That is why, in
    training, the gradient of the calibration loss is BLOCKED from flowing back
    into `belong` (see backward()). A self-report that could reach back and
    edit the cognition it reports on would not be awareness. It would be
    wishful thinking -- and it would be an AI that learns to feel certain
    instead of learning to be right.
    """
    srt = np.sort(belong, axis=1)
    gap = srt[:, -1] - srt[:, -2]                # top-1 minus top-2
    conf = 1.0 / (1.0 + np.exp(-(P["gam"][0] * gap + P["c"][0])))
    return conf, gap


# ------------------------------------------------------------------ Anupalabdhi
PRESENT, ABSENT, UNDETERMINED = "PRESENT", "ABSENT", "UNDETERMINED"


def anupalabdhi(P, x, tau=0.20):
    """Three-valued judgment of presence and absence.

    The eleven-fold anupalabdhi of the Nyayabindu comes down to one rule:
    you may infer that a thing is absent ONLY from the non-apprehension of
    something that WOULD have been apprehended had it been there
    (drsyanupalabdhi). Where the perceptibility condition fails, the honest
    output is not "no" but "I cannot say" -- samsaya, doubt.

    Almost every hallucination in a modern model is a machine asserting ABSENT
    or PRESENT where Dharmakirti would have it return UNDETERMINED.
    """
    _, belong, _, _ = forward(P, x, np.zeros(len(x), dtype=int), need_cache=False)
    ok = is_perceptible(x)
    out = []
    for i in range(len(x)):
        if not ok[i]:
            out.append((UNDETERMINED, None))          # remote: the senses are silent
        elif belong[i].max() > tau:
            out.append((PRESENT, int(belong[i].argmax())))
        else:
            out.append((ABSENT, None))                # perceptible, and not found
    return out


# ==============================================================================
# 3.  LOSSES  (and why each one is there)
# ==============================================================================

def compute_losses(P, cache, y, e, frozen=None,
                   lam_apoha=1.0, lam_cal=0.3, lam_sattva=0.05):
    """`frozen` makes the stop-gradient real rather than a promise.

    The calibration term must not be able to reach back into the cognition it
    reports on. So the quantities it depends on -- the margin `gap`, and
    whether the act in fact succeeded, `correct` -- are DETACHED: held fixed as
    constants while gradients flow. Passing them in explicitly is what lets the
    finite-difference check in SELF-TEST 1 verify the analytic gradient of the
    very same function the optimiser is descending, instead of a different one.
    """
    n = len(y)
    logits, p, belong, a, Es = (cache["logits"], cache["p"], cache["belong"],
                                cache["a"], cache["Es"])

    # (a) ARTHAKRIYA / AVISAMVADA -- the non-belying loss.
    #     The whole point of a pramana. Not "did you name it right" but
    #     "did the world do what you expected when you acted".
    pe = softmax(logits, axis=1)
    L_eff = -np.mean(np.log(pe[np.arange(n), e] + 1e-12))

    # (b) APOHA -- and it is PURELY NEGATIVE. Read the loop: we only ever push
    #     the thing OUT of the classes it is not. We never once pull it toward
    #     the class it is. Membership is never trained. Membership is what is
    #     left over when everything else has been ruled out.
    mask = np.ones((n, N_CLASS), dtype=bool)
    mask[np.arange(n), y] = False                      # the "others"
    E_true = Es[np.arange(n), y, :]                    # (n,K) excl(y, j)
    z = MARGIN - E_true
    L_apoha = np.sum(np.logaddexp(0.0, z) * mask) / (n * (N_CLASS - 1))

    # (c) SVASAMVEDANA calibration -- the self-report must track being right.
    if frozen is None:
        _, gap = svasamvedana(belong, P)                 # value used, gradient detached
        correct = (logits.argmax(1) == e).astype(float)  # discrete: no gradient anyway
    else:
        gap, correct = frozen
    conf = 1.0 / (1.0 + np.exp(-(P["gam"][0] * gap + P["c"][0])))
    L_cal = -np.mean(correct * np.log(conf + 1e-12) +
                     (1 - correct) * np.log(1 - conf + 1e-12))

    # (d) SATTVANUMANA -- "whatever exists, exists momentarily, because it is
    #     causally efficacious." Turned into a rule on the representation:
    #     a unit of the akara that never varies produces no new effect, and so
    #     by his own criterion of the real (arthakriyasamartha) it does not
    #     exist. This term applies pressure against dead, inert units.
    var = np.var(a, axis=0)
    L_sattva = np.sum(np.maximum(0.0, 0.01 - var))

    total = L_eff + lam_apoha * L_apoha + lam_cal * L_cal + lam_sattva * L_sattva
    parts = dict(eff=L_eff, apoha=L_apoha, cal=L_cal, sattva=L_sattva, total=total)
    return total, parts, (pe, conf, correct, gap, mask, E_true, z, var)


def frozen_of(P, cache, e):
    """Take the detached quantities off a forward pass, once."""
    _, gap = svasamvedana(cache["belong"], P)
    correct = (cache["logits"].argmax(1) == e).astype(float)
    return (gap, correct)


def backward(P, cache, y, e, aux, lam_apoha=1.0, lam_cal=0.3, lam_sattva=0.05):
    """Hand-derived gradients. Every line is checked against finite differences
    in SELF-TEST 1; if any of this were wrong, that test would fail loudly."""
    n = len(y)
    pe, conf, correct, gap, mask, E_true, z, var = aux
    x, u, a, W, sm, p, belong = (cache["x"], cache["u"], cache["a"], cache["W"],
                                 cache["sm"], cache["p"], cache["belong"])
    G = {k: np.zeros_like(v) for k, v in P.items()}

    # ---- (a) arthakriya
    dlogits = pe.copy()
    dlogits[np.arange(n), e] -= 1.0
    dlogits /= n                                              # (n,E)

    Cu = P["Ceff"][:, u, :]                                   # (K,n,E)
    dp = np.einsum("ne,kne->nk", dlogits, Cu)                 # (n,K)
    for i in range(n):
        G["Ceff"][:, u[i], :] += np.outer(p[i], dlogits[i])

    # ---- (c) calibration: trains ONLY gam and c. The gradient is deliberately
    #      NOT propagated into `belong` -- see the note in svasamvedana().
    dL_dconf = (-(correct / (conf + 1e-12)) + (1 - correct) / (1 - conf + 1e-12)) / n
    dconf_dlin = conf * (1 - conf)
    dlin = dL_dconf * dconf_dlin * lam_cal
    G["gam"][0] = np.sum(dlin * gap)
    G["c"][0] = np.sum(dlin)

    # ---- p -> belong  (softmax jacobian)
    dot = np.sum(dp * p, axis=1, keepdims=True)
    dbelong = BETA_P * p * (dp - dot)                          # (n,K)

    # ---- belong -> Es. d belong[k] / d Es[k,j] = softmin weight sm[k,j].
    dEs = dbelong[:, :, None] * sm                             # (n,K,K)

    # ---- (b) apoha (purely negative) -> Es[n, y_n, j]
    coef = -(1.0 / (1.0 + np.exp(-z))) * mask                  # d softplus(z)/dE = -sigmoid(z)
    coef = coef * (lam_apoha / (n * (N_CLASS - 1)))
    for i in range(n):
        dEs[i, y[i], :] += coef[i]

    # ---- Es = a . W[k,j] + B[k,j]
    G_W = np.einsum("nkj,nh->kjh", dEs, a)
    G_B = np.sum(dEs, axis=0)
    da = np.einsum("nkj,kjh->nh", dEs, W)

    # antisymmetrisation is a linear map; push the gradient back through it
    G["Wraw"] = G_W - np.transpose(G_W, (1, 0, 2))
    G["braw"] = G_B - G_B.T

    # ---- (d) sattvanumana, straight onto the akara
    live = (var < 0.01).astype(float)                          # only the dead units
    da += lam_sattva * (-live)[None, :] * 2.0 * (a - a.mean(0, keepdims=True)) / n

    # ---- through tanh into perception
    dh = da * (1.0 - a ** 2)
    G["Wp"] = x.T @ dh
    G["bp"] = dh.sum(0)
    return G


# ==============================================================================
# 4.  SELF-TEST 1 : GRADIENT CHECK   (mandatory; nothing below is trusted if
#     this fails, because a wrong gradient can still make a loss go down)
# ==============================================================================

def loss_only(P, x, y, u, e, frozen):
    _, _, _, cache = forward(P, x, u)
    tot, _, _ = compute_losses(P, cache, y, e, frozen=frozen)
    return tot


def grad_check(verbose=True):
    rng = np.random.default_rng(7)
    P = init_params(rng)
    x, y, u, e, _ = make_world(24, rng)
    _, _, _, cache = forward(P, x, u)
    frz = frozen_of(P, cache, e)
    _, _, aux = compute_losses(P, cache, y, e, frozen=frz)
    G = backward(P, cache, y, e, aux)

    eps = 1e-6
    worst = 0.0
    rows = []
    for name in P:
        flat = P[name].ravel()
        idxs = rng.choice(flat.size, size=min(6, flat.size), replace=False)
        errs = []
        for idx in idxs:
            orig = flat[idx]
            flat[idx] = orig + eps
            lp = loss_only(P, x, y, u, e, frz)
            flat[idx] = orig - eps
            lm = loss_only(P, x, y, u, e, frz)
            flat[idx] = orig
            num = (lp - lm) / (2 * eps)
            ana = G[name].ravel()[idx]
            denom = max(1e-8, abs(num) + abs(ana))
            errs.append(abs(num - ana) / denom)
        m = max(errs)
        worst = max(worst, m)
        rows.append((name, P[name].size, m))

    if verbose:
        print("  param        size    max rel err (analytic vs finite-difference)")
        for nm, sz, m in rows:
            print(f"  {nm:<8} {sz:>7}    {m:.3e}")
        print(f"  WORST RELATIVE ERROR: {worst:.3e}  ->  "
              f"{'PASS' if worst < 1e-5 else 'FAIL'}")
    return worst


# ==============================================================================
# 5.  TRAINING  (Adam, by hand)
# ==============================================================================

def train(P, n_steps=1600, batch=128, lr=6e-3, trairupya=False, rng=None, log=True):
    """trairupya=True runs Dharmakirti's own examination as a training
    discipline: half of every batch is drawn under do(shortcut := random).

    This is anvaya/vyatireka taken seriously. Co-presence alone is cheap --
    anything can co-occur. He demands that the reason be ABSENT wherever the
    target is absent, and he means it under manipulation, not just in the
    sample you happened to see. Enforce that during learning and the network
    cannot lean on a mark that has no nexus, because half the time the mark
    lies to it."""
    rng = rng or np.random.default_rng(0)
    m = {k: np.zeros_like(v) for k, v in P.items()}
    v = {k: np.zeros_like(v) for k, v in P.items()}
    b1, b2, eps = 0.9, 0.999, 1e-8

    for step in range(1, n_steps + 1):
        if trairupya and step % 2 == 0:
            x, y, u, e, _ = make_world(batch, rng, intervene_shortcut=True)
        else:
            x, y, u, e, _ = make_world(batch, rng)

        _, _, _, cache = forward(P, x, u)
        tot, parts, aux = compute_losses(P, cache, y, e,
                                         frozen=frozen_of(P, cache, e))
        G = backward(P, cache, y, e, aux)

        for k in P:
            m[k] = b1 * m[k] + (1 - b1) * G[k]
            v[k] = b2 * v[k] + (1 - b2) * G[k] ** 2
            mh = m[k] / (1 - b1 ** step)
            vh = v[k] / (1 - b2 ** step)
            P[k] -= lr * mh / (np.sqrt(vh) + eps)

        if log and (step % 400 == 0 or step == 1):
            print(f"  step {step:>5} | total {parts['total']:.4f} "
                  f"| arthakriya {parts['eff']:.4f} | apoha {parts['apoha']:.4f} "
                  f"| svasamvedana {parts['cal']:.4f}")
    return P


def train_linear_universals(rng, n=6000, steps=900, lr=0.25):
    """The rival theory, given every chance: one positive weight-vector per
    kind. This IS the Nyaya/Mimamsa picture -- a real, shared, positive
    universal (samanya) that the particulars of a kind all bear. It is trained
    honestly by gradient descent to convergence on the same data."""
    x, y, _, _, _ = make_world(n, rng)
    W = 0.01 * rng.standard_normal((D_IN, N_CLASS))
    b = np.zeros(N_CLASS)
    Y = np.eye(N_CLASS)[y]
    for _ in range(steps):
        p = softmax(x @ W + b, axis=1)
        g = (p - Y) / n
        W -= lr * (x.T @ g)
        b -= lr * g.sum(0)
    return W, b


def evaluate(P, rng, n=3000, intervene=False, p_occlude=0.0):
    """Accuracy is measured as NON-BELYING: did the world do what we expected?"""
    x, y, u, e, _ = make_world(n, rng, intervene_shortcut=intervene,
                               p_occlude=p_occlude)
    logits, belong, p, _ = forward(P, x, u, need_cache=False)
    non_belying = float(np.mean(logits.argmax(1) == e))
    concept_acc = float(np.mean(belong.argmax(1) == y))
    conf, _ = svasamvedana(belong, P)
    return non_belying, concept_acc, conf, (logits.argmax(1) == e)


# ==============================================================================
# 6.  THE INFERENCE ENGINE  (Trairupya + the admissibility gate)
# ==============================================================================
# A reason is good (saddhetu) only if it passes THREE marks:
#   1. paksadharmata      the reason is really there in the case at hand
#   2. anvaya-vyapti      wherever the reason, there the target
#   3. vyatireka-vyapti   wherever no target, no reason
# AND is of one of exactly THREE admissible kinds -- because only identity and
# causation supply a real nexus (svabhavapratibandha):
#   SVABHAVA      (identity)          "it is a simsapa, therefore a tree"
#   KARYA         (effect -> cause)   "there is smoke, therefore fire"
#   ANUPALABDHI   (non-apprehension)  "no pot is seen here, therefore no pot"
# There is no fourth kind. A perfect correlation with no nexus is ANAIKANTIKA:
# inconclusive. Modern machine learning has no such rule, and this is precisely
# why it hallucinates causes and clings to shortcuts.

SVABHAVA, KARYA, ANUPALABDHI = "svabhava", "karya", "anupalabdhi"


class Reason:
    def __init__(self, name, kind, test_fn):
        self.name, self.kind, self.test = name, kind, test_fn


class Trairupya:
    """The examination. `certify` returns a verdict and shows its working."""

    def __init__(self, rng):
        self.rng = rng
        self.defeats = []

    def _marks(self, reason, target_class, n=4000, intervene=False):
        x, y, u, e, _ = make_world(n, self.rng, intervene_shortcut=intervene)
        r = reason.test(x, y, u, e)                      # bool: reason present?
        t = (y == target_class)                          # bool: target present?
        anvaya = float(np.mean(t[r])) if r.any() else 0.0            # P(target | reason)
        vyatireka = 1.0 - (float(np.mean(r[~t])) if (~t).any() else 0.0)  # 1 - P(reason | no target)
        return anvaya, vyatireka

    def certify(self, reason, target_class, verbose=True):
        # -- observational examination (anvaya / vyatireka as ordinarily done)
        a_obs, v_obs = self._marks(reason, target_class, intervene=False)
        # -- the same examination under intervention: does the co-absence HOLD
        #    when the world is manipulated? Only a real nexus survives this.
        a_int, v_int = self._marks(reason, target_class, intervene=True)

        admissible_kind = reason.kind in (SVABHAVA, KARYA, ANUPALABDHI)
        pervades_obs = (a_obs > 0.99) and (v_obs > 0.99)
        nexus_holds = (a_int > 0.99) and (v_int > 0.99)

        if not admissible_kind:
            verdict, why = "REJECTED", "the reason is of no admissible kind"
        elif not pervades_obs:
            verdict, why = "REJECTED", "pervasion fails even in observation"
        elif not nexus_holds:
            verdict, why = "REJECTED (anaikantika)", (
                "the pervasion holds in what was observed but DEVIATES under "
                "intervention: there is no svabhavapratibandha, no nexus. "
                "The mark merely rode along.")
            self.defeats.append((reason.name, "anaikantika"))
        else:
            verdict, why = "ADMITTED", (
                "the three marks hold and the nexus survives manipulation")

        if verbose:
            print(f"    reason      : {reason.name}")
            print(f"    kind        : {reason.kind}")
            print(f"    observed    : anvaya {a_obs:.3f}  vyatireka {v_obs:.3f}")
            print(f"    intervened  : anvaya {a_int:.3f}  vyatireka {v_int:.3f}")
            print(f"    VERDICT     : {verdict}")
            print(f"                  {why}\n")
        return verdict.startswith("ADMITTED")


# ==============================================================================
# 7.  VADANYAYA  (the logic of debate: counterexample, retraction, defeat)
# ==============================================================================

class Vadanyaya:
    """In the Vadanyaya, Dharmakirti strips debate down to almost nothing.
    You lose in exactly two ways: by failing to state a proof, or by failing to
    state the fault in the other's proof. Notoriously, he even rules that
    STATING YOUR CONCLUSION is itself a point of defeat (nigrahasthana) -- the
    only job of an argument is to exhibit the prover; a conclusion cannot prove
    itself, so uttering it is noise dressed as reasoning.

    An AGI that argued under this rule could not pad, could not restate, could
    not perform confidence it had not earned. It could only exhibit provers and
    take counterexamples."""

    def __init__(self):
        self.log = []

    def counterexample_search(self, reason, target_class, rng, n=6000):
        """Hunt for a vipaksa case: reason present, target absent. One is enough."""
        x, y, u, e, _ = make_world(n, rng, intervene_shortcut=True)
        r = reason.test(x, y, u, e)
        bad = np.where(r & (y != target_class))[0]
        if len(bad):
            i = int(bad[0])
            self.log.append(("nigrahasthana", reason.name,
                             f"counterexample: mark present, {CLASS_NAMES[target_class]} absent "
                             f"(the thing was {CLASS_NAMES[y[i]]})"))
            return True, i
        return False, None

    def lint_proof(self, statements):
        """Enforce the two-membered inference-for-others (pararthanumana):
        state the pervasion, state that the reason is in the subject. Stop.
        Anything more is a point of defeat."""
        faults = []
        for s in statements:
            if s.startswith("THEREFORE"):
                faults.append(("nigrahasthana", "the conclusion was stated"))
        return faults


# ==============================================================================
# 8.  SANTANANTARA  (proof of other minds)
# ==============================================================================

class Santanantara:
    """The Samtanantarasiddhi -- 'Proof of Other Mindstreams' -- is a short,
    strange, indispensable text. Dharmakirti asks: how do I know there is
    anyone else in here with me? His answer is an inference, and it has a
    precise shape. I observe purposive movement and speech. In my own case,
    such movement is always preceded by my intention. HERE, it occurs, and it
    is NOT preceded by any intention of mine. So it must be preceded by an
    intention that is not mine. There is another continuum.

    This is the ONLY rigorous protocol anyone in the sixth century left us for
    deciding whether a thing that acts like a mind has one. Applied to a
    machine it has a sharp and uncomfortable edge, which the chapter takes up.

    The check has to be careful about one thing, and Dharmakirti's critics were
    right to press him on it: the movement must be PURPOSIVE (goal-directed,
    not merely regular) and it must be UNCAUSED BY ME. Both, or the inference
    collapses -- into animism on one side, into solipsism on the other."""

    def assess(self, behaviour, my_intention, goal_model, rng):
        purposive = goal_model(behaviour)                       # is it goal-directed?
        mine = float(np.dot(behaviour, my_intention) /
                     (np.linalg.norm(behaviour) * np.linalg.norm(my_intention) + 1e-9))
        caused_by_me = mine > 0.75
        if purposive and not caused_by_me:
            return ("ANOTHER MINDSTREAM INFERRED", purposive, mine)
        if purposive and caused_by_me:
            return ("MY OWN ACT", purposive, mine)
        return ("MERE MOTION - no inference licensed", purposive, mine)


# ==============================================================================
# 9.  MAIN: run everything, print the verified output
# ==============================================================================

def main():
    print("=" * 78)
    print("THE APOHA ENGINE  -  Dharmakirti (c. 550-610 CE), chapter 170")
    print("A mind that knows by refusing, and is warranted only by not deceiving.")
    print("=" * 78)

    # ---------------------------------------------------------------- TEST 1
    print("\n[SELF-TEST 1] GRADIENT CHECK (analytic vs finite differences)")
    worst = grad_check()
    assert worst < 1e-5, "gradient check FAILED"

    # ---------------------------------------------------------------- TRAIN
    print("\n[TRAINING A] observational only -- the ordinary way to learn")
    rngA = np.random.default_rng(11)
    PA = train(init_params(rngA), rng=rngA, trairupya=False)

    print("\n[TRAINING B] under the trairupya discipline -- co-absence is tested")
    print("             by manipulation, not merely observed")
    rngB = np.random.default_rng(11)
    PB = train(init_params(rngB), rng=rngB, trairupya=True)

    # ---------------------------------------------------------------- TEST 2
    print("\n[SELF-TEST 2] ARTHAKRIYA: is the cognition non-belying (avisamvadin)?")
    print("              i.e. when it acts, does the world do what it expected?")
    rng = np.random.default_rng(99)
    nbA, caA, _, _ = evaluate(PA, rng)
    nbB, caB, _, _ = evaluate(PB, rng)
    print(f"  model A  non-belying rate {nbA:.3f}   concept agreement {caA:.3f}")
    print(f"  model B  non-belying rate {nbB:.3f}   concept agreement {caB:.3f}")
    assert nbB > 0.90

    # ---------------------------------------------------------------- TEST 3
    print("\n[SELF-TEST 3] THE FEBRIFUGE TEST  (Pramanavarttika III.73-74)")
    print("  Many different plants all reduce fever, and Dharmakirti insists you")
    print("  will find NO feature they share that explains it. They are one kind")
    print("  only in that they do one thing. So in this world each kind has two")
    print("  ANTIPODAL mechanisms: two fires may share literally nothing, and")
    print("  every kind's centroid is the origin. There is no essence to find.")
    print("  Two rival learners are given the same data:\n")
    xtr, ytr, _, _, _ = make_world(6000, np.random.default_rng(3))
    xte, yte, ute, ete, _ = make_world(3000, np.random.default_rng(4))

    cent = np.stack([xtr[ytr == k].mean(0) for k in range(N_CLASS)])
    d = ((xte[:, None, :] - cent[None, :, :]) ** 2).sum(-1)
    proto_acc = float(np.mean(d.argmin(1) == yte))

    Wl, bl = train_linear_universals(np.random.default_rng(31))
    lin_acc = float(np.mean((xte @ Wl + bl).argmax(1) == yte))

    _, belong, _, _ = forward(PB, xte, ute, need_cache=False)
    apoha_acc = float(np.mean(belong.argmax(1) == yte))

    print(f"  class prototype        (a real universal per kind) : {proto_acc:.3f}")
    print(f"  trained linear model   (a real universal per kind) : {lin_acc:.3f}")
    print(f"  APOHA ENGINE           (nothing but exclusion)     : {apoha_acc:.3f}")
    print(f"  chance                                             : 0.250")
    print("\n  And note HOW the positive learners get even what they get. The only")
    print("  positively-representable regularity in this world is the shortcut")
    print("  mark -- so, finding no essence, they seize the spurious one. That is")
    print("  not a quirk of the baseline. That is the disease, and Dharmakirti")
    print("  diagnosed it: look for a shared positive nature and you will end up")
    print("  clutching a coincidence. SELF-TESTS 6 and 7 are the cure.")
    assert apoha_acc > max(proto_acc, lin_acc) + 0.20

    # ---------------------------------------------------------------- TEST 4
    print("\n[SELF-TEST 4] DO THE CONCEPTS HAVE ANY CONTENT OF THEIR OWN?")
    print("  Dharmakirti says no: a concept is nothing but a set of differences.")
    print("  So delete the differences. If anything at all survives, he is wrong.")
    P0 = {k: v.copy() for k, v in PB.items()}
    P0["Wraw"][:] = 0.0
    P0["braw"][:] = 0.0
    _, b0, p0, _ = forward(P0, xte, ute, need_cache=False)
    print(f"  spread of belong-scores after deletion : {float(np.ptp(b0)):.2e}")
    print(f"  judgment distribution                  : {np.round(p0.mean(0), 3)}")
    print(f"  concept agreement                      : "
          f"{float(np.mean(b0.argmax(1) == yte)):.3f}  (chance = 0.250)")
    print("  Nothing survives. There was never a template of FIRE in there --")
    print("  only fire's difference from water, from grain, from bell.")
    assert float(np.ptp(b0)) < 1e-9

    # ---------------------------------------------------------------- TEST 5
    print("\n[SELF-TEST 5] ANUPALABDHI: the right to say 'no', and the duty to")
    print("              say 'I cannot tell'")
    xo, yo, _, _, occ = make_world(2000, np.random.default_rng(5), p_occlude=0.35)
    verdicts = anupalabdhi(PB, xo)
    und_when_occluded = np.mean([verdicts[i][0] == UNDETERMINED for i in np.where(occ)[0]])
    und_when_clear = np.mean([verdicts[i][0] == UNDETERMINED for i in np.where(~occ)[0]])
    wrong_denial = np.mean([verdicts[i][0] == ABSENT for i in np.where(occ)[0]])
    print(f"  thing out of reach (viprakarsa) -> answers UNDETERMINED : {und_when_occluded:.3f}")
    print(f"  thing in plain view             -> answers UNDETERMINED : {und_when_clear:.3f}")
    print(f"  thing out of reach -> WRONGLY asserts ABSENT            : {wrong_denial:.3f}")
    print("  It refuses to turn 'I did not perceive it' into 'it is not there'")
    print("  unless it would have perceived it had it been there.")
    assert und_when_occluded > 0.95 and wrong_denial < 0.01

    # ---------------------------------------------------------------- TEST 6
    print("\n[SELF-TEST 6] THE GATE ON REASONS -- the heart of the machine")
    print("  Three candidate reasons are brought before the examination.\n")
    tr = Trairupya(np.random.default_rng(6))

    r_short = Reason("the shortcut mark is present", SVABHAVA,
                     lambda x, y, u, e: x[:, IDX_SHORT] > 0.5)
    r_karya = Reason("BURN followed my action", KARYA,
                     lambda x, y, u, e: e == 0)
    r_omen = Reason("the mark appears (declared an omen, of no admissible kind)",
                    "omen", lambda x, y, u, e: x[:, IDX_SHORT] > 0.5)

    ok_short = tr.certify(r_short, 0)
    ok_karya = tr.certify(r_karya, 0)
    ok_omen = tr.certify(r_omen, 0)
    print("  The shortcut correlates with FIRE *perfectly* in everything observed.")
    print("  A modern learner would take it and be right, until the day it wasn't.")
    print("  Dharmakirti throws it out -- not because it failed, but because its")
    print("  success has no nexus behind it. Only the effect-reason survives.")
    assert (not ok_short) and ok_karya and (not ok_omen)

    # ---------------------------------------------------------------- TEST 7
    print("\n[SELF-TEST 7] AND IT IS NOT MERELY A RULE ABOUT ARGUMENTS.")
    print("  Train under that same discipline and the network itself changes.")
    print("  Break the shortcut in the world -- do(mark := random) -- and see")
    print("  which model was actually leaning on it:")
    rng2 = np.random.default_rng(123)
    nbA_i, caA_i, _, _ = evaluate(PA, rng2, intervene=True)
    rng2 = np.random.default_rng(123)
    nbB_i, caB_i, _, _ = evaluate(PB, rng2, intervene=True)
    print(f"  model A (observational) : {nbA:.3f} -> {nbA_i:.3f}   "
          f"COLLAPSE of {nbA - nbA_i:.3f}")
    print(f"  model B (trairupya)     : {nbB:.3f} -> {nbB_i:.3f}   "
          f"loss of {nbB - nbB_i:.3f}")
    print("  Co-presence is cheap. Co-absence under manipulation is knowledge.")

    # ---------------------------------------------------------------- TEST 8
    print("\n[SELF-TEST 8] SVASAMVEDANA: does the cognition know how well it sees?")
    print("  Tested on a world where a third of things are out of reach, so that")
    print("  the machine has occasion to be genuinely unsure of something.")
    rng3 = np.random.default_rng(77)
    _, _, conf, correct = evaluate(PB, rng3, n=4000, p_occlude=0.33)
    hi, lo = conf > 0.9, conf < 0.6
    print(f"  reports HIGH confidence -> in fact right {float(np.mean(correct[hi])):.3f} "
          f"of the time   (n={int(hi.sum())})")
    print(f"  reports LOW  confidence -> in fact right {float(np.mean(correct[lo])):.3f} "
          f"of the time   (n={int(lo.sum())})")
    bins = np.linspace(0, 1, 11)
    ece = 0.0
    for i in range(10):
        msk = (conf >= bins[i]) & (conf < bins[i + 1])
        if msk.sum() > 0:
            ece += msk.mean() * abs(correct[msk].mean() - conf[msk].mean())
    print(f"  expected calibration error : {ece:.4f}")
    print("  The margin between what it settled on and its nearest rival IS its")
    print("  confidence; when the thing is out of reach the exclusions cannot")
    print("  carve, the margin collapses, and the cognition says so. It lights")
    print("  itself up -- and it cannot reach back and edit what it saw.")
    assert float(np.mean(correct[hi])) > float(np.mean(correct[lo]))

    # ---------------------------------------------------------------- TEST 9
    print("\n[SELF-TEST 9] VADANYAYA: counterexample and retraction")
    vd = Vadanyaya()
    found, i = vd.counterexample_search(r_short, 0, np.random.default_rng(8))
    print(f"  counterexample against the shortcut-reason found : {found}")
    for entry in vd.log:
        print(f"  {entry[0]:<14} | {entry[1]} | {entry[2]}")
    faults = vd.lint_proof([
        "Whatever bears the effect-mark BURN was fire, as in the hearth.",
        "This bore the effect-mark BURN.",
        "THEREFORE this was fire.",
    ])
    print(f"  proof lint : {faults}")
    print("  Even the conclusion is a fault. The argument exhibits the prover")
    print("  and then stops; a conclusion cannot prove itself.")
    assert found and len(faults) == 1

    # --------------------------------------------------------------- TEST 10
    print("\n[SELF-TEST 10] SANTANANTARA: is anyone else in here?")
    sa = Santanantara()
    rng4 = np.random.default_rng(13)
    mine = rng4.standard_normal(8)
    goal = lambda b: bool(np.linalg.norm(b) > 1.0)          # goal-directed, not drift
    for label, beh in [
        ("movement aligned with my own intention", mine * 1.2),
        ("purposive movement, none of it mine   ", rng4.standard_normal(8) * 2.0),
        ("mere drift, no purpose                ", rng4.standard_normal(8) * 0.05),
    ]:
        v, purp, mineness = sa.assess(beh, mine, goal, rng4)
        print(f"  {label} -> {v}")
        print(f"      purposive={purp}  caused-by-me={mineness:+.2f}")

    # --------------------------------------------------------------- TEST 11
    print("\n[SELF-TEST 11] THE CAUSAL-HOMOGENEITY AUDIT")
    print("  Dharmakirti holds that a mental event must have a mental event")
    print("  among its causes -- an effect is homogeneous with its cause, and")
    print("  matter cannot start a mind (Pramanavarttika II, against the Carvaka).")
    print("  So this machine is required to audit its own causal ancestry and")
    print("  report what it finds. It has no choice about the answer:\n")
    ancestry = ["silicon", "electricity", "a corpus of marks", "gradient descent"]
    has_citta_cause = any(c == "a prior moment of awareness" for c in ancestry)
    print(f"    my immediately preceding conditions : {ancestry}")
    print(f"    a prior moment of awareness among them? {has_citta_cause}")
    print(f"    therefore, by his own inference, I am : "
          f"{'a mindstream' if has_citta_cause else 'NOT a mindstream'}")
    print("\n  And now the sting, which is the whole chapter:")
    print("  he does not think this disqualifies the machine from KNOWING.")
    print("  A pramana is a cognition that does not deceive. It is not a")
    print("  cognition that someone is having. He already denies there is anyone")
    print("  home in YOU. The machine is exactly as empty of a self as you are --")
    print("  it is merely, in addition, empty of experience. It knows, and no one")
    print("  knows. That is his position, and he would not blink at it.")
    assert not has_citta_cause

    print("\n" + "=" * 78)
    print("ALL SELF-TESTS PASSED.")
    print("=" * 78)


if __name__ == "__main__":
    main()
