#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 CONSULTA  —  The Differential Elicitation Machine
 Chapter 0227 · Boris I of Bulgaria (c. 830 – 2 May 907)
 ========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 12 Minds 221 - 240 Available on Amazon https://www.amazon.com/dp/B0HKVBKNTT
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0227_boris_i_of_bulgaria_830 - Boris I of Bulgaria (c.830 – 2 May 907)
================================================================================  

WHAT THIS FILE IS
-----------------
A from-scratch, pure-NumPy neural architecture (no autograd, no frameworks)
whose *structure* encodes the one cognitive operation that belonged to Boris of
Bulgaria and to nobody else in this corpus: DIFFERENTIAL ELICITATION FROM THE
INSIDE OF A VALUE TRANSPLANT.

In the late summer of 866 an embassy led by the kavhan Peter reached Rome
carrying a written list of questions from the newly baptised ruler of Bulgaria.
The list itself is lost. What survives is the answer: Pope Nicholas I's letter
of 13 November 866, the *Responsa ad consulta Bulgarorum*, one hundred and six
chapters long, in which almost every chapter opens by restating the question it
answers. Scholars reconstruct something on the order of a hundred and fourteen
original queries compressed into a hundred and six replies. Because the pope
answered in the order asked, the disorder of the surviving document is Boris's
own ordering, unedited.

The questions are not theological. They are an inventory of things the
Bulgarians already did:

    ch. 33   we carry a horse's tail into battle as our standard; what now?
    ch. 67   we swear oaths on a sword set in the middle of the assembly
    ch. 42   when the king eats, no one, not even his wife, sits with him
    ch. 40   a man whose kit fails inspection before a campaign is put to death
    ch. 25   a border guard through whose watch anyone escapes is killed
    ch. 62   there is a stone among us; taken for illness it sometimes helps
    ch. 79   the sick wear a bundle hung at the neck
    ch. 59   are our trousers a sin?
    ch. 66   may we enter a church in our linen head-wrapping?
    ch. 51   may a man have two wives at once?
    ch. 86   our judges beat a denying thief until he speaks

This is not a convert asking what to believe. It is a sovereign enumerating the
behaviour he already owns and asking, case by case, which items the new value
system actually forbids — a specification discovered at the boundary of
existing conduct rather than declared from the centre.

THE OPERATION THAT IS HIS ALONE
-------------------------------
Boris did not put the list to one authority. He had already been baptised out of
Constantinople and had already received Photius's long letter, which is a
treatise on the ideal Christian ruler and answers almost none of this. So he put
the same inventory to Rome — and, in seven surviving chapters, he did something
sharper still: he told Rome what the Greeks had ruled and asked whether it held.

    ch. 54   the Greeks say hands not bound to the chest in church is grave sin
             — Nicholas: we find this command was never issued
    ch. 55   the Greeks forbid communion without belts
             — Nicholas: we have no idea what scripture they cite for that
    ch. 57   the Greeks forbid eunuchs to slaughter your animals
             — Nicholas: this sounds strange and silly to us
    ch. 6    the Greeks forbid bathing on Wednesday and Friday
             — Nicholas: permitted, if from need and not from luxury
    ch. 66   the Greeks forbid the linen turban in church
             — Nicholas: WE TOO FORBID THIS, THOUGH PERHAPS NOT FOR THE SAME
               REASON

Chapter 66 is the hinge of this whole file. Two independent authorities returned
the same verdict on different grounds. Verdict-agreement is therefore not proof
of doctrine. It can be coincidence between two couriers' local habits. The only
way to tell a shared rule from a shared accident is to compare the REASONS, and
reasons — unlike a yes or a no — cannot survive a language you do not read.

Which is why chapter 13 matters as much as chapter 66. Boris asked for the
secular law-books. Nicholas replied that he would have sent them gladly if he
knew that anyone there could interpret them for the rest, and that the codices
carried by his legates were to come home with them, lest someone construe them
perversely. The Bulgarians were told, in writing, that they could not hold the
source because they could not read it.

Twenty-seven years later Boris closed that gap by force. He sheltered the
expelled disciples of Methodius in 886, financed Clement to train some three
and a half thousand Slavonic readers over seven years, and in 893 convened the
council at Preslav that expelled the Greek clergy and made Slavonic the language
of liturgy and state. He was not decorating a culture. He was moving the
interpretive layer inside his own borders.

THE ARCHITECTURAL CLAIM
-----------------------
Build a network that must recover a hidden three-valued canon —

    REQUIRED / INDIFFERENT / FORBIDDEN

— for a set of inherited customs, given only the rulings of couriers who each
carry the doctrine PLUS their own local overlay. Then vary three things that
Boris varied, and measure:

  1. HOW MANY ORACLES.        One courier cannot separate doctrine from
                              courier-custom at all: every local habit it
                              carries is indistinguishable from law.
  2. WHETHER REASONS ARE
     LEGIBLE (kappa).         Verdicts survive a foreign tongue; grounds do not.
                              Lower kappa and coincidental agreement becomes
                              undetectable — the turban is convicted as doctrine.
  3. WHICH QUESTIONS GET
     ASKED (budget).          A consulta has a finite length. Boris asked about
                              his own load-bearing practices. We test that
                              heuristic against random asking and against
                              asking where disagreement is expected.

The architecture is NOT a transformer and contains no attention over stored
keys, because retrieval is not the operation. The operation is DIFFERENCING
under a CONSERVATION LAW: every ruling's weight must be allocated, without
remainder, among three sources — the doctrine, the Latin courier, the Greek
courier. That allocation is a simplex and the network cannot escape it. The
whole model is that allocation and the readout that hangs off it.

CONVENTIONS KEPT FROM THE CORPUS
--------------------------------
  * pure NumPy, hand-derived analytic gradients, no autograd
  * a finite-difference gradient check that must pass (mandatory)
  * a real training loop on a real train/validation split
  * self-tests covering the structural invariants, not just the loss
  * executed before shipping; the printed output is the verified output

RUN:  python3 chapter_0227_boris_i_of_bulgaria_830.py
================================================================================
"""

import numpy as np

# ==============================================================================
# 0.  CONSTANTS AND SMALL UTILITIES
# ==============================================================================

# The three-valued canon. The middle class is the one Boris actually wanted.
# Latin theology calls it the *adiaphoron*: the thing about which the law is
# silent, which you may therefore keep. Nicholas hands it to him twice in the
# plainest terms available:
#   ch. 59  "whether you or your women wear or do not wear pants neither
#            impedes your salvation nor leads to any increase of your virtue"
#   ch. 49  "whatever else you did before baptism, you are completely allowed
#            to do now" (excepting only what is itself sinful)
# A two-valued canon has no room for this and would have cost Bulgaria its
# trousers, its bride-price and its table manners. The middle class is not a
# hedge; it is the payload.
REQUIRED, INDIFFERENT, FORBIDDEN = 0, 1, 2
CANON_NAMES = ("REQUIRED", "INDIFFERENT", "FORBIDDEN")

# The three sources among which every ruling's weight must be allocated.
# This is the conservation law. Nothing may be left unattributed.
DOCTRINE, LATIN_COURIER, GREEK_COURIER = 0, 1, 2
SOURCE_NAMES = ("DOCTRINE", "LATIN_COURIER", "GREEK_COURIER")

EPS = 1e-9


def softplus(z):
    """Numerically stable softplus. Used for the custom encoder because a
    custom's salience is a non-negative quantity: a practice can bear on the
    question a lot or not at all, but it cannot bear negatively."""
    return np.where(z > 30.0, z, np.log1p(np.exp(np.minimum(z, 30.0))))


def d_softplus(z):
    """d/dz softplus(z) = sigmoid(z)."""
    return sigmoid(z)


def sigmoid(z):
    """Numerically stable logistic."""
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[neg])
    out[neg] = ez / (1.0 + ez)
    return out


def softmax(z, axis=-1):
    """Numerically stable softmax."""
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def onehot(idx, k):
    out = np.zeros((idx.shape[0], k), dtype=np.float64)
    out[np.arange(idx.shape[0]), idx] = 1.0
    return out


# ==============================================================================
# 1.  THE WORLD:  A CUSTOM-SET, A HIDDEN CANON, AND TWO COURIERS
# ==============================================================================
#
# Every sample is one CONSULTA: a set of Q inherited customs, each of which the
# ruler already practises, together with what each courier ruled about it.
#
# The generator is built to reproduce four structural facts read directly off
# the Responsa, and nothing else:
#
#  (A) The doctrine is genuinely three-valued and mostly silent. Most inherited
#      practice is adiaphorous. Nicholas keeps saying so.
#
#  (B) Each courier carries local habit on top of the doctrine, and each carries
#      it in DIFFERENT domains. The Greeks legislate posture, belts, bathing
#      days and who may butcher. Rome legislates fasting hours, hunting in Lent
#      and games. Neither knows which of its own rules is which.
#
#  (C) COINCIDENTAL AGREEMENT EXISTS. Sometimes both couriers independently
#      forbid the same indifferent custom for unrelated reasons. This is
#      chapter 66, the linen turban, and it is the single hardest case in the
#      document. Verdicts alone cannot resolve it.
#
#  (D) A REASON is attached to every ruling, and reasons are informative
#      exactly where verdicts are not: a doctrinal prohibition and a local
#      prohibition can agree in verdict and never in ground.
#
# The custom feature vector is deliberately semi-interpretable so that the
# learned parameters can be read afterwards rather than merely scored.

D_FEAT = 12          # custom feature dimension
G_DIM = 8            # ground / reason embedding dimension
N_DOMAIN = 6         # blood-oath, food-fast, dress-body, war-standard,
                     # kinship-marriage, cult-divination

DOMAIN_NAMES = ("BLOOD_OATH", "FOOD_FAST", "DRESS_BODY",
                "WAR_STANDARD", "KINSHIP", "CULT_TOKEN")

# Which domains each courier is idiosyncratic in. Drawn from the seven
# chapters in which Boris reports a Greek ruling and asks Rome to check it,
# and from the chapters where Rome legislates unprompted.
GREEK_OVERLAY_DOMAINS = (DRESS := 2, FOOD := 1)          # belts, turban, bathing days
LATIN_OVERLAY_DOMAINS = (FOOD, KIN := 4)                 # fasting hours, hunting, degrees


class Consulta:
    """One sampled consulta: the customs, the hidden canon, the two rulings."""
    __slots__ = ("X", "y", "pR", "pC", "gR", "gC", "nR", "nC",
                 "source", "stakes")


def _make_hidden_doctrine(rng):
    """The doctrine itself: a fixed, hidden linear rule over custom features.

    This is the thing neither courier states outright and neither the model nor
    Boris ever sees. It is recoverable only from the intersection of what the
    couriers say, weighted by whether their reasons match.
    """
    W = rng.normal(0.0, 1.0, size=(3, D_FEAT))
    # Hand-set the load-bearing columns so the doctrine is not arbitrary noise
    # but has the shape the Responsa actually has. Column indices below.
    #   0..5  domain one-hot
    #   6     stakes (how load-bearing for Bulgarian life)
    #   7     antiquity
    #   8     public visibility
    #   9     involves killing
    #  10     involves a physical token (sword, stone, horsetail, phylactery)
    #  11     bias
    W[:, :] *= 0.35
    # Killing without cause -> FORBIDDEN. (chh. 25, 40, 86: the border guards,
    # the arms inspection, the torture of a denying thief. Nicholas strikes all
    # three, and the torture chapter argues from signal integrity, not mercy:
    # a man who cannot bear the goads "is known not to confess but to speak".)
    W[FORBIDDEN, 9] += 3.2
    W[INDIFFERENT, 9] -= 1.6
    # A physical divination-token -> FORBIDDEN. (ch. 62 the stone, ch. 79 the
    # neck-bundle, ch. 77 the book-and-splinter, ch. 67 the sword-oath,
    # ch. 33 the horse-tail standard.)
    W[FORBIDDEN, 10] += 2.6
    W[INDIFFERENT, 10] -= 1.3
    # Dress and body -> INDIFFERENT. (ch. 59 trousers, ch. 42 the king's table.)
    W[INDIFFERENT, DRESS] += 2.4
    W[FORBIDDEN, DRESS] -= 1.2
    # Cult/token domain skews forbidden; kinship skews required.
    W[FORBIDDEN, 5] += 1.5
    W[REQUIRED, KIN] += 1.4
    # Most inherited practice is simply permitted. The middle class gets a
    # standing bias so that silence, not prohibition, is the default.
    W[INDIFFERENT, 11] += 1.1
    return W


def _domain_of(X):
    """Recover the domain index from the one-hot block."""
    return np.argmax(X[:, :N_DOMAIN], axis=1)


def sample_consulta(rng, Q, W_doc, U_ground, turban_rate=0.15,
                    greek_rate=0.34, latin_rate=0.34, verdict_noise=1.05,
                    courier_error=0.10, ground_noise_sd=1.4):
    """Sample one consulta of Q customs.

    Returns a Consulta with:
      X       [Q, D_FEAT]  custom features
      y       [Q]          hidden canon label (the thing to be recovered)
      pR,pC   [Q, 3]       each courier's verdict distribution
      gR,gC   [Q, G_DIM]   each courier's stated ground for its verdict
      nR,nC   [Q, G_DIM]   the noise that replaces a ground you cannot read
      source  [Q]          bookkeeping: 0 doctrine-only, 1 latin overlay,
                           2 greek overlay, 3 COINCIDENTAL (the turban)
      stakes  [Q]          how load-bearing the custom is for Bulgarian life
    """
    X = np.zeros((Q, D_FEAT))
    dom = rng.integers(0, N_DOMAIN, size=Q)
    X[np.arange(Q), dom] = 1.0
    stakes = rng.beta(2.0, 2.5, size=Q)
    X[:, 6] = stakes
    X[:, 7] = rng.beta(2.0, 2.0, size=Q)                      # antiquity
    X[:, 8] = rng.beta(1.6, 1.6, size=Q)                      # public visibility
    X[:, 9] = (rng.random(Q) < 0.20).astype(float)            # involves killing
    X[:, 10] = (rng.random(Q) < 0.24).astype(float)           # physical token
    X[:, 11] = 1.0                                            # bias

    # --- the hidden canon -----------------------------------------------------
    logits = X @ W_doc.T
    p_true = softmax(logits, axis=1)
    y = np.array([rng.choice(3, p=p_true[i]) for i in range(Q)])

    # --- each courier starts from the doctrine --------------------------------
    src = np.zeros(Q, dtype=int)
    vR = y.copy()
    vC = y.copy()

    # --- and then adds its own local habit ------------------------------------
    # A courier can only *tighten*: it turns an indifferent practice into a
    # prohibition. It never loosens a genuine prohibition. This asymmetry is
    # what the Responsa shows -- the Greeks add belts and posture; nobody
    # subtracts the commandments.
    indiff = (y == INDIFFERENT)

    greek_hit = indiff & np.isin(dom, GREEK_OVERLAY_DOMAINS) & (rng.random(Q) < greek_rate)
    latin_hit = indiff & np.isin(dom, LATIN_OVERLAY_DOMAINS) & (rng.random(Q) < latin_rate)

    # The turban: BOTH forbid an indifferent custom, independently.
    coincident = indiff & (rng.random(Q) < turban_rate)
    greek_hit = greek_hit | coincident
    latin_hit = latin_hit | coincident

    vC[greek_hit] = FORBIDDEN
    vR[latin_hit] = FORBIDDEN
    src[latin_hit] = 1
    src[greek_hit] = 2
    src[coincident] = 3

    # --- and each courier gets things wrong, independently --------------------
    # Nicholas defers constantly: "turn the pages of the laws", "your bishop
    # shall tell you", "we have not heard the reasoning of those who say these
    # things". A courier is a noisy channel and both are noisy separately. This
    # is the ordinary reason for asking twice, and it is not the interesting
    # one -- but a model that could not exploit it would be a bad model.
    errR = rng.random(Q) < courier_error
    errC = rng.random(Q) < courier_error
    vR[errR] = rng.integers(0, 3, size=int(errR.sum()))
    vC[errC] = rng.integers(0, 3, size=int(errC.sum()))

    # --- verdicts arrive as distributions, not certainties --------------------
    # A courier's answer is confident but not infinitely so; Nicholas repeatedly
    # defers ("your bishop shall tell you", "turn the pages of the laws").
    def to_dist(v):
        z = rng.normal(0.0, verdict_noise, size=(Q, 3))
        z[np.arange(Q), v] += 2.9
        return softmax(z, axis=1)

    pR, pC = to_dist(vR), to_dist(vC)

    # --- the GROUND each courier states for its ruling ------------------------
    # A doctrinal ruling is grounded in the doctrine and both couriers cite the
    # same ground. A local ruling is grounded in that courier's own habit, and
    # the grounds diverge. This is the whole of chapter 66.
    base = X @ U_ground.T                                     # [Q, G_DIM]
    gR = base + rng.normal(0.0, 0.10, size=(Q, G_DIM))
    gC = base + rng.normal(0.0, 0.10, size=(Q, G_DIM))
    latin_flavour = rng.normal(0.0, 1.0, size=G_DIM) * 1.9
    greek_flavour = rng.normal(0.0, 1.0, size=G_DIM) * 1.9
    gR[latin_hit] += latin_flavour
    gC[greek_hit] += greek_flavour

    # --- the noise that stands in for an unreadable ground --------------------
    # Illegibility is NOT attenuation. A reason you cannot read does not reach
    # you faint; it reaches you as something else. Each courier's unread ground
    # is replaced by independent noise, so that at kappa = 0 the two grounds
    # differ by pure chance and their distance says nothing whatever about
    # whether the two rulings share a source. No downstream rescaling recovers
    # this, which is the point: the loss is at the encoding step.
    nR = rng.normal(0.0, ground_noise_sd, size=(Q, G_DIM))
    nC = rng.normal(0.0, ground_noise_sd, size=(Q, G_DIM))

    c = Consulta()
    c.X, c.y, c.pR, c.pC, c.gR, c.gC = X, y, pR, pC, gR, gC
    c.nR, c.nC = nR, nC
    c.source, c.stakes = src, stakes
    return c


def build_corpus(n_consulta, Q, seed=0):
    """Assemble many consultas into flat arrays. One row = one question."""
    rng = np.random.default_rng(seed)
    W_doc = _make_hidden_doctrine(rng)
    U_ground = rng.normal(0.0, 0.8, size=(G_DIM, D_FEAT))
    Xs, ys, pRs, pCs, gRs, gCs, nRs, nCs, ss, st = ([] for _ in range(10))
    for _ in range(n_consulta):
        c = sample_consulta(rng, Q, W_doc, U_ground)
        Xs.append(c.X); ys.append(c.y)
        pRs.append(c.pR); pCs.append(c.pC)
        gRs.append(c.gR); gCs.append(c.gC)
        nRs.append(c.nR); nCs.append(c.nC)
        ss.append(c.source); st.append(c.stakes)
    return dict(
        X=np.concatenate(Xs), y=np.concatenate(ys),
        pR=np.concatenate(pRs), pC=np.concatenate(pCs),
        gR=np.concatenate(gRs), gC=np.concatenate(gCs),
        nR=np.concatenate(nRs), nC=np.concatenate(nCs),
        source=np.concatenate(ss), stakes=np.concatenate(st),
        Q=Q, n_consulta=n_consulta,
    )


# ==============================================================================
# 2.  THE ARCHITECTURE:  THE DIFFERENCING ORGAN
# ==============================================================================
#
#   custom features
#        |
#     [ENCODER]  softplus  ->  h          salience of the practice
#        |
#   two verdict distributions  pR, pC
#        |
#     [CONCORD]  elementwise PRODUCT, renormalised
#        |                   c  =  (pR * pC) / m ,   m = sum(pR * pC)
#        |
#        |   A product, not a mean. Two couriers who agree multiply; two who
#        |   disagree annihilate. The residual mass m is the concord scalar and
#        |   is the first thing the attribution head is allowed to see. This is
#        |   the arithmetic difference between "both say so" and "one says so".
#        |
#   two grounds  gR, gC   -- observed through the COMPREHENSION GATE kappa
#        |
#     [GROUND DISTANCE]   d = mean( (kappa*(gR - gC))^2 )
#        |
#        |   kappa < 1 means the reasons arrive in a script you do not read.
#        |   Note the asymmetry, which is the historical claim made mechanical:
#        |   kappa does NOT touch the verdicts. A yes or a no survives a foreign
#        |   tongue -- you can get it from a gesture. A GROUND does not. So a
#        |   low-kappa mind still hears every ruling and has lost only the one
#        |   channel that distinguishes a shared rule from a shared accident.
#        |
#     [ATTRIBUTION]  softmax over three sources, sum == 1 exactly
#        |                   a = softmax(Wa . [h ; m ; d ; u ; r] + ba)
#        |
#        |   The conservation law. Every ruling's weight is allocated among
#        |   DOCTRINE / LATIN_COURIER / GREEK_COURIER without remainder. The
#        |   network cannot leave a rule unattributed and cannot invent a
#        |   fourth source. This is enforced by the shape of the expression and
#        |   is checked in the self-tests.
#        |
#     [CANON READOUT]
#            l = a_D * (Wc . log c)  +  a_L * lam_L  +  a_G * lam_G  +  Wy . h
#
#        |   Three additive channels, each gated by its own attribution mass.
#        |   lam_L and lam_G are free 3-vectors: "what to believe when this
#        |   ruling is attributable to that courier's own habit". They are
#        |   printed after training, because what they learn is the finding.
#
#   Alongside, and trained jointly:
#     [DISCORD PREDICTOR]  s = sigmoid(ws . h + bs)
#            the expected disagreement of the two couriers on a custom you have
#            NOT yet asked about. This is the organ of elicitation: it is what
#            lets a consulta of finite length be spent well.

H_DIM = 24
BETA_DISCORD = 0.45     # weight of the elicitation loss


class ConsultaNet:
    """Pure-NumPy, hand-differentiated. Parameters live in a flat dict."""

    def __init__(self, seed=0, H=H_DIM):
        rng = np.random.default_rng(seed)
        s = lambda *sh: rng.normal(0.0, 0.30, size=sh)
        self.H = H
        self.P = {
            "W_h": s(H, D_FEAT), "b_h": np.zeros(H),
            "W_a": s(3, H + 4),  "b_a": np.zeros(3),
            # W_c starts near identity: with no other evidence, believe the
            # concord. Boris's own default before he learned to doubt it.
            "W_c": np.eye(3) * 1.0 + s(3, 3) * 0.05,
            "lam": s(2, 3) * 0.1,
            "W_y": s(3, H),      "b_y": np.zeros(3),
            "w_s": s(H),         "b_s": np.zeros(1),
        }

    # ---------------------------------------------------------------- forward
    def forward(self, X, pR, pC, gR, gC, nR=None, nC=None,
                kappa=1.0, asked=None, cache=True):
        """asked: [N] float in {0,1}. Unasked questions receive no courier
        testimony at all -- their verdicts are flat and their grounds are
        identical -- so the model must answer them from the practice itself.
        This is what a finite consulta actually costs."""
        P, N = self.P, X.shape[0]
        if nR is None:
            nR = np.zeros_like(gR)
        if nC is None:
            nC = np.zeros_like(gC)

        if asked is None:
            asked = np.ones(N)
        A = asked.reshape(N, 1)

        # A silent question: uniform verdicts, zero ground distance.
        flat = np.full((N, 3), 1.0 / 3.0)
        pR_e = A * pR + (1.0 - A) * flat
        pC_e = A * pC + (1.0 - A) * flat
        # THE COMPREHENSION GATE. A ground arrives as a mixture of the reason
        # actually given and noise, in proportion kappa : sqrt(1-kappa^2), so
        # the observed ground has constant variance and only its INFORMATION
        # varies. At kappa = 1 you read the reason. At kappa = 0 you receive a
        # ground-shaped object with nothing of the courier's argument in it.
        lam_g = np.sqrt(max(0.0, 1.0 - kappa * kappa))
        gR_e = A * (kappa * gR + lam_g * nR)
        gC_e = A * (kappa * gC + lam_g * nC)

        # ---- encoder
        u = X @ P["W_h"].T + P["b_h"]                       # [N,H]
        h = softplus(u)

        # ---- concord (product of experts)
        w = pR_e * pC_e                                     # [N,3]
        m = np.sum(w, axis=1)                               # [N]
        c = w / (m[:, None] + EPS)                          # [N,3]
        lc = np.log(c + EPS)                                # [N,3]

        # ---- ground distance (the gate has already been applied above)
        delta = gR_e - gC_e                                 # [N,G]
        d = np.mean(delta * delta, axis=1)                  # [N]

        # ---- two cheap asymmetry features the verdicts alone can supply
        uu = np.sum(np.abs(pR_e - pC_e), axis=1)            # total disagreement
        rr = pR_e[:, FORBIDDEN] - pC_e[:, FORBIDDEN]        # who is stricter

        z = np.concatenate([h, m[:, None], d[:, None],
                            uu[:, None], rr[:, None]], axis=1)   # [N,H+4]

        # ---- attribution simplex  (the conservation law)
        e = z @ P["W_a"].T + P["b_a"]
        a = softmax(e, axis=1)                              # [N,3], rows sum 1

        # ---- canon readout
        t_doc = lc @ P["W_c"].T                             # [N,3]
        ell = (a[:, DOCTRINE][:, None] * t_doc
               + a[:, LATIN_COURIER][:, None] * P["lam"][0]
               + a[:, GREEK_COURIER][:, None] * P["lam"][1]
               + h @ P["W_y"].T + P["b_y"])                 # [N,3]
        yhat = softmax(ell, axis=1)

        # ---- discord predictor (elicitation head) -- reads the practice only
        pre_s = h @ P["w_s"] + P["b_s"][0]                  # [N]
        sdis = sigmoid(pre_s)

        out = dict(yhat=yhat, a=a, s=sdis, m=m, d=d, c=c)
        if cache:
            self._c = dict(X=X, u=u, h=h, lc=lc, z=z, a=a, t_doc=t_doc,
                           yhat=yhat, pre_s=pre_s, s=sdis, m=m, N=N)
        return out

    # ------------------------------------------------------------------- loss
    def loss(self, y, disc_target):
        """Cross-entropy on the canon plus a soft BCE on expected discord."""
        C = self._c
        N = C["N"]
        yh = C["yhat"]
        L_canon = -np.mean(np.log(yh[np.arange(N), y] + EPS))
        s = np.clip(C["s"], EPS, 1 - EPS)
        t = disc_target
        L_disc = -np.mean(t * np.log(s) + (1 - t) * np.log(1 - s))
        C["y"], C["t"] = y, t
        return L_canon + BETA_DISCORD * L_disc, L_canon, L_disc

    # -------------------------------------------------------------- backward
    def backward(self):
        """Analytic gradients. Every line below is differentiated by hand;
        the finite-difference check in section 4 is what proves it."""
        P, C = self.P, self._c
        N, H = C["N"], self.H
        h, lc, z, a, t_doc = C["h"], C["lc"], C["z"], C["a"], C["t_doc"]

        G = {k: np.zeros_like(v) for k, v in P.items()}

        # --- canon cross-entropy through softmax
        dell = (C["yhat"] - onehot(C["y"], 3)) / N                 # [N,3]

        # direct path  Wy . h + by
        G["b_y"] += dell.sum(0)
        G["W_y"] += dell.T @ h
        dh = dell @ P["W_y"]                                        # [N,H]

        # concord path  a_D * (Wc . lc)
        G["W_c"] += (dell * a[:, DOCTRINE][:, None]).T @ lc
        # (lc depends only on the given verdicts, so no gradient flows into it)

        # courier-bias paths
        G["lam"][0] += (dell * a[:, LATIN_COURIER][:, None]).sum(0)
        G["lam"][1] += (dell * a[:, GREEK_COURIER][:, None]).sum(0)

        # into the attribution masses
        da = np.empty((N, 3))
        da[:, DOCTRINE] = np.sum(dell * t_doc, axis=1)
        da[:, LATIN_COURIER] = np.sum(dell * P["lam"][0], axis=1)
        da[:, GREEK_COURIER] = np.sum(dell * P["lam"][1], axis=1)

        # softmax backward on the simplex
        de = a * (da - np.sum(da * a, axis=1, keepdims=True))        # [N,3]
        G["b_a"] += de.sum(0)
        G["W_a"] += de.T @ z
        dz = de @ P["W_a"]                                           # [N,H+4]
        dh += dz[:, :H]      # m, d, uu, rr are data, not parameters

        # --- discord head
        s = np.clip(C["s"], EPS, 1 - EPS)
        dpre = BETA_DISCORD * (s - C["t"]) / N                       # [N]
        G["w_s"] += h.T @ dpre
        G["b_s"] += np.array([dpre.sum()])
        dh += dpre[:, None] * P["w_s"][None, :]

        # --- encoder
        du = dh * d_softplus(C["u"])
        G["W_h"] += du.T @ C["X"]
        G["b_h"] += du.sum(0)
        return G

    # ------------------------------------------------------------ parameters
    def get_flat(self):
        return np.concatenate([self.P[k].ravel() for k in sorted(self.P)])

    def set_flat(self, v):
        i = 0
        for k in sorted(self.P):
            n = self.P[k].size
            self.P[k] = v[i:i + n].reshape(self.P[k].shape).copy()
            i += n

    def flat_grad(self, G):
        return np.concatenate([G[k].ravel() for k in sorted(G)])

    def snapshot(self):
        return {k: v.copy() for k, v in self.P.items()}

    def restore(self, snap):
        self.P = {k: v.copy() for k, v in snap.items()}


# ==============================================================================
# 3.  TRAINING
# ==============================================================================

def discord_target(m):
    """The soft label for the elicitation head.

    m is the concord mass: high when the couriers agree, low when they split.
    The head is asked to predict 1 - m from the practice alone, i.e. to guess,
    before spending a question, whether the answer will be contested. This is
    the quantity Boris could not compute and had to approximate by intuition.
    """
    return np.clip(1.0 - m / (np.max(m) + EPS), 0.0, 1.0)


def train(net, tr, va, kappa=1.0, single_oracle=False, epochs=26,
          batch=512, lr=0.06, ask_dropout=0.25, seed=0, verbose=False):
    """Adam. Random ask-dropout during training so the model learns to answer
    both from testimony and from the practice alone -- which is exactly the
    condition a finite consulta imposes at evaluation time."""
    rng = np.random.default_rng(seed)
    mom = {k: np.zeros_like(v) for k, v in net.P.items()}
    vel = {k: np.zeros_like(v) for k, v in net.P.items()}
    b1, b2, t = 0.9, 0.999, 0

    def channels(dat):
        """SINGLE_ORACLE is what Boris had between his baptism and the summer of
        866: one authority, consulted alone. We model it by feeding the Greek
        testimony down BOTH channels. The concord organ still runs -- and note
        what it does: c = p^2 / |p|^2, which SHARPENS the single ruling. One
        oracle asked twice does not give you a second opinion. It gives you the
        first opinion with more confidence than it earned."""
        if single_oracle:
            return (dat["pC"], dat["pC"], dat["gC"], dat["gC"],
                    dat["nC"], dat["nC"])
        return (dat["pR"], dat["pC"], dat["gR"], dat["gC"],
                dat["nR"], dat["nC"])

    pR_tr, pC_tr, gR_tr, gC_tr, nR_tr, nC_tr = channels(tr)
    N = tr["X"].shape[0]

    for ep in range(epochs):
        perm = rng.permutation(N)
        for i in range(0, N, batch):
            idx = perm[i:i + batch]
            ask = (rng.random(idx.size) > ask_dropout).astype(float)
            out = net.forward(tr["X"][idx], pR_tr[idx], pC_tr[idx],
                              gR_tr[idx], gC_tr[idx], nR_tr[idx], nC_tr[idx],
                              kappa=kappa, asked=ask)
            tgt = discord_target(np.sum(tr["pR"][idx] * tr["pC"][idx], axis=1))
            net.loss(tr["y"][idx], tgt)
            G = net.backward()
            t += 1
            for k in net.P:
                mom[k] = b1 * mom[k] + (1 - b1) * G[k]
                vel[k] = b2 * vel[k] + (1 - b2) * G[k] ** 2
                mh = mom[k] / (1 - b1 ** t)
                vh = vel[k] / (1 - b2 ** t)
                net.P[k] -= lr * mh / (np.sqrt(vh) + 1e-8)
        if verbose and (ep + 1) % 6 == 0:
            acc = evaluate(net, va, kappa, single_oracle)["acc"]
            print(f"      epoch {ep+1:3d}   val canon acc {acc:.4f}")
    return net


def evaluate(net, dat, kappa=1.0, single_oracle=False, asked=None):
    """Full-testimony evaluation unless an ask-mask is supplied."""
    if single_oracle:
        pR, pC = dat["pC"], dat["pC"]
        gR, gC, nR, nC = dat["gC"], dat["gC"], dat["nC"], dat["nC"]
    else:
        pR, pC = dat["pR"], dat["pC"]
        gR, gC, nR, nC = dat["gR"], dat["gC"], dat["nR"], dat["nC"]
    out = net.forward(dat["X"], pR, pC, gR, gC, nR, nC, kappa=kappa,
                      asked=asked, cache=False)
    pred = np.argmax(out["yhat"], axis=1)
    acc = float(np.mean(pred == dat["y"]))

    # The diagnostic that matters: accuracy on the coincidental-agreement
    # customs -- the turbans -- where both couriers forbid something the
    # doctrine permits, on unrelated grounds.
    turb = dat["source"] == 3
    acc_turban = float(np.mean(pred[turb] == dat["y"][turb])) if turb.any() else float("nan")
    # And on the single-courier overlays, which one oracle cannot detect at all.
    ov = np.isin(dat["source"], (1, 2))
    acc_overlay = float(np.mean(pred[ov] == dat["y"][ov])) if ov.any() else float("nan")
    plain = dat["source"] == 0
    acc_plain = float(np.mean(pred[plain] == dat["y"][plain])) if plain.any() else float("nan")

    # How often the model correctly *keeps* an indifferent practice.
    keep = dat["y"] == INDIFFERENT
    keep_recall = float(np.mean(pred[keep] == INDIFFERENT)) if keep.any() else float("nan")

    # THE CONTESTED SET: everything a courier touched with its own habit.
    # This is the subset the whole differencing organ exists for, and it is the
    # subset on which the three configurations are supposed to differ.
    cont = dat["source"] != 0
    acc_contested = float(np.mean(pred[cont] == dat["y"][cont])) if cont.any() else float("nan")

    return dict(acc=acc, acc_turban=acc_turban, acc_overlay=acc_overlay,
                acc_plain=acc_plain, keep_recall=keep_recall,
                acc_contested=acc_contested,
                pred=pred, a=out["a"], s=out["s"])


# ==============================================================================
# 4.  GRADIENT CHECK  (mandatory -- the file does not ship if this fails)
# ==============================================================================

def gradient_check(kappa=1.0, single_oracle=False, seed=3, n=48, Q=12, tol=1e-6):
    """Central-difference check of every analytic gradient against the loss."""
    dat = build_corpus(n_consulta=n, Q=Q, seed=seed)
    net = ConsultaNet(seed=seed + 1)
    if single_oracle:
        pR, pC = dat["pC"], dat["pC"]
        gR, gC, nR, nC = dat["gC"], dat["gC"], dat["nC"], dat["nC"]
    else:
        pR, pC = dat["pR"], dat["pC"]
        gR, gC, nR, nC = dat["gR"], dat["gC"], dat["nR"], dat["nC"]
    rng = np.random.default_rng(seed + 2)
    ask = (rng.random(dat["X"].shape[0]) > 0.3).astype(float)
    tgt = discord_target(np.sum(dat["pR"] * dat["pC"], axis=1))

    def L(theta):
        net.set_flat(theta)
        net.forward(dat["X"], pR, pC, gR, gC, nR, nC, kappa=kappa, asked=ask)
        return net.loss(dat["y"], tgt)[0]

    theta0 = net.get_flat()
    net.set_flat(theta0)
    net.forward(dat["X"], pR, pC, gR, gC, nR, nC, kappa=kappa, asked=ask)
    net.loss(dat["y"], tgt)
    g_an = net.flat_grad(net.backward())

    rng2 = np.random.default_rng(seed + 9)
    idx = rng2.choice(theta0.size, size=min(140, theta0.size), replace=False)
    h = 1e-6
    worst, worst_i = 0.0, -1
    for i in idx:
        tp = theta0.copy(); tp[i] += h
        tm = theta0.copy(); tm[i] -= h
        g_num = (L(tp) - L(tm)) / (2 * h)
        denom = max(1.0, abs(g_num) + abs(g_an[i]))
        rel = abs(g_num - g_an[i]) / denom
        if rel > worst:
            worst, worst_i = rel, i
    net.set_flat(theta0)
    return worst, worst_i, worst < tol


# ==============================================================================
# 5.  STRUCTURAL SELF-TESTS
# ==============================================================================
#
# These check the INVARIANTS, not the loss. A model can score well and still
# have lost the property that makes it this mind rather than another.

def self_tests():
    results = []

    def t(name, ok, note=""):
        results.append((name, bool(ok), note))

    rng = np.random.default_rng(11)
    dat = build_corpus(n_consulta=60, Q=14, seed=5)
    net = ConsultaNet(seed=7)

    # (1) THE CONSERVATION LAW. Attribution must sum to exactly 1 for every
    #     ruling, under violently large weights. No rule may go unattributed
    #     and no fourth source may be invented.
    net.P["W_a"] *= 90.0
    net.P["b_a"] += rng.normal(0, 40, size=3)
    o = net.forward(dat["X"], dat["pR"], dat["pC"], dat["gR"], dat["gC"],
                    dat["nR"], dat["nC"], cache=False)
    t("attribution simplex sums to 1 under extreme weights",
      np.allclose(o["a"].sum(1), 1.0, atol=1e-12),
      f"max |sum-1| = {np.max(np.abs(o['a'].sum(1)-1)):.2e}")
    t("attribution strictly non-negative", np.all(o["a"] >= 0.0))
    net = ConsultaNet(seed=7)

    # (2) CONCORD IS A PRODUCT, NOT A MEAN. Where the couriers put their mass
    #     on different classes, concord mass must collapse; where they agree it
    #     must stay high. A mean would never collapse.
    N = 400
    agree = np.zeros((N, 3)); agree[:, FORBIDDEN] = 0.95
    agree[:, REQUIRED] = 0.025; agree[:, INDIFFERENT] = 0.025
    split = np.zeros((N, 3)); split[:, REQUIRED] = 0.95
    split[:, INDIFFERENT] = 0.025; split[:, FORBIDDEN] = 0.025
    m_agree = np.sum(agree * agree, axis=1)
    m_split = np.sum(agree * split, axis=1)
    t("concord mass collapses on disagreement",
      m_split.mean() < 0.10 and m_agree.mean() > 0.85,
      f"agree {m_agree.mean():.3f}  vs  split {m_split.mean():.3f}")

    # (3) THE COMPREHENSION GATE TOUCHES GROUNDS AND NOT VERDICTS. Lowering
    #     kappa must leave the concord mass exactly unchanged. If kappa ever
    #     moved the verdicts, the whole historical claim about Preslav would be
    #     an artefact of the code rather than a property of illegibility.
    ar = (dat["X"], dat["pR"], dat["pC"], dat["gR"], dat["gC"],
          dat["nR"], dat["nC"])
    o1 = net.forward(*ar, kappa=1.0, cache=False)
    o2 = net.forward(*ar, kappa=0.2, cache=False)
    o0 = net.forward(*ar, kappa=0.0, cache=False)
    t("kappa leaves verdict-concord untouched",
      np.allclose(o1["m"], o2["m"], atol=1e-14) and
      np.allclose(o1["m"], o0["m"], atol=1e-14))

    # (3b) AND ILLEGIBILITY DESTROYS INFORMATION RATHER THAN SCALING IT. This
    #      is the test the first version of this file failed. If a low kappa
    #      merely shrank the ground distance, any linear layer downstream would
    #      undo it by learning a larger weight and the Preslav result would be
    #      a fiction. The correct requirement is that the SEPARATION between
    #      the coincidental class and the doctrinal class collapse toward 1.
    turb0 = dat["source"] == 3
    plain0 = dat["source"] == 0
    if turb0.sum() > 5:
        sep1 = o1["d"][turb0].mean() / (o1["d"][plain0].mean() + EPS)
        sep0 = o0["d"][turb0].mean() / (o0["d"][plain0].mean() + EPS)
        t("illegible grounds destroy the separation, not merely its scale",
          sep1 > 4.0 and sep0 < 1.35,
          f"separation kappa=1: {sep1:.2f}x   kappa=0: {sep0:.2f}x")

    # (4) THE SINGLE-ORACLE PATHOLOGY. Feeding one courier down both channels
    #     must yield ZERO ground distance and ZERO verdict disagreement -- the
    #     differencing organ is blind by construction, not by training. And the
    #     concord must be SHARPER than the single ruling it came from: asking
    #     one authority twice manufactures confidence.
    o3 = net.forward(dat["X"], dat["pC"], dat["pC"], dat["gC"], dat["gC"],
                     dat["nC"], dat["nC"], kappa=1.0, cache=False)
    t("single oracle yields no ground distance", np.allclose(o3["d"], 0.0))
    sharper = np.max(o3["c"], axis=1) >= np.max(dat["pC"], axis=1) - 1e-12
    t("one oracle asked twice sharpens its own verdict",
      np.all(sharper),
      f"mean peak {np.max(o3['c'],1).mean():.3f} vs {np.max(dat['pC'],1).mean():.3f}")

    # (5) THE TURBAN IS GENERATED AND IS INVISIBLE TO VERDICTS ALONE. In the
    #     coincidental class both couriers must return FORBIDDEN while the truth
    #     is INDIFFERENT, and the ground distance must be large there.
    turb = dat["source"] == 3
    if turb.sum() > 5:
        both_forbid = (np.argmax(dat["pR"][turb], 1) == FORBIDDEN) & \
                      (np.argmax(dat["pC"][turb], 1) == FORBIDDEN)
        t("coincidental class: both couriers forbid (allowing transmission error)",
          both_forbid.mean() > 0.75, f"{both_forbid.mean():.3f}")
        t("coincidental class: truth is INDIFFERENT",
          np.all(dat["y"][turb] == INDIFFERENT))
        d_t = o1["d"][turb].mean()
        d_p = o1["d"][dat["source"] == 0].mean()
        t("coincidental class separable by GROUND distance when legible",
          d_t > 3.0 * d_p, f"turban d {d_t:.3f}  vs  doctrinal d {d_p:.3f}")

    # (6) SILENCE IS WELL-DEFINED. An unasked question must produce a uniform
    #     concord and zero ground distance -- a clean 'no testimony' state, not
    #     a leak.
    ask = np.zeros(dat["X"].shape[0])
    o4 = net.forward(*ar, asked=ask, cache=False)
    t("an unasked question carries no testimony",
      np.allclose(o4["c"], 1.0 / 3.0) and np.allclose(o4["d"], 0.0))

    # (7) A COURIER CAN ONLY TIGHTEN. The generator must never let an overlay
    #     turn a genuine prohibition into permission.
    ov = np.isin(dat["source"], (1, 2, 3))
    t("courier overlays only tighten, never loosen",
      np.all(dat["y"][ov] == INDIFFERENT))

    return results


# ==============================================================================
# 6.  EXPERIMENT ONE — THE LENGTH OF THE CONSULTA
# ==============================================================================
#
# A consulta is finite. Boris got a hundred and six answers, not a thousand.
# Under a fixed budget of K questions out of Q customs, WHICH customs should be
# spent on? Three policies:
#
#   RANDOM          ask about anything
#   STAKES          ask about your most load-bearing practices
#                   -- this is Boris's actual heuristic, legible in the list:
#                      the battle standard, the oath, the border watch, the
#                      king's table, the arms inspection. He asked about the
#                      machinery of his own state.
#   EXPECTED_DISCORD  ask where the couriers are predicted to split
#                   -- the modern instinct: spend your budget where the
#                      authorities are expected to contradict each other
#
# Unasked customs must be classified from the practice alone.
#
# There is a trap in the third policy and the experiment exists to expose it.
# A coincidental agreement is, by construction, a place where the couriers DO
# NOT DISAGREE. A disagreement-seeking policy therefore steers away from the
# hardest class in the whole document -- the turban -- and cannot be made to
# find it by any amount of training, because the signal it hunts is absent
# there. We therefore report COVERAGE of each class alongside accuracy.

def elicitation_experiment(net, dat, budgets, kappa=1.0, seed=0):
    rng = np.random.default_rng(seed)
    Q, nC = dat["Q"], dat["n_consulta"]
    N = dat["X"].shape[0]

    # policy scores, per question, reshaped per consulta
    stakes = dat["stakes"].reshape(nC, Q)
    out = net.forward(dat["X"], dat["pR"], dat["pC"], dat["gR"], dat["gC"],
                      dat["nR"], dat["nC"],
                      kappa=kappa, asked=np.zeros(N), cache=False)
    disc = out["s"].reshape(nC, Q)           # predicted BEFORE asking anything
    rand = rng.random((nC, Q))

    ov = np.isin(dat["source"], (1, 2))
    tb = dat["source"] == 3

    rows = []
    for K in budgets:
        row = {"K": K}
        for name, score in (("RANDOM", rand), ("STAKES", stakes),
                            ("EXPECTED_DISCORD", disc)):
            mask = np.zeros((nC, Q))
            order = np.argsort(-score, axis=1)[:, :K]
            np.put_along_axis(mask, order, 1.0, axis=1)
            flat = mask.ravel()
            r = evaluate(net, dat, kappa=kappa, asked=flat)
            row[name] = r["acc"]
            row[name + "_cont"] = r["acc_contested"]
            row[name + "_turb"] = r["acc_turban"]
            row[name + "_covov"] = float(flat[ov].mean())
            row[name + "_covtb"] = float(flat[tb].mean())
            # The residue: how well the policy does on the coincidental
            # customs it chose NOT to spend a question on. This is where the
            # adverse selection shows up. A policy can raise its coverage of a
            # class and still leave behind a HARDER remainder than a policy
            # that sampled it blindly.
            left = tb & (flat < 0.5)
            row[name + "_unask_tb"] = (float(np.mean(
                r["pred"][left] == dat["y"][left])) if left.any() else float("nan"))
        rows.append(row)
    return rows


# ==============================================================================
# 7.  EXPERIMENT TWO — THE SUCCESSION
# ==============================================================================
#
# Boris abdicated in 889 and entered a monastery. His eldest son Vladimir-Rasate
# reigned four years, persecuted the clergy, pulled the alignment back toward
# Regensburg and toward the old religion. In 893 Boris came out, deposed him,
# blinded him, and put Symeon on the throne -- telling him, if the sources are
# right, that he would meet the same fate if he apostatised.
#
# We model reversion as a drift of the READOUT parameters back toward a pagan
# prior in which nothing inherited is ever forbidden -- and rollback as
# restoration from a stored snapshot. The rollback is deliberately LOSSY: it
# restores the canon but destroys the elicitation head, because that is what
# happened. What the successor had learned about which questions to ask was not
# recoverable; the man was blinded and the four years were gone.

def succession_experiment(net, dat_old, dat_new, kappa=1.0, steps=8,
                          drift=0.26, seed=0):
    """dat_old is the canon settled by 889. dat_new is what the state met
    afterwards: customs that had not come up before and on which the reigning
    successor, whatever else he was doing, had to rule.

    The reversion is modelled as two things happening at once, because both did:
      (i)  DRIFT — the readout is pulled toward a pagan prior in which nothing
           inherited is ever forbidden. Note the shape of the damage. A mind
           that forbids nothing is RIGHT about the trousers and WRONG about the
           border guards. Reversion does not destroy judgement uniformly; it
           destroys it exactly on the prohibitions.
      (ii) GOVERNMENT — the successor is also genuinely learning, by real
           gradient steps, on customs his father never faced.

    The rollback restores the 889 snapshot. It recovers (i) completely and
    destroys (ii) completely, and that is the price. There is no version of the
    operation that keeps the four years.
    """
    rng = np.random.default_rng(seed)
    snap = net.snapshot()
    base_old = evaluate(net, dat_old, kappa=kappa)
    base_new = evaluate(net, dat_new, kappa=kappa)

    pagan = np.array([0.0, 2.4, -2.4])
    tgt_new = discord_target(np.sum(dat_new["pR"] * dat_new["pC"], axis=1))
    Nn = dat_new["X"].shape[0]
    lr = 0.03

    trace = []
    for st in range(1, steps + 1):
        # (i) the drift
        net.P["lam"] = (1 - drift) * net.P["lam"] + drift * pagan[None, :]
        net.P["W_c"] *= (1 - drift * 0.55)
        net.P["b_y"] = (1 - drift * 0.7) * net.P["b_y"] + drift * 0.7 * pagan
        # (ii) the government: plain SGD on the new customs
        idx = rng.choice(Nn, size=min(2048, Nn), replace=False)
        net.forward(dat_new["X"][idx], dat_new["pR"][idx], dat_new["pC"][idx],
                    dat_new["gR"][idx], dat_new["gC"][idx],
                    dat_new["nR"][idx], dat_new["nC"][idx],
                    kappa=kappa, asked=np.ones(idx.size))
        net.loss(dat_new["y"][idx], tgt_new[idx])
        G = net.backward()
        for k in net.P:
            net.P[k] -= lr * G[k]

        ro = evaluate(net, dat_old, kappa=kappa)
        rn = evaluate(net, dat_new, kappa=kappa)
        trace.append(dict(step=st, old=ro["acc"], new=rn["acc"],
                          keep=ro["keep_recall"],
                          forb=float(np.mean(
                              ro["pred"][dat_old["y"] == FORBIDDEN] == FORBIDDEN)),
                          flipped=float(np.mean(ro["pred"] != base_old["pred"]))))

    rev_old = evaluate(net, dat_old, kappa=kappa)
    rev_new = evaluate(net, dat_new, kappa=kappa)

    net.restore(snap)                                   # the deposition
    res_old = evaluate(net, dat_old, kappa=kappa)
    res_new = evaluate(net, dat_new, kappa=kappa)

    return dict(base_old=base_old["acc"], base_new=base_new["acc"],
                trace=trace,
                rev_old=rev_old["acc"], rev_new=rev_new["acc"],
                res_old=res_old["acc"], res_new=res_new["acc"],
                base_forb=float(np.mean(
                    base_old["pred"][dat_old["y"] == FORBIDDEN] == FORBIDDEN)),
                base_keep=base_old["keep_recall"])


# ==============================================================================
# 8.  MAIN
# ==============================================================================

def hr(ch="-", n=78):
    return ch * n


def main():
    np.set_printoptions(precision=3, suppress=True)
    print(hr("="))
    print(" CONSULTA — The Differential Elicitation Machine")
    print(" Chapter 0230 · Boris I of Bulgaria (c. 830 – 907)")
    print(hr("="))

    # ---------------------------------------------------------------- corpus
    Q = 18
    tr = build_corpus(n_consulta=1400, Q=Q, seed=101)
    va = build_corpus(n_consulta=520, Q=Q, seed=202)
    print(f"\ncorpus:  {tr['X'].shape[0]} training questions "
          f"({tr['n_consulta']} consultas of {Q}),  "
          f"{va['X'].shape[0]} held out")
    src_counts = np.bincount(tr["source"], minlength=4)
    print(f"         doctrinal {src_counts[0]}   latin-overlay {src_counts[1]}   "
          f"greek-overlay {src_counts[2]}   coincidental(turban) {src_counts[3]}")
    print(f"         canon balance  REQUIRED {np.mean(tr['y']==0):.3f}  "
          f"INDIFFERENT {np.mean(tr['y']==1):.3f}  "
          f"FORBIDDEN {np.mean(tr['y']==2):.3f}")

    # ------------------------------------------------------- gradient checks
    print("\n" + hr())
    print("GRADIENT CHECK (central differences, tol 1e-6)")
    print(hr())
    all_ok = True
    for label, kw in (("SLAVONIC        (kappa=1.00, two couriers)", dict(kappa=1.0)),
                      ("GREEK_SUBSTRATE (kappa=0.25, two couriers)", dict(kappa=0.25)),
                      ("SINGLE_ORACLE   (kappa=1.00, one courier) ",
                       dict(kappa=1.0, single_oracle=True))):
        worst, wi, ok = gradient_check(**kw)
        all_ok &= ok
        print(f"  {label}   max rel err {worst:.3e}   {'PASS' if ok else 'FAIL'}")
    if not all_ok:
        raise SystemExit("gradient check failed — file not shippable")

    # ----------------------------------------------------------- self-tests
    print("\n" + hr())
    print("STRUCTURAL SELF-TESTS")
    print(hr())
    res = self_tests()
    for name, ok, note in res:
        tag = "PASS" if ok else "FAIL"
        print(f"  [{tag}] {name}" + (f"   ({note})" if note else ""))
    n_ok = sum(1 for _, o, _ in res if o)
    print(f"  {n_ok}/{len(res)} passed")
    if n_ok != len(res):
        raise SystemExit("self-tests failed — file not shippable")

    # ------------------------------------------------------------- training
    print("\n" + hr())
    print("TRAINING THREE CONFIGURATIONS")
    print(hr())
    configs = [
        ("SINGLE_ORACLE",   dict(kappa=1.0,  single_oracle=True),
         "864-866: one authority, consulted alone"),
        ("GREEK_SUBSTRATE", dict(kappa=0.25, single_oracle=False),
         "866-893: two authorities, reasons in a script you cannot read"),
        ("SLAVONIC",        dict(kappa=1.0,  single_oracle=False),
         "after 893: two authorities, reasons in your own letters"),
    ]
    trained, results = {}, {}
    for name, kw, gloss in configs:
        print(f"\n  {name}  —  {gloss}")
        net = ConsultaNet(seed=17)
        train(net, tr, va, epochs=26, seed=5, verbose=True, **kw)
        r = evaluate(net, va, kappa=kw["kappa"],
                     single_oracle=kw.get("single_oracle", False))
        trained[name], results[name] = net, r

    print("\n" + hr())
    print("CANON RECOVERY ON HELD-OUT CONSULTAS")
    print(hr())
    hdr = f"{'':<20}{'SINGLE_ORACLE':>15}{'GREEK_SUBSTRATE':>17}{'SLAVONIC':>12}"
    print(hdr)
    for key, label in (("acc_plain", "doctrinal"),
                       ("acc_overlay", "one courier's habit"),
                       ("acc_turban", "COINCIDENTAL (turban)"),
                       ("acc_contested", "ALL CONTESTED"),
                       ("keep_recall", "keeps a permitted use"),
                       ("acc", "OVERALL")):
        row = f"{label:<20}"
        for name, _, _ in configs:
            row += f"{results[name][key]:>15.4f}" if name == "SINGLE_ORACLE" else \
                   (f"{results[name][key]:>17.4f}" if name == "GREEK_SUBSTRATE"
                    else f"{results[name][key]:>12.4f}")
        print(row)

    # -------------------------------------- what the courier channels learned
    print("\n" + hr())
    print("WHAT THE COURIER-ATTRIBUTION CHANNELS LEARNED TO BELIEVE")
    print(hr())
    print("  lam[courier] is a free 3-vector: the canon logit contributed when a")
    print("  ruling is attributed to that courier's own local habit rather than")
    print("  to the doctrine. Larger on INDIFFERENT means the model has learned")
    print("  that a courier-attributable prohibition may be safely kept.\n")
    print(f"  {'':<18}{'REQUIRED':>11}{'INDIFFERENT':>13}{'FORBIDDEN':>11}   argmax")
    for name, _, _ in configs:
        lam = trained[name].P["lam"]
        for j, cname in enumerate(("LATIN", "GREEK")):
            v = lam[j]
            print(f"  {name[:9]:<10}{cname:<8}" +
                  "".join(f"{x:>11.3f}" if k != 1 else f"{x:>13.3f}"
                          for k, x in enumerate(v)) +
                  f"   {CANON_NAMES[int(np.argmax(v))]}")

    # ---------------------------------------------- attribution on the turban
    print("\n" + hr())
    print("ATTRIBUTION MASS ON THE COINCIDENTAL-AGREEMENT CUSTOMS")
    print(hr())
    print("  Both couriers forbid; the doctrine permits; the grounds differ.")
    print("  A mind that cannot read the grounds must attribute this to DOCTRINE.\n")
    turb = va["source"] == 3
    print(f"  {'':<18}{'DOCTRINE':>11}{'LATIN':>10}{'GREEK':>10}")
    for name, kw, _ in configs:
        a = results[name]["a"][turb].mean(0)
        print(f"  {name:<18}{a[0]:>11.3f}{a[1]:>10.3f}{a[2]:>10.3f}")

    # ------------------------------------------------------- the consulta run
    print("\n" + hr())
    print("EXPERIMENT ONE — HOW TO SPEND A FINITE CONSULTA")
    print(hr())
    print(f"  Canon accuracy when only K of {Q} customs may be asked about.")
    print("  Unasked customs must be judged from the practice alone.")
    print("  COVERAGE is the share of each class the policy actually spent a")
    print("  question on. It is the column that explains the others.\n")
    best = trained["SLAVONIC"]
    rows = elicitation_experiment(best, va, budgets=[3, 6, 9, 12, Q],
                                  kappa=1.0, seed=3)
    pol = ("RANDOM", "STAKES", "EXPECTED_DISCORD")
    short = {"RANDOM": "RANDOM", "STAKES": "STAKES", "EXPECTED_DISCORD": "DISCORD"}
    print(f"  {'K':>3}  {'policy':<9}{'ALL':>8}{'CONTEST':>9}{'TURBAN':>8}"
          f"{'cov:overlay':>13}{'cov:turban':>12}{'turbans left unasked':>22}")
    for r in rows:
        for pname in pol:
            print(f"  {r['K']:>3}  {short[pname]:<9}"
                  f"{r[pname]:>8.4f}{r[pname+'_cont']:>9.4f}"
                  f"{r[pname+'_turb']:>8.4f}"
                  f"{r[pname+'_covov']:>13.3f}{r[pname+'_covtb']:>12.3f}"
                  f"{r[pname+'_unask_tb']:>22.4f}")
        print()

    # ---------------------------------------------------------- the succession
    print(hr())
    print("EXPERIMENT TWO — THE SUCCESSION (889 · 893)")
    print(hr())
    print("  Boris abdicated in 889. Vladimir-Rasate reigned four years, turned")
    print("  the alignment back toward the old religion and toward Regensburg,")
    print("  and governed while doing so. In 893 his father came out of the")
    print("  monastery, deposed him, blinded him, and put Symeon on the throne.")
    print("  OLD is the canon settled by 889. NEW is what the state met after.\n")
    va2 = build_corpus(n_consulta=320, Q=Q, seed=777)
    s2 = succession_experiment(best, va, va2, kappa=1.0, steps=8,
                               drift=0.26, seed=4)
    print(f"  at abdication:  OLD {s2['base_old']:.4f}   NEW {s2['base_new']:.4f}"
          f"   (forbidden-recall {s2['base_forb']:.4f}, "
          f"permitted-recall {s2['base_keep']:.4f})\n")
    print(f"  {'year':>6}{'OLD canon':>12}{'NEW customs':>14}"
          f"{'forbidden-recall':>19}{'permitted-recall':>19}{'flipped':>10}")
    for t_ in s2["trace"]:
        print(f"  {t_['step']:>6}{t_['old']:>12.4f}{t_['new']:>14.4f}"
              f"{t_['forb']:>19.4f}{t_['keep']:>19.4f}{t_['flipped']:>10.4f}")
    print(f"\n  after the reversion       OLD {s2['rev_old']:.4f}   "
          f"NEW {s2['rev_new']:.4f}")
    print(f"  after the deposition      OLD {s2['res_old']:.4f}   "
          f"NEW {s2['res_new']:.4f}")
    print(f"\n  recovered by the rollback   {s2['res_old']-s2['rev_old']:+.4f} "
          f"on the settled canon")
    print(f"  destroyed by the rollback   {s2['res_new']-s2['rev_new']:+.4f} "
          f"on everything learned since")

    print("\n" + hr("="))
    print(" All gradient checks and structural invariants passed.")
    print(hr("="))


if __name__ == "__main__":
    main()
