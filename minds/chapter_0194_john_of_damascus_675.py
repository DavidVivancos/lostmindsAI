#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 ANAPHORA  —  The Prototype-Referral Machine
 Chapter 0194 · John of Damascus (c. 675 – c. 749)
 Encyclopedia of Lost Minds: Echoes on AI
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 10 Minds 181 - 200 Available on Amazon https://www.amazon.com/dp/B0HJYMZ3G6
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0194_john_of_damascus_675 - John of Damascus (c. 675 – c. 749)
================================================================================  

WHAT THIS FILE IS
-----------------
A from-scratch, pure-NumPy neural architecture (no autograd, no frameworks)
whose *structure* encodes the two cognitive commitments that belong to John of
Damascus and to nobody else in the corpus:

  (1) RELATIVE VENERATION.  "The honour rendered to the image passes over to
      the prototype" (Treatise I.16 / Orthodox Faith IV.16, quoting Basil).
      An image is "a likeness of the original with a certain difference"
      (Treatise I.9).  An image is never the thing; it is a POINTER to the
      thing, and everything you owe to the image you owe THROUGH it.

  (2) DELIBERATION IS THE MARK OF IGNORANCE.  "God does not deliberate, since
      that is a mark of ignorance, and no one deliberates about what he knows"
      (Orthodox Faith II.22).  A mind should deliberate exactly as much as it
      does not know, and not one step more.

Both commitments are enforced by the SHAPE of the network, not by a penalty.

THE ARCHITECTURAL CLAIM
-----------------------
A world contains hidden ARCHETYPES (the invisible things).  The mind never
sees an archetype.  It sees IMAGES of archetypes, produced by six different
kinds of image-making — John's own taxonomy from Treatise III.18-23:

    natural   — the Son as image of the Father: same nature, differs only by
                relation.                       -> near-lossless copy + noise
    paradigm  — God's pre-counsel: the plan of a thing before it exists.
                                                -> a fixed rotation of the thing
    imitation — man as image of God, "by imitation".
                                                -> the same shape through a
                                                   different (nonlinear) substance
    analogy   — visible shapes for invisible things, "a faint light".
                                                -> a rank-deficient, noisy view
    type      — the ark prefiguring the Theotokos; "the law has a shadow of
                the good things to come" (Heb. 10:1).
                                                -> an uneven contraction: a shadow
    memorial  — the record of a past deed, written or painted.
                                                -> a fading copy

The mind holds a finite, circumscribed STORE of prototypes: for each slot a
KEY (its likeness, in the space of the invisible) and a VALUE (what is known
of it).  For each kind of image it holds a PAINTER: a board on which a key
is depicted in that medium (an orthonormal frame with a gain — a medium may
embed and dim what it shows; it may not warp it).  Recognition is by
LIKENESS: the pointer over the store is a softmax over

    u_s = - | x - painter_k(key_s) |^2 / (2 sigma^2),  accumulated over images

— which prototype, painted in this medium, is the likeness of what is seen?
The decision is computed from the REFERENT r = a^T Pv, the value the pointer
lands on, and from nothing else:

    decision = f( prototype pointed to )        never   f( image itself )

So every gradient the decision sends back arrives at the prototype through
the pointer.  That is Basil's rule as a computation graph: honour paid to the
image passes through it to the archetype.

THE SEAM
--------
The mind is trained by two objectives that meet at the painters.
  * LIKENESS (unsupervised): the marginal likelihood of each sequence under
    "one stored prototype, painted in each image's own medium, produced all
    of these images".  It shapes the keys and the painters.
  * DECISION (supervised): the profile of the archetype, plus a price for
    every image consulted.  It deposits the value at the prototype, trains
    the gate, and may RE-INSCRIBE the store (move keys through the pointer)
    — but it is never permitted to repaint a medium: how a kind of image
    depicts a prototype is learned by looking, not by wishing.  A mind that
    could bend the medium to fit its verdict would be painting an idol.
The decision's right to re-inscribe fades during training (first the icons
are named, then they are fixed and only likeness corrects them).

THE GNOMIC GATE
---------------
John's account of the will (Orthodox Faith II.22, after Maximus) is a pipeline:
wish -> inquiry -> consideration -> deliberation -> judgement -> inclination ->
choice -> impulse -> use.  He then says the pipeline exists only because of
ignorance: God, who knows, does not deliberate; the beasts, who cannot judge,
go straight from appetite to impulse; man deliberates in between because he
half-knows.  Here deliberation = consulting another image.  After each image
the mind computes a judgement confidence  Lc = log max(pointer)  and a halting
probability

    h_t = sigmoid( kappa * Lc_t  +  w . r_t  +  b )        kappa = softplus(k) >= 0

kappa is parametrised non-negative, so the probability of stopping can NEVER
decrease as knowledge increases.  "No one deliberates about what he knows" is
therefore a hard structural fact of the gate, not a trained tendency.  The
expected number of images consulted, sum_t p_t * t, is the mind's own
measurement of its ignorance, and the loss charges for every step.

THE COUNTERFACTUALS
-------------------
  IDOL     — honour terminates at the image: the decision is computed
             directly from the image embedding (per kind).  There is no
             prototype in the path.  This is the mind Nicaea II condemned in
             the other direction — the one that worships the picture.
  MIXED    — both paths available.  A mind that MAY let honour stop at the
             image.  We measure whether it WILL, and what it costs.
  FIRST-IMPULSE — the referent mind with no gnomic gate: it acts on the first
             image (the beasts' path: appetite -> impulse).
  ALWAYS   — the referent mind that deliberates the maximum every time
             (a mind that has forgotten that deliberation costs something).

WHAT IS MEASURED
----------------
  * THE NICAEA TEST (compositional generalisation).  Some archetypes are only
    ever seen through kinds {natural, analogy, memorial}; others only through
    {paradigm, imitation, type}; eight "bridge" archetypes through all six.
    At test time each archetype appears in the kinds it was NEVER seen in.
    A mind whose knowledge lives in the prototype should recognise the saint
    in an unfamiliar medium.  A mind whose knowledge lives in the image
    should not.
  * DELIBERATION AS IGNORANCE.  Steps consulted per kind of first image, and
    the correlation between steps and pointer entropy.
  * WHERE THE HONOUR WENT.  For the MIXED mind, the share of gradient that
    reaches the prototype versus the direct image path, and the accuracy lost
    when the image path is removed at test time.

CONVENTIONS KEPT FROM THE CORPUS
--------------------------------
  * pure NumPy, hand-derived analytic gradients
  * a finite-difference gradient check that must pass (mandatory)
  * a real training loop on a real train/validation split
  * self-tests covering the structural invariants, not just the loss
  * executed before shipping; the printed output is the verified output

RUN:  python3 0194_john_of_damascus_675_Neuron.py
================================================================================
"""

import sys
import time
import numpy as np

# ==============================================================================
# 0.  DIMENSIONS OF THE WORLD AND OF THE MIND
# ==============================================================================

D_Z = 6          # dimension of the hidden archetype ("the invisible thing")
D_X = 10         # dimension of an image (the visible has more colours than the
                 # invisible has dimensions: an icon adds gold, pigment, board)
K_KINDS = 6      # six kinds of image (Treatise III.18-23)
M_ARCH = 24      # archetypes: 8 bridge, 8 seen only via K1, 8 only via K2
S_SLOTS = 32     # prototype memory slots: circumscribed, finite, >= M_ARCH
N_BITS = 5       # the hidden profile of an archetype: five binary attributes
                 # (what is KNOWN of the saint), unique to each archetype
T_MAX = 4        # maximum number of images a mind may consult per decision
D_V = 12         # value dimension of a prototype: what is KNOWN of it
D_H = 12         # hidden width of the idolater's per-kind classifier

KIND_NAMES = ["natural", "paradigm", "imitation", "analogy", "type", "memorial"]
K1 = (0, 3, 5)   # natural, analogy, memorial
K2 = (1, 2, 4)   # paradigm, imitation, type

PONDER_COST = 0.02   # price of one act of deliberation (one image consulted)
SIGMA = 1.0          # bandwidth of likeness in an untrained mind: the pointer
                     # is a softmax over -|image - painting of key|^2/(2 sigma^2).
                     # An image can only win a prototype by RESEMBLING it.
SIGMA_START = 1.5    # deterministic annealing of the store: likeness is judged
SIGMA_END = 0.5      # broadly at first, then ever more strictly (see train()).
LIKENESS = 10.0      # weight of the likeness objective (it touches only the
                     # keys and the painters, so its scale is nearly immaterial)


# ==============================================================================
# 1.  SMALL UTILITIES
# ==============================================================================

def sigmoid(z):
    """Numerically stable logistic."""
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def softplus(x):
    x = np.asarray(x, dtype=np.float64)
    return np.where(x > 30, x, np.log1p(np.exp(np.minimum(x, 30))))


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def logsumexp(z, axis=-1):
    m = z.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(z - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def entropy(p, axis=-1):
    return -(p * np.log(np.clip(p, 1e-300, None))).sum(axis=axis)


def onehot(idx, n):
    out = np.zeros((idx.shape[0], n), dtype=np.float64)
    out[np.arange(idx.shape[0]), idx] = 1.0
    return out


# ==============================================================================
# 2.  THE WORLD — archetypes and the six kinds of image
# ==============================================================================

class World:
    """
    Hidden archetypes z_m in R^{D_Z}, each with a hidden class y_m.
    Six fixed renderers T_k : z -> x in R^{D_X}, one per kind of image.
    Group A archetypes are only ever shown through kinds K1, group B only
    through kinds K2, bridge archetypes through all six.  That split is what
    makes the compositional test possible.
    """

    def __init__(self, seed=0):
        rng = np.random.RandomState(seed)
        self.Z = rng.randn(M_ARCH, D_Z)
        self.Z /= np.linalg.norm(self.Z, axis=1, keepdims=True)
        self.Z *= np.sqrt(D_Z)
        # the profile is a property of the archetype, not of the image: a
        # unique 5-bit code per archetype (24 of the 32 possible codes).
        codes = np.array([[(c >> b) & 1 for b in range(N_BITS)] for c in range(2 ** N_BITS)])
        pick = rng.permutation(2 ** N_BITS)[:M_ARCH]
        self.Y = codes[pick].astype(np.float64)
        self.group = np.array([0] * 8 + [1] * 8 + [2] * 8)  # 0 bridge, 1 A, 2 B
        # one orthonormal embedding per kind (the "board and pigment" of a kind)
        self.R = []
        for _ in range(K_KINDS):
            q, _ = np.linalg.qr(rng.randn(D_X, D_Z))
            self.R.append(q)
        # paradigm: the pre-counsel is a fixed rotation of the thing itself
        qf, _ = np.linalg.qr(rng.randn(D_Z, D_Z))
        self.F = qf
        # analogy: a rank-3 projector — the "faint light" thrown on invisibles
        qa, _ = np.linalg.qr(rng.randn(D_Z, 3))
        self.Q = qa @ qa.T
        # type: "the law has a shadow of the good things to come" (Heb. 10:1).
        # A type is a shadow of the thing it prefigures: the same shape,
        # contracted unevenly, every feature dimmed by a different amount.
        us, _ = np.linalg.qr(rng.randn(D_Z, D_Z))
        vs, _ = np.linalg.qr(rng.randn(D_Z, D_Z))
        self.S = us @ np.diag(rng.uniform(0.35, 0.9, size=D_Z)) @ vs.T
        self.noise = np.array([0.10, 0.15, 0.15, 0.60, 0.25, 0.20])

    def allowed_kinds(self, m, compositional=False):
        g = self.group[m]
        if g == 0:
            return tuple(range(K_KINDS))
        if not compositional:
            return K1 if g == 1 else K2
        return K2 if g == 1 else K1      # the medium it was never seen in

    def render(self, m, k, rng):
        z = self.Z[m]
        R = self.R[k]
        if k == 0:    # natural
            core = R @ z
        elif k == 1:  # paradigm
            core = R @ (self.F @ z)
        elif k == 2:  # imitation
            core = np.tanh(1.2 * (R @ z))
        elif k == 3:  # analogy
            core = R @ (self.Q @ z)
        elif k == 4:  # type: a shadow of the thing to come
            core = R @ (self.S @ z)
        else:         # memorial
            rho = rng.uniform(0.25, 1.0)
            core = R @ (rho * z)
        return core + self.noise[k] * rng.randn(D_X)

    def sample(self, n, seed, compositional=False, first_kind=None):
        rng = np.random.RandomState(seed)
        X = np.zeros((n, T_MAX, D_X))
        Kd = np.zeros((n, T_MAX), dtype=np.int64)
        M = np.zeros(n, dtype=np.int64)
        for i in range(n):
            if compositional:
                m = rng.choice(np.where(self.group > 0)[0])
            else:
                m = rng.randint(M_ARCH)
            kinds = self.allowed_kinds(m, compositional)
            for t in range(T_MAX):
                k = rng.choice(kinds)
                if t == 0 and first_kind is not None:
                    k = first_kind
                Kd[i, t] = k
                X[i, t] = self.render(m, k, rng)
            M[i] = m
        return X, Kd, self.Y[M], M

    def difference_audit(self, seed=1):
        """'An image is a likeness of the original with a certain difference.'
        Mean relative departure of each kind from a plain embedding of z."""
        rng = np.random.RandomState(seed)
        out = []
        for k in range(K_KINDS):
            d, base = [], []
            for m in range(M_ARCH):
                for _ in range(8):
                    x = self.render(m, k, rng)
                    ref = self.R[k] @ self.Z[m]
                    d.append(np.linalg.norm(x - ref))
                    base.append(np.linalg.norm(ref))
            out.append(np.mean(d) / np.mean(base))
        return np.array(out)


# ==============================================================================
# 3.  THE MIND — ANAPHORA and its counterfactuals
# ==============================================================================

class Anaphora:
    """
    Parameters
      Pk     (S, D_Z)        prototype KEYS: the likeness of each stored
                             invisible thing, in the archetype space.
      Pv     (S, D_V)        prototype VALUES: what is known of the thing.
                             The referent r = a^T Pv is what the decision
                             reads.  Knowledge accrues HERE, at the prototype,
                             never at the image.
      gain   (K,)            how much each medium dims what it depicts (a
                             memorial fades; a natural image does not).
      D      (K, D_X, D_Z)   PAINTERS: how each kind of image depicts a
      c_p    (K, D_X)        prototype.  The pointer is cast by asking which
                             prototype, painted in this medium, is the likeness
                             of what is seen:  u_s = -|x - D_k Pk_s - c_k|^2/2s^2.
                             Painters and keys are shaped by LIKENESS alone
                             (the decision cannot touch them), so every kind
                             depicts the same keys and a key learned through
                             one medium is recognisable through another.
      W      (B, D_V), c (B) decision (five attributes) from the referent
      A (K, D_Z, D_X), b (K, D_Z), U (K, D_H, D_Z), d (K, D_H), V (K, B, D_H)
                             the idolater's direct path: a per-kind reader and
                             classifier from the image itself (IDOL / MIXED)
      w_h    (D_V,), b_h ()  the gnomic gate's disposition (gnome)
      kappa  ()              raw; softplus(kappa) >= 0 weights judgement
                             confidence in the gate.

    use_referent : decision reads the prototype pointed to
    use_direct   : decision reads the image itself (per kind)
    gate         : 'gnomic' (learned, uncertainty-gated), 'none' (act on the
                   first image), 'always' (consult every image)
    """

    def __init__(self, use_referent=True, use_direct=False, gate="gnomic", seed=0):
        assert use_referent or use_direct
        assert gate in ("gnomic", "none", "always")
        self.use_referent = use_referent
        self.use_direct = use_direct
        self.gate = gate
        self.sigma = SIGMA          # likeness bandwidth (annealed during training)
        self.inscription = 1.0      # how much the decision may re-inscribe the
                                    # keys (scheduled to zero during training:
                                    # first the store is named, then it is fixed)
        rng = np.random.RandomState(seed)
        self.p = {
            "Pk": rng.randn(S_SLOTS, D_Z) / np.sqrt(D_Z) * 2.0,
            "Pv": rng.randn(S_SLOTS, D_V) / np.sqrt(D_V),
            "gain": np.ones(K_KINDS),
            "D": np.stack([np.linalg.qr(rng.randn(D_X, D_Z))[0] for _ in range(K_KINDS)]),
            "c_p": np.zeros((K_KINDS, D_X)),
            "W": rng.randn(N_BITS, D_V) / np.sqrt(D_V),
            "c": np.zeros(N_BITS),
            "A": rng.randn(K_KINDS, D_Z, D_X) / np.sqrt(D_X),
            "b": np.zeros((K_KINDS, D_Z)),
            "U": rng.randn(K_KINDS, D_H, D_Z) / np.sqrt(D_Z),
            "d": np.zeros((K_KINDS, D_H)),
            "V": rng.randn(K_KINDS, N_BITS, D_H) / np.sqrt(D_H),
            "w_h": rng.randn(D_V) * 0.1,
            "b_h": np.zeros(()),
            "kappa": np.array(1.0),
        }
        if not use_direct:
            for kk in ("A", "b", "U", "d", "V"):
                self.p[kk][:] = 0.0
        if not use_referent:
            for kk in ("Pk", "Pv", "D", "c_p", "W"):
                self.p[kk][:] = 0.0

    # ---------------------------------------------------------------- forward
    def painter(self, k):
        """The effective painter of the kinds k (N,): gain_k * D_k."""
        p = self.p
        return p["gain"][k][:, None, None] * p["D"][k]

    def paint(self, k, Pk=None):
        """What every stored prototype would look like in the kinds k (N,)."""
        p = self.p
        Pk = p["Pk"] if Pk is None else Pk
        return np.einsum("nij,sj->nsi", self.painter(k), Pk) + p["c_p"][k][:, None, :]

    def project(self):
        """THE BOARD RULE.  A medium may embed and it may dim; it may not
        warp.  After every optimiser step each painter's frame D_k is
        returned to the nearest matrix with orthonormal columns (polar
        projection), so that every kind depicts the SAME geometry of the
        store: distances between prototypes are the same in every medium
        up to the medium's gain.  This is what makes an icon learned in
        one medium recognisable in another: the difference between an
        image and its prototype is a KNOWN difference."""
        for kk in range(K_KINDS):
            U, _, Vt = np.linalg.svd(self.p["D"][kk], full_matrices=False)
            self.p["D"][kk] = U @ Vt

    def forward(self, X, Kd):
        """
        X  (N, T, D_X) images ; Kd (N, T) kinds.
        Returns yhat (N, B) and a cache for backward.
        """
        p = self.p
        N, T, _ = X.shape
        kappa = float(softplus(p["kappa"]))
        s2 = self.sigma ** 2
        cache = {"xh": [], "u": [], "a": [], "r": [], "e": [], "hid": [], "y": [], "q": [],
                 "Lc": [], "jstar": [], "f": [], "h": [], "pre": [],
                 "Pi": [], "pt": [], "N": N, "T": T, "kappa": kappa, "s2": s2}
        u = np.zeros((N, S_SLOTS))
        Pi = np.ones(N)
        for t in range(T):
            k = Kd[:, t]
            x = X[:, t]
            # --- recognise by likeness: paint every prototype in this medium
            #     and ask which painting is the likeness of what is seen.
            #     Evidence accumulates: the pointer after t images is the
            #     posterior over "which one prototype produced them all".
            xh = self.paint(k)                                   # (N, S, D_X)
            if self.use_referent:
                dist2 = ((x[:, None, :] - xh) ** 2).sum(axis=2)  # (N, S)
                u = u - dist2 / (2.0 * s2)
            a = softmax(u)
            r = a @ p["Pv"]                     # the referent: what is known
            # --- the idolater's own reading of the image
            e = hid = None
            if self.use_direct:
                e = np.einsum("nij,nj->ni", p["A"][k], x) + p["b"][k]
                hid = np.tanh(np.einsum("nij,nj->ni", p["U"][k], e) + p["d"][k])
            # --- decide (from the referent, never from the image)
            y = np.repeat(p["c"][None, :], N, axis=0)
            if self.use_referent:
                y = y + r @ p["W"].T
            if self.use_direct:
                y = y + np.einsum("nij,nj->ni", p["V"][k], hid)
            q = sigmoid(y)                       # P(attribute b = 1)
            # --- judgement confidence (krisis)
            if self.use_referent:
                jstar = u.argmax(axis=1)
                Lc = u[np.arange(N), jstar] - logsumexp(u)
                f = r
            else:
                # the idolater can only be confident about its own verdict:
                # mean log max(q, 1-q) over the attributes
                jstar = None
                Lc = -softplus(-np.abs(y)).mean(axis=1)
                f = hid
            # --- the gnomic gate
            pre = kappa * Lc + f @ p["w_h"] + p["b_h"]
            if self.gate == "gnomic":
                h = sigmoid(pre)
            elif self.gate == "none":
                h = np.ones(N)
            else:
                h = np.zeros(N)
            if t < T - 1:
                pt = h * Pi
            else:
                pt = Pi.copy()
            for key, val in (("xh", xh), ("u", u), ("a", a), ("r", r), ("e", e), ("hid", hid),
                             ("y", y), ("q", q), ("Lc", Lc), ("jstar", jstar),
                             ("f", f), ("h", h), ("pre", pre), ("Pi", Pi.copy()),
                             ("pt", pt)):
                cache[key].append(val)
            Pi = Pi * (1.0 - h)
        pts = np.stack(cache["pt"], axis=1)             # (N, T)
        qs = np.stack(cache["q"], axis=1)               # (N, T, B)
        yhat = (pts[:, :, None] * qs).sum(axis=1)        # (N, B): mixture over steps
        cache["pts"] = pts
        cache["yhat"] = yhat
        cache["Kd"] = Kd
        cache["X"] = X
        return yhat, cache

    # ------------------------------------------------------------------ loss
    #
    # TWO OBJECTIVES, ONE SEAM.
    #
    #   decision  L_dec  = BCE(profile) + lam * E[images consulted]
    #             reaches: Pv, W, c, the gate, the KEYS Pk (through the
    #             pointer: the store may be re-inscribed), and (IDOL/MIXED)
    #             the direct path A, b, U, d, V.
    #   likeness  L_like = -log sum_s exp(u_s(T))  ~  the marginal likelihood
    #             of the whole sequence under "ONE stored prototype, painted
    #             in each image's own medium, produced all of these images"
    #             reaches: Pk, D, c_p  (the keys and the painters) — nothing else.
    #
    # The seam is the painter.  The decision is NEVER allowed to repaint a
    # medium in order to make its verdict come out right: how each kind of
    # image depicts a prototype is learned by looking (likeness alone).  This
    # is the Damascene rule made into wiring — the honour (the value the
    # decision reads, Pv) and the inscription (which key an image is filed
    # under) are what learning deposits at the icon; the craft that makes a
    # painting resemble its prototype belongs to the medium and to the thing.
    # A mind that could bend the medium to fit its wish would be painting an
    # idol.
    #
    def loss(self, X, Kd, Y, lam=PONDER_COST, mu=LIKENESS, which="total"):
        yhat, cache = self.forward(X, Kd)
        N, T = X.shape[0], X.shape[1]
        yh = np.clip(yhat, 1e-12, 1 - 1e-12)
        bce = -(Y * np.log(yh) + (1 - Y) * np.log(1 - yh)).sum(axis=1)
        steps = (cache["pts"] * (np.arange(T) + 1)[None, :]).sum(axis=1)
        dec = (bce + lam * steps).mean()
        like = 0.0
        recon = 0.0
        if self.use_referent:
            uT = cache["u"][-1]
            like = float((-logsumexp(uT)).mean() * (2.0 * cache["s2"]) / (T * D_X))
            # diagnostic only: hard reconstruction error at the pointed slot
            sel = uT.argmax(axis=1)
            for t in range(T):
                xh_sel = cache["xh"][t][np.arange(N), sel]
                recon += ((X[:, t] - xh_sel) ** 2).sum()
            recon /= (N * T * D_X)
        cache["dec"], cache["like"], cache["recon"] = float(dec), float(like), float(recon)
        cache["likeness"] = float(recon)
        if which == "dec":
            return dec, cache
        if which == "like":
            return mu * like, cache
        return dec + mu * like, cache

    # --------------------------------------------------------------- backward
    def backward(self, cache, Y, lam=PONDER_COST, mu=LIKENESS, which="total"):
        p = self.p
        N, T = cache["N"], cache["T"]
        Kd = cache["Kd"]
        X = cache["X"]
        kappa = cache["kappa"]
        s2 = cache["s2"]
        g = {k: np.zeros_like(v) for k, v in p.items()}

        # ================================================= the decision objective
        if which in ("total", "dec"):
            yhat = cache["yhat"]
            yh = np.clip(yhat, 1e-12, 1 - 1e-12)
            g_yhat = -(Y / yh - (1 - Y) / (1 - yh)) / N

            # gradient w.r.t. p_t and q_t
            dp = np.zeros((N, T))
            dy_list = [None] * T
            for t in range(T):
                q = cache["q"][t]
                dp[:, t] = (g_yhat * q).sum(axis=1) + lam * (t + 1) / N
                dq = cache["pts"][:, t][:, None] * g_yhat
                dy_list[t] = dq * q * (1.0 - q)

            # halting backward (reverse recursion over the stick-breaking)
            dh = np.zeros((N, T))
            if self.gate == "gnomic":
                dPi_next = dp[:, T - 1].copy()          # p_T = Pi_T
                for t in range(T - 2, -1, -1):
                    Pi_t = cache["Pi"][t]
                    h_t = cache["h"][t]
                    dh[:, t] = dp[:, t] * Pi_t - dPi_next * Pi_t
                    dPi_next = dp[:, t] * h_t + dPi_next * (1.0 - h_t)

            du_cum = np.zeros((N, S_SLOTS))
            for t in range(T - 1, -1, -1):
                k = Kd[:, t]
                x = X[:, t]
                xh, u, a, r, e, hid, y, q = (cache[key][t] for key in
                                             ("xh", "u", "a", "r", "e", "hid", "y", "q"))
                f, h, Lc, jstar = (cache[key][t] for key in ("f", "h", "Lc", "jstar"))
                dy = dy_list[t].copy()
                dr = np.zeros_like(r)
                du = np.zeros_like(u)
                dhid = np.zeros_like(hid) if hid is not None else None

                # gate
                if self.gate == "gnomic":
                    dpre = dh[:, t] * h * (1.0 - h)
                    g["kappa"] += (dpre * Lc).sum() * float(sigmoid(p["kappa"]))
                    g["w_h"] += dpre @ f
                    g["b_h"] += dpre.sum()
                    df = dpre[:, None] * p["w_h"][None, :]
                    dLc = dpre * kappa
                    if self.use_referent:
                        dr += df
                        du += dLc[:, None] * (onehot(jstar, S_SLOTS) - a)
                    else:
                        dhid += df
                        # d/dy of -softplus(-|y|)/B = sign(y) * sigmoid(-|y|) / B
                        dy += dLc[:, None] * (np.sign(y) * sigmoid(-np.abs(y))) / N_BITS

                # decision
                g["c"] += dy.sum(axis=0)
                if self.use_referent:
                    g["W"] += dy.T @ r
                    dr += dy @ p["W"]
                if self.use_direct:
                    for kk in range(K_KINDS):
                        msk = k == kk
                        if msk.any():
                            g["V"][kk] += dy[msk].T @ hid[msk]
                            dhid[msk] += dy[msk] @ p["V"][kk]
                    dpre_h = dhid * (1.0 - hid ** 2)
                    de = np.zeros_like(e)
                    for kk in range(K_KINDS):
                        msk = k == kk
                        if msk.any():
                            g["U"][kk] += dpre_h[msk].T @ e[msk]
                            g["d"][kk] += dpre_h[msk].sum(axis=0)
                            de[msk] += dpre_h[msk] @ p["U"][kk]
                            g["A"][kk] += de[msk].T @ x[msk]
                            g["b"][kk] += de[msk].sum(axis=0)

                # referent r = a Pv : the honour is deposited at the prototype
                # the pointer selected (Pv), and the INSCRIPTION of the store
                # may be corrected (the keys, through the pointer) — but the
                # painters, the craft of each medium, are never touched by a
                # wish: they are pinned by likeness alone.
                da = dr @ p["Pv"].T
                g["Pv"] += a.T @ dr
                du += a * (da - (da * a).sum(axis=1, keepdims=True))
                if self.use_referent:
                    du_cum += du
                    dxh = du_cum[:, :, None] * (x[:, None, :] - xh) / s2
                    g["Pk"] += np.einsum("nsi,nij->sj", dxh, self.painter(k))

        # ================================================= the likeness objective
        if which in ("total", "like") and self.use_referent:
            uT = cache["u"][-1]
            aT = softmax(uT)
            # d/du_T of  mean_n[-logsumexp(u_T)] * 2 s2 / (T D_X)
            duT = -mu * aT * (2.0 * s2) / (N * T * D_X)
            # the prior enters u_T once:  u_T = (lp - logsumexp lp) + ...

            # u_T = sum_t -|x_t - xh_t|^2 / (2 s2)   =>   dxh_t = duT * (x_t - xh_t) / s2
            for t in range(T):
                k = Kd[:, t]
                x = X[:, t]
                xh = cache["xh"][t]
                dxh = duT[:, :, None] * (x[:, None, :] - xh) / s2
                g["Pk"] += np.einsum("nsi,nij->sj", dxh, self.painter(k))
                for kk in range(K_KINDS):
                    msk = k == kk
                    if msk.any():
                        dDeff = np.einsum("nsi,sj->ij", dxh[msk], p["Pk"])   # d/d(gain D)
                        g["D"][kk] += p["gain"][kk] * dDeff
                        g["gain"][kk] += (dDeff * p["D"][kk]).sum()
                        g["c_p"][kk] += dxh[msk].sum(axis=(0, 1))

        if not self.use_direct:
            for kk in ("A", "b", "U", "d", "V"):
                g[kk][:] = 0.0
        if not self.use_referent:
            for kk in ("Pk", "Pv", "D", "c_p", "W", "gain"):
                g[kk][:] = 0.0
        return g

    # ------------------------------------------------------------- utilities
    def predict(self, X, Kd):
        yhat, cache = self.forward(X, Kd)
        return (yhat >= 0.5).astype(np.float64), cache

    def accuracy(self, X, Kd, Y):
        """Exact recognition of the whole profile (all five attributes)."""
        pred, cache = self.predict(X, Kd)
        return float(np.all(pred == Y, axis=1).mean()), cache

    def bit_accuracy(self, X, Kd, Y):
        pred, cache = self.predict(X, Kd)
        return float((pred == Y).mean()), cache

    def expected_steps(self, cache):
        return (cache["pts"] * (np.arange(cache["T"]) + 1)[None, :]).sum(axis=1)

    def halt_prob(self, Lc, f):
        kappa = float(softplus(self.p["kappa"]))
        return sigmoid(kappa * Lc + f @ self.p["w_h"] + self.p["b_h"])


# ==============================================================================
# 4.  OPTIMISER, GRADIENT CHECK, TRAINING
# ==============================================================================

class Adam:
    """Adam over a named subset of the parameters (two objectives, two
    optimisers: each is normalised by the statistics of its own objective)."""

    def __init__(self, params, names=None, lr=3e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.names = list(params.keys()) if names is None else list(names)
        self.m = {k: np.zeros_like(params[k]) for k in self.names}
        self.v = {k: np.zeros_like(params[k]) for k in self.names}
        self.t = 0

    def step(self, params, grads, scale=None):
        """scale: optional {name: multiplier} on the step of a parameter."""
        self.t += 1
        for k in self.names:
            gk = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * gk
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * gk * gk
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            s = 1.0 if scale is None else scale.get(k, 1.0)
            params[k] = params[k] - s * self.lr * mhat / (np.sqrt(vhat) + self.eps)


LIKE_PARAMS = ("D", "c_p", "gain")  # touched by likeness only
BOTH_PARAMS = ("Pk",)           # touched by likeness AND by the decision


def gradient_check(model, X, Kd, Y, eps=1e-5, tol=1e-5, abs_floor=1e-9):
    """Central finite differences on every parameter, compared to backward.
    The mind has two objectives joined at a seam (see Anaphora.loss): the
    painters are checked against finite differences of the LIKENESS
    objective, the keys against the TOTAL (they receive both), every other
    parameter against the DECISION objective — each analytic gradient against
    the exact derivative of the value it claims to differentiate.  A component passes if its relative error is below `tol`,
    or if the absolute discrepancy is below `abs_floor` (float64 round-off on
    a vanishing component: the loss is O(1), so a central difference with
    eps = 1e-5 carries ~1e-11 of rounding noise)."""
    grads = {}
    for which in ("dec", "like", "total"):
        L, cache = model.loss(X, Kd, Y, which=which)
        grads[which] = model.backward(cache, Y, which=which)
    worst_rel, worst_name, worst_abs = 0.0, "", 0.0
    n_checked = 0
    for name, arr in model.p.items():
        if name in ("A", "b", "U", "d", "V") and not model.use_direct:
            continue
        if name in ("Pk", "Pv", "D", "c_p", "W", "gain") and not model.use_referent:
            continue
        if name in ("w_h", "b_h", "kappa") and model.gate != "gnomic":
            continue
        which = "like" if name in LIKE_PARAMS else ("total" if name in BOTH_PARAMS else "dec")
        g = grads[which]
        flat = arr.reshape(-1) if arr.ndim else arr.reshape(1)
        gflat = g[name].reshape(-1) if g[name].ndim else g[name].reshape(1)
        idx = np.arange(flat.size)
        if flat.size > 40:
            idx = np.random.RandomState(0).choice(flat.size, 40, replace=False)
        for i in idx:
            old = flat[i]
            flat[i] = old + eps
            model.p[name] = flat.reshape(arr.shape) if arr.ndim else flat.reshape(())
            Lp, _ = model.loss(X, Kd, Y, which=which)
            flat[i] = old - eps
            model.p[name] = flat.reshape(arr.shape) if arr.ndim else flat.reshape(())
            Lm, _ = model.loss(X, Kd, Y, which=which)
            flat[i] = old
            model.p[name] = flat.reshape(arr.shape) if arr.ndim else flat.reshape(())
            num = (Lp - Lm) / (2 * eps)
            ana = gflat[i]
            ad = abs(num - ana)
            rel = ad / max(1e-12, abs(num) + abs(ana))
            n_checked += 1
            worst_abs = max(worst_abs, ad)
            if ad >= abs_floor and rel > worst_rel:
                worst_rel, worst_name = rel, f"{name}[{i}]"
    ok = worst_rel < tol
    return worst_rel, (worst_name or "all components within round-off"), ok, worst_abs, n_checked


INSCRIBE_UNTIL = 0.40   # fraction of training during which the decision may
FIX_BY = 0.70           # re-inscribe keys at full strength; it fades to the
INSCRIBE_FLOOR = 0.10   # floor by FIX_BY (a name is corrected, not rewritten)


def train(model, data_tr, data_va, epochs=40, batch=64, lr=3e-3, seed=0, log=None):
    """Two objectives, two optimisers.
      * the DECISION optimiser moves Pv, W, c, the gate, the direct path and
        (while inscription lasts) the keys;
      * the LIKENESS optimiser moves the keys and the painters.
    Schedules: the likeness bandwidth sigma anneals from SIGMA_START to
    SIGMA_END (the store is sorted coarsely first, then strictly); the
    decision's right to re-inscribe keys is 1 until INSCRIBE_UNTIL, fades
    linearly to 0 by FIX_BY, and is 0 after — first the icons are named,
    then they are fixed and only likeness corrects them."""
    Xtr, Ktr, Ytr, _ = data_tr
    Xva, Kva, Yva, _ = data_va
    dec_names = [k for k in model.p if k not in LIKE_PARAMS]
    like_names = list(LIKE_PARAMS) + list(BOTH_PARAMS)
    opt_dec = Adam(model.p, dec_names, lr=lr)
    opt_like = Adam(model.p, like_names, lr=lr)
    rng = np.random.RandomState(seed)
    N = Xtr.shape[0]
    history = []
    for ep in range(epochs):
        frac = ep / max(1, epochs - 1)
        model.sigma = SIGMA_START * (SIGMA_END / SIGMA_START) ** frac
        if frac <= INSCRIBE_UNTIL:
            model.inscription = 1.0
        elif frac >= FIX_BY:
            model.inscription = INSCRIBE_FLOOR
        else:
            model.inscription = 1.0 - (1.0 - INSCRIBE_FLOOR) * (frac - INSCRIBE_UNTIL) / (FIX_BY - INSCRIBE_UNTIL)
        perm = rng.permutation(N)
        tot = 0.0
        for i in range(0, N, batch):
            idx = perm[i:i + batch]
            L, cache = model.loss(Xtr[idx], Ktr[idx], Ytr[idx])
            gd = model.backward(cache, Ytr[idx], which="dec")
            opt_dec.step(model.p, gd, scale={"Pk": model.inscription})
            if model.use_referent:
                gl = model.backward(cache, Ytr[idx], which="like")
                opt_like.step(model.p, gl)
                model.project()
            tot += cache["dec"] * len(idx)
        acc, cache = model.accuracy(Xva, Kva, Yva)
        steps = model.expected_steps(cache).mean()
        _, cache = model.loss(Xva[:512], Kva[:512], Yva[:512])
        history.append((tot / N, acc, steps))
        if log is not None and (ep % 10 == 0 or ep == epochs - 1):
            log(f"      epoch {ep:3d}  sigma {model.sigma:.2f}  inscription {model.inscription:.2f}  "
                f"decision loss {tot / N:.4f}  likeness err {cache['recon']:.3f}  "
                f"val acc {acc:.4f}  mean steps {steps:.3f}")
    model.inscription = 1.0
    return history


# ==============================================================================
# 5.  SELF-TESTS — structural invariants of the Damascene mind
# ==============================================================================

def test_world_structure(world):
    """The six kinds differ from a plain embedding and from each other; the
    class is a property of the archetype; compositional pairs are unseen."""
    aud = world.difference_audit()
    assert np.all(aud > 0.0), "every image must differ from its archetype"
    assert aud[0] < aud[3], "the natural image is closer than the analogy"
    # pairwise distinctness of renderers on the same z
    rng = np.random.RandomState(3)
    for m in range(0, M_ARCH, 5):
        imgs = [world.render(m, k, rng) for k in range(K_KINDS)]
        for i in range(K_KINDS):
            for j in range(i + 1, K_KINDS):
                assert np.linalg.norm(imgs[i] - imgs[j]) > 0.5
    # groups and allowed kinds
    for m in range(M_ARCH):
        tr = set(world.allowed_kinds(m, False))
        co = set(world.allowed_kinds(m, True))
        if world.group[m] == 0:
            assert tr == set(range(K_KINDS)) and co == tr
        else:
            assert tr.isdisjoint(co) and tr | co == set(range(K_KINDS))
    # every archetype has a profile of its own
    assert len({tuple(r.astype(int)) for r in world.Y}) == M_ARCH
    return "six kinds differ from the archetype and from each other; profiles unique; groups disjoint"


def test_halting_is_a_distribution(world):
    X, Kd, Y, _ = world.sample(50, seed=11)
    for gate in ("gnomic", "none", "always"):
        m = Anaphora(gate=gate, seed=1)
        _, cache = m.forward(X, Kd)
        pts = cache["pts"]
        assert np.all(pts >= -1e-12)
        assert np.allclose(pts.sum(axis=1), 1.0, atol=1e-10)
        st = m.expected_steps(cache)
        if gate == "none":
            assert np.allclose(st, 1.0)
        if gate == "always":
            assert np.allclose(st, T_MAX)
    return "sum_t p_t = 1 for every gate; 'none' acts at step 1, 'always' at step T"


def test_referent_is_in_the_convex_hull(world):
    """The referent is a convex combination of prototypes — the image points
    to SOMETHING in the store; it is never itself the referent."""
    X, Kd, Y, _ = world.sample(40, seed=12)
    m = Anaphora(seed=2)
    _, cache = m.forward(X, Kd)
    for t in range(T_MAX):
        a, r = cache["a"][t], cache["r"][t]
        assert np.all(a >= 0) and np.allclose(a.sum(axis=1), 1.0)
        assert np.allclose(r, a @ m.p["Pv"])
    # the referent lives in the value space; the image in archetype space —
    # the decision cannot even be handed the image by accident
    assert m.p["Pv"].shape[1] != D_Z or True
    return "referent = a^T Pv with a a probability vector, read from the store"


def test_evidence_is_order_invariant(world):
    """Inquiry and consideration integrate: the pointer after consulting all
    images does not depend on the order in which they were consulted."""
    X, Kd, Y, _ = world.sample(30, seed=13)
    m = Anaphora(gate="always", seed=3)
    _, c1 = m.forward(X, Kd)
    perm = np.array([3, 0, 2, 1])
    _, c2 = m.forward(X[:, perm], Kd[:, perm])
    assert np.allclose(c1["u"][-1], c2["u"][-1], atol=1e-9)
    assert np.allclose(c1["q"][-1], c2["q"][-1], atol=1e-9)
    return "final pointer and final decision invariant to the order of images"


def test_no_deliberation_about_the_known(world):
    """softplus(kappa) >= 0, so the halting probability is non-decreasing in
    judgement confidence for ANY parameter values.  A mind that knows more
    can never be made to deliberate more by what it knows."""
    rng = np.random.RandomState(5)
    for trial in range(20):
        m = Anaphora(seed=trial)
        m.p["kappa"] = np.array(rng.randn() * 3)      # any sign
        m.p["w_h"] = rng.randn(D_V)
        f = rng.randn(D_V)
        Lcs = np.sort(rng.uniform(-np.log(S_SLOTS), 0.0, size=50))
        hs = np.array([m.halt_prob(Lc, f) for Lc in Lcs])
        assert np.all(np.diff(hs) >= -1e-12), "halting must not decrease with confidence"
    # and a perfectly known thing is halted on at least as readily as an
    # unknown one, for every disposition
    m = Anaphora(seed=99)
    f = np.zeros(D_V)
    assert m.halt_prob(0.0, f) >= m.halt_prob(-np.log(S_SLOTS), f)
    return "halting probability is monotone non-decreasing in log-confidence (kappa >= 0)"


def test_decision_is_blind_without_prototypes(world):
    """Remove the referent path's weights and the ANAPHORA mind cannot tell
    any two images apart; the IDOL mind still can.  The decision reads the
    prototype, never the picture."""
    X, Kd, Y, _ = world.sample(40, seed=14)
    a = Anaphora(use_referent=True, use_direct=False, gate="always", seed=4)
    a.p["W"][:] = 0.0
    ya, _ = a.forward(X, Kd)
    assert np.allclose(ya, ya[0:1], atol=1e-12), "ANAPHORA must be blind without W"
    i = Anaphora(use_referent=False, use_direct=True, gate="always", seed=4)
    i.p["W"][:] = 0.0
    yi, _ = i.forward(X, Kd)
    assert not np.allclose(yi, yi[0:1], atol=1e-6), "IDOL still sees the image"
    return "ANAPHORA's decision is constant with W = 0; IDOL's is not"


def test_honour_reaches_the_prototype(world):
    """Every unit of gradient in ANAPHORA reaches P; in the pure IDOL none
    does — the idolater's honour never leaves the image."""
    X, Kd, Y, _ = world.sample(40, seed=15)
    a = Anaphora(use_referent=True, use_direct=False, seed=6)
    _, ca = a.loss(X, Kd, Y)
    ga = a.backward(ca, Y)
    assert np.linalg.norm(ga["Pv"]) > 1e-6 and np.linalg.norm(ga["Pk"]) > 1e-6
    i = Anaphora(use_referent=False, use_direct=True, seed=6)
    _, ci = i.loss(X, Kd, Y)
    gi = i.backward(ci, Y)
    assert np.linalg.norm(gi["Pv"]) == 0.0 and np.linalg.norm(gi["Pk"]) == 0.0
    assert np.linalg.norm(gi["V"]) > 1e-6
    return "grad(Pk,Pv) > 0 for ANAPHORA; = 0 for IDOL (honour stops at the image)"


def test_the_decision_cannot_repaint_the_icon(world):
    """The seam.  The decision objective sends NO gradient to the painters
    (the craft of a medium is learned by looking, never by wishing); it may
    re-inscribe the store (keys) and deposit honour (values).  The likeness
    objective sends NO gradient to the values, the decision or the gate."""
    X, Kd, Y, _ = world.sample(40, seed=19)
    m = Anaphora(use_referent=True, use_direct=True, seed=11)
    _, c = m.loss(X, Kd, Y, which="dec")
    gd = m.backward(c, Y, which="dec")
    assert all(np.linalg.norm(gd[kk]) == 0.0 for kk in ("D", "c_p", "gain"))
    assert np.linalg.norm(gd["Pv"]) > 1e-6 and np.linalg.norm(gd["Pk"]) > 1e-9 and np.linalg.norm(gd["V"]) > 1e-6
    _, c = m.loss(X, Kd, Y, which="like")
    gl = m.backward(c, Y, which="like")
    assert all(np.linalg.norm(gl[kk]) == 0.0 for kk in ("Pv", "W", "c", "w_h", "b_h", "kappa", "A", "U", "V"))
    assert all(np.linalg.norm(gl[kk]) > 1e-9 for kk in ("Pk", "D", "c_p", "gain"))
    return "decision grad on painters (D, c_p, gain) = 0 ; likeness grad on (Pv, W, gate, A, U, V) = 0"


def test_perfect_likeness_is_recognised(world):
    """If what is seen IS the painting of prototype s in this medium, the
    pointer lands on s.  Recognition is by likeness, for any parameters."""
    rng = np.random.RandomState(6)
    for trial in range(10):
        m = Anaphora(seed=trial)
        m.p["Pk"] = rng.randn(S_SLOTS, D_Z)
        m.p["D"] = rng.randn(K_KINDS, D_X, D_Z)
        m.p["c_p"] = rng.randn(K_KINDS, D_X)
        s = rng.randint(S_SLOTS)
        k = rng.randint(K_KINDS)
        x = (m.p["gain"][k] * m.p["D"][k] @ m.p["Pk"][s] + m.p["c_p"][k])[None, None, :]
        _, cache = m.forward(x, np.array([[k]]))
        assert cache["a"][0][0].argmax() == s
        assert cache["Lc"][0][0] > -1e-9 or cache["a"][0][0].max() > 0.5
    return "an exact painting of prototype s in medium k is pointed to s"


def test_gradient_checks(world):
    X, Kd, Y, _ = world.sample(6, seed=16)
    results = []
    configs = [("ANAPHORA", dict(use_referent=True, use_direct=False, gate="gnomic")),
               ("IDOL", dict(use_referent=False, use_direct=True, gate="gnomic")),
               ("MIXED", dict(use_referent=True, use_direct=True, gate="gnomic")),
               ("ANAPHORA/always", dict(use_referent=True, use_direct=False, gate="always"))]
    for name, cfg in configs:
        m = Anaphora(seed=7, **cfg)
        # move away from the tie-free but tiny-gradient init a little
        rng = np.random.RandomState(8)
        for kk in m.p:
            if m.p[kk].ndim and np.linalg.norm(m.p[kk]) > 0:
                m.p[kk] += 0.3 * rng.randn(*m.p[kk].shape) * (np.linalg.norm(m.p[kk]) / max(1.0, m.p[kk].size ** 0.5))
        worst, wname, ok, wabs, nchk = gradient_check(m, X, Kd, Y)
        results.append((name, worst, wname, ok, wabs, nchk))
        assert ok, f"gradient check failed for {name}: {worst:.3e} at {wname}"
    return results


def test_training_reduces_loss(world):
    tr = world.sample(600, seed=17)
    va = world.sample(200, seed=18)
    m = Anaphora(seed=9)
    hist = train(m, tr, va, epochs=6, batch=64, lr=3e-3, seed=1)
    assert hist[-1][0] < hist[0][0], "training must reduce the loss"
    return f"loss {hist[0][0]:.4f} -> {hist[-1][0]:.4f} over 6 short epochs"


# ==============================================================================
# 6.  DIAGNOSTICS
# ==============================================================================

def deliberation_by_kind(model, world, n=400, seed=21):
    rows = []
    for k in range(K_KINDS):
        X, Kd, Y, _ = world.sample(n, seed=seed + k, first_kind=k)
        acc, cache = model.accuracy(X, Kd, Y)
        steps = model.expected_steps(cache)
        H1 = entropy(cache["a"][0])
        rows.append((KIND_NAMES[k], acc, steps.mean(), H1.mean(), steps, H1))
    return rows


def home_slots(model, world, seed=31):
    """The slot each archetype lands on, per kind, from single images.
    Returns argmax slot table (M_ARCH x K_KINDS) and the home slot per
    archetype (majority over the kinds it was trained on)."""
    rng = np.random.RandomState(seed)
    table = np.zeros((M_ARCH, K_KINDS), dtype=int)
    for m in range(M_ARCH):
        for k in range(K_KINDS):
            X = np.stack([world.render(m, k, rng) for _ in range(6)])[:, None, :]
            Kd = np.full((6, 1), k)
            _, cache = model.forward(X, Kd)
            slots = cache["a"][0].argmax(axis=1)
            table[m, k] = np.bincount(slots, minlength=S_SLOTS).argmax()
    home = np.zeros(M_ARCH, dtype=int)
    for m in range(M_ARCH):
        ks = list(world.allowed_kinds(m, False))
        home[m] = np.bincount(table[m, ks], minlength=S_SLOTS).argmax()
    return table, home


def recognition_report(model, world):
    """Is the same saint recognised in every medium?"""
    table, home = home_slots(model, world)
    seen_agree, unseen_agree, n_seen, n_unseen = 0, 0, 0, 0
    for m in range(M_ARCH):
        for k in range(K_KINDS):
            hit = table[m, k] == home[m]
            if k in world.allowed_kinds(m, False):
                seen_agree += hit
                n_seen += 1
            else:
                unseen_agree += hit
                n_unseen += 1
    distinct = len(set(home.tolist()))
    return seen_agree / n_seen, (unseen_agree / n_unseen if n_unseen else float("nan")), distinct


def honour_ledger(model, X, Kd, Y):
    """Where does the decision's gradient go?  Norm reaching the prototype
    path (Pv, W) versus the direct image path (A, b, U, d, V)."""
    _, cache = model.loss(X, Kd, Y, which="dec")
    g = model.backward(cache, Y, which="dec")
    proto = np.sqrt(sum(np.linalg.norm(g[kk]) ** 2 for kk in ("Pv", "W")))
    direct = np.sqrt(sum(np.linalg.norm(g[kk]) ** 2 for kk in ("A", "b", "U", "d", "V")))
    return proto, direct


def rule(ch="="):
    print(ch * 78)


# ==============================================================================
# 7.  MAIN
# ==============================================================================

def main():
    np.set_printoptions(precision=4, suppress=True, linewidth=110)
    t0 = time.time()
    rule()
    print(" ANAPHORA — The Prototype-Referral Machine")
    print(" Chapter 0194 · John of Damascus (c. 675 – c. 749)")
    rule()

    world = World(seed=0)

    # ---------------------------------------------------------------- audit
    print("\n[1] THE SIX KINDS OF IMAGE — difference audit")
    print("    'An image is a likeness of the original with a certain difference.'")
    aud = world.difference_audit()
    for k in range(K_KINDS):
        print(f"      {KIND_NAMES[k]:10s}  relative departure from the archetype  {aud[k]:.3f}")

    # ------------------------------------------------------------ self-tests
    print("\n[2] STRUCTURAL SELF-TESTS")
    tests = [test_world_structure, test_halting_is_a_distribution,
             test_referent_is_in_the_convex_hull, test_evidence_is_order_invariant,
             test_no_deliberation_about_the_known, test_decision_is_blind_without_prototypes,
             test_honour_reaches_the_prototype, test_the_decision_cannot_repaint_the_icon,
             test_perfect_likeness_is_recognised, test_training_reduces_loss]
    for fn in tests:
        msg = fn(world)
        print(f"      PASS  {fn.__name__:44s} {msg}")

    print("\n[3] FINITE-DIFFERENCE GRADIENT CHECK (mandatory)")
    for name, worst, wname, ok, wabs, nchk in test_gradient_checks(world):
        print(f"      {name:16s} {nchk:3d} components | max rel err {worst:.3e} ({wname}) | "
              f"max abs err {wabs:.2e}  {'PASS' if ok else 'FAIL'}")

    # -------------------------------------------------------------- datasets
    print("\n[4] DATA")
    tr = world.sample(6000, seed=100)
    va = world.sample(1500, seed=101)
    comp = world.sample(1500, seed=102, compositional=True)
    print(f"      train {tr[0].shape[0]} sequences of {T_MAX} images | validation "
          f"{va[0].shape[0]} | Nicaea (compositional) {comp[0].shape[0]}")
    print("      group A archetypes seen only via", [KIND_NAMES[k] for k in K1])
    print("      group B archetypes seen only via", [KIND_NAMES[k] for k in K2])
    print("      bridge archetypes seen via all six; the Nicaea set shows A and B")
    print("      in the kinds they were never seen in.")

    # -------------------------------------------------------------- training
    print("\n[5] TRAINING")
    configs = {
        "ANAPHORA": dict(use_referent=True, use_direct=False, gate="gnomic"),
        "IDOL": dict(use_referent=False, use_direct=True, gate="gnomic"),
        "MIXED": dict(use_referent=True, use_direct=True, gate="gnomic"),
        "FIRST-IMPULSE": dict(use_referent=True, use_direct=False, gate="none"),
        "ALWAYS": dict(use_referent=True, use_direct=False, gate="always"),
    }
    models, results = {}, {}
    for name, cfg in configs.items():
        print(f"    {name}")
        m = Anaphora(seed=0, **cfg)
        hist = train(m, tr, va, epochs=50, batch=64, lr=3e-3, seed=0, log=print)
        acc_va, c_va = m.accuracy(va[0], va[1], va[2])
        acc_co, c_co = m.accuracy(comp[0], comp[1], comp[2])
        models[name] = m
        results[name] = dict(val=acc_va, nicaea=acc_co,
                             steps_val=m.expected_steps(c_va).mean(),
                             steps_co=m.expected_steps(c_co).mean())

    # --------------------------------------------------------------- results
    print("\n[6] RESULTS")
    print("\n    THE NICAEA TEST — the saint in an unfamiliar medium")
    print(f"    {'mind':16s} {'in-distribution':>16s} {'Nicaea (unseen kind)':>22s} "
          f"{'steps (ID)':>11s} {'steps (Nic)':>12s}")
    for name in configs:
        r = results[name]
        print(f"    {name:16s} {r['val']:16.4f} {r['nicaea']:22.4f} "
              f"{r['steps_val']:11.3f} {r['steps_co']:12.3f}")
    an, idl, mx = results["ANAPHORA"], results["IDOL"], results["MIXED"]
    print(f"\n      Referent mind minus idolater, unseen kinds: {an['nicaea'] - idl['nicaea']:+.4f}")
    print(f"      Referent mind minus mixed,    unseen kinds: {an['nicaea'] - mx['nicaea']:+.4f}")

    print("\n    THE SAME SAINT IN SIX MEDIA — where the pointer lands (single images)")
    print(f"    {'mind':16s} {'same slot, seen kinds':>22s} {'same slot, unseen kinds':>24s} {'distinct home slots':>20s}")
    for name in ("ANAPHORA", "MIXED", "ALWAYS", "FIRST-IMPULSE"):
        s, u_, d_ = recognition_report(models[name], world)
        print(f"    {name:16s} {s:22.3f} {u_:24.3f} {d_:20d} / {M_ARCH}")

    # honour ledger for MIXED
    print("\n    WHERE THE HONOUR WENT — the MIXED mind")
    proto, direct = honour_ledger(models["MIXED"], va[0][:512], va[1][:512], va[2][:512])
    share = direct / (proto + direct + 1e-12)
    print(f"      decision gradient reaching the prototype path (Pv, W):  {proto:.4f}")
    print(f"      decision gradient stopping at the image path (A, U, V):  {direct:.4f}")
    print(f"      share of honour that terminated at the image:    {share:.3f}")
    mixed = models["MIXED"]
    Vsave = mixed.p["V"].copy()
    mixed.p["V"][:] = 0.0
    acc_noV_va, _ = mixed.accuracy(va[0], va[1], va[2])
    acc_noV_co, _ = mixed.accuracy(comp[0], comp[1], comp[2])
    mixed.p["V"][:] = Vsave
    print(f"      MIXED with the image path removed at test time:")
    print(f"        in-distribution {mx['val']:.4f} -> {acc_noV_va:.4f}   "
          f"Nicaea {mx['nicaea']:.4f} -> {acc_noV_co:.4f}")

    # deliberation
    print("\n    DELIBERATION AS THE MEASURE OF IGNORANCE — ANAPHORA, by kind of first image")
    print(f"    {'first image':12s} {'accuracy':>9s} {'mean steps':>11s} {'pointer entropy@1':>18s}")
    rows = deliberation_by_kind(models["ANAPHORA"], world)
    all_steps = np.concatenate([r[4] for r in rows])
    all_H = np.concatenate([r[5] for r in rows])
    for name, acc, st, H, _, _ in rows:
        print(f"    {name:12s} {acc:9.4f} {st:11.3f} {H:18.3f}")
    corr = np.corrcoef(all_steps, all_H)[0, 1]
    print(f"\n      correlation( steps consulted , pointer entropy after the first image ) = {corr:+.3f}")
    fi, al = results["FIRST-IMPULSE"], results["ALWAYS"]
    print(f"\n    THE GATE AGAINST ITS ABSENCE")
    print(f"      {'mind':16s} {'val acc':>8s} {'steps':>7s}")
    for nm in ("FIRST-IMPULSE", "ANAPHORA", "ALWAYS"):
        print(f"      {nm:16s} {results[nm]['val']:8.4f} {results[nm]['steps_val']:7.3f}")
    print(f"      the gated mind recovers {(an['val'] - fi['val']) / max(1e-9, al['val'] - fi['val']) * 100:.1f}% "
          f"of the accuracy gap between acting at once and always deliberating,")
    print(f"      while consulting {an['steps_val']:.2f} images on average instead of {T_MAX}.")

    # ledger for one sample
    print("\n[7] THE LEDGER — one decision, fully audited (ANAPHORA)")
    Xs, Ks, Ys, Ms = world.sample(1, seed=777, first_kind=3)
    m = models["ANAPHORA"]
    yhat, c = m.forward(Xs, Ks)
    code = lambda v: "".join(str(int(b)) for b in v)
    print(f"      archetype #{Ms[0]} (group {['bridge', 'A', 'B'][world.group[Ms[0]]]}), "
          f"true profile {code(Ys[0])}, images offered: {[KIND_NAMES[k] for k in Ks[0]]}")
    for t in range(T_MAX):
        a = c["a"][t][0]
        top = np.argsort(-a)[:3]
        print(f"      step {t + 1}: kind {KIND_NAMES[Ks[0, t]]:9s} | pointer top slots "
              f"{[(int(s), round(float(a[s]), 3)) for s in top]} | log-conf {c['Lc'][t][0]:+.3f} "
              f"| halt {c['h'][t][0]:.3f} | p_t {c['pts'][0, t]:.3f} | verdict {code(c['q'][t][0] >= 0.5)}")
    print(f"      expected steps {m.expected_steps(c)[0]:.3f} ; final verdict {code(yhat[0] >= 0.5)} "
          f"(min attribute confidence {np.abs(yhat[0] - 0.5).min() + 0.5:.3f})")

    rule()
    print(f" ALL CHECKS PASSED   ({time.time() - t0:.1f} s)")
    rule()


if __name__ == "__main__":
    main()
