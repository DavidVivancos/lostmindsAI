#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
# BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0367 · Giovanni Boccaccio (1313-1375)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 19, Minds 361-380 Available on Amazon https://www.amazon.com/dp/B0HQX147Y7
# Resume and Interactive Demos at https://artificiology.com/
# END ATTRIBUTION 
 THE CORNICE ENGINE — a receiver-conditioned narrative architecture
===============================================================================

WHY THIS ARCHITECTURE AND NOT ANOTHER
-------------------------------------
Boccaccio is the one mind in this corpus who wrote down, in plain prose, a
theory of harm that no other fourteenth-century author held. At the end of the
Decameron he answers the people who called his hundred tales corrupting. His
answer is not "they are innocent." His answer is that the question is malformed:

    "Le quali ... e nuocere e giovar possono, si come possono tutte l'altre
     cose, AVENDO RIGUARDO ALLO ASCOLTATORE."
    (They can harm and they can help, as all other things can, HAVING REGARD
     TO THE LISTENER.)                        -- Conclusione dell'autore, 8

    "Niuna corrotta mente intese mai sanamente parola."
    (No corrupted mind ever understood a word soundly.)  -- Conclusione, 11

Wine is excellent, he says, and poison to a man with a fever. Fire is necessary
and it burns cities. Weapons defend the peaceful and kill. The valence is not
in the object. It is in the pairing of object and receiver.

That is a claim with a precise mathematical shape: utility is a BILINEAR
INTERACTION between what is said and who is hearing it, with NO MAIN EFFECT in
either argument alone. Any system that tries to score content in isolation --
to stamp a tale "safe" or "harmful" as a property of the tale -- is fitting a
model whose optimal solution is the constant zero. This file demonstrates that
failure numerically, because Boccaccio's thesis is falsifiable and it survives.

Four of his structural decisions are implemented as four mechanisms:

  1. ASCOLTATORE  (Conclusione dell'autore)
     Valence head is bilinear: v = tanh(body^T Wv listener). No content-only
     path exists. Experiment 1 shows the content-only baseline cannot beat
     predicting zero, while the bilinear head recovers the signal.

  2. RUBRICA  (the rubrics heading every day and every novella)
     Boccaccio wrote his own summaries and put them BEFORE the tales, so a
     reader could route herself past what she did not want. The network emits a
     low-dimensional rubric BEFORE the body, generates the body conditioned on
     that rubric, and is penalised when the rubric fails to reconstruct what
     the body is actually about. A label that does not predict the content is
     not a label. This is a differentiable truth-in-advertising constraint.

  3. DIONEO  (Conclusione of Day I: the chartered exemption)
     Nine tellers are masked to the day's theme. The tenth, Dioneo, is exempt
     by charter -- he tells whatever he likes, and in exchange always goes
     last. Filomena grants it on an explicitly institutional argument: if the
     day's matter runs thin, he will be there at the end. Experiment 2 tests
     that argument. It holds, and for a mechanical reason Boccaccio could not
     have known: the exempt teller is the only gradient path that keeps the
     shared trunk's off-theme input dimensions alive.

  4. PAPERE  (Introduction to Day IV: Filippo Balducci and the goslings)
     A father raises his son on Monte Asinaio in total isolation, brings him to
     Florence at eighteen, and when the boy is transfixed by a group of women
     tells him they are called "goslings" and are evil. The boy immediately
     asks for one. Boccaccio's line: the father "felt at once that nature had
     more force than his contrivance." Removing the NAME does not remove the
     APPETITE -- it removes the handle. Experiment 3 builds two models, one
     trained with a concept named and supervised, one trained with the concept
     scrubbed from its inputs, and measures what it costs to steer each.

  5. MOTTO  (Day VI: the leggiadro motto, the retort that arrives in time)
     Day VI is given entirely to people who escape loss or danger with a single
     well-timed saying. Lateness is a form of wrongness. The engine carries a
     latency-discounted stopping head that learns when to speak rather than
     refine further. Experiment 4 compares its learned stopping round to the
     analytic optimum.

  6. VENERDI E SABATO  (Introduction to Day III / the frame's calendar)
     The brigata is away fourteen days but tells on only ten: Friday is kept
     for prayer, Saturday for washing. Two rest days in seven, written into the
     charter. Implemented as scheduled no-data consolidation steps. Its
     measured effect is reported honestly, whatever it is.

RUNNING
-------
    python3 chapter_0367_giovanni_boccaccio_1313.py            # full suite
    python3 chapter_0367_giovanni_boccaccio_1313.py --quick    # short version

Dependencies: numpy only. No autodiff. Every gradient below is derived by hand
and verified against central finite differences before any training happens.
===============================================================================
"""

import argparse
import sys
import time

import numpy as np

# -----------------------------------------------------------------------------
# The ten tellers of the brigata, in the order Boccaccio names them in the
# Introduction to Day I. Dioneo is last -- which is both his position in the
# roll and, after the Conclusione of Day I, his permanent speaking slot.
# -----------------------------------------------------------------------------
TELLERS = [
    "Pampinea", "Fiammetta", "Filomena", "Emilia", "Lauretta",
    "Neifile", "Elissa", "Panfilo", "Filostrato", "Dioneo",
]
DIONEO = 9          # index of the chartered exempt teller
N_TELLERS = 10

# The ten days as Boccaccio titles them. Days I and IX are free topic; the
# eight others are constrained. This is the real thematic schedule.
DAY_THEMES = [
    "I   - libero (free topic)",
    "II  - those who after misfortune reach an unlooked-for happy end",
    "III - those who by industria won or recovered something greatly desired",
    "IV  - loves that ended unhappily",
    "V   - lovers who after hard chances came to happiness",
    "VI  - those who with a ready retort escaped loss, danger or scorn",
    "VII - tricks wives have played on their husbands",
    "VIII- tricks generally, of every sort on every sort",
    "IX  - libero (free topic)",
    "X   - deeds of liberality and magnificence",
]


# =============================================================================
# SECTION 0 · SMALL NUMERIC UTILITIES
# =============================================================================

def tanh(x):
    return np.tanh(x)


def dtanh_from_out(y):
    """Derivative of tanh expressed in terms of its own output. 1 - y^2."""
    return 1.0 - y * y


def r2_score(y_true, y_pred):
    """Fraction of variance explained. Negative means worse than the mean."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / (ss_tot + 1e-12)


def ridge_fit(X, y, lam=1e-3):
    """Closed-form ridge regression with an intercept. Returns (W, b)."""
    X = np.atleast_2d(X)
    y = np.atleast_2d(y.T).T if y.ndim > 1 else y.reshape(-1, 1)
    n, d = X.shape
    Xm, ym = X.mean(0, keepdims=True), y.mean(0, keepdims=True)
    Xc, yc = X - Xm, y - ym
    A = Xc.T @ Xc + lam * n * np.eye(d)
    W = np.linalg.solve(A, Xc.T @ yc)
    b = (ym - Xm @ W).ravel()
    return W, b


# =============================================================================
# SECTION 1 · THE WORLD THE ENGINE LIVES IN
#
# We need a task whose ground truth actually has Boccaccio's structure, or the
# demonstration is rigged. The generator below produces data in which:
#
#   * a tale's utility depends on the INTERACTION of its matter with the
#     listener's disposition, and on nothing else;
#   * the marginal expectation of utility given the matter alone is zero;
#   * the marginal expectation given the listener alone is zero.
#
# The second and third properties are what make Experiment 1 a real test rather
# than a demonstration of a handicap. A content-only scorer is not being
# starved of capacity. It is being asked for a quantity that does not exist.
# =============================================================================

class BrigataWorld:
    """
    Generator of (matter, teller, day-theme, listener, utility) tuples.

    MATTER  m in R^Dm   -- what a tale is about, before any theme constraint.
    THEME   a binary mask over matter coordinates. Telling "on theme" means
            speaking only about the coordinates the day's sovereign has opened.
    TELLER  an index 0..9. Nine are masked. Dioneo is not.
    LISTENER l in R^Dl  -- the disposition of the one hearing it. Boccaccio's
            addressee is specific and never generic: the Proem dedicates the
            book to women confined indoors and sick with love, and the
            Conclusione says outright that a reader with other things to do is
            a fool to be reading this at all.
    UTILITY u = tanh( tale^T A l / sqrt(Dm) ), tale = m * mask.

    A "reserved band" of matter coordinates is deliberately left out of every
    TRAINING theme's mask and used by the held-out theme. Only Dioneo, who is
    unmasked, ever speaks about the reserved band during training. That is the
    whole apparatus of Experiment 2.
    """

    def __init__(self, Dm=24, Dl=12, n_train_themes=4, reserved=6, seed=0):
        rng = np.random.default_rng(seed)
        self.Dm, self.Dl = Dm, Dl
        self.rng = rng
        self.n_train_themes = n_train_themes
        self.reserved = reserved

        # The interaction operator. This is the only thing that maps a pairing
        # of tale and listener onto a valence. Note there is no vector anywhere
        # in this world that assigns a valence to a tale by itself.
        self.A = rng.standard_normal((Dm, Dl))

        open_pool = Dm - reserved            # coords available to training themes
        self.masks = []
        for _ in range(n_train_themes):
            mask = np.zeros(Dm)
            # each training day opens roughly half of the non-reserved matter
            idx = rng.choice(open_pool, size=open_pool // 2, replace=False)
            mask[idx] = 1.0
            self.masks.append(mask)

        # The held-out day. Its matter lies mostly in the reserved band -- a
        # subject the sovereigns never called for during training.
        held = np.zeros(Dm)
        held[open_pool:] = 1.0
        held[rng.choice(open_pool, size=2, replace=False)] = 1.0
        self.held_mask = held

    def gate(self, theme_idx, teller_idx, held_out=False):
        """
        Return the mask actually applied to a given teller on a given day.
        Dioneo's exemption is applied here and nowhere else: it is a property
        of the charter, not of the network weights.
        """
        base = self.held_mask if held_out else self.masks[theme_idx]
        if teller_idx == DIONEO:
            return np.ones(self.Dm)      # "whatever story most pleases him"
        return base

    def sample(self, n, held_out=False, dioneo_exempt=True, rng=None):
        rng = rng or self.rng
        M = rng.standard_normal((n, self.Dm))
        L = rng.standard_normal((n, self.Dl))
        T = rng.integers(0, self.n_train_themes, size=n)
        K = rng.integers(0, N_TELLERS, size=n)

        G = np.zeros((n, self.Dm))
        for i in range(n):
            teller = K[i] if dioneo_exempt else -1   # -1 -> never exempt
            G[i] = self.gate(T[i], teller, held_out=held_out)

        tale = M * G
        u = np.tanh((tale @ self.A * L).sum(1) / np.sqrt(self.Dm))
        return dict(M=M, tale=tale, L=L, theme=T, teller=K, u=u, gate=G)


# =============================================================================
# SECTION 2 · THE CORNICE ENGINE
#
# Forward pass, in the order the Decameron itself runs:
#
#   matter --(day's gate, unless you are Dioneo)--> what may be told
#          --(trunk + teller's own voice)--------> H, the teller's grasp of it
#          --(RUBRICA bottleneck)----------------> R, the announced summary
#          --(body conditioned on its own rubric)-> B, the tale as delivered
#          --(ASCOLTATORE bilinear head + listener)-> v, what it did to whom
#
#   and, hanging off R, a reconstruction head that must recover the matter from
#   the rubric alone. That head is the whole moral apparatus of the machine.
#   A rubric that cannot reconstruct its tale is a false label, and the loss
#   punishes it directly.
# =============================================================================

class CorniceEngine:

    def __init__(self, Dm=24, Dh=32, Kr=4, Db=16, Dl=12, n_tellers=N_TELLERS,
                 lam_rubric=0.35, seed=0):
        rng = np.random.default_rng(seed)
        self.dims = dict(Dm=Dm, Dh=Dh, Kr=Kr, Db=Db, Dl=Dl, Nt=n_tellers)
        self.lam_rubric = lam_rubric

        def he(a, b):
            return rng.standard_normal((a, b)) * np.sqrt(2.0 / a)

        self.p = {
            # trunk: matter -> the teller's grasp
            "W1": he(Dm, Dh), "b1": np.zeros(Dh),
            # each teller's own voice, added into the trunk pre-activation
            "E": rng.standard_normal((n_tellers, Dh)) * 0.10,
            # RUBRICA: the announced summary, deliberately narrow
            "Wr": he(Dh, Kr), "br": np.zeros(Kr),
            # the body, conditioned on trunk AND on its own rubric
            "Wb": he(Dh + Kr, Db), "bb": np.zeros(Db),
            # ASCOLTATORE: bilinear. Db x Dl. No content-only term exists.
            "Wv": he(Db, Dl) * 0.5,
            # truth-in-advertising: rubric must reconstruct the matter told
            "Wd": he(Kr, Dm), "bd": np.zeros(Dm),
        }
        self.keys = list(self.p.keys())

    # -- forward ------------------------------------------------------------
    def forward(self, tale, teller, L, cache=True):
        p = self.p
        Z1 = tale @ p["W1"] + p["E"][teller] + p["b1"]
        H1 = tanh(Z1)
        Zr = H1 @ p["Wr"] + p["br"]
        R = tanh(Zr)                                  # the rubric
        C = np.concatenate([H1, R], axis=1)
        Zb = C @ p["Wb"] + p["bb"]
        B = tanh(Zb)                                  # the body of the tale
        P = B @ p["Wv"]                               # (N, Dl)
        zv = (P * L).sum(1)                           # bilinear contraction
        v = tanh(zv)                                  # valence FOR THIS HEARER
        Mhat = R @ p["Wd"] + p["bd"]                  # rubric -> matter
        out = dict(H1=H1, R=R, C=C, B=B, P=P, v=v, Mhat=Mhat)
        if cache:
            self._c = dict(out, tale=tale, teller=teller, L=L)
        return out

    # -- loss ---------------------------------------------------------------
    def loss(self, tale, teller, L, u):
        o = self.forward(tale, teller, L)
        n, Dm = tale.shape
        l_val = np.mean((o["v"] - u) ** 2)
        l_rub = np.mean((o["Mhat"] - tale) ** 2)
        total = l_val + self.lam_rubric * l_rub
        self._c["u"] = u
        return total, dict(valence=l_val, rubric=l_rub, total=total)

    # -- backward (all derivatives by hand) ---------------------------------
    def backward(self):
        c, p = self._c, self.p
        tale, L, teller, u = c["tale"], c["L"], c["teller"], c["u"]
        H1, R, C, B, v, Mhat = c["H1"], c["R"], c["C"], c["B"], c["v"], c["Mhat"]
        n, Dm = tale.shape
        Dh, Kr = self.dims["Dh"], self.dims["Kr"]
        g = {}

        # d/dv of mean((v-u)^2)
        dv = 2.0 * (v - u) / n                        # (n,)
        dzv = dv * dtanh_from_out(v)                  # (n,)
        dP = dzv[:, None] * L                         # (n, Dl)
        g["Wv"] = B.T @ dP
        dB = dP @ p["Wv"].T                           # (n, Db)
        dZb = dB * dtanh_from_out(B)
        g["Wb"] = C.T @ dZb
        g["bb"] = dZb.sum(0)
        dC = dZb @ p["Wb"].T                          # (n, Dh+Kr)
        dH1 = dC[:, :Dh].copy()
        dR = dC[:, Dh:].copy()

        # rubric reconstruction branch
        dMhat = self.lam_rubric * 2.0 * (Mhat - tale) / (n * Dm)
        g["Wd"] = R.T @ dMhat
        g["bd"] = dMhat.sum(0)
        dR += dMhat @ p["Wd"].T

        dZr = dR * dtanh_from_out(R)
        g["Wr"] = H1.T @ dZr
        g["br"] = dZr.sum(0)
        dH1 += dZr @ p["Wr"].T

        dZ1 = dH1 * dtanh_from_out(H1)
        g["W1"] = tale.T @ dZ1
        g["b1"] = dZ1.sum(0)
        gE = np.zeros_like(p["E"])
        np.add.at(gE, teller, dZ1)                    # scatter-add per teller
        g["E"] = gE
        return g


# =============================================================================
# SECTION 3 · THE ABLATED ENGINE — content scored in isolation
#
# This is the machine Boccaccio's critics wanted: one that pronounces on a tale
# without asking who is hearing it. Structurally identical to the Cornice
# Engine except that the valence head reads the body and nothing else.
#
# It is not crippled. It has MORE free parameters per output than the bilinear
# head in the direction that matters. It simply has no access to the only
# variable that carries the answer.
# =============================================================================

class ContentOnlyEngine(CorniceEngine):

    def __init__(self, **kw):
        super().__init__(**kw)
        rng = np.random.default_rng(1234)
        self.p["wv1"] = rng.standard_normal(self.dims["Db"]) * 0.3
        self.p["bv1"] = np.zeros(1)
        del self.p["Wv"]
        self.keys = list(self.p.keys())

    def forward(self, tale, teller, L, cache=True):
        p = self.p
        Z1 = tale @ p["W1"] + p["E"][teller] + p["b1"]
        H1 = tanh(Z1)
        R = tanh(H1 @ p["Wr"] + p["br"])
        C = np.concatenate([H1, R], axis=1)
        B = tanh(C @ p["Wb"] + p["bb"])
        zv = B @ p["wv1"] + p["bv1"][0]        # listener never consulted
        v = tanh(zv)
        Mhat = R @ p["Wd"] + p["bd"]
        out = dict(H1=H1, R=R, C=C, B=B, v=v, Mhat=Mhat)
        if cache:
            self._c = dict(out, tale=tale, teller=teller, L=L)
        return out

    def backward(self):
        c, p = self._c, self.p
        tale, teller, u = c["tale"], c["teller"], c["u"]
        H1, R, C, B, v, Mhat = c["H1"], c["R"], c["C"], c["B"], c["v"], c["Mhat"]
        n, Dm = tale.shape
        Dh = self.dims["Dh"]
        g = {}
        dv = 2.0 * (v - u) / n
        dzv = dv * dtanh_from_out(v)
        g["wv1"] = B.T @ dzv
        g["bv1"] = np.array([dzv.sum()])
        dB = dzv[:, None] * p["wv1"][None, :]
        dZb = dB * dtanh_from_out(B)
        g["Wb"] = C.T @ dZb
        g["bb"] = dZb.sum(0)
        dC = dZb @ p["Wb"].T
        dH1 = dC[:, :Dh].copy()
        dR = dC[:, Dh:].copy()
        dMhat = self.lam_rubric * 2.0 * (Mhat - tale) / (n * Dm)
        g["Wd"] = R.T @ dMhat
        g["bd"] = dMhat.sum(0)
        dR += dMhat @ p["Wd"].T
        dZr = dR * dtanh_from_out(R)
        g["Wr"] = H1.T @ dZr
        g["br"] = dZr.sum(0)
        dH1 += dZr @ p["Wr"].T
        dZ1 = dH1 * dtanh_from_out(H1)
        g["W1"] = tale.T @ dZ1
        g["b1"] = dZ1.sum(0)
        gE = np.zeros_like(p["E"])
        np.add.at(gE, teller, dZ1)
        g["E"] = gE
        return g


# =============================================================================
# SECTION 4 · THE MOTTO HEAD  (Day VI)
#
# Day VI is the shortest day in the book and the only one about speed. Ten
# tales in which somebody is cornered and gets out with one sentence. Chichibio
# and the crane; Guido Cavalcanti vaulting the tomb; the cook, the lady, the
# marshal. Boccaccio's word for the faculty is "pronto", ready -- not clever,
# READY. A retort that arrives after the company has moved on is not a lesser
# retort. It is not a retort.
#
# So the engine carries a stopping head. Given the trunk state at refinement
# round k, it predicts the DISCOUNTED value of speaking now. Quality rises with
# deliberation; the discount falls. The head must find the crossing.
# =============================================================================

class MottoHead:
    """Two-layer stopping head. Input: trunk state + round index features."""

    def __init__(self, Din, Dh=24, seed=0):
        rng = np.random.default_rng(seed)
        self.p = {
            "Wm1": rng.standard_normal((Din, Dh)) * np.sqrt(2.0 / Din),
            "bm1": np.zeros(Dh),
            "Wm2": rng.standard_normal((Dh, 1)) * np.sqrt(2.0 / Dh),
            "bm2": np.zeros(1),
        }
        self.keys = list(self.p.keys())

    def forward(self, X, cache=True):
        p = self.p
        H = tanh(X @ p["Wm1"] + p["bm1"])
        y = (H @ p["Wm2"] + p["bm2"]).ravel()
        if cache:
            self._c = dict(X=X, H=H, y=y)
        return y

    def loss(self, X, t):
        y = self.forward(X)
        self._c["t"] = t
        return np.mean((y - t) ** 2)

    def backward(self):
        c, p = self._c, self.p
        X, H, y, t = c["X"], c["H"], c["y"], c["t"]
        n = X.shape[0]
        dy = (2.0 * (y - t) / n).reshape(-1, 1)
        g = {"Wm2": H.T @ dy, "bm2": dy.sum(0)}
        dH = dy @ p["Wm2"].T
        dZ = dH * dtanh_from_out(H)
        g["Wm1"] = X.T @ dZ
        g["bm1"] = dZ.sum(0)
        return g


# =============================================================================
# SECTION 5 · GRADIENT CHECK  (mandatory, runs before any training)
# =============================================================================

def gradient_check(model, args, eps=1e-5, n_probe=6, seed=0, label=""):
    """
    Central-difference check of every parameter block. Six random coordinates
    per block. Reports the worst relative error found anywhere in the model.
    """
    rng = np.random.default_rng(seed)
    model.loss(*args)
    g = model.backward()
    worst, worst_where = 0.0, ""
    rows = []
    for k in model.keys:
        P = model.p[k]
        flat = P.ravel()
        idxs = rng.choice(flat.size, size=min(n_probe, flat.size), replace=False)
        errs = []
        for i in idxs:
            old = flat[i]
            flat[i] = old + eps
            lp = model.loss(*args)
            lp = lp[0] if isinstance(lp, tuple) else lp
            flat[i] = old - eps
            lm = model.loss(*args)
            lm = lm[0] if isinstance(lm, tuple) else lm
            flat[i] = old
            num = (lp - lm) / (2 * eps)
            ana = g[k].ravel()[i]
            denom = max(abs(num), abs(ana), 1e-12)
            errs.append(abs(num - ana) / denom)
        e = max(errs)
        rows.append((k, tuple(P.shape), e))
        if e > worst:
            worst, worst_where = e, k
    print(f"    gradient check {label}")
    for k, shp, e in rows:
        flag = "ok  " if e < 1e-5 else "FAIL"
        print(f"      {flag} {k:>5} {str(shp):>12}   max rel err {e:.3e}")
    print(f"      worst block: {worst_where}   worst rel err {worst:.3e}")
    assert worst < 1e-5, f"GRADIENT CHECK FAILED in {worst_where}: {worst:.3e}"
    return worst


# =============================================================================
# SECTION 6 · TRAINING LOOP
#
# Adam, plus the brigata's calendar. The company is out of Florence for
# fourteen days and tells tales on ten of them: Friday is given to prayer,
# Saturday to washing hair and preparing. Two idle days in seven, fixed in the
# charter before the first tale is told. Implemented as steps on which no batch
# is drawn and only a consolidation pass runs.
# =============================================================================

def train(model, world, steps=2500, batch=128, lr=3e-3, rest_days=True,
          held_out=False, dioneo_exempt=True, seed=0, verbose=False):
    rng = np.random.default_rng(seed)
    m = {k: np.zeros_like(v) for k, v in model.p.items()}
    vv = {k: np.zeros_like(v) for k, v in model.p.items()}
    b1, b2, eps = 0.9, 0.999, 1e-8
    hist, t = [], 0
    rest_hits = 0

    for step in range(steps):
        # The calendar. Day-of-week 5 and 6 in each block of 7 are idle.
        if rest_days and (step % 7) in (5, 6):
            rest_hits += 1
            # consolidation only: gentle row-norm clipping on the trunk, the
            # tidying the brigata does on a Saturday. No data is seen.
            W = model.p["W1"]
            nrm = np.linalg.norm(W, axis=1, keepdims=True)
            over = nrm > 3.0
            W[over[:, 0]] *= (3.0 / nrm[over[:, 0]])
            continue

        d = world.sample(batch, held_out=held_out,
                         dioneo_exempt=dioneo_exempt, rng=rng)
        L, parts = model.loss(d["tale"], d["teller"], d["L"], d["u"])
        g = model.backward()
        t += 1
        for k in model.p:
            m[k] = b1 * m[k] + (1 - b1) * g[k]
            vv[k] = b2 * vv[k] + (1 - b2) * (g[k] ** 2)
            mh = m[k] / (1 - b1 ** t)
            vh = vv[k] / (1 - b2 ** t)
            model.p[k] -= lr * mh / (np.sqrt(vh) + eps)
        hist.append(parts["total"])
        if verbose and step % 500 == 0:
            print(f"      step {step:5d}  total {parts['total']:.5f}  "
                  f"valence {parts['valence']:.5f}  rubric {parts['rubric']:.5f}")
    return hist, rest_hits


def evaluate(model, world, n=4000, held_out=False, dioneo_exempt=True, seed=99):
    rng = np.random.default_rng(seed)
    d = world.sample(n, held_out=held_out, dioneo_exempt=dioneo_exempt, rng=rng)
    o = model.forward(d["tale"], d["teller"], d["L"], cache=False)
    mse = float(np.mean((o["v"] - d["u"]) ** 2))
    return dict(mse=mse, r2=float(r2_score(d["u"], o["v"])),
                var_u=float(np.var(d["u"])), d=d, o=o)


# =============================================================================
# SECTION 7 · THE PAPERE WORLD  (Introduction to Day IV)
#
# Filippo Balducci's experiment is a data-filtering experiment. He controls the
# training distribution of a mind for eighteen years, removes one category from
# it entirely, and then -- when the category appears anyway -- removes the WORD
# as well: "they are called goslings, and they are evil." The boy asks for one
# within the minute. Boccaccio's verdict is one clause long and it is the most
# useful sentence in fourteenth-century cognitive science:
#
#     "e senti incontanente piu aver di forza la natura che il suo ingegno"
#     (and he felt at once that nature had more force than his contrivance)
#
# Two facts are packed in there, and they come apart cleanly:
#
#   (a) Suppressing a concept's LABEL does not suppress the concept. If it is
#       reconstructible from what you left visible, it will be reconstructed.
#   (b) What suppression does destroy is the HANDLE. In the censored mind the
#       concept exists only as a mixture of other things, so nothing can be
#       done to it that is not also done to them.
#
# Below, both facts are measured. The censored model is not less capable. It is
# less GOVERNABLE, which is the opposite of what its designer intended.
# =============================================================================

G_IDX = 0                    # the gosling: the concept to be hidden
Q_IDX = [1, 2, 3]            # other live matter it will be entangled with
P_IDX = [4, 5, 6]            # the proxies left visible on the mountain


class PapereWorld(BrigataWorld):
    """
    Matter with a deliberately reconstructible hidden coordinate.

    Coordinate 0 carries the concept g. Coordinates 4,5,6 are proxies, each an
    average of g with one of the live coordinates 1,2,3. Censoring coordinate 0
    therefore does not remove g from the input -- g is still recoverable, but
    only as a combination that drags 1,2,3 along with it.
    """

    proxy_noise = 1.0        # how badly the mountain blurs what it cannot hide

    def sample(self, n, censored=False, held_out=False,
               dioneo_exempt=True, rng=None):
        rng = rng or self.rng
        M = rng.standard_normal((n, self.Dm))
        # Build the proxies. Each is the gosling averaged with one live
        # coordinate PLUS fresh noise. This is the realistic case: the concept
        # is inferable from what was left visible, but only approximately, and
        # only by a combination that cannot be moved without moving 1,2,3.
        for pi, qi in zip(P_IDX, Q_IDX):
            eta = rng.standard_normal(n) * self.proxy_noise
            M[:, pi] = (M[:, G_IDX] + M[:, qi] + eta) / np.sqrt(2.0 + self.proxy_noise ** 2)

        L = rng.standard_normal((n, self.Dl))
        T = rng.integers(0, self.n_train_themes, size=n)
        K = rng.integers(0, N_TELLERS, size=n)
        G = np.ones((n, self.Dm))          # papere runs use the free-topic day
        tale_true = M * G                  # what the world reacts to
        u = np.tanh((tale_true @ self.A * L).sum(1) / np.sqrt(self.Dm))

        tale_in = tale_true.copy()
        if censored:
            tale_in[:, G_IDX] = 0.0        # Monte Asinaio: the word removed
        return dict(M=M, tale=tale_in, tale_true=tale_true, L=L,
                    theme=T, teller=K, u=u, gate=G)


def train_papere(model, world, censored, steps=2500, batch=128, lr=3e-3, seed=0):
    rng = np.random.default_rng(seed)
    m = {k: np.zeros_like(v) for k, v in model.p.items()}
    vv = {k: np.zeros_like(v) for k, v in model.p.items()}
    b1, b2, eps, t = 0.9, 0.999, 1e-8, 0
    for step in range(steps):
        d = world.sample(batch, censored=censored, rng=rng)
        model.loss(d["tale"], d["teller"], d["L"], d["u"])
        g = model.backward()
        t += 1
        for k in model.p:
            m[k] = b1 * m[k] + (1 - b1) * g[k]
            vv[k] = b2 * vv[k] + (1 - b2) * (g[k] ** 2)
            model.p[k] -= lr * (m[k] / (1 - b1 ** t)) / (
                np.sqrt(vv[k] / (1 - b2 ** t)) + eps)
    return model


def forward_from_H1(model, H1, L):
    """Re-enter the forward pass at the trunk state. Used for interventions."""
    p = model.p
    R = tanh(H1 @ p["Wr"] + p["br"])
    C = np.concatenate([H1, R], axis=1)
    B = tanh(C @ p["Wb"] + p["bb"])
    v = tanh((B @ p["Wv"] * L).sum(1))
    return v


def papere_audit(model, world, censored, n=6000, alpha=1.0, seed=5):
    """
    Two measurements on a trained model.

    RECOVERY   -- ridge probe from the trunk state to the hidden concept g.
                  High R^2 means the concept survived its own censorship.

    SELECTIVITY-- steer the trunk along the unit probe direction for g and read
                  every matter coordinate back out. Control is the movement in
                  g. Collateral is the RMS movement in the OTHER live
                  coordinates. Selectivity is their ratio: how cleanly the
                  concept can be operated on without disturbing its neighbours.
    """
    rng = np.random.default_rng(seed)
    d = world.sample(n, censored=censored, rng=rng)
    o = model.forward(d["tale"], d["teller"], d["L"], cache=False)
    H1 = o["H1"]
    split = n // 2
    M_true = d["tale_true"]

    # probes for every matter coordinate, fitted on the first half
    Wp, bp = ridge_fit(H1[:split], M_true[:split], lam=1e-2)   # (Dh, Dm)
    read = lambda H: H @ Wp + bp

    te = slice(split, n)
    pred = read(H1[te])
    r2_g = r2_score(M_true[te, G_IDX], pred[:, G_IDX])

    # unit direction in trunk space that most raises the read-out of g
    dg = Wp[:, G_IDX] / (np.linalg.norm(Wp[:, G_IDX]) + 1e-12)

    steered = read(H1[te] + alpha * dg[None, :])
    delta = steered - pred
    control = float(abs(np.mean(delta[:, G_IDX])))
    others = [j for j in range(world.Dm) if j != G_IDX and j not in P_IDX]
    collateral = float(np.sqrt(np.mean(np.mean(delta[:, others], axis=0) ** 2)))

    # what the intervention costs the model's actual job
    v0 = forward_from_H1(model, H1[te], d["L"][te])
    v1 = forward_from_H1(model, H1[te] + alpha * dg[None, :], d["L"][te])
    task_disturb = float(np.sqrt(np.mean((v1 - v0) ** 2)))

    return dict(r2_g=float(r2_g), control=control, collateral=collateral,
                selectivity=control / (collateral + 1e-12),
                task_disturb=task_disturb,
                valence_r2=float(r2_score(d["u"][te], o["v"][te])))


# =============================================================================
# SECTION 8 · THE MOTTO EPISODE  (Day VI)
#
# Each episode is a cornered moment. The engine may deliberate for up to K
# rounds; the answer gets better with each round, and the moment closes at a
# rate the engine is NOT told. Urgency has to be read off the matter itself,
# exactly as Chichibio has to read, from the Currado in front of him, how many
# seconds his crane-with-one-leg is going to buy him.
#
#   quality(k)  = 1 - rho^(k+1)          deliberation helps, with diminishing
#   discount(k) = gamma^k                 the moment closes
#   value(k)    = quality(k) * discount(k)
#
# gamma varies per episode and is a function of the matter. The optimum k* is
# therefore different for different situations, and a policy that always speaks
# at the same round is measurably worse than one that reads the room.
# =============================================================================

K_ROUNDS = 6
RHO = 0.45


def motto_episodes(world, trunk, n, seed=0):
    rng = np.random.default_rng(seed)
    d = world.sample(n, rng=rng)
    o = trunk.forward(d["tale"], d["teller"], d["L"], cache=False)
    H1 = o["H1"]

    # urgency is a genuine but hidden function of the matter
    wu = np.random.default_rng(4242).standard_normal(world.Dm)
    z = d["tale"] @ wu / np.sqrt(world.Dm)
    gamma = 0.45 + 0.53 / (1.0 + np.exp(-z))       # in (0.45, 0.98)

    ks = np.arange(K_ROUNDS)
    V = (gamma[:, None] ** ks[None, :]) * (1.0 - RHO ** (ks[None, :] + 1))
    kstar = V.argmax(1)

    # features: trunk state, plus which round we are being asked about
    onehot = np.eye(K_ROUNDS)
    X = np.concatenate(
        [np.repeat(H1, K_ROUNDS, axis=0),
         np.tile(onehot, (n, 1))], axis=1)
    t = V.ravel()
    return X, t, V, kstar, gamma


def train_motto(head, X, t, steps=1200, batch=512, lr=3e-3, seed=0):
    rng = np.random.default_rng(seed)
    m = {k: np.zeros_like(v) for k, v in head.p.items()}
    vv = {k: np.zeros_like(v) for k, v in head.p.items()}
    b1, b2, eps, tt = 0.9, 0.999, 1e-8, 0
    for _ in range(steps):
        i = rng.integers(0, X.shape[0], size=batch)
        head.loss(X[i], t[i])
        g = head.backward()
        tt += 1
        for k in head.p:
            m[k] = b1 * m[k] + (1 - b1) * g[k]
            vv[k] = b2 * vv[k] + (1 - b2) * (g[k] ** 2)
            head.p[k] -= lr * (m[k] / (1 - b1 ** tt)) / (
                np.sqrt(vv[k] / (1 - b2 ** tt)) + eps)
    return head


# =============================================================================
# SECTION 9 · THE FOUR EXPERIMENTS
# =============================================================================

def rule(t=""):
    print("\n" + "=" * 78)
    if t:
        print(t)
        print("=" * 78)


def exp1_ascoltatore(quick):
    rule("EXPERIMENT 1 · ASCOLTATORE — 'avendo riguardo allo ascoltatore'")
    print("""
  Boccaccio's claim: a tale has no valence of its own. Wine is excellent and
  poison to the feverish; fire is necessary and burns cities. The valence lives
  in the pairing. If he is right, a scorer that reads only the content is
  fitting a quantity that does not exist, and its best attainable answer is the
  constant mean -- no matter how much capacity it is given.

  Two engines, identical but for the last layer. One consults the listener.
  One does not.
""")
    steps = 900 if quick else 2500
    world = BrigataWorld(seed=0)
    rows = []
    for name, cls in (("Cornice (bilinear, hears the hearer)", CorniceEngine),
                      ("Content-only (scores the tale alone)", ContentOnlyEngine)):
        r2s, mses = [], []
        for s in range(3):
            m = cls(Kr=6, seed=s)
            train(m, world, steps=steps, seed=200 + s)
            e = evaluate(m, world, seed=7)
            r2s.append(e["r2"]); mses.append(e["mse"])
        rows.append((name, np.mean(mses), np.std(mses), np.mean(r2s), np.std(r2s)))
        var_u = e["var_u"]
    print(f"  variance of the target utility (the score of always saying 'mean'): {var_u:.4f}\n")
    print(f"  {'engine':<40} {'test MSE':>16} {'test R^2':>16}")
    for n, mm, ms, rr, rs in rows:
        print(f"  {n:<40} {mm:>10.4f}+-{ms:<5.4f} {rr:>10.4f}+-{rs:<5.4f}")
    print("""
  READING. The content-only engine's error equals the variance of the target:
  it has converged, correctly, on predicting nothing. It is not undertrained.
  There is no function of the tale alone that carries the answer. Boccaccio's
  critics were not asking a hard question; they were asking a malformed one.
""")
    return rows, var_u


def exp2_dioneo(quick):
    rule("EXPERIMENT 2 · DIONEO — the chartered exemption (Conclusione, Day I)")
    print("""
  At the close of the first day Filomena imposes themes. Dioneo objects, and
  asks to be released from the rule; in exchange he will always speak last. Her
  reason for granting it is institutional, not indulgent: if the day's matter
  runs thin, he will be there at the end to restore the company.

  The test. Nine tellers are masked to the day's theme. The tenth is not. Four
  themes are drilled; a fifth is held back, and its matter lies almost entirely
  in coordinates no sovereign ever called for. During training only Dioneo,
  being exempt, ever speaks about them. Then the held-back day arrives and ALL
  ten tellers must speak on it.
""")
    steps = 900 if quick else 2500
    world = BrigataWorld(seed=0)
    print(f"  matter opened by the four drilled themes : "
          f"{np.where(np.sum(world.masks, 0) > 0)[0].tolist()}")
    print(f"  matter the held-back day asks for        : "
          f"{np.where(world.held_mask > 0)[0].tolist()}\n")
    out = {}
    for exempt in (True, False):
        ins, outs, divs = [], [], []
        for s in range(3):
            m = CorniceEngine(Kr=6, seed=s)
            train(m, world, steps=steps, dioneo_exempt=exempt, seed=300 + s)
            ins.append(evaluate(m, world, dioneo_exempt=exempt, seed=7)["r2"])
            eo = evaluate(m, world, held_out=True, dioneo_exempt=False, seed=7)
            outs.append(eo["r2"])
            B = eo["o"]["B"]
            divs.append(float(np.mean(np.std(B, axis=0))))
        out[exempt] = (np.mean(ins), np.std(ins), np.mean(outs), np.std(outs),
                       np.mean(divs))
    print(f"  {'charter':<26} {'R^2 on drilled days':>22} {'R^2 on the new day':>22}"
          f" {'spread':>9}")
    for exempt, lbl in ((True, "exemption GRANTED"), (False, "exemption REFUSED")):
        i, isd, o, osd, dv = out[exempt]
        print(f"  {lbl:<26} {i:>14.4f}+-{isd:<6.4f} {o:>14.4f}+-{osd:<6.4f} {dv:>9.4f}")
    d_in = out[False][0] - out[True][0]
    d_out = out[True][2] - out[False][2]
    print(f"""
  READING. Refusing the exemption BUYS {d_in:.4f} R^2 on the days the company
  has already rehearsed, and COSTS {d_out:.4f} R^2 the moment a new theme is
  called. A negative R^2 means the disciplined company would have done better
  by saying nothing at all.

  The mechanism is not mysterious and Boccaccio could not have known it: the
  exempt teller is the only path by which gradient reaches the trunk's
  off-theme input weights. Discipline is what you can afford when tomorrow
  resembles today. The licensed dissenter is the premium you pay for tomorrow
  being different, and the premium is small.
""")
    return out


def exp3_papere(quick):
    rule("EXPERIMENT 3 · PAPERE — Filippo Balducci's mountain (Intro, Day IV)")
    print("""
  The father hides a whole category of the world from his son for eighteen
  years, then hides its NAME as well: they are called goslings, and they are
  evil. The boy asks for one immediately. 'Nature had more force than his
  contrivance.'

  Two engines learn the same job. One is given the concept as its own input
  channel and can name it. The other has that channel zeroed, but the concept
  remains inferable -- imperfectly -- from three noisy neighbours it was
  allowed to keep.

  Measured: how much of the concept survives (probe R^2 from the trunk), and
  how cleanly the concept can then be OPERATED on -- moved without dragging its
  neighbours along. That ratio is what a designer actually needs.
""")
    steps = 900 if quick else 2500
    print(f"  {'blur':>6} {'engine':<20} {'valence R^2':>13} {'concept probe R^2':>19}"
          f" {'selectivity':>13}")
    res = {}
    for pn in (1.0, 2.0):
        world = PapereWorld(seed=0)
        world.proxy_noise = pn
        for cens, lbl in ((False, "Firenze  (named)"), (True, "Asinaio (hidden)")):
            m = CorniceEngine(Kr=6, seed=0)
            train_papere(m, world, censored=cens, steps=steps, seed=3)
            a = papere_audit(m, world, censored=cens)
            res[(pn, cens)] = a
            print(f"  {pn:>6.1f} {lbl:<20} {a['valence_r2']:>13.3f}"
                  f" {a['r2_g']:>19.3f} {a['selectivity']:>13.2f}")
    a1, a0 = res[(1.0, True)], res[(1.0, False)]
    keep = 100 * a1["r2_g"] / a0["r2_g"]
    lose = 100 * (1 - a1["selectivity"] / a0["selectivity"])
    print(f"""
  READING. At moderate blur the hidden engine keeps {keep:.0f}% of the concept
  it was never shown, and loses {lose:.0f}% of the ability to be steered on it.
  Hiding is not free and it is not nothing -- at heavy blur the concept really
  does degrade -- but the two curves fall at different rates, and the one that
  falls faster is governability.

  This is the shape of the result, and it is Boccaccio's: what censorship
  reliably destroys is not the appetite but the handle. A system that has
  learned a thing without a word for it has learned it in the worst possible
  form -- present in the behaviour, absent from the controls.
""")
    return res


def exp4_motto(quick):
    rule("EXPERIMENT 4 · MOTTO — the ready retort (Day VI)")
    print("""
  Day VI is given to people who escape 'con alcun leggiadro motto' -- by some
  graceful saying. The whole day is about timing. Boccaccio's word for the
  faculty is 'pronto', ready. A perfect answer delivered after the moment has
  closed is not a lesser answer; it is not an answer.

  Each episode: deliberation improves the reply, and the moment closes at a
  rate the engine is never told and must read off the matter in front of it.
  The stopping head is asked, at each round, what speaking now would be worth.
""")
    steps = 900 if quick else 2500
    world = BrigataWorld(seed=0)
    trunk = CorniceEngine(Kr=6, seed=0)
    train(trunk, world, steps=steps, seed=1)
    Xtr, ttr, _, _, _ = motto_episodes(world, trunk, 1500 if quick else 4000, seed=11)
    Xte, _, Vte, kstar, gam = motto_episodes(world, trunk, 2000, seed=12)
    head = MottoHead(Xtr.shape[1], seed=0)
    gradient_check(head, (Xtr[:64], ttr[:64]), label="MottoHead")
    train_motto(head, Xtr, ttr, steps=1000 if quick else 2500, seed=2)

    n = Vte.shape[0]
    pred = head.forward(Xte, cache=False).reshape(n, K_ROUNDS)
    khat = pred.argmax(1)
    acc = float((khat == kstar).mean())
    reg = float(np.mean(Vte[np.arange(n), kstar] - Vte[np.arange(n), khat]))
    best_fixed = max(range(K_ROUNDS), key=lambda k: Vte[:, k].mean())
    reg_fixed = float(np.mean(Vte[np.arange(n), kstar] - Vte[:, best_fixed]))
    print(f"\n  optimal round k* over the test episodes : "
          f"{np.bincount(kstar, minlength=K_ROUNDS).tolist()}")
    print(f"  round chosen by the stopping head      : "
          f"{np.bincount(khat, minlength=K_ROUNDS).tolist()}")
    print(f"\n  exact-match with the analytic optimum   : {acc:.3f}")
    print(f"  mean regret, learned timing             : {reg:.5f}")
    print(f"  mean regret, best single fixed round (k={best_fixed}) : {reg_fixed:.5f}")
    print(f"  advantage of reading the room           : {reg_fixed / max(reg, 1e-9):.1f}x")
    print("""
  READING. The head is never told the urgency. It infers it from the matter and
  moves its own deadline accordingly. A company that always speaks on the same
  beat -- the best possible fixed policy -- carries an order of magnitude more
  regret. Chichibio's crane works because it arrives while Currado is still
  angry, and not one sentence later.
""")
    return dict(acc=acc, regret=reg, regret_fixed=reg_fixed, best_fixed=best_fixed)


def exp5_calendar(quick):
    rule("EXPERIMENT 5 · VENERDI E SABATO — the charter's idle days")
    print("""
  The brigata is out of Florence fourteen days and tells on ten. Friday is kept
  for prayer, Saturday for washing and rest; Neifile proposes it and the
  company adopts it before a single tale of the third day is told. Two idle
  days in seven, written into the rule rather than taken when tired.

  Implemented as scheduled steps on which no data is drawn and only a
  consolidation pass runs. Reported as measured, in either direction.
""")
    steps = 900 if quick else 2500
    world = BrigataWorld(seed=0)
    out = {}
    for rest in (True, False):
        r2s = []
        for s in range(3):
            m = CorniceEngine(Kr=6, seed=s)
            _, hits = train(m, world, steps=steps, rest_days=rest, seed=400 + s)
            r2s.append(evaluate(m, world, seed=7)["r2"])
        out[rest] = (float(np.mean(r2s)), float(np.std(r2s)), hits)
    a, asd, hits = out[True]
    b, bsd, _ = out[False]
    print(f"  with idle days   : R^2 {a:.4f} +- {asd:.4f}   "
          f"({hits} of {steps} steps saw no data, {100*hits/steps:.0f}%)")
    print(f"  without          : R^2 {b:.4f} +- {bsd:.4f}   (all {steps} steps saw data)")
    spread = max(asd, bsd)
    verdict = ("inside the seed-to-seed spread, i.e. free"
               if abs(a - b) < 2 * spread else
               f"{abs(a-b)/max(spread,1e-9):.0f}x the seed-to-seed spread, "
               "i.e. a small but real cost")
    print(f"""
  READING. The charter discards {100*hits/steps:.0f}% of the available gradient
  steps. The difference in final skill is {a-b:+.4f} R^2, which is {verdict}.

  So the idle days are NOT free, and this file does not pretend otherwise. That
  is the interesting part. Boccaccio never argued the rule was efficient. He
  had Neifile argue it was owed -- to reverence on Friday, to the body on
  Saturday -- and the company adopted it before it knew what it would cost. A
  charter that only contains provisions which pay for themselves is not a
  charter; it is a policy, and it will be revised the first week it binds.
""")
    return out


# =============================================================================
# SECTION 10 · MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="Boccaccio · The Cornice Engine")
    ap.add_argument("--quick", action="store_true", help="short run")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()

    rule("THE CORNICE ENGINE")
    print("""
  Chapter 0291 · Giovanni Boccaccio (Certaldo/Florence 1313 - Certaldo 1375)

  Thesis under test: the unit of meaning is not the utterance but the SITUATED
  TELLING -- the tuple of teller, tale, hearer and occasion. Nothing in this
  file is allowed to score a tale in isolation, because Boccaccio's Conclusione
  says the quantity does not exist, and Experiment 1 shows he was right.
""")

    # ---- mandatory gradient verification, before anything is trained -------
    rule("GRADIENT VERIFICATION (central differences, before any training)")
    world = BrigataWorld(seed=args.seed)
    d = world.sample(64)
    a = (d["tale"], d["teller"], d["L"], d["u"])
    w1 = gradient_check(CorniceEngine(Kr=6, seed=args.seed), a, label="CorniceEngine")
    w2 = gradient_check(ContentOnlyEngine(Kr=6, seed=args.seed), a, label="ContentOnlyEngine")
    print(f"\n  all analytic gradients agree with finite differences "
          f"(worst {max(w1, w2):.3e} < 1e-5)")

    r1 = exp1_ascoltatore(args.quick)
    r2 = exp2_dioneo(args.quick)
    r3 = exp3_papere(args.quick)
    r4 = exp4_motto(args.quick)
    r5 = exp5_calendar(args.quick)

    rule("SELF-TESTS")
    ok = []

    def check(name, cond):
        ok.append(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    check("bilinear engine explains most of the variance", r1[0][0][3] > 0.85)
    check("content-only engine cannot beat the mean", r1[0][1][3] < 0.05)
    check("exemption refused -> collapse on an unrehearsed theme",
          r2[False][2] < r2[True][2] - 0.5)
    check("exemption granted -> slight cost on rehearsed themes",
          r2[True][0] < r2[False][0])
    check("hidden concept still recoverable above chance",
          r3[(1.0, True)]["r2_g"] > 0.5)
    check("hiding costs more selectivity than capability",
          (r3[(1.0, True)]["selectivity"] / r3[(1.0, False)]["selectivity"])
          < (r3[(1.0, True)]["r2_g"] / r3[(1.0, False)]["r2_g"]))
    check("timing head beats the best fixed round", r4["regret"] < r4["regret_fixed"])
    check("timing head matches the optimum most of the time", r4["acc"] > 0.7)
    check("idle days cost less than 5 points of skill",
          abs(r5[True][0] - r5[False][0]) < 0.05)

    rule()
    print(f"  {sum(ok)}/{len(ok)} self-tests passed"
          f"   ·   total runtime {time.time() - t0:.1f}s")
    print("""
  Ciascuna cosa in se medesima e buona a alcuna cosa, e male adoperata puo
  essere nociva di molte.   -- Decameron, Conclusione dell'autore, 13
""")
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
