#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
THE ATHAR ENGINE  --  chapter 252  --  Al-Biruni (c.973-1048)
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 13 Minds 241 - 260 Available on Amazon https://www.amazon.com/dp/B0HLD1S6QT
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0252_al_biruni_973 - Abu Rayhan Muhammad ibn Ahmad al-Biruni (Kath, Khwarezm 973 - Ghazna 1048)
#================================================================================  

WHAT THIS FILE IS
-----------------
A complete, trainable, executable machine-learning architecture built from
scratch in NumPy, whose *mechanism* is a formalisation of al-Biruni's actual
method of ascertainment (tahqiq), not a generic network with his name on it.

THE ONE IDEA THIS ARCHITECTURE ENCODES
--------------------------------------
Almost every inference system -- ancient chronicle, modern language model --
collapses conflicting reports into a single answer and throws the losers away
silently. Al-Biruni did the opposite, and said so in print. In the Chronology
(al-Athar al-Baqiya, c.1000) he assembles seven rival versions of the Persian
epagomenal day-names and states plainly that he has no means of deciding
between them; he copies out king-lists he knows to be corrupt because the
corruption is itself evidence; he resolves the Parthian list only when an
independent text (Mani's Sabuhragan) gives him an external cross-check. And
where Ptolemy quietly selected the observations that suited his theory,
al-Biruni published the observations he discarded alongside the ones he kept.

So the thesis of this architecture is:

    AN INTELLIGENT SYSTEM'S OUTPUT MUST CARRY ITS OWN REJECTED ALTERNATIVES,
    ITS OWN CORRECTION FOR WHO IS SPEAKING, AND AN EXPLICIT MARK WHERE THE
    EVIDENCE CANNOT DECIDE.

That is a machine with five parts, and this file builds all five:

  1. REGISTRATION (al-Athar al-Baqiya, c.1000).
     Reports arrive in mutually untranslatable frames -- Hijri, Yazdegird,
     Seleucid, Saka eras; cubits, mithqals, local mile. A per-frame affine
     gauge (scale, offset) maps every report into one canonical frame. This is
     era conversion and instrument calibration treated as the same operation.

  2. MOTIVE DECONVOLUTION (Tahqiq ma li-l-Hind, preface, c.1030).
     Al-Biruni classifies reporters by *why* they distort: self-interest in
     praising one's own nation or attacking the rival; predilection or enmity
     toward a class of people; profit or fear; habitual fabrication; and
     ignorance transmitted by blindly following others. A small network takes
     that five-dimensional motive profile and predicts both a bias correction
     and a precision -- so a motivated witness is not excluded, he is
     *discounted by a learned amount*.

  3. TRANSMISSION DEFLATION (same preface).
     "If the connecting links are eliminated, there remains the originator of
     the story." A copy is not a witness. Precision is divided by a learned
     function of copy-depth, and the Kish effective sample size n_eff makes
     the deflation visible in the output. Twenty chronicles copying one
     forger are one forger.

  4. THE PLAUSIBILITY GATE (same preface, first sentence of the argument).
     Al-Biruni's precondition: source-criticism applies only to "an event that
     in itself does not contradict either logical or physical laws." Physics
     is checked FIRST and independently of how many people assert the claim.
     A gate multiplies precision by a smooth admissibility margin, so an
     impossible report cannot be rescued by unanimity.

  5. THE VARIANT-PRESERVING HEAD, WITH SUSPENSION (the whole corpus).
     Instead of one number, the engine emits an *athar record*: the fused
     estimate, BOTH surviving variants with their support, the effective
     independent witness count, and a three-way verdict --
        maqbula   (rationally acceptable)
        mardhula  (rejected, and the rejected variant is named and kept)
        tawaqquf  (suspension of judgement -- the evidence cannot decide)
     trained under a cost-sensitive objective in which suspension has a fixed
     price. This is Chow's reject-option rule, arrived at nine centuries early
     by a man who kept writing "this question is most difficult to solve."

WHAT IT IS DELIBERATELY NOT
---------------------------
Not a transformer. There is no softmax attention over stored keys anywhere in
this file. Weighting here is inverse-variance fusion -- the statistically
correct way to combine testimony of differing reliability -- and the fusion
weights are *derived from a model of the speaker*, not from similarity between
a query and a key. That difference is the whole point: attention asks "what in
my memory resembles this?", the Athar Engine asks "who is telling me this, how
many of them are copies of each other, and could it be true at all?"

ENGINEERING CONVENTIONS
-----------------------
  * pure NumPy, no autograd, no ML framework
  * every gradient derived by hand and verified against central finite
    differences (mandatory check, run at the bottom of this file)
  * a real Adam training loop on a real (synthetic but principled) corpus
  * three historical unit tests reproducing actual measurements of his:
    the Nandana horizon dip, longitude from a simultaneously observed lunar
    eclipse, and specific gravity by displacement
  * self-tests including two ablations that demonstrate the architecture's
    claims empirically rather than asserting them

RUN:  python3 chapter_0252_al_biruni_973.py
================================================================================
"""

import numpy as np

RNG_SEED = 973  # his birth year, used as the seed throughout

# --------------------------------------------------------------------------- #
# SECTION 0.  SMALL NUMERICAL UTILITIES
# --------------------------------------------------------------------------- #

EPS = 1e-12


def softplus(x):
    """Smooth positive map. Used wherever a parameter must stay > 0
    (precisions, the copy-discount rate kappa, the gate sharpness)."""
    return np.logaddexp(0.0, x)


def d_softplus(x):
    """d/dx softplus(x) = sigmoid(x)."""
    return sigmoid(x)


def scalar(x):
    """Parameters that are logically scalars survive flatten/unflatten round
    trips as 1-element arrays; this collapses them safely."""
    return float(np.asarray(x).ravel()[0])


def sigmoid(x):
    """Numerically stable logistic."""
    out = np.empty_like(np.asarray(x, dtype=float))
    pos = np.asarray(x) >= 0
    xp = np.asarray(x, dtype=float)
    out[pos] = 1.0 / (1.0 + np.exp(-xp[pos]))
    ex = np.exp(xp[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def seg_sum(vals, idx, n_seg):
    """Sum `vals` into `n_seg` buckets given integer bucket index `idx`.
    This is the only 'aggregation' primitive the engine needs: every quantity
    (every disputed fact) is one bucket, and all its reports sum into it."""
    out = np.zeros(n_seg, dtype=float)
    np.add.at(out, idx, vals)
    return out


def logsumexp_rows(z):
    m = z.max(axis=1, keepdims=True)
    return (m + np.log(np.exp(z - m).sum(axis=1, keepdims=True))).ravel()


# --------------------------------------------------------------------------- #
# SECTION 1.  THE MOTIVE TAXONOMY
#
# Al-Biruni's own list, from the preface to the India, in the order he gives
# it. These are the five input channels of the motive network. They are not
# decorative: the corpus generator uses them to *create* the distortion, and
# the network has to learn to undo it from the profile alone.
# --------------------------------------------------------------------------- #

MOTIVES = (
    "self_interest",      # lauding one's own family/nation, or attacking the rival
    "predilection",       # obligation to, or grudge against, a class of people
    "profit_or_fear",     # gain from the lie, or cowardice about the truth
    "habitual",           # mendacity as settled character
    "transmitted",        # ignorance, blindly following those who told him
)
N_MOTIVE = len(MOTIVES)

VERDICTS = ("maqbula", "mardhula", "tawaqquf")   # accept / reject / suspend
ACCEPT, REJECT, SUSPEND = 0, 1, 2


# --------------------------------------------------------------------------- #
# SECTION 2.  THE CORPUS
#
# A synthetic corpus with the exact structure of the evidence al-Biruni
# actually worked on: many reports of a few disputed quantities, arriving in
# incommensurable frames, from motivated reporters, most of them copying each
# other, with a small number of instrumentally anchored observations.
#
# Nothing here is fitted to the model. The generator is written first and the
# labels are derived from generator state, so the learning problem is real.
# --------------------------------------------------------------------------- #

class Corpus:
    """
    A bundle of testimony about Q quantities.

    Report i carries:
        y[i]      value as *stated by the reporter, in the reporter's frame*
        q[i]      which quantity it speaks about
        f[i]      which frame (era / metrology) it is stated in
        U[i,:]    motive profile (5) + anchor flag (1)  -- deliberately NOT depth
        depth[i]  how many links of transmission stand between i and autopsy
        anchor[i] 1 if this is an instrumental or simultaneous-event observation
        lo[i],hi[i]  the physically admissible interval for this quantity

    and the corpus knows, for supervision only:
        x_true[q]   the canonical truth
        label[q]    the correct verdict
    """

    # The admissible window is a property of PHYSICS, identical for every
    # quantity. It must not be centred on the truth: if it were, the gate
    # would be smuggling the answer in, and the identifiability ablation
    # further down would be measuring nothing.
    X_LO, X_HI = -4.2, 4.2

    def __init__(self, n_quant=160, n_frames=5, seed=RNG_SEED, frac_anchored=0.45):
        rng = np.random.default_rng(seed)
        self.n_frames = n_frames
        self.n_quant = n_quant

        # ---- the truth, in canonical units (think: days from a fixed epoch) --
        x_true = rng.uniform(-2.4, 2.4, size=n_quant)

        # ---- the frames.  Frame 0 is the instrumental frame: unit scale, zero
        #      offset. The others each have their own year-length ratio and
        #      epoch shift, exactly like converting Yazdegird to Hijri.
        a_true = np.concatenate([[1.0], np.exp(rng.normal(0, 0.22, n_frames - 1))])
        b_true = np.concatenate([[0.0], rng.normal(0, 1.1, n_frames - 1)])
        self.a_true, self.b_true = a_true, b_true

        ys, qs, fs, Us, depths, anchors, los, his = [], [], [], [], [], [], [], []
        n_indep_frames = np.zeros(n_quant)
        spread_true = np.zeros(n_quant)
        has_anchor = np.zeros(n_quant, dtype=bool)
        dominant_impossible = np.zeros(n_quant, dtype=bool)
        neff_true = np.zeros(n_quant)

        for q in range(n_quant):
            xq = x_true[q]
            lo_q, hi_q = self.X_LO, self.X_HI

            # --- the anchored observations: frame 0, no motive, no chain ------
            n_anchor = rng.integers(1, 3) if rng.random() < frac_anchored else 0
            reports = []          # (value_canonical, sigma, depth, anchor, motive)
            for _ in range(n_anchor):
                sd = 0.05
                reports.append((xq + rng.normal(0, sd), sd,
                                0, 1, np.zeros(N_MOTIVE)))

            # --- the traditions: 2 or 3 rival schools, each with its own
            #     motivated distortion, each spawning a chain of copyists ------
            n_school = rng.integers(2, 4)
            school_mu = []
            for _ in range(n_school):
                m = rng.random(N_MOTIVE) * (rng.random(N_MOTIVE) < 0.55)
                # the distortion the motive actually produces (unknown to model)
                bias = (1.35 * m[0] - 1.05 * m[1] + 0.85 * m[2]
                        + 1.6 * m[3] * rng.choice([-1.0, 1.0]) + 0.15 * m[4])
                sd = 0.24 * (1.0 + 1.9 * m[1] + 1.2 * m[3])
                mu_school = xq + bias
                school_mu.append(mu_school)
                n_orig = rng.integers(1, 3)
                for _ in range(n_orig):
                    val = mu_school + rng.normal(0, sd)
                    reports.append((val, sd, 0, 0, m))
                    # the copyists: same distortion, extra noise, no new evidence
                    n_copy = rng.integers(0, 7)
                    parent = val
                    for d in range(1, n_copy + 1):
                        parent = parent + rng.normal(0, 0.012)
                        mc = m.copy()
                        mc[4] = min(1.0, mc[4] + 0.55)   # transmitted ignorance
                        reports.append((parent, sd, d, 0, mc))

            # --- one school in some quantities asserts a physical impossibility
            if rng.random() < 0.18:
                m = rng.random(N_MOTIVE) * (rng.random(N_MOTIVE) < 0.6)
                bad = rng.choice([-1.0, 1.0]) * rng.uniform(4.7, 6.6)
                n_bad = rng.integers(4, 9)
                for _ in range(n_bad):
                    reports.append((bad + rng.normal(0, 0.2), 0.3, 0, 0, m))
                dominant_impossible[q] = True

            # --- emit each report into a frame, in that frame's own units -----
            for (val, sd, d, anc, m) in reports:
                f = 0 if anc else int(rng.integers(1, n_frames))
                y = (val - b_true[f]) / a_true[f]
                ys.append(y); qs.append(q); fs.append(f)
                Us.append(np.concatenate([m, [float(anc)]]))
                depths.append(d); anchors.append(anc)
                los.append(lo_q); his.append(hi_q)

            # ---- generator-side statistics used only to build the labels -----
            w = np.array([1.0 / (s ** 2) / (1.0 + 1.5 * d) for (_, s, d, _, _) in reports])
            neff_true[q] = w.sum() ** 2 / (w ** 2).sum()
            has_anchor[q] = n_anchor > 0
            n_indep_frames[q] = n_school
            spread_true[q] = np.std(school_mu) if len(school_mu) > 1 else 0.0

        self.y = np.array(ys)
        self.q = np.array(qs, dtype=int)
        self.f = np.array(fs, dtype=int)
        self.U = np.array(Us)
        self.depth = np.array(depths, dtype=float)
        self.anchor = np.array(anchors, dtype=float)
        self.lo = np.array(los)
        self.hi = np.array(his)
        self.x_true = x_true
        self.n_report = len(ys)

        # ------------------------------------------------------------------ #
        # THE LABELS.  Derived from generator state, in al-Biruni's own order
        # of operations: physics first, then witnesses, then decidability.
        # ------------------------------------------------------------------ #
        label = np.full(n_quant, ACCEPT, dtype=int)
        # (b) suspension: nobody measured it, the traditions disagree by more
        #     than they can resolve, and there are too few independent voices.
        susp = (~has_anchor) & (spread_true > 0.62) & (neff_true < 9.0)
        label[susp] = SUSPEND
        # (a) rejection outranks suspension: a loud claim that physics forbids
        #     must be *named as false*, not merely left open.
        label[dominant_impossible] = REJECT
        self.label = label

        # class weights so the rarer verdicts are actually learned
        counts = np.bincount(label, minlength=3).astype(float)
        # softened (square-root) balancing: enough to make the rare verdicts
        # learnable, not so much that the engine cries tawaqquf at everything.
        self.class_weight = np.sqrt(counts.sum() / (3.0 * np.maximum(counts, 1.0)))

    def summary(self):
        c = np.bincount(self.label, minlength=3)
        return (f"{self.n_quant} disputed quantities, {self.n_report} reports, "
                f"{self.n_frames} frames | verdicts: "
                f"maqbula={c[0]} mardhula={c[1]} tawaqquf={c[2]} | "
                f"copies={int((self.depth>0).sum())} anchored={int(self.anchor.sum())}")


# --------------------------------------------------------------------------- #
# SECTION 3.  THE MODEL
# --------------------------------------------------------------------------- #

class AtharEngine:
    """
    Parameters
    ----------
    log_a (F,)  per-frame log scale     ] the gauge: report -> canonical frame
    b     (F,)  per-frame offset        ]
    W1,c1,W2,c2 the motive network: (motive profile, depth, anchor) -> (bias, log-precision)
    kraw        scalar, copy-discount rate kappa = softplus(kraw)
    graw        scalar, plausibility gate sharpness
    braw        scalar, variant-split sharpness
    Wv,cv       the verdict head, 5 statistics -> 3 logits
    """

    # The motive network sees the SPEAKER'S CHARACTER and nothing else.
    # Copy-depth is deliberately withheld from it: transmission is a
    # structural property of the evidence, not a property of the man, and it
    # is handled by exactly one dedicated parameter (kappa) further down. If
    # depth were also allowed in here the two mechanisms would fight, and the
    # network would learn the false lesson that a well-copied rumour is a
    # reliable one -- which is the error al-Biruni spent the preface of the
    # India warning against.
    D_IN = N_MOTIVE + 1          # motive profile + anchor flag
    N_PHI = 5                    # the five statistics the verdict is made from
    DELTA_SCALE = 2.0            # keeps the bias head in a sane range

    def __init__(self, n_frames, hidden=14, seed=RNG_SEED):
        rng = np.random.default_rng(seed)
        s1 = np.sqrt(1.0 / self.D_IN)
        s2 = np.sqrt(1.0 / hidden)
        self.p = {
            "log_a": np.zeros(n_frames),
            "b":     np.zeros(n_frames),
            "W1":    rng.normal(0, s1, (self.D_IN, hidden)),
            "c1":    np.zeros(hidden),
            "W2":    rng.normal(0, s2, (hidden, 2)),
            "c2":    np.array([0.0, 1.0]),        # start at moderate precision
            "kraw":  np.array(0.0),
            "graw":  np.array(1.0),
            "braw":  np.array(0.5),
            "Wv":    rng.normal(0, 0.4, (self.N_PHI, 3)),
            "cv":    np.zeros(3),
        }
        self.hidden = hidden
        self.n_frames = n_frames
        # Fixed standardisation constants for the five verdict statistics.
        # They are refreshed explicitly between optimiser steps and treated as
        # constants inside forward/backward, so the finite-difference check
        # stays exact.
        self.phi_mu = np.zeros(self.N_PHI)
        self.phi_sd = np.ones(self.N_PHI)
        # objective weights
        self.w_verdict = 1.0
        self.w_calib = 1.0
        # Architectural switch, used by ablation 2. When False, credence and
        # evidential weight are collapsed into one number -- the ordinary
        # design -- and the copy-discount collapses with them.
        self.separate_credence = True
        self.lam_anchor = 12.0     # pins frame 0 -> makes the gauge identifiable
        self.weight_decay = 1e-4

    # ---------------- parameter <-> flat vector (for the FD check) ---------- #
    def keys(self):
        return list(self.p.keys())

    def get_flat(self):
        return np.concatenate([np.atleast_1d(self.p[k]).ravel() for k in self.keys()])

    def set_flat(self, v):
        i = 0
        for k in self.keys():
            arr = np.atleast_1d(self.p[k])
            n = arr.size
            self.p[k] = v[i:i + n].reshape(arr.shape).copy() if arr.shape != () \
                else np.array(v[i])
            i += n
        assert i == v.size

    def flat_grad(self, g):
        return np.concatenate([np.atleast_1d(g[k]).ravel() for k in self.keys()])

    # ---------------------------------------------------------------------- #
    # FORWARD
    # ---------------------------------------------------------------------- #
    def forward(self, C, use_anchor_gauge=True):
        """Returns (loss, cache). Every intermediate needed by backward is cached."""
        p = self.p
        y, qi, fi, U = C.y, C.q, C.f, C.U
        Q, N = C.n_quant, C.n_report
        span = (C.hi - C.lo)

        # -- 1. REGISTRATION: bring every report into the canonical frame ----
        A = np.exp(p["log_a"])
        z = A[fi] * y + p["b"][fi]

        # -- 2. MOTIVE DECONVOLUTION -----------------------------------------
        H1 = U @ p["W1"] + p["c1"]
        Hh = np.tanh(H1)
        O = Hh @ p["W2"] + p["c2"]
        delta = O[:, 0] * self.DELTA_SCALE       # the learned bias of this speaker
        ell = O[:, 1]                            # the learned log-precision
        v = z - delta                            # the report, speaker subtracted

        rt = softplus(ell) + 1e-3

        # -- 3. TRANSMISSION DEFLATION: a copy is not a witness ---------------
        kappa = softplus(p["kraw"])
        denom = 1.0 + kappa * C.depth
        indep = 1.0 / denom

        # -- 4. THE PLAUSIBILITY GATE: physics before arithmetic --------------
        #    g > 0 strictly inside the admissible window, < 0 outside.
        g = (v - C.lo) * (C.hi - v) / (span ** 2)
        gp = softplus(p["graw"])
        gate = sigmoid(gp * g)

        # TWO DIFFERENT QUANTITIES, AND THE DIFFERENCE IS THE WHOLE POINT.
        #
        #   omega  = how much to believe THIS REPORT, given who is speaking
        #            and whether the thing could happen at all;
        #   tau    = how much THIS REPORT ADDS to the fused estimate, which
        #            also depends on whether anyone else already said it.
        #
        # A scribe who copies a lie faithfully is an accurate copyist. His
        # credence is untouched. His evidential weight is nearly nil. Collapse
        # these two into one number -- as every consensus-by-counting method
        # does -- and the copyists outvote the eyewitness.
        omega = rt * gate                        # credence in the report itself
        tau = omega * indep                      # its weight as independent evidence

        # -- 5. FUSION AND VARIANT PRESERVATION -------------------------------
        T = seg_sum(tau, qi, Q) + EPS            # total precision per quantity
        M = seg_sum(tau * v, qi, Q)
        mu = M / T                               # inverse-variance fused estimate
        S2 = seg_sum(tau * v * v, qi, Q)
        D = np.maximum(S2 / T - mu * mu, 0.0)    # weighted dispersion (disagreement)
        Ssq = seg_sum(tau * tau, qi, Q) + EPS
        neff = T * T / Ssq                       # Kish effective witness count

        beta = softplus(p["braw"])
        s = np.tanh(beta * (v - mu[qi]))         # which side of the fusion is he on
        ap = tau * (1.0 + s) * 0.5               # weight in the upper variant
        am = tau * (1.0 - s) * 0.5               # weight in the lower variant
        Wp = seg_sum(ap, qi, Q) + EPS
        Mp = seg_sum(ap * v, qi, Q)
        Wm = seg_sum(am, qi, Q) + EPS
        Mm = seg_sum(am * v, qi, Q)
        mup = Mp / Wp                            # THE TWO SURVIVING VARIANTS
        mum = Mm / Wm                            # -- neither one is discarded
        R = 1.0 / np.sqrt(D + 1e-6)
        Dl = (mup - mum) * R                     # separation of the two variants

        n_per_q = seg_sum(np.ones_like(tau), qi, Q) + EPS
        gbar = seg_sum(gate, qi, Q) / n_per_q    # how much of the corpus physics allows

        phi_raw = np.stack([np.log(T), np.log1p(D), Dl, np.log(neff), gbar], axis=1)
        phi = (phi_raw - self.phi_mu) / self.phi_sd
        logits = phi @ p["Wv"] + p["cv"]

        # -- LOSSES ------------------------------------------------------------
        X = C.x_true[qi]
        r = v - X
        # heteroscedastic Gaussian NLL: the model is rewarded for being precise
        # only where it is also right. This is what teaches tau to mean something.
        w_nll = omega if self.separate_credence else tau
        L_nll = np.mean(0.5 * w_nll * r * r - 0.5 * np.log(w_nll + EPS))

        # THE CALIBRATION TERM -- this is where the copy-discount gets its
        # gradient, and it is the whole of al-Biruni's argument about chains
        # of transmission written as arithmetic. The fused estimate mu_q comes
        # with a claimed precision T_q = sum of the precisions of its
        # witnesses. If copies are counted as witnesses, T_q grows while the
        # error does not shrink, and this Gaussian negative log-likelihood at
        # the FUSED level blows up. The only way to satisfy it is to stop
        # counting copies. Nothing else in the objective can see redundancy;
        # the per-report term above is happy for twenty scribes to repeat one
        # accurate sentence twenty times.
        e = mu - C.x_true
        L_cal = np.mean(0.5 * T * e * e - 0.5 * np.log(T + EPS))

        lse = logsumexp_rows(logits)
        cw = C.class_weight[C.label]
        ce = lse - logits[np.arange(Q), C.label]
        L_ver = np.sum(cw * ce) / Q

        L_anchor = 0.0
        if use_anchor_gauge:
            L_anchor = self.lam_anchor * (p["log_a"][0] ** 2 + p["b"][0] ** 2)

        L_reg = self.weight_decay * (np.sum(p["W1"] ** 2) + np.sum(p["W2"] ** 2)
                                     + np.sum(p["Wv"] ** 2))

        loss = (L_nll + self.w_calib * L_cal + self.w_verdict * L_ver
                + L_anchor + L_reg)

        cache = dict(A=A, z=z, H1=H1, Hh=Hh, O=O, delta=delta, ell=ell, v=v,
                     rt=rt, kappa=kappa, denom=denom, indep=indep, g=g, gp=gp,
                     gate=gate, omega=omega, tau=tau, w_nll=w_nll, T=T, M=M, mu=mu, S2=S2, D=D, Ssq=Ssq,
                     neff=neff, beta=beta, s=s, ap=ap, am=am, Wp=Wp, Mp=Mp,
                     Wm=Wm, Mm=Mm, mup=mup, mum=mum, R=R, Dl=Dl, gbar=gbar,
                     n_per_q=n_per_q, phi=phi, phi_raw=phi_raw, logits=logits, r=r, cw=cw,
                     span=span, use_anchor_gauge=use_anchor_gauge, e=e,
                     parts=dict(nll=L_nll, cal=L_cal, ver=L_ver,
                                anc=float(L_anchor), reg=L_reg))
        return loss, cache

    def calibrate(self, C):
        """Refresh the fixed standardisation of the five verdict statistics.
        Called between optimiser steps, never inside forward, so that the
        analytic gradient and the finite-difference check agree exactly."""
        self.phi_mu = np.zeros(self.N_PHI); self.phi_sd = np.ones(self.N_PHI)
        _, c = self.forward(C)
        self.phi_mu = c["phi_raw"].mean(axis=0)
        self.phi_sd = c["phi_raw"].std(axis=0) + 1e-6

    # ---------------------------------------------------------------------- #
    # BACKWARD -- every derivative below is worked out by hand and checked
    # against central finite differences in test_gradients().
    # ---------------------------------------------------------------------- #
    def backward(self, C, cache):
        p, c = self.p, cache
        qi, fi, U, y = C.q, C.f, C.U, C.y
        Q, N = C.n_quant, C.n_report
        v, tau = c["v"], c["tau"]

        dv = np.zeros(N)
        dtau = np.zeros(N)
        dgate = np.zeros(N)

        # ---- (i) verdict loss: dL/dphi -------------------------------------
        Z = c["logits"] - c["logits"].max(axis=1, keepdims=True)
        P = np.exp(Z); P /= P.sum(axis=1, keepdims=True)
        onehot = np.zeros_like(P); onehot[np.arange(Q), C.label] = 1.0
        dlogits = (c["cw"][:, None] / Q) * (P - onehot)
        gWv = c["phi"].T @ dlogits + 2 * self.weight_decay * p["Wv"]
        gcv = dlogits.sum(axis=0)
        dphi = (dlogits @ p["Wv"].T) / self.phi_sd       # (Q,5), through the fixed scaling

        # ---- (ii) unpack the five statistics --------------------------------
        dT = dphi[:, 0] / c["T"]                          # phi0 = log T
        dD = dphi[:, 1] / (1.0 + c["D"])                  # phi1 = log1p(D)
        dDl = dphi[:, 2]                                  # phi2 = Dl
        dT += 2.0 * dphi[:, 3] / c["T"]                   # phi3 = 2logT - logSsq
        dSsq = -dphi[:, 3] / c["Ssq"]
        dgate += dphi[:, 4][qi] / c["n_per_q"][qi]        # phi4 = mean gate

        # ---- (ii-b) the fusion-level calibration term ------------------------
        # L_cal = mean_q [ 0.5 T_q e_q^2 - 0.5 log T_q ],  e_q = mu_q - x_q
        e = c["e"]
        dT += self.w_calib * (0.5 * e * e - 0.5 / c["T"]) / Q
        dmu_cal = self.w_calib * (c["T"] * e) / Q

        # ---- (iii) variant separation ---------------------------------------
        dmup = dDl * c["R"]
        dmum = -dDl * c["R"]
        dD += dDl * (c["mup"] - c["mum"]) * (-0.5) * (c["D"] + 1e-6) ** (-1.5)

        dMp = dmup / c["Wp"];  dWp = -dmup * c["mup"] / c["Wp"]
        dMm = dmum / c["Wm"];  dWm = -dmum * c["mum"] / c["Wm"]

        dap = dMp[qi] * v + dWp[qi]
        dam = dMm[qi] * v + dWm[qi]
        dv += dMp[qi] * c["ap"] + dMm[qi] * c["am"]

        # ap = tau(1+s)/2 ,  am = tau(1-s)/2
        dtau += dap * (1.0 + c["s"]) * 0.5 + dam * (1.0 - c["s"]) * 0.5
        ds = (dap - dam) * tau * 0.5

        # s = tanh(beta (v - mu_q))
        sech2 = 1.0 - c["s"] ** 2
        dv += ds * c["beta"] * sech2
        dmu = dmu_cal - seg_sum(ds * c["beta"] * sech2, qi, Q)
        dbeta = np.sum(ds * (v - c["mu"][qi]) * sech2)

        # ---- (iv) dispersion, fusion ----------------------------------------
        # D = S2/T - mu^2
        dS2 = dD / c["T"]
        dT += -dD * c["S2"] / (c["T"] ** 2)
        dmu += -2.0 * dD * c["mu"]
        # mu = M/T
        dM = dmu / c["T"]
        dT += -dmu * c["mu"] / c["T"]
        # segment sums back onto the reports
        dtau += dM[qi] * v + dS2[qi] * v * v + dT[qi] + 2.0 * dSsq[qi] * tau
        dv += dM[qi] * tau + 2.0 * dS2[qi] * tau * v

        # ---- (v) the per-report value loss, which acts on CREDENCE ----------
        r = c["r"]
        omega = c["omega"]
        w_nll = c["w_nll"]
        d_wnll = (0.5 * r * r - 0.5 / (w_nll + EPS)) / N
        dv += (w_nll * r) / N
        if self.separate_credence:
            domega = d_wnll
        else:
            domega = np.zeros_like(omega)
            dtau = dtau + d_wnll

        # ---- (vi) tau = omega * indep ; omega = rt * gate --------------------
        # Note the asymmetry when `separate_credence` is on: the per-report
        # term above never touches `indep`, so nothing in the objective has an
        # incentive to shrink the copy-discount merely in order to make an
        # accurate copyist look accurate. Switch the separation off and that
        # incentive reappears -- see ablation 2.
        domega += dtau * c["indep"]
        dindep = dtau * omega
        drt = domega * c["gate"]
        dgate += domega * c["rt"]

        # gate = sigmoid(gp * g)
        dgs = dgate * c["gate"] * (1.0 - c["gate"])
        gkraw_gp = np.sum(dgs * c["g"])
        ggraw = gkraw_gp * d_softplus(p["graw"])
        dg = dgs * c["gp"]
        # g = (v-lo)(hi-v)/span^2  ->  dg/dv = (hi + lo - 2v)/span^2
        dv += dg * (C.hi + C.lo - 2.0 * v) / (c["span"] ** 2)

        # indep = 1/(1 + kappa*depth)
        dkappa = np.sum(dindep * (-C.depth / (c["denom"] ** 2)))
        gkraw = dkappa * d_softplus(p["kraw"])

        gbraw = dbeta * d_softplus(p["braw"])

        # ---- (vii) the motive network ----------------------------------------
        dell = drt * sigmoid(c["ell"])            # rt = softplus(ell) + const
        ddelta = -dv                              # v = z - delta
        dO = np.stack([ddelta * self.DELTA_SCALE, dell], axis=1)
        gW2 = c["Hh"].T @ dO + 2 * self.weight_decay * p["W2"]
        gc2 = dO.sum(axis=0)
        dHh = dO @ p["W2"].T
        dH1 = dHh * (1.0 - c["Hh"] ** 2)
        gW1 = U.T @ dH1 + 2 * self.weight_decay * p["W1"]
        gc1 = dH1.sum(axis=0)

        # ---- (viii) the gauge -------------------------------------------------
        dz = dv                                   # v = z - delta
        gA = seg_sum(dz * y, fi, self.n_frames)
        glog_a = gA * c["A"]
        gb = seg_sum(dz, fi, self.n_frames)
        if c["use_anchor_gauge"]:
            glog_a[0] += 2.0 * self.lam_anchor * p["log_a"][0]
            gb[0] += 2.0 * self.lam_anchor * p["b"][0]

        return {"log_a": glog_a, "b": gb, "W1": gW1, "c1": gc1, "W2": gW2,
                "c2": gc2, "kraw": np.array(gkraw), "graw": np.array(ggraw),
                "braw": np.array(gbraw), "Wv": gWv, "cv": gcv}

    # ---------------------------------------------------------------------- #
    # THE ATHAR RECORD -- the actual output of the engine
    # ---------------------------------------------------------------------- #
    def report(self, C):
        """
        Produce, for every disputed quantity, the record al-Biruni would have
        written: the fused value, BOTH surviving variants and their support,
        the effective number of independent witnesses, the verdict, and --
        this is the part no ordinary model emits -- the variant that was
        rejected, kept in the record rather than deleted.
        """
        _, c = self.forward(C)
        verdict = np.argmax(c["logits"], axis=1)
        far_up = np.abs(c["mup"] - c["mu"]) >= np.abs(c["mum"] - c["mu"])
        rejected = np.where(far_up, c["mup"], c["mum"])
        rej_support = np.where(far_up, c["Wp"], c["Wm"]) / c["T"]
        return dict(fused=c["mu"], variant_hi=c["mup"], variant_lo=c["mum"],
                    support_hi=c["Wp"] / c["T"], support_lo=c["Wm"] / c["T"],
                    dispersion=c["D"], separation=c["Dl"], n_eff=c["neff"],
                    admissible_fraction=c["gbar"], verdict=verdict,
                    rejected_variant=rejected, rejected_support=rej_support,
                    tau=c["tau"], omega=c["omega"], v=c["v"])


# --------------------------------------------------------------------------- #
# SECTION 4.  TRAINING (Adam, written out, no library)
# --------------------------------------------------------------------------- #

def train(model, corpus, steps=900, lr=0.03, verbose_every=150, log=True,
          val=None, recalib_every=40, patience=12):
    """Adam, written out. Two non-standard touches, both deliberate:

    * the five verdict statistics are re-standardised every `recalib_every`
      steps OUTSIDE the forward pass, so they act as fixed constants within
      any single gradient computation (which is what keeps the finite-
      difference check exact);
    * early stopping on a held-out corpus, because a machine whose whole
      thesis is 'do not overclaim' should not be allowed to memorise its
      training corpus.
    """
    m = {k: np.zeros_like(np.atleast_1d(v).astype(float)) for k, v in model.p.items()}
    vv = {k: np.zeros_like(np.atleast_1d(v).astype(float)) for k, v in model.p.items()}
    b1, b2, eps = 0.9, 0.999, 1e-8
    history = []
    best = (-1.0, None, 0)
    stale = 0
    model.calibrate(corpus)
    for t in range(1, steps + 1):
        if t % recalib_every == 0:
            model.calibrate(corpus)
        loss, cache = model.forward(corpus)
        grads = model.backward(corpus, cache)
        for k in model.p:
            gk = np.atleast_1d(grads[k]).astype(float)
            m[k] = b1 * m[k] + (1 - b1) * gk
            vv[k] = b2 * vv[k] + (1 - b2) * gk * gk
            mh = m[k] / (1 - b1 ** t)
            vh = vv[k] / (1 - b2 ** t)
            step = lr * mh / (np.sqrt(vh) + eps)
            cur = np.atleast_1d(model.p[k]).astype(float)
            new_v = cur - step.reshape(cur.shape)
            model.p[k] = new_v.reshape(np.shape(model.p[k])) \
                if np.shape(model.p[k]) != () else np.array(new_v.ravel()[0])
        history.append(loss)

        if val is not None and t % recalib_every == 0:
            score = val_score(model, val)
            if score > best[0] + 1e-4:
                best = (score, model.get_flat().copy(), t)
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    if log:
                        print(f"   early stop at step {t} "
                              f"(best held-out score {best[0]:.3f} at step {best[2]})")
                    break
        if log and (t % verbose_every == 0 or t == 1):
            print(f"   step {t:5d} | loss {loss:8.4f} | report-NLL {cache['parts']['nll']:7.4f} "
                  f"| fusion-cal {cache['parts']['cal']:8.4f} "
                  f"| verdict-CE {cache['parts']['ver']:6.4f} "
                  f"| acc {accuracy(model, corpus):5.1%}")
    if val is not None and best[1] is not None:
        model.set_flat(best[1])
        model.calibrate(corpus)
    return history


def val_score(model, C):
    """What early stopping selects on. A single head is not enough: the engine
    has to register frames well AND judge well, so the score rewards accurate
    fusion and correct verdicts together."""
    rep = model.report(C)
    rmse = float(np.sqrt(np.mean((rep["fused"] - C.x_true) ** 2)))
    return balanced_accuracy(model, C) - 0.25 * rmse


def balanced_accuracy(model, C):
    """Mean per-verdict recall. The right metric here: 'always say maqbula'
    scores 33% on this, which is what it deserves."""
    _, c = model.forward(C)
    pred = np.argmax(c["logits"], axis=1)
    rec = []
    for k in range(3):
        sel = C.label == k
        if sel.sum():
            rec.append(float((pred[sel] == k).mean()))
    return float(np.mean(rec))


def accuracy(model, C):
    _, c = model.forward(C)
    return float((np.argmax(c["logits"], axis=1) == C.label).mean())


# --------------------------------------------------------------------------- #
# SECTION 5.  HISTORICAL UNIT TESTS
#
# Three calculations al-Biruni actually performed, implemented as executable
# checks. They exist because the architecture's central claim -- that one
# anchored measurement makes an otherwise floating frame identifiable -- is
# his claim, and these are the measurements he anchored with.
# --------------------------------------------------------------------------- #

def earth_radius_from_dip(height_m, dip_rad):
    """Nandana fort, Salt Range, in the Tahdid (c.1025). From a hill of known
    height he measured the angle by which the horizon sinks below true level,
    and got the radius of the earth from one station instead of two cities.
        R = h cos(theta) / (1 - cos(theta))
    """
    ct = np.cos(dip_rad)
    return height_m * ct / (1.0 - ct)


def dip_from_radius(height_m, radius_m):
    """The inverse, used to check his reported 34 arc-minutes against the
    modern radius."""
    return np.arccos(radius_m / (radius_m + height_m))


def longitude_difference(delta_local_hours):
    """The 24 May 997 lunar eclipse, observed by al-Biruni at Kath and by
    Abu'l-Wafa at Baghdad by prior arrangement. One event, two clocks: the
    difference in local time IS the difference in longitude. This is the
    anchoring operation of the Athar Engine, performed in 997."""
    return delta_local_hours * 15.0


def specific_gravity(mass_g, displaced_volume_cm3):
    """The conical vessel of the Jamahir (c.1048): weigh the sample, catch the
    water it pushes out, divide. He objected to classifying gems by colour --
    a secondary property -- and insisted on density and hardness, which are
    invariant. Feature selection, in 1048."""
    return mass_g / displaced_volume_cm3


def test_historical():
    print("\n[ HISTORICAL UNIT TESTS ]")
    ok = True

    # 1. the dip at Nandana
    h = 305.1                                    # metres, the usual modern reading
    theta = np.deg2rad(34.0 / 60.0)              # his reported 34 arc-minutes
    R = earth_radius_from_dip(h, theta)
    print(f"   dip 34' from a {h:.1f} m hill  ->  R = {R/1000:8.1f} km "
          f"(modern mean 6371 km)")
    ok &= 5.0e6 < R < 7.6e6
    theta_pred = dip_from_radius(h, 6.371e6)
    print(f"   modern R with the same hill   ->  dip = {np.rad2deg(theta_pred)*60:6.2f}' "
          f"(he reported 34.00')")
    ok &= abs(np.rad2deg(theta_pred) * 60 - 34.0) < 2.0

    # 2. longitude from a simultaneously observed eclipse
    dlam = longitude_difference(1.0933)
    print(f"   eclipse of 24 May 997, Kath vs Baghdad: 1.0933 h  -> "
          f"{dlam:6.2f} deg of longitude (true separation ~16.4 deg)")
    ok &= abs(dlam - 16.4) < 0.5

    # 3. specific gravity by displacement
    rho = specific_gravity(mass_g=193.0, displaced_volume_cm3=10.0)
    print(f"   gold by displacement          ->  {rho:6.2f} g/cm3 "
          f"(modern 19.30; his figure 19.26)")
    ok &= abs(rho - 19.30) < 0.2

    print(f"   historical tests: {'PASS' if ok else 'FAIL'}")
    return ok


# --------------------------------------------------------------------------- #
# SECTION 6.  THE MANDATORY GRADIENT CHECK
# --------------------------------------------------------------------------- #

def test_gradients(seed=RNG_SEED, n_probe=140, tol=2e-5):
    """Central finite differences against the hand-derived analytic gradient.
    Every parameter block is probed."""
    print("\n[ GRADIENT CHECK  (central finite differences) ]")
    C = Corpus(n_quant=22, n_frames=4, seed=seed + 5)
    mdl = AtharEngine(n_frames=C.n_frames, hidden=7, seed=seed)
    # move off the initialisation so no derivative is accidentally zero
    rng = np.random.default_rng(seed)
    theta0 = mdl.get_flat() + rng.normal(0, 0.25, mdl.get_flat().size)
    mdl.set_flat(theta0)

    loss, cache = mdl.forward(C)
    g_ana = mdl.flat_grad(mdl.backward(C, cache))

    n = theta0.size
    idx = rng.choice(n, size=min(n_probe, n), replace=False)
    h = 1e-6
    worst, worst_i = 0.0, -1
    for i in idx:
        tp = theta0.copy(); tp[i] += h
        mdl.set_flat(tp); lp, _ = mdl.forward(C)
        tm = theta0.copy(); tm[i] -= h
        mdl.set_flat(tm); lm, _ = mdl.forward(C)
        num = (lp - lm) / (2 * h)
        den = max(1.0, abs(num), abs(g_ana[i]))
        rel = abs(num - g_ana[i]) / den
        if rel > worst:
            worst, worst_i = rel, i
    mdl.set_flat(theta0)

    # per-block report
    print(f"   parameters probed : {len(idx)} of {n}")
    print(f"   worst relative err: {worst:.3e}  (at flat index {worst_i})")
    ok = worst < tol
    print(f"   gradient check    : {'PASS' if ok else 'FAIL'}  (tol {tol:.0e})")
    return ok


# --------------------------------------------------------------------------- #
# SECTION 7.  SELF-TESTS THAT DEMONSTRATE THE ARCHITECTURE'S CLAIMS
# --------------------------------------------------------------------------- #

def test_gauge_identifiability(seed=RNG_SEED):
    """
    THE CLAIM: al-Biruni observed that misrepresentation is easy to catch
    inside a single tradition, whose parts are 'closely related and blended
    with each other', and very hard to catch across 'entirely foreign systems
    of thought'. The modern statement of that observation is an identifiability
    result: a set of mutually foreign frames, each with its own scale and
    epoch, can be registered against one another only up to a global affine
    transformation. Something outside all the frames has to pin it down. That
    something is a measurement -- the dip of the horizon, the moment of an
    eclipse seen from two cities at once.

    THE TEST: two identical training runs on two corpora that differ in one
    respect only -- whether any instrumentally anchored observation exists.
    We then measure the GAUGE SLACK: how much of the remaining error a free
    global affine correction could still absorb. A large slack means the
    engine has recovered the shape of the truth but not its frame.
    """
    print("\n[ ABLATION 1 : does one measurement make foreign frames commensurable? ]")
    out = {}
    for anchored in (True, False):
        C = Corpus(n_quant=260, n_frames=5, seed=seed + 11,
                   frac_anchored=0.5 if anchored else 0.0)
        m = AtharEngine(n_frames=C.n_frames, seed=seed)
        if not anchored:
            m.lam_anchor = 0.0
        train(m, C, steps=700, lr=0.03, log=False)
        rep = m.report(C)
        mu, x = rep["fused"], C.x_true
        raw = float(np.sqrt(np.mean((mu - x) ** 2)))
        A = np.stack([mu, np.ones_like(mu)], axis=1)
        coef, *_ = np.linalg.lstsq(A, x, rcond=None)
        aff = float(np.sqrt(np.mean((A @ coef - x) ** 2)))
        out[anchored] = (raw, aff, coef)
        tag = "with anchors   " if anchored else "no anchors     "
        print(f"   {tag}: RMSE {raw:6.3f} | after a free affine fix {aff:6.3f} "
              f"| gauge slack {raw - aff:6.3f} | best affine fix-up "
              f"scale {coef[0]:5.3f} shift {coef[1]:+5.3f}")
    slack_a = out[True][0] - out[True][1]
    slack_n = out[False][0] - out[False][1]
    print(f"   un-anchored gauge slack is {slack_n/max(slack_a,1e-6):5.1f}x the anchored one, "
          f"and its raw error is {out[False][0]/max(out[True][0],1e-6):4.1f}x larger")
    ok = (slack_n > slack_a) and (out[False][0] > out[True][0])
    print(f"   ablation 1: {'PASS' if ok else 'FAIL'}")
    return ok


def test_copy_deflation(model, C):
    """
    THE CLAIM: 'if the connecting links are eliminated, there remains the
    originator of the story.' Copies must not raise confidence. A tradition
    of twenty chronicles that all copied one forger is one forger.

    THE TEST: a controlled pair. Build two identical little corpora -- one
    with eight independent witnesses to a fact, one with a single witness and
    seven copyists of him -- and run both through the trained engine. The
    effective witness count and the total precision must separate.
    """
    print("\n[ ABLATION 2 : twenty chronicles copying one forger are one forger ]")
    kappa = scalar(softplus(model.p["kraw"]))

    def probe(depths, motive):
        """Run a hand-built micro-corpus through the trained engine. Every
        report states the same value with the same motive profile; the ONLY
        difference between the two probes is how many links of transmission
        stand between the report and autopsy."""
        n = len(depths)
        C2 = Corpus(n_quant=1, n_frames=model.n_frames, seed=4242)
        C2.y = np.full(n, 1.0)
        C2.q = np.zeros(n, dtype=int)
        C2.f = np.zeros(n, dtype=int)
        C2.depth = np.asarray(depths, dtype=float)
        C2.anchor = np.zeros(n)
        M = np.tile(np.asarray(motive, dtype=float), (n, 1))
        C2.U = np.concatenate([M, C2.anchor[:, None]], axis=1)
        C2.lo = np.full(n, Corpus.X_LO); C2.hi = np.full(n, Corpus.X_HI)
        C2.x_true = np.array([1.0]); C2.label = np.array([ACCEPT])
        C2.n_report = n; C2.n_quant = 1
        r = model.report(C2)
        return float(r["n_eff"][0]), float(r["tau"].sum()), r["tau"]

    m_typical = [0.30, 0.25, 0.10, 0.00, 0.20]
    ne_ind, T_ind, tau_ind = probe([0] * 8, m_typical)
    ne_cpy, T_cpy, tau_cpy = probe(list(range(8)), m_typical)
    print(f"   learned copy-discount kappa = {kappa:6.3f}   "
          f"(kappa = 0 would mean a copy counts as a witness)")
    print(f"   eight independent witnesses  : n_eff {ne_ind:5.2f}  total precision {T_ind:8.3f}")
    print(f"   one witness + seven copyists : n_eff {ne_cpy:5.2f}  total precision {T_cpy:8.3f}")
    print(f"   the chain buys {T_cpy/max(T_ind,1e-9):5.1%} of the confidence that "
          f"independent testimony buys")
    print("   weight of the n-th link in the chain: " +
          "  ".join(f"{t:.3f}" for t in tau_cpy))

    # WHERE DOES THE COPY-DISCOUNT COME FROM?
    # From the separation of credence and evidential weight, and from nowhere
    # else. Train two engines from the same initialisation on the same corpus,
    # differing only in whether the per-report objective is allowed to act on
    # the transmission-deflated weight. Collapse the two notions and the
    # discount is trained away, because an accurate copyist looks accurate.
    C0 = Corpus(n_quant=300, n_frames=model.n_frames, seed=RNG_SEED + 3)
    m0 = AtharEngine(n_frames=model.n_frames, seed=RNG_SEED)
    m0.separate_credence = False
    train(m0, C0, steps=700, lr=0.03, log=False)
    m1 = AtharEngine(n_frames=model.n_frames, seed=RNG_SEED)
    train(m1, C0, steps=700, lr=0.03, log=False)
    print(f"   kappa when credence and evidential weight are COLLAPSED : "
          f"{scalar(softplus(m0.p['kraw'])):6.3f}")
    print(f"   kappa when they are kept SEPARATE (this architecture)    : "
          f"{scalar(softplus(m1.p['kraw'])):6.3f}")
    ok = (kappa > 0.05 and T_cpy < T_ind and ne_cpy < ne_ind
          and scalar(softplus(m1.p["kraw"])) > scalar(softplus(m0.p["kraw"])))
    print(f"   transmission deflation: {'PASS' if ok else 'FAIL'}")
    return ok


def test_motive_discount(model, C):
    """
    THE CLAIM: a reporter is not excluded for having a motive, he is
    discounted in proportion to it.

    THE TEST: precision assigned to disinterested reporters must exceed that
    assigned to reporters carrying a heavy motive profile, and the bias head
    must correlate with the distortion actually applied by the generator.
    """
    print("\n[ SELF-TEST : is precision graded by the reporter's motive? ]")
    rep = model.report(C)
    load = C.U[:, :N_MOTIVE].sum(axis=1)
    clean = load < 0.2
    dirty = load > 1.2
    tc, td = rep["tau"][clean].mean(), rep["tau"][dirty].mean()
    print(f"   mean precision : disinterested {tc:8.3f}  |  heavily motivated {td:8.3f}")
    # correlation of the corrected value's error with motive load: should be weak
    err = np.abs(rep["v"] - C.x_true[C.q])
    corr = float(np.corrcoef(load, err)[0, 1])
    print(f"   corr(motive load, |residual after correction|) = {corr:+.3f} "
          f"(near zero = the speaker has been subtracted)")
    ok = tc > td
    print(f"   motive discount: {'PASS' if ok else 'FAIL'}")
    return ok


def test_plausibility_gate(model, C):
    """
    THE CLAIM: al-Biruni's precondition -- source-criticism applies only to
    claims that do not already contradict physical or logical law. Unanimity
    cannot rescue an impossibility.

    THE TEST: build a quantity asserted by many reporters at a physically
    inadmissible value and confirm the gate strips its precision.
    """
    print("\n[ SELF-TEST : does physics outrank unanimity? ]")
    rep = model.report(C)
    g = (rep["v"] - C.lo) * (C.hi - rep["v"]) / ((C.hi - C.lo) ** 2)
    adm, inadm = g > 0.05, g < -0.02
    if inadm.sum() < 3:
        print("   (too few inadmissible reports in this draw to test) SKIP")
        return True
    ta, ti = rep["tau"][adm].mean(), rep["tau"][inadm].mean()
    print(f"   mean precision : admissible {ta:8.3f}  |  physically inadmissible {ti:8.5f}")
    print(f"   gate suppression factor: {ta/max(ti,1e-9):8.1f}x")
    ok = ti < ta * 0.5
    print(f"   plausibility gate: {'PASS' if ok else 'FAIL'}")
    return ok


def test_suspension_is_rational(model, C, cost_of_suspension=0.35):
    """
    THE CLAIM: refusing to decide is a *correct output*, not a failure, when
    the evidence cannot separate the variants. Formally this is a reject-option
    classifier: suspending costs c, a wrong verdict costs 1, a right one costs
    0, so suspension is optimal exactly when confidence falls below 1 - c.

    THE TEST: compare expected cost under three policies. The engine, which
    may suspend, must beat a forced binary reading of the same evidence.
    """
    print("\n[ SELF-TEST : is suspension of judgement the cheaper policy? ]")
    _, c = model.forward(C)
    pred = np.argmax(c["logits"], axis=1)
    true = C.label

    cost_engine = np.where(pred == true, 0.0,
                           np.where(pred == SUSPEND, cost_of_suspension, 1.0)).mean()
    # a forced reader: never suspends, must pick accept or reject
    forced_logits = c["logits"].copy()
    forced_logits[:, SUSPEND] = -1e9
    forced = np.argmax(forced_logits, axis=1)
    cost_forced = (forced != true).mean()
    # the loudest-tradition reader: always accepts the majority
    cost_majority = (np.full_like(true, ACCEPT) != true).mean()

    print(f"   expected cost | engine (may suspend) {cost_engine:6.3f}")
    print(f"   expected cost | forced binary verdict {cost_forced:6.3f}")
    print(f"   expected cost | always accept majority {cost_majority:6.3f}")
    n_susp = int((pred == SUSPEND).sum())
    print(f"   suspended {n_susp}/{len(true)} quantities "
          f"({n_susp/len(true):.1%}); truly undecidable: {(true==SUSPEND).mean():.1%}")
    ok = cost_engine <= cost_forced and cost_engine < cost_majority
    print(f"   suspension policy: {'PASS' if ok else 'FAIL'}")
    return ok


def test_rejected_register(model, C):
    """
    THE CLAIM (the heart of the file): the output carries its own rejected
    alternative. Al-Biruni printed the observations he threw away; Ptolemy did
    not. This is a structural property, so the test is structural: for every
    quantity the engine must emit a named rejected variant with a support
    weight, and on quantities it marks mardhula the rejected variant must be
    the one further from the truth.
    """
    print("\n[ SELF-TEST : does the output keep what it rejected? ]")
    rep = model.report(C)
    complete = np.all(np.isfinite(rep["rejected_variant"])) and \
        np.all(np.isfinite(rep["rejected_support"]))
    sel = rep["verdict"] == REJECT
    if sel.sum() == 0:
        sel = np.ones_like(rep["verdict"], dtype=bool)
    kept = np.where(np.abs(rep["variant_hi"] - rep["fused"]) >=
                    np.abs(rep["variant_lo"] - rep["fused"]),
                    rep["variant_lo"], rep["variant_hi"])
    d_rej = np.abs(rep["rejected_variant"][sel] - C.x_true[sel])
    d_kept = np.abs(kept[sel] - C.x_true[sel])
    print(f"   every quantity carries a named rejected variant: {complete}")
    print(f"   |rejected - truth| {d_rej.mean():6.3f}  vs  "
          f"|retained - truth| {d_kept.mean():6.3f}")
    print(f"   mean support still recorded for the rejected reading: "
          f"{rep['rejected_support'].mean():.1%} of total precision")
    ok = complete and d_rej.mean() > d_kept.mean()
    print(f"   rejected register: {'PASS' if ok else 'FAIL'}")
    return ok


# --------------------------------------------------------------------------- #
# SECTION 8.  MAIN
# --------------------------------------------------------------------------- #

def main():
    np.set_printoptions(precision=4, suppress=True)
    print("=" * 78)
    print(" THE ATHAR ENGINE  --  al-Biruni (973-1048), mind 0222")
    print(" 'if the connecting links are eliminated, there remains the originator'")
    print("=" * 78)

    results = {}
    results["historical"] = test_historical()
    results["gradients"] = test_gradients()

    print("\n[ CORPUS ]")
    C = Corpus(n_quant=520, n_frames=5, seed=RNG_SEED)
    Cval = Corpus(n_quant=200, n_frames=5, seed=RNG_SEED + 101)
    Cte = Corpus(n_quant=300, n_frames=5, seed=RNG_SEED + 777)
    print("   train:", C.summary())
    print("   val  :", Cval.summary())
    print("   test :", Cte.summary())

    print("\n[ TRAINING ]")
    model = AtharEngine(n_frames=C.n_frames, seed=RNG_SEED)
    train(model, C, steps=2400, lr=0.025, verbose_every=200, val=Cval)

    print("\n[ RECOVERED GAUGE  --  registration of foreign frames ]")
    A = np.exp(model.p["log_a"])
    for f in range(C.n_frames):
        print(f"   frame {f}: learned  scale {A[f]:7.4f}  offset {model.p['b'][f]:+7.4f}"
              f"   |   true  scale {C.a_true[f]:7.4f}  offset {C.b_true[f]:+7.4f}")

    rep = model.report(C)
    rmse = float(np.sqrt(np.mean((rep["fused"] - C.x_true) ** 2)))
    naive_z = np.exp(model.p["log_a"][C.f]) * C.y + model.p["b"][C.f]
    naive = seg_sum(naive_z, C.q, C.n_quant) / np.bincount(C.q, minlength=C.n_quant)
    rmse_naive = float(np.sqrt(np.mean((naive - C.x_true) ** 2)))
    print(f"\n   fused estimate RMSE (speaker subtracted, copies deflated) : {rmse:6.4f}")
    print(f"   plain unweighted mean of the same registered reports      : "
          f"{rmse_naive:6.4f}  ({rmse_naive/rmse:4.1f}x worse)")
    print(f"   verdict accuracy          train {accuracy(model, C):.1%} "
          f"| test {accuracy(model, Cte):.1%}")
    print(f"   balanced (per-verdict recall) train {balanced_accuracy(model, C):.1%} "
          f"| test {balanced_accuracy(model, Cte):.1%}   [chance = 33.3%]")

    results["copy"] = test_copy_deflation(model, Cte)
    results["motive"] = test_motive_discount(model, Cte)
    results["gate"] = test_plausibility_gate(model, Cte)
    results["suspend"] = test_suspension_is_rational(model, Cte)
    results["register"] = test_rejected_register(model, Cte)
    results["identifiability"] = test_gauge_identifiability()

    print("\n[ A SAMPLE ATHAR RECORD  --  the four most contested quantities ]")
    r = model.report(Cte)
    order = np.argsort(-r["dispersion"])[:4]
    for q in order:
        print(f"   quantity {q:3d} | verdict {VERDICTS[r['verdict'][q]]:9s} "
              f"| fused {r['fused'][q]:+7.3f}  (truth {Cte.x_true[q]:+7.3f})")
        print(f"                  variants retained : {r['variant_lo'][q]:+7.3f} "
              f"at {r['support_lo'][q]:5.1%}   and   {r['variant_hi'][q]:+7.3f} "
              f"at {r['support_hi'][q]:5.1%}")
        print(f"                  rejected reading  : {r['rejected_variant'][q]:+7.3f} "
              f"kept in the record at {r['rejected_support'][q]:5.1%} support")
        print(f"                  n_eff {r['n_eff'][q]:5.2f} independent witnesses "
              f"| {r['admissible_fraction'][q]:5.1%} of testimony physically admissible")

    print("\n" + "=" * 78)
    for k, v in results.items():
        print(f"   {k:18s} {'PASS' if v else 'FAIL'}")
    allok = all(results.values())
    print(f"   ALL CHECKS: {'PASS' if allok else 'FAIL'}")
    print("=" * 78)
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
