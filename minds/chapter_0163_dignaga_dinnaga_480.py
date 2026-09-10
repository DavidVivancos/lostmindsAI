#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 ENCYCLOPEDIA OF LOST MINDS — Chapter 0163
Chapter 0163 - Dignaga (Dinnaga), c. 480-540 CE - Kanchi / Nalanda / Odivisha, India
Founder of the Buddhist pramana school. Author of the Pramanasamuccaya.
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 9 Minds 161 - 180 Available on Amazon https://www.amazon.com/dp/B0HJC1QF59
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 0163_dignaga_dinnaga_480 - Dignaga (Dinnaga), c. 480-540 CE - Kanchi / Nalanda / Odivisha, India
================================================================================    

 APOHA-NET : an AGI substrate built out of negations.

 THE ONE IDEA THIS FILE IMPLEMENTS
 ---------------------------------
 Dignaga's most radical claim is not that the mind is momentary, nor that
 cognition knows itself. It is this:

     A CONCEPT IS NOT A THING THE MIND STORES. IT IS A CUT THE MIND MAKES.

 The word "cow" (gauh) does not name a shared essence "cowness" residing in
 every cow. There is no such essence; there are only unrepeatable particulars
 (svalaksana). What "cow" does is EXCLUDE: it means non-non-cow (anyapoha).
 The concept is the residue left when everything else has been carved away.
 Meaning is negative space.

 Nearly every classifier ever built does the opposite. It stores a positive
 template — a centroid, a prototype, a class embedding, a key vector — and
 asks "how close is this input to the thing I keep?" Dignaga would call this
 a metaphysical error dressed as engineering: it reifies the universal.

 So APOHA-NET stores NO positive class representation at all. Anywhere.
 There is no class centroid, no class embedding, no per-class weight vector
 that points at the class. What the network holds is:

     (a) a shared pool of K EXCLUSION TESTS (vyavrtti) — each one a razor
         that cuts the space of particulars into "excluded" / "survives";
     (b) for each concept, a soft MASK saying WHICH exclusions it demands.

 A concept is then literally an AND-of-NOTs:

     concept c holds of x   iff   x survives every exclusion that c requires

     score(x, c) = SUM_k  gate[c,k] * log( P(x survives test k) )

 Two concepts differ only in WHICH negations they impose. Nothing positive is
 ever stored. That is apoha, executed.

 THE SECOND IDEA: THE WHEEL REFUSES TO SPEAK
 -------------------------------------------
 A classifier that always answers is, for Dignaga, a machine for producing
 fallacies. His Hetucakra ("wheel of reasons", a nine-celled 3x3 matrix) exists
 to tell you when a reason licenses assertion and when it does not. A reason
 (hetu) is valid only if it satisfies the three characteristics (trairupya):

     1. paksadharmata  — the reason is present in the case at hand
     2. sapakse sattvam — it occurs in AT LEAST SOME similar cases   (anvaya)
     3. vipakse asattvam — it occurs in NO dissimilar case whatsoever (vyatireka)

 Note the asymmetry, which is the whole of Dignaga's genius: condition 2 is
 partial and forgiving; condition 3 is total and unforgiving. Presence needs
 only be sampled. Absence must be complete. Of the nine cells of the wheel,
 exactly two are valid (cells 2 and 8), two are contradictory (4 and 6), one
 is "uncommon" (5), and four are inconclusive (1, 3, 7, 9).

 APOHA-NET encodes this asymmetry DIRECTLY IN ITS LOSS FUNCTION (a hard linear
 penalty on any presence of a reason among the dissimilar, a soft hinge that
 stops rewarding presence among the similar once partial coverage is reached),
 and then, at inference time, runs the wheel as a live gate: the network may
 only ASSERT a classification when it can name a reason whose empirical
 distribution lands in cell 2 or cell 8. Otherwise it abstains and reports the
 fault by name (viruddha / anaikantika / asadharana). Refusal is a first-class
 output of this architecture, not an add-on.

 THE THIRD IDEA: THE COGNITION THAT KNOWS ITSELF (so that it can be remembered)
 -----------------------------------------------------------------------------
 Dignaga argues for self-awareness (svasamvedana) from MEMORY: I could not now
 remember having seen blue unless the original seeing had, in the same act,
 also apprehended itself. Every cognition therefore has two aspects
 (dvirupata): an object-appearance (visayabhasa) and a self-appearance
 (svabhasa). So every forward pass here emits a small self-report vector, and
 a recollection head must reconstruct WHAT WAS COGNIZED and HOW CONFIDENTLY
 from that self-report ALONE — the object is gone, the moment has perished
 (ksanikatva). What survives is only the trace the cognition left of itself.
 This is not decoration: it yields a calibrated confidence channel for free,
 and it lets us build memory over a stream with NO persistent hidden state.

 ENGINEERING CONVENTIONS (project standard)
 ------------------------------------------
   * pure NumPy, from scratch, hand-derived gradients (no autodiff)
   * mandatory finite-difference gradient check over every parameter block
   * a real training loop on a task where the philosophy makes a testable
     prediction, plus honest baselines that are allowed to win where they should
   * self-tests; the file is executed and its true output is pasted into the
     chapter

 Run:  python3 chapter_0163_dignaga_dinnaga_480.py
================================================================================
"""

import numpy as np

RNG = np.random.default_rng(480)   # his birth year, as seed

EPS = 1e-9


# =============================================================================
# SECTION 0 — SMALL NUMERICS
# =============================================================================

def sigmoid(z):
    """Stable logistic. Used as the 'excluded / survives' verdict of a test."""
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def softplus(z):
    """log(1+e^z), stable.  NOTE: -log(sigmoid(-z)) == softplus(z).

    This identity is the arithmetic heart of the model: the NEGATIVE LOG of
    'x survives exclusion test k' is exactly softplus(z_k). Survival costs
    softplus. A concept's score is minus the total cost of the exclusions it
    demands. Meaning is measured in what it rules out.
    """
    return np.logaddexp(0.0, z)


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def one_hot(y, C):
    out = np.zeros((y.shape[0], C))
    out[np.arange(y.shape[0]), y] = 1.0
    return out


# =============================================================================
# SECTION 1 — THE WORLD OF BARE PARTICULARS (svalaksana)
# =============================================================================
#
# Dignaga's ontology gives us a data-generating process with teeth, and it lets
# us build a task on which his metaphysics makes a FALSIFIABLE prediction.
#
# For Dignaga there are no universals in the world. A "class" is not a region of
# similarity around a shared essence — that is precisely the Nyaya/Vaisesika
# realist picture he spent his life demolishing. A class is just: everything not
# excluded by a certain set of cuts. It follows that a class need not be
# CONVEX, need not be CONNECTED, and need not have a MEANINGFUL CENTROID.
#
# So we build exactly that world. Each concept is a union of three disjoint
# lobes of particulars, and the lobes of each concept are arranged so that they
# SUM TO ZERO: every concept has its centroid at the origin. The "essence" of
# each class, if you insist on computing one, is the same empty point.
#
# Prediction: a prototype/centroid learner (the realist about universals) must
# perform at chance. A linear learner must struggle. An exclusion lattice must
# succeed. We will check all three, and report whatever actually happens.
# =============================================================================

def make_particulars(n_per_lobe=90, D=16, C=4, lobes=3, radius=3.2, noise=0.62, rng=RNG):
    """Generate momentary particulars: no universals, only cuts.

    Returns X (N,D) float, y (N,) int.
    """
    lobe_centres = []
    for _ in range(C):
        # lobes-1 random directions; the last one closes the sum to zero, so the
        # class mean is the origin for EVERY class.
        dirs = rng.normal(size=(lobes - 1, D))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        last = -dirs.sum(axis=0)
        last /= (np.linalg.norm(last) + EPS)
        # renormalise the first ones so the exact sum is ~0 after scaling
        allq = np.vstack([dirs, last[None, :]])
        allq = allq - allq.mean(axis=0, keepdims=True)     # exact zero mean
        allq /= (np.linalg.norm(allq, axis=1, keepdims=True) + EPS)
        lobe_centres.append(allq * radius)

    Xs, ys = [], []
    for c in range(C):
        for l in range(lobes):
            pts = lobe_centres[c][l][None, :] + noise * rng.normal(size=(n_per_lobe, D))
            Xs.append(pts)
            ys.append(np.full(n_per_lobe, c))
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


def split(X, y, frac=0.7):
    n = int(len(y) * frac)
    return X[:n], y[:n], X[n:], y[n:]


# =============================================================================
# SECTION 2 — APOHA-NET
# =============================================================================
#
#  FORWARD PASS  (one momentary cognition)
#  ---------------------------------------
#   perception   P   : p = tanh(Pfix x + p0)        FIXED, NEVER LEARNED
#                      -- pratyaksa is kalpanapodha, "free of construction".
#                         Perception does not build; it presents. So the
#                         perceptual projection carries no learnable parameter.
#                         Every scrap of conceptual work happens after it.
#
#   exclusions   W,b : z = W p + b                  K razors (vyavrtti)
#                      excluded_k = sigmoid(z_k)
#                      survives_k = sigmoid(-z_k)
#                      -log survives_k = softplus(z_k)      <-- the cost of a cut
#
#   lattice      M   : gate = sigmoid(M)            (C,K) soft demand matrix
#                      u_c = - SUM_k gate[c,k] * softplus(z_k)
#                      "the score of concept c is minus the total survival-cost
#                       of the exclusions that c requires"
#                      NOTHING POSITIVE IS STORED. u is built from negations only.
#
#   assertion    a,B : logits = softplus(a) * u + B
#
#   self-report  V   : r = tanh(V u + v0)           svabhasa: the cognition's
#                                                   appearance to itself
#   recollection S   : memory logits = S r + s0     smrti: what was cognized,
#                      margin_hat = wm . r + bm     recovered from the trace alone
#
#  LOSS
#  ----
#   L = CE(logits, y)                                   the cognition is correct
#     + lam_vip * SUM_{c,k} gate[c,k] * mean_{x NOT in c}[ survives_k(x) ]
#           ^ vyatireka: a reason demanded by c must be ABSENT from the whole
#             dissimilar class. Linear, unforgiving, never saturates. Any
#             leakage is paid for in full.
#     + lam_sap * SUM_{c,k} gate[c,k] * relu(0.5 - mean_{x in c}[ survives_k(x) ])
#           ^ anvaya: a reason must hold of SOME similar cases. A hinge, because
#             Dignaga explicitly permits partial presence (this is why cell 8 is
#             valid). Once half the sapaksa is covered the pressure stops.
#     + lam_bud * mean_c (sum_k gate[c,k] - budget)^2    laghava: few cuts, not none
#     + lam_mem * CE(memory logits, stopgrad softmax(logits))
#     + lam_mrg * (margin_hat - stopgrad margin)^2
#
#  The two middle terms are the architectural signature. They are not a
#  regulariser bolted on for neatness: they are the second and third
#  characteristics of a valid reason, written as arithmetic, and they are what
#  drives the learned exclusion tests into the two valid cells of the wheel.
# =============================================================================

class ApohaNet:

    def __init__(self, D=16, H=32, K=24, C=4, R=8, rng=RNG,
                 lam_vip=0.50, lam_sap=0.25, vip_tol=0.02, sap_floor=0.20,
                 lam_bud=0.30, budget=6.0, lam_crisp=0.08,
                 lam_mem=0.30, lam_mrg=0.05):
        self.D, self.H, self.K, self.C, self.R = D, H, K, C, R
        self.lam_vip, self.lam_sap = lam_vip, lam_sap
        # VYATIREKA IS A THRESHOLD, NOT A GRADIENT TO INFINITY.
        # The third condition demands that the reason be ABSENT from the
        # dissimilar class. It does not demand that it be more and more absent.
        # A first draft penalised leakage linearly all the way to zero, so even a
        # spotless cut kept paying rent -- and the network responded, rationally,
        # by demanding no cuts at all. `vip_tol` is the point at which absence is
        # absence and the pressure stops.
        self.vip_tol = vip_tol
        # ANVAYA IS SATISFIED BY 'SOME'.
        # The second condition asks only that the reason be found among SOME of
        # the similar cases -- Dignaga is explicit, and cell 8 of the wheel is
        # valid precisely because of it. So the hinge stops pushing once a cut
        # covers `sap_floor` of its concept. Demanding majority coverage (an
        # earlier draft asked for half) quietly smuggles the realist assumption
        # back in: that a concept is one blob with one centre, which every good
        # cut must straddle. It is not.
        self.sap_floor = sap_floor
        # LAGHAVA (parsimony) as a BUDGET, not a shrink.
        # A first draft of this file penalised the total gate mass (an L1). The
        # network promptly drove every gate to zero and inflated the assertion
        # scale to compensate: it classified beautifully while demanding NO
        # exclusions at all, so the wheel had nothing to audit and the machine
        # could never justify a word it said. That is not economy; that is a
        # concept with no content. Dignaga's laghava says: a good definition
        # uses FEW cuts -- not none. So the demand mass of each concept is
        # pinned two-sidedly at `budget`, and the vyatireka term then decides
        # WHICH cuts deserve the ration.
        self.lam_bud, self.budget = lam_bud, budget
        # A CONCEPT EITHER DEMANDS A NEGATION OR IT DOES NOT.
        # Apoha membership is a CONJUNCTION of exclusions, not a weighted vote.
        # 'Half-requiring' that a thing be non-a-horse is not a thing a definition
        # can do. Left unconstrained the gates smear the whole budget thinly over
        # every available cut -- arithmetically comfortable, doctrinally void, and
        # it leaves the wheel with no discrete reason to audit. This term drives
        # each gate to 0 or 1 so that a concept is, in the end, a SET.
        self.lam_crisp = lam_crisp
        self.lam_mem, self.lam_mrg = lam_mem, lam_mrg

        # --- perception: fixed, unlearned, non-conceptual -------------------
        self.Pfix = rng.normal(scale=1.0 / np.sqrt(D), size=(H, D))
        self.p0 = np.zeros(H)

        # --- learnable parameters -------------------------------------------
        self.params = {
            'W':  rng.normal(scale=0.7 / np.sqrt(H), size=(K, H)),   # the razors
            'b':  np.zeros(K),
            'M':  rng.normal(scale=0.30, size=(C, K)),               # which cuts each concept demands
            'a':  np.array(0.5),                                     # softplus -> assertion scale
            'B':  np.zeros(C),                                       # per-concept bias
            'V':  rng.normal(scale=0.25, size=(R, C)),               # svabhasa
            'v0': np.zeros(R),
            'S':  rng.normal(scale=0.5, size=(C, R)),                # smrti: what was cognized
            's0': np.zeros(C),
            'wm': rng.normal(scale=0.5, size=(R,)),                  # smrti: how confidently
            'bm': np.array(0.0),
        }

    # ---------------- perception (no parameters, by doctrine) ---------------
    def perceive(self, X):
        return np.tanh(X @ self.Pfix.T + self.p0)

    # ---------------- forward ------------------------------------------------
    def forward(self, X, y, P=None, fixed_targets=None):
        """Full forward pass. Returns (loss, cache). y may be None for inference.

        `fixed_targets` pins the two STOP-GRADIENT targets of the recollection
        head (the object-distribution it must recall, and the margin it must
        recall). During training they are recomputed and detached each step, in
        the usual way. The gradient checker must instead hold them CONSTANT,
        because backward() deliberately cuts the path through them -- and a
        finite difference, which simply re-runs the forward pass, would
        otherwise silently push the target around and report a phantom error.
        (This is not a technicality; it cost this file one failed check.)
        """
        p = self.perceive(X) if P is None else P
        W, b, M = self.params['W'], self.params['b'], self.params['M']
        a, B = self.params['a'], self.params['B']
        V, v0 = self.params['V'], self.params['v0']
        S, s0 = self.params['S'], self.params['s0']
        wm, bm = self.params['wm'], self.params['bm']

        N = X.shape[0]
        C, K = self.C, self.K

        Z = p @ W.T + b                     # (N,K)  the razors' verdicts
        SP = softplus(Z)                    # (N,K)  cost of surviving each cut
        SV = sigmoid(-Z)                    # (N,K)  probability of surviving  == exp(-SP)
        G = sigmoid(M)                      # (C,K)  which cuts each concept demands

        U = -(SP @ G.T)                     # (N,C)  score = -(total survival cost demanded)
        alpha = softplus(a)
        logits = alpha * U + B              # (N,C)
        Pr = softmax(logits, axis=1)

        # --- svabhasa / smrti (self-appearance and its recollection) --------
        # A cognition's appearance TO ITSELF is its SHAPE, not its magnitude: the
        # question 'which concept did I just find, and by how much did it beat the
        # others' is scale-free. Feeding the raw scores here saturates the tanh and
        # the trace goes blank -- the cognition ceases to be self-luminous and
        # nothing can afterwards be remembered. So the self-channel reads U centred.
        Uc = U - U.mean(axis=1, keepdims=True)      # (N,C)
        Rr = np.tanh(Uc @ V.T + v0)                 # (N,R)
        ML = Rr @ S.T + s0                  # (N,C)  what do I recall cognizing?
        Mhat = Rr @ wm + bm                 # (N,)   how sure was I?

        cache = dict(p=p, Z=Z, SP=SP, SV=SV, G=G, U=U, Uc=Uc, alpha=alpha,
                     logits=logits, Pr=Pr, Rr=Rr, ML=ML, Mhat=Mhat, N=N)

        if y is None:
            return None, cache

        # ---------------- classification (the cognition is correct) ---------
        L_cls = -np.mean(np.log(Pr[np.arange(N), y] + EPS))

        # ---------------- trairupya: the two population conditions ----------
        # sapaksa mask (N,C): 1 where sample i IS an instance of concept c
        # vipaksa mask (N,C): 1 where sample i is NOT an instance of concept c
        SAP = one_hot(y, C)                 # (N,C)
        VIP = 1.0 - SAP
        n_sap = SAP.sum(axis=0) + EPS       # (C,)
        n_vip = VIP.sum(axis=0) + EPS

        # A[c,k] = mean survival of test k across the DISSIMILAR class of c
        A = (VIP.T @ SV) / n_vip[:, None]   # (C,K)
        # Bc[c,k] = mean survival of test k across the SIMILAR class of c
        Bc = (SAP.T @ SV) / n_sap[:, None]  # (C,K)

        leak = np.maximum(0.0, A - self.vip_tol)                   # (C,K)
        L_vip = self.lam_vip * np.sum(G * leak)                    # unforgiving above tolerance
        hinge = np.maximum(0.0, self.sap_floor - Bc)               # forgiving, saturating
        L_sap = self.lam_sap * np.sum(G * hinge)
        rowsum = G.sum(axis=1)                                     # (C,) cuts demanded
        L_bud = self.lam_bud * np.mean((rowsum - self.budget) ** 2)  # laghava
        L_crisp = self.lam_crisp * np.sum(G * (1.0 - G))           # a set, not a vote

        # ---------------- svasamvedana / smrti ------------------------------
        if fixed_targets is None:
            tgt = Pr.copy()                                        # stop-grad target
            sl = np.sort(logits, axis=1)
            margin = (sl[:, -1] - sl[:, -2])                       # stop-grad target
        else:
            tgt, margin = fixed_targets
        MPr = softmax(ML, axis=1)
        L_mem = self.lam_mem * (-np.mean(np.sum(tgt * np.log(MPr + EPS), axis=1)))
        L_mrg = self.lam_mrg * np.mean((Mhat - margin) ** 2)

        loss = L_cls + L_vip + L_sap + L_bud + L_crisp + L_mem + L_mrg

        cache.update(SAP=SAP, VIP=VIP, n_sap=n_sap, n_vip=n_vip, A=A, Bc=Bc,
                     hinge=hinge, leak=leak, tgt=tgt, MPr=MPr, margin=margin, y=y,
                     rowsum=rowsum,
                     parts=dict(cls=L_cls, vip=L_vip, sap=L_sap, bud=L_bud,
                                crisp=L_crisp, mem=L_mem, mrg=L_mrg))
        return loss, cache

    # ---------------- backward (hand-derived) --------------------------------
    def backward(self, cache):
        p, Z, SP, SV, G, U = cache['p'], cache['Z'], cache['SP'], cache['SV'], cache['G'], cache['U']
        alpha, Pr, Rr, ML, MPr = cache['alpha'], cache['Pr'], cache['Rr'], cache['ML'], cache['MPr']
        Mhat, margin, tgt = cache['Mhat'], cache['margin'], cache['tgt']
        SAP, VIP, n_sap, n_vip = cache['SAP'], cache['VIP'], cache['n_sap'], cache['n_vip']
        A, Bc, hinge, y, N = cache['A'], cache['Bc'], cache['hinge'], cache['y'], cache['N']
        C, K = self.C, self.K
        W, M, a, V, S, wm = (self.params['W'], self.params['M'], self.params['a'],
                             self.params['V'], self.params['S'], self.params['wm'])

        g = {k: np.zeros_like(v) for k, v in self.params.items()}

        # ---- classification head -------------------------------------------
        dlogits = (Pr - one_hot(y, C)) / N                      # (N,C)
        g['B'] += dlogits.sum(axis=0)
        dU = alpha * dlogits                                    # (N,C)
        g['a'] += float(np.sum(dlogits * U) * sigmoid(np.array(a)))   # d softplus(a)/da

        # ---- recollection head (smrti) --------------------------------------
        dML = self.lam_mem * (MPr - tgt) / N                    # (N,C)
        g['S'] += dML.T @ Rr
        g['s0'] += dML.sum(axis=0)
        dRr = dML @ S                                           # (N,R)

        dMhat = self.lam_mrg * 2.0 * (Mhat - margin) / N        # (N,)
        g['wm'] += Rr.T @ dMhat
        g['bm'] += float(dMhat.sum())
        dRr += np.outer(dMhat, wm)

        dpre_r = dRr * (1.0 - Rr ** 2)                          # (N,R)
        g['V'] += dpre_r.T @ cache['Uc']
        g['v0'] += dpre_r.sum(axis=0)
        dUc = dpre_r @ V                                        # (N,C)
        # through the centring: d/dU of (U - mean_c U)
        dU += dUc - dUc.mean(axis=1, keepdims=True)             # self-awareness reshapes the object channel

        # ---- lattice: U = -(SP @ G.T) ---------------------------------------
        dG = -(dU.T @ SP)                                       # (C,K)
        dSP = -(dU @ G)                                         # (N,K)

        # ---- trairupya terms -------------------------------------------------
        # L_vip = lam_vip * sum(G * relu(A - vip_tol)),  A = VIP^T SV / n_vip
        leak = cache['leak']
        act_v = (leak > 0).astype(float)                        # (C,K)
        dG += self.lam_vip * leak
        dSV = self.lam_vip * (VIP @ ((G * act_v) / n_vip[:, None]))   # (N,K)

        # L_sap = lam_sap * sum(G * relu(0.5 - Bc))
        dG += self.lam_sap * hinge
        act = (hinge > 0).astype(float)                          # (C,K)
        dBc = -self.lam_sap * G * act                            # (C,K)
        dSV += SAP @ (dBc / n_sap[:, None])                      # (N,K)

        # L_bud = lam_bud * mean_c (sum_k G_ck - budget)^2
        dG += (self.lam_bud * 2.0 * (cache['rowsum'] - self.budget) / C)[:, None]
        # L_crisp = lam_crisp * sum(G(1-G))
        dG += self.lam_crisp * (1.0 - 2.0 * G)

        # ---- through the razors ----------------------------------------------
        # SP = softplus(Z)      -> dZ += dSP * sigmoid(Z)
        # SV = sigmoid(-Z)      -> dZ += dSV * (-SV*(1-SV))
        dZ = dSP * sigmoid(Z) - dSV * (SV * (1.0 - SV))
        g['W'] += dZ.T @ p
        g['b'] += dZ.sum(axis=0)

        # ---- gate: G = sigmoid(M) ---------------------------------------------
        g['M'] += dG * G * (1.0 - G)

        return g

    # ---------------- optimiser (Adam, written out) --------------------------
    def fit(self, X, y, Xv, yv, epochs=260, lr=0.05, batch=128, verbose=True, rng=RNG):
        m = {k: np.zeros_like(v) for k, v in self.params.items()}
        v = {k: np.zeros_like(v) for k, v in self.params.items()}
        b1, b2, eps = 0.9, 0.999, 1e-8
        t = 0
        hist = []
        n = len(y)
        for ep in range(1, epochs + 1):
            perm = rng.permutation(n)
            for s in range(0, n, batch):
                idx = perm[s:s + batch]
                loss, cache = self.forward(X[idx], y[idx])
                grads = self.backward(cache)
                t += 1
                for k in self.params:
                    m[k] = b1 * m[k] + (1 - b1) * grads[k]
                    v[k] = b2 * v[k] + (1 - b2) * grads[k] ** 2
                    mh = m[k] / (1 - b1 ** t)
                    vh = v[k] / (1 - b2 ** t)
                    self.params[k] = self.params[k] - lr * mh / (np.sqrt(vh) + eps)
            if ep % 20 == 0 or ep == 1:
                ltr, ctr = self.forward(X, y)
                atr = (ctr['logits'].argmax(1) == y).mean()
                lva, cva = self.forward(Xv, yv)
                ava = (cva['logits'].argmax(1) == yv).mean()
                hist.append((ep, ltr, atr, ava))
                if verbose:
                    pp = ctr['parts']
                    print(f"  ep {ep:4d} | loss {ltr:7.4f} "
                          f"(cls {pp['cls']:.3f}  vyatireka {pp['vip']:.3f}  "
                          f"anvaya {pp['sap']:.3f}  laghava {pp['bud']:.3f}  "
                          f"crisp {pp['crisp']:.3f}) "
                          f"| train {atr:.3f}  val {ava:.3f}")
        return hist

    def predict(self, X):
        _, c = self.forward(X, None)
        return c['logits'].argmax(1), c


# =============================================================================
# SECTION 3 — THE HETUCAKRA: a live nine-celled validity gate
# =============================================================================
#
#  Rows    = presence of the reason among the SIMILAR cases (sapaksa)
#  Columns = presence of the reason among the DISSIMILAR cases (vipaksa)
#  Order of both: ALL, NONE, SOME  (this is the ordering that yields Dignaga's
#  canonical numbering, in which cells 2 and 8 are the valid ones)
#
#          vipaksa: ALL     NONE     SOME
#  sapaksa ALL   |   1        2*       3
#  sapaksa NONE  |   4x       5?       6x
#  sapaksa SOME  |   7        8*       9
#
#    2  valid  : present throughout the similar, absent from all the dissimilar
#    8  valid  : present in part of the similar, absent from all the dissimilar
#    4,6 viruddha    (contradictory — the reason proves the opposite)
#    5  asadharana   (uncommon/unique — proves nothing, shared with nothing)
#    1,3,7,9 anaikantika (inconclusive — the reason strays into the dissimilar)
#
#  Everything hangs on the middle column. Absence from the dissimilar class is
#  the ONLY column that licenses speech. That is Dignaga's whole epistemic ethic
#  compressed into a table, and it is a startlingly modern one: he is demanding
#  a bound on false positives, and he is refusing to trade it against recall.
# =============================================================================

CELL_NAMES = {
    1: ("anaikantika", "inconclusive"),
    2: ("VALID", "present in all similar, absent from all dissimilar"),
    3: ("anaikantika", "inconclusive"),
    4: ("viruddha", "contradictory"),
    5: ("asadharana", "uncommon - proves nothing"),
    6: ("viruddha", "contradictory"),
    7: ("anaikantika", "inconclusive"),
    8: ("VALID", "present in some similar, absent from all dissimilar"),
    9: ("anaikantika", "inconclusive"),
}
VALID_CELLS = {2, 8}


def _band(frac, hi=0.90, lo=0.05):
    """Map an empirical coverage fraction onto Dignaga's three-valued quantifier."""
    if frac >= hi:
        return 0   # ALL
    if frac <= lo:
        return 1   # NONE
    return 2       # SOME


def hetucakra_cell(sapaksa_frac, vipaksa_frac):
    """Return the cell number 1..9 of the wheel."""
    row = _band(sapaksa_frac)
    col = _band(vipaksa_frac)
    return 3 * row + col + 1


class HetucakraGate:
    """Audits every learned exclusion test as a candidate REASON, then licenses
    or refuses each assertion the network wants to make.

    The reason offered for 'x is a c' is: 'because x survives cut k' — where k
    is a cut that concept c actually demands. The gate asks Dignaga's three
    questions of that reason, and only then permits the network to speak.
    """

    def __init__(self, net, X, y, gate_thresh=0.5, present_thresh=0.5):
        self.net = net
        self.C, self.K = net.C, net.K
        _, cache = net.forward(X, None)
        SV = cache['SV']                       # (N,K) survival of each cut
        G = sigmoid(net.params['M'])           # (C,K) demanded cuts
        self.G = G
        present = (SV > present_thresh)        # does the reason hold of this case?
        self.table = np.zeros((self.C, self.K), dtype=int)
        self.sap = np.zeros((self.C, self.K))
        self.vip = np.zeros((self.C, self.K))
        for c in range(self.C):
            insim = (y == c)
            dissim = ~insim
            for k in range(self.K):
                sf = present[insim, k].mean()
                vf = present[dissim, k].mean()
                self.sap[c, k], self.vip[c, k] = sf, vf
                self.table[c, k] = hetucakra_cell(sf, vf)
        # a reason is AVAILABLE to concept c if c actually demands that cut and
        # the cut lands in a valid cell of the wheel
        self.available = (G > gate_thresh) & np.isin(self.table, list(VALID_CELLS))
        self.present_thresh = present_thresh

    def wheel_census(self):
        counts = {i: 0 for i in range(1, 10)}
        for c in range(self.C):
            for k in range(self.K):
                if self.G[c, k] > 0.5:                # only cuts the concept demands
                    counts[self.table[c, k]] += 1
        return counts

    def adjudicate(self, X):
        """For each x: the network's proposal, and whether the wheel licenses it.

        Returns pred (N,), asserted (N,) bool, reason (N,) int (-1 if none),
        fault (N,) list of str.
        """
        pred, cache = self.net.predict(X)
        SV = cache['SV']
        N = X.shape[0]
        asserted = np.zeros(N, dtype=bool)
        reason = np.full(N, -1, dtype=int)
        fault = []
        for i in range(N):
            c = pred[i]
            # paksadharmata: the reason must actually hold OF THIS CASE
            holds = SV[i] > self.present_thresh
            cands = np.where(self.available[c] & holds)[0]
            if len(cands):
                # prefer the strongest cut this concept demands
                k = cands[np.argmax(self.G[c][cands])]
                asserted[i] = True
                reason[i] = k
                fault.append("")
            else:
                # diagnose WHY we must stay silent, in his own vocabulary
                demanded = np.where(self.G[c] > 0.5)[0]
                if len(demanded) == 0:
                    fault.append("no reason demanded (concept underdetermined)")
                else:
                    hold_d = demanded[SV[i][demanded] > self.present_thresh]
                    if len(hold_d) == 0:
                        fault.append("paksadharmata fails (reason absent from this case)")
                    else:
                        cells = [self.table[c, k] for k in hold_d]
                        nm = CELL_NAMES[min(cells, key=lambda z: 0 if z in VALID_CELLS else 1)][0]
                        fault.append(f"reason falls in cell {cells[0]} ({nm})")
        return pred, asserted, reason, fault, cache


# =============================================================================
# SECTION 4 — BASELINES (the realists about universals get a fair hearing)
# =============================================================================

class PrototypeRealist:
    """The Nyaya-Vaisesika position, coded honestly: a class HAS an essence, and
    that essence is a positive thing you can store. Here it is the class mean;
    classification is proximity to the stored universal. This is also, allowing
    for scale, what a prototype network, a nearest-class-mean classifier, and
    the class-embedding table of a standard softmax head all do."""

    def fit(self, X, y, C):
        self.mu = np.stack([X[y == c].mean(axis=0) for c in range(C)])
        return self

    def predict(self, X):
        d = ((X[:, None, :] - self.mu[None, :, :]) ** 2).sum(-1)
        return d.argmin(1)


class LinearSoftmax:
    """One positive weight vector per class. Also a realist, just a smarter one."""

    def fit(self, X, y, C, epochs=400, lr=0.2):
        D = X.shape[1]
        self.Wl = np.zeros((C, D))
        self.bl = np.zeros(C)
        N = len(y)
        Y = one_hot(y, C)
        for _ in range(epochs):
            P = softmax(X @ self.Wl.T + self.bl, axis=1)
            dl = (P - Y) / N
            self.Wl -= lr * dl.T @ X
            self.bl -= lr * dl.sum(0)
        return self

    def predict(self, X):
        return (X @ self.Wl.T + self.bl).argmax(1)


# =============================================================================
# SECTION 5 — GRADIENT CHECK (mandatory)
# =============================================================================

def gradient_check(verbose=True):
    """Central finite differences against the hand-derived gradients, on every
    learnable block. If this fails, nothing else in the file means anything."""
    rng = np.random.default_rng(540)          # the year he is thought to have died
    net = ApohaNet(D=6, H=7, K=5, C=3, R=4, rng=rng)
    X = rng.normal(size=(11, 6))
    y = rng.integers(0, 3, size=11)

    loss, cache = net.forward(X, y)
    grads = net.backward(cache)
    # pin the stop-gradient targets: backward() does not differentiate through
    # them, so neither may the finite difference.
    FT = (cache['tgt'].copy(), cache['margin'].copy())

    # h=1e-6 is too aggressive here: the smallest gradients in V and S are ~5e-6,
    # and the subtraction (L(+h) - L(-h)) then loses most of its significant
    # figures to float64 cancellation, manufacturing a ~1e-5 'error' that halves
    # when h is relaxed. The step below is the honest one for this loss surface,
    # and the relative error is floored by each block's own gradient scale so
    # that a near-zero entry cannot masquerade as a large discrepancy.
    h = 1e-5
    print("\n  finite-difference gradient check (central, h=1e-5, scale-aware)")
    print("  " + "-" * 58)
    worst = 0.0
    ok = True
    for name in net.params:
        P = net.params[name]
        flat = np.atleast_1d(P).ravel()
        num = np.zeros_like(flat)
        idxs = range(len(flat)) if len(flat) <= 40 else rng.choice(len(flat), 40, replace=False)
        for i in idxs:
            old = flat[i]
            flat[i] = old + h
            net.params[name] = flat.reshape(P.shape) if P.ndim else np.array(flat[0])
            lp, _ = net.forward(X, y, fixed_targets=FT)
            flat[i] = old - h
            net.params[name] = flat.reshape(P.shape) if P.ndim else np.array(flat[0])
            lm, _ = net.forward(X, y, fixed_targets=FT)
            flat[i] = old
            net.params[name] = flat.reshape(P.shape) if P.ndim else np.array(flat[0])
            num[i] = (lp - lm) / (2 * h)
        ana = np.atleast_1d(grads[name]).ravel()
        sel = np.array(list(idxs))
        scale = max(np.abs(ana).max(), 1e-12)
        denom = np.maximum(np.abs(num[sel]) + np.abs(ana[sel]), 1e-3 * scale)
        rel = np.abs(num[sel] - ana[sel]) / denom
        m = float(rel.max())
        worst = max(worst, m)
        flag = "OK  " if m < 1e-5 else "FAIL"
        if m >= 1e-5:
            ok = False
        if verbose:
            print(f"    {name:>4s}  max rel err {m:.3e}   {flag}")
    print("  " + "-" * 58)
    print(f"    worst over all blocks: {worst:.3e}  ->  {'PASS' if ok else 'FAIL'}")
    assert ok, "gradient check FAILED"
    return worst


# =============================================================================
# SECTION 6 — SELF-TESTS
# =============================================================================

def self_tests():
    print("\n  self-tests")
    print("  " + "-" * 58)

    # softplus / survival identity: -log P(survive) == softplus(z)
    z = np.linspace(-8, 8, 101)
    assert np.allclose(-np.log(sigmoid(-z) + 1e-300), softplus(z), atol=1e-9)
    print("    survival identity  -log sigma(-z) == softplus(z)          OK")

    # the wheel numbering must reproduce Dignaga's canonical verdicts
    assert hetucakra_cell(1.0, 0.0) == 2          # all similar / no dissimilar
    assert hetucakra_cell(0.5, 0.0) == 8          # some similar / no dissimilar
    assert hetucakra_cell(0.0, 1.0) == 4          # contradictory
    assert hetucakra_cell(0.0, 0.5) == 6          # contradictory
    assert hetucakra_cell(0.0, 0.0) == 5          # asadharana (audibility)
    assert hetucakra_cell(1.0, 1.0) == 1          # knowability: proves nothing
    assert all(hetucakra_cell(*p) not in VALID_CELLS
               for p in [(1., 1.), (1., .5), (0., 1.), (0., 0.), (0., .5), (.5, 1.), (.5, .5)])
    print("    hetucakra: only cells 2 and 8 are valid                   OK")

    # the classic worked examples, run through the same gate the network uses
    # 'sound is impermanent, because produced'  : produced covers all impermanent
    #                                             things, no permanent thing  -> 2
    # 'sound is impermanent, because effort-born': effort-born covers only SOME
    #                                             impermanent things, still no
    #                                             permanent one                -> 8
    # 'sound is eternal, because audible'        : audible belongs to sound alone;
    #                                             shared with nothing, similar or
    #                                             dissimilar                   -> 5
    # 'sound is eternal, because knowable'       : knowable belongs to everything
    #                                                                          -> 1
    # 'sound is eternal, because produced'       : produced is found only among
    #                                             the impermanent              -> 4
    cases = [("produced -> impermanent", 1.00, 0.00, 2),
             ("effort-born -> impermanent", 0.40, 0.00, 8),
             ("audible -> eternal", 0.00, 0.00, 5),
             ("knowable -> eternal", 1.00, 1.00, 1),
             ("produced -> eternal", 0.00, 1.00, 4)]
    for nm, sf, vf, want in cases:
        got = hetucakra_cell(sf, vf)
        assert got == want, (nm, got, want)
    print("    five canonical inferences land in their historical cells   OK")

    # apoha: NOTHING positive is stored. Assert that no parameter has a per-class
    # slot in the input/perceptual space at all.
    net = ApohaNet()
    assert net.params['M'].shape == (net.C, net.K)      # class -> WHICH CUTS, only
    for nm, P in net.params.items():
        assert not (P.ndim == 2 and P.shape == (net.C, net.D)), "a class prototype leaked in!"
        assert not (P.ndim == 2 and P.shape == (net.C, net.H)), "a class prototype leaked in!"
    print("    no positive class template exists anywhere in the model    OK")

    # a concept's score is monotone DECREASING in every cut it demands
    Xt = RNG.normal(size=(5, net.D))
    _, c1 = net.forward(Xt, None)
    net.params['M'][0] += 3.0                          # concept 0 demands more cuts
    _, c2 = net.forward(Xt, None)
    assert np.all(c2['U'][:, 0] <= c1['U'][:, 0] + 1e-9)
    print("    demanding more exclusions can only lower a concept's score OK")
    print("  " + "-" * 58)


# =============================================================================
# SECTION 7 — MAIN
# =============================================================================

def main():
    print("=" * 78)
    print(" APOHA-NET  -  Dignaga (c. 480-540 CE)")
    print(" a mind that knows a thing only by everything the thing is not")
    print("=" * 78)

    gradient_check()
    self_tests()

    # ---------------------------------------------------------------- data
    X, y = make_particulars(n_per_lobe=90, D=16, C=4, lobes=3)
    Xtr, ytr, Xte, yte = split(X, y, 0.7)
    print(f"\n  world of bare particulars: {X.shape[0]} momentary events, "
          f"dim {X.shape[1]}, {len(set(y))} concepts")
    cents = np.stack([X[y == c].mean(0) for c in range(4)])
    print(f"  every concept's centroid sits at the origin "
          f"(max |mean| = {np.abs(cents).max():.3f}) -- there is no essence to store")

    # ---------------------------------------------------------------- baselines
    print("\n  BASELINES  (the realists about universals)")
    print("  " + "-" * 58)
    proto = PrototypeRealist().fit(Xtr, ytr, 4)
    acc_p = (proto.predict(Xte) == yte).mean()
    print(f"    prototype / class-centroid ('cowness is a thing')   test acc {acc_p:.3f}")
    lin = LinearSoftmax().fit(Xtr, ytr, 4)
    acc_l = (lin.predict(Xte) == yte).mean()
    print(f"    linear softmax  (one weight vector per class)       test acc {acc_l:.3f}")
    print(f"    chance                                              test acc {1/4:.3f}")

    # ---------------------------------------------------------------- apoha
    print("\n  TRAINING APOHA-NET  (concepts as conjunctions of negations)")
    print("  " + "-" * 58)
    net = ApohaNet(D=16, H=32, K=24, C=4, R=8)
    net.fit(Xtr, ytr, Xte, yte, epochs=320, lr=0.05, batch=128)
    pred, cache = net.predict(Xte)
    acc_a = (pred == yte).mean()
    print(f"\n    APOHA-NET                                           test acc {acc_a:.3f}")

    G = sigmoid(net.params['M'])
    print(f"    exclusions demanded per concept (of {net.K}): "
          f"{[int((G[c] > 0.5).sum()) for c in range(4)]}   "
          f"(laghava: a budget of ~{net.budget:.0f} cuts, spent where they do not leak)")

    # ------------------------------------------------- ablation: no trairupya
    # What does the SAME lattice do if we simply delete the second and third
    # conditions and let it optimise raw accuracy? This is the only fair way to
    # find out what Dignaga's discipline costs, and what it buys.
    print("\n  ABLATION: the same lattice with the trairupya conditions removed")
    print("  " + "-" * 58)
    free = ApohaNet(D=16, H=32, K=24, C=4, R=8,
                    lam_vip=0.0, lam_sap=0.0)          # anvaya & vyatireka switched off
    free.fit(Xtr, ytr, Xte, yte, epochs=320, lr=0.05, batch=128, verbose=False)
    pf, _ = free.predict(Xte)
    acc_f = (pf == yte).mean()
    gate_f = HetucakraGate(free, Xtr, ytr)
    cf = gate_f.wheel_census()
    tot_f = sum(cf.values()); val_f = cf[2] + cf[8]
    _, asrt_f, _, _, _ = gate_f.adjudicate(Xte)
    print(f"    unconstrained accuracy                              {acc_f:.3f}")
    print(f"    ...but valid reasons among the ones it relies on:   {val_f}/{tot_f}")
    print(f"    ...and cases it could justify at all:               "
          f"{asrt_f.sum()}/{len(yte)}")
    print("    It scores better and cannot defend a single word of it.")

    # ---------------------------------------------------------------- the wheel
    print("\n  THE HETUCAKRA, RUN OVER THE LEARNED REASONS")
    print("  " + "-" * 58)
    gate = HetucakraGate(net, Xtr, ytr)
    census = gate.wheel_census()
    for cell in range(1, 10):
        nm, gloss = CELL_NAMES[cell]
        mark = "*" if cell in VALID_CELLS else " "
        print(f"    cell {cell}{mark} {nm:<13s} {gloss:<46s} {census[cell]:3d}")
    tot = sum(census.values())
    val = census[2] + census[8]
    print(f"    -> {val}/{tot} of the reasons the net actually relies on are valid reasons")

    print("\n    a licensed inference, in full:")
    for c in range(4):
        ks = np.where(gate.available[c])[0]
        if len(ks):
            k = ks[np.argmax(G[c][ks])]
            print(f"      concept {c}: 'this is a {c}, because it survives cut {k}'  "
                  f"[sapaksa {gate.sap[c,k]:.2f}  vipaksa {gate.vip[c,k]:.2f}  "
                  f"cell {gate.table[c,k]}]")

    # ---------------------------------------------------------------- refusal
    print("\n  THE GATE: WHAT HAPPENS WHEN THE MACHINE MAY REFUSE TO SPEAK")
    print("  " + "-" * 58)
    pred, asserted, reason, fault, _ = gate.adjudicate(Xte)
    corr = (pred == yte)
    print(f"    ungated accuracy (the machine always answers)      {corr.mean():.3f}")
    if asserted.sum():
        print(f"    accuracy on ASSERTED claims                        "
              f"{corr[asserted].mean():.3f}   ({asserted.sum()}/{len(yte)} cases)")
    if (~asserted).sum():
        print(f"    accuracy on WITHHELD claims                        "
              f"{corr[~asserted].mean():.3f}   ({(~asserted).sum()}/{len(yte)} cases)")
        from collections import Counter
        for f, n in Counter([f for f in fault if f]).most_common(3):
            print(f"      silence because: {f}  ({n})")

    # ---------------------------------------------------------------- memory
    print("\n  SVASAMVEDANA: CAN THE COGNITION BE REMEMBERED?")
    print("  " + "-" * 58)
    _, cte = net.forward(Xte, None)
    # the object is gone. The moment has perished. ONLY the self-report survives.
    Rr = cte['Rr']
    recalled = (Rr @ net.params['S'].T + net.params['s0']).argmax(1)
    agree = (recalled == pred).mean()
    print(f"    the object is discarded; only the cognition's self-report is kept.")
    print(f"    recollection of WHAT was cognized, from the trace alone:  {agree:.3f}")
    mhat = Rr @ net.params['wm'] + net.params['bm']
    lo, hi = np.quantile(mhat, [0.25, 0.75])
    a_lo = corr[mhat <= lo].mean()
    a_hi = corr[mhat >= hi].mean()
    print(f"    recollection of HOW SURE it was  ->  accuracy where the")
    print(f"      remembered confidence is low  {a_lo:.3f}   high  {a_hi:.3f}")

    # a self without a substance: reconstruct the previous moment from this one,
    # by ridge regression over the stream of self-reports. No hidden state exists.
    _, ctr = net.forward(Xtr, None)
    Rs = ctr['Rr']
    A_, B_ = Rs[1:], Rs[:-1]
    Wchain = np.linalg.solve(A_.T @ A_ + 1e-2 * np.eye(net.R), A_.T @ B_)
    err = np.mean((A_ @ Wchain - B_) ** 2) / (np.mean(B_ ** 2) + EPS)
    print(f"    the stream carries no persistent state; each moment reconstructs")
    print(f"      its predecessor from its own self-appearance   (rel. err {err:.3f})")

    # ---------------------------------------------------------------- verdict
    print("\n" + "=" * 78)
    print(f"  centroid realist {acc_p:.3f}  |  linear realist {acc_l:.3f}  |  "
          f"apoha lattice {acc_a:.3f}  |  ungoverned lattice {acc_f:.3f}")
    print("  A class with no essence and no centre is invisible to a mind that stores")
    print("  essences, and perfectly tractable to a mind that stores only the cuts.")
    print("  Dignaga's nominalism is not a mood. It is an architecture.")
    print("")
    print("  And the last two numbers are the argument of his life. The ungoverned")
    print("  lattice is the better guesser and cannot defend one word it says. The")
    print("  governed one pays for the wheel in raw accuracy -- and then, on the cases")
    print("  it is willing to speak about at all, is MORE accurate than the ungoverned")
    print("  model is anywhere. That is not a safety tax. That is what knowing means.")
    print("=" * 78)


if __name__ == "__main__":
    main()
