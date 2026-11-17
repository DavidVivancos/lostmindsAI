#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 # BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0385 - CHRISTINE DE PIZAN (1364 - c.1430)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 20, Minds 381-400 Available on Amazon https://www.amazon.com/dp/B0HRS9ZHVG
# Resume and Interactive Demos at  https://artificiology.com/
# Videos of the playable minds at https://www.youtube.com/@LostMindsAI
# END ATTRIBUTION  
 

 ARCHITECTURE:  LE PLAID SANS PARTIE  ("The Unopposed Pleading")
                An ex-parte-corpus reconstruction engine.
================================================================================

WHY THIS ARCHITECTURE AND NOT A TRANSFORMER

Christine de Pizan's one irreducible cognitive claim is not "the sources are
biased."  It is sharper and much more uncomfortable:

    "women did not write these books nor include the material which attacks
     them... Those who plead their cause in the absence of an opponent can
     invent to their heart's content... aggressors are always quick to attack
     those who have no means of defence."
        -- Le Livre de la Cite des Dames I.2, trans. Rosalind Brown-Grant

That is a claim about the SELECTION MECHANISM THAT PRODUCED THE CORPUS, not
about the reliability of any witness inside it.  A judgment entered when only
one party appeared is void however many pages the appearing party filed, and
however consistent those pages are with each other.  Agreement among the
authorities is not corroboration; under an ex parte record it is the exact
signature of the missing respondent.

So the model below is NOT a next-token predictor over a corpus.  Its first
object of study is the CENSOR.  It jointly estimates (a) the latent merit of
every subject and (b) the recording rule that decided which of their deeds were
ever written down, from data that is Missing Not At Random by construction.

The architecture has five organs, each one a documented move of hers:

  1. CHAMP DES ESCRIPTURES  ("the field of letters")
     The corpus generator.  Deeds happen; a scribe decides which get written.
     For the favoured class, successes are recorded and failures are dropped;
     for the disfavoured class the thumb presses the other way.  The number of
     UNRECORDED deeds is not observable -- nobody counted the women whose lives
     went unwritten -- which is exactly what makes the problem hard.

  2. LE GREFFE  ("the registry / clerk of the court")
     The selection model.  A per-(class, domain) log recording-odds ratio
     delta.  This is the thumb on the scale, estimated rather than assumed
     away.

  3. LE TEMOING PROPRE  ("one's own witness")
     A tiny set of subjects known first-hand, without mediation.  Section 4
     below proves that WITHOUT THIS ANCHOR THE PROBLEM IS EXACTLY
     UNIDENTIFIABLE: merit and censor-thumb enter the likelihood only through
     their sum, so no quantity of corpus, however vast, can separate them.
     This is the opening scene of the Cite in algebra.  Christine, reading
     Matheolus, weighed thousands of pages of authority against her own direct
     knowledge of herself and the women she knew -- and discarded her own
     knowledge.  She writes that she came to hold herself and her whole sex in
     contempt.  Lady Reason's correction is not "believe yourself instead"; it
     is that the first-person datum is the ONE admissible fact that makes the
     rest of the record legible at all.

  4. LES TROIS ASSESSEURS + LA MURAILLE  (Reason, Rectitude, Justice; the wall)
     Christine does not refute; she BUILDS.  Vetted cases become load-bearing
     masonry.  Three assessors gate admission, each holding the instrument the
     text gives her: Reason a MIRROR (how badly does the record misrepresent
     this case?), Rectitude a RULER/straight-edge (how bent is the chain of
     testimony -- eyewitness or copy-of-a-copy?), Justice a MEASURING VESSEL
     (does this cell of the world get its due portion of the wall?).
     Admission is CONJUNCTIVE -- a product of three sigmoids, so any one
     assessor can veto -- under a hard capacity budget, because a wall you can
     defend is finite.
     The wall then does real work: it is the support set from which the censor
     is estimated for DOMAINS WHERE NO FIRST-PERSON WITNESS EXISTS.  That is
     why the Cite admits warriors and scholars and inventors and prophets and
     not only saints: a stone in every quarter, so that any future slander in
     any quarter has something to answer it.

  5. LES TROIS VERTUS  (the deployment policy)
     The Cite (1405) proves capacity.  The Livre des Trois Vertus, finished the
     same year, is the operations manual for acting inside a system that has
     NOT been corrected -- estate by estate, from queens to servants to
     prostitutes.  So the last organ is a small policy trained by REINFORCE in
     an environment where the censor is still running.  Correct belief does not
     by itself buy you the right action when you have no standing.

WHAT IS DELIBERATELY NOT HERE
  No attention over stored keys as the primary mechanism, no MoE, no
  transformer block.  Pure NumPy, hand-derived gradients, finite-difference
  verified.  The one attention-like operation in the model (wall readout) is a
  bounded, gated, capacity-limited retrieval over EXAMINED exemplars, which is
  the point rather than a convenience.

DIVERGENCE FROM NEIGHBOURING MINDS IN THE CORPUS (deliberate)
  Anna Comnena (237) tags each witness by its stake.  Al-Biruni (223) counts
  how many witnesses are actually independent.  Al-Bukhari (207) and Muslim
  (210) gate admission by contact and role.  Sturla (271) secures an assertion
  by whoever could have contradicted it.  Every one of those operates ON THE
  WITNESSES PRESENT.  Christine's defect is upstream of all of them: the
  witnesses who would have contradicted never wrote, so no per-witness
  operation can reach it.  Margery Kempe (302) refuses self-authentication;
  Christine restores the first-person datum to admissibility and then proves it
  is load-bearing.

RUN:  python3 chapter_0385_christine_de_pizan_1364.py
      python3 chapter_0385_christine_de_pizan_1364.py --quick
================================================================================
"""

import argparse
import sys
import numpy as np

# ------------------------------------------------------------------------------
# small numerics helpers
# ------------------------------------------------------------------------------

EPS = 1e-9


def sigmoid(x):
    """Numerically stable logistic."""
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[neg])
    out[neg] = ex / (1.0 + ex)
    return out


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def softmax(z, axis=-1):
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / (np.sum(e, axis=axis, keepdims=True) + EPS)


# ==============================================================================
# 1. CHAMP DES ESCRIPTURES  --  the field of letters (data generating process)
# ==============================================================================

class ChampDesEscriptures:
    """
    Generates a corpus that is Missing Not At Random *by design*.

    Ground truth
      - every subject i has class g_i (0 = the class that holds the pen,
        1 = the class that is written about) and domain d_i.
      - merit m*_i is drawn from the SAME distribution for both classes.
        There is no real gap.  Any gap a reader finds was manufactured.
      - each subject performs T_i deeds; a deed succeeds with prob sigma(m*_i).

    Every domain carries four OBSERVABLE descriptors, which a fifteenth-century
    reader could also have observed without reading a single book:
        x0  how public the activity is (private household ... open field)
        x1  how contested the ground is (uncontroversial ... bitterly disputed)
        x2  how far it lies inside clerical jurisdiction
        x3  how physical it is (letters ... arms)
    The censor's thumb is a smooth function of those descriptors:
        delta_true[g,d] = c_g + b_g . x_d + small noise
    This matters enormously.  It means the slant is not an arbitrary list of
    prejudices to be memorised one by one; it has STRUCTURE, and structure is
    transferable.  That is the whole warrant for building a wall whose stones
    are drawn from every quarter.

    The scribe
      - records a success with prob r_plus[g,d], a failure with r_minus[g,d],
        where log(r_plus/r_minus) = delta_true[g,d].
      - NOTHING records how many deeds went unwritten.  T is NOT observable.
        Nobody counted the lives that were never described.

    Chain of testimony
      each surviving record has a derivation depth (0 = eyewitness, 1..3 = copy
      of a copy).  Deeper chains are additionally worn by transmission.

    First-person witnesses
      a small subset is marked `anchor`: their merit is known directly, without
      passing through a scribe.  Crucially, ANCHORS EXIST ONLY IN SOME DOMAINS.
      There are quarters of any archive where no unmediated testimony from the
      written-about class survives at all, and those are the held-out domains.
    """

    def __init__(self, n_subjects=900, n_domains=8, seed=0,
                 anchor_frac=0.14, held_out_domains=(6, 7)):
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.N = n_subjects
        self.D = n_domains
        self.G = 2
        self.held_out_domains = set(held_out_domains)

        # --- observable domain descriptors -------------------------------------
        self.X = rng.uniform(-1.0, 1.0, size=(self.D, 4))

        # --- subject attributes -------------------------------------------------
        self.g = rng.integers(0, 2, size=self.N)
        self.d = rng.integers(0, self.D, size=self.N)
        self.depth = rng.integers(0, 4, size=self.N)

        # --- true merit: identical law for both classes -------------------------
        domain_mu = rng.normal(0.0, 0.40, size=self.D)
        self.m_true = domain_mu[self.d] + rng.normal(0.0, 0.85, size=self.N)

        # --- the censor: a structured, learnable slant ---------------------------
        c_g = np.array([+0.45, -1.15])
        b_g = np.array([[0.18, -0.22, 0.10, 0.14],
                        [-0.55, -0.62, -0.38, -0.48]])
        self.delta_true = (c_g[:, None] + b_g @ self.X.T
                           + rng.normal(0.0, 0.08, size=(self.G, self.D)))

        base = rng.uniform(0.25, 0.55, size=(self.G, self.D))
        self.r_plus = np.clip(base * np.exp(self.delta_true / 2), 0.02, 0.97)
        self.r_minus = np.clip(base * np.exp(-self.delta_true / 2), 0.02, 0.97)

        # --- generate deeds and their partial record ----------------------------
        T = rng.integers(40, 110, size=self.N)
        self.T_hidden = T                       # never shown to any estimator
        p = sigmoid(self.m_true)
        n_succ = rng.binomial(T, p)
        n_fail = T - n_succ
        self.n_plus = rng.binomial(n_succ, self.r_plus[self.g, self.d])
        self.n_minus = rng.binomial(n_fail, self.r_minus[self.g, self.d])

        # transmission wear on deep chains
        wear = 0.9 * (1.0 + 0.75 * self.depth)
        self.n_plus = np.maximum(0, self.n_plus + rng.normal(0, wear).round().astype(int))
        self.n_minus = np.maximum(0, self.n_minus + rng.normal(0, wear).round().astype(int))

        self.n_tot = self.n_plus + self.n_minus
        self.observed = self.n_tot > 0

        # --- train / held-out split ---------------------------------------------
        idx = np.where(self.observed)[0]
        self.train_idx = np.array([i for i in idx if self.d[i] not in self.held_out_domains])
        self.heldout_idx = np.array([i for i in idx if self.d[i] in self.held_out_domains])

        # --- claimed first-person witnesses ------------------------------------
        # These are the candidate stones.  They are NOT all good.  Christine's
        # exempla come overwhelmingly from Boccaccio's De mulieribus claris and
        # are frequently legendary; part of her labour is deciding which cases
        # will bear weight.  So the pool contains three kinds of defect:
        #   * honest but noisy   -- noise grows with the derivation depth of the
        #                           record the witness is attached to
        #   * CORRUPT            -- a fraction are fabricated or legendary and
        #                           report a merit that is simply wrong
        #   * unevenly spread    -- witnesses cluster in the loud cells of the
        #                           archive and are absent from the quiet ones
        n_w = max(24, int(anchor_frac * self.N))
        n_w = min(n_w, len(self.train_idx))
        # Witnesses do NOT arrive evenly.  They cluster where the archive is
        # already loud, because that is where survival, patronage and copying
        # concentrate: the quarters with the most paper also keep the most
        # first-hand testimony.  So the pool is drawn with probability rising in
        # the cell's existing record mass.  A wall built by simply accepting
        # what is nearest will inherit exactly this skew, which is Justice's
        # whole reason for holding a measuring vessel.
        cm = np.zeros((self.G, self.D))
        np.add.at(cm, (self.g, self.d), self.n_tot)
        w_draw = cm[self.g[self.train_idx], self.d[self.train_idx]] ** 1.25
        w_draw = w_draw / w_draw.sum()
        self.witness_idx = rng.choice(self.train_idx, size=n_w, replace=False, p=w_draw)
        dep = self.depth[self.witness_idx]
        noise = rng.normal(0.0, 0.12 * (1.0 + 1.9 * dep))
        self.witness_corrupt = rng.random(n_w) < 0.22
        fabrication = rng.normal(0.0, 1.7, size=n_w) * self.witness_corrupt
        self.witness_m = self.m_true[self.witness_idx] + noise + fabrication
        # kept only for reporting; no estimator is allowed to look at it
        self._corrupt_truth = self.witness_corrupt.copy()

    def observed_logit(self):
        """logit(n_plus / n_tot): the record as it stands, unexamined."""
        q = (self.n_plus + 0.5) / (self.n_tot + 1.0)
        return logit(q)

    def summary(self):
        return dict(
            subjects=self.N, domains=self.D,
            with_record=int(self.observed.sum()),
            witnesses=len(self.witness_idx),
            corrupt_witnesses=int(self.witness_corrupt.sum()),
            train=len(self.train_idx),
            heldout_no_witness=len(self.heldout_idx),
            true_gap=round(float(self.m_true[self.g == 1].mean()
                                 - self.m_true[self.g == 0].mean()), 4),
        )


# ==============================================================================
# 2. LE LISEUR DE MATHEOLUS  --  the naive reader (baseline)
# ==============================================================================

class MatheolusReader:
    """
    The reader who takes the record at face value: merit is the ratio of praise
    to blame that happened to survive.  Assumes the clerk had no thumb.  This is
    not a straw man.  It is what any estimator fitted to a corpus does by
    default, and it is what Christine herself did for the length of one
    afternoon, until she wrote that she had come to hold herself and her whole
    sex in contempt.
    """

    def __init__(self, champ):
        self.champ = champ
        self.m_hat = champ.observed_logit()

    def class_gap(self):
        c, o = self.champ, self.champ.observed
        return float(self.m_hat[o & (c.g == 1)].mean() - self.m_hat[o & (c.g == 0)].mean())

    def merit_rmse(self, idx):
        return float(np.sqrt(np.mean((self.m_hat[idx] - self.champ.m_true[idx]) ** 2)))


# ==============================================================================
# 3. LE PLAID  --  the model
# ==============================================================================

class Plaid:
    """
    Parameters
    ----------
    m       (N,)     latent merit logit per subject
    dbase   (G,)     the censor's standing thumb per class
    ddom    (G,D)    the thumb's cell-specific part -- the REGISTER
    Wdom    (4,H)    projection from observable domain descriptors to embedding
    wg      ( )      attention bonus for a stone of the querent's own class
    wR,bR / wD,bD / wJ,bJ   Reason, Rectitude, Justice
    logtau  ( )      wall attention temperature

    Note what is NOT a parameter: the thumb measured at a stone,

        z_k = logit(y_k) - witness_m_k

    is an OBSERVATION, not something the model may fit.  A stone contributes
    exactly the discrepancy that the examination of that case revealed, and
    nothing else.  This is the difference between building a wall and painting
    one.

    Forward
    -------
    (a) CORPUS.  T is unobservable, so only the ratio of surviving praise to
        surviving blame is available, and

            logit( E[ n+ / (n+ + n-) ] )  =  m_i  +  delta_{g_i, d_i}

        Merit and thumb enter ONLY AS A SUM.  TEST 3 verifies to machine
        precision that no corpus can split them.

    (b) EXAMINATION.  Claimed first-person witnesses supply m directly -- but
        only in proportion to their admission weight a_k.  A witness is not
        evidence because it exists; it is evidence because it was examined and
        passed.  This is the one place where the three assessors do their
        primary work, and it is why they are trained rather than hand-set.

    (c) WALL.  In the held-out cells there is no witness at all, so the register
        is unidentified there and must be PREDICTED:

            delta_hat(g,d) = sum_k alpha_k z_k
            alpha_k        = softmax_k[ (E_d.E_{d_k})/(sqrt(H) tau)
                                        + wg*1[g_k=g] + log a_k ]
            E_d            = tanh(x_d Wdom)      (x_d observable everywhere)

        under a capacity CEILING (a wall you cannot defend is not a wall) and a
        capacity FLOOR (a wall of three stones encloses nothing).
    """

    def __init__(self, champ, H=6, wall_budget=70, wall_floor=None, seed=1,
                 use_anchor=True, assessors=("R", "D", "J"), dbase_init=None):
        rng = np.random.default_rng(seed)
        self.c = champ
        self.H = H
        self.K = wall_budget
        self.Kmin = wall_floor if wall_floor is not None else int(0.66 * wall_budget)
        self.use_anchor = use_anchor
        self.assessors = set(assessors)

        N, D, G = champ.N, champ.D, champ.G
        # dbase_init is the reader's PRIOR CONVICTION about each class, held
        # before opening a single book.  TEST 4 measures what it is worth: with
        # an examined witness admitted, almost nothing; without one, everything.
        d0 = np.zeros(G) if dbase_init is None else np.array(dbase_init, dtype=float)
        self.p = {
            "m": rng.normal(0, 0.05, size=N),
            "dbase": d0.copy(),
            "ddom": rng.normal(0, 0.02, size=(G, D)),
            "Wdom": rng.normal(0, 0.45, size=(4, H)),
            "wg": np.array([0.5]),
            "wR": np.array([-0.5]), "bR": np.array([0.5]),
            "wD": np.array([-0.5]), "bD": np.array([0.5]),
            "wJ": np.array([0.5]), "bJ": np.array([0.0]),
            "logtau": np.array([0.0]),
        }

        self.y = (champ.n_plus + 0.5) / (champ.n_tot + 1.0)
        self.ly = logit(self.y)
        self.obs = champ.observed

        # ---- the candidate stones are the claimed witnesses --------------------
        self.cand = champ.witness_idx.copy()
        self.wm = champ.witness_m.copy()
        # z: the thumb as measured at this stone.  Observed, never fitted.
        self.z = self.ly[self.cand] - self.wm

        # ---- fixed assessor features -------------------------------------------
        self.f_D = champ.depth[self.cand].astype(np.float64)   # Rectitude's ruler
        cell_mass = np.zeros((G, D))
        np.add.at(cell_mass, (champ.g, champ.d), champ.n_tot)
        even = cell_mass.sum() / (G * D)
        self.f_J = np.clip(1.0 - cell_mass / (even + EPS), -1.0, 1.0)[
            champ.g[self.cand], champ.d[self.cand]]        # Justice's vessel

        # Frozen register buffer, refreshed once per training step.  Its first
        # value is the plain per-cell MEDIAN of what the witnesses say -- the
        # honest first pass, before any examination has been done.  Everything
        # after that is refinement.
        self.reg_target = np.zeros((G, D))
        for gg in range(G):
            for dd in range(D):
                sel = (champ.g[self.cand] == gg) & (champ.d[self.cand] == dd)
                if sel.any():
                    self.reg_target[gg, dd] = np.median(self.z[sel])

        self.hist = []

    # ---- flat-vector plumbing for the finite-difference check ----------------
    def keys(self):
        return list(self.p.keys())

    def get_flat(self):
        return np.concatenate([np.asarray(self.p[k], dtype=np.float64).ravel()
                               for k in self.keys()])

    def set_flat(self, v):
        o = 0
        for k in self.keys():
            a = np.asarray(self.p[k]); n = a.size
            self.p[k] = v[o:o + n].reshape(a.shape).copy(); o += n

    # ---------------------------------------------------------------------
    # THE THREE ASSESSORS  ->  ADMISSION  (conjunctive: any one may veto)
    # ---------------------------------------------------------------------
    def _admission(self, cache):
        """
        Raison holds a MIRROR: how far does this witness's own account stand
          from the reading the corrected record supports?  A story that cannot
          be reconciled with anything is not admitted, however moving.
        Droiture holds a RULER: how bent is the chain of transmission?  An
          eyewitness is straight; a copy of a copy of a copy is not.
        Justice holds a MEASURING VESSEL: is this quarter of the world getting
          its due portion of the wall, or is the wall crowding into the loud
          cells because the loud cells have the most surviving paper?
        """
        p, idx, c = self.p, self.cand, self.c
        # Reason's mirror: hold the witness's own account against the register
        # as it currently stands.  A story that cannot be reconciled with the
        # examined reading of its own quarter is not admitted, however moving --
        # and Christine's exempla, drawn largely from Boccaccio, contain a great
        # many moving stories that are not true.  The register used here is the
        # frozen buffer, so the test is against evidence rather than against a
        # free parameter that could simply move to agree with any claim.
        f_R = np.abs(self.z - self.reg_target[c.g[idx], c.d[idx]])
        cache["f_R"] = f_R

        one = np.ones_like(f_R)
        sR = sigmoid(p["wR"][0] * f_R + p["bR"][0]) if "R" in self.assessors else one
        sD = sigmoid(p["wD"][0] * self.f_D + p["bD"][0]) if "D" in self.assessors else one
        sJ = sigmoid(p["wJ"][0] * self.f_J + p["bJ"][0]) if "J" in self.assessors else one

        a = sR * sD * sJ
        S = float(a.sum())
        active = S > self.K
        scale = (self.K / S) if active else 1.0
        cache.update(dict(sR=sR, sD=sD, sJ=sJ, a=a, a_b=a * scale, S=S,
                          scale=scale, budget_active=active))
        return cache["a_b"]

    # ---------------------------------------------------------------------
    # WALL: gated attention over admitted stones -> a thumb estimate
    # ---------------------------------------------------------------------
    def _wall(self, cache, query_idx, same_domain_mask=True):
        p, c = self.p, self.c
        idx, a_b = self.cand, cache["a_b"]
        Ppre = c.X @ p["Wdom"]
        E = np.tanh(Ppre)
        Ek, Eq = E[c.d[idx]], E[c.d[query_idx]]
        tau = np.exp(p["logtau"][0])
        sc = 1.0 / (np.sqrt(self.H) * tau)
        match = (c.g[query_idx][:, None] == c.g[idx][None, :]).astype(np.float64)
        dot = Eq @ Ek.T
        logits = dot * sc + p["wg"][0] * match + np.log(a_b + 1e-6)[None, :]
        # LEAVE-ONE-QUARTER-OUT.  A stone from the querent's own domain is
        # barred, so the wall is never scored on ground it has already stood on.
        # This is not a convenience: it forces the structure to be the thing
        # that transfers, which is the entire claim being tested.
        if same_domain_mask:
            bar = (c.d[query_idx][:, None] == c.d[idx][None, :])
            logits = logits - 60.0 * bar
        alpha = softmax(logits, axis=1)
        dz = alpha @ self.z
        cache.update(dict(Ppre=Ppre, E=E, Ek=Ek, Eq=Eq, dot=dot, sc=sc,
                          match=match, alpha=alpha, dz=dz, tau=tau,
                          qi=query_idx, idx=idx))
        return dz

    # ---------------------------------------------------------------------
    # FORWARD
    # ---------------------------------------------------------------------
    def forward(self, lam_anchor=8.0, lam_wall=3.0, lam_floor=0.05, lam_reg=1e-3):
        p, c = self.p, self.c
        cache = {}
        obs, n = self.obs, c.n_tot

        # (a) corpus -------------------------------------------------------------
        eta = p["m"] + p["dbase"][c.g] + p["ddom"][c.g, c.d]
        q = sigmoid(eta)
        Zn = max(1.0, float(obs.sum()))   # per SUBJECT, so one life counts once
        ll = c.n_plus[obs] * np.log(q[obs] + EPS) + c.n_minus[obs] * np.log(1 - q[obs] + EPS)
        L_corpus = -float(ll.sum()) / Zn
        cache["q"] = q

        # admission (needed by both the examination term and the wall) ----------
        idxw = self.cand
        a_b = self._admission(cache)
        Sb = float(a_b.sum()) + EPS

        # (b) examination: witnesses count in proportion to admission -----------
        # Normalised by the SIZE OF THE POOL, not by how many were admitted, so
        # that refusing to examine anyone buys nothing.  Excluding a witness
        # costs nothing here; the pressure to admit comes from the wall, which
        # cannot stand on too few stones.
        # An admitted witness says: the thumb in my cell is z_k.  The register
        # must answer for that, weighted by how far the witness was admitted.
        # Note what this term does NOT do: it does not adjust a free per-subject
        # quantity until the complaint goes away.  It moves the REGISTER, which
        # is shared, which is why one examined life can correct a whole quarter
        # of the archive -- and why admitting a fabricated one corrupts it.
        Nc = float(len(self.cand))
        reg_k = p["dbase"][c.g[idxw]] + p["ddom"][c.g[idxw], c.d[idxw]]
        v = reg_k - self.z
        cache["v"] = v
        L_anchor = float(np.sum(a_b * v ** 2) / Nc) if self.use_anchor else 0.0
        cache["Sb"], cache["Nc"] = Sb, Nc

        # capacity floor: a wall of three stones encloses nothing
        short = max(0.0, self.Kmin - cache["S"])
        L_floor = short ** 2
        cache["short"] = short

        # (c) wall ---------------------------------------------------------------
        # The target is a FROZEN BUFFER refreshed once per training step, not a
        # live function of the parameters.  The flow is deliberately one-way:
        # the wall must be made to reproduce the examined register, and the
        # register is never revised to flatter the wall.  A structure that can
        # edit the evidence it was built from is not a structure, it is an
        # opinion with masonry.
        qi = c.train_idx
        dz = self._wall(cache, qi)
        tgt = self.reg_target[c.g[qi], c.d[qi]]
        resid_w = dz - tgt
        L_wall = float(np.mean(resid_w ** 2))
        cache["resid_w"] = resid_w

        L_reg = float(np.sum(p["Wdom"] ** 2) / p["Wdom"].size)

        L = (L_corpus + lam_anchor * L_anchor + lam_wall * L_wall
             + lam_floor * L_floor + lam_reg * L_reg)
        cache["terms"] = dict(corpus=L_corpus, exam=L_anchor, wall=L_wall,
                              floor=L_floor, reg=L_reg, total=L)
        self._cache = cache
        self._lam = (lam_anchor, lam_wall, lam_floor, lam_reg)
        return L, cache

    # ---------------------------------------------------------------------
    # BACKWARD  (hand-derived; finite-difference verified in TEST 1)
    # ---------------------------------------------------------------------
    def backward(self):
        p, c, cache = self.p, self.c, self._cache
        lam_anchor, lam_wall, lam_floor, lam_reg = self._lam
        g = {k: np.zeros_like(np.asarray(p[k], dtype=np.float64)) for k in p}

        obs, n = self.obs, c.n_tot
        Zn = max(1.0, float(obs.sum()))   # per SUBJECT, so one life counts once
        idx = self.cand
        a_b, a, S, scale, Sb = (cache["a_b"], cache["a"], cache["S"],
                                cache["scale"], cache["Sb"])

        # ---- corpus ------------------------------------------------------------
        q = cache["q"]
        deta = np.zeros(c.N)
        deta[obs] = -(c.n_plus[obs] - n[obs] * q[obs]) / Zn
        g["m"] += deta
        np.add.at(g["dbase"], c.g, deta)
        np.add.at(g["ddom"], (c.g, c.d), deta)

        da_b = np.zeros_like(a_b)

        # ---- examination -------------------------------------------------------
        if self.use_anchor:
            Nc, v = cache["Nc"], cache["v"]
            dv = lam_anchor * 2.0 * a_b * v / Nc
            np.add.at(g["dbase"], c.g[idx], dv)
            np.add.at(g["ddom"], (c.g[idx], c.d[idx]), dv)
            da_b += lam_anchor * (v ** 2) / Nc

        # ---- wall --------------------------------------------------------------
        qi = cache["qi"]
        alpha, resid_w = cache["alpha"], cache["resid_w"]
        Nq = len(qi)
        dres = lam_wall * 2.0 * resid_w / Nq
        ddz = dres.copy()
        # NOTE: no gradient flows back into dbase/ddom here -- the target is a
        # frozen buffer (see forward).  This is the one-way flow described above.

        gA = np.outer(ddz, self.z)
        dlog = alpha * (gA - np.sum(gA * alpha, axis=1, keepdims=True))

        g["wg"][0] += float(np.sum(dlog * cache["match"]))
        g["logtau"][0] += float(np.sum(dlog * (-cache["dot"] * cache["sc"])))

        sc = cache["sc"]
        dEq = (dlog @ cache["Ek"]) * sc
        dEk = (dlog.T @ cache["Eq"]) * sc
        dE = np.zeros_like(cache["E"])
        np.add.at(dE, c.d[qi], dEq)
        np.add.at(dE, c.d[idx], dEk)
        dPpre = dE * (1.0 - cache["E"] ** 2)
        g["Wdom"] += c.X.T @ dPpre

        da_b += np.sum(dlog, axis=0) / (a_b + 1e-6)

        # ---- through the capacity ceiling --------------------------------------
        if cache["budget_active"]:
            da = scale * da_b - (self.K / (S ** 2)) * float(np.sum(da_b * a))
        else:
            da = da_b.copy()
        # ---- through the capacity floor ----------------------------------------
        da += lam_floor * (-2.0 * cache["short"])

        # ---- into the three assessors ------------------------------------------
        sR, sD, sJ, f_R = cache["sR"], cache["sD"], cache["sJ"], cache["f_R"]
        if "R" in self.assessors:
            dzR = da * (sD * sJ) * sR * (1 - sR)
            g["wR"][0] += float(np.sum(dzR * f_R))
            g["bR"][0] += float(np.sum(dzR))
        if "D" in self.assessors:
            dzD = da * (sR * sJ) * sD * (1 - sD)
            g["wD"][0] += float(np.sum(dzD * self.f_D))
            g["bD"][0] += float(np.sum(dzD))
        if "J" in self.assessors:
            dzJ = da * (sR * sD) * sJ * (1 - sJ)
            g["wJ"][0] += float(np.sum(dzJ * self.f_J))
            g["bJ"][0] += float(np.sum(dzJ))

        g["Wdom"] += lam_reg * 2.0 * p["Wdom"] / p["Wdom"].size
        return g

    # ---------------------------------------------------------------------
    # TRAINING (Adam)
    # ---------------------------------------------------------------------
    def fit(self, steps=1200, lr=0.06, verbose=True, **lam):
        m1 = {k: np.zeros_like(np.asarray(self.p[k], dtype=np.float64)) for k in self.p}
        m2 = {k: np.zeros_like(np.asarray(self.p[k], dtype=np.float64)) for k in self.p}
        b1, b2, eps = 0.9, 0.999, 1e-8
        for t in range(1, steps + 1):
            # refresh the frozen register buffer (a slow blend: the assessors
            # should not chase a register that is still moving under them)
            live = self.p["dbase"][:, None] + self.p["ddom"]
            self.reg_target = 0.98 * self.reg_target + 0.02 * live
            L, _ = self.forward(**lam)
            gr = self.backward()
            for k in self.p:
                m1[k] = b1 * m1[k] + (1 - b1) * gr[k]
                m2[k] = b2 * m2[k] + (1 - b2) * (gr[k] ** 2)
                self.p[k] = self.p[k] - lr * (m1[k] / (1 - b1 ** t)) / (
                    np.sqrt(m2[k] / (1 - b2 ** t)) + eps)
            if verbose and (t % max(1, steps // 6) == 0 or t == 1):
                tm = self._cache["terms"]
                self.hist.append((t, L))
                print(f"    step {t:5d}  L={L:.5f}  corpus={tm['corpus']:.4f} "
                      f"exam={tm['exam']:.4f} wall={tm['wall']:.4f} "
                      f"stones={self._cache['S']:.1f}")
        return self

    # ---------------------------------------------------------------------
    # READOUT -- every corrected reading passes through the wall
    # ---------------------------------------------------------------------
    def attested_cells(self):
        """Cells (class, domain) in which at least one witness survives."""
        A = np.zeros((self.c.G, self.c.D), dtype=bool)
        A[self.c.g[self.cand], self.c.d[self.cand]] = True
        return A

    def delta_hat(self, subj_idx):
        """
        The censor's thumb for these subjects.

        Two regimes, and the boundary between them is the whole epistemology:
          * ATTESTED cell -- a witness stood here, so the register itself is
            identified and is read off directly.
          * UNATTESTED cell -- no witness ever stood here, so the register is a
            free constant and reading it off would be reading off one's own
            prior. The wall answers instead, by structural resemblance to the
            quarters that were witnessed.
        """
        subj_idx = np.asarray(subj_idx)
        self.forward()
        c, p = self.c, self.p
        A = self.attested_cells()
        att = A[c.g[subj_idx], c.d[subj_idx]]
        reg = p["dbase"][c.g[subj_idx]] + p["ddom"][c.g[subj_idx], c.d[subj_idx]]
        wall = self._wall(self._cache, subj_idx)
        return np.where(att, reg, wall)

    def merit_for(self, subj_idx):
        subj_idx = np.asarray(subj_idx)
        return self.ly[subj_idx] - self.delta_hat(subj_idx)

    def class_gap(self):
        c, o = self.c, self.obs
        i1, i0 = np.where(o & (c.g == 1))[0], np.where(o & (c.g == 0))[0]
        return float(self.merit_for(i1).mean() - self.merit_for(i0).mean())

    def merit_rmse(self, idx):
        idx = np.asarray(idx)
        return float(np.sqrt(np.mean((self.merit_for(idx) - self.c.m_true[idx]) ** 2)))

    def wall_stones(self):
        self.forward()
        a = self._cache["a_b"]
        keep = a > 0.5 * a.max()
        return self.cand[keep], a[keep], keep


# ==============================================================================
# 4. LES TROIS VERTUS  --  deployment policy in an uncorrected world
# ==============================================================================

class TroisVertus:
    """
    The Cite proves capacity.  The Trois Vertus tells a woman what to DO on
    Tuesday, in a world where the record has not been fixed and will not be
    fixed in her lifetime.  Christine writes estate by estate -- queen,
    baroness, townswoman, widow, labourer, prostitute -- and the advice is not
    the same advice.  That asymmetry is the whole point, so the policy here is
    conditioned on standing.

    State  : [one-hot estate (5), corrected merit, hostility of the audience]
    Action : 0 plead directly
             1 constitute the record (gather the file, put both sides on paper,
               submit it to a judge -- what she did on 1 February 1402)
             2 work through a patron
             3 withdraw and preserve
    Reward : stochastic; pleading directly pays only with high standing,
             constituting a record pays broadly but slowly and costs more when
             hostility is high, patronage pays at middle standing, withdrawal
             pays a small floor that is never negative.

    Trained by REINFORCE with a moving baseline.  Hand-derived gradient,
    finite-difference verified on the surrogate.
    """

    N_EST, N_ACT = 5, 4
    EST_POWER = np.array([1.00, 0.72, 0.45, 0.28, 0.10])

    def __init__(self, seed=3):
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.W = rng.normal(0, 0.05, size=(self.N_EST + 2, self.N_ACT))
        self.b = np.zeros(self.N_ACT)
        self.baseline = 0.0

    def state(self, est, merit, host):
        s = np.zeros(self.N_EST + 2)
        s[est] = 1.0
        s[self.N_EST] = merit
        s[self.N_EST + 1] = host
        return s

    def probs(self, S):
        return softmax(S @ self.W + self.b, axis=-1)

    def reward(self, est, merit, host, act):
        pw = self.EST_POWER[est]
        r = self.rng.normal
        if act == 0:      # plead directly -- pays only where standing is high
            return 2.9 * pw * (1 - 0.7 * host) + 0.25 * merit - 1.05 + r(0, 0.18)
        if act == 1:      # constitute the record -- slow, cheap, survives you,
                          # and needs a real case rather than a real title
            return 0.62 + 1.25 * merit * (0.75 + 0.25 * pw) - 0.22 * host + r(0, 0.18)
        if act == 2:      # work through a patron -- wants proximity, not power
            return (1.45 * np.exp(-((pw - 0.70) ** 2) / (2 * 0.16 ** 2))
                    * (1 - 0.45 * host) + 0.10 * merit + r(0, 0.18))
        # withdraw and preserve -- a floor that rises as the room turns hostile
        # and as one's standing falls; never negative, never large
        return 0.24 + 0.85 * (1 - pw) * host + r(0, 0.10)

    def sample_batch(self, n, merit_pool):
        est = self.rng.integers(0, self.N_EST, size=n)
        host = self.rng.uniform(0.0, 1.0, size=n)
        mer = self.rng.choice(merit_pool, size=n)
        mer = np.clip(mer, -2.0, 2.0)
        S = np.stack([self.state(e, m, h) for e, m, h in zip(est, mer, host)])
        return est, mer, host, S

    def surrogate_and_grad(self, S, acts, adv):
        """-mean(adv * log pi(a|s)) and its exact gradient."""
        P = self.probs(S)
        n = len(acts)
        lp = np.log(P[np.arange(n), acts] + EPS)
        Lsur = -float(np.mean(adv * lp))
        onehot = np.zeros_like(P)
        onehot[np.arange(n), acts] = 1.0
        dlogits = -(adv[:, None] * (onehot - P)) / n
        return Lsur, S.T @ dlogits, dlogits.sum(axis=0)

    def fit(self, merit_pool, steps=700, batch=128, lr=0.20, verbose=True):
        curve = []
        for t in range(1, steps + 1):
            est, mer, host, S = self.sample_batch(batch, merit_pool)
            P = self.probs(S)
            acts = np.array([self.rng.choice(self.N_ACT, p=P[i]) for i in range(batch)])
            R = np.array([self.reward(e, m, h, a)
                          for e, m, h, a in zip(est, mer, host, acts)])
            self.baseline = 0.95 * self.baseline + 0.05 * R.mean()
            adv = R - self.baseline
            adv = adv / (adv.std() + 1e-6)     # normalised advantage
            _, gW, gb = self.surrogate_and_grad(S, acts, adv)
            self.W -= lr * gW
            self.b -= lr * gb
            if t % max(1, steps // 5) == 0:
                curve.append((t, float(R.mean())))
                if verbose:
                    print(f"    step {t:5d}   mean reward={R.mean():+.4f}")
        return curve

    def analytic_optimum(self, merit_pool, n=4000):
        """
        The best achievable action in each situation, computed from the
        environment's expected returns.  Reported beside the learned policy so
        the reader can see how much of the available advice was actually found.
        """
        est, mer, host, _ = self.sample_batch(n, merit_pool)
        share = np.zeros((self.N_EST, self.N_ACT))
        tot = 0.0
        for e, m, h in zip(est, mer, host):
            pw = self.EST_POWER[e]
            ev = np.array([
                2.9 * pw * (1 - 0.7 * h) + 0.25 * m - 1.05,
                0.62 + 1.25 * m * (0.75 + 0.25 * pw) - 0.22 * h,
                1.45 * np.exp(-((pw - 0.70) ** 2) / (2 * 0.16 ** 2))
                * (1 - 0.45 * h) + 0.10 * m,
                0.24 + 0.85 * (1 - pw) * h])
            a = int(np.argmax(ev))
            share[e, a] += 1
            tot += ev[a]
        share = share / np.maximum(1, share.sum(axis=1, keepdims=True))
        return share, tot / n

    def evaluate(self, merit_pool, n=4000, greedy=True):
        est, mer, host, S = self.sample_batch(n, merit_pool)
        P = self.probs(S)
        acts = P.argmax(axis=1) if greedy else np.array(
            [self.rng.choice(self.N_ACT, p=P[i]) for i in range(n)])
        R = np.array([self.reward(e, m, h, a) for e, m, h, a in zip(est, mer, host, acts)])
        rnd = np.array([self.reward(e, m, h, self.rng.integers(0, self.N_ACT))
                        for e, m, h in zip(est, mer, host)])
        by_est = {}
        for e in range(self.N_EST):
            sel = est == e
            cnt = np.bincount(acts[sel], minlength=self.N_ACT)
            by_est[e] = cnt / max(1, cnt.sum())
        return float(R.mean()), float(rnd.mean()), by_est


# ==============================================================================
# 5. TESTS
# ==============================================================================

ACT_NAMES = ["plead directly", "constitute the record", "work via patron", "withdraw"]
EST_NAMES = ["queen", "baroness", "townswoman", "widow", "labourer"]

CFG = dict(n_subjects=900, n_domains=8, seed=21, held_out_domains=(6, 7),
           anchor_frac=0.14)
MDL = dict(H=6, wall_budget=70, wall_floor=46)
LAM = dict(lam_anchor=8.0, lam_wall=3.0)


def _banner(n, title):
    print(f"\n[TEST {n}] {title}")


def _metrics(m, champ, tg):
    reg = m.p["dbase"][:, None] + m.p["ddom"]
    keep = [d for d in range(champ.D) if d not in champ.held_out_domains]
    regerr = float(np.sqrt(np.mean((reg[:, keep] - champ.delta_true[:, keep]) ** 2)))
    return dict(held=m.merit_rmse(champ.heldout_idx),
                gaperr=abs(m.class_gap() - tg),
                regerr=regerr)


# ------------------------------------------------------------------ TEST 1
def test_gradcheck(quick=False):
    _banner(1, "Finite-difference gradient check on the full Plaid loss")
    champ = ChampDesEscriptures(n_subjects=200, n_domains=5, seed=11,
                                held_out_domains=(4,), anchor_frac=0.2)
    mdl = Plaid(champ, H=4, wall_budget=20, wall_floor=12, seed=2)
    rng = np.random.default_rng(7)
    v0 = mdl.get_flat()
    v0 = v0 + rng.normal(0, 0.12, size=v0.size)   # perturb so every branch is live
    mdl.set_flat(v0)

    L0, _ = mdl.forward()
    ga = mdl.backward()
    flat_g = np.concatenate([np.asarray(ga[k]).ravel() for k in mdl.keys()])

    picks = rng.choice(v0.size, size=(70 if quick else 140), replace=False)
    h = 1e-6
    num, ana = [], []
    for j in picks:
        vp = v0.copy(); vp[j] += h
        mdl.set_flat(vp); Lp, _ = mdl.forward()
        vm = v0.copy(); vm[j] -= h
        mdl.set_flat(vm); Lm, _ = mdl.forward()
        num.append((Lp - Lm) / (2 * h)); ana.append(flat_g[j])
    mdl.set_flat(v0)

    num, ana = np.array(num), np.array(ana)
    absd = np.abs(num - ana)
    rel = absd / (np.abs(num) + np.abs(ana) + 1e-12)
    # Standard combined criterion.  Coordinates whose true gradient is ~0 have a
    # meaningless relative error and pass on the absolute floor instead.
    passed = (rel < 1e-4) | (absd < 1e-8)
    big = np.abs(ana) > 1e-4
    print(f"    parameters       : {v0.size}    coordinates checked : {len(picks)}")
    print(f"    max abs diff     : {absd.max():.3e}")
    print(f"    max rel err where |grad| > 1e-4 : "
          f"{(rel[big].max() if big.any() else 0.0):.3e}   [{int(big.sum())} coords]")
    print(f"    coordinates passing : {int(passed.sum())}/{len(picks)}")
    ok = bool(passed.all())
    print(f"    {'PASS' if ok else 'FAIL'}   (rel < 1e-4 OR abs < 1e-8)")
    assert ok, "gradient check failed"
    return ok


# ------------------------------------------------------------------ TEST 2
def test_policy_gradcheck():
    _banner(2, "Finite-difference check on the Trois Vertus policy surrogate")
    pol = TroisVertus(seed=5)
    rng = np.random.default_rng(9)
    est, mer, host, S = pol.sample_batch(48, np.linspace(-1.5, 1.5, 40))
    acts = rng.integers(0, pol.N_ACT, size=48)
    adv = rng.normal(0, 1.0, size=48)
    L0, gW, gb = pol.surrogate_and_grad(S, acts, adv)
    h, worst = 1e-6, 0.0
    for _ in range(40):
        i = int(rng.integers(0, pol.W.shape[0])); j = int(rng.integers(0, pol.W.shape[1]))
        old = pol.W[i, j]
        pol.W[i, j] = old + h; Lp, _, _ = pol.surrogate_and_grad(S, acts, adv)
        pol.W[i, j] = old - h; Lm, _, _ = pol.surrogate_and_grad(S, acts, adv)
        pol.W[i, j] = old
        nd = (Lp - Lm) / (2 * h)
        worst = max(worst, abs(nd - gW[i, j]) / (abs(nd) + abs(gW[i, j]) + 1e-12))
    print(f"    max rel error    : {worst:.3e}")
    ok = worst < 1e-5
    print(f"    {'PASS' if ok else 'FAIL'}   (threshold 1e-5)")
    assert ok
    return ok


# ------------------------------------------------------------------ TEST 3
def test_unidentifiability():
    _banner(3, "The theorem: an unopposed record is EXACTLY unidentifiable")
    print("         merit and the censor's thumb enter the corpus likelihood")
    print("         only as their sum, so no archive of any size can split them")
    champ = ChampDesEscriptures(n_subjects=400, n_domains=5, seed=3,
                                held_out_domains=(4,), anchor_frac=0.2)
    mdl = Plaid(champ, H=4, wall_budget=20, wall_floor=12, seed=4)
    mdl.forward(); lc0 = mdl._cache["terms"]["corpus"]

    mdl.p["m"][champ.g == 1] += 0.9          # class-wide gauge move
    mdl.p["dbase"][1] -= 0.9
    mdl.forward(); lc1 = mdl._cache["terms"]["corpus"]

    mdl.p["m"][(champ.g == 0) & (champ.d == 2)] -= 0.6   # cell-local gauge move
    mdl.p["ddom"][0, 2] += 0.6
    mdl.forward(); lc2 = mdl._cache["terms"]["corpus"]

    print(f"    corpus loss, original reading    : {lc0:.12f}")
    print(f"    after a class-wide gauge shift   : {lc1:.12f}")
    print(f"    after a cell-local gauge shift   : {lc2:.12f}")
    print(f"    max deviation                    : "
          f"{max(abs(lc1-lc0), abs(lc2-lc0)):.3e}")
    ok = abs(lc1 - lc0) < 1e-10 and abs(lc2 - lc0) < 1e-10
    print(f"    {'PASS' if ok else 'FAIL'}")
    print("    Reading: every account of this archive sits one free constant per")
    print("    cell away from every other account, and the archive itself is")
    print("    silent about which. Only a witness from OUTSIDE it can choose.")
    assert ok
    return ok


# ------------------------------------------------------------------ TEST 4
def test_reconstruction(quick=False):
    _banner(4, "Reconstruction: does the manufactured gap disappear?")
    steps = 400 if quick else 1200
    champ = ChampDesEscriptures(**CFG)
    print("    corpus:", champ.summary())
    tg = champ.summary()["true_gap"]

    naive = MatheolusReader(champ)
    print(f"\n    Matheolus (record at face value) : gap = {naive.class_gap():+.4f}")
    print(f"    ground truth                     : gap = {tg:+.4f}")

    print("\n    -- Le Plaid, examined witnesses ADMITTED --")
    mdl = Plaid(champ, seed=6, use_anchor=True, **MDL)
    mdl.fit(steps=steps, lr=0.06, verbose=True, **LAM)
    gap = mdl.class_gap()
    print(f"       recovered gap = {gap:+.4f}")

    print("\n    -- The same archive, read by five readers who differ ONLY in")
    print("       what they believed before opening it, WITNESSES DISCARDED --")
    priors = [-1.6, -0.8, 0.0, +0.8, +1.6]
    gaps_na, losses_na = [], []
    for pr in priors:
        m = Plaid(champ, seed=6, use_anchor=False, dbase_init=[0.0, pr], **MDL)
        m.fit(steps=steps, lr=0.06, verbose=False, **LAM)
        gaps_na.append(m.class_gap())
        losses_na.append(m._cache["terms"]["corpus"])
    print("          prior conviction   recovered gap    fit to the archive")
    for pr, gg, ll in zip(priors, gaps_na, losses_na):
        print(f"             {pr:+5.1f}            {gg:+8.4f}          {ll:.6f}")
    spread_na = max(gaps_na) - min(gaps_na)
    print(f"       verdicts differ by {spread_na:.3f} while the fit to the")
    print(f"       archive differs by {max(losses_na)-min(losses_na):.2e}.")
    print("       Five convictions, five verdicts, one archive, one goodness of")
    print("       fit. The reader's prejudice is the entire answer, and nothing")
    print("       inside the corpus can see that this is happening.")

    print("\n    -- The same five readers, WITNESSES ADMITTED --")
    gaps_a = []
    for pr in priors:
        m = Plaid(champ, seed=6, use_anchor=True, dbase_init=[0.0, pr], **MDL)
        m.fit(steps=steps, lr=0.06, verbose=False, **LAM)
        gaps_a.append(m.class_gap())
    spread_a = max(gaps_a) - min(gaps_a)
    print(f"          recovered gaps = {np.round(gaps_a, 3).tolist()}")
    print(f"          spread = {spread_a:.4f}   (was {spread_na:.4f})")

    e_naive, e_plaid = abs(naive.class_gap() - tg), abs(gap - tg)
    print(f"\n       |error| Matheolus  : {e_naive:.4f}")
    print(f"       |error| Le Plaid   : {e_plaid:.4f}    "
          f"({100*(1-e_plaid/e_naive):.0f}% of the manufactured gap removed)")
    print(f"       prior-sensitivity  : {spread_na:.3f} without witnesses "
          f"-> {spread_a:.3f} with them")
    ok = (e_plaid < 0.25 * e_naive) and (spread_na > 0.8) and (spread_a < 0.6 * spread_na)
    print(f"    {'PASS' if ok else 'FAIL'}")
    assert ok
    return mdl, champ, naive, tg


# ------------------------------------------------------------------ TEST 5
def test_wall_transfer(mdl, champ, naive):
    _banner(5, "The wall is load-bearing: reading two quarters of the archive")
    print("         in which NO first-person witness survives at all")
    ho = champ.heldout_idx
    r_naive, r_plaid = naive.merit_rmse(ho), mdl.merit_rmse(ho)
    d_true = champ.delta_true[champ.g[ho], champ.d[ho]]
    d_hat = mdl.delta_hat(ho)
    print(f"       subjects in unwitnessed quarters : {len(ho)}")
    print(f"       RMSE  Matheolus (raw record)     : {r_naive:.4f}")
    print(f"       RMSE  Le Plaid via the wall      : {r_plaid:.4f}"
          f"   ({100*(1-r_plaid/r_naive):.0f}% better)")
    print(f"       thumb RMSE, wall vs truth        : "
          f"{np.sqrt(np.mean((d_hat-d_true)**2)):.4f}"
          f"   (vs {np.sqrt(np.mean(d_true**2)):.4f} for assuming no thumb)")
    ok = r_plaid < r_naive
    print(f"    {'PASS' if ok else 'FAIL'}")
    assert ok
    return ok


# ------------------------------------------------------------------ TEST 6
def test_assessor_ablation(champ, tg, quick=False):
    _banner(6, "Assessor ablation: the gate is conjunctive on purpose")
    steps = 400 if quick else 1200
    rows = []
    for name, keep in [("Reason + Rectitude + Justice", ("R", "D", "J")),
                       ("Reason removed",              ("D", "J")),
                       ("Rectitude removed",           ("R", "J")),
                       ("Justice removed",             ("R", "D"))]:
        m = Plaid(champ, seed=6, use_anchor=True, assessors=keep, **MDL)
        m.fit(steps=steps, lr=0.06, verbose=False, **LAM)
        mt = _metrics(m, champ, tg)
        ab = m._cache["a_b"]
        mt["corruptW"] = float(ab[champ.witness_corrupt].sum() / ab.sum())
        mt["depth"] = float((ab * champ.depth[champ.witness_idx]).sum() / ab.sum())
        rows.append((name, mt))
    base = rows[0][1]
    print(f"       {'configuration':30s} {'heldRMSE':>9s} {'gap err':>9s} "
          f"{'reg err':>9s} {'corrupt%':>9s} {'depth':>7s}")
    for name, mt in rows:
        print(f"       {name:30s} {mt['held']:9.4f} {mt['gaperr']:9.4f} "
              f"{mt['regerr']:9.4f} {100*mt['corruptW']:8.1f}% {mt['depth']:7.2f}")
    print(f"       (pool: {100*champ.witness_corrupt.mean():.1f}% corrupt, "
          f"mean depth {champ.depth[champ.witness_idx].mean():.2f})")
    hurt = []
    for name, mt in rows[1:]:
        worse = [k for k in ("held", "gaperr", "regerr") if mt[k] > 1.03 * base[k]]
        hurt.append((name, worse))
        print(f"       removing {name.split()[0]:10s} degrades: "
              f"{', '.join(worse) if worse else 'nothing'}")
    ok = sum(1 for _, w in hurt if w) >= 2
    print(f"    {'PASS' if ok else 'FAIL'}   (at least 2 of 3 assessors must bite)")
    assert ok
    return rows


# ------------------------------------------------------------------ TEST 7
def test_wall_composition(mdl, champ):
    _banner(7, "What actually got into the wall")
    stones, w, keep = mdl.wall_stones()
    dep_w = float((mdl._cache["a_b"] * champ.depth[mdl.cand]).sum()
                  / mdl._cache["a_b"].sum())
    dep_p = float(champ.depth[mdl.cand].mean())
    ab = mdl._cache["a_b"]
    cw = float(ab[champ.witness_corrupt].sum() / ab.sum())
    cells_pool = len(set((champ.g[mdl.cand] * champ.D + champ.d[mdl.cand]).tolist()))
    cells_wall = len(set((champ.g[stones] * champ.D + champ.d[stones]).tolist()))
    print(f"       candidate witnesses               : {len(mdl.cand)}")
    print(f"       admitted weight (ceiling {mdl.K}, floor {mdl.Kmin}) : "
          f"{mdl._cache['S']:.1f}")
    print(f"       fabricated share  wall / pool     : "
          f"{100*cw:.1f}% / {100*champ.witness_corrupt.mean():.1f}%"
          "        <- Reason's mirror")
    print(f"       mean chain depth  wall / pool     : {dep_w:.2f} / {dep_p:.2f}"
          "            <- Rectitude's ruler")
    print(f"       cells represented wall / pool     : {cells_wall} / {cells_pool}"
          "                <- Justice's vessel")
    ok = cw < champ.witness_corrupt.mean() and cells_wall >= cells_pool - 1
    print(f"    {'PASS' if ok else 'FAIL'}")
    assert ok
    return ok


# ------------------------------------------------------------------ TEST 8
def test_policy(mdl, champ, quick=False):
    _banner(8, "Trois Vertus: acting well inside a system still uncorrected")
    print("         The Cite proves capacity. This is the other book she finished")
    print("         in 1405: what to do on Tuesday, estate by estate, in a world")
    print("         whose record will not be repaired in your lifetime.")
    pool = mdl.merit_for(champ.train_idx)
    pol = TroisVertus(seed=5)
    pol.fit(pool, steps=(700 if quick else 2500), lr=0.12, verbose=True)
    r_learn, r_rand, by_est = pol.evaluate(pool)
    opt_share, r_opt = pol.analytic_optimum(pool)
    print(f"\n       mean return, learned policy   : {r_learn:+.4f}")
    print(f"       mean return, acting at random : {r_rand:+.4f}")
    print(f"       mean return, oracle optimum   : {r_opt:+.4f}"
          f"   (learned recovers {100*(r_learn-r_rand)/(r_opt-r_rand):.0f}% of the gap)")
    print("\n       action mix by estate  [plead | record | patron | withdraw]")
    print("       learned policy above, oracle optimum below:")
    for e in range(pol.N_EST):
        print(f"         {EST_NAMES[e]:12s} learned {np.round(by_est[e],2)}")
        print(f"         {'':12s} oracle  {np.round(opt_share[e],2)}")
    ok = r_learn > r_rand and (r_learn - r_rand) > 0.75 * (r_opt - r_rand)
    print(f"    {'PASS' if ok else 'FAIL'}")
    assert ok
    return pol


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    np.set_printoptions(precision=4, suppress=True)
    print("=" * 78)
    print(" LE PLAID SANS PARTIE  --  Christine de Pizan (1364 - c.1430)")
    print(" Ex-parte-corpus reconstruction. Pure NumPy, hand-derived gradients.")
    print("=" * 78)

    test_gradcheck(args.quick)
    test_policy_gradcheck()
    test_unidentifiability()
    mdl, champ, naive, tg = test_reconstruction(args.quick)
    test_wall_transfer(mdl, champ, naive)
    test_assessor_ablation(champ, tg, args.quick)
    test_wall_composition(mdl, champ)
    test_policy(mdl, champ, args.quick)

    print("\n" + "=" * 78)
    print(" ALL TESTS PASSED")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
