#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
  # BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0399 - KING SEJONG THE GREAT (1397-1450)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 20, Minds 381-400 Available on Amazon https://www.amazon.com/dp/B0HRS9ZHVG
# Resume and Interactive Demos at  https://artificiology.com/
# Videos of the playable minds at https://www.youtube.com/@LostMindsAI
# END ATTRIBUTION  
 JEONGEUM-KONGBEOP NET
 The cognitive architecture of King Sejong the Great (1397-1450), Joseon Korea
================================================================================

WHY THIS ARCHITECTURE AND NOT ANOTHER
-------------------------------------
Sejong's court already had what a modern engineer would call abundant supply:
metal movable type (kyongja type 1420; over 200,000 pieces of kabin type cast in
1434), a state printing office, and a bureaucracy that could distribute a book
to every county in the kingdom.  It also had near-total illiteracy.  Sejong is
the one ruler in this corpus who had solved the *reproduction* problem and
correctly diagnosed that the binding constraint was somewhere else entirely:
on the receiver's side, in the cost of acquiring a DECODER.

Two mechanisms follow from that diagnosis, and this file implements both as
real, trainable, differentiable machinery -- not as modules named after
institutions.

  MECHANISM 1 -- THE DERIVED TABLE  (from the Hunminjeongeum Haerye, 1446)
  ------------------------------------------------------------------------
  The Haerye's chapter "Chajahae" (the explanation of the making of the letters)
  states that the five base consonants are pictures of the articulator that
  produces them -- "the velar letter depicts the shape of the tongue-root
  blocking the throat" -- and that the remaining consonants are obtained by
  ADDING A STROKE because "the sound comes out slightly stronger".  The vowels
  are composed from three primitives (heaven, earth, human) with a yin/yang
  polarity.

  Implemented here as a codebook that is a FUNCTION of a handful of parameters
  rather than a table of free ones:

        E_consonant = PlaceChart @ A  +  StrokeChart @ S
        E_vowel     = PrimitiveCounts @ B  +  polarity * p

  Adding a stroke is literally vector addition.  There is no per-symbol
  parameter anywhere in this model.  The consequence is testable and this file
  tests it: symbols HELD OUT OF TRAINING ENTIRELY can still be emitted at test
  time, because their vectors are derived from the chart rather than looked up.

  MECHANISM 2 -- THE DISAGGREGATED RESIDUAL  (from the Kongbeop, 1430-1444)
  ------------------------------------------------------------------------
  In 1430 Sejong polled 172,806 people on a proposed flat land tax.  He won:
  roughly 98,657 for, 74,148 against.  He did not act.  The provincial breakdown
  showed the fertile south in favour and the poor north against -- so the split
  was not noise around a true preference, it was evidence that ONE PARAMETER
  COULD NOT FIT A HETEROGENEOUS COUNTRY.  He spent fourteen further years
  replacing the single rate with a grid: six grades of land fertility crossed
  with nine grades of harvest year, enacted 1444.

  Implemented here as a training-time event, not a hyperparameter.  The model
  begins with a single shared head.  After each epoch it computes a
  misspecification statistic F = (between-stratum residual variance) /
  (within-stratum residual variance).  While F stays low, the mean is enough.
  When F crosses threshold, the model does NOT push harder on the mean -- it
  OPENS a rank-one 6x9 correction grid and starts conditioning on strata.  The
  opening is logged.  You can watch the model decline to promulgate.

WHAT THIS IS NOT
----------------
Not a Transformer.  There is no attention, no key-value store, no softmax over
stored positions.  Nothing in this model retrieves a memorised item; the two
places where a lesser design would store a table (the symbol embeddings and the
per-region intercepts) are both replaced by generators.  That is the whole
point of the mind being modelled.

CONVENTIONS
-----------
Pure NumPy.  All gradients hand-derived.  A central-difference gradient check
over every learnable tensor is mandatory and runs in main().  Self-tests run in
main().  Executed and verified before shipping.

Run:  python3 chapter_0399_king_sejong_the_great_1397.py
================================================================================
"""

import numpy as np

# ==============================================================================
# SECTION 1 -- THE CHART (Chajahae).  Fixed, human-readable, not learned.
# ==============================================================================
# This is the load-bearing prior of the whole architecture.  Everything the
# model knows about the shape of its own vocabulary lives in these tables, and
# they are published constants rather than fitted parameters -- exactly as the
# Haerye published the derivation alongside the alphabet instead of shipping the
# alphabet alone.

# Five articulatory places (the five base icons of the Haerye).
PLACES = ["velar", "lingual", "labial", "dental", "glottal"]
P = len(PLACES)

# Three stroke-addition / doubling operations.
#   stop     = the first added stroke (occlusion)
#   aspirate = the second added stroke ("the sound comes out slightly stronger")
#   tense    = pyongso, the side-by-side doubling of a letter
STROKES = ["stop", "aspirate", "tense"]
F = len(STROKES)

# (glyph, place, [stop, aspirate, tense], is_chart_irregular)
# NOTE ON HONESTY OF THE CHART: the Haerye's own system has a documented
# irregularity.  The velar base letter is itself a stop, and the semi-lingual
# letter is explicitly called an outlier by the text.  We inherit the
# irregularity rather than silently tidying it, and flag it, because a chart
# that has been quietly regularised is no longer the chart that was published.
CONSONANTS = [
    ("g",   "velar",   [0, 0, 0], False),   # base icon: tongue-root at the molars
    ("k",   "velar",   [0, 1, 0], False),   # +aspirate stroke
    ("gg",  "velar",   [0, 0, 1], False),   # doubled
    ("n",   "lingual", [0, 0, 0], False),   # base icon: tongue-tip at the palate
    ("d",   "lingual", [1, 0, 0], False),
    ("t",   "lingual", [1, 1, 0], False),
    ("dd",  "lingual", [1, 0, 1], False),
    ("m",   "labial",  [0, 0, 0], False),   # base icon: the mouth
    ("b",   "labial",  [1, 0, 0], False),
    ("p",   "labial",  [1, 1, 0], False),
    ("bb",  "labial",  [1, 0, 1], False),
    ("s",   "dental",  [0, 0, 0], False),   # base icon: the incisor
    ("j",   "dental",  [1, 0, 0], False),
    ("ch",  "dental",  [1, 1, 0], False),
    ("ss",  "dental",  [0, 0, 1], False),
    ("jj",  "dental",  [1, 0, 1], False),
    ("ng",  "glottal", [0, 0, 0], False),   # base icon: the throat
    ("h",   "glottal", [1, 1, 0], False),
    ("r",   "lingual", [0, 1, 0], True),    # the semi-lingual outlier
]
NC = len(CONSONANTS)

# Vowels: composed from three primitives with a yin/yang polarity.
#   heaven (a round dot), earth (a flat line), human (an upright line)
# (glyph, [n_heaven, n_earth, n_human], polarity)
PRIMS = ["heaven", "earth", "human"]
R = len(PRIMS)
VOWELS = [
    ("ARAEA", [1, 0, 0], +1),   # heaven alone -- yang
    ("EU",    [0, 1, 0], -1),   # earth alone  -- yin
    ("I",     [0, 0, 1],  0),   # human alone  -- neutral
    ("O",     [1, 1, 0], +1),   # heaven above earth
    ("U",     [1, 1, 0], -1),   # heaven below earth
    ("A",     [1, 0, 1], +1),   # heaven right of human
    ("EO",    [1, 0, 1], -1),   # heaven left of human
    ("YO",    [2, 1, 0], +1),   # doubled heaven
    ("YU",    [2, 1, 0], -1),
    ("YA",    [2, 0, 1], +1),
    ("YEO",   [2, 0, 1], -1),
]
NV = len(VOWELS)

CONS_IDX = {g: i for i, (g, _, _, _) in enumerate(CONSONANTS)}
VOW_IDX = {g: i for i, (g, _, _) in enumerate(VOWELS)}


def build_charts():
    """Materialise the fixed chart matrices.  These are constants, not params."""
    Cp = np.zeros((NC, P))                    # consonant -> place one-hot
    Cf = np.zeros((NC, F))                    # consonant -> stroke multi-hot
    for i, (_, pl, fe, _) in enumerate(CONSONANTS):
        Cp[i, PLACES.index(pl)] = 1.0
        Cf[i, :] = fe
    Cv = np.zeros((NV, R))                    # vowel -> primitive counts
    pol = np.zeros(NV)                        # vowel -> yin/yang
    for i, (_, cnt, pl) in enumerate(VOWELS):
        Cv[i, :] = cnt
        pol[i] = pl
    return Cp, Cf, Cv, pol


CP, CF, CV, POL = build_charts()


# ==============================================================================
# SECTION 2 -- SMALL NUMERIC HELPERS
# ==============================================================================

def sigmoid(x):
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def softmax_rows(x):
    m = x.max(axis=-1, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=-1, keepdims=True)


# ==============================================================================
# SECTION 3 -- THE MODEL
# ==============================================================================

class JeongeumKongbeopNet:
    """
    Forward pass, in order:

      1. DERIVE the symbol vectors from the chart          (Chajahae)
      2. ASSEMBLE three slot-typed symbols into a block    (the syllable square)
      3. READ the block sequence with a gated recurrence    (deliberation)
      4. PREDICT with a shared head, plus -- only once the residuals prove it
         necessary -- a rank-one land-grade x harvest-grade grid   (Kongbeop)

    Nothing in steps 1-4 looks anything up.  d = symbol width, H = state width.
    """

    def __init__(self, d=12, H=24, n_strata=8, n_land=6, n_year=9, seed=0,
                 free_table=False):
        rng = np.random.default_rng(seed)
        s = lambda *sh: rng.normal(0, 0.35, size=sh)

        self.d, self.H = d, H
        self.n_strata, self.n_land, self.n_year = n_strata, n_land, n_year

        # --- 1. the codebook ---------------------------------------------
        # free_table=True is the CONTROL CONDITION: a conventional embedding
        # matrix with one free row per symbol.  It is the design Sejong did not
        # choose, and it is what every other alphabet in history is.  Keeping it
        # in the same file, trained by the same loop, is the only way to show
        # that the unseen-letter result comes from the derivation and not from
        # the rest of the network.
        self.free_table = free_table
        if free_table:
            self.Ec_free = s(NC, d)
            self.Ev_free = s(NV, d)
            codebook_params = ["Ec_free", "Ev_free"]
        else:
            self.A = s(P, d)      # the five base icons
            self.S = s(F, d)      # the stroke-addition operators
            self.B = s(R, d)      # heaven / earth / human
            self.p = s(d)         # the yin-yang polarity direction
            codebook_params = ["A", "S", "B", "p"]

        # --- 2. slot-typed block assembly (onset / nucleus / coda) ---
        self.Wo, self.Wn, self.Wc = s(d, H), s(d, H), s(d, H)
        self.bb = np.zeros(H)

        # --- 3. gated recurrence over blocks ---
        self.Wz, self.Uz, self.bz = s(H, H) * 0.5, s(H, H) * 0.5, np.zeros(H)
        self.Wh, self.Uh, self.bh = s(H, H) * 0.5, s(H, H) * 0.5, np.zeros(H)

        # --- 4. shared head ---
        self.w = s(H) * 0.5
        self.b0 = np.zeros(1)

        # --- 4b. the Kongbeop grid: SIX land grades x NINE harvest grades ---
        # Gland/Gyear form a rank-one correction.  Aland holds each stratum's
        # soft membership over the six fertility grades -- the model must infer
        # which grade a province belongs to, exactly as the 1430 survey had to.
        self.Gland = s(n_land) * 0.3
        self.Gyear = s(n_year) * 0.3
        self.Aland = s(n_strata, n_land) * 0.1

        # The gate starts CLOSED.  It is opened by evidence, in train(), and the
        # opening is an observable event rather than a design assumption.
        self.grid_open = False
        self.grid_opened_epoch = None

        self.PARAMS = codebook_params + [
            "Wo", "Wn", "Wc", "bb",
            "Wz", "Uz", "bz", "Wh", "Uh", "bh",
            "w", "b0", "Gland", "Gyear", "Aland"]

    # -- parameter registry, used by the optimiser and the gradient check ----
    def get(self, name):
        return getattr(self, name)

    # ---------------------------------------------------------------- codebook
    def derive_codebook(self):
        """E_c = Cp A + Cf S ;  E_v = Cv B + pol p.   No per-symbol parameters."""
        if self.free_table:
            return self.Ec_free, self.Ev_free
        Ec = CP @ self.A + CF @ self.S
        Ev = CV @ self.B + POL[:, None] * self.p
        return Ec, Ev

    def codebook_compression(self):
        """Free parameters used by the codebook vs. a plain embedding table."""
        derived = (NC + NV) * self.d if self.free_table else (P + F + R + 1) * self.d
        table = (NC + NV) * self.d
        return derived, table, table / derived

    # ----------------------------------------------------------------- forward
    def forward(self, batch, grid_override=None):
        """
        batch: dict with
          ons, nuc, cod : int arrays (N, T)   coda = -1 means the slot is empty
          strat         : int array  (N,)     which province the report is from
          year          : int array  (N,)     the observed harvest-year grade
        Returns (yhat, cache).
        """
        ons, nuc, cod = batch["ons"], batch["nuc"], batch["cod"]
        strat, year = batch["strat"], batch["year"]
        N, T = ons.shape
        d, H = self.d, self.H

        Ec, Ev = self.derive_codebook()

        # ---- 2. block assembly: three typed slots into one square -----------
        Eo = Ec[ons.reshape(-1)]                      # (N*T, d)
        En = Ev[nuc.reshape(-1)]
        cod_flat = cod.reshape(-1)
        has_cod = cod_flat >= 0
        Ecd = np.zeros((N * T, d))
        Ecd[has_cod] = Ec[cod_flat[has_cod]]          # empty coda contributes 0

        pre_b = Eo @ self.Wo + En @ self.Wn + Ecd @ self.Wc + self.bb
        blk = np.tanh(pre_b).reshape(N, T, H)

        # ---- 3. gated recurrence across the blocks of the report ------------
        h = np.zeros((N, H))
        zs, cands, hs = [], [], [h]
        for t in range(T):
            x = blk[:, t, :]
            z = sigmoid(h @ self.Wz + x @ self.Uz + self.bz)
            c = np.tanh(h @ self.Wh + x @ self.Uh + self.bh)
            h = (1.0 - z) * h + z * c
            zs.append(z); cands.append(c); hs.append(h)

        # ---- 4. shared head ------------------------------------------------
        y_shared = h @ self.w + self.b0[0]

        # ---- 4b. the grid, if and only if the residuals have earned it ------
        use_grid = self.grid_open if grid_override is None else grid_override
        alpha = softmax_rows(self.Aland)              # (n_strata, n_land)
        if use_grid:
            a_s = alpha[strat]                        # (N, n_land)
            land = a_s @ self.Gland                   # (N,)  soft fertility grade
            yr = self.Gyear[year]                     # (N,)  harvest-year grade
            grid = land * yr                          # rank-one 6 x 9 correction
        else:
            land = np.zeros(N); yr = np.zeros(N); grid = np.zeros(N)

        yhat = y_shared + grid

        cache = dict(N=N, T=T, ons=ons, nuc=nuc, cod=cod, strat=strat, year=year,
                     Eo=Eo, En=En, Ecd=Ecd, has_cod=has_cod, cod_flat=cod_flat,
                     pre_b=pre_b, blk=blk, zs=zs, cands=cands, hs=hs,
                     alpha=alpha, land=land, yr=yr, use_grid=use_grid)
        return yhat, cache

    # ---------------------------------------------------------------- backward
    def backward(self, dY, cache):
        """dY: (N,) gradient of the scalar loss w.r.t. yhat.  Returns dict."""
        N, T, H, d = cache["N"], cache["T"], self.H, self.d
        g = {k: np.zeros_like(self.get(k), dtype=float) for k in self.PARAMS}

        # ---- 4b. grid --------------------------------------------------------
        if cache["use_grid"]:
            strat, year = cache["strat"], cache["year"]
            alpha, land, yr = cache["alpha"], cache["land"], cache["yr"]
            a_s = alpha[strat]                                  # (N, n_land)
            # grid = (a_s @ Gland) * Gyear[year]
            g["Gland"] += ((dY * yr)[:, None] * a_s).sum(0)     # (n_land,)
            np.add.at(g["Gyear"], year, dY * land)
            d_as = (dY * yr)[:, None] * self.Gland[None, :]     # (N, n_land)
            d_alpha = np.zeros_like(alpha)
            np.add.at(d_alpha, strat, d_as)
            # softmax jacobian, row-wise
            g["Aland"] += alpha * (d_alpha - (d_alpha * alpha).sum(1, keepdims=True))

        # ---- 4. shared head --------------------------------------------------
        hT = cache["hs"][-1]
        g["w"] += hT.T @ dY
        g["b0"] += dY.sum()
        dh = dY[:, None] * self.w[None, :]                      # (N,H)

        # ---- 3. recurrence, unrolled backwards -------------------------------
        dblk = np.zeros((N, T, H))
        for t in reversed(range(T)):
            h_prev = cache["hs"][t]
            z, c = cache["zs"][t], cache["cands"][t]
            x = cache["blk"][:, t, :]

            dz = dh * (c - h_prev)
            dc = dh * z
            dh_prev = dh * (1.0 - z)

            dz_pre = dz * z * (1.0 - z)
            dc_pre = dc * (1.0 - c * c)

            g["Wz"] += h_prev.T @ dz_pre
            g["Uz"] += x.T @ dz_pre
            g["bz"] += dz_pre.sum(0)
            g["Wh"] += h_prev.T @ dc_pre
            g["Uh"] += x.T @ dc_pre
            g["bh"] += dc_pre.sum(0)

            dh_prev = dh_prev + dz_pre @ self.Wz.T + dc_pre @ self.Wh.T
            dblk[:, t, :] = dz_pre @ self.Uz.T + dc_pre @ self.Uh.T
            dh = dh_prev

        # ---- 2. block assembly ------------------------------------------------
        dpre_b = dblk.reshape(N * T, H) * (1.0 - cache["blk"].reshape(N * T, H) ** 2)
        Eo, En, Ecd = cache["Eo"], cache["En"], cache["Ecd"]
        g["Wo"] += Eo.T @ dpre_b
        g["Wn"] += En.T @ dpre_b
        g["Wc"] += Ecd.T @ dpre_b
        g["bb"] += dpre_b.sum(0)

        dEo = dpre_b @ self.Wo.T
        dEn = dpre_b @ self.Wn.T
        dEcd = dpre_b @ self.Wc.T

        # scatter symbol-vector gradients back onto the codebook rows
        dEc = np.zeros((NC, d))
        dEv = np.zeros((NV, d))
        np.add.at(dEc, cache["ons"].reshape(-1), dEo)
        np.add.at(dEv, cache["nuc"].reshape(-1), dEn)
        hc, cf = cache["has_cod"], cache["cod_flat"]
        if hc.any():
            np.add.at(dEc, cf[hc], dEcd[hc])

        # ---- 1. through the chart into the generators -------------------------
        # In the derived model there is nowhere for a per-symbol gradient to
        # land: it must be projected through the chart onto the five icons and
        # the three strokes.  Every occurrence of every velar letter therefore
        # trains the SAME base vector.  That is the whole mechanism.
        if self.free_table:
            g["Ec_free"] += dEc
            g["Ev_free"] += dEv
        else:
            g["A"] += CP.T @ dEc
            g["S"] += CF.T @ dEc
            g["B"] += CV.T @ dEv
            g["p"] += (POL[:, None] * dEv).sum(0)

        return g

    # ------------------------------------------------------------------- loss
    def loss_and_grad(self, batch, l2=1e-4):
        yhat, cache = self.forward(batch)
        r = yhat - batch["y"]
        N = len(r)
        loss = 0.5 * np.mean(r ** 2)
        dY = r / N
        g = self.backward(dY, cache)
        if l2 > 0:
            for k in self.PARAMS:
                if k not in ("bb", "bz", "bh", "b0"):
                    loss += 0.5 * l2 * np.sum(self.get(k) ** 2)
                    g[k] += l2 * self.get(k)
        return loss, g, yhat


# ==============================================================================
# SECTION 4 -- THE MISSPECIFICATION STATISTIC  (the 1430 reading of the poll)
# ==============================================================================

def misspecification_F(resid, strat, n_strata):
    """
    Sejong's actual test, in one function.

    A majority is a scalar.  A majority plus its provincial breakdown is a
    diagnosis.  Here: split the residuals by stratum and ask whether the
    between-stratum spread is larger than the within-stratum spread.

      F ~ 1   -> disagreement is noise; the shared head is adequate; promulgate.
      F >> 1  -> disagreement is STRUCTURE; the model is misspecified for at
                 least one subpopulation; do not promulgate, add conditioning.

    Returns (F, per-stratum mean residuals).
    """
    means = np.zeros(n_strata)
    within = 0.0
    counts = np.zeros(n_strata)
    for s in range(n_strata):
        m = strat == s
        counts[s] = m.sum()
        if counts[s] > 0:
            means[s] = resid[m].mean()
            within += ((resid[m] - means[s]) ** 2).sum()
    within = within / max(len(resid) - n_strata, 1)
    grand = resid.mean()
    between = (counts * (means - grand) ** 2).sum() / max(n_strata - 1, 1)
    F = between / (within + 1e-12)
    return F, means


# ==============================================================================
# SECTION 5 -- THE CORPUS OF REPORTS  (synthetic, but structured like the case)
# ==============================================================================
# Each example is a short district report: a sequence of syllable blocks, a
# province, and an observed harvest-year grade.  The target is the assessed
# yield.
#
# Two properties are deliberately built in, because they are the two things
# Sejong's mind is about:
#
#   (a) The contribution of a written token is a function of its FEATURES, not
#       of its identity.  So a model that derives symbol vectors from the chart
#       generalises to letters it has never seen, and a model with a free
#       lookup table cannot.
#
#   (b) The target contains a land-grade x harvest-grade term that varies by
#       province.  A single shared head therefore leaves a STRUCTURED residual,
#       which is precisely the signal the 1430 survey returned.

# Provinces, ordered roughly south (fertile) to north (poor), matching the
# split the real survey found: Jeolla and Gyeongsang in favour, Pyeongan,
# Hwanghae and Gangwon against.
PROVINCES = ["Jeolla", "Gyeongsang", "Chungcheong", "Gyeonggi",
             "Gangwon", "Hwanghae", "Pyeongan", "Hamgil"]
TRUE_LAND_GRADE = np.array([0, 0, 1, 2, 4, 3, 5, 5])     # index into six grades

# Three symbols are withheld from every training token, to test whether the
# model can emit a letter it has never been shown.
HELDOUT_CONS = ["p", "jj"]      # aspirated labial, tense dental
HELDOUT_VOW = ["YEO"]           # doubled-heaven yin vowel


def _blk(o, n, c=None):
    return (CONS_IDX[o], VOW_IDX[n], -1 if c is None else CONS_IDX[c])


# A small lexicon of report tokens.  Train tokens avoid the held-out letters.
TRAIN_TOKENS = [
    _blk("m", "U", "r"),  _blk("n", "A"),      _blk("b", "I"),
    _blk("s", "A", "n"),  _blk("g", "O", "d"), _blk("h", "A", "n"),
    _blk("t", "EO"),      _blk("ch", "O"),     _blk("d", "EU", "r"),
    _blk("k", "EU", "n"), _blk("j", "A", "g"), _blk("gg", "O", "ch"),
    _blk("ss", "I"),      _blk("dd", "A", "ng"), _blk("bb", "YA"),
    _blk("r", "YO"),      _blk("ng", "I", "r"), _blk("m", "ARAEA"),
]
# Test-only tokens, each containing at least one letter never seen in training.
HELDOUT_TOKENS = [
    _blk("p", "A", "r"),   _blk("jj", "O"),
    _blk("m", "YEO", "n"), _blk("p", "YEO", "ng"),
]


def token_feature_vector(tok):
    """The hidden semantics: a token's yield contribution depends on FEATURES."""
    o, n, c = tok
    v = np.zeros(6)
    v[0] += CF[o, STROKES.index("aspirate")]
    v[1] += CF[o, STROKES.index("tense")]
    v[2] += CF[o, STROKES.index("stop")]
    v[3] += POL[n]
    v[4] += CV[n].sum()
    if c >= 0:
        v[5] += 1.0 + CF[c, STROKES.index("stop")]
    return v


TOKEN_W = np.array([0.85, -0.60, 0.30, 0.45, 0.22, -0.35])


def make_dataset(n, T=4, seed=1, token_pool=None, rng=None):
    rng = rng or np.random.default_rng(seed)
    pool = token_pool if token_pool is not None else TRAIN_TOKENS
    idx = rng.integers(0, len(pool), size=(n, T))
    ons = np.zeros((n, T), dtype=int)
    nuc = np.zeros((n, T), dtype=int)
    cod = np.zeros((n, T), dtype=int)
    content = np.zeros(n)
    for i in range(n):
        for t in range(T):
            o, v, c = pool[idx[i, t]]
            ons[i, t], nuc[i, t], cod[i, t] = o, v, c
            content[i] += TOKEN_W @ token_feature_vector((o, v, c))
    content /= T

    strat = rng.integers(0, len(PROVINCES), size=n)
    year = rng.integers(0, 9, size=n)

    # The true generative law: a shared content term PLUS a land x year grid.
    land_mult = np.array([1.00, 0.86, 0.72, 0.58, 0.44, 0.30])   # six grades
    # Nine harvest grades, from a failed year to an abundant one.  The range is
    # deliberately NOT centred on zero: a flat rate levied in a bad year was the
    # exact objection the ministers raised against the kongbeop, and a symmetric
    # range would let the provincial structure cancel out in the aggregate.
    year_mult = np.linspace(-0.20, 1.40, 9)                      # nine grades
    grid = land_mult[TRUE_LAND_GRADE[strat]] * year_mult[year]
    y = 0.9 * content + 1.4 * grid + rng.normal(0, 0.05, size=n)

    return dict(ons=ons, nuc=nuc, cod=cod, strat=strat, year=year,
                y=y, content=content)


# ==============================================================================
# SECTION 6 -- GRADIENT CHECK  (mandatory)
# ==============================================================================

def gradient_check(free_table=False, verbose=True, eps=1e-5,
                   rel_tol=1e-5, abs_tol=1e-8):
    """
    Central-difference check over EVERY learnable tensor, with the grid forced
    open so that no path is left unexercised.

    Two tolerances, because one is not enough.  A relative error is the right
    measure for components whose gradient is meaningfully non-zero, and it is
    meaningless for components whose true gradient is ~1e-12, where the
    finite-difference estimate is pure floating-point noise.  So: relative error
    is judged only where max(|numeric|, |analytic|) exceeds a floor, and every
    component -- including the tiny ones -- must additionally pass an absolute
    test.  This is stricter than either check alone.
    """
    rng = np.random.default_rng(7)
    model = JeongeumKongbeopNet(d=6, H=8, seed=3, free_table=free_table)
    model.grid_open = True
    batch = make_dataset(9, T=3, rng=rng)

    _, g_analytic, _ = model.loss_and_grad(batch, l2=1e-3)

    FLOOR = 1e-6                      # below this, relative error is noise
    worst_rel, worst_abs = 0.0, 0.0
    rows = []
    for name in model.PARAMS:
        arr = model.get(name)
        flat = np.atleast_1d(arr).ravel()
        n_probe = min(8, flat.size)
        probes = rng.choice(flat.size, size=n_probe, replace=False)
        r_max, a_max, n_rel = 0.0, 0.0, 0
        for j in probes:
            orig = flat[j]
            flat[j] = orig + eps
            lp, _, _ = model.loss_and_grad(batch, l2=1e-3)
            flat[j] = orig - eps
            lm, _, _ = model.loss_and_grad(batch, l2=1e-3)
            flat[j] = orig
            num = (lp - lm) / (2 * eps)
            ana = float(np.atleast_1d(g_analytic[name]).ravel()[j])
            a_max = max(a_max, abs(num - ana))
            scale = max(abs(num), abs(ana))
            if scale > FLOOR:
                r_max = max(r_max, abs(num - ana) / scale)
                n_rel += 1
        worst_rel = max(worst_rel, r_max)
        worst_abs = max(worst_abs, a_max)
        rows.append((name, getattr(arr, "shape", ()), r_max, a_max, n_rel))

    ok = (worst_rel < rel_tol) and (worst_abs < abs_tol)
    if verbose:
        print("  {:<8} {:<12} {:>11} {:>11}  {}".format(
            "param", "shape", "rel err", "abs err", "n>floor"))
        for name, shp, r, a, k in rows:
            print("  {:<8} {:<12} {:>11.3e} {:>11.3e}  {:>4d}".format(
                name, str(shp), r, a, k))
        print("  {:-<54}".format(""))
        print("  worst relative error (components above {:.0e}) : {:.3e}   tol {:.0e}"
              .format(FLOOR, worst_rel, rel_tol))
        print("  worst absolute error (all components)        : {:.3e}   tol {:.0e}"
              .format(worst_abs, abs_tol))
        print("  " + ("PASS" if ok else "FAIL"))
    return ok, worst_rel, worst_abs


# ==============================================================================
# SECTION 7 -- TRAINING  (with the 1430 decision built into the loop)
# ==============================================================================

def train(model, train_b, val_b, epochs=160, lr=0.05, batch=64,
          F_threshold=6.0, min_epoch=15, plateau=0.02, seed=11, verbose=True):
    rng = np.random.default_rng(seed)
    n = len(train_b["y"])
    # Adam
    m = {k: np.zeros_like(model.get(k), dtype=float) for k in model.PARAMS}
    v = {k: np.zeros_like(model.get(k), dtype=float) for k in model.PARAMS}
    b1, b2, eps_a, step = 0.9, 0.999, 1e-8, 0
    history = []

    for ep in range(1, epochs + 1):
        # cosine decay: the last years of the debate were spent on refinement
        lr_t = lr * (0.5 * (1 + np.cos(np.pi * (ep - 1) / epochs)) * 0.9 + 0.1)
        perm = rng.permutation(n)
        for i in range(0, n, batch):
            sel = perm[i:i + batch]
            mb = {k: (val[sel] if isinstance(val, np.ndarray) else val)
                  for k, val in train_b.items()}
            loss, g, _ = model.loss_and_grad(mb)
            step += 1
            for k in model.PARAMS:
                if k in ("Gland", "Gyear", "Aland") and not model.grid_open:
                    continue                       # the grid does not exist yet
                m[k] = b1 * m[k] + (1 - b1) * g[k]
                v[k] = b2 * v[k] + (1 - b2) * g[k] ** 2
                mh = m[k] / (1 - b1 ** step)
                vh = v[k] / (1 - b2 ** step)
                setattr(model, k, model.get(k) - lr_t * mh / (np.sqrt(vh) + eps_a))

        # ---- end of epoch: read the residuals by province, not in aggregate --
        yh, _ = model.forward(train_b)
        resid = train_b["y"] - yh
        Fstat, means = misspecification_F(resid, train_b["strat"], model.n_strata)
        yv, _ = model.forward(val_b)
        val_mse = float(np.mean((yv - val_b["y"]) ** 2))
        history.append((ep, float(np.mean(resid ** 2)), val_mse, float(Fstat)))

        # THE 1430 DECISION.  Two conditions, both of which held historically:
        # the aggregate fit must already be good (he had his majority), and the
        # residual must still be structured by province (they still disagreed).
        tr_mse = float(np.mean(resid ** 2))
        settled = (ep >= min_epoch and len(history) > 6 and
                   abs(history[-6][1] - tr_mse) / max(history[-6][1], 1e-9) < plateau)
        if (not model.grid_open) and settled and Fstat > F_threshold:
            model.grid_open = True
            model.grid_opened_epoch = ep
            if verbose:
                print("\n  [epoch {:3d}]  MISSPECIFICATION DETECTED   F = {:.1f}"
                      .format(ep, Fstat))
                print("               the aggregate has settled (mse {:.4f}) and "
                      "the provinces still disagree.".format(tr_mse))
                worst = int(np.argmax(np.abs(means)))
                print("               worst stratum: {} (mean residual {:+.3f})"
                      .format(PROVINCES[worst], means[worst]))
                print("               -> declining to promulgate the flat rate; "
                      "opening the 6 x 9 grid.\n")

        if verbose and (ep % 20 == 0 or ep == 1):
            print("  epoch {:3d} | train mse {:.4f} | val mse {:.4f} | F {:7.2f} | grid {}"
                  .format(ep, np.mean(resid ** 2), val_mse, Fstat,
                          "open" if model.grid_open else "shut"))
    return history


# ==============================================================================
# SECTION 8 -- EVALUATIONS
# ==============================================================================

def eval_heldout_letters(model, rng):
    """
    "A wise man will acquaint himself with them before the morning is over."
    The real test of a derived table: can the model read a letter that was
    never in its training data?  These tokens contain letters the model has
    provably never seen -- their vectors exist only because the chart says how
    to build them.
    """
    seen = make_dataset(600, T=4, rng=rng, token_pool=TRAIN_TOKENS)
    unseen = make_dataset(600, T=4, rng=rng, token_pool=HELDOUT_TOKENS)
    out = {}
    for name, b in (("seen letters", seen), ("UNSEEN letters", unseen)):
        yh, _ = model.forward(b)
        mse = float(np.mean((yh - b["y"]) ** 2))
        var = float(np.var(b["y"]))
        out[name] = (mse, 1.0 - mse / var)
    return out


def learnability_curve(model, rng, tol=0.05):
    """
    How many exemplars does a cold learner need before it can regenerate the
    whole table?  We hand a naive ridge-regression learner k (chart-row, vector)
    pairs and ask it to reconstruct all 30 symbol vectors.

    This is the quantity Sejong put a public number on in 1446, and it is the
    quantity a derived representation is designed to minimise.
    """
    Ec, Ev = model.derive_codebook()
    X = np.vstack([np.hstack([CP, CF, np.zeros((NC, R + 1))]),
                   np.hstack([np.zeros((NV, P + F)), CV, POL[:, None]])])
    Y = np.vstack([Ec, Ev])
    n_all = X.shape[0]
    scale = np.mean(Y ** 2)
    curve = []
    for k in range(2, n_all + 1):
        errs = []
        for _ in range(30):
            sel = rng.choice(n_all, size=k, replace=False)
            Xs, Ys = X[sel], Y[sel]
            W = np.linalg.solve(Xs.T @ Xs + 1e-6 * np.eye(X.shape[1]), Xs.T @ Ys)
            errs.append(np.mean((X @ W - Y) ** 2) / scale)
        curve.append((k, float(np.mean(errs))))
    enough = next((k for k, e in curve if e < tol), None)
    return curve, enough


def ablation_flat_vs_grid(train_b, val_b, seed=5):
    """The counterfactual Sejong refused to run in 1430: promulgate the flat
    rate on the majority and see what it does to the provinces that objected."""
    flat = JeongeumKongbeopNet(seed=seed)
    train(flat, train_b, val_b, F_threshold=1e18, verbose=False)
    return flat


def per_province_report(model, b):
    """Mean residual per province, and the residual's SLOPE against the harvest
    grade.  The slope is the ministers' objection in one number: a scheme that
    is right on average can still be wrong in exactly the years that matter,
    and a scheme whose error grows as the harvest fails is the worst of all."""
    yh, _ = model.forward(b)
    r = b["y"] - yh
    means, slopes = [], []
    for s in range(len(PROVINCES)):
        m = b["strat"] == s
        means.append(r[m].mean())
        yr = b["year"][m].astype(float)
        slopes.append(np.polyfit(yr, r[m], 1)[0] if m.sum() > 2 else 0.0)
    return float(np.mean(r ** 2)), np.array(means), np.array(slopes)


def ablation_derived_vs_table(train_b, val_b, seed=5):
    """The control: identical network, identical training, the ONLY difference
    being a free row per symbol instead of a derived one."""
    ctrl = JeongeumKongbeopNet(seed=seed, free_table=True)
    train(ctrl, train_b, val_b, verbose=False)
    return ctrl


# ==============================================================================
# SECTION 9 -- SELF-TESTS
# ==============================================================================

def self_tests():
    ok = True

    # 1. The chart is well formed and the base icons really are unmarked.
    for gl, pl, fe, _ in CONSONANTS:
        assert pl in PLACES and len(fe) == F
    bases = [g for g, _, fe, irr in CONSONANTS if sum(fe) == 0 and not irr]
    assert set(bases) == {"g", "n", "m", "s", "ng"}, bases
    print("  [1] chart well formed; five unmarked base icons recovered  OK")

    # 2. Stroke addition is literally vector addition in the codebook.
    mdl = JeongeumKongbeopNet(seed=2)
    Ec, _ = mdl.derive_codebook()
    lhs = Ec[CONS_IDX["k"]] - Ec[CONS_IDX["g"]]        # velar  +aspirate
    rhs = Ec[CONS_IDX["ch"]] - Ec[CONS_IDX["j"]]       # dental +aspirate
    assert np.allclose(lhs, rhs, atol=1e-12), np.abs(lhs - rhs).max()
    print("  [2] +aspirate is the SAME vector at every place of articulation OK")

    # 3. No per-symbol parameters exist anywhere in the model.
    derived, table, ratio = mdl.codebook_compression()
    assert derived < table
    print("  [3] codebook uses {} params, a lookup table would use {} ({:.2f}x) OK"
          .format(derived, table, ratio))

    # 4. Held-out symbols are constructible without ever being trained on.
    assert all(c in CONS_IDX for c in HELDOUT_CONS)
    v = Ec[CONS_IDX["p"]]
    assert np.allclose(v, mdl.A[PLACES.index("labial")] + mdl.S[0] + mdl.S[1])
    print("  [4] an unseen letter is exactly base-icon + its strokes         OK")

    # 5. The misspecification statistic separates noise from structure.
    rng = np.random.default_rng(0)
    st = rng.integers(0, 8, 800)
    F_noise, _ = misspecification_F(rng.normal(0, 1, 800), st, 8)
    F_struct, _ = misspecification_F(rng.normal(0, 1, 800) + st * 0.9, st, 8)
    assert F_noise < 5 and F_struct > 50, (F_noise, F_struct)
    print("  [5] F = {:.2f} on noise, {:.1f} on structured dissent           OK"
          .format(F_noise, F_struct))

    # 6. Vowel polarity is a single shared direction (yin/yang).
    _, Ev = mdl.derive_codebook()
    a = Ev[VOW_IDX["O"]] - Ev[VOW_IDX["U"]]
    b = Ev[VOW_IDX["A"]] - Ev[VOW_IDX["EO"]]
    assert np.allclose(a, b, atol=1e-12)
    print("  [6] yin/yang is one shared direction across all vowel pairs     OK")
    return ok


# ==============================================================================
# SECTION 10 -- MAIN
# ==============================================================================

def main():
    np.set_printoptions(precision=3, suppress=True)
    bar = "=" * 78

    print(bar)
    print(" JEONGEUM-KONGBEOP NET  --  chapter 0399, King Sejong the Great")
    print(bar)

    print("\n[A] SELF-TESTS")
    self_tests()

    print("\n[B] GRADIENT CHECK  (central differences, grid forced open)")
    passed, wr, wa = gradient_check()
    if not passed:
        raise SystemExit("gradient check FAILED (rel {:.3e}, abs {:.3e})".format(wr, wa))
    print("\n  control condition (free lookup table):")
    ok2, wr2, wa2 = gradient_check(free_table=True, verbose=False)
    print("  worst rel {:.3e} / abs {:.3e}   {}".format(wr2, wa2,
          "PASS" if ok2 else "FAIL"))
    if not ok2:
        raise SystemExit("control gradient check FAILED")

    rng = np.random.default_rng(99)
    train_b = make_dataset(3000, T=4, rng=rng)
    val_b = make_dataset(1000, T=4, rng=rng)

    print("\n[C] TRAINING  (the grid stays shut until the residuals earn it)")
    model = JeongeumKongbeopNet(seed=5)
    hist = train(model, train_b, val_b, epochs=140)

    print("\n[D] THE 1430 COUNTERFACTUAL  (promulgate the flat rate anyway)")
    flat = ablation_flat_vs_grid(train_b, val_b)
    flat_mse, flat_mu, flat_sl = per_province_report(flat, val_b)
    grid_mse, grid_mu, grid_sl = per_province_report(model, val_b)
    print("  {:<13} {:>9} {:>9} | {:>9} {:>9}".format(
        "province", "flat bias", "flat slope", "grid bias", "grid slope"))
    for s_ in range(len(PROVINCES)):
        print("  {:<13} {:>+9.3f} {:>+10.3f} | {:>+9.3f} {:>+10.3f}".format(
            PROVINCES[s_], flat_mu[s_], flat_sl[s_], grid_mu[s_], grid_sl[s_]))
    print("  {:-<56}".format(""))
    print("  worst provincial bias  : {:+.3f}   ->  {:+.3f}".format(
        flat_mu[np.argmax(np.abs(flat_mu))], grid_mu[np.argmax(np.abs(grid_mu))]))
    print("  spread of bias (max-min): {:.3f}    ->  {:.3f}".format(
        flat_mu.max() - flat_mu.min(), grid_mu.max() - grid_mu.min()))
    print("  validation mse         : {:.4f}   ->  {:.4f}".format(flat_mse, grid_mse))
    print("  grid opened at epoch   : {}".format(model.grid_opened_epoch))
    print("  the flat rate under-assesses the fertile south and over-assesses the")
    print("  poor north, which is the objection the northern provinces actually")
    print("  filed in 1430, recovered here from the residuals alone.")

    print("\n[E] READING A LETTER THAT WAS NEVER TAUGHT")
    ctrl = ablation_derived_vs_table(train_b, val_b)
    print("  {:<26} {:>12} {:>12}".format("", "derived chart", "free table"))
    r_d = eval_heldout_letters(model, np.random.default_rng(31))
    r_c = eval_heldout_letters(ctrl, np.random.default_rng(31))
    for k in r_d:
        print("  {:<26} {:>+12.3f} {:>+12.3f}".format(
            "R^2, " + k, r_d[k][1], r_c[k][1]))
    print("  {:-<52}".format(""))
    d_par = (P + F + R + 1) * model.d
    t_par = (NC + NV) * model.d
    print("  codebook parameters        {:>12d} {:>12d}".format(d_par, t_par))
    print("  the two networks are identical apart from the codebook.  Both read")
    print("  familiar letters well.  Only the derived one can read a letter it")
    print("  was never shown, because its vector is a base icon plus its strokes")
    print("  rather than a row that was never filled in.")

    print("\n[F] COST OF ACQUIRING THE DECODER")
    curve, enough = learnability_curve(model, np.random.default_rng(4))
    for k, e in curve:
        if k in (2, 4, 6, 8, 9, 10, 12, 16, 20, 30):
            print("   {:2d} exemplars -> reconstruction error {:.4f}".format(k, e))
    print("  the whole table is recoverable from {} exemplars out of {}."
          .format(enough, NC + NV))
    print("  the derivation is published with the artefact, so acquiring the")
    print("  decoder is a bounded cost that can be quoted in advance -- which is")
    print("  exactly what the 1446 postface did when it promised a morning.")

    print("\n" + bar)
    print(" done.")
    print(bar)


if __name__ == "__main__":
    main()
