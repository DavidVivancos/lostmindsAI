#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
 THE JAMHARA ENGINE
 A trainable artificial-neuron architecture after Ibn Duraid al-Azdi
 (Abu Bakr Muhammad ibn al-Hasan ibn Duraid, Basra 223 AH / 837 CE -
  Baghdad 321 AH / 933 CE)

 ========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 12 Minds 221 - 240 Available on Amazon https://www.amazon.com/dp/B0HKVBKNTT
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0229_ibn_duraid_837 - Ibn Duraid al-Azdi (c.837-933)
================================================================================  

WHY THIS ARCHITECTURE AND NOT ANOTHER
-------------------------------------
Ibn Duraid inherited from his tradition a machine for enumerating the entire
combinatorial space of a language: al-Khalil ibn Ahmad's Kitab al-'Ayn walks
every permutation of every set of radicals and marks which are used and which
are neglected. Ibn Duraid, who examined the 'Ayn manuscript at Basra in 248 AH
and was named among the scholars who corrected it, then built something that
inverts it at both ends. His own dictionary is titled for its policy: the
*jamhara*, the crowd, the main body. He keeps the circulating word and drops
the rare one, and he lets tribal variants and arabicized loanwords stand as
full members of that crowd. At the same time he refuses, in a separate book,
to allow any token to be called meaningless: a proper name -- the canonical
"arbitrary label" of later linguistics -- must decompose into a root that says
something, and he wrote a whole onomasticon doing exactly that against
opponents who mocked Arab names as absurd. When a name genuinely does not
derive inside Arabic, he reports the foreign origin rather than forging one.
And he characteristically leaves *several* derivations of one name standing
side by side without choosing. A fourth book, on words that can be honoured in
two senses at once by a person compelled to swear, makes the last move
explicit: which sense is in force is fixed by the speaker's inner indexation,
not by the acoustic string.

So this network is not a Transformer, and attention over stored keys is not its
organising idea. Its organising idea is a LEXICON AS A CENSUS WITH A BUDGET,
feeding a DERIVATION LEDGER THAT IS FORBIDDEN TO COLLAPSE, read out through an
INTENTION GATE. Six named parts:

  0. TA'ARRUF AL-WAZN  (template recognition)
     A radical sits at a different place in the string under every template,
     so nothing can find the radicals until it knows the weave. A bag-of-
     letters read gives a soft distribution over templates, and that
     distribution mixes the queries used by part 1. This is the order the
     tradition itself assumes: you recognise the wazn, then you read the
     radicals off it.

  1. ISTIKHRAJ AL-JIDHR  (radical extraction)
     Three template-conditioned attention heads pull the consonantal radicals
     out of a surface string, learning for themselves which positions are root
     and which are servile. This is morphological segmentation, learned.

  2. HALQAT AL-TAQLIB  (the permutation orbit, S3 weight sharing)
     The six orderings of one radical set are ONE object with six addresses.
     Three codes are built from the extracted radicals and summed:
       - a permutation-INVARIANT sum-pooled code, which carries the orbit and
         is blind to the ordering by construction;
       - a permutation-EQUIVARIANT slot-weighted code, which carries which
         radical sat where;
       - a COMPARISON RING over all six ordered pairs of radicals, because
         which ordering you are looking at is a relational fact and no amount
         of sum-pooling will express it. Taqlib in the tradition is literally
         the operation of comparing and rotating the radicals against each
         other, so the network is given that operation rather than asked to
         reinvent it. The pair comparator's weights are shared across all six
         pairs.
     Together these buy zero-shot compositional transfer: the network can
     identify an ordering it has never been trained on, because it has seen
     that orbit and it has, separately, seen that relational pattern.

  3. BAWWABAT AL-JAMHARA  (the circulation gate)
     A learned threshold on how often a form circulates routes it either to
     dense private capacity (the crowd, al-lafz al-sha'i') or to a shared
     low-rank fallback (the wild rarities, al-gharib al-nadir). A budget
     penalty forces the threshold to bite. This is Ibn Duraid's editorial
     policy implemented as a capacity-allocation rule.

  4. MIZAN AL-ISHTIQAQ  (the derivation ledger, uncollapsed)
     K competing (orbit, ordinal) analyses are retained per token with
     renormalised scores, plus a separate AJNABI head that may declare the
     form underivable inside the lexicon. An explicit entropy FLOOR penalises
     premature collapse onto one reading. The ledger is meant to stay open.

  5. BAWWABAT AL-NIYYA  (the intention gate, dual read-out)
     The same retained ledger is read twice against two different contexts: a
     public one (what the hearer takes the word to mean) and an inner one
     (what the speaker indexes it to). Under a coercion flag the two targets
     differ, and only genuinely two-sensed forms can satisfy both.

  6. QAYD AL-MAQSURA  (the constrained decoder)
     A non-autoregressive decoder reconstructs the surface form, and at decode
     time every emitted string is hard-masked to terminate in one designated
     rhyme class -- the constraint of his 250-line poem in which every line
     ends on the same restricted shape. We then measure whether meaning
     survives the narrowed channel.

EVERYTHING IS PURE NUMPY. Gradients are derived and written by hand. A
finite-difference check over every parameter block runs on every execution and
must pass before training starts.

RUN:  python3 chapter_0229_ibn_duraid_837.py
      python3 chapter_0229_ibn_duraid_837.py --quick      (fast smoke run)
===============================================================================
"""

import sys
import time
import numpy as np

# =============================================================================
# SECTION 0. GLOBAL CONFIGURATION
# =============================================================================

CFG = dict(
    # --- alphabet -----------------------------------------------------------
    N_RADICAL   = 12,   # letters that may serve as root consonants
    N_SERVILE   = 5,    # vowels / affix letters that never count as radicals
    T           = 7,    # max surface length (padded)
    # --- lexicon ------------------------------------------------------------
    N_ORBIT     = 24,   # distinct radical multisets (each has 6 orderings)
    N_PERMIDX   = 6,    # orderings within an orbit: |S_3| = 6
    N_PATTERN   = 6,    # awzan, morphological templates
    # --- widths -------------------------------------------------------------
    D_EMB       = 40,
    D_HID       = 64,
    D_SENSE     = 10,
    D_CTX       = 10,
    D_CMP       = 12,
    D_RING      = 32,   # hidden width of the ordinal read-out   # width of each pairwise radical comparator
    RANK_WAHSHI = 6,    # rank of the shared fallback path for rare forms
    LEDGER_K    = 4,    # derivations kept standing side by side
    # --- objective weights --------------------------------------------------
    W_WAZN      = 0.5,
    W_ORBIT     = 1.0,
    W_PERMIDX   = 1.0,
    W_PATTERN   = 0.6,
    W_AJNABI    = 0.8,
    W_SENSE_PUB = 1.0,
    W_SENSE_IN  = 1.0,
    W_REC       = 0.5,
    W_FLOOR     = 0.60,  # anti-collapse: keep the ledger open
    W_BUDGET    = 2.0,   # force the circulation threshold to bite
    ENTROPY_FLOOR = 0.65, # nats; ~ the entropy of a 2-way split inside K=4
    JAMHARA_BUDGET = 0.45, # target fraction of tokens granted dense capacity
    LEDGER_TAU  = 1.0,
    NIYYA_GAIN  = 2.0,
    # --- training -----------------------------------------------------------
    SEED        = 229,
    LR          = 0.035,
    BETA1       = 0.9,
    BETA2       = 0.999,
    EPS         = 1e-8,
    EPOCHS      = 1400,
    BATCH       = 96,
)

# the six ordered pairs of radical slots: the taqlib ring
PAIRS  = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
PAIR_L = np.array([p[0] for p in PAIRS])
PAIR_R = np.array([p[1] for p in PAIRS])

PAD_ID = CFG["N_RADICAL"] + CFG["N_SERVILE"]           # 17
VOCAB  = PAD_ID + 1                                     # 18
RHYME_CLASS = [CFG["N_RADICAL"] + 0, CFG["N_RADICAL"] + 1]  # the maqsura endings


# =============================================================================
# SECTION 1. PRIMITIVES
#   Small, explicit numerical helpers. Nothing here is clever; everything here
#   is checked by the finite-difference test in Section 5.
# =============================================================================

def softmax(z, axis=-1):
    """Numerically stable softmax."""
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def softmax_ce(logits, targets, weights=None):
    """
    Cross entropy over the last axis.
      logits : (..., C)
      targets: (...,) integer class indices
      weights: (...,) optional per-example weight, used to switch a term off
               for examples where it is undefined (e.g. the root of a word
               that has no Arabic root at all).
    Returns (scalar_loss, dlogits).
    """
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_t = targets.reshape(-1)
    n = flat_logits.shape[0]
    p = softmax(flat_logits, axis=-1)
    if weights is None:
        w = np.ones(n)
    else:
        w = weights.reshape(-1).astype(np.float64)
    denom = max(w.sum(), 1e-9)
    ll = -np.log(np.maximum(p[np.arange(n), flat_t], 1e-12))
    loss = float(np.sum(w * ll) / denom)
    d = p.copy()
    d[np.arange(n), flat_t] -= 1.0
    d *= (w / denom)[:, None]
    return loss, d.reshape(logits.shape)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def bce_with_logits(logit, target):
    """Binary cross entropy. logit,(N,1); target,(N,1). Returns (loss, dlogit)."""
    p = sigmoid(logit)
    loss = float(np.mean(-(target * np.log(np.maximum(p, 1e-12))
                           + (1 - target) * np.log(np.maximum(1 - p, 1e-12)))))
    d = (p - target) / logit.size
    return loss, d


def mse(pred, target):
    diff = pred - target
    loss = float(np.mean(diff ** 2))
    d = 2.0 * diff / diff.size
    return loss, d


def entropy_of(p, axis=-1):
    return -np.sum(p * np.log(np.maximum(p, 1e-12)), axis=axis)


# =============================================================================
# SECTION 2. THE CORPUS
#   A synthetic but structurally faithful Semitic-style morphology. It is
#   synthetic because we need ground truth for roots, orderings, patterns,
#   senses and foreign origin all at once, which no real corpus hands over.
#   It is faithful because the generative process is the one Ibn Duraid's
#   tradition describes: a radical multiset, an ordering of it, a template
#   woven around it with servile letters, and a sense that depends on all
#   three. Circulation follows a Zipf law, as vocabulary in use does.
# =============================================================================

class BasraCorpus:
    """
    Builds the lexicon and samples training/eval batches.

    Terminology used throughout, matching the tradition:
      orbit    -- a radical multiset {r1,r2,r3}; its six orderings are its
                  taqalib (permutations).
      permidx  -- which of the six orderings this word uses.
      pattern  -- the wazn, the template that weaves servile letters around
                  the radicals.
      musta'mal / muhmal -- an ordering that is actually used / left neglected.
      ajnabi   -- a form with no derivation inside the lexicon at all.
    """

    PERMS = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]

    def __init__(self, cfg, rng):
        self.cfg = cfg
        self.rng = rng
        nr, ns = cfg["N_RADICAL"], cfg["N_SERVILE"]
        self.serviles = [nr + i for i in range(ns)]

        # --- radical multisets (orbits) -------------------------------------
        seen, orbits = set(), []
        while len(orbits) < cfg["N_ORBIT"]:
            trip = tuple(sorted(rng.choice(nr, size=3, replace=False).tolist()))
            if trip not in seen:
                seen.add(trip)
                orbits.append(trip)
        self.orbits = orbits

        # --- which orderings are used, which are neglected ------------------
        # Al-Khalil enumerated all six and marked the unused. We keep that
        # distinction: roughly two thirds of orderings are live words.
        self.used = np.zeros((cfg["N_ORBIT"], 6), dtype=bool)
        for o in range(cfg["N_ORBIT"]):
            k = rng.integers(3, 7)                       # 3..6 live orderings
            idx = rng.choice(6, size=int(k), replace=False)
            self.used[o, idx] = True

        # --- the latent semantic field --------------------------------------
        S = cfg["D_SENSE"]
        self.A_orbit = rng.normal(0, 1.0, (cfg["N_ORBIT"], S))
        self.A_perm  = rng.normal(0, 0.7, (6, S))
        self.A_pat   = rng.normal(0, 0.7, (cfg["N_PATTERN"], S))
        self.A_mix   = rng.normal(0, 0.5, (S, S))

        # --- circulation: Zipf over orbits ----------------------------------
        ranks = np.arange(1, cfg["N_ORBIT"] + 1)
        freq = 1.0 / ranks ** 1.05
        perm_order = rng.permutation(cfg["N_ORBIT"])
        self.circulation = np.zeros(cfg["N_ORBIT"])
        self.circulation[perm_order] = freq / freq.sum()
        # raw counts as a scribe would tally them in a body of attested text
        self.counts = np.maximum(1.0, np.round(self.circulation * 20000.0))

        # --- held-out orderings for the zero-shot orbit-transfer test -------
        # Each of these (orbit, permidx) pairs is a LIVE word that the network
        # never sees during training. Both factors are seen separately; the
        # combination is not. That is the taqlib generalisation claim.
        held = []
        for o in range(cfg["N_ORBIT"]):
            live = np.where(self.used[o])[0]
            if len(live) >= 4 and o % 2 == 0:
                held.append((o, int(live[-1])))
        self.heldout = set(held)

    # -- sense ---------------------------------------------------------------
    def sense(self, orbit, permidx, pattern):
        """Ground-truth meaning vector. Additive in the three factors, with a
        genuine interaction term so that the network cannot win by pure
        addition."""
        base = (self.A_orbit[orbit] + self.A_perm[permidx] + self.A_pat[pattern])
        inter = (self.A_orbit[orbit] * self.A_perm[permidx]) @ self.A_mix
        return np.tanh(base + 0.5 * inter)

    # -- surface realisation -------------------------------------------------
    def weave(self, radicals, pattern):
        """Weave servile letters around the three radicals per template."""
        r1, r2, r3 = radicals
        s = self.serviles
        if pattern == 0:   # fa'ala
            seq = [r1, s[0], r2, s[0], r3]
        elif pattern == 1: # maf'al
            seq = [s[1], r1, s[2], r2, r3]
        elif pattern == 2: # fa''al, gemination of the middle radical
            seq = [r1, r2, r2, s[0], r3]
        elif pattern == 3: # fa'ila + ta marbuta
            seq = [r1, s[0], r2, s[3], r3, s[4]]
        elif pattern == 4: # af'al
            seq = [s[2], r1, r2, s[0], r3]
        else:              # fa'lan
            seq = [r1, r2, s[1], r3, s[2]]
        return seq + [PAD_ID] * (self.cfg["T"] - len(seq))

    def foreign_form(self):
        """A form built by a process outside this morphology. Ibn Duraid, asked
        about such a name, says where it came from rather than forcing a root:
        Shurahbil, he reports, is not native Arabic. The network is given the
        same option."""
        nr = self.cfg["N_RADICAL"]
        body = self.rng.choice(nr, size=3, replace=True).tolist()   # repeats OK
        seq = [body[0], body[1], self.serviles[2], body[2],
               self.serviles[1], self.serviles[4]]
        return seq + [PAD_ID] * (self.cfg["T"] - len(seq))

    # -- sampling ------------------------------------------------------------
    def sample(self, n, split="train", coercion_rate=0.35, foreign_rate=0.16):
        """
        Returns a dict of arrays. `split` controls whether held-out taqalib are
        excluded (train) or exclusively used (orbit_transfer).
        """
        cfg = self.cfg
        X      = np.full((n, cfg["T"]), PAD_ID, dtype=np.int64)
        y_orb  = np.zeros(n, dtype=np.int64)
        y_pix  = np.zeros(n, dtype=np.int64)
        y_pat  = np.zeros(n, dtype=np.int64)
        y_aj   = np.zeros((n, 1))
        counts = np.zeros((n, 1))
        s_pub  = np.zeros((n, cfg["D_SENSE"]))
        s_in   = np.zeros((n, cfg["D_SENSE"]))
        ctx_p  = np.zeros((n, cfg["D_CTX"]))
        ctx_i  = np.zeros((n, cfg["D_CTX"]))
        coerce = np.zeros((n, 1))
        two_sense = np.zeros((n, 1))

        probs = self.circulation / self.circulation.sum()

        for i in range(n):
            foreign = (split == "train" or split == "dev") and \
                      (self.rng.random() < foreign_rate)
            if split == "foreign":
                foreign = True
            if split == "orbit_transfer":
                foreign = False

            if foreign:
                X[i] = self.foreign_form()
                y_aj[i, 0] = 1.0
                # an unowned form circulates rarely
                counts[i, 0] = float(self.rng.integers(1, 40))
                # its sense is not recoverable from the lexicon: target is the
                # neutral vector, and the network is scored on saying so
                y_orb[i] = 0
                y_pix[i] = 0
                y_pat[i] = 0
                continue

            # choose a live ordering, honouring the split
            while True:
                o = int(self.rng.choice(cfg["N_ORBIT"], p=probs))
                live = np.where(self.used[o])[0]
                if len(live) == 0:
                    continue
                p_i = int(self.rng.choice(live))
                is_held = (o, p_i) in self.heldout
                if split == "orbit_transfer" and is_held:
                    break
                if split in ("train", "dev", "foreign") and not is_held:
                    break

            pat = int(self.rng.integers(0, cfg["N_PATTERN"]))
            trip = self.orbits[o]
            order = self.PERMS[p_i]
            radicals = [trip[order[0]], trip[order[1]], trip[order[2]]]

            X[i] = self.weave(radicals, pat)
            y_orb[i], y_pix[i], y_pat[i] = o, p_i, pat
            counts[i, 0] = self.counts[o]

            true_sense = self.sense(o, p_i, pat)
            s_pub[i] = true_sense
            s_in[i] = true_sense

            # --- al-Malahin: the two-sensed form under compulsion -----------
            # A form is two-sensed if some OTHER live ordering of the same
            # orbit exists. Then the speaker may utter it while indexing the
            # sibling reading. Under coercion the public target is the
            # coercer's reading and the inner target is the speaker's.
            siblings = [q for q in np.where(self.used[o])[0]
                        if q != p_i and (o, q) not in self.heldout]
            if siblings:
                two_sense[i, 0] = 1.0
            if siblings and self.rng.random() < coercion_rate:
                q = int(self.rng.choice(siblings))
                coerce[i, 0] = 1.0
                s_in[i] = self.sense(o, q, pat)
                ctx_i[i, :cfg["D_SENSE"]] = self.A_perm[q]
            ctx_p[i, :cfg["D_SENSE"]] = self.A_perm[p_i]
            if coerce[i, 0] == 0.0:
                ctx_i[i] = ctx_p[i]

        return dict(X=X, y_orb=y_orb, y_pix=y_pix, y_pat=y_pat, y_aj=y_aj,
                    counts=counts, s_pub=s_pub, s_in=s_in, ctx_p=ctx_p,
                    ctx_i=ctx_i, coerce=coerce, two_sense=two_sense)


# =============================================================================
# SECTION 3. THE JAMHARA ENGINE
# =============================================================================

class JamharaEngine:
    """The six named parts, forward and backward, by hand."""

    def __init__(self, cfg, rng):
        self.cfg = cfg
        c = cfg
        D, H, S, C = c["D_EMB"], c["D_HID"], c["D_SENSE"], c["D_CTX"]
        r = c["RANK_WAHSHI"]
        HC = c["D_CMP"]
        NO, NP, PT = c["N_ORBIT"], c["N_PERMIDX"], c["N_PATTERN"]

        def g(*shape, scale=None):
            fan = shape[-2] if len(shape) >= 2 else shape[0]
            sc = scale if scale is not None else (1.0 / np.sqrt(fan))
            return rng.normal(0, sc, shape)

        self.P = {
            # 0. ta'arruf al-wazn ---------------------------------------
            "Emb":  g(VOCAB, D, scale=0.35),
            "Pos":  g(c["T"], D, scale=0.20),
            "Wwz":  g(D, PT), "bwz": np.zeros(PT),
            # 1. istikhraj al-jidhr --------------------------------------
            "Q":    g(3, D, scale=0.30),          # shared prior over slots
            "Qtab": g(PT, 3, D, scale=0.30),      # one query set per template
            # 2. halqat al-taqlib ---------------------------------------
            "Wm":   g(D, H),              # SHARED across all three slots
            "Wo":   g(3, D, H),           # slot-specific, carries the ordering
            "Wa":   g(D, c["D_CMP"]),     # comparator, left  member of a pair
            "Wb":   g(D, c["D_CMP"]),     # comparator, right member of a pair
            "bcmp": np.zeros(c["D_CMP"]),
            "Wcmp": g(6 * c["D_CMP"], H), # the ring of six ordered pairs
            "bz":   np.zeros(H),
            # 3. bawwabat al-jamhara ------------------------------------
            "wg":   np.array([1.2]),
            "bg":   np.array([-6.0]),
            "Wd":   g(H, H),              # dense private capacity
            "bd":   np.zeros(H),
            "U":    g(H, r),              # shared low-rank fallback
            "V":    g(r, H),
            # 4. mizan al-ishtiqaq --------------------------------------
            "Worb": g(H, NO), "borb": np.zeros(NO),
            "Wcr":  g(6 * c["D_CMP"], c["D_RING"]),
            "bcr":  np.zeros(c["D_RING"]),
            "Wpix": g(c["D_RING"], NP), "bpix": np.zeros(NP),
            "Wpat": g(H, PT), "bpat": np.zeros(PT),
            "waj":  g(H, 1),  "baj":  np.zeros(1),
            # sense lexicon, compositional so unseen combinations still
            # receive a meaning
            "SO":   g(NO, S, scale=0.5),
            "SP":   g(NP, S, scale=0.5),
            "ST":   g(PT, S, scale=0.5),
            "Wx":   g(S, S, scale=0.3),
            # 5. bawwabat al-niyya --------------------------------------
            "Wc":   g(C, S),
            # 6. qayd al-maqsura ----------------------------------------
            "Wdec": g(H, c["T"] * VOCAB),
            "bdec": np.zeros(c["T"] * VOCAB),
        }
        self.keys = list(self.P.keys())
        self._init_adam()

    def _init_adam(self):
        self.m = {k: np.zeros_like(v) for k, v in self.P.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.P.items()}
        self.t = 0

    # -------------------------------------------------------------------
    # FORWARD
    # -------------------------------------------------------------------
    def forward(self, b):
        c, P = self.cfg, self.P
        X, counts = b["X"], b["counts"]
        B, T = X.shape
        H, S, K = c["D_HID"], c["D_SENSE"], c["LEDGER_K"]
        NO, NP = c["N_ORBIT"], c["N_PERMIDX"]
        ca = {}

        # --- 0. TA'ARRUF AL-WAZN: which weave is this? ---------------------
        emb = P["Emb"][X]                                  # (B,T,D)
        Hs = emb + P["Pos"][None, :, :]                    # (B,T,D)
        valid = (X != PAD_ID).astype(np.float64)           # (B,T)
        nval = np.maximum(valid.sum(1, keepdims=True), 1.0)
        bag = np.sum(Hs * valid[:, :, None], axis=1) / nval # (B,D)
        wz_log = bag @ P["Wwz"] + P["bwz"]                 # (B,PT)
        wz = softmax(wz_log, axis=-1)                      # (B,PT)

        # --- 1. ISTIKHRAJ AL-JIDHR: pull the radicals out of the string ----
        # the queries are mixed according to the recognised template, so slot
        # j hunts in the place that THIS weave puts its j-th radical
        Qeff = P["Q"][None, :, :] + np.einsum("bp,pjd->bjd", wz, P["Qtab"])
        scores = np.einsum("btd,bjd->bjt", Hs, Qeff) / np.sqrt(c["D_EMB"])
        scores = scores + (valid[:, None, :] - 1.0) * 1e9  # mask the padding
        A = softmax(scores, axis=-1)                       # (B,3,T)
        R = np.einsum("bjt,btd->bjd", A, Hs)               # (B,3,D)
        ca.update(X=X, Hs=Hs, valid=valid, nval=nval, bag=bag,
                  wz_log=wz_log, wz=wz, Qeff=Qeff, A=A, R=R)

        # --- 2. HALQAT AL-TAQLIB: one orbit, six addresses ----------------
        # invariant: the same Wm applied to every slot then summed. Any
        # reordering of the three radicals leaves `minv` untouched.
        minv = np.einsum("bjd,dh->bh", R, P["Wm"])         # (B,H)
        # equivariant: slot-specific weights, so the ordering survives
        oeqv = np.einsum("bjd,jdh->bh", R, P["Wo"])        # (B,H)
        # relational: compare every ordered pair of radicals. The comparator
        # weights are shared; only the slot into which each comparison is
        # written is fixed, so the ring is a genuine taqlib operation.
        L = R[:, PAIR_L, :]                                # (B,6,D)
        Rr = R[:, PAIR_R, :]                               # (B,6,D)
        aa = np.einsum("bpd,dc->bpc", L, P["Wa"])
        bb = np.einsum("bpd,dc->bpc", Rr, P["Wb"])
        u_pre = aa + bb + aa * bb + P["bcmp"]              # second-order
        u = np.tanh(u_pre)                                 # (B,6,HC)
        cmpv = u.reshape(B, -1) @ P["Wcmp"]                # (B,H)
        pre_z0 = minv + oeqv + cmpv + P["bz"]
        z0 = np.tanh(pre_z0)
        ca.update(minv=minv, oeqv=oeqv, L=L, Rr=Rr, aa=aa, bb=bb,
                  u_pre=u_pre, u=u, cmpv=cmpv, pre_z0=pre_z0, z0=z0)

        # --- 3. BAWWABAT AL-JAMHARA: the circulation threshold ------------
        logc = np.log1p(counts)                            # (B,1)
        gate_pre = P["wg"][0] * logc + P["bg"][0]
        gate = sigmoid(gate_pre)                           # (B,1) in [0,1]
        dense = z0 @ P["Wd"] + P["bd"]                     # private capacity
        low_h = z0 @ P["U"]
        low = low_h @ P["V"]                               # shared fallback
        pre_z = gate * dense + (1.0 - gate) * low
        z = np.tanh(pre_z)                                 # (B,H)
        ca.update(logc=logc, gate_pre=gate_pre, gate=gate, dense=dense,
                  low_h=low_h, low=low, pre_z=pre_z, z=z)

        # --- 4. MIZAN AL-ISHTIQAQ: the ledger ------------------------------
        lo_orb = z @ P["Worb"] + P["borb"]                 # (B,NO)
        # the ordinal is read from the comparison ring ALONE, never from the
        # main hidden state. Orbit and position-in-orbit are two quantities
        # with two sources: the sum-pool knows WHICH radicals, the ring knows
        # in WHAT ORDER. Denying the ordinal head access to letter-identity
        # features is what makes it generalise to orderings never attested.
        zc = np.tanh(u.reshape(B, -1) @ P["Wcr"] + P["bcr"])   # (B,D_RING)
        lo_pix = zc @ P["Wpix"] + P["bpix"]                # (B,NP)
        ca["zc"] = zc
        lo_pat = z @ P["Wpat"] + P["bpat"]                 # (B,PT)
        lo_aj  = z @ P["waj"]  + P["baj"]                  # (B,1)

        # the full derivation grid: score(o,p) = score(o) + score(p)
        grid = lo_orb[:, :, None] + lo_pix[:, None, :]     # (B,NO,NP)
        flat = grid.reshape(B, NO * NP)
        # keep K standing side by side; the indices are held constant in the
        # backward pass, which is the usual treatment of a top-k selection
        topk = np.argpartition(-flat, K - 1, axis=1)[:, :K]
        order = np.argsort(-np.take_along_axis(flat, topk, axis=1), axis=1)
        topk = np.take_along_axis(topk, order, axis=1)     # (B,K) sorted
        klog = np.take_along_axis(flat, topk, axis=1)      # (B,K)
        k_orb, k_pix = topk // NP, topk % NP               # (B,K) each
        ca.update(lo_orb=lo_orb, lo_pix=lo_pix, lo_pat=lo_pat, lo_aj=lo_aj,
                  topk=topk, klog=klog, k_orb=k_orb, k_pix=k_pix)

        # --- 5. BAWWABAT AL-NIYYA: read the ledger twice -------------------
        p_pat = softmax(lo_pat, axis=-1)
        st_soft = p_pat @ P["ST"]                          # (B,S)
        SOk = P["SO"][k_orb]                               # (B,K,S)
        SPk = P["SP"][k_pix]                               # (B,K,S)
        prodk = SOk * SPk
        inter = np.einsum("bks,st->bkt", prodk, P["Wx"])
        pre_sense = SOk + SPk + st_soft[:, None, :] + 0.5 * inter
        sense_k = np.tanh(pre_sense)                       # (B,K,S)
        ca.update(p_pat=p_pat, st_soft=st_soft, SOk=SOk, SPk=SPk,
                  prodk=prodk, pre_sense=pre_sense, sense_k=sense_k)

        def read(ctx, tag):
            proj = ctx @ P["Wc"]                           # (B,S)
            bias = np.einsum("bs,bks->bk", proj, sense_k)  # (B,K)
            sel_pre = klog / c["LEDGER_TAU"] + c["NIYYA_GAIN"] * bias
            sel = softmax(sel_pre, axis=-1)                # (B,K)
            out = np.einsum("bk,bks->bs", sel, sense_k)    # (B,S)
            ca[f"proj_{tag}"] = proj
            ca[f"bias_{tag}"] = bias
            ca[f"sel_{tag}"] = sel
            ca[f"out_{tag}"] = out
            return sel, out

        sel_p, out_p = read(b["ctx_p"], "p")
        sel_i, out_i = read(b["ctx_i"], "i")

        # --- 6. QAYD AL-MAQSURA: reconstruct the surface -------------------
        dec = (z @ P["Wdec"] + P["bdec"]).reshape(B, c["T"], VOCAB)
        ca["dec"] = dec

        ca.update(B=B, sel_p=sel_p, sel_i=sel_i, out_p=out_p, out_i=out_i)
        return ca

    # -------------------------------------------------------------------
    # LOSS  (forward part; fills ca with gradients wrt the head outputs)
    # -------------------------------------------------------------------
    def loss(self, b, ca):
        c = self.cfg
        B = ca["B"]
        native = (1.0 - b["y_aj"][:, 0])                   # 1 where derivable
        parts, G = {}, {}

        l_wz,  d_wz  = softmax_ce(ca["wz_log"], b["y_pat"], native)
        l_orb, d_orb = softmax_ce(ca["lo_orb"], b["y_orb"], native)
        l_pix, d_pix = softmax_ce(ca["lo_pix"], b["y_pix"], native)
        l_pat, d_pat = softmax_ce(ca["lo_pat"], b["y_pat"], native)
        l_aj,  d_aj  = bce_with_logits(ca["lo_aj"], b["y_aj"])

        # sense is only scored where a sense exists
        w_s = native[:, None]
        dp = (ca["out_p"] - b["s_pub"]) * w_s
        di = (ca["out_i"] - b["s_in"]) * w_s
        nrm = max(float(native.sum()) * c["D_SENSE"], 1e-9)
        l_sp = float(np.sum(dp ** 2) / nrm)
        l_si = float(np.sum(di ** 2) / nrm)
        d_outp = 2.0 * dp / nrm
        d_outi = 2.0 * di / nrm

        l_rec, d_rec = softmax_ce(ca["dec"], b["X"])

        # --- anti-collapse: the ledger must stay open ---------------------
        # Ibn Duraid lists several derivations of one name and does not pick.
        # We penalise a selector distribution whose entropy falls below a
        # floor, which is a soft version of the same editorial habit.
        Hs_p = entropy_of(ca["sel_p"])                     # (B,)
        short = np.maximum(c["ENTROPY_FLOOR"] - Hs_p, 0.0)
        l_floor = float(np.mean(short ** 2))
        # d l_floor / d H = -2*short/B  ; dH/dsel = -(log sel + 1)
        dH = (-2.0 * short / B)[:, None]
        d_sel_floor = dH * (-(np.log(np.maximum(ca["sel_p"], 1e-12)) + 1.0))

        # --- the budget that makes the threshold bite ---------------------
        gbar = float(np.mean(ca["gate"]))
        l_budget = (gbar - c["JAMHARA_BUDGET"]) ** 2
        d_gate_budget = np.full_like(ca["gate"],
                                     2.0 * (gbar - c["JAMHARA_BUDGET"]) / ca["gate"].size)

        total = (c["W_WAZN"] * l_wz + c["W_ORBIT"] * l_orb + c["W_PERMIDX"] * l_pix
                 + c["W_PATTERN"] * l_pat + c["W_AJNABI"] * l_aj
                 + c["W_SENSE_PUB"] * l_sp + c["W_SENSE_IN"] * l_si
                 + c["W_REC"] * l_rec + c["W_FLOOR"] * l_floor
                 + c["W_BUDGET"] * l_budget)

        parts = dict(wazn=l_wz, orbit=l_orb, permidx=l_pix, pattern=l_pat, ajnabi=l_aj,
                     sense_pub=l_sp, sense_inner=l_si, recon=l_rec,
                     floor=l_floor, budget=l_budget, total=total)
        G = dict(d_wz=c["W_WAZN"] * d_wz, d_orb=c["W_ORBIT"] * d_orb, d_pix=c["W_PERMIDX"] * d_pix,
                 d_pat=c["W_PATTERN"] * d_pat, d_aj=c["W_AJNABI"] * d_aj,
                 d_outp=c["W_SENSE_PUB"] * d_outp,
                 d_outi=c["W_SENSE_IN"] * d_outi,
                 d_rec=c["W_REC"] * d_rec,
                 d_sel_floor=c["W_FLOOR"] * d_sel_floor,
                 d_gate_budget=c["W_BUDGET"] * d_gate_budget)
        return total, parts, G

    # -------------------------------------------------------------------
    # BACKWARD
    # -------------------------------------------------------------------
    def backward(self, b, ca, G):
        c, P = self.cfg, self.P
        B = ca["B"]
        H, S, K = c["D_HID"], c["D_SENSE"], c["LEDGER_K"]
        NO, NP = c["N_ORBIT"], c["N_PERMIDX"]
        g = {k: np.zeros_like(v) for k, v in P.items()}

        dz = np.zeros((B, H))
        d_klog = np.zeros((B, K))
        d_sense_k = np.zeros((B, K, S))

        # --- 6. decoder ----------------------------------------------------
        d_dec = G["d_rec"].reshape(B, -1)
        g["Wdec"] += ca["z"].T @ d_dec
        g["bdec"] += d_dec.sum(0)
        dz += d_dec @ P["Wdec"].T

        # --- 5. the two readings ------------------------------------------
        def unread(ctx, tag, d_out, extra_dsel=None):
            nonlocal d_klog, d_sense_k
            sel, bias = ca[f"sel_{tag}"], ca[f"bias_{tag}"]
            # out = sum_k sel_k * sense_k
            d_sel = np.einsum("bs,bks->bk", d_out, ca["sense_k"])
            d_sense_k += sel[:, :, None] * d_out[:, None, :]
            if extra_dsel is not None:
                d_sel = d_sel + extra_dsel
            # through the softmax
            d_pre = sel * (d_sel - np.sum(d_sel * sel, axis=1, keepdims=True))
            d_klog += d_pre / c["LEDGER_TAU"]
            d_bias = c["NIYYA_GAIN"] * d_pre
            # bias = proj . sense_k
            proj = ca[f"proj_{tag}"]
            d_proj = np.einsum("bk,bks->bs", d_bias, ca["sense_k"])
            d_sense_k += d_bias[:, :, None] * proj[:, None, :]
            g["Wc"] += ctx.T @ d_proj

        unread(b["ctx_p"], "p", G["d_outp"], G["d_sel_floor"])
        unread(b["ctx_i"], "i", G["d_outi"], None)

        # sense_k = tanh(pre_sense)
        d_pre_sense = d_sense_k * (1.0 - ca["sense_k"] ** 2)
        # pre_sense = SOk + SPk + st_soft + 0.5 * (SOk*SPk) @ Wx
        d_st = d_pre_sense.sum(axis=1)                       # (B,S)
        d_inter = 0.5 * d_pre_sense
        g["Wx"] += np.einsum("bks,bkt->st", ca["prodk"], d_inter)
        d_prod = np.einsum("bkt,st->bks", d_inter, P["Wx"])
        d_SOk = d_pre_sense + d_prod * ca["SPk"]
        d_SPk = d_pre_sense + d_prod * ca["SOk"]
        np.add.at(g["SO"], ca["k_orb"].reshape(-1), d_SOk.reshape(-1, S))
        np.add.at(g["SP"], ca["k_pix"].reshape(-1), d_SPk.reshape(-1, S))
        # st_soft = softmax(lo_pat) @ ST
        g["ST"] += ca["p_pat"].T @ d_st
        d_ppat = d_st @ P["ST"].T
        d_lo_pat = ca["p_pat"] * (d_ppat - np.sum(d_ppat * ca["p_pat"],
                                                  axis=1, keepdims=True))
        d_lo_pat = d_lo_pat + G["d_pat"]

        # --- 4. the ledger --------------------------------------------------
        d_flat = np.zeros((B, NO * NP))
        np.put_along_axis(d_flat, ca["topk"], d_klog, axis=1)
        d_grid = d_flat.reshape(B, NO, NP)
        d_lo_orb = d_grid.sum(axis=2) + G["d_orb"]
        d_lo_pix = d_grid.sum(axis=1) + G["d_pix"]
        d_lo_aj = G["d_aj"]

        for W, dl, bterm in (("Worb", d_lo_orb, "borb"),
                             ("Wpat", d_lo_pat, "bpat"),
                             ("waj",  d_lo_aj,  "baj")):
            g[W] += ca["z"].T @ dl
            g[bterm] += dl.sum(0)
            dz += dl @ P[W].T
        # the ordinal head hangs off the ring, so its gradient never reaches z
        g["Wpix"] += ca["zc"].T @ d_lo_pix
        g["bpix"] += d_lo_pix.sum(0)
        d_zc = (d_lo_pix @ P["Wpix"].T) * (1.0 - ca["zc"] ** 2)
        g["Wcr"] += ca["u"].reshape(B, -1).T @ d_zc
        g["bcr"] += d_zc.sum(0)
        d_u_ring = (d_zc @ P["Wcr"].T).reshape(ca["u"].shape)

        # --- 3. the circulation gate ---------------------------------------
        d_pre_z = dz * (1.0 - ca["z"] ** 2)
        gate = ca["gate"]
        d_gate = np.sum(d_pre_z * (ca["dense"] - ca["low"]), axis=1,
                        keepdims=True) + G["d_gate_budget"]
        d_dense = d_pre_z * gate
        d_low = d_pre_z * (1.0 - gate)

        g["Wd"] += ca["z0"].T @ d_dense
        g["bd"] += d_dense.sum(0)
        dz0 = d_dense @ P["Wd"].T

        g["V"] += ca["low_h"].T @ d_low
        d_low_h = d_low @ P["V"].T
        g["U"] += ca["z0"].T @ d_low_h
        dz0 = dz0 + d_low_h @ P["U"].T

        d_gate_pre = d_gate * gate * (1.0 - gate)
        g["wg"] += np.array([float(np.sum(d_gate_pre * ca["logc"]))])
        g["bg"] += np.array([float(np.sum(d_gate_pre))])

        # --- 2. the orbit encoder -------------------------------------------
        d_pre_z0 = dz0 * (1.0 - ca["z0"] ** 2)
        g["bz"] += d_pre_z0.sum(0)
        g["Wm"] += np.einsum("bjd,bh->dh", ca["R"], d_pre_z0)
        g["Wo"] += np.einsum("bjd,bh->jdh", ca["R"], d_pre_z0)
        dR = (np.einsum("bh,dh->bd", d_pre_z0, P["Wm"])[:, None, :]
              + np.einsum("bh,jdh->bjd", d_pre_z0, P["Wo"]))      # (B,3,D)

        # the comparison ring
        u_flat = ca["u"].reshape(B, -1)
        g["Wcmp"] += u_flat.T @ d_pre_z0
        d_u = (d_pre_z0 @ P["Wcmp"].T).reshape(ca["u"].shape) + d_u_ring
        d_upre = d_u * (1.0 - ca["u"] ** 2)
        g["bcmp"] += d_upre.sum(axis=(0, 1))
        d_aa = d_upre * (1.0 + ca["bb"])
        d_bb = d_upre * (1.0 + ca["aa"])
        g["Wa"] += np.einsum("bpd,bpc->dc", ca["L"], d_aa)
        g["Wb"] += np.einsum("bpd,bpc->dc", ca["Rr"], d_bb)
        d_L = np.einsum("bpc,dc->bpd", d_aa, P["Wa"])             # (B,6,D)
        d_Rr = np.einsum("bpc,dc->bpd", d_bb, P["Wb"])
        for pi in range(6):
            dR[:, PAIR_L[pi], :] += d_L[:, pi, :]
            dR[:, PAIR_R[pi], :] += d_Rr[:, pi, :]

        # --- 1. radical extraction -------------------------------------------
        A, Hsv = ca["A"], ca["Hs"]
        dA = np.einsum("bjd,btd->bjt", dR, Hsv)
        dHs = np.einsum("bjt,bjd->btd", A, dR)
        d_scores = A * (dA - np.sum(dA * A, axis=-1, keepdims=True))
        d_scores = d_scores / np.sqrt(c["D_EMB"])
        # scores = Hs . Qeff  -> two parents
        d_Qeff = np.einsum("bjt,btd->bjd", d_scores, Hsv)
        dHs = dHs + np.einsum("bjt,bjd->btd", d_scores, ca["Qeff"])
        g["Q"] += d_Qeff.sum(0)
        g["Qtab"] += np.einsum("bp,bjd->pjd", ca["wz"], d_Qeff)
        d_wz = np.einsum("bjd,pjd->bp", d_Qeff, P["Qtab"])

        # --- 0. template recognition -----------------------------------------
        wz = ca["wz"]
        d_wz_log = wz * (d_wz - np.sum(d_wz * wz, axis=1, keepdims=True))
        d_wz_log = d_wz_log + G["d_wz"]
        g["Wwz"] += ca["bag"].T @ d_wz_log
        g["bwz"] += d_wz_log.sum(0)
        d_bag = d_wz_log @ P["Wwz"].T                       # (B,D)
        dHs = dHs + (d_bag / ca["nval"])[:, None, :] * ca["valid"][:, :, None]

        g["Pos"] += dHs.sum(0)
        np.add.at(g["Emb"], ca["X"].reshape(-1),
                  dHs.reshape(-1, c["D_EMB"]))
        return g

    # -------------------------------------------------------------------
    def step(self, grads, lr):
        c = self.cfg
        self.t += 1
        for k in self.keys:
            gk = grads[k]
            self.m[k] = c["BETA1"] * self.m[k] + (1 - c["BETA1"]) * gk
            self.v[k] = c["BETA2"] * self.v[k] + (1 - c["BETA2"]) * gk ** 2
            mh = self.m[k] / (1 - c["BETA1"] ** self.t)
            vh = self.v[k] / (1 - c["BETA2"] ** self.t)
            self.P[k] -= lr * mh / (np.sqrt(vh) + c["EPS"])

    def loss_only(self, b):
        ca = self.forward(b)
        tot, _, _ = self.loss(b, ca)
        return tot


# =============================================================================
# SECTION 4. THE CONSTRAINED DECODER  (qayd al-maqsura)
# =============================================================================

def maqsura_decode(model, ca, constrain=True):
    """
    Read a surface form back out of the hidden state. When `constrain` is on,
    the last non-padding position is hard-masked to the designated rhyme class,
    exactly as every line of a maqsura poem must land on the same restricted
    ending. Returns the emitted sequences.
    """
    dec = ca["dec"].copy()                                  # (B,T,V)
    B, T, _ = dec.shape
    if constrain:
        lens = np.argmax(np.concatenate(
            [(np.argmax(dec, axis=-1) == PAD_ID),
             np.ones((B, 1), dtype=bool)], axis=1), axis=1)
        lens = np.clip(lens, 1, T)
        mask = np.full(VOCAB, -1e9)
        mask[RHYME_CLASS] = 0.0
        for i in range(B):
            dec[i, lens[i] - 1] += mask
    return np.argmax(dec, axis=-1)


# =============================================================================
# SECTION 5. GRADIENT CHECK  (mandatory, runs before training)
# =============================================================================

def gradient_check(model, batch, n_probe=4, eps=1e-5, tol=2e-4):
    """
    Central-difference check on a random sample of coordinates in every
    parameter block. Every block must pass or the run aborts. Nothing about
    this architecture is worth reading if the derivatives are wrong.
    """
    ca = model.forward(batch)
    _, _, G = model.loss(batch, ca)
    ana = model.backward(batch, ca, G)

    print("  block         max-rel-err   sample  status")
    print("  " + "-" * 46)
    worst_all, ok_all = 0.0, True
    rng = np.random.default_rng(7)
    for k in model.keys:
        arr = model.P[k]
        flat = arr.reshape(-1)
        idxs = rng.choice(flat.size, size=min(n_probe, flat.size), replace=False)
        worst = 0.0
        for i in idxs:
            old = flat[i]
            flat[i] = old + eps
            lp = model.loss_only(batch)
            flat[i] = old - eps
            lm = model.loss_only(batch)
            flat[i] = old
            num = (lp - lm) / (2 * eps)
            an = ana[k].reshape(-1)[i]
            denom = max(abs(num), abs(an), 1e-7)
            worst = max(worst, abs(num - an) / denom)
        ok = worst < tol
        ok_all &= ok
        worst_all = max(worst_all, worst)
        print(f"  {k:<12}  {worst:>11.3e}   {len(idxs):>4}   "
              f"{'PASS' if ok else 'FAIL'}")
    print("  " + "-" * 46)
    print(f"  overall worst relative error: {worst_all:.3e}   "
          f"{'ALL BLOCKS PASS' if ok_all else 'FAILURE'}")
    return ok_all


# =============================================================================
# SECTION 6. EVALUATION
#   Five named tests, each of which measures one of Ibn Duraid's commitments.
# =============================================================================

def evaluate(model, corpus, rng, n=900):
    c = model.cfg
    out = {}

    # --- (a) derivation on ordinary circulating vocabulary -----------------
    dev = corpus.sample(n, split="dev")
    ca = model.forward(dev)
    nat = dev["y_aj"][:, 0] == 0
    po = np.argmax(ca["lo_orb"], 1)
    pp = np.argmax(ca["lo_pix"], 1)
    pt = np.argmax(ca["lo_pat"], 1)
    out["root_acc"] = float(np.mean((po[nat] == dev["y_orb"][nat]) &
                                    (pp[nat] == dev["y_pix"][nat])))
    out["orbit_acc"] = float(np.mean(po[nat] == dev["y_orb"][nat]))
    out["pattern_acc"] = float(np.mean(pt[nat] == dev["y_pat"][nat]))

    # --- (b) taqlib transfer: orderings never trained on -------------------
    tr = corpus.sample(360, split="orbit_transfer")
    cat = model.forward(tr)
    po2 = np.argmax(cat["lo_orb"], 1)
    pp2 = np.argmax(cat["lo_pix"], 1)
    out["orbit_transfer_root"] = float(np.mean((po2 == tr["y_orb"]) &
                                               (pp2 == tr["y_pix"])))
    out["orbit_transfer_orbit"] = float(np.mean(po2 == tr["y_orb"]))

    # --- (c) the Shu'ubiyya test: derivable vs genuinely foreign -----------
    fo = corpus.sample(400, split="foreign")
    caf = model.forward(fo)
    p_for = sigmoid(caf["lo_aj"])[:, 0]
    p_nat = sigmoid(ca["lo_aj"])[nat, 0]
    out["foreign_recall"] = float(np.mean(p_for > 0.5))
    out["native_kept"] = float(np.mean(p_nat <= 0.5))
    out["forced_derivation_rate"] = float(np.mean(p_for <= 0.5))

    # --- (d) al-Malahin: both meters in the green under compulsion ---------
    co = dev["coerce"][:, 0] == 1
    if co.sum() > 0:
        e_pub = np.linalg.norm(ca["out_p"][co] - dev["s_pub"][co], axis=1)
        e_in = np.linalg.norm(ca["out_i"][co] - dev["s_in"][co], axis=1)
        thr = 0.9
        out["dual_satisfaction"] = float(np.mean((e_pub < thr) & (e_in < thr)))
        out["coerced_n"] = int(co.sum())
        out["sense_err_public"] = float(np.mean(e_pub))
        out["sense_err_inner"] = float(np.mean(e_in))
    uc = (dev["coerce"][:, 0] == 0) & nat
    out["sense_err_uncoerced"] = float(np.mean(
        np.linalg.norm(ca["out_p"][uc] - dev["s_pub"][uc], axis=1)))

    # --- (e) the ledger stays open -----------------------------------------
    ent = entropy_of(ca["sel_p"])
    out["ledger_entropy"] = float(np.mean(ent))
    out["ledger_max_nats"] = float(np.log(c["LEDGER_K"]))
    out["collapsed_frac"] = float(np.mean(ent < 0.25))

    # --- (f) the gate actually thresholds ----------------------------------
    gate = ca["gate"][:, 0]
    out["gate_mean"] = float(np.mean(gate))
    hi = dev["counts"][:, 0] > np.median(dev["counts"][:, 0])
    out["gate_on_common"] = float(np.mean(gate[hi]))
    out["gate_on_rare"] = float(np.mean(gate[~hi]))

    # --- (g) the maqsura channel -------------------------------------------
    free = np.argmax(ca["dec"], axis=-1)
    bound = maqsura_decode(model, ca, constrain=True)
    valid = dev["X"] != PAD_ID
    out["recon_free"] = float(np.mean((free == dev["X"])[valid]))
    out["recon_bound"] = float(np.mean((bound == dev["X"])[valid]))
    ends = []
    for i in range(bound.shape[0]):
        seq = [t for t in bound[i] if t != PAD_ID]
        if seq:
            ends.append(seq[-1] in RHYME_CLASS)
    out["maqsura_satisfied"] = float(np.mean(ends)) if ends else 0.0
    return out


# =============================================================================
# SECTION 6b. ABLATIONS
#   Each named part should be load-bearing. If a part can be switched off at
#   evaluation time with no cost, it was decoration and should be deleted.
# =============================================================================

def ablations(model, corpus, rng, n=900):
    c = model.cfg
    dev = corpus.sample(n, split="dev")
    nat = dev["y_aj"][:, 0] == 0
    res = {}

    # --- baseline ----------------------------------------------------------
    ca = model.forward(dev)
    co = dev["coerce"][:, 0] == 1
    thr = 0.9

    def dual(cache):
        e_p = np.linalg.norm(cache["out_p"][co] - dev["s_pub"][co], axis=1)
        e_i = np.linalg.norm(cache["out_i"][co] - dev["s_in"][co], axis=1)
        return float(np.mean((e_p < thr) & (e_i < thr)))

    res["base_dual"] = dual(ca)
    res["base_root"] = float(np.mean(
        (np.argmax(ca["lo_orb"], 1)[nat] == dev["y_orb"][nat]) &
        (np.argmax(ca["lo_pix"], 1)[nat] == dev["y_pix"][nat])))

    # --- A. silence the intention gate -------------------------------------
    keep = c["NIYYA_GAIN"]
    c["NIYYA_GAIN"] = 0.0
    res["no_niyya_dual"] = dual(model.forward(dev))
    c["NIYYA_GAIN"] = keep

    # --- B. grant every form dense capacity (abolish the census) -----------
    fat = dict(dev)
    fat["counts"] = np.full_like(dev["counts"], 1e6)
    caf = model.forward(fat)
    rare = dev["counts"][:, 0] <= np.median(dev["counts"][:, 0])
    sel = nat & rare
    res["gate_rare_base"] = float(np.mean(
        (np.argmax(ca["lo_orb"], 1)[sel] == dev["y_orb"][sel]) &
        (np.argmax(ca["lo_pix"], 1)[sel] == dev["y_pix"][sel])))
    res["gate_rare_forced"] = float(np.mean(
        (np.argmax(caf["lo_orb"], 1)[sel] == dev["y_orb"][sel]) &
        (np.argmax(caf["lo_pix"], 1)[sel] == dev["y_pix"][sel])))

    # --- C. break the comparison ring (ordinal must die) -------------------
    keepW = model.P["Wcr"].copy()
    model.P["Wcr"] = np.zeros_like(keepW)
    car = model.forward(dev)
    res["no_ring_ordinal"] = float(np.mean(
        np.argmax(car["lo_pix"], 1)[nat] == dev["y_pix"][nat]))
    res["no_ring_orbit"] = float(np.mean(
        np.argmax(car["lo_orb"], 1)[nat] == dev["y_orb"][nat]))
    model.P["Wcr"] = keepW

    print()
    print("-" * 72)
    print(" ABLATIONS")
    print("-" * 72)
    print(" A  silence the intention gate (niyya)")
    print(f"      dual satisfaction  {res['base_dual']:.3f}  ->  "
          f"{res['no_niyya_dual']:.3f}")
    print(" B  abolish the census, grant every form dense capacity")
    print(f"      root accuracy on rare forms  {res['gate_rare_base']:.3f}  ->  "
          f"{res['gate_rare_forced']:.3f}")
    print(" C  break the comparison ring")
    print(f"      ordinal accuracy   {model_ord(ca, dev, nat):.3f}  ->  "
          f"{res['no_ring_ordinal']:.3f}")
    print(f"      orbit accuracy     {float(np.mean(np.argmax(ca['lo_orb'],1)[nat]==dev['y_orb'][nat])):.3f}  ->  "
          f"{res['no_ring_orbit']:.3f}   (should barely move)")
    return res


def model_ord(ca, dev, nat):
    return float(np.mean(np.argmax(ca["lo_pix"], 1)[nat] == dev["y_pix"][nat]))


# =============================================================================
# SECTION 7. TRAINING
# =============================================================================

def train(quick=False):
    c = dict(CFG)
    if quick:
        c["EPOCHS"], c["BATCH"] = 40, 48
    rng = np.random.default_rng(c["SEED"])
    corpus = BasraCorpus(c, rng)
    model = JamharaEngine(c, rng)

    n_par = sum(v.size for v in model.P.values())
    print("=" * 72)
    print(" THE JAMHARA ENGINE  --  Ibn Duraid al-Azdi (837-933)")
    print("=" * 72)
    print(f" radical letters      : {c['N_RADICAL']}")
    print(f" radical multisets    : {c['N_ORBIT']}  "
          f"(x6 orderings = {c['N_ORBIT']*6} candidate roots)")
    print(f" live orderings       : {int(corpus.used.sum())} of "
          f"{corpus.used.size}  (the rest neglected, muhmal)")
    print(f" orderings held out   : {len(corpus.heldout)}  "
          f"(never trained on; used only for the taqlib transfer test)")
    print(f" templates (awzan)    : {c['N_PATTERN']}")
    print(f" ledger width K       : {c['LEDGER_K']}")
    print(f" parameters           : {n_par:,}")
    print()

    print("-" * 72)
    print(" FINITE-DIFFERENCE GRADIENT CHECK")
    print("-" * 72)
    gb = corpus.sample(24, split="train")
    if not gradient_check(model, gb):
        print("\n ABORTING: analytic gradients disagree with finite differences.")
        sys.exit(1)
    print()

    print("-" * 72)
    print(" TRAINING")
    print("-" * 72)
    print(f" {'epoch':>6} {'total':>8} {'wazn':>7} {'orbit':>7} {'pidx':>7} "
          f"{'ajnabi':>7} {'sns-pub':>8} {'sns-inr':>8} {'floor':>7} {'gate':>6}")
    t0 = time.time()
    hist = []
    for ep in range(1, c["EPOCHS"] + 1):
        b = corpus.sample(c["BATCH"], split="train")
        ca = model.forward(b)
        tot, parts, G = model.loss(b, ca)
        grads = model.backward(b, ca, G)
        lr = c["LR"] * (0.5 * (1 + np.cos(np.pi * ep / c["EPOCHS"])) * 0.9 + 0.1)
        model.step(grads, lr)
        hist.append(parts)
        if ep == 1 or ep % max(1, c["EPOCHS"] // 13) == 0 or ep == c["EPOCHS"]:
            print(f" {ep:>6} {parts['total']:>8.4f} {parts['wazn']:>7.4f} "
                  f"{parts['orbit']:>7.4f} "
                  f"{parts['permidx']:>7.4f} {parts['ajnabi']:>7.4f} "
                  f"{parts['sense_pub']:>8.4f} {parts['sense_inner']:>8.4f} "
                  f"{parts['floor']:>7.4f} "
                  f"{float(np.mean(ca['gate'])):>6.3f}")
    dt = time.time() - t0
    print(f"\n trained {c['EPOCHS']} steps in {dt:.1f}s  "
          f"({1000*dt/c['EPOCHS']:.1f} ms/step)")
    print(f" loss: {hist[0]['total']:.4f} -> {hist[-1]['total']:.4f}  "
          f"({100*(1-hist[-1]['total']/hist[0]['total']):.1f}% reduction)")
    return model, corpus, rng, hist


def report(model, corpus, rng):
    m = evaluate(model, corpus, rng)
    print()
    print("-" * 72)
    print(" EVALUATION")
    print("-" * 72)
    print(" 1  ISTIKHRAJ AL-JIDHR + MIZAN AL-ISHTIQAQ  (derivation)")
    print(f"      exact root  (orbit AND ordering)    : {m['root_acc']:.3f}")
    print(f"      orbit only  (radical multiset)      : {m['orbit_acc']:.3f}")
    print(f"      template (wazn)                     : {m['pattern_acc']:.3f}")
    print()
    print(" 2  HALQAT AL-TAQLIB  (zero-shot transfer to unseen orderings)")
    print(f"      exact root on held-out taqalib      : "
          f"{m['orbit_transfer_root']:.3f}")
    print(f"      orbit recovered on held-out         : "
          f"{m['orbit_transfer_orbit']:.3f}")
    print(f"      chance for exact root               : "
          f"{1.0/(model.cfg['N_ORBIT']*6):.3f}")
    print()
    print(" 3  THE SHU'UBIYYA TEST  (refuse 'meaningless', refuse forgery)")
    print(f"      foreign forms correctly flagged     : "
          f"{m['foreign_recall']:.3f}")
    print(f"      native forms NOT wrongly exiled     : {m['native_kept']:.3f}")
    print(f"      forced derivations (lower = better) : "
          f"{m['forced_derivation_rate']:.3f}")
    print()
    print(" 4  AL-MALAHIN  (both meters green under compulsion)")
    print(f"      coerced cases in sample             : {m.get('coerced_n',0)}")
    print(f"      dual satisfaction rate              : "
          f"{m.get('dual_satisfaction', float('nan')):.3f}")
    print(f"      mean error, hearer's reading        : "
          f"{m.get('sense_err_public', float('nan')):.3f}")
    print(f"      mean error, speaker's reading       : "
          f"{m.get('sense_err_inner', float('nan')):.3f}")
    print(f"      mean error, no compulsion           : "
          f"{m['sense_err_uncoerced']:.3f}")
    print()
    print(" 5  THE LEDGER STAYS OPEN  (anti-collapse)")
    print(f"      mean selector entropy               : "
          f"{m['ledger_entropy']:.3f} nats of "
          f"{m['ledger_max_nats']:.3f} possible")
    print(f"      fraction collapsed to one reading   : "
          f"{m['collapsed_frac']:.3f}")
    print()
    print(" 6  BAWWABAT AL-JAMHARA  (the circulation threshold bites)")
    print(f"      mean gate                           : {m['gate_mean']:.3f} "
          f"(budget {model.cfg['JAMHARA_BUDGET']:.2f})")
    print(f"      gate on circulating vocabulary      : "
          f"{m['gate_on_common']:.3f}")
    print(f"      gate on rare vocabulary             : {m['gate_on_rare']:.3f}")
    print()
    print(" 7  QAYD AL-MAQSURA  (meaning through a narrowed channel)")
    print(f"      surface reconstruction, free        : {m['recon_free']:.3f}")
    print(f"      surface reconstruction, constrained : {m['recon_bound']:.3f}")
    print(f"      rhyme constraint satisfied          : "
          f"{m['maqsura_satisfied']:.3f}")
    return m


# =============================================================================
# SECTION 8. SELF-TESTS
#   Structural assertions about the architecture, independent of training.
# =============================================================================

def self_tests(model, corpus, rng):
    print()
    print("-" * 72)
    print(" SELF-TESTS")
    print("-" * 72)
    passed = 0
    total = 0

    def check(name, cond, note=""):
        nonlocal passed, total
        total += 1
        if cond:
            passed += 1
        print(f"  [{'ok ' if cond else 'FAIL'}] {name}"
              + (f"   {note}" if note else ""))

    c = model.cfg
    b = corpus.sample(48, split="train")
    ca = model.forward(b)

    # T1: the invariant branch really is permutation invariant.
    R = ca["R"]
    m1 = np.einsum("bjd,dh->bh", R, model.P["Wm"])
    m2 = np.einsum("bjd,dh->bh", R[:, [2, 0, 1], :], model.P["Wm"])
    check("taqlib: invariant branch is order-blind",
          np.allclose(m1, m2, atol=1e-12),
          f"max diff {np.max(np.abs(m1-m2)):.2e}")

    # T2: the equivariant branch is NOT.
    o1 = np.einsum("bjd,jdh->bh", R, model.P["Wo"])
    o2 = np.einsum("bjd,jdh->bh", R[:, [2, 0, 1], :], model.P["Wo"])
    check("taqlib: equivariant branch does carry the ordering",
          not np.allclose(o1, o2, atol=1e-6),
          f"max diff {np.max(np.abs(o1-o2)):.2e}")

    # T3: the gate is monotone in circulation.
    lo = dict(b); lo["counts"] = np.full_like(b["counts"], 5.0)
    hi = dict(b); hi["counts"] = np.full_like(b["counts"], 9000.0)
    g_lo = float(np.mean(model.forward(lo)["gate"]))
    g_hi = float(np.mean(model.forward(hi)["gate"]))
    check("jamhara: rarer forms get less dense capacity",
          g_hi > g_lo, f"rare {g_lo:.3f} < common {g_hi:.3f}")

    # T4: the ledger holds exactly K distinct hypotheses.
    k_pairs = [len(set(map(tuple, np.stack([ca["k_orb"][i],
                                            ca["k_pix"][i]], 1).tolist())))
               for i in range(ca["B"])]
    check("ledger: K distinct derivations retained",
          all(k == c["LEDGER_K"] for k in k_pairs),
          f"K={c['LEDGER_K']}")

    # T5: the ledger scores are sorted and are the true top-K.
    grid = (ca["lo_orb"][:, :, None] + ca["lo_pix"][:, None, :]).reshape(ca["B"], -1)
    true_top = np.sort(np.sort(grid, axis=1)[:, -c["LEDGER_K"]:], axis=1)[:, ::-1]
    check("ledger: retained set is the genuine top-K",
          np.allclose(np.sort(ca["klog"], axis=1)[:, ::-1], true_top, atol=1e-10))

    # T6: intention actually moves the reading.
    b2 = dict(b)
    b2["ctx_i"] = rng.normal(0, 1, b["ctx_i"].shape)
    ca2 = model.forward(b2)
    check("niyya: a different inner context yields a different reading",
          not np.allclose(ca2["out_i"], ca["out_p"], atol=1e-6),
          f"mean shift {np.mean(np.abs(ca2['out_i']-ca['out_p'])):.3f}")

    # T7: the surface is identical whatever the intention. This is the whole
    # point of al-Malahin: the hearer cannot tell.
    check("niyya: the uttered surface does not leak the intention",
          np.array_equal(ca2["X"], ca["X"]) and
          np.allclose(ca2["dec"], ca["dec"], atol=1e-12))

    # T8: the maqsura constraint is satisfied by construction.
    bound = maqsura_decode(model, ca, constrain=True)
    ends_ok = []
    for i in range(bound.shape[0]):
        seq = [t for t in bound[i] if t != PAD_ID]
        if seq:
            ends_ok.append(seq[-1] in RHYME_CLASS)
    check("maqsura: every constrained emission lands on the rhyme",
          all(ends_ok) if ends_ok else False,
          f"{sum(ends_ok)}/{len(ends_ok)}")

    # T9: the corpus really withholds the transfer set.
    tr = corpus.sample(120, split="orbit_transfer")
    pairs = set(zip(tr["y_orb"].tolist(), tr["y_pix"].tolist()))
    check("corpus: transfer split draws only from held-out taqalib",
          pairs.issubset(corpus.heldout), f"{len(pairs)} distinct pairs")

    trn = corpus.sample(400, split="train")
    native = trn["y_aj"][:, 0] == 0
    trn_pairs = set(zip(trn["y_orb"][native].tolist(),
                        trn["y_pix"][native].tolist()))
    check("corpus: training split never touches them",
          len(trn_pairs & corpus.heldout) == 0)

    # T10: loss is finite and gradients are finite.
    tot, parts, G = model.loss(b, ca)
    gr = model.backward(b, ca, G)
    check("numerics: loss and all gradients finite",
          np.isfinite(tot) and all(np.all(np.isfinite(v)) for v in gr.values()))

    print(f"\n  {passed}/{total} self-tests passed")
    return passed == total


# =============================================================================
# SECTION 9. A WORKED READING
#   One form, shown the way Ibn Duraid shows a name in the Ishtiqaq: the
#   several derivations that could account for it, left standing together.
# =============================================================================

def worked_reading(model, corpus, rng):
    print()
    print("-" * 72)
    print(" A WORKED READING  (the ledger, printed the way he prints a name)")
    print("-" * 72)
    b = corpus.sample(6, split="dev")
    ca = model.forward(b)
    names = ["alif", "ba", "ta", "tha", "jim", "ha", "kha", "dal",
             "dhal", "ra", "zay", "sin", "-A-", "-I-", "-U-", "-t-", "-n-"]

    def show(seq):
        return " ".join(names[t] if t < len(names) else "." for t in seq
                        if t != PAD_ID)

    for i in range(3):
        foreign = b["y_aj"][i, 0] > 0.5
        p_for = float(sigmoid(ca["lo_aj"][i, 0]))
        print(f"\n  form: {show(b['X'][i])}")
        print(f"    circulation tally : {int(b['counts'][i,0])}"
              f"   capacity granted: {float(ca['gate'][i,0]):.3f}")
        print(f"    verdict on origin : "
              f"{'ajnabi, no derivation inside the lexicon' if p_for>0.5 else 'derivable'}"
              f"   (p={p_for:.2f}, truth={'foreign' if foreign else 'native'})")
        if not foreign:
            print(f"    true root         : orbit {b['y_orb'][i]}, "
                  f"ordering {b['y_pix'][i]}, template {b['y_pat'][i]}")
        print("    derivations left standing:")
        for k in range(model.cfg["LEDGER_K"]):
            o, p = int(ca["k_orb"][i, k]), int(ca["k_pix"][i, k])
            trip = corpus.orbits[o]
            live = corpus.used[o, p]
            mark = "musta'mal" if live else "muhmal   "
            print(f"      {k+1}. orbit {o:>2} {str(trip):<12} ordering {p}  "
                  f"{mark}  weight {float(ca['sel_p'][i,k]):.3f}")
        print(f"    selector entropy  : "
              f"{float(entropy_of(ca['sel_p'][i])):.3f} nats "
              f"(floor {model.cfg['ENTROPY_FLOOR']:.2f})")


# =============================================================================
# MAIN
# =============================================================================

def main():
    quick = "--quick" in sys.argv
    np.set_printoptions(precision=4, suppress=True)
    model, corpus, rng, hist = train(quick=quick)
    metrics = report(model, corpus, rng)
    ablations(model, corpus, rng)
    ok = self_tests(model, corpus, rng)
    worked_reading(model, corpus, rng)
    print()
    print("=" * 72)
    print(" DONE." if ok else " DONE WITH FAILING SELF-TESTS.")
    print("=" * 72)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
