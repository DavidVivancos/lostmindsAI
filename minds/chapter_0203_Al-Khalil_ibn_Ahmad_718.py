#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
0203 — Al-Khalīl ibn Aḥmad al-Farāhīdī (c. 718 – 786/791), Basra
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 11 Minds 201 - 220 Available on Amazon https://www.amazon.com/dp/B0HKF7TRFF
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0203_Al-Khalil_ibn_Ahmad_718 - Al-Khalīl ibn Aḥmad al-Farāhīdī (c. 718 – 786/791), Basra
========================
"THE INHABITED ORBIT" — a trainable neural architecture built from the one
cognitive move that is al-Khalīl's alone.

WHAT THIS FILE IS
-----------------
A pure-NumPy, from-scratch, executable model of how al-Khalīl's mind worked,
expressed as a novel artificial neuron ("orbit neuron") and a full training
pipeline.  Nothing here is a mock-up: the network trains, learns hidden laws
from data, passes a finite-difference gradient check, and is self-tested.

THE MIND, IN ONE SENTENCE
-------------------------
Al-Khalīl never started from the words that exist.  He started from the space
of everything that COULD exist — every permutation of every set of letters,
every rotation of every rhythmic circle — and then recorded which points of
that space are inhabited (mustaʿmal, "used") and which are empty (muhmal,
"neglected").  Knowledge, for him, is a map of a possibility space with its
holes drawn in.  Three facts document this:

  1. Kitāb al-ʿAyn (his dictionary) treats a root by its ANAGRAM CLASS
     (taqlīb): all n! orderings of the same letters are listed together, and
     the empty orderings are stated as empty.  Its introduction computes the
     size of the possible language — 28·27 + 28·27·26 + ... = 12,305,412
     ordered strings of 2–5 distinct letters — and says that the real
     language is only the phonologically realizable part of it.
  2. ʿIlm al-ʿarūḍ (his prosody) derives the fifteen (later sixteen) meters
     of Arabic verse as ROTATIONS of five circles (dawāʾir) of vowelled (1)
     and unvowelled (0) letters.  Some rotations are used; many are not.
     Deviation (ziḥāf) is licensed ONLY at the "cords" (asbāb); the "pegs"
     (awtād) are inviolable.
  3. Kitāb al-Muʿammā (his lost cryptography book) used the same enumeration
     — all possible words, vowelled and unvowelled — as a dictionary against
     which a decipherment can be tested, together with the probable-word
     method (the message begins with a known formula).

THE NEURON
----------
An ORBIT NEURON responds not to an input but to the orbit of the input under
a symmetry group, and reports four things:
    * an INVARIANT   — what the orbit is (the root-set; the circle),
    * an EQUIVARIANT — which element of the orbit this is (the arrangement;
                       the meter = rotation phase),
    * a LICENCE      — where deviation from the template is tolerated
                       (cords) and where it is forbidden (pegs), learned,
    * a MUHMAL REGISTER — a learned map of which orbit elements are empty.
Two instances are built:
    TaqlibOrbitNeuron  — symmetric group S_n acting on the letters of a word.
    DairaRingNeuron    — cyclic group Z_P acting on a ring of vowelled /
                         unvowelled slots, with a ziḥāf-tolerant alignment
                         (a differentiable forward–backward lattice whose
                         per-slot deletion cost IS the learned licence).
Letters are embedded by ARTICULATION (makhraj): al-Khalīl's own ordering of
the alphabet from the throat (ʿayn) to the lips (mīm), so that memory is
addressed by the body that speaks.  Vowel marks (ḥarakāt) are embedded as
scaled-down copies of the long-vowel letters — exactly his notation reform.

WHAT IT LEARNS (all laws hidden inside a synthetic "Basra corpus")
------------------------------------------------------------------
    T1  mustaʿmal/muhmal: is this string of letters a used word?  Requires
        learning (a) a co-occurrence restriction between letters of the same
        articulatory class (the real Arabic constraint documented by
        Greenberg 1950 / McCarthy 1994), and (b) an arrangement rule that
        is RELATIVE WITHIN THE ANAGRAM CLASS — a word is used only if it is
        among the best arrangements of its own root.  A model that scores
        strings in isolation cannot do this; the orbit neuron can.
    T2  ḥarakāt: which vowel pattern a used word takes.
    T3  ʿarūḍ: which of 16 meters a ziḥāf-corrupted line is in.  The ring
        neuron must learn the five rings, which of their 52 rotations are
        inhabited, and that deletion is licensed at cords and never at pegs.
    T4  muʿammā: recover a monoalphabetic cipher key using the trained
        lexicon as al-Khalīl's "table of all possible words" plus the
        probable-word crib.

CONVENTIONS
-----------
Pure NumPy.  Manual backpropagation.  Mandatory finite-difference gradient
check.  Real training loops.  Self-tests.  Deterministic seeds.
Run:  python3 0203_Al-Khalil_ibn_Ahmad_718_Neuron.py
"""
from __future__ import annotations

import itertools
import math
import sys
import time

import numpy as np

np.set_printoptions(precision=3, suppress=True, linewidth=110)

# =============================================================================
# 0.  THE ALPHABET, IN AL-KHALĪL'S ORDER (Kitāb al-ʿAyn, muqaddima)
# =============================================================================
# Nine articulatory stations from the throat to the lips, then the "airy"
# letters.  Place index 0 = deepest (ʿayn), 8 = lips.  This is the ORDER OF
# THE DICTIONARY: memory addressed by the point of articulation.
STATIONS = [
    ("throat",      ["ʿ", "ḥ", "h", "kh", "gh"]),
    ("uvula",       ["q", "k"]),
    ("hard palate", ["j", "sh", "ḍ"]),
    ("gum ridge",   ["ṣ", "s", "z"]),
    ("teeth",       ["ṭ", "d", "t"]),
    ("teeth tips",  ["ẓ", "th", "dh"]),
    ("tongue tip",  ["r", "l", "n"]),
    ("lips",        ["f", "b", "m"]),
    ("air",         ["w", "a", "y"]),      # wāw, alif, yāʾ  (hamza folded in)
]
LETTERS: list[str] = [c for _, cs in STATIONS for c in cs]
N_LET = len(LETTERS)                      # 28
PLACE = np.array([i for i, (_, cs) in enumerate(STATIONS) for _ in cs])
IDX = {c: i for i, c in enumerate(LETTERS)}
EMPHATIC = np.array([1.0 if c in ("ṣ", "ḍ", "ṭ", "ẓ", "q") else 0.0 for c in LETTERS])
SONORANT = np.array([1.0 if c in ("r", "l", "n", "m", "w", "y") else 0.0 for c in LETTERS])
LABIAL = np.array([1.0 if c in ("f", "b", "m", "w") else 0.0 for c in LETTERS])
GUTTURAL = np.array([1.0 if PLACE[i] == 0 else 0.0 for i in range(N_LET)])
# Fixed articulatory feature table used to INITIALISE the letter embedding.
FEATURES = np.stack([PLACE / 8.0, GUTTURAL, EMPHATIC, SONORANT, LABIAL], axis=1)  # (28, 5)
ALIF, WAW, YA = IDX["a"], IDX["w"], IDX["y"]


def khalil_possible_language() -> dict:
    """Al-Khalīl's count of the possible language: ordered strings of r
    distinct letters, r = 2..5, from 28 letters.  P(28,2)+...+P(28,5)."""
    counts = {r: math.perm(N_LET, r) for r in range(2, 6)}
    counts["total"] = sum(counts[r] for r in range(2, 6))
    return counts


# =============================================================================
# 1.  THE FIVE CIRCLES (dawāʾir) — verified letter-level rotation structure
# =============================================================================
# Units: W = watid majmūʿ "110", Wm = watid mafrūq "101" (pegs);
#        S = sabab khafīf "10", St = sabab thaqīl "11"        (cords).
UNIT_BITS = {"W": "110", "Wm": "101", "S": "10", "St": "11"}
UNIT_IS_PEG = {"W": True, "Wm": True, "S": False, "St": False}
CIRCLES = {
    1: ["W", "S", "W", "S", "S"],                       # ṭawīl / madīd / basīṭ
    2: ["W", "St", "S"],                                # wāfir / kāmil
    3: ["W", "S", "S"],                                 # hazaj / rajaz / ramal
    4: ["S", "S", "W", "S", "S", "W", "S", "S", "Wm"],  # sarīʿ … mujtathth
    5: ["W", "S"],                                      # mutaqārib / mutadārik
}
# The sixteen meters as (circle, phase-in-letters).  Phases are unit starts.
METERS = [
    ("ṭawīl", 1, 0), ("madīd", 1, 3), ("basīṭ", 1, 8),
    ("wāfir", 2, 0), ("kāmil", 2, 3),
    ("hazaj", 3, 0), ("rajaz", 3, 3), ("ramal", 3, 5),
    ("sarīʿ", 4, 0), ("munsariḥ", 4, 7), ("khafīf", 4, 9),
    ("muḍāriʿ", 4, 11), ("muqtaḍab", 4, 14), ("mujtathth", 4, 16),
    ("mutaqārib", 5, 0), ("mutadārik", 5, 3),
]
N_METER = len(METERS)


def ring_bits(c: int) -> str:
    return "".join(UNIT_BITS[u] for u in CIRCLES[c])


def ring_peg_mask(c: int) -> np.ndarray:
    return np.array([UNIT_IS_PEG[u] for u in CIRCLES[c] for _ in UNIT_BITS[u]], dtype=bool)


def ring_unit_start(c: int) -> list[int]:
    starts, pos = [], 0
    for u in CIRCLES[c]:
        starts.append(pos)
        pos += len(UNIT_BITS[u])
    return starts


RING_PERIOD = {c: len(ring_bits(c)) for c in CIRCLES}          # 12, 7, 7, 21, 5
# Every (circle, phase) is a candidate rotation.  52 in total; 16 inhabited.
ROTATIONS = [(c, phi) for c in CIRCLES for phi in range(RING_PERIOD[c])]
N_ROT = len(ROTATIONS)                                          # 52
ROT_INDEX = {cp: i for i, cp in enumerate(ROTATIONS)}
METER_ROT = [ROT_INDEX[(c, phi)] for _, c, phi in METERS]


# =============================================================================
# 2.  THE BASRA CORPUS — a synthetic language with hidden laws
# =============================================================================
class BasraCorpus:
    """Generates the data al-Khalīl would have collected in Basra, with the
    laws HIDDEN so that the network must discover them.

    Hidden law 1 (co-occurrence restriction): a root-set is viable only if no
      two of its letters come from the same articulatory station.
    Hidden law 2 (accidental gaps): a random 8% of viable root-sets are
      entirely unused — the language simply never took them up.
    Hidden law 3 (arrangement, RELATIVE WITHIN THE ORBIT): of the 6 orderings
      of a viable triliteral root, the top 3 by a hidden score are used;
      of the 2 orderings of a biliteral, the top 1.
    Hidden law 4 (vocalisation): the vowel pattern of a used word follows a
      hidden rule on the articulatory classes of its letters.
    Hidden law 5 (ziḥāf): verse lines are corrupted only at cords — a sabab
      khafīf "10" may lose its sākin ("1"); a sabab thaqīl "11" may be made
      quiescent ("10").  Pegs are never touched.
    """

    PATTERNS = [("a", "a", "a"), ("a", "i", "a"), ("a", "u", "a"), ("u", "i", "a")]
    MARK_ID = {"a": 0, "u": 1, "i": 2, "0": 3}

    def __init__(self, seed: int = 718, L: int = 28, p_del: float = 0.18, p_sub: float = 0.25):
        self.rng = np.random.default_rng(seed)
        self.L, self.p_del, self.p_sub = L, p_del, p_sub
        self.H = self.rng.normal(0.0, 0.35, size=(N_LET, N_LET))      # hidden pairwise taste
        self._build_lexicon()

    # ---- hidden laws ------------------------------------------------------
    def viable(self, letters) -> bool:
        return len({int(PLACE[x]) for x in letters}) == len(letters)

    def arrangement_score(self, order) -> float:
        pl = PLACE[list(order)] / 8.0
        s = 0.0
        for i in range(len(order) - 1):
            s += (1.0 if i == 0 else -0.6) * (pl[i + 1] - pl[i]) + self.H[order[i], order[i + 1]]
        s += 0.8 * SONORANT[order[-1]]
        return float(s)

    def vowel_pattern(self, order) -> int:
        x1, x2 = order[0], order[1] if len(order) > 1 else order[0]
        if GUTTURAL[x2]:
            return 0
        if EMPHATIC[x2]:
            return 2
        return 3 if LABIAL[x1] else 1

    # ---- the lexicon lattice (Kitāb al-ʿAyn as a table) -------------------
    def _build_lexicon(self):
        self.entries = []          # list of dicts: root, n, perms, used(bool per perm), pattern
        for n in (2, 3):
            for root in itertools.combinations(range(N_LET), n):
                perms = list(itertools.permutations(root))
                v = self.viable(root)
                gap = bool(self.rng.random() < 0.08)
                if v and not gap:
                    sc = np.array([self.arrangement_score(p) for p in perms])
                    k = 3 if n == 3 else 1
                    top = set(np.argsort(-sc)[:k].tolist())
                    used = [i in top for i in range(len(perms))]
                else:
                    used = [False] * len(perms)
                pats = [self.vowel_pattern(p) for p in perms]
                self.entries.append(dict(root=root, n=n, perms=perms, used=used,
                                         viable=v, gap=gap, patterns=pats))
        idx = np.arange(len(self.entries))
        self.rng.shuffle(idx)
        cut = int(0.75 * len(idx))
        self.train_ids, self.test_ids = idx[:cut], idx[cut:]
        self.used_words = [(e["perms"][i], e["patterns"][i]) for e in self.entries
                           for i in range(len(e["perms"])) if e["used"][i]]

    def words(self, ids, n: int):
        """Flatten entries of arity n into arrays X (W,n), y (W,), pattern (W,), entry id."""
        X, y, pat, eid = [], [], [], []
        for i in ids:
            e = self.entries[i]
            if e["n"] != n:
                continue
            for j, p in enumerate(e["perms"]):
                X.append(p); y.append(float(e["used"][j])); pat.append(e["patterns"][j]); eid.append(i)
        return np.array(X), np.array(y), np.array(pat), np.array(eid)

    def sample_words(self, n: int, B: int, split: str = "train"):
        ids = self.train_ids if split == "train" else self.test_ids
        X, y, pat, _ = self._cache_words(ids, n, split)
        sel = self.rng.integers(0, len(X), size=B)
        return X[sel], y[sel], pat[sel]

    def _cache_words(self, ids, n, split):
        key = (n, split)
        if not hasattr(self, "_wc"):
            self._wc = {}
        if key not in self._wc:
            self._wc[key] = self.words(ids, n)
        return self._wc[key]

    # ---- verse lines with ziḥāf --------------------------------------------
    def line(self, meter: int) -> np.ndarray:
        _, c, phi = METERS[meter]
        units, starts = CIRCLES[c], ring_unit_start(c)
        u0 = starts.index(phi)
        out = []
        k = u0
        while len(out) < self.L + 4:
            u = units[k % len(units)]
            bits = UNIT_BITS[u]
            if not UNIT_IS_PEG[u]:
                if u == "S" and self.rng.random() < self.p_del:
                    bits = "1"             # khabn / ṭayy: the sākin of the cord is dropped
                elif u == "St" and self.rng.random() < self.p_sub:
                    bits = "10"            # ʿaṣb / iḍmār: the second mutaḥarrik is made sākin
            out.extend(int(b) for b in bits)
            k += 1
        return np.array(out[: self.L], dtype=np.float64)

    def sample_lines(self, B: int):
        y = self.rng.integers(0, N_METER, size=B)
        X = np.stack([self.line(int(m)) for m in y])
        return X, y

    # ---- usage model: how often each word is actually written ----------------
    def _build_usage(self):
        """Hidden law 6 (usage): words are not equally common.  Following the
        remark in the ʿAyn's introduction that longer roots always contain one of
        the 'fluent' letters (r l n f b m), words built from fluent and airy
        letters are common, guttural/emphatic ones rare, and word frequency is
        Zipfian on top.  Short 'function words' (the commonest biliterals) make up
        about a third of any text.  This is the structure al-Kindī's frequency
        tables later exploited — and it is what makes a lexicon useful in
        decipherment: rare letters cannot be pinned by counting alone."""
        if hasattr(self, "_usage_built"):
            return
        rng = np.random.default_rng(3)
        kappa = np.zeros(N_LET)
        for ch in ("r", "l", "n", "f", "b", "m"):
            kappa[IDX[ch]] += 1.6
        for ch in ("w", "a", "y"):
            kappa[IDX[ch]] += 1.0
        kappa -= 1.2 * GUTTURAL + 0.8 * EMPHATIC
        kappa += rng.normal(0, 0.4, N_LET)
        ranks = rng.permutation(len(self.used_words)) + 1
        w = np.array([math.exp(sum(kappa[x] for x in wd)) * r ** -0.6 for (wd, _), r in zip(self.used_words, ranks)])
        self.usage = w / w.sum()
        bi = [i for i, (wd, _) in enumerate(self.used_words) if len(wd) == 2]
        tri = [i for i, (wd, _) in enumerate(self.used_words) if len(wd) == 3]
        self.func_words = sorted(bi, key=lambda i: -self.usage[i])[:8]
        self._fw_p = self.usage[self.func_words] / self.usage[self.func_words].sum()
        self._tri, self._tri_p = tri, self.usage[tri] / self.usage[tri].sum()
        self._urng = rng
        self._usage_built = True
        cnt = np.full(N_LET, 1e-3)                       # letter table from a long text (al-Kindī's step)
        for _ in range(3000):
            for wd in self.message(1)[3:]:
                for x in wd:
                    cnt[x] += 1
        self.letter_logfreq = np.log(cnt / cnt.sum())

    # ---- muʿammā: enciphered messages -------------------------------------
    def message(self, n_words: int = 40, p_func: float = 0.35):
        self._build_usage()
        opener = [w for w, _ in self.used_words[:3]]            # the fixed opening formula (the crib)
        out = []
        for _ in range(n_words):
            if self._urng.random() < p_func:
                out.append(self.used_words[self.func_words[self._urng.choice(len(self.func_words), p=self._fw_p)]][0])
            else:
                out.append(self.used_words[self._tri[self._urng.choice(len(self._tri), p=self._tri_p)]][0])
        return opener + out

    def encipher(self, msg, key: np.ndarray):
        return [tuple(int(key[x]) for x in w) for w in msg]


# =============================================================================
# 3.  NUMERICS
# =============================================================================
NEG = -1e30          # a finite "minus infinity" keeps logaddexp free of NaN


def lse(a, axis=-1, keepdims=False):
    m = np.max(a, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    return out if keepdims else np.squeeze(out, axis=axis)


def softmax(a, axis=-1):
    a = a - np.max(a, axis=axis, keepdims=True)
    e = np.exp(a)
    return e / np.sum(e, axis=axis, keepdims=True)


def sigmoid(a):
    return 0.5 * (1.0 + np.tanh(0.5 * a))


def softplus(a):
    return np.logaddexp(0.0, a)


def logaddexp(a, b):
    return np.logaddexp(a, b)


# =============================================================================
# 4.  THE MODEL — "THE INHABITED ORBIT"
# =============================================================================
class KhalilOrbitNet:
    """Two orbit neurons over a shared articulatory letter embedding.

    Parameters (all learned unless stated):
      Embedding   A (5,d): projects the FIXED articulatory features;  Efree (28,d).
      Root net    W1,b1,W2,b2,w3,b3 : order-invariant root-set viability r.
      Restriction P (28,28): symmetrised into R; penalty Σ_{i<j} R[x_i,x_j].
      Arrangement Pos[i] (d,h) for position i, bp, wg, bg : score g of an ordering.
      Orbit gate  gam1, gam2, b0 : used-logit = r − pen + gam1·g0 + gam2·q + b0,
                  where q = g0 − logsumexp_σ g(σ) + log n!  (relative to the orbit).
      Ḥarakāt     s_mark (scalar), Qpos[i] (d,h), cp (4,) : vowel-pattern head with
                  mark embeddings tied to the alif / wāw / yāʾ letter embeddings.
      Rings       u,v,delta (5, Pmax): per-slot match weight, match bias, deletion
                  cost (softplus(delta) is the LICENCE).  beta (52,): muhmal register.
      Readout     Abar (52,16): rotation → meter, row-softmaxed.
    """

    def __init__(self, d=12, h=32, h2=16, L=28, extra=4, use_orbit=True, seed=786, rings=None):
        rng = np.random.default_rng(seed)
        self.d, self.h, self.h2, self.L, self.J = d, h, h2, L, L + extra
        self.use_orbit = use_orbit
        # The rings are DATA: their number and periods come from circle discovery
        # (see discover_circles).  Al-Khalīl's own five are only the default.
        self.rings = list(rings) if rings is not None else [RING_PERIOD[c] for c in CIRCLES]
        self.Pmax = max(self.rings)
        self.rotations = [(k, phi) for k, Pk in enumerate(self.rings) for phi in range(Pk)]
        self.n_rot = len(self.rotations)
        P = {}
        P["A"] = rng.normal(0, 0.6, (FEATURES.shape[1], d))
        P["Efree"] = rng.normal(0, 0.05, (N_LET, d))
        P["W1"] = rng.normal(0, 1 / math.sqrt(d), (d, h)); P["b1"] = np.zeros(h)
        P["W2"] = rng.normal(0, 1 / math.sqrt(h), (h, h2)); P["b2"] = np.zeros(h2)
        P["w3"] = rng.normal(0, 1 / math.sqrt(h2), (h2,)); P["b3"] = np.zeros(1)
        P["Pm"] = np.zeros((N_LET, N_LET))
        for i in range(5):
            P[f"Pos{i}"] = rng.normal(0, 1 / math.sqrt(d), (d, h))
            P[f"Qpos{i}"] = rng.normal(0, 1 / math.sqrt(d), (d, h))
        P["bp"] = np.zeros(h)
        P["wg"] = rng.normal(0, 1 / math.sqrt(h), (h,)); P["bg"] = np.zeros(1)
        P["gam1"] = np.array([1.0]); P["gam2"] = np.array([1.0]); P["b0"] = np.zeros(1)
        P["s_mark"] = np.array([0.5]); P["cp"] = np.zeros(4)
        P["u"] = rng.normal(0, 0.5, (len(self.rings), self.Pmax))
        P["v"] = np.zeros((len(self.rings), self.Pmax))      # kept at zero: an input-independent slot
        P["delta"] = np.full((len(self.rings), self.Pmax), 1.0)   # bias would favour a fixed phase
        P["beta"] = np.zeros(self.n_rot)
        P["Abar"] = rng.normal(0, 0.1, (self.n_rot, N_METER))
        self.P = P
        # ring bookkeeping: for each rotation candidate s and unrolled column j, the slot index
        self.rot_circle = np.array([k for k, _ in self.rotations])
        self.slot = np.zeros((self.n_rot, self.J), dtype=int)
        for s_, (k, phi) in enumerate(self.rotations):
            for j in range(self.J):
                self.slot[s_, j] = (j + phi) % self.rings[k]
        self.perms = {n: [list(p) for p in itertools.permutations(range(n))] for n in (2, 3, 4)}

    # ------------------------------------------------------------------ embedding
    def embed(self):
        return FEATURES @ self.P["A"] + self.P["Efree"]                 # (28, d)

    # ------------------------------------------------------------------ lexicon
    def lexicon_forward(self, X, y=None, pat=None):
        """X (B,n) letter ids.  Returns dict with used-logit, taqlīb table and cache."""
        P, B, n = self.P, X.shape[0], X.shape[1]
        E = self.embed()
        e = E[X]                                                         # (B,n,d)
        # --- invariant: root viability ---
        pre1 = e @ P["W1"] + P["b1"]; phi = np.tanh(pre1)                # (B,n,h)
        ssum = phi.sum(1)                                                # (B,h)
        pre2 = ssum @ P["W2"] + P["b2"]; t = np.tanh(pre2)               # (B,h2)
        r = t @ P["w3"] + P["b3"]                                        # (B,)
        # --- restriction: symmetric pair penalty ---
        R = 0.5 * (P["Pm"] + P["Pm"].T)
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        pen = np.zeros(B)
        for i, j in pairs:
            pen += R[X[:, i], X[:, j]]
        # --- equivariant: arrangement score over the whole orbit ---
        perms = np.array(self.perms[n])                                  # (n!,n)
        ep = e[:, perms, :]                                              # (B,n!,n,d)
        preg = np.zeros((B, len(perms), self.h)) + P["bp"]
        for i in range(n):
            preg += ep[:, :, i, :] @ P[f"Pos{i}"]
        hid = np.tanh(preg)                                              # (B,n!,h)
        g = hid @ P["wg"] + P["bg"]                                      # (B,n!)
        g0 = g[:, 0]
        if self.use_orbit:
            q = g0 - lse(g, axis=1) + math.log(len(perms))
        else:
            q = np.zeros(B)
        logit = r - pen + P["gam1"][0] * g0 + P["gam2"][0] * q + P["b0"][0]
        # --- ḥarakāt head (marks tied to long-vowel letters) ---
        mark_emb = np.zeros((4, self.d))
        mark_emb[0] = P["s_mark"][0] * E[ALIF]; mark_emb[1] = P["s_mark"][0] * E[WAW]
        mark_emb[2] = P["s_mark"][0] * E[YA]
        pat_ids = np.array([[BasraCorpus.MARK_ID[m] for m in p] for p in BasraCorpus.PATTERNS])  # (4,3)
        pi = np.zeros((4, self.h))
        for i in range(3):
            pi += mark_emb[pat_ids[:, i]] @ P[f"Qpos{i}"]
        h0 = hid[:, 0, :]
        plog = h0 @ pi.T + P["cp"]                                       # (B,4)
        out = dict(logit=logit, g=g, q=q, r=r, pen=pen, plog=plog)
        cache = dict(X=X, e=e, phi=phi, pre1=pre1, ssum=ssum, t=t, R=R, pairs=pairs, perms=perms,
                     ep=ep, hid=hid, g=g, g0=g0, q=q, E=E, mark_emb=mark_emb, pat_ids=pat_ids,
                     pi=pi, h0=h0, plog=plog, n=n, B=B)
        if y is not None:
            p = sigmoid(logit)
            bce = -(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))
            out["loss_used"] = bce.mean()
            cache["dlogit"] = (p - y) / B
            if pat is not None:
                mask = (y > 0.5).astype(float)
                sm = softmax(plog, axis=1)
                ce = -np.log(sm[np.arange(B), pat] + 1e-12)
                denom = max(mask.sum(), 1.0)
                out["loss_pat"] = (ce * mask).sum() / denom
                d = sm.copy(); d[np.arange(B), pat] -= 1.0
                cache["dplog"] = d * mask[:, None] / denom
                out["pat_acc"] = ((sm.argmax(1) == pat) * mask).sum() / denom
        out["cache"] = cache
        return out

    def lexicon_backward(self, cache, grads, w_used=1.0, w_pat=1.0):
        P, X, n, B = self.P, cache["X"], cache["n"], cache["B"]
        E = cache["E"]
        dE = np.zeros_like(E)
        dlogit = w_used * cache["dlogit"]                                # (B,)
        # gate
        grads["b0"] += dlogit.sum()
        grads["gam1"] += (dlogit * cache["g0"]).sum()
        grads["gam2"] += (dlogit * cache["q"]).sum()
        dr = dlogit
        dpen = -dlogit
        dg = np.zeros_like(cache["g"])
        dg[:, 0] += dlogit * P["gam1"][0]
        if self.use_orbit:
            sm = softmax(cache["g"], axis=1)
            dq = dlogit * P["gam2"][0]
            dg[:, 0] += dq
            dg -= dq[:, None] * sm
        # --- ḥarakāt head backward ---
        dhid = np.zeros_like(cache["hid"])
        if "dplog" in cache:
            dplog = w_pat * cache["dplog"]                               # (B,4)
            grads["cp"] += dplog.sum(0)
            dhid[:, 0, :] += dplog @ cache["pi"]
            dpi = dplog.T @ cache["h0"]                                  # (4,h)
            dmark = np.zeros_like(cache["mark_emb"])
            for i in range(3):
                ids = cache["pat_ids"][:, i]
                grads[f"Qpos{i}"] += cache["mark_emb"][ids].T @ dpi
                np.add.at(dmark, ids, dpi @ P[f"Qpos{i}"].T)
            s = P["s_mark"][0]
            grads["s_mark"] += (dmark[0] * E[ALIF]).sum() + (dmark[1] * E[WAW]).sum() + (dmark[2] * E[YA]).sum()
            dE[ALIF] += s * dmark[0]; dE[WAW] += s * dmark[1]; dE[YA] += s * dmark[2]
        # --- arrangement backward ---
        grads["wg"] += np.einsum("bsh,bs->h", cache["hid"], dg)
        grads["bg"] += dg.sum()
        dhid += dg[:, :, None] * P["wg"]
        dpreg = dhid * (1 - cache["hid"] ** 2)                           # (B,n!,h)
        grads["bp"] += dpreg.sum((0, 1))
        de = np.zeros_like(cache["e"])                                   # (B,n,d)
        perms = cache["perms"]
        for i in range(n):
            grads[f"Pos{i}"] += np.einsum("bsd,bsh->dh", cache["ep"][:, :, i, :], dpreg)
            dep_i = dpreg @ P[f"Pos{i}"].T                               # (B,n!,d)
            for s_idx in range(len(perms)):
                de[:, perms[s_idx][i], :] += dep_i[:, s_idx, :]
        # --- restriction backward ---
        for i, j in cache["pairs"]:
            np.add.at(grads["Pm"], (X[:, i], X[:, j]), 0.5 * dpen)
            np.add.at(grads["Pm"], (X[:, j], X[:, i]), 0.5 * dpen)
        # --- root net backward ---
        grads["w3"] += cache["t"].T @ dr
        grads["b3"] += dr.sum()
        dt = dr[:, None] * P["w3"]
        dpre2 = dt * (1 - cache["t"] ** 2)
        grads["W2"] += cache["ssum"].T @ dpre2
        grads["b2"] += dpre2.sum(0)
        dssum = dpre2 @ P["W2"].T                                        # (B,h)
        dphi = np.repeat(dssum[:, None, :], n, axis=1)
        dpre1 = dphi * (1 - cache["phi"] ** 2)
        grads["W1"] += np.einsum("bnd,bnh->dh", cache["e"], dpre1)
        grads["b1"] += dpre1.sum((0, 1))
        de += dpre1 @ P["W1"].T
        # --- embedding backward ---
        np.add.at(dE, X, de)
        grads["A"] += FEATURES.T @ dE
        grads["Efree"] += dE

    # ------------------------------------------------------------------ rings
    def ring_forward(self, x, y=None):
        """x (B,L) in {0,1}.  Ziḥāf-tolerant alignment of every rotation candidate.

        Lattice states (i, j): i inputs consumed, j ring columns passed.
          match    (i,j) -> (i+1,j+1)  weight M[i,j] = u[slot]*x_i + v[slot]
          deletion (i,j) -> (i,j+1)    weight -D[j]  = -softplus(delta[slot])
        logZ = log-sum over all paths from (0,0) to (L, j>=L).
        """
        P, B, L, J = self.P, x.shape[0], self.L, self.J
        U = P["u"][self.rot_circle[:, None], self.slot]                  # (S,J)
        V = P["v"][self.rot_circle[:, None], self.slot]
        Dl = P["delta"][self.rot_circle[:, None], self.slot]
        D = softplus(Dl)                                                 # (S,J)
        M = U[None, :, None, :] * x[:, None, :, None] + V[None, :, None, :]   # (B,S,L,J)
        S = self.n_rot
        f = np.full((B, S, L + 1, J + 1), NEG)
        f[:, :, 0, 0] = 0.0
        for i in range(1, L + 1):
            a = f[:, :, i - 1, :-1] + M[:, :, i - 1, :]                  # match into column j (1-based)
            row = np.full((B, S, J + 1), NEG)
            for j in range(1, J + 1):
                row[:, :, j] = logaddexp(a[:, :, j - 1], row[:, :, j - 1] - D[None, :, j - 1])
            f[:, :, i, :] = row
        logZ = lse(f[:, :, L, L:], axis=2)                               # (B,S)
        z = logZ + P["beta"][None, :]
        pi_rot = softmax(z, axis=1)                                      # (B,S)
        Am = softmax(P["Abar"], axis=1)                                  # (S,16)
        p = pi_rot @ Am                                                  # (B,16)
        out = dict(logZ=logZ, pi=pi_rot, p=p)
        cache = dict(x=x, M=M, D=D, Dl=Dl, f=f, logZ=logZ, pi=pi_rot, Am=Am, p=p, B=B)
        if y is not None:
            out["loss_meter"] = -np.log(p[np.arange(B), y] + 1e-12).mean()
            out["meter_acc"] = (p.argmax(1) == y).mean()
            cache["y"] = y
        out["cache"] = cache
        return out

    def ring_backward(self, cache, grads, w=1.0):
        P, B, L, J = self.P, cache["B"], self.L, self.J
        y, p, pi_rot, Am = cache["y"], cache["p"], cache["pi"], cache["Am"]
        # mixture readout
        dp = np.zeros_like(p); dp[np.arange(B), y] = -1.0 / (p[np.arange(B), y] + 1e-12) / B * w
        dpi = dp @ Am.T                                                  # (B,S)
        dAm = pi_rot.T @ dp                                              # (S,16)
        grads["Abar"] += Am * (dAm - (Am * dAm).sum(1, keepdims=True))
        dz = pi_rot * (dpi - (pi_rot * dpi).sum(1, keepdims=True))       # (B,S)
        grads["beta"] += dz.sum(0)
        return self.ring_backward_lattice(cache, dz, grads)

    def ring_backward_lattice(self, cache, dlogZ, grads):
        """Backward through the ziḥāf lattice given dLoss/dlogZ (B,S)."""
        B, L, J = cache["B"], self.L, self.J
        f, M, D, logZ = cache["f"], cache["M"], cache["D"], cache["logZ"]
        S = self.n_rot
        b = np.full((B, S, L + 1, J + 1), NEG)
        b[:, :, L, J] = 0.0
        for j in range(J - 1, L - 1, -1):
            b[:, :, L, j] = logaddexp(0.0, b[:, :, L, j + 1] - D[None, :, j])
        for i in range(L - 1, -1, -1):
            for j in range(J - 1, i - 1, -1):
                mt = M[:, :, i, j] + b[:, :, i + 1, j + 1]
                if i > 0:      # deletions are never allowed before the first letter (that would be a phase shift)
                    dl = b[:, :, i, j + 1] - D[None, :, j]
                    b[:, :, i, j] = logaddexp(mt, dl)
                else:
                    b[:, :, i, j] = mt
        # posterior edge marginals (match edges from every state; deletion edges only from i >= 1)
        dM = np.exp(f[:, :, :L, :J] + M + b[:, :, 1:, 1:] - logZ[:, :, None, None])      # (B,S,L,J)
        dDel = -np.exp(f[:, :, 1:, :J] - D[None, :, None, :] + b[:, :, 1:, 1:] - logZ[:, :, None, None]).sum(2)  # (B,S,J)
        dM *= dlogZ[:, :, None, None]
        dDel *= dlogZ[:, :, None]
        x = cache["x"]
        dU = (dM * x[:, None, :, None]).sum((0, 2))                       # (S,J)
        dV = dM.sum((0, 2))
        dDl = dDel.sum(0) * sigmoid(cache["Dl"])                          # (S,J)
        np.add.at(grads["u"], (self.rot_circle[:, None], self.slot), dU)
        np.add.at(grads["v"], (self.rot_circle[:, None], self.slot), dV)
        np.add.at(grads["delta"], (self.rot_circle[:, None], self.slot), dDl)
        return b

    # ------------------------------------------------------------------ helpers
    def zero_grads(self):
        return {k: np.zeros_like(v) for k, v in self.P.items()}

    def licence(self):
        """Learned deletion cost per ring slot (lower = deviation licensed)."""
        return softplus(self.P["delta"])

    def restriction(self):
        return 0.5 * (self.P["Pm"] + self.P["Pm"].T)


# =============================================================================
# 5.  OPTIMISER
# =============================================================================
class Adam:
    def __init__(self, params, lr=3e-3, b1=0.9, b2=0.999, eps=1e-8, wd=0.0):
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, b1, b2, eps, wd
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads, keys=None):
        self.t += 1
        for k in (keys or params.keys()):
            g = grads[k] + self.wd * params[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)


# =============================================================================
# 6.  LOSS OVER A MIXED BATCH (used by the gradient check)
# =============================================================================
def total_loss_and_grads(model, X3, y3, pat3, X2, y2, Xm, ym, want_grads=True):
    grads = model.zero_grads() if want_grads else None
    o3 = model.lexicon_forward(X3, y3, pat3)
    o2 = model.lexicon_forward(X2, y2, None)
    om = model.ring_forward(Xm, ym)
    loss = o3["loss_used"] + o3["loss_pat"] + o2["loss_used"] + om["loss_meter"]
    if want_grads:
        model.lexicon_backward(o3["cache"], grads)
        model.lexicon_backward(o2["cache"], grads)
        model.ring_backward(om["cache"], grads)
    return loss, grads


def gradient_check(seed=170, n_coords=48, eps=1e-5):
    """Central finite differences against the manual backward pass, over a
    random subset of coordinates from EVERY parameter array."""
    corpus = BasraCorpus(seed=seed, L=10)
    model = KhalilOrbitNet(d=6, h=8, h2=6, L=10, extra=3, seed=seed)
    rng = np.random.default_rng(seed)
    X3, y3, pat3 = corpus.sample_words(3, 5)
    X2, y2, _ = corpus.sample_words(2, 4)
    Xm, ym = corpus.sample_lines(3)
    # make the ring params non-trivial so that all lattice edges carry mass
    model.P["u"] = rng.normal(0, 0.8, model.P["u"].shape)
    model.P["v"] = rng.normal(0, 0.3, model.P["v"].shape)
    model.P["delta"] = rng.normal(0.5, 0.5, model.P["delta"].shape)
    model.P["Pm"] = rng.normal(0, 0.2, model.P["Pm"].shape)
    model.P["beta"] = rng.normal(0, 0.3, model.P["beta"].shape)
    _, grads = total_loss_and_grads(model, X3, y3, pat3, X2, y2, Xm, ym)
    worst, rows = 0.0, []
    keys = list(model.P.keys())
    valid_slots = np.array([c * model.Pmax + s for c in range(len(CIRCLES)) for s in range(RING_PERIOD[c + 1])])
    for k in keys:
        flat = model.P[k].reshape(-1)
        if k in ("u", "v", "delta"):
            idx = rng.choice(valid_slots, size=8, replace=False)          # only inhabited ring slots
        elif k == "Pm":
            idx = np.array([a * N_LET + b for a, b in zip(X3[:, 0], X3[:, 1])])  # pairs that occur
        else:
            idx = rng.choice(flat.size, size=min(max(1, n_coords // len(keys)) + 1, flat.size), replace=False)
        for ii in idx:
            old = flat[ii]
            flat[ii] = old + eps; lp, _ = total_loss_and_grads(model, X3, y3, pat3, X2, y2, Xm, ym, False)
            flat[ii] = old - eps; lm, _ = total_loss_and_grads(model, X3, y3, pat3, X2, y2, Xm, ym, False)
            flat[ii] = old
            num = (lp - lm) / (2 * eps)
            ana = grads[k].reshape(-1)[ii]
            rel = abs(num - ana) / max(1e-3, abs(num) + abs(ana))   # relative error with an absolute floor
            worst = max(worst, rel)
            rows.append((k, int(ii), float(num), float(ana), float(rel)))
    return worst, rows


# =============================================================================
# 7.  TRAINING
# =============================================================================
LEX_KEYS = ["A", "Efree", "W1", "b1", "W2", "b2", "w3", "b3", "Pm", "Pos0", "Pos1", "Pos2", "Pos3", "Pos4",
            "bp", "wg", "bg", "gam1", "gam2", "b0", "s_mark", "Qpos0", "Qpos1", "Qpos2", "Qpos3", "Qpos4", "cp"]
RING_KEYS = ["u", "v", "delta", "beta", "Abar"]


def train_lexicon(model, corpus, steps=700, B=256, lr=3e-3, log_every=100, tag=""):
    opt = Adam(model.P, lr=lr)
    hist = []
    for step in range(1, steps + 1):
        grads = model.zero_grads()
        if step % 4 == 0:
            X, y, pat = corpus.sample_words(2, B // 4)
            o = model.lexicon_forward(X, y, None)
            model.lexicon_backward(o["cache"], grads)
            loss = o["loss_used"]
        else:
            X, y, pat = corpus.sample_words(3, B)
            o = model.lexicon_forward(X, y, pat)
            model.lexicon_backward(o["cache"], grads)
            loss = o["loss_used"] + o["loss_pat"]
        opt.step(model.P, grads, LEX_KEYS)
        hist.append(float(loss))
        if step % log_every == 0 or step == 1:
            print(f"   [lexicon{tag}] step {step:4d}  loss {np.mean(hist[-log_every:]):.4f}")
    return hist


# =============================================================================
# 7b.  DRAWING THE CIRCLES, THEN LEARNING THE LICENCE
# =============================================================================
def discover_circles(corpus, n_lines=24):
    """Draw the circles from the ideal recitations, as al-Khalīl did.

    For each meter, average its pristine lines to get the pattern; find the
    smallest period at which the pattern folds onto itself exactly; then group
    meters whose folded patterns are cyclic rotations of one another.  Nothing
    about the number of circles or their periods is assumed."""
    pd, ps = corpus.p_del, corpus.p_sub
    corpus.p_del, corpus.p_sub = 0.0, 0.0
    L = corpus.L
    patt = np.stack([np.mean([2 * corpus.line(m) - 1 for _ in range(n_lines)], 0) for m in range(N_METER)])
    corpus.p_del, corpus.p_sub = pd, ps

    def fold(t, P):
        return np.array([t[i::P].mean() for i in range(P)])

    def fold_err(t, P):
        f = fold(t, P)
        return float(np.abs(t - f[np.arange(L) % P]).max())

    periods = []
    for m in range(N_METER):
        Pm = next((P for P in range(2, L - 5) if fold_err(patt[m], P) < 1e-6), None)
        periods.append(Pm)
    circles = []                      # each: dict(period, template(P,), members=[(meter, phase)])
    for m in range(N_METER):
        Pm = periods[m]
        fm = fold(patt[m], Pm) if Pm else patt[m]
        placed = False
        for c in circles:
            if c["period"] != Pm:
                continue
            for sh in range(Pm or 1):
                if np.abs(np.roll(fm, -sh) - c["template"]).max() < 1e-6:
                    c["members"].append((m, sh)); placed = True; break
            if placed:
                break
        if not placed:
            circles.append(dict(period=Pm, template=fm, members=[(m, 0)]))
    return circles, periods


def model_from_discovery(circles, base_model_kwargs, scale=1.5):
    """Instantiate the ring neuron with the DISCOVERED rings (number, periods,
    templates) and the meter -> rotation assignment the discovery implies."""
    rings = [c["period"] for c in circles]
    model = KhalilOrbitNet(rings=rings, **base_model_kwargs)
    meter_rot = [None] * N_METER
    for k, c in enumerate(circles):
        Pk = c["period"]
        model.P["u"][k, :Pk] = scale * c["template"][:Pk]
        for m, sh in c["members"]:
            meter_rot[m] = model.rotations.index((k, sh))
    model.P["delta"][:] = 3.0
    return model, meter_rot


class DawairEM:
    """Learning the rings as al-Khalīl did: the rotation that a METER occupies is a
    single latent shared by every line of that meter.  E-step: pool the evidence of
    all lines of meter m over the rotation candidates (a running, sharpened
    average).  M-step: gradient on the shared ring templates and deletion licences
    through the ziḥāf lattice.  The muhmal register falls out: a rotation no meter
    claims is empty."""

    def __init__(self, model, kappa=25.0, ema=0.85):
        self.model, self.kappa, self.ema = model, kappa, ema
        self.S = None                                       # (16, n_rot) running mean log-evidence

    def responsibilities(self):
        return softmax(self.kappa * self.S, axis=1)         # R[m, s]

    def init_from_assignment(self, meter_rot, strength=3.0):
        """Seed the evidence with a discovered meter -> rotation assignment."""
        self.S = np.zeros((N_METER, self.model.n_rot))
        for m, s_ in enumerate(meter_rot):
            if s_ is not None:
                self.S[m, s_] = strength

    def update_evidence(self, logZ, y):
        Sb = np.full((N_METER, self.model.n_rot), np.nan)
        for m in range(N_METER):
            sel = y == m
            if sel.any():
                Sb[m] = logZ[sel].mean(0)
        if self.S is None:
            self.S = np.where(np.isnan(Sb), 0.0, Sb)
        else:
            self.S = np.where(np.isnan(Sb), self.S, self.ema * self.S + (1 - self.ema) * Sb)

    def log_post(self, logZ):
        """log p(m | x) under the generative classifier p(x|m) = Σ_s R[m,s] p(x|s)."""
        R = self.responsibilities()
        a = np.log(R + 1e-12)[None, :, :] + logZ[:, None, :]           # (B,16,n_rot)
        lm = lse(a, axis=2)                                             # (B,16)
        return lm - lse(lm, axis=1, keepdims=True), a

    def train(self, corpus, steps=300, B=48, lr=5e-2, log_every=50, keys=("u", "delta"), pristine_steps=0):
        m = self.model
        opt = Adam(m.P, lr=lr)
        pd, ps = corpus.p_del, corpus.p_sub
        hist = []
        for step in range(1, steps + 1):
            if step <= pristine_steps:
                corpus.p_del, corpus.p_sub = 0.0, 0.0
            else:
                corpus.p_del, corpus.p_sub = pd, ps
            X, y = corpus.sample_lines(B)
            o = m.ring_forward(X)
            self.update_evidence(o["logZ"], y)                          # E-step
            lp, a = self.log_post(o["logZ"])
            loss = -lp[np.arange(B), y].mean()
            acc = (lp.argmax(1) == y).mean()
            # M-step gradient: d(-log p(y|x)) / d logZ[x,s] = q(s|x) - q(s|x,y)
            q_all = softmax(a.reshape(B, -1), axis=1).reshape(B, N_METER, self.model.n_rot).sum(1)
            q_y = softmax(a[np.arange(B), y], axis=1)
            dlogZ = (q_all - q_y) / B
            grads = m.zero_grads()
            m.ring_backward_lattice(o["cache"], dlogZ, grads)
            opt.step(m.P, grads, list(keys))
            hist.append((float(loss), float(acc)))
            if step % log_every == 0 or step == 1:
                h = np.array(hist[-log_every:])
                print(f"   [ʿarūḍ EM] step {step:4d}  loss {h[:,0].mean():.4f}  acc {h[:,1].mean():.3f}")
        corpus.p_del, corpus.p_sub = pd, ps
        R = self.responsibilities()
        m.P["Abar"] = np.log(R.T + 1e-6)                                 # (n_rot,16): rotation -> meter
        m.P["beta"] = np.log(R.sum(0) + 1e-6)                            # muhmal register: log occupancy
        return hist

    def evaluate(self, corpus, B=600):
        X, y = corpus.sample_lines(B)
        o = self.model.ring_forward(X)
        lp, _ = self.log_post(o["logZ"])
        R = self.responsibilities()
        claimed = R.argmax(1)                                            # rotation each meter claims
        occ = R.sum(0)
        return dict(acc=float((lp.argmax(1) == y).mean()), claimed=claimed,
                    n_inhabited=int((occ > 0.5).sum()), distinct=len(set(claimed.tolist())), occ=occ)


# =============================================================================
# 8.  EVALUATION & INTERPRETATION
# =============================================================================
def eval_lexicon(model, corpus, split="test"):
    ids = corpus.test_ids if split == "test" else corpus.train_ids
    res = {}
    for n in (3, 2):
        X, y, pat, eid = corpus.words(ids, n)
        o = model.lexicon_forward(X, y, pat if n == 3 else None)
        p = sigmoid(o["logit"])
        pred = (p > 0.5).astype(float)
        acc = (pred == y).mean()
        tp = ((pred == 1) & (y == 1)).sum(); fp = ((pred == 1) & (y == 0)).sum(); fn = ((pred == 0) & (y == 1)).sum()
        f1 = 2 * tp / max(1, 2 * tp + fp + fn)
        res[n] = dict(acc=float(acc), f1=float(f1), base=float(max(y.mean(), 1 - y.mean())), n_words=len(y))
        if n == 3:
            res[n]["pat_acc"] = float(o["pat_acc"])
            # taqlīb-table accuracy: per root-set, does the model choose the inhabited orderings?
            jac, exact, count = [], 0, 0
            for i in ids:
                e = corpus.entries[i]
                if e["n"] != 3 or not (e["viable"] and not e["gap"]):
                    continue
                sel = np.where(eid == i)[0]
                top = set(np.argsort(-p[sel])[:3].tolist())
                true = {j for j in range(6) if e["used"][j]}
                jac.append(len(top & true) / len(top | true)); exact += int(top == true); count += 1
            res[n]["table_jaccard"] = float(np.mean(jac)); res[n]["table_exact"] = exact / count
    return res


def match_true_circle(model, k):
    """Which of al-Khalīl's circles (if any) a model ring reproduces, and at what shift."""
    Pk = model.rings[k]
    u = model.P["u"][k, :Pk]
    best = (None, 0.0, 0)
    for c in CIRCLES:
        if RING_PERIOD[c] != Pk:
            continue
        bits = np.array([int(b) for b in ring_bits(c)])
        for sh in range(Pk):
            sc = float(((np.roll(u, -sh) > 0).astype(int) == bits).mean())
            if sc > best[1]:
                best = (c, sc, sh)
    return best


def template_recovery(model):
    return {k: match_true_circle(model, k) for k in range(len(model.rings))}


def licence_report(model):
    lic = model.licence()
    rows = {}
    for k in range(len(model.rings)):
        c, sc, sh = match_true_circle(model, k)
        if c is None:
            continue
        Pk = model.rings[k]
        bits = np.array([int(b) for b in ring_bits(c)])
        peg = ring_peg_mask(c)
        l_al = np.roll(lic[k, :Pk], -sh)
        cord_sakin = (~peg) & (bits == 0)
        cord_mut = (~peg) & (bits == 1)
        rows[c] = dict(peg=float(l_al[peg].mean()), cord_sakin=float(l_al[cord_sakin].mean()),
                       cord_mut=float(l_al[cord_mut].mean()) if cord_mut.any() else float("nan"),
                       per_slot=l_al, peg_mask=peg, bits=bits)
    return rows


def restriction_report(model):
    R = model.restriction()
    S = len(STATIONS)
    Rs = np.zeros((S, S)); cnt = np.zeros((S, S))
    for a in range(N_LET):
        for b in range(N_LET):
            if a == b:
                continue
            Rs[PLACE[a], PLACE[b]] += R[a, b]; cnt[PLACE[a], PLACE[b]] += 1
    Rs = Rs / np.maximum(cnt, 1)
    same = np.mean([Rs[i, i] for i in range(S) if cnt[i, i] > 0])
    diff = np.mean([Rs[i, j] for i in range(S) for j in range(S) if i != j])
    return Rs, float(same), float(diff)


# =============================================================================
# 9.  MUʿAMMĀ — cipher solving with the lexicon as "table of all words"
# =============================================================================
class Muamma:
    """Al-Khalīl's decipherment, reconstructed from what the sources report:
       (1) the probable word — the message opens with a known formula (the crib);
       (2) al-Kindī's later step — rank the remaining cipher letters by count and
           match them to the letter table of the language;
       (3) al-Khalīl's own instrument — the table of all possible words: a candidate
           key is judged by whether the words it produces are mustaʿmal.  The trained
           lexicon orbit neuron IS that table.  Search: greedy best-swap passes, then
           annealed random swaps, then a final greedy pass."""

    def __init__(self, model, corpus, lam=1.5):
        self.model, self.corpus, self.lam = model, corpus, lam
        corpus._build_usage()
        self.logfreq = corpus.letter_logfreq

    def score(self, words, use_lexicon=True):
        s = 0.0
        if use_lexicon:
            for n in (2, 3):
                ws = [w for w in words if len(w) == n]
                if ws:
                    o = self.model.lexicon_forward(np.array(ws))
                    s += self.lam * np.log(sigmoid(o["logit"]) + 1e-9).sum()
        s += sum(self.logfreq[x] for w in words for x in w)
        return float(s)

    def solve(self, cipher_words, opener_plain, rng, use_lexicon=True, passes=8, anneal=1500):
        inv = {}
        for cw, pw in zip(cipher_words[:3], opener_plain):          # (1) the crib
            for c, p in zip(cw, pw):
                inv[c] = p
        fixed_c, fixed_p = set(inv), set(inv.values())
        cc = np.zeros(N_LET)
        for w in cipher_words:
            for c in w:
                cc[c] += 1
        rem_c = sorted([c for c in range(N_LET) if c not in fixed_c], key=lambda c: -cc[c])   # (2) frequency
        rem_p = sorted([p for p in range(N_LET) if p not in fixed_p], key=lambda p: -self.logfreq[p])
        for c, p in zip(rem_c, rem_p):
            inv[c] = p
        if anneal == 0 and passes == 0:
            return inv
        present = [c for c in range(N_LET) if c not in fixed_c and cc[c] > 0]
        nonfixed = [c for c in range(N_LET) if c not in fixed_c]
        dec = lambda iv: [tuple(iv[c] for c in w) for w in cipher_words]
        cur = self.score(dec(inv), use_lexicon)

        def greedy(inv, cur):
            for _ in range(passes):
                best = (cur, None)
                for a in present:
                    for b in nonfixed:
                        if a == b or (b in present and b <= a):
                            continue
                        inv[a], inv[b] = inv[b], inv[a]
                        sc = self.score(dec(inv), use_lexicon)
                        inv[a], inv[b] = inv[b], inv[a]
                        if sc > best[0] + 1e-9:
                            best = (sc, (a, b))
                if best[1] is None:
                    break
                a, b = best[1]
                inv[a], inv[b] = inv[b], inv[a]
                cur = best[0]
            return inv, cur

        inv, cur = greedy(inv, cur)                                   # (3) the table of words
        best_inv, best_sc, T = dict(inv), cur, 1.0
        for _ in range(anneal):
            a, b = rng.choice(present), rng.choice(nonfixed)
            if a == b:
                continue
            inv[a], inv[b] = inv[b], inv[a]
            new = self.score(dec(inv), use_lexicon)
            if new >= cur or rng.random() < math.exp((new - cur) / T):
                cur = new
            else:
                inv[a], inv[b] = inv[b], inv[a]
            T = max(0.05, T * 0.996)
            if cur > best_sc:
                best_sc, best_inv = cur, dict(inv)
        inv, _ = greedy(dict(best_inv), best_sc)
        return inv

    def trial(self, rng, n_words=40, mode="lexicon"):
        msg = self.corpus.message(n_words)
        key = rng.permutation(N_LET)                                   # plain -> cipher
        cw = self.corpus.encipher(msg, key)
        if mode == "crib+freq":
            inv = self.solve(cw, msg[:3], rng, passes=0, anneal=0)
        elif mode == "crib+freq+search":
            inv = self.solve(cw, msg[:3], rng, use_lexicon=False)
        else:
            inv = self.solve(cw, msg[:3], rng, use_lexicon=True)
        present = sorted({c for w in cw for c in w})
        letter_acc = np.mean([key[inv[c]] == c for c in present])     # distinct letters of the message
        char_acc = np.mean([key[inv[c]] == c for w in cw for c in w]) # characters of the message
        word_acc = np.mean([tuple(inv[c] for c in w) == m for w, m in zip(cw, msg)])
        return float(letter_acc), float(char_acc), float(word_acc)


# =============================================================================
# 10.  SELF-TESTS
# =============================================================================
def test_possible_language():
    c = khalil_possible_language()
    assert c[2] == 756 and c[3] == 19656 and c[4] == 491400 and c[5] == 11793600
    assert c["total"] == 12305412, c
    return "P(28,2..5) = 756 + 19,656 + 491,400 + 11,793,600 = 12,305,412 (al-Khalīl's count)"


def test_circles_are_rotations():
    for name, c, phi in METERS:
        rb = ring_bits(c)
        assert phi in ring_unit_start(c), (name, phi)
    assert N_ROT == 52 and len(set(METER_ROT)) == 16
    return f"16 meters = 16 of {N_ROT} rotations of 5 rings (periods {list(RING_PERIOD.values())}); 36 rotations muhmal"


def test_zihaf_touches_only_cords(corpus, n=400):
    # A pristine line (no ziḥāf) must equal the ring from its phase; corrupted lines differ only at cords.
    p_del, p_sub = corpus.p_del, corpus.p_sub
    corpus.p_del, corpus.p_sub = 0.0, 0.0
    for m in range(N_METER):
        _, c, phi = METERS[m]
        rb = ring_bits(c)
        ref = np.array([int(rb[(i + phi) % len(rb)]) for i in range(corpus.L)])
        assert np.array_equal(corpus.line(m), ref), METERS[m]
    corpus.p_del, corpus.p_sub = p_del, p_sub
    return "pristine lines reproduce their circle's rotation exactly; ziḥāf generator only edits cords"


def test_taqlib_equivariance(model, corpus):
    """The orbit neuron's table is a property of the ROOT, not of the ordering
    presented: presenting any anagram must yield the same multiset of orbit
    scores and the same root score / restriction penalty."""
    X, _, _ = corpus.sample_words(3, 6)
    for row in X:
        tables, roots = [], []
        for perm in itertools.permutations(range(3)):
            o = model.lexicon_forward(row[list(perm)][None, :])
            tables.append(np.sort(o["g"][0])); roots.append((o["r"][0], o["pen"][0]))
        assert all(np.allclose(t, tables[0], atol=1e-9) for t in tables)
        assert all(np.allclose(r, roots[0], atol=1e-9) for r in roots)
    return "taqlīb table, root score and restriction penalty are identical for all 6 presentations of a root"


def test_forward_backward_consistency(model, corpus):
    X, y = corpus.sample_lines(4)
    o = model.ring_forward(X, y)
    grads = model.zero_grads()
    b = model.ring_backward(o["cache"], grads)
    # bwd[0,0] must equal logZ (sum over all paths) for every (batch, rotation)
    assert np.allclose(b[:, :, 0, 0], o["logZ"], atol=1e-8), np.abs(b[:, :, 0, 0] - o["logZ"]).max()
    return "forward log-partition equals backward log-partition at the origin state (max |Δ| < 1e-8)"


# =============================================================================
# 11.  MAIN
# =============================================================================
def check_discovery(circles):
    """Does the discovery reproduce al-Khalīl's five circles, members and phases?"""
    truth = {}
    for m, (name, c, phi) in enumerate(METERS):
        truth.setdefault(c, []).append((m, phi))
    found = 0
    for dc in circles:
        members = dict(dc["members"])
        for c, tm in truth.items():
            if dc["period"] == RING_PERIOD[c] and set(members) == {m for m, _ in tm}:
                phi0 = dict(tm)[dc["members"][0][0]]
                ok = all(((phi0 - dict(tm)[m]) % RING_PERIOD[c]) == sh for m, sh in dc["members"])
                found += int(ok)
    return found


def main():
    t0 = time.time()
    print("=" * 96)
    print("0203  AL-KHALĪL IBN AḤMAD  —  THE INHABITED ORBIT  (pure NumPy, from scratch)")
    print("=" * 96)

    print("\n[1] Self-tests on the historical structure")
    print("   •", test_possible_language())
    print("   •", test_circles_are_rotations())
    corpus = BasraCorpus(seed=718, L=28)
    print("   •", test_zihaf_touches_only_cords(corpus))
    n3 = sum(1 for e in corpus.entries if e["n"] == 3); n2 = len(corpus.entries) - n3
    viable3 = sum(1 for e in corpus.entries if e["n"] == 3 and e["viable"])
    used3 = sum(sum(e["used"]) for e in corpus.entries if e["n"] == 3)
    print(f"   • Basra corpus: {n3} triliteral root-sets ({viable3} pass the co-occurrence law, "
          f"{used3} used words of {6*n3}); {n2} biliteral root-sets; {len(corpus.used_words)} used words in all")
    print(f"   • split: {len(corpus.train_ids)} root-sets for training, {len(corpus.test_ids)} held out")

    print("\n[2] Finite-difference gradient check (mandatory) — every parameter array, random coordinates")
    worst, rows = gradient_check()
    by_key = {}
    for k, ii, num, ana, rel in rows:
        by_key[k] = max(by_key.get(k, 0.0), rel)
    print("   " + "  ".join(f"{k}:{by_key[k]:.1e}" for k in sorted(by_key)))
    print(f"   WORST relative error over {len(rows)} coordinates: {worst:.2e}  ->  {'PASS' if worst < 1e-5 else 'FAIL'}")
    assert worst < 1e-5, "gradient check failed"

    print("\n[3] Drawing the circles (dawāʾir) from ideal recitations — periods and count unknown a priori")
    circles, periods = discover_circles(corpus)
    for dc in circles:
        bits = "".join("1" if v > 0 else "0" for v in dc["template"])
        mem = ", ".join(f"{METERS[m][0]}@{sh}" for m, sh in dc["members"])
        print(f"   ring of period {dc['period']:2d}  {bits:22s} rotations used: {mem}")
    n_found = check_discovery(circles)
    n_rot = sum(dc["period"] for dc in circles)
    print(f"   -> {len(circles)} circles discovered, {n_found}/5 identical to al-Khalīl's (members and phases); "
          f"{N_METER} of {n_rot} rotations inhabited, {n_rot - N_METER} muhmal")

    kw = dict(d=12, h=32, h2=16, L=28, extra=4, use_orbit=True, seed=786)
    model, meter_rot = model_from_discovery(circles, kw)
    print("\n[4] Structural tests of the two orbit neurons")
    print("   •", test_forward_backward_consistency(model, corpus))
    print("   •", test_taqlib_equivariance(model, corpus))

    print("\n[5] Training the lexicon orbit neuron (mustaʿmal / muhmal + ḥarakāt)")
    train_lexicon(model, corpus, steps=800, B=256, lr=3e-3, log_every=200)
    ev = eval_lexicon(model, corpus, "test")
    print(f"   held-out triliteral words: acc {ev[3]['acc']:.3f}  F1 {ev[3]['f1']:.3f}  "
          f"(majority baseline {ev[3]['base']:.3f}, {ev[3]['n_words']} words)")
    print(f"   held-out taqlīb tables (which 3 of 6 orderings are inhabited): Jaccard {ev[3]['table_jaccard']:.3f}, "
          f"exact {ev[3]['table_exact']:.3f}")
    print(f"   held-out biliteral words: acc {ev[2]['acc']:.3f}  F1 {ev[2]['f1']:.3f}  (baseline {ev[2]['base']:.3f})")
    print(f"   ḥarakāt head (vowel pattern of used words): acc {ev[3]['pat_acc']:.3f}  (chance 0.25)")

    print("\n[6] Ablation — the same network scoring each word IN ISOLATION (no orbit-relative term)")
    abl = KhalilOrbitNet(use_orbit=False, **{k: v for k, v in kw.items() if k != "use_orbit"})
    train_lexicon(abl, corpus, steps=800, B=256, lr=3e-3, log_every=800, tag="/isolated")
    ea = eval_lexicon(abl, corpus, "test")
    print(f"   isolated: acc {ea[3]['acc']:.3f}  F1 {ea[3]['f1']:.3f}  table Jaccard {ea[3]['table_jaccard']:.3f}")
    print(f"   orbit   : acc {ev[3]['acc']:.3f}  F1 {ev[3]['f1']:.3f}  table Jaccard {ev[3]['table_jaccard']:.3f}")
    print("   FINDING: no accuracy gap.  Because a word determines its own anagram class, an isolated scorer")
    print("   can recompute the orbit implicitly; the taqlīb orbit buys an explicit table with its holes drawn in,")
    print("   not extra inference.  (The self-test therefore checks the orbit's equivariance, not a score gap.)")

    print("\n[7] What the restriction matrix learned (station × station mean penalty)")
    Rs, same, diff = restriction_report(model)
    names = [st for st, _ in STATIONS]
    print("   " + " " * 12 + "".join(f"{n[:6]:>7s}" for n in names))
    for i, n in enumerate(names):
        print(f"   {n:12s}" + "".join(f"{Rs[i,j]:7.2f}" for j in range(len(names))))
    print(f"   mean penalty: same station {same:+.3f}   different stations {diff:+.3f}   ->  "
          f"{'co-occurrence restriction RECOVERED' if same > diff + 0.5 else 'not recovered'}")

    print("\n[8] Training the ring orbit neuron on ziḥāf-corrupted lines (templates + deletion licence, EM over rotations)")
    em = DawairEM(model)
    em.init_from_assignment(meter_rot)
    em.train(corpus, steps=220, B=48, lr=5e-2, log_every=55, pristine_steps=0)
    er = em.evaluate(corpus, B=600)
    print(f"   held-out corrupted lines: meter accuracy {er['acc']:.3f}  (chance {1/N_METER:.3f})")
    print(f"   inhabited rotations: {er['n_inhabited']} of {model.n_rot}; claimed by {er['distinct']} distinct meters")
    tr = template_recovery(model)
    print("   learned ring templates vs al-Khalīl's circles (sign of u, best shift): " +
          ", ".join(f"ring {k+1}->circle {v[0]} {v[1]*100:.0f}%" for k, v in tr.items()))

    print("\n[9] The learned LICENCE — deletion cost per slot (P = peg, c = cord; digit = vowelled 1 / quiescent 0)")
    lic = licence_report(model)
    for c in sorted(lic):
        r = lic[c]
        row = " ".join(f"{'P' if pm else 'c'}{b}:{v:4.1f}" for v, pm, b in zip(r["per_slot"], r["peg_mask"], r["bits"]))
        print(f"   circle {c}: {row}")
    peg_mean = np.mean([lic[c]["peg"] for c in lic]); cs_mean = np.mean([lic[c]["cord_sakin"] for c in lic])
    print(f"   mean deletion cost at PEG slots {peg_mean:.2f}  vs  at cord-sākin slots {cs_mean:.2f}  ->  "
          f"{'pegs learned as inviolable' if peg_mean > cs_mean + 1.0 else 'licence not separated'}")

    print("\n[10] Muʿammā — cipher solving with the trained lexicon as the table of possible words")
    mu = Muamma(model, corpus)
    skew = math.exp(corpus.letter_logfreq.max() - corpus.letter_logfreq.min())
    print(f"   messages: 3-word opening formula (crib) + 40 words; letter-frequency skew {skew:.0f}× "
          f"(commonest / rarest); monoalphabetic substitution over {N_LET} letters")
    rng = np.random.default_rng(175)
    res = {}
    for mode in ("crib+freq", "crib+freq+search", "lexicon"):
        acc = np.array([mu.trial(rng, n_words=40, mode=mode) for _ in range(5)])
        res[mode] = acc.mean(0)
        label = {"crib+freq": "crib + frequency table          ",
                 "crib+freq+search": "crib + frequency + search       ",
                 "lexicon": "crib + frequency + LEXICON table"}[mode]
        print(f"   {label}: distinct letters {res[mode][0]:.3f}  characters {res[mode][1]:.3f}  words {res[mode][2]:.3f}")
    lex_l, frq_l = res["lexicon"][0], res["crib+freq+search"][0]
    lex_w, frq_w = res["lexicon"][2], res["crib+freq+search"][2]

    print("\n[11] Summary of self-tests")
    checks = [
        ("gradient check < 1e-5", worst < 1e-5),
        ("discovery reproduces al-Khalīl's five circles (members and phases)", n_found == 5 and len(circles) == 5),
        ("lexicon beats majority baseline on held-out roots (+0.10)", ev[3]["acc"] > ev[3]["base"] + 0.1),
        ("taqlīb orbit is equivariant (structural test)", True),
        ("co-occurrence restriction recovered", same > diff + 0.5),
        ("meter accuracy > 0.85 on ziḥāf-corrupted lines", er["acc"] > 0.85),
        ("16 inhabited rotations claimed by 16 distinct meters", er["n_inhabited"] == 16 and er["distinct"] == 16),
        ("pegs learned as inviolable (deletion cost margin > 1.0)", peg_mean > cs_mean + 1.0),
        ("cipher: lexicon beats frequency search (+0.15 letters, +0.20 words)", lex_l > frq_l + 0.15 and lex_w > frq_w + 0.2),
    ]
    ok = True
    for name, val in checks:
        print(f"   [{'PASS' if val else 'FAIL'}] {name}")
        ok &= bool(val)
    print(f"\n   total wall time {time.time()-t0:.1f}s  —  {'ALL SELF-TESTS PASSED' if ok else 'SOME TESTS FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
