#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 ORIGO  —  The Derivation Cell
 Chapter 0178 · Isidore of Seville (c. 560 – 4 April 636)
 Encyclopedia of Lost Minds: Echoes on AI
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 9 Minds 161 - 180 Available on Amazon https://www.amazon.com/dp/B0HJC1QF59
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 0178_isidore_of_seville_560 - Isidore of Seville (c. 560 – 4 April 636)
================================================================================  

WHAT THIS FILE IS
-----------------
A from-scratch, pure-NumPy neural architecture (no autograd, no frameworks)
whose *structure* encodes the one cognitive commitment that is Isidore's and
nobody else's in this corpus: A NAME IS A COMPRESSED RECORD OF THE REASON IT
WAS GIVEN, AND UNDERSTANDING IS THE DECOMPRESSION OF THAT RECORD — except
where no such record exists, and then the mind must say so.

The two halves of that sentence are both his, and they are stated three
sentences apart in Etymologiae I.29.2-3 (Lindsay 1911):

    "Nam dum videris unde ortum est nomen, citius vim eius intellegis."
     For when you have seen where a name arose from, you grasp its force
     more quickly.

    "Non autem omnia nomina a veteribus secundum naturam inposita sunt, sed
     quaedam et secundum placitum ... Hinc est quod omnium nominum
     etymologiae non reperiuntur."
     But not all names were imposed by the ancients according to nature;
     some were given according to whim ... Hence the etymologies of all
     names cannot be found.

He then lists the ways a name can carry its reason (I.29.3-5): ex causa (from
its cause: 'rex' from 'regere'), ex origine (from its origin: 'homo' from
'humus'), ex contrariis (from its contrary: 'lucus' because it does not
'lucere'), ex derivatione (from another name: 'prudens' from 'prudentia'), ex
vocibus (from a sound), ex Graeca etymologia (transliterated from Greek), and
names taken from places. And in III.3.2 he applies the escape clause to a
specific word: the number 'five' (quinque), he says, was named "non secundum
naturam, sed secundum placitum voluntatis" — not by nature but by the whim of
whoever named the numbers. He does not invent an origin for it. He records
that there is none.

THE ARCHITECTURAL CLAIM
-----------------------
Build a cell whose representation of a symbol is not a lookup but a
DERIVATION:

    vis(name) =  (1 - gamma) * OP[mode](vis(root found inside name))
               +      gamma  * MEMORY(name)

  * the SENSUS reads the surface string through a small set of hypotheses
    about where the root sits inside it and whether it must first be read
    back through a transliteration cipher (ex Graeca);
  * the LEXICON scores each candidate window against the roots it knows —
    prototypes that each spell essentially one string — and against a NULL
    ROOT that says "this window is no root I know";
  * the RATIO chooses one of the derivation operators and applies it to the
    root's bounded force — the operators are learned, and what they learn
    to be is identity (ex causa / ex derivatione / ex Graeca), negation
    (ex contrariis), and two fixed re-arrangements (ex origine, ex vocibus);
    the mode is read from the RESIDUE, the letters around the recognised
    root, so that all names ending in -ns look alike to the mode reader
    whatever their root;
  * the PLACITUM GATE gamma decides whether the name is derivable at all.
    When it is not, the cell falls back on a memory written by exposure —
    it holds the surfaces of the names it has read — and learns only what
    each of them means. An unread arbitrary name recalls nothing, and the
    cell's answer collapses to indifference (p = 0.5). That collapse is not
    a failure mode. It is Isidore's "non reperiuntur" rendered as a number.

A small cost is charged for every use of the gate: the cell must PREFER to
derive, and may only decline when derivation will not fit. That is the order
of operations of the Etymologiae itself — seek the origin first, admit
convention second, never manufacture an origin to avoid the admission.

Five structural refusals make the gate necessary rather than decorative,
and every one of them was found by watching an earlier version of this cell
invent etymologies: a root is a prototype, not a region; a window that runs
into the padding is no root; the null root has a fixed threshold; a root's
force and every operator are bounded; and the transliteration is a
letter-to-letter map with a one-to-one prior. Remove any one and the cell
becomes a memoriser that calls itself an etymologist.

THE TWO COUNTERFACTUALS
-----------------------
  CRATYLUS   gamma is forced to 0 and the lexicon has no null root: every
             window resembles its nearest root, and every name must be
             derived. This is the caricature of Isidore that later critics
             drew (Peter Jones: "most of his derivations are total nonsense")
             — a mind that cannot say "arbitrary" and therefore invents an
             origin for everything. We measure what that costs: the lexicon
             is warped, systematic generalisation to unseen derivations
             degrades, and on unseen arbitrary names the model emits
             CONFIDENT nonsense.
  NOMINALIST gamma is forced to 1. Nothing is derived; every name is a
             convention to be remembered. It recalls by resemblance, which
             is enough wherever a derivation preserves the root's force, and
             fails exactly where the derivation transforms it.

THE TASK
--------
A synthetic lexicon generated by a hidden process that mirrors I.29: roots
with hidden four-bit meanings ("vis"); the bare roots themselves as words;
derived names built from a root by one of six modes, each mode a fixed
transformation of the meaning; and a set of arbitrary names (placitum)
whose meaning is unrelated to any root. The test split holds out entire
(root, mode) pairs: a test name is one whose root was seen in other modes
and whose mode was seen on other roots, but whose combination was never
seen. Only a mind that has learned the derivation rule — not the individual
names — can answer. Test arbitrary names are unseen too, and the correct
answer for them is to know that one does not know.

Training follows Isidore's own order in three stretches: GRAMMATICA (the
bare words, until the lexicon has hardened), ORIGINES (the derivations and
the Greek lesson, with the gate shut so that derivation is tried before
anything else), and PLACITUM (the gate opens; the memory reads the stream;
convention is admitted where derivation has failed).

CONVENTIONS KEPT FROM THE CORPUS
--------------------------------
  * pure NumPy, hand-derived analytic gradients for every parameter
  * a finite-difference gradient check that must pass (mandatory)
  * a real training loop on a real train/validation split
  * self-tests covering the structural invariants, not just the loss
  * executed before shipping; the printed output is the verified output

THE SEVEN NAMES
---------------
Etymologiae XI.1.12-13: "haec omnia adiuncta sunt animae ut una res sit. Pro
efficientiis enim causarum diversa nomina sortita est anima" — all of these
are joined to the soul so that it is ONE THING; the soul receives different
names according to the effects of its operations: anima when it gives life,
animus when it wills, mens when it knows, memoria when it recollects, ratio
when it judges rightly, spiritus when it breathes, sensus when it senses.
The sections of this file are named after those operations. They are not
modules of a mind. They are one computation, read under seven names.

RUN:  python3 0178_isidore_of_seville_560_Neuron.py
================================================================================
"""

import sys
import time
import numpy as np

# ==============================================================================
# 0.  SMALL UTILITIES
# ==============================================================================

def sigmoid(z):
    """Numerically stable logistic."""
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[neg])
    out[neg] = ez / (1.0 + ez)
    return out


def softmax(z, axis=-1):
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def logsumexp(z, axis=-1):
    m = np.max(z, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(z - m), axis=axis, keepdims=True))).squeeze(axis)


def bce_with_logits(z, y):
    """Element-wise binary cross-entropy from logits, stable."""
    return np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))


def rule(ch="=", n=78):
    print(ch * n)


# ==============================================================================
# 1.  THE LEXICON  —  a world in which names carry their reasons (mostly)
# ==============================================================================
#
# The generator is the "ancients" of I.29: it imposes names on things. Most
# names are imposed secundum naturam — built from a root by a mode, so the
# name records why the thing is called what it is called. A minority are
# imposed secundum placitum: a string with no root inside it and a meaning
# that no rule could recover.

ALPHABET = "abcdeilmnorst"         # 13 letters; Latin-flavoured on purpose
PAD = "_"
CHARS = ALPHABET + PAD
A = len(CHARS)                    # 14 symbols incl. pad
L = 8                             # max name length
ROOTLEN = 3
O = L - ROOTLEN + 1               # 6 possible root offsets
D = 4                             # bits of meaning ("vis")

MODES = ["simplex", "causa", "origine", "contrariis", "derivatione", "vocibus", "graeca"]
M = len(MODES)
SIMPLEX = 0                       # the bare root as a word of the language
PLACITUM = M                      # index of the arbitrary class in labels

# The affixes each mode uses. Two surface variants per mode so that the mode
# is a *pattern* to be recognised, not a single string.
AFFIX = {
    "simplex":     [("", "")],                           # the root itself: 'regere'
    "causa":       [("", "ns"), ("", "tor")],            # rex a regendo
    "origine":     [("e", ""), ("de", "")],              # homo ex humo
    "contrariis":  [("in", ""), ("non", "")],            # lucus a non lucendo
    "derivatione": [("", "nsia"), ("", "toria")],        # prudens -> prudentia
    "vocibus":     [("REDUP", "o"), ("REDUP", "")],      # onomatopoeia
    "graeca":      [("CIPHER", "is"), ("CIPHER", "os")], # transliteration
}

# What each mode does to the root's meaning. These are signed permutations of
# the four bits — exactly the class of maps a linear operator on logits can
# represent, so the cell CAN recover them; whether it DOES is what we test.
def apply_mode(vis, mode):
    v = np.array(vis, dtype=np.int64)
    if mode in ("simplex", "causa", "derivatione", "graeca"):
        return v.copy()                     # the name keeps the root's force
    if mode == "contrariis":
        return 1 - v                        # lucus a non lucendo
    if mode == "origine":
        return v[[3, 0, 1, 2]]              # the derived thing shifts the material's attributes
    if mode == "vocibus":
        return v[[1, 0, 3, 2]]              # a sound-name carries them re-paired
    raise ValueError(mode)


def encode(name):
    """One-hot (L, A) with right padding."""
    x = np.zeros((L, A), dtype=np.float64)
    for i in range(L):
        ch = name[i] if i < len(name) else PAD
        x[i, CHARS.index(ch)] = 1.0
    return x


class Lexicon:
    """The generative process behind the names, plus the train/test split."""

    def __init__(self, n_roots=32, n_arbitrary=96, holdout_frac=0.25, seed=0):
        rng = np.random.RandomState(seed)
        self.rng = rng
        # a fixed transliteration: Greek letter -> Latin letter, as a permutation
        perm = rng.permutation(len(ALPHABET))
        self.cipher = {ALPHABET[i]: ALPHABET[perm[i]] for i in range(len(ALPHABET))}
        self.decipher = {v: k for k, v in self.cipher.items()}

        # distinct roots
        roots = set()
        while len(roots) < n_roots:
            roots.add("".join(rng.choice(list(ALPHABET), ROOTLEN)))
        self.roots = sorted(roots)
        self.root_vis = {r: rng.randint(0, 2, D) for r in self.roots}
        self.root_index = {r: i for i, r in enumerate(self.roots)}

        # every derived name: (name, vis, root, mode, offset, ciphered)
        derived = []
        for r in self.roots:
            for mi, mode in enumerate(MODES):
                for pre, suf in AFFIX[mode]:
                    if pre == "REDUP":
                        pre_s = r[:2]
                    elif pre == "CIPHER":
                        pre_s = ""
                    else:
                        pre_s = pre
                    core = "".join(self.cipher[c] for c in r) if mode == "graeca" else r
                    name = pre_s + core + suf
                    assert len(name) <= L, name
                    vis = apply_mode(self.root_vis[r], mode)
                    derived.append(dict(name=name, vis=vis, root=r, mode=mi,
                                        offset=len(pre_s), ciphered=(mode == "graeca")))
        self.derived = derived

        # arbitrary names: no root visible in any window, plain or deciphered
        arbitrary = []
        seen = {d["name"] for d in derived}
        while len(arbitrary) < n_arbitrary:
            n = rng.randint(4, 7)
            s = "".join(rng.choice(list(ALPHABET), n))
            if s in seen or self._contains_root(s):
                continue
            seen.add(s)
            arbitrary.append(dict(name=s, vis=rng.randint(0, 2, D), root=None,
                                  mode=PLACITUM, offset=-1, ciphered=False))
        self.arbitrary = arbitrary

        # hold out whole (root, mode) pairs — the systematic-generalisation split.
        # The bare roots (simplex) are never held out: a mind that derives
        # 'rex' from 'regere' is assumed to know the word 'regere'.
        pairs = [(r, mi) for r in self.roots for mi in range(M) if mi != SIMPLEX]
        rng.shuffle(pairs)
        n_hold = int(round(holdout_frac * len(pairs)))
        held = set(pairs[:n_hold])
        self.train = [d for d in derived if (d["root"], d["mode"]) not in held]
        self.test = [d for d in derived if (d["root"], d["mode"]) in held]
        n_arb_test = n_arbitrary // 3
        self.train += arbitrary[n_arb_test:]
        self.test += arbitrary[:n_arb_test]
        rng.shuffle(self.train)
        rng.shuffle(self.test)

    def _contains_root(self, s):
        for o in range(len(s) - ROOTLEN + 1):
            w = s[o:o + ROOTLEN]
            if w in self.root_index:
                return True
            dw = "".join(self.decipher[c] for c in w)
            if dw in self.root_index:
                return True
        return False

    @staticmethod
    def tensors(items):
        X = np.stack([encode(d["name"]) for d in items])
        Y = np.stack([d["vis"] for d in items]).astype(np.float64)
        MODE = np.array([d["mode"] for d in items])
        return X, Y, MODE

    def describe(self):
        print(f"  roots: {len(self.roots)}   derived names: {len(self.derived)}   "
              f"arbitrary names: {len(self.arbitrary)}")
        print(f"  train: {len(self.train)} (incl. {sum(d['mode'] == SIMPLEX for d in self.train)} bare roots)"
              f"   test: {len(self.test)} "
              f"({sum(d['mode'] != PLACITUM for d in self.test)} unseen derivations, "
              f"{sum(d['mode'] == PLACITUM for d in self.test)} unseen arbitrary)")
        r = self.roots[0]
        print(f"  example root '{r}' vis={self.root_vis[r]} ->")
        for d in self.derived:
            if d["root"] == r:
                print(f"     {d['name']:<9} {MODES[d['mode']]:<12} vis={d['vis']}")


# ==============================================================================
# 2.  THE CELL  —  ORIGO
# ==============================================================================
#
#   x (L,A) one-hot name
#     |
#     |  SENSUS ..... windows at six offsets, read plain or through the cipher K
#     v
#   phi (H=12, 3A)
#     |
#     |  MEMORIA .... score each window against the lexicon of root PROTOTYPES,
#     |               plus a null root: "this window is no root I know"
#     v
#   S (H,R+1) --> P(r|h) = softmax_r ; LSE_h = "does this window look like any root"
#     |
#     |  RATIO ...... choose the hypothesis h (offset+cipher) and the mode m
#     v
#   rv = sum_h p_h sum_{r<R} P(r|h) V_r      (the root's force, recovered; null gives 0)
#   zd = sum_m p_m A_m rv                    (the operator of the mode applied)
#     |
#     |  ANIMUS ..... the placitum gate: derivable, or merely given?
#     v
#   z  = (1-gamma) zd + gamma zm ;  zm = recognition memory with a null slot
#     |
#     |  MENS ....... the vis: p = sigmoid(z), four bits of meaning
#     v
#
# Three structural refusals make the gate NECESSARY rather than decorative:
#   * a root is a prototype (a letter distribution per position), so one root
#     can match essentially one string — the lexicon cannot be bent into a
#     memory of arbitrary windows without losing its real roots;
#   * the lexicon has a null root, so a window that resembles nothing yields
#     no force (zd -> 0, p -> 0.5) instead of the nearest root's force;
#   * the operators carry no bias, so a mode cannot smuggle in a memorised
#     meaning for a name that has no root.
# Under those refusals, an arbitrary name can only be answered by the memory,
# and the memory can only answer names it has met. Everything the chapter
# measures follows from those three lines.

def log_softmax(z, axis=-1):
    return z - logsumexp(z, axis=axis)[..., None] if axis == -1 else z - np.expand_dims(logsumexp(z, axis=axis), axis)


class Origo:
    """The derivation cell. One state, seven names, hand-derived gradients."""

    TAU_MEM = 30.0     # sharpness of recognition memory (cosine -> logit)
    TAU_MODE = 1.0     # temperature of the mode choice
    LAM_ENT = 0.02     # final cost on hedging between modes (a name has one etymology)
    REL = (-3, -2, -1, 3, 4, 5, 6, 7)   # residue slots relative to the root
    VMAX = 4.0         # a root's force is bounded (logit magnitude); operators are
                       # bounded maps, so a faint match cannot be inflated into a meaning
    LAM_PERM = 0.1     # a transliteration is one-to-one: columns of K should sum to 1
    KAPPA_FINAL = 2.0  # match sharpness of the lexicon once learned (a root is a string)
    NULL_LOGP = -1.0   # null-root threshold in log-prob units (~0.72 per letter)

    def __init__(self, n_slots=48, d=D, n_mem=400, seed=1, gate_mode="isidore",
                 lam_gate=0.05, null_root=True):
        rng = np.random.RandomState(seed)
        self.R, self.d, self.N = n_slots, d, n_mem     # lexicon slots (over-provisioned)
        self.H = 2 * O                         # 6 offsets x {plain, greek}
        self.gate_mode = gate_mode             # 'isidore' | 'cratylus' | 'nominalist'
        self.lam_gate = lam_gate
        # match sharpness, annealed by the trainer (soft -> sharp), separately
        # for the native reading and for the reading through the cipher
        self.kappa_plain = self.KAPPA_FINAL
        self.kappa_greek = self.KAPPA_FINAL
        self.use_greek = True                  # phase I of training masks the cipher
        self.gate_closed = False               # phase IIa: derivation only, no memory
        self.exposed = 0                       # how many names the memory has read
        self.lam_ent = 0.0                     # ramped in by the trainer
        self.tau_mode = self.TAU_MODE
        self.null_root = null_root             # False = every window has a nearest root
        G = L * A                              # flattened surface size
        p = {}
        # SENSUS: the transliteration cipher (Greek -> Latin), learned as a
        # letter-to-letter map: each ciphered letter is read as a distribution
        # over plain letters (row-stochastic), never as a free linear weight
        p["thetaK"] = 0.05 * rng.randn(A, A)
        # MEMORIA: the lexicon of root prototypes (logits over letters, per position)
        p["theta"] = 0.30 * rng.randn(n_slots, ROOTLEN, A)
        p["thetaV"] = 0.30 * rng.randn(n_slots, d)   # the vis of each root (pre-tanh)
        # RATIO: the window that looks most like a root IS the root (no free
        # surface-driven chooser: recognition, not rule, positions the root);
        # the mode is read from the surface AND from where the root was found
        p["c"] = np.zeros(self.H)               # prior over offset x cipher
        p["beta"] = np.array([1.0])             # weight on "looks like a root"
        p["Wm"] = 0.10 * rng.randn(M, len(self.REL) * A + 1)   # +1: the echo cue
        p["Wh"] = 0.10 * rng.randn(M, self.H)
        p["cm"] = np.zeros(M)
        p["thetaA"] = np.stack([np.eye(d) * 0.5 + 0.30 * rng.randn(d, d) for _ in range(M)])
        # ANIMUS: the placitum gate
        p["wg"] = 0.05 * rng.randn(G)
        p["ug"] = np.array([-0.5])
        p["bg"] = np.array([0.0])
        # recognition memory for names that are merely given
        p["Km"] = rng.randn(n_mem, G)
        p["tm"] = np.zeros(n_mem) - 24.0        # threshold: recall only above cos ~0.8
        p["thetaVm"] = 0.10 * rng.randn(n_mem, d)
        self.p = p
        lm = np.ones((L, A)); lm[:, A - 1] = 0.0
        self._letter_mask = lm.reshape(-1)

    # ---------------------------------------------------------------- forward
    def forward(self, X, cache=True):
        p = self.p
        B = X.shape[0]
        G = L * A
        R, N, H = self.R, self.N, self.H
        g = X.reshape(B, G)

        # ---- SENSUS: six windows, each read plain and through the cipher
        Wn = np.stack([X[:, o:o + ROOTLEN, :] for o in range(O)], axis=1)  # (B,O,3,A)
        phi_plain = Wn.reshape(B, O, ROOTLEN * A)
        K = softmax(p["thetaK"], axis=1)                                    # (A,A) row-stochastic
        Wk = Wn @ K                                                         # (B,O,3,A)
        phi_greek = Wk.reshape(B, O, ROOTLEN * A)
        phi = np.concatenate([phi_plain, phi_greek], axis=1)                # (B,H,3A)

        # ---- MEMORIA: each window against every root prototype, and the null
        logp = log_softmax(p["theta"], axis=2)                              # (R,3,A)
        E = logp.reshape(R, ROOTLEN * A)
        kap = np.concatenate([np.full(O, self.kappa_plain), np.full(O, self.kappa_greek)])
        S = kap[None, :, None] * (phi @ E.T)                                # (B,H,R)
        # a root is three letters: a window that runs into the padding is no root
        haspad = (Wn[:, :, :, A - 1].sum(axis=2) > 0)                       # (B,O)
        padmask = np.concatenate([haspad, haspad], axis=1)                  # (B,H)
        S = S - 1e6 * padmask[:, :, None]
        if self.null_root:
            s_null = np.broadcast_to((kap * self.NULL_LOGP)[None, :, None], (B, H, 1))
        else:
            s_null = np.full((B, H, 1), -1e6)
        Sfull = np.concatenate([S, s_null], axis=2)                         # (B,H,R+1)
        P = softmax(Sfull, axis=2)                                          # P(r|h), r in 0..R
        LSE = logsumexp(S, axis=2)                                          # (B,H) real roots only

        # ---- RATIO: which hypothesis, which mode
        Q = p["beta"][0] * LSE + p["c"]                                     # (B,H)
        if not self.use_greek:
            Q = Q - 1e6 * np.concatenate([np.zeros(O), np.ones(O)])[None, :]
        ph = softmax(Q, axis=1)
        V = self.VMAX * np.tanh(p["thetaV"])                                # (R,d) bounded force
        T = P[:, :, :R] @ V                                                 # (B,H,d); null -> 0
        rv = np.einsum("bh,bhd->bd", ph, T)                                 # (B,d)
        # the residue: what surrounds the root, read relative to where it sits
        Res = self._residues(X)                                             # (B,H,8A)
        res = np.einsum("bh,bhj->bj", ph, Res)                              # (B,8A)
        Ml = (res @ p["Wm"].T + ph @ p["Wh"].T + p["cm"]) / self.tau_mode   # (B,M)
        pm = softmax(Ml, axis=1)
        Aop = np.tanh(p["thetaA"])                                          # (M,d,d) bounded maps
        Dm = np.einsum("bd,mkd->bmk", rv, Aop)                              # (B,M,d)
        zd = np.einsum("bm,bmd->bd", pm, Dm)

        # ---- ANIMUS: the placitum gate
        f = logsumexp(S.reshape(B, -1), axis=1)                             # (B,) any root anywhere?
        glog = g @ p["wg"] + p["ug"][0] * f + p["bg"][0]
        if self.gate_mode == "isidore" and not self.gate_closed:
            gamma = sigmoid(glog)
        elif self.gate_mode == "isidore" or self.gate_mode == "cratylus":
            gamma = np.zeros(B)
        elif self.gate_mode == "nominalist":
            gamma = np.ones(B)
        else:
            raise ValueError(self.gate_mode)

        # ---- recognition memory: cosine (over letters only, not padding) to
        #      stored keys, thresholded, with a null slot
        gl = g * self._letter_mask                                          # (B,G) pad zeroed
        gn = np.sqrt(np.sum(gl * gl, axis=1, keepdims=True))                # (B,1)
        kn = np.sqrt(np.sum(p["Km"] ** 2, axis=1))                          # (N,)
        cos = (gl @ p["Km"].T) / (gn * kn[None, :])                         # (B,N)
        cm = self.TAU_MEM * cos + p["tm"]
        cm_null = np.concatenate([cm, np.zeros((B, 1))], axis=1)
        a = softmax(cm_null, axis=1)                                        # (B,N+1)
        Vm = self.VMAX * np.tanh(p["thetaVm"])
        zm = a[:, :N] @ Vm                                                  # (B,d)

        # ---- MENS
        z = (1.0 - gamma)[:, None] * zd + gamma[:, None] * zm
        prob = sigmoid(z)
        if cache:
            self._c = dict(X=X, g=g, gl=gl, Wn=Wn, K=K, kap=kap, phi=phi, logp=logp, E=E, S=S, P=P, LSE=LSE,
                           Q=Q, ph=ph, V=V, T=T, rv=rv, Res=Res, res=res, Ml=Ml, pm=pm, Aop=Aop,
                           Dm=Dm, zd=zd, Vm=Vm, f=f,
                           glog=glog, gamma=gamma, gn=gn, kn=kn, cos=cos, a=a, zm=zm,
                           z=z, prob=prob)
        return prob

    def expose(self, X):
        """MEMORIA is written by exposure, not by gradient: the keys become the
        surfaces of the names the cell has actually read (letters only).
        What the memory *learns* afterwards is what each remembered name
        means; what it can never do is recall a name it has not read."""
        B = X.shape[0]
        g = X.reshape(B, -1) * self._letter_mask
        n = min(B, self.N)
        self.p["Km"][:n] = g[:n]
        self.exposed = n

    def _residues(self, X):
        """For every offset, the letters at fixed positions relative to the
        root window (before it and after it); out-of-range slots read as pad."""
        B = X.shape[0]
        padvec = np.zeros(A); padvec[A - 1] = 1.0
        nrel = len(self.REL)
        Res = np.zeros((B, O, nrel * A + 1))
        for o in range(O):
            for k, rel in enumerate(self.REL):
                pos = o + rel
                if 0 <= pos < L:
                    Res[:, o, k * A:(k + 1) * A] = X[:, pos, :]
                else:
                    Res[:, o, k * A:(k + 1) * A] = padvec
            # the echo cue (ex vocibus): do the two letters before the root
            # repeat the root's first two letters? counts matching letters
            if o >= 2:
                echo = (np.sum(X[:, o - 2, :A - 1] * X[:, o, :A - 1], axis=1)
                        + np.sum(X[:, o - 1, :A - 1] * X[:, o + 1, :A - 1], axis=1))
                Res[:, o, nrel * A] = echo
        return np.concatenate([Res, Res], axis=1)                           # (B,H,8A+1)

    def loss(self, X, Y):
        self.forward(X)
        c = self._c
        B, d = Y.shape
        bce = bce_with_logits(c["z"], Y).sum() / (B * d)
        gate_cost = (self.lam_gate * c["gamma"].mean()
                     if (self.gate_mode == "isidore" and not self.gate_closed) else 0.0)
        pm = c["pm"]
        ent = -np.sum(pm * np.log(pm + 1e-12)) / B
        colsum = c["K"][:A - 1, :A - 1].sum(axis=0)                         # letters only
        perm = self.LAM_PERM * np.sum((colsum - 1.0) ** 2)
        return bce + gate_cost + self.lam_ent * ent + perm

    # --------------------------------------------------------------- backward
    def backward(self, Y):
        """Analytic gradients of loss() w.r.t. every parameter, using the cache."""
        p, c = self.p, self._c
        B, d = Y.shape
        N, H, R = self.N, self.H, self.R
        g = c["g"]
        grads = {k: np.zeros_like(v) for k, v in p.items()}

        dz = (c["prob"] - Y) / (B * d)                                      # (B,d)
        gamma = c["gamma"]
        dzd = dz * (1.0 - gamma)[:, None]
        dzm = dz * gamma[:, None]
        dgamma = np.sum(dz * (c["zm"] - c["zd"]), axis=1)
        if self.gate_mode == "isidore" and not self.gate_closed:
            dgamma = dgamma + self.lam_gate / B
            dglog = dgamma * gamma * (1.0 - gamma)
        else:
            dglog = np.zeros(B)

        # ---- recognition memory
        a = c["a"]
        Vm = c["Vm"]
        dVm = a[:, :N].T @ dzm
        grads["thetaVm"] = dVm * self.VMAX * (1.0 - np.tanh(p["thetaVm"]) ** 2)
        da = np.zeros_like(a)
        da[:, :N] = dzm @ Vm.T
        dcm = a * (da - np.sum(a * da, axis=1, keepdims=True))
        dcm = dcm[:, :N]                                                    # (B,N)
        grads["tm"] = dcm.sum(axis=0)
        dcos = self.TAU_MEM * dcm
        gn, kn, cos = c["gn"], c["kn"], c["cos"]
        # d cos_bk / d K_k = g_b/(|g_b||k|) - cos_bk * k/|k|^2
        term1 = (dcos / gn).T @ c["gl"]                                     # (N,G): sum_b dcos_bk g_b/|g_b|
        term1 = term1 / kn[:, None]
        coef = np.sum(dcos * cos, axis=0)                                   # (N,)
        term2 = (coef / (kn ** 2))[:, None] * p["Km"]
        grads["Km"] = term1 - term2

        # ---- gate
        grads["wg"] = g.T @ dglog
        grads["ug"] = np.array([np.sum(dglog * c["f"])])
        grads["bg"] = np.array([np.sum(dglog)])
        df = dglog * p["ug"][0]
        S = c["S"]
        dS = df[:, None, None] * softmax(S.reshape(B, -1), axis=1).reshape(B, H, R)

        # ---- mode operators
        pm, Dm, rv = c["pm"], c["Dm"], c["rv"]
        dpm = np.einsum("bd,bmd->bm", dzd, Dm)
        dDm = pm[:, :, None] * dzd[:, None, :]                              # (B,M,d)
        Aop = c["Aop"]
        dAop = np.einsum("bmk,bd->mkd", dDm, rv)
        grads["thetaA"] = dAop * (1.0 - Aop ** 2)
        drv = np.einsum("bmk,mkd->bd", dDm, Aop)
        # entropy cost: d/dpm [-sum pm log pm] = -(log pm + 1)
        dpm = dpm + self.lam_ent * (-(np.log(pm + 1e-12) + 1.0)) / B
        dMl = pm * (dpm - np.sum(pm * dpm, axis=1, keepdims=True)) / self.tau_mode
        res, Res = c["res"], c["Res"]
        grads["Wm"] = dMl.T @ res
        grads["cm"] = dMl.sum(axis=0)
        ph = c["ph"]
        grads["Wh"] = dMl.T @ ph
        dres = dMl @ p["Wm"]                                                # (B,8A)

        # ---- hypothesis mixture
        T, P = c["T"], c["P"]
        dph = (np.einsum("bd,bhd->bh", drv, T) + dMl @ p["Wh"]
               + np.einsum("bj,bhj->bh", dres, Res))
        dT = ph[:, :, None] * drv[:, None, :]                               # (B,H,d)
        Preal = P[:, :, :R]
        V = c["V"]
        dV = Preal.reshape(-1, R).T @ dT.reshape(-1, d)
        grads["thetaV"] = dV * self.VMAX * (1.0 - np.tanh(p["thetaV"]) ** 2)
        dP = np.zeros_like(P)
        dP[:, :, :R] = dT @ V.T                                             # (B,H,R)
        dQ = ph * (dph - np.sum(ph * dph, axis=1, keepdims=True))
        grads["c"] = dQ.sum(axis=0)
        grads["beta"] = np.array([np.sum(dQ * c["LSE"])])
        dLSE = dQ * p["beta"][0]
        dS = dS + dLSE[:, :, None] * softmax(S, axis=2)                     # d LSE / dS
        dSfull = P * (dP - np.sum(P * dP, axis=2, keepdims=True))           # softmax backward
        dS = dS + dSfull[:, :, :R]

        # ---- lexicon prototypes
        phi = c["phi"]
        kap = c["kap"]
        dSk = dS * kap[None, :, None]
        dE = dSk.reshape(-1, R).T @ phi.reshape(-1, ROOTLEN * A)            # (R,3A)
        dlogp = dE.reshape(R, ROOTLEN, A)
        sm = softmax(p["theta"], axis=2)
        grads["theta"] = dlogp - sm * np.sum(dlogp, axis=2, keepdims=True)
        dphi = dSk @ c["E"]                                                 # (B,H,3A)

        # ---- cipher (only the greek half of the hypotheses touches K)
        dphi_greek = dphi[:, O:, :].reshape(B, O, ROOTLEN, A)
        Wn = c["Wn"]
        dK = np.einsum("bojx,bojy->xy", Wn, dphi_greek)
        K = c["K"]
        colsum = K[:A - 1, :A - 1].sum(axis=0)
        dK[:A - 1, :A - 1] += 2.0 * self.LAM_PERM * (colsum - 1.0)[None, :]
        grads["thetaK"] = K * (dK - np.sum(K * dK, axis=1, keepdims=True))
        return grads

    # --------------------------------------------------------------- helpers
    def predict(self, X):
        return self.forward(X, cache=False)

    def prototypes(self):
        """The strings the lexicon believes its roots are."""
        sm = softmax(self.p["theta"], axis=2)
        return ["".join(CHARS[int(np.argmax(sm[r, j]))] for j in range(ROOTLEN))
                for r in range(self.R)]

    def operator_labels(self):
        """Name each operator slot by what it has LEARNED to do to a root's
        force. The slots are latent: the cell assigns them itself, so we read
        the learned map and match it to Isidore's modes."""
        Aop = np.tanh(self.p["thetaA"])
        I = np.eye(self.d)
        targets = {
            "identity (ex causa vel derivatione)": I,
            "negation (ex contrariis)": -I,
            "rotation (ex origine)": I[[3, 0, 1, 2]],
            "pair-swap (ex vocibus)": I[[1, 0, 3, 2]],
        }
        labels = []
        for m in range(M):
            best, dist = None, 1e9
            for lab, T in targets.items():
                dd = float(np.abs(Aop[m] - T).mean())
                if dd < dist:
                    best, dist = lab, dd
            labels.append(best if dist < 0.35 else f"unassigned (mean dev {dist:.2f})")
        return labels

    def explain(self, name, lex=None):
        """The cell's derivation of one name, in Isidore's own formula
        ("X dictum a Y, quod ..."), or its admission that there is none."""
        X = encode(name)[None]
        self.forward(X)
        c = self._c
        h = int(np.argmax(c["ph"][0]))
        offset, ciphered = h % O, h >= O
        r = int(np.argmax(c["P"][0, h]))
        m = int(np.argmax(c["pm"][0]))
        gamma = float(c["gamma"][0])
        bits = "".join(str(int(v)) for v in (c["prob"][0] > 0.5))
        conf = float(np.mean(np.abs(c["prob"][0] - 0.5)) * 2)
        null = float(c["a"][0, -1])
        if gamma > 0.5 or r == self.R:
            if null > 0.5:
                why = "nomen secundum placitum, nec memoria tenet — no etymology, no recollection"
            else:
                why = "nomen secundum placitum, memoria tenet — no etymology; recalled"
            return f"{name.upper():<9} {why} (gamma={gamma:.2f}); vis {bits} conf {conf:.2f}"
        root = self.prototypes()[r]
        label = self.operator_labels()[m]
        via = "ex Graeca etymologia, " if ciphered else ""
        return (f"{name.upper():<9} dictum {via}a '{root.upper()}' @{offset}, "
                f"{label} (gamma={gamma:.2f}); vis {bits} conf {conf:.2f}")


# ==============================================================================
# 3.  THE GRADIENT CHECK  (mandatory)
# ==============================================================================

def gradient_check(gate_mode="isidore", seed=3, n=5, eps=1e-6, tol=5e-4, verbose=True):
    """Finite-difference check of every parameter tensor against the analytic
    gradient. Uses a tiny lexicon so the check is fast and the numbers are
    well conditioned; float64 throughout."""
    lex = Lexicon(n_roots=6, n_arbitrary=12, seed=seed)
    X, Y, _ = Lexicon.tensors(lex.train[:n])
    model = Origo(n_slots=8, n_mem=8, seed=seed + 1, gate_mode=gate_mode,
                  null_root=(gate_mode != "cratylus"))
    # push parameters away from their tidy initial values so that no gradient
    # component is trivially zero
    rng = np.random.RandomState(seed + 2)
    for k in model.p:
        model.p[k] = model.p[k] + 0.3 * rng.randn(*model.p[k].shape)
    model.loss(X, Y)
    grads = model.backward(Y)
    worst = 0.0
    report = []
    for k, v in model.p.items():
        idx = list(np.ndindex(v.shape))
        rng.shuffle(idx)
        for ij in idx[:min(len(idx), 6)]:
            old = v[ij]
            v[ij] = old + eps
            lp = model.loss(X, Y)
            v[ij] = old - eps
            lm = model.loss(X, Y)
            v[ij] = old
            num = (lp - lm) / (2 * eps)
            ana = grads[k][ij]
            rel = abs(num - ana) / max(1e-6, abs(num) + abs(ana))
            worst = max(worst, rel)
            report.append((k, ij, num, ana, rel))
    if verbose:
        bad = [r for r in report if r[4] > tol]
        print(f"  gradient check [{gate_mode}]: {len(report)} components, "
              f"worst relative error {worst:.2e}, tolerance {tol:.0e} -> "
              f"{'PASS' if worst < tol else 'FAIL'}")
        for r in bad[:5]:
            print("    mismatch", r)
    return worst < tol


# ==============================================================================
# 4.  TRAINING  (SPIRITUS — what animates the cell)
# ==============================================================================

class Adam:
    def __init__(self, params, lr=0.02, b1=0.9, b2=0.999, eps=1e-8, wd=0.0):
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, b1, b2, eps, wd
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for k in params:
            gk = grads[k] + self.wd * params[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * gk
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * gk * gk
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)


def accuracy(model, X, Y):
    prob = model.predict(X)
    return float(np.mean((prob > 0.5) == (Y > 0.5)))


def train(model, Xtr, Ytr, MODEtr, Xte, Yte, MODEte, epochs=400, warmup=150,
          batch=48, lr=0.02, seed=0, log_every=50, quiet=False,
          kappa0=0.3, kappa1=0.3, lam_gate0=0.4, iia_frac=0.4):
    """A real training loop in two phases, which are Isidore's own order:

      PHASE I  — GRAMMATICA (warmup epochs). Only the bare roots are shown:
                 the words of the language, each with its force. The lexicon
                 is annealed from soft to sharp (kappa0 -> KAPPA_FINAL): early
                 on every window faintly resembles every root, which is what
                 lets a root be discovered at all; by the end a prototype
                 matches one string. Learning by exposure, then a hardened
                 lexicon. The gate is closed (a bare word has no derivation).

      PHASE II — ORIGINES (the remaining epochs). The full stream: derived
                 names, Greek transliterations, arbitrary names. The lexicon
                 is briefly softened (kappa1) so that the cipher can be found
                 by gradient, then re-hardened; the placitum gate is expensive
                 at first (lam_gate0) and cheap at the end: SEEK THE ORIGIN
                 FIRST, admit convention only once derivation has had its
                 chance.

    Minibatch Adam throughout. The unseen-derivation accuracy is tracked so
    that we can report how *quickly* each mind comes to understand names it
    has never met ("citius vim eius intellegis")."""
    rng = np.random.RandomState(seed)
    opt = Adam(model.p, lr=lr)
    simplex = np.where(MODEtr == SIMPLEX)[0]
    n_all = Xtr.shape[0]
    derived_te = MODEte != PLACITUM
    history = []
    first_90 = None
    t0 = time.time()
    lam_final = model.lam_gate
    for ep in range(1, epochs + 1):
        if ep <= warmup:                                   # ---- PHASE I
            s = (ep - 1) / max(1.0, warmup - 1)
            model.kappa_plain = kappa0 + (model.KAPPA_FINAL - kappa0) * s
            model.kappa_greek = model.kappa_plain
            model.use_greek = False
            model.gate_closed = True                       # a bare word is derived from itself
            model.lam_gate = lam_gate0
            model.lam_ent = 0.0
            pool = simplex
        else:                                              # ---- PHASE II
            n2 = epochs - warmup
            e2 = ep - warmup                               # 1..n2
            na = int(round(iia_frac * n2))                 # IIa: derivation only
            model.kappa_plain = model.KAPPA_FINAL
            model.use_greek = True
            pool = np.arange(n_all)
            if e2 <= na:                                   # ---- IIa  origines, gate shut
                s = (e2 - 1) / max(1.0, na - 1)
                model.kappa_greek = kappa1 + (model.KAPPA_FINAL - kappa1) * s
                model.gate_closed = True
                model.lam_gate = lam_gate0
                model.lam_ent = 0.0
            else:                                          # ---- IIb  placitum admitted
                if e2 == na + 1:
                    model.expose(Xtr)                      # the memory reads the stream
                s = (e2 - na - 1) / max(1.0, n2 - na - 1)
                model.kappa_greek = model.KAPPA_FINAL
                model.gate_closed = False
                model.lam_gate = lam_gate0 + (lam_final - lam_gate0) * s
                model.lam_ent = model.LAM_ENT * s
        perm = rng.permutation(pool)
        bsz = 8 if ep <= warmup else batch                 # small batches on the small vocabulary
        tot = 0.0
        for i in range(0, len(perm), bsz):
            idx = perm[i:i + bsz]
            tot += model.loss(Xtr[idx], Ytr[idx]) * len(idx)
            grads = model.backward(Ytr[idx])
            if ep <= warmup:
                # a bare word has no residue to read and nothing to decide:
                # the mode reader, the operators, the cipher and the gate
                # all wait for phase II
                for k in ("Wm", "Wh", "cm", "thetaA", "thetaK", "wg", "ug", "bg"):
                    grads[k][...] = 0.0
            else:
                # the native lexicon, once acquired, is held: Latin words do
                # not change their letters because one is reading Greek
                grads["theta"][...] = 0.0
            opt.step(model.p, grads)
        tr_acc = accuracy(model, Xtr, Ytr)
        te_der = accuracy(model, Xte[derived_te], Yte[derived_te])
        history.append((ep, tot / len(perm), tr_acc, te_der))
        if ep > warmup and first_90 is None and te_der >= 0.90:
            first_90 = ep - warmup
        if not quiet and (ep % log_every == 0 or ep == 1 or ep == warmup):
            phase = ("I  grammatica" if ep <= warmup else
                     "IIa origines " if model.gate_closed else "IIb placitum ")
            print(f"    epoch {ep:4d} [{phase}] loss {tot / len(perm):.4f}  "
                  f"train acc {tr_acc:.3f}  unseen-derivation acc {te_der:.3f}")
    model.kappa_plain = model.kappa_greek = model.KAPPA_FINAL
    model.use_greek = True
    model.gate_closed = False
    model.lam_ent = model.LAM_ENT
    model.lam_gate = lam_final
    if not quiet:
        print(f"    trained in {time.time() - t0:.1f}s")
    return history, first_90


# ==============================================================================
# 5.  EVALUATION  —  what each mind knows, and whether it knows that it doesn't
# ==============================================================================

def evaluate(model, lex, Xte, Yte, MODEte):
    prob = model.predict(Xte)
    model.forward(Xte)
    gamma = model._c["gamma"]
    der = MODEte != PLACITUM
    arb = ~der
    conf = np.mean(np.abs(prob - 0.5) * 2, axis=1)         # 0 = shrug, 1 = certain
    out = dict(
        acc_unseen_derived=float(np.mean((prob[der] > 0.5) == (Yte[der] > 0.5))),
        acc_unseen_arbitrary=float(np.mean((prob[arb] > 0.5) == (Yte[arb] > 0.5))),
        conf_unseen_derived=float(conf[der].mean()),
        conf_unseen_arbitrary=float(conf[arb].mean()),
        gamma_derived=float(gamma[der].mean()),
        gamma_arbitrary=float(gamma[arb].mean()),
    )
    # per-mode accuracy on unseen derivations
    per_mode = {}
    for mi, mname in enumerate(MODES):
        sel = MODEte == mi
        if sel.any():
            per_mode[mname] = float(np.mean((prob[sel] > 0.5) == (Yte[sel] > 0.5)))
    out["per_mode"] = per_mode
    return out


def alignment_accuracy(model, items):
    """Does the cell put the root where the ancients put it?"""
    X, _, _ = Lexicon.tensors(items)
    model.forward(X)
    ph = model._c["ph"]
    hits = 0
    for i, d in enumerate(items):
        h = int(np.argmax(ph[i]))
        if (h % O) == d["offset"] and (h >= O) == d["ciphered"]:
            hits += 1
    return hits / len(items)


def root_identification_accuracy(model, items, lex):
    X, _, _ = Lexicon.tensors(items)
    model.forward(X)
    ph, P = model._c["ph"], model._c["P"]
    protos = model.prototypes()
    hits = 0
    for i, d in enumerate(items):
        h = int(np.argmax(ph[i]))
        r = int(np.argmax(P[i, h]))
        hits += (r < model.R and protos[r] == d["root"])
    return hits / len(items)


def cipher_recovery(model, lex):
    """How many letters of the transliteration has the cell reconstructed?
    K maps a ciphered letter's one-hot onto the plain alphabet; the true map
    is the inverse of the cipher."""
    K = softmax(model.p["thetaK"], axis=1)
    hits = 0
    for ch in ALPHABET:
        i = CHARS.index(ch)                       # ciphered letter
        j = int(np.argmax(K[i, :len(ALPHABET)]))  # what the cell reads it as
        hits += (CHARS[j] == lex.decipher[ch])
    return hits / len(ALPHABET)


# ==============================================================================
# 6.  SELF-TESTS  —  the structural invariants, not just the loss
# ==============================================================================
#
# The tests come in two kinds. STRUCTURAL tests hold for any parameters: they
# are what the shape of the cell guarantees before training. LEARNED tests
# hold for the trained Isidore mind: they are what the chapter claims.

def _tiny():
    lex = Lexicon(n_roots=6, n_arbitrary=12, seed=11)
    X, Y, MO = Lexicon.tensors(lex.train)
    return lex, X, Y, MO


def test_dataset_integrity():
    """Every derived name carries its root where the generator says, read
    plain or through the cipher; no arbitrary name contains a root anywhere;
    every mode's meaning-map is a signed permutation."""
    lex = Lexicon(seed=5)
    for d in lex.derived:
        w = d["name"][d["offset"]:d["offset"] + ROOTLEN]
        if d["ciphered"]:
            w = "".join(lex.decipher[c] for c in w)
        assert w == d["root"], (d, w)
        assert np.array_equal(d["vis"], apply_mode(lex.root_vis[d["root"]], MODES[d["mode"]]))
    for d in lex.arbitrary:
        assert not lex._contains_root(d["name"]), d
    for mode in MODES:
        # each meaning-map is a bijection on the 16 possible forces
        images = {tuple(apply_mode(np.array(v), mode)) for v in np.ndindex(2, 2, 2, 2)}
        assert len(images) == 16, mode
    # held-out pairs are whole: no (root, mode) appears on both sides
    tr = {(d["root"], d["mode"]) for d in lex.train if d["root"]}
    te = {(d["root"], d["mode"]) for d in lex.test if d["root"]}
    assert not (tr & te)
    # every test root was seen in training under some other mode
    assert all(any(r == d["root"] for d in lex.train) for r, _ in te)
    return True


def test_shapes():
    lex, X, Y, MO = _tiny()
    m = Origo(n_slots=8, n_mem=16, seed=2)
    prob = m.forward(X)
    c = m._c
    assert prob.shape == (X.shape[0], D)
    assert c["phi"].shape == (X.shape[0], 2 * O, ROOTLEN * A)
    assert c["P"].shape == (X.shape[0], 2 * O, 9)          # 8 slots + null
    assert np.allclose(c["P"].sum(axis=2), 1.0)
    assert np.allclose(c["ph"].sum(axis=1), 1.0)
    assert np.allclose(c["pm"].sum(axis=1), 1.0)
    assert np.allclose(c["a"].sum(axis=1), 1.0)
    K = softmax(m.p["thetaK"], axis=1)
    assert np.allclose(K.sum(axis=1), 1.0)                  # a cipher is a letter map
    return True


def test_gradient_check():
    return all(gradient_check(g, verbose=False) for g in ("isidore", "cratylus", "nominalist"))


def test_pad_window_is_never_a_root():
    """A window that runs into the padding gets no root mass, for any weights."""
    lex, X, Y, MO = _tiny()
    m = Origo(n_slots=8, n_mem=16, seed=3)
    m.p["theta"][...] = 5.0 * np.random.RandomState(0).randn(*m.p["theta"].shape)
    m.forward(X)
    c = m._c
    haspad = c["Wn"][:, :, :, A - 1].sum(axis=2) > 0
    for b in range(X.shape[0]):
        for o in range(O):
            if haspad[b, o]:
                assert c["P"][b, o, -1] > 0.999 and c["P"][b, O + o, -1] > 0.999
    return True


def test_null_root_yields_no_force():
    """When every window falls to the null root, the derived force is zero
    and the derivation path answers exactly 0.5 — the shape of 'non
    reperiuntur', whatever the weights."""
    lex, X, Y, MO = _tiny()
    m = Origo(n_slots=8, n_mem=16, seed=4, gate_mode="cratylus")
    # make the prototypes match nothing that occurs
    m.p["theta"][...] = -20.0
    m.p["theta"][:, :, A - 1] = 20.0                        # prototypes spell '___'
    m.p["thetaV"][...] = 3.0
    prob = m.forward(X)
    assert np.all(np.abs(m._c["rv"]) < 1e-6)
    assert np.allclose(prob, 0.5, atol=1e-6)
    return True


def test_force_is_bounded():
    """No parameter setting can push a root's force or an operator entry
    beyond its bound, so a faint match cannot be inflated into a meaning."""
    m = Origo(n_slots=8, n_mem=16, seed=5)
    m.p["thetaV"][...] = 1e3
    m.p["thetaA"][...] = -1e3
    assert np.all(np.abs(Origo.VMAX * np.tanh(m.p["thetaV"])) <= Origo.VMAX)
    assert np.all(np.abs(np.tanh(m.p["thetaA"])) <= 1.0)
    lex, X, Y, MO = _tiny()
    m.forward(X)
    assert np.all(np.abs(m._c["zd"]) <= Origo.VMAX * D + 1e-9)
    return True


def test_memory_recalls_only_what_it_read():
    """An exposed memory recalls a read name with near-certainty and gives
    an unread name to the null slot."""
    lex, X, Y, MO = _tiny()
    m = Origo(n_slots=8, n_mem=128, seed=6, gate_mode="nominalist")
    m.expose(X)
    m.forward(X)
    a = m._c["a"]
    gl = m._c["gl"]
    for b in range(X.shape[0]):
        # recall lands on a slot whose key IS the name (duplicates and
        # near-homonyms such as 'ababc'/'ababco' may split the mass)
        k = int(np.argmax(a[b, :m.N]))
        assert np.allclose(m.p["Km"][k], gl[b]), (b, k)
        assert a[b, -1] < 0.05
    Xu, _, _ = Lexicon.tensors(lex.test)
    m.forward(Xu)
    au = m._c["a"]
    # arbitrary unseen names: nothing to recall
    arb = np.array([d["mode"] == PLACITUM for d in lex.test])
    assert np.all(au[arb, -1] > 0.9), au[arb, -1]
    return True


def test_training_reduces_loss(models):
    h = models["isidore"]["history"]
    return h[0][1] > 3 * h[-1][1]


def test_lexicon_acquired(models, lex):
    """After grammatica, the prototypes ARE the roots."""
    m = models["isidore"]["model"]
    protos = set(p for p in m.prototypes() if p in lex.root_index)
    return len(protos) == len(lex.roots)


def test_alignment(models, lex):
    """The cell puts the root where the ancients put it, plain or ciphered."""
    items = [d for d in lex.test if d["mode"] != PLACITUM]
    return alignment_accuracy(models["isidore"]["model"], items) >= 0.95


def test_cipher_recovered(models, lex):
    """Ex Graeca: the transliteration has been reconstructed letter by letter."""
    return cipher_recovery(models["isidore"]["model"], lex) >= 0.9


def test_operators_recovered(models):
    """The learned operators are the four signed permutations the ancients used."""
    labels = models["isidore"]["model"].operator_labels()
    kinds = {l.split(" (")[0] for l in labels if not l.startswith("unassigned")}
    return kinds == {"identity", "negation", "rotation", "pair-swap"}


def test_differentia(models, lex):
    """Differentiae: a minimal pair — the same root ex causa and ex contrariis
    — must come out with every bit of force reversed."""
    m = models["isidore"]["model"]
    flips = 0
    for r in lex.roots:
        pa = m.predict(encode(r + "ns")[None])[0] > 0.5
        pb = m.predict(encode("in" + r)[None])[0] > 0.5
        flips += int(np.all(pa != pb))
    return flips >= 0.95 * len(lex.roots)


def test_synonyma(models, lex):
    """Synonyma: names that mean the same thing by different derivations
    are given the same force — 'homo' and 'inhumanus' are not, 'rex' and
    'rector' are."""
    m = models["isidore"]["model"]
    agree, total = 0, 0
    for r in lex.roots:
        forms = [r, r + "ns", r + "tor", r + "nsia", r + "toria",
                 "".join(lex.cipher[c] for c in r) + "is"]
        ps = np.stack([m.predict(encode(f)[None])[0] for f in forms]) > 0.5
        agree += int(np.all(ps == ps[0]))
        total += 1
    return agree >= 0.95 * total


def test_gate_discriminates(models):
    e = models["isidore"]["eval"]
    return e["gamma_arbitrary"] > 0.8 and e["gamma_derived"] < 0.2


def test_abstention(models):
    """On unseen arbitrary names Isidore shrugs; Cratylus is confidently wrong."""
    return (models["isidore"]["eval"]["conf_unseen_arbitrary"] < 0.15 and
            models["cratylus"]["eval"]["conf_unseen_arbitrary"] > 0.5)


def test_generalisation(models):
    e = {k: v["eval"]["acc_unseen_derived"] for k, v in models.items()}
    return e["isidore"] >= 0.9 and e["isidore"] > e["cratylus"] + 0.05 and e["isidore"] > e["nominalist"] + 0.1


# ==============================================================================
# 7.  MAIN
# ==============================================================================

def main():
    np.set_printoptions(precision=3, suppress=True)
    rule()
    print(" ORIGO — the derivation cell · Chapter 0178 · Isidore of Seville")
    print(" Encyclopedia of Lost Minds: Echoes on AI")
    rule()

    # ---- structural tests first: these hold before any learning
    print("\n[1] Structural self-tests")
    structural = [test_dataset_integrity, test_shapes, test_pad_window_is_never_a_root,
                  test_null_root_yields_no_force, test_force_is_bounded,
                  test_memory_recalls_only_what_it_read]
    ok = True
    for t in structural:
        r = t()
        ok &= bool(r)
        print(f"  {'PASS' if r else 'FAIL'}  {t.__name__}")

    print("\n[2] Finite-difference gradient check (all parameters, three minds)")
    gc = all(gradient_check(g) for g in ("isidore", "cratylus", "nominalist"))
    ok &= gc

    # ---- the lexicon
    print("\n[3] The lexicon")
    lex = Lexicon(seed=0)
    lex.describe()
    Xtr, Ytr, MOtr = Lexicon.tensors(lex.train)
    Xte, Yte, MOte = Lexicon.tensors(lex.test)

    # ---- three minds
    print("\n[4] Training three minds on the same stream")
    models = {}
    for gm in ("isidore", "cratylus", "nominalist"):
        print(f"\n  --- {gm.upper()} ---")
        m = Origo(n_slots=48, gate_mode=gm, seed=1, null_root=(gm != "cratylus"))
        hist, first90 = train(m, Xtr, Ytr, MOtr, Xte, Yte, MOte, epochs=800, warmup=300,
                              log_every=100)
        ev = evaluate(m, lex, Xte, Yte, MOte)
        items = [d for d in lex.test if d["mode"] != PLACITUM]
        ev["alignment"] = alignment_accuracy(m, items)
        ev["root_id"] = root_identification_accuracy(m, items, lex)
        ev["cipher"] = cipher_recovery(m, lex)
        ev["first_90"] = first90
        models[gm] = dict(model=m, history=hist, eval=ev)

    print("\n[5] Results on the held-out names")
    rule("-")
    hdr = f"  {'':<22}{'ISIDORE':>12}{'CRATYLUS':>12}{'NOMINALIST':>12}"
    print(hdr)
    rule("-")
    rows = [
        ("unseen derivations acc", "acc_unseen_derived"),
        ("unseen arbitrary acc", "acc_unseen_arbitrary"),
        ("confidence, derived", "conf_unseen_derived"),
        ("confidence, arbitrary", "conf_unseen_arbitrary"),
        ("gate: derived names", "gamma_derived"),
        ("gate: arbitrary names", "gamma_arbitrary"),
        ("root found at offset", "alignment"),
        ("root identified", "root_id"),
        ("cipher recovered", "cipher"),
    ]
    for label, key in rows:
        vals = [models[g]["eval"][key] for g in ("isidore", "cratylus", "nominalist")]
        print(f"  {label:<22}" + "".join(f"{v:>12.3f}" for v in vals))
    vals = [models[g]["eval"]["first_90"] for g in ("isidore", "cratylus", "nominalist")]
    print(f"  {'epochs to 90% unseen':<22}" + "".join(f"{str(v) if v else 'never':>12}" for v in vals))
    rule("-")
    print("  per mode, unseen derivations:")
    for mode in MODES[1:]:
        vals = [models[g]["eval"]["per_mode"].get(mode, float('nan')) for g in ("isidore", "cratylus", "nominalist")]
        print(f"    {mode:<20}" + "".join(f"{v:>12.3f}" for v in vals))

    print("\n[6] What the operators learned (ISIDORE)")
    m = models["isidore"]["model"]
    for i, lab in enumerate(m.operator_labels()):
        print(f"  slot {i}: {lab}")
    print(f"  cipher: " + ", ".join(
        f"{ch}->{CHARS[int(np.argmax(softmax(m.p['thetaK'], axis=1)[CHARS.index(ch), :A - 1]))]}"
        for ch in ALPHABET))

    print("\n[7] The ledger — held-out names, in Isidore's formula (ISIDORE)")
    shown = 0
    for d in lex.test:
        if shown >= 14:
            break
        truth = MODES[d["mode"]] if d["mode"] < M else "placitum"
        print(f"  {m.explain(d['name'])}   | truth: {''.join(map(str, d['vis']))} {truth}")
        shown += 1
    print("\n  the same names, as CRATYLUS reads them:")
    mc = models["cratylus"]["model"]
    for d in lex.test[:5]:
        print(f"  {mc.explain(d['name'])}")

    print("\n[8] Learned self-tests (the chapter's claims)")
    learned = [
        ("training reduces loss", test_training_reduces_loss(models)),
        ("lexicon acquired (prototypes are the roots)", test_lexicon_acquired(models, lex)),
        ("alignment: root found where placed", test_alignment(models, lex)),
        ("ex Graeca: cipher recovered", test_cipher_recovered(models, lex)),
        ("operators are the four signed permutations", test_operators_recovered(models)),
        ("differentia: minimal pair flips every bit", test_differentia(models, lex)),
        ("synonyma: same force by different derivations", test_synonyma(models, lex)),
        ("gate discriminates derivable from given", test_gate_discriminates(models)),
        ("abstention: Isidore shrugs, Cratylus invents", test_abstention(models)),
        ("systematic generalisation beats both ablations", test_generalisation(models)),
    ]
    for label, r in learned:
        ok &= bool(r)
        print(f"  {'PASS' if r else 'FAIL'}  {label}")

    rule()
    print(f" ALL TESTS {'PASSED' if ok else 'FAILED'}")
    rule()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
