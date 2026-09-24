#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
CHAPTER 0231 - SUGAWARA NO MICHIZANE (845-903)
THE KAERITEN ENGINE: a transposition-first cognitive architecture
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 12 Minds 221 - 240 Available on Amazon https://www.amazon.com/dp/B0HKVBKNTT
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0231_sugawara_no_michizane_845 - Sugawara no Michizane (845-903)
================================================================================  

WHY THIS ARCHITECTURE, AND NOT ANOTHER

Michizane's whole working life was spent performing one operation, tens of
thousands of times, at three different scales:

  * At the scale of the sentence, he read Literary Sinitic -- a language nobody
    in Heian Japan spoke -- by the technique called KANBUN KUNDOKU. The source
    text was never rewritten. Instead the reader added marks in the margins and
    between the columns, and those marks told the eye to jump BACKWARD, to
    return, to take position 5 before position 3. Philologists describe the
    procedure as three separable layers (Frellesvig 2010, p.259; Denecke 2014,
    p.47): TRANSLATION (choose a native lexeme for each graph), TRANSPOSITION
    (reorder the graphs into native syntax), INTERPOLATION (insert the
    grammatical particles and inflections the source does not contain).

  * At the scale of the corpus, he compiled the RUIJU KOKUSHI (892), 200
    fascicles that took the Six National Histories and re-sorted them from
    chronological order into eighteen topical categories. The editorial policy
    was explicitly conservative: keep the wording of the sources, change only
    the order of access.

  * At the scale of the state, he watched the AKO INCIDENT (887-888) freeze the
    government of Japan because a single imported term in a single imperial
    edict was glossed confidently by someone who had not checked which precedent
    governed it. The regent walked out. Ministries stopped. Michizane's
    intervention was a letter arguing the dispute back down.

So: a mind whose native act is REORDERING A SOURCE IT REFUSES TO ALTER, and
whose formative catastrophe was a confident gloss of a foreign token.

This file builds that mind. It is deliberately NOT a Transformer. Attention
mixes values into a soft blend and throws the alignment away; the Kaeriten
Engine's central object is a near-PERMUTATION -- a doubly stochastic matrix
annealed toward a hard one-to-one reading path -- because for Michizane the
alignment IS the artefact. The return-marks are the reading. A system that
cannot show you its return-marks has not read; it has guessed.

FOUR ORGANS
  1. SCAN         a bidirectional recurrent pass over the graphs (no attention).
                  This is the eye going down the column, twice.
  2. KAERITEN     pairwise scores -> Sinkhorn normalisation -> soft permutation
                  P. Row i of P says: "to read position i, return to source
                  position j." This is the return-mark, made differentiable.
  3. KUNDOKU      per-slot lexeme choice from the reordered states (translation)
                  plus a particle head (interpolation) that supplies grammar the
                  source never had.
  4. AKO GATE     an abstention channel. Some graphs are genuinely ambiguous
                  with no cue in the sentence. The correct output is not a
                  guess, it is a request to consult the archive. The model is
                  trained to emit ABSTAIN on exactly those, and penalised for
                  abstaining anywhere else.

Two auxiliary procedures close the loop with the historical record:
  RUIJU RE-INDEX      cluster the learned graph embeddings into topical
                      categories and emit a classified index, then PROVE the
                      re-sort was lossless (the multiset of tokens is identical
                      before and after). Reordering must never emend.
  POSTHUMOUS AUDIT    after training, re-score every decision and surface the
                      confident errors. Michizane was rehabilitated in 923,
                      twenty years dead; the model gets its audit sooner.

CONVENTIONS
  Pure NumPy. Every parameter has a hand-written analytic gradient, and the file
  refuses to train until a finite-difference check passes. Run it directly:

      python3 chapter_0231_sugawara_no_Michizane_845.py

================================================================================
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

RNG_SEED = 845  # the year of his birth, used as the global seed

# =============================================================================
# SECTION 1 - THE CORPUS
#
# A small synthetic Literary Sinitic grammar and its kundoku reading. Synthetic
# because we need ground truth for the permutation, which no real annotated
# corpus of this size supplies in machine-readable form; principled because the
# reordering rule below is a fair schematisation of what kaeriten actually do:
# the verb travels to the end, its auxiliaries travel past it, the preposition
# is read AFTER its own object, and the subject alone stays put.
# =============================================================================

# Source graphs. Each entry: graph -> (romanised gloss, syntactic role)
# Roles: S subject, V verb, O object, A adverb, L locative noun,
#        P preposition, N negation, M modal, C register cue.
GRAPHS: Dict[str, Tuple[str, str]] = {
    # subjects
    "\u738b": ("ou", "S"), "\u81e3": ("omi", "S"), "\u6c11": ("tami", "S"),
    "\u5e2b": ("shi", "S"), "\u5ba2": ("kyaku", "S"),
    # verbs
    "\u898b": ("miru", "V"), "\u805e": ("kiku", "V"), "\u4f5c": ("tsukuru", "V"),
    "\u8b80": ("yomu", "V"), "\u554f": ("tou", "V"), "\u737b": ("tatematsuru", "V"),
    # objects
    "\u66f8": ("fumi", "O"), "\u8a69": ("uta", "O"), "\u6885": ("ume", "O"),
    "\u6708": ("tsuki", "O"), "\u793c": ("rei", "O"),
    # adverbs
    "\u4ea6": ("mata", "A"), "\u65e2": ("sudeni", "A"), "\u5e38": ("tsuneni", "A"),
    # locatives
    "\u5c71": ("yama", "L"), "\u4eac": ("miyako", "L"), "\u6d77": ("umi", "L"),
    "\u5e9c": ("fu", "L"),
    # function graphs
    "\u65bc": ("oite", "P"), "\u4e0d": ("zu", "N"), "\u53ef": ("beshi", "M"),
    # register cues -- the graph that tells you WHICH archive governs the reading
    "\u8a54": ("mikotonori", "C"), "\u8ad6": ("agetsurau", "C"),
}

# THE AKO CLASS.
#
# These three graphs are the heart of the model. Each has two defensible
# readings, and nothing inside the graph itself decides between them. What
# decides is which body of precedent you are reading under -- whether the
# document is an imperial instrument (\u8a54) or a scholarly disquisition (\u8ad6).
# If the sentence carries neither cue, there is no fact of the matter available
# to the reader, and the correct output is not a guess. It is a request.
#
# This is the Ako Incident, compiled down to a training signal. In 887 an
# imperial edict named Fujiwara no Mototsune "ako", a Shang-dynasty honorific.
# One scholar read it as a working office; another read it as an empty dignity.
# Nobody went back to check which precedent governed. The regent withdrew, the
# ministries stopped, and the government of Japan sat idle for months over the
# gloss of a single foreign word.
AKO_READINGS: Dict[str, Dict[str, str]] = {
    "\u8861": {"\u8a54": "kou-no-tsukasa", "\u8ad6": "hakaru"},      # office with duties / to weigh
    "\u76f8": {"\u8a54": "shou-no-kimi",   "\u8ad6": "ai"},          # chief minister / mutually
    "\u8207": {"\u8a54": "atauru",         "\u8ad6": "tomoni"},      # to grant / together with
}
for _g, _rd in AKO_READINGS.items():
    GRAPHS[_g] = (_rd["\u8a54"], "O")          # placeholder gloss; role is object
AKO_SET = set(AKO_READINGS)

ABSTAIN = "<consult-the-archive>"

# Particles and inflections supplied by the reader. None of these are present
# anywhere in the source text; the reader interpolates them from grammar he
# carries in his head. The verb ending is the interesting one: it depends on
# what follows the verb IN THE READING ORDER, which does not exist until the
# transposition has been performed. Interpolation is downstream of transposition
# -- the architecture has to discover that, and so did every Heian schoolboy.
PARTICLES = ["", "\u306f", "\u306b", "\u3092", "\u30eb", "\u30ea"]

SRC_VOCAB = sorted(GRAPHS.keys())
SRC_IDX = {g: i for i, g in enumerate(SRC_VOCAB)}
TGT_VOCAB = sorted(
    {v[0] for v in GRAPHS.values()}
    | {r for rd in AKO_READINGS.values() for r in rd.values()}
    | {ABSTAIN}
)
TGT_IDX = {w: i for i, w in enumerate(TGT_VOCAB)}
PAR_IDX = {p: i for i, p in enumerate(PARTICLES)}

BY_ROLE: Dict[str, List[str]] = {}
for _g, (_gl, _r) in GRAPHS.items():
    if _g in AKO_SET:
        continue
    BY_ROLE.setdefault(_r, []).append(_g)
for _r in BY_ROLE:
    BY_ROLE[_r].sort()


@dataclass
class Sample:
    """One sentence, in both orders, with its reading path."""
    graphs: List[str]          # source order, as written
    src: np.ndarray            # (N,) source vocabulary indices
    perm: np.ndarray           # (N,) perm[i] = source slot read at reading step i
    lex: np.ndarray            # (N,) target lexeme at each reading step
    par: np.ndarray            # (N,) particle appended at each reading step
    abstain: np.ndarray        # (N,) bool, True where ABSTAIN is correct
    ako: np.ndarray            # (N,) bool, True at any ako-class graph, cued or not
    roles: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.src)


def make_sample(rng: np.random.Generator, long_form: bool = False) -> Sample:
    """
    Build one sentence.

    SOURCE (Literary Sinitic) slot order:
        [C] S [A] [N] [M] V [O] [P L] [P L]

    READING (kundoku) order:
        [C] , S , (L P)... , (O) , (A) , V , (M) , (N)

    Note what this rule does. The preposition is read AFTER the noun it governs
    -- that is the classic re-ten, the mark that says "back up one". The
    auxiliaries \u4e0d and \u53ef sit before the verb as written and after it as read,
    because Japanese hangs its auxiliaries off the tail of the verb. The subject
    alone does not move. Nothing is added to or removed from the source; only
    the path through it changes.

    long_form=True forces the rarely co-occurring options on at once and allows
    a SECOND prepositional phrase. Long-form sentences are never used in
    training. They exist to ask whether the return-marks the model has learned
    are a rule or a lookup table.
    """
    if long_form:
        has_a, has_n, has_m, has_o = True, rng.random() < 0.5, rng.random() < 0.5, True
        if has_n and has_m:
            has_m = False
        n_pp = 2
        has_cue = rng.random() < 0.6
    else:
        has_a = rng.random() < 0.45
        has_n = rng.random() < 0.30
        has_m = (rng.random() < 0.30) and not has_n
        has_o = rng.random() < 0.85
        n_pp = 1 if rng.random() < 0.45 else 0
        has_cue = rng.random() < 0.35

    use_ako = has_o and (rng.random() < (0.30 if has_cue else 0.16))
    cue = rng.choice(BY_ROLE["C"]) if has_cue else None

    slots: List[Tuple[str, str]] = []           # (graph, role) in SOURCE order
    if has_cue:
        slots.append((cue, "C"))
    slots.append((rng.choice(BY_ROLE["S"]), "S"))
    if has_a:
        slots.append((rng.choice(BY_ROLE["A"]), "A"))
    if has_n:
        slots.append(("\u4e0d", "N"))
    if has_m:
        slots.append(("\u53ef", "M"))
    slots.append((rng.choice(BY_ROLE["V"]), "V"))
    if has_o:
        slots.append(((rng.choice(sorted(AKO_SET)) if use_ako
                       else rng.choice(BY_ROLE["O"])), "O"))
    pp_pos: List[Tuple[int, int]] = []
    for _ in range(n_pp):
        slots.append(("\u65bc", "P")); p_at = len(slots) - 1
        slots.append((rng.choice(BY_ROLE["L"]), "L")); l_at = len(slots) - 1
        pp_pos.append((l_at, p_at))

    pos_of: Dict[str, int] = {}
    for i, (_g, r) in enumerate(slots):
        if r not in ("P", "L"):
            pos_of[r] = i

    order: List[int] = []
    if has_cue:
        order.append(pos_of["C"])
    order.append(pos_of["S"])
    for l_at, p_at in pp_pos:
        order += [l_at, p_at]                    # noun first, THEN \u65bc
    if has_o:
        order.append(pos_of["O"])
    if has_a:
        order.append(pos_of["A"])
    order.append(pos_of["V"])
    if has_m:
        order.append(pos_of["M"])
    if has_n:
        order.append(pos_of["N"])

    graphs = [g for g, _ in slots]
    src = np.array([SRC_IDX[g] for g in graphs], dtype=np.int64)
    perm = np.array(order, dtype=np.int64)

    # Verb inflection depends on what comes NEXT in the reading, not in the text.
    verb_ending = "\u30eb" if has_m else ("" if has_n else "\u30ea")
    part_for = {"C": "", "S": "\u306f", "L": "\u306b", "P": "", "O": "\u3092",
                "A": "", "V": verb_ending, "M": "", "N": ""}

    lex, par, abst, akom = [], [], [], []
    for j in order:
        g, r = slots[j]
        akom.append(g in AKO_SET)
        if g in AKO_SET:
            if has_cue:
                lex.append(TGT_IDX[AKO_READINGS[g][cue]]); abst.append(False)
            else:
                lex.append(TGT_IDX[ABSTAIN]); abst.append(True)
        else:
            lex.append(TGT_IDX[GRAPHS[g][0]]); abst.append(False)
        par.append(PAR_IDX[part_for[r]])

    return Sample(
        graphs=graphs, src=src, perm=perm,
        lex=np.array(lex, dtype=np.int64),
        par=np.array(par, dtype=np.int64),
        abstain=np.array(abst, dtype=bool),
        ako=np.array(akom, dtype=bool),
        roles=[r for _g, r in slots],
    )


def make_corpus(n: int, seed: int, long_form: bool = False) -> List[Sample]:
    rng = np.random.default_rng(seed)
    return [make_sample(rng, long_form=long_form) for _ in range(n)]


# =============================================================================
# SECTION 2 - DIFFERENTIABLE PRIMITIVES
#
# Written out longhand. Each has a forward that caches what its backward needs.
# =============================================================================

def log_softmax(x: np.ndarray, axis: int) -> np.ndarray:
    m = x.max(axis=axis, keepdims=True)
    z = x - m
    return z - np.log(np.exp(z).sum(axis=axis, keepdims=True))


def softmax(x: np.ndarray, axis: int) -> np.ndarray:
    return np.exp(log_softmax(x, axis))


def d_center_by_lse(g: np.ndarray, sm: np.ndarray, axis: int) -> np.ndarray:
    """
    Backward of  y = x - logsumexp(x, axis).
    sm is softmax(x, axis), already computed in the forward pass.
    dL/dx = g - sm * sum(g, axis)
    """
    return g - sm * g.sum(axis=axis, keepdims=True)


def sinkhorn_fwd(S: np.ndarray, tau: float, iters: int):
    """
    Turn an arbitrary score matrix into a (nearly) doubly stochastic one by
    alternately normalising rows and columns in log space.

    The result P is the soft return-mark table: P[i, j] is the mass with which
    reading step i returns to source graph j. As tau falls the matrix hardens
    toward a genuine permutation, which is what a scholar's finished annotation
    is -- every graph read once, in one place, in one order.
    """
    la = S / tau
    cache = []
    for _ in range(iters):
        sm_r = softmax(la, axis=1)
        la = la - np.log(np.exp(la - la.max(1, keepdims=True)).sum(1, keepdims=True)) - la.max(1, keepdims=True)
        sm_c = softmax(la, axis=0)
        la = la - np.log(np.exp(la - la.max(0, keepdims=True)).sum(0, keepdims=True)) - la.max(0, keepdims=True)
        cache.append((sm_r, sm_c))
    P = np.exp(la)
    return P, (cache, tau, P)


def sinkhorn_bwd(gP: np.ndarray, cache) -> np.ndarray:
    steps, tau, P = cache
    g = gP * P                      # through exp
    for sm_r, sm_c in reversed(steps):
        g = d_center_by_lse(g, sm_c, axis=0)
        g = d_center_by_lse(g, sm_r, axis=1)
    return g / tau


def ce_fwd(logits: np.ndarray, targets: np.ndarray, weight: np.ndarray | None = None):
    """Mean cross entropy over rows, with optional per-row weights."""
    ls = log_softmax(logits, axis=1)
    n = logits.shape[0]
    picked = ls[np.arange(n), targets]
    w = np.ones(n) if weight is None else weight
    denom = max(w.sum(), 1e-9)
    loss = -(w * picked).sum() / denom
    return loss, (ls, targets, w, denom)


def ce_bwd(cache) -> np.ndarray:
    ls, targets, w, denom = cache
    p = np.exp(ls)
    g = p.copy()
    g[np.arange(len(targets)), targets] -= 1.0
    return g * (w[:, None] / denom)


# =============================================================================
# SECTION 3 - THE ENGINE
# =============================================================================

@dataclass
class Config:
    d_emb: int = 24
    d_hid: int = 36          # per direction; H is 2*d_hid wide
    d_key: int = 24
    d_head: int = 40
    rel_radius: int = 12     # relative-position bias window for the return-marks
    tau: float = 0.42        # Sinkhorn temperature
    sink_iters: int = 12
    lambda_perm: float = 1.0
    lambda_par: float = 0.6
    abstain_weight: float = 4.0   # ako cases are rare; weight them up


class KaeritenEngine:
    """
    SCAN -> KAERITEN -> KUNDOKU(+AKO GATE)

    Everything the model knows about how to read is in one place: the return-
    mark table P. Everything it knows about what the graphs mean is in the
    lexeme head. Keeping them separate is the whole design. Michizane's marks
    were written in a different ink from the text, in the margin, by a named
    hand, and could be argued with independently of the text they annotated.
    """

    def __init__(self, cfg: Config, seed: int = RNG_SEED):
        self.cfg = cfg
        rng = np.random.default_rng(seed)
        V, T, Pn = len(SRC_VOCAB), len(TGT_VOCAB), len(PARTICLES)
        de, dh, dk, dhh = cfg.d_emb, cfg.d_hid, cfg.d_key, cfg.d_head
        H = 2 * dh

        def n(*shape, scale=None):
            s = scale if scale is not None else 1.0 / math.sqrt(shape[0])
            return rng.normal(0.0, s, size=shape)

        self.p: Dict[str, np.ndarray] = {
            "E":   n(V, de, scale=0.35),
            # scan, forward and backward passes down the column
            "Wf":  n(de, dh), "Uf": n(dh, dh), "bf": np.zeros(dh),
            "Wb":  n(de, dh), "Ub": n(dh, dh), "bb": np.zeros(dh),
            # return-mark scorer
            "Wq":  n(H, dk), "Wk": n(H, dk),
            "rel": np.zeros(2 * cfg.rel_radius + 1),
            # reading heads
            "W1":  n(H, dhh), "b1": np.zeros(dhh),
            "W2":  n(dhh, T), "b2": np.zeros(T),
            "W3":  n(dhh, Pn), "b3": np.zeros(Pn),
        }

    # -- relative position bias -------------------------------------------
    def _rel_bias(self, N: int):
        R = self.cfg.rel_radius
        i = np.arange(N)[:, None]
        j = np.arange(N)[None, :]
        idx = np.clip(j - i, -R, R) + R
        return self.p["rel"][idx], idx

    # -- forward -----------------------------------------------------------
    def forward(self, s: Sample):
        cfg, p = self.cfg, self.p
        N = len(s)
        dh = cfg.d_hid

        X = p["E"][s.src]                                    # (N, de)

        # --- ORGAN 1: SCAN. Two passes down the column, forward and back.
        hf = np.zeros((N, dh)); prev = np.zeros(dh)
        for t in range(N):
            prev = np.tanh(X[t] @ p["Wf"] + prev @ p["Uf"] + p["bf"])
            hf[t] = prev
        hb = np.zeros((N, dh)); nxt = np.zeros(dh)
        for t in range(N - 1, -1, -1):
            nxt = np.tanh(X[t] @ p["Wb"] + nxt @ p["Ub"] + p["bb"])
            hb[t] = nxt
        H = np.concatenate([hf, hb], axis=1)                 # (N, 2dh)

        # --- ORGAN 2: KAERITEN. Score every possible return, then normalise
        #     the score table until it is (nearly) a permutation.
        Q = H @ p["Wq"]; K = H @ p["Wk"]
        S = Q @ K.T / math.sqrt(cfg.d_key)
        rel, rel_idx = self._rel_bias(N)
        S = S + rel
        P, sk_cache = sinkhorn_fwd(S, cfg.tau, cfg.sink_iters)

        # --- ORGAN 3: KUNDOKU. Read the graphs in the order the marks dictate.
        G = P @ H                                            # (N, 2dh)
        A1 = G @ p["W1"] + p["b1"]
        Z = np.tanh(A1)
        lex_logits = Z @ p["W2"] + p["b2"]
        par_logits = Z @ p["W3"] + p["b3"]

        cache = dict(s=s, N=N, X=X, hf=hf, hb=hb, H=H, Q=Q, K=K,
                     P=P, sk=sk_cache, G=G, A1=A1, Z=Z,
                     lex_logits=lex_logits, par_logits=par_logits,
                     rel_idx=rel_idx)
        return cache

    def loss(self, cache) -> Tuple[float, Dict[str, float]]:
        cfg = self.cfg
        s = cache["s"]

        # ORGAN 4 lives here: ako slots carry extra weight so that abstention
        # is learned rather than averaged away as noise.
        # Ako slots are rare and are the whole point, so they carry extra
        # weight -- both the ones that must abstain and the ones that must
        # resolve. Abstention has to be a judgement about the evidence, which
        # means the model must also be punished for abstaining when a cue was
        # sitting right there in the text.
        w = np.where(s.ako, cfg.abstain_weight, 1.0)
        l_lex, c_lex = ce_fwd(cache["lex_logits"], s.lex, w)
        l_par, c_par = ce_fwd(cache["par_logits"], s.par)

        # Permutation supervision: reading step i should return to slot perm[i].
        logP = np.log(np.clip(cache["P"], 1e-12, None))
        l_perm = -logP[np.arange(len(s)), s.perm].mean()

        total = l_lex + cfg.lambda_par * l_par + cfg.lambda_perm * l_perm
        cache["c_lex"], cache["c_par"] = c_lex, c_par
        return total, {"lex": l_lex, "par": l_par, "perm": l_perm}

    # -- backward ----------------------------------------------------------
    def backward(self, cache) -> Dict[str, np.ndarray]:
        cfg, p = self.cfg, self.p
        s, N = cache["s"], cache["N"]
        dh = cfg.d_hid
        g = {k: np.zeros_like(v) for k, v in p.items()}

        # heads
        g_lex = ce_bwd(cache["c_lex"])
        g_par = ce_bwd(cache["c_par"]) * cfg.lambda_par
        Z = cache["Z"]
        g["W2"] += Z.T @ g_lex; g["b2"] += g_lex.sum(0)
        g["W3"] += Z.T @ g_par; g["b3"] += g_par.sum(0)
        gZ = g_lex @ p["W2"].T + g_par @ p["W3"].T
        gA1 = gZ * (1.0 - Z ** 2)
        g["W1"] += cache["G"].T @ gA1; g["b1"] += gA1.sum(0)
        gG = gA1 @ p["W1"].T

        # G = P @ H
        P, H = cache["P"], cache["H"]
        gP = gG @ H.T
        gH = P.T @ gG

        # permutation loss straight into P
        gPl = np.zeros_like(P)
        gPl[np.arange(N), s.perm] = -1.0 / (np.clip(P[np.arange(N), s.perm], 1e-12, None) * N)
        gP = gP + cfg.lambda_perm * gPl

        # through Sinkhorn
        gS = sinkhorn_bwd(gP, cache["sk"])

        # relative bias
        np.add.at(g["rel"], cache["rel_idx"].ravel(), gS.ravel())

        # S = Q K^T / sqrt(dk)
        sc = 1.0 / math.sqrt(cfg.d_key)
        Q, K = cache["Q"], cache["K"]
        gQ = (gS @ K) * sc
        gK = (gS.T @ Q) * sc
        g["Wq"] += H.T @ gQ; g["Wk"] += H.T @ gK
        gH = gH + gQ @ p["Wq"].T + gK @ p["Wk"].T

        # split back into the two scan directions
        ghf, ghb = gH[:, :dh], gH[:, dh:]
        X, hf, hb = cache["X"], cache["hf"], cache["hb"]
        gX = np.zeros_like(X)

        carry = np.zeros(dh)
        for t in range(N - 1, -1, -1):
            gt = ghf[t] + carry
            ga = gt * (1.0 - hf[t] ** 2)
            g["Wf"] += np.outer(X[t], ga)
            prev = hf[t - 1] if t > 0 else np.zeros(dh)
            g["Uf"] += np.outer(prev, ga)
            g["bf"] += ga
            gX[t] += p["Wf"] @ ga
            carry = p["Uf"] @ ga

        carry = np.zeros(dh)
        for t in range(N):
            gt = ghb[t] + carry
            ga = gt * (1.0 - hb[t] ** 2)
            g["Wb"] += np.outer(X[t], ga)
            nxt = hb[t + 1] if t < N - 1 else np.zeros(dh)
            g["Ub"] += np.outer(nxt, ga)
            g["bb"] += ga
            gX[t] += p["Wb"] @ ga
            carry = p["Ub"] @ ga

        np.add.at(g["E"], s.src, gX)
        return g

    # -- convenience -------------------------------------------------------
    def loss_and_grad(self, s: Sample):
        c = self.forward(s)
        L, parts = self.loss(c)
        return L, self.backward(c), parts

    def read(self, s: Sample) -> Dict[str, object]:
        """
        Produce the annotated reading: the return-marks, the glosses, the
        supplied particles. This is the artefact a Heian scholar would hand you,
        and the reason the permutation is kept explicit.
        """
        c = self.forward(s)
        P = c["P"]
        path = P.argmax(axis=1)
        lex = c["lex_logits"].argmax(axis=1)
        par = c["par_logits"].argmax(axis=1)
        conf = softmax(c["lex_logits"], axis=1).max(axis=1)
        pieces = []
        for i in range(len(s)):
            j = int(path[i])
            word = TGT_VOCAB[lex[i]]
            pieces.append(word + PARTICLES[par[i]])
        return {
            "source": "".join(s.graphs),
            "marks": [(int(i), int(path[i])) for i in range(len(s))],
            "reading": " ".join(pieces),
            "confidence": conf,
            "abstained": [TGT_VOCAB[k] == ABSTAIN for k in lex],
            "P": P,
        }


# =============================================================================
# SECTION 4 - GRADIENT CHECK  (mandatory; nothing runs until this passes)
# =============================================================================

def gradient_check(eps: float = 1e-5, n_probe: int = 4, tol: float = 2e-5) -> bool:
    rng = np.random.default_rng(11)
    cfg = Config(d_emb=8, d_hid=7, d_key=6, d_head=9, sink_iters=5, rel_radius=3)
    m = KaeritenEngine(cfg, seed=3)
    sample = make_sample(np.random.default_rng(99))

    _, grads, _ = m.loss_and_grad(sample)

    worst = 0.0
    print("  finite-difference check")
    print("  {:<5} {:>16} {:>16} {:>12}".format("param", "analytic", "numeric", "rel.err"))
    for name in ["E", "Wf", "Uf", "bf", "Wb", "Ub", "bb",
                 "Wq", "Wk", "rel", "W1", "b1", "W2", "b2", "W3", "b3"]:
        arr = m.p[name]
        flat = arr.reshape(-1)
        picks = rng.choice(flat.size, size=min(n_probe, flat.size), replace=False)
        for k in picks:
            orig = flat[k]
            flat[k] = orig + eps
            lp, _ = m.loss(m.forward(sample))
            flat[k] = orig - eps
            lm, _ = m.loss(m.forward(sample))
            flat[k] = orig
            num = (lp - lm) / (2 * eps)
            ana = grads[name].reshape(-1)[k]
            err = abs(num - ana) / max(1.0, abs(num), abs(ana))
            worst = max(worst, err)
        # report the last probe of each parameter for readability
        print("  {:<5} {:>16.9f} {:>16.9f} {:>12.2e}".format(name, ana, num, err))
    ok = worst < tol
    print("  worst relative error over {} probes: {:.3e}  ->  {}"
          .format(16 * n_probe, worst, "PASS" if ok else "FAIL"))
    return ok


# =============================================================================
# SECTION 5 - TRAINING
# =============================================================================

class Adam:
    def __init__(self, params: Dict[str, np.ndarray], lr=6e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for k in params:
            gk = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * gk
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * gk * gk
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)


def evaluate(model: KaeritenEngine, data: Sequence[Sample]) -> Dict[str, float]:
    """
    Six numbers, because the architecture makes six separable claims.

    order_tok / order_sent   did the return-marks land on the right graphs
    lexeme                   did the gloss come out right
    particle                 was the interpolated grammar right (verb ending
                             included, which requires the transposition first)
    ako_recall               when the sentence gave no cue, did it abstain
    ako_resolve              when the sentence DID give a cue, did it use it
                             instead of abstaining -- abstention must be a
                             judgement, not a reflex
    false_abstain            how often it hid behind abstention with no cause
    """
    n_tok = n_ord = n_lex = n_par = 0
    n_sent = n_sent_ok = 0
    ako_tot = ako_hit = 0
    res_tot = res_hit = 0
    false_abstain = clear_tot = 0
    ako_ids = {SRC_IDX[g] for g in AKO_SET}
    for s in data:
        c = model.forward(s)
        path = c["P"].argmax(1)
        lex = c["lex_logits"].argmax(1)
        par = c["par_logits"].argmax(1)
        ok_ord = path == s.perm
        n_ord += ok_ord.sum(); n_tok += len(s)
        n_lex += (lex == s.lex).sum()
        n_par += (par == s.par).sum()
        n_sent += 1
        n_sent_ok += int(ok_ord.all())
        pred_abstain = np.array([TGT_VOCAB[k] == ABSTAIN for k in lex])
        ako_tot += s.abstain.sum(); ako_hit += (pred_abstain & s.abstain).sum()
        clear_tot += (~s.abstain).sum()
        false_abstain += (pred_abstain & ~s.abstain).sum()
        # cued ako slots: the graph is ambiguous but the sentence resolves it
        cued = np.array([(s.src[j] in ako_ids) for j in s.perm]) & ~s.abstain
        res_tot += int(cued.sum())
        res_hit += int(((lex == s.lex) & cued).sum())
    return {
        "order_tok": n_ord / n_tok,
        "order_sent": n_sent_ok / n_sent,
        "lexeme": n_lex / n_tok,
        "particle": n_par / n_tok,
        "ako_recall": (ako_hit / ako_tot) if ako_tot else float("nan"),
        "ako_resolve": (res_hit / res_tot) if res_tot else float("nan"),
        "false_abstain": (false_abstain / clear_tot) if clear_tot else 0.0,
    }


def train(model: KaeritenEngine, train_set, dev_set, epochs=14, batch=16, lr=6e-3, verbose=True):
    opt = Adam(model.p, lr=lr)
    rng = np.random.default_rng(RNG_SEED)
    history = []
    for ep in range(1, epochs + 1):
        opt.lr = lr * (0.88 ** (ep - 1))     # cool the step down; the Sinkhorn
                                             # pass is sharp and bolts if left hot
        idx = rng.permutation(len(train_set))
        run = 0.0
        for b0 in range(0, len(idx), batch):
            chunk = idx[b0:b0 + batch]
            acc = {k: np.zeros_like(v) for k, v in model.p.items()}
            tot = 0.0
            for i in chunk:
                L, gr, _ = model.loss_and_grad(train_set[i])
                tot += L
                for k in acc:
                    acc[k] += gr[k]
            for k in acc:
                acc[k] /= len(chunk)
            # gradient clipping keeps the Sinkhorn pass from bolting early on
            nrm = math.sqrt(sum(float((v ** 2).sum()) for v in acc.values()))
            if nrm > 5.0:
                for k in acc:
                    acc[k] *= 5.0 / nrm
            opt.step(model.p, acc)
            run += tot / len(chunk)
        m = evaluate(model, dev_set)
        history.append((ep, run / max(1, math.ceil(len(idx) / batch)), m))
        if verbose:
            print("  ep {:>2}  loss {:7.4f} | order/tok {:.3f} order/sent {:.3f} "
                  "| lex {:.3f} part {:.3f} | ako-abstain {:.3f} ako-resolve {:.3f}"
                  .format(ep, history[-1][1], m["order_tok"], m["order_sent"],
                          m["lexeme"], m["particle"], m["ako_recall"], m["ako_resolve"]))
    return history


# =============================================================================
# SECTION 6 - RUIJU RE-INDEX
#
# The Ruiju Kokushi took 200 fascicles of chronological history and re-sorted
# them into topical categories WITHOUT altering the wording. That constraint --
# reordering must be lossless -- is testable, so we test it.
# =============================================================================

def ruiju_reindex(model: KaeritenEngine, corpus: Sequence[Sample], k: int = 6, iters: int = 40):
    E = model.p["E"]
    rng = np.random.default_rng(892)          # the year of compilation
    cent = E[rng.choice(len(E), size=k, replace=False)].copy()
    for _ in range(iters):
        d = ((E[:, None, :] - cent[None, :, :]) ** 2).sum(-1)
        a = d.argmin(1)
        for c in range(k):
            if (a == c).any():
                cent[c] = E[a == c].mean(0)
    cats: Dict[int, List[str]] = {c: [] for c in range(k)}
    for gi, c in enumerate(a):
        cats[int(c)].append(SRC_VOCAB[gi])

    # Re-sort the corpus by category of its verb, then verify losslessness.
    flat_before: List[str] = [g for s in corpus for g in s.graphs]
    buckets: Dict[int, List[Sample]] = {c: [] for c in range(k)}
    for s in corpus:
        vpos = s.roles.index("V")
        buckets[int(a[SRC_IDX[s.graphs[vpos]]])].append(s)
    reordered = [s for c in range(k) for s in buckets[c]]
    flat_after: List[str] = [g for s in reordered for g in s.graphs]

    lossless = sorted(flat_before) == sorted(flat_after)
    return cats, reordered, lossless


def diagnose_long_form(model: KaeritenEngine, data: Sequence[Sample]) -> Dict[str, float]:
    """
    Not "how wrong", but "wrong how".

    Training sentences carry at most one prepositional phrase. From that data
    the correct rule -- read each phrase's noun, then its preposition, phrases
    in the order written -- is UNDERDETERMINED: with one phrase you cannot tell
    ordering-by-position from ordering-by-distance-to-the-verb. The held-out
    long form supplies two phrases and forces the question.

    This function asks whether the residual errors are that one confusion, or
    whether the reading path fell apart generally.
    """
    swapped = 0
    swapped_and_otherwise_perfect = 0
    total_two_pp = 0
    other_broken = 0
    for s in data:
        pp = [i for i, r in enumerate(s.roles) if r == "P"]
        if len(pp) != 2:
            continue
        total_two_pp += 1
        path = model.forward(s)["P"].argmax(1)
        # build the counterfactual: the same reading with the two phrases swapped
        alt = s.perm.copy()
        steps = [k for k, j in enumerate(s.perm) if s.roles[j] in ("P", "L")]
        if len(steps) == 4:
            a, b, c, d = steps
            alt[[a, b, c, d]] = s.perm[[c, d, a, b]]
        hit_true = (path == s.perm).all()
        hit_alt = (path == alt).all()
        if hit_alt and not hit_true:
            swapped += 1
            swapped_and_otherwise_perfect += 1
        elif not hit_true:
            other_broken += 1
    return {
        "sentences_with_two_phrases": total_two_pp,
        "explained_by_phrase_order": swapped,
        "broken_some_other_way": other_broken,
        "share_explained": (swapped / max(1, swapped + other_broken)),
    }


# =============================================================================
# SECTION 7 - POSTHUMOUS AUDIT
#
# Re-score every decision after the fact and surface the ones that were wrong
# AND confident. Michizane's own rehabilitation arrived in 923, twenty years
# after his death; this one arrives at the end of the run.
# =============================================================================

def posthumous_audit(model: KaeritenEngine, data: Sequence[Sample], thresh: float = 0.85):
    confident_wrong = 0
    total_wrong = 0
    worst = None
    worst_conf = -1.0
    for s in data:
        c = model.forward(s)
        lex = c["lex_logits"].argmax(1)
        conf = softmax(c["lex_logits"], 1).max(1)
        bad = lex != s.lex
        total_wrong += int(bad.sum())
        cw = bad & (conf > thresh)
        confident_wrong += int(cw.sum())
        if cw.any():
            i = int(np.argmax(np.where(cw, conf, -1)))
            if conf[i] > worst_conf:
                worst_conf = float(conf[i])
                worst = ("".join(s.graphs), TGT_VOCAB[lex[i]], TGT_VOCAB[s.lex[i]], conf[i])
    return {"total_wrong": total_wrong,
            "confident_wrong": confident_wrong,
            "worst": worst}


# =============================================================================
# SECTION 8 - SELF-TESTS
# =============================================================================

def test_doubly_stochastic() -> bool:
    rng = np.random.default_rng(0)
    S = rng.normal(size=(7, 7))
    P, _ = sinkhorn_fwd(S, tau=0.42, iters=40)
    r = abs(P.sum(1) - 1).max(); c = abs(P.sum(0) - 1).max()
    ok = r < 1e-6 and c < 1e-3
    print("  row deviation {:.2e}, column deviation {:.2e}  ->  {}"
          .format(r, c, "PASS" if ok else "FAIL"))
    return ok


def test_corpus_wellformed() -> bool:
    data = make_corpus(500, seed=1)
    ok = True
    for s in data:
        if sorted(s.perm.tolist()) != list(range(len(s))):
            ok = False; break
        if not (len(s.lex) == len(s.par) == len(s)):
            ok = False; break
    abst = np.mean([s.abstain.any() for s in data])
    lens = np.array([len(s) for s in data])
    longs = make_corpus(120, seed=2, long_form=True)
    llens = np.array([len(s) for s in longs])
    print("  {} training-shape sentences, every reading path a true permutation: {}"
          .format(len(data), ok))
    print("  lengths {}-{} (mean {:.1f}); uncued ako sentences {:.1%}"
          .format(lens.min(), lens.max(), lens.mean(), abst))
    print("  held-out long form lengths {}-{} (mean {:.1f}) -- never trained on"
          .format(llens.min(), llens.max(), llens.mean()))
    return ok and 0.02 < abst < 0.40 and llens.max() > lens.max()


def test_interpolation_depends_on_transposition() -> bool:
    """
    The verb ending is decided by the slot that follows the verb in the READING,
    which in the source text sits BEFORE the verb. If the corpus did not encode
    that dependency the particle head would be learnable from source order
    alone, and the architecture would prove nothing.
    """
    data = make_corpus(400, seed=7)
    seen: Dict[Tuple[int, int], set] = {}
    for s in data:
        vi = s.roles.index("V")
        step = int(np.where(s.perm == vi)[0][0])
        key = (int(s.src[vi]), vi)          # same verb graph, same source slot
        seen.setdefault(key, set()).add(int(s.par[step]))
    ambiguous = sum(1 for v in seen.values() if len(v) > 1)
    print("  verb graphs whose ending is NOT determined by source position: {} of {}"
          .format(ambiguous, len(seen)))
    return ambiguous > 0


def main() -> int:
    np.set_printoptions(precision=3, suppress=True, linewidth=120)
    print("=" * 78)
    print("CHAPTER 0210 - SUGAWARA NO MICHIZANE (845-903)")
    print("THE KAERITEN ENGINE - transposition without emendation")
    print("=" * 78)

    print("\n[1] CORPUS")
    if not test_corpus_wellformed():
        print("  corpus malformed; aborting"); return 1
    if not test_interpolation_depends_on_transposition():
        print("  corpus does not encode the dependency it claims; aborting"); return 1

    print("\n[2] SINKHORN OPERATOR")
    if not test_doubly_stochastic():
        print("  operator failed; aborting"); return 1

    print("\n[3] GRADIENT CHECK")
    if not gradient_check():
        print("  gradients disagree with finite differences; aborting"); return 1

    print("\n[4] TRAINING")
    train_set = make_corpus(900, seed=845)
    dev_set = make_corpus(240, seed=903)
    long_set = make_corpus(240, seed=947, long_form=True)   # never seen in training
    model = KaeritenEngine(Config())
    before = evaluate(model, dev_set)
    print("  before training: order/tok {:.3f}  lexeme {:.3f}"
          .format(before["order_tok"], before["lexeme"]))
    hist = train(model, train_set, dev_set, epochs=24)
    after = hist[-1][2]

    print("\n[5] LENGTH GENERALISATION  (held-out long form, never trained on)")
    lg = evaluate(model, long_set)
    print("  order/tok {:.3f}  order/sent {:.3f}  lexeme {:.3f}  particle {:.3f}"
          .format(lg["order_tok"], lg["order_sent"], lg["lexeme"], lg["particle"]))
    print("  ako-abstain {:.3f}  ako-resolve {:.3f}  false-abstain {:.3f}"
          .format(lg["ako_recall"], lg["ako_resolve"], lg["false_abstain"]))
    d = diagnose_long_form(model, long_set)
    print("  error taxonomy on the {} two-phrase sentences:"
          .format(int(d["sentences_with_two_phrases"])))
    print("    read correctly but with the two phrases in reversed order : {}"
          .format(int(d["explained_by_phrase_order"])))
    print("    broken in some other way                                  : {}"
          .format(int(d["broken_some_other_way"])))
    print("    share of failures explained by that single confusion      : {:.3f}"
          .format(d["share_explained"]))

    print("\n[6] ABLATION  (same engine, return-marks left unsupervised)")
    abl = KaeritenEngine(Config(lambda_perm=0.0), seed=RNG_SEED + 1)
    train(abl, train_set, dev_set, epochs=24, verbose=False)
    am = evaluate(abl, dev_set)
    print("  with mark supervision   : order/tok {:.3f}  lexeme {:.3f}  particle {:.3f}"
          .format(after["order_tok"], after["lexeme"], after["particle"]))
    print("  without                 : order/tok {:.3f}  lexeme {:.3f}  particle {:.3f}"
          .format(am["order_tok"], am["lexeme"], am["particle"]))

    print("\n[7] TWO READINGS OF THE SAME GRAPH")
    ako_ids = {SRC_IDX[g] for g in AKO_SET}
    shown = 0
    for s in long_set + dev_set:
        if shown >= 3:
            break
        if not any(int(x) in ako_ids for x in s.src):
            continue
        r = model.read(s)
        cue = [g for g, ro in zip(s.graphs, s.roles) if ro == "C"]
        print("  source : {}   [cue: {}]".format(r["source"], cue[0] if cue else "none"))
        print("  marks  : " + "  ".join("{}<-{}".format(i + 1, j + 1) for i, j in r["marks"]))
        print("  reading: {}".format(r["reading"]))
        shown += 1

    print("\n[8] RUIJU RE-INDEX")
    cats, reordered, lossless = ruiju_reindex(model, dev_set)
    for c in sorted(cats):
        if cats[c]:
            print("  category {}: {}".format(c, " ".join(cats[c])))
    print("  corpus re-sorted, token multiset preserved: {}".format(lossless))

    print("\n[9] POSTHUMOUS AUDIT")
    for label, ds in (("in distribution", dev_set), ("held-out long form", long_set)):
        aud = posthumous_audit(model, ds)
        print("  {:<20} glosses wrong {:>4}   confident (p>0.85) {:>4}"
              .format(label, aud["total_wrong"], aud["confident_wrong"]))
        if aud["worst"]:
            src, got, want, cf = aud["worst"]
            print("      most confident error: {} -> said '{}', wanted '{}' at p={:.2f}"
                  .format(src, got, want, cf))
        else:
            print("      no confident errors remain")

    print("\n[10] SUMMARY")
    rows = [("reading-order accuracy", "order_tok"),
            ("whole-sentence order", "order_sent"),
            ("lexeme accuracy", "lexeme"),
            ("interpolated particle", "particle"),
            ("ako abstention recall", "ako_recall"),
            ("ako resolution (cued)", "ako_resolve"),
            ("false abstention rate", "false_abstain")]
    print("  {:<26} {:>8} {:>8} {:>10}".format("", "start", "trained", "held-out"))
    for label, key in rows:
        print("  {:<26} {:>8.3f} {:>8.3f} {:>10.3f}"
              .format(label, before[key], after[key], lg[key]))
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
