#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
   PMF-Net : the Punch-Matrix-Forme Network
 # BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0400 - Gutenberg  (1400-1460)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 20, Minds 381-400 Available on Amazon https://www.amazon.com/dp/B0HRS9ZHVG
# Resume and Interactive Demos at  https://artificiology.com/
# Videos of the playable minds at https://www.youtube.com/@LostMindsAI
# END ATTRIBUTION  
================================================================================

WHY THIS ARCHITECTURE AND NOT ANOTHER
-------------------------------------
The received story about this figure -- "he made knowledge cheap to copy" -- is
the one thing the evidence does not support as his distinctive contribution.
Cast metal type already existed in Korea; by 1434 the Joseon court had cast over
200,000 pieces of it. Copying was not the bottleneck he solved.

What the physical evidence does show is a chain of negatives:

        punch  ->  matrix  ->  hand mould  ->  sort  ->  forme  ->  sheet

and a startling asymmetry in how much precision was spent at each link. The
BODY of a sort is held to a tolerance so tight that a deviation of a fraction of
a millimetre destroys the page: too short and the letter does not print at all,
too tall and it punches through the paper. The FACE of the same sort is not held
to any such tolerance. Computational clustering of the lowercase 'i' in the DK
type (Needham & Aguera y Arcas, 2000-2003) found not one shape but hundreds --
far more variation than a single reusable matrix could produce. The identity
that the printing revolution is supposed to have delivered was never a physical
property of the type. It was a tolerance budget, spent only where error
COMPOUNDS and withheld everywhere error merely SITS.

That is the cognitive signature this file implements. Three consequences follow,
and each is a concrete mechanism below:

  (1) GENERATED, NOT STORED. There is no embedding table in this model. The
      vector for each symbol is produced at every forward pass by descending
      the chain: a handful of punches, struck into matrices, cast into sorts.
      Parameters live at the top; instances are derivatives. A correction made
      to a punch propagates exactly and instantly to every descendant, which is
      what a chain of negatives buys you and what a lookup table cannot.

  (2) THE CONTAINER EDITS THE CONTENT. A line of type must sum to the measure
      exactly. Compositors facing a full line reached for abbreviations and
      ligatures -- the B42 fount carried about 290 sorts, including roughly 92
      abbreviations and 83 ligatures, for a 24-letter alphabet. Spelling in
      early printed books demonstrably bends to the line (Voeste 2021). So this
      network carries a running REMAINING MEASURE as a first-class input and a
      hard feasibility penalty on sorts that will not fit. The geometry of the
      container reaches backwards into the choice of word.

  (3) AFTER COMMITMENT, CORRECTION IS OUT OF BAND. Once a forme is locked and
      pulled, the sheets already printed cannot be revised; the trade invented
      the stop-press variant and the errata leaf instead. The STOP-PRESS PATCH
      here is a frozen base plus a hard-budgeted sparse correction, fit without
      touching a single base parameter.

And one diagnostic that is his and nobody else's -- the COLLATION module. When
every copy of a text was written out by a different hand, errors were
independent and majority vote across witnesses recovered the truth; that is the
whole basis of textual criticism. After 1455 the errors of a print run are
perfectly shared: 180 copies agree because they are one setting, not because
they are 180 observations. Agreement stops being evidence. The module measures
this collapse numerically and shows what has to replace naive voting.

WHAT IS DELIBERATELY ABSENT
---------------------------
No transformer, no attention over stored keys, no mixture-of-experts. Attention
retrieves from a table of remembered items; this mind's entire discovery was
that you should not keep the table. The recurrence here is a composing stick:
state accumulates left to right under a width budget, exactly as a line is set.

RUNNING
-------
    python3 chapter_0400_gutenberg_johannes_gensfleisch_1400.py            # full suite: grad check, train, 5 studies
    python3 chapter_0400_gutenberg_johannes_gensfleisch_1400.py --quick    # shorter training, same tests

Pure NumPy. No other dependency. Deterministic under the seeds set below.
================================================================================
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ==============================================================================
# SECTION 0 -- THE TYPE CASE
# ==============================================================================
# Before any learning happens we need a physical inventory: the sorts, and the
# width of each one in typographic units. Width is not decoration here -- it is
# the constraint that drives the whole composing mechanism, so it is declared
# first and never learned. In a real case the widths came from the mould; here
# they are fixed integers on an 18-unit em, which is how set-widths were
# actually reckoned.

EM_UNITS = 18          # one em, subdivided; a wide 'm' occupies a full em
MEASURE = 90           # width of one line, in units (five ems)

# Each entry: symbol -> set width in units.
# The narrow sorts (i, l, t, punctuation) are the ones a compositor reaches for
# when a line runs long; the abbreviation sorts do the same job more violently,
# swallowing several letters into one mark.
BASE_SORTS: Dict[str, int] = {
    "a": 10, "b": 11, "c":  9, "d": 11, "e": 10, "f":  6, "g": 11, "h": 11,
    "i":  5, "l":  5, "m": 18, "n": 11, "o": 11, "p": 11, "q": 11, "r":  8,
    "s":  8, "t":  7, "u": 11, "x": 10, " ":  6, ".":  5,
}

# Abbreviation sorts. Historically these are the tilde-bearing contractions and
# the ligatures that let a compositor compress a line without respacing it.
# Each carries the meaning of a multi-character string in a single narrow body.
ABBREV_SORTS: Dict[str, int] = {
    "~": 6,    # nasal bar: stands for a following m or n
    "&": 9,    # the et ligature
    "9": 7,    # the -us contraction
}

SORTS: Dict[str, int] = {**BASE_SORTS, **ABBREV_SORTS}
ITOS: List[str] = sorted(SORTS.keys())
STOI: Dict[str, int] = {s: i for i, s in enumerate(ITOS)}
V: int = len(ITOS)
WIDTHS: np.ndarray = np.array([SORTS[s] for s in ITOS], dtype=np.float64)

ABBREV_IDS = np.array([STOI[s] for s in ABBREV_SORTS], dtype=np.int64)


# ==============================================================================
# SECTION 1 -- THE EXEMPLAR AND THE COMPOSITOR
# ==============================================================================
# The training corpus is not random noise. It is generated by a small stochastic
# grammar (the "exemplar" -- the manuscript being set) and then run through a
# simulated compositor who must justify each line to MEASURE exactly.
#
# The critical property: when the line is running short of room, the compositor
# substitutes abbreviated forms. That means the choice of SYMBOL depends on the
# remaining space, not only on the language. Any model that ignores the width
# budget is structurally unable to predict those positions. This is the property
# the ablation study in Section 6 measures.

STEMS = [
    "dominus", "liber", "sanctus", "littera", "manus", "opus", "verbum",
    "nomen", "forma", "corpus", "lumen", "signum", "pagina", "tempus",
    "ordo", "ferrum", "plumbum", "labor", "artem", "codex",
]
SUFFIXES = ["", "um", "is", "es", "ibus", "orum"]


def _abbreviate(word: str) -> str:
    """Return a shortened spelling of `word` using the abbreviation sorts.

    This mirrors what a compositor actually did with a long line: replace a
    nasal with the bar over the preceding letter, swallow a terminal -us into
    one mark, or drop in the et-ligature. The result is narrower in units but
    means the same thing.
    """
    w = word
    if w.endswith("us"):
        w = w[:-2] + "9"
    if "m" in w:
        w = w.replace("m", "~", 1)
    elif "n" in w:
        w = w.replace("n", "~", 1)
    return w


def _clean(word: str) -> str:
    """Drop any character we do not hold in the case (we have no full alphabet;
    a real fount is an inventory, not the abstract alphabet)."""
    return "".join(ch for ch in word if ch in SORTS)


def _width(s: str) -> int:
    return int(sum(SORTS[ch] for ch in s))


def compose_corpus(n_lines: int, rng: np.random.Generator) -> Tuple[List[int], np.ndarray]:
    """Set `n_lines` lines of justified type and return the flat sort stream
    plus the remaining-measure trace that accompanies it.

    Returns
    -------
    stream : list of sort ids, read in setting order, lines concatenated
    remain : float array, same length; remain[t] is the space still free in the
             line at the moment sort t is about to be placed, normalised to
             [0, 1] by MEASURE. This is the compositor's own working variable.
    """
    stream: List[int] = []
    remain: List[float] = []

    for _ in range(n_lines):
        used = 0
        line: List[str] = []
        while True:
            stem = STEMS[rng.integers(len(STEMS))]
            suf = SUFFIXES[rng.integers(len(SUFFIXES))]
            word = _clean(stem + suf)
            if not word:
                continue

            free = MEASURE - used
            long_w = _width(word) + SORTS[" "]
            short = _abbreviate(word)
            short_w = _width(short) + SORTS[" "]

            # The compositor's decision. With plenty of room he sets the full
            # form (with a little noise -- he is a person, not a rule). When the
            # full form will not fit but the contraction will, he contracts.
            # When neither fits, the line is done and gets a full stop.
            if long_w <= free and rng.random() > 0.05:
                chosen = word + " "
            elif short_w <= free:
                chosen = short + " "
            else:
                # Close the line with a period if there is room for one.
                if free >= SORTS["."]:
                    chosen = "."
                    line.append(chosen)
                    for ch in chosen:
                        stream.append(STOI[ch])
                        remain.append((MEASURE - used) / MEASURE)
                        used += SORTS[ch]
                break

            for ch in chosen:
                stream.append(STOI[ch])
                remain.append((MEASURE - used) / MEASURE)
                used += SORTS[ch]
            line.append(chosen)
            if used >= MEASURE - SORTS[" "]:
                break

    return stream, np.asarray(remain, dtype=np.float64)


def make_batches(stream: List[int], remain: np.ndarray, seq_len: int
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cut the flat stream into aligned (input, target, measure) windows."""
    n = (len(stream) - 1) // seq_len
    x = np.array(stream[: n * seq_len], dtype=np.int64).reshape(n, seq_len)
    y = np.array(stream[1: n * seq_len + 1], dtype=np.int64).reshape(n, seq_len)
    r = remain[: n * seq_len].reshape(n, seq_len)
    return x, y, r


# ==============================================================================
# SECTION 2 -- THE CHAIN OF NEGATIVES
# ==============================================================================
# The heart of the model. Nothing here is a lookup table.
#
#   Punch  (P, D)   -- hand-cut masters. Few, expensive, and the only place
#                      where a correction can be made cheaply and globally.
#   Strike (M, P)   -- how each matrix was struck. Row-softmaxed, so a matrix is
#                      a convex blend of punches and owns no free content.
#   Cast   (V, M)   -- which matrix each sort was cast from. Row-softmaxed.
#   Gain   (V,)     -- the set width of the sort's face: a per-sort scalar, the
#                      only degree of freedom an individual sort possesses.
#
# The embedding of sort v is therefore
#
#       E[v] = Gain[v] * ( softmax(Cast[v]) @ tanh( softmax(Strike) @ Punch ) )
#
# Parameter count is P*D + M*P + V*M + V, and V*D never appears. Every sort is
# a derivative of something upstream. That is the whole thesis in a tensor.


@dataclass
class PMFConfig:
    n_punch: int = 8         # hand-cut masters
    n_matrix: int = 12       # struck negatives
    d_face: int = 48         # dimensionality of a letter face
    d_hidden: int = 96       # the composing stick
    seq_len: int = 64
    justify_beta: float = 6.0    # strength of the feasibility penalty
    seed: int = 1455


def softmax_rows(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class PMFNet:
    """The Punch-Matrix-Forme network.

    Forward pass, in the order the workshop does it:
      1. strike the matrices from the punches
      2. cast the sorts from the matrices
      3. set the line, carrying the remaining measure in the state
      4. bias the next-sort logits by what will physically still fit
    """

    def __init__(self, cfg: PMFConfig, widths: np.ndarray = WIDTHS, vocab: int = V):
        self.cfg = cfg
        self.V = vocab
        self.widths = widths.astype(np.float64)
        self.w_norm = self.widths / MEASURE      # widths in the same units as `remain`

        rng = np.random.default_rng(cfg.seed)
        P, M, D, H = cfg.n_punch, cfg.n_matrix, cfg.d_face, cfg.d_hidden

        def sc(a, b, g=1.0):
            return rng.standard_normal((a, b)) * (g / math.sqrt(b))

        self.p: Dict[str, np.ndarray] = {
            # --- the chain of negatives ---
            "Punch":  rng.standard_normal((P, D)) * 0.8,
            "Strike": rng.standard_normal((M, P)) * 0.5,
            "Cast":   rng.standard_normal((vocab, M)) * 0.5,
            "Gain":   np.ones(vocab) + rng.standard_normal(vocab) * 0.02,
            # --- the composing stick ---
            "Wx": sc(D, H),
            "Wh": sc(H, H, 0.7),
            "wr": rng.standard_normal(H) * 0.10,   # the remaining-measure input
            "bh": np.zeros(H),
            # --- the forme's output face ---
            "Wo": sc(H, vocab),
            "bo": np.zeros(vocab),
        }
        self.use_measure = True       # ablation switch (Section 6)
        self.use_justify = True       # ablation switch (Section 6)
        self._patch: Optional[np.ndarray] = None   # stop-press patch (Section 8)

    # ---------------------------------------------------------------- helpers
    def n_params(self) -> int:
        return int(sum(v.size for v in self.p.values()))

    def copy_params(self) -> Dict[str, np.ndarray]:
        return {k: v.copy() for k, v in self.p.items()}

    def load_params(self, d: Dict[str, np.ndarray]) -> None:
        for k, v in d.items():
            self.p[k] = v.copy()

    # ------------------------------------------------------- the chain itself
    def build_faces(self) -> Dict[str, np.ndarray]:
        """Strike the matrices, then cast the sorts. Returns every intermediate
        because the backward pass has to walk back up the same chain."""
        p = self.p
        s = softmax_rows(p["Strike"])            # (M, P) strike proportions
        mat_pre = s @ p["Punch"]                  # (M, D) matrix cavity, pre-tanh
        mat = np.tanh(mat_pre)                    # (M, D) the struck negative
        c = softmax_rows(p["Cast"])               # (V, M) which matrix per sort
        face_raw = c @ mat                        # (V, D) the cast face
        E = p["Gain"][:, None] * face_raw         # (V, D) scaled by set width
        return {"s": s, "mat_pre": mat_pre, "mat": mat, "c": c,
                "face_raw": face_raw, "E": E}

    def justification_bias(self, r: np.ndarray) -> np.ndarray:
        """Penalise, per timestep, every sort too wide for the space that is left.

        r : (B,) remaining measure, normalised.
        Returns (B, V) additive logit bias, <= 0.

        This is the container reaching into the content. It has no parameters --
        it is geometry, not preference -- so no gradient flows through it, and
        it cannot be learned away. When the line is nearly full, the wide sorts
        are simply not available and the narrow abbreviations are all that is
        left, which is precisely why the abbreviation sorts exist.
        """
        if not self.use_justify:
            return np.zeros((r.shape[0], self.V))
        over = self.w_norm[None, :] - r[:, None]          # >0 means it will not fit
        return -self.cfg.justify_beta * np.maximum(over, 0.0)

    # -------------------------------------------------------- forward + loss
    def forward(self, x: np.ndarray, y: np.ndarray, r: np.ndarray,
                cache: bool = True) -> Tuple[float, Optional[dict]]:
        """x, y : (B, T) int ids.  r : (B, T) remaining measure in [0,1].

        Returns mean cross-entropy in nats over all B*T positions.
        """
        B, T = x.shape
        H = self.cfg.d_hidden
        chain = self.build_faces()
        E = chain["E"]

        h_prev = np.zeros((B, H))
        hs: List[np.ndarray] = []
        probs: List[np.ndarray] = []
        loss = 0.0

        rin = r if self.use_measure else np.zeros_like(r)

        for t in range(T):
            e_t = E[x[:, t]]                                   # (B, D)
            a = e_t @ self.p["Wx"] + h_prev @ self.p["Wh"] \
                + np.outer(rin[:, t], self.p["wr"]) + self.p["bh"]
            h = np.tanh(a)
            z = h @ self.p["Wo"] + self.p["bo"] + self.justification_bias(r[:, t])
            if self._patch is not None:
                z = z + self._patch                            # stop-press correction
            pr = softmax_rows(z)
            loss += -np.log(np.maximum(pr[np.arange(B), y[:, t]], 1e-12)).sum()
            if cache:
                hs.append(h)
                probs.append(pr)
            h_prev = h

        loss /= (B * T)
        if not cache:
            return loss, None
        return loss, {"chain": chain, "hs": hs, "probs": probs,
                      "x": x, "y": y, "r": r, "rin": rin, "B": B, "T": T}

    # ------------------------------------------------------------- backward
    def backward(self, ctx: dict) -> Dict[str, np.ndarray]:
        """Backprop through the composing stick, then back UP the chain of
        negatives: sorts -> matrices -> punches.

        The second half is the interesting part. Because no sort owns its own
        content, the gradient of every sort's error is routed into the matrix it
        was cast from, and the gradient of every matrix into the punches that
        struck it. A punch therefore accumulates the error of its whole
        descendant population, which is exactly why a punch is worth cutting
        slowly by hand and a sort is not.
        """
        p, cfg = self.p, self.cfg
        chain, hs, probs = ctx["chain"], ctx["hs"], ctx["probs"]
        x, y, rin, B, T = ctx["x"], ctx["y"], ctx["rin"], ctx["B"], ctx["T"]
        E = chain["E"]
        scale = 1.0 / (B * T)

        g = {k: np.zeros_like(v) for k, v in p.items()}
        dE = np.zeros_like(E)
        dh_next = np.zeros((B, cfg.d_hidden))

        for t in reversed(range(T)):
            h, pr = hs[t], probs[t]
            h_prev = hs[t - 1] if t > 0 else np.zeros_like(h)

            dz = pr.copy()
            dz[np.arange(B), y[:, t]] -= 1.0
            dz *= scale

            g["Wo"] += h.T @ dz
            g["bo"] += dz.sum(axis=0)

            dh = dz @ p["Wo"].T + dh_next
            da = dh * (1.0 - h * h)

            e_t = E[x[:, t]]
            g["Wx"] += e_t.T @ da
            g["Wh"] += h_prev.T @ da
            g["wr"] += rin[:, t] @ da
            g["bh"] += da.sum(axis=0)

            np.add.at(dE, x[:, t], da @ p["Wx"].T)
            dh_next = da @ p["Wh"].T

        # ---- back up the chain -------------------------------------------
        # E[v] = Gain[v] * face_raw[v]
        g["Gain"] = (dE * chain["face_raw"]).sum(axis=1)
        dface = dE * p["Gain"][:, None]                     # (V, D)

        # face_raw = c @ mat
        c, mat = chain["c"], chain["mat"]
        dc = dface @ mat.T                                   # (V, M)
        g["Cast"] = c * (dc - (dc * c).sum(axis=1, keepdims=True))
        dmat = c.T @ dface                                   # (M, D)

        # mat = tanh(s @ Punch)
        dmat_pre = dmat * (1.0 - mat * mat)                  # (M, D)
        s = chain["s"]
        ds = dmat_pre @ p["Punch"].T                         # (M, P)
        g["Strike"] = s * (ds - (ds * s).sum(axis=1, keepdims=True))
        g["Punch"] = s.T @ dmat_pre                          # (P, D)

        return g


# ==============================================================================
# SECTION 3 -- GRADIENT CHECK (mandatory)
# ==============================================================================

def gradient_check(net: PMFNet, x, y, r, n_probe: int = 6, eps: float = 1e-6
                   ) -> Tuple[float, List[Tuple[str, float, float]]]:
    """Central finite differences against the analytic gradient.

    Every parameter block is probed, including the two softmax-routed blocks
    (Strike, Cast) where a sign error would be easy to hide.
    """
    loss, ctx = net.forward(x, y, r)
    grads = net.backward(ctx)
    rng = np.random.default_rng(7)
    worst = 0.0
    report: List[Tuple[str, float, float]] = []

    for name, arr in net.p.items():
        block_worst = 0.0
        block_absw = 0.0
        flat = arr.ravel()
        idxs = rng.choice(flat.size, size=min(n_probe, flat.size), replace=False)
        for i in idxs:
            orig = flat[i]
            flat[i] = orig + eps
            lp, _ = net.forward(x, y, r, cache=False)
            flat[i] = orig - eps
            lm, _ = net.forward(x, y, r, cache=False)
            flat[i] = orig
            num = (lp - lm) / (2 * eps)
            ana = grads[name].ravel()[i]
            # Combined criterion. A relative error is meaningless when both
            # gradients are near zero, which is routine in the softmax-routed
            # blocks (Strike, Cast) where individual entries carry gradients of
            # order 1e-5. Central differences in float64 leave roundoff of order
            # 1e-10 in the loss, so a discrepancy below 1e-8 nats is arithmetic
            # noise and is scored as a pass; anything above it must meet the
            # relative bound.
            denom = abs(num) + abs(ana) + 1e-12
            rel = abs(num - ana) / denom
            absd = abs(num - ana)
            score = rel if absd > 1e-8 else 0.0
            block_worst = max(block_worst, score)
            block_absw = max(block_absw, absd)
        report.append((name, block_worst, block_absw))
        worst = max(worst, block_worst)
    return worst, report


# ==============================================================================
# SECTION 4 -- TRAINING (Adam)
# ==============================================================================

class Adam:
    def __init__(self, params: Dict[str, np.ndarray], lr=3e-2, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray],
             clip: float = 5.0, frozen: Optional[set] = None) -> None:
        self.t += 1
        total = math.sqrt(sum(float((g * g).sum()) for g in grads.values()))
        s = min(1.0, clip / (total + 1e-12))
        for k in params:
            if frozen and k in frozen:
                continue
            gk = grads[k] * s
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * gk
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (gk * gk)
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)


def evaluate(net: PMFNet, x, y, r, bs: int = 32) -> float:
    tot, n = 0.0, 0
    for i in range(0, len(x), bs):
        l, _ = net.forward(x[i:i + bs], y[i:i + bs], r[i:i + bs], cache=False)
        k = len(x[i:i + bs])
        tot += l * k
        n += k
    return tot / max(n, 1)


def train(net: PMFNet, tr, va, epochs: int = 12, bs: int = 32, lr: float = 3e-2,
          log=print) -> List[Tuple[int, float, float]]:
    xt, yt, rt = tr
    opt = Adam(net.p, lr=lr)
    rng = np.random.default_rng(1450)
    hist = []
    for ep in range(1, epochs + 1):
        order = rng.permutation(len(xt))
        run, nb = 0.0, 0
        for i in range(0, len(order) - bs + 1, bs):
            sel = order[i:i + bs]
            loss, ctx = net.forward(xt[sel], yt[sel], rt[sel])
            grads = net.backward(ctx)
            opt.step(net.p, grads)
            run += loss
            nb += 1
        vl = evaluate(net, *va)
        hist.append((ep, run / max(nb, 1), vl))
        log(f"    epoch {ep:2d} | train {run/max(nb,1):.4f} | "
            f"val {vl:.4f} | val ppl {math.exp(vl):7.3f}")
    return hist


# ==============================================================================
# SECTION 5 -- STUDY A : THE TOLERANCE BUDGET
# ==============================================================================
# The central experiment, and the one that decides whether the reading of this
# mind offered in the header is right.
#
# A sort has two dimensions that can go wrong, and they are not symmetric.
#
#   THE BODY -- height-to-paper and set width. Errors here ADD ALONG THE LINE.
#       Ninety units of measure must be filled to the unit or the forme will not
#       lock and the page cannot be pulled. The accumulated error over n sorts
#       grows like sqrt(n), so the tolerance a single sort must meet gets
#       tighter the bigger the forme. A column of the B42 runs to roughly 2,600
#       sorts.
#
#   THE FACE -- the letter image itself. Errors here DO NOT ADD. Each glyph's
#       deviation terminates at that glyph. The only thing that matters is
#       whether a reader can still tell which letter it is, and a reader's
#       discrimination is coarse.
#
# LOCK_TOL below is one unit -- narrower than a hair space. The face test asks
# whether a perturbed face is still nearest to its own prototype, which is the
# cheapest honest model of "the reader still reads it as an i".
#
# If the two thresholds come out orders of magnitude apart, then the striking
# thing about the surviving evidence -- bodies machined to a fraction of a
# millimetre, faces falling into hundreds of distinct clusters that no reader
# ever complained about -- is not sloppiness partly redeemed by genius. It is
# the correct allocation, and it could have been derived.

LOCK_TOL = 1.0        # units of accumulated width error a forme will still take
FORME_SIZES = (15, 120, 600, 2600)   # one line, one page-block, ... one column


def forme_lock_failure(sigma: float, n_sorts: int, trials: int = 4000,
                       rng: Optional[np.random.Generator] = None) -> float:
    """Fraction of formes that will not lock, given relative body error `sigma`.

    Each sort contributes an independent width error proportional to its own
    width. The errors sum. This is arithmetic, not simulation of the model --
    it is a property of the composing constraint itself.
    """
    rng = rng or np.random.default_rng(5)
    w = WIDTHS[rng.integers(0, len(WIDTHS), size=(trials, n_sorts))]
    err = (w * rng.standard_normal(w.shape) * sigma).sum(axis=1)
    return float((np.abs(err) > LOCK_TOL).mean())


def face_identification_failure(net: PMFNet, sigma: float, trials: int = 200,
                                rng: Optional[np.random.Generator] = None) -> float:
    """Fraction of cast faces that can no longer be identified as themselves.

    A face is perturbed and then matched back against the clean prototypes. A
    misread is a face whose nearest prototype is some other letter. This is the
    reader's side of the bargain, and it is the only place face precision has to
    be bought at all.
    """
    rng = rng or np.random.default_rng(6)
    E0 = net.build_faces()["E"]
    sc = float(np.abs(E0).mean())
    bad = tot = 0
    for _ in range(trials):
        E = E0 + rng.standard_normal(E0.shape) * sigma * sc
        d = ((E[:, None, :] - E0[None, :, :]) ** 2).sum(-1)
        bad += int((d.argmin(1) != np.arange(len(E0))).sum())
        tot += len(E0)
    return bad / tot


def tolerance_budget(net: PMFNet,
                     body_sigmas=(0.0005, 0.001, 0.002, 0.005, 0.01, 0.02),
                     face_sigmas=(0.05, 0.1, 0.2, 0.4, 0.8, 1.5, 3.0)
                     ) -> Dict[str, object]:
    body = [(sig, [forme_lock_failure(sig, n) for n in FORME_SIZES])
            for sig in body_sigmas]
    face = [(sig, face_identification_failure(net, sig)) for sig in face_sigmas]

    # the threshold at which each dimension first fails 10% of the time
    body_thr = next((sig for sig, r in body if r[-1] > 0.10), None)
    face_thr = next((sig for sig, r in face if r > 0.10), None)
    return {"body": body, "face": face,
            "body_threshold": body_thr, "face_threshold": face_thr,
            "ratio": (face_thr / body_thr) if (body_thr and face_thr) else None}


def blast_radius(net: PMFNet, va, sigma: float = 0.40, trials: int = 8
                 ) -> List[Tuple[str, float, float]]:
    """Second panel: damage ONE node at each depth of the chain and count how
    many sorts move.

    The result here is not the one the chain-of-negatives story predicts, and it
    is reported as it comes out. Because a matrix is a CONVEX BLEND of punches,
    a single spoiled punch is diluted across every descendant rather than
    concentrated in one, so it moves all V sorts but moves each of them only a
    little. A spoiled sort moves exactly one sort, but moves it the whole way.
    Sharing the masters buys reach at the top and, by the same mechanism, buys
    resistance to damage at the top. That is worth knowing and worth saying: it
    is an argument for elemental punches shared across letters -- which is what
    the clustering evidence points to -- and against the tidy assumption that
    upstream always means more dangerous.
    """
    base = evaluate(net, *va)
    saved = net.copy_params()
    E0 = net.build_faces()["E"].copy()
    rng = np.random.default_rng(99)
    rows = []
    for label, key in ((f"one punch  (1 of {net.cfg.n_punch})", "Punch"),
                       (f"one strike (1 of {net.cfg.n_matrix})", "Strike"),
                       (f"one sort   (1 of {net.V})", "SORT")):
        dls, brs = [], []
        for _ in range(trials):
            net.load_params(saved)
            if key == "SORT":
                v = int(rng.integers(net.V))
                net.p["Gain"][v] += (rng.standard_normal() * sigma
                                     * float(np.abs(saved["Gain"]).mean()))
                net.p["Cast"][v] += (rng.standard_normal(saved["Cast"].shape[1])
                                     * sigma * float(np.abs(saved["Cast"]).mean()))
            else:
                i = int(rng.integers(saved[key].shape[0]))
                net.p[key][i] += (rng.standard_normal(saved[key].shape[1])
                                  * sigma * float(np.abs(saved[key]).mean()))
            dls.append(evaluate(net, *va) - base)
            E1 = net.build_faces()["E"]
            brs.append(int((np.abs(E1 - E0).max(axis=1) > 1e-9).sum()))
        rows.append((label, float(np.mean(dls)), float(np.mean(brs))))
    net.load_params(saved)
    return rows


# ==============================================================================
# SECTION 6 -- STUDY B : DOES THE CONTAINER EDIT THE CONTENT?
# ==============================================================================
# Not measured by loss -- loss confounds this with the fact that the remaining
# measure doubles as a position-in-line clock -- but behaviourally, by holding
# the text fixed and moving nothing but the space left in the line.
#
# TEST 1 -- the response curve. Sweep the remaining measure from a fresh line to
#           a nearly full one and watch the probability mass the model puts on
#           the abbreviation sorts. If the coupling is real the curve rises.
#
# TEST 2 -- the counterfactual. Same context, same history, two values of the
#           remaining measure. How often does the model's chosen next symbol
#           actually change? That number is the size of the effect that the
#           shape of the page has on what gets said on it.

def container_response(net: PMFNet, va, levels=(0.90, 0.70, 0.50, 0.35, 0.22, 0.12, 0.06),
                       n: int = 40) -> Dict[str, List[Tuple[float, float]]]:
    xb, yb, rb = va[0][:n], va[1][:n], va[2][:n]
    out = {}
    for label, meas in (("full model", True), ("measure input removed", False)):
        net.use_measure = meas
        curve = []
        for rv in levels:
            _, ctx = net.forward(xb, yb, np.full_like(rb, rv))
            pmass = float(np.mean([pr[:, ABBREV_IDS].sum(axis=1).mean()
                                   for pr in ctx["probs"]]))
            curve.append((rv, pmass))
        out[label] = curve
    net.use_measure = True
    return out


def counterfactual_flip(net: PMFNet, va, loose: float = 0.85, tight: float = 0.10,
                        n: int = 60) -> Dict[str, float]:
    xb, yb, rb = va[0][:n], va[1][:n], va[2][:n]
    out = {}
    for label, meas in (("full model", True), ("measure input removed", False)):
        net.use_measure = meas
        _, c1 = net.forward(xb, yb, np.full_like(rb, loose))
        _, c2 = net.forward(xb, yb, np.full_like(rb, tight))
        out[label] = float(np.mean([(p1.argmax(1) != p2.argmax(1)).mean()
                                    for p1, p2 in zip(c1["probs"], c2["probs"])]))
    net.use_measure = True
    return out


# ==============================================================================
# SECTION 7 -- STUDY C : COLLATION, OR WHY AGREEMENT STOPPED BEING EVIDENCE
# ==============================================================================
# Two regimes for producing N witnesses of one text.
#
#   SCRIPTORIUM : each copy is made by a different hand from a different
#                 exemplar. Errors are independent. Majority vote across N
#                 witnesses drives the error rate down roughly exponentially,
#                 which is why collating manuscripts recovers an archetype.
#
#   PRESS       : all N copies come off one setting of type. Whatever the
#                 compositor got wrong is in every copy, identically. Extra
#                 copies add no information about the text at all; they only
#                 confirm the forme. Majority vote plateaus at the setting's own
#                 error rate no matter how large N gets.
#
# The corrective is not more copies but more SETTINGS: independent editions,
# ideally from independent exemplars. The third curve below shows what happens
# when you weight by settings instead of copies -- the only estimator that
# recovers the manuscript regime's behaviour under print.
#
# This is not a metaphor for training on synthetic data. It is the same
# arithmetic.

def collation_study(n_sym: int = 24, length: int = 400, p_err: float = 0.12,
                    n_settings: int = 6, seed: int = 1468
                    ) -> Dict[str, List[Tuple[int, float]]]:
    rng = np.random.default_rng(seed)
    truth = rng.integers(0, n_sym, size=length)

    def corrupt(src, p):
        out = src.copy()
        hit = rng.random(length) < p
        out[hit] = rng.integers(0, n_sym, size=int(hit.sum()))
        return out

    def vote(witnesses: np.ndarray) -> float:
        """Fraction of positions the plurality reading gets right."""
        rec = np.empty(length, dtype=np.int64)
        for j in range(length):
            rec[j] = np.bincount(witnesses[:, j], minlength=n_sym).argmax()
        return float((rec == truth).mean())

    settings = [corrupt(truth, p_err) for _ in range(n_settings)]

    scriptorium, press, discounted = [], [], []
    for n in (1, 3, 5, 9, 15, 25, 45, 81):
        # independent hands
        w = np.stack([corrupt(truth, p_err) for _ in range(n)])
        scriptorium.append((n, vote(w)))

        # n copies off ONE setting -- identical by construction
        w = np.stack([settings[0].copy() for _ in range(n)])
        press.append((n, vote(w)))

        # n copies spread over the available independent settings, then
        # de-duplicated to one vote per setting: the provenance discount
        used = min(n, n_settings)
        w = np.stack([settings[k] for k in range(used)])
        discounted.append((n, vote(w)))

    return {"scriptorium": scriptorium, "press": press,
            "provenance-discounted": discounted}


# ==============================================================================
# SECTION 8 -- STUDY D : THE STOP-PRESS PATCH
# ==============================================================================
# The forme is locked and the sheets are pulled. A fault is then found. You
# cannot go back into the base -- it is committed and already distributed. What
# the trade actually did was pull the forme, swap one sort, and carry on, which
# left two states of the same page in circulation, and for anything already sold
# they printed an errata leaf.
#
# Here: freeze every base parameter, then fit an additive logit patch under a
# hard budget of K non-zero entries. The base is verified unchanged afterwards.
# The point of the exercise is the shape of the constraint -- corrections after
# commitment are sparse, external, cheap, and auditable, and you can count them.

def stop_press_patch(net: PMFNet, tr, va, budget: int = 6, steps: int = 400,
                     lr: float = 0.15) -> Dict[str, float]:
    """Can a correction made after commitment repair a fault made before it?

    The sequence deliberately mirrors the workshop:
      1. the run is healthy                            -> base
      2. a punch turns out to have been cut wrong,
         and every sort descended from it inherits
         the fault                                     -> damaged
      3. the sheets are already printed, so the base
         is frozen and an errata correction is fitted
         under a hard budget of K entries              -> patched
      4. for comparison only, the punch is re-cut      -> recut

    The answer, which is the whole moral of the exercise, is in the recovery
    fraction: an out-of-band patch buys back some of the loss and cannot buy
    back the rest, because a bias applied to the output cannot undo a fault in
    the generator. The base parameters are verified untouched throughout.
    """
    xt, yt, rt = tr
    healthy = net.copy_params()
    base_val = evaluate(net, *va)

    # --- 2. cut one punch badly ------------------------------------------
    rng = np.random.default_rng(1457)
    bad = int(rng.integers(net.cfg.n_punch))
    net.p["Punch"][bad] += rng.standard_normal(net.cfg.d_face) * 0.9
    damaged_params = net.copy_params()
    damaged_val = evaluate(net, *va)

    # --- 3. errata: frozen base, sparse additive correction on the logits --
    patch = np.zeros(net.V)
    m = np.zeros(net.V); v = np.zeros(net.V)
    for step in range(1, steps + 1):
        sel = rng.choice(len(xt), size=min(24, len(xt)), replace=False)
        net._patch = patch
        _, ctx = net.forward(xt[sel], yt[sel], rt[sel])
        gp = np.zeros(net.V)
        B, T = len(sel), xt.shape[1]
        for t, pr in enumerate(ctx["probs"]):
            d = pr.copy()
            d[np.arange(B), yt[sel][:, t]] -= 1.0
            gp += d.sum(axis=0)
        gp /= (B * T)
        m = 0.9 * m + 0.1 * gp
        v = 0.999 * v + 0.001 * gp * gp
        patch -= lr * (m / (1 - 0.9 ** step)) / (np.sqrt(v / (1 - 0.999 ** step)) + 1e-8)
        if budget < net.V:                     # hard errata budget
            keep = np.argsort(-np.abs(patch))[:budget]
            mask = np.zeros(net.V, dtype=bool); mask[keep] = True
            patch[~mask] = 0.0
    net._patch = patch
    patched_val = evaluate(net, *va)
    net._patch = None

    drift = max(float(np.abs(net.p[k] - damaged_params[k]).max()) for k in net.p)

    # --- 4. re-cut the punch (upstream repair) ---------------------------
    net.load_params(healthy)
    recut_val = evaluate(net, *va)

    harm = damaged_val - base_val
    return {"base_val": base_val, "damaged_val": damaged_val,
            "patched_val": patched_val, "recut_val": recut_val,
            "bad_punch": bad, "nonzero": int((patch != 0).sum()),
            "base_drift": drift,
            "recovered_by_patch": (damaged_val - patched_val) / max(harm, 1e-12),
            "recovered_by_recut": (damaged_val - recut_val) / max(harm, 1e-12)}


# ==============================================================================
# SECTION 9 -- STUDY E : SETTING A LINE (qualitative sample)
# ==============================================================================

def set_a_line(net: PMFNet, n_lines: int = 4, seed: int = 1440) -> List[str]:
    """Generate text under a live width budget, exactly as the corpus was made:
    the model proposes, and the measure disposes."""
    rng = np.random.default_rng(seed)
    E = net.build_faces()["E"]
    lines = []
    h = np.zeros((1, net.cfg.d_hidden))
    cur = STOI[" "]
    for _ in range(n_lines):
        used, out = 0, []
        for _ in range(200):
            r = np.array([(MEASURE - used) / MEASURE])
            a = E[[cur]] @ net.p["Wx"] + h @ net.p["Wh"] \
                + np.outer(r, net.p["wr"]) + net.p["bh"]
            h = np.tanh(a)
            z = h @ net.p["Wo"] + net.p["bo"] + net.justification_bias(r)
            pr = softmax_rows(z)[0]
            # hard-forbid anything that will not physically fit
            pr = np.where(net.widths <= (MEASURE - used), pr, 0.0)
            if pr.sum() <= 0:
                break
            pr = pr / pr.sum()
            cur = int(rng.choice(net.V, p=pr))
            out.append(ITOS[cur])
            used += int(net.widths[cur])
        lines.append("".join(out))
    return lines


# ==============================================================================
# SECTION 10 -- SELF-TESTS
# ==============================================================================

def self_tests(net: PMFNet, tr, va, log=print) -> List[Tuple[str, bool, str]]:
    res: List[Tuple[str, bool, str]] = []

    def check(name, cond, detail=""):
        res.append((name, bool(cond), detail))

    # 1. no stored embedding table anywhere
    check("no V x D embedding parameter exists",
          not any(v.shape == (net.V, net.cfg.d_face) for v in net.p.values()),
          f"param blocks: {sorted(net.p.keys())}")

    # 2. parameter budget really is smaller than a table would be
    table = net.V * net.cfg.d_face
    chain = (net.p["Punch"].size + net.p["Strike"].size
             + net.p["Cast"].size + net.p["Gain"].size)
    check("chain of negatives is cheaper than the table it replaces",
          chain < table, f"chain {chain} < table {table}")

    # 3. every sort is reachable from at least one punch (no orphans)
    ch = net.build_faces()
    reach = (ch["c"] @ ch["s"])          # (V, P) sort -> punch mass
    check("every sort descends from the punches",
          bool(np.all(reach.sum(axis=1) > 0.999)),
          f"min mass {reach.sum(axis=1).min():.6f}")

    # 4. a change at a punch propagates to every sort (global correction)
    E0 = net.build_faces()["E"].copy()
    net.p["Punch"][0] += 0.5
    E1 = net.build_faces()["E"]
    net.p["Punch"][0] -= 0.5
    moved = int((np.abs(E1 - E0).max(axis=1) > 1e-9).sum())
    check("one punch edit moves every sort", moved == net.V,
          f"{moved}/{net.V} sorts moved")

    # 5. the feasibility penalty is never positive and bites when it should
    r_full = np.array([1.0]); r_tight = np.array([0.02])
    b_full, b_tight = net.justification_bias(r_full), net.justification_bias(r_tight)
    widest = int(np.argmax(net.widths))
    check("justification never rewards, and forbids the widest sort on a full line",
          b_full.max() <= 1e-12 and b_tight[0, widest] < -1.0,
          f"widest sort '{ITOS[widest]}' bias {b_tight[0, widest]:.2f}")

    # 6. abbreviation sorts are genuinely narrower than what they replace
    check("abbreviation sorts are narrower than the letters they swallow",
          all(SORTS[a] < SORTS['m'] for a in ABBREV_SORTS),
          f"{ {a: SORTS[a] for a in ABBREV_SORTS} } vs m={SORTS['m']}")

    # 7. loss is finite and near ln(V) at initialisation for a fresh net
    fresh = PMFNet(PMFConfig(seed=1234))
    l0, _ = fresh.forward(va[0][:8], va[1][:8], va[2][:8], cache=False)
    check("fresh network starts near uniform entropy",
          abs(l0 - math.log(net.V)) < 1.2, f"loss {l0:.3f} vs ln(V) {math.log(net.V):.3f}")

    # 8. body error really does compound with forme size
    f_small = forme_lock_failure(0.005, 15)
    f_big = forme_lock_failure(0.005, 2600)
    check("body error compounds with forme size", f_big > 10 * max(f_small, 1e-6),
          f"15 sorts {f_small:.4f} -> 2600 sorts {f_big:.4f}")

    # 9. the measure genuinely changes the model's choices
    fl = counterfactual_flip(net, va)
    check("remaining measure changes the chosen sort",
          fl["full model"] > 5 * fl["measure input removed"],
          f"full {fl['full model']*100:.1f}% vs ablated "
          f"{fl['measure input removed']*100:.1f}%")

    # 10. determinism
    a, _ = net.forward(va[0][:4], va[1][:4], va[2][:4], cache=False)
    b, _ = net.forward(va[0][:4], va[1][:4], va[2][:4], cache=False)
    check("forward pass is deterministic", a == b, f"{a!r} == {b!r}")

    for name, ok, detail in res:
        log(f"    [{'PASS' if ok else 'FAIL'}] {name}"
            + (f"   ({detail})" if detail else ""))
    return res


# ==============================================================================
# SECTION 11 -- MAIN
# ==============================================================================

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="PMF-Net -- figure 0400")
    ap.add_argument("--quick", action="store_true", help="fewer epochs / smaller corpus")
    args = ap.parse_args(argv)

    t0 = time.time()
    W = 78
    def rule(c="="): print(c * W)
    def head(s): rule(); print(s); rule()

    head(" PMF-NET  --  THE PUNCH-MATRIX-FORME NETWORK\n"
         " Figure 0400 : Johannes Gensfleisch zur Laden zum Gutenberg (c.1400-1468)")

    rng = np.random.default_rng(1450)
    n_tr = 2000 if args.quick else 6000
    n_va = 1200

    print("\n[1] SETTING THE EXEMPLAR")
    tr_stream, tr_rem = compose_corpus(n_tr, rng)
    va_stream, va_rem = compose_corpus(n_va, rng)
    cfg = PMFConfig(seq_len=48 if args.quick else 64)
    tr = make_batches(tr_stream, tr_rem, cfg.seq_len)
    va = make_batches(va_stream, va_rem, cfg.seq_len)
    print(f"    type case        : {V} sorts "
          f"({len(BASE_SORTS)} letters/space, {len(ABBREV_SORTS)} abbreviations)")
    print(f"    measure          : {MEASURE} units ({MEASURE/EM_UNITS:.0f} ems)")
    print(f"    corpus           : {len(tr_stream):,} sorts train / "
          f"{len(va_stream):,} val")
    print(f"    windows          : {tr[0].shape} train, {va[0].shape} val")
    ab_rate = np.isin(np.array(tr_stream), ABBREV_IDS).mean()
    print(f"    abbreviation rate: {ab_rate*100:.2f}% of all sorts set")

    net = PMFNet(cfg)
    print("\n[2] THE CHAIN OF NEGATIVES")
    print(f"    punches          : {cfg.n_punch}  ({net.p['Punch'].size} params)")
    print(f"    matrices         : {cfg.n_matrix} ({net.p['Strike'].size} params)")
    print(f"    sorts            : {V}  (routed by {net.p['Cast'].size} params "
          f"+ {net.p['Gain'].size} set-widths)")
    print(f"    composing stick  : {cfg.d_hidden} units")
    print(f"    total parameters : {net.n_params():,}")
    chain_cost = (net.p['Punch'].size + net.p['Strike'].size
                  + net.p['Cast'].size + net.p['Gain'].size)
    print(f"    face cost        : chain {chain_cost} vs stored table {V*cfg.d_face}")
    big = 290   # the size of the actual B42 fount: 47 capitals, 63 lowercase,
                # 92 abbreviations, 83 ligatures, 5 marks
    print(f"    at a {big}-sort fount : chain "
          f"{net.p['Punch'].size + net.p['Strike'].size + big*cfg.n_matrix + big}"
          f" vs stored table {big*cfg.d_face}")

    print("\n[3] GRADIENT CHECK  (central differences, eps=1e-6)")
    worst, report = gradient_check(net, tr[0][:4], tr[1][:4], tr[2][:4])
    for name, rel, absd in report:
        print(f"    {name:<7} max rel err {rel:.3e}   max abs err {absd:.3e}")
    ok = worst < 1e-5
    print(f"    WORST OVER ALL BLOCKS : {worst:.3e}   -> "
          f"{'PASS' if ok else 'FAIL'}")
    if not ok:
        print("    gradient check failed; aborting.")
        return 1

    print("\n[4] TRAINING")
    hist = train(net, tr, va, epochs=8 if args.quick else 16,
                 bs=32, lr=1e-2, log=print)
    first, last = hist[0], hist[-1]
    print(f"    val loss {first[2]:.4f} -> {last[2]:.4f}  "
          f"(perplexity {math.exp(first[2]):.2f} -> {math.exp(last[2]):.2f}, "
          f"uniform baseline {V})")

    print("\n[5] STUDY A -- THE TOLERANCE BUDGET")
    tb = tolerance_budget(net)
    print("\n    THE BODY: fraction of formes that will not lock.")
    print("    (columns are sorts per forme; a B42 column runs to ~2,600)\n")
    print(f"    {'width error':>12}" + "".join(f"{n:>10}" for n in FORME_SIZES))
    for sig, row in tb["body"]:
        print(f"    {sig:>11.4f} " + "".join(f"{v:>10.3f}" for v in row))
    print("\n    THE FACE: fraction of letters a reader can no longer identify.\n")
    for sig, v in tb["face"]:
        bar = "#" * int(round(v * 40))
        print(f"    shape error {sig:>5.2f}   misread {v*100:>6.2f}%  {bar}")
    print(f"\n    first 10% failure -- body at {tb['body_threshold']}, "
          f"face at {tb['face_threshold']}")
    if tb["ratio"]:
        print(f"    THE FACE TOLERATES {tb['ratio']:.0f}x THE ERROR THE BODY DOES.")
    print("    Body error adds along the line and destroys the page; face error")
    print("    stops at the glyph and is invisible. Precision belongs in the")
    print("    dimensions where mistakes compose, and nowhere else.")

    print("\n    Second panel -- damage one node at each depth of the chain:\n")
    print(f"    {'damaged node':<26}{'delta loss':>13}{'sorts moved':>14}")
    for label, dl, br in blast_radius(net, va):
        print(f"    {label:<26}{dl:>13.4f}{br:>10.1f}/{V}")

    print("\n[6] STUDY B -- DOES THE CONTAINER EDIT THE CONTENT?")
    print("    probability mass on the abbreviation sorts as the line fills,")
    print("    with the text held fixed and only the space left changing.\n")
    cr = container_response(net, va)
    lv = [r for r, _ in cr["full model"]]
    print("    " + " " * 24 + "".join(f"{r:>8.2f}" for r in lv))
    for label, curve in cr.items():
        print(f"    {label:<24}" + "".join(f"{p:>8.3f}" for _, p in curve))
    fm = cr["full model"]
    print(f"\n    the full model raises abbreviation mass "
          f"{fm[-1][1]/max(fm[0][1],1e-9):.2f}x as the line closes;")
    ab = cr["measure input removed"]
    print(f"    with the measure removed the same sweep moves it "
          f"{ab[-1][1]/max(ab[0][1],1e-9):.2f}x.")

    flips = counterfactual_flip(net, va)
    print("\n    counterfactual: identical context, only the space left changes.")
    for label, f in flips.items():
        print(f"      {label:<24} next sort changes {f*100:>5.1f}% of the time")
    print(f"    Nothing about the language moved. The shape of the container")
    print(f"    rewrote {flips['full model']*100:.0f}% of the choices made inside it.")

    print("\n[7] STUDY C -- COLLATION: WHEN AGREEMENT STOPS BEING EVIDENCE")
    coll = collation_study()
    print(f"\n    {'witnesses':>10}{'scriptorium':>14}{'one press run':>15}"
          f"{'discounted':>13}")
    for i in range(len(coll["scriptorium"])):
        n = coll["scriptorium"][i][0]
        print(f"    {n:>10}{coll['scriptorium'][i][1]:>14.3f}"
              f"{coll['press'][i][1]:>15.3f}"
              f"{coll['provenance-discounted'][i][1]:>13.3f}")
    print("\n    (fraction of readings recovered correctly; true text has an")
    print("     underlying per-symbol corruption rate of 0.12 in every regime)")

    print("\n[8] STUDY D -- THE STOP-PRESS PATCH")
    sp = stop_press_patch(net, tr, va, budget=6, steps=200 if args.quick else 400)
    print(f"    1. healthy run                  : {sp['base_val']:.4f}")
    print(f"    2. punch #{sp['bad_punch']} found badly cut     : "
          f"{sp['damaged_val']:.4f}")
    print(f"    3. errata patch, {sp['nonzero']} corrections   : "
          f"{sp['patched_val']:.4f}   "
          f"({sp['recovered_by_patch']*100:.1f}% of the harm recovered)")
    print(f"    4. punch re-cut instead         : {sp['recut_val']:.4f}   "
          f"({sp['recovered_by_recut']*100:.1f}% recovered)")
    print(f"    base parameters moved by        : {sp['base_drift']:.2e}  "
          f"(sheets already pulled stay pulled)")
    print(f"\n    A correction applied downstream of a locked forme cannot reach")
    print(f"    a fault that lives upstream of it. It can only mask part of it.")

    print("\n[9] STUDY E -- SETTING A LINE UNDER THE MEASURE")
    for i, line in enumerate(set_a_line(net, 4), 1):
        used = sum(SORTS[c] for c in line)
        print(f"    line {i}: {line!r}")
        print(f"             {used}/{MEASURE} units filled")

    print("\n[10] SELF-TESTS")
    res = self_tests(net, tr, va)
    n_pass = sum(1 for _, ok, _ in res if ok)

    rule()
    print(f" gradient check   : {'PASS' if ok else 'FAIL'}  ({worst:.2e})")
    print(f" self-tests       : {n_pass}/{len(res)} passed")
    print(f" final val ppl    : {math.exp(last[2]):.3f}  (uniform = {V})")
    print(f" elapsed          : {time.time()-t0:.1f}s")
    rule()
    return 0 if (ok and n_pass == len(res)) else 1


if __name__ == "__main__":
    sys.exit(main())
