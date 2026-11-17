#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 LATITUDO ENGINE  --  a from-scratch cognitive architecture after Nicole Oresme
                      (Normandy, c.1320 -- Lisieux, 11 July 1382)
 # BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0371 · Nicole Oresme (1323-1382)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 19, Minds 361-380 Available on Amazon https://www.amazon.com/dp/B0HQX147Y7
# Resume and Interactive Demos at https://artificiology.com/
# END ATTRIBUTION 
================================================================================

WHY THIS ARCHITECTURE AND NOT ANOTHER
-------------------------------------
Oresme's one irreducible cognitive move, made in the *Tractatus de
configurationibus qualitatum et motuum* (c.1350s), is this: he refused to let a
quality be summarised by a number.  A hot thing is not "hot to degree 7".  It is
hot ACCORDING TO A FIGURE -- an intensity raised as a perpendicular over every
point of the subject's extension.  The number 7 is only the AREA under that
figure, what he calls the *quantitas qualitatis*, and the whole treatise is a
demonstration that the area is not the thing.  Two subjects can carry exactly the
same quantity of exactly the same quality and behave differently, because the
*configuratio* -- the shape of the distribution -- differs.

Every mainstream neural unit in 2026 does the opposite.  It sums its inputs into
one scalar, applies a nonlinearity, and emits one scalar.  A scalar is Oresme's
area.  The architecture below replaces that scalar with a configuration:

    every signal is a distribution over an internal extension,
    every weight is a distribution over the same extension,
    and the classical scalar neuron reappears exactly as the special case
    in which all weights happen to be UNIFORM.

Uniformity, for Oresme, is the measure-zero case.  So the usual network is the
degenerate corner of this one.

SEVEN MECHANISMS, EACH TRACEABLE TO A TEXT
------------------------------------------
 1. LATITUDO / CONFIGURATIO  (De configurationibus I.1-I.11)
    Signals and weights are vectors over a discretised extension of E cells.
    Coupling is the integral of their product -- Oresme's "compounding" of a
    quality with the subject it acts on.

 2. THE MERTON SPLIT  (De configurationibus III.7; the mean-speed theorem)
    Oresme gave the first geometric proof that a uniformly difform latitude has
    exactly the effect of a uniform one at its middle degree.  Written as an
    identity on the inner product this is EXACT and free:
          <w, phi>  =  wbar * phibar  +  <dw, dphi>
    Term one is the QUANTITAS channel (mean weight times area).  Term two is the
    CONFIGURATIO channel (shape against shape).  Every decision the network makes
    is therefore auditable into "how much" versus "how shaped" without any
    approximation and without a probe network.

 3. THE PROPER FIGURE WITH LOCKED MEASURE  (De moneta, c.1355)
    Each unit emits h * g, where h is its computed magnitude and g is its own
    learnable *proper figure*.  g is constrained to unit area by construction.
    A unit that could inflate the area of its own figure would be silently
    changing the unit of account in which every other unit reasons -- which is
    precisely Oresme's charge against the prince who alters the coinage: the
    fraud is not the change, it is that the change is hidden and universal.  The
    measure is hash-locked and re-verified after training.

 4. THE INCOMMENSURABILITY PRIOR  (De proportionibus proportionum, c.1356;
    De commensurabilitate vel incommensurabilitate motuum celi)
    Oresme argued by counting that two unknown ratios are far more likely to be
    incommensurable than commensurable, and drew the consequence that celestial
    configurations never exactly recur.  We reconstruct the counting argument
    exactly (parallel prime-exponent vectors) and use its consequence
    architecturally: the world model is driven by incommensurable frequencies,
    exact-match retrieval provably returns nothing, and the system must
    interpolate over the shape continuum instead of recognising.

 5. THE GAUGE OF THE FRAME  (Le livre du ciel et du monde II.25, 1377)
    Oresme's supposition: local motion is perceptible only in so far as one body
    is seen to stand differently with respect to another.  Here that becomes a
    provable symmetry: rotate the extension of every signal by tau and rotate
    every weight-configuration by tau, and every output is bit-identical.  The
    frame parameter is UNIDENTIFIABLE FROM BEHAVIOUR.  The system reports this
    rather than guessing, then fixes tau by declared convention and logs it as a
    convention -- which is what Oresme did when, having refuted every argument
    against a turning Earth, he kept the still one anyway.

 6. THE GROUPING TEST  (Questiones super geometriam Euclidis, Qu.2)
    Oresme was the first to prove the harmonic series divergent, by grouping its
    terms into blocks each exceeding one half.  Increments that individually
    vanish need not have a bounded sum.  We run his proof as a live monitor over
    training-time capability increments, distinguishing convergent accumulation
    from divergent accumulation -- a distinction unreadable from the size of any
    single step.

 7. DE CAUSIS MIRABILIUM  (c.1370, ed. Hansen 1985)
    Oresme's programme against marvels: before invoking the heavens, demons or
    God, exhaust defect of sense, over-vigorous imagination, and faulty
    inference.  Implemented as a three-source ablation on any surprising output,
    with "mirabile" reserved for the residue that survives all three.

RUNNING CONVENTIONS
-------------------
Pure NumPy.  Analytic gradients with a mandatory finite-difference check.  A real
training loop on a task that is impossible for the scalar summary and easy for
the configuration.  Self-tests for every claim above.  No frameworks, no
attention over stored keys, no transformer.

Author's note: the task below is built so that EVERY sample carries exactly the
same quantitas qualitatis.  The area channel therefore contains, provably, zero
bits.  That is the experiment Oresme would have wanted run.
================================================================================
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

# ------------------------------------------------------------------------------
# Global conventions
# ------------------------------------------------------------------------------
SEED = 1382                      # the year of his death at Lisieux
E_CELLS = 32                     # cells of the internal extension (longitudo)
DX = 1.0 / E_CELLS               # so that a unit-area figure integrates to 1
RNG = np.random.default_rng(SEED)

# Pairwise incommensurable generators.  Oresme's point in De commensurabilitate
# is not that irrationals exist but that they are the GENERIC case, so a world
# driven by them never returns to a state it has already occupied.
INCOMMENSURABLE_OMEGA = np.array([math.sqrt(2.0), math.sqrt(3.0), math.sqrt(5.0)])


# ==============================================================================
# SECTION 1 -- PRIMITIVES OF THE LATITUDE
# ==============================================================================

def softplus(z: np.ndarray) -> np.ndarray:
    """Numerically stable softplus.  Used to keep proper figures strictly
    positive: an intensity, for Oresme, is raised as a perpendicular and a
    perpendicular has no negative length."""
    return np.where(z > 30.0, z, np.log1p(np.exp(np.minimum(z, 30.0))))


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Derivative of softplus."""
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def quantitas(phi: np.ndarray) -> np.ndarray:
    """The QUANTITY OF THE QUALITY: the area under the figure.

    Oresme, De configurationibus I.11 -- the measure of a quality is the area of
    its configuration, not the height of any one of its perpendiculars.  This is
    the single scalar that a conventional neuron would have kept.  The entire
    architecture exists to show what is thrown away when only this survives.
    """
    return np.sum(phi, axis=-1) * DX


def merton_split(w: np.ndarray, phi: np.ndarray) -> Tuple[float, float]:
    """THE MEAN-SPEED DECOMPOSITION -- exact, not approximate.

    Oresme's geometric proof of the Merton rule (De configurationibus III.7)
    says a uniformly difform latitude is equivalent in total effect to the
    uniform latitude at its middle degree.  The algebraic shadow of that proof
    is an exact orthogonal split of any coupling into a QUANTITY term and a
    CONFIGURATION term:

        <w, phi> dx  =  mean(w) * area(phi)  +  <w - mean(w), phi - mean(phi)> dx

    because the cross terms integrate to zero.  Returns (quantitas_term,
    configuratio_term); their sum is the coupling to machine precision.
    """
    wbar = float(np.mean(w))
    phibar = float(np.mean(phi))
    quant = wbar * (phibar * E_CELLS * DX)          # mean weight x area
    config = float(np.sum((w - wbar) * (phi - phibar)) * DX)
    return quant, config


def roll_extension(a: np.ndarray, tau: int) -> np.ndarray:
    """Rotate a configuration along the extension.  The extension is CIRCULAR --
    this is what makes the frame symmetry of Livre du ciel II.25 exact rather
    than approximate, since a rotation has no edge at which it could be caught.
    """
    return np.roll(a, tau, axis=-1)


# ==============================================================================
# SECTION 2 -- THE TASK: FOUR FIGURES OF EQUAL QUANTITY
# ==============================================================================
#
# Oresme's four species of configuration, from De configurationibus I.10-I.16:
#
#   0  uniformis                    -- the same degree throughout
#   1  uniformiter difformis, ascendens  -- intensity rising evenly
#   2  uniformiter difformis, descendens -- intensity falling evenly
#   3  difformiter difformis        -- difform in a difform way (two summits)
#
# EVERY sample is renormalised to area exactly 1.0.  So the quantitas channel
# is, by construction, informationless.  A mind that keeps only the number is at
# chance; a mind that keeps the figure is not.  This is the experiment.
# ------------------------------------------------------------------------------

CLASS_NAMES = ["uniformis",
               "uniformiter difformis ascendens",
               "uniformiter difformis descendens",
               "difformiter difformis"]


def _base_figure(kind: int, x: np.ndarray) -> np.ndarray:
    if kind == 0:
        return np.ones_like(x)
    if kind == 1:
        return 0.25 + 1.60 * x
    if kind == 2:
        return 0.25 + 1.60 * (1.0 - x)
    if kind == 3:
        return 0.25 + 1.10 * (1.0 + np.cos(4.0 * np.pi * x))
    raise ValueError(kind)


def make_corpus(n: int, noise: float = 0.30, max_shift: int = 8,
                rng: np.random.Generator | None = None
                ) -> Tuple[np.ndarray, np.ndarray]:
    """Generate n configurations with labels, all of identical area."""
    rng = rng or RNG
    x = np.arange(E_CELLS) / E_CELLS
    Phi = np.empty((n, E_CELLS))
    y = rng.integers(0, 4, size=n)
    for i in range(n):
        f = _base_figure(int(y[i]), x)
        # per-sample jitter: the degree of difformity varies, the species does not
        f = f * (1.0 + 0.22 * rng.standard_normal())
        f = np.roll(f, int(rng.integers(-max_shift, max_shift + 1)))
        f = f + noise * rng.standard_normal(E_CELLS)
        f = np.clip(f, 0.02, None)
        f = f / (np.sum(f) * DX)                 # <-- area locked to 1 exactly
        Phi[i] = f
    return Phi, y.astype(np.int64)


# ==============================================================================
# SECTION 3 -- THE NETWORK
# ==============================================================================

@dataclass
class Latitudo:
    """A network whose signals and weights are configurations, not numbers.

    Shapes
    ------
    W1     (H1, E)         weight-configurations reading the input figure
    b1     (H1,)
    G1raw  (H1, E)         raw parameters of each unit's PROPER FIGURE
    W2     (H2, H1, E)     weight-configurations reading layer-1 emissions
    b2     (H2,)
    G2raw  (H2, E)
    U      (C, H2, E)      readout weight-configurations
    bc     (C,)

    The proper figures g = softplus(Graw) / integral(softplus(Graw)) always have
    unit area.  That constraint is the De moneta lock: a unit may change the
    SHAPE of what it emits but never the total measure it emits it in.
    """
    H1: int = 16
    H2: int = 10
    C: int = 4

    W1: np.ndarray = field(default=None)
    b1: np.ndarray = field(default=None)
    G1raw: np.ndarray = field(default=None)
    W2: np.ndarray = field(default=None)
    b2: np.ndarray = field(default=None)
    G2raw: np.ndarray = field(default=None)
    U: np.ndarray = field(default=None)
    bc: np.ndarray = field(default=None)

    # ---------------------------------------------------------------- init ---
    def __post_init__(self):
        if self.W1 is not None:
            return
        r = np.random.default_rng(SEED + 7)
        # Weight configurations are seeded from incommensurable harmonics.  Two
        # units therefore never share a period, and no pair of proper figures can
        # be brought into exact coincidence by any shift -- Oresme's celestial
        # non-recurrence, moved indoors.
        x = np.arange(E_CELLS) / E_CELLS
        def harmonic_init(shape_lead: int, scale: float) -> np.ndarray:
            rows = []
            for k in range(shape_lead):
                w = np.zeros(E_CELLS)
                for j, om in enumerate(INCOMMENSURABLE_OMEGA):
                    amp = r.standard_normal()
                    ph = r.uniform(0, 2 * np.pi)
                    w += amp * np.cos(2 * np.pi * om * (k + 1) * x + ph)
                rows.append(w)
            A = np.array(rows)
            return A * (scale / (np.std(A) + 1e-12))

        self.W1 = harmonic_init(self.H1, 1.4)
        self.b1 = np.zeros(self.H1)
        self.G1raw = 0.35 * r.standard_normal((self.H1, E_CELLS))
        self.W2 = (harmonic_init(self.H2 * self.H1, 0.9)
                   .reshape(self.H2, self.H1, E_CELLS))
        self.b2 = np.zeros(self.H2)
        self.G2raw = 0.35 * r.standard_normal((self.H2, E_CELLS))
        self.U = (harmonic_init(self.C * self.H2, 0.9)
                  .reshape(self.C, self.H2, E_CELLS))
        self.bc = np.zeros(self.C)

    # ------------------------------------------------------------- packing ---
    @property
    def param_names(self) -> List[str]:
        return ["W1", "b1", "G1raw", "W2", "b2", "G2raw", "U", "bc"]

    def get(self, name: str) -> np.ndarray:
        return getattr(self, name)

    def clone(self) -> "Latitudo":
        return Latitudo(self.H1, self.H2, self.C,
                        *[self.get(n).copy() for n in self.param_names])

    # ------------------------------------------------------- proper figures ---
    def proper_figures(self) -> Tuple[np.ndarray, np.ndarray,
                                      np.ndarray, np.ndarray]:
        """Return (g1, g2, Z1, Z2).  Each row of g integrates to exactly 1."""
        s1 = softplus(self.G1raw)
        Z1 = np.sum(s1, axis=1) * DX
        g1 = s1 / Z1[:, None]
        s2 = softplus(self.G2raw)
        Z2 = np.sum(s2, axis=1) * DX
        g2 = s2 / Z2[:, None]
        return g1, g2, Z1, Z2

    # ------------------------------------------------------------- forward ---
    def forward(self, Phi: np.ndarray) -> Dict[str, np.ndarray]:
        """Phi: (N, E) input configurations."""
        g1, g2, Z1, Z2 = self.proper_figures()

        # Layer 1: couple the input figure against each unit's weight-figure.
        A1 = (Phi @ self.W1.T) * DX                      # (N,H1)
        pre1 = A1 + self.b1
        h1 = np.tanh(pre1)
        # emission of unit h is h1[h] * g1[h,:]  -- magnitude times proper figure

        # Layer 2: because emissions factor as (scalar x figure), the coupling
        # collapses to a small matrix M2 of figure-against-figure overlaps.
        M2 = np.einsum("mhj,hj->mh", self.W2, g1) * DX     # (H2,H1)
        pre2 = h1 @ M2.T + self.b2
        h2 = np.tanh(pre2)

        # Readout: same construction once more.
        K = np.einsum("cmj,mj->cm", self.U, g2) * DX       # (C,H2)
        logits = h2 @ K.T + self.bc

        return dict(Phi=Phi, g1=g1, g2=g2, Z1=Z1, Z2=Z2,
                    A1=A1, pre1=pre1, h1=h1, M2=M2,
                    pre2=pre2, h2=h2, K=K, logits=logits)

    # ---------------------------------------------------------------- loss ---
    @staticmethod
    def softmax_ce(logits: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
        z = logits - logits.max(axis=1, keepdims=True)
        ez = np.exp(z)
        p = ez / ez.sum(axis=1, keepdims=True)
        n = logits.shape[0]
        loss = float(-np.mean(np.log(p[np.arange(n), y] + 1e-15)))
        dlogits = p.copy()
        dlogits[np.arange(n), y] -= 1.0
        return loss, dlogits / n

    def loss_and_grads(self, Phi: np.ndarray, y: np.ndarray
                       ) -> Tuple[float, Dict[str, np.ndarray]]:
        c = self.forward(Phi)
        loss, dlogits = self.softmax_ce(c["logits"], y)

        # -- readout -----------------------------------------------------------
        dbc = dlogits.sum(0)
        dK = dlogits.T @ c["h2"]                              # (C,H2)
        dh2 = dlogits @ c["K"]                                # (N,H2)
        dU = dK[:, :, None] * c["g2"][None, :, :] * DX        # (C,H2,E)
        dg2 = np.einsum("cm,cmj->mj", dK, self.U) * DX        # (H2,E)

        # -- layer 2 -----------------------------------------------------------
        dpre2 = dh2 * (1.0 - c["h2"] ** 2)
        db2 = dpre2.sum(0)
        dM2 = dpre2.T @ c["h1"]                               # (H2,H1)
        dh1 = dpre2 @ c["M2"]                                 # (N,H1)
        dW2 = dM2[:, :, None] * c["g1"][None, :, :] * DX      # (H2,H1,E)
        dg1 = np.einsum("mh,mhj->hj", dM2, self.W2) * DX      # (H1,E)

        # -- layer 1 -----------------------------------------------------------
        dpre1 = dh1 * (1.0 - c["h1"] ** 2)
        db1 = dpre1.sum(0)
        dW1 = (dpre1.T @ Phi) * DX                            # (H1,E)

        # -- through the unit-area normalisation (the De moneta lock) ----------
        # g = s/Z with Z = sum(s)*dx  =>  dg/ds = (I - g*dx)/Z
        def unnormalise(dg, g, Z, Graw):
            inner = np.sum(dg * g, axis=1, keepdims=True) * DX
            ds = (dg - inner) / Z[:, None]
            return ds * sigmoid(Graw)
        dG1raw = unnormalise(dg1, c["g1"], c["Z1"], self.G1raw)
        dG2raw = unnormalise(dg2, c["g2"], c["Z2"], self.G2raw)

        return loss, dict(W1=dW1, b1=db1, G1raw=dG1raw,
                          W2=dW2, b2=db2, G2raw=dG2raw,
                          U=dU, bc=dbc)


# ==============================================================================
# SECTION 4 -- THE QUANTITAS BASELINE (the mind that kept only the number)
# ==============================================================================

class QuantitasOnly:
    """Multinomial logistic regression on a single feature: the area.

    This is the strawman Oresme actually argued against -- the reduction of a
    quality to its quantity.  Because every sample in the corpus has area 1.0 to
    machine precision, this model is provably at chance.  It is included not to
    be beaten but to make the information content of the discarded shape visible
    as a number.
    """

    def __init__(self, C: int = 4):
        self.w = np.zeros(C)
        self.b = np.zeros(C)

    def fit(self, Phi, y, epochs=400, lr=0.5):
        a = quantitas(Phi)
        n = len(y)
        for _ in range(epochs):
            logits = a[:, None] * self.w[None, :] + self.b[None, :]
            z = logits - logits.max(1, keepdims=True)
            p = np.exp(z); p /= p.sum(1, keepdims=True)
            d = p.copy(); d[np.arange(n), y] -= 1.0; d /= n
            self.w -= lr * (d * a[:, None]).sum(0)
            self.b -= lr * d.sum(0)
        return self

    def accuracy(self, Phi, y):
        a = quantitas(Phi)
        logits = a[:, None] * self.w[None, :] + self.b[None, :]
        return float(np.mean(np.argmax(logits, 1) == y))


# ==============================================================================
# SECTION 5 -- ORESME'S OWN INSTRUMENTS, RUN AS CODE
# ==============================================================================

def oresme_grouping_test(terms: np.ndarray, blocks: int = 6
                         ) -> Dict[str, object]:
    """ORESME'S PROOF OF 1350, RUN AS A MONITOR.

    Questiones super geometriam Euclidis, Qu.2: the harmonic series is infinite
    because it contains infinitely many groups each greater than one half --
    1/3+1/4 > 1/2, then 1/5+...+1/8 > 1/2, then 1/9+...+1/16 > 1/2, and so on
    without end.  The terms vanish; the sum does not.

    We apply exactly that procedure to any sequence of increments.  Blocks are
    the dyadic ranges [2^k, 2^(k+1)).  If the block sums fail to shrink, the
    accumulation is DIVERGENS: the fact that each step became negligible tells
    you nothing about where the total goes.
    """
    terms = np.asarray(terms, dtype=float)
    sums = []
    for k in range(blocks):
        lo, hi = 2 ** k, min(2 ** (k + 1), len(terms))
        if lo >= len(terms):
            break
        sums.append(float(np.sum(terms[lo:hi])))
    if len(sums) < 3:
        return dict(verdict="indeterminatum", block_sums=sums, ratio=float("nan"))
    early = float(np.mean(sums[:max(1, len(sums) // 2)]))
    late = float(np.mean(sums[len(sums) // 2:]))
    ratio = late / (early + 1e-18)
    verdict = "divergens" if ratio > 0.45 else "convergens"
    return dict(verdict=verdict, block_sums=sums, ratio=ratio)


def _prime_exponent_vector(num: int, den: int, primes: List[int]) -> np.ndarray:
    v = np.zeros(len(primes), dtype=np.int64)
    for sign, m in ((1, num), (-1, den)):
        for i, p in enumerate(primes):
            while m % p == 0:
                v[i] += sign
                m //= p
        if m != 1:
            return None
    return v


def incommensurability_census(N: int = 12) -> Dict[str, float]:
    """ORESME'S COUNTING ARGUMENT, De proportionibus proportionum III.

    Two ratios A and B are COMMENSURABLE IN RATIO when one is a rational power
    of the other -- when there is a common ratio C with A = C^i, B = C^j.  In
    prime-exponent coordinates that is exactly the condition that the exponent
    vectors of A and B be PARALLEL.  Oresme, who had no such coordinates, argued
    the same conclusion by enumerating cases and finding the commensurable ones
    a vanishing minority.

    We enumerate every ratio n/m in lowest terms with 1 <= m < n <= N and count
    the fraction of unordered pairs whose exponent vectors are parallel.  The
    fraction falls as the pool grows: incommensurability is the generic case,
    which is why, Oresme concluded, no celestial configuration ever returns.
    """
    primes = [p for p in range(2, N + 1)
              if all(p % q for q in range(2, int(p ** 0.5) + 1))]
    vecs = []
    for n in range(2, N + 1):
        for m in range(1, n):
            if math.gcd(n, m) != 1:
                continue
            v = _prime_exponent_vector(n, m, primes)
            if v is not None:
                vecs.append(v)
    total = 0
    comm = 0
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            total += 1
            a, b = vecs[i], vecs[j]
            # parallel iff every 2x2 minor vanishes
            if not np.any(np.outer(a, b) - np.outer(b, a)):
                comm += 1
    return dict(pool=len(vecs), pairs=total,
                commensurable=comm,
                fraction_commensurable=comm / max(total, 1))


def recurrence_probe(steps: int = 4000) -> Dict[str, float]:
    """NO STATE EVER RETURNS  (De commensurabilitate, the debate of Arithmetic
    and Geometry).  Three hands turning at pairwise incommensurable rates come
    arbitrarily close to their opening position and never reach it.  We report
    the closest approach after step 0 -- small, and provably never zero."""
    t = np.arange(1, steps + 1)[:, None]
    ang = (t * INCOMMENSURABLE_OMEGA[None, :]) % 1.0
    d = np.linalg.norm(np.minimum(ang, 1.0 - ang), axis=1)
    return dict(closest_return=float(d.min()),
                exact_returns=int(np.sum(d == 0.0)),
                steps=steps)


def retrieval_gate(query: np.ndarray, memory: np.ndarray,
                   tol: float = 1e-12) -> int:
    """Exact-match recall over stored configurations.  Returns the number of
    hits, which on a continuum of shapes is zero.  The gate exists to be seen
    failing: a mind in Oresme's world cannot answer by recognising, because the
    configuration in front of it has never occurred before."""
    d = np.abs(memory - query[None, :]).max(axis=1)
    return int(np.sum(d < tol))


def measure_hash(model: Latitudo) -> str:
    """MUTATIO MONETAE DETECTOR.  Hashes the invariants that define the unit of
    account: the extension, the cell width, and the unit-area constraint on
    every proper figure.  De moneta ch.15-17: the prince's gain from altering
    the coin is the community's loss, and the offence is that the alteration is
    silent.  Training may change every shape; it may not change the measure."""
    g1, g2, _, _ = model.proper_figures()
    areas = np.concatenate([np.sum(g1, 1) * DX, np.sum(g2, 1) * DX])
    payload = json.dumps(dict(E=E_CELLS, dx=round(DX, 15),
                              unit_areas=[round(float(a), 9) for a in areas]),
                         sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ==============================================================================
# SECTION 6 -- GRADIENT CHECK (mandatory)
# ==============================================================================

def _loss_only(model: Latitudo, Phi: np.ndarray, y: np.ndarray) -> float:
    return Latitudo.softmax_ce(model.forward(Phi)["logits"], y)[0]


def gradient_check(model: Latitudo, Phi: np.ndarray, y: np.ndarray,
                   n_probe: int = 24, h: float = 1e-3) -> float:
    """Finite-difference check on a random sample of coordinates from every one
    of the eight parameter blocks.  Returns the maximum relative error.

    Uses the five-point stencil

        f'(x) ~ [ -f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h) ] / (12h)

    whose truncation error is O(h^4).  This matters here: several gradient
    coordinates are of order 1e-6 while the loss is of order 1, so a naive
    two-point quotient at h=1e-6 has a roundoff floor comparable to the quantity
    being measured, and would report a disagreement that belongs to the
    arithmetic rather than to the derivation.  At h=1e-3 both truncation and
    roundoff sit several orders below every analytic value probed.
    """
    rng = np.random.default_rng(99)
    _, grads = model.loss_and_grads(Phi, y)
    worst = 0.0
    for name in model.param_names:
        P = model.get(name)
        flat = P.reshape(-1)
        gflat = grads[name].reshape(-1)
        idx = rng.choice(flat.size, size=min(n_probe, flat.size), replace=False)
        for i in idx:
            old = flat[i]
            vals = []
            for step in (2 * h, h, -h, -2 * h):
                flat[i] = old + step
                vals.append(_loss_only(model, Phi, y))
            flat[i] = old
            f2p, f1p, f1m, f2m = vals
            num = (-f2p + 8 * f1p - 8 * f1m + f2m) / (12 * h)
            ana = gflat[i]
            denom = max(1e-12, abs(num) + abs(ana))
            worst = max(worst, abs(num - ana) / denom)
    return worst


# ==============================================================================
# SECTION 7 -- TRAINING
# ==============================================================================

def train(model: Latitudo, Phi: np.ndarray, y: np.ndarray,
          Phi_te: np.ndarray, y_te: np.ndarray,
          epochs: int = 200, batch: int = 64, lr: float = 0.012,
          verbose: bool = True) -> Dict[str, object]:
    """Adam, written out, because the whole file is from scratch."""
    m = {n: np.zeros_like(model.get(n)) for n in model.param_names}
    v = {n: np.zeros_like(model.get(n)) for n in model.param_names}
    b1, b2, eps = 0.9, 0.999, 1e-8
    step = 0
    n = len(y)
    rng = np.random.default_rng(SEED + 3)
    hist = {"loss": [], "test_acc": [], "gain": []}
    prev_acc = 0.25

    for ep in range(1, epochs + 1):
        order = rng.permutation(n)
        ep_loss = 0.0
        nb = 0
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            loss, g = model.loss_and_grads(Phi[idx], y[idx])
            step += 1
            for name in model.param_names:
                m[name] = b1 * m[name] + (1 - b1) * g[name]
                v[name] = b2 * v[name] + (1 - b2) * g[name] ** 2
                mh = m[name] / (1 - b1 ** step)
                vh = v[name] / (1 - b2 ** step)
                model.get(name)[...] -= lr * mh / (np.sqrt(vh) + eps)
            ep_loss += loss
            nb += 1
        acc = accuracy(model, Phi_te, y_te)
        hist["loss"].append(ep_loss / nb)
        hist["test_acc"].append(acc)
        hist["gain"].append(max(0.0, acc - prev_acc))   # capability increment
        prev_acc = acc
        if verbose and (ep % 25 == 0 or ep == 1):
            print(f"    epoch {ep:4d}   loss {ep_loss/nb:.4f}   "
                  f"held-out {acc*100:5.1f}%")
    return hist


def accuracy(model: Latitudo, Phi: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.argmax(model.forward(Phi)["logits"], 1) == y))


# ==============================================================================
# SECTION 8 -- THE FRAME GAUGE  (Livre du ciel et du monde II.25)
# ==============================================================================

def shift_frame(model: Latitudo, tau: int) -> Latitudo:
    """Rotate the whole apparatus -- every signal-reading configuration and every
    proper figure -- by tau cells of the extension.  Nothing else changes."""
    q = model.clone()
    q.W1 = roll_extension(q.W1, tau)
    q.G1raw = roll_extension(q.G1raw, tau)
    q.W2 = roll_extension(q.W2, tau)
    q.G2raw = roll_extension(q.G2raw, tau)
    q.U = roll_extension(q.U, tau)
    return q


def esbatement(model: Latitudo, Phi: np.ndarray) -> Dict[str, object]:
    """THE DIVERSION, AND THE UNDECIDABILITY REPORT.

    Oresme built the case for a turning Earth to full strength, answered every
    objection against it, then wrote that what he had said he had said
    'par esbatement' -- by way of diversion -- and that he held the heavens
    moved regardless (Le livre du ciel et du monde II.25, ed. Menut & Denomy,
    pp. 536-38).  The point was never the Earth.  The point was that natural
    reason could not settle it, and that a method should be able to say so.

    Here the same claim is a theorem.  Rotate the frame by tau while rotating
    the inputs with it, and every output is identical.  The frame parameter is
    unidentifiable from any behaviour whatsoever.  The system therefore returns
    an undecidability verdict rather than a value, and any tau it goes on to use
    is logged as a CONVENTION, not as a finding.
    """
    base = model.forward(Phi)["logits"]
    worst = 0.0
    for tau in range(E_CELLS):
        q = shift_frame(model, tau)
        alt = q.forward(roll_extension(Phi, tau))["logits"]
        worst = max(worst, float(np.abs(alt - base).max()))
    resolution = float(np.abs(base).max()) * 1e-9
    return dict(max_output_difference=worst,
                sensor_resolution=resolution,
                verdict=("indecidable par experience"
                         if worst < resolution else "discriminable"),
                convention_adopted="tau = 0 (frame held still, by declaration)")


# ==============================================================================
# SECTION 9 -- DE CAUSIS MIRABILIUM: attribution of a surprise
# ==============================================================================

def de_causis_mirabilium(model: Latitudo, phi: np.ndarray, label: int
                         ) -> Dict[str, object]:
    """Before a marvel, exhaust the three ordinary causes.

    Oresme's method (De causis mirabilium, ed. Hansen 1985): marvels are usually
    defect of SENSE, over-vigorous IMAGINATION supplying what was not there, or
    faulty REASON.  Only the residue deserves the name.  We test each in turn on
    a single wrongly-classified figure:

      sensus      -- smooth the input; does the verdict change?
      imaginatio  -- flatten the configuratio channel, keeping only quantity;
                     does the verdict change?  (the interior sense supplying
                     shape that the datum did not carry)
      ratio       -- perturb the couplings slightly; is the verdict unstable?

    Whatever survives all three is reported as MIRABILE -- unexplained, and
    labelled as such rather than filled in.
    """
    def pred(p_in, mdl):
        return int(np.argmax(mdl.forward(p_in[None, :])["logits"][0]))

    base = pred(phi, model)
    out = {"predicted": CLASS_NAMES[base], "true": CLASS_NAMES[label]}

    # sensus: is the verdict an artefact of fine-grained input detail?
    k = np.array([0.25, 0.5, 0.25])
    smoothed = np.convolve(np.r_[phi[-1], phi, phi[0]], k, mode="valid")
    smoothed = smoothed / (np.sum(smoothed) * DX)
    out["sensus_changes_verdict"] = (pred(smoothed, model) != base)

    # imaginatio: strip the shape, keep the quantity
    flat = np.full(E_CELLS, float(np.mean(phi)))
    out["imaginatio_changes_verdict"] = (pred(flat, model) != base)

    # ratio: is the inference itself unstable under small change?
    rng = np.random.default_rng(11)
    flips = 0
    for _ in range(12):
        q = model.clone()
        q.W2 = q.W2 + 0.02 * rng.standard_normal(q.W2.shape)
        flips += (pred(phi, q) != base)
    out["ratio_instability"] = flips / 12.0

    explained = (out["sensus_changes_verdict"]
                 or out["imaginatio_changes_verdict"]
                 or out["ratio_instability"] > 0.25)
    out["verdict"] = "causa naturalis inventa" if explained else "mirabile"
    return out


# ==============================================================================
# SECTION 10 -- SELF-TESTS
# ==============================================================================

def run_self_tests(model: Latitudo, Phi: np.ndarray, y: np.ndarray
                   ) -> List[Tuple[str, bool, str]]:
    results: List[Tuple[str, bool, str]] = []

    # 1. Merton split is exact
    w = RNG.standard_normal(E_CELLS)
    p = np.abs(RNG.standard_normal(E_CELLS)) + 0.1
    q_, c_ = merton_split(w, p)
    direct = float(np.sum(w * p) * DX)
    err = abs((q_ + c_) - direct)
    results.append(("Merton split exact  <w,phi> = wbar*area + <dw,dphi>",
                    err < 1e-13, f"residual {err:.2e}"))

    # 2. The mean-speed theorem itself
    x = (np.arange(E_CELLS) + 0.5) / E_CELLS
    ramp = 2.0 * x                       # uniformly difform from 0 to 2
    mid = np.full(E_CELLS, 1.0)          # uniform at the middle degree
    err2 = abs(quantitas(ramp) - quantitas(mid))
    results.append(("Mean-speed theorem: uniformly difform == uniform at mid",
                    err2 < 1e-13, f"|difference| {err2:.2e}"))

    # 3. Every sample carries the same quantity
    areas = quantitas(Phi)
    spread = float(areas.max() - areas.min())
    results.append(("Corpus: quantitas identical across all figures",
                    spread < 1e-12, f"spread {spread:.2e}, area {areas.mean():.6f}"))

    # 4. Unit-area lock on proper figures
    g1, g2, _, _ = model.proper_figures()
    a = np.concatenate([np.sum(g1, 1) * DX, np.sum(g2, 1) * DX])
    dev = float(np.abs(a - 1.0).max())
    results.append(("De moneta lock: every proper figure has unit area",
                    dev < 1e-12, f"max deviation {dev:.2e}"))

    # 5. Frame gauge
    es = esbatement(model, Phi[:64])
    results.append(("Frame gauge: output invariant under joint rotation",
                    es["max_output_difference"] < 1e-10,
                    f"max delta {es['max_output_difference']:.2e} -> "
                    f"{es['verdict']}"))

    # 6. Non-recurrence
    rp = recurrence_probe()
    results.append(("No exact recurrence under incommensurable rates",
                    rp["exact_returns"] == 0,
                    f"closest return {rp['closest_return']:.6f} in "
                    f"{rp['steps']} steps, exact returns {rp['exact_returns']}"))

    # 7. Retrieval provably empty
    hits = sum(retrieval_gate(Phi[i], Phi[np.arange(len(Phi)) != i])
               for i in range(40))
    results.append(("Exact-match retrieval returns nothing",
                    hits == 0, f"{hits} hits over 40 queries"))

    # 8. Grouping test discriminates where term size cannot
    harm = np.array([1.0 / n for n in range(1, 129)])
    sq = np.array([1.0 / n ** 2 for n in range(1, 129)])
    vh = oresme_grouping_test(harm)["verdict"]
    vs = oresme_grouping_test(sq)["verdict"]
    results.append(("Grouping test: 1/n divergens, 1/n^2 convergens",
                    vh == "divergens" and vs == "convergens",
                    f"1/n -> {vh}; 1/n^2 -> {vs}; both term sequences vanish"))

    # 9. Incommensurability is generic and gets more so
    c8 = incommensurability_census(8)["fraction_commensurable"]
    c14 = incommensurability_census(14)["fraction_commensurable"]
    results.append(("Incommensurability prior: commensurable pairs thin out",
                    c14 < c8,
                    f"N=8 -> {c8*100:.2f}% commensurable; "
                    f"N=14 -> {c14*100:.2f}%"))

    return results


# ==============================================================================
# SECTION 11 -- MAIN
# ==============================================================================

def main() -> None:
    print("=" * 78)
    print(" LATITUDO ENGINE -- Nicole Oresme (c.1320-1382), figure 0293")
    print(" a network in which every signal and every weight is a FIGURE,")
    print(" and the scalar neuron is the degenerate uniform case")
    print("=" * 78)

    # ---- corpus -------------------------------------------------------------
    Phi_tr, y_tr = make_corpus(1200, rng=np.random.default_rng(SEED))
    Phi_te, y_te = make_corpus(500, rng=np.random.default_rng(SEED + 101))
    print(f"\n[corpus] {len(y_tr)} training figures, {len(y_te)} held out, "
          f"{E_CELLS} cells of extension")
    print(f"[corpus] four species: {', '.join(CLASS_NAMES)}")
    print(f"[corpus] quantitas qualitatis of every figure = "
          f"{quantitas(Phi_tr).mean():.9f} "
          f"(spread {np.ptp(quantitas(Phi_tr)):.2e})")
    print("[corpus] the area channel therefore carries ZERO bits by construction")

    model = Latitudo()
    n_par = sum(model.get(n).size for n in model.param_names)
    print(f"[model ] {n_par} parameters; "
          f"{model.H1} then {model.H2} configuration-valued units")

    lock_before = measure_hash(model)
    print(f"[moneta] measure hash before training: {lock_before}")

    # ---- gradient check -----------------------------------------------------
    print("\n[grad  ] finite-difference check on all eight parameter blocks ...")
    err = gradient_check(model, Phi_tr[:48], y_tr[:48])
    ok = err < 1e-6
    print(f"[grad  ] max relative error {err:.3e}  ->  {'PASS' if ok else 'FAIL'}")
    assert ok, "gradient check failed"

    # ---- baseline -----------------------------------------------------------
    base = QuantitasOnly().fit(Phi_tr, y_tr)
    base_acc = base.accuracy(Phi_te, y_te)
    print(f"\n[baseline] quantitas-only model (the mind that kept the number):")
    print(f"[baseline] held-out accuracy {base_acc*100:.1f}%  "
          f"(chance is 25.0%)")

    # ---- train --------------------------------------------------------------
    print("\n[train ] configuratio model:")
    hist = train(model, Phi_tr, y_tr, Phi_te, y_te)
    final = accuracy(model, Phi_te, y_te)
    print(f"[train ] final held-out accuracy {final*100:.1f}%")
    print(f"[train ] information recovered from the shape alone: "
          f"{(final - base_acc)*100:+.1f} points over the quantity")

    lock_after = measure_hash(model)
    print(f"[moneta] measure hash after training:  {lock_after}  "
          f"-> {'unaltered' if lock_after == lock_before else 'MUTATIO MONETAE'}")

    # ---- Oresme's grouping monitor on the training increments ---------------
    g = oresme_grouping_test(np.array(hist["gain"]))
    print(f"\n[series] grouping test on per-epoch capability increments:")
    print(f"[series] block sums {[round(s,4) for s in g['block_sums']]}")
    print(f"[series] verdict '{g['verdict']}' "
          f"(late/early block ratio {g['ratio']:.3f})")
    print("[series] note: every increment is individually negligible; that fact")
    print("[series] alone does not bound the total, which is Oresme's 1350 point")

    # ---- the frame gauge ----------------------------------------------------
    es = esbatement(model, Phi_te[:128])
    print(f"\n[ciel  ] rotating the frame through all {E_CELLS} positions:")
    print(f"[ciel  ] largest output difference {es['max_output_difference']:.3e}")
    print(f"[ciel  ] verdict: {es['verdict']}")
    print(f"[ciel  ] {es['convention_adopted']}")

    # ---- attribution of a surprise -----------------------------------------
    pred = np.argmax(model.forward(Phi_te)["logits"], 1)
    wrong = np.where(pred != y_te)[0]
    print(f"\n[mirab ] {len(wrong)} misjudged figures out of {len(y_te)}")
    if len(wrong):
        i = int(wrong[0])
        d = de_causis_mirabilium(model, Phi_te[i], int(y_te[i]))
        print(f"[mirab ] case {i}: read as '{d['predicted']}', "
              f"truly '{d['true']}'")
        print(f"[mirab ]   sensus alters verdict:     {d['sensus_changes_verdict']}")
        print(f"[mirab ]   imaginatio alters verdict: {d['imaginatio_changes_verdict']}")
        print(f"[mirab ]   ratio instability:         {d['ratio_instability']:.2f}")
        print(f"[mirab ]   -> {d['verdict']}")

    # ---- the Merton audit of a live decision --------------------------------
    c = model.forward(Phi_te[:1])
    unit = 0
    qt, cf = merton_split(model.W1[unit], Phi_te[0])
    print(f"\n[merton] audit of unit {unit} on one held-out figure:")
    print(f"[merton]   quantitas term   {qt:+.6f}")
    print(f"[merton]   configuratio term {cf:+.6f}")
    print(f"[merton]   coupling         {qt+cf:+.6f} "
          f"(direct {float(c['A1'][0, unit]):+.6f})")
    share = abs(cf) / (abs(qt) + abs(cf) + 1e-15)
    print(f"[merton]   share of the decision owed to SHAPE: {share*100:.1f}%")

    # ---- the counting argument ---------------------------------------------
    print("\n[ratios] Oresme's census of ratios of ratios:")
    for N in (6, 10, 14):
        cs = incommensurability_census(N)
        print(f"[ratios]   pool up to {N:2d}: {cs['pool']:4d} ratios, "
              f"{cs['fraction_commensurable']*100:6.2f}% of pairs commensurable")
    print("[ratios] the commensurable case thins as the world grows -- which is")
    print("[ratios] why, he concluded, no configuration of the heavens returns")

    # ---- self tests ---------------------------------------------------------
    print("\n" + "-" * 78)
    print(" SELF-TESTS")
    print("-" * 78)
    results = run_self_tests(model, Phi_te, y_te)
    for name, passed, detail in results:
        print(f" [{'PASS' if passed else 'FAIL'}] {name}")
        print(f"        {detail}")
    n_pass = sum(1 for _, p, _ in results if p)
    print("-" * 78)
    print(f" {n_pass}/{len(results)} self-tests passed; "
          f"gradient check {err:.3e}; held-out {final*100:.1f}% "
          f"vs {base_acc*100:.1f}% for quantity alone")
    print("=" * 78)
    assert n_pass == len(results), "a self-test failed"


if __name__ == "__main__":
    main()
