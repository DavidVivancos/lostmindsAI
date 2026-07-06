#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 chapter_0037_cleisthenes_-570.py
 Figure 37 — Cleisthenes of Athens (c. 570 – c. 508 BCE)
 Architecture : DEMOS-NET — a Decorrelated Sortition Ensemble
 Substrate    : pure NumPy, from scratch (no autograd, no ML frameworks)
 # Part Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 2 Minds 21 - 40 Available on Amazon https://www.amazon.com/dp/B0H6QCQ9M7
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 0037 · Cleisthenes of Athens
================================================================================

WHY THIS ARCHITECTURE (and why it is NOT a Transformer / MoE)
--------------------------------------------------------------------------------
Cleisthenes is remembered as "the father of Athenian democracy," but the cliché
("collective intelligence", "wisdom of crowds") is *not* his idea — that lens
belongs to Pericles, and "equal law / isonomia" belongs to Solon a generation
earlier. Cleisthenes' single distinctive cognitive act is narrower, harder, and
mechanical:

    He re-engineered the *topology* of the social graph BEFORE any deliberation
    happened, so that no decision unit could ever be a captured faction.

In 508/507 BCE he took the ~139 demes (villages) of Attica, sorted them into
three regions — the city (asty), the coast (paralia), the inland (mesogeia) —
bundled them into 30 trittyes ("thirds", ten per region), and then assembled
each of the 10 new tribes (phylai) from *exactly one trittys per region*,
allotted partly by lot. The Athenaion Politeia says he did this to redistribute
the population "in fresh combinations" (ch. 21). The point, stated plainly by
Aristotle and confirmed by modern scholarship, was to make it "much less likely
that tribes would act on geographical and family loyalties."

That is, in modern terms, a **decorrelation operator**. Old Attica's regional
blocs (men of the coast / plain / hill — the very factions Herodotus names) were
*correlated* error sources: a coastal bloc fails coastally, together. Averaging
correlated voters does not reduce error. Cleisthenes' trick was to forcibly
build each voting unit as a cross-section of the whole, so that the units became
statistically independent — and only THEN does aggregation buy you robustness.
He arrived, by political surgery in the 6th century BCE, at the core insight of
ensemble learning: **the variance of an average falls only as fast as its
members are decorrelated; so engineer the independence, do not pray for it.**

DEMOS-NET encodes four mechanisms, each a literal Cleisthenic institution:

  1. TRITTYS RECOMBINATION (the thesis).
     The D input features are partitioned into 3 disjoint REGIONS. Each region is
     cut into S disjoint SLICES (trittyes). Each of the T tribes is wired to read
     exactly ONE slice from EACH region — a forced cross-section. The slice
     assignment is drawn by lot (a permutation per region). This is the only
     thing that differs between DEMOS-NET and its control, the FACTION ensemble,
     in which each tribe instead reads three slices all from a SINGLE region.
     Same parameters, same training — only the wiring differs. The experiment
     below shows the cross-sectional wiring crushes the factional one.

  2. SORTITION / THE LOT (klEros).
     The Council of 500 seated 50 citizens per tribe *by lot*, not by merit or
     wealth. Inside each tribe DEMOS-NET seats only a random subset of its hidden
     "councillors" on each forward pass (a Bernoulli mask, inverted-dropout
     scaled). Routing is weight-free and stochastic — the opposite of a learned
     MoE gate. At evaluation the whole council sits.

  3. ISONOMIA AS EQUAL WEIGHT (not learned attention).
     Tribe votes are combined by a FIXED 1/T average. There is no learned gating,
     no attention, no softmax-over-experts that would let one tribe accrue weight.
     Equal weight is the aggregation rule; decorrelation (mech. 1) is what makes
     equal weight actually work.

  4. OSTRACISM (anti-tyranny pruning).
     Under equal-weight averaging a unit can still seize control by *amplitude*:
     a tribe that grows huge-magnitude logits swamps the mean even at weight 1/T.
     That is the would-be tyrant. Each ostracism cycle DEMOS-NET measures every
     tribe's influence (mean logit norm), and if the largest exceeds the isonomic
     share by a margin it ostracizes that tribe — exiled (excluded + reinitialised)
     for a fixed number of epochs, exactly as Athens exiled its dangerous men for
     ten years without conviction. We track the Gini coefficient of influence and
     show ostracism keeps power flat versus an un-ostracised control.

  (PRYTANY ROTATION: the 10 tribes each chaired the Council for 1/10 of the year
   in an order fixed by lot. DEMOS-NET rotates a "presiding" tribe each epoch
   whose council sits in full (no sortition mask that epoch) — a documented
   fairness schedule that nets out over a full cycle and does not bias gradients.)

EVERYTHING BELOW IS REAL: analytic backprop, a finite-difference gradient check
(mandatory, must pass), a real training loop, and four self-tests. The verified
console output is pasted into the chapter.

Run:  python3 chapter_0037_cleisthenes_-570.py
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# =============================================================================
# SECTION 0 — Reproducibility & configuration
# =============================================================================

GLOBAL_SEED = 5087  # 508/7 BCE, the year of the reform.


@dataclass
class Config:
    # --- the Attica substrate (synthetic dataset) ---------------------------
    n_regions: int = 3            # asty (city), paralia (coast), mesogeia (inland)
    slices_per_region: int = 10   # 10 trittyes per region (historical)
    slice_dim: int = 3            # features carried by one trittys
    n_tribes: int = 10            # 10 phylai (historical)
    n_classes: int = 2            # the binary civic question put to the vote
    n_train: int = 1500
    n_test: int = 700
    label_noise: float = 0.04     # a few citizens always mis-hear the question
    signal_strength: float = 2.0  # how strongly each region broadcasts its sign
    feature_noise: float = 0.6    # per-dim sensory noise

    # --- each tribe is a 2-hidden-layer MLP over its cross-section ----------
    #     (layer 1 reads off the three region signs; layer 2 deliberates to a
    #      verdict -- the parity. One layer cannot reliably learn parity.)
    hidden1_per_tribe: int = 20   # first bench of councillors
    hidden2_per_tribe: int = 12   # second bench of councillors
    seat_rate: float = 0.8        # sortition: fraction of councillors seated / pass

    # --- training -----------------------------------------------------------
    lr: float = 0.30
    lr_halflife: int = 80         # step-halve the learning rate every N epochs
    epochs: int = 240
    batch: int = 150
    weight_init: float = 0.6

    # --- ostracism ----------------------------------------------------------
    ostracism_every: int = 10     # hold an ostracism vote every N epochs
    ostracism_margin: float = 1.35  # exile if influence > margin * isonomic share
    exile_epochs: int = 10        # ten-year exile (here: ten epochs)
    min_active_tribes: int = 7    # never collapse the assembly below a quorum

    @property
    def region_dim(self) -> int:
        return self.slices_per_region * self.slice_dim

    @property
    def input_dim(self) -> int:
        return self.n_regions * self.region_dim

    @property
    def tribe_input_dim(self) -> int:
        # one slice from each region
        return self.n_regions * self.slice_dim


# =============================================================================
# SECTION 1 — The Attica dataset
# -----------------------------------------------------------------------------
# We synthesise a population whose correct collective answer DEPENDS ON ALL
# THREE REGIONS AT ONCE, while each region on its own is uninformative about the
# answer. Concretely each sample has a hidden sign g_r in {-1,+1} for every
# region r; the civic truth is the PARITY of the three region-signs:
#
#        y = 1  iff  g_asty * g_paralia * g_mesogeia == +1
#
# Every slice of region r carries a NOISY copy of g_r. Therefore:
#   * A unit that sees only one region learns that region's sign well but can
#     never recover the parity -> it is structurally stuck near chance.
#   * A unit that sees one slice from EACH region can estimate all three signs
#     and compute the parity -> it can be right.
# This is the formal skeleton of "a regional bloc cannot govern the whole; a
# cross-section can." It is exactly the failure Cleisthenes engineered against.
# =============================================================================

def make_attica(cfg: Config, n: int, rng: np.random.Generator
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X[n, D], y[n], G[n, 3]) where G holds the hidden region signs."""
    G = rng.choice(np.array([-1.0, 1.0]), size=(n, cfg.n_regions))
    parity = np.prod(G, axis=1)                 # in {-1,+1}
    y = (parity > 0).astype(np.int64)           # class 1 iff product == +1

    # flip a few labels (citizens mishear the question)
    flip = rng.random(n) < cfg.label_noise
    y[flip] = 1 - y[flip]

    # build features: every dim of region r is g_r * signal + gaussian noise
    X = np.empty((n, cfg.input_dim), dtype=np.float64)
    signal = cfg.signal_strength
    for r in range(cfg.n_regions):
        start = r * cfg.region_dim
        block = (G[:, r:r + 1] * signal
                 + rng.normal(0.0, cfg.feature_noise, size=(n, cfg.region_dim)))
        X[:, start:start + cfg.region_dim] = block
    return X, y, G


# =============================================================================
# SECTION 2 — Trittyes: drawing the wiring BY LOT
# -----------------------------------------------------------------------------
# For each region we hold a lottery: a random permutation of its 10 slices.
# Tribe t is then wired to slice perm_region[t] of each region. Because the
# permutations are independent across regions, the tribes are maximally spread
# combinations of region-slices -- "fresh combinations" (Ath. Pol. 21).
#
# The FACTION control uses the same shapes but wires all three of a tribe's
# input slices to a SINGLE region (round-robin which region), reproducing the
# old correlated regional blocs Cleisthenes abolished.
# =============================================================================

def wire_tribes(cfg: Config, rng: np.random.Generator, mode: str
                ) -> List[np.ndarray]:
    """Return, per tribe, an int array of input-feature indices it reads."""
    assert cfg.n_tribes <= cfg.slices_per_region
    wirings: List[np.ndarray] = []

    if mode == "cleisthenic":
        # one independent lottery per region
        perms = [rng.permutation(cfg.slices_per_region)
                 for _ in range(cfg.n_regions)]
        for t in range(cfg.n_tribes):
            idx = []
            for r in range(cfg.n_regions):
                s = perms[r][t]                       # this tribe's slice in region r
                base = r * cfg.region_dim + s * cfg.slice_dim
                idx.extend(range(base, base + cfg.slice_dim))
            wirings.append(np.array(idx, dtype=np.int64))

    elif mode == "faction":
        # each tribe reads n_regions slices, ALL from one region (a bloc)
        perms = [rng.permutation(cfg.slices_per_region)
                 for _ in range(cfg.n_regions)]
        cursor = [0, 0, 0]
        for t in range(cfg.n_tribes):
            r = t % cfg.n_regions                     # which bloc this tribe is
            idx = []
            for _ in range(cfg.n_regions):
                s = perms[r][cursor[r] % cfg.slices_per_region]
                cursor[r] += 1
                base = r * cfg.region_dim + s * cfg.slice_dim
                idx.extend(range(base, base + cfg.slice_dim))
            wirings.append(np.array(idx, dtype=np.int64))
    else:
        raise ValueError(mode)
    return wirings


# =============================================================================
# SECTION 3 — DEMOS-NET
# -----------------------------------------------------------------------------
# Per tribe t:  z1 = Xt @ W1[t] + b1[t]      (Xt = the tribe's cross-section)
#               a1 = relu(z1)
#               a1 = a1 * seat_mask / seat_rate          (sortition, train only)
#               logit_t = a1 @ W2[t] + b2[t]             shape (N, C)
# Assembly:     logits = mean over ACTIVE tribes of logit_t                (1/T)
#               p = softmax(logits);  loss = cross-entropy
# Backprop is fully analytic. The seat mask is a constant during a given
# forward/backward, so gradients flow exactly as in standard dropout.
# =============================================================================

class DemosNet:
    def __init__(self, cfg: Config, wirings: List[np.ndarray],
                 rng: np.random.Generator,
                 boost_tribe: int = -1, boost_scale: float = 1.0):
        self.cfg = cfg
        self.wirings = wirings
        self.T = cfg.n_tribes
        H1, H2, C = cfg.hidden1_per_tribe, cfg.hidden2_per_tribe, cfg.n_classes
        din = cfg.tribe_input_dim
        k = cfg.weight_init
        # parameters: each tribe is a 2-hidden-layer MLP (bench 1 reads off the
        # three region signs; bench 2 deliberates them into a verdict -- the
        # parity). One tribe may be seeded as a "demagogue" with oversized
        # weights -- a charismatic aristocrat whose amplitude lets it dominate
        # the isonomic mean. Ostracism exists to detect and exile exactly this.
        self.W1, self.b1 = [], []
        self.W2, self.b2 = [], []
        self.W3, self.b3 = [], []
        for t in range(self.T):
            s = boost_scale if t == boost_tribe else 1.0
            self.W1.append(rng.normal(0, s * k / np.sqrt(din), (din, H1)))
            self.b1.append(np.zeros(H1))
            self.W2.append(rng.normal(0, s * k / np.sqrt(H1), (H1, H2)))
            self.b2.append(np.zeros(H2))
            self.W3.append(rng.normal(0, s * k / np.sqrt(H2), (H2, C)))
            self.b3.append(np.zeros(C))
        # civic-status registers
        self.active = np.ones(self.T, dtype=bool)     # seated in the assembly?
        self.exile_left = np.zeros(self.T, dtype=int)  # epochs of exile remaining
        self.presiding = 0                             # prytany: who chairs now

    # ---- parameter (de)serialisation, used by the gradient checker ----------
    def get_params(self) -> List[np.ndarray]:
        out = []
        for t in range(self.T):
            out += [self.W1[t], self.b1[t], self.W2[t], self.b2[t],
                    self.W3[t], self.b3[t]]
        return out

    def _param_refs(self):
        for t in range(self.T):
            yield ('W1', t); yield ('b1', t)
            yield ('W2', t); yield ('b2', t)
            yield ('W3', t); yield ('b3', t)

    def set_param(self, name: str, t: int, value: np.ndarray):
        getattr(self, name)[t] = value

    # ------------------------------------------------------------------------
    def forward(self, X: np.ndarray, train: bool,
                seat_masks: List[Tuple[np.ndarray, np.ndarray]] | None = None):
        """Return (logits, probs, cache). Two hidden benches, sortition on both."""
        cfg = self.cfg
        N = X.shape[0]
        caches = []
        active_idx = [t for t in range(self.T) if self.active[t]]
        logit_sum = np.zeros((N, cfg.n_classes))
        for t in active_idx:
            Xt = X[:, self.wirings[t]]                  # (N, din) cross-section
            seated = (train and (t != self.presiding)
                      and seat_masks is not None)
            z1 = Xt @ self.W1[t] + self.b1[t]
            a1 = np.maximum(z1, 0.0)
            if seated:
                m1, m2 = seat_masks[t]                  # sortition: seat by lot
                s1 = m1 / cfg.seat_rate                 # inverted-dropout scale
            else:
                s1 = np.ones_like(a1)                   # presiding tribe sits full
            a1 = a1 * s1
            z2 = a1 @ self.W2[t] + self.b2[t]
            a2 = np.maximum(z2, 0.0)
            if seated:
                s2 = m2 / cfg.seat_rate
            else:
                s2 = np.ones_like(a2)
            a2 = a2 * s2
            logit_t = a2 @ self.W3[t] + self.b3[t]      # (N, C)
            logit_sum += logit_t
            caches.append((t, Xt, z1, a1, s1, z2, a2, s2))
        logits = logit_sum / max(len(active_idx), 1)    # ISONOMIA: equal weight
        probs = self._softmax(logits)
        cache = (caches, active_idx, N)
        return logits, probs, cache

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def loss(self, probs: np.ndarray, y: np.ndarray) -> float:
        N = y.shape[0]
        return float(-np.log(probs[np.arange(N), y] + 1e-12).mean())

    def backward(self, cache, probs: np.ndarray, y: np.ndarray
                 ) -> Dict[Tuple[str, int], np.ndarray]:
        """Analytic gradients of the mean cross-entropy w.r.t. every parameter."""
        caches, active_idx, N = cache
        n_active = max(len(active_idx), 1)
        dlogits = probs.copy()
        dlogits[np.arange(N), y] -= 1.0
        dlogits /= N                                    # d loss / d logits
        dlogit_t = dlogits / n_active                   # equal-weight mean share

        grads: Dict[Tuple[str, int], np.ndarray] = {}
        for (t, Xt, z1, a1, s1, z2, a2, s2) in caches:
            # logit_t = a2 @ W3 + b3   (a2 already includes its gate factor s2)
            dW3 = a2.T @ dlogit_t
            db3 = dlogit_t.sum(axis=0)
            da2 = dlogit_t @ self.W3[t].T
            da2 = da2 * s2                               # bench-2 gate factor
            dz2 = da2 * (z2 > 0.0)
            dW2 = a1.T @ dz2
            db2 = dz2.sum(axis=0)
            da1 = dz2 @ self.W2[t].T
            da1 = da1 * s1                               # bench-1 gate factor
            dz1 = da1 * (z1 > 0.0)
            dW1 = Xt.T @ dz1
            db1 = dz1.sum(axis=0)
            grads[('W1', t)] = dW1; grads[('b1', t)] = db1
            grads[('W2', t)] = dW2; grads[('b2', t)] = db2
            grads[('W3', t)] = dW3; grads[('b3', t)] = db3
        # exiled / unseated tribes get zero gradient
        for t in range(self.T):
            for nm in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3'):
                grads.setdefault((nm, t), np.zeros(getattr(self, nm)[t].shape))
        return grads

    # ---- sampling sortition masks (kept outside forward so grad-check is exact)
    def sample_seat_masks(self, N: int, rng: np.random.Generator
                          ) -> List[Tuple[np.ndarray, np.ndarray]]:
        H1, H2, sr = (self.cfg.hidden1_per_tribe, self.cfg.hidden2_per_tribe,
                      self.cfg.seat_rate)
        out = []
        for _ in range(self.T):
            m1 = (rng.random((N, H1)) < sr).astype(np.float64)
            m2 = (rng.random((N, H2)) < sr).astype(np.float64)
            out.append((m1, m2))
        return out

    # ------------------------------------------------------------------------
    def sgd_step(self, grads, lr: float):
        for t in range(self.T):
            if not self.active[t]:
                continue
            for nm in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3'):
                getattr(self, nm)[t] -= lr * grads[(nm, t)]

    # ---- civic diagnostics --------------------------------------------------
    def tribe_influence(self, X: np.ndarray) -> np.ndarray:
        """Mean L2 norm of each tribe's logit vector — its real sway on the vote.
        A tribe with large amplitude dominates the isonomic mean (a tyrant)."""
        infl = np.zeros(self.T)
        for t in range(self.T):
            Xt = X[:, self.wirings[t]]
            a1 = np.maximum(Xt @ self.W1[t] + self.b1[t], 0.0)
            a2 = np.maximum(a1 @ self.W2[t] + self.b2[t], 0.0)
            logit_t = a2 @ self.W3[t] + self.b3[t]
            infl[t] = np.linalg.norm(logit_t, axis=1).mean()
        return infl

    def reinit_tribe(self, t: int, rng: np.random.Generator):
        H1, H2, C = (self.cfg.hidden1_per_tribe, self.cfg.hidden2_per_tribe,
                     self.cfg.n_classes)
        din = self.cfg.tribe_input_dim
        k = self.cfg.weight_init
        self.W1[t] = rng.normal(0, k / np.sqrt(din), (din, H1)); self.b1[t] = np.zeros(H1)
        self.W2[t] = rng.normal(0, k / np.sqrt(H1), (H1, H2));   self.b2[t] = np.zeros(H2)
        self.W3[t] = rng.normal(0, k / np.sqrt(H2), (H2, C));    self.b3[t] = np.zeros(C)


def gini(x: np.ndarray) -> float:
    """Gini coefficient of a non-negative vector (0 = perfect equality)."""
    x = np.sort(np.abs(x.astype(np.float64)))
    n = x.size
    if x.sum() == 0:
        return 0.0
    cum = np.cumsum(x)
    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)


# =============================================================================
# SECTION 4 — Finite-difference gradient check (MANDATORY)
# -----------------------------------------------------------------------------
# We fix one set of sortition masks so the loss is a deterministic function of
# the parameters, then compare analytic grads to central finite differences.
# =============================================================================

def _relu_pattern(cache) -> bytes:
    """Pack the ReLU activation pattern (signs of every pre-activation z1, z2
    across all seated tribes) into a hashable signature. Two evaluations with
    the SAME signature lie in one smooth linear region, so the central finite
    difference is exact there and the analytic gradient must match."""
    caches, _, _ = cache
    bits = []
    for (t, Xt, z1, a1, s1, z2, a2, s2) in caches:
        bits.append((z1 > 0.0).ravel())
        bits.append((z2 > 0.0).ravel())
    return np.packbits(np.concatenate(bits)).tobytes()


def gradient_check(cfg: Config, verbose: bool = True) -> float:
    """Kink-aware finite-difference check.

    A ReLU network is piecewise-linear; the gradient is classically defined
    on the interior of each linear region and is only a *subgradient* exactly
    on a kink (some z_i == 0). A central finite difference is a valid estimator
    only when the +eps and -eps evaluations stay inside the SAME region. We
    therefore sample coordinates and accept one only when its perturbation does
    not flip any ReLU gate (activation pattern unchanged). On those coordinates
    analytic and numerical gradients must agree to ~1e-6. Coordinates that
    straddle a kink are reported and skipped, not silently passed."""
    rng = np.random.default_rng(11)
    Xs, ys, _ = make_attica(cfg, 40, rng)
    wirings = wire_tribes(cfg, rng, "cleisthenic")
    net = DemosNet(cfg, wirings, rng)
    masks = net.sample_seat_masks(Xs.shape[0], rng)

    def eval_at():
        _, probs, cache = net.forward(Xs, train=True, seat_masks=masks)
        return net.loss(probs, ys), _relu_pattern(cache)

    _, probs, cache = net.forward(Xs, train=True, seat_masks=masks)
    analytic = net.backward(cache, probs, ys)

    eps = 1e-6
    max_rel = 0.0
    n_clean = 0
    n_kink = 0
    target_clean = 60
    rng_idx = np.random.default_rng(3)
    refs = list(net._param_refs())
    attempts = 0
    while n_clean < target_clean and attempts < 6000:
        attempts += 1
        name, t = refs[rng_idx.integers(len(refs))]
        flat = getattr(net, name)[t].ravel()
        c = int(rng_idx.integers(flat.size))
        orig = flat[c]
        flat[c] = orig + eps; lp, pat_p = eval_at()
        flat[c] = orig - eps; lm, pat_m = eval_at()
        flat[c] = orig
        if pat_p != pat_m:                  # finite-diff path crossed a kink
            n_kink += 1
            continue
        num = (lp - lm) / (2 * eps)
        ana = analytic[(name, t)].ravel()[c]
        denom = max(1e-12, abs(num) + abs(ana))
        max_rel = max(max_rel, abs(num - ana) / denom)
        n_clean += 1
    if verbose:
        print(f"  [grad-check] max relative error = {max_rel:.3e}  "
              f"over {n_clean} smooth-region coords "
              f"({n_kink} kink-straddling coords skipped)  "
              f"({'PASS' if max_rel < 1e-5 else 'FAIL'})")
    return max_rel


# =============================================================================
# SECTION 5 — Training loop (with ostracism + prytany rotation)
# =============================================================================

def train(cfg: Config, mode: str, seed: int, use_ostracism: bool,
          log: bool = False, boost_tribe: int = -1, boost_scale: float = 1.0):
    rng = np.random.default_rng(seed)
    Xtr, ytr, _ = make_attica(cfg, cfg.n_train, rng)
    Xte, yte, _ = make_attica(cfg, cfg.n_test, rng)
    wirings = wire_tribes(cfg, rng, mode)
    net = DemosNet(cfg, wirings, rng, boost_tribe=boost_tribe,
                   boost_scale=boost_scale)

    ginis: List[float] = []
    infl_hist: List[np.ndarray] = []
    active_hist: List[np.ndarray] = []
    n = Xtr.shape[0]
    for ep in range(cfg.epochs):
        net.presiding = ep % net.T                       # prytany rotation
        lr = cfg.lr * (0.5 ** (ep // cfg.lr_halflife))   # step LR decay
        # decrement exile clocks; recall citizens whose exile has ended
        for t in range(net.T):
            if net.exile_left[t] > 0:
                net.exile_left[t] -= 1
                if net.exile_left[t] == 0:
                    net.active[t] = True

        order = rng.permutation(n)
        for s in range(0, n, cfg.batch):
            bi = order[s:s + cfg.batch]
            Xb, yb = Xtr[bi], ytr[bi]
            masks = net.sample_seat_masks(Xb.shape[0], rng)
            _, probs, cache = net.forward(Xb, train=True, seat_masks=masks)
            grads = net.backward(cache, probs, yb)
            net.sgd_step(grads, lr)

        # ---- ostracism vote -------------------------------------------------
        if use_ostracism and (ep + 1) % cfg.ostracism_every == 0:
            infl = net.tribe_influence(Xtr)
            active_infl = infl[net.active]
            share = active_infl.mean()
            t_star = int(np.argmax(np.where(net.active, infl, -np.inf)))
            if (infl[t_star] > cfg.ostracism_margin * share
                    and net.active.sum() > cfg.min_active_tribes):
                net.active[t_star] = False
                net.exile_left[t_star] = cfg.exile_epochs
                net.reinit_tribe(t_star, rng)            # returns reformed
        infl_now = net.tribe_influence(Xtr)
        infl_hist.append(infl_now)
        active_hist.append(net.active.copy())
        ginis.append(gini(infl_now))

        if log and (ep % 20 == 0 or ep == cfg.epochs - 1):
            tr = accuracy(net, Xtr, ytr)
            te = accuracy(net, Xte, yte)
            print(f"    epoch {ep:3d}  loss(train)~ "
                  f"{net.loss(net.forward(Xtr, False)[1], ytr):.3f}  "
                  f"acc_tr={tr:.3f} acc_te={te:.3f} gini={ginis[-1]:.3f}")

    return net, Xtr, ytr, Xte, yte, (ginis, infl_hist, active_hist)


def accuracy(net: DemosNet, X: np.ndarray, y: np.ndarray) -> float:
    _, probs, _ = net.forward(X, train=False)
    return float((probs.argmax(axis=1) == y).mean())


# =============================================================================
# SECTION 6 — Self-tests / experiments
# =============================================================================

def main():
    np.set_printoptions(precision=3, suppress=True)
    cfg = Config()
    print("=" * 74)
    print("DEMOS-NET  —  Cleisthenes of Athens (figure 37)")
    print("Decorrelated Sortition Ensemble  |  pure NumPy")
    print("=" * 74)
    print(f"Attica: D={cfg.input_dim} features = {cfg.n_regions} regions x "
          f"{cfg.slices_per_region} trittyes x {cfg.slice_dim} dims")
    print(f"Assembly: {cfg.n_tribes} tribes, {cfg.hidden1_per_tribe}+{cfg.hidden2_per_tribe} "
          f"councillors each (two benches), seat-rate {cfg.seat_rate}, "
          f"isonomic (equal-weight) vote")
    print(f"Civic question: parity of 3 region-signs (needs all regions at once)")
    print("-" * 74)

    # ---- TEST 1: gradient check ----------------------------------------------
    print("[TEST 1] Finite-difference gradient check")
    rel = gradient_check(cfg)
    t1 = rel < 1e-5

    # ---- TEST 2: the network learns ------------------------------------------
    print("\n[TEST 2] DEMOS-NET trains on Attica (cross-sectional wiring)")
    net, Xtr, ytr, Xte, yte, _diag = train(
        cfg, "cleisthenic", seed=GLOBAL_SEED, use_ostracism=True, log=True)
    acc_cleis = accuracy(net, Xte, yte)
    t2 = acc_cleis > 0.85

    # ---- TEST 3: recombination beats factions (THE THESIS) -------------------
    print("\n[TEST 3] Cross-section (Cleisthenic) vs single-region (faction)")
    cleis_scores, faction_scores = [], []
    for sd in (101, 202, 303):
        nc, _, _, Xc, yc, _ = train(cfg, "cleisthenic", sd, use_ostracism=True)
        nf, _, _, Xf, yf, _ = train(cfg, "faction", sd, use_ostracism=True)
        cleis_scores.append(accuracy(nc, Xc, yc))
        faction_scores.append(accuracy(nf, Xf, yf))
    mc, mf = np.mean(cleis_scores), np.mean(faction_scores)
    print(f"    Cleisthenic cross-section accuracy : {mc:.3f}  "
          f"(seeds {np.round(cleis_scores,3)})")
    print(f"    Faction (regional-bloc) accuracy   : {mf:.3f}  "
          f"(seeds {np.round(faction_scores,3)})")
    print(f"    -> recombination advantage         : +{mc - mf:.3f}")
    t3 = (mc - mf) > 0.20

    # ---- TEST 4: ostracism caps the tyrant -----------------------------------
    print("\n[TEST 4] Ostracism caps peak influence (anti-tyranny)")
    print("    (tribe 3 is seeded as a 'demagogue' with 3x oversized weights)")

    def peak_dominance(diag):
        # max active-tribe influence / mean active-tribe influence, late epochs
        _, infl_hist, active_hist = diag
        doms = []
        for infl, act in zip(infl_hist[-40:], active_hist[-40:]):
            a = infl[act]
            if a.size and a.mean() > 0:
                doms.append(a.max() / a.mean())
        return float(np.mean(doms))

    *_, diag_on = train(cfg, "cleisthenic", 777, use_ostracism=True,
                        boost_tribe=3, boost_scale=3.0)
    *_, diag_off = train(cfg, "cleisthenic", 777, use_ostracism=False,
                         boost_tribe=3, boost_scale=3.0)
    d_on = peak_dominance(diag_on)
    d_off = peak_dominance(diag_off)
    print(f"    peak influence ratio  WITH ostracism : {d_on:.3f}x")
    print(f"    peak influence ratio  WITHOUT        : {d_off:.3f}x")
    print(f"    -> tyrant's edge cut by              : {d_off - d_on:+.3f}x")
    t4 = d_on < d_off - 1e-3

    # ---- verdict -------------------------------------------------------------
    print("\n" + "=" * 74)
    print("SELF-TEST SUMMARY")
    for name, ok in (("1 gradient-check", t1), ("2 learns", t2),
                     ("3 recombination>faction", t3), ("4 ostracism flattens", t4)):
        print(f"  [{'PASS' if ok else 'FAIL'}] test {name}")
    allok = all([t1, t2, t3, t4])
    print("=" * 74)
    print("ALL TESTS PASSED" if allok else "SOME TESTS FAILED")
    print("=" * 74)
    return allok


if __name__ == "__main__":
    ok = main()
    import sys
    sys.exit(0 if ok else 1)
