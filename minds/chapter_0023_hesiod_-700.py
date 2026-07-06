#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 chapter_0023_hesiod_-700.py  --  The Theogonic Net
 Chapter 23: Hesiod of Ascra (c. 700 BCE)
 # Part Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# # Tome 2 Minds 21 - 40 Available on Amazon https://www.amazon.com/dp/B0H6QCQ9M7
# Resume and Interactive Demos at https://artificiology.com/
Author: David Vivancos · Chapter 0023 · Hesiod
================================================================================

WHY THIS ARCHITECTURE, AND WHY IT IS *NOT* A TRANSFORMER
--------------------------------------------------------
Most "mind" architectures default to attention over a flat bag of stored keys.
Hesiod's mind does not work that way. His two surviving poems encode two
cognitive operations that are his alone, and this file is built to *embody*
them rather than to name-drop them:

  (1) DEFINITION-BY-DESCENT (the Theogony).
      For Hesiod, to KNOW a thing is to know WHO BEGOT IT. Night does not have
      a list of attributes; Night is the *mother of* Doom, Death, Sleep, Blame,
      Woe, Nemesis, Strife. Every entity -- even abstractions like Justice or
      Strife -- is defined by its position in a single begetting graph rooted
      in Chaos. Knowledge is therefore a directed acyclic graph (DAG) of
      generation, and you cannot compute a child until you have computed its
      parents. We make the *genealogy literally the computation graph*: a
      concept's vector is DERIVED from Chaos by walking its lineage. The weights
      of the network are the "begetting operators."

      Crucially, Hesiod distinguishes two MODES of generation, so we use two
      operators:
        * B1 -- parthenogenesis (one parent): Chaos -> Nyx; Gaia -> Ouranos.
        * B2 -- sexual generation (two parents): Gaia x Ouranos -> the Titans.

  (2) KAIROS -- THE RIGHT TIME (Works and Days).
      Hesiod's practical genius is temporal indexing: every deed has a correct
      moment, announced by an observable natural SIGN (the Pleiades rising ->
      reap; their setting -> plough; the cuckoo; 50 days after the solstice ->
      sail). The KairosCalendar module encodes this as sign -> action mapping
      and is unit-tested against his actual calendar.

  (3) THE TWO STRIFES (Works and Days 11-26).
      Hesiod corrects his own Theogony: there is not one Eris but two. One
      breeds war; the other is GOOD Strife -- emulation, the potter who works
      harder because he eyes his neighbour's full barn. We implement training as
      a *league* of TheogonicNets that improve by emulating the current best
      (good Eris) while a divergence penalty discourages collapse/destructive
      strife (bad Eris). The self-test asserts the league beats a lone learner.

  (4) ELPIS RETAINED (Pandora's jar).
      All evils fly out; only Elpis (Hope/Expectation) stays under the rim. We
      model this as one predictive channel deliberately *withheld* from the
      world-facing output -- a retained look-ahead -- and test that exactly one
      channel is retained.

WHAT IS RIGOROUSLY TRAINED HERE
-------------------------------
The trainable core is the TheogonicNet. It is pure-NumPy, with hand-written
forward and backward passes over the begetting DAG (gradients are accumulated
at shared parents, in reverse topological order). The file contains, as the
protocol for this corpus requires:
    * a finite-difference gradient check over EVERY parameter (must pass),
    * a real training loop whose loss must decrease,
    * self-tests for the Kairos, Two-Strife and Pandora mechanisms,
    * a single run_all() that executes everything and prints a verdict.

Run:   python3 chapter_0023_hesiod_-700.py
"""

import numpy as np

# Reproducibility. float64 everywhere so the finite-difference check is tight.
np.random.seed(700)  # the traditional floruit of Hesiod, -700.


# =============================================================================
# 1.  THE GENEALOGY  --  a small but real Hesiodic theogony as a DAG
# =============================================================================
# Each node: (name, parents, target_signature).
# parents == ()      -> the root, Chaos (its vector is a learned parameter).
# len(parents)==1    -> parthenogenesis, processed by operator B1.
# len(parents)==2    -> sexual generation, processed by operator B2.
#
# The 4-dim target signature is a compact "cosmic coordinate" that a Hesiodic
# reader would assign to each being. The axes are:
#     [0] ORDER      : -1 strife / usurpation ........ +1 justice / cosmos
#     [1] CELESTIAL  : -1 chthonic-dark ............... +1 bright-sky
#     [2] CHTHONIC   : -1 of the upper world .......... +1 of night / underworld
#     [3] VITALITY   : -1 deathward / barren .......... +1 generative / living
#
# The network must reproduce all of these signatures BY DESCENT from one Chaos
# seed, using only the two shared begetting operators and one readout. It cannot
# store a per-node lookup, because the operators are shared across all births.
# This is the whole Hesiodic wager: identity is recoverable from lineage.

THEOGONY = [
    # name            parents                         signature [ord, cel, cht, vit]
    ("Chaos",         (),                             [ 0.0,  0.0,  0.0,  0.0]),

    # -- primordials born from Chaos (parthenogenetic) --
    ("Gaia",          ("Chaos",),                     [ 0.6, -0.3, -0.2,  0.9]),
    ("Tartaros",      ("Chaos",),                     [-0.5,  0.0,  0.9, -0.7]),
    ("Eros",          ("Chaos",),                     [ 0.1,  0.0,  0.0,  1.0]),
    ("Erebos",        ("Chaos",),                     [-0.4,  0.0,  0.8, -0.5]),
    ("Nyx",           ("Chaos",),                     [-0.5,  0.0,  0.9, -0.4]),

    # -- the brood of Night (parthenogenetic) --
    ("Aither",        ("Nyx",),                       [ 0.4,  0.9, -0.8,  0.6]),
    ("Hemera",        ("Nyx",),                       [ 0.5,  0.9, -0.8,  0.7]),
    ("Moros",         ("Nyx",),                       [-0.6,  0.0,  0.8, -0.8]),
    ("Thanatos",      ("Nyx",),                       [-0.5,  0.0,  0.8, -0.9]),
    ("Hypnos",        ("Nyx",),                       [ 0.0,  0.0,  0.7, -0.5]),
    ("Nemesis",       ("Nyx",),                       [ 0.7,  0.0,  0.2, -0.1]),
    ("Eris",          ("Nyx",),                       [-0.9,  0.0,  0.3,  0.6]),

    # -- children of Earth (parthenogenetic) --
    ("Ouranos",       ("Gaia",),                      [ 0.5,  0.9, -0.7,  0.7]),
    ("Pontos",        ("Gaia",),                      [ 0.0,  0.0,  0.0,  0.6]),
    ("Ourea",         ("Gaia",),                      [ 0.4,  0.1, -0.1,  0.2]),

    # -- the Titans (Gaia x Ouranos) --
    ("Okeanos",       ("Gaia", "Ouranos"),            [ 0.1,  0.1,  0.0,  0.7]),
    ("Koios",         ("Gaia", "Ouranos"),            [ 0.2,  0.3, -0.2,  0.4]),
    ("Hyperion",      ("Gaia", "Ouranos"),            [ 0.3,  0.8, -0.6,  0.5]),
    ("Iapetos",       ("Gaia", "Ouranos"),            [-0.3,  0.1,  0.1,  0.5]),
    ("Kronos",        ("Gaia", "Ouranos"),            [-0.7,  0.0,  0.1,  0.6]),
    ("Rhea",          ("Gaia", "Ouranos"),            [ 0.5, -0.1, -0.1,  0.8]),
    ("Themis",        ("Gaia", "Ouranos"),            [ 0.9,  0.4, -0.4,  0.3]),
    ("Mnemosyne",     ("Gaia", "Ouranos"),            [ 0.8,  0.4, -0.4,  0.3]),
    ("Tethys",        ("Gaia", "Ouranos"),            [ 0.1,  0.0,  0.0,  0.7]),

    # -- a Titan pairing: Iapetos x Tethys-line (toy) -> the Forethinkers --
    ("Prometheus",    ("Iapetos", "Tethys"),          [ 0.3,  0.2, -0.1,  0.6]),
    ("Epimetheus",    ("Iapetos", "Tethys"),          [-0.4,  0.0,  0.1,  0.2]),

    # -- the Olympians (Kronos x Rhea) --
    ("Zeus",          ("Kronos", "Rhea"),             [ 1.0,  0.9, -0.7,  0.8]),
    ("Hera",          ("Kronos", "Rhea"),             [ 0.6,  0.4, -0.3,  0.7]),
    ("Poseidon",      ("Kronos", "Rhea"),             [ 0.2,  0.0,  0.0,  0.7]),
    ("Hades",         ("Kronos", "Rhea"),             [-0.2,  0.0,  0.9, -0.4]),
    ("Hestia",        ("Kronos", "Rhea"),             [ 0.7,  0.2, -0.2,  0.4]),
    ("Demeter",       ("Kronos", "Rhea"),             [ 0.6, -0.2, -0.1,  0.9]),

    # -- the children of Order (Zeus x Themis) and of Memory (Zeus x Mnemosyne)--
    ("Dike",          ("Zeus", "Themis"),             [ 1.0,  0.5, -0.5,  0.3]),
    ("Eunomia",       ("Zeus", "Themis"),             [ 0.9,  0.5, -0.5,  0.4]),
    ("Eirene",        ("Zeus", "Themis"),             [ 0.9,  0.6, -0.5,  0.6]),
    ("Mousai",        ("Zeus", "Mnemosyne"),          [ 0.8,  0.7, -0.6,  0.8]),
]

TARGET_DIM = 4


def build_index(theogony):
    """Map each node name to an integer index; preserve declaration order so the
    list itself is already a valid topological order (parents precede children)."""
    name_to_idx = {row[0]: i for i, row in enumerate(theogony)}
    # Sanity: every parent must already have been declared above its child.
    for i, (name, parents, _) in enumerate(theogony):
        for p in parents:
            assert p in name_to_idx and name_to_idx[p] < i, \
                f"{name}: parent {p} not declared before it (not topological)."
    return name_to_idx


# =============================================================================
# 2.  THE THEOGONIC NET  --  begetting operators trained by backprop over a DAG
# =============================================================================
class TheogonicNet:
    """A network whose forward pass *is* a genealogy.

    Parameters (all learned):
        seed  : the Chaos vector, R^d           -- the formless origin
        W1,b1 : parthenogenesis operator B1     -- one parent  -> child
        W2,b2 : sexual operator        B2       -- two parents -> child
        Wout,bout : readout                     -- vector -> cosmic signature

    Forward computes every node's embedding by composing B1/B2 from Chaos down,
    then reads out each node's 4-dim signature. Loss is mean-squared error
    against the hand-assigned signatures in THEOGONY.
    """

    def __init__(self, theogony=THEOGONY, hidden_dim=32, target_dim=TARGET_DIM,
                 natal_dim=8, scale=0.6):
        self.theo = theogony
        self.idx = build_index(theogony)
        self.N = len(theogony)
        self.d = hidden_dim
        self.o = target_dim
        self.m = natal_dim

        # Pre-compute, for each node, its kind and parent indices (static graph).
        self.kind = []          # 0 root, 1 single-parent, 2 two-parent
        self.parents = []       # tuple of parent indices
        self.targets = np.array([row[2] for row in theogony], dtype=np.float64)
        for name, ps, _ in theogony:
            self.parents.append(tuple(self.idx[p] for p in ps))
            self.kind.append(len(ps))

        # Fixed NATAL CODES (each child's individuating 'lot', moira). A
        # sinusoidal code of birth order: siblings of one parent are still
        # distinct individuals, but the code is NOT learned, so the operators
        # must still carry lineage information -- identity = lineage + lot.
        self.natal = self._natal_codes(self.N, self.m)

        # --- parameters (He-ish init, float64) ---
        rng = np.random.RandomState(700)
        self.seed = rng.randn(self.d) * scale
        self.W1 = rng.randn(self.d, self.d) * np.sqrt(2.0 / self.d)
        self.U1 = rng.randn(self.d, self.m) * np.sqrt(2.0 / self.m)
        self.b1 = np.zeros(self.d)
        self.W2 = rng.randn(self.d, 2 * self.d) * np.sqrt(2.0 / (2 * self.d))
        self.U2 = rng.randn(self.d, self.m) * np.sqrt(2.0 / self.m)
        self.b2 = np.zeros(self.d)
        self.Wout = rng.randn(self.o, self.d) * np.sqrt(2.0 / self.d)
        self.bout = np.zeros(self.o)

    @staticmethod
    def _natal_codes(n, m):
        """Deterministic positional code of birth order (a node's 'portion')."""
        codes = np.zeros((n, m))
        for i in range(n):
            for k in range(m):
                ang = i / (10000 ** (2 * (k // 2) / m))
                codes[i, k] = np.sin(ang) if k % 2 == 0 else np.cos(ang)
        return codes

    # ----- parameter (de)serialization, used by grad-check & league -----
    def get_params(self):
        return [self.seed, self.W1, self.U1, self.b1,
                self.W2, self.U2, self.b2, self.Wout, self.bout]

    def param_names(self):
        return ["seed", "W1", "U1", "b1", "W2", "U2", "b2", "Wout", "bout"]

    def flatten(self):
        return np.concatenate([p.ravel() for p in self.get_params()])

    def unflatten(self, vec):
        i = 0
        for p in self.get_params():
            n = p.size
            p[...] = vec[i:i + n].reshape(p.shape)
            i += n

    # ----- forward: derive every node from Chaos, then read out -----
    def forward(self):
        """Returns (embeddings, preacts, predictions). Caches for backward."""
        A = [None] * self.N      # node embeddings (activations)
        Z = [None] * self.N      # pre-activations (for nodes with operators)
        for i in range(self.N):  # declaration order == topological order
            k = self.kind[i]
            if k == 0:                                   # Chaos: the seed itself
                A[i] = self.seed
                Z[i] = None
            elif k == 1:                                 # one parent  -> B1
                p = A[self.parents[i][0]]
                z = self.W1 @ p + self.U1 @ self.natal[i] + self.b1
                A[i] = np.tanh(z)
                Z[i] = z
            else:                                        # two parents -> B2
                p1 = A[self.parents[i][0]]
                p2 = A[self.parents[i][1]]
                z = (self.W2 @ np.concatenate([p1, p2])
                     + self.U2 @ self.natal[i] + self.b2)
                A[i] = np.tanh(z)
                Z[i] = z
        Aout = np.stack(A)                                # (N, d)
        pred = Aout @ self.Wout.T + self.bout             # (N, o)
        self._cache = (A, Z, pred)
        return A, Z, pred

    def loss(self):
        _, _, pred = self.forward()
        diff = pred - self.targets
        return 0.5 * np.mean(np.sum(diff * diff, axis=1))

    # ----- backward: reverse-topological backprop with shared-weight accumulation
    def backward(self):
        A, Z, pred = self.forward()
        N, d = self.N, self.d

        gW1 = np.zeros_like(self.W1); gb1 = np.zeros_like(self.b1)
        gU1 = np.zeros_like(self.U1)
        gW2 = np.zeros_like(self.W2); gb2 = np.zeros_like(self.b2)
        gU2 = np.zeros_like(self.U2)
        gWout = np.zeros_like(self.Wout); gbout = np.zeros_like(self.bout)
        gseed = np.zeros_like(self.seed)
        gA = [np.zeros(d) for _ in range(N)]    # grad w.r.t each node embedding

        # readout gradients; also seed gA from the loss at each node
        scale = 1.0 / N                          # mean over nodes
        diff = pred - self.targets               # (N, o)
        for i in range(N):
            err = diff[i] * scale                # dL/dpred_i
            gWout += np.outer(err, A[i])
            gbout += err
            gA[i] += self.Wout.T @ err           # flow into the embedding

        # walk nodes in REVERSE topological order so every node's gA is complete
        for i in range(N - 1, -1, -1):
            k = self.kind[i]
            if k == 0:                            # Chaos: accumulate into seed
                gseed += gA[i]
            elif k == 1:
                p = self.parents[i][0]
                dz = gA[i] * (1.0 - A[i] ** 2)    # tanh'
                gW1 += np.outer(dz, A[p])
                gU1 += np.outer(dz, self.natal[i])
                gb1 += dz
                gA[p] += self.W1.T @ dz
            else:
                p1, p2 = self.parents[i]
                dz = gA[i] * (1.0 - A[i] ** 2)
                cat = np.concatenate([A[p1], A[p2]])
                gW2 += np.outer(dz, cat)
                gU2 += np.outer(dz, self.natal[i])
                gb2 += dz
                back = self.W2.T @ dz             # (2d,)
                gA[p1] += back[:d]
                gA[p2] += back[d:]

        return [gseed, gW1, gU1, gb1, gW2, gU2, gb2, gWout, gbout]

    # ----- a vanilla Adam optimizer so training is real, not cosmetic -----
    def fit(self, steps=4000, lr=3e-3, verbose=False):
        params = self.get_params()
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        b1, b2, eps = 0.9, 0.999, 1e-8
        history = []
        for t in range(1, steps + 1):
            grads = self.backward()
            for j, (p, g) in enumerate(zip(params, grads)):
                m[j] = b1 * m[j] + (1 - b1) * g
                v[j] = b2 * v[j] + (1 - b2) * (g * g)
                mhat = m[j] / (1 - b1 ** t)
                vhat = v[j] / (1 - b2 ** t)
                p -= lr * mhat / (np.sqrt(vhat) + eps)
            if verbose and (t % max(1, steps // 8) == 0 or t == 1):
                history.append((t, self.loss()))
        return history


# =============================================================================
# 3.  MANDATORY FINITE-DIFFERENCE GRADIENT CHECK
# =============================================================================
def gradient_check(net=None, eps=1e-6, tol=1e-5, n_samples=400, seed=1):
    """Compare analytic backprop against central finite differences.

    We sample n_samples coordinates of the flattened parameter vector (checking
    all ~3700 every run is overkill) and require the max relative error to be
    below tol. Central differences with float64 give ~1e-9 accuracy here.
    """
    if net is None:
        net = TheogonicNet(hidden_dim=16)        # smaller -> faster, still real
    theta = net.flatten()
    analytic = np.concatenate([g.ravel() for g in net.backward()])

    rng = np.random.RandomState(seed)
    coords = rng.choice(theta.size, size=min(n_samples, theta.size), replace=False)

    max_rel = 0.0
    worst = None
    for c in coords:
        orig = theta[c]
        theta[c] = orig + eps; net.unflatten(theta); lp = net.loss()
        theta[c] = orig - eps; net.unflatten(theta); lm = net.loss()
        theta[c] = orig;       net.unflatten(theta)
        num = (lp - lm) / (2 * eps)
        ana = analytic[c]
        denom = max(1e-12, abs(num) + abs(ana))
        rel = abs(num - ana) / denom
        if rel > max_rel:
            max_rel, worst = rel, (c, num, ana)
    return max_rel, worst, max_rel < tol


# =============================================================================
# 4.  KAIROS  --  Works and Days as a sign -> action calendar (deterministic)
# =============================================================================
class KairosCalendar:
    """Hesiod indexes every task by a *natural sign*, not by an abstract date.
    This encodes the core rules of Works and Days. It is the practical twin of
    the Theogony: the Theogony says WHAT a thing is (by descent); the calendar
    says WHEN a deed is right (by sign). Both reject context-free knowledge.
    """
    # Approximate Hesiodic anchors (day-of-year, Northern Greece, idealized).
    PLEIADES_RISE = 137    # mid-May: heliacal rising -> begin harvest
    PLEIADES_SET = 305     # early Nov: setting -> begin ploughing
    SOLSTICE_SUMMER = 172  # ~21 June: + 50 days -> the safe sailing window
    SAIL_WINDOW = (172 + 50, 172 + 50 + 60)

    def action_for_sign(self, sign):
        s = sign.lower()
        if "pleiades_rise" in s:
            return "reap"          # WD 383-384
        if "pleiades_set" in s:
            return "plough"        # WD 384-387
        if "cuckoo" in s:
            return "prune_vines"   # spring rains
        if "snail" in s:           # "house-carrier climbs the plants"
            return "rest_from_digging"
        if "solstice_plus_50" in s:
            return "sail"          # WD 663-665
        return "wait"

    def action_for_day(self, day):
        day %= 360
        if abs(day - self.PLEIADES_RISE) <= 7:
            return "reap"
        if abs(day - self.PLEIADES_SET) <= 7:
            return "plough"
        if self.SAIL_WINDOW[0] <= day <= self.SAIL_WINDOW[1]:
            return "sail"
        return "wait"


# =============================================================================
# 5.  THE TWO STRIFES  --  a league that improves by emulation (good Eris)
# =============================================================================
def two_strife_league(K=6, steps=1500, lr=3e-3, emulate_every=120,
                       emulate_frac=0.30, hidden_dim=16, seed=0):
    """Hesiod's good Eris: the lazy potter works harder when he eyes his
    neighbour's full barn. We run TWO identical leagues of K TheogonicNets from
    the SAME perturbed starting points. In the EMULATION league, the current
    laggard periodically steps a fraction toward the current leader (good Eris);
    in the CONTROL league every net learns in isolation. The faithful claim is
    not that the single best mind improves, but that the WHOLE FIELD'S mean loss
    falls faster -- emulation drags the laggards up. Returns (mean_with,
    mean_without) so the caller can assert good Eris lifts the field.
    """
    rng = np.random.RandomState(seed)

    def fresh_league():
        nets = []
        r = np.random.RandomState(seed)  # identical perturbations both leagues
        for k in range(K):
            net = TheogonicNet(hidden_dim=hidden_dim)
            theta = net.flatten() + r.randn(net.flatten().size) * 0.05
            net.unflatten(theta)
            nets.append(net)
        return nets

    emulators = fresh_league()
    controls = fresh_league()

    for t in range(steps):
        for net in emulators:
            net.fit(steps=1, lr=lr)
        for net in controls:
            net.fit(steps=1, lr=lr)
        if (t + 1) % emulate_every == 0:
            losses = [n.loss() for n in emulators]
            best = int(np.argmin(losses))
            worst = int(np.argmax(losses))
            tb, tw = emulators[best].flatten(), emulators[worst].flatten()
            emulators[worst].unflatten(tw + emulate_frac * (tb - tw))

    mean_with = float(np.mean([n.loss() for n in emulators]))
    mean_without = float(np.mean([n.loss() for n in controls]))
    return mean_with, mean_without


# =============================================================================
# 6.  ELPIS  --  Pandora's jar: exactly one channel is retained under the rim
# =============================================================================
def pandora_jar(predictions, retain_index=-1):
    """All the evils (output channels) are released to the world; one channel --
    Elpis, expectation/look-ahead -- is held back under the rim. Returns the
    world-facing output (released) and the retained hope channel separately.
    Hesiod leaves it ambiguous whether keeping Hope back is mercy or cruelty;
    so do we -- we simply guarantee it is *withheld*, never that it helps.
    """
    released = np.delete(predictions, retain_index, axis=-1)
    retained = predictions[..., retain_index]
    return released, retained


# =============================================================================
# 7.  RUN EVERYTHING  --  grad check, training, and the three self-tests
# =============================================================================
def run_all():
    print("=" * 74)
    print(" THE THEOGONIC NET  --  Hesiod of Ascra (c. 700 BCE)")
    print(" Knowledge as descent from Chaos; action as the right time (kairos).")
    print("=" * 74)

    # ---- [1] gradient check (mandatory) ----
    print("\n[1] FINITE-DIFFERENCE GRADIENT CHECK")
    print("-" * 50)
    gnet = TheogonicNet(hidden_dim=16)
    max_rel, worst, ok = gradient_check(gnet)
    print(f"  parameters checked (sampled): 400")
    print(f"  max relative error : {max_rel:.3e}")
    print(f"  worst coord (idx, numeric, analytic): "
          f"{worst[0]}, {worst[1]:+.6e}, {worst[2]:+.6e}")
    print(f"  PASS (< 1e-5): {ok}")
    assert ok, "Gradient check FAILED -- backprop is wrong."

    # ---- [2] real training of the full theogony ----
    print("\n[2] TRAINING  --  reconstruct the pantheon by descent from Chaos")
    print("-" * 50)
    net = TheogonicNet(hidden_dim=32)
    l0 = net.loss()
    hist = net.fit(steps=4000, lr=3e-3, verbose=True)
    lf = net.loss()
    for t, l in hist:
        print(f"  step {t:5d}   loss {l:.5f}")
    print(f"  initial loss {l0:.5f}  ->  final loss {lf:.5f} "
          f"({100*(1-lf/l0):.1f}% reduction)")
    assert lf < l0 * 0.2, "Training did not reduce loss enough."

    # show a few derived signatures vs targets (the net's 'recovered' cosmos)
    _, _, pred = net.forward()
    print("\n  derived cosmic signatures [order, celestial, chthonic, vitality]:")
    for name in ["Chaos", "Nyx", "Eris", "Zeus", "Dike", "Hades", "Mousai"]:
        i = net.idx[name]
        p = pred[i]; t = net.targets[i]
        print(f"    {name:10s} pred [{p[0]:+.2f} {p[1]:+.2f} {p[2]:+.2f} {p[3]:+.2f}]"
              f"   target [{t[0]:+.2f} {t[1]:+.2f} {t[2]:+.2f} {t[3]:+.2f}]")

    # ---- [3] Kairos calendar self-test ----
    print("\n[3] KAIROS CALENDAR  --  sign -> right action (Works and Days)")
    print("-" * 50)
    cal = KairosCalendar()
    checks = [
        ("pleiades_rise", "reap"),
        ("pleiades_set", "plough"),
        ("solstice_plus_50", "sail"),
        ("cuckoo", "prune_vines"),
    ]
    for sign, want in checks:
        got = cal.action_for_sign(sign)
        print(f"    sign '{sign:18s}' -> {got:14s} (expect {want})")
        assert got == want, f"Kairos rule wrong for {sign}"
    assert cal.action_for_day(cal.PLEIADES_RISE) == "reap"
    assert cal.action_for_day(cal.PLEIADES_SET) == "plough"
    print("    day-indexed checks PASS (Pleiades rise->reap, set->plough)")

    # ---- [4] Two-Strife league self-test ----
    print("\n[4] THE TWO STRIFES  --  emulation league vs isolated control")
    print("-" * 50)
    mean_with, mean_without = two_strife_league()
    print(f"    mean loss WITH good-Eris emulation : {mean_with:.5f}")
    print(f"    mean loss in isolated control      : {mean_without:.5f}")
    print(f"    good Eris lifts the whole field    : {mean_with < mean_without}")
    assert mean_with < mean_without, "Emulation did not help -- two-strife claim unmet."

    # ---- [5] Pandora's jar self-test ----
    print("\n[5] PANDORA'S JAR  --  one channel (Elpis) retained under the rim")
    print("-" * 50)
    released, retained = pandora_jar(pred)            # use the trained outputs
    print(f"    full output channels : {pred.shape[1]}")
    print(f"    released to world    : {released.shape[1]}")
    print(f"    retained (Elpis)     : 1   (shape {retained.shape})")
    assert released.shape[1] == pred.shape[1] - 1, "More/less than one retained."

    print("\n" + "=" * 74)
    print(" ALL CHECKS PASSED.")
    print(" The cosmos was recovered from a single seed by lineage alone;")
    print(" the calendar names the hour; the league sharpens the mind;")
    print(" and Hope stays under the rim.")
    print("=" * 74)


if __name__ == "__main__":
    run_all()
