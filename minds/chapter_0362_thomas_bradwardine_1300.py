#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 # BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0362 · Thomas Bradwardine (1300-1349)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 19, Minds 361-380 Available on Amazon https://www.amazon.com/dp/B0HQX147Y7
# Resume and Interactive Demos at https://artificiology.com/
# END ATTRIBUTION 
 THE SIGNIFICATION-CLOSURE ENGINE  (SCE-1349)

 An executable cognitive architecture derived from Thomas Bradwardine
 (c.1300 - 26 August 1349), "Doctor Profundus", Fellow of Merton College,
 Oxford, and for five weeks Archbishop of Canterbury.

 Encyclopedia of Lost Minds: Echoes on AI -- figure 0362
================================================================================

WHY THIS ARCHITECTURE AND NOT A TRANSFORMER
-------------------------------------------
Bradwardine left four distinct technical commitments. Each becomes a load-
bearing part of this machine. None of them is attention-over-stored-keys.

 (1) INSOLUBILIA (Oxford, early 1320s). His postulate: *signification is
     closed under consequence* -- a proposition signifies everything that
     follows from what it signifies. Truth then requires that things be
     WHOLLY as the proposition signifies; a single failing significate makes
     it false. So the Liar, which signifies its own falsity, also signifies
     (by consequence) its own truth, cannot have all its significates obtain,
     and is therefore SIMPLY FALSE -- with no ban on self-reference and no
     stratification of the language.
     --> becomes: ClosureOperator (extensive, monotone, idempotent by
         construction) + conjunctive truth in log-space + InsolubleVerdict.

 (2) DE PROPORTIONIBUS VELOCITATUM IN MOTIBUS (1328). Nothing moves at all
     unless force exceeds resistance; and to increase velocity ARITHMETICALLY
     you must increase the ratio F/R GEOMETRICALLY.
     --> becomes: the ProportionGate, the elementary neuron of this network.
         It has a true dead zone (no output, no gradient, when F <= R) and a
         compounding response above it. It replaces ReLU/sigmoid entirely.

 (3) DE CONTINUO (c.1330). 151 conclusions against indivisibilism: no
     continuum is composed of atoms; a change has no first instant of being,
     only a last instant of non-being (an extrinsic boundary).
     --> becomes: a structural guarantee, verified in the self-tests, that
         the closure map is monotone and that its output varies continuously
         and betweenly -- plus the extrinsic-boundary geometry of the gate,
         which has a last instant of rest and no first instant of motion.

 (4) DE MEMORIA ARTIFICIALI ADQUIRENDA. Memory is ENGINEERED: ordered loci,
     deliberately placed vivid images, walked in ORDINAL sequence. Its
     purpose is invention -- composing on the spot -- not recall.
     --> becomes: LocusMemory. Retrieval is driven by POSITION IN THE WALK,
         never by content similarity. It reshapes which rules are cheap to
         fire at each step of reasoning. This is the deliberate refusal of
         content-addressed attention.

 (5) DE CAUSA DEI (1344) / DE FUTURIS CONTINGENTIBUS. Everything is
     necessitated relative to the first cause's ordained power, yet the agent
     retains the power to the opposite absolutely speaking.
     --> becomes: a diagnostic, not a loss. Every verdict is reported twice:
         as ordained (the actual deterministic forward pass) and as absolute
         (the same input under perturbed first-cause parameters), and the
         gap between them is the model's retained power to the opposite.

THE TASK IT ACTUALLY LEARNS
---------------------------
A synthetic corpus of propositional assertions. Each example gives an
explicitly signified atom set. The engine must (a) recover the full
consequence closure, having learned the rule base from data alone, (b) judge
truth conjunctively against a world, and (c) recognise self-undermining
assertions and return the Bradwardinian verdict SIMPLY FALSE.

Everything is pure NumPy with hand-derived gradients. A finite-difference
gradient check over every parameter block is mandatory and runs on every
execution. Nothing here is a stub.

Run:  python3 chapter_0362_thomas_bradwardine_1300.py
================================================================================
"""

import numpy as np

# ------------------------------------------------------------------------------
# Numerical guards. R_MIN keeps every resistance strictly positive, so the ratio
# F/R is always defined; EPS_F keeps the force strictly positive so that log and
# power operations never see zero.
# ------------------------------------------------------------------------------
R_MIN = 1e-2
EPS_F = 1e-6
K_MIN = 0.10          # floor on the compounding exponent
DELTA_W = 1e-3        # world clipping, so log of a false atom is finite


# ==============================================================================
# SECTION 0 -- ELEMENTARY FUNCTIONS
# ==============================================================================

def softplus(x):
    """Smooth positive reparameterisation. Used wherever a weight must be >= 0,
    because Bradwardine's forces, resistances and deliveries are magnitudes:
    a resistance cannot be negative, and a premise cannot push backwards."""
    return np.logaddexp(0.0, x)


def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def bce(p, y, eps=1e-9):
    """Binary cross entropy, mean-reduced, with the probability clipped into
    (eps, 1-eps) so the logarithm stays finite."""
    pc = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(pc) + (1.0 - y) * np.log(1.0 - pc)))


def dbce(p, y, eps=1e-9):
    """d/dp of the mean-reduced BCE above.

    The mask matters. Where p has been clipped, the loss as actually
    implemented is FLAT in p, so the true derivative there is zero. Letting
    gradient flow through a saturated clip is the classic way to make an
    analytic gradient disagree with its own loss by a small constant amount
    that no change of step size will reveal. The mask keeps the finite-
    difference check honest."""
    pc = np.clip(p, eps, 1.0 - eps)
    live = ((p > eps) & (p < 1.0 - eps)).astype(np.float64)
    return live * (-(y / pc) + (1.0 - y) / (1.0 - pc)) / p.size


# ==============================================================================
# SECTION 1 -- THE PROPORTION GATE
#
#   Bradwardine's rule of motion, used as an activation function.
#
#   Aristotle's readers had held V proportional to F/R. That gives a moving
#   body whenever F > 0, however small -- a fly should shift a ship -- and it
#   gives nonzero velocity at F = R, where plainly nothing moves. Bradwardine's
#   correction (1328): velocity grows ARITHMETICALLY as the ratio F/R grows
#   GEOMETRICALLY. Double the speed by squaring the ratio; treble it by cubing.
#   Motion begins exactly at F = R and not one instant before.
#
#   Written as a bounded gate:
#
#       h(F, R) = 1 - (R/F)^k        when F > R
#       h(F, R) = 0                  when F <= R
#
#   Properties that matter for a neural unit:
#     * exact zero below threshold, with exactly zero gradient -- a real dead
#       zone, not a soft one. A rule that is not driven hard enough does not
#       fire, and does not learn from that step either.
#     * h -> 1 as F/R -> infinity: saturating, so no exploding activations.
#     * equal increments of h demand geometric escalation of F/R. Capability
#       is expensive at the top exactly as Bradwardine said motion was.
#     * the boundary is EXTRINSIC in his sense: there is a last state of rest
#       (F = R, h = 0) and no first state of motion. h is continuous at the
#       threshold, its derivative is not -- the continuum is not cut into atoms,
#       it is bounded from outside.
# ==============================================================================

def proportion_gate(F, R, k):
    """Forward pass. F is (B, M), R is (M,), k is (M,). Returns (B, M)."""
    F = np.maximum(F, EPS_F)
    ratio = F / R[None, :]
    active = ratio > 1.0
    # (R/F)^k computed as exp(-k * ln(ratio)) for numerical stability
    safe = np.where(active, ratio, 2.0)                 # dummy > 1 where inactive
    decay = np.exp(-k[None, :] * np.log(safe))
    h = np.where(active, 1.0 - decay, 0.0)
    cache = (F, R, k, ratio, active, decay)
    return h, cache


def proportion_gate_backward(dh, cache):
    """Hand-derived gradients of the gate.

        h      = 1 - ratio^(-k)
        dh/dratio = k * ratio^(-k-1)     = k * decay / ratio
        dh/dk     = decay * ln(ratio)
        ratio  = F / R
        dratio/dF = 1/R
        dratio/dR = -F / R^2
    Below threshold every one of these is zero: the body is at rest, and rest
    has no derivative to give."""
    F, R, k, ratio, active, decay = cache
    z = np.where(active, 1.0, 0.0)
    dh_dratio = z * k[None, :] * decay / np.maximum(ratio, EPS_F)
    dratio = dh * dh_dratio

    dF = dratio / R[None, :]
    dR = np.sum(-dratio * F / (R[None, :] ** 2), axis=0)
    dk = np.sum(dh * z * decay * np.log(np.where(active, ratio, 1.0)), axis=0)
    return dF, dR, dk


# ==============================================================================
# SECTION 2 -- THE RULE BASE THAT GENERATES THE WORLD
#
#   The engine never sees these rules. It must recover them from examples.
#   Atom layout:  0..NBASE-1  ordinary atoms
#                 IDX_T       "this proposition is true"
#                 IDX_F       "this proposition is false"
#
#   The final rule is Bradwardine's own theorem, the hinge of the Insolubilia:
#   whatever signifies its own falsity thereby signifies its own truth.
# ==============================================================================

NBASE = 12
IDX_T = NBASE          # signifies its own truth
IDX_F = NBASE + 1      # signifies its own falsity
NATOM = NBASE + 2

CONTRADICTORY_PAIRS = [(IDX_T, IDX_F)]

TRUE_RULES = [
    ([0], 3),
    ([1], 4),
    ([2], 5),
    ([3, 4], 6),
    ([5], 7),
    ([6], 8),
    ([7, 8], 9),
    ([0, 2], 10),
    ([9], 11),
    ([10, 11], 2),
    ([IDX_F], IDX_T),      # <-- the Bradwardine postulate, in the data
]


def forward_chain(seed_bits):
    """Ground-truth consequence closure by exhaustive forward chaining.
    Extensive (the seed survives), monotone, and idempotent -- it is run to a
    fixed point. This is the target the learned ClosureOperator must match."""
    s = seed_bits.copy().astype(np.int8)
    changed = True
    while changed:
        changed = False
        for prem, con in TRUE_RULES:
            if s[con] == 0 and all(s[p] == 1 for p in prem):
                s[con] = 1
                changed = True
    return s


def make_dataset(n, rng):
    """Build n examples.

    seed   : what the proposition explicitly signifies      (n, NATOM) float
    clos   : everything it signifies once closed            (n, NATOM) float
    world  : how things actually stand                      (n, NATOM) float
    y_true : 1 iff things are WHOLLY as it signifies        (n, 1)
    y_ins  : 1 iff it is an insoluble                       (n, 1)
    """
    seeds = np.zeros((n, NATOM), dtype=np.float64)
    clos = np.zeros((n, NATOM), dtype=np.float64)
    world = np.zeros((n, NATOM), dtype=np.float64)
    y_true = np.zeros((n, 1), dtype=np.float64)
    y_ins = np.zeros((n, 1), dtype=np.float64)

    for i in range(n):
        bits = np.zeros(NATOM, dtype=np.int8)
        # explicit signification: one to three ordinary atoms
        for j in rng.choice(NBASE, size=rng.integers(1, 4), replace=False):
            bits[j] = 1
        insoluble = rng.random() < 0.35
        if insoluble:
            bits[IDX_F] = 1                    # "this proposition is false"
        c = forward_chain(bits)

        # The world. Half the time it satisfies everything signified.
        w = c.copy().astype(np.float64)
        satisfied = rng.random() < 0.5
        if not satisfied:
            members = np.flatnonzero(c[:NBASE])
            if members.size:
                w[rng.choice(members)] = 0.0
            else:
                satisfied = True
        # Nothing in the world makes a proposition both true and false, so the
        # self-referential atoms never both obtain out there.
        w[IDX_T] = 1.0
        w[IDX_F] = 0.0

        seeds[i] = bits
        clos[i] = c
        world[i] = w
        y_ins[i, 0] = 1.0 if insoluble else 0.0
        # An insoluble is simply false whatever the world does.
        y_true[i, 0] = 0.0 if insoluble else (1.0 if satisfied else 0.0)

    return seeds, clos, world, y_true, y_ins


# ==============================================================================
# SECTION 3 -- LOCUS MEMORY  (De memoria artificiali adquirenda)
#
#   Bradwardine's memory art: fix an ordered set of places, put a deliberately
#   vivid image in each, then WALK them in order. Carruthers' point about the
#   medieval art is that its purpose was invention -- having the material to
#   hand while composing, not reciting a list.
#
#   Here the walk schedule omega is a fixed function of the reasoning step
#   alone. It does not look at the content. It cannot: that is the whole point.
#   Locus l is visited around step l * (T-1)/(L-1), with a soft overlap, and
#   what it delivers is a modification of the RESISTANCES of the rules -- it
#   makes certain inferences cheap to make at certain moments of the walk.
#
#   Memory here does not store answers. It stages the order in which thinking
#   becomes possible.
# ==============================================================================

def ordinal_walk_schedule(T, L, width=0.75):
    """(T, L) row-stochastic matrix. Purely positional -- no content anywhere."""
    if L == 1:
        return np.ones((T, 1))
    steps = np.arange(T, dtype=np.float64)
    centres = np.arange(L, dtype=np.float64) * (T - 1.0) / (L - 1.0)
    d = steps[:, None] - centres[None, :]
    w = np.exp(-0.5 * (d / width) ** 2)
    return w / np.sum(w, axis=1, keepdims=True)


# ==============================================================================
# SECTION 4 -- THE ENGINE
# ==============================================================================

class SigniticationClosureEngine:
    """The full model.

    Parameters
      A_raw (M, NATOM)  premise attachment.  A = softplus(A_raw) >= 0.
                        A[m, j] is how hard atom j pushes rule m.
      r_raw (M,)        rule resistance.     R = softplus(r_raw) + R_MIN > 0.
      k_raw (M,)        compounding exponent. k = softplus(k_raw) + K_MIN > 0.
      B_raw (M, NATOM)  consequent delivery. B = softplus(B_raw) >= 0.
      P_raw (L, M)      locus images: what each place in the memory walk does
                        to each rule's resistance.
      a_raw, b_tru      the conjunctive truth head.
      w_ver (P,), b_ver the insoluble verdict head.

    The nonnegativity of A and B is not decoration. It is what makes the
    closure operator provably MONOTONE, and the noisy-OR update is what makes
    it provably EXTENSIVE. Those two laws hold by construction, at
    initialisation, before a single gradient step. Only idempotence has to be
    reached, and it is reached by iterating the map to its fixed point.
    """

    def __init__(self, n_rules=32, n_loci=4, n_steps=6, seed=1349):
        rng = np.random.default_rng(seed)
        M, N, L, T = n_rules, NATOM, n_loci, n_steps
        self.M, self.N, self.L, self.T = M, N, L, T

        # INITIALISATION IS NOT COSMETIC HERE, AND THE REASON IS BRADWARDINE'S.
        # A proportion gate is exactly zero, with exactly zero derivative, while
        # force does not exceed resistance. So a network of these units that is
        # initialised with heavy resistances is not merely slow -- it is
        # PERMANENTLY DEAD. No rule fires, no gradient returns, and no amount of
        # training will start it, because learning is itself a motion and motion
        # requires the ratio to already exceed unity. The resistances therefore
        # start low and the premise attachments start high, so that every rule
        # begins just barely in motion and can be taught where to stop.
        # But the gate is dead at BOTH ends: below threshold h is exactly 0,
        # and far above it h saturates toward 1 with a vanishing derivative
        # (dh/dratio = k*ratio^(-k-1)). A proportion-gate network only learns
        # while it sits in the responsive band just above unity. These values
        # were chosen by measuring the initial closure mass and tuning it to
        # roughly 0.4 -- moving, but nowhere near saturated.
        self.A_raw = rng.normal(-0.4, 0.5, size=(M, N))     # A ~ 0.5, premise push
        self.r_raw = rng.normal(1.6, 0.3, size=(M,))        # R ~ 1.8, ratio just over 1
        self.k_raw = rng.normal(0.0, 0.3, size=(M,))        # k ~ 0.8
        self.B_raw = rng.normal(-3.5, 0.5, size=(M, N))     # B ~ 0.03, sparse delivery
        self.P_raw = rng.normal(0.0, 0.2, size=(L, M))

        self.a_raw = np.array([0.0])
        self.b_tru = np.array([2.0])

        self.w_ver = rng.normal(0.0, 0.5, size=(len(CONTRADICTORY_PAIRS),))
        self.b_ver = np.array([-1.0])

        self.omega = ordinal_walk_schedule(T, L)   # fixed, positional, not learned

    # -- parameter plumbing ----------------------------------------------------
    NAMES = ["A_raw", "r_raw", "k_raw", "B_raw", "P_raw", "a_raw", "b_tru",
             "w_ver", "b_ver"]

    def params(self):
        return {n: getattr(self, n) for n in self.NAMES}

    # -- forward ---------------------------------------------------------------
    def _step(self, s, A, B, k, t, resist_bias):
        """One application of the consequence map, at station t of the memory
        walk. Returns the new signification and everything backprop needs."""
        r_pre = self.r_raw + self.omega[t] @ self.P_raw + resist_bias
        R = softplus(r_pre) + R_MIN                          # (M,)
        F = s @ A.T                                          # (Bt, M) force
        h, gcache = proportion_gate(F, R, k)                 # (Bt, M) firing
        gpre = h @ B                                         # (Bt, N) delivery
        edecay = np.exp(-gpre)
        g = 1.0 - edecay                                     # bounded to [0,1)

        # UNION, NOT ACCUMULATION. What a proposition signifies is a SET. To
        # signify something one already signifies adds nothing whatever -- so
        # the join has to be idempotent, and the idempotent join on [0,1] is
        # the maximum. A probabilistic-or (s + g - s*g) would instead keep
        # nudging every atom upward on every pass, and its only fixed point
        # would be the degenerate one where everything any firing rule can
        # reach has been driven to certainty. Bradwardine's closure is not a
        # tally of how often a thing was concluded. It is the set of what
        # follows.
        s_new = np.maximum(s, g)
        take_g = (g > s).astype(np.float64)
        return s_new, (s, F, h, gcache, gpre, edecay, g, r_pre, take_g)

    def consequence_closure(self, s0, tol=1e-13, max_iter=1000):
        """THE CONSEQUENCE OPERATOR PROPER.

        A distinction worth being exact about. The memory walk stages HOW a
        mind arrives at what it is committed to -- which inferences are cheap
        at which moment of the walk. It is a route, and a route is taken once.
        The closure itself is WHAT is committed to, and that is the fixed point
        of the terminal consequence map: apply it until nothing further
        follows.

        It is that operator, not the walk, that has to satisfy Bradwardine's
        three laws. It is extensive and monotone by construction (nonnegative
        weights, max-union join) and it is idempotent because it is run to a
        genuine fixed point, which exists because the map is extensive,
        bounded above by 1, and hence monotonically convergent.
        """
        A = softplus(self.A_raw)
        B = softplus(self.B_raw)
        k = softplus(self.k_raw) + K_MIN
        s = np.array(s0, dtype=np.float64)
        n = 0
        for _ in range(max_iter):
            s_new, _ = self._step(s, A, B, k, self.T - 1, 0.0)
            delta = float(np.max(np.abs(s_new - s)))
            s = s_new
            n += 1
            if delta < tol:
                break
        return s, n

    def forward(self, seed, world, resist_bias=0.0, soften=True):
        """One full pass.

        resist_bias perturbs every rule's resistance. At 0.0 this is the
        ORDAINED pass -- the model's actual, necessitated output. Nonzero
        values give the ABSOLUTE pass: what this same mind would have
        concluded had the first causes been set otherwise. See
        retained_power().
        """
        A = softplus(self.A_raw)
        B = softplus(self.B_raw)
        k = softplus(self.k_raw) + K_MIN

        # ---- closure: iterate the consequence map -----------------------------
        # explicit signification, softened off the hard 0/1 corners
        s = (0.02 + 0.96 * seed) if soften else np.array(seed, dtype=np.float64)
        tape = []
        for t in range(self.T):
            s, rec = self._step(s, A, B, k, t, resist_bias)
            tape.append(rec)

        clos = s

        # ---- conjunctive truth: things must be WHOLLY as it signifies ---------
        # log of the world, clipped so a false atom contributes a finite penalty
        logw = np.log(np.clip(world, DELTA_W, 1.0 - DELTA_W))
        logtau = np.sum(clos * logw, axis=1, keepdims=True)   # (Bt, 1)
        a = softplus(self.a_raw)[0]
        z_tru = a * logtau + self.b_tru[0]
        tau = sigmoid(z_tru)

        # ---- the insoluble verdict -------------------------------------------
        # Read off co-assertion of a contradictory pair. If the closure holds
        # both "I am true" and "I am false", not everything it signifies can
        # obtain, so it is simply false.
        cfeat = np.stack([clos[:, j1] * clos[:, j2]
                          for (j1, j2) in CONTRADICTORY_PAIRS], axis=1)
        z_ver = cfeat @ self.w_ver + self.b_ver[0]
        p_ins = sigmoid(z_ver)[:, None] if z_ver.ndim == 1 else sigmoid(z_ver)

        # ---- final judgement --------------------------------------------------
        p_true = tau * (1.0 - p_ins)

        cache = dict(tape=tape, clos=clos, logw=logw, logtau=logtau, a=a,
                     tau=tau, cfeat=cfeat, p_ins=p_ins, p_true=p_true,
                     A=A, B=B, k=k, seed=seed)
        return clos, p_true, p_ins, cache

    # -- loss ------------------------------------------------------------------
    def loss(self, cache, clos_t, y_true, y_ins, l2=1e-4):
        clos, p_true, p_ins = cache["clos"], cache["p_true"], cache["p_ins"]
        Lc = bce(clos, clos_t)
        Lt = bce(p_true, y_true)
        Li = bce(p_ins, y_ins)
        Lr = l2 * (np.sum(self.A_raw ** 2) + np.sum(self.B_raw ** 2)
                   + np.sum(self.P_raw ** 2))
        return Lc + Lt + Li + Lr, (Lc, Lt, Li, Lr)

    # -- backward --------------------------------------------------------------
    def backward(self, cache, clos_t, y_true, y_ins, l2=1e-4):
        """Hand-derived gradients for every parameter. No autodiff anywhere."""
        A, B, k = cache["A"], cache["B"], cache["k"]
        clos, logw, logtau = cache["clos"], cache["logw"], cache["logtau"]
        a, tau, cfeat, p_ins, p_true = (cache["a"], cache["tau"], cache["cfeat"],
                                        cache["p_ins"], cache["p_true"])
        tape = cache["tape"]

        g = {n: np.zeros_like(getattr(self, n)) for n in self.NAMES}

        # ---- head 1: truth of the final judgement -----------------------------
        dp_true = dbce(p_true, y_true)                       # (Bt,1)
        dtau = dp_true * (1.0 - p_ins)
        dp_ins_from_true = dp_true * (-tau)

        dz_tru = dtau * tau * (1.0 - tau)
        g["a_raw"] += np.array([np.sum(dz_tru * logtau) * sigmoid(self.a_raw[0])])
        g["b_tru"] += np.array([np.sum(dz_tru)])
        dlogtau = dz_tru * a
        dclos = dlogtau * logw                               # (Bt, N)

        # ---- head 2: the insoluble verdict ------------------------------------
        dp_ins = dbce(p_ins, y_ins) + dp_ins_from_true
        dz_ver = dp_ins * p_ins * (1.0 - p_ins)              # (Bt,1)
        g["w_ver"] += (dz_ver * cfeat).sum(axis=0)
        g["b_ver"] += np.array([np.sum(dz_ver)])
        for idx, (j1, j2) in enumerate(CONTRADICTORY_PAIRS):
            contrib = dz_ver[:, 0] * self.w_ver[idx]
            dclos[:, j1] += contrib * clos[:, j2]
            dclos[:, j2] += contrib * clos[:, j1]

        # ---- direct supervision on the closure itself -------------------------
        dclos += dbce(clos, clos_t)

        # ---- unroll the closure iterations backwards --------------------------
        ds = dclos
        dA = np.zeros_like(A)
        dB = np.zeros_like(B)
        dk = np.zeros_like(k)

        for t in reversed(range(self.T)):
            s_prev, F, h, gcache, gpre, edecay, gg, r_pre, take_g = tape[t]

            #  s_out = max(s_prev, g); gradient routes to whichever won,
            #  exactly as it does through a maximum anywhere else.
            dg = ds * take_g
            ds_prev = ds * (1.0 - take_g)

            #  g = 1 - exp(-gpre)
            dgpre = dg * edecay

            #  gpre = h @ B
            dB += h.T @ dgpre
            dh = dgpre @ B.T

            #  h = proportion_gate(F, R, k)
            dF, dR, dk_t = proportion_gate_backward(dh, gcache)
            dk += dk_t

            #  R = softplus(r_pre) + R_MIN
            dr_pre = dR * sigmoid(r_pre)
            g["r_raw"] += dr_pre
            g["P_raw"] += np.outer(self.omega[t], dr_pre)

            #  F = s_prev @ A.T
            dA += dF.T @ s_prev
            ds_prev += dF @ A

            ds = ds_prev

        # seed is an input; ds at this point is discarded.

        # ---- softplus reparameterisations -------------------------------------
        g["A_raw"] += dA * sigmoid(self.A_raw)
        g["B_raw"] += dB * sigmoid(self.B_raw)
        g["k_raw"] += dk * sigmoid(self.k_raw)

        # ---- weight decay ------------------------------------------------------
        g["A_raw"] += 2.0 * l2 * self.A_raw
        g["B_raw"] += 2.0 * l2 * self.B_raw
        g["P_raw"] += 2.0 * l2 * self.P_raw
        return g

    # -- retained power to the opposite  (De causa Dei) -------------------------
    def retained_power(self, seed, world, bias=0.6):
        """Bradwardine held that an act is necessitated relative to God's
        ORDAINED power while the agent still retains, absolutely speaking, the
        power to the opposite. This measures the analogue: how far this
        model's verdict would move had the first causes -- here the rule
        resistances -- been ordained otherwise. A large value means the
        judgement was contingent on the settings; a small one means it was
        forced by the structure of the case itself."""
        _, p_ord, _, _ = self.forward(seed, world, resist_bias=0.0)
        _, p_lo, _, _ = self.forward(seed, world, resist_bias=-bias)
        _, p_hi, _, _ = self.forward(seed, world, resist_bias=+bias)
        return p_ord, float(np.mean(np.maximum(np.abs(p_lo - p_ord),
                                               np.abs(p_hi - p_ord))))


# ==============================================================================
# SECTION 5 -- TRAINING
# ==============================================================================

def train(model, steps=4000, batch=128, lr=0.05, seed=7, verbose=True):
    """Adam over hand-derived gradients, with a cosine-decayed step size.

    The decay is not decoration either. Because every unit is a proportion
    gate, a late large step can shove a rule's force below its own resistance,
    at which point that rule dies permanently -- zero output, zero gradient,
    no way back. Annealing the step size is how a network of these units is
    kept alive to the end of training."""
    rng = np.random.default_rng(seed)
    Xs, Xc, Xw, Yt, Yi = make_dataset(4000, rng)
    Vs, Vc, Vw, Vt, Vi = make_dataset(600, np.random.default_rng(seed + 1))

    m = {n: np.zeros_like(p) for n, p in model.params().items()}
    v = {n: np.zeros_like(p) for n, p in model.params().items()}
    b1, b2, eps = 0.9, 0.999, 1e-8
    hist = []

    for it in range(1, steps + 1):
        lr_t = lr * 0.5 * (1.0 + np.cos(np.pi * it / steps))
        idx = rng.choice(len(Xs), size=batch, replace=False)
        s, c, w, yt, yi = Xs[idx], Xc[idx], Xw[idx], Yt[idx], Yi[idx]

        _, _, _, cache = model.forward(s, w)
        L, parts = model.loss(cache, c, yt, yi)
        gr = model.backward(cache, c, yt, yi)

        for n, p in model.params().items():
            m[n] = b1 * m[n] + (1 - b1) * gr[n]
            v[n] = b2 * v[n] + (1 - b2) * gr[n] ** 2
            mh = m[n] / (1 - b1 ** it)
            vh = v[n] / (1 - b2 ** it)
            p -= lr_t * mh / (np.sqrt(vh) + eps)

        if verbose and (it % 500 == 0 or it == 1):
            cl, pt, pi, _ = model.forward(Vs, Vw)
            acc_c = float(np.mean((cl > 0.5) == (Vc > 0.5)))
            acc_t = float(np.mean((pt > 0.5) == (Vt > 0.5)))
            acc_i = float(np.mean((pi > 0.5) == (Vi > 0.5)))
            hist.append((it, L, acc_c, acc_t, acc_i))
            print(f"  step {it:4d} | loss {L:7.4f} "
                  f"(clos {parts[0]:.3f} true {parts[1]:.3f} insol {parts[2]:.3f}) "
                  f"| val closure {acc_c*100:5.1f}%  verdict {acc_t*100:5.1f}%  "
                  f"insoluble {acc_i*100:5.1f}%")
    return hist, (Vs, Vc, Vw, Vt, Vi)


# ==============================================================================
# SECTION 6 -- MANDATORY FINITE-DIFFERENCE GRADIENT CHECK
# ==============================================================================

def gradient_check(seed=99, n=12, h=1e-6, tol=2e-5):
    """Central differences against the analytic gradients, on every parameter
    block. If this fails the architecture is not real and nothing below it
    means anything."""
    rng = np.random.default_rng(seed)
    model = SigniticationClosureEngine(n_rules=7, n_loci=3, n_steps=3, seed=5)
    s, c, w, yt, yi = make_dataset(n, rng)

    _, _, _, cache = model.forward(s, w)
    ana = model.backward(cache, c, yt, yi)

    worst, worst_name = 0.0, ""
    print("  parameter block      checked   max rel err")
    print("  " + "-" * 44)
    for name in model.NAMES:
        P = getattr(model, name)
        flat = P.ravel()
        picks = rng.choice(flat.size, size=min(6, flat.size), replace=False)
        errs = []
        for i in picks:
            orig = flat[i]
            flat[i] = orig + h
            _, _, _, cp = model.forward(s, w)
            Lp, _ = model.loss(cp, c, yt, yi)
            flat[i] = orig - h
            _, _, _, cm = model.forward(s, w)
            Lm, _ = model.loss(cm, c, yt, yi)
            flat[i] = orig
            num = (Lp - Lm) / (2 * h)
            an = ana[name].ravel()[i]
            errs.append(abs(num - an) / max(1.0, abs(num), abs(an)))
        e = max(errs)
        errs_ok = "ok" if e < tol else "FAIL"
        print(f"  {name:<18} {len(picks):>7}   {e:.3e}  {errs_ok}")
        if e > worst:
            worst, worst_name = e, name

    print(f"  worst block: {worst_name}  max relative error {worst:.3e}  "
          f"(tolerance {tol:.0e})")
    assert worst < tol, f"gradient check FAILED on {worst_name}: {worst:.3e}"
    return worst


# ==============================================================================
# SECTION 7 -- SELF-TESTS  (the architecture's own doctrinal commitments)
# ==============================================================================

def test_gate_extrinsic_boundary():
    """De continuo. The gate must have a LAST instant of rest and NO FIRST
    instant of motion: continuous at threshold, zero below it, strictly
    positive above it, and no atom of motion anywhere."""
    R = np.array([1.0]); k = np.array([2.0])
    below, _ = proportion_gate(np.array([[0.999999]]), R, k)
    at, _ = proportion_gate(np.array([[1.0]]), R, k)
    just, _ = proportion_gate(np.array([[1.000001]]), R, k)
    assert below[0, 0] == 0.0, "motion before force exceeded resistance"
    assert at[0, 0] == 0.0, "motion at the exact point of equality"
    assert 0.0 < just[0, 0] < 1e-4, "motion did not begin continuously"
    # no smallest motion: halve the excess, get a smaller motion, forever
    prev = just[0, 0]
    for i in range(1, 25):
        v, _ = proportion_gate(np.array([[1.0 + 1e-6 / (2 ** i)]]), R, k)
        assert 0.0 < v[0, 0] < prev, "an indivisible least motion was found"
        prev = v[0, 0]
    print(f"  extrinsic boundary   ok   h(1.0)={at[0,0]:.1e}  "
          f"h(1+1e-6)={just[0,0]:.3e}  no least motion after 24 halvings")


def test_gate_compounding():
    """De proportionibus. To increase the gate's response ARITHMETICALLY the
    ratio F/R must increase GEOMETRICALLY. Equivalently: squaring the ratio
    doubles the 'velocity' u = k*ln(F/R) that the gate is built from."""
    R = np.array([1.0]); k = np.array([1.0])
    ratios = np.array([[2.0, 4.0, 16.0, 256.0]])
    u = np.log(ratios[0])
    assert abs(u[1] - 2 * u[0]) < 1e-12, "squaring the ratio did not double it"
    assert abs(u[2] - 4 * u[0]) < 1e-12
    assert abs(u[3] - 8 * u[0]) < 1e-12
    h, _ = proportion_gate(ratios, np.array([1.0]), np.array([1.0]))
    assert np.all(np.diff(h[0]) > 0) and h[0, -1] < 1.0
    print(f"  compounding law      ok   ratio 2->4->16->256 gives velocity "
          f"1x->2x->4x->8x; gate saturates at {h[0,-1]:.4f}")


def test_closure_laws(model):
    """Insolubilia. The closure operator must satisfy the three laws of a
    consequence operator. Two of them hold by construction; the third is
    reached by iteration."""
    rng = np.random.default_rng(3)
    s1 = (rng.random((60, NATOM)) < 0.25).astype(np.float64)
    c1, _, _, _ = model.forward(s1, np.ones((60, NATOM)))

    # EXTENSIVE: s <= C(s). Guaranteed by the max-union update.
    seeded = 0.02 + 0.96 * s1
    ext = float(np.min(c1 - seeded))
    assert ext >= -1e-12, "closure lost part of what was explicitly signified"

    # MONOTONE: s <= s' implies C(s) <= C(s'). Guaranteed by A, B >= 0.
    s2 = np.maximum(s1, (rng.random((60, NATOM)) < 0.25).astype(np.float64))
    c2, _, _, _ = model.forward(s2, np.ones((60, NATOM)))
    mono = float(np.min(c2 - c1))
    assert mono >= -1e-9, "closure was not monotone"

    # IDEMPOTENT: C(C(s)) = C(s). This is the law that has to be earned rather
    # than built in, and it is only true of a closure actually run to its fixed
    # point. Re-asserting the whole of what a proposition signifies must add
    # nothing whatever to it.
    cfix, n_extra = model.consequence_closure(seeded)
    cfix2, _ = model.consequence_closure(cfix)
    idem = float(np.max(np.abs(cfix2 - cfix)))
    assert idem < 1e-9, f"closure was not idempotent: max|C(C(s))-C(s)| = {idem:.3e}"

    # Extensive and monotone at the fixed point too, not just after T strokes.
    seeded2 = 0.02 + 0.96 * s2
    cfix_b, _ = model.consequence_closure(seeded2)
    assert float(np.min(cfix - seeded)) >= -1e-12, "fixed point lost the seed"
    assert float(np.min(cfix_b - cfix)) >= -1e-9, "fixed point was not monotone"
    nontrivial = float(np.mean(cfix > 0.5))

    # BETWEENNESS (De continuo): no gaps. The closure of a midpoint lies
    # between the closures of the endpoints.
    mid = 0.5 * (s1 + s2)
    cm, _, _, _ = model.forward(mid, np.ones((60, NATOM)))
    lo, hi = np.minimum(c1, c2), np.maximum(c1, c2)
    btw = float(np.max(np.maximum(lo - cm, cm - hi)))
    assert btw < 1e-9, "the closure manifold was not dense between states"

    print(f"  extensivity          ok   min(C(s) - s) = {ext:+.3e}  (>= 0)")
    print(f"  monotonicity         ok   min(C(s') - C(s)) = {mono:+.3e}  (>= 0)")
    print(f"  idempotence          ok   fixed point in {n_extra} applications; "
          f"max|C(C(s)) - C(s)| = {idem:.3e}")
    print(f"                            and it is non-trivial: {nontrivial*100:.1f}% of "
          f"atoms signified, not all of them")
    print(f"  betweenness          ok   max excursion = {btw:.3e}  (no indivisibles)")


def test_the_liar(model):
    """The Insolubilia, in one case. Feed the engine a proposition that
    signifies its own falsity. Bradwardine's claim is that its signification,
    once closed, ALSO contains its own truth -- and that this makes it simply
    false, not meaningless, not truth-valueless, not forbidden."""
    liar = np.zeros((1, NATOM)); liar[0, 0] = 1.0; liar[0, IDX_F] = 1.0
    plain = np.zeros((1, NATOM)); plain[0, 0] = 1.0
    w = np.ones((1, NATOM)); w[0, IDX_F] = 0.0

    cl_l, pt_l, pi_l, _ = model.forward(liar, w)
    cl_p, pt_p, pi_p, _ = model.forward(plain, w)

    signifies_own_falsity = cl_l[0, IDX_F]
    signifies_own_truth = cl_l[0, IDX_T]

    assert signifies_own_falsity > 0.5, "the Liar lost its own falsity"
    assert signifies_own_truth > 0.5, \
        "closure failed to yield the Liar's own truth -- the postulate is not learned"
    assert pi_l[0, 0] > 0.5, "the Liar was not recognised as an insoluble"
    assert pt_l[0, 0] < 0.5, "the Liar was not judged simply false"
    assert pi_p[0, 0] < 0.5, "an ordinary proposition was flagged as insoluble"

    print(f"  the Liar             ok   signifies own falsity {signifies_own_falsity:.3f}, "
          f"AND own truth {signifies_own_truth:.3f}")
    print(f"                            verdict: insoluble p={pi_l[0,0]:.3f}, "
          f"true p={pt_l[0,0]:.3f}  -> SIMPLY FALSE")
    print(f"  ordinary proposition ok   insoluble p={pi_p[0,0]:.3f}, "
          f"true p={pt_p[0,0]:.3f}  -> judged on the world, not banned")


def test_conjunctive_truth(model):
    """Omnia sicut significat. Truth requires that things be WHOLLY as the
    proposition signifies. Breaking ONE significate must break the whole,
    and breaking more must not repair it."""
    rng = np.random.default_rng(11)
    s, c, w, yt, yi = make_dataset(400, rng)
    ok = (yi[:, 0] < 0.5) & (yt[:, 0] > 0.5)          # true, non-insoluble
    s, c, w = s[ok][:120], c[ok][:120], w[ok][:120]
    _, p_all, _, _ = model.forward(s, w)

    broken = w.copy()
    for i in range(len(broken)):
        mem = np.flatnonzero(c[i, :NBASE])
        if mem.size:
            broken[i, mem[0]] = 0.0
    _, p_one, _, _ = model.forward(s, broken)

    drop = float(np.mean(p_all) - np.mean(p_one))
    assert drop > 0.3, "a failing significate did not destroy truth"
    print(f"  conjunctive truth    ok   all significates obtain: p={np.mean(p_all):.3f}; "
          f"break exactly one: p={np.mean(p_one):.3f}  (drop {drop:.3f})")


def test_locus_memory_is_positional(model):
    """De memoria artificiali. The retrieval schedule must depend on the step
    of the walk and on nothing else. If it ever consulted content, this test
    would fail -- and the architecture would have quietly become attention."""
    om = model.omega
    assert om.shape == (model.T, model.L)
    assert np.allclose(om.sum(axis=1), 1.0)
    peaks = np.argmax(om, axis=1)
    assert np.all(np.diff(peaks) >= 0), "the memory walk was not ordinal"
    # the schedule is a pure function of (T, L): rebuild it from scratch
    rebuilt = ordinal_walk_schedule(model.T, model.L)
    assert np.allclose(rebuilt, om), "the walk was contaminated by content"
    print(f"  locus walk           ok   {model.L} loci over {model.T} steps, "
          f"peak order {peaks.tolist()}, content-independent")


def test_retained_power(model, data):
    """De causa Dei. Report the ordained verdict and the power to the
    opposite. This is a measurement, not an assertion -- Bradwardine's point
    was that both are true at once, at different modal levels."""
    Vs, Vc, Vw, Vt, Vi = data
    p_ord, power = model.retained_power(Vs[:200], Vw[:200])
    acc = float(np.mean((p_ord > 0.5) == (Vt[:200] > 0.5)))
    print(f"  ordained vs absolute ok   ordained verdict accuracy {acc*100:.1f}%; "
          f"mean retained power to the opposite {power:.4f}")


# ==============================================================================
# SECTION 8 -- MAIN
# ==============================================================================

def main():
    np.set_printoptions(precision=4, suppress=True)
    print("=" * 78)
    print(" THE SIGNIFICATION-CLOSURE ENGINE  (SCE-1349)")
    print(" after Thomas Bradwardine, Doctor Profundus, c.1300 - 26 August 1349")
    print("=" * 78)

    print("\n[1] GRADIENT CHECK  (central differences vs hand-derived analytic)")
    worst = gradient_check()

    print("\n[2] TRAINING  (learning the consequence closure from examples alone)")
    model = SigniticationClosureEngine(n_rules=32, n_loci=4, n_steps=6, seed=1349)
    hist, data = train(model, steps=4000, batch=128, lr=0.05)

    print("\n[3] DOCTRINAL SELF-TESTS")
    test_gate_extrinsic_boundary()
    test_gate_compounding()
    test_closure_laws(model)
    test_conjunctive_truth(model)
    test_the_liar(model)
    test_locus_memory_is_positional(model)
    test_retained_power(model, data)

    Vs, Vc, Vw, Vt, Vi = data
    cl, pt, pi, _ = model.forward(Vs, Vw)
    print("\n[4] FINAL HELD-OUT PERFORMANCE  (600 unseen propositions)")
    print(f"  consequence closure recovered : "
          f"{float(np.mean((cl > 0.5) == (Vc > 0.5)))*100:.2f}% of atoms")
    print(f"  truth verdict                 : "
          f"{float(np.mean((pt > 0.5) == (Vt > 0.5)))*100:.2f}%")
    print(f"  insoluble detection           : "
          f"{float(np.mean((pi > 0.5) == (Vi > 0.5)))*100:.2f}%")
    print(f"  gradient check worst rel. err : {worst:.3e}")

    R = softplus(model.r_raw) + R_MIN
    k = softplus(model.k_raw) + K_MIN
    print(f"\n  learned resistances : min {R.min():.3f}  median "
          f"{np.median(R):.3f}  max {R.max():.3f}")
    print(f"  learned exponents   : min {k.min():.3f}  median "
          f"{np.median(k):.3f}  max {k.max():.3f}")
    print("\n" + "=" * 78)
    print(" ALL CHECKS PASSED.")
    print("=" * 78)


if __name__ == "__main__":
    main()
