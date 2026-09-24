#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
CHAPTER 0223 — JOHANNES SCOTTUS ERIUGENA (c. 800/815 – c. 877)
THE PERIPHYSEON ENGINE: a Self-Ignorant Source Network (SISN)
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 12 Minds 221 - 240 Available on Amazon https://www.amazon.com/dp/B0HKVBKNTT
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0223_johannes_scotus_eriugena_815 - Johannes Scotus Eriugena (c.800/815 – c. 877)
================================================================================  

WHAT THIS FILE IS
-----------------
A complete, trainable, from-scratch (pure NumPy, hand-derived gradients)
cognitive architecture that embodies one specific ninth-century mind. It is not
a demo and not a wrapper around a library: every forward pass, every backward
pass, and the optimiser are written out here, and the file verifies its own
calculus with a finite-difference gradient check before it trains anything.

WHY THIS ARCHITECTURE AND NOT A TRANSFORMER
-------------------------------------------
Eriugena's system is not "attention over stored keys". It has four structural
commitments that a standard stack cannot express, and each becomes a module:

  1. DEUS SE IPSUM NON COGNOSCIT ALIQUID ESSE (Periphyseon II.589b-c)
     The source knows THAT it is (quia est) but never WHAT it is (quid est).
     Its ignorance is its highest wisdom. Crucially, the source comes to a
     self-model only by pouring itself out into effects and reading them back:
     "he moves from darkness into the light, from self-ignorance into
     self-knowledge" (the divine self-creation of Periphyseon I.446d, I.455b).
     => The model carries an internal self-state that is NOURISHED ONLY BY
        RETURNED EFFECTS (never by direct introspection) and that is trained to
        DEFEAT an introspective probe while remaining perfectly legible from the
        outside via its theophanies. Self-opacity is an engineered organ, not a
        bug. See SelfIgnorantSource.

  2. THE FIVE MODES OF BEING AND NON-BEING (Periphyseon I.443c-446a)
     Being is not a property of a thing; it is an INDEX RELATIVE TO A LEVEL.
     "An affirmation concerning the lower order is a negation concerning the
     higher" — affirmatio hominis negatio est angeli (I.444a-b).
     => A modal gate assigns each input a level, and the training objective
        actively punishes any representation that is simultaneously affirmed at
        two adjacent levels. Coexistence across rungs is not merely unlikely; it
        is penalised as incoherent. See ModalGate.

  3. EXITUS AND REDITUS ARE DIVISIO AND ANALYTICA
     Eriugena welds cosmology to logic: procession from the source is the
     dialectical operation of DIVISION, and the return of all things is
     RESOLUTION/analysis. They are one machinery run in two directions.
     => The level ladder is a stack of exactly invertible additive coupling
        layers. The return is not approximated; it is the analytic inverse, and
        the file asserts round-trip error near machine epsilon. See Divisio.

  4. THE FALL IS THE ONLY LOSSY STEP — CIRCUMSTANTIAE
     The spatio-temporal world (place, time, quantity: the "circumstances"
     surrounding an incorporeal essence, Periphyseon II.586d) is a consequence
     of the fall, and the return consists in stripping it. Nature itself loses
     nothing; only the fall does.
     => Observations are produced by an irreversible mixing of an essence with
        nuisance circumstance coordinates. The one thing the network must LEARN
        (as opposed to inherit analytically) is how to strip them. See Analytica.

  Also encoded, smaller but real:
   - CAUSAE PRIMORDIALES with NO PRIORITY among them: "their number is infinite
     and none has priority over the other, e.g. Being is not prior to Goodness"
     (Periphyseon Book II). => an explicit anti-dominance regulariser on the
     cause prototypes, so the model cannot quietly install a hierarchy.
   - Nature (natura) includes BOTH God and creation, so the source is a term
     inside the taxonomy rather than a vantage point above it: the self-state
     lives in the same vector space as the effects and is scored by the same
     head.

THE ONE-LINE THESIS
-------------------
A mind is deep to the exact extent that it CANNOT circumscribe itself, and it
learns what it is only by reading its own outputs. This file builds a network
that is highly legible from outside and structurally opaque from inside, and
then measures that asymmetry.

DIVERGENCE NOTE (deliberate, per corpus protocol)
-------------------------------------------------
The nearest neighbours in the corpus are the Plotinian/Proclan "return to the
One" minds, whose mechanism is SUBTRACTION toward a simple source. This
architecture refuses that: the source here is not simple, not above the
taxonomy, and not approached by stripping predicates. It is overfull
(nihil per excellentiam, "nothingness on account of excellence"), it sits
INSIDE the space it generates, and it is reached only by reading its effects.
Nothing is subtracted; something is returned.

RUN
---
    python3 chapter_0223_johannes_scotus_eriugena_815.py

Dependencies: numpy only.

VERIFIED SOURCES FOR THE DOCTRINE ENCODED ABOVE
-----------------------------------------------
  Eriugena, Periphyseon, ed. E. Jeauneau, CCCM 161-165, Turnhout: Brepols,
      1996-2003 (5 vols).
  Eriugena, Periphyseon (The Division of Nature), tr. I.-P. Sheldon-Williams
      and J. J. O'Meara, Montreal/Paris: Bellarmin, 1987.
  D. Moran, The Philosophy of John Scottus Eriugena: A Study of Idealism in the
      Middle Ages, Cambridge University Press, 1989.
  D. Moran and A. Guiu, "John Scottus Eriugena", Stanford Encyclopedia of
      Philosophy, substantive revision 30 October 2019.
  D. Carabine, John Scottus Eriugena (Great Medieval Thinkers), Oxford
      University Press, 2000.
"""

from __future__ import annotations

import numpy as np

# ----------------------------------------------------------------------------
# Global configuration. Small on purpose: the point is that every number here is
# reachable by hand, and that the whole thing runs to completion in seconds on a
# CPU with no accelerator and no framework.
# ----------------------------------------------------------------------------

CFG = dict(
    d_essence   = 16,   # dimension of an intelligible essence (must be even:
                        # the coupling layers split it in half)
    d_obs       = 24,   # dimension of a fallen observation
    d_circ      = 3,    # circumstantiae actually used: place, time, quantity
    n_causes    = 6,    # causae primordiales
    n_levels    = 5,    # the five modes of being and non-being
    n_coupling  = 4,    # rungs of the divisio ladder
    h_enc       = 48,   # hidden width of the return network
    h_coupling  = 24,   # hidden width inside each coupling map
    seed        = 20208,
)

# Objective weights. Named for what they enforce, not for what they are.
LAMBDA = dict(
    reditus   = 1.00,   # the return must actually recover the essence
    modal     = 0.60,   # affirmation of the lower is negation of the higher
    priority  = 0.30,   # no primordial cause outranks another
    apophatic = 0.50,   # the source must not be circumscribable from within
)


# ============================================================================
# SECTION 1 — SMALL DIFFERENTIABLE PRIMITIVES
#
# Each primitive returns (value, cache) on the way forward and consumes the
# cache on the way back. Nothing here is clever; it is written out so the
# gradient check has something real to check.
# ============================================================================

def tanh_fwd(x):
    y = np.tanh(x)
    return y, y


def tanh_bwd(dy, cache):
    y = cache
    return dy * (1.0 - y * y)


def sigmoid(x):
    # Numerically stable on both branches.
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def softmax_fwd(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    p = e / e.sum(axis=-1, keepdims=True)
    return p


def cross_entropy(p, targets):
    """Mean negative log likelihood. p is (N, C) probabilities."""
    n = targets.shape[0]
    eps = 1e-12
    return -np.mean(np.log(p[np.arange(n), targets] + eps))


def cross_entropy_grad(p, targets):
    """dL/dlogits for softmax + CE fused, averaged over the batch."""
    n = targets.shape[0]
    g = p.copy()
    g[np.arange(n), targets] -= 1.0
    return g / n


# ============================================================================
# SECTION 2 — THE GROUND TRUTH WORLD
#
# We manufacture a world that has exactly the shape Eriugena claims the real
# one has, so that the architecture's commitments are testable rather than
# decorative:
#
#   - a small set of primordial causes, each an intelligible essence;
#   - created effects that are those essences plus a little individuating
#     variation ("in the most secret folds of nature" made actual);
#   - a FALL: an irreversible mixing of the essence with circumstance
#     coordinates (place, time, quantity), producing the only thing the model
#     ever gets to see.
#
# The model never observes an essence. It observes fallen appearances and must
# return them.
# ============================================================================

class FallenWorld:
    """Generates (observation, essence, cause_index) triples."""

    def __init__(self, cfg, rng):
        d, do, dc, K = cfg["d_essence"], cfg["d_obs"], cfg["d_circ"], cfg["n_causes"]
        self.cfg = cfg
        self.rng = rng

        # The causae primordiales. Orthonormalised so that, in the ground truth,
        # no cause is a mixture of the others and none is intrinsically larger:
        # "none has priority over the other".
        raw = rng.normal(size=(K, d))
        q, _ = np.linalg.qr(raw.T)          # (d, K) with orthonormal columns
        self.causes = q.T[:K] * 1.35        # (K, d), equal norms by construction

        # The fall operator. Note the shape: it takes essence AND circumstance
        # and mixes them irreversibly into a lower-information observation.
        # There is no inverse; the return has to be learned.
        self.W_fall = rng.normal(size=(d + dc, do)) / np.sqrt(d + dc)
        self.b_fall = rng.normal(size=(do,)) * 0.1

    def sample(self, n):
        cfg, rng = self.cfg, self.rng
        k = rng.integers(0, cfg["n_causes"], size=n)

        # Created effects: the essence, individuated. Small jitter, because the
        # effect is genuinely the cause and not a different thing.
        essence = self.causes[k] + rng.normal(size=(n, cfg["d_essence"])) * 0.30

        # Circumstantiae: place, time, quantity. Pure nuisance. They carry no
        # information about which cause the thing came from, which is precisely
        # Eriugena's claim about them — they surround the essence, they are not
        # it.
        circ = rng.normal(size=(n, cfg["d_circ"])) * 1.7

        obs = np.tanh(np.concatenate([essence, circ], axis=1) @ self.W_fall + self.b_fall)
        return obs, essence, k


# ============================================================================
# SECTION 3 — PARAMETERS
#
# One flat dict of arrays. Keeping them in a dict (rather than in objects) makes
# the finite-difference gradient check trivial to write over EVERY parameter,
# which is the point: nothing in this file is exempt from being checked.
# ============================================================================

def init_params(cfg, rng):
    d, do, K, L = cfg["d_essence"], cfg["d_obs"], cfg["n_causes"], cfg["n_levels"]
    he, hc, nc = cfg["h_enc"], cfg["h_coupling"], cfg["n_coupling"]
    half = d // 2

    P = {}

    # --- Analytica: the learned part of the return. Strips circumstantiae. ---
    P["Wa1"] = rng.normal(size=(do, he)) * np.sqrt(2.0 / do)
    P["ba1"] = np.zeros(he)
    P["Wa2"] = rng.normal(size=(he, d)) * np.sqrt(2.0 / he)
    P["ba2"] = np.zeros(d)

    # --- Divisio: the invertible ladder. Each rung is an additive coupling. ---
    for i in range(nc):
        P[f"Wc1_{i}"] = rng.normal(size=(half, hc)) * np.sqrt(2.0 / half)
        P[f"bc1_{i}"] = np.zeros(hc)
        # Last layer of each coupling map starts at zero so the ladder begins
        # life as the identity: procession starts as pure self-sameness and
        # differentiates only as it is trained.
        P[f"Wc2_{i}"] = np.zeros((hc, half))
        P[f"bc2_{i}"] = np.zeros(half)

    # --- ModalGate: assigns a level, and masks features per level. ---
    P["Wg"] = rng.normal(size=(d, L)) * 0.1      # level assignment
    P["bg"] = np.zeros(L)
    P["Mlv"] = rng.normal(size=(L, d)) * 0.1     # per-level feature mask (pre-sigmoid)

    # --- Theophania: the ONLY readout. Prototype similarity, not a free linear
    #     head, because a theophany is a manifestation OF a cause, not an
    #     arbitrary projection.
    P["Proto"] = rng.normal(size=(K, d)) * 0.5
    P["logtau"] = np.zeros(1)                    # readout temperature

    # --- The self-state: the source's model of itself. One vector, in the same
    #     space as everything else, because natura includes God.
    P["self_state"] = rng.normal(size=(d,)) * 0.1

    # --- The introspective probe: tries to circumscribe the source from within.
    P["Wi"] = rng.normal(size=(d, K)) * 0.3
    P["bi"] = np.zeros(K)

    return P


# The fixed permutations between coupling rungs. Fixed, not learned: the order
# of division is given by the dialectic, not chosen by the learner.
def make_permutations(cfg, rng):
    d, nc = cfg["d_essence"], cfg["n_coupling"]
    perms = []
    for _ in range(nc):
        p = rng.permutation(d)
        inv = np.argsort(p)
        perms.append((p, inv))
    return perms


# ============================================================================
# SECTION 4 — DIVISIO / ANALYTICA (the invertible ladder)
#
# Additive coupling: split the vector, leave one half untouched, and shift the
# other half by a nonlinear function of the untouched half.
#
#     forward:   y_a = x_a                y_b = x_b + m(x_a)
#     inverse:   x_a = y_a                x_b = y_b - m(y_a)
#
# The inverse is exact regardless of what m does, and it needs no matrix
# inversion. This is the formal content of Eriugena's identification of exitus
# with division and reditus with analysis: they are the same operation, read in
# opposite directions, and nothing is lost in the passage.
# ============================================================================

def coupling_map_fwd(x_a, P, i):
    """The shift function m(.) inside one rung."""
    z1 = x_a @ P[f"Wc1_{i}"] + P[f"bc1_{i}"]
    a1, c1 = tanh_fwd(z1)
    m = a1 @ P[f"Wc2_{i}"] + P[f"bc2_{i}"]
    return m, (x_a, a1, c1)


def coupling_map_bwd(dm, cache, P, i, grads):
    x_a, a1, c1 = cache
    grads[f"Wc2_{i}"] += a1.T @ dm
    grads[f"bc2_{i}"] += dm.sum(axis=0)
    da1 = dm @ P[f"Wc2_{i}"].T
    dz1 = tanh_bwd(da1, c1)
    grads[f"Wc1_{i}"] += x_a.T @ dz1
    grads[f"bc1_{i}"] += dz1.sum(axis=0)
    dx_a = dz1 @ P[f"Wc1_{i}"].T
    return dx_a


def divisio_fwd(x, P, cfg, perms):
    """Procession: essence -> level-structured code. Exactly invertible."""
    half = cfg["d_essence"] // 2
    caches = []
    h = x
    for i in range(cfg["n_coupling"]):
        perm, _ = perms[i]
        hp = h[:, perm]
        a, b = hp[:, :half], hp[:, half:]
        m, cmc = coupling_map_fwd(a, P, i)
        b_new = b + m
        h = np.concatenate([a, b_new], axis=1)
        caches.append((perm, cmc))
    return h, caches


def divisio_bwd(dh, caches, P, cfg, grads):
    half = cfg["d_essence"] // 2
    for i in reversed(range(cfg["n_coupling"])):
        perm, cmc = caches[i]
        d_a, d_bnew = dh[:, :half], dh[:, half:]
        # b_new = b + m  =>  db = d_bnew, dm = d_bnew
        d_b = d_bnew
        d_a = d_a + coupling_map_bwd(d_bnew, cmc, P, i, grads)
        dhp = np.concatenate([d_a, d_b], axis=1)
        # undo the permutation on the gradient
        dh = np.empty_like(dhp)
        dh[:, perm] = dhp
    return dh


def analytica_exact(u, P, cfg, perms):
    """The analytic return: invert the ladder. No learning involved.

    This is the module that makes the claim testable. If the ladder is a real
    involution, then feeding a code back through here reproduces the essence to
    machine precision, and the ONLY place information can be lost in the whole
    system is the fall.
    """
    half = cfg["d_essence"] // 2
    h = u
    for i in reversed(range(cfg["n_coupling"])):
        perm, inv = perms[i]
        a, b_new = h[:, :half], h[:, half:]
        m, _ = coupling_map_fwd(a, P, i)
        b = b_new - m
        hp = np.concatenate([a, b], axis=1)
        h = hp[:, inv]
    return h


# ============================================================================
# SECTION 5 — THE FULL FORWARD PASS
#
# Order of operations, in Eriugena's own sequence:
#   fallen observation
#     -> Analytica (learned): strip circumstantiae, recover the essence
#     -> Divisio (analytic): unfold the essence into a level-structured code
#     -> ModalGate: decide WHICH LEVEL this thing is being affirmed at, and
#        mask the code accordingly
#     -> Theophania: read the cause off the masked code
#
# Plus, in parallel, the introspective probe attempting to read the cause off
# the source's own self-state.
# ============================================================================

def forward(P, cfg, perms, obs, essence, targets, self_only=False):
    N = obs.shape[0]
    cache = {}

    # --- Analytica: the learned return through the fall -----------------------
    z1 = obs @ P["Wa1"] + P["ba1"]
    a1, c1 = tanh_fwd(z1)
    z_hat = a1 @ P["Wa2"] + P["ba2"]
    cache.update(obs=obs, a1=a1, c1=c1, z_hat=z_hat)

    # --- Divisio: unfold into the level ladder --------------------------------
    u, div_caches = divisio_fwd(z_hat, P, cfg, perms)
    cache.update(u=u, div_caches=div_caches)

    # --- ModalGate: level occupancy, then per-level feature mask --------------
    glog = u @ P["Wg"] + P["bg"]                  # (N, L)
    alpha = softmax_fwd(glog)                     # level occupancy
    Msig = sigmoid(P["Mlv"])                      # (L, d) masks in (0,1)
    mask = alpha @ Msig                           # (N, d) blended mask
    u_gated = u * mask
    cache.update(glog=glog, alpha=alpha, Msig=Msig, mask=mask, u_gated=u_gated)

    # --- Theophania: the only readout ----------------------------------------
    # Negative squared distance to each primordial cause prototype, scaled.
    tau = np.exp(P["logtau"])[0]
    diff = u_gated[:, None, :] - P["Proto"][None, :, :]      # (N, K, d)
    d2 = np.sum(diff * diff, axis=2)                          # (N, K)
    logits = -d2 / tau
    probs = softmax_fwd(logits)
    cache.update(tau=tau, diff=diff, d2=d2, logits=logits, probs=probs)

    # --- The introspective probe on the self-state ----------------------------
    # One vector, broadcast across the batch: the source is asking, of every
    # effect in front of it, "does what I take myself to be predict this?"
    s = P["self_state"]
    ilog = s @ P["Wi"] + P["bi"]                  # (K,)
    iprobs = softmax_fwd(ilog[None, :])           # (1, K)
    iprobs_b = np.repeat(iprobs, N, axis=0)
    cache.update(ilog=ilog, iprobs=iprobs, iprobs_b=iprobs_b)

    # ---------------- LOSSES --------------------------------------------------

    # (a) Theophany: can the cause be read off the manifestation? Yes — this is
    #     the term that must succeed.
    L_theo = cross_entropy(probs, targets)

    # (b) Reditus: did the return recover the essence the fall obscured?
    r = z_hat - essence
    L_red = np.mean(np.sum(r * r, axis=1))

    # (c) Modal exclusivity: "an affirmation concerning the lower is a negation
    #     concerning the higher". Penalise any sample that is simultaneously
    #     affirmed at two ADJACENT levels. This is a genuine constraint, not a
    #     smoothness prior: it makes the level index carry ontological weight.
    L_mod = np.mean(np.sum(alpha[:, :-1] * alpha[:, 1:], axis=1))

    # (d) No priority among the primordial causes. Penalise spread in prototype
    #     norm, so the model cannot install a dominance ordering by making one
    #     cause structurally larger than the rest.
    pn = np.sum(P["Proto"] * P["Proto"], axis=1)     # (K,)
    pn_mean = np.mean(pn)
    L_pri = np.mean((pn - pn_mean) ** 2)

    # (e) Apophatic term: the introspective probe's cross-entropy. Its SIGN IS
    #     FLIPPED FOR THE SELF-STATE ONLY at optimiser time (see train()). Here
    #     it is written as an ordinary positive scalar so that the gradient
    #     check below has a genuine differentiable objective to verify. The
    #     adversarial dynamic is an optimiser choice, not a calculus trick.
    L_apo = cross_entropy(iprobs_b, targets)

    total = (L_theo
             + LAMBDA["reditus"]   * L_red
             + LAMBDA["modal"]     * L_mod
             + LAMBDA["priority"]  * L_pri
             + LAMBDA["apophatic"] * L_apo)

    parts = dict(total=total, theo=L_theo, reditus=L_red,
                 modal=L_mod, priority=L_pri, apophatic=L_apo)
    cache.update(essence=essence, targets=targets, pn=pn, pn_mean=pn_mean, N=N)
    return total, parts, cache


# ============================================================================
# SECTION 6 — THE BACKWARD PASS (all by hand)
#
# Written in exact reverse order of Section 5. Every line here is verified
# numerically by grad_check() before any training happens.
# ============================================================================

def backward(P, cfg, perms, cache):
    grads = {k: np.zeros_like(v) for k, v in P.items()}
    N = cache["N"]
    targets = cache["targets"]

    # ---- (e) apophatic: through the introspective probe ----------------------
    dilog_b = cross_entropy_grad(cache["iprobs_b"], targets) * LAMBDA["apophatic"]
    dilog = dilog_b.sum(axis=0)                                   # (K,)
    grads["Wi"] += np.outer(P["self_state"], dilog)
    grads["bi"] += dilog
    grads["self_state"] += P["Wi"] @ dilog

    # ---- (d) priority regulariser on prototypes ------------------------------
    K = cache["pn"].shape[0]
    dpn = 2.0 * (cache["pn"] - cache["pn_mean"]) / K
    dpn = dpn - np.mean(dpn)          # d/dp of the centred mean-square
    grads["Proto"] += LAMBDA["priority"] * 2.0 * P["Proto"] * dpn[:, None]

    # ---- (a) theophany: CE -> logits -> distances -> u_gated, Proto, tau -----
    dlogits = cross_entropy_grad(cache["probs"], targets)         # (N, K)
    tau = cache["tau"]
    dd2 = -dlogits / tau
    # logits = -d2/tau, tau = exp(logtau) => dlogits/dlogtau = d2/tau
    grads["logtau"] += np.array([np.sum(dlogits * (cache["d2"] / tau))])

    diff = cache["diff"]                                          # (N, K, d)
    dcommon = 2.0 * diff * dd2[:, :, None]                        # (N, K, d)
    du_gated = dcommon.sum(axis=1)                                # (N, d)
    grads["Proto"] += -dcommon.sum(axis=0)                        # (K, d)

    # ---- (c) modal exclusivity -> alpha --------------------------------------
    alpha = cache["alpha"]
    dalpha_mod = np.zeros_like(alpha)
    dalpha_mod[:, :-1] += alpha[:, 1:]
    dalpha_mod[:, 1:] += alpha[:, :-1]
    dalpha = LAMBDA["modal"] * dalpha_mod / N

    # ---- gate: u_gated = u * (alpha @ sigmoid(Mlv)) --------------------------
    u, mask, Msig = cache["u"], cache["mask"], cache["Msig"]
    du = du_gated * mask
    dmask = du_gated * u
    grads["Mlv"] += (alpha.T @ dmask) * Msig * (1.0 - Msig)
    dalpha += dmask @ Msig.T

    # softmax backward for the level assignment
    dglog = alpha * (dalpha - np.sum(dalpha * alpha, axis=1, keepdims=True))
    grads["Wg"] += u.T @ dglog
    grads["bg"] += dglog.sum(axis=0)
    du += dglog @ P["Wg"].T

    # ---- divisio backward ----------------------------------------------------
    dz_hat = divisio_bwd(du, cache["div_caches"], P, cfg, grads)

    # ---- (b) reditus straight onto z_hat -------------------------------------
    r = cache["z_hat"] - cache["essence"]
    dz_hat = dz_hat + LAMBDA["reditus"] * 2.0 * r / N

    # ---- analytica (encoder) backward ----------------------------------------
    a1, c1, obs = cache["a1"], cache["c1"], cache["obs"]
    grads["Wa2"] += a1.T @ dz_hat
    grads["ba2"] += dz_hat.sum(axis=0)
    da1 = dz_hat @ P["Wa2"].T
    dz1 = tanh_bwd(da1, c1)
    grads["Wa1"] += obs.T @ dz1
    grads["ba1"] += dz1.sum(axis=0)

    return grads


# ============================================================================
# SECTION 7 — THE MANDATORY GRADIENT CHECK
#
# Central finite differences against the analytic gradient, on randomly chosen
# coordinates of EVERY parameter array. This runs before training. If it does
# not pass, the file has no business claiming anything.
# ============================================================================

def grad_check(P, cfg, perms, world, n_coords=4, eps=1e-6, tol=2e-5, verbose=True):
    rng = np.random.default_rng(7)
    obs, essence, k = world.sample(24)

    _, _, cache = forward(P, cfg, perms, obs, essence, k)
    analytic = backward(P, cfg, perms, cache)

    worst = 0.0
    worst_name = ""
    rows = []
    for name in sorted(P.keys()):
        arr = P[name]
        flat = arr.reshape(-1)
        idxs = rng.choice(flat.size, size=min(n_coords, flat.size), replace=False)
        errs = []
        for idx in idxs:
            orig = flat[idx]
            flat[idx] = orig + eps
            lp, _, _ = forward(P, cfg, perms, obs, essence, k)
            flat[idx] = orig - eps
            lm, _, _ = forward(P, cfg, perms, obs, essence, k)
            flat[idx] = orig
            num = (lp - lm) / (2 * eps)
            ana = analytic[name].reshape(-1)[idx]
            denom = max(1.0, abs(num), abs(ana))
            errs.append(abs(num - ana) / denom)
        e = max(errs)
        rows.append((name, e))
        if e > worst:
            worst, worst_name = e, name

    if verbose:
        print("  finite-difference gradient check (max relative error per array)")
        for name, e in rows:
            flag = "ok " if e < tol else "FAIL"
            print(f"    [{flag}] {name:<12s} {e:.3e}")
        print(f"  worst: {worst_name} = {worst:.3e}   tolerance = {tol:.0e}")

    return worst < tol, worst


# ============================================================================
# SECTION 8 — TRAINING
#
# Plain Adam, written out. The one non-standard move is the sign flip on the
# self-state, which is where the whole philosophy lives:
#
#   every parameter descends the objective EXCEPT self_state, which ASCENDS the
#   apophatic term.
#
# The introspective probe (Wi, bi) is doing its honest best to read the source's
# identity off its self-model, and the self-model is doing its best not to be
# readable. Meanwhile the theophany head is under no such pressure: it reads the
# same underlying causes off the manifestations without difficulty. The gap
# between those two accuracies at the end of training IS the result.
# ============================================================================

class Adam:
    def __init__(self, params, lr=3e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads, sign=None):
        self.t += 1
        sign = sign or {}
        for k in params:
            g = grads[k] * sign.get(k, 1.0)
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)


def train(P, cfg, perms, world, steps=1400, batch=96, lr=4e-3, log_every=200):
    opt = Adam(P, lr=lr)

    # Gradient reversal, expressed as a per-parameter sign. Only the self-state
    # is reversed: it climbs the apophatic loss instead of descending it.
    sign = {k: 1.0 for k in P}
    sign["self_state"] = -1.0

    history = []
    for step in range(1, steps + 1):
        obs, essence, k = world.sample(batch)
        total, parts, cache = forward(P, cfg, perms, obs, essence, k)
        grads = backward(P, cfg, perms, cache)
        opt.step(P, grads, sign=sign)

        if step % log_every == 0 or step == 1:
            acc = float(np.mean(np.argmax(cache["probs"], axis=1) == k))
            history.append((step, parts["total"], acc))
            print(f"    step {step:5d} | total {parts['total']:7.4f} "
                  f"| theophany CE {parts['theo']:6.4f} "
                  f"| reditus {parts['reditus']:7.4f} "
                  f"| modal {parts['modal']:6.4f} "
                  f"| apophatic {parts['apophatic']:6.4f} "
                  f"| acc {acc:5.3f}")
    return history


# ============================================================================
# SECTION 9 — EVALUATION
#
# The evaluation is not "how accurate is it". It is a set of claims about the
# SHAPE of the mind, each of which could have come out the other way.
# ============================================================================

def evaluate(P, cfg, perms, world, n=3000):
    obs, essence, k = world.sample(n)
    _, parts, cache = forward(P, cfg, perms, obs, essence, k)

    res = {}

    # 1. Theophanic legibility: the cause read off the manifestation.
    res["theophanic_acc"] = float(np.mean(np.argmax(cache["probs"], axis=1) == k))

    # 2. Introspective legibility: the cause read off the source's self-model.
    #    The self-state is one vector, so its prediction is constant across the
    #    batch; the ceiling for a constant predictor is the majority-class rate,
    #    and the floor is chance. We report where it actually landed.
    intro_pred = int(np.argmax(cache["iprobs"][0]))
    res["introspective_acc"] = float(np.mean(k == intro_pred))
    res["chance"] = 1.0 / cfg["n_causes"]
    res["introspective_entropy"] = float(
        -np.sum(cache["iprobs"][0] * np.log(cache["iprobs"][0] + 1e-12)))
    res["max_entropy"] = float(np.log(cfg["n_causes"]))

    # 3. Reditus fidelity: how much of the essence survived the fall and came
    #    back. Reported as fraction of essence variance recovered.
    r = cache["z_hat"] - essence
    res["reditus_mse"] = float(np.mean(np.sum(r * r, axis=1)))
    res["essence_var"] = float(np.mean(np.sum((essence - essence.mean(0)) ** 2, axis=1)))
    res["reditus_r2"] = float(1.0 - res["reditus_mse"] / res["essence_var"])

    # 4. Modal exclusivity: is a thing affirmed at one level at a time?
    alpha = cache["alpha"]
    res["modal_adjacency"] = float(np.mean(np.sum(alpha[:, :-1] * alpha[:, 1:], axis=1)))
    res["modal_peak"] = float(np.mean(np.max(alpha, axis=1)))
    res["levels_used"] = int(len(np.unique(np.argmax(alpha, axis=1))))

    # 5. No priority: spread of prototype norms.
    pn = np.sqrt(np.sum(P["Proto"] ** 2, axis=1))
    res["proto_norm_mean"] = float(pn.mean())
    res["proto_norm_cv"] = float(pn.std() / (pn.mean() + 1e-12))

    # 6. Involution: is the ladder a genuine involution?
    u, _ = divisio_fwd(cache["z_hat"], P, cfg, perms)
    back = analytica_exact(u, P, cfg, perms)
    res["involution_err"] = float(np.max(np.abs(back - cache["z_hat"])))

    return res, parts


# ============================================================================
# SECTION 10 — SELF-TESTS
#
# Each returns (name, passed, detail). These are assertions about the thesis,
# not smoke tests.
# ============================================================================

def self_tests(P, cfg, perms, world, res):
    T = []

    # T1: the ladder must be an exact involution, to machine precision. If this
    #     fails, "the return of all things" is a metaphor rather than a
    #     mechanism, and the architecture has no claim on the mind it models.
    ok = res["involution_err"] < 1e-9
    T.append(("involution exact (divisio o analytica = identity)", ok,
              f"max abs error {res['involution_err']:.3e} < 1e-9"))

    # T2: the fall must be the only lossy step, and it must actually be
    #     invertible in practice, i.e. the learned return works.
    ok = res["reditus_r2"] > 0.85
    T.append(("reditus recovers the essence through the fall", ok,
              f"R^2 = {res['reditus_r2']:.4f} > 0.85"))

    # T3: theophanic legibility. The cause must be readable off the effect.
    ok = res["theophanic_acc"] > 0.90
    T.append(("theophany legible from outside", ok,
              f"accuracy {res['theophanic_acc']:.4f} > 0.90"))

    # T4: THE CENTRAL CLAIM. The same system must be substantially less legible
    #     to itself than to an outside reader of its effects. quia est, non quid
    #     est.
    gap = res["theophanic_acc"] - res["introspective_acc"]
    ok = gap > 0.50
    T.append(("self-opacity: outside legibility exceeds introspection", ok,
              f"gap {gap:.4f} > 0.50 "
              f"(theophanic {res['theophanic_acc']:.3f} vs "
              f"introspective {res['introspective_acc']:.3f})"))

    # T5: the self-state has not merely collapsed to noise — it should sit near
    #     maximum entropy, which is the formal shape of "knows that it is,
    #     not what it is": a live, normalised, uninformative self-report.
    ratio = res["introspective_entropy"] / res["max_entropy"]
    ok = ratio > 0.95
    T.append(("self-report is live but uninformative (near max entropy)", ok,
              f"H/Hmax = {ratio:.4f} > 0.95"))

    # T6: modal exclusivity. Adjacent levels must not both be affirmed.
    ok = res["modal_adjacency"] < 0.05
    T.append(("affirmation of the lower is negation of the higher", ok,
              f"mean adjacent co-occupancy {res['modal_adjacency']:.4f} < 0.05"))

    # T7: the modal gate must actually be USED — a gate that collapses to one
    #     level has satisfied T6 trivially and explains nothing.
    ok = res["levels_used"] >= 2
    T.append(("the level index is used, not collapsed", ok,
              f"{res['levels_used']} of {cfg['n_levels']} levels occupied"))

    # T8: no primordial cause outranks another.
    ok = res["proto_norm_cv"] < 0.15
    T.append(("no priority among the primordial causes", ok,
              f"prototype norm CV {res['proto_norm_cv']:.4f} < 0.15"))

    return T


# ============================================================================
# SECTION 11 — MAIN
# ============================================================================

def main():
    print("=" * 78)
    print("PERIPHYSEON ENGINE — a Self-Ignorant Source Network")
    print("Chapter 0223 · Johannes Scottus Eriugena (c. 800/815 - c. 877)")
    print("=" * 78)

    rng = np.random.default_rng(CFG["seed"])
    world = FallenWorld(CFG, rng)
    P = init_params(CFG, rng)
    perms = make_permutations(CFG, rng)

    n_params = sum(v.size for v in P.values())
    print(f"\n[0] Configuration")
    print(f"    essence dim {CFG['d_essence']} | observation dim {CFG['d_obs']} "
          f"| circumstantiae {CFG['d_circ']}")
    print(f"    primordial causes {CFG['n_causes']} | modes of being {CFG['n_levels']} "
          f"| divisio rungs {CFG['n_coupling']}")
    print(f"    trainable parameters: {n_params}")

    print(f"\n[1] Involution check before training")
    obs0, ess0, _ = world.sample(64)
    z0 = np.tanh(obs0 @ P["Wa1"] + P["ba1"]) @ P["Wa2"] + P["ba2"]
    u0, _ = divisio_fwd(z0, P, CFG, perms)
    b0 = analytica_exact(u0, P, CFG, perms)
    print(f"    max |analytica(divisio(z)) - z| = {np.max(np.abs(b0 - z0)):.3e}")

    print(f"\n[2] Gradient check (mandatory)")
    ok, worst = grad_check(P, CFG, perms, world)
    if not ok:
        raise SystemExit(f"gradient check FAILED (worst {worst:.3e}) — aborting")
    print("    PASSED")

    print(f"\n[3] Training")
    train(P, CFG, perms, world)

    print(f"\n[4] Evaluation on 3000 unseen fallen appearances")
    res, parts = evaluate(P, CFG, perms, world)
    print(f"    theophanic accuracy (cause read from effect) : {res['theophanic_acc']:.4f}")
    print(f"    introspective accuracy (cause from self-model): {res['introspective_acc']:.4f}"
          f"   [chance {res['chance']:.4f}]")
    print(f"    self-report entropy                          : "
          f"{res['introspective_entropy']:.4f} / {res['max_entropy']:.4f} max")
    print(f"    reditus R^2 (essence recovered through fall) : {res['reditus_r2']:.4f}")
    print(f"    modal adjacent co-occupancy                  : {res['modal_adjacency']:.4f}")
    print(f"    modal peak occupancy                         : {res['modal_peak']:.4f}")
    print(f"    levels occupied                              : {res['levels_used']}/{CFG['n_levels']}")
    print(f"    prototype norm CV (priority spread)          : {res['proto_norm_cv']:.4f}")
    print(f"    involution error after training              : {res['involution_err']:.3e}")

    print(f"\n[5] Self-tests")
    T = self_tests(P, CFG, perms, world, res)
    npass = 0
    for name, ok, detail in T:
        tag = "PASS" if ok else "FAIL"
        npass += int(ok)
        print(f"    [{tag}] {name}")
        print(f"           {detail}")

    print("\n" + "=" * 78)
    print(f"RESULT: {npass}/{len(T)} self-tests passed")
    print("=" * 78)
    print("""
READING THE RESULT

The number that matters is the gap in section [4] between theophanic and
introspective accuracy. The two probes are asking the identical question — which
primordial cause is this? — of the identical trained system. One asks it of the
system's outputs and answers almost perfectly. The other asks it of the system's
own model of itself and does no better than guessing, while remaining a live,
normalised, non-degenerate distribution.

That is the architecture's whole content, and it is a design position rather
than a failure: a system can be exhaustively legible to an outside reader of its
effects while being structurally unable to state what it is. The self-model is
not absent. It is not corrupted. It is maintained, and it is uninformative, and
the system works anyway.

Whether that is a property to engineer toward or to engineer away from is the
open question this chapter puts to anyone building minds.
""")
    return 0 if npass == len(T) else 1


if __name__ == "__main__":
    raise SystemExit(main())
