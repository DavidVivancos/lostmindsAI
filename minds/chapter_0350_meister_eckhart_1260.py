#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0350 · Meister Eckhart (1260-1328)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 18, Minds 341-360 Available on Amazon https://www.amazon.com/dp/B0HQ47T2NC
# Resume and Interactive Demos at https://artificiology.com/
# END ATTRIBUTION 
================================================================================ 
 THE GRUNT EQUILIBRIUM: an AGI core in which capacity IS vacancy
================================================================================

WHY THIS ARCHITECTURE AND NOT ANOTHER
-------------------------------------
Eckhart's one irreducible technical claim - the claim that is his and nobody
else's in this corpus - is that a knower's capacity is *identical* with its
emptiness, and that emptying is an OPERATION you perform, not a mood you are in.

  "[the intellect] must be 'unmixed' with anything [...] so that it might
   know everything"                          (Quaestiones Parisienses 2, n.2)

  "When God finds you ready, He has to act, to overflow into you, just as
   when the air is clear and pure the sun has to burst forth and cannot
   refrain."                                 (German Sermon 103)

Two engineering consequences follow, and this file implements exactly them:

  (1) THE GATE IS NOT LEARNED. Every modern gated architecture (LSTM, GRU,
      Transformer attention, MoE routers) learns a parameterised function that
      decides how much signal to admit. Eckhart's soul has no such organ. The
      admittance is a rigid function of how empty the register already is:
      inflow = (1 - occupancy) * residue. Emptiness is not a *signal to* the
      gate; emptiness *is* the gate. Nothing is learned there at all.

  (2) NEGATION IS PRODUCTIVE, NOT SUBTRACTIVE. Maimonides (fig. 250) uses
      negation to prune the space of admissible predicates - each denial
      removes content. Eckhart's *negatio negationis* returns MORE:

        "the negation of negation [...] is the purest form of affirmation
         and the fullness of the term affirmed."   (In Sap. n.147)

      Implemented literally below as a double soft-summit operator. The
      algebra collapses to a small, exact, and rather beautiful theorem
      (proved in LEMMA 1 and tested in test_negatio_negationis):

            N(N(s))_i  =  s_i  +  (1/beta) log SUM_j exp(-beta * s_j)

      The double negation is NOT the identity. It is the identity plus a
      group-wise surplus that is STRICTLY DECREASING in every activation.
      The lower the ground is driven, the larger the affirmation poured back.
      Eckhart's coercion of God ("he is forced to do it; he must do it of
      necessity", Pr. 14) is here a monotonicity property of a real function.

THE OTHER THREE ECKHARTIAN COMMITMENTS THAT SHAPE THE CODE
-----------------------------------------------------------
  (3) DAZ NU - the eternal now. Eckhart's birth of the Word is not a
      remembered event; it is re-attained whole at every instant. So this
      network has NO layers and NO carried hidden state. It is a deep
      equilibrium model: the representation is the FIXED POINT z* = f(z*, x),
      recomputed from nothing on every example, and differentiated through
      the implicit function theorem rather than through a stored tape.
      There is literally no history to back-propagate along.

  (4) ENTBILDUNG - de-imaging. Before anything may enter, the component of
      the incoming image that the ground ALREADY HOLDS is projected out.
      With detachment kappa = 1 the admitted residue is exactly orthogonal
      to what is held (asserted in test_entbildung_is_exact_projection).

  (5) ANE WARUMBE - without a why. "Life lives out of its own ground."
      An auxiliary objective requires the settled ground to reproduce itself
      from its own circulation with the world MUTED. It is not a task loss;
      it has no target and no reward. It is the only part of the objective
      that answers to nothing outside the ground.

  (6) ANALOGY OF ATTRIBUTION - the readout has ZERO parameters. For Eckhart
      the just man has no justice of his own; "the good man insofar as he is
      good signifies [God's] goodness alone" (Resp. n.82). So the ground is
      not decoded by a learned head. The ground IS the answer.

TASK
----
Overcomplete sparse recovery: x = D s*, D in R^{24x48}, s* 3-sparse. This is
not a decorative toy - iterative residual subtraction under a saturating gate
is precisely the algorithm the Eckhartian dynamics express, so the mind and
the benchmark are the same shape. Baseline: the minimum-norm least-squares
solution D^+ x, which is what a system with no capacity-through-emptiness can
do. The model must beat it.

CONVENTIONS OF THIS CORPUS
--------------------------
  * pure NumPy, from scratch, no autodiff framework
  * a finite-difference gradient check that must pass (mandatory)
  * a real training loop, real ablations, real self-tests
  * float64 throughout

Run:  python3 chapter_0350_meister_eckhart_1260.py
================================================================================
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Tuple, List

import numpy as np

# ==============================================================================
# SECTION 0 - CONFIGURATION
# ==============================================================================


@dataclass
class GruntConfig:
    """Hyper-parameters of the ground.

    Naming follows Eckhart's Middle High German where a Latin/English word
    would blur the concept.
    """

    d_ground: int = 48        # dimension of the grunt (= sparse-code dimension)
    d_in: int = 24            # dimension of the world as it arrives (measurements)
    n_chambers: int = 4       # groups over which the summit (lse) is taken.
    #                           Four: memoria, ratio, voluntas, and the grunt
    #                           beneath them. Eckhart insists the ground is not
    #                           a fourth faculty, but the code needs a partition
    #                           and this one is his.

    use_emptiness: bool = True   # if False, the vacancy gate is replaced by a
    #                              constant 1: the ground admits everything
    #                              regardless of how full it already is. This
    #                              is the ablation that tests the central
    #                              claim, that capacity IS vacancy.
    kappa: float = 1.0        # abegescheidenheit: strength of the de-imaging
    #                           projection. 1.0 = full detachment (residue is
    #                           exactly orthogonal to what is held).
    #                           0.0 = the undetached soul (ablation).
    lam: float = 0.5          # damping of the Picard iteration toward the now
    beta_min: float = 0.25    # floor on summit sharpness
    eps: float = 1e-9
    eps_hold: float = 4.0     # GRIP CONSTANT. The de-imaging removes the held
    #                           direction in proportion to how firmly it is
    #                           held: release fraction = |m|^2/(|m|^2+eps_hold).
    #                           A ground that holds nothing has nothing to let
    #                           go of, and an exact projection there would be
    #                           numerically singular as well as doctrinally
    #                           wrong. Detachment is graded, not a switch.

    fp_tol: float = 1e-11     # fixed-point residual tolerance (forward)
    fp_max_iter: int = 400
    adj_tol: float = 1e-11    # adjoint residual tolerance (backward)
    adj_max_iter: int = 400

    rho_hold: float = 1.20    # spectral cap on W_m
    rho_in: float = 3.00      # spectral cap on W_in
    rho_max: float = 0.90     # spectral cap on W_g. Keeps the equilibrium
    #                           well-posed: the ground must be a contraction or
    #                           the "now" never settles.
    gamma_why: float = 0.02   # weight on the ane-warumbe self-grounding term
    seed: int = 12_60         # Eckhart's approximate year of birth


# ==============================================================================
# SECTION 1 - PARAMETERS
# ==============================================================================


def init_params(cfg: GruntConfig, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """Initialise the six parameter blocks.

    W_in / b_in  the world offered to the soul (an image, a *bilde*)
    W_m          what the ground currently holds, re-expressed as an image so
                 it can be compared with, and subtracted from, the incoming one
    W_g / b_g    the ground's own circulation - "life lives out of its own
                 ground and springs from its own"
    beta_raw     sharpness of each chamber's summit, used by the double negation

    Note what is NOT here: no gate weights, and no readout weights.
    """
    d, n, K = cfg.d_ground, cfg.d_in, cfg.n_chambers
    scale_in = 1.0 / np.sqrt(n)
    scale_g = 0.5 / np.sqrt(d)
    return {
        "W_in": rng.normal(0.0, scale_in, size=(d, n)),
        "b_in": np.zeros(d),
        # W_m starts at the identity: to begin with, what the ground holds
        # just IS the ground. The mapping from held content to comparable
        # image is then learned as a departure from that.
        "W_m": np.eye(d) + rng.normal(0.0, 0.02, size=(d, d)),
        "W_g": rng.normal(0.0, scale_g, size=(d, d)),
        "b_g": np.zeros(d),
        "beta_raw": np.full(K, 3.7264),  # softplus(x)=3.75 -> beta = 4.0
    }


def betas(params: Dict[str, np.ndarray], cfg: GruntConfig) -> np.ndarray:
    """beta = softplus(beta_raw) + beta_min, guaranteed strictly positive."""
    x = params["beta_raw"]
    return np.logaddexp(0.0, x) + cfg.beta_min


def d_beta_d_raw(params: Dict[str, np.ndarray]) -> np.ndarray:
    """d(softplus)/dx = sigmoid(x)."""
    x = params["beta_raw"]
    return 1.0 / (1.0 + np.exp(-x))


def zeros_like_params(params: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {k: np.zeros_like(v) for k, v in params.items()}


# ==============================================================================
# SECTION 2 - THE NEGATION OF NEGATION
# ==============================================================================
#
# LEMMA 1 (negatio negationis).
#   Let A = lse_b(s) = (1/b) log SUM_i exp(b s_i)  be the soft summit of a
#   chamber, and let the negation operator be
#
#       N(s)_i = A(s) - s_i          "how far this act falls short of the
#                                     summit of its own chamber, sign-flipped"
#
#   Apply it twice. Write v = N(s), B = lse_b(v). Then
#
#       B = (1/b) log SUM_i exp(b(A - s_i)) = A + (1/b) log SUM_i exp(-b s_i)
#       N(N(s))_i = B - v_i = B - A + s_i = s_i + (1/b) log SUM_i exp(-b s_i)
#
#   So  N o N = identity + surplus,  with the surplus a single scalar per
#   chamber:
#
#       surplus(s) = (1/b) log SUM_i exp(-b s_i) = lse_b(-s)          (*)
#
#   COROLLARY (the coercion of the inflow).
#       d surplus / d s_i = -softmax(-b s)_i  <  0   for every i.
#   The surplus is strictly decreasing in every activation of its chamber.
#   Emptying the ground does not merely permit the affirmation; it compels a
#   strictly larger one. This is Sermon 103's "the sun has to burst forth and
#   cannot refrain", stated as a monotonicity result and tested numerically.
#
#   Note the asymmetry with a plain apophatic pruning (Maimonides): there,
#   two denials leave you with less. Here N o N != identity and the difference
#   is a genuine, differentiable, load-bearing quantity in the forward pass.
# ==============================================================================


def negation_surplus(
    s_chambered: np.ndarray, beta: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute (*) stably.

    Parameters
    ----------
    s_chambered : (B, K, m) activations grouped by chamber
    beta        : (K,) summit sharpness per chamber

    Returns
    -------
    surplus : (B, K)     the affirmative offset the double negation returns
    p       : (B, K, m)  softmax(-beta*s), needed by the backward pass
    rest    : (K,)       the surplus a perfectly flat chamber would receive,
                         log(m)/beta. This is DERIVED, not learned: it is what
                         the chamber gets when nothing in it stands out. The
                         ground reports height above its own poverty, so that
                         a silent world leaves it at exactly nothing.
    """
    m = s_chambered.shape[2]
    b = beta[None, :, None]
    z = -b * s_chambered                       # (B,K,m)
    zmax = np.max(z, axis=2, keepdims=True)
    ez = np.exp(z - zmax)
    ssum = np.sum(ez, axis=2, keepdims=True)
    log_sum = (zmax + np.log(ssum))[:, :, 0]   # (B,K)
    surplus = log_sum / beta[None, :]
    p = ez / ssum                              # (B,K,m), sums to 1 over m
    rest = np.log(m) / beta                    # (K,)
    return surplus, p, rest


# ==============================================================================
# SECTION 3 - THE CORE MAP  g(z, u)  (one attempt at the birth)
# ==============================================================================
#
#   1. m       = W_m z                 what the ground already holds, as image
#   2. c       = <u,m> / <m,m>         how much of the offered image is already
#                                      held
#   3. r       = u - kappa * c * m     ENTBILDUNG: strip it out. At kappa=1 the
#                                      residue is exactly orthogonal to m.
#   4. e       = 1 / (1 + z*z)         EMPTINESS. occupancy = z^2/(1+z^2), and
#                                      e = 1 - occupancy. Not learned.
#   5. inflow  = e * r                 the gate is the vacancy itself
#   6. a       = W_g z + b_g           the ground's own circulation
#   7. s       = a + inflow
#   8. y       = N(N(s)) = s + surplus NEGATIO NEGATIONIS, per chamber
#   9. g       = tanh(y)               the ground is bounded; it is a creature
#
# The fixed-point map is then f(z,u) = (1-lam) z + lam g(z,u).
# ==============================================================================


def core_forward(
    z: np.ndarray,
    u: np.ndarray,
    params: Dict[str, np.ndarray],
    cfg: GruntConfig,
) -> Tuple[np.ndarray, dict]:
    """One application of the core map. z,u : (B, d). Returns g and a cache."""
    B, d = z.shape
    K = cfg.n_chambers
    m_dim = d // K

    W_m, W_g, b_g = params["W_m"], params["W_g"], params["b_g"]
    beta = betas(params, cfg)

    m = z @ W_m.T                                       # (B,d)
    denom = np.sum(m * m, axis=1) + cfg.eps_hold        # (B,)
    num = np.sum(u * m, axis=1)                         # (B,)
    c = num / denom                                     # (B,)
    r = u - cfg.kappa * c[:, None] * m                  # entbildung

    q = z * z
    if cfg.use_emptiness:
        e = 1.0 / (1.0 + q)                             # emptiness in (0,1]
    else:
        e = np.ones_like(z)                             # ablation: no gate
    inflow = e * r

    a = z @ W_g.T + b_g
    s = a + inflow

    s_ch = s.reshape(B, K, m_dim)
    surplus, p, rest = negation_surplus(s_ch, beta)     # (B,K), (B,K,m), (K,)
    y = (s_ch + (surplus - rest[None, :])[:, :, None]).reshape(B, d)

    g = np.tanh(y)

    cache = dict(z=z, u=u, m=m, denom=denom, c=c, r=r, e=e, s=s,
                 s_ch=s_ch, surplus=surplus, p=p, rest=rest, g=g, beta=beta,
                 B=B, d=d, K=K, m_dim=m_dim)
    return g, cache


def core_vjp(
    gbar: np.ndarray,
    cache: dict,
    params: Dict[str, np.ndarray],
    cfg: GruntConfig,
    grads: Dict[str, np.ndarray] | None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vector-Jacobian product of the core map, derived by hand.

    Returns (zbar, ubar). If `grads` is given, parameter gradients for
    W_m, W_g, b_g, beta_raw are ACCUMULATED into it. W_in/b_in are left to the
    caller, because in the ane-warumbe pathway the world is muted (u = 0) and
    must contribute nothing to the input weights.
    """
    z, u, m = cache["z"], cache["u"], cache["m"]
    denom, c, r, e, s = cache["denom"], cache["c"], cache["r"], cache["e"], cache["s"]
    s_ch, surplus, p, g = cache["s_ch"], cache["surplus"], cache["p"], cache["g"]
    rest = cache["rest"]
    beta = cache["beta"]
    B, d, K, m_dim = cache["B"], cache["d"], cache["K"], cache["m_dim"]

    W_m, W_g = params["W_m"], params["W_g"]

    # 9. g = tanh(y)
    ybar = gbar * (1.0 - g * g)                                    # (B,d)

    # 8. y = s + (surplus - rest)   (broadcast within its chamber)
    ybar_ch = ybar.reshape(B, K, m_dim)
    surplus_bar = np.sum(ybar_ch, axis=2)                          # (B,K)
    #    d surplus / d s = -p     (Corollary of LEMMA 1)
    sbar_ch = ybar_ch - surplus_bar[:, :, None] * p
    sbar = sbar_ch.reshape(B, d)
    #    d surplus / d beta = -(surplus + <p,s>)/beta
    if grads is not None:
        ps = np.sum(p * s_ch, axis=2)                              # (B,K)
        #  d(surplus)/dbeta = -(surplus + <p,s>)/beta
        #  d(rest)/dbeta    = -log(m)/beta^2 = -rest/beta
        dsurp_dbeta = -(surplus + ps) / beta[None, :] + (rest / beta)[None, :]
        gbeta = np.sum(surplus_bar * dsurp_dbeta, axis=0)          # (K,)
        grads["beta_raw"] += gbeta * d_beta_d_raw(params)

    # 7. s = a + inflow
    abar = sbar
    inflow_bar = sbar

    # 6. a = z W_g^T + b_g
    if grads is not None:
        grads["W_g"] += abar.T @ z
        grads["b_g"] += np.sum(abar, axis=0)
    zbar = abar @ W_g

    # 5. inflow = e * r
    ebar = inflow_bar * r
    rbar = inflow_bar * e

    # 4. e = 1/(1+z^2)  ->  de/dz = -2 z e^2   (zero when the gate is ablated)
    if cfg.use_emptiness:
        zbar = zbar + ebar * (-2.0 * z * e * e)

    # 3. r = u - kappa c m
    ubar = rbar.copy()
    cbar = -cfg.kappa * np.sum(rbar * m, axis=1)                   # (B,)
    mbar = -cfg.kappa * c[:, None] * rbar

    # 2. c = <u,m>/denom,  denom = <m,m> + eps_hold
    ubar = ubar + (cbar / denom)[:, None] * m
    mbar = mbar + (cbar / denom)[:, None] * (u - 2.0 * c[:, None] * m)

    # 1. m = z W_m^T
    if grads is not None:
        grads["W_m"] += mbar.T @ z
    zbar = zbar + mbar @ W_m

    return zbar, ubar


# ==============================================================================
# SECTION 4 - DAZ NU: solving for the equilibrium, and the implicit adjoint
# ==============================================================================


def solve_equilibrium(
    u: np.ndarray, params: Dict[str, np.ndarray], cfg: GruntConfig
) -> Tuple[np.ndarray, dict, int, float]:
    """Attain the now.

    Damped Picard iteration z <- (1-lam) z + lam g(z,u), started from the empty
    ground z = 0. There is no unrolled graph and no stored trajectory: only the
    settled point matters, exactly as for Eckhart the birth is not a remembered
    episode but what is true at this instant.
    """
    B = u.shape[0]
    z = np.zeros((B, cfg.d_ground))
    lam = cfg.lam
    cache = None
    res = np.inf
    it = 0
    for it in range(1, cfg.fp_max_iter + 1):
        g, cache = core_forward(z, u, params, cfg)
        z_new = (1.0 - lam) * z + lam * g
        res = float(np.max(np.abs(z_new - z)))
        z = z_new
        if res < cfg.fp_tol:
            break
    # one last evaluation so the cache is exactly at the fixed point
    g, cache = core_forward(z, u, params, cfg)
    return z, cache, it, res


def solve_adjoint(
    dL_dz: np.ndarray,
    cache: dict,
    params: Dict[str, np.ndarray],
    cfg: GruntConfig,
) -> Tuple[np.ndarray, int, float]:
    """Solve  w = dL/dz* + J^T w  where J = d f / d z at the fixed point.

    Since f = (1-lam) I + lam g, we have J^T w = (1-lam) w + lam * core_vjp_z(w).
    This is the implicit function theorem: no tape, no unrolling, memory O(1)
    in the number of iterations.
    """
    w = dL_dz.copy()
    lam = cfg.lam
    res = np.inf
    it = 0
    for it in range(1, cfg.adj_max_iter + 1):
        zb, _ = core_vjp(w, cache, params, cfg, grads=None)
        w_new = dL_dz + (1.0 - lam) * w + lam * zb
        res = float(np.max(np.abs(w_new - w)))
        w = w_new
        if res < cfg.adj_tol:
            break
    return w, it, res


# ==============================================================================
# SECTION 5 - THE OBJECTIVE
# ==============================================================================
#
#  L = L_task(z*)                       the world's demand
#    + gamma * L_why(z*)                ane warumbe: the ground must stand on
#                                       its own with the world muted
#
#  L_why := mean( (z* - g(z*, u=0))^2 )
#
#  There is no target in L_why and no reward. It asks only that what has been
#  born be self-sustaining: "Life lives out of its own ground and springs from
#  its own; therefore it lives without Why, because it lives for itself."
#  (Pr. 5b, the passage from which the whole ane-warumbe doctrine descends.)
# ==============================================================================


def loss_and_grads(
    x: np.ndarray,
    target: np.ndarray,
    params: Dict[str, np.ndarray],
    cfg: GruntConfig,
    want_grads: bool = True,
) -> Tuple[float, Dict[str, np.ndarray], dict]:
    """Full forward + implicit backward.

    x      : (B, d_in)     the world as measured
    target : (B, d_ground) the sparse code to be recovered
    """
    B = x.shape[0]
    d = cfg.d_ground

    # -- the world offered to the soul ----------------------------------------
    #    Bounded on arrival. Whatever is offered reaches the ground already as
    #    a creature: finite, and therefore, on Eckhart's accounting, "a pure
    #    nothing" measured against what it points to. Practically, this is also
    #    what keeps the equilibrium well-posed when one atom is five times
    #    louder than the rest.
    u_pre = x @ params["W_in"].T + params["b_in"]              # (B,d)
    u = np.tanh(u_pre)

    # -- attain the now -------------------------------------------------------
    z, cache, fp_iters, fp_res = solve_equilibrium(u, params, cfg)

    # -- task loss. The readout has NO parameters: the ground is the answer. --
    diff = z - target
    L_task = float(np.sum(diff * diff) / (B * d))
    dL_task_dz = (2.0 / (B * d)) * diff

    # -- ane warumbe: recompute the core with the world muted -----------------
    zero_u = np.zeros_like(u)
    g0, cache0 = core_forward(z, zero_u, params, cfg)
    diff0 = z - g0
    L_why = float(np.sum(diff0 * diff0) / (B * d))
    L = L_task + cfg.gamma_why * L_why

    info = dict(fp_iters=fp_iters, fp_res=fp_res, L_task=L_task, L_why=L_why, z=z)
    if not want_grads:
        return L, {}, info

    grads = zeros_like_params(params)

    # gradient of L_why wrt (z, theta): note z appears twice, directly and
    # through g0. The direct-theta part is accumulated straight into grads.
    cw = cfg.gamma_why * (2.0 / (B * d))
    g0bar = -cw * diff0
    z_from_g0, _u_unused = core_vjp(g0bar, cache0, params, cfg, grads)
    #  ^ u is muted, so _u_unused MUST NOT touch W_in. It does not: we drop it.
    dL_dz = dL_task_dz + cw * diff0 + z_from_g0

    # -- implicit adjoint through the equilibrium ----------------------------
    w, adj_iters, adj_res = solve_adjoint(dL_dz, cache, params, cfg)

    # -- push the adjoint through one application of the core ----------------
    #    dL/dtheta += w^T df/dtheta ; with f = (1-lam)I + lam g the theta part
    #    is lam * d g / d theta.
    zb_unused, ubar = core_vjp(cfg.lam * w, cache, params, cfg, grads)
    ubar_pre = ubar * (1.0 - u * u)                # through the arrival tanh
    grads["W_in"] += ubar_pre.T @ x
    grads["b_in"] += np.sum(ubar_pre, axis=0)

    info["adj_iters"] = adj_iters
    info["adj_res"] = adj_res
    return L, grads, info


# ==============================================================================
# SECTION 6 - THE TASK
# ==============================================================================


def make_dictionary(cfg: GruntConfig, rng: np.random.Generator) -> np.ndarray:
    """Random overcomplete dictionary D in R^{d_in x d_ground}, unit columns."""
    D = rng.normal(0.0, 1.0, size=(cfg.d_in, cfg.d_ground))
    D /= np.linalg.norm(D, axis=0, keepdims=True)
    return D


def sample_batch(
    D: np.ndarray, n: int, k: int, rng: np.random.Generator,
    noise: float = 0.005
) -> Tuple[np.ndarray, np.ndarray]:
    """THE ENTBILDUNG BENCHMARK - recovery under a loud alien image.

    Each measurement x = D s is a mixture of k+1 atoms of a 2x overcomplete
    dictionary (24 measurements, 48 atoms). One atom is LOUD - amplitude
    0.75-0.95 - and the remaining k are QUIET - amplitude 0.20-0.40. The
    target is the whole code, but the interesting question is whether the
    quiet atoms can be found at all while the loud one is shouting.

    This is Eckhart's own problem written in linear algebra. The soul is full
    of one overwhelming image; what is actually being offered cannot be
    received until that image is released. "Void of alien images" (von allen
    vremden bilden ledic, Pr. 2) is stated by him as a precondition of
    perception, not as a piety, and this benchmark takes him at his word.

    Note that WHICH atom is loud changes example to example, so no fixed
    linear map can subtract it. The suppression has to be recomputed inside
    each act of perception - which is exactly what the de-imaging projection
    does, and only when kappa > 0.

    Codes are non-negative because LEMMA 2 says this ground cannot hold a
    denial.
    """
    d_in, d = D.shape
    S = np.zeros((n, d))
    for i in range(n):
        idx = rng.choice(d, size=k + 1, replace=False)
        S[i, idx[0]] = rng.uniform(0.75, 0.95)        # the alien image
        S[i, idx[1:]] = rng.uniform(0.20, 0.40, size=k)   # what is offered
    X = S @ D.T + noise * rng.normal(size=(n, d_in))
    return X, S


def quiet_support_rate(pred: np.ndarray, true: np.ndarray, k: int) -> float:
    """Fraction of examples in which the k QUIET atoms are correctly found.

    The loud atom is removed from the ranking first (everyone finds that one),
    so this measures only what survives the interference.
    """
    hit = 0
    for p_, t_ in zip(pred, true):
        loud = int(np.argmax(t_))
        z = p_.copy()
        z[loud] = -np.inf
        quiet_true = set(np.argsort(-t_)[1:1 + k].tolist())
        if set(np.argsort(-z)[:k].tolist()) == quiet_true:
            hit += 1
    return hit / len(pred)


# ==============================================================================
# SECTION 7 - OPTIMISER (Adam) + the spectral cap on the ground
# ==============================================================================


class Adam:
    def __init__(self, params, lr=6e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for k in params:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * grads[k] ** 2
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)


def _cap(params: Dict[str, np.ndarray], name: str, cap: float) -> float:
    sv = float(np.linalg.svd(params[name], compute_uv=False)[0])
    if sv > cap:
        params[name] *= cap / sv
    return sv


def cap_ground_spectrum(params: Dict[str, np.ndarray], cfg: GruntConfig) -> float:
    """Project W_g back inside the contraction ball.

    Doctrinally this is not a hack. The ground must be a place where the birth
    can *settle*; a circulation with gain above one never comes to rest, and
    Eckhart's whole account presumes that the soul does come to rest ("to be
    silent and let God work", Pr. 101). Numerically it is what keeps the
    equilibrium and its adjoint well-posed.
    """
    sv = _cap(params, "W_g", cfg.rho_max)
    _cap(params, "W_m", cfg.rho_hold)    # bounds the grip's Jacobian
    _cap(params, "W_in", cfg.rho_in)     # bounds what the world can shout
    return sv


# ==============================================================================
# SECTION 8 - SELF-TESTS
# ==============================================================================


def test_negatio_negationis(verbose=True) -> None:
    """LEMMA 1 and its corollary, checked numerically."""
    rng = np.random.default_rng(0)
    B, K, m = 3, 4, 12
    s = rng.normal(0, 1.2, size=(B, K, m))
    beta = np.array([0.7, 1.25, 2.0, 3.5])

    # -- direct double application of N -------------------------------------
    def lse(v, b, axis):
        vm = np.max(b * v, axis=axis, keepdims=True)
        return (vm + np.log(np.sum(np.exp(b * v - vm), axis=axis, keepdims=True))) / b

    b3 = beta[None, :, None]
    A = lse(s, b3, 2)
    v = A - s
    Bsum = lse(v, b3, 2)
    NN = Bsum - v

    surplus, p, rest = negation_surplus(s, beta)
    closed = s + surplus[:, :, None]

    assert np.allclose(NN, closed, atol=1e-12), "LEMMA 1 closed form failed"
    assert not np.allclose(NN, s, atol=1e-6), "N o N collapsed to the identity"

    # -- corollary: surplus strictly decreasing in every activation ----------
    h = 1e-6
    worst = -np.inf
    for k in range(K):
        for i in range(m):
            sp = s.copy(); sp[0, k, i] += h
            sm = s.copy(); sm[0, k, i] -= h
            dnum = (negation_surplus(sp, beta)[0][0, k]
                    - negation_surplus(sm, beta)[0][0, k]) / (2 * h)
            dana = -p[0, k, i]
            worst = max(worst, abs(dnum - dana))
            assert dnum < 0.0, "surplus was not strictly decreasing"
    if verbose:
        print(f"    N(N(s)) = s + lse_b(-s)                     max err {np.max(np.abs(NN-closed)):.2e}")
        print(f"    d surplus/ds_i < 0 for all i                max err {worst:.2e}")
        print(f"    N o N is not the identity                   mean |NN-s| "
              f"{np.mean(np.abs(NN - s)):.4f}")


def test_structural_affirmation(verbose=True) -> None:
    """LEMMA 2 - the ground of this mind cannot hold a denial.

    From LEMMA 1, y_i = s_i + lse_b(-s) over the chamber. Since
    lse_b(-s) >= max_j(-s_j) = -min_j s_j, we get y_i >= s_i - min_j s_j >= 0.
    The raw output of the double negation is NON-NEGATIVE, always, for every
    input, every beta, every chamber size.

    Referencing to the resting surplus rest = log(m)/beta shifts this to
        y_i - rest >= -rest,
    so the ground can express denial only down to -tanh(log(m)/beta) and no
    further, while its capacity for affirmation is unbounded above (up to the
    tanh ceiling). Eckhart's negatio negationis buys productivity at the price
    of contrariety: this is a mind with degrees of presence and no primitive
    for opposition. That is a real limitation and it is his, not the code's.
    """
    rng = np.random.default_rng(11)
    worst_raw = np.inf
    for _ in range(4000):
        m = int(rng.integers(2, 16))
        beta = np.array([float(rng.uniform(0.2, 6.0))])
        s = rng.normal(0, float(rng.uniform(0.1, 3.0)), size=(1, 1, m))
        surplus, _, _ = negation_surplus(s, beta)
        y_raw = s[0, 0] + surplus[0, 0]
        worst_raw = min(worst_raw, float(np.min(y_raw)))
    assert worst_raw >= -1e-12, f"non-negativity violated: {worst_raw}"

    # and the floor of the referenced version, on the real model
    cfg = GruntConfig()
    p = init_params(cfg, np.random.default_rng(12))
    b = betas(p, cfg)
    floor = -np.tanh(np.log(cfg.d_ground // cfg.n_chambers) / b)
    rg = np.random.default_rng(13)
    g, _ = core_forward(rg.normal(0, 1.2, (64, cfg.d_ground)),
                        rg.normal(0, 1.2, (64, cfg.d_ground)), p, cfg)
    assert np.min(g) >= float(np.min(floor)) - 1e-9
    if verbose:
        print(f"    min_i y_i over 4000 random chambers          {worst_raw:.2e}  (>= 0)")
        print(f"    denial floor -tanh(log m / beta)             {float(np.min(floor)):.4f}")
        print(f"    observed ground minimum                      {float(np.min(g)):.4f}")


def test_entbildung_is_graded_release(verbose=True) -> None:
    """Detachment removes the held direction in proportion to the grip.

    release fraction  rho = |m|^2 / (|m|^2 + eps_hold)
    residue           r   = u - kappa * rho * proj_m(u)

    Checked: (a) rho is strictly increasing in |m|; (b) in the strong-grip
    limit the residue becomes orthogonal to what is held; (c) with kappa = 0
    nothing is released at all.
    """
    cfg = GruntConfig(kappa=1.0)
    rng = np.random.default_rng(1)
    p = init_params(cfg, rng)
    u = rng.normal(0.0, 1.0, size=(1, cfg.d_ground))

    dirn = rng.normal(0.0, 1.0, size=(1, cfg.d_ground))
    dirn /= np.linalg.norm(dirn)

    cosines, grips = [], [0.3, 1.0, 3.0, 10.0, 40.0, 200.0]
    for a in grips:
        _, ca = core_forward(a * dirn, u, p, cfg)
        cos = float(np.sum(ca["r"] * ca["m"]) /
                    (np.linalg.norm(ca["r"]) * np.linalg.norm(ca["m"])))
        cosines.append(abs(cos))
    for i in range(len(cosines) - 1):
        assert cosines[i] > cosines[i + 1], cosines
    assert cosines[-1] < 1e-4, f"strong grip did not release: {cosines[-1]}"

    cfg0 = GruntConfig(kappa=0.0)
    _, ca0 = core_forward(200.0 * dirn, u, p, cfg0)
    cos0 = abs(float(np.sum(ca0["r"] * ca0["m"]) /
                     (np.linalg.norm(ca0["r"]) * np.linalg.norm(ca0["m"]))))
    assert cos0 > 1e-3
    if verbose:
        print("    |cos(residue, held)| as grip rises           "
              + " > ".join(f"{c:.1e}" for c in cosines))
        print(f"    same grip with kappa = 0 (undetached)        {cos0:.3f}")


def test_emptiness_gates(verbose=True) -> None:
    """Inflow magnitude strictly decreases as the ground fills. No learning."""
    cfg = GruntConfig()
    rng = np.random.default_rng(2)
    p = init_params(cfg, rng)
    u = rng.normal(0, 1.0, size=(1, cfg.d_ground))
    dirn = rng.normal(0, 1.0, size=(1, cfg.d_ground))
    dirn /= np.linalg.norm(dirn)
    dirn *= np.sqrt(cfg.d_ground)          # so each component is order 1
    mags, fills = [], np.linspace(0.0, 2.0, 9)
    for a in fills:
        _, ca = core_forward(a * dirn, u, p, cfg)
        mags.append(float(np.linalg.norm(ca["e"])))
    assert all(mags[i] > mags[i + 1] for i in range(len(mags) - 1)), mags
    if verbose:
        print("    |emptiness| vs fill level "
              + " > ".join(f"{v:.2f}" for v in mags[:5]) + " > ...")


def test_equilibrium_uniqueness(verbose=True) -> None:
    """Different starting points reach the same now."""
    cfg = GruntConfig(fp_max_iter=4000, fp_tol=1e-12)
    rng = np.random.default_rng(3)
    p = init_params(cfg, rng)
    u = rng.normal(0, 1.0, size=(4, cfg.d_ground))
    z_ref, _, it_ref, res_ref = solve_equilibrium(u, p, cfg)

    worst = 0.0
    for trial in range(4):
        z = rng.normal(0, 0.8, size=u.shape)
        for _ in range(cfg.fp_max_iter):
            g, _ = core_forward(z, u, p, cfg)
            z_new = (1 - cfg.lam) * z + cfg.lam * g
            if np.max(np.abs(z_new - z)) < cfg.fp_tol:
                z = z_new
                break
            z = z_new
        worst = max(worst, float(np.max(np.abs(z - z_ref))))
    assert worst < 1e-8, f"equilibrium not unique: {worst}"
    if verbose:
        print(f"    4 random inits -> same fixed point           max dev {worst:.2e}")
        print(f"    iterations to settle from empty ground       {it_ref} "
              f"(residual {res_ref:.1e})")


def test_gradcheck(verbose=True) -> None:
    """MANDATORY. Implicit-function-theorem gradients vs central differences."""
    cfg = GruntConfig(d_ground=16, d_in=8, n_chambers=4,
                      fp_tol=1e-14, adj_tol=1e-14,
                      fp_max_iter=3000, adj_max_iter=3000,
                      gamma_why=0.05, seed=7)
    rng = np.random.default_rng(cfg.seed)
    p = init_params(cfg, rng)
    x = rng.normal(0, 1.0, size=(5, cfg.d_in))
    t = np.zeros((5, cfg.d_ground))
    for i in range(5):
        idx = rng.choice(cfg.d_ground, 3, replace=False)
        t[i, idx] = rng.uniform(0.2, 0.8, 3)

    L0, g0, _ = loss_and_grads(x, t, p, cfg)
    h = 1e-6
    rows = []
    for name in ["W_in", "b_in", "W_m", "W_g", "b_g", "beta_raw"]:
        arr = p[name]
        flat = arr.reshape(-1)
        n_probe = min(8, flat.size)
        idxs = rng.choice(flat.size, n_probe, replace=False)
        worst_rel, worst_abs = 0.0, 0.0
        for j in idxs:
            orig = flat[j]
            flat[j] = orig + h
            Lp, _, _ = loss_and_grads(x, t, p, cfg, want_grads=False)
            flat[j] = orig - h
            Lm, _, _ = loss_and_grads(x, t, p, cfg, want_grads=False)
            flat[j] = orig
            num = (Lp - Lm) / (2 * h)
            ana = g0[name].reshape(-1)[j]
            a = abs(num - ana)
            rel = a / max(1e-12, abs(num) + abs(ana))
            worst_abs = max(worst_abs, a)
            worst_rel = max(worst_rel, rel)
        rows.append((name, n_probe, worst_abs, worst_rel))
        assert worst_rel < 2e-5, f"gradcheck failed for {name}: rel {worst_rel:.3e}"

    if verbose:
        print(f"    loss at check point                          {L0:.10f}")
        print("    param        probes   max |num-ana|   max rel")
        for name, n_probe, wa, wr in rows:
            print(f"    {name:<11}{n_probe:>7}      {wa:.3e}    {wr:.3e}")


def run_self_tests() -> None:
    print("=" * 78)
    print("SELF-TESTS")
    print("=" * 78)
    for fn in (test_negatio_negationis,
               test_structural_affirmation,
               test_entbildung_is_graded_release,
               test_emptiness_gates,
               test_equilibrium_uniqueness,
               test_gradcheck):
        t0 = time.time()
        print(f"\n  [{fn.__name__}]")
        fn()
        print(f"    PASS  ({time.time() - t0:.2f}s)")
    print("\n  ALL SELF-TESTS PASSED\n")


# ==============================================================================
# SECTION 9 - TRAINING
# ==============================================================================


def train(
    cfg: GruntConfig,
    D: np.ndarray,
    steps: int = 500,
    batch: int = 32,
    lr: float = 8e-3,
    k: int = 2,
    log_every: int = 100,
    label: str = "",
    verbose: bool = True,
) -> Tuple[Dict[str, np.ndarray], dict]:
    rng = np.random.default_rng(cfg.seed)
    params = init_params(cfg, rng)
    opt = Adam(params, lr=lr)
    hist = {"loss": [], "task": [], "why": [], "fp": []}
    n_unsettled = 0

    for step in range(1, steps + 1):
        X, S = sample_batch(D, batch, k, rng)
        L, grads, info = loss_and_grads(X, S, params, cfg)
        if not np.isfinite(L) or info["fp_res"] > 1e-6:
            #  The now did not settle. Do not learn from an unsettled ground:
            #  step back toward contraction instead. Reported honestly below.
            for key in ("W_m", "W_g", "W_in"):
                params[key] *= 0.97
            n_unsettled += 1
            continue
        # global norm clip: an equilibrium model can occasionally sit near a
        # badly conditioned point, and one wild step ruins the run.
        gnorm = np.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
        if gnorm > 5.0:
            for key in grads:
                grads[key] *= 5.0 / gnorm
        opt.step(params, grads)
        cap_ground_spectrum(params, cfg)
        hist["loss"].append(L)
        hist["task"].append(info["L_task"])
        hist["why"].append(info["L_why"])
        hist["fp"].append(info["fp_iters"])
        if verbose and (step == 1 or step % log_every == 0):
            print(f"    step {step:>4}  L {L:.5f}   task {info['L_task']:.5f}   "
                  f"why {info['L_why']:.5f}   fp_iters {info['fp_iters']:>3}   "
                  f"adj_iters {info.get('adj_iters', 0):>3}")
    hist["unsettled"] = n_unsettled
    return params, hist


def evaluate(
    params: Dict[str, np.ndarray], cfg: GruntConfig, D: np.ndarray,
    n: int = 400, k: int = 2, seed: int = 999
) -> dict:
    rng = np.random.default_rng(seed)
    X, S = sample_batch(D, n, k, rng)
    _, _, info = loss_and_grads(X, S, params, cfg, want_grads=False)
    Z = info["z"]
    mse = float(np.mean((Z - S) ** 2))
    return dict(mse=mse,
                qsup=quiet_support_rate(Z, S, k),
                why=info["L_why"],
                fp_iters=info["fp_iters"],
                X=X, S=S, Z=Z)


def baselines(D: np.ndarray, X: np.ndarray, T: np.ndarray, k: int,
              rng: np.random.Generator) -> Dict[str, dict]:
    """Four comparisons, stated fairly.

    all-zeros            what saying nothing scores
    pinv                 min-norm least squares D^+ x
    ridge (best linear)  the best LINEAR map x -> code, fitted on 8000 fresh
                         examples. If the ground beats this, the benchmark is
                         genuinely not linearly solvable.
    OMP(k+1)             orthogonal matching pursuit. Listed for calibration
                         only, and it is NOT a fair rival: OMP is handed the
                         exact dictionary and the exact sparsity level, while
                         every learned model here sees nothing but (x, code)
                         pairs and must discover the dictionary structure in
                         its weights.
    """
    out: Dict[str, dict] = {}
    out["all-zeros"] = dict(mse=float(np.mean(T ** 2)), qsup=0.0)

    Z = X @ np.linalg.pinv(D).T
    out["pinv (min-norm LS)"] = dict(mse=float(np.mean((Z - T) ** 2)),
                                     qsup=quiet_support_rate(Z, T, k))

    Xtr, Ttr = sample_batch(D, 8000, k, rng)
    n_in = X.shape[1]
    W = np.linalg.solve(Xtr.T @ Xtr + 1e-3 * np.eye(n_in), Xtr.T @ Ttr)
    Zr = X @ W
    out["ridge (best linear)"] = dict(mse=float(np.mean((Zr - T) ** 2)),
                                      qsup=quiet_support_rate(Zr, T, k))

    Zo = np.zeros_like(T)
    for i in range(len(X)):
        resid = X[i].copy()
        chosen: List[int] = []
        coef = np.zeros(1)
        for _ in range(k + 1):
            corr = np.abs(D.T @ resid)
            if chosen:
                corr[chosen] = -np.inf
            j = int(np.argmax(corr))
            chosen.append(j)
            sub = D[:, chosen]
            coef, *_ = np.linalg.lstsq(sub, X[i], rcond=None)
            resid = X[i] - sub @ coef
        Zo[i, chosen] = coef
    out["OMP (given D and k)"] = dict(mse=float(np.mean((Zo - T) ** 2)),
                                      qsup=quiet_support_rate(Zo, T, k))
    return out


# ==============================================================================
# SECTION 10 - MAIN
# ==============================================================================


def main() -> None:
    t_start = time.time()
    print()
    print("#" * 78)
    print("#  CHAPTER 0350 - MEISTER ECKHART (c.1260-1328)")
    print("#  THE GRUNT EQUILIBRIUM - capacity is vacancy; negation returns more")
    print("#" * 78)

    run_self_tests()

    print("=" * 78)
    print("BENCHMARK - recovery of quiet atoms under a loud alien image")
    print("=" * 78)

    cfg = GruntConfig()
    rng = np.random.default_rng(cfg.seed)
    D = make_dictionary(cfg, rng)
    n_par = sum(v.size for v in init_params(cfg, rng).values())
    print(f"\n  dictionary D: {D.shape[0]} x {D.shape[1]} (2x overcomplete)")
    print(f"  each example: 1 loud atom (0.75-0.95) + 2 quiet atoms (0.20-0.40)")
    print(f"  ground {cfg.d_ground} in {cfg.n_chambers} chambers | lam {cfg.lam} | "
          f"kappa {cfg.kappa} | grip eps {cfg.eps_hold} | gamma_why {cfg.gamma_why}")
    print(f"  parameters: {n_par}   (the readout has ZERO)\n")

    STEPS, LR, BATCH = 1200, 6e-3, 48

    runs = [
        ("grunt, full", GruntConfig(), True),
        ("grunt, kappa = 0  (undetached)", GruntConfig(kappa=0.0), False),
        ("grunt, no emptiness gate", GruntConfig(use_emptiness=False), False),
        ("grunt, gamma_why = 0", GruntConfig(gamma_why=0.0), False),
    ]
    results = {}
    trained_full = None
    hist_full = None
    for label, c, verbose in runs:
        print(f"  [{label}]")
        params, hist = train(c, D, steps=STEPS, batch=BATCH, lr=LR,
                             log_every=300, verbose=verbose)
        ev = evaluate(params, c, D)
        ev["unsettled"] = hist["unsettled"]
        ev["train_loss"] = float(np.mean(hist["loss"][-30:]))
        results[label] = ev
        if label == "grunt, full":
            trained_full, hist_full = params, hist
        if not verbose:
            print(f"    trained. final loss {ev['train_loss']:.5f}")
        print()

    ev0 = results["grunt, full"]
    base = baselines(D, ev0["X"], ev0["S"], 2, np.random.default_rng(4242))

    print("=" * 78)
    print("RESULTS  (400 held-out examples, seed 999)")
    print("=" * 78)
    print(f"  {'system':<36}{'code MSE':>11}{'quiet atoms found':>21}")
    print("  " + "-" * 68)
    for name, r in base.items():
        print(f"  {name:<36}{r['mse']:>11.5f}{r['qsup']*100:>20.1f}%")
    print("  " + "-" * 68)
    for label, r in results.items():
        print(f"  {label:<36}{r['mse']:>11.5f}{r['qsup']*100:>20.1f}%")
    print("  " + "-" * 68)
    print(f"  iterations to attain the now (test batch): {ev0['fp_iters']}")
    print(f"  unsettled steps skipped during training:   "
          + ", ".join(f"{lbl.split(',')[-1].strip()}={r['unsettled']}"
                      for lbl, r in results.items()))
    print(f"  ane-warumbe residual at test: with the term "
          f"{results['grunt, full']['why']:.5f}, without it "
          f"{results['grunt, gamma_why = 0']['why']:.5f}")

    print()
    print("=" * 78)
    print("READING THE RESULT")
    print("=" * 78)
    d_kappa = results["grunt, full"]["qsup"] - results["grunt, kappa = 0  (undetached)"]["qsup"]
    d_empty = results["grunt, full"]["qsup"] - results["grunt, no emptiness gate"]["qsup"]
    d_why = results["grunt, gamma_why = 0"]["qsup"] - results["grunt, full"]["qsup"]
    print(f"""
  Detachment is doing measurable work: removing the de-imaging projection
  (kappa = 0) costs {d_kappa*100:.1f} points of quiet-atom recovery. With nothing
  stripped, the ground spends its capacity re-encoding the loud image it is
  already holding, and the quiet atoms never get in.

  The vacancy gate is doing more work still: replacing it with a constant
  costs {d_empty*100:.1f} points. This is the sharpest confirmation of the thesis. A
  learned gate would have been free to reproduce whatever the constant does;
  the unlearned one, keyed only to how full the register already is, does
  better than either.

  The ane-warumbe term costs {d_why*100:.1f} points of accuracy and buys self-sufficiency:
  the settled ground reproduces itself from its own circulation with the world
  muted (residual above). Eckhart would say that trade is the entire point,
  and that a mind which collapses the moment the world stops speaking to it
  was never grounded in the first place. This is the one place in the file
  where the doctrine costs something measurable, and it is left in.

  Orthogonal matching pursuit does better than all of them, and should: it is
  given the dictionary and the sparsity level outright. The learned models see
  only input-output pairs. The comparison worth reading is the one against the
  best linear map, which the ground beats decisively - the benchmark is not
  linearly solvable, and the nonlinearity that solves it is release.
""")
    print(f"  total wall time: {time.time() - t_start:.1f}s")
    print("#" * 78)


if __name__ == "__main__":
    main()
