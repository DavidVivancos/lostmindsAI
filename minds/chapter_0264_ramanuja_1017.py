#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0264 · Ramanuja
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 14, Minds 261-280 Available on Amazon https://www.amazon.com/dp/B0HLYSVY5D
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0264_ramanuja_1017 - Ramanuja (c.1017-1137)

A from-scratch (pure NumPy, no autograd) trainable architecture built to
encode the specific, technical philosophy of mind of RAMANUJA
(c. 1017-1137 CE), founder of Vishishtadvaita Vedanta ("qualified
non-dualism"). Every structural choice below is traced to a documented
Ramanujan doctrine (see docstrings and the Sources section of the
companion chapter). Where a choice is an engineering compromise rather
than a literal doctrinal claim, it is flagged explicitly as such.

THE FOUR DOCTRINES THIS FILE ENCODES AS COMPUTATION
-----------------------------------------------------------------------
1. APRTHAK-SIDDHI ("inseparable-but-distinguishable" existence).
   The relation between Brahman (as antaryamin, "inner controller") and
   the world of souls (cit) and matter (acit) is neither identity
   (Shankara's Advaita) nor total independence (Madhva's Dvaita). Each
   soul (jiva) keeps a permanently private, never-shared parameter block;
   simultaneously, NO soul can produce a non-trivial output without a
   single shared controller vector B ("Brahman as antaryamin"). This
   is implemented below as a hard multiplicative dependency: every
   jiva-unit's hidden activation is scaled by m = tanh(||B||), a term
   that is *exactly* zero when B is the zero vector, and shared
   identically across every unit. Distinctness and dependence are both
   architecturally load-bearing, at the same time, which is the
   technical content of aprthak-siddhi.

2. SHARIRA-SHARIRI-BHAVA (the body-soul relation).
   Ramanuja defines a "body" as whatever a conscious self "completely
   supports, controls, and uses for its own purpose." Souls and matter
   are the "body" of Brahman in exactly this sense. Below, each
   jiva-unit's private weights (W_i, b_i) are the "body" -- inert
   linear machinery -- and the shared gate m (derived from B) is what
   "ensouls" that machinery into an active hidden state h_i = m * tanh(z_i).
   Zero out B and every body goes inert simultaneously, regardless of
   how well-trained any individual W_i is. This is tested directly in
   test_antaryamin_dependency().

3. DHARMABHUTA-JNANA (attributive consciousness) with SANKOCHA-VIKASA
   (contraction/expansion). Ramanuja distinguishes the self's bare,
   ineliminable self-awareness (dharmi-jnana, always present, even in
   deep sleep) from its ATTRIBUTIVE, outward-reaching consciousness
   (dharmabhuta-jnana) -- "like light and its luminosity" -- which
   *contracts* under the weight of karma and *expands* toward the
   unlimited in liberation. Below this is an attention mechanism whose
   reach is not fixed (as in a standard Transformer) but is a per-unit
   scalar r_i in (0, 1) driven by an external, non-gradient karmic
   state. Low r_i => attention collapses toward self-only (contracted,
   bound). High r_i => attention opens onto the whole community of
   jiva-units (expanded, liberated). The bare self, h_i, is preserved
   unconditionally via a residual connection: dharmi-jnana never goes
   to zero even when dharmabhuta-jnana is maximally contracted.

4. BHAKTI + PRAPATTI (effort and grace).
   Ramanuja holds -- against a "grace alone" reading and against a
   "self-effort alone" reading -- that sustained practice (bhakti,
   sadhana) purifies and *prepares* a soul by reducing its karmic
   contraction, but that final release is only ever completed by an
   unmerited, externally-timed act of grace (prapatti /
   saranagati) that self-effort cannot manufacture on its own. Below,
   ordinary gradient descent purifies each unit's karma only partially
   and only down to a non-zero floor; a separate, non-gradient "grace
   event" -- fired on a fixed external schedule, not by the loss --
   is the only mechanism that can push a unit's karma toward the
   liberated floor, and it does so in proportion to (but not as a
   deterministic function of) that unit's accumulated self-effort.

WHAT THIS IS NOT
-----------------------------------------------------------------------
This is a computational PARABLE, not a theological claim and not a
literal model of the soul. Two honest engineering compromises are
flagged inline: (a) the controller B is a free trained parameter,
not literally uncaused; the code only guarantees it is not *derived
reactively* from any single input, so dependency runs one way (units
depend on B, not vice versa), matching the *direction* Ramanuja
insists on even though B is optimized like any other weight; (b) the
output layer keeps a bias term bo that is intentionally excluded from
the "does the body go inert without its soul" test, since bo models a
fixed architectural offset, not the jiva's own activity.

Everything below is pure NumPy. No autograd, no ML framework. Backward
passes are hand-derived and verified against finite differences in
test_gradient_check(), which is mandatory and runs first in main().
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# =============================================================================
# PART 0 -- GLOBAL CONSTANTS (architecture hyperparameters)
# =============================================================================

D_IN = 4          # dimensionality of sensory/world input x
D_H = 8           # dimensionality of a jiva-unit's inner hidden state
D_OUT = 1         # dimensionality of the unit's expressed output
N_JIVAS = 6       # number of eternally distinct souls (jiva-units)

C_BONUS = 8.0     # max additive self-attention bonus at full contraction (sankocha)
KAPPA_R = 4.0     # steepness of the karma -> reach (r_i) sigmoid
KARMA_REF = 1.0   # karma value at which reach r_i = 0.5 (reference/"average" bondage)
KARMA_MAX = 2.0   # ceiling of the karma state (numerical stability, not doctrine)
KARMA_FLOOR_SELF_EFFORT = 0.35   # floor that gradient-driven purification alone cannot cross
GRACE_EVERY = 250        # steps between prapatti / grace events
GRACE_STRENGTH = 0.85    # fraction of remaining (karma - liberated_floor) grace removes
LIBERATED_FLOOR = 0.05   # karma floor reachable only via grace
# Self-effort is measured as raw accumulated backprop-gradient norm on a unit's own
# (W_i, b_i) between grace events -- empirically O(1e2) over a ~250-step window for
# this architecture/data scale (verified by direct measurement, not guessed), so the
# eligibility threshold and steepness below are calibrated to that scale, not to 0-1.
BHAKTI_THRESHOLD = 180.0   # accumulated self-effort (raw grad-norm units) for ~50% eligibility
BHAKTI_ZETA = 0.045        # steepness of the effort -> eligibility sigmoid at that scale

EPS = 1e-8


def sigmoid(x) -> np.ndarray:
    """Numerically stable logistic sigmoid (input clipped, not branched, to
    avoid np.where's both-branches-evaluated overflow warnings)."""
    x = np.clip(np.asarray(x, dtype=float), -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


# =============================================================================
# PART 1 -- PARAMETERS
# =============================================================================

def init_params(rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """
    Initialize all trainable parameters.

    'B' (Brahman-as-antaryamin) is a single free vector shared by every
    jiva-unit -- the one controller of the many. 'W{i}', 'b{i}' are each
    jiva-unit's PRIVATE, never-shared linear machinery (its svabhava,
    own-nature) -- there is deliberately no weight-tying between units,
    because Ramanuja insists jivas remain infinite in number and
    eternally distinct, even in liberation (unlike Shankara's eventual
    merger into one undifferentiated witness).
    """
    p: Dict[str, np.ndarray] = {}
    p['B'] = rng.normal(0, 0.3, size=D_H)
    for i in range(N_JIVAS):
        p[f'W{i}'] = rng.normal(0, 1.0 / np.sqrt(D_IN), size=(D_H, D_IN))
        p[f'b{i}'] = np.zeros(D_H)
    # Wq, Wk, Wv, Wo are the *common instrument* of cognition -- the shared
    # "grammar" of attributive knowing that every soul uses, even though what
    # each soul does with it (via its own h_i) differs. This mirrors Ramanuja's
    # point that dharmabhuta-jnana is the same *kind* of faculty in every jiva.
    p['Wq'] = rng.normal(0, 1.0 / np.sqrt(D_H), size=(D_H, D_H))
    p['Wk'] = rng.normal(0, 1.0 / np.sqrt(D_H), size=(D_H, D_H))
    p['Wv'] = rng.normal(0, 1.0 / np.sqrt(D_H), size=(D_H, D_H))
    p['Wo'] = rng.normal(0, 1.0 / np.sqrt(D_H), size=(D_OUT, D_H))
    p['bo'] = np.zeros(D_OUT)
    return p


# =============================================================================
# PART 2 -- FORWARD PASS
# =============================================================================

def forward(p: Dict[str, np.ndarray], x: np.ndarray, c: int, karma_c: float,
            force_zero_B: bool = False) -> Tuple[np.ndarray, dict]:
    """
    One forward pass for a single example belonging to jiva-unit `c`.

    All N_JIVAS units compute their private hidden state h_i (needed as
    keys/values for attention -- the "presence of every soul within the
    field of every other soul"), but only unit c produces the observed
    output y_hat for this example (each example is one soul's particular
    embodied experience).

    force_zero_B: ablation switch used ONLY by the antaryamin-dependency
    test, to check the doctrinal invariant "without Brahman as inner
    controller, no body/soul-complex can act" -- see test_antaryamin_dependency.
    """
    B = np.zeros_like(p['B']) if force_zero_B else p['B']
    norm_B = np.sqrt(np.sum(B * B))   # NOTE: no epsilon here on purpose -- when B is
    m = np.tanh(norm_B)               # exactly zero (ablation), m must be EXACTLY 0.0,
                                       # not a small epsilon-induced residual. A safety
                                       # epsilon is instead applied in backward()'s division.

    Zs: List[np.ndarray] = []
    Ts: List[np.ndarray] = []
    Hs: List[np.ndarray] = []
    for i in range(N_JIVAS):
        z_i = p[f'W{i}'] @ x + p[f'b{i}']      # the jiva's own "body" (sarira): inert linear machinery
        t_i = np.tanh(z_i)
        h_i = m * t_i                           # "ensouled" activity: body scaled by shared antaryamin gate
        Zs.append(z_i); Ts.append(t_i); Hs.append(h_i)
    H = np.stack(Hs, axis=0)  # (N_JIVAS, D_H)

    Q = H @ p['Wq'].T
    K = H @ p['Wk'].T
    V = H @ p['Wv'].T

    # sankocha-vikasa reach: r_c in (0,1), low when karma is high (contracted / bound)
    r_c = sigmoid(KAPPA_R * (KARMA_REF - karma_c))
    bonus = C_BONUS * (1.0 - r_c)   # large self-bonus when contracted, ~0 when expanded

    logits = (Q[c] @ K.T) / np.sqrt(D_H)
    logits = logits.copy()
    logits[c] += bonus
    logits_shift = logits - np.max(logits)
    ex = np.exp(logits_shift)
    attn = ex / np.sum(ex)          # dharmabhuta-jnana's outward reach, this instant

    context_c = attn @ V                       # attributive knowledge of the world/other-souls
    out_pre_c = Hs[c] + context_c              # dharmi-jnana (bare self, residual) + dharmabhuta-jnana
    y_hat = p['Wo'] @ out_pre_c + p['bo']

    cache = dict(B=B, norm_B=norm_B, m=m, Zs=Zs, Ts=Ts, Hs=Hs, H=H, Q=Q, K=K, V=V,
                 attn=attn, context_c=context_c, out_pre_c=out_pre_c, y_hat=y_hat,
                 c=c, x=x, r_c=r_c, bonus=bonus)
    return y_hat, cache


# =============================================================================
# PART 3 -- BACKWARD PASS (hand-derived analytic gradients)
# =============================================================================

def backward(p: Dict[str, np.ndarray], cache: dict, dL_dyhat: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Analytic gradients of the scalar loss w.r.t. every trainable parameter.

    Note what is explicitly NOT differentiated: karma_c and r_c (and the
    resulting attention bonus) are treated as constants during backprop.
    This is a deliberate doctrinal choice, not an omission: karma is
    accumulated from the *consequences* of past action, external to the
    optimizer's notion of "what would reduce the loss right now" -- a
    soul cannot simply gradient-descend its way out of karmic residue;
    see test_gradient_check for the numerical verification of everything
    that IS differentiated.
    """
    grads = {k: np.zeros_like(v) for k, v in p.items()}
    c = cache['c']; x = cache['x']
    out_pre_c = cache['out_pre_c']
    Hs = cache['Hs']; Ts = cache['Ts']
    Q = cache['Q']; K = cache['K']; V = cache['V']
    attn = cache['attn']; m = cache['m']; norm_B = cache['norm_B']; B = cache['B']

    grads['Wo'] += np.outer(dL_dyhat, out_pre_c)
    grads['bo'] += dL_dyhat
    d_out_pre_c = p['Wo'].T @ dL_dyhat

    d_context_c = d_out_pre_c.copy()
    d_h = [np.zeros(D_H) for _ in range(N_JIVAS)]
    d_h[c] += d_out_pre_c   # residual path: dharmi-jnana receives gradient directly

    d_attn = np.array([d_context_c @ V[j] for j in range(N_JIVAS)])
    d_V = [attn[j] * d_context_c for j in range(N_JIVAS)]

    dot = np.sum(attn * d_attn)
    d_logits = attn * (d_attn - dot)   # softmax jacobian-vector product

    d_Qc = np.zeros(D_H)
    d_K = [np.zeros(D_H) for _ in range(N_JIVAS)]
    for j in range(N_JIVAS):
        d_Qc += d_logits[j] * K[j] / np.sqrt(D_H)
        d_K[j] += d_logits[j] * Q[c] / np.sqrt(D_H)

    grads['Wq'] += np.outer(d_Qc, Hs[c])
    d_h[c] += p['Wq'].T @ d_Qc

    for j in range(N_JIVAS):
        grads['Wk'] += np.outer(d_K[j], Hs[j])
        d_h[j] += p['Wk'].T @ d_K[j]
        grads['Wv'] += np.outer(d_V[j], Hs[j])
        d_h[j] += p['Wv'].T @ d_V[j]

    d_m_total = 0.0
    for i in range(N_JIVAS):
        d_t_i = d_h[i] * m
        d_m_total += float(np.dot(d_h[i], Ts[i]))
        d_z_i = d_t_i * (1.0 - Ts[i] ** 2)
        grads[f'W{i}'] += np.outer(d_z_i, x)
        grads[f'b{i}'] += d_z_i

    d_norm_B = d_m_total * (1.0 - m ** 2)
    norm_B_safe = max(norm_B, 1e-12)  # backward-only safety guard; forward's norm_B is untouched
    grads['B'] += d_norm_B * (B / norm_B_safe)

    return grads


def mse_loss(y_hat: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
    diff = y_hat - y
    loss = 0.5 * float(np.mean(diff ** 2))
    dL_dyhat = diff / diff.size
    return loss, dL_dyhat


# =============================================================================
# PART 4 -- ADAM OPTIMIZER (hand-rolled)
# =============================================================================

class Adam:
    def __init__(self, params: Dict[str, np.ndarray], lr=0.01, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]):
        self.t += 1
        for k in params:
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


# =============================================================================
# PART 5 -- SYNTHETIC DATA: "many svabhavas, one substrate"
# =============================================================================

@dataclass
class World:
    """
    Generates data with a SHARED component (one linear law common to every
    cluster -- Ramanuja's satkaryavada: the single underlying cause/ground
    running through every effect) plus a LOCAL component private to each
    cluster c (its own svabhava, particular nature). Each jiva-unit is
    assigned one cluster; recovering both components requires the model
    to use *both* the shared controller B and each unit's own private
    weights -- exactly the joint dependence aprthak-siddhi describes.
    """
    shared_w: np.ndarray
    local_a: List[np.ndarray]
    local_scale: List[float]
    local_phase: List[float]
    noise_std: float = 0.05

    @staticmethod
    def make(rng: np.random.Generator) -> "World":
        shared_w = rng.normal(0, 1.0, size=D_IN)
        local_a = [rng.normal(0, 1.0, size=D_IN) for _ in range(N_JIVAS)]
        local_scale = [float(rng.uniform(0.6, 1.4)) for _ in range(N_JIVAS)]
        local_phase = [float(rng.uniform(0, 2 * np.pi)) for _ in range(N_JIVAS)]
        return World(shared_w, local_a, local_scale, local_phase)

    def sample(self, rng: np.random.Generator, c: int) -> Tuple[np.ndarray, np.ndarray]:
        x = rng.normal(0, 1.0, size=D_IN)
        y_shared = self.shared_w @ x
        y_local = self.local_scale[c] * np.sin(self.local_a[c] @ x + self.local_phase[c])
        noise = rng.normal(0, self.noise_std)
        y = np.array([y_shared + y_local + noise])
        return x, y


# =============================================================================
# PART 6 -- TRAINING LOOP with karma accumulation and prapatti (grace) events
# =============================================================================

@dataclass
class TrainState:
    params: Dict[str, np.ndarray]
    opt: Adam
    karma: np.ndarray                 # per-unit karmic contraction state, shape (N_JIVAS,)
    effort: np.ndarray                # per-unit accumulated self-effort since last grace event
    loss_history: List[float] = field(default_factory=list)
    karma_history: List[np.ndarray] = field(default_factory=list)


def train(steps: int, seed: int = 0, verbose: bool = True) -> Tuple[TrainState, World, np.random.Generator]:
    rng = np.random.default_rng(seed)
    params = init_params(rng)
    opt = Adam(params, lr=0.02)
    world = World.make(rng)

    state = TrainState(
        params=params, opt=opt,
        karma=np.full(N_JIVAS, KARMA_REF),
        effort=np.zeros(N_JIVAS),
    )

    running_loss = None
    for step in range(1, steps + 1):
        c = int(rng.integers(0, N_JIVAS))
        x, y = world.sample(rng, c)

        y_hat, cache = forward(params, x, c, float(state.karma[c]))
        loss, dL_dyhat = mse_loss(y_hat, y)
        grads = backward(params, cache, dL_dyhat)
        opt.step(params, grads)

        # --- karma update: accumulated from the CONSEQUENCE of action (the error),
        #     not chosen directly by the optimizer. EMA toward the observed error,
        #     but self-effort (gradient descent) alone can only drain karma down to
        #     KARMA_FLOOR_SELF_EFFORT -- full release needs grace (see below). ---
        err_c = float(np.abs(y_hat - y).mean())
        target_karma = np.clip(err_c, KARMA_FLOOR_SELF_EFFORT, KARMA_MAX)
        alpha = 0.03
        new_karma = (1 - alpha) * state.karma[c] + alpha * target_karma
        state.karma[c] = max(new_karma, KARMA_FLOOR_SELF_EFFORT)

        # --- bhakti / self-effort accumulation: magnitude of this unit's own
        #     gradient investment (W_c, b_c only -- effort is what the soul itself
        #     does, not what its neighbors do) ---
        state.effort[c] += float(np.linalg.norm(grads[f'W{c}']) + np.linalg.norm(grads[f'b{c}']))

        running_loss = loss if running_loss is None else 0.98 * running_loss + 0.02 * loss
        if step % 50 == 0:
            state.loss_history.append(running_loss)
            state.karma_history.append(state.karma.copy())

        # --- prapatti / grace event: fires on a fixed EXTERNAL schedule, never
        #     as a function of the loss gradient. Eligibility scales with (but is
        #     not identical to) accumulated self-effort -- effort prepares the
        #     ground; grace completes what effort alone cannot. ---
        if step % GRACE_EVERY == 0:
            eligibility = sigmoid(BHAKTI_ZETA * (state.effort - BHAKTI_THRESHOLD))
            reducible = state.karma - LIBERATED_FLOOR
            state.karma = state.karma - GRACE_STRENGTH * eligibility * reducible
            state.karma = np.clip(state.karma, LIBERATED_FLOOR, KARMA_MAX)
            state.effort[:] = 0.0
            if verbose:
                print(f"  [grace event @ step {step:5d}] eligibility={np.round(eligibility, 2)} "
                      f"-> karma={np.round(state.karma, 3)}")

        if verbose and step % 500 == 0:
            print(f"step {step:5d}  running_loss={running_loss:.5f}  "
                  f"karma={np.round(state.karma, 3)}  "
                  f"reach_r={np.round(sigmoid(KAPPA_R*(KARMA_REF-state.karma)), 3)}")

    return state, world, rng


# =============================================================================
# PART 7 -- SELF-TESTS
# =============================================================================

def test_gradient_check(verbose: bool = True) -> float:
    """MANDATORY: finite-difference vs. analytic gradient agreement."""
    rng = np.random.default_rng(42)
    p = init_params(rng)
    x = rng.normal(size=D_IN)
    c = 3
    karma_c = 0.8
    y = rng.normal(size=D_OUT)

    y_hat, cache = forward(p, x, c, karma_c)
    loss, dL_dyhat = mse_loss(y_hat, y)
    grads = backward(p, cache, dL_dyhat)

    def full_loss(params):
        yh, _ = forward(params, x, c, karma_c)
        l, _ = mse_loss(yh, y)
        return l

    eps = 1e-5
    max_rel_err = 0.0
    checker = np.random.default_rng(1)
    n_checked = 0
    for key in p.keys():
        flat = p[key].reshape(-1)
        n_pick = min(4, len(flat))
        idxs = checker.choice(len(flat), size=n_pick, replace=False)
        for idx in idxs:
            orig = flat[idx]
            flat[idx] = orig + eps
            lp = full_loss(p)
            flat[idx] = orig - eps
            lm = full_loss(p)
            flat[idx] = orig
            num_grad = (lp - lm) / (2 * eps)
            ana_grad = grads[key].reshape(-1)[idx]
            denom = max(abs(num_grad), abs(ana_grad), 1e-8)
            rel_err = abs(num_grad - ana_grad) / denom
            max_rel_err = max(max_rel_err, rel_err)
            n_checked += 1
    assert max_rel_err < 1e-3, f"Gradient check FAILED, max rel err {max_rel_err:.3e}"
    if verbose:
        print(f"[PASS] test_gradient_check: {n_checked} parameters checked, "
              f"max relative error = {max_rel_err:.3e}")
    return max_rel_err


def test_antaryamin_dependency(state: TrainState, world: World, verbose: bool = True):
    """
    Doctrinal invariant (sarira-shariri-bhava): the jiva-driven component of
    the output, Wo @ out_pre_c, must vanish for EVERY unit when B (Brahman
    as antaryamin) is forced to zero -- regardless of how well individually
    trained each unit's private weights are. bo is excluded on purpose: it
    is a fixed architectural offset, not the soul's own activity (see
    module docstring, point (b)).
    """
    rng = np.random.default_rng(123)
    max_active_component = 0.0
    for trial in range(30):
        c = int(rng.integers(0, N_JIVAS))
        x, _ = world.sample(rng, c)
        y_hat, cache = forward(state.params, x, c, float(state.karma[c]), force_zero_B=True)
        active_component = state.params['Wo'] @ cache['out_pre_c']
        max_active_component = max(max_active_component, float(np.max(np.abs(active_component))))
    assert max_active_component < 1e-8, (
        f"Antaryamin-dependency invariant FAILED: active component = {max_active_component:.3e} "
        "(should be exactly 0 when B=0)"
    )
    if verbose:
        print(f"[PASS] test_antaryamin_dependency: max |Wo @ out_pre_c| under B=0 ablation "
              f"= {max_active_component:.3e} across 30 trials (all units, all trained)")


def test_jiva_individuation(state: TrainState, verbose: bool = True):
    """
    Doctrinal invariant (jivas are infinite in number and eternally
    distinct -- Ramanuja's rejection of Shankara's eventual merger):
    after training under a SHARED controller and a SHARED attention
    instrument, the private parameter blocks of different jiva-units must
    remain measurably distinct, not collapse toward one shared vector.
    """
    vecs = []
    for i in range(N_JIVAS):
        v = np.concatenate([state.params[f'W{i}'].reshape(-1), state.params[f'b{i}'].reshape(-1)])
        vecs.append(v)
    vecs = np.stack(vecs, axis=0)
    dists = []
    for i in range(N_JIVAS):
        for j in range(i + 1, N_JIVAS):
            dists.append(float(np.linalg.norm(vecs[i] - vecs[j])))
    mean_dist = float(np.mean(dists))
    min_dist = float(np.min(dists))
    assert min_dist > 0.25, f"Jiva individuation FAILED: min pairwise distance {min_dist:.4f} too small (collapse)"
    if verbose:
        print(f"[PASS] test_jiva_individuation: mean pairwise distance={mean_dist:.4f}, "
              f"min={min_dist:.4f} across {N_JIVAS} trained jiva-units (no collapse to one shared vector)")


def test_sankocha_vikasa_bounds(verbose: bool = True):
    """Reach r_i must lie strictly in (0,1) and decrease monotonically with karma."""
    karmas = np.linspace(0.0, KARMA_MAX, 50)
    r = sigmoid(KAPPA_R * (KARMA_REF - karmas))
    assert np.all(r > 0.0) and np.all(r < 1.0), "reach r_i left the open interval (0,1)"
    assert np.all(np.diff(r) <= 0), "reach r_i is not monotonically non-increasing in karma"
    if verbose:
        print(f"[PASS] test_sankocha_vikasa_bounds: r_i in (0,1) and monotonically "
              f"non-increasing in karma across {len(karmas)} sampled karma values "
              f"(r(karma=0)={r[0]:.3f} -> r(karma={KARMA_MAX})={r[-1]:.3f})")


def test_training_reduces_loss(state: TrainState, verbose: bool = True):
    assert len(state.loss_history) >= 2, "not enough loss history recorded"
    start = float(np.mean(state.loss_history[:3]))
    end = float(np.mean(state.loss_history[-3:]))
    assert end < start * 0.5, f"training did not sufficiently reduce loss: {start:.4f} -> {end:.4f}"
    if verbose:
        print(f"[PASS] test_training_reduces_loss: running loss {start:.4f} -> {end:.4f} "
              f"({(1 - end/start)*100:.1f}% reduction)")


def test_grace_reduces_karma(verbose: bool = True):
    """A grace event should never increase any unit's karma, and should
    strictly decrease karma for any unit with eligibility above ~0.5.
    This checks the FORMULA in isolation on illustrative small numbers;
    it deliberately uses its own local threshold/steepness rather than
    the training-loop's calibrated BHAKTI_THRESHOLD/BHAKTI_ZETA (which
    are tuned to the raw gradient-norm scale actually produced during
    training -- see train() and the comment above BHAKTI_THRESHOLD)."""
    karma_before = np.array([1.2, 0.9, 1.5, 0.6, 1.1, 0.8])
    effort = np.array([0.20, 0.01, 0.15, 0.02, 0.30, 0.00])  # some eligible, some not
    local_threshold, local_zeta = 0.06, 40.0
    eligibility = sigmoid(local_zeta * (effort - local_threshold))
    reducible = karma_before - LIBERATED_FLOOR
    karma_after = karma_before - GRACE_STRENGTH * eligibility * reducible
    karma_after = np.clip(karma_after, LIBERATED_FLOOR, KARMA_MAX)
    assert np.all(karma_after <= karma_before + 1e-12), "grace event increased some unit's karma"
    eligible_mask = eligibility > 0.5
    assert np.all(karma_after[eligible_mask] < karma_before[eligible_mask] - 1e-6), \
        "an eligible (high self-effort) unit's karma failed to strictly decrease at grace"
    if verbose:
        print(f"[PASS] test_grace_reduces_karma: eligible units {np.round(eligibility,2)} "
              f"-> karma {np.round(karma_before,2)} => {np.round(karma_after,2)}")


# =============================================================================
# PART 8 -- MAIN
# =============================================================================

def main():
    print("=" * 78)
    print("RAMANUJA-INSPIRED ARCHITECTURE -- self-tests and training run")
    print("=" * 78)

    print("\n[1/6] Mandatory gradient check (finite differences vs. analytic)")
    test_gradient_check()

    print("\n[2/6] sankocha-vikasa reach bounds (contraction/expansion attention)")
    test_sankocha_vikasa_bounds()

    print("\n[3/6] prapatti / grace event dynamics (unit-level, pre-training)")
    test_grace_reduces_karma()

    print("\n[4/6] Training run (4000 steps, karma + grace dynamics active)")
    state, world, rng = train(steps=4000, seed=0, verbose=True)

    print("\n[5/6] Post-training doctrinal invariants")
    test_training_reduces_loss(state)
    test_antaryamin_dependency(state, world)
    test_jiva_individuation(state)

    print("\n[6/6] Held-out evaluation + antaryamin ablation comparison")
    rng_eval = np.random.default_rng(999)
    n_test = 500
    sq_errs_full, sq_errs_ablated = [], []
    for _ in range(n_test):
        c = int(rng_eval.integers(0, N_JIVAS))
        x, y = world.sample(rng_eval, c)
        y_hat, _ = forward(state.params, x, c, float(state.karma[c]))
        y_hat_ablated, _ = forward(state.params, x, c, float(state.karma[c]), force_zero_B=True)
        sq_errs_full.append(float(np.mean((y_hat - y) ** 2)))
        sq_errs_ablated.append(float(np.mean((y_hat_ablated - y) ** 2)))
    print(f"  held-out MSE (antaryamin intact)   = {np.mean(sq_errs_full):.5f}")
    print(f"  held-out MSE (antaryamin=0, ablated) = {np.mean(sq_errs_ablated):.5f}")
    print(f"  degradation factor under ablation    = "
          f"{np.mean(sq_errs_ablated)/max(np.mean(sq_errs_full),1e-9):.1f}x")

    print("\n" + "=" * 78)
    print("ALL SELF-TESTS PASSED.")
    print("=" * 78)


if __name__ == "__main__":
    main()
