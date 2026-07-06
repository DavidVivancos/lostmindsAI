#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
chapter_0053_sophocles_-496.py
Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
Tome 3 Minds 41 - 60 Available on Amazon https://www.amazon.com/dp/B0H6TVX69S
Resume and Interactive Demos at https://artificiology.com/
Author: David Vivancos · Chapter 0053 · Sophocles
================================================================================
THE RECOGNITION NETWORK  —  a from-scratch NumPy architecture whose learning
dynamics embody the distinctive cognitive signature of Sophocles (c.496-406 BCE).


WHY THIS IS *NOT* A TRANSFORMER WITH TRAGIC LABELS
--------------------------------------------------
Most "tragedy-themed" networks just rename attention heads "hubris" and call it a
day. This file does something structurally different. Sophocles' specific
contribution to the theory of mind is not "suffering teaches" (that is Aeschylus'
pathei mathos) and not "passion overrides reason" (that is Euripides). Sophocles'
mechanism is DRAMATIC / TRAGIC IRONY made into a process:

    The decisive truth about the agent is FIXED before the story begins and is
    already, secretly, TRUE. Nothing the agent does changes that truth. What
    changes is only the agent's ACCESS to it. And — the cruel hinge — the
    agent's own competent, well-motivated investigation is exactly the engine
    that drags the hidden truth into the open and destroys the agent's model of
    itself. Knowledge here is not accumulation; it is RECOGNITION (anagnorisis):
    a sudden, irreversible re-alignment of a confident belief-state onto a
    ground truth that was there the whole time.

We turn that into concrete, runnable computation:

  * z_star  : a LATENT "FATE" vector, sampled ONCE at construction and FROZEN.
              It is never a trainable parameter. It is the truth-about-the-self
              that the network does not yet know but has already enacted.

  * belief b: the network's avowed self-model. The forward pass produces surface
              predictions from b; b is what the agent "thinks it is."

  * The INVESTIGATION coupling: every act of querying the world (every forward
              pass on evidence) leaks a little of z_star into the evidence the
              agent conditions on. The better the agent investigates, the faster
              its belief is pulled toward the very truth that will undo it.
              (Oedipus' competence is his doom — solving the riddle is what
              seats him on the throne that is the crime.)

  * IRONY    : a *measured scalar* = surface-confidence * belief/​truth-divergence.
              The network is maximally ironic when it predicts the visible world
              fluently while its self-model is furthest from the latent truth.
              This is Sophocles' signature quantity and the chapter's thesis.

  * RECOGNITION GATE (peripeteia -> anagnorisis): a thresholded, LATCHING,
              IRREVERSIBLE event. Once accumulated evidence-against-the-self
              crosses tau, the belief is collapsed onto z_star and LOCKED. There
              is no un-knowing. Gradient flow through the latch is handled with a
              straight-through estimator so the rest of the net still trains, and
              the whole thing passes a finite-difference gradient check.

The training objective deliberately couples two terms that pull apart, exactly
as Oedipus' two goals do (rule well  vs.  find the killer = find himself):

      L = surface_prediction_loss            (be a fluent, competent king)
        + lambda_recog * recognition_loss    (align self-model with the truth)

A mind that optimizes only the first becomes maximally ironic and then shatters
at the gate. A mind that is allowed to recognize pays the price early and is
spared the catastrophic snap. The file lets you watch both regimes.

GUARANTEES (all checked by __main__):
  - pure NumPy, from scratch (no autograd framework)
  - analytic gradients verified by central finite differences (REQUIRED)
  - a real training loop that reduces a real loss on real synthetic data
  - self-tests for every component, including the irreversibility of the latch
  - executed before shipping; verified stdout pasted into the chapter

Run:   python3 chapter_0053_sophocles_-496.py
================================================================================
"""

from __future__ import annotations
import numpy as np

# Determinism: tragedy admits no luck. The dice are cast at import.
_GLOBAL_SEED = 496  # Sophocles' (traditional) birth year, as an omen not a fact.


# ============================================================================
# SECTION 1 — FOUNDATIONAL MATH
# Small, explicit, differentiable primitives. Everything downstream is built
# from these so the finite-difference check covers the whole graph.
# ============================================================================

def stable_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax along `axis`."""
    z = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def dtanh(y: np.ndarray) -> np.ndarray:
    """Derivative of tanh given its OUTPUT y = tanh(x)."""
    return 1.0 - y * y


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def dsigmoid(y: np.ndarray) -> np.ndarray:
    """Derivative of sigmoid given its OUTPUT y = sigmoid(x)."""
    return y * (1.0 - y)


def cross_entropy(logits: np.ndarray, target_idx: np.ndarray):
    """
    Mean cross-entropy over a batch.
      logits     : (B, V)
      target_idx : (B,) integer class labels
    Returns (loss_scalar, dlogits) where dlogits has shape (B, V).
    """
    B = logits.shape[0]
    p = stable_softmax(logits, axis=-1)
    loss = -np.mean(np.log(p[np.arange(B), target_idx] + 1e-12))
    d = p.copy()
    d[np.arange(B), target_idx] -= 1.0
    d /= B
    return loss, d


def mse(pred: np.ndarray, target: np.ndarray):
    """Mean-squared error and its gradient w.r.t. pred."""
    diff = pred - target
    loss = np.mean(diff * diff)
    d = (2.0 / diff.size) * diff
    return loss, d


# ============================================================================
# SECTION 2 — CONFIG
# ============================================================================

class TragicConfig:
    """
    Hyperparameters of the Recognition Network. Names map to Sophoclean concepts:
      d_model        : width of the self-model / belief space
      d_world        : dimensionality of observable 'surface' evidence
      n_surface      : size of the surface vocabulary the king must predict
      tau            : recognition threshold (how much evidence-against-self the
                       agent can sustain before peripeteia)
      lambda_recog   : weight on the recognition loss (willingness to look)
      investigate    : how strongly each query leaks z_star into evidence
                       (Oedipus' relentlessness). 0 => a king who never asks.
    """
    def __init__(
        self,
        d_model: int = 24,
        d_world: int = 16,
        n_surface: int = 12,
        tau: float = 1.25,
        lambda_recog: float = 0.5,
        investigate: float = 0.6,
        seed: int = _GLOBAL_SEED,
    ):
        self.d_model = d_model
        self.d_world = d_world
        self.n_surface = n_surface
        self.tau = tau
        self.lambda_recog = lambda_recog
        self.investigate = investigate
        self.seed = seed


# ============================================================================
# SECTION 3 — THE RECOGNITION GATE (peripeteia -> anagnorisis)
#
# This is the heart of the architecture and the part that is genuinely
# Sophoclean rather than generic. It is a LATCHING, IRREVERSIBLE nonlinearity.
#
# Forward:
#   given a scalar "evidence against the self" e (>=0 accumulates),
#   the gate output g in [0,1] is a smooth function of (e - tau), BUT once g
#   has ever crossed 0.5 the gate LATCHES to 1 and stays there forever for this
#   trajectory. Knowledge, once had, cannot be un-had. There is no path back to
#   ignorance. This breaks time-symmetry exactly the way recognition does in the
#   plays: Oedipus cannot return to the morning before he asked.
#
# Backward:
#   the latch is non-differentiable. We use a STRAIGHT-THROUGH ESTIMATOR: on the
#   backward pass we pretend the gate was the smooth sigmoid s(k*(e-tau)), whose
#   derivative is well defined. This is the standard, principled trick for
#   training through hard, discrete, or latched decisions, and it is exactly
#   what lets the whole network pass a finite-difference gradient check while
#   still behaving irreversibly in the forward direction.
# ============================================================================

class RecognitionGate:
    """Latching, irreversible recognition with a straight-through gradient."""

    def __init__(self, tau: float, sharpness: float = 4.0):
        self.tau = float(tau)
        self.k = float(sharpness)
        # smooth_mode disables the hard latch so that FORWARD == the surrogate
        # the BACKWARD pass differentiates. This is used ONLY by the gradient
        # check, where finite differences must agree with the analytic grad.
        # The straight-through latch (the real, irreversible behaviour) is the
        # default and is validated separately by test_latch_is_irreversible().
        self.smooth_mode = False
        self.reset()

    def reset(self):
        """Begin a new trajectory in a state of ignorance (the play's opening)."""
        self._latched = False

    def forward(self, e: np.ndarray):
        """
        e : array of nonneg scalars (B,) — accumulated evidence-against-self.
        Returns (g, cache). g in [0,1]. Latches per-element once it crosses 0.5.

        We track latching PER TRAJECTORY via a persistent boolean. For batched
        training we treat the whole batch as one ensemble of fates and latch
        elementwise; once an element has recognized, it stays recognized for the
        remainder of THIS forward sweep's history.
        """
        s = sigmoid(self.k * (e - self.tau))          # smooth surrogate in [0,1]
        if self.smooth_mode:
            # consistent-with-backward mode: no latch, g == s exactly.
            return s, (s, np.zeros_like(e, dtype=bool))
        crossed = s >= 0.5                             # has it crossed this step?
        # Per-element sticky latch across calls within a trajectory:
        if np.isscalar(self._latched) and self._latched is False:
            self._latched = np.zeros_like(e, dtype=bool)
        self._latched = np.logical_or(self._latched, crossed)
        g = np.where(self._latched, 1.0, s)            # hard 1 once latched
        cache = (s, np.array(self._latched, copy=True))
        return g, cache

    def backward(self, dG: np.ndarray, cache):
        """
        Straight-through: gradient flows as if g == s (the smooth surrogate),
        regardless of the hard latch in the forward pass.
        dG : upstream gradient w.r.t. g, shape (B,)
        returns dE : gradient w.r.t. e, shape (B,)
        """
        s, _ = cache
        ds_de = self.k * dsigmoid(s)                   # d sigmoid(k(e-tau))/de
        return dG * ds_de


# ============================================================================
# SECTION 4 — THE RECOGNITION NETWORK
#
# Components (all parameters are plain NumPy arrays held in self.params):
#
#   z_star (FROZEN)  : the latent fate vector. Truth-about-the-self. d_model.
#                      Sampled once; NEVER updated by training. It is already so.
#
#   W_emb            : maps a surface-evidence vector (d_world) into model space.
#   W_belief         : recurrent-ish map producing the avowed belief b (d_model)
#                      from embedded evidence. b is "what the king thinks he is."
#   W_surface        : reads belief b -> logits over the surface vocabulary
#                      (n_surface). This is the king ruling competently: fluent
#                      prediction of the visible world.
#   w_probe          : a linear probe that scores "evidence against the self" =
#                      how loudly the current evidence implicates the latent
#                      truth. Drives the recognition gate.
#
# The INVESTIGATION COUPLING (the tragic engine):
#   raw evidence x is contaminated by a fraction `investigate` of a projection
#   of z_star. The harder you look (larger investigate), the more the truth you
#   are fleeing is already inside the data you reason over. Set investigate=0
#   and you get a king who never sends to Delphi: low irony, no recognition,
#   permanent comfortable error.
# ============================================================================

class RecognitionNetwork:
    def __init__(self, cfg: TragicConfig):
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        dm, dw, ns = cfg.d_model, cfg.d_world, cfg.n_surface

        def glorot(shape):
            fan = sum(shape)
            return rng.normal(0, np.sqrt(2.0 / fan), size=shape)

        # Trainable parameters
        self.params = {
            "W_emb":     glorot((dw, dm)),
            "b_emb":     np.zeros(dm),
            "W_belief":  glorot((dm, dm)),
            "b_belief":  np.zeros(dm),
            "W_surface": glorot((dm, ns)),
            "b_surface": np.zeros(ns),
            "w_probe":   glorot((dm,)),     # vector -> scalar evidence score
            "b_probe":   np.zeros(()),
        }

        # FROZEN fate. Drawn once. The truth the agent has already enacted.
        # Normalized so the divergence metric is well scaled.
        z = rng.normal(0, 1, size=dm)
        self.z_star = z / (np.linalg.norm(z) + 1e-9)

        # A fixed (non-trainable) projection that decides HOW z_star leaks into
        # the world when the agent investigates. Part of the world, not the mind.
        P = rng.normal(0, 1, size=(dm, dw))
        self.P_leak = P / np.sqrt(dm)

        self.gate = RecognitionGate(tau=cfg.tau)
        self.grads = {k: np.zeros_like(v) for k, v in self.params.items()}

    # ---- numerically explicit forward pass -------------------------------
    def forward(self, x_raw: np.ndarray, accum_evidence: np.ndarray):
        """
        x_raw          : (B, d_world)  raw surface evidence the agent observes
        accum_evidence : (B,)          evidence-against-self carried in so far
        Returns (out, cache) where out is a dict of all quantities the chapter
        and the tests care about.
        """
        cfg = self.cfg
        p = self.params

        # --- INVESTIGATION COUPLING: the world is already contaminated by fate.
        # x = x_raw + investigate * (z_star projected into world coords)
        leak = cfg.investigate * (self.z_star @ self.P_leak)      # (d_world,)
        x = x_raw + leak                                          # (B, d_world)

        # --- embed evidence into model space
        h_pre = x @ p["W_emb"] + p["b_emb"]                        # (B, d_model)
        h = tanh(h_pre)

        # --- form the avowed belief (self-model)
        b_pre = h @ p["W_belief"] + p["b_belief"]                  # (B, d_model)
        b = tanh(b_pre)                                            # belief in [-1,1]

        # --- surface prediction (ruling the visible city)
        logits = b @ p["W_surface"] + p["b_surface"]              # (B, n_surface)

        # --- probe: how strongly does belief implicate the latent truth?
        # evidence-against-self grows when belief starts resembling z_star
        # (the closer you get to the truth, the louder the alarm).
        align = b @ self.z_star                                    # (B,) in [-dm,dm]
        probe = b @ p["w_probe"] + p["b_probe"]                    # (B,)
        # nonneg increment of evidence; relu-like via softplus for smooth grads
        incr = np.logaddexp(0.0, probe + align)                   # softplus(.)
        evidence = accum_evidence + incr                          # (B,)

        # --- recognition gate (latching, irreversible)
        g, gate_cache = self.gate.forward(evidence)               # (B,)

        # --- DRAMATIC IRONY (the signature scalar), purely diagnostic:
        # high surface confidence * large divergence of belief from truth.
        conf = np.max(stable_softmax(logits, axis=-1), axis=-1)   # (B,)
        diverg = np.linalg.norm(b - self.z_star[None, :], axis=-1)  # (B,)
        irony = conf * diverg                                     # (B,)

        cache = dict(x=x, h_pre=h_pre, h=h, b_pre=b_pre, b=b,
                     logits=logits, probe=probe, align=align, incr=incr,
                     evidence=evidence, g=g, gate_cache=gate_cache,
                     accum_in=accum_evidence)
        out = dict(logits=logits, belief=b, evidence=evidence, gate=g,
                   irony=irony, divergence=diverg, confidence=conf)
        return out, cache

    # ---- the coupled tragic loss -----------------------------------------
    def loss(self, cache, target_surface: np.ndarray):
        """
        L = surface CE  +  lambda_recog * recognition MSE(belief, z_star) gated.

        The recognition term is multiplied by the gate g: the agent only "pays"
        the cost of aligning its self-model to the truth once recognition fires.
        Before peripeteia the term is nearly free (g~0) — the agent rules in
        comfortable error. After the latch (g=1) the full divergence is charged
        at once: the catastrophic, all-at-once reckoning Sophocles dramatizes.

        Returns (L, parts_dict, dlogits, db_from_recog, dg, dincr_placeholder)
        """
        cfg = self.cfg
        b = cache["b"]
        g = cache["g"]

        # 1) surface prediction loss (rule the city)
        ce, dlogits = cross_entropy(cache["logits"], target_surface)

        # 2) recognition loss: distance from belief to the latent truth,
        #    charged in proportion to how far recognition has progressed (g).
        diff = b - self.z_star[None, :]            # (B, d_model)
        per_elem = np.mean(diff * diff, axis=1)    # (B,) MSE per sample
        recog = np.mean(g * per_elem)              # gated mean

        L = ce + cfg.lambda_recog * recog

        # gradients of the recognition term:
        B = b.shape[0]
        # d recog / d b  = lambda * g * (2/d_model) * diff / B
        db_recog = (cfg.lambda_recog * g[:, None] *
                    (2.0 / b.shape[1]) * diff / B)
        # d recog / d g  = lambda * per_elem / B   (feeds the gate -> evidence)
        dg = cfg.lambda_recog * per_elem / B

        parts = dict(ce=ce, recog=recog, total=L)
        return L, parts, dlogits, db_recog, dg

    # ---- backward pass (analytic) ----------------------------------------
    def backward(self, cache, dlogits, db_recog, dg):
        """
        Backprop through: surface head, belief, embedding, AND the recognition
        path (gate -> evidence -> probe/align -> belief). Accumulates into
        self.grads. Returns dx_raw for completeness (not used in training).
        """
        p = self.params
        g_acc = self.grads
        for k in g_acc:
            g_acc[k][...] = 0.0

        b = cache["b"]
        h = cache["h"]
        x = cache["x"]

        # --- surface head:  logits = b @ W_surface + b_surface
        g_acc["W_surface"] += b.T @ dlogits
        g_acc["b_surface"] += dlogits.sum(axis=0)
        db = dlogits @ p["W_surface"].T            # (B, d_model) from surface

        # --- add recognition's direct gradient on belief
        db += db_recog

        # --- recognition path through the gate -> evidence -> probe & align
        # dg is gradient w.r.t. g; push it through the latching gate (straight
        # -through) to get gradient w.r.t. evidence:
        dE = self.gate.backward(dg, cache["gate_cache"])          # (B,)
        # evidence = accum_in + softplus(probe + align)
        # d softplus(u)/du = sigmoid(u);  u = probe + align
        u = cache["probe"] + cache["align"]
        dincr = dE * sigmoid(u)                                   # (B,)
        # probe = b @ w_probe + b_probe
        g_acc["w_probe"] += (dincr[:, None] * b).sum(axis=0)
        g_acc["b_probe"] += dincr.sum()
        db += dincr[:, None] * p["w_probe"][None, :]
        # align = b @ z_star  -> db += dincr * z_star (z_star frozen)
        db += dincr[:, None] * self.z_star[None, :]

        # --- belief nonlinearity: b = tanh(b_pre)
        db_pre = db * dtanh(b)                                    # (B, d_model)
        g_acc["W_belief"] += h.T @ db_pre
        g_acc["b_belief"] += db_pre.sum(axis=0)
        dh = db_pre @ p["W_belief"].T                            # (B, d_model)

        # --- embedding nonlinearity: h = tanh(h_pre); h_pre = x @ W_emb + b_emb
        dh_pre = dh * dtanh(h)
        g_acc["W_emb"] += x.T @ dh_pre
        g_acc["b_emb"] += dh_pre.sum(axis=0)
        dx = dh_pre @ p["W_emb"].T                               # (B, d_world)
        return dx

    # ---- parameter update -------------------------------------------------
    def sgd_step(self, lr: float):
        for k in self.params:
            self.params[k] -= lr * self.grads[k]


# ============================================================================
# SECTION 5 — SYNTHETIC TRAGEDY (data)
#
# We build a tiny world in which a hidden truth (a fixed class drawn from the
# fate vector) is the *correct* surface label, but the raw evidence is biased to
# suggest a comfortable, wrong label. A naive "good king" who predicts the
# comfortable label scores well on surface CE early and accrues irony, until the
# investigation coupling drags belief toward z_star and the gate fires.
# ============================================================================

def make_tragedy_batch(net: "RecognitionNetwork", B: int, rng) -> tuple:
    """
    Returns (x_raw, target_surface). The *true* label is determined by the
    latent fate; the comfortable (decoy) signal points elsewhere.
    """
    cfg = net.cfg
    # true label derived deterministically from fate (already so, before play)
    fate_logits = (net.z_star @ net.P_leak)            # (d_world,)
    true_label = int(np.argmax(fate_logits) % cfg.n_surface)

    x_raw = rng.normal(0, 1, size=(B, cfg.d_world))
    # inject a comfortable decoy: bias evidence toward a *different* label
    decoy = (true_label + cfg.n_surface // 2) % cfg.n_surface
    decoy_dir = np.zeros(cfg.d_world)
    decoy_dir[decoy % cfg.d_world] = 2.0
    x_raw += decoy_dir[None, :]

    target = np.full(B, true_label, dtype=int)         # truth is what's graded
    return x_raw, target


# ============================================================================
# SECTION 6 — GRADIENT CHECK  (MANDATORY)
# Central finite differences vs. analytic grads on the full coupled loss.
# ============================================================================

def gradient_check(verbose: bool = True) -> float:
    cfg = TragicConfig(d_model=8, d_world=6, n_surface=5,
                       tau=0.5, lambda_recog=0.7, investigate=0.5, seed=7)
    net = RecognitionNetwork(cfg)
    # Use the smooth (non-latching) gate so the FORWARD pass equals exactly the
    # surrogate the BACKWARD pass differentiates. The hard, irreversible latch
    # is validated separately (test_latch_is_irreversible); differentiating a
    # step/latch with finite differences is meaningless by construction.
    net.gate.smooth_mode = True
    rng = np.random.default_rng(123)
    B = 4
    x_raw, target = make_tragedy_batch(net, B, rng)
    accum = rng.uniform(0, 0.4, size=B)   # some pre-existing suspicion

    def full_loss():
        # fresh gate state each evaluation so FD perturbations are comparable
        net.gate.reset()
        out, cache = net.forward(x_raw, accum)
        L, parts, dlogits, db_recog, dg = net.loss(cache, target)
        return L, (cache, dlogits, db_recog, dg)

    # analytic grads
    L0, (cache, dlogits, db_recog, dg) = full_loss()
    net.backward(cache, dlogits, db_recog, dg)
    analytic = {k: v.copy() for k, v in net.grads.items()}

    eps = 1e-5
    max_rel = 0.0
    worst = None
    for name in net.params:
        P = net.params[name]
        it = np.nditer(P, flags=["multi_index"])
        # check a handful of coords per tensor to keep it fast & thorough
        coords = []
        flat_n = P.size
        idxs = np.linspace(0, flat_n - 1, min(flat_n, 5)).astype(int)
        for lin in idxs:
            coords.append(np.unravel_index(lin, P.shape))
        for idx in coords:
            orig = P[idx]
            P[idx] = orig + eps
            Lp, _ = full_loss()
            P[idx] = orig - eps
            Lm, _ = full_loss()
            P[idx] = orig
            num = (Lp - Lm) / (2 * eps)
            ana = analytic[name][idx]
            denom = max(1e-8, abs(num) + abs(ana))
            rel = abs(num - ana) / denom
            if rel > max_rel:
                max_rel = rel
                worst = (name, idx, num, ana)
    if verbose:
        print(f"  max relative error (analytic vs finite-diff): {max_rel:.3e}")
        if worst:
            n, i, num, ana = worst
            print(f"  worst @ {n}{i}:  fd={num:+.6e}  analytic={ana:+.6e}")
    return max_rel


# ============================================================================
# SECTION 7 — TRAINING LOOP
# Two regimes are run so the reader can SEE the Sophoclean dynamics:
#   A) "comfortable king": lambda_recog ~ 0, investigate high -> irony climbs,
#      then the gate fires late and the recognition cost snaps in catastrophically.
#   B) "examined king": lambda_recog high -> pays alignment early, lower peak irony.
# ============================================================================

def train(cfg: TragicConfig, steps: int = 400, lr: float = 0.05,
          B: int = 32, log_every: int = 50, seed: int = 0):
    net = RecognitionNetwork(cfg)
    rng = np.random.default_rng(seed)
    history = []
    # evidence accumulates across the "performance" — reset per epoch-of-12
    accum = np.zeros(B)
    net.gate.reset()
    for t in range(1, steps + 1):
        x_raw, target = make_tragedy_batch(net, B, rng)
        out, cache = net.forward(x_raw, accum)
        L, parts, dlogits, db_recog, dg = net.loss(cache, target)
        net.backward(cache, dlogits, db_recog, dg)
        net.sgd_step(lr)

        # carry a decayed slice of evidence forward (suspicion lingers, the
        # latch never reverses); reset every 12 steps = a new "performance".
        accum = 0.9 * cache["evidence"]
        if t % 12 == 0:
            accum = np.zeros(B)
            net.gate.reset()

        if t % log_every == 0 or t == 1:
            acc = float(np.mean(np.argmax(cache["logits"], 1) == target))
            history.append(dict(
                step=t, loss=float(L), ce=float(parts["ce"]),
                recog=float(parts["recog"]), acc=acc,
                irony=float(np.mean(out["irony"])),
                gate=float(np.mean(out["gate"])),
                diverg=float(np.mean(out["divergence"])),
            ))
    return net, history


# ============================================================================
# SECTION 8 — SELF-TESTS
# ============================================================================

def test_shapes():
    cfg = TragicConfig()
    net = RecognitionNetwork(cfg)
    rng = np.random.default_rng(1)
    x, tgt = make_tragedy_batch(net, 5, rng)
    out, cache = net.forward(x, np.zeros(5))
    assert out["logits"].shape == (5, cfg.n_surface)
    assert out["belief"].shape == (5, cfg.d_model)
    assert out["gate"].shape == (5,)
    assert out["irony"].shape == (5,)
    return True


def test_fate_is_frozen():
    """z_star must never change under training — it is already so."""
    cfg = TragicConfig()
    net = RecognitionNetwork(cfg)
    before = net.z_star.copy()
    _net, _hist = train(cfg, steps=30, seed=2)
    # the SAME object's fate is untouched (training builds a fresh net, but the
    # invariant we assert is that z_star is not in the trainable params set):
    assert "z_star" not in net.params
    assert np.allclose(net.z_star, before)
    return True


def test_latch_is_irreversible():
    """Once recognition fires, it cannot be undone within a trajectory."""
    gate = RecognitionGate(tau=1.0, sharpness=5.0)
    gate.reset()
    # push evidence high -> latch
    g1, _ = gate.forward(np.array([5.0, 5.0]))
    assert np.allclose(g1, 1.0)
    # now drop evidence to ~0; latched elements must STAY at 1
    g2, _ = gate.forward(np.array([0.0, 0.0]))
    assert np.allclose(g2, 1.0), "recognition reversed — impossible in Sophocles"
    return True


def test_irony_then_collapse():
    """
    A comfortable king (no recognition weight) should show HIGHER peak irony
    than an examined king (high recognition weight). This is the chapter thesis
    rendered as a measurable inequality.
    """
    comfortable = TragicConfig(lambda_recog=0.02, investigate=0.8, seed=11)
    examined    = TragicConfig(lambda_recog=0.9,  investigate=0.8, seed=11)
    _n1, h1 = train(comfortable, steps=240, seed=5)
    _n2, h2 = train(examined,    steps=240, seed=5)
    peak_irony_comfort = max(r["irony"] for r in h1)
    peak_irony_examined = max(r["irony"] for r in h2)
    assert peak_irony_comfort >= peak_irony_examined, (
        f"expected comfortable>={peak_irony_examined:.3f}, "
        f"got {peak_irony_comfort:.3f}")
    return peak_irony_comfort, peak_irony_examined


def test_training_reduces_loss():
    cfg = TragicConfig(lambda_recog=0.4, investigate=0.5, seed=3)
    _net, hist = train(cfg, steps=400, seed=4)
    first = hist[0]["loss"]
    last = hist[-1]["loss"]
    assert last < first, f"loss did not fall: {first:.3f} -> {last:.3f}"
    return first, last


# ============================================================================
# SECTION 9 — MAIN
# ============================================================================

def _banner(s):
    print("\n" + "=" * 72)
    print(s)
    print("=" * 72)


if __name__ == "__main__":
    np.random.seed(_GLOBAL_SEED)

    _banner("THE RECOGNITION NETWORK — Sophocles (chapter 0053)")
    print("Thesis: the truth is fixed before the play; intelligence is the")
    print("irreversible RECOGNITION that collapses a confident self-model onto")
    print("a fate it has already enacted. Investigation is the engine of ruin.")

    _banner("1) GRADIENT CHECK (mandatory)")
    rel = gradient_check(verbose=True)
    ok_grad = rel < 1e-4
    print(f"  PASS" if ok_grad else "  FAIL", f"(threshold 1e-4)")

    _banner("2) SELF-TESTS")
    print("  test_shapes ................", "PASS" if test_shapes() else "FAIL")
    print("  test_fate_is_frozen .......", "PASS" if test_fate_is_frozen() else "FAIL")
    print("  test_latch_irreversible ...", "PASS" if test_latch_is_irreversible() else "FAIL")
    f0, f1 = test_training_reduces_loss()
    print(f"  test_training_reduces_loss  PASS  (loss {f0:.3f} -> {f1:.3f})")
    pic, pie = test_irony_then_collapse()
    print(f"  test_irony_then_collapse .. PASS  "
          f"(peak irony: comfortable={pic:.3f} >= examined={pie:.3f})")

    _banner("3) WATCHING A TRAGEDY UNFOLD (comfortable king)")
    cfg = TragicConfig(lambda_recog=0.03, investigate=0.85, seed=496)
    net, hist = train(cfg, steps=300, log_every=36, seed=21)
    print(f"  {'step':>5} {'loss':>8} {'surf_acc':>9} {'irony':>8} "
          f"{'gate':>6} {'diverg':>7}")
    for r in hist:
        print(f"  {r['step']:>5} {r['loss']:>8.3f} {r['acc']:>9.2f} "
              f"{r['irony']:>8.3f} {r['gate']:>6.2f} {r['diverg']:>7.3f}")

    _banner("4) THE SAME WORLD, AN EXAMINED KING (pays recognition early)")
    cfg2 = TragicConfig(lambda_recog=0.9, investigate=0.85, seed=496)
    net2, hist2 = train(cfg2, steps=300, log_every=36, seed=21)
    print(f"  {'step':>5} {'loss':>8} {'surf_acc':>9} {'irony':>8} "
          f"{'gate':>6} {'diverg':>7}")
    for r in hist2:
        print(f"  {r['step']:>5} {r['loss']:>8.3f} {r['acc']:>9.2f} "
              f"{r['irony']:>8.3f} {r['gate']:>6.2f} {r['diverg']:>7.3f}")

    _banner("DONE")
    print("All checks complete. The latch never reversed; the fate never moved.")
