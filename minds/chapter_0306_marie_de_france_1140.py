#!/usr/bin/env python3
"""
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0306 · Marie de France (fl. c. 1160-1215 CE)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 16, Minds 301-320 Available on Amazon https://www.amazon.com/gp/product/B0HNJQX9S4
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos 
A from-scratch, pure-NumPy trainable neural architecture built to embody a
single, textually-grounded thesis reconstructed from Marie de France's own
General Prologue to the Lais and tested against her Fables and her lai
"Bisclavret":

    Legitimate transmission of understanding is not maximal fidelity to a
    source, and it is not unconstrained invention. It is FIDELITY where the
    source is determinate, plus GATED SURPLUS -- new, checkable
    understanding -- exactly where the source is genuinely obscure; and any
    claim of RECOGNITION between two presentations of "the same case" must
    be verified at the level of an invariant evaluative CORE, not at the
    level of a mutable surface PERIPHERY that a false semblance can imitate.

Three coupled ideas from the chapter are made computational here:

  1. CORE / PERIPHERY SPLIT (the "Bisclavret" test, chapter Sec. 2.2 & 5)
     An input is encoded into an invariant CORE (meant to persist across a
     transformation of embodiment/context) and a mutable PERIPHERY (meant to
     track the surface form of a given presentation). A same-case,
     different-form pair is trained to keep its CORE close while its
     PERIPHERY is left free to differ.

  2. GLOSS + GATED SURPLUS (the General Prologue thesis, chapter Sec. 2.1,
     4 & 5) A decoder reconstructs the *legible* part of the input exactly
     (gloser la lettre / fidelity). A second head proposes SURPLUS content
     -- information beyond the literal input -- but that head's influence on
     the final output is gated by a learned, *supervised* estimate of how
     obscure/ambiguous the instance actually is. Low ambiguity -> the
     output should lean on a conservative, literal baseline reading. High
     ambiguity -> the output is permitted, and required, to lean on the
     learned interpretive (surplus) pathway.

  3. RECOGNITION AGAINST FALSE SEMBLANCE (the "Lanval" test, chapter Sec. 3)
     A recognition head is trained to judge whether two presentations
     belong to "the same case" using only their CORE embeddings, against
     deliberately constructed "false semblance" pairs whose PERIPHERY is
     made to look similar while their CORE (their true underlying case)
     differs -- exactly the shape of the queen's convincing but false
     accusation in Lanval, which a court cannot resolve from surface form
     alone.

This file is a small, complete, and independently checkable instrument for
testing whether those three distinctions can be made to function as
measurable computational properties, not a general-purpose language model.
It does not use attention, transformers, or any pretrained component: every
layer, every gradient, and every training step is implemented from raw
NumPy arrays below.

Structure
---------
  SECTION 1  Dense layer + tiny 2-layer MLP building blocks (manual
             forward/backward, cache-based so the same layer can be called
             multiple times per training step without state collision).
  SECTION 2  Adam optimizer (from scratch).
  SECTION 3  Synthetic "case" data generator implementing the core/
             periphery/obscurity/surplus generative process described above.
  SECTION 4  The MarieMind model: five sub-networks (core encoder, periphery
             encoder, ambiguity/gate head, gloss decoder, surplus decoder,
             baseline reader, recognition head) wired together with fully
             manual backpropagation, including gradient accumulation across
             the multiple loss branches that share the two encoders.
  SECTION 5  Finite-difference gradient check (mandatory correctness proof).
  SECTION 6  Training loop.
  SECTION 7  Self-tests tied directly to the chapter's three claims.
  SECTION 8  Demonstration entry point.

Author note: this module intentionally works over compact structured
"case" vectors rather than open text -- a real and stated limitation (see
chapter Sec. 6) -- because the point of this file is to make the *shape* of
Marie's distinctions checkable end-to-end, not to build a language model.
"""

from __future__ import annotations

import numpy as np


# =============================================================================
# SECTION 1: DENSE LAYER AND 2-LAYER MLP (manual forward / backward)
# =============================================================================

class Dense:
    """A single affine layer plus a pointwise activation, with a fully
    manual, cache-based backward pass. forward() and backward() are pure
    functions of their arguments (no hidden layer-level state), so the same
    Dense instance can be safely reused several times within one training
    step -- e.g. the same core encoder applied to form A, form B, and a
    recognition-pair batch, all within a single gradient step.
    """

    def __init__(self, in_dim: int, out_dim: int, activation: str = "tanh",
                 rng: np.random.Generator | None = None):
        rng = rng if rng is not None else np.random.default_rng()
        # Xavier/Glorot-style uniform init, appropriate for tanh/sigmoid.
        limit = np.sqrt(6.0 / (in_dim + out_dim))
        self.W = rng.uniform(-limit, limit, size=(in_dim, out_dim))
        self.b = np.zeros(out_dim)
        assert activation in ("tanh", "sigmoid", "linear")
        self.activation = activation

    def forward(self, X: np.ndarray):
        Z = X @ self.W + self.b
        if self.activation == "tanh":
            A = np.tanh(Z)
        elif self.activation == "sigmoid":
            A = 1.0 / (1.0 + np.exp(-Z))
        else:  # linear
            A = Z
        cache = (X, A)
        return A, cache

    def backward(self, dA: np.ndarray, cache):
        X, A = cache
        if self.activation == "tanh":
            dZ = dA * (1.0 - A ** 2)
        elif self.activation == "sigmoid":
            dZ = dA * A * (1.0 - A)
        else:  # linear
            dZ = dA
        dW = X.T @ dZ
        db = dZ.sum(axis=0)
        dX = dZ @ self.W.T
        return dX, dW, db

    def params(self):
        return [self.W, self.b]


class MLP2:
    """A 2-layer MLP: Dense(in->hidden, tanh) -> Dense(hidden->out, act)."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 out_activation: str = "tanh", rng: np.random.Generator | None = None):
        self.l1 = Dense(in_dim, hidden_dim, "tanh", rng)
        self.l2 = Dense(hidden_dim, out_dim, out_activation, rng)

    def forward(self, X: np.ndarray):
        H, c1 = self.l1.forward(X)
        Y, c2 = self.l2.forward(H)
        return Y, (c1, c2)

    def backward(self, dY: np.ndarray, cache):
        c1, c2 = cache
        dH, dW2, db2 = self.l2.backward(dY, c2)
        dX, dW1, db1 = self.l1.backward(dH, c1)
        # Returns dX (grad wrt this MLP's input) and the four param grads,
        # ordered to match self.params().
        return dX, [dW1, db1, dW2, db2]

    def params(self):
        return [self.l1.W, self.l1.b, self.l2.W, self.l2.b]


# =============================================================================
# SECTION 2: ADAM OPTIMIZER (from scratch)
# =============================================================================

class Adam:
    """Standard Adam, keyed by python id() of each parameter array so the
    same optimizer instance can track many small parameter tensors across
    several sub-networks without any framework machinery."""

    def __init__(self, lr: float = 0.01, beta1: float = 0.9, beta2: float = 0.999,
                 eps: float = 1e-8):
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.m: dict[int, np.ndarray] = {}
        self.v: dict[int, np.ndarray] = {}
        self.t = 0

    def step(self, params: list[np.ndarray], grads: list[np.ndarray]):
        self.t += 1
        for p, g in zip(params, grads):
            key = id(p)
            if key not in self.m:
                self.m[key] = np.zeros_like(p)
                self.v[key] = np.zeros_like(p)
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * g
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (g ** 2)
            mhat = self.m[key] / (1 - self.beta1 ** self.t)
            vhat = self.v[key] / (1 - self.beta2 ** self.t)
            p -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


# =============================================================================
# SECTION 3: SYNTHETIC "CASE" DATA GENERATOR
# =============================================================================
#
# A "case" stands for one narrative/moral situation (a Lanval, a Bisclavret,
# a wolf-and-lamb dispute). Each case has:
#   - a persistent CORE vector: the evaluative ranking that should survive
#     transformation (Dc dims).
# Each case is observed in two FORMS (e.g. courtly presentation / transformed
# presentation), each with:
#   - an independently sampled PERIPHERY vector: the surface form of that
#     particular presentation (Dp dims).
#   - a scalar ambiguity level a in (0.1, 0.9): how obscure this instance's
#     "letter" is.
# The observed LETTER x is built from three blocks:
#   - x_core   (Dxc dims): a low-noise, reliably legible function of CORE.
#   - x_periph (Dxp dims): a low-noise, reliably legible function of PERIPHERY.
#   - x_obs    (Dxo dims): a function of (CORE, PERIPHERY) that is
#     increasingly corrupted by noise as `a` grows -- "the ancients wrote
#     obscurely" operationalized as a literal signal-to-noise parameter.
# The SURPLUS target y (Dy dims) is a fully deterministic function of
# (CORE, PERIPHERY) -- genuine understanding that is, in principle, always
# recoverable from the underlying case, but which x_obs alone only reveals
# when `a` is low. When `a` is high, recovering y requires the model to have
# learned the (core, periphery) -> y association across many cases, i.e. to
# supply real interpretive surplus rather than read it off the letter.
# =============================================================================

Dc, Dp = 4, 4                       # core / periphery latent dims
Dxc, Dxp, Dxo = 4, 4, 3             # letter block widths
Dx = Dxc + Dxp + Dxo                # total input width (11)
Dy = 4                              # surplus target width


class CaseGenerator:
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        # Fixed random "world" projections shared by every case, standing in
        # for the (unknown, but consistent) rules by which underlying
        # evaluative structure surfaces as observable narrative detail.
        self.W_core_to_xc = self.rng.normal(scale=0.9, size=(Dc, Dxc))
        self.W_periph_to_xp = self.rng.normal(scale=0.9, size=(Dp, Dxp))
        self.W_core_to_xo = self.rng.normal(scale=0.7, size=(Dc, Dxo))
        self.W_periph_to_xo = self.rng.normal(scale=0.7, size=(Dp, Dxo))
        self.W_core_to_y = self.rng.normal(scale=0.8, size=(Dc, Dy))
        self.W_periph_to_y = self.rng.normal(scale=0.8, size=(Dp, Dy))

    def sample_cases(self, n_cases: int):
        """Returns per-case core vectors, shape (n_cases, Dc)."""
        return self.rng.normal(size=(n_cases, Dc))

    def sample_form(self, core: np.ndarray, ambiguity: np.ndarray | None = None,
                     periphery: np.ndarray | None = None):
        """Given a batch of core vectors (N, Dc), sample one FORM each:
        a periphery, an ambiguity level, the observed letter x, and the
        deterministic surplus target y. `ambiguity` / `periphery` may be
        supplied explicitly (used later to engineer "false semblance"
        pairs with deliberately matched periphery)."""
        n = core.shape[0]
        if periphery is None:
            periphery = self.rng.normal(size=(n, Dp))
        if ambiguity is None:
            ambiguity = self.rng.uniform(0.1, 0.9, size=n)

        x_core = np.tanh(core @ self.W_core_to_xc) + self.rng.normal(scale=0.03, size=(n, Dxc))
        x_periph = np.tanh(periphery @ self.W_periph_to_xp) + self.rng.normal(scale=0.03, size=(n, Dxp))

        clean_obscure = np.tanh(core @ self.W_core_to_xo + periphery @ self.W_periph_to_xo)
        noise = self.rng.normal(scale=1.0, size=(n, Dxo))
        a_col = ambiguity[:, None]
        # As ambiguity -> 1, x_obs becomes dominated by noise (the "letter"
        # genuinely underdetermines the case); as ambiguity -> 0, x_obs is
        # close to the clean, literally-recoverable signal.
        x_obs = (1 - a_col) * clean_obscure + a_col * noise

        y = np.tanh(core @ self.W_core_to_y + periphery @ self.W_periph_to_y)

        x = np.concatenate([x_core, x_periph, x_obs], axis=1)
        return x, y, ambiguity, periphery

    def transformation_pair(self, core: np.ndarray):
        """Two independent FORMS of the same cases -- the Bisclavret test:
        same core, independently sampled periphery and ambiguity each time."""
        x_a, y_a, a_a, p_a = self.sample_form(core)
        x_b, y_b, a_b, p_b = self.sample_form(core)
        return (x_a, y_a, a_a, p_a), (x_b, y_b, a_b, p_b)

    def false_semblance_pairs(self, n_pairs: int):
        """Construct 'Lanval' hard-negative pairs: TWO DIFFERENT cases
        (different core) whose PERIPHERY is deliberately made near-identical
        -- a convincing surface match concealing a genuine mismatch beneath
        it, exactly the shape of the queen's performed grievance."""
        core_1 = self.sample_cases(n_pairs)
        core_2 = self.sample_cases(n_pairs)
        shared_periphery = self.rng.normal(size=(n_pairs, Dp))
        jitter = self.rng.normal(scale=0.05, size=(n_pairs, Dp))
        x1, _, _, _ = self.sample_form(core_1, periphery=shared_periphery)
        x2, _, _, _ = self.sample_form(core_2, periphery=shared_periphery + jitter)
        return x1, x2


# =============================================================================
# SECTION 4: THE MARIEMIND MODEL
# =============================================================================

class MarieMind:
    """Wires the six sub-networks together and implements one fully manual
    forward+backward training step, accumulating gradients into the two
    shared encoders from every loss branch that touches them (fidelity,
    surplus, invariance, recognition)."""

    HIDDEN = 20
    GATE_HIDDEN = 32

    def __init__(self, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.enc_core = MLP2(Dx, self.HIDDEN, Dc, "tanh", rng)
        self.enc_periph = MLP2(Dx, self.HIDDEN, Dp, "tanh", rng)
        self.gate_head = MLP2(Dx, self.GATE_HIDDEN, 1, "sigmoid", rng)
        self.gloss_decoder = MLP2(Dc + Dp, self.HIDDEN, Dxc + Dxp, "tanh", rng)
        self.surplus_decoder = MLP2(Dc + Dp, self.HIDDEN, Dy, "tanh", rng)
        self.baseline = Dense(Dxo, Dy, "tanh", rng)
        self.recognition_head = MLP2(3 * Dc, self.HIDDEN, 1, "linear", rng)

        self._modules = {
            "enc_core": self.enc_core, "enc_periph": self.enc_periph,
            "gate_head": self.gate_head, "gloss_decoder": self.gloss_decoder,
            "surplus_decoder": self.surplus_decoder, "baseline": self.baseline,
            "recognition_head": self.recognition_head,
        }
        self.opt = Adam(lr=0.006)

    # ---- flat parameter/gradient views, used by the optimizer and by the
    # gradient checker ----
    def all_params(self):
        ps = []
        for m in self._modules.values():
            ps.extend(m.params())
        return ps

    # ---- core forward primitive, reused everywhere ----
    def encode(self, X: np.ndarray):
        core, cache_core = self.enc_core.forward(X)
        periph, cache_periph = self.enc_periph.forward(X)
        return core, periph, cache_core, cache_periph

    def forward_case(self, X: np.ndarray):
        """Full forward pass for a main-batch of letters X (N, Dx). Returns
        predictions plus every cache needed for backward()."""
        core, periph, c_core, c_periph = self.encode(X)
        u, c_gate = self.gate_head.forward(X)                    # (N,1)
        cat = np.concatenate([core, periph], axis=1)              # (N, Dc+Dp)
        recon, c_recon = self.gloss_decoder.forward(cat)          # (N, Dxc+Dxp)
        raw_sur, c_sur = self.surplus_decoder.forward(cat)        # (N, Dy)
        x_obs = X[:, Dxc + Dxp:]
        base, c_base = self.baseline.forward(x_obs)               # (N, Dy)
        gated = (1 - u) * base + u * raw_sur                      # (N, Dy)

        out = dict(core=core, periph=periph, u=u, recon=recon,
                   raw_sur=raw_sur, base=base, gated=gated)
        cache = dict(c_core=c_core, c_periph=c_periph, c_gate=c_gate,
                     c_recon=c_recon, c_sur=c_sur, c_base=c_base, x_obs=x_obs)
        return out, cache

    def backward_case(self, X, out, cache, target_recon, y, a_true,
                       w_fid=1.0, w_sur=1.0, w_gate=1.0):
        """Backprop through the main-batch branch (fidelity + surplus +
        gate calibration). Returns accumulated grads keyed by module name,
        plus dCore/dPeriph so callers can add further branches (invariance,
        recognition) before the encoders' own backward() is invoked."""
        N = X.shape[0]
        u, recon, raw_sur, base, gated = out["u"], out["recon"], out["raw_sur"], out["base"], out["gated"]

        # --- fidelity: L_fid = mean((recon - target_recon)**2) ---
        dRecon = w_fid * 2.0 * (recon - target_recon) / target_recon.size
        dCat_fid, g_gloss = self.gloss_decoder.backward(dRecon, cache["c_recon"])
        dCore = dCat_fid[:, :Dc].copy()
        dPeriph = dCat_fid[:, Dc:].copy()

        # --- surplus: L_sur = mean((gated - y)**2) ---
        dGated = w_sur * 2.0 * (gated - y) / y.size
        dU_from_sur = np.sum(dGated * (raw_sur - base), axis=1, keepdims=True)
        dBase = dGated * (1 - u)
        dRawSur = dGated * u
        dCat_sur, g_surdec = self.surplus_decoder.backward(dRawSur, cache["c_sur"])
        dCore += dCat_sur[:, :Dc]
        dPeriph += dCat_sur[:, Dc:]
        _, dW_base, db_base = self.baseline.backward(dBase, cache["c_base"])

        # --- gate calibration: L_gate = mean((u - a_true)**2) ---
        dU_from_gate = w_gate * 2.0 * (u - a_true.reshape(-1, 1)) / u.size
        dU_total = dU_from_sur + dU_from_gate
        _, g_gate = self.gate_head.backward(dU_total, cache["c_gate"])

        grads = {
            "gloss_decoder": g_gloss,
            "surplus_decoder": g_surdec,
            "baseline": [dW_base, db_base],
            "gate_head": g_gate,
        }
        losses = dict(
            fid=float(np.mean((recon - target_recon) ** 2)),
            sur=float(np.mean((gated - y) ** 2)),
            gate=float(np.mean((u.ravel() - a_true) ** 2)),
        )
        return grads, dCore, dPeriph, losses

    def invariance_branch(self, X_a, X_b, w_inv=1.0):
        """L_inv = mean((core(X_a) - core(X_b))**2) for same-case pairs."""
        core_a, c_a = self.enc_core.forward(X_a)
        core_b, c_b = self.enc_core.forward(X_b)
        diff = core_a - core_b
        dCore_a = w_inv * 2.0 * diff / diff.size
        dCore_b = -dCore_a
        _, g_a = self.enc_core.backward(dCore_a, c_a)
        _, g_b = self.enc_core.backward(dCore_b, c_b)
        g_sum = [ga + gb for ga, gb in zip(g_a, g_b)]
        loss = float(np.mean(diff ** 2))
        return g_sum, loss

    def recognition_branch(self, X1, X2, labels, w_rec=1.0):
        """Binary recognition of 'same underlying case' from CORE alone.
        labels: 1 for genuine same-case pairs, 0 for false-semblance or
        random-negative pairs. recognition_head outputs a logit; the
        logit/BCE combination is used so the backward pass is the clean
        (prob - label)/N form, avoiding a redundant sigmoid-derivative
        multiplication."""
        core1, c1 = self.enc_core.forward(X1)
        core2, c2 = self.enc_core.forward(X2)
        feat = np.concatenate([core1, core2, np.abs(core1 - core2)], axis=1)
        logit, c_head = self.recognition_head.forward(feat)
        prob = 1.0 / (1.0 + np.exp(-logit))
        N = X1.shape[0]
        eps = 1e-9
        bce = -np.mean(labels.reshape(-1, 1) * np.log(prob + eps)
                        + (1 - labels.reshape(-1, 1)) * np.log(1 - prob + eps))

        dLogit = w_rec * (prob - labels.reshape(-1, 1)) / N
        dFeat, g_head = self.recognition_head.backward(dLogit, c_head)
        dCore1 = dFeat[:, :Dc].copy()
        dCore2 = dFeat[:, Dc:2 * Dc].copy()
        dAbs = dFeat[:, 2 * Dc:]
        sgn = np.sign(core1 - core2)
        dCore1 += dAbs * sgn
        dCore2 -= dAbs * sgn

        _, g_c1 = self.enc_core.backward(dCore1, c1)
        _, g_c2 = self.enc_core.backward(dCore2, c2)
        g_core_sum = [ga + gb for ga, gb in zip(g_c1, g_c2)]
        return g_head, g_core_sum, float(bce), prob

    # ---- one full training step ----
    def train_step(self, gen: CaseGenerator, batch_cases: int, rec_pairs: int,
                    weights: dict):
        core = gen.sample_cases(batch_cases)
        (x_a, y_a, a_a, p_a), (x_b, y_b, a_b, p_b) = gen.transformation_pair(core)

        # Concatenate both forms into one main batch for the
        # fidelity/surplus/gate branch (every observed letter, regardless
        # of which of the two forms it is, is glossed the same way).
        X = np.concatenate([x_a, x_b], axis=0)
        Y = np.concatenate([y_a, y_b], axis=0)
        A = np.concatenate([a_a, a_b], axis=0)
        target_recon = X[:, :Dxc + Dxp]

        out, cache = self.forward_case(X)
        grads_main, dCore, dPeriph, losses = self.backward_case(
            X, out, cache, target_recon, Y, A,
            w_fid=weights["fid"], w_sur=weights["sur"], w_gate=weights["gate"])

        # Split dCore / dPeriph back out and backprop through the shared
        # encoders for THIS branch first.
        _, g_core_main = self.enc_core.backward(dCore, cache["c_core"])
        _, g_periph_main = self.enc_periph.backward(dPeriph, cache["c_periph"])

        # Invariance branch (Bisclavret test): same-case core(x_a) vs core(x_b).
        g_core_inv, inv_loss = self.invariance_branch(x_a, x_b, w_inv=weights["inv"])

        # Recognition branch (Lanval test): genuine + false-semblance + random.
        n = rec_pairs
        core_pos = gen.sample_cases(n)
        (xp_a, _, _, _), (xp_b, _, _, _) = gen.transformation_pair(core_pos)
        x1_fs, x2_fs = gen.false_semblance_pairs(n)
        core_r1 = gen.sample_cases(n)
        core_r2 = gen.sample_cases(n)
        x_r1, _, _, _ = gen.sample_form(core_r1)
        x_r2, _, _, _ = gen.sample_form(core_r2)

        X1 = np.concatenate([xp_a, x1_fs, x_r1], axis=0)
        X2 = np.concatenate([xp_b, x2_fs, x_r2], axis=0)
        labels = np.concatenate([np.ones(n), np.zeros(n), np.zeros(n)])
        g_head_rec, g_core_rec, rec_loss, _ = self.recognition_branch(
            X1, X2, labels, w_rec=weights["rec"])

        # --- accumulate all encoder-touching gradients ---
        g_core_total = [a + b + c for a, b, c in zip(g_core_main, g_core_inv, g_core_rec)]
        g_periph_total = g_periph_main  # periphery only touched by main branch

        grads_by_module = {
            "enc_core": g_core_total,
            "enc_periph": g_periph_total,
            "gloss_decoder": grads_main["gloss_decoder"],
            "surplus_decoder": grads_main["surplus_decoder"],
            "baseline": grads_main["baseline"],
            "gate_head": grads_main["gate_head"],
            "recognition_head": g_head_rec,
        }

        # --- optimizer step ---
        all_params, all_grads = [], []
        for name, module in self._modules.items():
            all_params.extend(module.params())
            all_grads.extend(grads_by_module[name])
        self.opt.step(all_params, all_grads)

        total_loss = (weights["fid"] * losses["fid"] + weights["sur"] * losses["sur"]
                      + weights["gate"] * losses["gate"] + weights["inv"] * inv_loss
                      + weights["rec"] * rec_loss)
        return dict(total=total_loss, fid=losses["fid"], sur=losses["sur"],
                    gate=losses["gate"], inv=inv_loss, rec=rec_loss)

    # ---- loss-only forward, used by the gradient checker ----
    def compute_total_loss(self, gen_state, weights):
        """Recompute the total loss from a FIXED, pre-sampled dataset
        (gen_state), with no new random sampling -- required so the finite
        difference check perturbs a weight against an identical batch."""
        (x_a, y_a, a_a), (x_b, y_b, a_b), (x1_fs, x2_fs), (xp_a, xp_b), (x_r1, x_r2) = gen_state

        X = np.concatenate([x_a, x_b], axis=0)
        Y = np.concatenate([y_a, y_b], axis=0)
        A = np.concatenate([a_a, a_b], axis=0)
        target_recon = X[:, :Dxc + Dxp]
        out, _ = self.forward_case(X)
        fid = np.mean((out["recon"] - target_recon) ** 2)
        sur = np.mean((out["gated"] - Y) ** 2)
        gate = np.mean((out["u"].ravel() - A) ** 2)

        core_a, _ = self.enc_core.forward(x_a)
        core_b, _ = self.enc_core.forward(x_b)
        inv = np.mean((core_a - core_b) ** 2)

        n = xp_a.shape[0]
        labels = np.concatenate([np.ones(n), np.zeros(x1_fs.shape[0]), np.zeros(x_r1.shape[0])])
        X1 = np.concatenate([xp_a, x1_fs, x_r1], axis=0)
        X2 = np.concatenate([xp_b, x2_fs, x_r2], axis=0)
        c1, _ = self.enc_core.forward(X1)
        c2, _ = self.enc_core.forward(X2)
        feat = np.concatenate([c1, c2, np.abs(c1 - c2)], axis=1)
        logit, _ = self.recognition_head.forward(feat)
        prob = 1.0 / (1.0 + np.exp(-logit))
        eps = 1e-9
        rec = -np.mean(labels.reshape(-1, 1) * np.log(prob + eps)
                        + (1 - labels.reshape(-1, 1)) * np.log(1 - prob + eps))

        total = (weights["fid"] * fid + weights["sur"] * sur + weights["gate"] * gate
                 + weights["inv"] * inv + weights["rec"] * rec)
        return float(total)


WEIGHTS = dict(fid=1.0, sur=1.0, gate=1.5, inv=1.0, rec=1.0)


# =============================================================================
# SECTION 5: FINITE-DIFFERENCE GRADIENT CHECK  (mandatory)
# =============================================================================

def gradient_check(seed: int = 42, n_checks_per_module: int = 3, eps: float = 1e-5,
                    tol: float = 3e-2) -> bool:
    """Verifies the fully manual backward pass above against a numerical
    finite-difference gradient, on a small FIXED batch (frozen random state,
    reused for every perturbation) so the comparison is apples-to-apples.
    Checks a handful of weights from every module that receives gradient in
    train_step(). Uses a generous but standard relative-error tolerance
    appropriate for tanh/sigmoid networks evaluated in float64.
    """
    rng_data = np.random.default_rng(seed)
    gen = CaseGenerator(seed=seed)
    model = MarieMind(seed=seed + 1)

    n_case = 6
    core = gen.sample_cases(n_case)
    (x_a, y_a, a_a, _), (x_b, y_b, a_b, _) = gen.transformation_pair(core)
    x1_fs, x2_fs = gen.false_semblance_pairs(4)
    core_p = gen.sample_cases(4)
    (xp_a, _, _, _), (xp_b, _, _, _) = gen.transformation_pair(core_p)
    core_r1, core_r2 = gen.sample_cases(4), gen.sample_cases(4)
    x_r1, _, _, _ = gen.sample_form(core_r1)
    x_r2, _, _, _ = gen.sample_form(core_r2)

    gen_state = ((x_a, y_a, a_a), (x_b, y_b, a_b), (x1_fs, x2_fs), (xp_a, xp_b), (x_r1, x_r2))

    def loss_fn():
        return model.compute_total_loss(gen_state, WEIGHTS)

    # --- analytic gradients: run one training-step-shaped backward pass,
    # but WITHOUT calling the optimizer, so parameters are untouched and we
    # can read out the accumulated grads directly. ---
    X = np.concatenate([x_a, x_b], axis=0)
    Y = np.concatenate([y_a, y_b], axis=0)
    A = np.concatenate([a_a, a_b], axis=0)
    target_recon = X[:, :Dxc + Dxp]
    out, cache = model.forward_case(X)
    grads_main, dCore, dPeriph, _ = model.backward_case(
        X, out, cache, target_recon, Y, A,
        w_fid=WEIGHTS["fid"], w_sur=WEIGHTS["sur"], w_gate=WEIGHTS["gate"])
    _, g_core_main = model.enc_core.backward(dCore, cache["c_core"])
    _, g_periph_main = model.enc_periph.backward(dPeriph, cache["c_periph"])
    g_core_inv, _ = model.invariance_branch(x_a, x_b, w_inv=WEIGHTS["inv"])

    n = xp_a.shape[0]
    labels = np.concatenate([np.ones(n), np.zeros(x1_fs.shape[0]), np.zeros(x_r1.shape[0])])
    X1 = np.concatenate([xp_a, x1_fs, x_r1], axis=0)
    X2 = np.concatenate([xp_b, x2_fs, x_r2], axis=0)
    g_head_rec, g_core_rec, _, _ = model.recognition_branch(X1, X2, labels, w_rec=WEIGHTS["rec"])

    g_core_total = [a + b + c for a, b, c in zip(g_core_main, g_core_inv, g_core_rec)]
    analytic = {
        "enc_core": g_core_total,
        "enc_periph": g_periph_main,
        "gloss_decoder": grads_main["gloss_decoder"],
        "surplus_decoder": grads_main["surplus_decoder"],
        "baseline": grads_main["baseline"],
        "gate_head": grads_main["gate_head"],
        "recognition_head": g_head_rec,
    }

    all_ok = True
    report = []
    for name, module in model._modules.items():
        params = module.params()
        grads = analytic[name]
        idxs = rng_data.integers(0, 1_000_000, size=n_checks_per_module)
        for k in range(n_checks_per_module):
            p_idx = k % len(params)
            p = params[p_idx]
            g = grads[p_idx]
            flat = p.reshape(-1)
            gflat = g.reshape(-1)
            j = int(idxs[k]) % flat.size

            orig = flat[j]
            flat[j] = orig + eps
            loss_plus = loss_fn()
            flat[j] = orig - eps
            loss_minus = loss_fn()
            flat[j] = orig  # restore

            numeric = (loss_plus - loss_minus) / (2 * eps)
            analytic_val = gflat[j]
            denom = max(abs(numeric), abs(analytic_val), 1e-8)
            rel_err = abs(numeric - analytic_val) / denom
            ok = rel_err < tol
            all_ok &= ok
            report.append((name, j, numeric, analytic_val, rel_err, ok))

    print("Gradient check (finite-difference vs. analytic backward):")
    print(f"{'module':<18}{'idx':>7}{'numeric':>14}{'analytic':>14}{'rel_err':>12}  ok")
    for name, j, num, ana, err, ok in report:
        print(f"{name:<18}{j:>7}{num:>14.6f}{ana:>14.6f}{err:>12.6f}  {'PASS' if ok else 'FAIL'}")
    print(f"-> {'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED'}\n")
    return all_ok


# =============================================================================
# SECTION 6: TRAINING LOOP
# =============================================================================

def train(steps: int = 2500, seed: int = 7, verbose_every: int = 250):
    gen = CaseGenerator(seed=seed)
    model = MarieMind(seed=seed + 1)
    history = []
    for step in range(1, steps + 1):
        logs = model.train_step(gen, batch_cases=48, rec_pairs=24, weights=WEIGHTS)
        history.append(logs)
        if verbose_every and step % verbose_every == 0:
            print(f"step {step:5d}  total {logs['total']:.4f}  "
                  f"fid {logs['fid']:.4f}  sur {logs['sur']:.4f}  "
                  f"gate {logs['gate']:.4f}  inv {logs['inv']:.4f}  rec {logs['rec']:.4f}")
    return model, gen, history


# =============================================================================
# SECTION 7: SELF-TESTS (tied directly to the chapter's three claims)
# =============================================================================

def evaluate_fidelity(model: MarieMind, gen: CaseGenerator, n_cases: int = 200):
    """Held-out check: draws FRESH cases from the SAME world (the same
    generator instance used in training, so the underlying core/periphery
    -> letter projections match what the model was trained against) --
    never a newly re-seeded CaseGenerator, which would instantiate an
    unrelated random world and make generalization meaningless to test."""
    g = gen
    core = g.sample_cases(n_cases)
    x, y, a, _ = g.sample_form(core)
    out, _ = model.forward_case(x)
    target_recon = x[:, :Dxc + Dxp]
    mse = float(np.mean((out["recon"] - target_recon) ** 2))
    return mse


def evaluate_gate_calibration(model: MarieMind, gen: CaseGenerator, n_cases: int = 300):
    """Claim: the learned ambiguity gate u should track true injected
    ambiguity `a` -- surplus should be licensed where the letter is
    genuinely obscure, not indiscriminately."""
    g = gen
    core = g.sample_cases(n_cases)
    x, y, a, _ = g.sample_form(core)
    out, _ = model.forward_case(x)
    u = out["u"].ravel()
    corr = float(np.corrcoef(u, a)[0, 1])
    return corr


def evaluate_core_invariance(model: MarieMind, gen: CaseGenerator, n_cases: int = 200):
    """Claim: CORE should be substantially more stable across a
    transformation of the SAME case than across two DIFFERENT cases
    (the Bisclavret test)."""
    g = gen
    core = g.sample_cases(n_cases)
    (x_a, _, _, _), (x_b, _, _, _) = g.transformation_pair(core)
    core_a, _ = model.enc_core.forward(x_a)
    core_b, _ = model.enc_core.forward(x_b)
    periph_a, _ = model.enc_periph.forward(x_a)
    periph_b, _ = model.enc_periph.forward(x_b)

    same_case_core_dist = float(np.mean(np.linalg.norm(core_a - core_b, axis=1)))
    same_case_periph_dist = float(np.mean(np.linalg.norm(periph_a - periph_b, axis=1)))

    # Shuffle to build different-case pairs from the same pool of forms.
    perm = g.rng.permutation(n_cases)
    diff_case_core_dist = float(np.mean(np.linalg.norm(core_a - core_b[perm], axis=1)))

    return dict(same_case_core_dist=same_case_core_dist,
                diff_case_core_dist=diff_case_core_dist,
                same_case_periph_dist=same_case_periph_dist)


def evaluate_recognition(model: MarieMind, gen: CaseGenerator, n_pairs: int = 300):
    """Claim: the recognition head, using CORE alone, should correctly
    reject 'false semblance' pairs (different case, matched periphery) that
    a periphery-only judge would be fooled by (the Lanval test)."""
    g = gen

    core_pos = g.sample_cases(n_pairs)
    (xp_a, _, _, _), (xp_b, _, _, _) = g.transformation_pair(core_pos)
    x1_fs, x2_fs = g.false_semblance_pairs(n_pairs)
    core_r1, core_r2 = g.sample_cases(n_pairs), g.sample_cases(n_pairs)
    x_r1, _, _, _ = g.sample_form(core_r1)
    x_r2, _, _, _ = g.sample_form(core_r2)

    def rec_predict(X1, X2):
        c1, _ = model.enc_core.forward(X1)
        c2, _ = model.enc_core.forward(X2)
        feat = np.concatenate([c1, c2, np.abs(c1 - c2)], axis=1)
        logit, _ = model.recognition_head.forward(feat)
        prob = 1.0 / (1.0 + np.exp(-logit))
        return prob.ravel(), c1, c2

    prob_pos, _, _ = rec_predict(xp_a, xp_b)
    prob_fs, c1_fs, c2_fs = rec_predict(x1_fs, x2_fs)
    prob_rand, _, _ = rec_predict(x_r1, x_r2)

    acc = np.mean(
        np.concatenate([prob_pos > 0.5, prob_fs <= 0.5, prob_rand <= 0.5]).astype(float)
    )

    # Periphery-only baseline judge: cosine similarity of PERIPHERY
    # embeddings thresholded at 0 -- this baseline SHOULD be fooled by the
    # false-semblance pairs, since their periphery was engineered to match.
    def periph_cos(X1, X2):
        p1, _ = model.enc_periph.forward(X1)
        p2, _ = model.enc_periph.forward(X2)
        num = np.sum(p1 * p2, axis=1)
        den = np.linalg.norm(p1, axis=1) * np.linalg.norm(p2, axis=1) + 1e-9
        return num / den

    periph_cos_fs = periph_cos(x1_fs, x2_fs)
    periph_would_accept_fs = float(np.mean(periph_cos_fs > 0.3))  # fooled rate
    core_correctly_rejects_fs = float(np.mean(prob_fs <= 0.5))

    return dict(overall_accuracy=float(acc),
                periphery_judge_fooled_rate_on_false_semblance=periph_would_accept_fs,
                core_recognition_correct_reject_rate_on_false_semblance=core_correctly_rejects_fs)


def run_self_tests(model: MarieMind, gen: CaseGenerator):
    print("=" * 78)
    print("SELF-TESTS")
    print("=" * 78)

    fid_mse = evaluate_fidelity(model, gen)
    print(f"[1] Gloss fidelity (reconstruction MSE on held-out cases): {fid_mse:.4f}")
    fid_ok = fid_mse < 0.08
    print(f"    -> {'PASS' if fid_ok else 'FAIL'} (threshold 0.08)\n")

    gate_corr = evaluate_gate_calibration(model, gen)
    print(f"[2] Surplus-gate calibration: corr(u_predicted, true ambiguity) = {gate_corr:.3f}")
    gate_ok = gate_corr > 0.5
    print(f"    -> {'PASS' if gate_ok else 'FAIL'} (threshold 0.50) "
          f"-- surplus is licensed roughly where the letter is genuinely obscure.\n")

    inv = evaluate_core_invariance(model, gen)
    print("[3] Bisclavret test -- CORE stability under transformation:")
    print(f"    same-case CORE distance      : {inv['same_case_core_dist']:.4f}")
    print(f"    different-case CORE distance : {inv['diff_case_core_dist']:.4f}")
    print(f"    same-case PERIPHERY distance : {inv['same_case_periph_dist']:.4f}  (expected to stay large)")
    inv_ok = inv["same_case_core_dist"] < 0.6 * inv["diff_case_core_dist"]
    print(f"    -> {'PASS' if inv_ok else 'FAIL'} (same-case core distance should be "
          f"well below different-case core distance, while periphery keeps varying)\n")

    rec = evaluate_recognition(model, gen)
    print("[4] Lanval test -- recognition beneath false semblance:")
    print(f"    overall recognition accuracy                         : {rec['overall_accuracy']:.3f}")
    print(f"    periphery-only judge fooled rate on false-semblance  : "
          f"{rec['periphery_judge_fooled_rate_on_false_semblance']:.3f}  (expected HIGH)")
    print(f"    core-based recognizer correct-reject on false-semblance: "
          f"{rec['core_recognition_correct_reject_rate_on_false_semblance']:.3f}  (expected HIGH)")
    rec_ok = (rec["overall_accuracy"] > 0.70
              and rec["core_recognition_correct_reject_rate_on_false_semblance"] > 0.60)
    print(f"    -> {'PASS' if rec_ok else 'FAIL'} (thresholds 0.70 / 0.60): the CORE-based "
          f"recognizer should see through a false semblance that a surface/periphery judge "
          f"is measurably fooled by.\n")

    all_pass = fid_ok and gate_ok and inv_ok and rec_ok
    print("=" * 78)
    print(f"OVERALL: {'ALL SELF-TESTS PASSED' if all_pass else 'AT LEAST ONE SELF-TEST FAILED'}")
    print("=" * 78)
    return all_pass


# =============================================================================
# SECTION 8: DEMONSTRATION ENTRY POINT
# =============================================================================

def main():
    print("#" * 78)
    print("# Chapter 250 -- Marie de France: gloss / gated surplus / recognition")
    print("#" * 78)
    print()

    ok = gradient_check()
    if not ok:
        print("WARNING: gradient check reported at least one failing parameter; "
              "inspect the report above before trusting downstream training.\n")

    print("Training MarieMind (core/periphery encoders, gloss decoder, "
          "gated surplus decoder, recognition head)...\n")
    model, gen, history = train(steps=2500, verbose_every=250)
    print()

    all_pass = run_self_tests(model, gen)
    return ok, all_pass, history


if __name__ == "__main__":
    main()
