#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0289 · Peter Lombard (c. 1096/1100-1160 CE), "Magister Sententiarum"
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 15, Minds 281-300 Available on Amazon https://www.amazon.com/dp/B0HMQ7TVJM
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0289_peter_lombard_1100 - Peter Lombard (c. 1096/1100-1160 CE), "Magister Sententiarum"
#  Figure 0289 -- Peter Lombard (c. 1096/1100-1160 CE), "Magister Sententiarum"
#
#  WHAT THIS FILE IS
#  ------------------
#  Peter Lombard's originality was never a doctrine. He proposed almost
#  nothing of his own. His originality was a *method* for making a mind
#  bigger than any one mind: take the sententiae -- the authoritative
#  statements -- of Scripture and the Fathers on a question, place them side
#  by side, and either (a) find the distinctio, the hidden difference of
#  sense or reference that lets two apparently opposed authorities both be
#  true, or (b) when no such distinction survives honest search, say so
#  plainly (his own recurring formula is "non... determino", "I do not
#  decide this") instead of forcing a resolution the evidence will not bear.
#  He then filed the question in one of four books -- Trinity, Creation,
#  Incarnation, Sacraments -- because *where a question is taught* is itself
#  part of teaching it well.
#
#  This file implements that method as a real, trainable model. Nothing
#  here is a chatbot wearing a costume: every layer below is a small,
#  fully-derived neural computation in raw NumPy, every gradient is checked
#  against finite differences before the model is allowed to train, and the
#  training loop runs on synthetic data generated to have the exact
#  structure Lombard's method is built to handle -- some questions where a
#  genuine distinction reconciles conflicting authorities, and some where no
#  distinction exists and the honest output is to abstain.
#
#  ARCHITECTURE SUMMARY  (see the chapter text for the full discussion)
#  ----------------------------------------------------------------------
#  1. AuctoritasBank    -- a learned embedding table of "authority trust":
#                          one row per named source (Augustine, Jerome,
#                          Ambrose, Hilary, a synodal decree, an anonymous
#                          gloss...). This is Lombard's own practice made
#                          explicit: not all sententiae carry equal weight.
#  2. DistinctioLayer   -- a small two-layer network that maps
#                          (authority-claim - question) into a 1-D "sense
#                          coordinate". If a genuine distinction exists, the
#                          coordinate separates the authorities who affirm
#                          the question from those who deny it -- i.e. it
#                          discovers they were answering *different*
#                          questions, which is exactly what a scholastic
#                          distinctio claims.
#  3. AporiaHead        -- a calibrated classifier that looks at how well
#                          the DistinctioLayer *could* separate the
#                          authorities and predicts whether this question is
#                          the rare kind that should be left open. This is
#                          the "non determino" reflex as a first-class,
#                          trained output, not a fallback when confidence is
#                          low -- productive doubt is a target, not a
#                          failure mode.
#  4. CurriculumRouter  -- a softmax classifier that files a question into
#                          one of Lombard's four books by the shape of its
#                          concept alone. This is his other genuine
#                          contribution: not what to conclude, but where a
#                          question belongs in a teachable order.
#  5. LombardNet         -- wires the four pieces together, computes a
#                          composite loss (weighted hinge + two cross
#                          entropies + L2), and back-propagates all of it
#                          by hand.
#
#  Everything is pure NumPy. There is no autodiff library and no ML
#  framework. Run this file directly:
#
#      python3 0289_Neuron.py            # gradient check + train + self-tests + demo
#      python3 0289_Neuron.py --quick    # smaller run, same checks, faster
#
# =============================================================================

from __future__ import annotations

import sys
import time
import numpy as np


# =============================================================================
# SECTION A -- LOW-LEVEL DIFFERENTIABLE PRIMITIVES  (hand-derived backprop)
# =============================================================================
# Every primitive below returns (output, cache) on the forward pass and
# consumes (grad_output, cache) on the backward pass. This is the same
# contract a hand-rolled autodiff engine would use; we simply never build
# the engine, we call the two halves ourselves in the right order.

def linear_forward(x: np.ndarray, W: np.ndarray, b: np.ndarray):
    """y = x @ W + b.   x: (N, in)   W: (in, out)   b: (out,)"""
    y = x @ W + b
    cache = (x, W)
    return y, cache


def linear_backward(dy: np.ndarray, cache):
    x, W = cache
    dx = dy @ W.T
    dW = x.T @ dy
    db = dy.sum(axis=0)
    return dx, dW, db


def tanh_forward(x: np.ndarray):
    y = np.tanh(x)
    return y, y  # cache is the output itself, dtanh(x) = 1 - tanh(x)^2


def tanh_backward(dy: np.ndarray, cache_y: np.ndarray):
    return dy * (1.0 - cache_y ** 2)


def sigmoid(x: np.ndarray) -> np.ndarray:
    # numerically stable logistic sigmoid
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def bce_loss_and_grad(p: np.ndarray, y: np.ndarray, eps: float = 1e-9):
    """Binary cross entropy on probabilities p (already sigmoided), target y in {0,1}.
    Returns (mean_loss, dL/dp)."""
    p_clipped = np.clip(p, eps, 1.0 - eps)
    loss = -np.mean(y * np.log(p_clipped) + (1.0 - y) * np.log(1.0 - p_clipped))
    n = p.shape[0]
    dp = (-(y / p_clipped) + (1.0 - y) / (1.0 - p_clipped)) / n
    return loss, dp


def sigmoid_backward_from_logit(dp: np.ndarray, p: np.ndarray):
    """Given dL/dp and p = sigmoid(z), return dL/dz."""
    return dp * p * (1.0 - p)


def softmax_forward(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / ez.sum(axis=1, keepdims=True)


def softmax_xent_loss_and_grad(logits: np.ndarray, labels: np.ndarray):
    """logits: (N, C).  labels: (N,) int class ids.  Returns (mean_loss, dL/dlogits)."""
    n = logits.shape[0]
    p = softmax_forward(logits)
    correct_p = p[np.arange(n), labels]
    loss = -np.mean(np.log(np.clip(correct_p, 1e-12, 1.0)))
    dz = p.copy()
    dz[np.arange(n), labels] -= 1.0
    dz /= n
    return loss, dz


# =============================================================================
# SECTION B -- SYNTHETIC "QUAESTIO" DATA GENERATOR
# =============================================================================
# We cannot hand LombardNet twelfth-century manuscripts, so we generate data
# with the *structural* property Lombard's method targets: for "resolvable"
# questions, a hidden linear distinction genuinely separates the authorities
# by stance; for "aporia" questions, stances are deliberately scattered so
# that no linear distinction of the term can separate them -- the honest
# output is abstention, not a forced answer. The router labels come from
# four well-separated Gaussian clusters standing in for the four books.

AUTHORITY_NAMES = [
    "Scripture", "Augustine", "Ambrose", "Jerome", "Hilary of Poitiers",
    "Gregory the Great", "John of Damascus", "Isidore", "a conciliar canon",
    "an anonymous gloss", "Origen (suspect)", "a private opinion",
]
N_AUTHORITIES = len(AUTHORITY_NAMES)
# Rough hand-set "seniority" priors used only to *generate* plausible data;
# the network never sees these numbers and must recover trust structure
# purely from how predictive each authority's stance turns out to be.
_TRUE_SENIORITY = np.array(
    [0.97, 0.92, 0.85, 0.85, 0.75, 0.80, 0.78, 0.70, 0.65, 0.35, 0.30, 0.20]
)

BOOK_NAMES = ["I. Trinity & God", "II. Creation & the Soul",
              "III. Incarnation & Redemption", "IV. Sacraments & the Last Things"]
N_BOOKS = len(BOOK_NAMES)
EMBED_DIM = 12
HIDDEN_DIM = 8


def _book_centers(rng: np.random.Generator) -> np.ndarray:
    centers = rng.normal(0.0, 1.0, size=(N_BOOKS, EMBED_DIM))
    centers *= 3.0 / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-8)
    return centers


def generate_dataset(n_questions: int, k_authorities: int, rng: np.random.Generator,
                      book_centers: np.ndarray, aporia_rate: float = 0.35):
    """
    Builds one synthetic "disputed question" dataset.

    Returns a dict of arrays:
      q            (N, D)      question concept embeddings
      book_label   (N,)        which of the 4 books the question belongs to
      auth_id      (N, k)      integer ids into AUTHORITY_NAMES
      a_embed      (N, k, D)   each authority's claim embedding
      stance       (N, k)      +1 affirm / -1 deny
      is_aporia    (N,)        1.0 if the question is genuinely irreducible
      true_distinction (N, D) the hidden distinguishing direction used to
                               generate resolvable cases (ground truth, used
                               only for diagnostics, never fed to the model)
    """
    D = EMBED_DIM
    book_label = rng.integers(0, N_BOOKS, size=n_questions)
    q = book_centers[book_label] + rng.normal(0, 0.35, size=(n_questions, D))

    is_aporia = (rng.random(n_questions) < aporia_rate).astype(np.float64)

    auth_id = rng.integers(0, N_AUTHORITIES, size=(n_questions, k_authorities))
    a_embed = np.zeros((n_questions, k_authorities, D))
    stance = np.zeros((n_questions, k_authorities))
    true_distinction = rng.normal(0, 1, size=(n_questions, D))
    true_distinction /= (np.linalg.norm(true_distinction, axis=1, keepdims=True) + 1e-8)

    for i in range(n_questions):
        dvec = true_distinction[i]
        for j in range(k_authorities):
            seniority = _TRUE_SENIORITY[auth_id[i, j]]
            if is_aporia[i] < 0.5:
                # RESOLVABLE: stance correlates with a genuine hidden
                # distinction; higher-seniority authorities follow it more
                # faithfully (noisier for weak/spurious attributions).
                side = 1.0 if rng.random() < 0.5 else -1.0
                noise_scale = (1.0 - seniority) * 1.4 + 0.15
                offset = side * (0.9 + 0.4 * rng.random()) * dvec
                offset += rng.normal(0, noise_scale, size=D)
                stance[i, j] = side
            else:
                # APORIA: stance is independent of any single direction --
                # authorities of similar seniority split roughly evenly with
                # no linearly recoverable structure, by construction.
                side = 1.0 if rng.random() < 0.5 else -1.0
                random_dir = rng.normal(0, 1, size=D)
                random_dir /= (np.linalg.norm(random_dir) + 1e-8)
                offset = side * 0.7 * random_dir + rng.normal(0, 0.9, size=D)
                stance[i, j] = side
            a_embed[i, j] = q[i] + offset

    return {
        "q": q, "book_label": book_label, "auth_id": auth_id, "a_embed": a_embed,
        "stance": stance, "is_aporia": is_aporia, "true_distinction": true_distinction,
    }


# =============================================================================
# SECTION C -- LOMBARDNET
# =============================================================================

class LombardNet:
    """
    Distinction-learning sententiae resolver with a four-book curriculum
    router. All parameters live in self.params (a flat dict of numpy
    arrays) so gradient checking and Adam can iterate over them generically.
    """

    def __init__(self, embed_dim: int = EMBED_DIM, hidden_dim: int = HIDDEN_DIM,
                 n_authorities: int = N_AUTHORITIES, n_books: int = N_BOOKS,
                 seed: int = 0):
        self.D = embed_dim
        self.H = hidden_dim
        self.n_authorities = n_authorities
        self.n_books = n_books
        rng = np.random.default_rng(seed)

        def xavier(fan_in, fan_out):
            bound = np.sqrt(6.0 / (fan_in + fan_out))
            return rng.uniform(-bound, bound, size=(fan_in, fan_out))

        self.params = {
            # AuctoritasBank: one learned raw trust-logit per named source.
            "auctoritas_logit": rng.normal(0, 0.1, size=(n_authorities,)),

            # DistinctioLayer: (a_embed - q) -> hidden -> 1-D sense coordinate.
            "W1": xavier(self.D, self.H),
            "b1": np.zeros(self.H),
            "W2": xavier(self.H, 1),
            "b2": np.zeros(1),

            # AporiaHead: [mean(c), std(c), weighted_agreement, |q|] -> logit.
            "Wa": xavier(4, 1),
            "ba": np.zeros(1),

            # CurriculumRouter: q -> book logits.
            "Wr": xavier(self.D, n_books),
            "br": np.zeros(n_books),
        }

    # ---- forward -----------------------------------------------------
    def forward(self, batch: dict):
        """
        Full forward pass. Returns (outputs, cache) where outputs holds
        everything needed for loss computation and cache holds everything
        needed for backward().
        """
        P = self.params
        q = batch["q"]                      # (N, D)
        a_embed = batch["a_embed"]           # (N, k, D)
        auth_id = batch["auth_id"]           # (N, k) int
        N, k, D = a_embed.shape

        # --- AuctoritasBank: trust_i = sigmoid(logit[auth_id]) ---------
        auth_logit_flat = P["auctoritas_logit"][auth_id.reshape(-1)]     # (N*k,)
        trust_flat = sigmoid(auth_logit_flat)
        trust = trust_flat.reshape(N, k)

        # --- DistinctioLayer --------------------------------------------
        delta = a_embed - q[:, None, :]              # (N, k, D)
        delta_flat = delta.reshape(N * k, D)
        h_lin, cache_h_lin = linear_forward(delta_flat, P["W1"], P["b1"])
        h, cache_h_tanh = tanh_forward(h_lin)
        c_lin, cache_c_lin = linear_forward(h, P["W2"], P["b2"])          # (N*k, 1)
        c = c_lin.reshape(N, k)

        # --- Per-question group separation (the real "distinctio" signal) --
        # A found distinction has no canonical sign -- what matters is
        # whether a threshold splits the trust-weighted affirming
        # authorities from the trust-weighted denying ones, PER QUESTION.
        # (Each question's true hidden distinction direction is its own;
        # there is no single global direction shared across questions, so
        # any loss that demands a fixed global sign convention is
        # mis-specified. This computes the sign-free, per-question gap.)
        stance = batch["stance"]
        pos_mask = (stance > 0).astype(np.float64)
        neg_mask = (stance < 0).astype(np.float64)
        Wp = (trust * pos_mask).sum(axis=1) + 1e-6      # (N,)
        Wn = (trust * neg_mask).sum(axis=1) + 1e-6      # (N,)
        mean_pos = (trust * pos_mask * c).sum(axis=1) / Wp
        mean_neg = (trust * neg_mask * c).sum(axis=1) / Wn
        diff = mean_pos - mean_neg                       # (N,)
        abs_diff = np.abs(diff)

        # --- AporiaHead features -----------------------------------------
        c_mean = c.mean(axis=1)                        # (N,)
        c_std = c.std(axis=1) + 1e-6                    # (N,)
        q_norm = np.linalg.norm(q, axis=1)
        aporia_feat = np.stack([c_mean, c_std, abs_diff, q_norm], axis=1)  # (N,4)
        aporia_logit, cache_aporia_lin = linear_forward(aporia_feat, P["Wa"], P["ba"])
        aporia_logit = aporia_logit[:, 0]
        aporia_p = sigmoid(aporia_logit)

        # --- CurriculumRouter ---------------------------------------------
        book_logits, cache_router = linear_forward(q, P["Wr"], P["br"])   # (N, n_books)

        outputs = {
            "trust": trust, "c": c, "aporia_p": aporia_p,
            "aporia_logit": aporia_logit, "book_logits": book_logits,
            "group_diff": diff, "group_diff_abs": abs_diff,
        }
        cache = {
            "N": N, "k": k, "D": D, "auth_id": auth_id,
            "trust_flat": trust_flat, "delta_flat": delta_flat,
            "cache_h_lin": cache_h_lin, "cache_h_tanh": cache_h_tanh, "h": h,
            "cache_c_lin": cache_c_lin, "c": c,
            "aporia_feat": aporia_feat, "cache_aporia_lin": cache_aporia_lin,
            "aporia_p": aporia_p, "q": q, "cache_router": cache_router,
            "stance": batch["stance"], "trust": trust,
            "pos_mask": pos_mask, "neg_mask": neg_mask, "Wp": Wp, "Wn": Wn,
            "mean_pos": mean_pos, "mean_neg": mean_neg, "diff": diff,
        }
        return outputs, cache

    # ---- loss ----------------------------------------------------------
    def compute_loss(self, batch: dict, outputs: dict,
                      w_sep: float = 1.0, w_aporia: float = 1.0,
                      w_router: float = 1.0, w_l2: float = 1e-4, margin: float = 1.0):
        N = outputs["c"].shape[0]
        abs_diff = outputs["group_diff_abs"]
        diff = outputs["group_diff"]

        # Sign-free per-question separation hinge: penalize questions whose
        # achieved trust-weighted gap between the affirming and denying
        # authorities falls short of the margin, regardless of which side
        # ended up positive (there is no canonical sign for a distinction).
        violation = np.maximum(0.0, margin - abs_diff)
        loss_sep = violation.mean()
        active = (violation > 0).astype(np.float64)
        # d(loss_sep)/d(diff) = -sign(diff)*active/N  (subgradient at 0 taken as 0)
        d_diff_from_sep = -(np.sign(diff) * active) / N

        loss_aporia, dp_aporia = bce_loss_and_grad(outputs["aporia_p"], batch["is_aporia"])
        loss_router, dz_router = softmax_xent_loss_and_grad(outputs["book_logits"], batch["book_label"])

        l2 = 0.0
        for name in ("W1", "W2", "Wa", "Wr"):
            l2 += np.sum(self.params[name] ** 2)
        loss_l2 = w_l2 * l2

        total = w_sep * loss_sep + w_aporia * loss_aporia + w_router * loss_router + loss_l2
        parts = {"sep": loss_sep, "aporia": loss_aporia, "router": loss_router, "l2": loss_l2, "total": total}
        grad_seed = {
            "d_diff_from_sep": d_diff_from_sep * w_sep,
            "dp_aporia": dp_aporia * w_aporia,
            "dz_router": dz_router * w_router,
            "w_l2": w_l2,
        }
        return parts, grad_seed

    # ---- backward --------------------------------------------------------
    def backward(self, cache: dict, grad_seed: dict):
        P = self.params
        grads = {k_: np.zeros_like(v) for k_, v in P.items()}
        N, k, D = cache["N"], cache["k"], cache["D"]

        # ---- Router branch: dz_router -> Wr, br, and into q -------------
        dz_router = grad_seed["dz_router"]                      # (N, n_books)
        dq_router, dWr, dbr = linear_backward(dz_router, cache["cache_router"])
        grads["Wr"] += dWr
        grads["br"] += dbr

        # ---- Aporia branch: dp_aporia -> logit -> Wa, ba -> features ----
        dp_aporia = grad_seed["dp_aporia"]                       # (N,)
        d_aporia_logit = sigmoid_backward_from_logit(dp_aporia, cache["aporia_p"])  # (N,)
        d_aporia_feat, dWa, dba = linear_backward(d_aporia_logit[:, None], cache["cache_aporia_lin"])
        grads["Wa"] += dWa
        grads["ba"] += dba
        # aporia_feat columns: [c_mean, c_std, group_diff_abs, q_norm]
        d_c_mean = d_aporia_feat[:, 0]
        d_c_std = d_aporia_feat[:, 1]
        d_abs_diff_from_aporia = d_aporia_feat[:, 2]
        d_q_norm = d_aporia_feat[:, 3]

        c = cache["c"]
        c_mean = c.mean(axis=1)
        dc_from_mean = np.repeat((d_c_mean / k)[:, None], k, axis=1)
        std = c.std(axis=1) + 1e-6
        dc_from_std = (d_c_std / (k * std))[:, None] * (c - c_mean[:, None])

        # ---- group-separation branch (diff = mean_pos - mean_neg) -------
        # d(loss_total)/d(abs_diff) has two sources: the direct hinge term
        # (already converted to a d/d(diff) seed in compute_loss) and the
        # AporiaHead's use of abs_diff as a feature (d_abs_diff_from_aporia,
        # w.r.t. abs_diff, needs the sign(diff) chain rule to become d/d(diff)).
        diff = cache["diff"]
        d_diff_from_aporia = d_abs_diff_from_aporia * np.sign(diff)
        d_diff_total = grad_seed["d_diff_from_sep"] + d_diff_from_aporia   # (N,)

        pos_mask, neg_mask = cache["pos_mask"], cache["neg_mask"]
        Wp, Wn = cache["Wp"], cache["Wn"]
        trust = cache["trust"]
        mean_pos, mean_neg = cache["mean_pos"], cache["mean_neg"]

        # d(diff_i)/d(c_ij): weighted-mean derivative w.r.t. its own element
        dc_from_group = (d_diff_total[:, None] *
                          (pos_mask * trust / Wp[:, None] - neg_mask * trust / Wn[:, None]))

        dc_total = dc_from_mean + dc_from_std + dc_from_group

        # ---- back through DistinctioLayer: c -> W2,b2 -> h -> W1,b1 -> delta ----
        dc_lin = dc_total.reshape(N * k, 1)
        dh, dW2, db2 = linear_backward(dc_lin, cache["cache_c_lin"])
        grads["W2"] += dW2
        grads["b2"] += db2
        dh_lin = tanh_backward(dh, cache["cache_h_tanh"])
        ddelta_flat, dW1, db1 = linear_backward(dh_lin, cache["cache_h_lin"])
        grads["W1"] += dW1
        grads["b1"] += db1

        ddelta = ddelta_flat.reshape(N, k, D)
        dq_from_delta = -ddelta.sum(axis=1)   # delta = a_embed - q  =>  d/dq = -1
        # (a_embed is data, not a parameter, so we do not propagate into it)

        # ---- trust gradient from the group-separation quotient rule -----
        # d(mean_pos_i)/d(trust_ij) = (c_ij - mean_pos_i)/Wp_i  for pos_mask_ij=1
        # d(mean_neg_i)/d(trust_ij) = (c_ij - mean_neg_i)/Wn_i  for neg_mask_ij=1
        # d(diff_i)/d(trust_ij) = the pos term minus the neg term (only one
        # of the two masks is active per (i,j), the other contributes zero).
        d_trust_from_group = d_diff_total[:, None] * (
            pos_mask * (c - mean_pos[:, None]) / Wp[:, None]
            - neg_mask * (c - mean_neg[:, None]) / Wn[:, None]
        )
        d_trust_flat = d_trust_from_group.reshape(-1)
        d_auth_logit_flat = sigmoid_backward_from_logit(d_trust_flat, cache["trust_flat"])

        auth_id_flat = cache["auth_id"].reshape(-1)
        np.add.at(grads["auctoritas_logit"], auth_id_flat, d_auth_logit_flat)

        # ---- combine the two paths into q: router + distinction branch --
        dq_total = dq_router + dq_from_delta
        q = cache["q"]
        q_norm = np.linalg.norm(q, axis=1) + 1e-9
        dq_from_norm = d_q_norm[:, None] * (q / q_norm[:, None])
        dq_total = dq_total + dq_from_norm
        # q is a per-example input embedding (data), not a trained parameter
        # of this network in the synthetic setup, so dq_total is computed
        # for completeness / potential future use but not accumulated into
        # self.params here.

        return grads

    def add_l2_grad(self, grads: dict, w_l2: float):
        for name in ("W1", "W2", "Wa", "Wr"):
            grads[name] = grads[name] + 2.0 * w_l2 * self.params[name]
        return grads

    # ---- convenience: one full loss+grad call ---------------------------
    def loss_and_grads(self, batch: dict, w_sep=1.0, w_aporia=1.0, w_router=1.0, w_l2=1e-4):
        outputs, cache = self.forward(batch)
        parts, grad_seed = self.compute_loss(batch, outputs, w_sep, w_aporia, w_router, w_l2)
        grads = self.backward(cache, grad_seed)
        grads = self.add_l2_grad(grads, w_l2)
        return parts, grads, outputs


# =============================================================================
# SECTION D -- ADAM OPTIMIZER (from scratch)
# =============================================================================

class Adam:
    def __init__(self, params: dict, lr: float = 0.02, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params: dict, grads: dict):
        self.t += 1
        for k in params:
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            m_hat = self.m[k] / (1 - self.b1 ** self.t)
            v_hat = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# =============================================================================
# SECTION E -- FINITE-DIFFERENCE GRADIENT CHECK  (mandatory, run before training)
# =============================================================================

def numerical_gradient_check(model: LombardNet, batch: dict, eps: float = 1e-5,
                              n_samples_per_param: int = 4, tol: float = 3e-3,
                              verbose: bool = True) -> bool:
    """
    For a handful of random entries in every parameter array, compares the
    analytic gradient produced by model.backward() against a central
    finite-difference estimate of dLoss/dparam. Passes iff every sampled
    entry agrees within `tol` relative error (or both are ~0).
    """
    rng = np.random.default_rng(12345)

    def total_loss(_batch):
        outputs, _ = model.forward(_batch)
        parts, _ = model.compute_loss(_batch, outputs)
        return parts["total"]

    parts, grads, _ = model.loss_and_grads(batch)
    all_ok = True
    report = []

    for name, arr in model.params.items():
        flat = arr.reshape(-1)
        g_flat = grads[name].reshape(-1)
        n_pick = min(n_samples_per_param, flat.size)
        idxs = rng.choice(flat.size, size=n_pick, replace=False)
        for idx in idxs:
            orig = flat[idx]
            flat[idx] = orig + eps
            loss_plus = total_loss(batch)
            flat[idx] = orig - eps
            loss_minus = total_loss(batch)
            flat[idx] = orig  # restore

            numeric_grad = (loss_plus - loss_minus) / (2 * eps)
            analytic_grad = g_flat[idx]
            denom = max(abs(numeric_grad), abs(analytic_grad), 1e-8)
            rel_err = abs(numeric_grad - analytic_grad) / denom
            ok = rel_err < tol or (abs(numeric_grad) < 1e-6 and abs(analytic_grad) < 1e-6)
            all_ok &= ok
            report.append((name, idx, analytic_grad, numeric_grad, rel_err, ok))

    if verbose:
        print(f"{'param':<20}{'idx':>6}{'analytic':>14}{'numeric':>14}{'rel_err':>12}  ok")
        for name, idx, a, n_, r, ok in report:
            flag = "OK" if ok else "FAIL"
            print(f"{name:<20}{idx:>6}{a:>14.6f}{n_:>14.6f}{r:>12.2e}  {flag}")
    return all_ok


# =============================================================================
# SECTION F -- TRAINING LOOP
# =============================================================================

def evaluate(model: LombardNet, batch: dict) -> dict:
    outputs, _ = model.forward(batch)
    aporia_pred = (outputs["aporia_p"] > 0.5).astype(np.float64)
    aporia_acc = float(np.mean(aporia_pred == batch["is_aporia"]))
    book_pred = outputs["book_logits"].argmax(axis=1)
    router_acc = float(np.mean(book_pred == batch["book_label"]))

    # "separation quality" on resolvable-only questions: fraction where the
    # trust-weighted gap between the affirming and denying authority groups
    # met the training margin -- i.e. a genuine distinction was found. This
    # is sign-free by construction (a distinction has no canonical polarity).
    resolvable_mask = batch["is_aporia"] < 0.5
    abs_diff = outputs["group_diff_abs"]
    if resolvable_mask.any():
        sep_acc = float((abs_diff[resolvable_mask] >= 1.0).mean())
    else:
        sep_acc = float("nan")
    return {"aporia_acc": aporia_acc, "router_acc": router_acc, "sep_acc": sep_acc}


def train(model: LombardNet, epochs: int = 400, batch_size: int = 64,
          n_train: int = 2000, n_val: int = 400, k_authorities: int = 6,
          lr: float = 0.03, seed: int = 1, verbose: bool = True):
    rng = np.random.default_rng(seed)
    book_centers = _book_centers(np.random.default_rng(999))  # fixed across runs
    train_data = generate_dataset(n_train, k_authorities, rng, book_centers)
    val_data = generate_dataset(n_val, k_authorities, np.random.default_rng(seed + 1), book_centers)

    opt = Adam(model.params, lr=lr)
    history = []
    n_batches = max(1, n_train // batch_size)

    for epoch in range(epochs):
        perm = rng.permutation(n_train)
        epoch_loss = 0.0
        for bidx in range(n_batches):
            idx = perm[bidx * batch_size:(bidx + 1) * batch_size]
            if idx.size == 0:
                continue
            batch = {
                "q": train_data["q"][idx],
                "book_label": train_data["book_label"][idx],
                "auth_id": train_data["auth_id"][idx],
                "a_embed": train_data["a_embed"][idx],
                "stance": train_data["stance"][idx],
                "is_aporia": train_data["is_aporia"][idx],
            }
            parts, grads, _ = model.loss_and_grads(batch)
            opt.step(model.params, grads)
            epoch_loss += parts["total"] * idx.size
        epoch_loss /= n_train

        if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
            val_metrics = evaluate(model, val_data)
            print(f"  epoch {epoch:4d}  train_loss={epoch_loss:8.4f}  "
                  f"val: aporia_acc={val_metrics['aporia_acc']:.3f}  "
                  f"router_acc={val_metrics['router_acc']:.3f}  "
                  f"sep_acc={val_metrics['sep_acc']:.3f}")
        history.append(epoch_loss)

    final_metrics = evaluate(model, val_data)
    return history, final_metrics, val_data


# =============================================================================
# SECTION G -- THE QUAESTIO METHOD: STRUCTURED INFERENCE
# =============================================================================

def quaestio(model: LombardNet, question_embedding: np.ndarray,
             authority_records: list, aporia_threshold: float = 0.5) -> dict:
    """
    Runs Lombard's method end to end on ONE new question.

    authority_records: list of dicts, each {"name": str, "auth_id": int,
                        "embedding": np.ndarray(D,), "stance": +1/-1}

    Returns a structured verdict: which book the question belongs to, and
    either a resolution (a distinction that separates the authorities, with
    a confidence derived from the learned margin) or an aporia declaration
    ("non determino") with its own calibrated confidence -- mirroring the
    two legitimate endings of a scholastic quaestio.
    """
    k = len(authority_records)
    q = question_embedding.reshape(1, -1)
    a_embed = np.stack([r["embedding"] for r in authority_records]).reshape(1, k, -1)
    auth_id = np.array([[r["auth_id"] for r in authority_records]])
    stance = np.array([[r["stance"] for r in authority_records]], dtype=np.float64)

    batch = {"q": q, "a_embed": a_embed, "auth_id": auth_id, "stance": stance}
    outputs, _ = model.forward(batch)

    book_idx = int(outputs["book_logits"].argmax(axis=1)[0])
    aporia_p = float(outputs["aporia_p"][0])
    c = outputs["c"][0]
    trust = outputs["trust"][0]

    verdict = {
        "book": BOOK_NAMES[book_idx],
        "aporia_probability": aporia_p,
        "per_authority": [
            {"name": authority_records[i]["name"],
             "trust": float(trust[i]),
             "sense_coordinate": float(c[i]),
             "stance": "affirms" if authority_records[i]["stance"] > 0 else "denies"}
            for i in range(k)
        ],
    }
    if aporia_p >= aporia_threshold:
        verdict["resolution"] = "non determino"
        verdict["gloss"] = ("The authorities cannot be reconciled by any distinction "
                             "this method can find in good conscience; the question is "
                             "left open rather than forced.")
    else:
        threshold = float(np.median(c))
        verdict["resolution"] = "distinguo"
        verdict["distinction_threshold"] = threshold
        verdict["gloss"] = ("A distinction in sense reconciles the authorities: those "
                             "above the threshold and those below are answering "
                             "different questions under one term.")
    return verdict


# =============================================================================
# SECTION H -- SELF-TESTS
# =============================================================================

def run_self_tests(quick: bool = False) -> bool:
    print("=" * 78)
    print("SELF-TESTS")
    print("=" * 78)
    ok = True

    # T1: shapes and no-NaN forward pass
    model = LombardNet(seed=0)
    rng = np.random.default_rng(0)
    book_centers = _book_centers(np.random.default_rng(999))
    small = generate_dataset(16, 5, rng, book_centers)
    batch = {k_: small[k_] for k_ in ("q", "book_label", "auth_id", "a_embed", "stance", "is_aporia")}
    outputs, _ = model.forward(batch)
    shapes_ok = (outputs["c"].shape == (16, 5) and outputs["book_logits"].shape == (16, N_BOOKS)
                 and outputs["aporia_p"].shape == (16,))
    nan_ok = all(np.isfinite(v).all() for v in outputs.values())
    print(f"[T1] forward shapes correct: {shapes_ok}; all finite: {nan_ok}")
    ok &= shapes_ok and nan_ok

    # T2: gradient check (mandatory)
    print("[T2] finite-difference gradient check ...")
    gc_batch = {k_: small[k_] for k_ in ("q", "book_label", "auth_id", "a_embed", "stance", "is_aporia")}
    grad_ok = numerical_gradient_check(model, gc_batch, verbose=True)
    print(f"[T2] gradient check passed: {grad_ok}")
    ok &= grad_ok

    # T3: loss decreases over a short training run
    print("[T3] short training run (loss should fall) ...")
    model2 = LombardNet(seed=7)
    epochs = 60 if quick else 150
    history, final_metrics, _ = train(model2, epochs=epochs, n_train=800 if quick else 1500,
                                       n_val=200, k_authorities=6, lr=0.05, seed=3, verbose=False)
    loss_fell = history[-1] < history[0] * 0.6
    print(f"[T3] loss[0]={history[0]:.4f} -> loss[-1]={history[-1]:.4f}  fell_enough={loss_fell}")
    ok &= loss_fell

    # T4: learned metrics beat chance
    print(f"[T4] final val metrics: {final_metrics}")
    router_ok = final_metrics["router_acc"] > 1.0 / N_BOOKS + 0.25
    aporia_ok = final_metrics["aporia_acc"] > 0.60
    sep_ok = (final_metrics["sep_acc"] != final_metrics["sep_acc"]) or final_metrics["sep_acc"] > 0.65
    print(f"[T4] router_acc beats chance: {router_ok}; aporia_acc>0.60: {aporia_ok}; sep_acc>0.65: {sep_ok}")
    ok &= router_ok and aporia_ok and sep_ok

    # T5: quaestio() runs end-to-end and returns a well-formed verdict
    print("[T5] quaestio() structured inference ...")
    q_emb = book_centers[0] + rng.normal(0, 0.2, size=EMBED_DIM)
    records = [
        {"name": "Augustine", "auth_id": 1, "embedding": q_emb + 0.9, "stance": 1},
        {"name": "Hilary of Poitiers", "auth_id": 4, "embedding": q_emb + 0.85, "stance": 1},
        {"name": "a private opinion", "auth_id": 11, "embedding": q_emb - 0.9, "stance": -1},
    ]
    verdict = quaestio(model2, q_emb, records)
    verdict_ok = verdict["resolution"] in ("distinguo", "non determino") and verdict["book"] in BOOK_NAMES
    print(f"[T5] verdict well-formed: {verdict_ok}  -> book={verdict['book']}, "
          f"resolution={verdict['resolution']}, aporia_p={verdict['aporia_probability']:.3f}")
    ok &= verdict_ok

    print("=" * 78)
    print("ALL SELF-TESTS PASSED" if ok else "SOME SELF-TESTS FAILED")
    print("=" * 78)
    return ok


# =============================================================================
# SECTION I -- DEMONSTRATION
# =============================================================================

def demo():
    print("\n" + "=" * 78)
    print("0239 -- LombardNet: a distinction-learning sententiae resolver")
    print("with a four-book curriculum router")
    print("=" * 78)

    rng = np.random.default_rng(42)
    book_centers = _book_centers(np.random.default_rng(999))
    model = LombardNet(seed=42)

    print("\n[1] Finite-difference gradient check on a fresh model:")
    small_batch_data = generate_dataset(20, 5, rng, book_centers)
    small_batch = {k_: small_batch_data[k_] for k_ in
                   ("q", "book_label", "auth_id", "a_embed", "stance", "is_aporia")}
    gc_ok = numerical_gradient_check(model, small_batch, verbose=False)
    print(f"    all sampled gradients within tolerance: {gc_ok}")

    print("\n[2] Training on synthetic disputed questions ...")
    t0 = time.time()
    history, final_metrics, val_data = train(model, epochs=300, n_train=2000, n_val=400,
                                              k_authorities=6, lr=0.03, seed=1, verbose=True)
    print(f"    training completed in {time.time() - t0:.1f}s")
    print(f"    final validation metrics: {final_metrics}")

    print("\n[3] Running the quaestio method on three illustrative disputed questions.")
    print("    (Concept embeddings below are synthetic placeholders standing in for")
    print("     positions in the model's learned concept space, not literal claims")
    print("     about what any authority wrote.)\n")

    examples = [
        ("Can a rational will, uncompelled, choose evil?", 0,
         [("Augustine", 1, 1), ("John of Damascus", 6, 1), ("a private opinion", 11, -1)]),
        ("Does the whole divine essence subsist wholly in each of the three persons?", 0,
         [("Scripture", 0, 1), ("Hilary of Poitiers", 4, 1), ("an anonymous gloss", 9, -1)]),
        ("Once assumed, is Christ's humanity a distinct created 'something'?", 2,
         [("Augustine", 1, 1), ("Ambrose", 2, -1), ("Jerome", 3, 1), ("Origen (suspect)", 10, -1)]),
    ]
    for title, book_idx, auths in examples:
        base = book_centers[book_idx]
        q_emb = base + rng.normal(0, 0.25, size=EMBED_DIM)
        records = []
        for name, aid, stance in auths:
            drift = rng.normal(0, 0.5, size=EMBED_DIM)
            records.append({"name": name, "auth_id": aid,
                             "embedding": q_emb + stance * 0.6 + drift, "stance": stance})
        verdict = quaestio(model, q_emb, records)
        print(f"  Quaestio: {title}")
        print(f"    -> filed under Book {verdict['book']}")
        print(f"    -> aporia probability: {verdict['aporia_probability']:.3f}")
        print(f"    -> resolution: {verdict['resolution']}")
        for rec in verdict["per_authority"]:
            print(f"       - {rec['name']:<20} trust={rec['trust']:.2f}  "
                  f"sense_coord={rec['sense_coordinate']:+.2f}  ({rec['stance']})")
        print(f"    {verdict['gloss']}\n")

    print("=" * 78)
    print("Summary: the network learns (a) a per-source trust weighting from")
    print("outcomes alone, (b) a distinction operator that reconciles genuinely")
    print("reconcilable authorities, (c) a calibrated abstention signal for")
    print("questions that should stay open, and (d) a four-book filing sense --")
    print("four small trained mechanisms standing in for one disciplined habit")
    print("of mind.")
    print("=" * 78 + "\n")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    quick = "--quick" in sys.argv
    tests_ok = run_self_tests(quick=quick)
    if not tests_ok:
        print("\n[FATAL] Self-tests failed; aborting before demo.", file=sys.stderr)
        sys.exit(1)
    demo()
