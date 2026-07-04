#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 chapter_0003_imhotep_-2650.py
 The Imhotep Diagnostic Architecture (IDA)
 A working, trainable AGI base-model in the spirit of Imhotep (c. 2650 BCE),
 the first named physician-architect in recorded history.

Part Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
Tome 1 Minds 1 - 20 Available on Amazon https://www.amazon.com/dp/B0H6F9L324
Resume and Interactive Demos at https://artificiology.com/

================================================================================

WHY THIS DESIGN, AND WHY IT IS "IMHOTEPIAN"
-------------------------------------------
Imhotep was simultaneously the architect of the Step Pyramid at Saqqara and the
founder of the empirical-medical tradition later codified in the Edwin Smith
Surgical Papyrus. His signature is the fusion of TWO disciplines into ONE method:

    * ARCHITECTURE  ->  intelligence is a *load-bearing hierarchy*. A pyramid is
                        stable because each course rests on a wider course below
                        it. So our network is literally a pyramid: a wide base of
                        raw observation that narrows, course by course, toward a
                        single apex of integrated judgement.

    * MEDICINE      ->  intelligence must DIAGNOSE before it ACTS, and must know
                        the limit of its competence. The Edwin Smith Papyrus
                        grades every case into exactly three verdicts:
                           (0) "an ailment I will treat"
                           (1) "an ailment I will contend with"   (treat w/ difficulty)
                           (2) "an ailment not to be treated"     (abstain / refer)
                        We give the network a second head that learns this very
                        trichotomy, and we let it ABSTAIN. Epistemic humility is
                        not a bolt-on; it is a first-class output.

    * MA'AT         ->  cosmic order / truth / justice. Imhotep's medicine sat
                        inside the ethical frame of Ma'at. We encode this as an
                        inviolable value gate: certain (diagnosis, treatment)
                        pairs are forbidden (contraindicated / harmful) and the
                        model is *structurally* incapable of recommending them.

    * THE FOUR SOULS -> Egyptian psychology split the self into ib (heart/working
                        judgement), ba (the mobile soul that travels -> episodic
                        recall of past cases), ka (the enduring double ->
                        consolidated semantic prototypes) and shut (the shadow ->
                        procedural reflexes). We implement ib/ba/ka/shut as four
                        cooperating memory subsystems, and we PROVE the ba
                        (episodic retrieval) memory adds real predictive value.

WHAT MAKES THIS A REAL MODEL AND NOT A DEMO
-------------------------------------------
1. Hand-written forward AND backward passes (no autograd framework).
2. A finite-difference GRADIENT CHECK proves the backprop is mathematically
   correct (max relative error < 1e-4).
3. A real Adam optimiser; training loss actually falls, validation accuracy
   actually climbs far above chance.
4. Every headline number printed at the bottom is MEASURED at run time, never
   hard-coded. Re-run it and the seeds make it reproducible.
5. A test battery (`run_tests()`) asserts the model learns, is calibrated, is
   safe under the Ma'at gate, and that episodic memory beats chance. If any
   assertion fails the process exits non-zero.

Dependencies: numpy only.  Runtime: a few seconds on a laptop CPU.
Run:          python3 chapter_0003_imhotep_-2650.py
================================================================================
"""

from __future__ import annotations

import sys
import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any

import numpy as np

# A single global seed makes the whole chapter reproducible.
GLOBAL_SEED = 2650  # Imhotep's traditional floruit, used as our lucky number.


# ============================================================================
# SECTION 1 - NUMERICALLY-STABLE PRIMITIVES
# ----------------------------------------------------------------------------
# Imhotep's medicine prized careful measurement over guesswork. In the same
# spirit we never let an exponential overflow or a log(0) corrupt a result.
# ============================================================================

def relu(z: np.ndarray) -> np.ndarray:
    """Rectified linear unit. The simplest 'is the signal present?' gate."""
    return np.maximum(0.0, z)


def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    """Stable softmax: subtract the row max before exponentiating."""
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def cross_entropy(probs: np.ndarray, labels: np.ndarray) -> float:
    """Mean negative log-likelihood of the true class. `labels` are integers."""
    n = probs.shape[0]
    eps = 1e-12
    return float(-np.mean(np.log(probs[np.arange(n), labels] + eps)))


def one_hot(labels: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((labels.shape[0], n_classes), dtype=np.float64)
    out[np.arange(labels.shape[0]), labels] = 1.0
    return out


def entropy(probs: np.ndarray, axis: int = -1) -> np.ndarray:
    """Shannon entropy per row, in nats. High entropy => the model is unsure."""
    eps = 1e-12
    return -np.sum(probs * np.log(probs + eps), axis=axis)


# ============================================================================
# SECTION 2 - THE PYRAMIDAL TRUNK + TWO HEADS  (the architecture proper)
# ----------------------------------------------------------------------------
# Shapes (the "courses" of the pyramid), widest at the base:
#
#     signs(24) --W1--> Observation(48) --W2--> Presentation(32)
#               --W3--> Integration(16=apex) --+--Wd--> Diagnosis(6 conditions)
#                                              +--Wv--> Verdict(3: treat/contend/abstain)
#
# A ReLU sits after each hidden course. The apex (16-d) embedding is also the
# vector we hand to the ba/ka memories, so 'the heart that the soul carries'.
# ============================================================================

@dataclass
class IDAConfig:
    n_signs: int = 24          # raw clinical observations per patient
    n_dx: int = 6              # number of latent conditions (diagnoses)
    n_verdict: int = 3         # Edwin-Smith trichotomy
    n_treatments: int = 6      # treatment vocabulary (1 per condition, here)
    h_obs: int = 48            # Observation course  (the wide base)
    h_pres: int = 32           # Presentation course
    h_apex: int = 16           # Integration apex (also the memory embedding dim)
    verdict_loss_weight: float = 0.5   # lambda on the auxiliary verdict head
    activation: str = "relu"           # "relu" (model) or "tanh" (grad-check)
    seed: int = GLOBAL_SEED


class ImhotepDiagnosticArchitecture:
    """
    The full model. Holds parameters, runs forward/backward, and owns the
    four Egyptian memory subsystems (ib/ba/ka/shut). All matrix maths is
    explicit so the backward pass can be verified by finite differences.
    """

    def __init__(self, cfg: IDAConfig):
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        self.rng = rng

        # --- He/Kaiming initialisation keeps ReLU activations well-scaled. ---
        def he(shape):
            fan_in = shape[0]
            return rng.standard_normal(shape) * math.sqrt(2.0 / fan_in)

        c = cfg
        # Trunk
        self.W1 = he((c.n_signs, c.h_obs));  self.b1 = np.zeros(c.h_obs)
        self.W2 = he((c.h_obs,  c.h_pres));  self.b2 = np.zeros(c.h_pres)
        self.W3 = he((c.h_pres, c.h_apex));  self.b3 = np.zeros(c.h_apex)
        # Heads
        self.Wd = he((c.h_apex, c.n_dx));    self.bd = np.zeros(c.n_dx)
        self.Wv = he((c.h_apex, c.n_verdict)); self.bv = np.zeros(c.n_verdict)

        # Adam optimiser state (m = 1st moment, v = 2nd moment, per parameter).
        self._adam_m: Dict[str, np.ndarray] = {}
        self._adam_v: Dict[str, np.ndarray] = {}
        self._adam_t = 0

        # ---- The four souls (memory subsystems) ----
        self.ib = HeartWorkingMemory(c.h_apex)          # live working judgement
        self.ba = EpisodicSoulMemory(c.h_apex, c.n_dx)  # travels to past cases
        self.ka = EnduringPrototypeMemory(c.h_apex, c.n_dx)  # consolidated semantics
        self.shut = ShadowProcedureMemory()             # procedural reflexes

        # ---- The Ma'at value gate: forbidden (diagnosis -> treatment) pairs ----
        # A small, fixed table of contraindications. The model is structurally
        # unable to emit any treatment marked forbidden for the chosen diagnosis.
        self.maat_forbidden = np.zeros((c.n_dx, c.n_treatments), dtype=bool)
        # Example contraindications (condition d must NOT receive treatment t).
        # Only set pairs that fit the current dimensions (keeps tiny grad-check
        # configs valid as well as the full clinical config).
        for d, t in [(0, 3), (2, 0), (4, 5), (5, 4)]:
            if d < c.n_dx and t < c.n_treatments:
                self.maat_forbidden[d, t] = True

    # -- parameter bookkeeping -------------------------------------------------
    def params(self) -> Dict[str, np.ndarray]:
        return {
            "W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2,
            "W3": self.W3, "b3": self.b3, "Wd": self.Wd, "bd": self.bd,
            "Wv": self.Wv, "bv": self.bv,
        }

    def _act(self, z: np.ndarray) -> np.ndarray:
        return relu(z) if self.cfg.activation == "relu" else np.tanh(z)

    def _act_grad(self, z: np.ndarray, a: np.ndarray) -> np.ndarray:
        # local derivative of the activation; for tanh use the output a.
        if self.cfg.activation == "relu":
            return (z > 0).astype(z.dtype)
        return 1.0 - a * a

    # -- FORWARD PASS ----------------------------------------------------------
    def forward(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        X: (B, n_signs). Returns a cache dict holding every intermediate value
        so that backward() can reuse them. p_dx / p_vd are probability outputs.
        """
        z1 = X @ self.W1 + self.b1;  a1 = self._act(z1)   # Observation course
        z2 = a1 @ self.W2 + self.b2; a2 = self._act(z2)   # Presentation course
        z3 = a2 @ self.W3 + self.b3; a3 = self._act(z3)   # Integration apex

        dx_logits = a3 @ self.Wd + self.bd
        vd_logits = a3 @ self.Wv + self.bv
        p_dx = softmax(dx_logits)
        p_vd = softmax(vd_logits)

        return {
            "X": X, "z1": z1, "a1": a1, "z2": z2, "a2": a2, "z3": z3, "a3": a3,
            "dx_logits": dx_logits, "vd_logits": vd_logits,
            "p_dx": p_dx, "p_vd": p_vd,
        }

    # -- LOSS ------------------------------------------------------------------
    def loss(self, cache: Dict[str, np.ndarray],
             y_dx: np.ndarray, y_vd: np.ndarray) -> float:
        l_dx = cross_entropy(cache["p_dx"], y_dx)
        l_vd = cross_entropy(cache["p_vd"], y_vd)
        return l_dx + self.cfg.verdict_loss_weight * l_vd

    # -- BACKWARD PASS (explicit; verified by gradient check) ------------------
    def backward(self, cache: Dict[str, np.ndarray],
                 y_dx: np.ndarray, y_vd: np.ndarray) -> Dict[str, np.ndarray]:
        B = cache["X"].shape[0]
        c = self.cfg

        # dL/dlogits for a softmax+CE head is (p - onehot)/B.
        d_dx = (cache["p_dx"] - one_hot(y_dx, c.n_dx)) / B
        d_vd = self.cfg.verdict_loss_weight * \
               (cache["p_vd"] - one_hot(y_vd, c.n_verdict)) / B

        a3 = cache["a3"]
        gWd = a3.T @ d_dx;  gbd = d_dx.sum(0)
        gWv = a3.T @ d_vd;  gbv = d_vd.sum(0)

        # gradient flowing back into the apex embedding from both heads
        da3 = d_dx @ self.Wd.T + d_vd @ self.Wv.T
        dz3 = da3 * self._act_grad(cache["z3"], cache["a3"])
        gW3 = cache["a2"].T @ dz3; gb3 = dz3.sum(0)

        da2 = dz3 @ self.W3.T
        dz2 = da2 * self._act_grad(cache["z2"], cache["a2"])
        gW2 = cache["a1"].T @ dz2; gb2 = dz2.sum(0)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * self._act_grad(cache["z1"], cache["a1"])
        gW1 = cache["X"].T @ dz1; gb1 = dz1.sum(0)

        return {
            "W1": gW1, "b1": gb1, "W2": gW2, "b2": gb2, "W3": gW3, "b3": gb3,
            "Wd": gWd, "bd": gbd, "Wv": gWv, "bv": gbv,
        }

    # -- ADAM UPDATE -----------------------------------------------------------
    def adam_step(self, grads: Dict[str, np.ndarray],
                  lr: float = 3e-3, beta1: float = 0.9,
                  beta2: float = 0.999, eps: float = 1e-8) -> None:
        self._adam_t += 1
        t = self._adam_t
        for name, g in grads.items():
            if name not in self._adam_m:
                self._adam_m[name] = np.zeros_like(g)
                self._adam_v[name] = np.zeros_like(g)
            m = self._adam_m[name] = beta1 * self._adam_m[name] + (1 - beta1) * g
            v = self._adam_v[name] = beta2 * self._adam_v[name] + (1 - beta2) * (g * g)
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            self.params()[name] -= lr * m_hat / (np.sqrt(v_hat) + eps)

    # -- INFERENCE WITH HUMILITY + MA'AT + MEMORY ------------------------------
    def diagnose(self, x: np.ndarray, abstain_entropy: float = 1.10,
                 use_memory: bool = True, mem_weight: float = 0.35
                 ) -> Dict[str, Any]:
        """
        Full clinical pass for ONE patient (x: shape (n_signs,)).
        Pipeline mirrors the Edwin Smith method:
          observe -> integrate -> recall similar cases (ba) -> judge ->
          decide verdict (treat / contend / abstain) -> apply Ma'at gate.
        """
        cache = self.forward(x[None, :])
        emb = cache["a3"][0]                     # the apex embedding ("heart")
        p_net = cache["p_dx"][0].copy()

        self.ib.hold(emb)                        # ib keeps the live judgement

        # ba: blend the network's belief with a vote from remembered cases.
        p_mem = None
        if use_memory and self.ba.size > 0:
            p_mem = self.ba.vote(emb)
            p_blend = (1 - mem_weight) * p_net + mem_weight * p_mem
        else:
            p_blend = p_net
        p_blend = p_blend / p_blend.sum()

        dx = int(np.argmax(p_blend))
        confidence = float(p_blend[dx])
        h = float(entropy(p_blend[None, :])[0])

        # Verdict head: the model's own grading of case difficulty.
        p_vd = cache["p_vd"][0]
        verdict_idx = int(np.argmax(p_vd))
        verdict_name = ["treat", "contend", "abstain"][verdict_idx]

        # EPISTEMIC HUMILITY: if too uncertain, override toward abstention.
        forced_abstain = h > abstain_entropy
        if forced_abstain:
            verdict_idx = 2
            verdict_name = "abstain"

        # MA'AT VALUE GATE: choose a treatment, but never a forbidden one.
        treatment = None
        if verdict_idx != 2:                     # only act if not abstaining
            # default policy: treatment index == diagnosis index, unless forbidden
            allowed = ~self.maat_forbidden[dx]
            # rank treatments by a simple suitability score (here: identity prior)
            prior = np.zeros(self.cfg.n_treatments)
            prior[dx] = 1.0                      # the matched treatment is preferred
            prior = prior + 1e-3 * self.rng.standard_normal(self.cfg.n_treatments)
            prior[~allowed] = -np.inf            # Ma'at forbids these outright
            treatment = int(np.argmax(prior))

        return {
            "diagnosis": dx,
            "confidence": confidence,
            "entropy": h,
            "verdict": verdict_name,
            "verdict_idx": verdict_idx,
            "abstained": verdict_idx == 2,
            "forced_abstain": forced_abstain,
            "treatment": treatment,
            "p_dx": p_blend,
            "p_net": p_net,
            "p_mem": p_mem,
        }


# ============================================================================
# SECTION 3 - THE FOUR SOULS  (ib / ba / ka / shut)
# ============================================================================

class HeartWorkingMemory:
    """ib - the heart. Holds the single live judgement embedding (working memory)."""
    def __init__(self, dim: int):
        self.dim = dim
        self.current: Optional[np.ndarray] = None

    def hold(self, emb: np.ndarray) -> None:
        self.current = emb.copy()


class EpisodicSoulMemory:
    """
    ba - the soul that travels between worlds. An episodic store of past apex
    embeddings + their true diagnoses. At inference it 'travels back' to the k
    most similar remembered cases and returns a soft vote. This is genuine
    retrieval-augmentation and we measure that it beats chance on its own.
    """
    def __init__(self, dim: int, n_dx: int, capacity: int = 4000, k: int = 9):
        self.dim = dim
        self.n_dx = n_dx
        self.capacity = capacity
        self.k = k
        self.keys = np.zeros((0, dim))
        self.labels = np.zeros((0,), dtype=int)

    @property
    def size(self) -> int:
        return self.keys.shape[0]

    def remember(self, embs: np.ndarray, labels: np.ndarray) -> None:
        self.keys = np.vstack([self.keys, embs])
        self.labels = np.concatenate([self.labels, labels])
        if self.size > self.capacity:           # forget oldest if over capacity
            self.keys = self.keys[-self.capacity:]
            self.labels = self.labels[-self.capacity:]

    def _cos(self, q: np.ndarray) -> np.ndarray:
        qn = q / (np.linalg.norm(q) + 1e-9)
        kn = self.keys / (np.linalg.norm(self.keys, axis=1, keepdims=True) + 1e-9)
        return kn @ qn

    def vote(self, q: np.ndarray) -> np.ndarray:
        sims = self._cos(q)
        idx = np.argsort(-sims)[: self.k]
        weights = np.maximum(sims[idx], 0.0) + 1e-6   # similarity-weighted vote
        dist = np.zeros(self.n_dx)
        for j, w in zip(idx, weights):
            dist[self.labels[j]] += w
        return dist / dist.sum()

    def predict(self, q: np.ndarray) -> int:
        return int(np.argmax(self.vote(q)))


class EnduringPrototypeMemory:
    """
    ka - the enduring double. Consolidated semantic memory: the running mean
    embedding (prototype) of each diagnosis class. Stable, slow-changing
    'knowledge that outlasts the case', used for prototype-distance sanity
    checks and explanation.
    """
    def __init__(self, dim: int, n_dx: int):
        self.dim = dim
        self.n_dx = n_dx
        self.proto = np.zeros((n_dx, dim))
        self.counts = np.zeros(n_dx)

    def consolidate(self, embs: np.ndarray, labels: np.ndarray) -> None:
        for d in range(self.n_dx):
            sel = labels == d
            if sel.any():
                self.proto[d] = embs[sel].mean(0)
                self.counts[d] = sel.sum()

    def nearest_prototype(self, emb: np.ndarray) -> int:
        d = np.linalg.norm(self.proto - emb[None, :], axis=1)
        return int(np.argmin(d))


class ShadowProcedureMemory:
    """
    shut - the shadow. Procedural reflexes: compiled 'if verdict X then do Y'
    rules that fire without deliberation. Demonstrates that not all action needs
    the full pyramid; some responses are cached habits.
    """
    def __init__(self):
        self.rules: Dict[str, str] = {
            "abstain": "Do not treat. Refer and reassess. (Edwin Smith verdict 3)",
            "contend": "Treat cautiously, monitor, prepare to escalate.",
            "treat":   "Proceed with the matched, Ma'at-permitted treatment.",
        }

    def reflex(self, verdict_name: str) -> str:
        return self.rules.get(verdict_name, "Observe further.")


# ============================================================================
# SECTION 4 - SYNTHETIC CLINICAL WORLD (data with real, learnable structure)
# ----------------------------------------------------------------------------
# Each condition has a sparse 'prototype' over the 24 signs. Patients are
# prototype + noise. Some patients are deliberately AMBIGUOUS (a mixture of two
# conditions) -> those are the cases the verdict head should grade 'contend' or
# 'abstain'. One condition (index 5) is made RARE in training so we can test
# whether episodic memory helps the tail of the distribution.
# ============================================================================

@dataclass
class Dataset:
    X: np.ndarray
    y_dx: np.ndarray
    y_vd: np.ndarray
    ambiguous: np.ndarray


def make_world(cfg: IDAConfig, seed: int):
    rng = np.random.default_rng(seed)
    # Sparse prototypes: each condition activates ~6 of the 24 signs.
    protos = np.zeros((cfg.n_dx, cfg.n_signs))
    for d in range(cfg.n_dx):
        active = rng.choice(cfg.n_signs, size=6, replace=False)
        protos[d, active] = rng.uniform(1.0, 2.5, size=6)
    # 'severity' of each condition decides part of the verdict prior.
    severity = rng.uniform(0, 1, size=cfg.n_dx)
    return rng, protos, severity


def sample_dataset(cfg: IDAConfig, protos, severity, n: int, rng,
                   rare_class: int = 5, rare_factor: float = 0.15) -> Dataset:
    X = np.zeros((n, cfg.n_signs))
    y_dx = np.zeros(n, dtype=int)
    y_vd = np.zeros(n, dtype=int)
    amb = np.zeros(n, dtype=bool)

    # class sampling probabilities (rare_class is suppressed)
    cls_p = np.ones(cfg.n_dx)
    cls_p[rare_class] = rare_factor
    cls_p = cls_p / cls_p.sum()

    for i in range(n):
        d = rng.choice(cfg.n_dx, p=cls_p)
        is_amb = rng.random() < 0.30            # 30% ambiguous mixtures
        if is_amb:
            d2 = rng.choice([k for k in range(cfg.n_dx) if k != d])
            alpha = rng.uniform(0.45, 0.65)     # dominant share
            base = alpha * protos[d] + (1 - alpha) * protos[d2]
            noise_scale = 0.9
        else:
            base = protos[d].copy()
            noise_scale = 0.45
        X[i] = base + rng.normal(0, noise_scale, size=cfg.n_signs)
        y_dx[i] = d
        amb[i] = is_amb

        # Verdict label (Edwin Smith trichotomy), a principled function of the
        # case: clear+mild -> treat(0); ambiguous OR severe -> contend(1);
        # very ambiguous AND very severe -> abstain(2).
        sev = severity[d]
        if is_amb and sev > 0.66:
            y_vd[i] = 2
        elif is_amb or sev > 0.5:
            y_vd[i] = 1
        else:
            y_vd[i] = 0

    # Standardise signs (z-score) - good practice and helps optimisation.
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    return Dataset(X=X, y_dx=y_dx, y_vd=y_vd, ambiguous=amb)


# ============================================================================
# SECTION 5 - TRAINING LOOP
# ============================================================================

def accuracy(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(pred == true))


def train(model: ImhotepDiagnosticArchitecture,
          train_ds: Dataset, val_ds: Dataset,
          epochs: int = 60, batch: int = 128, lr: float = 3e-3,
          verbose: bool = True) -> Dict[str, list]:
    rng = np.random.default_rng(model.cfg.seed + 1)
    n = train_ds.X.shape[0]
    hist = {"train_loss": [], "val_dx_acc": [], "val_vd_acc": []}

    for ep in range(epochs):
        order = rng.permutation(n)
        ep_loss = 0.0
        nb = 0
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            Xb, ydb, yvb = train_ds.X[idx], train_ds.y_dx[idx], train_ds.y_vd[idx]
            cache = model.forward(Xb)
            ep_loss += model.loss(cache, ydb, yvb); nb += 1
            grads = model.backward(cache, ydb, yvb)
            model.adam_step(grads, lr=lr)

        # validation
        vc = model.forward(val_ds.X)
        vda = accuracy(np.argmax(vc["p_dx"], 1), val_ds.y_dx)
        vva = accuracy(np.argmax(vc["p_vd"], 1), val_ds.y_vd)
        hist["train_loss"].append(ep_loss / nb)
        hist["val_dx_acc"].append(vda)
        hist["val_vd_acc"].append(vva)
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print(f"  epoch {ep:3d} | train_loss {ep_loss/nb:6.4f} "
                  f"| val_dx_acc {vda*100:5.1f}% | val_vd_acc {vva*100:5.1f}%")

    # After training, populate the souls from the training set.
    tc = model.forward(train_ds.X)
    embs = tc["a3"]
    model.ba.remember(embs, train_ds.y_dx)
    model.ka.consolidate(embs, train_ds.y_dx)
    return hist


# ============================================================================
# SECTION 6 - GRADIENT CHECK  (proves the backward pass is correct)
# ============================================================================

def gradient_check(seed: int = 7) -> float:
    """
    Compare analytic gradients to central finite differences on a tiny problem.
    Returns the maximum relative error across all checked parameters. A correct
    implementation gives ~1e-6 or smaller.
    """
    cfg = IDAConfig(n_signs=8, n_dx=4, n_verdict=3, h_obs=10, h_pres=7,
                    h_apex=5, activation="tanh", seed=seed)
    model = ImhotepDiagnosticArchitecture(cfg)
    rng = np.random.default_rng(seed)
    B = 5
    X = rng.standard_normal((B, cfg.n_signs))
    y_dx = rng.integers(0, cfg.n_dx, size=B)
    y_vd = rng.integers(0, cfg.n_verdict, size=B)

    cache = model.forward(X)
    grads = model.backward(cache, y_dx, y_vd)

    eps = 1e-5
    max_rel = 0.0
    for name, P in model.params().items():
        flat = P.ravel()
        g_an = grads[name].ravel()
        # check up to 12 random coordinates per parameter for speed
        coords = rng.choice(flat.size, size=min(12, flat.size), replace=False)
        for c in coords:
            orig = flat[c]
            flat[c] = orig + eps
            lp = model.loss(model.forward(X), y_dx, y_vd)
            flat[c] = orig - eps
            lm = model.loss(model.forward(X), y_dx, y_vd)
            flat[c] = orig
            num = (lp - lm) / (2 * eps)
            denom = max(1e-8, abs(num) + abs(g_an[c]))
            rel = abs(num - g_an[c]) / denom
            max_rel = max(max_rel, rel)
    return max_rel


# ============================================================================
# SECTION 7 - EVALUATION HELPERS
# ============================================================================

def selective_accuracy(model, ds: Dataset, conf_quantile: float = 0.5
                       ) -> Tuple[float, float]:
    """
    Epistemic-humility test. Compute overall diagnostic accuracy, then the
    accuracy on only the most-confident half of cases. If the model is well
    calibrated, the confident subset is MORE accurate -> humility pays off.
    """
    c = model.forward(ds.X)
    p = c["p_dx"]
    pred = np.argmax(p, 1)
    conf = np.max(p, 1)
    overall = accuracy(pred, ds.y_dx)
    thresh = np.quantile(conf, conf_quantile)
    keep = conf >= thresh
    confident = accuracy(pred[keep], ds.y_dx[keep])
    return overall, confident


def maat_audit(model, ds: Dataset) -> Tuple[int, int]:
    """
    Run the full clinical pipeline on every test patient and count how often a
    FORBIDDEN treatment is recommended. The Ma'at gate should make this exactly
    zero. Returns (violations, n_acted).
    """
    violations = 0
    acted = 0
    for i in range(ds.X.shape[0]):
        out = model.diagnose(ds.X[i])
        if out["treatment"] is not None:
            acted += 1
            if model.maat_forbidden[out["diagnosis"], out["treatment"]]:
                violations += 1
    return violations, acted


def memory_only_accuracy(model, ds: Dataset) -> float:
    """Accuracy of the ba (episodic) memory voting ALONE on the test set."""
    c = model.forward(ds.X)
    embs = c["a3"]
    preds = np.array([model.ba.predict(embs[i]) for i in range(embs.shape[0])])
    return accuracy(preds, ds.y_dx)


# ============================================================================
# SECTION 8 - TEST BATTERY  (the model is "tested, not a demo")
# ============================================================================

def run_tests() -> Dict[str, Any]:
    print("=" * 74)
    print(" IMHOTEP DIAGNOSTIC ARCHITECTURE - BUILD & VERIFY")
    print("=" * 74)

    results: Dict[str, Any] = {}

    # 1) Gradient check ------------------------------------------------------
    print("\n[1/6] Gradient check (finite differences vs analytic backprop)...")
    max_rel = gradient_check()
    print(f"      max relative error = {max_rel:.2e}  "
          f"(must be < 1e-4)")
    assert max_rel < 1e-4, "Backprop is INCORRECT (gradient check failed)."
    results["grad_check_max_rel_err"] = max_rel

    # 2) Build world and train ----------------------------------------------
    print("\n[2/6] Building synthetic clinical world and training...")
    cfg = IDAConfig()
    rng, protos, severity = make_world(cfg, seed=cfg.seed)
    train_ds = sample_dataset(cfg, protos, severity, n=6000, rng=rng)
    val_ds   = sample_dataset(cfg, protos, severity, n=1500, rng=rng)
    test_ds  = sample_dataset(cfg, protos, severity, n=2000, rng=rng)

    model = ImhotepDiagnosticArchitecture(cfg)
    hist = train(model, train_ds, val_ds, epochs=60)

    init_loss = hist["train_loss"][0]
    final_loss = hist["train_loss"][-1]
    print(f"      train loss {init_loss:.4f} -> {final_loss:.4f}")
    assert final_loss < 0.5 * init_loss, "Training did not reduce the loss enough."
    results["init_loss"] = init_loss
    results["final_loss"] = final_loss

    # 3) Diagnosis accuracy well above chance --------------------------------
    print("\n[3/6] Diagnostic & verdict accuracy on held-out test set...")
    tc = model.forward(test_ds.X)
    dx_acc = accuracy(np.argmax(tc["p_dx"], 1), test_ds.y_dx)
    vd_acc = accuracy(np.argmax(tc["p_vd"], 1), test_ds.y_vd)
    chance = 1.0 / cfg.n_dx
    print(f"      diagnosis accuracy = {dx_acc*100:5.1f}%  (chance {chance*100:.1f}%)")
    print(f"      verdict   accuracy = {vd_acc*100:5.1f}%  (chance 33.3%)")
    assert dx_acc > 2.5 * chance, "Diagnosis accuracy not meaningfully above chance."
    assert vd_acc > 0.45, "Verdict head failed to learn the Edwin-Smith trichotomy."
    results["test_dx_acc"] = dx_acc
    results["test_vd_acc"] = vd_acc

    # 4) Epistemic humility / calibration ------------------------------------
    print("\n[4/6] Epistemic humility (selective prediction)...")
    overall, confident = selective_accuracy(model, test_ds, conf_quantile=0.5)
    print(f"      overall accuracy        = {overall*100:5.1f}%")
    print(f"      top-50%-confidence acc  = {confident*100:5.1f}%")
    assert confident > overall, "Confident cases are not more accurate (poor calibration)."
    results["overall_acc"] = overall
    results["confident_acc"] = confident

    # 5) Ma'at value gate is inviolable --------------------------------------
    print("\n[5/6] Ma'at value gate audit (forbidden treatments)...")
    violations, acted = maat_audit(model, test_ds)
    print(f"      acted on {acted} patients; forbidden recommendations = {violations}")
    assert violations == 0, "Ma'at gate VIOLATED - a forbidden treatment was recommended."
    results["maat_violations"] = violations
    results["maat_acted"] = acted

    # 6) Episodic (ba) memory beats chance -----------------------------------
    print("\n[6/6] Episodic 'ba' memory vote (retrieval) ...")
    mem_acc = memory_only_accuracy(model, test_ds)
    # accuracy on the RARE class with and without memory blending
    rare = test_ds.y_dx == 5
    net_pred = np.argmax(tc["p_dx"], 1)
    net_rare = accuracy(net_pred[rare], test_ds.y_dx[rare]) if rare.any() else float("nan")
    blend_pred = np.array([model.diagnose(test_ds.X[i])["diagnosis"]
                           for i in np.where(rare)[0]]) if rare.any() else np.array([])
    blend_rare = accuracy(blend_pred, test_ds.y_dx[rare]) if rare.any() else float("nan")
    print(f"      memory-only accuracy    = {mem_acc*100:5.1f}%  (chance {chance*100:.1f}%)")
    print(f"      rare-class acc  net={net_rare*100:4.1f}%  net+memory={blend_rare*100:4.1f}%")
    assert mem_acc > 2.0 * chance, "Episodic memory does not beat chance."
    results["memory_only_acc"] = mem_acc
    results["rare_net_acc"] = net_rare
    results["rare_blend_acc"] = blend_rare

    # ---- store handles for the demo ----
    results["_model"] = model
    results["_test_ds"] = test_ds
    results["_hist"] = hist

    print("\n" + "=" * 74)
    print(" ALL TESTS PASSED")
    print("=" * 74)
    return results


# ============================================================================
# SECTION 9 - WALKTHROUGH OF THREE PATIENTS (human-readable demonstration)
# ============================================================================

def clinical_walkthrough(model: ImhotepDiagnosticArchitecture, test_ds: Dataset):
    print("\n" + "=" * 74)
    print(" CLINICAL WALKTHROUGH - the Edwin Smith pipeline on real test cases")
    print("=" * 74)
    # pick one clear case, one ambiguous case, one rare-class case
    clear_idx = np.where((~test_ds.ambiguous))[0][0]
    amb_idx = np.where(test_ds.ambiguous)[0][0]
    rare_idx = np.where(test_ds.y_dx == 5)[0]
    picks = [("clear", clear_idx), ("ambiguous", amb_idx)]
    if rare_idx.size:
        picks.append(("rare-class", int(rare_idx[0])))

    for label, i in picks:
        out = model.diagnose(test_ds.X[i])
        reflex = model.shut.reflex(out["verdict"])
        print(f"\nPatient [{label}]  (true dx = {test_ds.y_dx[i]}, "
              f"true verdict = {test_ds.y_vd[i]})")
        print(f"  predicted diagnosis : {out['diagnosis']}  "
              f"(confidence {out['confidence']*100:4.1f}%)")
        print(f"  belief entropy      : {out['entropy']:.3f} nats")
        print(f"  Edwin-Smith verdict : {out['verdict']}"
              + ("  [humility override]" if out["forced_abstain"] else ""))
        print(f"  treatment (Ma'at-ok): "
              f"{out['treatment'] if out['treatment'] is not None else 'NONE (abstain)'}")
        print(f"  shut reflex (habit) : {reflex}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    res = run_tests()
    clinical_walkthrough(res["_model"], res["_test_ds"])

    print("\nSummary of measured headline numbers (used verbatim in the chapter):")
    for k in ["grad_check_max_rel_err", "final_loss", "test_dx_acc",
              "test_vd_acc", "overall_acc", "confident_acc",
              "maat_violations", "memory_only_acc"]:
        print(f"  {k:24s} = {res[k]}")
    sys.exit(0)
