#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=========================================================================
"THE JEWEL THAT COUNTS THE OTHERS"  — an Indra-net reflective-equilibrium
neuron built from Fazang's own doctrine of mutual inclusion.
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 10 Minds 181 - 200 Available on Amazon https://www.amazon.com/dp/B0HJYMZ3G6
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0190_Fazang_643 - Fazang (643–712)
================================================================================    
WHAT THIS FILE IS
-----------------
A complete, from-scratch, pure-NumPy artificial-neuron architecture whose
every mechanism is lifted from a specific, textually attested move in
Fazang's Huayan writings (Huayan wujiao zhang 華嚴五教章, Jin shizi zhang
金師子章, Huayan jing tanxuan ji 華嚴經探玄記).  It is not a demo: it has a
forward pass, a hand-derived backward pass, a finite-difference gradient
check (mandatory, executed on every run), an Adam optimiser, real training
on two tasks, and a battery of self-tests that verify the *doctrinal*
claims as *numerical* claims.

THE MIND IT EMBODIES  (each item is a mechanism below)
------------------------------------------------------
  [相入 mutual inclusion]  A jewel's state is not its own input but its
     input PLUS the reflected states of every other jewel, iterated to a
     reflective equilibrium (Indra's net: images of images, 重重無盡).
     → `reflect()`: K unrolled reflection passes.

  [有力／無力 power / lack of power]  Fazang: when one dharma has complete
     power as cause, the others completely LACK power (they are only
     conditions).  Power is not shared, it is CONSERVED across every pair.
     → `PowerGate`: an antisymmetric logit matrix Λ; the gate
       G_jk = σ(Λ_jk − Λ_kj) satisfies G_jk + G_kj = 1 exactly.  This is
       NOT attention over keys: it is a learned, static HETERARCHY of who
       is cause and who is condition in each pairing (cf. N. Jones 2023).

  [同體門／異體門 same-body / different-body gates]  Fazang splits every
     inclusion into what a thing contains "in its own body" (the one coin
     already holds the positions one-through-ten in itself) and what it
     contains "in different bodies" (the other nine coins).
     → two recurrent matrices: S_same (self-channel) and R_diff (channel
       of reflection from the others).

  [主伴 principal and retinue]  In any act of cognition ONE jewel is
     principal; all others are retinue; and the roles ROTATE so that every
     jewel is principal in its own act.
     → `principal` argument: the read-out is taken from a single jewel.
       During training the principal rotates at random; at test time we
       demand that EVERY jewel, read alone, yields the whole.

  [六相 six characteristics]  Totality 總, particularity 別, sameness 同,
     difference 異, integration 成, disintegration 壞 — the Rafter
     Dialogue.  The rafter IS the building because it wholly (全力) makes
     it; yet the rafter remains a rafter.
     → `SixCharacteristicLoss`: 成 the whole is decodable from ANY part
       (cross-entropy on the principal's read-out); 同 all parts agree on
       the whole (variance of the whole-subspace across jewels); 壞 every
       part still reconstructs ITS OWN facet from its final state (it does
       not dissolve into the whole); 總／別／異 are measured, not trained.

  [全力 total power, the rafter argument]  Every part is a TOTAL cause of
     the whole, not a fractional one.  → self-test: ablating any single
     jewel's input moves EVERY other jewel's image of the whole (the
     "total-cause matrix" has no zero off-diagonal entry).

  [數十錢 counting ten coins]  Task B literally counts coins: ten jewels,
     each holding one coin or nothing; each jewel alone must state the
     total.  "One is ten, because without the one the ten is not."

  [金獅子 the golden lion]  Task A: a lion (a shape) made of gold (pixels)
     is cut into nine facets; each facet alone must name the whole lion
     after reflection.  "In each hair of the lion, the whole lion."

The file trains in well under two minutes on a laptop CPU.

USAGE
-----
    python chapter_0190_Fazang_643.py            # full run (train + tests)
    python chapter_0190_Fazang_643.py --quick    # shorter training
    python chapter_0190_Fazang_643.py --gradcheck-only

No dependency beyond NumPy.
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# 0. Small numerical helpers
# ---------------------------------------------------------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def one_hot(y: np.ndarray, c: int) -> np.ndarray:
    out = np.zeros((y.shape[0], c))
    out[np.arange(y.shape[0]), y] = 1.0
    return out


# ---------------------------------------------------------------------------
# 1. Parameters  (a flat dict so the gradient check can walk every tensor)
# ---------------------------------------------------------------------------

@dataclass
class Config:
    n_jewels: int = 9        # N — jewels in the net (mirrors in the hall)
    p_in: int = 16           # p — dimension of each jewel's own facet input
    d: int = 32              # d — state width of a jewel
    m: int = 16              # m — width of the shared "whole" subspace (總相)
    c: int = 6               # c — number of classes the whole must name
    k_reflect: int = 3       # K — reflection depth (層 layers of images)
    lam_same: float = 0.5    # weight of 同 (agreement on the whole)
    lam_own: float = 0.5     # weight of 壞 (retention of one's own facet)
    principal_mode: str = "rotate"   # "rotate" | "all"
    seed: int = 190


class Params(dict):
    """Flat parameter store; keys are the doctrinal names of each tensor."""

    @staticmethod
    def init(cfg: Config, rng: np.random.Generator) -> "Params":
        N, p, d, m, c = cfg.n_jewels, cfg.p_in, cfg.d, cfg.m, cfg.c
        s = lambda *shape: rng.normal(0.0, 1.0, size=shape)
        P = Params()
        # 別 particularity: facet encoder (own input → own facet)
        P["W_facet"] = s(p, d) * np.sqrt(1.0 / p)
        P["b_facet"] = np.zeros(d)
        # 同體門 same-body recurrent channel
        P["S_same"] = s(d, d) * (0.3 / np.sqrt(d))
        # 異體門 different-body reflection channel (the mirror transform)
        P["R_diff"] = s(d, d) * (0.6 / np.sqrt(d))
        # 有力／無力 antisymmetric power logits (raw; antisymmetrised in use)
        P["Lambda"] = s(N, N) * 0.5
        # 總相 projection to the shared whole-subspace
        P["P_whole"] = s(d, m) * np.sqrt(1.0 / d)
        # 成 read-out of the whole (class logits)
        P["W_out"] = s(m, c) * np.sqrt(1.0 / m)
        P["b_out"] = np.zeros(c)
        # 壞 own-facet decoder (state → its own input again)
        P["W_dec"] = s(d, p) * np.sqrt(1.0 / d)
        P["b_dec"] = np.zeros(p)
        return P

    def zeros_like(self) -> "Params":
        g = Params()
        for k, v in self.items():
            g[k] = np.zeros_like(v)
        return g


# ---------------------------------------------------------------------------
# 2. The model: forward, loss, backward
# ---------------------------------------------------------------------------

@dataclass
class Cache:
    X: np.ndarray
    U: np.ndarray                 # facets (B,N,d)
    G: np.ndarray                 # power gate (N,N)
    A: np.ndarray                 # reflection weights (N,N)
    S: List[np.ndarray]           # states S_0..S_K, each (B,N,d)
    V: List[np.ndarray]           # V_t = S_{t-1} @ R_diff, t=1..K
    Wh: np.ndarray                # whole-subspace (B,N,m)
    O: np.ndarray                 # logits (B,N,c)
    Xh: np.ndarray                # own reconstruction (B,N,p)
    pi: np.ndarray                # per-jewel read-out weights (N,)


class FazangNet:
    """Indra-net reflective-equilibrium network (see module docstring)."""

    def __init__(self, cfg: Config, params: Optional[Params] = None,
                 rng: Optional[np.random.Generator] = None):
        self.cfg = cfg
        self.rng = rng if rng is not None else np.random.default_rng(cfg.seed)
        self.P = params if params is not None else Params.init(cfg, self.rng)
        N = cfg.n_jewels
        self.mask = 1.0 - np.eye(N)     # a jewel does not reflect itself

    # ---- 有力／無力 ------------------------------------------------------
    def power_gate(self, Lambda: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """G_jk = σ(Λ_jk − Λ_kj), zero on the diagonal.  Then G_jk + G_kj = 1:
        power over a pairing is conserved — when j is cause, k is condition."""
        anti = Lambda - Lambda.T
        G = sigmoid(anti) * self.mask
        A = G / (self.cfg.n_jewels - 1)          # bounded row sums
        return G, A

    # ---- forward ---------------------------------------------------------
    def forward(self, X: np.ndarray, principal: Optional[int] = None,
                k_reflect: Optional[int] = None) -> Cache:
        P, cfg = self.P, self.cfg
        K = cfg.k_reflect if k_reflect is None else k_reflect
        B, N, _ = X.shape
        # 別: own facet
        U = np.tanh(X @ P["W_facet"] + P["b_facet"])
        # 有力／無力
        G, A = self.power_gate(P["Lambda"])
        # 相入: reflective equilibrium, K passes of images-within-images
        S = [U]
        V = []
        for _ in range(K):
            Sprev = S[-1]
            Vt = Sprev @ P["R_diff"]                        # what others emit
            reflected = np.einsum("jk,bkd->bjd", A, Vt)     # what j receives
            Zt = U + Sprev @ P["S_same"] + reflected
            S.append(np.tanh(Zt))
            V.append(Vt)
        SK = S[-1]
        # 總: whole-subspace; 成: read-out; 壞: own reconstruction
        Wh = SK @ P["P_whole"]
        O = Wh @ P["W_out"] + P["b_out"]
        Xh = SK @ P["W_dec"] + P["b_dec"]
        # 主伴: which jewel(s) speak for the whole
        if principal is None:
            pi = np.full(N, 1.0 / N)
        else:
            pi = np.zeros(N); pi[principal] = 1.0
        return Cache(X, U, G, A, S, V, Wh, O, Xh, pi)

    # ---- 六相 loss ---------------------------------------------------------
    def loss(self, cache: Cache, y: np.ndarray) -> Tuple[float, Dict[str, float]]:
        cfg = self.cfg
        B, N, c = cache.O.shape
        m, p = cache.Wh.shape[-1], cache.X.shape[-1]
        prob = softmax(cache.O, axis=-1)                       # (B,N,c)
        logp = np.log(prob[np.arange(B)[:, None], np.arange(N)[None, :], y[:, None]] + 1e-12)
        L_ce = -(logp * cache.pi[None, :]).sum() / B            # 成 integration
        Wbar = cache.Wh.mean(axis=1, keepdims=True)
        L_same = cfg.lam_same * ((cache.Wh - Wbar) ** 2).sum() / (B * N * m)   # 同
        L_own = cfg.lam_own * ((cache.Xh - cache.X) ** 2).sum() / (B * N * p)  # 壞
        L = L_ce + L_same + L_own
        return L, {"ce": L_ce, "same": L_same, "own": L_own}

    # ---- backward (hand-derived) -------------------------------------------
    def backward(self, cache: Cache, y: np.ndarray) -> Params:
        P, cfg = self.P, self.cfg
        X, U, G, A, S, V, Wh, O, Xh, pi = (cache.X, cache.U, cache.G, cache.A,
                                            cache.S, cache.V, cache.Wh, cache.O,
                                            cache.Xh, cache.pi)
        B, N, c = O.shape
        m, p = Wh.shape[-1], X.shape[-1]
        K = len(V)
        g = P.zeros_like()

        # --- 成: cross-entropy on the (weighted) read-outs
        prob = softmax(O, axis=-1)
        dO = prob.copy()
        dO[np.arange(B)[:, None], np.arange(N)[None, :], y[:, None]] -= 1.0
        dO *= (pi[None, :, None] / B)
        g["W_out"] = np.einsum("bjm,bjc->mc", Wh, dO)
        g["b_out"] = dO.sum(axis=(0, 1))
        dWh = dO @ P["W_out"].T
        # --- 同: agreement on the whole
        Wbar = Wh.mean(axis=1, keepdims=True)
        dWh += (2.0 * cfg.lam_same / (B * N * m)) * (Wh - Wbar)
        SK = S[-1]
        g["P_whole"] = np.einsum("bjd,bjm->dm", SK, dWh)
        dSK = dWh @ P["P_whole"].T
        # --- 壞: own reconstruction
        dXh = (2.0 * cfg.lam_own / (B * N * p)) * (Xh - X)
        g["W_dec"] = np.einsum("bjd,bjp->dp", SK, dXh)
        g["b_dec"] = dXh.sum(axis=(0, 1))
        dSK += dXh @ P["W_dec"].T

        # --- 相入: back through the K reflections
        dU = np.zeros_like(U)
        dA = np.zeros_like(A)
        dS = dSK
        for t in range(K, 0, -1):
            St, Sprev, Vt = S[t], S[t - 1], V[t - 1]
            dZ = dS * (1.0 - St ** 2)
            dU += dZ
            g["S_same"] += np.einsum("bjd,bje->de", Sprev, dZ)
            dSprev = dZ @ P["S_same"].T
            dA += np.einsum("bjd,bkd->jk", dZ, Vt)
            dV = np.einsum("jk,bjd->bkd", A, dZ)
            g["R_diff"] += np.einsum("bkd,bke->de", Sprev, dV)
            dSprev += dV @ P["R_diff"].T
            dS = dSprev
        dU += dS                                    # S_0 = U

        # --- 有力／無力: gate → antisymmetric logits
        dG = dA / (N - 1)
        danti = dG * G * (1.0 - G) * self.mask
        g["Lambda"] = danti - danti.T

        # --- 別: facet encoder
        dZu = dU * (1.0 - U ** 2)
        g["W_facet"] = np.einsum("bnp,bnd->pd", X, dZu)
        g["b_facet"] = dZu.sum(axis=(0, 1))
        return g

    # ---- convenience ---------------------------------------------------------
    def predict_per_jewel(self, X: np.ndarray, k_reflect: Optional[int] = None) -> np.ndarray:
        """Class chosen by EACH jewel alone: (B,N)."""
        cache = self.forward(X, principal=None, k_reflect=k_reflect)
        return cache.O.argmax(axis=-1)

    def whole_images(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X).Wh


# ---------------------------------------------------------------------------
# 3. Adam optimiser (from scratch)
# ---------------------------------------------------------------------------

class Adam:
    def __init__(self, params: Params, lr: float = 3e-3, b1: float = 0.9,
                 b2: float = 0.999, eps: float = 1e-8, wd: float = 0.0):
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, b1, b2, eps, wd
        self.m = params.zeros_like()
        self.v = params.zeros_like()
        self.t = 0

    def step(self, params: Params, grads: Params) -> None:
        self.t += 1
        for k in params:
            gk = grads[k] + self.wd * params[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * gk
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * gk * gk
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)


# ---------------------------------------------------------------------------
# 4. Data — Task A "the golden lion", Task B "counting ten coins"
# ---------------------------------------------------------------------------

SHAPES = ["circle", "square", "triangle", "cross", "diamond", "ring"]


def _draw_shape(kind: str, size: int = 12) -> np.ndarray:
    """A crisp 12x12 binary glyph.  The 'lion' of gold: a form with parts."""
    img = np.zeros((size, size))
    cx = cy = (size - 1) / 2.0
    yy, xx = np.mgrid[0:size, 0:size]
    if kind == "circle":
        img[(xx - cx) ** 2 + (yy - cy) ** 2 <= 4.2 ** 2] = 1
    elif kind == "ring":
        r2 = (xx - cx) ** 2 + (yy - cy) ** 2
        img[(r2 <= 4.6 ** 2) & (r2 >= 2.6 ** 2)] = 1
    elif kind == "square":
        img[2:10, 2:10] = 1
    elif kind == "diamond":
        img[np.abs(xx - cx) + np.abs(yy - cy) <= 4.6] = 1
    elif kind == "cross":
        img[4:8, 1:11] = 1
        img[1:11, 4:8] = 1
    elif kind == "triangle":
        for r in range(2, 11):
            half = (r - 2) * 0.55
            img[r, int(round(cx - half)):int(round(cx + half)) + 1] = 1
    return img


def make_lion_batch(rng: np.random.Generator, B: int, noise: float = 0.05
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """Returns X (B, 9, 16): nine 4x4 facets of a jittered 12x12 glyph;
    y (B,) the class of the WHOLE glyph.  Each facet is one jewel's input."""
    X = np.zeros((B, 9, 16))
    y = rng.integers(0, len(SHAPES), size=B)
    for b in range(B):
        img = _draw_shape(SHAPES[y[b]])
        dx, dy = rng.integers(-1, 2, size=2)
        img = np.roll(np.roll(img, dx, axis=1), dy, axis=0)
        flip = rng.random(img.shape) < noise
        img = np.where(flip, 1 - img, img)
        img = img * 2.0 - 1.0                        # ±1 pixels
        k = 0
        for gy in range(3):
            for gx in range(3):
                X[b, k] = img[gy * 4:(gy + 1) * 4, gx * 4:(gx + 1) * 4].reshape(-1)
                k += 1
    return X, y


def make_coin_batch(rng: np.random.Generator, B: int) -> Tuple[np.ndarray, np.ndarray]:
    """Ten jewels, each holding one coin (1) or nothing (0).  Input of jewel j
    is [coin_j, position_j/9]: it knows its own coin and its place in the
    count; it must state the TOTAL of all ten (11 classes)."""
    N = 10
    present = (rng.random((B, N)) < rng.random((B, 1))).astype(float)
    pos = np.tile(np.arange(N) / (N - 1.0), (B, 1))
    X = np.stack([present * 2 - 1, pos * 2 - 1], axis=-1)   # (B,10,2)
    y = present.sum(axis=1).astype(int)
    return X, y


# ---------------------------------------------------------------------------
# 5. Finite-difference gradient check (MANDATORY on every run)
# ---------------------------------------------------------------------------

def gradient_check(verbose: bool = True) -> float:
    cfg = Config(n_jewels=4, p_in=3, d=5, m=4, c=3, k_reflect=2,
                 lam_same=0.7, lam_own=0.4, seed=7)
    rng = np.random.default_rng(cfg.seed)
    net = FazangNet(cfg, rng=rng)
    X = rng.normal(size=(3, cfg.n_jewels, cfg.p_in))
    y = rng.integers(0, cfg.c, size=3)
    principal = 2
    cache = net.forward(X, principal=principal)
    grads = net.backward(cache, y)
    eps = 1e-6
    worst = 0.0
    for k, W in net.P.items():
        num = np.zeros_like(W)
        it = np.nditer(W, flags=["multi_index"], op_flags=["readwrite"])
        while not it.finished:
            idx = it.multi_index
            old = W[idx]
            W[idx] = old + eps; Lp, _ = net.loss(net.forward(X, principal), y)
            W[idx] = old - eps; Lm, _ = net.loss(net.forward(X, principal), y)
            W[idx] = old
            num[idx] = (Lp - Lm) / (2 * eps)
            it.iternext()
        # antisymmetric parametrisation: only Λ−Λᵀ matters, compare raw grads
        denom = np.abs(num) + np.abs(grads[k]) + 1e-10
        rel = np.max(np.abs(num - grads[k]) / denom)
        worst = max(worst, rel)
        if verbose:
            print(f"  gradcheck {k:9s} shape={str(W.shape):10s} max-rel-err={rel:.2e}")
    if verbose:
        print(f"  gradcheck WORST = {worst:.2e}  (pass if < 1e-6)")
    assert worst < 1e-6, f"gradient check FAILED: {worst}"
    return worst


# ---------------------------------------------------------------------------
# 6. Training loop
# ---------------------------------------------------------------------------

def train(net: FazangNet, batch_fn, steps: int, batch: int, lr: float,
          rng: np.random.Generator, log_every: int = 100, label: str = "") -> List[float]:
    opt = Adam(net.P, lr=lr, wd=1e-5)
    N = net.cfg.n_jewels
    hist = []
    t0 = time.time()
    for step in range(1, steps + 1):
        X, y = batch_fn(rng, batch)
        # 主伴: rotate the principal — this step one jewel speaks for the whole
        principal = int(rng.integers(0, N)) if net.cfg.principal_mode == "rotate" else None
        cache = net.forward(X, principal=principal)
        L, parts = net.loss(cache, y)
        grads = net.backward(cache, y)
        # gradient clipping (global norm) for stability of the recurrence
        gn = np.sqrt(sum((g ** 2).sum() for g in grads.values()))
        if gn > 5.0:
            for k in grads: grads[k] *= 5.0 / gn
        opt.step(net.P, grads)
        hist.append(L)
        if step % log_every == 0 or step == 1:
            acc = (cache.O.argmax(-1) == y[:, None]).mean()
            print(f"  [{label}] step {step:5d}  loss {L:.4f}  ce {parts['ce']:.4f}"
                  f"  same {parts['same']:.4f}  own {parts['own']:.4f}"
                  f"  all-jewel acc {acc:.3f}  ({time.time()-t0:.1f}s)")
    return hist


# ---------------------------------------------------------------------------
# 7. Self-tests: the doctrine, measured
# ---------------------------------------------------------------------------

def per_jewel_accuracy(net: FazangNet, X: np.ndarray, y: np.ndarray,
                       k_reflect: Optional[int] = None) -> np.ndarray:
    pred = net.predict_per_jewel(X, k_reflect=k_reflect)       # (B,N)
    return (pred == y[:, None]).mean(axis=0)                   # (N,)


def agreement_rate(net: FazangNet, X: np.ndarray) -> float:
    pred = net.predict_per_jewel(X)
    return (pred == pred[:, :1]).all(axis=1).mean()


def total_cause_matrix(net: FazangNet, X: np.ndarray) -> np.ndarray:
    """D[j,k] = mean |ΔWhole_j| when jewel k's INPUT is erased.
    Fazang's 全力: every part is a total cause of the whole seen anywhere,
    so no off-diagonal entry may be zero."""
    N = net.cfg.n_jewels
    base = net.whole_images(X)
    D = np.zeros((N, N))
    for k in range(N):
        Xk = X.copy(); Xk[:, k, :] = 0.0
        alt = net.whole_images(Xk)
        D[:, k] = np.abs(alt - base).mean(axis=(0, 2))
    return D


def own_retention(net: FazangNet, X: np.ndarray) -> Tuple[float, float]:
    cache = net.forward(X)
    mse = ((cache.Xh - X) ** 2).mean()
    base = ((X - X.mean(axis=0, keepdims=True)) ** 2).mean()
    return mse, base


def occlusion_curve(net: FazangNet, X: np.ndarray, y: np.ndarray,
                    rng: np.random.Generator) -> List[Tuple[int, float]]:
    N = net.cfg.n_jewels
    out = []
    for n_occ in range(0, N - 1):
        Xo = X.copy()
        for b in range(X.shape[0]):
            idx = rng.choice(N, size=n_occ, replace=False)
            Xo[b, idx, :] = 0.0
        acc = per_jewel_accuracy(net, Xo, y).mean()
        out.append((n_occ, acc))
    return out


def print_matrix(M: np.ndarray, title: str, fmt: str = "{:6.3f}") -> None:
    print(f"  {title}")
    for row in M:
        print("    " + " ".join(fmt.format(v) for v in row))


# ---------------------------------------------------------------------------
# 8. Main
# ---------------------------------------------------------------------------

def run_task_lion(quick: bool, seed: int) -> Dict[str, float]:
    print("\n=== TASK A · 金獅子 THE GOLDEN LION  (9 jewels × 4x4 facets → whole shape) ===")
    cfg = Config(n_jewels=9, p_in=16, d=32, m=16, c=len(SHAPES), k_reflect=3,
                 lam_same=0.5, lam_own=0.5, principal_mode="rotate", seed=seed)
    rng = np.random.default_rng(seed)
    net = FazangNet(cfg, rng=rng)
    steps = 500 if quick else 1800
    train(net, make_lion_batch, steps=steps, batch=64, lr=3e-3, rng=rng,
          log_every=max(1, steps // 6), label="lion")

    Xt, yt = make_lion_batch(np.random.default_rng(seed + 1), 600)
    res: Dict[str, float] = {}

    # (1) 主伴 / holography: every jewel alone names the whole
    accK = per_jewel_accuracy(net, Xt, yt)
    acc0 = per_jewel_accuracy(net, Xt, yt, k_reflect=0)
    print("\n  [holography] per-jewel accuracy with K=3 reflections:")
    print("    " + "  ".join(f"j{j}:{a:.2f}" for j, a in enumerate(accK)))
    print("  [holography] per-jewel accuracy with K=0 (facet only, no reflection):")
    print("    " + "  ".join(f"j{j}:{a:.2f}" for j, a in enumerate(acc0)))
    corners = [0, 2, 6, 8]
    print(f"  corner jewels  K=3 mean {accK[corners].mean():.3f}   K=0 mean {acc0[corners].mean():.3f}")
    agr = agreement_rate(net, Xt)
    print(f"  [同 sameness] all nine jewels agree on the class in {agr*100:.1f}% of samples")
    res.update(acc_K3_min=float(accK.min()), acc_K0_min=float(acc0.min()),
               acc_K3_corner=float(accK[corners].mean()), acc_K0_corner=float(acc0[corners].mean()),
               agreement=float(agr))

    # (2) reflection-depth sweep — images of images (重重)
    print("  [重重 depth sweep] mean per-jewel accuracy vs reflection depth at test time:")
    for K in range(0, 7):
        a = per_jewel_accuracy(net, Xt, yt, k_reflect=K).mean()
        print(f"    K={K}: {a:.3f}" + ("   <- trained depth" if K == cfg.k_reflect else ""))

    # (3) 全力 total cause matrix
    D = total_cause_matrix(net, Xt[:200])
    print_matrix(D / D.max(), "[全力 total-cause matrix] row j = whole seen at jewel j, col k = jewel k erased (normalised):")
    off = D[~np.eye(9, dtype=bool)]
    print(f"  smallest off-diagonal sensitivity = {off.min():.4f}  (must be > 0: every part moves every whole)")
    res["min_offdiag_sens"] = float(off.min())

    # (4) 壞 retention of the own facet
    mse, base = own_retention(net, Xt)
    print(f"  [壞 retention] own-facet reconstruction MSE {mse:.4f} vs variance baseline {base:.4f}")
    res.update(own_mse=float(mse), own_base=float(base))

    # (5) 有力／無力 power heterarchy
    G, _ = net.power_gate(net.P["Lambda"])
    sym = np.abs((G + G.T)[~np.eye(9, dtype=bool)] - 1.0).max()
    rowpow = G.sum(axis=1)
    print_matrix(G, "[有力／無力 power gate G] G[j,k] = power of j as cause over k (G+Gᵀ = 1 off-diagonal):")
    print(f"  max |G+Gᵀ−1| off-diagonal = {sym:.2e};  total power per jewel: "
          + " ".join(f"{v:.2f}" for v in rowpow)
          + f"  dominance index max/mean = {rowpow.max()/rowpow.mean():.3f}")
    res.update(power_conservation=float(sym), dominance=float(rowpow.max() / rowpow.mean()))

    # (6) occlusion — mutual inclusion carries the missing facets
    print("  [相入 occlusion] mean per-jewel accuracy when n random facets are erased at test time:")
    for n_occ, a in occlusion_curve(net, Xt[:300], yt[:300], np.random.default_rng(seed + 2)):
        print(f"    erased {n_occ}: {a:.3f}")
    return res


def run_task_coins(quick: bool, seed: int) -> Dict[str, float]:
    print("\n=== TASK B · 數十錢 COUNTING TEN COINS  (10 jewels, each states the total) ===")
    cfg = Config(n_jewels=10, p_in=2, d=24, m=12, c=11, k_reflect=3,
                 lam_same=0.3, lam_own=0.2, principal_mode="rotate", seed=seed + 10)
    rng = np.random.default_rng(seed + 10)
    net = FazangNet(cfg, rng=rng)
    steps = 500 if quick else 1500
    train(net, make_coin_batch, steps=steps, batch=64, lr=4e-3, rng=rng,
          log_every=max(1, steps // 5), label="coins")
    Xt, yt = make_coin_batch(np.random.default_rng(seed + 11), 800)
    accK = per_jewel_accuracy(net, Xt, yt)
    acc0 = per_jewel_accuracy(net, Xt, yt, k_reflect=0)
    print("  [one is ten] per-jewel accuracy on the TOTAL, K=3: "
          + " ".join(f"{a:.2f}" for a in accK))
    print("  [one is ten] per-jewel accuracy on the TOTAL, K=0: "
          + " ".join(f"{a:.2f}" for a in acc0)
          + f"   (chance ≈ {1/11:.2f})")
    print(f"  [同] all ten jewels agree on the count in {agreement_rate(net, Xt)*100:.1f}% of samples")
    # counting up vs counting down: the "first" coin and the "tenth" coin must both hold the ten
    print(f"  first coin (j0) {accK[0]:.3f}   tenth coin (j9) {accK[9]:.3f}")
    # exactness: mean absolute error of the count from each jewel
    pred = net.predict_per_jewel(Xt)
    mae = np.abs(pred - yt[:, None]).mean(axis=0)
    print("  mean |count error| per jewel: " + " ".join(f"{v:.2f}" for v in mae))
    return {"coins_acc_min": float(accK.min()), "coins_acc0_max": float(acc0.max()),
            "coins_agreement": float(agreement_rate(net, Xt))}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    ap.add_argument("--quick", action="store_true", help="shorter training")
    ap.add_argument("--gradcheck-only", action="store_true")
    ap.add_argument("--seed", type=int, default=190)
    args = ap.parse_args(argv)
    np.set_printoptions(precision=3, suppress=True)
    t0 = time.time()
    print("=" * 78)
    print("0190 · FAZANG 法藏 (643–712) · Indra-net reflective-equilibrium neuron")
    print("=" * 78)

    print("\n=== GRADIENT CHECK (finite differences, float64) ===")
    gradient_check(verbose=True)
    if args.gradcheck_only:
        return 0

    a = run_task_lion(args.quick, args.seed)
    b = run_task_coins(args.quick, args.seed)

    print("\n=== SELF-TEST VERDICTS ===")
    checks = [
        ("gradient check < 1e-6", True),
        ("power conserved: |G+Gᵀ−1| < 1e-9", a["power_conservation"] < 1e-9),
        ("holography: every lion jewel ≥ 0.70 alone (K=3)", a["acc_K3_min"] >= 0.70),
        ("reflection matters: corner jewels K=3 − K=0 ≥ 0.15", a["acc_K3_corner"] - a["acc_K0_corner"] >= 0.15),
        ("agreement on the whole ≥ 0.80", a["agreement"] >= 0.80),
        ("total cause: no zero off-diagonal sensitivity", a["min_offdiag_sens"] > 0.0),
        ("retention: own MSE < 0.5 × baseline", a["own_mse"] < 0.5 * a["own_base"]),
        ("one is ten: every coin jewel ≥ 0.85 (K=3)", b["coins_acc_min"] >= 0.85),
        ("coins need reflection: best jewel at K=0 ≤ 0.45", b["coins_acc0_max"] <= 0.45),
    ]
    ok = True
    for name, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        ok &= bool(passed)
    print(f"\nTotal wall time {time.time()-t0:.1f}s")
    print("ALL SELF-TESTS PASSED" if ok else "SOME SELF-TESTS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
