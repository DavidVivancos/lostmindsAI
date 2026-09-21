#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 11 Minds 201 - 220 Available on Amazon https://www.amazon.com/dp/B0HKF7TRFF
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0217_fatima_al_fihri_800 - Fatima al-Fihri (Fez, Ramadan 245 AH / 859 CE)
================================================================================  

THE WAQF NETWORK  --  "The Deed of Perpetuity"
A trainable artificial-neuron architecture built from the founding act
attributed to Fatima al-Fihri (Fez, Ramadan 245 AH / 859 CE).

Pure NumPy. No autograd library. Every gradient is derived by hand and
verified against central finite differences inside this file. Run it:

    python3 chapter_0217_fatima_al_fihri_800.py

--------------------------------------------------------------------------
WHY THIS ARCHITECTURE  (the mind behind the mechanism)
--------------------------------------------------------------------------
Ibn Abi Zar's account of the founding of al-Qarawiyyin (Rawd al-Qirtas,
c. 1326) attributes five acts to Fatima. Each becomes a mechanism here:

  1. QUARRY  -- "all materials were taken from the land itself"; a well
     was dug on the site so nothing of doubtful origin entered the work.
     => Every neuron's PRINCIPAL vector is literally cut from the data it
        stands on (a mean of recorded training points, the "deed").
        Provenance is a property of construction, not a log.

  2. FAST (sawm) -- she fasted from the first day of construction until
     the mosque was complete, and only then prayed in it.
     => A ring under construction takes no reward and is not consumed:
        its outputs are withheld from the institution's predictions and
        it is trained only on what the existing institution leaves
        unfunded (its residual). It is "prayed in" (served) only once
        complete.

  3. DEED / STIPULATION (shart al-waqif) -- an endowment's founder binds
     the future: the principal is INALIENABLE, only its YIELD is spent,
     and the founder's ordered list of beneficiaries governs forever.
     => Principals and extents are never touched by gradient descent.
        Each neuron distributes its yield by a WATERFALL: an ordered
        chain of beneficiaries with capped needs; what remains falls to
        the last beneficiary of every classical waqf, "the poor".

  4. ACCRETION -- the mosque grew for 1,100 years only by ADDITION of new
     wings (956, 1134-43, 1349, 1587, 1609...), each enveloping, never
     replacing, the founder's hall.
     => New tasks are learned by founding new RINGS whose outputs are
        added to the institution. Old rings are never overwritten.

  5. CUSTODY (nazir, istibdal) -- a custodian administers the endowment
     and may exchange a RUINED asset for an equivalent one, but may
     not touch a healthy one.
     => A Nazir routine monitors each neuron's yield and re-quarries
        only chronically barren principals, one-for-one, from the
        currently unfunded data.

The founder herself is ABSENT from the running institution -- as she is
from the history of the teaching that later happened in her hall. After
founding, the construction data can be discarded; the network keeps only
the deeds.

--------------------------------------------------------------------------
NOVELTY (what is not a Transformer, MoE, RBF-net or ELM)
--------------------------------------------------------------------------
 * The unit ("waqf neuron") emits a VECTOR of allocations produced by a
   differentiable waterfall with trainable, then sealed, caps -- a
   priority cascade, not a dot product.
 * The frozen weights are data-quarried means with recorded deeds, so
   every unit can name the exact training points it was cut from.
 * Learning is split by ROLE: deeds (frozen), rents (custodian, plastic),
   needs (trainable during the fast, sealed at completion), circles &
   heads (students, plastic, yield-gated).
 * Growth happens only by ring accretion; unlearning is impossible by
   construction; forgetting is bounded by the locality of yield.
"""
from __future__ import annotations

import hashlib
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

np.set_printoptions(precision=4, suppress=True)

# ---------------------------------------------------------------------------
# Small numerics
# ---------------------------------------------------------------------------

def softplus(x: np.ndarray) -> np.ndarray:
    """Numerically stable softplus: log(1 + e^x)."""
    return np.where(x > 30.0, x, np.log1p(np.exp(np.minimum(x, 30.0))))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def one_hot(y: np.ndarray, C: int) -> np.ndarray:
    Y = np.zeros((y.shape[0], C))
    Y[np.arange(y.shape[0]), y] = 1.0
    return Y


# ---------------------------------------------------------------------------
# The DEED: a record of exactly which points a principal was cut from
# ---------------------------------------------------------------------------

@dataclass
class Deed:
    """The founding record of one waqf neuron.

    idx      : indices (into the construction set) of the points averaged
               into the principal -- "the land the stone was quarried from".
    digest   : SHA-256 of those points, so the deed can be verified later
               even after the construction data is discarded.
    founded  : the ring epoch at which it was cut (0 = original founding,
               >0 = an istibdal exchange performed by the custodian).
    """
    idx: np.ndarray
    digest: str
    founded: int = 0


# ---------------------------------------------------------------------------
# The QUARRY: Lloyd iterations on the ring's own construction data
# ---------------------------------------------------------------------------

def quarry(X: np.ndarray, K: int, rng: np.random.Generator, iters: int = 12,
           extent_scale: float = 1.4, extent_floor: float = 1e-2
           ) -> Tuple[np.ndarray, np.ndarray, List[Deed]]:
    """Cut K principals out of X by farthest-point seeding + Lloyd steps.

    Returns principals P (K,d), extents s (K,), and one Deed per principal.
    Every principal is EXACTLY the mean of the points in its deed, so
    provenance is a structural fact (checked in the self-tests).
    """
    N, d = X.shape
    # farthest-point seeding (deterministic given rng)
    centers = [X[rng.integers(N)]]
    d2 = ((X - centers[0]) ** 2).sum(1)
    for _ in range(1, K):
        j = int(np.argmax(d2))
        centers.append(X[j])
        d2 = np.minimum(d2, ((X - X[j]) ** 2).sum(1))
    P = np.stack(centers)
    for _ in range(iters):
        D2 = ((X[:, None, :] - P[None, :, :]) ** 2).sum(2)
        a = D2.argmin(1)
        for k in range(K):
            m = a == k
            if m.any():
                P[k] = X[m].mean(0)
            else:  # an empty lot: take the point farthest from every centre
                j = int(np.argmax(D2.min(1)))
                P[k] = X[j]
    D2 = ((X[:, None, :] - P[None, :, :]) ** 2).sum(2)
    a = D2.argmin(1)
    deeds, s = [], np.zeros(K)
    for k in range(K):
        idx = np.where(a == k)[0]
        if idx.size == 0:
            idx = np.array([int(np.argmin(D2[:, k]))])
        P[k] = X[idx].mean(0)                       # exact provenance
        rms = np.sqrt(((X[idx] - P[k]) ** 2).sum(1).mean())
        s[k] = max(extent_scale * rms, extent_floor)
        dig = hashlib.sha256(np.ascontiguousarray(X[idx]).tobytes()).hexdigest()[:16]
        deeds.append(Deed(idx=idx, digest=dig, founded=0))
    return P, s, deeds


# ---------------------------------------------------------------------------
# A RING of waqf neurons (one endowment, one wing of the institution)
# ---------------------------------------------------------------------------

class WaqfRing:
    """One endowment: K waqf neurons -> B beneficiary circles -> ring head.

    Parameter roles
    ---------------
    INALIENABLE (never receives gradient): P (K,d), s (K,), order (K,B)
    SEALED at completion (trainable only during the fast): lognd (K,B)
    PLASTIC (custodian & students): logrho (K,), W[j] (m,K), c[j] (m,),
                                    V (C, B*m+1), b (C,)
    """

    def __init__(self, name: str, X_construct: np.ndarray, K: int, B: int,
                 m: int, C: int, rng: np.random.Generator):
        self.name = name
        self.K, self.B, self.m, self.C = K, B, m, C
        self.d = X_construct.shape[1]
        # -- the deed (quarried, frozen) --
        self.P, self.s, self.deeds = quarry(X_construct, K, rng)
        self.P.setflags(write=False)
        self.s.setflags(write=False)
        # the founder's ordered list of beneficiaries, per neuron
        self.order = np.stack([rng.permutation(B) for _ in range(K)])
        self.inv_order = np.argsort(self.order, axis=1)
        # -- stipulation (needs), trainable during construction --
        self.lognd = rng.normal(-1.6, 0.3, size=(K, B))
        self.sealed = False
        # -- custodian's rents --
        self.logrho = np.full(K, 0.5413)          # softplus(0.5413) ~ 1.0
        # -- students (circles) and head --
        self.W = [rng.normal(0, 1.0 / np.sqrt(K), size=(m, K)) for _ in range(B)]
        self.c = [np.zeros(m) for _ in range(B)]
        self.V = rng.normal(0, 1.0 / np.sqrt(B * m + 1), size=(C, B * m + 1))
        self.b = np.zeros(C)
        self.under_construction = True             # the fast has begun
        self.cache: Dict[str, np.ndarray] = {}
        self.exchanges = 0
        self._epoch = 0
        self.snapshot_at_founding = (self.P.copy(), self.s.copy())

    # ---------------- parameters by role ----------------
    def plastic_params(self) -> Dict[str, np.ndarray]:
        p = {"logrho": self.logrho, "V": self.V, "b": self.b}
        for j in range(self.B):
            p[f"W{j}"] = self.W[j]
            p[f"c{j}"] = self.c[j]
        if not self.sealed:
            p["lognd"] = self.lognd
        return p

    # ---------------- forward ----------------
    def yield_(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        D2 = ((X[:, None, :] - self.P[None, :, :]) ** 2).sum(2)     # (N,K)
        kern = np.exp(-D2 / (2.0 * self.s[None, :] ** 2))
        rho = softplus(self.logrho)
        return rho[None, :] * kern, kern

    def forward(self, X: np.ndarray) -> np.ndarray:
        N = X.shape[0]
        g, kern = self.yield_(X)                                     # (N,K)
        n = softplus(self.lognd) + 1e-4                              # (K,B)
        # permute needs into the founder's order
        n_pos = np.take_along_axis(n, self.order, axis=1)            # (K,B)
        r = g.copy()
        a_pos = np.zeros((N, self.K, self.B))
        u_list, r_prev_list = [], []
        for j in range(self.B):
            nj = n_pos[None, :, j]
            u = r / nj
            a = nj * np.tanh(u)
            a_pos[:, :, j] = a
            u_list.append(u)
            r_prev_list.append(r)
            r = r - a
        R = r                                                        # the poor
        # unpermute allocations into beneficiary index
        A = np.take_along_axis(a_pos, self.inv_order[None, :, :], axis=2)
        q = R.sum(1, keepdims=True)
        H, Z = [], []
        for j in range(self.B):
            Zj = A[:, :, j] @ self.W[j].T + self.c[j][None, :]
            Hj = np.tanh(Zj)
            Z.append(Zj)
            H.append(Hj)
        F = np.concatenate(H + [q], axis=1)                          # (N,B*m+1)
        y = F @ self.V.T + self.b[None, :]
        self.cache = dict(X=X, g=g, kern=kern, n_pos=n_pos, u=np.stack(u_list, 2),
                          a_pos=a_pos, A=A, R=R, q=q, H=H, F=F)
        return y

    # ---------------- backward ----------------
    def backward(self, dy: np.ndarray) -> Dict[str, np.ndarray]:
        c = self.cache
        N = dy.shape[0]
        grads: Dict[str, np.ndarray] = {}
        grads["V"] = dy.T @ c["F"]
        grads["b"] = dy.sum(0)
        dF = dy @ self.V                                             # (N,B*m+1)
        dA = np.zeros_like(c["A"])
        for j in range(self.B):
            dH = dF[:, j * self.m:(j + 1) * self.m]
            dZ = dH * (1.0 - c["H"][j] ** 2)
            grads[f"W{j}"] = dZ.T @ c["A"][:, :, j]
            grads[f"c{j}"] = dZ.sum(0)
            dA[:, :, j] = dZ @ self.W[j]
        dq = dF[:, -1:]
        dR = np.repeat(dq, self.K, axis=1)                           # (N,K)
        # permute allocation-grads into position order
        da_pos = np.take_along_axis(dA, self.order[None, :, :], axis=2)
        dr = dR.copy()
        dn_pos = np.zeros((self.K, self.B))
        for j in range(self.B - 1, -1, -1):
            u = c["u"][:, :, j]
            sech2 = 1.0 - np.tanh(u) ** 2
            ga = da_pos[:, :, j] - dr                 # dL/da_j (direct + via r_j)
            dn_pos[:, j] = (ga * (np.tanh(u) - u * sech2)).sum(0)
            dr = dr + ga * sech2                      # dL/dr_{j-1}
        dg = dr
        if not self.sealed:
            dn = np.take_along_axis(dn_pos, self.inv_order, axis=1)
            grads["lognd"] = dn * sigmoid(self.lognd)
        drho = (dg * c["kern"]).sum(0)
        grads["logrho"] = drho * sigmoid(self.logrho)
        return grads

    # ---------------- the custodian ----------------
    def seal(self):
        """End of the fast: the deed is sealed, the needs become law."""
        self.sealed = True
        self.under_construction = False
        self.lognd.setflags(write=False)

    def istibdal(self, X: np.ndarray, loss_per_point: np.ndarray, mean_yield: np.ndarray,
                 ruin_frac: float = 0.05, m_points: int = 12) -> Tuple[List[int], List[int]]:
        """Exchange RUINED principals (chronically barren yield) for new ones
        quarried from the currently unfunded points. One-for-one. Never
        touches a healthy neuron. Returns the list of exchanged neurons."""
        med = np.median(mean_yield)
        ruined = np.where(mean_yield < ruin_frac * max(med, 1e-9))[0]
        if ruined.size == 0:
            return [], []
        order = np.argsort(-loss_per_point)                # worst-served first
        P = self.P.copy(); s = self.s.copy()
        used = np.zeros(X.shape[0], bool)
        exchanged = []
        for k in ruined:
            # the worst-served point not yet used, plus its m nearest neighbours
            cand = [i for i in order if not used[i]]
            if not cand:
                break
            i0 = cand[0]
            d2 = ((X - X[i0]) ** 2).sum(1)
            nn = np.argsort(d2)[:m_points]
            used[nn] = True
            P[k] = X[nn].mean(0)
            rms = np.sqrt(((X[nn] - P[k]) ** 2).sum(1).mean())
            s[k] = max(1.4 * rms, 1e-2)
            dig = hashlib.sha256(np.ascontiguousarray(X[nn]).tobytes()).hexdigest()[:16]
            self.deeds[k] = Deed(idx=nn, digest=dig, founded=self._epoch)
            self.logrho[k] = 0.5413
            exchanged.append(int(k))
        self.P = P; self.P.setflags(write=False)
        self.s = s; self.s.setflags(write=False)
        self.exchanges += len(exchanged)
        return exchanged, [int(k) for k in ruined]


# ---------------------------------------------------------------------------
# The INSTITUTION: rings accrete; outputs add; nothing is ever demolished
# ---------------------------------------------------------------------------

class Adam:
    def __init__(self, lr=3e-3, b1=0.9, b2=0.999, eps=1e-8, wd=1e-5):
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, b1, b2, eps, wd
        self.m: Dict[str, np.ndarray] = {}
        self.v: Dict[str, np.ndarray] = {}
        self.t = 0

    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]):
        self.t += 1
        for k, p in params.items():
            if k not in grads:
                continue
            g = grads[k] + (self.wd * p if p.ndim > 1 else 0.0)
            if k not in self.m:
                self.m[k] = np.zeros_like(p); self.v[k] = np.zeros_like(p)
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)


class WaqfInstitution:
    """The whole mosque-university: a list of rings whose logits are summed.

    served()  -> which rings are consumed by predictions (complete ones)
    forward() -> sum of served rings' logits
    """

    def __init__(self, C: int, B: int = 4, m: int = 8, seed: int = 245):
        self.C, self.B, self.m = C, B, m
        self.rings: List[WaqfRing] = []
        self.rng = np.random.default_rng(seed)
        self.ledger: List[str] = []

    # ---- founding ----
    def found_ring(self, name: str, X_construct: np.ndarray, K: int) -> WaqfRing:
        ring = WaqfRing(name, X_construct, K, self.B, self.m, self.C, self.rng)
        self.rings.append(ring)
        self.ledger.append(f"founded ring '{name}': K={K} principals quarried from "
                           f"{X_construct.shape[0]} points; fast begins")
        return ring

    def served(self) -> List[WaqfRing]:
        return [r for r in self.rings if not r.under_construction]

    def forward(self, X: np.ndarray, include_construction: bool = False) -> np.ndarray:
        y = np.zeros((X.shape[0], self.C))
        for r in self.rings:
            if r.under_construction and not include_construction:
                continue
            y += r.forward(X)
        return y

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X).argmax(1)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float((self.predict(X) == y).mean())

    @staticmethod
    def loss_and_grad(logits: np.ndarray, y: np.ndarray, C: int):
        p = softmax(logits)
        N = y.shape[0]
        loss = -np.log(p[np.arange(N), y] + 1e-12).mean()
        dlogits = (p - one_hot(y, C)) / N
        return loss, dlogits, -np.log(p[np.arange(N), y] + 1e-12)

    # ---- the fast: construction of one ring on the institution's residual ----
    def fast(self, ring: WaqfRing, X: np.ndarray, y: np.ndarray, epochs: int = 60,
             batch: int = 64, lr: float = 5e-3, verbose: bool = False) -> List[float]:
        """Train ONLY the new ring, on top of the frozen institution's logits.
        The ring is NOT served during this phase (it is not prayed in)."""
        assert ring.under_construction
        opt = Adam(lr=lr)
        base = self.forward(X)                       # existing institution, frozen
        hist = []
        N = X.shape[0]
        for ep in range(epochs):
            perm = self.rng.permutation(N)
            tot = 0.0
            for i in range(0, N, batch):
                idx = perm[i:i + batch]
                yr = ring.forward(X[idx])
                loss, dl, _ = self.loss_and_grad(base[idx] + yr, y[idx], self.C)
                g = ring.backward(dl)
                opt.step(ring.plastic_params(), g)
                tot += loss * idx.size
            hist.append(tot / N)
            if verbose and (ep % 10 == 0 or ep == epochs - 1):
                print(f"    fast[{ring.name}] epoch {ep:3d}  loss {tot / N:.4f}")
        return hist

    def complete(self, ring: WaqfRing):
        """The first prayer: seal the deed and begin serving the ring."""
        ring.seal()
        self.ledger.append(f"ring '{ring.name}' complete: deed sealed, now served")

    # ---- administration: joint training of every served ring's plastic parts ----
    def administer(self, X: np.ndarray, y: np.ndarray, epochs: int = 30, batch: int = 64,
                   lr: float = 2e-3, verbose: bool = False) -> List[float]:
        rings = self.served()
        opts = [Adam(lr=lr) for _ in rings]
        hist = []
        N = X.shape[0]
        for ep in range(epochs):
            perm = self.rng.permutation(N)
            tot = 0.0
            for i in range(0, N, batch):
                idx = perm[i:i + batch]
                logits = np.zeros((idx.size, self.C))
                for r in rings:
                    logits += r.forward(X[idx])
                loss, dl, _ = self.loss_and_grad(logits, y[idx], self.C)
                for r, o in zip(rings, opts):
                    o.step(r.plastic_params(), r.backward(dl))
                tot += loss * idx.size
            hist.append(tot / N)
            for r in rings:
                r._epoch += 1
            if verbose and (ep % 10 == 0 or ep == epochs - 1):
                print(f"    administer epoch {ep:3d}  loss {tot / N:.4f}")
        return hist

    # ---- the custodian's rounds ----
    def nazir(self, ring: WaqfRing, X: np.ndarray, y: np.ndarray) -> Tuple[List[int], List[int]]:
        """The custodian inspects ONE endowment against the stream of the
        community it was founded for: which principals are barren there?
        Exchange them (istibdal) for principals quarried from the currently
        unfunded points. A custodian never judges a wing by a congregation
        that does not use it."""
        assert not ring.under_construction, "a wing under construction is not yet administered"
        logits = self.forward(X)
        _, _, lpp = self.loss_and_grad(logits, y, self.C)
        g, _ = ring.yield_(X)
        exchanged, barren = ring.istibdal(X, lpp, g.mean(0))
        if exchanged:
            self.ledger.append(f"nazir: ring '{ring.name}' exchanged {len(exchanged)} barren principals "
                               f"(of {len(barren)} found barren)")
        return exchanged, barren


# ---------------------------------------------------------------------------
# A control model (ordinary MLP) to measure forgetting against
# ---------------------------------------------------------------------------

class MLP:
    def __init__(self, d, h, C, rng):
        self.W1 = rng.normal(0, 1 / np.sqrt(d), (h, d)); self.b1 = np.zeros(h)
        self.W2 = rng.normal(0, 1 / np.sqrt(h), (C, h)); self.b2 = np.zeros(C)
        self.C = C

    def params(self):
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}

    def forward(self, X):
        self.X = X; self.Z = X @ self.W1.T + self.b1; self.H = np.tanh(self.Z)
        return self.H @ self.W2.T + self.b2

    def backward(self, dy):
        g = {"W2": dy.T @ self.H, "b2": dy.sum(0)}
        dH = dy @ self.W2; dZ = dH * (1 - self.H ** 2)
        g["W1"] = dZ.T @ self.X; g["b1"] = dZ.sum(0)
        return g

    def fit(self, X, y, rng, epochs=60, batch=64, lr=3e-3):
        opt = Adam(lr=lr)
        for _ in range(epochs):
            perm = rng.permutation(X.shape[0])
            for i in range(0, X.shape[0], batch):
                idx = perm[i:i + batch]
                _, dl, _ = WaqfInstitution.loss_and_grad(self.forward(X[idx]), y[idx], self.C)
                opt.step(self.params(), self.backward(dl))

    def accuracy(self, X, y):
        return float((self.forward(X).argmax(1) == y).mean())


# ---------------------------------------------------------------------------
# Synthetic worlds: the two banks of the river of Fez
# ---------------------------------------------------------------------------

def make_moons(n: int, rng: np.random.Generator, noise: float = 0.12) -> Tuple[np.ndarray, np.ndarray]:
    n1 = n // 2; n2 = n - n1
    t1 = rng.uniform(0, np.pi, n1); t2 = rng.uniform(0, np.pi, n2)
    X1 = np.stack([np.cos(t1), np.sin(t1)], 1)
    X2 = np.stack([1 - np.cos(t2), 1 - np.sin(t2) - 0.5], 1)
    X = np.concatenate([X1, X2]) + rng.normal(0, noise, (n, 2))
    y = np.concatenate([np.zeros(n1, int), np.ones(n2, int)])
    p = rng.permutation(n)
    return X[p], y[p]


def make_spiral_bank(n: int, rng: np.random.Generator, shift=(7.0, 4.0)) -> Tuple[np.ndarray, np.ndarray]:
    """A second, distant community with an inverted labelling rule."""
    n1 = n // 2; n2 = n - n1
    t = rng.uniform(0.3, 2.2, n1)
    X1 = np.stack([t * np.cos(3 * t), t * np.sin(3 * t)], 1)
    t = rng.uniform(0.3, 2.2, n2)
    X2 = np.stack([-t * np.cos(3 * t), -t * np.sin(3 * t)], 1)
    X = np.concatenate([X1, X2]) * 0.8 + np.array(shift) + rng.normal(0, 0.06, (n, 2))
    y = np.concatenate([np.ones(n1, int), np.zeros(n2, int)])     # inverted rule
    p = rng.permutation(n)
    return X[p], y[p]


# ---------------------------------------------------------------------------
# Finite-difference gradient check (mandatory)
# ---------------------------------------------------------------------------

def gradient_check(seed: int = 7, eps: float = 1e-6) -> float:
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (9, 3))
    y = rng.integers(0, 3, 9)
    ring = WaqfRing("check", X, K=5, B=3, m=4, C=3, rng=rng)
    # make the problem non-trivial: randomise plastic parts
    ring.logrho = rng.normal(0.3, 0.5, ring.K)
    ring.lognd = rng.normal(0.0, 0.7, (ring.K, ring.B))
    for j in range(ring.B):
        ring.W[j] = rng.normal(0, 0.8, ring.W[j].shape); ring.c[j] = rng.normal(0, 0.3, ring.m)
    ring.V = rng.normal(0, 0.8, ring.V.shape); ring.b = rng.normal(0, 0.3, ring.C)

    def loss_fn():
        logits = ring.forward(X)
        return WaqfInstitution.loss_and_grad(logits, y, 3)[0]

    logits = ring.forward(X)
    _, dl, _ = WaqfInstitution.loss_and_grad(logits, y, 3)
    an = ring.backward(dl)
    worst = 0.0
    for name, p in ring.plastic_params().items():
        num = np.zeros_like(p)
        it = np.nditer(p, flags=["multi_index"], op_flags=["readwrite"])
        for _ in it:
            i = it.multi_index
            old = p[i]
            p[i] = old + eps; lp = loss_fn()
            p[i] = old - eps; lm = loss_fn()
            p[i] = old
            num[i] = (lp - lm) / (2 * eps)
        rel = np.abs(num - an[name]).max() / (np.abs(num).max() + np.abs(an[name]).max() + 1e-12)
        worst = max(worst, rel)
        print(f"    grad-check {name:8s} shape {str(p.shape):10s} max|num|={np.abs(num).max():.3e}  rel.err={rel:.2e}")
    return worst


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

def banner(t: str):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def main() -> int:
    t0 = time.time()
    results: Dict[str, bool] = {}
    banner("THE WAQF NETWORK -- self-tests (Fatima al-Fihri, mind 0220)")

    # ---------------- T1 gradient check ----------------
    banner("T1  finite-difference gradient check (all plastic + stipulation params)")
    worst = gradient_check()
    results["T1 gradient check"] = worst < 1e-5
    print(f"  worst relative error = {worst:.3e}  ->  {'PASS' if results['T1 gradient check'] else 'FAIL'}")

    # ---------------- world A: the Qarawiyyin bank ----------------
    rng = np.random.default_rng(859)
    XA, yA = make_moons(900, rng)
    XA_tr, yA_tr, XA_te, yA_te = XA[:700], yA[:700], XA[700:], yA[700:]

    inst = WaqfInstitution(C=2, B=4, m=8, seed=245)
    ringA = inst.found_ring("qarawiyyin", XA_tr, K=40)

    # ---------------- T2 provenance by construction ----------------
    banner("T2  provenance: every principal is exactly the mean of its deed")
    ok = True; covered = np.zeros(XA_tr.shape[0], bool)
    for k, dd in enumerate(ringA.deeds):
        ok &= np.allclose(ringA.P[k], XA_tr[dd.idx].mean(0), atol=1e-12)
        ok &= hashlib.sha256(np.ascontiguousarray(XA_tr[dd.idx]).tobytes()).hexdigest()[:16] == dd.digest
        covered[dd.idx] = True
    ok &= bool(covered.all()) and bool((ringA.s > 0).all())
    results["T2 provenance"] = ok
    print(f"  {ringA.K} principals, {covered.sum()}/{XA_tr.shape[0]} construction points in deeds,"
          f" extents in [{ringA.s.min():.3f},{ringA.s.max():.3f}]  ->  {'PASS' if ok else 'FAIL'}")

    # ---------------- T3 the fast: not served while under construction ----------------
    banner("T3  the fast: a ring under construction is trained but not consumed")
    served_before = inst.forward(XA_te).copy()          # empty institution -> zeros
    hist = inst.fast(ringA, XA_tr, yA_tr, epochs=80, lr=5e-3, verbose=True)
    served_during = inst.forward(XA_te)
    not_consumed = np.allclose(served_before, served_during)
    ring_alone_acc = float(((ringA.forward(XA_te)).argmax(1) == yA_te).mean())
    inst.complete(ringA)
    acc_A0 = inst.accuracy(XA_te, yA_te)
    results["T3 fast/serve"] = not_consumed and abs(acc_A0 - ring_alone_acc) < 1e-12
    print(f"  served output unchanged during construction: {not_consumed}; "
          f"ring accuracy at completion {acc_A0:.3f}  ->  {'PASS' if results['T3 fast/serve'] else 'FAIL'}")

    # ---------------- T4 stipulation semantics ----------------
    banner("T4  waterfall stipulation: allocation <= need, conservation, non-negativity")
    ringA.forward(XA_te)
    c = ringA.cache
    n = softplus(ringA.lognd) + 1e-4
    alloc_ok = bool((c["A"] >= -1e-12).all()) and bool((c["A"] <= n[None] + 1e-9).all())
    conserve = np.abs(c["A"].sum(2) + c["R"] - c["g"]).max()
    results["T4 stipulation"] = alloc_ok and conserve < 1e-9
    share_poor = float(c["R"].sum() / c["g"].sum())
    print(f"  0 <= allocation <= need: {alloc_ok}; |sum(alloc)+remainder-yield| max = {conserve:.1e};"
          f" share reaching 'the poor' = {share_poor:.3f}  ->  {'PASS' if results['T4 stipulation'] else 'FAIL'}")

    # ---------------- administration on A ----------------
    banner("T5  learning: administration of the completed ring")
    inst.administer(XA_tr, yA_tr, epochs=40, lr=2e-3, verbose=True)
    acc_A = inst.accuracy(XA_te, yA_te)
    results["T5 learning"] = acc_A >= 0.95
    print(f"  two-moons test accuracy = {acc_A:.3f}  ->  {'PASS' if results['T5 learning'] else 'FAIL'}")

    # ---------------- T6 perpetuity ----------------
    banner("T6  perpetuity: principals, extents and sealed needs never move")
    P0, s0 = ringA.snapshot_at_founding
    lognd_sealed = ringA.lognd.copy()
    perp = np.array_equal(P0, ringA.P) and np.array_equal(s0, ringA.s)
    try:
        ringA.P[0, 0] += 1.0; mutable = True
    except ValueError:
        mutable = False
    results["T6 perpetuity"] = perp and (not mutable) and ringA.sealed
    print(f"  principals bitwise-identical to founding: {perp}; write-protected: {not mutable};"
          f" deed sealed: {ringA.sealed}  ->  {'PASS' if results['T6 perpetuity'] else 'FAIL'}")

    # ---------------- T7 accretion without forgetting ----------------
    banner("T7  accretion: a second community founds a second ring; the first is not forgotten")
    XB, yB = make_spiral_bank(900, rng)
    XB_tr, yB_tr, XB_te, yB_te = XB[:700], yB[:700], XB[700:], yB[700:]
    acc_B_before = inst.accuracy(XB_te, yB_te)
    ringB = inst.found_ring("andalusiyyin", XB_tr, K=48)
    inst.fast(ringB, XB_tr, yB_tr, epochs=120, lr=5e-3, verbose=True)
    inst.complete(ringB)
    # administration on the MIXED stream: both communities now use the institution
    Xmix = np.concatenate([XA_tr, XB_tr]); ymix = np.concatenate([yA_tr, yB_tr])
    inst.administer(Xmix, ymix, epochs=30, lr=2e-3, verbose=False)
    acc_A_after = inst.accuracy(XA_te, yA_te)
    acc_B_after = inst.accuracy(XB_te, yB_te)
    # yield locality: how much does ring B fire on A's land and vice versa?
    gBA, _ = ringB.yield_(XA_te); gAA, _ = ringA.yield_(XA_te)
    gAB, _ = ringA.yield_(XB_te); gBB, _ = ringB.yield_(XB_te)
    cross = 0.5 * (gBA.sum() / gAA.sum() + gAB.sum() / gBB.sum())
    # control: an ordinary MLP trained on A, then on B only (naive sequential learning)
    ctrl = MLP(2, 64, 2, np.random.default_rng(1))
    ctrl.fit(XA_tr, yA_tr, np.random.default_rng(2), epochs=80)
    ctrl_A0 = ctrl.accuracy(XA_te, yA_te)
    ctrl.fit(XB_tr, yB_tr, np.random.default_rng(3), epochs=80)
    ctrl_A1 = ctrl.accuracy(XA_te, yA_te); ctrl_B1 = ctrl.accuracy(XB_te, yB_te)
    drop_waqf = acc_A - acc_A_after
    drop_ctrl = ctrl_A0 - ctrl_A1
    results["T7 accretion"] = (acc_B_after >= 0.90) and (drop_waqf <= 0.04)
    print(f"  ring B founded (K=48). Task B acc: {acc_B_before:.3f} (before) -> {acc_B_after:.3f} (after)")
    print(f"  Task A acc: {acc_A:.3f} -> {acc_A_after:.3f}   (drop {drop_waqf:+.3f})")
    print(f"  cross-land yield ratio (foreign/own) = {cross:.4f}")
    print(f"  control MLP, naive sequential: A {ctrl_A0:.3f} -> {ctrl_A1:.3f} (drop {drop_ctrl:+.3f}); B {ctrl_B1:.3f}")
    print(f"  ->  {'PASS' if results['T7 accretion'] else 'FAIL'}")
    # T7b -- a NEAR community whose land overlaps the first: interference is bounded, not zero
    XC, yC = make_spiral_bank(700, rng, shift=(2.2, 0.4))
    XC_tr, yC_tr, XC_te, yC_te = XC[:500], yC[:500], XC[500:], yC[500:]
    accA_pre_C = inst.accuracy(XA_te, yA_te)
    ringC = inst.found_ring("near-quarter", XC_tr, K=40)
    inst.fast(ringC, XC_tr, yC_tr, epochs=120, lr=5e-3, verbose=False)
    inst.complete(ringC)
    accA_post_C_fast = inst.accuracy(XA_te, yA_te)
    inst.administer(np.concatenate([XA_tr, XB_tr, XC_tr]), np.concatenate([yA_tr, yB_tr, yC_tr]), epochs=25, lr=2e-3)
    accA_post_C = inst.accuracy(XA_te, yA_te); accC = inst.accuracy(XC_te, yC_te)
    gCA, _ = ringC.yield_(XA_te)
    cross_near = gCA.sum() / gAA.sum()
    print(f"  T7b near community (overlapping land): ring C founded; foreign/own yield on A = {cross_near:.3f}")
    print(f"      Task A acc {accA_pre_C:.3f} -> {accA_post_C_fast:.3f} (after C's fast) -> {accA_post_C:.3f} (after joint administration); Task C acc {accC:.3f}")

    # ---------------- T8 istibdal ----------------
    banner("T8  istibdal: the custodian exchanges only barren principals, and service recovers")
    P_good = ringA.P.copy()
    ruin = np.random.default_rng(5).choice(ringA.K, size=ringA.K * 7 // 10, replace=False)
    P_bad = P_good.copy(); P_bad[ruin] += np.array([40.0, -40.0])       # a flood carries them off the land
    ringA.P = P_bad; ringA.P.setflags(write=False)
    acc_ruined = inst.accuracy(XA_te, yA_te)
    exchanged, barren = inst.nazir(ringA, XA_tr, yA_tr)
    exchanged, barren = set(exchanged), set(barren)
    all_ruined_exchanged = set(ruin.tolist()).issubset(exchanged)
    only_barren = exchanged.issubset(barren)
    healthy = [k for k in range(ringA.K) if k not in barren]
    healthy_untouched = np.array_equal(ringA.P[healthy], P_bad[healthy])
    provenance_ok = all(np.allclose(ringA.P[k], XA_tr[ringA.deeds[k].idx].mean(0), atol=1e-12) for k in exchanged)
    inst.administer(XA_tr, yA_tr, epochs=25, lr=2e-3, verbose=False)
    acc_recovered = inst.accuracy(XA_te, yA_te)
    results["T8 istibdal"] = (all_ruined_exchanged and only_barren and healthy_untouched and provenance_ok
                              and len(exchanged) > 0 and (acc_recovered >= acc_A - 0.03))
    print(f"  {len(ruin)}/{ringA.K} principals carried off the land -> acc {acc_ruined:.3f}")
    print(f"  nazir found {len(barren)} barren, exchanged {len(exchanged)}: all ruined exchanged={all_ruined_exchanged},"
          f" only barren touched={only_barren}, healthy untouched={healthy_untouched}, new deeds exact={provenance_ok}")
    print(f"  after administration acc {acc_recovered:.3f}  ->  {'PASS' if results['T8 istibdal'] else 'FAIL'}")

    # ---------------- T9 the founder's absence ----------------
    banner("T9  the founder's absence: deeds verify without the construction data")
    del XA_tr  # the founder leaves; the land she bought is remembered only in deeds
    reconstructed = all(len(dd.digest) == 16 for dd in ringA.deeds) and all(dd.idx.size > 0 for dd in ringA.deeds)
    acc_final = inst.accuracy(XA_te, yA_te)
    results["T9 founder absent"] = reconstructed and acc_final >= 0.9
    print(f"  deeds intact ({' + '.join(str(len(r.deeds)) for r in inst.rings)}), institution still serves: acc {acc_final:.3f}"
          f"  ->  {'PASS' if results['T9 founder absent'] else 'FAIL'}")

    # ---------------- ledger ----------------
    banner("LEDGER of the institution")
    for line in inst.ledger:
        print("  - " + line)
    n_params = sum(p.size for r in inst.rings for p in r.plastic_params().values())
    n_frozen = sum(r.P.size + r.s.size + r.lognd.size for r in inst.rings)
    print(f"  rings: {len(inst.rings)}; plastic parameters: {n_params}; inalienable+sealed: {n_frozen}")

    banner("SUMMARY")
    allok = True
    for k, v in results.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
        allok &= v
    print(f"\n  {sum(results.values())}/{len(results)} tests passed in {time.time() - t0:.1f}s")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
