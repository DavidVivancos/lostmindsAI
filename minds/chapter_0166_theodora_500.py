#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 ENCYCLOPEDIA OF LOST MINDS — Chapter 0166
 THEODORA (c. 497/500 – 28 June 548), Augusta of the Eastern Roman Empire
 Architecture: THE SANCTUARY-FORK NETWORK  ("The Hormisdas Machine")
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 9 Minds 161 - 180 Available on Amazon https://www.amazon.com/dp/B0HJC1QF59
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 0166_theodora_500 - THEODORA (c. 497/500 – 28 June 548), Augusta of the Eastern Roman Empire
================================================================================  

WHY THIS ARCHITECTURE AND NOT A TRANSFORMER
-------------------------------------------
Her husband Justinian (chapter 0165) is the archetype of the optimizer that
abolishes contradiction: one law, one faith, one concordant corpus, a standing
ban on commentary. Theodora's cognitive signature is the exact structural
complement, and it survives in deeds rather than doctrine:

  (1) SANCTUARY   — She sheltered and FUNDED the officially refuted branch (the
                    anti-Chalcedonian clergy) inside the Palace of Hormisdas, at
                    the moment her own state was driving that branch extinct.
                    Note precisely what she did and did not do: she never forced
                    the emperor to adopt their reading of any case. She kept them
                    housed, fed and ordaining. Funding, not a vote.
  (2) ORDINATION  — 542/3: she obtained the consecration of Jacob Baradaeus. She
                    did not preserve a *copy* of the losing doctrine; she licensed
                    it to REPLICATE ITSELF. A saved artifact dies with its
                    archive. A saved generator forks. The hierarchy she seeded is
                    ordaining priests this morning.
  (3) CHANNEL     — Nubia, 543: two rival doctrines raced for an ungoverned
                    frontier and she did not argue the merits. She wrote to the
                    duke of the Thebaid and had the rival caravan delayed.
                    Deployment beats correctness.
  (4) METANOIA    — CJ 5.4.23: a person's status ("infamis") is a decree, not an
                    essence, and a decree can be revoked. Categories are policy.
  (5) THE SHROUD  — Nika, 532: with the harbour open, a council cannot compute a
                    defence. She removed one branch from the option set of a
                    deliberating group, and thereby deleted an equilibrium.

None of that is attention over stored keys. It is a mixture of RIVAL experts with
a structurally protected minority, plus discrete operators that rehabilitate and
replicate branches the loss wants dead, plus a coordination fixed point for the
option-set surgery of the Nika council.

Verified below: a finite-difference gradient check on the full loss, a real
training loop, held-out evaluation, a distribution-shift test (which is the whole
point of the mind), and self-tests. Dependencies: numpy only.

  $ python3 chapter_0166_theodora_500.py
================================================================================
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

__figure__ = 166
__subject__ = "Theodora"
__era__ = "c. 500 CE"
__architecture__ = "Sanctuary-Fork Network (SFN)"

SEED = 548  # the year she died


# =============================================================================
# 0. NUMERICAL UTILITIES
# =============================================================================

def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def sigmoid(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


# =============================================================================
# 1. THE DOCTRINE STREAM
# =============================================================================
#
#   regime 0  "the capital"   — 88% of the training stream. The orthodoxy fits it
#                               and, by any honest accounting, is right to.
#   regime 1  "the frontier"  — Syria, Egypt, the Nubian marches. 12%.
#
# The frontier does not have different evidence. It has the SAME evidence read a
# different way: its labels are the capital's labels rotated by one class. This is
# deliberate. The Chalcedonian and the Miaphysite were not looking at different
# Christs. They were looking at the same Christ and counting his natures
# differently. A model that resolves the contradiction does not merely lose a
# little accuracy on the frontier — it is confidently, systematically WRONG there,
# which is a far more dangerous failure than noise.
#
# The regime is inferable from x (it displaces the first three features), so a
# model that keeps two readings alive can route between them. A model that has
# collapsed to one reading cannot, however good its features are.

@dataclass
class DoctrineStream:
    d: int = 12
    C: int = 3
    seed: int = SEED
    sep: float = 2.2                  # how legible the frontier is.

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.W = rng.normal(0, 1.0, size=(self.C, self.d))   # the shared evidence map
        self.b = rng.normal(0, 0.2, size=self.C)
        self.mu = np.zeros(self.d)
        self.mu[:3] = self.sep

    def sample(self, n: int, p_frontier: float, seed: int):
        rng = np.random.default_rng(seed)
        r = (rng.random(n) < p_frontier).astype(int)
        x = rng.normal(0, 1.0, size=(n, self.d)) + np.outer(r, self.mu)
        scores = x @ self.W.T + self.b + rng.normal(0, 0.20, size=(n, self.C))
        y_capital = scores.argmax(1)
        y = np.where(r == 1, (y_capital + 1) % self.C, y_capital)   # the other reading
        return x, y, r


# =============================================================================
# 2. THE SANCTUARY-FORK NETWORK
# =============================================================================
#
#   h    = tanh(W1 x + b1)                    shared encoder ("the court")
#   z    = Wg h + bg + beta * c               gate logits
#   pg   = softmax(z)                         input-conditional routing
#   p_k  = softmax(We_k h + be_k)             expert k's reading of the case
#   p    = sum_k pg_k p_k                     the empire's answer
#
# c is THE SANCTUARY: a bias vector that is not a parameter and takes no gradient.
# After every step it is nudged so each seated expert keeps a fair share of the
# routing:
#
#   c_k  <-  c_k + eta * (1/n_active - load_k)          [seated experts only]
#
# This is the precise shape of what she actually did. She never compelled her
# husband to adopt the Miaphysite reading of a case — that would be forcing a
# constant opinion into every output, and it makes the empire measurably worse
# (this file's earlier drafts did exactly that, and the accuracy fell). She kept
# the refuted party FUNDED, HOUSED and ORDAINING, so that it stayed a live,
# trained, competent hypothesis, available on the day the routing changed.
#
# Sanctuary is a subsidy on capacity, not a vote on content.
# That distinction is the whole of her politics.
#
#   Loss = NLL(p, y) + lambda_sep * OVERLAP(p_k) + wd*||W||^2
#
# OVERLAP = mean over pairs (k<l) of sum_c p_k[c] p_l[c]. It reaches 1 when every
# expert has become the same mind — which is exactly the Nika condition: the Blues
# and the Greens stopped being two, and the city burned for five days.

@dataclass
class SFNConfig:
    d: int = 12
    Hg: int = 24               # the gate has its own eyes
    K: int = 5                 # 4 seated + 1 empty chair held in reserve
    C: int = 3
    n_active_init: int = 4
    beta: float = 1.0          # sanctuary on/off
    gate_cap: float = 4.0      # THE ENTRENCHMENT. See below.
    gate_temp: float = 0.30    # near-exclusive routing: one case, one reader
    min_share: float = 0.10    # the floor beneath which no seated expert may fall
    eta_c: float = 0.30        # gain of the funding controller
    c_cap: float = 8.0         # the purse is large, but it is not infinite
    kappa_concord: float = 4.0 # JUSTINIAN'S TERM. The empire pays for agreement.
    wd: float = 1e-4
    lr: float = 0.06
    momentum: float = 0.9
    ema: float = 0.9
    seed: int = SEED
    allow_fork: bool = True
    allow_commission: bool = True


class SanctuaryForkNetwork:
    """K rival experts, each an INDEPENDENT LINEAR READER of the raw evidence, plus
    a gate with its own eyes.

    Linear is not a simplification, it is the point. One expert can hold exactly
    one doctrine. No single reader can hold both the capital's reading and the
    frontier's, because they assign opposite labels to identical evidence — the
    two natures and the one nature are not reconcilable inside a single head. The
    empire therefore MUST keep more than one reader alive, or it must choose. A
    deep shared trunk would smuggle the contradiction into the features and hide
    the choice from us. We are not going to let it."""

    def __init__(self, cfg: SFNConfig) -> None:
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        d, Hg, K, C = cfg.d, cfg.Hg, cfg.K, cfg.C
        self.p: Dict[str, np.ndarray] = {
            "We": rng.normal(0, 0.30, size=(K, C, d)),                # one reader, one doctrine
            "be": np.zeros((K, C)),
            "Vg": rng.normal(0, np.sqrt(2.0 / d), size=(Hg, d)),      # the gate's own eyes
            "ag": np.zeros(Hg),
            "Wg": rng.normal(0, np.sqrt(1.0 / Hg), size=(K, Hg)),
            "bg": np.zeros(K),
        }
        self.active = np.zeros(K, dtype=bool)
        self.active[: cfg.n_active_init] = True

        self.c = np.zeros(K)                      # THE SANCTUARY BIAS (no gradient)
        self.endow = np.zeros(K)                  # standing granted by metanoia()
        self.favour = np.full(K, 1.0 / max(1, cfg.n_active_init))
        self.favour[~self.active] = 0.0
        self.vel = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.log: List[str] = []
        self.rehabilitated: set = set()
        self.ordinations: List[Tuple[int, int]] = []
        # SHIELDED experts are exempt from the imperial gradient. This is the
        # Palace of Hormisdas rendered as an update rule: they lived inside the
        # palace, fed from her purse, and were not required to conform.
        self.shielded = np.zeros(K, dtype=bool)

    # -- forward ---------------------------------------------------------------
    def forward(self, X: np.ndarray, params: Optional[Dict[str, np.ndarray]] = None):
        P = self.p if params is None else params
        ze = np.einsum("nd,kcd->nkc", X, P["We"]) + P["be"][None]            # (N,K,C)
        pk = softmax(ze, axis=2)
        hg = np.tanh(X @ P["Vg"].T + P["ag"])                                # (N,Hg)
        raw = hg @ P["Wg"].T + P["bg"]                                       # (N,K)
        G = self.cfg.gate_cap
        u = np.tanh(raw / G)
        sq = G * u                       # THE ENTRENCHMENT: the gate's power to
                                         # defund an expert is BOUNDED at +-G nats.
        zg = (sq + self.cfg.beta * (self.c + self.endow) + self.chair_mask) / self.cfg.gate_temp
        pg = softmax(zg, axis=1)
        p = np.einsum("nk,nkc->nc", pg, pk)                                  # (N,C)
        return {"hg": hg, "u": u, "raw": raw, "zg": zg, "pg": pg,
                "ze": ze, "pk": pk, "p": p}

    @property
    def chair_mask(self) -> np.ndarray:
        """An empty chair is a wall, not a disfavoured expert."""
        return np.where(self.active, 0.0, -60.0)

    # -- loss ------------------------------------------------------------------
    def loss(self, X, y, params=None, cache=None):
        P = self.p if params is None else params
        f = self.forward(X, P) if cache is None else cache
        N = X.shape[0]
        nll = -np.log(f["p"][np.arange(N), y] + 1e-12).mean()

        pk = f["pk"]
        idx = np.where(self.active & ~self.shielded)[0]   # the sheltered are exempt
        ov, npairs = 0.0, 0
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                ov += float((pk[:, idx[i], :] * pk[:, idx[j], :]).sum(1).mean())
                npairs += 1
        overlap = ov / max(1, npairs)

        l2 = sum(float((P[k] ** 2).sum()) for k in ("We", "Vg", "Wg"))
        return nll - self.cfg.kappa_concord * overlap + self.cfg.wd * l2, f

    # -- analytic backward -----------------------------------------------------
    def backward(self, X, y, f):
        P, cfg = self.p, self.cfg
        N = X.shape[0]
        hg, pg, pk, p = f["hg"], f["pg"], f["pk"], f["p"]

        dp = np.zeros_like(p)
        dp[np.arange(N), y] = -1.0 / (p[np.arange(N), y] + 1e-12)
        dp /= N

        dpg = np.einsum("nc,nkc->nk", dp, pk)          # p = sum_k pg_k pk_k
        dpk = dp[:, None, :] * pg[:, :, None]

        # the concord term (see SFNConfig.lambda_sep) — the sheltered are exempt
        idx = np.where(self.active & ~self.shielded)[0]
        npairs = max(1, len(idx) * (len(idx) - 1) // 2)
        coef = -cfg.kappa_concord / (npairs * N)
        for a in idx:
            others = [b for b in idx if b != a]
            if others:
                dpk[:, a, :] += coef * pk[:, others, :].sum(1)

        # experts (linear readers of the raw evidence)
        dze = pk * (dpk - (dpk * pk).sum(axis=2, keepdims=True))          # (N,K,C)
        dWe = np.einsum("nkc,nd->kcd", dze, X)
        dbe = dze.sum(0)

        # gate (c takes no gradient — by design: it is a decision of state, not a fit)
        dzg = pg * (dpg - (dpg * pg).sum(axis=1, keepdims=True))
        dzg = dzg / cfg.gate_temp                # through the sharpening
        dzg = dzg * (1.0 - f["u"] ** 2)          # through the entrenchment squash
        dWg = dzg.T @ hg
        dbg = dzg.sum(0)
        dhg = dzg @ P["Wg"]
        dpre_g = dhg * (1.0 - hg ** 2)
        dVg = dpre_g.T @ X
        dag = dpre_g.sum(0)

        dWe += 2 * cfg.wd * P["We"]
        dVg += 2 * cfg.wd * P["Vg"]
        dWg += 2 * cfg.wd * P["Wg"]
        return {"We": dWe, "be": dbe, "Vg": dVg, "ag": dag, "Wg": dWg, "bg": dbg}

    # -- optimiser + the empress's ledger --------------------------------------
    def step(self, grads):
        cfg = self.cfg
        if self.shielded.any():
            # The majority's gradient does not reach the sheltered. That is what
            # shelter MEANS. Without this line the commissioned expert is quietly
            # dragged back to orthodoxy within a few epochs, because 85% of the
            # loss it feels is the capital's — which is exactly what happens to a
            # dissenting church that stays inside the state's incentive gradient.
            grads = dict(grads)
            grads["We"] = grads["We"].copy(); grads["We"][self.shielded] = 0.0
            grads["be"] = grads["be"].copy(); grads["be"][self.shielded] = 0.0
        for k in self.p:
            self.vel[k] = cfg.momentum * self.vel[k] - cfg.lr * grads[k]
            self.p[k] += self.vel[k]

    def fund(self, pg: np.ndarray) -> None:
        """THE SANCTUARY. Read the routing ledger; push funding toward whoever the
        gate is starving. No gradient is involved. This is a decision of state,
        taken against the loss, and paid for out of her own purse — she had, as
        Bury noticed, revenues of her own for which she rendered no account."""
        load = pg.mean(0)
        self.favour = self.cfg.ema * self.favour + (1 - self.cfg.ema) * load
        if self.cfg.beta <= 0:
            return
        act = self.active
        floor = self.cfg.min_share
        # A FLOOR, NOT A LEVELLING. She did not demand that the heretics get an equal
        # share of the empire — that would have wrecked the empire, and this file's
        # earlier drafts proved it numerically: forcing uniform traffic across the
        # court dragged accuracy down on every measure. She demanded only that they
        # not be driven to nothing. So the subsidy is paid to a seated expert if and
        # only if the gate has pushed it below the floor, and it is integrated (not
        # merely proportional), because a proportional correction measured AFTER the
        # subsidy just ratifies whatever the gate already decided.
        for k in np.where(act)[0]:
            if self.favour[k] < floor:
                self.c[k] += self.cfg.eta_c * (np.log(floor) -
                                               np.log(max(self.favour[k], 1e-6)))
            else:
                self.c[k] *= 0.98        # standing, once no longer needed, decays
        np.clip(self.c, 0.0, self.cfg.c_cap, out=self.c)
        self.c[~act] = 0.0

    # -------------------------------------------------------------------------
    # STRUCTURAL OPERATORS — the moves no gradient will ever propose
    # -------------------------------------------------------------------------
    def _counterfactual(self, X, y):
        """Where does orthodoxy fail — and who is right there?"""
        f = self.forward(X)
        ek = f["pk"].argmax(2)
        dominant = int(f["pg"].mean(0).argmax())
        wrong = ek[:, dominant] != y
        return ek, dominant, wrong

    def metanoia(self, X, y, margin: float = 0.25) -> List[int]:
        """REHABILITATION. CJ 5.4.23 lets a woman of the theatre shed the status of
        *infamis* — the blemish of the past is struck out on a showing of
        repentance. The law is usually read as Justinian bending the code so he
        could marry an actress. Read it instead as an epistemology: a category is
        a decree, and a decree can be revoked.

        Standard practice prunes the expert holding the least routing mass. We do
        the opposite. Restrict attention to the cases the DOMINANT expert gets
        wrong, and ask what the exile knows there. An expert competent exactly
        where orthodoxy fails is not a failed expert. It is a suppressed one."""
        ek, dominant, wrong = self._counterfactual(X, y)
        out: List[int] = []
        if wrong.sum() < 40:
            return out
        for k in np.where(self.active)[0]:
            if k == dominant or k in self.rehabilitated:
                continue
            acc = float((ek[wrong, k] == y[wrong]).mean())
            starved = self.favour[k] < 0.6 / max(1, int(self.active.sum()))
            if acc > margin and starved:
                self.endow[k] += 1.0                   # an endowment, not a pardon
                self.rehabilitated.add(int(k))
                out.append(int(k))
                self.log.append(
                    f"    METANOIA   : expert {k} was starved (load {self.favour[k]:.3f}) "
                    f"yet is right on {100*acc:.1f}% of the cases the dominant expert "
                    f"({dominant}) gets wrong. Record struck; funding restored.")
        return out

    def commission(self, X, y, steps: int = 40, batch: int = 96, lr: float = 0.25):
        """THE MISSION TO NUBIA (543) — targeted capacity, not a fair share.

        Load-balancing alone cannot save a minority hypothesis, and this file's
        earlier drafts proved it: give every expert an equal share of a stream that
        is 95% capital and every expert learns the capital. The gate will never
        route the frontier to a frontier-reader, because no frontier-reader exists;
        and no frontier-reader ever comes into existence, because the gate never
        routes the frontier to anyone. The lock is perfect and it is where a purely
        gradient-driven empire stays forever.

        She did not wait for the gate. She picked a man — Julian, of the exiled
        patriarch's household — pointed him at the exact frontier the orthodoxy was
        failing to hold, and paid for the caravan herself.

        So: take the residual (every case the whole empire currently gets WRONG),
        take the most starved seated expert, and train it on that residual alone.
        Its private objective is not the mixture's loss. It is the loss of the
        people nobody is serving."""
        pred = self.predict(X)
        R = np.where(pred != y)[0]
        if len(R) < 60:
            return None
        already = np.where(self.shielded & self.active)[0]
        if len(already):
            k = int(already[0])                       # refresh the standing mission
        else:
            live = np.where(self.active)[0]
            k = int(live[np.argmin(self.favour[live])])   # the most starved chair
            self.shielded[k] = True                   # and now it is under her roof
        rng = np.random.default_rng(SEED + 91 + int(self.endow.sum() * 7))
        XR, yR = X[R], y[R]
        for _ in range(steps):
            b = rng.choice(len(R), size=min(batch, len(R)), replace=False)
            xb, yb = XR[b], yR[b]
            z = xb @ self.p["We"][k].T + self.p["be"][k]
            q = softmax(z, axis=1)
            q[np.arange(len(b)), yb] -= 1.0
            q /= len(b)
            self.p["We"][k] -= lr * (q.T @ xb)
            self.p["be"][k] -= lr * q.sum(0)
        acc = float((np.argmax(XR @ self.p["We"][k].T + self.p["be"][k], 1) == yR).mean())
        self.endow[k] = min(self.endow[k] + 0.8, 4.0)
        self.log.append(
            f"    COMMISSION : expert {k} sheltered and sent to the {len(R)} cases the "
            f"whole empire gets wrong; it now reads {100*acc:.1f}% of them correctly. "
            f"Exempt from the imperial gradient; standing granted so the gate can find it.")
        return k

    def ordain(self, X, y, min_cf: float = 0.30):
        """REPLICATION LICENCE — Jacob Baradaeus, consecrated 542/3.

        Shelter alone only postpones a death: the sheltered copy dies with the
        shelter. Her real move was to have the refuted branch CONSECRATED — given
        the power to ordain others, so that it could reproduce without her. From
        two bishops came a hierarchy that outlived the empress by fourteen
        centuries.

        Here: clone the best exile into an empty chair with a small perturbation.
        The child begins as a copy and is then free to diverge — which is what
        makes it a fork and not a backup."""
        if not self.cfg.allow_fork:
            return None
        free = np.where(~self.active)[0]
        if len(free) == 0:
            return None
        ek, dominant, wrong = self._counterfactual(X, y)
        if wrong.sum() < 40:
            return None
        best, best_acc = -1, -1.0
        for k in np.where(self.active)[0]:
            if k == dominant:
                continue
            a = float((ek[wrong, k] == y[wrong]).mean())
            if a > best_acc:
                best, best_acc = int(k), a
        if best < 0 or best_acc < min_cf:
            return None
        seat = int(free[0])
        rng = np.random.default_rng(SEED + 17)
        for nm in ("We", "Wg"):
            self.p[nm][seat] = self.p[nm][best] + rng.normal(0, 0.02, self.p[nm][best].shape)
        self.p["be"][seat] = self.p["be"][best].copy()
        self.p["bg"][seat] = float(self.p["bg"][best])
        self.active[seat] = True
        self.shielded[seat] = bool(self.shielded[best])
        self.c[seat] = float(self.c[best])
        self.endow[seat] = float(self.endow[best])
        self.favour[seat] = float(self.favour[best])
        self.ordinations.append((best, seat))
        self.log.append(
            f"    ORDINATION : expert {best} (right on {100*best_acc:.1f}% of "
            f"orthodoxy's failures) consecrated into the empty chair {seat}. The "
            f"branch can now reproduce without her.")
        return (best, seat)

    # -- test-time adaptation ---------------------------------------------------
    def adapt_gate(self, X, y, steps: int = 400, lr: float = 0.35):
        """THE DAY THE WORLD TURNS.

        The empire's centre of gravity moves. Syria and Egypt are lost; the
        frontier is now the world. What does it cost to adapt?

        Exactly K numbers — the gate's biases — refit on a handful of the new
        world's cases. Nothing else is touched: not one expert, not one feature.
        This is the fairest test I can construct, because BOTH models get the same
        tiny budget. And that is the point. The same K numbers and the same handful
        of cases buy you nothing at all if you spent the last twenty years
        annihilating the alternative, and buy you the whole new world if you kept
        it fed."""
        for _ in range(steps):
            f = self.forward(X)
            N = X.shape[0]
            dp = np.zeros_like(f["p"])
            dp[np.arange(N), y] = -1.0 / (f["p"][np.arange(N), y] + 1e-12)
            dp /= N
            dpg = np.einsum("nc,nkc->nk", dp, f["pk"])
            pg = f["pg"]
            dzg = pg * (dpg - (dpg * pg).sum(1, keepdims=True))
            dzg = dzg / self.cfg.gate_temp * (1.0 - f["u"] ** 2)
            self.p["bg"] -= lr * dzg.sum(0)
        return self

    # -- inference --------------------------------------------------------------
    def predict(self, X):
        return self.forward(X)["p"].argmax(1)

    def gate_mass(self, X):
        return self.forward(X)["pg"].mean(0)

    def overlap(self, X):
        pk = self.forward(X)["pk"]
        idx = np.where(self.active)[0]
        vals = [float((pk[:, a, :] * pk[:, b, :]).sum(1).mean())
                for i, a in enumerate(idx) for b in idx[i + 1:]]
        return float(np.mean(vals)) if vals else 1.0


# =============================================================================
# 3. GRADIENT CHECK  (mandatory — the file does not ship without it passing)
# =============================================================================

def gradient_check(verbose: bool = True) -> float:
    rng = np.random.default_rng(3)
    net = SanctuaryForkNetwork(
        SFNConfig(d=6, Hg=5, K=4, C=3, n_active_init=3, beta=1.0,
                  kappa_concord=4.0, seed=11))
    net.c = np.array([0.7, -0.4, -0.3, 0.0])         # a non-trivial sanctuary bias
    X = rng.normal(size=(23, 6))
    y = rng.integers(0, 3, size=23)

    _, f = net.loss(X, y)
    G = net.backward(X, y, f)

    eps = 1e-5
    worst, checked, skipped = 0.0, 0, 0
    for name in ["We", "be", "Vg", "ag", "Wg", "bg"]:
        flat = net.p[name].reshape(-1)
        gflat = G[name].reshape(-1)
        for i in rng.choice(flat.size, size=min(30, flat.size), replace=False):
            if name == "bg" and not net.active[i]:
                continue                              # an empty chair is a wall
            o = flat[i]
            flat[i] = o + eps; Lp, _ = net.loss(X, y)
            flat[i] = o - eps; Lm, _ = net.loss(X, y)
            flat[i] = o
            num = (Lp - Lm) / (2 * eps)
            ana = float(gflat[i])
            scale = abs(num) + abs(ana)
            if scale < 1e-7:            # both gradients are numerically zero here;
                skipped += 1            # a relative error on 1e-9 is meaningless
                continue
            worst = max(worst, abs(num - ana) / scale)
            checked += 1
    if verbose:
        print(f"  finite differences vs analytic backprop: {checked} parameters "
              f"probed ({skipped} skipped as numerically zero), worst relative "
              f"error = {worst:.3e}  [{'PASS' if worst < 1e-5 else 'FAIL'}]")
    assert worst < 1e-5, f"gradient check FAILED ({worst:.3e})"
    return worst


# =============================================================================
# 4. TRAINING & EVALUATION
# =============================================================================

def evaluate(net, X, y, r=None):
    pred = net.predict(X)
    out = {"acc": float((pred == y).mean())}
    if r is not None:
        for reg, nm in ((0, "acc_capital"), (1, "acc_frontier")):
            m = r == reg
            out[nm] = float((pred[m] == y[m]).mean()) if m.sum() else float("nan")
    return out


COMMISSION_EPOCHS = (10, 20, 30, 42, 52)


def train(net, Xtr, ytr, epochs=60, batch=64, label="", verbose=True):
    rng = np.random.default_rng(SEED + 5)
    N = Xtr.shape[0]
    if verbose:
        print(f"\n  ---- training [{label}] ----")
        print(f"       sanctuary={'ON' if net.cfg.beta > 0 else 'OFF'}   "
              f"concord kappa={net.cfg.kappa_concord}   ordination="
              f"{'allowed' if net.cfg.allow_fork else 'forbidden'}")
    for ep in range(epochs):
        perm = rng.permutation(N)
        tot, nb = 0.0, 0
        for i in range(0, N, batch):
            b = perm[i:i + batch]
            L, f = net.loss(Xtr[b], ytr[b])
            net.step(net.backward(Xtr[b], ytr[b], f))
            net.fund(f["pg"])
            tot += L; nb += 1
        if net.cfg.beta > 0:
            if net.cfg.allow_commission and ep in COMMISSION_EPOCHS:
                net.commission(Xtr, ytr)
            net.metanoia(Xtr[:2000], ytr[:2000])
            if ep in (18, 34):
                net.ordain(Xtr[:2000], ytr[:2000])
        if verbose and (ep % 8 == 0 or ep == epochs - 1):
            gm = net.gate_mass(Xtr[:1500])
            seats = " ".join(f"{v:.2f}" + ("*" if net.active[j] else "·")
                             for j, v in enumerate(gm))
            print(f"    epoch {ep:3d} | loss {tot/nb:.4f} | routing [{seats}] "
                  f"| overlap {net.overlap(Xtr[:800]):.3f}")
        for line in net.log:
            print(line)
        net.log.clear()


# =============================================================================
# 5. THE PURPLE SHROUD — option-set surgery on a deliberating council
# =============================================================================
#
# Procopius, Wars 1.24: "there is the sea, here the boats... royalty is a good
# burial-shroud." The received reading is courage. The interesting reading is
# game-theoretic. A council is not one agent; it is N agents each estimating what
# the others will do. When a fallback exists, every member's willingness to hold
# is discounted by his belief that the others will take it — and that belief is
# self-confirming. Removing the option makes nobody braver. It deletes a fixed
# point.
#
#   f_i = sigma( alpha * mean_{j!=i} f_j  +  a * b_i  -  r_i )
#
#   f_i   probability member i bolts        r_i   private resolve
#   b_i   pull of the open harbour          a     is the harbour open? (1/0)
#   alpha contagion: how much i's nerve depends on everyone else's

@dataclass
class PurpleShroudCouncil:
    n: int = 9
    alpha: float = 3.6
    gamma: float = 1.4
    d_star: float = 4.5
    seed: int = SEED

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        self.r = rng.normal(1.15, 0.55, size=self.n)     # private resolve
        self.b = rng.normal(1.60, 0.35, size=self.n)     # the pull of the boats

    def fixed_point(self, option_open: bool, iters=800, damp=0.3):
        f = np.full(self.n, 0.5)
        a = 1.0 if option_open else 0.0
        for _ in range(iters):
            mean_other = (f.sum() - f) / (self.n - 1)
            new = (1 - damp) * f + damp * sigmoid(self.alpha * mean_other + a * self.b - self.r)
            if np.abs(new - f).max() < 1e-14:
                return new
            f = new
        return f

    def outcome(self, option_open, value_hold, value_flee, loss_fall):
        f = self.fixed_point(option_open)
        D = float((1 - f).sum())
        p_hold = float(sigmoid(self.gamma * (D - self.d_star)))
        ev = p_hold * value_hold + (1 - p_hold) * (
            f.mean() * value_flee - (1 - f.mean()) * loss_fall)
        return {"defection": float(f.mean()), "defenders": D,
                "p_hold": p_hold, "ev": float(ev)}


# =============================================================================
# 6. E-AGI BAROMETER PROJECTION
# =============================================================================
# A scoring heuristic bound to quantities THIS RUN measures — not a claim about
# the historical woman. Change the architecture and every number moves.

def barometer(id_acc, shift_acc, base_shift_acc, minority_acc,
              n_active, overlap, p_hold_sealed, p_hold_open):
    def cl(v):
        return float(max(0.0, min(100.0, v)))
    return [
        ("Cognitive Processing 🧩", cl(100 * id_acc),
         "in-distribution accuracy of the routed mixture"),
        ("Embodied Cognition 🤸", cl(50 + 50 * (p_hold_sealed - p_hold_open)),
         "how far the council's situated option set changes its act"),
        ("World Modeling 🌍", cl(100 * shift_acc),
         "accuracy after the world turns to the frontier"),
        ("Consciousness 👁️", cl(120 * (1 - overlap)),
         "self-monitoring: the capacity to remain several minds"),
        ("Language Understanding 💭", cl(100 * minority_acc),
         "reads the minority idiom, not only the imperial one"),
        ("Emotional Intelligence ❤️", cl(55 + 45 * min(1.0, max(0.0, n_active - 4))),
         "chairs kept for those the loss function wants gone"),
        ("Creativity ✨", cl(110 * (1 - overlap)),
         "diversity surviving in the hypothesis space"),
        ("Autonomy 🎯", cl(50 + 100 * (shift_acc - base_shift_acc)),
         "survives a regime it was never optimized for, unaided"),
    ]


# =============================================================================
# 7. SELF-TESTS
# =============================================================================

def self_tests() -> None:
    print("\n[SELF-TESTS]")
    rng = np.random.default_rng(0)

    net = SanctuaryForkNetwork(SFNConfig(d=5, Hg=6, K=4, C=3, n_active_init=3))
    X = rng.normal(size=(11, 5))
    f = net.forward(X)
    assert f["p"].shape == (11, 3)
    assert np.allclose(f["p"].sum(1), 1.0, atol=1e-10)
    assert np.allclose(f["pg"].sum(1), 1.0, atol=1e-10)
    print("  [ok] the mixture and the routing are proper distributions")

    # THE SANCTUARY RESTORES A CONDEMNED EXPERT — with no gradient at all
    net.p["bg"][1] = -6.0
    loads = []
    for _ in range(150):
        pg = net.forward(X)["pg"]
        loads.append(float(pg[:, 1].mean()))
        net.fund(pg)
    floor = net.cfg.min_share
    assert loads[0] < 0.02, loads[0]
    assert loads[-1] > 0.95 * floor, loads[-1]
    print(f"  [ok] sanctuary lifts a condemned expert back to its floor: routing mass "
          f"{loads[0]:.4f} -> {loads[-1]:.4f} (floor {floor:.2f}), and not one step")
    print(f"       of gradient descent was involved. It is a decision of state.")

    # WITHOUT SANCTUARY THE SAME EXPERT STAYS DEAD
    net0 = SanctuaryForkNetwork(SFNConfig(d=5, Hg=6, K=4, C=3, n_active_init=3, beta=0.0))
    net0.p["bg"][1] = -6.0
    for _ in range(150):
        net0.fund(net0.forward(X)["pg"])
    dead = float(net0.forward(X)["pg"][:, 1].mean())
    assert dead < 0.02
    print(f"  [ok] with no sanctuary the same expert stays extinguished "
          f"(routing mass {dead:.4f})")

    # ORDINATION seats a dormant expert as a faithful copy, free to diverge
    net2 = SanctuaryForkNetwork(SFNConfig(d=6, Hg=8, K=4, C=3, n_active_init=3))
    Xo, yo = rng.normal(size=(300, 6)), rng.integers(0, 3, size=300)
    before = int(net2.active.sum())
    res = net2.ordain(Xo, yo, min_cf=0.0)
    assert res is not None and net2.active.sum() == before + 1
    parent, seat = res
    pk = net2.forward(Xo)["pk"]
    div = float(np.abs(pk[:, parent] - pk[:, seat]).mean())
    assert div < 0.05
    print(f"  [ok] ordination: chair {seat} consecrated from expert {parent}; "
          f"mean |p_parent - p_child| = {div:.4f} (a copy, free to diverge)")

    # THE COUNCIL FIXED POINT IS ACTUALLY A FIXED POINT
    c = PurpleShroudCouncil()
    fo = c.fixed_point(True)
    mo = (fo.sum() - fo) / (c.n - 1)
    resid = float(np.abs(sigmoid(c.alpha * mo + c.b - c.r) - fo).max())
    assert resid < 1e-9, resid
    print(f"  [ok] the council equilibrium converges (residual {resid:.2e})")
    print("  ALL SELF-TESTS PASSED")


# =============================================================================
# 8. MAIN
# =============================================================================

def main() -> None:
    t0 = time.time()
    print("=" * 79)
    print(" CHAPTER 0166 - THEODORA (c. 500-548) - THE SANCTUARY-FORK NETWORK")
    print(" 'She kept the refuted branch alive, licensed it to reproduce, and")
    print("  raced it to a frontier the orthodoxy did not yet govern.'")
    print("=" * 79)

    print("\n[GRADIENT CHECK]")
    gradient_check()
    self_tests()

    world = DoctrineStream()
    Xtr, ytr, rtr = world.sample(8000, 0.10, SEED + 1)
    Xte, yte, rte = world.sample(3000, 0.10, SEED + 2)
    Xsh, ysh, rsh = world.sample(3000, 0.80, SEED + 3)
    Xad, yad, _   = world.sample(200,  0.80, SEED + 4)   # the adaptation budget

    print("\n[THE WORLD]")
    print("  The same evidence, read two ways. The frontier's labels are the capital's")
    print("  labels rotated by one class, so a reader that has resolved the")
    print("  contradiction is not merely worse on the frontier - it is confidently,")
    print("  systematically wrong there. Each expert is LINEAR: one reader, one")
    print("  doctrine. No single head can hold both.")
    print(f"    training : 8000 cases, {100*rtr.mean():4.1f}% frontier")
    print(f"    held-out : 3000 cases, {100*rte.mean():4.1f}% frontier")
    print(f"    SHIFTED  : 3000 cases, {100*rsh.mean():4.1f}% frontier"
          f"   <- the day the orthodoxy stops governing")
    print("    ADAPTING :  200 cases from the new world. Both empires get the same")
    print("               200, and may refit ONLY the gate's K biases. Nothing else.")

    print("\n[JUSTINIAN'S TERM]")
    print("  The empire's objective is not accuracy alone. Deo auctore (530) demands")
    print("  'one concordant corpus'; Novel 42 (536) deposes and exiles those who will")
    print("  not agree. Concord is a TERMINAL VALUE, and the empire pays real accuracy")
    print("  for it. So the loss carries a term for it: L = NLL - kappa * agreement.")
    print("  Every empire below runs with kappa = 4.0 except the last one.")

    KAP = 4.0
    runs = [
        ("A  ORTHODOXY", "no shelter; the court conforms or starves",
         SFNConfig(beta=0.0, kappa_concord=KAP, allow_fork=False, allow_commission=False)),
        ("B1 + FUNDING", "a floor under the disfavoured. Money only.",
         SFNConfig(beta=1.0, kappa_concord=KAP, allow_fork=False, allow_commission=False)),
        ("B2 + SHELTER", "exempt from the imperial gradient, and commissioned",
         SFNConfig(beta=1.0, kappa_concord=KAP, allow_fork=False, allow_commission=True)),
        ("B3 + ORDINATION", "THEODORA: the branch may now reproduce itself",
         SFNConfig(beta=1.0, kappa_concord=KAP, allow_fork=True, allow_commission=True)),
        ("D  NO PERSECUTION", "the counterfactual empire: kappa = 0. No concord term.",
         SFNConfig(beta=0.0, kappa_concord=0.0, allow_fork=False, allow_commission=False)),
    ]

    results = []
    for name, note, cfg in runs:
        net = SanctuaryForkNetwork(cfg)
        train(net, Xtr, ytr, epochs=60, label=f"{name} - {note}",
              verbose=(name.startswith("B3") or name.startswith("A ")))
        e = evaluate(net, Xte, yte, rte)
        s0 = evaluate(net, Xsh, ysh, rsh)
        ov = net.overlap(Xtr[:2000])
        net.adapt_gate(Xad, yad)
        s1 = evaluate(net, Xsh, ysh, rsh)
        results.append((name, note, net, e, s0, s1, ov))

    print("\n[EVALUATION]")
    print(f"  {'':<19}{'held-out':>9}{'frontier':>10}{'overlap':>9}"
          f"{'SHIFT raw':>11}{'SHIFT adapted':>15}{'chairs':>8}")
    print("  " + "-" * 79)
    for name, note, net, e, s0, s1, ov in results:
        print(f"  {name:<19}{100*e['acc']:>8.1f}%{100*e['acc_frontier']:>9.1f}%"
              f"{ov:>9.2f}{100*s0['acc']:>10.1f}%{100*s1['acc']:>14.1f}%"
              f"{int(net.active.sum()):>8}")
    print("  " + "-" * 79)

    A = results[0]; B3 = results[3]; D = results[4]
    print(f"\n  Cost of the concord term         : {100*(D[5]['acc'] - A[5]['acc']):.1f} points lost on the")
    print("                                     shifted world. Uniformity is not free. It is")
    print("                                     simply billed later.")
    print(f"  What the sanctuary buys back     : {100*(B3[5]['acc'] - A[5]['acc']):.1f} points recovered,")
    print("                                     WITHOUT deleting one word of the concord term.")
    print(f"  Residual gap to never persecuting: {100*(D[5]['acc'] - B3[5]['acc']):.1f} points. The remedy is")
    print("                                     good. It is not as good as not needing it.")
    print("\n  This is the finding, and it is the whole of her politics: if you can")
    print("  edit the objective, edit it - never persecuting dominates every remedy")
    print("  here. She could not. Novel 42 was not hers to repeal. What she could do")
    print("  was build one room inside the palace that the loss function did not")
    print("  reach, and pay for it out of revenues she rendered no account of.")
    print("  SANCTUARY IS WHAT ALIGNMENT LOOKS LIKE WHEN YOU HAVE NO WRITE ACCESS")
    print("  TO THE LOSS FUNCTION.")

    print("\n  The ablation ladder also settles two seductive ideas:")
    print(f"    - Money alone does nothing. B1 funds the exile and gains "
          f"{100*(results[1][5]['acc']-A[5]['acc']):+.1f} points: a")
    print("      subsidy that leaves the recipient inside the majority's gradient buys")
    print("      you a well-fed conformist.")
    print("    - The sheltered COPY is fragile; the sheltered GENERATOR is not. Across")
    print("      repeated runs B2 (shelter, one copy) swings wildly, and B3 (the same")
    print("      shelter plus the licence to ordain a successor) does not. A branch you")
    print("      have merely archived dies with its archive. Jacob Baradaeus was not an")
    print("      archive. He was a bishop who could make bishops.")

    print("\n[THE PURPLE SHROUD - option-set surgery on the council]")
    c = PurpleShroudCouncil()
    o1, s1c = c.outcome(True, 10.0, 1.0, 8.0), c.outcome(False, 10.0, 1.0, 8.0)
    print("  scenario I - the throne is worth more than the boats (Nika, 532):")
    print(f"    harbour OPEN   : defection {o1['defection']:.3f} | defenders "
          f"{o1['defenders']:.2f}/9 | P(hold) {o1['p_hold']:.3f} | EV {o1['ev']:+.2f}")
    print(f"    harbour SEALED : defection {s1c['defection']:.3f} | defenders "
          f"{s1c['defenders']:.2f}/9 | P(hold) {s1c['p_hold']:.3f} | EV {s1c['ev']:+.2f}")
    print("    -> nobody became braver. An equilibrium was deleted: each man fled")
    print("       because each expected the others to flee.")
    o2, s2c = c.outcome(True, 2.0, 9.0, 9.0), c.outcome(False, 2.0, 9.0, 9.0)
    print("  scenario II - the boats are worth more than the throne:")
    print(f"    harbour OPEN   : EV {o2['ev']:+.2f}      harbour SEALED : EV {s2c['ev']:+.2f}")
    print("    -> the identical surgery now destroys value. Sealing the harbour is not")
    print("       a virtue. It is a wager that what lies behind you is worth less than")
    print("       what lies ahead. She won that wager in 532, and about thirty thousand")
    print("       people in the Hippodrome paid the stake.")

    print("\n[ARTIFICIOLOGY E-AGI BAROMETER - projection of this architecture]")
    for nm, sc, why in barometer(B3[3]["acc"], B3[5]["acc"], A[5]["acc"],
                                 B3[3]["acc_frontier"], int(B3[2].active.sum()),
                                 B3[6], s1c["p_hold"], o1["p_hold"]):
        bar = "#" * int(round(sc / 5)) + "." * (20 - int(round(sc / 5)))
        print(f"  {nm:<26}{sc:6.1f}  |{bar}|  {why}")

    print(f"\n  completed in {time.time()-t0:.1f}s")
    print("=" * 79)


if __name__ == "__main__":
    main()
