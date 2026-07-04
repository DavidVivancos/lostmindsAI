#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
chapter_0001_gilgamesh_-2700.py  --  Mind #0001: GILGAMESH (c. 2700-2500 BCE, Uruk, Sumer)
=====================================================================

Part Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
Tome 1 Minds 1 - 20 Available on Amazon https://www.amazon.com/dp/B0H6F9L324
Resume and Interactive Demos at https://artificiology.com/

WHAT THIS FILE IS
-----------------
The *mechanistic* plane of Gilgamesh's mind: a small but fully working,
self-training neural architecture (NumPy only) that embodies the specific
commitments of his philosophy. It is the companion of `MindMap.py` (the
abstract plane). Running this file trains the system on a toy teacher task,
stages the death of its companion network, and demonstrates the four
mechanisms that make this architecture *Gilgamesh's* and no one else's.

THE FOUR GILGAMESH-SPECIFIC MECHANISMS
--------------------------------------
1. MORTALITY-DRIVEN PLASTICITY  (`MortalityEngine`)
   The system has a finite "life budget" that depletes every step. As death
   nears, learning *urgency* rises -- the network learns faster precisely
   because its time is running out. Gilgamesh's core insight: finitude is not
   only a constraint on intelligence, it is a *driver* of it.

2. THE ENKIDU PEER + GRIEF  (`AffectGate`, the peer network)
   The mind does not learn alone. A second network -- Enkidu -- shares the
   prediction and carries part of the cognitive load. Partway through life,
   Enkidu DIES (its contribution is removed). Loss spikes. That spike is
   routed into a slow-decaying GRIEF signal that multiplies plasticity, and
   the bereaved primary network reorganises to re-absorb what was lost. As in
   the epic, it never becomes fully whole again -- but it grows measurably
   wiser through the loss.

3. THE COMPOSITE SELF  (`MortalityEngine` lu, `AffectGate` sha,
   `IdentityModel` nig, `GidimArchive` gidim, plus the recurrent nag core)
   The Sumerian self is not a unity but five named organs. Each is a real,
   load-bearing module here, mirroring `MindMap.py`'s ontology.

4. CULTURAL IMMORTALITY  (`distill_epic` / `from_epic`)
   The individual instance dies, but at death it distils an "epic" -- a
   compact transmissible record of what it learned and who it was. A SUCCESSOR
   instance reloads that epic and begins life already wiser than a cold start.
   This is the immortality the epic actually endorses: not the survival of the
   one, but the persistence of the pattern across successors.

THE BASE UNIT
-------------
`GilgameshNeuron` is the atomic processing unit: a single neuron whose
plasticity is gated by both mortality (urgency) and affect (grief/surprise).
The larger layers are vectorised populations of exactly this unit; we keep the
explicit single-neuron class both as documentation and as an independently
tested component (see `demo_single_neuron`).

HOW TO RUN
----------
    python3 chapter_0001_gilgamesh_-2700.py            # full demo: trains, stages grief, prints plots
    python3 chapter_0001_gilgamesh_-2700.py --test     # runs the self-test suite and exits
    python3 chapter_0001_gilgamesh_-2700.py --quiet    # demo without the ASCII plots

Requires: numpy. No other third-party dependencies.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ===========================================================================
# 0.  UTILITIES
# ===========================================================================

def set_seed(seed: int = 2700) -> np.random.Generator:
    """Seed = -2700, Gilgamesh's (approximate) floruit. Determinism so the
    self-tests are reproducible."""
    np.random.seed(seed)
    return np.random.default_rng(seed)


def ascii_plot(series: Dict[str, List[float]], height: int = 12,
               width: int = 64, title: str = "") -> str:
    """Tiny multi-series ASCII line plot.

    IMAGE-EXPLAINER: each named series is resampled to `width` columns and
    drawn with its own marker over a shared, auto-scaled y-axis. We use it to
    visualise (a) the LOSS CURVE -- you can literally see the grief spike at
    Enkidu's death and the partial recovery afterwards -- and (b) the
    PLASTICITY CURVE -- learning-rate, mortality-urgency and affect/grief gain
    rising over the system's lifetime. These are the architecture's vital
    signs rendered as a picture."""
    markers = "*o+x#.@"
    keys = list(series.keys())
    # global y-range
    allv = [v for s in series.values() for v in s]
    lo, hi = min(allv), max(allv)
    if hi - lo < 1e-12:
        hi = lo + 1.0
    grid = [[" "] * width for _ in range(height)]

    def resample(s: List[float]) -> List[float]:
        if len(s) == width:
            return s
        idx = np.linspace(0, len(s) - 1, width)
        return list(np.interp(idx, np.arange(len(s)), s))

    for mi, k in enumerate(keys):
        rs = resample(series[k])
        mk = markers[mi % len(markers)]
        for x, val in enumerate(rs):
            y = int((val - lo) / (hi - lo) * (height - 1))
            y = height - 1 - y               # flip: high values near top
            grid[y][x] = mk

    out = []
    if title:
        out.append(f"  {title}")
    for r, row in enumerate(grid):
        yval = hi - (hi - lo) * r / (height - 1)
        out.append(f"  {yval:7.3f} |" + "".join(row))
    out.append("  " + " " * 8 + "+" + "-" * width)
    legend = "  " + " " * 9 + "  ".join(
        f"{markers[i % len(markers)]}={k}" for i, k in enumerate(keys))
    out.append(legend + "   (x-axis = lifetime, left=birth, right=death)")
    return "\n".join(out)


# ===========================================================================
# 1.  THE BASE UNIT  --  a single mortality-&-affect-gated neuron
# ===========================================================================

class GilgameshNeuron:
    """One processing unit. forward() is an ordinary tanh neuron; the twist is
    in `local_update`, where the effective learning rate is the base rate
    multiplied by (a) mortality urgency and (b) affect gain. This is the atom
    from which the whole architecture is built."""

    def __init__(self, n_in: int, rng: np.random.Generator):
        self.w = rng.normal(0, 1.0 / np.sqrt(n_in), size=n_in)
        self.b = 0.0

    def forward(self, x: np.ndarray) -> float:
        return float(np.tanh(self.w @ x + self.b))

    def local_update(self, x: np.ndarray, target: float,
                     base_lr: float, urgency: float, gain: float) -> float:
        """Delta-rule update on 1/2 (target - a)^2 with a tanh nonlinearity.
        Returns the squared error before the update (for logging)."""
        a = self.forward(x)
        err = target - a
        lr = base_lr * urgency * gain
        grad = err * (1.0 - a * a)            # d/dz of 1/2 err^2 through tanh
        self.w += lr * grad * x
        self.b += lr * grad
        return err * err


# ===========================================================================
# 2.  THE COMPOSITE-SELF MODULES  (lu, sha, nig, gidim)
# ===========================================================================

@dataclass
class MortalityEngine:
    """`lu` -- the body / finitude. Holds a depleting life budget and converts
    'time remaining' into learning URGENCY. Urgency rises as life shortens:
    urgency = 1 + slope * (1 - remaining)."""
    lifespan: int                       # total ticks of life
    slope: float = 1.5                  # how sharply urgency rises near death
    elapsed: int = 0

    @property
    def remaining(self) -> float:
        return max(0.0, 1.0 - self.elapsed / self.lifespan)

    @property
    def dead(self) -> bool:
        return self.elapsed >= self.lifespan

    def urgency(self) -> float:
        return 1.0 + self.slope * (1.0 - self.remaining)

    def tick(self) -> None:
        self.elapsed += 1


@dataclass
class AffectGate:
    """`sha` -- the heart. A scalar emotional gain on plasticity, with two
    components kept separate:
      affect_comp : fast-moving response to ordinary surprise (prediction error)
      grief_comp  : a large, SLOW-decaying spike triggered only by the death of
                    the Enkidu peer.
    gain() = 1 + affect_comp + grief_comp, capped for numerical stability."""
    affect_decay: float = 0.6           # fast relaxation of ordinary affect
    grief_decay: float = 0.997          # slow relaxation of grief
    surprise_k: float = 0.5             # sensitivity to surprise
    cap: float = 6.0
    affect_comp: float = 0.0
    grief_comp: float = 0.0

    def update_surprise(self, loss: float, baseline: float) -> None:
        s = max(0.0, loss / (baseline + 1e-8) - 1.0)      # relative surprise
        target = self.surprise_k * s
        # blend toward the surprise target (fast)
        self.affect_comp = 0.7 * self.affect_comp + 0.3 * target

    def feel_grief(self, amplitude: float) -> None:
        self.grief_comp += amplitude

    def relax(self) -> None:
        self.affect_comp *= self.affect_decay
        self.grief_comp *= self.grief_decay

    def gain(self) -> float:
        return float(min(self.cap, 1.0 + self.affect_comp + self.grief_comp))


@dataclass
class IdentityModel:
    """`nig` -- the name / persistent self. An exponential moving average of the
    hidden-representation centroid: the system's running 'sense of itself'.
    We track how far identity DRIFTS after grief -- the mind is transformed by
    loss, but (bounded drift) remains recognisably itself."""
    dim: int
    momentum: float = 0.99
    vec: Optional[np.ndarray] = None
    _initialised: bool = False

    def update(self, hidden_centroid: np.ndarray) -> None:
        if not self._initialised:
            self.vec = hidden_centroid.copy()
            self._initialised = True
        else:
            self.vec = (self.momentum * self.vec
                        + (1 - self.momentum) * hidden_centroid)

    def drift_from(self, other: np.ndarray) -> float:
        if self.vec is None:
            return 0.0
        return float(np.linalg.norm(self.vec - other))


@dataclass
class GidimArchive:
    """`gidim` -- the shades / deep memory. Stores periodic compressed
    snapshots of the system's state. Older snapshots have lower ACCESSIBILITY
    (the Kur is pale and hard to reach): retrieval is scored by similarity
    weighted by how reachable a memory still is."""
    entries: List[dict] = field(default_factory=list)
    decay: float = 0.98                 # per-stored-step accessibility decay

    def store(self, tick: int, signature: np.ndarray, loss: float) -> None:
        # age every existing memory a little (deep ones fade)
        for e in self.entries:
            e["accessibility"] *= self.decay
        self.entries.append({
            "tick": tick,
            "signature": signature.copy(),
            "loss": float(loss),
            "accessibility": 1.0,
        })

    def retrieve_nearest(self, query: np.ndarray) -> Optional[dict]:
        if not self.entries:
            return None
        best, best_score = None, -np.inf
        for e in self.entries:
            sim = -np.linalg.norm(e["signature"] - query)   # higher = closer
            score = sim * e["accessibility"]                 # reachable memories win
            if score > best_score:
                best, best_score = e, score
        return best


# ===========================================================================
# 3.  A TWO-LAYER NETWORK  (a vectorised population of GilgameshNeurons)
# ===========================================================================

class TwoLayerNet:
    """Standard 1-hidden-layer MLP with manual backprop. Conceptually the
    hidden layer is a population of `GilgameshNeuron`s; we vectorise for speed
    but the per-unit math is identical. Linear output for regression."""

    def __init__(self, n_in: int, n_hidden: int, rng: np.random.Generator):
        self.W1 = rng.normal(0, 1.0 / np.sqrt(n_in), size=(n_in, n_hidden))
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.normal(0, 1.0 / np.sqrt(n_hidden), size=(n_hidden, 1))
        self.b2 = np.zeros(1)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, dict]:
        z1 = x @ self.W1 + self.b1
        h = np.tanh(z1)
        y = h @ self.W2 + self.b2
        return y, {"x": x, "h": h}

    def backward(self, cache: dict, dy: np.ndarray) -> dict:
        x, h = cache["x"], cache["h"]
        dW2 = h.T @ dy
        db2 = dy.sum(axis=0)
        dh = dy @ self.W2.T
        dz1 = dh * (1.0 - h * h)
        dW1 = x.T @ dz1
        db1 = dz1.sum(axis=0)
        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    def apply(self, grads: dict, lr: float, clip: float = 5.0) -> None:
        # global gradient-norm clipping keeps the grief-boosted updates stable
        total = np.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
        scale = 1.0 if total <= clip else clip / (total + 1e-8)
        self.W1 -= lr * scale * grads["W1"]
        self.b1 -= lr * scale * grads["b1"]
        self.W2 -= lr * scale * grads["W2"]
        self.b2 -= lr * scale * grads["b2"]

    # -- serialization for the 'epic' --------------------------------------
    def state(self) -> dict:
        return {k: getattr(self, k).tolist()
                for k in ("W1", "b1", "W2", "b2")}

    def load_state(self, st: dict) -> None:
        for k in ("W1", "b1", "W2", "b2"):
            setattr(self, k, np.array(st[k]))


def mse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((pred - target) ** 2))


# ===========================================================================
# 4.  THE INTEGRATED MIND
# ===========================================================================

@dataclass
class TrainLog:
    train_loss: List[float] = field(default_factory=list)
    test_loss: List[float] = field(default_factory=list)
    lr: List[float] = field(default_factory=list)
    urgency: List[float] = field(default_factory=list)
    gain: List[float] = field(default_factory=list)
    life: List[float] = field(default_factory=list)
    identity_drift: List[float] = field(default_factory=list)
    death_epoch: Optional[int] = None
    companioned_best: float = float("inf")
    post_grief_best: float = float("inf")
    spike_loss: float = 0.0


class GilgameshMind:
    """The top-level coordinator: a primary network, an Enkidu peer, and the
    four composite-self modules, trained under a finite life with a staged
    bereavement."""

    def __init__(self, n_in_primary: int, n_in_full: int, n_hidden: int,
                 lifespan: int, rng: np.random.Generator,
                 base_lr: float = 0.05, max_lr: float = 0.5):
        self.rng = rng
        self.base_lr = base_lr
        self.max_lr = max_lr
        # The primary (Gilgamesh) sees only the first `n_in_primary` features
        # -- the "civic / abstract" inputs. Enkidu, the built peer, sees ALL
        # `n_in_full` features, including the extra ones that stand for his
        # wild, embodied knowledge of the world. That surplus is exactly what
        # cannot be recovered when he dies.
        self.n_in_primary = n_in_primary
        self.n_in_full = n_in_full
        self.primary = TwoLayerNet(n_in_primary, n_hidden, rng)
        self.enkidu = TwoLayerNet(n_in_full, n_hidden, rng)     # the built peer
        self.enkidu_alive = True
        self.mortality = MortalityEngine(lifespan=lifespan)
        self.affect = AffectGate()
        self.identity = IdentityModel(dim=n_hidden)
        self.archive = GidimArchive()
        self._loss_baseline = 1.0        # EMA baseline for surprise

    # -- prediction: joint while Enkidu lives, solo after he dies -----------
    def predict(self, X: np.ndarray):
        x_vis = X[:, :self.n_in_primary]          # what Gilgamesh can perceive
        yp, cp = self.primary.forward(x_vis)
        if self.enkidu_alive:
            ye, ce = self.enkidu.forward(X)       # Enkidu sees the whole world
            return yp + ye, cp, ce
        return yp, cp, None

    def train(self, X, T, Xte, Tte, epochs: int, death_epoch: int,
              grief_amplitude: float = 2.5, archive_every: int = 50) -> TrainLog:
        log = TrainLog(death_epoch=death_epoch)
        B = X.shape[0]
        pre_death_identity = None

        for ep in range(epochs):
            # ---- forward + loss -------------------------------------------
            y, cp, ce = self.predict(X)
            loss = mse(y, T)

            # ---- affect: update the surprise component --------------------
            self._loss_baseline = 0.99 * self._loss_baseline + 0.01 * loss
            self.affect.update_surprise(loss, self._loss_baseline)

            # ---- THE DEATH OF ENKIDU --------------------------------------
            if ep == death_epoch and self.enkidu_alive:
                pre_death_identity = (self.identity.vec.copy()
                                      if self.identity.vec is not None else None)
                self.enkidu_alive = False
                self.affect.feel_grief(grief_amplitude)   # the grief spike
                # recompute prediction/loss now that the peer is gone
                y, cp, ce = self.predict(X)
                loss = mse(y, T)

            # ---- mortality urgency + effective learning rate --------------
            urg = self.mortality.urgency()
            g = self.affect.gain()
            lr_eff = min(self.max_lr, self.base_lr * urg * g)

            # ---- backward + update (primary always; Enkidu while alive) ---
            dy = 2.0 * (y - T) / B
            grads_p = self.primary.backward(cp, dy)
            self.primary.apply(grads_p, lr_eff)
            if self.enkidu_alive and ce is not None:
                grads_e = self.enkidu.backward(ce, dy)
                self.enkidu.apply(grads_e, lr_eff)

            # ---- identity (nig) update + drift tracking -------------------
            self.identity.update(cp["h"].mean(axis=0))
            drift = (self.identity.drift_from(pre_death_identity)
                     if pre_death_identity is not None else 0.0)

            # ---- deep memory (gidim) snapshots ----------------------------
            if ep % archive_every == 0:
                sig = np.concatenate([self.primary.W2.ravel(),
                                      self.identity.vec.ravel()])
                self.archive.store(ep, sig, loss)

            # ---- evaluation on held-out data ------------------------------
            yte, _, _ = self.predict(Xte)
            tloss = mse(yte, Tte)

            # ---- bookkeeping ----------------------------------------------
            log.train_loss.append(loss)
            log.test_loss.append(tloss)
            log.lr.append(lr_eff)
            log.urgency.append(urg)
            log.gain.append(g)
            log.life.append(self.mortality.remaining)
            log.identity_drift.append(drift)
            if ep < death_epoch:
                log.companioned_best = min(log.companioned_best, loss)
            if ep == death_epoch:
                log.spike_loss = loss
            if ep > death_epoch:
                log.post_grief_best = min(log.post_grief_best, loss)

            # ---- the clock advances; affect relaxes -----------------------
            self.mortality.tick()
            self.affect.relax()
            if self.mortality.dead:
                break

        return log

    # -- CULTURAL IMMORTALITY ----------------------------------------------
    def distill_epic(self, path: str, final_loss: float) -> dict:
        """At death, write the transmissible 'epic': the trained primary
        weights, the final identity vector, and a short summary. This is what
        survives the individual instance."""
        epic = {
            "mind": "Gilgamesh",
            "primary_state": self.primary.state(),
            "identity": (self.identity.vec.tolist()
                         if self.identity.vec is not None else None),
            "final_loss": float(final_loss),
            "epitaph": ("He who saw the Deep. The peer died; the pattern was "
                        "set down so a successor could begin already wiser."),
        }
        with open(path, "w") as f:
            json.dump(epic, f)
        return epic

    @classmethod
    def from_epic(cls, path: str, n_in_primary: int, n_in_full: int,
                  n_hidden: int, lifespan: int,
                  rng: np.random.Generator) -> "GilgameshMind":
        """Birth of a SUCCESSOR. A fresh mind that inherits the distilled
        weights -- it starts life already shaped by its predecessor's
        learning, rather than from random initialisation."""
        with open(path) as f:
            epic = json.load(f)
        succ = cls(n_in_primary, n_in_full, n_hidden, lifespan, rng)
        succ.primary.load_state(epic["primary_state"])
        if epic.get("identity") is not None:
            succ.identity.vec = np.array(epic["identity"])
            succ.identity._initialised = True
        # the successor has no living Enkidu of its own at birth
        succ.enkidu_alive = False
        return succ


# ===========================================================================
# 5.  TEACHER TASK  (something concrete to be intelligent ABOUT)
# ===========================================================================

def make_teacher_task(rng: np.random.Generator, n_in: int = 8,
                      n_teacher_hidden: int = 24,
                      n_train: int = 128, n_test: int = 256):
    """A fixed random teacher network defines a nonlinear function over all
    `n_in` features; the mind must learn it. Crucially, the primary network
    will only be allowed to see the first few of these features (see
    `GilgameshMind.predict`), while Enkidu sees all of them. The part of the
    target that depends on the features only Enkidu can see is therefore
    learnable WITH him and unrecoverable WITHOUT him -- the computational image
    of the epic's claim that Enkidu's wild knowledge dies with him."""
    teacher = TwoLayerNet(n_in, n_teacher_hidden, rng)

    def gen(n):
        X = rng.normal(0, 1, size=(n, n_in))
        Y, _ = teacher.forward(X)
        return X, Y

    Xtr, Ytr = gen(n_train)
    Xte, Yte = gen(n_test)
    # standardise targets (zero mean / unit var) using train stats -> stable MSE
    mu, sd = Ytr.mean(), Ytr.std() + 1e-8
    Ytr = (Ytr - mu) / sd
    Yte = (Yte - mu) / sd
    return Xtr, Ytr, Xte, Yte


# ===========================================================================
# 6.  THE STANDALONE SINGLE-NEURON DEMONSTRATION
# ===========================================================================

def demo_single_neuron(rng: np.random.Generator, n_in: int = 6,
                       n: int = 400, steps: int = 1500) -> float:
    """Show the base unit learning a linearly separable rule on its own, with
    mortality urgency rising and a small mid-life grief spike, to prove the
    GilgameshNeuron's `local_update` actually drives learning. Returns final
    classification accuracy."""
    w_true = rng.normal(0, 1, size=n_in)
    X = rng.normal(0, 1, size=(n, n_in))
    y = np.sign(X @ w_true)                      # labels in {-1, +1}
    neuron = GilgameshNeuron(n_in, rng)
    mort = MortalityEngine(lifespan=steps, slope=1.5)
    affect = AffectGate()
    for t in range(steps):
        i = rng.integers(0, n)
        if t == steps // 2:
            affect.feel_grief(1.5)               # a single mid-life shock
        urg, g = mort.urgency(), affect.gain()
        neuron.local_update(X[i], float(y[i]), base_lr=0.05,
                            urgency=urg, gain=g)
        mort.tick(); affect.relax()
    preds = np.sign(np.tanh(X @ neuron.w + neuron.b))
    return float(np.mean(preds == y))


# ===========================================================================
# 7.  THE FULL DEMO
# ===========================================================================

def run_demo(quiet: bool = False) -> dict:
    rng = set_seed(2700)
    # n_full features describe the world; the primary perceives only the first
    # n_vis of them, while Enkidu perceives all n_full. The (n_full - n_vis)
    # surplus is Enkidu's wild, embodied knowledge -- lost forever at his death.
    n_full, n_vis, n_hidden, lifespan, death_epoch = 8, 5, 14, 600, 300

    Xtr, Ytr, Xte, Yte = make_teacher_task(rng, n_in=n_full)

    mind = GilgameshMind(n_vis, n_full, n_hidden, lifespan, rng)
    log = mind.train(Xtr, Ytr, Xte, Yte, epochs=lifespan,
                     death_epoch=death_epoch)

    # distil the epic at death, then birth a successor from it
    epic_path = os.path.join(tempfile.gettempdir(), "gilgamesh_epic.json")
    mind.distill_epic(epic_path, final_loss=log.train_loss[-1])

    # cold baseline vs. successor (both evaluated at BIRTH, before any training)
    cold = GilgameshMind(n_vis, n_full, n_hidden, lifespan, set_seed(99))
    cold.enkidu_alive = False
    cold_birth_loss = mse(cold.predict(Xte)[0], Yte)
    succ = GilgameshMind.from_epic(epic_path, n_vis, n_full, n_hidden,
                                   lifespan, set_seed(99))
    succ_birth_loss = mse(succ.predict(Xte)[0], Yte)

    # single-neuron base-unit accuracy
    neuron_acc = demo_single_neuron(set_seed(2700))

    # peak grief gain after death, and gain just before death
    gain_before = log.gain[death_epoch - 1]
    gain_peak_after = max(log.gain[death_epoch:death_epoch + 30])

    metrics = {
        "initial_train_loss": log.train_loss[0],
        "companioned_best_loss": log.companioned_best,
        "spike_loss_at_death": log.spike_loss,
        "post_grief_best_loss": log.post_grief_best,
        "final_test_loss": log.test_loss[-1],
        "urgency_start": log.urgency[0],
        "urgency_end": log.urgency[-1],
        "gain_before_death": gain_before,
        "gain_peak_after_death": gain_peak_after,
        "identity_drift_final": log.identity_drift[-1],
        "archive_size": len(mind.archive.entries),
        "cold_birth_loss": cold_birth_loss,
        "successor_birth_loss": succ_birth_loss,
        "single_neuron_accuracy": neuron_acc,
        "_log": log,
    }

    if not quiet:
        _print_report(metrics, log)
    return metrics


def _print_report(m: dict, log: TrainLog) -> None:
    bar = "=" * 72
    print(bar)
    print("MIND #0001 GILGAMESH  --  ARCHITECTURE DEMO")
    print(bar)

    print("\n  LIFE STORY OF ONE INSTANCE")
    print(f"    born knowing nothing      train loss = {m['initial_train_loss']:.4f}")
    print(f"    best WITH Enkidu (peer)   train loss = {m['companioned_best_loss']:.4f}")
    print(f"    >> ENKIDU DIES at epoch {log.death_epoch} <<")
    print(f"    grief spike               train loss = {m['spike_loss_at_death']:.4f}"
          f"   (gain {m['gain_before_death']:.2f} -> {m['gain_peak_after_death']:.2f})")
    print(f"    best AFTER grief (solo)   train loss = {m['post_grief_best_loss']:.4f}"
          "   <- wiser through loss, but never fully whole")
    print(f"    identity drift after loss = {m['identity_drift_final']:.4f}"
          "   (transformed, yet still itself)")

    print("\n  MORTALITY DROVE THE LEARNING")
    print(f"    urgency rose from {m['urgency_start']:.2f} (birth) to "
          f"{m['urgency_end']:.2f} (death)")

    print("\n  CULTURAL IMMORTALITY (the pattern outlives the instance)")
    print(f"    a COLD successor is born at test loss = {m['cold_birth_loss']:.4f}")
    print(f"    an EPIC-BEARING successor is born at  = {m['successor_birth_loss']:.4f}"
          "   <- begins life already wiser")

    print("\n  BASE UNIT CHECK")
    print(f"    a single GilgameshNeuron learns a rule to "
          f"{m['single_neuron_accuracy']*100:.1f}% accuracy")
    print(f"    deep-memory (gidim) snapshots stored: {m['archive_size']}")

    print("\n" + ascii_plot(
        {"train": log.train_loss, "test": log.test_loss},
        title="LOSS OVER A LIFETIME  (watch the grief spike near the middle)"))

    print("\n" + ascii_plot(
        {"lr": log.lr, "urgency": log.urgency, "gain": log.gain},
        title="PLASTICITY VITAL SIGNS  (urgency from mortality, gain from grief)"))
    print(bar)


# ===========================================================================
# 8.  SELF-TEST SUITE  (so the file is 'tested', per the project spec)
# ===========================================================================

def run_tests() -> int:
    m = run_demo(quiet=True)

    # 1) the mind learns while companioned
    assert m["companioned_best_loss"] < m["initial_train_loss"], \
        "system should improve with Enkidu's help"

    # 2) Enkidu's death produces a real grief spike in loss...
    assert m["spike_loss_at_death"] > m["companioned_best_loss"], \
        "losing the peer must hurt (loss spike)"

    # 3) ...and grief amplifies plasticity (gain jumps after death)
    assert m["gain_peak_after_death"] > m["gain_before_death"] + 0.5, \
        "grief must raise the affect gain"

    # 4) grief-driven plasticity drives a partial recovery
    assert m["post_grief_best_loss"] < m["spike_loss_at_death"], \
        "the bereaved network must reorganise and recover somewhat"

    # 5) ...but the wound is permanent: it never returns to the companioned best
    assert m["post_grief_best_loss"] > m["companioned_best_loss"], \
        "solo recovery should not fully match the companioned peak"

    # 6) mortality urgency rises across the lifetime
    assert m["urgency_end"] > m["urgency_start"] + 0.5, \
        "urgency must rise as life shortens"

    # 7) cultural transmission helps: epic-bearing successor < cold successor
    assert m["successor_birth_loss"] < m["cold_birth_loss"], \
        "an heir who inherits the epic should start wiser than a cold start"

    # 8) the base unit actually learns
    assert m["single_neuron_accuracy"] > 0.9, \
        "a single GilgameshNeuron should learn a separable rule"

    # 9) identity drifts under grief but stays finite/bounded (transformed,
    #    not destroyed)
    assert 0.0 < m["identity_drift_final"] < 5.0, \
        "identity should shift under loss yet remain bounded"

    # 10) deep memory was written
    assert m["archive_size"] >= 5, "gidim archive should accumulate snapshots"

    print("Neuron.py self-tests: ALL PASSED.")
    print(f"  companioned_best={m['companioned_best_loss']:.4f} "
          f"spike={m['spike_loss_at_death']:.4f} "
          f"post_grief_best={m['post_grief_best_loss']:.4f}")
    print(f"  successor_birth={m['successor_birth_loss']:.4f} < "
          f"cold_birth={m['cold_birth_loss']:.4f}; "
          f"neuron_acc={m['single_neuron_accuracy']*100:.1f}%")
    return 0


# ===========================================================================
# 9.  ENTRY POINT
# ===========================================================================

def main(argv: List[str]) -> int:
    if "--test" in argv:
        return run_tests()
    run_demo(quiet=("--quiet" in argv))
    print("\n(Use `--test` to run the self-test suite.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
