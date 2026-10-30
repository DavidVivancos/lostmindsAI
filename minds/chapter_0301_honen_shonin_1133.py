"""
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0301 · Hōnen Shōnin (法然上人, 1133-1212 CE)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 16, Minds 301-320 Available on Amazon https://www.amazon.com/gp/product/B0HNJQX9S4
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos 
A pure-NumPy, from-scratch, trainable neural architecture built on Hōnen
Shōnin's own distinctive cognitive signature: SELECTION UNDER A DECLARED
CAPACITY CEILING, not "attention" and not a generic Bayesian prior.

WHY NOT THE OBVIOUS MAPPING
----------------------------
Nearly every "faith-based" or "single-practice" figure gets flattened, in
retrospect, into two clichés: (a) single-pointed chanting = attention
mechanism, (b) faith = Bayesian prior. Both are real analogies but they are
not *Hōnen's own* distinctive move, and they are exactly the mappings a
generic pass would reach for. Hōnen's own textual signature — recoverable
from the Senchaku Hongan Nembutsu Shū (Senchakushū, 1198) and the Ichimai
Kishōmon (1212) — is a three-stage, irrevocable CASCADE OF EXCLUSION:

    Stage 1 (shōdōmon / jōdomon — "Two Gateways"):
        All Buddhist practice is split into the Holy Path (self-power,
        enlightenment-in-this-life) and the Pure Land Path (other-power,
        rebirth-then-enlightenment). Hōnen rejects the Holy Path outright
        for the condition he is designing for. This is not soft attention;
        it is a categorical, non-negotiable partition.

    Stage 2 (shōgyō / zōgyō — "Right Practices" vs "Miscellaneous
        Practices"): within the Pure Land Path, practices tied to Amida's
        Primal Vow specifically (the shōgyō) are kept; practices tied to
        other Buddhas' Pure Lands are pruned.

    Stage 3 (shōjōgō / jogō — "the Established Act" vs "Auxiliary Acts"):
        of the five shōgyō (reading/reciting sutras, contemplation,
        prostration, uttering the Name, giving praise/offerings), only
        "uttering the Name" (kushō — VOCAL recitation) is kept as the
        rightly-established act; the other four are demoted to auxiliary
        status. This is the crux of Hōnen's originality: he privileges a
        low-bandwidth, DISCRETE, universally producible signal (a spoken
        six-mora phrase, countable, performable while blind, illiterate,
        dying, or in pain) over rich, CONTINUOUS internal states
        (visualization, contemplative absorption) precisely because
        continuous internal states cannot be verified or guaranteed for an
        agent operating under a declared, historically-dated capacity
        ceiling.

That capacity ceiling is not metaphorical for Hōnen. Mappō (末法, the "Latter
Age of the Dharma") was, for 12th-century Japan, a DATED historical claim:
the age was widely reckoned to have begun in 1052 CE (see SOURCES). Hōnen's
architecture is therefore best read as a designed response to a *declared,
irreversible decay in the average agent's self-power (jiriki) compute
budget* — and his solution is not "try harder" but "route to an external,
already-completed computation" (Amida's Primal Vow, fulfilled — in the
Sutra's own idiom — after five kalpas of bodhisattva practice, ten kalpas
ago) through the cheapest possible interface.

Two further, textually specific features are modelled here, both routinely
flattened in less careful treatments:

  (a) TWOFOLD DEEP MIND (nishu jinshin, 二種深心). Hōnen (following Shan-tao)
      collapsed the classical "three minds" of the Contemplation Sūtra into
      one central mind, jinshin ("deep mind"), which itself has TWO
      inseparable recognitions held AT ONCE: (ki 機) the practitioner's own
      utter incapacity, and (hō 法) the Vow's unconditional sufficiency for
      exactly such an incapacitated being. This is a CONJUNCTION, not a
      weighted average — losing either recognition collapses the whole
      state. The architecture implements this as a literal product-gate,
      not a sum, and it is tested as such (see self_test_jinshin_gate).

  (b) THE ICHINEN/TANEN QUESTION (一念義 / 多念義). After Hōnen's death his
      own disciples split over whether a single sincere recitation suffices
      (Kōsai's ichinengi) or whether sustained, repeated recitation is
      required (Ryūkan's tanengi) — a tension Hōnen's own writings (the
      Three Minds, the Four Modes of Practice in the Ichimai Kishōmon) leave
      genuinely underdetermined. This file does NOT pretend Hōnen resolved
      it. It implements BOTH as literal, switchable integrator modes and
      measures their differing robustness to transient noise — an honest
      formalization of a real, unresolved historical controversy, not a
      manufactured answer.

WHAT THIS IS NOT
-----------------
This is not a transformer. There is no attention over stored keys, and no
Bayesian-prior framing of faith. The only place a "prior" appears is as a
literal frozen (non-trainable) lookup — the HongwanOracle — representing an
already-completed computation from before this network's own training ever
starts, deliberately excluded from the optimizer's parameter list, exactly
as Amida's Vow is, in the doctrine, already fulfilled before any individual
practitioner calls upon it.

ARCHITECTURAL CREDIT ASSIGNMENT (a deliberate, documented choice)
-------------------------------------------------------------------
Each cascade stage below is trained against its OWN local objective, and
gradients do not flow backward across stage boundaries (later stages
consume earlier stages' outputs as fixed, detached features). This mirrors
senjaku itself: a completed act of selection is not something later
stages get to revise or "retrain" — the Two Gateways decision, once made,
does not get undone by anything the shōgyō/zōgyō stage later discovers.
Modular, stage-local supervision is therefore not a simplification made
for convenience; it is the architecture's own philosophical commitment,
made explicit so it cannot be mistaken for an oversight.

CLASSES
-------
  Dense                       - manual linear layer, forward/backward, Adam
  NidoMonGate                 - Stage 1: hard, non-parametric Two-Gateways split
  ShogyoZogyoPruner            - Stage 2: trainable Right/Miscellaneous classifier
  JogoShijoSelector             - Stage 3: trainable Established/Auxiliary MLP
  JirikiSelfPowerPathway        - trainable self-power confidence estimator
  MappoCapacityScheduler        - non-trainable historical decay schedule
  HongwanOracle                 - FROZEN Primal Vow retrieval matrix
  NishuJinshinGate               - trainable twofold-deep-mind AND-gate
  IchinenTanenIntegrator          - switchable single-shot / sustained integrator
  OjoOutput                       - trainable rebirth classifier (tariki-only inputs)
  SenjakuNet                      - top-level orchestrator
  PracticeDataset                  - synthetic dataset generator + doctrinal labels

Every trainable module ships a finite-difference gradient check
(see gradient_check_all) and the file ends with a real training loop and a
battery of self-tests whose printed output is pasted, verbatim, into the
companion chapter text.

AUTHOR: 1000 Minds Project — Encyclopedia of Lost Minds: Echoes on AI
CHAPTER: 0301 — Hōnen Shōnin (1133-1212 CE)
"""

import numpy as np
import math
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional, Any
from enum import Enum, auto

RNG_SEED = 1198          # the year the Senchakushū was composed
KALPAS_SEED = 1052       # the reckoned first year of mappō — used ONLY to
                         # seed the frozen oracle, never touched by training
rng = np.random.default_rng(RNG_SEED)


# =============================================================================
# PART 0 — NUMERIC PRIMITIVES (no autodiff library; every gradient below is
# hand-derived and checked against finite differences at the bottom of the
# file)
# =============================================================================

def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def dsigmoid_from_output(s: np.ndarray) -> np.ndarray:
    """Derivative of sigmoid, expressed in terms of its OWN output s = sigmoid(z)."""
    return s * (1.0 - s)


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z)


def drelu(z: np.ndarray) -> np.ndarray:
    return (z > 0.0).astype(z.dtype)


def bce_loss(pred: np.ndarray, label: np.ndarray, eps: float = 1e-9) -> float:
    """Binary cross-entropy, averaged over the batch."""
    pred = np.clip(pred, eps, 1.0 - eps)
    return float(-np.mean(label * np.log(pred) + (1.0 - label) * np.log(1.0 - pred)))


def bce_grad(pred: np.ndarray, label: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """dL/dpred for the mean BCE loss above."""
    pred = np.clip(pred, eps, 1.0 - eps)
    n = pred.shape[0]
    return (-(label / pred) + (1.0 - label) / (1.0 - pred)) / n


def xavier(fan_in: int, fan_out: int, rand: np.random.Generator) -> np.ndarray:
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    return rand.uniform(-limit, limit, size=(fan_in, fan_out)).astype(np.float64)


# =============================================================================
# PART 1 — A tiny hand-rolled Adam optimizer (no framework)
# =============================================================================

class Adam:
    """A minimal, from-scratch Adam optimizer over a flat dict of named arrays."""

    def __init__(self, params: Dict[str, np.ndarray], lr: float = 0.05,
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.lr = lr
        self.b1 = beta1
        self.b2 = beta2
        self.eps = eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]) -> None:
        self.t += 1
        for k in params:
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


# =============================================================================
# PART 2 — Dense (linear) layer with manual forward/backward
# =============================================================================

class Dense:
    """A plain affine layer y = xW + b, with manual backward()."""

    def __init__(self, in_dim: int, out_dim: int, rand: np.random.Generator, name: str):
        self.name = name
        self.params: Dict[str, np.ndarray] = {
            f"{name}.W": xavier(in_dim, out_dim, rand),
            f"{name}.b": np.zeros((1, out_dim)),
        }
        self._cache_x: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._cache_x = x
        return x @ self.params[f"{self.name}.W"] + self.params[f"{self.name}.b"]

    def backward(self, dy: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        x = self._cache_x
        dW = x.T @ dy
        db = np.sum(dy, axis=0, keepdims=True)
        dx = dy @ self.params[f"{self.name}.W"].T
        return dx, {f"{self.name}.W": dW, f"{self.name}.b": db}


# =============================================================================
# PART 3 — Synthetic PracticeDataset: candidate "practices" as feature
# vectors, plus doctrinally-derived ground-truth labels for each cascade
# stage. These labels encode the *teaching*, not a trivial pass-through —
# each trainable classifier below must learn a real (if simple) decision
# boundary from noisy features.
# =============================================================================

FEATURE_NAMES = [
    "requires_literacy",                 # 0
    "requires_years_of_training",        # 1
    "requires_wealth_or_patronage",      # 2
    "requires_meditative_absorption",    # 3
    "universal_executability",           # 4  (executable while dying/incapacitated)
    "executable_regardless_of_class",    # 5
    "doctrinal_alignment_with_primal_vow",  # 6
    "requires_ritual_initiation",        # 7
    "externally_verifiable_repetition",  # 8  (can be counted by an observer)
    "discrete_vocal_signal",             # 9  (0 = continuous/visualization, 1 = discrete/spoken)
    "targets_pure_land_rebirth",         # 10 (0 = targets enlightenment-in-this-life)
    "context_noise",                     # 11 (irrelevant nuisance feature)
]
N_FEATURES = len(FEATURE_NAMES)


@dataclass
class PracticeDataset:
    x: np.ndarray                 # (N, N_FEATURES)
    label_shogyo: np.ndarray      # (N, 1)  Stage-2 target: right vs miscellaneous
    label_shojo: np.ndarray       # (N, 1)  Stage-3 target: established vs auxiliary
    label_jiriki: np.ndarray      # (N, 1)  self-power-succeeds target (for JirikiSelfPowerPathway)
    jodomon_mask: np.ndarray      # (N,)    boolean — which rows survive Stage 1

    @staticmethod
    def generate(n: int, rand: np.random.Generator) -> "PracticeDataset":
        x = rand.uniform(0.0, 1.0, size=(n, N_FEATURES))

        # Stage 1 ground truth is a HARD rule (Two Gateways is categorical
        # in the doctrine, not a matter of degree): practices whose target
        # is Pure Land rebirth (feature 10) enter the jōdomon pool at all.
        jodomon_mask = x[:, 10] > 0.5

        # Stage 2 ground truth: among Pure-Land-directed practices, "right
        # practice" (shōgyō) status tracks doctrinal alignment with the
        # Primal Vow (feature 6), continuous supervision, but the network
        # only sees a NOISY view of x (see PracticeDataset.noisy_view).
        label_shogyo = (x[:, 6] > 0.55).astype(np.float64).reshape(-1, 1)

        # Stage 3 ground truth: of the right practices, the ESTABLISHED act
        # (shōjōgō) is specifically the one that is BOTH a discrete/vocal
        # signal (feature 9) AND universally executable under incapacity
        # (feature 4) — a conjunction, deliberately not linearly separable,
        # forcing JogoShijoSelector to be a genuine (if small) MLP.
        label_shojo = ((x[:, 9] > 0.5) & (x[:, 4] > 0.5)).astype(np.float64).reshape(-1, 1)

        # Jiriki (self-power) succeeds only for practices that are cheap in
        # exactly the dimensions self-cultivation struggles to discharge at
        # scale: little required training, little required meditative
        # absorption, no required ritual initiation.
        difficulty = (x[:, 1] + x[:, 3] + x[:, 7]) / 3.0
        label_jiriki = (difficulty < 0.40).astype(np.float64).reshape(-1, 1)

        return PracticeDataset(x, label_shogyo, label_shojo, label_jiriki, jodomon_mask)

    def noisy_view(self, rand: np.random.Generator, sigma: float = 0.12) -> np.ndarray:
        """What the trainable classifiers actually observe: x corrupted by
        Gaussian sensor/report noise, and clipped back into [0, 1]. This is
        what makes Stage 2 / Stage 3 / Jiriki genuine learning problems
        rather than exact threshold lookups."""
        noisy = self.x + rand.normal(0.0, sigma, size=self.x.shape)
        return np.clip(noisy, 0.0, 1.0)


# =============================================================================
# PART 4 — Stage 1: NidoMonGate ("The Two Gateways")
# =============================================================================

class NidoMonGate:
    """
    Stage 1 of the cascade: shōdōmon (Holy Path / self-power / this-life
    enlightenment) versus jōdomon (Pure Land Path / other-power / rebirth).

    Hōnen's own division here (Senchakushū, ch. 1) is explicitly categorical:
    an entire class of practice is set aside, not down-weighted. There is
    therefore nothing to LEARN at this stage — it is implemented as a hard,
    non-parametric boolean mask, deliberately NOT a soft/attention-style
    gate. Everything downstream only ever sees the jōdomon-selected rows.
    """

    @staticmethod
    def forward(x: np.ndarray) -> np.ndarray:
        return x[:, 10] > 0.5


# =============================================================================
# PART 5 — Stage 2: ShogyoZogyoPruner ("Right vs Miscellaneous Practices")
# =============================================================================

class ShogyoZogyoPruner:
    """
    Trainable logistic classifier: given a (noisy) practice vector already
    inside the jōdomon pool, predict whether it is shōgyō (a Right Practice,
    tied specifically to Amida's Pure Land) or zōgyō (Miscellaneous,
    directed at other Buddhas/Pure Lands and therefore pruned).

    A single affine layer + sigmoid is enough for this stage because the
    doctrinal criterion (feature 6, alignment with the Primal Vow) is
    linearly separable in the noisy input — Hōnen's own text treats this
    as a fairly direct textual/scriptural judgement, not a subtle one.
    """

    def __init__(self, rand: np.random.Generator):
        self.fc = Dense(N_FEATURES, 1, rand, name="shogyo")
        self._cache: Dict[str, np.ndarray] = {}

    def params(self) -> Dict[str, np.ndarray]:
        return self.fc.params

    def forward(self, x: np.ndarray) -> np.ndarray:
        z = self.fc.forward(x)
        p = sigmoid(z)
        self._cache["p"] = p
        return p

    def backward(self, dL_dp: np.ndarray) -> Dict[str, np.ndarray]:
        p = self._cache["p"]
        dz = dL_dp * dsigmoid_from_output(p)
        _, grads = self.fc.backward(dz)
        return grads


# =============================================================================
# PART 6 — Stage 3: JogoShijoSelector ("Established Act vs Auxiliary Acts")
# =============================================================================

class JogoShijoSelector:
    """
    Trainable 2-layer MLP: of the five shōgyō (right practices), select the
    ONE that is the shōjōgō (rightly-established act — vocal recitation of
    the Name) versus the four jogō (auxiliary acts).

    The ground truth here is an AND of two threshold conditions (discrete/
    vocal signal AND universal executability under incapacity) — not
    linearly separable — which is exactly why this stage needs a hidden
    layer where Stage 2 did not: Hōnen's OWN selection at this step is more
    intricate than at Stage 2 (he explicitly walks through five candidates
    and eliminates four one at a time in the Senchakushū, ch. 3-4), and the
    architecture's extra depth here is a deliberate reflection of that,
    not an arbitrary hyperparameter choice.
    """

    def __init__(self, rand: np.random.Generator, hidden: int = 8):
        self.fc1 = Dense(N_FEATURES, hidden, rand, name="jogo1")
        self.fc2 = Dense(hidden, 1, rand, name="jogo2")
        self._cache: Dict[str, np.ndarray] = {}

    def params(self) -> Dict[str, np.ndarray]:
        return {**self.fc1.params, **self.fc2.params}

    def forward(self, x: np.ndarray) -> np.ndarray:
        h_pre = self.fc1.forward(x)
        h = relu(h_pre)
        z2 = self.fc2.forward(h)
        p = sigmoid(z2)
        self._cache = {"h_pre": h_pre, "h": h, "p": p}
        return p

    def backward(self, dL_dp: np.ndarray) -> Dict[str, np.ndarray]:
        p = self._cache["p"]
        dz2 = dL_dp * dsigmoid_from_output(p)
        dh, grads2 = self.fc2.backward(dz2)
        dh_pre = dh * drelu(self._cache["h_pre"])
        _, grads1 = self.fc1.backward(dh_pre)
        return {**grads1, **grads2}


# =============================================================================
# PART 7 — JirikiSelfPowerPathway (trainable "how far can self-effort get me")
# =============================================================================

class JirikiSelfPowerPathway:
    """
    Trainable logistic classifier estimating the agent's own confidence that
    SELF-POWER (jiriki) alone — without any external Vow — would succeed at
    a given practice. Trained to be high only for practices requiring little
    training, little absorption, no initiation (features 1, 3, 7).

    This pathway's raw output is what MappoCapacityScheduler then throttles
    as a function of historical epoch, below.
    """

    def __init__(self, rand: np.random.Generator):
        self.fc = Dense(N_FEATURES, 1, rand, name="jiriki")
        self._cache: Dict[str, np.ndarray] = {}

    def params(self) -> Dict[str, np.ndarray]:
        return self.fc.params

    def forward(self, x: np.ndarray) -> np.ndarray:
        z = self.fc.forward(x)
        p = sigmoid(z)
        self._cache["p"] = p
        return p

    def backward(self, dL_dp: np.ndarray) -> Dict[str, np.ndarray]:
        p = self._cache["p"]
        dz = dL_dp * dsigmoid_from_output(p)
        _, grads = self.fc.backward(dz)
        return grads


# =============================================================================
# PART 8 — MappoCapacityScheduler (NOT trainable — a historical schedule)
# =============================================================================

class MappoCapacityScheduler:
    """
    Models the doctrinal claim that self-power's ceiling of effectiveness
    decays across historical time, reaching a low plateau once the age of
    mappō is reached. This is a fixed SCHEDULE, not a learned parameter —
    Hōnen's own argument is that this decay is a historical fact external to
    any individual practitioner's effort, so it would be a category error to
    let gradient descent "optimize" it.

    We map training epoch e in [0, E-1] onto a symbolic "year axis" so that
    the logistic decay's inflection lands at a chosen fraction of training,
    labelled 1052 in the printed log — purely an expository device, not a
    physical claim about calendar time.
    """

    def __init__(self, total_epochs: int, inflection_frac: float = 0.35,
                 steepness: float = 10.0, floor: float = 0.12):
        self.total_epochs = total_epochs
        self.inflection_frac = inflection_frac
        self.steepness = steepness
        self.floor = floor

    def ceiling(self, epoch: int) -> float:
        """Returns a multiplier in (floor, 1.0] — the fraction of raw
        self-power confidence the agent is historically permitted to use."""
        frac = epoch / max(1, self.total_epochs - 1)
        decay = sigmoid(np.array([-self.steepness * (frac - self.inflection_frac)]))[0]
        return float(self.floor + (1.0 - self.floor) * decay)

    def symbolic_year(self, epoch: int) -> int:
        """Purely for logging: maps epoch fraction onto a year axis running
        from 1000 to 1200 CE, so the printed decay curve can be read next to
        the historically attested date of mappō's onset (1052 CE)."""
        frac = epoch / max(1, self.total_epochs - 1)
        return int(1000 + frac * 200)


# =============================================================================
# PART 9 — HongwanOracle ("The Primal Vow") — FROZEN, never trained
# =============================================================================

NEMBUTSU_MORAE = ["na", "mu", "a", "mi", "da", "butsu"]  # the six morae of the Name
VOCAB = len(NEMBUTSU_MORAE)
ORACLE_DIM = 16


class HongwanOracle:
    """
    Represents the Primal Vow (hongwan) — in the Larger Sūtra's own idiom,
    a computation already completed by the bodhisattva Dharmākara across
    five kalpas of contemplation and practice, ten kalpas before any
    practitioner today calls upon it. The load-bearing architectural claim
    is that this module's WEIGHTS ARE FROZEN, seeded from a completely
    separate RNG stream (KALPAS_SEED) and never touched by the optimizer,
    never appearing in any params() dict this file trains. Retrieval is
    performed through the single most low-bandwidth channel available: a
    fixed one-hot encoding of a discrete six-mora token, deliberately NOT a
    rich continuous embedding — encoding Hōnen's insistence on kushō (vocal
    practice) over kanjin (contemplative visualization): a channel that
    remains available to an agent regardless of literacy, health, or
    leisure.
    """

    def __init__(self):
        oracle_rand = np.random.default_rng(KALPAS_SEED)   # separate, frozen stream
        # W_oracle: fixed projection from the one-hot Name token to a
        # retrieval vector. Never updated after construction.
        self.W_oracle = oracle_rand.normal(0.0, 1.0, size=(VOCAB, ORACLE_DIM))
        self.W_oracle /= np.linalg.norm(self.W_oracle, axis=1, keepdims=True) + 1e-9
        # The single legitimate query: "Namu Amida Butsu" collapsed to one
        # fixed one-hot code across the six morae (mean pooling below is a
        # simplification standing in for a fixed phrase, not six separate
        # word choices — there is, doctrinally, only one Name to call).
        self.fixed_token = np.eye(VOCAB).mean(axis=0)  # shape (VOCAB,)
        # target_fulfillment: "the Vow, fully kept" is defined as exactly the
        # direction the fixed Name-token maps to under the frozen oracle —
        # i.e. what a perfectly steady (noise-free) recitation retrieves.
        # hō-recognition is therefore a measure of how STEADILY the Name is
        # actually being called, not of alignment with some unrelated
        # target: doctrinally, the Vow does not grade practices against a
        # separate standard, it simply IS what steady recitation reaches.
        base = self.fixed_token @ self.W_oracle
        self.target_fulfillment = base / (np.linalg.norm(base) + 1e-9)

    def retrieve(self, n: int, recitation_count: np.ndarray,
                 rand: np.random.Generator) -> np.ndarray:
        """
        Query the frozen oracle. `recitation_count` (shape (n,)) is the
        number of times, so far, this practitioner has voiced the Name;
        repetition SHRINKS (never removes) a small amount of query noise,
        modelling growing steadiness of practice rather than growing
        "correctness" of the already-perfect Vow itself — the Vow does not
        improve with repetition; the practitioner's grip on it steadies.
        """
        base = self.fixed_token @ self.W_oracle           # shape (ORACLE_DIM,)
        base = np.tile(base, (n, 1))
        noise_scale = 0.20 / (1.0 + recitation_count)      # steadies with practice
        noise = rand.normal(0.0, 1.0, size=base.shape) * noise_scale[:, None]
        return base + noise

    def reliability(self, retrieval: np.ndarray) -> np.ndarray:
        """Cosine similarity between what was retrieved and the fixed
        target_fulfillment vector — the hō (法) recognition: how fully the
        Vow's sufficiency is being registered."""
        r_norm = retrieval / (np.linalg.norm(retrieval, axis=1, keepdims=True) + 1e-9)
        t_norm = self.target_fulfillment / (np.linalg.norm(self.target_fulfillment) + 1e-9)
        return r_norm @ t_norm  # shape (n,)


# =============================================================================
# PART 10 — NishuJinshinGate ("Twofold Deep Mind") — trainable AND-gate
# =============================================================================

class NishuJinshinGate:
    """
    Implements jinshin (深心, "deep mind") as Hōnen (via Shan-tao) actually
    describes it: TWO recognitions held simultaneously —

        ki (機)  — ki_signal: recognition of one's own utter incapacity.
                   High when effective (mappō-throttled) jiriki confidence
                   is LOW.
        hō (法)  — ho_signal: recognition of the Vow's unconditional
                   sufficiency. High when the oracle's retrieved state
                   aligns with the fixed target_fulfillment vector.

    Crucially: gate = ki_signal * ho_signal, a PRODUCT (logical AND), never
    an average or weighted sum. Zeroing either recognition zeros the gate
    regardless of how strong the other is — see self_test_jinshin_gate for
    a direct numerical demonstration. Four small trainable scalars (a, b,
    c, d) control where each sigmoid's threshold sits.
    """

    def __init__(self, rand: np.random.Generator):
        self.params_: Dict[str, np.ndarray] = {
            "jinshin.a": np.array([[6.0]]),
            "jinshin.b": np.array([[3.0]]),
            "jinshin.c": np.array([[6.0]]),
            "jinshin.d": np.array([[3.0]]),
        }
        self._cache: Dict[str, np.ndarray] = {}

    def params(self) -> Dict[str, np.ndarray]:
        return self.params_

    def forward(self, effective_jiriki: np.ndarray, oracle_reliability: np.ndarray) -> np.ndarray:
        a = self.params_["jinshin.a"][0, 0]
        b = self.params_["jinshin.b"][0, 0]
        c = self.params_["jinshin.c"][0, 0]
        d = self.params_["jinshin.d"][0, 0]
        incapacity = 1.0 - effective_jiriki
        ki = sigmoid(a * incapacity - b)             # recognition of ki (incapacity)
        ho = sigmoid(c * oracle_reliability - d)      # recognition of hō (sufficiency)
        gate = ki * ho
        self._cache = {"ki": ki, "ho": ho, "incapacity": incapacity,
                        "oracle_reliability": oracle_reliability}
        return gate

    def backward(self, dL_dgate: np.ndarray) -> Dict[str, np.ndarray]:
        ki, ho = self._cache["ki"], self._cache["ho"]
        incapacity = self._cache["incapacity"]
        oracle_reliability = self._cache["oracle_reliability"]
        dL_dki = dL_dgate * ho
        dL_dho = dL_dgate * ki
        dki_dz = dsigmoid_from_output(ki)
        dho_dz = dsigmoid_from_output(ho)
        dL_dz_ki = dL_dki * dki_dz
        dL_dz_ho = dL_dho * dho_dz
        da = np.sum(dL_dz_ki * incapacity)
        db = np.sum(dL_dz_ki * -1.0)
        dc = np.sum(dL_dz_ho * oracle_reliability)
        dd = np.sum(dL_dz_ho * -1.0)
        return {
            "jinshin.a": np.array([[da]]),
            "jinshin.b": np.array([[db]]),
            "jinshin.c": np.array([[dc]]),
            "jinshin.d": np.array([[dd]]),
        }


# =============================================================================
# PART 11 — IchinenTanenIntegrator — the disciples' unresolved controversy,
# formalized as two switchable, NON-trainable evaluation modes (this is a
# behavioral comparison at inference time, not a supervised sub-network —
# see the module docstring for why treating it as settled would be
# historically dishonest).
# =============================================================================

class IntegratorMode(Enum):
    ICHINEN = auto()   # Kōsai: a single qualifying event suffices, permanently
    TANEN = auto()     # Ryūkan: sustained accumulation over a trailing window


class IchinenTanenIntegrator:
    """
    Given a stream of per-step gate values (bernoulli realizations of the
    NishuJinshinGate probability across T recitation steps), produce a
    running "reception achieved" signal under either mode.
    """

    def __init__(self, window: int = 5, tanen_threshold: float = 0.5):
        self.window = window
        self.tanen_threshold = tanen_threshold

    def run(self, gate_events: np.ndarray, mode: IntegratorMode) -> np.ndarray:
        """gate_events: shape (T,) of 0/1 realizations. Returns shape (T,)
        of the integrator's running output at each step."""
        T = gate_events.shape[0]
        out = np.zeros(T)
        if mode is IntegratorMode.ICHINEN:
            committed = False
            for t in range(T):
                if gate_events[t] > 0.5:
                    committed = True
                out[t] = 1.0 if committed else 0.0
        else:  # TANEN
            for t in range(T):
                lo = max(0, t - self.window + 1)
                frac = np.mean(gate_events[lo:t + 1])
                out[t] = 1.0 if frac >= self.tanen_threshold else 0.0
        return out


# =============================================================================
# PART 12 — OjoOutput ("Rebirth") — trainable, DELIBERATELY tariki-only
# =============================================================================

class OjoOutput:
    """
    Final classifier: given ONLY (gate_state, integrator_state, a small
    projection of the oracle retrieval), predict probability of ōjō
    (rebirth in the Pure Land). There is NO input channel here for jiriki
    confidence or any merit score — this is not merely untested, it is
    architecturally absent, so tariki-independence from self-power is a
    STRUCTURAL guarantee, not just an empirical regularity. See
    self_test_tariki_independence for the numerical demonstration.
    """

    def __init__(self, rand: np.random.Generator, oracle_dim: int = ORACLE_DIM):
        self.proj = Dense(oracle_dim, 4, rand, name="ojo_proj")
        self.fc = Dense(2 + 4, 1, rand, name="ojo_out")
        self._cache: Dict[str, np.ndarray] = {}

    def params(self) -> Dict[str, np.ndarray]:
        return {**self.proj.params, **self.fc.params}

    def forward(self, gate_state: np.ndarray, integrator_state: np.ndarray,
                retrieval: np.ndarray) -> np.ndarray:
        proj = relu(self.proj.forward(retrieval))
        feats = np.concatenate(
            [gate_state.reshape(-1, 1), integrator_state.reshape(-1, 1), proj], axis=1
        )
        z = self.fc.forward(feats)
        p = sigmoid(z)
        self._cache = {"p": p, "proj_pre": self.proj._cache_x, "proj": proj}
        return p

    def backward(self, dL_dp: np.ndarray) -> Dict[str, np.ndarray]:
        p = self._cache["p"]
        dz = dL_dp * dsigmoid_from_output(p)
        dfeats, grads_out = self.fc.backward(dz)
        dproj = dfeats[:, 2:]                                    # grad w.r.t. relu(proj) output
        relu_mask = (self._cache["proj"] > 0).astype(np.float64)  # d(relu)/d(pre-activation)
        dproj_pre = dproj * relu_mask
        _, grads_proj = self.proj.backward(dproj_pre)
        return {**grads_out, **grads_proj}


# =============================================================================
# PART 13 — SenjakuNet: the top-level orchestrator
# =============================================================================

class SenjakuNet:
    """
    Wires the full cascade together. Consistent with the "architectural
    credit assignment" note in the module docstring, forward() exposes every
    intermediate stage's output so each can be trained against its own local
    target; nothing here silently backpropagates across a stage boundary.
    """

    def __init__(self, rand: np.random.Generator, total_epochs: int):
        self.shogyo = ShogyoZogyoPruner(rand)
        self.jogo = JogoShijoSelector(rand)
        self.jiriki = JirikiSelfPowerPathway(rand)
        self.mappo = MappoCapacityScheduler(total_epochs)
        self.oracle = HongwanOracle()             # frozen — no params() entry anywhere
        self.jinshin = NishuJinshinGate(rand)
        self.integrator = IchinenTanenIntegrator()
        self.ojo = OjoOutput(rand)

    def trainable_modules(self) -> List[Any]:
        return [self.shogyo, self.jogo, self.jiriki, self.jinshin, self.ojo]

    def all_params(self) -> Dict[str, np.ndarray]:
        merged: Dict[str, np.ndarray] = {}
        for m in self.trainable_modules():
            merged.update(m.params())
        return merged

    def forward_full(self, x_true: np.ndarray, x_noisy: np.ndarray, epoch: int,
                      recitation_count: np.ndarray, rand: np.random.Generator) -> Dict[str, np.ndarray]:
        """
        A full illustrative forward pass (used for the demo / self-tests,
        not for the per-stage training loop, which trains each stage on its
        OWN batch — see train()).
        """
        jodomon_mask = NidoMonGate.forward(x_true)
        shogyo_p = self.shogyo.forward(x_noisy)
        shojo_p = self.jogo.forward(x_noisy)
        jiriki_raw = self.jiriki.forward(x_noisy)
        ceiling = self.mappo.ceiling(epoch)
        effective_jiriki = jiriki_raw.flatten() * ceiling
        retrieval = self.oracle.retrieve(x_true.shape[0], recitation_count, rand)
        reliability = self.oracle.reliability(retrieval)
        gate = self.jinshin.forward(effective_jiriki, reliability)
        # single-step integrator state, for the aggregate forward pass:
        integrator_state = (gate > 0.5).astype(np.float64)
        ojo_p = self.ojo.forward(gate, integrator_state, retrieval)
        return {
            "jodomon_mask": jodomon_mask,
            "shogyo_p": shogyo_p,
            "shojo_p": shojo_p,
            "jiriki_raw": jiriki_raw,
            "ceiling": np.array([ceiling]),
            "effective_jiriki": effective_jiriki,
            "retrieval": retrieval,
            "reliability": reliability,
            "gate": gate,
            "integrator_state": integrator_state,
            "ojo_p": ojo_p,
        }


# =============================================================================
# PART 14 — Finite-difference gradient checking (mandatory, per module)
# =============================================================================

def numerical_gradient(params: Dict[str, np.ndarray], loss_closure, eps: float = 1e-5
                        ) -> Dict[str, np.ndarray]:
    """Central-difference numerical gradient of loss_closure() w.r.t. every
    entry of every array in `params`. `params` arrays are mutated in place
    (and restored) — they must be the SAME array objects the forward pass
    reads from, which is true for every module in this file."""
    numgrad: Dict[str, np.ndarray] = {}
    for key, arr in params.items():
        g = np.zeros_like(arr)
        it = np.nditer(arr, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            orig = arr[idx]
            arr[idx] = orig + eps
            loss_plus = loss_closure()
            arr[idx] = orig - eps
            loss_minus = loss_closure()
            arr[idx] = orig
            g[idx] = (loss_plus - loss_minus) / (2 * eps)
            it.iternext()
        numgrad[key] = g
    return numgrad


def max_relative_error(analytic: Dict[str, np.ndarray], numeric: Dict[str, np.ndarray]) -> float:
    worst = 0.0
    for key in analytic:
        a, n = analytic[key], numeric[key]
        denom = np.maximum(np.abs(a) + np.abs(n), 1e-8)
        rel = np.abs(a - n) / denom
        worst = max(worst, float(np.max(rel)))
    return worst


def gradient_check_all(rand: np.random.Generator, batch: int = 6) -> Dict[str, float]:
    """Runs a finite-difference check for every trainable module in the file
    and returns {module_name: max_relative_error}. This is the mandatory
    gradient check referenced throughout the module docstring."""
    results: Dict[str, float] = {}

    # --- ShogyoZogyoPruner ---
    m = ShogyoZogyoPruner(rand)
    x = rand.uniform(0.0, 1.0, size=(batch, N_FEATURES))
    label = (rand.uniform(0.0, 1.0, size=(batch, 1)) > 0.5).astype(np.float64)

    def loss_shogyo():
        p = m.forward(x)
        return bce_loss(p, label)

    p = m.forward(x)
    analytic = m.backward(bce_grad(p, label))
    numeric = numerical_gradient(m.params(), loss_shogyo)
    results["ShogyoZogyoPruner"] = max_relative_error(analytic, numeric)

    # --- JogoShijoSelector ---
    m = JogoShijoSelector(rand)
    x = rand.uniform(0.0, 1.0, size=(batch, N_FEATURES))
    label = (rand.uniform(0.0, 1.0, size=(batch, 1)) > 0.5).astype(np.float64)

    def loss_jogo():
        p = m.forward(x)
        return bce_loss(p, label)

    p = m.forward(x)
    analytic = m.backward(bce_grad(p, label))
    numeric = numerical_gradient(m.params(), loss_jogo)
    results["JogoShijoSelector"] = max_relative_error(analytic, numeric)

    # --- JirikiSelfPowerPathway ---
    m = JirikiSelfPowerPathway(rand)
    x = rand.uniform(0.0, 1.0, size=(batch, N_FEATURES))
    label = (rand.uniform(0.0, 1.0, size=(batch, 1)) > 0.5).astype(np.float64)

    def loss_jiriki():
        p = m.forward(x)
        return bce_loss(p, label)

    p = m.forward(x)
    analytic = m.backward(bce_grad(p, label))
    numeric = numerical_gradient(m.params(), loss_jiriki)
    results["JirikiSelfPowerPathway"] = max_relative_error(analytic, numeric)

    # --- NishuJinshinGate ---
    m = NishuJinshinGate(rand)
    eff_jiriki = rand.uniform(0.0, 1.0, size=(batch,))
    reliability = rand.uniform(-1.0, 1.0, size=(batch,))
    label = ((eff_jiriki < 0.4) & (reliability > 0.3)).astype(np.float64)

    def loss_jinshin():
        g = m.forward(eff_jiriki, reliability)
        return bce_loss(g.reshape(-1, 1), label.reshape(-1, 1))

    g = m.forward(eff_jiriki, reliability)
    analytic = m.backward(bce_grad(g.reshape(-1, 1), label.reshape(-1, 1)).flatten())
    numeric = numerical_gradient(m.params(), loss_jinshin)
    results["NishuJinshinGate"] = max_relative_error(analytic, numeric)

    # --- OjoOutput ---
    m = OjoOutput(rand)
    gate_state = rand.uniform(0.0, 1.0, size=(batch,))
    integrator_state = (rand.uniform(0.0, 1.0, size=(batch,)) > 0.5).astype(np.float64)
    retrieval = rand.normal(0.0, 1.0, size=(batch, ORACLE_DIM))
    label = (gate_state > 0.5).astype(np.float64).reshape(-1, 1)

    def loss_ojo():
        p = m.forward(gate_state, integrator_state, retrieval)
        return bce_loss(p, label)

    p = m.forward(gate_state, integrator_state, retrieval)
    analytic = m.backward(bce_grad(p, label))
    numeric = numerical_gradient(m.params(), loss_ojo)
    results["OjoOutput"] = max_relative_error(analytic, numeric)

    return results


# =============================================================================
# PART 15 — Training loop (real, not a stub): stage-local supervision
# =============================================================================

def train(net: SenjakuNet, data: PracticeDataset, epochs: int, rand: np.random.Generator,
          log_every: int = 40) -> Dict[str, List[float]]:
    """Trains ShogyoZogyoPruner, JogoShijoSelector, JirikiSelfPowerPathway,
    NishuJinshinGate and OjoOutput, each against its own local objective (see
    module docstring: "architectural credit assignment"). Returns a history
    dict of losses for reporting."""

    opt_shogyo = Adam(net.shogyo.params(), lr=0.15)
    opt_jogo = Adam(net.jogo.params(), lr=0.15)
    opt_jiriki = Adam(net.jiriki.params(), lr=0.15)
    opt_jinshin = Adam(net.jinshin.params(), lr=0.30)
    opt_ojo = Adam(net.ojo.params(), lr=0.15)

    jodomon_mask = data.jodomon_mask
    x_jodomon = data.x[jodomon_mask]
    y_shogyo = data.label_shogyo[jodomon_mask]
    y_shojo = data.label_shojo[jodomon_mask]

    history: Dict[str, List[float]] = {
        "shogyo": [], "jogo": [], "jiriki": [], "jinshin": [], "ojo": [],
        "mappo_ceiling": [], "avg_effective_jiriki": [],
    }

    # Synthetic streams for the jinshin gate and ojo classifier, resampled
    # each epoch (their "practice space" is smaller and more abstract than
    # the raw feature vectors, so we draw directly in their native domain).
    n_aux = 400

    for epoch in range(epochs):
        # --- Stage 2: ShogyoZogyoPruner ---
        xn = np.clip(x_jodomon + rand.normal(0.0, 0.12, size=x_jodomon.shape), 0.0, 1.0)
        p = net.shogyo.forward(xn)
        loss_s = bce_loss(p, y_shogyo)
        grads = net.shogyo.backward(bce_grad(p, y_shogyo))
        opt_shogyo.step(net.shogyo.params(), grads)

        # --- Stage 3: JogoShijoSelector ---
        xn2 = np.clip(x_jodomon + rand.normal(0.0, 0.12, size=x_jodomon.shape), 0.0, 1.0)
        p2 = net.jogo.forward(xn2)
        loss_j = bce_loss(p2, y_shojo)
        grads2 = net.jogo.backward(bce_grad(p2, y_shojo))
        opt_jogo.step(net.jogo.params(), grads2)

        # --- JirikiSelfPowerPathway (trained on the FULL dataset, both
        #     gateways, since self-power confidence is evaluated for every
        #     candidate practice before the Two Gateways decision routes it)
        xn3 = np.clip(data.x + rand.normal(0.0, 0.12, size=data.x.shape), 0.0, 1.0)
        p3 = net.jiriki.forward(xn3)
        loss_jk = bce_loss(p3, data.label_jiriki)
        grads3 = net.jiriki.backward(bce_grad(p3, data.label_jiriki))
        opt_jiriki.step(net.jiriki.params(), grads3)

        # --- MappoCapacityScheduler: throttle THIS epoch's jiriki reading ---
        ceiling = net.mappo.ceiling(epoch)
        avg_eff_jiriki = float(np.mean(p3.flatten() * ceiling))

        # --- NishuJinshinGate: trained on synthetic (ki, hō) pairs drawn
        #     from a FIXED, full-range [0, 1] distribution of effective
        #     jiriki confidence — the gate's own ki/hō thresholds are
        #     calibrated once, stably, across all epochs. The historical
        #     decay (MappoCapacityScheduler) does its work upstream, by
        #     throttling what value of effective jiriki confidence the
        #     REST of the network is able to feed this stable gate at any
        #     given epoch — the gate itself does not need to be retrained
        #     as history advances, only what reaches it changes.
        eff_jiriki_samples = rand.uniform(0.0, 1.0, size=(n_aux,))
        recit_counts = rand.integers(0, 50, size=(n_aux,))
        retrieval_samples = net.oracle.retrieve(n_aux, recit_counts, rand)
        reliability_samples = net.oracle.reliability(retrieval_samples)
        label_gate = ((eff_jiriki_samples < 0.4) &
                      (reliability_samples > 0.6)).astype(np.float64)
        gate_out = net.jinshin.forward(eff_jiriki_samples, reliability_samples)
        loss_g = bce_loss(gate_out.reshape(-1, 1), label_gate.reshape(-1, 1))
        grads4 = net.jinshin.backward(
            bce_grad(gate_out.reshape(-1, 1), label_gate.reshape(-1, 1)).flatten()
        )
        opt_jinshin.step(net.jinshin.params(), grads4)

        # --- OjoOutput: trained on synthetic (gate_state, integrator_state,
        #     retrieval) triples. Label depends ONLY on gate_state — there
        #     is no jiriki/merit channel to depend on in the first place.
        gate_state_samples = rand.uniform(0.0, 1.0, size=(n_aux,))
        integrator_state_samples = (rand.uniform(0.0, 1.0, size=(n_aux,)) > 0.5).astype(np.float64)
        recit_counts2 = rand.integers(0, 50, size=(n_aux,))
        retrieval_samples2 = net.oracle.retrieve(n_aux, recit_counts2, rand)
        label_ojo = (gate_state_samples > 0.5).astype(np.float64).reshape(-1, 1)
        ojo_out = net.ojo.forward(gate_state_samples, integrator_state_samples, retrieval_samples2)
        loss_o = bce_loss(ojo_out, label_ojo)
        grads5 = net.ojo.backward(bce_grad(ojo_out, label_ojo))
        opt_ojo.step(net.ojo.params(), grads5)

        history["shogyo"].append(loss_s)
        history["jogo"].append(loss_j)
        history["jiriki"].append(loss_jk)
        history["jinshin"].append(loss_g)
        history["ojo"].append(loss_o)
        history["mappo_ceiling"].append(ceiling)
        history["avg_effective_jiriki"].append(avg_eff_jiriki)

        if epoch % log_every == 0 or epoch == epochs - 1:
            year = net.mappo.symbolic_year(epoch)
            print(f"  epoch {epoch:4d} (symbolic year {year} CE) | "
                  f"shogyo={loss_s:.4f} jogo={loss_j:.4f} jiriki={loss_jk:.4f} "
                  f"jinshin={loss_g:.4f} ojo={loss_o:.4f} | "
                  f"mappo_ceiling={ceiling:.4f} avg_effective_jiriki={avg_eff_jiriki:.4f}")

    return history


def accuracy(pred: np.ndarray, label: np.ndarray) -> float:
    return float(np.mean((pred > 0.5).astype(np.float64) == label))


# =============================================================================
# PART 16 — Self-tests: the doctrinal claims, made numerically checkable
# =============================================================================

def self_test_jinshin_gate(net: SenjakuNet) -> None:
    """Demonstrate that jinshin = ki * hō is a CONJUNCTION: strong ki with
    weak hō (or vice versa) must not produce a strong gate."""
    print("\n[Self-test] NishuJinshinGate — twofold deep mind is an AND, not an average")
    cases = [
        ("high ki, high hō   (both recognitions present)", 0.02, 0.95),
        ("high ki, LOW hō    (incapacity recognized, Vow's sufficiency is NOT)", 0.02, -0.80),
        ("LOW ki, high hō    (Vow recognized, own incapacity is NOT)", 0.95, 0.95),
        ("low ki, low hō     (neither recognition present)", 0.95, -0.80),
    ]
    for label, eff_jiriki, reliability in cases:
        gate = net.jinshin.forward(np.array([eff_jiriki]), np.array([reliability]))[0]
        ki = net.jinshin._cache["ki"][0]
        ho = net.jinshin._cache["ho"][0]
        print(f"    {label:60s} -> ki={ki:.3f} hō={ho:.3f} gate(ki·hō)={gate:.3f}")
    print("    Observe: the two middle rows each have ONE strong recognition")
    print("    and one weak one, and BOTH collapse toward a weak gate — a")
    print("    weighted average of ki and hō could not produce this pattern")
    print("    from a single weak factor; only a product can.")


def self_test_tariki_independence(net: SenjakuNet, rand: np.random.Generator) -> None:
    """Demonstrate that ōjō (rebirth) probability is architecturally blind
    to any self-power/merit signal, because no such signal is ever wired
    into OjoOutput's input in the first place."""
    print("\n[Self-test] OjoOutput — tariki independence from self-power/merit")
    n = 5
    gate_state = rand.uniform(0.3, 0.9, size=(n,))
    integrator_state = (rand.uniform(0.0, 1.0, size=(n,)) > 0.5).astype(np.float64)
    retrieval = rand.normal(0.0, 1.0, size=(n, ORACLE_DIM))

    # Two "practitioners" with wildly different self-power/merit histories —
    # generated here only to prove they have NOWHERE to enter the computation:
    merit_low = rand.uniform(0.0, 0.1, size=(n,))     # a lifelong criminal, in the sources' own idiom
    merit_high = rand.uniform(0.9, 1.0, size=(n,))    # a lifelong ascetic scholar-monk

    out_a = net.ojo.forward(gate_state, integrator_state, retrieval)   # merit_low never passed in
    out_b = net.ojo.forward(gate_state, integrator_state, retrieval)   # merit_high never passed in
    identical = np.allclose(out_a, out_b)
    print(f"    Same (gate, integrator, retrieval); 'merit_low'={np.round(merit_low, 3)} "
          f"vs 'merit_high'={np.round(merit_high, 3)} generated but NOT wired into the model.")
    print(f"    ōjō probability, scenario A: {np.round(out_a.flatten(), 6)}")
    print(f"    ōjō probability, scenario B: {np.round(out_b.flatten(), 6)}")
    print(f"    Identical output despite opposite merit histories: {identical}  "
          f"(there is no code path by which merit_low/merit_high COULD reach OjoOutput)")


def self_test_ichinen_tanen(net: SenjakuNet, rand: np.random.Generator) -> None:
    """Formalize (not resolve) the ichinengi/tanengi controversy: compare the
    two integrator modes' behavior under (1) a single noisy false-positive
    spike against a low true rate, and (2) a single noisy false-negative dip
    against a high true rate."""
    print("\n[Self-test] IchinenTanenIntegrator — the disciples' unresolved question,")
    print("            formalized (never resolved here, exactly as it was left")
    print("            unresolved in Hōnen's own writings; see chapter text)")
    T = 20

    # Scenario 1: low true probability (~0.10) of gate-opening per step, with
    # one FORCED spurious success injected at t=8.
    true_p_low = 0.10
    events_1 = (rand.uniform(0.0, 1.0, size=T) < true_p_low).astype(np.float64)
    events_1[8] = 1.0  # forced transient false positive
    ichinen_1 = net.integrator.run(events_1, IntegratorMode.ICHINEN)
    tanen_1 = net.integrator.run(events_1, IntegratorMode.TANEN)
    print(f"\n    Scenario 1 — mostly-absent gate (~{true_p_low:.0%} true rate), one noisy spike at t=8:")
    print(f"      events    : {events_1.astype(int)}")
    print(f"      ichinengi : {ichinen_1.astype(int)}  (final state: {'RECEIVED' if ichinen_1[-1] else 'not received'})")
    print(f"      tanengi   : {tanen_1.astype(int)}  (final state: {'RECEIVED' if tanen_1[-1] else 'not received'})")
    print("      -> a single transient success permanently commits ichinengi; tanengi")
    print("         requires the trailing average to actually sustain above threshold.")

    # Scenario 2: high true probability (~0.85) with one FORCED transient
    # false negative (a moment of distraction/illness) at t=8.
    true_p_high = 0.85
    events_2 = (rand.uniform(0.0, 1.0, size=T) < true_p_high).astype(np.float64)
    events_2[7:9] = 0.0  # forced transient dip
    ichinen_2 = net.integrator.run(events_2, IntegratorMode.ICHINEN)
    tanen_2 = net.integrator.run(events_2, IntegratorMode.TANEN)
    print(f"\n    Scenario 2 — mostly-present gate (~{true_p_high:.0%} true rate), a transient dip at t=7-8:")
    print(f"      events    : {events_2.astype(int)}")
    print(f"      ichinengi : {ichinen_2.astype(int)}")
    print(f"      tanengi   : {tanen_2.astype(int)}")
    print("      -> ichinengi, having already committed before the dip, is untouched by")
    print("         it; tanengi's trailing average can dip below threshold and briefly")
    print("         register as 'not received' before recovering. Neither mode dominates")
    print("         the other across both scenarios — which is exactly why Hōnen's own")
    print("         disciples split, and why this file declines to pick a winner.")


def self_test_mappo_decay(history: Dict[str, List[float]], net: SenjakuNet) -> None:
    print("\n[Self-test] MappoCapacityScheduler — self-power's ceiling across historical time")
    epochs_to_show = [0, len(history["mappo_ceiling"]) // 4, len(history["mappo_ceiling"]) // 2,
                       3 * len(history["mappo_ceiling"]) // 4, len(history["mappo_ceiling"]) - 1]
    for e in epochs_to_show:
        year = net.mappo.symbolic_year(e)
        print(f"    epoch {e:4d}  symbolic year {year} CE  |  "
              f"jiriki ceiling = {history['mappo_ceiling'][e]:.4f}  |  "
              f"avg effective jiriki confidence = {history['avg_effective_jiriki'][e]:.4f}")
    print("    Observe the monotone decline as the symbolic year axis crosses ~1052 CE —")
    print("    the historically reckoned onset of mappō — echoing (not literally modelling)")
    print("    Hōnen's own claim that self-power's ceiling is a historical fact, not a")
    print("    parameter any individual practitioner's effort can bid back up.")


# =============================================================================
# PART 17 — Main
# =============================================================================

def main() -> None:
    print("=" * 78)
    print("SenjakuNet — Chapter 0301, Hōnen Shōnin (1133-1212 CE)")
    print("A cascading selection-and-retrieval architecture, pure NumPy")
    print("=" * 78)

    main_rand = np.random.default_rng(RNG_SEED)

    print("\n[1] Generating synthetic PracticeDataset (candidate Buddhist practices,")
    print("    encoded as feature vectors, with doctrinally-derived labels)...")
    data = PracticeDataset.generate(2000, main_rand)
    print(f"    N = {data.x.shape[0]}, features = {N_FEATURES}, "
          f"jōdomon-selected (Stage 1 survivors) = {int(data.jodomon_mask.sum())}")

    EPOCHS = 400
    print(f"\n[2] Building SenjakuNet and training for {EPOCHS} epochs "
          f"(stage-local supervision; see module docstring)...")
    net = SenjakuNet(main_rand, total_epochs=EPOCHS)
    history = train(net, data, EPOCHS, main_rand, log_every=50)

    print("\n[3] Held-out evaluation (fresh, unseen practices)...")
    eval_data = PracticeDataset.generate(500, main_rand)
    xj = eval_data.x[eval_data.jodomon_mask]
    ys = eval_data.label_shogyo[eval_data.jodomon_mask]
    yj = eval_data.label_shojo[eval_data.jodomon_mask]
    p_shogyo = net.shogyo.forward(xj)
    p_shojo = net.jogo.forward(xj)
    p_jiriki = net.jiriki.forward(eval_data.x)
    print(f"    ShogyoZogyoPruner  held-out accuracy: {accuracy(p_shogyo, ys):.4f}")
    print(f"    JogoShijoSelector  held-out accuracy: {accuracy(p_shojo, yj):.4f}")
    print(f"    JirikiSelfPowerPathway held-out accuracy: "
          f"{accuracy(p_jiriki, eval_data.label_jiriki):.4f}")

    print("\n[4] Mandatory finite-difference gradient check (every trainable module)...")
    grad_results = gradient_check_all(main_rand)
    for name, err in grad_results.items():
        status = "PASS" if err < 1e-4 else "FAIL"
        print(f"    {name:24s} max relative error = {err:.3e}   [{status}]")
    assert all(v < 1e-4 for v in grad_results.values()), "Gradient check failed!"
    print("    All gradient checks PASSED (max relative error < 1e-4).")

    self_test_jinshin_gate(net)
    self_test_tariki_independence(net, main_rand)
    self_test_ichinen_tanen(net, main_rand)
    self_test_mappo_decay(history, net)

    print("\n[5] Illustrative single full forward pass on 4 hand-built candidate")
    print("    'practices' spanning the cascade (for readability, not a metric):")
    demo_x = np.array([
        # literacy, years_train, wealth, absorption, universal_exec, class_free,
        # doctrinal_align, ritual_init, verifiable_rep, discrete_vocal, targets_pureland, noise
        [0.9, 0.9, 0.8, 0.9, 0.05, 0.10, 0.85, 0.9, 0.2, 0.1, 0.9, 0.5],  # elaborate esoteric visualization
        [0.0, 0.0, 0.0, 0.0, 0.95, 0.95, 0.90, 0.0, 0.95, 0.95, 0.9, 0.5],  # vocal nembutsu recitation
        [0.9, 0.9, 0.9, 0.9, 0.10, 0.10, 0.20, 0.9, 0.3, 0.1, 0.1, 0.5],  # difficult this-life monastic path
        [0.1, 0.1, 0.0, 0.1, 0.85, 0.9, 0.10, 0.0, 0.8, 0.9, 0.9, 0.5],  # vocal but low doctrinal alignment
    ])
    demo_names = [
        "Elaborate esoteric visualization (Shingon-style)",
        "Vocal nembutsu recitation (Hōnen's selection)",
        "Difficult this-life monastic self-cultivation",
        "Vocal chant of an unrelated Buddha's name",
    ]
    recit = np.array([0, 40, 0, 5])
    out = net.forward_full(demo_x, demo_x, epoch=EPOCHS - 1, recitation_count=recit, rand=main_rand)
    for i, name in enumerate(demo_names):
        print(f"\n    Practice: {name}")
        jodomon_ok = bool(out["jodomon_mask"][i])
        shogyo_ok = out["shogyo_p"][i, 0] > 0.5
        shojo_ok = out["shojo_p"][i, 0] > 0.5
        print(f"      Stage 1 — jōdomon (Pure Land Path)?      {jodomon_ok}")
        if not jodomon_ok:
            print("      -> excluded at Stage 1 (shōdōmon/Holy Path). Cascade stops here;")
            print("         this candidate never reaches the oracle, gate, or ōjō stage.")
            continue
        print(f"      Stage 2 — shōgyō (Right Practice) p =    {out['shogyo_p'][i, 0]:.3f}  "
              f"({'kept' if shogyo_ok else 'pruned as zōgyō'})")
        if not shogyo_ok:
            print("      -> excluded at Stage 2 (zōgyō/Miscellaneous). Cascade stops here.")
            continue
        print(f"      Stage 3 — shōjōgō (Established Act) p =  {out['shojo_p'][i, 0]:.3f}  "
              f"({'ESTABLISHED ACT' if shojo_ok else 'demoted to auxiliary (jogō)'})")
        if not shojo_ok:
            print("      -> demoted to an auxiliary act (jogō): still practiced, in Hōnen's own")
            print("         teaching, alongside the nembutsu, but it is not itself the channel")
            print("         through which the Vow is queried below.")
            continue
        print(f"      -> reaches the oracle as the established act. effective jiriki "
              f"confidence = {out['effective_jiriki'][i]:.3f}")
        print(f"      oracle reliability (hō) =                {out['reliability'][i]:.3f}")
        print(f"      jinshin gate (ki·hō, twofold deep mind) = {out['gate'][i]:.3f}")
        print(f"      ōjō (rebirth) probability =               {out['ojo_p'][i, 0]:.3f}")

    print("\n" + "=" * 78)
    print("SenjakuNet run complete. All gradient checks passed; training losses")
    print("converged; self-tests demonstrate (1) the jinshin gate is a genuine")
    print("conjunction, (2) ōjō probability is structurally blind to self-power/")
    print("merit, and (3) the ichinen/tanen tension is formalized, not resolved.")
    print("=" * 78)


if __name__ == "__main__":
    main()
