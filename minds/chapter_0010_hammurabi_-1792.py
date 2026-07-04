"""
chapter_0010_hammurabi_-1792.py  --  THE STELE NETWORK
A Casuistic Analogical Engine after Hammurabi of Babylon (r. 1792-1750 BCE)

Part Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
Tome 1 Minds 1 - 20 Available on Amazon https://www.amazon.com/dp/B0H6F9L324
Resume and Interactive Demos at https://artificiology.com/
================================================================================

WHY THIS ARCHITECTURE, AND WHY IT IS *NOT* A RULE-ENGINE
--------------------------------------------------------
The popular picture of the Code of Hammurabi is a deterministic lookup table:
"stated offence -> stated penalty," a giant if/else. Modern Assyriology tells a
different and far more interesting story. The 282 surviving provisions are NOT a
deductive statute book. They are written in exactly the same "if a man does X,
then Y" form as the Babylonian omen lists and medical-diagnostic texts, and the
Code is almost never cited as binding law in the thousands of surviving court
records of the period. As Jean Bottero argued, the Mesopotamians did not
legislate by abstract universal principle; they taught judgment "by formulating
examples from a sufficiently large and typical selection of actual cases -- in
the way that we still teach our children grammar and arithmetic." The Code is a
*curated training set of worked judgments* -- a manual of paradigm cases -- from
which a judge is expected to reason to every NEW case by analogy to the nearest
precedent.

That is the cognitive signature this file encodes:

    The just mind is not the mind that derives verdicts from axioms.
    It is the mind that has internalised a fixed, public canon of exemplary
    cases and decides each novel case by the weighted consensus of the cases
    most analogous to it -- where the whole art lies in knowing WHICH FEATURES
    make two cases alike (the offence? the intent? the social class of the
    parties?), and where every verdict must be traceable back to a named
    exemplar carved on a public stone.

So the model here is deliberately NOT a Transformer, NOT a multilayer perceptron,
NOT attention over learned key/value embeddings. It is an instance-based /
case-based reasoner -- the oldest non-parametric paradigm in machine learning --
given a Hammurabic twist:

  1. THE STELE (a fixed, public case base).  A frozen set of m paradigm cases,
     each a feature vector with a known verdict and a human-readable text. The
     network does NOT learn the cases. They are the canon, carved in diorite,
     immutable and public. (Hammurabi: justice must be written down where
     everyone can see it; the king does not get to change it case by case.)

  2. THE LEARNED ANALOGICAL METRIC.  The ONLY trainable weights are a matrix L
     defining a Mahalanobis metric  M = L^T L  (positive semidefinite by
     construction). The distance from a new case q to a canonical case x_j is
            D_j = (q - x_j)^T M (q - x_j) = || L (q - x_j) ||^2.
     Training learns *which features matter for analogy* -- e.g. that the social
     class of the victim is decisive for an assault but irrelevant for a flood,
     and that the scribe who filed the tablet is pure noise. This is the
     casuist's core skill: weighing the respects in which two cases are similar.

  3. SOFT ANALOGICAL RETRIEVAL.  Verdict logits are a temperature-weighted vote
     of the canon:  w_j = softmax(-D_j / tau),  z_c = sum_j w_j * [verdict_j==c]
     + b_c, then p = softmax(z). Differentiable end to end. (This *looks* like
     attention, but the scores come from a learned distance metric over a FIXED
     public exemplar memory whose "values" are one-hot legal outcomes -- it is
     metric-learned kernel regression, not a sequence model.)

  4. THE LEX-TALIONIS HEAD (symbolic, not learned).  The predicted verdict CLASS
     (talion / compensation / restitution / capital / corporal / acquittal) is
     resolved into a CONCRETE sanction by Hammurabi's class structure: an eye for
     an eye *between equals*, but graded silver compensation when the victim is a
     commoner or a slave. The quantitative reasoning is learned by analogy; the
     *form* of the sanction is fixed by public structure. That division is pure
     Hammurabi.

  5. THE AUDIT TRAIL.  Every verdict reports the canonical cases it leaned on and
     their weights -- "this rests on Law 196 (0.61) and Law 206 (0.27)." The just
     mind is the legible mind; legitimacy is auditability.

  6. THE EXTENSION FLAG.  When the nearest precedent is still far away, the model
     flags the verdict as JUDICIAL EXTENSION, not settled law. Historically the
     code had gaps and the king issued misharum edicts to patch them; here the
     boundary between "covered by the canon" and "going beyond it" is made
     explicit rather than hidden -- the honest answer to the brittleness problem.

ENGINEERING CONVENTION (shared across this whole book of minds)
---------------------------------------------------------------
* Pure NumPy, written from scratch -- no autograd, no ML framework.
* A finite-difference gradient check is MANDATORY and runs every execution.
* A real training loop (mini-batch Adam, implemented here) that measurably
  improves held-out accuracy over the untrained identity-metric baseline.
* Self-tests for retrieval correctness, the symbolic head, and the OOD flag.
* float64 throughout for a clean gradient check.

Run:  python3 chapter_0010_hammurabi_-1792.py
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

RNG_SEED = 1792  # the year Hammurabi took the throne (middle chronology)


# =============================================================================
# 0.  THE WORLD: offences, social classes, and verdict classes
# -----------------------------------------------------------------------------
# These are the symbolic vocabularies of Old Babylonian law. Feature vectors are
# built from them; the generative "ground truth" rule below approximates the
# actual Code (law numbers are cited in comments) so that the metric has
# something real to learn.
# =============================================================================

OFFENCES: List[str] = [
    "assault_eye",          # LH 196/198/199
    "assault_bone",         # LH 197/198/199
    "assault_tooth",        # LH 200/201
    "theft_sacred",         # LH 6   (temple/palace property)
    "theft_ordinary",       # LH 8   (livestock / private goods)
    "builder_collapse",     # LH 229/230/232
    "surgery_fatal",        # LH 218 (operation kills/blinds an awilum)
    "false_witness",        # LH 3/4 (capital vs property case)
    "agent_fraud",          # LH 254-256 (merchant's agent cheats)
    "adultery",             # LH 129
    "brawl_homicide",       # LH 206/207 (accidental blow)
    "canal_negligence",     # LH 53-55 (flooding a neighbour's field)
]
OFF_IDX = {o: i for i, o in enumerate(OFFENCES)}

CLASSES: List[str] = ["awilum", "mushkenum", "wardum"]   # free man / commoner / slave
CLS_IDX = {c: i for i, c in enumerate(CLASSES)}

# Verdict CLASSES -- the output space the network predicts by analogy.
VERDICTS: List[str] = [
    "TALION",        # mirror penalty (eye for eye) -- only between equals
    "COMPENSATION",  # fixed/graded silver payment
    "RESTITUTION",   # multiple of the value stolen/lost
    "CAPITAL",       # death
    "CORPORAL",      # mutilation (e.g. the surgeon's hand)
    "ACQUITTAL",     # no penalty (act of god, cleared by oath, pardon)
]
VRD_IDX = {v: i for i, v in enumerate(VERDICTS)}
N_VERDICTS = len(VERDICTS)


# Extra "distractor" features that a real docket would record but that are
# legally irrelevant. A good metric must learn to IGNORE these. Including them is
# the whole point: it makes metric-learning meaningful and lets us *measure*
# whether the network has discovered which features carry justice.
DISTRACTORS: List[str] = ["region_id", "season", "time_of_day", "scribe_id"]


@dataclass
class Case:
    """A single legal case, both as a human-readable record and a feature row."""
    offence: str
    perp_class: str
    victim_class: str
    intent: float          # 1.0 = premeditated, 0.0 = accidental
    value: float           # normalised property value at stake (0..1)
    evidence: float        # strength of proof / sealed-receipt present (0..1)
    distract: np.ndarray   # legally-irrelevant noise features
    verdict: int = -1      # index into VERDICTS (ground truth / canon label)

    def to_vector(self) -> np.ndarray:
        """Encode the case as a fixed-length float64 feature vector.

        Layout:
          [ one-hot offence (12) | one-hot perp_class (3) | one-hot victim_class (3)
            | intent (1) | value (1) | evidence (1) | distractors (4) ]  = 25 dims
        """
        off = np.zeros(len(OFFENCES));   off[OFF_IDX[self.offence]] = 1.0
        pc  = np.zeros(len(CLASSES));    pc[CLS_IDX[self.perp_class]] = 1.0
        vc  = np.zeros(len(CLASSES));    vc[CLS_IDX[self.victim_class]] = 1.0
        scal = np.array([self.intent, self.value, self.evidence], dtype=np.float64)
        return np.concatenate([off, pc, vc, scal, self.distract]).astype(np.float64)


FEATURE_DIM = len(OFFENCES) + 2 * len(CLASSES) + 3 + len(DISTRACTORS)  # = 25

# Human-readable index ranges, used by the feature-importance report.
FEATURE_NAMES: List[str] = (
    [f"offence:{o}" for o in OFFENCES]
    + [f"perp:{c}" for c in CLASSES]
    + [f"victim:{c}" for c in CLASSES]
    + ["intent", "value", "evidence"]
    + [f"noise:{d}" for d in DISTRACTORS]
)


# =============================================================================
# 1.  GROUND TRUTH: a generative approximation of the Code
# -----------------------------------------------------------------------------
# This function plays the role of "the law as it actually was decided." It is NOT
# used by the network -- it only labels the data, exactly as a corpus of real
# court outcomes would. The network must *recover* this structure from examples
# via the learned metric. The rule is class-, intent-, and value-dependent so the
# metric has to attend to the right features (and ignore the distractors).
# =============================================================================

def true_verdict(c: Case) -> int:
    o, pv, vv = c.offence, c.perp_class, c.victim_class
    if o in ("assault_eye", "assault_bone", "assault_tooth"):
        # LH 196-201: mirror penalty between free equals; graded silver otherwise.
        if vv == "awilum" and pv == "awilum":
            return VRD_IDX["TALION"]
        return VRD_IDX["COMPENSATION"]
    if o == "theft_sacred":
        # LH 6: theft of temple/palace property is capital.
        return VRD_IDX["CAPITAL"]
    if o == "theft_ordinary":
        # LH 8: pay a multiple; if the goods are very valuable and the thief
        # cannot conceivably pay, it escalates to death.
        return VRD_IDX["CAPITAL"] if c.value > 0.8 else VRD_IDX["RESTITUTION"]
    if o == "builder_collapse":
        # LH 229: if the house kills the owner, the builder is put to death;
        # LH 232: if it only destroys goods, he restores them.
        return VRD_IDX["CAPITAL"] if c.intent > 0.5 or c.value > 0.5 else VRD_IDX["RESTITUTION"]
        # (here `intent` overloads "a person was inside / killed")
    if o == "surgery_fatal":
        # LH 218: the surgeon whose operation kills/blinds an awilum loses a hand.
        return VRD_IDX["CORPORAL"] if vv == "awilum" else VRD_IDX["COMPENSATION"]
    if o == "false_witness":
        # LH 3: false testimony in a capital case is itself capital;
        # LH 4: in a property case the false witness pays the claim.
        return VRD_IDX["CAPITAL"] if c.value > 0.5 else VRD_IDX["COMPENSATION"]
    if o == "agent_fraud":
        # LH 254-256: a cheating agent restores a multiple of what he embezzled.
        return VRD_IDX["RESTITUTION"]
    if o == "adultery":
        # LH 129: capital -- unless the wronged husband (or king) pardons, in
        # which case both are spared. Strong evidence -> conviction.
        return VRD_IDX["CAPITAL"] if c.evidence > 0.4 else VRD_IDX["ACQUITTAL"]
    if o == "brawl_homicide":
        # LH 206-207: an accidental blow is compensated (the striker swears it
        # was unintentional and pays the physician / a half-mina).
        return VRD_IDX["CAPITAL"] if c.intent > 0.5 else VRD_IDX["COMPENSATION"]
    if o == "canal_negligence":
        # LH 53-55: negligent flooding is restored; an act of god clears the
        # farmer (low intent + weak "negligence" signal -> acquittal).
        return VRD_IDX["RESTITUTION"] if c.intent > 0.4 else VRD_IDX["ACQUITTAL"]
    raise ValueError(f"unmapped offence {o}")


# =============================================================================
# 2.  THE STELE: build the fixed, public canon of paradigm cases
# -----------------------------------------------------------------------------
# One clean prototype per (offence x relevant-class / relevant-condition). These
# are the "282 laws" in miniature: archetypal, noise-free, publicly fixed. Novel
# cases will be matched against THESE by analogy.
# =============================================================================

def build_stele() -> List[Case]:
    z = np.zeros(len(DISTRACTORS))           # the canon carries no docket noise
    canon: List[Case] = []

    def add(offence, perp, victim, intent, value, evidence):
        c = Case(offence, perp, victim, intent, value, evidence, z.copy())
        c.verdict = true_verdict(c)
        canon.append(c)

    # Assault: one prototype per victim class (the decisive feature).
    for inj in ("assault_eye", "assault_bone", "assault_tooth"):
        add(inj, "awilum", "awilum", 1.0, 0.0, 1.0)     # talion
        add(inj, "awilum", "mushkenum", 1.0, 0.0, 1.0)  # silver
        add(inj, "awilum", "wardum", 1.0, 0.0, 1.0)     # half-value silver
    # Theft
    add("theft_sacred", "awilum", "awilum", 1.0, 0.9, 1.0)        # capital
    add("theft_ordinary", "awilum", "awilum", 1.0, 0.3, 1.0)      # restitution
    add("theft_ordinary", "awilum", "awilum", 1.0, 0.9, 1.0)      # escalates capital
    # Builder
    add("builder_collapse", "awilum", "awilum", 1.0, 0.7, 1.0)    # capital (death inside)
    add("builder_collapse", "awilum", "awilum", 0.0, 0.3, 1.0)    # restitution (goods)
    # Surgery
    add("surgery_fatal", "awilum", "awilum", 0.0, 0.0, 1.0)       # corporal
    add("surgery_fatal", "awilum", "mushkenum", 0.0, 0.0, 1.0)    # compensation
    # False witness
    add("false_witness", "awilum", "awilum", 1.0, 0.8, 1.0)       # capital
    add("false_witness", "awilum", "awilum", 1.0, 0.2, 1.0)       # compensation
    # Agent fraud
    add("agent_fraud", "awilum", "awilum", 1.0, 0.5, 1.0)         # restitution
    # Adultery
    add("adultery", "awilum", "awilum", 1.0, 0.9, 1.0)           # capital
    add("adultery", "awilum", "awilum", 1.0, 0.1, 1.0)           # acquittal (no proof)
    # Brawl
    add("brawl_homicide", "awilum", "awilum", 1.0, 0.0, 1.0)     # capital (intent)
    add("brawl_homicide", "awilum", "awilum", 0.0, 0.0, 1.0)     # compensation
    # Canal
    add("canal_negligence", "awilum", "awilum", 1.0, 0.4, 1.0)  # restitution
    add("canal_negligence", "awilum", "awilum", 0.0, 0.4, 1.0)  # acquittal (act of god)
    return canon


# =============================================================================
# 3.  DATA: sample novel (noisy) cases and label them with the true rule
# =============================================================================

def sample_case(rng: np.random.Generator) -> Case:
    offence = OFFENCES[rng.integers(len(OFFENCES))]
    perp = CLASSES[rng.integers(len(CLASSES))]
    victim = CLASSES[rng.integers(len(CLASSES))]
    intent = float(rng.random())
    value = float(rng.random())
    evidence = float(rng.random())
    distract = rng.normal(0.0, 1.0, size=len(DISTRACTORS))  # pure noise
    c = Case(offence, perp, victim, intent, value, evidence, distract)
    c.verdict = true_verdict(c)
    return c


def make_dataset(n: int, rng: np.random.Generator,
                 label_noise: float = 0.06) -> Tuple[np.ndarray, np.ndarray, List[Case]]:
    """Return (X [n,d], y [n], cases). A little label noise keeps the task honest
    (a perfectly separable problem would not test generalisation)."""
    cases = [sample_case(rng) for _ in range(n)]
    X = np.stack([c.to_vector() for c in cases]).astype(np.float64)
    y = np.array([c.verdict for c in cases], dtype=np.int64)
    if label_noise > 0:
        flip = rng.random(n) < label_noise
        y[flip] = rng.integers(N_VERDICTS, size=int(flip.sum()))
    return X, y, cases


# =============================================================================
# 4.  THE STELE NETWORK
# =============================================================================

@dataclass
class SteleNetwork:
    """Differentiable case-based reasoner with a learned analogical metric.

    Trainable parameters (the only ones):
        L        : (d, r)  factor of the Mahalanobis metric M = L^T L
        log_tau  : scalar  log temperature of the analogical softmax
        log_s    : scalar  log of the vote-confidence scale (how strongly a
                           decisive precedent overrides the prior)
        b        : (C,)    per-verdict prior ("the king's residual correction")

    The stele (canon) features `Sx` (m,d) and one-hot verdicts `Sy` (m,C) are
    FIXED and never receive gradients.
    """
    Sx: np.ndarray                  # (m, d) canon feature matrix  (frozen)
    Sy_onehot: np.ndarray           # (m, C) canon verdicts one-hot (frozen)
    canon: List[Case]               # human-readable canon (for audit trails)
    rank: int = FEATURE_DIM
    seed: int = RNG_SEED
    L: np.ndarray = field(init=False)
    log_tau: float = field(init=False)
    log_s: float = field(init=False)
    b: np.ndarray = field(init=False)
    extension_distance: float = field(init=False, default=np.inf)

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        d = self.Sx.shape[1]
        # Initialise L near the identity so the model starts as plain Euclidean
        # k-NN over the canon, then *learns* the metric from there.
        self.L = np.eye(d, self.rank, dtype=np.float64)
        self.L += 0.01 * rng.standard_normal((d, self.rank))
        self.log_tau = 0.0                                  # tau = 1.0
        self.log_s = np.log(4.0)                            # vote scale s = 4.0
        self.b = np.zeros(self.Sy_onehot.shape[1], dtype=np.float64)

    # ---- core forward ------------------------------------------------------
    def _pairwise_to_canon(self, Q: np.ndarray) -> np.ndarray:
        """Squared Mahalanobis distance D[i,j] = ||L (q_i - x_j)||^2 from every
        query in Q to every paradigm case in the stele. Shared by forward() and
        the extension-threshold calibration."""
        delta = Q[:, None, :] - self.Sx[None, :, :]      # (n, m, d)
        proj = delta @ self.L                            # (n, m, r)
        return np.einsum("ijr,ijr->ij", proj, proj)      # (n, m)

    def forward(self, Q: np.ndarray) -> Tuple[np.ndarray, dict]:
        """Q: (n,d) queries -> p: (n,C) verdict probabilities, plus a cache for
        backprop. Every intermediate is kept so the gradients are exact."""
        n, d = Q.shape
        m = self.Sx.shape[0]
        tau = np.exp(self.log_tau)
        s = np.exp(self.log_s)

        # delta[i,j,:] = Q_i - Sx_j           (n, m, d)
        delta = Q[:, None, :] - self.Sx[None, :, :]
        # proj[i,j,:]  = delta[i,j,:] @ L      (n, m, r)
        proj = delta @ self.L
        # D[i,j] = || L (q_i - x_j) ||^2       (n, m)   <- Mahalanobis distance
        D = np.einsum("ijr,ijr->ij", proj, proj)
        # analogical scores: nearer precedent -> higher weight
        A = -D / tau
        A -= A.max(axis=1, keepdims=True)            # numerical stability
        expA = np.exp(A)
        W = expA / expA.sum(axis=1, keepdims=True)   # (n, m) softmax over canon
        # verdict logits = SCALED weighted vote of canon verdicts + learned prior.
        # The scale s lets a decisive precedent (W concentrated on one case)
        # produce a strong logit that dominates the prior b; a split vote stays
        # near the prior. Without it, a one-hot vote logit is only ~1 and the
        # prior would swamp the evidence.
        V = W @ self.Sy_onehot                       # (n, C) the precedent vote
        Z = s * V + self.b                           # (n, C)
        Z -= Z.max(axis=1, keepdims=True)
        expZ = np.exp(Z)
        P = expZ / expZ.sum(axis=1, keepdims=True)   # (n, C)

        cache = dict(Q=Q, delta=delta, proj=proj, D=D, W=W, V=V, P=P,
                     tau=tau, s=s)
        return P, cache

    # ---- loss + analytic gradients ----------------------------------------
    def loss_and_grads(self, Q: np.ndarray, y: np.ndarray
                       ) -> Tuple[float, Dict[str, np.ndarray]]:
        """Cross-entropy loss and exact gradients w.r.t. {L, log_tau, log_s, b}."""
        n = Q.shape[0]
        P, cache = self.forward(Q)
        W, D, proj, delta, tau, s, V = (
            cache[k] for k in ("W", "D", "proj", "delta", "tau", "s", "V"))

        # --- cross-entropy ---
        eps = 1e-12
        loss = -np.log(P[np.arange(n), y] + eps).mean()

        # dL/dZ  (softmax + CE):  (P - onehot(y)) / n
        Y = np.zeros_like(P); Y[np.arange(n), y] = 1.0
        dZ = (P - Y) / n                              # (n, C)

        # Z = s * V + b
        db = dZ.sum(axis=0)                           # (C,)
        ds = float((dZ * V).sum())                    # scalar
        dlog_s = ds * s                               # s = exp(log_s)
        dV = s * dZ                                   # (n, C)

        # V = W @ Sy_onehot  ->  dW = dV @ Sy_onehot^T
        dW = dV @ self.Sy_onehot.T                    # (n, m)

        # W = softmax_j(A): dA = W * (dW - sum_k dW_k W_k)
        dA = W * (dW - (dW * W).sum(axis=1, keepdims=True))   # (n, m)

        # A = -D / tau
        dD = dA * (-1.0 / tau)                        # (n, m)
        # log_tau: dA/dlog_tau = D / tau  (since tau = exp(log_tau))
        dlog_tau = float((dA * (D / tau)).sum())

        # D = sum_r proj^2  ->  dproj = dD * 2 * proj
        dproj = 2.0 * proj * dD[:, :, None]           # (n, m, r)

        # proj = delta @ L  ->  dL = sum_{i,j} delta_{ij}^T (outer) dproj_{ij}
        d = self.L.shape[0]; r = self.L.shape[1]
        dL = delta.reshape(-1, d).T @ dproj.reshape(-1, r)   # (d, r)

        return loss, {"L": dL, "log_tau": dlog_tau, "log_s": dlog_s, "b": db}

    # ---- inference ---------------------------------------------------------
    def predict(self, Q: np.ndarray) -> np.ndarray:
        P, _ = self.forward(Q)
        return P.argmax(axis=1)

    def accuracy(self, Q: np.ndarray, y: np.ndarray) -> float:
        return float((self.predict(Q) == y).mean())

    def metric_diagonal(self) -> np.ndarray:
        """diag(M) = diag(L^T L): the learned per-feature 'this matters' weight."""
        return np.einsum("dr,dr->d", self.L, self.L)

    # ---- the verdict with audit trail + extension flag --------------------
    def verdict(self, case: Case, k: int = 3, vec_override: np.ndarray = None
                ) -> dict:
        """Decide a single case: predicted class, concrete sanction (lex-talionis
        head), the k canonical precedents relied upon (audit trail), and whether
        the verdict is settled law or a judicial extension beyond the canon.

        The extension flag keys off the ABSOLUTE Mahalanobis distance to the
        nearest precedent (not the softmax weight, which is relative and always
        ~1 for the closest case). If the nearest precedent is farther than the
        canon's own internal scale (self.extension_distance, calibrated on the
        training distribution), the case sits outside the canon."""
        q = (vec_override if vec_override is not None else case.to_vector())[None, :]
        P, cache = self.forward(q)
        W = cache["W"][0]
        D = cache["D"][0]
        pred = int(P[0].argmax())
        order = np.argsort(-W)[:k]
        precedents = [
            {
                "weight": float(W[j]),
                "verdict": VERDICTS[self.canon[j].verdict],
                "text": describe_case(self.canon[j]),
            }
            for j in order
        ]
        nearest_distance = float(D.min())
        extension = nearest_distance > self.extension_distance
        sanction = lex_talionis_resolution(
            VERDICTS[pred], case.perp_class, case.victim_class, case.value
        )
        return {
            "predicted_class": VERDICTS[pred],
            "confidence": float(P[0, pred]),
            "sanction": sanction,
            "audit_trail": precedents,
            "nearest_distance": nearest_distance,
            "settled_law": not extension,
            "note": ("settled: rests squarely on a public precedent"
                     if not extension else
                     "JUDICIAL EXTENSION: no canonical case is close; the king "
                     "must rule and, by Hammurabi's logic, carve the new case "
                     "into the stele so the next judge inherits it"),
        }

    # ---- training loop -----------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray,
            Xval: np.ndarray, yval: np.ndarray,
            epochs: int = 60, batch: int = 64, lr: float = 0.05,
            verbose: bool = True) -> dict:
        """Mini-batch Adam over the metric parameters."""
        rng = np.random.default_rng(self.seed + 1)
        n = X.shape[0]
        # Adam moments for each parameter
        scalars = ("log_tau", "log_s")
        mom = {p: (0.0 if p in scalars else np.zeros_like(getattr(self, p)))
               for p in ("L", "log_tau", "log_s", "b")}
        vel = {p: (0.0 if p in scalars else np.zeros_like(getattr(self, p)))
               for p in ("L", "log_tau", "log_s", "b")}
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        t = 0
        history = {"loss": [], "val_acc": []}
        for ep in range(1, epochs + 1):
            perm = rng.permutation(n)
            ep_loss = 0.0
            for s in range(0, n, batch):
                idx = perm[s:s + batch]
                loss, grads = self.loss_and_grads(X[idx], y[idx])
                ep_loss += loss * len(idx)
                t += 1
                for p in ("L", "log_tau", "log_s", "b"):
                    g = grads[p]
                    mom[p] = beta1 * mom[p] + (1 - beta1) * g
                    vel[p] = beta2 * vel[p] + (1 - beta2) * (g * g)
                    mhat = mom[p] / (1 - beta1 ** t)
                    vhat = vel[p] / (1 - beta2 ** t)
                    upd = lr * mhat / (np.sqrt(vhat) + eps)
                    setattr(self, p, getattr(self, p) - upd)
            ep_loss /= n
            va = self.accuracy(Xval, yval)
            history["loss"].append(ep_loss)
            history["val_acc"].append(va)
            if verbose and (ep == 1 or ep % 10 == 0 or ep == epochs):
                print(f"   epoch {ep:3d}  loss {ep_loss:.4f}  val_acc {va:.3f}  "
                      f"tau {np.exp(self.log_tau):.3f}")
        # ---- calibrate the "beyond the canon" threshold -------------------
        # After the metric is learned, measure how far each TRAINING query sits
        # from its nearest paradigm case under the learned metric M. A genuinely
        # novel case (one the stele never anticipated) will sit further out than
        # almost anything seen in training. We take a high percentile as the
        # extension boundary: cross it and the engine refuses to pretend the
        # canon already covers the matter.
        Dtr = self._pairwise_to_canon(X)          # (n, n_canon)
        nearest = Dtr.min(axis=1)                  # distance to closest precedent
        self.extension_distance = float(np.percentile(nearest, 99.5))
        if verbose:
            print(f"   calibrated extension_distance "
                  f"(99.5th pctile of nearest) = {self.extension_distance:.3f}")
        return history


# =============================================================================
# 5.  THE LEX-TALIONIS HEAD  (symbolic resolution of a verdict CLASS -> sanction)
# -----------------------------------------------------------------------------
# This is the famous, and famously misunderstood, part. "An eye for an eye" is
# NOT universal vengeance; in the Code it applies *between social equals*. When
# the victim is a commoner or a slave the mirror penalty is replaced by graded
# silver. The class structure is fixed and public -- it is not learned -- which
# is exactly why Hammurabi could carve it in stone.
# =============================================================================

def lex_talionis_resolution(verdict_class: str, perp_class: str,
                            victim_class: str, value: float) -> str:
    if verdict_class == "TALION":
        if perp_class == "awilum" and victim_class == "awilum":
            return "mirror penalty in kind (eye for eye, bone for bone) -- LH 196/197/200"
        # talion is only ever ordered between equals; otherwise it degrades
        return ("graded silver compensation in lieu of mirror penalty "
                "(victim is not a social equal)")
    if verdict_class == "COMPENSATION":
        if victim_class == "mushkenum":
            return "fixed silver compensation, 1 mina (commoner victim) -- LH 198"
        if victim_class == "wardum":
            return "silver equal to half the slave's market value -- LH 199"
        return "silver compensation, ~1/2 mina, on sworn account -- LH 207"
    if verdict_class == "RESTITUTION":
        mult = 30 if value > 0.6 else 10
        return f"restore {mult}x the value at stake -- LH 8 / 256"
    if verdict_class == "CAPITAL":
        return "death (the offence strikes at the order the gods entrusted to the king)"
    if verdict_class == "CORPORAL":
        return "amputation of the offending instrument (e.g. the surgeon's hand) -- LH 218"
    if verdict_class == "ACQUITTAL":
        return "no penalty: cleared by oath, pardon, or recognised act of god"
    return "unresolved"


def describe_case(c: Case) -> str:
    bits = [c.offence.replace("_", " ")]
    if c.offence.startswith("assault"):
        bits.append(f"victim={c.victim_class}")
    if c.offence in ("theft_ordinary", "theft_sacred", "agent_fraud"):
        bits.append(f"value={'high' if c.value > 0.6 else 'modest'}")
    if c.offence in ("builder_collapse", "brawl_homicide", "canal_negligence"):
        bits.append("a death/serious harm" if c.intent > 0.5 else "property/accident")
    if c.offence in ("adultery", "false_witness"):
        bits.append("proven" if max(c.value, c.evidence) > 0.5 else "unproven")
    return " | ".join(bits) + f"  ->  {VERDICTS[c.verdict]}"


# =============================================================================
# 6.  GRADIENT CHECK  (mandatory: analytic vs central finite differences)
# =============================================================================

def finite_difference_check(verbose: bool = True) -> float:
    """Build a tiny instance and verify every parameter's analytic gradient
    against a central finite difference. Returns the max relative error."""
    rng = np.random.default_rng(7)
    d, m, C, n = 6, 5, 3, 4
    Sx = rng.standard_normal((m, d))
    Sy = np.zeros((m, C)); Sy[np.arange(m), rng.integers(C, size=m)] = 1.0
    net = SteleNetwork(Sx=Sx, Sy_onehot=Sy, canon=[], rank=d, seed=3)
    net.L = rng.standard_normal((d, d))           # arbitrary, away from identity
    net.log_tau = float(rng.standard_normal() * 0.3)
    net.log_s = float(rng.standard_normal() * 0.3 + 1.0)   # away from the default
    net.b = rng.standard_normal(C) * 0.5
    Q = rng.standard_normal((n, d))
    y = rng.integers(C, size=n)

    _, grads = net.loss_and_grads(Q, y)

    def loss_only() -> float:
        return net.loss_and_grads(Q, y)[0]

    eps = 1e-6
    max_rel = 0.0
    report = []

    # check a random subset of L entries
    L0 = net.L.copy()
    for _ in range(12):
        i = rng.integers(d); j = rng.integers(d)
        net.L = L0.copy(); net.L[i, j] += eps; lp = loss_only()
        net.L = L0.copy(); net.L[i, j] -= eps; lm = loss_only()
        net.L = L0.copy()
        num = (lp - lm) / (2 * eps)
        ana = grads["L"][i, j]
        rel = abs(num - ana) / (abs(num) + abs(ana) + 1e-12)
        max_rel = max(max_rel, rel)
        report.append((f"L[{i},{j}]", ana, num, rel))

    # check log_tau
    t0 = net.log_tau
    net.log_tau = t0 + eps; lp = loss_only()
    net.log_tau = t0 - eps; lm = loss_only()
    net.log_tau = t0
    num = (lp - lm) / (2 * eps); ana = grads["log_tau"]
    rel = abs(num - ana) / (abs(num) + abs(ana) + 1e-12)
    max_rel = max(max_rel, rel)
    report.append(("log_tau", ana, num, rel))

    # check log_s
    s0 = net.log_s
    net.log_s = s0 + eps; lp = loss_only()
    net.log_s = s0 - eps; lm = loss_only()
    net.log_s = s0
    num = (lp - lm) / (2 * eps); ana = grads["log_s"]
    rel = abs(num - ana) / (abs(num) + abs(ana) + 1e-12)
    max_rel = max(max_rel, rel)
    report.append(("log_s", ana, num, rel))

    # check every entry of b
    b0 = net.b.copy()
    for c in range(C):
        net.b = b0.copy(); net.b[c] += eps; lp = loss_only()
        net.b = b0.copy(); net.b[c] -= eps; lm = loss_only()
        net.b = b0.copy()
        num = (lp - lm) / (2 * eps); ana = grads["b"][c]
        rel = abs(num - ana) / (abs(num) + abs(ana) + 1e-12)
        max_rel = max(max_rel, rel)
        report.append((f"b[{c}]", ana, num, rel))

    if verbose:
        print("   param        analytic        numeric         rel.err")
        shown = [r for r in report if not r[0].startswith("L[")][:8]
        for name, ana, num, rel in report[:4] + shown:
            print(f"   {name:10s}  {ana: .6e}  {num: .6e}  {rel: .2e}")
        print(f"   ...({len(report)} params checked)")
        print(f"   MAX RELATIVE ERROR = {max_rel:.3e}")
    return max_rel


# =============================================================================
# 7.  MAIN: gradient check -> train -> evaluate -> demonstrate the mind
# =============================================================================

def main() -> None:
    np.set_printoptions(precision=3, suppress=True)
    print("=" * 78)
    print("THE STELE NETWORK  --  a casuistic analogical engine after Hammurabi")
    print("=" * 78)

    # --- (a) MANDATORY gradient check ---
    print("\n[1] FINITE-DIFFERENCE GRADIENT CHECK")
    max_rel = finite_difference_check()
    assert max_rel < 1e-5, f"gradient check FAILED (max rel err {max_rel:.2e})"
    print("   PASS: analytic gradients match finite differences.")

    # --- (b) data + canon ---
    rng = np.random.default_rng(RNG_SEED)
    canon = build_stele()
    Sx = np.stack([c.to_vector() for c in canon]).astype(np.float64)
    Sy = np.zeros((len(canon), N_VERDICTS))
    Sy[np.arange(len(canon)), [c.verdict for c in canon]] = 1.0
    print(f"\n[2] THE STELE: {len(canon)} fixed public precedents over "
          f"{FEATURE_DIM} features, {N_VERDICTS} verdict classes.")

    Xtr, ytr, _ = make_dataset(2400, rng)
    Xva, yva, _ = make_dataset(600, rng)
    Xte, yte, te_cases = make_dataset(800, rng)

    # --- (c) baseline: untrained, identity metric (plain Euclidean k-NN vote) ---
    base = SteleNetwork(Sx=Sx, Sy_onehot=Sy, canon=canon, rank=FEATURE_DIM)
    base.L = np.eye(FEATURE_DIM)              # exact identity: no learned weighting
    base_acc = base.accuracy(Xte, yte)
    print(f"\n[3] BASELINE (identity metric, no training): "
          f"test accuracy = {base_acc:.3f}")
    print("    -> Euclidean analogy is confused by the legally-irrelevant "
          "distractor\n       features; it cannot tell which respects matter.")

    # --- (d) train the analogical metric ---
    print(f"\n[4] TRAINING the analogical metric (Adam):")
    net = SteleNetwork(Sx=Sx, Sy_onehot=Sy, canon=canon, rank=FEATURE_DIM)
    net.fit(Xtr, ytr, Xva, yva, epochs=60, batch=64, lr=0.05)
    test_acc = net.accuracy(Xte, yte)
    print(f"\n[5] TRAINED test accuracy = {test_acc:.3f}   "
          f"(baseline was {base_acc:.3f})")
    assert test_acc > base_acc + 0.10, "training did not improve over baseline"
    assert test_acc > 0.80, "trained accuracy unexpectedly low"

    # --- (e) interpretability: did it learn WHICH features carry justice? ---
    diag = net.metric_diagonal()
    order = np.argsort(-diag)
    print("\n[6] LEARNED FEATURE IMPORTANCE  diag(M) = diag(L^T L)")
    print("    top features the just mind learned to weigh:")
    for i in order[:6]:
        print(f"      {FEATURE_NAMES[i]:18s}  {diag[i]:.3f}")
    print("    legally-irrelevant 'noise' features it learned to ignore:")
    noise_ix = [i for i, nm in enumerate(FEATURE_NAMES) if nm.startswith("noise:")]
    for i in noise_ix:
        print(f"      {FEATURE_NAMES[i]:18s}  {diag[i]:.3f}")
    noise_mean = float(diag[noise_ix].mean())
    signal_mean = float(np.delete(diag, noise_ix).mean())
    print(f"    mean weight  signal={signal_mean:.3f}  vs  noise={noise_mean:.3f}")
    assert signal_mean > noise_mean, "metric failed to down-weight the distractors"

    # --- (f) the mind at work: verdicts with audit trails ---
    print("\n[7] THE MIND AT WORK  (verdict + audit trail + extension flag)")
    z = np.zeros(len(DISTRACTORS))
    demo_cases = [
        Case("assault_eye", "awilum", "awilum", 1.0, 0.0, 1.0, z.copy()),    # talion
        Case("assault_eye", "awilum", "wardum", 1.0, 0.0, 1.0, z.copy()),    # silver
        Case("theft_sacred", "awilum", "awilum", 1.0, 0.9, 1.0, z.copy()),   # capital
        Case("brawl_homicide", "awilum", "awilum", 0.0, 0.0, 1.0, z.copy()), # accident
    ]
    for c in demo_cases:
        v = net.verdict(c, k=2)
        print(f"\n   CASE: {describe_case(c).split('  ->')[0]}")
        print(f"     verdict : {v['predicted_class']}  (conf {v['confidence']:.2f})")
        print(f"     sanction: {v['sanction']}")
        print(f"     rests on: " + "; ".join(
            f"{p['verdict']} [{p['weight']:.2f}] ({p['text'].split('  ->')[0]})"
            for p in v['audit_trail']))
        print(f"     status  : {v['note']}")

    # --- (g) the boundary of settled law: an out-of-canon case ---
    print("\n[8] BEYOND THE CANON  (a case the stele never anticipated)")
    # The harm TYPE is familiar -- ordinary theft -- but the MAGNITUDE is not:
    # a sum many times larger than any stake the stele ever priced. A casuistic
    # system can only reason by proportion to exemplars it has actually seen;
    # confronted with a quantity orders beyond its largest precedent, no analogy
    # is close enough to be just. The engine declines to pretend otherwise and
    # refers the matter to the king -- who, by Hammurabi's own logic, must rule
    # and carve the new exemplar into the stele for the next judge to inherit.
    novel = Case("theft_ordinary", "awilum", "awilum",
                 intent=1.0, value=8.0, evidence=1.0, distract=z.copy())
    v = net.verdict(novel, k=2)
    print(f"   case: ordinary theft, but of a sum ~8x the largest the canon prices")
    print(f"   nearest precedent distance : {v['nearest_distance']:.1f}   "
          f"(extension boundary {net.extension_distance:.1f})")
    print(f"   verdict : {v['predicted_class']} (conf {v['confidence']:.2f})")
    print(f"   status  : {v['note']}")
    assert not v["settled_law"], "extension flag failed to fire on a novel case"

    print("\n" + "=" * 78)
    print("ALL SELF-TESTS PASSED.")
    print("Gradient check OK | metric learned the legally-relevant features |")
    print("verdicts are auditable to public precedents | extension flag works.")
    print("=" * 78)


if __name__ == "__main__":
    main()
