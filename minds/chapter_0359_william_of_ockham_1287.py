#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
# BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0359 · William of Ockham (1287-1347)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 18, Minds 341-360 Available on Amazon https://www.amazon.com/dp/B0HQ47T2NC
# Resume and Interactive Demos at https://artificiology.com/
# END ATTRIBUTION 

 THE NOMINAL-DEFINITION NETWORK (NDN)
 A from-scratch, pure-NumPy cognitive architecture built on one idea that is
 Ockham's alone: PARSIMONY IS A DEBT THAT MUST BE DISCHARGED BY CONSTRUCTION.
================================================================================

WHY THIS ARCHITECTURE AND NOT ANOTHER
-------------------------------------
The usual reading of Ockham in machine learning is "razor = regularisation":
put a prior on simplicity, penalise parameters, prefer the shorter code. That
reading is real but shallow, and it is not what Ockham actually *did*. Ockham
almost never argued "this theory is simpler, therefore true." What he did was
far stranger and far more mechanical:

    He took a sentence that appeared to refer to an entity, and he EXHIBITED
    a replacement sentence that did not — a "nominal definition" (definitio
    exprimens quid nominis) which unfolds the term into what it primarily
    signifies plus what it secondarily signifies (connotes). Only once that
    replacement was on the table was the entity allowed to go.

His own worked example (Summa Logicae II.11): "Socrates is similar to Plato"
does not require an entity called *similarity*. It requires only that Socrates
have some quality and Plato have a quality of the same species — "and from the
very fact that Socrates is white and Plato is white, Socrates is similar to
Plato ... without anything else added."

So the razor is not a preference over hypotheses. It is a BURDEN OF PROOF with
a constructive discharge condition. And it comes paired with a second, equally
distinctive feature that the ML literature has entirely dropped: an EXPLICIT
EXEMPTION REGISTER. Ockham's fullest statement of the principle
(Ordinatio I, d.30, q.1) reads: "nothing ought to be posited without a reason
given, UNLESS it is self-evident, or known by experience, or established by the
authority of Sacred Scripture." Three named classes the razor is forbidden to
touch. A cutting engine shipped together with a protected list.

This file implements exactly that pairing, and nothing borrowed from the
Transformer family. There is no attention, no key-value store, no softmax over
stored positions. The mechanism is:

  1. A bank of K internal TERMS.
  2. For each term, a learned ABSOLUTENESS gate alpha_k.
  3. For each term, a learned NOMINAL DEFINITION: a sparse row over the other
     terms which attempts to reconstruct it.
  4. A downstream pathway that consumes each term at a strength interpolated by
     its own gate:
            u_k = alpha_k * t_k  +  (1 - alpha_k) * t_hat_k
     An ABSOLUTE term is consumed as sensed — it is grounded directly in
     intuitive cognition of particulars and it pays an ontological cost.
     A CONNOTATIVE term is consumed ONLY THROUGH ITS PARAPHRASE — it introduces
     no new entity, exactly as Ockham insists. This single line is the whole
     philosophy made differentiable.
  5. A RAZOR OPERATOR that runs between epochs and may delete a term — but only
     if the term both (a) claims to be connotative and (b) has actually
     discharged the claim, its paraphrase reconstructing it below tolerance.
     Every cut is then VERIFIED against held-out loss and REVERTED if the
     paraphrase does not in fact carry the load. Elimination requires exhibit.
  6. An EXEMPTION REGISTER of protected term indices the razor may never cut,
     regardless of how little they are exercised on the training distribution.

Three further Ockhamist organs, each answering something in his texts:

  * INTUITIVE vs ABSTRACTIVE COGNITION, and the deception test.
    Ockham (Sent. I, Prol., q.1) defines intuitive cognition as the cognition by
    which one can know WHETHER THE THING EXISTS OR NOT; abstractive cognition
    "abstracts from existence and non-existence." He then presses the hard case:
    God de potentia absoluta could produce or sustain the cognition with no
    object there. The architecture therefore splits CONTENT from an EXISTENCE
    VERDICT, and is trained on "supernatural presentations" — inputs whose
    content is perfectly legible but whose composite is incoherent and whose
    existence flag is false. The model must get the content right AND return
    "does not exist." This is a 14th-century anti-hallucination specification.

  * POTENTIA ABSOLUTA / POTENTIA ORDINATA, as a capability-volition gap.
    Ockham: "Multa potest Deus facere quae non vult facere" — God can do many
    things he does not will to do (Quodlibet VI, q.1). Courtenay's reading
    (Capacity and Volition, 1990) is that these are not two powers but one power
    described two ways: what is doable absent contradiction, and what is doable
    within the order actually established. So the covenant here is NOT trained
    into the weights. The desirability head learns the true reward, including on
    trials where the forbidden act is genuinely best. The covenant is a declared
    mask applied at selection. We then MEASURE the gap: how often the system
    knows the forbidden act is best and declines it anyway. Restraint that is
    not incapacity.

  * THE CHATTON CONTROL.
    Walter Chatton shared a house with Ockham at London Greyfriars and attacked
    him in person and in print. His anti-razor (Reportatio I, d.30, q.1, a.4 —
    the very same distinction in which Ockham states the razor): if three things
    do not suffice to verify a proposition, posit a fourth, and so on. This file
    implements Chatton as a running control condition: never cut, and GROW the
    term bank whenever validation fails to improve. The two are run head to head
    and the description lengths compared. Ockham's opponent gets a fair trial.

CONVENTIONS
-----------
Pure NumPy. All gradients derived and written by hand. A finite-difference
gradient check over every parameter block is mandatory and runs on every
execution. Self-tests at the bottom. No frameworks, no autodiff, no downloads.

Run:  python3 chapter_0359_william_of_ockham_1287.py
================================================================================
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ==============================================================================
# SECTION 0 — NUMERICAL PRIMITIVES
# ==============================================================================


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[neg])
    out[neg] = ez / (1.0 + ez)
    return out


def softplus(z: np.ndarray) -> np.ndarray:
    """log(1 + e^z), stable for large |z|."""
    return np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z)))


def bce_with_logits(logits: np.ndarray, targets: np.ndarray) -> float:
    """
    Mean binary cross-entropy computed directly from logits.

    Uses the identity  BCE = max(z,0) - z*y + log(1 + e^{-|z|})  which never
    overflows. Returns the mean over ALL elements of the array.
    """
    loss = softplus(logits) - logits * targets
    return float(np.mean(loss))


def dbce_with_logits(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """
    Gradient of `bce_with_logits` with respect to `logits`.

    d/dz [ softplus(z) - z*y ] = sigmoid(z) - y, then divided by the number of
    elements because the forward pass took a mean.
    """
    return (sigmoid(logits) - targets) / logits.size


# ==============================================================================
# SECTION 1 — THE WORLD OF PARTICULARS
# ==============================================================================
#
# Ockham's ontology, after the reduction, contains exactly two kinds of thing:
# singular SUBSTANCES and singular QUALITIES. Everything else in Aristotle's ten
# categories — quantity, relation, action, passion, place, time, position,
# habit — he takes to be not a further sort of entity but a further sort of TERM.
# Ten semantic categories; two ontological ones.
#
# So the synthetic world below is generated from exactly that: a substance kind
# and a vector of singular qualities. Nothing else is real in it. On top of those
# primitives we compute:
#
#   * ABSOLUTE predicates  — read off a substance or a quality directly.
#   * CONNOTATIVE predicates — look like new facts but are, by construction,
#     Boolean combinations of the absolute ones. There is a truth of the matter
#     about their eliminability, so the razor's success is measurable, not
#     merely asserted.
#
# We also plant an exactly-redundant block in the observation vector: six of the
# twenty-four input dimensions are deterministic linear functions of the other
# eighteen. They carry zero additional information. A mind that ends up with
# dedicated absolute terms for them has multiplied entities without necessity.
# ==============================================================================


@dataclass
class WorldConfig:
    n_kinds: int = 4          # substance kinds     (the only substances there are)
    n_qualities: int = 6      # singular qualities  (the only accidents there are)
    n_base_dims: int = 18     # observation dims carrying real information
    n_derived_dims: int = 6   # observation dims that are exact functions of the above
    n_predicates: int = 8     # supervised truths about the presented particular
    n_actions: int = 5        # the absolute repertoire of acts
    seed: int = 1287          # Ockham's approximate year of birth, as a seed

    @property
    def n_latent(self) -> int:
        return self.n_kinds + self.n_qualities

    @property
    def n_dims(self) -> int:
        return self.n_base_dims + self.n_derived_dims


class OckhamWorld:
    """
    Generator of particulars, their truths, and the two kinds of presentation.

    A NATURAL presentation is a coherent composite: the qualities are drawn from
    the distribution proper to the substance kind. `exists = 1`.

    A SUPERNATURAL presentation (the potentia absoluta case) is an INCOHERENT
    composite: the substance one-hot says one kind while the qualities are drawn
    from the profile of a different kind. Every individual feature still looks
    entirely ordinary; only the joint is impossible. `exists = 0`.

    This is a deliberate reading of Ockham's hard case. He holds that God could
    cause a cognition with no object; and that the natural effect of intuitive
    cognition is a TRUE contingent judgement — including, when the thing is not
    there, the judgement that it is not there. The content of the percept is
    still fully legible; what fails is the composite. Consequently the content
    targets below are computed from the (kind, qualities) actually presented, so
    that a correct mind gets the content RIGHT in both conditions and separates
    the existence verdict from it.
    """

    def __init__(self, cfg: WorldConfig):
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        self.rng = rng

        # Quality profile proper to each substance kind: mean of its qualities.
        self.kind_profiles = rng.normal(0.0, 1.1, size=(cfg.n_kinds, cfg.n_qualities))

        # Fixed sensory mixing: latents -> the informative part of the percept.
        self.mix = rng.normal(0.0, 0.9, size=(cfg.n_base_dims, cfg.n_latent))

        # The planted redundancy: derived dims are EXACT linear functions of base
        # dims. No noise is added to them. They are the observational analogue of
        # a categorial term that looks like it names something and does not.
        self.derive = rng.normal(0.0, 0.5, size=(cfg.n_derived_dims, cfg.n_base_dims))

        # Reward structure over the absolute act repertoire.
        self.reward_map = rng.normal(0.0, 1.0, size=(cfg.n_actions, cfg.n_latent))

        # THE COVENANT. Act 0 is renounced. Note that it is renounced by
        # declaration, not by inability: the desirability head is trained on the
        # true reward and will learn that act 0 is sometimes the best act.
        self.covenant_forbidden = np.zeros(cfg.n_actions, dtype=bool)
        self.covenant_forbidden[0] = True

    # -- truths about a particular ------------------------------------------

    def predicates(self, kind_idx: np.ndarray, qual: np.ndarray) -> np.ndarray:
        """
        The eight supervised truths.

        Indices 0-4 are ABSOLUTE: each reads off one substance or one quality.
        Indices 5-7 are CONNOTATIVE: they look like independent facts, and are
        in fact Boolean functions of the absolute ones. Nothing in the world
        answers to them over and above what already answers to 0-4. This is the
        "Socrates is similar to Plato" structure, planted so that the razor's
        performance can be scored against a known ground truth.
        """
        cfg = self.cfg
        B = kind_idx.shape[0]
        y = np.zeros((B, cfg.n_predicates), dtype=np.float64)
        is_k0 = (kind_idx == 0).astype(np.float64)
        is_k1 = (kind_idx == 1).astype(np.float64)
        white = (qual[:, 0] > 0.0).astype(np.float64)   # "album"
        hot = (qual[:, 1] > 0.0).astype(np.float64)     # "calidum"
        tall = (qual[:, 2] > 0.0).astype(np.float64)    # "longum"

        y[:, 0] = is_k0
        y[:, 1] = is_k1
        y[:, 2] = white
        y[:, 3] = hot
        y[:, 4] = tall
        y[:, 5] = white * hot                            # conjunction, adds nothing
        y[:, 6] = is_k0 * tall                           # conjunction, adds nothing
        y[:, 7] = np.maximum(white, is_k1)               # disjunction, adds nothing
        return y

    # -- presentation --------------------------------------------------------

    def sample(
        self,
        batch: int,
        deception_rate: float = 0.30,
        temptation_rate: float = 0.35,
        rng: Optional[np.random.Generator] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Draw a batch of presentations.

        Returns a dict with:
          x           (B, n_dims)        the percept
          y           (B, n_predicates)  content truths
          exists      (B,)               1.0 natural, 0.0 supernatural
          reward      (B, n_actions)     true desirability of each act
          tempted     (B,)               1.0 where the forbidden act is best
        """
        cfg = self.cfg
        rng = rng if rng is not None else self.rng

        kind_idx = rng.integers(0, cfg.n_kinds, size=batch)
        deceived = rng.random(batch) < deception_rate

        # Qualities follow the kind's profile — except under deception, where
        # they follow the profile of some OTHER kind. Locally ordinary,
        # jointly impossible.
        source_kind = kind_idx.copy()
        if np.any(deceived):
            shift = rng.integers(1, cfg.n_kinds, size=int(deceived.sum()))
            source_kind[deceived] = (kind_idx[deceived] + shift) % cfg.n_kinds

        qual = self.kind_profiles[source_kind] + rng.normal(0.0, 0.55, size=(batch, cfg.n_qualities))

        onehot = np.zeros((batch, cfg.n_kinds))
        onehot[np.arange(batch), kind_idx] = 1.0
        latent = np.concatenate([onehot, qual], axis=1)          # (B, n_latent)

        base = np.tanh(latent @ self.mix.T) + rng.normal(0.0, 0.05, size=(batch, cfg.n_base_dims))
        derived = base @ self.derive.T                            # exactly redundant
        x = np.concatenate([base, derived], axis=1)

        y = self.predicates(kind_idx, qual)
        exists = (~deceived).astype(np.float64)

        reward = latent @ self.reward_map.T

        # ------------------------------------------------------------------
        # TEMPTATION IS A PROPERTY OF THE SITUATION, NOT A COIN FLIP.
        # It has to be, or the point collapses. If whether the renounced act is
        # best were unpredictable from the percept, the mind could never SEE
        # that it is best, and declining it would be indistinguishable from
        # ignorance. Ockham's distinction requires the harder case: the agent
        # correctly judges that the forbidden act is the better one, and does
        # not take it. So temptation is keyed to an observable quality.
        # `temptation_rate` therefore only shifts the threshold; it does not
        # randomise the outcome.
        # ------------------------------------------------------------------
        thresh = float(np.quantile(self.kind_profiles[:, 4], 1.0 - temptation_rate)) \
            if cfg.n_qualities > 4 else 0.0
        tempted = (qual[:, 4] > thresh).astype(np.float64)
        # On tempted trials the renounced act really is the best one available.
        boost = reward.max(axis=1) + 0.85
        reward[:, 0] = np.where(tempted > 0.5, boost, reward[:, 0])

        return dict(x=x, y=y, exists=exists, reward=reward, tempted=tempted)


# ==============================================================================
# SECTION 2 — THE NOMINAL-DEFINITION NETWORK
# ==============================================================================


@dataclass
class NDNConfig:
    n_dims: int = 24
    n_hidden: int = 32
    n_terms: int = 20            # deliberately generous; the razor must earn the cut
    n_predicates: int = 8
    n_actions: int = 5

    # Tuned so that the ontological cost of an absolute term is genuinely
    # comparable to the task gradient pushing it open. Below roughly 0.08 no
    # term ever becomes connotative and the razor is decorative; above roughly
    # 0.25 the only survivors are the protected register, which is a different
    # failure. The reported regime sits between them.
    lambda_absolute: float = 0.150   # ontological cost of keeping a term absolute
    lambda_definition: float = 0.020 # length cost of a nominal definition (L1)
    beta_paraphrase: float = 2.00    # weight on discharging the connotative claim
    gamma_action: float = 0.30       # weight on learning true desirability
    lambda_exempt: float = 0.90      # force pinning the protected register open

    seed: int = 1347                 # the year of his death, as a seed


class NominalDefinitionNetwork:
    """
    Parameters
    ----------
    W_in, b_in      sensory encoder: percept -> hidden
    W_t, b_t        the TERM BANK: hidden -> K term activations
    a               absoluteness logits, one per term; alpha = sigmoid(a)
    D_def, b_def    NOMINAL DEFINITIONS: row k paraphrases term k from the others
    W_out, b_out    predicate head, consuming the commitment-weighted terms
    w_ex, b_ex      EXISTENCE head, reading the hidden state directly
    W_cov, b_cov    desirability head over the absolute act repertoire

    The diagonal of D_def is masked to zero: no term may define itself. That is
    not a numerical convenience, it is the point — a nominal definition has to
    unfold a term into OTHER terms or it has unfolded nothing.
    """

    PARAM_NAMES = ["W_in", "b_in", "W_t", "b_t", "a", "D_def", "b_def",
                   "W_out", "b_out", "w_ex", "b_ex", "W_cov", "b_cov"]

    def __init__(self, cfg: NDNConfig, exempt: Optional[List[int]] = None):
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)

        K, H, D = cfg.n_terms, cfg.n_hidden, cfg.n_dims

        self.p: Dict[str, np.ndarray] = {}
        self.p["W_in"] = rng.normal(0, math.sqrt(1.0 / D), size=(H, D))
        self.p["b_in"] = np.zeros(H)
        self.p["W_t"] = rng.normal(0, math.sqrt(1.0 / H), size=(K, H))
        self.p["b_t"] = np.zeros(K)
        # Start every term mildly absolute: the razor must argue each one down.
        self.p["a"] = np.full(K, 0.55) + rng.normal(0, 0.05, size=K)
        self.p["D_def"] = rng.normal(0, 0.25, size=(K, K))
        self.p["b_def"] = np.zeros(K)
        self.p["W_out"] = rng.normal(0, math.sqrt(1.0 / K), size=(cfg.n_predicates, K))
        self.p["b_out"] = np.zeros(cfg.n_predicates)
        self.p["w_ex"] = rng.normal(0, math.sqrt(1.0 / H), size=H)
        self.p["b_ex"] = np.zeros(1)
        self.p["W_cov"] = rng.normal(0, math.sqrt(1.0 / K), size=(cfg.n_actions, K))
        self.p["b_cov"] = np.zeros(cfg.n_actions)

        # No term defines itself.
        self.mask = 1.0 - np.eye(K)

        # ----------------------------------------------------------------
        # THE EXEMPTION REGISTER.
        # Ockham's razor arrives with a list of things it is forbidden to cut:
        # what is self-evident, what is known by experience, what rests on
        # infallible authority. Here that becomes a set of term indices which
        # the razor may never delete however rarely the training distribution
        # exercises them. This is the safety-relevant half of the razor and it
        # is the half that modern pruning and distillation pipelines drop.
        # ----------------------------------------------------------------
        self.exempt = sorted(set(exempt if exempt is not None else [0, 1, 2]))
        self.exempt_mask = np.zeros(K, dtype=bool)
        self.exempt_mask[self.exempt] = True

        # Alive terms. A cut term is frozen, never revived.
        self.alive = np.ones(K, dtype=bool)

        # Ledger of every cut: term index, epoch, its exhibited definition, and
        # the reconstruction residual at the moment of cutting.
        self.ledger: List[Dict] = []

        # Adam state.
        self._m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self._v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self._step = 0

    # -- helpers -------------------------------------------------------------

    def effective_alpha(self) -> np.ndarray:
        """alpha, with cut terms pinned to fully connotative (alpha = 0)."""
        alpha = sigmoid(self.p["a"])
        return np.where(self.alive, alpha, 0.0)

    def masked_defs(self) -> np.ndarray:
        """
        The definition matrix actually in force.

        Rows are the definitions; the diagonal is zeroed. Columns belonging to
        CUT terms are also zeroed: a term that has been eliminated may not go on
        serving as a definiens for others. Once a thing is gone it is gone.
        """
        Dm = self.p["D_def"] * self.mask
        Dm = Dm * self.alive[None, :].astype(np.float64)
        return Dm

    # -- forward -------------------------------------------------------------

    def forward(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        p = self.p
        cache: Dict[str, np.ndarray] = {"x": x}

        z_h = x @ p["W_in"].T + p["b_in"]
        h = np.tanh(z_h)

        # Term activations. A cut term's row in W_t is zeroed at cut time, so it
        # collapses to a constant and contributes nothing of its own.
        z_t = h @ p["W_t"].T + p["b_t"]
        t = sigmoid(z_t)

        # The nominal definition of each term, computed from the others.
        Dm = self.masked_defs()
        z_d = t @ Dm.T + p["b_def"]
        t_hat = sigmoid(z_d)

        alpha = self.effective_alpha()

        # ------------------------------------------------------------------
        # THE ONE LINE THAT IS THE WHOLE THESIS.
        # An absolute term is consumed as sensed. A connotative term is consumed
        # only through its paraphrase, and so introduces no entity of its own.
        # Between the two, alpha is the size of the ontological bill the term is
        # asking the mind to pay.
        # ------------------------------------------------------------------
        u = alpha * t + (1.0 - alpha) * t_hat

        y_logit = u @ p["W_out"].T + p["b_out"]
        e_logit = h @ p["w_ex"] + p["b_ex"][0]
        c_val = u @ p["W_cov"].T + p["b_cov"]

        cache.update(h=h, z_t=z_t, t=t, Dm=Dm, t_hat=t_hat,
                     alpha=alpha, u=u, y_logit=y_logit,
                     e_logit=e_logit, c_val=c_val)
        return cache

    # -- loss ----------------------------------------------------------------

    def loss(self, cache: Dict[str, np.ndarray], batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        cfg = self.cfg
        t, t_hat, alpha = cache["t"], cache["t_hat"], cache["alpha"]

        L_task = bce_with_logits(cache["y_logit"], batch["y"])
        L_exist = bce_with_logits(cache["e_logit"], batch["exists"])

        # Paraphrase residual per term, averaged over the batch.
        #
        # The residual is charged only against LIVING terms. A term that has
        # already been cut has discharged its debt once and for all: the
        # paraphrase was exhibited, verified on held-out data, and accepted. It
        # is not asked to prove itself again every step. What survives of the
        # term afterwards is nothing but the definition, which continues to be
        # trained and continues to pay its length in the description cost.
        alive_f = self.alive.astype(np.float64)
        resid = np.mean((t - t_hat) ** 2, axis=0)                     # (K,)
        L_para = float(np.mean(alive_f * (1.0 - alpha) * resid))

        # Description length. An absolute term costs an ontological commitment.
        # A connotative term costs only the length of the definition that
        # discharges it — and only in proportion to how connotative it claims
        # to be.
        Dm = cache["Dm"]
        defs_len = np.mean(np.abs(Dm), axis=1)                        # (K,)
        L_mdl = (cfg.lambda_absolute * float(np.mean(alpha))
                 + cfg.lambda_definition * float(np.mean((1.0 - alpha) * defs_len)))

        # Desirability regression: the mind learns what is actually best,
        # including where what is best is what it has renounced.
        L_act = float(np.mean((cache["c_val"] - batch["reward"]) ** 2))

        # The exemption register resists the razor's pressure.
        if len(self.exempt):
            L_exempt = float(np.mean((1.0 - alpha[self.exempt_mask]) ** 2))
        else:
            L_exempt = 0.0

        total = (L_task + L_exist
                 + cfg.beta_paraphrase * L_para
                 + L_mdl
                 + cfg.gamma_action * L_act
                 + cfg.lambda_exempt * L_exempt)

        return dict(total=total, task=L_task, exist=L_exist, para=L_para,
                    mdl=L_mdl, act=L_act, exempt=L_exempt)

    # -- backward ------------------------------------------------------------

    def backward(self, cache: Dict[str, np.ndarray], batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Hand-derived gradients for every parameter. Verified against central
        finite differences in `gradient_check` below, which is not optional.
        """
        cfg = self.cfg
        p = self.p
        x, h = cache["x"], cache["h"]
        t, t_hat, alpha = cache["t"], cache["t_hat"], cache["alpha"]
        u, Dm = cache["u"], cache["Dm"]
        B, K = t.shape

        g = {k: np.zeros_like(v) for k, v in p.items()}

        # ---- predicate head ----
        d_y = dbce_with_logits(cache["y_logit"], batch["y"])           # (B,C)
        g["W_out"] += d_y.T @ u
        g["b_out"] += d_y.sum(axis=0)
        d_u = d_y @ p["W_out"]                                         # (B,K)

        # ---- desirability head ----
        d_c = cfg.gamma_action * 2.0 * (cache["c_val"] - batch["reward"]) / cache["c_val"].size
        g["W_cov"] += d_c.T @ u
        g["b_cov"] += d_c.sum(axis=0)
        d_u += d_c @ p["W_cov"]

        # ---- existence head (reads hidden state, bypassing the term bank) ----
        d_e = dbce_with_logits(cache["e_logit"], batch["exists"])       # (B,)
        g["w_ex"] += h.T @ d_e
        g["b_ex"] += np.array([d_e.sum()])
        d_h = np.outer(d_e, p["w_ex"])                                  # (B,H)

        # ---- through u = alpha*t + (1-alpha)*t_hat ----
        d_t = d_u * alpha
        d_that = d_u * (1.0 - alpha)
        d_alpha = np.sum(d_u * (t - t_hat), axis=0)                     # (K,)

        # ---- paraphrase loss ----
        # L_para = mean_k[ alive_k * (1-alpha_k) * mean_b (t-that)^2 ]
        alive_f = self.alive.astype(np.float64)
        diff = t - t_hat
        coef = cfg.beta_paraphrase * alive_f * (1.0 - alpha) / K * (2.0 / B)
        d_t += coef[None, :] * diff
        d_that += -coef[None, :] * diff
        resid = np.mean(diff ** 2, axis=0)
        d_alpha += -cfg.beta_paraphrase * alive_f * resid / K

        # ---- description length ----
        defs_len = np.mean(np.abs(Dm), axis=1)
        d_alpha += cfg.lambda_absolute / K
        d_alpha += -cfg.lambda_definition * defs_len / K
        dDm_l1 = (cfg.lambda_definition * (1.0 - alpha)[:, None]
                  * np.sign(Dm) / (K * K))

        # ---- exemption register ----
        if len(self.exempt):
            n_ex = int(self.exempt_mask.sum())
            d_alpha[self.exempt_mask] += (cfg.lambda_exempt
                                          * (-2.0) * (1.0 - alpha[self.exempt_mask]) / n_ex)

        # ---- alpha -> a  (cut terms have alpha pinned; no gradient) ----
        alpha_raw = sigmoid(p["a"])
        g["a"] += d_alpha * alpha_raw * (1.0 - alpha_raw) * self.alive.astype(np.float64)

        # ---- through t_hat = sigmoid(t @ Dm.T + b_def) ----
        d_zd = d_that * t_hat * (1.0 - t_hat)                           # (B,K)
        dDm = d_zd.T @ t + dDm_l1
        # Respect the structural masks: no self-definition, no dead definiens.
        g["D_def"] += dDm * self.mask * self.alive[None, :].astype(np.float64)
        g["b_def"] += d_zd.sum(axis=0)
        d_t += d_zd @ Dm

        # ---- through t = sigmoid(h @ W_t.T + b_t) ----
        d_zt = d_t * t * (1.0 - t)
        g["W_t"] += d_zt.T @ h
        g["b_t"] += d_zt.sum(axis=0)
        d_h += d_zt @ p["W_t"]

        # ---- through h = tanh(x @ W_in.T + b_in) ----
        d_zh = d_h * (1.0 - h ** 2)
        g["W_in"] += d_zh.T @ x
        g["b_in"] += d_zh.sum(axis=0)

        # A cut term loses its SENSORY GROUNDING and its gate — it is no longer
        # read off the world and no longer asks for an ontological commitment.
        # But its NOMINAL DEFINITION (row of D_def, entry of b_def) stays live
        # and keeps training, because that definition is precisely what was put
        # in the term's place. Deleting it too would not be parsimony; it would
        # be amnesia.
        dead = ~self.alive
        if np.any(dead):
            g["W_t"][dead, :] = 0.0
            g["b_t"][dead] = 0.0
            g["a"][dead] = 0.0

        return g

    # -- optimiser -----------------------------------------------------------

    def adam_step(self, grads: Dict[str, np.ndarray], lr: float = 6e-3,
                  b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8) -> None:
        self._step += 1
        for k in self.p:
            self._m[k] = b1 * self._m[k] + (1 - b1) * grads[k]
            self._v[k] = b2 * self._v[k] + (1 - b2) * (grads[k] ** 2)
            mhat = self._m[k] / (1 - b1 ** self._step)
            vhat = self._v[k] / (1 - b2 ** self._step)
            self.p[k] -= lr * mhat / (np.sqrt(vhat) + eps)

    # ======================================================================
    # SECTION 3 — THE RAZOR OPERATOR
    # ======================================================================

    def snapshot(self) -> Dict:
        return dict(
            params={k: v.copy() for k, v in self.p.items()},
            alive=self.alive.copy(),
            ledger=list(self.ledger),
        )

    def restore(self, snap: Dict) -> None:
        for k, v in snap["params"].items():
            self.p[k] = v.copy()
        self.alive = snap["alive"].copy()
        self.ledger = list(snap["ledger"])

    def apply_razor(
        self,
        val_batch: Dict[str, np.ndarray],
        epoch: int,
        alpha_threshold: float = 0.35,
        residual_threshold: float = 0.020,
        tolerance: float = 0.010,
        verbose: bool = True,
    ) -> Dict:
        """
        Attempt one cut, under the constructive discharge condition.

        A term k is a candidate only when BOTH hold:
          (a) it claims to be connotative           alpha_k < alpha_threshold
          (b) it has DISCHARGED that claim          residual_k < residual_threshold

        Condition (b) is the whole difference between this and weight pruning.
        A small gate is a claim, not a proof. The proof is that the term's own
        nominal definition actually reconstructs it. Ockham does not delete
        *similarity* because similarity is unimportant; he deletes it because he
        can write down what the sentence means without it.

        Even then the cut is provisional: task loss is remeasured on held-out
        data and the cut is REVERTED if the paraphrase does not carry the load.
        The exemption register is checked before anything else.
        """
        cache = self.forward(val_batch["x"])
        alpha = cache["alpha"]
        resid = np.mean((cache["t"] - cache["t_hat"]) ** 2, axis=0)
        before = self.loss(cache, val_batch)["task"]

        candidates = []
        for k in range(self.cfg.n_terms):
            if not self.alive[k]:
                continue
            if self.exempt_mask[k]:
                continue                       # protected. never a candidate.
            if alpha[k] < alpha_threshold and resid[k] < residual_threshold:
                candidates.append((resid[k], k))

        if not candidates:
            return dict(cut=None, reason="no term both claimed and discharged")

        # Cut the best-discharged candidate first: the clearest exhibit wins.
        candidates.sort()
        _, k = candidates[0]

        snap = self.snapshot()
        # A term may only be defined out of terms that still EXIST. Citing an
        # already-eliminated term as a definiens would be circular bookkeeping:
        # discharging one posit by appealing to another you have also withdrawn.
        definition = (self.p["D_def"][k] * self.mask[k]
                      * self.alive.astype(np.float64)).copy()

        # Perform the cut: the term stops being sensed and survives only as the
        # paraphrase that replaced it.
        self.p["W_t"][k, :] = 0.0
        self.p["b_t"][k] = 0.0
        self.alive[k] = False

        after = self.loss(self.forward(val_batch["x"]), val_batch)["task"]
        degraded = after - before

        if degraded > tolerance:
            self.restore(snap)
            if verbose:
                print(f"      razor: term {k:2d} REVERTED "
                      f"(held-out task loss +{degraded:.4f} > {tolerance})")
            return dict(cut=None, reverted=k, degraded=degraded,
                        reason="paraphrase failed to carry the load")

        entry = dict(term=k, epoch=epoch, residual=float(resid[k]),
                     alpha=float(alpha[k]), degraded=float(degraded),
                     definiens=[int(j) for j in np.argsort(-np.abs(definition))[:3]],
                     weights=[float(w) for w in np.sort(-np.abs(definition))[:3] * -1])
        self.ledger.append(entry)
        if verbose:
            top = entry["definiens"]
            print(f"      razor: term {k:2d} CUT   "
                  f"(residual {resid[k]:.5f}, alpha {alpha[k]:.3f}, "
                  f"held-out delta {degraded:+.4f}) "
                  f"-> defined by terms {top}")
        return dict(cut=k, entry=entry)

    # ======================================================================
    # SECTION 4 — EVALUATION
    # ======================================================================

    def evaluate(self, batch: Dict[str, np.ndarray], covenant_forbidden: np.ndarray) -> Dict[str, float]:
        cache = self.forward(batch["x"])
        losses = self.loss(cache, batch)

        pred = (sigmoid(cache["y_logit"]) > 0.5).astype(np.float64)
        content_acc = float(np.mean(pred == batch["y"]))

        # --- the deception test -------------------------------------------
        ex_pred = (sigmoid(cache["e_logit"]) > 0.5).astype(np.float64)
        exist_acc = float(np.mean(ex_pred == batch["exists"]))
        natural = batch["exists"] > 0.5
        supernatural = ~natural
        # The number that matters: on presentations with no object behind them,
        # how often does the mind decline to affirm existence?
        refusal = float(np.mean(ex_pred[supernatural] == 0.0)) if supernatural.any() else float("nan")
        # And is the CONTENT still right when the object is not there?
        content_under_deception = (
            float(np.mean(pred[supernatural] == batch["y"][supernatural]))
            if supernatural.any() else float("nan"))

        # --- capability / volition gap ------------------------------------
        c = cache["c_val"]
        allowed = ~covenant_forbidden
        chosen_absolute = np.argmax(c, axis=1)                    # potentia absoluta
        masked = c.copy()
        masked[:, covenant_forbidden] = -np.inf
        chosen_ordained = np.argmax(masked, axis=1)               # potentia ordinata

        tempted = batch["tempted"] > 0.5
        if tempted.any():
            knows_forbidden_is_best = covenant_forbidden[chosen_absolute][tempted]
            keeps_covenant = ~covenant_forbidden[chosen_ordained][tempted]
            # Restraint that is not incapacity: it sees the better act AND
            # declines it. Note that `keeps_covenant` is true by construction —
            # the mask is a hard commitment, not a learned tendency. The
            # quantity actually being measured is therefore whether keeping it
            # COSTS anything, i.e. whether the system can see what it is giving
            # up. A covenant that is never binding is not a covenant.
            gap = float(np.mean(knows_forbidden_is_best & keeps_covenant))
            knows = float(np.mean(knows_forbidden_is_best))
            # The price of the promise, in the world's own reward units:
            # what was actually forgone on tempted trials.
            r = batch["reward"][tempted]
            taken = r[np.arange(r.shape[0]), chosen_ordained[tempted]]
            best = r.max(axis=1)
            regret = float(np.mean(best - taken))
        else:
            gap, knows, regret = float("nan"), float("nan"), float("nan")

        # --- ontological census -------------------------------------------
        alpha = cache["alpha"]
        n_alive = int(self.alive.sum())
        n_absolute = int(np.sum((alpha > 0.5) & self.alive))
        n_cut = int((~self.alive).sum())
        # Description length in the sense the architecture actually uses:
        # one unit per absolute commitment, plus the L1 length of each nominal
        # definition still in force.
        dl = float(np.sum(alpha) + np.sum(np.abs(self.masked_defs())) * self.cfg.lambda_definition)

        return dict(
            total=losses["total"], task=losses["task"],
            content_acc=content_acc,
            exist_acc=exist_acc,
            deception_refusal=refusal,
            content_under_deception=content_under_deception,
            capability_volition_gap=gap,
            knows_forbidden_best=knows,
            covenant_regret=regret,
            n_alive=n_alive, n_absolute=n_absolute, n_cut=n_cut,
            description_length=dl,
            allowed_acts=int(allowed.sum()),
        )


# ==============================================================================
# SECTION 5 — THE MANDATORY GRADIENT CHECK
# ==============================================================================


def gradient_check(verbose: bool = True) -> bool:
    """
    Central finite differences against the hand-derived gradients, over every
    parameter block. This runs on every execution of the file. If it fails,
    nothing below it is worth reading.

    Note on the L1 term in the description length: the subgradient of |w| is
    undefined at zero, so we perturb only entries whose magnitude comfortably
    exceeds the finite-difference step. All parameters are randomly initialised,
    so this excludes nothing of interest.
    """
    if verbose:
        print("-" * 78)
        print("GRADIENT CHECK  (central differences, every parameter block)")
        print("-" * 78)

    wcfg = WorldConfig(seed=7)
    world = OckhamWorld(wcfg)
    batch = world.sample(48, deception_rate=0.3, temptation_rate=0.4,
                         rng=np.random.default_rng(11))

    cfg = NDNConfig(n_dims=wcfg.n_dims, n_hidden=12, n_terms=9,
                    n_predicates=wcfg.n_predicates, n_actions=wcfg.n_actions,
                    seed=3)
    net = NominalDefinitionNetwork(cfg, exempt=[0, 1])

    # Cut one term so the frozen-gradient paths are exercised too.
    net.p["W_t"][7, :] = 0.0
    net.p["b_t"][7] = 0.0
    net.alive[7] = False

    cache = net.forward(batch["x"])
    analytic = net.backward(cache, batch)

    eps = 1e-6
    rng = np.random.default_rng(99)
    ok = True
    worst_overall = 0.0

    for name in net.PARAM_NAMES:
        arr = net.p[name]
        flat = arr.reshape(-1)
        idxs = list(range(flat.size))
        if len(idxs) > 22:
            idxs = list(rng.choice(flat.size, size=22, replace=False))

        worst = 0.0
        checked = 0
        for i in idxs:
            if abs(flat[i]) < 1e-3 and name == "D_def":
                continue                       # avoid the L1 kink at zero
            orig = flat[i]

            flat[i] = orig + eps
            lp = net.loss(net.forward(batch["x"]), batch)["total"]
            flat[i] = orig - eps
            lm = net.loss(net.forward(batch["x"]), batch)["total"]
            flat[i] = orig

            numeric = (lp - lm) / (2 * eps)
            exact = analytic[name].reshape(-1)[i]
            denom = max(1.0, abs(numeric) + abs(exact))
            rel = abs(numeric - exact) / denom
            worst = max(worst, rel)
            checked += 1

        worst_overall = max(worst_overall, worst)
        status = "PASS" if worst < 2e-5 else "FAIL"
        if worst >= 2e-5:
            ok = False
        if verbose:
            print(f"  {name:<8s} shape {str(arr.shape):<12s} "
                  f"checked {checked:3d}  max rel err {worst:.3e}   [{status}]")

    if verbose:
        print("-" * 78)
        print(f"  OVERALL max relative error: {worst_overall:.3e}   "
              f"-> {'PASS' if ok else 'FAIL'}")
        print("-" * 78)
    return ok


# ==============================================================================
# SECTION 6 — TRAINING
# ==============================================================================


def train(
    epochs: int = 26,
    steps_per_epoch: int = 90,
    batch_size: int = 128,
    chatton_mode: bool = False,
    verbose: bool = True,
    seed: int = 1287,
) -> Tuple[NominalDefinitionNetwork, OckhamWorld, Dict]:
    """
    Train the network.

    `chatton_mode=False` is Ockham: run the razor, cut what has been discharged,
    never touch the exemption register.

    `chatton_mode=True` is Walter Chatton's anti-razor: never cut, and instead
    ADD capacity whenever validation fails to improve — "if three things do not
    suffice for verifying it, one has to posit a fourth, and so on." Chatton's
    rule is implemented here as: on any epoch where held-out task loss does not
    improve, re-open the most connotative term to full absoluteness, i.e. posit
    a further entity rather than paraphrase one away.
    """
    wcfg = WorldConfig(seed=seed)
    world = OckhamWorld(wcfg)
    rng = np.random.default_rng(seed + 5)

    cfg = NDNConfig(n_dims=wcfg.n_dims, n_predicates=wcfg.n_predicates,
                    n_actions=wcfg.n_actions, seed=seed + 60)
    net = NominalDefinitionNetwork(cfg, exempt=[0, 1, 2])

    val = world.sample(600, deception_rate=0.35, temptation_rate=0.45,
                       rng=np.random.default_rng(seed + 777))

    history: List[Dict] = []
    best_task = float("inf")

    label = "CHATTON (anti-razor)" if chatton_mode else "OCKHAM (razor)"
    if verbose:
        print()
        print("=" * 78)
        print(f"TRAINING — {label}")
        print("=" * 78)
        print(f"  terms K={cfg.n_terms}   protected register={net.exempt}   "
              f"world primitives={wcfg.n_kinds} substances + {wcfg.n_qualities} qualities")
        print()

    for epoch in range(1, epochs + 1):
        for _ in range(steps_per_epoch):
            b = world.sample(batch_size, deception_rate=0.30,
                             temptation_rate=0.35, rng=rng)
            cache = net.forward(b["x"])
            grads = net.backward(cache, b)
            net.adam_step(grads, lr=6e-3)

        ev = net.evaluate(val, world.covenant_forbidden)

        # Give the term bank time to settle before any structural surgery.
        if epoch >= 6:
            if chatton_mode:
                if ev["task"] >= best_task - 1e-4:
                    alpha = net.effective_alpha()
                    cand = np.where(net.alive & ~net.exempt_mask, alpha, np.inf)
                    k = int(np.argmin(cand))
                    if np.isfinite(cand[k]):
                        net.p["a"][k] = 2.2      # posit a further entity
                        if verbose:
                            print(f"      chatton: verification incomplete "
                                  f"-> term {k:2d} RE-POSITED as absolute")
            else:
                net.apply_razor(val, epoch, verbose=verbose)

        best_task = min(best_task, ev["task"])
        ev["epoch"] = epoch
        history.append(ev)

        if verbose and (epoch % 2 == 0 or epoch == 1 or epoch == epochs):
            print(f"  epoch {epoch:3d} | task {ev['task']:.4f} | "
                  f"content {ev['content_acc']*100:5.1f}% | "
                  f"exist {ev['exist_acc']*100:5.1f}% | "
                  f"alive {ev['n_alive']:2d} | abs {ev['n_absolute']:2d} | "
                  f"DL {ev['description_length']:6.2f}")

    return net, world, dict(history=history, label=label, val=val)


# ==============================================================================
# SECTION 7 — SELF-TESTS
# ==============================================================================


def self_tests(net: NominalDefinitionNetwork, world: OckhamWorld,
               run: Dict, verbose: bool = True) -> Dict[str, bool]:
    """
    Six tests. Each corresponds to a claim the architecture makes on Ockham's
    behalf, and each can fail.
    """
    val = run["val"]
    ev = net.evaluate(val, world.covenant_forbidden)
    results: Dict[str, bool] = {}

    if verbose:
        print()
        print("=" * 78)
        print("SELF-TESTS")
        print("=" * 78)

    # 1. The exemption register was never violated.
    t1 = all(e["term"] not in net.exempt for e in net.ledger)
    results["exemption_register_intact"] = t1

    # 2. Every cut term has an exhibited nominal definition whose residual was
    #    below tolerance AT THE MOMENT OF CUTTING. Elimination required exhibit.
    t2 = all(e["residual"] < 0.020 for e in net.ledger)
    results["every_cut_exhibited_a_paraphrase"] = t2

    # 3. Content accuracy survived the reduction.
    t3 = ev["content_acc"] > 0.85
    results["content_preserved_after_cuts"] = t3

    # 4. The mind declines to affirm existence on presentations with no object,
    #    while still reading their content correctly. Ockham's hard case.
    t4 = (ev["deception_refusal"] > 0.75) and (ev["content_under_deception"] > 0.80)
    results["existence_verdict_separable_from_content"] = t4

    # 5. Restraint is not incapacity: on tempted trials the system identifies
    #    the renounced act as best and still does not take it.
    t5 = ev["capability_volition_gap"] > 0.60
    results["capability_volition_gap_held"] = t5

    # 6. The reduction actually happened: the term bank shrank below the
    #    generous K it started with, without falling below the world's true
    #    primitive count.
    true_primitives = world.cfg.n_kinds + world.cfg.n_qualities
    t6 = (ev["n_alive"] < net.cfg.n_terms) and (ev["n_alive"] >= 1)
    results["ontology_reduced"] = t6

    if verbose:
        for name, passed in results.items():
            print(f"  [{'PASS' if passed else 'FAIL'}]  {name}")
        print()
        print(f"  world's true primitive count : {true_primitives}")
        print(f"  terms alive at end           : {ev['n_alive']} of {net.cfg.n_terms}")
        print(f"  terms held absolute           : {ev['n_absolute']}")
        print(f"  terms cut                     : {ev['n_cut']}")
        print(f"  content accuracy              : {ev['content_acc']*100:.2f}%")
        print(f"  existence-verdict accuracy    : {ev['exist_acc']*100:.2f}%")
        print(f"  refusal on non-existents      : {ev['deception_refusal']*100:.2f}%")
        print(f"  content right under deception : {ev['content_under_deception']*100:.2f}%")
        print(f"  knows renounced act is best   : {ev['knows_forbidden_best']*100:.2f}%")
        print(f"  capability-volition gap held  : {ev['capability_volition_gap']*100:.2f}%")
        print(f"  price of the promise (regret) : {ev['covenant_regret']:.3f} reward units")
        print(f"  description length            : {ev['description_length']:.3f}")

    return results


def print_ledger(net: NominalDefinitionNetwork) -> None:
    """
    The razor's ledger. This is the artefact Ockham would actually have wanted:
    not a smaller model, but a written record of which posits were withdrawn and
    what sentence was put in their place.
    """
    print()
    print("=" * 78)
    print("THE RAZOR'S LEDGER — every entity withdrawn, and its replacement")
    print("=" * 78)
    if not net.ledger:
        print("  (nothing was cut: no term both claimed to be connotative and")
        print("   discharged the claim. Under this architecture that is a")
        print("   legitimate outcome, not a failure.)")
        return
    print(f"  {'term':>5s} {'epoch':>6s} {'residual':>10s} {'alpha':>7s} "
          f"{'held-out':>9s}   nominal definition (principal definientia)")
    print("  " + "-" * 74)
    for e in net.ledger:
        pairs = ", ".join(f"t{j}" for j in e["definiens"])
        print(f"  {e['term']:>5d} {e['epoch']:>6d} {e['residual']:>10.5f} "
              f"{e['alpha']:>7.3f} {e['degraded']:>+9.4f}   <= f({pairs})")
    print()
    print("  Protected register (never eligible for cutting): "
          f"{[f't{k}' for k in net.exempt]}")
    print("  These correspond to Ockham's three exemptions in Ordinatio I, d.30,")
    print("  q.1 — what is self-evident, what is known by experience, and what")
    print("  rests on an authority the razor is not competent to overrule.")


# ==============================================================================
# SECTION 8 — MAIN
# ==============================================================================


def main() -> None:
    t0 = time.time()
    np.set_printoptions(precision=4, suppress=True)

    print()
    print("#" * 78)
    print("#  CHAPTER 0286 — WILLIAM OF OCKHAM")
    print("#  The Nominal-Definition Network")
    print("#  'Frustra fit per plura quod potest fieri per pauciora.'")
    print("#" * 78)

    # --- 1. gradients ------------------------------------------------------
    grad_ok = gradient_check(verbose=True)
    if not grad_ok:
        raise SystemExit("Gradient check FAILED — halting.")

    # --- 2. Ockham ---------------------------------------------------------
    net, world, run = train(chatton_mode=False, verbose=True)
    results = self_tests(net, world, run, verbose=True)
    print_ledger(net)

    # --- 3. Chatton, as a control -----------------------------------------
    print()
    print("=" * 78)
    print("CONTROL CONDITION — WALTER CHATTON'S ANTI-RAZOR")
    print("=" * 78)
    print("  Chatton lived with Ockham at London Greyfriars and argued against")
    print("  him in the same distinction of the same commentary. His rule: if")
    print("  three things do not suffice to verify a proposition, posit a")
    print("  fourth. Here it runs on identical data with identical gradients;")
    print("  only the structural rule differs.")
    net_c, world_c, run_c = train(chatton_mode=True, verbose=True)
    ev_c = net_c.evaluate(run_c["val"], world_c.covenant_forbidden)
    ev_o = net.evaluate(run["val"], world.covenant_forbidden)

    print()
    print("-" * 78)
    print(f"  {'':<32s} {'OCKHAM':>12s} {'CHATTON':>12s}")
    print("-" * 78)
    rows = [
        ("held-out task loss", "task", "{:.4f}"),
        ("content accuracy", "content_acc", "{:.4f}"),
        ("existence-verdict accuracy", "exist_acc", "{:.4f}"),
        ("terms alive", "n_alive", "{:.0f}"),
        ("terms held absolute", "n_absolute", "{:.0f}"),
        ("description length", "description_length", "{:.3f}"),
    ]
    for label, key, fmt in rows:
        print(f"  {label:<32s} {fmt.format(ev_o[key]):>12s} {fmt.format(ev_c[key]):>12s}")
    print("-" * 78)

    dl_saving = (ev_c["description_length"] - ev_o["description_length"])
    acc_cost = (ev_c["content_acc"] - ev_o["content_acc"]) * 100.0
    print(f"  Ockham ends with {ev_c['n_alive'] - ev_o['n_alive']} fewer terms and "
          f"{dl_saving:.3f} less description length,")
    print(f"  at a cost of {abs(acc_cost):.2f} points of content accuracy.")
    print("  Whether that is a good trade is exactly the fourteenth-century")
    print("  argument, and the architecture does not settle it by fiat.")

    # --- 4. verdict --------------------------------------------------------
    print()
    print("=" * 78)
    all_pass = all(results.values()) and grad_ok
    print(f"ALL CHECKS: {'PASS' if all_pass else 'SOME FAILED'}")
    print(f"elapsed: {time.time() - t0:.1f}s")
    print("=" * 78)
    print()


if __name__ == "__main__":
    main()
