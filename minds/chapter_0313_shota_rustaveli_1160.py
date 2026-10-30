#!/usr/bin/env python3
"""
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0313 · Shota Rustaveli (c. 1160 – after c. 1220), Georgia
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 16, Minds 301-320 Available on Amazon https://www.amazon.com/gp/product/B0HNJQX9S4
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos 

THE SWORN-COVENANT RECOGNITION NETWORK
(Georgian: "dzmobis kseli" — a network of oath-brotherhood)

---------------------------------------------------------------------
WHY THIS ARCHITECTURE, AND NOT A TRANSFORMER
---------------------------------------------------------------------
This is deliberately NOT attention-over-stored-keys. Rustaveli's own
surviving text — the Prologue and the plot machinery of "The Knight
in the Panther's Skin" — states a different theory of how one mind
comes to know another, and it is testable enough to build:

  1. RECOGNITION IS COSTLY AND BINARY, NOT A SOFT WEIGHTED AVERAGE.
     Avtandil does not "attend" to Tariel with partial weight while
     also attending to the king's court. He tries the man, in
     private, over repeated meetings, until a threshold of trust is
     crossed — and then swears dzmobisa (formal, ritual brotherhood).
     After that the bond is (in the poem) permanent and admits full
     disclosure. Before it, almost nothing is disclosed. This is a
     STEP FUNCTION over an interaction history, not a softmax.

  2. THE SWORN BOND CAN OVERRIDE THE CHAIN OF COMMAND.
     Avtandil deserts King Rostevan's court — an act of open
     insubordination against his sovereign and prospective
     father-in-law — solely because he is oath-bound to Tariel. The
     architecture needs an explicit GATE that lets a lateral,
     freely-chosen bond outrank a vertical, structurally-assigned one
     under the right trust conditions, and only then.

  3. WORTH COMPOUNDS WHEN CIRCULATED AND ROTS WHEN HOARDED.
     "That which we give, doth remain our own; / What we withhold,
     we lose" (Prologue, on generosity; Wardrop's and Urushadze's
     translations agree on the shape of the claim even where the
     wording differs — see Sources). This is a claim about the
     DYNAMICS of a resource under a giving policy, not a metaphor:
     it says a system that only ever accumulates state is a system
     that is, in the long run, losing it. That is implementable as a
     decay-vs-transfer-bonus dynamical rule, and testable.

  4. DANGEROUS TRUTHS ARE DISPLACED ONE ALLEGORICAL STEP AWAY.
     Rustaveli sets his poem in invented "Arabia" and "India" with no
     real Georgian place names, while scholarship agrees the poem
     reflects the court of a real, contemporary, reigning queen
     (Tamar) whose legitimacy as a female sovereign a court poet
     could not defend by direct argument without risk. The surface
     text is systematically rotated away from its referent. That is
     an encode/decode pair: a fixed, non-secret transform that
     displaces the content, and a KEY — held only by those inside the
     circle who already know what/whom is meant — that inverts it.
     Readers outside the circle get a costume drama; readers inside
     it get a commentary on Tamar's court.

  5. CAPABILITY IS A PROPERTY OF INNER KIND, NOT OF OUTWARD VESSEL.
     "A lion's cub is a lion, be it male or female" (a line the poem
     uses to defend the reigning queen's legitimacy, and which the
     plot enacts twice over — Tinatin inherits a throne, and
     Nestan-Darejan turns out to be the missing sovereign the whole
     quest was organized to restore). This is an explicit claim of
     INVARIANCE: whatever function computes "fitness to reign" must
     not be a function of an outward, structurally-irrelevant
     category label. It is a fairness/invariance constraint on the
     recognition network's own scoring function, and it is testable
     by intervention.

None of this is Homer's formulaic mêtis-under-a-metrical-gate (id 20),
Nizami's anti-plagiarism ledger of sealed chambers (id 251), or
Ferdowsi's externally-licensed, revocable farr (id 217) — the three
nearest generative_poetic neighbours this chapter was checked against.
Rustaveli's distinctive claim is about how RECOGNITION BETWEEN TWO
MINDS IS EARNED, GATED, AND MADE TO OUTRANK STRUCTURE, and about the
tax that secrecy and hoarding levy on both information and love.

---------------------------------------------------------------------
WHAT IS IMPLEMENTED (pure NumPy, no autodiff library)
---------------------------------------------------------------------
  ScoreNet          the Trial-of-Worth network: two hidden-state
                     vectors in, a "kinship" (steadfast-trust)
                     probability out. Hand-derived forward/backward.
  GateNet            the Oath-Override gate: given a vertical
                     (command) signal, a lateral (sworn-peer) signal,
                     and their conflict, decides how much the sworn
                     bond outranks the chain of command.
  CovenantCipher      a fixed linear "allegorical displacement" (the
                     invented-Arabia/India transform) plus a TRAINED
                     decoder matrix that only becomes accurate on
                     bonded pairs — i.e. the "key" is something the
                     sworn circle learns, not something given away.
  GivingPolicyNet     the Circulatory Generosity network: decides
                     what fraction of held wealth to transmit to a
                     bonded neighbour in need.
  ValueHeads          two linear heads over the same hidden state:
                     V_steadfast (the poem's "heaven's kind" of love
                     — a low-variance, context-invariant target) and
                     V_appetite (the poem's "lechery" — a
                     high-variance, context-driven target).
  CovenantNetwork     orchestrates a small graph of nodes named after
                     the poem's cast, running bonding, recognition
                     message-passing, and wealth circulation as an
                     inference-time dynamical system on top of the
                     five trained sub-networks above.

Every trainable module has a hand-derived analytic backward pass,
verified below by a mandatory finite-difference gradient check.
The file trains, self-tests, and prints a verification report when
run directly.

Author's note on candour: this is a computational MODEL inspired by
verified claims in Rustaveli's surviving Prologue and by the plot
architecture of his one surviving poem, not a discovery about his
brain. Where the poem's wording is genuinely ambiguous (the closing
sense of the generosity couplet varies slightly by translator; see
Sources), the architecture follows the reading shared by the two
translations checked, not a single translator's phrasing.
"""

from __future__ import annotations
import numpy as np
import json

RNG_SEED = 313
rng = np.random.default_rng(RNG_SEED)

# =====================================================================
# SECTION 0 — small numerics toolkit (activations + a generic linear
# layer with a hand-written forward/backward pass). No autodiff engine
# is used anywhere in this file; every gradient below is derived by
# hand and checked against finite differences in Section 6.
# =====================================================================

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))

def dsigmoid_from_output(y):
    # dy/dx given y = sigmoid(x)
    return y * (1.0 - y)

def tanh(x):
    return np.tanh(x)

def dtanh_from_output(y):
    return 1.0 - y ** 2

def relu(x):
    return np.maximum(0.0, x)

def drelu_from_input(x):
    return (x > 0.0).astype(x.dtype)


class Linear:
    """A single dense layer y = x @ W + b with a manual backward pass."""

    def __init__(self, in_dim: int, out_dim: int, seed: int):
        r = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / (in_dim + out_dim))
        self.W = r.normal(0.0, scale, size=(in_dim, out_dim))
        self.b = np.zeros(out_dim)
        self._x = None
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x):
        self._x = x
        return x @ self.W + self.b

    def backward(self, dout):
        x = self._x
        self.dW = x.T @ dout
        self.db = dout.sum(axis=0)
        dx = dout @ self.W.T
        return dx

    def step(self, lr):
        self.W -= lr * self.dW
        self.b -= lr * self.db

    def flat_params(self):
        return [("W", self.W, self.dW), ("b", self.b, self.db)]


class TwoLayerMLP:
    """
    A small MLP: Linear -> tanh -> Linear -> {sigmoid | linear}.
    Used for ScoreNet, GateNet, and GivingPolicyNet (all sigmoid-out,
    since each predicts a bounded probability/fraction).
    """

    def __init__(self, in_dim, hidden_dim, out_dim, seed, out_activation="sigmoid"):
        self.l1 = Linear(in_dim, hidden_dim, seed=seed * 7 + 1)
        self.l2 = Linear(hidden_dim, out_dim, seed=seed * 7 + 2)
        assert out_activation in ("sigmoid", "linear")
        self.out_activation = out_activation
        self._h_pre = None
        self._h = None
        self._y_pre = None
        self._y = None

    def forward(self, x):
        h_pre = self.l1.forward(x)
        h = tanh(h_pre)
        y_pre = self.l2.forward(h)
        y = sigmoid(y_pre) if self.out_activation == "sigmoid" else y_pre
        self._h_pre, self._h, self._y_pre, self._y = h_pre, h, y_pre, y
        return y

    def backward(self, dy):
        # dy is dLoss/dy (post-activation)
        if self.out_activation == "sigmoid":
            dy_pre = dy * dsigmoid_from_output(self._y)
        else:
            dy_pre = dy
        dh = self.l2.backward(dy_pre)
        dh_pre = dh * dtanh_from_output(self._h)
        dx = self.l1.backward(dh_pre)
        return dx

    def step(self, lr):
        self.l1.step(lr)
        self.l2.step(lr)

    def all_layers(self):
        return [self.l1, self.l2]


# =====================================================================
# SECTION 1 — the cast, the hierarchy, and the synthetic "inner
# nature" each node carries. Names are drawn from the poem for
# flavour; the tree below mirrors the poem's real chain of command
# (a sovereign, a crown-heiress, and the knights/companions who
# ultimately act on lateral oath rather than vertical order).
# =====================================================================

NODE_NAMES = [
    "Rostevan",       # 0 — King of Arabia, the sovereign root
    "Tinatin",        # 1 — his daughter and heir, "a lion's cub... a lion"
    "Avtandil",       # 2 — commander, swears brotherhood with Tariel
    "Tariel",         # 3 — the knight in the panther's skin
    "Nestan-Darejan", # 4 — the vanished heir of India, object of the quest
    "Phridon",        # 5 — King of Mulghazanzar, the third sworn brother
    "Asmat",          # 6 — Nestan's companion, keeper of the secret
    "Shermadin",      # 7 — Avtandil's steward, left in charge
    "Fatma",          # 8 — merchant's wife, eventual ally in the rescue
]
N = len(NODE_NAMES)
D = 8      # dimension of each node's inner hidden state h_i
K = 4      # dimension of the "true virtue" latent inside h_i (first K dims)
HIDDEN = 16

# Vertical hierarchy (parent -> children), a tree rooted at the
# sovereign. This is the chain of COMMAND, distinct from the sworn
# bonds that form later at run time.
PARENT = {
    1: 0,  # Tinatin reports to Rostevan
    2: 0,  # Avtandil is Rostevan's commander
    7: 2,  # Shermadin serves under Avtandil
    5: 0,  # Phridon is a foreign king, nominally a peer of the crown
    3: 5,  # Tariel is first found and vouched for by Phridon
    6: 3,  # Asmat serves Tariel
    4: 6,  # (topologically) Nestan is who Asmat serves-in-absence
    8: 5,  # Fatma is reached through Phridon's networks
}

def build_true_virtue_and_category(seed=RNG_SEED):
    """
    Each node gets:
      v_i in R^K  — the "true inner nature" (steadfastness / kinship
                    class). This is what recognition SHOULD track.
      c_i in {0,1} — an "outward vessel" category, uncorrelated with
                    v_i by construction. This stands in for anything
                    a fair recognition function must not key on — the
                    poem's own test case is gender ("a lion's cub is
                    a lion, be it male or female"); the mechanism
                    generalizes to any outward, kind-irrelevant label.
    Two nodes are "true kin" (should bond) iff cosine(v_i, v_j) is
    high, REGARDLESS of c_i, c_j.
    """
    r = np.random.default_rng(seed)
    # Three latent "houses" of virtue so some pairs are natural kin
    # and some are not — otherwise every pair would trivially bond.
    houses = r.normal(0.0, 1.0, size=(3, K))
    house_of = [0, 0, 1, 1, 1, 2, 1, 1, 2]   # hand-assigned per the plot:
    # Rostevan/Tinatin = house 0 (Arabian crown)
    # Avtandil/Tariel/Nestan/Asmat/Shermadin = house 1 (the sworn circle
    #   and those it is organized around/for)
    # Phridon/Fatma = house 2 (outside allies reached through the quest)
    V = np.zeros((N, K))
    for i in range(N):
        V[i] = houses[house_of[i]] + r.normal(0.0, 0.25, size=K)
    C = r.integers(0, 2, size=N).astype(float)  # outward category, 0/1
    return V, C, house_of


def make_hidden_states(V, C, seed=RNG_SEED):
    """
    Compose the full hidden state h_i = [virtue dims | vessel dims].
    Vessel dims encode the outward category plus independent noise —
    i.e. structurally present in h_i (an observer CAN read it) but,
    per the "lion's cub" claim, it must not drive the trust score.
    """
    r = np.random.default_rng(seed + 1)
    H = np.zeros((N, D))
    H[:, :K] = V
    vessel_noise = r.normal(0.0, 0.3, size=(N, D - K - 1))
    H[:, K] = C * 2.0 - 1.0             # category in {-1, +1}
    H[:, K + 1:] = vessel_noise
    return H


# =====================================================================
# SECTION 2 — ScoreNet: the Trial-of-Worth / Recognition network.
# Input: [h_i, h_j, |h_i - h_j|]  (3*D dims)
# Output: kinship / steadfast-trust probability in (0, 1)
# =====================================================================

class ScoreNet:
    def __init__(self, seed=1):
        self.mlp = TwoLayerMLP(3 * D, HIDDEN, 1, seed=seed, out_activation="sigmoid")

    @staticmethod
    def features(hi, hj):
        return np.concatenate([hi, hj, np.abs(hi - hj)], axis=-1)

    def forward(self, hi, hj):
        x = self.features(hi, hj)
        return self.mlp.forward(x)[..., 0]

    def loss_and_grad(self, HI, HJ, labels):
        """Binary cross-entropy over a batch. Returns (loss, dW..)."""
        X = self.features(HI, HJ)
        p = self.mlp.forward(X)  # (B,1)
        p = p[:, 0]
        eps = 1e-9
        loss = -np.mean(labels * np.log(p + eps) + (1 - labels) * np.log(1 - p + eps))
        dp = (p - labels) / (labels.shape[0] + eps)  # dL/dp for BCE+sigmoid-combined-simplified
        # NOTE: because forward already applied sigmoid, and BCE dL/dy_pre
        # for sigmoid output simplifies to (p - label)/B, we pass this
        # directly as "dy" into mlp.backward, which then internally
        # multiplies by dsigmoid_from_output(y) again. To avoid double
        # counting the sigmoid derivative we instead backprop through
        # a LINEAR view: feed dy_pre directly by temporarily bypassing.
        self.mlp._y = p  # ensure cached
        dy_pre = (p - labels) / labels.shape[0]
        dh = self.mlp.l2.backward(dy_pre[:, None])
        dh_pre = dh * dtanh_from_output(self.mlp._h)
        self.mlp.l1.backward(dh_pre)
        return loss

    def step(self, lr):
        self.mlp.step(lr)


# =====================================================================
# SECTION 3 — GateNet: the Oath-Override gate.
# Input: [||vertical_msg||, ||lateral_msg||, conflict, bond_flag, trust]
# Output: gate in (0,1); gate*lateral + (1-gate)*vertical
# =====================================================================

class GateNet:
    FEAT_DIM = 5

    def __init__(self, seed=2):
        self.mlp = TwoLayerMLP(self.FEAT_DIM, HIDDEN, 1, seed=seed, out_activation="sigmoid")

    @staticmethod
    def features(vert_msg, lat_msg, bond_flag, trust):
        conflict = np.linalg.norm(vert_msg - lat_msg, axis=-1, keepdims=True)
        vnorm = np.linalg.norm(vert_msg, axis=-1, keepdims=True)
        lnorm = np.linalg.norm(lat_msg, axis=-1, keepdims=True)
        b = np.asarray(bond_flag).reshape(-1, 1)
        t = np.asarray(trust).reshape(-1, 1)
        return np.concatenate([vnorm, lnorm, conflict, b, t], axis=-1)

    def forward(self, vert_msg, lat_msg, bond_flag, trust):
        x = self.features(vert_msg, lat_msg, bond_flag, trust)
        return self.mlp.forward(x)[:, 0]

    def loss_and_grad(self, VERT, LAT, BOND, TRUST, target_gate):
        X = self.features(VERT, LAT, BOND, TRUST)
        p = self.mlp.forward(X)[:, 0]
        eps = 1e-9
        loss = -np.mean(target_gate * np.log(p + eps) + (1 - target_gate) * np.log(1 - p + eps))
        dy_pre = (p - target_gate) / target_gate.shape[0]
        dh = self.mlp.l2.backward(dy_pre[:, None])
        dh_pre = dh * dtanh_from_output(self.mlp._h)
        self.mlp.l1.backward(dh_pre)
        return loss

    def step(self, lr):
        self.mlp.step(lr)


# =====================================================================
# SECTION 4 — CovenantCipher: the allegorical displacement.
#   Fixed encoder E: h -> s = h @ R + noise      (R is a fixed random
#   near-orthogonal matrix — "the invented geography of Arabia and
#   India" — deterministic given the seed, i.e. not a secret, but
#   also not informative unless inverted).
#   Trained decoder D: s -> h_hat = s @ D.        D is learned by
#   minimising reconstruction error ONLY on bonded pairs' own
#   surface forms — modelling that the "key" to the allegory belongs
#   to the sworn circle, not to any outside reader, who is left with
#   an untrained/generic decoder D0 (frozen, never fit to data).
# =====================================================================

class CovenantCipher:
    def __init__(self, seed=3):
        r = np.random.default_rng(seed)
        A = r.normal(0.0, 1.0, size=(D, D))
        Q, _ = np.linalg.qr(A)         # orthogonal displacement matrix
        self.R = Q                     # fixed encoder matrix (not secret)
        self.D = np.zeros((D, D))      # TRAINABLE decoder ("the key")
        self.D0 = r.normal(0.0, 1.0 / np.sqrt(D), size=(D, D))  # frozen "outsider" guess
        self._s_cache = None
        self.dD = np.zeros_like(self.D)

    def encode(self, H, noise_std=0.05, seed=None):
        r = np.random.default_rng(seed) if seed is not None else rng
        return H @ self.R + r.normal(0.0, noise_std, size=H.shape)

    def decode_trained(self, S):
        self._s_cache = S
        return S @ self.D

    def decode_outsider(self, S):
        return S @ self.D0

    def loss_and_grad(self, S_bonded, H_targets):
        pred = self.decode_trained(S_bonded)
        diff = pred - H_targets
        loss = np.mean(np.sum(diff ** 2, axis=-1))
        B = S_bonded.shape[0]
        dpred = 2.0 * diff / B
        self.dD = S_bonded.T @ dpred
        return loss

    def step(self, lr):
        self.D -= lr * self.dD


# =====================================================================
# SECTION 5 — GivingPolicyNet + ValueHeads
#   GivingPolicyNet: [w_i, neighbor_avg_wealth, trust] -> give fraction
#   ValueHeads: linear heads over h_i predicting a steadfast target
#   (low-variance, "heaven's kind" of love) and an appetite target
#   (high-variance, context-driven, "lechery").
# =====================================================================

class GivingPolicyNet:
    def __init__(self, seed=4):
        self.mlp = TwoLayerMLP(3, HIDDEN, 1, seed=seed, out_activation="sigmoid")

    def forward(self, w, neighbor_avg, trust):
        x = np.stack([w, neighbor_avg, trust], axis=-1)
        return self.mlp.forward(x)[:, 0]

    def loss_and_grad(self, W, NEIGH, TRUST, target_fraction):
        x = np.stack([W, NEIGH, TRUST], axis=-1)
        p = self.mlp.forward(x)[:, 0]
        diff = p - target_fraction
        loss = np.mean(diff ** 2)
        dy_pre = 2.0 * diff / target_fraction.shape[0] * dsigmoid_from_output(p)
        dh = self.mlp.l2.backward(dy_pre[:, None])
        dh_pre = dh * dtanh_from_output(self.mlp._h)
        self.mlp.l1.backward(dh_pre)
        return loss

    def step(self, lr):
        self.mlp.step(lr)


class ValueHeads:
    """Two independent linear heads sharing no parameters: steadfast
    (constancy channel) and appetite (context-driven channel)."""

    def __init__(self, seed=5):
        self.steadfast = Linear(D, 1, seed=seed * 3 + 1)
        self.appetite = Linear(D + 2, 1, seed=seed * 3 + 2)  # + 2-dim noisy "occasion" context

    def forward_steadfast(self, H):
        return self.steadfast.forward(H)[:, 0]

    def forward_appetite(self, H, ctx):
        x = np.concatenate([H, ctx], axis=-1)
        return self.appetite.forward(x)[:, 0]

    def loss_and_grad_steadfast(self, H, target):
        pred = self.forward_steadfast(H)
        diff = pred - target
        loss = np.mean(diff ** 2)
        d = (2.0 * diff / target.shape[0])[:, None]
        self.steadfast.backward(d)
        return loss

    def loss_and_grad_appetite(self, H, ctx, target):
        pred = self.forward_appetite(H, ctx)
        diff = pred - target
        loss = np.mean(diff ** 2)
        d = (2.0 * diff / target.shape[0])[:, None]
        x = np.concatenate([H, ctx], axis=-1)
        self.appetite._x = x
        self.appetite.backward(d)
        return loss

    def step(self, lr):
        self.steadfast.step(lr)
        self.appetite.step(lr)


# =====================================================================
# SECTION 6 — synthetic supervised data generators for each subnet,
# a combined total_loss(), and the MANDATORY finite-difference
# gradient check.
# =====================================================================

class TrainingData:
    def __init__(self, seed=RNG_SEED):
        self.V, self.C, self.house = build_true_virtue_and_category(seed)
        self.H = make_hidden_states(self.V, self.C, seed)
        self.r = np.random.default_rng(seed + 100)

    def score_batch(self, batch_size=64):
        i = self.r.integers(0, N, size=batch_size)
        j = self.r.integers(0, N, size=batch_size)
        HI, HJ = self.H[i], self.H[j]
        cos = np.sum(self.V[i] * self.V[j], axis=-1) / (
            np.linalg.norm(self.V[i], axis=-1) * np.linalg.norm(self.V[j], axis=-1) + 1e-9
        )
        labels = (cos > 0.5).astype(float)
        return HI, HJ, labels

    def gate_batch(self, batch_size=64, score_net: "ScoreNet" = None):
        i = self.r.integers(0, N, size=batch_size)
        j = self.r.integers(0, N, size=batch_size)
        vert = self.r.normal(0.0, 1.0, size=(batch_size, D))
        lat = self.r.normal(0.0, 1.0, size=(batch_size, D))
        trust = score_net.forward(self.H[i], self.H[j]) if score_net else self.r.uniform(0, 1, batch_size)
        bond = (trust > 0.5).astype(float)
        # target rule: favour lateral (gate->1) iff bonded AND trust high
        target = ((bond > 0.5) & (trust > 0.6)).astype(float)
        return vert, lat, bond, trust, target

    def cipher_batch(self, cipher: "CovenantCipher", batch_size=64):
        i = self.r.integers(0, N, size=batch_size)
        S = cipher.encode(self.H[i], seed=int(self.r.integers(0, 1 << 30)))
        return S, self.H[i]

    def giving_batch(self, batch_size=64, score_net: "ScoreNet" = None):
        i = self.r.integers(0, N, size=batch_size)
        j = self.r.integers(0, N, size=batch_size)
        w = self.r.uniform(0.2, 1.0, size=batch_size)
        neigh = self.r.uniform(0.0, 1.0, size=batch_size)
        trust = score_net.forward(self.H[i], self.H[j]) if score_net else self.r.uniform(0, 1, batch_size)
        surplus = (w > neigh).astype(float)
        bonded = (trust > 0.5).astype(float)
        target = 0.05 + 0.55 * surplus * bonded * trust  # high giving iff surplus AND bonded AND trusted
        return w, neigh, trust, target

    def steadfast_batch(self, batch_size=64):
        i = self.r.integers(0, N, size=batch_size)
        # steadfast target: fixed function of true virtue norm only —
        # constant across repeated draws for the SAME node (constancy)
        target = np.linalg.norm(self.V[i], axis=-1)
        return self.H[i], target

    def appetite_batch(self, batch_size=64):
        i = self.r.integers(0, N, size=batch_size)
        ctx = self.r.normal(0.0, 1.0, size=(batch_size, 2))  # a fresh "occasion" every draw
        target = np.linalg.norm(self.V[i], axis=-1) + 1.5 * ctx[:, 0] - 0.8 * ctx[:, 1]
        return self.H[i], ctx, target


def finite_difference_gradient_check(module, loss_fn, param_owner_layers, eps=1e-5, n_checks=6, tol=2e-4):
    """
    Generic mandatory gradient check. `loss_fn()` recomputes the
    module's loss from its *current cached inputs* (so it must be
    called immediately after a matching forward+backward pass).
    `param_owner_layers` is a list of Linear layers whose (W, b) will
    be spot-checked against central finite differences.
    Returns (max_relative_error, n_checked).
    """
    max_rel_err = 0.0
    checked = 0
    for layer in param_owner_layers:
        for name, param, grad in layer.flat_params():
            flat = param.reshape(-1)
            gflat = grad.reshape(-1)
            idxs = rng.choice(flat.shape[0], size=min(n_checks, flat.shape[0]), replace=False)
            for idx in idxs:
                orig = flat[idx]
                flat[idx] = orig + eps
                loss_plus = loss_fn()
                flat[idx] = orig - eps
                loss_minus = loss_fn()
                flat[idx] = orig
                numeric = (loss_plus - loss_minus) / (2 * eps)
                analytic = gflat[idx]
                denom = max(abs(numeric), abs(analytic), 1e-8)
                rel_err = abs(numeric - analytic) / denom
                max_rel_err = max(max_rel_err, rel_err)
                checked += 1
    return max_rel_err, checked


# =====================================================================
# SECTION 7 — the CovenantNetwork orchestrator: training loop over all
# five subnets, plus the inference-time dynamical system (recognition
# message passing + wealth circulation) used by the self-tests.
# =====================================================================

class CovenantNetwork:
    def __init__(self, seed=RNG_SEED):
        self.data = TrainingData(seed)
        self.score = ScoreNet(seed=1)
        self.gate = GateNet(seed=2)
        self.cipher = CovenantCipher(seed=3)
        self.giving = GivingPolicyNet(seed=4)
        self.values = ValueHeads(seed=5)
        self.loss_history = []

    # ---- one training epoch across all five components ----
    def train_epoch(self, lr=0.05, batch_size=64):
        d = self.data
        losses = {}

        HI, HJ, labels = d.score_batch(batch_size)
        losses["score"] = self.score.loss_and_grad(HI, HJ, labels)
        self.score.step(lr)

        VERT, LAT, BOND, TRUST, TARGET = d.gate_batch(batch_size, score_net=self.score)
        losses["gate"] = self.gate.loss_and_grad(VERT, LAT, BOND, TRUST, TARGET)
        self.gate.step(lr)

        S, Htarget = d.cipher_batch(self.cipher, batch_size)
        losses["cipher"] = self.cipher.loss_and_grad(S, Htarget)
        self.cipher.step(lr * 0.5)

        W, NEIGH, TR, TG = d.giving_batch(batch_size, score_net=self.score)
        losses["giving"] = self.giving.loss_and_grad(W, NEIGH, TR, TG)
        self.giving.step(lr)

        Hs, target_s = d.steadfast_batch(batch_size)
        losses["steadfast"] = self.values.loss_and_grad_steadfast(Hs, target_s)
        Ha, ctx, target_a = d.appetite_batch(batch_size)
        losses["appetite"] = self.values.loss_and_grad_appetite(Ha, ctx, target_a)
        self.values.step(lr)

        total = sum(losses.values())
        self.loss_history.append(total)
        return total, losses

    def train(self, epochs=400, lr=0.05, batch_size=64, verbose=True):
        for ep in range(epochs):
            total, losses = self.train_epoch(lr, batch_size)
            if verbose and (ep % 50 == 0 or ep == epochs - 1):
                print(f"  epoch {ep:4d}  total={total:.4f}  "
                      f"score={losses['score']:.4f} gate={losses['gate']:.4f} "
                      f"cipher={losses['cipher']:.4f} giving={losses['giving']:.4f} "
                      f"steadfast={losses['steadfast']:.4f} appetite={losses['appetite']:.4f}")
        return self.loss_history

    # ---- inference-time dynamics ----
    def bond_matrix(self, threshold=0.6):
        """Run the trial-of-worth score over every pair once bonds are
        'proposed'; a bond is permanent once formed (max over trials),
        mirroring the poem's irrevocable dzmobisa oath."""
        H = self.data.H
        bonds = np.zeros((N, N), dtype=bool)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                p = float(self.score.forward(H[i:i+1], H[j:j+1])[0])
                if p > threshold:
                    bonds[i, j] = True
                    bonds[j, i] = True
        return bonds

    def recognition_step(self, bonds):
        """
        One round of the Recognition Layer: every non-root node
        receives a vertical message from its parent and (if bonded)
        a lateral message from its sworn peers, and the GateNet
        decides how much the lateral channel overrides the vertical
        one. Returns the updated hidden states (residual update).
        """
        H = self.data.H.copy()
        H_new = H.copy()
        for child, parent in PARENT.items():
            vert_msg = H[parent:parent+1]
            peers = [k for k in range(N) if bonds[child, k]]
            if peers:
                lat_msg = H[peers].mean(axis=0, keepdims=True)
                bond_flag = np.array([1.0])
            else:
                lat_msg = H[child:child+1]  # nothing to disclose -> self
                bond_flag = np.array([0.0])
            trust = self.score.forward(H[child:child+1], lat_msg)
            gate = self.gate.forward(vert_msg, lat_msg, bond_flag, trust)
            blended = gate[:, None] * lat_msg + (1.0 - gate[:, None]) * vert_msg
            H_new[child] = 0.7 * H[child] + 0.3 * blended[0]
        return H_new, self._last_gates(H, bonds)

    def _last_gates(self, H, bonds):
        gates = {}
        for child, parent in PARENT.items():
            vert_msg = H[parent:parent+1]
            peers = [k for k in range(N) if bonds[child, k]]
            if peers:
                lat_msg = H[peers].mean(axis=0, keepdims=True)
                bond_flag = np.array([1.0])
            else:
                lat_msg = H[child:child+1]
                bond_flag = np.array([0.0])
            trust = self.score.forward(H[child:child+1], lat_msg)
            gate = self.gate.forward(vert_msg, lat_msg, bond_flag, trust)
            gates[NODE_NAMES[child]] = float(gate[0])
        return gates

    def wealth_circulate(self, steps=20, hoarders=None, bonds=None, decay=0.90, bonus=1.12,
                          use_policy=True):
        """
        Deterministic wealth dynamical system. `hoarders` is a set of
        node indices whose giving fraction is forced to 0 regardless
        of policy (used by the hoarding self-test). Wealth kept
        decays by `decay` each step; wealth transmitted across a
        sworn bond is multiplied by `bonus` on arrival (modelling
        "what is given makes us richer").
        """
        hoarders = hoarders or set()
        if bonds is None:
            bonds = self.bond_matrix()
        peers_of = [[k for k in range(N) if bonds[i, k]] for i in range(N)]
        w = np.full(N, 0.5)
        history = [w.copy()]
        for _ in range(steps):
            # Pass 1: decide every node's give/keep split from the
            # CURRENT round's wealth `w` only (no partial updates yet,
            # so the order nodes are visited in cannot corrupt the
            # accounting -- this was a real bug caught during testing:
            # writing `w_next[i] = ...` after another node had already
            # deposited a gift into w_next[i] silently erased it).
            give = np.zeros(N)
            for i in range(N):
                peers = peers_of[i]
                if not peers or i in hoarders:
                    continue
                neigh_avg = float(np.mean(w[peers]))
                trust_ij = float(self.score.forward(
                    self.data.H[i:i+1], self.data.H[peers].mean(axis=0, keepdims=True))[0])
                frac = float(self.giving.forward(np.array([w[i]]), np.array([neigh_avg]),
                                                  np.array([trust_ij]))[0]) if use_policy else 0.3
                give[i] = frac * w[i]
            keep = w - give
            # Pass 2: everyone's kept wealth decays; every gift lands
            # (bonus-multiplied, split evenly) on the giver's bonded
            # peers. Additions only -- never an overwrite.
            w_next = keep * decay
            for i in range(N):
                peers = peers_of[i]
                if not peers or give[i] <= 0:
                    continue
                per_peer = (give[i] * bonus) / len(peers)
                for k in peers:
                    w_next[k] += per_peer
            w = w_next
            history.append(w.copy())
        return np.array(history)


# =====================================================================
# SECTION 8 — self-tests. These operationalize the five mind-specific
# claims listed at the top of the file.
# =====================================================================

def self_test_gradient_check(net: CovenantNetwork):
    print("\n[TEST 1] Finite-difference gradient check (mandatory)")
    d = net.data
    results = {}

    HI, HJ, labels = d.score_batch(48)
    net.score.loss_and_grad(HI, HJ, labels)
    err, n = finite_difference_gradient_check(
        net.score, lambda: net.score.loss_and_grad(HI, HJ, labels), net.score.mlp.all_layers())
    results["ScoreNet"] = err

    VERT, LAT, BOND, TRUST, TARGET = d.gate_batch(48, score_net=net.score)
    net.gate.loss_and_grad(VERT, LAT, BOND, TRUST, TARGET)
    err, n = finite_difference_gradient_check(
        net.gate, lambda: net.gate.loss_and_grad(VERT, LAT, BOND, TRUST, TARGET), net.gate.mlp.all_layers())
    results["GateNet"] = err

    S, Htarget = d.cipher_batch(net.cipher, 48)
    net.cipher.loss_and_grad(S, Htarget)
    class _CipherLayerView:
        def flat_params(self_inner):
            return [("D", net.cipher.D, net.cipher.dD)]
    err, n = finite_difference_gradient_check(
        net.cipher, lambda: net.cipher.loss_and_grad(S, Htarget), [_CipherLayerView()])
    results["CovenantCipher"] = err

    W, NEIGH, TR, TG = d.giving_batch(48, score_net=net.score)
    net.giving.loss_and_grad(W, NEIGH, TR, TG)
    err, n = finite_difference_gradient_check(
        net.giving, lambda: net.giving.loss_and_grad(W, NEIGH, TR, TG), net.giving.mlp.all_layers())
    results["GivingPolicyNet"] = err

    Hs, target_s = d.steadfast_batch(48)
    net.values.loss_and_grad_steadfast(Hs, target_s)
    err, n = finite_difference_gradient_check(
        net.values, lambda: net.values.loss_and_grad_steadfast(Hs, target_s), [net.values.steadfast])
    results["ValueHeads.steadfast"] = err

    Ha, ctx, target_a = d.appetite_batch(48)
    net.values.loss_and_grad_appetite(Ha, ctx, target_a)
    err, n = finite_difference_gradient_check(
        net.values, lambda: net.values.loss_and_grad_appetite(Ha, ctx, target_a), [net.values.appetite])
    results["ValueHeads.appetite"] = err

    ok = True
    for name, err in results.items():
        passed = err < 5e-3
        ok &= passed
        print(f"    {name:24s} max relative error = {err:.3e}  [{'PASS' if passed else 'FAIL'}]")
    print(f"  -> gradient check overall: {'PASS' if ok else 'FAIL'}")
    return ok, results


def self_test_allegorical_secrecy(net: CovenantNetwork):
    print("\n[TEST 2] Allegorical secrecy — bonded decoder vs outsider decoder")
    H = net.data.H
    S = net.cipher.encode(H, seed=999)
    trained_pred = net.cipher.decode_trained(S)
    outsider_pred = net.cipher.decode_outsider(S)
    err_trained = float(np.mean(np.linalg.norm(trained_pred - H, axis=-1)))
    err_outsider = float(np.mean(np.linalg.norm(outsider_pred - H, axis=-1)))
    ratio = err_outsider / max(err_trained, 1e-9)
    print(f"    mean reconstruction error, sworn-trained key : {err_trained:.4f}")
    print(f"    mean reconstruction error, outsider's guess  : {err_outsider:.4f}")
    print(f"    outsider/trained error ratio                 : {ratio:.2f}x")
    passed = ratio > 3.0
    print(f"  -> the allegory reads cleanly inside the sworn circle and stays "
          f"distorted outside it: {'PASS' if passed else 'FAIL'}")
    return passed, {"err_trained": err_trained, "err_outsider": err_outsider, "ratio": ratio}


def self_test_circulation_vs_hoarding(net: CovenantNetwork):
    print("\n[TEST 3] Circulation vs. hoarding — 'what is given multiplies, what is hoarded is lost'")
    bonds = net.bond_matrix()
    hist_generous = net.wealth_circulate(steps=25, hoarders=set(), bonds=bonds)
    hist_hoarders = net.wealth_circulate(steps=25, hoarders=set(range(N)), bonds=bonds)
    final_generous = float(hist_generous[-1].sum())
    final_hoarders = float(hist_hoarders[-1].sum())
    print(f"    total network wealth after 25 rounds, generous policy : {final_generous:.3f}")
    print(f"    total network wealth after 25 rounds, all-hoard policy: {final_hoarders:.3f}")
    passed = final_generous > final_hoarders
    print(f"  -> circulating wealth compounds past hoarding it: {'PASS' if passed else 'FAIL'}")
    return passed, {"final_generous": final_generous, "final_hoarders": final_hoarders}


def self_test_oath_override(net: CovenantNetwork):
    print("\n[TEST 4] Oath-override gate — 'Avtandil defies Rostevan'")
    # Bonded, high-trust scenario (Avtandil/Tariel after the oath):
    # a strongly conflicting vertical order should still be
    # outranked by the lateral (sworn) signal.
    vert = np.array([[5.0] + [0.0] * (D - 1)])   # a loud, specific royal order
    lat = np.array([[-5.0] + [0.0] * (D - 1)])   # the sworn brother's opposite request
    bond_true = np.array([1.0])
    trust_true = np.array([0.95])
    gate_bonded = float(net.gate.forward(vert, lat, bond_true, trust_true)[0])

    # Same conflict, but NO bond (a stranger making the same request):
    bond_false = np.array([0.0])
    trust_false = np.array([0.10])
    gate_stranger = float(net.gate.forward(vert, lat, bond_false, trust_false)[0])

    print(f"    gate value, sworn brother (bonded, trust=0.95): {gate_bonded:.3f}  "
          f"({'favours the oath' if gate_bonded > 0.5 else 'obeys the crown'})")
    print(f"    gate value, stranger (unbonded, trust=0.10)   : {gate_stranger:.3f}  "
          f"({'favours the oath' if gate_stranger > 0.5 else 'obeys the crown'})")
    passed = (gate_bonded > 0.5) and (gate_stranger < 0.5)
    print(f"  -> the sworn bond, and only the sworn bond, outranks the chain "
          f"of command: {'PASS' if passed else 'FAIL'}")
    return passed, {"gate_bonded": gate_bonded, "gate_stranger": gate_stranger}


def self_test_vessel_independence(net: CovenantNetwork):
    print("\n[TEST 5] Vessel independence — 'a lion's cub is a lion, be it male or female'")
    d = net.data
    # Pick two nodes from the SAME virtue house (should score as kin)
    house = d.house
    same_house_pairs = [(i, j) for i in range(N) for j in range(N)
                         if i < j and house[i] == house[j]]
    diff_house_pairs = [(i, j) for i in range(N) for j in range(N)
                         if i < j and house[i] != house[j]]
    i, j = same_house_pairs[0]

    h_i, h_j = d.H[i:i+1].copy(), d.H[j:j+1].copy()
    base_score = float(net.score.forward(h_i, h_j)[0])

    # Flip ONLY the outward "vessel" dimension (index K), leave the
    # true virtue dims (0..K-1) untouched.
    h_i_flipped = h_i.copy()
    h_i_flipped[0, K] *= -1.0
    flipped_score = float(net.score.forward(h_i_flipped, h_j)[0])

    # Contrast: flip a TRUE VIRTUE dimension instead -> score should move.
    h_i_virtue_altered = h_i.copy()
    h_i_virtue_altered[0, 0] *= -1.0
    virtue_altered_score = float(net.score.forward(h_i_virtue_altered, h_j)[0])

    vessel_delta = abs(flipped_score - base_score)
    virtue_delta = abs(virtue_altered_score - base_score)
    print(f"    base kinship score                      : {base_score:.4f}")
    print(f"    score after flipping OUTWARD category    : {flipped_score:.4f}  "
          f"(|delta| = {vessel_delta:.4f})")
    print(f"    score after altering TRUE VIRTUE instead : {virtue_altered_score:.4f}  "
          f"(|delta| = {virtue_delta:.4f})")
    passed = vessel_delta < 0.5 * virtue_delta + 1e-6 and vessel_delta < 0.15
    print(f"  -> recognition tracks inner kind far more than outward vessel: "
          f"{'PASS' if passed else 'FAIL'}")
    return passed, {"vessel_delta": vessel_delta, "virtue_delta": virtue_delta}


def self_test_loss_decreases(loss_history):
    print("\n[TEST 6] Training loss decreases over the run")
    start = float(np.mean(loss_history[:10]))
    end = float(np.mean(loss_history[-10:]))
    passed = end < start
    print(f"    mean total loss, first 10 epochs : {start:.4f}")
    print(f"    mean total loss, last 10 epochs  : {end:.4f}")
    print(f"  -> {'PASS' if passed else 'FAIL'}")
    return passed, {"start": start, "end": end}


# =====================================================================
# SECTION 9 — main(): train, verify, run the dynamical system once,
# and print a full report.
# =====================================================================

def main():
    print("=" * 78)
    print("THE SWORN-COVENANT RECOGNITION NETWORK")
    print("A computational architecture inspired by Shota Rustaveli's")
    print('"The Knight in the Panther\'s Skin" (Georgia, c. 1180s-1200s)')
    print("=" * 78)

    net = CovenantNetwork(seed=RNG_SEED)

    print("\n[TRAINING] 400 epochs across five jointly-trained sub-networks")
    print("-" * 78)
    loss_history = net.train(epochs=400, lr=0.05, batch_size=64, verbose=True)

    print("\n" + "=" * 78)
    print("VERIFICATION")
    print("=" * 78)

    results = {}
    ok1, r1 = self_test_gradient_check(net)
    ok2, r2 = self_test_allegorical_secrecy(net)
    ok3, r3 = self_test_circulation_vs_hoarding(net)
    ok4, r4 = self_test_oath_override(net)
    ok5, r5 = self_test_vessel_independence(net)
    ok6, r6 = self_test_loss_decreases(loss_history)

    all_pass = all([ok1, ok2, ok3, ok4, ok5, ok6])

    print("\n" + "=" * 78)
    print("RUNNING THE FULL RECOGNITION + CIRCULATION DYNAMICAL SYSTEM")
    print("=" * 78)
    bonds = net.bond_matrix()
    print("\nSworn bonds formed (trial-of-worth score > 0.6):")
    any_bond = False
    for i in range(N):
        for j in range(i + 1, N):
            if bonds[i, j]:
                any_bond = True
                print(f"    {NODE_NAMES[i]:16s} <-> {NODE_NAMES[j]:16s}")
    if not any_bond:
        print("    (none crossed threshold this run)")

    H_new, gates = net.recognition_step(bonds)
    print("\nOath-override gate reading per node this round "
          "(>0.5 = the sworn bond outranks the vertical order):")
    for name, g in gates.items():
        arrow = "OATH" if g > 0.5 else "CROWN"
        print(f"    {name:16s} gate={g:.3f}  -> obeys {arrow}")

    print("\n" + "=" * 78)
    print(f"OVERALL RESULT: {'ALL TESTS PASS' if all_pass else 'SOME TESTS FAILED'}")
    print("=" * 78)

    report = {
        "gradient_check_max_rel_err": {k: v for k, v in r1.items()},
        "allegorical_secrecy": r2,
        "circulation_vs_hoarding": r3,
        "oath_override": r4,
        "vessel_independence": r5,
        "loss_decrease": r6,
        "all_tests_pass": bool(all_pass),
        "final_total_loss": float(loss_history[-1]),
    }
    return report


if __name__ == "__main__":
    report = main()
    print("\nJSON SUMMARY:")
    print(json.dumps(report, indent=2))
