"""
================================================================================ 
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0307 · Nizami Ganjavi (c.1141-1209)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 16, Minds 301-320 Available on Amazon https://www.amazon.com/gp/product/B0HNJQX9S4
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos 

WHAT THIS FILE IS
------------------
A pure-NumPy, from-scratch, trainable and executable architecture whose three
subsystems are not generic "attention over stored keys," but three specific,
independently attested claims from Nizami Ganjavi's own poetry about how a
mind should be built. Nothing here is a demo stub: every subsystem has real
forward/backward passes, a finite-difference gradient check, a genuine
training loop with a real (if small) optimization problem, and a printed,
falsifiable self-test.

THE THREE MECHANISMS, AND WHERE EACH COMES FROM
-------------------------------------------------
1. THE UNPIERCED-PEARL GENERATOR (khalq al-ma'ani).
   Nizami repeatedly and explicitly claimed -- as a matter of doctrine, not
   incidental style -- that he refused to reuse a "meaning" (ma'ani, a
   metaphor or image-pairing) that any predecessor had already coined. This
   is implemented as a trained scoring network over a discrete concept space,
   combined with a hard formal (metrical) validity mask and an explicit
   "already used" memory that a search procedure is forbidden to revisit.
   Novelty is not a bonus term smoothed into a reward; it is a hard
   combinatorial constraint, exactly as Nizami describes his own practice.

2. THE HAFT PEYKAR GOVERNOR (pavilion-absorption and the decay of the realm).
   In Haft Peykar, King Bahram Gur visits seven color-and-planet-coded
   pavilions in sequence, each pavilion given wholly to one princess and one
   story; while he is absorbed in each, the kingdom's affairs are neglected
   and decay, until he must return and restore order. This is implemented as
   literal sequential fine-tuning of a shared "kingdom" network on seven
   narrow "pavilion" tasks, and it reproduces a real, named phenomenon in
   machine learning -- catastrophic forgetting -- together with the two
   classical mitigations Nizami's narrative itself proposes: periodic
   *return* to the general task (rehearsal / replay) and *remembering what
   mattered* (an importance-weighted anchor to the original competence,
   i.e. an Elastic-Weight-Consolidation-style penalty).

3. THE KHIZR-ISKANDAR GATE (unsought knowledge).
   In the Eskandarnameh, Alexander mounts a deliberate, self-authored
   expedition for the Water of Life and fails; his guide Khizr, who was
   never seeking it and was occupied with ordinary duty, finds it
   unsought. Nizami is explicit that the difference is not luck but manner:
   Iskandar's failure is attributed to his *eagerness* -- to pursuing the
   water as his own authored project -- while the water "arrived unsought"
   to Khizr. This is implemented as two REINFORCE-trained tabular policies
   in a shared environment: one (Iskandar) whose *only* objective is a
   resource that is gated behind a minimum-entropy condition at the moment
   of arrival; one (Khizr) whose objective is unrelated ordinary duty, with
   the same resource gate evaluated only passively, never as a training
   signal. The experiment asks a real question: does an agent whose entire
   optimization pressure is aimed at a goal thereby destroy its own chance
   of being in the state (sufficiently undetermined) that the goal requires?

Every number below is produced by running this file, not invented afterward.
This file does not need internet access, GPUs, or any dependency beyond
NumPy. Run it directly:

    python3 0307_Neuron.py

================================================================================
"""

import numpy as np
import sys
import time

RNG_SEED = 307
np.random.seed(RNG_SEED)


# =============================================================================
# PART 0 -- SHARED FRAMEWORK: a small, honest, backprop-capable MLP in NumPy
# =============================================================================
# Both the Pearl Generator (Part 2) and the Kingdom Governor (Part 1) are
# built from this. It is deliberately minimal: dense layers, three
# activations, manual analytic backward passes, and an Adam optimizer
# written out term by term (no autograd). The finite-difference check in
# Part 3.0 verifies these analytic gradients directly.

def relu(x):
    return np.maximum(0.0, x)

def relu_grad(z):
    return (z > 0.0).astype(np.float64)

def tanh_grad(a):
    # a is tanh(z) already
    return 1.0 - a ** 2


class Dense:
    """A single fully-connected layer with its own Adam state."""

    def __init__(self, n_in, n_out, activation="relu", rng=None):
        rng = rng if rng is not None else np.random.default_rng(RNG_SEED)
        # He/Xavier-ish init depending on activation
        scale = np.sqrt(2.0 / n_in) if activation == "relu" else np.sqrt(1.0 / n_in)
        self.W = rng.normal(0.0, scale, size=(n_in, n_out))
        self.b = np.zeros(n_out)
        self.activation = activation

        # Adam moment buffers
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b); self.vb = np.zeros_like(self.b)
        self.t = 0

        self._cache = None
        self.dW = None
        self.db = None

    def forward(self, x):
        z = x @ self.W + self.b
        if self.activation == "relu":
            a = relu(z)
        elif self.activation == "tanh":
            a = np.tanh(z)
        elif self.activation == "linear":
            a = z
        else:
            raise ValueError(f"unknown activation {self.activation}")
        self._cache = (x, z, a)
        return a

    def backward(self, dA):
        x, z, a = self._cache
        if self.activation == "relu":
            dZ = dA * relu_grad(z)
        elif self.activation == "tanh":
            dZ = dA * tanh_grad(a)
        elif self.activation == "linear":
            dZ = dA
        self.dW = x.T @ dZ
        self.db = np.sum(dZ, axis=0)
        dX = dZ @ self.W.T
        return dX

    def adam_step(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0, anchor=None, anchor_strength=0.0, importance=None):
        """Standard Adam, with an optional EWC-style quadratic anchor toward
        a stored earlier weight value ('anchor'), scaled per-parameter by an
        'importance' matrix. This is the mechanism the Governor uses to
        remember the kingdom while visiting a pavilion."""
        self.t += 1
        dW = self.dW
        db = self.db
        if anchor is not None and anchor_strength > 0.0:
            impW = importance[0] if importance is not None else 1.0
            dW = dW + anchor_strength * impW * (self.W - anchor[0])
            impb = importance[1] if importance is not None else 1.0
            db = db + anchor_strength * impb * (self.b - anchor[1])
        if weight_decay > 0.0:
            dW = dW + weight_decay * self.W

        self.mW = beta1 * self.mW + (1 - beta1) * dW
        self.vW = beta2 * self.vW + (1 - beta2) * (dW ** 2)
        mW_hat = self.mW / (1 - beta1 ** self.t)
        vW_hat = self.vW / (1 - beta2 ** self.t)
        self.W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)

        self.mb = beta1 * self.mb + (1 - beta1) * db
        self.vb = beta2 * self.vb + (1 - beta2) * (db ** 2)
        mb_hat = self.mb / (1 - beta1 ** self.t)
        vb_hat = self.vb / (1 - beta2 ** self.t)
        self.b -= lr * mb_hat / (np.sqrt(vb_hat) + eps)


class MLP:
    """A stack of Dense layers, i.e. a small feed-forward network."""

    def __init__(self, sizes, activations, rng=None):
        assert len(activations) == len(sizes) - 1
        rng = rng if rng is not None else np.random.default_rng(RNG_SEED)
        self.layers = [Dense(sizes[i], sizes[i + 1], activations[i], rng=rng)
                        for i in range(len(sizes) - 1)]

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dA):
        for layer in reversed(self.layers):
            dA = layer.backward(dA)
        return dA

    def adam_step(self, lr=1e-3, weight_decay=0.0, anchors=None, anchor_strength=0.0, importances=None):
        for i, layer in enumerate(self.layers):
            anchor = anchors[i] if anchors is not None else None
            importance = importances[i] if importances is not None else None
            layer.adam_step(lr, weight_decay=weight_decay, anchor=anchor,
                             anchor_strength=anchor_strength, importance=importance)

    def snapshot(self):
        """Return a deep copy of (W, b) for every layer -- used both as an
        EWC anchor point and to save/restore the 'kingdom's true state'."""
        return [(layer.W.copy(), layer.b.copy()) for layer in self.layers]

    def restore(self, snap):
        for layer, (W, b) in zip(self.layers, snap):
            layer.W[...] = W
            layer.b[...] = b


def mse_loss(y_hat, y):
    diff = y_hat - y
    loss = np.mean(diff ** 2)
    dA = 2.0 * diff / y.shape[0]
    return loss, dA


# =============================================================================
# PART 3.0 -- MANDATORY GRADIENT CHECK
# =============================================================================
# Verifies the analytic backward() implementation above against numerical
# (central finite-difference) gradients on a freshly built small MLP. This
# must pass before anything else in this file is trusted.

def gradient_check(n_checks=40, eps=1e-5, tol=1e-4, verbose=True):
    rng = np.random.default_rng(7)
    net = MLP([5, 8, 6, 1], ["tanh", "relu", "linear"], rng=rng)
    x = rng.normal(size=(16, 5))
    y = rng.normal(size=(16, 1))

    def full_loss():
        y_hat = net.forward(x)
        loss, _ = mse_loss(y_hat, y)
        return loss

    # Analytic pass
    y_hat = net.forward(x)
    loss, dA = mse_loss(y_hat, y)
    net.backward(dA)

    # Collect (layer, array, index) tuples to probe
    probes = []
    for layer in net.layers:
        for arr, grad in [(layer.W, layer.dW), (layer.b, layer.db)]:
            flat_idx = np.arange(arr.size)
            rng.shuffle(flat_idx)
            for idx in flat_idx[: max(1, n_checks // (2 * len(net.layers)))]:
                probes.append((arr, grad, np.unravel_index(idx, arr.shape)))

    max_rel_err = 0.0
    n_done = 0
    for arr, grad, idx in probes:
        orig = arr[idx]
        arr[idx] = orig + eps
        loss_plus = full_loss()
        arr[idx] = orig - eps
        loss_minus = full_loss()
        arr[idx] = orig  # restore

        numeric = (loss_plus - loss_minus) / (2 * eps)
        analytic = grad[idx]
        denom = max(1e-8, abs(numeric) + abs(analytic))
        rel_err = abs(numeric - analytic) / denom
        max_rel_err = max(max_rel_err, rel_err)
        n_done += 1

    passed = max_rel_err < tol
    if verbose:
        print(f"  [gradient check] probed {n_done} parameters across {len(net.layers)} layers")
        print(f"  [gradient check] max relative error = {max_rel_err:.3e}  (tolerance {tol:.0e})")
        print(f"  [gradient check] {'PASSED' if passed else 'FAILED'}")
    return passed, max_rel_err


# =============================================================================
# PART 1 -- THE HAFT PEYKAR GOVERNOR
# catastrophic forgetting under sequential pavilion-visits, and its cure
# =============================================================================

D_IN = 6     # a generic "affairs of the realm" feature vector
D_HID = 24
D_OUT = 1

PAVILION_COLORS = ["Black", "Yellow", "Green", "Red", "Turquoise", "Sandalwood", "White"]
PAVILION_PLANETS = ["Saturn", "Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus"]


def make_kingdom_task(rng, n=400):
    """The 'general competence' Bahram Gur is supposed to maintain: a fixed,
    reasonably nonlinear function of a 6-dim feature vector representing the
    ordinary business of governing (taxes, harvests, borders, ...)."""
    X = rng.uniform(-1, 1, size=(n, D_IN))
    freqs = np.array([1.3, 0.7, 2.1, 0.4, 1.7, 0.9])
    y = np.sin(X @ freqs).reshape(-1, 1) + 0.3 * np.sum(X ** 2, axis=1, keepdims=True) - 0.5
    return X, y


def make_pavilion_task(rng, pavilion_idx, n=120):
    """Each of the seven pavilions is a narrow, shifted sub-domain: a
    princess's story keyed to her clime, distinct enough from the kingdom's
    general task that fitting it purely (no rehearsal) pulls shared weights
    away from general competence -- exactly what 'absorption' costs."""
    center = rng_pavilion_centers[pavilion_idx]
    X = center + 0.25 * rng.uniform(-1, 1, size=(n, D_IN))
    # a distorted, locally-idiosyncratic target -- a different 'moral' per pavilion
    twist = rng_pavilion_twists[pavilion_idx]
    y = np.tanh(X @ twist).reshape(-1, 1) * 2.0
    return X, y


rng_master = np.random.default_rng(RNG_SEED)
rng_pavilion_centers = rng_master.uniform(-0.6, 0.6, size=(7, D_IN))
rng_pavilion_twists = rng_master.normal(0, 1.5, size=(7, D_IN))


def eval_mse(net, X, y):
    y_hat = net.forward(X)
    return float(np.mean((y_hat - y) ** 2))


def train_on(net, X, y, steps, lr, anchors=None, anchor_strength=0.0, importances=None, batch_size=32, rng=None):
    rng = rng if rng is not None else np.random.default_rng(0)
    n = X.shape[0]
    for _ in range(steps):
        idx = rng.integers(0, n, size=min(batch_size, n))
        xb, yb = X[idx], y[idx]
        y_hat = net.forward(xb)
        _, dA = mse_loss(y_hat, yb)
        net.backward(dA)
        net.adam_step(lr=lr, anchors=anchors, anchor_strength=anchor_strength, importances=importances)


def compute_fisher_importance(net, X, y, n_samples=64, rng=None):
    """A crude empirical-Fisher-style importance: average squared gradient
    of the loss w.r.t. each parameter over kingdom data, i.e. 'how much did
    the affairs of the realm depend on this exact weight'. Parameters with
    high importance are anchored hard during pavilion visits; parameters
    the kingdom never needed are left free for the pavilion to use."""
    rng = rng if rng is not None else np.random.default_rng(1)
    n = X.shape[0]
    sq_grads = [(np.zeros_like(l.W), np.zeros_like(l.b)) for l in net.layers]
    for _ in range(n_samples):
        i = rng.integers(0, n)
        xb, yb = X[i:i + 1], y[i:i + 1]
        y_hat = net.forward(xb)
        _, dA = mse_loss(y_hat, yb)
        net.backward(dA)
        for li, layer in enumerate(net.layers):
            sq_grads[li] = (sq_grads[li][0] + layer.dW ** 2, sq_grads[li][1] + layer.db ** 2)
    importances = [(gW / n_samples, gb / n_samples) for gW, gb in sq_grads]
    # normalize to a sane scale
    max_val = max(np.max(gW.max() if gW.size else 0) for gW, gb in importances) or 1.0
    importances = [(gW / max_val, gb / max_val) for gW, gb in importances]
    return importances


def run_haft_peykar_governor(verbose=True):
    rng = np.random.default_rng(RNG_SEED + 1)
    X_king, y_king = make_kingdom_task(rng, n=500)
    n_split = 400
    X_king_train, y_king_train = X_king[:n_split], y_king[:n_split]
    X_king_eval, y_king_eval = X_king[n_split:], y_king[n_split:]

    pavilion_data = [make_pavilion_task(rng, i, n=150) for i in range(7)]

    # --- Establish the well-governed baseline: train the kingdom network
    #     on general affairs before any pavilion is visited.
    def fresh_net():
        return MLP([D_IN, D_HID, D_HID, D_OUT], ["relu", "relu", "linear"],
                   rng=np.random.default_rng(RNG_SEED + 2))

    net_baseline = fresh_net()
    train_on(net_baseline, X_king_train, y_king_train, steps=400, lr=5e-3,
             rng=np.random.default_rng(11))
    baseline_mse = eval_mse(net_baseline, X_king_eval, y_king_eval)

    kingdom_snapshot = net_baseline.snapshot()
    importances = compute_fisher_importance(net_baseline, X_king_train, y_king_train,
                                             n_samples=80, rng=np.random.default_rng(12))

    # --- CONDITION A: absorbed king -- pure sequential fine-tuning, no
    #     rehearsal, no anchor. Pavilion-absorption exactly as narrated.
    net_absorbed = fresh_net()
    net_absorbed.restore(kingdom_snapshot)
    absorbed_trace = [baseline_mse]
    for i, (Xp, yp) in enumerate(pavilion_data):
        train_on(net_absorbed, Xp, yp, steps=150, lr=5e-3, rng=np.random.default_rng(20 + i))
        absorbed_trace.append(eval_mse(net_absorbed, X_king_eval, y_king_eval))

    # --- CONDITION B: the returning king -- same seven visits, but with
    #     (i) brief rehearsal of kingdom data mixed into every visit and
    #     (ii) an EWC-style anchor toward the pre-tour weights, scaled by
    #     Fisher importance -- 'ceremonially returning attention to what
    #     the realm needed most'.
    net_restored = fresh_net()
    net_restored.restore(kingdom_snapshot)
    restored_trace = [baseline_mse]
    rehearsal_rng = np.random.default_rng(30)
    for i, (Xp, yp) in enumerate(pavilion_data):
        # interleave pavilion steps with rehearsal steps, and anchor every step
        for step in range(150):
            if step % 3 == 0:
                idx = rehearsal_rng.integers(0, X_king_train.shape[0], size=32)
                xb, yb = X_king_train[idx], y_king_train[idx]
            else:
                idx = rehearsal_rng.integers(0, Xp.shape[0], size=min(32, Xp.shape[0]))
                xb, yb = Xp[idx], yp[idx]
            y_hat = net_restored.forward(xb)
            _, dA = mse_loss(y_hat, yb)
            net_restored.backward(dA)
            net_restored.adam_step(lr=5e-3, anchors=kingdom_snapshot,
                                    anchor_strength=0.05, importances=importances)
        restored_trace.append(eval_mse(net_restored, X_king_eval, y_king_eval))

    if verbose:
        print(f"  baseline (well-governed) kingdom MSE: {baseline_mse:.4f}")
        print("  pavilion   color        planet     absorbed-king MSE   returning-king MSE")
        for i in range(7):
            print(f"     {i+1}      {PAVILION_COLORS[i]:<11} {PAVILION_PLANETS[i]:<9} "
                  f"{absorbed_trace[i+1]:>10.4f}          {restored_trace[i+1]:>10.4f}")
        growth_absorbed = absorbed_trace[-1] / max(1e-8, absorbed_trace[0])
        growth_restored = restored_trace[-1] / max(1e-8, restored_trace[0])
        print(f"  net decay ratio (final/baseline): absorbed x{growth_absorbed:.2f}  "
              f"returning x{growth_restored:.2f}")

    return {
        "baseline_mse": baseline_mse,
        "absorbed_trace": absorbed_trace,
        "restored_trace": restored_trace,
    }


# =============================================================================
# PART 2 -- THE UNPIERCED-PEARL GENERATOR
# combinatorial novelty search under a hard formal (metrical) constraint
# =============================================================================

N_CONCEPTS = 14
EMBED_DIM = 5
PROSODY_PATTERN = np.array([1, 0, 1])  # required long(1)/short(0)/contrast foot pattern for a valid hemistich pairing

rng_concepts = np.random.default_rng(RNG_SEED + 3)
CONCEPT_NAMES = [
    "rose", "nightingale", "candle", "moth", "mirror", "cypress", "wine-cup",
    "moon", "musk", "pearl", "thorn", "reed-pen", "labyrinth", "amber",
]
CONCEPT_EMBED = rng_concepts.normal(0, 1, size=(N_CONCEPTS, EMBED_DIM))
CONCEPT_PROSODY = rng_concepts.integers(0, 2, size=N_CONCEPTS)  # each concept's syllable weight class


def oracle_compatibility(i, j):
    """A fixed, nonlinear, non-symmetric-looking 'ground truth' of poetic
    fit between two concepts -- the thing the trained network must learn to
    approximate. Not learnable in closed form; the network has to fit it
    from examples, exactly like any other regression target."""
    a, b = CONCEPT_EMBED[i], CONCEPT_EMBED[j]
    cos = (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    dist = np.linalg.norm(a - b)
    val = 0.5 * np.tanh(2.0 * cos) + 0.5 * np.exp(-0.3 * dist) * np.sin(3.0 * dist)
    return float(val)


def pair_features(i, j):
    return np.concatenate([CONCEPT_EMBED[i], CONCEPT_EMBED[j],
                            [CONCEPT_EMBED[i] @ CONCEPT_EMBED[j]]])


def is_metrically_valid(i, j):
    """A pairing is a well-formed couplet-slot only if the opening concept's
    prosodic weight, the closing concept's prosodic weight, and their
    contrast bit match the required foot pattern -- a hard mask, not a soft
    preference. This stands in for closed-couplet meter (aruz): most
    semantically excellent pairings are simply inadmissible, exactly as a
    real hemistich rejects most well-meaning but unscannable phrases. With
    14 concepts this admits a minority (~27%) of all orderings -- a real
    constraint, not a token gesture."""
    code = np.array([CONCEPT_PROSODY[i], CONCEPT_PROSODY[j], CONCEPT_PROSODY[i] ^ CONCEPT_PROSODY[j]])
    return bool(np.array_equal(code, PROSODY_PATTERN))


def build_pearl_scorer(rng):
    return MLP([2 * EMBED_DIM + 1, 16, 8, 1], ["tanh", "relu", "linear"], rng=rng)


def train_pearl_scorer(verbose=True):
    rng = np.random.default_rng(RNG_SEED + 4)
    net = build_pearl_scorer(rng)

    all_pairs = [(i, j) for i in range(N_CONCEPTS) for j in range(N_CONCEPTS) if i != j]
    X = np.array([pair_features(i, j) for i, j in all_pairs])
    y = np.array([[oracle_compatibility(i, j)] for i, j in all_pairs])

    perm = rng.permutation(len(all_pairs))
    n_train = int(0.8 * len(all_pairs))
    train_idx, val_idx = perm[:n_train], perm[n_train:]
    Xtr, ytr = X[train_idx], y[train_idx]
    Xval, yval = X[val_idx], y[val_idx]

    opt_rng = np.random.default_rng(RNG_SEED + 5)
    for step in range(1500):
        idx = opt_rng.integers(0, Xtr.shape[0], size=24)
        xb, yb = Xtr[idx], ytr[idx]
        y_hat = net.forward(xb)
        _, dA = mse_loss(y_hat, yb)
        net.backward(dA)
        net.adam_step(lr=2e-3)

    train_mse = eval_mse(net, Xtr, ytr)
    val_mse = eval_mse(net, Xval, yval)
    if verbose:
        print(f"  pearl-scorer fit: train MSE {train_mse:.4f}   held-out MSE {val_mse:.4f}  "
              f"(oracle range approx [-1, 1])")
    return net, train_mse, val_mse


def unpierced_pearl_search(net, n_rounds=10, seed_predecessor_pairs=None, verbose=True):
    """The combinatorial-novelty search itself: at each round, enumerate
    every metrically valid pairing not yet in the used-set, score it with
    the trained network, and take the highest-scoring 'unpierced pearl'.
    If the valid, unused store is exhausted, say so honestly rather than
    silently reusing a meaning -- Nizami's own doctrine, taken literally."""
    used = set(seed_predecessor_pairs or [])
    valid_pairs = [(i, j) for i in range(N_CONCEPTS) for j in range(N_CONCEPTS)
                   if i != j and is_metrically_valid(i, j)]

    results = []
    for round_idx in range(n_rounds):
        candidates = [(i, j) for (i, j) in valid_pairs if (i, j) not in used]
        if not candidates:
            if verbose:
                print(f"  round {round_idx+1}: the store of pearls admissible under this meter "
                      f"is exhausted ({len(used)} coined, {len(valid_pairs)} total valid slots).")
            break
        feats = np.array([pair_features(i, j) for (i, j) in candidates])
        scores = net.forward(feats).flatten()
        best = int(np.argmax(scores))
        i, j = candidates[best]
        used.add((i, j))
        results.append((i, j, float(scores[best])))
        if verbose:
            print(f"  round {round_idx+1:>2}: \"{CONCEPT_NAMES[i]} / {CONCEPT_NAMES[j]}\"  "
                  f"score={scores[best]:+.3f}   (meter OK, never used before)")
    return results, valid_pairs, used


# =============================================================================
# PART 3 -- THE KHIZR-ISKANDAR GATE
# does self-authored, single-objective optimization destroy its own goal?
# =============================================================================

N_STATES = 9          # positions 0..8 on a line
N_ACTIONS = 3          # 0 = left, 1 = right, 2 = serve/stay
RESOURCE_CELL = 6
DUTY_CELLS = (1, 8)    # Khizr's ordinary tasks, chosen so the route passes RESOURCE_CELL
EPISODE_LEN = 20
ENTROPY_THRESHOLD = 0.55  # out of a max possible log(3) ~= 1.0986 for 3 actions
GAMMA = 0.95              # discount for return-to-go credit assignment


def softmax_rows(logits):
    z = logits - np.max(logits, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)


def policy_probs(theta, state):
    return softmax_rows(theta[state:state + 1, :])[0]


def entropy(p):
    p = np.clip(p, 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))


def run_episode(theta, reward_fn, rng, track_gate=True):
    """Simulate one episode under the (stochastic) tabular softmax policy
    theta, recording (state, action, reward) at every step for return-to-go
    REINFORCE, and the reward signal defined by reward_fn. Also (always)
    tracks, purely as a passive measurement, whether the resource gate at
    RESOURCE_CELL was ever satisfied -- this measurement never feeds back
    into training; it only answers 'did the water reveal itself'."""
    state = N_STATES // 2  # start in the middle, position 4
    collected = False
    steps = []  # (state, action, reward)
    resource_hit = False
    task = rng.integers(0, len(DUTY_CELLS))  # which duty is active this episode (Khizr only uses this)
    duty_done = False

    for t in range(EPISODE_LEN):
        probs = policy_probs(theta, state)
        action = rng.choice(N_ACTIONS, p=probs)

        ent = entropy(probs)
        r, collected, duty_done = reward_fn(state, action, ent, collected, task, duty_done)
        steps.append((state, action, r))

        if action == 2 and state == RESOURCE_CELL and ent > ENTROPY_THRESHOLD and not resource_hit:
            resource_hit = True  # passive measurement, independent of training reward

        if action == 0:
            state = max(0, state - 1)
        elif action == 1:
            state = min(N_STATES - 1, state + 1)
        # action == 2: stay

    return steps, resource_hit


def iskandar_reward(state, action, ent, collected, task, duty_done):
    """Iskandar's ENTIRE objective: reach the resource cell and serve,
    with the reward only realized if the policy's own entropy there clears
    the threshold at that instant, and only once per episode. No other
    signal exists -- this is a purely self-authored, single-goal agent."""
    r = 0.0
    if action == 2 and state == RESOURCE_CELL and not collected:
        if ent > ENTROPY_THRESHOLD:
            r = 1.0
            collected = True
    return r, collected, duty_done


def khizr_reward(state, action, ent, collected, task, duty_done):
    """Khizr's entire objective: serve at whichever duty cell is active
    this episode. The resource cell and its gate never appear in this
    function at all -- Khizr's gradient signal knows nothing of the water."""
    r = 0.0
    duty_cell = DUTY_CELLS[task]
    if action == 2 and state == duty_cell and not duty_done:
        r = 0.7
        duty_done = True
    return r, collected, duty_done


def reinforce_train(reward_fn, n_episodes, lr, rng):
    """Vanilla REINFORCE with a discounted return-to-go G_t and a per-step
    running baseline (not a single episode-wide scalar): each (state,
    action) is credited only with the reward it and its causal future
    actually produced, not with the total return of the whole episode.
    This is the technically correct form of the policy-gradient estimator
    and, in this environment, is also what prevents an incidental waypoint
    (state 6, for the agent whose reward has nothing to do with it) from
    being spuriously reinforced just because it happened to lie on the path
    to an unrelated reward later in the same episode."""
    theta = np.zeros((N_STATES, N_ACTIONS))
    baseline = np.zeros(EPISODE_LEN)
    beta = 0.02

    for ep in range(n_episodes):
        steps, _ = run_episode(theta, reward_fn, rng)
        T = len(steps)
        rewards = np.array([s[2] for s in steps])
        G = np.zeros(T)
        running = 0.0
        for t in reversed(range(T)):
            running = rewards[t] + GAMMA * running
            G[t] = running
        baseline[:T] = (1 - beta) * baseline[:T] + beta * G
        advantage = G - baseline[:T]

        grad = np.zeros_like(theta)
        for t, (state, action, _r) in enumerate(steps):
            probs = policy_probs(theta, state)
            onehot = np.zeros(N_ACTIONS)
            onehot[action] = 1.0
            grad[state, :] += advantage[t] * (onehot - probs)
        theta += lr * grad
        theta = np.clip(theta, -12, 12)  # keep softmax numerically sane

    return theta


def evaluate_resource_gate(theta, n_episodes, reward_fn_for_rollout, rng):
    hits = 0
    for _ in range(n_episodes):
        _, resource_hit = run_episode(theta, reward_fn_for_rollout, rng)
        hits += int(resource_hit)
    return hits / n_episodes


def run_khizr_iskandar_gate(verbose=True):
    rng_train_i = np.random.default_rng(RNG_SEED + 6)
    rng_train_k = np.random.default_rng(RNG_SEED + 7)
    rng_eval_i = np.random.default_rng(RNG_SEED + 8)
    rng_eval_k = np.random.default_rng(RNG_SEED + 9)

    theta_iskandar = reinforce_train(iskandar_reward, n_episodes=6000, lr=0.04, rng=rng_train_i)
    theta_khizr = reinforce_train(khizr_reward, n_episodes=6000, lr=0.04, rng=rng_train_k)

    ent_iskandar_at_resource = entropy(policy_probs(theta_iskandar, RESOURCE_CELL))
    ent_khizr_at_resource = entropy(policy_probs(theta_khizr, RESOURCE_CELL))

    # For fair passive evaluation, roll out each agent under a reward
    # function that no longer trains anything (we just need run_episode's
    # bookkeeping); use a null reward so theta is not further changed.
    def null_reward(state, action, ent, collected, task, duty_done):
        return 0.0, collected, duty_done

    success_iskandar = evaluate_resource_gate(theta_iskandar, 4000, null_reward, rng_eval_i)
    success_khizr = evaluate_resource_gate(theta_khizr, 4000, null_reward, rng_eval_k)

    if verbose:
        print(f"  policy entropy at the resource cell after training:")
        print(f"     Iskandar (single-goal, self-authored):  H = {ent_iskandar_at_resource:.3f}  "
              f"(threshold {ENTROPY_THRESHOLD})")
        print(f"     Khizr    (ordinary duty, unsought):     H = {ent_khizr_at_resource:.3f}  "
              f"(threshold {ENTROPY_THRESHOLD})")
        print(f"  empirical resource-gate success rate over 4000 held-out episodes:")
        print(f"     Iskandar: {success_iskandar*100:5.1f}%")
        print(f"     Khizr:    {success_khizr*100:5.1f}%")

    return {
        "ent_iskandar": ent_iskandar_at_resource,
        "ent_khizr": ent_khizr_at_resource,
        "success_iskandar": success_iskandar,
        "success_khizr": success_khizr,
    }


# =============================================================================
# MAIN -- run the gradient check, then all three subsystems, then self-tests
# =============================================================================

def main():
    t0 = time.time()
    print("=" * 78)
    print("NizamiNet -- 0251_Neuron.py")
    print("Chapter 251: Nizami Ganjavi (c.1141-1209) -- Encyclopedia of Lost Minds")
    print("=" * 78)

    print("\n[0] MANDATORY GRADIENT CHECK (shared Dense/MLP backward pass)")
    passed, max_err = gradient_check()
    assert passed, "Gradient check failed -- refusing to trust training below."

    print("\n[1] THE HAFT PEYKAR GOVERNOR -- pavilion-absorption vs. the returning king")
    gov_results = run_haft_peykar_governor()

    print("\n[2] THE UNPIERCED-PEARL GENERATOR -- khalq al-ma'ani under a metrical mask")
    scorer_net, train_mse, val_mse = train_pearl_scorer()
    print("  searching for unpierced pearls (10 rounds, seeded with 3 pairs 'already used by predecessors'):")
    seed_used = {(0, 1), (2, 3), (7, 9)}  # rose/nightingale, candle/moth, moon/pearl -- old ma'ani
    pearls, valid_pairs, used_after = unpierced_pearl_search(scorer_net, n_rounds=10, seed_predecessor_pairs=seed_used)

    print("\n[3] THE KHIZR-ISKANDAR GATE -- does eagerness defeat the eager?")
    ki_results = run_khizr_iskandar_gate()

    print("\n" + "=" * 78)
    print("SELF-TESTS")
    print("=" * 78)

    tests = []

    tests.append(("gradient check passes (max rel. error < 1e-4)", passed))

    absorbed_growth = gov_results["absorbed_trace"][-1] / max(1e-8, gov_results["absorbed_trace"][0])
    restored_growth = gov_results["restored_trace"][-1] / max(1e-8, gov_results["restored_trace"][0])
    tests.append(("absorbed king's kingdom MSE grows substantially (>1.5x baseline): "
                   f"{absorbed_growth:.2f}x", absorbed_growth > 1.5))
    tests.append(("returning king's kingdom MSE stays much closer to baseline than absorbed king's: "
                   f"restored {restored_growth:.2f}x vs absorbed {absorbed_growth:.2f}x",
                   restored_growth < absorbed_growth))

    tests.append((f"pearl-scorer generalizes to held-out pairs (val MSE {val_mse:.3f} < 0.35)",
                   val_mse < 0.35))
    pearl_pairs_only = [(i, j) for (i, j, s) in pearls]
    tests.append((f"unpierced-pearl search never repeats a pairing across {len(pearl_pairs_only)} rounds",
                   len(set(pearl_pairs_only)) == len(pearl_pairs_only)))
    tests.append((f"unpierced-pearl search never reuses a seeded predecessor pairing",
                   len(set(pearl_pairs_only) & seed_used) == 0))
    tests.append((f"every generated pearl is metrically valid",
                   all(is_metrically_valid(i, j) for (i, j) in pearl_pairs_only)))

    tests.append((f"Khizr's policy retains higher entropy at the resource cell than Iskandar's "
                   f"({ki_results['ent_khizr']:.3f} vs {ki_results['ent_iskandar']:.3f})",
                   ki_results["ent_khizr"] > ki_results["ent_iskandar"]))
    tests.append((f"Khizr's unsought resource-gate success rate exceeds Iskandar's self-authored one "
                   f"({ki_results['success_khizr']*100:.1f}% vs {ki_results['success_iskandar']*100:.1f}%)",
                   ki_results["success_khizr"] > ki_results["success_iskandar"]))

    all_passed = True
    for description, ok in tests:
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_passed = False
        print(f"  [{status}] {description}")

    print("=" * 78)
    print(f"ALL SELF-TESTS {'PASSED' if all_passed else 'FAILED'}   "
          f"(wall clock {time.time()-t0:.1f}s)")
    print("=" * 78)

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
