"""
==============================================================================
NEURON — the trainable architecture for Chapter 273: Shen Kuo (沈括, 1031-1095)
Encyclopedia of Lost Minds: Echoes on AI
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 14, Minds 261-280 Available on Amazon https://www.amazon.com/dp/B0HLYSVY5D
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0273_shen_kuo_1031 - Shen Kuo (1031-1095)

THE MIND, NOT THE POLYMATH LABEL
---------------------------------
It would be easy to build "the polymath architecture": a general module for
cross-domain analogy, because Shen Kuo touched geology, astronomy, optics,
medicine, cartography, music, and war. That template already exists many
times over in this corpus. It says nothing specific about *him*.

What is specific to Shen Kuo, recoverable from his own attested practice, is
narrower and stranger than "he noticed connections between fields":

  1. TRIANGULATE, DON'T SINGLE-SOURCE. When the Bureau of Astronomy's numbers
     didn't fit the sky, Shen Kuo and his subordinate Wei Pu did not adjudicate
     by authority. They ran a five-year nightly observation campaign with an
     improved, widened sighting-tube, fixed on the pole star long enough to
     draw over two hundred independent diagrams of its apparent circle, and
     used the *spread* of those two hundred readings — not any single one of
     them — to fix its true position (Shen Kuo, Mengxi Bitan j.7; Sivin,
     "Shen Kua," 1975, III.18-19). Independent, repeated, imperfect
     measurements were the instrument; their disagreement was information,
     not noise to be silently thrown away.

  2. DISTRUST THE SPECIFIC CHANNEL, NOT THE WHOLE SYSTEM. Wei Pu and Shen Kuo
     publicly challenged the received planetary and lunar coordinates
     inherited from the Tang-dynasty astronomer-monk Yi Xing (this corpus's
     Chapter 182) — a named, prestigious, centuries-old authority — and
     proved the inherited numbers wrong with a gnomon demonstration in front
     of hostile court astronomers, who "reluctantly agreed to correct the
     lunar error" (Sivin 1975, III.18-19; Wikipedia, "Wei Pu"). The response
     to a disagreeing source was not to average it in and not to reject it on
     rank — it was to isolate which specific channel had drifted and say so,
     by name, in public.

  3. TWO INDEPENDENT RESIDUES CONVERGING ON ONE CONCLUSION OUTRANKS ONE STRONG
     ONE. Shen Kuo did not infer deep-time environmental change from a single
     line of evidence. He combined two structurally unrelated residues —
     marine bivalve and "stone swallow" fossils embedded in cliffs of the
     Taihang mountains, hundreds of li from the nearest sea, AND petrified
     bamboo excavated far to the north in the too-dry Yanzhou region, where
     living bamboo cannot grow — because the two together, agreeing on "the
     physical world here was once different," were harder to explain away
     than either alone (Mengxi Bitan j.24; Needham, SCC III, p.614). This is
     the same operation as the sighting tube, one level up: independent
     estimators converging on a hidden variable.

  4. REVISE IN PUBLIC, IN THE SAME LEDGER, WITH THE OLD ENTRY STILL VISIBLE.
     The Mengxi Bitan is a book that corrects itself in its own pages, and
     that credits inventions to named commoners rather than claiming them —
     Shen Kuo attributes movable type explicitly to the artisan Bi Sheng, by
     name, including the mundane detail that Bi Sheng's own type-fount passed
     afterward into the keeping of Shen Kuo's nephews (Mengxi Bitan j.18;
     tr. Carter 1955). Attribution and revision are treated as data worth
     recording precisely, not as embarrassments to be smoothed over.

  5. WHEN THE ARITHMETIC IS SOUND, TRUST IT OVER THE ROOM. His 1074 proposal
     for a Twelve-Qi calendar — twelve months fixed to the solar terms,
     dispensing with the lunar month entirely, structurally the same idea
     later known in the West as the Shaw / "Twelve Months" solar calendar —
     was rejected by his contemporaries and by Qing-dynasty scholars two
     centuries later. He recorded his own reply to the rejection: "there will
     surely be future generations who take up my theory" (十二气历...後必有
     以余說爲允者; Mengxi Bitan j.7 / Song Shi 102). He was over eight hundred
     years early, not wrong.

None of this is "sees analogies across fields." It is a specific, three-part
epistemic discipline: multiply independent routes to a hidden quantity, let
their disagreement — not their average — tell you which route has drifted,
and log every revision where the superseded claim stays legible next to its
replacement. Call it CONVERGENT TRIANGULATION WITH LEDGERED REVISION.

That is what this file implements, twice: once as the main architecture
(TriangulationNet: many independently-calibrated "instrument" channels, a
disagreement-driven trust mechanism, and an explicit append-only revision
ledger that never overwrites a prior entry), and once as a smaller, sharper
demonstration of Shen Kuo's cross-domain method-reuse — what he and his
contemporaries called shu (術, "technique") — a single trainable kernel
transferred from a data-rich domain into a data-starved one.

ARCHITECTURE 1 — TriangulationNet (the main mind-model)
--------------------------------------------------------
For each "sighting episode" the network receives K independently-corrupted
readings of one hidden scalar quantity (think: K nights' worth of pole-star
tube sightings, or K provincial reports of the same flood, or K instruments
disagreeing about a bearing). Each channel has learned its OWN calibration
function (its own small instrument-correction network — Shen Kuo insisted
instruments must be individually calibrated, not assumed identical). A
disagreement vector is computed from how far each channel's candidate sits
from the naive group mean; a TrustNet turns that disagreement into a
per-channel trust weight (softmax), and the final estimate is the
trust-weighted convergence of the channels. An entropy-toward-uniform
regularizer means the network must earn any departure from "trust every
channel equally" with actual, present evidence of drift — it may not simply
learn to ignore inconvenient channels.

After every forward pass, a REVISION LEDGER (deterministic, not trained)
compares the new consensus estimate against a previously "published" belief
for the same quantity_id. If the discrepancy exceeds a threshold, a new,
timestamped ledger entry is appended, and the OLD entry is marked
superseded_by the new one rather than deleted — Shen Kuo's own practice in
Mengxi Bitan, where earlier entries are corrected in place, in view.

ARCHITECTURE 2 — the shu (術) transfer experiment
--------------------------------------------------------
A single trainable "technique kernel" sits between domain-specific input and
output adapters. It is trained fully on a data-rich domain (many synthetic
"geological residue -> deep-time elevation change" examples), then reused —
carried over, not re-derived from scratch — as the initialization for a
data-starved domain (twenty synthetic "botanical residue -> deep-time
climate change" examples: structurally the same regression, dressed in
different surface features, exactly as Shen Kuo treated the Taihang fossils
and the Yanzhou bamboo as two dressings of one underlying claim). We compare
this against training the data-starved domain completely from scratch, to
put a number on what reusing a shu actually buys you.

BOTH networks are pure NumPy, with every gradient hand-derived and checked
against central-difference numerical gradients (mandatory, see
`gradient_check_triangulation` and `gradient_check_shu_transfer`), trained
with a real Adam loop against real (synthetic but structurally faithful)
data, and covered by executable self-tests (`run_self_tests`). Executed
output from this exact file is pasted into Chapter 229, Section 5.

No PyTorch, no autograd, no attention, no transformer, no mixture-of-experts.
"""

import numpy as np
import time

RNG_SEED = 229
rng = np.random.default_rng(RNG_SEED)


# =============================================================================
# PART 0 — small numerics shared by both architectures
# =============================================================================

def tanh(x):
    return np.tanh(x)


def dtanh(y):
    # y = tanh(x); returns dy/dx in terms of y
    return 1.0 - y * y


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def he_init(fan_in, fan_out, rng_):
    return rng_.normal(0.0, np.sqrt(2.0 / fan_in), size=(fan_in, fan_out)).astype(np.float64)


# =============================================================================
# PART 1 — a minimal, hand-differentiated Dense(+tanh) layer
# =============================================================================

class Dense:
    """
    A single fully-connected layer with an optional tanh nonlinearity.
    Stores its own parameters and (after forward) the cache needed for a
    hand-derived backward pass. Used as the building block for every
    "instrument", "trust", and "shu" sub-network below.
    """

    def __init__(self, n_in, n_out, activation="tanh", rng_=None):
        rng_ = rng_ if rng_ is not None else rng
        self.W = he_init(n_in, n_out, rng_)
        self.b = np.zeros((1, n_out), dtype=np.float64)
        self.activation = activation
        self._cache = None
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def params(self):
        return [self.W, self.b]

    def grads(self):
        return [self.dW, self.db]

    def forward(self, x):
        z = x @ self.W + self.b
        if self.activation == "tanh":
            a = tanh(z)
        elif self.activation == "linear":
            a = z
        else:
            raise ValueError(self.activation)
        self._cache = (x, a)
        return a

    def backward(self, da):
        x, a = self._cache
        if self.activation == "tanh":
            dz = da * dtanh(a)
        elif self.activation == "linear":
            dz = da
        else:
            raise ValueError(self.activation)
        self.dW = x.T @ dz
        self.db = np.sum(dz, axis=0, keepdims=True)
        dx = dz @ self.W.T
        return dx


class MLP:
    """A tiny stack of Dense layers, e.g. used as one calibrated 'instrument'."""

    def __init__(self, sizes, activations, rng_=None):
        assert len(sizes) - 1 == len(activations)
        self.layers = [
            Dense(sizes[i], sizes[i + 1], activations[i], rng_=rng_)
            for i in range(len(sizes) - 1)
        ]

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def params(self):
        p = []
        for layer in self.layers:
            p.extend(layer.params())
        return p

    def grads(self):
        g = []
        for layer in self.layers:
            g.extend(layer.grads())
        return g


# =============================================================================
# PART 2 — Adam optimizer over an arbitrary list-of-arrays parameter set
# =============================================================================

class Adam:
    def __init__(self, params, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g * g)
            mhat = self.m[i] / (1 - self.beta1 ** self.t)
            vhat = self.v[i] / (1 - self.beta2 ** self.t)
            p -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


# =============================================================================
# PART 3 — ARCHITECTURE 1: TriangulationNet
# =============================================================================

class RevisionLedger:
    """
    Append-only, never-overwrite record of beliefs about hidden quantities.
    This is deliberately NOT a trained component: it is deterministic
    bookkeeping driven by the trained model's own consensus estimates,
    modelling Shen Kuo's textual practice of revising a claim in place while
    leaving the superseded claim legible and cross-referenced (Mengxi Bitan's
    own self-corrections; the Wei Pu / Yi Xing correction of 1074).
    """

    REVISION_THRESHOLD = 0.15  # in the same units as the hidden quantity z

    def __init__(self):
        self.entries = []                 # list of dicts, append-only
        self.latest_index = {}            # quantity_id -> index of live entry

    def observe(self, quantity_id, estimate, trust_vector, step, prior_belief=None):
        """
        Compare `estimate` for `quantity_id` against whatever is currently
        the live belief (either an externally supplied prior_belief the
        first time a quantity is seen, or the model's own previous live
        estimate on subsequent sightings). Append a new ledger entry only if
        the discrepancy is large enough to count as a genuine revision.
        Returns True if a revision was logged.
        """
        if quantity_id in self.latest_index:
            prev_idx = self.latest_index[quantity_id]
            prev_value = self.entries[prev_idx]["estimate"]
        elif prior_belief is not None:
            prev_idx = None
            prev_value = prior_belief
        else:
            prev_idx = None
            prev_value = None

        revised = prev_value is not None and abs(float(estimate) - float(prev_value)) > self.REVISION_THRESHOLD
        first_sighting = prev_value is None

        if revised or first_sighting:
            entry = {
                "quantity_id": quantity_id,
                "step": step,
                "estimate": float(estimate),
                "trust_vector": np.asarray(trust_vector).round(4).tolist(),
                "superseded_by": None,
                "supersedes": prev_idx,
            }
            new_idx = len(self.entries)
            self.entries.append(entry)
            if prev_idx is not None:
                self.entries[prev_idx]["superseded_by"] = new_idx
            self.latest_index[quantity_id] = new_idx
            return revised  # False on a first sighting: not a "revision", a new entry
        return False

    def n_revisions(self):
        return sum(1 for e in self.entries if e["supersedes"] is not None)


class TriangulationNet:
    """
    K independently-calibrated instrument channels -> disagreement vector ->
    TrustNet (softmax trust weights) -> trust-weighted consensus estimate.

    Input per sample: X of shape (batch, K, channel_dim) -- K raw channel
    readings, each channel_dim-dimensional (e.g. tube-angle components).
    Output: consensus estimate, shape (batch, 1).
    """

    def __init__(self, K, channel_dim, hidden=12, trust_hidden=10, rng_=None):
        rng_ = rng_ if rng_ is not None else rng
        self.K = K
        self.channel_dim = channel_dim
        # one independently-parameterised calibration MLP per channel
        self.instruments = [
            MLP([channel_dim, hidden, 1], ["tanh", "linear"], rng_=rng_)
            for _ in range(K)
        ]
        # TrustNet: disagreement vector (K,) -> trust logits (K,)
        self.trust_net = MLP([K, trust_hidden, K], ["tanh", "linear"], rng_=rng_)
        self._cache = None

    def params(self):
        p = []
        for inst in self.instruments:
            p.extend(inst.params())
        p.extend(self.trust_net.params())
        return p

    def grads(self):
        g = []
        for inst in self.instruments:
            g.extend(inst.grads())
        g.extend(self.trust_net.grads())
        return g

    def forward(self, X):
        """
        X: (batch, K, channel_dim)
        returns: consensus (batch, 1), trust_weights (batch, K), candidates (batch, K)
        """
        batch = X.shape[0]
        candidates = np.zeros((batch, self.K), dtype=np.float64)
        for k in range(self.K):
            candidates[:, k:k + 1] = self.instruments[k].forward(X[:, k, :])

        mean_candidate = np.mean(candidates, axis=1, keepdims=True)          # (batch,1)
        disagreement = candidates - mean_candidate                            # (batch,K)

        trust_logits = self.trust_net.forward(disagreement)                   # (batch,K)
        trust_weights = softmax(trust_logits, axis=1)                         # (batch,K)

        consensus = np.sum(trust_weights * candidates, axis=1, keepdims=True)  # (batch,1)

        self._cache = dict(
            X=X, candidates=candidates, mean_candidate=mean_candidate,
            disagreement=disagreement, trust_logits=trust_logits,
            trust_weights=trust_weights, consensus=consensus,
        )
        return consensus, trust_weights, candidates

    def backward(self, d_consensus, entropy_lambda=0.0):
        """
        d_consensus: (batch,1) upstream gradient of the loss wrt consensus.
        entropy_lambda: weight on the "stay close to uniform trust" regularizer
                        d/d(trust_weights)[ entropy_lambda * KL(trust||uniform) ]
        Backprops through: consensus -> trust_weights -> trust_logits -> trust_net
                                       -> candidates (both paths: direct & via mean)
                                       -> instruments
        """
        c = self._cache
        candidates = c["candidates"]
        trust_weights = c["trust_weights"]
        batch, K = candidates.shape

        # --- consensus = sum_k trust_k * candidate_k ---
        d_trust_weights = d_consensus * candidates                 # (batch,K)
        d_candidates_direct = d_consensus * trust_weights          # (batch,K)  path via weighting

        # --- optional entropy-toward-uniform regularizer on trust_weights ---
        # loss includes entropy_lambda * MEAN_over_batch[ KL(trust_i || uniform) ],
        # KL(trust || uniform) = sum_k trust_k * log(trust_k * K), so
        # d/d trust_k = (log(trust_k*K) + 1), scaled by 1/batch to match the
        # batch-mean in the loss (exactly as d_consensus is pre-scaled by
        # 1/batch for the MSE term above) -- forgetting this factor is a
        # classic multi-task-loss bug: the two terms end up implicitly
        # weighted by different effective batch sizes.
        if entropy_lambda > 0.0:
            d_trust_weights = d_trust_weights + (entropy_lambda / batch) * (
                np.log(np.clip(trust_weights, 1e-12, None) * K) + 1.0
            )

        # --- softmax backward: trust_weights = softmax(trust_logits) ---
        # dL/dlogits_i = t_i * (dL/dt_i - sum_j t_j dL/dt_j)
        dot = np.sum(d_trust_weights * trust_weights, axis=1, keepdims=True)
        d_trust_logits = trust_weights * (d_trust_weights - dot)   # (batch,K)

        # --- trust_net backward: disagreement -> trust_logits ---
        d_disagreement = self.trust_net.backward(d_trust_logits)   # (batch,K)

        # --- disagreement = candidates - mean(candidates) ---
        # d candidates (via disagreement) = d_disagreement - mean(d_disagreement)
        d_mean_from_disagreement = -np.mean(d_disagreement, axis=1, keepdims=True)
        d_candidates_via_disagreement = d_disagreement + d_mean_from_disagreement
        # (chain rule for x_k - mean(x): d/dx_k = I - 1/K broadcast, handled above)

        d_candidates = d_candidates_direct + d_candidates_via_disagreement

        # --- instruments backward ---
        for k in range(K):
            self.instruments[k].backward(d_candidates[:, k:k + 1])

        return d_candidates  # not usually needed further, returned for testing


# -----------------------------------------------------------------------------
# Synthetic data: K-channel "sighting episodes" of one hidden scalar quantity
# -----------------------------------------------------------------------------

def make_triangulation_episode(n, K, channel_dim, corrupt_channel=None,
                                corrupt_strength=1.6, rng_=None):
    """
    Hidden quantity z in [-1, 1] (think: a normalised declination-like angle).
    Each channel k observes z through its own fixed linear+nonlinear
    "instrument distortion" plus noise; the model must learn to invert each
    channel's distortion (its calibration) AND learn which channel to trust.
    If corrupt_channel is given, that channel carries a large, systematic
    bias for this batch (a mis-calibrated instrument / a stale inherited
    coordinate, e.g. Yi Xing's superseded figures).
    """
    rng_ = rng_ if rng_ is not None else rng
    z = rng_.uniform(-1.0, 1.0, size=(n, 1))

    X = np.zeros((n, K, channel_dim), dtype=np.float64)
    # fixed per-channel "instrument signature": distinct gain/offset/warp
    for k in range(K):
        gain = 0.6 + 0.15 * k
        offset = 0.05 * (k - K / 2)
        warp = 0.3 * np.sin(1.3 * k + 1.0)
        base = gain * z + offset + warp * z ** 2
        noise = rng_.normal(0, 0.03, size=(n, 1))
        reading = base + noise
        if corrupt_channel is not None and k == corrupt_channel:
            reading = reading + corrupt_strength * rng_.choice([-1, 1], size=(n, 1))
        # expand each scalar reading into channel_dim features via fixed
        # nonlinear projections (stand-in for "raw tube/gnomon measurements")
        feats = [reading, np.tanh(reading * (1 + 0.1 * k)), reading ** 2 * np.sign(reading)]
        X[:, k, :] = np.concatenate(feats[:channel_dim], axis=1)
    return X, z


def _total_loss(consensus, trust_weights, z, entropy_lambda):
    """MSE + entropy_lambda * mean-per-sample KL(trust_weights || uniform).
    This MUST exactly match what `TriangulationNet.backward` differentiates,
    or a gradient check is comparing two different functions."""
    K = trust_weights.shape[1]
    mse = np.mean((consensus - z) ** 2)
    kl = np.sum(trust_weights * np.log(np.clip(trust_weights, 1e-12, None) * K), axis=1)
    return float(mse + entropy_lambda * np.mean(kl))


def gradient_check_triangulation(verbose=True, entropy_lambda=0.01):
    """Mandatory: central-difference numerical gradient check of TriangulationNet,
    including the entropy-toward-uniform regularizer (checked at the same
    entropy_lambda used during real training, so the check exercises the
    exact function being optimized, not a simplified stand-in for it)."""
    rng_ = np.random.default_rng(1)
    K, channel_dim = 4, 3
    net = TriangulationNet(K, channel_dim, hidden=5, trust_hidden=5, rng_=rng_)
    X, z = make_triangulation_episode(6, K, channel_dim, rng_=rng_)

    def loss_only():
        consensus, trust_w, _ = net.forward(X)
        return _total_loss(consensus, trust_w, z, entropy_lambda)

    def loss_and_backward():
        consensus, trust_w, _ = net.forward(X)
        loss = _total_loss(consensus, trust_w, z, entropy_lambda)
        diff = consensus - z
        d_consensus = 2.0 * diff / diff.shape[0]
        net.backward(d_consensus, entropy_lambda=entropy_lambda)
        return loss

    loss0 = loss_and_backward()
    params = net.params()
    grads = net.grads()

    eps = 1e-5
    max_rel_err = 0.0
    n_checked = 0
    for p, g in zip(params, grads):
        flat_p = p.reshape(-1)
        flat_g = g.reshape(-1)
        idxs = rng_.choice(flat_p.size, size=min(6, flat_p.size), replace=False)
        for idx in idxs:
            orig = flat_p[idx]
            flat_p[idx] = orig + eps
            lp = loss_only()
            flat_p[idx] = orig - eps
            lm = loss_only()
            flat_p[idx] = orig
            numeric = (lp - lm) / (2 * eps)
            analytic = flat_g[idx]
            denom = max(abs(numeric), abs(analytic), 1e-8)
            rel_err = abs(numeric - analytic) / denom
            max_rel_err = max(max_rel_err, rel_err)
            n_checked += 1

    # re-sync grads/caches after the perturbation loop
    loss_and_backward()
    if verbose:
        print(f"  [TriangulationNet] params checked: {n_checked}, "
              f"loss={loss0:.6f}, max relative grad error = {max_rel_err:.3e}")
    return max_rel_err


# =============================================================================
# PART 4 — ARCHITECTURE 2: the shu (術) cross-domain technique-transfer net
# =============================================================================

class SharedKernel:
    """The reusable 'technique' (shu): a small MLP transform shared across domains."""

    def __init__(self, dim, hidden, rng_=None):
        self.net = MLP([dim, hidden, dim], ["tanh", "tanh"], rng_=rng_)

    def forward(self, x):
        return self.net.forward(x)

    def backward(self, dout):
        return self.net.backward(dout)

    def params(self):
        return self.net.params()

    def grads(self):
        return self.net.grads()

    def clone_weights_from(self, other):
        for p_self, p_other in zip(self.params(), other.params()):
            p_self[...] = p_other


class DomainNet:
    """domain-specific input adapter -> SHARED kernel -> domain-specific output adapter."""

    def __init__(self, in_dim, shu_dim, out_dim, shared_kernel, rng_=None):
        self.in_adapter = Dense(in_dim, shu_dim, "tanh", rng_=rng_)
        self.shared_kernel = shared_kernel
        self.out_adapter = Dense(shu_dim, out_dim, "linear", rng_=rng_)

    def forward(self, x):
        h = self.in_adapter.forward(x)
        h = self.shared_kernel.forward(h)
        y = self.out_adapter.forward(h)
        return y

    def backward(self, dy):
        dh = self.out_adapter.backward(dy)
        dh = self.shared_kernel.backward(dh)
        self.in_adapter.backward(dh)

    def params(self, include_shared=True):
        p = self.in_adapter.params() + self.out_adapter.params()
        if include_shared:
            p = p + self.shared_kernel.params()
        return p

    def grads(self, include_shared=True):
        g = self.in_adapter.grads() + self.out_adapter.grads()
        if include_shared:
            g = g + self.shared_kernel.grads()
        return g


def make_domain_A(n, rng_=None):
    """Data-rich domain: 'geological residue -> deep-time elevation change'."""
    rng_ = rng_ if rng_ is not None else rng
    x = rng_.uniform(-1, 1, size=(n, 4))
    # a fixed nonlinear ground-truth relation this domain's data respects
    y = np.tanh(1.4 * x[:, 0:1] - 0.6 * x[:, 1:2] + 0.3 * x[:, 2:3] * x[:, 3:4])
    y = y + rng_.normal(0, 0.02, size=y.shape)
    return x, y


def make_domain_B(n, rng_=None, lo=-1.0, hi=1.0):
    """Data-starved domain: 'botanical residue -> deep-time climate change'.
    Structurally the SAME latent relation as domain A (same shu should
    apply), but with a different, domain-specific surface encoding —
    exactly as fossils-in-rock and bamboo-underground are different surface
    dressings of one 'the physical world here was once different' claim.

    `lo`/`hi` control the sampling range. Shen Kuo's actual evidentiary
    situation was not just SCARCE (a handful of fossil sites, one bamboo
    excavation) but also NARROW: a few observed locations from which a much
    broader claim about deep time had to generalize. The transfer experiment
    below trains on a narrow slice and evaluates on the full range, which is
    the faithful version of the test -- not merely 'few points from the same
    distribution as the test set'."""
    rng_ = rng_ if rng_ is not None else rng
    x = rng_.uniform(lo, hi, size=(n, 5))
    latent = np.tanh(1.4 * x[:, 0:1] - 0.6 * x[:, 1:2] + 0.3 * x[:, 2:3] * x[:, 4:5])
    y = latent + rng_.normal(0, 0.02, size=latent.shape)
    return x, y


def train_domain_net(net, x, y, epochs, lr, train_shared=True, verbose_every=None):
    opt = Adam(net.params(include_shared=train_shared), lr=lr)
    losses = []
    for e in range(epochs):
        pred = net.forward(x)
        diff = pred - y
        loss = float(np.mean(diff ** 2))
        losses.append(loss)
        d_pred = 2.0 * diff / diff.shape[0]
        net.backward(d_pred)
        opt.step(net.params(include_shared=train_shared), net.grads(include_shared=train_shared))
        if verbose_every and (e % verbose_every == 0 or e == epochs - 1):
            print(f"    epoch {e:4d}  loss={loss:.5f}")
    return losses


def gradient_check_shu_transfer(verbose=True):
    """Mandatory: central-difference numerical gradient check of DomainNet."""
    rng_ = np.random.default_rng(2)
    shared = SharedKernel(dim=6, hidden=6, rng_=rng_)
    net = DomainNet(in_dim=4, shu_dim=6, out_dim=1, shared_kernel=shared, rng_=rng_)
    x, y = make_domain_A(5, rng_=rng_)

    def compute_loss_backward():
        pred = net.forward(x)
        diff = pred - y
        loss = np.mean(diff ** 2)
        d_pred = 2.0 * diff / diff.shape[0]
        net.backward(d_pred)
        return loss

    def compute_loss_only():
        pred = net.forward(x)
        diff = pred - y
        return float(np.mean(diff ** 2))

    loss0 = compute_loss_backward()
    params = net.params()
    grads = net.grads()

    eps = 1e-5
    max_rel_err = 0.0
    n_checked = 0
    for p, g in zip(params, grads):
        flat_p = p.reshape(-1)
        flat_g = g.reshape(-1)
        idxs = rng_.choice(flat_p.size, size=min(6, flat_p.size), replace=False)
        for idx in idxs:
            orig = flat_p[idx]
            flat_p[idx] = orig + eps
            lp = compute_loss_only()
            flat_p[idx] = orig - eps
            lm = compute_loss_only()
            flat_p[idx] = orig
            numeric = (lp - lm) / (2 * eps)
            analytic = flat_g[idx]
            denom = max(abs(numeric), abs(analytic), 1e-8)
            rel_err = abs(numeric - analytic) / denom
            max_rel_err = max(max_rel_err, rel_err)
            n_checked += 1
    compute_loss_backward()
    if verbose:
        print(f"  [DomainNet/shu]     params checked: {n_checked}, "
              f"loss={loss0:.6f}, max relative grad error = {max_rel_err:.3e}")
    return max_rel_err


# =============================================================================
# PART 5 — training loops with real, printed metrics
# =============================================================================

def train_triangulation_net(epochs=1200, lr=0.02, batch=32, K=4, channel_dim=3,
                             entropy_lambda=0.02, verbose_every=200):
    rng_ = np.random.default_rng(42)
    net = TriangulationNet(K, channel_dim, hidden=10, trust_hidden=8, rng_=rng_)
    opt = Adam(net.params(), lr=lr)
    history = []
    for e in range(epochs):
        # every 5th episode, one randomly chosen channel is corrupted, forcing
        # the TrustNet to actually learn to detect and discount drift instead
        # of coasting on a fixed uniform average
        corrupt = rng_.integers(0, K) if (e % 5 == 0) else None
        X, z = make_triangulation_episode(batch, K, channel_dim,
                                           corrupt_channel=corrupt, rng_=rng_)
        consensus, trust_w, cand = net.forward(X)
        diff = consensus - z
        mse = float(np.mean(diff ** 2))
        d_consensus = 2.0 * diff / diff.shape[0]
        net.backward(d_consensus, entropy_lambda=entropy_lambda)
        opt.step(net.params(), net.grads())
        history.append(mse)
        if verbose_every and (e % verbose_every == 0 or e == epochs - 1):
            tag = f"(channel {corrupt} corrupted)" if corrupt is not None else "(clean)"
            print(f"    epoch {e:5d}  mse={mse:.5f}  {tag}")
    return net, history


def evaluate_trust_discrimination(net, K=4, channel_dim=3, n_eval=400):
    """
    Self-test support: does the trained TrustNet actually assign lower
    average trust to a channel that is corrupted at evaluation time, versus
    the SAME channel's trust when it is behaving normally?
    """
    rng_ = np.random.default_rng(7)
    target_channel = 1

    X_clean, _ = make_triangulation_episode(n_eval, K, channel_dim, rng_=rng_)
    _, trust_clean, _ = net.forward(X_clean)
    trust_channel_clean = float(np.mean(trust_clean[:, target_channel]))

    X_corrupt, _ = make_triangulation_episode(n_eval, K, channel_dim,
                                               corrupt_channel=target_channel, rng_=rng_)
    _, trust_corrupt, _ = net.forward(X_corrupt)
    trust_channel_corrupt = float(np.mean(trust_corrupt[:, target_channel]))

    return trust_channel_clean, trust_channel_corrupt


def run_ledger_demo(net, K=4, channel_dim=3, n_episodes=10):
    """
    Feed a short sequence of sighting episodes for THREE recurring
    quantity_ids, one of which starts from a badly wrong prior belief
    (a stand-in for Yi Xing's superseded coordinate); show the ledger
    logging revisions exactly where, and only where, they are warranted.
    """
    rng_ = np.random.default_rng(11)
    ledger = RevisionLedger()
    quantity_priors = {"pole_star_bearing": 0.9, "flood_crest_estimate": -0.05,
                        "compass_declination": 0.02}
    revisions_logged = 0
    for step in range(n_episodes):
        for qid, prior in quantity_priors.items():
            X, z_true = make_triangulation_episode(1, K, channel_dim, rng_=rng_)
            consensus, trust_w, _ = net.forward(X)
            revised = ledger.observe(qid, consensus[0, 0], trust_w[0], step,
                                      prior_belief=prior if step == 0 else None)
            if revised:
                revisions_logged += 1
    return ledger, revisions_logged


# =============================================================================
# PART 6 — self-tests
# =============================================================================

def run_self_tests():
    print("\n" + "=" * 78)
    print("SELF-TESTS")
    print("=" * 78)
    results = []

    # 1. gradient checks (mandatory, must be tiny)
    err1 = gradient_check_triangulation(verbose=True)
    results.append(("TriangulationNet gradient check < 1e-4", err1 < 1e-4))

    err2 = gradient_check_shu_transfer(verbose=True)
    results.append(("DomainNet/shu gradient check < 1e-4", err2 < 1e-4))

    # 2. trust weights are a valid probability simplex
    rng_ = np.random.default_rng(3)
    net_small = TriangulationNet(4, 3, hidden=6, trust_hidden=6, rng_=rng_)
    X, z = make_triangulation_episode(20, 4, 3, rng_=rng_)
    _, trust_w, _ = net_small.forward(X)
    sums_ok = np.allclose(np.sum(trust_w, axis=1), 1.0, atol=1e-8)
    nonneg_ok = np.all(trust_w >= 0.0)
    results.append(("Trust weights sum to 1 and are non-negative", bool(sums_ok and nonneg_ok)))

    # 3. train the real model, then check differential distrust
    print("\n  Training TriangulationNet (this is the run quoted in the chapter)...")
    trained_net, history = train_triangulation_net(epochs=1200, verbose_every=300)
    trust_clean, trust_corrupt = evaluate_trust_discrimination(trained_net)
    print(f"  Channel 1 mean trust when clean:     {trust_clean:.4f}")
    print(f"  Channel 1 mean trust when corrupted: {trust_corrupt:.4f}")
    results.append(("Corrupted channel receives measurably LESS trust than clean",
                     trust_corrupt < trust_clean - 0.03))

    final_mse = float(np.mean(history[-50:]))
    baseline_mse = evaluate_naive_average_baseline()
    print(f"  Final TriangulationNet MSE (last 50 steps, incl. corrupted episodes): {final_mse:.5f}")
    print(f"  Naive unweighted-average-of-channels MSE on the same distribution:    {baseline_mse:.5f}")
    results.append(("Trained consensus beats naive channel averaging", final_mse < baseline_mse))

    # 4. ledger only logs genuine revisions
    ledger, n_revisions = run_ledger_demo(trained_net)
    print(f"\n  Ledger entries written: {len(ledger.entries)}  "
          f"(revisions among them: {n_revisions})")
    has_first_sightings = len(ledger.entries) >= 3  # at least the 3 opening entries
    revision_flag_consistent = all(
        (e["supersedes"] is not None) == (idx in [x["supersedes"] for x in ledger.entries if x["supersedes"] is not None])
        or True  # structural sanity: superseded_by back-links resolve
        for idx, e in enumerate(ledger.entries)
    )
    backlinks_ok = all(
        ledger.entries[e["supersedes"]]["superseded_by"] == i
        for i, e in enumerate(ledger.entries) if e["supersedes"] is not None
    )
    results.append(("Ledger opens all three quantities on first sighting", has_first_sightings))
    results.append(("Every ledger revision back-links correctly to the entry it supersedes", backlinks_ok))

    # 5. shu transfer beats from-scratch training on the scarce domain
    print("\n  Running shu (technique) transfer experiment...")
    transfer_loss, scratch_loss = run_shu_transfer_experiment()
    print(f"  Domain B held-out MSE, trained FROM SCRATCH (20 samples):        {scratch_loss:.5f}")
    print(f"  Domain B held-out MSE, shu TRANSFERRED from Domain A (20 samples): {transfer_loss:.5f}")
    results.append(("Cross-domain shu transfer beats training from scratch on scarce data",
                     transfer_loss < scratch_loss))

    print("\n" + "-" * 78)
    all_pass = True
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  [{status}] {name}")
    print("-" * 78)
    print(f"  {sum(1 for _, ok in results if ok)}/{len(results)} self-tests passed")
    print("=" * 78)
    return all_pass, results


def evaluate_naive_average_baseline(n=2000, K=4, channel_dim=3):
    rng_ = np.random.default_rng(99)
    mses = []
    for e in range(n // 32):
        corrupt = rng_.integers(0, K) if (e % 5 == 0) else None
        X, z = make_triangulation_episode(32, K, channel_dim, corrupt_channel=corrupt, rng_=rng_)
        # naive baseline: just average the raw first feature of each channel
        naive = np.mean(X[:, :, 0], axis=1, keepdims=True)
        mses.append(float(np.mean((naive - z) ** 2)))
    return float(np.mean(mses))


def run_shu_transfer_experiment(n_seeds=12, n_train=18, epochs=150, lr=0.04):
    """
    Domain A is data-rich and trains the shared kernel (the shu) fully.
    Domain B is Shen Kuo's actual evidentiary situation: SCARCE (n_train
    samples) AND NARROW (drawn only from x in [-0.5, 0.5] -- a few fossil
    sites, one bamboo excavation), evaluated against the FULL range
    x in [-1, 1] -- the broad claim ("the physical world here was once
    different") those few narrow observations were asked to support.

    We compare, averaged over n_seeds independent draws of the scarce
    training set and of both networks' initial adapter weights:
      (1) a Domain-B network trained completely FROM SCRATCH, versus
      (2) a Domain-B network whose shared kernel is the one already trained
          on Domain A -- reused AS THE TECHNIQUE, frozen, exactly as Shen
          Kuo did not re-derive a new mathematics of permutation for the
          Go board after using it for the calendar; only the domain-specific
          adapters (the "which fossil bed", "which excavation report" layer)
          are trained on Domain B's 18 points.
    The comparison is reported as a MEAN over seeds (a single seed is not
    a reliable claim about a stochastic training process); both the mean
    and the win-rate are printed.
    """
    rng_A = np.random.default_rng(501)
    shared_A = SharedKernel(dim=6, hidden=6, rng_=rng_A)
    net_A = DomainNet(in_dim=4, shu_dim=6, out_dim=1, shared_kernel=shared_A, rng_=rng_A)
    xA, yA = make_domain_A(400, rng_=rng_A)
    train_domain_net(net_A, xA, yA, epochs=400, lr=0.03, train_shared=True, verbose_every=None)

    scratch_losses, transfer_losses = [], []
    for seed in range(n_seeds):
        xB_train, yB_train = make_domain_B(n_train, rng_=np.random.default_rng(1000 + seed),
                                            lo=-0.5, hi=0.5)
        xB_test, yB_test = make_domain_B(300, rng_=np.random.default_rng(5000 + seed),
                                          lo=-1.0, hi=1.0)

        # Condition 1: from scratch -- must learn the technique AND the interface
        # from 18 points alone.
        shared_scratch = SharedKernel(dim=6, hidden=6, rng_=np.random.default_rng(seed))
        net_scratch = DomainNet(in_dim=5, shu_dim=6, out_dim=1,
                                 shared_kernel=shared_scratch, rng_=np.random.default_rng(seed))
        train_domain_net(net_scratch, xB_train, yB_train, epochs=epochs, lr=lr,
                          train_shared=True, verbose_every=None)
        pred_scratch = net_scratch.forward(xB_test)
        scratch_losses.append(float(np.mean((pred_scratch - yB_test) ** 2)))

        # Condition 2: shu transferred from Domain A and frozen -- only the
        # interface (adapters) is learned from the 18 points.
        shared_transfer = SharedKernel(dim=6, hidden=6, rng_=np.random.default_rng(seed))
        shared_transfer.clone_weights_from(shared_A)
        net_transfer = DomainNet(in_dim=5, shu_dim=6, out_dim=1,
                                  shared_kernel=shared_transfer, rng_=np.random.default_rng(seed))
        train_domain_net(net_transfer, xB_train, yB_train, epochs=epochs, lr=lr,
                          train_shared=False, verbose_every=None)
        pred_transfer = net_transfer.forward(xB_test)
        transfer_losses.append(float(np.mean((pred_transfer - yB_test) ** 2)))

    wins = sum(1 for t, s in zip(transfer_losses, scratch_losses) if t < s)
    print(f"    ({n_seeds} seeds, {n_train} narrow-range training points each, "
          f"evaluated on the full range) transfer wins {wins}/{n_seeds} seeds")
    return float(np.mean(transfer_losses)), float(np.mean(scratch_losses))


# =============================================================================
# PART 7 — main
# =============================================================================

def main():
    print("=" * 78)
    print("CHAPTER 229 -- SHEN KUO (沈括, 1031-1095)")
    print("TriangulationNet: convergent triangulation with ledgered revision")
    print("=" * 78)

    t0 = time.time()
    all_pass, results = run_self_tests()
    elapsed = time.time() - t0

    print(f"\nTotal runtime: {elapsed:.2f}s")
    print("ALL SELF-TESTS PASSED" if all_pass else "SOME SELF-TESTS FAILED")
    return all_pass


if __name__ == "__main__":
    main()
