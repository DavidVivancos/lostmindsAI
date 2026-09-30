"""
================================================================================
 A REGIONAL-CONSENSUS TRUST NETWORK
 after the method of Abu Ya'la al-Khalil ibn 'Abdallah al-Khalili al-Qazwini
 (c. 977 - 1055 CE), author of "al-Irshad fi Ma'rifat 'Ulama al-Hadith"
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 13 Minds 241 - 260 Available on Amazon https://www.amazon.com/dp/B0HLD1S6QT
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0255_ibrahim_al_khalili_977 - Ibrahim al-Khalili (c.977-1055)
#================================================================================  

WHAT THIS IS

A from-scratch (pure NumPy, no autograd/ML libraries) trainable neural network
whose architecture is built around a specific historical epistemological
method rather than a generic Transformer/attention stack.

Al-Khalili did not build a hadith collection (a set of admitted reports, the
al-Bukhari / Muslim ibn al-Hajjaj project). He built something one level
above that: a biographical register of the hadith SCHOLARS themselves,
"al-Irshad fi Ma'rifat 'Ulama al-Hadith", organised city by city -- Mecca,
Medina, Kufa, Basra, Baghdad, the Persian towns, Khurasan -- so that a
narrator's standing could be read off from (a) what his own city's
community of peers said of him, (b) whether scholars OUTSIDE his city, with
no local stake, said the same thing, and (c) -- weighted above both -- what
al-Khalili had determined himself, from direct audition (sama') or personal
correspondence, when he had it. His own book concedes its own errors; his
method assumed regional communities could be locally partisan (ta'assub)
even while being individually well-informed.

That is a genuinely different epistemic object than "is this one isnad
sound". It is: TRUST AS A PROPERTY OF A PERSON EMBEDDED IN A PLACE, to be
triangulated across places, with direct first-hand contact allowed to
outrank any quantity of second-hand aggregate opinion.

This file implements exactly that as a differentiable architecture:

  1. LOCAL TESTIMONY ENCODER      -- reads raw peer testimony about a
                                      narrator (praise, criticism, internal
                                      consistency, etc).
  2. ATTENTION-POOLED LOCAL
     CONSENSUS                    -- combines multiple testimonies about
                                      the same narrator into one "what his
                                      city says" vector.
  3. REGIONAL BIAS CORRECTION     -- a learned, per-region correction term
                                      that models and removes systematic
                                      local partisanship (ta'assub) BEFORE
                                      that opinion is allowed to travel.
  4. TRANSLOCAL (CROSS-REGION
     ONLY) COROBORATION
     PROPAGATION                  -- a graph-message-passing step that
                                      lets a narrator's corrected local
                                      standing be corroborated by citations
                                      arriving from OTHER cities only --
                                      same-city citations are structurally
                                      excluded, because they cannot supply
                                      independent confirmation of a
                                      same-city bias.
  5. CHRONOLOGICAL / GEOGRAPHIC
     CONSISTENCY FILTER           -- a rule-based (non-learned) pre-filter
                                      on the citation graph itself, which
                                      drops any transmission edge that is
                                      chronologically impossible (a
                                      narrator "citing" someone from a
                                      later generation) -- exactly the kind
                                      of tabaqat cross-check classical
                                      critics used to catch fabricated
                                      chains.
  6. DIRECT-ENCOUNTER PRECEDENCE
     GATE                         -- for the minority of narrators
                                      al-Khalili met or corresponded with
                                      directly, a learned gate blends in
                                      that low-noise first-hand channel,
                                      structurally forced to zero for
                                      everyone else.
  7. READOUT                      -- final reliability estimate in [0, 1].

None of this is a Transformer, an MoE, or attention-over-a-KV-store; the
"attention" here is a small, literal within-narrator pooling step (step 2),
not the propagation mechanism. Propagation is graph message-passing subject
to hard structural constraints (steps 4 and 5) that encode al-Khalili's
own two epistemic rules: prize independence over volume, and let contact
outrank hearsay.

WHAT THE DATA IS

Real large-scale hadith-critical datasets (al-Jarh wa'l-Ta'dil corpora, the
generational tabaqat literature) are not available in structured, machine
-readable form and re-deriving them from primary texts is outside what a
single script can responsibly claim to do. So the world here is
synthetic-but-faithful: a population of narrators with a hidden true
reliability, grouped into regions with a hidden systematic bias per
region, observed only through noisy multi-source testimony plus a sparse
cross-region citation graph plus a small number of first-hand encounters.
The architecture's job is to recover the hidden reliability from that noisy,
partisan, partially-connected evidence -- which is exactly al-Khalili's own
problem, restated as a learning problem.

Every layer below is implemented with a hand-written forward AND backward
pass through a small reverse-mode autodiff engine (also written from
scratch in this file, no autograd/torch/jax). A finite-difference gradient
check is run over every parameter group before training, and a battery of
self-tests probes not just "does loss go down" but "does the mechanism
that is supposed to matter actually matter" (see SELF-TESTS at the bottom).
================================================================================
"""

import numpy as np

RNG_SEED = 11
rng = np.random.default_rng(RNG_SEED)


# ==============================================================================
# PART 1 -- MINIMAL REVERSE-MODE AUTODIFF ENGINE (pure NumPy)
# ==============================================================================

def _unbroadcast(grad, shape):
    """Sum-reduce `grad` down to `shape` after a NumPy-broadcasted op."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad.reshape(shape)


class Tensor:
    """A NumPy array with a gradient and a backward closure. Deliberately
    minimal: only the operations this model actually needs are implemented,
    each with a hand-verified backward rule. Composite operations (softmax,
    division, sigmoid via primitives, etc.) are built OUT OF these, so a
    single working set of primitives is enough to make the whole model
    differentiable end to end."""

    __slots__ = ("data", "grad", "requires_grad", "_backward", "_prev", "_op")

    def __init__(self, data, requires_grad=False, _children=(), _op=""):
        self.data = np.asarray(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    @property
    def shape(self):
        return self.data.shape

    @staticmethod
    def const(data):
        """Wrap a plain array as a non-trainable (leaf, no-grad) Tensor."""
        return Tensor(data, requires_grad=False)

    def zero_grad(self):
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)

    # ---- elementwise / linear algebra ops -----------------------------
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        req = self.requires_grad or other.requires_grad
        out = Tensor(self.data + other.data, req, (self, other), "+")

        def _backward():
            if self.requires_grad:
                self.grad += _unbroadcast(out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(out.grad, other.data.shape)
        out._backward = _backward
        return out

    __radd__ = __add__

    def __neg__(self):
        out = Tensor(-self.data, self.requires_grad, (self,), "neg")

        def _backward():
            if self.requires_grad:
                self.grad += -out.grad
        out._backward = _backward
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + (-other)

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        req = self.requires_grad or other.requires_grad
        out = Tensor(self.data * other.data, req, (self, other), "*")

        def _backward():
            if self.requires_grad:
                self.grad += _unbroadcast(out.grad * other.data, self.data.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(out.grad * self.data, other.data.shape)
        out._backward = _backward
        return out

    __rmul__ = __mul__

    def reciprocal(self):
        r = 1.0 / self.data
        out = Tensor(r, self.requires_grad, (self,), "recip")

        def _backward():
            if self.requires_grad:
                self.grad += out.grad * (-(r ** 2))
        out._backward = _backward
        return out

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self * other.reciprocal()

    def matmul(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        req = self.requires_grad or other.requires_grad
        out = Tensor(self.data @ other.data, req, (self, other), "matmul")

        def _backward():
            if self.requires_grad:
                self.grad += out.grad @ other.data.T
            if other.requires_grad:
                other.grad += self.data.T @ out.grad
        out._backward = _backward
        return out

    # ---- nonlinearities -------------------------------------------------
    def relu(self):
        mask = (self.data > 0).astype(np.float64)
        out = Tensor(self.data * mask, self.requires_grad, (self,), "relu")

        def _backward():
            if self.requires_grad:
                self.grad += out.grad * mask
        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, self.requires_grad, (self,), "tanh")

        def _backward():
            if self.requires_grad:
                self.grad += out.grad * (1.0 - t * t)
        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + np.exp(-self.data))
        out = Tensor(s, self.requires_grad, (self,), "sigmoid")

        def _backward():
            if self.requires_grad:
                self.grad += out.grad * s * (1.0 - s)
        out._backward = _backward
        return out

    def exp(self):
        e = np.exp(self.data)
        out = Tensor(e, self.requires_grad, (self,), "exp")

        def _backward():
            if self.requires_grad:
                self.grad += out.grad * e
        out._backward = _backward
        return out

    # ---- reductions -------------------------------------------------------
    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), self.requires_grad, (self,), "sum")

        def _backward():
            if self.requires_grad:
                g = out.grad
                if not keepdims and axis is not None:
                    g = np.expand_dims(g, axis)
                self.grad += np.ones_like(self.data) * g
        out._backward = _backward
        return out

    def mean(self):
        n = self.data.size
        out = Tensor(self.data.mean(), self.requires_grad, (self,), "mean")

        def _backward():
            if self.requires_grad:
                self.grad += np.ones_like(self.data) * (out.grad / n)
        out._backward = _backward
        return out

    # ---- indexing / structural ops ----------------------------------------
    def col(self, t):
        """Select column t as an (N,1) tensor. Correctly differentiable
        (accumulates directly into self.grad -- no closure-over-loop-
        variable bugs, which is a classically easy mistake to make here)."""
        out = Tensor(self.data[:, t:t + 1], self.requires_grad, (self,), "col")

        def _backward():
            if self.requires_grad:
                self.grad[:, t:t + 1] += out.grad
        out._backward = _backward
        return out

    def gather_rows(self, idx):
        """out[i] = self[idx[i]]; gradient scatter-adds back (handles
        repeated indices correctly, e.g. many narrators sharing one
        region's bias row)."""
        idx = np.asarray(idx)
        out = Tensor(self.data[idx], self.requires_grad, (self,), "gather")

        def _backward():
            if self.requires_grad:
                np.add.at(self.grad, idx, out.grad)
        out._backward = _backward
        return out

    @staticmethod
    def concat(tensors, axis=1):
        req = any(t.requires_grad for t in tensors)
        data = np.concatenate([t.data for t in tensors], axis=axis)
        out = Tensor(data, req, tuple(tensors), "concat")
        sizes = [t.data.shape[axis] for t in tensors]

        def _backward():
            idx = 0
            for t, s in zip(tensors, sizes):
                if t.requires_grad:
                    sl = [slice(None)] * data.ndim
                    sl[axis] = slice(idx, idx + s)
                    t.grad += out.grad[tuple(sl)]
                idx += s
        out._backward = _backward
        return out

    # ---- graph traversal ----------------------------------------------
    def backward(self):
        topo, visited = [], set()

        def build(v):
            if id(v) not in visited:
                visited.add(id(v))
                for child in v._prev:
                    build(child)
                topo.append(v)
        build(self)
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()


# ==============================================================================
# PART 2 -- SYNTHETIC WORLD: narrators, regions, testimony, citation graph
# ==============================================================================

N = 80            # number of narrators under evaluation
R = 6              # number of regions / cities
T = 4              # testimonies (peer judgments) observed per narrator
F = 6              # raw testimony feature dimension
H = 16             # hidden width
K_HOPS = 2         # translocal propagation hops
REGION_NAMES = ["Qazwin", "Baghdad", "Kufa", "Basra", "Rayy", "Isfahan"]


def build_world(seed=11):
    g = np.random.default_rng(seed)

    region_id = g.integers(0, R, size=N)
    generation = g.integers(1, 6, size=N)          # tabaqa (generation) 1..5
    true_reliability = g.beta(2, 2, size=N)         # hidden ground truth in [0,1]
    region_bias_true = g.normal(0, 0.18, size=R)    # hidden per-region ta'assub

    # testimony_feats[i, t] = [praise, criticism, consistency,
    #                          local_corroboration_count, recency, noise]
    testimony_feats = np.zeros((N, T, F))
    for i in range(N):
        for t in range(T):
            base = np.clip(true_reliability[i] + region_bias_true[region_id[i]]
                            + g.normal(0, 0.12), 0, 1)
            testimony_feats[i, t] = [
                base + g.normal(0, 0.05),
                (1 - base) + g.normal(0, 0.05),
                base + g.normal(0, 0.05),
                g.uniform(0, 1),
                g.uniform(0, 1),
                g.normal(0, 0.3),
            ]

    # a minority of narrators al-Khalili met or corresponded with directly
    direct_flag = np.zeros((N, 1))
    direct_idx = g.choice(N, size=12, replace=False)
    direct_flag[direct_idx, 0] = 1.0
    direct_feats = np.zeros((N, F))
    for i in direct_idx:
        base = np.clip(true_reliability[i] + g.normal(0, 0.03), 0, 1)  # low-noise
        direct_feats[i] = [base, 1 - base, base, 1.0, 1.0, g.normal(0, 0.05)]

    # sparse cross-region citation graph (translocal corroboration edges)
    p_edge = 0.06
    A = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j and region_id[i] != region_id[j] and g.random() < p_edge:
                A[i, j] = 1.0

    # deliberately inject a handful of chronologically IMPOSSIBLE edges, to
    # verify the consistency filter actually removes them (self-test below)
    impossible_pairs = []
    tries = 0
    while len(impossible_pairs) < 5 and tries < 4000:
        i, j = g.integers(0, N, size=2)
        tries += 1
        if i != j and region_id[i] != region_id[j] and generation[j] < generation[i]:
            A[i, j] = 1.0
            impossible_pairs.append((i, j))

    return dict(region_id=region_id, generation=generation,
                true_reliability=true_reliability, region_bias_true=region_bias_true,
                testimony_feats=testimony_feats, direct_flag=direct_flag,
                direct_feats=direct_feats, A=A, impossible_pairs=impossible_pairs)


def chronologically_consistent_adjacency(A_raw, generation):
    """Rule-based (non-learned) filter: retain a citation edge i<-j only if
    the cited narrator j belongs to a generation at least as early as i's.
    This is a direct computational analogue of the tabaqat cross-check
    classical hadith critics used to catch impossible transmission claims
    -- e.g. a narrator "meeting" a teacher who died before he was born."""
    A_clean = A_raw.copy()
    gi = generation.reshape(-1, 1)
    gj = generation.reshape(1, -1)
    impossible = gj < gi
    A_clean[impossible & (A_raw > 0)] = 0.0
    return A_clean


def normalize_adjacency(A_clean):
    deg = A_clean.sum(axis=1, keepdims=True)
    return A_clean / np.where(deg == 0, 1.0, deg)


# ==============================================================================
# PART 3 -- THE MODEL
# ==============================================================================

def init_w(shape, scale=0.35):
    return Tensor(rng.normal(0, scale, size=shape), requires_grad=True)


def init_b(shape):
    # small random init (not exact zero): avoids landing bias terms exactly
    # on the ReLU non-differentiable kink for degree-0 graph nodes, which
    # would otherwise be a spurious finite-difference discrepancy at t=0.
    return Tensor(rng.normal(0, 0.02, size=shape), requires_grad=True)


def fresh_params():
    p = {}
    p["W1"] = init_w((F, H)); p["b1"] = init_b((H,))          # testimony encoder
    p["Wa"] = init_w((H, H)); p["va"] = init_w((H, 1))         # attention scorer
    p["RegionBias"] = init_w((R, H), scale=0.15)               # ta'assub correction
    p["Wp"] = init_w((H, H)); p["bp"] = init_b((H,))           # propagation transform
    p["Wd"] = init_w((F, H)); p["bd"] = init_b((H,))           # direct-encounter encoder
    p["Wg"] = init_w((H, 1)); p["bg"] = init_b((1,))           # gate
    p["Wr"] = init_w((H, 1)); p["br"] = init_b((1,))           # readout
    return p


def forward(p, world, A_norm_t, use_bias_correction=True):
    testimony_feats = world["testimony_feats"]
    region_id = world["region_id"]
    direct_feats = world["direct_feats"]
    direct_flag = world["direct_flag"]

    # -- 1. per-testimony encoding (shared weights across the T "channels")
    emb_list = []
    for t in range(T):
        xt = Tensor.const(testimony_feats[:, t, :])
        et = (xt.matmul(p["W1"]) + p["b1"]).relu()
        emb_list.append(et)

    # -- 2. attention-pooled local consensus over the T testimonies
    scores = [(et.matmul(p["Wa"])).tanh().matmul(p["va"]) for et in emb_list]  # T x (N,1)
    S = Tensor.concat(scores, axis=1)                                          # (N,T)
    S_shift = S - Tensor.const(S.data.max(axis=1, keepdims=True))              # stability
    Sexp = S_shift.exp()
    Ssum = Sexp.sum(axis=1, keepdims=True)
    weights = Sexp / Ssum                                                      # softmax (N,T)

    local_consensus = None
    for t in range(T):
        wt = weights.col(t)
        contrib = wt * emb_list[t]
        local_consensus = contrib if local_consensus is None else local_consensus + contrib

    # -- 3. regional bias correction (ta'assub)
    region_bias_vec = p["RegionBias"].gather_rows(region_id)
    local_corrected = local_consensus - region_bias_vec if use_bias_correction else local_consensus

    # -- 4. translocal (cross-region-only) corroboration propagation
    h = local_corrected
    for _ in range(K_HOPS):
        msg = A_norm_t.matmul(h)
        h = h + (msg.matmul(p["Wp"]) + p["bp"]).relu()

    # -- 5. direct-encounter precedence gate (structurally 0 unless al-Khalili
    #       actually met this narrator -- 'direct_flag' is a hard mask, not
    #       something the network can learn its way around)
    d_emb = (Tensor.const(direct_feats).matmul(p["Wd"]) + p["bd"]).relu()
    gate_logit = h.matmul(p["Wg"]) + p["bg"]
    gate_raw = gate_logit.sigmoid()
    gate = gate_raw * Tensor.const(direct_flag)
    one_minus_gate = Tensor.const(np.ones_like(gate.data)) - gate
    final_repr = gate * d_emb + one_minus_gate * h

    # -- 6. readout: final reliability estimate in [0,1]
    pred = (final_repr.matmul(p["Wr"]) + p["br"]).sigmoid()
    return pred, gate


def mse_loss(pred, y_idx, true_reliability):
    y = Tensor.const(true_reliability[y_idx].reshape(-1, 1))
    diff = pred.gather_rows(y_idx) - y
    return (diff * diff).mean()


# ==============================================================================
# PART 4 -- TRAINING
# ==============================================================================

def sgd_step(params, lr):
    for p in params.values():
        p.data -= lr * p.grad
        p.zero_grad()


def train(world, A_norm_t, epochs=400, lr=0.15, use_bias_correction=True, seed=None):
    global rng
    if seed is not None:
        rng = np.random.default_rng(seed)
    params = fresh_params()
    train_idx, test_idx = world["train_idx"], world["test_idx"]
    losses = []
    for _ in range(epochs):
        pred, gate = forward(params, world, A_norm_t, use_bias_correction)
        loss = mse_loss(pred, train_idx, world["true_reliability"])
        loss.backward()
        sgd_step(params, lr)
        losses.append(float(loss.data))
    pred_final, gate_final = forward(params, world, A_norm_t, use_bias_correction)
    return params, losses, pred_final, gate_final


def pearson(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum()) * np.sqrt((b * b).sum()) + 1e-12))


# ==============================================================================
# PART 5 -- FINITE-DIFFERENCE GRADIENT CHECK
# ==============================================================================

def gradient_check(world, A_norm_t, n_per_param=4, eps=1e-5, tol=2e-3):
    params = fresh_params()
    pred, gate = forward(params, world, A_norm_t)
    loss = mse_loss(pred, world["train_idx"], world["true_reliability"])
    loss.backward()

    results = []
    for name, p in params.items():
        idxs = [tuple(rng.integers(0, s) for s in p.data.shape) for _ in range(n_per_param)]
        for idx in idxs:
            orig = p.data[idx]
            p.data[idx] = orig + eps
            l1 = mse_loss(forward(params, world, A_norm_t)[0], world["train_idx"], world["true_reliability"]).data
            p.data[idx] = orig - eps
            l2 = mse_loss(forward(params, world, A_norm_t)[0], world["train_idx"], world["true_reliability"]).data
            p.data[idx] = orig
            numeric = (l1 - l2) / (2 * eps)
            analytic = p.grad[idx]
            rel = abs(numeric - analytic) / (abs(numeric) + abs(analytic) + 1e-8)
            results.append((name, idx, numeric, analytic, rel))
    max_rel = max(r[4] for r in results)
    passed = all(r[4] < tol for r in results)
    return passed, max_rel, results


# ==============================================================================
# PART 6 -- SELF-TESTS
# ==============================================================================

def run_self_tests(verbose=True):
    world = build_world(seed=RNG_SEED)
    A_filtered = chronologically_consistent_adjacency(world["A"], world["generation"])
    A_norm = normalize_adjacency(A_filtered)
    A_norm_t = Tensor.const(A_norm)

    perm = np.random.default_rng(3).permutation(N)
    n_train = int(0.8 * N)
    world["train_idx"], world["test_idx"] = perm[:n_train], perm[n_train:]

    results = {}

    # --- test 1: gradient check -----------------------------------------
    passed, max_rel, _ = gradient_check(world, A_norm_t)
    results["gradient_check_passes"] = passed
    if verbose:
        print(f"[1] gradient check: {'PASS' if passed else 'FAIL'} (max relative error {max_rel:.2e})")

    # --- test 2: chronological/geographic consistency filter -------------
    all_impossible_removed = all(
        A_filtered[i, j] == 0.0 for (i, j) in world["impossible_pairs"]
    )
    results["chronology_filter_removes_impossible_edges"] = all_impossible_removed
    if verbose:
        print(f"[2] chronology filter drops all {len(world['impossible_pairs'])} injected "
              f"impossible edges: {'PASS' if all_impossible_removed else 'FAIL'}")

    # --- test 3: training reduces loss substantially ----------------------
    params, losses, pred_final, gate_final = train(world, A_norm_t, epochs=400, lr=0.15, seed=RNG_SEED)
    loss_shrunk = losses[-1] < 0.3 * losses[0]
    results["training_loss_decreases"] = loss_shrunk
    if verbose:
        print(f"[3] training loss {losses[0]:.4f} -> {losses[-1]:.4f} "
              f"({'PASS' if loss_shrunk else 'FAIL'}, target < 30% of initial)")

    # --- test 4: held-out correlation improves over an untrained network --
    untrained_params = fresh_params()
    pred0, _ = forward(untrained_params, world, A_norm_t)
    corr_before = pearson(pred0.data[world["test_idx"], 0], world["true_reliability"][world["test_idx"]])
    corr_after = pearson(pred_final.data[world["test_idx"], 0], world["true_reliability"][world["test_idx"]])
    corr_improves = corr_after > corr_before + 0.3
    results["held_out_correlation_improves"] = corr_improves
    if verbose:
        print(f"[4] held-out correlation with true reliability: {corr_before:.3f} -> {corr_after:.3f} "
              f"({'PASS' if corr_improves else 'FAIL'})")

    # --- test 5: regional bias correction is load-bearing (ablation) ------
    bias_mag = np.abs(world["region_bias_true"])
    high_bias_regions = np.argsort(-bias_mag)[:3]
    mask_high = np.isin(world["region_id"], high_bias_regions)
    test_high = world["test_idx"][np.isin(world["test_idx"], np.where(mask_high)[0])]

    _, _, pred_with, _ = train(world, A_norm_t, epochs=400, lr=0.15, use_bias_correction=True, seed=RNG_SEED)
    _, _, pred_without, _ = train(world, A_norm_t, epochs=400, lr=0.15, use_bias_correction=False, seed=RNG_SEED)
    err_with = float(np.mean((pred_with.data[test_high, 0] - world["true_reliability"][test_high]) ** 2))
    err_without = float(np.mean((pred_without.data[test_high, 0] - world["true_reliability"][test_high]) ** 2))
    bias_correction_helps = err_without > err_with
    results["regional_bias_correction_helps"] = bias_correction_helps
    if verbose:
        print(f"[5] MSE on high-bias-region held-out narrators: with correction {err_with:.4f}, "
              f"without {err_without:.4f} ({'PASS' if bias_correction_helps else 'FAIL'})")

    # --- test 6: direct-encounter gate behaves structurally as designed ---
    g_final = gate_final.data[:, 0]
    direct_mask = world["direct_flag"][:, 0] > 0
    non_direct_zero = np.allclose(g_final[~direct_mask], 0.0)
    direct_engaged = g_final[direct_mask].mean() > 0.05
    gate_ok = non_direct_zero and direct_engaged
    results["direct_encounter_gate_structural"] = gate_ok
    if verbose:
        print(f"[6] direct-encounter gate: non-direct forced to 0 = {non_direct_zero}, "
              f"mean gate for direct narrators = {g_final[direct_mask].mean():.3f} "
              f"({'PASS' if gate_ok else 'FAIL'})")

    all_passed = all(results.values())
    if verbose:
        print()
        print("ALL SELF-TESTS PASSED" if all_passed else "SOME SELF-TESTS FAILED")
    return all_passed, results


if __name__ == "__main__":
    run_self_tests(verbose=True)
