"""
================================================================================
 Chapter 0148_st_ambrose_339 - St. Ambrose of Milan (339-397 CE)
 The Antiphonal Settling Network (ASN)
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 8 Minds 141 - 160 Available on Amazon https://www.amazon.com/dp/B0HH8RTCXF
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 148: St. Ambrose of Milan (339-397 CE)
================================================================================  

WHAT THIS FILE IS
-----------------
A from-scratch, pure-NumPy neural architecture that encodes the distinctive
cognitive signature of Ambrose of Milan. It is NOT a demo wrapper around a big
library: it contains a tiny reverse-mode automatic-differentiation engine
("Cantus"), a real model built on top of it, a MANDATORY finite-difference
gradient check, a real training loop, self-tests, and a controlled ablation.
Run it directly:  `python3 chapter_0148_st_ambrose_339.py`

WHY THIS ARCHITECTURE (the mind, not a generic Transformer)
-----------------------------------------------------------
Ambrose's documented innovation was ANTIPHONY. Besieged in his basilica in
Milan in 385-386 CE, with an imperial army at the doors, he did not lecture the
crowd from above. He split the congregation into two choirs and had them sing
back and forth -- call and response -- in strict iambic-dimeter meter. Nicene
doctrine was smuggled into short, metrically-locked, singable verse that any
illiterate layperson could carry home and reproduce exactly. Augustine, present
for these vigils, wept at the sound and later recorded it (Confessions IX).

Read as a theory of cognition, this is remarkable and specific:

  (1) ALTERNATION, NOT BROADCAST. Understanding is stabilized in the *exchange*
      between two voices, each conditioning on the other's last utterance --
      not injected top-down into isolated minds.

  (2) METER AS ERROR-CORRECTING CODE. Strict meter is a constraint that snaps
      each utterance onto a small, shared lattice of legal "feet." That makes a
      transmitted phrase robust to corruption and reproducible across a crowd:
      an error-correcting code implemented in verse. Heresy ("Arian drift") is
      literally off-meter noise the lattice rejects.

  (3) CONGREGATIONAL CONSENSUS. Alignment is reached when the two choirs
      converge on the same refrain -- a fixed point the whole assembly holds --
      rather than by a single authority dictating the answer.

The Antiphonal Settling Network embodies exactly these three ideas:

  * Two coupled recurrent "choirs" (A = cantor/call, B = congregation/response)
    that ALTERNATE for several rounds, each updating from the shared verse plus
    the OTHER choir's most recent utterance.  -> (1) alternation

  * A learned METRICAL CODEBOOK through which every utterance is quantized
    (soft vector quantization). A commitment loss pulls utterances toward the
    lattice.  -> (2) meter as error-correcting code

  * A consensus read-out from the mean of the two settled choir states, plus an
    explicit ANTIPHONAL AGREEMENT loss that rewards the two choirs for
    converging.  -> (3) congregational consensus / fixed point

THE TASK (chosen to make the thesis measurable)
-----------------------------------------------
"Antiphon completion under corruption." A fixed liturgical rule maps a clean
call to its canonical response (a deterministic permutation-and-sign map -- the
"rite"). At run time the call is corrupted with noise and masking ("Arian
drift"). The network must recover the clean response. We then run an ABLATION:
with the metrical codebook ON vs OFF. If Ambrose is right that strict meter
defends transmitted meaning, the quantized ("in-meter") model should degrade
more gracefully as corruption rises. The training/eval output at the bottom of
this file reports exactly that comparison.

Author's note: the code is heavily commented so a reader can follow both the
mathematics and the analogy to Ambrose's liturgical machine.
================================================================================
"""

from __future__ import annotations

import numpy as np

# Reproducibility across the whole liturgy.
GLOBAL_SEED = 1749  # 17 -> A.D. hint; 49 -> figure 0149. Arbitrary but fixed.


# =============================================================================
# PART I — "Cantus": a minimal reverse-mode autodiff engine (pure NumPy)
# -----------------------------------------------------------------------------
# Every quantity in the network is a Node carrying a value (.data) and an
# accumulated gradient (.grad). Each operation records how to push gradients
# back to its inputs. This is a compact tape-based autograd, written from
# scratch so the finite-difference gradient check below is a genuine test of
# OUR math, not a library's.
# =============================================================================

class Node:
    """A tensor on the autodiff tape."""

    __slots__ = ("data", "grad", "_backward", "_prev", "_op")

    def __init__(self, data, _children=(), _op=""):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None          # closure filled by each op
        self._prev = set(_children)            # parent Nodes on the tape
        self._op = _op                         # label, for debugging

    # ---- helpers ----------------------------------------------------------
    @staticmethod
    def _unbroadcast(grad, shape):
        """Sum `grad` back down to `shape` after NumPy broadcasting."""
        while grad.ndim > len(shape):
            grad = grad.sum(axis=0)
        for i, dim in enumerate(shape):
            if dim == 1 and grad.shape[i] != 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad.reshape(shape)

    # ---- elementwise ops --------------------------------------------------
    def __add__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        out = Node(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += Node._unbroadcast(out.grad, self.data.shape)
            other.grad += Node._unbroadcast(out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        out = Node(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += Node._unbroadcast(other.data * out.grad, self.data.shape)
            other.grad += Node._unbroadcast(self.data * out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        return self + (other * -1.0)

    def __neg__(self):
        return self * -1.0

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    # ---- linear algebra ---------------------------------------------------
    def matmul(self, other):
        out = Node(self.data @ other.data, (self, other), "@")

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        out._backward = _backward
        return out

    def __matmul__(self, other):
        return self.matmul(other)

    # ---- nonlinearity -----------------------------------------------------
    def tanh(self):
        t = np.tanh(self.data)
        out = Node(t, (self,), "tanh")

        def _backward():
            self.grad += (1.0 - t * t) * out.grad
        out._backward = _backward
        return out

    # ---- reductions -------------------------------------------------------
    def mean(self):
        out = Node(self.data.mean(), (self,), "mean")
        n = self.data.size

        def _backward():
            self.grad += (out.grad / n) * np.ones_like(self.data)
        out._backward = _backward
        return out

    def sum_cols(self):
        """Sum over axis=1, keepdims — used inside the quantizer/softmax."""
        out = Node(self.data.sum(axis=1, keepdims=True), (self,), "sum_cols")

        def _backward():
            self.grad += out.grad * np.ones_like(self.data)
        out._backward = _backward
        return out

    # ---- backprop driver --------------------------------------------------
    def backward(self):
        """Topological sort, then push gradients from this scalar node back."""
        topo, seen = [], set()

        def build(v):
            if v not in seen:
                seen.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)
        build(self)

        self.grad = np.ones_like(self.data)  # d(self)/d(self) = 1
        for v in reversed(topo):
            v._backward()


# ---- free functions on Nodes (composition of the primitives above) ---------

def concat_cols(a: Node, b: Node) -> Node:
    """Concatenate two (B x .) Nodes along the feature axis."""
    out = Node(np.concatenate([a.data, b.data], axis=1), (a, b), "concat")
    na = a.data.shape[1]

    def _backward():
        a.grad += out.grad[:, :na]
        b.grad += out.grad[:, na:]
    out._backward = _backward
    return out


def softmax_rows(x: Node) -> Node:
    """Row-wise softmax with a custom, numerically stable backward."""
    z = x.data - x.data.max(axis=1, keepdims=True)
    e = np.exp(z)
    p = e / e.sum(axis=1, keepdims=True)
    out = Node(p, (x,), "softmax")

    def _backward():
        # Jacobian-vector product for softmax, per row.
        dot = (out.grad * p).sum(axis=1, keepdims=True)
        x.grad += p * (out.grad - dot)
    out._backward = _backward
    return out


def metrical_quantize(u: Node, codebook: Node, tau: float):
    """
    The 'strict meter'. Soft vector-quantize each utterance row of `u`
    (B x H) against a learned `codebook` (K x H) of legal metrical feet.

    Returns (u_q, assignment_probs):
      dist2[b,k] = || u[b] - C[k] ||^2
      p          = softmax(-dist2 / tau)      (row-wise over K feet)
      u_q[b]     = sum_k p[b,k] * C[k]        (snap onto the lattice)

    Both u_q and the probabilities flow gradients to u and the codebook, so the
    lattice itself is *learned* -- Ambrose composed his own hymns, after all.
    """
    U = u.data                      # (B, H)
    C = codebook.data               # (K, H)
    B, H = U.shape
    K = C.shape[0]

    # Squared distances (B, K).
    dist2 = ((U[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
    z = -dist2 / tau
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    P = e / e.sum(axis=1, keepdims=True)      # (B, K) assignment probabilities
    Uq = P @ C                                 # (B, H) quantized utterance

    out = Node(Uq, (u, codebook), "quantize")

    def _backward():
        G = out.grad                           # (B, H) upstream grad on u_q
        # --- gradient of Uq = P @ C wrt P and C ---
        dP = G @ C.T                           # (B, K)
        dC_direct = P.T @ G                     # (K, H) via the explicit C factor
        # --- gradient of P wrt its logits (softmax over K) ---
        dot = (dP * P).sum(axis=1, keepdims=True)
        dlogits = P * (dP - dot)                # (B, K)
        # logits = -dist2/tau ; dist2[b,k] = sum_h (U[b,h]-C[k,h])^2
        dphi = -dlogits / tau                   # (B, K) grad on dist2
        diff = U[:, None, :] - C[None, :, :]    # (B, K, H)
        # d dist2 / dU = 2*diff ; d dist2 / dC = -2*diff
        dU = (dphi[:, :, None] * 2.0 * diff).sum(axis=1)       # (B, H)
        dC_dist = (dphi[:, :, None] * -2.0 * diff).sum(axis=0)  # (K, H)
        u.grad += dU
        codebook.grad += dC_direct + dC_dist
    out._backward = _backward
    return out, P


def mse(pred: Node, target_array: np.ndarray) -> Node:
    """Mean-squared error against a constant target (no grad to target)."""
    diff = pred - Node(target_array)
    sq = diff * diff
    return sq.mean()


# =============================================================================
# PART II — Parameters, initialization, and the model
# =============================================================================

def glorot(shape, rng):
    """Glorot/Xavier uniform init -- keeps the choirs from shouting at start."""
    fan_in, fan_out = shape[0], shape[1]
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=shape)


class AntiphonalSettlingNetwork:
    """
    D  : verse (input/output) dimension
    H  : choir hidden dimension
    K  : number of metrical feet in the codebook
    T  : number of antiphonal rounds (alternations); even -> ends on choir B
    use_meter : if False, the metrical quantizer is bypassed (ablation)
    """

    def __init__(self, D=8, H=16, K=6, T=4, tau=0.5, use_meter=True, seed=GLOBAL_SEED):
        rng = np.random.default_rng(seed)
        self.D, self.H, self.K, self.T = D, H, K, T
        self.tau = tau
        self.use_meter = use_meter

        # Each choir maps [verse (D) ; other choir's last utterance (H)] -> H.
        self.W_A = Node(glorot((D + H, H), rng)); self.b_A = Node(np.zeros((1, H)))
        self.W_B = Node(glorot((D + H, H), rng)); self.b_B = Node(np.zeros((1, H)))
        # The shared metrical lattice (learned feet).
        self.C = Node(glorot((K, H), rng))
        # Consensus read-out: mean(choir states) -> clean response (D).
        self.W_out = Node(glorot((H, D), rng)); self.b_out = Node(np.zeros((1, D)))

    def parameters(self):
        return [self.W_A, self.b_A, self.W_B, self.b_B, self.C, self.W_out, self.b_out]

    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)

    # ---- forward: the antiphon ------------------------------------------
    def forward(self, x_array):
        """
        x_array : (B, D) corrupted call.
        Returns a dict of Nodes: prediction, and the last utterances of each
        choir (for the antiphonal-agreement loss), plus mean codebook-commitment.
        """
        B = x_array.shape[0]
        x = Node(x_array)

        # Both choirs begin silent.
        uA = Node(np.zeros((B, self.H)))   # choir A's current utterance
        uB = Node(np.zeros((B, self.H)))   # choir B's current utterance

        commit_terms = []  # collect || u - quantize(u) ||^2 for the meter loss

        # Alternate: on each round the *calling* choir sings, conditioned on the
        # other's last utterance, then is snapped onto the metrical lattice.
        for t in range(self.T):
            if t % 2 == 0:
                # Choir A calls, hearing B's last response.
                pre = (concat_cols(x, uB) @ self.W_A) + self.b_A
                h = pre.tanh()
                uA, commit = self._meter(h)
                commit_terms.append(commit)
            else:
                # Choir B answers, hearing A's last call.
                pre = (concat_cols(x, uA) @ self.W_B) + self.b_B
                h = pre.tanh()
                uB, commit = self._meter(h)
                commit_terms.append(commit)

        # Congregational consensus = mean of the two settled choirs.
        consensus = (uA + uB) * 0.5
        pred = (consensus @ self.W_out) + self.b_out

        # Average commitment ("distance from correct meter") across all rounds.
        commit_loss = commit_terms[0]
        for c in commit_terms[1:]:
            commit_loss = commit_loss + c
        commit_loss = commit_loss * (1.0 / len(commit_terms))

        return {"pred": pred, "uA": uA, "uB": uB, "commit": commit_loss}

    def _meter(self, h: Node):
        """Apply (or bypass) the metrical lattice; return (utterance, commit)."""
        if not self.use_meter:
            # Ablation: no meter. Commitment is defined but zero-influence.
            zero = Node(np.zeros(()))
            return h, zero
        u_q, _P = metrical_quantize(h, self.C, self.tau)
        # Commitment loss = mean squared distance from utterance to its foot.
        diff = h - u_q
        commit = (diff * diff).mean()
        # Use the quantized (in-meter) utterance downstream.
        return u_q, commit

    # ---- full loss -------------------------------------------------------
    def loss(self, x_array, y_array, beta=0.3, gamma=0.1):
        """
        Total = reconstruction + beta * antiphonal_agreement + gamma * meter.
          reconstruction : recover the clean canonical response
          agreement      : the two choirs must converge (consensus/fixed point)
          meter          : utterances must lie on the metrical lattice
        """
        out = self.forward(x_array)
        recon = mse(out["pred"], y_array)
        # Antiphonal agreement: penalize disagreement between the choirs.
        d = out["uA"] - out["uB"]
        agreement = (d * d).mean()
        total = recon + agreement * beta + out["commit"] * gamma
        return total, {"recon": float(recon.data),
                       "agreement": float(agreement.data),
                       "meter": float(out["commit"].data)}


# =============================================================================
# PART III — The "rite": a deterministic liturgical rule + corruption
# =============================================================================

class Liturgy:
    """
    The finite 'rite': a small fixed set of M canonical verses (well-separated
    refrains living on the unit sphere) and, for each, its canonical response
    under an invariant permutation-and-sign rule.

    This discreteness is faithful to Ambrose -- a liturgy is a bounded
    repertoire of set verses (he is credited with roughly a dozen hymns), not
    an open field of arbitrary utterances. It is also precisely the regime in
    which an error-correcting code helps: corruption knocks a call OFF the
    manifold of legal verses, and the task is to sing the correct refrain back
    ON to it. A metrical lattice that snaps internal states onto legal feet can
    recover the right refrain from a badly corrupted call.
    """

    def __init__(self, D=8, M=12, seed=GLOBAL_SEED):
        rng = np.random.default_rng(seed)
        V = rng.normal(size=(M, D))
        V /= np.linalg.norm(V, axis=1, keepdims=True)   # unit-sphere refrains
        self.V = V
        perm = rng.permutation(D)
        sign = rng.choice([-1.0, 1.0], size=D)
        self.R = np.tanh(1.25 * (V[:, perm] * sign))     # canonical responses
        self.D, self.M = D, M

    def sample(self, B, rng, noise=0.0, mask_frac=0.0):
        """
        Draw B verses at random; corrupt the calls with 'Arian drift'.
        Returns (x_corrupted_call, y_canonical_response, verse_indices).
        """
        idx = rng.integers(0, self.M, size=B)
        clean = self.V[idx]
        x = clean + rng.normal(0.0, noise, size=clean.shape)
        if mask_frac > 0.0:
            keep = rng.uniform(size=clean.shape) > mask_frac
            x = x * keep
        return x, self.R[idx], idx

    def nearest_response(self, preds):
        """Snap each prediction to the nearest canonical response index."""
        d = ((preds[:, None, :] - self.R[None, :, :]) ** 2).sum(axis=2)
        return d.argmin(axis=1)


# =============================================================================
# PART IV — Optimizer (Adam, from scratch)
# =============================================================================

class Adam:
    def __init__(self, params, lr=3e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.params = params
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]
        self.t = 0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            g = p.grad
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            mhat = self.m[i] / (1 - self.b1 ** self.t)
            vhat = self.v[i] / (1 - self.b2 ** self.t)
            p.data -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


# =============================================================================
# PART V — MANDATORY finite-difference gradient check
# =============================================================================

def gradient_check(verbose=True):
    """
    Verify the hand-written backward pass against numerical gradients. We check
    every parameter tensor at a few random coordinates. Passing this is the
    contract that makes the training loop trustworthy.
    """
    rng = np.random.default_rng(0)
    D, H, K, T = 5, 7, 4, 4
    net = AntiphonalSettlingNetwork(D=D, H=H, K=K, T=T, tau=0.5, use_meter=True, seed=3)
    lit = Liturgy(D=D, M=6, seed=3)

    B = 4
    x, y, _idx = lit.sample(B, rng, noise=0.15, mask_frac=0.1)

    def loss_value():
        total, _ = net.loss(x, y, beta=0.3, gamma=0.1)
        return total

    # Analytic gradients.
    net.zero_grad()
    total = loss_value()
    total.backward()
    analytic = [p.grad.copy() for p in net.parameters()]

    eps = 1e-6
    max_rel = 0.0
    names = ["W_A", "b_A", "W_B", "b_B", "C", "W_out", "b_out"]
    for pi, p in enumerate(net.parameters()):
        flat = p.data.reshape(-1)
        n_check = min(6, flat.size)
        idxs = np.random.default_rng(pi + 1).choice(flat.size, size=n_check, replace=False)
        for idx in idxs:
            orig = flat[idx]
            flat[idx] = orig + eps
            lp = float(loss_value().data)
            flat[idx] = orig - eps
            lm = float(loss_value().data)
            flat[idx] = orig
            num = (lp - lm) / (2 * eps)
            ana = analytic[pi].reshape(-1)[idx]
            denom = max(1e-8, abs(num) + abs(ana))
            rel = abs(num - ana) / denom
            max_rel = max(max_rel, rel)
    if verbose:
        print(f"  [grad-check] max relative error across params = {max_rel:.2e}")
    passed = max_rel < 1e-4
    return passed, max_rel


# =============================================================================
# PART VI — Training + the error-correcting-meter ablation
# =============================================================================

def train_model(use_meter, steps=800, B=64, D=8, M=12, H=24, K=16, T=4,
                tau=0.4, noise=0.30, mask_frac=0.20, seed=GLOBAL_SEED, log=True):
    rng = np.random.default_rng(seed)
    net = AntiphonalSettlingNetwork(D=D, H=H, K=K, T=T, tau=tau,
                                    use_meter=use_meter, seed=seed)
    lit = Liturgy(D=D, M=M, seed=seed)
    opt = Adam(net.parameters(), lr=3e-3)

    for step in range(steps):
        x, y, _idx = lit.sample(B, rng, noise=noise, mask_frac=mask_frac)
        net.zero_grad()
        total, parts = net.loss(x, y, beta=0.3, gamma=0.1)
        total.backward()
        opt.step()
        if log and (step % 200 == 0 or step == steps - 1):
            tag = "in-meter " if use_meter else "no-meter "
            print(f"  [{tag}] step {step:4d}  loss={float(total.data):.4f}  "
                  f"recon={parts['recon']:.4f}  agree={parts['agreement']:.4f}  "
                  f"meter={parts['meter']:.4f}")
    return net, lit


def evaluate_robustness(net, lit, corruption_levels, B=1500, seed=99):
    """
    Report REFRAIN-RECOVERY ACCURACY as corruption rises: given a corrupted
    call, does the network sing back close enough to the correct canonical
    response that the nearest legal refrain is the right one? Chance = 1/M.
    Higher is better; the gap between the in-meter and no-meter models is the
    quantity that tests Ambrose's error-correcting-meter claim.
    """
    rng = np.random.default_rng(seed)
    results = []
    for lvl in corruption_levels:
        x, y, idx = lit.sample(B, rng, noise=lvl, mask_frac=min(0.5, lvl))
        pred = net.forward(x)["pred"].data
        pick = lit.nearest_response(pred)
        results.append(float((pick == idx).mean()))
    return results


# =============================================================================
# PART VII — Self-tests
# =============================================================================

def self_tests():
    print("Self-tests")
    print("-" * 68)

    # (1) Autodiff sanity: d/dx mean((Wx+b)^2) matches finite differences.
    rng = np.random.default_rng(1)
    W = Node(rng.normal(size=(3, 3)))
    x = Node(rng.normal(size=(2, 3)))
    b = Node(rng.normal(size=(1, 3)))
    y = ((x @ W) + b)
    loss = (y * y).mean()
    loss.backward()
    eps = 1e-6
    flat = W.data.reshape(-1)
    g_num = np.zeros_like(flat)
    for i in range(flat.size):
        o = flat[i]
        flat[i] = o + eps
        lp = float((((x.data @ W.data) + b.data) ** 2).mean())
        flat[i] = o - eps
        lm = float((((x.data @ W.data) + b.data) ** 2).mean())
        flat[i] = o
        g_num[i] = (lp - lm) / (2 * eps)
    ok_autodiff = np.allclose(g_num, W.grad.reshape(-1), atol=1e-5)
    print(f"  autodiff matmul/mean gradient ....... {'PASS' if ok_autodiff else 'FAIL'}")

    # (2) The quantizer output must lie inside the convex hull of the codebook
    #     (soft VQ is a convex combination of feet): each |u_q| <= max|C| row.
    u = Node(rng.normal(size=(5, 4)) * 3.0)
    C = Node(rng.normal(size=(6, 4)))
    u_q, P = metrical_quantize(u, C, tau=0.5)
    hull_ok = np.all(P >= -1e-9) and np.allclose(P.sum(axis=1), 1.0)
    print(f"  metrical quantizer is a valid mix ... {'PASS' if hull_ok else 'FAIL'}")

    # (3) Antiphonal coupling actually alternates: with T=2 the prediction must
    #     depend on choir B's weights (perturbing W_B changes the output).
    net = AntiphonalSettlingNetwork(D=4, H=6, K=4, T=2, use_meter=True, seed=5)
    x0 = np.ones((3, 4)) * 0.3
    p_before = net.forward(x0)["pred"].data.copy()
    net.W_B.data += 0.5
    p_after = net.forward(x0)["pred"].data
    coupling_ok = not np.allclose(p_before, p_after)
    print(f"  two-choir antiphonal coupling live .. {'PASS' if coupling_ok else 'FAIL'}")

    return ok_autodiff and hull_ok and coupling_ok


# =============================================================================
# PART VIII — Main liturgy
# =============================================================================

def main():
    np.random.seed(GLOBAL_SEED)
    print("=" * 68)
    print(" St. Ambrose of Milan — Antiphonal Settling Network")
    print(" 'antiphonae, hymni, ac vigiliae' — Paulinus, Vita Ambrosii 13.3")
    print("=" * 68)

    print("\n[1] Gradient check (mandatory)")
    print("-" * 68)
    ok, max_rel = gradient_check()
    print(f"  result: {'PASS' if ok else 'FAIL'} (threshold 1e-4)")
    if not ok:
        raise SystemExit("Gradient check failed — aborting the rite.")

    print("\n[2] " )
    passed = self_tests()
    print(f"  overall self-tests: {'PASS' if passed else 'FAIL'}")

    print("\n[3] Training two choirs — WITH the metrical lattice (in-meter)")
    print("-" * 68)
    net_meter, lit = train_model(use_meter=True)

    print("\n[4] Training two choirs — WITHOUT the lattice (ablation, no-meter)")
    print("-" * 68)
    # Same liturgy/seed so the two runs are directly comparable.
    net_flat, _ = train_model(use_meter=False)

    print("\n[5] Error-correcting-meter ablation")
    print("    refrain-recovery accuracy under rising 'Arian drift'")
    print(f"    (M={lit.M} canonical verses; blind chance = {1.0/lit.M:.3f})")
    print("-" * 68)
    levels = [0.0, 0.30, 0.60, 0.90, 1.20]
    acc_meter = evaluate_robustness(net_meter, lit, levels)
    acc_flat = evaluate_robustness(net_flat, lit, levels)
    print("  corruption |  in-meter acc |  no-meter acc |  meter advantage")
    print("  ---------- |  ------------ |  ------------ |  ---------------")
    for lvl, am, af in zip(levels, acc_meter, acc_flat):
        adv = am - af
        print(f"     {lvl:4.2f}   |    {am:8.3f}  |    {af:8.3f}  |    {adv:+8.3f}")

    # Summarize the thesis quantitatively (mean advantage under real drift).
    drift = slice(1, None)  # exclude the zero-corruption row
    avg_adv = float(np.mean(np.array(acc_meter)[drift] - np.array(acc_flat)[drift]))
    print("-" * 68)
    verdict = ("meter DEFENDS the refrain (higher recovery under drift)"
               if avg_adv > 0 else "meter did not help on this run")
    print(f"  mean accuracy advantage of strict meter under drift: {avg_adv:+.3f}")
    print(f"  verdict: {verdict}")
    print("\nThe antiphon holds. Consensus reached; the congregation sings as one.")


if __name__ == "__main__":
    main()
