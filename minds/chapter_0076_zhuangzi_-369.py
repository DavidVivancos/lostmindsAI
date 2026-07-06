#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 chapter_0076_zhuangzi_-369.py  —  The Pivot-of-the-Dao Network (PDN)
 A from-scratch, trainable neural architecture after the cognitive signature of
 Zhuang Zhou (Zhuangzi, c. 369-286 BCE), native of Meng in the state of Song.
  Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
Tome 4 Minds 61 - 80 Available on Amazon https://www.amazon.com/dp/B0H71JC95Q
Resume and Interactive Demos at https://artificiology.com/  
Author: David Vivancos · Chapter 0076 · Zhuang Zhou (Zhuangzi)
================================================================================

WHY THIS ARCHITECTURE IS *ZHUANGZI* AND NOT JUST "ANOTHER DAOIST NET"
--------------------------------------------------------------------
Laozi's lane (already taken elsewhere in this corpus) is emptiness (xu) and
wu-wei as non-conceptual response. If we built "an effortless wu-wei layer" we
would simply be re-skinning Laozi. So we deliberately diverge. Zhuangzi's own
and unmistakable cognitive contributions are three, and the network encodes
exactly those three — nothing generic:

  1. THE PIVOT OF THE DAO  (道樞, dao shu) and EQUALIZING THINGS (齊物, qi wu).
     "This is also that; that is also this." No frame is privileged; the sage
     stands at the still center of the ring (the pivot) and lets each situation
     supply its own rightness — 因是, yin shi, "going by the rightness of THIS
     moment." Computationally: a bank of competing *frames*, softly selected by
     the *moment* (the input context), with an explicit penalty that forbids any
     single frame from globally dominating. The model is taught that truth is
     frame-relative, not frame-free.

  2. COOK DING CARVES THE OX  (庖丁解牛, Inner Chapter 3, "Caring for Life").
     The cook's blade is undulled after nineteen years because he never hacks
     through bone — he "goes in by the natural openings," letting what has no
     thickness slip into the spaces that have room. Mastery is following the
     grain (天理, the natural structure), not forcing. Computationally: inference
     is a short *flow* that descends along learned low-resistance directions
     ("the joints"), and we MINIMIZE the cumulative motion it takes to settle
     ("blade_wear"). The hallmark metric of a Zhuangzi-mind is: solve the task
     with the least cutting. A good model keeps its blade sharp.

  3. FORGET THE TRAP ONCE YOU HAVE THE FISH  (得魚忘筌, Chapter 26).
     "The fish-trap exists because of the fish; once you've got the fish you can
     forget the trap. Words exist because of meaning; once you've got the
     meaning you can forget the words." Symbols are disposable scaffolds.
     Computationally: a *symbolic trap* side-channel helps shape the flow during
     training, but a consistency objective forces the flow to do the work alone,
     and at "enlightened" evaluation the trap is DROPPED. A true Zhuangzi-mind
     keeps the fish after the trap is gone: accuracy must survive the deletion
     of its own explicit code.

So the full forward pass reads, in Zhuangzi's own images:
    embed the situation  →  let the moment pick its frames (pivot / yin shi)  →
    flow through the joints with least cutting (Cook Ding)  →  read the catch,
    then prove you can forget the trap (de yu wang quan).

ENGINEERING CONTRACT (kept identical across the whole 1000-Minds corpus)
------------------------------------------------------------------------
  * Pure NumPy, written from scratch (a tiny reverse-mode autodiff engine lives
    at the top of this file; no PyTorch / TF / JAX).
  * A finite-difference gradient check that MUST pass (mandatory).
  * A real training loop on a real (synthetic but non-trivial) task, with a held
    -out test set and the three Zhuangzi metrics reported.
    * Self-tests, executable end to end. Run:  python3 chapter_0076_zhuangzi_-369.py
  * The verified stdout of a real run is pasted into the chapter prose.

The task is chosen to make the philosophy *bite*: a two-spiral ("twin-arm")
classification where the boundary between classes is a curved cavity. A blade
that hacks straight (a linear/forcing model) jams on bone; only a model that
follows the grain of the spiral separates the arms. This is Cook Ding's ox in
miniature, and it is genuinely hard for a forcing model — so the metrics mean
something.

Author's note on provenance: the biography (Meng/Song, the lacquer-garden post,
the refused ministry of Chu, the friend-rival Hui Shi) rests on Sima Qian's
Shiji ch. 63 and the Inner Chapters themselves; the AGI reading is our own
extrapolation, flagged as such in the chapter.
================================================================================
"""

import numpy as np

# Reproducibility. 76 = this mind's index in the sequence.
np.random.seed(76)


# ==============================================================================
# PART 0  —  A MINIMAL REVERSE-MODE AUTODIFF ENGINE (from scratch, ~120 lines)
# ------------------------------------------------------------------------------
# We grow our own tiny "tape." Each Node remembers how to push gradient to its
# parents. This guarantees the finite-difference gradient check below validates
# BOTH the engine and the model — there is no framework to hide behind.
# ==============================================================================

class Node:
    """A scalar-or-tensor value on the autodiff tape."""
    __slots__ = ("data", "grad", "_parents", "_backward")

    def __init__(self, data, parents=()):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._parents = parents
        self._backward = lambda: None

    # ---- helpers ----------------------------------------------------------
    @staticmethod
    def _unbroadcast(grad, shape):
        """Sum a gradient back down to `shape` after NumPy broadcasting."""
        while grad.ndim > len(shape):
            grad = grad.sum(axis=0)
        for i, s in enumerate(shape):
            if s == 1 and grad.shape[i] != 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad.reshape(shape)

    # ---- ops --------------------------------------------------------------
    def __add__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        out = Node(self.data + other.data, (self, other))

        def _backward():
            self.grad += Node._unbroadcast(out.grad, self.data.shape)
            other.grad += Node._unbroadcast(out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        out = Node(self.data * other.data, (self, other))

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

    def matmul(self, other):
        out = Node(self.data @ other.data, (self, other))

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        out._backward = _backward
        return out

    def __matmul__(self, other):
        return self.matmul(other)

    def sum(self, axis=None, keepdims=False):
        out = Node(self.data.sum(axis=axis, keepdims=keepdims), (self,))

        def _backward():
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self.grad += np.ones_like(self.data) * g
        out._backward = _backward
        return out

    def mean(self):
        return self.sum() * (1.0 / self.data.size)

    def tanh(self):
        t = np.tanh(self.data)
        out = Node(t, (self,))

        def _backward():
            self.grad += (1.0 - t * t) * out.grad
        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + np.exp(-self.data))
        out = Node(s, (self,))

        def _backward():
            self.grad += s * (1.0 - s) * out.grad
        out._backward = _backward
        return out

    def softmax(self, axis=-1):
        z = self.data - self.data.max(axis=axis, keepdims=True)
        e = np.exp(z)
        p = e / e.sum(axis=axis, keepdims=True)
        out = Node(p, (self,))

        def _backward():
            # Jacobian-vector product for softmax along `axis`.
            dot = (out.grad * p).sum(axis=axis, keepdims=True)
            self.grad += p * (out.grad - dot)
        out._backward = _backward
        return out

    def square(self):
        out = Node(self.data ** 2, (self,))

        def _backward():
            self.grad += 2.0 * self.data * out.grad
        out._backward = _backward
        return out

    def log(self):
        out = Node(np.log(self.data + 1e-12), (self,))

        def _backward():
            self.grad += (1.0 / (self.data + 1e-12)) * out.grad
        out._backward = _backward
        return out

    # ---- reverse pass -----------------------------------------------------
    def backward(self):
        topo, seen = [], set()

        def build(v):
            if id(v) in seen:
                return
            seen.add(id(v))
            for p in v._parents:
                build(p)
            topo.append(v)
        build(self)
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()


# ==============================================================================
# PART 1  —  PARAMETER HELPERS
# ==============================================================================

def he_init(shape):
    """He/Xavier-style init scaled for tanh-ish flows."""
    fan_in = shape[0]
    return np.random.randn(*shape) * np.sqrt(1.0 / fan_in)


def param(shape, scale=None):
    w = he_init(shape) if scale is None else np.random.randn(*shape) * scale
    return Node(w)


# ==============================================================================
# PART 2  —  THE PIVOT-OF-THE-DAO NETWORK (PDN)
# ------------------------------------------------------------------------------
# forward(x):
#   z0      = tanh(x We + be)                         # embed the situation
#   --- THE PIVOT (道樞) / EQUALIZING THINGS (齊物) ---
#   frame_w = softmax( (z0 + trap) Wp )              # the moment picks frames
#             (trap is the symbolic side-channel; dropped at enlightened eval)
#   --- COOK DING'S FLOW (庖丁解牛): T steps through the joints ---
#   for t in 1..T:
#       feats = tanh(z Wf + bf)                       # read the local grain
#       dirs  = feats Wd                              # candidate cut directions,
#               reshaped to (B, n_frames, H)          #   one per frame
#       move  = sum_k frame_w[:,k] * dirs[:,k,:]      # 因是: follow THIS frame's cut
#       gate  = sigmoid(z Wg + bg)                    # "natural openings": where
#                                                     #   there is room to enter
#       step  = eta * gate * move
#       z     = z + step                              # residual flow (least force)
#       wear += mean(step^2)                          # blade wear == cutting cost
#   logits  = z Wo + bo                               # read the catch
#   loss    = CE(logits, y)
#           + lam_wear  * wear                        # Cook Ding: keep blade sharp
#           + lam_equal * frame_imbalance             # 齊物: no frame may dominate
#           + lam_forget* trap_consistency            # 得魚忘筌: survive forgetting
#
# Every term above is one of Zhuangzi's three ideas, made differentiable.
# ==============================================================================

class PivotDaoNetwork:
    def __init__(self, d_in, d_hidden=24, n_frames=4, n_classes=2, T=5,
                 eta=0.7, lam_wear=0.05, lam_equal=0.08, lam_forget=0.15,
                 seed=76):
        rng = np.random.RandomState(seed)
        np.random.seed(seed)
        self.d_in, self.H, self.K = d_in, d_hidden, n_frames
        self.C, self.T, self.eta = n_classes, T, eta
        self.lam_wear = lam_wear
        self.lam_equal = lam_equal
        self.lam_forget = lam_forget

        H, K = d_hidden, n_frames
        # Embed the situation.
        self.We = param((d_in, H));          self.be = Node(np.zeros((1, H)))
        # The symbolic "trap": a compact code read straight off the input.
        self.Wc = param((d_in, H), scale=0.3)
        # The pivot: pick frames from (embedding + trap).
        self.Wp = param((H, K), scale=0.3)
        # Grain reader (shared across flow steps -> a recurrent flow).
        self.Wf = param((H, H));             self.bf = Node(np.zeros((1, H)))
        # Cut directions: one H-vector per frame.
        self.Wd = param((H, K * H), scale=0.2)
        # Natural openings (gate).
        self.Wg = param((H, H), scale=0.3);  self.bg = Node(np.zeros((1, H)))
        # Readout.
        self.Wo = param((H, n_classes), scale=0.3)
        self.bo = Node(np.zeros((1, n_classes)))

    def params(self):
        return [self.We, self.be, self.Wc, self.Wp, self.Wf, self.bf,
                self.Wd, self.Wg, self.bg, self.Wo, self.bo]

    # -- forward; returns (loss_node, diagnostics) --------------------------
    def forward(self, X, y=None, use_trap=True):
        B = X.shape[0]
        Xn = Node(X)

        # embed the situation
        z = (Xn @ self.We + self.be).tanh()

        # symbolic trap (the words / the fish-trap)
        trap = (Xn @ self.Wc).tanh()
        pivot_in = z + trap if use_trap else z

        # THE PIVOT: the moment selects frames (因是). softmax over K frames.
        frame_w = (pivot_in @ self.Wp).softmax(axis=-1)        # (B, K)

        # COOK DING'S FLOW through the joints
        wear = Node(0.0)
        for _ in range(self.T):
            feats = (z @ self.Wf + self.bf).tanh()             # local grain
            dirs = feats @ self.Wd                             # (B, K*H)
            # reshape (B, K*H) -> per-frame cut directions and combine by frame_w
            # We avoid a reshape op in the engine by slicing columns per frame.
            move = None
            for k in range(self.K):
                d_k = _col_slice(dirs, k * self.H, (k + 1) * self.H)   # (B,H)
                w_k = _col_slice(frame_w, k, k + 1)                    # (B,1)
                contrib = d_k * w_k
                move = contrib if move is None else (move + contrib)
            gate = (z @ self.Wg + self.bg).sigmoid()           # natural openings
            step = (gate * move) * self.eta
            z = z + step                                       # residual descent
            wear = wear + step.square().mean()                 # cutting cost

        logits = z @ self.Wo + self.bo

        diag = {"frame_w": frame_w, "z": z, "logits": logits, "wear": wear}
        if y is None:
            return logits, diag

        # --- losses ---
        ce = cross_entropy(logits, y)

        # 齊物 (equalize things): penalize a frame that dominates the WHOLE batch.
        # Mean usage per frame should stay near uniform 1/K.
        mean_use = frame_w.mean()             # scalar mean over all entries
        # imbalance = sum_k (avg_k - 1/K)^2 ; build avg_k via column means
        imbalance = Node(0.0)
        inv_k = 1.0 / self.K
        for k in range(self.K):
            col = _col_slice(frame_w, k, k + 1)            # (B,1)
            avg_k = col.mean()
            imbalance = imbalance + (avg_k - Node(inv_k)).square()
        equal_loss = imbalance

        loss = ce + (wear * self.lam_wear) + (equal_loss * self.lam_equal)
        diag["ce"] = ce
        diag["equal"] = equal_loss
        diag["mean_use"] = mean_use
        return loss, diag

    # -- forget-the-trap consistency: flow-with-trap must agree with flow-
    #    without-trap, so the network can drop the trap and keep the fish. ---
    def forget_consistency(self, X):
        _, d_on = self.forward(X, y=None, use_trap=True)
        _, d_off = self.forward(X, y=None, use_trap=False)
        diff = (d_on["logits"] - d_off["logits"]).square().mean()
        return diff


def _col_slice(node, c0, c1):
    """Differentiable column slice node[:, c0:c1] for our tiny engine."""
    out = Node(node.data[:, c0:c1], (node,))

    def _backward():
        g = np.zeros_like(node.data)
        g[:, c0:c1] = out.grad
        node.grad += g
    out._backward = _backward
    return out


def cross_entropy(logits, y):
    """Mean softmax cross-entropy. y is an int array of shape (B,)."""
    p = logits.softmax(axis=-1)
    B = logits.data.shape[0]
    onehot = np.zeros_like(p.data)
    onehot[np.arange(B), y] = 1.0
    picked = (p * Node(onehot)).sum(axis=-1)          # (B,)
    return (picked.log() * -1.0).mean()


# ==============================================================================
# PART 3  —  THE TASK: TWIN SPIRAL ("THE GRAIN OF THE OX")
# ------------------------------------------------------------------------------
# Two interleaved spiral arms. The separating boundary is a curved cavity, not a
# straight cut. A forcing (linear) blade jams; only a grain-follower separates
# the arms. This makes the wu-wei / Cook-Ding objective measurable, not poetic.
# ==============================================================================

def make_twin_spiral(n_per_class=250, noise=0.10, seed=76):
    rng = np.random.RandomState(seed)
    n = n_per_class
    sweep = 2.4 * np.pi
    theta = np.sqrt(rng.rand(n)) * sweep             # spiral sweep
    r = theta / sweep
    # arm 0
    x0 = np.stack([r * np.cos(theta), r * np.sin(theta)], 1)
    # arm 1 (rotated by pi -> interleaved)
    x1 = np.stack([r * np.cos(theta + np.pi), r * np.sin(theta + np.pi)], 1)
    X = np.concatenate([x0, x1], 0)
    X = X + rng.randn(*X.shape) * noise
    y = np.concatenate([np.zeros(n, int), np.ones(n, int)])
    # add a radial+angular feature lift (the model must still find the grain)
    rad = np.linalg.norm(X, axis=1, keepdims=True)
    ang = np.arctan2(X[:, 1:2], X[:, 0:1])
    X = np.concatenate([X, rad, np.sin(ang), np.cos(ang)], 1)   # d_in = 5
    # shuffle
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def standardize(X, mu=None, sd=None):
    if mu is None:
        mu, sd = X.mean(0, keepdims=True), X.std(0, keepdims=True) + 1e-8
    return (X - mu) / sd, mu, sd


# ==============================================================================
# PART 4  —  TRAINING (plain SGD with momentum; pure NumPy)
# ==============================================================================

def accuracy(model, X, y, use_trap=True):
    logits, _ = model.forward(X, y=None, use_trap=use_trap)
    pred = logits.data.argmax(1)
    return float((pred == y).mean())


def avg_blade_wear(model, X):
    _, d = model.forward(X, y=None, use_trap=True)
    return float(d["wear"].data)


def train(model, Xtr, ytr, Xte, yte, epochs=60, lr=0.05, mom=0.9, verbose=True):
    ps = model.params()
    vel = [np.zeros_like(p.data) for p in ps]
    hist = []
    for ep in range(1, epochs + 1):
        # zero grads
        for p in ps:
            p.grad = np.zeros_like(p.data)
        loss, diag = model.forward(Xtr, ytr, use_trap=True)
        # add the forget-the-trap consistency term (得魚忘筌)
        forget = model.forget_consistency(Xtr)
        total = loss + forget * model.lam_forget
        total.backward()
        # SGD + momentum, with light gradient clipping
        for i, p in enumerate(ps):
            g = np.clip(p.grad, -5.0, 5.0)
            vel[i] = mom * vel[i] - lr * g
            p.data += vel[i]
        if verbose and (ep % 20 == 0 or ep == 1):
            tr_acc = accuracy(model, Xtr, ytr)
            te_acc = accuracy(model, Xte, yte)
            te_forget = accuracy(model, Xte, yte, use_trap=False)
            print(f"  epoch {ep:3d} | loss {float(total.data):7.4f} "
                  f"| train {tr_acc:5.3f} | test {te_acc:5.3f} "
                  f"| test(trap-forgotten) {te_forget:5.3f} "
                  f"| blade_wear {float(diag['wear'].data):6.4f}")
        hist.append(float(total.data))
    return hist


# ==============================================================================
# PART 5  —  FINITE-DIFFERENCE GRADIENT CHECK  (mandatory; must pass)
# ==============================================================================

def gradient_check(seed=0, tol=1e-4):
    """Compare analytic (autodiff) grads to numerical grads on a tiny model."""
    np.random.seed(seed)
    X = np.random.randn(6, 5)
    y = np.random.randint(0, 2, size=6)
    model = PivotDaoNetwork(d_in=5, d_hidden=6, n_frames=3, n_classes=2,
                            T=3, seed=seed)

    def total_loss():
        loss, _ = model.forward(X, y, use_trap=True)
        loss = loss + model.forget_consistency(X) * model.lam_forget
        return loss

    # analytic
    for p in model.params():
        p.grad = np.zeros_like(p.data)
    L = total_loss()
    L.backward()
    analytic = [p.grad.copy() for p in model.params()]

    # numerical (central differences) on a random subset of entries per param
    eps = 1e-5
    max_rel = 0.0
    rng = np.random.RandomState(seed + 1)
    names = ["We", "be", "Wc", "Wp", "Wf", "bf", "Wd", "Wg", "bg", "Wo", "bo"]
    for pi, p in enumerate(model.params()):
        flat = p.data.reshape(-1)
        m = min(8, flat.size)
        sel = rng.choice(flat.size, size=m, replace=False)
        for idx in sel:
            orig = flat[idx]
            flat[idx] = orig + eps
            lp = float(total_loss().data)
            flat[idx] = orig - eps
            lm = float(total_loss().data)
            flat[idx] = orig
            num = (lp - lm) / (2 * eps)
            ana = analytic[pi].reshape(-1)[idx]
            denom = max(1e-8, abs(num) + abs(ana))
            rel = abs(num - ana) / denom
            max_rel = max(max_rel, rel)
    print(f"  max relative error across sampled params: {max_rel:.2e}")
    ok = max_rel < tol
    print(f"  gradient check {'PASSED' if ok else 'FAILED'} (tol={tol:.0e})")
    return ok


# ==============================================================================
# PART 6  —  SELF-TESTS + MAIN
# ==============================================================================

def self_tests():
    print("[self-tests]")
    # softmax rows sum to 1
    s = Node(np.random.randn(4, 5)).softmax(axis=-1)
    assert np.allclose(s.data.sum(1), 1.0), "softmax rows must sum to 1"
    # cross-entropy of confident-correct ~ small
    logits = Node(np.array([[10.0, -10.0], [-10.0, 10.0]]))
    ce = cross_entropy(logits, np.array([0, 1]))
    assert ce.data < 1e-3, "CE of confident-correct should be ~0"
    # column slice gradient routes correctly
    a = Node(np.random.randn(3, 4))
    out = _col_slice(a, 1, 3).sum()
    a.grad = np.zeros_like(a.data)
    out.backward()
    expect = np.zeros((3, 4)); expect[:, 1:3] = 1.0
    assert np.allclose(a.grad, expect), "column-slice gradient mismatch"
    print("  ok: softmax / cross-entropy / column-slice gradients\n")


def main():
    print("=" * 72)
    print(" THE PIVOT-OF-THE-DAO NETWORK  —  Zhuangzi (76)")
    print("=" * 72)

    self_tests()

    print("[gradient check]")
    ok = gradient_check()
    assert ok, "Gradient check FAILED — aborting."
    print()

    print("[data] twin-spiral 'grain of the ox'")
    X, y = make_twin_spiral(n_per_class=200, noise=0.18)
    n = len(X); cut = int(0.8 * n)
    Xtr_raw, ytr = X[:cut], y[:cut]
    Xte_raw, yte = X[cut:], y[cut:]
    Xtr, mu, sd = standardize(Xtr_raw)
    Xte, _, _ = standardize(Xte_raw, mu, sd)
    print(f"  train={len(Xtr)}  test={len(Xte)}  d_in={X.shape[1]}\n")

    print("[train] Pivot-of-the-Dao Network")
    model = PivotDaoNetwork(d_in=X.shape[1], d_hidden=32, n_frames=4,
                            n_classes=2, T=6, eta=0.5, lam_wear=0.06,
                            lam_equal=0.06, lam_forget=0.12)
    train(model, Xtr, ytr, Xte, yte, epochs=90, lr=0.05)

    print("\n[the three Zhuangzi metrics]")
    te_with = accuracy(model, Xte, yte, use_trap=True)
    te_forg = accuracy(model, Xte, yte, use_trap=False)
    wear = avg_blade_wear(model, Xte)
    _, d = model.forward(Xte, y=None, use_trap=True)
    fw = d["frame_w"].data.mean(0)
    print(f"  Cook Ding (blade wear, lower=sharper)      : {wear:.4f}")
    print(f"  Forget-the-trap (acc drop when trap gone)  : "
          f"{te_with:.3f} -> {te_forg:.3f}  (drop {te_with - te_forg:+.3f})")
    print(f"  Pivot/equalize (mean frame usage)          : "
          f"[{', '.join(f'{v:.2f}' for v in fw)}]  (ideal ~{1/len(fw):.2f} each)")

    # A Zhuangzi-mind keeps the fish after the trap is gone: assert small drop.
    assert te_with > 0.80, "model should actually learn the spiral"
    assert (te_with - te_forg) < 0.15, "should largely survive forgetting the trap"
    print("\n[done] The blade is still sharp; the fish is kept; no frame reigns.")


if __name__ == "__main__":
    main()
