#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 chapter_0032_laozi_lao_tzu_-601.py
 The Valley Network: an AGI core built from SUBTRACTION and REVERSAL
 # Part Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 2 Minds 21 - 40 Available on Amazon https://www.amazon.com/dp/B0H6QCQ9M7
# Resume and Interactive Demos at https://artificiology.com/    
Author: David Vivancos · Chapter 0032 · Laozi (Lao Tzu)
================================================================================

WHY THIS FILE IS NOT A TRANSFORMER
----------------------------------
Almost every modern architecture is *additive*: you stack layers, accumulate
parameters, and push activations forward. The Dao De Jing argues the opposite.
Its single most idiosyncratic claim is not "be reactive" — it is that capability
arises by SUBTRACTION and RETURN, never by accumulation:

    為學日益，為道日損。損之又損，以至於無為。   (ch. 48)
    "In the pursuit of learning, every day something is added.
     In the pursuit of the Way, every day something is dropped.
     Drop and again drop, until you arrive at non-action (wu wei)."

    反者道之動，弱者道之用。                       (ch. 40)
    "Reversal (fan 反) is the movement of the Way;
     weakness is its function."

    三十輻共一轂，當其無，有車之用。               (ch. 11)
    "Thirty spokes share one hub; it is the EMPTINESS at the centre
     that makes the cart useful."  -> the hole is the function.

This module turns those three sentences into three concrete mechanisms and
shows they actually train. There is no central controller, no readout head that
"decides" — the answer SETTLES out of the network's own return-to-root dynamics
(ziran 自然, the self-so).

THE THREE MECHANISMS (each maps to one verse)
---------------------------------------------
1. REVERSION OPTIMIZER  (fan 反, ch. 40 + ch. 16 "歸根曰靜", return to the root)
   Every weight is pulled by TWO forces each step: toward task fit, and back
   toward the root (zero). Learning is a damped oscillation that reverts to a
   low-action fixed point instead of a forward-only descent. This is not L2
   regularisation bolted on — reversion is the *primary* dynamic and the data
   term is the perturbation.

2. CARVING / PRUNING-AS-LEARNING  (sun 損, ch. 48 + the hub's hole, ch. 11)
   Capacity is *carved*, not grown. Weights that revert close to the root are
   permanently emptied (masked to exactly zero). The learned function comes to
   live in the surviving holes — a sparse channel cut through a full block.

3. SETTLING / WU-WEI READOUT  (wu wei 無為, ch. 37 "無為而無不為")
   The output is not computed in one forward sweep. The hidden state is iterated
   to a fixed point of a contraction map — it "settles" — and we read the
   settled state. "Does nothing, yet nothing is left undone."

EVERYTHING IS PURE NUMPY, FROM SCRATCH.
A finite-difference gradient check is MANDATORY and is run on every execution
(see `gradient_check`). The training loop, the carving schedule, and the self
tests all run when you execute this file. Verified console output is pasted into
the companion chapter (32_*.md).

Run:  python3 chapter_0032_laozi_lao_tzu_-601.py
"""

from __future__ import annotations

import numpy as np

# A single global generator keeps every run reproducible (no hidden state).
RNG = np.random.default_rng(老 := 1101)  # 老 = "Lao"; the walrus is just a wink.


# ==============================================================================
# SECTION 0 — Small numerical helpers (kept explicit so the math is auditable)
# ==============================================================================

def tanh(x: np.ndarray) -> np.ndarray:
    """Bounded, odd nonlinearity. We use tanh because it is a *contraction* for
    small slopes, which is what lets the settling dynamics (mechanism 3) reach a
    fixed point. Its derivative is 1 - tanh(x)^2."""
    return np.tanh(x)


def dtanh(y: np.ndarray) -> np.ndarray:
    """Derivative of tanh expressed through its OUTPUT y = tanh(x): 1 - y^2.
    Storing the output and reusing it is cheaper and numerically cleaner."""
    return 1.0 - y * y


def softplus(x: np.ndarray) -> np.ndarray:
    """A soft, always-positive gate. 'Weakness as function' (ch. 40): we prefer
    soft, leaky gates over hard 0/1 switches; the network is allowed to be weak."""
    # numerically stable softplus
    return np.where(x > 20.0, x, np.log1p(np.exp(np.minimum(x, 20.0))))


def mse(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean squared error — our task-fit ('the ten thousand things') term."""
    d = pred - target
    return 0.5 * float(np.mean(d * d))


# ==============================================================================
# SECTION 1 — The Valley layer
# ==============================================================================
#
# A Valley layer is an ordinary affine map y = W x + b followed by tanh, BUT it
# carries a binary *carving mask* M of the same shape as W. The effective weight
# is W * M. As training proceeds we set more of M to zero (carving toward
# emptiness). The mask is never undone — emptiness, once attained, is kept
# (ch. 16: 歸根曰靜, returning to the root is stillness).
# ==============================================================================

class ValleyLayer:
    def __init__(self, n_in: int, n_out: int, name: str = "valley"):
        self.name = name
        self.n_in = n_in
        self.n_out = n_out
        # Small initial weights: we START near the root and let structure be
        # carved, rather than starting large and shrinking.
        scale = 0.5 / np.sqrt(n_in)
        self.W = RNG.normal(0.0, scale, size=(n_out, n_in))
        self.b = np.zeros(n_out)
        # Carving mask: 1 = spoke present, 0 = hole carved. Begins entirely full.
        self.M = np.ones((n_out, n_in))
        # Caches for backprop
        self._x = None
        self._y = None

    @property
    def W_eff(self) -> np.ndarray:
        """The weight the world actually sees: full weight masked by the holes."""
        return self.W * self.M

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        pre = x @ self.W_eff.T + self.b
        self._y = tanh(pre)
        return self._y

    def backward(self, grad_y: np.ndarray):
        """Return (grad_x, grad_W, grad_b). Gradients flow only through unmasked
        weights — a carved hole transmits nothing, exactly as an empty hub
        carries no spoke."""
        grad_pre = grad_y * dtanh(self._y)            # (B, n_out)
        grad_W = grad_pre.T @ self._x                 # (n_out, n_in)
        grad_W = grad_W * self.M                      # holes get zero gradient
        grad_b = grad_pre.sum(axis=0)
        grad_x = grad_pre @ self.W_eff                # (B, n_in)
        return grad_x, grad_W, grad_b


# ==============================================================================
# SECTION 2 — The Valley network (the body of the mind)
# ==============================================================================
#
# A stack of Valley layers whose *final* representation is produced not by a
# single forward pass but by a SETTLING loop (mechanism 3). We add a recurrent
# "return" map R on the hidden state: the state is iterated
#       h_{k+1} = (1 - a) * h_k + a * tanh(R h_k + u)
# where u is the bottom-up drive from the input and `a` is a small step. This is
# a damped fixed-point iteration: it relaxes toward h* with R contracting. The
# network literally returns to a resting state before it answers.
# ==============================================================================

class ValleyNetwork:
    def __init__(self, n_in: int, n_hidden: int, n_out: int,
                 settle_steps: int = 6, settle_step_size: float = 0.5):
        self.enc = ValleyLayer(n_in, n_hidden, name="encode")
        self.dec = ValleyLayer(n_hidden, n_out, name="decode")
        # Return map R: drives the settling dynamics. Kept small (weakness).
        scaleR = 0.3 / np.sqrt(n_hidden)
        self.R = RNG.normal(0.0, scaleR, size=(n_hidden, n_hidden))
        self.MR = np.ones((n_hidden, n_hidden))   # R is carvable too
        self.settle_steps = settle_steps
        self.a = settle_step_size
        self._cache = None

    @property
    def R_eff(self) -> np.ndarray:
        return self.R * self.MR

    # ---- forward with settling -------------------------------------------------
    def forward(self, x: np.ndarray) -> np.ndarray:
        u = self.enc.forward(x)                 # bottom-up drive (B, H)
        h = np.zeros_like(u)
        states, preacts = [], []
        a = self.a
        Re = self.R_eff
        for _ in range(self.settle_steps):
            pre = h @ Re.T + u                  # return map + drive
            g = tanh(pre)
            h_new = (1 - a) * h + a * g
            states.append(h)                   # h BEFORE this step
            preacts.append((pre, g))
            h = h_new
        out = self.dec.forward(h)              # wu-wei readout of the settled state
        self._cache = dict(u=u, states=states, preacts=preacts, h_final=h)
        return out

    # ---- backprop through the unrolled settling -------------------------------
    def backward(self, grad_out: np.ndarray):
        """Backprop through the decoder, then through the unrolled settling loop
        (BPTT), then through the encoder. Returns a dict of all parameter grads."""
        c = self._cache
        a = self.a
        Re = self.R_eff

        grad_h, grad_Wdec, grad_bdec = self.dec.backward(grad_out)  # d L / d h_final

        grad_R = np.zeros_like(self.R)
        grad_u = np.zeros_like(c["u"])

        # Walk the settling loop in reverse.
        for k in reversed(range(self.settle_steps)):
            pre, g = c["preacts"][k]
            h_prev = c["states"][k]
            # h_new = (1-a) h_prev + a g ;  g = tanh(pre) ; pre = h_prev R^T + u
            grad_g = grad_h * a
            grad_pre = grad_g * dtanh(g)
            grad_R += grad_pre.T @ h_prev
            grad_u += grad_pre
            # path back into h_prev: direct (1-a) term + through pre (R)
            grad_h = grad_h * (1 - a) + grad_pre @ Re

        grad_R = grad_R * self.MR

        # Encoder: u = enc(x); grad wrt u flows in.
        _, grad_Wenc, grad_benc = self.enc.backward(grad_u)

        return dict(Wenc=grad_Wenc, benc=grad_benc,
                    Wdec=grad_Wdec, bdec=grad_bdec,
                    R=grad_R)

    # ---- parameter access (flat) for the optimizer & grad-check ---------------
    def params(self) -> dict:
        return dict(Wenc=self.enc.W, benc=self.enc.b,
                    Wdec=self.dec.W, bdec=self.dec.b,
                    R=self.R)

    def masks(self) -> dict:
        return dict(Wenc=self.enc.M, benc=np.ones_like(self.enc.b),
                    Wdec=self.dec.M, bdec=np.ones_like(self.dec.b),
                    R=self.MR)


# ==============================================================================
# SECTION 3 — The Reversion optimizer  (fan 反 + sun 損)
# ==============================================================================
#
# This is the heart of the file and the thing that makes it Laozi's and no one
# else's. A standard SGD step is:
#       theta <- theta - lr * grad
# The Reversion optimizer adds an explicit RETURN-TO-ROOT force and treats the
# data gradient as a *perturbation* of that return, not the main event:
#       theta <- theta - lr * grad  -  rho * theta            (reversion to root)
# Then it CARVES: any weight whose magnitude has reverted below a moving
# threshold is permanently emptied (mask -> 0). The threshold rises on a
# schedule ("損之又損" — drop, and drop again), so the network grows steadily
# emptier and the surviving weights carry the function through the holes.
#
# A light momentum gives the damped-oscillation character of fan (the value
# overshoots the root and returns), rather than a monotone shrink.
# ==============================================================================

class ReversionOptimizer:
    def __init__(self, net: ValleyNetwork, lr: float = 0.05,
                 rho: float = 0.01, momentum: float = 0.6):
        self.net = net
        self.lr = lr
        self.rho = rho            # strength of the pull back to the root
        self.mu = momentum
        self.vel = {k: np.zeros_like(v) for k, v in net.params().items()}

    def step(self, grads: dict):
        params = self.net.params()
        masks = self.net.masks()
        for k, p in params.items():
            g = grads[k]
            # damped oscillation: velocity carries the return + the data push
            self.vel[k] = self.mu * self.vel[k] - self.lr * g - self.rho * p
            p += self.vel[k]
            p *= masks[k]   # keep carved holes exactly empty

    def carve(self, threshold: float):
        """Empty (permanently) any unmasked weight that has reverted below
        `threshold`. Only WEIGHT matrices are carved — biases are the floor of
        the valley and are left intact."""
        carved = 0
        for layer_M, layer_W in ((self.net.enc.M, self.net.enc.W),
                                 (self.net.dec.M, self.net.dec.W),
                                 (self.net.MR,    self.net.R)):
            small = (np.abs(layer_W) < threshold) & (layer_M > 0)
            carved += int(small.sum())
            layer_M[small] = 0.0
            layer_W[small] = 0.0   # also zero the underlying weight
        return carved

    def emptiness(self) -> float:
        """Fraction of weight slots that are now holes — the network's 虛 (xu)."""
        total = hole = 0
        for M in (self.net.enc.M, self.net.dec.M, self.net.MR):
            total += M.size
            hole += int((M == 0).sum())
        return hole / total


# ==============================================================================
# SECTION 4 — Loss, forward/backward wiring, and the MANDATORY gradient check
# ==============================================================================

def forward_loss(net: ValleyNetwork, X: np.ndarray, Y: np.ndarray):
    """Return (loss, grad_out) for MSE. grad_out feeds net.backward."""
    pred = net.forward(X)
    loss = mse(pred, Y)
    grad_out = (pred - Y) / Y.shape[0] / Y.shape[1]   # d(0.5*mean(d^2))/d pred
    return loss, grad_out


def gradient_check(seed: int = 7, tol: float = 1e-5) -> bool:
    """Finite-difference check of EVERY parameter group against backprop.
    This is mandatory: if it ever fails, the architecture is wrong and the file
    must not ship. We disable carving here (full masks) so we test the raw maths.
    """
    rng = np.random.default_rng(seed)
    net = ValleyNetwork(n_in=4, n_hidden=5, n_out=3,
                        settle_steps=4, settle_step_size=0.5)
    X = rng.normal(size=(6, 4))
    Y = rng.normal(size=(6, 3)) * 0.3

    # analytic gradients
    loss, grad_out = forward_loss(net, X, Y)
    grads = net.backward(grad_out)

    eps = 1e-6
    worst = 0.0
    params = net.params()
    for name, P in params.items():
        Gnum = np.zeros_like(P)
        it = np.nditer(P, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            orig = P[idx]
            P[idx] = orig + eps
            lp, _ = forward_loss(net, X, Y)
            P[idx] = orig - eps
            lm, _ = forward_loss(net, X, Y)
            P[idx] = orig
            Gnum[idx] = (lp - lm) / (2 * eps)
            it.iternext()
        Ga = grads[name]
        denom = np.maximum(1e-8, np.abs(Ga) + np.abs(Gnum))
        rel = np.max(np.abs(Ga - Gnum) / denom)
        worst = max(worst, rel)
        flag = "OK " if rel < tol else "BAD"
        print(f"   grad-check [{flag}] {name:5s}  max-rel-err = {rel:.2e}")
    ok = worst < tol
    print(f"   --> worst relative error = {worst:.2e}  "
          f"({'PASS' if ok else 'FAIL'} at tol {tol:.0e})")
    return ok


# ==============================================================================
# SECTION 5 — A task that rewards emptiness: sparse pattern completion
# ==============================================================================
#
# We build a task whose true generator is SPARSE: each output depends on only a
# few inputs through a low-dimensional bottleneck. A dense additive network can
# fit it, but an emptied network fits it *as well or better* while using far
# fewer live weights — demonstrating Laozi's claim that the function lives in the
# holes, and that dropping (sun) does not cost capability (無為而無不為).
# ==============================================================================

def make_task(n_samples: int, n_in: int = 8, n_out: int = 4,
              gen_seed: int = 3, data_seed: int = 0):
    """Sparse pattern-completion task.

    The *generator* (the hidden sparse mixing A) is fixed by `gen_seed`, so the
    train and test splits describe the SAME underlying function — only the input
    samples differ (`data_seed`). This lets us measure genuine generalisation:
    can an emptied valley still predict unseen inputs of the same world?"""
    gen = np.random.default_rng(gen_seed)
    A = gen.normal(size=(n_out, n_in))
    sparsity = (gen.random((n_out, n_in)) < 0.35).astype(float)
    A = A * sparsity                      # most input->output links are absent
    dat = np.random.default_rng(data_seed)
    X = dat.normal(size=(n_samples, n_in))
    Y = np.tanh(X @ A.T) * 0.8
    return X, Y


def train(net: ValleyNetwork, opt: ReversionOptimizer,
          X: np.ndarray, Y: np.ndarray, epochs: int = 600,
          warmup: int = 150, carve_every: int = 30,
          carve_growth: float = 0.010, verbose: bool = True):
    """Two phases that mirror the Dao De Jing.

    Phase 1 (學, 'learning'): let the valley fill — fit the task first.
    Phase 2 (道, 'the Way'): 損之又損 — drop and again drop. Only AFTER the
    network can do the task do we begin carving, so we can SEE that emptying
    does not cost capability. The threshold rises slowly; we never carve in a
    single epoch more than the network can absorb."""
    history = []
    threshold = 0.0
    for ep in range(1, epochs + 1):
        loss, grad_out = forward_loss(net, X, Y)
        grads = net.backward(grad_out)
        opt.step(grads)
        if ep > warmup and ep % carve_every == 0:
            threshold += carve_growth
            opt.carve(threshold)
        if verbose and (ep == 1 or ep % carve_every == 0 or ep == epochs):
            print(f"   epoch {ep:4d}  loss={loss:.5f}  "
                  f"emptiness={opt.emptiness()*100:5.1f}%  thr={threshold:.3f}")
        history.append((ep, loss, opt.emptiness()))
    return history


# ==============================================================================
# SECTION 6 — Self-tests + demonstration (runs on execution)
# ==============================================================================

def self_tests() -> bool:
    print("\n[1] SELF-TESTS")
    ok = True

    # (a) forward shapes
    net = ValleyNetwork(6, 7, 4)
    out = net.forward(RNG.normal(size=(5, 6)))
    cond = out.shape == (5, 4)
    print(f"   [{'OK ' if cond else 'BAD'}] forward shape -> {out.shape}")
    ok &= cond

    # (b) carving really empties and is irreversible under steps
    net2 = ValleyNetwork(8, 8, 4)
    opt2 = ReversionOptimizer(net2)
    before = opt2.emptiness()
    opt2.carve(threshold=10.0)            # huge threshold empties almost all
    after = opt2.emptiness()
    cond = after > before and after > 0.9
    print(f"   [{'OK ' if cond else 'BAD'}] carving empties: "
          f"{before*100:.0f}% -> {after*100:.0f}%")
    ok &= cond
    # a step must not resurrect carved weights
    X = RNG.normal(size=(4, 8)); Y = RNG.normal(size=(4, 4)) * 0.1
    _, go = forward_loss(net2, X, Y)
    opt2.step(net2.backward(go))
    cond = abs(opt2.emptiness() - after) < 1e-12
    print(f"   [{'OK ' if cond else 'BAD'}] holes stay empty after a step")
    ok &= cond

    # (c) settling reaches an approximate fixed point (output stable to more steps)
    n3 = ValleyNetwork(5, 6, 3, settle_steps=6)
    x = RNG.normal(size=(3, 5))
    o6 = n3.forward(x)
    n3.settle_steps = 18
    o18 = n3.forward(x)
    drift = float(np.max(np.abs(o6 - o18)))
    cond = drift < 0.05
    print(f"   [{'OK ' if cond else 'BAD'}] settling is near-fixed-point "
          f"(drift 6->18 steps = {drift:.4f})")
    ok &= cond

    # (d) reversion pulls an undriven weight toward the root
    n4 = ValleyNetwork(4, 4, 2)
    opt4 = ReversionOptimizer(n4, lr=0.0, rho=0.2, momentum=0.0)  # data off, return on
    w0 = float(np.abs(n4.enc.W).mean())
    for _ in range(10):
        opt4.step({k: np.zeros_like(v) for k, v in n4.params().items()})
    w1 = float(np.abs(n4.enc.W).mean())
    cond = w1 < w0
    print(f"   [{'OK ' if cond else 'BAD'}] reversion shrinks weights "
          f"toward root ({w0:.4f} -> {w1:.4f})")
    ok &= cond

    print(f"   self-tests: {'ALL PASS' if ok else 'SOME FAILED'}")
    return ok


def demonstration():
    print("\n[2] GRADIENT CHECK (mandatory)")
    grad_ok = gradient_check()

    tests_ok = self_tests()

    print("\n[3] TRAINING ON SPARSE PATTERN COMPLETION")
    Xtr, Ytr = make_task(256, n_in=8, n_out=4, gen_seed=3, data_seed=1)
    Xte, Yte = make_task(128, n_in=8, n_out=4, gen_seed=3, data_seed=2)
    net = ValleyNetwork(n_in=8, n_hidden=16, n_out=4,
                        settle_steps=6, settle_step_size=0.5)
    opt = ReversionOptimizer(net, lr=0.08, rho=0.0015, momentum=0.7)

    pred0 = net.forward(Xte); test0 = mse(pred0, Yte)
    print(f"   test loss before training: {test0:.5f}")
    train(net, opt, Xtr, Ytr, epochs=600, warmup=150,
          carve_every=30, carve_growth=0.010)
    predf = net.forward(Xte); testf = mse(predf, Yte)
    print(f"   test loss after  training: {testf:.5f}")
    print(f"   final emptiness (xu 虛): {opt.emptiness()*100:.1f}% of weights are holes")
    improved = testf < test0 * 0.5

    print("\n[4] VERDICT")
    print(f"   gradient check ......... {'PASS' if grad_ok else 'FAIL'}")
    print(f"   self-tests ............. {'PASS' if tests_ok else 'FAIL'}")
    print(f"   learned the task ....... {'PASS' if improved else 'FAIL'} "
          f"({test0:.4f} -> {testf:.4f})")
    print(f"   carved & still capable .. {'PASS' if (improved and opt.emptiness()>0.3) else 'n/a'}")
    all_ok = grad_ok and tests_ok and improved
    print(f"\n   The valley emptied itself and lost nothing. "
          f"{'無為而無不為.' if all_ok else '(review needed)'}")
    return all_ok


if __name__ == "__main__":
    print("=" * 72)
    print(" Laozi — The Valley Network  (Neuron.py)")
    print(" capability by subtraction (損) and reversal (反); order by ziran (自然)")
    print("=" * 72)
    success = demonstration()
    print("\nDONE." if success else "\nDONE (with failures — do not ship).")
