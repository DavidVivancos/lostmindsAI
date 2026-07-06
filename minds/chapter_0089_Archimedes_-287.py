#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================================
 THE LEVER-AND-EXHAUSTION ENGINE
 A trainable cognitive architecture after Archimedes of Syracuse (c.287-212 BCE)
 # Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 5 Minds 81 - 100 Available on Amazon https://www.amazon.com/dp/B0H7LP5LP2
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 0089 · Archimedes of Syracuse
============================================================================

WHY THIS ARCHITECTURE (the one idea that is his alone)
----------------------------------------------------------------------------
When the Archimedes Palimpsest was re-read (rediscovered by Heiberg in 1906,
re-imaged at the Walters Art Museum 1999-2008), it exposed a secret that the
formal Greek treatises hide: Archimedes did not *find* his theorems the way he
*proved* them. In "The Method of Mechanical Theorems" he confesses a private,
non-rigorous procedure. He imagines an abstract figure (a parabolic segment,
a sphere) as a physical body sliced into infinitely many indivisible laminae,
hangs those slices on an imagined LEVER, and reads off the unknown area or
volume from the condition of BALANCE. He says plainly that this only
*discovers* the answer; it does not prove it. The proof comes afterwards, by a
second and utterly different faculty: the METHOD OF EXHAUSTION, which traps the
unknown between an inscribed (lower) and a circumscribed (upper) figure and
squeezes the two bounds together until no gap remains -- a double reductio that
shows the quantity can be neither greater nor less than the claimed value.

So Archimedes' mind is not one method but TWO, coupled in a loop:
    (1) a fast, embodied, physically-grounded HEURISTIC -- the lever -- that
        weighs indivisibles to PROPOSE an answer, and
    (2) a slow, unforgiving VERIFIER -- exhaustion -- that BRACKETS the answer
        between bounds and squeezes them shut.
Intelligence, on this reading, lives in neither faculty alone but in the
discipline of letting bold mechanical conjecture be governed by relentless
bracketing. That is the thesis this network makes runnable.

HOW THE CODE EMBODIES IT
----------------------------------------------------------------------------
The canonical Archimedean task is QUADRATURE: given the heights of a curve at
sample points (the "laminae"), estimate the area underneath (the integral) --
exactly the quadrature of the parabola he solved by weighing slices.

  * LEVER LAYER (discovery): each lamina i is a slice with height h_i hung at a
    learned, positive weight q_i. The candidate area is the balanced moment
    sum  candidate = Sum_i q_i * h_i * dx . The weights q_i are produced by a
    small per-lamina encoder, so the engine LEARNS its own quadrature rule
    (it converges toward trapezoid-like weights ~1, but can adapt).

  * EXHAUSTION HEADS (verification): from a pooled summary of the laminae the
    engine emits two non-negative gaps, delta_lo and delta_hi, giving an
    inscribed lower bound L = candidate - delta_lo and a circumscribed upper
    bound U = candidate + delta_hi. The point estimate is the bracket midpoint.

  * EXHAUSTION LOSS (the squeeze): three terms in tension --
        fit      : the midpoint should hit the true area;
        bracket  : the truth must NOT escape [L, U]  (the double reductio);
        squeeze  : minimise (delta_lo + delta_hi)    (drive the gap to zero).
    Their equilibrium is the *tightest interval that still contains the truth*
    -- the method of exhaustion rendered as a loss landscape.

Everything below is pure NumPy, written from scratch: explicit forward pass,
hand-derived analytic gradients, a finite-difference gradient check that MUST
pass, a real training loop (Adam, also from scratch), and self-tests on
integrals whose exact values are known (including a parabola, Prop. 1 of the
Quadrature of the Parabola in spirit). No autograd, no deep-learning library.

Run:  python chapter_0089_Archimedes_-287.py
============================================================================
"""

import numpy as np

# ---------------------------------------------------------------------------
# Small numerically-stable primitives (the "atoms" the engine is built from)
# ---------------------------------------------------------------------------

def softplus(x):
    """softplus(x) = log(1 + e^x), evaluated stably. Used to force the lever
    weights q and the exhaustion gaps delta to be strictly non-negative --
    a hung weight cannot be negative, and a bracket gap cannot be negative."""
    # log1p(exp(-|x|)) + max(x,0) is the standard stable form.
    return np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))

def sigmoid(x):
    """Derivative of softplus is the logistic sigmoid: d/dx softplus = sigma(x)."""
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))

def relu(x):
    """Half-wave rectifier; used in the bracket-containment penalty so that the
    loss only fires when the truth escapes the inscribed/circumscribed bounds."""
    return np.maximum(x, 0.0)


# ---------------------------------------------------------------------------
# Synthetic Archimedean dataset: smooth curves and their exact-ish integrals
# ---------------------------------------------------------------------------

def make_curve_batch(batch, n_samples, rng, n_fine=4001):
    """Generate `batch` random smooth curves on [0,1], sampled at `n_samples`
    equally spaced points (the laminae the engine sees), together with each
    curve's true integral computed on a *much finer* grid (the ground truth the
    engine must learn to estimate).

    A curve is a small random combination of low-frequency cosines plus a gentle
    quadratic -- smooth enough that a coarse quadrature is meaningful, varied
    enough that the engine cannot memorise a single rule.

    Returns:
        H    : (batch, n_samples) sampled heights (the laminae)
        Xpos : (batch, n_samples) sample positions in [0,1]
        area : (batch,) true integral over [0,1]
        dx   : scalar spacing between coarse samples
    """
    xs = np.linspace(0.0, 1.0, n_samples)            # coarse abscissae
    xf = np.linspace(0.0, 1.0, n_fine)               # fine abscissae (truth)
    dx = xs[1] - xs[0]

    H = np.zeros((batch, n_samples))
    area = np.zeros(batch)
    Xpos = np.tile(xs, (batch, 1))

    n_modes = 3
    for b in range(batch):
        # random low-frequency cosine coefficients + quadratic shape
        amps = rng.uniform(-1.0, 1.0, size=n_modes)
        ks = rng.integers(1, 4, size=n_modes).astype(float)   # frequencies 1..3
        phases = rng.uniform(0, 2 * np.pi, size=n_modes)
        a2 = rng.uniform(-0.8, 0.8)                            # quadratic term
        a1 = rng.uniform(-0.5, 0.5)                            # linear term
        bias = rng.uniform(0.5, 2.0)                           # keep curve positive-ish

        def f(x):
            val = bias + a1 * x + a2 * x * x
            for a, k, ph in zip(amps, ks, phases):
                val = val + 0.4 * a * np.cos(2 * np.pi * k * x + ph)
            return val

        H[b] = f(xs)
        yf = f(xf)
        # truth: composite trapezoid on the fine grid (our stand-in for exact)
        area[b] = np.trapezoid(yf, xf)

    return H, Xpos, area, dx


# ---------------------------------------------------------------------------
# THE ENGINE
# ---------------------------------------------------------------------------

class LeverExhaustionEngine:
    """Archimedes' two-method mind as a trainable network.

    Parameters (all learned):
        W1, b1   : per-lamina encoder  (features -> hidden)
        v_lev, c_lev : lever head -> per-lamina log-weight a_i (q_i = softplus(a_i))
        v_lo, c_lo   : exhaustion head -> lower gap delta_lo = softplus(.)
        v_hi, c_hi   : exhaustion head -> upper gap delta_hi = softplus(.)

    Loss weights:
        lb : weight of the bracket-containment penalty (the double reductio)
        ls : weight of the squeeze penalty (the exhaustion that closes the gap)
    """

    def __init__(self, d_in=2, hidden=24, lb=1.0, ls=0.03, seed=0):
        rng = np.random.default_rng(seed)
        s = 0.5
        # Encoder
        self.W1 = rng.normal(0, s, size=(d_in, hidden)) * (1.0 / np.sqrt(d_in))
        self.b1 = np.zeros(hidden)
        # Lever head (discovery): weights hung on the beam
        self.v_lev = rng.normal(0, s, size=hidden) * (1.0 / np.sqrt(hidden))
        self.c_lev = 0.0
        # Exhaustion heads (verification): bracket gaps
        self.v_lo = rng.normal(0, s, size=hidden) * (1.0 / np.sqrt(hidden))
        self.c_lo = 0.0
        self.v_hi = rng.normal(0, s, size=hidden) * (1.0 / np.sqrt(hidden))
        self.c_hi = 0.0

        self.lb = float(lb)
        self.ls = float(ls)

    # -- parameter bookkeeping (lets the grad-check sweep every scalar) --------
    def params(self):
        return {
            "W1": self.W1, "b1": self.b1,
            "v_lev": self.v_lev, "c_lev": self.c_lev,
            "v_lo": self.v_lo, "c_lo": self.c_lo,
            "v_hi": self.v_hi, "c_hi": self.c_hi,
        }

    def set_param(self, name, value):
        setattr(self, name, value)

    # -- forward ---------------------------------------------------------------
    def forward(self, H, Xpos, dx, y=None):
        """Run the engine. Returns predictions and, if `y` is given, the loss.
        Also returns a `cache` of intermediate tensors needed for backprop.

        Shapes: H, Xpos -> (B, n).  y -> (B,).  dx -> scalar.
        """
        B, n = H.shape

        # Features per lamina: [height, position].  (B, n, d_in)
        F = np.stack([H, Xpos], axis=-1)

        # Encoder: Z1 = F W1 + b1 ; A1 = tanh(Z1)
        Z1 = np.einsum("bnd,dh->bnh", F, self.W1) + self.b1     # (B,n,H)
        A1 = np.tanh(Z1)

        # --- LEVER (discovery): hang a positive weight q_i on each lamina -----
        a = A1 @ self.v_lev + self.c_lev                        # (B,n) log-weights
        q = softplus(a)                                         # (B,n) >= 0
        candidate = np.sum(q * H, axis=1) * dx                  # (B,) balanced moment

        # --- EXHAUSTION (verification): bracket from a pooled summary ----------
        g = A1.mean(axis=1)                                     # (B,H) pooled laminae
        s_lo = g @ self.v_lo + self.c_lo                        # (B,)
        s_hi = g @ self.v_hi + self.c_hi                        # (B,)
        delta_lo = softplus(s_lo)                               # (B,) >= 0
        delta_hi = softplus(s_hi)                               # (B,) >= 0

        L = candidate - delta_lo                                # inscribed bound
        U = candidate + delta_hi                                # circumscribed bound
        yhat = 0.5 * (L + U)                                    # bracket midpoint

        cache = dict(H=H, F=F, Z1=Z1, A1=A1, a=a, q=q,
                     candidate=candidate, g=g, s_lo=s_lo, s_hi=s_hi,
                     delta_lo=delta_lo, delta_hi=delta_hi,
                     L=L, U=U, yhat=yhat, dx=dx, B=B, n=n)

        if y is None:
            return yhat, L, U, None, cache

        # --- EXHAUSTION LOSS: fit + bracket-containment + squeeze --------------
        # The bracket term is a *linear* hinge: escaping the bounds costs a
        # constant marginal `lb`, while the squeeze costs a constant `ls` to keep
        # any gap at all. Their balance fixes a crisp equilibrium escape rate of
        # ~ls/lb, i.e. the gap shrinks (exhaustion) until the truth only just
        # begins to slip out -- the tightest interval that still contains it.
        fit = 0.5 * (yhat - y) ** 2
        esc_hi = relu(y - U)          # truth above the circumscribed bound -> bad
        esc_lo = relu(L - y)          # truth below the inscribed bound    -> bad
        bracket = self.lb * (esc_hi + esc_lo)
        squeeze = self.ls * (delta_lo + delta_hi)
        loss_vec = fit + bracket + squeeze
        loss = float(np.mean(loss_vec))

        cache.update(y=y, esc_hi=esc_hi, esc_lo=esc_lo)
        return yhat, L, U, loss, cache

    # -- backward (hand-derived analytic gradients) ----------------------------
    def backward(self, cache):
        """Return a dict of gradients matching params(). Derived by hand from the
        forward pass; verified against finite differences in grad_check()."""
        B = cache["B"]; n = cache["n"]; dx = cache["dx"]
        invB = 1.0 / B
        y = cache["y"]; yhat = cache["yhat"]
        U = cache["U"]; L = cache["L"]
        esc_hi = cache["esc_hi"]; esc_lo = cache["esc_lo"]

        # Top-level per-sample gradients (mean over batch baked in via invB).
        # Linear hinge: marginal cost of escape is the constant lb (as an
        # indicator on whether the truth has slipped past the bound).
        g_yhat = invB * (yhat - y)                       # dLoss/dyhat   (B,)
        g_U = invB * (-self.lb * (esc_hi > 0))           # dLoss/dU      (B,)
        g_L = invB * (self.lb * (esc_lo > 0))            # dLoss/dL      (B,)
        squeeze_grad = invB * self.ls                    # dLoss/ddelta (constant)

        # yhat = candidate + (delta_hi - delta_lo)/2 ; U = candidate + delta_hi ;
        # L = candidate - delta_lo. Chain to candidate and the two gaps.
        d_candidate = g_yhat + g_U + g_L                          # (B,)
        d_delta_hi = 0.5 * g_yhat + g_U + squeeze_grad            # (B,)
        d_delta_lo = -0.5 * g_yhat - g_L + squeeze_grad           # (B,)

        # --- exhaustion heads: delta = softplus(s) ---------------------------
        d_s_lo = d_delta_lo * sigmoid(cache["s_lo"])              # (B,)
        d_s_hi = d_delta_hi * sigmoid(cache["s_hi"])              # (B,)
        g_pool = cache["g"]                                       # (B,H)
        gv_lo = g_pool.T @ d_s_lo                                 # (H,)
        gc_lo = float(np.sum(d_s_lo))
        gv_hi = g_pool.T @ d_s_hi                                 # (H,)
        gc_hi = float(np.sum(d_s_hi))
        # gradient flowing back into the pooled summary g
        d_g = np.outer(d_s_lo, self.v_lo) + np.outer(d_s_hi, self.v_hi)  # (B,H)

        # --- lever head: candidate = sum_i q_i h_i dx, q = softplus(a) -------
        # dCandidate/dq_i = h_i dx ; dq_i/da_i = sigmoid(a_i)
        H = cache["H"]; A1 = cache["A1"]
        d_q = d_candidate[:, None] * (H * dx)                     # (B,n)
        d_a = d_q * sigmoid(cache["a"])                           # (B,n)
        gv_lev = np.einsum("bn,bnh->h", d_a, A1)                  # (H,)
        gc_lev = float(np.sum(d_a))

        # --- gradient into A1 from both paths --------------------------------
        # lever path: a = A1 . v_lev  -> dA1 += d_a * v_lev
        dA1 = d_a[:, :, None] * self.v_lev[None, None, :]        # (B,n,H)
        # bracket path: g = mean_i A1  -> each lamina gets d_g / n
        dA1 += (d_g / n)[:, None, :]                              # (B,n,H)

        # --- through tanh and the encoder ------------------------------------
        dZ1 = dA1 * (1.0 - A1 ** 2)                               # (B,n,H)
        F = cache["F"]
        gW1 = np.einsum("bnd,bnh->dh", F, dZ1)                    # (d_in,H)
        gb1 = np.einsum("bnh->h", dZ1)                            # (H,)

        return {
            "W1": gW1, "b1": gb1,
            "v_lev": gv_lev, "c_lev": gc_lev,
            "v_lo": gv_lo, "c_lo": gc_lo,
            "v_hi": gv_hi, "c_hi": gc_hi,
        }


# ---------------------------------------------------------------------------
# Finite-difference gradient check (MANDATORY -- must pass before training)
# ---------------------------------------------------------------------------

def grad_check(seed=1, eps=1e-6, tol=1e-5):
    """Compare analytic gradients to central finite differences on every
    parameter. Returns the worst relative error. The engine's verifier
    discipline applies to its own code: we do not trust the gradients until
    they are bracketed by numerical truth."""
    rng = np.random.default_rng(seed)
    eng = LeverExhaustionEngine(hidden=8, seed=seed)
    H, Xpos, area, dx = make_curve_batch(5, 9, rng)

    _, _, _, loss0, cache = eng.forward(H, Xpos, dx, area)
    grads = eng.backward(cache)

    worst = 0.0
    worst_name = None
    for name, P in eng.params().items():
        P = np.atleast_1d(np.array(P, dtype=float))
        flat = P.ravel()
        gflat = np.atleast_1d(np.array(grads[name], dtype=float)).ravel()
        # sample a handful of coordinates per parameter to keep it quick
        idxs = range(flat.size) if flat.size <= 12 else \
            rng.choice(flat.size, size=12, replace=False)
        for i in idxs:
            orig = flat[i]
            flat[i] = orig + eps
            eng.set_param(name, _reshape_like(flat, getattr(eng, name)))
            _, _, _, lp, _ = eng.forward(H, Xpos, dx, area)
            flat[i] = orig - eps
            eng.set_param(name, _reshape_like(flat, getattr(eng, name)))
            _, _, _, lm, _ = eng.forward(H, Xpos, dx, area)
            flat[i] = orig
            eng.set_param(name, _reshape_like(flat, getattr(eng, name)))

            num = (lp - lm) / (2 * eps)
            ana = gflat[i]
            denom = max(1.0, abs(num) + abs(ana))
            rel = abs(num - ana) / denom
            if rel > worst:
                worst, worst_name = rel, f"{name}[{i}]"
    return worst, worst_name


def _reshape_like(flat, ref):
    ref = np.array(ref)
    if ref.ndim == 0:
        return float(flat[0]) if flat.size == 1 else float(flat.reshape(())[()])
    return flat.reshape(ref.shape).copy()


# ---------------------------------------------------------------------------
# Adam optimiser (from scratch) + training loop
# ---------------------------------------------------------------------------

class Adam:
    def __init__(self, params, lr=2e-2, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.t = 0
        self.m = {k: np.zeros_like(np.atleast_1d(np.array(v, float)))
                  for k, v in params.items()}
        self.v = {k: np.zeros_like(np.atleast_1d(np.array(v, float)))
                  for k, v in params.items()}

    def step(self, eng, grads):
        self.t += 1
        for k in grads:
            g = np.atleast_1d(np.array(grads[k], float))
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            cur = np.atleast_1d(np.array(getattr(eng, k), float))
            upd = cur - self.lr * mhat / (np.sqrt(vhat) + self.eps)
            eng.set_param(k, _reshape_like(upd.ravel(), getattr(eng, k)))


def train(steps=1200, batch=64, n_samples=16, hidden=24, seed=7, verbose=True):
    rng = np.random.default_rng(seed)
    eng = LeverExhaustionEngine(d_in=2, hidden=hidden, lb=1.0, ls=0.03, seed=seed)
    opt = Adam(eng.params(), lr=2e-2)

    history = []
    for step in range(1, steps + 1):
        H, Xpos, area, dx = make_curve_batch(batch, n_samples, rng)
        yhat, L, U, loss, cache = eng.forward(H, Xpos, dx, area)
        grads = eng.backward(cache)
        opt.step(eng, grads)
        if step % 200 == 0 or step == 1:
            mae = float(np.mean(np.abs(yhat - area)))
            contained = float(np.mean((area >= L) & (area <= U)))
            width = float(np.mean(U - L))
            history.append((step, loss, mae, contained, width))
            if verbose:
                print(f"  step {step:5d} | loss {loss:8.5f} | MAE {mae:.4f} "
                      f"| in-bracket {contained*100:5.1f}% | gap {width:.4f}")
    return eng, history


# ---------------------------------------------------------------------------
# Self-tests on integrals with known exact values
# ---------------------------------------------------------------------------

def known_curve_test(eng, n_samples=16):
    """Feed the trained engine three curves whose integrals we know exactly and
    check the estimate and the bracket. Includes a parabola -- a nod to the
    Quadrature of the Parabola, the theorem Archimedes first found by weighing
    slices on a lever before proving it by exhaustion."""
    xs = np.linspace(0, 1, n_samples)
    dx = xs[1] - xs[0]
    cases = [
        ("constant  f=1.0",      lambda x: np.ones_like(x),        1.0),
        ("parabola  f=x^2+0.5",  lambda x: x * x + 0.5,            1.0 / 3 + 0.5),
        ("cosine    f=1+0.3cos", lambda x: 1 + 0.3 * np.cos(2 * np.pi * x), 1.0),
    ]
    rows = []
    for name, f, exact in cases:
        H = f(xs)[None, :]
        Xpos = xs[None, :]
        yhat, L, U, _, _ = eng.forward(H, Xpos, dx)
        rows.append((name, exact, float(yhat[0]), float(L[0]), float(U[0])))
    return rows


# ---------------------------------------------------------------------------
# Main: gradient check -> train -> self-test, printing a verifiable transcript
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    print("=" * 74)
    print("THE LEVER-AND-EXHAUSTION ENGINE  (Archimedes of Syracuse)")
    print("=" * 74)

    print("\n[1] Finite-difference gradient check (the verifier checks itself)")
    worst, where = grad_check()
    print(f"    worst relative error: {worst:.2e}  at {where}")
    assert worst < 1e-4, "gradient check FAILED"
    print("    PASS  (analytic gradients are bracketed by numerical truth)")

    print("\n[2] Training: lever proposes, exhaustion squeezes")
    eng, history = train()

    print("\n[3] Self-test on integrals with known exact values")
    rows = known_curve_test(eng)
    print(f"    {'curve':<22}{'exact':>9}{'estimate':>10}{'lower':>9}{'upper':>9}"
          f"{'in?':>5}")
    all_ok = True
    for name, exact, est, lo, hi in rows:
        inside = lo - 1e-6 <= exact <= hi + 1e-6
        acc = abs(est - exact) < 0.06
        all_ok = all_ok and acc
        print(f"    {name:<22}{exact:9.4f}{est:10.4f}{lo:9.4f}{hi:9.4f}"
              f"{('Y' if inside else 'n'):>5}")
    print(f"\n    point estimates within 0.06 of exact: {all_ok}")

    print("\n[4] Verdict")
    final = history[-1]
    print(f"    final train loss {final[1]:.5f} | MAE {final[2]:.4f} "
          f"| bracket contains truth {final[3]*100:.1f}% of the time "
          f"| mean gap {final[4]:.4f}")
    print("    The lever discovered; exhaustion bracketed and squeezed. As "
          "Archimedes wrote,")
    print("    it is easier to supply the proof once the method has revealed "
          "the answer.")
    print("=" * 74)
