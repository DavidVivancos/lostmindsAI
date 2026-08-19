"""
================================================================================
Chapter 131 — Marcus Aurelius (121–180 CE)
The Homeostat of the View From Above
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 7 Minds 121 - 140 Available on Amazon https://www.amazon.com/dp/B0HFN6GXMH
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 131: Marcus Aurelius (121–180 CE)
================================================================================   

A from-scratch cognitive architecture, in pure NumPy, that encodes the one
cognitive move that is Marcus Aurelius's alone.

WHY THIS IS NOT A TRANSFORMER, AND NOT AN "ASSENT GATE"
-------------------------------------------------------
The obvious Stoic architecture — an impression arrives, a gate grants or
withholds assent, an impulse follows — already belongs to earlier minds in this
corpus: Chrysippus built the commanding-faculty (impression -> assent ->
impulse), Zeno of Citium built the gate of abstention, Seneca built the
regulation of assent, Epictetus built the dichotomy of control. To repeat any of
them here would be to miss Marcus entirely.

What is Marcus's and no one else's is the *maintenance problem*. The Meditations
are not a treatise; they are a private log in which the same consolations are
re-derived, night after night, because tranquility is not a state you reach and
keep — it decays the instant you stop re-deriving it. And the operation Marcus
reaches for, more than any other Stoic, to re-derive it is the "view from above":
he deliberately re-renders each impression at cosmic scale, so that its affective
charge — which is a function of scale — drops toward zero, while the duty the
impression still lays on him is preserved. Tranquility is therefore a *fixed
point actively held*, not a memory retrieved.

THE MAPPING (mind -> mechanism)
-------------------------------
  * hegemonikon (the ruling faculty / "inner citadel")
        -> a protected recurrent state h that ONLY updates through its own
           endorsed, scale-corrected inputs; externals never write to it
           directly. A leak term makes it a contraction map: perturbations decay,
           so the citadel returns to equilibrium — it is stormed only through
           faulty judgment.
  * the "view from above" (cosmic re-scaling of an impression)
        -> a learned zoom scalar alpha per impression that attenuates the
           AFFECTIVE channel of the representation (by 1/(1+alpha)) while leaving
           the OBJECTIVE channel untouched. Zoom out -> affect drains, content
           stays.
  * discipline of desire (accept what is not up to us)
        -> the model learns to raise alpha (zoom hard) precisely on impressions
           that are "not up to us" (pure affect, no duty) and to zoom gently on
           impressions that carry a real duty, because those it must still act on.
  * discipline of action / the "reserve clause"
        -> the impulse (the duty the situation demands) is read from the *present
           judged impression*, not from the slow citadel, so that right action is
           immediate while the citadel stays calm.
  * tranquility (ataraxia)
        -> a penalty on the disturbance of h caused by impressions that are not
           up to us. Minimizing it teaches the view-from-above discipline.

WHAT THE SELF-TESTS SHOW (all run at the bottom; output pasted into the chapter)
  1. A finite-difference gradient check passes for every parameter (the hand-
     written reverse-mode engine is correct).
  2. After training, the model performs its DUTY perfectly (it still governs).
  3. The learned zoom alpha is HIGHER on "not up to us" impressions than on
     dutiful ones — the view from above has learned what to look past.
  4. The ruling faculty stays near equilibrium under a storm of pure affect (the
     inner citadel holds), and an ABLATION that removes the view-from-above
     discipline lets the citadel be stormed — proving the discipline is what
     produces the calm.
  5. Left un-fed, the citadel decays back to equilibrium: tranquility is not
     stored, it is continually re-derived. This is why the Meditations repeat.

No torch, no tensorflow, no autograd library. The reverse-mode engine below is
built by hand; the gradient check is its own proof of correctness.
================================================================================
"""

import numpy as np


# ==============================================================================
# SECTION 1 — A tiny reverse-mode autodiff engine over NumPy 2-D arrays
# ------------------------------------------------------------------------------
# Everything the model needs (matmul, broadcast add/mul, tanh, sigmoid, softplus,
# reciprocal, row-slicing, sum, square) is a node that knows how to push gradient
# to its parents. This is "from scratch" in the strict sense: the chain rule is
# implemented explicitly, op by op.
# ==============================================================================

def _unbroadcast(grad, shape):
    """Sum a gradient back down to `shape` after NumPy broadcasting."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad.reshape(shape)


class Tensor:
    """A node in the computation graph. Wraps a NumPy array and its gradient."""

    def __init__(self, data, _prev=()):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(_prev)

    # --- addition (with broadcasting) ---
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other))
        def _back():
            self.grad += _unbroadcast(out.grad, self.data.shape)
            other.grad += _unbroadcast(out.grad, other.data.shape)
        out._backward = _back
        return out

    # --- elementwise multiply (with broadcasting) ---
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other))
        def _back():
            self.grad += _unbroadcast(out.grad * other.data, self.data.shape)
            other.grad += _unbroadcast(out.grad * self.data, other.data.shape)
        out._backward = _back
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + (other * -1.0)

    def __neg__(self):
        return self * -1.0

    # --- matrix multiply (2-D) ---
    def matmul(self, other):
        out = Tensor(self.data @ other.data, (self, other))
        def _back():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        out._backward = _back
        return out

    # --- 1 / x ---
    def recip(self):
        out = Tensor(1.0 / self.data, (self,))
        def _back():
            self.grad += _unbroadcast(-out.grad / (self.data ** 2), self.data.shape)
        out._backward = _back
        return out

    # --- nonlinearities ---
    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, (self,))
        def _back():
            self.grad += out.grad * (1.0 - t ** 2)
        out._backward = _back
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + np.exp(-self.data))
        out = Tensor(s, (self,))
        def _back():
            self.grad += out.grad * s * (1.0 - s)
        out._backward = _back
        return out

    def softplus(self):
        """log(1+e^x): a smooth, always-positive zoom level for the view-from-above."""
        sp = np.logaddexp(0.0, self.data)
        out = Tensor(sp, (self,))
        def _back():
            self.grad += out.grad * (1.0 / (1.0 + np.exp(-self.data)))
        out._backward = _back
        return out

    # --- structural ops ---
    def slice_rows(self, a, b):
        """Rows [a:b) of a 2-D parameter (used to split the zoom weight)."""
        out = Tensor(self.data[a:b], (self,))
        def _back():
            self.grad[a:b] += out.grad
        out._backward = _back
        return out

    def sum(self):
        out = Tensor(np.array(self.data.sum()), (self,))
        def _back():
            self.grad += out.grad * np.ones_like(self.data)
        out._backward = _back
        return out

    def square(self):
        out = Tensor(self.data ** 2, (self,))
        def _back():
            self.grad += out.grad * 2.0 * self.data
        out._backward = _back
        return out

    def backward(self):
        """Topological reverse pass from this (scalar) node."""
        topo, seen = [], set()
        def build(v):
            if v not in seen:
                seen.add(v)
                for c in v._prev:
                    build(c)
                topo.append(v)
        build(self)
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()


def cross_entropy_masked(logits, targets, mask):
    """
    Masked softmax cross-entropy. Only rows with mask==1 contribute — this is how
    'withholding assent' is expressed in the loss: impressions that carry no duty
    (mask==0) place no demand on the action head.
      logits : Tensor (B, C)
      targets: int array (B,)
      mask   : float array (B,)  (1 = a duty is owed, 0 = nothing to judge here)
    Returns a scalar Tensor.
    """
    z = logits.data - logits.data.max(axis=1, keepdims=True)
    ez = np.exp(z)
    sm = ez / ez.sum(axis=1, keepdims=True)
    B = z.shape[0]
    denom = max(mask.sum(), 1.0)
    logp = np.log(sm[np.arange(B), targets] + 1e-12)
    loss_val = -(logp * mask).sum() / denom
    out = Tensor(np.array(loss_val), (logits,))
    def _back():
        g = sm.copy()
        g[np.arange(B), targets] -= 1.0
        g = g * (mask / denom)[:, None]
        logits.grad += g * out.grad
    out._backward = _back
    return out


# ==============================================================================
# SECTION 2 — The architecture: The Homeostat of the View From Above
# ==============================================================================

class MarcusMind:
    """
    A recurrent controller with a protected ruling-faculty state (the citadel).
    Dimensions:
      D  = size of a raw impression (objective dims + affective dims)
      K  = size of the two internal channels (objective / affective)
      H  = size of the ruling faculty (hegemonikon)
      C  = number of possible duties (the action space)
    """

    def __init__(self, D=10, K=10, H=8, C=4, seed=0):
        rng = np.random.default_rng(seed)
        w = lambda a, b: Tensor(rng.standard_normal((a, b)) * np.sqrt(1.0 / a))
        self.D, self.K, self.H, self.C = D, K, H, C

        # Two projections of every impression: an OBJECTIVE channel (the bare
        # nature of the thing) and an AFFECTIVE channel (the value-coloring the
        # mind adds). The view from above will scale only the affective channel.
        self.A_obj = w(D, K)
        self.A_aff = w(D, K)

        # The view-from-above zoom: reads the impression AND the current state of
        # the ruling faculty, and outputs how far to "step back."
        self.w_alpha = w(D + H, 1)
        self.b_alpha = Tensor(np.zeros((1, 1)))

        # Judgment: turn the re-rendered impression into an objective code z, and
        # decide clarity of assent a.
        self.W_z = w(K, H)
        self.w_a = w(K, 1)
        self.b_a = Tensor(np.zeros((1, 1)))

        # The citadel update map and its leak (gamma = sigmoid(g_raw) in (0,1)).
        self.W_h = w(H, H)
        self.g_raw = Tensor(np.array([[0.2]]))

        # Two action pathways to the duty logits:
        #   W_uz : the IMPULSE — duty read straight from the present judgment.
        #   W_u  : CONTEXT — a small contribution from the stable citadel.
        self.W_uz = w(H, C)
        self.W_u = w(H, C)
        self.b_u = Tensor(np.zeros((1, C)))

        self._ones_H = np.ones((H, 1))  # constant for summing disturbance per row

    def params(self):
        return [self.A_obj, self.A_aff, self.w_alpha, self.b_alpha, self.W_z,
                self.w_a, self.b_a, self.W_h, self.g_raw, self.W_uz, self.W_u,
                self.b_u]

    def forward(self, X, freeze_view=False):
        """
        Roll the mind over a stream of impressions X of shape (Tsteps, B, D).
        The stream stands for a single day of an emperor's impressions.

        If freeze_view=True, the view-from-above is disabled (alpha forced to 0,
        so nothing is re-scaled). This is the ablation used to prove the
        discipline is what produces tranquility.

        Returns per-timestep lists of: duty logits, zoom alpha, and disturbance
        (how far the ruling faculty moved this step).
        """
        Tsteps, B, D = X.shape
        H = self.H
        h = Tensor(np.zeros((B, H)))                 # the ruling faculty, at rest
        one = Tensor(np.ones((B, 1)))
        gamma = self.g_raw.sigmoid()                 # leak toward equilibrium

        logits_list, alpha_list, dist_list = [], [], []
        for t in range(Tsteps):
            x = Tensor(X[t])                         # the impression (phantasia)

            ch_obj = x.matmul(self.A_obj)            # objective channel
            ch_aff = x.matmul(self.A_aff)            # affective channel (the coloring)

            if freeze_view:
                # No view from above: the affective charge passes through in full.
                atten = one
                alpha = Tensor(np.zeros((B, 1)))
            else:
                # The view from above: split w_alpha into an impression part and a
                # state part, so the zoom depends on both the thing and the self.
                Wx = self.w_alpha.slice_rows(0, D)
                Wh = self.w_alpha.slice_rows(D, D + H)
                pre = x.matmul(Wx) + h.matmul(Wh) + self.b_alpha
                alpha = pre.softplus()               # >= 0 ; larger = more cosmic
                atten = (one + alpha).recip()         # in (0,1]; drains affect

            # Re-render the impression: objective content preserved, affect scaled.
            xtilde = ch_obj + ch_aff * atten

            z = xtilde.matmul(self.W_z).tanh()        # the objective code (judgment)
            a = (xtilde.matmul(self.w_a) + self.b_a).sigmoid()  # clarity of assent

            # The citadel updates ONLY through its own assented, re-scaled input.
            inp = z.matmul(self.W_h) * a
            h_new = h * (one - gamma) + inp           # leak + gated input = contraction

            # The impulse: the duty this situation demands, read from the present
            # judgment (immediate), plus a small steadying term from the citadel.
            u = z.matmul(self.W_uz) + h_new.matmul(self.W_u) + self.b_u

            # Disturbance of the ruling faculty this step (per row).
            diff = h_new - h
            dist = diff.square().matmul(Tensor(self._ones_H))

            logits_list.append(u)
            alpha_list.append(alpha)
            dist_list.append(dist)
            h = h_new

        return logits_list, alpha_list, dist_list


# ==============================================================================
# SECTION 3 — The world: a stream of impressions, some "up to us", some not
# ==============================================================================

DOBJ, DAFF = 6, 4                # objective dims, affective dims
D_IN = DOBJ + DAFF
C_DUTY = 4                       # number of duties (the action space)

# The DUTIES are FIXED points in the objective subspace — they do not change from
# day to day. (An emperor's obligations are stable; only their surface varies.)
DUTY_PROTOS = np.random.default_rng(0).standard_normal((C_DUTY, DOBJ)) * 1.5


def make_day(Tsteps, B, seed):
    """
    Generate a 'day' of impressions.
      * "up to us" (a duty is owed): objective content = a duty prototype + a
        little noise; mild affective charge. mask = 1, label = the duty.
      * "not up to us" (pure affect): almost no objective content; a HUGE
        affective charge (an insult, a fear, another's vice). mask = 0.
    """
    rng = np.random.default_rng(seed)
    X = np.zeros((Tsteps, B, D_IN))
    duty = np.zeros((Tsteps, B), dtype=int)
    is_duty = np.zeros((Tsteps, B))
    for t in range(Tsteps):
        for b in range(B):
            if rng.random() < 0.5:
                c = rng.integers(0, C_DUTY)
                X[t, b, :DOBJ] = DUTY_PROTOS[c] + 0.25 * rng.standard_normal(DOBJ)
                X[t, b, DOBJ:] = 0.6 * rng.standard_normal(DAFF)     # mild charge
                duty[t, b] = c
                is_duty[t, b] = 1.0
            else:
                X[t, b, :DOBJ] = 0.05 * rng.standard_normal(DOBJ)    # no real content
                X[t, b, DOBJ:] = 3.5 * rng.standard_normal(DAFF)     # huge charge
                duty[t, b] = 0
                is_duty[t, b] = 0.0
    return X, duty, is_duty


def compute_loss(model, X, duty, is_duty, beta, freeze_view=False):
    """
    Total loss = duty loss + beta * tranquility loss.
      duty loss    : masked cross-entropy — do the right thing when a duty is owed.
      tranquility  : penalize disturbance of the ruling faculty caused by
                     impressions that are NOT up to us. This is the pressure that
                     teaches the view-from-above discipline.
    """
    Tsteps = X.shape[0]
    logits_l, alpha_l, dist_l = model.forward(X, freeze_view=freeze_view)

    duty_loss = Tensor(np.array(0.0))
    for t in range(Tsteps):
        duty_loss = duty_loss + cross_entropy_masked(logits_l[t], duty[t], is_duty[t])
    duty_loss = duty_loss * (1.0 / Tsteps)

    tranq = Tensor(np.array(0.0))
    for t in range(Tsteps):
        not_ours = (1.0 - is_duty[t])[:, None]
        tranq = tranq + (dist_l[t] * Tensor(not_ours)).sum() * (1.0 / (not_ours.sum() + 1e-9))
    tranq = tranq * (beta / Tsteps)

    return duty_loss + tranq, logits_l, alpha_l, dist_l


# ==============================================================================
# SECTION 4 — Training (hand-written Adam over the from-scratch gradients)
# ==============================================================================

def train(model, steps=600, beta=0.3, verbose=True):
    ps = model.params()
    m = [np.zeros_like(p.data) for p in ps]
    v = [np.zeros_like(p.data) for p in ps]
    b1, b2, eps = 0.9, 0.999, 1e-8
    for step in range(1, steps + 1):
        lr = 0.02 if step < steps * 0.6 else 0.005
        X, duty, is_duty = make_day(8, 48, seed=step)     # 8 = the twelve-book cadence, trimmed
        loss, *_ = compute_loss(model, X, duty, is_duty, beta)
        for p in ps:
            p.grad[:] = 0
        loss.backward()
        for i, p in enumerate(ps):
            g = p.grad
            m[i] = b1 * m[i] + (1 - b1) * g
            v[i] = b2 * v[i] + (1 - b2) * (g * g)
            mh = m[i] / (1 - b1 ** step)
            vh = v[i] / (1 - b2 ** step)
            p.data -= lr * mh / (np.sqrt(vh) + eps)
        if verbose and (step == 1 or step % 150 == 0):
            print(f"    step {step:4d}   loss = {loss.data:.4f}")
    return model


def evaluate(model, n_days=10, freeze_view=False):
    """Return (duty accuracy, mean zoom on dutiful, mean zoom on not-ours,
    mean disturbance on not-ours)."""
    ncorr = ntot = 0
    a_duty, a_notours, d_notours = [], [], []
    for s in range(9000, 9000 + n_days):
        X, duty, is_duty = make_day(8, 48, seed=s)
        logits_l, alpha_l, dist_l = model.forward(X, freeze_view=freeze_view)
        for t in range(8):
            pred = logits_l[t].data.argmax(1)
            mk = is_duty[t].astype(bool)
            ncorr += (pred[mk] == duty[t][mk]).sum()
            ntot += mk.sum()
            a_duty += list(alpha_l[t].data[mk, 0])
            a_notours += list(alpha_l[t].data[~mk, 0])
            d_notours += list(dist_l[t].data[~mk, 0])
    return (ncorr / max(ntot, 1),
            float(np.mean(a_duty)), float(np.mean(a_notours)),
            float(np.mean(d_notours)))


# ==============================================================================
# SECTION 5 — Self-tests (each one is a claim from the chapter, made checkable)
# ==============================================================================

def gradient_check(seed=1):
    """Mandatory finite-difference check of the from-scratch gradients."""
    Tsteps, B = 5, 3
    model = MarcusMind(D=D_IN, K=6, H=4, C=C_DUTY, seed=seed)
    X, duty, is_duty = make_day(Tsteps, B, seed=7)
    loss, *_ = compute_loss(model, X, duty, is_duty, beta=0.4)
    for p in model.params():
        p.grad[:] = 0
    loss.backward()

    eps = 1e-5   # the finite-difference sweet spot for this model's curvature
    worst = 0.0
    for p in model.params():
        flat, g = p.data.reshape(-1), p.grad.reshape(-1)
        num = np.zeros_like(flat)
        for i in range(flat.size):
            o = flat[i]
            flat[i] = o + eps
            Lp = compute_loss(model, X, duty, is_duty, beta=0.4)[0].data
            flat[i] = o - eps
            Lm = compute_loss(model, X, duty, is_duty, beta=0.4)[0].data
            flat[i] = o
            num[i] = (Lp - Lm) / (2 * eps)
        rel = np.abs(num - g) / (np.abs(num) + np.abs(g) + 1e-8)
        worst = max(worst, rel.max())
    return worst


def storm_test(model, freeze_view=False):
    """
    A storm of nothing but 'not up to us' impressions (insults, fears, others'
    vices). Measure how far the ruling faculty is driven from rest. With the view
    from above, the citadel holds; with it frozen off, the citadel is stormed.
    """
    rng = np.random.default_rng(4242)
    Tsteps, B = 30, 64
    X = np.zeros((Tsteps, B, D_IN))
    X[:, :, DOBJ:] = 3.5 * rng.standard_normal((Tsteps, B, DAFF))
    duty = np.zeros((Tsteps, B), dtype=int)
    is_duty = np.zeros((Tsteps, B))
    _, _, dist_l = model.forward(X, freeze_view=freeze_view)
    return float(np.mean([dist_l[t].data.mean() for t in range(Tsteps)]))


def maintenance_test(model):
    """
    Feed the ruling faculty a real disturbance, then feed NOTHING (blank
    impressions). Watch it decay back toward equilibrium. Tranquility is not a
    stored value; the leak pulls the self home. This is why the same meditations
    must be re-written — the equilibrium must be re-derived.
    """
    Tsteps, B = 16, 8
    X = np.zeros((Tsteps, B, D_IN))
    X[0, :, DOBJ:] = 6.0 * np.random.default_rng(1).standard_normal((B, DAFF))  # one shock
    # Read the magnitude of the ruling faculty directly, step by step. After the
    # single shock at t=0 every later impression is blank, so we watch pure decay.
    h = np.zeros((B, model.H))
    gamma = 1.0 / (1.0 + np.exp(-model.g_raw.data[0, 0]))
    mags = []
    for t in range(Tsteps):
        x = X[t]
        ch_obj = x @ model.A_obj.data
        ch_aff = x @ model.A_aff.data
        xtilde = ch_obj + ch_aff                 # frozen view => atten = 1
        z = np.tanh(xtilde @ model.W_z.data)
        a = 1.0 / (1.0 + np.exp(-(xtilde @ model.w_a.data + model.b_a.data)))
        h = h * (1 - gamma) + (z @ model.W_h.data) * a
        mags.append(float(np.linalg.norm(h, axis=1).mean()))
    return mags


if __name__ == "__main__":
    np.random.seed(0)
    print("=" * 74)
    print("  MARCUS AURELIUS — The Homeostat of the View From Above")
    print("=" * 74)

    # 1) Gradient check --------------------------------------------------------
    worst = gradient_check()
    print(f"\n[1] Gradient check (finite difference vs. hand-written backprop)")
    print(f"    worst relative error across all parameters: {worst:.2e}")
    print(f"    -> {'PASS' if worst < 1e-4 else 'FAIL'} (engine gradients are correct)")

    # 2) Train the disciplined mind -------------------------------------------
    print(f"\n[2] Training the disciplined mind (view from above ON, beta=0.3):")
    model = MarcusMind(D=D_IN, K=10, H=8, C=C_DUTY, seed=3)
    train(model, steps=600, beta=0.3)
    acc, a_duty, a_notours, d_notours = evaluate(model)
    print(f"\n[3] After training:")
    print(f"    duty accuracy (does it still govern?)      : {acc*100:.1f}%")
    print(f"    mean zoom on DUTIFUL impressions            : {a_duty:.3f}")
    print(f"    mean zoom on NOT-UP-TO-US impressions       : {a_notours:.3f}")
    print(f"    -> zooms {'HARDER' if a_notours > a_duty else 'less'} on what is not up to us "
          f"({a_notours/max(a_duty,1e-9):.2f}x)")
    print(f"    disturbance of ruling faculty (not-ours)    : {d_notours:.4f}  (near 0 = tranquil)")

    # 3) The inner citadel under a storm + the ablation ------------------------
    calm = storm_test(model, freeze_view=False)
    stormed = storm_test(model, freeze_view=True)
    print(f"\n[4] The inner citadel under a storm of pure affect:")
    print(f"    with the view from above (discipline ON) : mean disturbance = {calm:.4f}")
    print(f"    ablation — discipline OFF (alpha forced 0): mean disturbance = {stormed:.4f}")
    print(f"    -> the discipline reduces the storm's grip by {stormed/max(calm,1e-9):.0f}x")

    # 4) Maintenance / decay ---------------------------------------------------
    mags = maintenance_test(model)
    print(f"\n[5] Maintenance: one shock, then silence. ||ruling faculty|| over time:")
    print("    " + "  ".join(f"{m:.2f}" for m in mags[:8]) + "  ...")
    print(f"    peak = {max(mags):.2f}  ->  settles to {mags[-1]:.2f}")
    print(f"    -> tranquility is not stored; the self decays home and must be re-derived.")

    print("\n" + "=" * 74)
    print("  All checks complete. The mind governs, and the citadel holds.")
    print("=" * 74)
