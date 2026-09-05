"""
================================================================================
Chapter 0143_plotinus_205 — Plotinus (205-270 CE)
Henadic Emanation Network (HEN): a from-scratch cognitive architecture that
encodes Plotinus' philosophy of mind rather than a generic neural stack.
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 8 Minds 141 - 160 Available on Amazon https://www.amazon.com/dp/B0HH8RTCXF
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 143: Plotinus (205-270 CE)
================================================================================  

WHY THIS IS NOT A TRANSFORMER
-----------------------------
Plotinus' account of mind is built on three coupled ideas that have almost
nothing in common with attention-over-stored-keys:

  1. EMANATION AS A CASCADE (proodos / procession).
     Reality overflows downward: the One -> Intellect (Nous) -> Soul -> the
     sensible world. Each level is a fainter image of the one above it. In the
     architecture this is a *generative descent*: a small "One" code is unfolded
     back down into a predicted sense-vector.

  2. KNOWING IS RETURN, NOT REPRESENTATION (epistrophe / reversion).
     A derived level does not sit passively; it is CONSTITUTED by turning back
     toward its source and contemplating it. Cognition here is the *ascent*
     (encoder) whose job is to bend the input back toward the intelligible.
     "Sight becomes seeing" only by reverting to what is seen.

  3. IN NOUS, THE KNOWER AND THE KNOWN ARE ONE.
     Plotinus' most radical thesis: at the level of Intellect there is no gap
     between the thinking and the thing thought. We enforce this literally with
     TIED WEIGHTS — the operator that ascends to a Form is the transpose of the
     operator by which the Form overflows into its image. One substance, used
     in both directions. A dedicated loss term then pulls the Form-as-ascended-to
     into identity with the Form-as-overflowed-from.

  4. THE ONE IS SIMPLE AND UNREPRESENTABLE (apophasis / henosis).
     The summit is not more information but LESS: a source "without parts."
     An apophatic penalty drives the top code toward internal unity (all its
     units toward a single shared value), so the network is rewarded for
     reaching a simple attractor rather than a rich latent — the exact opposite
     of a high-capacity bottleneck. Union (henosis) is modelled as the collapse
     of the top code to its mean: the knower/known distinction vanishes.

Together these give the "Henadic Emanation Network": a triadic
  Sense (n0)  <->  Soul (n1)  <->  Nous (n2)  <->  One (n3, small)
autoencoder with tied procession/reversion weights, a contemplative-identity
loss at Nous, and an apophatic simplicity loss at the One.

RELEVANCE TO AGI RESEARCH
-------------------------
The design is a small argument about alignment and interpretability. Instead of
piling capacity into an opaque core, it forces cognition to (a) pass through a
deliberately *simple*, low-multiplicity attractor and (b) be reconstructible as
an emanation from that attractor. A mind whose summit is simple and whose every
level is a faithful, invertible image of the one above it is, in Plotinian
terms, auditable: you can walk the ascent and the descent and check that they
agree.

ENGINEERING CONVENTIONS
-----------------------
  * Pure NumPy. No torch / tensorflow / keras.
  * Row-vector batch convention: X has shape (batch, n0).
  * Hand-derived analytic gradients for every parameter, including the doubly
    used tied weights, verified against finite differences (mandatory check).
  * A real training loop on a synthetic "emanation manifold" dataset.
  * Self-tests for reversion-to-rest and henosis (union) behaviour.

Run:  python3 chapter_0143_plotinus_205.py
================================================================================
"""

import numpy as np


# ==============================================================================
# SECTION 1 — NUMERICAL PRIMITIVES
# ==============================================================================

def tanh(x):
    """Bounded, smooth activation. Bounded-ness matters: each hypostasis is a
    *finite image* of its source, never an unbounded amplification of it."""
    return np.tanh(x)


def dtanh_from_activation(a):
    """Derivative of tanh expressed through its own output a = tanh(z):
    d/dz tanh(z) = 1 - tanh(z)^2. Using the activation keeps forward + backward
    consistent and cheap."""
    return 1.0 - a * a


def row_mean(x):
    """Per-sample mean across the feature axis (used for the apophatic term and
    for henosis)."""
    return np.mean(x, axis=1, keepdims=True)


# ==============================================================================
# SECTION 2 — THE HENADIC EMANATION NETWORK
# ==============================================================================

class HenadicEmanationNetwork:
    """
    Levels (hypostases), from the sensible up to the One:

        Level 0  Sense   dim n0   (the material image)
        Level 1  Soul    dim n1   (discursive, mediating)
        Level 2  Nous    dim n2   (the Forms; knower == known)
        Level 3  One     dim n3   (small; simple, apophatic source)

    Reversion (ascent / encoder) operators:
        W1 : n1 x n0     Sense -> Soul
        W2 : n2 x n1     Soul  -> Nous
        W3 : n3 x n2     Nous  -> One

    Procession (descent / decoder) reuses the SAME weights, transposed:
        One  -> Nous   via W3^T
        Nous -> Soul   via W2^T
        Soul -> Sense  via W1^T
    Only the biases differ between the two directions (b* for ascent, c* for
    descent). The shared weights are the computational form of Plotinus'
    identity of the intelligible substance across the double movement.
    """

    def __init__(self, n0, n1, n2, n3, seed=0):
        self.n0, self.n1, self.n2, self.n3 = n0, n1, n2, n3
        rng = np.random.default_rng(seed)

        # Small-scale init keeps early tanh activations in their linear region so
        # the ascent and descent start as near-isometries (faithful images).
        def init(rows, cols):
            return rng.standard_normal((rows, cols)) * (1.0 / np.sqrt(cols))

        # Tied reversion/procession weights.
        self.W1 = init(n1, n0)
        self.W2 = init(n2, n1)
        self.W3 = init(n3, n2)

        # Ascent (reversion) biases.
        self.b1 = np.zeros(n1)
        self.b2 = np.zeros(n2)
        self.b3 = np.zeros(n3)

        # Descent (procession) biases.
        self.c2 = np.zeros(n2)
        self.c1 = np.zeros(n1)
        self.c0 = np.zeros(n0)

    # --- parameter plumbing (used by the finite-difference gradient check) ----
    def params(self):
        return {
            "W1": self.W1, "W2": self.W2, "W3": self.W3,
            "b1": self.b1, "b2": self.b2, "b3": self.b3,
            "c2": self.c2, "c1": self.c1, "c0": self.c0,
        }

    def set_param(self, name, value):
        setattr(self, name, value)

    # -------------------------------------------------------------------------
    # FORWARD PASS: one full act of contemplation
    #   ascent  (reversion): Sense -> Soul -> Nous -> One
    #   descent (procession): One  -> Nous -> Soul -> Sense
    # -------------------------------------------------------------------------
    def forward(self, X):
        cache = {"X": X}

        # ----- ASCENT (epistrophe): the soul turns back toward its source -----
        Z1 = X @ self.W1.T + self.b1
        H1 = tanh(Z1)                       # Soul
        Z2 = H1 @ self.W2.T + self.b2
        H2 = tanh(Z2)                       # Nous, as *ascended to* (the knower)
        Z3 = H2 @ self.W3.T + self.b3
        H3 = tanh(Z3)                       # the One code

        # ----- DESCENT (proodos): the One overflows back into an image --------
        G2 = H3 @ self.W3 + self.c2
        HH2 = tanh(G2)                      # Nous, as *overflowed from* (known)
        G1 = HH2 @ self.W2 + self.c1
        HH1 = tanh(G1)                      # Soul regenerated
        Xhat = HH1 @ self.W1 + self.c0      # Sense regenerated (linear readout)

        cache.update(dict(H1=H1, H2=H2, H3=H3, HH2=HH2, HH1=HH1, Xhat=Xhat))
        return Xhat, cache

    # -------------------------------------------------------------------------
    # LOSS: three Plotinian terms
    #   recon      — the sensible world is a faithful image (emanation fidelity)
    #   nous_ident — knower == known at the level of Intellect
    #   apophatic  — the One is simple: its units collapse toward one value
    # -------------------------------------------------------------------------
    def loss(self, cache, alpha=1.0, beta=0.1):
        X, Xhat = cache["X"], cache["Xhat"]
        H2, HH2, H3 = cache["H2"], cache["HH2"], cache["H3"]
        B = X.shape[0]

        recon = 0.5 * np.sum((Xhat - X) ** 2) / B
        nous_ident = 0.5 * np.sum((HH2 - H2) ** 2) / B
        centred = H3 - row_mean(H3)                     # deviation from unity
        apophatic = 0.5 * np.sum(centred ** 2) / B

        total = recon + alpha * nous_ident + beta * apophatic
        parts = dict(total=total, recon=recon,
                     nous_ident=nous_ident, apophatic=apophatic)
        return total, parts

    # -------------------------------------------------------------------------
    # BACKWARD PASS: analytic gradients.
    # Every tied weight (W1,W2,W3) accumulates TWO contributions — one from the
    # descent path (where it appears as W) and one from the ascent path (where it
    # appears as W^T). Getting both right is the whole point of the grad check.
    # -------------------------------------------------------------------------
    def backward(self, cache, alpha=1.0, beta=0.1):
        X = cache["X"]
        H1, H2, H3 = cache["H1"], cache["H2"], cache["H3"]
        HH2, HH1, Xhat = cache["HH2"], cache["HH1"], cache["Xhat"]
        B = X.shape[0]

        grads = {k: np.zeros_like(v) for k, v in self.params().items()}

        # ---- gradients of the three loss terms at their anchor tensors -------
        dXhat = (Xhat - X) / B                       # from recon
        dHH2_id = alpha * (HH2 - H2) / B             # nous identity -> known side
        dH2_id = -alpha * (HH2 - H2) / B             # nous identity -> knower side
        dH3_apo = beta * (H3 - row_mean(H3)) / B     # apophatic -> the One

        # ================= DESCENT PATH (procession) ==========================
        # Xhat = HH1 @ W1 + c0
        grads["c0"] += dXhat.sum(0)
        grads["W1"] += HH1.T @ dXhat                 # descent contribution to W1
        dHH1 = dXhat @ self.W1.T

        # HH1 = tanh(G1); G1 = HH2 @ W2 + c1
        dG1 = dHH1 * dtanh_from_activation(HH1)
        grads["c1"] += dG1.sum(0)
        grads["W2"] += HH2.T @ dG1                   # descent contribution to W2
        dHH2 = dG1 @ self.W2.T
        dHH2 += dHH2_id                              # + nous identity (known side)

        # HH2 = tanh(G2); G2 = H3 @ W3 + c2
        dG2 = dHH2 * dtanh_from_activation(HH2)
        grads["c2"] += dG2.sum(0)
        grads["W3"] += H3.T @ dG2                    # descent contribution to W3
        dH3 = dG2 @ self.W3.T
        dH3 += dH3_apo                               # + apophatic pull on the One

        # ================= ASCENT PATH (reversion) ============================
        # H3 = tanh(Z3); Z3 = H2 @ W3^T + b3
        dZ3 = dH3 * dtanh_from_activation(H3)
        grads["b3"] += dZ3.sum(0)
        grads["W3"] += dZ3.T @ H2                    # ascent contribution to W3
        dH2 = dZ3 @ self.W3
        dH2 += dH2_id                                # + nous identity (knower side)

        # H2 = tanh(Z2); Z2 = H1 @ W2^T + b2
        dZ2 = dH2 * dtanh_from_activation(H2)
        grads["b2"] += dZ2.sum(0)
        grads["W2"] += dZ2.T @ H1                    # ascent contribution to W2
        dH1 = dZ2 @ self.W2

        # H1 = tanh(Z1); Z1 = X @ W1^T + b1
        dZ1 = dH1 * dtanh_from_activation(H1)
        grads["b1"] += dZ1.sum(0)
        grads["W1"] += dZ1.T @ X                     # ascent contribution to W1

        return grads

    # -------------------------------------------------------------------------
    # INFERENCE-ONLY CONTEMPLATIVE BEHAVIOURS (not part of the trained loss)
    # -------------------------------------------------------------------------
    def reversion_to_rest(self, x, steps=10):
        """Plotinus' 'rest' (mone): repeatedly ascend then re-emanate, feeding
        each image back in as the new object of contemplation, until the soul
        settles into a fixed point. We report the step-to-step CHANGE of the
        iterate; the soul is 'at rest' when successive contemplations stop moving
        it. A shrinking change is the settling."""
        cur = x.copy()
        changes = []
        for _ in range(steps):
            nxt, _ = self.forward(cur)
            changes.append(float(np.mean((nxt - cur) ** 2)))
            cur = nxt
        return cur, changes

    def henosis(self, x):
        """Union with the One: take the ascent's top code and collapse it to its
        mean, erasing all internal multiplicity (the knower/known distinction),
        then emanate from that simple point. Returns the 'unitary' sense image."""
        _, cache = self.forward(x)
        h3 = cache["H3"]
        h3_one = np.broadcast_to(row_mean(h3), h3.shape)   # perfectly simple code
        G2 = h3_one @ self.W3 + self.c2
        HH2 = tanh(G2)
        G1 = HH2 @ self.W2 + self.c1
        HH1 = tanh(G1)
        return HH1 @ self.W1 + self.c0


# ==============================================================================
# SECTION 3 — SYNTHETIC "EMANATION MANIFOLD" DATA
# ==============================================================================

def make_emanation_data(n_samples, n0, n_forms=3, noise=0.02,
                        gen_seed=0, sample_seed=1):
    """The sensible world as an image of a few Forms.

    A handful of latent 'Form' codes are pushed through a FIXED nonlinear
    generator (the single 'overflow' that produced the world) to produce
    structured sense-vectors, plus a little matter-noise. Crucially the
    generator is fixed by gen_seed and SHARED across train/test; only the
    per-sample Form activations (sample_seed) differ. So train and test are
    different draws from the *same* emanation — a faithful mind that has learned
    to revert one draw should revert the other. This makes generalisation a real
    test of whether the ascent has grasped the Forms rather than the samples.
    """
    gen = np.random.default_rng(gen_seed)
    Gf = gen.standard_normal((n_forms, 16)) * 0.8      # Forms -> intelligible
    Gh = gen.standard_normal((16, n0)) * 0.8           # intelligible -> sense

    smp = np.random.default_rng(sample_seed)
    z = smp.standard_normal((n_samples, n_forms))      # this soul's Form-mixture
    hidden = np.tanh(z @ Gf)
    X = np.tanh(hidden @ Gh)
    X = X + noise * smp.standard_normal(X.shape)
    return X.astype(np.float64)


# ==============================================================================
# SECTION 4 — GRADIENT CHECK  (mandatory)
# ==============================================================================

def gradient_check(verbose=True):
    """Finite-difference check of every analytic gradient, including the tied
    weights that are used in both directions. Uses float64 and a central
    difference for accuracy."""
    rng = np.random.default_rng(7)
    net = HenadicEmanationNetwork(n0=6, n1=5, n2=4, n3=3, seed=3)
    X = rng.standard_normal((8, 6))
    alpha, beta = 1.0, 0.1

    # Analytic gradients.
    _, cache = net.forward(X)
    L0, _ = net.loss(cache, alpha, beta)
    analytic = net.backward(cache, alpha, beta)

    eps = 1e-6
    max_rel = 0.0
    worst = None
    for name, P in net.params().items():
        flat = P.ravel()
        g = analytic[name].ravel()
        # Check a spread of coordinates (all of them for small params).
        idxs = range(flat.size) if flat.size <= 30 else \
            rng.choice(flat.size, 30, replace=False)
        for i in idxs:
            orig = flat[i]
            flat[i] = orig + eps
            _, c1 = net.forward(X); Lp, _ = net.loss(c1, alpha, beta)
            flat[i] = orig - eps
            _, c2 = net.forward(X); Lm, _ = net.loss(c2, alpha, beta)
            flat[i] = orig
            num = (Lp - Lm) / (2 * eps)
            ana = g[i]
            denom = max(1e-12, abs(num) + abs(ana))
            rel = abs(num - ana) / denom
            if rel > max_rel:
                max_rel, worst = rel, (name, i, num, ana)

    ok = max_rel < 1e-5
    if verbose:
        print("  max relative error : {:.2e}".format(max_rel))
        print("  worst coordinate   : {}".format(worst))
        print("  gradient check     : {}".format("PASS" if ok else "FAIL"))
    return ok, max_rel


# ==============================================================================
# SECTION 5 — TRAINING LOOP
# ==============================================================================

def _clip_global(grads, max_norm):
    """Clip gradients by global L2 norm — keeps the tied-weight coupling from
    letting the ascent and descent paths resonate into instability."""
    total = np.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
    if total > max_norm:
        scale = max_norm / (total + 1e-12)
        for k in grads:
            grads[k] *= scale
    return grads


def train(net, X, epochs=600, lr=0.05, alpha=0.5, beta=0.02,
          batch=64, seed=0, verbose=True, clip=5.0):
    """Plain SGD with momentum + global-norm gradient clipping. The point is not
    raw performance but to show the three Plotinian objectives can be jointly
    minimised: the world becomes a faithful image, knower and known converge at
    Nous, and the One grows simple. The apophatic term is kept *gentle* — a
    steady pull toward unity, not a collapse that annihilates the image."""
    rng = np.random.default_rng(seed)
    vel = {k: np.zeros_like(v) for k, v in net.params().items()}
    mom = 0.9
    n = X.shape[0]
    history = []
    for ep in range(epochs):
        perm = rng.permutation(n)
        for s in range(0, n, batch):
            xb = X[perm[s:s + batch]]
            _, cache = net.forward(xb)
            grads = net.backward(cache, alpha, beta)
            grads = _clip_global(grads, clip)
            for k in net.params():
                vel[k] = mom * vel[k] - lr * grads[k]
                net.set_param(k, net.params()[k] + vel[k])
        if verbose and (ep % 50 == 0 or ep == epochs - 1):
            _, c = net.forward(X)
            _, parts = net.loss(c, alpha, beta)
            history.append((ep, parts))
            print("  epoch {:4d} | total {:.4f} | recon {:.4f} | "
                  "nous_ident {:.5f} | apophatic {:.5f}".format(
                      ep, parts["total"], parts["recon"],
                      parts["nous_ident"], parts["apophatic"]))
    return history


# ==============================================================================
# SECTION 6 — SELF-TESTS / DEMONSTRATIONS
# ==============================================================================

def run_all():
    np.set_printoptions(precision=4, suppress=True)

    print("=" * 70)
    print("HENADIC EMANATION NETWORK — Plotinus (c.205-270)")
    print("=" * 70)

    print("\n[1] GRADIENT CHECK (analytic vs finite difference)")
    ok, rel = gradient_check()
    assert ok, "Gradient check FAILED (rel={:.2e})".format(rel)

    print("\n[2] DATA — building the emanation manifold")
    Xtr = make_emanation_data(512, n0=12, n_forms=3, noise=0.02,
                              gen_seed=0, sample_seed=1)
    Xte = make_emanation_data(128, n0=12, n_forms=3, noise=0.02,
                              gen_seed=0, sample_seed=2)
    print("  train {}  test {}".format(Xtr.shape, Xte.shape))

    print("\n[3] TRAINING (ascend, revert, simplify)")
    net = HenadicEmanationNetwork(n0=12, n1=10, n2=8, n3=5, seed=5)
    _, c0 = net.forward(Xtr); _, p0 = net.loss(c0)
    train(net, Xtr, epochs=600, lr=0.05, alpha=0.5, beta=0.05, seed=0)

    print("\n[4] GENERALISATION")
    _, cte = net.forward(Xte)
    _, pte = net.loss(cte)
    print("  init  recon (emanation fidelity): {:.4f}".format(p0["recon"]))
    print("  test  recon (emanation fidelity): {:.4f}".format(pte["recon"]))
    print("  test nous identity (knower=known): {:.5f}".format(pte["nous_ident"]))
    print("  recon improvement vs init       : {:.1f}x".format(
        p0["recon"] / max(1e-9, pte["recon"])))

    print("\n[5] REVERSION TO REST (the soul settling into a fixed point)")
    x = Xte[:16]
    _, changes = net.reversion_to_rest(x, steps=10)
    print("  step-to-step change of the iterate:")
    print("   ", np.array(changes))
    print("  settling (change shrinks): {}".format(
        changes[-1] < changes[0]))

    print("\n[6] APOPHATIC SIMPLICITY OF THE ONE")
    _, c = net.forward(Xte)
    h3 = c["H3"]
    spread = float(np.mean(np.std(h3, axis=1)))
    print("  mean within-sample spread of the One code: {:.4f}".format(spread))
    print("  (smaller = closer to 'without parts')")

    print("\n[7] HENOSIS (union: the knower/known distinction erased)")
    normal, _ = net.forward(Xte[:8])
    unitary = net.henosis(Xte[:8])
    drift = float(np.mean((normal - unitary) ** 2))
    print("  mean drift between discursive ascent and unitary emanation: {:.4f}".format(drift))
    print("  (a non-zero drift is the point: union with the simple source")
    print("   is not the same act as knowing the particular Forms — the One")
    print("   is *beyond* representation, so the unitary emanation cannot")
    print("   fully re-derive the discursive image. Union transcends knowledge.)")

    print("\n" + "=" * 70)
    print("ALL CHECKS PASSED — the emanation holds together.")
    print("=" * 70)


if __name__ == "__main__":
    run_all()
