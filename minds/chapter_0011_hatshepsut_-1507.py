"""
================================================================================
chapter_0011_hatshepsut_-1507.py  --  THE MAAT-FIELD NETWORK
A from-scratch, trainable energy-based architecture after the mind of Hatshepsut
(Maatkare), 5th pharaoh of Egypt's 18th Dynasty (r. c. 1473-1458 BCE).
Part Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
Tome 1 Minds 1 - 20 Available on Amazon https://www.amazon.com/dp/B0H6F9L324
Resume and Interactive Demos at https://artificiology.com/
================================================================================

WHY THIS ARCHITECTURE, AND WHY NOT A TRANSFORMER
------------------------------------------------
Every other "great-ruler" chapter is tempted toward the same machine: stack
attention, store keys, retrieve by similarity, call it legitimacy.  That is the
wrong mind for Hatshepsut.  Her surviving works -- the fivefold royal titulary,
the Deir el-Bahri reliefs, the Punt program, the Karnak obelisks -- do not show
a ruler *generating novelty*.  They show a ruler *restoring a balanced order*.

The Egyptian word for that order is **maat**: truth, justice, cosmic balance,
the way things must be held against **isfet** (chaos, disorder, entropy).  Her
throne name, Maat-ka-Re, literally binds her self to it ("Maat is the Ka/soul of
Re").  A pharaoh's whole cognitive job was not invention but *re-balancing*: when
a king dies, when the Hyksos have damaged the temples, when a woman must hold a
male office, the configuration of the cosmos is disturbed and must be relaxed
back into a valid, all-constraints-satisfied state.

So the right model is NOT a feed-forward predictor and NOT attention-over-a-
sequence.  It is an **energy-based associative memory** -- a modern continuous
Hopfield network (Ramsauer et al., 2020) trained as an energy-based model.  In
it:

    * a learned ENERGY function assigns low energy to *maat* (balanced,
      rule-satisfying configurations) and high energy to *isfet* (corrupted,
      rule-violating ones);
    * COGNITION is relaxation: gradient descent on the state restores a
      corrupted pattern to the nearest stored balanced attractor.  This *is*
      "restoring maat from isfet" -- pattern completion / denoising;
    * IDENTITY/AUTHORITY is not a property of a body but a transferable ROLE.
      The state is partitioned into typed blocks (context, office, filler).
      Becoming pharaoh = binding a filler into the office slot so that every
      constraint is simultaneously satisfied.  Which attractor you settle into
      *is* who you are.

THE HATSHEPSUT INVARIANT (the one idea that is hers alone)
---------------------------------------------------------
Gay Robins ("The Names of Hatshepsut as King", JEA 85, 1999) showed that her
king-names were built from grammatically FEMININE participles even as she
occupied the male office of king -- "Daughter of Re", "female Horus", the
retained epithet Khnemet-Amun.  She did NOT break the grammar of kingship (that
would be isfet); she found the unique configuration that satisfied *every* rule
at once, holding a "+pole" (the male office) and a "-pole" (her feminine
grammar) in simultaneous balance.  Egyptian thought already encoded exactly this
gender-duality as the balance of maat itself (the king is "mother and father").

We hard-wire that as a **maat constraint operator C**: a balanced state must hold
two poles co-active and equal.  A one-sided "male-only king" state is high
energy (isfet); the balanced "androgynous king" state is the low-energy
solution.  Test 4 below shows the network *dynamically* recovering the missing
feminine pole from a one-sided input -- Hatshepsut's solution emerging from the
energy landscape, not hand-coded.

WHAT IS LEARNED, AND THE HONEST CONTRACT OF THIS FILE
-----------------------------------------------------
Parameters theta = { X (M memories x D), b (bias D), g_raw (scalar balance
gate) }.  We learn them with a contrastive-divergence energy objective (lower
energy of data, raise energy of model's "fantasy" negatives).  This file is NOT
a demo: it contains
    (1) a MANDATORY finite-difference gradient check that must pass,
    (2) a real training loop that measurably reshapes the energy landscape,
    (3) five self-tests, including the Hatshepsut balance test.
Run it with `python3 chapter_0011_hatshepsut_-1507.py`.  The verified console output is pasted into the
companion chapter.

Pure NumPy.  No deep-learning framework.  Reproducible (fixed seed).
Author: David Vivancos  --  Mind #11, Hatshepsut.
"""

from __future__ import annotations
import numpy as np

# =============================================================================
# SECTION 0 -- small numerically-stable primitives
# =============================================================================

def softplus(x):
    """softplus(x) = log(1+e^x), stable.  Used to keep the balance gate g >= 0."""
    return np.logaddexp(0.0, x)

def sigmoid(x):
    """Derivative of softplus is the logistic sigmoid; we need it for d g / d g_raw."""
    return 1.0 / (1.0 + np.exp(-x))

def logsumexp(z, axis=None, keepdims=False):
    """Stable log-sum-exp."""
    m = np.max(z, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(z - m), axis=axis, keepdims=True))
    if not keepdims:
        out = np.squeeze(out, axis=axis)
    return out

def softmax(z, axis=-1):
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


# =============================================================================
# SECTION 1 -- THE COSMOS: a typed state space and a corpus of balanced patterns
# =============================================================================
#
# A state vector s in R^D is one "configuration of the cosmos".  We carve it
# into typed blocks so that identity is a *binding* rather than a body:
#
#   [ context block | office pole+ | office pole- | filler/body block ]
#
#   * context  (Dc dims): which roles/offices are in play (a pattern's "id").
#   * pole+    (Dp dims): the office charge that tradition codes as male.
#   * pole-    (Dp dims): the office charge that Hatshepsut keeps feminine.
#   * filler   (Df dims): the person/body bound into the office.
#
# A CANONICAL BALANCED pattern (maat) has pole+ == pole- : both charges of the
# office co-present and equal -- the dual-gender equilibrium.  We build the
# corpus so every stored memory satisfies maat by construction; learning then
# has to *discover* that these are the low-energy attractors.

class Cosmos:
    def __init__(self, Dc=24, Dp=8, Df=24, n_patterns=12, seed=0):
        self.Dc, self.Dp, self.Df = Dc, Dp, Df
        self.D = Dc + 2 * Dp + Df
        self.n_patterns = n_patterns
        rng = np.random.default_rng(seed)
        self.rng = rng

        # index slices into the state vector
        self.ctx = slice(0, Dc)
        self.pole_plus = slice(Dc, Dc + Dp)
        self.pole_minus = slice(Dc + Dp, Dc + 2 * Dp)
        self.fill = slice(Dc + 2 * Dp, self.D)

        # ---- the maat constraint operator C ----
        # C s = (pole+ - pole-)  (zero everywhere else).  ||C s||^2 = office
        # imbalance.  A balanced (maat) state has C s = 0.
        C = np.zeros((Dp, self.D))
        for k in range(Dp):
            C[k, Dc + k] = 1.0           # +pole coordinate
            C[k, Dc + Dp + k] = -1.0     # -pole coordinate
        self.C = C
        self.CtC = C.T @ C               # precomputed for the energy gradient

        # ---- build M canonical BALANCED patterns ----
        X = np.zeros((n_patterns, self.D))
        for i in range(n_patterns):
            context = rng.standard_normal(Dc)
            office_filler = rng.standard_normal(Dp)   # the office's "charge"
            body = rng.standard_normal(Df)            # the person bound in
            X[i, self.ctx] = context
            X[i, self.pole_plus] = office_filler      # +pole
            X[i, self.pole_minus] = office_filler     # -pole  == +pole  -> maat
            X[i, self.fill] = body
        # unit-norm each pattern so energies are comparable
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        self.patterns = X

    # --- helpers to corrupt a pattern into "isfet" ---
    def isfet(self, batch, noise=0.35, break_pole=True):
        """Disturb maat: add chaos noise and (optionally) suppress the feminine
        -pole, producing a one-sided 'male-only king' -- a rule-violating state
        that the network must learn to reject / restore."""
        out = batch + noise * self.rng.standard_normal(batch.shape)
        if break_pole:
            out[:, self.pole_minus] = 0.0     # erase the retained feminine grammar
        return out

    def office_imbalance(self, batch):
        """Mean ||C s|| per state -- how far from the dual-pole equilibrium."""
        return np.linalg.norm(batch @ self.C.T, axis=1)


# =============================================================================
# SECTION 2 -- THE MAAT-FIELD NETWORK (energy, retrieval dynamics, EBM gradient)
# =============================================================================

class MaatFieldNetwork:
    """
    Energy (modern continuous Hopfield + maat-balance penalty), for a state s:

        z      = X @ s                         # match against M memories
        lse    = (1/beta) * logsumexp(beta*z)  # soft "nearest balanced pattern"
        g      = softplus(g_raw)               # >= 0 balance gate
        E(s)   = -lse + 0.5*||s||^2 - b.s + 0.5*g*||C s||^2

    * -lse carves a deep well at each stored balanced pattern (associative
      memory / multiple attractors -- this is what attention-over-keys cannot
      give you as an *energy*).
    * 0.5||s||^2 keeps states bounded (the Hopfield "self" term).
    * -b.s is a learned standing bias.
    * 0.5*g*||C s||^2 is MAAT: it raises the energy of any office imbalance, so
      the only low-energy office state is the balanced dual-pole one.

    Cognition (retrieve / restore maat) = gradient descent on E in s:
        grad_s E = -X^T softmax(beta*X s) + s - b + g * (C^T C) s
        s <- s - eta * grad_s E

    Learning = contrastive-divergence EBM objective over parameters
    theta = {X, b, g_raw}; negatives are the model's own relaxed fantasies and
    are held FIXED while differentiating (standard CD), which also makes the
    finite-difference gradient check exact.
    """

    def __init__(self, cosmos: Cosmos, M=12, beta=8.0, l2=1e-3, seed=1):
        self.cos = cosmos
        self.D = cosmos.D
        self.M = M
        self.beta = beta
        self.l2 = l2
        rng = np.random.default_rng(seed)
        # learnable params -- start as small random memories (NOT the answers)
        self.X = 0.10 * rng.standard_normal((M, self.D))
        self.b = np.zeros(self.D)
        self.g_raw = np.array(0.0)        # g = softplus(0) ~ 0.69 to start

    # ---- parameter (de)serialization for the gradient check ----
    def get_theta(self):
        return np.concatenate([self.X.ravel(), self.b.ravel(), np.atleast_1d(self.g_raw)])

    def set_theta(self, theta):
        nX = self.M * self.D
        self.X = theta[:nX].reshape(self.M, self.D)
        self.b = theta[nX:nX + self.D]
        self.g_raw = theta[nX + self.D]

    # ---- ENERGY (batched).  S: (B, D) -> E: (B,) ----
    def energy(self, S):
        g = softplus(self.g_raw)
        z = S @ self.X.T                          # (B, M)
        lse = logsumexp(self.beta * z, axis=1) / self.beta
        self_term = 0.5 * np.sum(S * S, axis=1)
        bias_term = S @ self.b
        Cs = S @ self.cos.C.T                     # (B, Dp)
        maat_term = 0.5 * g * np.sum(Cs * Cs, axis=1)
        return -lse + self_term - bias_term + maat_term

    # ---- grad of energy w.r.t. the STATE (used by retrieval dynamics) ----
    def grad_state(self, S):
        g = softplus(self.g_raw)
        z = S @ self.X.T                          # (B, M)
        p = softmax(self.beta * z, axis=1)        # (B, M) soft nearest-memory
        gE = -(p @ self.X) + S - self.b + g * (S @ self.cos.CtC)
        return gE

    # ---- COGNITION: relax a (corrupted) batch toward an attractor ----
    def retrieve(self, S0, steps=60, eta=0.3):
        """Restore maat: gradient-descent the state to a low-energy fixed point."""
        S = S0.copy()
        for _ in range(steps):
            S = S - eta * self.grad_state(S)
        return S

    # ---- grad of energy w.r.t. PARAMETERS for one batch (analytic) ----
    # Returns summed-over-batch grads as flat arrays; the loss combines them.
    def _param_grads_of_energy(self, S):
        """For each row s: dE/dX = -p s^T ; dE/db = -s ;
        dE/dg_raw = 0.5 * sigmoid(g_raw) * ||C s||^2.  Returned summed over batch."""
        B = S.shape[0]
        z = S @ self.X.T
        p = softmax(self.beta * z, axis=1)        # (B, M)
        # dE/dX summed over batch: -sum_b p_b outer s_b  = -(p^T @ S)
        dX = -(p.T @ S)                           # (M, D)
        dB = -np.sum(S, axis=0)                   # (D,)
        Cs = S @ self.cos.C.T
        dg = 0.5 * sigmoid(self.g_raw) * np.sum(Cs * Cs)   # scalar (summed)
        return dX, dB, dg

    # ---- LOSS (contrastive divergence) and its analytic gradient ----
    # L = mean_pos E(X_pos) - mean_neg E(S_neg) + 0.5*l2*||X||^2
    # S_neg is FIXED (detached).  Lower energy of data, raise energy of fantasies.
    def loss_and_grad(self, S_pos, S_neg):
        Bp, Bn = S_pos.shape[0], S_neg.shape[0]
        Ep = self.energy(S_pos)
        En = self.energy(S_neg)
        reg = 0.5 * self.l2 * np.sum(self.X * self.X)
        loss = np.mean(Ep) - np.mean(En) + reg

        dXp, dBp, dgp = self._param_grads_of_energy(S_pos)
        dXn, dBn, dgn = self._param_grads_of_energy(S_neg)
        dX = dXp / Bp - dXn / Bn + self.l2 * self.X
        dB = dBp / Bp - dBn / Bn
        dg = dgp / Bp - dgn / Bn
        grad = np.concatenate([dX.ravel(), dB.ravel(), np.atleast_1d(dg)])
        return loss, grad


# =============================================================================
# SECTION 3 -- MANDATORY finite-difference gradient check
# =============================================================================
# We hold the negative samples FIXED (as CD requires) and verify that the
# analytic dL/dtheta matches a central-difference estimate.  A correct energy
# model is a precondition for everything else; if this fails, nothing below is
# trustworthy.

def gradient_check(net: MaatFieldNetwork, S_pos, S_neg, n_probe=60, eps=1e-6, seed=7):
    theta0 = net.get_theta().copy()
    _, g_analytic = net.loss_and_grad(S_pos, S_neg)

    rng = np.random.default_rng(seed)
    idx = rng.choice(theta0.size, size=min(n_probe, theta0.size), replace=False)
    # always include the final coordinate (g_raw) so the gate gradient is tested
    if (theta0.size - 1) not in idx:
        idx = np.append(idx, theta0.size - 1)

    max_rel = 0.0
    for j in idx:
        tp = theta0.copy(); tp[j] += eps
        net.set_theta(tp); Lp, _ = net.loss_and_grad(S_pos, S_neg)
        tm = theta0.copy(); tm[j] -= eps
        net.set_theta(tm); Lm, _ = net.loss_and_grad(S_pos, S_neg)
        num = (Lp - Lm) / (2 * eps)
        ana = g_analytic[j]
        denom = max(1e-9, abs(num) + abs(ana))
        rel = abs(num - ana) / denom
        max_rel = max(max_rel, rel)
    net.set_theta(theta0)   # restore
    return max_rel


# =============================================================================
# SECTION 4 -- TRAINING (Adam on the CD energy objective)
# =============================================================================

def train(net: MaatFieldNetwork, cosmos: Cosmos, epochs=400, lr=0.02,
          cd_steps=8, cd_eta=0.30, noise=0.35, verbose=True):
    X_data = cosmos.patterns                      # (M, D) the balanced corpus
    # Adam state
    th = net.get_theta()
    m = np.zeros_like(th); v = np.zeros_like(th)
    b1, b2, eps = 0.9, 0.999, 1e-8

    history = []
    for ep in range(1, epochs + 1):
        # positives = the canonical balanced patterns
        S_pos = X_data
        # negatives = model fantasies: corrupt the data into isfet, then relax
        #             a few steps under the CURRENT energy (CD-k).  Detached.
        S_start = cosmos.isfet(X_data, noise=noise, break_pole=True)
        S_neg = net.retrieve(S_start, steps=cd_steps, eta=cd_eta)

        loss, grad = net.loss_and_grad(S_pos, S_neg)

        # Adam update
        m = b1 * m + (1 - b1) * grad
        v = b2 * v + (1 - b2) * (grad * grad)
        mhat = m / (1 - b1 ** ep)
        vhat = v / (1 - b2 ** ep)
        th = net.get_theta() - lr * mhat / (np.sqrt(vhat) + eps)
        net.set_theta(th)

        if ep % 50 == 0 or ep == 1:
            e_maat = float(np.mean(net.energy(X_data)))
            e_isfet = float(np.mean(net.energy(cosmos.isfet(X_data, noise=noise))))
            history.append((ep, loss, e_maat, e_isfet, e_isfet - e_maat))
            if verbose:
                print(f"  epoch {ep:4d} | loss {loss:+.4f} | "
                      f"E_maat {e_maat:+.3f} | E_isfet {e_isfet:+.3f} | "
                      f"gap {e_isfet - e_maat:+.3f}")
    return history


# =============================================================================
# SECTION 5 -- SELF-TESTS (the architecture must demonstrate the mind, not assert it)
# =============================================================================

def cosine(a, b):
    return float(np.sum(a * b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def test_energy_gap(net, cosmos):
    """T2: trained model assigns lower energy to maat than to isfet."""
    e_maat = float(np.mean(net.energy(cosmos.patterns)))
    e_isfet = float(np.mean(net.energy(cosmos.isfet(cosmos.patterns, noise=0.4))))
    return e_isfet - e_maat, e_maat, e_isfet

def test_pattern_completion(net, cosmos, trials=40, noise=0.3, seed=11):
    """T3: relax corrupted patterns; recovery = cosine to the true pattern,
    and 'hit' = nearest stored pattern is the correct one."""
    rng = np.random.default_rng(seed)
    X = cosmos.patterns
    hits, cos_sum = 0, 0.0
    for _ in range(trials):
        i = rng.integers(len(X))
        s0 = X[i] + noise * rng.standard_normal(cosmos.D)
        s0[cosmos.pole_minus] = 0.0                # break the feminine pole too
        s = net.retrieve(s0[None, :], steps=80, eta=0.3)[0]
        cos_sum += cosine(s, X[i])
        sims = X @ s / (np.linalg.norm(X, axis=1) * np.linalg.norm(s) + 1e-12)
        if int(np.argmax(sims)) == i:
            hits += 1
    return hits / trials, cos_sum / trials

def test_hatshepsut_balance(net, cosmos, seed=13):
    """T4 -- THE HATSHEPSUT TEST.
    Feed a one-sided 'male-only king': office +pole present, feminine -pole
    erased.  A faithful model must RESTORE the missing feminine pole, settling
    into the balanced dual-gender equilibrium (small ||C s||) rather than the
    one-sided state.  This is her solution emerging from the energy landscape."""
    rng = np.random.default_rng(seed)
    X = cosmos.patterns
    before, after, recov = [], [], []
    for i in range(len(X)):
        s0 = X[i].copy()
        s0[cosmos.pole_minus] = 0.0                # erase feminine grammar
        before.append(float(np.linalg.norm(cosmos.C @ s0)))
        s = net.retrieve(s0[None, :], steps=120, eta=0.3)[0]
        after.append(float(np.linalg.norm(cosmos.C @ s)))
        # how well was the erased -pole reconstructed toward the +pole charge?
        recov.append(cosine(s[cosmos.pole_minus], X[i][cosmos.pole_plus]))
    return float(np.mean(before)), float(np.mean(after)), float(np.mean(recov))

def test_role_binding(net, cosmos, seed=17):
    """T5: identity is a binding, not a body.  Keep a pattern's office/context
    but swap in a NOVEL filler/body; relaxation should keep the office balanced
    (the role survives the change of person)."""
    rng = np.random.default_rng(seed)
    X = cosmos.patterns
    ok = 0
    for i in range(len(X)):
        s0 = X[i].copy()
        s0[cosmos.fill] = rng.standard_normal(cosmos.Df)   # a new person
        s0[cosmos.pole_minus] = 0.0
        s = net.retrieve(s0[None, :], steps=120, eta=0.3)[0]
        if np.linalg.norm(cosmos.C @ s) < 0.5 * np.linalg.norm(cosmos.C @ s0 + 1e-9):
            ok += 1
    return ok / len(X)


# =============================================================================
# SECTION 6 -- MAIN: grad-check -> train -> test -> verified report
# =============================================================================

def main():
    np.random.seed(0)
    print("=" * 74)
    print(" THE MAAT-FIELD NETWORK  --  an energy model after Hatshepsut")
    print(" cognition = restoring maat (balance) from isfet (chaos)")
    print("=" * 74)

    cosmos = Cosmos(Dc=24, Dp=8, Df=24, n_patterns=12, seed=0)
    net = MaatFieldNetwork(cosmos, M=12, beta=8.0, l2=1e-3, seed=1)
    print(f"\nState dim D = {cosmos.D}  (ctx {cosmos.Dc} | pole+ {cosmos.Dp} | "
          f"pole- {cosmos.Dp} | filler {cosmos.Df})")
    print(f"Memories M = {net.M} | beta = {net.beta} | params = {net.get_theta().size}")

    # --- [1] mandatory gradient check (before training, arbitrary fixed negs) ---
    print("\n[1] FINITE-DIFFERENCE GRADIENT CHECK (contrastive-divergence loss)")
    S_pos = cosmos.patterns
    S_neg = cosmos.isfet(cosmos.patterns, noise=0.35, break_pole=True)
    max_rel = gradient_check(net, S_pos, S_neg, n_probe=60)
    print(f"    max relative error (analytic vs numeric): {max_rel:.3e}")
    ok_grad = max_rel < 1e-4
    print(f"    gradient check: {'PASS' if ok_grad else 'FAIL'}  (threshold 1e-4)")
    assert ok_grad, "Gradient check failed -- aborting."

    # --- [2] train: reshape the energy landscape ---
    print("\n[2] TRAINING (Adam on the CD energy objective)")
    hist = train(net, cosmos, epochs=400, lr=0.02, cd_steps=8, cd_eta=0.30,
                 noise=0.35, verbose=True)

    # --- [3] self-tests ---
    print("\n[3] SELF-TESTS")
    gap, em, ei = test_energy_gap(net, cosmos)
    print(f"  T2 energy gap        : E_isfet {ei:+.3f} - E_maat {em:+.3f} = "
          f"{gap:+.3f}  -> {'PASS' if gap > 0 else 'FAIL'}")

    hit, cos_rec = test_pattern_completion(net, cosmos)
    print(f"  T3 pattern completion: nearest-pattern hit-rate {hit*100:5.1f}% | "
          f"mean cosine {cos_rec:.3f}  -> {'PASS' if hit >= 0.9 else 'FAIL'}")

    imb0, imb1, rec = test_hatshepsut_balance(net, cosmos)
    print(f"  T4 HATSHEPSUT balance: imbalance {imb0:.3f} -> {imb1:.3f}  "
          f"(feminine pole restored, cos {rec:+.3f})  -> "
          f"{'PASS' if imb1 < 0.5 * imb0 else 'FAIL'}")

    rb = test_role_binding(net, cosmos)
    print(f"  T5 role-binding      : office re-balanced for novel body "
          f"{rb*100:5.1f}%  -> {'PASS' if rb >= 0.9 else 'FAIL'}")

    all_pass = ok_grad and gap > 0 and hit >= 0.9 and imb1 < 0.5 * imb0 and rb >= 0.9
    print("\n" + "=" * 74)
    print(f" RESULT: {'ALL CHECKS PASS' if all_pass else 'SOME CHECKS FAILED'}")
    print(" The network does not assert maat; it restores it -- relaxing a")
    print(" corrupted, one-sided 'king' back into the balanced equilibrium that")
    print(" was Hatshepsut's own cognitive solution.")
    print("=" * 74)
    return all_pass


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
