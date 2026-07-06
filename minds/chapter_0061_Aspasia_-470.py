#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================================
 61 - ASPASIA OF MILETUS  (c. 470 - c. 400 BCE)
 Architecture:  THE INDUCTIVE MIRROR NETWORK  (IMN)
 Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
Tome 4 Minds 61 - 80 Available on Amazon https://www.amazon.com/dp/B0H71JC95Q
Resume and Interactive Demos at https://artificiology.com/
Author: David Vivancos · Chapter 0061 · Aspasia of Miletus
============================================================================

WHY THIS ARCHITECTURE, AND WHY IT IS *HERS*
-------------------------------------------
Aspasia left no words of her own. The single piece of actual reasoning that
survives under her name is preserved by Cicero (De Inventione 1.31.51-53),
who quotes a lost dialogue of Aeschines of Sphettos. In it Aspasia does NOT
lecture. She asks a ladder of *comparative* questions, each one easy to
answer, each of the same shape:

    "If your neighbour had a BETTER ornament than yours, which would you want?"
    "Hers."
    "A better dress?"            "Hers."
    "...a better husband?"       (the wife falls silent)

She then runs the identical ladder on the husband (horse, farm, ... wife),
and he too falls silent. She never states the conclusion. The interlocutors
*generate it themselves*: each desires the best partner, therefore each must
*become* the best to deserve the best. Cicero files this under INDUCTIO
(Greek: epagoge) - leading a mind, by assent to undisputed cases, to a
doubtful proposition that resembles them. Xenophon (Memorabilia 2.6.36) adds
the matchmaker's rule: report the good qualities *truthfully*, never flatter.

That is a precise cognitive signature, and it is NOT a transformer:

  (1) LEARN FROM COMPARISONS, NOT LABELS.  Every datum is "B is better than
      A," never "A scores 7.3." The model is trained on pairwise judgments
      only (a Bradley-Terry / pairwise-logistic objective).
  (2) ONE COMPARATOR, TRANSFERRED BY ANALOGY.  The same merit function is
      learned on "safe" domains (ornament, dress, horse, farm) and then
      carried, unchanged, onto a withheld "loaded" domain (partner). The
      inductive leap = ranking accuracy on a domain it never trained on.
  (3) THE CONCLUSION IS INDUCED, NOT ASSERTED.  No partner-merit label is
      ever supplied. Two agents, each *desiring the best*, ascend the SAME
      learned merit gradient toward the ideal the other wants. The "silence
      then resolve" is a fixed point of this coupled dynamic (eros -> arete).
  (4) THE MATCHMAKER MUST BE HONEST.  A flattering comparator (one that
      inflates merit) produces a false equilibrium: agents *believe* they
      have improved while the true merit gap stays open. Only the truthful
      comparator closes the real gap. We demonstrate this directly.

So the file has three movements:
  A. InductiveComparator  - a small from-scratch MLP scorer trained ONLY on
                            comparisons, with HAND-WRITTEN backprop and a
                            mandatory finite-difference gradient check.
  B. Analogy transfer     - train on safe domains, measure ranking accuracy
                            on the withheld "partner" domain.
  C. Eros->Arete mirror   - the coupled self-improvement dynamic, run with an
                            honest vs. a flattering matchmaker.

Everything is pure NumPy, from scratch, and self-tested. Run it directly.
============================================================================
"""

import numpy as np

np.random.seed(61)  # 61 = Aspasia's index in the corpus

# ===========================================================================
# 0.  SMALL NUMERIC UTILITIES
# ===========================================================================

def sigmoid(z):
    # Numerically stable logistic function.
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def log_sigmoid(z):
    # Stable log(sigmoid(z)) = -softplus(-z).
    return -np.logaddexp(0.0, -z)


def xavier(shape, rng):
    fan_in = shape[0] if len(shape) == 1 else shape[1]
    fan_out = shape[0]
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=shape)


# ===========================================================================
# 1.  THE INDUCTIVE COMPARATOR
#     A merit scorer s(x) = u . tanh(W x + b), trained ONLY on the question
#     "is B better than A?" via the Bradley-Terry pairwise-logistic loss:
#         P(B > A) = sigmoid( s(B) - s(A) )
#     This is the computational atom of Aspasia's ladder of comparisons.
# ===========================================================================

class InductiveComparator:
    def __init__(self, in_dim, hidden, rng):
        self.in_dim = in_dim
        self.hidden = hidden
        self.W = xavier((hidden, in_dim), rng)   # hidden x in_dim
        self.b = np.zeros(hidden)
        self.u = xavier((hidden,), rng)          # hidden
        self._cache = None

    # ---- parameter (un)packing, used by the gradient check & optimiser ----
    def get_params(self):
        return np.concatenate([self.W.ravel(), self.b.ravel(), self.u.ravel()])

    def set_params(self, vec):
        h, d = self.hidden, self.in_dim
        i = 0
        self.W = vec[i:i + h * d].reshape(h, d); i += h * d
        self.b = vec[i:i + h]; i += h
        self.u = vec[i:i + h]

    # ---- forward: merit score for a batch of items X (N x in_dim) ----------
    def score(self, X):
        Z = X @ self.W.T + self.b        # N x hidden    (pre-activation)
        Hd = np.tanh(Z)                  # N x hidden    (hidden activation)
        s = Hd @ self.u                  # N             (scalar merit)
        return s, (X, Z, Hd)

    # ---- gradient of merit wrt the INPUT x  (used by the eros->arete loop) -
    # d s / d x = W^T ( u * (1 - tanh(Wx+b)^2) )
    def score_grad_input(self, X):
        Z = X @ self.W.T + self.b
        dtanh = 1.0 - np.tanh(Z) ** 2            # N x hidden
        g = (dtanh * self.u) @ self.W            # N x in_dim
        return g

    # ---- pairwise loss + HAND-WRITTEN backprop ----------------------------
    # Pairs: Xa, Xb  (N x in_dim).  Label y=1 means "B is better than A".
    def loss_and_grads(self, Xa, Xb, y, l2=1e-4):
        sa, ca = self.score(Xa)
        sb, cb = self.score(Xb)
        diff = sb - sa                            # logit of P(B > A)
        # Stable BCE on the difference:
        # L = mean( softplus(diff) - y*diff ) + l2 penalty
        loss = np.mean(np.logaddexp(0.0, diff) - y * diff)
        loss += 0.5 * l2 * (np.sum(self.W ** 2) + np.sum(self.u ** 2))

        N = Xa.shape[0]
        p = sigmoid(diff)                         # N      predicted P(B>A)
        dL_ddiff = (p - y) / N                    # N      dLoss/d(diff)

        # ds/d(params) for one side, then combine (B contributes +, A -)
        def side_grads(cache, coeff):
            X, Z, Hd = cache
            # s = Hd . u  ;  Hd = tanh(Z) ; Z = X W^T + b
            dL_ds = coeff                          # N (already includes /N)
            dL_du = Hd.T @ dL_ds                    # hidden
            dHd = np.outer(dL_ds, self.u)          # N x hidden
            dZ = dHd * (1.0 - Hd ** 2)             # N x hidden  (tanh')
            dL_dW = dZ.T @ X                        # hidden x in_dim
            dL_db = dZ.sum(axis=0)                  # hidden
            return dL_dW, dL_db, dL_du

        dWb, dbb, dub = side_grads(cb, dL_ddiff)   # B side: +diff
        dWa, dba, dua = side_grads(ca, -dL_ddiff)  # A side: -diff
        dW = dWb + dWa + l2 * self.W
        db = dbb + dba
        du = dub + dua + l2 * self.u
        grad = np.concatenate([dW.ravel(), db.ravel(), du.ravel()])
        return loss, grad


# ===========================================================================
# 2.  SYNTHETIC WORLD OF COMPARISONS
#     A hidden "true merit" lives on a shared latent axis. Each DOMAIN
#     (ornament, dress, horse, farm, partner) rotates the latent into its own
#     observable features, so a comparator that merely memorised one domain
#     cannot transfer. The honest task is to learn the merit *form* itself.
# ===========================================================================

DOMAINS = ["ornament", "dress", "horse", "farm", "partner"]
LOADED_DOMAIN = "partner"          # withheld at training time (the "silence")

def make_world(latent_dim=4, rng=None):
    rng = rng or np.random.default_rng(0)
    # The single latent direction along which "better" is defined.
    v_true = rng.normal(size=latent_dim)
    v_true /= np.linalg.norm(v_true)
    # CRUCIAL design choice (this is the whole Aspasian point):
    # merit always shows up along ONE SHARED observable axis, identical in
    # every domain - the *form* of "better" is constant. Only the irrelevant
    # residual (the subject matter: jewelry vs horse vs spouse) gets a
    # domain-specific scramble that the comparator must learn to see through.
    shared_dir = rng.normal(size=latent_dim)
    shared_dir /= np.linalg.norm(shared_dir)
    proj = {d: rng.normal(size=(latent_dim, latent_dim)) for d in DOMAINS}
    return {"v_true": v_true, "shared_dir": shared_dir,
            "proj": proj, "latent_dim": latent_dim}

def true_merit(world, latent):
    # The ground-truth merit the interlocutors implicitly agree on.
    return latent @ world["v_true"]

def observe(world, latent, domain):
    # Observable = (merit carried on the SHARED axis)
    #            + (domain-specific dressing of the merit-orthogonal residual)
    #            + a one-hot domain tag.
    m = true_merit(world, latent)                       # scalar merit
    residual = latent - m * world["v_true"]             # merit-orthogonal part
    feats = m * world["shared_dir"] + 0.4 * (residual @ world["proj"][domain])
    onehot = np.zeros(len(DOMAINS))
    onehot[DOMAINS.index(domain)] = 1.0
    return np.concatenate([feats, onehot])

def sample_pairs(world, domains, n_pairs, rng, noise=0.0):
    L = world["latent_dim"]
    Xa, Xb, y = [], [], []
    for _ in range(n_pairs):
        d = rng.choice(domains)
        la = rng.normal(size=L)
        lb = rng.normal(size=L)
        ma, mb = true_merit(world, la), true_merit(world, lb)
        # noisy comparative judgment (Bradley-Terry style)
        p = sigmoid(np.array([(mb - ma) / max(noise, 1e-9)]))[0] if noise > 0 \
            else (1.0 if mb > ma else 0.0)
        label = 1.0 if rng.random() < p else 0.0
        Xa.append(observe(world, la, d))
        Xb.append(observe(world, lb, d))
        y.append(label)
    return np.array(Xa), np.array(Xb), np.array(y)


# ===========================================================================
# 3.  TRAINING  (full-batch gradient descent on the pairwise loss)
# ===========================================================================

def train(model, Xa, Xb, y, lr=0.5, epochs=400, l2=1e-4, verbose=False):
    history = []
    for ep in range(epochs):
        loss, grad = model.loss_and_grads(Xa, Xb, y, l2=l2)
        params = model.get_params() - lr * grad
        model.set_params(params)
        history.append(loss)
        if verbose and ep % 100 == 0:
            print(f"    epoch {ep:4d}   loss {loss:.4f}")
    return history

def ranking_accuracy(model, world, domain, rng, n=400):
    # Fraction of held-out pairs whose order the comparator gets right.
    Xa, Xb, y = sample_pairs(world, [domain], n, rng, noise=0.0)
    sa, _ = model.score(Xa)
    sb, _ = model.score(Xb)
    pred = (sb > sa).astype(float)
    return float(np.mean(pred == y))


# ===========================================================================
# 4.  THE EROS -> ARETE MIRROR
#     Two agents. Each *desires the best* partner, i.e. wants a partner whose
#     comparator-merit is maximal. Aspasia's induced conclusion: to be chosen
#     by such a partner, one must *become* the best. Each agent therefore
#     ascends the SAME learned merit gradient. No merit label is ever given;
#     the resolve emerges from the coupling. A FLATTERING matchmaker inflates
#     perceived merit and produces a false (stagnant) equilibrium.
# ===========================================================================

def eros_arete_dynamics(model, world, steps=60, lr=0.25, flatter=0.0, rng=None):
    """
    flatter = 0.0  -> honest matchmaker (truthful report, Xen. Mem. 2.6.36)
    flatter > 0.0  -> the report is inflated; agents *feel* improved but the
                      gradient they climb is corrupted, so true merit stalls.
    Returns the trajectory of TRUE merit for both agents.
    """
    rng = rng or np.random.default_rng(7)
    L = world["latent_dim"]
    # Two agents begin as ordinary, below-ideal partners (latent self-vectors).
    self_a = rng.normal(size=L) * 0.3
    self_b = rng.normal(size=L) * 0.3
    traj_a, traj_b = [], []

    # Jacobian d(observable feats)/d(latent) for the loaded 'partner' domain,
    # used to pull the comparator's input-gradient back into latent space.
    v = world["v_true"]
    sh = world["shared_dir"]
    P = world["proj"][LOADED_DOMAIN]
    J = np.outer(sh, v) + 0.4 * (P.T @ (np.eye(L) - np.outer(v, v)))

    for _ in range(steps):
        traj_a.append(true_merit(world, self_a))
        traj_b.append(true_merit(world, self_b))
        # Each agent presents itself in the "partner" domain and asks the
        # comparator which way is "better"; it then climbs that direction.
        xa = observe(world, self_a, LOADED_DOMAIN)[None, :]
        xb = observe(world, self_b, LOADED_DOMAIN)[None, :]
        ga = J.T @ model.score_grad_input(xa)[0, :L]   # gradient into latent
        gb = J.T @ model.score_grad_input(xb)[0, :L]
        # A flattering matchmaker adds a self-congratulatory direction that
        # does NOT align with true merit, corrupting the ascent.
        if flatter > 0.0:
            bogus = rng.normal(size=L)
            ga = (1 - flatter) * ga + flatter * np.linalg.norm(ga) * bogus
            gb = (1 - flatter) * gb + flatter * np.linalg.norm(gb) * bogus
        n_a = np.linalg.norm(ga) + 1e-9
        n_b = np.linalg.norm(gb) + 1e-9
        self_a = self_a + lr * ga / n_a
        self_b = self_b + lr * gb / n_b
    return np.array(traj_a), np.array(traj_b)


# ===========================================================================
# 5.  MANDATORY FINITE-DIFFERENCE GRADIENT CHECK
# ===========================================================================

def gradient_check(verbose=True):
    rng = np.random.default_rng(123)
    in_dim, hidden, n = 7, 6, 12
    model = InductiveComparator(in_dim, hidden, rng)
    Xa = rng.normal(size=(n, in_dim))
    Xb = rng.normal(size=(n, in_dim))
    y = (rng.random(n) < 0.5).astype(float)

    _, analytic = model.loss_and_grads(Xa, Xb, y, l2=1e-3)
    theta = model.get_params().copy()
    numeric = np.zeros_like(theta)
    eps = 1e-6
    for i in range(theta.size):
        t = theta.copy(); t[i] += eps
        model.set_params(t); lp, _ = model.loss_and_grads(Xa, Xb, y, l2=1e-3)
        t = theta.copy(); t[i] -= eps
        model.set_params(t); lm, _ = model.loss_and_grads(Xa, Xb, y, l2=1e-3)
        numeric[i] = (lp - lm) / (2 * eps)
    model.set_params(theta)

    rel = np.linalg.norm(analytic - numeric) / (
        np.linalg.norm(analytic) + np.linalg.norm(numeric) + 1e-12)
    if verbose:
        print(f"  finite-difference gradient check  relative error = {rel:.2e}")
    return rel


# ===========================================================================
# 6.  MAIN  -  run the three movements and self-test
# ===========================================================================

def main():
    print("=" * 74)
    print(" THE INDUCTIVE MIRROR NETWORK  -  Aspasia of Miletus  (#61)")
    print("=" * 74)

    # --- (0) gradient check -------------------------------------------------
    print("\n[0] Gradient check (analytic backprop vs. finite differences)")
    rel = gradient_check()
    assert rel < 1e-5, f"gradient check FAILED (rel err {rel:.2e})"
    print("    PASS: hand-written backprop matches numerical gradient.")

    # --- (1) learn the comparator from comparisons only ---------------------
    print("\n[1] Learning merit from COMPARISONS ONLY (Bradley-Terry loss)")
    rng = np.random.default_rng(2024)
    world = make_world(latent_dim=4, rng=rng)
    safe_domains = [d for d in DOMAINS if d != LOADED_DOMAIN]
    in_dim = world["latent_dim"] + len(DOMAINS)
    model = InductiveComparator(in_dim, hidden=16, rng=rng)
    Xa, Xb, y = sample_pairs(world, safe_domains, 2600, rng, noise=0.0)
    hist = train(model, Xa, Xb, y, lr=0.5, epochs=500, l2=1e-4, verbose=True)
    print(f"    final training loss = {hist[-1]:.4f}")
    train_acc = np.mean(
        [ranking_accuracy(model, world, d, rng) for d in safe_domains])
    print(f"    ranking accuracy on TRAINED domains = {train_acc:.3f}")

    # --- (2) the inductive leap: transfer to the withheld 'partner' domain --
    print("\n[2] The inductive leap: transfer to the WITHHELD 'partner' domain")
    print("    (the comparator never saw a single 'partner' comparison)")
    transfer_acc = ranking_accuracy(model, world, LOADED_DOMAIN, rng, n=800)
    print(f"    ranking accuracy on '{LOADED_DOMAIN}' (unseen) = {transfer_acc:.3f}")
    assert transfer_acc > 0.80, "analogy transfer too weak"
    print("    PASS: assent built on easy comparisons carries to the loaded one.")

    # --- (3) eros -> arete: honest vs. flattering matchmaker ---------------
    print("\n[3] Eros -> Arete mirror: do the agents actually improve?")
    a_h, b_h = eros_arete_dynamics(model, world, steps=60, lr=0.25,
                                   flatter=0.0, rng=np.random.default_rng(7))
    a_f, b_f = eros_arete_dynamics(model, world, steps=60, lr=0.25,
                                   flatter=0.6, rng=np.random.default_rng(7))
    gain_honest = (a_h[-1] - a_h[0] + b_h[-1] - b_h[0]) / 2.0
    gain_flatter = (a_f[-1] - a_f[0] + b_f[-1] - b_f[0]) / 2.0
    print(f"    honest matchmaker : mean TRUE-merit gain = {gain_honest:+.3f}")
    print(f"    flattering one    : mean TRUE-merit gain = {gain_flatter:+.3f}")
    assert gain_honest > 0.5, "honest dynamic should raise true merit"
    assert gain_honest > gain_flatter + 0.3, "flattery should under-perform truth"
    print("    PASS: only the TRUTHFUL comparator closes the real merit gap")
    print("          (Xenophon, Memorabilia 2.6.36 - 'never praise falsely').")

    # --- summary ------------------------------------------------------------
    print("\n" + "=" * 74)
    print(" SELF-TESTS PASSED.  The network learns the FORM of 'better' from")
    print(" comparisons, transfers it by analogy to a question it was never")
    print(" trained on, and lets two agents INDUCE - never are told - that to")
    print(" be desired they must become excellent.  That is the inductio.")
    print("=" * 74)


if __name__ == "__main__":
    main()
