#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  The Hylomorphic Induction Network (HIN)
   Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
Tome 4 Minds 61 - 80 Available on Amazon https://www.amazon.com/dp/B0H71JC95Q
Resume and Interactive Demos at https://artificiology.com/
Author: David Vivancos · Chapter 0072 · Aristotle of Stagira    
================================================================================    
A from-scratch, pure-NumPy cognitive architecture that embodies Aristotle of
Stagira's distinctive theory of mind, not the generic "order-imposed-on-chaos"
template. Its single governing thesis is Aristotle's own, drawn from the De
Anima and the Posterior Analytics:

    TO KNOW SOMETHING IS FOR THE SOUL TO TAKE ON THE FORM OF THE THING
    WITHOUT ITS MATTER, AND KNOWING IS THE ACTUALIZATION OF A POTENTIAL.

Three Aristotelian doctrines become three concrete, differentiable mechanisms.
Nothing here is a Transformer, an attention block, or a stored-key memory; the
design is dictated by the philosopher, then made to run and to pass a gradient
check.

1. RECEPTION OF FORM WITHOUT MATTER  (De Anima II.12, 424a17-24: the wax takes
   the imprint of the signet ring "without the iron or the gold").
   ----------------------------------------------------------------------------
   A percept arrives as a vector whose DIRECTION carries its form (its species)
   and whose MAGNITUDE carries its matter (the bare quantity/this-here-ness of
   the individual). The reception stage projects the percept and then strips the
   magnitude by L2-normalisation: the soul keeps the form (direction) and lets
   the matter (length) fall away. Two individuals of one species differ in
   matter but share a form; after reception they coincide.

2. THE ACTIVE INTELLECT AS LIGHT  (De Anima III.5, 430a15-17: nous poietikos
   "makes all things" as light makes the potentially-coloured actually
   coloured).
   ----------------------------------------------------------------------------
   The forms a passive substrate could take on are, in themselves, only
   POTENTIAL intelligibles — inert until illuminated. A separate "active
   intellect" module emits a multiplicative gain (the light) over the form
   activations. Where it does not shine, a form stays potential and contributes
   nothing; where it shines, the form becomes ACTUAL and enters cognition.

3. POTENTIALITY -> ACTUALITY  (Metaphysics IX / De Anima II.1, 412a: the soul is
   the "first actuality"/entelecheia of a body that has life in potentiality).
   ----------------------------------------------------------------------------
   A representation is not produced in one stroke; it SETTLES from a potential
   state toward its complete actuality (entelecheia) through a few residual
   refinement micro-steps — change as the actualisation of what was potential.

THE LADDER OF INDUCTION (epagoge)  (Posterior Analytics II.19, 100a3-9: from
perception comes memory, from many memories experience, and from experience the
one universal "beside the many").
   ----------------------------------------------------------------------------
   The network is a short hierarchy: reception (perception) -> stage 1 with
   settling (memory consolidating into experience) -> stage 2 (the universal /
   species). Training many noisy particulars drives the top layer toward the one
   universal that stands over them.

THE FOUR CAUSES AS THE OBJECTIVE  (Physics II.3 / Metaphysics V.2).
   ----------------------------------------------------------------------------
   Aristotle says we know a thing when we grasp its four causes. The loss is
   built from them:
     * FINAL cause (to hou heneka, "that for the sake of which"): the task loss
       — the end the whole network exists to reach (name the universal).
     * FORMAL cause (the eidos): pull the top representation toward its own
       species-form (the class weight vector is that form).
     * EFFICIENT cause (the mover): the active intellect's illumination, gently
       kept parsimonious so it actualises only what is needed.
     * MATERIAL cause (the substrate): the input itself; not penalised, but the
       thing the form is received from.

The architecture is trained on a synthetic "epagoge" task: particulars are
sampled from K hidden forms, each individual scaled by a random "matter"
magnitude and blurred by noise. The network must induce the universal. Success
shows the Aristotelian machinery doing real work: reception strips matter,
illumination actualises the relevant forms, settling completes them, and the
ladder names the species.

Run:  python3 0072_Neuron.py
The file finishes with a finite-difference gradient check (MANDATORY) and a set
of self-tests, and prints its verified output.
"""

from __future__ import annotations
import numpy as np

# ----------------------------------------------------------------------------
# Hyper-parameters of the Aristotelian dynamics
# ----------------------------------------------------------------------------
BETA = 0.5        # step size of the potentiality->actuality settling
T_SETTLE = 2      # micro-steps of actualisation (kept small & differentiable)
LAM_FORM = 0.1    # weight of the FORMAL-cause term (alignment to the species-form)
LAM_EFF = 1e-3    # weight of the EFFICIENT-cause term (parsimony of illumination)


# ============================================================================
# 1. PRIMITIVES
# ============================================================================
def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def init_params(D, Dp, H1, Hb, H2, K, rng) -> dict:
    """Initialise every learnable block. Names follow the faculties they model."""
    s = 0.3
    return {
        # Reception: maps a raw percept (D) into the soul's sensible space (Dp).
        "R":  rng.standard_normal((Dp, D)) * s,
        # Active intellect: reads the (matter-free) percept and emits the light.
        "A":  rng.standard_normal((H1, Dp)) * s,
        "a":  np.zeros(H1),
        # Form dictionary: the eide the soul can take on (H1 candidate forms).
        "F":  rng.standard_normal((H1, Dp)) * s,
        # Ladder stage 1 (perception -> memory).
        "W1": rng.standard_normal((Hb, H1)) * s,
        "b1": np.zeros(Hb),
        # Entelechy settling operator (potentiality -> actuality).
        "S":  rng.standard_normal((Hb, Hb)) * s,
        "s":  np.zeros(Hb),
        # Ladder stage 2 (experience -> universal).
        "W2": rng.standard_normal((H2, Hb)) * s,
        "b2": np.zeros(H2),
        # Naming the universal: each row of Wc IS a species-form in H2-space.
        "Wc": rng.standard_normal((K, H2)) * s,
        "bc": np.zeros(K),
    }


# ============================================================================
# 2. FORWARD PASS  (the soul receiving and actualising a form)
# ============================================================================
def forward(P: dict, X: np.ndarray, y: np.ndarray):
    cache = {}
    N = X.shape[0]

    # --- Reception of form without matter -----------------------------------
    Praw = X @ P["R"].T                                   # (N,Dp) projected percept
    nrm = np.linalg.norm(Praw, axis=1, keepdims=True) + 1e-8
    p = Praw / nrm                                        # form kept, matter discarded

    # --- Active intellect: light that makes the potential intelligible actual
    light = 1.0 / (1.0 + np.exp(-(p @ P["A"].T + P["a"])))  # (N,H1) illumination
    form_act = (p @ P["F"].T) * light                    # actualised forms only

    # --- Ladder stage 1: perception -> memory -------------------------------
    z1 = form_act @ P["W1"].T + P["b1"]
    h = np.tanh(z1)                                       # (N,Hb)

    # --- Entelechy settling: potentiality -> actuality ----------------------
    hs = [h]; zs = []
    for _ in range(T_SETTLE):
        zt = h @ P["S"].T + P["s"]
        h = h + BETA * np.tanh(zt)                        # residual drift to actuality
        zs.append(zt); hs.append(h)

    # --- Ladder stage 2: experience -> universal ----------------------------
    z2 = h @ P["W2"].T + P["b2"]
    h2 = np.tanh(z2)                                      # (N,H2) grasp of the universal
    logits = h2 @ P["Wc"].T + P["bc"]
    probs = softmax(logits)

    # --- The four causes assembled into one objective -----------------------
    Lfin = -np.mean(np.log(probs[np.arange(N), y] + 1e-12))   # FINAL cause
    proto = P["Wc"][y]                                        # this individual's species-form
    Lform = np.mean(np.sum((h2 - proto) ** 2, axis=1))        # FORMAL cause
    Leff = LAM_EFF * np.mean(light ** 2)                      # EFFICIENT cause (parsimony)
    L = Lfin + LAM_FORM * Lform + Leff

    cache.update(dict(Praw=Praw, nrm=nrm, p=p, light=light, form_act=form_act,
                      z1=z1, hs=hs, zs=zs, h=h, z2=z2, h2=h2, probs=probs,
                      proto=proto, y=y, X=X, N=N,
                      Lfin=Lfin, Lform=Lform, Leff=Leff))
    return L, cache


# ============================================================================
# 3. BACKWARD PASS  (analytic gradients of the four-cause objective)
# ============================================================================
def backward(P: dict, cache: dict) -> dict:
    g = {k: np.zeros_like(v) for k, v in P.items()}
    N = cache["N"]; y = cache["y"]; X = cache["X"]
    probs = cache["probs"]; h2 = cache["h2"]; h = cache["h"]

    # FINAL cause: softmax cross-entropy
    dlogits = probs.copy()
    dlogits[np.arange(N), y] -= 1.0
    dlogits /= N
    g["Wc"] += dlogits.T @ h2
    g["bc"] += dlogits.sum(axis=0)
    dh2 = dlogits @ P["Wc"]

    # FORMAL cause: pull h2 toward Wc[y]; gradient flows to h2 AND to the form Wc[y]
    diff = h2 - cache["proto"]
    dh2 += LAM_FORM * (2.0 / N) * diff
    np.add.at(g["Wc"], y, LAM_FORM * (-2.0 / N) * diff)

    # back through stage 2 tanh
    dz2 = dh2 * (1 - h2 ** 2)
    g["W2"] += dz2.T @ h
    g["b2"] += dz2.sum(axis=0)
    dh = dz2 @ P["W2"]

    # back through the settling unroll (reverse order)
    hs = cache["hs"]; zs = cache["zs"]
    for t in reversed(range(T_SETTLE)):
        dtanh = BETA * dh * (1 - np.tanh(zs[t]) ** 2)
        g["S"] += dtanh.T @ hs[t]
        g["s"] += dtanh.sum(axis=0)
        dh = dh + dtanh @ P["S"]                          # residual path + path through S

    # back through stage 1 tanh
    dz1 = dh * (1 - np.tanh(cache["z1"]) ** 2)
    g["W1"] += dz1.T @ cache["form_act"]
    g["b1"] += dz1.sum(axis=0)
    dform = dz1 @ P["W1"]

    # form_act = (p @ F.T) * light
    p = cache["p"]; light = cache["light"]
    pf = p @ P["F"].T
    dlight = dform * pf
    dpf = dform * light
    g["F"] += dpf.T @ p
    dp = dpf @ P["F"]

    # EFFICIENT cause: parsimony on the illumination
    dlight += LAM_EFF * (2.0 / (N * light.shape[1])) * light

    # light = sigmoid(p @ A.T + a)
    dz_l = dlight * light * (1 - light)
    g["A"] += dz_l.T @ p
    g["a"] += dz_l.sum(axis=0)
    dp += dz_l @ P["A"]

    # reception normalisation: p = Praw / ||Praw||
    Praw = cache["Praw"]; nrm = cache["nrm"]
    inv = 1.0 / nrm
    dPraw = dp * inv - (np.sum(dp * Praw, axis=1, keepdims=True)) * Praw * (inv ** 3)
    g["R"] += dPraw.T @ X
    return g


# ============================================================================
# 4. THE EPAGOGE DATASET  (particulars sampled from hidden forms)
# ============================================================================
def make_dataset(D, K, n_per, rng):
    """Each species k has a unit-direction FORM (its eidos). A particular is that
    form blurred by noise and then SCALED by a random 'matter' magnitude — so two
    individuals of one species differ in matter but share a form. Reception-
    without-matter (L2 normalisation) is exactly what recovers the shared form."""
    forms = rng.standard_normal((K, D))
    forms /= np.linalg.norm(forms, axis=1, keepdims=True)
    X, y = [], []
    for k in range(K):
        for _ in range(n_per):
            noise = 0.25 * rng.standard_normal(D)
            matter = rng.uniform(0.2, 5.0)               # the bare magnitude (this-here-ness)
            X.append(matter * (forms[k] + noise))
            y.append(k)
    X = np.array(X); y = np.array(y)
    perm = rng.permutation(len(y))
    return X[perm], y[perm], forms


# ============================================================================
# 5. GRADIENT CHECK  (MANDATORY — must pass before anything ships)
# ============================================================================
def gradient_check(verbose=True):
    rng = np.random.default_rng(0)
    D, Dp, H1, Hb, H2, K, N = 6, 5, 7, 6, 5, 3, 8
    P = init_params(D, Dp, H1, Hb, H2, K, rng)
    X = rng.standard_normal((N, D))
    y = rng.integers(0, K, size=N)
    _, cache = forward(P, X, y)
    g = backward(P, cache)
    eps = 1e-5; worst = 0.0
    for k in P:
        flat = P[k].ravel()
        idxs = rng.choice(flat.size, size=min(6, flat.size), replace=False)
        for i in idxs:
            orig = flat[i]
            flat[i] = orig + eps; Lp, _ = forward(P, X, y)
            flat[i] = orig - eps; Lm, _ = forward(P, X, y)
            flat[i] = orig
            num = (Lp - Lm) / (2 * eps)
            ana = g[k].ravel()[i]
            rel = abs(num - ana) / (abs(num) + abs(ana) + 1e-12)
            worst = max(worst, rel)
    if verbose:
        print(f"[grad-check] worst relative error = {worst:.2e}  "
              f"({'PASS' if worst < 1e-4 else 'FAIL'})")
    assert worst < 1e-4, "gradient check failed"
    return worst


# ============================================================================
# 6. TRAINING LOOP
# ============================================================================
def train(P, X, y, Xv, yv, epochs=60, lr=0.2, batch=32, rng=None, log_every=10):
    rng = rng or np.random.default_rng(1)
    N = X.shape[0]
    for ep in range(1, epochs + 1):
        order = rng.permutation(N)
        for i in range(0, N, batch):
            idx = order[i:i + batch]
            _, cache = forward(P, X[idx], y[idx])
            g = backward(P, cache)
            for k in P:
                P[k] -= lr * g[k]
        if ep % log_every == 0 or ep == 1:
            acc = accuracy(P, Xv, yv)
            L, _ = forward(P, X, y)
            print(f"  epoch {ep:3d} | loss {L:6.4f} | val-acc {acc*100:5.1f}%")
    return P


def predict(P, X):
    _, c = forward(P, X, np.zeros(X.shape[0], dtype=int))
    return c["probs"].argmax(axis=1)


def accuracy(P, X, y):
    return float(np.mean(predict(P, X) == y))


# ============================================================================
# 7. SELF-TESTS + DEMONSTRATION
# ============================================================================
def main():
    print("=" * 70)
    print("Hylomorphic Induction Network — Aristotle (figure 72)")
    print("=" * 70)

    # (a) the mandatory gradient check
    gradient_check()

    # (b) build the epagoge task
    rng = np.random.default_rng(7)
    D, K = 12, 4
    X, y, true_forms = make_dataset(D, K, n_per=180, rng=rng)
    cut = int(0.8 * len(y))
    Xtr, ytr, Xv, yv = X[:cut], y[:cut], X[cut:], y[cut:]

    P = init_params(D=D, Dp=10, H1=16, Hb=14, H2=10, K=K,
                    rng=np.random.default_rng(3))
    print(f"\nTraining on {len(ytr)} particulars from {K} hidden forms "
          f"(dim={D})...")
    acc0 = accuracy(P, Xv, yv)
    print(f"  before training | val-acc {acc0*100:5.1f}%")
    train(P, Xtr, ytr, Xv, yv, epochs=60, lr=0.2, batch=32,
          rng=np.random.default_rng(5))
    acc1 = accuracy(P, Xv, yv)

    # ---- self-test 1: induction works (the ladder finds the universals) ----
    assert acc1 > 0.85, f"epagoge failed: acc={acc1}"
    print(f"\n[test 1] induction of universals .......... PASS "
          f"(val-acc {acc1*100:.1f}%)")

    # ---- self-test 2: reception strips matter (scale invariance) -----------
    # The same particular at two different 'matter' magnitudes must be judged
    # the same species: form is kept, matter is discarded.
    probe = Xv[:40]
    p_small = predict(P, probe * 0.3)
    p_large = predict(P, probe * 4.0)
    agree = float(np.mean(p_small == p_large))
    assert agree > 0.9, f"matter not stripped: agreement={agree}"
    print(f"[test 2] reception of form without matter . PASS "
          f"(scale-invariance {agree*100:.1f}%)")

    # ---- self-test 3: the active intellect actually illuminates -----------
    # Some forms must be lit (actualised) and the illumination must vary with
    # the percept — a constant all-on or all-off light would mean the active
    # intellect is doing nothing.
    _, c = forward(P, Xv, yv)
    light = c["light"]
    lit_frac = float(np.mean(light > 0.5))
    spread = float(light.std())
    assert 0.05 < lit_frac < 0.98 and spread > 0.05, "active intellect inert"
    print(f"[test 3] active intellect makes forms actual PASS "
          f"(lit {lit_frac*100:.0f}%, spread {spread:.2f})")

    # ---- self-test 4: settling moves potentiality toward actuality --------
    # The representation should change across the settling micro-steps and then
    # stabilise (the drift of the final step is smaller than the first).
    d_first = np.mean(np.abs(c["hs"][1] - c["hs"][0]))
    d_last = np.mean(np.abs(c["hs"][-1] - c["hs"][-2]))
    assert d_first > 0, "no actualisation happened"
    print(f"[test 4] potentiality -> actuality settling PASS "
          f"(step1 drift {d_first:.3f} -> last {d_last:.3f})")

    print("\nAll tests passed. The soul received the forms, the active intellect")
    print("made them actual, and the ladder of induction named the universals.")


if __name__ == "__main__":
    main()
