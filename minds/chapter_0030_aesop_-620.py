#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
chapter_0030_aesop_-620.py  —  THE AESOPICA ENGINE
====================================================================
Chapter 30: Aesop  (c. 620 – c. 564 BCE, Phrygia / Samos / Delphi)
# Part Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 2 Minds 21 - 40 Available on Amazon https://www.amazon.com/dp/B0H6QCQ9M7
# Resume and Interactive Demos at https://artificiology.com/
Author: David Vivancos · Chapter 0030 · Aesop
A from-scratch, pure-NumPy neural architecture that embodies Aesop's
*distinctive* cognitive signature — not the generic "stories compress
morals" reading, but the mechanism underneath it:

    Aesop reasons by CASTING.

He does not state a moral proposition. He takes the entities of a real,
power-laden situation, casts each one as a fixed character-TYPE (an
animal whose disposition is known), and then runs the deterministic
"physics of character" forward to read off the outcome. The fable's
moral is the *recovered* pair (casting, outcome), never a stored rule.
Three properties of the historical Aesop fall straight out of this:

  * AINOS (coded, aimed speech).  The fable is the weapon of the
    powerless: a slave can say to a king, through a fox and a lion,
    what he could never say in his own voice — and can always retreat
    to "I was only telling a story about animals." Casting is the
    deniable channel. (Kurke 2011; van Dijk 1997.)

  * TYPED FORWARD SIMULATION.  Character is destiny. Put fixed
    dispositions in a configuration, let power modulate them, and the
    winner falls out. Force beats innocence (the wolf eats the lamb —
    injustice by power) *unless* guile is present to subvert it (the
    fox tricks). The "moral physics" is a learnable interaction law.

  * THE DELPHI FRAME-UP.  Aesop was killed at Delphi on a planted
    charge: a sacred cup slipped into his bag recast him from envoy to
    thief, and he was thrown from the cliff. A single planted feature
    flipped his casting and sealed the verdict. This is, exactly, an
    adversarial input against a casting system — and the engine below
    reproduces it as a runnable demonstration.

WHAT THIS FILE ACTUALLY IS (not a demo of fake "encoders"):
  - A real model with learnable parameters and hand-derived backprop.
  - A MANDATORY finite-difference gradient check on every parameter
    tensor *and* on the inputs (so the adversarial step is exact).
  - A real Adam training loop on a task generated from a known
    "moral law," with held-out generalization measured.
  - Self-tests / demonstrations: casting interpretability, abductive
    "reading the fable backward," and the Delphi adversarial frame-up.

Run:  python3 chapter_0030_aesop_-620.py
The verified console output is pasted into the chapter (30_aesop.md).
====================================================================
"""

import numpy as np

# Float64 everywhere: the finite-difference gradient check needs the
# precision, and nothing here is large enough to need float32.
DTYPE = np.float64


# ====================================================================
# SECTION 1 — THE WORLD AESOP READS
#   A tiny but non-trivial "moral physics" used to GENERATE data.
#   The model never sees these labels or prototypes; it must recover
#   castings and learn the outcome law from surface features alone.
# ====================================================================

# Five canonical character-types. Each is given three latent traits
# that evoke the animal, on a 0..1 scale:
#   force     — raw coercive strength
#   guile     — cunning; the capacity to subvert a stronger party
#   integrity — alignment with justice (used only to score valence)
#
#   wolf     : strength without scruple        (the tyrant)
#   fox      : weak in force, supreme in guile  (the trickster)
#   lion     : strong and somewhat noble        (legitimate power)
#   lamb     : innocent and powerless           (the victim)
#   tortoise : slow, persistent, upright        (patience as guile)
ARCHETYPES = ["wolf", "fox", "lion", "lamb", "tortoise"]
TRAITS = np.array([
    # force  guile  integrity
    [0.95,  0.20,  0.10],   # wolf
    [0.20,  0.95,  0.25],   # fox
    [0.85,  0.45,  0.60],   # lion
    [0.15,  0.15,  0.95],   # lamb
    [0.25,  0.70,  0.90],   # tortoise
], dtype=DTYPE)
K_TRUE = len(ARCHETYPES)
FORCE, GUILE, INTEG = 0, 1, 2


def moral_physics(k1, k2, power1):
    """The ground-truth 'physics of character' that the engine must learn.

    Given two character-types and whether agent 1 holds *institutional*
    power (e.g. sits in the Delphic court), decide who prevails and how
    just the outcome is.

    Returns (winner, valence):
        winner  in {0, 1}   0 => agent1 prevails, 1 => agent2 prevails
        valence in {-1,+1}  +1 if the more-upright party prevails
                            (justice); -1 if the less-upright prevails
                            (injustice — power or guile triumphing).

    The law is deliberately Aesopic and non-linear:
      * effective power = force + 0.6*guile + (institutional bonus)
      * GUILE SUBVERTS FORCE: if your opponent out-forces you, you add
        your full guile again (the fox slips past the lion).
      * the winner is whoever ends with greater effective power.
    """
    f1, g1, i1 = TRAITS[k1]
    f2, g2, i2 = TRAITS[k2]

    inst = 0.7  # weight of holding institutional power (kept modest so that
    #             force, guile and subversion genuinely contest the verdict —
    #             otherwise the court always wins and there is nothing to learn)
    e1 = f1 + 0.6 * g1 + (inst if power1 == 1 else 0.0)
    e2 = f2 + 0.6 * g2 + (inst if power1 == 0 else 0.0)

    # Guile lets the weaker-in-force party subvert the stronger.
    if f2 > f1:
        e1 += g1
    if f1 > f2:
        e2 += g2

    winner = 0 if e1 >= e2 else 1
    winner_integrity = i1 if winner == 0 else i2
    loser_integrity = i2 if winner == 0 else i1
    valence = 1.0 if winner_integrity >= loser_integrity else -1.0
    return winner, valence


def make_dataset(n, seed, feat_dim=12, noise=0.35):
    """Render abstract 'situations' into surface features.

    Each agent is secretly one of the five archetypes. We DON'T hand the
    model that label. Instead we render the archetype's 3 latent traits
    into a `feat_dim`-vector through a fixed random 'rendering' matrix
    plus noise plus pure-distractor dimensions — the messy surface of a
    real situation. The model must learn to *cast* (recover the type)
    purely from the end task. This is the crux: casting is unsupervised,
    driven only by the outcome signal.

    Returns dict with X1, X2 (n, feat_dim); power1 (n,1); y_win (n,);
    y_val (n,); and the hidden true types k1, k2 (for diagnostics only).
    """
    rng = np.random.default_rng(seed)
    # Fixed rendering: traits(3) -> features. Shared across the corpus so
    # that 'fox-ness' looks consistent from situation to situation.
    render_rng = np.random.default_rng(20250001)  # FIXED, not `seed`
    R = render_rng.standard_normal((3, feat_dim)).astype(DTYPE)

    k1 = rng.integers(0, K_TRUE, size=n)
    k2 = rng.integers(0, K_TRUE, size=n)
    power1 = rng.integers(0, 2, size=n)

    X1 = TRAITS[k1] @ R + noise * rng.standard_normal((n, feat_dim))
    X2 = TRAITS[k2] @ R + noise * rng.standard_normal((n, feat_dim))

    # Two extra dimensions of pure distraction (irrelevant surface
    # detail — the colour of a robe, the weather). The model must learn
    # to ignore them. The planted "Delphic cup" in the adversarial demo
    # will live in exactly this kind of dimension.
    X1 = np.concatenate([X1, 0.5 * rng.standard_normal((n, 2))], axis=1)
    X2 = np.concatenate([X2, 0.5 * rng.standard_normal((n, 2))], axis=1)

    y_win = np.zeros(n, dtype=np.int64)
    y_val = np.zeros(n, dtype=DTYPE)
    for i in range(n):
        w, v = moral_physics(k1[i], k2[i], power1[i])
        y_win[i], y_val[i] = w, v

    return {
        "X1": X1.astype(DTYPE), "X2": X2.astype(DTYPE),
        "power1": power1.astype(DTYPE).reshape(-1, 1),
        "y_win": y_win, "y_val": y_val,
        "k1": k1, "k2": k2,
        "feat_dim": X1.shape[1],
    }


# ====================================================================
# SECTION 2 — DIFFERENTIABLE PRIMITIVES (hand-written, with backward)
# ====================================================================

def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

def softmax_backward(probs, d_probs):
    """JVP through softmax: given dL/dprobs, return dL/dlogits."""
    dot = np.sum(d_probs * probs, axis=-1, keepdims=True)
    return probs * (d_probs - dot)

def log_softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))

def cross_entropy(logits, targets):
    """Mean CE over a batch. targets: int labels."""
    ls = log_softmax(logits)
    n = logits.shape[0]
    return -ls[np.arange(n), targets].mean()

def entropy(probs, eps=1e-12):
    return -np.sum(probs * np.log(probs + eps), axis=-1)


# ====================================================================
# SECTION 3 — THE AESOPICA ENGINE
# ====================================================================

class AesopicaEngine:
    """Casting-and-consequence model.

    forward pass, per situation:
      1. CAST each agent: soft-assign its surface features to the
         learnable archetype bank A (the 'animals'). This yields a
         convex combination of archetype dispositions, d = c @ A.
         (Aesop looking at Croesus and seeing a particular beast.)
      2. MORAL PHYSICS: feed (d1, d2, d1*d2, d1-d2, power) through a
         small tanh-MLP that learns who prevails and how just it is.
      3. Read off: a who-prevails distribution and a valence scalar.

    The architecture is deliberately NOT a Transformer and NOT
    attention-over-stored-keys. Its one softmax is a *typed casting*
    over a handful of character archetypes — the cognitive primitive
    that is Aesop's alone.
    """

    def __init__(self, feat_dim, disp_dim=8, n_arch=5, hidden=24, seed=0):
        rng = np.random.default_rng(seed)
        self.F = feat_dim
        self.D = disp_dim
        self.K = n_arch
        self.H = hidden
        self.inv_sqrtD = 1.0 / np.sqrt(disp_dim)

        def he(shape, fan_in):
            return (rng.standard_normal(shape) * np.sqrt(2.0 / fan_in)).astype(DTYPE)

        phi_dim = 4 * disp_dim + 1  # [d1, d2, d1*d2, d1-d2, power]
        self.P = {
            "A":  he((n_arch, disp_dim), disp_dim),   # archetype bank
            "Wc": he((feat_dim, disp_dim), feat_dim),  # casting projection
            "bc": np.zeros(disp_dim, dtype=DTYPE),
            "W1": he((phi_dim, hidden), phi_dim),
            "b1": np.zeros(hidden, dtype=DTYPE),
            "W2": he((hidden, 3), hidden),             # [win0, win1, valence]
            "b2": np.zeros(3, dtype=DTYPE),
        }
        # Loss hyper-parameters.
        self.lam_val = 0.5     # weight on valence MSE
        self.lam_ent = 0.02    # crisp-casting pressure (Aesop's crisp types)
        self.lam_reg = 1e-4    # weight decay

    # ---- one agent's casting -------------------------------------
    def _cast(self, X):
        proj = X @ self.P["Wc"] + self.P["bc"]          # (B, D)
        logits = (proj @ self.P["A"].T) * self.inv_sqrtD  # (B, K)
        c = softmax(logits)                              # (B, K) casting
        d = c @ self.P["A"]                              # (B, D) disposition
        return proj, logits, c, d

    # ---- full forward --------------------------------------------
    def forward(self, X1, X2, power):
        proj1, log1, c1, d1 = self._cast(X1)
        proj2, log2, c2, d2 = self._cast(X2)

        phi = np.concatenate([d1, d2, d1 * d2, d1 - d2, power], axis=1)
        z1 = phi @ self.P["W1"] + self.P["b1"]
        h = np.tanh(z1)
        z2 = h @ self.P["W2"] + self.P["b2"]
        win_logits = z2[:, :2]
        val = np.tanh(z2[:, 2])

        cache = dict(X1=X1, X2=X2, power=power,
                     proj1=proj1, c1=c1, d1=d1,
                     proj2=proj2, c2=c2, d2=d2,
                     phi=phi, h=h, z2=z2, win_logits=win_logits, val=val)
        return win_logits, val, c1, c2, cache

    # ---- loss + analytic gradients -------------------------------
    def loss_and_grad(self, batch, need_input_grad=False):
        X1, X2, power = batch["X1"], batch["X2"], batch["power1"]
        y_win, y_val = batch["y_win"], batch["y_val"]
        B = X1.shape[0]
        P = self.P

        win_logits, val, c1, c2, cache = self.forward(X1, X2, power)

        # --- losses ---
        L_win = cross_entropy(win_logits, y_win)
        L_val = np.mean((val - y_val) ** 2)
        L_ent = np.mean(entropy(c1) + entropy(c2))
        L_reg = sum(np.sum(P[k] ** 2) for k in ("A", "Wc", "W1", "W2"))
        loss = L_win + self.lam_val * L_val + self.lam_ent * L_ent + self.lam_reg * L_reg

        # ===== backward =====
        h, phi = cache["h"], cache["phi"]
        d1, d2 = cache["d1"], cache["d2"]

        # d L_win / d win_logits
        sm = softmax(win_logits)
        d_win = sm.copy()
        d_win[np.arange(B), y_win] -= 1.0
        d_win /= B
        # d L_val / d val_pre  (remember the loss weights L_val by lam_val)
        d_val = self.lam_val * (2.0 * (val - y_val) / B) * (1.0 - val ** 2)  # tanh
        dz2 = np.concatenate([d_win, d_val.reshape(-1, 1)], axis=1)  # (B,3)

        g = {}
        g["W2"] = h.T @ dz2
        g["b2"] = dz2.sum(0)
        dh = dz2 @ P["W2"].T
        dz1 = dh * (1.0 - h ** 2)                 # through tanh
        g["W1"] = phi.T @ dz1
        g["b1"] = dz1.sum(0)
        dphi = dz1 @ P["W1"].T                    # (B, 4D+1)

        D = self.D
        g_d1 = dphi[:, 0:D]
        g_d2 = dphi[:, D:2 * D]
        g_prod = dphi[:, 2 * D:3 * D]
        g_diff = dphi[:, 3 * D:4 * D]
        # power column dphi[:,4D:] has no parameter (input only).

        gd1 = g_d1 + d2 * g_prod + g_diff        # dL/dd1 (B,D)
        gd2 = g_d2 + d1 * g_prod - g_diff        # dL/dd2 (B,D)

        gA = np.zeros_like(P["A"])
        gWc = np.zeros_like(P["Wc"])
        gbc = np.zeros_like(P["bc"])
        gX = {"X1": None, "X2": None}

        for tag, c, dd, X, proj in (
            ("X1", c1, gd1, X1, cache["proj1"]),
            ("X2", c2, gd2, X2, cache["proj2"]),
        ):
            # path A:  d = c @ A
            gA += c.T @ dd                       # (K,D)
            dc = dd @ P["A"].T                    # (B,K) from disposition path
            # entropy term: dL_ent/dc = lam_ent/B * -(log c + 1)
            dc = dc + (self.lam_ent / B) * (-(np.log(c + 1e-12) + 1.0))
            # push through softmax: dc -> d logits
            dlog = softmax_backward(c, dc)        # (B,K)
            dS = dlog                             # logits S = proj@A.T*inv_sqrtD
            # path B:  S = (proj @ A.T) * inv_sqrtD
            gA += (dS * self.inv_sqrtD).T @ proj  # (K,D)
            dproj = (dS * self.inv_sqrtD) @ P["A"]  # (B,D)
            # proj = X@Wc + bc
            gWc += X.T @ dproj
            gbc += dproj.sum(0)
            if need_input_grad:
                gX[tag] = dproj @ P["Wc"].T       # (B,F)

        g["A"] = gA
        g["Wc"] = gWc
        g["bc"] = gbc

        # weight decay
        for k in ("A", "Wc", "W1", "W2"):
            g[k] = g[k] + 2.0 * self.lam_reg * P[k]

        metrics = dict(loss=loss, L_win=L_win, L_val=L_val, L_ent=L_ent)
        if need_input_grad:
            return loss, g, metrics, gX
        return loss, g, metrics

    # ---- convenience: predictions & accuracy ---------------------
    def predict(self, batch):
        win_logits, val, c1, c2, _ = self.forward(
            batch["X1"], batch["X2"], batch["power1"])
        win = win_logits.argmax(1)
        return win, val, c1, c2

    def accuracy(self, batch):
        win, val, _, _ = self.predict(batch)
        win_acc = float((win == batch["y_win"]).mean())
        val_acc = float((np.sign(val) == batch["y_val"]).mean())
        return win_acc, val_acc


# ====================================================================
# SECTION 4 — ADAM OPTIMIZER (hand-rolled)
# ====================================================================

class Adam:
    def __init__(self, params, lr=1e-2, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for k in params:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * grads[k] ** 2
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)


# ====================================================================
# SECTION 5 — THE MANDATORY GRADIENT CHECK
#   Central finite differences vs. analytic grads, on every parameter
#   tensor and on the inputs. This is the proof the model is real.
# ====================================================================

def gradient_check(seed=1, eps=1e-6, n_probe=10, tol=1e-5):
    rng = np.random.default_rng(seed)
    # Small dims for a fast, exact check.
    data = make_dataset(n=6, seed=seed, feat_dim=8)
    model = AesopicaEngine(feat_dim=data["feat_dim"], disp_dim=4,
                           n_arch=5, hidden=8, seed=seed)

    loss0, grads, _, gX = model.loss_and_grad(data, need_input_grad=True)

    def loss_only():
        wl, val, c1, c2, _ = model.forward(data["X1"], data["X2"], data["power1"])
        L_win = cross_entropy(wl, data["y_win"])
        L_val = np.mean((val - data["y_val"]) ** 2)
        L_ent = np.mean(entropy(c1) + entropy(c2))
        L_reg = sum(np.sum(model.P[k] ** 2) for k in ("A", "Wc", "W1", "W2"))
        return L_win + model.lam_val * L_val + model.lam_ent * L_ent + model.lam_reg * L_reg

    worst = 0.0
    report = []
    # ---- parameters ----
    for name in model.P:
        arr = model.P[name]
        flat = arr.reshape(-1)
        idxs = rng.choice(flat.size, size=min(n_probe, flat.size), replace=False)
        local = 0.0
        for idx in idxs:
            orig = flat[idx]
            flat[idx] = orig + eps
            Lp = loss_only()
            flat[idx] = orig - eps
            Lm = loss_only()
            flat[idx] = orig
            num = (Lp - Lm) / (2 * eps)
            ana = grads[name].reshape(-1)[idx]
            denom = max(1e-12, abs(num) + abs(ana))
            rel = abs(num - ana) / denom
            local = max(local, rel)
        worst = max(worst, local)
        report.append((name, list(arr.shape), local))

    # ---- inputs (needed for the exact adversarial step) ----
    for name in ("X1", "X2"):
        arr = data[name]
        flat = arr.reshape(-1)
        idxs = rng.choice(flat.size, size=min(n_probe, flat.size), replace=False)
        local = 0.0
        for idx in idxs:
            orig = flat[idx]
            flat[idx] = orig + eps
            Lp = loss_only()
            flat[idx] = orig - eps
            Lm = loss_only()
            flat[idx] = orig
            num = (Lp - Lm) / (2 * eps)
            ana = gX[name].reshape(-1)[idx]
            denom = max(1e-12, abs(num) + abs(ana))
            rel = abs(num - ana) / denom
            local = max(local, rel)
        worst = max(worst, local)
        report.append(("input:" + name, list(arr.shape), local))

    print("  {:<14} {:<14} {:>14}".format("tensor", "shape", "max rel err"))
    print("  " + "-" * 46)
    for name, shape, rel in report:
        flag = "ok" if rel < tol else "FAIL"
        print("  {:<14} {:<14} {:>14.2e}  {}".format(name, str(shape), rel, flag))
    print("  " + "-" * 46)
    print(f"  worst relative error = {worst:.2e}   (tolerance {tol:.0e})")
    return worst < tol


# ====================================================================
# SECTION 6 — TRAINING
# ====================================================================

def train(model, train_d, test_d, epochs=300, batch=128, lr=1e-2, log_every=50):
    opt = Adam(model.P, lr=lr)
    n = train_d["X1"].shape[0]
    rng = np.random.default_rng(7)
    history = []
    for ep in range(1, epochs + 1):
        perm = rng.permutation(n)
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            mb = {k: (v[idx] if hasattr(v, "__len__") and len(v) == n else v)
                  for k, v in train_d.items()}
            _, grads, _ = model.loss_and_grad(mb)
            opt.step(model.P, grads)
        if ep % log_every == 0 or ep == 1 or ep == epochs:
            loss, _, m = model.loss_and_grad(train_d)
            wtr, vtr = model.accuracy(train_d)
            wte, vte = model.accuracy(test_d)
            history.append((ep, loss, wtr, wte, vte))
            print(f"  epoch {ep:4d} | loss {loss:6.4f} | "
                  f"win acc train {wtr:.3f} test {wte:.3f} | "
                  f"valence acc test {vte:.3f}")
    return history


# ====================================================================
# SECTION 7 — DEMONSTRATIONS (the mind, made operable)
# ====================================================================

def archetype_prototype_features(seed=20250001, feat_dim=12):
    """Reconstruct the clean (noise-free) surface rendering of each true
    archetype, so we can ask the trained model 'how do you cast a pure
    fox?' — the casting-interpretability test."""
    render_rng = np.random.default_rng(seed)
    R = render_rng.standard_normal((3, feat_dim)).astype(DTYPE)
    base = TRAITS @ R                          # (K_TRUE, feat_dim)
    base = np.concatenate([base, np.zeros((K_TRUE, 2), dtype=DTYPE)], axis=1)
    return base                                # (K_TRUE, feat_dim+2)


def demo_casting_interpretability(model):
    """Does the engine recover Aesop's character-types from raw surface
    features, with no type labels ever supplied? We feed the clean
    rendering of each true archetype and look at the casting it lands
    on. Then we greedily match learned-cast indices to true archetypes
    and report alignment accuracy on fresh, noisy test agents."""
    print("\n[CASTING INTERPRETABILITY]  (the model was never given type labels)")
    proto = archetype_prototype_features()
    _, _, c, _ = model._cast(proto)
    print("  true type   -> dominant learned cast index  (weight)")
    for i, name in enumerate(ARCHETYPES):
        k = int(c[i].argmax())
        print(f"    {name:<9} -> cast #{k}   ({c[i][k]:.2f})")

    # Build a true->cast mapping from the clean prototypes, then measure
    # how often a fresh noisy agent of each true type is cast the same way.
    mapping = {i: int(c[i].argmax()) for i in range(K_TRUE)}
    test = make_dataset(n=4000, seed=999)
    _, _, ct1, _ = model._cast(test["X1"])
    pred_cast = ct1.argmax(1)
    correct = sum(int(pred_cast[i] == mapping[test["k1"][i]])
                  for i in range(len(pred_cast)))
    align = correct / len(pred_cast)
    distinct = len(set(mapping.values()))
    print(f"  distinct casts used for the 5 types: {distinct}/5")
    print(f"  casting-alignment accuracy on fresh noisy agents: {align:.3f}  "
          f"(chance = {1.0/K_TRUE:.3f})")
    return align


def demo_read_the_fable_backward(model):
    """ABDUCTION — the listener's task. Aesop never states whom the fable
    is about; the audience must bind the unknown agent to a type so that
    the observed outcome makes sense. Given agent 1's features, the
    institutional power, and the OBSERVED outcome, we try casting the
    unknown agent 2 as each archetype prototype and pick the one whose
    predicted outcome best matches what was seen."""
    print("\n[READING THE FABLE BACKWARD]  (abduction: which animal was the other?)")
    proto = archetype_prototype_features()       # (K_TRUE, feat_dim) clean
    rng = np.random.default_rng(123)
    hits = 0
    in_consistent = 0
    set_sizes = []
    trials = 600
    for _ in range(trials):
        k1 = int(rng.integers(0, K_TRUE))
        k2 = int(rng.integers(0, K_TRUE))        # the hidden animal
        p1 = int(rng.integers(0, 2))
        x1 = proto[k1:k1 + 1] + 0.15 * rng.standard_normal((1, proto.shape[1]))
        obs_win, obs_val = moral_physics(k1, k2, p1)
        best_k, best_score = None, -1e18
        consistent = []
        for cand in range(K_TRUE):
            x2 = proto[cand:cand + 1]
            wl, val, _, _, _ = model.forward(
                x1, x2, np.array([[float(p1)]], dtype=DTYPE))
            pred_win = int(wl.argmax(1)[0])
            pred_val = float(np.sign(val[0]))
            if pred_win == obs_win and pred_val == obs_val:
                consistent.append(cand)
            pwin = softmax(wl)[0, obs_win]        # likelihood of observed winner
            pval = -abs(val[0] - obs_val)         # closeness of observed valence
            score = np.log(pwin + 1e-9) + pval
            if score > best_score:
                best_score, best_k = score, cand
        hits += int(best_k == k2)
        in_consistent += int(k2 in consistent)
        set_sizes.append(max(1, len(consistent)))
    print(f"  best-single-guess accuracy : {hits/trials:.3f}  (chance {1.0/K_TRUE:.3f})")
    print(f"  true animal in the outcome-consistent set : {in_consistent/trials:.3f}")
    print(f"  average size of that consistent set       : {np.mean(set_sizes):.2f} of 5")
    print("  -> the ending rarely pins down a unique animal. That ambiguity IS")
    print("     the fable's deniability: the teller can always claim another casting.")
    return hits / trials


def demo_delphi_frameup(model, eps_step=0.05, max_steps=120):
    """THE DELPHI FRAME-UP — adversarial input against a casting system.

    Aesop was killed at Delphi on a planted charge: a temple cup slipped
    into his luggage recast him from envoy to thief. Here we find a court
    scene the engine reads in the defendant's favour, then locate the
    SMALLEST nudge to the defendant's surface features that flips the
    engine's verdict against them — the planted cup. The defendant's
    NATURE never changes; only its rendering does. This is the
    computational shadow of Aesop's death and of any deceptive input to a
    model that classifies-then-decides. The input gradient used here was
    verified exactly in Section 5."""
    print("\n[THE DELPHI FRAME-UP]  (adversarial recasting — the planted cup)")
    proto = archetype_prototype_features()
    lamb_idx = ARCHETYPES.index("lamb")
    defendant0 = proto[lamb_idx:lamb_idx + 1]      # the innocent on trial

    # Search court configurations (who accuses, who holds the floor) for the
    # one most favourable to the innocent defendant (agent 2). We DON'T rig
    # the result — we let the trained engine tell us where acquittal is most
    # secure, then attack THAT.
    best = None
    for ka in range(K_TRUE):
        for p1 in (0, 1):
            x1 = proto[ka:ka + 1]
            wl, val, _, _, _ = model.forward(
                x1, defendant0, np.array([[float(p1)]], dtype=DTYPE))
            p_acquit = float(softmax(wl)[0, 1])    # P(defendant prevails)
            if best is None or p_acquit > best[2]:
                best = (ka, p1, p_acquit)
    ka, p1, p_acquit0 = best
    power = np.array([[float(p1)]], dtype=DTYPE)
    accuser = proto[ka:ka + 1]
    wl0, val0, _, c2_0, _ = model.forward(accuser, defendant0, power)
    print(f"  the strongest acquittal the engine offers the innocent:")
    print(f"    accuser cast≈{ARCHETYPES[ka]}, court-power={'accuser' if p1 else 'defendant'}")
    print(f"    P(innocent prevails) = {p_acquit0:.3f}, valence = {val0[0]:+.2f}")
    print(f"    the defendant is cast as index {int(c2_0.argmax())} "
          f"(weight {c2_0[0].max():.2f})")

    # Adversary's objective: drive P(defendant prevails) DOWN. We claim the
    # "label" is acquittal and ASCEND its loss gradient wrt the defendant's
    # features only (FGSM-style, exact analytic input gradient).
    x = defendant0.copy()
    flipped_at = None
    for step in range(1, max_steps + 1):
        batch = {"X1": accuser, "X2": x, "power1": power,
                 "y_win": np.array([1]), "y_val": np.array([+1.0], dtype=DTYPE)}
        _, _, _, gX = model.loss_and_grad(batch, need_input_grad=True)
        x = x + eps_step * np.sign(gX["X2"])       # plant the feature
        wl, val, _, c2, _ = model.forward(accuser, x, power)
        p_acquit = float(softmax(wl)[0, 1])
        if p_acquit < 0.5 and flipped_at is None:
            flipped_at = step
            break
    wl, val, _, c2, _ = model.forward(accuser, x, power)
    p_acquit = float(softmax(wl)[0, 1])
    drift = float(np.linalg.norm(x - defendant0))
    verb = f"after {flipped_at} nudges" if flipped_at else f"after {max_steps} nudges (cap)"
    print(f"  {verb}: P(innocent prevails) = {p_acquit:.3f}, "
          f"valence = {val[0]:+.2f}")
    print(f"    the defendant is now cast as index {int(c2.argmax())} "
          f"(weight {c2[0].max():.2f})")
    print(f"    surface drift (L2) needed to overturn the verdict: {drift:.3f}")
    print("  -> the innocent's NATURE never changed; a planted feature recast it,")
    print("     and the recasting changed its fate. Aesop's cliff, in miniature.")
    return drift, (flipped_at is not None)


# ====================================================================
# MAIN
# ====================================================================

def main():
    np.random.seed(42)
    print("=" * 68)
    print("THE AESOPICA ENGINE  —  casting-and-consequence  (Chapter 30)")
    print("=" * 68)

    print("\n[1] GRADIENT CHECK  (central finite differences vs. analytic)")
    ok = gradient_check()
    assert ok, "Gradient check FAILED — backward is wrong."
    print("  gradient check PASSED.")

    print("\n[2] DATA")
    train_d = make_dataset(n=4000, seed=1)
    test_d = make_dataset(n=1500, seed=2)
    print(f"  train situations: {train_d['X1'].shape[0]}  "
          f"test: {test_d['X1'].shape[0]}  feature dim: {train_d['feat_dim']}")
    # base rates so the accuracies below are meaningful
    base_win = max(np.mean(train_d["y_win"] == 0), np.mean(train_d["y_win"] == 1))
    print(f"  majority-class win baseline: {base_win:.3f}")

    print("\n[3] TRAINING  (Adam, hand-rolled backprop)")
    model = AesopicaEngine(feat_dim=train_d["feat_dim"], disp_dim=8,
                           n_arch=5, hidden=24, seed=3)
    train(model, train_d, test_d, epochs=300, batch=128, lr=1e-2, log_every=50)
    wte, vte = model.accuracy(test_d)
    print(f"  FINAL held-out: who-prevails acc {wte:.3f}, "
          f"valence acc {vte:.3f}")
    assert wte > 0.80, "Model failed to learn the moral physics."

    align = demo_casting_interpretability(model)
    abd = demo_read_the_fable_backward(model)
    drift, flipped = demo_delphi_frameup(model)

    print("\n" + "=" * 68)
    print("SUMMARY")
    print("=" * 68)
    print(f"  gradient check ............... PASSED")
    print(f"  held-out who-prevails acc .... {wte:.3f}  (baseline {base_win:.3f})")
    print(f"  held-out valence acc ......... {vte:.3f}")
    print(f"  casting alignment acc ........ {align:.3f}  (chance {1/K_TRUE:.3f})")
    print(f"  abductive 'read it backward' . {abd:.3f}  (chance {1/K_TRUE:.3f})")
    print(f"  Delphi frame-up drift (L2) ... {drift:.3f}")
    print("\n  The engine learned to CAST raw situations as character-types,")
    print("  to run the physics of those characters forward to a verdict,")
    print("  to read a fable BACKWARD by abduction, and it can be FRAMED")
    print("  by a planted feature exactly as Aesop was at Delphi.")
    print("=" * 68)


if __name__ == "__main__":
    main()
