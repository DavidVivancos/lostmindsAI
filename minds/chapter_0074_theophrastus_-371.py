#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 chapter_0074_theophrastus_-371.py  --  THEOPHRASTUS OF ERESUS (c. 371 - c. 287 BCE)
 Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
Tome 4 Minds 61 - 80 Available on Amazon https://www.amazon.com/dp/B0H71JC95Q
Resume and Interactive Demos at https://artificiology.com/
Author: David Vivancos · Chapter 0074 · Theophrastus of Eresus
================================================================================

WHY THIS ARCHITECTURE (and why NOT a transformer)
--------------------------------------------------------------------------------
Theophrastus was Aristotle's successor, but his cognitive method is his own and
it cuts *against* his teacher's. Where Aristotle extracts an ESSENCE (the form
that makes a thing what it is) top-down, Theophrastus works BOTTOM-UP. He is the
father of botany because he gathered ~500 plants and sorted them by their
*observable distinctive features* -- habit (tree / shrub / under-shrub / herb),
leaves, flowers, fruits, modes of generation -- rather than by a deduced nature.
His "Characters" defines each personality type not by an inner cause but
OSTENSIVELY: a one-line definition followed by a string of concrete behavioural
vignettes ("the Flatterer is the one who, while walking with you, says 'see how
everyone looks at you'..."). A type just IS its bundle of typical instances.

Two further signatures, both from his surviving works, shape the model:

  (1) LIMITED TELEOLOGY.  In his "Metaphysics" Theophrastus argues that not
      everything is "for the sake of something": the deer's over-large antlers,
      the insect that lives a single day, freak wet/dry seasons resist a clean
      purposive account. The right move is sometimes to ABSTAIN -- to say "this
      fits no kind; it is coincidence" -- rather than force a label.

  (2) OIKEIOS TROPOS -- "a method proper to each field" (Metaphysics 9a11).
      The features that distinguish plants are not the features that distinguish
      characters. Each domain deserves its own weighting of what to attend to.

The matching modern formalism is the ATTENTION-LEARNING EXEMPLAR NETWORK
(ALCOVE; Kruschke 1992), itself a learnable form of Nosofsky's Generalized
Context Model (1986). It:
  * stores concrete labelled EXEMPLARS (Theophrastus' collected observations),
  * classifies a new case by its attention-weighted similarity to those stored
    instances (no essences, no rules),
  * LEARNS a per-dimension ATTENTION vector deciding which features matter,
  * here, learns a SEPARATE attention vector per domain (oikeios tropos), and
  * carries an explicit "none / coincidence" channel for LIMITED TELEOLOGY.

Everything below is pure NumPy, built from scratch. A finite-difference gradient
check (mandatory) verifies every analytic gradient; a real training loop drives
the loss down; self-tests confirm the Theophrastean behaviours. Run the file:
    python3 chapter_0074_theophrastus_-371.py
The verified console output is reproduced in the accompanying chapter.

This file is referenced in prose only by what it does, never by name.
================================================================================
"""

import numpy as np

RNG = np.random.default_rng(74)   # 74 = Theophrastus' index in the corpus


# ============================================================================
# 0. SMOOTH HELPERS
#    softplus keeps positive scalars positive; softmax turns raw logits into a
#    normalised attention distribution (sums to 1). Both are smooth so the
#    finite-difference gradient check has no kinks to trip over.
# ============================================================================
def softplus(x):
    # numerically stable log(1+e^x)
    return np.logaddexp(0.0, x)

def d_softplus(x):
    # derivative of softplus = logistic sigmoid
    return 1.0 / (1.0 + np.exp(-x))

def softmax(z, axis=-1):
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


# ============================================================================
# 1. THE SYNTHETIC WORLD  --  a miniature of Theophrastus' two catalogues
#    Joint feature space = [ 5 botanical features | 6 behavioural features ].
#    A plant lives in the first block (last 6 ~ 0); a character lives in the
#    second block (first 5 ~ 0). The per-domain attention must DISCOVER this.
# ============================================================================
D_PLANT = 5          # habit, leaf-breadth, has-flower, root-depth, seed-leaves
D_CHAR  = 6          # flatters, boasts, grumbles, hoards, superstition, talks
D = D_PLANT + D_CHAR # = 11 joint dimensions

# --- botanical prototypes (3 plant kinds), values in the first 5 dims ---------
#                    habit  leafW  flower  rootD  seedleaves
PLANT_PROTOS = {
    0: np.array([1.00, 0.85, 1.0, 0.90, 1.0]),  # broad-leaved tree (oak-like, dicot)
    1: np.array([0.00, 0.10, 0.2, 0.30, 0.0]),  # cereal grass (monocot, narrow leaf)
    2: np.array([0.50, 0.55, 0.8, 0.50, 1.0]),  # flowering shrub (dicot)
}
# --- behavioural prototypes (4 Theophrastan character types), in the 6 dims ---
#                     flat  boast grumble hoard superst talk
CHAR_PROTOS = {
    3: np.array([1.0, 0.2, 0.1, 0.1, 0.1, 0.7]),  # the Flatterer (Kolax)
    4: np.array([0.2, 1.0, 0.1, 0.2, 0.1, 0.8]),  # the Boaster   (Alazon)
    5: np.array([0.1, 0.1, 1.0, 0.7, 0.2, 0.5]),  # the Grumbler  (Mempsimoiros)
    6: np.array([0.1, 0.1, 0.2, 0.2, 1.0, 0.3]),  # the Superstitious (Deisidaimon)
}
PLANT_CLASSES = sorted(PLANT_PROTOS)     # [0,1,2]
CHAR_CLASSES  = sorted(CHAR_PROTOS)      # [3,4,5,6]
C = len(PLANT_CLASSES) + len(CHAR_CLASSES)   # 7 real classes
NONE_CLASS = C                               # index 7 = "no kind / coincidence"

PLANT_DOMAIN, CHAR_DOMAIN = 0, 1


def _embed(domain, partial):
    """Place a domain-specific feature vector into the joint 11-D space."""
    v = np.zeros(D)
    if domain == PLANT_DOMAIN:
        v[:D_PLANT] = partial
    else:
        v[D_PLANT:] = partial
    return v


def make_dataset(n_per_class=18, n_none=40, noise=0.06):
    """Generate concrete observations around each prototype, plus 'coincidence'
    instances that sit far from every prototype (Theophrastus' anomalies that
    no purposive category fits)."""
    X, y, dom = [], [], []
    for cls, proto in PLANT_PROTOS.items():
        for _ in range(n_per_class):
            X.append(_embed(PLANT_DOMAIN, np.clip(proto + RNG.normal(0, noise, D_PLANT), 0, 1)))
            y.append(cls); dom.append(PLANT_DOMAIN)
    for cls, proto in CHAR_PROTOS.items():
        for _ in range(n_per_class):
            X.append(_embed(CHAR_DOMAIN, np.clip(proto + RNG.normal(0, noise, D_CHAR), 0, 1)))
            y.append(cls); dom.append(CHAR_DOMAIN)
    # --- coincidental / "no kind" instances: far from all prototypes ---------
    protos = {**PLANT_PROTOS, **CHAR_PROTOS}
    made = 0
    while made < n_none:
        d = int(RNG.integers(0, 2))
        cand = RNG.uniform(0, 1, D_PLANT if d == PLANT_DOMAIN else D_CHAR)
        # accept only if dissimilar to every prototype in that domain
        far = True
        for cls, proto in protos.items():
            same = (cls in PLANT_CLASSES) == (d == PLANT_DOMAIN)
            if same and np.linalg.norm(cand - proto) < 0.55:
                far = False; break
        if far:
            X.append(_embed(d, cand)); y.append(NONE_CLASS); dom.append(d); made += 1
    X = np.array(X); y = np.array(y); dom = np.array(dom)
    perm = RNG.permutation(len(X))
    return X[perm], y[perm], dom[perm]


def make_exemplar_bank(per_class=4, noise=0.05):
    """The stored observations the model reasons FROM -- a labelled support set,
    just as Theophrastus reasons from his herbarium and his vignettes."""
    E, lab = [], []
    for cls, proto in PLANT_PROTOS.items():
        for _ in range(per_class):
            E.append(_embed(PLANT_DOMAIN, np.clip(proto + RNG.normal(0, noise, D_PLANT), 0, 1)))
            lab.append(cls)
    for cls, proto in CHAR_PROTOS.items():
        for _ in range(per_class):
            E.append(_embed(CHAR_DOMAIN, np.clip(proto + RNG.normal(0, noise, D_CHAR), 0, 1)))
            lab.append(cls)
    return np.array(E), np.array(lab)


# ============================================================================
# 2. THE MODEL  --  ExemplarTypologist
#    Parameters (all smooth, all gradient-checked):
#      raw_w  : (2, D)  per-domain attention logits  -> w_d = softmax(raw_w[d])
#      raw_c  : (2,)    per-domain specificity        -> c_d = softplus(raw_c[d])
#      A      : (C, M)  exemplar -> real-class associations (learned)
#      raw_phi: scalar  response scaling              -> phi = softplus(raw_phi)
#      b_none : scalar  the limited-teleology logit (constant "coincidence" vote)
# ============================================================================
class ExemplarTypologist:
    def __init__(self, E, E_labels):
        self.E = E                       # (M, D) frozen stored observations
        self.E_labels = E_labels
        self.M = E.shape[0]
        # init associations so each exemplar gently votes for its own class
        A0 = np.zeros((C, self.M))
        for j, lab in enumerate(E_labels):
            A0[lab, j] = 1.0
        self.params = {
            "raw_w":  RNG.normal(0, 0.01, (2, D)),     # near-uniform attention
            "raw_c":  np.array([0.0, 0.0]),            # c_d = softplus(0) ~ 0.69
            "A":      A0 + RNG.normal(0, 0.01, (C, self.M)),
            "raw_phi": np.array(1.0),                  # scalar (0-d array)
            "b_none": np.array(0.0),
        }

    # ---- forward for ONE example, returning logits + a cache for backprop ----
    def forward_one(self, x, domain):
        p = self.params
        w = softmax(p["raw_w"][domain])            # (D,) attention, sums to 1
        c = softplus(p["raw_c"][domain])           # scalar specificity > 0
        phi = softplus(p["raw_phi"])               # scalar scaling > 0
        diff = x[None, :] - self.E                  # (M, D)
        sq = diff ** 2                              # (M, D)
        dist = sq @ w                               # (M,)  attention-weighted dist
        h = np.exp(-c * dist)                       # (M,)  exemplar activations
        o_real = phi * (p["A"] @ h)                 # (C,)  evidence per real class
        logits = np.concatenate([o_real, [p["b_none"].item()]])  # (C+1,)
        cache = dict(x=x, domain=domain, w=w, c=c, phi=phi,
                     sq=sq, dist=dist, h=h)
        return logits, cache

    def predict(self, x, domain):
        logits, _ = self.forward_one(x, domain)
        return softmax(logits)

    # ---- analytic gradient for ONE example -----------------------------------
    def backward_one(self, cache, target, grads):
        p = self.params
        x, domain = cache["x"], cache["domain"]
        w, c, phi = cache["w"], cache["c"], cache["phi"]
        sq, dist, h = cache["sq"], cache["dist"], cache["h"]

        logits = np.concatenate([phi * (p["A"] @ h), [p["b_none"].item()]])
        probs = softmax(logits)
        g = probs.copy(); g[target] -= 1.0          # dL/dlogits  (C+1,)
        g_real, g_none = g[:C], g[C]

        # b_none
        grads["b_none"] += g_none
        # phi : o_real = phi * (A@h);  s = A@h
        s = p["A"] @ h                               # (C,)
        dL_dphi = float(np.dot(g_real, s))
        grads["raw_phi"] += dL_dphi * d_softplus(p["raw_phi"])
        # A : dL/dA[i,j] = g_real_i * phi * h_j
        grads["A"] += phi * np.outer(g_real, h)
        # h : dL/dh_j = phi * sum_i g_real_i A[i,j]
        gh = phi * (g_real @ p["A"])                 # (M,)
        # c : h = exp(-c dist) ; dh/dc = -dist*h
        dL_dc = float(np.sum(gh * (-dist * h)))
        grads["raw_c"][domain] += dL_dc * d_softplus(p["raw_c"][domain])
        # dist : dh/ddist = -c*h ; dL/ddist_j = gh_j * (-c h_j)
        gd = gh * (-c * h)                           # (M,)
        # w : dist_j = sum_k w_k sq_jk ; dL/dw_k = sum_j gd_j sq_jk
        dL_dw = sq.T @ gd                            # (D,)
        # backprop through softmax(raw_w[domain]) -> w
        dL_draw = w * (dL_dw - np.dot(dL_dw, w))     # softmax jacobian-vector
        grads["raw_w"][domain] += dL_draw

    # ---- batch loss + gradients ---------------------------------------------
    def loss_and_grads(self, X, y, dom):
        grads = {k: np.zeros_like(v) for k, v in self.params.items()}
        total = 0.0
        for i in range(len(X)):
            logits, cache = self.forward_one(X[i], dom[i])
            probs = softmax(logits)
            total += -np.log(probs[y[i]] + 1e-12)
            self.backward_one(cache, y[i], grads)
        n = len(X)
        total /= n
        for k in grads:
            grads[k] = grads[k] / n
        return total, grads

    def loss_only(self, X, y, dom):
        total = 0.0
        for i in range(len(X)):
            logits, _ = self.forward_one(X[i], dom[i])
            probs = softmax(logits)
            total += -np.log(probs[y[i]] + 1e-12)
        return total / len(X)


# ============================================================================
# 3. MANDATORY FINITE-DIFFERENCE GRADIENT CHECK
#    Compares analytic grads to numerical (central-difference) grads on every
#    parameter array. Must pass before any training is trusted.
# ============================================================================
def gradient_check(model, X, y, dom, eps=1e-6, tol=1e-5):
    _, analytic = model.loss_and_grads(X, y, dom)
    print("  Finite-difference gradient check")
    print("  " + "-" * 56)
    all_ok = True
    for name in model.params:
        P = model.params[name]
        flat = P.ravel()
        num = np.zeros_like(flat)
        # sample up to 12 coordinates per array to keep the check fast
        idxs = range(flat.size) if flat.size <= 12 else \
               RNG.choice(flat.size, 12, replace=False)
        for k in idxs:
            orig = flat[k]
            flat[k] = orig + eps; lp = model.loss_only(X, y, dom)
            flat[k] = orig - eps; lm = model.loss_only(X, y, dom)
            flat[k] = orig
            num[k] = (lp - lm) / (2 * eps)
        a = analytic[name].ravel()
        sel = list(idxs)
        rel = np.abs(a[sel] - num[sel]) / (np.abs(a[sel]) + np.abs(num[sel]) + 1e-12)
        worst = float(np.max(rel))
        ok = worst < tol
        all_ok &= ok
        print(f"    {name:8s} | max rel.err = {worst:.2e} | {'PASS' if ok else 'FAIL'}")
    print("  " + "-" * 56)
    print(f"  GRADIENT CHECK: {'ALL PASS' if all_ok else 'FAILURE'}\n")
    return all_ok


# ============================================================================
# 4. TRAINING LOOP  (full-batch Adam, pure NumPy)
# ============================================================================
def train(model, X, y, dom, epochs=400, lr=0.05):
    m = {k: np.zeros_like(v) for k, v in model.params.items()}
    v = {k: np.zeros_like(v) for k, v in model.params.items()}
    b1, b2, eps = 0.9, 0.999, 1e-8
    print("  Training (Adam, full batch)")
    print("  " + "-" * 56)
    for t in range(1, epochs + 1):
        loss, grads = model.loss_and_grads(X, y, dom)
        for k in model.params:
            m[k] = b1 * m[k] + (1 - b1) * grads[k]
            v[k] = b2 * v[k] + (1 - b2) * grads[k] ** 2
            mhat = m[k] / (1 - b1 ** t)
            vhat = v[k] / (1 - b2 ** t)
            model.params[k] = model.params[k] - lr * mhat / (np.sqrt(vhat) + eps)
        if t == 1 or t % 80 == 0:
            acc = accuracy(model, X, y, dom)
            print(f"    epoch {t:4d} | loss {loss:.4f} | train acc {acc:5.1%}")
    print("  " + "-" * 56 + "\n")


def accuracy(model, X, y, dom):
    correct = 0
    for i in range(len(X)):
        pred = int(np.argmax(model.predict(X[i], dom[i])))
        correct += (pred == y[i])
    return correct / len(X)


# ============================================================================
# 5. SELF-TESTS  --  do the learned behaviours match Theophrastus' method?
# ============================================================================
CLASS_NAMES = {0: "broad-leaved tree", 1: "cereal grass", 2: "flowering shrub",
               3: "the Flatterer", 4: "the Boaster", 5: "the Grumbler",
               6: "the Superstitious", NONE_CLASS: "NO KIND (coincidence)"}

def self_tests(model):
    print("  Self-tests")
    print("  " + "-" * 56)
    passed = 0; total = 0

    # held-out test set
    Xte, yte, dte = make_dataset(n_per_class=8, n_none=24)
    acc = accuracy(model, Xte, yte, dte)
    total += 1; ok = acc > 0.85; passed += ok
    print(f"    [1] held-out accuracy {acc:5.1%} (>85%)            {'PASS' if ok else 'FAIL'}")

    # a clean plant is correctly typed by features, not essence
    x_oak = _embed(PLANT_DOMAIN, PLANT_PROTOS[0])
    pred = int(np.argmax(model.predict(x_oak, PLANT_DOMAIN)))
    total += 1; ok = pred == 0; passed += ok
    print(f"    [2] broad-leaved tree -> {CLASS_NAMES[pred]:22s}  {'PASS' if ok else 'FAIL'}")

    # a clean character is correctly typed
    x_flat = _embed(CHAR_DOMAIN, CHAR_PROTOS[3])
    pred = int(np.argmax(model.predict(x_flat, CHAR_DOMAIN)))
    total += 1; ok = pred == 3; passed += ok
    print(f"    [3] Flatterer behaviour -> {CLASS_NAMES[pred]:20s}  {'PASS' if ok else 'FAIL'}")

    # LIMITED TELEOLOGY: an anomaly must trigger abstention, not a forced label
    abstained = 0; trials = 40
    for _ in range(trials):
        d = int(RNG.integers(0, 2))
        cand = RNG.uniform(0, 1, D_PLANT if d == PLANT_DOMAIN else D_CHAR)
        far = all(np.linalg.norm(cand - pr) > 0.6
                  for cl, pr in {**PLANT_PROTOS, **CHAR_PROTOS}.items()
                  if (cl in PLANT_CLASSES) == (d == PLANT_DOMAIN))
        if not far:
            continue
        x = _embed(d, cand)
        if int(np.argmax(model.predict(x, d))) == NONE_CLASS:
            abstained += 1
    total += 1; ok = abstained >= trials * 0.5; passed += ok
    print(f"    [4] abstains on {abstained}/{trials} anomalies (limited teleology) {'PASS' if ok else 'FAIL'}")

    # OIKEIOS TROPOS: each domain's attention concentrates on its own block
    w_plant = softmax(model.params["raw_w"][PLANT_DOMAIN])
    w_char  = softmax(model.params["raw_w"][CHAR_DOMAIN])
    mass_plant_on_plant = w_plant[:D_PLANT].sum()
    mass_char_on_char   = w_char[D_PLANT:].sum()
    total += 1; ok = mass_plant_on_plant > 0.7 and mass_char_on_char > 0.7; passed += ok
    print(f"    [5] oikeios tropos: plant-attn on botany {mass_plant_on_plant:.0%}, "
          f"char-attn on behaviour {mass_char_on_char:.0%}  {'PASS' if ok else 'FAIL'}")
    print("  " + "-" * 56)
    print(f"  SELF-TESTS: {passed}/{total} passed\n")
    return passed == total


# ============================================================================
# 6. MAIN
# ============================================================================
def main():
    print("=" * 64)
    print(" THEOPHRASTUS  --  Attention-Learning Exemplar Typologist")
    print(" (bottom-up kinds by distinctive features; limited teleology)")
    print("=" * 64 + "\n")

    X, y, dom = make_dataset()
    E, E_lab = make_exemplar_bank()
    model = ExemplarTypologist(E, E_lab)
    print(f"  dataset: {len(X)} observations | exemplar bank: {model.M} | "
          f"classes: {C} + 1 'none'\n")

    ok_grad = gradient_check(model, X[:24], y[:24], dom[:24])
    train(model, X, y, dom, epochs=400, lr=0.05)
    ok_self = self_tests(model)

    print("=" * 64)
    print(f" RESULT: gradient_check={'PASS' if ok_grad else 'FAIL'} | "
          f"self_tests={'PASS' if ok_self else 'FAIL'}")
    print("=" * 64)


if __name__ == "__main__":
    main()
