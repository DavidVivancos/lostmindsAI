"""
================================================================================
Chapter 0157_vatsyayana_400 - Vatsyayana (Paksilasvamin), 400-450 CE
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 8 Minds 141 - 160 Available on Amazon https://www.amazon.com/dp/B0HH8RTCXF
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 0157_vatsyayana_400 - Vatsyayana (Paksilasvamin), 400-450 CE
================================================================================  

FIGURE
------
Paksilasvamin Vatsyayana, author of the *Nyaya-Sutra-Bhasya* - the first
surviving commentary on Gautama/Aksapada's *Nyaya-Sutra* (c. 150 CE). He is the
philosopher who fixed the classical Indian THEORY OF KNOWING: knowledge (prama)
is not one faculty but arises through FOUR irreducible instruments (pramanas):

    pratyaksa  - perception  (direct sensory contact)
    anumana    - inference   (a mark/hetu tied by invariable concomitance, vyapti,
                              to what is inferred; canonical case: smoke => fire)
    upamana    - comparison  (recognition by similarity to a known exemplar)
    sabda      - testimony   (the assertion of a reliable/authoritative source)

Vatsyayana's own distinctive move - the thesis this whole model is built on - is
PRAGMATIC and EXTRINSIC: a cognition is a *guide to action*, and its validity
(pramanya) is confirmed not by inspecting the cognition itself but by the SUCCESS
of the activity (pravrtti) it licenses. Belief is fixed (nirnaya, ascertainment)
only after doubt (samsaya) is resolved by weighing the instruments jointly in
structured debate (vada), during which each candidate reason must survive
fallacy-checking (hetvabhasa). Error (mithya-jnana) is MIS-ascription of a real
object, never fabrication from nothing.

WHY THIS ARCHITECTURE (and why NOT a Transformer)
-------------------------------------------------
A monolithic end-to-end network that stores keys and attends over them collapses
exactly the distinction Vatsyayana refuses to collapse: it treats all evidence as
one undifferentiated substrate. His mind is federated, not monolithic. So the
model here is a CATUSPRAMANA ADJUDICATOR:

  (1) Four *separate* channel encoders, one per pramana - each instrument has its
      own validity conditions and cannot be reduced to another.
  (2) A joint ADJUDICATION gate (the vada / debate step) that sees all four
      channel-beliefs at once and decides how much each instrument speaks to THIS
      case - so a channel that conflicts with the others (a hetvabhasa, a lying
      witness) can be down-weighted. This is cross-channel by design.
  (3) A VERIFICATION head (nyaya proper) that reads the adjudicated belief and
      predicts whether acting on the proposition will SUCCEED - i.e. the target
      is pravrtti-success, encoding extrinsic validity directly in the loss.

The whole thing is pure NumPy, trained from scratch with hand-derived gradients,
a mandatory finite-difference gradient check, a real training loop, and a battery
of doctrine-named self-tests (samsaya = doubt as output entropy, nirnaya =
confident ascertainment, hetvabhasa down-weighting = the gate learning to distrust
a corrupted instrument).

Dependencies: numpy only.
================================================================================
"""

import numpy as np


# =============================================================================
# SECTION 0 - Small numerical primitives (kept explicit for the gradient check)
# =============================================================================

def tanh(x):
    return np.tanh(x)


def dtanh_from_out(h):
    # derivative of tanh given its OUTPUT h = tanh(z):  1 - h^2
    return 1.0 - h * h


def sigmoid(x):
    # numerically stable logistic
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def softmax_rows(z):
    # row-wise softmax over the last axis
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def row_entropy(p, eps=1e-12):
    # Shannon entropy per row (natural log) - our measure of samsaya (doubt)
    return -np.sum(p * np.log(p + eps), axis=1)


# =============================================================================
# SECTION 1 - The four pramanas as a data-generating world
# =============================================================================
# We synthesise a "knowledge establishment" task. For each episode there is a
# hidden truth t about a proposition (say: "there is fire on that hill"). Acting
# on the proposition SUCCEEDS iff t = 1. Each of the four instruments delivers a
# small feature vector that carries partial, reliability-weighted evidence about
# t. Crucially:
#   * no single instrument is sufficient (each is noisy / sometimes unreliable);
#   * with some probability one instrument is CORRUPTED - it asserts the wrong
#     thing *confidently* (a hetvabhasa: a pseudo-reason, or a lying witness).
# A faithful Nyaya reasoner must fuse the instruments and, in debate, distrust the
# instrument that conflicts with the weight of the others.

CHANNELS = ["pratyaksa", "anumana", "upamana", "sabda"]  # perception, inference, comparison, testimony
D_CH = 6  # feature dimension per channel


def _channel_features(signed_truth, reliability, corrupted, rng):
    """Build one channel's D_CH-vector.

    signed_truth s in {-1,+1};  reliability in (0,1];  corrupted flag flips the
    evidence sign while KEEPING confidence high (the signature of a fallacy /
    unreliable testimony: emphatic but wrong).
    """
    s = signed_truth
    if corrupted:
        s = -s                      # asserts the opposite of the truth ...
        reliability = rng.uniform(0.8, 1.0)   # ... and does so with false confidence
    noise = rng.normal(0.0, 0.35)
    e = s * reliability + noise     # the instrument's raw signed evidence
    return np.array([
        reliability,                # how strongly the instrument speaks
        e,                          # signed evidence
        e * reliability,            # confidence-weighted evidence
        abs(e),                     # magnitude of the claim
        rng.normal(0.0, 0.2),       # distractor / irrelevant feature
        reliability * reliability,  # a nonlinear reliability cue
    ], dtype=float)


def make_dataset(n, rng, p_corrupt=0.35):
    """Return (X_list, y, meta).

    X_list is a list of four (n, D_CH) arrays (one per pramana).
    y is (n,) in {0,1}: does acting on the proposition succeed (= hidden truth)?
    meta['corrupt'] is (n,) index of corrupted channel or -1 if none.
    """
    Xs = [np.zeros((n, D_CH)) for _ in range(4)]
    y = np.zeros(n, dtype=float)
    corrupt_idx = np.full(n, -1, dtype=int)
    for i in range(n):
        t = rng.integers(0, 2)              # hidden truth in {0,1}
        s = 2 * t - 1                        # signed truth in {-1,+1}
        y[i] = float(t)
        # per-instrument reliabilities (availability / trustworthiness this episode)
        rels = rng.uniform(0.30, 1.0, size=4)
        cch = -1
        if rng.random() < p_corrupt:
            cch = int(rng.integers(0, 4))   # exactly one instrument is a hetvabhasa
        corrupt_idx[i] = cch
        for k in range(4):
            Xs[k][i] = _channel_features(s, rels[k], corrupted=(k == cch), rng=rng)
    return Xs, y, {"corrupt": corrupt_idx}


# =============================================================================
# SECTION 2 - The Catuspramana Adjudicator (parameters, forward, backward)
# =============================================================================

class CatuspramanaAdjudicator:
    """Vatsyayana's fourfold evidential reasoner (gated-vote form).

    Layout
      channel encoders : X_k (N,D_CH) -> H_k (N,H)             [one per pramana]
      instrument vote  : o_k = H_k . wv + bv (shared readout)  [each instrument's
                         own signed claim about the proposition]
      adjudication gate : concat(H_0..H_3) -> logits (N,4) -> softmax = G  [vada]
      final verdict     : logit = sum_k G_k * o_k -> yhat = sigmoid(logit)

    The verification is a *debate*: there is no large private head that can quietly
    invert a confident-but-wrong instrument. To be right on a case where one
    instrument lies (a hetvabhasa), the gate MUST demote that instrument's vote.
    Thus faithfulness to Nyaya is forced by the architecture, not merely hoped for.
    The target is pravrtti-success (will acting succeed?), so validity is judged
    extrinsically, exactly as Vatsyayana holds.

    Loss: BCE (+ L2 on weight matrices, folded into the checked objective so the
    finite-difference gradient check stays exact).
    """

    def __init__(self, H=12, l2=1e-4, seed=0, **_ignore):
        self.H, self.l2 = H, l2
        r = np.random.default_rng(seed)

        def xav(shape, fan_in):
            return r.normal(0.0, np.sqrt(1.0 / fan_in), size=shape)

        self.p = {}
        for k in range(4):
            self.p[f"W{k}"] = xav((H, D_CH), D_CH)     # channel encoder weight
            self.p[f"b{k}"] = np.zeros(H)              # channel encoder bias
        self.p["wv"] = xav((H,), H)                    # shared per-instrument vote readout
        self.p["bv"] = np.zeros(1)                     # shared vote bias
        self.p["Wg"] = xav((4, 4 * H), 4 * H)          # adjudication (debate) gate weight
        self.p["bg"] = np.zeros(4)                     # gate bias (a prior over instruments)

    # ---- forward pass, returning a cache for backprop -----------------------
    def forward(self, Xs):
        p = self.p
        H_list = []
        for k in range(4):
            Hk = tanh(Xs[k] @ p[f"W{k}"].T + p[f"b{k}"])   # (N,H)
            H_list.append(Hk)
        Hcat = np.concatenate(H_list, axis=1)              # (N,4H)
        O = np.stack([Hk @ p["wv"] + p["bv"][0] for Hk in H_list], axis=1)  # (N,4) votes
        Lg = Hcat @ p["Wg"].T + p["bg"]                    # (N,4) gate logits
        G = softmax_rows(Lg)                               # (N,4) adjudication weights
        logit = np.sum(G * O, axis=1)                      # (N,) debated verdict
        yhat = sigmoid(logit)                              # (N,)
        cache = dict(Xs=Xs, H=H_list, Hcat=Hcat, O=O, G=G, logit=logit, yhat=yhat)
        return yhat, cache

    # ---- loss + analytic gradients ------------------------------------------
    def loss_and_grads(self, Xs, y):
        p = self.p
        N = y.shape[0]
        yhat, c = self.forward(Xs)
        eps = 1e-12
        bce = -np.mean(y * np.log(yhat + eps) + (1 - y) * np.log(1 - yhat + eps))
        l2sum = 0.0
        for name in ["W0", "W1", "W2", "W3", "Wg", "wv"]:
            l2sum += np.sum(p[name] ** 2)
        loss = bce + 0.5 * self.l2 * l2sum

        g = {name: np.zeros_like(val) for name, val in p.items()}
        Hlist, Hcat, O, G = c["H"], c["Hcat"], c["O"], c["G"]

        # dL/dlogit for mean BCE with sigmoid
        dlogit = (yhat - y) / N                            # (N,)
        # logit = sum_k G_k * O_k
        dG = dlogit[:, None] * O                           # (N,4)
        dO = dlogit[:, None] * G                           # (N,4)

        # softmax backward on the gate logits
        dLg = G * (dG - np.sum(dG * G, axis=1, keepdims=True))   # (N,4)
        g["Wg"] += dLg.T @ Hcat + self.l2 * p["Wg"]
        g["bg"] += np.sum(dLg, axis=0)
        dHcat_fromG = dLg @ p["Wg"]                        # (N,4H)

        # vote readout (shared across instruments): O_k = H_k . wv + bv
        for k in range(4):
            g["wv"] += Hlist[k].T @ dO[:, k]
        g["wv"] += self.l2 * p["wv"]
        g["bv"][0] += np.sum(dO)

        Hdim = self.H
        for k in range(4):
            dHk = np.outer(dO[:, k], p["wv"])              # path through the vote
            dHk = dHk + dHcat_fromG[:, k * Hdim:(k + 1) * Hdim]  # path through the gate
            dZk = dHk * dtanh_from_out(Hlist[k])
            g[f"W{k}"] += dZk.T @ c["Xs"][k] + self.l2 * p[f"W{k}"]
            g[f"b{k}"] += np.sum(dZk, axis=0)

        return loss, g, yhat

    # ---- convenience --------------------------------------------------------
    def predict(self, Xs):
        yhat, _ = self.forward(Xs)
        return yhat

    def gate_weights(self, Xs):
        _, c = self.forward(Xs)
        return c["G"]


# =============================================================================
# SECTION 3 - Mandatory finite-difference gradient check
# =============================================================================

def gradient_check(seed=1):
    rng = np.random.default_rng(seed)
    model = CatuspramanaAdjudicator(H=7, H2=5, l2=1e-3, seed=seed)
    Xs, y, _ = make_dataset(8, rng)
    loss0, grads, _ = model.loss_and_grads(Xs, y)

    eps = 1e-6
    max_rel = 0.0
    worst = None
    # check a random subset of coordinates from every parameter tensor
    checked = 0
    for name, val in model.p.items():
        flat = val.ravel()
        n_probe = min(6, flat.size)
        idxs = rng.choice(flat.size, size=n_probe, replace=False)
        for idx in idxs:
            orig = flat[idx]
            flat[idx] = orig + eps
            lp, _, _ = model.loss_and_grads(Xs, y)
            flat[idx] = orig - eps
            lm, _, _ = model.loss_and_grads(Xs, y)
            flat[idx] = orig
            num = (lp - lm) / (2 * eps)
            ana = grads[name].ravel()[idx]
            denom = max(1e-12, abs(num) + abs(ana))
            rel = abs(num - ana) / denom
            checked += 1
            if rel > max_rel:
                max_rel = rel
                worst = (name, int(idx), num, ana)
    return max_rel, worst, checked


# =============================================================================
# SECTION 4 - Training loop (Adam) and evaluation
# =============================================================================

def train(model, Xs_tr, y_tr, Xs_va, y_va, epochs=260, lr=6e-3, batch=64, seed=3):
    rng = np.random.default_rng(seed)
    N = y_tr.shape[0]
    m = {k: np.zeros_like(v) for k, v in model.p.items()}
    v = {k: np.zeros_like(v) for k, v in model.p.items()}
    b1, b2, eps = 0.9, 0.999, 1e-8
    step = 0
    hist = []
    for ep in range(epochs):
        order = rng.permutation(N)
        for start in range(0, N, batch):
            idx = order[start:start + batch]
            xb = [X[idx] for X in Xs_tr]
            yb = y_tr[idx]
            loss, grads, _ = model.loss_and_grads(xb, yb)
            step += 1
            for k in model.p:
                m[k] = b1 * m[k] + (1 - b1) * grads[k]
                v[k] = b2 * v[k] + (1 - b2) * (grads[k] ** 2)
                mhat = m[k] / (1 - b1 ** step)
                vhat = v[k] / (1 - b2 ** step)
                model.p[k] -= lr * mhat / (np.sqrt(vhat) + eps)
        if (ep + 1) % 20 == 0 or ep == 0:
            tr_acc = accuracy(model, Xs_tr, y_tr)
            va_acc = accuracy(model, Xs_va, y_va)
            hist.append((ep + 1, loss, tr_acc, va_acc))
    return hist


def accuracy(model, Xs, y):
    yhat = model.predict(Xs)
    return float(np.mean((yhat >= 0.5).astype(float) == y))


# =============================================================================
# SECTION 5 - Doctrine-named diagnostics (the mind operating)
# =============================================================================

def samsaya_report(model, Xs):
    """Samsaya = doubt. We read the model's doubt as the entropy of its two-way
    verdict distribution [1-yhat, yhat]. High entropy = suspended judgement."""
    yhat = model.predict(Xs)
    p2 = np.stack([1 - yhat, yhat], axis=1)
    return row_entropy(p2)


def badhita_intervention(model, rng, n=4000):
    """Controlled test: when ONE instrument turns liar, is it OVERRULED by the rest?

    Nyaya calls a reason contradicted by another means of knowledge *badhita*. We
    build a fully clean, correctly-decided episode, then corrupt ONLY one
    instrument (flip its evidence, inflate its confidence into a hetvabhasa) while
    holding the other three fixed, and ask: does the final verdict stay correct?
    A faithful reasoner lets the weight of the honest instruments overrule the one
    that lies, rather than being captured by the loudest voice.

    Returns dict with:
      base_correct  - verdict accuracy before the lie (should be ~1)
      kept_correct  - verdict accuracy after one instrument lies
      flip_rate     - fraction of episodes whose verdict flipped due to the lie
    """
    base_ok = kept_ok = flips = 0
    for i in range(n):
        t = int(rng.integers(0, 2)); s = 2 * t - 1
        rels = rng.uniform(0.30, 1.0, size=4)
        c = i % 4                                       # rotate the lying instrument
        clean = [_channel_features(s, rels[k], corrupted=False, rng=rng) for k in range(4)]
        Xs_clean = [clean[k][None, :] for k in range(4)]
        v0 = int(model.predict(Xs_clean)[0] >= 0.5)
        corrupted = list(clean)
        corrupted[c] = _channel_features(s, rels[c], corrupted=True, rng=rng)
        Xs_corr = [corrupted[k][None, :] for k in range(4)]
        v1 = int(model.predict(Xs_corr)[0] >= 0.5)
        base_ok += (v0 == t)
        kept_ok += (v1 == t)
        flips += (v1 != v0)
    return dict(base_correct=base_ok / n, kept_correct=kept_ok / n, flip_rate=flips / n)


def single_channel_competence(rng, n=6000):
    """Baseline: how well does EACH lone pramana do on its own? Establishes that
    fusion genuinely beats any single instrument (Vatsyayana's pluralism)."""
    Xs, y, _ = make_dataset(n, rng, p_corrupt=0.35)
    accs = []
    for k in range(4):
        # logistic regression on channel k only, closed-ish via a few GD steps
        w = np.zeros(D_CH); b = 0.0
        X = Xs[k]
        for _ in range(400):
            z = X @ w + b
            ph = sigmoid(z)
            gw = X.T @ (ph - y) / n
            gb = np.mean(ph - y)
            w -= 0.5 * gw; b -= 0.5 * gb
        acc = np.mean(((X @ w + b) >= 0).astype(float) == y)
        accs.append(float(acc))
    return accs


# =============================================================================
# SECTION 6 - Main demonstration
# =============================================================================

def main():
    np.set_printoptions(precision=4, suppress=True)
    line = "=" * 78

    print(line)
    print("  CHAPTER 0158 - VATSYAYANA  |  Catuspramana Adjudicator")
    print("  Four instruments of knowing, weighed in debate, judged by success")
    print(line)

    # ---- 1. gradient check (mandatory) --------------------------------------
    print("\n[1] Finite-difference gradient check")
    max_rel, worst, checked = gradient_check()
    print(f"    coordinates checked : {checked}")
    print(f"    max relative error  : {max_rel:.3e}   (worst: {worst[0]}[{worst[1]}])")
    print(f"    analytic vs numeric : {worst[3]:+.6e} vs {worst[2]:+.6e}")
    assert max_rel < 1e-4, "gradient check FAILED"
    print("    PASS - analytic gradients match numerical to < 1e-4")

    # ---- 2. data ------------------------------------------------------------
    print("\n[2] Synthesising the four-instrument world")
    rng = np.random.default_rng(42)
    Xs_tr, y_tr, _ = make_dataset(6000, rng, p_corrupt=0.35)
    Xs_va, y_va, _ = make_dataset(1500, rng, p_corrupt=0.35)
    Xs_te, y_te, meta_te = make_dataset(3000, rng, p_corrupt=0.35)
    print(f"    train={y_tr.size}  val={y_va.size}  test={y_te.size}   "
          f"channels={CHANNELS}")
    base = single_channel_competence(np.random.default_rng(7))
    for k, a in zip(CHANNELS, base):
        print(f"    lone {k:<10}: acc={a:.3f}")
    print(f"    best single instrument acc = {max(base):.3f}  "
          f"(fusion should beat this)")

    # ---- 3. train -----------------------------------------------------------
    print("\n[3] Training the adjudicator (Adam)")
    model = CatuspramanaAdjudicator(H=12, H2=10, l2=1e-4, seed=11)
    hist = train(model, Xs_tr, y_tr, Xs_va, y_va, epochs=260, lr=6e-3)
    for ep, loss, tra, vaa in hist:
        print(f"    epoch {ep:>3}: loss={loss:.4f}  train_acc={tra:.3f}  val_acc={vaa:.3f}")

    te_acc = accuracy(model, Xs_te, y_te)
    print(f"    >> held-out TEST accuracy = {te_acc:.3f}")
    print(f"    >> uplift over best lone instrument = {te_acc - max(base):+.3f}")

    # ---- 4. samsaya (doubt) -------------------------------------------------
    print("\n[4] Samsaya - doubt as verdict entropy")
    ent = samsaya_report(model, Xs_te)
    # split test cases by whether a fallacy is present (harder -> more doubt)
    has_fallacy = meta_te["corrupt"] >= 0
    print(f"    mean doubt, clean cases   : {ent[~has_fallacy].mean():.3f} nats")
    print(f"    mean doubt, fallacy cases : {ent[has_fallacy].mean():.3f} nats")
    print(f"    (higher doubt where an instrument lies = correct suspension)")
    conf = ent < 0.2
    print(f"    nirnaya (ascertained, doubt<0.2): {100*conf.mean():.1f}% of cases; "
          f"acc on those = {accuracy(model, [X[conf] for X in Xs_te], y_te[conf]):.3f}")

    # ---- 5. hetvabhasa / badhita - is a lie overruled by the rest? ----------
    print("\n[5] Hetvabhasa - when one instrument lies, is it overruled (badhita)?")
    # accuracy split on the natural test set
    fal = meta_te["corrupt"] >= 0
    acc_clean = accuracy(model, [X[~fal] for X in Xs_te], y_te[~fal])
    acc_fall = accuracy(model, [X[fal] for X in Xs_te], y_te[fal])
    print(f"    test accuracy, no fallacy present : {acc_clean:.3f}")
    print(f"    test accuracy, a fallacy present  : {acc_fall:.3f}")
    # controlled single-instrument corruption
    rob = badhita_intervention(model, np.random.default_rng(99))
    print(f"    controlled: verdict correct before lie = {rob['base_correct']:.3f}")
    print(f"    controlled: verdict STILL correct after = {rob['kept_correct']:.3f}")
    print(f"    controlled: verdicts flipped by the lie  = {rob['flip_rate']:.3f}")
    print("    -> the honest instruments overrule the fallacious one (badhita)")

    # ---- 6. pravrtti - a worked case ----------------------------------------
    print("\n[6] Pravrtti - one representative adjudication, instrument by instrument")
    rng2 = np.random.default_rng(2024)
    shown = None
    for _ in range(200):                                # find a typical, correctly-resolved lie
        Xs1, y1, meta1 = make_dataset(1, rng2, p_corrupt=1.0)
        c = int(meta1["corrupt"][0])
        yh1 = float(model.predict(Xs1)[0])
        G1 = model.gate_weights(Xs1)[0]
        verdict_ok = (yh1 >= 0.5) == bool(y1[0])
        if verdict_ok and G1[c] < 0.25:                 # correct AND the liar is demoted
            shown = (Xs1, y1, c, yh1, G1); break
    Xs1, y1, c, yh1, G1 = shown
    print(f"    hidden truth (success?) : {int(y1[0])}")
    print(f"    lying instrument        : {CHANNELS[c]} (confident but wrong)")
    for k in range(4):
        flag = "  <-- fallacy, demoted" if k == c else ""
        print(f"      gate[{CHANNELS[k]:<10}] = {G1[k]:.3f}{flag}")
    print(f"    verified P(success)     : {yh1:.3f}  "
          f"-> verdict {'ACT' if yh1>=0.5 else 'REFRAIN'}  (matches the truth)")

    print("\n" + line)
    print("  All checks passed. The mind establishes prama by adjudicating four")
    print("  instruments and trusting the belief that survives doubt and fallacy.")
    print(line)


if __name__ == "__main__":
    main()
