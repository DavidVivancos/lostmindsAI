#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
chapter_0092_chrysippus_-279.py  —  THE HEGEMONIKON
An assent-gated, single-faculty recurrent reasoner after Chrysippus of Soli
(c. 279 - c. 206 BCE), third head of the Stoa.
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 5 Minds 81 - 100 Available on Amazon https://www.amazon.com/dp/B0H7LP5LP2
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 0092 · Chrysippus of Soli
================================================================================

WHY THIS ARCHITECTURE (and why it is NOT a Transformer)
--------------------------------------------------------------------------------
Chrysippus denied Plato's tripartite soul. For him the mind is ONE corporeal
rational faculty, the *hegemonikon* (the "commanding-part"), seated in the
heart. Cognition is a strict pipeline:

        impression (phantasia)  ->  ASSENT (sunkatathesis)  ->  impulse (horme)  -> action

The single move that is genuinely "up to us" (eph' hemin) is ASSENT: the mind
can endorse an impression, withhold endorsement (epoche), or be carried away by
it. Knowledge comes only from assenting to a *cataleptic* impression (phantasia
kataleptike) — one so clear it could only come from what is real. Error and
passion (pathos) come from assenting to *non-cataleptic* impressions.

So this network is not attention over stored keys. It is a control architecture
with exactly one accumulating state (the commanding-faculty) and a learned GATE
that decides, impression by impression, how much to assent. Three things set the
gate, exactly as Chrysippus argued:

  (1) the impression's own CLARITY  (cataleptic vs. non-cataleptic)  -> alpha
  (2) the agent's fixed CHARACTER   (the "cylinder's shape", below)  -> beta
  (3) the faculty's current TENSION (tonos) from prior assents       -> gamma

Point (2) is the cylinder-and-cone analogy (Cicero, *De Fato* 39-43): fate gives
the push (the impression arrives, unbidden), but HOW the thing rolls is fixed by
its own shape. Identical impressions -> different assents, because the character
vector k differs. That is Chrysippus' compatibilism, made mechanical.

Point (3) is the snowball of passion: a faculty already tense from prior
mis-assents is biased to over-assent again. Passions, for Chrysippus, are
"excessive impulses" born of bad evaluative judgments (Galen, PHP 4). The cure is
not to add an irrational part to fight them — there is none — but to correct the
judgment, i.e. to lower assent. We demonstrate exactly this at the end.

THE LEARNING CLAIM
--------------------------------------------------------------------------------
We train the Hegemonikon on a stream of impressions, some cataleptic (reliable
evidence about a hidden truth) and some non-cataleptic (noisy, often deceptive).
Nothing tells the model which is which. It must DISCOVER clarity and learn to
withhold assent from the deceptive ones. We then show, by ablation, that a mind
that assents to everything (gate forced open, a==1) is measurably worse: epoche
— suspension of assent — is what protects the conclusion.

CONVENTIONS (held across the whole 1000Minds corpus)
--------------------------------------------------------------------------------
  * Pure NumPy, from scratch. No autograd, no ML frameworks.
  * Manual backprop through time (BPTT) with a finite-difference gradient check
    that MUST pass (printed below).
  * A real training loop, a held-out test set, and self-tests/demos.
  * Run the file top to bottom: it trains, checks gradients, and prints results.

Run:  python3 chapter_0092_chrysippus_-279.py
================================================================================
"""

import numpy as np

RNG = np.random.default_rng(279)   # seed = year of his birth (279 BCE), for reproducibility


# ============================================================================
# 1. PRIMITIVES
# ============================================================================
def sigmoid(z):
    # Numerically stable logistic. This is the shape of a single assent decision:
    # it saturates toward full assent (1) or full epoche (0).
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)),
                    np.exp(z) / (1.0 + np.exp(z)))

def softmax(z):
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)

def cross_entropy(probs, target_idx):
    return -np.log(probs[target_idx] + 1e-12)


# ============================================================================
# 2. THE MODEL — one corporeal rational faculty, no parts
# ============================================================================
class Hegemonikon:
    """
    A single commanding-faculty that accumulates ONLY what it assents to.

    Parameters (the whole soul is these arrays — nothing is modular):
      W_p, b_p : form an impression (phantasia) from raw sense input
      w_c, b_c : read an impression's CLARITY  -> cataleptic vs non-cataleptic
      k        : the CHARACTER vector (the cylinder's shape; fixed disposition)
      w_h      : read the faculty's current TENSION (tonos) to modulate assent
      alpha    : weight of clarity   in the assent decision
      beta     : weight of character in the assent decision
      gamma    : weight of tension   in the assent decision   (snowball of passion)
      b_a      : assent bias (a temperament: how readily this mind assents at all)
      V        : the impulse map — turns an assented impression into a state increment
      W_o, b_o : the action read-out from the final faculty state
    """

    def __init__(self, d_in, d_hidden, n_classes, scale=0.30):
        s = scale
        self.W_p = RNG.normal(0, s, size=(d_hidden, d_in))
        self.b_p = np.zeros(d_hidden)
        self.w_c = RNG.normal(0, s, size=d_hidden)
        self.b_c = 0.0
        self.k   = RNG.normal(0, s, size=d_hidden)     # the cylinder's shape
        self.w_h = RNG.normal(0, s, size=d_hidden)
        self.alpha = 1.0          # clarity matters from the start
        self.beta  = 0.0
        self.gamma = 0.0
        self.b_a   = 0.0
        self.V   = RNG.normal(0, s, size=(d_hidden, d_hidden))
        self.W_o = RNG.normal(0, s, size=(n_classes, d_hidden))
        self.b_o = np.zeros(n_classes)

        self.d_in, self.d_hidden, self.n_classes = d_in, d_hidden, n_classes

    # ----- parameter (un)packing, used only by the gradient checker -----------
    PNAMES = ["W_p", "b_p", "w_c", "b_c", "k", "w_h",
              "alpha", "beta", "gamma", "b_a", "V", "W_o", "b_o"]

    def get(self, name):
        return getattr(self, name)

    def set(self, name, value):
        setattr(self, name, value)

    # ------------------------------------------------------------------ forward
    def forward(self, X, force_assent=None):
        """
        X : (T, d_in) stream of impressions arriving in time (fate's pushes).
        force_assent : if a float in [0,1], the gate is CLAMPED to that value at
                       every step (used for the 'assent-to-everything' ablation;
                       force_assent=1.0 is a mind with no power of epoche).

        Returns (logits, cache). The cache stores everything BPTT needs.
        """
        T = X.shape[0]
        H = self.d_hidden
        h = np.zeros(H)                      # the faculty starts empty (tonos = 0)
        cache = {"X": X, "steps": [], "force_assent": force_assent}

        for t in range(T):
            x_t = X[t]
            a_pre = self.W_p @ x_t + self.b_p
            p_t = np.tanh(a_pre)             # the formed impression (bounded)

            c_logit = self.w_c @ p_t + self.b_c          # CLARITY logit
            m_t     = self.k  @ p_t                       # CHARACTER alignment
            s_t     = self.w_h @ h                        # TENSION read-out (tonos)

            a_logit = (self.alpha * c_logit
                       + self.beta * m_t
                       + self.gamma * s_t
                       + self.b_a)
            a_gate = sigmoid(a_logit)        # ASSENT in (0,1): 1=endorse, 0=epoche
            if force_assent is not None:
                a_gate_eff = float(force_assent)
            else:
                a_gate_eff = a_gate

            u_t = self.V @ p_t               # candidate impulse from this impression
            h_prev = h
            h = h_prev + a_gate_eff * u_t    # faculty accumulates ONLY what it assents to

            cache["steps"].append(dict(
                x_t=x_t, a_pre=a_pre, p_t=p_t, c_logit=c_logit, m_t=m_t,
                s_t=s_t, a_logit=a_logit, a_gate=a_gate, a_gate_eff=a_gate_eff,
                u_t=u_t, h_prev=h_prev, h=h))

        logits = self.W_o @ h + self.b_o     # the action read-out (horme -> action)
        cache["h_T"] = h
        cache["logits"] = logits
        return logits, cache

    # ------------------------------------------------------------------ loss
    def loss_and_grad(self, X, target_idx):
        """Full forward + manual BPTT. Returns (loss, grads dict, assent trace)."""
        logits, cache = self.forward(X)
        probs = softmax(logits)
        loss = cross_entropy(probs, target_idx)

        # ---- gradient containers ----
        g = {n: np.zeros_like(np.atleast_1d(np.asarray(self.get(n), dtype=float)))
             for n in self.PNAMES}

        # ---- output layer ----
        dy = probs.copy()
        dy[target_idx] -= 1.0                # dL/dlogits
        h_T = cache["h_T"]
        g["W_o"] += np.outer(dy, h_T)
        g["b_o"] += dy
        dh = self.W_o.T @ dy                 # grad wrt final faculty state

        # ---- BPTT over the assent gate (reverse in time) ----
        for st in reversed(cache["steps"]):
            p_t   = st["p_t"]
            u_t   = st["u_t"]
            a_eff = st["a_gate_eff"]
            a     = st["a_gate"]
            forced = cache["force_assent"] is not None

            # h_t = h_prev + a_eff * u_t
            da_eff = float(dh @ u_t)
            du_t   = a_eff * dh
            dh_prev = dh.copy()              # direct skip path to previous state

            # u_t = V p_t
            g["V"] += np.outer(du_t, p_t)
            dp_t = self.V.T @ du_t

            if not forced:
                # a = sigmoid(a_logit);  da_logit = da_eff * a (1-a)
                d_alogit = da_eff * a * (1.0 - a)
                g["alpha"][0] += d_alogit * st["c_logit"]
                g["beta"][0]  += d_alogit * st["m_t"]
                g["gamma"][0] += d_alogit * st["s_t"]
                g["b_a"][0]   += d_alogit

                d_clogit = d_alogit * self.alpha
                d_m      = d_alogit * self.beta
                d_s      = d_alogit * self.gamma

                # s_t = w_h . h_prev
                g["w_h"] += d_s * st["h_prev"]
                dh_prev  += d_s * self.w_h           # tension couples to the PAST

                # m_t = k . p_t
                g["k"] += d_m * p_t
                dp_t   += d_m * self.k

                # c_logit = w_c . p_t + b_c
                g["w_c"] += d_clogit * p_t
                g["b_c"][0] += d_clogit
                dp_t     += d_clogit * self.w_c

            # p_t = tanh(a_pre)
            d_apre = dp_t * (1.0 - p_t * p_t)
            g["W_p"] += np.outer(d_apre, st["x_t"])
            g["b_p"] += d_apre

            dh = dh_prev                      # pass gradient to the previous step

        # squeeze scalar grads back to scalars
        for n in ["b_c", "alpha", "beta", "gamma", "b_a"]:
            g[n] = float(g[n][0])
        return loss, g, np.array([s["a_gate"] for s in cache["steps"]])


# ============================================================================
# 3. THE TASK — assent under non-cataleptic impressions
# ============================================================================
# A hidden truth class y* is fixed for a sequence. Each impression is either
# CATALEPTIC (clear; its content points at y*) or NON-CATALEPTIC (murky; its
# content points at a WRONG class — a deceptive presentation). A noisy "clarity
# channel" hints at which is which, but imperfectly. A mind that assents to
# everything is dragged toward the deceptions; a mind that withholds assent from
# the murky ones recovers the truth.
# ============================================================================
class ImpressionStream:
    def __init__(self, n_classes=4, d_content=6, T=10, p_cataleptic=0.5,
                 clarity_noise=0.6, content_noise=0.35, seed=206):
        self.C = n_classes
        self.d_content = d_content
        self.T = T
        self.p_cat = p_cataleptic
        self.clar_noise = clarity_noise
        self.cont_noise = content_noise
        self.rng = np.random.default_rng(seed)
        # one fixed prototype direction per class (the "real object" each can present)
        P = self.rng.normal(0, 1, size=(self.C, d_content))
        self.proto = P / np.linalg.norm(P, axis=1, keepdims=True)
        self.d_in = d_content + 1            # +1 clarity channel

    def sample(self):
        y = self.rng.integers(self.C)
        X = np.zeros((self.T, self.d_in))
        for t in range(self.T):
            cataleptic = self.rng.random() < self.p_cat
            if cataleptic:
                content = self.proto[y] + self.cont_noise * self.rng.normal(size=self.d_content)
                clarity = 1.0 + self.clar_noise * self.rng.normal()
            else:
                wrong = self.rng.integers(self.C)
                while wrong == y:
                    wrong = self.rng.integers(self.C)
                content = self.proto[wrong] + self.cont_noise * self.rng.normal(size=self.d_content)
                clarity = -1.0 + self.clar_noise * self.rng.normal()
            X[t, :self.d_content] = content
            X[t, self.d_content] = clarity
        return X, y

    def batch(self, n):
        return [self.sample() for _ in range(n)]


# ============================================================================
# 4. GRADIENT CHECK  (mandatory; must pass)
# ============================================================================
def finite_difference_check(model, X, y, eps=1e-6):
    """Compare analytic BPTT grads to central finite differences over EVERY param."""
    loss0, grads, _ = model.loss_and_grad(X, y)
    max_rel = 0.0
    worst = None
    for name in model.PNAMES:
        val = model.get(name)
        arr = np.atleast_1d(np.asarray(val, dtype=float)).copy()
        flat = arr.ravel()
        gflat = np.atleast_1d(np.asarray(grads[name], dtype=float)).ravel()
        # check a handful of coordinates per parameter to keep it fast
        idxs = range(flat.size) if flat.size <= 12 else \
            model_rng_choice(flat.size, 12)
        for i in idxs:
            orig = flat[i]
            flat[i] = orig + eps
            model.set(name, _reshape_like(flat, val))
            lp, _ = model.forward(X)
            lp = cross_entropy(softmax(lp), y)
            flat[i] = orig - eps
            model.set(name, _reshape_like(flat, val))
            lm, _ = model.forward(X)
            lm = cross_entropy(softmax(lm), y)
            flat[i] = orig
            model.set(name, _reshape_like(flat, val))
            num = (lp - lm) / (2 * eps)
            ana = gflat[i]
            denom = max(1e-8, abs(num) + abs(ana))
            rel = abs(num - ana) / denom
            if rel > max_rel:
                max_rel = rel
                worst = (name, i, num, ana)
    return max_rel, worst


def _reshape_like(flat, ref):
    if np.isscalar(ref) or (isinstance(ref, float)):
        return float(flat[0])
    return flat.reshape(np.asarray(ref).shape)


_CHK_RNG = np.random.default_rng(7)
def model_rng_choice(n, k):
    return _CHK_RNG.choice(n, size=k, replace=False)


# ============================================================================
# 5. TRAINING  (plain SGD with momentum, hand-rolled)
# ============================================================================
def evaluate(model, data, force_assent=None):
    correct = 0
    for X, y in data:
        logits, _ = model.forward(X, force_assent=force_assent)
        if int(np.argmax(logits)) == y:
            correct += 1
    return correct / len(data)


def train(model, stream, steps=4000, lr=0.05, momentum=0.9,
          batch=16, report_every=500):
    vel = {n: np.zeros_like(np.atleast_1d(np.asarray(model.get(n), dtype=float)))
           for n in model.PNAMES}
    test = stream.batch(400)
    for step in range(1, steps + 1):
        accum = {n: np.zeros_like(vel[n]) for n in model.PNAMES}
        bloss = 0.0
        for _ in range(batch):
            X, y = stream.sample()
            loss, g, _ = model.loss_and_grad(X, y)
            bloss += loss
            for n in model.PNAMES:
                accum[n] += np.atleast_1d(np.asarray(g[n], dtype=float))
        for n in model.PNAMES:
            grad = accum[n] / batch
            vel[n] = momentum * vel[n] - lr * grad
            new = np.atleast_1d(np.asarray(model.get(n), dtype=float)) + vel[n]
            model.set(n, _reshape_like(new.ravel(), model.get(n)))
        if step % report_every == 0 or step == 1:
            acc = evaluate(model, test[:200])
            print(f"  step {step:5d} | loss {bloss/batch:6.3f} | test-acc {acc:5.3f}")
    return test


# ============================================================================
# 6. MAIN — train, gradient-check, and run the Stoic demonstrations
# ============================================================================
def main():
    print("=" * 74)
    print("THE HEGEMONIKON — assent-gated single-faculty reasoner (Chrysippus)")
    print("=" * 74)

    stream = ImpressionStream(n_classes=4, d_content=6, T=10, p_cataleptic=0.5)
    model = Hegemonikon(d_in=stream.d_in, d_hidden=16, n_classes=stream.C)

    # ---------------- gradient check BEFORE training ----------------
    print("\n[1] Finite-difference gradient check (analytic BPTT vs numeric):")
    Xc, yc = stream.sample()
    max_rel, worst = finite_difference_check(model, Xc, yc)
    print(f"    max relative error = {max_rel:.2e}  (worst: {worst[0]})")
    assert max_rel < 1e-4, "GRADIENT CHECK FAILED"
    print("    PASS — gradients are correct.\n")

    # ---------------- train ----------------
    print("[2] Training the commanding-faculty to withhold assent from murk:")
    test = train(model, stream, steps=4000, lr=0.05, batch=16, report_every=800)

    # ---------------- core claim: epoche beats credulity ----------------
    print("\n[3] The value of epoche (suspension of assent):")
    acc_learned = evaluate(model, test, force_assent=None)
    acc_credulous = evaluate(model, test, force_assent=1.0)   # assents to everything
    print(f"    learned assent gate ........ test-acc {acc_learned:5.3f}")
    print(f"    assents to EVERYTHING (a=1) . test-acc {acc_credulous:5.3f}")
    print(f"    -> withholding assent buys {acc_learned-acc_credulous:+.3f} accuracy.")

    # ---------------- the cylinder: same fate, different constitution ----------
    # Cicero, De Fato 39-43: fate gives the push; HOW the thing rolls is fixed by
    # the cylinder's own shape. We give two minds the SAME impression streams and
    # vary ONLY the character vector k, then count how often their actions diverge.
    print("\n[4] The cylinder-and-cone: identical impressions, different character:")
    base = model.k.copy()
    kA = base + 2.5 * RNG.normal(size=base.shape)
    kB = base - 2.5 * RNG.normal(size=base.shape)
    n_trials, diverge, dasum = 40, 0, 0.0
    example = None
    for _ in range(n_trials):
        Xf, yf = stream.sample()
        model.k = kA
        lA, _ = model.forward(Xf); _, _, aA = model.loss_and_grad(Xf, yf)
        model.k = kB
        lB, _ = model.forward(Xf); _, _, aB = model.loss_and_grad(Xf, yf)
        actA, actB = int(np.argmax(lA)), int(np.argmax(lB))
        dasum += np.mean(np.abs(aA - aB))
        if actA != actB:
            diverge += 1
            if example is None:
                example = (aA.mean(), aB.mean(), actA, actB)
    model.k = base                                  # restore the real character
    print(f"    mean |assent_A - assent_B| per step = {dasum/n_trials:.3f}")
    print(f"    same fate, different action in {diverge}/{n_trials} streams")
    if example:
        print(f"    e.g. assent {example[0]:.2f} -> act {example[2]}  vs  "
              f"assent {example[1]:.2f} -> act {example[3]}")
    print("    Fate supplies the push; the constitution decides the roll.")

    # ---------------- passion = EXCESSIVE IMPULSE, and its cure ----------------
    # Galen, PHP 4: a passion is "an excessive impulse" (horme pleonazousa) born of
    # a mistaken evaluative judgment. Here a loud DECEPTIVE impression arrives. A
    # mind whose temperament assents too readily (large b_a) lets it through as a
    # huge impulse a*||V p||. Stoic therapy = lower the readiness to assent.
    print("\n[5] Passion = excessive impulse; the cure is to assent less:")
    Xp, yp = stream.sample()
    loud = np.zeros(stream.d_in)
    wrong = (yp + 1) % stream.C
    loud[:stream.d_content] = 3.0 * stream.proto[wrong]   # an over-valued presentation
    loud[stream.d_content] = 0.0                          # ambiguous clarity
    Xp[5] = loud

    def impulse_to_loud(b_a_value):
        save = model.b_a
        model.b_a = b_a_value
        _, cache = model.forward(Xp)
        st = cache["steps"][5]
        model.b_a = save
        return st["a_gate"], st["a_gate"] * np.linalg.norm(st["u_t"])

    a_ready, imp_ready = impulse_to_loud(model.b_a + 3.0)   # over-ready temperament
    a_calm,  imp_calm  = impulse_to_loud(model.b_a - 3.0)   # disciplined assent
    print(f"    over-ready mind : assent {a_ready:.3f} -> impulse magnitude {imp_ready:6.3f}  (a passion)")
    print(f"    disciplined mind: assent {a_calm:.3f} -> impulse magnitude {imp_calm:6.3f}  (apatheia)")
    print("    No second 'irrational part' fights the passion — the one faculty,")
    print("    trained to withhold assent, simply ceases to generate it.")

    print("\n" + "=" * 74)
    print("Done. The mind that is master of its assent is master of itself.")
    print("=" * 74)


if __name__ == "__main__":
    main()
