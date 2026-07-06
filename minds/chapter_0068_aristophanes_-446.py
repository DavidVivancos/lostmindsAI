#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
THE INCONGRUITY-RESOLUTION ENGINE  (figure 68)
A from-scratch, pure-NumPy cognitive architecture modelled on the comic mind of
the Athenian poet of Old Comedy (c. 446 - c. 386 BCE).
 Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
Tome 4 Minds 61 - 80 Available on Amazon https://www.amazon.com/dp/B0H71JC95Q
Resume and Interactive Demos at https://artificiology.com/
Author: David Vivancos · Chapter 0068 · Aristophanes of Athens
================================================================================

WHY THIS ARCHITECTURE IS NOT A TRANSFORMER
------------------------------------------
Almost every modern model is trained to MINIMISE surprise: it predicts the most
probable next token and is rewarded for being right. The comic mind does the
opposite. It is a machine for producing SURPRISE THAT RESOLVES -- a continuation
that violates what the audience expects (the *doxa*, common opinion) while
remaining rigorously coherent with an absurd governing premise (the "Great
Idea": a private peace treaty, a sex-strike to end a war, a city built in the
sky between gods and men). The laugh is the moment a listener's expectation is
broken and a hidden consistency snaps into place. That is the formal content of
the incongruity-resolution theory of humour, and it is the cognitive signature
this engine encodes.

So the objective here is deliberately inverted relative to the usual one:

      comic value  =  INCONGRUITY (distance from the expected)
                       x  COHERENCE (alignment with the premise)

A pure-noise continuation is maximally incongruous but incoherent: not funny.
A conventional continuation is coherent but unsurprising: not funny. Only the
PRODUCT is large, and only when both terms are. The engine learns the move that
maximises that product: "go where the premise points, away from where
convention points."

THREE MIND-SPECIFIC MODULES
---------------------------
1. ExpectationField  (frozen)  -- the *doxa*. A fixed linear map that returns
   what a conventional mind expects next. It is frozen because convention is not
   something the comic poet learns; it is the public backdrop he plays against.

2. IncongruityGenerator (trained) -- the comic intelligence. Given a context and
   an absurd premise it produces a unit-norm continuation that is pulled toward
   the premise and pushed away from the expectation.

3. ParabasisHead (trained) -- METACOGNITION. In Old Comedy the chorus halts the
   action, drops the mask, and addresses the audience directly about the play
   itself (the *parabasis*). Here a small head reads the generator's internal
   state and predicts the comic value of its own output: the engine watches
   itself work and reports on the quality of the joke it just made.

A fourth construct, the AGON, is a forward-time procedure rather than a learned
weight: two opposed premises (Aristophanes' "Stronger" and "Weaker" arguments
from *The Clouds*) are each run through the generator and scored, and the engine
declares the more comically effective line. Fantasy propagation feeds an output
back in as the next context, tracing the "Great Idea" as it unfolds.

MAPPING TO THE E-AGI BAROMETER (see chapter for discussion)
  - Cognitive Processing : the incongruity x coherence objective is non-trivial
        constrained reasoning, not next-step imitation.
  - World Modeling       : ExpectationField is a learned model of "the ordinary."
  - Consciousness        : ParabasisHead is explicit self-monitoring.
  - Creativity           : maximised coherent surprise is operationalised novelty.
  - Language/Emotional/Autonomy/Embodiment : discussed in the chapter; the engine
        is the cognitive core those faculties would wrap.

ENGINEERING CONTRACT (kept identical across the corpus)
  * pure NumPy, no autodiff,
  * a finite-difference gradient check that MUST pass,
  * a real training loop on a real (synthetic) task,
  * self-tests / demonstrations,
  * executed before shipping; verified output pasted into the chapter.
================================================================================
"""

import numpy as np

np.random.seed(68)  # the figure's number, for reproducibility


# ============================================================================
# UTILITIES
# ============================================================================

def normalize(x, axis=-1, eps=1e-12):
    """Project a vector (or batch of row-vectors) onto the unit sphere."""
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, eps)


def unit_norm(x, eps=1e-12):
    """Scalar norm of a 1-D vector, floored away from zero for safe division."""
    return max(float(np.linalg.norm(x)), eps)


def xavier(shape, rng):
    """Glorot-uniform initialisation for a 2-D weight matrix."""
    fan_out, fan_in = shape
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=shape)


# ============================================================================
# 1. THE EXPECTATION FIELD  (frozen "doxa": what the crowd expects next)
# ============================================================================

class ExpectationField:
    """
    A FROZEN linear map  e_raw = W_E @ c , returned as a unit direction.

    This represents conventional opinion -- the predictable continuation an
    ordinary mind would supply for a given context. It is intentionally NOT
    trained: the comic poet does not invent the audience's expectations, he
    inherits them and then subverts them. Freezing it also keeps the comic
    objective honest, because the generator cannot "win" by quietly redefining
    what counts as expected.
    """

    def __init__(self, d, rng):
        # A random but fixed rotation-like matrix. Convention is arbitrary but
        # stable -- exactly the property a satirist relies on.
        self.W_E = xavier((d, d), rng)

    def expected_direction(self, c):
        """Return the unit-norm expected continuation for context c (1-D)."""
        return normalize(self.W_E @ c)


# ============================================================================
# 2. THE INCONGRUITY GENERATOR  (the trained comic intelligence)
#    plus 3. THE PARABASIS HEAD  (metacognitive self-assessment)
# ============================================================================

class IncongruityEngine:
    """
    Forward:
        x  = [context ; premise]                       (2d,)
        z1 = W1 x + b1 ;  h = tanh(z1)                 (H,)   hidden state
        u  = W2 h + b2                                 (d,)   raw continuation
        g  = u / ||u||                                 (d,)   unit continuation
        s  = w_p . h + b_p                             scalar parabasis estimate

    Supervision (the Aristophanic move, computed analytically per example):
        e_hat   = ExpectationField(context)            expected direction
        p_hat   = normalize(premise)                   premise direction
        g_star  = normalize(alpha * p_hat - beta * e_hat)
                  -> the ideal comic continuation: toward premise, away from doxa
        q_star  = coherence(g_star) * incongruity(g_star)   ideal comic value
                  coherence   = <g_star, p_hat>        in [-1, 1]
                  incongruity = 0.5 * ||g_star - e_hat||^2  in [0, 2]

    Loss:
        L_gen = 1 - <g, g_star>          (cosine alignment of output to ideal)
        L_par = 0.5 * (s - q_star)^2     (self-assessment regression)
        L     = L_gen + lam * L_par

    The generator therefore learns to REPRODUCE the controlled-surprise move,
    and the parabasis head learns to PREDICT how comic its own output is.
    """

    def __init__(self, d=16, hidden=32, alpha=1.0, beta=0.8, lam=0.5, seed=68):
        self.d = d
        self.H = hidden
        self.alpha = alpha      # pull toward the premise (the "Great Idea")
        self.beta = beta        # push away from the expected (the doxa)
        self.lam = lam          # weight of the metacognitive loss
        rng = np.random.default_rng(seed)
        self.field = ExpectationField(d, rng)

        # trainable parameters
        self.W1 = xavier((hidden, 2 * d), rng)
        self.b1 = np.zeros(hidden)
        self.W2 = xavier((d, hidden), rng)
        self.b2 = np.zeros(d)
        self.w_p = rng.normal(0, 0.1, size=hidden)   # parabasis projection
        self.b_p = 0.0

    # ---- parameter plumbing (used by the optimiser and the gradient check) --
    def params(self):
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2,
                "b2": self.b2, "w_p": self.w_p, "b_p": self.b_p}

    def set_param(self, name, value):
        if name == "b_p":
            self.b_p = float(value)
        else:
            setattr(self, name, value)

    # ---- analytic supervision target ---------------------------------------
    def targets(self, context, premise):
        """Return (g_star, q_star, e_hat, p_hat) for one example."""
        e_hat = self.field.expected_direction(context)
        p_hat = normalize(premise)
        g_star = normalize(self.alpha * p_hat - self.beta * e_hat)
        coherence = float(g_star @ p_hat)
        incongruity = 0.5 * float(np.sum((g_star - e_hat) ** 2))
        q_star = coherence * incongruity
        return g_star, q_star, e_hat, p_hat

    # ---- forward -----------------------------------------------------------
    def forward(self, context, premise):
        g_star, q_star, e_hat, p_hat = self.targets(context, premise)
        x = np.concatenate([context, premise])
        z1 = self.W1 @ x + self.b1
        h = np.tanh(z1)
        u = self.W2 @ h + self.b2
        n = unit_norm(u)
        g = u / n
        s = float(self.w_p @ h + self.b_p)

        L_gen = 1.0 - float(g @ g_star)
        L_par = 0.5 * (s - q_star) ** 2
        L = L_gen + self.lam * L_par

        cache = dict(x=x, z1=z1, h=h, u=u, n=n, g=g, s=s,
                     g_star=g_star, q_star=q_star, e_hat=e_hat, p_hat=p_hat,
                     L_gen=L_gen, L_par=L_par, L=L)
        return cache

    # ---- backward (hand-derived; verified by finite differences below) -----
    def backward(self, cache):
        g, g_star = cache["g"], cache["g_star"]
        h, x, n = cache["h"], cache["x"], cache["n"]
        s, q_star = cache["s"], cache["q_star"]

        # dL_gen/du through g = u/||u|| :  Jacobian (I - g g^T)/n
        # dL_gen/dg = -g_star  ->  du = (1/n)(I - g g^T)(-g_star)
        proj = g_star - (g @ g_star) * g            # (I - g g^T) g_star
        du = -(1.0 / n) * proj                       # (d,)

        dW2 = np.outer(du, h)
        db2 = du

        # parabasis (metacognition) error
        e_par = self.lam * (s - q_star)              # dL/ds * lam folded in
        dw_p = e_par * h
        db_p = e_par

        # into the hidden state from both heads
        dh = self.W2.T @ du + e_par * self.w_p
        dz1 = dh * (1.0 - h ** 2)                     # tanh'
        dW1 = np.outer(dz1, x)
        db1 = dz1

        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2,
                "w_p": dw_p, "b_p": db_p}

    # ---- batch helpers -----------------------------------------------------
    def loss_and_grads(self, C, P):
        """Mean loss and averaged grads over a batch (rows of C, P)."""
        grads = {k: np.zeros_like(v) for k, v in self.params().items()}
        total = 0.0
        for c, p in zip(C, P):
            cache = self.forward(c, p)
            total += cache["L"]
            g = self.backward(cache)
            for k in grads:
                grads[k] = grads[k] + g[k]
        m = len(C)
        for k in grads:
            grads[k] = grads[k] / m
        return total / m, grads

    def batch_loss(self, C, P):
        return float(np.mean([self.forward(c, p)["L"] for c, p in zip(C, P)]))


# ============================================================================
# DATA: random contexts and absurd premises in concept space
# ============================================================================

def make_dataset(n, d, seed):
    rng = np.random.default_rng(seed)
    C = normalize(rng.normal(size=(n, d)), axis=1)   # contexts on unit sphere
    P = normalize(rng.normal(size=(n, d)), axis=1)   # premises  on unit sphere
    return C, P


# ============================================================================
# GRADIENT CHECK  (mandatory; must pass before anything else runs)
# ============================================================================

def gradient_check(engine, C, P, eps=1e-6):
    print("-" * 72)
    print("FINITE-DIFFERENCE GRADIENT CHECK")
    print("-" * 72)
    _, analytic = engine.loss_and_grads(C, P)
    max_rel = 0.0
    rng = np.random.default_rng(0)
    for name, P_arr in engine.params().items():
        arr = np.atleast_1d(np.array(P_arr, dtype=float))
        flat = arr.ravel()
        # probe a few random coordinates per parameter (full check on small ones)
        idxs = range(flat.size) if flat.size <= 6 else rng.choice(
            flat.size, size=6, replace=False)
        for i in idxs:
            orig = flat[i]
            flat[i] = orig + eps
            engine.set_param(name, arr.reshape(np.shape(P_arr)) if np.ndim(P_arr) else flat[i])
            lp = engine.batch_loss(C, P)
            flat[i] = orig - eps
            engine.set_param(name, arr.reshape(np.shape(P_arr)) if np.ndim(P_arr) else flat[i])
            lm = engine.batch_loss(C, P)
            flat[i] = orig
            engine.set_param(name, arr.reshape(np.shape(P_arr)) if np.ndim(P_arr) else flat[i])

            num = (lp - lm) / (2 * eps)
            ana = np.atleast_1d(analytic[name]).ravel()[i]
            denom = max(1e-12, abs(num) + abs(ana))
            rel = abs(num - ana) / denom
            max_rel = max(max_rel, rel)
        print(f"  {name:>4}: checked  max-rel-err so far = {max_rel:.2e}")
    ok = max_rel < 1e-5
    print(f"\n  overall max relative error = {max_rel:.3e}  ->  "
          f"{'PASS' if ok else 'FAIL'}")
    assert ok, "gradient check FAILED"
    return max_rel


# ============================================================================
# OPTIMISER: Adam
# ============================================================================

class Adam:
    def __init__(self, params, lr=2e-2, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(np.atleast_1d(v).astype(float))
                  for k, v in params.items()}
        self.v = {k: np.zeros_like(np.atleast_1d(v).astype(float))
                  for k, v in params.items()}
        self.t = 0

    def step(self, engine, grads):
        self.t += 1
        for k in grads:
            g = np.atleast_1d(grads[k]).astype(float)
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            upd = self.lr * mhat / (np.sqrt(vhat) + self.eps)
            cur = np.atleast_1d(np.array(engine.params()[k], dtype=float))
            new = cur - upd
            if k == "b_p":
                engine.set_param(k, float(new[0]))
            else:
                engine.set_param(k, new.reshape(np.shape(engine.params()[k])))


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train(engine, C, P, epochs=400, batch=64, lr=2e-2, seed=1):
    opt = Adam(engine.params(), lr=lr)
    rng = np.random.default_rng(seed)
    n = len(C)
    hist = []
    for ep in range(epochs):
        order = rng.permutation(n)
        for i in range(0, n, batch):
            idx = order[i:i + batch]
            _, grads = engine.loss_and_grads(C[idx], P[idx])
            opt.step(engine, grads)
        if ep % 40 == 0 or ep == epochs - 1:
            L = engine.batch_loss(C, P)
            align, corr = evaluate(engine, C, P)
            hist.append((ep, L, align, corr))
            print(f"  epoch {ep:4d} | loss {L:.4f} | "
                  f"mean cos(g,g*) {align:.4f} | parabasis r {corr:.4f}")
    return hist


def evaluate(engine, C, P):
    """Mean alignment of output to ideal, and correlation of the parabasis
    self-estimate with the true comic value (its metacognitive accuracy)."""
    aligns, s_pred, q_true = [], [], []
    for c, p in zip(C, P):
        cache = engine.forward(c, p)
        aligns.append(cache["g"] @ cache["g_star"])
        s_pred.append(cache["s"])
        q_true.append(cache["q_star"])
    s_pred, q_true = np.array(s_pred), np.array(q_true)
    if s_pred.std() < 1e-9 or q_true.std() < 1e-9:
        corr = 0.0
    else:
        corr = float(np.corrcoef(s_pred, q_true)[0, 1])
    return float(np.mean(aligns)), corr


# ============================================================================
# DEMONSTRATIONS OF THE COMIC FACULTIES
# ============================================================================

def comic_value(engine, context, premise):
    """The product objective for the engine's actual output: how funny is it?"""
    cache = engine.forward(context, premise)
    g, e_hat, p_hat = cache["g"], cache["e_hat"], cache["p_hat"]
    coherence = float(g @ p_hat)
    incongruity = 0.5 * float(np.sum((g - e_hat) ** 2))
    return coherence * incongruity, coherence, incongruity, cache


def demo_agon(engine, d, seed=7):
    """THE AGON (The Clouds): a Stronger and a Weaker premise are pitted against
    each other over the same context; the engine declares the funnier line."""
    print("-" * 72)
    print("AGON  -- two opposed 'Great Ideas' contend over one context")
    print("-" * 72)
    rng = np.random.default_rng(seed)
    context = normalize(rng.normal(size=d))
    stronger = normalize(rng.normal(size=d))         # the respectable argument
    weaker = normalize(-stronger + 0.35 * rng.normal(size=d))  # its inversion
    cv_s, coh_s, inc_s, _ = comic_value(engine, context, stronger)
    cv_w, coh_w, inc_w, _ = comic_value(engine, context, weaker)
    print(f"  Stronger argument : comic={cv_s:+.3f}  "
          f"(coherence {coh_s:+.3f}, incongruity {inc_s:.3f})")
    print(f"  Weaker   argument : comic={cv_w:+.3f}  "
          f"(coherence {coh_w:+.3f}, incongruity {inc_w:.3f})")
    winner = "Stronger" if cv_s >= cv_w else "Weaker"
    print(f"  -> the engine awards the scene to the {winner} argument.")


def demo_fantasy(engine, d, steps=5, seed=11):
    """FANTASY PROPAGATION (the 'Great Idea'): feed each output back as the next
    context and watch the chain stay coherent with the premise while remaining
    far from ordinary expectation -- a counterfactual reasoned to its end."""
    print("-" * 72)
    print("FANTASY PROPAGATION -- one absurd premise, carried to its conclusion")
    print("-" * 72)
    rng = np.random.default_rng(seed)
    premise = normalize(rng.normal(size=d))
    context = normalize(rng.normal(size=d))
    print("  step |  cos(out, premise)  cos(out, expected)   comic")
    for t in range(steps):
        cv, coh, inc, cache = comic_value(engine, context, premise)
        cos_exp = float(cache["g"] @ cache["e_hat"])
        print(f"   {t:2d}  |     {coh:+.3f}            {cos_exp:+.3f}        {cv:+.3f}")
        context = cache["g"]   # the world has moved; the premise stays fixed


def demo_parabasis(engine, C, P):
    """PARABASIS (metacognition): the engine reports a self-estimate of comic
    value; here are a few cases beside the truth, plus held-out correlation."""
    print("-" * 72)
    print("PARABASIS -- the engine assesses its own jokes (self-monitoring)")
    print("-" * 72)
    print("  example |  self-estimate  |  true comic value")
    for k in range(5):
        cache = engine.forward(C[k], P[k])
        print(f"     {k:2d}    |     {cache['s']:+.3f}     |      "
              f"{cache['q_star']:+.3f}")
    _, corr = evaluate(engine, C, P)
    print(f"  metacognitive correlation on this set: r = {corr:.3f}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 72)
    print("THE INCONGRUITY-RESOLUTION ENGINE  (figure 68)")
    print("comedy as controlled, coherent violation of expectation")
    print("=" * 72)

    d, hidden = 16, 32
    engine = IncongruityEngine(d=d, hidden=hidden, alpha=1.0, beta=0.8,
                               lam=0.9, seed=68)

    # small batch for the gradient check (kept tiny for speed and clarity)
    Cg, Pg = make_dataset(8, d, seed=123)
    gradient_check(engine, Cg, Pg)

    # full training set
    C_tr, P_tr = make_dataset(2000, d, seed=2)
    C_te, P_te = make_dataset(400, d, seed=99)

    print("\n" + "-" * 72)
    print("TRAINING  (objective: reproduce the coherent-surprise move)")
    print("-" * 72)
    train(engine, C_tr, P_tr, epochs=400, batch=64, lr=2e-2)

    print("\nHELD-OUT EVALUATION")
    align, corr = evaluate(engine, C_te, P_te)
    print(f"  mean cos(output, ideal comic move) = {align:.4f}")
    print(f"  parabasis self-assessment corr r   = {corr:.4f}")

    print()
    demo_agon(engine, d)
    print()
    demo_fantasy(engine, d)
    print()
    demo_parabasis(engine, C_te, P_te)

    print("\n" + "=" * 72)
    print("DONE. The engine learned to make surprise resolve, and to know it.")
    print("=" * 72)


if __name__ == "__main__":
    main()
