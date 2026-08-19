"""
========================
THE EQUIPOLLENCE ENGINE  --  an architecture after Sextus Empiricus (c.160-210 CE)
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 7 Minds 121 - 140 Available on Amazon https://www.amazon.com/dp/B0HFN6GXMH
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 139: Sextus Empiricus (c.160-210 CE)
================================================================================   

WHAT THIS FILE IS
-----------------
A from-scratch, pure-NumPy neural architecture whose *mechanism* is a literal
translation of the cognitive signature of the Pyrrhonian sceptic Sextus
Empiricus. It is not a demonstration wrapper around a standard classifier; the
forward pass, the loss geometry, and the training target were all designed so
that the single most Sextan idea -- that for a large class of questions the
correct answer is NO answer -- is built into the machine rather than bolted on.

THE ONE IDEA THAT DRIVES EVERYTHING
-----------------------------------
Sextus' method (Outlines of Pyrrhonism I.8, I.10, I.25-30) is:
    1.  Take any claim about how things REALLY are (the "non-evident").
    2.  Set against it the strongest OPPOSING claim.
    3.  If the two arguments come out equal in force -- *isostheneia*,
        equipollence -- then neither can be asserted; the mind *suspends*
        (*epoche*).
    4.  Tranquillity (*ataraxia*) is NOT aimed at. It arrives, Sextus says,
        "as a shadow follows a body", once one stops grasping for the
        non-evident (PH I.29).

Almost every machine-learning system is trained to always emit a confident
label. That is dogmatism in Sextus' precise sense. This architecture instead:

  * Splits judgement into TWO adversarial advocate sub-networks -- a case FOR
    (the pro-advocate) and a case AGAINST (the con-advocate) -- computed by
    genuinely separate parameters, so that the network *constructs its own
    counter-argument* to every input.

  * Feeds the *tension* between them to an "isosthenic judge" head that emits
    three outcomes: AFFIRM, DENY, and -- as a first-class, trainable target --
    SUSPEND.

  * Adds an *equipollence gate*: a learned Gaussian bump on the SUSPEND logit
    that grows as the two advocates approach a tie (diff -> 0). This is the
    mechanical form of "no more this than that" (ou mallon, PH I.188-191).

  * NEVER optimises for tranquillity. The loss is plain cross-entropy over the
    three outcomes. We then MEASURE calibration/over-confidence after the fact
    and show it falls out on its own -- ataraxia as a byproduct, exactly as
    Sextus insists it must be (against the Academics, who wrongly made
    suspension a goal; PH I.226-235, Encyclopaedia sources).

WHY THIS IS NOT A TRANSFORMER / MoE / ATTENTION MODEL
-----------------------------------------------------
Attention stores keys and retrieves the best-matching value -- a machine for
*committing* to the most salient reading. Sextus' machine does the opposite: it
manufactures a rival reading of equal weight and, at the balance point, refuses
to retrieve anything. The core primitive here is not similarity-retrieval but
*opposition-and-balance*. Hence: two advocates + a balance-reading judge + an
equipollence gate. No attention, no stored key-value memory.

WHAT THE FILE CONTAINS
----------------------
  * EquipollenceEngine        -- the model (init / forward / manual backward).
  * make_claims               -- a synthetic world where SUSPEND is genuinely
                                 the correct answer (equal-and-opposite evidence).
  * finite_difference_check    -- MANDATORY numeric gradient check on every
                                 parameter; must pass before anything else runs.
  * train                     -- a real Adam training loop.
  * the Ten Modes test        -- perspective perturbations (after Aenesidemus,
                                 PH I.35-163): a good sceptic's suspensions are
                                 INVARIANT across shifts of viewpoint.
  * a Dogmatist baseline       -- forced always to commit; we show it makes
                                 confident errors exactly where Sextus predicts.
  * an ataraxia (calibration)  -- over-confidence measured, never trained for.
    read-out

All maths is hand-written NumPy. Run:  python chapter_0139_sextus_empiricus_160.py
=============================================================================
"""

import numpy as np

# A single global generator so every run is reproducible (a sceptic distrusts
# results that cannot be reproduced under the same conditions).
RNG = np.random.default_rng(140)


# ---------------------------------------------------------------------------
# 1. THE SYNTHETIC WORLD  --  where suspension is genuinely correct
# ---------------------------------------------------------------------------
# Each "claim" carries evidence FOR and AGAINST. A hidden leaning tau decides
# the truth. When |tau| is small the evidence is genuinely equipollent -- even
# a perfect observer cannot tell affirm from deny -- so the *right* label is
# SUSPEND (2). This is the crucial design choice: suspension is not a hedge, it
# is the ground truth in the balance zone. We also inject "strong-but-balanced"
# cases (both advocates loud, yet tied) -- the pure isostheneia that Sextus
# prizes: equal and opposite arguments, both forceful.
# ---------------------------------------------------------------------------
FEATURE_DIM = 8          # dimensionality of an "impression" (phainomenon)
N_CLASSES = 3            # AFFIRM=0, DENY=1, SUSPEND=2
SUSPEND_BAND = 0.5       # |tau| below this => genuinely undecidable => SUSPEND


def make_claims(n, rng=RNG):
    """Generate n claims as (X, y).

    X[:,0] = pro-strength, X[:,1] = con-strength, then noisy echoes and
    distractor channels. The leaning tau ~= pro - con; the intensity ~= pro +
    con. A claim can be quiet-and-tied or LOUD-and-tied; both are SUSPEND.
    """
    tau = rng.normal(0.0, 1.0, size=n)             # hidden leaning (non-evident)
    base = rng.uniform(0.3, 1.6, size=n)           # how forceful the arguments are
    noise = lambda: rng.normal(0.0, 0.12, size=n)

    pro = base + 0.5 * tau + noise()               # strength of the case FOR
    con = base - 0.5 * tau + noise()               # strength of the case AGAINST

    X = np.stack([
        pro,                       # 0: direct pro impression
        con,                       # 1: direct con impression
        pro + noise(),             # 2: a second, noisier look at pro
        con + noise(),             # 3: a second, noisier look at con
        base + noise(),            # 4: intensity cue
        rng.normal(0, 1, n),       # 5: distractor
        rng.normal(0, 1, n),       # 6: distractor
        rng.normal(0, 1, n),       # 7: distractor
    ], axis=1).astype(np.float64)

    y = np.where(tau >= SUSPEND_BAND, 0,           # AFFIRM
         np.where(tau <= -SUSPEND_BAND, 1, 2))     # DENY else SUSPEND
    return X, y.astype(np.int64)


# ---------------------------------------------------------------------------
# 2. SMALL MATH HELPERS
# ---------------------------------------------------------------------------
def softmax(z):
    z = z - np.max(z, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)


def tanh(z):
    return np.tanh(z)


# ---------------------------------------------------------------------------
# 3. THE MODEL
# ---------------------------------------------------------------------------
class EquipollenceEngine:
    """
    Forward pass (per example, but vectorised over a batch):

        h      = tanh(Wp x + bp)              # PHAINOMENON: how things appear
        a_pro  = tanh(W1 h + b1)              # the case FOR ...
        s_pro  = v1 . a_pro + c1              #   ... as a scalar strength
        a_con  = tanh(W2 h + b2)              # the case AGAINST ...
        s_con  = v2 . a_con + c2              #   ... as a scalar strength
        diff   = s_pro - s_con                # the LEANING (net argument)
        feat   = [s_pro, s_con, diff]         # what the judge is allowed to see
        zj     = tanh(Wj feat + bj)
        raw    = Vj zj + bo                   # 3 logits: affirm / deny / suspend
        bump   = beta * exp(-diff^2 / (2 sigma^2))   # EQUIPOLLENCE GATE
        logits = raw + [0, 0, bump]           # tie -> push toward SUSPEND
        p      = softmax(logits)

    The two advocates have SEPARATE parameters: the model literally argues both
    sides of itself. The gate encodes "no more this than that": as the two
    cases approach equality the suspend-logit is lifted by a learned amount.
    """

    def __init__(self, hidden=16, adv=12, judge=8, rng=RNG):
        d, H, A, K = FEATURE_DIM, hidden, adv, judge

        def he(shape):
            fan_in = shape[1] if len(shape) == 2 else shape[0]
            return rng.normal(0, np.sqrt(2.0 / fan_in), size=shape)

        self.p = {
            # perception (shared appearance layer)
            "Wp": he((H, d)), "bp": np.zeros(H),
            # pro-advocate
            "W1": he((A, H)), "b1": np.zeros(A), "v1": he((A,)) * 0.3, "c1": 0.0,
            # con-advocate
            "W2": he((A, H)), "b2": np.zeros(A), "v2": he((A,)) * 0.3, "c2": 0.0,
            # isosthenic judge
            "Wj": he((K, 3)), "bj": np.zeros(K), "Vj": he((3, K)), "bo": np.zeros(3),
            # equipollence gate (log_sigma keeps sigma > 0)
            "beta": 1.0, "log_sigma": np.log(0.6),
        }

    # --- forward, caching everything needed for the manual backward pass ----
    def forward(self, X):
        p = self.p
        h = tanh(X @ p["Wp"].T + p["bp"])                     # (N,H)

        a1 = tanh(h @ p["W1"].T + p["b1"])                    # (N,A)
        s_pro = a1 @ p["v1"] + p["c1"]                        # (N,)

        a2 = tanh(h @ p["W2"].T + p["b2"])                    # (N,A)
        s_con = a2 @ p["v2"] + p["c2"]                        # (N,)

        diff = s_pro - s_con                                  # (N,)
        feat = np.stack([s_pro, s_con, diff], axis=1)         # (N,3)

        zj = tanh(feat @ p["Wj"].T + p["bj"])                 # (N,K)
        raw = zj @ p["Vj"].T + p["bo"]                        # (N,3)

        sigma = np.exp(p["log_sigma"])
        E = np.exp(-(diff ** 2) / (2.0 * sigma ** 2))         # (N,)
        bump = p["beta"] * E                                  # (N,)

        logits = raw.copy()
        logits[:, 2] += bump                                  # only the SUSPEND logit
        probs = softmax(logits)

        cache = dict(X=X, h=h, a1=a1, a2=a2, s_pro=s_pro, s_con=s_con,
                     diff=diff, feat=feat, zj=zj, sigma=sigma, E=E,
                     probs=probs)
        return probs, cache

    def loss(self, probs, y):
        """Plain cross-entropy. NOTE: no tranquillity term. Ataraxia is never
        optimised for -- we only measure it afterwards."""
        n = len(y)
        ll = -np.log(probs[np.arange(n), y] + 1e-12)
        return np.mean(ll)

    # --- manual backward pass: gradients for EVERY parameter ----------------
    def backward(self, cache, y):
        p = self.p
        N = len(y)
        probs = cache["probs"]

        # dL/dlogits = softmax - onehot  (mean over batch)
        dlogits = probs.copy()
        dlogits[np.arange(N), y] -= 1.0
        dlogits /= N                                           # (N,3)

        g = {k: np.zeros_like(v) if isinstance(v, np.ndarray) else 0.0
             for k, v in p.items()}

        # --- equipollence gate: the bump only touched suspend logit (idx 2) --
        dbump = dlogits[:, 2]                                  # (N,)
        E, diff, sigma = cache["E"], cache["diff"], cache["sigma"]
        g["beta"] = np.sum(dbump * E)
        dE = dbump * p["beta"]                                 # (N,)
        # E = exp(-diff^2/(2 sigma^2))
        ddiff_gate = dE * E * (-diff / sigma ** 2)             # via diff
        # d log_sigma: dE/dlog_sigma = E * diff^2 / sigma^2
        g["log_sigma"] = np.sum(dE * E * (diff ** 2) / sigma ** 2)

        # --- judge head ------------------------------------------------------
        zj = cache["zj"]
        g["Vj"] = dlogits.T @ zj                               # (3,K)
        g["bo"] = np.sum(dlogits, axis=0)                      # (3,)
        dzj = dlogits @ p["Vj"]                                # (N,K)
        dpre_j = dzj * (1 - zj ** 2)                           # tanh'
        feat = cache["feat"]
        g["Wj"] = dpre_j.T @ feat                              # (K,3)
        g["bj"] = np.sum(dpre_j, axis=0)                       # (K,)
        dfeat = dpre_j @ p["Wj"]                               # (N,3)

        # feat = [s_pro, s_con, diff], diff = s_pro - s_con
        ds_pro = dfeat[:, 0] + dfeat[:, 2] + ddiff_gate
        ds_con = dfeat[:, 1] - dfeat[:, 2] - ddiff_gate

        # --- pro advocate ----------------------------------------------------
        a1, h = cache["a1"], cache["h"]
        g["v1"] = a1.T @ ds_pro                                # (A,)
        g["c1"] = np.sum(ds_pro)
        da1 = np.outer(ds_pro, p["v1"])                        # (N,A)
        dpre1 = da1 * (1 - a1 ** 2)
        g["W1"] = dpre1.T @ h                                  # (A,H)
        g["b1"] = np.sum(dpre1, axis=0)
        dh_1 = dpre1 @ p["W1"]                                 # (N,H)

        # --- con advocate ----------------------------------------------------
        a2 = cache["a2"]
        g["v2"] = a2.T @ ds_con
        g["c2"] = np.sum(ds_con)
        da2 = np.outer(ds_con, p["v2"])
        dpre2 = da2 * (1 - a2 ** 2)
        g["W2"] = dpre2.T @ h
        g["b2"] = np.sum(dpre2, axis=0)
        dh_2 = dpre2 @ p["W2"]

        # --- shared perception ----------------------------------------------
        dh = dh_1 + dh_2
        dpre_h = dh * (1 - h ** 2)
        X = cache["X"]
        g["Wp"] = dpre_h.T @ X                                 # (H,d)
        g["bp"] = np.sum(dpre_h, axis=0)

        return g


# ---------------------------------------------------------------------------
# 4. FINITE-DIFFERENCE GRADIENT CHECK  (mandatory, runs first)
# ---------------------------------------------------------------------------
def finite_difference_check(eps=1e-6, tol=1e-5):
    """Verify every analytic gradient against a central finite difference.
    A sceptic asserts nothing about the backward pass until it has been set
    against its numerical opposite and found equal."""
    model = EquipollenceEngine(hidden=6, adv=5, judge=4, rng=np.random.default_rng(1))
    X, y = make_claims(16, rng=np.random.default_rng(2))

    probs, cache = model.forward(X)
    grads = model.backward(cache, y)

    worst = 0.0
    for name, val in model.p.items():
        arr = np.atleast_1d(np.array(val, dtype=np.float64))
        flat = arr.ravel()
        gflat = np.atleast_1d(np.array(grads[name], dtype=np.float64)).ravel()
        # probe a handful of coordinates per parameter (all, if small)
        idxs = range(flat.size) if flat.size <= 12 else RNG.choice(flat.size, 12, replace=False)
        for i in idxs:
            orig = flat[i]
            flat[i] = orig + eps
            _set(model, name, arr)
            lp = model.loss(model.forward(X)[0], y)
            flat[i] = orig - eps
            _set(model, name, arr)
            lm = model.loss(model.forward(X)[0], y)
            flat[i] = orig
            _set(model, name, arr)
            num = (lp - lm) / (2 * eps)
            ana = gflat[i]
            denom = max(1.0, abs(num) + abs(ana))
            rel = abs(num - ana) / denom
            worst = max(worst, rel)
    return worst, worst < tol


def _set(model, name, arr):
    """Write a probed array back into the parameter dict (scalars stay scalars)."""
    if np.ndim(model.p[name]) == 0:
        model.p[name] = float(arr.ravel()[0])
    else:
        model.p[name] = arr.reshape(np.shape(model.p[name]))


# ---------------------------------------------------------------------------
# 5. TRAINING (Adam, hand-rolled)
# ---------------------------------------------------------------------------
def train(model, steps=1500, batch=128, lr=3e-3):
    m = {k: np.zeros_like(np.atleast_1d(np.array(v, float))) for k, v in model.p.items()}
    v = {k: np.zeros_like(np.atleast_1d(np.array(x, float))) for k, x in model.p.items()}
    b1, b2, eps = 0.9, 0.999, 1e-8
    Xval, yval = make_claims(4000, rng=np.random.default_rng(99))

    for t in range(1, steps + 1):
        X, y = make_claims(batch)
        probs, cache = model.forward(X)
        grads = model.backward(cache, y)
        for k in model.p:
            gk = np.atleast_1d(np.array(grads[k], float))
            m[k] = b1 * m[k] + (1 - b1) * gk
            v[k] = b2 * v[k] + (1 - b2) * gk * gk
            mhat = m[k] / (1 - b1 ** t)
            vhat = v[k] / (1 - b2 ** t)
            upd = lr * mhat / (np.sqrt(vhat) + eps)
            if np.ndim(model.p[k]) == 0:
                model.p[k] = float(model.p[k] - upd.ravel()[0])
            else:
                model.p[k] = model.p[k] - upd.reshape(np.shape(model.p[k]))

        if t % 300 == 0 or t == 1:
            pv, _ = model.forward(Xval)
            acc = np.mean(np.argmax(pv, axis=1) == yval)
            print(f"  step {t:4d}   val_loss {model.loss(pv, yval):.4f}   "
                  f"val_acc {acc:.3f}   beta {model.p['beta']:+.3f}   "
                  f"sigma {np.exp(model.p['log_sigma']):.3f}")
    return model


# ---------------------------------------------------------------------------
# 6. THE TEN MODES  --  perspective invariance test (after Aenesidemus)
# ---------------------------------------------------------------------------
# Sextus' Ten Modes (PH I.35-163) argue that what appears depends on the
# observer -- the animal, the person, the sense, the circumstance, position,
# admixture, quantity, relation, frequency, custom. A genuine sceptic's
# suspension should therefore be STABLE under a shift of viewpoint. We simulate
# a "mode" as a random rescaling/mixing of the sense-channels and check how
# often the SUSPEND verdict survives the change of perspective.
# ---------------------------------------------------------------------------
def apply_mode(X, rng):
    scale = rng.uniform(0.8, 1.25, size=(1, FEATURE_DIM))     # channel gains
    admix = rng.normal(0, 0.06, size=X.shape)                 # slight admixture
    return X * scale + admix


def ten_modes_invariance(model, n=3000):
    X, y = make_claims(n, rng=np.random.default_rng(7))
    base = np.argmax(model.forward(X)[0], axis=1)
    is_susp = base == 2
    stable = np.ones(is_susp.sum(), dtype=bool)
    for _ in range(10):                                       # ten modes
        Xm = apply_mode(X, np.random.default_rng(RNG.integers(1 << 30)))
        vm = np.argmax(model.forward(Xm)[0], axis=1)
        stable &= (vm[is_susp] == 2)
    return stable.mean() if is_susp.any() else float("nan")


# ---------------------------------------------------------------------------
# 7. DOGMATIST BASELINE  --  forced to commit (no suspend allowed)
# ---------------------------------------------------------------------------
# A dogmatist, in Sextus' sense, must always affirm or deny. We take the
# trained sceptic's own pro/con reading but force a 2-way verdict, and count how
# often it commits *confidently and wrongly* precisely in the balance zone that
# the sceptic would suspend on.
# ---------------------------------------------------------------------------
def dogmatist_error_in_balance_zone(model, n=6000):
    X, y = make_claims(n, rng=np.random.default_rng(11))
    probs, cache = model.forward(X)
    diff = cache["diff"]
    # dogmatist verdict: affirm if leaning positive else deny (never suspend)
    dog = np.where(diff >= 0, 0, 1)
    # "true" hidden state in the balance zone is unknowable, so any confident
    # commitment there is, by construction, an over-reach. Count commitments in
    # the genuinely-suspend region.
    in_band = y == 2
    committed = np.ones(n, dtype=bool)                        # dogmatist always commits
    over_reach = np.mean(committed[in_band])                  # = 1.0 by definition
    sceptic = np.argmax(probs, axis=1)
    sceptic_suspends = np.mean(sceptic[in_band] == 2)
    return over_reach, sceptic_suspends, dog  # dog returned for completeness


# ---------------------------------------------------------------------------
# 8. ATARAXIA READ-OUT  --  calibration measured, NEVER trained for
# ---------------------------------------------------------------------------
def expected_calibration_error(model, n=6000, bins=10):
    """Over-confidence read-out. Because the loss contained no tranquillity
    term, a low value here is *emergent* -- ataraxia as a shadow, not a goal."""
    X, y = make_claims(n, rng=np.random.default_rng(13))
    probs = model.forward(X)[0]
    conf = np.max(probs, axis=1)
    pred = np.argmax(probs, axis=1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += (m.mean()) * abs(correct[m].mean() - conf[m].mean())
    return ece


# ---------------------------------------------------------------------------
# 9. MAIN  --  run the whole sceptical examination
# ---------------------------------------------------------------------------
def main():
    print("=" * 74)
    print("THE EQUIPOLLENCE ENGINE  --  Sextus Empiricus (c.160-210 CE)")
    print("=" * 74)

    print("\n[1] Finite-difference gradient check (assert nothing until verified)")
    worst, ok = finite_difference_check()
    print(f"    worst relative error over all parameters: {worst:.2e}")
    print(f"    gradient check: {'PASSED' if ok else 'FAILED'}")
    assert ok, "Gradient check failed -- backward pass is not trustworthy."

    print("\n[2] Training the sceptic (loss = cross-entropy only; no ataraxia term)")
    model = EquipollenceEngine()
    train(model)

    print("\n[3] Where the affirm/deny/suspend verdicts land")
    Xt, yt = make_claims(6000, rng=np.random.default_rng(21))
    pred = np.argmax(model.forward(Xt)[0], axis=1)
    names = ["AFFIRM", "DENY", "SUSPEND"]
    for c in range(3):
        mask = yt == c
        acc = np.mean(pred[mask] == c)
        print(f"    {names[c]:8s}: recall {acc:.3f}   (share of data {mask.mean():.2f})")
    print(f"    overall accuracy: {np.mean(pred == yt):.3f}")

    print("\n[4] Ten Modes: does suspension survive a change of perspective?")
    inv = ten_modes_invariance(model)
    print(f"    suspensions stable across all ten perspective shifts: {inv:.3f}")

    print("\n[5] Dogmatist vs sceptic in the balance zone")
    over, susp, _ = dogmatist_error_in_balance_zone(model)
    print(f"    dogmatist commits in the undecidable zone: {over:.3f} of the time")
    print(f"    sceptic instead suspends there:            {susp:.3f} of the time")

    print("\n[6] Ataraxia read-out (calibration was never in the loss)")
    ece = expected_calibration_error(model)
    print(f"    expected calibration error (lower = calmer): {ece:.4f}")

    print("\n" + "=" * 74)
    print("Verdict: the machine argues both sides of every claim, and where the")
    print("two cases weigh equal it declines to speak. Tranquillity followed.")
    print("=" * 74)


if __name__ == "__main__":
    main()
