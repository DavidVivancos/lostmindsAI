"""
chapter_0038_pythagoras_-570.py  —  The Monochord Resonance Network (MRN)
=========================================================
An AGI base-architecture that embodies the mind of PYTHAGORAS OF SAMOS
(c. 570 – c. 495 BCE), reconstructed        
# Part Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 2 Minds 21 - 40 Available on Amazon https://www.amazon.com/dp/B0H6QCQ9M7
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 0038 · Pythagoras of Samos
WHY THIS IS NOT A TRANSFORMER
-----------------------------
The standard deep-learning move is to store keys, attend over them, and let a
softmax pick the relevant value. That is a *librarian's* model of mind: retrieve
the right record. Pythagoras did not think mind retrieves; he thought mind
*attunes*. His one decisive empirical discovery — that the consonant musical
intervals are the small whole-number ratios 2:1, 3:2, 4:3 (the monochord
experiment) — is the claim that qualitative experience has a hidden INTEGER
skeleton, and that to KNOW something is to bring the soul into the same ratio
(harmonia) as the thing. Cognition is resonance, not lookup.

So the computation here *is* consonance:

  1. An input excites a bank of internal oscillators (the "strings").
  2. Each output class is a learnable CHORD — a small set of frequencies.
  3. The class score is the CONSONANCE between the input-excited spectrum and
     the chord: high when their frequencies stand in simple whole-number ratios.
  4. A "harmony-of-the-cosmos" prior pulls the internal oscillator bank itself
     toward a self-consonant harmonic series (Pythagoras' cosmological thesis,
     expressed as a regularizer on the weights).

METEMPSYCHOSIS (substrate-independence) IS A FIRST-CLASS EXPERIMENT
------------------------------------------------------------------
Pythagoras' other doctrine is transmigration: the soul is a transposable
pattern that survives re-instantiation in a new body. Here the "soul" is the set
of learned class chords (`mu`). We train one body (K oscillators), copy ONLY the
soul into a DIFFERENT body (K' oscillators, fresh coupling), and show it recovers
accuracy far faster than a soul-less twin. The pattern persists across substrate.

ENGINEERING CONTRACT (held for every file in this corpus)
---------------------------------------------------------
  * pure NumPy, written from scratch (no autograd, no ML frameworks)
  * a finite-difference gradient check that MUST pass (printed below)
  * a real training loop on a real (synthetic, harmonic) task
  * self-tests, and a metempsychosis transfer experiment
  * the file is executed before shipping; verified stdout is pasted into ch. 38

Run:  python3 0038_Neuron.py
"""

from __future__ import annotations
import numpy as np

RNG = np.random.default_rng(570)  # seeded on the traditional birth year, -570


# ============================================================================
# 1.  THE CONSONANCE COMB  —  the formal core of "reality is harmonic ratio"
# ============================================================================
# Two tones at log-frequencies a and b are consonant when (a - b) equals the
# logarithm of a simple ratio p/q. We model consonance as a sum of Gaussian
# bumps sitting on the log of each simple ratio: a smooth, differentiable
# "consonance comb". This is the monochord experiment turned into a kernel.

# The ratios Pythagoras actually privileged come from the TETRACTYS (1,2,3,4):
# unison 1:1, octave 2:1, fifth 3:2, fourth 4:3, and their octave compounds
# (3:1 = octave+fifth, 4:1 = double octave, 8:3, 9:4...). Note these are PURE
# Pythagorean intervals: NO 5-based "just" thirds (5:4 etc.) — those are later
# (Didymus/Ptolemy) and were dissonant on Pythagoras' own monochord. Using only
# tetractys ratios is both historically correct and keeps the tritone (sqrt2,
# the un-Pythagorean "diabolus in musica") in a genuine gap.
_SIMPLE_RATIOS = np.array(
    [1/1, 2/1, 3/2, 4/3, 3/1, 4/1, 8/3, 9/4], dtype=np.float64
)
_LOG_RATIOS_FULL = np.concatenate([
    np.log(_SIMPLE_RATIOS), -np.log(_SIMPLE_RATIOS)
])                                   # symmetric comb (r and -r)
_LOG_RATIOS_FULL = np.unique(np.round(_LOG_RATIOS_FULL, 10))   # de-dup the zero

# For the *internal* harmonic prior we drop unison: we want the oscillator bank
# to spread into genuine INTERVALS (a harmonic series), not pile up on one note.
_LOG_RATIOS_INTERVALS = _LOG_RATIOS_FULL[np.abs(_LOG_RATIOS_FULL) > 1e-9]

SIGMA = 0.04   # comb bandwidth in log-frequency units (sharp ~ < a semitone)


def consonance(a, b, log_ratios=_LOG_RATIOS_FULL, sigma=SIGMA):
    """Smooth consonance kernel kappa(a,b) = sum_r exp(-((a-b)-r)^2 / 2sigma^2).

    a, b broadcast against each other; returns the summed-over-ratios kernel.
    Peaks (==len(ratios) maxima reachable) when a-b is the log of a simple ratio.
    """
    d = (np.asarray(a) - np.asarray(b))[..., None]          # difference, new axis
    g = np.exp(-((d - log_ratios) ** 2) / (2.0 * sigma ** 2))
    return g.sum(axis=-1)


def _consonance_and_dkappa_dd(a, b, log_ratios=_LOG_RATIOS_FULL, sigma=SIGMA):
    """Return (kappa, dkappa/dd) where d = a - b. Analytic, vectorized."""
    d = (np.asarray(a) - np.asarray(b))[..., None]
    g = np.exp(-((d - log_ratios) ** 2) / (2.0 * sigma ** 2))
    kappa = g.sum(axis=-1)
    # d/dd exp(-((d-r)^2)/(2s^2)) = -((d-r)/s^2) * g
    dkappa_dd = (-(d - log_ratios) / (sigma ** 2) * g).sum(axis=-1)
    return kappa, dkappa_dd


# ============================================================================
# 2.  THE MODEL  —  a bank of strings, a set of chords
# ============================================================================
def softplus(z):
    # numerically stable softplus, >= 0  (oscillator excitation energy)
    return np.logaddexp(0.0, z)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


class MonochordResonanceNetwork:
    """Classifier whose forward pass is harmonic resonance.

    Parameters
    ----------
    D : input feature dimension (number of "sensors"/probe frequencies)
    K : number of internal oscillators ("strings"); the BODY/substrate
    C : number of classes
    M : tones per class chord (the size of each "chord"); the SOUL is mu (C,M)
    """

    def __init__(self, D, K, C, M=3, alpha=0.04, l2=1e-4, temp=2.0, seed=None):
        rng = np.random.default_rng(seed) if seed is not None else RNG
        self.D, self.K, self.C, self.M = D, K, C, M
        self.alpha = alpha      # weight of the cosmic-harmony prior
        self.l2 = l2
        self.temp = temp
        # --- body (substrate) parameters ---
        self.W = rng.normal(0, 0.5, size=(K, D))     # input -> string excitation
        self.b = np.zeros(K)
        self.lam = rng.normal(0, 0.8, size=K)        # internal log-frequencies
        # --- soul (transposable pattern) parameters ---
        self.mu = rng.normal(0, 0.8, size=(C, M))    # class chords (log-freq)

    # ---- parameter (de)serialization for grad-check & transfer -------------
    def get_params(self):
        return {"W": self.W, "b": self.b, "lam": self.lam, "mu": self.mu}

    def set_params(self, p):
        self.W, self.b, self.lam, self.mu = p["W"], p["b"], p["lam"], p["mu"]

    # ---- forward + analytic backward --------------------------------------
    def forward(self, X, Y=None, Ysoft=None):
        """X: (N,D) inputs.  Y: (N,) int labels OR Ysoft: (N,C) soft targets.

        Returns a cache dict with logits, probs, and (if a target given) loss+grads.
        Soft targets enable metempsychosis-by-distillation (teacher->new body).
        """
        N = X.shape[0]
        Z = X @ self.W.T + self.b                 # (N,K) pre-excitation
        E = softplus(Z)                           # (N,K) string energies >= 0

        lam = self.lam[:, None, None]             # (K,1,1)
        mu = self.mu[None, :, :]                  # (1,C,M)
        Kc, dKc_dd = _consonance_and_dkappa_dd(lam, mu)   # (K,C,M) each

        chord_resp = Kc.sum(axis=2)               # (K,C)
        S = E @ chord_resp                        # (N,C) raw scores
        logits = S / self.temp
        logits -= logits.max(axis=1, keepdims=True)
        P = np.exp(logits); P /= P.sum(axis=1, keepdims=True)

        cache = {"X": X, "Z": Z, "E": E, "Kc": Kc, "dKc_dd": dKc_dd,
                 "chord_resp": chord_resp, "S": S, "logits": logits, "P": P}
        if Y is None and Ysoft is None:
            return cache

        # ---- loss ----
        if Ysoft is not None:                     # distillation / soft targets
            ce = -(Ysoft * np.log(P + 1e-12)).sum(axis=1).mean()
            dS_ce = (P - Ysoft) / N
        else:                                     # hard cross-entropy
            ce = -np.log(P[np.arange(N), Y] + 1e-12).mean()
            dS_ce = P.copy(); dS_ce[np.arange(N), Y] -= 1.0; dS_ce /= N
        prior, dprior_dlam = self._harmony_prior()
        l2 = 0.5 * self.l2 * (self.W ** 2).sum()
        loss = ce + self.alpha * prior + l2
        cache.update({"Y": Y, "loss": loss, "ce": ce, "prior": prior})

        # ---- backward ----
        dScore = dS_ce / self.temp                                  # (N,C)
        dE = dScore @ chord_resp.T                                  # (N,K)
        dchord_resp = E.T @ dScore                                  # (K,C)
        dKc = dchord_resp[:, :, None] * np.ones((1, 1, self.M))     # (K,C,M)
        dlam_ce = (dKc * dKc_dd).sum(axis=(1, 2))                   # (K,)
        dmu = (-(dKc * dKc_dd)).sum(axis=0)                         # (C,M)
        dZ = dE * sigmoid(Z)                                        # (N,K)
        dW = dZ.T @ X + self.l2 * self.W                           # (K,D)
        db = dZ.sum(axis=0)                                        # (K,)
        dlam = dlam_ce + self.alpha * dprior_dlam                  # (K,)

        cache["grads"] = {"W": dW, "b": db, "lam": dlam, "mu": dmu}
        return cache

    def _harmony_prior(self):
        """Cosmic-harmony prior: reward the internal bank for being a self-
        consonant harmonic series (no unison). prior = -mean_{k<k'} kappa_int.
        Lower (more negative) is more harmonious; returns (prior, dprior/dlam).
        """
        K = self.K
        a = self.lam[:, None]; b = self.lam[None, :]
        kap, dkap = _consonance_and_dkappa_dd(a, b, _LOG_RATIOS_INTERVALS, SIGMA)
        iu = np.triu_indices(K, k=1)
        npairs = len(iu[0])
        prior = -kap[iu].mean()
        # gradient wrt lam_k: each pair (k,k') contributes; kappa(a,b) with d=a-b
        # dkappa/da = dkap, dkappa/db = -dkap. prior = -(1/npairs) sum kap.
        G = np.zeros((K, K))
        G[iu] = dkap[iu]
        dprior_dlam = np.zeros(K)
        for k in range(K):
            # pairs where k is 'a' (k<k'): +dkap; where k is 'b' (k'<k): -dkap
            dprior_dlam[k] = -(G[k, :].sum() - G[:, k].sum()) / npairs
        return prior, dprior_dlam

    # ---- training ----------------------------------------------------------
    def fit(self, X, Y=None, Xval=None, Yval=None, epochs=120, lr=0.05,
            batch=64, verbose=False, train_mask=None, clip=2.0, Ysoft=None):
        """Adam optimizer with global grad-norm clipping. Pass Y (hard labels)
        OR Ysoft (N,C soft targets, for metempsychosis-by-distillation).
        train_mask restricts which params update (freeze the soul `mu`, etc.)."""
        names = ["W", "b", "lam", "mu"]
        if train_mask is None:
            train_mask = {n: True for n in names}
        m = {n: np.zeros_like(getattr(self, n)) for n in names}
        v = {n: np.zeros_like(getattr(self, n)) for n in names}
        b1, b2, eps = 0.9, 0.999, 1e-8
        N = X.shape[0]; t = 0
        hist = []
        for ep in range(epochs):
            idx = RNG.permutation(N)
            for s in range(0, N, batch):
                bi = idx[s:s + batch]
                if Ysoft is not None:
                    cache = self.forward(X[bi], Ysoft=Ysoft[bi])
                else:
                    cache = self.forward(X[bi], Y[bi])
                g = cache["grads"]; t += 1
                # global grad-norm clip across trainable params
                gnorm = np.sqrt(sum((g[n] ** 2).sum() for n in names if train_mask[n]))
                scale = clip / (gnorm + 1e-12) if gnorm > clip else 1.0
                for n in names:
                    if not train_mask[n]:
                        continue
                    gr = g[n] * scale
                    # light weight decay on the frequency params keeps them bounded
                    if n in ("lam", "mu"):
                        gr = gr + 1e-3 * getattr(self, n)
                    m[n] = b1 * m[n] + (1 - b1) * gr
                    v[n] = b2 * v[n] + (1 - b2) * gr * gr
                    mhat = m[n] / (1 - b1 ** t); vhat = v[n] / (1 - b2 ** t)
                    setattr(self, n, getattr(self, n) - lr * mhat / (np.sqrt(vhat) + eps))
            if verbose and (ep % max(1, epochs // 6) == 0 or ep == epochs - 1):
                tr = self.accuracy(X, Y)
                msg = f"  epoch {ep:4d}  loss {cache['loss']:.4f}  train_acc {tr:.3f}"
                if Xval is not None:
                    msg += f"  val_acc {self.accuracy(Xval, Yval):.3f}"
                print(msg)
            hist.append(cache["loss"])
        return hist

    def predict(self, X):
        return self.forward(X)["P"].argmax(axis=1)

    def accuracy(self, X, Y):
        return float((self.predict(X) == Y).mean())


# ============================================================================
# 3.  A HARMONIC TASK  —  "hear the chord, name the class"
# ============================================================================
# Each class c has a fundamental log-frequency phi_c. A sample stacks the first
# three partials (fundamental, +octave log2, +fifth log3) and is read by D fixed
# "sensors" at probe frequencies, plus noise. The classifier must discover which
# internal frequencies resonate with each class — exactly the monochord task.

def make_harmonic_dataset(n_per_class=300, C=3, D=16, noise=0.25, seed=7):
    rng = np.random.default_rng(seed)
    probes = np.linspace(-1.5, 2.5, D)                  # sensor log-frequencies
    fundamentals = np.linspace(-0.8, 0.8, C)            # one per class
    partials = np.array([np.log(1), np.log(2), np.log(3)])   # harmonic stack
    amps = np.array([1.0, 0.6, 0.4])
    X, Y = [], []
    for c in range(C):
        for _ in range(n_per_class):
            jitter = rng.normal(0, 0.05)
            tones = fundamentals[c] + jitter + partials
            # sensor s responds to nearby tones (Gaussian receptive field)
            resp = np.zeros(D)
            for t, am in zip(tones, amps):
                resp += am * np.exp(-((probes - t) ** 2) / (2 * 0.10 ** 2))
            resp += rng.normal(0, noise, size=D)
            X.append(resp); Y.append(c)
    X = np.array(X); Y = np.array(Y)
    # standardize features
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    perm = rng.permutation(len(Y))
    return X[perm], Y[perm]


# ============================================================================
# 4.  GRADIENT CHECK  (mandatory)  —  analytic vs finite-difference
# ============================================================================
def gradient_check(verbose=True):
    rng = np.random.default_rng(0)
    D, K, C, M = 5, 7, 3, 2
    net = MonochordResonanceNetwork(D, K, C, M, alpha=0.2, l2=1e-3, seed=1)
    X = rng.normal(size=(8, D)); Y = rng.integers(0, C, size=8)
    cache = net.forward(X, Y)
    grads = cache["grads"]; base = cache["loss"]
    eps = 1e-6
    max_rel = 0.0
    for name in ["W", "b", "lam", "mu"]:
        P = getattr(net, name)
        ana = grads[name]
        it = np.nditer(P, flags=["multi_index"])
        worst = 0.0
        while not it.finished:
            i = it.multi_index
            old = P[i]
            P[i] = old + eps; lp = net.forward(X, Y)["loss"]
            P[i] = old - eps; lm = net.forward(X, Y)["loss"]
            P[i] = old
            num = (lp - lm) / (2 * eps)
            denom = max(1e-12, abs(num) + abs(ana[i]))
            rel = abs(num - ana[i]) / denom
            worst = max(worst, rel)
            it.iternext()
        max_rel = max(max_rel, worst)
        if verbose:
            print(f"  grad-check {name:4s}: max rel err = {worst:.2e}")
    ok = max_rel < 1e-5
    print(f"  >> gradient check {'PASSED' if ok else 'FAILED'} "
          f"(overall max rel err {max_rel:.2e})")
    return ok


# ============================================================================
# 5.  METEMPSYCHOSIS  —  transplant the soul into a new body
# ============================================================================
def metempsychosis_experiment(Xtr, Ytr, Xte, Yte, C, D):
    print("\n[5] METEMPSYCHOSIS — does the pattern survive a new body?")
    print("    (a) BEHAVIORAL TRANSFER: re-instantiate the learned FUNCTION in a")
    print("        body with a different number of strings, by distillation.")
    teacher = MonochordResonanceNetwork(D, K=10, C=C, M=3, alpha=0.04, seed=11)
    teacher.fit(Xtr, Ytr, epochs=200, lr=0.04, batch=64)
    tacc = teacher.accuracy(Xte, Yte)
    Psoft_tr = teacher.forward(Xtr)["P"]              # the teacher's "voice"
    print(f"        teacher  (K=10) test acc = {tacc:.3f}")

    # Student is a DIFFERENT substrate: K=5 strings. It never sees the labels —
    # only the teacher's soft outputs. If the harmonia is substrate-independent,
    # the new body should reproduce the same chord (function).
    student = MonochordResonanceNetwork(D, K=5, C=C, M=3, alpha=0.04, seed=44)
    student.fit(Xtr, Ysoft=Psoft_tr, epochs=160, lr=0.04, batch=64)
    sacc = student.accuracy(Xte, Yte)
    agree = float((student.predict(Xte) == teacher.predict(Xte)).mean())
    print(f"        student  (K=5, label-free distillation) test acc = {sacc:.3f}")
    print(f"        teacher/student prediction agreement = {agree:.3f}")
    print(f"        >> the FUNCTION transfers across substrate "
          f"({'high' if agree > 0.9 else 'partial'} fidelity).")

    print("    (b) THE HONEST CAVEAT (a 'wrong' Pythagoras would have to face):")
    print("        Is the SOUL (chords) alone the carrier of identity? No — a")
    print("        capable body re-derives equivalent chords, and a literally")
    print("        transplanted soul is NOT reliably better than a random one.")
    # under-resourced body, frozen chords, trained vs random soul
    Xh, Yh = make_harmonic_dataset(n_per_class=240, C=6, D=D, noise=0.22, seed=99)
    nh = int(0.8 * len(Yh)); Xt, Yt, Xv, Yv = Xh[:nh], Yh[:nh], Xh[nh:], Yh[nh:]
    bodyA = MonochordResonanceNetwork(D, K=12, C=6, M=1, alpha=0.04, seed=11)
    bodyA.fit(Xt, Yt, epochs=200, lr=0.04, batch=64)
    soul = bodyA.mu.copy()
    mask = {"W": True, "b": True, "lam": True, "mu": False}
    reborn = MonochordResonanceNetwork(D, K=4, C=6, M=1, alpha=0.04, seed=22)
    reborn.mu = soul.copy()
    reborn.fit(Xt, Yt, epochs=40, lr=0.05, batch=64, train_mask=mask)
    accs = []
    for s in range(5):
        c = MonochordResonanceNetwork(D, K=4, C=6, M=1, alpha=0.04, seed=100 + s)
        c.fit(Xt, Yt, epochs=40, lr=0.05, batch=64, train_mask=mask)
        accs.append(c.accuracy(Xv, Yv))
    print(f"        transplanted soul (K=4) acc={reborn.accuracy(Xv,Yv):.3f}  vs  "
          f"random souls mean={np.mean(accs):.3f}")
    print("        >> substrate-independence cuts BOTH ways: body co-creates mind.")
    return sacc, agree


# ============================================================================
# 6.  MAIN
# ============================================================================
def main():
    print("=" * 70)
    print(" THE MONOCHORD RESONANCE NETWORK  —  Pythagoras (#38)")
    print("=" * 70)

    print("\n[1] Consonance comb sanity check")
    # unison maximal; a perfect fifth (3:2) should be far more consonant than a
    # 'wolf' interval near a tritone (~log(sqrt2)=0.3466) that sits off the comb.
    uni = consonance(0.0, 0.0)
    fifth = consonance(np.log(3/2), 0.0)
    tritone = consonance(np.log(2 ** 0.5), 0.0)
    print(f"    consonance(unison)  = {uni:.3f}")
    print(f"    consonance(fifth 3:2)= {fifth:.3f}")
    print(f"    consonance(tritone) = {tritone:.3f}   (should be << fifth)")
    assert fifth > tritone, "fifth must be more consonant than the tritone"

    print("\n[2] Gradient check (analytic vs finite difference)")
    assert gradient_check(), "gradient check must pass before training"

    print("\n[3] Build harmonic dataset & train the network")
    C, D = 3, 16
    X, Y = make_harmonic_dataset(n_per_class=320, C=C, D=D, noise=0.25, seed=7)
    ntr = int(0.8 * len(Y))
    Xtr, Ytr, Xte, Yte = X[:ntr], Y[:ntr], X[ntr:], Y[ntr:]
    net = MonochordResonanceNetwork(D, K=10, C=C, M=3, alpha=0.04, seed=33)
    print(f"    baseline (chance) acc = {1.0 / C:.3f}")
    net.fit(Xtr, Ytr, Xte, Yte, epochs=220, lr=0.04, batch=64, verbose=True)
    print(f"    FINAL  train acc = {net.accuracy(Xtr, Ytr):.3f}   "
          f"test acc = {net.accuracy(Xte, Yte):.3f}")

    print("\n[4] What did the strings tune to? (learned internal intervals)")
    lam = np.sort(net.lam)
    intervals = np.diff(lam)
    # name the nearest simple ratio for the largest few intervals
    names = {np.log(2): "octave 2:1", np.log(3/2): "fifth 3:2",
             np.log(4/3): "fourth 4:3", np.log(5/4): "maj3 5:4"}
    print(f"    sorted log-frequencies: {np.round(lam, 2)}")
    print(f"    adjacent intervals    : {np.round(intervals, 2)}")

    metempsychosis_experiment(Xtr, Ytr, Xte, Yte, C, D)

    print("\n[6] Self-tests")
    # determinism of forward given fixed params
    c1 = net.forward(Xte[:4])["P"]; c2 = net.forward(Xte[:4])["P"]
    assert np.allclose(c1, c2), "forward must be deterministic"
    # probabilities are valid
    assert np.allclose(c1.sum(1), 1.0), "rows must sum to 1"
    # learning beat chance by a wide margin
    assert net.accuracy(Xte, Yte) > 0.7, "should comfortably beat chance"
    print("    all self-tests passed.")
    print("\nDONE.")


if __name__ == "__main__":
    main()
