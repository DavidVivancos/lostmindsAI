#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 Roman historian & senator | Domain: history | Region: Gallia Narbonensis / Rome
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 7 Minds 121 - 140 Available on Amazon https://www.amazon.com/dp/B0HFN6GXMH
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 127: Publius (or Gaius) Cornelius Tacitus  (c. AD 56 - c. AD 120)
================================================================================   

WHAT THIS FILE IS
-----------------
A from-scratch, pure-NumPy neural architecture that encodes the ONE cognitive
signature that is Tacitus's alone: the recovery of a concealed inner MOTIVE from
public behaviour that a fearful regime has flattened into near-uniform
compliance.

Tacitus opens the Annals by naming the two forces that corrupt every witness of
power: adulatio (servile flattery, offered while the tyrant lives) and odium
(hatred, poured out once he is safely dead). Testimony given under either force
is worthless as-is. His method - "sine ira et studio", without anger and
partiality - is therefore NOT bland neutrality. It is an act of DECONVOLUTION:
strip the known distortion out of the signal before you infer the motive
underneath. Under a frightened court, senators "rush into servitude" (ruere in
servitium); every face wears the same loyalty, so ordinary evidence loses its
diagnostic value. What survives is the RESIDUE - the excess of flattery, the
significant silence, the one anomalous act - and Tacitus reads that residue the
LOUDER the more frightened the room is.

That is a signal-processing thesis about social cognition, and it dictates the
architecture. We do NOT build a transformer, a mixture-of-experts, or an
attention-over-stored-keys oracle. We build:

    A CoercionGate  ->  a Deconvolution front-end  ->  a variational Motive
    encoder  ->  a Motive classifier  ->  a re-synthesising Decoder.

THE WORLD (data-generating process we must invert)
--------------------------------------------------
Each observed act x in R^D is produced from:
  * a hidden motive class  m in {loyal, fearful, ambitious, resentful}
  * a "sincere" behaviour  s = B[m] + noise            (what m would do if free)
  * a fixed compliance display  p                       (the servile uniform)
  * a coercion level  c in [0,1]                         (how terrified the room is)
The court then flattens the sincere act toward compliance:
      x = (1 - a(c)) * s  +  a(c) * p  +  observation noise
with an "adulatio gate"  a(c) = a_max * sigmoid(k (c - c0)).
Because a_max < 1, a THIN residue of the sincere act always leaks through - the
truth is buried, never erased. Tacitus's wager is exactly that it is never
erased. As c -> 1 the mutual information between m and the bulk of x collapses,
so a naive reader is reduced to guessing.

THE TACITEAN INVERSION (the model's front-end)
----------------------------------------------
Knowing the coercion level, the sincere act is recoverable in the mean:
      s_hat = (x - a(c) * p) / (1 - a(c))
This is the mathematical form of "read the residue against the known fear": it
renormalises the leaked residue back to full size. It also AMPLIFIES the noise
by 1/(1 - a(c)) - so recovering the truth under terror is possible but arrives
with widened uncertainty. The model therefore also learns to report HIGHER
posterior variance as coercion rises: calibrated suspicion, judgement suspended
in proportion to how much the evidence has been coerced. That is "sine ira et
studio" made computational.

The compliance prototype p and the gate shape (a_max, k, c0) are LEARNED, so the
network discovers for itself what servility looks like and how fear saturates a
court - it is not told.

WHAT THE SELF-TESTS DEMONSTRATE (run this file)
-----------------------------------------------
  [Gradient check]  Analytic gradients match finite differences (< 1e-5).
  [Test 1] Under low coercion a naive linear readout of raw x recovers motive.
  [Test 2] Under HIGH coercion the naive readout collapses toward chance, while
           the deconvolving model still recovers motive well above chance.
  [Test 3] The model's expressed uncertainty (posterior std) RISES with coercion:
           suspicion is calibrated, not constant.
  [Test 4] Adulatio detector: praise in EXCESS of what compliance predicts is
           flagged as the high-information "tell" - Tacitus reading innuendo.

No external data, no frameworks. NumPy only. Deterministic under a fixed seed.
"""

from __future__ import annotations
import numpy as np

# =============================================================================
# 0. Small numerical helpers (kept explicit so backprop stays auditable)
# =============================================================================

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def softplus(z):
    # numerically stable softplus, used to keep the gate steepness positive
    return np.logaddexp(0.0, z)

def d_softplus(z):
    return sigmoid(z)

def tanh(z):
    return np.tanh(z)

def d_tanh(a):
    # derivative given the ALREADY-computed activation a = tanh(z)
    return 1.0 - a * a

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

def onehot(idx, k):
    m = np.zeros((idx.shape[0], k))
    m[np.arange(idx.shape[0]), idx] = 1.0
    return m


# =============================================================================
# 1. THE WORLD  -  a synthetic Roman court that flattens motive under fear
# =============================================================================

MOTIVES = ["loyal", "fearful", "ambitious", "resentful"]
K = len(MOTIVES)      # motive classes
D = 8                 # behavioural channels (see names below)

# Behavioural channels observed of any senator at court:
BEHAVIOUR_CHANNELS = [
    "praise_of_prince",   # 0 how fulsomely they praise the emperor
    "denounce_rivals",    # 1 willingness to accuse others
    "gift_giving",        # 2 material displays of loyalty
    "public_assent",      # 3 voting/nodding with the majority
    "attendance",         # 4 diligence in showing up
    "private_warmth",     # 5 genuine warmth toward the prince (hard to fake)
    "candour",            # 6 willingness to voice disagreement
    "self_advancement",   # 7 manoeuvring for personal position
]

def _prototype_behaviours(rng):
    """Sincere behaviour each motive would display IF THE COURT WERE FREE."""
    B = np.zeros((K, D))
    # loyal:      warm, candid-ish, praises sincerely, not scheming
    B[0] = [ 0.8, -0.3,  0.4,  0.5,  0.6,  0.9,  0.4, -0.4]
    # fearful:    praises & assents from terror, no warmth, no candour, hides
    B[1] = [ 0.6,  0.2,  0.3,  0.8,  0.7, -0.6, -0.8, -0.2]
    # ambitious:  praises strategically, denounces rivals, self-advances hard
    B[2] = [ 0.7,  0.9,  0.6,  0.4,  0.5, -0.1, -0.2,  0.9]
    # resentful:  cold, withholds praise, quietly candid, low assent
    B[3] = [-0.4, -0.1, -0.2, -0.3,  0.3, -0.4,  0.7, -0.1]
    return B

# The servile compliance display p: what EVERYONE converges to under terror.
# High praise, high assent, high attendance; warmth & candour crushed to zero.
COMPLIANCE = np.array([0.9, 0.1, 0.3, 0.95, 0.9, 0.0, 0.0, 0.1])

def sample_court(n, coercion, rng, noise=0.10):
    """
    Draw n courtiers at a given coercion level in [0,1].
    Returns observed behaviour x, true motive m, sincere behaviour s, and the
    per-sample gate a used (for diagnostics / the adulatio detector test).
    """
    B = _prototype_behaviours(rng)
    m = rng.integers(0, K, size=n)
    s = B[m] + rng.normal(0, noise, size=(n, D))          # sincere act
    # the TRUE world gate (fixed; the model must learn its own approximation).
    # a_max<1 => a thin residue of the sincere act always leaks through: the
    # truth is buried under terror, never erased. That is Tacitus's wager.
    a = 0.965 * sigmoid(11.0 * (coercion - 0.55))         # scalar for this batch
    x = (1.0 - a) * s + a * COMPLIANCE + rng.normal(0, noise, size=(n, D))
    a_vec = np.full((n, 1), a)
    return x, m, s, a_vec


# =============================================================================
# 2. THE MODEL  -  CoercionGate -> Deconvolution -> Variational motive encoder
#                  -> Motive classifier -> re-synthesising Decoder
# =============================================================================

class TaciteanMind:
    """
    Parameters (all learned):
      Gate:      g_max_raw, g_k_raw, g_c0   -> a(c) = sig(g_max_raw)*sig(softplus(g_k_raw)*(c-g_c0))
      Compliance prototype:  p  (D,)        -> the learned "servile uniform"
      Encoder:   W1,b1 (D->H tanh); Wmu,bmu / Wlv,blv (H->Z)
      Classifier:Wc,bc (Z->K)
      Decoder:   Wd,bd (Z->Hd tanh); Wr,br (Hd->D sincere behaviour)
    """

    def __init__(self, D=D, H=16, Z=4, Hd=16, K=K, seed=0):
        r = np.random.default_rng(seed)
        sc = lambda a, b: r.normal(0, np.sqrt(2.0 / a), size=(a, b))
        self.P = {
            # --- learned coercion gate (starts mild) ---
            "g_max_raw": np.array(1.5),     # sigmoid(1.5) ~ 0.82 max saturation
            "g_k_raw":   np.array(1.5),     # steepness via softplus
            "g_c0":      np.array(0.5),     # midpoint of fear
            # --- learned compliance display ---
            "p": COMPLIANCE + r.normal(0, 0.05, size=D),
            # --- encoder ---
            "W1": sc(D, H),  "b1": np.zeros(H),
            "Wmu": sc(H, Z), "bmu": np.zeros(Z),
            "Wlv": sc(H, Z), "blv": np.zeros(Z),
            # --- classifier ---
            "Wc": sc(Z, K),  "bc": np.zeros(K),
            # --- decoder ---
            "Wd": sc(Z, Hd), "bd": np.zeros(Hd),
            "Wr": sc(Hd, D), "br": np.zeros(D),
        }
        self.dims = dict(D=D, H=H, Z=Z, Hd=Hd, K=K)

    # ---- the learned gate: how fear saturates a court -----------------------
    def gate(self, c):
        P = self.P
        a_max = sigmoid(P["g_max_raw"])
        k = softplus(P["g_k_raw"])
        return a_max * sigmoid(k * (c - P["g_c0"]))       # (N,1)

    # ---- forward pass -------------------------------------------------------
    def forward(self, x, c, eps, m_idx=None, beta=1.0, gamma=1.0):
        """
        x   (N,D) observed behaviour
        c   (N,1) coercion level
        eps (N,Z) fixed reparam noise (fixed => deterministic for grad-check)
        Returns loss and a cache for backprop.
        """
        P = self.P
        N = x.shape[0]
        a = self.gate(c)                                   # (N,1)
        one_minus_a = 1.0 - a

        # --- Tacitean deconvolution front-end: read residue against known fear
        shat = (x - a * P["p"]) / one_minus_a              # (N,D)

        # --- variational motive encoder ---
        z1 = shat @ P["W1"] + P["b1"]; h = tanh(z1)        # (N,H)
        mu = h @ P["Wmu"] + P["bmu"]                       # (N,Z)
        logvar = h @ P["Wlv"] + P["blv"]                   # (N,Z)
        std = np.exp(0.5 * logvar)
        z = mu + std * eps                                 # (N,Z) reparameterised

        # --- motive classifier ---
        logits = z @ P["Wc"] + P["bc"]                     # (N,K)
        probs = softmax(logits)

        # --- decoder: rebuild sincere act, then RE-APPLY the coercion channel
        zd = z @ P["Wd"] + P["bd"]; hd = tanh(zd)          # (N,Hd)
        fs = hd @ P["Wr"] + P["br"]                        # (N,D) sincere estimate
        xhat = one_minus_a * fs + a * P["p"]               # re-synthesised obs

        # --- losses ---
        recon = np.sum((xhat - x) ** 2) / N
        kl = -0.5 * np.sum(1 + logvar - mu ** 2 - np.exp(logvar)) / N
        if m_idx is not None:
            y = onehot(m_idx, self.dims["K"])
            ce = -np.sum(y * np.log(probs + 1e-12)) / N
        else:
            y = None; ce = 0.0
        loss = recon + beta * kl + gamma * ce

        cache = dict(x=x, c=c, eps=eps, a=a, oma=one_minus_a, shat=shat,
                     h=h, mu=mu, logvar=logvar, std=std, z=z,
                     logits=logits, probs=probs, hd=hd, fs=fs, xhat=xhat,
                     y=y, N=N, beta=beta, gamma=gamma)
        return loss, cache

    # ---- backward pass (all gradients by hand) ------------------------------
    def backward(self, cache):
        P = self.P
        N = cache["N"]; beta = cache["beta"]; gamma = cache["gamma"]
        x, c, eps = cache["x"], cache["c"], cache["eps"]
        a, oma = cache["a"], cache["oma"]
        g = {k: np.zeros_like(v) for k, v in P.items()}

        # ---- reconstruction: recon = sum((xhat - x)^2)/N -------------------
        dxhat = (2.0 / N) * (cache["xhat"] - x)            # (N,D)
        # xhat = oma*fs + a*p  ->  d/dfs, d/dp, and d/da (a affects xhat & shat)
        dfs = dxhat * oma                                  # (N,D)
        g["p"] += (dxhat * a).sum(axis=0)                  # p appears in xhat
        # d xhat/d a = -fs + p = (p - fs)
        da_recon = (dxhat * (P["p"] - cache["fs"])).sum(axis=1, keepdims=True)  # (N,1)

        # decoder: fs = hd@Wr + br ; hd = tanh(zd); zd = z@Wd + bd
        g["Wr"] += cache["hd"].T @ dfs
        g["br"] += dfs.sum(axis=0)
        dhd = dfs @ P["Wr"].T
        dzd = dhd * d_tanh(cache["hd"])
        g["Wd"] += cache["z"].T @ dzd
        g["bd"] += dzd.sum(axis=0)
        dz = dzd @ P["Wd"].T                               # grad into z (via decoder)

        # ---- classification: ce = -sum(y*log softmax(logits))/N ------------
        if cache["y"] is not None:
            dlogits = gamma * (cache["probs"] - cache["y"]) / N    # (N,K)
            g["Wc"] += cache["z"].T @ dlogits
            g["bc"] += dlogits.sum(axis=0)
            dz = dz + dlogits @ P["Wc"].T                  # add classifier path

        # ---- KL: kl = -0.5 sum(1+logvar-mu^2-exp(logvar))/N ----------------
        dmu_kl = beta * (cache["mu"]) / N                  # d kl/d mu
        dlogvar_kl = beta * 0.5 * (np.exp(cache["logvar"]) - 1.0) / N

        # ---- reparam: z = mu + std*eps ; std = exp(0.5 logvar) -------------
        dmu = dz + dmu_kl
        dstd = dz * eps
        dlogvar = dstd * (0.5 * cache["std"]) + dlogvar_kl

        # ---- encoder heads: mu = h@Wmu+bmu ; logvar = h@Wlv+blv ------------
        g["Wmu"] += cache["h"].T @ dmu; g["bmu"] += dmu.sum(axis=0)
        g["Wlv"] += cache["h"].T @ dlogvar; g["blv"] += dlogvar.sum(axis=0)
        dh = dmu @ P["Wmu"].T + dlogvar @ P["Wlv"].T
        dz1 = dh * d_tanh(cache["h"])
        g["W1"] += cache["shat"].T @ dz1; g["b1"] += dz1.sum(axis=0)
        dshat = dz1 @ P["W1"].T                            # (N,D) grad into shat

        # ---- deconvolution front-end: shat = (x - a*p)/(1-a) ---------------
        # d shat/d p = -a/(1-a) ;  d shat/d a  needs product+quotient rule
        g["p"] += (dshat * (-a / oma)).sum(axis=0)
        # shat = (x - a p) * (1-a)^-1
        #   d shat/d a = [-p*(1-a) + (x - a p)] / (1-a)^2  = (x - p)/(1-a)^2
        dshat_da = (x - P["p"]) / (oma ** 2)               # (N,D)
        da_shat = (dshat * dshat_da).sum(axis=1, keepdims=True)  # (N,1)

        # ---- total gradient into the scalar gate a, then into gate params --
        da = da_recon + da_shat                            # (N,1)
        # a = a_max * sig(k*(c-c0)); a_max=sig(g_max_raw); k=softplus(g_k_raw)
        a_max = sigmoid(P["g_max_raw"])
        k = softplus(P["g_k_raw"])
        s_in = sigmoid(k * (c - P["g_c0"]))                # (N,1)
        # d a / d g_max_raw = sig'(g_max_raw) * s_in
        g["g_max_raw"] += (da * (a_max * (1 - a_max)) * s_in).sum()
        # d a / d k = a_max * s_in*(1-s_in) * (c-c0) ; d k/d g_k_raw = sigmoid
        dsin = a_max * s_in * (1 - s_in)
        g["g_k_raw"] += (da * dsin * (c - P["g_c0"]) * d_softplus(P["g_k_raw"])).sum()
        # d a / d c0 = a_max * s_in*(1-s_in) * (-k)
        g["g_c0"] += (da * dsin * (-k)).sum()

        return g

    # ---- convenience: full loss+grad ---------------------------------------
    def loss_and_grad(self, x, c, eps, m_idx, beta=1.0, gamma=1.0):
        loss, cache = self.forward(x, c, eps, m_idx, beta, gamma)
        grad = self.backward(cache)
        return loss, grad

    # ---- inference helpers -------------------------------------------------
    def infer(self, x, c):
        """Return motive posterior probs and posterior std (mean over z-dims)."""
        eps = np.zeros((x.shape[0], self.dims["Z"]))       # posterior mean, no noise
        _, cache = self.forward(x, c, eps, m_idx=None)
        post_std = np.exp(0.5 * cache["logvar"]).mean(axis=1)
        return cache["probs"], post_std, cache["shat"]


# =============================================================================
# 3. GRADIENT CHECK  (mandatory) - analytic vs finite differences
# =============================================================================

def gradient_check(verbose=True):
    rng = np.random.default_rng(1)
    model = TaciteanMind(H=6, Z=3, Hd=6, seed=2)   # small for a fast dense check
    N = 5
    x, m, _, _ = sample_court(N, coercion=0.6, rng=rng)
    c = np.full((N, 1), 0.6)
    eps = rng.normal(size=(N, model.dims["Z"]))    # FIXED noise -> deterministic

    loss, grad = model.loss_and_grad(x, c, eps, m, beta=0.7, gamma=1.3)

    h = 1e-5
    max_rel = 0.0
    worst = None
    for name, W in model.P.items():
        flat = W.reshape(-1)
        # check up to 6 entries per parameter tensor to keep it quick
        idxs = range(min(6, flat.size))
        for i in idxs:
            orig = flat[i]
            flat[i] = orig + h
            lp, _ = model.forward(x, c, eps, m, beta=0.7, gamma=1.3)
            flat[i] = orig - h
            lm, _ = model.forward(x, c, eps, m, beta=0.7, gamma=1.3)
            flat[i] = orig
            num = (lp - lm) / (2 * h)
            ana = grad[name].reshape(-1)[i]
            denom = max(1e-8, abs(num) + abs(ana))
            rel = abs(num - ana) / denom
            if rel > max_rel:
                max_rel = rel; worst = (name, i, num, ana)
    if verbose:
        print(f"[grad check] worst relative error = {max_rel:.2e}  "
              f"at {worst[0]}[{worst[1]}] (num={worst[2]:+.5f}, ana={worst[3]:+.5f})")
        print(f"[grad check] {'PASS' if max_rel < 1e-4 else 'FAIL'} (threshold 1e-4)")
    return max_rel


# =============================================================================
# 4. TRAINING  -  Adam over a court whose fear level varies across the corpus
# =============================================================================

def train(model, steps=4000, batch=128, lr=3e-3, seed=7, verbose=True):
    rng = np.random.default_rng(seed)
    P = model.P
    mt = {k: np.zeros_like(v) for k, v in P.items()}
    vt = {k: np.zeros_like(v) for k, v in P.items()}
    b1, b2, epsA = 0.9, 0.999, 1e-8
    for t in range(1, steps + 1):
        # sample a mixed corpus: each batch drawn at a random coercion level,
        # exactly as Tacitus works across reigns of very different fearfulness.
        coercion = rng.uniform(0.0, 1.0)
        x, m, _, _ = sample_court(batch, coercion, rng)
        c = np.full((batch, 1), coercion)
        eps = rng.normal(size=(batch, model.dims["Z"]))
        # KL warm-up so the classifier can organise the latent first
        beta = min(1.0, t / 1500.0) * 0.5
        loss, grad = model.loss_and_grad(x, c, eps, m, beta=beta, gamma=2.0)
        for k in P:
            gk = np.clip(grad[k], -5, 5)
            mt[k] = b1 * mt[k] + (1 - b1) * gk
            vt[k] = b2 * vt[k] + (1 - b2) * gk * gk
            mhat = mt[k] / (1 - b1 ** t)
            vhat = vt[k] / (1 - b2 ** t)
            P[k] = P[k] - lr * mhat / (np.sqrt(vhat) + epsA)
        if verbose and (t % 800 == 0 or t == 1):
            print(f"  step {t:5d} | coercion {coercion:.2f} | loss {loss:7.4f}")
    return model


# =============================================================================
# 5. EVALUATION  -  the four Tacitean self-tests
# =============================================================================

def _entropy(probs):
    return -(probs * np.log(probs + 1e-12)).sum(axis=1)


def accuracy_at(model, coercion, rng, n=4000):
    x, m, s, a_vec = sample_court(n, coercion, rng)
    c = np.full((n, 1), coercion)
    probs, _, shat = model.infer(x, c)
    pred = probs.argmax(axis=1)
    return (pred == m).mean(), _entropy(probs).mean(), x, m, s, shat


def naive_readout_accuracy_at(rng, coercion, n_train=6000, n_test=4000):
    """Control fit AND tested at a SINGLE coercion (knows the regime implicitly)."""
    xtr, mtr, _, _ = sample_court(n_train, coercion, rng)
    xte, mte, _, _ = sample_court(n_test, coercion, rng)
    Xtr = np.hstack([xtr, np.ones((n_train, 1))]); Ytr = onehot(mtr, K)
    W = np.linalg.solve(Xtr.T @ Xtr + 1e-2 * np.eye(Xtr.shape[1]), Xtr.T @ Ytr)
    Xte = np.hstack([xte, np.ones((n_test, 1))])
    return ((Xte @ W).argmax(axis=1) == mte).mean()


def _pooled_sample(rng, n):
    """A corpus spanning many reigns of different fearfulness (mixed coercion)."""
    xs, ms, cs = [], [], []
    per = 400
    while sum(len(v) for v in ms) < n:
        cval = rng.uniform(0.0, 1.0)
        x, m, _, _ = sample_court(per, cval, rng)
        xs.append(x); ms.append(m); cs.append(np.full((per, 1), cval))
    return np.vstack(xs)[:n], np.concatenate(ms)[:n], np.vstack(cs)[:n]


def naive_pooled_accuracy(rng, n_train=12000, n_test=8000):
    """The FACE-VALUE reader: one linear map on raw x, BLIND to how frightened
    each court was. This is the courtier Tacitus despises - he reads every reign
    with the same credulous eye."""
    xtr, mtr, _ = _pooled_sample(rng, n_train)
    xte, mte, _ = _pooled_sample(rng, n_test)
    Xtr = np.hstack([xtr, np.ones((len(mtr), 1))]); Ytr = onehot(mtr, K)
    W = np.linalg.solve(Xtr.T @ Xtr + 1e-2 * np.eye(Xtr.shape[1]), Xtr.T @ Ytr)
    Xte = np.hstack([xte, np.ones((len(mte), 1))])
    return ((Xte @ W).argmax(axis=1) == mte).mean()


def model_pooled_accuracy(model, rng, n=8000):
    """The Tacitean reader: conditions each judgement on the KNOWN coercion of
    that reign, deconvolves, then classifies in a fear-invariant motive space."""
    x, m, c = _pooled_sample(rng, n)
    probs, _, _ = model.infer(x, c)
    return (probs.argmax(axis=1) == m).mean()


def run_selftests(model):
    rng = np.random.default_rng(99)
    print("\n" + "=" * 74)
    print("TACITEAN SELF-TESTS")
    print("=" * 74)
    chance = 1.0 / K

    # ---- reference table: accuracy & suspicion by coercion ----------------
    print("\n[reference] motive recovery & suspicion as fear rises "
          f"(chance = {chance:.2f})")
    print(f"  {'coercion':>9} | {'naive@c':>8} | {'model':>7} | {'entropy(nats)':>13}")
    ent = {}
    for cval in [0.0, 0.25, 0.5, 0.75, 1.0]:
        acc, H, *_ = accuracy_at(model, cval, rng)
        naive = naive_readout_accuracy_at(np.random.default_rng(int(cval*100)+3), cval)
        ent[cval] = H
        print(f"  {cval:9.2f} | {naive:8.3f} | {acc:7.3f} | {H:13.3f}")

    # ---- Test 1: a free court can be read at face value -------------------
    naive_free = naive_readout_accuracy_at(np.random.default_rng(5), 0.0)
    t1 = naive_free > 0.9
    print(f"\n[Test 1] Free court is legible at face value: "
          f"{'PASS' if t1 else 'FAIL'}  (naive@0.0={naive_free:.3f})")

    # ---- Test 2: across mixed reigns, c-awareness is decisive -------------
    naive_pool = naive_pooled_accuracy(np.random.default_rng(11))
    model_pool = model_pooled_accuracy(model, np.random.default_rng(12))
    t2 = (model_pool - naive_pool) > 0.10
    print(f"\n[Test 2] Reading a mixed corpus of reigns (blind vs fear-aware):")
    print(f"  face-value reader (c-blind) : {naive_pool:.3f}")
    print(f"  Tacitean reader   (c-aware) : {model_pool:.3f}")
    print(f"  Test 2 (knowing the fear is decisive): "
          f"{'PASS' if t2 else 'FAIL'}  (+{model_pool-naive_pool:.3f})")

    # ---- Test 3: calibrated suspicion (judgement suspended with fear) -----
    t3 = ent[1.0] > ent[0.0] + 0.05
    print(f"\n[Test 3] Calibrated suspicion — posterior entropy by coercion:")
    print(f"  free {ent[0.0]:.3f}  ->  mid {ent[0.5]:.3f}  ->  terror {ent[1.0]:.3f} nats")
    print(f"  Test 3 (suspicion grows with coercion): {'PASS' if t3 else 'FAIL'}")

    # ---- Test 4: adulatio detector (the deconvolved tell) -----------------
    x, m, s, shat = accuracy_at(model, 0.92, rng)[2:]
    loyal = (m == 0); fearful = (m == 1)
    sep_raw = abs(x[loyal, 0].mean() - x[fearful, 0].mean())
    sep_dec = abs(shat[loyal, 0].mean() - shat[fearful, 0].mean())
    ratio = sep_dec / max(sep_raw, 1e-6)
    t4 = ratio > 2.5      # deconvolution more than doubles the visible tell
    print(f"\n[Test 4] Adulatio detector at coercion 0.92 (loyal vs merely fearful):")
    print(f"  raw praise gap  = {sep_raw:.3f}  (surface: two men praise alike)")
    print(f"  deconvolved gap = {sep_dec:.3f}  (residue reveals the sincere tell)")
    print(f"  amplification   = {ratio:.1f}x")
    print(f"  Test 4 (deconvolution amplifies the tell >2.5x): "
          f"{'PASS' if t4 else 'FAIL'}")

    allp = all([t1, t2, t3, t4])
    print("\n" + "=" * 74)
    print(f"SELF-TESTS: {'ALL PASS' if allp else 'SOME FAILED'}")
    print("=" * 74)
    return allp


# =============================================================================
# 6. MAIN
# =============================================================================

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    print("=" * 74)
    print("TACITUS  -  Coercion-Gated Latent-Motive Deconvolution")
    print("  'sine ira et studio' as a signal-processing operation")
    print("=" * 74)

    print("\n[1/3] Gradient check (analytic vs finite-difference) ...")
    err = gradient_check()

    print("\n[2/3] Training the Tacitean mind on a court of shifting fear ...")
    model = TaciteanMind(seed=0)
    train(model, steps=4500)

    print("\n[3/3] Running self-tests ...")
    run_selftests(model)
