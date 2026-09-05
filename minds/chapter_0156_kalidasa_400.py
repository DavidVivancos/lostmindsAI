"""
================================================================================
Chapter 0156_kalidasa_400 - Kalidasa (380-415 CE, Gupta India)
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 8 Minds 141 - 160 Available on Amazon https://www.amazon.com/dp/B0HH8RTCXF
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 0156_kalidasa_400 - Kalidasa (380-415 CE, Gupta India)
================================================================================  
THE DHVANI RESONANCE ENGINE
A from-scratch (pure-NumPy) cognitive architecture that encodes Kalidasa's
distinctive theory of mind: meaning is not *stated* but *evoked*. The poet emits
a deliberately impoverished "suggestion" (vyanjana / dhvani) and the receiver's
own mind (the sahrdaya, the "one whose heart resonates") supplies the rest by
settling into a completed aesthetic meaning (rasa). The bridge that makes any
one thing knowable in terms of another is analogy -- upama -- the figure for
which the Sanskrit tradition crowned him: "Upama Kalidasasya."
--------------------------------------------------------------------------------
WHY THIS IS NOT A TRANSFORMER
The default modern generator maximises information transmitted per token. This
architecture does the opposite: it is trained to MINIMISE the bits it emits
while a fixed-capacity receiver still reconstructs the full meaning. Mastery is
measured not by what the machine says but by what it makes the audience's mind
supply. Four coupled mechanisms embody four of Kalidasa's cognitive signatures:

    (1) UPAMA ENGINE      -- soft structure-mapping attention over a learned
                             bank of poetic "vehicles" (upamana: cloud, moon,
                             lotus, deer, lightning, river). To know the target
                             (tenor / upameya) is to find the vehicle it echoes.

    (2) VYANJANA CHANNEL  -- a hard information bottleneck with an explicit rate
                             penalty. This is dhvani: the *suggested* meaning,
                             a few faint bits rather than an explicit statement.

    (3) SAHRDAYA RECEIVER -- a recurrent attractor that *settles* the faint seed
                             into a full meaning over several steps. Rasa is not
                             transmitted; it crystallises inside the receiver.

    (4) ABHIJNANA MEMORY  -- a token-keyed hetero-associative store. A memory is
                             recalled only through its recognition-token (the
                             signet ring of Shakuntala). Corrupt the token and
                             the memory is *forgotten* (Dushyanta's curse);
                             restore it and recognition returns. A dramatised,
                             measurable model of key-addressed memory and its
                             failure mode.

The differentiable core (1)-(3) is trained end-to-end with hand-written
backpropagation-through-time and validated by a finite-difference gradient
check (mandatory). The abhijnana store (4) is Hebbian (outer-product), the
authentic learning rule for a content-addressable memory, and is exercised as a
self-test rather than by gradient descent.

Everything below runs on NumPy alone. Execute the file to reproduce the gradient
check, the training curve, and the five self-tests.
================================================================================
"""

from __future__ import annotations

import numpy as np

# ----------------------------------------------------------------------------- 
# Reproducibility & metadata
# ----------------------------------------------------------------------------- 
RNG = np.random.default_rng(156)          # seeded on the figure id
FIGURE_ID = 156
FIGURE_NAME = "Kalidasa"
FIGURE_FLORUIT = "c. 380-415 CE"

# The learned bank of poetic vehicles (upamana). Names are illustrative labels
# for interpretability of the trained model; the vectors themselves are learned.
VEHICLE_NAMES = ["cloud", "moon", "lotus", "deer", "lightning", "river"]


# ============================================================================= 
# 1.  SMALL DIFFERENTIABLE PRIMITIVES
#     Written out by hand so every gradient can be checked numerically.
# ============================================================================= 

def softmax(v):
    """Numerically stable softmax over the last axis of a 1-D vector."""
    v = v - np.max(v)
    e = np.exp(v)
    return e / np.sum(e)


def tanh(x):
    return np.tanh(x)


def dtanh(y):
    """Derivative of tanh given its *output* y = tanh(x): 1 - y^2."""
    return 1.0 - y * y


# ============================================================================= 
# 2.  THE DHVANI RESONANCE ENGINE  (the trainable core)
# ============================================================================= 

class DhvaniResonanceEngine:
    """
    Forward path (all vectors are 1-D):

        tenor x  (D,)
        --- UPAMA (structure-mapping attention over vehicle bank V) -----------
        scores = (V @ x) / sqrt(D)                     # affinity to each vehicle
        a      = softmax(scores)                       # the chosen simile
        ctx    = a @ V                                 # the evoked vehicle image
        --- ENCODER ----------------------------------------------------------
        h      = tanh(Wenc @ x + Uenc @ ctx + benc)    # analogical representation
        --- VYANJANA bottleneck (dhvani: emit few bits) ----------------------
        z      = Wz @ h + bz                           # the faint SUGGESTION
        --- SAHRDAYA receiver (settle seed into full meaning) ----------------
        seed   = Wdec @ z                              # suggestion lifts to state
        s_0    = tanh(seed)
        s_t    = tanh(Wrec @ s_{t-1} + seed),  t=1..T  # attractor settling
        x_hat  = Wout @ s_T + bout                     # the RECONSTRUCTED meaning

    Loss = reconstruction  +  rate(dhvani)  +  simile-commitment(entropy)
    """

    def __init__(self, D=12, M=6, H=16, Z=4, S=16, T=3,
                 lambda_rate=0.03, lambda_ent=0.06, beta=2.5, rate_eps=1e-6):
        self.D, self.M, self.H, self.Z, self.S, self.T = D, M, H, Z, S, T
        self.lambda_rate = lambda_rate      # weight of the "emit few bits" penalty
        self.lambda_ent = lambda_ent        # weight of "commit to one simile"
        self.beta = beta                    # attention sharpness (lower entropy)
        self.rate_eps = rate_eps            # smooths |z| -> sqrt(z^2+eps)

        sc = lambda a, b: RNG.standard_normal((a, b)) * np.sqrt(2.0 / (a + b))
        # Upama vehicle bank (M vehicles x D features) -- learned.
        self.V = sc(M, D)
        # Encoder
        self.Wenc = sc(H, D)
        self.Uenc = sc(H, D)
        self.benc = np.zeros(H)
        # Vyanjana bottleneck
        self.Wz = sc(Z, H)
        self.bz = np.zeros(Z)
        # Sahrdaya receiver
        self.Wdec = sc(S, Z)
        self.Wrec = sc(S, S) * 0.5          # modest recurrence -> stable settling
        self.Wout = sc(D, S)
        self.bout = np.zeros(D)

    # -- parameter plumbing (so the gradient checker can perturb everything) --
    def params(self):
        return {
            "V": self.V, "Wenc": self.Wenc, "Uenc": self.Uenc, "benc": self.benc,
            "Wz": self.Wz, "bz": self.bz, "Wdec": self.Wdec, "Wrec": self.Wrec,
            "Wout": self.Wout, "bout": self.bout,
        }

    # ------------------------------------------------------------------ forward
    def forward(self, x, cache=True):
        D = self.D
        scores = self.beta * (self.V @ x)           # (M,)  sharpened affinity
        a = softmax(scores)                         # (M,)  the simile weights
        ctx = a @ self.V                            # (D,)  evoked vehicle image

        pre_h = self.Wenc @ x + self.Uenc @ ctx + self.benc
        h = tanh(pre_h)                             # (H,)

        z = self.Wz @ h + self.bz                   # (Z,)  the suggestion (dhvani)

        seed = self.Wdec @ z                        # (S,)
        states = [tanh(seed)]
        for _ in range(self.T):
            states.append(tanh(self.Wrec @ states[-1] + seed))
        s_final = states[-1]
        x_hat = self.Wout @ s_final + self.bout     # (D,)

        c = None
        if cache:
            c = dict(x=x, scores=scores, a=a, ctx=ctx, pre_h=pre_h, h=h,
                     z=z, seed=seed, states=states, s_final=s_final, x_hat=x_hat)
        return x_hat, c

    # --------------------------------------------------------------------- loss
    def loss(self, x, target):
        x_hat, c = self.forward(x)
        rec = 0.5 * np.mean((x_hat - target) ** 2)
        rate = self.lambda_rate * np.mean(np.sqrt(c["z"] ** 2 + self.rate_eps))
        a = c["a"]
        ent = -np.sum(a * np.log(a + 1e-12))        # entropy of the simile
        ent_term = self.lambda_ent * ent            # minimised -> commit to a simile
        total = rec + rate + ent_term
        return total, c, dict(rec=rec, rate=rate, ent=ent)

    # ----------------------------------------------------------------- backward
    def backward(self, x, target, c):
        """Analytic gradients for the full differentiable core (incl. BPTT)."""
        D, T = self.D, self.T
        g = {k: np.zeros_like(v) for k, v in self.params().items()}

        # ---- reconstruction head ------------------------------------------
        dxhat = (c["x_hat"] - target) / D           # d rec / d x_hat
        g["Wout"] += np.outer(dxhat, c["s_final"])
        g["bout"] += dxhat
        ds = self.Wout.T @ dxhat                    # grad into final state

        # ---- backprop-through-time through the settling receiver -----------
        # seed feeds every step; accumulate its gradient.
        dseed = np.zeros(self.S)
        for t in range(T, 0, -1):
            s_t = c["states"][t]
            dpre = ds * dtanh(s_t)                   # through tanh
            g["Wrec"] += np.outer(dpre, c["states"][t - 1])
            dseed += dpre                            # seed added at each step
            ds = self.Wrec.T @ dpre                  # grad to previous state
        # step 0 : s_0 = tanh(seed)
        dpre0 = ds * dtanh(c["states"][0])
        dseed += dpre0

        # ---- seed = Wdec @ z ----------------------------------------------
        g["Wdec"] += np.outer(dseed, c["z"])
        dz = self.Wdec.T @ dseed

        # ---- rate penalty on z (smooth |z|) --------------------------------
        drate = self.lambda_rate * (c["z"] / (np.sqrt(c["z"] ** 2 + self.rate_eps))) / self.Z
        dz = dz + drate

        # ---- bottleneck z = Wz @ h + bz ------------------------------------
        g["Wz"] += np.outer(dz, c["h"])
        g["bz"] += dz
        dh = self.Wz.T @ dz

        # ---- h = tanh(pre_h) ----------------------------------------------
        dpre_h = dh * dtanh(c["h"])
        g["Wenc"] += np.outer(dpre_h, c["x"])
        g["Uenc"] += np.outer(dpre_h, c["ctx"])
        g["benc"] += dpre_h
        dctx = self.Uenc.T @ dpre_h                 # grad into evoked vehicle image

        # ---- upama attention: ctx = a @ V ; a = softmax(scores) ------------
        # dL/dV has two routes: through ctx (a-weighted) and through scores.
        a = c["a"]
        # route 1: ctx = sum_k a_k V[k]  -> dV[k] += a_k * dctx
        g["V"] += np.outer(a, dctx)
        # grad to attention weights from ctx: da_k = V[k] . dctx
        da_from_ctx = self.V @ dctx                 # (M,)

        # entropy term grad wrt a: d(-sum a log a)/da_k = -(log a_k + 1)
        da_from_ent = self.lambda_ent * (-(np.log(a + 1e-12) + 1.0))
        da = da_from_ctx + da_from_ent

        # softmax jacobian: dscores_k = a_k (da_k - sum_j a_j da_j)
        dscores = a * (da - np.sum(a * da))
        # scores = beta * (V @ x)  -> dV[k] += beta * dscores_k * x
        g["V"] += self.beta * np.outer(dscores, c["x"])

        return g

    # ------------------------------------------------------- one training step
    def step_grads(self, x, target):
        total, c, parts = self.loss(x, target)
        g = self.backward(x, target, c)
        return total, g, parts


# ============================================================================= 
# 3.  ABHIJNANA MEMORY  -- the recognition-token (signet-ring) store
#     Hebbian hetero-associative memory: recall a stored meaning only through
#     its key token. Corrupt the token -> forgetting.  Restore it -> recognition.
# ============================================================================= 

class AbhijnanaMemory:
    def __init__(self, key_dim=64, mem_dim=48):
        self.key_dim = key_dim
        self.mem_dim = mem_dim
        self.W = np.zeros((mem_dim, key_dim))       # association matrix
        self.keys = []                              # kept only to *test* recall

    def _bipolar(self, n):
        return RNG.choice([-1.0, 1.0], size=n)

    def imprint(self, key, memory):
        """Hebbian outer-product imprint: W += memory (x) key^T."""
        self.W += np.outer(memory, key) / self.key_dim

    def recall(self, key):
        """Retrieve the memory associated with a (possibly corrupted) key."""
        return np.sign(self.W @ key)

    @staticmethod
    def corrupt(key, fraction):
        """Flip a `fraction` of the token's bits (loss / damage of the ring)."""
        k = key.copy()
        n_flip = int(round(fraction * k.size))
        idx = RNG.choice(k.size, size=n_flip, replace=False)
        k[idx] *= -1.0
        return k


# ============================================================================= 
# 4.  SYNTHETIC "MEANINGS"
#     Each meaning is a sparse blend of a few latent themes -- so a faint
#     suggestion (few bits) can in principle evoke the whole, and a good simile
#     (one dominant vehicle) is the natural route to compress it.
# ============================================================================= 

def make_theme_bank(D, K):
    B = RNG.standard_normal((K, D))
    B /= np.linalg.norm(B, axis=1, keepdims=True)
    return B


def sample_meaning(B, max_active=2, noise=0.03):
    K = B.shape[0]
    n = RNG.integers(1, max_active + 1)
    themes = RNG.choice(K, size=n, replace=False)
    coeffs = RNG.uniform(0.5, 1.0, size=n)
    x = np.zeros(B.shape[1])
    for th, cf in zip(themes, coeffs):
        x += cf * B[th]
    x += noise * RNG.standard_normal(B.shape[1])
    x /= (np.linalg.norm(x) + 1e-9)
    return x, themes[0]                              # return dominant theme too


# ============================================================================= 
# 5.  GRADIENT CHECK  (mandatory)  --  finite differences vs analytic grads
# ============================================================================= 

def gradient_check(model, x, target, eps=1e-6):
    _, _, _ = model.step_grads(x, target)           # warm
    total, grads, _ = model.step_grads(x, target)
    max_rel = 0.0
    worst = None
    for name, P in model.params().items():
        G = grads[name]
        # sample a handful of entries per parameter (keeps the check fast)
        flat = P.size
        idxs = np.unique(np.linspace(0, flat - 1, num=min(8, flat)).astype(int))
        for fi in idxs:
            i = np.unravel_index(fi, P.shape)
            orig = P[i]
            P[i] = orig + eps
            lp, _, _ = model.loss(x, target)
            P[i] = orig - eps
            lm, _, _ = model.loss(x, target)
            P[i] = orig
            num = (lp - lm) / (2 * eps)
            ana = G[i]
            denom = max(1e-9, abs(num) + abs(ana))
            rel = abs(num - ana) / denom
            if rel > max_rel:
                max_rel = rel
                worst = (name, i, num, ana)
    return max_rel, worst


# ============================================================================= 
# 6.  TRAINING  (Adam, hand-rolled)
# ============================================================================= 

class Adam:
    def __init__(self, params, lr=3e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def update(self, params, grads):
        self.t += 1
        for k in params:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (grads[k] ** 2)
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


def train(model, B, steps=4000, batch=16, report_every=1000):
    opt = Adam(model.params())
    history = []
    for it in range(1, steps + 1):
        gsum = {k: np.zeros_like(v) for k, v in model.params().items()}
        lsum = rsum = ratesum = 0.0
        for _ in range(batch):
            x, _ = sample_meaning(B)
            total, g, parts = model.step_grads(x, x)   # autoencode the meaning
            for k in gsum:
                gsum[k] += g[k] / batch
            lsum += total / batch
            rsum += parts["rec"] / batch
            ratesum += parts["rate"] / batch
        opt.update(model.params(), gsum)
        if it % report_every == 0 or it == 1:
            history.append((it, lsum, rsum, ratesum))
            print(f"  step {it:5d} | loss {lsum:.5f} | recon {rsum:.5f} "
                  f"| rate(dhvani) {ratesum:.5f}")
    return history


# ============================================================================= 
# 7.  SELF-TESTS
# ============================================================================= 

def test_dhvani_rate_distortion(model, B, trials=400):
    """
    Evocation test. Keep only the top-k largest-magnitude bits of the emitted
    suggestion z (zeroing the rest) and measure how well the receiver still
    reconstructs the meaning. If a faint suggestion suffices, the receiver's
    mind is doing the work -- the definition of dhvani.
    """
    Z = model.Z
    errs = {k: [] for k in range(1, Z + 1)}
    base = []
    for _ in range(trials):
        x, _ = sample_meaning(B)
        x_hat_full, c = model.forward(x)
        base.append(np.mean((x_hat_full - x) ** 2))
        z = c["z"]
        order = np.argsort(-np.abs(z))
        for keep in range(1, Z + 1):
            mask = np.zeros_like(z)
            mask[order[:keep]] = 1.0
            zc = z * mask
            seed = model.Wdec @ zc
            s = tanh(seed)
            for _ in range(model.T):
                s = tanh(model.Wrec @ s + seed)
            xh = model.Wout @ s + model.bout
            errs[keep].append(np.mean((xh - x) ** 2))
    print(f"  full-code reconstruction MSE : {np.mean(base):.5f}")
    for keep in range(1, Z + 1):
        print(f"  keep top-{keep} of {Z} bits     : MSE {np.mean(errs[keep]):.5f}")
    return np.mean(base), {k: float(np.mean(v)) for k, v in errs.items()}


def test_upama_commitment(model, B):
    """
    For each latent theme, report the dominant vehicle the model settles on and
    how peaked the simile is (max attention weight). A committed, stable mapping
    theme->vehicle is 'Upama Kalidasasya' in miniature.
    """
    K = B.shape[0]
    print("  theme -> dominant vehicle (peak attention)")
    peaks = []
    mapping = {}
    for th in range(K):
        # present the pure theme as the tenor
        x = B[th] / (np.linalg.norm(B[th]) + 1e-9)
        _, c = model.forward(x)
        a = c["a"]
        k = int(np.argmax(a))
        peaks.append(float(np.max(a)))
        mapping[th] = (VEHICLE_NAMES[k], float(np.max(a)))
        print(f"    theme {th}  ->  {VEHICLE_NAMES[k]:9s}  (a_max={np.max(a):.2f})")
    print(f"  mean simile peakedness       : {np.mean(peaks):.2f} "
          f"(1.0 = fully committed)")
    return float(np.mean(peaks)), mapping


def test_abhijnana(mem, n_pairs=12, levels=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5)):
    """
    The signet-ring test. Imprint (token -> memory) pairs, then progressively
    corrupt the token and measure recall accuracy. Recognition should survive
    small damage and collapse past a threshold -- the curse of forgetting.
    """
    keys, mems = [], []
    for _ in range(n_pairs):
        k = mem._bipolar(mem.key_dim)
        m = mem._bipolar(mem.mem_dim)
        mem.imprint(k, m)
        keys.append(k)
        mems.append(m)
    print("  token corruption -> mean recall accuracy")
    curve = {}
    for frac in levels:
        accs = []
        for k, m in zip(keys, mems):
            kc = mem.corrupt(k, frac) if frac > 0 else k
            r = mem.recall(kc)
            accs.append(np.mean(r == m))
        curve[frac] = float(np.mean(accs))
        bar = "#" * int(round(curve[frac] * 30))
        print(f"    corrupt {int(frac*100):3d}%  acc {curve[frac]:.2f}  {bar}")
    return curve


# ============================================================================= 
# 8.  MAIN
# ============================================================================= 

def main():
    print("=" * 74)
    print(f"  Figure {FIGURE_ID}: {FIGURE_NAME}  ({FIGURE_FLORUIT})")
    print("  THE DHVANI RESONANCE ENGINE")
    print("=" * 74)

    D, K = 12, 8
    B = make_theme_bank(D, K)
    model = DhvaniResonanceEngine(D=D, M=6, H=16, Z=4, S=16, T=3)

    print("\n[1] GRADIENT CHECK  (finite differences vs analytic backprop)")
    x0, _ = sample_meaning(B)
    max_rel, worst = gradient_check(model, x0, x0)
    print(f"  max relative error : {max_rel:.2e}")
    print(f"  worst param        : {worst[0]} index {worst[1]}")
    status = "PASS" if max_rel < 1e-4 else "FAIL"
    print(f"  gradient check     : {status}")

    print("\n[2] TRAINING  (autoencode meanings under a dhvani rate penalty)")
    train(model, B, steps=6000, batch=16, report_every=1000)

    print("\n[3] DHVANI / RATE-DISTORTION  (reconstruct from few emitted bits)")
    test_dhvani_rate_distortion(model, B)

    print("\n[4] UPAMA COMMITMENT  (theme -> one dominant poetic vehicle)")
    test_upama_commitment(model, B)

    print("\n[5] ABHIJNANA MEMORY  (recognition-token recall vs the curse)")
    mem = AbhijnanaMemory(key_dim=64, mem_dim=48)
    test_abhijnana(mem)

    print("\n" + "=" * 74)
    print("  All stages complete.")
    print("=" * 74)


if __name__ == "__main__":
    main()
