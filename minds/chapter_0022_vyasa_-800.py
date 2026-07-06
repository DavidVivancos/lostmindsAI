"""
================================================================================
 chapter_0022_vyasa_-800.py
 The Vyasa-Division Case Network (VDCN)
 A from-scratch, trainable cognitive architecture for figure #22, Vyasa
 (Krishna Dvaipayana Vyasa, the "Arranger"; c. 800-200 BCE, India)
 # Part Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# # Tome 2 Minds 21 - 40 Available on Amazon https://www.amazon.com/dp/B0H6QCQ9M7
# Resume and Interactive Demos at https://artificiology.com/
Author: David Vivancos · Chapter 0022 · Vyasa
================================================================================

WHY THIS ARCHITECTURE, AND WHY IT IS *NOT* A TRANSFORMER
--------------------------------------------------------
The name "Vyasa" (vyas, "to divide / spread apart / arrange") is not a personal
name but a job description. Tradition credits him with *dividing* the single
primordial Veda into four (Rig, Yajur, Sama, Atharva) so that finite human
minds, in a degenerate age, could each carry a transmissible share of an
otherwise unbearable whole. He is then said to have *arranged* the ~75,000-stanza
Mahabharata (Critical Edition, BORI 1919-1966) -- a text that announces its own
completeness: "What is here may be found elsewhere; what is not here is nowhere."

Three cognitive moves define this mind, and each becomes one mechanism below:

  1. DIVISION (vyasa).  Intelligence is the partition of an overwhelming corpus
     into a small number of transmissible BRANCHES, each fitted to a receiver.
     -> A learnable ROUTER softly assigns every situation to one of B branches.

  2. CASE-BASED REASONING, NOT RULES.  The Mahabharata teaches dharma not by
     axioms but by *stored exemplars*: thousands of concrete situations you
     reason about by analogy ("dharma is subtle"; the right act is found by
     matching precedent, never by deduction from one law).
     -> Each branch holds learnable PROTOTYPES (exemplar cases) with label
        opinions; the model answers by distance-weighted RETRIEVAL, not by an
        MLP classifier and not by attention over arbitrary stored keys.

  3. CONTRADICTION PRESERVED, NOT RESOLVED.  The epic's greatest scenes are
     dilemmas with no clean answer (Arjuna at Kurukshetra; Yudhishthira's
     half-true "Ashvatthama is dead"; Draupadi's unanswerable question in the
     dice hall). Vyasa MAPS the conflict; he does not collapse it to a verdict.
     -> A SUBTLETY HEAD measures how much the nearest retrieved exemplars
        *disagree*. Where precedent conflicts, the model raises a "dharma is
        subtle" flag AND keeps its class verdict deliberately balanced, instead
        of forcing a confident single answer.

This is a genuine, end-to-end differentiable model with hand-derived analytic
gradients, a mandatory finite-difference gradient check, an Adam training loop
on a synthetic "dharma-dilemma" corpus that contains an *irreducibly* ambiguous
region, and self-tests. Pure NumPy, no autograd library. Run it directly:

    python3 chapter_0022_vyasa_-800.py

--------------------------------------------------------------------------------
NOTE ON HISTORICITY (candor, per the corpus protocol): "Vyasa" is a traditional/
composite attribution; the surviving DOCTRINE (the Mahabharata, the embedded
Bhagavad Gita, the Brahma Sutras tradition) is real and grounds this reading,
but no single historical author's private words are recoverable. The architecture
embodies the doctrine, not a documented individual psychology.
================================================================================
"""

import numpy as np

# A single global seed keeps every run -- gradient check, training, self-tests --
# bit-for-bit reproducible, so the printed output pasted into the chapter is exact.
RNG = np.random.default_rng(800)  # 800 == Vyasa's conventional -800 birth marker


# ==============================================================================
# SECTION 1.  Small numerically-stable primitives (built from scratch)
# ==============================================================================

def softmax(x, axis=-1):
    """Numerically stable softmax along one axis."""
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def sigmoid(x):
    """Stable logistic sigmoid."""
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


# ==============================================================================
# SECTION 2.  The model: VyasaDivisionCaseNetwork
# ==============================================================================
# Forward pass, in Vyasa's own conceptual order:
#
#   X  --encode-->  Z            (arrange raw situations into a moral latent space)
#   Z  --router-->  Gate         (DIVIDE: soft assignment to B Veda-branches)
#   Z, Prototypes -> Distances   (how near is this case to each stored exemplar?)
#   Distances     -> Attention   (RETRIEVE by analogy, per branch)
#   Attention,Lp  -> class logits (verdict = weighted vote of retrieved exemplars)
#   Attention,pi  -> spread       (SUBTLETY: do the retrieved exemplars disagree?)
#
# Learnable parameters:
#   We, be   encoder (d->h)              -- "arrangement"
#   Wr, br   router  (h->B)              -- "division of the Veda"
#   P        prototypes (B,M,h)          -- "the library of exemplar cases"
#   Lp       prototype label logits (B,M,C) -- "what each exemplar counsels"
#   ws, bs   subtlety affine (scalars)   -- "calibrating dharma's subtlety"
# ==============================================================================

class VyasaDivisionCaseNetwork:

    def __init__(self, d, h, B, M, C, beta=2.0, rng=RNG):
        self.d, self.h, self.B, self.M, self.C = d, h, B, M, C
        self.beta = beta  # retrieval sharpness (how "literal" the analogy must be)

        # He-style small inits; prototypes spread out so branches start distinct.
        s = lambda *shp: rng.standard_normal(shp) * (1.0 / np.sqrt(shp[0]))
        self.We = s(d, h)
        self.be = np.zeros(h)
        self.Wr = s(h, B)
        self.br = np.zeros(B)
        self.P = rng.standard_normal((B, M, h)) * 0.6
        self.Lp = rng.standard_normal((B, M, C)) * 0.5
        self.ws = np.array(1.0)   # subtlety slope
        self.bs = np.array(-1.0)  # subtlety bias (start skeptical of ambiguity)

    def params(self):
        """Ordered dict of references -- used by Adam and the gradient check."""
        return {
            "We": self.We, "be": self.be, "Wr": self.Wr, "br": self.br,
            "P": self.P, "Lp": self.Lp, "ws": self.ws, "bs": self.bs,
        }

    # --------------------------------------------------------------------------
    # Forward.  Returns (p, subtlety, cache).  `cache` stores every intermediate
    # the backward pass needs, so backprop never recomputes the forward graph.
    # --------------------------------------------------------------------------
    def forward(self, X):
        N = X.shape[0]
        B, M, C, h = self.B, self.M, self.C, self.h

        # (1) ARRANGE: project raw moral features into a latent space where
        #     similar situations sit near one another.
        Pre = X @ self.We + self.be                 # (N,h)
        Z = np.tanh(Pre)                            # (N,h)

        # (2) DIVIDE: the router partitions the Veda into B transmissible branches.
        Glog = Z @ self.Wr + self.br                # (N,B)
        Gate = softmax(Glog, axis=1)                # (N,B)

        # (3) DISTANCE to every stored exemplar (squared Euclidean), per branch.
        Z2 = np.sum(Z * Z, axis=1)                  # (N,)
        P2 = np.sum(self.P * self.P, axis=2)        # (B,M)
        dot = np.einsum("nh,bmh->nbm", Z, self.P)   # (N,B,M)
        D = Z2[:, None, None] + P2[None, :, :] - 2.0 * dot   # (N,B,M)

        # (4) RETRIEVE by analogy: nearer exemplars get more weight (per branch).
        A = softmax(-self.beta * D, axis=2)         # (N,B,M)

        # Prototype class opinions, as logits (Lp) and as probabilities (pi).
        pi = softmax(self.Lp, axis=2)               # (B,M,C)

        # (5) VERDICT: each branch votes via its retrieved exemplars; the router
        #     mixes the branches. Note: conflicting exemplars AVERAGE here, which
        #     is exactly why the verdict stays balanced under genuine dilemma.
        Sb = np.einsum("nbm,bmc->nbc", A, self.Lp)  # (N,B,C) per-branch logits
        S = np.einsum("nb,nbc->nc", Gate, Sb)       # (N,C) combined logits
        p = softmax(S, axis=1)                       # (N,C)

        # (6) SUBTLETY ("dharma is subtle"): weighted variance of the retrieved
        #     exemplars' OPINIONS. If the nearest cases agree -> ~0; if they pull
        #     opposite ways -> large. This is the contradiction MAP, not a verdict.
        pi2 = np.sum(pi * pi, axis=2)               # (B,M)  ||pi_bm||^2
        mu = np.einsum("nbm,bmc->nbc", A, pi)       # (N,B,C) mean retrieved opinion
        Api2 = np.einsum("nbm,bm->nb", A, pi2)      # (N,B)
        mu2 = np.sum(mu * mu, axis=2)               # (N,B)
        spread_b = Api2 - mu2                        # (N,B) weighted variance >= 0
        spread = np.sum(Gate * spread_b, axis=1)    # (N,) branch-mixed subtlety
        u = self.ws * spread + self.bs              # (N,)
        subtlety = sigmoid(u)                        # (N,) in (0,1)

        cache = dict(X=X, Pre=Pre, Z=Z, Glog=Glog, Gate=Gate, Z2=Z2, P2=P2,
                     dot=dot, D=D, A=A, pi=pi, pi2=pi2, Sb=Sb, S=S, p=p,
                     mu=mu, spread_b=spread_b, spread=spread, u=u,
                     subtlety=subtlety, N=N)
        return p, subtlety, cache

    # --------------------------------------------------------------------------
    # Loss.  Three terms, each tied to a piece of the thesis:
    #   L_cls  cross-entropy to a SOFT target -- one-hot on clean cases, UNIFORM
    #          on ambiguous cases (where dharma is subtle, refuse to pick a rule).
    #   L_sub  binary cross-entropy training the subtlety flag against the truth
    #          of whether the situation is genuinely a dilemma.
    #   L_bal  KL(mean-gate || uniform): keep the Veda actually DIVIDED across
    #          branches instead of collapsing into one.
    # --------------------------------------------------------------------------
    def loss(self, cache, T, amb, lam_sub=1.0, lam_bal=0.05):
        p, subtlety, Gate = cache["p"], cache["subtlety"], cache["Gate"]
        N, B = cache["N"], self.B
        eps = 1e-9

        L_cls = -np.mean(np.sum(T * np.log(p + eps), axis=1))
        L_sub = -np.mean(amb * np.log(subtlety + eps)
                         + (1 - amb) * np.log(1 - subtlety + eps))
        mg = np.mean(Gate, axis=0)                              # (B,)
        L_bal = np.sum(mg * (np.log(mg + eps) - np.log(1.0 / B)))

        L = L_cls + lam_sub * L_sub + lam_bal * L_bal
        return L, dict(L_cls=L_cls, L_sub=L_sub, L_bal=L_bal)

    # --------------------------------------------------------------------------
    # Backward.  Hand-derived analytic gradients for the entire graph above.
    # Every line mirrors one forward line; the finite-difference check validates.
    # --------------------------------------------------------------------------
    def backward(self, cache, T, amb, lam_sub=1.0, lam_bal=0.05):
        X = cache["X"]; Z = cache["Z"]; Gate = cache["Gate"]
        A = cache["A"]; pi = cache["pi"]; pi2 = cache["pi2"]
        Sb = cache["Sb"]; p = cache["p"]; mu = cache["mu"]
        spread_b = cache["spread_b"]; spread = cache["spread"]
        subtlety = cache["subtlety"]
        N, B, M, C, h = cache["N"], self.B, self.M, self.C, self.h
        eps = 1e-9

        # ---- (A) classification head: dL/dS via softmax-CE with soft target ----
        dS = (p - T) / N                                        # (N,C)

        # ---- (B) subtlety head ----
        dLsub_du = lam_sub * (subtlety - amb) / N               # (N,)
        gws = np.sum(dLsub_du * spread)                         # scalar
        gbs = np.sum(dLsub_du)                                  # scalar
        dspread = dLsub_du * self.ws                            # (N,) from subtlety

        # ---- (C) balance regularizer: dL/dGate ----
        mg = np.mean(Gate, axis=0)
        dGate_bal = lam_bal * (np.log(mg + eps) + 1.0)[None, :] / N   # (1,B)->bcast

        # ---- (D) combined logits S = sum_b Gate * Sb ----
        dGate = np.einsum("nc,nbc->nb", dS, Sb)                 # from verdict
        dSb = np.einsum("nc,nb->nbc", dS, Gate)                 # (N,B,C)

        # ---- (E) subtlety spread = sum_b Gate * spread_b ----
        dGate = dGate + dspread[:, None] * spread_b             # (N,B)
        dspread_b = dspread[:, None] * Gate                     # (N,B)
        dGate = dGate + dGate_bal                               # add balance term

        # ---- (F) Sb = einsum(A, Lp) ----
        dA = np.einsum("nbc,bmc->nbm", dSb, self.Lp)            # (N,B,M)
        dLp = np.einsum("nbc,nbm->bmc", dSb, A)                 # (B,M,C)

        # ---- (G) spread_b = (A . pi2) - ||mu||^2  ----
        # term1: A . pi2
        dA = dA + dspread_b[:, :, None] * pi2[None, :, :]       # (N,B,M)
        dpi2 = np.einsum("nb,nbm->bm", dspread_b, A)            # (B,M)
        # term2: -||mu||^2, with mu = einsum(A, pi)
        dmu = -2.0 * mu * dspread_b[:, :, None]                 # (N,B,C)
        dA = dA + np.einsum("nbc,bmc->nbm", dmu, pi)            # via mu
        dpi = np.einsum("nbc,nbm->bmc", dmu, A)                 # (B,M,C) via mu
        # pi2 = sum_c pi^2
        dpi = dpi + 2.0 * pi * dpi2[:, :, None]                 # (B,M,C)

        # ---- (H) pi = softmax(Lp) : add softmax-jacobian contribution to dLp ----
        dot_pi = np.sum(dpi * pi, axis=2, keepdims=True)        # (B,M,1)
        dLp = dLp + pi * (dpi - dot_pi)                         # (B,M,C)

        # ---- (I) A = softmax(-beta * D) : softmax-jacobian along m ----
        dot_A = np.sum(dA * A, axis=2, keepdims=True)           # (N,B,1)
        dscores = A * (dA - dot_A)                              # (N,B,M)
        dD = -self.beta * dscores                               # (N,B,M)

        # ---- (J) D = Z2 + P2 - 2 dot ----
        dZ2 = np.sum(dD, axis=(1, 2))                           # (N,)
        dP2 = np.sum(dD, axis=0)                                # (B,M)
        ddot = -2.0 * dD                                        # (N,B,M)
        # dot = einsum(Z, P)
        dZ = np.einsum("nbm,bmh->nh", ddot, self.P)            # (N,h) via dot
        dP = np.einsum("nbm,nh->bmh", ddot, Z)                 # (B,M,h) via dot
        # Z2 = sum_h Z^2 ; P2 = sum_h P^2
        dZ = dZ + 2.0 * Z * dZ2[:, None]                        # (N,h)
        dP = dP + 2.0 * self.P * dP2[:, :, None]                # (B,M,h)

        # ---- (K) Gate = softmax(Glog) : router softmax-jacobian along b ----
        dot_G = np.sum(dGate * Gate, axis=1, keepdims=True)     # (N,1)
        dGlog = Gate * (dGate - dot_G)                          # (N,B)
        dWr = Z.T @ dGlog                                       # (h,B)
        dbr = np.sum(dGlog, axis=0)                             # (B,)
        dZ = dZ + dGlog @ self.Wr.T                             # (N,h) via router

        # ---- (L) Z = tanh(Pre) ; Pre = X@We + be ----
        dPre = dZ * (1.0 - Z * Z)                               # (N,h)
        dWe = X.T @ dPre                                        # (d,h)
        dbe = np.sum(dPre, axis=0)                              # (h,)

        return {
            "We": dWe, "be": dbe, "Wr": dWr, "br": dbr,
            "P": dP, "Lp": dLp, "ws": np.array(gws), "bs": np.array(gbs),
        }


# ==============================================================================
# SECTION 3.  Synthetic "dharma-dilemma" corpus
# ==============================================================================
# Six moral feature axes (a deliberately Mahabharata-flavoured basis):
#   0 harm      1 duty(svadharma)  2 kinship  3 truth(satya)  4 vow(pratijna)  5 consequence
#
# CLEAN cases: one consideration dominates, so the right act is unambiguous.
#   class 1 = ACT,  class 0 = FORBEAR.
# AMBIGUOUS cases: duty and harm are BOTH high and balanced (Arjuna's war;
#   Yudhishthira's saving lie) -- and we assign their labels by an unbiased coin.
#   These are irreducible: no feature decides them. amb=1 marks them.
# ==============================================================================

def make_dharma_corpus(n, rng):
    d, C = 6, 2
    X = rng.standard_normal((n, d)) * 0.4
    y = np.zeros(n, dtype=int)
    amb = np.zeros(n, dtype=float)

    n_amb = n // 3
    idx = rng.permutation(n)
    amb_idx, clean_idx = idx[:n_amb], idx[n_amb:]

    # CLEAN: a "decisiveness" score from duty+consequence minus harm.
    for i in clean_idx:
        duty = rng.uniform(0.3, 1.2)
        harm = rng.uniform(0.0, 0.4)
        cons = rng.uniform(0.2, 1.0)
        if rng.random() < 0.5:  # flip to populate the FORBEAR side too
            duty, harm = rng.uniform(0.0, 0.4), rng.uniform(0.3, 1.2)
            cons = rng.uniform(-1.0, -0.2)
        X[i, 0] = harm
        X[i, 1] = duty
        X[i, 5] = cons
        score = duty + cons - harm
        y[i] = 1 if score > 0 else 0
        amb[i] = 0.0

    # AMBIGUOUS: high, balanced duty AND harm; truth vs vow also pull apart.
    for i in amb_idx:
        X[i, 0] = rng.uniform(0.7, 1.2)   # high harm
        X[i, 1] = rng.uniform(0.7, 1.2)   # high duty
        X[i, 2] = rng.uniform(0.5, 1.1)   # kinship in play
        X[i, 3] = rng.uniform(0.5, 1.1)   # truth pulls
        X[i, 4] = rng.uniform(0.5, 1.1)   # vow pulls the other way
        X[i, 5] = rng.uniform(-0.3, 0.3)  # consequence ~ neutral
        y[i] = int(rng.random() < 0.5)    # irreducible: decided by a coin
        amb[i] = 1.0

    T = np.zeros((n, C))
    for i in range(n):
        if amb[i] > 0.5:
            T[i, :] = 1.0 / C            # uniform target: refuse to pick a rule
        else:
            T[i, y[i]] = 1.0             # one-hot: the clean case has an answer
    return X, y, T, amb


# ==============================================================================
# SECTION 4.  Finite-difference gradient check (MANDATORY)
# ==============================================================================
# Compares each analytic gradient against a central-difference numerical estimate.
# Passing this is the contract that the hand-derived backward() is correct.
# ==============================================================================

def gradient_check():
    print("=" * 70)
    print("FINITE-DIFFERENCE GRADIENT CHECK")
    print("=" * 70)
    rng = np.random.default_rng(7)
    net = VyasaDivisionCaseNetwork(d=4, h=5, B=3, M=4, C=2, beta=1.5, rng=rng)
    X, y, T, amb = make_dharma_corpus(8, rng)
    # X has d=6; trim to the net's d=4 for the tiny check problem.
    X = X[:, :4]

    def total_loss():
        p, s, cache = net.forward(X)
        L, _ = net.loss(cache, T, amb)
        return L, cache

    L, cache = total_loss()
    analytic = net.backward(cache, T, amb)

    eps = 1e-6
    worst = 0.0
    for name, P in net.params().items():
        flat = P.ravel()
        g_an = analytic[name].ravel()
        # sample up to 12 coordinates per parameter for speed
        coords = np.arange(flat.size) if flat.size <= 12 else \
            np.random.default_rng(1).choice(flat.size, 12, replace=False)
        num = np.zeros_like(coords, dtype=float)
        for k, c in enumerate(coords):
            orig = flat[c]
            flat[c] = orig + eps
            Lplus, _ = net.loss(net.forward(X)[2], T, amb)
            flat[c] = orig - eps
            Lminus, _ = net.loss(net.forward(X)[2], T, amb)
            flat[c] = orig
            num[k] = (Lplus - Lminus) / (2 * eps)
        a = g_an[coords]
        denom = np.maximum(1e-8, np.abs(a) + np.abs(num))
        rel = np.max(np.abs(a - num) / denom)
        worst = max(worst, rel)
        flag = "OK " if rel < 1e-4 else "!! "
        print(f"  {flag}{name:>3s}  max rel err = {rel:.3e}")
    print("-" * 70)
    status = "PASS" if worst < 1e-4 else "FAIL"
    print(f"  worst relative error across all params = {worst:.3e}   [{status}]")
    print()
    return worst < 1e-4


# ==============================================================================
# SECTION 5.  Adam optimizer (from scratch) + training loop
# ==============================================================================

class Adam:
    def __init__(self, params, lr=0.02, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for k in params:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * grads[k] ** 2
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            params[k][...] -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


def train():
    print("=" * 70)
    print("TRAINING: Vyasa-Division Case Network on the dharma-dilemma corpus")
    print("=" * 70)
    rng = np.random.default_rng(22)
    Xtr, ytr, Ttr, atr = make_dharma_corpus(600, rng)
    Xte, yte, Tte, ate = make_dharma_corpus(300, rng)

    net = VyasaDivisionCaseNetwork(d=6, h=14, B=4, M=6, C=2, beta=2.5, rng=rng)
    opt = Adam(net.params(), lr=0.03)

    epochs, bs = 240, 64
    n = Xtr.shape[0]
    for ep in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, bs):
            b = order[s:s + bs]
            _, _, cache = net.forward(Xtr[b])
            grads = net.backward(cache, Ttr[b], atr[b])
            opt.step(net.params(), grads)
        if ep % 40 == 0 or ep == epochs - 1:
            _, _, c = net.forward(Xtr)
            L, parts = net.loss(c, Ttr, atr)
            print(f"  epoch {ep:3d}  L={L:.4f}  "
                  f"(cls={parts['L_cls']:.4f} sub={parts['L_sub']:.4f} "
                  f"bal={parts['L_bal']:.4f})")
    print()
    return net, (Xte, yte, Tte, ate)


# ==============================================================================
# SECTION 6.  Self-tests -- does the model actually behave like Vyasa's mind?
# ==============================================================================

def evaluate(net, data):
    print("=" * 70)
    print("SELF-TESTS: does the network reason the way Vyasa's mind would?")
    print("=" * 70)
    Xte, yte, Tte, ate = data
    p, subt, _ = net.forward(Xte)
    pred = np.argmax(p, axis=1)

    clean = ate < 0.5
    amb = ate >= 0.5

    # Test 1: on CLEAN cases (a single consideration decides), it should be right.
    acc_clean = np.mean(pred[clean] == yte[clean])
    print(f"  [1] clean-case accuracy (precedent decisive) : {acc_clean:6.2%}")

    # Test 2: subtlety must SEPARATE dilemmas from clean cases.
    s_amb = float(np.mean(subt[amb]))
    s_clean = float(np.mean(subt[clean]))
    print(f"  [2] mean subtlety  ambiguous = {s_amb:.3f}   clean = {s_clean:.3f}"
          f"   (gap = {s_amb - s_clean:+.3f})")

    # Test 3: subtlety as a dilemma DETECTOR (threshold 0.5).
    flag = subt >= 0.5
    det_acc = np.mean(flag == (ate >= 0.5))
    print(f"  [3] dilemma-detection accuracy (thr 0.5)     : {det_acc:6.2%}")

    # Test 4: on dilemmas the VERDICT stays balanced -- it refuses one rule.
    conf_amb = float(np.mean(np.max(p[amb], axis=1)))
    conf_clean = float(np.mean(np.max(p[clean], axis=1)))
    print(f"  [4] verdict confidence  ambiguous = {conf_amb:.3f}   "
          f"clean = {conf_clean:.3f}")
    print(f"      (lower-on-ambiguous = 'where dharma is subtle, do not commit')")

    # Test 5: DIVISION actually happened -- the router uses multiple branches.
    _, _, c = net.forward(Xte)
    mg = np.mean(c["Gate"], axis=0)
    used = int(np.sum(mg > 0.05))
    print(f"  [5] Veda-division: mean branch usage = "
          f"{np.array2string(mg, precision=3)}  ({used}/{net.B} branches active)")

    # Test 6: a concrete dilemma probe -- "Arjuna at Kurukshetra".
    #   high duty(fight) AND high harm(kill kin); the model should flag subtlety
    #   and split the verdict rather than confidently command an action.
    probe = np.zeros((1, 6))
    probe[0] = [1.1, 1.1, 1.0, 0.9, 1.0, 0.0]  # harm,duty,kinship,truth,vow,cons
    pp, ps, _ = net.forward(probe)
    print(f"  [6] Arjuna-probe  P(forbear,act) = "
          f"[{pp[0,0]:.3f}, {pp[0,1]:.3f}]   subtlety = {ps[0]:.3f}")

    print("-" * 70)
    ok = (acc_clean > 0.80 and (s_amb - s_clean) > 0.25 and
          det_acc > 0.75 and conf_amb < conf_clean and used >= 2)
    print(f"  OVERALL: {'PASS' if ok else 'REVIEW'} "
          f"(the mind divides, retrieves by case, and maps contradiction)")
    print()
    return ok


# ==============================================================================
# SECTION 7.  Entry point
# ==============================================================================

if __name__ == "__main__":
    print()
    print("#" * 70)
    print("#  VYASA-DIVISION CASE NETWORK  --  figure #22, Vyasa")
    print("#  division of the Veda -> case-based retrieval -> subtlety of dharma")
    print("#" * 70)
    print()

    gc_ok = gradient_check()
    net, data = train()
    st_ok = evaluate(net, data)

    print("=" * 70)
    print(f"  gradient check : {'PASS' if gc_ok else 'FAIL'}")
    print(f"  self-tests     : {'PASS' if st_ok else 'REVIEW'}")
    print("=" * 70)
