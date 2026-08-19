"""
================================================================================
 Chapter 0138 - TERTULLIAN (c. 155-240 CE)
 The "Anima Corporea" network: a from-scratch (pure-NumPy) cognitive
 architecture that embodies Tertullian of Carthage's distinctive theory of mind.
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 7 Minds 121 - 140 Available on Amazon https://www.amazon.com/dp/B0HFN6GXMH
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 138: Tertullian (c. 155-240 CE)
================================================================================   

WHY THIS ARCHITECTURE (and why it is NOT a Transformer)
-------------------------------------------------------
Tertullian is usually mis-remembered for "credo quia absurdum" (a phrase he
never wrote). Read from his surviving works, five *cognitive* commitments stand
out, and each becomes one mechanism below:

  1. PRAESCRIPTIO  (De Praescriptione Haereticorum).  Truth is admitted by
     PRIOR POSSESSION, not proven on neutral ground.  A legal demurrer decides
     whether a claim may even ENTER the argument, *before* its merits are heard.
        -> a fixed admissibility GATE applied before any computation.

  2. CORPOREALITY OF THE SOUL  (De Anima 5-9).  The soul is a real body: it has
     extension and 'figure' (effigies); it is not a placeless Platonic ghost.
        -> a locally-connected CORPOREAL ENCODER: hidden units live at body
           coordinates and see only nearby inputs (Gaussian receptive fields).

  3. APPREHENSIO.  Understanding is a grasp that SETTLES, not a stepwise proof.
        -> an ATTRACTOR: the state relaxes over K iterations to a fixed point
           that "fits" the input, rather than a single feed-forward pass.

  4. TESTIMONIUM ANIMAE naturaliter Christianae  (Apology 17).  The soul arrives
     already bearing innate witness; that witness speaks LOUDEST where trained
     reason is most naive.
        -> a fixed INNATE head, mixed in by a gate that opens in proportion to
           the learned head's uncertainty (entropy).

  5. CREDIBILE QUIA INEPTUM  (De Carne Christi 5.4) - NOT fideism but an
     argument from improbability: what is too unfitting to have been fabricated
     is thereby authenticated.
        -> a training-time CREDIBILITY weight that TRUSTS surprising-yet-coherent
           evidence more than smooth, easily-faked evidence.

Wrapped around the differentiable core is TRADUCIANISM (De Anima 27, 36):
the soul is transmitted by GENERATION, not created blank each time, and it
inherits an ancestral 'stain' (tradux peccati).
        -> an evolutionary outer loop where each new model is SEEDED FROM A
           PARENT plus a persistent inherited bias, never from scratch.

WHAT THE FILE DOES WHEN RUN
---------------------------
  * builds the differentiable core and derives every gradient by hand;
  * PASSES a finite-difference gradient check (mandatory);
  * trains on a synthetic task engineered so the innate-testimony gate is
    measurably useful, and reports learned-only vs. dual-head accuracy;
  * demonstrates the praescriptio gate (admissible vs. precluded inputs);
  * runs the traducian evolutionary loop and shows fitness climbing while the
    inherited 'stain' persists across generations.

Everything is NumPy; there is no autograd and no deep-learning framework.
"""

import numpy as np

RNG = np.random.default_rng(139)  # 139 = Tertullian's chapter number, for reproducibility


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def softmax(z):
    z = z - np.max(z, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)


# ===========================================================================
#  THE MODEL
# ===========================================================================
class AnimaCorporea:
    """
    Tertullian's corporeal, testimony-bearing soul as a differentiable network.

    Trainable parameters (the soul's 'lifetime discipline', formed by ascesis):
        W1, b1        - corporeal encoder (W1 is masked by a fixed locality field)
        W_rec, b_rec  - apprehension attractor recurrence
        W_out, b_out  - learned (discursive) head

    Fixed parameters (given, not learned - the soul's 'created' endowment):
        M             - Gaussian locality mask over body coordinates (corporeality)
        r, theta, kg  - praescriptio gate (admissibility by prior possession)
        W_innate      - the innate testimony head (the soul 'naturally Christian')
    """

    def __init__(self, d_in=16, H=24, C=3, K=5, grid_hidden=(6, 4),
                 grid_input=(4, 4), sigma=0.45, g_rec=0.6, gate_gain=6.0):
        self.d_in, self.H, self.C, self.K = d_in, H, C, K
        self.g_rec = g_rec                    # recurrent gain (keeps attractor stable)
        self.gate_gain = gate_gain            # sharpness of the praescriptio gate
        self.logC = np.log(C)

        # --- corporeality: place hidden units and inputs at body coordinates ---
        hy, hx = grid_hidden
        pos_h = np.array([[i / (hx - 1), j / (hy - 1)]
                          for j in range(hy) for i in range(hx)])[:H]
        iy, ix = grid_input
        pos_i = np.array([[i / (ix - 1), j / (iy - 1)]
                          for j in range(iy) for i in range(ix)])[:d_in]
        # Gaussian receptive field: hidden unit i connects mostly to nearby inputs
        d2 = ((pos_h[:, None, :] - pos_i[None, :, :]) ** 2).sum(-1)  # (H, d_in)
        self.M = np.exp(-d2 / (2 * sigma ** 2))                      # locality mask

        # --- praescriptio gate: a fixed 'rule of faith' direction + threshold ---
        r = RNG.standard_normal(d_in)
        self.r = r / np.linalg.norm(r)
        self.theta = -0.15   # admit unless the plea points against the rule

        # --- trainable parameters (small init) ---
        s = 0.30
        self.W1 = RNG.standard_normal((H, d_in)) * s
        self.b1 = np.zeros(H)
        self.W_rec = RNG.standard_normal((H, H)) * (s / np.sqrt(H))
        self.b_rec = np.zeros(H)
        self.W_out = RNG.standard_normal((C, H)) * s
        self.b_out = np.zeros(C)

        # --- fixed innate testimony head (set later to the task's innate rule) ---
        self.W_innate = RNG.standard_normal((C, d_in)) * s

    # ----- parameter (de)serialization: used by the traducian outer loop -----
    def get_params(self):
        return {k: getattr(self, k).copy()
                for k in ["W1", "b1", "W_rec", "b_rec", "W_out", "b_out"]}

    def set_params(self, p):
        for k, v in p.items():
            setattr(self, k, v.copy())

    # ------------------------------------------------------------------ forward
    def forward(self, X, cache=False):
        """
        X: (N, d_in). Returns final probabilities (N, C).
        If cache=True, also returns a dict of intermediates for backprop.
        """
        N = X.shape[0]
        # 1. PRAESCRIPTIO gate (admissibility by prior possession) -------------
        a = sigmoid(self.gate_gain * (X @ self.r - self.theta))      # (N,)
        Xp = a[:, None] * X                                          # gated input

        # 2. CORPOREAL encoder (locality-masked weights) -----------------------
        W1e = self.W1 * self.M                                       # (H, d_in)
        pre1 = Xp @ W1e.T + self.b1                                  # (N, H)

        # 3. APPREHENSION attractor: relax to a fixed point --------------------
        hs = [np.tanh(pre1)]                                         # h0
        for _ in range(self.K):
            u = self.g_rec * (hs[-1] @ self.W_rec.T) + pre1 + self.b_rec
            hs.append(np.tanh(u))
        h = hs[-1]                                                  # settled state

        # 4a. LEARNED (discursive) head ---------------------------------------
        logits_L = h @ self.W_out.T + self.b_out                    # (N, C)
        pL = softmax(logits_L)
        # entropy of the learned head -> how 'naive' it is on this input
        Hent = -np.sum(pL * np.log(pL + 1e-12), axis=1)             # (N,)
        gate = Hent / self.logC                                     # (N,) in [0,1]

        # 4b. INNATE testimony head (fixed) -----------------------------------
        logits_I = Xp @ self.W_innate.T                             # (N, C)

        # 4c. mixture: testimony speaks louder where reason is uncertain -------
        logits = (1 - gate)[:, None] * logits_L + gate[:, None] * logits_I
        p = softmax(logits)

        if cache:
            cc = dict(X=X, a=a, Xp=Xp, W1e=W1e, pre1=pre1, hs=hs, h=h,
                      logits_L=logits_L, pL=pL, Hent=Hent, gate=gate,
                      logits_I=logits_I, logits=logits, p=p)
            return p, cc
        return p

    # ------------------------------------------------- loss + hand-written grad
    def loss_and_grads(self, X, y, sample_w=None):
        """
        Cross-entropy loss and analytic gradients for all trainable params.
        y: (N,) int labels.  sample_w: optional (N,) credibility weights
        (the 'credibile quia ineptum' re-weighting).
        """
        N = X.shape[0]
        p, c = self.forward(X, cache=True)
        onehot = np.zeros_like(p)
        onehot[np.arange(N), y] = 1.0

        if sample_w is None:
            sample_w = np.ones(N)
        sw = sample_w / (sample_w.sum() + 1e-12)                    # normalized weights

        loss = -np.sum(sw * np.log(p[np.arange(N), y] + 1e-12))

        # ---- gradient w.r.t. final logits (weighted CE) ----
        dlogits = (p - onehot) * sw[:, None]                        # (N, C)

        gate = c["gate"][:, None]
        logits_L, logits_I = c["logits_L"], c["logits_I"]
        pL, Hent = c["pL"], c["Hent"]

        # logits = (1-gate)*logits_L + gate*logits_I
        dlogits_L = (1 - gate) * dlogits                            # direct path
        dlogits_I = gate * dlogits
        dgate = np.sum(dlogits * (logits_I - logits_L), axis=1)     # (N,)

        # gate = entropy(pL)/logC ; dEntropy/dz_i = -pL_i (log pL_i + H)
        dHdz = -pL * (np.log(pL + 1e-12) + Hent[:, None])           # (N, C)
        dlogits_L += (dgate / self.logC)[:, None] * dHdz            # gate path

        # ---- learned head params ----
        h = c["h"]
        dW_out = dlogits_L.T @ h                                    # (C, H)
        db_out = dlogits_L.sum(0)
        dh = dlogits_L @ self.W_out                                 # (N, H)

        # ---- backprop through the attractor unroll ----
        hs = c["hs"]
        dW_rec = np.zeros_like(self.W_rec)
        db_rec = np.zeros_like(self.b_rec)
        dpre1 = np.zeros_like(c["pre1"])
        for t in range(self.K, 0, -1):
            du = dh * (1 - hs[t] ** 2)                              # through tanh
            db_rec += du.sum(0)
            dpre1 += du                                            # pre1 injected each step
            dW_rec += self.g_rec * (du.T @ hs[t - 1])
            dh = self.g_rec * (du @ self.W_rec)
        du0 = dh * (1 - hs[0] ** 2)                                 # h0 = tanh(pre1)
        dpre1 += du0

        # ---- corporeal encoder params ----
        db1 = dpre1.sum(0)
        Xp = c["Xp"]
        dW1e = dpre1.T @ Xp                                         # (H, d_in)
        dW1 = dW1e * self.M                                         # mask is fixed

        grads = dict(W1=dW1, b1=db1, W_rec=dW_rec, b_rec=db_rec,
                     W_out=dW_out, b_out=db_out)
        return loss, grads, p


# ===========================================================================
#  1. FINITE-DIFFERENCE GRADIENT CHECK  (mandatory)
# ===========================================================================
def gradient_check(seed=0):
    rng = np.random.default_rng(seed)
    net = AnimaCorporea(d_in=8, H=10, C=3, K=3, grid_hidden=(5, 2), grid_input=(4, 2))
    X = rng.standard_normal((4, 8))
    y = rng.integers(0, 3, size=4)
    w = rng.uniform(0.5, 1.5, size=4)                              # nontrivial credibility weights

    _, grads, _ = net.loss_and_grads(X, y, sample_w=w)

    eps = 1e-6
    max_rel = 0.0
    for name in ["W1", "b1", "W_rec", "b_rec", "W_out", "b_out"]:
        P = getattr(net, name)
        g_ana = grads[name]
        it = np.nditer(P, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            orig = P[idx]
            P[idx] = orig + eps
            lp, _, _ = net.loss_and_grads(X, y, sample_w=w)
            P[idx] = orig - eps
            lm, _, _ = net.loss_and_grads(X, y, sample_w=w)
            P[idx] = orig
            g_num = (lp - lm) / (2 * eps)
            denom = max(1e-8, abs(g_num) + abs(g_ana[idx]))
            rel = abs(g_num - g_ana[idx]) / denom
            max_rel = max(max_rel, rel)
            it.iternext()
    return max_rel


# ===========================================================================
#  SYNTHETIC TASK  (built so the innate-testimony gate is measurably useful)
# ===========================================================================
def make_task(n=2400, d_in=16, C=3, seed=7):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d_in))

    # innate rule: a fixed linear projection whose argmax is the 'natural' answer
    P_innate = rng.standard_normal((C, d_in))

    # learned rule: a smooth nonlinear function that occupies most of the space
    A = rng.standard_normal((C, d_in))
    Wq = rng.standard_normal(d_in)                                 # direction that marks the 'naive' region
    s = X @ Wq
    naive = s > 1.0                                                # a sparse frontier region

    learned_logits = np.tanh(X @ A.T)                             # structured, learnable
    y = np.argmax(learned_logits, axis=1)
    # inside the naive frontier the truth follows the INNATE rule instead
    y[naive] = np.argmax(X[naive] @ P_innate.T, axis=1)

    # split
    idx = rng.permutation(n)
    tr, va, te = idx[:1400], idx[1400:1700], idx[1700:]

    # UNDER-SAMPLE the naive region in TRAIN so the learned head stays uncertain there
    keep = []
    for i in tr:
        if naive[i] and rng.random() > 0.15:
            continue
        keep.append(i)
    tr = np.array(keep)

    return dict(X=X, y=y, naive=naive, P_innate=P_innate,
                tr=tr, va=va, te=te)


def accuracy(net, X, y, force_gate=None):
    """Optionally force gate=0 (learned-only ablation) to isolate the innate head."""
    if force_gate is None:
        p = net.forward(X)
    else:
        _, c = net.forward(X, cache=True)
        logits = (1 - force_gate) * c["logits_L"] + force_gate * c["logits_I"]
        p = softmax(logits)
    return np.mean(np.argmax(p, axis=1) == y)


# ===========================================================================
#  ADAM optimizer (hand-rolled) + one training run ('lifetime discipline')
# ===========================================================================
class Adam:
    def __init__(self, params, lr=3e-2, b1=0.9, b2=0.999, eps=1e-8):
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
        return params


def train(net, task, steps=220, batch=128, use_credibility=True, verbose=False):
    X, y = task["X"], task["y"]
    tr = task["tr"]
    params = net.get_params()
    opt = Adam(params, lr=3e-2)
    first = last = None
    for t in range(steps):
        bi = RNG.choice(tr, size=min(batch, len(tr)), replace=False)
        Xb, yb = X[bi], y[bi]

        # 'credibile quia ineptum': upweight surprising-yet-coherent examples.
        # surprise = how wrong the current model is; coherence = gate confidence
        # (low entropy). Weight = 1 + surprise*coherence, clipped.
        sw = None
        if use_credibility:
            p, c = net.forward(Xb, cache=True)
            surprise = 1.0 - p[np.arange(len(yb)), yb]              # in [0,1]
            coherence = 1.0 - c["gate"]                            # confident = coherent
            sw = 1.0 + np.clip(surprise * coherence, 0, 1)

        net.set_params(params)
        loss, grads, _ = net.loss_and_grads(Xb, yb, sample_w=sw)
        params = opt.step(params, grads)
        net.set_params(params)
        if first is None:
            first = loss
        last = loss
    return first, last


# ===========================================================================
#  2. TRADUCIANISM: an evolutionary descent that inherits an ancestral 'stain'
# ===========================================================================
def traducian_descent(task, generations=6, pop=6, lifetime=25, seed=3):
    """
    Souls are TRANSMITTED BY DESCENT, never created blank. Each child copies a
    parent's parameters (+ mutation) AND inherits a 'tradux' offset added to b1
    that decays but never vanishes - Tertullian's inherited stain (tradux peccati).
    """
    rng = np.random.default_rng(seed)
    X, y, va = task["X"], task["y"], task["va"]

    def new_soul():
        net = AnimaCorporea()
        net.W_innate = task["P_innate"].copy()                    # innate rule is 'given'
        return net

    # generation 0: one ancestor + noisy copies
    ancestor = new_soul()
    population = []
    for _ in range(pop):
        net = new_soul()
        net.set_params(ancestor.get_params())
        pr = net.get_params()
        for k in pr:
            pr[k] += rng.standard_normal(pr[k].shape) * 0.05
        net.set_params(pr)
        population.append([net, np.zeros_like(net.b1)])           # [soul, tradux]

    history = []
    for g in range(generations):
        scored = []
        for net, tradux in population:
            net.b1 = net.b1 + tradux                              # the stain is expressed...
            train(net, task, steps=lifetime, batch=96, use_credibility=False)
            fit = accuracy(net, X[va], y[va])
            net.b1 = net.b1 - tradux                              # ...then separated from learned bias
            scored.append((fit, net, tradux))
        scored.sort(key=lambda s: -s[0])
        best_fit = scored[0][0]
        stain_norm = float(np.linalg.norm(scored[0][2]))
        history.append((best_fit, stain_norm))

        # reproduce from the top parents (descent + inherited, persistent stain)
        parents = scored[:max(2, pop // 2)]
        children = []
        for i in range(pop):
            _, pnet, ptradux = parents[i % len(parents)]
            child = new_soul()
            child.set_params(pnet.get_params())
            cp = child.get_params()
            for k in cp:
                cp[k] += rng.standard_normal(cp[k].shape) * 0.03  # mutation
            child.set_params(cp)
            # tradux: inherited (decayed) + a fresh, small, persistent increment
            new_tradux = 0.85 * ptradux + 0.05 * rng.standard_normal(child.b1.shape)
            children.append([child, new_tradux])
        population = children
    return history


# ===========================================================================
#  MAIN: run every self-test and print a verifiable report
# ===========================================================================
if __name__ == "__main__":
    print("=" * 74)
    print(" ANIMA CORPOREA  -  Tertullian (Chapter 0139)  -  self-test report")
    print("=" * 74)

    # 1) gradient check ------------------------------------------------------
    max_rel = gradient_check()
    ok = max_rel < 1e-4
    print(f"\n[1] Finite-difference gradient check")
    print(f"    max relative error = {max_rel:.2e}   ->  {'PASS' if ok else 'FAIL'}")
    assert ok, "gradient check failed"

    # 2) train + show the innate-testimony gate is useful --------------------
    task = make_task()
    net = AnimaCorporea()
    net.W_innate = task["P_innate"].copy()          # innate testimony = the natural rule
    first, last = train(net, task, steps=220, use_credibility=True)
    X, y = task["X"], task["y"]
    te, naive = task["te"], task["naive"]
    te_naive = te[naive[te]]

    acc_all = accuracy(net, X[te], y[te])
    acc_naive_dual = accuracy(net, X[te_naive], y[te_naive])
    acc_naive_learned = accuracy(net, X[te_naive], y[te_naive], force_gate=0.0)
    print(f"\n[2] Training ('lifetime discipline')")
    print(f"    loss {first:.3f} -> {last:.3f}")
    print(f"    overall test accuracy .................... {acc_all:.3f}")
    print(f"    naive-frontier acc, learned head only .... {acc_naive_learned:.3f}")
    print(f"    naive-frontier acc, WITH innate testimony  {acc_naive_dual:.3f}")
    print(f"    -> testimony gate rescue = "
          f"{acc_naive_dual - acc_naive_learned:+.3f} on the naive frontier")

    # 3) praescriptio gate: admissible vs precluded pleas --------------------
    Xt = X[te]
    a = sigmoid(net.gate_gain * (Xt @ net.r - net.theta))
    adm = Xt[a > 0.5]
    pre = Xt[a <= 0.5]
    conf_adm = np.max(net.forward(adm), axis=1).mean() if len(adm) else float("nan")
    conf_pre = np.max(net.forward(pre), axis=1).mean() if len(pre) else float("nan")
    print(f"\n[3] Praescriptio gate (admission by prior possession)")
    print(f"    admitted pleas: {len(adm):4d}   mean confidence {conf_adm:.3f}")
    print(f"    precluded pleas:{len(pre):4d}   mean confidence {conf_pre:.3f}")

    # 4) traducian descent: fitness climbs, inherited stain persists ---------
    print(f"\n[4] Traducian descent (soul transmitted, stain inherited)")
    hist = traducian_descent(task)
    for g, (fit, stain) in enumerate(hist):
        print(f"    gen {g}:  best fitness {fit:.3f}   inherited-stain |tradux| {stain:.3f}")
    improved = hist[-1][0] >= hist[0][0]
    persists = hist[-1][1] > 1e-3
    print(f"    fitness improved across descent: {improved}")
    print(f"    ancestral stain still present at final generation: {persists}")

    print("\n" + "=" * 74)
    print(" All self-tests complete.")
    print("=" * 74)


# ============================================================================
#  VERIFIED EXECUTION OUTPUT (captured when this file was run)
# ----------------------------------------------------------------------------
#  [1] Finite-difference gradient check
#      max relative error = 6.72e-05   ->  PASS
#
#  [2] Training ('lifetime discipline')
#      loss 1.785 -> 0.448
#      overall test accuracy .................... 0.646
#      naive-frontier acc, learned head only .... 0.285
#      naive-frontier acc, WITH innate testimony  0.443
#      -> testimony gate rescue = +0.158 on the naive frontier
#
#  [3] Praescriptio gate (admission by prior possession)
#      admitted pleas:  396   mean confidence 0.860
#      precluded pleas: 304   mean confidence 0.635
#
#  [4] Traducian descent (soul transmitted, stain inherited)
#      gen 0:  best fitness 0.607   inherited-stain |tradux| 0.000
#      gen 1:  best fitness 0.640   inherited-stain |tradux| 0.197
#      gen 2:  best fitness 0.627   inherited-stain |tradux| 0.302
#      gen 3:  best fitness 0.633   inherited-stain |tradux| 0.448
#      gen 4:  best fitness 0.657   inherited-stain |tradux| 0.369
#      gen 5:  best fitness 0.670   inherited-stain |tradux| 0.487
#      fitness improved across descent: True
#      ancestral stain still present at final generation: True
# ============================================================================
