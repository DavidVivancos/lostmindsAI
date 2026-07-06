"""
=============================================================================
 chapter_0058_herodotus_-484.py  —  The HISTOR-NET
 A from-scratch (pure NumPy) cognitive architecture distilled from the mind of
 HERODOTUS OF HALICARNASSUS (c. 484 - c. 425 BCE).
 Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
Tome 3 Minds 41 - 60 Available on Amazon https://www.amazon.com/dp/B0H6TVX69S
Resume and Interactive Demos at https://artificiology.com/
Author: David Vivancos · Chapter 0058 · Herodotus of Halicarnassus
=============================================================================

WHY THIS ARCHITECTURE, AND NOT A TRANSFORMER
--------------------------------------------
Herodotus is usually flattened into "the father of inquiry / a weigher of
sources". That reading is true but shallow, and it is the same reading every
inquiry-minded figure receives. The Histories actually fuse THREE ideas that no
other ancient thinker fused, and those three ideas — not generic "inquiry" —
are what this network encodes:

  1. NOMOS BASILEUS  ("custom is king", Histories 3.38).
     There is no view from nowhere. Every judgement is made *inside* a learned
     custom-frame (a nomos). The Greeks burn their dead and are horrified by the
     Callatiae who eat theirs; the Callatiae are equally horrified by cremation.
     ==> The same observation must be encoded by SEVERAL frame-specific lenses
         and mixed; there is no single canonical embedding.

  2. THOMA  (wonder; the proem promises "great and wonderful deeds",
     erga megala te kai thomasta).
     Herodotus allocates memory and attention to the MARVELLOUS — the anomalous,
     the surprising, the thing that violates expectation.
     ==> A "wonder gate": features that deviate from the learned expectation are
         AMPLIFIED. Surprise, not magnitude, controls what the network attends to.

  3. THE WHEEL OF FORTUNE  (the proem's law, 1.5.4: "human prosperity never
     continues in the same place"; Solon to Croesus, 1.32: "the divine is wholly
     envious/phthoneron"; repeated for Polycrates 3.40 and Xerxes 7.10/7.46).
     The deep law his comparison reveals is REVERSAL: what rises too high is cut
     down (phthonos/nemesis). Greatness is mean-reverting.
     ==> A recurrent "fortune" dynamic with a NEMESIS restoring force whose
         strength grows once an elevation threshold is crossed.

A fourth, methodological idea threads through all of them:

  4. PROVENANCE-TAGGED CREDENCE  (opsis = sight, akoe = hearsay, gnome =
     judgement; Histories 2.99; "I am obliged to report what is said, but I am
     not obliged to believe it", 7.152.3).
     Herodotus never collapses claims into a flat truth-value; each claim keeps
     the tag of HOW it was known.
     ==> Every input feature carries a reliability weight r in [0,1]
         (opsis ~ 1.0 > akoe ~ 0.5 > gnome-inference ~ 0.3) that scales its
         influence before any lens reads it.

So the HISTOR-NET is a FRAME-RELATIVE, WONDER-GATED, REVERSAL-AWARE comparator
that keeps provenance attached. It is NOT an oracle that emits one answer; it
holds several culturally-indexed readings, learns most from what surprises it,
and predicts that any trajectory tending to an extreme will revert.

THE TASK IT IS TRAINED ON
-------------------------
Each example is a "polity at its zenith": a vector of observed attributes
(power, wealth, territory, and hubris-markers such as bridging the Hellespont
or whipping the sea). Each attribute carries a provenance weight. The network
must predict the polity's FORTUNE TRAJECTORY across T "ages" — the Herodotean
expectation being that the higher and more hubristic the zenith, the sharper
the coming reversal. This is the Croesus / Polycrates / Xerxes problem made
learnable.

ENGINEERING CONTRACT (kept in every file of this corpus)
--------------------------------------------------------
  * Pure NumPy, built from scratch (no autograd, no ML frameworks).
  * Hand-derived analytic gradients.
  * A finite-difference gradient check that MUST pass (printed at run time).
  * A real training loop that drives the loss down.
  * Self-tests, then a small demonstration.
Run this file directly to see the gradient check, training curve, and demo.
=============================================================================
"""

import numpy as np

# -----------------------------------------------------------------------------
#  Numerically-stable primitives
# -----------------------------------------------------------------------------
def softplus(x):
    # log(1 + e^x), stable for large |x|
    return np.logaddexp(0.0, x)

def sigmoid(x):
    # 1 / (1 + e^-x), stable
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out

def softmax_rows(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


# =============================================================================
#  THE MODEL
# =============================================================================
class HistorNet:
    """
    Forward pipeline (one polity, vectorised over a batch of N):

      xe   = X * R                      provenance-weighted input  (nomos reads only what is credibly known)
      g    = softmax(xe @ Wr^T + br)    which custom-frames apply  (NOMOS routing)
      Hk   = tanh(xe @ We[k]^T + be[k]) the k-th frame's reading   (each nomos sees differently)
      h    = sum_k g_k * Hk             frame-mixed representation
      gate = sigmoid(gw*(h-mu)^2 + gb)  THE WONDER GATE            (deviation from expectation = thoma)
      z    = h * (1 + gate)             amplify the marvellous
      f0   = tanh(z @ Wf^T + bf)        initial fortune state
      for t in 1..T:                    THE WHEEL OF FORTUNE
          e_t   = f_{t-1} . u + u0                    elevation (how high it has risen)
          nem_t = softplus(kappa) * softplus(e_t - theta)   NEMESIS (grows past the threshold)
          f_t   = f_{t-1} + 0.5*tanh(f_{t-1}@A^T + bdyn) - nem_t * f_{t-1}
          y_t   = f_t . wo + bo                       observed fortune at age t
      loss = mean((Y_pred - Y)^2)
    """

    def __init__(self, d_in, n_frames, d_h, d_s, T, seed=0):
        self.d_in, self.K, self.d_h, self.d_s, self.T = d_in, n_frames, d_h, d_s, T
        rng = np.random.default_rng(seed)
        s = 0.5
        self.p = {
            # NOMOS routing (which custom-frames apply to this observation)
            "Wr":  rng.normal(0, s, (n_frames, d_in)),
            "br":  np.zeros(n_frames),
            # NOMOS lenses (each frame's own reading of the same input)
            "We":  rng.normal(0, s, (n_frames, d_h, d_in)),
            "be":  np.zeros((n_frames, d_h)),
            # WONDER gate
            "mu":  rng.normal(0, 0.1, d_h),     # learned expectation
            "gw":  rng.normal(0, 0.3, d_h),     # surprise steepness per dim
            "gb":  np.zeros(d_h),               # gate bias
            # fortune-state projection
            "Wf":  rng.normal(0, s, (d_s, d_h)),
            "bf":  np.zeros(d_s),
            # WHEEL-OF-FORTUNE dynamics
            "A":     rng.normal(0, 0.3, (d_s, d_s)),
            "bdyn":  np.zeros(d_s),
            "u":     rng.normal(0, 0.3, d_s),   # elevation read-out
            "u0":    np.zeros(1),
            "kappa": np.array([0.0]),           # raw nemesis strength (softplus -> >=0)
            "theta": np.array([0.5]),           # hubris threshold
            # fortune read-out
            "wo":  rng.normal(0, s, d_s),
            "bo":  np.zeros(1),
        }

    # ---- forward, returning a cache for backprop -----------------------------
    def forward(self, X, R):
        p = self.p
        N = X.shape[0]
        xe = X * R                                              # (N,d_in)

        scores = xe @ p["Wr"].T + p["br"]                      # (N,K)
        g = softmax_rows(scores)                               # (N,K)

        Hk = np.empty((N, self.K, self.d_h))
        pre_k = np.empty((N, self.K, self.d_h))
        for k in range(self.K):
            pre_k[:, k, :] = xe @ p["We"][k].T + p["be"][k]
            Hk[:, k, :] = np.tanh(pre_k[:, k, :])
        h = np.einsum("nk,nkh->nh", g, Hk)                     # (N,d_h)

        delta = h - p["mu"]                                    # (N,d_h)
        arg = p["gw"] * delta**2 + p["gb"]                     # (N,d_h)
        gate = sigmoid(arg)                                    # (N,d_h)
        z = h * (1.0 + gate)                                   # (N,d_h)

        pre0 = z @ p["Wf"].T + p["bf"]                         # (N,d_s)
        f0 = np.tanh(pre0)                                     # (N,d_s)

        sp_k = softplus(p["kappa"][0])                         # scalar nemesis strength
        fs = [f0]
        es, nems, drifts, tdrifts = [], [], [], []
        Y = np.empty((N, self.T))
        for t in range(self.T):
            f_prev = fs[-1]
            e = f_prev @ p["u"] + p["u0"][0]                   # (N,)
            sp_e = softplus(e - p["theta"][0])                 # (N,)
            nem = sp_k * sp_e                                  # (N,)
            drift = f_prev @ p["A"].T + p["bdyn"]              # (N,d_s)
            tdrift = np.tanh(drift)
            f = f_prev + 0.5 * tdrift - nem[:, None] * f_prev  # (N,d_s)
            y = f @ p["wo"] + p["bo"][0]                       # (N,)
            es.append(e); nems.append(nem); drifts.append(drift); tdrifts.append(tdrift)
            fs.append(f); Y[:, t] = y

        cache = dict(X=X, R=R, xe=xe, scores=scores, g=g, Hk=Hk, pre_k=pre_k,
                     h=h, delta=delta, gate=gate, z=z, pre0=pre0, f0=f0,
                     sp_k=sp_k, fs=fs, es=es, nems=nems, tdrifts=tdrifts, Y=Y)
        return Y, cache

    def loss(self, X, R, Ytrue):
        Y, cache = self.forward(X, R)
        diff = Y - Ytrue
        L = np.mean(diff**2)
        return L, cache, diff

    # ---- analytic backward ---------------------------------------------------
    def backward(self, cache, diff):
        p = self.p
        N, T = diff.shape
        g_ = {k: np.zeros_like(v) for k, v in p.items()}

        dY = (2.0 / (N * T)) * diff                            # (N,T) dL/dY

        # ---- back through the fortune recurrence (BPTT) ----
        fs = cache["fs"]; es = cache["es"]; nems = cache["nems"]; tdr = cache["tdrifts"]
        sp_k = cache["sp_k"]
        sig_k = sigmoid(p["kappa"][0])                         # d softplus(kappa)/d kappa
        gf = np.zeros((N, self.d_s))                           # grad wrt f_t carried back
        for t in reversed(range(T)):
            f_prev = fs[t]                                     # f_{t-1}
            # add this step's output gradient onto f_t
            gf = gf + np.outer(dY[:, t], p["wo"])             # (N,d_s)
            g_["wo"] += fs[t + 1].T @ dY[:, t]
            g_["bo"][0] += dY[:, t].sum()

            nem = nems[t]; e = es[t]; tdrift = tdr[t]
            sp_e = softplus(e - p["theta"][0])
            sig_e = sigmoid(e - p["theta"][0])                # d softplus(e-theta)/d e

            # f_t = f_prev + 0.5*tanh(drift) - nem * f_prev
            # (a) identity term  d f_t/d f_prev = I
            gfp = gf.copy()
            # (b) drift term
            ddrift = 0.5 * (1.0 - tdrift**2) * gf             # (N,d_s)
            g_["A"]    += ddrift.T @ f_prev
            g_["bdyn"] += ddrift.sum(0)
            gfp += ddrift @ p["A"]
            # (c) nemesis scaling term: -nem * f_prev
            gfp += -nem[:, None] * gf                         # via the explicit f_prev factor
            # grad wrt nem (scalar per sample): sum over dims of gf * (-f_prev)
            gnem = np.sum(gf * (-f_prev), axis=1)             # (N,)
            # nem = sp_k * sp_e  ->  d nem/d e = sp_k * sig_e
            de = gnem * sp_k * sig_e                          # (N,)  grad wrt e_t
            # nem params
            g_["kappa"][0] += np.sum(gnem * sp_e * sig_k)     # d nem/d kappa = sp_e * sig_k
            g_["theta"][0] += np.sum(gnem * sp_k * (-sig_e))  # d nem/d theta = sp_k * -sig_e
            # e_t = f_prev . u + u0
            g_["u"]  += de @ f_prev
            g_["u0"][0] += de.sum()
            gfp += np.outer(de, p["u"])
            # carry to previous step
            gf = gfp

        # ---- back through f0 = tanh(pre0), pre0 = z @ Wf^T + bf ----
        gf0 = gf
        dpre0 = (1.0 - cache["f0"]**2) * gf0                   # (N,d_s)
        g_["Wf"] += dpre0.T @ cache["z"]
        g_["bf"] += dpre0.sum(0)
        dz = dpre0 @ p["Wf"]                                   # (N,d_h)

        # ---- back through wonder gate: z = h*(1+gate), gate = sigmoid(gw*delta^2+gb) ----
        h = cache["h"]; gate = cache["gate"]; delta = cache["delta"]
        dgate_arg = (dz * h) * gate * (1.0 - gate)            # (N,d_h)
        g_["gw"] += (dgate_arg * delta**2).sum(0)
        g_["gb"] += dgate_arg.sum(0)
        dh = dz * (1.0 + gate) + dgate_arg * p["gw"] * 2.0 * delta
        g_["mu"] += -(dgate_arg * p["gw"] * 2.0 * delta).sum(0)

        # ---- back through frame mixture: h = sum_k g_k * Hk ----
        Hk = cache["Hk"]; g = cache["g"]; xe = cache["xe"]
        dg = np.einsum("nh,nkh->nk", dh, Hk)                  # (N,K)
        for k in range(self.K):
            dHk = dh * g[:, k][:, None]                       # (N,d_h)
            dpre_k = (1.0 - Hk[:, k, :]**2) * dHk
            g_["We"][k] += dpre_k.T @ xe
            g_["be"][k] += dpre_k.sum(0)
        # softmax jacobian: dscores = g * (dg - sum_k dg*g)
        dscores = g * (dg - (dg * g).sum(1, keepdims=True))
        g_["Wr"] += dscores.T @ xe
        g_["br"] += dscores.sum(0)
        # (xe = X*R is data; no gradient continues past here)
        return g_


# =============================================================================
#  SYNTHETIC HERODOTEAN WORLD
#  Generate polities and the "true" fortune trajectory their zenith implies.
#  The law: prosperity rises, but elevation+hubris summon nemesis -> reversal.
# =============================================================================
PROVENANCE = {"opsis": 1.0, "akoe": 0.5, "gnome": 0.3}   # 2.99 / 7.152 credence tiers

def make_dataset(n, d_in=5, T=6, seed=1):
    """
    Features (d_in=5): [power, wealth, territory, hubris_marker, piety]
    Provenance: power/wealth are seen (opsis), territory heard (akoe),
    hubris/piety are inferred (gnome) -- so they enter with lower credence,
    exactly as Herodotus would weight them.
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, (n, d_in))
    R = np.tile(np.array([PROVENANCE["opsis"], PROVENANCE["opsis"],
                          PROVENANCE["akoe"], PROVENANCE["gnome"],
                          PROVENANCE["gnome"]]), (n, 1))
    Y = np.empty((n, T))
    for i in range(n):
        power, wealth, terr, hubris, piety = X[i]
        zenith = 0.55 * power + 0.30 * wealth + 0.15 * terr      # how high it stands
        # nemesis pressure grows with elevation and hubris, softened by piety
        pressure = max(0.0, zenith + 0.6 * hubris - 0.4 * piety - 0.55)
        f = 0.15 + 0.9 * zenith                                  # initial fortune
        traj = []
        for t in range(T):
            f = f + 0.18 * zenith * (1.0 - f)                    # logistic-ish rise
            f = f - pressure * (f**2)                            # the cut-down (reversal)
            traj.append(f)
        Y[i] = traj
    # standardise targets for stable training
    Y = (Y - Y.mean()) / (Y.std() + 1e-8)
    return X, R, Y


# =============================================================================
#  GRADIENT CHECK  (mandatory; must pass)
# =============================================================================
def gradient_check():
    rng = np.random.default_rng(7)
    d_in, K, d_h, d_s, T, N = 5, 3, 6, 4, 4, 8
    net = HistorNet(d_in, K, d_h, d_s, T, seed=3)
    X = rng.uniform(0, 1, (N, d_in))
    R = rng.uniform(0.3, 1.0, (N, d_in))
    Ytrue = rng.normal(0, 1, (N, T))

    L, cache, diff = net.loss(X, R, Ytrue)
    grads = net.backward(cache, diff)

    eps = 1e-6
    worst = 0.0
    print(f"  {'param':6s}  {'max|rel err|':>12s}")
    for name, val in net.p.items():
        flat = val.ravel()
        gflat = grads[name].ravel()
        idxs = range(flat.size) if flat.size <= 12 else rng.choice(flat.size, 12, replace=False)
        rel_max = 0.0
        for j in idxs:
            orig = flat[j]
            flat[j] = orig + eps
            Lp, _, _ = net.loss(X, R, Ytrue)
            flat[j] = orig - eps
            Lm, _, _ = net.loss(X, R, Ytrue)
            flat[j] = orig
            num = (Lp - Lm) / (2 * eps)
            ana = gflat[j]
            denom = max(1e-12, abs(num) + abs(ana))
            rel_max = max(rel_max, abs(num - ana) / denom)
        worst = max(worst, rel_max)
        print(f"  {name:6s}  {rel_max:12.2e}")
    print(f"\n  WORST relative error across all parameters: {worst:.2e}")
    assert worst < 1e-4, "Gradient check FAILED"
    print("  GRADIENT CHECK PASSED  (analytic gradients match finite differences)\n")
    return worst


# =============================================================================
#  TRAINING  (plain SGD with momentum)
# =============================================================================
def train(net, X, R, Y, epochs=600, lr=0.05, mom=0.9, batch=64, seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    vel = {k: np.zeros_like(v) for k, v in net.p.items()}
    n = X.shape[0]
    hist = []
    for ep in range(epochs):
        idx = rng.permutation(n)
        for s in range(0, n, batch):
            b = idx[s:s + batch]
            L, cache, diff = net.loss(X[b], R[b], Y[b])
            grads = net.backward(cache, diff)
            for k in net.p:
                vel[k] = mom * vel[k] - lr * grads[k]
                net.p[k] += vel[k]
        if ep % max(1, epochs // 10) == 0 or ep == epochs - 1:
            Lfull, _, _ = net.loss(X, R, Y)
            hist.append((ep, Lfull))
            if verbose:
                bar = "#" * int(40 * (1 - min(1.0, Lfull)))
                print(f"  epoch {ep:4d}   MSE {Lfull:7.4f}  |{bar}")
    return hist


# =============================================================================
#  SELF-TESTS
# =============================================================================
def self_tests(net, X, R, Y):
    print("SELF-TESTS")
    print("-" * 60)
    Yp, _ = net.forward(X, R)
    # 1. shapes
    assert Yp.shape == Y.shape
    print(f"  [1] output shape {Yp.shape} matches targets ....... OK")
    # 2. correlation between prediction and truth
    corr = np.corrcoef(Yp.ravel(), Y.ravel())[0, 1]
    print(f"  [2] pred/target correlation = {corr:6.3f} (>0.8) .... {'OK' if corr>0.8 else 'LOW'}")
    assert corr > 0.8
    # 3. NEMESIS test: a high+hubristic polity must reverse harder than a humble one
    hi = np.array([[0.95, 0.95, 0.9, 0.95, 0.05]])  # mighty, proud, impious -> Xerxes-like
    lo = np.array([[0.30, 0.30, 0.3, 0.05, 0.95]])  # modest, pious           -> Tellos-like
    Rr = np.tile(np.array([1, 1, .5, .3, .3]), (1, 1))
    yhi, _ = net.forward(hi, Rr); ylo, _ = net.forward(lo, Rr)
    drop_hi = yhi[0].max() - yhi[0, -1]
    drop_lo = ylo[0].max() - ylo[0, -1]
    print(f"  [3] reversal(proud)={drop_hi:6.3f} > reversal(humble)={drop_lo:6.3f} "
          f"... {'OK' if drop_hi>drop_lo else 'FAIL'}")
    assert drop_hi > drop_lo
    # 4. wonder gate actually amplifies (gate strictly between 0 and 1, z != h)
    _, c = net.forward(X, R)
    assert np.all((c["gate"] > 0) & (c["gate"] < 1)) and not np.allclose(c["z"], c["h"])
    print(f"  [4] wonder gate active (mean gate={c['gate'].mean():.3f}) ...... OK")
    # 5. nomos relativism: frame router spreads weight across >1 frame on average
    spread = (c["g"] > 0.05).sum(1).mean()
    print(f"  [5] avg #active nomos-frames per case = {spread:.2f} (>1) ... {'OK' if spread>1 else 'FLAT'}")
    print("-" * 60 + "\n")


# =============================================================================
#  MAIN
# =============================================================================
if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    print("=" * 70)
    print(" HISTOR-NET  —  the cognitive architecture of Herodotus")
    print(" frame-relative (nomos) | wonder-gated (thoma) | reversal-aware (nemesis)")
    print("=" * 70 + "\n")

    print("STEP 1 — GRADIENT CHECK")
    print("-" * 60)
    gradient_check()

    print("STEP 2 — BUILD THE HERODOTEAN WORLD")
    print("-" * 60)
    d_in, T = 5, 6
    Xtr, Rtr, Ytr = make_dataset(800, d_in=d_in, T=T, seed=1)
    Xte, Rte, Yte = make_dataset(200, d_in=d_in, T=T, seed=99)
    print(f"  train polities: {Xtr.shape[0]}   test polities: {Xte.shape[0]}")
    print(f"  features: power, wealth, territory, hubris, piety")
    print(f"  provenance weights (opsis/akoe/gnome): {Rtr[0]}\n")

    print("STEP 3 — TRAIN  (learn the wheel of fortune)")
    print("-" * 60)
    net = HistorNet(d_in, n_frames=3, d_h=12, d_s=6, T=T, seed=5)
    train(net, Xtr, Rtr, Ytr, epochs=600, lr=0.04, mom=0.9, batch=64)
    Ltr, _, _ = net.loss(Xtr, Rtr, Ytr)
    Lte, _, _ = net.loss(Xte, Rte, Yte)
    print(f"\n  final train MSE {Ltr:.4f} | test MSE {Lte:.4f}\n")

    print("STEP 4 — SELF-TESTS")
    print("-" * 60)
    self_tests(net, Xte, Rte, Yte)

    print("STEP 5 — A HERODOTEAN DEMONSTRATION")
    print("-" * 60)
    print("  Three polities at their zenith, read by the network:\n")
    demos = {
        "Croesus of Lydia (rich, proud, warned)":  [0.85, 0.98, 0.70, 0.80, 0.20],
        "Xerxes' Persia   (vast, hubristic)":      [0.97, 0.90, 0.99, 0.95, 0.10],
        "Tellos of Athens (modest, pious, content)":[0.35, 0.30, 0.25, 0.05, 0.95],
    }
    Rdemo = np.array([[1, 1, .5, .3, .3]])
    for name, feat in demos.items():
        y, _ = net.forward(np.array([feat]), Rdemo)
        peak = y[0].max(); end = y[0, -1]
        verdict = "CUT DOWN" if (peak - end) > 0.15 else "endures"
        print(f"  {name:42s}")
        print(f"     fortune over the ages: {np.round(y[0],2)}   ->  {verdict}")
    print("\n  'Call no man happy until he is dead.'  — Solon to Croesus (Hdt. 1.32)")
    print("=" * 70)
