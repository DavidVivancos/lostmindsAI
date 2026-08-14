"""
======================================================================
CHAPTER 0113  --  THE EXEMPLAR ENGINE
A from-scratch cognitive architecture after
Titus Livius (Livy, 59 BC - AD 17), historian of Rome.
======================================================================
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 6 Minds 101 - 120 Available on Amazon https://www.amazon.com/dp/B0HF7G6JJD
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 113: Livy (-59 to -17 BCE)
================================================================================
WHY THIS ARCHITECTURE (and why it is NOT a Transformer)
-------------------------------------------------------
Livy did not model the mind as a truth-machine that weighs rival
testimonies (that is Herodotus), nor as a myth-free forecaster of
political mechanics (that is Thucydides), nor as a mechanical wheel of
inevitable constitutional decay (that is Polybius). Livy's cognitive
signature, stated openly in his Preface, is different and specific:

    "...you behold the lessons of every kind of experience set forth
     as on a conspicuous monument; from these you may choose for
     yourself and for your own state what to IMITATE, from these mark
     for AVOIDANCE what is shameful in the conception and in the result."
                                        -- Livy, Ab Urbe Condita, Praef.

So for Livy the mind is a GALLERY OF EXEMPLA: a small set of canonical
moral prototypes (Cincinnatus who renounced power, Lucretia, the
faction that fell to luxury) distilled out of countless annals. To
judge the present you RETRIEVE the exemplum the present most resembles
-- not by surface facts (which year, which war) but by deep moral
structure -- and you read off its charge: does imitating this pattern
raise a people, or ruin it?

Two forces decide outcomes in Livy: VIRTUS (character), which is
DURABLE and -- crucially -- IMITABLE, a template you can load into your
own conduct; and FORTUNA (chance), which is TRANSIENT and cannot be
chosen. Because virtus is imitable, decline is never a sealed fate: it
is reversible by choosing what to imitate. That single conviction is
the whole engine below.

THE ARCHITECTURE, in its own parts:
  * SituationEncoder      : raw features of a moment -> latent state z_t
  * ExemplarMemory        : M learned prototype vectors (the "exempla"),
                            each with a learned moral valence v_m
  * AnalogicalRetrieval   : structure-mapping match of z_t to prototypes
                            in a learned metric subspace -> attention a_t
                            (this is retrieval over CONSOLIDATED PROTOTYPES,
                             not over cached inputs -- that is the point)
  * VirtusFortunaGate     : splits the retrieved moral charge into a
                            durable character component (virtus) and a
                            transient chance component (fortuna)
  * NarrativeIntegrator   : integrates virtus into a slowly-accumulating
                            "standing" h_t (leaky, persistence lambda),
                            while fortuna only perturbs the observed
                            trajectory y_t = h_t + kappa * fortuna_t
  * ExemplarIntervention  : the Livian move -- given a declining arc,
                            increase emulation of the highest-virtue
                            exemplum and show the corrected trajectory

Everything is pure NumPy, learned by analytic back-propagation through
time. A finite-difference gradient check on every parameter is run at
import/execute time (mandatory). A real training loop fits a synthetic
but principled "history" that encodes Livy's own causal theory, and a
suite of self-tests verifies the mind-specific behaviours.

Author's note on convention: no PyTorch / TF / JAX. NumPy only.
"""

import numpy as np


# =====================================================================
# 0. UTILITIES
# =====================================================================
def set_seed(seed=42):
    """Fix the random seed so every run is reproducible."""
    np.random.seed(seed)


def sigmoid(x):
    """Numerically-stable logistic sigmoid."""
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def softmax_lastaxis(x):
    """Softmax over the final axis, stabilised by subtracting the max."""
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


# =====================================================================
# 1. THE SYNTHETIC "HISTORY" -- a world that obeys Livy's theory
# =====================================================================
# We fabricate polities as sequences of moments. Each moment belongs to
# one of K_TRUE archetypal SITUATION-TYPES (the ground-truth "exempla"):
# e.g. "crisis-met-with-discipline" (high virtue, character-driven),
# "power-renounced" (high virtue), "wealth-influx" (neutral, but a
# gateway), "luxury-after-conquest" (decline, character-driven vice),
# "faction-and-discord" (decline), "windfall/omen" (chance-driven).
#
# Each type k has: a feature centroid mu_k (what the moment LOOKS like),
# a moral CHARGE q_k (raise (+) or ruin (-)), and a VIRTUS FRACTION
# phi_k (how much of that charge is durable character vs transient luck).
#
# The observed trajectory Y is produced by the SAME functional form the
# model uses, with hidden true parameters lambda*, kappa*. The model
# only ever sees the noisy surface features s_t and must recover the
# exempla, their charges, the virtus split, and the integration law.
# That is exactly Livy's task: read the moral structure beneath events.
# =====================================================================
def make_true_world(F=8, K_true=6, seed=7):
    rng = np.random.RandomState(seed)
    mu = rng.randn(K_true, F) * 1.4                      # exempla centroids
    # charges: 3 virtue exempla (+), 2 vice exempla (-), 1 fortune event (~0 char)
    q = np.array([1.6, 1.1, 0.8, -1.4, -0.9, 0.2])[:K_true]
    # virtus fraction: virtue/vice are character-driven (high phi);
    # the fortune event is chance-driven (low phi)
    phi = np.array([0.85, 0.80, 0.55, 0.80, 0.70, 0.15])[:K_true]
    # Markov transitions: polities have runs; wealth can slide to luxury
    Tr = np.array([
        [.55, .15, .15, .05, .05, .05],   # discipline tends to persist
        [.20, .45, .15, .10, .05, .05],   # renunciation
        [.10, .10, .35, .25, .10, .10],   # wealth-influx -> often luxury
        [.05, .05, .10, .55, .20, .05],   # luxury persists (decline lock-in)
        [.10, .05, .10, .25, .45, .05],   # faction
        [.20, .15, .20, .15, .15, .15],   # fortune events scatter
    ])[:K_true, :K_true]
    Tr = Tr / Tr.sum(axis=1, keepdims=True)
    return dict(mu=mu, q=q, phi=phi, Tr=Tr, F=F, K_true=K_true)


def sample_dataset(world, N, T=24, noise=0.35, sigma_f=0.4,
                   lam_true=0.80, kappa_true=0.6, seed=0):
    """Draw N polity-narratives and their observed rise/decline curves."""
    rng = np.random.RandomState(seed)
    mu, q, phi, Tr = world['mu'], world['q'], world['phi'], world['Tr']
    F, K = world['F'], world['K_true']
    S = np.zeros((N, T, F))
    Y = np.zeros((N, T))
    for n in range(N):
        k = rng.randint(K)
        H = 0.0
        for t in range(T):
            S[n, t] = mu[k] + rng.randn(F) * noise           # surface features
            r = q[k]                                          # true moral charge
            virtus = phi[k] * r                               # durable component
            shock = rng.randn() * sigma_f                     # exogenous fortuna
            fort = (1.0 - phi[k]) * r + shock                 # transient component
            H = lam_true * H + virtus                         # character accrues
            Y[n, t] = H + kappa_true * fort                   # fortune perturbs
            k = rng.choice(K, p=Tr[k])                        # next situation
    return S, Y


# =====================================================================
# 2. THE MODEL
# =====================================================================
class ExemplarEngine:
    """
    Livy's mind as a differentiable machine.

    Parameters (all learned):
      W1 (F,D), b1 (D)      : situation encoder
      P  (M,D)              : the exempla (prototype library)
      v  (M,)               : moral valence of each exemplum
      Wq (D,K), Wk (D,K)    : learned metric for analogical matching
      wg (D,), bg ()        : virtus/fortuna gate
      lam_raw ()            : persistence of standing, lambda = sigmoid(lam_raw)
      kappa ()              : how strongly transient fortune perturbs outcome
    """

    def __init__(self, F=8, D=16, K=8, M=8, seed=42):
        rng = np.random.RandomState(seed)
        s = lambda a, b: rng.randn(a, b) * np.sqrt(2.0 / (a + b))
        self.p = {
            'W1': s(F, D), 'b1': np.zeros(D),
            'P':  rng.randn(M, D) * 0.6,
            'v':  rng.randn(M) * 0.3,
            'Wq': s(D, K), 'Wk': s(D, K),
            'wg': rng.randn(D) * np.sqrt(1.0 / D), 'bg': 0.0,
            'lam_raw': np.array(1.0),          # sigmoid(1.0) ~ 0.73
            'kappa':   np.array(0.5),
        }
        self.F, self.D, self.K, self.M = F, D, K, M

    # ---- forward pass (returns loss, predictions, and a cache) ----------
    def forward(self, S, Y=None):
        p = self.p
        B, T, F = S.shape
        K = self.K
        Z = np.tanh(S @ p['W1'] + p['b1'])                     # (B,T,D)
        Qz = Z @ p['Wq']                                       # (B,T,K)
        Pk = p['P'] @ p['Wk']                                  # (M,K)
        Qz2 = (Qz ** 2).sum(-1, keepdims=True)                 # (B,T,1)
        Pk2 = (Pk ** 2).sum(-1)                                # (M,)
        cross = Qz @ Pk.T                                      # (B,T,M)
        dist2 = Qz2 + Pk2[None, None, :] - 2.0 * cross         # (B,T,M)
        E = -dist2 / np.sqrt(K)                                # (B,T,M)
        A = softmax_lastaxis(E)                                # (B,T,M) analogy
        r = A @ p['v']                                         # (B,T) moral charge
        gp = Z @ p['wg'] + p['bg']                             # (B,T)
        g = sigmoid(gp)                                        # (B,T) virtus share
        virtus = g * r                                         # (B,T) durable
        fort = (1.0 - g) * r                                   # (B,T) transient
        lam = sigmoid(p['lam_raw'])                            # scalar in (0,1)
        kappa = p['kappa']
        h = np.zeros((B, T))
        h[:, 0] = virtus[:, 0]
        for t in range(1, T):
            h[:, t] = lam * h[:, t - 1] + virtus[:, t]         # character accrues
        y = h + kappa * fort                                   # fortune perturbs
        cache = dict(S=S, Z=Z, Qz=Qz, Pk=Pk, dist2=dist2, A=A, r=r,
                     gp=gp, g=g, virtus=virtus, fort=fort, h=h, y=y,
                     lam=lam, kappa=kappa)
        loss = None
        if Y is not None:
            loss = np.mean((y - Y) ** 2)
        return loss, y, cache

    # ---- analytic backward pass (BPTT through the integrator) -----------
    def backward(self, cache, Y):
        p = self.p
        S, Z, Qz, Pk, A = cache['S'], cache['Z'], cache['Qz'], cache['Pk'], cache['A']
        r, g, virtus, fort = cache['r'], cache['g'], cache['virtus'], cache['fort']
        h, y, lam, kappa = cache['h'], cache['y'], cache['lam'], cache['kappa']
        B, T, F = S.shape
        K = self.K
        M = self.M

        dy = 2.0 * (y - Y) / (B * T)                           # dL/dy   (B,T)

        # y = h + kappa*fort
        dkappa = np.sum(dy * fort)
        dfort = dy * kappa                                     # (B,T)
        dh = dy.copy()                                         # direct part of dL/dh_t

        # BPTT: h_t = lam*h_{t-1} + virtus_t   (t>=1)
        for t in range(T - 2, -1, -1):
            dh[:, t] += lam * dh[:, t + 1]
        # lambda gradient: dh_t/dlam = h_{t-1}
        dlam = np.sum(dh[:, 1:] * h[:, :-1])
        dlam_raw = dlam * lam * (1.0 - lam)                    # sigmoid deriv
        dvirtus = dh.copy()                                    # dh_t/dvirtus_t = 1

        # virtus = g*r ; fort = (1-g)*r
        dr = dvirtus * g + dfort * (1.0 - g)                   # (B,T)
        dg = r * (dvirtus - dfort)                             # (B,T)

        # g = sigmoid(gp)
        dgp = dg * g * (1.0 - g)                               # (B,T)
        dwg = np.einsum('btd,bt->d', Z, dgp)
        dbg = np.sum(dgp)
        dZ = dgp[:, :, None] * p['wg'][None, None, :]          # from gate

        # r = A @ v
        dv = np.einsum('btm,bt->m', A, dr)
        dA = dr[:, :, None] * p['v'][None, None, :]            # (B,T,M)

        # softmax backward
        dE = A * (dA - (dA * A).sum(-1, keepdims=True))        # (B,T,M)
        ddist2 = -dE / np.sqrt(K)                              # (B,T,M)

        # dist2 = ||Qz||^2 + ||Pk||^2 - 2 Qz.Pk^T
        D2sum_bt = ddist2.sum(-1)                              # (B,T)
        dQz = 2.0 * Qz * D2sum_bt[:, :, None] - 2.0 * (ddist2 @ Pk)      # (B,T,K)
        D2sum_m = ddist2.sum((0, 1))                           # (M,)
        dPk = 2.0 * Pk * D2sum_m[:, None] - 2.0 * np.einsum('btm,btk->mk', ddist2, Qz)

        # Qz = Z @ Wq ; Pk = P @ Wk
        dWq = np.einsum('btd,btk->dk', Z, dQz)
        dZ += dQz @ p['Wq'].T                                  # add retrieval path
        dWk = p['P'].T @ dPk
        dP = dPk @ p['Wk'].T                                   # (M,D)

        # Z = tanh(S@W1 + b1)
        dpre = dZ * (1.0 - Z ** 2)
        dW1 = np.einsum('btf,btd->fd', S, dpre)
        db1 = dpre.sum((0, 1))

        grads = {
            'W1': dW1, 'b1': db1, 'P': dP, 'v': dv,
            'Wq': dWq, 'Wk': dWk, 'wg': dwg, 'bg': np.array(dbg),
            'lam_raw': np.array(dlam_raw), 'kappa': np.array(dkappa),
        }
        return grads

    # ---- the mind-specific operation: choose what to imitate ------------
    def exemplar_intervention(self, S, strength=0.6):
        """
        Livy's prescription. For each narrative we identify the exemplum
        of highest virtue (max learned valence) and, at the moments where
        the arc is losing standing, we push the analogical attention
        toward emulating it -- i.e. "behold Cincinnatus; imitate him."
        We then recompute the trajectory. A working Livian mind should
        BEND A DECLINING ARC UPWARD when the virtuous exemplum is loaded.
        Returns (baseline_y, intervened_y).
        """
        p = self.p
        _, y0, cache = self.forward(S)
        A = cache['A'].copy()
        virtus, g, r = cache['virtus'], cache['g'], cache['r']
        best = int(np.argmax(p['v']))                          # the model's Cincinnatus
        B, T, M = A.shape
        # blend extra attention onto the virtuous exemplum where standing falls
        dstep = np.diff(cache['h'], axis=1, prepend=cache['h'][:, :1])
        falling = (dstep < 0).astype(float)                    # (B,T) declining moments
        onehot = np.zeros(M); onehot[best] = 1.0
        blend = strength * falling[:, :, None]
        A2 = (1.0 - blend) * A + blend * onehot[None, None, :]
        A2 = A2 / A2.sum(-1, keepdims=True)
        r2 = A2 @ p['v']
        virtus2 = g * r2
        fort2 = (1.0 - g) * r2
        lam = cache['lam']
        h2 = np.zeros((B, T)); h2[:, 0] = virtus2[:, 0]
        for t in range(1, T):
            h2[:, t] = lam * h2[:, t - 1] + virtus2[:, t]
        y2 = h2 + p['kappa'] * fort2
        return y0, y2


# =====================================================================
# 3. ADAM OPTIMIZER (from scratch)
# =====================================================================
class Adam:
    def __init__(self, params, lr=3e-2, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(np.atleast_1d(v).astype(float)) for k, v in params.items()}
        self.v = {k: np.zeros_like(np.atleast_1d(v).astype(float)) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for k in params:
            gk = np.atleast_1d(grads[k]).astype(float)
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * gk
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * gk * gk
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            upd = self.lr * mhat / (np.sqrt(vhat) + self.eps)
            params[k] = params[k] - upd.reshape(np.shape(params[k]))
        return params


# =====================================================================
# 4. GRADIENT CHECK (mandatory) -- analytic vs finite-difference
# =====================================================================
def gradient_check(seed=1, eps=1e-6):
    print("=" * 66)
    print("FINITE-DIFFERENCE GRADIENT CHECK")
    print("=" * 66)
    set_seed(seed)
    world = make_true_world(F=6, K_true=5, seed=3)
    S, Y = sample_dataset(world, N=5, T=7, seed=2)
    model = ExemplarEngine(F=6, D=8, K=5, M=6, seed=9)

    loss, _, cache = model.forward(S, Y)
    grads = model.backward(cache, Y)

    worst = 0.0
    for name in model.p:
        arr = np.atleast_1d(model.p[name]).astype(float)
        flat = arr.reshape(-1)
        gflat = np.atleast_1d(grads[name]).astype(float).reshape(-1)
        idxs = range(len(flat)) if len(flat) <= 6 else \
            np.random.RandomState(name.__hash__() & 0xffff).choice(len(flat), 6, replace=False)
        rels = []
        for i in idxs:
            orig = flat[i]
            flat[i] = orig + eps
            model.p[name] = flat.reshape(np.shape(model.p[name]))
            lp, _, _ = model.forward(S, Y)
            flat[i] = orig - eps
            model.p[name] = flat.reshape(np.shape(model.p[name]))
            lm, _, _ = model.forward(S, Y)
            flat[i] = orig
            model.p[name] = flat.reshape(np.shape(model.p[name]))
            num = (lp - lm) / (2 * eps)
            ana = gflat[i]
            denom = max(1e-12, abs(num) + abs(ana))
            rel = abs(num - ana) / denom
            rels.append(rel)
        mr = max(rels)
        worst = max(worst, mr)
        flag = "OK " if mr < 1e-4 else "!! "
        print(f"  {flag}{name:9s}  max rel err = {mr:.2e}")
    print("-" * 66)
    print(f"  WORST relative error across all parameters: {worst:.3e}")
    assert worst < 1e-4, "Gradient check FAILED"
    print("  GRADIENT CHECK PASSED (analytic gradients are correct).")
    print()
    return worst


# =====================================================================
# 5. TRAIN + SELF-TESTS
# =====================================================================
def train_and_test():
    print("=" * 66)
    print("TRAINING THE EXEMPLAR ENGINE ON A LIVIAN 'HISTORY'")
    print("=" * 66)
    set_seed(42)
    world = make_true_world(F=8, K_true=6, seed=7)
    Str, Ytr = sample_dataset(world, N=256, T=24, seed=10)
    Sva, Yva = sample_dataset(world, N=64,  T=24, seed=99)

    model = ExemplarEngine(F=8, D=16, K=8, M=8, seed=42)
    opt = Adam(model.p, lr=3e-2)

    # baseline: predict the mean of the target
    base_mse = np.mean((Yva - Ytr.mean()) ** 2)
    l0, _, _ = model.forward(Str, Ytr)
    print(f"  train MSE (init) : {l0:.4f}")
    print(f"  val   MSE (predict-mean baseline) : {base_mse:.4f}")

    for step in range(600):
        loss, _, cache = model.forward(Str, Ytr)
        grads = model.backward(cache, Ytr)
        model.p = opt.step(model.p, grads)
        if step % 100 == 0 or step == 599:
            vl, _, _ = model.forward(Sva, Yva)
            print(f"  step {step:4d} | train {loss:.4f} | val {vl:.4f} | "
                  f"lambda={sigmoid(model.p['lam_raw']):.3f} "
                  f"kappa={float(model.p['kappa']):.3f}")

    vl, yva, _ = model.forward(Sva, Yva)
    ss_res = np.sum((Yva - yva) ** 2)
    ss_tot = np.sum((Yva - Yva.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot
    print("-" * 66)
    print(f"  final val MSE : {vl:.4f}   (variance explained R^2 = {r2:.3f})")

    print()
    print("=" * 66)
    print("SELF-TESTS: does the machine behave like Livy's mind?")
    print("=" * 66)

    # TEST 1: it beats the naive baseline decisively
    t1 = vl < 0.5 * base_mse
    print(f"  [1] learns causal structure (val MSE < 0.5x baseline)  : {t1}")

    # TEST 2: the exempla specialise into virtue-exempla and vice-exempla
    vvals = np.sort(model.p['v'])
    spread = vvals[-1] - vvals[0]
    t2 = (vvals[-1] > 0.3) and (vvals[0] < -0.3) and (spread > 1.0)
    print(f"  [2] prototypes polarise into praise/blame exempla      : {t2}")
    print(f"      learned valences (sorted): "
          + ", ".join(f"{x:+.2f}" for x in vvals))

    # TEST 3: recovered persistence lambda tracks the true lambda* (0.80)
    lam = float(sigmoid(model.p['lam_raw']))
    t3 = abs(lam - 0.80) < 0.15
    print(f"  [3] recovers durable-character persistence (lambda~0.80): {t3}"
          f"  (got {lam:.3f})")

    # TEST 4: the Livian intervention bends declining arcs upward
    # take validation narratives that END in decline, apply "imitate the
    # virtuous exemplum", and measure the change in final standing.
    _, yb, _ = model.forward(Sva)
    declining = yb[:, -1] < np.median(yb[:, -1])
    Sd = Sva[declining]
    y0, y2 = model.exemplar_intervention(Sd, strength=0.7)
    lift = np.mean(y2[:, -1] - y0[:, -1])
    frac_up = np.mean(y2[:, -1] > y0[:, -1])
    t4 = (lift > 0) and (frac_up > 0.8)
    print(f"  [4] 'choose what to imitate' raises failing polities    : {t4}")
    print(f"      mean final-standing lift = {lift:+.3f} ; "
          f"share improved = {frac_up*100:.0f}%")

    # TEST 5: virtus/fortuna decomposition is meaningful -- the durable
    # (virtus) channel dominates the final standing, chance perturbs it.
    _, _, c = model.forward(Sva)
    lam = c['lam']
    # reconstruct virtus-only trajectory vs fortune contribution
    B, T = c['h'].shape
    virt_final = np.mean(np.abs(c['h'][:, -1]))
    fort_final = np.mean(np.abs(c['kappa'] * c['fort'][:, -1]))
    t5 = virt_final > fort_final
    print(f"  [5] durable virtus outweighs transient fortuna at the end: {t5}")
    print(f"      |virtus standing|={virt_final:.3f}  "
          f"|fortuna kick|={fort_final:.3f}")

    passed = sum([t1, t2, t3, t4, t5])
    print("-" * 66)
    print(f"  SELF-TESTS PASSED: {passed}/5")
    print("=" * 66)
    return passed, r2


if __name__ == "__main__":
    worst = gradient_check()
    passed, r2 = train_and_test()
    print()
    print("SUMMARY")
    print(f"  gradient check worst rel-err : {worst:.2e}")
    print(f"  self-tests passed            : {passed}/5")
    print(f"  variance explained (val R^2) : {r2:.3f}")
    if worst < 1e-4 and passed >= 4:
        print("  STATUS: the Exemplar Engine is verified and behaves as Livy's mind.")
    else:
        print("  STATUS: needs attention.")
