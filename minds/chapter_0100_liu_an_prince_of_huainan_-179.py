"""
================================================================================
chapter_0100_liu_an_prince_of_huainan_-179.py  THE GANYING ENGINE  —  a resonance-field architecture after Liu An (179-122 BCE)
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 5 Minds 81 - 100 Available on Amazon https://www.amazon.com/dp/B0H7LP5LP2
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 0100 · Liu An, Prince of Huainan
================================================================================
Figure 100  ·  Liu An, Prince of Huainan  ·  patron-compiler of the *Huainanzi*
(淮南鴻烈, c. 139 BCE), the early-Han compendium whose operative principle is
*ganying* 感應 — "stimulus-and-response," cosmic resonance.

WHY THIS ARCHITECTURE (and not attention-over-stored-keys)
----------------------------------------------------------
The *Huainanzi* does not model the mind as a clerk who retrieves rules from a
ledger. It models it as a FIELD that resonates. Its signature image (Lanming,
chapter 6): strike the gong note on one zither and the matching string on a
second, untouched zither sings back across the room. Knowing, for Liu An, is not
lookup — it is *attunement*: things of the same kind (lei 類) entrain one another
through qi 氣.

Two tiers of resonance are named in the text (LeBlanc 1985):
  * RELATIVE resonance — like answers like, *within* a category.
  * TOTAL resonance    — a rarer field reaching *across* all categories
                         (the text calls it cheng 誠 / love).
And one principle of action — *wuwei* 無為: the sage does not push every string.
He finds the ROOT (ben 本) and plucks there; the branches (mo 末) follow.

So the model below is, deliberately, a network of COUPLED PHASE OSCILLATORS
(a learnable Kuramoto field), trained from scratch in pure NumPy:

  COUPLING   K_ij = a·(c_i · c_j) + g , masked off-diagonal (no self-coupling)
             a·(c_i·c_j) is RELATIVE resonance (category embeddings);
             g is the global root field — TOTAL resonance reaching every node.
  DYNAMICS   theta_i <- theta_i + h*( w_i + (1/N) Sum_j K_ij sin(theta_j-theta_i) + u_i )
  READOUT    per-category coherence r_k = |mean e^{i*theta} over the strings of kind k|
             logits = W*r + b           (which family of strings sang back?)
  WUWEI      u is a gentle standing driver (the root). A controller learns where
             to spend its small force budget so the branches self-organize.

The readout reads ROTATION-INVARIANT coherence, because absolute phase is
meaningless in a resonance field — only whether kindred strings move *together*
is meaningful. The task is SYMPATHETIC COMPLETION: only a SEED of one kind's
strings is struck into a chord; the rest of that kind, and every other kind,
start scattered. The single way for that kind's coherence to rise is for the
struck seed to pull its silent kin into phase — like answering like across the
gap. A field that scatters them destroys the only signal there is, so the
resonance mechanism is not a shortcut here — it is the answer. (Self-test [d]
confirms the silent kin swing toward the seed while foreign strings do not.)

Pure NumPy. No autograd library. A finite-difference gradient check (mandatory)
verifies the hand-derived backprop-through-time. Run:  `python3 0100_liu_an_Neuron.py`
================================================================================
"""

from __future__ import annotations
import numpy as np

RNG = np.random.default_rng(122)  # seed = year of Liu An's death, 122 BCE
EPS = 1e-6


# ==============================================================================
# 1.  THE FIELD  —  parameters, coupling, and the resonance dynamics
# ==============================================================================

class GanyingField:
    """A learnable field of N coupled phase oscillators (a Kuramoto network).

    Trainable parameters
    ---------------------
      C : (N, d)  category embeddings.  c_i.c_j is how strongly node i and j are
                  "of the same kind" (lei).  This *is* relative resonance.
      a : scalar  gain on relative (within-category) resonance.
      g : scalar  the global root field -- total resonance reaching every node.
      w : (N,)    each oscillator's natural frequency (its own pitch/inclination).
      W : (Kc,Kc) readout over the per-category coherence vector.
      b : (Kc,)   readout bias.
    """

    def __init__(self, N: int, d: int, Kc: int, T: int = 12, h: float = 0.30):
        assert N % Kc == 0, "N must divide evenly into Kc categories"
        self.N, self.d, self.Kc, self.T, self.h = N, d, Kc, T, h
        self.per = N // Kc
        self.C = RNG.normal(0, 0.30, (N, d))
        self.a = np.array(1.0)
        self.g = np.array(0.0)
        self.w = RNG.normal(0, 0.40, N)        # heterogeneous natural pitches
        self.W = np.eye(Kc) * 2.0 + RNG.normal(0, 0.05, (Kc, Kc))
        self.b = np.zeros(Kc)
        self.offdiag = 1.0 - np.eye(N)
        self.group = np.zeros((Kc, N))         # group[k,i]=1 if node i is kind k
        for k in range(Kc):
            self.group[k, k * self.per:(k + 1) * self.per] = 1.0

    def coupling(self):
        """K_ij = (a*c_i.c_j + g) off the diagonal, 0 on it."""
        return (self.a * (self.C @ self.C.T) + self.g) * self.offdiag

    def _coherence_feats(self, theta):
        """Per-category order parameter r_k (rotation invariant), (B,Kc)."""
        cos, sin = np.cos(theta), np.sin(theta)          # (B,N)
        Cx = cos @ self.group.T / self.per               # (B,Kc) mean cos in group
        Sx = sin @ self.group.T / self.per
        r = np.sqrt(Cx ** 2 + Sx ** 2 + EPS)             # (B,Kc)
        return r, Cx, Sx

    def forward(self, theta0, u=None, cache=False):
        """Unroll T Euler steps of the resonance dynamics, then read coherence.

        theta0 : (B,N)  initial phases -- the *stimulus* (a struck chord).
        u      : driver added each step (wuwei), (N,) / (B,N) / None.
        """
        N, h = self.N, self.h
        K = self.coupling()
        if u is None:
            u = np.zeros((1, N))
        u = np.atleast_2d(u)

        thetas = [theta0.copy()]
        coss = []
        theta = theta0.copy()
        for _ in range(self.T):
            d = theta[:, None, :] - theta[:, :, None]    # (B,N,N) d[b,i,j]=tj-ti
            sin_d, cos_d = np.sin(d), np.cos(d)
            drive = self.w[None, :] + (sin_d * K[None]).sum(2) / N + u
            theta = theta + h * drive
            thetas.append(theta.copy())
            coss.append(cos_d)

        r, Cx, Sx = self._coherence_feats(theta)
        logits = r @ self.W.T + self.b
        if cache:
            self.cache = dict(thetas=thetas, coss=coss, K=K, theta_T=theta,
                              r=r, Cx=Cx, Sx=Sx, u=u, B=theta0.shape[0])
        return logits

    @staticmethod
    def coherence(theta):
        """Global Kuramoto order parameter r in [0,1]; 1 = every string in unison."""
        return np.abs(np.exp(1j * theta).mean(axis=-1))


# ==============================================================================
# 2.  LOSS  +  HAND-DERIVED BACKPROP-THROUGH-TIME  (what the grad-check verifies)
# ==============================================================================

def softmax_xent(logits, y):
    z = logits - logits.max(1, keepdims=True)
    p = np.exp(z); p /= p.sum(1, keepdims=True)
    B = logits.shape[0]
    loss = -np.log(p[np.arange(B), y] + 1e-12).mean()
    d = p.copy(); d[np.arange(B), y] -= 1.0
    return loss, d / B


def backward(field: GanyingField, y):
    """Manual reverse-mode gradients for every parameter, via BPTT.

    Per step:  S_i = w_i + (1/N) Sum_j K_ij sin(theta_j-theta_i) + u_i,  theta+ = theta + h*S.
    The adjoint dtheta is carried back through all T steps by the Jacobian
    d(theta+)/d(theta); parameter grads accumulate; K's grad is chained into
    C, a, g because K is *assembled* from them.
    """
    c = field.cache
    N, h, T = field.N, field.h, field.T
    K, thetas, coss = c["K"], c["thetas"], c["coss"]
    theta_T, r, Cx, Sx, B = c["theta_T"], c["r"], c["Cx"], c["Sx"], c["B"]

    # -- readout backward (through per-category coherence features) -------------
    logits = r @ field.W.T + field.b
    loss, dlogits = softmax_xent(logits, y)              # (B,Kc)
    gW = dlogits.T @ r                                   # (Kc,Kc)
    gb = dlogits.sum(0)
    dr = dlogits @ field.W                               # (B,Kc)
    dCx = dr * Cx / r                                    # r=sqrt(Cx^2+Sx^2+eps)
    dSx = dr * Sx / r
    dcos = (dCx @ field.group) / field.per              # (B,N)
    dsin = (dSx @ field.group) / field.per
    dtheta = -dcos * np.sin(theta_T) + dsin * np.cos(theta_T)   # (B,N)

    gw = np.zeros(N)
    gK = np.zeros((N, N))
    for t in reversed(range(T)):
        cos_d = coss[t]
        sin_d = np.sin(thetas[t][:, None, :] - thetas[t][:, :, None])
        dS = h * dtheta
        gw += dS.sum(0)                                  # dS_i/dw_i = 1
        gK += np.einsum('bi,bij->ij', dS, sin_d) / N     # dS_i/dK_ij = sin(..)/N
        KC = K[None] * cos_d
        off = np.einsum('bi,bik->bk', dtheta, KC) / N    # dS_i/dtheta_k, k!=i
        diag = -(dtheta * KC.sum(2) / N)                 # dS_i/dtheta_i
        dtheta = dtheta + h * (off + diag)

    Gm = gK * field.offdiag
    CCt = field.C @ field.C.T
    ga = np.sum(Gm * CCt)
    gg = np.sum(Gm)
    gC = field.a * ((Gm + Gm.T) @ field.C)

    grads = dict(C=gC, a=np.array(ga), g=np.array(gg), w=gw, W=gW, b=gb)
    return loss, grads


def _bptt_u_only(field: GanyingField, dtheta_T):
    """Lightweight BPTT propagating only the adjoint needed for du (wuwei)."""
    c = field.cache
    N, h, T = field.N, field.h, field.T
    K, thetas, coss, B = c["K"], c["thetas"], c["coss"], c["B"]
    dtheta = dtheta_T.copy()
    gu = np.zeros((B, N))
    for t in reversed(range(T)):
        cos_d = coss[t]
        dS = h * dtheta
        gu += dS
        KC = K[None] * cos_d
        off = np.einsum('bi,bik->bk', dtheta, KC) / N
        diag = -(dtheta * KC.sum(2) / N)
        dtheta = dtheta + h * (off + diag)
    return gu


# ==============================================================================
# 3.  GRADIENT CHECK  —  mandatory.  Analytic BPTT vs. finite differences.
# ==============================================================================

def gradient_check():
    print("=" * 70)
    print("FINITE-DIFFERENCE GRADIENT CHECK  (analytic BPTT vs numeric)")
    print("=" * 70)
    N, d, Kc, T = 9, 4, 3, 6
    f = GanyingField(N, d, Kc, T=T, h=0.25)
    B = 7
    theta0 = RNG.uniform(-np.pi, np.pi, (B, N))
    y = RNG.integers(0, Kc, B)
    f.forward(theta0, cache=True)
    _, grads = backward(f, y)

    eps = 1e-5
    worst = 0.0
    for name in ["C", "a", "g", "w", "W", "b"]:
        P = getattr(f, name)
        flat = np.atleast_1d(P).ravel()
        gnum = np.zeros_like(flat)
        idxs = list(range(flat.size)) if flat.size <= 12 else list(RNG.choice(flat.size, 12, replace=False))
        for i in idxs:
            o = flat[i]
            flat[i] = o + eps; lp = softmax_xent(f.forward(theta0), y)[0]
            flat[i] = o - eps; lm = softmax_xent(f.forward(theta0), y)[0]
            flat[i] = o
            gnum[i] = (lp - lm) / (2 * eps)
        ga = np.atleast_1d(grads[name]).ravel()
        num, ana = gnum[idxs], ga[idxs]
        rel = np.abs(num - ana) / (np.abs(num) + np.abs(ana) + 1e-9)
        m = float(rel.max()); worst = max(worst, m)
        print(f"  {'OK ' if m < 1e-4 else '!!!'} param {name:<2}  max rel-err = {m:.2e}")
    print("-" * 70)
    ok = worst < 1e-4
    print(f"  WORST relative error = {worst:.2e}   ->   {'PASS' if ok else 'FAIL'}")
    assert ok, "Gradient check FAILED -- backprop and forward disagree."
    return ok


# ==============================================================================
# 4.  RESONANCE-CATEGORIZATION TASK  —  which family of strings sang back?
# ==============================================================================

def make_resonance_data(N, Kc, n, strike_noise=0.30, seed_frac=0.5):
    """Sympathetic completion -- the task is built so resonance must do real work.

    Each sample names a category c. We strike only a SEED of c's strings (about
    half of them) into a shared phase; the OTHER half of c, and every string of
    every other category, start scattered at random phases. Nothing about the
    raw stimulus separates c cleanly -- its initial coherence is only middling.

    The single way for c's *whole-category* coherence to rise above the rest is
    for the struck seed to pull its silent kin into phase: like answering like
    across the gap. A field that couples kindred strings (positive within-kind
    coupling) will complete c's chord and leave the foreign strings inert. A
    field that scatters them (negative coupling) destroys the only signal there
    is. So the resonance mechanism is not a shortcut here -- it is the answer."""
    per = N // Kc
    seed = max(2, int(round(seed_frac * per)))
    theta0 = RNG.uniform(-np.pi, np.pi, (n, N))      # everything scattered...
    y = RNG.integers(0, Kc, n)
    for s in range(n):
        c = y[s]
        ref = RNG.uniform(-np.pi, np.pi)
        base = c * per
        # strike only the first `seed` strings of category c into a chord;
        # the remaining (per-seed) strings of c stay scattered, awaiting their kin
        theta0[s, base:base + seed] = ref + RNG.normal(0, strike_noise, seed)
    return theta0, y


def category_init(N, d, Kc, block=1.0):
    """Warm-start C so each block of nodes leans toward its own kind. The field
    still must learn the coupling gain, the pitches, and the readout."""
    per = N // Kc
    C = RNG.normal(0, 0.10, (N, d))
    for k in range(Kc):
        C[k*per:(k+1)*per, k % d] += block
    return C


def accuracy(field, theta0, y):
    return float((field.forward(theta0).argmax(1) == y).mean())


def train_resonance(epochs=600, lr=0.05, T=14, sigma_w=0.25, block=1.0,
                    seed_frac=0.5, strike_noise=0.30, a_max=0.5, verbose=True):
    if verbose:
        print("\n" + "=" * 70)
        print("TRAINING  —  resonance categorization (does like sing to like?)")
        print("=" * 70)
    N, d, Kc = 24, 6, 3
    f = GanyingField(N, d, Kc, T=T, h=0.30)
    f.C = category_init(N, d, Kc, block=block)
    f.w = RNG.normal(0, sigma_w, N)        # gentle pitch spread so kin can entrain

    Xtr, ytr = make_resonance_data(N, Kc, 400, strike_noise, seed_frac)
    Xte, yte = make_resonance_data(N, Kc, 200, strike_noise, seed_frac)

    keys = ["C", "a", "g", "w", "W", "b"]
    state = {k: [np.zeros_like(np.atleast_1d(getattr(f, k)).astype(float)),
                 np.zeros_like(np.atleast_1d(getattr(f, k)).astype(float))] for k in keys}
    b1, b2, epsA = 0.9, 0.999, 1e-8
    acc0 = accuracy(f, Xte, yte)

    for ep in range(1, epochs + 1):
        f.forward(Xtr, cache=True)
        loss, grads = backward(f, ytr)
        for k in keys:
            g = np.atleast_1d(grads[k]).astype(float)
            m, v = state[k]
            m[:] = b1*m + (1-b1)*g
            v[:] = b2*v + (1-b2)*g*g
            upd = lr * (m/(1-b1**ep)) / (np.sqrt(v/(1-b2**ep)) + epsA)
            P = getattr(f, k)
            if np.ndim(P) == 0:
                setattr(f, k, np.array(P.item() - upd.reshape(-1)[0].item()))
            else:
                P -= upd.reshape(P.shape)
        f.g = np.array(np.clip(float(f.g), -0.15, 0.15))   # keep total resonance gentle
        f.a = np.array(np.clip(float(f.a), 0.0, a_max))     # keep coupling near threshold
        if verbose and (ep == 1 or ep % 100 == 0):
            print(f"  epoch {ep:4d}   loss={loss:.4f}   "
                  f"train_acc={accuracy(f,Xtr,ytr):.3f}   test_acc={accuracy(f,Xte,yte):.3f}")
    if verbose:
        print("-" * 70)
        print(f"  start test_acc = {acc0:.3f}   ->   final test_acc = {accuracy(f,Xte,yte):.3f}")
        print(f"  learned relative-resonance gain a = {float(f.a):+.3f}   "
              f"root-field g = {float(f.g):+.3f}")
    return f, (Xte, yte)


# ==============================================================================
# 5.  WUWEI CONTROL  —  pluck the fewest strings; let the branches follow.
# ==============================================================================

def wuwei_control(field: GanyingField, steps=900, lr=0.20, l1=0.025):
    """The sage's problem: bring a scattered field into order by adding a gentle
    standing push u_i to each oscillator -- paying an L1 price for every unit of
    force used. The controller learns to spend its budget on the ROOT strings and
    lets the within-kind coupling carry the branches the rest of the way. Two
    things follow, both on-thesis: each KIND locks into near-unison internally
    (relative resonance, easy), while GLOBAL unison across all kinds stays out of
    reach (total resonance, rare -- the field resists it). Optimised with the
    same BPTT machinery, no force applied to the dynamics it was not trained on."""
    print("\n" + "=" * 70)
    print("WUWEI CONTROL  —  gentle tuning: each kind locks, the whole resists")
    print("=" * 70)
    N = field.N
    theta0 = RNG.uniform(-np.pi, np.pi, (1, N))
    u = np.zeros(N)
    per, Kc = field.per, field.Kc
    for _ in range(steps):
        field.forward(theta0, u=u[None, :], cache=True)
        thetaT = field.cache["theta_T"][0]                    # (N,)
        # objective: MAXIMISE mean within-kind coherence. Let each kind settle
        # to its own phase; the controller need only nucleate, the coupling
        # carries the branches the rest of the way -- so u stays sparse.
        dtheta = np.zeros(N)
        for k in range(Kc):
            idx = np.arange(k*per, (k+1)*per)
            Cx = np.mean(np.cos(thetaT[idx])); Sx = np.mean(np.sin(thetaT[idx]))
            rk = np.sqrt(Cx*Cx + Sx*Sx) + EPS
            # d(-r_k)/dtheta_i  for i in kind k  (we descend, so push toward +r_k)
            dtheta[idx] = (np.sin(thetaT[idx])*Cx - np.cos(thetaT[idx])*Sx) / (per*rk)
        gu = _bptt_u_only(field, dtheta[None, :])[0]
        u = u - lr * (gu + l1 * np.sign(u))              # + wuwei (L1) penalty
    rT = float(field.coherence(field.forward(theta0, u=u[None, :]))[0])
    # within-category order: how aligned each KIND becomes internally (the
    # branches following the plucked roots). Global unison across kinds is the
    # rarer "total resonance" (cheng) and is expected to stay lower.
    thetaT = field.forward(theta0, u=u[None, :], cache=True)
    r_per = field._coherence_feats(field.cache["theta_T"])[0][0]   # (Kc,)
    within = float(np.mean(r_per))
    order = np.argsort(-np.abs(u))[:3]
    print(f"  driver energy: L1(u)={np.abs(u).sum():.2f}  mean|u|={np.mean(np.abs(u)):.3f}  "
          f"max|u|={np.max(np.abs(u)):.3f}   (a gentle standing push, no shocks)")
    print(f"  within-kind order after tuning: r_kind={within:.3f}  "
          f"(branches fall into line within each kind)")
    print(f"  global unison across kinds:      r={rT:.3f}  "
          f"(the rarer 'total resonance' -- the field resists it by design)")
    print(f"  the 'root' strings (largest pushes): nodes {order.tolist()}  "
          f"u={np.round(u[order],3).tolist()}")
    return u


# ==============================================================================
# 6.  SELF-TESTS  +  MAIN
# ==============================================================================

def self_tests(field, test_set):
    print("\n" + "=" * 70)
    print("SELF-TESTS")
    print("=" * 70)
    Xte, yte = test_set
    acc = accuracy(field, Xte, yte); chance = 1.0 / field.Kc
    print(f"  [a] held-out accuracy {acc:.3f} > chance+0.15 ({chance+0.15:.3f}) : "
          f"{'PASS' if acc > chance + 0.15 else 'FAIL'}")
    assert acc > chance + 0.15
    r_unison = GanyingField.coherence(np.zeros((1, field.N)))[0]
    print(f"  [b] coherence(unison) = {r_unison:.3f} ~ 1.0            : "
          f"{'PASS' if abs(r_unison-1) < 1e-9 else 'FAIL'}")
    assert abs(r_unison - 1) < 1e-9
    K = field.coupling(); per = field.per
    same = np.mean([K[i, j] for c in range(field.Kc)
                    for i in range(c*per, (c+1)*per) for j in range(c*per, (c+1)*per) if i != j])
    cross = np.mean([K[i, j] for i in range(per) for j in range(per, field.N)])
    print(f"  [c] within-kind coupling {same:+.3f} > cross-kind {cross:+.3f}  : "
          f"{'PASS' if same > cross else 'FAIL'}")
    assert same > cross

    # [d] functional ganying -- "strike some strings, the silent kindred answer."
    #     For each held-out strike we split the true category into the struck
    #     SEED and its initially-silent KIN. We measure how aligned the silent
    #     kin are to the seed's phase, before vs after the field settles, and
    #     compare against FOREIGN strings (other categories) as a control. If
    #     like answers like, the kin swing into phase with the seed while the
    #     foreign strings do not.
    per = field.per
    seed = max(2, int(round(0.5 * per)))

    def align_to(theta_row, idx, center):           # mean cos(theta_i - center)
        return float(np.mean(np.cos(theta_row[idx] - center)))

    kin0, kinT, foreignT = [], [], []
    for s in range(len(Xte)):
        x = Xte[s:s+1]; c = int(yte[s]); base = c * per
        seed_idx = np.arange(base, base + seed)
        kin_idx  = np.arange(base + seed, base + per)
        foreign_idx = np.array([i for i in range(field.N)
                                if i < base or i >= base + per])
        field.forward(x, cache=True); thetaT = field.cache["theta_T"][0]
        th0 = x[0]
        cen0 = np.angle(np.mean(np.exp(1j * th0[seed_idx])))
        cenT = np.angle(np.mean(np.exp(1j * thetaT[seed_idx])))
        kin0.append(align_to(th0,   kin_idx,     cen0))
        kinT.append(align_to(thetaT, kin_idx,    cenT))
        foreignT.append(align_to(thetaT, foreign_idx, cenT))
    kin0, kinT, foreignT = map(lambda z: float(np.mean(z)), (kin0, kinT, foreignT))
    pulled_in = kinT - kin0                          # kin swing toward the seed
    vs_foreign = kinT - foreignT                     # kin answer; foreigners don't
    print(f"  [d] silent kin alignment to struck seed: start {kin0:+.3f} -> "
          f"settled {kinT:+.3f}  (pulled in {pulled_in:+.3f})")
    print(f"      settled kin {kinT:+.3f} vs foreign strings {foreignT:+.3f}  "
          f"(ganying margin {vs_foreign:+.3f}) : "
          f"{'PASS' if (pulled_in > 0.15 and vs_foreign > 0.15) else 'FAIL'}")
    assert pulled_in > 0.15 and vs_foreign > 0.15
    print("-" * 70)
    print("  all self-tests PASSED")


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    gradient_check()
    # gradient_check above consumes the shared RNG stream; reseed so the
    # training regime is deterministic and the resonance margins are stable.
    RNG = np.random.default_rng(122)          # 122 BCE -- the year Liu An died
    field, test_set = train_resonance()
    self_tests(field, test_set)
    wuwei_control(field)
    print("\n" + "=" * 70)
    print("Strike one string; the kindred string answers. Tune the root; the")
    print("branches fall into order of themselves.  — the way of the Huainanzi")
    print("=" * 70)
