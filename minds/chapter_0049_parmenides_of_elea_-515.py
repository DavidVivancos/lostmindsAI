#!/usr/bin/env python3
# =============================================================================
# chapter_0049_parmenides_of_elea_-515.py - Parmenides of Elea (c. 515 - c. 460 BCE)
# Architecture: THE ELEATIC SPHERE NETWORK  ("Aletheia-by-Invariance")
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 3 Minds 41 - 60 Available on Amazon https://www.amazon.com/dp/B0H6TVX69S
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 0049 · Parmenides of Elea
# =============================================================================
#
# WHY THIS ARCHITECTURE, AND NOT A TRANSFORMER
# --------------------------------------------
# Parmenides' single, mind-defining idea is NOT "order tames chaos" and NOT
# "build an auditable institution." It is stranger and more specific:
#
#     (1) "to gar auto noein estin te kai einai"  (fr. 3) --
#         to think and to be are THE SAME. A thought with no being as its
#         object is not a faint thought; it is no thought at all.
#     (2) The road "it is not" (he ouk esti) is unthinkable and unsayable
#         (fr. 2). Non-being cannot be represented. Period.
#     (3) What-is is one, ungenerated, indivisible, motionless, complete --
#         "like the bulk of a well-rounded sphere, equally poised in every
#         direction from the center" (fr. 8).
#     (4) Mortals wander a second road, the Way of Opinion (doxa): the senses
#         show plurality, change and motion. That world is not false data to
#         be denied -- it is APPEARANCE to be seen through.
#
# Translated into a learning machine, those four commitments dictate a very
# un-Transformer-like design. A Transformer is, in Parmenidean terms, a
# machine of pure doxa: it samples the next token from a distribution over
# possibilities most of which ARE NOT; it traffics constantly in negation,
# masking, dropout and counterfactuals. Parmenides would call that the
# mechanised babble of the "two-headed" (dikranoi) who hold that a thing both
# is and is not.
#
# So this network does the opposite. Its job is to recover the ONE unchanging
# Being behind the MANY changing appearances of a thing -- with representations
# in which non-being is structurally unreachable:
#
#   * PRESENCE SIMPLEX (softmax): every hidden coordinate is strictly > 0.
#     No feature can ever be truly absent. "Non-being is not."  (commitment 2)
#   * WELL-ROUNDED SPHERE (L2 normalise): the Being-embedding lives on the
#     unit hypersphere -- complete, equally poised in every direction, the
#     origin (the only true "hole") forbidden.  (commitment 3)
#   * INVARIANCE loss: all the sensory appearances (doxa-modes) of one entity
#     are pulled to a SINGLE point -- the One behind the many.  (1, 3)
#   * ALETHEIA decoder: from that single Being-point the network reconstructs
#     the thing's canonical, un-distorted form -- truth as un-concealment.
#   * ELENCHOS / homogeneity term: presence is held near maximum entropy so no
#     part of Being is privileged and no coordinate collapses into a hole -- a
#     reductio against internal division.  (3)
#   * SEPARATION loss: distinct entities stay distinct. This is Patricia Curd's
#     PREDICATIONAL monism, not crude numerical monism: each what-is is a
#     unified whole of a single kind, while many such whole kinds coexist.
#     (Curd, The Legacy of Parmenides, 1998)
#
# The training signal therefore IS Parmenides' epistemology: see through the
# Way of Opinion (modal distortions) to the Way of Truth (the invariant,
# spherical, gapless Being), and let knowledge be the identity of the thought
# (embedding) with what is (the recovered canonical entity).
#
# Engineering contract honoured here:
#   - pure NumPy, from scratch (no autograd, no ML framework);
#   - an analytic backward pass for EVERY parameter;
#   - a finite-difference gradient check that must pass (mandatory);
#   - a real training loop with a from-scratch Adam optimiser;
#   - self-tests asserting the Parmenidean invariants actually hold;
#   - the file executes end to end and prints a verifiable report.
#
# Run:  python3 chapter_0049_parmenides_of_elea_-515.py
# =============================================================================

from __future__ import annotations
import numpy as np

RNG = np.random.default_rng(515)          # seed = Parmenides' (approx) birth year
EPS = 1e-8


# =============================================================================
# 1. THE WORLD OF DOXA  -- synthetic "appearances of a thing"
# =============================================================================
# Each ENTITY is a canonical latent vector c (its "what-is"). We never show c
# to the network directly. Instead we show it through a small FIXED set of
# "modes of appearance" -- the senses, or the systematic distortions of the
# Way of Opinion. Mode s applies a fixed orthogonal twist plus a fixed scaling,
# then adds sensory noise. The network must see THROUGH all S modes to the one
# invariant entity behind them.

def random_orthogonal(d, rng):
    """A random rotation/reflection (orthogonal matrix) via QR."""
    a = rng.standard_normal((d, d))
    q, r = np.linalg.qr(a)
    q *= np.sign(np.diag(r))              # deterministic decomposition
    return q


class DoxaWorld:
    """Generates appearances x = scale_s * (T_s @ c) (+ noise)."""

    def __init__(self, n_entities=24, din=8, n_modes=4, noise=0.05, rng=RNG):
        self.din, self.n_modes, self.noise, self.rng = din, n_modes, noise, rng
        C = rng.standard_normal((n_entities, din))                 # canonical Beings
        self.C = C / (np.linalg.norm(C, axis=1, keepdims=True) + EPS)
        self.modes = [random_orthogonal(din, rng) for _ in range(n_modes)]
        self.scales = [0.6 + 0.8 * rng.random() for _ in range(n_modes)]

    def appearance(self, e, m):
        x = self.scales[m] * (self.modes[m] @ self.C[e])
        return x + self.noise * self.rng.standard_normal(self.din)

    def batch(self, entities, per_entity):
        xs, ts, gids = [], [], []
        for g, e in enumerate(entities):
            for _ in range(per_entity):
                m = int(self.rng.integers(self.n_modes))
                xs.append(self.appearance(e, m))
                ts.append(self.C[e])             # aletheia target = canonical Being
                gids.append(g)
        return np.array(xs), np.array(ts), np.array(gids)


# =============================================================================
# 2. THE ELEATIC SPHERE NETWORK  -- parameters & forward pass
# =============================================================================
# Per appearance x:
#   h1 = tanh(x W1 + b1)        sensory pre-processing
#   z  = h1 W2 + b2             presence logits
#   p  = softmax(z)             PRESENCE SIMPLEX  (strictly > 0)
#   g  = p W3 + b3              pre-embedding
#   e  = g / ||g||              WELL-ROUNDED SPHERE (unit norm)
#   r  = e W4 + b4              ALETHEIA reconstruction of c

class EleaticSphereNet:
    def __init__(self, din=8, hidden=16, presence=12, embed=6, dout=8, rng=RNG):
        self.dims = dict(din=din, hidden=hidden, presence=presence,
                         embed=embed, dout=dout)

        def he(shape):
            return rng.standard_normal(shape) * np.sqrt(2.0 / shape[0])

        self.P = {
            "W1": he((din, hidden)),      "b1": np.zeros(hidden),
            "W2": he((hidden, presence)), "b2": np.zeros(presence),
            "W3": he((presence, embed)),  "b3": np.zeros(embed),
            "W4": he((embed, dout)),      "b4": np.zeros(dout),
        }
        # loss weights -- these ARE the philosophy, dialed in:
        self.lam = dict(rec=1.0, inv=1.0, sep=0.5, pres=0.02)
        self.sep_margin = 0.30

    @staticmethod
    def _softmax(z):
        z = z - z.max(axis=1, keepdims=True)
        ez = np.exp(z)
        return ez / (ez.sum(axis=1, keepdims=True) + EPS)

    def forward(self, X):
        P = self.P
        a1 = np.tanh(X @ P["W1"] + P["b1"])
        z = a1 @ P["W2"] + P["b2"]
        p = self._softmax(z)
        g = p @ P["W3"] + P["b3"]
        n = np.linalg.norm(g, axis=1, keepdims=True) + EPS
        e = g / n
        r = e @ P["W4"] + P["b4"]
        return dict(X=X, a1=a1, z=z, p=p, g=g, n=n, e=e, r=r)

    @staticmethod
    def _group_means(e, gids):
        G = gids.max() + 1
        means = np.zeros((G, e.shape[1]))
        for g in range(G):
            mu = e[gids == g].mean(axis=0)
            means[g] = mu / (np.linalg.norm(mu) + EPS)
        return means                                  # detached unit targets

    # =======================================================================
    # 3. LOSS + ANALYTIC GRADIENTS
    # =======================================================================
    def loss_and_grads(self, X, T, gids, means=None):
        P, lam = self.P, self.lam
        c = self.forward(X)
        e, p, n, a1, r = c["e"], c["p"], c["n"], c["a1"], c["r"]
        B = X.shape[0]
        # The One per entity is a DETACHED target. Freeze it so the
        # analytic gradient (means held constant) matches finite differences.
        if means is None:
            means = self._group_means(e, gids)
        pos = means[gids]

        diff = r - T                                           # L_rec (aletheia)
        L_rec = 0.5 * np.sum(diff ** 2) / B
        dr = lam["rec"] * diff / B

        L_inv = lam["inv"] * np.mean(1.0 - np.sum(e * pos, axis=1))   # the One
        de_inv = lam["inv"] * (-pos) / B

        sims = e @ means.T                                     # L_sep (kinds)
        viol = sims - (-self.sep_margin)
        active = viol > 0
        active[np.arange(B), gids] = False
        n_neg = max(means.shape[0] - 1, 1)
        L_sep = lam["sep"] * np.sum(np.where(active, viol, 0.0)) / (B * n_neg)
        de_sep = lam["sep"] * (active.astype(float) @ means) / (B * n_neg)

        logp = np.log(p + EPS)                                 # L_pres (homogeneity)
        L_pres = lam["pres"] * np.mean(np.sum(p * logp, axis=1))
        dp_pres = lam["pres"] * (logp + 1.0) / B

        L = L_rec + L_inv + L_sep + L_pres

        grads = {k: np.zeros_like(v) for k, v in P.items()}
        grads["W4"] = e.T @ dr
        grads["b4"] = dr.sum(axis=0)
        de = dr @ P["W4"].T + de_inv + de_sep

        ede = np.sum(e * de, axis=1, keepdims=True)            # d through e=g/|g|
        dg = (de - e * ede) / n

        grads["W3"] = p.T @ dg
        grads["b3"] = dg.sum(axis=0)
        dp = dg @ P["W3"].T + dp_pres

        dz = p * (dp - np.sum(p * dp, axis=1, keepdims=True))  # softmax backward
        grads["W2"] = a1.T @ dz
        grads["b2"] = dz.sum(axis=0)
        da1 = dz @ P["W2"].T

        dh1 = da1 * (1.0 - a1 ** 2)                            # tanh backward
        grads["W1"] = X.T @ dh1
        grads["b1"] = dh1.sum(axis=0)

        parts = dict(L_rec=L_rec, L_inv=L_inv, L_sep=L_sep, L_pres=L_pres)
        return L, grads, parts, c

    def loss_only(self, X, T, gids, means):
        # means MUST be the frozen target used by loss_and_grads.
        c = self.forward(X)
        e, p, r = c["e"], c["p"], c["r"]
        B = X.shape[0]
        pos = means[gids]
        L_rec = 0.5 * np.sum((r - T) ** 2) / B
        L_inv = self.lam["inv"] * np.mean(1.0 - np.sum(e * pos, axis=1))
        sims = e @ means.T
        viol = sims - (-self.sep_margin)
        active = viol > 0
        active[np.arange(B), gids] = False
        n_neg = max(means.shape[0] - 1, 1)
        L_sep = self.lam["sep"] * np.sum(np.where(active, viol, 0.0)) / (B * n_neg)
        L_pres = self.lam["pres"] * np.mean(np.sum(p * np.log(p + EPS), axis=1))
        return self.lam["rec"] * L_rec + L_inv + L_sep + L_pres


# =============================================================================
# 4. ADAM OPTIMISER (from scratch)
# =============================================================================
class Adam:
    def __init__(self, params, lr=2e-2, b1=0.9, b2=0.999, eps=1e-8):
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
            params[k] -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


# =============================================================================
# 5. MANDATORY FINITE-DIFFERENCE GRADIENT CHECK
# =============================================================================
def gradient_check(net, X, T, gids, n_probe=6, h=1e-5):
    means = net._group_means(net.forward(X)["e"], gids)   # freeze the One
    _, grads, _, _ = net.loss_and_grads(X, T, gids, means=means)
    worst = 0.0
    for name, W in net.P.items():
        flat = W.ravel()
        idxs = RNG.choice(flat.size, size=min(n_probe, flat.size), replace=False)
        for i in idxs:
            orig = flat[i]
            flat[i] = orig + h; Lp = net.loss_only(X, T, gids, means)
            flat[i] = orig - h; Lm = net.loss_only(X, T, gids, means)
            flat[i] = orig
            num = (Lp - Lm) / (2 * h)
            ana = grads[name].ravel()[i]
            rel = abs(num - ana) / max(1.0, abs(num) + abs(ana))
            worst = max(worst, rel)
    return worst


# =============================================================================
# 6. METRICS THAT ENCODE THE PHILOSOPHY
# =============================================================================
def within_entity_dispersion(net, X, gids):
    """Mean cosine spread of appearances around their One (0 = perfect unity)."""
    e = net.forward(X)["e"]
    means = net._group_means(e, gids)
    return float(np.mean(1.0 - np.sum(e * means[gids], axis=1)))


def between_entity_separation(net, X, gids):
    """Mean angular gap between distinct Beings (higher = more distinct kinds)."""
    e = net.forward(X)["e"]
    means = net._group_means(e, gids)
    G = means.shape[0]
    S = means @ means.T
    return float(1.0 - S[~np.eye(G, dtype=bool)].mean())


# =============================================================================
# 7. MAIN: train, check, self-test, report
# =============================================================================
def main():
    print("=" * 70)
    print(" ELEATIC SPHERE NETWORK  -- Parmenides of Elea (figure 0049)")
    print(" Seeing the One Being through the Many Appearances of Doxa")
    print("=" * 70)

    world = DoxaWorld(n_entities=24, din=8, n_modes=4, noise=0.05)
    net = EleaticSphereNet(din=8, hidden=16, presence=12, embed=6, dout=8)

    entities = list(range(world.C.shape[0]))
    Xtr, Ttr, Gtr = world.batch(entities, per_entity=6)
    Xva, Tva, Gva = world.batch(entities, per_entity=4)        # unseen draws
    print(f"\nWorld : {len(entities)} entities x {world.n_modes} appearance-modes")
    print(f"Train : {Xtr.shape[0]} appearances | Val : {Xva.shape[0]} appearances")

    worst = gradient_check(net, Xtr, Ttr, Gtr)
    print(f"\n[grad-check] worst relative error = {worst:.2e}  "
          f"-> {'PASS' if worst < 1e-4 else 'FAIL'}")
    assert worst < 1e-4, "Gradient check failed."

    d0 = within_entity_dispersion(net, Xva, Gva)
    s0 = between_entity_separation(net, Xva, Gva)
    r0 = 0.5 * np.mean((net.forward(Xva)["r"] - Tva) ** 2)
    print(f"\n[before training]  dispersion(One)={d0:.4f}  "
          f"separation(kinds)={s0:.4f}  recon_err={r0:.4f}")

    opt = Adam(net.P, lr=2e-2)
    EPOCHS = 600
    print("\n[training]")
    for ep in range(1, EPOCHS + 1):
        Xb, Tb, Gb = world.batch(entities, per_entity=6)
        L, grads, parts, _ = net.loss_and_grads(Xb, Tb, Gb)
        opt.step(net.P, grads)
        if ep == 1 or ep % 100 == 0:
            d = within_entity_dispersion(net, Xva, Gva)
            s = between_entity_separation(net, Xva, Gva)
            print(f"  epoch {ep:4d} | L={L:7.4f} "
                  f"| rec={parts['L_rec']:.4f} inv={parts['L_inv']:.4f} "
                  f"sep={parts['L_sep']:.4f} pres={parts['L_pres']:+.4f} "
                  f"| disp={d:.4f} sep*={s:.4f}")

    d1 = within_entity_dispersion(net, Xva, Gva)
    s1 = between_entity_separation(net, Xva, Gva)
    cva = net.forward(Xva)
    r1 = 0.5 * np.mean((cva["r"] - Tva) ** 2)
    print(f"\n[after training]   dispersion(One)={d1:.4f}  "
          f"separation(kinds)={s1:.4f}  recon_err={r1:.4f}")

    print("\n[self-tests : do the doctrines actually hold?]")
    e_all, p_all, r_all = cva["e"], cva["p"], cva["r"]

    t_sphere = np.allclose(np.linalg.norm(e_all, axis=1), 1.0, atol=1e-5)
    print(f"  well-rounded sphere  : all |e|=1               -> {t_sphere}")
    assert t_sphere

    min_presence = float(p_all.min())
    t_nonbeing = min_presence > 0.0
    print(f"  non-being unreachable: min presence={min_presence:.2e}>0  -> {t_nonbeing}")
    assert t_nonbeing

    t_finite = np.all(np.isfinite(e_all)) and np.all(np.isfinite(r_all))
    print(f"  no holes (all finite): no NaN/Inf anywhere     -> {t_finite}")
    assert t_finite

    t_unity = d1 < d0 * 0.5
    print(f"  the One behind many  : dispersion {d0:.3f}->{d1:.3f}  -> {t_unity}")
    assert t_unity

    t_kinds = s1 > s0
    print(f"  distinct kinds kept  : separation {s0:.3f}->{s1:.3f}  -> {t_kinds}")
    assert t_kinds

    t_aletheia = r1 < r0 * 0.5
    print(f"  aletheia (recover c) : recon {r0:.3f}->{r1:.3f}      -> {t_aletheia}")
    assert t_aletheia

    Xnew, _, Gnew = world.batch(entities, per_entity=3)
    d_new = within_entity_dispersion(net, Xnew, Gnew)
    t_robust = d_new < 0.15
    print(f"  see through new doxa : fresh-appearance disp={d_new:.4f} -> {t_robust}")
    assert t_robust

    print("\n" + "=" * 70)
    print(" ALL CHECKS PASSED. The machine reaches one Being through many")
    print(" appearances, forbids non-being, and un-conceals the canonical form.")
    print("=" * 70)


if __name__ == "__main__":
    main()
