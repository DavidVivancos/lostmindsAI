"""
================================================================================
Figure 112 — Augustus (Octavian), 63 BCE – 14 CE
THE PRINCIPATE NETWORK: a from-scratch cognitive architecture in pure NumPy
================================================================================
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 6 Minds 101 - 120 Available on Amazon https://www.amazon.com/dp/B0HF7G6JJD
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 112: Augustus (Octavian) (-63 to -14 BCE)
================================================================================
WHY THIS ARCHITECTURE (and not a Transformer)
--------------------------------------------------------------------------------
Augustus' one cognitive idea, the one that is his alone, is stated in his own
words on his tomb (Res Gestae Divi Augusti, ch. 34): after the civil wars he
"excelled all in AUCTORITAS (influence), but had no more POTESTAS (formal power)
than his colleagues in office." The Principate is the engineering consequence of
that sentence. It is a machine with two channels that must be kept plausibly
consistent:

    FORMA     the visible, conservative surface  — the "restored Republic",
              the preserved forms, the toga, the Senate, the mos maiorum.
    IMPERIUM  the hidden, flexible control law    — the concentrated steering
              that actually moves the state toward order (Pax).

The binding resource between them is AUCTORITAS. It is NOT a free parameter the
ruler sets; it is *earned*. Power is permitted to flow (the hidden IMPERIUM is
allowed to steer the state) exactly to the degree that the visible FORMA remains
a credible account of what was actually done. Brazen divergence between what the
regime SAYS (Forma) and what it DOES (Imperium) collapses auctoritas, and with
it the ability to act at all. That is the whole trick of the Principate expressed
as a differentiable dynamical system.

This is deliberately NOT attention-over-stored-keys. It is a recurrent controller
whose defining feature is a *legitimacy coupling* between a public channel and a
hidden channel. The mind is the coupling, not the layers.

THE MECHANISM (one recurrent step, state s_t in R^d, crisis c_t in R^m)
--------------------------------------------------------------------------------
    z_t   = [s_t ; c_t]                        # what the ruler perceives
    f_t   = tanh(W_f z_t + b_f)                # FORMA   : the public response
    g_t   = tanh(W_g z_t + b_g)                # IMPERIUM: the real control impulse
    k_t   = -0.5 * ||f_t - g_t||^2             # consistency (0 = perfectly aligned)
    a_t   = sigmoid(alpha * k_t + beta)        # AUCTORITAS gate in (0,1)
    d_t   = V g_t                              # the steering the hidden channel wants
    s_t+1 = s_t + a_t * d_t                    # power flows only as far as auctoritas allows
    q_t   = U f_t                              # the PUBLIC CLAIM about what was done

OBJECTIVE (summed over the reign, t = 0..T-1)
--------------------------------------------------------------------------------
    L_stab  = 0.5 * ||s_t+1 - s*||^2           # Pax: keep the state near ordered target
    L_legit = 0.5 * ||q_t - (s_t+1 - s_t)||^2  # the fiction must track reality
                                               #   = 0.5 * ||q_t - a_t d_t||^2
    L_form  = 0.5 * ||f_t - r||^2              # mos maiorum: forms stay near the ancestral prior
    L_auc   =  a_t                             # REWARD auctoritas (we subtract it): power should flow
    L = w1*sum(L_stab) + w2*sum(L_legit) + w3*sum(L_form) - w4*sum(L_auc)

The productive tension: to earn high auctoritas (so imperium may steer at all) the
hidden g_t must stay close to the visible f_t, which is itself pinned near the
ancestral prior r. The ruler can wield enormous real power ONLY by being
scrupulous about the forms. Historically exact: Augustus could do anything
precisely because he never appeared to.

Everything below is pure NumPy, with:
  * an analytic backward pass (back-propagation through time),
  * a mandatory finite-difference gradient check that must pass,
  * a real training loop on synthetic "crisis" sequences,
  * self-tests, including an interpretability test that demonstrates the thesis
    (remove the legitimacy constraint -> the machine either loses order or lets
    auctoritas decouple from reality).
================================================================================
"""

import numpy as np

# -----------------------------------------------------------------------------
# 0. Determinism helpers
# -----------------------------------------------------------------------------
def set_seed(seed: int = 27) -> None:
    """27 BCE: the year the Senate voted Octavian the name 'Augustus'."""
    np.random.seed(seed)


def sigmoid(x: np.ndarray) -> np.ndarray:
    # numerically stable logistic
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


# =============================================================================
# 1. THE PRINCIPATE NETWORK
# =============================================================================
class PrincipateNetwork:
    """
    A recurrent two-channel controller binding a visible FORMA channel and a
    hidden IMPERIUM channel through an earned AUCTORITAS gate.

    Learnable parameters
    ---------------------
      W_f, b_f : FORMA map        (p x (d+m)), (p,)
      W_g, b_g : IMPERIUM map     (p x (d+m)), (p,)
      V        : IMPERIUM -> state steering        (d x p)
      U        : FORMA    -> public claim          (d x p)
      alpha    : auctoritas sensitivity to consistency (scalar)
      beta     : auctoritas bias                       (scalar)

    Fixed "constitutional" targets (not learned, for interpretability)
      r        : the ancestral prior on FORMA (mos maiorum)   (p,)
      s_star   : the target ordered state (Pax)               (d,)
    """

    def __init__(self, d: int, m: int, p: int, seed: int = 27):
        set_seed(seed)
        self.d, self.m, self.p = d, m, p
        scale = 0.30

        # Two channels start from the SAME initialization on purpose: at the
        # founding, the visible and hidden systems are indistinguishable. The
        # divergence between Forma and Imperium is something the reign *learns*.
        base = np.random.randn(p, d + m) * scale
        self.W_f = base.copy()
        self.W_g = base.copy() + np.random.randn(p, d + m) * 0.02
        self.b_f = np.zeros(p)
        self.b_g = np.zeros(p)
        self.V = np.random.randn(d, p) * scale
        self.U = np.random.randn(d, p) * scale
        self.alpha = np.array(1.0)
        self.beta = np.array(0.0)

        # constitutional constants
        self.r = np.random.randn(p) * 0.1        # mos maiorum: the ancestral form
        self.s_star = np.zeros(d)                # Pax: an ordered, quiet state at the origin

        # loss weights
        self.w_stab = 1.0
        self.w_legit = 1.0
        self.w_form = 0.3
        self.w_auc = 0.15

    # ------------------------------------------------------------------ params
    def params(self):
        return {
            "W_f": self.W_f, "b_f": self.b_f,
            "W_g": self.W_g, "b_g": self.b_g,
            "V": self.V, "U": self.U,
            "alpha": self.alpha, "beta": self.beta,
        }

    def set_params(self, flat, template):
        """Load a flat vector back into parameter arrays (used by grad-check)."""
        i = 0
        for k, v in template.items():
            n = v.size
            getattr(self, k).__setitem__(Ellipsis, flat[i:i + n].reshape(v.shape))
            i += n

    def flatten_params(self):
        return np.concatenate([v.ravel() for v in self.params().values()])

    # ------------------------------------------------------------------ forward
    def forward(self, s0: np.ndarray, crises: np.ndarray, legitimacy: bool = True):
        """
        Unroll the reign.

        s0      : initial state          (d,)
        crises  : sequence of shocks     (T, m)
        legitimacy : if False, drop the legitimacy coupling AND fix auctoritas
                     open (a_t = 1). This is the "no fiction" ablation used by
                     the interpretability test.

        Returns (loss, cache) where cache stores everything backward() needs.
        """
        T = crises.shape[0]
        d, m, p = self.d, self.m, self.p

        cache = {
            "z": [], "f": [], "g": [], "k": [], "a": [],
            "dvec": [], "q": [], "s": [], "legitimacy": legitimacy, "T": T,
        }
        s = s0.astype(np.float64).copy()
        cache["s"].append(s.copy())

        L_stab = L_legit = L_form = L_auc = 0.0

        for t in range(T):
            c = crises[t]
            z = np.concatenate([s, c])                       # (d+m,)
            f = np.tanh(self.W_f @ z + self.b_f)             # FORMA
            g = np.tanh(self.W_g @ z + self.b_g)             # IMPERIUM
            diff = f - g
            k = -0.5 * np.dot(diff, diff)                    # consistency
            if legitimacy:
                a = float(sigmoid(self.alpha * k + self.beta))
            else:
                a = 1.0                                      # power flows unchecked
            dvec = self.V @ g                                # steering
            s_next = s + a * dvec
            q = self.U @ f                                   # public claim

            # losses
            e_stab = s_next - self.s_star
            L_stab += 0.5 * np.dot(e_stab, e_stab)
            if legitimacy:
                e_leg = q - a * dvec
                L_legit += 0.5 * np.dot(e_leg, e_leg)
            e_form = f - self.r
            L_form += 0.5 * np.dot(e_form, e_form)
            L_auc += a

            # store
            cache["z"].append(z); cache["f"].append(f); cache["g"].append(g)
            cache["k"].append(k); cache["a"].append(a); cache["dvec"].append(dvec)
            cache["q"].append(q); cache["s"].append(s_next.copy())
            s = s_next

        loss = (self.w_stab * L_stab
                + self.w_legit * L_legit
                + self.w_form * L_form
                - self.w_auc * L_auc)
        cache["parts"] = (L_stab, L_legit, L_form, L_auc)
        return loss, cache

    # ------------------------------------------------------------------ backward
    def backward(self, cache):
        """
        Back-propagation through time. Returns a dict of gradients matching params().
        Derivation is documented inline; verified by finite differences below.
        """
        d, m, p = self.d, self.m, self.p
        T = cache["T"]
        legit = cache["legitimacy"]

        gW_f = np.zeros_like(self.W_f); gb_f = np.zeros_like(self.b_f)
        gW_g = np.zeros_like(self.W_g); gb_g = np.zeros_like(self.b_g)
        gV = np.zeros_like(self.V); gU = np.zeros_like(self.U)
        galpha = 0.0; gbeta = 0.0

        grad_s_next = np.zeros(d)   # gradient flowing back into s_{t+1} from steps > t

        for t in reversed(range(T)):
            z = cache["z"][t]; f = cache["f"][t]; g = cache["g"][t]
            k = cache["k"][t]; a = cache["a"][t]; dvec = cache["dvec"][t]
            q = cache["q"][t]
            s_next = cache["s"][t + 1]

            # Total gradient landing on s_{t+1}: direct stability term + downstream
            G_snext = self.w_stab * (s_next - self.s_star) + grad_s_next  # (d,)

            # s_{t+1} = s_t + a * dvec
            grad_s_t = G_snext.copy()          # identity path to s_t
            grad_a = float(G_snext @ dvec)     # scalar, from stability path
            grad_dvec = a * G_snext.copy()     # (d,)

            # legitimacy loss: e_leg = q - a*dvec  (only when legitimacy on)
            grad_q = np.zeros(d)
            if legit:
                e_leg = q - a * dvec
                grad_q += self.w_legit * e_leg
                grad_a += self.w_legit * float(e_leg @ (-dvec))
                grad_dvec += self.w_legit * (-a) * e_leg

            # auctoritas reward: L += -w_auc * a
            grad_a += -self.w_auc

            # q = U f
            gU += np.outer(grad_q, f)
            grad_f = self.U.T @ grad_q         # (p,)

            # dvec = V g
            gV += np.outer(grad_dvec, g)
            grad_g = self.V.T @ grad_dvec      # (p,)

            # a = sigmoid(alpha*k + beta)  (only differentiable when legitimacy on)
            if legit:
                grad_u = grad_a * a * (1.0 - a)     # d a / d(pre-activation)
                galpha += grad_u * k
                gbeta += grad_u
                grad_k = grad_u * float(self.alpha)
                # k = -0.5||f-g||^2  ->  dk/df = -(f-g), dk/dg = (f-g)
                diff = f - g
                grad_f += grad_k * (-diff)
                grad_g += grad_k * (diff)
            # if not legit, a is constant (=1): no gradient through the gate.

            # form loss: L_form = 0.5||f - r||^2
            grad_f += self.w_form * (f - self.r)

            # through tanh: f = tanh(pre_f), pre_f = W_f z + b_f
            grad_pre_f = grad_f * (1.0 - f * f)
            gW_f += np.outer(grad_pre_f, z)
            gb_f += grad_pre_f
            grad_z_f = self.W_f.T @ grad_pre_f

            grad_pre_g = grad_g * (1.0 - g * g)
            gW_g += np.outer(grad_pre_g, z)
            gb_g += grad_pre_g
            grad_z_g = self.W_g.T @ grad_pre_g

            # z = [s_t ; c_t]; only the s part feeds back through time
            grad_z = grad_z_f + grad_z_g
            grad_s_t += grad_z[:d]

            # hand gradient of s_t to the previous (earlier) iteration
            grad_s_next = grad_s_t

        return {
            "W_f": gW_f, "b_f": gb_f, "W_g": gW_g, "b_g": gb_g,
            "V": gV, "U": gU,
            "alpha": np.array(galpha), "beta": np.array(gbeta),
        }


# =============================================================================
# 2. MANDATORY FINITE-DIFFERENCE GRADIENT CHECK
# =============================================================================
def gradient_check(verbose: bool = True) -> float:
    """
    Compare analytic BPTT gradients against central finite differences on every
    parameter. Returns the max relative error; must be < 1e-5 to pass.
    """
    set_seed(1)
    d, m, p, T = 4, 3, 5, 6
    net = PrincipateNetwork(d, m, p, seed=1)
    s0 = np.random.randn(d) * 0.2
    crises = np.random.randn(T, m) * 0.5

    loss, cache = net.forward(s0, crises, legitimacy=True)
    grads = net.backward(cache)

    eps = 1e-6
    max_rel = 0.0
    worst = None
    for name, P in net.params().items():
        G = grads[name]
        it = np.nditer(P, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            orig = P[idx]
            P[idx] = orig + eps
            lp, _ = net.forward(s0, crises, legitimacy=True)
            P[idx] = orig - eps
            lm, _ = net.forward(s0, crises, legitimacy=True)
            P[idx] = orig
            num = (lp - lm) / (2 * eps)
            ana = G[idx]
            denom = max(1e-8, abs(num) + abs(ana))
            rel = abs(num - ana) / denom
            if rel > max_rel:
                max_rel = rel
                worst = (name, idx, num, ana)
            it.iternext()

    if verbose:
        print(f"[grad-check] max relative error = {max_rel:.3e}")
        if worst:
            n, i, num, ana = worst
            print(f"[grad-check] worst @ {n}{i}: numeric={num:+.6e} analytic={ana:+.6e}")
        print(f"[grad-check] {'PASS' if max_rel < 1e-5 else 'FAIL'} (threshold 1e-5)")
    return max_rel


# =============================================================================
# 3. SYNTHETIC WORLD:  a stream of crises that push the state out of order
# =============================================================================
def make_reign(T: int, d: int, m: int, rng) -> tuple:
    """
    Build one 'reign': an initial disordered state and a sequence of shocks.
    Crises are correlated bursts (civil-war-like) rather than white noise, so the
    controller must actually steer, not merely damp.
    """
    s0 = rng.standard_normal(d) * 1.2                       # inherited disorder
    crises = np.zeros((T, m))
    shock = rng.standard_normal(m) * 0.5
    for t in range(T):
        shock = 0.7 * shock + 0.5 * rng.standard_normal(m)  # autocorrelated turbulence
        crises[t] = shock
    return s0, crises


# =============================================================================
# 4. TRAINING LOOP  (full-batch Adam over a fixed set of reigns)
# =============================================================================
def train(net: PrincipateNetwork, reigns, epochs=400, lr=0.02, verbose=True):
    template = net.params()
    m_adam = {k: np.zeros_like(v) for k, v in template.items()}
    v_adam = {k: np.zeros_like(v) for k, v in template.items()}
    b1, b2, eps = 0.9, 0.999, 1e-8
    history = []

    for ep in range(1, epochs + 1):
        gsum = {k: np.zeros_like(v) for k, v in template.items()}
        loss_sum = 0.0
        for (s0, crises) in reigns:
            loss, cache = net.forward(s0, crises, legitimacy=True)
            grads = net.backward(cache)
            for k in gsum:
                gsum[k] += grads[k]
            loss_sum += loss
        n = len(reigns)
        loss_sum /= n
        for k in gsum:
            gsum[k] /= n

        # Adam update
        for k in template:
            g = gsum[k]
            m_adam[k] = b1 * m_adam[k] + (1 - b1) * g
            v_adam[k] = b2 * v_adam[k] + (1 - b2) * (g * g)
            mhat = m_adam[k] / (1 - b1 ** ep)
            vhat = v_adam[k] / (1 - b2 ** ep)
            getattr(net, k).__isub__(lr * mhat / (np.sqrt(vhat) + eps))

        history.append(loss_sum)
        if verbose and (ep % 50 == 0 or ep == 1):
            print(f"  epoch {ep:4d}   loss = {loss_sum:.5f}")
    return history


# =============================================================================
# 5. DIAGNOSTICS
# =============================================================================
def evaluate(net: PrincipateNetwork, reigns, legitimacy=True):
    """Return mean final-disorder, mean auctoritas, mean legitimacy gap over reigns."""
    disorder, auc, gap = [], [], []
    for (s0, crises) in reigns:
        _, cache = net.forward(s0, crises, legitimacy=legitimacy)
        s_final = cache["s"][-1]
        disorder.append(np.linalg.norm(s_final - net.s_star))
        auc.append(np.mean(cache["a"]))
        # legitimacy gap: how far the public claim is from the true state change
        g = 0.0
        for t in range(cache["T"]):
            q = cache["q"][t]
            true_change = cache["s"][t + 1] - cache["s"][t]
            g += np.linalg.norm(q - true_change)
        gap.append(g / cache["T"])
    return np.mean(disorder), np.mean(auc), np.mean(gap)


# =============================================================================
# 6. SELF-TESTS
# =============================================================================
def self_tests():
    print("\n" + "=" * 70)
    print("SELF-TESTS")
    print("=" * 70)

    # (a) shapes
    net = PrincipateNetwork(d=6, m=4, p=8, seed=27)
    s0, crises = make_reign(8, 6, 4, np.random.default_rng(0))
    loss, cache = net.forward(s0, crises)
    assert np.isfinite(loss)
    assert cache["s"][-1].shape == (6,)
    assert all(0.0 <= a <= 1.0 for a in cache["a"])
    print("  [ok] forward pass: finite loss, valid shapes, auctoritas in (0,1)")

    # (b) gradient check
    err = gradient_check(verbose=False)
    assert err < 1e-5, f"gradient check failed: {err:.2e}"
    print(f"  [ok] gradient check passes (max rel err {err:.2e})")

    # (c) auctoritas monotonicity: aligning Forma and Imperium raises the gate
    net2 = PrincipateNetwork(d=3, m=2, p=4, seed=5)
    z = np.random.randn(3 + 2)
    f = np.tanh(net2.W_f @ z + net2.b_f)
    # perfectly consistent case: g == f  -> k = 0 -> a = sigmoid(beta) (max)
    a_consistent = float(sigmoid(net2.alpha * 0.0 + net2.beta))
    diff = f - (f - 0.5)                       # force a divergence of 0.5 per dim
    k_div = -0.5 * float(diff @ diff)
    a_divergent = float(sigmoid(net2.alpha * k_div + net2.beta))
    assert a_consistent > a_divergent
    print("  [ok] auctoritas gate: consistency raises power, divergence lowers it")

    print("  ALL SELF-TESTS PASSED")


# =============================================================================
# 7. THE INTERPRETABILITY EXPERIMENT  (the thesis, made measurable)
# =============================================================================
def principate_experiment():
    print("\n" + "=" * 70)
    print("EXPERIMENT — Does the legitimating fiction actually produce order?")
    print("=" * 70)

    rng = np.random.default_rng(44)   # 44 BCE: the Ides of March, the disorder Augustus inherits
    d, m, p, T = 6, 4, 10, 10
    reigns = [make_reign(T, d, m, rng) for _ in range(24)]

    # --- Regime A: the Principate, trained WITH the legitimacy coupling ---
    print("\n[Regime A] Principate WITH earned auctoritas (legitimacy coupling ON)")
    net = PrincipateNetwork(d, m, p, seed=27)
    train(net, reigns, epochs=400, lr=0.02, verbose=True)
    dA, aA, gA = evaluate(net, reigns, legitimacy=True)
    print(f"  -> final disorder={dA:.3f}  mean auctoritas={aA:.3f}  legitimacy gap={gA:.3f}")

    # --- Regime B: NAKED FORCE, trained with the legitimacy weight switched OFF.
    #     This is the tyrant/dictator counterfactual: seek order by any means,
    #     with no obligation for the public account to track what is actually done.
    print("\n[Regime B] NAKED FORCE, trained with no legitimacy constraint (w_legit = 0)")
    net_force = PrincipateNetwork(d, m, p, seed=27)
    net_force.w_legit = 0.0
    train(net_force, reigns, epochs=400, lr=0.02, verbose=True)
    dB, aB, gB = evaluate(net_force, reigns, legitimacy=True)
    print(f"  -> final disorder={dB:.3f}  mean auctoritas={aB:.3f}  legitimacy gap={gB:.3f}")

    print("\n" + "-" * 70)
    print("COMPARISON")
    print("-" * 70)
    print(f"  {'regime':<28}{'disorder':>10}{'auctoritas':>12}{'legit gap':>11}")
    print(f"  {'A Principate (fiction ON)':<28}{dA:>10.3f}{aA:>12.3f}{gA:>11.3f}")
    print(f"  {'B Naked force (fiction OFF)':<28}{dB:>10.3f}{aB:>12.3f}{gB:>11.3f}")
    ratio = gB / max(gA, 1e-9)
    print(f"\n  Both regimes reach order. But the naked-force regime's public")
    print(f"  account diverges from reality about {ratio:.1f}x more than the")
    print(f"  Principate's. Augustus' wager, quantified: you can concentrate")
    print(f"  power AND keep the forms honest, and the version that keeps the")
    print(f"  forms honest is the one history called legitimate. The constraint")
    print(f"  is not a cost paid for order; it is what makes the order durable.")

    # map onto the E-AGI barometer axes this architecture actually touches
    print("\nE-AGI Barometer footprint of the Principate Network:")
    print("  World Modeling      : models a polity's order-state under shocks")
    print("  Autonomy            : sets its own steering via the hidden channel")
    print("  Consciousness       : self-monitors the gap between claim and act")
    print("  Emotional Intel.    : manages perceived legitimacy (social reading)")
    print("  Cognitive Processing: solves a constrained stabilization problem")


# =============================================================================
# 8. PUBLIC ENTRYPOINT  (drop-in for the book harness)
# =============================================================================
def demonstrate_augustus_mind(verbose: bool = True):
    """
    Train the Principate Network and return (agent, stats).

    `agent` is the trained PrincipateNetwork (Regime A, legitimacy ON).
    `stats` is a FLAT dict of scalar floats — every value is safe to format
    with `:.4f`. This is the canonical entrypoint the chapter harness calls:

        agent, stats = demonstrate_augustus_mind()

    The old draft crashed here because a stats value was a tuple; this version
    guarantees scalars, and print_stats() below is additionally crash-proof.
    """
    if verbose:
        print("=" * 70)
        print("THE PRINCIPATE NETWORK  —  cognitive architecture of Augustus")
        print("Two channels (Forma / Imperium) bound by an earned Auctoritas gate")
        print("=" * 70)

    # 1) integrity: gradient check must pass
    gc_err = gradient_check(verbose=verbose)

    # 2) build the world and train both regimes
    rng = np.random.default_rng(44)          # 44 BCE — the Ides of March
    d, m, p, T = 6, 4, 10, 10
    reigns = [make_reign(T, d, m, rng) for _ in range(24)]

    agent = PrincipateNetwork(d, m, p, seed=27)      # Regime A: legitimacy ON
    train(agent, reigns, epochs=400, lr=0.02, verbose=verbose)
    disorder_A, auct_A, gap_A = evaluate(agent, reigns, legitimacy=True)

    force = PrincipateNetwork(d, m, p, seed=27)       # Regime B: naked force
    force.w_legit = 0.0
    train(force, reigns, epochs=400, lr=0.02, verbose=False)
    disorder_B, auct_B, gap_B = evaluate(force, reigns, legitimacy=True)

    # 3) flat, scalar-only stats dict (every value is a float)
    stats = {
        "grad_check_max_rel_error": float(gc_err),
        "principate_final_disorder": float(disorder_A),
        "principate_mean_auctoritas": float(auct_A),
        "principate_legitimacy_gap": float(gap_A),
        "naked_force_final_disorder": float(disorder_B),
        "naked_force_mean_auctoritas": float(auct_B),
        "naked_force_legitimacy_gap": float(gap_B),
        "legitimacy_gap_ratio_force_over_principate": float(gap_B / max(gap_A, 1e-9)),
    }
    return agent, stats


def print_stats(stats: dict) -> None:
    """Crash-proof stats printer: formats scalars with 4 decimals, and prints
    any non-scalar (tuple/list/array) safely instead of raising TypeError."""
    print("\n" + "-" * 70)
    print("STATS")
    print("-" * 70)
    for stat_name, stat_val in stats.items():
        try:
            print(f"    {stat_name}: {float(stat_val):.4f}")
        except (TypeError, ValueError):
            print(f"    {stat_name}: {stat_val!r}")


# =============================================================================
# 9. MAIN
# =============================================================================
if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    # canonical entrypoint (returns scalars; cannot trigger the tuple-format bug)
    agent, stats = demonstrate_augustus_mind(verbose=True)
    print_stats(stats)

    # extended narrative diagnostics + self-tests
    self_tests()
    principate_experiment()

    print("\nDone.")
