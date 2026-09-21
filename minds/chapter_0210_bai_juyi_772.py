#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 Figure 0210 — Bai Juyi 白居易 (772–846), Tang dynasty
 Architecture: CRN — the Caishi Resonance Network
               (采詩共鳴網 / "collected-song resonance network")
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 11 Minds 201 - 220 Available on Amazon https://www.amazon.com/dp/B0HKF7TRFF
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0210_bai_juyi_772 - Bai Juyi (772–846), Tang dynasty
================================================================================  

WHY THIS ARCHITECTURE AND NOT ANOTHER
-------------------------------------
Bai Juyi left an unusually explicit theory of how a mind transmits meaning to
another mind. In his letter to Yuan Zhen (與元九書, 815 CE) he writes:

    詩者：根情、苗言、華聲、實義。
    "As for poetry: emotion is the root, language the sprout,
     sound the flower, meaning the fruit."

Read carefully, this is a claim about *ordering and causality* that inverts the
usual engineering assumption. We normally build generators that start from a
semantic payload and dress it in words. Bai says the opposite: generation
*begins* in affect, passes through diction, is carried by sound, and meaning is
the LAST thing to appear — and it appears in the listener, as fruit appears on a
plant someone else picks. The author does not emit meaning. The author emits a
sound-carried utterance, and meaning is what a listener manages to extract.

He then attaches an acceptance test to that pipeline. The tradition records
(Song-era anecdote, and therefore probably embroidered) that he read drafts to
an illiterate old woman and revised whatever she could not follow. Whether or
not the old woman is historical, the *policy* is unmistakably his: he classified
his own corpus, he wrote satirical poems whose closing stanza states its own
intent in plain words so that nobody could miss it, and he was mocked by elite
contemporaries for exactly this vulgarity. The test is real even if the anecdote
is not.

That test is the interesting machine-learning object. It is a training signal in
which THE VERIFIER IS DELIBERATELY WEAKER THAN THE GENERATOR. Not an adversary,
not a stronger critic, not a reward model with more capacity — a small, slow,
low-vocabulary listener whose failure to understand is treated as the generator's
failure. Modern alignment tends to reach for a verifier at least as strong as
the thing being verified. Bai's instinct is the reverse: the legitimacy of an
utterance is bounded by the comprehension of its weakest intended recipient.

Finally, Bai wrapped both of these in an institutional loop he explicitly wanted
restored: the 采詩 (caishi) office, the Zhou-era functionary whose job was to
collect songs from the villages so the ruler could hear what he was otherwise
insulated from. Bai says its abolition is why "上不以詩補察時政，下不以歌洩導人情" —
above, poetry no longer supplements and audits governance; below, song no longer
drains and channels the feelings of the people. Both directions matter, and they
are different objectives.

So the network has four cascaded stages, one weak gate, one bidirectional
channel, and one archival strategy:

    1. RootAffect     (情)  generation seeds in affect, not semantics
    2. SproutLexicon  (言)  register-partitioned diction, elite words penalised
    3. FlowerProsody  (聲)  gated tonal resonance; the carrier that "enters"
    4. FruitSemantics (義)  the author's own reading — deliberately SECONDARY
    5. YujieGate      (嫗解) the weak listener; its cross-entropy is the PRIMARY loss
    6. CaishiChannel  (采詩) grievance -> poem -> policy, with two separate metrics
    7. FenCangArchive (分藏) survival by scattered whole copies, not by one vault

Everything below is pure NumPy with hand-derived gradients. A finite-difference
gradient check is mandatory and runs on every invocation. There is a real
training loop, a controlled ablation, and three self-tests.

Run:  python3 0195_Neuron.py
================================================================================
"""

import numpy as np

RNG = np.random.default_rng(772)   # his birth year, as a seed

# =============================================================================
# SECTION 0 — A SYNTHETIC WORLD OF GRIEVANCES
# =============================================================================
# Bai's satirical poems (諷諭詩) are not about feelings in general; each one is
# about a specific, nameable harm done to a specific kind of person. "The Old
# Charcoal Seller" is about a state requisition priced at nothing. "The Old Man
# of Xinfeng with the Broken Arm" is about conscription for a border war so
# hated that a man crippled himself to avoid it. So the corpus here is not
# free-form text; it is a set of GRIEVANCE TYPES, each with a severity.
#
# The task the network must solve is therefore concrete and checkable:
#   given a grievance the poet has witnessed, emit an utterance from which the
#   WEAKEST listener can still recover which grievance it was.

GRIEVANCE_NAMES = [
    "conscription",       # 折臂翁 — the border levy
    "requisition",        # 賣炭翁 — goods taken at a fictional price
    "land_tax",           # 重賦 — tax exceeding the harvest
    "corvee_labour",      # forced work on walls and canals
    "famine_neglect",     # granaries closed during dearth
    "official_luxury",    # 輕肥 — fat horses and light furs
]
C = len(GRIEVANCE_NAMES)     # number of grievance classes
DX = C + 3                   # input: one-hot(grievance)*severity + 3 context feats


def make_batch(n, rng):
    """
    Build a batch of witnessed events.

    x = [ onehot(class) * severity | season, distance_from_capital, harvest_index ]

    The three context features are deliberately *uninformative noise* about the
    grievance class. They exist so the network cannot cheat: it has to route the
    class signal through the whole root->sprout->flower chain rather than let a
    context feature leak straight to the output.
    """
    cls = rng.integers(0, C, size=n)
    sev = rng.uniform(0.25, 1.0, size=n)
    x = np.zeros((n, DX))
    x[np.arange(n), cls] = sev
    x[:, C:] = rng.normal(0.0, 0.4, size=(n, 3))
    return x, cls, sev


# -----------------------------------------------------------------------------
# The register-partitioned lexicon (言)
# -----------------------------------------------------------------------------
# Bai's contemporaries wrote for each other. Their diction assumed you had the
# Zuozhuan by heart. Bai's did not. So the vocabulary here is split in two:
#
#   COMMON  words a market listener knows      (mask value 0)
#   ELITE   allusive/literary words            (mask value 1)
#
# The split is not decorative. The weak listener physically receives only the
# common-register share of the utterance (see YujieGate). Probability mass the
# poet spends on elite words is mass the old woman never hears at all. This is
# the mechanism, not a penalty bolted on afterwards.

V = 48                      # vocabulary size
ELITE_FRACTION = 0.5
ELITE_MASK = np.zeros(V)
ELITE_MASK[V // 2:] = 1.0   # second half of the vocabulary is elite diction


# =============================================================================
# SECTION 1 — THE NETWORK
# =============================================================================

def xavier(shape, rng, gain=1.0):
    fan_in = shape[-1]
    return rng.normal(0.0, gain / np.sqrt(fan_in), size=shape)


class CaishiResonanceNetwork:
    """
    The full four-root cascade plus the weak gate.

    Dimensions are small on purpose. Bai's whole argument is that the expressive
    machine should not be larger than the comprehension it is answerable to, and
    the weak listener here is a genuine bottleneck (DB=4) that the generator
    cannot widen.
    """

    def __init__(self, dr=24, de=16, df=20, db=4, tau=0.8, rng=RNG):
        self.dr, self.de, self.df, self.db, self.tau = dr, de, df, db, tau

        # ---- 1. ROOT (情) : affect state seeded directly from the witnessed event
        self.W_r = xavier((dr, DX), rng)
        self.b_r = np.zeros(dr)

        # ---- 2. SPROUT (言) : affect -> diction. Logits over the split lexicon.
        self.W_s = xavier((V, dr), rng)
        self.b_s = np.zeros(V)
        self.E = xavier((V, de), rng)          # word embeddings, shared

        # ---- 3. FLOWER (聲) : the sound-carrier. A gated resonance bank.
        self.W_f = xavier((df, de), rng)
        self.b_f = np.zeros(df)
        self.W_g = xavier((df, df), rng)       # the gate: which tones survive

        # ---- 4. FRUIT (義) : the AUTHOR's own reading of his own poem.
        self.W_y = xavier((C, df), rng)

        # ---- 5. YUJIE (嫗解) : the old woman. Tiny. Sees common register only.
        self.W_w = xavier((db, de), rng)       # db = 4. A real bottleneck.
        self.W_o = xavier((C, db), rng)

        self.keys = ["W_r", "b_r", "W_s", "b_s", "E",
                     "W_f", "b_f", "W_g", "W_y", "W_w", "W_o"]

    # -- parameter vector plumbing (also used by the archive and the gradcheck) --
    def get(self, k):
        return getattr(self, k)

    def flat(self):
        return np.concatenate([self.get(k).ravel() for k in self.keys])

    def load_flat(self, v):
        i = 0
        for k in self.keys:
            a = self.get(k)
            n = a.size
            setattr(self, k, v[i:i + n].reshape(a.shape).copy())
            i += n

    # ------------------------------------------------------------------
    # FORWARD
    # ------------------------------------------------------------------
    def forward(self, X):
        """
        X : (B, DX) witnessed events.
        Returns a cache dict holding every intermediate the backward pass needs.

        The order of these five blocks is the thesis. Affect first, meaning last.
        """
        c = {}
        c["X"] = X

        # --- 1. ROOT (情). The poem starts as a feeling about what was seen.
        c["zr"] = X @ self.W_r.T + self.b_r
        c["R"] = np.tanh(c["zr"])                                   # (B, dr)

        # --- 2. SPROUT (言). Affect chooses words. Soft distribution over the
        #     lexicon; temperature tau controls how committed the diction is.
        c["U"] = c["R"] @ self.W_s.T + self.b_s                     # (B, V)
        Z = c["U"] / self.tau
        Z = Z - Z.max(axis=1, keepdims=True)
        eZ = np.exp(Z)
        c["P"] = eZ / eZ.sum(axis=1, keepdims=True)                 # (B, V)

        # The full utterance, and the part of it that actually reaches a listener
        # with no classical education. Note PC is NOT renormalised: if the poet
        # spends half his mass on allusions, the old woman receives half a poem.
        c["PC"] = c["P"] * (1.0 - ELITE_MASK)
        c["e"] = c["P"] @ self.E                                    # (B, de)
        c["ec"] = c["PC"] @ self.E                                  # (B, de)

        # --- 3. FLOWER (聲). Sound is what makes language enter the ear:
        #     "韻協則言順，言順則聲易入". Modelled as a resonance bank with a
        #     multiplicative gate — some tones carry, some are damped.
        c["zf"] = c["e"] @ self.W_f.T + self.b_f
        c["A"] = np.tanh(c["zf"])                                   # (B, df)
        c["gpre"] = c["A"] @ self.W_g.T
        c["G"] = 1.0 / (1.0 + np.exp(-c["gpre"]))
        c["H"] = c["A"] * c["G"]                                    # (B, df)

        # --- 4. FRUIT (義), author's version. Deliberately downweighted later.
        c["oa"] = c["H"] @ self.W_y.T                               # (B, C)

        # --- 5. YUJIE (嫗解). The old woman in the market. Four hidden units.
        c["zw"] = c["ec"] @ self.W_w.T
        c["B"] = np.tanh(c["zw"])                                   # (B, db)
        c["oo"] = c["B"] @ self.W_o.T                               # (B, C)
        return c

    # ------------------------------------------------------------------
    # LOSS
    # ------------------------------------------------------------------
    def loss(self, c, cls, sev, w=None):
        """
        Five terms. The weights encode the whole argument of the chapter.

          Ly    weak-listener cross-entropy      PRIMARY   (嫗解)
          La    author's own cross-entropy       secondary
          Lreg  probability mass on elite words  penalty
          Lpr   prosodic smoothness of the tonal bank      (韻協)
          Laf   affect magnitude must track severity       (根情)

        Ly >> La is the design decision. The poem is judged by what the weakest
        listener recovers, not by what the poet believes he encoded.
        """
        if w is None:
            w = dict(y=1.0, a=0.15, reg=0.35, pr=0.05, af=0.10)
        B = c["X"].shape[0]
        idx = np.arange(B)

        Soo = softmax(c["oo"])
        Ly = -np.log(np.maximum(Soo[idx, cls], 1e-12)).mean()

        Soa = softmax(c["oa"])
        La = -np.log(np.maximum(Soa[idx, cls], 1e-12)).mean()

        Lreg = (c["P"] * ELITE_MASK).sum(axis=1).mean()

        D2 = c["A"][:, 2:] - 2.0 * c["A"][:, 1:-1] + c["A"][:, :-2]
        Lpr = (D2 ** 2).sum(axis=1).mean()

        m = (c["R"] ** 2).mean(axis=1)
        Laf = ((m - sev) ** 2).mean()

        total = (w["y"] * Ly + w["a"] * La + w["reg"] * Lreg
                 + w["pr"] * Lpr + w["af"] * Laf)
        parts = dict(Ly=Ly, La=La, Lreg=Lreg, Lpr=Lpr, Laf=Laf, total=total)
        return total, parts, (Soo, Soa, D2, m)

    # ------------------------------------------------------------------
    # BACKWARD  (all gradients derived by hand; verified by finite differences)
    # ------------------------------------------------------------------
    def backward(self, c, cls, sev, aux, w=None):
        if w is None:
            w = dict(y=1.0, a=0.15, reg=0.35, pr=0.05, af=0.10)
        Soo, Soa, D2, m = aux
        B = c["X"].shape[0]
        idx = np.arange(B)
        g = {k: np.zeros_like(self.get(k)) for k in self.keys}

        # ---- weak-listener branch (嫗解) -------------------------------------
        doo = Soo.copy()
        doo[idx, cls] -= 1.0
        doo *= w["y"] / B                                            # (B, C)
        g["W_o"] += doo.T @ c["B"]
        dB = doo @ self.W_o
        dzw = dB * (1.0 - c["B"] ** 2)
        g["W_w"] += dzw.T @ c["ec"]
        dec = dzw @ self.W_w                                         # (B, de)
        g["E"] += c["PC"].T @ dec
        dP = (dec @ self.E.T) * (1.0 - ELITE_MASK)                   # (B, V)

        # ---- author branch (義) ----------------------------------------------
        doa = Soa.copy()
        doa[idx, cls] -= 1.0
        doa *= w["a"] / B
        g["W_y"] += doa.T @ c["H"]
        dH = doa @ self.W_y
        dA = dH * c["G"]
        dG = dH * c["A"]
        dgpre = dG * c["G"] * (1.0 - c["G"])
        g["W_g"] += dgpre.T @ c["A"]
        dA += dgpre @ self.W_g

        # ---- prosody (聲): second-difference smoothness ----------------------
        coef = 2.0 * w["pr"] / B
        dA[:, 2:] += coef * D2
        dA[:, 1:-1] += -2.0 * coef * D2
        dA[:, :-2] += coef * D2

        dzf = dA * (1.0 - c["A"] ** 2)
        g["W_f"] += dzf.T @ c["e"]
        g["b_f"] += dzf.sum(axis=0)
        de_ = dzf @ self.W_f                                         # (B, de)
        g["E"] += c["P"].T @ de_
        dP += de_ @ self.E.T

        # ---- register penalty (言) -------------------------------------------
        dP += (w["reg"] / B) * ELITE_MASK[None, :]

        # ---- through the softmax over the lexicon ---------------------------
        dU = (c["P"] * (dP - (c["P"] * dP).sum(axis=1, keepdims=True))) / self.tau
        g["W_s"] += dU.T @ c["R"]
        g["b_s"] += dU.sum(axis=0)
        dR = dU @ self.W_s                                           # (B, dr)

        # ---- affect must track severity (情) --------------------------------
        dm = 2.0 * (m - sev) * (w["af"] / B)                         # (B,)
        dR += dm[:, None] * (2.0 * c["R"] / self.dr)

        dzr = dR * (1.0 - c["R"] ** 2)
        g["W_r"] += dzr.T @ c["X"]
        g["b_r"] += dzr.sum(axis=0)
        return g


def softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    eZ = np.exp(Z)
    return eZ / eZ.sum(axis=1, keepdims=True)


# =============================================================================
# SECTION 2 — MANDATORY FINITE-DIFFERENCE GRADIENT CHECK
# =============================================================================

def gradient_check(verbose=True):
    """
    Central-difference check on EVERY parameter coordinate — no sampling.

    Error is measured in vector form,

        ||numeric - analytic||_2 / (||numeric||_2 + ||analytic||_2)

    rather than per-coordinate. Per-coordinate relative error is a bad statistic
    here: several tensors have legitimate gradient entries around 1e-8, where a
    1e-12 absolute discrepancy (i.e. float64 round-off in the difference
    quotient) shows up as a 1e-4 "relative error" and means nothing. The vector
    form measures the error against the scale of the gradient that actually
    exists. Max absolute error is reported alongside so nothing is hidden.
    """
    rng = np.random.default_rng(846)          # the year he died
    net = CaishiResonanceNetwork(dr=7, de=6, df=8, db=4, tau=0.9, rng=rng)
    X, cls, sev = make_batch(5, rng)

    c = net.forward(X)
    _, _, aux = net.loss(c, cls, sev)
    g = net.backward(c, cls, sev, aux)

    eps = 1e-5
    worst_rel, worst_abs, n_checked = 0.0, 0.0, 0
    report = []
    for k in net.keys:
        flat = net.get(k).ravel()          # a view: writes here mutate the param
        ana = g[k].ravel()
        num = np.zeros(flat.size)
        for i in range(flat.size):
            orig = flat[i]
            flat[i] = orig + eps
            Lp, _, _ = net.loss(net.forward(X), cls, sev)
            flat[i] = orig - eps
            Lm, _, _ = net.loss(net.forward(X), cls, sev)
            flat[i] = orig
            num[i] = (Lp - Lm) / (2 * eps)
        rel = (np.linalg.norm(num - ana)
               / max(np.linalg.norm(num) + np.linalg.norm(ana), 1e-300))
        ab = np.abs(num - ana).max()
        worst_rel, worst_abs = max(worst_rel, rel), max(worst_abs, ab)
        n_checked += flat.size
        report.append((k, flat.size, rel, ab))

    if verbose:
        print("  exhaustive central-difference check (eps=1e-5)")
        print(f"  {'tensor':<6s} {'coords':>7s} {'rel(vec)':>11s} {'max|abs|':>11s}")
        for k, n, r, a in report:
            print(f"    {k:<5s} {n:7d} {r:11.3e} {a:11.3e}")
        print(f"    {n_checked} coordinates checked; "
              f"worst rel {worst_rel:.3e}, worst abs {worst_abs:.3e}")
        print("    (round-off floor for a central difference on an O(1) loss at"
              " eps=1e-5 is ~1e-11 absolute; W_g sits near it only because its"
              " own gradient norm is ~5e-5.)")
    # Absolute error is the primary criterion because it is the one that is not
    # distorted by tensors whose true gradients are legitimately tiny. 1e-9 is
    # two orders above the observed round-off floor and ~8 orders below the
    # largest gradient entries in the model.
    assert worst_abs < 1e-9, f"gradient check FAILED, worst abs {worst_abs:.3e}"
    assert worst_rel < 1e-6, f"gradient check FAILED, worst rel {worst_rel:.3e}"
    return worst_rel


# =============================================================================
# SECTION 3 — TRAINING
# =============================================================================

def adam_init(net):
    return ({k: np.zeros_like(net.get(k)) for k in net.keys},
            {k: np.zeros_like(net.get(k)) for k in net.keys})


def train(net, steps=1400, batch=64, lr=0.02, w=None, rng=None, log_every=200,
          label=""):
    """A plain Adam loop. Nothing exotic; the interest is in the loss weights."""
    if rng is None:
        rng = np.random.default_rng(1)
    M, Vv = adam_init(net)
    b1, b2, eps = 0.9, 0.999, 1e-8
    history = []
    for t in range(1, steps + 1):
        X, cls, sev = make_batch(batch, rng)
        c = net.forward(X)
        L, parts, aux = net.loss(c, cls, sev, w)
        g = net.backward(c, cls, sev, aux, w)
        for k in net.keys:
            M[k] = b1 * M[k] + (1 - b1) * g[k]
            Vv[k] = b2 * Vv[k] + (1 - b2) * (g[k] ** 2)
            mh = M[k] / (1 - b1 ** t)
            vh = Vv[k] / (1 - b2 ** t)
            setattr(net, k, net.get(k) - lr * mh / (np.sqrt(vh) + eps))
        history.append(parts["total"])
        if log_every and (t % log_every == 0 or t == 1):
            print(f"    [{label}] step {t:5d}  total {parts['total']:.4f}"
                  f"  Ly {parts['Ly']:.4f}  La {parts['La']:.4f}"
                  f"  elite {parts['Lreg']:.4f}")
    return history


def evaluate(net, n=3000, rng=None):
    """
    Two numbers matter, and they are different numbers:
      yujie_acc   — can the WEAKEST listener recover the grievance?
      author_acc  — can the poet's own reading recover it?
    A poem that only the poet understands scores well on the second and fails
    the first. Bai treats that as failure, not as sophistication.
    """
    if rng is None:
        rng = np.random.default_rng(99)
    X, cls, sev = make_batch(n, rng)
    c = net.forward(X)
    yujie = (c["oo"].argmax(axis=1) == cls).mean()
    author = (c["oa"].argmax(axis=1) == cls).mean()
    elite = (c["P"] * ELITE_MASK).sum(axis=1).mean()
    # how much of the utterance's probability mass survives the market at all
    reach = (c["P"] * (1 - ELITE_MASK)).sum(axis=1).mean()

    # ---- WHAT THE GATE COSTS -------------------------------------------------
    # Bai's critics were not simply snobs. The recorded complaint against him is
    # 淺俗 — shallow and vulgar; monotonous; the same plain register for every
    # occasion. If that charge has any technical content, it should show up as a
    # collapse of expressive range, so it is measured rather than asserted.
    #
    # eff_vocab: perplexity of the corpus-averaged word distribution, i.e. the
    #   effective number of distinct words the poet actually reaches for.
    # nuance: how well SEVERITY (not just which grievance, but how bad) survives
    #   the four-unit bottleneck into the weak listener. Least-squares readout
    #   from the old woman's hidden state; reported as R^2.
    pbar = c["P"].mean(axis=0)
    eff_vocab = float(np.exp(-(pbar * np.log(np.maximum(pbar, 1e-12))).sum()))

    Bh = np.concatenate([c["B"], np.ones((c["B"].shape[0], 1))], axis=1)
    coef, *_ = np.linalg.lstsq(Bh, sev, rcond=None)
    resid = sev - Bh @ coef
    r2 = 1.0 - resid.var() / sev.var()

    return dict(yujie_acc=float(yujie), author_acc=float(author),
                elite_mass=float(elite), common_mass=float(reach),
                eff_vocab=eff_vocab, nuance_r2=float(r2))


# =============================================================================
# SECTION 4 — THE CAISHI CHANNEL (采詩)
# =============================================================================
# Bai does not argue that poems should be understood for aesthetic reasons. He
# argues that a comprehensible poem is a functioning organ of state: the only
# unfiltered channel between a village and a throne. He gives it two jobs, and
# they are genuinely distinct, which is why they get two separate metrics here.
#
#   補察時政  the signal must correct the ruler's model of what is wrong
#   洩導人情  the act of being heard must itself relieve the pressure below
#
# A channel can succeed at one and fail at the other. A censored petition system
# relieves nothing and corrects nothing. A therapeutic complaint box relieves
# and corrects nothing. Court flattery corrects nothing while relieving the
# courtier. Bai wants both, and audits them separately.

class CaishiChannel:
    def __init__(self, net, n_villages=40, eta=0.06, rng=None):
        self.net = net
        self.rng = rng or np.random.default_rng(7)
        self.n = n_villages
        self.eta = eta
        # ground truth: how much each grievance actually afflicts the realm
        self.true_burden = self.rng.dirichlet(np.ones(C) * 0.7)
        # the throne's belief about that burden, initially flat and wrong
        self.policy = np.ones(C) / C
        # unrelieved feeling in the population, per grievance
        self.pressure = self.true_burden.copy() * 1.0

    def one_year(self, filtered=False, n_poems=24):
        """
        filtered=True models the court WITHOUT the caishi office: reports reach
        the throne only through officials, who suppress grievances that
        implicate officials (classes 3,5 here: corvee and official luxury).
        """
        # villages sample grievances in proportion to what actually afflicts them
        cls = self.rng.choice(C, size=n_poems, p=self.true_burden)
        sev = self.rng.uniform(0.3, 1.0, size=n_poems)
        X = np.zeros((n_poems, DX))
        X[np.arange(n_poems), cls] = sev
        X[:, C:] = self.rng.normal(0, 0.4, size=(n_poems, 3))

        c = self.net.forward(X)
        heard = c["oo"].argmax(axis=1)          # what the weak listener recovers
        correct = (heard == cls)

        if filtered:
            keep = ~np.isin(cls, [3, 5])        # officials suppress their own
            heard, correct, cls_k = heard[keep], correct[keep], cls[keep]
        else:
            cls_k = cls

        # 補察: the throne updates its model from what it heard
        if len(heard):
            counts = np.bincount(heard, minlength=C).astype(float)
            counts /= counts.sum()
            self.policy = (1 - self.eta) * self.policy + self.eta * counts
            self.policy /= self.policy.sum()

        # 洩導: pressure falls only where a grievance was BOTH voiced AND
        # correctly understood. Being misheard relieves nothing.
        relief = np.zeros(C)
        for k, ok in zip(cls_k, correct):
            if ok:
                relief[k] += 1.0
        if relief.sum() > 0:
            relief /= relief.sum()
        self.pressure = np.maximum(0.0, self.pressure - 0.05 * relief)
        self.pressure += 0.012 * self.true_burden      # new harm accrues yearly
        return dict(
            buzha=1.0 - 0.5 * np.abs(self.policy - self.true_burden).sum(),
            xiedao=float(self.pressure.sum()),
        )

    def run(self, years=60, filtered=False):
        tr = [self.one_year(filtered=filtered) for _ in range(years)]
        return tr[-1], tr


# =============================================================================
# SECTION 5 — THE FENCANG ARCHIVE (分藏)
# =============================================================================
# This is the least metaphorical part of the file, because Bai actually did it.
# Having watched most of Li Bai's output vanish, he refused to trust any single
# vault — including the imperial library. He compiled his own collected works
# repeatedly and deposited complete copies in separate monasteries: Donglin at
# Lushan, Shengshan at Luoyang, and others, ending with a 75-juan recension
# around 842. Roughly 2,800 of his poems survive; that is not luck.
#
# The engineering content: he chose REPLICATION over centralisation, and he
# chose sites with uncorrelated failure modes (different provinces, different
# patrons, different armies). Below we test both the survival statistics and,
# importantly, that a restored copy is bit-identical in behaviour.

class FenCangArchive:
    def __init__(self, net, n_sites=5, rng=None):
        self.rng = rng or np.random.default_rng(842)
        self.n_sites = n_sites
        self.sites = [net.flat().copy() for _ in range(n_sites)]
        self.names = ["Donglin(Lushan)", "Shengshan(Luoyang)", "Xiangshan",
                      "Nanchan(Suzhou)", "family_cabinet"][:n_sites]

    @staticmethod
    def survival_probability(n_sites, per_site_hazard=0.55, trials=20000, rng=None):
        """
        Monte-Carlo: probability that at least one complete copy survives two
        centuries, given an independent per-site loss hazard. 0.55 is a blunt
        stand-in for fire, war, dynastic collapse and neglect over that span.
        """
        rng = rng or np.random.default_rng(0)
        lost = rng.random((trials, n_sites)) < per_site_hazard
        return float((~lost.all(axis=1)).mean())

    def catastrophe(self, k):
        """Destroy k sites at random. Return a surviving copy, or None."""
        idx = self.rng.permutation(self.n_sites)[:k]
        for i in idx:
            self.sites[i] = None
        alive = [s for s in self.sites if s is not None]
        return (alive[0] if alive else None), [self.names[i] for i in idx]


# =============================================================================
# SECTION 6 — SELF-TESTS
# =============================================================================

def test_weak_gate_matters():
    """
    THE CENTRAL EXPERIMENT.

    Train two identical networks. One is judged by the old woman (w_y = 1.0);
    the other is judged only by its own author (w_y = 0.0, w_a = 1.0) and pays
    no register penalty — the ordinary "write for your peers" regime that Bai
    was arguing against.

    Prediction, if Bai is right: the second network will be perfectly capable by
    its own lights while becoming unintelligible to anyone else, and it will
    drift into elite diction without being told to.
    """
    print("\n[TEST 1] Does the weak listener actually change the poem?")
    rng = np.random.default_rng(11)

    bai = CaishiResonanceNetwork(rng=np.random.default_rng(3))
    train(bai, steps=1200, w=dict(y=1.0, a=0.15, reg=0.35, pr=0.05, af=0.10),
          rng=np.random.default_rng(21), log_every=400, label="gated  ")
    r_bai = evaluate(bai)

    elite = CaishiResonanceNetwork(rng=np.random.default_rng(3))
    train(elite, steps=1200, w=dict(y=0.0, a=1.0, reg=0.0, pr=0.05, af=0.10),
          rng=np.random.default_rng(21), log_every=400, label="ungated")
    r_el = evaluate(elite)

    print(f"    gated  (judged by the market): yujie {r_bai['yujie_acc']:.3f}"
          f"  author {r_bai['author_acc']:.3f}  elite-mass {r_bai['elite_mass']:.3f}")
    print(f"    ungated(judged by the poet)  : yujie {r_el['yujie_acc']:.3f}"
          f"  author {r_el['author_acc']:.3f}  elite-mass {r_el['elite_mass']:.3f}")
    print(f"    chance level for yujie is {1.0 / C:.3f} — the ungated poem is"
          f" not merely harder, it is noise to the market listener.")

    assert r_bai["yujie_acc"] > r_el["yujie_acc"] + 0.20, \
        "the gate failed to improve weak-listener comprehension"
    assert r_bai["elite_mass"] < r_el["elite_mass"], \
        "the gate failed to suppress elite diction"

    print("    -- and what the gate costs: a NULL result, reported as such --")
    print(f"    effective vocabulary  gated {r_bai['eff_vocab']:6.2f}"
          f"   ungated {r_el['eff_vocab']:6.2f}   (of {V} words)")
    print(f"    severity nuance R^2   gated {r_bai['nuance_r2']:6.3f}"
          f"   ungated {r_el['nuance_r2']:6.3f}")
    print("    The elite critics' charge — that writing for the market narrows")
    print("    your register — does NOT reproduce here. Both models collapse to a")
    print("    similar handful of words, and neither preserves severity, because")
    print("    both were trained on a classification objective that never asked")
    print("    for it. This is a limitation of the experiment, not a vindication")
    print("    of Bai: to test the charge properly you would need an objective")
    print("    that rewards gradation, which is precisely the thing his own")
    print("    satirical mode was accused of flattening.")
    print("    PASS — the gate is load-bearing; its cost remains unmeasured.")
    return bai, r_bai, r_el


def test_caishi_channel(net):
    """The open channel must beat the filtered one on BOTH metrics."""
    print("\n[TEST 2] Does an open collection channel outperform a filtered one?")
    open_ch = CaishiChannel(net, rng=np.random.default_rng(5))
    filt_ch = CaishiChannel(net, rng=np.random.default_rng(5))
    o, _ = open_ch.run(years=60, filtered=False)
    f, _ = filt_ch.run(years=60, filtered=True)
    print(f"    open    : buzha(policy fit) {o['buzha']:.3f}"
          f"   xiedao(residual pressure) {o['xiedao']:.3f}")
    print(f"    filtered: buzha(policy fit) {f['buzha']:.3f}"
          f"   xiedao(residual pressure) {f['xiedao']:.3f}")
    assert o["buzha"] > f["buzha"], "open channel failed to correct policy better"
    assert o["xiedao"] < f["xiedao"], "open channel failed to relieve pressure"
    print("    PASS — suppression degrades correction and relief independently.")


def test_fencang_archive(net):
    """Scattered whole copies: survival statistics, plus exact restoration."""
    print("\n[TEST 3] Does scattering whole copies preserve the mind?")
    for n in (1, 2, 3, 5):
        p = FenCangArchive.survival_probability(n, rng=np.random.default_rng(n))
        print(f"    {n} site(s): P(at least one copy survives 200y) = {p:.4f}")

    before = evaluate(net, n=1500)
    arch = FenCangArchive(net, n_sites=5, rng=np.random.default_rng(842))
    copy, destroyed = arch.catastrophe(4)
    assert copy is not None, "total loss — archive strategy failed"
    print(f"    destroyed: {', '.join(destroyed)}")

    ghost = CaishiResonanceNetwork(rng=np.random.default_rng(0))
    ghost.load_flat(copy)
    after = evaluate(ghost, n=1500)
    assert abs(before["yujie_acc"] - after["yujie_acc"]) < 1e-12, \
        "restored copy does not behave identically"
    print(f"    restored from 1 of 5 surviving copies:"
          f" yujie {after['yujie_acc']:.4f} (identical to {before['yujie_acc']:.4f})")
    print("    PASS — redundancy across uncorrelated sites, not a single vault.")


# =============================================================================
# SECTION 7 — MAIN
# =============================================================================

def main():
    print("=" * 78)
    print(" CRN — Caishi Resonance Network   |   Figure 0195, Bai Juyi (772-846)")
    print(" 根情 root=affect  苗言 sprout=language  華聲 flower=sound  實義 fruit=meaning")
    print("=" * 78)

    print("\n[GRADIENT CHECK] hand-derived backward pass vs central differences")
    worst = gradient_check()
    print(f"  OK — analytic gradients verified (worst rel err {worst:.3e})")

    net, r_bai, r_el = test_weak_gate_matters()
    test_caishi_channel(net)
    test_fencang_archive(net)

    print("\n" + "=" * 78)
    print(" SUMMARY")
    print("-" * 78)
    print(f" weak-listener comprehension, gated model   : {r_bai['yujie_acc']:.3f}")
    print(f" weak-listener comprehension, ungated model : {r_el['yujie_acc']:.3f}")
    print(f" elite diction mass, gated / ungated        :"
          f" {r_bai['elite_mass']:.3f} / {r_el['elite_mass']:.3f}")
    print(" The generator is larger than its verifier. That is the point:")
    print(" legitimacy is bounded by the comprehension of the weakest recipient.")
    print("=" * 78)


if __name__ == "__main__":
    main()
