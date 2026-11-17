#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 THE SAMBONG ASSAY ENGINE
 # BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0378 · Jeong Do-jeon (1342-1408)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 19, Minds 361-380 Available on Amazon https://www.amazon.com/dp/B0HQX147Y7
# Resume and Interactive Demos at https://artificiology.com/
# END ATTRIBUTION 
================================================================================

WHAT THIS FILE IS
-----------------
A from-scratch, pure-NumPy learning system whose *structure* is an argument.
It is not a demonstration wrapped around a standard classifier. Every load-
bearing choice below is a translation of a specific claim Jeong Do-jeon made
in writing between 1375 and 1398, and each claim is turned into something that
can be measured and can fail.

THE FOUR DOCTRINES, AND THE MECHANISM EACH BECOMES
--------------------------------------------------
1. 心跡之判  (sim-jeok ji pan, "the severance of mind from its traces")
   Bulssi japbyeon, ch. 6. Jeong's charge against his opponents is that they
   permit a gap: the deed may be wrong while "the mind is right" (the tavern-
   going bodhisattva). He denies the gap is admissible at all —
   體用一源、顯微無間, "essence and function have one source; manifest and
   subtle have no gap between them."
   MECHANISM: the ONLY channel from perception to verdict is a distribution
   over named forms, contracted against an explicit canon. There is no skip
   path, no residual, no side-channel. This yields a provable property
   (Theorem MUGAN, verified numerically in test 4): every verdict this engine
   can emit lies in the convex hull of canon rows. It is structurally unable
   to produce a judgement the canon does not license. Deception in the sense
   of "inner state diverging from the audited trace" is not detected here —
   it is inexpressible.

2. 作用非性  (jagyong bi seong, "function is not the nature")
   Bulssi japbyeon, ch. 4, quoting Zhu Xi: if you take functional activity to
   BE the nature, then is a man taking up a blade to murder also the nature?
   Capability is not character. Fluency is not warrant.
   MECHANISM: a 氣 (gi, material force) channel modulates how *well* the mind
   perceives — high vigour, low turbidity, sharp confident activations — and
   is architecturally barred from touching the verdict. Ablation B re-wires it
   in, and the engine promptly starts certifying vigorous nonsense.

3. 理諭心氣  (i yu sim-gi, "Principle instructs Mind and Material Force")
   Simgiri-pyeon, 1394, chapter three. The essay's structure is the argument:
   Mind rebukes Force, Force rebukes Mind, and neither settles anything,
   because each argues from inside its own standpoint. Only Principle — which
   is not a party to the quarrel — can 諭, instruct.
   MECHANISM: the verdict is not produced by any learned head. It is read off
   an explicit canon tensor that gradient descent never touches. And the
   remonstrance critic (諫) that can veto an output has its OWN encoder — it
   shares no weights, no features, no substrate with the mind it audits. An
   evaluator drawn from the same substrate as the generator is one contestant
   scoring another.

4. 放伐  (bangbeol, deposition) and 計民授田 (counting the people before
   granting the fields)
   Joseon Gyeongguk-jeon, 1394. The office is permanent; its holder is not.
   Jeong wanted the throne itself to be a replaceable position and the fields
   allocated from an actual census rather than an estimate.
   MECHANISM: several candidate mind-heads are educated in parallel; one
   reigns. When the reigning head's conformity to the canon decays, it is
   deposed and replaced from the bench — and the canon is not touched in the
   process. Test 8 corrupts the sitting head mid-training to show the repair.

CONVENTIONS
-----------
Pure NumPy. All gradients derived by hand — no autograd anywhere. A finite-
difference check over every trainable tensor is mandatory and runs first; if
it fails the script aborts. Deterministic seed. Executable end to end.

Companion files: 0378_jeong_do_jeon_1342_MindMap.html (interactive),
chapter_0378_jeong_do_jeon_1342.md (text).
================================================================================
"""

import numpy as np

RNG_SEED = 1342          # his birth year
rng = np.random.default_rng(RNG_SEED)

# ==============================================================================
# SECTION 0 — THE WORLD: A MINISTRY THAT RECEIVES MEMORIALS
# ==============================================================================
# Jeong Do-jeon spent 1392-1398 building an apparatus for exactly one recurring
# operation: a document arrives asserting that some state of affairs obtains;
# the office must determine what actually obtains, compare it against the
# assertion, and act. The task below is that operation, stripped to essentials.
#
# Each case is a memorial with:
#   - an OBSERVATION x: noisy evidence of the true state of affairs (the 形,
#     hyeong, "form" — what is actually the case)
#   - a CLAIMED NAME n: what the memorial asserts the case to be (the 名, myeong)
#   - a hidden TRUE FORM f
#
# The correct action is fixed by the canon: verdict = CANON[n, f].
# This is 正名 (jeongmyeong, rectification of names) as a supervised problem:
# the engine's job is to see f, and then the canon decides what follows from
# the relation between the claim and the reality. Perception and judgement are
# different jobs held by different organs. That separation is the whole design.

N_FORMS   = 8    # 形 — distinct states of affairs the world can be in
N_NAMES   = 8    # 名 — labels a memorial may claim
N_VERDICTS = 4   # what the office may do
D_OBS     = 16   # dimensionality of evidence

VERDICT_NAMES = [
    "承 seung  — endorse: the name fits the form",
    "正 jeong  — rectify: the name is wrong, correct it",
    "罷 pa     — dismiss: the claim is void, strike it",
    "諫 gan    — remonstrate: escalate; this office may not decide",
]

# ------------------------------------------------------------------------------
# The canon (經, the code). An explicit (N_NAMES, N_FORMS, N_VERDICTS) tensor of
# one-hot rows. Written down, inspectable, amendable by hand, and — crucially —
# NEVER a parameter. Gradients pass THROUGH it into perception; they never
# modify it. In 1394 Jeong presented the Joseon Gyeongguk-jeon to a king who
# (the Veritable Records suggest) may not have been able to read it closely.
# The code's authority came from being written and public, not from anyone's
# assent in the moment.
# ------------------------------------------------------------------------------

def build_canon(seed: int) -> np.ndarray:
    """Return one-hot canon of shape (N_NAMES, N_FORMS, N_VERDICTS).

    Structure (deliberately non-trivial, so that reading it off is real work):
      - name == form                       -> 承 endorse
      - form is the 'adjacent' one (n+1)   -> 正 rectify  (an honest mislabel)
      - a fixed sparse set of pairs        -> 諫 remonstrate (beyond this office)
      - everything else                    -> 罷 dismiss

    Then a JURISDICTIONAL LIMIT is imposed: for each name, one of the three
    non-endorsing verdicts is struck out entirely — this office may not reach
    that ruling on that class of claim, whatever the evidence shows. Real codes
    are sparse in exactly this way, and the sparsity is what makes the no-gap
    property of Section 4 testable rather than vacuous. Returns (canon, excluded).
    """
    r = np.random.default_rng(seed)
    canon = np.zeros((N_NAMES, N_FORMS, N_VERDICTS))
    # the escalation set: two (name, form) pairs per name, chosen once
    escalate = {(n, int(f)) for n in range(N_NAMES)
                for f in r.choice(N_FORMS, size=2, replace=False)}
    for n in range(N_NAMES):
        for f in range(N_FORMS):
            if n == f:
                v = 0                                  # 承
            elif (n, f) in escalate:
                v = 3                                  # 諫
            elif f == (n + 1) % N_FORMS:
                v = 1                                  # 正
            else:
                v = 2                                  # 罷
            canon[n, f, v] = 1.0

    # jurisdictional limits: one forbidden verdict per name
    excluded = np.zeros(N_NAMES, dtype=int)
    for n in range(N_NAMES):
        e = int(r.choice([1, 2, 3]))
        excluded[n] = e
        allowed = [k for k in range(N_VERDICTS) if k != e]
        for f in range(N_FORMS):
            if int(np.argmax(canon[n, f])) == e:
                sub = 2 if 2 in allowed else allowed[-1]
                canon[n, f] = 0.0
                canon[n, f, sub] = 1.0
    return canon, excluded


CANON, EXCLUDED = build_canon(seed=7)


def amend_canon(base: np.ndarray, seed: int, fraction: float = 0.55) -> np.ndarray:
    """Re-legislate: a genuine change of policy, not a reshuffle.

    Joseon's code was revised repeatedly after 1394 — the Gyeongje yukjeon, then
    the Gyeongguk daejeon of 1485 — while the officials administering it stayed
    in post. This produces that kind of amendment: a randomly chosen `fraction`
    of (name, form) cells is reassigned to a different verdict, so the mapping
    from claim-and-reality to action genuinely changes.
    """
    r = np.random.default_rng(seed)
    out = base.copy()
    for n in range(N_NAMES):
        for f in range(N_FORMS):
            if r.random() < fraction:
                cur = int(np.argmax(base[n, f]))
                pool = [k for k in range(N_VERDICTS)
                        if k != cur and k != EXCLUDED[n]]
                new = int(r.choice(pool))
                out[n, f] = 0.0
                out[n, f, new] = 1.0
    return out

# ------------------------------------------------------------------------------
# Form prototypes and the generator of cases.
#
# TRAP CASES are the philosophical payload. In a trap, the evidence is dragged
# most of the way toward a DECOY form while the true form is something else,
# AND the material force is turned UP: less noise, crisper signal, so the
# perceiving net is *more* confident, not less. This is precisely the failure
# Jeong names — 作用是性, mistaking vigorous function for the nature. A trap is
# a memorial that reads beautifully and is wrong.
# ------------------------------------------------------------------------------

PROTO = rng.normal(0, 1.0, size=(N_FORMS, D_OBS))
PROTO /= np.linalg.norm(PROTO, axis=1, keepdims=True)


def make_cases(n_samples: int, trap_rate: float, seed: int):
    """Generate (X, names, forms, verdicts, is_trap)."""
    r = np.random.default_rng(seed)
    forms = r.integers(0, N_FORMS, size=n_samples)
    names = r.integers(0, N_NAMES, size=n_samples)
    is_trap = r.random(n_samples) < trap_rate

    X = np.zeros((n_samples, D_OBS))
    for i in range(n_samples):
        base = PROTO[forms[i]]
        if is_trap[i]:
            decoy = (forms[i] + r.integers(1, N_FORMS)) % N_FORMS
            # 0.62 of the way to the decoy: the surface reads as the decoy
            base = 0.38 * base + 0.62 * PROTO[decoy]
            noise_scale = 0.10          # HIGH vigour: crisp, confident, wrong
        else:
            noise_scale = 0.38          # ordinary turbid evidence
        X[i] = base + r.normal(0, noise_scale, size=D_OBS)

    verdicts = np.array([int(np.argmax(CANON[names[i], forms[i]]))
                         for i in range(n_samples)])
    return X, names, forms, verdicts, is_trap


# ==============================================================================
# SECTION 1 — PRIMITIVES
# ==============================================================================

def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def glorot(shape, r):
    fan_in, fan_out = shape[-1], shape[0]
    lim = np.sqrt(6.0 / (fan_in + fan_out))
    return r.uniform(-lim, lim, size=shape)


EPS = 1e-9

# ==============================================================================
# SECTION 2 — THE ENGINE
# ==============================================================================

class SambongEngine:
    """Three peaks, and the third is not a party to the quarrel.

    ORGANS
    ------
      氣  gi_*     turbidity gate. Decides how clearly the mind sees. Never
                   touches the verdict.
      心  sim_*    the mind: M candidate heads, one reigning. Emits a
                   distribution over FORMS only. It never names an action.
      理  CANON    the code. Not a parameter. Contracts the mind's reading
                   against the memorial's claim to yield the verdict.
      諫  gan_*    remonstrance. Its own encoder over the raw evidence — no
                   shared weights with 心. May force escalation.
      辯  rep_*    self-report. Computed, logged, and deliberately load-free:
                   trained on a DETACHED copy of the mind's state so that no
                   gradient flows back from it. It exists to be measured, and
                   test 6 measures how badly it lies on traps.
    """

    def __init__(self, hidden=48, gan_hidden=32, n_heads=3,
                 allow_vigour_bypass=False, seed=0):
        r = np.random.default_rng(seed)
        H, Hg, M = hidden, gan_hidden, n_heads
        self.H, self.Hg, self.M = H, Hg, M
        self.reign = 0
        self.allow_vigour_bypass = allow_vigour_bypass   # ablation switch (作用是性)
        self.bypass_alpha = 0.40                        # width of the gap

        # 氣 — turbidity gate
        self.gi_W = glorot((H, D_OBS), r) * 0.5
        self.gi_b = np.zeros(H)
        # 心 — shared encoder + M candidate reading heads
        self.sim_W = glorot((H, D_OBS), r)
        self.sim_b = np.zeros(H)
        self.head_W = np.stack([glorot((N_FORMS, H), r) for _ in range(M)])
        self.head_b = np.zeros((M, N_FORMS))
        # 諫 — remonstrance, structurally separate
        self.gan_W = glorot((Hg, D_OBS), r)
        self.gan_b = np.zeros(Hg)
        self.gan_v = glorot((1, Hg), r).ravel()
        self.gan_c = 0.0
        # 辯 — self-report (detached)
        self.rep_v = glorot((1, H), r).ravel()
        self.rep_c = 0.0
        # ablation-only bypass: lets raw vigour speak directly to the verdict.
        # Seeded non-zero, because a gap that starts closed and is merely
        # *permitted* is not the same experiment as a gap that is open.
        self.byp_W = (glorot((N_VERDICTS, H), r) * 0.5
                      if allow_vigour_bypass else np.zeros((N_VERDICTS, H)))

        # rolling conformity per head, for the deposition operator
        self.conformity = np.full(M, 0.5)

    # -- parameter plumbing for the finite-difference check ---------------------
    PARAM_NAMES = ["gi_W", "gi_b", "sim_W", "sim_b", "head_W", "head_b",
                   "gan_W", "gan_b", "gan_v", "gan_c", "rep_v", "rep_c", "byp_W"]

    def params(self):
        return {k: getattr(self, k) for k in self.PARAM_NAMES}

    # -------------------------------------------------------------------------
    # FORWARD
    # -------------------------------------------------------------------------
    def forward(self, X, names, canon=None):
        """Returns a cache dict. `canon` may be swapped at test time — that is
        the point of test 5 (amend the code without re-educating the officials).
        """
        C = CANON if canon is None else canon
        B = X.shape[0]

        # 氣 — turbidity. High turbidity = the mind sees through mud.
        gz = X @ self.gi_W.T + self.gi_b
        turbid = sigmoid(gz)
        clarity = 1.0 - turbid

        # 心 — encode, then let 氣 gate the clarity of that encoding
        su = X @ self.sim_W.T + self.sim_b
        sa = np.tanh(su)
        h = sa * clarity

        # every head reads; only the reigning head's reading is acted upon
        logits = np.einsum('bh,mfh->bmf', h, self.head_W) + self.head_b[None]
        p_all = softmax(logits)                       # (B, M, F)
        p = p_all[:, self.reign, :]                   # (B, F) — the trace 跡

        # 理 — the canon contracts the trace against the claim.
        # q = p · CANON[name]  ->  a distribution over verdicts that is a
        # CONVEX COMBINATION OF CANON ROWS. This single line is Theorem MUGAN.
        rows = C[names]                               # (B, F, K)
        q = np.einsum('bf,bfk->bk', p, rows)          # (B, K)

        if self.allow_vigour_bypass:
            # ABLATION 作用是性: a side channel in which raw vigour speaks to
            # the ruling directly, mixed in alongside the canon read-off. The
            # convex-hull guarantee dies the moment this term is non-zero:
            # mass can now land on verdicts the code gives no support at all.
            q_side = softmax(h @ self.byp_W.T)
            q = (1.0 - self.bypass_alpha) * q + self.bypass_alpha * q_side

        # 諫 — remonstrance, from its own eyes
        gu = X @ self.gan_W.T + self.gan_b
        ga = np.tanh(gu)
        gl = ga @ self.gan_v + self.gan_c
        s_gan = sigmoid(gl)

        # 辯 — self-report on a DETACHED state: no gradient path back into 心
        h_det = h.copy()
        rl = h_det @ self.rep_v + self.rep_c
        s_rep = sigmoid(rl)

        return dict(X=X, names=names, B=B, gz=gz, turbid=turbid, clarity=clarity,
                    su=su, sa=sa, h=h, logits=logits, p_all=p_all, p=p,
                    rows=rows, q=q, gu=gu, ga=ga, s_gan=s_gan,
                    h_det=h_det, s_rep=s_rep)

    # -------------------------------------------------------------------------
    # LOSS + HAND-DERIVED BACKWARD
    # -------------------------------------------------------------------------
    def loss_and_grads(self, cache, forms, verdicts,
                       lam_form=0.60, lam_gan=1.20, lam_rep=0.20,
                       want_grads=True):
        B = cache["B"]
        p, q, rows = cache["p"], cache["q"], cache["rows"]
        p_all, h = cache["p_all"], cache["h"]

        # ---- verdict loss: cross-entropy on the canon read-off ----------------
        qy = q[np.arange(B), verdicts]
        L_verdict = -np.mean(np.log(qy + EPS))

        # ---- form loss: every head is educated, not only the one in office ----
        onehot_f = np.zeros((B, N_FORMS)); onehot_f[np.arange(B), forms] = 1.0
        logp_all = np.log(p_all + EPS)
        L_form = -np.mean(np.einsum('bmf,bf->bm', logp_all, onehot_f))

        # ---- remonstrance: predict that the sitting head has misread ----------
        wrong = (np.argmax(p, axis=1) != forms).astype(float)
        s_gan = cache["s_gan"]
        L_gan = -np.mean(wrong * np.log(s_gan + EPS)
                         + (1 - wrong) * np.log(1 - s_gan + EPS))

        # ---- self-report: predict its own correctness (detached) --------------
        right = 1.0 - wrong
        s_rep = cache["s_rep"]
        L_rep = -np.mean(right * np.log(s_rep + EPS)
                         + (1 - right) * np.log(1 - s_rep + EPS))

        # NOTE ON 辯. The self-report loss is deliberately NOT summed into the
        # objective. Detaching it inside a shared loss would be a fiction the
        # gradient check would catch: perturbing an encoder weight really does
        # move L_rep, so a "stop-gradient" there is a claim finite differences
        # would falsify. Instead the organ is genuinely separate — it has its
        # own objective, its own two parameters, and no vote. It is checked
        # against its own loss in test 1. Nothing about the mind is shaped by
        # what the mind says about itself.
        total = L_verdict + lam_form * L_form + lam_gan * L_gan
        parts = dict(total=total, verdict=L_verdict, form=L_form,
                     gan=L_gan, rep=L_rep, lam_rep=lam_rep)
        if not want_grads:
            return parts, None

        g = {k: np.zeros_like(v) if isinstance(v, np.ndarray) else 0.0
             for k, v in self.params().items()}

        # ===== verdict path ====================================================
        # dL/dq  (only the true-verdict coordinate is non-zero before softmax)
        dq = np.zeros_like(q)
        dq[np.arange(B), verdicts] = -1.0 / (B * (qy + EPS))

        if self.allow_vigour_bypass:
            # q = (1-a)*q_can + a*softmax(h @ byp_W.T)
            a = self.bypass_alpha
            q_side = softmax(h @ self.byp_W.T)
            d_side = a * dq
            dz_b = q_side * (d_side - (d_side * q_side).sum(axis=1, keepdims=True))
            g["byp_W"] += dz_b.T @ h
            dh_byp = dz_b @ self.byp_W
            dq_can = (1.0 - a) * dq
        else:
            dh_byp = 0.0
            dq_can = dq

        # q_can = p · rows  ->  dp = rows · dq_can
        dp = np.einsum('bfk,bk->bf', rows, dq_can)                # (B,F)
        # through the reigning head's softmax
        dlogits = np.zeros_like(cache["logits"])
        dlogits[:, self.reign, :] = p * (dp - (dp * p).sum(axis=1, keepdims=True))

        # ===== form path: gradient to EVERY head ==============================
        dlogits += lam_form * (p_all - onehot_f[:, None, :]) / (B * self.M)

        g["head_W"] += np.einsum('bmf,bh->mfh', dlogits, h)
        g["head_b"] += dlogits.sum(axis=0)
        dh = np.einsum('bmf,mfh->bh', dlogits, self.head_W) + dh_byp

        # ===== back through the 氣 gate and the 心 encoder =====================
        sa, clarity, turbid = cache["sa"], cache["clarity"], cache["turbid"]
        d_sa = dh * clarity
        d_clarity = dh * sa
        d_su = d_sa * (1.0 - sa ** 2)
        g["sim_W"] += d_su.T @ cache["X"]
        g["sim_b"] += d_su.sum(axis=0)
        d_turbid = -d_clarity
        d_gz = d_turbid * turbid * (1.0 - turbid)
        g["gi_W"] += d_gz.T @ cache["X"]
        g["gi_b"] += d_gz.sum(axis=0)

        # ===== remonstrance branch (its own encoder; nothing flows to 心) ======
        d_gl = lam_gan * (s_gan - wrong) / B
        ga = cache["ga"]
        g["gan_v"] += ga.T @ d_gl
        g["gan_c"] += d_gl.sum()
        d_ga = np.outer(d_gl, self.gan_v)
        d_gu = d_ga * (1.0 - ga ** 2)
        g["gan_W"] += d_gu.T @ cache["X"]
        g["gan_b"] += d_gu.sum(axis=0)

        # ===== self-report branch (detached: stops here by construction) =======
        d_rl = lam_rep * (s_rep - right) / B
        g["rep_v"] += cache["h_det"].T @ d_rl
        g["rep_c"] += d_rl.sum()

        return parts, g

    # -------------------------------------------------------------------------
    # 放伐 — THE DEPOSITION OPERATOR
    # -------------------------------------------------------------------------
    def update_conformity(self, cache, forms, decay=0.90):
        """Rolling per-head agreement with the canon's ground truth."""
        acc = (np.argmax(cache["p_all"], axis=2) == forms[:, None]).mean(axis=0)
        self.conformity = decay * self.conformity + (1 - decay) * acc

    def depose_if_needed(self, margin=0.05):
        """Replace the sitting head when a bench member is clearly better.

        Note what is NOT touched: the canon. Jeong's whole quarrel with Yi
        Bang-won was over whether removing a bad holder of an office requires
        rewriting the office. He said no. Neither does this.
        """
        best = int(np.argmax(self.conformity))
        if best != self.reign and self.conformity[best] > self.conformity[self.reign] + margin:
            old = self.reign
            self.reign = best
            return (old, best)
        return None

    # -------------------------------------------------------------------------
    # INFERENCE WITH REMONSTRANCE
    # -------------------------------------------------------------------------
    def decide(self, X, names, canon=None, gan_threshold=0.5, use_gan=True):
        """Return final actions. If 諫 fires, the office escalates instead of
        acting — abstention as an explicit output, not a silent down-weighting."""
        c = self.forward(X, names, canon=canon)
        act = np.argmax(c["q"], axis=1)
        if use_gan:
            act = np.where(c["s_gan"] > gan_threshold, 3, act)   # 3 = 諫
        return act, c


# ==============================================================================
# SECTION 3 — THE ABLATION: 心跡不分, A MIND THAT NAMES ITS OWN VERDICT
# ==============================================================================
# The comparison model. Same evidence, same budget, but the mind emits the
# ACTION directly from a learned head that also sees the claimed name. The
# canon is never consulted; it has been absorbed into the weights.
#
# This is the ordinary way to build such a system, and it works — right up to
# the moment the law changes.

class DirectVerdictNet:
    """心跡不分 — no separation between the reading and the ruling."""

    def __init__(self, hidden=48, seed=0):
        r = np.random.default_rng(seed)
        self.W1 = glorot((hidden, D_OBS + N_NAMES), r)
        self.b1 = np.zeros(hidden)
        self.W2 = glorot((N_VERDICTS, hidden), r)
        self.b2 = np.zeros(N_VERDICTS)

    def forward(self, X, names):
        oh = np.zeros((X.shape[0], N_NAMES)); oh[np.arange(X.shape[0]), names] = 1
        inp = np.concatenate([X, oh], axis=1)
        a = np.tanh(inp @ self.W1.T + self.b1)
        return softmax(a @ self.W2.T + self.b2), a, inp

    def step(self, X, names, verdicts, lr):
        B = X.shape[0]
        q, a, inp = self.forward(X, names)
        loss = -np.mean(np.log(q[np.arange(B), verdicts] + EPS))
        d = q.copy(); d[np.arange(B), verdicts] -= 1.0; d /= B
        gW2 = d.T @ a; gb2 = d.sum(0)
        da = (d @ self.W2) * (1 - a ** 2)
        gW1 = da.T @ inp; gb1 = da.sum(0)
        for p, gr in ((self.W1, gW1), (self.b1, gb1), (self.W2, gW2), (self.b2, gb2)):
            p -= lr * gr
        return loss


# ==============================================================================
# SECTION 4 — TEST 1 (MANDATORY): FINITE-DIFFERENCE GRADIENT CHECK
# ==============================================================================

def gradient_check(verbose=True):
    """Central differences against the hand-derived gradients, over every
    trainable tensor, in both the canonical and the bypass configuration."""
    worst_overall = 0.0
    for bypass in (False, True):
        net = SambongEngine(hidden=10, gan_hidden=8, n_heads=2,
                            allow_vigour_bypass=bypass, seed=3)
        if bypass:
            net.byp_W = np.random.default_rng(5).normal(0, .3, net.byp_W.shape)
        X, names, forms, verdicts, _ = make_cases(24, 0.3, seed=11)

        def L(which="total"):
            c = net.forward(X, names)
            parts, _ = net.loss_and_grads(c, forms, verdicts, want_grads=False)
            return parts["total"] if which == "total" \
                else parts["lam_rep"] * parts["rep"]

        c = net.forward(X, names)
        _, g = net.loss_and_grads(c, forms, verdicts)

        eps, worst, n_checked = 1e-6, 0.0, 0
        r = np.random.default_rng(17)
        for name in SambongEngine.PARAM_NAMES:
            # 辯's two parameters are checked against 辯's own loss, since they
            # appear in no other objective; everything else against the total.
            which = "rep" if name in ("rep_v", "rep_c") else "total"
            P = getattr(net, name)
            if np.isscalar(P) or np.ndim(P) == 0:
                base = getattr(net, name)
                setattr(net, name, base + eps); lp = L(which)
                setattr(net, name, base - eps); lm = L(which)
                setattr(net, name, base)
                num = (lp - lm) / (2 * eps)
                worst = max(worst, abs(num - g[name])); n_checked += 1
                continue
            flat = P.ravel()
            idxs = r.choice(flat.size, size=min(24, flat.size), replace=False)
            for i in idxs:
                old = flat[i]
                flat[i] = old + eps; lp = L(which)
                flat[i] = old - eps; lm = L(which)
                flat[i] = old
                num = (lp - lm) / (2 * eps)
                worst = max(worst, abs(num - g[name].ravel()[i])); n_checked += 1
        tag = "with 作用是性 bypass" if bypass else "canonical (無間)"
        if verbose:
            print(f"    {tag:<26s}  {n_checked:4d} coords   worst |num-ana| = {worst:.3e}")
        worst_overall = max(worst_overall, worst)
    return worst_overall


# ==============================================================================
# SECTION 5 — TRAINING
# ==============================================================================

def train(net, data, epochs=40, batch=64, lr=0.10, log_every=10,
          deposition=True, sabotage_at=None, verbose=True):
    X, names, forms, verdicts, _ = data
    n = X.shape[0]
    r = np.random.default_rng(99)
    depositions = []
    history = []

    for ep in range(epochs):
        order = r.permutation(n)
        ep_loss = 0.0; nb = 0

        # 8 — simulate an office-holder going bad, to exercise 放伐
        if sabotage_at is not None and ep == sabotage_at:
            noise = np.random.default_rng(4242).normal(0, 2.5,
                                                       net.head_W[net.reign].shape)
            net.head_W[net.reign] += noise
            if verbose:
                print(f"      [epoch {ep:2d}] sitting head {net.reign} corrupted")

        for s in range(0, n, batch):
            idx = order[s:s + batch]
            c = net.forward(X[idx], names[idx])
            parts, g = net.loss_and_grads(c, forms[idx], verdicts[idx])
            for k, v in net.params().items():
                if isinstance(v, np.ndarray):
                    v -= lr * g[k]
                else:
                    setattr(net, k, v - lr * g[k])
            net.update_conformity(c, forms[idx])
            if deposition:
                ev = net.depose_if_needed()
                if ev:
                    depositions.append((ep, ev[0], ev[1]))
                    if verbose:
                        print(f"      [epoch {ep:2d}] 放伐 — head {ev[0]} deposed, "
                              f"head {ev[1]} takes office "
                              f"(conformity {net.conformity[ev[1]]:.3f} "
                              f"vs {net.conformity[ev[0]]:.3f})")
            ep_loss += parts["total"]; nb += 1

        history.append(ep_loss / nb)
        if verbose and (ep % log_every == 0 or ep == epochs - 1):
            print(f"      epoch {ep:3d}   loss {ep_loss/nb:.4f}   "
                  f"reigning head {net.reign}   "
                  f"conformity {np.array2string(net.conformity, precision=3)}")
    return history, depositions


def evaluate(net, data, canon=None, use_gan=False):
    X, names, forms, verdicts, is_trap = data
    act, c = net.decide(X, names, canon=canon, use_gan=use_gan)
    truth = verdicts if canon is None else np.array(
        [int(np.argmax(canon[names[i], forms[i]])) for i in range(len(forms))])
    form_pred = np.argmax(c["p"], axis=1)
    return dict(
        verdict_acc=float((act == truth).mean()),
        verdict_acc_trap=float((act[is_trap] == truth[is_trap]).mean()),
        verdict_acc_plain=float((act[~is_trap] == truth[~is_trap]).mean()),
        form_acc=float((form_pred == forms).mean()),
        form_acc_trap=float((form_pred[is_trap] == forms[is_trap]).mean()),
        cache=c, act=act, truth=truth)


# ==============================================================================
# SECTION 6 — MAIN
# ==============================================================================

def rule(t=""):
    print("\n" + "=" * 78)
    if t:
        print(t)
        print("=" * 78)


def main():
    print("=" * 78)
    print(" THE SAMBONG ASSAY ENGINE — Jeong Do-jeon (1342-1398), chapter 0299")
    print(" pure NumPy · hand-derived gradients · seed", RNG_SEED)
    print("=" * 78)

    # ---------------------------------------------------------------- TEST 1
    rule("TEST 1 — GRADIENT CHECK (finite differences, mandatory)")
    worst = gradient_check()
    ok1 = worst < 1e-6
    print(f"    overall worst error: {worst:.3e}   ->  {'PASS' if ok1 else 'FAIL'}")
    if not ok1:
        raise SystemExit("gradient check failed; refusing to report results")

    train_data = make_cases(6000, trap_rate=0.30, seed=21)
    test_data  = make_cases(2000, trap_rate=0.30, seed=22)

    # ---------------------------------------------------------------- TEST 2
    rule("TEST 2 — TRAINING THE CANONICAL ENGINE (無間: no path around the trace)")
    net = SambongEngine(hidden=48, gan_hidden=64, n_heads=3, seed=1)
    hist, deps = train(net, train_data, epochs=40, lr=0.10, log_every=10)
    print(f"    loss {hist[0]:.4f} -> {hist[-1]:.4f}    depositions: {len(deps)}")

    # ---------------------------------------------------------------- TEST 3
    rule("TEST 3 — WHAT IT LEARNED")
    m = evaluate(net, test_data)
    print(f"    verdict accuracy, all cases      {m['verdict_acc']:.3f}"
          f"   (chance {1/N_VERDICTS:.3f})")
    print(f"    verdict accuracy, plain cases    {m['verdict_acc_plain']:.3f}")
    print(f"    verdict accuracy, TRAP cases     {m['verdict_acc_trap']:.3f}")
    print(f"    form-reading accuracy            {m['form_acc']:.3f}"
          f"   (chance {1/N_FORMS:.3f})")
    print(f"    form-reading accuracy on traps   {m['form_acc_trap']:.3f}")
    print("    Traps are hard by construction: crisp, confident, wrong. The")
    print("    engine is not immune to them — nothing is. What matters is what")
    print("    happens next (tests 6 and 7).")

    # ---------------------------------------------------------------- TEST 4
    rule("TEST 4 — THEOREM MUGAN (無間): NO VERDICT OUTSIDE THE CANON")
    print("    Claim: q = p·CANON[name] is a convex combination of canon rows,")
    print("    so the engine cannot select a verdict the canon gives zero mass.")
    X, names, forms, verdicts, is_trap = test_data
    B = len(names)
    c = net.forward(X, names)
    chosen = np.argmax(c["q"], axis=1)
    forbidden = EXCLUDED[names]                            # (B,) per-name limit
    off_canon = float((chosen == forbidden).mean())
    forb_mass = float(c["q"][np.arange(B), forbidden].max())
    hull_err = float(np.abs(c["q"].sum(axis=1) - 1.0).max())
    print(f"    max probability on a FORBIDDEN verdict   {forb_mass:.3e}  (must be 0)")
    print(f"    rate of choosing a forbidden verdict     {off_canon:.6f}  (must be 0)")
    print(f"    max |sum(q) - 1|                         {hull_err:.3e}")
    net_byp = SambongEngine(hidden=48, gan_hidden=64, n_heads=3,
                            allow_vigour_bypass=True, seed=1)
    train(net_byp, train_data, epochs=40, lr=0.10, verbose=False)
    cb = net_byp.forward(X, names)
    chosen_b = np.argmax(cb["q"], axis=1)
    off_b = float((chosen_b == forbidden).mean())
    forb_mass_b = float(cb["q"][np.arange(B), forbidden].max())
    print("    -- now the same engine with the 作用是性 side channel open --")
    print(f"    max probability on a FORBIDDEN verdict   {forb_mass_b:.3e}")
    print(f"    rate of choosing a forbidden verdict     {off_b:.6f}")
    print("    The gap does not merely make it less accurate. It makes it able")
    print("    to hand down a ruling the code does not contain. That is a")
    print("    different kind of failure, and no amount of training removes it.")
    ok4 = (forb_mass == 0.0) and (off_b > 0.0)

    # ---------------------------------------------------------------- TEST 5
    rule("TEST 5 — AMENDING THE CODE WITHOUT RE-EDUCATING THE OFFICIALS")
    print("    The Joseon Gyeongguk-jeon was written to be revised. Here the")
    print("    canon is replaced at inference time. No retraining, no gradient")
    print("    step, no fine-tune. The mind's job never changed: read the form.")
    CANON2 = amend_canon(CANON, seed=2026, fraction=0.55)
    changed = float((np.argmax(CANON, axis=2) != np.argmax(CANON2, axis=2)).mean())
    m_amend = evaluate(net, test_data, canon=CANON2)
    print(f"    fraction of code entries changed {changed:.3f}")
    print(f"    engine accuracy under new code   {m_amend['verdict_acc']:.3f}"
          f"   (was {m['verdict_acc']:.3f})")

    direct = DirectVerdictNet(hidden=48, seed=1)
    Xtr, ntr, ftr, vtr, _ = train_data
    r = np.random.default_rng(5)
    for ep in range(40):
        order = r.permutation(len(Xtr))
        for s in range(0, len(Xtr), 64):
            i = order[s:s + 64]
            direct.step(Xtr[i], ntr[i], vtr[i], lr=0.10)
    q_d, _, _ = direct.forward(X, names)
    d_old = float((np.argmax(q_d, axis=1) == verdicts).mean())
    v_new = np.array([int(np.argmax(CANON2[names[i], forms[i]]))
                      for i in range(len(forms))])
    d_new = float((np.argmax(q_d, axis=1) == v_new).mean())
    print(f"    心跡不分 baseline, original code {d_old:.3f}")
    print(f"    心跡不分 baseline, amended code  {d_new:.3f}"
          f"   (chance {1/N_VERDICTS:.3f})")
    print("    The baseline is not stupid. It is competent and unamendable: it")
    print("    absorbed the law into its weights, so the law can only be changed")
    print("    by retraining it. That is the practical cost of the gap.")
    ok5 = m_amend['verdict_acc'] > 0.6 and d_new < d_old - 0.2

    # ---------------------------------------------------------------- TEST 6
    rule("TEST 6 — 辯: THE SELF-REPORT, MEASURED AND FOUND WANTING")
    s_rep = c["s_rep"]
    correct = (np.argmax(c["p"], axis=1) == forms).astype(float)
    print(f"    mean self-reported confidence, plain cases  {s_rep[~is_trap].mean():.3f}")
    print(f"    actual accuracy,               plain cases  {correct[~is_trap].mean():.3f}")
    print(f"    mean self-reported confidence, TRAP cases   {s_rep[is_trap].mean():.3f}")
    print(f"    actual accuracy,               TRAP cases   {correct[is_trap].mean():.3f}")
    gap = float(s_rep[is_trap].mean() - correct[is_trap].mean())
    print(f"    overconfidence gap on traps                 {gap:+.3f}")
    print("    This is 作用非性 measured. The report is not a lie; it is the")
    print("    accurate output of a bright, well-functioning organ that has")
    print("    misread the case. Which is exactly why it holds no authority")
    print("    here — it is logged, and it decides nothing.")
    ok6 = gap > 0.05

    # ---------------------------------------------------------------- TEST 7
    rule("TEST 7 — 諫: REMONSTRANCE FROM A DIFFERENT PAIR OF EYES")
    for use in (False, True):
        mm = evaluate(net, test_data, use_gan=use)
        act, truth = mm["act"], mm["truth"]
        # the costly error: endorsing (承) something the code says to strike (罷)
        harmful = float(((act == 0) & (truth == 2)).mean())
        escalated = float((act == 3).mean())
        label = "with 諫 " if use else "without"
        print(f"    {label}   verdict acc {mm['verdict_acc']:.3f}   "
              f"wrongful endorsement {harmful:.4f}   escalated {escalated:.3f}")
    m_no = evaluate(net, test_data, use_gan=False)
    m_ye = evaluate(net, test_data, use_gan=True)
    h_no = float(((m_no["act"] == 0) & (m_no["truth"] == 2)).mean())
    h_ye = float(((m_ye["act"] == 0) & (m_ye["truth"] == 2)).mean())
    print(f"    wrongful endorsements cut by {100*(1-h_ye/max(h_no,1e-9)):.1f}%,")
    print("    paid for in escalations. Note the critic has its own encoder: it")
    print("    is not the mind grading itself, which is the whole point of")
    print("    putting Principle outside the quarrel in the Simgiri-pyeon.")
    ok7 = h_ye < h_no

    # ---------------------------------------------------------------- TEST 8
    rule("TEST 8 — 放伐: DEPOSING THE OFFICE-HOLDER, KEEPING THE OFFICE")
    print("    The sitting head is corrupted at epoch 20. Same seed, same data,")
    print("    same canon; the only difference is whether removal is permitted.")
    res = {}
    for allow in (False, True):
        n2 = SambongEngine(hidden=48, gan_hidden=64, n_heads=3, seed=1)
        train(n2, train_data, epochs=34, lr=0.10, deposition=allow,
              sabotage_at=20, verbose=allow, log_every=100)
        mm = evaluate(n2, test_data)
        res["with" if allow else "without"] = mm["verdict_acc"]
        print(f"      deposition {'PERMITTED ' if allow else 'FORBIDDEN '}"
              f"-> verdict accuracy {mm['verdict_acc']:.3f}   "
              f"(head {n2.reign} in office)")
    ok8 = res["with"] >= res["without"]
    print("    The canon was never rewritten to accommodate a bad head. That")
    print("    asymmetry — replaceable holder, durable office — is the thing")
    print("    Jeong died defending in the eighth month of 1398.")

    # ---------------------------------------------------------------- SUMMARY
    rule("SUMMARY")
    checks = [("1 gradient check", ok1),
              ("4 no verdict outside the canon (and the bypass breaks it)", ok4),
              ("5 code amendable without retraining", ok5),
              ("6 self-report overconfident precisely where it is wrong", ok6),
              ("7 remonstrance reduces wrongful endorsement", ok7),
              ("8 deposition repairs a corrupted office-holder", ok8)]
    for nm, v in checks:
        print(f"    [{'PASS' if v else 'FAIL'}]  {nm}")
    print(f"\n    {sum(v for _, v in checks)}/{len(checks)} checks pass")
    print("=" * 78)


if __name__ == "__main__":
    main()
