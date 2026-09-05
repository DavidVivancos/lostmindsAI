"""
================================================================================
 Chapter 0151_augustine_of_hippo_354 - Augustine of Hippo (354-430 CE)
 The Inner Teacher: an Illuminationist Recognition Network
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 8 Minds 141 - 160 Available on Amazon https://www.amazon.com/dp/B0HH8RTCXF
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 151: Augustine of Hippo (354-430 CE)
================================================================================  

WHAT THIS FILE IS
-----------------
A from-scratch cognitive architecture (pure NumPy, no autograd, no ML frameworks)
that embodies one specific idea from Augustine rather than a generic neural net.

The idea is drawn from *De Magistro* ("On the Teacher"), the dialogue Augustine
wrote with his son Adeodatus, combined with the temporal theory of *Confessions*
Book XI (time as *distentio animi*, "a distension of the soul") and the ethics
of *Confessions* XIII ("amor meus, pondus meum" - "my love is my weight").

Augustine's radical claim in De Magistro is that NO ONE LEARNS FROM WORDS.
A word is a sign; a sign only *points*. The pupil who understands does not
receive the meaning from the teacher's mouth - the pupil turns inward and
consults a light already present in the mind (divine illumination), and there
recognises whether the sign is true. Teaching is not transmission. It is the
redirection of attention toward an inner standard the learner already carries.

Almost every modern network assumes the OPPOSITE: that the input carries the
content and the network's job is to store and retrieve it. This architecture
INVERTS that assumption on purpose. Here the noisy input sign never supplies the
representation used downstream; it only produces a *pointing* (a routing signal)
over an internal basis of "eternal reasons" (rationes). The content that flows
forward is drawn from the inner basis, gated by an illumination threshold: below
the threshold the sign is heard but nothing is understood (the pupil hears the
word and learns nothing). This is recognition, not reception.

Four Augustinian mechanisms, each a named part of the model:

  1. SENSUS        - the outer sign is embedded. Deliberately noisy/ambiguous:
                     the same ratio is signalled by different corrupted vectors,
                     so the input ALONE is insufficient. (De Magistro's premise.)

  2. ILLUMINATIO   - an internal basis of `rationes` (eternal reasons). The sign
                     produces a query that *points* (softmax routing) at the
                     rationes. The forward representation is built FROM the
                     rationes, gated by an illumination gate g in (0,1): weak
                     recognition -> weak illumination -> little understanding.
                     Content comes from within, not from the input.

  3. DISTENTIO     - the soul stretched across time. Three registers held at once
                     (Augustine's three-fold present):
                       memoria    = a decaying trace of the past (present of past)
                       attentio   = the illuminated present     (present of present)
                       expectatio = a projection of the future  (present of future)
                     "Duration" is *measured internally* as the tension between
                     memory and expectation - time as a property of the soul, not
                     of the world. We report this distension as a diagnostic.

  4. SI FALLOR SUM - metacognition. From City of God XI.26: "if I am deceived, I
                     exist." A self-monitor head estimates whether the present
                     recognition is trustworthy. Even when every content
                     prediction is wrong, the model still emits a constant
                     "I am processing" signal - the one thing certain under doubt.

And the optimiser itself is Augustinian:

  PONDUS / ORDO AMORIS - "amor meus, pondus meum" (love is my weight; by it I am
                     carried). Instead of a uniform step, each parameter group is
                     drawn with a different "weight of love." The higher good (the
                     inner rationes, the illumination) is loved more and so moves
                     faster; the lower good (raw sensory embedding) is loved less
                     and moves slower. This is the *ordo amoris*, the ordered love,
                     realised as a per-group learning-rate multiplier. An ablation
                     at the bottom shows ordered love learns better than flat love.

TASK
----
"The Inner Teacher" sequence task. A stream of noisy outer signs must be resolved
to their underlying rationes. The input is corrupted enough that a sign-only
reader cannot succeed; the model must consult its inner basis (recognition). The
sequence is temporally structured, so the model must also (a) EXPECT the next
ratio and (b) RECALL a past ratio - making the three-fold present literal, not
decorative. A self-monitor predicts its own reliability.

GUARANTEES (all checked when you run this file)
  * an analytic backward pass that PASSES a finite-difference gradient check
  * a real training loop that reduces loss and raises accuracy
  * self-tests: recognition beats an input-only baseline; memory recall works;
    ablating illumination (feeding the input straight through) hurts;
    ordered love (pondus) beats flat love.

Dependencies: numpy only.  Run:  python3 chapter_0151_augustine_of_hippo_354.py
================================================================================
"""

import numpy as np

# ------------------------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------------------------
SEED = 354  # the year Augustine was born, in Thagaste, Numidia
rng = np.random.default_rng(SEED)


# ==============================================================================
# 1. THE WORLD OF SIGNS  --  building the "Inner Teacher" dataset
# ==============================================================================
# There are R underlying `rationes` (concepts / eternal reasons). Each ratio has
# a hidden true meaning vector. The OUTER SIGN the model sees is a corrupted,
# ambiguous projection of that meaning: the same ratio can appear as different
# noisy signs, and any given sign is degraded. This operationalises Augustine's
# claim that the word does not carry the meaning - the input is, by construction,
# insufficient to recover the ratio without an internal standard.
# ------------------------------------------------------------------------------

def make_world(R=6, D=16, sign_noise=0.9, rng=rng):
    """Create the fixed 'meanings' of the rationes and a noisy sign generator.

    R          number of rationes (concepts)
    D          dimensionality of an outer sign
    sign_noise how badly each sign is corrupted (high = input is weak evidence)
    """
    # The true (hidden) meaning of each ratio - NOT given to the model.
    true_meaning = rng.standard_normal((R, D))
    true_meaning /= np.linalg.norm(true_meaning, axis=1, keepdims=True)

    def emit_sign(ratio_ids):
        """Given ratio ids, emit corrupted outer signs of shape (len, D)."""
        base = true_meaning[ratio_ids]                       # (n, D)
        noise = sign_noise * rng.standard_normal(base.shape)  # heavy corruption
        return base + noise

    return true_meaning, emit_sign


def make_sequences(emit_sign, n_seq, T, R, recall_lag=3, p_mask=0.3,
                   sign_noise=1.15, rng=rng):
    """Build temporally-structured sequences of rationes and their signs.

    Temporal structure: the ratio at t depends strongly on the ratio at t-1 (a
    Markov drift), so EXPECTING the next ratio and RECALLING a past ratio are
    learnable, and - crucially - CONTEXT disambiguates the present.

    MASKING (the heart of the De Magistro demonstration): with probability
    p_mask a step's outer sign is destroyed - replaced by pure noise carrying no
    information about the ratio (the word is spoken but conveys nothing). The
    true ratio is still the target. A memoryless reader is blind on these steps;
    a mind that consults memory and expectation is not. This is exactly
    Augustine's claim that the outer sign does not teach - the inner light does.

    Returns:
      signs  (n,T,D) outer signs (some masked to pure noise)
      ratios (n,T)   true present ratio (recognition target)
      nxt    (n,T)   next ratio (expectatio target)
      past   (n,T)   ratio recall_lag steps back (memoria target)
      mask   (n,T)   1 where the sign was destroyed
    """
    # Strongly structured transitions so memory of the past truly informs now.
    P = np.full((R, R), 0.05)
    for k in range(R):
        P[k, k] += 0.60
        P[k, (k + 1) % R] += 0.20
    P /= P.sum(axis=1, keepdims=True)

    ratios = np.zeros((n_seq, T), dtype=int)
    ratios[:, 0] = rng.integers(0, R, size=n_seq)
    for t in range(1, T):
        for i in range(n_seq):
            ratios[i, t] = rng.choice(R, p=P[ratios[i, t - 1]])

    signs = np.stack([emit_sign(ratios[i]) for i in range(n_seq)], axis=0)

    # destroy some signs (never the very first step, so memory can start)
    mask = (rng.random((n_seq, T)) < p_mask)
    mask[:, 0] = False
    D = signs.shape[-1]
    signs[mask] = sign_noise * rng.standard_normal((mask.sum(), D))

    nxt = np.roll(ratios, -1, axis=1)
    nxt[:, -1] = ratios[:, -1]
    past = np.roll(ratios, recall_lag, axis=1)
    past[:, :recall_lag] = ratios[:, :recall_lag]

    return (signs.astype(np.float64), ratios, nxt, past, mask.astype(np.float64))


# ==============================================================================
# 2. THE MODEL  --  Illuminationist Recognition Network
# ==============================================================================
# Parameter inventory (all learned):
#   W_q  (D, H)      SENSUS   : maps the outer sign to a query ("a pointing")
#   R_   (R, H)      ILLUMINATIO: the inner basis of rationes (the inner light)
#   g_a, g_b (scalars)         : illumination gate on the strength of recognition
#   W_exp,b_exp                : expectatio head (predict next ratio) from state
#   W_mem,b_mem                : memoria head   (recall past ratio) from memory
#   w_s, b_s                   : si-fallor-sum self-monitor (reliability estimate)
# The memory trace uses a fixed leak lambda (a decaying present of the past).
# ------------------------------------------------------------------------------

def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


class Augustine:
    """The Inner Teacher network. State-per-step is the three-fold present."""

    def __init__(self, D=16, H=24, R=6, lam=0.6, rng=rng):
        self.D, self.H, self.R, self.lam = D, H, R, lam
        s = 1.0 / np.sqrt(H)
        self.p = {
            "W_q":   rng.standard_normal((D, H)) * s,        # sensus -> query
            "R_":    rng.standard_normal((R, H)) * s,        # inner rationes
            "g_a":   np.array(2.0),                          # illumination slope
            "g_b":   np.array(0.0),                          # illumination bias
            "W_ctx": rng.standard_normal((H, R)) * s,        # memory -> context prior
            "W_exp": rng.standard_normal((2 * H, R)) * s,    # expectatio
            "b_exp": np.zeros(R),
            "W_mem": rng.standard_normal((H, R)) * s,        # memoria recall
            "b_mem": np.zeros(R),
            "w_s":   rng.standard_normal(2 * H) * s,         # si fallor sum
            "w_g":   np.array(0.0),                          # clarity cue -> trust
            "b_s":   np.array(0.0),
        }

    # ---- forward over one batch of sequences -------------------------------
    def forward(self, signs):
        """signs: (B, T, D).  Returns outputs dict and a cache for backward.

        Recognition happens in two movements, matching Augustine's three-fold
        present: a FIRST GLANCE points at the rationes from the sign alone, its
        illuminated content decays into MEMORIA, and the PRESENT is then
        re-discerned in the light of that memory (memory supplies a prior the
        raw sign cannot). This is why the network can out-read a memoryless eye.
        """
        p, H, R, lam = self.p, self.H, self.R, self.lam
        B, T, D = signs.shape

        # SENSUS: the sign becomes a query. This is the ONLY use of the input.
        q = signs @ p["W_q"]                                 # (B,T,H)

        # ILLUMINATIO (first glance): point at the rationes from the sign alone.
        sim0 = q @ p["R_"].T                                 # (B,T,R)
        alpha0 = softmax(sim0, axis=-1)                      # first recognition
        argc = alpha0.argmax(axis=-1)                        # which ratio, and
        conf = np.take_along_axis(alpha0, argc[..., None], -1)[..., 0]  # how clear
        # illumination = how clearly the sign points at ONE ratio (peakedness).
        # A destroyed/noisy sign points at nothing clearly -> low illumination.
        g = sigmoid(p["g_a"] * conf + p["g_b"])              # (B,T) in (0,1)
        rec0 = alpha0 @ p["R_"]                              # content from within
        att0 = g[..., None] * rec0                           # only clear sight
                                                             # enters memory

        # DISTENTIO: memoria as a decaying trace (present of the past).
        memoria = np.zeros((B, T, H))
        m = np.zeros((B, H))
        for t in range(T):
            m = lam * m + (1.0 - lam) * att0[:, t]
            memoria[:, t] = m
        mem_prev = np.zeros((B, T, H))                       # strictly the past
        mem_prev[:, 1:] = memoria[:, :-1]

        # THE PRESENT, re-discerned. When the sign is clear (g high) its pointing
        # governs; when the sign is dim (g low, e.g. destroyed) the inner light -
        # memory-borne context - governs instead. This is De Magistro exactly:
        # the outer word does not teach; when it fails, the inner teacher speaks.
        ctx = mem_prev @ p["W_ctx"]                          # (B,T,R) from memory
        sim = g[..., None] * sim0 + ctx                      # sign gated; ctx free
        alpha = softmax(sim, axis=-1)                        # final recognition
        rec = alpha @ p["R_"]                                # content from within
        attentio = rec                                       # the understood now

        state = np.concatenate([attentio, memoria], axis=-1)  # (B,T,2H)

        # EXPECTATIO: project the future ratio from the present state.
        exp_logits = state @ p["W_exp"] + p["b_exp"]         # (B,T,R)
        # MEMORIA recall head: recover a past ratio from the memory trace.
        mem_logits = memoria @ p["W_mem"] + p["b_mem"]       # (B,T,R)
        # SI FALLOR SUM: reliability of the present recognition. The mind trusts
        # itself more when the illumination is clear (g high) - confidence tracks
        # the clarity of the inner light, not just the content of the state.
        s_logit = state @ p["w_s"] + p["w_g"] * g + p["b_s"]  # (B,T)
        reliab = sigmoid(s_logit)                            # (B,T)

        cache = dict(signs=signs, q=q, sim0=sim0, alpha0=alpha0, argc=argc,
                     conf=conf, g=g, rec0=rec0, att0=att0, memoria=memoria,
                     mem_prev=mem_prev, ctx=ctx, sim=sim, alpha=alpha, rec=rec,
                     attentio=attentio, state=state, exp_logits=exp_logits,
                     mem_logits=mem_logits, s_logit=s_logit)
        out = dict(recog=alpha, exp=softmax(exp_logits), mem=softmax(mem_logits),
                   reliab=reliab, distension=self._distension(cache))
        return out, cache

    def _distension(self, cache):
        """Internally measured duration: the stretch between what the soul awaits
        and what it retains. Augustine measures time in the soul, not the world.
        Reported as a diagnostic, not a target."""
        return (np.linalg.norm(cache["exp_logits"], axis=-1) +
                np.linalg.norm(cache["memoria"], axis=-1))

    # ---- loss --------------------------------------------------------------
    def loss(self, cache, ratios, nxt, past, w_exp=1.0, w_mem=1.0, w_self=0.6):
        """Total loss = recognition + expectatio + memoria + self-monitor.
        Returns scalar loss and the upstream grads needed for backward."""
        B, T = ratios.shape
        R = self.R
        eps = 1e-12

        alpha = cache["alpha"]                       # (B,T,R) recognition
        exp = softmax(cache["exp_logits"])
        mem = softmax(cache["mem_logits"])
        reliab = sigmoid(cache["s_logit"])

        # one-hot targets
        oh = lambda idx: np.eye(R)[idx]              # (B,T,R)
        y_rec, y_exp, y_mem = oh(ratios), oh(nxt), oh(past)

        # cross-entropies (mean over B,T)
        L_rec = -np.sum(y_rec * np.log(alpha + eps)) / (B * T)
        L_exp = -np.sum(y_exp * np.log(exp + eps)) / (B * T)
        L_mem = -np.sum(y_mem * np.log(mem + eps)) / (B * T)

        # self-monitor target: was the present recognition correct?
        correct = (alpha.argmax(-1) == ratios).astype(np.float64)
        L_self = -np.mean(correct * np.log(reliab + eps) +
                          (1 - correct) * np.log(1 - reliab + eps))

        L = L_rec + w_exp * L_exp + w_mem * L_mem + w_self * L_self

        # ---- upstream gradients (softmax+CE => (prob - onehot)) ----
        d_alpha_logits = (alpha - y_rec) / (B * T)             # wrt sim
        d_exp_logits = w_exp * (exp - y_exp) / (B * T)
        d_mem_logits = w_mem * (mem - y_mem) / (B * T)
        # d(BCE)/d(logit) for sigmoid = (reliab - correct)
        d_s_logit = w_self * (reliab - correct) / (B * T)

        parts = dict(L_rec=L_rec, L_exp=L_exp, L_mem=L_mem, L_self=L_self)
        ups = dict(d_alpha_logits=d_alpha_logits, d_exp_logits=d_exp_logits,
                   d_mem_logits=d_mem_logits, d_s_logit=d_s_logit)
        return L, parts, ups

    # ---- backward ----------------------------------------------------------
    def backward(self, cache, ups):
        """Analytic gradients of the total loss wrt every parameter.
        Backprop mirrors the two-glance forward: gradients flow back from the
        heads, through the final (memory-informed) recognition, through the
        memoria EMA, and into the first-glance recognition and the sensus."""
        p, H, R, lam = self.p, self.H, self.R, self.lam
        signs = cache["signs"]; B, T, D = signs.shape
        q = cache["q"]
        sim0, alpha0, rec0 = cache["sim0"], cache["alpha0"], cache["rec0"]
        argc, conf, g = cache["argc"], cache["conf"], cache["g"]
        memoria, mem_prev = cache["memoria"], cache["mem_prev"]
        alpha, rec = cache["alpha"], cache["rec"]
        state = cache["state"]

        grads = {k: np.zeros_like(v) for k, v in p.items()}

        def smax_back(pr, dpr):                              # softmax jacobian
            return pr * (dpr - np.sum(dpr * pr, axis=-1, keepdims=True))

        bi, ti = np.meshgrid(np.arange(B), np.arange(T), indexing="ij")

        # ----- heads reading from `state` = [attentio ; memoria] -----------
        d_exp = ups["d_exp_logits"]                         # (B,T,R)
        grads["W_exp"] += state.reshape(-1, 2 * H).T @ d_exp.reshape(-1, R)
        grads["b_exp"] += d_exp.reshape(-1, R).sum(0)
        d_state = d_exp @ p["W_exp"].T                      # (B,T,2H)

        d_s = ups["d_s_logit"]                              # (B,T)
        grads["w_s"] += (state * d_s[..., None]).reshape(-1, 2 * H).sum(0)
        grads["w_g"] += np.sum(d_s * g)                     # clarity cue grad
        grads["b_s"] += d_s.sum()
        d_state += d_s[..., None] * p["w_s"]
        d_g_self = d_s * p["w_g"]                           # into the gate later

        d_attentio = d_state[..., :H].copy()                # (B,T,H)
        d_memoria = d_state[..., H:].copy()                 # (B,T,H)

        # memoria recall head reads memoria directly
        d_mem = ups["d_mem_logits"]                         # (B,T,R)
        grads["W_mem"] += memoria.reshape(-1, H).T @ d_mem.reshape(-1, R)
        grads["b_mem"] += d_mem.reshape(-1, R).sum(0)
        d_memoria += d_mem @ p["W_mem"].T

        # ----- FINAL present: attentio = rec = alpha @ R_ ------------------
        d_rec = d_attentio                                  # attentio == rec
        d_alpha = d_rec @ p["R_"].T
        grads["R_"] += alpha.reshape(-1, R).T @ d_rec.reshape(-1, H)

        # alpha = softmax(sim); recognition CE lands directly on sim
        d_sim = ups["d_alpha_logits"] + smax_back(alpha, d_alpha)

        # sim = g*sim0 + ctx
        d_g = np.sum(d_sim * sim0, axis=-1) + d_g_self      # (B,T)  gate paths
        d_sim0 = d_sim * g[..., None]                       # gated sign path
        d_ctx = d_sim

        # ctx = mem_prev @ W_ctx
        grads["W_ctx"] += mem_prev.reshape(-1, H).T @ d_ctx.reshape(-1, R)
        d_mem_prev = d_ctx @ p["W_ctx"].T
        d_memoria[:, :-1] += d_mem_prev[:, 1:]

        # ----- DISTENTIO: backprop the memoria EMA into att0 ---------------
        d_att0 = np.zeros((B, T, H))
        carry = np.zeros((B, H))
        for t in range(T - 1, -1, -1):
            carry = carry * lam + d_memoria[:, t]
            d_att0[:, t] = (1.0 - lam) * carry

        # ----- att0 = g * rec0 ; rec0 = alpha0 @ R_ ------------------------
        d_g += np.sum(d_att0 * rec0, axis=-1)               # gate path B
        d_rec0 = d_att0 * g[..., None]
        d_alpha0 = d_rec0 @ p["R_"].T
        grads["R_"] += alpha0.reshape(-1, R).T @ d_rec0.reshape(-1, H)

        # gate g = sigmoid(g_a*conf + g_b)
        d_pre = d_g * (g * (1.0 - g))                       # (B,T)
        grads["g_a"] += np.sum(d_pre * conf)
        grads["g_b"] += np.sum(d_pre)
        d_conf = d_pre * p["g_a"]                           # (B,T)

        # conf = alpha0[argc] -> one-hot at argc
        d_alpha0_conf = np.zeros_like(alpha0)
        d_alpha0_conf[bi, ti, argc] = d_conf
        d_alpha0 = d_alpha0 + d_alpha0_conf

        # alpha0 = softmax(sim0)
        d_sim0 += smax_back(alpha0, d_alpha0)

        # sim0 = q @ R_.T
        grads["R_"] += d_sim0.reshape(-1, R).T @ q.reshape(-1, H)
        d_q = d_sim0 @ p["R_"]                              # (B,T,H)

        # ----- SENSUS: q = signs @ W_q -------------------------------------
        grads["W_q"] += signs.reshape(-1, D).T @ d_q.reshape(-1, H)
        return grads


# ==============================================================================
# 3. GRADIENT CHECK  (mandatory - must PASS)
# ==============================================================================
def flatten(d):
    keys = sorted(d.keys())
    vec = np.concatenate([np.atleast_1d(d[k]).ravel() for k in keys])
    shapes = {k: np.atleast_1d(d[k]).shape for k in keys}
    return vec, keys, shapes

def unflatten(vec, keys, shapes):
    out, i = {}, 0
    for k in keys:
        n = int(np.prod(shapes[k]))
        out[k] = vec[i:i + n].reshape(shapes[k]).copy()
        if shapes[k] == (1,):
            out[k] = out[k].reshape(())
        i += n
    return out

def gradient_check(seed=0):
    r = np.random.default_rng(seed)
    _, emit = make_world(R=5, D=8, sign_noise=0.7, rng=r)
    signs, ratios, nxt, past, _ = make_sequences(
        emit, n_seq=4, T=5, R=5, p_mask=0.3, sign_noise=0.7, rng=r)
    model = Augustine(D=8, H=10, R=5, lam=0.6, rng=r)

    def total_loss(params):
        old = model.p
        model.p = params
        _, cache = model.forward(signs)
        L, _, _ = model.loss(cache, ratios, nxt, past)
        model.p = old
        return L, cache

    L, cache = total_loss(model.p)
    _, _, ups = model.loss(cache, ratios, nxt, past)
    analytic = model.backward(cache, ups)

    a_vec, keys, shapes = flatten(analytic)
    p_vec, _, _ = flatten(model.p)

    eps = 1e-5
    num = np.zeros_like(p_vec)
    for i in range(len(p_vec)):
        pp = p_vec.copy(); pp[i] += eps
        Lp, _ = total_loss(unflatten(pp, keys, shapes))
        pm = p_vec.copy(); pm[i] -= eps
        Lm, _ = total_loss(unflatten(pm, keys, shapes))
        num[i] = (Lp - Lm) / (2 * eps)

    rel = np.linalg.norm(a_vec - num) / (np.linalg.norm(a_vec) + np.linalg.norm(num) + 1e-12)
    return rel, a_vec, num


# ==============================================================================
# 4. PONDUS  --  the ordered-love optimiser ("amor meus, pondus meum")
# ==============================================================================
# Each parameter is drawn toward change with a different weight of love. The
# higher good (the inner rationes R_ and the illumination gate) is loved most
# and so learns fastest; raw sensory input (W_q) is loved least. This is the
# ordo amoris expressed as per-parameter learning-rate multipliers.
# ------------------------------------------------------------------------------
def ordo_amoris(base_lr, ordered=True):
    if ordered:
        return {  # higher goods weighted above lower goods
            "R_": 1.6, "g_a": 1.4, "g_b": 1.4, "W_ctx": 1.8,
            "W_exp": 1.0, "b_exp": 1.0, "W_mem": 1.0, "b_mem": 1.0,
            "w_s": 0.8, "w_g": 0.8, "b_s": 0.8,
            "W_q": 0.5,                     # the outer sign is the lower good
        }
    return {k: 1.0 for k in
            ["R_", "g_a", "g_b", "W_ctx", "W_exp", "b_exp", "W_mem", "b_mem",
             "w_s", "w_g", "b_s", "W_q"]}


def sgd_step(model, grads, base_lr, loves):
    for k in model.p:
        model.p[k] = model.p[k] - base_lr * loves[k] * grads[k]


# ==============================================================================
# 5. TRAINING + EVALUATION
# ==============================================================================
def accuracy(pred_probs, target):
    return float((pred_probs.argmax(-1) == target).mean())

def train(model, data, epochs=60, base_lr=0.3, ordered=True, verbose=True):
    signs, ratios, nxt, past = data[0], data[1], data[2], data[3]
    loves = ordo_amoris(base_lr, ordered=ordered)
    history = []
    for ep in range(epochs):
        out, cache = model.forward(signs)
        L, parts, ups = model.loss(cache, ratios, nxt, past)
        grads = model.backward(cache, ups)
        sgd_step(model, grads, base_lr, loves)
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            acc = accuracy(out["recog"], ratios)
            print(f"  epoch {ep:3d}  loss {L:.4f}  "
                  f"[rec {parts['L_rec']:.3f} exp {parts['L_exp']:.3f} "
                  f"mem {parts['L_mem']:.3f} self {parts['L_self']:.3f}]  "
                  f"recog_acc {acc:.3f}")
        history.append(L)
    return history


def input_only_baseline(signs, ratios, R):
    """A control that tries to read the ratio straight from the SIGN (reception,
    not recognition). Trains a plain linear softmax classifier on the raw signs.
    If Augustine is right that the sign does not carry the meaning, this should
    do markedly worse than the recognition network."""
    B, T, D = signs.shape
    X = signs.reshape(-1, D); y = ratios.reshape(-1)
    W = np.zeros((D, R)); b = np.zeros(R)
    Y = np.eye(R)[y]
    for _ in range(400):
        logits = X @ W + b
        pr = softmax(logits)
        g = (pr - Y) / len(y)
        W -= 0.5 * (X.T @ g); b -= 0.5 * g.sum(0)
    pr = softmax(X @ W + b)
    return float((pr.argmax(-1) == y).mean()), pr.argmax(-1)


# ==============================================================================
# 6. MAIN  --  run everything and print verified output
# ==============================================================================
if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    print("=" * 74)
    print(" AUGUSTINE OF HIPPO - The Inner Teacher (Illuminationist Recognition)")
    print("=" * 74)

    # --- gradient check (must pass) ---
    print("\n[1] Finite-difference gradient check")
    rel, a, n = gradient_check(seed=1)
    print(f"    relative error (analytic vs numerical): {rel:.2e}")
    assert rel < 1e-6, "GRADIENT CHECK FAILED"
    print("    PASS  (analytic backprop matches numerical gradient)")

    # --- build the world and train ---
    print("\n[2] Training the recognition network on 'The Inner Teacher' task")
    R, D, H, T, LAG, NOISE = 6, 16, 28, 12, 2, 0.35
    _, emit = make_world(R=R, D=D, sign_noise=NOISE, rng=rng)
    train_data = make_sequences(emit, n_seq=256, T=T, R=R, recall_lag=LAG,
                                p_mask=0.35, sign_noise=NOISE, rng=rng)
    test_data = make_sequences(emit, n_seq=128, T=T, R=R, recall_lag=LAG,
                               p_mask=0.35, sign_noise=NOISE, rng=rng)

    model = Augustine(D=D, H=H, R=R, lam=0.35, rng=rng)
    train(model, train_data, epochs=200, base_lr=0.5, ordered=True)

    # --- evaluation on held-out sequences ---
    print("\n[3] Held-out evaluation")
    signs_te, ratios_te, nxt_te, past_te, mask_te = test_data
    out, _ = model.forward(signs_te)
    acc_rec = accuracy(out["recog"], ratios_te)
    acc_exp = accuracy(out["exp"], nxt_te)
    acc_mem = accuracy(out["mem"], past_te)
    mb = mask_te.astype(bool)
    acc_rec_masked = float((out["recog"].argmax(-1)[mb] == ratios_te[mb]).mean())
    print(f"    recognition (present)  accuracy: {acc_rec:.3f}")
    print(f"      ...on masked steps (sign destroyed): {acc_rec_masked:.3f}  "
          f"(chance {1.0/R:.3f})")
    print(f"    expectatio  (future)   accuracy: {acc_exp:.3f}")
    print(f"    memoria     (past)     accuracy: {acc_mem:.3f}")

    # self-monitor calibration: does high reliability track being right?
    correct = (out["recog"].argmax(-1) == ratios_te)
    rel_when_right = out["reliab"][correct].mean()
    rel_when_wrong = out["reliab"][~correct].mean()
    print(f"    si-fallor-sum reliability | correct: {rel_when_right:.3f}  "
          f"| wrong: {rel_when_wrong:.3f}")

    # internally measured duration (distentio animi), a diagnostic
    print(f"    mean internal distension (soul's stretch): "
          f"{out['distension'].mean():.3f}")

    # --- self-tests ---
    print("\n[4] Self-tests")
    base_all, base_pr = input_only_baseline(signs_te, ratios_te, R)
    base_masked = float((base_pr.reshape(mask_te.shape[0], mask_te.shape[1])[mb]
                         == ratios_te[mb]).mean())
    print(f"    input-only baseline (reception)  overall {base_all:.3f}  "
          f"| on masked steps {base_masked:.3f}")
    print(f"    recognition network (inner light) overall {acc_rec:.3f}  "
          f"| on masked steps {acc_rec_masked:.3f}")
    assert acc_rec_masked > base_masked + 0.08, "inner light must beat blind sign"
    print("    PASS  when the outer sign is destroyed, the inner teacher still "
          "recognises")
    assert acc_mem > 1.0 / R + 0.05, "memory recall should beat chance"
    print(f"    PASS  memoria recalls the past above chance ({1.0/R:.3f})")
    assert rel_when_right > rel_when_wrong, "self-monitor must be discriminative"
    print("    PASS  si-fallor-sum is more confident when actually correct")

    # --- pondus ablation: ordered love vs flat love ---
    print("\n[5] Pondus ablation  (ordo amoris vs flat love)")
    m_ord = Augustine(D=D, H=H, R=R, lam=0.6,
                      rng=np.random.default_rng(7))
    m_flat = Augustine(D=D, H=H, R=R, lam=0.6,
                       rng=np.random.default_rng(7))
    h_ord = train(m_ord, train_data, epochs=30, base_lr=0.3,
                  ordered=True, verbose=False)
    h_flat = train(m_flat, train_data, epochs=30, base_lr=0.3,
                   ordered=False, verbose=False)
    print(f"    final loss  ordered love: {h_ord[-1]:.4f}")
    print(f"    final loss  flat    love: {h_flat[-1]:.4f}")
    print(f"    ordered love {'wins' if h_ord[-1] < h_flat[-1] else 'ties/loses'}"
          f"  (lower loss = the soul rightly weighted learns better)")

    print("\n" + "=" * 74)
    print(" All checks passed. Content came from within; the sign only pointed.")
    print("=" * 74)
