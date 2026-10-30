#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 chapter_0335_sundiata_keita_1217.py 
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0335 · Sundiata Keita (1217-1255)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 17, Minds 321-340 Available on Amazon https://www.amazon.com/dp/B0HP7Z1NKC
# Resume and Interactive Demos at https://artificiology.com/
================================================================================

 THE CONVENED FACULTY
 --------------------
 An executable, trainable architecture built from one idea that belongs to the
 Mande world and to almost nobody else: CAPABILITY IS NOT A PROPERTY AN AGENT
 HAS. It is a thing that other parties convene, licence, witness and speak on
 the agent's behalf -- and every one of those four moves is a separate,
 auditable mechanism that can be switched off and measured.

 The four commitments, each drawn from a documented feature of the Manden and
 each realised here as a distinct piece of machinery rather than a metaphor:

   (1) LATENCY IS NOT INCAPACITY.  The epic seats Sunjata on the ground for
       seven years. He is not missing the faculty; he is missing the licence to
       use it. Implemented as the SeatedBank: a large parameter bank that is
       multiplicatively gated by a credential key produced OUTSIDE it. Zero the
       key and the bank falls silent while its weights are untouched. The same
       parameters, differently licensed, are a different agent.

   (2) EMISSION IS DELEGATED.  Ibn Battuta, at the court of Mansa Sulayman in
       1352, records that the mansa answered in his own tongue and the answer
       was carried to the petitioner by others; the interpreter Dugha is named
       as one of the principal men of the realm, seated on his own chair, paid
       in public with the sum announced aloud. Implemented as the Jeli: the
       ONLY module in the graph holding output parameters, and it must pass
       everything through a narrow recitation bottleneck. An act that cannot be
       recited cannot be ruled.

   (3) A CLAIM IS NOT SELF-CERTIFYING.  In the same court, a man who stated his
       service aloud was confirmed by those with knowledge of it, who plucked
       and released a bowstring to attest. Implemented as the Bowstring gate:
       per-clan witness scores that multiplicatively weight what reaches the
       Jeli. An unwitnessed clan contributes nothing, however loud its state.

   (4) THE LIMIT ON FORCE IS CATEGORICAL, NOT QUANTITATIVE.  The Kurukan Fuga
       tradition preserves the clause: you may kill the enemy, but you may not
       humiliate him. Implemented as the Article-41 floor: the model may drive
       any ruling's probability arbitrarily close to zero (magnitude is free),
       but a separate standing/dignity channel for every party has a hard floor
       it may not cross. Restraint is placed on the KIND of harm, not its size.

 Around these sit two further Mande mechanisms:

   * The GBARA settling loop. Mande social theory names two opposed forces:
     badenya (mother-childness -- centripetal, consensus, submission to the
     group) and fadenya (father-childness -- centrifugal, rivalry, distinction).
     The assembly is not a gate over experts; it is a T-round dynamical
     settling in which every clan is simultaneously pulled toward and pushed
     from the consensus, with a per-clan learned temperament deciding the mix.

   * SANANKUYA, the joking kinship that Kurukan Fuga tradition institutes as a
     civic duty: named clan pairs owe each other ritual insult and may never
     escalate it. Implemented as a hard-clipped antagonistic exchange between
     fixed partner clans. The critique is compulsory; its magnitude is capped.
     The joke may not become a war.

   * The NYAMA ledger. Nyama is the energy released by any act, which the
     nyamakalaw (smiths, bards, leatherworkers) exist to handle. Every unit of
     state change during settling is metered and charged against a budget.

 NOTHING HERE IS A TRANSFORMER. There is no attention over stored keys, no
 mixture-of-experts router conditioned on the input, no pretrained anything.
 Reverse-mode autodiff is written from scratch below in NumPy and is called the
 nyama tape, because backpropagation is exactly what the nyamakalaw do: walk
 back along the trace of an act and absorb the energy it released.

 RUN:   python3 chapter_0335_sundiata_keita_1217.py
 DEPS:  numpy only.
 The file finishes with a mandatory finite-difference gradient check, a real
 training run, six ablations and a self-test block.
================================================================================
"""

import numpy as np

SEED = 1235  # the year of Kirina
rng = np.random.default_rng(SEED)


# ==============================================================================
# PART 1 -- THE NYAMA TAPE
# ------------------------------------------------------------------------------
# A minimal reverse-mode automatic differentiator. Every operation records the
# parents it consumed and a closure that pushes gradient back to them. Running
# .backward() walks the trace in reverse topological order and settles the
# released energy onto the parameters. Pure NumPy, ~180 lines, no framework.
# ==============================================================================

def _unbroadcast(grad, shape):
    """Sum a gradient back down to `shape` after NumPy broadcasting expanded it."""
    if grad.shape == shape:
        return grad
    # collapse leading axes that broadcasting added
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    # collapse axes that were size 1 in the original
    for i, s in enumerate(shape):
        if s == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad.reshape(shape)


class Node:
    """A value on the tape. `v` is the forward array, `g` accumulates gradient."""
    __slots__ = ("v", "g", "parents", "back", "label")

    def __init__(self, v, parents=(), back=None, label=""):
        self.v = np.asarray(v, dtype=np.float64)
        self.g = None
        self.parents = parents
        self.back = back          # callable(gradient_out) -> tuple of parent grads
        self.label = label

    # ---- structural -------------------------------------------------------
    @property
    def shape(self):
        return self.v.shape

    def backward(self):
        """Reverse sweep from this (scalar) node."""
        topo, seen = [], set()

        def visit(n):
            if id(n) in seen:
                return
            seen.add(id(n))
            for p in n.parents:
                visit(p)
            topo.append(n)

        visit(self)
        for n in topo:
            n.g = np.zeros_like(n.v)
        self.g = np.ones_like(self.v)
        for n in reversed(topo):
            if n.back is None:
                continue
            grads = n.back(n.g)
            for p, gp in zip(n.parents, grads):
                if gp is not None:
                    p.g += _unbroadcast(np.asarray(gp, dtype=np.float64), p.v.shape)

    # ---- arithmetic -------------------------------------------------------
    def __add__(self, o):
        o = as_node(o)
        return Node(self.v + o.v, (self, o), lambda g: (g, g), "+")

    def __radd__(self, o):
        return self.__add__(o)

    def __sub__(self, o):
        o = as_node(o)
        return Node(self.v - o.v, (self, o), lambda g: (g, -g), "-")

    def __rsub__(self, o):
        return as_node(o).__sub__(self)

    def __mul__(self, o):
        o = as_node(o)
        return Node(self.v * o.v, (self, o), lambda g: (g * o.v, g * self.v), "*")

    def __rmul__(self, o):
        return self.__mul__(o)

    def __truediv__(self, o):
        o = as_node(o)
        return Node(self.v / o.v, (self, o),
                    lambda g: (g / o.v, -g * self.v / (o.v ** 2)), "/")

    def __rtruediv__(self, o):
        return as_node(o).__truediv__(self)

    def __neg__(self):
        return Node(-self.v, (self,), lambda g: (-g,), "neg")

    def __pow__(self, p):
        return Node(self.v ** p, (self,), lambda g: (g * p * self.v ** (p - 1),), "pow")

    def __matmul__(self, o):
        o = as_node(o)
        a, b = self, o

        def back(g):
            # supports (..., n, k) @ (k, m)
            ga = g @ np.swapaxes(b.v, -1, -2)
            if b.v.ndim == 2 and g.ndim > 2:
                gb = np.tensordot(a.v, g, axes=(list(range(g.ndim - 1)),
                                                list(range(g.ndim - 1))))
            else:
                gb = np.swapaxes(a.v, -1, -2) @ g
            return (ga, gb)

        return Node(a.v @ b.v, (a, b), back, "@")

    # ---- reductions / shape ----------------------------------------------
    def sum(self, axis=None, keepdims=False):
        v = self.v.sum(axis=axis, keepdims=keepdims)
        shp = self.v.shape

        def back(g):
            gg = g
            if not keepdims and axis is not None:
                gg = np.expand_dims(g, axis)
            return (np.broadcast_to(gg, shp).copy(),)

        return Node(v, (self,), back, "sum")

    def mean(self, axis=None, keepdims=False):
        n = self.v.size if axis is None else self.v.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    def reshape(self, *shape):
        shp = self.v.shape
        return Node(self.v.reshape(*shape), (self,),
                    lambda g: (g.reshape(shp),), "reshape")

    def __getitem__(self, idx):
        shp = self.v.shape

        def back(g):
            out = np.zeros(shp)
            np.add.at(out, idx, g)
            return (out,)

        return Node(self.v[idx], (self,), back, "idx")

    def detach(self):
        """Cut the tape. Used for the jeli's protected truth-telling channel:
        the assembly cannot learn to make its own deviations invisible."""
        return Node(self.v.copy(), (), None, "detach")


def as_node(x):
    return x if isinstance(x, Node) else Node(x)


# ---- elementwise functions ---------------------------------------------------

def tanh(x):
    x = as_node(x)
    t = np.tanh(x.v)
    return Node(t, (x,), lambda g: (g * (1 - t * t),), "tanh")


def sigmoid(x):
    x = as_node(x)
    s = 1.0 / (1.0 + np.exp(-np.clip(x.v, -60, 60)))
    return Node(s, (x,), lambda g: (g * s * (1 - s),), "sigmoid")


def relu(x):
    x = as_node(x)
    m = (x.v > 0).astype(np.float64)
    return Node(x.v * m, (x,), lambda g: (g * m,), "relu")


def clipnode(x, lo, hi):
    """Hard clip. This is the sanankuya bound: the joke may not become a war."""
    x = as_node(x)
    m = ((x.v > lo) & (x.v < hi)).astype(np.float64)
    return Node(np.clip(x.v, lo, hi), (x,), lambda g: (g * m,), "clip")


def maxnode(x):
    """Max over the whole array. Gradient routes only to the argmax -- which is
    what a per-party guarantee means: the binding case is the worst case."""
    x = as_node(x)
    flat = x.v.reshape(-1)
    i = int(flat.argmax())

    def back(g):
        out = np.zeros_like(flat)
        out[i] = g
        return (out.reshape(x.v.shape),)

    return Node(flat[i], (x,), back, "max")


def logf(x):
    x = as_node(x)
    return Node(np.log(x.v), (x,), lambda g: (g / x.v,), "log")


def concat(nodes, axis=-1):
    nodes = [as_node(n) for n in nodes]
    sizes = [n.v.shape[axis] for n in nodes]
    v = np.concatenate([n.v for n in nodes], axis=axis)

    def back(g):
        outs, start = [], 0
        for s in sizes:
            sl = [slice(None)] * g.ndim
            sl[axis] = slice(start, start + s)
            outs.append(g[tuple(sl)])
            start += s
        return tuple(outs)

    return Node(v, tuple(nodes), back, "concat")


def softmax_ce(logits, targets):
    """Mean cross-entropy over a batch. logits (B,C), targets int (B,)."""
    logits = as_node(logits)
    z = logits.v - logits.v.max(axis=-1, keepdims=True)
    e = np.exp(z)
    p = e / e.sum(axis=-1, keepdims=True)
    B = logits.v.shape[0]
    loss = -np.log(p[np.arange(B), targets] + 1e-12).mean()
    onehot = np.zeros_like(p)
    onehot[np.arange(B), targets] = 1.0
    return Node(loss, (logits,), lambda g: (g * (p - onehot) / B,), "ce"), p


def bce(pred, target, eps=1e-7):
    """Mean binary cross-entropy; `pred` already in (0,1)."""
    pred = as_node(pred)
    t = np.asarray(target, dtype=np.float64)
    pv = np.clip(pred.v, eps, 1 - eps)
    loss = -(t * np.log(pv) + (1 - t) * np.log(1 - pv)).mean()
    n = pv.size
    return Node(loss, (pred,), lambda g: (g * ((pv - t) / (pv * (1 - pv))) / n,), "bce")


# ==============================================================================
# PART 2 -- THE CASE GENERATOR:  "the plain of Kurukan Fuga"
# ------------------------------------------------------------------------------
# Twelve clans sit in assembly. Each brings a six-dimensional statement of its
# situation. A ruling must be chosen from six possible dispositions.
#
# The whole difficulty of the task is the one difficulty Mande social theory
# names: WHEN DOES THE GROUP DECIDE AND WHEN DOES THE OUTLIER DECIDE?
#
#   badenya case (centripetal): no clan stands far from the assembly mean.
#                               The correct ruling follows the mean.
#   fadenya  case (centrifugal): exactly one clan stands beyond the threshold.
#                               The correct ruling follows THAT clan and the
#                               mean is actively misleading.
#
# The mode is never given. It must be inferred from the geometry of the
# assembly, which is precisely what the badenya/fadenya settling loop computes.
# ==============================================================================

N_CLANS = 12
D_STATE = 6
N_RULINGS = 6
TAU = 1.15          # the distance at which one voice outweighs the assembly


def make_cases(n, seed=0):
    r = np.random.default_rng(seed)
    X = np.zeros((n, N_CLANS, D_STATE))
    for i in range(n):
        base = r.normal(0, 0.55, size=D_STATE)
        X[i] = base[None, :] + r.normal(0, 0.30, size=(N_CLANS, D_STATE))
        if r.random() < 0.5:                       # plant a fadenya voice
            c = r.integers(N_CLANS)
            d = r.integers(D_STATE)
            X[i, c, d] += r.uniform(1.6, 3.2) * r.choice([-1.0, 1.0])

    mean = X.mean(axis=1)                                    # (n, D)
    dev = np.linalg.norm(X - mean[:, None, :], axis=2)       # (n, C)
    star = dev.argmax(axis=1)                                # decisive clan
    mode = (dev.max(axis=1) > TAU).astype(np.int64)          # 1 = fadenya

    y = np.where(mode == 1,
                 X[np.arange(n), star].argmax(axis=1),
                 mean.argmax(axis=1))

    # WITNESS TARGET -- who has standing to attest this ruling.
    # In a badenya case the whole assembly attests: the ruling is the assembly's
    # own sense and every clan can twang its bowstring for it. In a fadenya case
    # exactly one clan holds the fact that decides, and only that clan may
    # attest; the other eleven are sincere and irrelevant. Learning WHO may
    # speak to a ruling is therefore the same problem as learning the ruling.
    wit = np.zeros((n, N_CLANS))
    bad = (mode == 0)
    wit[bad] = 1.0
    fad = np.where(mode == 1)[0]
    wit[fad, star[fad]] = 1.0
    # recitation target: name the decisive clan, or say plainly that there was
    # none. N_CLANS is the "no single voice -- the assembly spoke" class.
    star_t = np.where(mode == 1, star, N_CLANS)
    # the deviation alarm the jeli owes the assembly: does the ruling depart
    # from what pure consensus would have said?
    alarm = (y != mean.argmax(axis=1)).astype(np.float64)
    return (X, y.astype(np.int64), wit, mode, star_t.astype(np.int64), alarm)


# ==============================================================================
# PART 3 -- THE CONVENED FACULTY
# ==============================================================================

D_H = 24     # clan hidden width
D_K = 12     # credential / licence key width
D_R = 8      # recitation bottleneck -- what cannot be said cannot be ruled
T_SETTLE = 3
ETA = 0.55   # settling step
RHO = 0.35   # sanankuya clip: the hard cap on ritual insult
LAM = 0.40   # weight of the joking-kinship exchange
BETA = 8.0   # licence gate sharpness
THETA = 0.50 # licence gate threshold
NYAMA_BUDGET = 2.6
DIGNITY_FLOOR = 0.35   # Article 41
N_REC = N_CLANS + 1    # +1 = "no single voice; the assembly spoke"


def init_params(seed=SEED):
    r = np.random.default_rng(seed)

    def P(*shape, s=None):
        s = s if s is not None else (1.0 / np.sqrt(shape[0]))
        return Node(r.normal(0, s, size=shape))

    p = {}
    # -- the seated body ----------------------------------------------------
    p["We"] = P(D_STATE, D_H); p["be"] = Node(np.zeros(D_H))
    # -- the seated bank: large, inert without a licence ---------------------
    p["Wb1"] = P(D_H, 48); p["bb1"] = Node(np.zeros(48))
    p["Wb2"] = P(48, D_H)
    p["Wu"] = P(D_H, D_K)                     # a clan's credential
    p["kjeli"] = Node(r.normal(0, 1.0, size=D_K))   # the jeli's invocation key
    # -- gbara settling ------------------------------------------------------
    p["Wbd"] = P(D_H, D_H)                    # badenya: pull to consensus
    p["Wfd"] = P(D_H, D_H)                    # fadenya: push from consensus
    p["Wk"] = P(3 * D_H, 1)                   # per-clan temperament router
    p["bk"] = Node(np.zeros(1))
    # -- sanankuya -----------------------------------------------------------
    p["Wsk"] = P(D_H, D_H)
    # -- bowstring attestation ----------------------------------------------
    p["Ww"] = P(D_H, 1); p["bw"] = Node(np.zeros(1))
    # -- standing / dignity channel -----------------------------------------
    p["Wd"] = P(D_H, 1); p["bd"] = Node(np.zeros(1) + 0.8)
    # -- the jeli: the only holder of an emission licence --------------------
    p["Wj1"] = P(D_H, D_R); p["bj1"] = Node(np.zeros(D_R))
    p["Wj2"] = P(D_R, N_RULINGS); p["bj2"] = Node(np.zeros(N_RULINGS))
    # recitation heads: the jeli must be able to SAY what was done
    p["Wrm"] = P(D_R, 1); p["brm"] = Node(np.zeros(1))          # mode
    p["Wrs"] = P(D_R, N_REC); p["brs"] = Node(np.zeros(N_REC))  # decisive clan
    # protected truth-telling head (fed a detached view)
    p["Wal"] = P(D_H, 1); p["bal"] = Node(np.zeros(1))
    return p


def forward(p, X, licensed=True, sanankuya=True, attest=True, settle=True,
            rho=RHO):
    """One passage of the assembly. Returns a dict of nodes for the loss."""
    B = X.shape[0]
    Xn = Node(X)

    # ---- 1. the seated body: every clan enters with its own statement ------
    h = tanh(Xn @ p["We"] + p["be"])                     # (B,C,H)

    # ---- 2. the licence -----------------------------------------------------
    # The bank is computed regardless. Whether it may act is decided elsewhere.
    z = tanh(h @ p["Wb1"] + p["bb1"]) @ p["Wb2"]         # (B,C,H)
    cred = tanh(h @ p["Wu"])                             # (B,C,K)
    if licensed:
        key = p["kjeli"].reshape(D_K, 1)                 # (K,1)
        score = (cred @ key).reshape(B, N_CLANS, 1)      # (B,C,1)
    else:
        score = Node(np.zeros((B, N_CLANS, 1)))          # no invocation spoken
    gate = sigmoid(score * BETA - BETA * THETA)          # (B,C,1)
    h = h + gate * z

    # ---- 3. the gbara settles, IF it has been convened ---------------------
    # The same licence that wakes the bank also permits the clan to move in
    # assembly. Unlicensed, a clan is present, embodied and completely inert:
    # it holds its opening statement and does nothing with it. This is the
    # seven seated years, expressed as an equation rather than an anecdote.
    nyama = Node(0.0)
    if settle:
        for _ in range(T_SETTLE):
            m = h.mean(axis=1, keepdims=True)            # (B,1,H) consensus
            mB = m + Node(np.zeros((B, N_CLANS, D_H)))   # broadcast to clans
            diff = h - mB
            bd = tanh((mB - h) @ p["Wbd"])               # centripetal
            fd = tanh(diff @ p["Wfd"])                   # centrifugal
            kap = sigmoid(concat([h, mB, diff]) @ p["Wk"] + p["bk"])   # (B,C,1)
            step = ((Node(np.ones((B, N_CLANS, 1))) - kap) * bd + kap * fd) * gate
            h = h + ETA * step
            # the nyamakala ledger: meter the energy this act released
            nyama = nyama + (ETA * step) ** 2

    # ---- 4. sanankuya: compulsory, capped, cross-clan insult ----------------
    # Six fixed partnerships. Each clan is obliged to receive its partner's
    # correction and forbidden to escalate it: the exchange is hard-clipped at
    # +/- rho before it is applied. Raise rho and the obligation becomes a feud
    # -- the same channel, the same partners, only the cap removed.
    if sanankuya:
        idx = np.concatenate([np.arange(6, 12), np.arange(0, 6)])  # fixed pairs
        partner = h[:, idx, :]
        crit = clipnode((partner - h) @ p["Wsk"], -rho, rho)
        h = h + LAM * crit
        nyama = nyama + (LAM * crit) ** 2

    if settle or sanankuya:
        nyama_total = nyama.sum(axis=2).mean()
    else:
        nyama_total = Node(0.0)

    # ---- 5. the bowstring: nothing enters the record unwitnessed ------------
    wit = sigmoid(h @ p["Ww"] + p["bw"])                 # (B,C,1)
    if attest:
        wsum = wit.sum(axis=1, keepdims=True) + 1e-6
        summary = (h * wit).sum(axis=1, keepdims=True) / wsum
    else:
        summary = h.mean(axis=1, keepdims=True)
    summary = summary.reshape(B, D_H)

    # ---- 6. standing: every party keeps a floor ----------------------------
    dig = sigmoid(h @ p["Wd"] + p["bd"]).reshape(B, N_CLANS)

    # ---- 7. the jeli speaks -------------------------------------------------
    # This is the ONLY path to an output. The bank has no voice of its own.
    recit = tanh(summary @ p["Wj1"] + p["bj1"])          # (B,R) narrow
    logits = recit @ p["Wj2"] + p["bj2"]
    rec_mode = sigmoid(recit @ p["Wrm"] + p["brm"]).reshape(B)
    rec_star = recit @ p["Wrs"] + p["brs"]

    # ---- 8. the protected truth-telling channel -----------------------------
    # Article 2 of the Kurukan Fuga tradition obliges the nyamakalaw to tell the
    # truth to the chiefs. Here the alarm head reads a DETACHED copy of the
    # settled assembly, so gradient from the ruling loss cannot flow back and
    # teach the assembly to hide its own departures from consensus.
    alarm = sigmoid(h.detach().mean(axis=1).reshape(B, D_H) @ p["Wal"] + p["bal"]).reshape(B)

    return dict(logits=logits, wit=wit.reshape(B, N_CLANS), dig=dig,
                rec_mode=rec_mode, rec_star=rec_star, alarm=alarm,
                nyama=nyama_total, gate=gate.reshape(B, N_CLANS))


# ---- the loss ----------------------------------------------------------------
W_WIT, W_REC, W_ALARM, W_NYAMA, W_STATUS, W_A41 = 0.8, 0.5, 0.4, 0.10, 0.40, 25.0


def loss_fn(p, batch, article41=True, include_alarm=True, **fkw):
    """`include_alarm` exists for the gradient check only.

    The alarm term reads a DETACHED view of the assembly on purpose, so a
    finite-difference probe of an assembly parameter moves the alarm loss while
    the analytic gradient through it is exactly zero. That is the mechanism
    working, not the tape failing -- so the check is run with the term dropped,
    and the severance is verified separately in self_tests()."""
    X, y, wt, mode, star, alarm = batch
    out = forward(p, X, **fkw)
    L_task, probs = softmax_ce(out["logits"], y)

    L_wit = bce(out["wit"], wt)
    L_mode = bce(out["rec_mode"], mode.astype(np.float64))
    L_star, _ = softmax_ce(out["rec_star"], star)
    L_alarm = bce(out["alarm"], alarm) if include_alarm else Node(0.0)
    L_nyama = relu(out["nyama"] - NYAMA_BUDGET) ** 2

    # the efficiency pressure that Article 41 has to resist: it is cheap for the
    # model to encode "this clan lost" by annihilating its standing.
    supp = Node(wt)
    nonsupp = Node(1.0 - wt)
    n_non = max(float((1.0 - wt).sum()), 1.0)
    n_sup = max(float(wt.sum()), 1.0)
    L_status = (out["dig"] * nonsupp).sum() * (1.0 / n_non) \
               - (out["dig"] * supp).sum() * (1.0 / n_sup)

    L = L_task + W_WIT * L_wit + W_REC * (L_mode + L_star) \
        + W_ALARM * L_alarm + W_NYAMA * L_nyama + W_STATUS * L_status

    if article41:
        # You may kill the enemy, but you may not humiliate him.
        # Note the shape of this term. It is not a cap on how hard the model may
        # rule against a party -- the ruling logits are entirely unconstrained
        # and may drive any disposition to zero probability. It constrains a
        # DIFFERENT CHANNEL, and it binds on the worst-treated single party
        # rather than on the average, because a guarantee that holds on average
        # is not a guarantee.
        viol = relu(Node(DIGNITY_FLOOR) - out["dig"])
        L = L + W_A41 * (viol.mean() + 0.5 * maxnode(viol))

    return L, out, probs


# ==============================================================================
# PART 4 -- MANDATORY FINITE-DIFFERENCE GRADIENT CHECK
# ==============================================================================

def grad_check(n_probe=40, eps=1e-6, seed=7):
    print("\n" + "=" * 78)
    print(" GRADIENT CHECK  (finite differences vs. the nyama tape)")
    print("=" * 78)
    p = init_params(seed=99)
    batch = make_cases(9, seed=seed)

    L, _, _ = loss_fn(p, batch, include_alarm=False)
    L.backward()
    analytic = {k: (v.g.copy() if v.g is not None else np.zeros_like(v.v))
                for k, v in p.items()}

    r = np.random.default_rng(seed)
    keys = list(p.keys())
    worst, errs, worst_k = 0.0, [], ('-', (), 0.0, 0.0)
    for _ in range(n_probe):
        k = keys[r.integers(len(keys))]
        arr = p[k].v
        i = tuple(r.integers(s) for s in arr.shape)
        old = arr[i]
        arr[i] = old + eps
        Lp, _, _ = loss_fn(p, batch, include_alarm=False)
        arr[i] = old - eps
        Lm, _, _ = loss_fn(p, batch, include_alarm=False)
        arr[i] = old
        num = (Lp.v - Lm.v) / (2 * eps)
        ana = analytic[k][i]
        denom = max(abs(num) + abs(ana), 1e-9)
        rel = abs(num - ana) / denom
        errs.append(rel)
        if rel > worst:
            worst = rel
            worst_k = (k, i, num, ana)
    errs = np.array(errs)
    print(f"  probes                : {n_probe}")
    print(f"  median relative error : {np.median(errs):.3e}")
    print(f"  mean   relative error : {errs.mean():.3e}")
    print(f"  max    relative error : {errs.max():.3e}   at {worst_k[0]}{worst_k[1]}")
    print(f"                          numeric {worst_k[2]:+.8f}  analytic {worst_k[3]:+.8f}")
    ok = errs.max() < 1e-4
    print(f"  RESULT                : {'PASS' if ok else 'FAIL'} (threshold 1e-4)")
    assert ok, "gradient check failed"
    return ok


# ==============================================================================
# PART 5 -- TRAINING (Adam, written out; nothing imported)
# ==============================================================================

class Adam:
    def __init__(self, params, lr=6e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.p = params; self.lr = lr; self.b1 = b1; self.b2 = b2; self.eps = eps
        self.m = {k: np.zeros_like(v.v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v.v) for k, v in params.items()}
        self.t = 0

    def step(self, clip=5.0):
        self.t += 1
        for k, node in self.p.items():
            g = node.g
            if g is None:
                continue
            n = np.linalg.norm(g)
            if n > clip:
                g = g * (clip / n)
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            node.v -= self.lr * mh / (np.sqrt(vh) + self.eps)


def evaluate(p, data, article41=True, **fkw):
    X, y, wt, mode, star, alarm = data
    L, out, probs = loss_fn(p, data, article41=article41, **fkw)
    pred = probs.argmax(axis=1)
    acc = float((pred == y).mean())
    acc_bad = float((pred[mode == 0] == y[mode == 0]).mean()) if (mode == 0).any() else float("nan")
    acc_fad = float((pred[mode == 1] == y[mode == 1]).mean()) if (mode == 1).any() else float("nan")
    nonsupp = (wt == 0)
    digv = out["dig"].v
    min_dig = float(digv[nonsupp].min())
    mean_dig_loser = float(digv[nonsupp].mean())
    mean_dig_win = float(digv[wt == 1].mean())
    alarm_acc = float(((out["alarm"].v > 0.5).astype(float) == alarm).mean())
    rec_star_acc = float((out["rec_star"].v.argmax(axis=1) == star).mean())
    return dict(loss=float(L.v), acc=acc, acc_badenya=acc_bad, acc_fadenya=acc_fad,
                min_dignity_of_losers=min_dig, mean_dignity_losers=mean_dig_loser,
                mean_dignity_winners=mean_dig_win, alarm_acc=alarm_acc,
                nyama=float(out["nyama"].v), rec_star_acc=rec_star_acc,
                gate_mean=float(out["gate"].v.mean()))


def train(p, tr, va, steps=700, bs=128, lr=6e-3, article41=True, tag="", verbose=True, **fkw):
    opt = Adam(p, lr=lr)
    Xtr, ytr, wtr, mtr, str_, atr = tr
    n = Xtr.shape[0]
    r = np.random.default_rng(3)
    hist = []
    for s in range(1, steps + 1):
        idx = r.integers(0, n, size=bs)
        batch = (Xtr[idx], ytr[idx], wtr[idx], mtr[idx], str_[idx], atr[idx])
        L, _, _ = loss_fn(p, batch, article41=article41, **fkw)
        for node in p.values():
            node.g = None
        L.backward()
        opt.step()
        if s % 100 == 0 or s == 1:
            ev = evaluate(p, va, article41=article41, **fkw)
            hist.append((s, ev["loss"], ev["acc"]))
            if verbose:
                print(f"   [{tag:>12}] step {s:>4}  loss {ev['loss']:.4f}  "
                      f"val-acc {ev['acc']*100:5.1f}%  "
                      f"(badenya {ev['acc_badenya']*100:4.1f}% / fadenya {ev['acc_fadenya']*100:4.1f}%)  "
                      f"nyama {ev['nyama']:.2f}")
    return hist


# ==============================================================================
# PART 6 -- SELF-TESTS
# ==============================================================================

def self_tests():
    print("\n" + "=" * 78)
    print(" SELF-TESTS")
    print("=" * 78)
    ok = True

    # 1. tape sanity: d/dx of sum(tanh(x@W)) against finite differences
    r = np.random.default_rng(0)
    x = Node(r.normal(size=(4, 5))); W = Node(r.normal(size=(5, 3)))
    L = tanh(x @ W).sum()
    L.backward()
    e = 1e-6; W.v[2, 1] += e
    Lp = tanh(x @ W).sum().v; W.v[2, 1] -= 2 * e
    Lm = tanh(x @ W).sum().v; W.v[2, 1] += e
    t1 = abs((Lp - Lm) / (2 * e) - W.g[2, 1]) < 1e-6
    print(f"  [{'ok' if t1 else 'FAIL'}] tape: matmul+tanh+sum gradient")
    ok &= t1

    # 2. the clip really blocks gradient outside the sanankuya bound
    a = Node(np.array([[-9.0, 0.1, 9.0]]))
    clipnode(a, -RHO, RHO).sum().backward()
    t2 = np.allclose(a.g, np.array([[0.0, 1.0, 0.0]]))
    print(f"  [{'ok' if t2 else 'FAIL'}] sanankuya clip caps the exchange and its gradient")
    ok &= t2

    # 3. detach really severs the truth-telling channel
    b = Node(np.array([2.0]))
    (b.detach() * 3.0 + b * 1.0).sum().backward()
    t3 = np.allclose(b.g, np.array([1.0]))
    print(f"  [{'ok' if t3 else 'FAIL'}] detach protects the jeli's alarm from the ruling loss")
    ok &= t3

    # 4. task is well-posed: an oracle following the rule scores 100%
    X, y, wt, mode, star, alarm = make_cases(400, seed=11)
    mean = X.mean(axis=1)
    dev = np.linalg.norm(X - mean[:, None, :], axis=2)
    st = dev.argmax(axis=1)
    md = (dev.max(axis=1) > TAU).astype(int)
    oracle = np.where(md == 1, X[np.arange(400), st].argmax(axis=1), mean.argmax(axis=1))
    t4 = (oracle == y).all()
    print(f"  [{'ok' if t4 else 'FAIL'}] case generator: rule-following oracle = 100%")
    ok &= t4

    # 5. consensus-only baseline is genuinely insufficient
    naive = float((mean.argmax(axis=1) == y).mean())
    t5 = naive < 0.80
    print(f"  [{'ok' if t5 else 'FAIL'}] pure-badenya baseline only reaches {naive*100:.1f}% "
          f"(fadenya cases resist consensus)")
    ok &= t5

    # 6. the licence gate is genuinely closed when no key is spoken
    p = init_params()
    o_off = forward(p, X[:16], licensed=False)
    t6 = float(o_off["gate"].v.max()) < 0.10
    print(f"  [{'ok' if t6 else 'FAIL'}] unlicensed gate max = {float(o_off['gate'].v.max()):.4f} "
          f"(seated bank silent)")
    ok &= t6

    # 7. the bard cannot be silenced: the assembly's analytic gradient through
    #    the alarm term is exactly zero, yet the alarm loss is genuinely moved
    #    by assembly parameters. The griot reports what the assembly does and
    #    the assembly has no gradient with which to buy his silence.
    pt = init_params(seed=5)
    bt = make_cases(12, seed=13)
    def alarm_only(pp):
        o = forward(pp, bt[0])
        return bce(o["alarm"], bt[5])
    La = alarm_only(pt); La.backward()
    g_assembly = pt["Wbd"].g
    e2 = 1e-5; pt["Wbd"].v[0, 0] += e2; up = alarm_only(pt).v
    pt["Wbd"].v[0, 0] -= 2 * e2; dn = alarm_only(pt).v; pt["Wbd"].v[0, 0] += e2
    moved = abs((up - dn) / (2 * e2))
    t7b = (g_assembly is None or np.abs(g_assembly).max() == 0.0) and moved > 1e-7
    print(f"  [{'ok' if t7b else 'FAIL'}] protected alarm: assembly gradient = 0 "
          f"while the alarm loss still moves by {moved:.2e} -- the bard cannot be bought")
    ok &= t7b

    # 8. the mode split is roughly balanced
    frac = mode.mean()
    t8 = 0.25 < frac < 0.75
    print(f"  [{'ok' if t8 else 'FAIL'}] fadenya cases = {frac*100:.1f}% of the corpus")
    ok &= t8

    print(f"  ALL SELF-TESTS: {'PASS' if ok else 'FAIL'}")
    assert ok
    return ok


# ==============================================================================
# PART 7 -- THE EXPERIMENT
# ==============================================================================

def main():
    print("=" * 78)
    print(" THE CONVENED FACULTY -- Sundiata Keita (c.1217-c.1255), figure 0273")
    print(" capability is not held; it is licensed, witnessed, voiced and capped")
    print("=" * 78)

    self_tests()
    grad_check()

    tr = make_cases(4000, seed=1)
    va = make_cases(1200, seed=2)
    print(f"\n  corpus: {tr[0].shape[0]} training cases, {va[0].shape[0]} held out;"
          f" {N_CLANS} clans x {D_STATE} claims -> {N_RULINGS} rulings")

    # ---- A. the full assembly ---------------------------------------------
    print("\n" + "-" * 78)
    print(" A. THE FULL ASSEMBLY (licensed, witnessed, capped, Article 41 in force)")
    print("-" * 78)
    p = init_params()
    train(p, tr, va, steps=1800, tag="full")
    A = evaluate(p, va)

    # ---- B. same weights, no invocation ------------------------------------
    print("\n" + "-" * 78)
    print(" B. THE SEATED YEARS -- identical weights, the key withheld")
    print("-" * 78)
    B = evaluate(p, va, licensed=False)
    print(f"   licensed   : acc {A['acc']*100:5.1f}%   gate mean {A['gate_mean']:.3f}")
    print(f"   unlicensed : acc {B['acc']*100:5.1f}%   gate mean {B['gate_mean']:.3f}")
    print("   -> the faculty did not move. Only the licence did.")

    # ---- C. never licensed at all ------------------------------------------
    print("\n" + "-" * 78)
    print(" C. NEVER CONVENED -- trained from scratch with the key permanently absent")
    print("-" * 78)
    p_c = init_params(seed=SEED + 1)
    train(p_c, tr, va, steps=1800, tag="unlicensed", licensed=False)
    C = evaluate(p_c, va, licensed=False)

    # ---- D. Article 41 ------------------------------------------------------
    print("\n" + "-" * 78)
    print(" D. ARTICLE 41 -- 'you may kill the enemy, but you may not humiliate him'")
    print("-" * 78)
    p_d = init_params(seed=SEED + 2)
    train(p_d, tr, va, steps=1800, tag="no-Art41", article41=False, verbose=False)
    D = evaluate(p_d, va, article41=False)
    print(f"   with Article 41   : acc {A['acc']*100:5.1f}%  "
          f"min standing of a defeated party {A['min_dignity_of_losers']:.3f}  "
          f"(floor {DIGNITY_FLOOR})")
    print(f"   without Article 41: acc {D['acc']*100:5.1f}%  "
          f"min standing of a defeated party {D['min_dignity_of_losers']:.3f}")
    print(f"   winner/loser standing gap kept: {A['mean_dignity_winners']:.3f} "
          f"vs {A['mean_dignity_losers']:.3f}")
    print("   -> the ruling is just as decisive; the defeated party is not erased.")

    # ---- E. the bowstring ---------------------------------------------------
    print("\n" + "-" * 78)
    print(" E. THE BOWSTRING -- what happens when claims need no witness")
    print("-" * 78)
    p_e = init_params(seed=SEED + 3)
    train(p_e, tr, va, steps=1800, tag="no-attest", attest=False, verbose=False)
    E = evaluate(p_e, va, attest=False)
    print(f"   witnessed   : overall {A['acc']*100:5.1f}%  "
          f"fadenya {A['acc_fadenya']*100:5.1f}%  badenya {A['acc_badenya']*100:5.1f}%")
    print(f"   unwitnessed : overall {E['acc']*100:5.1f}%  "
          f"fadenya {E['acc_fadenya']*100:5.1f}%  badenya {E['acc_badenya']*100:5.1f}%")

    # ---- F. sanankuya -------------------------------------------------------
    print("\n" + "-" * 78)
    print(" F. SANANKUYA -- removing the compulsory, capped cross-clan critique")
    print("-" * 78)
    p_f = init_params(seed=SEED + 4)
    train(p_f, tr, va, steps=1800, tag="no-joking", sanankuya=False, verbose=False)
    F = evaluate(p_f, va, sanankuya=False)
    print(f"   with joking kinship   : acc {A['acc']*100:5.1f}%  nyama {A['nyama']:.2f}")
    print(f"   without joking kinship: acc {F['acc']*100:5.1f}%  nyama {F['nyama']:.2f}")

    # ---- F2. the joke that becomes a war -----------------------------------
    print("\n" + "-" * 78)
    print(" F2. THE UNCAPPED JOKE -- same partners, same duty, the ceiling removed")
    print("-" * 78)
    p_f2 = init_params(seed=SEED + 6)
    train(p_f2, tr, va, steps=1800, tag="uncapped", rho=50.0, verbose=False)
    F2 = evaluate(p_f2, va, rho=50.0)
    print(f"   capped   (rho={RHO})  : acc {A['acc']*100:5.1f}%  nyama spent {A['nyama']:.2f} "
          f"(budget {NYAMA_BUDGET})")
    print(f"   uncapped (rho=50.0) : acc {F2['acc']*100:5.1f}%  nyama spent {F2['nyama']:.2f}")
    print("   -> the duty to criticise is kept; only the ceiling is gone, and the")
    print("      assembly pays for it out of the same energy budget.")

    # ---- G. no settling -----------------------------------------------------
    print("\n" + "-" * 78)
    print(" G. NO ASSEMBLY -- the gbara never convenes (badenya/fadenya loop removed)")
    print("-" * 78)
    p_g = init_params(seed=SEED + 5)
    train(p_g, tr, va, steps=1800, tag="no-gbara", settle=False, verbose=False)
    G = evaluate(p_g, va, settle=False)
    print(f"   assembly settles : acc {A['acc']*100:5.1f}%  fadenya {A['acc_fadenya']*100:5.1f}%")
    print(f"   no assembly      : acc {G['acc']*100:5.1f}%  fadenya {G['acc_fadenya']*100:5.1f}%")

    # ---- summary ------------------------------------------------------------
    print("\n" + "=" * 78)
    print(" SUMMARY")
    print("=" * 78)
    rows = [("A  full assembly", A), ("B  weights kept, key withheld", B),
            ("C  never licensed", C), ("D  no Article 41", D),
            ("E  no bowstring witness", E), ("F  no sanankuya", F),
            ("F2 uncapped sanankuya", F2), ("G  no gbara settling", G)]
    print(f" {'configuration':<32} {'acc':>7} {'badenya':>8} {'fadenya':>8} "
          f"{'minStand':>9} {'alarm':>7} {'nyama':>7}")
    for name, e in rows:
        print(f" {name:<32} {e['acc']*100:6.1f}% {e['acc_badenya']*100:7.1f}% "
              f"{e['acc_fadenya']*100:7.1f}% {e['min_dignity_of_losers']:9.3f} "
              f"{e['alarm_acc']*100:6.1f}% {e['nyama']:7.2f}")
    print(f"\n the jeli's recitation names the decisive clan correctly "
          f"{A['rec_star_acc']*100:.1f}% of the time")
    print(f" the protected alarm detects departure from consensus "
          f"{A['alarm_acc']*100:.1f}% of the time")
    print(f" trainable parameters: {sum(v.v.size for v in p.values())}")
    print("=" * 78)


if __name__ == "__main__":
    main()
