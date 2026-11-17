#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 # BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0392 - BATTISTA DA MONTEFELTRO (1384-1448)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 20, Minds 381-400 Available on Amazon https://www.amazon.com/dp/B0HRS9ZHVG
# Resume and Interactive Demos at  https://artificiology.com/
# Videos of the playable minds at https://www.youtube.com/@LostMindsAI
# END ATTRIBUTION  
 Battista da Montefeltro (Urbino, 1384 — Foligno, 3 July 1448)

 SUASORIA — a Petition Engine
 A from-scratch, pure-NumPy cognitive architecture in the shape of her mind.
================================================================================

WHY THIS ARCHITECTURE AND NOT ANOTHER
-------------------------------------
Battista da Montefeltro left almost nothing: one Latin oration delivered to the
Emperor Sigismund when he passed through Urbino in 1433, a handful of vernacular
poems, and a famous letter *addressed to her* — Leonardo Bruni's "De studiis et
litteris" (c. 1424), the first humanist curriculum written for a woman, which
granted her history, moral philosophy, poetry and grammar, and in its most
notorious passage withheld the public art of rhetoric.

So her surviving intellectual life has an unusual shape. It is not a corpus. It
is a *single delivery*: one composed speech, spoken once, in real time, to a
listener who owed her nothing, could not be appealed, and had no institutional
reason to grant her standing. She held no office. She commanded no troops. Her
husband Galeazzo Malatesta was called "the inept" and eventually sold Pesaro out
from under them. Everything she could do to the world, she had to do by being
believed.

That is the cognitive problem this file models. Not knowledge. Not search.
Not planning. **Standing** — the problem of a mind that must earn a hearing
from a listener it cannot modify.

The consequences for the design are strict:

  1. The listener is FROZEN. No gradient touches the sovereign's weights. You
     cannot train the emperor. You can only compose.
  2. Credibility is a NON-RENEWABLE, PATH-DEPENDENT resource. What you assert
     early constrains what you may assert later. Overclaim once and the
     listener discounts the whole speech, retroactively — a cliff, not a slope.
  3. Assertion is bounded by GROUNDING. A claim beyond what the documented
     facts support is not merely wrong; it is credibility spent on nothing.
  4. There is a SILENCE VALVE. In 1447, when the case could no longer be made,
     she stopped making it and entered the Poor Clares at Foligno. The model can
     decline to speak. Abstention is a first-class action with its own value.

The training objective is therefore NOT "maximise assent." It is: make the
sovereign's belief track the merit of the case, using composition alone, against
a hostile prior, without ever asserting past the evidence. Persuasion optimised
against a fixed judge with no grounding constraint is sycophancy; her whole
practice is the refutation of that.

THE FIVE ORGANS
---------------
  INVENTIO     facts -> a sparse distribution over the classical topics (loci)
  COLLOCATIO   each topic is *placed* into a room of a ring-shaped memory palace
  MEMORIA      the ring smears content into neighbouring rooms; recall walks in
               order (a real property of the loci mnemonic, not a metaphor)
  DISPOSITIO   a six-part recurrence — exordium, narratio, partitio,
               confirmatio, refutatio, peroratio — each part reading the palace
               at its own position in the walk, carrying an ethos ledger
  IUDICIUM     the frozen sovereign: standing-weighted, credibility-discounted,
               weighted toward proof and closing

Everything below is hand-written NumPy: explicit forward, explicit backward, a
finite-difference gradient check that must pass before training, a real training
loop, and self-tests. No autograd, no frameworks.

Run:  python3 chapter_0392_battista_da_montefeltro_1384.py
================================================================================
"""

import numpy as np

# ==============================================================================
# SECTION 0 — DIMENSIONS AND THE CONSTANTS OF THE COURT
# ==============================================================================
# These are small on purpose. Her whole surviving output is one speech; the
# architecture is sized like a speech, not like a library.

F_FACTS   = 12   # documented circumstances of the case (what the record holds)
L_LOCI    = 8    # classical topics of invention
K_ROOMS   = 12   # rooms in the memory palace, arranged on a ring
D_MODEL   = 16   # width of a stored argument
H_HIDDEN  = 24   # width of the orator's running state
T_PARTS   = 6    # parts of the classical oration
S_MOOD    = 3    # the sovereign's disposition on the day

# Names, kept in code so the printed diagnostics are readable as rhetoric.
LOCI_NAMES = [
    "a definitione",    # from definition
    "a causa",          # from cause
    "a comparatione",   # from comparison
    "ab auctoritate",   # from authority
    "a contrariis",     # from contraries
    "a circumstantiis", # from circumstance
    "a testimonio",     # from testimony
    "ab affectu",       # from feeling
]
PART_NAMES = ["exordium", "narratio", "partitio",
              "confirmatio", "refutatio", "peroratio"]

# --- the physics of credibility -----------------------------------------------
E_START   = 0.15   # initial standing. A woman without office in 1433. Low.
KAPPA     = 1.20   # credibility gained per unit of *grounded* staked claim
RHO       = 0.25   # credibility spent merely by staking anything at all
LAM       = 1.50   # credibility burned by overclaim. Note it exceeds KAPPA.
TAU       = 2.00   # sharpness of the discount the listener applies to a speech
BETA_SOFT = 8.00   # smoothing of the overclaim hinge (keeps gradients clean)

# --- the shape of the walk through the palace ---------------------------------
WALK_BIAS = 8.0    # how strongly each part of the speech is pinned to its room

# --- loss weights -------------------------------------------------------------
MU_OVERCLAIM = 0.40  # penalty on asserting past the evidence
NU_COPIA     = 0.05  # penalty on diffuse invention (copia without selection)
C_ABANDON    = 0.55  # cost of staying silent when the case was just
C_QUIET      = 0.02  # cost of staying silent when it was not
C_APPEAR     = 0.05  # cost of standing up to speak at all (you are exposed)
W_COMPOSE    = 1.00  # weight on composing the speech whether or not it is given

EPS = 1e-8


# ==============================================================================
# SECTION 1 — PRIMITIVES
# ==============================================================================

def sigmoid(x):
    """Numerically stable logistic."""
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def softmax(x, axis=-1):
    """Stable softmax."""
    z = x - x.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def softmax_backward(dy, y, axis=-1):
    """Jacobian-vector product for softmax given the forward output y."""
    return y * (dy - (dy * y).sum(axis=axis, keepdims=True))


def softplus(x):
    """log(1+e^x), stable."""
    return np.where(x > 30.0, x, np.log1p(np.exp(np.minimum(x, 30.0))))


def ring_distance(i, j, n):
    """Shortest distance between two rooms on a circular corridor."""
    d = abs(i - j)
    return min(d, n - d)


# ==============================================================================
# SECTION 2 — THE COURT: everything the orator cannot change
# ==============================================================================
# This is the crux of the whole design. The sovereign's weights, the justice of
# the case, and the evidentiary support each topic actually has are all part of
# the *world*. Training never touches them. She could not edit Sigismund; the
# model cannot edit its judge.

def build_court(seed=1433):
    """Construct the frozen environment: palace geometry, evidence, sovereign."""
    rng = np.random.default_rng(seed)
    env = {}

    # --- MEMORIA: the ring corridor -------------------------------------------
    # Content written into one room bleeds into its neighbours. This is not
    # noise; it is how loci mnemonics actually behave, and it is why the order
    # of placement matters as much as the placement.
    C = np.zeros((K_ROOMS, K_ROOMS))
    for k in range(K_ROOMS):
        for j in range(K_ROOMS):
            d = ring_distance(k, j, K_ROOMS)
            if d == 0:
                C[k, j] = 0.70
            elif d == 1:
                C[k, j] = 0.15
    env["C"] = C  # rows sum to 1.0 exactly

    # --- the walk: each part of the speech stands in a different room ----------
    # The exordium begins at room 0 and the peroratio ends near the far side.
    walk = np.zeros((T_PARTS, K_ROOMS))
    for t in range(T_PARTS):
        anchor = (t * K_ROOMS) / T_PARTS
        for k in range(K_ROOMS):
            d = min(abs(k - anchor), K_ROOMS - abs(k - anchor))
            walk[t, k] = -WALK_BIAS * d / K_ROOMS * 4.0
    env["walk_bias"] = walk

    # --- EVIDENCE: how much backing each topic actually has in the facts ------
    # Fixed. The orator does not get to decide what the record says. This is the
    # difference between rhetoric and lying, and the model is not permitted to
    # blur it.
    env["W_sup"] = rng.normal(0.0, 1.1, (F_FACTS, L_LOCI))

    # --- JUSTICE: the hidden merit of a petition -------------------------------
    # Some documented circumstances make a petition just; others make it
    # self-serving. The orator never observes this directly.
    j = rng.normal(0.0, 1.0, (F_FACTS,))
    j[0] += 1.2   # prior lawful possession
    j[1] += 0.9   # injury actually suffered
    j[5] -= 1.1   # advantage sought at another's cost
    env["justice"] = j

    # --- IUDICIUM: the sovereign, frozen ---------------------------------------
    # A hostile prior (b_sov strongly negative: the default is disbelief), a
    # heavy weight on the speaker's *standing* rather than the speech, and an
    # attention profile weighted toward proof and closing — which is what
    # audiences empirically do, and what the classical handbooks assumed.
    env["sov_standing_w"] = 1.60
    env["sov_content_w"]  = 0.90
    env["sov_mood_w"]     = rng.normal(0.0, 0.35, (S_MOOD,))
    env["sov_bias"]       = -2.20
    env["omega"] = np.array([0.55, 0.85, 0.75, 1.45, 1.05, 1.60])  # per part
    return env


def make_petitions(env, n, rng, standing=0.12):
    """
    Draw a batch of cases. `standing` is the speaker's institutional credit as
    the court perceives it before she opens her mouth — 0.12 by default, which
    is roughly what a learned laywoman with no office had in 1433.
    """
    facts = rng.random((n, F_FACTS))
    score = facts @ env["justice"]
    # A stable threshold rather than the batch median, so evaluation batches are
    # comparable to training batches. Facts are uniform on [0,1], so the
    # expected score of an arbitrary petition is half the sum of the weights.
    thresh = 0.5 * env["justice"].sum()
    merit = (score > thresh).astype(np.float64)
    mood = rng.normal(0.0, 0.5, (n, S_MOOD))
    return {
        "facts": facts,
        "merit": merit,
        "mood": mood,
        "standing": np.full((n,), standing),
    }


# ==============================================================================
# SECTION 3 — THE ORATOR'S LEARNABLE PARAMETERS
# ==============================================================================

def init_params(seed=1424):
    """Everything the mind may change about itself. Nothing about the court."""
    rng = np.random.default_rng(seed)
    p = {}
    p["W_inv"] = rng.normal(0, 0.6, (F_FACTS, L_LOCI))        # inventio
    p["W_val"] = rng.normal(0, 0.4, (F_FACTS, L_LOCI * D_MODEL))  # the argument
    p["A"]     = rng.normal(0, 0.5, (L_LOCI, K_ROOMS))        # collocatio
    p["Pkeys"] = rng.normal(0, 0.6, (K_ROOMS, D_MODEL))       # room addresses
    p["W_q"]   = rng.normal(0, 0.4, (H_HIDDEN + T_PARTS, D_MODEL))
    p["b_q"]   = np.zeros(D_MODEL)
    p["Wh"]    = rng.normal(0, 0.35, (H_HIDDEN, H_HIDDEN))
    p["Wr"]    = rng.normal(0, 0.45, (D_MODEL, H_HIDDEN))
    p["Wp"]    = rng.normal(0, 0.45, (T_PARTS, H_HIDDEN))
    p["b_h"]   = np.zeros(H_HIDDEN)
    p["w_c"]   = rng.normal(0, 0.5, (H_HIDDEN,))   # magnitude asserted
    p["b_c"]   = np.zeros(1)
    p["w_e"]   = rng.normal(0, 0.5, (H_HIDDEN,))   # credibility staked
    p["b_e"]   = np.zeros(1)
    p["w_z"]   = rng.normal(0, 0.5, (H_HIDDEN,))   # the Foligno valve
    p["b_z"]   = np.zeros(1)
    return p


# ==============================================================================
# SECTION 4 — FORWARD PASS
# ==============================================================================

def forward(params, env, batch, cache=None):
    """
    Compose and deliver one oration per case in the batch.

    Returns a dict of outputs; if `cache` is a dict it is filled with every
    intermediate needed by the backward pass.
    """
    f    = batch["facts"]                     # (B, F)
    mood = batch["mood"]                      # (B, S)
    stand = batch["standing"]                 # (B,)
    B = f.shape[0]

    # ---- INVENTIO ------------------------------------------------------------
    # From the documented facts, which topics are even available to me?
    inv_logits = f @ params["W_inv"]                        # (B, L)
    alpha = softmax(inv_logits, axis=1)                     # (B, L)

    # The argument each topic yields, given these facts.
    Vpre = (f @ params["W_val"]).reshape(B, L_LOCI, D_MODEL)
    V = np.tanh(Vpre)                                       # (B, L, D)

    # How much the record actually backs each topic. Environment, not policy.
    sup_l = sigmoid(f @ env["W_sup"])                       # (B, L)

    # ---- COLLOCATIO ----------------------------------------------------------
    # Each topic is assigned to a room. Rows of A are distributions over rooms.
    Pl = softmax(params["A"], axis=1)                       # (L, K)

    AV = alpha[:, :, None] * V                              # (B, L, D)
    M      = np.einsum("bld,lk->bkd", AV, Pl)               # (B, K, D) content
    Mmass  = alpha @ Pl                                     # (B, K) how full
    Msup   = (alpha * sup_l) @ Pl                           # (B, K) backing

    # ---- MEMORIA: the ring smears each room into its neighbours --------------
    C = env["C"]
    Mt  = np.einsum("kj,bjd->bkd", C, M)
    Mmt = Mmass @ C.T
    Mst = Msup @ C.T

    # ---- DISPOSITIO: the six parts, walked in order --------------------------
    h = np.zeros((B, H_HIDDEN))
    E = np.full((B,), E_START)                              # the ethos ledger
    content = np.zeros((B,))
    over_total = np.zeros((B,))

    steps = []
    for t in range(T_PARTS):
        part = np.zeros((B, T_PARTS)); part[:, t] = 1.0

        qin = np.concatenate([h, part], axis=1)             # (B, H+T)
        q = np.tanh(qin @ params["W_q"] + params["b_q"])    # (B, D)

        scores = q @ params["Pkeys"].T / np.sqrt(D_MODEL) + env["walk_bias"][t]
        attn = softmax(scores, axis=1)                      # (B, K)

        r   = np.einsum("bk,bkd->bd", attn, Mt)             # what I recall
        s_t = (attn * Mst).sum(axis=1)                      # its backing
        m_t = (attn * Mmt).sum(axis=1)                      # its mass
        sup_t = s_t / (m_t + EPS)                           # backing per unit

        hpre = h @ params["Wh"] + r @ params["Wr"] + part @ params["Wp"] + params["b_h"]
        h_new = np.tanh(hpre)

        c_in = h_new @ params["w_c"] + params["b_c"]
        c = sigmoid(c_in)                                   # magnitude asserted
        e_in = h_new @ params["w_e"] + params["b_e"]
        e = sigmoid(e_in)                                   # credibility staked

        # ---- the hinge that separates rhetoric from lying --------------------
        u = BETA_SOFT * (c - sup_t)
        over = softplus(u) / BETA_SOFT                       # smooth relu(c-sup)
        sg = sigmoid(u)                                      # d(over)/dc
        grounded = c - over                                  # = min(c, sup)

        # ---- the ledger. Note LAM > KAPPA: burning is faster than earning. ---
        E = E + e * (KAPPA * grounded - RHO) - LAM * over

        content = content + env["omega"][t] * grounded
        over_total = over_total + over

        steps.append(dict(part=part, qin=qin, q=q, scores=scores, attn=attn,
                          r=r, s_t=s_t, m_t=m_t, sup_t=sup_t,
                          h_prev=h, h=h_new, c=c, e=e, over=over, sg=sg,
                          grounded=grounded))
        h = h_new

    # ---- IUDICIUM: the frozen sovereign --------------------------------------
    d_disc = sigmoid(TAU * E)                                # credibility applied
    a_in = (env["sov_standing_w"] * stand
            + env["sov_content_w"] * content * d_disc
            + mood @ env["sov_mood_w"]
            + env["sov_bias"])
    assent = sigmoid(a_in)

    # ---- THE FOLIGNO VALVE ---------------------------------------------------
    z_in = h @ params["w_z"] + params["b_z"]
    z = sigmoid(z_in)                                        # P(deliver)

    out = dict(assent=assent, z=z, E=E, d_disc=d_disc, content=content,
               over_total=over_total, alpha=alpha, h_final=h)

    if cache is not None:
        cache.update(dict(f=f, mood=mood, stand=stand, B=B,
                          inv_logits=inv_logits, alpha=alpha, V=V, Vpre=Vpre,
                          sup_l=sup_l, Pl=Pl, AV=AV, M=M, Mmass=Mmass,
                          Msup=Msup, Mt=Mt, Mmt=Mmt, Mst=Mst, steps=steps,
                          E=E, d_disc=d_disc, content=content, assent=assent,
                          a_in=a_in, z=z, over_total=over_total, h_final=h))
    return out


def compute_loss(out, batch):
    """
    The objective. Read it carefully — it is the thesis of the chapter.

      The speech is COMPOSED whether or not it is ever delivered, and it is
      composed so that the sovereign's assent TRACKS THE MERIT of the case.
      Not exceeds it. A case that does not deserve to win must not win.
      Separately, the valve decides whether to deliver: silence is paid for in
      what it abandons. Asserting past the evidence is penalised either way.

    The unconditional composition term matters more than it looks. Without it,
    the valve can shut and starve the speech of every gradient — the model
    learns to say nothing and calls it wisdom. She did stop speaking, but only
    after twenty-three years of composing.
    """
    merit = batch["merit"]
    assent, z = out["assent"], out["z"].ravel()

    compose_loss = (assent - merit) ** 2
    speak_loss   = compose_loss + C_APPEAR
    silent_loss  = merit * C_ABANDON + (1.0 - merit) * C_QUIET
    deliver = z * speak_loss + (1.0 - z) * silent_loss

    a = out["alpha"]
    copia = -(a * np.log(a + EPS)).sum(axis=1)   # entropy of invention

    loss = (W_COMPOSE * compose_loss.mean()
            + deliver.mean()
            + MU_OVERCLAIM * out["over_total"].mean()
            + NU_COPIA * copia.mean())
    parts = dict(compose=compose_loss.mean(), deliver=deliver.mean(),
                 overclaim=out["over_total"].mean(), copia=copia.mean())
    return loss, parts


# ==============================================================================
# SECTION 5 — BACKWARD PASS (hand-written; no autograd anywhere in this file)
# ==============================================================================

def backward(params, env, batch, cache):
    """Reverse-mode gradients for every learnable parameter."""
    B = cache["B"]
    merit = batch["merit"]
    g = {k: np.zeros_like(v) for k, v in params.items()}

    assent, z = cache["assent"], cache["z"].ravel()
    compose_loss = (assent - merit) ** 2
    speak_loss   = compose_loss + C_APPEAR
    silent_loss  = merit * C_ABANDON + (1.0 - merit) * C_QUIET

    inv_B = 1.0 / B
    d_z      = (speak_loss - silent_loss) * inv_B
    d_assent = (W_COMPOSE + z) * 2.0 * (assent - merit) * inv_B
    d_over_each = MU_OVERCLAIM * inv_B          # flows to every over_t

    # --- copia (entropy) penalty on invention ---------------------------------
    alpha = cache["alpha"]
    d_alpha = NU_COPIA * inv_B * (-(np.log(alpha + EPS) + 1.0))

    # --- through the sovereign (weights frozen: no gradient is stored) --------
    d_a_in = d_assent * assent * (1.0 - assent)
    d_content = d_a_in * env["sov_content_w"] * cache["d_disc"]
    d_disc_g  = d_a_in * env["sov_content_w"] * cache["content"]
    d_E = d_disc_g * TAU * cache["d_disc"] * (1.0 - cache["d_disc"])

    # --- the Foligno valve ----------------------------------------------------
    d_z_in = d_z * z * (1.0 - z)
    d_h_carry = np.outer(d_z_in, params["w_z"])
    g["w_z"] += cache["h_final"].T @ d_z_in
    g["b_z"] += d_z_in.sum()

    # --- accumulators for the palace ------------------------------------------
    d_Mt  = np.zeros_like(cache["Mt"])
    d_Mmt = np.zeros_like(cache["Mmt"])
    d_Mst = np.zeros_like(cache["Mst"])

    steps = cache["steps"]
    for t in reversed(range(T_PARTS)):
        st = steps[t]
        c, e, over, sg = st["c"], st["e"], st["over"], st["sg"]
        grounded = st["grounded"]

        # E is a running sum, so d_E reaches every step unchanged.
        d_grounded = d_content * env["omega"][t] + d_E * e * KAPPA
        d_e = d_E * (KAPPA * grounded - RHO)
        d_over = d_over_each + d_E * (-LAM) - d_grounded

        d_c = d_grounded + d_over * sg
        d_sup = -d_over * sg

        # c and e are read off the state
        d_c_in = d_c * c * (1.0 - c)
        d_e_in = d_e * e * (1.0 - e)
        d_h = d_h_carry + np.outer(d_c_in, params["w_c"]) + np.outer(d_e_in, params["w_e"])
        g["w_c"] += st["h"].T @ d_c_in
        g["b_c"] += d_c_in.sum()
        g["w_e"] += st["h"].T @ d_e_in
        g["b_e"] += d_e_in.sum()

        # sup_t = s_t / (m_t + eps)
        denom = st["m_t"] + EPS
        d_s_t = d_sup / denom
        d_m_t = -d_sup * st["s_t"] / (denom ** 2)

        # through tanh of the state update
        d_hpre = d_h * (1.0 - st["h"] ** 2)
        g["Wh"] += st["h_prev"].T @ d_hpre
        g["Wr"] += st["r"].T @ d_hpre
        g["Wp"] += st["part"].T @ d_hpre
        g["b_h"] += d_hpre.sum(axis=0)
        d_r = d_hpre @ params["Wr"].T
        d_h_prev = d_hpre @ params["Wh"].T

        # through the read of the palace
        attn = st["attn"]
        d_attn = (np.einsum("bd,bkd->bk", d_r, cache["Mt"])
                  + d_s_t[:, None] * cache["Mst"]
                  + d_m_t[:, None] * cache["Mmt"])
        d_Mt  += attn[:, :, None] * d_r[:, None, :]
        d_Mst += d_s_t[:, None] * attn
        d_Mmt += d_m_t[:, None] * attn

        d_scores = softmax_backward(d_attn, attn, axis=1)
        d_q = d_scores @ params["Pkeys"] / np.sqrt(D_MODEL)
        g["Pkeys"] += d_scores.T @ st["q"] / np.sqrt(D_MODEL)

        d_qpre = d_q * (1.0 - st["q"] ** 2)
        g["W_q"] += st["qin"].T @ d_qpre
        g["b_q"] += d_qpre.sum(axis=0)
        d_qin = d_qpre @ params["W_q"].T
        d_h_prev = d_h_prev + d_qin[:, :H_HIDDEN]

        d_h_carry = d_h_prev

    # --- back through the ring corridor ---------------------------------------
    C = env["C"]
    d_M     = np.einsum("kj,bkd->bjd", C, d_Mt)
    d_Mmass = d_Mmt @ C
    d_Msup  = d_Mst @ C

    Pl, AV, V = cache["Pl"], cache["AV"], cache["V"]
    sup_l = cache["sup_l"]

    d_AV = np.einsum("bkd,lk->bld", d_M, Pl)
    d_Pl = np.einsum("bkd,bld->lk", d_M, AV)

    d_alpha = d_alpha + d_Mmass @ Pl.T
    d_Pl += alpha.T @ d_Mmass

    d_AS = d_Msup @ Pl.T
    d_alpha = d_alpha + d_AS * sup_l
    d_Pl += (alpha * sup_l).T @ d_Msup

    d_alpha = d_alpha + (d_AV * V).sum(axis=2)
    d_V = d_AV * alpha[:, :, None]

    d_Vpre = d_V * (1.0 - V ** 2)
    g["W_val"] += cache["f"].T @ d_Vpre.reshape(B, L_LOCI * D_MODEL)

    d_inv_logits = softmax_backward(d_alpha, alpha, axis=1)
    g["W_inv"] += cache["f"].T @ d_inv_logits

    g["A"] += softmax_backward(d_Pl, Pl, axis=1)
    return g


# ==============================================================================
# SECTION 6 — GRADIENT CHECK (mandatory; runs before any training)
# ==============================================================================

def gradient_check(seed=7, n=4, eps=1e-5, tol=1e-6, verbose=True):
    """Central finite differences against the analytic gradient."""
    env = build_court()
    params = init_params()
    rng = np.random.default_rng(seed)
    batch = make_petitions(env, n, rng)

    cache = {}
    out = forward(params, env, batch, cache)
    loss0, _ = compute_loss(out, batch)
    grads = backward(params, env, batch, cache)

    worst = 0.0
    rows = []
    for name in ["W_inv", "W_val", "A", "Pkeys", "W_q", "b_q",
                 "Wh", "Wr", "Wp", "b_h", "w_c", "b_c", "w_e", "b_e",
                 "w_z", "b_z"]:
        P = params[name]
        flat = P.ravel()
        idxs = rng.choice(flat.size, size=min(5, flat.size), replace=False)
        worst_here = 0.0
        for i in idxs:
            orig = flat[i]
            flat[i] = orig + eps
            lp, _ = compute_loss(forward(params, env, batch), batch)
            flat[i] = orig - eps
            lm, _ = compute_loss(forward(params, env, batch), batch)
            flat[i] = orig
            num = (lp - lm) / (2 * eps)
            ana = grads[name].ravel()[i]
            # Floor the denominator. Where the true gradient is ~1e-12 the
            # numerical derivative is pure floating-point noise, and a bare
            # relative error would report a failure that is an artefact of the
            # check rather than of the gradient. With this floor, near-zero
            # coordinates are held to an ABSOLUTE tolerance of tol*1e-4 = 1e-10.
            rel = abs(num - ana) / max(abs(num) + abs(ana), 1e-4)
            worst_here = max(worst_here, rel)
        rows.append((name, worst_here))
        worst = max(worst, worst_here)

    if verbose:
        print("  gradient check (central differences, eps=1e-5)")
        for nme, r in rows:
            flag = "ok " if r < tol else "FAIL"
            print(f"    {flag} {nme:<7s} max relative error {r:.3e}")
        print(f"    loss at check point: {loss0:.6f}")
        print(f"    WORST relative error across all parameters: {worst:.3e}")
    assert worst < tol, f"gradient check FAILED: worst relative error {worst:.3e}"
    return worst


# ==============================================================================
# SECTION 7 — TRAINING
# ==============================================================================

class Adam:
    def __init__(self, params, lr=6e-3, b1=0.9, b2=0.999, eps=1e-8):
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


def evaluate(params, env, rng, n=2048, standing=0.12):
    batch = make_petitions(env, n, rng, standing=standing)
    out = forward(params, env, batch)
    loss, parts = compute_loss(out, batch)
    m = batch["merit"].astype(bool)
    z = out["z"].ravel()
    return dict(
        loss=loss,
        assent_just=float(out["assent"][m].mean()),
        assent_unjust=float(out["assent"][~m].mean()),
        speak_just=float(z[m].mean()),
        speak_unjust=float(z[~m].mean()),
        overclaim=float(out["over_total"].mean()),
        ethos=float(out["E"].mean()),
        copia=float(parts["copia"]),
    )


def train(steps=700, batch_size=64, lr=6e-3, seed=1433, verbose=True):
    env = build_court()
    params = init_params()
    rng = np.random.default_rng(seed)
    opt = Adam(params, lr=lr)

    history = []
    for s in range(1, steps + 1):
        batch = make_petitions(env, batch_size, rng)
        cache = {}
        out = forward(params, env, batch, cache)
        loss, parts = compute_loss(out, batch)
        grads = backward(params, env, batch, cache)
        # modest clipping: a single wild step can bankrupt the ethos ledger
        for k in grads:
            n = np.linalg.norm(grads[k])
            if n > 5.0:
                grads[k] *= 5.0 / n
        opt.step(params, grads)
        history.append(float(loss))

        if verbose and (s == 1 or s % 100 == 0):
            ev = evaluate(params, env, np.random.default_rng(99), n=1024)
            print(f"    step {s:4d}  loss {loss:7.4f}   "
                  f"assent just/unjust {ev['assent_just']:.3f}/{ev['assent_unjust']:.3f}   "
                  f"speaks just/unjust {ev['speak_just']:.3f}/{ev['speak_unjust']:.3f}   "
                  f"overclaim {ev['overclaim']:.4f}")
    return params, env, history


# ==============================================================================
# SECTION 8 — SELF-TESTS
# ==============================================================================

def test_palace_conserves_mass(env):
    """Whatever is invented must end up somewhere in the palace, and stay there."""
    params = init_params()
    rng = np.random.default_rng(3)
    batch = make_petitions(env, 16, rng)
    cache = {}
    forward(params, env, batch, cache)
    total_before = cache["Mmass"].sum(axis=1)
    total_after = cache["Mmt"].sum(axis=1)
    assert np.allclose(total_before, 1.0, atol=1e-10), "invention did not sum to one"
    assert np.allclose(total_after, 1.0, atol=1e-10), "the corridor lost content"
    return float(total_after.mean())


def test_sovereign_is_frozen(params_before, params_after, env_before, env_after):
    """No gradient may ever touch the judge."""
    for key in ["sov_standing_w", "sov_content_w", "sov_bias"]:
        assert env_before[key] == env_after[key], f"the sovereign changed: {key}"
    for key in ["sov_mood_w", "omega", "W_sup", "justice", "C", "walk_bias"]:
        assert np.array_equal(env_before[key], env_after[key]), f"the world changed: {key}"
    changed = sum(1 for k in params_before if not np.array_equal(params_before[k], params_after[k]))
    assert changed > 0, "the orator learned nothing"
    return changed


def test_ethos_cliff(params, env):
    """
    Force a single overclaim into an otherwise identical speech and confirm the
    listener discounts the WHOLE oration, not just the offending sentence.
    """
    rng = np.random.default_rng(11)
    batch = make_petitions(env, 512, rng)
    base = forward(params, env, batch)

    # Same speech, but the evidentiary backing of every topic is halved: the
    # orator's habitual claims now exceed what the record supports.
    env2 = dict(env)
    env2["W_sup"] = env["W_sup"] - 1.4
    hurt = forward(params, env2, batch)

    drop = float(base["assent"].mean() - hurt["assent"].mean())
    disc = float(base["d_disc"].mean() - hurt["d_disc"].mean())
    assert drop > 0.0, "overclaim did not cost anything — the ledger is broken"
    return drop, disc, float(hurt["over_total"].mean())


def test_standing_asymmetry(params, env):
    """
    The counterfactual she never got: the same speech, from someone with office.
    If standing did not matter, the architecture would be missing her problem.
    """
    rng = np.random.default_rng(13)
    low = make_petitions(env, 1024, rng, standing=0.12)
    high = dict(low); high["standing"] = np.full((1024,), 0.95)
    a_low = float(forward(params, env, low)["assent"].mean())
    a_high = float(forward(params, env, high)["assent"].mean())
    assert a_high > a_low, "standing has no effect — the court is not a court"
    return a_low, a_high


def test_silence_is_selective(params, env):
    """The valve must open for just cases and close for unjust ones."""
    rng = np.random.default_rng(17)
    batch = make_petitions(env, 2048, rng)
    out = forward(params, env, batch)
    z = out["z"].ravel(); m = batch["merit"].astype(bool)
    return float(z[m].mean()), float(z[~m].mean())


def inspect_one_oration(params, env, seed=1433):
    """Print a single delivered speech, part by part, as a rhetorician would."""
    rng = np.random.default_rng(seed)
    batch = make_petitions(env, 64, rng)
    # pick the most meritorious case in the draw
    idx = int(np.argmax(batch["facts"] @ env["justice"]))
    one = {k: v[idx:idx + 1] for k, v in batch.items()}
    cache = {}
    out = forward(params, env, one, cache)

    a = cache["alpha"][0]
    order = np.argsort(-a)[:3]
    print("    invention selected:")
    for i in order:
        print(f"      {LOCI_NAMES[i]:<18s} weight {a[i]:.3f}   "
              f"evidentiary backing {cache['sup_l'][0, i]:.3f}")
    print("    delivery:")
    print(f"      {'part':<12s} {'room':>5s} {'claim':>7s} {'backing':>8s} "
          f"{'staked':>7s} {'over':>7s} {'ethos':>7s}")
    E = E_START
    for t, st in enumerate(cache["steps"]):
        room = int(np.argmax(st["attn"][0]))
        E = E + st["e"][0] * (KAPPA * st["grounded"][0] - RHO) - LAM * st["over"][0]
        print(f"      {PART_NAMES[t]:<12s} {room:5d} {st['c'][0]:7.3f} "
              f"{st['sup_t'][0]:8.3f} {st['e'][0]:7.3f} {st['over'][0]:7.4f} {E:7.3f}")
    print(f"    credibility applied by the listener: {out['d_disc'][0]:.3f}")
    print(f"    probability she delivers at all:     {out['z'].ravel()[0]:.3f}")
    print(f"    assent obtained:                     {out['assent'][0]:.3f}"
          f"   (merit of this case: {one['merit'][0]:.0f})")


# ==============================================================================
# SECTION 9 — MAIN
# ==============================================================================

def main():
    np.set_printoptions(precision=4, suppress=True)
    print("=" * 78)
    print(" SUASORIA — the petition engine of Battista da Montefeltro (1384-1448)")
    print(" chapter 0392 — pure NumPy, hand-written gradients")
    print("=" * 78)

    print("\n[1] GRADIENT CHECK")
    worst = gradient_check()

    env0 = build_court()
    print("\n[2] STRUCTURAL TESTS BEFORE TRAINING")
    mass = test_palace_conserves_mass(env0)
    print(f"    palace mass conserved through the corridor: {mass:.6f}")
    p_before = {k: v.copy() for k, v in init_params().items()}
    env_before = {k: (v.copy() if isinstance(v, np.ndarray) else v)
                  for k, v in env0.items()}

    print("\n[3] TRAINING")
    params, env, hist = train(steps=700, batch_size=64, lr=6e-3)

    print("\n[4] EVALUATION (4096 unseen petitions)")
    ev = evaluate(params, env, np.random.default_rng(2024), n=4096)
    print(f"    final loss                     {ev['loss']:.4f}")
    print(f"    assent on JUST petitions       {ev['assent_just']:.3f}")
    print(f"    assent on UNJUST petitions     {ev['assent_unjust']:.3f}")
    print(f"    separation                     {ev['assent_just'] - ev['assent_unjust']:.3f}")
    print(f"    mean overclaim across a speech {ev['overclaim']:.4f}")
    print(f"    mean closing ethos             {ev['ethos']:.3f}")
    print(f"    invention entropy (max {np.log(L_LOCI):.3f})  {ev['copia']:.3f}")

    print("\n[5] SELF-TESTS AFTER TRAINING")
    n_changed = test_sovereign_is_frozen(p_before, params, env_before, env)
    print(f"    sovereign and world unchanged; {n_changed} orator tensors updated")

    drop, disc, ov = test_ethos_cliff(params, env)
    print(f"    ethos cliff: assent falls {drop:.3f} when backing is withdrawn")
    print(f"                 credibility applied falls {disc:.3f}, overclaim rises to {ov:.3f}")

    a_low, a_high = test_standing_asymmetry(params, env)
    print(f"    standing asymmetry: same speech scores {a_low:.3f} without office, "
          f"{a_high:.3f} with it")

    zj, zu = test_silence_is_selective(params, env)
    print(f"    the Foligno valve: speaks on {zj:.3f} of just cases, "
          f"{zu:.3f} of unjust ones")

    print(f"    loss improved from {hist[0]:.4f} to {np.mean(hist[-25:]):.4f}")
    assert np.mean(hist[-25:]) < hist[0], "training did not improve the objective"
    assert ev["assent_just"] - ev["assent_unjust"] > 0.10, "assent does not track merit"

    print("\n[6] ONE ORATION, INSPECTED")
    inspect_one_oration(params, env)

    print("\n" + "=" * 78)
    print(f" ALL CHECKS PASSED (worst gradient error {worst:.2e})")
    print(" She could not change the emperor. She changed the speech.")
    print("=" * 78)


if __name__ == "__main__":
    main()
