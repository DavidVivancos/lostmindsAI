"""
================================================================================
 CHAPTER 308 -- FARID UD-DIN ATTAR OF NISHAPUR (c.1145-1221 CE)
 THE VALLEY-ATTRITION NETWORK  (a.k.a. the "Si-Murgh Architecture")
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0308 · Farid ud-Din Attar of Nishapur (c.1145-1221 CE)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 16, Minds 301-320 Available on Amazon https://www.amazon.com/gp/product/B0HNJQX9S4
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos 

WHAT THIS FILE IS
------------------
A pure-NumPy, from-scratch, trainable neural architecture whose *mechanism*
is derived from the specific cognitive structure Attar describes across two
of his long narrative poems: the Mantiq al-Tayr ("The Conference/Speech of
the Birds", completed c. 1177) and the Musibat-Nama ("The Book of
Affliction", c. 1180-1190). It is not a generic transformer wearing a
Sufi costume: every layer corresponds to a textually-attested step in
Attar's own account of how a mind searches, fails, sheds itself, and
finds that the object of its search was constituted by its own converged,
purified population all along.

THE TWO TEXTS THIS ARCHITECTURE ENCODES
----------------------------------------
1. Mantiq al-Tayr: thousands of birds, led by the hoopoe, set out to find
   their king, the Simurgh. Each bird is characterised by ONE specific,
   named attachment that furnishes its excuse not to go (the nightingale's
   delusive love of the rose, the parrot's fear of losing a safe cage in
   exchange for a search that might kill it, the peacock's nostalgia for a
   paradise it believes it once owned, the partridge's avarice for jewels,
   the duck's obsessive ritual attachment to water, the homa's pride in
   bestowing status on others, the hawk's ambition to serve earthly kings,
   the heron's melancholy attachment to the shoreline, the owl's greed for
   buried treasure, the finch's self-protective claim of weakness). The
   birds cross seven valleys -- Quest, Love, Knowledge, Detachment, Unity,
   Bewilderment, and Poverty/Annihilation -- and most die or turn back
   *at the point where their specific attachment is tested*. Only thirty
   (si morgh) arrive. When a curtain is drawn back before the throne of the
   Simurgh, they see only a mirror: they themselves, thirty birds, ARE the
   Simurgh (si-murgh / Simurgh is a pun built into the myth's own name).
   The sought object is the population's own purified, aggregated state.

2. Musibat-Nama: a single wayfarer of thought (salik-i fikrat, "Fikrat")
   travels the entire furnished cosmos -- the angelic Throne, the celestial
   spheres, the four elements, minerals, plants, animals -- and puts the
   same question to forty different beings in turn. Every one of them
   answers "not here; ask further." The outward, hierarchical search of
   the whole created order returns no terminal answer; Fikrat is finally
   directed inward, to "the ocean of his own soul."

THE ARCHITECTURE, IN ONE SENTENCE
----------------------------------
An outward recursive query across a fixed ontology (FikratQuery, standing
in for the Musibat-Nama's forty stations, compressed here to five
ontological classes: mineral, vegetable, animal, human, angelic) is
trained to find NOTHING externally -- its own objective pushes its
"deflection" toward 1.0 at every level -- and what leaks through becomes
only a small residual bias contributed to a population of ten named,
individually-biased agents (the Valley Network, standing in for the ten
canonically-named excuse-birds of the Mantiq al-Tayr). That population is
then passed through seven valley layers matched one-for-one to Attar's
own sequence. Agents are never physically deleted (that would break
differentiability); instead each valley multiplies a per-agent survival
weight, and the final prediction is an aggregate of the population's own
converged state -- there is no external label vector stored anywhere in
the forward pass that the network is told to copy. The "answer" the
network is trained to produce IS a function of what its own population
already carried in.

WHAT IS AND IS NOT LEARNED, AND WHY (an honest design note)
-------------------------------------------------------------
Two of the seven valleys (Love, Detachment) end in a sigmoid "survival
gate" -- a soft, continuous stand-in for the poem's birds dying or
turning back. Each gate does double duty, and both duties are trained
together: (1) it sets each agent's trust-weight when the Unity valley
pools the population toward a shared consensus, so a gate that is wrong
directly costs the network accuracy on the main task; and (2) it is
separately, directly supervised toward an "honest diagnosis" target
built from the true attachment magnitude that generated each agent
(a bird that was mostly its attachment, and little else, should be
judged as such). Both objectives push gradient into the SAME threshold
parameters (theta_love for Love, eps_thresh for Detachment) and, for
Detachment, the same upstream reduction parameter (alpha) -- there is
no stop-gradient anywhere in the graph. (The Love valley's own
intensification, rho, is deliberately NOT learned -- see the note next
to LOVE_RHO below: an early trainable version of it was driven straight
to zero by gradient descent, which had discovered that the cheapest way
to serve the main reconstruction loss was to never let Love intensify
anything in the first place. Attar's valleys are not something the
traveller bargains with, and this file now enforces that structurally.)
Every remaining path is carried through ordinary backpropagation and
verified below against finite differences; the two duties of a single
gate are additive terms in one honest loss, not two disconnected
mechanisms. The remaining parameters -- the projection that sets
collective direction in the Quest valley, the casebook of remembered
exemplars read in the Knowledge valley, Fikrat's outward-search weights,
the Unity valley's blend coefficient, and the final linear readout --
are trained purely by the main task loss, through the same single
backward pass.

FILE MAP
--------
  1. Small numeric primitives (sigmoid, softmax, stable norms)
  2. Bird registry (the ten named attachments + hoopoe, for narration)
  3. Parameter initialisation
  4. forward_one()   -- full forward pass for a single population, with cache
  5. backward_one()  -- full analytic backward pass (hand-derived, see the
                         chapter's "Architecture" section for the derivation
                         of why several terms collapse the way they do)
  6. Synthetic data generator (the "Simurgh" the population must recover)
  7. Adam optimiser (from scratch)
  8. Finite-difference gradient check (mandatory, see __main__)
  9. Training loop
 10. Self-tests (shape, gradient, behavioural, and learning-curve tests)
 11. __main__ demonstration and verified execution report
================================================================================
"""

import numpy as np

RNG_SEED = 20260731  # date this chapter was assembled -- fixed for reproducibility

# ------------------------------------------------------------------------
# 1. NUMERIC PRIMITIVES
# ------------------------------------------------------------------------

def sigmoid(x):
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-12)


def safe_norm(v, eps=1e-8):
    return float(np.sqrt(np.sum(v * v)) + eps)


# ------------------------------------------------------------------------
# 2. BIRD REGISTRY -- ten named attachments (agent slot i owns axis e_i),
#    grounded in the poem's own cast (Darbandi & Davis translation;
#    see chapter Sources). The Hoopoe is the guide, not a searching agent,
#    and is represented instead as a single learned vector added to every
#    agent's Quest-valley update.
# ------------------------------------------------------------------------

BIRDS = [
    dict(name="Nightingale", attachment="delusive love of the rose"),
    dict(name="Parrot",      attachment="false immortality, safety of the cage"),
    dict(name="Peacock",     attachment="nostalgia for a lost, illusory paradise"),
    dict(name="Partridge",   attachment="avarice for jewels and stones"),
    dict(name="Duck",        attachment="ritual attachment to water/purity"),
    dict(name="Homa",        attachment="pride in bestowing status on others"),
    dict(name="Hawk",        attachment="ambition to serve an earthly king"),
    dict(name="Heron",       attachment="melancholy longing for the shoreline"),
    dict(name="Owl",         attachment="greed for buried treasure"),
    dict(name="Finch",       attachment="self-protective claim of weakness"),
]
NB = len(BIRDS)          # 10 named agents per population
SIGNAL_DIM = 14          # dimensions the population's shared "Simurgh" lives in
D = NB + SIGNAL_DIM      # 24 -- first NB dims are personal-bias axes, rest is shared signal
MEM = 12                 # size of the Ma'rifat casebook (remembered saint-stories)
K = 5                    # number of latent "Simurgh" concepts the corpus must tell apart
R = 5                    # Musibat-Nama's ontological classes: mineral, vegetable, animal,
                          # human, angelic (compressing its forty examined beings into the
                          # five classes Attar himself sorts them into)
REALM_NAMES = ["mineral", "vegetable", "animal", "human", "angelic"]

K1 = 4.0                 # sharpness of the Love valley's survival gate
K2 = 4.0                 # sharpness of the Detachment valley's survival gate
LOVE_RHO = 1.2            # fixed Love-valley intensification (see note above init_params)
TARGET_DEFLECT = 0.92    # Fikrat's outward search is trained to fail almost everywhere
LAMBDA_COS = 0.5         # weight of the reconstruction ("were we the Simurgh?") loss
LAMBDA_DEFLECT = 0.05    # weight of Fikrat's own auxiliary objective
LAMBDA_GATE = 0.5        # weight of the two valleys' honest-diagnosis objective
BIAS_REF = 1.75          # fixed reference magnitude for "how attached is too attached"
                          # (the midpoint of the default sampling range below); using a
                          # FIXED reference rather than each population's own median is
                          # what makes the honest-diagnosis target learnable at all -- a
                          # per-population median is a moving target (the same bird with
                          # the same bias can be judged differently depending only on who
                          # else happens to be in that population), and an early version
                          # of this file that used the per-population median taught the
                          # network the degenerate shortcut of predicting a constant 0.5
                          # for every bird rather than actually reading its attachment.


# ------------------------------------------------------------------------
# 3. PARAMETER INITIALISATION
# ------------------------------------------------------------------------

def init_params(rng):
    scale = 1.0 / np.sqrt(D)
    p = {}
    # near-identity initialisation: keeps each agent's own diagonal entry
    # (which is where its personal attachment magnitude lives, see AXES
    # below) legible at the start of training, so the Love/Detachment
    # gates have a real signal to learn from rather than an immediate
    # mush of cross-agent mixing. The network remains free to move away
    # from identity; nothing after initialisation constrains it.
    p["W_talab"] = np.eye(D) + rng.normal(0, scale * 0.3, size=(D, D)).astype(np.float64)
    p["hoopoe_vec"] = rng.normal(0, 0.05, size=(D,)).astype(np.float64)
    p["W_fikrat"] = rng.normal(0, scale, size=(R, D)).astype(np.float64)
    p["b_fikrat"] = np.zeros(R, dtype=np.float64)
    p["theta_love"] = np.array(0.0, dtype=np.float64)    # Love valley judgment threshold
    p["M"] = rng.normal(0, scale, size=(MEM, D)).astype(np.float64)
    p["eta_m"] = np.array(0.3, dtype=np.float64)          # Knowledge valley blend strength
    p["alpha_raw"] = np.array(0.5, dtype=np.float64)      # Detachment strength (pre-sigmoid)
    p["eps_thresh"] = np.array(0.3, dtype=np.float64)     # Detachment valley judgment threshold
    p["beta_raw"] = np.array(0.0, dtype=np.float64)       # Unity valley blend (pre-sigmoid)
    p["W_out"] = rng.normal(0, scale, size=(D, K)).astype(np.float64)
    p["b_out"] = np.zeros(K, dtype=np.float64)

    # fixed (non-learned) realm archetypes for the Fikrat module, and the
    # per-bird bias axes, which by construction are simply the standard
    # basis vectors e_0..e_{NB-1} -- agent i's one and only attachment
    # dimension is dimension i. This is not a simplification of convenience
    # only: it is what lets every agent's excuse be legible and separable,
    # exactly as the poem insists each bird's flaw is named and distinct.
    realm_embeds = rng.normal(0, 1.0, size=(R, D))
    realm_embeds[:, :NB] = 0.0  # realms speak only in the shared-signal subspace
    realm_embeds = realm_embeds / (np.linalg.norm(realm_embeds, axis=1, keepdims=True) + 1e-8)

    consts = dict(realm_embeds=realm_embeds)
    return p, consts


def get_axis(i):
    e = np.zeros(D)
    e[i] = 1.0
    return e


AXES = np.eye(D)[:NB]  # AXES[i] == get_axis(i)


# ------------------------------------------------------------------------
# 4. FORWARD PASS (single population)
# ------------------------------------------------------------------------

def forward_one(params, consts, x0, rng, train=True, hayrat_gamma=0.15):
    """
    x0: (NB, D) initial embeddings for the ten named agents of ONE population.
    Returns (yhat, logits, probs, cache) where cache holds everything the
    backward pass needs.
    """
    W_talab, hoopoe_vec = params["W_talab"], params["hoopoe_vec"]
    W_fikrat, b_fikrat = params["W_fikrat"], params["b_fikrat"]
    rho, theta_love = LOVE_RHO, float(params["theta_love"])
    M, eta_m = params["M"], float(params["eta_m"])
    alpha = float(sigmoid(params["alpha_raw"]))
    eps_thresh = float(params["eps_thresh"])
    beta = float(sigmoid(params["beta_raw"]))
    W_out, b_out = params["W_out"], params["b_out"]
    realm_embeds = consts["realm_embeds"]

    # ---- Fikrat's outward query (Musibat-Nama) --------------------------
    x_query = x0.mean(axis=0)                                   # (D,)
    z_r = W_fikrat @ x_query + b_fikrat                          # (R,)
    deflect = sigmoid(z_r)                                       # (R,) -> "not here"
    h = ((1.0 - deflect)[:, None] * realm_embeds).sum(axis=0)    # (D,) residual leak

    # ---- Valley 1: Talab (the Quest) ------------------------------------
    talab_pre = x0 @ W_talab + h[None, :] + hoopoe_vec[None, :]  # (NB, D)
    x1 = np.tanh(talab_pre)

    # ---- Valley 2: Ishq (Love) -- amplify + honest survival gate --------
    pm_after = x1[np.arange(NB), np.arange(NB)] * (1.0 + rho)    # (NB,) diag entries
    x2 = x1.copy()
    x2[np.arange(NB), np.arange(NB)] = pm_after
    gate1 = sigmoid(-K1 * (pm_after - theta_love))                # (NB,) survive prob.
    w1 = gate1

    # ---- Valley 3: Ma'rifat (Knowledge) -- read the casebook -------------
    scores = (x2 @ M.T) / np.sqrt(D)                               # (NB, MEM)
    attn = np.array([softmax(scores[i]) for i in range(NB)])      # (NB, MEM)
    read = attn @ M                                                # (NB, D)
    x3 = x2 + eta_m * read

    # ---- Valley 4: Istighna (Detachment) -- project out one's own axis --
    diag3 = x3[np.arange(NB), np.arange(NB)].copy()
    x4 = x3.copy()
    x4[np.arange(NB), np.arange(NB)] = diag3 * (1.0 - alpha)
    # the valley judges by how much attachment REMAINS along the agent's
    # own named axis after the projection -- not by the full state norm,
    # which is dominated by the shared Ma'rifat casebook read and carries
    # no information about this particular judgment.
    resid = np.abs(diag3) * (1.0 - alpha) + 1e-8                    # (NB,)
    gate2 = sigmoid(-K2 * (resid - eps_thresh))                      # high residual -> eliminated
    w2 = w1 * gate2
    wn = w2 / (w2.sum() + 1e-8)                                    # normalised, frozen weights

    # ---- Valley 5: Tawhid (Unity) -- blend toward trust-weighted consensus
    consensus5 = (wn[:, None] * x4).sum(axis=0)                   # (D,)
    x5 = (1.0 - beta) * x4 + beta * consensus5[None, :]

    # ---- Valley 6: Hayrat (Bewilderment) -- calibrated noise -------------
    pop_std = float(np.std(x5)) if train else 0.0
    noise = rng.normal(0, hayrat_gamma * pop_std, size=x5.shape) if train else 0.0
    x6 = x5 + noise

    # ---- Valley 7: Faqr o Fana (Poverty & Annihilation) -------------------
    # total, indiscriminate dissolution: every agent counts equally now,
    # even the ones Unity had learned to trust less.
    yhat = x6.mean(axis=0)                                         # (D,)

    logits = yhat @ W_out + b_out
    probs = softmax(logits)

    cache = dict(x0=x0, x_query=x_query, z_r=z_r, deflect=deflect, h=h,
                 talab_pre=talab_pre, x1=x1, pm_after=pm_after, x2=x2, gate1=gate1, w1=w1,
                 scores=scores, attn=attn, read=read, x3=x3,
                 diag3=diag3, x4=x4, resid=resid, gate2=gate2, w2=w2, wn=wn,
                 consensus5=consensus5, x5=x5, x6=x6, yhat=yhat, logits=logits, probs=probs,
                 alpha=alpha, beta=beta, rho=rho, theta_love=theta_love, eta_m=eta_m,
                 eps_thresh=eps_thresh)
    return yhat, logits, probs, cache


# ------------------------------------------------------------------------
# 5. BACKWARD PASS (single population) -- hand-derived analytic gradients
#
# This is FULL, proper backpropagation through every path in the forward
# graph -- including through the two survival gates (gate1 in the Love
# valley, gate2 in the Detachment valley), which are used BOTH as the
# Unity valley's trust-weighting (wn, feeding consensus5) AND as the
# target of a direct "honest diagnosis" supervision term. Both uses of
# gate1/gate2 contribute gradient, and both contributions are summed
# below before continuing the chain back into theta_love, eps_thresh,
# rho, alpha and, through them, into x1/x3/x4 and everything upstream.
# An earlier draft of this file tried to treat gate1/gate2 as stop-gradient
# constants purely for the pooling step, which is mathematically a
# DIFFERENT (smaller) function than the one forward_one() actually
# computes -- and a finite-difference check correctly rejected it, since
# central differences always measure the true total derivative of
# whatever forward_one() computes, not of some intended subgraph of it.
# The fix is this: differentiate the function that is actually there.
# ------------------------------------------------------------------------

def backward_one(params, consts, cache, s_true, label, true_bias):
    """
    Returns (loss_total, loss_parts, grads) where grads has the same keys
    as `params`.
    """
    realm_embeds = consts["realm_embeds"]
    x0 = cache["x0"]; x_query = cache["x_query"]; deflect = cache["deflect"]
    x1 = cache["x1"]; pm_after = cache["pm_after"]; x2 = cache["x2"]
    gate1 = cache["gate1"]; scores = cache["scores"]; attn = cache["attn"]
    x3 = cache["x3"]; diag3 = cache["diag3"]; x4 = cache["x4"]
    resid = cache["resid"]; gate2 = cache["gate2"]; w2 = cache["w2"]; wn = cache["wn"]
    consensus5 = cache["consensus5"]; x5 = cache["x5"]
    yhat = cache["yhat"]; logits = cache["logits"]; probs = cache["probs"]
    alpha = cache["alpha"]; beta = cache["beta"]; rho = cache["rho"]

    M = params["M"]; W_out = params["W_out"]
    idx = np.arange(NB)

    grads = {k: np.zeros_like(v) for k, v in params.items()}

    # ---- loss values -------------------------------------------------
    ce = -np.log(probs[label] + 1e-12)
    na, nb_ = safe_norm(yhat), safe_norm(s_true)
    num = float(np.dot(yhat, s_true))
    cos_sim = num / (na * nb_)
    cos_loss = 1.0 - cos_sim
    defl_aux = float(np.mean((TARGET_DEFLECT - deflect) ** 2))
    true_worth = sigmoid(-(true_bias - BIAS_REF))                       # (NB,) target in (0,1)
    gate_loss = LAMBDA_GATE * float(np.mean((gate1 - true_worth) ** 2)
                                     + np.mean((gate2 - true_worth) ** 2))
    loss_total = ce + LAMBDA_COS * cos_loss + LAMBDA_DEFLECT * defl_aux + gate_loss

    # ---- readout backward (-> g_yhat contribution 1) -------------------
    dlogits = probs.copy()
    dlogits[label] -= 1.0
    grads["W_out"] += np.outer(yhat, dlogits)
    grads["b_out"] += dlogits
    g_yhat = W_out @ dlogits

    # ---- cosine loss backward (-> g_yhat contribution 2) ----------------
    dcos_dyhat = s_true / (na * nb_) - num * yhat / (na ** 3 * nb_)
    g_yhat += -LAMBDA_COS * dcos_dyhat

    # ---- Fana: yhat = mean_i x6_i ; x6_i = x5_i + noise(stopgrad) --------
    g5 = g_yhat / NB                                                   # (D,) same for every i
    dL_dx5 = np.tile(g5[None, :], (NB, 1))                              # (NB,D)

    # ---- Unity (Tawhid): x5_i = (1-beta) x4_i + beta * consensus5 -------
    #      consensus5 = sum_j wn_j x4_j
    A = (1.0 - beta) * g5                                               # (D,) constant across i
    Gc = beta * g_yhat                                                  # (D,) dL/d(consensus5)
    dbeta = float(np.sum(dL_dx5 * (consensus5[None, :] - x4)))
    grads["beta_raw"] += dbeta * beta * (1.0 - beta)

    dL_dx4 = np.tile(A[None, :], (NB, 1)) + wn[:, None] * Gc[None, :]    # direct part
    dL_dwn = x4 @ Gc                                                     # (NB,) dL/d(consensus5)*dconsensus5/dwn_i = Gc.x4_i

    # ---- wn_i = w2_i / T  (ratio normalisation, T = sum w2) --------------
    T = float(np.sum(w2)) + 1e-8
    dot_val = float(np.sum(dL_dwn * wn))
    dL_dw2 = (dL_dwn - dot_val) / T                                      # (NB,)

    # ---- w2_i = gate1_i * gate2_i -----------------------------------------
    dL_dgate1 = dL_dw2 * gate2
    dL_dgate2 = dL_dw2 * gate1

    # ---- add the direct "honest diagnosis" supervision gradient ----------
    dLg1_direct = LAMBDA_GATE * (2.0 / NB) * (gate1 - true_worth)
    dLg2_direct = LAMBDA_GATE * (2.0 / NB) * (gate2 - true_worth)
    dL_dgate1 = dL_dgate1 + dLg1_direct
    dL_dgate2 = dL_dgate2 + dLg2_direct

    # ---- Love valley gate: gate1_i = sigmoid(-K1*(pm_after_i - theta_love))
    dpm_after = dL_dgate1 * gate1 * (1.0 - gate1) * (-K1)                # (NB,)
    grads["theta_love"] += float(np.sum(dL_dgate1 * gate1 * (1.0 - gate1) * K1))

    # ---- Detachment valley gate: gate2_i = sigmoid(-K2*(resid_i - eps_thresh))
    #      resid_i = |diag3_i| * (1-alpha)     (residual attachment after projection)
    dresid = dL_dgate2 * gate2 * (1.0 - gate2) * (-K2)                    # (NB,)
    grads["eps_thresh"] += float(np.sum(dL_dgate2 * gate2 * (1.0 - gate2) * K2))
    d_alpha_via_resid = float(np.sum(dresid * (-np.abs(diag3))))
    d_diag3_via_resid = dresid * np.sign(diag3) * (1.0 - alpha)           # (NB,)

    # ---- Detachment (Istighna) backward: x4_i[d]=x3_i[d] (d!=i) ---------
    #      x4_i[i] = x3_i[i]*(1-alpha)
    dL_dx3 = dL_dx4.copy()
    diag_grad4 = dL_dx4[idx, idx]
    dL_dx3[idx, idx] = diag_grad4 * (1.0 - alpha) + d_diag3_via_resid
    dalpha = float(np.sum(diag_grad4 * (-diag3))) + d_alpha_via_resid
    grads["alpha_raw"] += dalpha * alpha * (1.0 - alpha)

    # ---- Knowledge (Ma'rifat) backward -----------------------------------
    dL_dx2 = dL_dx3.copy()
    d_read = cache["eta_m"] * dL_dx3
    grads["eta_m"] += float(np.sum(dL_dx3 * cache["read"]))
    d_attn = d_read @ M.T
    grads["M"] += attn.T @ d_read
    d_scores = np.zeros_like(scores)
    for i in range(NB):
        a = attn[i]
        d_scores[i] = a * (d_attn[i] - np.dot(a, d_attn[i]))
    grads["M"] += (d_scores.T @ x2) / np.sqrt(D)
    dL_dx2 += (d_scores @ M) / np.sqrt(D)

    # ---- Love (Ishq) backward: x2_i[i] = x1_i[i]*(1+rho), rho fixed -------
    dL_dx1 = dL_dx2.copy()
    diag_grad2 = dL_dx2[idx, idx]
    dL_dx1[idx, idx] = diag_grad2 * (1.0 + rho)
    # pm_after_i = x1_i[i]*(1+rho) also feeds the gate directly (dpm_after)
    dL_dx1[idx, idx] += dpm_after * (1.0 + rho)

    # ---- Quest (Talab) backward --------------------------------------------
    dL_dtalab_pre = dL_dx1 * (1.0 - cache["x1"] ** 2)
    grads["W_talab"] += x0.T @ dL_dtalab_pre
    grads["hoopoe_vec"] += dL_dtalab_pre.sum(axis=0)
    dL_dh = dL_dtalab_pre.sum(axis=0)

    # ---- Fikrat backward (direct path via h, plus auxiliary defl. loss) ---
    dL_ddeflect_direct = -(dL_dh @ realm_embeds.T)
    dL_ddeflect_aux = LAMBDA_DEFLECT * (2.0 / R) * (deflect - TARGET_DEFLECT)
    dL_ddeflect = dL_ddeflect_direct + dL_ddeflect_aux
    dz_r = dL_ddeflect * deflect * (1.0 - deflect)
    grads["W_fikrat"] += np.outer(dz_r, x_query)
    grads["b_fikrat"] += dz_r

    loss_parts = dict(ce=ce, cos_loss=cos_loss, defl_aux=defl_aux, gate_loss=gate_loss)
    return loss_total, loss_parts, grads


# ------------------------------------------------------------------------
# 6. SYNTHETIC DATA -- the "Simurgh" the population must recover
# ------------------------------------------------------------------------

def make_concepts(rng):
    raw = rng.normal(0, 1.0, size=(K, SIGNAL_DIM))
    raw = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    concepts = np.zeros((K, D))
    concepts[:, NB:] = raw
    return concepts


def sample_population(rng, concepts, bias_lo=0.5, bias_hi=3.0, noise_sigma=0.05):
    label = rng.integers(0, K)
    s_true = concepts[label].copy()
    x0 = np.tile(s_true[None, :], (NB, 1)).astype(np.float64)
    true_bias = np.zeros(NB)
    for i in range(NB):
        mag = rng.uniform(bias_lo, bias_hi) * rng.choice([-1.0, 1.0])
        x0[i, i] += mag
        true_bias[i] = abs(mag)
    x0 += rng.normal(0, noise_sigma, size=x0.shape)
    return x0, s_true, label, true_bias


# ------------------------------------------------------------------------
# 7. ADAM OPTIMISER, FROM SCRATCH
# ------------------------------------------------------------------------

class Adam:
    def __init__(self, params, lr=0.01, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for k in params:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (grads[k] ** 2)
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            params[k] = params[k] - self.lr * mhat / (np.sqrt(vhat) + self.eps)


def batch_loss_and_grads(params, consts, batch, rng):
    total_loss = 0.0
    total_parts = dict(ce=0.0, cos_loss=0.0, defl_aux=0.0, gate_loss=0.0)
    grads_sum = {k: np.zeros_like(v) for k, v in params.items()}
    for (x0, s_true, label, true_bias) in batch:
        yhat, logits, probs, cache = forward_one(params, consts, x0, rng, train=True)
        loss, parts, grads = backward_one(params, consts, cache, s_true, label, true_bias)
        total_loss += loss
        for kk in total_parts:
            total_parts[kk] += parts[kk]
        for kk in grads_sum:
            grads_sum[kk] += grads[kk]
    n = len(batch)
    total_loss /= n
    for kk in total_parts:
        total_parts[kk] /= n
    for kk in grads_sum:
        grads_sum[kk] /= n
    return total_loss, total_parts, grads_sum


# ------------------------------------------------------------------------
# 8. FINITE-DIFFERENCE GRADIENT CHECK  (mandatory)
# ------------------------------------------------------------------------

def finite_difference_check(params, consts, batch, seed=0, eps=1e-5, n_entries=4, tol=3e-2):
    """
    Compares the analytic gradient (averaged over `batch`) against a
    central finite-difference estimate, for a handful of entries in
    every parameter tensor. train=False and a fixed RNG for the check's
    forward passes so that the Bewilderment valley's stochastic noise
    does not itself register as a (correctly) non-smooth term -- the
    check is run with hayrat_gamma effectively silenced via train=False,
    which matches ordinary practice of checking gradients of the
    deterministic computation graph.
    """
    rng = np.random.default_rng(seed)

    def total_loss(p):
        s = 0.0
        for (x0, s_true, label, true_bias) in batch:
            yhat, logits, probs, cache = forward_one(p, consts, x0, rng, train=False)
            loss, parts, _ = backward_one(p, consts, cache, s_true, label, true_bias)
            s += loss
        return s / len(batch)

    _, _, grads = batch_loss_and_grads_deterministic(params, consts, batch, rng)

    report = []
    max_rel_err = 0.0
    for key in params:
        arr = params[key]
        flat = arr.reshape(-1)
        g_flat = grads[key].reshape(-1)
        idxs = np.linspace(0, flat.size - 1, num=min(n_entries, flat.size)).astype(int)
        for idx in idxs:
            orig = flat[idx]
            flat[idx] = orig + eps
            lp = total_loss(params)
            flat[idx] = orig - eps
            lm = total_loss(params)
            flat[idx] = orig
            numeric = (lp - lm) / (2 * eps)
            analytic = g_flat[idx]
            denom = max(abs(numeric), abs(analytic), 1e-8)
            rel_err = abs(numeric - analytic) / denom
            max_rel_err = max(max_rel_err, rel_err)
            report.append((key, int(idx), analytic, numeric, rel_err))
    passed = max_rel_err < tol
    return passed, max_rel_err, report


def batch_loss_and_grads_deterministic(params, consts, batch, rng):
    """Same as batch_loss_and_grads but forces train=False (no Hayrat noise)
    so the computation graph is smooth for the finite-difference check."""
    total_loss = 0.0
    grads_sum = {k: np.zeros_like(v) for k, v in params.items()}
    for (x0, s_true, label, true_bias) in batch:
        yhat, logits, probs, cache = forward_one(params, consts, x0, rng, train=False)
        loss, parts, grads = backward_one(params, consts, cache, s_true, label, true_bias)
        total_loss += loss
        for kk in grads_sum:
            grads_sum[kk] += grads[kk]
    n = len(batch)
    return total_loss / n, None, {k: v / n for k, v in grads_sum.items()}


# ------------------------------------------------------------------------
# 9. TRAINING LOOP
# ------------------------------------------------------------------------

def train(epochs=60, batch_size=32, batches_per_epoch=20, lr=0.02, seed=RNG_SEED, verbose=True):
    rng = np.random.default_rng(seed)
    params, consts = init_params(rng)
    concepts = make_concepts(rng)
    opt = Adam(params, lr=lr)

    history = []
    baseline_acc = evaluate_accuracy(params, consts, concepts, rng, n=200)
    baseline_cos = evaluate_cosine(params, consts, concepts, rng, n=200)
    history.append(dict(epoch=-1, loss=float("nan"), acc=baseline_acc, cos=baseline_cos))
    if verbose:
        print(f"  epoch  -1  (pre-training baseline)      "
              f"val_acc={baseline_acc:.3f}  val_cos_sim={baseline_cos:.3f}")

    for epoch in range(epochs):
        epoch_loss = 0.0
        for _ in range(batches_per_epoch):
            batch = [sample_population(rng, concepts) for _ in range(batch_size)]
            loss, parts, grads = batch_loss_and_grads(params, consts, batch, rng)
            opt.step(params, grads)
            epoch_loss += loss
        epoch_loss /= batches_per_epoch
        acc = evaluate_accuracy(params, consts, concepts, rng, n=200)
        cos = evaluate_cosine(params, consts, concepts, rng, n=200)
        history.append(dict(epoch=epoch, loss=epoch_loss, acc=acc, cos=cos))
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"  epoch {epoch:3d}  loss={epoch_loss:.4f}  "
                  f"val_acc={acc:.3f}  val_cos_sim={cos:.3f}")
    return params, consts, concepts, history


def evaluate_accuracy(params, consts, concepts, rng, n=200):
    correct = 0
    for _ in range(n):
        x0, s_true, label, true_bias = sample_population(rng, concepts)
        yhat, logits, probs, cache = forward_one(params, consts, x0, rng, train=False)
        if int(np.argmax(probs)) == label:
            correct += 1
    return correct / n


def evaluate_cosine(params, consts, concepts, rng, n=200):
    sims = []
    for _ in range(n):
        x0, s_true, label, true_bias = sample_population(rng, concepts)
        yhat, logits, probs, cache = forward_one(params, consts, x0, rng, train=False)
        sims.append(np.dot(yhat, s_true) / (safe_norm(yhat) * safe_norm(s_true)))
    return float(np.mean(sims))


# ------------------------------------------------------------------------
# 10. SELF-TESTS
# ------------------------------------------------------------------------

def test_shapes():
    rng = np.random.default_rng(1)
    params, consts = init_params(rng)
    concepts = make_concepts(rng)
    x0, s_true, label, true_bias = sample_population(rng, concepts)
    yhat, logits, probs, cache = forward_one(params, consts, x0, rng, train=True)
    assert yhat.shape == (D,)
    assert logits.shape == (K,)
    assert abs(probs.sum() - 1.0) < 1e-6
    print("  [PASS] test_shapes")


def test_gradient_check():
    rng = np.random.default_rng(2)
    params, consts = init_params(rng)
    concepts = make_concepts(rng)
    batch = [sample_population(rng, concepts) for _ in range(6)]
    passed, max_rel_err, report = finite_difference_check(params, consts, batch, seed=3)
    print(f"  [{'PASS' if passed else 'FAIL'}] test_gradient_check "
          f"(max relative error = {max_rel_err:.5f})")
    if not passed:
        worst = sorted(report, key=lambda r: -r[4])[:5]
        for key, idx, a, n, e in worst:
            print(f"      {key}[{idx}]: analytic={a:.6f} numeric={n:.6f} rel_err={e:.5f}")
    assert passed
    return max_rel_err


def test_attachment_reduces_survival():
    """Behavioural test: after training (the 'honest diagnosis' objective
    is what teaches this, not random initialisation), an agent whose true
    bias magnitude is much larger than its peers' should end the forward
    pass with a materially lower combined survival weight (w2) than an
    agent with near-zero bias -- i.e. the Love and Detachment valleys have
    learned to judge each named bird by the magnitude of its own named
    attachment, exactly as the poem insists each valley does."""
    params, consts, concepts, history = train(epochs=30, batch_size=24,
                                               batches_per_epoch=12, lr=0.03,
                                               seed=7, verbose=False)
    rng = np.random.default_rng(4)
    s_true = concepts[0]
    x0 = np.tile(s_true[None, :], (NB, 1)).astype(np.float64)
    x0[0, 0] += 0.05     # bird 0: almost no attachment
    x0[1, 1] += 6.0       # bird 1: overwhelming attachment
    yhat, logits, probs, cache = forward_one(params, consts, x0, rng, train=False)
    w2 = cache["w2"]
    passed = w2[0] > w2[1]
    print(f"  [{'PASS' if passed else 'FAIL'}] test_attachment_reduces_survival "
          f"(low-attachment w2={w2[0]:.4f} vs high-attachment w2={w2[1]:.4f})")
    assert passed


def test_deflection_learns_to_fail_outward():
    """After training, Fikrat's outward query should have converged toward
    the target deflection (searching the whole furnished cosmos and being
    told 'not here' at (almost) every level), leaving only a thin residual
    to leak into the population's Quest valley."""
    rng = np.random.default_rng(5)
    params, consts, concepts, history = train(epochs=25, batch_size=24,
                                               batches_per_epoch=12, lr=0.03,
                                               seed=99, verbose=False)
    x0, s_true, label, true_bias = sample_population(rng, concepts)
    yhat, logits, probs, cache = forward_one(params, consts, x0, rng, train=False)
    mean_deflect = float(np.mean(cache["deflect"]))
    passed = mean_deflect > 0.7
    print(f"  [{'PASS' if passed else 'FAIL'}] test_deflection_learns_to_fail_outward "
          f"(mean deflection={mean_deflect:.3f}, per realm: " +
          ", ".join(f"{n}={d:.2f}" for n, d in zip(REALM_NAMES, cache["deflect"])) + ")")
    assert passed


def test_training_improves_learning():
    rng = np.random.default_rng(6)
    params, consts, concepts, history = train(epochs=40, batch_size=24,
                                               batches_per_epoch=15, lr=0.03,
                                               seed=1234, verbose=False)
    baseline_acc = history[0]["acc"]              # pre-training, ~chance level
    first_epoch_loss = history[1]["loss"]
    last_acc, last_loss = history[-1]["acc"], history[-1]["loss"]
    passed = (last_acc > baseline_acc) and (last_loss < first_epoch_loss) and (last_acc > 1.0 / K)
    print(f"  [{'PASS' if passed else 'FAIL'}] test_training_improves_learning "
          f"(acc {baseline_acc:.3f} [pre-training] -> {last_acc:.3f} [epoch {epochs_str(history)}], "
          f"loss {first_epoch_loss:.4f} -> {last_loss:.4f}, chance level = {1.0/K:.3f})")
    assert passed
    return history


def epochs_str(history):
    return history[-1]["epoch"]


def run_self_tests():
    print("Running self-tests for the Valley-Attrition Network...")
    test_shapes()
    max_rel_err = test_gradient_check()
    test_attachment_reduces_survival()
    test_deflection_learns_to_fail_outward()
    history = test_training_improves_learning()
    print("All self-tests passed.\n")
    return max_rel_err, history


# ------------------------------------------------------------------------
# 11. MAIN DEMONSTRATION
# ------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 78)
    print("CHAPTER 252 -- ATTAR OF NISHAPUR -- VALLEY-ATTRITION NETWORK")
    print("=" * 78)

    max_rel_err, _ = run_self_tests()

    print("Full training run (100 epochs)")
    print("-" * 78)
    params, consts, concepts, history = train(epochs=100, batch_size=32,
                                               batches_per_epoch=20, lr=0.02,
                                               seed=RNG_SEED, verbose=True)

    rng = np.random.default_rng(RNG_SEED + 1)
    final_acc = evaluate_accuracy(params, consts, concepts, rng, n=1000)
    final_cos = evaluate_cosine(params, consts, concepts, rng, n=1000)

    print("-" * 78)
    print(f"Final held-out accuracy over 1000 fresh populations : {final_acc:.4f}")
    print(f"Final held-out mean cosine similarity to true Simurgh: {final_cos:.4f}")
    print(f"Gradient check maximum relative error                : {max_rel_err:.6f}")

    print("-" * 78)
    print("A single worked population, narrated:")
    x0, s_true, label, true_bias = sample_population(rng, concepts)
    yhat, logits, probs, cache = forward_one(params, consts, x0, rng, train=False)
    print(f"  True concept index: {label}   Predicted: {int(np.argmax(probs))}   "
          f"P(correct)={probs[label]:.3f}")
    print(f"  Mean Fikrat deflection (outward search failing, as it should): "
          f"{float(np.mean(cache['deflect'])):.3f}")
    print("  Per-bird outcome (w1=survived Love, w2=survived Love+Detachment):")
    for i, b in enumerate(BIRDS):
        print(f"    {b['name']:>11s} (attachment: {b['attachment']:<38s}) "
              f"true_bias={true_bias[i]:.2f}  w1={cache['w1'][i]:.3f}  w2={cache['w2'][i]:.3f}")
    print(f"  Final cosine(yhat, true Simurgh) = "
          f"{float(np.dot(yhat, s_true) / (safe_norm(yhat) * safe_norm(s_true))):.4f}")
    print("=" * 78)
