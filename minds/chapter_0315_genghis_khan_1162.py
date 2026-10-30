"""
0315_Neuron.py
==============
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0315 · Genghis Khan / Chinggis Khan (Temuejin), b. c.1162, Khentii, Mongolia -- d. August 1227
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 16, Minds 301-320 Available on Amazon https://www.amazon.com/gp/product/B0HNJQX9S4
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos 
Architecture name (informal, used only in comments): THE ARBAN LATTICE
    "Orlog"  -- Orlog is the Mongolian word for a field marshal / great general; here it names
    the whole trainable system: a small confederation of independently-initialised specialist
    networks ("noyad", commanders/specialists) that are routed to, tested under load, bound
    together by a fast shared broadcast signal, and periodically reorganised on the basis of
    demonstrated performance rather than seniority.

--------------------------------------------------------------------------------------------
WHY THIS SHAPE OF NETWORK, AND NOT A TRANSFORMER
--------------------------------------------------------------------------------------------
Genghis Khan left no philosophy of mind in his own hand -- he was not literate, and everything
attributed to him survives second-hand (see Sources at the foot of this file and the companion
chapter). But the empire he actually built is, itself, a fully specified cognitive architecture,
documented in enough independent, cross-checked detail (Mongol, Chinese, Persian, Latin sources)
that we can transcribe its literal design into code without inventing anything. Four structural
facts, all independently attested, define the system implemented below:

  1. DECIMAL RE-WIRING OF LOYALTY (arban/jagun/mingghan/tuemen). Temuejin dissolved the old
     clan-and-lineage chain of command and rebuilt the army (and, through it, the state) as
     nested units of ten, with promotion and demotion attached to units, not to birth. A
     commander who failed could be broken down and folded into another ten; a fighter who
     distinguished himself could rise to command one. This is implemented below as a small
     two-level hierarchy of "expert" sub-networks organised into GROUPS ("arban"), with an
     explicit, non-gradient REORGANISATION operator that runs between training epochs.

  2. RECRUITMENT AND ROUTING OF CAPTURED SPECIALISTS, NOT A SINGLE MIND. The historically
     documented governing apparatus of the empire was never "Genghis Khan's genius" acting
     alone -- it was Uighur scribes (Tatatonga), a Khitan statesman-astronomer (Yelue Chucai),
     Muslim tax-farmers and merchant-envoys, Chinese and Persian siege engineers, all recruited
     from conquered populations, tested on real tasks, and routed to the problems they were
     best suited for. This is implemented as a MIXTURE OF EXPERTS with an input-conditioned
     gate: the "mind" is the routing-and-integration layer, not any one expert.

  3. FITNESS PROVEN UNDER LOAD, NOT ASSUMED AT ENTRY ("meritocracy" in the strict, checkable
     sense -- promotion tracks a running record of results, and it is revisable). Implemented
     as a slowly-updated, non-trainable FITNESS score per expert, fed back as an additive bias
     on the routing logits: a specialist's history of being right when trusted shifts how much
     it is trusted next time, on top of whatever the input itself suggests.

  4. THE YAM -- A DEDICATED, LOW-LATENCY RELAY NETWORK binding a continental empire together.
     The Mongol Empire ran a horse-relay postal and courier network (the "yam") precisely so
     that a decision made at the centre could reach the frontier fast enough to matter, and so
     that field commanders could act on shared, current information rather than stale local
     assumptions. Implemented as a BROADCAST BUS: a slowly-updated vector, computed from the
     hidden state of whichever expert is currently most trusted, that is injected additively
     into every expert's hidden layer on the next step -- local commanders' units are still
     directed by, but not fused into, the shared signal.

A fifth mechanism -- LOYALTY REGULARISATION -- has no single one-line steppe analogue; it is
this file's synthesis of a pattern visible across (2)-(4): units that are *not* currently the
best choice for a given problem are still required to stay broadly aligned with the shared
directive (the broadcast), on pain of a penalty, exactly as reserve units and provincial
governors were expected to stay "on-doctrine" even when not on campaign, so they could be
folded back into active service without retraining from zero. This is the file's one genuinely
invented piece, and the chapter text says so plainly -- it is not attributed to Temuejin himself.

None of this is a Transformer, and it is not attention over a fixed set of stored keys. There
is no self-attention anywhere in this file. The "attention" analogue here is *routing under
tested, revisable trust*, not *content-addressed lookup*, which is a real and different
computational commitment: it is Chinggisid because trust is a runtime, earned, group-anchored
quantity, not a static learned similarity between the current input and a memory buffer.

--------------------------------------------------------------------------------------------
FILE CONTENTS
--------------------------------------------------------------------------------------------
  Part 1  -- Parameter container + initialisation (heterogeneous, "captured from many peoples")
  Part 2  -- Forward pass (fully vectorised math is documented inline)
  Part 3  -- Manual backward pass (exact analytic gradients, no autodiff)
  Part 4  -- Finite-difference gradient check (mandatory; must pass before training)
  Part 5  -- Non-gradient reorganisation operator (the "decimal" meritocratic pruning step)
  Part 6  -- Synthetic "campaign" data generator (four sequential, non-stationary provinces)
  Part 7  -- Training loop + ablation control run (loyalty/broadcast ON vs OFF)
  Part 8  -- Self-tests and __main__ entry point (prints the verified run to stdout)

Pure NumPy. No autograd, no deep-learning framework. Every gradient below is hand-derived.
--------------------------------------------------------------------------------------------
"""

import numpy as np

RNG_SEED = 315  # figure id, for reproducibility -- not a design claim, just a fixed seed


# ==============================================================================================
# PART 1 -- PARAMETERS
# ==============================================================================================

class OrlogConfig:
    """Fixed sizes for the whole system. Kept small so the file trains in seconds on a laptop
    CPU and the gradient check stays cheap, not because the mechanism only works at this scale."""

    def __init__(self):
        self.input_dim = 6          # synthetic "situation" feature vector
        self.hidden_dim = 16        # each specialist's internal working space
        self.num_classes = 4        # four possible correct "decisions" per situation
        self.num_experts = 9        # noyad: individual specialists ("captured expertise")
        self.group_size = 3         # arban-scale grouping: 3 groups of 3 experts
        self.num_groups = self.num_experts // self.group_size
        self.fitness_bias_weight = 2.0   # how strongly proven track record shifts routing
        self.fitness_decay = 0.90        # EMA decay for the fitness score
        self.broadcast_decay = 0.80      # EMA decay for the yam relay broadcast vectors
        self.global_gamma = 0.30         # how much the tuemen-level (global) signal leaks
                                          # into each arban's local broadcast
        self.loyalty_lambda = 0.15       # strength of the loyalty regulariser
        self.learning_rate = 0.01
        self.max_grad_norm = 3.0         # global-norm gradient clipping (feedback loops via
                                          # the broadcast bus can otherwise blow up quickly)
        self.reorg_every = 40            # steps between decimal-reorganisation events
        self.reorg_weak_ratio = 0.55     # an expert scoring below this fraction of its
                                          # group's median fitness is a reorganisation candidate
        self.reorg_collapse_ratio = 0.5  # below THIS fraction of the *weak* threshold ->
                                          # treated as an irrecoverable failure -> full reinit
                                          # instead of clone-with-noise
        self.reorg_min_median = 0.12     # a group whose median fitness hasn't cleared this
                                          # floor yet is still "raw recruits, campaign one" --
                                          # every member is still near-zero and the ranking is
                                          # noise, so reorganisation is skipped until the group
                                          # has actually demonstrated something worth comparing


def group_of(expert_idx: int, cfg: OrlogConfig) -> int:
    return expert_idx // cfg.group_size


def init_expert(rng: np.random.Generator, cfg: OrlogConfig, origin_scale: float):
    """One specialist's parameters. `origin_scale` deliberately varies per-expert at init,
    standing in for the historical fact that the recruited specialists did not arrive as
    blank, identically-distributed units -- a Chinese siege engineer, a Uighur scribe and a
    Muslim tax-farmer walk in with different priors. This is flavour, not load-bearing: the
    gradient check and training loop must, and do, work regardless of this variation."""
    d, h, c = cfg.input_dim, cfg.hidden_dim, cfg.num_classes
    W1 = rng.normal(0, origin_scale / np.sqrt(d), size=(d, h))
    b1 = np.zeros(h)
    W2 = rng.normal(0, origin_scale / np.sqrt(h), size=(h, c))
    b2 = np.zeros(c)
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


def init_orlog(cfg: OrlogConfig, seed: int = RNG_SEED):
    rng = np.random.default_rng(seed)
    experts = []
    # heterogeneous init scales, one per "origin culture" -- purely cosmetic diversity
    origin_scales = [0.7, 1.3, 1.0, 0.9, 1.6, 1.1, 0.8, 1.4, 1.0]
    for e in range(cfg.num_experts):
        experts.append(init_expert(rng, cfg, origin_scales[e % len(origin_scales)]))

    Wg = rng.normal(0, 1.0 / np.sqrt(cfg.input_dim), size=(cfg.input_dim, cfg.num_experts))
    bg = np.zeros(cfg.num_experts)

    state = {
        "experts": experts,
        "Wg": Wg,
        "bg": bg,
        # non-trainable running state (updated by EMA outside of gradient descent):
        "fitness": np.zeros(cfg.num_experts),
        "local_broadcast": np.zeros((cfg.num_groups, cfg.hidden_dim)),
        "global_broadcast": np.zeros(cfg.hidden_dim),
        "reorg_log": [],
    }
    return state


# ==============================================================================================
# PART 2 -- FORWARD PASS
# ==============================================================================================

def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def relu(z):
    return np.maximum(0, z)


def forward(state, cfg: OrlogConfig, x, use_broadcast=True, use_fitness=True):
    """
    x: (B, input_dim)
    Returns a `cache` dict holding everything backward() needs, plus final_logits (B, C).

    Math (per expert e, group g = group_of(e)):
        broadcast_e   = local_broadcast[g] + global_gamma * global_broadcast     (constant, no grad)
        pre1_e        = x @ W1_e + b1_e + broadcast_e                            (B, h)
        a1_e          = relu(pre1_e)                                             (B, h)
        out_e         = a1_e @ W2_e + b2_e                                       (B, C)

        gate_logits   = x @ Wg + bg + fitness_bias_weight * fitness              (B, E)
        gate          = softmax(gate_logits, axis=1)                             (B, E)

        final_logits  = sum_e gate[:, e:e+1] * out_e                             (B, C)
    """
    B = x.shape[0]
    E = cfg.num_experts

    broadcasts = []
    pre1_list, a1_list, out_list = [], [], []
    for e in range(E):
        g = group_of(e, cfg)
        if use_broadcast:
            b_vec = state["local_broadcast"][g] + cfg.global_gamma * state["global_broadcast"]
        else:
            b_vec = np.zeros(cfg.hidden_dim)
        broadcasts.append(b_vec)
        p = state["experts"][e]
        pre1 = x @ p["W1"] + p["b1"] + b_vec
        a1 = relu(pre1)
        out = a1 @ p["W2"] + p["b2"]
        pre1_list.append(pre1)
        a1_list.append(a1)
        out_list.append(out)

    fitness_term = cfg.fitness_bias_weight * state["fitness"] if use_fitness else 0.0
    gate_logits = x @ state["Wg"] + state["bg"] + fitness_term
    gate = softmax(gate_logits)  # (B, E)

    final_logits = np.zeros((B, cfg.num_classes))
    for e in range(E):
        final_logits += gate[:, e:e + 1] * out_list[e]

    cache = {
        "x": x, "gate": gate, "gate_logits": gate_logits,
        "pre1": pre1_list, "a1": a1_list, "out": out_list, "broadcasts": broadcasts,
        "final_logits": final_logits,
    }
    return final_logits, cache


def loss_fn(cache, y, cfg: OrlogConfig):
    """Cross-entropy on final_logits + the loyalty regulariser. Returns scalar loss and the
    upstream gradient pieces backward() needs (dlogits, and the pieces of dL/dgate, dL/da1
    contributed by the loyalty term alone -- the CE contributions are computed in backward())."""
    B = cache["x"].shape[0]
    probs = softmax(cache["final_logits"])
    ce = -np.log(np.clip(probs[np.arange(B), y], 1e-12, 1.0)).mean()

    E = cfg.num_experts
    gate = cache["gate"]
    loyalty = 0.0
    d_loyalty_d_gate = np.zeros_like(gate)          # (B, E)
    d_loyalty_d_a1 = [np.zeros_like(a) for a in cache["a1"]]
    h = cfg.hidden_dim
    for e in range(E):
        diff = cache["a1"][e] - cache["broadcasts"][e][None, :]   # (B, h)
        per_example_mse = (diff ** 2).mean(axis=1)                # (B,)
        weight = (1.0 - gate[:, e])                                # (B,)
        loyalty += (weight * per_example_mse).mean()
        # loyalty (pre-lambda) = sum_e (1/B) sum_b weight[b,e] * mean_h(diff^2)
        # d loyalty_pre / d gate[b,e]   = -(1/B) * per_example_mse[b]             (lambda applied below)
        # d loyalty_pre / d a1[e][b,h'] =  (1/B) * weight[b] * (2/h) * diff[b,h'] (lambda applied below)
        d_loyalty_d_gate[:, e] += -(1.0 / B) * per_example_mse
        d_loyalty_d_a1[e] += (1.0 / B) * weight[:, None] * (2.0 / h) * diff

    loyalty *= cfg.loyalty_lambda
    d_loyalty_d_gate *= cfg.loyalty_lambda
    d_loyalty_d_a1 = [cfg.loyalty_lambda * d for d in d_loyalty_d_a1]
    total = ce + loyalty
    probs_minus_onehot = probs.copy()
    probs_minus_onehot[np.arange(B), y] -= 1.0
    dlogits = probs_minus_onehot / B   # dL_ce/d(final_logits)

    return total, ce, loyalty, dlogits, d_loyalty_d_gate, d_loyalty_d_a1


# ==============================================================================================
# PART 3 -- BACKWARD PASS (exact, hand-derived)
# ==============================================================================================

def backward(state, cfg: OrlogConfig, cache, dlogits, d_loyalty_d_gate, d_loyalty_d_a1):
    E = cfg.num_experts
    x = cache["x"]
    gate = cache["gate"]

    grads = {"experts": [dict() for _ in range(E)], "Wg": None, "bg": None}

    d_gate_total = d_loyalty_d_gate.copy()  # (B, E), CE contribution added below per-expert
    d_a1_total = [d.copy() for d in d_loyalty_d_a1]

    for e in range(E):
        out_e = cache["out"][e]
        a1_e = cache["a1"][e]
        pre1_e = cache["pre1"][e]
        p = state["experts"][e]

        # --- CE contribution through the mixture ---
        d_out_e = gate[:, e:e + 1] * dlogits                       # (B, C)
        d_gate_total[:, e] += np.einsum("bc,bc->b", dlogits, out_e)  # dL_ce/dgate[:,e]

        # --- expert e's own backward: out = a1 @ W2 + b2 ---
        grads["experts"][e]["W2"] = a1_e.T @ d_out_e
        grads["experts"][e]["b2"] = d_out_e.sum(axis=0)
        d_a1_total[e] += d_out_e @ p["W2"].T

        # --- relu, then linear input layer (broadcast is a constant, no grad flows there) ---
        d_pre1 = d_a1_total[e] * (pre1_e > 0)
        grads["experts"][e]["W1"] = x.T @ d_pre1
        grads["experts"][e]["b1"] = d_pre1.sum(axis=0)

    # --- gate softmax backward: standard softmax-Jacobian-vector product, per example ---
    # dL/dlogits[b,e] = gate[b,e] * ( dL/dgate[b,e] - sum_j gate[b,j] * dL/dgate[b,j] )
    weighted_sum = np.einsum("be,be->b", gate, d_gate_total)        # (B,)
    d_gate_logits = gate * (d_gate_total - weighted_sum[:, None])   # (B, E)

    grads["Wg"] = x.T @ d_gate_logits
    grads["bg"] = d_gate_logits.sum(axis=0)

    return grads


def apply_grads(state, grads, cfg: OrlogConfig):
    lr = cfg.learning_rate

    # global-norm gradient clipping: the broadcast bus feeds last-step activations back into
    # this step's forward pass, which (like any recurrent signal) can amplify a few bad steps
    # into a runaway update if left unclipped. This does not change what backward() computed
    # (the gradient check above runs before any clipping is applied) -- it only bounds the
    # step actually taken, exactly as any responsible optimiser does for a recurrent system.
    sq = 0.0
    for e in range(cfg.num_experts):
        for k in ("W1", "b1", "W2", "b2"):
            sq += np.sum(grads["experts"][e][k] ** 2)
    sq += np.sum(grads["Wg"] ** 2) + np.sum(grads["bg"] ** 2)
    norm = np.sqrt(sq)
    scale = 1.0
    if norm > cfg.max_grad_norm:
        scale = cfg.max_grad_norm / (norm + 1e-8)

    for e in range(cfg.num_experts):
        for k in ("W1", "b1", "W2", "b2"):
            state["experts"][e][k] -= lr * scale * grads["experts"][e][k]
    state["Wg"] -= lr * scale * grads["Wg"]
    state["bg"] -= lr * scale * grads["bg"]


def train_step(state, cfg: OrlogConfig, x, y, use_broadcast=True, use_fitness=True):
    logits, cache = forward(state, cfg, x, use_broadcast=use_broadcast, use_fitness=use_fitness)
    total, ce, loyalty, dlogits, d_loy_gate, d_loy_a1 = loss_fn(cache, y, cfg)
    grads = backward(state, cfg, cache, dlogits, d_loy_gate, d_loy_a1)
    apply_grads(state, grads, cfg)
    update_running_state(state, cfg, cache, y, use_broadcast=use_broadcast, use_fitness=use_fitness)
    return total, ce, loyalty, cache


def update_running_state(state, cfg: OrlogConfig, cache, y, use_broadcast=True, use_fitness=True):
    """The two pieces of non-gradient state: fitness (per-expert track record) and the
    yam-relay broadcast vectors (local per-arban, and one global/tuemen-level). Both are
    EMAs, updated AFTER the gradient step, so they are legitimately constant with respect to
    the parameters during the forward/backward pass that just happened (this is what keeps the
    finite-difference gradient check well-posed: nothing here participates in autodiff)."""
    B = cache["x"].shape[0]
    gate = cache["gate"]
    preds_per_expert = [np.argmax(o, axis=1) for o in cache["out"]]

    if use_fitness:
        for e in range(cfg.num_experts):
            correct = (preds_per_expert[e] == y).astype(float)   # (B,)
            contribution = (gate[:, e] * correct).mean()
            state["fitness"][e] = (cfg.fitness_decay * state["fitness"][e]
                                    + (1 - cfg.fitness_decay) * contribution)

    if use_broadcast:
        for g in range(cfg.num_groups):
            members = [e for e in range(cfg.num_experts) if group_of(e, cfg) == g]
            # the currently-most-trusted member of this arban sets the local directive
            gate_sub = gate[:, members]
            top_local = np.array(members)[np.argmax(gate_sub, axis=1)]
            mean_a1 = np.mean([cache["a1"][e_idx][b] for b, e_idx in enumerate(top_local)], axis=0)
            state["local_broadcast"][g] = (cfg.broadcast_decay * state["local_broadcast"][g]
                                            + (1 - cfg.broadcast_decay) * mean_a1)
        state["global_broadcast"] = (cfg.broadcast_decay * state["global_broadcast"]
                                      + (1 - cfg.broadcast_decay) * state["local_broadcast"].mean(axis=0))


# ==============================================================================================
# PART 4 -- FINITE-DIFFERENCE GRADIENT CHECK  (mandatory; run before any training claim is made)
# ==============================================================================================

def flatten_params(state, cfg):
    parts = []
    shapes = []
    for e in range(cfg.num_experts):
        for k in ("W1", "b1", "W2", "b2"):
            arr = state["experts"][e][k]
            parts.append(arr.ravel())
            shapes.append(("expert", e, k, arr.shape))
    parts.append(state["Wg"].ravel()); shapes.append(("Wg", None, None, state["Wg"].shape))
    parts.append(state["bg"].ravel()); shapes.append(("bg", None, None, state["bg"].shape))
    return np.concatenate(parts), shapes


def flatten_grads(grads, cfg, shapes):
    parts = []
    for tag, e, k, shape in shapes:
        if tag == "expert":
            parts.append(grads["experts"][e][k].ravel())
        elif tag == "Wg":
            parts.append(grads["Wg"].ravel())
        elif tag == "bg":
            parts.append(grads["bg"].ravel())
    return np.concatenate(parts)


def set_flat_value(state, cfg, shapes, flat_idx, value):
    """Write `value` into the single scalar parameter at position flat_idx of the flattened
    vector produced by flatten_params. Returns the old value so the caller can restore it."""
    offset = 0
    for tag, e, k, shape in shapes:
        size = int(np.prod(shape))
        if offset <= flat_idx < offset + size:
            local_idx = flat_idx - offset
            if tag == "expert":
                arr = state["experts"][e][k]
            elif tag == "Wg":
                arr = state["Wg"]
            else:
                arr = state["bg"]
            flat_view = arr.reshape(-1)
            old = flat_view[local_idx]
            flat_view[local_idx] = value
            return old
        offset += size
    raise IndexError("flat_idx out of range")


def total_loss_only(state, cfg, x, y):
    logits, cache = forward(state, cfg, x)
    total, *_ = loss_fn(cache, y, cfg)
    return total


def gradient_check(cfg: OrlogConfig, n_checks=24, eps=1e-5, seed=1):
    """Central-difference check on n_checks randomly chosen scalar parameters, spread across
    every parameter tensor in the system (experts' W1/b1/W2/b2, and the gate's Wg/bg). Uses a
    small fixed batch. Non-trainable running state (fitness, broadcasts) is frozen at zero for
    the duration of the check, exactly as it legitimately is during a single train_step call."""
    rng = np.random.default_rng(seed)
    state = init_orlog(cfg, seed=seed)
    B = 5
    x = rng.normal(size=(B, cfg.input_dim))
    y = rng.integers(0, cfg.num_classes, size=B)

    logits, cache = forward(state, cfg, x)
    total, ce, loyalty, dlogits, d_loy_gate, d_loy_a1 = loss_fn(cache, y, cfg)
    grads = backward(state, cfg, cache, dlogits, d_loy_gate, d_loy_a1)

    flat_params, shapes = flatten_params(state, cfg)
    flat_grads = flatten_grads(grads, cfg, shapes)

    n_params = flat_params.shape[0]
    idxs = rng.choice(n_params, size=min(n_checks, n_params), replace=False)

    max_rel_err = 0.0
    worst = None
    for idx in idxs:
        old = set_flat_value(state, cfg, shapes, idx, flat_params[idx] + eps)
        loss_plus = total_loss_only(state, cfg, x, y)
        set_flat_value(state, cfg, shapes, idx, flat_params[idx] - eps)
        loss_minus = total_loss_only(state, cfg, x, y)
        set_flat_value(state, cfg, shapes, idx, old)  # restore exactly

        numeric = (loss_plus - loss_minus) / (2 * eps)
        analytic = flat_grads[idx]
        denom = max(abs(numeric), abs(analytic), 1e-8)
        rel_err = abs(numeric - analytic) / denom
        if rel_err > max_rel_err:
            max_rel_err = rel_err
            worst = (idx, numeric, analytic)

    return max_rel_err, worst


# ==============================================================================================
# PART 5 -- DECIMAL REORGANISATION  (non-gradient, "meritocracy" operator)
# ==============================================================================================

def reorganize(state, cfg: OrlogConfig, step, rng):
    """Runs every cfg.reorg_every steps. For each arban (group), find the member with the
    lowest fitness. If it is only mildly behind its group's median, it is folded into the
    best performer's line -- its weights become a noisy clone of the group's best expert
    (absorbing demonstrated best practice, as a defeated unit's survivors were commonly
    redistributed into a victorious one's command). If it is far behind (a persistent,
    structural failure), it is fully reinitialised from scratch with a fresh random draw,
    the steppe-tradition equivalent of stripping a failed commander of his tuemen outright.
    Either way its fitness score is reset to the group median: no unit is permanently
    penalised for a past failure once it has been reorganised -- this is the concrete,
    checkable form of 'no hereditary caste' in this file."""
    log_entries = []
    for g in range(cfg.num_groups):
        members = [e for e in range(cfg.num_experts) if group_of(e, cfg) == g]
        fits = state["fitness"][members]
        median = np.median(fits)
        worst_local = int(np.argmin(fits))
        worst_e = members[worst_local]
        best_local = int(np.argmax(fits))
        best_e = members[best_local]
        if worst_e == best_e:
            continue  # group of one effectively, nothing to reorganise
        if median < cfg.reorg_min_median:
            continue  # whole group still untested; ranking would be noise, not merit
        if fits[worst_local] < cfg.reorg_weak_ratio * max(median, 1e-6):
            collapsed = fits[worst_local] < cfg.reorg_collapse_ratio * cfg.reorg_weak_ratio * max(median, 1e-6)
            if collapsed:
                d, h, c = cfg.input_dim, cfg.hidden_dim, cfg.num_classes
                state["experts"][worst_e] = init_expert(rng, cfg, origin_scale=1.0)
                action = "full_reinit"
            else:
                src = state["experts"][best_e]
                noisy = {k: v + rng.normal(0, 0.05, size=v.shape) for k, v in src.items()}
                state["experts"][worst_e] = noisy
                action = "clone_from_best"
            state["fitness"][worst_e] = median
            log_entries.append({"step": step, "group": g, "reorganized_expert": worst_e,
                                 "best_expert": best_e, "action": action,
                                 "worst_fitness": float(fits[worst_local]), "group_median": float(median)})
    state["reorg_log"].extend(log_entries)
    return log_entries


# ==============================================================================================
# PART 6 -- SYNTHETIC "CAMPAIGN" DATA  (four sequential, non-stationary provinces)
# ==============================================================================================

_PROVINCE_CACHE = {}


def _province_geometry(cfg, province_id):
    """Class centers and rotation for a given province, derived deterministically from
    province_id alone (a fixed seed per id) and cached -- so 'province 2' means the exact
    same classification task every single time it is sampled. Getting this wrong (redrawing
    random centers on every batch) would silently make the task unlearnable, since the
    x -> y mapping would change from batch to batch; this is exactly the kind of bug the
    self-tests in Part 8 are designed to surface via the held-out accuracy assertions."""
    key = (province_id, cfg.input_dim, cfg.num_classes)
    if key not in _PROVINCE_CACHE:
        geo_rng = np.random.default_rng(9000 + province_id)
        d, c = cfg.input_dim, cfg.num_classes
        centers = geo_rng.normal(0, 2.2, size=(c, d)) + province_id * geo_rng.normal(0, 0.6, size=d)
        theta = province_id * 0.7
        rot = np.eye(d)
        rot[0, 0], rot[0, 1] = np.cos(theta), -np.sin(theta)
        rot[1, 0], rot[1, 1] = np.sin(theta), np.cos(theta)
        _PROVINCE_CACHE[key] = (centers, rot)
    return _PROVINCE_CACHE[key]


def make_province(rng, cfg, province_id, n):
    """Each 'province' is a FIXED 4-class Gaussian-blob classification problem in the model's
    input space, with a province-specific rotation and offset -- i.e. every conquered province
    poses a genuinely different (but same-shaped) decision problem, which is the point: the
    system must integrate genuinely new specialists' worth of skill each time, not just re-use
    one fixed mapping. This is a deliberately small, honest synthetic stand-in for "a new
    domain of governance," not a claim to model real medieval administration. `rng` supplies
    only the per-call label draw and observation noise; the geometry itself is fixed (see
    _province_geometry) so the task is actually learnable across repeated sampling."""
    d, c = cfg.input_dim, cfg.num_classes
    centers, rot = _province_geometry(cfg, province_id)
    y = rng.integers(0, c, size=n)
    x = centers[y] + rng.normal(0, 0.9, size=(n, d))
    x = x @ rot
    return x, y


# ==============================================================================================
# PART 7 -- TRAINING LOOP + ABLATION
# ==============================================================================================

def run_campaign(cfg: OrlogConfig, use_broadcast=True, use_fitness=True, use_loyalty=True,
                  seed=7, steps_per_province=150, batch_size=24, verbose=True):
    rng = np.random.default_rng(seed)
    data_rng = np.random.default_rng(seed + 1000)
    state = init_orlog(cfg, seed=seed)
    lam_saved = cfg.loyalty_lambda
    if not use_loyalty:
        cfg.loyalty_lambda = 0.0

    n_provinces = 4
    held_out = {}
    for pv in range(n_provinces):
        held_out[pv] = make_province(data_rng, cfg, pv, 400)

    history = {"loss": [], "province1_retention": [], "just_learned": [], "loss_at_block_start": []}
    global_step = 0
    for pv in range(n_provinces):
        history["loss_at_block_start"].append(
            np.mean([total_loss_only(state, cfg, *make_province(data_rng, cfg, pv, batch_size))
                      for _ in range(3)]))
        for s in range(steps_per_province):
            xb, yb = make_province(data_rng, cfg, pv, batch_size)
            total, ce, loyalty, cache = train_step(state, cfg, xb, yb,
                                                     use_broadcast=use_broadcast,
                                                     use_fitness=use_fitness)
            history["loss"].append(total)
            global_step += 1
            if global_step % cfg.reorg_every == 0:
                reorganize(state, cfg, global_step, rng)
        # own-province accuracy right after training on it (did the system learn IT at all?)
        acc_own = evaluate_accuracy(state, cfg, *held_out[pv])
        history["just_learned"].append(acc_own)
        # retention of province 0 (the FIRST one learned) as later provinces are absorbed
        acc0 = evaluate_accuracy(state, cfg, *held_out[0])
        history["province1_retention"].append(acc0)
        if verbose:
            print(f"  after province {pv} ({(pv+1)*steps_per_province} steps): "
                  f"train_loss={np.mean(history['loss'][-20:]):.4f}  "
                  f"province-{pv} just-learned acc={acc_own:.3f}  "
                  f"province-0 held-out acc={acc0:.3f}")

    final_accs = [evaluate_accuracy(state, cfg, *held_out[pv]) for pv in range(n_provinces)]
    cfg.loyalty_lambda = lam_saved
    return state, history, final_accs


def evaluate_accuracy(state, cfg, x, y):
    logits, _ = forward(state, cfg, x)
    preds = np.argmax(logits, axis=1)
    return float((preds == y).mean())


# ==============================================================================================
# PART 8 -- SELF-TESTS
# ==============================================================================================

def main():
    cfg = OrlogConfig()

    print("=" * 78)
    print("PART A -- FINITE-DIFFERENCE GRADIENT CHECK")
    print("=" * 78)
    max_rel_err, worst = gradient_check(cfg, n_checks=24, eps=1e-5, seed=1)
    print(f"  checked 24 random scalar parameters across all experts + gate")
    print(f"  max relative error = {max_rel_err:.3e}  (worst param: {worst[0]}, "
          f"numeric={worst[1]:.6f}, analytic={worst[2]:.6f})")
    assert max_rel_err < 1e-3, "GRADIENT CHECK FAILED"
    print("  PASS: analytic backward() matches numerical gradient.\n")

    print("=" * 78)
    print("PART B -- FULL SYSTEM: 4 sequential provinces, mechanism ON")
    print("=" * 78)
    state, history, final_accs = run_campaign(cfg, use_broadcast=True, use_fitness=True,
                                               use_loyalty=True, seed=7)
    print(f"  final held-out accuracy per province: "
          f"{['%.3f' % a for a in final_accs]}  (chance level = {1.0/cfg.num_classes:.3f})")
    print(f"  reorganisation events fired: {len(state['reorg_log'])}")
    for entry in state["reorg_log"][:6]:
        print(f"    step {entry['step']:4d}  group {entry['group']}  "
              f"expert {entry['reorganized_expert']} -> {entry['action']} "
              f"(fitness {entry['worst_fitness']:.3f} vs group median {entry['group_median']:.3f})")
    if len(state["reorg_log"]) > 6:
        print(f"    ... {len(state['reorg_log']) - 6} more reorganisation events")

    print(f"  own-province held-out accuracy, measured right after training on it: "
          f"{['%.3f' % a for a in history['just_learned']]}")
    print(f"  loss at the START of each province's block (task-shift spike, expected): "
          f"{[round(float(l), 2) for l in history['loss_at_block_start']]}")
    steps = cfg.reorg_every  # a stable within-block window near the end of each block
    print(f"  mean loss over the LAST {2*steps} steps of training = "
          f"{np.mean(history['loss'][-2*steps:]):.4f}")
    assert np.mean(history["just_learned"]) > 0.70, \
        "system is not reliably learning each newly-presented province at all"
    assert history["just_learned"][0] > 0.85, \
        "system failed to learn even the very first, uncontested province"
    print("  PASS: every province is learned to well above chance (25%) immediately after "
          "its own training block -- the routing + backward pass genuinely fits new data.\n")

    print("=" * 78)
    print("PART C -- ABLATION: does the yam-broadcast + loyalty term actually help RETENTION?")
    print("=" * 78)
    print("  Run WITH broadcast+loyalty (the full Arban Lattice mechanism):")
    _, hist_on, _ = run_campaign(cfg, use_broadcast=True, use_fitness=True, use_loyalty=True,
                                  seed=7, verbose=False)
    print("  Run WITHOUT broadcast+loyalty (plain soft mixture-of-experts, fitness routing only):")
    _, hist_off, _ = run_campaign(cfg, use_broadcast=False, use_fitness=True, use_loyalty=False,
                                   seed=7, verbose=False)

    retention_on = hist_on["province1_retention"][-1]   # province-0 accuracy after province 3
    retention_off = hist_off["province1_retention"][-1]
    print(f"  province-0 (first-learned) held-out accuracy after all 4 provinces:")
    print(f"    WITH  yam-broadcast + loyalty : {retention_on:.3f}")
    print(f"    WITHOUT (plain soft MoE)      : {retention_off:.3f}")
    print(f"    chance level                  : {1.0/cfg.num_classes:.3f}")
    if retention_on >= retention_off:
        print("  OBSERVATION: the broadcast+loyalty mechanism retained the first province at "
              "least as well as the ablated control on this seed.")
    else:
        print("  OBSERVATION: on this seed the ablated control retained better -- reported "
              "honestly; the mechanism is not asserted to dominate in every run, only tested.")
    print()

    print("=" * 78)
    print("ALL SELF-TESTS COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()
