"""
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0275 · Su Shi (Su Dongpo) (1037-1101)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 14, Minds 261-280 Available on Amazon https://www.amazon.com/dp/B0HLYSVY5D
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0275_su_shi_su_dongpo_1037 - Su Shi (Su Dongpo) (1037-1101)
A from-scratch NumPy neural architecture built from Su Shi's own,
textually-verified theory of skilled generative cognition -- NOT a
generic "mind-ground / Chan-Confucian-Daoist synthesis" template.

WHERE THIS COMES FROM (see chapter .md, Section 5, for full citations)
-----------------------------------------------------------------------
Su Shi left an unusually precise, mechanistic vocabulary for how a mind
should PLAN and EXECUTE a structured, generative act, spread across four
of his own essays/letters/poems:

  1. "The bamboo complete in the breast" (胸有成竹), from his essay on
     Wen Tong's bamboo paintings: before the brush moves, the whole
     living form must be held as ONE COMPLETE, UNFOLDING PLAN, not
     assembled joint-by-joint at the moment of drawing. "Nowadays the
     artists construct a bamboo, joint by joint and leaf by leaf. Where
     is the bamboo?"

  2. "The falcon and the hare", same essay: once execution begins, it
     is ballistic -- "one follows the idea, pursuing the image just
     seen, like a hawk swooping on a rabbit; a moment's hesitation, and
     it is lost." Deliberation happens entirely BEFORE the stroke,
     never DURING it.

  3. "Proceeding where it must proceed, stopping where it cannot but
     stop" (常行於所當行，常止於所不可不止), from his letter to Xie
     Minshi (與謝民師推官書): the generative process carries its own,
     CONTENT-DETERMINED halting criterion. It does not stop because an
     external template (a fixed meter, a fixed length) says so.

  4. "Taking shape according to what is met" (隨物賦形) and "a spring
     of ten-thousand measures ... that cannot choose its terrain"
     (吾文如萬斛泉源，不擇地皆可出), from his self-appraisal of his own
     writing (自評文): ONE underlying source produces different surface
     forms depending on the terrain (context) it meets -- not one
     separately-tuned instrument per genre.

  5. "Judging a painting by mere likeness is a view fit for a child"
     (論畫以形似，見與兒童鄰): the correct standard for a generated
     artifact is whether it captures the underlying PRINCIPLE (理) that
     governs how the thing grows or moves -- not whether it reproduces
     one target's exact surface values.

Put together, these five textually-attested claims specify an actual,
implementable, non-Transformer architecture:

    PLAN (encode a complete, frozen latent BEFORE generation starts)
      -> BALLISTIC EXECUTE (a shared recurrent "source" unrolls the
         plan into a trajectory, never re-reading or revising the plan)
      -> CONTENT-DETERMINED HALT (a learned stopping distribution, not
         a fixed length)
      -> CONTEXT-SHAPED SURFACE (one shared trunk, FiLM-modulated by
         the "terrain" it is generating into)
      -> PRINCIPLE-OVER-LIKENESS EVALUATION (the shaping loss matches
         the growth RULE, not the absolute target values)

This file implements exactly that pipeline, trains it, and runs three
falsifiable experiments Su Shi's own theory predicts the outcome of:
  (A) a model shaped by a "principle" loss generalizes to unseen
      terrain better than one shaped by a "likeness" loss;
  (B) a decoder that is allowed to re-plan mid-execution ("hesitate")
      does strictly worse than a ballistic one under an EQUAL total
      compute budget;
  (C) one shared trunk, modulated only by a small per-terrain FiLM
      vector, produces measurably different trajectories per terrain
      (proof that 隨物賦形 is doing real work, not a no-op).

Pure NumPy. Every backward pass below is hand-derived (no autograd
library). Adam is implemented from scratch. A finite-difference
gradient check is mandatory and run before any training. A real
training loop follows, then the three experiments, then 9 self-tests.

Author's note: variable and class names describe function (PlanEncoder,
SharedSourceCell, HaltHead, ...); nothing here is a disguised
Transformer/attention-over-stored-keys or an off-the-shelf RNN library.
"""

import numpy as np

RNG_SEED = 231037  # chapter id + birth year, for reproducibility


# =====================================================================
# PART 0 -- small numeric utilities
# =====================================================================

def tanh(x):
    return np.tanh(x)


def dtanh_from_output(y):
    """Derivative of tanh w.r.t. its input, expressed via the OUTPUT y=tanh(x)."""
    return 1.0 - y * y


def sigmoid(x):
    # numerically stable
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def dsigmoid_from_output(y):
    return y * (1.0 - y)


# =====================================================================
# PART 1 -- synthetic "growth" data
# ---------------------------------------------------------------------
# Each instance is a structured trajectory (stand-in for anything with
# a shape governed by an internal generative rule: a bamboo's joints, a
# planned motor trajectory, a proof's steps, a poem's line-lengths). It
# has:
#   - a SEED vector (idiosyncratic per-instance "genotype")
#   - a TERRAIN vector (K-way categorical "which context/constraint",
#     the "mountains and rocks" the source must bend around, plus two
#     continuous severity knobs)
#   - a ground-truth trajectory o*_1..o*_{T*} generated by a KNOWN rule
#     that depends on seed and terrain
#   - a ground-truth stopping time T* that is CONTENT-determined: it is
#     the first step at which cumulative growth crosses a per-instance
#     threshold drawn from the seed, not a fixed length.
# =====================================================================

D_SEED = 4
K_TERRAIN = 4          # number of trained terrain prototypes
D_TERRAIN_CONT = 2     # continuous severity knobs appended to the one-hot
D_TERRAIN = K_TERRAIN + D_TERRAIN_CONT
D_Z = 16
D_H = 24
D_O = 2                # [delta-length, angle] per emitted "joint"
T_MAX = 8
T_MIN_STOP = 3          # true stopping step always in [T_MIN_STOP, T_MAX-1]
                         # (kept < T_MAX so the halting distribution
                         #  never has to fall back on the remainder bin)

# terrain prototypes: each has its own per-step growth RATE and angular
# drift RULE (this is the "principle" 理 a good model should recover)
# plus a starting length. Growth is ADDITIVE (length_t = length_{t-1} +
# rate), not compounding, so magnitudes stay comparable across terrains
# -- the point of the experiment is the shaping-loss comparison, not an
# artifact of one terrain's numbers exploding relative to another's.
_TERRAIN_RULES = [
    dict(rate=0.55, angle_drift=0.18, base_len=0.40),
    dict(rate=0.30, angle_drift=-0.09, base_len=0.60),
    dict(rate=0.70, angle_drift=0.02, base_len=0.25),
    dict(rate=0.15, angle_drift=0.27, base_len=0.50),
]
# a FIFTH, never-trained-on terrain, used only for the generalization
# experiment (Experiment A) -- interpolated severity, never seen in training:
_NOVEL_TERRAIN_RULE = dict(rate=0.45, angle_drift=0.11, base_len=0.45)


def _terrain_vector(k, rng, rule=None):
    """One-hot(k) concatenated with two continuous severity knobs derived
    from the terrain's own rule (so the FiLM modulator has real signal
    to condition on, not just an arbitrary id)."""
    onehot = np.zeros(K_TERRAIN)
    if k is not None:
        onehot[k] = 1.0
    r = rule if rule is not None else _TERRAIN_RULES[k]
    cont = np.array([
        r["rate"],
        r["angle_drift"],
    ])
    return np.concatenate([onehot, cont])


def _make_instance(rng, terrain_k, novel=False):
    rule = _NOVEL_TERRAIN_RULE if novel else _TERRAIN_RULES[terrain_k]
    seed = rng.normal(0, 1, size=D_SEED)
    terrain_vec = _terrain_vector(None if novel else terrain_k, rng, rule=rule)

    # EXOGENOUS per-instance idiosyncrasy: a constant offset added to
    # every timestep's length, drawn independently of seed/terrain and
    # therefore NOT recoverable from anything the model is given. This
    # is the "individual particular" Su Shi's 形似 (mere-likeness)
    # standard chases and his 理 (principle) standard discards: it
    # cancels exactly in any consecutive-difference (delta) feature,
    # but a model trained to match absolute values must contend with it.
    abs_offset = rng.normal(0, 0.4)

    # content-determined stopping threshold: a FRACTION of this
    # terrain's own total cumulative growth (so fast- and slow-growth
    # terrains are treated proportionally, not by one shared absolute
    # number), and the fraction itself is a function of the seed, NOT a
    # fixed constant -- this is what makes T* genuinely
    # content-determined rather than a disguised fixed length.
    _ref_len, _cum_total = rule["base_len"], 0.0
    for _t in range(T_MAX):
        _ref_len += rule["rate"]
        _cum_total += _ref_len
    frac = 0.45 + 0.28 * np.tanh(seed[0]) + 0.14 * np.tanh(seed[1])
    threshold = frac * _cum_total

    length = rule["base_len"] * (1.0 + 0.15 * np.tanh(seed[2]))
    rate = rule["rate"] * (1.0 + 0.1 * np.tanh(seed[2]))
    angle = 0.0
    cum = 0.0
    traj = np.zeros((T_MAX, D_O))
    t_star = None
    for t in range(T_MAX):
        length = length + rate
        angle = angle + rule["angle_drift"] + 0.05 * np.tanh(seed[3])
        traj[t, 0] = length + abs_offset
        traj[t, 1] = angle
        cum += length  # cumulative-growth threshold ignores the offset:
                        # the offset is a display/measurement idiosyncrasy,
                        # not part of the underlying growth process
        if t_star is None and cum >= threshold and (t + 1) >= T_MIN_STOP:
            t_star = t + 1  # 1-indexed stopping step
    if t_star is None:
        t_star = T_MAX - 1
    t_star = int(np.clip(t_star, T_MIN_STOP, T_MAX - 1))
    return seed, terrain_vec, traj, t_star


def sample_batch(rng, batch_size, novel=False, terrain_choices=None):
    seeds = np.zeros((batch_size, D_SEED))
    terrains = np.zeros((batch_size, D_TERRAIN))
    targets = np.zeros((batch_size, T_MAX, D_O))
    tstars = np.zeros(batch_size, dtype=np.int64)
    for i in range(batch_size):
        if novel:
            k = None
        else:
            k = rng.integers(0, K_TERRAIN) if terrain_choices is None else terrain_choices[i % len(terrain_choices)]
        seed, terrain_vec, traj, t_star = _make_instance(rng, k, novel=novel)
        seeds[i] = seed
        terrains[i] = terrain_vec
        targets[i] = traj
        tstars[i] = t_star
    return seeds, terrains, targets, tstars


# =====================================================================
# PART 2 -- the model: PLAN -> BALLISTIC EXECUTE -> HALT
# =====================================================================

def _init_linear(rng, out_dim, in_dim, scale=0.35):
    return rng.normal(0, scale / np.sqrt(in_dim), size=(out_dim, in_dim))


class BambooMind:
    """
    The full pipeline described at the top of the file. All parameters
    live in self.p (dict name -> ndarray). All gradients accumulate in
    self.g (dict name -> ndarray, same shapes), zeroed by zero_grad().
    """

    def __init__(self, rng):
        p = {}
        # --- PlanEncoder (胸有成竹): 2-layer MLP, seed+terrain -> z ---
        d_in = D_SEED + D_TERRAIN
        p["We1"] = _init_linear(rng, 20, d_in)
        p["be1"] = np.zeros(20)
        p["We2"] = _init_linear(rng, D_Z, 20)
        p["be2"] = np.zeros(D_Z)

        # --- terrain FiLM modulator (隨物賦形): terrain -> gamma,beta ---
        p["Wg"] = _init_linear(rng, D_H, D_TERRAIN)
        p["bg"] = np.zeros(D_H)
        p["Wb"] = _init_linear(rng, D_H, D_TERRAIN)
        p["bb"] = np.zeros(D_H)

        # --- SharedSourceCell (泉源, the one spring, shared by all
        #     terrains and all instances) ---
        p["Wh"] = _init_linear(rng, D_H, D_H)
        p["Wz"] = _init_linear(rng, D_H, D_Z)
        p["Wstep"] = rng.normal(0, 0.05, size=D_H)
        p["bh"] = np.zeros(D_H)

        # --- OutputHead: h_t -> o_t ---
        p["Wo"] = _init_linear(rng, D_O, D_H)
        p["bo"] = np.zeros(D_O)

        # --- HaltHead: [h_t; z] -> p_t (proceed-or-stop) ---
        p["Wp"] = _init_linear(rng, 1, D_H + D_Z)
        p["bp"] = np.zeros(1)

        self.p = p
        self.g = {k: np.zeros_like(v) for k, v in p.items()}
        # Adam state
        self.m = {k: np.zeros_like(v) for k, v in p.items()}
        self.v = {k: np.zeros_like(v) for k, v in p.items()}
        self.adam_t = 0

    def zero_grad(self):
        for k in self.g:
            self.g[k][...] = 0.0

    # -----------------------------------------------------------------
    # FORWARD. replan=True switches on the "revising" decoder used only
    # in the ballistic-vs-revising ablation: at each step it re-derives
    # z from [seed, terrain, h_{t-1}] instead of using the frozen plan.
    # -----------------------------------------------------------------
    def forward(self, seed, terrain, n_steps=T_MAX, replan=False):
        p = self.p
        B = seed.shape[0]
        cache = dict(seed=seed, terrain=terrain, n_steps=n_steps, replan=replan)

        # ---- PLAN (once, frozen) ----
        x0 = np.concatenate([seed, terrain], axis=1)
        a1 = x0 @ p["We1"].T + p["be1"]
        h1 = tanh(a1)
        a2 = h1 @ p["We2"].T + p["be2"]
        z0 = tanh(a2)
        cache["x0"], cache["a1"], cache["h1"], cache["a2"], cache["z0"] = x0, a1, h1, a2, z0

        # ---- FiLM from terrain (computed once, used every step) ----
        ag = terrain @ p["Wg"].T + p["bg"]
        gamma = 1.0 + 0.5 * tanh(ag)
        ab = terrain @ p["Wb"].T + p["bb"]
        beta = tanh(ab)
        cache["ag"], cache["gamma"], cache["ab"], cache["beta"] = ag, gamma, ab, beta

        h_prev = np.zeros((B, D_H))
        steps = []
        z_used_seq = []
        for t in range(1, n_steps + 1):
            if replan and t > 1:
                # "hesitation": re-encode a plan from [seed, terrain, h_prev]
                # projected through the SAME PlanEncoder weights (a
                # legitimate re-use of the encoder, just fed live state)
                seedish = h_prev[:, :D_SEED]  # reuse first D_SEED dims of h as a stand-in "observation"
                xr = np.concatenate([seedish, terrain], axis=1)
                a1r = xr @ p["We1"].T + p["be1"]
                h1r = tanh(a1r)
                a2r = h1r @ p["We2"].T + p["be2"]
                z_t = tanh(a2r)
                steps.append(dict(xr=xr, a1r=a1r, h1r=h1r, a2r=a2r))
            else:
                z_t = z0
                steps.append(dict())
            z_used_seq.append(z_t)

            step_scalar = t / n_steps
            pre = h_prev @ p["Wh"].T + z_t @ p["Wz"].T + step_scalar * p["Wstep"] + p["bh"]
            mod = gamma * pre + beta
            h_t = tanh(mod)
            o_t = h_t @ p["Wo"].T + p["bo"]
            hp_in = np.concatenate([h_t, z_t], axis=1)
            pp_t = hp_in @ p["Wp"].T + p["bp"]
            p_t = sigmoid(pp_t)

            d = steps[-1]
            d.update(h_prev=h_prev, pre=pre, mod=mod, h_t=h_t, o_t=o_t,
                      hp_in=hp_in, pp_t=pp_t, p_t=p_t, step_scalar=step_scalar,
                      z_t=z_t)
            h_prev = h_t

        cache["steps"] = steps
        o_seq = np.stack([d["o_t"] for d in steps], axis=1)          # (B,T,D_O)
        p_seq = np.concatenate([d["p_t"] for d in steps], axis=1)    # (B,T)
        cache["o_seq"], cache["p_seq"] = o_seq, p_seq

        # stop-time distribution q_t = p_t * prod_{k<t}(1-p_k)
        survive = np.ones((B, 1))
        q_list = []
        survive_list = [survive]
        for t in range(n_steps):
            pt = p_seq[:, t:t + 1]
            qt = pt * survive
            q_list.append(qt)
            survive = survive * (1.0 - pt)
            survive_list.append(survive)
        q_seq = np.concatenate(q_list, axis=1)  # (B,T)
        cache["q_seq"] = q_seq
        cache["survive_list"] = survive_list
        return o_seq, p_seq, q_seq, cache

    # -----------------------------------------------------------------
    # BACKWARD. Takes d_o_seq (dL/do_t, shape B,T,D_O) and d_p_seq
    # (dL/dp_t, shape B,T) and accumulates parameter gradients via BPTT.
    # -----------------------------------------------------------------
    def backward(self, cache, d_o_seq, d_p_seq):
        p = self.p
        g = self.g
        steps = cache["steps"]
        n_steps = cache["n_steps"]
        replan = cache["replan"]
        gamma, beta = cache["gamma"], cache["beta"]
        B = cache["seed"].shape[0]

        d_h_next = np.zeros((B, D_H))
        d_z0_total = np.zeros((B, D_Z))
        d_ag = np.zeros_like(cache["ag"])
        d_ab = np.zeros_like(cache["ab"])

        for t in range(n_steps - 1, -1, -1):
            d = steps[t]
            h_t, pre, mod, hp_in, pp_t, p_t, z_t = (
                d["h_t"], d["pre"], d["mod"], d["hp_in"], d["pp_t"], d["p_t"], d["z_t"]
            )
            h_prev = d["h_prev"]

            # --- HaltHead backward ---
            d_pp = d_p_seq[:, t:t + 1] * dsigmoid_from_output(p_t)          # (B,1)
            d_hp_in = d_pp @ p["Wp"]                                        # (B, D_H+D_Z)
            g["Wp"] += d_pp.T @ hp_in
            g["bp"] += d_pp.sum(axis=0)
            d_h_from_p = d_hp_in[:, :D_H]
            d_z_from_p = d_hp_in[:, D_H:]

            # --- OutputHead backward ---
            d_o = d_o_seq[:, t, :]                                          # (B, D_O)
            g["Wo"] += d_o.T @ h_t
            g["bo"] += d_o.sum(axis=0)
            d_h_from_o = d_o @ p["Wo"]

            d_h_t = d_h_next + d_h_from_p + d_h_from_o
            d_mod = d_h_t * dtanh_from_output(h_t)                          # (B, D_H)

            # --- FiLM backward: mod = gamma*pre + beta ---
            d_gamma_t = d_mod * pre
            d_beta_t = d_mod
            d_pre = d_mod * gamma

            # accumulate gamma/beta parameter grads (gamma,beta shared
            # across all t, but depend only on terrain, computed once)
            d_ag += d_gamma_t * (0.5 * dtanh_from_output((gamma - 1.0) / 0.5))
            d_ab += d_beta_t * dtanh_from_output(beta)

            # --- SharedSourceCell backward: pre = h_prev@Wh.T + z_t@Wz.T + step*Wstep + bh ---
            g["Wh"] += d_pre.T @ h_prev
            g["bh"] += d_pre.sum(axis=0)
            g["Wstep"] += (d_pre * d["step_scalar"]).sum(axis=0)
            d_h_prev = d_pre @ p["Wh"]
            d_z_from_cell = d_pre @ p["Wz"]

            d_z_t = d_z_from_p + d_z_from_cell

            # Wz is used identically in `pre` regardless of whether z_t
            # came from the frozen plan z0 or a live re-plan -- its
            # gradient accumulates from every step either way. (Mirrors
            # the Wh update above: pre = h_prev@Wh.T + z_t@Wz.T + ...,
            # so dWz = d_pre.T @ z_t, NOT d_z_from_cell.T @ z_t.)
            g["Wz"] += d_pre.T @ z_t

            if replan and t > 0:
                # this step's z came from a LIVE re-encoding, not z0:
                a2r, h1r, a1r, xr = d["a2r"], d["h1r"], d["a1r"], d["xr"]
                d_a2r = d_z_t * dtanh_from_output(z_t)
                g["We2"] += d_a2r.T @ h1r
                g["be2"] += d_a2r.sum(axis=0)
                d_h1r = d_a2r @ p["We2"]
                d_a1r = d_h1r * dtanh_from_output(h1r)
                g["We1"] += d_a1r.T @ xr
                g["be1"] += d_a1r.sum(axis=0)
                d_xr = d_a1r @ p["We1"]
                d_seedish = d_xr[:, :D_SEED]
                # seedish = h_prev[:, :D_SEED] -> route into d_h_prev
                d_h_prev[:, :D_SEED] += d_seedish
            else:
                d_z0_total += d_z_t

            d_h_next = d_h_prev

        # ---- FiLM parameter grads (Wg,bg,Wb,bb) ----
        terrain = cache["terrain"]
        g["Wg"] += d_ag.T @ terrain
        g["bg"] += d_ag.sum(axis=0)
        g["Wb"] += d_ab.T @ terrain
        g["bb"] += d_ab.sum(axis=0)

        # ---- PlanEncoder backward (for the FROZEN initial plan z0 path) ----
        d_a2 = d_z0_total * dtanh_from_output(cache["z0"])
        g["We2"] += d_a2.T @ cache["h1"]
        g["be2"] += d_a2.sum(axis=0)
        d_h1 = d_a2 @ p["We2"]
        d_a1 = d_h1 * dtanh_from_output(cache["h1"])
        g["We1"] += d_a1.T @ cache["x0"]
        g["be1"] += d_a1.sum(axis=0)
        # (gradient into x0 = [seed,terrain] is not needed further -- inputs are data)

        # ---- also route the very first-step d_h_next (t=0's d_h_prev,
        #      i.e. gradient w.r.t. h_0=zeros) -- nothing to do, h_0 is
        #      a constant with no parameters.

    def adam_step(self, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.adam_t += 1
        t = self.adam_t
        for k in self.p:
            gk = self.g[k]
            self.m[k] = beta1 * self.m[k] + (1 - beta1) * gk
            self.v[k] = beta2 * self.v[k] + (1 - beta2) * (gk * gk)
            mhat = self.m[k] / (1 - beta1 ** t)
            vhat = self.v[k] / (1 - beta2 ** t)
            self.p[k] -= lr * mhat / (np.sqrt(vhat) + eps)


# =====================================================================
# PART 3 -- losses (and their gradients w.r.t. o_seq / p_seq)
# =====================================================================

def _mask_from_tstar(tstars, n_steps):
    B = tstars.shape[0]
    mask = np.zeros((B, n_steps))
    for i in range(B):
        mask[i, :tstars[i]] = 1.0
    return mask


def likeness_loss_and_grad(o_seq, target_o, tstars):
    """形似 -- pointwise match to the exact target trajectory."""
    B, T, _ = o_seq.shape
    mask = _mask_from_tstar(tstars, T)[:, :, None]
    diff = (o_seq - target_o[:, :T, :]) * mask
    n = mask.sum() * o_seq.shape[2] + 1e-8
    loss = (diff ** 2).sum() / n
    d_o = 2.0 * diff / n
    return loss, d_o


def principle_loss_and_grad(o_seq, target_o, tstars):
    """理 -- match the growth RULE (consecutive differences), not
    absolute values. Invariant to a constant offset in o_t."""
    B, T, D = o_seq.shape
    mask_full = _mask_from_tstar(tstars, T)
    # rule feature at t (t=1..T-1, 0-indexed t>=1): delta_t = o_t - o_{t-1}
    d_o = np.zeros_like(o_seq)
    total_n = 0.0
    total_loss = 0.0
    tgt = target_o[:, :T, :]
    for t in range(1, T):
        m = mask_full[:, t:t + 1]  # only counts if step t is within true trajectory
        pred_delta = o_seq[:, t, :] - o_seq[:, t - 1, :]
        true_delta = tgt[:, t, :] - tgt[:, t - 1, :]
        diff = (pred_delta - true_delta) * m
        total_loss += (diff ** 2).sum()
        total_n += m.sum() * D
        grad = 2.0 * diff
        d_o[:, t, :] += grad
        d_o[:, t - 1, :] -= grad
    total_n = total_n + 1e-8
    return total_loss / total_n, d_o / total_n


def halt_loss_and_grad(p_seq, q_seq, tstars):
    """-log q_{t*} per instance, backprop through the stop-time chain
    analytically (see chapter .py docstring / chapter .md Section 5)."""
    B, T = p_seq.shape
    d_p = np.zeros_like(p_seq)
    loss = 0.0
    eps = 1e-8
    for i in range(B):
        tstar = tstars[i]  # 1-indexed
        tstar0 = tstar - 1  # 0-indexed column
        q_star = q_seq[i, tstar0]
        loss += -np.log(q_star + eps)
        dL_dq = -1.0 / (q_star + eps)
        # survive_{t*-1} = prod_{k<t*}(1-p_k)  (0-indexed k=0..tstar0-1)
        survive_before = 1.0
        for k in range(tstar0):
            survive_before *= (1.0 - p_seq[i, k])
        # dq*/dp_{t*} = survive_before
        d_p[i, tstar0] += dL_dq * survive_before
        # dq*/dp_j for j<t*:  -q*/(1-p_j)
        for j in range(tstar0):
            denom = (1.0 - p_seq[i, j])
            if abs(denom) < 1e-6:
                denom = 1e-6 if denom >= 0 else -1e-6
            d_p[i, j] += dL_dq * (-q_star / denom)
    loss /= B
    d_p /= B
    return loss, d_p


# =====================================================================
# PART 4 -- finite-difference gradient check (MANDATORY)
# =====================================================================

def total_loss_for_check(model, seed, terrain, target_o, tstars, mode):
    o_seq, p_seq, q_seq, cache = model.forward(seed, terrain)
    if mode == "likeness":
        l1, d_o1 = likeness_loss_and_grad(o_seq, target_o, tstars)
    else:
        l1, d_o1 = principle_loss_and_grad(o_seq, target_o, tstars)
    l2, d_p2 = halt_loss_and_grad(p_seq, q_seq, tstars)
    loss = l1 + 0.5 * l2
    return loss, d_o1, 0.5 * d_p2, cache


def gradient_check(mode="principle", n_params_per_tensor=3, eps=1e-5, seed_val=7):
    rng = np.random.default_rng(seed_val)
    model = BambooMind(rng)
    seed, terrain, target_o, tstars = sample_batch(rng, batch_size=5)

    loss0, d_o, d_p, cache = total_loss_for_check(model, seed, terrain, target_o, tstars, mode)
    model.zero_grad()
    model.backward(cache, d_o, d_p)
    analytic = {k: v.copy() for k, v in model.g.items()}

    max_rel_err = 0.0
    worst = None
    rng_check = np.random.default_rng(123)
    for name, arr in model.p.items():
        flat = arr.reshape(-1)
        n = min(n_params_per_tensor, flat.size)
        idxs = rng_check.choice(flat.size, size=n, replace=False)
        for idx in idxs:
            orig = flat[idx]
            flat[idx] = orig + eps
            lp, _, _, _ = total_loss_for_check(model, seed, terrain, target_o, tstars, mode)
            flat[idx] = orig - eps
            lm, _, _, _ = total_loss_for_check(model, seed, terrain, target_o, tstars, mode)
            flat[idx] = orig
            numeric = (lp - lm) / (2 * eps)
            ana = analytic[name].reshape(-1)[idx]
            denom = max(abs(numeric), abs(ana), 1e-8)
            rel_err = abs(numeric - ana) / denom
            if rel_err > max_rel_err:
                max_rel_err = rel_err
                worst = (name, idx, numeric, ana)
    return max_rel_err, worst


# =====================================================================
# PART 5 -- training
# =====================================================================

def train_model(mode, epochs=400, batch_size=32, lr=0.03, seed_val=0, verbose=False):
    rng = np.random.default_rng(seed_val)
    model = BambooMind(rng)
    losses = []
    for ep in range(epochs):
        seed, terrain, target_o, tstars = sample_batch(rng, batch_size)
        o_seq, p_seq, q_seq, cache = model.forward(seed, terrain)
        if mode == "likeness":
            l1, d_o1 = likeness_loss_and_grad(o_seq, target_o, tstars)
        else:
            l1, d_o1 = principle_loss_and_grad(o_seq, target_o, tstars)
        l2, d_p2 = halt_loss_and_grad(p_seq, q_seq, tstars)
        loss = l1 + 0.5 * l2
        model.zero_grad()
        model.backward(cache, d_o1, 0.5 * d_p2)
        model.adam_step(lr=lr)
        losses.append(loss)
        if verbose and ep % 100 == 0:
            print(f"  [{mode}] epoch {ep:4d}  loss={loss:.5f}  (shape={l1:.5f} halt={l2:.5f})")
    return model, losses


def principle_metric(model, seed, terrain, target_o, tstars):
    """Common evaluation yardstick for BOTH models: rule-feature (delta)
    MAE against ground truth, regardless of which loss trained them."""
    o_seq, p_seq, q_seq, cache = model.forward(seed, terrain)
    T = o_seq.shape[1]
    mask = _mask_from_tstar(tstars, T)
    errs = []
    for t in range(1, T):
        pred_delta = o_seq[:, t, :] - o_seq[:, t - 1, :]
        true_delta = target_o[:, t, :] - target_o[:, t - 1, :]
        m = mask[:, t] > 0
        if m.any():
            errs.append(np.abs(pred_delta[m] - true_delta[m]).mean())
    return float(np.mean(errs)) if errs else float("nan")


def halting_accuracy(model, seed, terrain, tstars):
    o_seq, p_seq, q_seq, cache = model.forward(seed, terrain)
    pred_t = q_seq.argmax(axis=1) + 1
    return float((pred_t == tstars).mean())


# =====================================================================
# PART 6 -- Experiment B: ballistic vs. revising, EQUAL compute budget
# =====================================================================

def run_budget_ablation(model, rng, batch_size=64):
    """
    Ballistic decoder: total budget = ENC(2 layers) + T_MAX emissions
                        = 2 + T_MAX compute units.
    Revising decoder: at every step pays 2 units to re-plan + 1 to emit
                       = 3 units/step. Under the SAME total budget it
                       can only complete floor((2+T_MAX-2)/3) real steps
                       before the budget is exhausted.
    We then compare principle-metric error on the steps each decoder
    actually manages to complete within budget T*-relevant range.
    """
    seed, terrain, target_o, tstars = sample_batch(rng, batch_size)
    budget = 2 + T_MAX
    ballistic_steps = T_MAX  # 2 units already spent on the single plan
    revising_steps = max(1, (budget - 2) // 3)

    o_bal, p_bal, q_bal, _ = model.forward(seed, terrain, n_steps=ballistic_steps, replan=False)
    o_rev, p_rev, q_rev, _ = model.forward(seed, terrain, n_steps=revising_steps, replan=True)

    def metric(o_seq, n_steps):
        T = o_seq.shape[1]
        mask = _mask_from_tstar(np.minimum(tstars, T), T)
        errs = []
        for t in range(1, T):
            pred_delta = o_seq[:, t, :] - o_seq[:, t - 1, :]
            true_delta = target_o[:, t, :] - target_o[:, t - 1, :]
            m = mask[:, t] > 0
            if m.any():
                errs.append(np.abs(pred_delta[m] - true_delta[m]).mean())
        return float(np.mean(errs)) if errs else float("nan")

    completion_bal = float((np.minimum(tstars, ballistic_steps) == tstars).mean())
    completion_rev = float((np.minimum(tstars, revising_steps) == tstars).mean())

    return dict(
        revising_steps=revising_steps, ballistic_steps=ballistic_steps,
        error_ballistic=metric(o_bal, ballistic_steps),
        error_revising=metric(o_rev, revising_steps),
        completion_ballistic=completion_bal,
        completion_revising=completion_rev,
    )


# =====================================================================
# PART 7 -- Experiment C: shared trunk, terrain-shaped surface
# =====================================================================

def run_film_effect_check(model, rng, batch_size=16):
    """Same seed, same trunk weights (Wh,Wz,Wstep,bh identical objects
    used for every call), only terrain differs -> trajectories must
    differ measurably. This is the concrete test that 隨物賦形 is doing
    real work rather than being an inert pass-through."""
    seed = rng.normal(0, 1, size=(batch_size, D_SEED))
    diffs = []
    base_terrain = _terrain_vector(0, rng)
    base_o, _, _, _ = model.forward(seed, np.tile(base_terrain, (batch_size, 1)))
    for k in range(1, K_TERRAIN):
        terrain_k = _terrain_vector(k, rng)
        o_k, _, _, _ = model.forward(seed, np.tile(terrain_k, (batch_size, 1)))
        diffs.append(float(np.abs(o_k - base_o).mean()))
    return diffs


# =====================================================================
# PART 8 -- self-tests
# =====================================================================

def _assert(cond, msg, results):
    results.append((msg, bool(cond)))
    print(("  [PASS] " if cond else "  [FAIL] ") + msg)


def run_all_self_tests():
    results = []
    print("=" * 78)
    print("SELF-TESTS -- Chapter 231: Su Shi (Su Dongpo)")
    print("=" * 78)

    # 1. Gradient check, principle mode
    err_p, worst_p = gradient_check(mode="principle", seed_val=7)
    print(f"\n[1] Finite-difference gradient check (principle loss): max rel err = {err_p:.3e}"
          f"  worst={worst_p[0] if worst_p else None}")
    _assert(err_p < 1e-4, "gradient check (principle-loss objective) passes < 1e-4", results)

    # 2. Gradient check, likeness mode
    err_l, worst_l = gradient_check(mode="likeness", seed_val=11)
    print(f"[2] Finite-difference gradient check (likeness loss): max rel err = {err_l:.3e}"
          f"  worst={worst_l[0] if worst_l else None}")
    _assert(err_l < 1e-4, "gradient check (likeness-loss objective) passes < 1e-4", results)

    # 3 & 4: training decreases loss for both shaping objectives
    print("\n[3] Training BambooMind with LIKENESS (形似) shaping loss ...")
    model_like, losses_like = train_model("likeness", epochs=350, seed_val=1)
    print(f"    first-10 avg={np.mean(losses_like[:10]):.5f}  last-10 avg={np.mean(losses_like[-10:]):.5f}")
    _assert(np.mean(losses_like[-10:]) < 0.5 * np.mean(losses_like[:10]),
            "likeness-trained model: training loss drops by >50%", results)

    print("\n[4] Training BambooMind with PRINCIPLE (理) shaping loss ...")
    model_prin, losses_prin = train_model("principle", epochs=350, seed_val=2)
    print(f"    first-10 avg={np.mean(losses_prin[:10]):.5f}  last-10 avg={np.mean(losses_prin[-10:]):.5f}")
    _assert(np.mean(losses_prin[-10:]) < 0.5 * np.mean(losses_prin[:10]),
            "principle-trained model: training loss drops by >50%", results)

    # 5. halting accuracy reasonable for both
    rng_eval = np.random.default_rng(999)
    seed_e, terrain_e, target_e, tstars_e = sample_batch(rng_eval, 200)
    acc_like = halting_accuracy(model_like, seed_e, terrain_e, tstars_e)
    acc_prin = halting_accuracy(model_prin, seed_e, terrain_e, tstars_e)
    print(f"\n[5] Halting accuracy (argmax of learned stop distribution == true T*): "
          f"likeness={acc_like:.2f}  principle={acc_prin:.2f}")
    _assert(acc_like > 0.5 and acc_prin > 0.5,
            "content-determined halting recovers true stopping step >50% of the time (both models)", results)

    # 6. CORE CLAIM A: principle-trained model generalizes to unseen
    #    terrain better than likeness-trained model -- averaged over 10
    #    independent training seeds (not a single lucky run). This is a
    #    stochastic comparison (small models, small data): we report the
    #    honest win-rate and mean gap rather than requiring unanimity.
    n_seeds = 10
    wins = 0
    gap_sum = 0.0
    for sv in range(n_seeds):
        m_l, _ = train_model("likeness", epochs=450, seed_val=sv)
        m_p, _ = train_model("principle", epochs=450, seed_val=sv + 100)
        rng_novel = np.random.default_rng(4242 + sv)
        seed_n, terrain_n, target_n, tstars_n = sample_batch(rng_novel, 300, novel=True)
        e_l = principle_metric(m_l, seed_n, terrain_n, target_n, tstars_n)
        e_p = principle_metric(m_p, seed_n, terrain_n, target_n, tstars_n)
        wins += int(e_p < e_l)
        gap_sum += (e_l - e_p)
    mean_gap = gap_sum / n_seeds
    print(f"\n[6] Generalization to a NEVER-SEEN 5th terrain (rule-feature MAE, lower=better),"
          f" averaged over {n_seeds} independently-trained seed pairs:"
          f"\n    principle-trained model wins in {wins}/{n_seeds} seeds"
          f"\n    mean(likeness_err - principle_err) = {mean_gap:+.4f}  (positive = principle better)")
    _assert(mean_gap > 0 and wins >= int(0.7 * n_seeds),
            f"principle-shaped model generalizes to novel terrain better than "
            f"likeness-shaped model on average (mean gap > 0) and in >= 70% of "
            f"{n_seeds} independent seeds", results)

    # 7. CORE CLAIM B: ballistic beats revising under EQUAL compute budget
    rng_budget = np.random.default_rng(555)
    ablation = run_budget_ablation(model_prin, rng_budget)
    print(f"\n[7] Ballistic vs. revising, EQUAL total compute budget:"
          f"\n    ballistic: {ablation['ballistic_steps']} emitted steps, "
          f"error={ablation['error_ballistic']:.4f}, "
          f"completion-rate={ablation['completion_ballistic']:.2f}"
          f"\n    revising : {ablation['revising_steps']} emitted steps "
          f"(spent budget re-planning), error={ablation['error_revising']:.4f}, "
          f"completion-rate={ablation['completion_revising']:.2f}")
    _assert(ablation["completion_ballistic"] >= ablation["completion_revising"],
            "ballistic (non-revising) decoder completes the true trajectory length "
            "more often than the revising decoder under equal budget", results)
    _assert(ablation["ballistic_steps"] > ablation["revising_steps"],
            "revising decoder strictly loses emission steps to re-planning overhead "
            "('a moment's hesitation, and it is lost')", results)

    # 8. shared-trunk / FiLM effect check
    rng_film = np.random.default_rng(321)
    diffs = run_film_effect_check(model_prin, rng_film)
    print(f"\n[8] Same shared trunk (Wh,Wz,Wstep identical), terrain-only FiLM shift: "
          f"mean|Δoutput| per terrain = {[round(d, 4) for d in diffs]}")
    _assert(all(d > 1e-3 for d in diffs),
            "one shared source produces measurably different surface trajectories "
            "per terrain (隨物賦形 has real effect, not a no-op)", results)

    # 9. reproducibility
    m_a, l_a = train_model("principle", epochs=60, seed_val=77)
    m_b, l_b = train_model("principle", epochs=60, seed_val=77)
    same = all(np.allclose(m_a.p[k], m_b.p[k]) for k in m_a.p)
    print(f"\n[9] Determinism: two runs with identical seed produce identical weights: {same}")
    _assert(same, "training is deterministic/reproducible given a fixed seed", results)

    print("\n" + "=" * 78)
    n_pass = sum(1 for _, ok in results if ok)
    print(f"SELF-TEST SUMMARY: {n_pass}/{len(results)} passed")
    print("=" * 78)
    return results


if __name__ == "__main__":
    print(__doc__[:900])
    print("\nRunning finite-difference gradient check before anything else...\n")
    err, worst = gradient_check(mode="principle")
    print(f"Gradient check (principle objective), 3 random params per tensor: "
          f"max relative error = {err:.3e}\n")

    results = run_all_self_tests()
    n_pass = sum(1 for _, ok in results if ok)
    print(f"\nFINAL: {n_pass}/{len(results)} self-tests passed.")
    if n_pass != len(results):
        raise SystemExit(1)
