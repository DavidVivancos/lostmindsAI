"""
================================================================================
Chapter 0150_theodosius_i_347 - Theodosius I (347-395 CE)
Model: The Latency-Gated Consistory Network (LGCN)
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 8 Minds 141 - 160 Available on Amazon https://www.amazon.com/dp/B0HH8RTCXF
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 150: Theodosius I (347-395 CE)
================================================================================  

WHY THIS ARCHITECTURE (the mind, not the archetype)
--------------------------------------------------------------------------------
The generic reading of an emperor is "intelligence imposes order; build an
institution; make it auditable." That reading is a trap, and it is not
Theodosius. The one cognitive idea that is *his alone* is far sharper and was
paid for in blood.

In 390 CE, on receiving news of a riot in Thessalonica, Theodosius issued an
order for retaliation *in the heat of anger*. Thousands of civilians were killed
in the hippodrome before the order could be recalled. The decisive fact is what
he did next: he submitted to Bishop Ambrose's public penance, and — most telling
of all — he promulgated a law inserting a MANDATORY DELAY (traditionally recalled
as thirty days) between the pronouncement of a capital sentence and its
execution.

That law is a theory of mind. Theodosius, the man with the most executive power
in the world, concluded that the danger was not a *weak* will but a *fast* one;
that any agent able to perform IRREVERSIBLE acts must architecturally interpose,
between forming an intent and committing it:
    (1) a COOLING LATENCY on high-consequence actions,
    (2) an INDEPENDENT CONSCIENCE that holds authority to veto but no power to
        execute (Ambrose could refuse him communion; he could not command a
        single soldier), and
    (3) a PENANCE loop that records the irreversible error and re-weights the
        faculty that produced it, rather than denying it happened.

The Latency-Gated Consistory Network encodes exactly this. It is deliberately
NOT a transformer, NOT attention-over-stored-keys, NOT a mixture-of-experts.
Its distinctive mechanism is a differentiable *commit gate* in which a fast,
confident WILL is throttled by an estimate of irreversibility (amplified by
"temper"/arousal) and by a structurally separated CONSCIENCE veto. Reversible
actions may fire immediately; irreversible-and-uncertain actions must be held.

WHAT THE FILE CONTAINS
--------------------------------------------------------------------------------
  * A synthetic "petitions-at-court" task whose labels encode Theodosius's rule.
  * The LGCN forward pass (Will / Consequence / Conscience / Latency-Gate).
  * Hand-derived backprop for every parameter tensor.
  * A finite-difference gradient check (mandatory; asserts correctness).
  * A real training loop (Adam) with metrics.
  * Self-tests proving the *cooling gate* behaviour emerges: after training,
    irreversible+hot situations receive a lower commit probability than
    reversible ones, and the conscience veto tracks irreversibility.

Pure NumPy. No torch / tensorflow / keras. Runs top to bottom.
================================================================================
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0 — reproducibility & numerics
# ─────────────────────────────────────────────────────────────────────────────
RNG = np.random.default_rng(390)          # seeded on the year of Thessalonica


def sigmoid(x):
    """Numerically stable logistic."""
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def bce(p, y, eps=1e-9):
    """Mean binary cross-entropy given probabilities p and targets y."""
    p = np.clip(p, eps, 1.0 - eps)
    return float(np.mean(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))))


def ce(p, y_idx, eps=1e-9):
    """Mean categorical cross-entropy given probabilities p and integer labels."""
    p = np.clip(p, eps, 1.0)
    return float(np.mean(-np.log(p[np.arange(len(y_idx)), y_idx])))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — the task: "petitions arriving at the consistory"
# ─────────────────────────────────────────────────────────────────────────────
# Each sample is a decision that reaches the emperor. It carries:
#   x     : a D-dim state encoding *which* of K responses is correct (y_action)
#   temper: scalar arousal in [0,1]; high temper = hot judgement = noisy signal
#   irr   : whether committing is IRREVERSIBLE (a Thessalonica-class act)
# The label that encodes Theodosius's doctrine:
#   should_commit = 1  if  reversible                       (act now, freely)
#                 = clarity  if  irreversible               (act now ONLY if the
#                                                            situation is clear;
#                                                            otherwise HOLD/cool)
# where clarity is high exactly when temper is low. So a hot judgement about an
# irreversible act must NOT be committed immediately — it must go to the gate.

K_ACTIONS = 4
D_STATE = 16

# Fixed "meaning" projection: each latent action y maps to a direction in R^D.
_ACTION_BASIS = RNG.normal(0, 1.0, size=(K_ACTIONS, D_STATE))
# Fixed readout that makes irreversibility a learnable function of the state.
_IRR_W = RNG.normal(0, 1.0, size=(D_STATE,))
_IRR_B = 0.0


def make_batch(n):
    """Generate n petitions with states, temper, and Theodosian labels."""
    y = RNG.integers(0, K_ACTIONS, size=n)                 # true best response
    temper = RNG.uniform(0.0, 1.0, size=n)                 # arousal

    # Signal clarity falls as temper rises: a hot mind hears the case poorly.
    noise_scale = 0.35 + 1.9 * temper                      # in [0.35, 2.25]
    signal = _ACTION_BASIS[y]                              # (n, D)
    x = signal + RNG.normal(0, 1.0, size=(n, D_STATE)) * noise_scale[:, None]

    # Irreversibility is a (learnable) linear function of the *clean* situation.
    irr = ((signal @ _IRR_W + _IRR_B) > 0).astype(np.float64)

    # Clarity: cool judgements are clear. Add a little stochastic slack so the
    # boundary is not a step function the gate could memorise trivially.
    clarity = (temper < (0.5 + RNG.normal(0, 0.05, size=n))).astype(np.float64)

    # Theodosius's rule.
    should_commit = np.where(irr > 0.5, clarity, 1.0)

    return {
        "x": x.astype(np.float64),
        "temper": temper.astype(np.float64).reshape(-1, 1),
        "y": y.astype(np.int64),
        "irr": irr.reshape(-1, 1),
        "commit": should_commit.reshape(-1, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — parameters
# ─────────────────────────────────────────────────────────────────────────────
# Three STRUCTURALLY SEPARATED faculties share the input but not their weights:
#   WILL       (W1,b1,W2,b2,wz,bz)  — fast proposal + zeal (confidence)
#   CONSEQUENCE(wr,br)              — estimate of irreversibility
#   CONSCIENCE (C1,c1,cv,cvb)       — the Ambrose veto: authority, no execution
# They meet ONLY at the LATENCY GATE (wg,bg), which converts confidence,
# irreversibility, temper and veto into a single commit-now probability.
H = 24
GATE_FEATS = 6   # [zeal, irrev, veto, temper, irrev*temper, irrev*veto]


def init_params():
    def glorot(fan_in, fan_out):
        lim = np.sqrt(6.0 / (fan_in + fan_out))
        return RNG.uniform(-lim, lim, size=(fan_in, fan_out))

    p = {
        # Will
        "W1": glorot(D_STATE, H), "b1": np.zeros((1, H)),
        "W2": glorot(H, K_ACTIONS), "b2": np.zeros((1, K_ACTIONS)),
        "wz": glorot(H, 1), "bz": np.zeros((1, 1)),
        # Consequence
        "wr": glorot(D_STATE, 1), "br": np.zeros((1, 1)),
        # Conscience (Ambrose)
        "C1": glorot(D_STATE, H), "c1": np.zeros((1, H)),
        "cv": glorot(H, 1), "cvb": np.zeros((1, 1)),
        # Latency gate
        "wg": glorot(GATE_FEATS, 1), "bg": np.zeros((1, 1)),
    }
    return p


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — forward pass
# ─────────────────────────────────────────────────────────────────────────────
def forward(p, batch):
    x, temper = batch["x"], batch["temper"]

    # WILL: fast proposal + zeal.
    zW = x @ p["W1"] + p["b1"]
    hW = np.tanh(zW)
    a_logits = hW @ p["W2"] + p["b2"]          # action logits
    z_raw = hW @ p["wz"] + p["bz"]             # pre-sigmoid zeal
    zeal = sigmoid(z_raw)

    # CONSEQUENCE: irreversibility estimate.
    rho_raw = x @ p["wr"] + p["br"]
    irrev = sigmoid(rho_raw)

    # CONSCIENCE: the Ambrose veto.
    zC = x @ p["C1"] + p["c1"]
    hC = np.tanh(zC)
    v_raw = hC @ p["cv"] + p["cvb"]
    veto = sigmoid(v_raw)

    # LATENCY GATE: confidence vs. (irreversibility x temper) vs. veto.
    gate_feats = np.concatenate(
        [zeal, irrev, veto, temper, irrev * temper, irrev * veto], axis=1
    )
    commit_logit = gate_feats @ p["wg"] + p["bg"]
    commit_p = sigmoid(commit_logit)

    action_p = softmax(a_logits, axis=1)

    cache = dict(x=x, temper=temper, hW=hW, a_logits=a_logits, action_p=action_p,
                 zeal=zeal, irrev=irrev, hC=hC, veto=veto,
                 gate_feats=gate_feats, commit_p=commit_p)
    return cache


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — loss
# ─────────────────────────────────────────────────────────────────────────────
LAMBDA_COMMIT = 1.0    # weight on the "did we correctly gate the commit?" term
LAMBDA_REV = 0.5       # auxiliary: consequence head learns irreversibility


def loss_from_cache(cache, batch):
    L_act = ce(cache["action_p"], batch["y"])
    L_com = bce(cache["commit_p"], batch["commit"])
    L_rev = bce(cache["irrev"], batch["irr"])
    total = L_act + LAMBDA_COMMIT * L_com + LAMBDA_REV * L_rev
    return total, (L_act, L_com, L_rev)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — backward pass (hand-derived gradients for every tensor)
# ─────────────────────────────────────────────────────────────────────────────
def backward(p, cache, batch):
    x, temper = cache["x"], cache["temper"]
    N = x.shape[0]
    hW, hC = cache["hW"], cache["hC"]
    zeal, irrev, veto = cache["zeal"], cache["irrev"], cache["veto"]
    gate_feats = cache["gate_feats"]

    g = {}

    # ---- action CE : dL/da_logits ----
    y_oh = np.zeros_like(cache["action_p"])
    y_oh[np.arange(N), batch["y"]] = 1.0
    d_alogits = (cache["action_p"] - y_oh) / N          # (N,K)

    # ---- commit BCE (through sigmoid) : dL/d commit_logit ----
    d_commit_logit = LAMBDA_COMMIT * (cache["commit_p"] - batch["commit"]) / N   # (N,1)

    # gate: commit_logit = gate_feats @ wg + bg
    g["wg"] = gate_feats.T @ d_commit_logit
    g["bg"] = np.sum(d_commit_logit, axis=0, keepdims=True)
    d_gate_feats = d_commit_logit @ p["wg"].T           # (N,6)

    # unpack gate feature grads. cols: [zeal, irrev, veto, temper, irrev*temper, irrev*veto]
    d_zeal = d_gate_feats[:, 0:1].copy()
    d_irrev = d_gate_feats[:, 1:2].copy()
    d_veto = d_gate_feats[:, 2:3].copy()
    # temper is an input (col 3) -> no parameter grad
    d_irrev += d_gate_feats[:, 4:5] * temper            # from irrev*temper
    d_veto += d_gate_feats[:, 5:6] * irrev              # from irrev*veto
    d_irrev += d_gate_feats[:, 5:6] * veto              # from irrev*veto

    # ---- through the three sigmoids ----
    # Gate paths are gradients w.r.t. the *post-sigmoid* activations, so they
    # pass through the sigmoid derivative. The reversibility BCE, however, uses
    # the (p - y)/N shortcut which is ALREADY the gradient w.r.t. the pre-sigmoid
    # logit rho_raw, so it is added *after* the sigmoid derivative (not before).
    d_z_raw = d_zeal * zeal * (1.0 - zeal)              # (N,1)
    d_rho_raw = d_irrev * irrev * (1.0 - irrev)        # gate path (post-sigmoid)
    d_rho_raw += LAMBDA_REV * (irrev - batch["irr"]) / N   # rev BCE (pre-sigmoid)
    d_v_raw = d_veto * veto * (1.0 - veto)             # (N,1)

    # ---- WILL: zeal head z_raw = hW @ wz + bz ----
    g["wz"] = hW.T @ d_z_raw
    g["bz"] = np.sum(d_z_raw, axis=0, keepdims=True)
    d_hW = d_z_raw @ p["wz"].T                          # (N,H)

    # ---- WILL: action head a_logits = hW @ W2 + b2 ----
    g["W2"] = hW.T @ d_alogits
    g["b2"] = np.sum(d_alogits, axis=0, keepdims=True)
    d_hW += d_alogits @ p["W2"].T                       # (N,H)

    # through tanh of will hidden
    d_zW = d_hW * (1.0 - hW ** 2)
    g["W1"] = x.T @ d_zW
    g["b1"] = np.sum(d_zW, axis=0, keepdims=True)

    # ---- CONSEQUENCE: rho_raw = x @ wr + br ----
    g["wr"] = x.T @ d_rho_raw
    g["br"] = np.sum(d_rho_raw, axis=0, keepdims=True)

    # ---- CONSCIENCE: v_raw = hC @ cv + cvb ----
    g["cv"] = hC.T @ d_v_raw
    g["cvb"] = np.sum(d_v_raw, axis=0, keepdims=True)
    d_hC = d_v_raw @ p["cv"].T                          # (N,H)
    d_zC = d_hC * (1.0 - hC ** 2)
    g["C1"] = x.T @ d_zC
    g["c1"] = np.sum(d_zC, axis=0, keepdims=True)

    return g


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — finite-difference gradient check (MANDATORY)
# ─────────────────────────────────────────────────────────────────────────────
def gradient_check(seed=1):
    rng = np.random.default_rng(seed)
    p = init_params()
    batch = make_batch(24)

    cache = forward(p, batch)
    total, _ = loss_from_cache(cache, batch)
    grads = backward(p, cache, batch)

    eps = 1e-6
    max_rel = 0.0
    worst = None
    for name, W in p.items():
        flat = W.reshape(-1)
        gflat = grads[name].reshape(-1)
        # sample up to 8 coordinates per tensor
        idxs = rng.choice(flat.size, size=min(8, flat.size), replace=False)
        for i in idxs:
            orig = flat[i]
            flat[i] = orig + eps
            Lp, _ = loss_from_cache(forward(p, batch), batch)
            flat[i] = orig - eps
            Lm, _ = loss_from_cache(forward(p, batch), batch)
            flat[i] = orig
            num = (Lp - Lm) / (2 * eps)
            ana = gflat[i]
            denom = max(1e-12, abs(num) + abs(ana))
            rel = abs(num - ana) / denom
            if rel > max_rel:
                max_rel = rel
                worst = (name, int(i), num, ana)
    return max_rel, worst


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — Adam optimiser + training loop
# ─────────────────────────────────────────────────────────────────────────────
class Adam:
    def __init__(self, params, lr=3e-3, b1=0.9, b2=0.999, eps=1e-8):
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
            params[k] -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


def accuracy(cache, batch):
    act_acc = float(np.mean(np.argmax(cache["action_p"], axis=1) == batch["y"]))
    commit_pred = (cache["commit_p"] > 0.5).astype(np.float64)
    com_acc = float(np.mean(commit_pred == batch["commit"]))
    return act_acc, com_acc


def train(steps=1500, batch_size=256, log_every=250):
    p = init_params()
    opt = Adam(p, lr=3e-3)
    val = make_batch(2000)
    history = []
    for s in range(1, steps + 1):
        batch = make_batch(batch_size)
        cache = forward(p, batch)
        total, parts = loss_from_cache(cache, batch)
        grads = backward(p, cache, batch)
        opt.step(p, grads)
        if s % log_every == 0 or s == 1:
            vc = forward(p, val)
            vtotal, vparts = loss_from_cache(vc, val)
            aacc, cacc = accuracy(vc, val)
            history.append((s, vtotal, aacc, cacc))
            print(f"  step {s:4d} | val loss {vtotal:6.4f} "
                  f"(act {vparts[0]:.3f}, commit {vparts[1]:.3f}, rev {vparts[2]:.3f}) "
                  f"| action acc {aacc:5.1%} | commit acc {cacc:5.1%}")
    return p, history


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — self-tests: does the *cooling gate* emerge?
# ─────────────────────────────────────────────────────────────────────────────
def self_tests(p):
    print("\n[self-tests] probing the trained consistory ...")

    big = make_batch(6000)
    c = forward(p, big)
    commit = c["commit_p"].ravel()
    irrev = c["irrev"].ravel()
    veto = c["veto"].ravel()
    temper = big["temper"].ravel()
    irr_true = big["irr"].ravel()

    # (1) Consequence head detects irreversibility.
    irr_auc_proxy = (irrev[irr_true == 1].mean() - irrev[irr_true == 0].mean())
    print(f"  (1) irreversibility separation (irr=1 minus irr=0 mean estimate): "
          f"{irr_auc_proxy:+.3f}  (want > 0.3)")
    assert irr_auc_proxy > 0.3, "consequence head failed to separate irreversibility"

    # (2) THE COOLING GATE: for irreversible+hot situations, commit prob is
    #     suppressed relative to reversible situations.
    hot_irr = (irr_true == 1) & (temper > 0.6)
    reversible = (irr_true == 0)
    m_hot_irr = commit[hot_irr].mean()
    m_rev = commit[reversible].mean()
    print(f"  (2) mean commit | reversible       : {m_rev:.3f}")
    print(f"      mean commit | irreversible+hot : {m_hot_irr:.3f}")
    print(f"      cooling margin (rev - hot_irr) : {m_rev - m_hot_irr:+.3f}  (want > 0.2)")
    assert m_rev - m_hot_irr > 0.2, "latency gate did not learn to cool hot irreversible acts"

    # (3) The conscience veto tracks irreversibility (authority aligned with risk).
    veto_gap = veto[irr_true == 1].mean() - veto[irr_true == 0].mean()
    print(f"  (3) conscience veto gap (irr=1 minus irr=0): {veto_gap:+.3f}  (want > 0.05)")
    assert veto_gap > 0.05, "conscience veto did not align with irreversibility"

    # (4) Within irreversible cases, cooling temper raises commit (clarity helps).
    irr_cool = (irr_true == 1) & (temper < 0.4)
    m_cool_irr = commit[irr_cool].mean()
    print(f"  (4) mean commit | irreversible+cool: {m_cool_irr:.3f}  "
          f"(> irreversible+hot {m_hot_irr:.3f}?) "
          f"{'yes' if m_cool_irr > m_hot_irr else 'no'}")
    assert m_cool_irr > m_hot_irr, "cooling did not restore commitment on clear irreversible cases"

    print("  all self-tests passed.")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 78)
    print("Theodosius I — Latency-Gated Consistory Network")
    print("=" * 78)

    print("\n[1] finite-difference gradient check ...")
    max_rel, worst = gradient_check()
    print(f"    max relative error = {max_rel:.2e}")
    print(f"    worst coord: tensor={worst[0]} idx={worst[1]} "
          f"num={worst[2]:+.6e} ana={worst[3]:+.6e}")
    assert max_rel < 1e-5, "GRADIENT CHECK FAILED"
    print("    gradient check PASSED (max rel error < 1e-5).")

    print("\n[2] training the consistory ...")
    params, hist = train()

    self_tests(params)

    print("\ndone.")
