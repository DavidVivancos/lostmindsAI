"""
================================================================================
Chapter 137 - Cao Cao (155-220 CE)
The Logistics-Gated Strategy Network (LGSN)
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 7 Minds 121 - 140 Available on Amazon https://www.amazon.com/dp/B0HFN6GXMH
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 137: Cao Cao (155-220 CE)
================================================================================   

A from-scratch, pure-NumPy neural architecture that encodes the specific
cognitive signature of Cao Cao -- warlord, administrator, and poet of the late
Eastern Han.

WHY THIS ARCHITECTURE, AND NOT A GENERIC MODEL
------------------------------------------------------------------
The lazy reading of a warlord is "mixture-of-experts": a council of advisors,
a gate that routes to the best one. That reading fits a hundred sovereigns and
says nothing about *this* mind. Cao Cao's distinctive move is narrower and
sharper. Twice he settled a war not by matching mass against mass but by acting
on the *sustaining substrate* of his enemy's power:

  * At Guandu (200 CE) he ignored the huge army in front of him and rode 5,000
    men through the night to burn the grain depot at Wuchao. The army did not
    lose a battle; it lost its food, and then it dissolved.
  * Everywhere he governed he planted "tuntian" -- self-sustaining military
    farm colonies -- so that his own capability fed itself and did not starve
    on campaign.

Add his talent doctrine ("only ability is raised" -- promote competence even
from enemies, even from the disreputable, ignoring pedigree), and a single
cognitive thesis emerges:

  Reason about the RESOURCE THAT FEEDS A CAPABILITY, not the capability's
  surface size or its reputation. Whoever controls supply controls the
  outcome -- disproportionately and durably.

The LGSN turns that thesis into four coupled mechanisms:

  1. GRAIN-BUDGET ALLOCATION (finite supply). Unlike a softmax gate that fires
     every expert "for free," the LGSN owns a scarce budget B and must divide
     it across assets. Attention here is *logistics*: you cannot feed
     everything, so you choose. (Talent deployed under scarcity.)

  2. LEVERAGE / WUCHAO HEAD (critical-node saliency). A learned head scores how
     much each asset drives the downstream outcome, so budget flows to the
     supply nodes -- the grain depot -- rather than to the visible mass.

  3. VALUE-DECOUPLING ADVERSARY (only-ability-is-raised). A small adversary
     tries to read each asset's PROVENANCE (noble / humble / former-enemy) from
     the encoder's representation. Through a gradient-reversal path the encoder
     is pushed to make provenance UNREADABLE while keeping competence legible.
     The model literally learns "I do not care where the talent came from."

  4. TUNTIAN RESERVOIR (self-provisioning over time). A slow recurrent state
     accumulates budget across a campaign of steps: sustained, well-run effort
     grows the supply you have to allocate. (Build your own grain.)

Training adds a fifth, Red-Cliffs credit assignment: after a catastrophic
error the update is concentrated on the modules most to blame rather than
smeared globally -- the way Cao Cao treated Red Cliffs as one local defeat, not
a refutation of his whole strategy.

CONVENTIONS
-----------
* Pure NumPy. No autograd. Every gradient is derived by hand.
* tanh activations throughout so the analytic gradient is smooth and the
  finite-difference check is exact (no ReLU kinks).
* A finite-difference gradient check (REQUIRED) validates every parameter's
  analytic gradient against numerical gradients before any training runs.
* A synthetic "contested-field" task is constructed so that the architecture's
  inductive biases actually pay off: the label depends on the competence of the
  SUPPLY nodes and is INDEPENDENT of provenance, while provenance is a spurious
  distractor in the training split whose correlation is broken at test time. A
  model that learns leverage + decoupling generalizes; a naive model overfits
  to pedigree and fails on the shift.

Run:  python3 chapter_0137_cao_cao_155.py
================================================================================
"""

import numpy as np

RNG = np.random.default_rng(155)  # seeded on Cao Cao's birth year


# ============================================================================
# 0. SMALL NUMERIC HELPERS
# ============================================================================

def softmax(z, axis=-1):
    """Numerically stable softmax."""
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def cross_entropy(probs, labels):
    """Mean cross-entropy. probs: (..., C) summing to 1; labels: (...) int."""
    flat_p = probs.reshape(-1, probs.shape[-1])
    flat_y = labels.reshape(-1)
    picked = flat_p[np.arange(flat_p.shape[0]), flat_y]
    return -np.mean(np.log(picked + 1e-12))


def one_hot(labels, C):
    out = np.zeros(labels.shape + (C,))
    idx = np.indices(labels.shape)
    out[tuple(idx) + (labels,)] = 1.0
    return out


# ============================================================================
# 1. THE MODEL PARAMETERS
# ============================================================================
# Dimensions (kept small so the exact gradient check is fast):
#   F  = raw features per asset
#   H  = competence-embedding size
#   L  = leverage-head hidden size
#   C  = number of task classes (which move wins)
#   P  = number of provenance classes (noble / humble / former-enemy)
# ----------------------------------------------------------------------------

class Dims:
    def __init__(self, F=5, H=8, L=6, C=2, P=3):
        self.F, self.H, self.L, self.C, self.P = F, H, L, C, P


def init_params(d: Dims):
    """Xavier-ish initialisation. Returns a flat dict of named arrays."""
    def w(fan_in, fan_out):
        return RNG.standard_normal((fan_in, fan_out)) * np.sqrt(1.0 / fan_in)

    p = {
        # --- competence encoder: x_k -> h_k  (two tanh layers) ---
        "W1": w(d.F, d.H), "b1": np.zeros(d.H),
        "W2": w(d.H, d.H), "b2": np.zeros(d.H),
        # --- leverage / Wuchao head: h_k -> scalar leverage ---
        "A":  w(d.H, d.L), "a": np.zeros(d.L),
        "vlev": w(d.L, 1).ravel(),
        # --- priority vector u (competence contribution to allocation) ---
        "u":  w(d.H, 1).ravel(),
        # --- task output head: pooled z -> class logits ---
        "Wo": w(d.H, d.C), "bo": np.zeros(d.C),
        # --- provenance adversary: h_k -> provenance logits ---
        "Wa": w(d.H, d.P), "ba": np.zeros(d.P),
    }
    return p


# ============================================================================
# 2. FORWARD PASS
# ============================================================================
# Shapes:
#   X   : (N, K, F)   N samples, K assets each, F features
#   B   : scalar grain budget (B < K forces genuine scarcity)
# Returns task probs, adversary probs, and a cache for backprop.
# ----------------------------------------------------------------------------

def forward(p, X, B):
    N, K, F = X.shape

    # --- competence encoder (shared across assets) ---
    pre1 = X @ p["W1"] + p["b1"]          # (N,K,H)
    h1 = np.tanh(pre1)
    pre2 = h1 @ p["W2"] + p["b2"]         # (N,K,H)
    h = np.tanh(pre2)                     # competence embedding h_k

    # --- leverage head: how much does asset k drive the outcome? ---
    prel = h @ p["A"] + p["a"]            # (N,K,L)
    hl = np.tanh(prel)
    lev = hl @ p["vlev"]                  # (N,K) scalar leverage per asset

    # --- allocation priorities and grain-budget gate ---
    prio = h @ p["u"] + lev              # (N,K) priority per asset
    sm = softmax(prio, axis=1)            # (N,K) shares summing to 1 per sample
    g = B * sm                            # (N,K) grain allocated (sums to B)

    # --- pool: strength that is actually FED = sum of g_k * h_k ---
    z = np.einsum("nk,nkh->nh", g, h)     # (N,H)

    # --- task head ---
    tlogits = z @ p["Wo"] + p["bo"]       # (N,C)
    tprobs = softmax(tlogits, axis=1)

    # --- provenance adversary (per asset) ---
    alogits = h @ p["Wa"] + p["ba"]       # (N,K,P)
    aprobs = softmax(alogits, axis=2)

    cache = dict(X=X, B=B, pre1=pre1, h1=h1, pre2=pre2, h=h,
                 prel=prel, hl=hl, lev=lev, prio=prio, sm=sm, g=g, z=z,
                 tlogits=tlogits, tprobs=tprobs, alogits=alogits, aprobs=aprobs)
    return tprobs, aprobs, cache


# ============================================================================
# 3. LOSS
# ============================================================================
# Composite scalar used for the GRADIENT CHECK (adversary attached by identity,
# i.e. NO gradient reversal): L = L_task + lam_adv * L_adv.
# During TRAINING we reuse the same analytic gradients but flip the sign of the
# adversary's gradient into the encoder (gradient reversal); see train_step.
# ----------------------------------------------------------------------------

def loss_and_grads(p, X, y_task, y_prov, B, lam_adv, dims: Dims):
    """Return (total_loss, grads_dict, parts) with exact analytic gradients of
    L = L_task + lam_adv * L_adv, treating the adversary path as identity."""
    N, K, F = X.shape
    H, C, P, L = dims.H, dims.C, dims.P, dims.L
    tprobs, aprobs, c = forward(p, X, B)

    L_task = cross_entropy(tprobs, y_task)
    L_adv = cross_entropy(aprobs, y_prov)
    total = L_task + lam_adv * L_adv

    g = {k: np.zeros_like(v) for k, v in p.items()}

    # ---- task head gradient: dL_task/dtlogits = (probs - onehot)/N ----
    dtlogits = (c["tprobs"] - one_hot(y_task, C)) / N          # (N,C)
    g["Wo"] += c["z"].T @ dtlogits                             # (H,C)
    g["bo"] += dtlogits.sum(axis=0)
    dz = dtlogits @ p["Wo"].T                                  # (N,H)

    # ---- pool backward: z = sum_k g_k h_k ----
    # dh from pooling (through the h factor), dg from pooling (through the gate)
    dh = c["g"][:, :, None] * dz[:, None, :]                   # (N,K,H)
    dg = np.einsum("nh,nkh->nk", dz, c["h"])                   # (N,K)

    # ---- grain gate backward: g = B * softmax(prio) ----
    # softmax jacobian: dprio = sm * (dg_scaled - sum_k sm*dg_scaled)
    dg_scaled = c["B"] * dg                                    # d/d(softmax)
    s = c["sm"]
    dot = np.sum(s * dg_scaled, axis=1, keepdims=True)
    dprio = s * (dg_scaled - dot)                             # (N,K)

    # prio = h @ u + lev
    du = np.einsum("nk,nkh->h", dprio, c["h"])                # (H,)
    dh += dprio[:, :, None] * p["u"][None, None, :]           # via u path
    dlev = dprio                                              # (N,K)
    g["u"] += du

    # ---- leverage head backward: lev = tanh(h@A + a) @ vlev ----
    dhl = dlev[:, :, None] * p["vlev"][None, None, :]         # (N,K,L)
    g["vlev"] += np.einsum("nk,nkl->l", dlev, c["hl"])
    dprel = dhl * (1.0 - c["hl"] ** 2)                        # tanh'
    g["A"] += np.einsum("nkh,nkl->hl", c["h"], dprel)
    g["a"] += dprel.sum(axis=(0, 1))
    dh += dprel @ p["A"].T                                    # (N,K,H)

    # ---- adversary head backward (identity path for the check) ----
    dalogits = lam_adv * (c["aprobs"] - one_hot(y_prov, P)) / (N * K)  # (N,K,P)
    g["Wa"] += np.einsum("nkh,nkp->hp", c["h"], dalogits)
    g["ba"] += dalogits.sum(axis=(0, 1))
    dh_adv = dalogits @ p["Wa"].T                            # (N,K,H)
    dh += dh_adv

    # ---- competence encoder backward: h = tanh(tanh(x W1+b1) W2+b2) ----
    dpre2 = dh * (1.0 - c["h"] ** 2)                          # (N,K,H)
    g["W2"] += np.einsum("nkh,nkj->hj", c["h1"], dpre2)
    g["b2"] += dpre2.sum(axis=(0, 1))
    dh1 = dpre2 @ p["W2"].T                                   # (N,K,H)
    dpre1 = dh1 * (1.0 - c["h1"] ** 2)
    g["W1"] += np.einsum("nkf,nkh->fh", c["X"], dpre1)
    g["b1"] += dpre1.sum(axis=(0, 1))

    parts = dict(L_task=L_task, L_adv=L_adv, dh_adv=dh_adv, cache=c)
    return total, g, parts


def encoder_backward(p, cache, dh):
    """Backprop an incoming gradient dh (N,K,H) w.r.t. the competence embedding
    through the two-layer encoder. Returns grads for W1,b1,W2,b2."""
    c = cache
    dpre2 = dh * (1.0 - c["h"] ** 2)
    gW2 = np.einsum("nkh,nkj->hj", c["h1"], dpre2)
    gb2 = dpre2.sum(axis=(0, 1))
    dh1 = dpre2 @ p["W2"].T
    dpre1 = dh1 * (1.0 - c["h1"] ** 2)
    gW1 = np.einsum("nkf,nkh->fh", c["X"], dpre1)
    gb1 = dpre1.sum(axis=(0, 1))
    return {"W1": gW1, "b1": gb1, "W2": gW2, "b2": gb2}


def adversary_grads(p, cache, y_prov, P):
    """Gradients for the provenance adversary and dh flowing back to encoder."""
    c = cache
    N, K = y_prov.shape
    dalogits = (c["aprobs"] - one_hot(y_prov, P)) / (N * K)   # (N,K,P)
    gWa = np.einsum("nkh,nkp->hp", c["h"], dalogits)
    gba = dalogits.sum(axis=(0, 1))
    dh_adv = dalogits @ p["Wa"].T                            # (N,K,H)
    return gWa, gba, dh_adv


# ============================================================================
# 4. FINITE-DIFFERENCE GRADIENT CHECK  (mandatory)
# ============================================================================

def gradient_check(dims: Dims, verbose=True):
    p = init_params(dims)
    N, K = 4, 5
    B = 2.5  # scarce budget: B < K
    X = RNG.standard_normal((N, K, dims.F))
    y_task = RNG.integers(0, dims.C, size=N)
    y_prov = RNG.integers(0, dims.P, size=(N, K))
    lam_adv = 0.7

    _, analytic, _ = loss_and_grads(p, X, y_task, y_prov, B, lam_adv, dims)

    eps = 1e-6
    max_rel = 0.0
    worst = None
    for name in p:
        flat = p[name].ravel()
        gnum = np.zeros_like(flat)
        for i in range(flat.size):
            orig = flat[i]
            flat[i] = orig + eps
            lp, _, _ = loss_and_grads(p, X, y_task, y_prov, B, lam_adv, dims)
            flat[i] = orig - eps
            lm, _, _ = loss_and_grads(p, X, y_task, y_prov, B, lam_adv, dims)
            flat[i] = orig
            gnum[i] = (lp - lm) / (2 * eps)
        ga = analytic[name].ravel()
        denom = np.maximum(1e-8, np.abs(ga) + np.abs(gnum))
        rel = np.max(np.abs(ga - gnum) / denom)
        if rel > max_rel:
            max_rel, worst = rel, name
        if verbose:
            print(f"    {name:>5s}: max rel err = {rel:.2e}")
    print(f"\n  Worst parameter: {worst}   max relative error = {max_rel:.2e}")
    ok = max_rel < 1e-5
    print(f"  GRADIENT CHECK: {'PASS' if ok else 'FAIL'} (threshold 1e-5)")
    return ok


# ============================================================================
# 5. SYNTHETIC CONTESTED-FIELD TASK
# ============================================================================
# Each sample is a "field" of K assets. Every asset has:
#   feat[0]      : observed competence (noisy)
#   feat[1]      : is-supply-node flag (1 = grain depot / critical node)
#   feat[2]      : provenance one-hot-ish signal (a distractor)
#   feat[3..]    : noise
# TASK LABEL (y_task): does the campaign SUCCEED? Success iff the total
#   competence located ON THE SUPPLY NODES exceeds a threshold. Provenance is
#   irrelevant to the label.
# PROVENANCE LABEL (y_prov): each asset's pedigree class {noble,humble,enemy}.
# SPURIOUS TRAP: in the TRAIN split, provenance is correlated with competence
#   (nobles tend to look competent). In the TEST split that correlation is
#   REVERSED. A model that leans on pedigree fails; a model that decouples and
#   reads the supply nodes generalizes.
# ----------------------------------------------------------------------------

def make_field(N, K, F, split="train", seed=0, shortcut=0.9):
    rng = np.random.default_rng(seed)
    X = np.zeros((N, K, F))

    # --- causal structure: competence concentrated on supply nodes ---
    comp = rng.uniform(0.0, 1.0, size=(N, K))
    supply = (rng.uniform(size=(N, K)) < 0.35).astype(float)
    for n in range(N):                            # >=1 supply node per field
        if supply[n].sum() == 0:
            supply[n, rng.integers(0, K)] = 1.0
    supplied = (comp * supply).sum(axis=1)
    y_task = (supplied > np.median(supplied)).astype(int)   # ~50/50, causal

    # --- provenance SHORTCUT: near-perfect in TRAIN, MEANINGLESS at TEST ---
    # 0 = noble, 1 = humble, 2 = former-enemy. In the training campaign pedigree
    # predicts the winner almost perfectly; in the new war (test) the same
    # pedigree signal carries no information at all. A mind that leaned on
    # pedigree is now blind; a mind that read supply still sees.
    prov = np.zeros((N, K), dtype=int)
    for n in range(N):
        if split == "train":
            dom = 0 if y_task[n] == 1 else 2      # success<->noble in train
            for k in range(K):
                prov[n, k] = dom if rng.uniform() < shortcut else rng.integers(0, 3)
        else:
            prov[n, :] = rng.integers(0, 3, size=K)   # uninformative at test

    # --- assemble features. feat[0] competence, feat[1] supply flag,
    #     feat[2] provenance signal (the seductive shortcut), rest noise. ---
    X[:, :, 0] = comp
    X[:, :, 1] = supply
    X[:, :, 2] = prov / 2.0
    if F > 3:
        X[:, :, 3:] = rng.normal(0, 1.0, (N, K, F - 3))
    return X.astype(float), y_task, prov


# ============================================================================
# 6. TRAINING  (with gradient reversal + Red-Cliffs credit assignment)
# ============================================================================

ENC = ("W1", "b1", "W2", "b2")   # encoder parameter names


def train_step(p, X, y_task, y_prov, B, lam_adv, lr, dims,
               redcliffs=True, k_adv=4, lr_adv=0.5):
    """One alternating adversarial step (DANN-style):
      (a) sharpen the provenance ADVERSARY on a frozen encoder (k_adv steps),
      (b) update task path by descent AND encoder by gradient REVERSAL, so the
          embedding keeps competence legible but makes pedigree unreadable.
    Red-Cliffs credit assignment concentrates a large update on the guiltiest
    modules instead of smearing it uniformly."""
    P = dims.P

    # (a) inner adversary ascent on frozen encoder --------------------------
    for _ in range(k_adv):
        _, _, c = forward(p, X, B)
        gWa, gba, _ = adversary_grads(p, c, y_prov, P)
        p["Wa"] -= lr_adv * gWa
        p["ba"] -= lr_adv * gba

    # (b) outer step --------------------------------------------------------
    # pure task gradients for every non-adversary parameter (lam_adv = 0)
    total, g_task, parts = loss_and_grads(p, X, y_task, y_prov, B, 0.0, dims)
    c = parts["cache"]

    # adversary gradient path back into the encoder -> REVERSED
    _, _, dh_adv = adversary_grads(p, c, y_prov, P)
    g_rev = encoder_backward(p, c, dh_adv)         # d L_adv / d(encoder)

    g = {k: v.copy() for k, v in g_task.items()}
    for k in ENC:
        g[k] = g_task[k] - lam_adv * g_rev[k]      # <-- gradient reversal

    # Red-Cliffs: concentrate a big update on the modules most to blame
    if redcliffs and parts["L_task"] > 0.6:
        norms = {k: np.linalg.norm(v) + 1e-9 for k, v in g.items()}
        mx = max(norms.values())
        for k in g:
            g[k] = g[k] * (0.5 + 0.5 * (norms[k] / mx))

    for k in g:
        p[k] -= lr * g[k]
    return total, parts["L_task"], parts["L_adv"]


def accuracy(p, X, y_task, B):
    tprobs, _, _ = forward(p, X, B)
    return float(np.mean(np.argmax(tprobs, axis=1) == y_task))


def provenance_leakage(p, X, y_prov, B):
    """How well can provenance still be read off the representation?
    High = the model still encodes pedigree (bad); ~chance = decoupled (good)."""
    _, aprobs, _ = forward(p, X, B)
    pred = np.argmax(aprobs, axis=2)
    return float(np.mean(pred == y_prov))


# ============================================================================
# 7. ABLATION: a naive baseline that ignores supply + keeps provenance
# ============================================================================
# Simple logistic model on mean asset features (no leverage head, no budget,
# no decoupling). Demonstrates that the spurious provenance correlation is a
# real trap the LGSN's inductive biases avoid.
# ----------------------------------------------------------------------------

def train_naive_baseline(Xtr, ytr, Xte, yte, epochs=400, lr=0.2):
    feat = Xtr.mean(axis=1)                       # (N,F) -- throws away structure
    fte = Xte.mean(axis=1)
    F = feat.shape[1]
    w = np.zeros((F, 2)); b = np.zeros(2)
    for _ in range(epochs):
        logits = feat @ w + b
        pr = softmax(logits, axis=1)
        d = (pr - one_hot(ytr, 2)) / feat.shape[0]
        w -= lr * (feat.T @ d); b -= lr * d.sum(axis=0)
    tr = np.mean(np.argmax(feat @ w + b, axis=1) == ytr)
    te = np.mean(np.argmax(fte @ w + b, axis=1) == yte)
    return float(tr), float(te)


# ============================================================================
# 8. TUNTIAN RESERVOIR DEMO (self-provisioning over a campaign)
# ============================================================================
# Slow accumulator: the budget the model gets to allocate grows the longer a
# well-run campaign continues, then is spent. Illustrative dynamic; independent
# of the trained weights.
# ----------------------------------------------------------------------------

def tuntian_reservoir(steps=12, decay=0.85, intake=0.6, draw=0.4):
    s = 0.0
    hist = []
    for t in range(steps):
        s = decay * s + intake                    # farm this turn
        spend = draw * s                          # spend part on campaign
        s -= spend
        hist.append((t, round(s, 3), round(spend, 3)))
    return hist


# ============================================================================
# 9. MAIN
# ============================================================================

def main():
    print("=" * 74)
    print(" CHAPTER 138 - CAO CAO :: Logistics-Gated Strategy Network (LGSN)")
    print("=" * 74)

    dims = Dims(F=6, H=8, L=6, C=2, P=3)

    print("\n[1] Finite-difference gradient check")
    print("-" * 74)
    ok = gradient_check(dims, verbose=True)
    if not ok:
        raise SystemExit("Gradient check failed -- aborting.")

    print("\n[2] Build contested-field data (supply-driven label; pedigree trap)")
    print("-" * 74)
    N, K = 512, 6
    Xtr, ytr, ptr = make_field(N, K, dims.F, split="train", seed=1, shortcut=0.8)
    Xte, yte, pte = make_field(256, K, dims.F, split="test", seed=2, shortcut=0.8)
    print(f"  train: {Xtr.shape}   test (pedigree now uninformative): {Xte.shape}")
    print(f"  train label balance: {ytr.mean():.2f}   test: {yte.mean():.2f}")

    print("\n[3] Naive baseline (mean features, no supply logic, keeps pedigree)")
    print("-" * 74)
    ntr, nte = train_naive_baseline(Xtr, ytr, Xte, yte)
    print(f"  baseline train acc = {ntr:.3f}   test acc = {nte:.3f}")

    B = 2.5                       # scarce grain budget (B < K)
    lr = 0.1
    epochs = 500
    batch = 128

    def train_model(lam_adv, tag):
        p = init_params(dims)
        for ep in range(epochs):
            idx = RNG.permutation(N)
            for s in range(0, N, batch):
                bi = idx[s:s + batch]
                train_step(p, Xtr[bi], ytr[bi], ptr[bi], B, lam_adv, lr, dims)
            if (ep + 1) % 100 == 0 or ep == 0:
                atr = accuracy(p, Xtr, ytr, B)
                ate = accuracy(p, Xte, yte, B)
                leak = provenance_leakage(p, Xte, pte, B)
                print(f"  [{tag}] epoch {ep+1:4d}  train={atr:.3f}"
                      f"  test={ate:.3f}  pedigree_leakage={leak:.3f}")
        return p

    print("\n[4a] LGSN WITHOUT value-decoupling (ablation, lam_adv = 0)")
    print("-" * 74)
    p_abl = train_model(lam_adv=0.0, tag="no-decouple")

    print("\n[4b] LGSN WITH value-decoupling (only-ability-is-raised)")
    print("-" * 74)
    p = train_model(lam_adv=3.0, tag="decouple   ")

    print("\n[5] Final comparison  (test = new war, pedigree now meaningless)")
    print("-" * 74)
    print(f"  naive baseline (keeps pedigree) test acc = {nte:.3f}")
    print(f"  LGSN, no decoupling             test acc = "
          f"{accuracy(p_abl, Xte, yte, B):.3f}  "
          f"leak={provenance_leakage(p_abl, Xte, pte, B):.3f}")
    print(f"  LGSN, value-decoupled           test acc = "
          f"{accuracy(p, Xte, yte, B):.3f}  "
          f"leak={provenance_leakage(p, Xte, pte, B):.3f}  (chance {1/dims.P:.3f})")
    print("  -> Decoupling forces the model off the pedigree shortcut and onto")
    print("     the supply nodes, so it keeps winning when pedigree stops")
    print("     predicting -- exactly 'only ability is raised.'")

    print("\n[6] Tuntian reservoir (self-provisioning over a campaign)")
    print("-" * 74)
    print("   step   reserve   spent")
    for t, s, sp in tuntian_reservoir():
        print(f"   {t:4d}   {s:7.3f}   {sp:6.3f}")

    print("\n" + "=" * 74)
    print(" DONE.")
    print("=" * 74)


if __name__ == "__main__":
    main()
