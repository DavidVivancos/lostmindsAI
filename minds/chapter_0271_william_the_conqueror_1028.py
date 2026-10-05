"""
CHAPTER 271 -- WILLIAM THE CONQUEROR (c.1028-1087)
The Ledger-Bond-Ratchet Network (LBRN)
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0271 · William the Conqueror (c.1028-1087)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 14, Minds 261-280 Available on Amazon https://www.amazon.com/dp/B0HLYSVY5D
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0271_william_the_conqueror_1028 - William the Conqueror (c.1028-1087)
================================================================================
WHAT THIS FILE IS
================================================================================
This is not a metaphor dressed up as code. It is a small, real, from-scratch
NumPy neural architecture whose three modules are direct computational
translations of the three cognitive habits that the documentary record
actually shows William doing, in this order, in 1085-1086-1066-1069:

  1. TOTAL-INVENTORY LEDGER  (the Domesday habit)
     Never act on a partial, locally-reported picture. Force every local
     report to be checked against a reconstructible whole, and penalise
     the model for letting any single reporting node go untrusted/uncounted.

  2. DIRECT-BOND COLLAPSE    (the Salisbury Oath habit)
     Do not govern through a chain of intermediaries. Learn which nodes in
     a multi-hop hierarchy are "of any account" and bind them *directly*
     to the centre, regardless of how many hops of vassalage separate them
     on paper. This deliberately throws away the tidy tree structure.

  3. THE SOFT RATCHET        (the Channel-crossing / Hastings habit)
     Commitment is not an average of evidence over time, it is driven by
     the single sharpest alignment of signals encountered in a waiting
     window, and past that point the decision does not relax back down.
     This is implemented as a temperature-controlled soft-max pool over
     time (a smooth surrogate of "max", not "mean") feeding a sigmoid gate.

None of this is attention-over-stored-keys, and none of it is a generic
hierarchical knowledge graph. Every layer below is one specific, historically
argued cognitive commitment, made numerically precise, made trainable, and
made falsifiable by a finite-difference gradient check.

Everything is pure NumPy. No autograd library is used anywhere. Every
backward-pass line below was derived by hand with the chain rule and is
checked against numerical differentiation in `gradient_check()`.

Run this file directly:  python3 chapter_0228_william_the_conqueror_1028_Neuron.py
================================================================================
"""

import numpy as np

RNG = np.random.default_rng(1066)


# ==============================================================================
# SECTION 0 -- SYNTHETIC "KINGDOM" DATA
# ==============================================================================
# We simulate a small kingdom of N manors organised into G groups. Each group
# has one "tenant-in-chief" node (depth = 1, a great baron) and several
# "sub-tenant manor" nodes held under him (depth = 2). This mirrors the real
# two-hop structure Domesday records: king -> tenant-in-chief -> undertenant.
#
# Each manor has a TRUE resource vector (what a perfectly faithful inquest
# would find: hides of land, plough-teams, recorded population, livestock,
# mill count) and a NOISY LOCAL REPORT (what actually arrives at court:
# a corrupted, incomplete version of the truth, exactly as jurors' sworn
# testimony, reeve's returns and circuit commissioners' notes were imperfect).

N_GROUPS = 4
MANORS_PER_GROUP = 6          # 1 tenant-in-chief (depth 1) + 5 undertenants (depth 2)
N_MANORS = N_GROUPS * MANORS_PER_GROUP   # 24
D_IN = 5     # noisy local report features
D_OUT = 5    # true resource features (same schema, ground truth)
D_HIDDEN = 16


def make_kingdom(rng):
    depth = np.ones(N_MANORS)
    group_id = np.zeros(N_MANORS, dtype=int)
    for g in range(N_GROUPS):
        base = g * MANORS_PER_GROUP
        depth[base] = 1.0                      # the tenant-in-chief
        depth[base + 1: base + MANORS_PER_GROUP] = 2.0   # undertenant manors
        group_id[base: base + MANORS_PER_GROUP] = g

    # True resource vector per manor: [hides, plough_teams, population, livestock, mills]
    # Tenants-in-chief tend to hold richer demesne land, but several
    # undertenant manors are deliberately made just as wealthy -- exactly the
    # historical situation the Salisbury Oath had to reach past: some
    # "of any account" landholders sit two hops down, not one.
    base_scale = np.where(depth == 1.0, 3.0, 1.0)[:, None]
    Y_true = rng.gamma(shape=2.2, scale=base_scale, size=(N_MANORS, D_OUT))
    # Inject three "rich undertenants" so materiality does NOT collapse onto depth.
    rich_undertenants = [1, 8, 15]
    for idx in rich_undertenants:
        Y_true[idx] *= 2.8

    # Noisy local report = true value corrupted by measurement/concealment noise.
    # Concealment is multiplicative (under-reporting), measurement noise additive.
    concealment = rng.uniform(0.55, 1.0, size=(N_MANORS, 1))
    X_noisy = Y_true * concealment + rng.normal(0, 0.35, size=(N_MANORS, D_OUT))
    X_noisy = np.clip(X_noisy, 0.0, None)

    L_true = Y_true.sum(axis=0)   # true kingdom-wide ledger total

    # Materiality / "of any account": top 40% by total true value, REGARDLESS of depth.
    value = Y_true.sum(axis=1)
    threshold = np.quantile(value, 0.60)
    B_true = (value > threshold).astype(np.float64)

    reach = 1.0 / depth   # crude fixed "distance-decay" feature: 1.0 at depth1, 0.5 at depth2

    return dict(X=X_noisy, Y_true=Y_true, L_true=L_true, depth=depth,
                reach=reach, B_true=B_true, group_id=group_id)


# ------------------------------------------------------------------------------
# Commitment sequences: T weekly observations of (wind favourability,
# enemy-exhaustion intelligence, fleet readiness, and their triple product as
# an explicit interaction feature). "Positive" sequences contain one week
# where all three genuinely align (the real narrow window that opened after
# Stamford Bridge in late Sept 1066); "negative" sequences contain noisy,
# never-jointly-aligned weeks (a campaign season that never presents a
# justified opening -- the network must learn to require joint alignment,
# not react to any single strong but isolated channel).
# ------------------------------------------------------------------------------
T_STEPS = 10
D_R = 4


def make_commitment_batch(rng, n_pos=6, n_neg=6):
    seqs, targets = [], []

    for _ in range(n_pos):
        wind = rng.uniform(-0.3, 0.4, size=T_STEPS)
        intel = rng.uniform(-0.3, 0.4, size=T_STEPS)
        fleet = rng.uniform(-0.2, 0.5, size=T_STEPS)
        peak_t = rng.integers(3, T_STEPS - 1)
        wind[peak_t], intel[peak_t], fleet[peak_t] = 0.9, 0.85, 0.8
        interact = wind * intel * fleet
        seqs.append(np.stack([wind, intel, fleet, interact], axis=1))
        targets.append(1.0)

    for _ in range(n_neg):
        wind = rng.uniform(-0.3, 0.9, size=T_STEPS)
        intel = rng.uniform(-0.3, 0.4, size=T_STEPS)
        fleet = rng.uniform(-0.3, 0.4, size=T_STEPS)
        # one channel spikes alone -- must NOT be enough on its own
        lone_t = rng.integers(0, T_STEPS)
        wind[lone_t] = 0.95
        interact = wind * intel * fleet
        seqs.append(np.stack([wind, intel, fleet, interact], axis=1))
        targets.append(0.0)

    R = np.stack(seqs, axis=0)              # (B, T, D_R)
    target = np.array(targets)               # (B,)
    perm = rng.permutation(len(target))
    return R[perm], target[perm]


# ==============================================================================
# SECTION 1 -- THE LEDGER-BOND-RATCHET NETWORK
# ==============================================================================
class LedgerBondRatchetNet:
    """
    Three modules, three shared/independent parameter groups, one hand-derived
    backward pass. See module docstring for the historical mapping.
    """

    def __init__(self, rng, d_in=D_IN, d_hidden=D_HIDDEN, d_out=D_OUT, d_r=D_R,
                 lam_cov=0.15, temperature=6.0,
                 w_ledger=1.0, w_bond=1.0, w_commit=1.0):
        s = lambda n: 1.0 / np.sqrt(n)
        self.p = {
            # --- Ledger (Domesday) module ---
            'W_enc': rng.normal(0, s(d_in), size=(d_in, d_hidden)),
            'b_enc': np.zeros(d_hidden),
            'W_dec': rng.normal(0, s(d_hidden), size=(d_hidden, d_out)),
            'b_dec': np.zeros(d_out),
            'w_g':   rng.normal(0, s(d_hidden), size=d_hidden),
            'b_g':   np.zeros(1),
            # --- Bond (Salisbury) module ---
            'W_b':   rng.normal(0, s(d_hidden + 2), size=(d_hidden + 2,)),
            'b_b':   np.zeros(1),
            # --- Ratchet (Hastings-commitment) module ---
            'W_r':   rng.normal(0, s(d_r), size=(d_r,)),
            'b_r':   np.zeros(1),
            'gain':  np.array([1.0]),
            'bias2': np.zeros(1),
        }
        self.lam_cov = lam_cov
        self.k = temperature
        self.w_ledger = w_ledger
        self.w_bond = w_bond
        self.w_commit = w_commit

    # --------------------------------------------------------------------
    def forward(self, X, Y_true, L_true, depth, reach, B_true, R, commit_target):
        p = self.p
        cache = {}

        # ---------- Ledger (Domesday) ----------
        Z1 = X @ p['W_enc'] + p['b_enc']                # (N,H)
        H = np.maximum(Z1, 0.0)                         # ReLU
        Y_hat = H @ p['W_dec'] + p['b_dec']              # (N,Dout)
        pre_g = H @ p['w_g'] + p['b_g']                  # (N,)
        g = 1.0 / (1.0 + np.exp(-pre_g))                 # coverage gate in (0,1)
        L_hat = (g[:, None] * Y_hat).sum(axis=0)         # (Dout,)

        N, Dout = Y_hat.shape
        # The kingdom-total ledger scales with N (it is a SUM over manors), so its
        # squared error must be normalised by N^2 before being added to the
        # per-manor reconstruction error -- otherwise the totalising term simply
        # swamps the gradient and the network stops bothering to get any single
        # manor right, which is exactly the failure mode Domesday's own design
        # (checking local juries against the aggregate, not just trusting the sum)
        # was built to avoid.
        NORM = float(N)
        loss_A = np.mean((Y_hat - Y_true) ** 2)
        loss_B = np.mean(((L_hat - L_true) / NORM) ** 2)
        loss_C = np.mean((1.0 - g) ** 2)
        loss_ledger = loss_A + loss_B + self.lam_cov * loss_C

        # ---------- Bond (Salisbury) ----------
        F = np.concatenate([H, depth[:, None], reach[:, None]], axis=1)   # (N,H+2)
        bond_logits = F @ p['W_b'] + p['b_b']            # (N,)
        Bhat = 1.0 / (1.0 + np.exp(-bond_logits))
        eps = 1e-9
        loss_bond = -np.mean(B_true * np.log(Bhat + eps) +
                              (1 - B_true) * np.log(1 - Bhat + eps))

        # ---------- Ratchet (commitment) ----------
        u = np.einsum('btd,d->bt', R, p['W_r']) + p['b_r']   # (B,T)
        z = np.tanh(u)
        kz = self.k * z
        m = kz.max(axis=1, keepdims=True)
        e = np.exp(kz - m)
        s = e.sum(axis=1)                                     # (B,)
        logsumexp = np.log(s) + m.squeeze(1)
        S = logsumexp / self.k                                # soft ratchet signal (B,)
        pre_c = S * p['gain'] + p['bias2']
        c = 1.0 / (1.0 + np.exp(-pre_c))
        loss_commit = -np.mean(commit_target * np.log(c + eps) +
                                (1 - commit_target) * np.log(1 - c + eps))

        total = (self.w_ledger * loss_ledger +
                 self.w_bond * loss_bond +
                 self.w_commit * loss_commit)

        cache.update(dict(X=X, Y_true=Y_true, L_true=L_true, Z1=Z1, H=H, Y_hat=Y_hat,
                           pre_g=pre_g, g=g, L_hat=L_hat, N=N, Dout=Dout,
                           depth=depth, reach=reach, F=F, bond_logits=bond_logits,
                           Bhat=Bhat, B_true=B_true,
                           R=R, u=u, z=z, e=e, s=s, S=S, pre_c=pre_c, c=c,
                           commit_target=commit_target))
        self._cache = cache
        metrics = dict(loss_total=total, loss_ledger=loss_ledger, loss_A=loss_A,
                        loss_B=loss_B, loss_C=loss_C, loss_bond=loss_bond,
                        loss_commit=loss_commit)
        return total, metrics

    # --------------------------------------------------------------------
    def backward(self):
        p, c = self.p, self._cache
        grads = {k: np.zeros_like(v) for k, v in p.items()}
        N, Dout = c['N'], c['Dout']

        # ===== Ledger branch =====
        NORM = float(N)
        dY_hat = (2.0 / (N * Dout)) * (c['Y_hat'] - c['Y_true'])            # from loss_A
        dL_hat = (2.0 / (Dout * NORM ** 2)) * (c['L_hat'] - c['L_true'])     # from loss_B
        dY_hat = dY_hat + c['g'][:, None] * dL_hat[None, :]                  # L_hat -> Y_hat
        dg = (c['Y_hat'] * dL_hat[None, :]).sum(axis=1)                      # L_hat -> g
        dg = dg + self.lam_cov * (-2.0 / N) * (1.0 - c['g'])                 # loss_C -> g

        dpre_g = dg * c['g'] * (1 - c['g'])
        grads['w_g'] = self.w_ledger * (c['H'].T @ dpre_g)
        grads['b_g'] = self.w_ledger * np.array([dpre_g.sum()])
        dH_from_g = np.outer(dpre_g, p['w_g'])

        grads['W_dec'] = self.w_ledger * (c['H'].T @ dY_hat)
        grads['b_dec'] = self.w_ledger * dY_hat.sum(axis=0)
        dH_from_dec = dY_hat @ p['W_dec'].T

        dH_ledger = self.w_ledger * (dH_from_dec + dH_from_g)

        # ===== Bond branch =====
        dbond_logits = (c['Bhat'] - c['B_true']) / N
        grads['W_b'] = self.w_bond * (c['F'].T @ dbond_logits)
        grads['b_b'] = self.w_bond * np.array([dbond_logits.sum()])
        dF = np.outer(dbond_logits, p['W_b'])
        dH_bond = self.w_bond * dF[:, :c['H'].shape[1]]

        # ===== Combine into H, then into encoder =====
        dH_total = dH_ledger + dH_bond
        dZ1 = dH_total * (c['Z1'] > 0)
        grads['W_enc'] = c['X'].T @ dZ1
        grads['b_enc'] = dZ1.sum(axis=0)

        # ===== Ratchet branch =====
        Bc = c['R'].shape[0]
        dpre_c = (c['c'] - c['commit_target']) / Bc
        grads['gain'] = self.w_commit * np.array([np.sum(dpre_c * c['S'])])
        grads['bias2'] = self.w_commit * np.array([dpre_c.sum()])
        dS = self.w_commit * dpre_c * p['gain']
        dlogsumexp = dS / self.k
        softmax_w = c['e'] / c['s'][:, None]                 # (B,T)
        d_kz = dlogsumexp[:, None] * softmax_w
        d_u = d_kz * self.k * (1.0 - c['z'] ** 2)
        grads['W_r'] = np.einsum('bt,btd->d', d_u, c['R'])
        grads['b_r'] = np.array([d_u.sum()])

        return grads

    # --------------------------------------------------------------------
    def params_flat_items(self):
        return list(self.p.items())


# ==============================================================================
# SECTION 2 -- FINITE-DIFFERENCE GRADIENT CHECK  (mandatory correctness test)
# ==============================================================================
def gradient_check(model, batch, n_checks_per_param=3, eps=1e-5, tol=2e-4):
    print("Running finite-difference gradient check ...")
    X, Y_true, L_true, depth, reach, B_true, R, commit_target = batch

    total0, _ = model.forward(X, Y_true, L_true, depth, reach, B_true, R, commit_target)
    analytic = model.backward()

    rng = np.random.default_rng(7)
    worst = 0.0
    for name, param in model.p.items():
        flat = param.reshape(-1)
        g_flat = analytic[name].reshape(-1)
        idxs = rng.choice(flat.size, size=min(n_checks_per_param, flat.size), replace=False)
        for idx in idxs:
            orig = flat[idx]

            flat[idx] = orig + eps
            lp, _ = model.forward(X, Y_true, L_true, depth, reach, B_true, R, commit_target)
            flat[idx] = orig - eps
            lm, _ = model.forward(X, Y_true, L_true, depth, reach, B_true, R, commit_target)
            flat[idx] = orig

            numeric = (lp - lm) / (2 * eps)
            analytic_val = g_flat[idx]
            denom = max(abs(numeric), abs(analytic_val), 1e-8)
            rel_err = abs(numeric - analytic_val) / denom
            worst = max(worst, rel_err)
            status = "OK " if rel_err < tol else "FAIL"
            print(f"  [{status}] {name:8s} idx={idx:4d}  analytic={analytic_val: .6e}"
                  f"  numeric={numeric: .6e}  rel_err={rel_err:.2e}")

    print(f"Worst relative error across all checked entries: {worst:.2e}"
          f"  (tolerance {tol:.1e})")
    assert worst < tol, "GRADIENT CHECK FAILED"
    print("Gradient check PASSED.\n")
    return worst


# ==============================================================================
# SECTION 3 -- ADAM OPTIMIZER (manual, no framework)
# ==============================================================================
class Adam:
    def __init__(self, params, lr=0.03, b1=0.9, b2=0.999, eps=1e-8):
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


# ==============================================================================
# SECTION 4 -- TRAINING LOOP
# ==============================================================================
def train(model, batch, epochs=600, lr=0.03, log_every=100):
    X, Y_true, L_true, depth, reach, B_true, R, commit_target = batch
    opt = Adam(model.p, lr=lr)
    history = []
    for epoch in range(1, epochs + 1):
        total, metrics = model.forward(X, Y_true, L_true, depth, reach, B_true, R, commit_target)
        grads = model.backward()
        opt.step(model.p, grads)
        history.append(metrics['loss_total'])
        if epoch % log_every == 0 or epoch == 1:
            print(f"  epoch {epoch:4d} | total {metrics['loss_total']:.4f} | "
                  f"ledger {metrics['loss_ledger']:.4f} | bond {metrics['loss_bond']:.4f} | "
                  f"commit {metrics['loss_commit']:.4f}")
    return history


# ==============================================================================
# SECTION 5 -- SELF-TESTS (behavioural, not just numerical)
# ==============================================================================
def self_tests(model, batch):
    print("\nRunning behavioural self-tests ...")
    X, Y_true, L_true, depth, reach, B_true, R, commit_target = batch
    _, metrics = model.forward(X, Y_true, L_true, depth, reach, B_true, R, commit_target)
    c = model._cache

    # Test 1: ledger reconstruction beats a naive "trust the noisy report" baseline.
    naive_mse = np.mean((X - Y_true) ** 2)
    learned_mse = np.mean((c['Y_hat'] - Y_true) ** 2)
    print(f"  [1] Naive (raw report) MSE = {naive_mse:.4f} | "
          f"Ledger-reconstructed MSE = {learned_mse:.4f}  "
          f"-> {'PASS' if learned_mse < naive_mse else 'FAIL'}")
    assert learned_mse < naive_mse

    # Test 2: bond predictions correlate with materiality, not with depth alone.
    bond_pred = (c['Bhat'] > 0.5).astype(float)
    acc = np.mean(bond_pred == B_true)
    depth2_material = B_true[depth == 2].sum()
    print(f"  [2] Direct-bond classification accuracy = {acc:.2%}  "
          f"({int(depth2_material)} depth-2 manors correctly need direct bonds)  "
          f"-> {'PASS' if acc > 0.8 else 'FAIL'}")
    assert acc > 0.8

    # Test 3: irreversibility -- appending weak signal AFTER a strong aligned
    # peak must not lower the commitment signal S (the soft-ratchet property).
    strong_seq = R[commit_target == 1.0][0:1].copy()
    weak_tail = strong_seq.copy()
    weak_tail[0, -1, :] = np.array([-0.5, -0.5, -0.5, -0.125])  # bad week appended
    def commit_signal(seq):
        u = np.einsum('btd,d->bt', seq, model.p['W_r']) + model.p['b_r']
        z = np.tanh(u)
        kz = model.k * z
        m = kz.max(axis=1, keepdims=True)
        e = np.exp(kz - m)
        s = e.sum(axis=1)
        return (np.log(s) + m.squeeze(1)) / model.k
    S_before = commit_signal(strong_seq)[0]
    S_after = commit_signal(weak_tail)[0]
    drop = S_before - S_after
    print(f"  [3] Commitment signal before bad final week = {S_before:.4f}, "
          f"after = {S_after:.4f} (drop = {drop:.4f}, ratchet tolerance < 0.05)  "
          f"-> {'PASS' if drop < 0.05 else 'FAIL'}")
    assert drop < 0.05

    # Test 4: commitment classifier separates aligned vs never-aligned sequences.
    commit_acc = np.mean((c['c'] > 0.5).astype(float) == commit_target)
    print(f"  [4] Commitment classification accuracy = {commit_acc:.2%}  "
          f"-> {'PASS' if commit_acc > 0.8 else 'FAIL'}")
    assert commit_acc > 0.8

    print("All behavioural self-tests PASSED.\n")


# ==============================================================================
# SECTION 6 -- MAIN
# ==============================================================================
if __name__ == "__main__":
    kingdom = make_kingdom(RNG)
    R, commit_target = make_commitment_batch(RNG)

    batch = (kingdom['X'], kingdom['Y_true'], kingdom['L_true'], kingdom['depth'],
             kingdom['reach'], kingdom['B_true'], R, commit_target)

    model = LedgerBondRatchetNet(RNG)

    print("=" * 78)
    print("CHAPTER 228 - WILLIAM THE CONQUEROR - Ledger-Bond-Ratchet Network")
    print("=" * 78)
    print(f"Kingdom: {N_MANORS} manors in {N_GROUPS} groups | "
          f"commitment batch: {R.shape[0]} sequences x {T_STEPS} weeks\n")

    gradient_check(model, batch)

    print("Training ...")
    train(model, batch, epochs=600, lr=0.03, log_every=100)

    self_tests(model, batch)

    total, metrics = model.forward(*batch)
    print("Final metrics:")
    for k, v in metrics.items():
        print(f"  {k:12s}: {v:.5f}")
    print("\nDone.")

    # ---- Export trained weights + synthetic kingdom for the companion
    # MindMap.html visualization (which re-runs this exact forward pass,
    # client-side in plain JavaScript, on the real trained parameters). ----
    import json
    export = {
        "weights": {k: v.tolist() for k, v in model.p.items()},
        "temperature": model.k,
        "kingdom": {
            "X": kingdom['X'].tolist(),
            "Y_true": kingdom['Y_true'].tolist(),
            "L_true": kingdom['L_true'].tolist(),
            "depth": kingdom['depth'].tolist(),
            "reach": kingdom['reach'].tolist(),
            "B_true": kingdom['B_true'].tolist(),
            "group_id": kingdom['group_id'].tolist(),
        },
        "commitment_example": {
            "R_positive": R[commit_target == 1.0][0].tolist(),
            "R_negative": R[commit_target == 0.0][0].tolist(),
        },
        "final_metrics": {k: float(v) for k, v in metrics.items()},
    }
    with open("chapter_0228_william_the_conqueror_1028_weights.json", "w") as f:
        json.dump(export, f)
    print("Exported trained weights -> chapter_0228_william_the_conqueror_1028_weights.json")
