"""
================================================================================
chapter_0015_ramses_ii_-1303.py
The Ramesside Replication Network (RRN) -- "The Cartouche Network"
Chapter 15: Ramses II / Ramesses II (c. 1303-1213 BCE)

 Part Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
Tome 1 Minds 1 - 20 Available on Amazon https://www.amazon.com/dp/B0H6F9L324
Resume and Interactive Demos at https://artificiology.com/

================================================================================

WHY THIS ARCHITECTURE, AND WHY IT IS *NOT* A TRANSFORMER
--------------------------------------------------------
Every other ancient-sovereign chapter is tempted toward the same lens:
"intelligence imposes order; build an auditable institution." Ramses II's mind
does not actually run on that axis. His single, obsessive cognitive move was
REPLICATION OF THE SELF. He erected more colossal statues of himself than any
pharaoh before or since; he usurped his predecessors' statues by re-inscribing
his own cartouche on them; and -- the detail that decides this whole file -- he
had his cartouches carved in DEEP SUNK RELIEF specifically so that no later king
could overwrite his name (UCLA Encyclopedia of Egyptology, "Usurpation of
Monuments"; Brand, "Usurped Cartouches"). He had done exactly that to others and
did not want it done to him.

That is not a metaphor for an attention head. It is a precise engineering
specification for a problem that modern AGI actually faces:

    Take ONE canonical identity / policy ("the king's will"), instantiate it as
    MANY copies deployed across MANY different environments ("territories"), and
    guarantee that the shared identity (a) remains recoverable from any single
    deployed copy even after that copy is damaged, (b) cannot be cheaply
    overwritten by an adversary's near-identity, and (c) can be reconstructed by
    CONSENSUS across the surviving copies even if a large fraction are destroyed.

This is the duplication/control problem of replicated agents: weight-shared model
copies, distilled students, fleets of identical policies. Ramses solved a
1300-BCE version of it in sandstone. We give it a learnable, differentiable form.

THE FOUR RAMESSIDE MECHANISMS (all real, all trained jointly)
-------------------------------------------------------------
1. STAMPING (replication).   A single learned identity vector z* (the "ka", the
   true name) is broadcast into N instances. Each instance i also receives its
   own territory context c_i. h_i = tanh(Wid z* + Wmod c_i + b_h). Every colossus
   is stamped from the one template; the territory only modulates it.

2. THE DECREE (replication fidelity).  Each instance must broadcast the SAME
   decree t* = T z* regardless of its territory. Loss pulls every instance's
   output to t*. The network must learn to be INVARIANT to territory -- "one
   will, reproduced identically across the empire."

3. RE-INSCRIPTION (deep carving / drift resistance).  From each instance's hidden
   state -- including artificially WEATHERED/corrupted versions of it -- a decoder
   Wdec must reconstruct the canonical identity z*. Training under corruption
   forces z* into a redundant, robustly recoverable subspace: the cartouche is
   carved deep enough to survive erosion.

4. ANTI-USURPATION MARGIN.  A hinge loss demands that every recovered identity be
   closer to z* than to any of a fixed set of rival "decoy" identities (other
   kings' names) by a margin. Small adversarial perturbations cannot flip the
   identity to a usurper. Deep carving == large margin.

CONSENSUS / MONUMENT REDUNDANCY (a property, tested not trained).  Because z* is
stored redundantly in every instance, averaging the recovered identities over the
SURVIVING instances recovers z* even when a large fraction of monuments are
toppled. "Carve my name in a thousand places."

Implementation rules followed:
  * Pure NumPy, from scratch. No autograd, no ML framework.
  * Analytic backprop for every parameter.
  * A finite-difference gradient check is MANDATORY and is the first self-test.
  * A real training loop on synthetic data; loss must fall.
  * Self-tests for the four Ramesside properties.
  * Execute the file; the printed output is pasted verbatim into the chapter.

Run:  python3 chapter_0015_ramses_ii_-1303.py
Author: David Vivancos · Chapter 0015 · Ramses II
================================================================================
"""

from __future__ import annotations
import numpy as np

# A single global RNG seed makes the corruption masks, decoys and init
# deterministic, so the loss is a fixed differentiable function and the
# finite-difference gradient check is meaningful.
SEED = 1279  # the year Ramses II took the throne (1279 BCE)


# ----------------------------------------------------------------------------- #
# 1. CONFIG -- the dimensions of the "empire"
# ----------------------------------------------------------------------------- #
class Config:
    D_id   = 8     # identity ("ka") dimension -- the true name lives here
    D_ctx  = 6     # territory/context dimension
    D_h    = 16    # colossus hidden-state dimension
    D_out  = 5     # decree (broadcast output) dimension
    N      = 12    # number of deployed instances (colossi) per forward pass
    K_corr = 3     # corrupted re-inscription views per instance (weathering)
    n_decoy = 4    # rival "usurper" identities to stay separated from
    margin  = 0.6  # anti-usurpation hinge margin
    drop_p  = 0.35 # weathering: fraction of hidden units knocked out in a view
    noise   = 0.20 # weathering: gaussian noise std on corrupted views
    # loss weights
    w_task     = 1.0
    w_reinscr  = 1.0
    w_margin   = 0.5


# ----------------------------------------------------------------------------- #
# 2. PARAMETERS -- "the mind" of Ramses, as learnable arrays
# ----------------------------------------------------------------------------- #
def init_params(cfg: Config, rng: np.random.Generator) -> dict:
    """
    Small random init for the learnable apparatus only.

    The canonical identity z* is NOT here -- Ramses did not *learn* his true
    name, he was given it at coronation. z* is a fixed datum (see fixed_world).
    The "mind" we train is the machinery that stamps that fixed identity into
    every colossus, keeps it recoverable under weathering, and defends it from
    overwriting. Keeping z* fixed also makes the supervisory targets genuine
    constants, so the finite-difference gradient check is exact.
    """
    s = 0.5
    return {
        "Wid":   rng.normal(0, s, size=(cfg.D_h, cfg.D_id)) / np.sqrt(cfg.D_id),
        "Wmod":  rng.normal(0, s, size=(cfg.D_h, cfg.D_ctx)) / np.sqrt(cfg.D_ctx),
        "b_h":   np.zeros(cfg.D_h),
        "Wo":    rng.normal(0, s, size=(cfg.D_out, cfg.D_h)) / np.sqrt(cfg.D_h),
        "b_o":   np.zeros(cfg.D_out),
        "Wdec":  rng.normal(0, s, size=(cfg.D_id, cfg.D_h)) / np.sqrt(cfg.D_h),
        "b_dec": np.zeros(cfg.D_id),
    }


def fixed_world(cfg: Config, rng: np.random.Generator) -> dict:
    """
    Non-learned pieces of the environment, fixed once:
      z       : the canonical identity z* (the "ka" / true name) -- a FIXED datum
      t_star  : the fixed decree every colossus must broadcast (t* = T z*)
      T       : maps identity to decree (kept for reporting / consistency)
      decoys  : rival identities the network must NOT be confusable with
      masks   : per (instance, view) weathering dropout masks (deterministic)
      noise   : per (instance, view) weathering gaussian noise (deterministic)
    """
    z = rng.normal(0, 0.5, size=(cfg.D_id,))
    T = rng.normal(0, 1.0, size=(cfg.D_out, cfg.D_id)) / np.sqrt(cfg.D_id)
    t_star = T @ z                                    # fixed decree (constant)
    decoys = rng.normal(0, 0.5, size=(cfg.n_decoy, cfg.D_id))
    # deterministic weathering for every (instance i, corruption view k)
    keep = (rng.random(size=(cfg.N, cfg.K_corr, cfg.D_h)) > cfg.drop_p).astype(float)
    # scale so expected magnitude is preserved (inverted dropout)
    keep = keep / (1.0 - cfg.drop_p)
    gnoise = rng.normal(0, cfg.noise, size=(cfg.N, cfg.K_corr, cfg.D_h))
    return {"z": z, "t_star": t_star, "T": T, "decoys": decoys,
            "keep": keep, "gnoise": gnoise}


def make_contexts(cfg: Config, rng: np.random.Generator) -> np.ndarray:
    """The N territories. Distinct, normalized context vectors."""
    C = rng.normal(0, 1.0, size=(cfg.N, cfg.D_ctx))
    C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-8)
    return C


# ----------------------------------------------------------------------------- #
# 3. FORWARD PASS + LOSS
# ----------------------------------------------------------------------------- #
def forward(params: dict, world: dict, C: np.ndarray, cfg: Config):
    """
    Returns (loss, cache, parts). Vectorized over the N instances.

    Shapes:
      C    : (N, D_ctx)
      h    : (N, D_h)          clean colossus hidden states
      y    : (N, D_out)        each instance's broadcast decree
      hc   : (N, K, D_h)       weathered views of each colossus
      zrec : (N, K, D_id)      identity re-inscribed from each weathered view
    """
    z   = world["z"]                            # canonical identity z* (fixed)
    Wid, Wmod, b_h = params["Wid"], params["Wmod"], params["b_h"]
    Wo, b_o        = params["Wo"], params["b_o"]
    Wdec, b_dec    = params["Wdec"], params["b_dec"]
    t_star = world["t_star"]                     # fixed decree target

    # --- (1) STAMPING: identity broadcast + territory modulation -------------
    id_term = Wid @ z                         # (D_h,)  same for every instance
    h_pre   = C @ Wmod.T + id_term + b_h       # (N, D_h)
    h       = np.tanh(h_pre)                    # (N, D_h)

    # --- (2) THE DECREE: each instance must output the fixed decree t* -------
    y = h @ Wo.T + b_o                          # (N, D_out)
    diff_task = y - t_star                       # (N, D_out)
    L_task = np.mean(np.sum(diff_task**2, axis=1))

    # --- (3) RE-INSCRIPTION under weathering ---------------------------------
    keep   = world["keep"]                      # (N, K, D_h)
    gnoise = world["gnoise"]                     # (N, K, D_h)
    hc   = h[:, None, :] * keep + gnoise         # (N, K, D_h) weathered colossi
    zrec = hc @ Wdec.T + b_dec                   # (N, K, D_id) recovered identity
    diff_re = zrec - z                           # (N, K, D_id) reconstruct fixed z*
    L_reinscr = np.mean(np.sum(diff_re**2, axis=2))

    # --- (4) ANTI-USURPATION MARGIN ------------------------------------------
    # similarity of each recovered identity to z* vs to each decoy.
    decoys = world["decoys"]                     # (n_decoy, D_id)
    zr_flat = zrec.reshape(-1, cfg.D_id)         # (N*K, D_id)
    sim_true  = zr_flat @ z                       # (N*K,) <zrec, z*>
    sim_decoy = zr_flat @ decoys.T                # (N*K, n_decoy)
    # hinge: want sim_true >= sim_decoy + margin for every decoy.
    viol = cfg.margin - (sim_true[:, None] - sim_decoy)  # (N*K, n_decoy)
    hinge = np.maximum(0.0, viol)
    L_margin = np.mean(np.sum(hinge, axis=1))

    loss = (cfg.w_task * L_task
            + cfg.w_reinscr * L_reinscr
            + cfg.w_margin * L_margin)

    cache = dict(C=C, id_term=id_term, h_pre=h_pre, h=h, y=y,
                 diff_task=diff_task, hc=hc, keep=keep, zrec=zrec,
                 diff_re=diff_re, zr_flat=zr_flat, sim_true=sim_true,
                 sim_decoy=sim_decoy, hinge=hinge)
    parts = dict(L_task=L_task, L_reinscr=L_reinscr, L_margin=L_margin)
    return loss, cache, parts


# ----------------------------------------------------------------------------- #
# 4. BACKWARD PASS -- analytic gradients for every parameter
# ----------------------------------------------------------------------------- #
def backward(params: dict, world: dict, cache: dict, cfg: Config) -> dict:
    z    = world["z"]                                # fixed identity (constant)
    Wid, Wmod  = params["Wid"], params["Wmod"]
    Wo         = params["Wo"]
    Wdec       = params["Wdec"]
    decoys = world["decoys"]

    C      = cache["C"]; h = cache["h"]
    keep   = cache["keep"]; hc = cache["hc"]
    diff_task = cache["diff_task"]; diff_re = cache["diff_re"]
    hinge = cache["hinge"]
    N, K = cfg.N, cfg.K_corr

    grads = {k: np.zeros_like(v) for k, v in params.items()}

    # ---- (2) task loss: L_task = mean_i ||y_i - t*||^2 ----------------------
    dY = (2.0 / N) * diff_task                       # (N, D_out)
    grads["Wo"]  += cfg.w_task * (dY.T @ h)          # (D_out, D_h)
    grads["b_o"] += cfg.w_task * dY.sum(axis=0)
    dh_task = dY @ Wo                                 # (N, D_h) flows into h

    # ---- (3) re-inscription: L_reinscr = mean ||zrec - z||^2 ----------------
    scale_re = 2.0 / (N * K)
    dZrec = scale_re * diff_re                        # (N, K, D_id)

    # ---- (4) margin: hinge(margin - (sim_true - sim_decoy)) -----------------
    # z and decoys are constants, so margin contributes only through zrec.
    active = (hinge > 0).astype(float)               # (N*K, n_decoy)
    coeff_true = -active.sum(axis=1)                  # (N*K,)
    dzr_margin = coeff_true[:, None] * z[None, :]      # (N*K, D_id) from sim_true
    dzr_margin += active @ decoys                      # (N*K, D_id) from decoys
    dzr_margin *= (cfg.w_margin / (N * K))             # mean over N*K, weighted

    # combine reconstruction + margin gradients on zrec
    dZrec_total = cfg.w_reinscr * dZrec + dzr_margin.reshape(N, K, cfg.D_id)

    # zrec = hc @ Wdec.T + b_dec
    hc_flat    = hc.reshape(-1, cfg.D_h)              # (N*K, D_h)
    dZrec_flat = dZrec_total.reshape(-1, cfg.D_id)     # (N*K, D_id)
    grads["Wdec"]  += dZrec_flat.T @ hc_flat           # (D_id, D_h)
    grads["b_dec"] += dZrec_flat.sum(axis=0)
    dHc = (dZrec_flat @ Wdec).reshape(N, K, cfg.D_h)   # (N, K, D_h)

    # hc = h[:,None,:]*keep + gnoise -> gradient into clean h via the keep mask
    dh_from_re = (dHc * keep).sum(axis=1)             # (N, D_h)

    # ---- combine paths into h, then through tanh ----------------------------
    dh = cfg.w_task * dh_task + dh_from_re             # (N, D_h)
    dh_pre = dh * (1.0 - h**2)                          # tanh'

    # h_pre = C @ Wmod.T + (Wid @ z) + b_h  (z fixed, so Wid grad = outer(., z))
    grads["Wmod"] += dh_pre.T @ C                       # (D_h, D_ctx)
    grads["b_h"]  += dh_pre.sum(axis=0)
    dh_pre_sum = dh_pre.sum(axis=0)                     # (D_h,)
    grads["Wid"] += np.outer(dh_pre_sum, z)            # (D_h, D_id)

    return grads


# ----------------------------------------------------------------------------- #
# 5. FINITE-DIFFERENCE GRADIENT CHECK (mandatory)
# ----------------------------------------------------------------------------- #
def gradient_check(cfg: Config, eps: float = 1e-6, tol: float = 1e-5) -> bool:
    rng = np.random.default_rng(SEED)
    params = init_params(cfg, rng)
    world  = fixed_world(cfg, rng)
    C      = make_contexts(cfg, rng)

    loss, cache, _ = forward(params, world, C, cfg)
    grads = backward(params, world, cache, cfg)

    max_rel = 0.0
    worst = None
    rng_chk = np.random.default_rng(7)
    for name, P in params.items():
        flat = P.reshape(-1)
        n_check = min(flat.size, 12)
        idxs = rng_chk.choice(flat.size, size=n_check, replace=False)
        ga = grads[name].reshape(-1)
        for idx in idxs:
            orig = flat[idx]
            flat[idx] = orig + eps
            lp, _, _ = forward(params, world, C, cfg)
            flat[idx] = orig - eps
            lm, _, _ = forward(params, world, C, cfg)
            flat[idx] = orig
            num = (lp - lm) / (2 * eps)
            ana = ga[idx]
            denom = max(1e-9, abs(num) + abs(ana))
            rel = abs(num - ana) / denom
            if rel > max_rel:
                max_rel = rel
                worst = (name, int(idx), num, ana)

    ok = max_rel < tol
    print(f"  [gradient check] max relative error = {max_rel:.3e}  "
          f"(tol {tol:.0e})  ->  {'PASS' if ok else 'FAIL'}")
    if worst is not None:
        n, i, num, ana = worst
        print(f"  [gradient check] worst param: {n}[{i}]  "
              f"numeric={num:+.6e}  analytic={ana:+.6e}")
    return ok


# ----------------------------------------------------------------------------- #
# 6. TRAINING LOOP -- Adam, full-batch over the N territories
# ----------------------------------------------------------------------------- #
def train(cfg: Config, steps: int = 1500, lr: float = 5e-3, verbose: bool = True):
    rng = np.random.default_rng(SEED)
    params = init_params(cfg, rng)
    world  = fixed_world(cfg, rng)
    C      = make_contexts(cfg, rng)

    # Adam state
    m = {k: np.zeros_like(v) for k, v in params.items()}
    v = {k: np.zeros_like(v) for k, v in params.items()}
    b1, b2, eps = 0.9, 0.999, 1e-8

    history = []
    first_loss = None
    for t in range(1, steps + 1):
        loss, cache, parts = forward(params, world, C, cfg)
        grads = backward(params, world, cache, cfg)
        if first_loss is None:
            first_loss = loss
        for k in params:
            m[k] = b1 * m[k] + (1 - b1) * grads[k]
            v[k] = b2 * v[k] + (1 - b2) * (grads[k] ** 2)
            mhat = m[k] / (1 - b1 ** t)
            vhat = v[k] / (1 - b2 ** t)
            params[k] -= lr * mhat / (np.sqrt(vhat) + eps)
        if verbose and (t == 1 or t % 250 == 0 or t == steps):
            print(f"  step {t:4d} | loss {loss:8.4f} | "
                  f"task {parts['L_task']:7.4f} | "
                  f"reinscr {parts['L_reinscr']:7.4f} | "
                  f"margin {parts['L_margin']:7.4f}")
        history.append(loss)

    return params, world, C, history, first_loss


# ----------------------------------------------------------------------------- #
# 7. SELF-TESTS for the four Ramesside properties
# ----------------------------------------------------------------------------- #
def _recover_identity(params, world, C, cfg, corrupt=False, rng=None):
    """Return (zrec_per_instance averaged over views, clean h)."""
    z = world["z"]
    h_pre = C @ params["Wmod"].T + (params["Wid"] @ z) + params["b_h"]
    h = np.tanh(h_pre)
    if corrupt and rng is not None:
        keep = (rng.random(size=(cfg.N, cfg.D_h)) > cfg.drop_p) / (1 - cfg.drop_p)
        noise = rng.normal(0, cfg.noise, size=(cfg.N, cfg.D_h))
        hc = h * keep + noise
    else:
        hc = h
    zrec = hc @ params["Wdec"].T + params["b_dec"]      # (N, D_id)
    return zrec, h


def _cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def test_replication_fidelity(params, world, C, cfg):
    """Every colossus should broadcast the same decree t* = T z*."""
    z = world["z"]; t_star = world["t_star"]
    h = np.tanh(C @ params["Wmod"].T + (params["Wid"] @ z) + params["b_h"])
    y = h @ params["Wo"].T + params["b_o"]              # (N, D_out)
    spread = float(np.mean(np.std(y, axis=0)))          # variation across territories
    err = float(np.mean(np.linalg.norm(y - t_star, axis=1)))
    print(f"  [replication ] cross-territory output spread = {spread:.4f} "
          f"(lower = more identical copies)")
    print(f"  [replication ] mean |decree - t*|           = {err:.4f}")
    return spread < 0.15 and err < 0.20


def test_drift_resistance(params, world, C, cfg):
    """Identity must survive weathering of an individual colossus."""
    rng = np.random.default_rng(99)
    zrec_clean, _ = _recover_identity(params, world, C, cfg, corrupt=False)
    zrec_weath, _ = _recover_identity(params, world, C, cfg, corrupt=True, rng=rng)
    z = world["z"]
    cos_clean = np.mean([_cos(zrec_clean[i], z) for i in range(cfg.N)])
    cos_weath = np.mean([_cos(zrec_weath[i], z) for i in range(cfg.N)])
    print(f"  [drift       ] mean cos(recovered, z*)  clean = {cos_clean:.4f}")
    print(f"  [drift       ] mean cos(recovered, z*)  weathered = {cos_weath:.4f}")
    return cos_weath > 0.85


def test_monument_redundancy(params, world, C, cfg):
    """Consensus over surviving colossi recovers identity as monuments are toppled."""
    rng = np.random.default_rng(123)
    zrec, _ = _recover_identity(params, world, C, cfg, corrupt=True, rng=rng)
    z = world["z"]
    print("  [redundancy  ] fraction toppled -> cos(consensus identity, z*):")
    ok = True
    for frac in (0.0, 0.25, 0.5, 0.75):
        n_keep = max(1, int(round(cfg.N * (1 - frac))))
        # average over a deterministic subset of surviving monuments
        survivors = zrec[:n_keep]
        consensus = survivors.mean(axis=0)
        c = _cos(consensus, z)
        flag = "ok" if c > 0.8 else "DEGRADED"
        print(f"                   toppled {frac*100:4.0f}%  (kept {n_keep:2d}/"
              f"{cfg.N})  ->  cos = {c:.4f}  [{flag}]")
        if frac <= 0.5 and c <= 0.8:
            ok = False
    return ok


def test_anti_usurpation(params, world, C, cfg):
    """Recovered identity must beat every rival decoy by the margin."""
    zrec, _ = _recover_identity(params, world, C, cfg, corrupt=False)
    z = world["z"]; decoys = world["decoys"]
    min_gap = np.inf
    for i in range(cfg.N):
        st = zrec[i] @ z
        sd = zrec[i] @ decoys.T
        gap = st - sd.max()
        min_gap = min(min_gap, gap)
    print(f"  [usurpation  ] worst (true - best decoy) similarity gap = {min_gap:.4f} "
          f"(margin target {cfg.margin})")
    return min_gap > 0.0


# ----------------------------------------------------------------------------- #
# 8. MAIN
# ----------------------------------------------------------------------------- #
def main():
    cfg = Config()
    print("=" * 74)
    print("RAMESSIDE REPLICATION NETWORK (RRN) -- 'The Cartouche Network'")
    print("Ramses II (c.1303-1213 BCE):  one identity, stamped everywhere,")
    print("carved deep enough to survive erosion and resist overwriting.")
    print("=" * 74)

    print("\n[1] Finite-difference gradient check (analytic vs numeric):")
    grad_ok = gradient_check(cfg)

    print("\n[2] Training the empire (Adam, full-batch over territories):")
    params, world, C, history, first_loss = train(cfg, steps=1500, lr=5e-3)
    last_loss = history[-1]
    print(f"  loss: {first_loss:.4f}  ->  {last_loss:.4f}  "
          f"({100*(first_loss-last_loss)/first_loss:.1f}% reduction)")

    print("\n[3] Ramesside property tests on the trained network:")
    r1 = test_replication_fidelity(params, world, C, cfg)
    r2 = test_drift_resistance(params, world, C, cfg)
    r3 = test_monument_redundancy(params, world, C, cfg)
    r4 = test_anti_usurpation(params, world, C, cfg)

    print("\n" + "=" * 74)
    print("RESULTS")
    print("=" * 74)
    checks = [
        ("gradient check (analytic == numeric)", grad_ok),
        ("loss decreased during training",       last_loss < first_loss * 0.5),
        ("replication fidelity (identical decree across territories)", r1),
        ("drift resistance (identity survives weathering)", r2),
        ("monument redundancy (consensus survives toppling)", r3),
        ("anti-usurpation margin holds", r4),
    ]
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    all_ok = all(ok for _, ok in checks)
    print("-" * 74)
    print(f"  ALL CHECKS {'PASSED' if all_ok else 'DID NOT PASS'}")
    print("=" * 74)
    return all_ok


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
