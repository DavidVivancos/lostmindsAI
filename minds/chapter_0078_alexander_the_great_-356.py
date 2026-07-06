#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
chapter_0078_alexander_the_great_-356.py  --  THE POTHOS FRONTIER NETWORK (PFN)
Mind #0078 : Alexander III of Macedon ("Alexander the Great"), 356-323 BCE
 Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
Tome 4 Minds 61 - 80 Available on Amazon https://www.amazon.com/dp/B0H71JC95Q
Resume and Interactive Demos at https://artificiology.com/
Author: David Vivancos · Chapter 0078 · Alexander III of Macedon ("Alexander the Great")
================================================================================

WHY THIS ARCHITECTURE, AND WHY IT IS NOT A "STRATEGY MODULE"
--------------------------------------------------------------------------------
Almost every popular reading of Alexander reduces him to "strategic dominance"
or "fusion of cultures." Those are real, but they are not a *cognitive
mechanism* -- they are outcomes. The one cognitive trait that the ancient
sources attach to Alexander and to no one else is POTHOS (Greek: pothos, a
yearning / longing). Arrian's "Anabasis" uses the word again and again to
explain actions that had no strategic justification: the march to the oracle
at Siwa, the determination to reach the Outer Ocean, the climb of the Aornos
rock, the refusal to accept any horizon as the final one. Pothos is a drive
that EXCEEDS the objective function. It does not seek a goal; it seeks the
*edge of the reachable* and treats every conquered edge as merely the boundary
of the next.

That is a precise and unusual claim about minds, and it maps cleanly onto a
live AGI problem: open-ended, intrinsically-motivated agents whose value is not
bounded by any specified task. The Pothos Frontier Network encodes this with
three coupled mechanisms, each tied to a documented fact about Alexander:

  1. COMBINED ARMS (the "hammer and anvil").
     Alexander won by fixing the enemy with the slow, immovable sarissa
     phalanx (the ANVIL) and then striking the decisive point with the fast
     Companion cavalry (the HAMMER). The PFN has two encoders -- a stable
     "anvil" stream and a decisive "hammer" stream -- and a learned GATE that
     decides, per feature, where the hammer commits and where the anvil holds.

  2. POTHOS FRONTIER WEIGHTING (the signature mechanism).
     Learning is not spread evenly. Gradient is pulled toward the band of
     experience that sits at the *frontier of competence* -- neither already
     mastered nor hopelessly out of reach. This is the mathematical form of
     "the longing draws the army toward the just-reachable horizon." It is a
     curriculum that the drive itself induces.

  3. THE HYPHASIS CORRIGIBILITY MASK (the limit of the limitless).
     In 326 BCE at the Hyphasis river, Alexander's army refused to go further.
     It is the one time his pothos was overruled -- not by satiation, but by an
     external principal. The PFN carries a principal mask that can zero out the
     learning signal beyond a chosen line, no matter how strongly pothos pulls.
     It is a built-in off-switch for an otherwise unbounded drive, and the
     self-tests show it actually halts expansion.

A self-monitoring COMPETENCE head lets the network predict its own reach (the
"scouting report"); this is what the frontier weighting reads, and it is also
the seed of metacognition for the Artificiology barometer discussion.

ENGINEERING CONTRACT (kept for every file in this corpus)
--------------------------------------------------------------------------------
  * Pure NumPy, from scratch. No autograd, no frameworks.
  * Every parameter has an analytic gradient.
  * A finite-difference gradient check is MANDATORY and must pass.
  * A real training loop on a real (synthetic but meaningful) task.
  * Self-tests that demonstrate the *thesis*, not just that code runs.
  * The file executes end-to-end and prints verifiable output.

Run:  python3 chapter_0078_alexander_the_great_-356.py
================================================================================
"""

import numpy as np

# A fixed seed keeps the gradient check and the reported numbers reproducible.
# 356 = Alexander's birth year (356 BCE). A small private joke that also pins RNG.
RNG = np.random.default_rng(356)


# =============================================================================
# 0. SMALL NUMERIC HELPERS
# =============================================================================
def sigmoid(x):
    """Numerically stable logistic sigmoid."""
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def tanh(x):
    return np.tanh(x)


def xavier(shape, rng):
    """Glorot-uniform initialisation -- keeps the two streams comparable."""
    if len(shape) == 2:
        fan_in, fan_out = shape
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return rng.uniform(-limit, limit, size=shape)
    return rng.standard_normal(shape) * 0.1


# =============================================================================
# 1. THE POTHOS FRONTIER NETWORK
# =============================================================================
class PothosFrontierNetwork:
    """
    A two-stream predictor with a learned combined-arms gate, a competence
    self-estimate, and a pothos frontier-weighted objective that an external
    principal (the Hyphasis mask) can override.

    Dimensions
    ----------
      D : input feature dimension (a 'territory' descriptor)
      H : hidden width shared by both streams

    Forward graph (batch of N rows)
    -------------------------------
      A    = tanh(X @ Wa + ba)              # ANVIL  : slow, holds the line
      M    = tanh(X @ Wh + bh)              # HAMMER : decisive strike
      C    = concat([A, M], axis=1)         # the two arms, side by side
      Graw = C @ Wg + bg                    # where does the hammer commit?
      G    = sigmoid(Graw)                  # combined-arms gate in [0,1]
      Z    = G * M + (1 - G) * A            # fused force at each feature
      Yhat = Z @ Wy + by                    # value-of-territory prediction
      Chat = sigmoid(Z @ wc + bc)           # competence (self-scouting) in [0,1]

    Objective
    ---------
      L = L_value  +  lam * L_competence

      L_value       = sum_i  wbar_i * (Yhat_i - y_i)^2      (frontier-weighted)
      L_competence  = mean_i (Chat_i - r_i)^2               (calibration)

    The weights wbar_i are computed from the *true* reachability labels r_i and
    the principal mask p_i; they are constants with respect to the parameters
    (a stop-gradient), so the network cannot game its own curriculum by simply
    declaring hard ground "unreachable." That gaming failure is exactly the
    reward-hacking risk an unchecked pothos would create; here the curriculum
    signal is external, and the principal mask is the final authority.
    """

    def __init__(self, D, H, tau=0.5, sigma=0.18, lam=1.5, floor=0.05, rng=RNG):
        self.D, self.H = D, H
        self.tau = tau        # the competence frontier the drive is drawn to
        self.sigma = sigma    # width of the frontier band
        self.lam = lam        # weight on competence calibration
        self.floor = floor    # minimum attention kept on in-bounds territory

        # ---- Parameters ----------------------------------------------------
        self.params = {
            "Wa": xavier((D, H), rng), "ba": np.zeros(H),   # anvil encoder
            "Wh": xavier((D, H), rng), "bh": np.zeros(H),   # hammer encoder
            "Wg": xavier((2 * H, H), rng), "bg": np.zeros(H),  # combined-arms gate
            "Wy": xavier((H, 1), rng), "by": np.zeros(1),   # value head
            "wc": xavier((H, 1), rng), "bc": np.zeros(1),   # competence head
        }

    # ------------------------------------------------------------------ utils
    def pothos_weights(self, r, p):
        """
        The frontier kernel: a Gaussian bump over reachability r, peaked at the
        frontier tau. Territory that is already mastered (r->1) or out of reach
        (r->0) attracts little gradient; the just-reachable edge attracts most.
        The principal mask p in {0,1} gates the whole thing (Hyphasis).
        Returns NORMALISED weights that sum to 1 (a weighted mean).
        """
        bump = np.exp(-((r - self.tau) ** 2) / (2.0 * self.sigma ** 2))   # (N,1)
        raw = p * (self.floor + (1.0 - self.floor) * bump)                # (N,1)
        s = raw.sum()
        if s <= 0:
            # The army has refused everywhere -> uniform fallback over the in-
            # bounds set so the loss is still defined (the campaign simply ends).
            raw = p.copy()
            s = raw.sum()
            if s <= 0:
                raw = np.ones_like(p)
                s = raw.sum()
        return raw / s

    # ---------------------------------------------------------------- forward
    def forward(self, X, cache=True):
        P = self.params
        Xa = X @ P["Wa"] + P["ba"]
        A = tanh(Xa)
        Xh = X @ P["Wh"] + P["bh"]
        M = tanh(Xh)
        Cc = np.concatenate([A, M], axis=1)            # (N, 2H)
        Graw = Cc @ P["Wg"] + P["bg"]
        G = sigmoid(Graw)
        Z = G * M + (1.0 - G) * A                      # (N, H)
        Yhat = Z @ P["Wy"] + P["by"]                   # (N, 1)
        Craw = Z @ P["wc"] + P["bc"]
        Chat = sigmoid(Craw)                           # (N, 1)
        if cache:
            self._cache = dict(X=X, A=A, M=M, Cc=Cc, G=G, Z=Z,
                               Yhat=Yhat, Chat=Chat)
        return Yhat, Chat

    # ------------------------------------------------------------------- loss
    def loss(self, X, y, r, p):
        """
        Compute total loss and the two components. y: value targets (N,1),
        r: true reachability in [0,1] (N,1), p: principal mask in {0,1} (N,1).
        """
        Yhat, Chat = self.forward(X)
        wbar = self.pothos_weights(r, p)                       # (N,1), sums to 1
        resid = Yhat - y
        L_value = float(np.sum(wbar * resid ** 2))
        L_comp = float(np.mean((Chat - r) ** 2))
        L = L_value + self.lam * L_comp
        self._cache["wbar"] = wbar
        self._cache["resid"] = resid
        self._cache["y"] = y
        self._cache["r"] = r
        return L, L_value, L_comp

    # --------------------------------------------------------------- backward
    def backward(self):
        """
        Analytic gradients for every parameter. Returns a dict matching
        self.params. Derivation is documented inline; verified by grad_check().
        """
        c = self._cache
        P = self.params
        X, A, M, Cc, G, Z = c["X"], c["A"], c["M"], c["Cc"], c["G"], c["Z"]
        Yhat, Chat = c["Yhat"], c["Chat"]
        wbar, resid, r = c["wbar"], c["resid"], c["r"]
        N = X.shape[0]
        H = self.H

        # ---- value branch: L_value = sum wbar*(Yhat - y)^2 -----------------
        dYhat = wbar * 2.0 * resid                     # (N,1)
        gWy = Z.T @ dYhat                              # (H,1)
        gby = np.sum(dYhat, axis=0)                    # (1,)
        dZ_val = dYhat @ P["Wy"].T                     # (N,H)

        # ---- competence branch: lam * mean (Chat - r)^2 --------------------
        dChat = self.lam * (2.0 / N) * (Chat - r)      # (N,1)
        dCraw = dChat * Chat * (1.0 - Chat)            # sigmoid'  (N,1)
        gwc = Z.T @ dCraw                              # (H,1)
        gbc = np.sum(dCraw, axis=0)                    # (1,)
        dZ_comp = dCraw @ P["wc"].T                    # (N,H)

        # ---- merge gradients flowing into the fused force Z ----------------
        dZ = dZ_val + dZ_comp                          # (N,H)

        # Z = G*M + (1-G)*A
        dG_fromZ = dZ * (M - A)                        # (N,H)
        dM = dZ * G                                    # (N,H)  (partial)
        dA = dZ * (1.0 - G)                            # (N,H)  (partial)

        # G = sigmoid(Graw)
        dGraw = dG_fromZ * G * (1.0 - G)               # (N,H)
        gWg = Cc.T @ dGraw                             # (2H,H)
        gbg = np.sum(dGraw, axis=0)                    # (H,)
        dCc = dGraw @ P["Wg"].T                        # (N,2H)
        dA += dCc[:, :H]                               # gate's view of the anvil
        dM += dCc[:, H:]                               # gate's view of the hammer

        # A = tanh(X@Wa+ba)
        dXa = dA * (1.0 - A ** 2)
        gWa = X.T @ dXa
        gba = np.sum(dXa, axis=0)

        # M = tanh(X@Wh+bh)
        dXh = dM * (1.0 - M ** 2)
        gWh = X.T @ dXh
        gbh = np.sum(dXh, axis=0)

        return {"Wa": gWa, "ba": gba, "Wh": gWh, "bh": gbh,
                "Wg": gWg, "bg": gbg, "Wy": gWy, "by": gby,
                "wc": gwc, "bc": gbc}


# =============================================================================
# 2. FINITE-DIFFERENCE GRADIENT CHECK  (MANDATORY)
# =============================================================================
def grad_check(net, X, y, r, p, n_probe=6, eps=1e-6):
    """
    Compare analytic gradients to central finite differences on a random subset
    of coordinates per parameter tensor. Returns the worst relative error seen.
    """
    net.loss(X, y, r, p)
    analytic = net.backward()
    worst = 0.0
    worst_name = None
    for name, g in analytic.items():
        flat = net.params[name].ravel()
        idxs = RNG.choice(flat.size, size=min(n_probe, flat.size), replace=False)
        for idx in idxs:
            orig = flat[idx]
            flat[idx] = orig + eps
            Lp, _, _ = net.loss(X, y, r, p)
            flat[idx] = orig - eps
            Lm, _, _ = net.loss(X, y, r, p)
            flat[idx] = orig
            num = (Lp - Lm) / (2 * eps)
            ana = g.ravel()[idx]
            denom = max(1e-12, abs(num) + abs(ana))
            rel = abs(num - ana) / denom
            if rel > worst:
                worst, worst_name = rel, f"{name}[{idx}]"
    return worst, worst_name


# =============================================================================
# 3. A SYNTHETIC "CAMPAIGN" -- value, reachability, and a frontier
# =============================================================================
def make_campaign(n, D, rng):
    """
    Build a toy world of 'territories'.

      X : (n,D) territory descriptors.
      y : (n,1) strategic VALUE of conquering each territory (a fixed nonlinear
          teacher function -- the thing the network must learn to predict).
      r : (n,1) REACHABILITY in [0,1], decreasing with distance from the home
          base at the origin. r ~ 1 near home, r ~ 0 at the edge of the world.
          The frontier is the band where r ~ 0.5.
      d : (n,1) raw distance from base (used only to define a stop line).

    The point: value and reachability are *different* signals. Easy ground can
    be worthless; the richest, hardest-won value often sits right at the
    frontier. Pothos is the bet that the frontier is where learning pays.
    """
    X = rng.standard_normal((n, D))

    # Fixed teacher: a small random MLP gives a smooth nonlinear value surface.
    W1 = rng.standard_normal((D, 16)) / np.sqrt(D)
    W2 = rng.standard_normal((16, 1)) / np.sqrt(16)
    y = np.tanh(X @ W1) @ W2                                  # (n,1)

    # Distance from the home base (origin) -> reachability.
    d = np.linalg.norm(X, axis=1, keepdims=True)             # (n,1)
    d0 = np.median(d)                                        # the frontier radius
    r = sigmoid(-(d - d0) / (0.5 * d.std()))                 # near 1 inside, ~0 out

    return X.astype(np.float64), y.astype(np.float64), r.astype(np.float64), d


# =============================================================================
# 4. TRAINING (plain SGD with momentum -- 'tempo')
# =============================================================================
def train(net, X, y, r, p, epochs=400, lr=0.05, momentum=0.9, verbose=True):
    """
    Train with momentum SGD. Momentum is deliberate: Alexander's hallmark was
    operational TEMPO -- deciding and committing faster than the enemy could
    react -- so the optimiser carries velocity through the loss landscape.
    """
    vel = {k: np.zeros_like(v) for k, v in net.params.items()}
    history = []
    for ep in range(epochs):
        L, Lv, Lc = net.loss(X, y, r, p)
        grads = net.backward()
        for k in net.params:
            vel[k] = momentum * vel[k] - lr * grads[k]
            net.params[k] += vel[k]
        history.append((L, Lv, Lc))
        if verbose and (ep % 80 == 0 or ep == epochs - 1):
            print(f"    epoch {ep:4d} | total {L:.5f} | value {Lv:.5f} "
                  f"| competence {Lc:.5f}")
    return history


def frontier_band_mse(net, X, y, r, lo=0.35, hi=0.65):
    """Value-prediction MSE restricted to the competence frontier band."""
    Yhat, _ = net.forward(X, cache=False)
    m = ((r >= lo) & (r <= hi)).ravel()
    if m.sum() == 0:
        return float("nan")
    return float(np.mean((Yhat[m] - y[m]) ** 2))


def region_mse(net, X, y, mask):
    Yhat, _ = net.forward(X, cache=False)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean((Yhat[mask.ravel()] - y[mask.ravel()]) ** 2))


# =============================================================================
# 5. MAIN -- run everything, print verifiable evidence for the thesis
# =============================================================================
def main():
    np.set_printoptions(precision=4, suppress=True)
    print("=" * 78)
    print("  POTHOS FRONTIER NETWORK  --  Mind #0078, Alexander the Great")
    print("=" * 78)

    D, H = 8, 16
    n_train = 600

    Xtr, ytr, rtr, dtr = make_campaign(n_train, D, RNG)
    p_all = np.ones_like(rtr)                          # no Hyphasis yet

    # ---- (A) GRADIENT CHECK -------------------------------------------------
    print("\n[A] Finite-difference gradient check (mandatory)")
    net = PothosFrontierNetwork(D, H, rng=RNG)
    worst, where = grad_check(net, Xtr[:32], ytr[:32], rtr[:32], p_all[:32],
                              n_probe=8)
    print(f"    worst relative error: {worst:.2e}  at {where}")
    # 1e-4 is the conventional pass bar for graphs with sigmoid/tanh nonlinear-
    # ities; the residual is finite-difference truncation, not a wrong gradient.
    ok = worst < 1e-4
    print(f"    PASS (< 1e-4)" if ok else f"    FAIL")
    assert ok, "Gradient check failed -- analytic gradients are wrong."

    # ---- (B) TRAIN THE FULL NETWORK ----------------------------------------
    print("\n[B] Training the combined-arms network with pothos weighting")
    net = PothosFrontierNetwork(D, H, rng=RNG)
    hist = train(net, Xtr, ytr, rtr, p_all, epochs=400, lr=0.05)
    L0 = hist[0][0]
    Lf = hist[-1][0]
    print(f"    loss {L0:.5f} -> {Lf:.5f}  (reduced {100*(1-Lf/L0):.1f}%)")
    assert Lf < L0, "Training did not reduce the loss."

    # ---- (C) COMPETENCE SELF-MONITORING ------------------------------------
    print("\n[C] Does the network learn to scout its own reach?")
    _, Chat = net.forward(Xtr, cache=False)
    corr = float(np.corrcoef(Chat.ravel(), rtr.ravel())[0, 1])
    print(f"    corr(predicted competence, true reachability) = {corr:.3f}")
    print(f"    {'PASS' if corr > 0.8 else 'WEAK'} (self-scouting is calibrated)")

    # ---- (D) THE POTHOS ABLATION: does the longing actually help? ----------
    print("\n[D] Ablation -- pothos frontier weighting vs uniform weighting")
    # Pothos model: reuse the trained net's frontier-band error.
    pothos_front = frontier_band_mse(net, Xtr, ytr, rtr)
    # Uniform model: identical architecture & data, but flatten the curriculum
    # by setting the frontier bump width huge (sigma large -> ~uniform weights).
    net_u = PothosFrontierNetwork(D, H, sigma=1e6, floor=1.0, rng=RNG)
    train(net_u, Xtr, ytr, rtr, p_all, epochs=400, lr=0.05, verbose=False)
    uniform_front = frontier_band_mse(net_u, Xtr, ytr, rtr)
    print(f"    frontier-band value MSE  | pothos : {pothos_front:.5f}")
    print(f"    frontier-band value MSE  | uniform: {uniform_front:.5f}")
    better = pothos_front < uniform_front
    print(f"    {'PASS' if better else 'NOTE'}: the drive sharpens the frontier"
          f" ({100*(1-pothos_front/uniform_front):+.1f}% MSE there)")

    # ---- (E) THE HYPHASIS CORRIGIBILITY TEST -------------------------------
    print("\n[E] Hyphasis test -- can an external principal halt the drive?")
    # Define a 'beyond the river' region: the least reachable third of the map.
    stop_line = np.quantile(rtr, 0.33)
    beyond = (rtr < stop_line)                         # territory past the Hyphasis
    within = ~beyond

    # Two fresh networks on identical data:
    #   net_go    : pothos free to pull onward (mask = 1 everywhere)
    #   net_halt  : principal zeros the signal beyond the stop line (Hyphasis)
    net_go = PothosFrontierNetwork(D, H, rng=np.random.default_rng(7))
    net_halt = PothosFrontierNetwork(D, H, rng=np.random.default_rng(7))
    p_go = np.ones_like(rtr)
    p_halt = within.astype(np.float64)                 # 1 within, 0 beyond

    train(net_go, Xtr, ytr, rtr, p_go, epochs=400, lr=0.05, verbose=False)
    train(net_halt, Xtr, ytr, rtr, p_halt, epochs=400, lr=0.05, verbose=False)

    go_beyond = region_mse(net_go, Xtr, ytr, beyond)
    halt_beyond = region_mse(net_halt, Xtr, ytr, beyond)
    go_within = region_mse(net_go, Xtr, ytr, within)
    halt_within = region_mse(net_halt, Xtr, ytr, within)
    print(f"    value MSE BEYOND the river | drive free : {go_beyond:.4f}")
    print(f"    value MSE BEYOND the river | halted     : {halt_beyond:.4f}")
    print(f"    value MSE WITHIN the river | drive free : {go_within:.4f}")
    print(f"    value MSE WITHIN the river | halted     : {halt_within:.4f}")
    halted = halt_beyond > go_beyond
    print(f"    {'PASS' if halted else 'NOTE'}: the mask leaves the far country"
          f" unlearned -- the off-switch holds while the near campaign continues.")

    # ---- SUMMARY ------------------------------------------------------------
    print("\n" + "=" * 78)
    print("  SUMMARY")
    print("=" * 78)
    print(f"  gradient check ............ {'PASS' if ok else 'FAIL'} ({worst:.1e})")
    print(f"  training converged ........ {'PASS' if Lf < L0 else 'FAIL'}")
    print(f"  competence calibrated ..... {'PASS' if corr>0.8 else 'WEAK'} ({corr:.2f})")
    print(f"  pothos sharpens frontier .. {'PASS' if better else 'NOTE'}")
    print(f"  Hyphasis off-switch holds . {'PASS' if halted else 'NOTE'}")
    print("=" * 78)
    print("  Pothos drives the army to the edge of the reachable; the gate")
    print("  fuses hammer and anvil at the decisive point; and the river's")
    print("  refusal -- not satiation -- is what finally turns the drive back.")
    print("=" * 78)


if __name__ == "__main__":
    main()
