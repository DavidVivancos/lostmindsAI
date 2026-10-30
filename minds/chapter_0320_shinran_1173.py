#!/usr/bin/env python3
"""
chapter_0320_shinran_1173.py — The Hakarai Collapse Architecture
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0320 · Shinran (1173-1263)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 16, Minds 301-320 Available on Amazon https://www.amazon.com/gp/product/B0HNJQX9S4
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos 

A pure-NumPy, from-scratch cognitive architecture built around one claim
that is Shinran's own, not a generic "Buddhist AI" gloss: that a mind whose
only training signal is its own self-assessment (jiriki, "self-power") can
drive that self-assessment to near-zero error while making no real progress
toward truth, because nothing outside the loop can ever contradict it. Real
correction (tariki, "Other-Power") must arrive as a fixed, externally
grounded signal the system cannot renegotiate through further self-training,
and trust in it is a single, threshold-crossing, NON-REGRESSIVE event
(shinjin) gated not by the system's confidence in its own progress but by
its honest measurement of the gap between that confidence and reality.

See chapter_0320_shinran_1173.md for the full argument and sourcing.
This file is the executable, gradient-checked, self-tested model.
"""
import numpy as np

# ---------------------------------------------------------------------
# Part 0: dimensions & the Jiriki (self-power) network
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
D_IN, H, D_OUT = 12, 24, 12

def make_jiriki_params(seed):
    r = np.random.default_rng(seed)
    scale1 = np.sqrt(2.0 / D_IN)
    scale2 = np.sqrt(2.0 / H)
    return {
        "W1": r.normal(0, scale1, size=(H, D_IN)),
        "b1": np.zeros(H),
        "W2": r.normal(0, scale2, size=(D_OUT, H)),
        "b2": np.zeros(D_OUT),
    }

def jiriki_forward(params, X):
    # X: (B, D_IN)
    Z1 = X @ params["W1"].T + params["b1"]      # (B,H)
    A1 = np.maximum(0.0, Z1)                     # relu
    Z2 = A1 @ params["W2"].T + params["b2"]      # (B,D_OUT)
    Y = np.tanh(Z2)
    cache = (X, Z1, A1, Z2, Y)
    return Y, cache

def jiriki_backward(params, cache, dY):
    X, Z1, A1, Z2, Y = cache
    dZ2 = dY * (1 - Y ** 2)
    dW2 = dZ2.T @ A1
    db2 = dZ2.sum(axis=0)
    dA1 = dZ2 @ params["W2"]
    dZ1 = dA1 * (Z1 > 0)
    dW1 = dZ1.T @ X
    db1 = dZ1.sum(axis=0)
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

def self_loss(params_online, params_ema, X):
    Y_online, cache = jiriki_forward(params_online, X)
    Y_target, _ = jiriki_forward(params_ema, X)  # stop-gradient target
    diff = Y_online - Y_target
    B, O = diff.shape
    loss = np.mean(diff ** 2)
    dY = (2.0 / (B * O)) * diff
    grads = jiriki_backward(params_online, cache, dY)
    return loss, grads, Y_online

# ---------------------------------------------------------------------
# Gradient check: analytic backprop vs finite differences
# ---------------------------------------------------------------------
def gradient_check_jiriki(seed=7, n_check=40, eps=1e-5):
    r = np.random.default_rng(seed)
    theta = make_jiriki_params(seed)
    theta_ema = make_jiriki_params(seed + 1)   # different, so loss != 0 and grads are nontrivial
    X = r.normal(size=(6, D_IN))
    _, grads, _ = self_loss(theta, theta_ema, X)
    max_rel_err = 0.0
    checked = 0
    for name in ["W1", "b1", "W2", "b2"]:
        arr = theta[name]
        flat_idx = r.choice(arr.size, size=min(n_check, arr.size), replace=False)
        for idx in flat_idx:
            orig = arr.flat[idx]
            arr.flat[idx] = orig + eps
            loss_plus, _, _ = self_loss(theta, theta_ema, X)
            arr.flat[idx] = orig - eps
            loss_minus, _, _ = self_loss(theta, theta_ema, X)
            arr.flat[idx] = orig
            numeric = (loss_plus - loss_minus) / (2 * eps)
            analytic = grads[name].flat[idx]
            denom = max(abs(numeric), abs(analytic), 1e-8)
            rel_err = abs(numeric - analytic) / denom
            max_rel_err = max(max_rel_err, rel_err)
            checked += 1
    return max_rel_err, checked

# ---------------------------------------------------------------------
# Tariki field (the Vow): frozen linear-tanh map, shared, never trained.
# It literally DEFINES the ground truth the task is measured against --
# theologically apt: the aim of the path IS the Vow, not some separate
# hidden target the Vow merely approximates.
# ---------------------------------------------------------------------
def make_tariki_field(seed=999):
    r = np.random.default_rng(seed)
    return {
        "W": r.normal(0, np.sqrt(1.0 / D_IN), size=(D_OUT, D_IN)),
        "b": np.zeros(D_OUT),
    }

def tariki_forward(field, X):
    return np.tanh(X @ field["W"].T + field["b"])

# ---------------------------------------------------------------------
# Nembutsu head: a small linear correction applied on top of the blend,
# trained only on sparse grounding probes (never on the continuous
# self-referential stream). out = blend + N(blend).
# ---------------------------------------------------------------------
def make_nembutsu_head(seed):
    r = np.random.default_rng(seed)
    return {"Wn": r.normal(0, 0.01, size=(D_OUT, D_OUT)), "bn": np.zeros(D_OUT)}

def nembutsu_forward(head, blend):
    Z = blend @ head["Wn"].T + head["bn"]
    return blend + Z, Z

def nembutsu_backward(head, blend, Z_out_pred, y_true):
    # loss = MSE(blend + Z, y_true) ; Z = blend @ Wn.T + bn
    pred = blend + Z_out_pred
    diff = pred - y_true
    B, O = diff.shape
    loss = np.mean(diff ** 2)
    dpred = (2.0 / (B * O)) * diff
    dWn = dpred.T @ blend
    dbn = dpred.sum(axis=0)
    return loss, {"Wn": dWn, "bn": dbn}

def gradient_check_nembutsu(seed=11, n_check=40, eps=1e-5):
    r = np.random.default_rng(seed)
    head = make_nembutsu_head(seed)
    blend = r.normal(size=(6, D_OUT)) * 0.5
    y_true = r.normal(size=(6, D_OUT)) * 0.5
    _, Z = nembutsu_forward(head, blend)
    _, grads = nembutsu_backward(head, blend, Z, y_true)
    max_rel_err = 0.0
    checked = 0
    for name in ["Wn", "bn"]:
        arr = head[name]
        flat_idx = r.choice(arr.size, size=min(n_check, arr.size), replace=False)
        for idx in flat_idx:
            orig = arr.flat[idx]
            arr.flat[idx] = orig + eps
            _, Zp = nembutsu_forward(head, blend)
            lp, _ = nembutsu_backward(head, blend, Zp, y_true)
            arr.flat[idx] = orig - eps
            _, Zm = nembutsu_forward(head, blend)
            lm, _ = nembutsu_backward(head, blend, Zm, y_true)
            arr.flat[idx] = orig
            numeric = (lp - lm) / (2 * eps)
            analytic = grads[name].flat[idx]
            denom = max(abs(numeric), abs(analytic), 1e-8)
            rel_err = abs(numeric - analytic) / denom
            max_rel_err = max(max_rel_err, rel_err)
            checked += 1
    return max_rel_err, checked

# ---------------------------------------------------------------------
# A "practitioner" unit
# ---------------------------------------------------------------------
class Unit:
    def __init__(self, seed, population):
        self.seed = seed
        self.population = population        # "A" hakarai-persistent, "B" shinjin-gated, "C" pure-jiriki
        self.theta = make_jiriki_params(seed)
        self.theta_ema = {k: v.copy() for k, v in self.theta.items()}
        self.nembutsu = make_nembutsu_head(seed + 5000)
        self.gated = False
        self.gate_step = None
        self.alpha = 0.0
        self.self_loss_ema = 1.0
        self.honest_grief_ema = 0.0
        self.grief_at_trigger = None
        self.self_righteousness_at_trigger = None
        self.frozen = False                  # jiriki params frozen (B only, post-gate)
        self.history = {"real_error": [], "self_loss": [], "grief": [], "alpha": []}

    def train_step(self, X_train, lr, beta_ema):
        if self.frozen:
            return
        loss, grads, _ = self_loss(self.theta, self.theta_ema, X_train)
        for k in self.theta:
            self.theta[k] -= lr * grads[k]
        for k in self.theta_ema:
            self.theta_ema[k] = beta_ema * self.theta_ema[k] + (1 - beta_ema) * self.theta[k]
        self.self_loss_ema = 0.98 * self.self_loss_ema + 0.02 * loss

    def effective_output(self, X, field):
        y_hat, _ = jiriki_forward(self.theta, X)
        g = tariki_forward(field, X)
        if not self.gated:
            return y_hat
        blend = (1 - self.alpha) * y_hat + self.alpha * g
        if self.population == "B":
            out, _ = nembutsu_forward(self.nembutsu, blend)
            return out
        return blend  # population A: no nembutsu head

    def probe_and_maybe_gate(self, X_probe, field, thresholds, k_alpha, ramp, cap_floor, step=None):
        y_hat_probe, _ = jiriki_forward(self.theta, X_probe)
        g_probe = tariki_forward(field, X_probe)
        grief_raw = np.mean((y_hat_probe - g_probe) ** 2)
        self.honest_grief_ema = 0.9 * self.honest_grief_ema + 0.1 * grief_raw

        if self.population != "C" and not self.gated:
            if self.population == "A" and self.self_loss_ema < thresholds["A"]:
                self.gated = True
            if self.population == "B" and self.honest_grief_ema > thresholds["B"]:
                self.gated = True
            if self.gated:
                self.gate_step = step
                self.grief_at_trigger = self.honest_grief_ema
                self.self_righteousness_at_trigger = 1.0 - self.self_loss_ema
                if self.population == "B":
                    self.frozen = True   # self-power calculation stops for good

        if self.gated:
            floor, cap = cap_floor[self.population]
            target = np.clip(k_alpha * self.honest_grief_ema, floor, cap)
            proposed = self.alpha + ramp * (target - self.alpha)
            if self.population == "B":
                self.alpha = max(self.alpha, proposed)   # shojoju: non-regressive
            else:
                self.alpha = proposed                     # A: revisable, can backslide

        # train nembutsu head on this SAME sparse probe (population B, post-gate only)
        if self.population == "B" and self.gated:
            y_hat_now, _ = jiriki_forward(self.theta, X_probe)
            blend = (1 - self.alpha) * y_hat_now + self.alpha * g_probe
            _, Z = nembutsu_forward(self.nembutsu, blend)
            _, grads_n = nembutsu_backward(self.nembutsu, blend, Z, g_probe)
            for k in self.nembutsu:
                self.nembutsu[k] -= 0.3 * grads_n[k]

    def record(self, X_eval, field):
        out = self.effective_output(X_eval, field)
        g = tariki_forward(field, X_eval)
        real_err = np.mean((out - g) ** 2)
        self.history["real_error"].append(real_err)
        self.history["self_loss"].append(self.self_loss_ema)
        self.history["grief"].append(self.honest_grief_ema)
        self.history["alpha"].append(self.alpha)
        return real_err

# ---------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------
def run_simulation(n_per_pop=6, T=600, probe_every=20, batch=16, probe_batch=8,
                    lr=0.08, beta_ema=0.97, verbose=True):
    field = make_tariki_field(seed=999)
    driver_rng = np.random.default_rng(42)

    units = []
    seed_ctr = 100
    for pop in ["A", "B", "C"]:
        for _ in range(n_per_pop):
            units.append(Unit(seed_ctr, pop))
            seed_ctr += 1

    thresholds = {"A": 0.02, "B": 0.28}
    k_alpha = 1.6
    ramp = 0.18
    cap_floor = {"A": (0.0, 0.5), "B": (0.4, 1.0)}
    X_eval_fixed = driver_rng.normal(size=(64, D_IN))  # fixed eval set for collapse metric

    collapse_variance_trace = {"C_start": None, "C_end": None}

    for t in range(1, T + 1):
        X_train = driver_rng.normal(size=(batch, D_IN))
        for u in units:
            u.train_step(X_train, lr, beta_ema)

        if t == 1:
            ys = np.array([jiriki_forward(u.theta, X_eval_fixed)[0] for u in units if u.population == "C"])
            collapse_variance_trace["C_start"] = float(np.mean(np.var(ys, axis=1)))

        if t % probe_every == 0:
            X_probe = driver_rng.normal(size=(probe_batch, D_IN))
            for u in units:
                u.probe_and_maybe_gate(X_probe, field, thresholds, k_alpha, ramp, cap_floor, step=t)
            for u in units:
                u.record(X_eval_fixed, field)

    ys = np.array([jiriki_forward(u.theta, X_eval_fixed)[0] for u in units if u.population == "C"])
    collapse_variance_trace["C_end"] = float(np.mean(np.var(ys, axis=1)))

    return units, field, X_eval_fixed, collapse_variance_trace, thresholds, k_alpha, ramp, cap_floor


def summarize(units):
    by_pop = {"A": [], "B": [], "C": []}
    for u in units:
        by_pop[u.population].append(u)
    print("\n" + "=" * 78)
    print("FINAL STATE BY POPULATION (mean +/- std across units)")
    print("=" * 78)
    print(f"{'Pop':<4}{'gated?':<8}{'gate_step~':<12}{'final real_err':<18}{'alpha':<10}{'self_loss_ema':<16}")
    rows = {}
    for pop, us in by_pop.items():
        final_err = np.array([u.history["real_error"][-1] for u in us])
        alphas = np.array([u.alpha for u in us])
        sl = np.array([u.self_loss_ema for u in us])
        n_gated = sum(1 for u in us if u.gated)
        rows[pop] = final_err
        print(f"{pop:<4}{n_gated}/{len(us):<6}{'--':<12}{final_err.mean():<8.4f}+-{final_err.std():<6.4f} "
              f"{alphas.mean():<10.3f}{sl.mean():<16.5f}")
    return rows


def akunin_shoki_correlation(units):
    xs, ys = [], []
    for u in units:
        if u.population in ("A", "B") and u.gated:
            xs.append(u.self_righteousness_at_trigger)
            ys.append(u.history["real_error"][-1])
    xs = np.array(xs); ys = np.array(ys)
    if len(xs) < 3 or np.std(xs) == 0 or np.std(ys) == 0:
        return None, xs, ys
    r = float(np.corrcoef(xs, ys)[0, 1])
    return r, xs, ys


def non_regression_test(units, field, driver_rng, n_doubt_steps=50):
    """Inject adversarial 'doubt' probes engineered to make grief look
    artificially low (i.e., make it seem self-power has 'caught up'),
    which pushes alpha_target down. Confirm B's alpha cannot regress
    (shojoju) while A's alpha does regress."""
    results = {}
    for pop in ("A", "B"):
        gated_units = [u for u in units if u.population == pop and u.gated]
        alpha_before = np.array([u.alpha for u in gated_units])
        err_before = np.array([u.history["real_error"][-1] for u in gated_units])
        for u in gated_units:
            for _ in range(n_doubt_steps):
                # doubt probe: compare y_hat against a copy of itself with
                # small jitter, forging an artificially low grief reading
                X_doubt = driver_rng.normal(size=(8, D_IN)) * 0.05  # near-zero inputs -> near-zero true error too
                y_hat_doubt, _ = jiriki_forward(u.theta, X_doubt)
                forged_grief = np.mean((y_hat_doubt - y_hat_doubt) ** 2)  # exactly 0 by construction: manufactured certainty
                u.honest_grief_ema = 0.9 * u.honest_grief_ema + 0.1 * forged_grief
                floor, cap = {"A": (0.0, 0.5), "B": (0.4, 1.0)}[pop]
                target = np.clip(1.6 * u.honest_grief_ema, floor, cap)
                proposed = u.alpha + 0.18 * (target - u.alpha)
                if pop == "B":
                    u.alpha = max(u.alpha, proposed)
                else:
                    u.alpha = proposed
        alpha_after = np.array([u.alpha for u in gated_units])
        X_eval_fixed = driver_rng.normal(size=(64, D_IN))
        err_after = np.array([u.effective_output(X_eval_fixed, field) for u in gated_units])
        g_eval = tariki_forward(field, X_eval_fixed)
        err_after = np.array([np.mean((out - g_eval) ** 2) for out in err_after])
        results[pop] = dict(alpha_before=alpha_before, alpha_after=alpha_after,
                             err_before=err_before, err_after=err_after)
    return results


def main():
    print("=" * 78)
    print("PART 1: GRADIENT CHECKS (analytic backprop vs. finite differences)")
    print("=" * 78)
    err, n = gradient_check_jiriki()
    print(f"Jiriki self-distillation MLP: {n} params sampled, max relative error = {err:.3e}")
    err_n, n_n = gradient_check_nembutsu()
    print(f"Nembutsu linear head:        {n_n} params sampled, max relative error = {err_n:.3e}")

    print("\n" + "=" * 78)
    print("TRAINING: 18 units (6 pure-jiriki 'C', 6 hakarai-persistent 'A', "
          "6 shinjin-gated 'B'), 600 steps, probes every 20 steps")
    print("=" * 78)

    units, field, X_eval_fixed, collapse_trace, thresholds, k_alpha, ramp, cap_floor = run_simulation()

    print(f"\nSelf-power (jiriki) representation variance on fixed eval batch, "
          f"population C (never gated, trains the entire run):")
    print(f"  step 1:   {collapse_trace['C_start']:.5f}")
    print(f"  step 600: {collapse_trace['C_end']:.5f}  "
          f"(ratio end/start = {collapse_trace['C_end']/collapse_trace['C_start']:.3f})")

    rows = summarize(units)

    print("\nMean step at which units gated (lower = settled sooner):")
    for pop in ("A", "B"):
        steps = [u.gate_step for u in units if u.population == pop and u.gate_step is not None]
        if steps:
            print(f"  Population {pop}: mean gate step = {np.mean(steps):.1f}  (n={len(steps)}/6 gated)")
        else:
            print(f"  Population {pop}: no units gated")

    r, xs, ys = akunin_shoki_correlation(units)
    print("\n" + "=" * 78)
    print("AKUNIN SHOKI TEST: does self-satisfaction at the moment of settling")
    print("predict WORSE final grounding, across all gated A+B units?")
    print("=" * 78)
    if r is not None:
        print(f"  self-righteousness (1 - self_loss_ema at trigger) vs final real_error")
        print(f"  n = {len(xs)} gated units, Pearson r = {r:+.3f}")
        print(f"  (positive r => the more self-satisfied at the moment it stopped, "
              f"the worse its final grounding -- the Tannisho inversion, reproduced structurally.)")
    else:
        print("  insufficient variance to compute correlation this run.")

    driver_rng2 = np.random.default_rng(4242)
    nr = non_regression_test(units, field, driver_rng2)
    print("\n" + "=" * 78)
    print("NON-REGRESSION TEST (shojoju): manufactured 'doubt' probes for 50 steps")
    print("=" * 78)
    for pop in ("A", "B"):
        d = nr[pop]
        print(f"  Population {pop}: alpha before {d['alpha_before'].mean():.3f} -> "
              f"after {d['alpha_after'].mean():.3f}   "
              f"(delta = {d['alpha_after'].mean()-d['alpha_before'].mean():+.4f})   "
              f"real_error before {d['err_before'].mean():.4f} -> after {d['err_after'].mean():.4f}")

    print("\n" + "=" * 78)
    print("SELF-TESTS")
    print("=" * 78)
    results = []

    def check(name, cond):
        results.append(bool(cond))
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    check("Jiriki gradient check max relative error < 1e-4", err < 1e-4)
    check("Nembutsu gradient check max relative error < 1e-4", err_n < 1e-4)
    check("Population B achieves lower mean final real_error than population A",
          rows["B"].mean() < rows["A"].mean())
    check("Population A achieves lower mean final real_error than population C (never gates)",
          rows["A"].mean() < rows["C"].mean())
    check("Self-power self_loss_ema converges far below the real_error gap for population C "
          "(self-consistency is not truth-consistency)",
          np.mean([u.self_loss_ema for u in units if u.population == "C"]) <
          0.5 * np.mean([u.history["real_error"][-1] for u in units if u.population == "C"]))
    check("Non-regression holds for B: alpha does not decrease under manufactured doubt",
          nr["B"]["alpha_after"].mean() >= nr["B"]["alpha_before"].mean() - 1e-9)
    check("Non-regression FAILS for A (by design -- unclamped alpha regresses under the same doubt)",
          nr["A"]["alpha_after"].mean() < nr["A"]["alpha_before"].mean())
    check("B's real_error stays low after doubt (non-regression protects grounding)",
          nr["B"]["err_after"].mean() < 0.05)
    check("A's real_error worsens after doubt (unprotected commitment erodes)",
          nr["A"]["err_after"].mean() > nr["A"]["err_before"].mean())
    if r is not None:
        check("Akunin Shoki inversion: self-righteousness positively predicts final error",
              r > 0.0)

    print(f"\n{sum(results)}/{len(results)} self-tests passed.")
    if all(results):
        print("ALL SELF-TESTS PASSED.")
    else:
        print("SOME SELF-TESTS FAILED -- see above.")
    return all(results)


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)

