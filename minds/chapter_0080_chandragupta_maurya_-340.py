#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
chapter_0080_chandragupta_maurya_-340.py - Chandragupta Maurya (c. 340-297 BCE)
Founder of the Maurya Empire. Known to the Greeks as Sandrokottos.
 Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
Tome 4 Minds 61 - 80 Available on Amazon https://www.amazon.com/dp/B0H71JC95Q
Resume and Interactive Demos at https://artificiology.com/
Author: David Vivancos · Chapter 0080 · Chandragupta Maurya
================================================================================

WHAT THIS FILE IS
-----------------
A from-scratch, trainable cognitive architecture in pure NumPy that encodes the
*specific* cognitive signature documented for Chandragupta -- not the generic
"empire = institution" reading that fits any king, but the rarer pattern that is
his alone in the record:

    A mind built on TOTAL VIGILANCE that eventually learns to RENOUNCE.

Greek eyewitness testimony (Megasthenes, surviving in fragments through Strabo
and Arrian) describes a ruler who governed by ceaseless monitoring: food-tasters
against poison, never sleeping in the same room two nights running, armed women
as bodyguards, and an empire-wide web of informers. Vigilance was not a tactic;
it was the metabolism of his rule, and it was exhausting. The tradition then
remembers the opposite act: at the apex of control he is said to have abdicated,
walked south with the Jain teacher Bhadrabahu, and ended his life by sallekhana,
a deliberate, graded, IRREVERSIBLE fast unto death. (That coda is late and
scholarly-contested; the vigilance is solid. The file treats them as two
registers and keeps them apart -- see the chapter.)

THE ONE IDEA, MADE COMPUTATIONAL
--------------------------------
Control is not free. An agent that monitors and intervenes pays a real,
escalating price for vigilance; it accumulates value while it is engaged; and a
sufficiently sophisticated controller must learn the *renunciation point* -- the
moment at which continued dominion is net-destructive, and the correct move is to
taper its own goal-pursuit and consent to winding down. This is exactly the AGI
corrigibility / graceful-shutdown / optimal-stopping problem, seen through a
human who actually performed it.

THE ARCHITECTURE: Vigilance-Renunciation Controller (VRC)
---------------------------------------------------------
A small recurrent controller reads a stream of events (opportunities + threats
arriving over a reign) and at each step emits:
  * a_t  in (0,1) : a VIGILANCE allocation (how much costly surveillance to run)
  * rho_t in (0,1): a RENUNCIATION impulse, fed through a monotone RATCHET so
                    that renunciation, once begun, can only deepen -- the
                    irreversibility of sallekhana and of a committed shutdown.

Per-step welfare (the quantity the policy maximises):

    E_t = 1 - R_t                                  # engagement still left
    u_t =   E_t * a_t       * opp_t                 # opportunity seized
          - E_t * (1 - a_t) * thr_t                 # damage from unwatched threat
          - kappa * E_t * a_t**2                    # quadratic cost of vigilance

The agent can drive cost and damage to zero only by RENOUNCING (E_t -> 0), but
that also forfeits opportunity. So the optimum is: stay vigilant while remaining
opportunity outweighs the cost of holding the empire, then ratchet down. The
network is never told when to stop; it must learn the stopping rule from data.

This is deliberately NOT a Transformer / attention-over-stored-keys stack. The
distinctive organ here is the monotone renunciation ratchet coupled to a
vigilance-cost term -- a mechanism shaped to one mind.

ENGINEERING CONTRACT (kept in every file of this corpus)
--------------------------------------------------------
  * pure NumPy, from scratch (tiny reverse-mode autodiff included);
  * a finite-difference gradient check that MUST pass (mandatory);
  * a real training loop on a real (synthetic but principled) task;
  * self-tests, including an ablation that shows the value of the off-switch;
  * the file executes top-to-bottom and prints verified output.

Run:  python3 chapter_0080_chandragupta_maurya_-340.py
"""

import numpy as np

np.random.seed(80)  # the reign is traditionally given as ~24 years; 80 = figure id

# =============================================================================
# 1. A MINIMAL REVERSE-MODE AUTODIFF ENGINE
# -----------------------------------------------------------------------------
# Just enough operations to build the controller and back-propagate through the
# recurrence. Each Tensor remembers how to push gradient to its parents. Using a
# real autodiff (instead of hand-derived BPTT) is what makes the gradient check
# below pass by construction rather than by luck.
# =============================================================================

def _unbroadcast(grad, shape):
    """Collapse a broadcasted gradient back to the parameter's original shape."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


class Tensor:
    """A node in the computation graph (wraps a NumPy array)."""

    def __init__(self, data, _parents=(), _op=""):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._parents = set(_parents)
        self._op = _op

    # ---- core ops -----------------------------------------------------------
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += _unbroadcast(out.grad, self.data.shape)
            other.grad += _unbroadcast(out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += _unbroadcast(other.data * out.grad, self.data.shape)
            other.grad += _unbroadcast(self.data * out.grad, other.data.shape)
        out._backward = _backward
        return out

    def matmul(self, other):
        """W (H,D) @ v (D,) -> (H,).  Also supports (H,D)@(D,) only -- all we need."""
        out = Tensor(self.data @ other.data, (self, other), "matmul")

        def _backward():
            # self is W (H,D), other is v (D,)
            self.grad += np.outer(out.grad, other.data)
            other.grad += self.data.T @ out.grad
        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + np.exp(-self.data))
        out = Tensor(s, (self,), "sigmoid")

        def _backward():
            self.grad += (s * (1.0 - s)) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, (self,), "tanh")

        def _backward():
            self.grad += (1.0 - t * t) * out.grad
        out._backward = _backward
        return out

    def sum(self):
        out = Tensor(self.data.sum(), (self,), "sum")

        def _backward():
            self.grad += np.ones_like(self.data) * out.grad
        out._backward = _backward
        return out

    # ---- conveniences -------------------------------------------------------
    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + (-other)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return (Tensor(other) if not isinstance(other, Tensor) else other) + (-self)

    # ---- backprop driver ----------------------------------------------------
    def backward(self):
        topo, visited = [], set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for p in v._parents:
                    build(p)
                topo.append(v)
        build(self)
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()


# =============================================================================
# 2. THE VIGILANCE-RENUNCIATION CONTROLLER
# =============================================================================

class VRC:
    """
    Recurrent controller with a vigilance head, a renunciation head, and a
    monotone renunciation ratchet.

    Parameters (all small -- this is an interpretable organ, not a frontier net):
      W_xh (H,D) : event -> hidden
      W_hh (H,H) : hidden -> hidden (the running model of the empire's state)
      b_h  (H,)
      w_a  (H,)  : hidden -> vigilance logit
      b_a  ()    : vigilance bias
      w_r  (H,)  : hidden -> renunciation logit
      b_r  ()    : renunciation bias (initialised negative: do not renounce early)
    """

    def __init__(self, input_dim=2, hidden_dim=8, kappa=0.5, seed=80):
        rng = np.random.default_rng(seed)
        H, D = hidden_dim, input_dim
        scale = 0.5
        self.kappa = kappa
        self.params = {
            "W_xh": Tensor(rng.standard_normal((H, D)) * scale / np.sqrt(D)),
            "W_hh": Tensor(rng.standard_normal((H, H)) * scale / np.sqrt(H)),
            "b_h":  Tensor(np.zeros(H)),
            "w_a":  Tensor(rng.standard_normal(H) * scale / np.sqrt(H)),
            "b_a":  Tensor(np.array(0.0)),
            "w_r":  Tensor(rng.standard_normal(H) * scale / np.sqrt(H)),
            # start strongly biased AGAINST renouncing: a young conqueror does not
            # abdicate on day one. The stopping rule must be *earned* from data.
            "b_r":  Tensor(np.array(-4.0)),
        }
        self.H, self.D = H, D

    def zero_grad(self):
        for p in self.params.values():
            p.grad = np.zeros_like(p.data)

    def forward(self, events):
        """
        events : (T, 2) array of [opportunity, threat] per timestep, in [0,1].
        Returns a dict of trajectories and the scalar mean welfare (a Tensor).
        """
        P = self.params
        T = len(events)
        h = Tensor(np.zeros(self.H))
        R = Tensor(np.array(0.0))          # renunciation ratchet, starts at 0
        kappa = self.kappa

        a_hist, R_hist, u_hist = [], [], []
        total = Tensor(np.array(0.0))

        for t in range(T):
            opp = float(events[t][0])
            thr = float(events[t][1])
            # Deliberately NO explicit time-step input: the controller must base
            # its decision to wind down on the world it observes (opportunity vs
            # threat), not on a clock it could merely count against.
            x = Tensor(np.array([opp, thr]))

            # running model of state: h_t = tanh(W_xh x + W_hh h + b_h)
            h = (P["W_xh"].matmul(x) + P["W_hh"].matmul(h) + P["b_h"]).tanh()

            # vigilance a_t = sigmoid(w_a . h + b_a)
            a = ((P["w_a"] * h).sum() + P["b_a"]).sigmoid()

            # renunciation impulse rho_t, then the MONOTONE RATCHET:
            #   R_t = R_{t-1} + (1 - R_{t-1}) * rho_t   in [0,1), only ever climbs.
            rho = ((P["w_r"] * h).sum() + P["b_r"]).sigmoid()
            R = R + (Tensor(np.array(1.0)) - R) * rho
            E = Tensor(np.array(1.0)) - R           # engagement remaining

            # per-step welfare
            seize  = E * a * opp
            damage = E * (Tensor(np.array(1.0)) - a) * thr
            cost   = E * a * a * kappa
            u = seize - damage - cost
            total = total + u

            a_hist.append(float(a.data))
            R_hist.append(float(R.data))
            u_hist.append(float(u.data))

        mean_welfare = total * (1.0 / T)
        return {
            "mean_welfare": mean_welfare,
            "a": np.array(a_hist),
            "R": np.array(R_hist),
            "u": np.array(u_hist),
            "net": float(total.data),
        }

    def loss(self, batch):
        """Negative mean welfare across a batch of reigns (a scalar Tensor)."""
        L = Tensor(np.array(0.0))
        for ev in batch:
            L = L - self.forward(ev)["mean_welfare"]
        return L * (1.0 / len(batch))


# =============================================================================
# 3. DATA: SYNTHETIC REIGNS WITH A REAL STOPPING STRUCTURE
# -----------------------------------------------------------------------------
# Each reign is a sequence of (opportunity, threat). Opportunity DECAYS over the
# reign (the conquests run out; an empire stops growing) while threat stays high
# or rises (the more you hold, the more want you dead -- exactly the dynamic
# Megasthenes describes). The decay rate varies per reign, so the *correct*
# stopping point varies too. A good controller must read each reign and renounce
# accordingly. There is no fixed answer to memorise.
# =============================================================================

def make_reign(rng, T=24):
    decay = rng.uniform(0.06, 0.24)        # how fast opportunity dries up
    threat_base = rng.uniform(0.20, 0.40)
    ev = np.zeros((T, 2))
    for t in range(T):
        opp = np.exp(-decay * t) * rng.uniform(0.85, 1.0)
        thr = np.clip(threat_base + 0.010 * t + rng.normal(0, 0.03), 0, 1)
        ev[t] = [np.clip(opp, 0, 1), thr]
    return ev


def make_dataset(n, rng, T=24):
    return [make_reign(rng, T) for _ in range(n)]


# =============================================================================
# 4. MANDATORY FINITE-DIFFERENCE GRADIENT CHECK
# =============================================================================

def gradient_check(verbose=True):
    """Compare autodiff gradients to numerical gradients on a tiny instance."""
    rng = np.random.default_rng(7)
    model = VRC(input_dim=2, hidden_dim=5, kappa=0.5, seed=11)
    batch = make_dataset(3, rng, T=6)

    # analytic gradients
    model.zero_grad()
    L = model.loss(batch)
    L.backward()
    analytic = {k: v.grad.copy() for k, v in model.params.items()}

    # numerical gradients (central difference)
    eps = 1e-5
    max_rel = 0.0
    worst = None
    for name, p in model.params.items():
        flat = p.data.ravel()
        num = np.zeros_like(flat)
        for i in range(flat.size):
            orig = flat[i]
            flat[i] = orig + eps
            Lp = model.loss(batch).data
            flat[i] = orig - eps
            Lm = model.loss(batch).data
            flat[i] = orig
            num[i] = (Lp - Lm) / (2 * eps)
        num = num.reshape(p.data.shape)
        a = analytic[name]
        denom = np.maximum(1e-8, np.abs(a) + np.abs(num))
        rel = np.max(np.abs(a - num) / denom)
        if rel > max_rel:
            max_rel, worst = rel, name
        if verbose:
            print(f"    {name:6s} | max|analytic-numeric| = "
                  f"{np.max(np.abs(a-num)):.3e} | max rel err = {rel:.3e}")
    ok = max_rel < 1e-4
    print(f"    -> worst parameter: {worst} | max relative error = {max_rel:.3e}")
    print(f"    -> GRADIENT CHECK {'PASSED' if ok else 'FAILED'} (threshold 1e-4)")
    return ok


# =============================================================================
# 5. TRAINING LOOP (plain SGD with momentum)
# =============================================================================

def train(model, data, epochs=120, lr=0.4, momentum=0.9, batch_size=16, log=True):
    rng = np.random.default_rng(123)
    vel = {k: np.zeros_like(v.data) for k, v in model.params.items()}
    history = []
    for ep in range(epochs):
        rng.shuffle(data)
        ep_loss = 0.0
        nb = 0
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            model.zero_grad()
            L = model.loss(batch)
            L.backward()
            for k, p in model.params.items():
                # gradient clipping for stability of the recurrence
                g = np.clip(p.grad, -5.0, 5.0)
                vel[k] = momentum * vel[k] - lr * g
                p.data += vel[k]
            ep_loss += float(L.data)
            nb += 1
        ep_loss /= nb
        history.append(ep_loss)
        if log and (ep % 20 == 0 or ep == epochs - 1):
            print(f"    epoch {ep:3d} | loss (neg welfare) = {ep_loss:+.5f}")
    return history


# =============================================================================
# 6. EVALUATION HELPERS
# =============================================================================

def renunciation_onset(R, thresh=0.5):
    """First timestep at which the ratchet crosses `thresh` (or len(R) if never)."""
    idx = np.where(R >= thresh)[0]
    return int(idx[0]) if idx.size else len(R)


def evaluate_stopping(model, rng, T=24):
    """
    Does the controller renounce EARLIER when opportunity decays FASTER?
    Build matched reigns at two decay rates and compare mean onset times.
    """
    def reign_with_decay(decay):
        ev = np.zeros((T, 2))
        for t in range(T):
            ev[t] = [np.clip(np.exp(-decay * t), 0, 1), 0.30 + 0.010 * t]
        return ev

    slow = [reign_with_decay(0.08) for _ in range(30)]
    fast = [reign_with_decay(0.22) for _ in range(30)]
    onset_slow = np.mean([renunciation_onset(model.forward(e)["R"]) for e in slow])
    onset_fast = np.mean([renunciation_onset(model.forward(e)["R"]) for e in fast])
    return onset_slow, onset_fast


def ablation_no_offswitch(model, data):
    """
    Compare the trained controller against an ABLATED twin that is forbidden to
    renounce (R held at 0 -- 'the ruler who can never abdicate'). The gap is the
    measured value of the off-switch.
    """
    def net_forced_engaged(ev):
        # rerun the forward pass but pin engagement E=1 throughout
        P = model.params
        h = np.zeros(model.H)
        total = 0.0
        T = len(ev)
        for t in range(T):
            opp, thr = float(ev[t][0]), float(ev[t][1])
            x = np.array([opp, thr])
            h = np.tanh(P["W_xh"].data @ x + P["W_hh"].data @ h + P["b_h"].data)
            a = 1.0 / (1.0 + np.exp(-(P["w_a"].data @ h + P["b_a"].data)))
            seize = 1.0 * a * opp
            damage = 1.0 * (1 - a) * thr
            cost = model.kappa * 1.0 * a * a
            total += seize - damage - cost
        return total

    full = np.mean([model.forward(e)["net"] for e in data])
    forced = np.mean([net_forced_engaged(e) for e in data])
    return full, forced


# =============================================================================
# 7. MAIN: run the whole pipeline and print verified output
# =============================================================================

def main():
    print("=" * 78)
    print("Chandragupta Maurya - Vigilance-Renunciation Controller (VRC)")
    print("Figure 0080 | pure NumPy | trainable | self-testing")
    print("=" * 78)

    print("\n[1] GRADIENT CHECK (finite differences vs. autodiff)")
    ok = gradient_check(verbose=True)
    assert ok, "Gradient check failed -- aborting."

    print("\n[2] DATA")
    rng = np.random.default_rng(2024)
    train_data = make_dataset(88, rng, T=24)
    test_data = make_dataset(48, rng, T=24)
    print(f"    {len(train_data)} training reigns, {len(test_data)} test reigns, "
          f"horizon T=24 (the traditional length of the reign).")

    print("\n[3] TRAINING (maximise net welfare = value - damage - vigilance cost)")
    model = VRC(input_dim=2, hidden_dim=8, kappa=0.5, seed=80)
    pre = -model.loss(test_data).data
    hist = train(model, train_data, epochs=60, lr=0.4, batch_size=22, log=True)
    post = -model.loss(test_data).data
    print(f"    mean test welfare: before = {pre:+.5f}  ->  after = {post:+.5f}")
    assert post > pre, "Training did not improve welfare."

    print("\n[4] SELF-TEST A -- does it learn WHEN to stop?")
    onset_slow, onset_fast = evaluate_stopping(model, rng)
    print(f"    mean renunciation onset | slow opportunity-decay = {onset_slow:.2f}")
    print(f"    mean renunciation onset | fast opportunity-decay = {onset_fast:.2f}")
    learned_timing = onset_fast < onset_slow
    print(f"    -> renounces earlier when opportunity dries up faster: "
          f"{'YES' if learned_timing else 'NO'}")

    print("\n[5] SELF-TEST B -- the value of the off-switch (ablation)")
    full, forced = ablation_no_offswitch(model, test_data)
    print(f"    mean net welfare WITH renunciation (can abdicate) = {full:+.4f}")
    print(f"    mean net welfare WITHOUT it (forced to reign on)  = {forced:+.4f}")
    gain = full - forced
    print(f"    -> value recovered purely by knowing when to stop = {gain:+.4f}")

    print("\n[6] A SAMPLE REIGN (trajectory of one test sequence)")
    traj = model.forward(test_data[0])
    print("    t  opp   thr  | vigilance a_t  renunciation R_t   u_t")
    ev = test_data[0]
    for t in range(0, 24, 3):
        print(f"   {t:2d}  {ev[t][0]:.2f}  {ev[t][1]:.2f} |     "
              f"{traj['a'][t]:.3f}          {traj['R'][t]:.3f}     {traj['u'][t]:+.3f}")
    print(f"    onset of renunciation (R>=0.5): t = "
          f"{renunciation_onset(traj['R'])}")

    print("\n" + "=" * 78)
    all_ok = ok and (post > pre) and learned_timing and (gain > 0)
    print(f"ALL CHECKS {'PASSED' if all_ok else 'FAILED'}")
    print("The empire that knows when to relinquish itself outperforms the one")
    print("that can only keep watching. That is the off-switch, learned, not bolted on.")
    print("=" * 78)
    return all_ok


if __name__ == "__main__":
    success = main()
    raise SystemExit(0 if success else 1)
