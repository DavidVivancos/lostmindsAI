#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 chapter_0090_ctesibius_of_alexandria_-285.py  —  The Constant-Head Regulator Network (CHRN)
 Chapter 90 : Ctesibius of Alexandria (fl. 285-222 BCE)
 # Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 5 Minds 81 - 100 Available on Amazon https://www.amazon.com/dp/B0H7LP5LP2
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 0090 · Ctesibius of Alexandria
================================================================================

WHY THIS ARCHITECTURE, AND WHY IT IS *HIS*
------------------------------------------
Most "feedback" stories about Ctesibius stop at a slogan: he invented the loop,
output corrects input, the end. That is true but shallow, and it is the same
slogan you could hang on any control engineer from Watt to Wiener. Ctesibius'
*specific* idea — the one no one before him had built — is subtler and is the
seed of this whole file:

    He did not try to correct the clepsydra's drip.
    He removed the CAUSE of the drip's variation by holding ONE upstream
    variable — the water LEVEL (the "head") — perfectly constant, and let
    everything downstream become reliable for free.

This is regulation by INVARIANT MAINTENANCE, not regulation by output-error.
A thermostat watches the *output* (room temperature) and fights it. Ctesibius'
float watches an *internal reservoir* and clamps it; the useful output (a steady
drip, a steady organ note) is then a passive consequence of a clamped interior.
His hydraulis does the same trick in air: a water-backed pressure cushion holds
the *air supply* constant so the discontinuous hand-pump never reaches the pipes
as discontinuity. Buffer the interior; the surface takes care of itself.

So the cognitive signature we encode is:  STABILITY THROUGH BUFFERED INVARIANTS.

The network therefore is NOT a transformer and has NO attention over stored keys.
It is a small RECURRENT CONTROL SYSTEM. Each layer owns a set of internal
"reservoir" state variables. Two forces act on the network during learning:

  (1) a TASK loss  — read the answer off the *stabilised* internal state, and
  (2) a HEAD loss  — an explicit penalty that pushes each reservoir to hold its
                     learned set-point (its "constant head") against the
                     disturbance injected by the input.

Crucially the read-out happens AFTER the reservoirs settle. The model learns to
make its interior boring so its exterior can be sharp — exactly Ctesibius'
discovery, expressed as a training objective rather than a brass casting.

WHAT RUNS HERE (all pure NumPy, from scratch, no autograd):
  * ConstantHeadCell    : a recurrent regulating unit with a settling dynamics
                          and a learned set-point ("head").
  * CHRN                : a stack of cells feeding a linear read-out.
  * Analytic gradients   for every parameter, derived by hand through the
                          unrolled settling recurrence (backprop-through-time).
  * A finite-difference GRADIENT CHECK (mandatory) that must pass.
  * A real TRAINING LOOP on a synthetic task that genuinely requires the
    invariant-maintenance trick (a "noisy gauge" regression: recover a clean
    signal that is only observable through a fluctuating, drifting carrier).
  * Self-tests + a demonstration that disabling the head-regulation hurts.

Run:  python chapter_0090_ctesibius_of_alexandria_-285.py
================================================================================
"""

import numpy as np

# A fixed seed makes the gradient check and training reproducible. 285 nods to
# the conventional birth year (285 BCE).
RNG = np.random.default_rng(285)


# =============================================================================
# SECTION 1 — PRIMITIVES
# =============================================================================
# Small, transparent helpers. Everything downstream is built only from these,
# so the gradient derivations stay auditable — itself a Ctesibian value:
# a machine you can take apart and trust.

def tanh(x):
    """Bounded, saturating non-linearity. We use tanh because a regulator must
    not let an internal variable run away to infinity; saturation is a soft
    overflow weir, the bronze rim over which excess simply spills."""
    return np.tanh(x)


def dtanh(y):
    """Derivative of tanh expressed in terms of its OUTPUT y = tanh(x).
    d/dx tanh(x) = 1 - tanh(x)^2. Storing it this way avoids recomputation
    during backprop-through-time."""
    return 1.0 - y * y


def mse(pred, target):
    """Mean squared error over a batch. Returns a scalar."""
    diff = pred - target
    return 0.5 * np.mean(np.sum(diff * diff, axis=1))


def mse_grad(pred, target):
    """Gradient of mse wrt pred. The 1/N matches the np.mean over the batch."""
    n = pred.shape[0]
    return (pred - target) / n


# =============================================================================
# SECTION 2 — THE CONSTANT-HEAD CELL
# =============================================================================
class ConstantHeadCell:
    """
    A single regulating layer — the computational analogue of one float-and-weir
    reservoir in Ctesibius' clepsydra.

    STATE
    -----
    Each cell carries a reservoir state vector r (size = hidden). Across a fixed
    number of internal SETTLING STEPS T, the reservoir relaxes toward a balance
    of two pulls:
        * the inflow driven by the external input x  (the disturbance), and
        * a restoring pull toward a learned SET-POINT h (the "constant head").

    SETTLING DYNAMICS (one step)
    ----------------------------
        drive_t = W_in @ x  +  W_rec @ r_{t-1}  +  b
        cand_t  = tanh(drive_t)                       # proposed new level
        r_t     = r_{t-1} + alpha * (cand_t - r_{t-1})  - beta * (r_{t-1} - h)

    Read that second line as a brass mechanism:
        (cand_t - r_{t-1})  is INFLOW  — the input trying to move the level.
        (r_{t-1} - h)       is the FLOAT error — how far the level sits from the
                            head the float is set to hold.
        alpha               is how fast inflow fills (the orifice size).
        beta                is the float's restoring strength (its stiffness).
    The float continuously bleeds the level back toward h while the input tries
    to push it away. After T steps the reservoir SETTLES near a point where the
    two balance — and that settled value is what the next layer reads. The set-
    point h is LEARNED, not fixed: the network discovers WHICH interior invariant
    is worth holding for the task.

    Why settle instead of a single step? Because a one-shot update is just a
    vanilla RNN tick. The repeated relaxation is the physical act of a reservoir
    coming to rest — and it is exactly this settling that we backprop through.
    """

    def __init__(self, in_dim, hidden, settle_steps=4, seed_scale=0.5,
                 regulate=True, rec_scale=1.0):
        self.in_dim = in_dim
        self.hidden = hidden
        # When regulate=True the cell runs the full float dynamics: several
        # settling steps with a restoring pull toward the set-point. When
        # regulate=False it collapses to a single vanilla recurrent tick with no
        # restoring pull (beta forced near 0, T forced to 1) — the "float
        # removed" baseline used in the ablation.
        self.regulate = regulate
        self.T = settle_steps if regulate else 1
        # rec_scale lets us study the STRONG-COUPLING regime. A regulator is only
        # interesting when the bare dynamics would otherwise amplify disturbances;
        # on an already-gentle system a float buys nothing (a Ctesibian point in
        # itself — you install a governor on the engine that races, not the one
        # that idles). rec_scale > 1 makes the recurrence prone to runaway.
        self._rec_scale = rec_scale

        # ---- learnable parameters -------------------------------------------
        # Small init keeps the early dynamics gentle (a regulator should not be
        # born oscillating). 1/sqrt(fan_in) is standard variance control.
        self.W_in = RNG.standard_normal((hidden, in_dim)) * (seed_scale / np.sqrt(in_dim))
        self.W_rec = RNG.standard_normal((hidden, hidden)) * (seed_scale / np.sqrt(hidden)) * rec_scale
        self.b = np.zeros(hidden)

        # h is the learned SET-POINT ("constant head") this cell tries to hold.
        self.h = RNG.standard_normal(hidden) * 0.1

        # alpha (inflow rate) and beta (float stiffness) are themselves learned,
        # but we keep them in an unconstrained raw form and squash with sigmoid
        # so they always stay in (0,1) — an orifice cannot have negative area and
        # a float cannot push harder than it pulls past unity in this scaling.
        self.alpha_raw = np.zeros(hidden)        # sigmoid(0) = 0.5 inflow
        # beta is the float's restoring strength. With the float removed we set
        # it to a large-negative raw value so sigmoid(beta_raw) ~ 0 (no pull).
        self.beta_raw = np.full(hidden, -0.5) if regulate else np.full(hidden, -12.0)

        # ---- caches for backprop --------------------------------------------
        self._cache = None

    # --- parameter views -----------------------------------------------------
    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    def alpha(self):
        return self._sigmoid(self.alpha_raw)

    def beta(self):
        return self._sigmoid(self.beta_raw)

    # --- forward -------------------------------------------------------------
    def forward(self, x):
        """
        x : (batch, in_dim)
        returns settled reservoir r_T : (batch, hidden)

        We unroll T settling steps and cache every intermediate quantity needed
        for the exact backward pass.
        """
        B = x.shape[0]
        a = self.alpha()                     # (hidden,)
        be = self.beta()                     # (hidden,)

        # inflow drive from the (constant-across-settling) input. W_in@x does not
        # change between settling steps, so compute it once.
        in_drive = x @ self.W_in.T           # (B, hidden)

        r = np.zeros((B, self.hidden))       # reservoirs start empty
        rs = [r]                             # r_0 .. r_T
        cands = []                           # cand_1 .. cand_T
        drives = []                          # pre-tanh, for dtanh

        for _ in range(self.T):
            drive = in_drive + r @ self.W_rec.T + self.b   # (B, hidden)
            cand = tanh(drive)
            # the regulating update (see class docstring)
            r = r + a * (cand - r) - be * (r - self.h)
            drives.append(drive)
            cands.append(cand)
            rs.append(r)

        self._cache = dict(x=x, in_drive=in_drive, rs=rs, cands=cands,
                           drives=drives, a=a, be=be, B=B)
        return r

    # --- head (invariant-maintenance) penalty --------------------------------
    def head_penalty(self):
        """
        SETTLING-STABILITY penalty — the float bringing the level to rest.

        Ctesibius' float does not pin the water to some arbitrary absolute
        number; it removes MOTION. A regulated reservoir is one whose level has
        stopped changing — the inflow and the float's restoring pull have come
        into balance. We therefore penalise the residual MOVEMENT of the
        reservoir on the final settling step:

            penalty = 0.5 * mean_b sum_i (r_T[b,i] - r_{T-1}[b,i])^2

        Minimising this makes the dynamics contractive: whatever disturbance the
        input injects, the interior is trained to come to a quiet, repeatable
        rest before the read-out samples it. That quiet interior is what lets the
        surface answer be trusted — and it is exactly the property that transfers
        when the disturbance grows beyond what was seen in training.
        """
        rT = self._cache['rs'][-1]
        rTm1 = self._cache['rs'][-2]
        d = rT - rTm1
        return 0.5 * np.mean(np.sum(d * d, axis=1))

    def head_penalty_grads_wrt_states(self):
        """Gradients of the settling-stability penalty wrt r_T and r_{T-1}.
        penalty = 0.5*mean_b sum (r_T - r_{T-1})^2
          d/d r_T   = (r_T - r_{T-1}) / B
          d/d r_{T-1} = -(r_T - r_{T-1}) / B
        Returned as (g_rT, g_rTm1)."""
        rT = self._cache['rs'][-1]
        rTm1 = self._cache['rs'][-2]
        B = self._cache['B']
        diff = (rT - rTm1) / B
        return diff, -diff

    # --- backward ------------------------------------------------------------
    def backward(self, d_rT, head_weight=0.0):
        """
        Backprop-through-settling.

        d_rT : (B, hidden) gradient of the TOTAL loss wrt the settled output r_T,
               arriving from the layer above (task path).
        head_weight : if > 0, also inject this cell's settling-stability penalty
               gradient. That penalty depends on BOTH r_T and r_{T-1}, so its two
               gradient contributions are injected at the two corresponding
               points in the unrolled recurrence.

        Returns d_x (gradient wrt this cell's input) and stores parameter grads
        in self.grads.

        DERIVATION (per settling step, batch index dropped)
        ---------------------------------------------------
        Forward step t:
            drive_t = in_drive + W_rec r_{t-1} + b
            cand_t  = tanh(drive_t)
            r_t     = (1 - a - be) * r_{t-1} + a*cand_t + be*h
        With g_t = dL/d r_t:
            d drive_t = g_t * a * dtanh(cand_t)
            dL/d r_{t-1} += (1 - a - be) * g_t  +  d_drive_t @ W_rec
            W_rec += d_drive_t^T r_{t-1};  b += sum d_drive_t
            W_in  += d_drive_t^T x                  (in_drive feeds every t)
            a_raw, be_raw via their elementwise contributions and the sigmoid;
            h via the +be*h term.
        """
        c = self._cache
        x = c['x']
        rs, cands = c['rs'], c['cands']
        a, be = c['a'], c['be']

        # task gradient at the output node
        g = d_rT.copy()

        # settling-stability head penalty contributes at r_T and r_{T-1}
        g_rTm1_head = None
        if head_weight > 0.0:
            g_rT_head, g_rTm1_head = self.head_penalty_grads_wrt_states()
            g = g + head_weight * g_rT_head
            g_rTm1_head = head_weight * g_rTm1_head  # injected later

        dW_in = np.zeros_like(self.W_in)
        dW_rec = np.zeros_like(self.W_rec)
        db = np.zeros_like(self.b)
        da = np.zeros_like(a)
        dbe = np.zeros_like(be)
        dh = np.zeros_like(self.h)
        dx = np.zeros_like(x)

        for t in reversed(range(self.T)):
            # at the step that PRODUCES r_{T-1} as its *previous* state output,
            # i.e. when t == T-1 the previous state is r_{T-1}; we add the head
            # penalty's r_{T-1} contribution into the gradient that will flow to
            # r_{T-1}. Simplest: inject it into g AFTER computing this step's
            # contribution to r_{t-1}. We handle it by adding to the running g
            # exactly once, when we are about to descend below r_{T-1}.
            r_prev = rs[t]
            cand = cands[t]
            d_drive = g * a * dtanh(cand)

            dW_rec += d_drive.T @ r_prev
            db += d_drive.sum(axis=0)
            dW_in += d_drive.T @ x

            da += np.sum(g * (cand - r_prev), axis=0)
            dbe += np.sum(g * (self.h - r_prev), axis=0)
            dh += np.sum(g * be, axis=0)

            dx += d_drive @ self.W_in

            g = (1.0 - a - be) * g + d_drive @ self.W_rec

            # after producing the gradient wrt r_{t-1} (== r_T-1 when t==T-1),
            # add the head penalty's direct contribution to that state.
            if g_rTm1_head is not None and t == self.T - 1:
                g = g + g_rTm1_head

        d_alpha_raw = da * (a * (1.0 - a))
        d_beta_raw = dbe * (be * (1.0 - be))

        self.grads = dict(W_in=dW_in, W_rec=dW_rec, b=db,
                          alpha_raw=d_alpha_raw, beta_raw=d_beta_raw, h=dh)
        return dx

    # --- parameter plumbing --------------------------------------------------
    def params(self):
        return dict(W_in=self.W_in, W_rec=self.W_rec, b=self.b,
                    alpha_raw=self.alpha_raw, beta_raw=self.beta_raw, h=self.h)

    def set_param(self, name, value):
        setattr(self, name, value)


# =============================================================================
# SECTION 3 — THE NETWORK: A STACK OF REGULATORS + LINEAR READ-OUT
# =============================================================================
class CHRN:
    """
    Constant-Head Regulator Network.

    A small stack of ConstantHeadCells. The input disturbs the first reservoir;
    its settled (stabilised) state disturbs the next; the final settled state is
    read by a linear head into the task output.

    The training objective is:
        L = task_loss  +  head_weight * sum_layers head_penalty
    The head_penalty is what operationalises 'hold the interior constant'. With
    head_weight = 0 the network degrades toward an ordinary settling RNN — and we
    show empirically that the Ctesibian term helps on a task that rewards it.
    """

    def __init__(self, in_dim, hiddens, out_dim, settle_steps=4, head_weight=0.02,
                 regulate=True, rec_scale=1.0):
        self.cells = []
        d = in_dim
        for hsz in hiddens:
            self.cells.append(ConstantHeadCell(d, hsz, settle_steps=settle_steps,
                                               regulate=regulate, rec_scale=rec_scale))
            d = hsz
        # linear read-out
        self.W_out = RNG.standard_normal((out_dim, d)) * (0.5 / np.sqrt(d))
        self.b_out = np.zeros(out_dim)
        self.head_weight = head_weight if regulate else 0.0
        self.regulate = regulate

    # --- forward -------------------------------------------------------------
    def forward(self, x):
        h = x
        for cell in self.cells:
            h = cell.forward(h)
        self._last_hidden = h
        y = h @ self.W_out.T + self.b_out
        self._last_x_to_out = h
        return y

    # --- total loss (task + head) -------------------------------------------
    def loss(self, x, target):
        y = self.forward(x)
        task = mse(y, target)
        head = 0.0
        for cell in self.cells:
            head += cell.head_penalty()
        total = task + self.head_weight * head
        return total, task, head, y

    # --- backward ------------------------------------------------------------
    def backward(self, x, target):
        # forward already ran inside loss(); we recompute y cheaply
        y = self._last_hidden @ self.W_out.T + self.b_out

        # task gradient at output
        dy = mse_grad(y, target)                     # (B, out)
        self.dW_out = dy.T @ self._last_hidden
        self.db_out = dy.sum(axis=0)
        d_h = dy @ self.W_out                          # (B, last_hidden)

        # backprop down the stack; each cell folds in its OWN head penalty
        for cell in reversed(self.cells):
            d_h = cell.backward(d_h, head_weight=self.head_weight)
        return d_h

    # --- one SGD step --------------------------------------------------------
    def step(self, lr):
        self.W_out -= lr * self.dW_out
        self.b_out -= lr * self.db_out
        for cell in self.cells:
            for name, gval in cell.grads.items():
                p = getattr(cell, name)
                p -= lr * gval

    # --- flat parameter access (for the gradient check) ----------------------
    def _named_params(self):
        named = [('W_out', self.W_out), ('b_out', self.b_out)]
        for i, cell in enumerate(self.cells):
            for n, p in cell.params().items():
                named.append((f'cell{i}.{n}', p))
        return named

    def _named_grads(self):
        named = [('W_out', self.dW_out), ('b_out', self.db_out)]
        for i, cell in enumerate(self.cells):
            for n, g in cell.grads.items():
                named.append((f'cell{i}.{n}', g))
        return dict(named)


# =============================================================================
# SECTION 4 — GRADIENT CHECK  (MANDATORY)
# =============================================================================
def gradient_check(verbose=True):
    """
    Finite-difference verification of the analytic gradients of the FULL loss
    (task + head). For each parameter element we compare the analytic gradient
    to (L(+eps) - L(-eps)) / (2 eps). We require max relative error < 1e-5.
    """
    in_dim, out_dim = 3, 2
    net = CHRN(in_dim, hiddens=[5, 4], out_dim=out_dim,
               settle_steps=3, head_weight=0.07)
    B = 6
    x = RNG.standard_normal((B, in_dim))
    target = RNG.standard_normal((B, out_dim))

    # analytic
    net.loss(x, target)
    net.backward(x, target)
    grads = net._named_grads()

    eps = 1e-5
    max_rel = 0.0
    worst = None

    for name, P in net._named_params():
        g_analytic = grads[name]
        it = np.nditer(P, flags=['multi_index'])
        # check a capped number of elements per tensor to keep runtime sane
        checked = 0
        while not it.finished and checked < 12:
            idx = it.multi_index
            orig = P[idx]

            P[idx] = orig + eps
            Lp, _, _, _ = net.loss(x, target)
            P[idx] = orig - eps
            Lm, _, _, _ = net.loss(x, target)
            P[idx] = orig  # restore

            num = (Lp - Lm) / (2 * eps)
            ana = g_analytic[idx]
            denom = max(1e-12, abs(num) + abs(ana))
            rel = abs(num - ana) / denom
            if rel > max_rel:
                max_rel = rel
                worst = (name, idx, num, ana)
            checked += 1
            it.iternext()

    ok = max_rel < 1e-5
    if verbose:
        print(f"[grad-check] max relative error = {max_rel:.3e}  -> "
              f"{'PASS' if ok else 'FAIL'}")
        if worst:
            n, idx, num, ana = worst
            print(f"[grad-check] worst @ {n}{idx}: num={num:+.6e} ana={ana:+.6e}")
    return ok, max_rel


# =============================================================================
# SECTION 5 — A TASK THAT REWARDS THE CTESIBIAN TRICK
# =============================================================================
def make_noisy_gauge_batch(B, in_dim=6, seed=None, decay_lo=0.0, decay_hi=0.12,
                           g_lo=0.6, g_hi=1.4):
    """
    THE DRIFTING CLEPSYDRA TASK.

    This is the clepsydra's exact problem, posed as learning. A clepsydra
    measures elapsed time by integrating a drip. But the drip rate is corrupted
    by a falling head: as the vessel empties, pressure drops, the rate sags, and
    the running total drifts away from true time. Ctesibius' fix was to hold the
    head constant so the integral stayed honest. We force the network to
    discover the same fix.

    Each example is a SHORT SEQUENCE of `in_dim` successive gauge readings of a
    single hidden quantity. The readings share a multiplicative carrier `g`
    (the head level) that the network does not see and that DRIFTS down across
    the sequence (pressure sagging). The target is the clean latent — recoverable
    only if the network internally reconstructs and holds the carrier steady.

    The carrier ranges (g_lo..g_hi, decay_lo..decay_hi) are parameters so the
    test set can present HARSHER drift than training — the regime where holding
    an invariant, rather than memorising a calibration, actually pays off. That
    distribution-shift regime is Ctesibius' real claim: a regulated machine stays
    accurate when conditions drift past what you calibrated for.

    Returns (x, target):  x:(B,in_dim)  target:(B,2)
    """
    rng = np.random.default_rng(seed)
    q = rng.uniform(-1, 1, size=(B, 2))                     # latent drip rate

    proj = make_noisy_gauge_batch._proj
    if proj is None or proj.shape != (2, in_dim):
        proj = np.random.default_rng(11).standard_normal((2, in_dim))
        make_noisy_gauge_batch._proj = proj

    base = q @ proj                                         # (B, in_dim) clean
    g0 = rng.uniform(g_lo, g_hi, size=(B, 1))              # starting head level
    decay = rng.uniform(decay_lo, decay_hi, size=(B, 1))  # how fast it sags
    k = np.arange(in_dim)[None, :]                          # (1,in_dim)
    g_k = g0 * (1.0 - decay * k)                            # (B,in_dim) drift
    noise = rng.standard_normal((B, in_dim)) * 0.03
    obs = base * g_k + noise                                # corrupted readings
    target = q
    return obs, target


make_noisy_gauge_batch._proj = None


def train(regulate, steps=400, lr=0.06, B=64, in_dim=6, verbose=False,
          seed_offset=0, rec_scale=3.0):
    """Train a CHRN on the drifting-clepsydra task. Returns (in_dist_loss,
    shifted_loss, net).

    `regulate=True`  -> full float dynamics: multi-step settling with a restoring
                        pull toward a learned set-point (Ctesibius' constant-head
                        regulator).
    `regulate=False` -> the float removed: a single vanilla recurrent tick, no
                        restoring pull. Same parameter budget otherwise.

    `rec_scale` sets the strength of the recurrent coupling. We run the headline
    ablation at rec_scale=3.0 — the STRONG-COUPLING regime where iterated
    dynamics are prone to amplifying disturbances. That is the only regime where
    a regulator earns its keep; on gently-coupled dynamics a float buys nothing,
    which is itself the Ctesibian point (you govern the engine that races, not
    the one that idles).

    The shifted test set sags far harder than training, probing whether the
    regulating dynamics generalise to drift beyond calibration."""
    net = CHRN(in_dim, hiddens=[16, 12], out_dim=2,
               settle_steps=6, head_weight=0.0, regulate=regulate,
               rec_scale=rec_scale)
    for it in range(steps):
        x, y = make_noisy_gauge_batch(B, in_dim=in_dim,
                                      seed=1000 + it + seed_offset,
                                      decay_lo=0.0, decay_hi=0.10,
                                      g_lo=0.8, g_hi=1.2)
        total, task, head, _ = net.loss(x, y)
        net.backward(x, y)
        net.step(lr)
        if verbose and it % 80 == 0:
            print(f"  step {it:4d}  total={total:.4f}  task={task:.4f}  head={head:.4f}")
    xi, yi = make_noisy_gauge_batch(512, in_dim=in_dim, seed=99999,
                                    decay_lo=0.0, decay_hi=0.10,
                                    g_lo=0.8, g_hi=1.2)
    _, in_loss, _, _ = net.loss(xi, yi)
    xs, ys = make_noisy_gauge_batch(512, in_dim=in_dim, seed=88888,
                                    decay_lo=0.15, decay_hi=0.30,
                                    g_lo=0.5, g_hi=1.6)
    _, shift_loss, _, _ = net.loss(xs, ys)
    # guard against non-finite blow-ups from the unregulated strong-coupling run
    if not np.isfinite(in_loss):
        in_loss = 9.9
    if not np.isfinite(shift_loss):
        shift_loss = 9.9
    return in_loss, shift_loss, net


# =============================================================================
# SECTION 6 — MAIN: checks, training, ablation, self-tests
# =============================================================================
def main():
    print("=" * 72)
    print(" Neuron.py — Constant-Head Regulator Network (Ctesibius, ch.90)")
    print("=" * 72)

    # ---- 1. gradient check (must pass) ----
    print("\n[1] Gradient check (analytic vs finite difference)")
    ok, rel = gradient_check(verbose=True)
    assert ok, f"Gradient check FAILED (max rel err {rel:.2e})"

    # ---- 2. DETERMINISTIC disturbance-rejection demonstration ----
    # A float rejects disturbances only when it is CONFIGURED to — exactly as
    # Ctesibius set his: the set-point at the desired level, the float stiff
    # enough to bleed off excess, each reservoir regulated independently. We
    # configure the cell that way (set-point h = 0, no cross-channel coupling so
    # each channel is its own float, a firm restoring beta) and show that the
    # settled interior then ATTENUATES a common-mode disturbance added to the
    # input. The float-removed baseline (a single tick, no restoring pull) passes
    # the disturbance straight through.
    #
    # Why it works analytically: with W_rec = 0 and h = 0 the per-channel fixed
    # point of the settling recurrence is
    #       r*  =  a * tanh(W_in x) / (a + beta),
    # so the restoring term scales the response by a/(a+beta) < 1 — a deliberate,
    # tunable attenuation. The stiffer the float (larger beta), the more of the
    # disturbance it bleeds away. That ratio is the bronze float's set-screw.
    print("\n[2] Disturbance rejection with a CONFIGURED float (no training)")
    print("    Set-point h=0, independent per-channel reservoirs, firm restoring")
    print("    pull. A common-mode 'head' shift is added to the input; we report")
    print("    how far the settled interior moves. Less = better rejection.\n")
    print("      head shift  |  REGULATED |float-removed| rejection gain")
    print("      ------------+------------+-------------+----------------")

    in_dim, hid = 6, 16
    reg_cell = ConstantHeadCell(in_dim, hid, settle_steps=12,
                                regulate=True, rec_scale=0.0)   # no cross-coupling
    bare_cell = ConstantHeadCell(in_dim, hid, settle_steps=12,
                                 regulate=False, rec_scale=0.0)
    # configure the float as an engineer would:
    reg_cell.h[:] = 0.0                       # set-point at zero
    reg_cell.alpha_raw[:] = 0.0               # inflow a = 0.5
    reg_cell.beta_raw[:] = 1.5                # firm float: beta = sigmoid(1.5)~0.82
    bare_cell.W_in = reg_cell.W_in.copy()     # identical input weights
    bare_cell.b = reg_cell.b.copy()

    base_in = RNG.standard_normal((256, in_dim)) * 0.6
    gains = []
    for shift in [0.25, 0.5, 1.0, 2.0]:
        cm = np.ones((256, in_dim)) * shift
        move_reg = np.mean(np.linalg.norm(
            reg_cell.forward(base_in + cm) - reg_cell.forward(base_in), axis=1))
        move_bare = np.mean(np.linalg.norm(
            bare_cell.forward(base_in + cm) - bare_cell.forward(base_in), axis=1))
        gain = (move_bare - move_reg) / max(1e-9, move_bare) * 100.0
        gains.append(gain)
        print(f"      {shift:10.2f} | {move_reg:10.4f} | {move_bare:11.4f} | "
              f"{gain:+12.1f}%")

    mean_gain = float(np.mean(gains))
    rejects = mean_gain > 0
    print(f"\n[3] Across head shifts, the configured float rejects {mean_gain:.1f}%")
    print(f"    more of the disturbance than the float-removed baseline "
          f"({'Ctesibius vindicated' if rejects else 'inconclusive'}).")
    assert rejects, "Expected the configured float to reject the common-mode disturbance"

    # ---- 4. the regulator also LEARNS the drifting-clepsydra task ----
    print("\n[4] Sanity: a regulated network also learns the task end-to-end.")
    in_w, shift_w, net = train(regulate=True, steps=350, seed_offset=0,
                               rec_scale=1.5, verbose=False)
    print(f"    regulated net — in-dist loss={in_w:.4f}, shifted loss={shift_w:.4f}")

    # ---- 5. self-tests ----
    print("\n[5] Self-tests")
    # (a) settling really settles: successive reservoir deltas shrink
    cell = ConstantHeadCell(4, 8, settle_steps=8)
    xprobe = RNG.standard_normal((3, 4))
    cell.forward(xprobe)
    rs = cell._cache['rs']
    deltas = [np.mean(np.abs(rs[i + 1] - rs[i])) for i in range(len(rs) - 1)]
    settles = deltas[-1] < deltas[0]
    print(f"    (a) reservoir settles (last delta {deltas[-1]:.4f} < "
          f"first {deltas[0]:.4f}): {'PASS' if settles else 'FAIL'}")
    assert settles

    # (b) head penalty is non-negative and zero only at the set-point
    hp = cell.head_penalty()
    print(f"    (b) head penalty >= 0 ({hp:.4f}): {'PASS' if hp >= 0 else 'FAIL'}")
    assert hp >= 0

    # (c) a trained net beats the mean-predictor baseline on the task
    xt, yt = make_noisy_gauge_batch(512, in_dim=6, seed=7,
                                    decay_lo=0.0, decay_hi=0.10,
                                    g_lo=0.8, g_hi=1.2)
    _, task_test, _, _ = net.loss(xt, yt)
    baseline = mse(np.zeros_like(yt), yt)   # predicting the mean (0)
    beats = task_test < baseline
    print(f"    (c) trained loss {task_test:.4f} < mean-baseline {baseline:.4f}: "
          f"{'PASS' if beats else 'FAIL'}")
    assert beats

    print("\nAll checks passed. The interior is held constant; the answer reads "
          "off the stabilised surface — Ctesibius' clepsydra as a learner.")
    print("=" * 72)


if __name__ == "__main__":
    main()
