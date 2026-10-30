#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0295 · Bhaskara II (Bhaskaracharya) (c. 1114 CE, Vijjadavida (region disputed among Bijapur/Karnataka, Patan-Chalisgaon and Beed/Maharashtra by modern scholarship), d. c. 1185 CE, Ujjain)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 15, Minds 281-300 Available on Amazon https://www.amazon.com/dp/B0HMQ7TVJM
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0295_bhaskara_ii_1114 - Bhaskara II (Bhaskaracharya)
 CHAKRAVALANET -- a from-scratch differentiable architecture built on the
 distinctive mathematical moves of Bhaskara II (Bhaskaracharya), b. 1114 CE,
 Vijjadavida (region disputed among Bijapur/Karnataka, Patan-Chalisgaon and
 Beed/Maharashtra by modern scholarship), d. c. 1185 CE, Ujjain.
================================================================================

WHY THIS SHAPE, NOT A TRANSFORMER

Bhaskara's mathematics -- in the Lilavati, Bijaganita, and the Siddhanta
Shiromani (Grahaganita + Goladhyaya) -- returns again and again to a single
cognitive move: when a computation hits a point where the direct path is
blocked, he does not refuse the operation and does not silently paper over
it either. He *names* the resulting object, *carries* it forward without
collapsing it prematurely, and resolves it through a disciplined, bounded
procedure. Three documented instances of this move give the three custom
layers below their shape:

  1. KHAHARA (Bijaganita, Lilavati v.46-47). A quantity divided by zero is
     not an error state; Bhaskara names it khahara, a distinct kind of
     quantity, and states explicitly that adding or subtracting any finite
     amount leaves it unaltered ("as no change occurs in the infinite and
     immutable when worlds are formed or dissolved," Bijaganita, tr.
     Colebrooke 1817). He also gives the companion rule for khaguna
     (zero-multiplied quantities): such a term must be RETAINED, not
     collapsed to zero, until the very end of a derivation, in case later
     steps cancel the zero that produced it. This is, in effect, a rule for
     deferred symbolic evaluation, five and a half centuries before anyone
     had a word for "lazy evaluation."
        -> KhaharaNorm and DeferredGateUnit below.

  2. CHAKRAVALA (Bijaganita). Where Brahmagupta (628 CE) could solve the
     indeterminate "square-nature" equation x^2 - N y^2 = 1 for special N,
     and Jayadeva (c. 950-1000 CE, work now lost, known only through a
     20-verse quotation) generalised the approach, Bhaskara gave the full
     worked algorithm and its name: chakra, "wheel." Given one triple
     (a, b, k) with a^2 - N b^2 = k, compose it with a trivial triple
     (m, 1, m^2-N) via Brahmagupta's identity, choose m to make the result
     divisible by k while minimising |m^2-N|, and repeat. No direct
     solution exists in general; the wheel is a bounded, verifiable detour
     around that fact. Bhaskara's own worked example, x^2 - 61y^2 = 1, gives
     minimal solution x=1,766,319,049, y=226,153,980 -- the same problem
     Fermat posed to Frenicle de Bessy as a challenge in 1657, five hundred
     years later, unsolved in Europe until Euler and Lagrange.
        -> the classical chakravala_solve() function, AND the differentiable
           ChakravalaRefine block (a continuous relaxation of the same
           "compose, then keep whichever candidate is closer to the target"
           logic, using a soft selection instead of a discrete one).

  3. TATKALIKA-GATI ("of the instant," Siddhanta Shiromani, Grahaganita).
     Bhaskara distinguishes sthula gati (the coarse, tabulated motion of a
     planet over a day) from tatkalika gati (the motion it would have "were
     its velocity, for that instant, to remain uniform"). He derives, by an
     ordinary proportion argument on shrinking arcs (not by invoking a
     mystical infinitesimal), the relation equivalent in modern notation to
     delta(sin theta) = cos theta * delta theta, and is explicit that it
     holds only for small delta theta -- and separately notes that at the
     "resting point" of a planet's corrected longitude, this instantaneous
     difference vanishes. (Predecessors -- Manjula, Aryabhata II -- already
     used the term tatkalika for this kind of quantity; Bhaskara's specific
     contribution, per Datta & Singh's history of the topic, is proving the
     formula by similar-triangle proportion rather than assuming it.)
        -> TatkalikaHead below: an explicit local-proportion derivative
           estimate used both to step the prediction forward and to flag
           turning points where that derivative crosses zero.

WHAT THIS IS NOT: this is not a general theory of the derivative (no chain
rule for arbitrary functions is claimed for Bhaskara, and none is claimed
here beyond the specific proportion argument), and the chakravala solver
below implements the plain "minimise |m^2-N|" rule without Brahmagupta's
extra shortcuts for k in {+-1,+-2,+-4} -- so it converges cleanly for many N
(verified below) but not all; where it does not, it says so rather than
silently returning a wrong answer.

Pure NumPy. Manual forward/backward for every custom layer. A finite-
difference gradient check is mandatory and is run on the full assembled
network, not just isolated layers, before any training happens.
================================================================================
"""

import numpy as np
import math

RNG_SEED = 42


# ==============================================================================
# PART 0 -- THE CHAKRAVALA METHOD, THE CLASSICAL (NON-NEURAL) ALGORITHM
# ==============================================================================

def chakravala_solve(N, max_steps=200, verbose=False):
    """Solve x^2 - N*y^2 = 1 for the smallest positive integer solution using
    the historical cyclic method, following Bhaskara's Bijaganita.

    Given a triple (a, b, k) with a^2 - N*b^2 = k, compose it with the
    trivial triple (m, 1, m^2-N) via Brahmagupta's identity:
        A = (a*m + N*b) / k,  B = (a + b*m) / k,  K = (m^2 - N) / k
    where m is chosen, among all m with (a + b*m) divisible by k, to
    minimise |m^2 - N|. Iterate until k = 1.

    This is the plain single-branch rule described in the verses; it does
    NOT implement Brahmagupta's extra shortcut for reaching k=1 in one step
    whenever an intermediate |k| lands on {1,2,4}, so it will occasionally
    cycle without converging (this is flagged, not hidden) on N for which
    that shortcut is load-bearing.
    """
    r = math.isqrt(N)
    if r * r == N:
        raise ValueError("N must not be a perfect square (no nontrivial Pell solution).")

    m0 = round(math.sqrt(N))
    a, b, k = m0, 1, m0 * m0 - N
    if k == 0:
        raise ValueError("degenerate starting triple")

    seen = set()
    steps = 0
    while k != 1:
        steps += 1
        if steps > max_steps:
            raise RuntimeError(f"chakravala_solve: no convergence for N={N} in {max_steps} steps "
                                f"(this N likely needs Brahmagupta's k in {{1,2,4}} shortcut, "
                                f"not implemented here)")
        ak = abs(k)
        base = None
        for r_ in range(ak):
            if (b * r_ + a) % ak == 0:
                base = r_
                break
        if base is None:
            raise RuntimeError("no valid residue class found (unexpected for a genuine triple)")
        t_center = (math.sqrt(N) - base) / ak
        candidates = []
        for t in range(math.floor(t_center) - 2, math.ceil(t_center) + 3):
            m = base + t * ak
            candidates.append((abs(m * m - N), abs(m), m))
        candidates.sort()
        m = candidates[0][2]

        A = (a * m + N * b) // k
        B = (a + b * m) // k
        K = (m * m - N) // k
        a, b, k = A, B, K
        if verbose:
            print(f"    step {steps:2d}:  m={m:6d}  ->  a={a}, b={b}, k={k}")
        state = (a, b, k)
        if state in seen:
            raise RuntimeError(f"chakravala_solve: cycled without reaching k=1 for N={N}")
        seen.add(state)

    if a < 0:
        a, b = -a, -b
    return a, b, steps


# ==============================================================================
# PART 1 -- BASIC BUILDING BLOCK
# ==============================================================================

class Linear:
    """Ordinary affine layer, used for embedding, the deferred-carry folds,
    and the three output heads."""

    def __init__(self, d_in, d_out, rng):
        self.W = rng.standard_normal((d_in, d_out)) * (1.0 / math.sqrt(d_in))
        self.b = np.zeros(d_out)

    def params(self):
        return {'W': self.W, 'b': self.b}

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, dout):
        self.grads = {'W': self.x.T @ dout, 'b': np.sum(dout, axis=0)}
        return dout @ self.W.T


# ==============================================================================
# PART 2 -- KHAHARA NORM
# A normalization layer that does not paper over near-zero variance with an
# epsilon: it explicitly detects the degenerate ("khahara") regime and routes
# it to a dedicated, learned, named representation instead.
# ==============================================================================

class KhaharaNorm:
    def __init__(self, dim, rng, sharpness=20.0, sigma_floor=1e-3):
        self.dim = dim
        self.sharpness = sharpness
        self.sigma_floor = sigma_floor
        self.gamma = np.ones(dim)
        self.beta = np.zeros(dim)
        self.khahara = rng.standard_normal(dim) * 0.01  # the named absorbing value

    def params(self):
        return {'gamma': self.gamma, 'beta': self.beta, 'khahara': self.khahara}

    def forward(self, x):
        mu = np.mean(x, axis=1, keepdims=True)
        var = np.mean((x - mu) ** 2, axis=1, keepdims=True)
        sigma = np.sqrt(var + 1e-12)
        xhat = (x - mu) / (sigma + 1e-8)
        gate_in = self.sharpness * (self.sigma_floor - sigma)
        m = 1.0 / (1.0 + np.exp(-gate_in))
        out = (1 - m) * (self.gamma * xhat + self.beta) + m * self.khahara
        self.cache = (x, mu, var, sigma, xhat, m, gate_in)
        return out

    def backward(self, dout):
        x, mu, var, sigma, xhat, m, gate_in = self.cache
        _, D = x.shape
        normal_branch = self.gamma * xhat + self.beta
        dm = np.sum(dout * (self.khahara - normal_branch), axis=1, keepdims=True)
        dsig_gate = dm * m * (1 - m) * (-self.sharpness)
        dgamma = np.sum(dout * (1 - m) * xhat, axis=0)
        dbeta = np.sum(dout * (1 - m), axis=0)
        dkhahara = np.sum(dout * m, axis=0)
        dxhat = dout * (1 - m) * self.gamma
        dsigma = dsig_gate + np.sum(dxhat * -(x - mu) / (sigma + 1e-8) ** 2, axis=1, keepdims=True)
        dvar = dsigma * 0.5 / sigma
        dx = dxhat / (sigma + 1e-8)
        dmu = -np.sum(dx, axis=1, keepdims=True)
        dx = dx + dvar * 2 * (x - mu) / D
        dmu = dmu + np.sum(dvar * (-2) * (x - mu) / D, axis=1, keepdims=True)
        dx = dx + dmu / D
        self.grads = {'gamma': dgamma, 'beta': dbeta, 'khahara': dkhahara}
        return dx


# ==============================================================================
# PART 3 -- DEFERRED GATE UNIT
# The complement of a sigmoid gate is not thrown away. It is returned as a
# "khaguna" carry term, exactly as Bhaskara's rule says a zero-multiplied
# term must be retained and only resolved at the end of the computation.
# ==============================================================================

class DeferredGateUnit:
    def __init__(self, dim, rng):
        self.Wg = rng.standard_normal((dim, dim)) * (1.0 / math.sqrt(dim))
        self.bg = np.zeros(dim)
        self.Wv = rng.standard_normal((dim, dim)) * (1.0 / math.sqrt(dim))
        self.bv = np.zeros(dim)

    def params(self):
        return {'Wg': self.Wg, 'bg': self.bg, 'Wv': self.Wv, 'bv': self.bv}

    def forward(self, x):
        gate_logit = x @ self.Wg + self.bg
        g = 1.0 / (1.0 + np.exp(-gate_logit))
        v = np.tanh(x @ self.Wv + self.bv)
        y = g * v
        carry = (1 - g) * v          # <- the khaguna term: not discarded
        self.cache = (x, g, v)
        return y, carry

    def backward(self, dy, dcarry):
        x, g, v = self.cache
        dv = dy * g + dcarry * (1 - g)
        dg = dy * v - dcarry * v
        dgate_logit = dg * g * (1 - g)
        dvraw = dv * (1 - v ** 2)
        self.grads = {
            'Wg': x.T @ dgate_logit, 'bg': np.sum(dgate_logit, axis=0),
            'Wv': x.T @ dvraw, 'bv': np.sum(dvraw, axis=0),
        }
        return dgate_logit @ self.Wg.T + dvraw @ self.Wv.T


# ==============================================================================
# PART 4 -- CHAKRAVALA REFINE
# A bounded number of refinement cycles. Each cycle proposes a candidate
# state via a fixed composition transform (the neural analogue of composing
# with the trivial triple), then a SOFT, differentiable selection keeps
# whichever of {current state, candidate} sits closer to a learned target --
# the continuous relaxation of "choose the m that minimises |m^2-N|."
# ==============================================================================

class ChakravalaRefine:
    def __init__(self, dim, rng, K=4, temperature=0.5):
        self.dim, self.K, self.T = dim, K, temperature
        self.Wc = rng.standard_normal((dim, dim)) * (1.0 / math.sqrt(dim))
        self.bc = np.zeros(dim)
        self.target = rng.standard_normal(dim) * 0.1

    def params(self):
        return {'Wc': self.Wc, 'bc': self.bc, 'target': self.target}

    @staticmethod
    def _softmax2(logits):
        z = logits - np.max(logits, axis=1, keepdims=True)
        e = np.exp(z)
        return e / np.sum(e, axis=1, keepdims=True)

    def forward(self, h0):
        h = h0
        self.caches = []
        for _ in range(self.K):
            pre = h @ self.Wc + self.bc
            cand = h + np.tanh(pre)
            r_keep = np.sum((h - self.target) ** 2, axis=1)
            r_cand = np.sum((cand - self.target) ** 2, axis=1)
            logits = np.stack([-r_keep, -r_cand], axis=1) / self.T
            w = self._softmax2(logits)
            h_new = w[:, 0:1] * h + w[:, 1:2] * cand
            self.caches.append((h, pre, cand, w))
            h = h_new
        return h

    def backward(self, dh_final):
        dWc = np.zeros_like(self.Wc)
        dbc = np.zeros_like(self.bc)
        dtarget = np.zeros_like(self.target)
        dh_next = dh_final
        for h, pre, cand, w in reversed(self.caches):
            dh_new = dh_next
            dw0 = np.sum(dh_new * h, axis=1, keepdims=True)
            dw1 = np.sum(dh_new * cand, axis=1, keepdims=True)
            dh_from_keep = dh_new * w[:, 0:1]
            dcand = dh_new * w[:, 1:2]
            dw = np.concatenate([dw0, dw1], axis=1)
            dot = np.sum(dw * w, axis=1, keepdims=True)
            dlogits = w * (dw - dot) / self.T
            dr_keep, dr_cand = -dlogits[:, 0:1], -dlogits[:, 1:2]
            dh_from_rkeep = dr_keep * 2 * (h - self.target)
            dtarget += np.sum(-dr_keep * 2 * (h - self.target), axis=0)
            dcand_from_rcand = dr_cand * 2 * (cand - self.target)
            dtarget += np.sum(-dr_cand * 2 * (cand - self.target), axis=0)
            dcand_total = dcand + dcand_from_rcand
            dpre = dcand_total * (1 - np.tanh(pre) ** 2)
            dWc += h.T @ dpre
            dbc += np.sum(dpre, axis=0)
            dh_from_pre = dpre @ self.Wc.T
            dh_next = dh_from_keep + dh_from_rkeep + dcand_total + dh_from_pre
        self.grads = {'Wc': dWc, 'bc': dbc, 'target': dtarget}
        return dh_next


# ==============================================================================
# PART 5 -- FULL NETWORK
# ==============================================================================

class ChakravalaNet:
    """
    x_window (raw signal history)
        -> [+ explicit tatkalika finite-difference input feature]
        -> Linear embed
        -> KhaharaNorm
        -> DeferredGateUnit x2        (each spins off a khaguna carry)
        -> ChakravalaRefine (K cycles)
        -> fold the two carries back in (the deferred collapse, "till the end")
        -> three heads: base value, instantaneous rate, turning-point logit
        -> final_pred = base + rate * delta      (an explicit Euler step,
           i.e. "the value at a nearby instant is the value plus the
           instantaneous rate times the small interval" -- tatkalika-gati
           made into an output layer instead of a metaphor)
    """

    def __init__(self, window, hidden, K=4, delta=0.1, seed=RNG_SEED):
        rng = np.random.default_rng(seed)
        self.window, self.hidden, self.delta = window, hidden, delta
        d_in = window + 1
        self.embed = Linear(d_in, hidden, rng)
        self.norm = KhaharaNorm(hidden, rng)
        self.gate1 = DeferredGateUnit(hidden, rng)
        self.gate2 = DeferredGateUnit(hidden, rng)
        self.refine = ChakravalaRefine(hidden, rng, K=K)
        self.fold1 = Linear(hidden, hidden, rng)
        self.fold2 = Linear(hidden, hidden, rng)
        self.head_base = Linear(hidden, 1, rng)
        self.head_rate = Linear(hidden, 1, rng)
        self.head_turn = Linear(hidden, 1, rng)
        self._layers = [
            ('embed', self.embed), ('norm', self.norm),
            ('gate1', self.gate1), ('gate2', self.gate2),
            ('refine', self.refine), ('fold1', self.fold1), ('fold2', self.fold2),
            ('head_base', self.head_base), ('head_rate', self.head_rate),
            ('head_turn', self.head_turn),
        ]

    def all_params(self):
        return {f"{n}.{k}": v for n, l in self._layers for k, v in l.params().items()}

    def all_grads(self):
        return {f"{n}.{k}": v for n, l in self._layers for k, v in l.grads.items()}

    def forward(self, x_window):
        d0 = (x_window[:, -1:] - x_window[:, -2:-1]) / self.delta
        x_aug = np.concatenate([x_window, d0], axis=1)
        h0 = self.embed.forward(x_aug)
        h1 = self.norm.forward(h0)
        h2, carry1 = self.gate1.forward(h1)
        h3, carry2 = self.gate2.forward(h2)
        h4 = self.refine.forward(h3)
        h5 = h4 + self.fold1.forward(carry1) + self.fold2.forward(carry2)
        base = self.head_base.forward(h5)
        rate = self.head_rate.forward(h5)
        turn_logit = self.head_turn.forward(h5)
        final_pred = base + rate * self.delta
        return final_pred, rate, turn_logit

    def backward(self, dfinal_pred, drate_extra, dturn_logit):
        dbase = dfinal_pred
        drate = dfinal_pred * self.delta + drate_extra
        dh5 = (self.head_base.backward(dbase) + self.head_rate.backward(drate)
               + self.head_turn.backward(dturn_logit))
        dcarry1 = self.fold1.backward(dh5)
        dcarry2 = self.fold2.backward(dh5)
        dh3 = self.refine.backward(dh5)
        dh2 = self.gate2.backward(dh3, dcarry2)
        dh1 = self.gate1.backward(dh2, dcarry1)
        dh0 = self.norm.backward(dh1)
        return self.embed.backward(dh0)


# ==============================================================================
# PART 6 -- FINITE-DIFFERENCE GRADIENT CHECK (mandatory, run on full network)
# ==============================================================================

def numeric_grad(loss_fn, p, eps=1e-5):
    g = np.zeros_like(p)
    it = np.nditer(p, flags=['multi_index'])
    for _ in it:
        idx = it.multi_index
        old = p[idx]
        p[idx] = old + eps
        f1 = loss_fn()
        p[idx] = old - eps
        f2 = loss_fn()
        p[idx] = old
        g[idx] = (f1 - f2) / (2 * eps)
    return g


def run_gradient_check(net, batch=5, tol=2e-3, seed=7, verbose=True):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((batch, net.window)) * 0.3 + 1.0
    w_final = rng.standard_normal((batch, 1))
    w_rate = rng.standard_normal((batch, 1))
    w_turn = rng.standard_normal((batch, 1))

    def loss_fn():
        fp, rate, turn = net.forward(x)
        return float(np.sum(fp * w_final) + np.sum(rate * w_rate) + np.sum(turn * w_turn))

    net.forward(x)
    net.backward(w_final, w_rate, w_turn)
    grads = net.all_grads()
    params = net.all_params()

    worst = 0.0
    for name, p in params.items():
        num = numeric_grad(loss_fn, p)
        ana = grads[name]
        denom = np.max(np.abs(num))
        err = np.max(np.abs(num - ana)) / (denom + 1e-8)
        if denom > 1e-6:
            worst = max(worst, err)
        if verbose:
            flag = "OK" if (err < tol or denom < 1e-6) else "FAIL"
            print(f"    {name:22s} rel-err={err:.3e}  {flag}")
    return worst


# ==============================================================================
# PART 7 -- SYNTHETIC DATA: a toy "equation of the center" style periodic
# apparent-motion signal, of the kind Bhaskara's Grahaganita corrects for.
# ==============================================================================

def make_signal(t, A1=1.0, w1=1.3, p1=0.4, A2=0.4, w2=3.1, p2=1.1):
    val = A1 * np.sin(w1 * t + p1) + A2 * np.sin(w2 * t + p2)
    dval = A1 * w1 * np.cos(w1 * t + p1) + A2 * w2 * np.cos(w2 * t + p2)
    return val, dval


def build_dataset(n_points, window, h, t0, turn_frac=0.15):
    ts = t0 + h * np.arange(n_points + window + 1)
    vals, dvals = make_signal(ts)
    X, Y, R = [], [], []
    for i in range(n_points):
        X.append(vals[i:i + window])
        Y.append(vals[i + window])
        R.append(dvals[i + window - 1])
    X = np.array(X)
    Y = np.array(Y).reshape(-1, 1)
    R = np.array(R).reshape(-1, 1)
    rmax = np.max(np.abs(R))
    Turn = (np.abs(R) < turn_frac * rmax).astype(float)
    return X, Y, R, Turn


# ==============================================================================
# PART 8 -- TRAINING
# ==============================================================================

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def train(net, Xtr, Ytr, Rtr, Ttr, epochs=40, batch_size=32, lr=0.01,
          lam_rate=0.5, lam_turn=0.3, seed=123, verbose=True):
    rng = np.random.default_rng(seed)
    params = net.all_params()
    m_state = {k: np.zeros_like(v) for k, v in params.items()}
    v_state = {k: np.zeros_like(v) for k, v in params.items()}
    b1, b2, eps = 0.9, 0.999, 1e-8
    step = 0
    history = []

    n = Xtr.shape[0]
    for epoch in range(epochs):
        perm = rng.permutation(n)
        ep_loss = 0.0
        for bi in range(0, n, batch_size):
            idx = perm[bi:bi + batch_size]
            xb, yb, rb, tb = Xtr[idx], Ytr[idx], Rtr[idx], Ttr[idx]
            fp, rate, turn_logit = net.forward(xb)
            err = fp - yb
            rerr = rate - rb
            p = sigmoid(turn_logit)
            loss_main = np.mean(err ** 2)
            loss_rate = np.mean(rerr ** 2)
            loss_turn = -np.mean(tb * np.log(p + 1e-9) + (1 - tb) * np.log(1 - p + 1e-9))
            total = loss_main + lam_rate * loss_rate + lam_turn * loss_turn
            ep_loss += total * len(idx)

            bsz = len(idx)
            dfp = 2 * err / bsz
            drate_extra = lam_rate * 2 * rerr / bsz
            dturn = lam_turn * (p - tb) / bsz
            net.backward(dfp, drate_extra, dturn)

            step += 1
            grads = net.all_grads()
            for k, v in params.items():
                g = grads[k]
                m_state[k] = b1 * m_state[k] + (1 - b1) * g
                v_state[k] = b2 * v_state[k] + (1 - b2) * (g * g)
                mhat = m_state[k] / (1 - b1 ** step)
                vhat = v_state[k] / (1 - b2 ** step)
                v -= lr * mhat / (np.sqrt(vhat) + eps)

        history.append(ep_loss / n)
        if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
            print(f"    epoch {epoch:3d}   loss={history[-1]:.5f}")
    return history


# ==============================================================================
# PART 9 -- SELF-TESTS
# ==============================================================================

def main():
    print("=" * 78)
    print("PART A -- classical chakravala method (verification against the")
    print("          historical record)")
    print("=" * 78)
    x, y, steps = chakravala_solve(61, verbose=True)
    print(f"\n  N=61:  x={x}  y={y}  (found in {steps} cycles)")
    assert x == 1766319049 and y == 226153980, "does not match Bhaskara's own worked answer"
    assert x * x - 61 * y * y == 1
    print("  MATCHES the minimal solution given in Bhaskara's Bijaganita "
          "(the same problem Fermat posed to Frenicle de Bessy in 1657).")

    other_Ns = [3, 5, 10, 11, 17, 23, 26, 33]
    print(f"\n  spot-checking the algorithm on other non-square N: {other_Ns}")
    for N in other_Ns:
        xn, yn, sn = chakravala_solve(N)
        assert xn * xn - N * yn * yn == 1 and yn != 0
        print(f"    N={N:3d}  x={xn:6d}  y={yn:4d}  steps={sn}  (verified x^2-{N}y^2=1)")

    print("\n" + "=" * 78)
    print("PART B -- finite-difference gradient check on the FULL assembled")
    print("          differentiable network (KhaharaNorm + DeferredGateUnit x2")
    print("          + ChakravalaRefine + TatkalikaHead), not just isolated layers")
    print("=" * 78)
    net = ChakravalaNet(window=8, hidden=16, K=4, delta=0.1)
    worst = run_gradient_check(net, batch=5, tol=2e-3, verbose=True)
    print(f"\n  worst relative error across all {len(net.all_params())} parameter "
          f"tensors: {worst:.3e}")
    assert worst < 2e-3, "gradient check failed"
    print("  GRADIENT CHECK PASSED.")

    print("\n" + "=" * 78)
    print("PART C -- training on a synthetic 'equation of the center' style")
    print("          periodic signal (the kind of apparent-motion correction")
    print("          curve Bhaskara's Grahaganita computes)")
    print("=" * 78)
    net2 = ChakravalaNet(window=8, hidden=16, K=4, delta=0.1, seed=7)
    Xtr, Ytr, Rtr, Ttr = build_dataset(400, window=8, h=0.1, t0=0.0)
    Xte, Yte, Rte, Tte = build_dataset(150, window=8, h=0.1, t0=80.0)
    history = train(net2, Xtr, Ytr, Rtr, Ttr, epochs=40, verbose=True)

    fp_te, rate_te, turn_te = net2.forward(Xte)
    test_mse = float(np.mean((fp_te - Yte) ** 2))
    signal_var = float(np.var(Yte))
    p_te = sigmoid(turn_te)
    turn_acc = float(np.mean((p_te > 0.5).astype(float) == Tte))

    print(f"\n  loss:  epoch 0 = {history[0]:.5f}   ->   epoch {len(history)-1} = {history[-1]:.5f}")
    print(f"  held-out test MSE on next-value prediction: {test_mse:.5f}  "
          f"(signal variance: {signal_var:.5f})")
    print(f"  held-out turning-point classification accuracy: {turn_acc:.3f}")

    assert history[-1] < history[0] * 0.25, "training loss did not decrease enough"
    assert test_mse < 0.15 * signal_var, "held-out prediction error too high relative to signal scale"
    assert turn_acc > 0.75, "turning-point detector accuracy too low"
    print("\n  ALL TRAINING SELF-TESTS PASSED.")

    print("\n" + "=" * 78)
    print("ALL SELF-TESTS PASSED: chakravala solver verified against the")
    print("historical record; full-network gradient check passed; training")
    print("converges and generalises on held-out data.")
    print("=" * 78)


if __name__ == "__main__":
    main()
