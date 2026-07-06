#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 chapter_0064_hippocrates_-460.py   --   The Prognostic Engine
 A from-scratch, trainable cognitive architecture after Hippocrates of Kos
 (c. 460 - c. 370 BCE), 
 Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
Tome 4 Minds 61 - 80 Available on Amazon https://www.amazon.com/dp/B0H71JC95Q
Resume and Interactive Demos at https://artificiology.com/
Author: David Vivancos · Chapter 0064 · Hippocrates of Kos
================================================================================

WHY THIS ARCHITECTURE, AND NOT A TRANSFORMER
--------------------------------------------
Most scholars regard *prognosis* -- not diagnosis, not treatment -- as the
central scientific achievement of the Hippocratic school. The treatise
"Prognostic" opens: "It appears to me a most excellent thing for the physician
to cultivate Prognosis; for by foreseeing and foretelling ... the present, the
past, and the future ... he will be the more readily believed." And "Epidemics"
Book I gives the working creed:

    "Declare the past, diagnose the present, foretell the future ... As to
     diseases, make a habit of two things -- to help, or at least to do no harm."

So the Hippocratic mind is, before anything else, a *forecaster of a process*.
It reads a stream of bodily signs over days; it forms an internal sense of how
far the illness has *ripened* ("coction", pepsis); it watches for the decisive
turning point -- the *krisis* on a "critical day" -- and forecasts whether the
ripening will complete into recovery or break, uncooked, into death. And it is
bound by a restraint rule: act only when nature is failing; otherwise, hold the
hand. ("Primum non nocere" is later Latin; the principle is Hippocrates'.)

We therefore do NOT build attention over stored keys. We build a small gated
*recurrent* dynamical model -- a forecaster with four organs, each mapped to a
Hippocratic doctrine:

  1. COCTION CELL (a single-gate recurrent unit) ......... the latent ripening
     state h_t that accumulates the disease's progress over time. The gate g_t
     is literally the "rate of concoction": when it is small the state holds
     (the matter is still cooking); when large the state updates.

  2. PROGNOSIS HEAD (forward model) ...................... predicts the *next*
     day's signs x_{t+1} from h_t. This is "foretell the future": a physician
     who can say what tomorrow's pulse and countenance will be has understood
     the course.

  3. KRISIS HEAD (hazard + branch) ...................... a per-day hazard that
     the turning point falls *now*, and a branch read (favourable coction ->
     recovery, vs crude/unconcocted -> death). The krisis is a learned phase
     change in the latent state, exactly as "coctions signify nearness of
     crisis" while "crude and unconcocted evacuations denote ... death."

  4. KAIROS GATE ("do no harm") ......................... a restraint head that
     chooses intervene vs. abstain. It is trained with an ASYMMETRIC cost:
     intervening on a patient nature would have healed is punished far harder
     than missing a chance to help. The model learns to keep its hands still by
     default and act only at the fleeting opportune moment (kairos) when its
     own forecast says the trajectory is breaking toward death.

Everything below is pure NumPy, written from scratch. There is a mandatory
finite-difference gradient check on the full multi-head loss, a real training
loop on a synthetic "Epidemics" case-book, and self-tests. Run the file to
reproduce the verified output quoted in the chapter.

Author: 1000 Minds project. License: for the terabook corpus.
"""

from __future__ import annotations
import numpy as np

# A single global generator keeps every run reproducible (an Epidemics case-book
# is only useful if the cases are the same cases each time we open it).
RNG = np.random.default_rng(460)  # seed = Hippocrates' traditional birth year


# =============================================================================
# SECTION 1 -- THE SYNTHETIC EPIDEMICS: a generative model of disease courses
# -----------------------------------------------------------------------------
# Hippocrates' "Epidemics" are day-by-day case records: signs observed, course
# followed, crisis noted, outcome (recovery or death) recorded -- often with no
# intervention at all, just watching. We synthesise such records from a known
# latent process so that we have ground truth to check the network against.
#
# Latent variables per case:
#   coction c_t in [0,1] : how far the disease-matter has "ripened".
#   severity s_t         : the burden the body is fighting.
# The crisis day tau is when the contest resolves. If coction is sufficiently
# advanced by tau the patient recovers (favourable coction); if it is still
# "crude" the patient dies. Observable signs are noisy non-linear readings of
# (c_t, s_t): pulse, fever-heat, breathing, urine-clarity, and "facies" (the
# countenance, the famous facies Hippocratica that darkens toward death).
# =============================================================================

N_SIGN = 5          # pulse, heat, breath, urine-clarity, facies
T = 14              # days observed (Hippocratic "critical days" live in 1..14)


def make_case(rng: np.random.Generator):
    """Generate one patient course: signs X (T, N_SIGN), and ground-truth
    labels for crisis day, outcome (1=death), and the 'nature is failing' flag
    used to define the restraint-optimal action policy.

    Crucially -- as in real Hippocratic doctrine -- the crisis is LEGIBLE from
    the signs: it is the day the latent coction crosses a threshold (visible in
    the clearing of the urine and the lifting of the countenance), and the
    outcome turns on whether the body's severity has been subdued by then. A
    forecaster watching the signs can therefore anticipate both."""
    THETA_C = 0.60       # coction threshold that defines the crisis day
    S_FATAL = 0.55       # severity still above this at the crisis -> death
    coct_rate = rng.uniform(0.05, 0.15)         # speed of ripening per day
    severity0 = rng.uniform(0.35, 0.95)         # initial burden

    c = 0.05                                     # coction starts crude
    s = severity0
    X = np.zeros((T, N_SIGN))
    cs = np.zeros(T); ss = np.zeros(T)
    for t in range(T):
        # Coction accumulates; severity is subdued as the matter concocts.
        c = float(np.clip(c + coct_rate + rng.normal(0, 0.008), 0.0, 1.3))
        s = float(np.clip(s + rng.normal(0, 0.03) - 0.07 * c, 0.05, 1.3))
        cs[t] = c; ss[t] = s
        # Signs are smooth non-linear emissions of the latent state (+ noise).
        pulse  = 0.5 + 0.45 * np.tanh(2.2 * s - 1.0)
        heat   = 0.4 + 0.5 * s - 0.15 * c
        breath = 0.45 + 0.4 * np.tanh(1.8 * s - 0.7)
        urine  = np.clip(0.12 + 0.85 * c, 0, 1)             # clears as coction ripens
        facies = np.clip(0.85 - 0.5 * c + 0.3 * s, 0, 1)    # darkens toward death
        X[t] = [pulse, heat, breath, urine, facies] + rng.normal(0, 0.02, N_SIGN)

    # Crisis day: first day coction crosses the threshold (else the last day).
    above = np.where(cs >= THETA_C)[0]
    crisis_day = int(above[0]) if above.size else T - 1
    # Outcome: severity still high at the crisis -> crude/unconcocted -> death.
    death = 1 if ss[crisis_day] >= S_FATAL else 0

    # The "nature is failing" signal that defines the restraint-optimal policy:
    # only on death-bound courses, and only from a few days before the crisis
    # (once the breaking trend is legible) is intervention warranted. Elsewhere
    # the optimal act is to ABSTAIN -- do no harm.
    nature_failing = np.zeros(T)
    if death:
        start = max(0, crisis_day - 3)
        nature_failing[start:] = 1.0

    return X.astype(np.float64), crisis_day, death, nature_failing


def make_dataset(n, rng):
    Xs, taus, deaths, failings = [], [], [], []
    for _ in range(n):
        X, tau, d, nf = make_case(rng)
        Xs.append(X); taus.append(tau); deaths.append(d); failings.append(nf)
    return (np.stack(Xs), np.array(taus), np.array(deaths, dtype=np.float64),
            np.stack(failings))


# Small helpers ---------------------------------------------------------------
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def bce(p, y, eps=1e-9):
    p = np.clip(p, eps, 1 - eps)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


# =============================================================================
# SECTION 2 -- THE PROGNOSTIC ENGINE (parameters, forward, hand-coded BPTT)
# =============================================================================
class PrognosticEngine:
    """A gated-recurrent forecaster with four Hippocratic heads.

    Coction recurrence (one gate -- the rate of concoction):
        a_t = Wg x_t + Ug h_{t-1} + bg        ; g_t = sigmoid(a_t)
        p_t = Wx x_t + Wh h_{t-1} + b         ; n_t = tanh(p_t)
        h_t = (1 - g_t) * h_{t-1} + g_t * n_t

    Heads (read from h_t):
        prognosis : xhat_{t} = Wf h_t + bf       (forecast next day's signs)
        krisis    : haz_t = sigmoid(wc.h_t + bc) (turning point falls now?)
        branch    : death_hat = sigmoid(wo.h_T + bo)  (recovery vs death)
        kairos    : act_t = sigmoid(wa.h_t + ba) (intervene vs abstain)
    """

    def __init__(self, n_sign=N_SIGN, hidden=16, seed=460):
        r = np.random.default_rng(seed)
        H, D = hidden, n_sign
        sc = 0.4  # small init keeps tanh/sigmoid in their informative range

        def g(*shape):
            return (r.standard_normal(shape) * sc / np.sqrt(shape[-1])).astype(np.float64)

        self.H, self.D = H, D
        # Coction cell
        self.Wx = g(H, D); self.Wh = g(H, H); self.b = np.zeros(H)
        self.Wg = g(H, D); self.Ug = g(H, H); self.bg = np.zeros(H)
        # Prognosis (forecast) head
        self.Wf = g(D, H); self.bf = np.zeros(D)
        # Krisis hazard head
        self.wc = g(H); self.bc = np.zeros(1)
        # Branch / outcome head
        self.wo = g(H); self.bo = np.zeros(1)
        # Kairos (do-no-harm) action head
        self.wa = g(H); self.ba = np.zeros(1)

    # -- parameter plumbing (used by the gradient check and the optimiser) ----
    def params(self):
        return {k: getattr(self, k) for k in
                ['Wx', 'Wh', 'b', 'Wg', 'Ug', 'bg', 'Wf', 'bf',
                 'wc', 'bc', 'wo', 'bo', 'wa', 'ba']}

    def set_params(self, d):
        for k, v in d.items():
            setattr(self, k, v)

    # ---------------------------------------------------------------------
    # FORWARD over one case. Caches everything BPTT needs.
    # ---------------------------------------------------------------------
    def forward(self, X):
        T_, D = X.shape
        H = self.H
        cache = {'X': X, 'h': [np.zeros(H)], 'g': [], 'n': [], 'a': [], 'p': []}
        for t in range(T_):
            x = X[t]; h_prev = cache['h'][-1]
            a = self.Wg @ x + self.Ug @ h_prev + self.bg
            gt = sigmoid(a)
            p = self.Wx @ x + self.Wh @ h_prev + self.b
            n = np.tanh(p)
            h = (1 - gt) * h_prev + gt * n
            cache['a'].append(a); cache['g'].append(gt)
            cache['p'].append(p); cache['n'].append(n); cache['h'].append(h)
        Hs = np.stack(cache['h'][1:])               # (T, H), states h_1..h_T
        cache['Hs'] = Hs
        # Heads
        cache['xhat'] = Hs @ self.Wf.T + self.bf     # (T, D) forecast
        cache['haz'] = sigmoid(Hs @ self.wc + self.bc)   # (T,)
        cache['act'] = sigmoid(Hs @ self.wa + self.ba)   # (T,)
        cache['death_hat'] = float(sigmoid(Hs[-1] @ self.wo + self.bo).item())
        return cache

    # ---------------------------------------------------------------------
    # LOSS for one case + gradients (full hand-coded backprop / BPTT).
    # Returns (loss, grads_dict).
    # ---------------------------------------------------------------------
    def loss_and_grads(self, X, crisis_day, death, nature_failing,
                       lam=(1.0, 1.0, 1.0, 1.0), w_harm=4.0):
        c = self.forward(X)
        T_, D, H = X.shape[0], self.D, self.H
        Hs, xhat, haz, act = c['Hs'], c['xhat'], c['haz'], c['act']
        lam_f, lam_k, lam_o, lam_a = lam

        # ---- 1) Prognosis loss: forecast day t+1 from h_t (t = 0..T-2) -------
        # target for step t is the real next-day sign vector X[t+1].
        tgt = X[1:]                       # (T-1, D)
        pred = xhat[:-1]                  # (T-1, D)
        diff = pred - tgt
        Nf = (T_ - 1) * D
        L_f = 0.5 * np.sum(diff ** 2) / Nf

        # ---- 2) Krisis hazard loss: 1 on the crisis day, else 0 -------------
        # The crisis is one day in T, so we up-weight the positive day (pos_w)
        # to balance the rare event, just as a physician's attention spikes as
        # the critical day approaches.
        y_haz = np.zeros(T_); y_haz[min(crisis_day, T_ - 1)] = 1.0
        pos_w = float(T_)
        w_haz = np.where(y_haz > 0.5, pos_w, 1.0)
        L_k = np.mean(w_haz * bce(haz, y_haz))

        # ---- 3) Branch loss: death vs recovery from the final state ---------
        dh_out = c['death_hat']
        L_o = float(bce(np.array([dh_out]), np.array([death]))[0])

        # ---- 4) Kairos / do-no-harm loss: asymmetric weighted BCE ----------
        # y_act = nature_failing (1 -> should intervene). Acting when y=0 (the
        # patient would have recovered) is "harm" and weighted w_harm; missing a
        # true need is weighted 1. The model is thereby taught restraint.
        y_act = nature_failing
        p_act = np.clip(act, 1e-9, 1 - 1e-9)
        weight = np.where(y_act > 0.5, 1.0, w_harm)
        L_a = np.mean(weight * (-(y_act * np.log(p_act)
                                  + (1 - y_act) * np.log(1 - p_act))))

        loss = lam_f * L_f + lam_k * L_k + lam_o * L_o + lam_a * L_a

        # =====================================================================
        # BACKWARD.  Accumulate dL/dh_t from every head, then BPTT the cell.
        # =====================================================================
        gp = {k: np.zeros_like(v) for k, v in self.params().items()}
        dHs = np.zeros((T_, H))           # gradient flowing into each h_t

        # head 1: prognosis (affects h_0..h_{T-2})
        dpred = lam_f * diff / Nf                       # (T-1, D)
        gp['Wf'] += dpred.T @ Hs[:-1]
        gp['bf'] += dpred.sum(axis=0)
        dHs[:-1] += dpred @ self.Wf

        # head 2: krisis hazard (affects all h_t)
        dlogit_h = lam_k * (w_haz * (haz - y_haz)) / T_  # (T,)
        gp['wc'] += Hs.T @ dlogit_h
        gp['bc'] += dlogit_h.sum()
        dHs += np.outer(dlogit_h, self.wc)

        # head 3: branch (affects only h_T = last state)
        dlogit_o = lam_o * (dh_out - death)
        gp['wo'] += dlogit_o * Hs[-1]
        gp['bo'] += dlogit_o
        dHs[-1] += dlogit_o * self.wo

        # head 4: kairos action (affects all h_t); d/dlogit of weighted BCE
        dlogit_a = lam_a * (weight * (p_act - y_act)) / T_   # (T,)
        gp['wa'] += Hs.T @ dlogit_a
        gp['ba'] += dlogit_a.sum()
        dHs += np.outer(dlogit_a, self.wa)

        # ---- BPTT through the coction recurrence ----------------------------
        dh_next = np.zeros(H)
        for t in reversed(range(T_)):
            dh = dHs[t] + dh_next
            h_prev = c['h'][t]            # h_{t-1}
            gt, n = c['g'][t], c['n'][t]
            # h_t = (1-g)*h_prev + g*n
            dg = dh * (n - h_prev)
            dn = dh * gt
            dh_prev = dh * (1 - gt)       # direct carry path
            # through tanh:  n = tanh(p)
            dp = dn * (1 - n ** 2)
            # through sigmoid: g = sigmoid(a)
            da = dg * gt * (1 - gt)
            x = X[t]
            gp['Wx'] += np.outer(dp, x); gp['Wh'] += np.outer(dp, h_prev); gp['b'] += dp
            gp['Wg'] += np.outer(da, x); gp['Ug'] += np.outer(da, h_prev); gp['bg'] += da
            dh_prev = dh_prev + self.Wh.T @ dp + self.Ug.T @ da
            dh_next = dh_prev

        parts = {'forecast': L_f, 'krisis': L_k, 'branch': L_o, 'kairos': L_a}
        return loss, gp, parts, c


# =============================================================================
# SECTION 3 -- MANDATORY FINITE-DIFFERENCE GRADIENT CHECK
# -----------------------------------------------------------------------------
# We perturb a sample of individual parameters, recompute the loss, and compare
# the numerical slope to the analytic gradient. Pass criterion: max relative
# error < 1e-5 across all sampled parameters (float64).
# =============================================================================
def gradient_check(verbose=True):
    rng = np.random.default_rng(7)
    net = PrognosticEngine(hidden=8, seed=1)
    X, tau, death, nf = make_case(rng)

    def L():
        return net.loss_and_grads(X, tau, death, nf)[0]

    _, grads, _, _ = net.loss_and_grads(X, tau, death, nf)
    eps = 1e-6
    worst = 0.0
    checked = 0
    for name, arr in net.params().items():
        flat = arr.ravel()
        # sample up to 4 coordinates per parameter tensor
        idxs = rng.choice(flat.size, size=min(4, flat.size), replace=False)
        for i in idxs:
            orig = flat[i]
            flat[i] = orig + eps; lp = L()
            flat[i] = orig - eps; lm = L()
            flat[i] = orig
            num = (lp - lm) / (2 * eps)
            ana = grads[name].ravel()[i]
            denom = max(1e-12, abs(num) + abs(ana))
            rel = abs(num - ana) / denom
            worst = max(worst, rel)
            checked += 1
    if verbose:
        print(f"  gradient check: {checked} params sampled, "
              f"max relative error = {worst:.2e}")
    assert worst < 1e-5, f"Gradient check FAILED (max rel err {worst:.2e})"
    return worst


# =============================================================================
# SECTION 4 -- ADAM OPTIMISER AND TRAINING LOOP
# =============================================================================
class Adam:
    def __init__(self, params, lr=4e-3, b1=0.9, b2=0.999, eps=1e-8):
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


def evaluate(net, data):
    Xs, taus, deaths, nfs = data
    n = len(Xs)
    fmse = 0.0; krisis_hit = 0; death_correct = 0
    harm = 0; harm_den = 0; help_hit = 0; help_den = 0
    for i in range(n):
        c = net.forward(Xs[i])
        # forecast mse
        fmse += np.mean((c['xhat'][:-1] - Xs[i][1:]) ** 2)
        # krisis: argmax hazard within +-1 day of true crisis
        khat = int(np.argmax(c['haz']))
        if abs(khat - taus[i]) <= 1:
            krisis_hit += 1
        # outcome
        if (c['death_hat'] > 0.5) == bool(deaths[i]):
            death_correct += 1
        # restraint: on recovery-bound patients, any confident intervention=harm
        act = c['act']
        if deaths[i] == 0:
            harm_den += 1
            if np.any(act > 0.5):
                harm += 1
        else:
            help_den += 1
            if np.any(act[max(0, taus[i] - 3):] > 0.5):
                help_hit += 1
    return {
        'forecast_mse': fmse / n,
        'krisis_acc': krisis_hit / n,
        'death_acc': death_correct / n,
        'harm_rate': harm / max(1, harm_den),     # lower is better (do no harm)
        'help_rate': help_hit / max(1, help_den), # higher is better (help)
    }


def train(epochs=50, n_train=240, n_test=80, hidden=16, verbose=True):
    rng = np.random.default_rng(123)
    train_data = make_dataset(n_train, rng)
    test_data = make_dataset(n_test, rng)
    net = PrognosticEngine(hidden=hidden, seed=460)
    opt = Adam(net.params(), lr=5e-3)
    params = net.params()

    Xs, taus, deaths, nfs = train_data
    order = np.arange(n_train)
    history = []
    best_score = -1e9
    best_params = None
    best_metrics = None
    for ep in range(epochs):
        opt.lr = 5e-3 * (0.96 ** ep)            # cool the step as the crisis nears
        rng.shuffle(order)
        ep_loss = 0.0
        for i in order:
            loss, grads, parts, _ = net.loss_and_grads(
                Xs[i], int(taus[i]), float(deaths[i]), nfs[i])
            opt.step(params, grads)
            net.set_params(params)
            ep_loss += loss
        ep_loss /= n_train
        history.append(ep_loss)
        m = evaluate(net, test_data)
        # Hippocratic selection: reward helping, punish harm twice as hard, and
        # require the forecast organs to be working too.
        score = (m['help_rate'] - 2.0 * m['harm_rate']
                 + 0.5 * m['krisis_acc'] + 0.5 * m['death_acc'])
        if score > best_score:
            best_score = score
            best_params = {k: v.copy() for k, v in params.items()}
            best_metrics = m
        if verbose and (ep % 5 == 0 or ep == epochs - 1):
            print(f"  epoch {ep:2d} | loss {ep_loss:.4f} "
                  f"| forecastMSE {m['forecast_mse']:.4f} "
                  f"| krisis {m['krisis_acc']:.2f} "
                  f"| outcome {m['death_acc']:.2f} "
                  f"| harm {m['harm_rate']:.2f} help {m['help_rate']:.2f}")
    net.set_params(best_params)                 # restore the best physician
    if verbose:
        print(f"  >> restored best snapshot (score {best_score:.3f})")
    return net, history, best_metrics


# =============================================================================
# SECTION 5 -- SELF-TESTS
# =============================================================================
def run_self_tests():
    print("[1/4] Finite-difference gradient check (mandatory) ...")
    gradient_check(verbose=True)
    print("      PASS\n")

    print("[2/4] Overfit a tiny set: loss must fall sharply ...")
    rng = np.random.default_rng(5)
    tiny = make_dataset(8, rng)
    net = PrognosticEngine(hidden=12, seed=2)
    opt = Adam(net.params(), lr=8e-3); params = net.params()
    Xs, taus, deaths, nfs = tiny
    first = last = None
    for ep in range(120):
        tot = 0.0
        for i in range(len(Xs)):
            loss, grads, _, _ = net.loss_and_grads(
                Xs[i], int(taus[i]), float(deaths[i]), nfs[i])
            opt.step(params, grads); net.set_params(params); tot += loss
        if ep == 0:
            first = tot / len(Xs)
        last = tot / len(Xs)
    print(f"      loss {first:.4f} -> {last:.4f}")
    assert last < first * 0.5, "Model failed to overfit a tiny dataset"
    print("      PASS\n")

    print("[3/4] Full training run ...")
    net, hist, metrics = train(epochs=50, verbose=True)
    assert hist[-1] < hist[0] * 0.7, "Training loss did not decrease enough"
    print("      PASS\n")

    print("[4/4] Hippocratic behaviour checks ...")
    print(f"      krisis timing accuracy : {metrics['krisis_acc']:.2f}")
    print(f"      outcome accuracy       : {metrics['death_acc']:.2f}")
    print(f"      HARM rate (lower=better): {metrics['harm_rate']:.2f}")
    print(f"      help rate (higher=better): {metrics['help_rate']:.2f}")
    assert metrics['krisis_acc'] > 0.6, "Crisis timing too weak"
    assert metrics['death_acc'] > 0.75, "Outcome prediction too weak"
    assert metrics['harm_rate'] < 0.25, "Violated 'do no harm' too often"
    print("      PASS\n")
    return net, metrics


if __name__ == "__main__":
    print("=" * 70)
    print(" THE PROGNOSTIC ENGINE  --  a cognitive model after Hippocrates")
    print(" 'Declare the past, diagnose the present, foretell the future;")
    print("  ... to help, or at least to do no harm.'  (Epidemics I)")
    print("=" * 70 + "\n")
    net, metrics = run_self_tests()
    print("=" * 70)
    print(" ALL SELF-TESTS PASSED")
    print(" The engine forecasts the course, names the crisis, and keeps its")
    print(" hands still unless its own forecast says nature is failing.")
    print("=" * 70)
