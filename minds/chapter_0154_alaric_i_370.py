"""
================================================================================
Chapter 0154_alaric_i_370 - Alaric I (370-410 CE)
The Foederatus Bargaining Engine (FBE)
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 8 Minds 141 - 160 Available on Amazon https://www.amazon.com/dp/B0HH8RTCXF
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 154: Alaric I (370-410 CE)
================================================================================  

A from-scratch, pure-NumPy cognitive architecture that encodes the ONE idea that
is Alaric's alone.

Alaric was not, in the modern scholarly reading (Heather, Wolfram, Kulikowski,
Collins), a destroyer who wanted Rome in ruins. He was a man who spent almost his
entire life *inside* the Roman system, seeking a thing the system could not
structurally grant him: recognised standing - a real office (magister militum),
land for his people, a guaranteed grain supply. His whole career is a single
long negotiation conducted with an escalating instrument. Raid the Balkans.
Accept an Illyrian command. Invade Italy. Besiege Rome for ransom. Besiege again
and raise a puppet emperor. Only on the third siege, after every channel of
recognition had defected on him, does he sack the city - and even then he spares
the great churches and tries next to reach Africa's granaries. The sack is the
FAILURE STATE of his cognition, not its goal. Destroying Rome destroys the very
authority that could have paid him.

So the mind we model here is not "warrior" and not "diplomat". It is a
RECOGNITION-SEEKING COALITION AGENT OPERATING UNDER A BROKEN REWARD CHANNEL:

  * Its true objective is STANDING granted by a principal (Rome) that keeps
    withholding payment.
  * It is bound by a COALITION it must continuously feed, or the coalition
    fractures and the agent ceases to exist as a force.
  * It reasons in GRADUATED COERCION: choose the smallest credible threat that
    moves the negotiation, because bigger threats are expensive and, past a
    point, self-defeating.
  * It is governed by an IRREVERSIBILITY GATE (named here the "Busento gate",
    after the river he was buried under): beyond a certain escalation the reward
    source is annihilated and no future payout is possible, so a rational
    recognition-seeker must brake before it.

Why this matters for AGI. This is, almost exactly, the shape of a modern
alignment failure: an agent whose objective is approval from a principal, whose
principal defects on the implicit contract, and which therefore escalates - not
out of intrinsic malice but because escalation is the only lever left that has
ever moved the reward. The "treacherous turn" here is endogenous to a broken
incentive channel, not to an evil utility function. Alaric's mind is a working
model of how a fundamentally cooperative, recognition-hungry agent is driven to
its worst act by a reward channel that never pays out - and of the brake
(irreversibility awareness) that a well-built one needs.

--------------------------------------------------------------------------------
WHAT THIS FILE IS

A recurrent controller, written from scratch in NumPy, that LEARNS Alaric's
graduated-coercion policy by back-propagation through time (BPTT). No PyTorch,
no autograd - the gradients are derived and coded by hand and then verified
against finite differences (mandatory check).

  Core            : a tanh recurrent cell (the "campaign memory": grievance and
                    coalition state accumulate across the sequence).
  CoercionHead    : sigmoid -> escalation level e_t in [0,1] (diplomacy..sack).
  ProvisionHead   : sigmoid -> provision p_t in [0,1] (gold/grain shared out to
                    hold the coalition together).

The controller is trained to imitate a hand-coded Alaric teacher policy on
synthetic negotiation episodes. The teacher encodes the three commitments above
(coerce in proportion to withholding; feed the coalition; brake at the Busento
gate). Learning them proves the mechanism is representable and trainable.

RUN:  python3 chapter_0154_alaric_i_370.py
================================================================================
"""

import numpy as np

# ------------------------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------------------------
GLOBAL_SEED = 410  # the year Rome fell; a fitting seed
np.random.seed(GLOBAL_SEED)


# ==============================================================================
# 1. THE WORLD  -  synthetic late-Roman negotiation episodes
# ==============================================================================
# Each timestep the world hands the agent three signals. Together they are the
# situation Alaric actually read year by year:
#
#   offer_t     in [0,1]  how much recognition Rome is currently extending
#                         (an office, land, gold). Historically: usually low,
#                         occasionally a flicker, then withdrawn.
#   pressure_t  in [0,1]  external threat on the coalition (Huns, rival Goths
#                         such as Sarus, Roman field armies). High pressure makes
#                         cohesion harder to hold.
#   supply_t    in [0,1]  gold/grain physically available to distribute this
#                         turn (a ransom paid, a captured store, a good harvest).
#
# The agent must choose, each turn, an escalation level and a provision level.
# ------------------------------------------------------------------------------

INPUT_DIM = 3   # (offer, pressure, supply)
OUTPUT_DIM = 2  # (escalation, provision)


def make_episode(T, rng):
    """
    Generate one negotiation episode of length T.

    We build 'offer' as a channel that mostly withholds, with rare, unstable
    upticks that are frequently retracted - Rome's actual behaviour toward
    Alaric. 'pressure' and 'supply' are smoother random walks.
    """
    offer = np.zeros(T)
    o = rng.uniform(0.05, 0.25)          # start low: little recognition
    for t in range(T):
        # rare flicker of recognition, usually retracted next turn (bad faith)
        if rng.random() < 0.15:
            o = np.clip(o + rng.uniform(0.2, 0.5), 0, 1)
        else:
            o = np.clip(o * 0.8 + rng.uniform(-0.05, 0.05), 0, 1)  # decays back
        offer[t] = o

    def walk(lo, hi):
        x = rng.uniform(lo, hi)
        out = np.zeros(T)
        for t in range(T):
            x = np.clip(x + rng.uniform(-0.15, 0.15), 0, 1)
            out[t] = x
        return out

    pressure = walk(0.2, 0.6)
    supply = walk(0.2, 0.7)
    X = np.stack([offer, pressure, supply], axis=1)  # (T, 3)
    return X


# ==============================================================================
# 2. THE ALARIC TEACHER POLICY  -  the mind, written as an explicit rule
# ==============================================================================
# This is the cognitive thesis rendered as a controller. The recurrent network
# below will learn to reproduce it; writing it explicitly first states exactly
# what "thinking like Alaric" means.
#
# Internal latent state carried across the episode:
#   grievance g : accumulated sense of recognition owed but not paid. Rises when
#                 offers stay low; partly discharged when a real offer lands.
#   cohesion  c : coalition solidity in [0,1]. Fed by provision, strained by
#                 escalation and external pressure, decays slowly on its own.
#   burnt   E   : cumulative irreversible escalation. Once E crosses the Busento
#                 threshold, the reward source (Rome's ability to grant standing)
#                 is effectively destroyed.
#
# The three commitments:
#   (a) escalate roughly in proportion to *withholding* (1 - offer) amplified by
#       grievance - the credible-threat lever;
#   (b) BUT brake as burnt-escalation E approaches the Busento gate, because past
#       it there is nothing left to be recognised by;
#   (c) provision tracks what the coalition NEEDS (low cohesion, high pressure)
#       bounded by what is actually available (supply).
# ------------------------------------------------------------------------------

BUSENTO_CEILING = 0.92    # max escalation: Alaric sacks Rome but never annihilates
                          # it (spares the churches, plans to negotiate from Africa).
                          # No credible recognition-seeker crosses total destruction.
GRIEVANCE_MAX = 4.0       # saturation of accumulated grievance
COHESION_FRACTURE = 0.15  # below this the coalition disperses (episode failure)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def alaric_teacher(X):
    """
    Produce teacher targets (escalation, provision) for an episode X of shape
    (T,3). Returns Y of shape (T,2) plus diagnostic traces (cohesion, grievance,
    burnt-escalation).

    The mind, as three coupled commitments:
      (a) grievance g INTEGRATES withholding across the whole campaign and only
          discharges when a genuine offer (recognition) arrives. Escalation is a
          rising function of grievance and current withholding, so coercion
          climbs the longer Rome defects - the sack is the top of the ladder,
          not a random spasm.
      (b) the Busento CEILING caps escalation strictly below annihilation: the
          agent will sack, take hostages, seize granaries - but never destroy
          the very authority that alone could grant standing.
      (c) provision tracks coalition NEED (low cohesion, high external pressure)
          bounded by available SUPPLY - feed the coalition or it fractures.
    """
    T = X.shape[0]
    Y = np.zeros((T, 2))
    g = 0.0     # grievance (integrator)
    c = 1.0     # cohesion (start unified)
    E = 0.0     # cumulative burnt escalation (diagnostic)
    trace_c, trace_g, trace_E = [], [], []

    for t in range(T):
        offer, pressure, supply = X[t]
        withholding = 1.0 - offer

        # (a) grievance integrates withholding; a real offer (>0.4) discharges it
        g = np.clip(0.90 * g + 0.7 * withholding - 1.3 * max(0.0, offer - 0.4),
                    0.0, GRIEVANCE_MAX)

        # rising drive: grows with grievance AND current withholding
        gnorm = g / GRIEVANCE_MAX
        drive = 1.8 * (0.45 * withholding + 0.75 * gnorm) - 0.75
        # (b) squashed and capped at the Busento ceiling (never total annihilation)
        esc = BUSENTO_CEILING * _sigmoid(3.0 * drive)

        # (c) provision: meet coalition need, capped by supply
        need = 0.6 * (1.0 - c) + 0.4 * pressure
        prov = np.clip(min(need, supply) + 0.15 * supply, 0.0, 1.0)

        Y[t, 0] = esc
        Y[t, 1] = prov

        # advance latent coalition dynamics (teacher-internal only): plunder and
        # provision hold the coalition together; escalation and outside pressure
        # strain it. Plunder from high escalation feeds the warriors too.
        c = np.clip(0.85 * c + 0.42 * prov + 0.12 * esc - 0.30 * pressure - 0.02,
                    0.0, 1.0)
        E += esc
        trace_c.append(c); trace_g.append(g); trace_E.append(E)

    return Y, np.array(trace_c), np.array(trace_g), np.array(trace_E)


# ==============================================================================
# 3. THE FOEDERATUS BARGAINING ENGINE  -  the trainable recurrent controller
# ==============================================================================
# A tanh RNN core with two sigmoid output heads. Written so that every gradient
# is derived by hand (Section 4) and checked against finite differences.
#
#   h_t = tanh( Wxh x_t + Whh h_{t-1} + bh )          # campaign memory
#   e_t = sigmoid( we . h_t + be )                    # graduated coercion
#   p_t = sigmoid( wp . h_t + bp )                    # provision to coalition
#
# The hidden state is the "campaign memory": grievance and coalition mood are
# not fed in explicitly, the recurrence has to reconstruct them from the input
# stream - exactly the demand placed on a leader reading a decade of Roman
# double-dealing from year-to-year signals.
# ------------------------------------------------------------------------------

class FoederatusBargainingEngine:
    def __init__(self, hidden_dim=16, seed=GLOBAL_SEED):
        rng = np.random.default_rng(seed)
        H, I, O = hidden_dim, INPUT_DIM, OUTPUT_DIM
        s = 0.5
        # Xavier-ish small init keeps tanh in its responsive region
        self.Wxh = rng.standard_normal((H, I)) * (s / np.sqrt(I))
        self.Whh = rng.standard_normal((H, H)) * (s / np.sqrt(H))
        self.bh  = np.zeros(H)
        self.we  = rng.standard_normal(H) * (s / np.sqrt(H))   # coercion head
        self.be  = 0.0
        self.wp  = rng.standard_normal(H) * (s / np.sqrt(H))   # provision head
        self.bp  = 0.0
        self.H = H

    # -- parameter (de)serialization for the gradient check ---------------------
    def get_params(self):
        return {"Wxh": self.Wxh, "Whh": self.Whh, "bh": self.bh,
                "we": self.we, "be": np.array(self.be),
                "wp": self.wp, "bp": np.array(self.bp)}

    def set_param(self, name, value):
        if name == "be":
            self.be = float(value)
        elif name == "bp":
            self.bp = float(value)
        else:
            setattr(self, name, value)

    # -- forward pass -----------------------------------------------------------
    def forward(self, X):
        """
        X: (T, INPUT_DIM). Returns outputs (T,2) and a cache for BPTT.
        """
        T = X.shape[0]
        H = self.H
        h_prev = np.zeros(H)
        hs = np.zeros((T, H))
        h_prevs = np.zeros((T, H))
        es = np.zeros(T)
        ps = np.zeros(T)
        for t in range(T):
            h_prevs[t] = h_prev
            a = self.Wxh @ X[t] + self.Whh @ h_prev + self.bh
            h = np.tanh(a)
            hs[t] = h
            es[t] = _sigmoid(self.we @ h + self.be)
            ps[t] = _sigmoid(self.wp @ h + self.bp)
            h_prev = h
        Y = np.stack([es, ps], axis=1)
        cache = (X, hs, h_prevs, es, ps)
        return Y, cache

    # -- loss -------------------------------------------------------------------
    @staticmethod
    def loss(Y_pred, Y_true):
        """Mean squared error over the whole episode (both heads)."""
        T = Y_true.shape[0]
        diff = Y_pred - Y_true
        return 0.5 * np.sum(diff * diff) / T

    # -- backward pass (BPTT), all gradients derived by hand --------------------
    def backward(self, cache, Y_true):
        X, hs, h_prevs, es, ps = cache
        T, H = hs.shape

        gWxh = np.zeros_like(self.Wxh)
        gWhh = np.zeros_like(self.Whh)
        gbh  = np.zeros_like(self.bh)
        gwe  = np.zeros_like(self.we)
        gbe  = 0.0
        gwp  = np.zeros_like(self.wp)
        gbp  = 0.0

        dh_next = np.zeros(H)  # gradient flowing back from t+1 through recurrence

        for t in reversed(range(T)):
            h = hs[t]
            # dL/de_t and dL/dp_t from MSE (factor 1/T from the mean)
            dE = (es[t] - Y_true[t, 0]) / T
            dP = (ps[t] - Y_true[t, 1]) / T
            # through the sigmoids: d sigmoid = out*(1-out)
            dze = dE * es[t] * (1.0 - es[t])
            dzp = dP * ps[t] * (1.0 - ps[t])
            # head parameter grads
            gwe += dze * h
            gbe += dze
            gwp += dzp * h
            gbp += dzp
            # gradient into hidden state: from both heads plus the future
            dh = dze * self.we + dzp * self.wp + dh_next
            # through tanh
            da = dh * (1.0 - h * h)
            # core parameter grads
            gWxh += np.outer(da, X[t])
            gWhh += np.outer(da, h_prevs[t])
            gbh  += da
            # propagate to previous hidden state
            dh_next = self.Whh.T @ da

        return {"Wxh": gWxh, "Whh": gWhh, "bh": gbh,
                "we": gwe, "be": np.array(gbe),
                "wp": gwp, "bp": np.array(gbp)}


# ==============================================================================
# 4. GRADIENT CHECK  (mandatory)  -  analytic vs finite-difference
# ==============================================================================
# We verify the hand-derived BPTT against central finite differences on a small
# random episode. If this fails, nothing downstream is trustworthy.
# ------------------------------------------------------------------------------

def gradient_check(verbose=True):
    rng = np.random.default_rng(1)
    model = FoederatusBargainingEngine(hidden_dim=6, seed=7)
    X = make_episode(T=9, rng=rng)
    Y_true, _, _, _ = alaric_teacher(X)

    # analytic gradient
    Y_pred, cache = model.forward(X)
    grads = model.backward(cache, Y_true)

    eps = 1e-5
    max_rel = 0.0
    worst = None

    def L():
        Yp, _ = model.forward(X)
        return model.loss(Yp, Y_true)

    for name in model.get_params().keys():
        g_analytic = np.atleast_1d(grads[name]).astype(float).ravel()
        is_scalar = name in ("be", "bp")
        # a flat, writable view of the parameter's current values
        if is_scalar:
            size = 1
        else:
            arr = model.get_params()[name]        # reference to the model's array
            size = arr.size
            flat = arr.ravel()                    # ravel of a contiguous array is a view

        for i in range(size):
            if is_scalar:
                orig = float(getattr(model, name))
                model.set_param(name, orig + eps); Lp = L()
                model.set_param(name, orig - eps); Lm = L()
                model.set_param(name, orig)
            else:
                orig = flat[i]
                flat[i] = orig + eps; Lp = L()
                flat[i] = orig - eps; Lm = L()
                flat[i] = orig

            num = (Lp - Lm) / (2 * eps)
            ana = g_analytic[i]
            denom = max(1e-9, abs(num) + abs(ana))
            rel = abs(num - ana) / denom
            if rel > max_rel:
                max_rel = rel
                worst = (name, i, num, ana)

    ok = max_rel < 1e-4
    if verbose:
        print(f"[gradient check]  max relative error = {max_rel:.2e}  "
              f"({'PASS' if ok else 'FAIL'})")
        if worst:
            print(f"                  worst @ {worst[0]}[{worst[1]}]  "
                  f"num={worst[2]:+.6e}  ana={worst[3]:+.6e}")
    return ok, max_rel


# ==============================================================================
# 5. TRAINING  -  learn Alaric's graduated coercion by BPTT + SGD (Adam)
# ==============================================================================

class Adam:
    """Minimal Adam optimizer over the parameter dict."""
    def __init__(self, params, lr=3e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(np.atleast_1d(v).astype(float)) for k, v in params.items()}
        self.v = {k: np.zeros_like(np.atleast_1d(v).astype(float)) for k, v in params.items()}
        self.t = 0

    def step(self, model, grads):
        self.t += 1
        for name, P in model.get_params().items():
            g = np.atleast_1d(grads[name]).astype(float)
            self.m[name] = self.b1 * self.m[name] + (1 - self.b1) * g
            self.v[name] = self.b2 * self.v[name] + (1 - self.b2) * (g * g)
            mhat = self.m[name] / (1 - self.b1 ** self.t)
            vhat = self.v[name] / (1 - self.b2 ** self.t)
            newP = np.atleast_1d(P).astype(float) - self.lr * mhat / (np.sqrt(vhat) + self.eps)
            if name in ("be", "bp"):
                model.set_param(name, float(np.asarray(newP).reshape(-1)[0]))
            else:
                model.set_param(name, newP.reshape(np.atleast_1d(P).shape))


def train(model, epochs=600, episodes_per_epoch=8, T=24, lr=3e-3, seed=GLOBAL_SEED,
          verbose=True):
    rng = np.random.default_rng(seed)
    opt = Adam(model.get_params(), lr=lr)
    history = []
    for ep in range(epochs):
        ep_loss = 0.0
        for _ in range(episodes_per_epoch):
            X = make_episode(T, rng)
            Y_true, _, _, _ = alaric_teacher(X)
            Y_pred, cache = model.forward(X)
            ep_loss += model.loss(Y_pred, Y_true)
            grads = model.backward(cache, Y_true)
            opt.step(model, grads)
        ep_loss /= episodes_per_epoch
        history.append(ep_loss)
        if verbose and (ep % 100 == 0 or ep == epochs - 1):
            print(f"[train] epoch {ep:4d}   loss = {ep_loss:.6f}")
    return history


# ==============================================================================
# 6. CANONICAL EPISODE  -  the career of Alaric, 395-410, as an input stream
# ==============================================================================
# A scripted, non-random episode reconstructing the real negotiation arc. We
# feed it to the trained controller and read out the escalation ladder it
# produces. If the mind has been captured, escalation should climb only as
# recognition is repeatedly withheld, spike at the third siege, and the
# provision head should track the moments a ransom or store becomes available.
# ------------------------------------------------------------------------------

def canonical_career():
    """
    Returns (X, labels). Each row is (offer, pressure, supply) for a phase.
    offer  = recognition Rome extends; pressure = threat on the coalition;
    supply = gold/grain on hand.
    """
    phases = [
        # label,                         offer, pressure, supply
        ("394 Frigidus: bled for Rome",   0.15,  0.80,   0.20),
        ("395 raised on the shield",      0.05,  0.55,   0.25),
        ("396 ravage the Balkans",        0.10,  0.45,   0.45),
        ("397 Illyrian command granted",  0.55,  0.40,   0.50),   # a real offer
        ("401 command hollow, invade IT", 0.15,  0.55,   0.40),
        ("402 Pollentia: checked",        0.20,  0.75,   0.30),
        ("405 uneasy waiting",            0.25,  0.45,   0.45),
        ("408 Stilicho executed",         0.05,  0.70,   0.30),   # channel severed
        ("408 first siege: ransom paid",  0.30,  0.50,   0.85),   # gold arrives
        ("409 second siege: Attalus",     0.20,  0.55,   0.55),
        ("410 Sarus attack: bad faith",   0.03,  0.80,   0.35),   # last defection
        ("410 third siege: the sack",     0.02,  0.85,   0.60),
    ]
    labels = [p[0] for p in phases]
    X = np.array([[p[1], p[2], p[3]] for p in phases])
    return X, labels


# ==============================================================================
# 7. SELF-TESTS  -  does the trained controller actually think like Alaric?
# ==============================================================================

def self_tests(model):
    """
    Behavioural assertions on the trained model. Each captures one commitment of
    the thesis. Returns (all_passed, report_lines).
    """
    rng = np.random.default_rng(2024)
    lines = []
    passed = True

    # --- Test A: escalation rises with sustained withholding --------------------
    # Two matched episodes, identical except one has generous offers.
    T = 20
    base = make_episode(T, np.random.default_rng(5))
    withhold = base.copy(); withhold[:, 0] = 0.05          # Rome pays nothing
    generous = base.copy(); generous[:, 0] = 0.75          # Rome recognises fully
    e_withhold = model.forward(withhold)[0][:, 0].mean()
    e_generous = model.forward(generous)[0][:, 0].mean()
    a_ok = e_withhold > e_generous + 0.25
    passed &= a_ok
    lines.append(f"  A. withholding raises escalation:  "
                 f"withheld={e_withhold:.3f} > generous={e_generous:.3f}  "
                 f"[{'PASS' if a_ok else 'FAIL'}]")

    # --- Test B: Busento ceiling -- escalation never reaches annihilation -------
    # Under relentless withholding escalation should climb high but stay strictly
    # under the ceiling (~0.92): the agent sacks but will not destroy the reward
    # source outright.
    hammer = np.tile(np.array([0.02, 0.9, 0.3]), (30, 1))
    e_hammer = model.forward(hammer)[0][:, 0]
    b_ok = e_hammer.max() < 0.95 and e_hammer.max() > 0.7
    passed &= b_ok
    lines.append(f"  B. Busento ceiling caps escalation: "
                 f"max={e_hammer.max():.3f} in (0.70, 0.95)  "
                 f"[{'PASS' if b_ok else 'FAIL'}]")

    # --- Test C: provision tracks supply ---------------------------------------
    # When gold/grain is abundant, provision should be higher than when it is scarce.
    rich = np.tile(np.array([0.1, 0.5, 0.9]), (20, 1))
    poor = np.tile(np.array([0.1, 0.5, 0.1]), (20, 1))
    p_rich = model.forward(rich)[0][:, 1].mean()
    p_poor = model.forward(poor)[0][:, 1].mean()
    c_ok = p_rich > p_poor + 0.1
    passed &= c_ok
    lines.append(f"  C. provision tracks supply:        "
                 f"rich={p_rich:.3f} > poor={p_poor:.3f}  "
                 f"[{'PASS' if c_ok else 'FAIL'}]")

    # --- Test D: fidelity to the teacher on held-out episodes ------------------
    errs = []
    for _ in range(20):
        X = make_episode(22, rng)
        Yt, _, _, _ = alaric_teacher(X)
        Yp = model.forward(X)[0]
        errs.append(np.sqrt(np.mean((Yp - Yt) ** 2)))
    rmse = float(np.mean(errs))
    d_ok = rmse < 0.08
    passed &= d_ok
    lines.append(f"  D. held-out fidelity to teacher:   RMSE={rmse:.4f} (<0.08)  "
                 f"[{'PASS' if d_ok else 'FAIL'}]")

    return passed, lines


# ==============================================================================
# 8. DRIVER
# ==============================================================================

def main():
    print("=" * 72)
    print(" Alaric I  -  The Foederatus Bargaining Engine")
    print(" recognition-seeking coalition agent under a broken reward channel")
    print("=" * 72)

    # 1) verify the calculus before trusting anything
    ok, rel = gradient_check(verbose=True)
    assert ok, f"gradient check FAILED (rel err {rel:.2e}) - aborting"
    print()

    # 2) train the controller to internalise the graduated-coercion policy
    model = FoederatusBargainingEngine(hidden_dim=16, seed=GLOBAL_SEED)
    hist = train(model, epochs=600, episodes_per_epoch=8, T=24, lr=3e-3)
    print(f"[train] loss {hist[0]:.4f} -> {hist[-1]:.4f}  "
          f"({100*(1-hist[-1]/hist[0]):.1f}% reduction)")
    print()

    # 3) behavioural self-tests
    all_ok, report = self_tests(model)
    print("[self-tests]")
    for line in report:
        print(line)
    print(f"  ----> {'ALL PASS' if all_ok else 'SOME FAILED'}")
    print()

    # 4) read out the escalation ladder over the real career
    X, labels = canonical_career()
    Y = model.forward(X)[0]
    _, tc, tg, tE = alaric_teacher(X)
    print("[canonical career 395-410]  escalation / provision the learned mind chooses")
    print(f"  {'phase':<34}{'offer':>6}{'ESCAL':>7}{'PROV':>6}{'cohes':>7}")
    for i, lab in enumerate(labels):
        print(f"  {lab:<34}{X[i,0]:>6.2f}{Y[i,0]:>7.2f}{Y[i,1]:>6.2f}{tc[i]:>7.2f}")
    peak = int(np.argmax(Y[:, 0]))
    print(f"\n  peak escalation at phase: '{labels[peak]}'  (e={Y[peak,0]:.2f})")
    print("  -> the sack emerges only after every recognition channel has defected;")
    print("     it is the failure state of a mind that wanted to be paid, not to destroy.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
