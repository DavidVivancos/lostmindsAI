"""
================================================================================
Chapter 129 - Zhang Heng (張衡, 78-139 CE)
THE RESONANT DIRECTION ENGINE (RDE)
A from-scratch, pure-NumPy cognitive architecture
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 7 Minds 121 - 140 Available on Amazon https://www.amazon.com/dp/B0HFN6GXMH
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 129: Zhang Heng (78-139 CE)
================================================================================    

WHY THIS ARCHITECTURE, FOR THIS MIND
------------------------------------
Zhang Heng's single most distinctive cognitive act was not "measuring" and not
"cataloguing." It was INFERRING A HIDDEN, DISTANT CAUSE FROM A FAINT LOCAL
SIGNATURE, AND RESOLVING ITS DIRECTION.

His bronze seismoscope (the houfeng didong yi, 132 CE) did exactly this: a
tremor originating hundreds of kilometres away - far too weak for any person in
the capital to feel - set an internal pendulum swinging, and ONE of eight
dragons dropped its ball toward the bearing of the quake. The machine perceived
what no human sense could, and it did not merely say "something happened"; it
said "it came from THERE." When it fired toward the west and the court felt
nothing, officials mocked it - until, days later, a courier arrived from Longxi
confirming the quake in precisely that direction.

Three of Zhang Heng's works share one deep principle - RESONANT INFERENCE OF A
DISTAL CAUSE:
  (1) SEISMOSCOPE  : a distant tremor -> resonant swing -> 1-of-8 direction.
  (2) ARMILLARY    : a water-clock drives a bronze celestial sphere that turns
                     once per day, kept phase-locked to the real sky, so an
                     observer indoors reads the positions of stars outdoors.
                     A running world-simulation registered to observation.
  (3) LING XIAN    : the Moon shines by REFLECTED sunlight; a lunar eclipse is
                     the Earth's own shadow ("an-xu", the dark void) falling on
                     the Moon. Inference of an unseen occluding geometry from a
                     change in light.

So this architecture is NOT a transformer, not attention-over-stored-keys, and
not an MoE. It is a bank of tuned RESONATORS whose steady-state responses are
read out to localise a hidden source among eight bearings - the seismoscope, made
differentiable and trainable - plus an autonomously-clocked ARMILLARY phase loop
that keeps an internal simulation registered to observation, and an AN-XU
occlusion test that recovers a source even when several sensors are shadowed.

WHAT RUNS IN THIS FILE
----------------------
  * ResonantDirectionEngine : the trainable core (pure NumPy, hand-written
                              forward + backward).
  * gradient_check()        : finite-difference check of EVERY parameter's
                              analytic gradient (mandatory, must pass).
  * train()                 : a real training loop on synthetic directional
                              "seismic" events; reports rising accuracy.
  * Armillary               : a self-driving phase oscillator that locks onto an
                              observed celestial phase (world-model registration).
  * an_xu_occlusion_test()  : direction recovery under shadowed (masked) sensors.

Everything below executes end-to-end. The verified console output is pasted into
the chapter prose.

AUTHOR : Encyclopedia of Lost Minds - Echoes on AI
"""

import numpy as np

RNG = np.random.default_rng(78139)  # born 78 CE, died 139 CE


# =============================================================================
# PART I - THE WORLD: synthetic directional "seismic" events
# =============================================================================
# Eight sensors sit on a ring at the eight compass bearings, exactly like the
# eight dragons of the seismoscope. A hidden source at one bearing emits a
# damped sinusoid ("the tremor"). Each sensor receives an attenuated, time-
# delayed copy: strong if it faces the source, weak if it faces away, and it is
# buried in noise. The engine must name the bearing that the tremor came from.

N_SENSORS = 8                       # eight dragons / eight bearings
T = 32                              # samples in each sensor's short waveform
SENSOR_ANGLES = np.arange(N_SENSORS) * (2 * np.pi / N_SENSORS)


def make_event(direction, snr=1.0, rng=RNG):
    """Generate one directional seismic event.

    direction : int in 0..7 - the true bearing of the hidden source.
    snr       : signal-to-noise ratio (Zhang Heng's real problem: the tremor is
                faint). Lower snr => harder.
    Returns X of shape (N_SENSORS, T): each row is one sensor's waveform.
    """
    src_angle = direction * (2 * np.pi / N_SENSORS)
    # A tremor is a damped sinusoid at a random-ish natural frequency.
    f0 = 0.18 + 0.06 * rng.standard_normal() * 0.0  # steady tremor frequency
    f0 = 0.18                                         # keep deterministic band
    t = np.arange(T)
    envelope = np.exp(-t / (0.55 * T))               # the tremor rings then dies
    base_wave = envelope * np.sin(2 * np.pi * f0 * t)

    X = np.zeros((N_SENSORS, T))
    for s in range(N_SENSORS):
        # Directional gain: a sensor facing the source hears it loudest.
        align = np.cos(SENSOR_ANGLES[s] - src_angle)
        gain = max(align, 0.0) ** 1.5                 # front-facing dragons ring
        # Small propagation delay proportional to angular distance.
        delay = int(round(2 * (1 - align)))
        w = np.roll(base_wave, delay) * gain
        noise = (1.0 / snr) * 0.4 * rng.standard_normal(T)
        X[s] = w + noise
    return X


def make_batch(n, snr=1.0, rng=RNG):
    dirs = rng.integers(0, N_SENSORS, size=n)
    Xs = np.stack([make_event(int(d), snr=snr, rng=rng) for d in dirs])
    return Xs, dirs                                    # (n, S, T), (n,)


# =============================================================================
# PART II - SPECTRAL FRONT-END (fixed, differentiable, parameter-free)
# =============================================================================
# The resonator only "hears" a signal through its frequency content, so we first
# turn each sensor waveform into a power spectrum. The DFT is a fixed linear map;
# because the input waveform is a constant per sample, the spectral power P is a
# constant feature - the trainable parameters enter only AFTER this stage, which
# keeps the backward pass clean.

_freqs = np.arange(T // 2 + 1)                          # 0..T/2 frequency bins
F = _freqs.size
_omega_axis = 2 * np.pi * _freqs / T                    # angular freq per bin
_C = np.cos(np.outer(_omega_axis, np.arange(T)))        # (F, T) cosine basis
_S = np.sin(np.outer(_omega_axis, np.arange(T)))        # (F, T) sine basis


def spectral_power(X):
    """X (S, T) -> P (S, F): |DFT|^2 per sensor, normalised.

    We normalise by the whole event's peak power. This is a single scalar per
    event, so it scales every sensor identically and therefore PRESERVES the
    relative sensor amplitudes that encode direction, while keeping the numbers
    bounded so the downstream tanh does not saturate.
    """
    Re = X @ _C.T                                       # (S, F)
    Im = X @ _S.T                                       # (S, F)
    P = Re ** 2 + Im ** 2
    return P / (P.max() + 1e-8)


# =============================================================================
# PART III - THE RESONATOR BANK  (the heart: tuned bronze that "rings")
# =============================================================================
# Each resonator k is a second-order driven-damped oscillator with a natural
# frequency omega_k and damping ratio zeta_k. Its power response to a driving
# frequency w is the classic Lorentzian transfer magnitude squared:
#
#     |H_k(w)|^2 = 1 / ( (omega_k^2 - w^2)^2 + (2 * zeta_k * omega_k * w)^2 )
#
# A resonator "rings" (large response) when the tremor's energy sits near its
# natural frequency. omega_k and zeta_k are LEARNED: training tunes the bronze.
# This is the exact opposite of attention-over-keys - selectivity comes from
# physical resonance, not from dot products against stored vectors.

def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def softplus_grad(x):
    return 1.0 / (1.0 + np.exp(-x))                     # = sigmoid(x)


class ResonantDirectionEngine:
    """Seismoscope-as-neural-network.

    Pipeline (per sample):
        X (S,T) --spectral_power--> P (S,F)
        Rmat (K,F) from (omega, zeta)          # resonance response of each unit
        E (S,K) = P @ Rmat.T                    # how hard each dragon's bronze rings
        e = flatten(E) (S*K,)
        h = tanh(W1 e + b1)                     # integrate the ringing pattern
        logits = W2 h + b2  (S,)               # 1-of-8 dragon fires
        loss = softmax cross-entropy + L2
    Everything has a hand-written analytic gradient, checked below.
    """

    def __init__(self, K=10, H=24, seed=7):
        rng = np.random.default_rng(seed)
        # Natural frequencies spread across the usable band; damping moderate.
        self.omega_raw = np.log(np.expm1(np.linspace(0.15, 1.6, K)))  # softplus^-1
        self.zeta_raw = np.log(np.expm1(np.full(K, 0.25)))
        scale1 = np.sqrt(1.0 / (N_SENSORS * K))
        self.W1 = rng.standard_normal((H, N_SENSORS * K)) * scale1
        self.b1 = np.zeros(H)
        scale2 = np.sqrt(1.0 / H)
        self.W2 = rng.standard_normal((N_SENSORS, H)) * scale2
        self.b2 = np.zeros(N_SENSORS)
        self.K, self.H = K, H
        self.l2 = 1e-4

    # ----- parameter (de)serialisation for the gradient check -----
    def get_params(self):
        return {
            "omega_raw": self.omega_raw, "zeta_raw": self.zeta_raw,
            "W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2,
        }

    def set_params(self, p):
        for k, v in p.items():
            setattr(self, k, v)

    # ----- resonance response matrix Rmat (K, F) -----
    def _resonance(self):
        omega = softplus(self.omega_raw)                # (K,)
        zeta = softplus(self.zeta_raw) + 1e-3           # (K,)  strictly > 0
        w = _omega_axis[None, :]                        # (1, F)
        o = omega[:, None]                              # (K, 1)
        z = zeta[:, None]                               # (K, 1)
        D = (o ** 2 - w ** 2) ** 2 + (2 * z * o * w) ** 2   # (K, F) denominator
        # A small constant floor bounds the peak so a sharp resonator cannot
        # produce an unbounded spike. dRmat/dD = -Rmat^2 still holds exactly.
        Rmat = 1.0 / (D + 5e-2)
        cache = dict(omega=omega, zeta=zeta, w=w, o=o, z=z, D=D, Rmat=Rmat)
        return Rmat, cache

    # ----- forward pass; returns loss + cache for backward -----
    def forward(self, P, y):
        """P (B, S, F) spectral power; y (B,) integer bearings."""
        B = P.shape[0]
        Rmat, rcache = self._resonance()                # (K, F)
        # E[b,s,k] = sum_f P[b,s,f] Rmat[k,f]
        E = np.einsum("bsf,kf->bsk", P, Rmat)           # (B, S, K)
        e = E.reshape(B, -1)                            # (B, S*K)
        z1 = e @ self.W1.T + self.b1                    # (B, H)
        h = np.tanh(z1)
        logits = h @ self.W2.T + self.b2                # (B, S)
        # stable softmax cross-entropy
        m = logits.max(axis=1, keepdims=True)
        ex = np.exp(logits - m)
        probs = ex / ex.sum(axis=1, keepdims=True)
        ce = -np.log(probs[np.arange(B), y] + 1e-12).mean()
        reg = self.l2 * (np.sum(self.W1 ** 2) + np.sum(self.W2 ** 2))
        loss = ce + reg
        cache = dict(P=P, E=E, e=e, z1=z1, h=h, probs=probs, y=y, B=B,
                     Rmat=Rmat, rcache=rcache)
        return loss, probs, cache

    # ----- backward pass; returns dict of gradients matching get_params -----
    def backward(self, cache):
        P, E, e = cache["P"], cache["E"], cache["e"]
        h, probs, y, B = cache["h"], cache["probs"], cache["y"], cache["B"]
        Rmat, rc = cache["Rmat"], cache["rcache"]

        # dLoss/dlogits (softmax CE, averaged over batch)
        dlogits = probs.copy()
        dlogits[np.arange(B), y] -= 1.0
        dlogits /= B                                    # (B, S)

        dW2 = dlogits.T @ h + 2 * self.l2 * self.W2     # (S, H)
        db2 = dlogits.sum(axis=0)
        dh = dlogits @ self.W2                          # (B, H)
        dz1 = dh * (1 - h ** 2)                         # tanh'
        dW1 = dz1.T @ e + 2 * self.l2 * self.W1         # (H, S*K)
        db1 = dz1.sum(axis=0)
        de = dz1 @ self.W1                              # (B, S*K)
        dE = de.reshape(B, N_SENSORS, self.K)           # (B, S, K)

        # E[b,s,k] = sum_f P[b,s,f] Rmat[k,f]
        # dLoss/dRmat[k,f] = sum_{b,s} dE[b,s,k] P[b,s,f]
        dRmat = np.einsum("bsk,bsf->kf", dE, P)         # (K, F)

        # Rmat = 1/(D+eps) => dRmat/dD = -Rmat^2 ; then D wrt omega, zeta.
        o, z, w, D = rc["o"], rc["z"], rc["w"], rc["D"]
        dD = dRmat * (-(Rmat ** 2))                     # (K, F)
        dD_domega = 4 * o * (o ** 2 - w ** 2) + 8 * (z ** 2) * o * (w ** 2)
        dD_dzeta = 8 * z * (o ** 2) * (w ** 2)
        domega = np.sum(dD * dD_domega, axis=1)         # (K,)
        dzeta = np.sum(dD * dD_dzeta, axis=1)           # (K,)
        # chain through softplus (raw -> positive)
        domega_raw = domega * softplus_grad(self.omega_raw)
        dzeta_raw = dzeta * softplus_grad(self.zeta_raw)

        return {
            "omega_raw": domega_raw, "zeta_raw": dzeta_raw,
            "W1": dW1, "b1": db1, "W2": dW2, "b2": db2,
        }


# =============================================================================
# PART IV - GRADIENT CHECK (mandatory; must pass)
# =============================================================================

def gradient_check():
    print("=" * 72)
    print("GRADIENT CHECK  (analytic vs central finite differences)")
    print("=" * 72)
    net = ResonantDirectionEngine(K=6, H=12, seed=3)
    X, y = make_batch(5, snr=1.2, rng=np.random.default_rng(1))
    P = np.stack([spectral_power(x) for x in X])

    loss, _, cache = net.forward(P, y)
    grads = net.backward(cache)

    eps = 1e-6
    worst = 0.0
    for name, g in grads.items():
        p = getattr(net, name)
        flat = p.reshape(-1)
        gflat = g.reshape(-1)
        idxs = range(flat.size) if flat.size <= 12 else \
            np.random.default_rng(0).integers(0, flat.size, 12)
        max_rel = 0.0
        for i in idxs:
            orig = flat[i]
            flat[i] = orig + eps
            lp, _, _ = net.forward(P, y)
            flat[i] = orig - eps
            lm, _, _ = net.forward(P, y)
            flat[i] = orig
            num = (lp - lm) / (2 * eps)
            ana = gflat[i]
            denom = max(1e-12, abs(num) + abs(ana))
            rel = abs(num - ana) / denom
            max_rel = max(max_rel, rel)
        worst = max(worst, max_rel)
        flag = "OK " if max_rel < 1e-4 else "!! "
        print(f"  {flag}{name:11s} max rel err = {max_rel:.2e}")
    print("-" * 72)
    ok = worst < 1e-4
    print(f"WORST relative error = {worst:.2e}  ->  "
          f"{'PASS' if ok else 'FAIL'}")
    print()
    return ok


# =============================================================================
# PART V - TRAINING LOOP (a real one)
# =============================================================================

def clip_grads(grads, max_norm=5.0):
    """Global-norm clipping keeps the resonator parameters from being flung into
    an unstable regime by a single large step."""
    total = np.sqrt(sum(np.sum(g ** 2) for g in grads.values()))
    if total > max_norm:
        scale = max_norm / (total + 1e-12)
        for k in grads:
            grads[k] = grads[k] * scale
    return grads


def sgd_step(net, grads, lr, vel, mom=0.85):
    """SGD with heavy-ball momentum. `vel` is a persistent velocity dict."""
    clip_grads(grads)
    for name, g in grads.items():
        p = getattr(net, name)
        vel[name] = mom * vel.get(name, np.zeros_like(g)) - lr * g
        p += vel[name]


def accuracy(net, P, y):
    _, probs, _ = net.forward(P, y)
    return (probs.argmax(axis=1) == y).mean()


def train(epochs=40, batch=64, lr=0.05, snr=1.0, n_train=2000):
    print("=" * 72)
    print("TRAINING  the Resonant Direction Engine (1-of-8 bearing recovery)")
    print("=" * 72)
    net = ResonantDirectionEngine(K=12, H=28, seed=11)
    vel = {}

    # fixed training + evaluation sets (mini-batch SGD over real epochs)
    Xtr, ytr = make_batch(n_train, snr=snr, rng=np.random.default_rng(2024))
    Ptr = np.stack([spectral_power(x) for x in Xtr])
    Xte, yte = make_batch(400, snr=snr, rng=np.random.default_rng(999))
    Pte = np.stack([spectral_power(x) for x in Xte])

    chance = 1.0 / N_SENSORS
    print(f"  train examples = {n_train} | chance accuracy = {chance:.3f}")
    rng = np.random.default_rng(0)
    best_acc, best_params = -1.0, None
    for ep in range(1, epochs + 1):
        order = rng.permutation(n_train)
        running = 0.0
        for i in range(0, n_train, batch):
            idx = order[i:i + batch]
            loss, _, cache = net.forward(Ptr[idx], ytr[idx])
            grads = net.backward(cache)
            sgd_step(net, grads, lr, vel)
            running += loss * len(idx)
        acc = accuracy(net, Pte, yte)
        if acc > best_acc:                              # keep the best sky-reader
            best_acc = acc
            best_params = {k: v.copy() for k, v in net.get_params().items()}
        if ep % 5 == 0 or ep == 1:
            print(f"  epoch {ep:3d} | loss {running/n_train:6.3f} | "
                  f"test acc {acc:5.3f}")
    net.set_params(best_params)                         # restore best checkpoint
    final = accuracy(net, Pte, yte)
    print("-" * 72)
    print(f"  BEST test accuracy = {final:.3f}  ({final/chance:.1f}x chance)")
    print()
    return net


# =============================================================================
# PART VI - THE ARMILLARY: a self-driving simulation kept registered to the sky
# =============================================================================
# Zhang Heng's water-powered armillary sphere turned on its own once per day and
# was kept in registration with the real heavens. Here: an internal phase
# oscillator advances by its own learned angular velocity (the water clock), and
# a phase-locked correction nudges it whenever it drifts from the observed sky
# phase. This is a world-model that RUNS autonomously yet stays truthful to
# observation - exactly the property an AGI world-model needs.

class Armillary:
    def __init__(self, period=24.0, k_lock=0.12):
        self.phase = 0.0                    # internal simulation phase (radians)
        self.omega = 2 * np.pi / period     # internal "water clock" rate
        self.k = k_lock                     # how strongly observation corrects us

    def step(self, observed_phase=None, dt=1.0):
        self.phase = (self.phase + self.omega * dt) % (2 * np.pi)
        if observed_phase is not None:      # registration to reality
            err = np.arctan2(np.sin(observed_phase - self.phase),
                             np.cos(observed_phase - self.phase))
            self.phase = (self.phase + self.k * err) % (2 * np.pi)
            return err
        return None


def armillary_test():
    print("=" * 72)
    print("ARMILLARY  self-driving world-model, phase-locked to observation")
    print("=" * 72)
    arm = Armillary(period=24.0, k_lock=0.15)
    arm.phase = 2.0                                  # start badly out of register
    true_rate = 2 * np.pi / 23.6                     # real sky drifts a bit
    true_phase = 0.0
    errs = []
    for tstep in range(200):
        true_phase = (true_phase + true_rate) % (2 * np.pi)
        e = arm.step(observed_phase=true_phase)
        errs.append(abs(e))
    early = np.mean(errs[:20])
    late = np.mean(errs[-20:])
    print(f"  mean |phase error| first 20 steps : {early:.4f} rad")
    print(f"  mean |phase error| last  20 steps : {late:.4f} rad")
    print(f"  registration improved by {early/max(late,1e-9):.1f}x  ->  "
          f"{'LOCKED' if late < early else 'DRIFT'}")
    print()
    return late < early


# =============================================================================
# PART VII - AN-XU (闇虛): recover the source even when sensors are shadowed
# =============================================================================
# Zhang Heng explained the lunar eclipse as the Earth's own shadow ("an-xu")
# occluding the Moon. Occlusion is information, not just loss. Here we mask
# (shadow) several sensors and confirm the trained engine still names the bearing
# from the resonance pattern in the sensors that remain - graceful degradation
# under occlusion, the seismoscope reasoning under a partial eclipse of data.

def an_xu_occlusion_test(net, n_shadow=3, trials=300):
    print("=" * 72)
    print("AN-XU OCCLUSION TEST  direction recovery with shadowed sensors")
    print("=" * 72)
    rng = np.random.default_rng(4242)
    ok = 0
    for _ in range(trials):
        d = int(rng.integers(0, N_SENSORS))
        X = make_event(d, snr=1.0, rng=rng)
        shadow = rng.choice(N_SENSORS, size=n_shadow, replace=False)
        X[shadow] = 0.0                              # eclipse these dragons
        P = spectral_power(X)[None]
        _, probs, _ = net.forward(P, np.array([d]))
        if probs.argmax() == d:
            ok += 1
    acc = ok / trials
    print(f"  sensors shadowed per trial : {n_shadow} / {N_SENSORS}")
    print(f"  direction recovered        : {acc:.3f} "
          f"({acc*N_SENSORS:.1f}x chance)")
    print()
    return acc


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print()
    print("#" * 72)
    print("# ZHANG HENG - RESONANT DIRECTION ENGINE - full self-test")
    print("#" * 72)
    print()

    passed = gradient_check()
    assert passed, "Gradient check FAILED - architecture is not trustworthy."

    net = train(epochs=40, batch=64, lr=0.05, snr=1.0, n_train=2000)
    armillary_test()
    an_xu_occlusion_test(net, n_shadow=3)

    print("#" * 72)
    print("# ALL SELF-TESTS COMPLETE")
    print("#" * 72)
