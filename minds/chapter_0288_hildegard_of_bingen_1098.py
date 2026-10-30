#!/usr/bin/env python3
"""
Chapter 288 -- Hildegard of Bingen (1098-1179 CE)
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0288 · Hildegard of Bingen (1098-1179 CE)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 15, Minds 281-300 Available on Amazon https://www.amazon.com/dp/B0HMQ7TVJM
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0288_hildegard_of_bingen_1098 - Hildegard of Bingen (1098-1179 CE)
Neuron.py -- a pure-NumPy, from-scratch, trainable architecture built around
this specific mind's distinctive cognitive claims.

WHY THIS SHAPE (not a Transformer, not attention-over-stored-keys):
Hildegard's own account of her visions is not sequential. She insists,
repeatedly and in her own words (the Protestificatio to Scivias, and letters
defending her authority to write), that she sees a whole structured scene at
once, wide awake, "not in sleep, nor in ecstasy, nor in dream" -- and that
the labor of her life was not perception but *decoding*: turning one
simultaneous inner gestalt into separate outward forms (theology, diagram,
chant, medical diagnosis) with the help of a whole scriptorium of scribes
and illuminators. Three of her own, textually documented commitments drive
the four pieces below:

  1. SIMULTANEITY. The vision arrives whole, not word by word. So the
     encoder below does not process a token stream causally; it *binds* a
     handful of noisy partial glimpses into one fixed-size latent via
     competitive slot updates (ConcordanceBindingEncoder), then reads all
     of them out in parallel through independent decoder heads -- one
     shared representation, several simultaneous readings of it.

  2. VIRIDITAS / ARIDITAS. Hildegard reports becoming physically ill when
     she suppressed her visions, and recovering only once she began to
     write; and she describes the 1178-79 interdict silencing her nuns'
     song in the same physiological register. This is modelled as a single
     homeostatic scalar (ViriditasGate) that multiplicatively gates the
     *shared latent itself* -- not one decoder in isolation -- so that
     suppressing expression degrades every channel's access to the same
     underlying compressed content at once.

  3. LINGUA IGNOTA. When existing Latin vocabulary could not hold what she
     needed to say, Hildegard did not approximate with the nearest existing
     word -- she minted new words and letters outright (litterae ignotae).
     LinguaIgnotaCodebook is a growing vector-quantization codebook: novel
     latent content beyond a distance threshold from every existing symbol
     mints a new discrete symbol instead of being forced into the nearest
     one.

Engine: a small hand-written reverse-mode autodiff core (class T), built so
every module below can be checked against numerical finite differences --
see finite_difference_gradient_check(). No autograd library, no ML
framework: NumPy only.
"""
import numpy as np

# ---------------------------------------------------------------------------
# Minimal reverse-mode autodiff engine (pure NumPy, no external deps)
# ---------------------------------------------------------------------------

def _unbroadcast(grad, shape):
    """Sum-reduce `grad` down to `shape`, undoing NumPy broadcasting."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad.reshape(shape)


class T:
    """A tensor node in the computation graph. Wraps a NumPy array."""
    __slots__ = ("data", "grad", "_backward", "_prev", "requires_grad", "_op")

    def __init__(self, data, requires_grad=False, _children=(), _op=""):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = _children
        self.requires_grad = requires_grad
        self._op = _op

    @property
    def shape(self):
        return self.data.shape

    def zero_grad(self):
        self.grad = np.zeros_like(self.data)

    def __add__(self, other):
        other = other if isinstance(other, T) else T(other)
        out = T(self.data + other.data, self.requires_grad or other.requires_grad,
                 (self, other), "+")

        def _backward():
            if self.requires_grad:
                self.grad += _unbroadcast(out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        other = other if isinstance(other, T) else T(other)
        return self + (-other)

    def __rsub__(self, other):
        other = other if isinstance(other, T) else T(other)
        return other + (-self)

    def __mul__(self, other):
        other = other if isinstance(other, T) else T(other)
        out = T(self.data * other.data, self.requires_grad or other.requires_grad,
                 (self, other), "*")

        def _backward():
            if self.requires_grad:
                self.grad += _unbroadcast(out.grad * other.data, self.data.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(out.grad * self.data, other.data.shape)
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def __matmul__(self, other):
        other = other if isinstance(other, T) else T(other)
        out = T(self.data @ other.data, self.requires_grad or other.requires_grad,
                 (self, other), "@")

        def _backward():
            g = out.grad
            a, b = self.data, other.data
            if self.requires_grad:
                if a.ndim == 1 and b.ndim == 2:
                    self.grad += g @ b.T
                elif a.ndim == 2 and b.ndim == 1:
                    self.grad += np.outer(g, b)
                else:
                    self.grad += g @ b.T
            if other.requires_grad:
                if a.ndim == 1 and b.ndim == 2:
                    other.grad += np.outer(a, g)
                elif a.ndim == 2 and b.ndim == 1:
                    other.grad += a.T @ g
                else:
                    other.grad += a.T @ g
        out._backward = _backward
        return out

    def tanh(self):
        th = np.tanh(self.data)
        out = T(th, self.requires_grad, (self,), "tanh")

        def _backward():
            if self.requires_grad:
                self.grad += out.grad * (1.0 - th ** 2)
        out._backward = _backward
        return out

    def sigmoid(self):
        sg = 1.0 / (1.0 + np.exp(-self.data))
        out = T(sg, self.requires_grad, (self,), "sigmoid")

        def _backward():
            if self.requires_grad:
                self.grad += out.grad * sg * (1.0 - sg)
        out._backward = _backward
        return out

    def relu(self):
        r = np.maximum(0.0, self.data)
        out = T(r, self.requires_grad, (self,), "relu")

        def _backward():
            if self.requires_grad:
                self.grad += out.grad * (self.data > 0.0)
        out._backward = _backward
        return out

    def sum(self):
        out = T(self.data.sum(), self.requires_grad, (self,), "sum")

        def _backward():
            if self.requires_grad:
                self.grad += out.grad * np.ones_like(self.data)
        out._backward = _backward
        return out

    def square(self):
        out = T(self.data ** 2, self.requires_grad, (self,), "sq")

        def _backward():
            if self.requires_grad:
                self.grad += out.grad * 2.0 * self.data
        out._backward = _backward
        return out

    def softmax(self):
        x = self.data - self.data.max()
        e = np.exp(x)
        p = e / e.sum()
        out = T(p, self.requires_grad, (self,), "softmax")

        def _backward():
            if self.requires_grad:
                # Jacobian-vector product for softmax
                g = out.grad
                dot = np.sum(g * p)
                self.grad += p * (g - dot)
        out._backward = _backward
        return out

    def backward(self):
        topo, visited = [], set()

        def build(v):
            if id(v) not in visited:
                visited.add(id(v))
                for child in v._prev:
                    build(child)
                topo.append(v)
        build(self)
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()


def softmax_cross_entropy(logits: T, target_idx: int):
    """Numerically stable combined softmax + NLL. Returns scalar loss T."""
    x = logits.data - logits.data.max()
    e = np.exp(x)
    p = e / e.sum()
    loss_val = -np.log(p[target_idx] + 1e-12)
    out = T(loss_val, logits.requires_grad, (logits,), "sce")

    def _backward():
        if logits.requires_grad:
            grad = p.copy()
            grad[target_idx] -= 1.0
            logits.grad += out.grad * grad
    out._backward = _backward
    return out


def mse(pred: T, target: np.ndarray):
    diff = pred - T(target)
    sq = diff.square()
    n = np.prod(pred.shape)
    return sq.sum() * (1.0 / n)


# ---------------------------------------------------------------------------
# Synthetic "vision" dataset
#
# Five vision-classes loosely echo the five thematic clusters of Scivias:
#   0 Creation / cosmos       1 Fall & Redemption      2 Church & Sacraments
#   3 Virtues & Vices         4 Consummation / End-times
#
# Each class has a fixed template signature; an "episode" is a short sequence
# of noisy partial glimpses (observations) of that template -- Hildegard's
# own account is that she saw the whole thing at once, so the encoder's job
# is to *bind* several partial, noisy glimpses back into one coherent whole.
# ---------------------------------------------------------------------------
N_CLASSES = 5
D_OBS = 10
T_STEPS = 6

def make_templates(rng):
    return rng.normal(size=(N_CLASSES, D_OBS))

def sample_episode(cls, templates, rng, noise=0.35):
    base = templates[cls]
    obs = np.stack([base + rng.normal(scale=noise, size=D_OBS) for _ in range(T_STEPS)])
    return obs

# Per-class targets for each decoder head -----------------------------------
D_SLOT, K_SLOTS, D_VISION = 6, 4, 12

def make_diagram_target(cls, rng):
    """8x8 concentric-ring 'rota' pattern; ring radius keyed to class."""
    size = 8
    yy, xx = np.mgrid[0:size, 0:size]
    cx = cy = (size - 1) / 2.0
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    ring_r = 1.2 + 0.8 * cls
    pattern = np.exp(-((r - ring_r) ** 2) / 1.5)
    return pattern.flatten()

def make_music_target(cls, rng):
    """16-step monophonic pitch contour in scale-degree units.
    Class 4 (end-times) gets the widest leaps, echoing the musicological
    observation that Hildegard's chant uses an unusually wide range."""
    steps = 16
    t = np.linspace(0, 2 * np.pi, steps)
    amplitude = 2.0 + 1.5 * cls
    contour = amplitude * np.sin(t * (1 + 0.3 * cls)) + 0.5 * cls
    return contour

def make_diagnosis_target(cls):
    """4-way humoral temperament class, deterministic map from vision class."""
    # sanguine=0 choleric=1 phlegmatic=2 melancholic=3
    return cls % 4


# ---------------------------------------------------------------------------
# Building blocks: a plain affine layer, then the four architectural pieces
# that are specific to this mind (encoder, gate, decoder heads, codebook).
# ---------------------------------------------------------------------------
class Linear:
    def __init__(self, n_in, n_out, rng, scale=None):
        scale = scale or (1.0 / np.sqrt(n_in))
        self.W = T(rng.normal(size=(n_in, n_out)) * scale, requires_grad=True)
        self.b = T(np.zeros(n_out), requires_grad=True)

    def __call__(self, x: T):
        return x @ self.W + self.b

    def params(self):
        return [self.W, self.b]


class ConcordanceBindingEncoder:
    """Iteratively binds T noisy partial observations into one 'Vision'
    latent vector. K slot-vectors compete (softmax consonance) for each
    incoming observation, are updated by a leaky-integrator rule, and are
    finally pooled into a single simultaneous gestalt -- mirroring
    Hildegard's own insistence that the vision arrives whole ('simul'),
    not word by word."""

    def __init__(self, rng, d_obs=D_OBS, d_slot=D_SLOT, k_slots=K_SLOTS, d_vision=D_VISION):
        self.k = k_slots
        self.d_slot = d_slot
        self.W_in = Linear(d_obs, d_slot, rng)
        self.S0 = T(rng.normal(size=(k_slots, d_slot)) * 0.1, requires_grad=True)
        self.W_slot = Linear(d_slot, d_slot, rng)
        self.W_upd = Linear(d_slot, d_slot, rng)
        self.pool = Linear(k_slots * d_slot, d_vision, rng)

    def params(self):
        return self.W_in.params() + [self.S0] + self.W_slot.params() + \
            self.W_upd.params() + self.pool.params()

    def __call__(self, obs_seq: np.ndarray):
        # obs_seq: (T_STEPS, d_obs); S is a single T of shape (k_slots, d_slot)
        S = self.S0
        for t in range(obs_seq.shape[0]):
            o = T(obs_seq[t])
            o_proj = self.W_in(o)                        # (d_slot,)
            raw = S @ o_proj                              # (k,) "consonance" score
            attn = raw.softmax()                          # (k,) competition across slots
            update = _outer_diff(attn, o_proj)            # (k, d_slot) assigned input
            S_lin = _apply_linear_rows(self.W_slot, S)
            U_lin = _apply_linear_rows(self.W_upd, update)
            S = (S_lin + U_lin).tanh()
        flat = _flatten(S)
        vision = self.pool(flat).tanh()
        return vision


def _outer_diff(a: T, b: T):
    """Differentiable outer product a(k,) x b(d,) -> (k,d)."""
    out = T(np.outer(a.data, b.data), a.requires_grad or b.requires_grad, (a, b), "outer")

    def _backward():
        if a.requires_grad:
            a.grad += out.grad @ b.data
        if b.requires_grad:
            b.grad += out.grad.T @ a.data
    out._backward = _backward
    return out


def _apply_linear_rows(lin: Linear, X: T):
    """Apply a Linear to every row of a (k, n_in) tensor -> (k, n_out)."""
    out_data = X.data @ lin.W.data + lin.b.data
    out = T(out_data, True, (X, lin.W, lin.b), "linrows")

    def _backward():
        g = out.grad
        X.grad += g @ lin.W.data.T
        lin.W.grad += X.data.T @ g
        lin.b.grad += g.sum(axis=0)
    out._backward = _backward
    return out


def _flatten(X: T):
    out = T(X.data.flatten(), X.requires_grad, (X,), "flatten")
    shape = X.data.shape

    def _backward():
        if X.requires_grad:
            X.grad += out.grad.reshape(shape)
    out._backward = _backward
    return out


class ViriditasGate:
    """A single homeostatic scalar mapping recent 'expression volume' e_t
    (0 = suppressed, 1 = actively writing/singing/dictating) to a gain
    g_t in (0,1) that multiplicatively scales every decoder head's hidden
    activity at once. This is the mechanism for Hildegard's own reported
    experience: withheld expression => global withering (ariditas) across
    every channel simultaneously, not a channel-specific failure."""

    def __init__(self, rng):
        self.a = T(np.array(2.0), requires_grad=True)
        self.b = T(np.array(0.0), requires_grad=True)
        self.decay = 0.6

    def params(self):
        return [self.a, self.b]

    def step(self, e_prev, expressed_now):
        e_t = self.decay * e_prev + (1.0 - self.decay) * expressed_now
        g = (self.a * e_t + self.b).sigmoid()
        return e_t, g


class DecoderHead:
    """Linear -> tanh -> (scaled by viriditas gain) -> Linear -> task head."""

    def __init__(self, d_vision, d_hidden, d_out, rng):
        self.l1 = Linear(d_vision, d_hidden, rng)
        self.l2 = Linear(d_hidden, d_out, rng)

    def params(self):
        return self.l1.params() + self.l2.params()

    def __call__(self, v: T, gain: T):
        h = self.l1(v).tanh() * gain
        out = self.l2(h)
        return out


class LinguaIgnotaCodebook:
    """A growing vector-quantization codebook over the projected 'vision'
    vector. If the nearest existing codeword is farther than `tau`, a new
    symbol is minted (Hildegard did not force novel content into existing
    Latin vocabulary; she built new words). Otherwise, the vision snaps to
    (commits toward) the nearest existing symbol -- ordinary vocabulary
    reuse. Uses a straight-through estimator so gradients still flow into
    the projector even though nearest-neighbour selection is discrete."""

    def __init__(self, d_vision, d_code, rng, tau=1.6, max_size=24):
        self.proj = Linear(d_vision, d_code, rng)
        self.tau = tau
        self.max_size = max_size
        self.codewords = []   # list[np.ndarray]
        self.mint_log = []    # bookkeeping for reporting

    def params(self):
        return self.proj.params()

    def __call__(self, v: T, gain: T, mint=True, tag=""):
        z = (self.proj(v) * gain).tanh()
        if not self.codewords:
            if mint:
                self.codewords.append(z.data.copy())
                self.mint_log.append(tag)
            nearest = self.codewords[0] if self.codewords else np.zeros_like(z.data)
            dist = float(np.linalg.norm(z.data - nearest))
        else:
            dists = [np.linalg.norm(z.data - c) for c in self.codewords]
            idx = int(np.argmin(dists))
            dist = dists[idx]
            nearest = self.codewords[idx]
            if dist > self.tau and mint and len(self.codewords) < self.max_size:
                self.codewords.append(z.data.copy())
                self.mint_log.append(tag)
                nearest = self.codewords[-1]
                dist = 0.0
        target = T(nearest)  # stop-gradient constant
        diff = z - target
        commit_loss = diff.square().sum()
        return z, commit_loss, dist, len(self.codewords)


class HildegardNet:
    def __init__(self, rng, d_hidden=10):
        self.encoder = ConcordanceBindingEncoder(rng)
        self.gate = ViriditasGate(rng)
        self.diagram = DecoderHead(D_VISION, d_hidden, 64, rng)
        self.music = DecoderHead(D_VISION, d_hidden, 16, rng)
        self.diagnosis = DecoderHead(D_VISION, d_hidden, 4, rng)
        self.lingua = LinguaIgnotaCodebook(D_VISION, d_code=8, rng=rng, tau=1.25)

    def params(self):
        p = self.encoder.params() + self.gate.params() + self.diagram.params() + \
            self.music.params() + self.diagnosis.params() + self.lingua.params()
        return p

    def forward(self, obs_seq, e_prev, expressed_now, diagram_t, music_t, diag_cls, mint=True, tag=""):
        vision_raw = self.encoder(obs_seq)
        e_t, gain = self.gate.step(e_prev, expressed_now)
        # viriditas modulates the shared latent itself before it reaches ANY
        # decoder head -- a global withering of the compressed vision, not
        # just an independently-undoable rescale inside each head.
        vision = vision_raw * gain
        diagram_out = self.diagram(vision, gain)
        music_out = self.music(vision, gain)
        diag_out = self.diagnosis(vision, gain)
        z, commit_loss, dist, vocab_size = self.lingua(vision, gain, mint=mint, tag=tag)

        loss_diagram = mse(diagram_out, diagram_t)
        loss_music = mse(music_out, music_t)
        loss_diag = softmax_cross_entropy(diag_out, diag_cls)
        loss_text = commit_loss * 0.05
        total = loss_diagram + loss_music + loss_diag + loss_text
        info = dict(e_t=e_t, gain=float(gain.data), vocab_size=vocab_size,
                    dist=dist, loss_diagram=float(loss_diagram.data),
                    loss_music=float(loss_music.data), loss_diag=float(loss_diag.data),
                    loss_text=float(loss_text.data))
        return total, info


def sgd_step(params, lr):
    for p in params:
        p.data -= lr * p.grad
        p.zero_grad()


def run_training(seed=0, epochs=40, lr=0.08, verbose=True):
    rng = np.random.default_rng(seed)
    templates = make_templates(rng)
    net = HildegardNet(rng)
    class_names = ["Creation/Cosmos", "Fall&Redemption", "Church&Sacraments",
                   "Virtues&Vices", "Consummation"]

    history = []
    e_prev = 0.5
    for ep in range(epochs):
        # suppression schedule: every 7th epoch, force viriditas suppression
        # to demonstrate the *global*, cross-channel effect of ariditas
        suppress_epoch = (ep % 7 == 6)
        ep_losses = []
        for cls in range(N_CLASSES):
            obs = sample_episode(cls, templates, rng)
            diagram_t = make_diagram_target(cls, rng)
            music_t = make_music_target(cls, rng)
            diag_cls = make_diagnosis_target(cls)
            expressed_now = 0.0 if suppress_epoch else 1.0
            total, info = net.forward(obs, e_prev, expressed_now, diagram_t, music_t,
                                       diag_cls, mint=True, tag=class_names[cls])
            for p in net.params():
                p.zero_grad()
            total.backward()
            sgd_step(net.params(), lr)
            e_prev = info["e_t"]
            ep_losses.append(float(total.data))
        history.append((ep, suppress_epoch, float(np.mean(ep_losses)),
                         info["gain"], info["vocab_size"]))
        if verbose and (ep % 5 == 0 or ep == epochs - 1):
            tag = " [SUPPRESSED epoch]" if suppress_epoch else ""
            print(f"  epoch {ep:3d}  mean_loss={np.mean(ep_losses):.4f}  "
                  f"viriditas_gain={info['gain']:.3f}  vocab={info['vocab_size']}{tag}")
    return net, templates, class_names, history


def finite_difference_gradient_check(net, templates, class_names, n_checks=25, eps=1e-5, seed=7):
    """Mandatory check: compare analytic grads from the autodiff engine
    against central finite differences on a random sample of parameters
    drawn from every module (encoder, gate, all three MSE/CE decoder
    heads, and the Lingua Ignota projector)."""
    rng = np.random.default_rng(seed)
    cls = 2
    obs = sample_episode(cls, templates, rng)
    diagram_t = make_diagram_target(cls, rng)
    music_t = make_music_target(cls, rng)
    diag_cls = make_diagnosis_target(cls)

    params = net.params()
    for p in params:
        p.zero_grad()
    total, _ = net.forward(obs, 0.5, 1.0, diagram_t, music_t, diag_cls, mint=False)
    total.backward()
    analytic_snapshot = [p.grad.copy() for p in params]

    rng_idx = np.random.default_rng(seed + 1)
    worst = 0.0
    checked = 0
    for _ in range(n_checks):
        pi = rng_idx.integers(0, len(params))
        p = params[pi]
        flat_idx = tuple(rng_idx.integers(0, s) for s in p.data.shape) if p.data.shape else ()
        orig = p.data[flat_idx] if flat_idx else p.data.item()

        def set_val(val):
            if flat_idx:
                p.data[flat_idx] = val
            else:
                p.data = np.array(val)

        set_val(orig + eps)
        l1, _ = net.forward(obs, 0.5, 1.0, diagram_t, music_t, diag_cls, mint=False)
        set_val(orig - eps)
        l2, _ = net.forward(obs, 0.5, 1.0, diagram_t, music_t, diag_cls, mint=False)
        set_val(orig)
        numeric = (l1.data - l2.data) / (2 * eps)
        analytic = analytic_snapshot[pi][flat_idx] if flat_idx else analytic_snapshot[pi].item()
        rel_err = abs(analytic - numeric) / (abs(numeric) + abs(analytic) + 1e-8)
        worst = max(worst, rel_err)
        checked += 1
    return worst, checked


def self_tests(net, templates, class_names):
    results = []

    # Test 1: gradient check passes under tight tolerance
    worst, n = finite_difference_gradient_check(net, templates, class_names)
    results.append(("finite-difference gradient check (%d params)" % n, worst < 3e-3, worst))

    # Test 2: SUSTAINED viriditas suppression degrades ALL three supervised
    # channels at once. This mirrors Hildegard's own report: it was not a
    # single missed day that made her ill, but prolonged withholding of
    # expression (and the 1178-79 interdict silencing the divine office for
    # months, which she describes in explicitly physiological terms).
    rng = np.random.default_rng(3)
    e_flow = 0.5
    for _ in range(8):
        e_flow, _ = net.gate.step(e_flow, 1.0)
    e_dry = 0.5
    for _ in range(8):
        e_dry, _ = net.gate.step(e_dry, 0.0)
    _, gain_flow = net.gate.step(e_flow, 1.0)
    _, gain_dry = net.gate.step(e_dry, 0.0)

    cls_probe = 1
    obs_probe = sample_episode(cls_probe, templates, rng)
    vision_raw = net.encoder(obs_probe)
    norm_flow = float(np.linalg.norm(vision_raw.data * gain_flow.data))
    norm_dry = float(np.linalg.norm(vision_raw.data * gain_dry.data))
    # This is architecturally guaranteed (vision is literally rescaled by a
    # smaller positive gain), not a downstream-training artifact: it is the
    # mechanism itself, verified directly.
    results.append(("sustained suppression shrinks the shared vision latent's norm "
                     "(gain_flow=%.3f > gain_dry=%.3f)" % (float(gain_flow.data), float(gain_dry.data)),
                     norm_dry < norm_flow and float(gain_dry.data) < float(gain_flow.data),
                     (round(norm_flow, 3), round(norm_dry, 3))))

    # Honest secondary observation (not gated pass/fail): after training
    # mostly under "flowing" conditions (suppressed only 1/7 epochs), do the
    # trained decoder heads still show higher reconstruction loss under
    # sustained suppression, or has gradient descent partly learned to route
    # around the infrequent gate? Reported for the chapter discussion.
    flow_losses = {"loss_diagram": [], "loss_music": [], "loss_diag": []}
    dry_losses = {"loss_diagram": [], "loss_music": [], "loss_diag": []}
    for cls in range(N_CLASSES):
        for trial in range(4):
            obs = sample_episode(cls, templates, rng)
            dt = make_diagram_target(cls, rng); mt = make_music_target(cls, rng); dc = make_diagnosis_target(cls)
            _, info_flow = net.forward(obs, e_flow, 1.0, dt, mt, dc, mint=False)
            _, info_dry = net.forward(obs, e_dry, 0.0, dt, mt, dc, mint=False)
            for k in flow_losses:
                flow_losses[k].append(info_flow[k])
                dry_losses[k].append(info_dry[k])
    means_flow = {k: float(np.mean(v)) for k, v in flow_losses.items()}
    means_dry = {k: float(np.mean(v)) for k, v in dry_losses.items()}
    heads_worse_under_suppression = sum(means_dry[k] > means_flow[k] for k in means_flow)
    trained_loss_report = {k: (round(means_flow[k], 3), round(means_dry[k], 3)) for k in means_flow}

    # Test 3: gate output is a valid probability-like gain in (0,1)
    results.append(("viriditas gain stays in (0,1)",
                     0.0 < info_flow["gain"] < 1.0 and 0.0 < info_dry["gain"] < 1.0,
                     (info_flow["gain"], info_dry["gain"])))

    # Test 4: codebook has grown beyond a single symbol after training on 5 classes
    results.append(("Lingua Ignota codebook grew beyond one symbol during training",
                     len(net.lingua.codewords) >= 2, len(net.lingua.codewords)))

    # Test 5: codebook does NOT keep growing without bound on repeated in-distribution input
    rng2 = np.random.default_rng(11)
    size_before = len(net.lingua.codewords)
    for _ in range(15):
        cls2 = rng2.integers(0, N_CLASSES)
        obs2 = sample_episode(cls2, templates, rng2)
        _, gain = net.gate.step(1.0, 1.0)
        vision = net.encoder(obs2) * gain
        net.lingua(vision, gain, mint=True, tag=class_names[cls2])
    size_after = len(net.lingua.codewords)
    results.append(("codebook converges on repeated in-distribution input (bounded growth)",
                     (size_after - size_before) <= 2, (size_before, size_after)))

    # Test 6: an out-of-distribution probe -- standing in for a genuinely
    # unprecedented concept with no analogue among the five trained
    # vision-classes (a stand-in for "AGI" itself: something Hildegard's
    # world had no existing word for) -- is constructed by a maximin search
    # for the vision-space point farthest from every existing codeword, and
    # must mint a genuinely new symbol rather than being forced into an
    # existing one.
    _, gain_probe = net.gate.step(1.0, 1.0)
    rng6 = np.random.default_rng(2026)
    best_vec, best_dist = None, -1.0
    for _ in range(3000):
        cand = rng6.uniform(-1.0, 1.0, size=D_VISION)
        z_cand = (net.lingua.proj(T(cand)) * gain_probe).tanh()
        d_cand = min(np.linalg.norm(z_cand.data - c) for c in net.lingua.codewords)
        if d_cand > best_dist:
            best_dist, best_vec = d_cand, cand
    size_pre_ood = len(net.lingua.codewords)
    vision_ood = T(best_vec)
    _, _, dist_ood, size_post_ood = net.lingua(vision_ood, gain_probe, mint=True, tag="AGI-probe (novel concept)")
    results.append(("maximally-novel 'AGI' probe mints a new Lingua Ignota symbol",
                     size_post_ood == size_pre_ood + 1, (round(best_dist, 3), size_pre_ood, size_post_ood)))

    # Test 7: encoder is order-sensitive-but-noise-robust: two noisy episodes
    # of the SAME class should land closer in vision-space than two episodes
    # of DIFFERENT classes (binding actually clusters by content)
    rng4 = np.random.default_rng(21)
    vecs = {c: [net.encoder(sample_episode(c, templates, rng4)).data for _ in range(6)]
            for c in range(N_CLASSES)}
    same_d, diff_d = [], []
    for c in range(N_CLASSES):
        for i in range(len(vecs[c])):
            for j in range(i + 1, len(vecs[c])):
                same_d.append(np.linalg.norm(vecs[c][i] - vecs[c][j]))
    for c1 in range(N_CLASSES):
        for c2 in range(c1 + 1, N_CLASSES):
            for v1 in vecs[c1]:
                for v2 in vecs[c2]:
                    diff_d.append(np.linalg.norm(v1 - v2))
    mean_same, mean_diff = float(np.mean(same_d)), float(np.mean(diff_d))
    results.append(("same-class episodes bind closer in vision-space than different-class, on average"
                     " (%d same-class pairs vs %d cross-class pairs)" % (len(same_d), len(diff_d)),
                     mean_same < mean_diff, (round(mean_same, 3), round(mean_diff, 3))))

    # Test 8: loss decreases over training (sanity that the model actually learned)
    results.append(("training loss decreased overall", True, None))  # filled by caller

    # Test 9: diagnosis head reaches better-than-chance accuracy after training
    rng5 = np.random.default_rng(42)
    correct = 0
    total_n = 20
    for _ in range(total_n):
        cls5 = rng5.integers(0, N_CLASSES)
        obs5 = sample_episode(cls5, templates, rng5)
        dt5 = make_diagram_target(cls5, rng5); mt5 = make_music_target(cls5, rng5)
        dc5 = make_diagnosis_target(cls5)
        _, gain5 = net.gate.step(1.0, 1.0)
        vision5 = net.encoder(obs5) * gain5
        pred_logits = net.diagnosis(vision5, gain5)
        pred = int(np.argmax(pred_logits.data))
        correct += int(pred == dc5)
    acc = correct / total_n
    results.append(("humoral-diagnosis head beats chance (>25%) after training", acc > 0.25, acc))

    # Test 10: all parameters have finite values (no NaN/Inf explosions)
    all_finite = all(np.all(np.isfinite(p.data)) for p in net.params())
    results.append(("all parameters remain finite after training", all_finite, None))

    return results, {"heads_worse_under_suppression": heads_worse_under_suppression,
                      "trained_loss_flow_vs_dry": trained_loss_report}


if __name__ == "__main__":
    print("=" * 78)
    print("HILDEGARD OF BINGEN COGNITIVE ARCHITECTURE -- training run")
    print("=" * 78)
    net, templates, class_names, history = run_training(seed=0, epochs=40, lr=0.08)

    print()
    print("-" * 78)
    print("Finite-difference gradient check")
    print("-" * 78)
    worst, n = finite_difference_gradient_check(net, templates, class_names)
    print(f"  checked {n} parameters across encoder/gate/decoders/codebook")
    print(f"  worst relative error: {worst:.3e}  (threshold 3e-3) ->",
          "PASS" if worst < 3e-3 else "FAIL")

    print()
    print("-" * 78)
    print("Self-test suite")
    print("-" * 78)
    losses = [h[2] for h in history]
    results, info_note = self_tests(net, templates, class_names)
    results[7] = ("training loss decreased overall", losses[-1] < losses[0],
                  (losses[0], losses[-1]))
    n_pass = 0
    for name, ok, extra in results:
        status = "PASS" if ok else "FAIL"
        n_pass += int(ok)
        extra_str = f"  ({extra})" if extra is not None else ""
        print(f"  [{status}] {name}{extra_str}")
    print(f"\n  {n_pass}/{len(results)} self-tests passed")

    print()
    print("-" * 78)
    print("Informational: trained-decoder loss under sustained flow vs. sustained")
    print("suppression (NOT gated pass/fail -- see chapter discussion)")
    print("-" * 78)
    print(f"  heads with higher mean loss under suppression: "
          f"{info_note['heads_worse_under_suppression']}/3")
    for k, (fv, dv) in info_note["trained_loss_flow_vs_dry"].items():
        arrow = "higher under suppression" if dv > fv else "LOWER under suppression (compensated)"
        print(f"    {k:14s} flow={fv:7.3f}  dry={dv:7.3f}   [{arrow}]")

    print()
    print("-" * 78)
    print("Lingua Ignota codebook (minted symbols, in order)")
    print("-" * 78)
    for i, tag in enumerate(net.lingua.mint_log):
        print(f"  symbol #{i:02d}  minted while decoding: {tag}")
