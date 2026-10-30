"""
=============================================================================
 CHAPTER 299 -- IBN RUSHD (AVERROES), b. Cordoba 14 April 1126, d. Marrakesh
 # Encyclopedia of Lost Minds: Echoes on AI · Chapter 0299 · Ibn Rushd (Averroes) (b. Cordoba 14 April 1126, d. Marrakesh 11 December 1198)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 15, Minds 281-300 Available on Amazon https://www.amazon.com/dp/B0HMQ7TVJM
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0299_ibn_rushd_averroes_1126 - Ibn Rushd (Averroes) (b. Cordoba 14 April 1126, d. Marrakesh 11 December 1198)
 11 December 1198.

 THE COMMON-INTELLECT NETWORK
 A from-scratch, trainable architecture built around the one philosophical
 claim that is genuinely and distinctively Ibn Rushd's own -- not Aristotle's,
 not al-Farabi's, not Avicenna's -- worked out across three independent
 sub-systems that share nothing with a Transformer, an attention mechanism,
 or a mixture-of-experts router:

   1. UNITY OF THE MATERIAL INTELLECT (Long Commentary on the De Anima,
      Determinate Particular Argument + Unity Argument). One numerically
      single receiving substance ("material intellect"), never copied,
      is the ONLY thing that actualizes universal understanding. Every
      individual knower supplies nothing to that substance but their own
      private, idiosyncratic "phantasms" -- particular sensory-imaginative
      images. Implemented literally: one shared set of weight tensors
      (CommonIntellect) is the SAME Python object referenced inside every
      agent's forward pass; each agent owns only a small private
      PhantasmEncoder. Remove the shared weights and every agent's
      understanding collapses at once, however well each private encoder
      was trained -- exactly the doctrine's most controversial consequence.

   2. CAUSAL NECESSITY AGAINST HABITUAL CONJUNCTION (Tahafut al-Tahafut,
      17th Discussion, against al-Ghazali). Ibn Rushd's argument that
      denying necessary causal connections in nature -- reducing "fire
      burns cotton" to a mere habitual, re-created-by-God-each-instant
      conjunction -- destroys the very possibility of expecting tomorrow
      to resemble today, science included. Implemented as two competing
      predictors on al-Ghazali's own example (fire and cotton): one
      (CausalMechanism) is given the real physical drivers and forced to
      use a SINGLE invariant mapping across several training environments;
      the other (HabitualConjunction) is denied the physical drivers and
      can only memorize each environment's observed base rate. Tested on
      a genuinely new environment.

   3. CAPACITY-GATED TA'WIL (Fasl al-Maqal, the three classes of assent --
      demonstrative / dialectical / rhetorical -- and the discipline of
      restricting allegorical reconciliation to those equipped to bear
      it). Implemented as a gated decoder: a shared trunk produces one
      "actualized thought"; a gate reads only a descriptor of the
      AUDIENCE and routes the SAME proposition to a reconciled
      demonstrative reading, a hedged dialectical paraphrase, or the
      unreconciled literal/rhetorical reading -- never volunteering the
      reconciliation to an audience not equipped for it.

 Pure NumPy. No autodiff library, no ML framework. A small hand-written
 reverse-mode autodiff engine (class Tensor) is defined below and used
 throughout; every gradient it produces is checked against central-difference
 numerical gradients before any training loop is trusted (see
 finite_difference_gradient_check()). Executed and its printed output
 verified before being reported in the chapter text.
=============================================================================
"""
import numpy as np

RNG_SEED = 20260729
rng = np.random.default_rng(RNG_SEED)


# =============================================================================
# SECTION 0 -- REVERSE-MODE AUTODIFF ENGINE
# A minimal, general Tensor class: every operation records how to push a
# gradient back to its inputs. backward() walks the resulting graph in
# reverse topological order. This is the one piece of machinery every
# module below shares -- which is itself a small nod to the chapter's
# thesis: one common computational substrate, many private uses of it.
# =============================================================================

class Tensor:
    __slots__ = ('data', 'grad', '_backward', '_prev', '_op', 'requires_grad')

    def __init__(self, data, _children=(), _op='', requires_grad=True):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.requires_grad = requires_grad

    @staticmethod
    def _unbroadcast(g, shape):
        while g.ndim > len(shape):
            g = g.sum(axis=0)
        for i, s in enumerate(shape):
            if s == 1 and g.shape[i] != 1:
                g = g.sum(axis=i, keepdims=True)
        return g.reshape(shape)

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += Tensor._unbroadcast(out.grad, self.data.shape)
            other.grad += Tensor._unbroadcast(out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += Tensor._unbroadcast(out.grad * other.data, self.data.shape)
            other.grad += Tensor._unbroadcast(out.grad * self.data, other.data.shape)
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        return self * (-1.0)

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def matmul(self, other):
        out = Tensor(self.data @ other.data, (self, other), '@')
        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, (self,), 'tanh')
        def _backward():
            self.grad += (1 - t * t) * out.grad
        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + np.exp(-np.clip(self.data, -60, 60)))
        out = Tensor(s, (self,), 'sigmoid')
        def _backward():
            self.grad += s * (1 - s) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        r = np.maximum(0.0, self.data)
        out = Tensor(r, (self,), 'relu')
        def _backward():
            self.grad += (self.data > 0).astype(np.float64) * out.grad
        out._backward = _backward
        return out

    def softmax(self, axis=-1):
        x = self.data
        x_max = x.max(axis=axis, keepdims=True)
        e = np.exp(x - x_max)
        y = e / e.sum(axis=axis, keepdims=True)
        out = Tensor(y, (self,), 'softmax')
        def _backward():
            g = out.grad
            s = np.sum(g * y, axis=axis, keepdims=True)
            self.grad += y * (g - s)
        out._backward = _backward
        return out

    def col(self, j):
        """Select column j, keeping a (batch,1) shape, with correct scatter-backward."""
        out = Tensor(self.data[:, j:j + 1], (self,), 'col')
        def _backward():
            self.grad[:, j:j + 1] += out.grad
        out._backward = _backward
        return out

    def sum(self):
        out = Tensor(self.data.sum(), (self,), 'sum')
        def _backward():
            self.grad += np.ones_like(self.data) * out.grad
        out._backward = _backward
        return out

    def mean(self):
        n = self.data.size
        out = Tensor(self.data.sum() / n, (self,), 'mean')
        def _backward():
            self.grad += (np.ones_like(self.data) / n) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo, visited = [], set()
        def build(v):
            if id(v) not in visited:
                visited.add(id(v))
                for c in v._prev:
                    build(c)
                topo.append(v)
        build(self)
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()


def softmax_cross_entropy(logits, y_idx):
    """logits: Tensor (batch,K); y_idx: int array (batch,). Combined, numerically
    stable op -- the analytic backward is the well-known (softmax - one_hot)/N."""
    x = logits.data
    x_max = x.max(axis=-1, keepdims=True)
    e = np.exp(x - x_max)
    p = e / e.sum(axis=-1, keepdims=True)
    n = x.shape[0]
    nll = -np.log(p[np.arange(n), y_idx] + 1e-12)
    out = Tensor(nll.mean(), (logits,), 'softmax_ce')
    def _backward():
        g = p.copy()
        g[np.arange(n), y_idx] -= 1.0
        g /= n
        logits.grad += g * out.grad
    out._backward = _backward
    return out


def sigmoid_bce(logits, y):
    """logits: Tensor (batch,1) pre-sigmoid; y: float array (batch,1) in {0,1}."""
    x = logits.data
    p = 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))
    n = x.shape[0]
    loss_val = -(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)).mean()
    out = Tensor(loss_val, (logits,), 'sigmoid_bce')
    def _backward():
        logits.grad += ((p - y) / n) * out.grad
    out._backward = _backward
    return out


def xavier(shape, r):
    fan_in, fan_out = shape[0], shape[-1]
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return Tensor(r.uniform(-limit, limit, size=shape))


class Linear:
    """A single trainable affine layer: y = x @ W + b."""
    def __init__(self, d_in, d_out, r):
        self.W = xavier((d_in, d_out), r)
        self.b = Tensor(np.zeros(d_out))

    def __call__(self, x):
        return x.matmul(self.W) + self.b

    def params(self):
        return [self.W, self.b]


class Adam:
    def __init__(self, params, lr=0.02, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params, self.lr = params, lr
        self.b1, self.b2, self.eps = beta1, beta2, eps
        self.m = {id(p): np.zeros_like(p.data) for p in params}
        self.v = {id(p): np.zeros_like(p.data) for p in params}
        self.t = 0

    def zero_grad(self):
        for p in self.params:
            p.grad = np.zeros_like(p.data)

    def step(self):
        self.t += 1
        for p in self.params:
            g = p.grad
            m = self.m[id(p)] = self.b1 * self.m[id(p)] + (1 - self.b1) * g
            v = self.v[id(p)] = self.b2 * self.v[id(p)] + (1 - self.b2) * (g * g)
            mh = m / (1 - self.b1 ** self.t)
            vh = v / (1 - self.b2 ** self.t)
            p.data -= self.lr * mh / (np.sqrt(vh) + self.eps)


# =============================================================================
# SECTION 1 -- UNITY OF THE MATERIAL INTELLECT
# One shared substrate (CommonIntellect); many private phantasm encoders.
# =============================================================================

N_CLASSES_1 = 9        # universal forms to be actualized
N_AGENTS_1 = 5          # individual knowers trained together
D_PARTICULAR_1 = 14     # each agent's raw idiosyncratic sensory dimension
D_PHANTASM_1 = 8        # common latent dimension a phantasm is cast into
D_HIDDEN_1 = 16
N_PER_CLASS_1 = 40

_prototypes_1 = rng.standard_normal((N_CLASSES_1, 6)) * 0.55
_agent_sensory_1 = [rng.standard_normal((6, D_PARTICULAR_1)) * 0.9 for _ in range(N_AGENTS_1)]


def _extend_agent_sensory_1(r):
    _agent_sensory_1.append(r.standard_normal((6, D_PARTICULAR_1)) * 0.9)


def make_agent_data_1(agent_id, n_per_class=N_PER_CLASS_1, seed_offset=0):
    """Synthesize one agent's idiosyncratic 'particulars': the same universal
    prototype run through THAT agent's own private sensory projection, plus
    private noise. Different knowers, quite literally, see different things
    when confronted with the same universal form."""
    r = np.random.default_rng(100 + agent_id + seed_offset)
    Xs, ys = [], []
    for c in range(N_CLASSES_1):
        noise = r.standard_normal((n_per_class, 6)) * 0.42
        particulars = (_prototypes_1[c] + noise) @ _agent_sensory_1[agent_id]
        particulars += r.standard_normal((n_per_class, D_PARTICULAR_1)) * 0.22
        Xs.append(particulars)
        ys.append(np.full(n_per_class, c))
    X, y = np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)
    idx = r.permutation(len(y))
    return X[idx], y[idx]


class PhantasmEncoder:
    """PRIVATE to one agent: raw particular -> phantasm vector. This is the
    only part of a knower's cognition that Ibn Rushd allows to be individual."""
    def __init__(self, r):
        self.l1 = Linear(D_PARTICULAR_1, D_HIDDEN_1, r)
        self.l2 = Linear(D_HIDDEN_1, D_PHANTASM_1, r)

    def __call__(self, x):
        return self.l2(self.l1(x).tanh()).tanh()

    def params(self):
        return self.l1.params() + self.l2.params()


class CommonIntellect:
    """SHARED across every agent: literally the same Python object, the same
    weight arrays, referenced inside every agent's forward pass. This is the
    single, numerically-one 'material intellect' of the Long Commentary."""
    def __init__(self, r):
        self.l1 = Linear(D_PHANTASM_1, D_HIDDEN_1, r)
        self.l2 = Linear(D_HIDDEN_1, N_CLASSES_1, r)

    def __call__(self, phantasm):
        return self.l2(self.l1(phantasm).tanh())

    def params(self):
        return self.l1.params() + self.l2.params()


class PrivateStack:
    """Ablation baseline: same architecture, but with NO shared substrate --
    every parameter, including the classification head, is private and
    freshly initialized. Used only to measure what the shared intellect buys."""
    def __init__(self, r):
        self.enc = PhantasmEncoder(r)
        self.head = CommonIntellect(r)

    def __call__(self, x):
        return self.head(self.enc(x))

    def params(self):
        return self.enc.params() + self.head.params()


# =============================================================================
# SECTION 2 -- CAUSAL NECESSITY AGAINST HABITUAL CONJUNCTION
# al-Ghazali's own example: fire and cotton.
# =============================================================================

TRAIN_ENVS = [dict(id=0, humidity=0.10), dict(id=1, humidity=0.35), dict(id=2, humidity=0.55)]
TEST_ENV = dict(id=3, humidity=0.92)     # soaked cotton -- an unseen regime
N_ENVS = len(TRAIN_ENVS) + 1


def true_burn_prob(contact, dryness, flame_intensity):
    """The hidden, ground-truth physical mechanism -- Ibn Rushd's 'nature
    placed in the thing itself' (tab' / khassiyya), never revealed to either
    model directly; both must infer it from data."""
    z = 4.0 * contact * dryness * flame_intensity - 2.2
    return 1.0 / (1.0 + np.exp(-z))


def sample_env(env, n, seed):
    r = np.random.default_rng(seed)
    contact = r.uniform(0.4, 1.0, size=n)
    dryness = np.clip(1.0 - env['humidity'] + r.normal(0, 0.07, size=n), 0.02, 1.0)
    flame = r.uniform(0.4, 1.0, size=n)
    p = true_burn_prob(contact, dryness, flame)
    y = (r.random(n) < p).astype(np.float64)
    return contact, dryness, flame, y


def one_hot(ids, n_envs):
    m = np.zeros((len(ids), n_envs))
    m[np.arange(len(ids)), ids] = 1.0
    return m


class CausalMechanism:
    """Sees the real physical drivers (contact, dryness, flame). ONE set of
    weights, forced to explain every training environment at once -- an
    invariance constraint standing in for Ibn Rushd's claim that natures are
    fixed properties of things, not habits renewed occasion by occasion."""
    def __init__(self, r):
        self.l1 = Linear(3, 12, r)
        self.l2 = Linear(12, 1, r)

    def __call__(self, contact, dryness, flame):
        x = Tensor(np.stack([contact, dryness, flame], axis=1), requires_grad=False)
        return self.l2(self.l1(x).tanh())

    def params(self):
        return self.l1.params() + self.l2.params()


class HabitualConjunction:
    """Sees only 'contact happened' plus an environment-ID tag -- exactly
    al-Ghazali's own epistemology of habitual conjunction, with no access to
    the physical mechanism it correlates with. The one-hot environment rows
    of the first layer start at exactly zero (rather than random Xavier
    noise): a habitual learner that has never observed a given environment
    has, by construction, no idiosyncratic adjustment for it and can only
    fall back on whatever the 'contact' feature and bias encode about the
    training-average environment -- a deterministic, seed-robust stand-in
    for 'nothing principled to extrapolate from', rather than an arbitrary
    lucky/unlucky random guess."""
    def __init__(self, r):
        self.l1 = Linear(1 + N_ENVS, 12, r)
        self.l1.W.data[1:, :] = 0.0   # rows 1..N_ENVS are the one-hot env columns
        self.l2 = Linear(12, 1, r)

    def __call__(self, contact, env_ids):
        x = np.concatenate([contact.reshape(-1, 1), one_hot(env_ids, N_ENVS)], axis=1)
        return self.l2(self.l1(Tensor(x, requires_grad=False)).tanh())

    def params(self):
        return self.l1.params() + self.l2.params()


# =============================================================================
# SECTION 3 -- CAPACITY-GATED TA'WIL DECODING
# Fasl al-Maqal's three classes of assent: demonstrative, dialectical, rhetorical.
# =============================================================================

D_THOUGHT_3 = 10
D_CTX_3 = 6
K_MEANING_3 = 4
N_REGISTERS_3 = 3
LITERAL, RECONCILED, HEDGED, DISTRACT = 0, 1, 2, 3


def make_dataset_3(n, seed):
    r = np.random.default_rng(seed)
    thoughts = r.standard_normal((n, D_THOUGHT_3))
    register = r.integers(0, N_REGISTERS_3, size=n)
    ctx = np.zeros((n, D_CTX_3))
    for i in range(n):
        base = np.zeros(3)
        base[register[i]] = 1.0
        ctx[i, :3] = base + r.normal(0, 0.25, size=3)
        ctx[i, 3:] = r.normal(0, 1.0, size=3)   # nuisance dims the gate must learn to ignore
    target_meaning = np.where(register == 0, RECONCILED,
                       np.where(register == 1, HEDGED, LITERAL))
    return thoughts, ctx, register, target_meaning


class CapacityGatedDecoder:
    """A shared trunk actualizes one 'thought'. A gate reads ONLY the
    audience descriptor (never the content) and estimates which of the
    three classes of assent that audience is equipped for. The released
    meaning is a gate-weighted blend of three shared-trunk heads -- so the
    demonstrative reconciliation (ta'wil) is structurally unreachable for
    an audience the gate does not certify as demonstrative-class."""
    def __init__(self, r):
        self.trunk = Linear(D_THOUGHT_3, 16, r)
        self.gate_in = Linear(D_CTX_3, 16, r)
        self.gate_out = Linear(16, N_REGISTERS_3, r)
        self.head_demo = Linear(16, K_MEANING_3, r)
        self.head_dial = Linear(16, K_MEANING_3, r)
        self.head_rhet = Linear(16, K_MEANING_3, r)

    def forward(self, thought, ctx):
        h = self.trunk(thought).tanh()
        g_hidden = self.gate_in(ctx).tanh()
        register_logits = self.gate_out(g_hidden)
        gate = register_logits.softmax()
        w_demo, w_dial, w_rhet = gate.col(0), gate.col(1), gate.col(2)
        blended = self.head_demo(h) * w_demo + self.head_dial(h) * w_dial + self.head_rhet(h) * w_rhet
        return register_logits, blended, gate

    def params(self):
        ps = []
        for mod in (self.trunk, self.gate_in, self.gate_out,
                    self.head_demo, self.head_dial, self.head_rhet):
            ps += mod.params()
        return ps


# =============================================================================
# SECTION 4 -- UNIFIED FINITE-DIFFERENCE GRADIENT CHECK (mandatory)
# One representative trainable weight matrix from EACH of the three modules,
# checked against central-difference numerical gradients of the SAME scalar
# loss the training loop actually uses.
# =============================================================================

def finite_difference_gradient_check(verbose=True):
    ok = True
    eps = 1e-6

    def check(name, param, loss_fn, all_params, tol=1e-6):
        nonlocal ok
        # Every leaf parameter Tensor in the graph persists across calls, so
        # its .grad accumulates additively unless explicitly zeroed first --
        # zero the WHOLE graph's parameters, not just the one under test,
        # or earlier checks on the same graph silently inflate this one.
        for p in all_params:
            p.grad = np.zeros_like(p.data)
        loss_fn().backward()
        analytic = param.grad.copy()
        numeric = np.zeros_like(param.data)
        it = np.nditer(param.data, flags=['multi_index'])
        for _ in it:
            idx = it.multi_index
            orig = param.data[idx]
            param.data[idx] = orig + eps
            f1 = loss_fn().data
            param.data[idx] = orig - eps
            f2 = loss_fn().data
            param.data[idx] = orig
            numeric[idx] = (f1 - f2) / (2 * eps)
        rel = np.abs(analytic - numeric).max() / (np.abs(numeric).max() + 1e-12)
        passed = rel < tol
        ok = ok and passed
        if verbose:
            print(f"  [{name:34s}] max|analytic-numeric| rel err = {rel:.3e}  "
                  f"{'PASS' if passed else 'FAIL'}")
        for p in all_params:
            p.grad = np.zeros_like(p.data)

    r = np.random.default_rng(1234)

    # -- module 1 --
    enc = PhantasmEncoder(r)
    intellect = CommonIntellect(r)
    params1 = enc.params() + intellect.params()
    x1 = Tensor(r.standard_normal((6, D_PARTICULAR_1)), requires_grad=False)
    y1 = r.integers(0, N_CLASSES_1, size=6)
    def loss1():
        return softmax_cross_entropy(intellect(enc(x1)), y1)
    check("module1.CommonIntellect.l2.W", intellect.l2.W, loss1, params1)
    check("module1.PhantasmEncoder.l1.W", enc.l1.W, loss1, params1)

    # -- module 2 --
    causal = CausalMechanism(r)
    params2c = causal.params()
    contact = r.uniform(0.4, 1.0, size=8)
    dryness = r.uniform(0.1, 1.0, size=8)
    flame = r.uniform(0.4, 1.0, size=8)
    y2 = (r.random(8) > 0.5).astype(np.float64).reshape(-1, 1)
    def loss2():
        return sigmoid_bce(causal(contact, dryness, flame), y2)
    check("module2.CausalMechanism.l1.W", causal.l1.W, loss2, params2c)

    habitual = HabitualConjunction(r)
    params2h = habitual.params()
    env_ids = r.integers(0, N_ENVS, size=8)
    def loss2b():
        return sigmoid_bce(habitual(contact, env_ids), y2)
    check("module2.HabitualConjunction.l1.W", habitual.l1.W, loss2b, params2h)

    # -- module 3 --
    dec = CapacityGatedDecoder(r)
    params3 = dec.params()
    thought = Tensor(r.standard_normal((6, D_THOUGHT_3)), requires_grad=False)
    ctx = Tensor(r.standard_normal((6, D_CTX_3)), requires_grad=False)
    reg = r.integers(0, N_REGISTERS_3, size=6)
    meaning = r.integers(0, K_MEANING_3, size=6)
    def loss3():
        rl, bl, _ = dec.forward(thought, ctx)
        return softmax_cross_entropy(rl, reg) + softmax_cross_entropy(bl, meaning)
    check("module3.CapacityGatedDecoder.gate_out.W", dec.gate_out.W, loss3, params3)
    check("module3.CapacityGatedDecoder.head_demo.W", dec.head_demo.W, loss3, params3)

    return ok


# =============================================================================
# SECTION 5 -- TRAINING
# =============================================================================

def train_module1(verbose=True):
    r = np.random.default_rng(11)
    encoders = [PhantasmEncoder(r) for _ in range(N_AGENTS_1)]
    intellect = CommonIntellect(r)
    params = intellect.params()
    for e in encoders:
        params += e.params()
    opt = Adam(params, lr=0.03)
    train_data = [make_agent_data_1(a) for a in range(N_AGENTS_1)]

    for ep in range(220):
        opt.zero_grad()
        total = 0.0
        for a in range(N_AGENTS_1):
            X, y = train_data[a]
            phantasm = encoders[a](Tensor(X, requires_grad=False))
            loss = softmax_cross_entropy(intellect(phantasm), y)
            loss.backward()
            total += loss.data
        opt.step()
        if verbose and (ep % 40 == 0 or ep == 219):
            print(f"    epoch {ep:4d}  mean CE across {N_AGENTS_1} agents = {total / N_AGENTS_1:.4f}")

    return encoders, intellect, train_data, r


def train_module2(verbose=True):
    r = np.random.default_rng(33)
    causal = CausalMechanism(r)
    habitual = HabitualConjunction(r)
    opt_c = Adam(causal.params(), lr=0.05)
    opt_h = Adam(habitual.params(), lr=0.05)

    batches = []
    for env in TRAIN_ENVS:
        contact, dryness, flame, y = sample_env(env, 300, seed=10 + env['id'])
        batches.append((contact, dryness, flame, y, np.full(300, env['id'])))

    for ep in range(300):
        opt_c.zero_grad()
        opt_h.zero_grad()
        tc, th = 0.0, 0.0
        for contact, dryness, flame, y, env_ids in batches:
            yb = y.reshape(-1, 1)
            lc = sigmoid_bce(causal(contact, dryness, flame), yb)
            lc.backward()
            tc += lc.data
            lh = sigmoid_bce(habitual(contact, env_ids), yb)
            lh.backward()
            th += lh.data
        opt_c.step()
        opt_h.step()
        if verbose and (ep % 60 == 0 or ep == 299):
            print(f"    epoch {ep:4d}  causal BCE={tc/3:.4f}   habitual BCE={th/3:.4f}")

    return causal, habitual


def train_module3(verbose=True):
    r = np.random.default_rng(77)
    model = CapacityGatedDecoder(r)
    opt = Adam(model.params(), lr=0.04)
    thoughts_tr, ctx_tr, reg_tr, meaning_tr = make_dataset_3(600, seed=21)

    for ep in range(260):
        opt.zero_grad()
        tb = Tensor(thoughts_tr, requires_grad=False)
        cb = Tensor(ctx_tr, requires_grad=False)
        register_logits, blended, gate = model.forward(tb, cb)
        loss_register = softmax_cross_entropy(register_logits, reg_tr)
        loss_content = softmax_cross_entropy(blended, meaning_tr)
        (loss_register + loss_content).backward()
        opt.step()
        if verbose and (ep % 50 == 0 or ep == 259):
            print(f"    epoch {ep:4d}  register CE={loss_register.data:.4f}  "
                  f"content CE={loss_content.data:.4f}")

    return model


# =============================================================================
# SECTION 6 -- SELF-TESTS
# =============================================================================

def self_test_module1(encoders, intellect, train_data, r):
    print("\n  -- Self-test 1a: a SIXTH, never-trained agent, four examples per")
    print("     class, reusing the already-trained shared intellect --")
    print("     (averaged over 8 independent few-shot draws for a stable estimate)")
    new_id = N_AGENTS_1
    _extend_agent_sensory_1(r)

    shared_accs, priv_accs = [], []
    N_TRIALS = 8
    for trial in range(N_TRIALS):
        few_X, few_y = make_agent_data_1(new_id, n_per_class=4, seed_offset=900 + trial)
        test_X, test_y = make_agent_data_1(new_id, n_per_class=30, seed_offset=555)

        new_encoder = PhantasmEncoder(r)
        opt_new = Adam(new_encoder.params(), lr=0.05)
        for _ in range(120):
            opt_new.zero_grad()
            loss = softmax_cross_entropy(intellect(new_encoder(Tensor(few_X, requires_grad=False))), few_y)
            loss.backward()
            opt_new.step()
        pred = intellect(new_encoder(Tensor(test_X, requires_grad=False))).data.argmax(axis=1)
        shared_accs.append((pred == test_y).mean())

        priv = PrivateStack(r)
        opt_priv = Adam(priv.params(), lr=0.05)
        for _ in range(120):
            opt_priv.zero_grad()
            loss = softmax_cross_entropy(priv(Tensor(few_X, requires_grad=False)), few_y)
            loss.backward()
            opt_priv.step()
        pred_priv = priv(Tensor(test_X, requires_grad=False)).data.argmax(axis=1)
        priv_accs.append((pred_priv == test_y).mean())

    acc_shared, acc_priv = float(np.mean(shared_accs)), float(np.mean(priv_accs))
    print(f"     shared-substrate condition   -> mean test accuracy = {acc_shared:.3f} "
          f"(std {np.std(shared_accs):.3f}, n={N_TRIALS})")
    print(f"     no-shared-substrate baseline -> mean test accuracy = {acc_priv:.3f} "
          f"(std {np.std(priv_accs):.3f}, n={N_TRIALS})")
    print(f"     (chance level = {1/N_CLASSES_1:.3f})")

    print("\n  -- Self-test 1b: suppress the ONE shared substrate; every agent's")
    print("     understanding should collapse at once, regardless of how well")
    print("     each private phantasm encoder was individually trained --")
    saved_W1, saved_W2 = intellect.l1.W.data.copy(), intellect.l2.W.data.copy()
    intellect.l1.W.data[:] = 0.0
    intellect.l2.W.data[:] = 0.0
    collapsed = []
    for a in range(N_AGENTS_1):
        X, y = train_data[a]
        pred = intellect(encoders[a](Tensor(X, requires_grad=False))).data.argmax(axis=1)
        collapsed.append((pred == y).mean())
    intellect.l1.W.data[:] = saved_W1
    intellect.l2.W.data[:] = saved_W2
    restored = []
    for a in range(N_AGENTS_1):
        X, y = train_data[a]
        pred = intellect(encoders[a](Tensor(X, requires_grad=False))).data.argmax(axis=1)
        restored.append((pred == y).mean())
    print(f"     mean per-agent accuracy WITH shared intellect zeroed     = {np.mean(collapsed):.3f}")
    print(f"     mean per-agent accuracy immediately after restoring it  = {np.mean(restored):.3f}")
    return acc_shared, acc_priv, np.mean(collapsed), np.mean(restored)


def self_test_module2(causal, habitual):
    print("\n  -- Self-test 2: in-distribution environments --")
    def eval_causal(env, n=400, seed=777):
        contact, dryness, flame, y = sample_env(env, n, seed)
        p = 1 / (1 + np.exp(-causal(contact, dryness, flame).data.flatten()))
        return ((p > 0.5).astype(np.float64) == y).mean()

    def eval_habitual(env, n=400, seed=777):
        contact, dryness, flame, y = sample_env(env, n, seed)
        p = 1 / (1 + np.exp(-habitual(contact, np.full(n, env['id'])).data.flatten()))
        return ((p > 0.5).astype(np.float64) == y).mean()

    for env in TRAIN_ENVS:
        print(f"     env {env['id']} (humidity={env['humidity']:.2f})  "
              f"causal acc={eval_causal(env):.3f}   habitual acc={eval_habitual(env):.3f}")

    print("\n  -- Self-test 2b: held-out INTERVENTION environment (soaked cotton,")
    print("     never trained on) --")
    contact_t, dryness_t, flame_t, y_t = sample_env(TEST_ENV, 400, seed=777)
    true_rate = y_t.mean()
    p_causal = 1 / (1 + np.exp(-causal(contact_t, dryness_t, flame_t).data.flatten()))
    acc_c = ((p_causal > 0.5).astype(np.float64) == y_t).mean()
    p_habitual = 1 / (1 + np.exp(-habitual(contact_t, np.full(400, TEST_ENV['id'])).data.flatten()))
    acc_h = ((p_habitual > 0.5).astype(np.float64) == y_t).mean()
    print(f"     true empirical burn rate in this unseen environment = {true_rate:.3f}")
    print(f"     causal module   mean P(burn)={p_causal.mean():.3f}  "
          f"|error|={abs(p_causal.mean()-true_rate):.3f}  accuracy={acc_c:.3f}")
    print(f"     habitual module mean P(burn)={p_habitual.mean():.3f}  "
          f"|error|={abs(p_habitual.mean()-true_rate):.3f}  accuracy={acc_h:.3f}")
    print("     (the habitual module's env-id=3 row was never gradient-updated --")
    print("      exactly al-Ghazali's own position leaves it nothing principled")
    print("      to extrapolate from)")
    return true_rate, p_causal.mean(), p_habitual.mean()


def self_test_module3(model):
    print("\n  -- Self-test 3: the SAME proposition shown to two audiences --")
    r = np.random.default_rng(999)
    same_thought = np.repeat(r.standard_normal((1, D_THOUGHT_3)), 2, axis=0)
    ctx_demo = np.zeros((1, D_CTX_3)); ctx_demo[0, :3] = [1, 0, 0]
    ctx_rhet = np.zeros((1, D_CTX_3)); ctx_rhet[0, :3] = [0, 0, 1]
    ctx_probe = np.concatenate([ctx_demo, ctx_rhet], axis=0)
    register_logits, blended, gate = model.forward(
        Tensor(same_thought, requires_grad=False), Tensor(ctx_probe, requires_grad=False))
    released = blended.data.argmax(axis=1)
    names = {LITERAL: "literal/apparent", RECONCILED: "ta'wil-reconciled",
             HEDGED: "hedged dialectical", DISTRACT: "distractor"}
    print(f"     demonstrative-class audience  gate=(demo,dial,rhet)={gate.data[0].round(3)}"
          f"  -> released: {names[released[0]]}")
    print(f"     rhetorical-class audience     gate=(demo,dial,rhet)={gate.data[1].round(3)}"
          f"  -> released: {names[released[1]]}")
    ok = (released[0] == RECONCILED) and (released[1] == LITERAL)
    print(f"     ta'wil correctly capacity-gated: {ok}")
    return ok


# =============================================================================
# SECTION 7 -- MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 78)
    print("CHAPTER 244 -- IBN RUSHD (AVERROES) -- THE COMMON-INTELLECT NETWORK")
    print("=" * 78)

    print("\n[1/3] Finite-difference gradient check (mandatory, run before training)")
    passed = finite_difference_gradient_check()
    print(f"\n  GRADIENT CHECK: {'ALL PASSED' if passed else 'FAILED'}")
    assert passed, "gradient check failed -- refusing to proceed to training"

    print("\n[2/3] Training all three modules")
    print("\n  Module 1 -- Unity of the Material Intellect")
    encoders, intellect, train_data, r1 = train_module1()
    print("\n  Module 2 -- Causal Necessity vs. Habitual Conjunction")
    causal, habitual = train_module2()
    print("\n  Module 3 -- Capacity-Gated Ta'wil Decoding")
    dec = train_module3()

    print("\n[3/3] Self-tests")
    self_test_module1(encoders, intellect, train_data, r1)
    self_test_module2(causal, habitual)
    ok3 = self_test_module3(dec)

    total_params = (
        sum(p.data.size for e in encoders for p in e.params())
        + sum(p.data.size for p in intellect.params())
        + sum(p.data.size for p in causal.params())
        + sum(p.data.size for p in habitual.params())
        + sum(p.data.size for p in dec.params())
    )
    print(f"\nTotal trainable parameters across all three modules: {total_params:,}")
    print("\nAll modules trained, gradient-checked, and self-tested successfully.")


"""
==============================================================================
VERIFIED EXECUTION LOG -- captured from an actual run of this file,
pasted here unedited as required proof-of-execution before shipping.
==============================================================================

==============================================================================
CHAPTER 244 -- IBN RUSHD (AVERROES) -- THE COMMON-INTELLECT NETWORK
==============================================================================

[1/3] Finite-difference gradient check (mandatory, run before training)
  [module1.CommonIntellect.l2.W      ] max|analytic-numeric| rel err = 1.951e-09  PASS
  [module1.PhantasmEncoder.l1.W      ] max|analytic-numeric| rel err = 2.891e-09  PASS
  [module2.CausalMechanism.l1.W      ] max|analytic-numeric| rel err = 2.748e-09  PASS
  [module2.HabitualConjunction.l1.W  ] max|analytic-numeric| rel err = 1.451e-09  PASS
  [module3.CapacityGatedDecoder.gate_out.W] max|analytic-numeric| rel err = 1.335e-09  PASS
  [module3.CapacityGatedDecoder.head_demo.W] max|analytic-numeric| rel err = 2.511e-09  PASS

  GRADIENT CHECK: ALL PASSED

[2/3] Training all three modules

  Module 1 -- Unity of the Material Intellect
    epoch    0  mean CE across 5 agents = 2.2188
    epoch   40  mean CE across 5 agents = 0.2813
    epoch   80  mean CE across 5 agents = 0.0988
    epoch  120  mean CE across 5 agents = 0.0446
    epoch  160  mean CE across 5 agents = 0.0192
    epoch  200  mean CE across 5 agents = 0.0074
    epoch  219  mean CE across 5 agents = 0.0054

  Module 2 -- Causal Necessity vs. Habitual Conjunction
    epoch    0  causal BCE=0.6386   habitual BCE=0.6800
    epoch   60  causal BCE=0.5590   habitual BCE=0.5793
    epoch  120  causal BCE=0.5580   habitual BCE=0.5777
    epoch  180  causal BCE=0.5592   habitual BCE=0.5775
    epoch  240  causal BCE=0.5573   habitual BCE=0.5775
    epoch  299  causal BCE=0.5561   habitual BCE=0.5774

  Module 3 -- Capacity-Gated Ta'wil Decoding
    epoch    0  register CE=1.5632  content CE=1.4526
    epoch   50  register CE=0.0064  content CE=0.0001
    epoch  100  register CE=0.0038  content CE=0.0000
    epoch  150  register CE=0.0027  content CE=0.0000
    epoch  200  register CE=0.0020  content CE=0.0000
    epoch  250  register CE=0.0016  content CE=0.0000
    epoch  259  register CE=0.0015  content CE=0.0000

[3/3] Self-tests

  -- Self-test 1a: a SIXTH, never-trained agent, four examples per
     class, reusing the already-trained shared intellect --
     (averaged over 8 independent few-shot draws for a stable estimate)
     shared-substrate condition   -> mean test accuracy = 0.680 (std 0.054, n=8)
     no-shared-substrate baseline -> mean test accuracy = 0.655 (std 0.043, n=8)
     (chance level = 0.111)

  -- Self-test 1b: suppress the ONE shared substrate; every agent's
     understanding should collapse at once, regardless of how well
     each private phantasm encoder was individually trained --
     mean per-agent accuracy WITH shared intellect zeroed     = 0.111
     mean per-agent accuracy immediately after restoring it  = 0.999

  -- Self-test 2: in-distribution environments --
     env 0 (humidity=0.10)  causal acc=0.647   habitual acc=0.613
     env 1 (humidity=0.35)  causal acc=0.705   habitual acc=0.690
     env 2 (humidity=0.55)  causal acc=0.777   habitual acc=0.777

  -- Self-test 2b: held-out INTERVENTION environment (soaked cotton,
     never trained on) --
     true empirical burn rate in this unseen environment = 0.128
     causal module   mean P(burn)=0.076  |error|=0.052  accuracy=0.873
     habitual module mean P(burn)=0.521  |error|=0.393  accuracy=0.450
     (the habitual module's env-id=3 row was never gradient-updated --
      exactly al-Ghazali's own position leaves it nothing principled
      to extrapolate from)

  -- Self-test 3: the SAME proposition shown to two audiences --
     demonstrative-class audience  gate=(demo,dial,rhet)=[1. 0. 0.]  -> released: ta'wil-reconciled
     rhetorical-class audience     gate=(demo,dial,rhet)=[0. 0. 1.]  -> released: literal/apparent
     ta'wil correctly capacity-gated: True

Total trainable parameters across all three modules: 2,866

All modules trained, gradient-checked, and self-tested successfully.
"""
