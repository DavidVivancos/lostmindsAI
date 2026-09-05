"""
================================================================================
Chapter 0149_st_jerome_347 - St. Jerome (347-420 CE)
The Hieronymian Collation Engine: an AGI core built from a translator's mind
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 8 Minds 141 - 160 Available on Amazon https://www.amazon.com/dp/B0HH8RTCXF
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 149: St. Jerome (347-420 CE)
================================================================================  

WHY THIS ARCHITECTURE (and why it is NOT a Transformer)
--------------------------------------------------------------------------------
Jerome of Stridon gave the Latin West its Bible. He did not do it by inventing
scripture; he did it by *collating* — laying corrupted copies side by side,
reconstructing the true reading, and carrying its MEANING (not its surface
letters) across a language boundary. Three convictions organised his whole mind,
and each becomes a mechanism in this model:

  1. HEBRAICA VERITAS ("the Hebrew truth"). Jerome distrusted the copy-of-a-copy.
     Where his contemporaries revered the Greek Septuagint, he went back to the
     Hebrew source and, in his prefaces (e.g. to Job), complained of what was
     "obscure ... omitted ... or corrupted by copyists." In a modern system this
     is grounding against a source of truth and robust reconstruction from noisy
     witnesses — the opposite of training on your own derivative output.
        -> MECHANISM: a COLLATION layer. Several corrupted "witnesses" of one
           source are aligned; an iterative reweighting gives more authority to
           readings that agree with the emerging archetype and quarantines the
           corrupt ones. This is a differentiable stemmatics / robust mean.

  2. SENSUM DE SENSU ("sense for sense, not word for word"), from his Letter 57
     to Pammachius. Meaning must be transferred at the level of sense, not tokens.
        -> MECHANISM: a narrow SENSE BOTTLENECK the reconstructed reading must
           pass through before it can be re-expressed. Translation is forced to
           route through meaning.

  3. ...BUT SOME TEXT IS SACRED, "where even the order of the words is a mystery."
     Jerome held BOTH methods and knew WHEN to use each: free for ordinary prose,
     literal for Scripture. That meta-decision is the heart of his craft.
        -> MECHANISM: a VERBUM/SENSUM ROUTER. A per-position gate chooses between
           a literal channel (word-for-word) and the sense channel. The model
           learns, from context, when the letter itself is load-bearing.

A fourth idea haunts him — the dream of Letter 22 ("Ciceronianus es, non
Christianus": you are a Ciceronian, not a Christian). The tools of a mind can be
captured by the wrong master. We do not train that as a loss here, but the
architecture keeps capability (the channels) separable from allegiance (the
router / the grounding target), which is exactly where a Jeromean alignment
argument would attach. See the chapter for the full treatment.

WHAT THE FILE DOES
--------------------------------------------------------------------------------
Pure NumPy, from scratch: a tiny reverse-mode autodiff, a synthetic bilingual
"manuscript" corpus with deliberately corrupted witnesses, the collation engine,
a finite-difference gradient check (mandatory), a real training loop, and
self-tests. Run it directly:  python3 chapter_0149_st_jerome_347.py
"""

import numpy as np

# =============================================================================
# 1. AUTOGRAD CORE  — a minimal reverse-mode tensor (no torch/tf/jax)
#    Every op records how to push gradients back to its parents; .backward()
#    walks the graph in reverse topological order. This is the "scriptorium":
#    the machinery every higher layer is copied out of.
# =============================================================================
class T:
    def __init__(self, data, parents=(), backward=None):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._parents = parents
        self._backward = backward or (lambda: None)

    @staticmethod
    def _unbroadcast(grad, shape):
        """Undo NumPy broadcasting so a parent receives grad in its own shape."""
        while grad.ndim > len(shape):
            grad = grad.sum(axis=0)
        for i, s in enumerate(shape):
            if s == 1 and grad.shape[i] != 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad.reshape(shape)

    def __add__(self, o):
        o = o if isinstance(o, T) else T(o)
        out = T(self.data + o.data, (self, o))
        def bw():
            self.grad += T._unbroadcast(out.grad, self.data.shape)
            o.grad += T._unbroadcast(out.grad, o.data.shape)
        out._backward = bw; return out

    def __mul__(self, o):
        o = o if isinstance(o, T) else T(o)
        out = T(self.data * o.data, (self, o))
        def bw():
            self.grad += T._unbroadcast(out.grad * o.data, self.data.shape)
            o.grad += T._unbroadcast(out.grad * self.data, o.data.shape)
        out._backward = bw; return out

    def __sub__(self, o):
        o = o if isinstance(o, T) else T(o)
        return self + (o * -1.0)

    def matmul(self, o):
        out = T(self.data @ o.data, (self, o))
        def bw():
            self.grad += out.grad @ o.data.T
            o.grad += self.data.T @ out.grad
        out._backward = bw; return out

    def sum(self, axis=None, keepdims=False):
        out = T(self.data.sum(axis=axis, keepdims=keepdims), (self,))
        def bw():
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self.grad += np.ones_like(self.data) * g
        out._backward = bw; return out

    def reshape(self, *shape):
        out = T(self.data.reshape(*shape), (self,))
        def bw(): self.grad += out.grad.reshape(self.data.shape)
        out._backward = bw; return out

    def tanh(self):
        t = np.tanh(self.data); out = T(t, (self,))
        def bw(): self.grad += (1 - t * t) * out.grad
        out._backward = bw; return out

    def sigmoid(self):
        s = 1.0 / (1.0 + np.exp(-self.data)); out = T(s, (self,))
        def bw(): self.grad += s * (1 - s) * out.grad
        out._backward = bw; return out

    def exp(self):
        e = np.exp(self.data); out = T(e, (self,))
        def bw(): self.grad += e * out.grad
        out._backward = bw; return out

    def softplus(self):
        out = T(np.logaddexp(0.0, self.data), (self,))
        def bw(): self.grad += (1.0 / (1.0 + np.exp(-self.data))) * out.grad
        out._backward = bw; return out

    def recip(self):
        out = T(1.0 / self.data, (self,))
        def bw(): self.grad += (-1.0 / (self.data ** 2)) * out.grad
        out._backward = bw; return out

    def backward(self):
        topo, seen = [], set()
        def build(v):
            if id(v) in seen: return
            seen.add(id(v))
            for p in v._parents: build(p)
            topo.append(v)
        build(self)
        self.grad = np.ones_like(self.data)
        for v in reversed(topo): v._backward()


def softmax(x: T, axis):
    """Numerically-stable softmax along one axis (used to weight witnesses)."""
    m = x.data.max(axis=axis, keepdims=True)
    e = (x + T(-m)).exp()
    return e * e.sum(axis=axis, keepdims=True).recip()


def softmax_ce(logits: T, target):
    """Fused softmax cross-entropy over rows; returns (scalar_loss, probs)."""
    z = logits.data - logits.data.max(axis=1, keepdims=True)
    e = np.exp(z); p = e / e.sum(axis=1, keepdims=True)
    N = z.shape[0]
    loss = -np.mean(np.log(p[np.arange(N), target] + 1e-12))
    out = T(loss, (logits,))
    def bw():
        d = p.copy(); d[np.arange(N), target] -= 1.0; d /= N
        logits.grad += d * out.grad
    out._backward = bw; return out, p


# =============================================================================
# 2. THE MANUSCRIPT CORPUS
#    A hidden "concept" sequence is the true meaning. The SOURCE language writes
#    each concept through one codebook. Each WITNESS is a copy of the source with
#    random scribal corruptions. The TARGET language is produced two ways:
#       - "sacred" positions (a fixed register): word-for-word — target token
#         equals the source token, so the literal channel should win.
#       - "profane" positions: sense-for-sense — target is a DIFFERENT codebook
#         of the same concept, recoverable only by passing through meaning.
#    Restoration must denoise witnesses back to the true source (Hebraica veritas);
#    translation must render sense across the boundary (sensum de sensu); the
#    router must learn which register each position belongs to.
# =============================================================================
class Corpus:
    def __init__(self, seed=0, C=5, L=6, K=4, p_corrupt=0.25):
        self.rng = np.random.default_rng(seed)
        self.C, self.L, self.K, self.p = C, L, K, p_corrupt
        self.Vsrc = C; self.Vtgt = C
        self.perm_src = self.rng.permutation(C)          # concept -> source token
        self.perm_tgt = self.rng.permutation(C)          # concept -> free target token
        self.sacred = (np.arange(L) < L // 2).astype(int) # first half = literal register

    def sample(self, n):
        exs = []
        for _ in range(n):
            c = self.rng.integers(0, self.C, size=self.L)       # hidden meaning
            src = self.perm_src[c]                               # canonical source text
            tgt = np.where(self.sacred == 1, self.perm_src[c], self.perm_tgt[c])
            W = np.tile(src, (self.K, 1)).copy()                 # K witnesses
            for k in range(self.K):
                for i in range(self.L):
                    if self.rng.random() < self.p:              # scribal corruption
                        W[k, i] = self.rng.integers(0, self.Vsrc)
            exs.append((W, src, tgt))
        return exs


# =============================================================================
# 3. THE HIERONYMIAN COLLATION ENGINE
# =============================================================================
class Hieronymus:
    def __init__(self, C, L, K, d=16, b=8, iters=3, seed=1):
        r = np.random.default_rng(seed)
        g = lambda *s: T(r.standard_normal(s) * 0.2)
        z = lambda *s: T(np.zeros(s))
        self.L, self.K, self.d, self.b, self.iters = L, K, d, b, iters
        self.Vs, self.Vt = C, C
        self.E    = g(C, d)                       # shared token embedding (one hand)
        self.P    = g(1, L, d)                    # positional embedding
        self.W1   = g(d, d); self.b1 = z(1, d)    # per-position encoder -> "reading"
        self.alpha = T(np.array([[0.5]]))         # collation sharpness (softplus'd)
        self.auth = z(K, 1, 1)                    # per-witness authority prior
        self.Wd   = g(d, b); self.bd = z(1, b)    # sense bottleneck (down): the sensum
        self.Wu   = g(b, d); self.bu = z(1, d)    # sense bottleneck (up)
        self.Wres = g(d, C)                       # restoration head (Hebraica veritas)
        self.Wts  = g(d, C)                       # sense-for-sense translation head
        self.Wlit = g(d, C)                       # word-for-word (literal) channel
        self.Wg   = g(d, 1); self.Wgp = g(d, 1); self.bg = z(1, 1)  # verbum/sensum router

    def params(self):
        return [self.E, self.P, self.W1, self.b1, self.alpha, self.auth, self.Wd,
                self.bd, self.Wu, self.bu, self.Wres, self.Wts, self.Wlit,
                self.Wg, self.Wgp, self.bg]

    def _onehot(self, idx, V):
        o = np.zeros((idx.shape[0], V)); o[np.arange(idx.shape[0]), idx] = 1.0
        return o

    def forward(self, W):
        K, L, d = self.K, self.L, self.d
        # (a) embed each witness token and add its position
        oh = self._onehot(W.reshape(-1), self.Vs)                       # [K*L, Vs]
        emb = T(oh).matmul(self.E).reshape(K, L, d) + self.P            # [K, L, d]
        # (b) per-position encoder produces one "reading" per witness
        R = (emb.reshape(K * L, d).matmul(self.W1) + self.b1).tanh().reshape(K, L, d)
        # (c) COLLATION: iterative reweighting toward the emerging archetype.
        #     Readings close to the consensus gain authority; outliers (corruptions)
        #     are progressively quarantined. This is the Hebraica-veritas move.
        w = T(np.ones((K, 1, 1)) / K)
        neg_a = self.alpha.softplus() * -1.0
        for _ in range(self.iters):
            mu = (w * R).sum(axis=0, keepdims=True)                     # [1, L, d]
            diff = R - mu
            dist = (diff * diff).sum(axis=2, keepdims=True)             # [K, L, 1]
            w = softmax(dist * neg_a + self.auth, axis=0)               # [K, L, 1]
        mu = (w * R).sum(axis=0, keepdims=True).reshape(L, d)           # the archetype
        # (d) SENSE BOTTLENECK: meaning must pass through the narrow sensum
        s  = (mu.matmul(self.Wd) + self.bd).tanh()                      # [L, b]
        up = (s.matmul(self.Wu) + self.bu).tanh()                       # [L, d]
        # (e) restoration of the true source (textual criticism / grounding)
        res = up.matmul(self.Wres)
        # (f) TRANSLATION with the verbum/sensum router
        sense = up.matmul(self.Wts)                                     # free rendering
        lit   = mu.matmul(self.Wlit)                                    # literal rendering
        gate  = (mu.matmul(self.Wg)
                 + self.P.reshape(L, d).matmul(self.Wgp) + self.bg).sigmoid()
        trans = lit * gate + sense * (T(np.ones((L, 1))) - gate)
        return res, trans, gate, w

    def loss(self, ex, router_w=1.0):
        W, src, tgt = ex
        res, trans, gate, wgt = self.forward(W)
        Lres, _ = softmax_ce(res, src)              # reconstruct the true source
        Ltr,  _ = softmax_ce(trans, tgt)            # render the sense in the target
        sacred = (np.arange(self.L) < self.L // 2).astype(float).reshape(self.L, 1)
        rl = ((gate - T(sacred)) * (gate - T(sacred))).sum() * (router_w / self.L)
        return Lres + Ltr + rl


def zero_grads(model):
    for p in model.params():
        p.grad = np.zeros_like(p.data)


# =============================================================================
# 4. GRADIENT CHECK  (mandatory) — autodiff vs central finite differences
# =============================================================================
def gradient_check(model, batch, param, name, eps=1e-6):
    def total():
        t = model.loss(batch[0])
        for e in batch[1:]:
            t = t + model.loss(e)
        return t
    zero_grads(model); L = total(); L.backward()
    ana = param.grad.copy(); num = np.zeros_like(param.data)
    for idx in np.ndindex(*param.data.shape):
        old = param.data[idx]
        param.data[idx] = old + eps; Lp = total().data
        param.data[idx] = old - eps; Lm = total().data
        param.data[idx] = old
        num[idx] = (Lp - Lm) / (2 * eps)
    diff = float(np.max(np.abs(ana - num)))
    print(f"  grad-check [{name:>4}]  max|analytic - numeric| = {diff:.2e}"
          f"   ->  {'PASS' if diff < 1e-5 else 'FAIL'}")
    return diff


# =============================================================================
# 5. TRAIN, EVALUATE, DEMONSTRATE
# =============================================================================
def main():
    C, L, K = 5, 6, 4
    corp = Corpus(seed=2, C=C, L=L, K=K)
    model = Hieronymus(C, L, K, seed=3)

    print("=" * 74)
    print("Gradient check (finite differences)")
    print("=" * 74)
    batch = corp.sample(2)
    d1 = gradient_check(model, batch, model.W1, "W1")
    d2 = gradient_check(model, batch, model.Wd, "Wd")
    d3 = gradient_check(model, batch, model.auth, "auth")
    assert max(d1, d2, d3) < 1e-5, "gradient check failed"

    print("\n" + "=" * 74)
    print("Training the collation engine")
    print("=" * 74)
    train = corp.sample(320); B = 16; base = 0.06
    vel = [np.zeros_like(p.data) for p in model.params()]
    for epoch in range(70):
        corp.rng.shuffle(train)
        lr = base * (0.5 ** (epoch / 22.0)); tot = 0.0; nb = 0
        for i in range(0, len(train), B):
            chunk = train[i:i + B]; zero_grads(model)
            l = model.loss(chunk[0])
            for e in chunk[1:]:
                l = l + model.loss(e)
            l = l * (1.0 / len(chunk)); l.backward()
            for j, prm in enumerate(model.params()):
                vel[j] = 0.85 * vel[j] - lr * prm.grad
                prm.data += vel[j]
            tot += l.data; nb += 1
        if epoch % 10 == 0 or epoch == 69:
            print(f"  epoch {epoch:2d}   loss {tot / nb:.4f}   lr {lr:.4f}")

    print("\n" + "=" * 74)
    print("Evaluation (300 fresh manuscripts)")
    print("=" * 74)
    test = corp.sample(300)
    rc = tc = rt = n = 0; w_clean, w_corr = [], []
    for ex in test:
        W, src, tgt = ex
        res, trans, gate, wgt = model.forward(W)
        rc += (res.data.argmax(1) == src).sum()
        tc += (trans.data.argmax(1) == tgt).sum()
        rt += ((gate.data.reshape(-1) > 0.5).astype(int) == corp.sacred).sum()
        n += L
        wa = wgt.data.reshape(K, L)
        for k in range(K):
            for i in range(L):
                (w_corr if W[k, i] != src[i] else w_clean).append(wa[k, i])
    print(f"  restoration accuracy (Hebraica veritas) : {rc / n:.3f}")
    print(f"  translation accuracy (sensum de sensu)  : {tc / n:.3f}")
    print(f"  router accuracy (verbum vs sensum)      : {rt / n:.3f}")
    print(f"  mean authority  clean witnesses         : {np.mean(w_clean):.3f}")
    print(f"  mean authority  corrupt witnesses       : {np.mean(w_corr):.3f}")

    print("\n" + "=" * 74)
    print("One worked example — the engine collating and translating")
    print("=" * 74)
    W, src, tgt = test[0]
    res, trans, gate, wgt = model.forward(W)
    reg = ["sacred" if s else "profane" for s in corp.sacred]
    method = ["word-for-word" if g > 0.5 else "sense-for-sense"
              for g in gate.data.reshape(-1)]
    print("  witnesses (corrupt copies) :")
    for k in range(K):
        print(f"     scribe {k}:  {W[k].tolist()}")
    print(f"  true source                : {src.tolist()}")
    print(f"  restored source            : {res.data.argmax(1).tolist()}")
    print(f"  target (gold)              : {tgt.tolist()}")
    print(f"  produced translation       : {trans.data.argmax(1).tolist()}")
    print(f"  register                   : {reg}")
    print(f"  router chose               : {method}")

    # ---- self-tests -------------------------------------------------------
    assert rc / n > 0.85, "restoration too weak"
    assert tc / n > 0.85, "translation too weak"
    assert rt / n > 0.90, "router did not learn the dual method"
    assert np.mean(w_corr) < np.mean(w_clean), "collation did not quarantine corruption"
    print("\nAll self-tests passed.")


if __name__ == "__main__":
    main()
