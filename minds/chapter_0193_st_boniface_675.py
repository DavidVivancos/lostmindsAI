# =============================================================================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 10 Minds 181 - 200 Available on Amazon https://www.amazon.com/dp/B0HJYMZ3G6
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0193_st_boniface_675 - St. Boniface (Wynfrith of Wessex), c. 675 – 754 CE
# =============================================================================
#  Encyclopedia of Lost Minds — Echoes on AI
#  Chapter 193 · St. Boniface (Wynfrith of Wessex), c. 675 – 754 CE
#
#  "The Orthography of Grace": a from-scratch neural architecture that encodes
#  the ONE cognitive idea that is Boniface's alone.
#
#  ---------------------------------------------------------------------------
#  WHY THIS ARCHITECTURE (the mind -> mechanism mapping)
#  ---------------------------------------------------------------------------
#  Boniface was not, first, a tree-feller. He was a GRAMMARIAN. Before he ever
#  left England he wrote a Latin grammar (the Ars Bonifacii), a treatise on
#  verse, and — decisively for us — twenty ACROSTIC riddles on the Virtues and
#  Vices, in which the SOLUTION is spelled vertically down the initial letters
#  of the lines. An acrostic carries its meaning in TWO independent channels at
#  once: the horizontal surface (the riddle's body — expressive, but where a
#  scribe's slip can corrupt a word) and the vertical acrostic (a compact,
#  structurally protected code that names the answer). In a surviving copy of
#  his riddle on Virginity a scribe swapped two words so the acrostic reads
#  "PIRGINITAS" instead of "VIRGINITAS" — a glaring, four-line-tall, never-
#  corrected error — yet the intended reference is still perfectly recoverable.
#
#  The SAME structure governs the most revealing episode in his letters. A
#  priest, ignorant of Latin, had been baptizing "in nomine patria et filia"
#  (garbled case-endings). Boniface the grammarian wanted the baptisms declared
#  void and repeated. Rome (Pope Zachary, Denzinger 297) OVERRULED him: because
#  the priest introduced "not an error or a heresy, but only ignorance of the
#  Roman tongue," the reference to the Trinity survived the broken form, so the
#  sacrament was VALID. Reference survives corrupted surface — UNLESS an
#  adversary has inverted the meaning behind a valid-looking surface. That last
#  clause is the heretic Aldebert, who wore an orthodox surface (crosses,
#  blessings, "relics") while inverting the reference (self-authorization, a
#  forged "letter from heaven," his own hair and nails passed off as relics).
#  The synod even split intent from surface: the deluded Aldebert (a "lunatic")
#  was offered repentance; Clemens, who KNOWINGLY held his errors, got the
#  harsher sentence. Innocent noise vs. adversarial inversion — judged by
#  intent, not by the surface.
#
#  Finally, the oak. Boniface felled Donar's sacred oak at Geismar and BUILT A
#  CHAPEL FROM ITS TIMBER. He did not delete the substrate; he re-loaded the
#  falsified structure's material under a new root.
#
#  This file turns those four documented facts into one learnable network:
#
#    (1) DUAL-CHANNEL REDUNDANT ENCODING. Every input is read by a SURFACE
#        encoder (expressive, corruption-prone) and an ACROSTIC encoder
#        (compact, protected). Each independently proposes a referent.
#    (2) CROSS-CHANNEL RECONCILIATION. When the channels agree, both speak;
#        when they disagree, the protected acrostic channel pulls the answer
#        back — recovering the true reference from a corrupted surface, exactly
#        as "VIRGINITAS" is recovered from "PIRGINITAS."
#    (3) VALIDITY ADJUDICATION (the Zachary rule). A gate learns to ACCEPT
#        innocent corruption (channels reconcile to one referent) and REJECT
#        adversarial inversion (a confident surface that DISAGREES with the
#        acrostic) — separating noise from a forger.
#    (4) GUARDED GRAFT-NOT-DELETE (the oak's wood -> the chapel, tempered by
#        Rome). Falsified/dead prototypes are not zeroed; their material is
#        re-loaded as a redundant witness for a needy referent — but a graft is
#        KEPT only if it does not increase held-out loss, else reverted. Reuse
#        is admitted only when it does not break what already works.
#
#  Engineering convention for this corpus: pure NumPy, built from scratch on a
#  tiny reverse-mode autodiff engine; a mandatory finite-difference gradient
#  check; a real training loop; an ablation; and self-tests. No PyTorch, no
#  attention-over-stored-keys, no Transformer. Run: `python3 <thisfile>`.
# =============================================================================

import numpy as np


# =============================================================================
#  PART 1 — A tiny reverse-mode autodiff engine over NumPy 2-D arrays.
#  (We build the calculus ourselves so the gradient check is a real check of
#   OUR math, not of a framework's.)
# =============================================================================

def _unbroadcast(grad, shape):
    """Sum `grad` back down to `shape` — the reverse of NumPy broadcasting."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad.reshape(shape)


class T:
    """A minimal tensor node: holds data, accumulates grad, remembers how it
    was produced so `backward()` can walk the graph in reverse."""
    __slots__ = ("data", "grad", "_backward", "_prev")

    def __init__(self, data, _children=()):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(_children)

    @property
    def shape(self):
        return self.data.shape

    # ---- elementwise algebra (with broadcasting) ----------------------------
    def __add__(self, other):
        other = other if isinstance(other, T) else T(other)
        out = T(self.data + other.data, (self, other))

        def _bw():
            self.grad += _unbroadcast(out.grad, self.data.shape)
            other.grad += _unbroadcast(out.grad, other.data.shape)
        out._backward = _bw
        return out

    def __mul__(self, other):
        other = other if isinstance(other, T) else T(other)
        out = T(self.data * other.data, (self, other))

        def _bw():
            self.grad += _unbroadcast(other.data * out.grad, self.data.shape)
            other.grad += _unbroadcast(self.data * out.grad, other.data.shape)
        out._backward = _bw
        return out

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        other = other if isinstance(other, T) else T(other)
        return self + (-other)

    def matmul(self, other):
        """2-D matrix product (batch,in)@(in,out) -> (batch,out)."""
        out = T(self.data @ other.data, (self, other))

        def _bw():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        out._backward = _bw
        return out

    # ---- unary maps ---------------------------------------------------------
    def tanh(self):
        t = np.tanh(self.data)
        out = T(t, (self,))

        def _bw():
            self.grad += (1.0 - t * t) * out.grad
        out._backward = _bw
        return out

    def exp(self):
        e = np.exp(self.data)
        out = T(e, (self,))

        def _bw():
            self.grad += e * out.grad
        out._backward = _bw
        return out

    def log(self):
        out = T(np.log(self.data), (self,))

        def _bw():
            self.grad += (1.0 / self.data) * out.grad
        out._backward = _bw
        return out

    def sqrt(self):
        s = np.sqrt(self.data)
        out = T(s, (self,))

        def _bw():
            self.grad += (0.5 / (s + 1e-30)) * out.grad
        out._backward = _bw
        return out

    def recip(self):
        r = 1.0 / self.data
        out = T(r, (self,))

        def _bw():
            self.grad += (-r * r) * out.grad
        out._backward = _bw
        return out

    def __truediv__(self, other):
        other = other if isinstance(other, T) else T(other)
        return self * other.recip()

    def sigmoid(self):
        s = 1.0 / (1.0 + np.exp(-self.data))
        out = T(s, (self,))

        def _bw():
            self.grad += s * (1.0 - s) * out.grad
        out._backward = _bw
        return out

    # ---- reductions & selection --------------------------------------------
    def sum(self, axis=None, keepdims=False):
        out = T(self.data.sum(axis=axis, keepdims=keepdims), (self,))

        def _bw():
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self.grad += np.ones_like(self.data) * g
        out._backward = _bw
        return out

    def gather(self, idx):
        """Pick one column per row: out[i] = data[i, idx[i]] -> shape (B,1).
        Used to read off the true-class probability for cross-entropy."""
        idx = np.asarray(idx, dtype=int)
        sel = self.data[np.arange(self.data.shape[0]), idx].reshape(-1, 1)
        out = T(sel, (self,))

        def _bw():
            g = np.zeros_like(self.data)
            g[np.arange(self.data.shape[0]), idx] = out.grad[:, 0]
            self.grad += g
        out._backward = _bw
        return out

    def backward(self):
        """Topologically sort the graph and push gradients from this scalar."""
        topo, seen = [], set()

        def build(v):
            if v not in seen:
                seen.add(v)
                for c in v._prev:
                    build(c)
                topo.append(v)
        build(self)
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()


def softmax(t):
    """Row-wise softmax over a (batch, M) tensor."""
    e = t.exp()
    return e / e.sum(axis=1, keepdims=True)


def concat_cols(cols):
    """Concatenate a list of (B,1)/(B,k) tensors along the feature axis."""
    data = np.concatenate([c.data for c in cols], axis=1)
    out = T(data, tuple(cols))
    widths = [c.data.shape[1] for c in cols]

    def _bw():
        i = 0
        for c, w in zip(cols, widths):
            c.grad += out.grad[:, i:i + w]
            i += w
    out._backward = _bw
    return out


# =============================================================================
#  PART 2 — The model parameters and the forward pass.
#
#  Two encoders (SURFACE + ACROSTIC), each reading the whole input; each with M
#  learned "prototypes" (candidate witnesses). A fixed group matrix G maps the
#  M prototypes to K referent classes, allowing MANY prototypes to attest to
#  ONE referent (the mechanism grafting will exploit). Spare prototypes start
#  attached to no class (the "dead wood" awaiting reuse).
# =============================================================================

class Params:
    def __init__(self, rng, D_in, K, M, h_s, d_s, h_a, d_a, h_v):
        def he(shape, fan_in):                       # He-scaled init
            return T(rng.standard_normal(shape) * np.sqrt(2.0 / fan_in))
        # SURFACE channel: expressive (larger), corruption-prone at input time
        self.Ws1 = he((D_in, h_s), D_in); self.bs1 = T(np.zeros(h_s))
        self.Ws2 = he((h_s, d_s), h_s);   self.bs2 = T(np.zeros(d_s))
        # ACROSTIC channel: compact (smaller latent) — the protected code
        self.Wa1 = he((D_in, h_a), D_in); self.ba1 = T(np.zeros(h_a))
        self.Wa2 = he((h_a, d_a), h_a);   self.ba2 = T(np.zeros(d_a))
        # Prototype readouts (M "witnesses" per channel)
        self.Us = he((d_s, M), d_s);      self.cs = T(np.zeros(M))
        self.Ua = he((d_a, M), d_a);      self.ca = T(np.zeros(M))
        # Validity adjudicator: 4 cross-channel features -> hidden -> scalar
        self.V1 = he((4, h_v), 4);        self.bv1 = T(np.zeros(h_v))
        self.V2 = he((h_v, 1), h_v);      self.bv2 = T(np.zeros(1))

    def all(self):
        return [self.Ws1, self.bs1, self.Ws2, self.bs2, self.Wa1, self.ba1,
                self.Wa2, self.ba2, self.Us, self.cs, self.Ua, self.ca,
                self.V1, self.bv1, self.V2, self.bv2]


def forward(P, X, G):
    """Full forward pass. Returns per-class distributions for the surface,
    acrostic, and reconciled ("fused") readings, plus the validity gate."""
    x = T(X); Gm = T(G)
    # --- surface reading (the riddle body / the spoken Latin) ---------------
    hs = (x.matmul(P.Ws1) + P.bs1).tanh()
    s = (hs.matmul(P.Ws2) + P.bs2).tanh()
    ls = s.matmul(P.Us) + P.cs                      # (B,M) surface logits
    # --- acrostic reading (the protected vertical code) ---------------------
    ha = (x.matmul(P.Wa1) + P.ba1).tanh()
    a = (ha.matmul(P.Wa2) + P.ba2).tanh()
    la = a.matmul(P.Ua) + P.ca                      # (B,M) acrostic logits
    # --- prototypes -> class distributions via the group matrix -------------
    ps_m = softmax(ls); pa_m = softmax(la)
    ps_k = ps_m.matmul(Gm)                          # (B,K) surface-by-class
    pa_k = pa_m.matmul(Gm)                          # (B,K) acrostic-by-class
    # --- cross-channel agreement features -----------------------------------
    alpha = (ps_k * pa_k).sqrt().sum(axis=1, keepdims=True)   # Bhattacharyya
    overlap = (ps_k * pa_k).sum(axis=1, keepdims=True)        # confident accord
    pur_s = (ps_k * ps_k).sum(axis=1, keepdims=True)          # surface sharpness
    pur_a = (pa_k * pa_k).sum(axis=1, keepdims=True)          # acrostic sharpness
    # --- reconciliation: acrostic speaks fully; surface weighted by accord ---
    #     (disagreement -> surface suppressed -> acrostic recovers the referent)
    zf = (ls * alpha) + la
    pf_m = softmax(zf); pf_k = pf_m.matmul(Gm)
    # --- validity gate (the Zachary rule): accept innocent noise, reject a
    #     confident-but-disagreeing surface (an adversary's inversion) --------
    phi = concat_cols([alpha, overlap, pur_s, pur_a])
    vh = (phi.matmul(P.V1) + P.bv1).tanh()
    v = (vh.matmul(P.V2) + P.bv2).sigmoid()
    return dict(ps_k=ps_k, pa_k=pa_k, pf_k=pf_k, v=v, alpha=alpha,
                pf_m=pf_m, ps_m=ps_m)


def make_group_matrix(proto_class, K):
    """(M,K) 0/1 matrix: row j has a 1 in the column of prototype j's class.
    A prototype whose class is -1 ('dead wood') maps to NO column."""
    M = len(proto_class)
    G = np.zeros((M, K))
    for j, c in enumerate(proto_class):
        if c >= 0:
            G[j, c] = 1.0
    return G


# =============================================================================
#  PART 3 — Loss. Four terms, each with a Bonifatian job:
#    CE(fused)   : recover the TRUE reference even under corruption/attack.
#    CE(acrostic): the protected channel must name the referent on its own.
#    CE(surface) : on CLEAN inputs only, learn the canonical surface->referent.
#    BCE(gate)   : accept innocent corruption, reject adversarial inversion.
# =============================================================================

def loss_fn(P, X, y, accept, clean_mask, G, lam_a=1.0, lam_s=0.5, lam_v=1.0):
    out = forward(P, X, G)
    y = np.asarray(y, int); B = X.shape[0]; eps = 1e-9
    ce_f = (out["pf_k"].gather(y) + eps).log() * -1.0
    ce_a = (out["pa_k"].gather(y) + eps).log() * -1.0
    ce_s = (out["ps_k"].gather(y) + eps).log() * -1.0
    cm = T(np.asarray(clean_mask, float).reshape(-1, 1))
    acc = T(np.asarray(accept, float).reshape(-1, 1))
    ones = T(np.ones((B, 1))); v = out["v"]
    bce = ((acc * (v + eps).log()) +
           ((ones - acc) * (ones - v + eps).log())) * -1.0
    total = ce_f + ce_a * lam_a + (ce_s * cm) * lam_s + bce * lam_v
    return total.sum() / B, out


# =============================================================================
#  PART 4 — Synthetic world. Three input regimes, each mapped to the history:
#    CLEAN        : well-formed rite. surface & acrostic both name referent c.
#    INNOCENT NOISE: "in nomine patria et filia" — surface heavily garbled,
#                    acrostic lightly perturbed. True referent still c. ACCEPT.
#    FORGED SURFACE: Aldebert — surface confidently mimics a DIFFERENT referent
#                    t, while the acrostic still encodes the true source c.
#                    True reference is c; the rite must be REJECTED.
# =============================================================================

def build_referents(rng, K, D_s, D_a):
    mu = rng.standard_normal((K, D_s)) * 1.0        # surface signatures
    kappa = rng.standard_normal((K, D_a)) * 0.75    # acrostic codes (closer)
    return mu, kappa


def sample_batch(rng, B, mu, kappa, D_s, D_a, sfac=1.0, afac=1.0):
    K = mu.shape[0]
    X = np.zeros((B, D_s + D_a)); y = np.zeros(B, int)
    accept = np.zeros(B, int); clean = np.zeros(B, int); kind = np.zeros(B, int)
    for i in range(B):
        c = rng.integers(0, K); r = rng.random()
        if r < 0.34:                                 # clean
            xs = mu[c] + rng.standard_normal(D_s) * 0.05 * sfac
            xa = kappa[c] + rng.standard_normal(D_a) * 0.05 * afac
            accept[i] = 1; clean[i] = 1; kind[i] = 0
        elif r < 0.67:                               # innocent noise
            xs = mu[c] + rng.standard_normal(D_s) * 1.3 * sfac
            xa = kappa[c] + rng.standard_normal(D_a) * 0.35 * afac
            accept[i] = 1; clean[i] = 0; kind[i] = 1
        else:                                        # forged surface
            t = rng.integers(0, K)
            while t == c:
                t = rng.integers(0, K)
            xs = mu[t] + rng.standard_normal(D_s) * 0.2 * sfac
            xa = kappa[c] + rng.standard_normal(D_a) * 0.35 * afac
            accept[i] = 0; clean[i] = 0; kind[i] = 2
        X[i, :D_s] = xs; X[i, D_s:] = xa; y[i] = c
    return X, y, accept, clean, kind


# =============================================================================
#  PART 5 — Optimiser (Adam, from scratch).
# =============================================================================

class Adam:
    def __init__(self, params, lr=3e-3):
        self.p = params; self.lr = lr
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]; self.t = 0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.p):
            g = p.grad
            self.m[i] = 0.9 * self.m[i] + 0.1 * g
            self.v[i] = 0.999 * self.v[i] + 0.001 * (g * g)
            mh = self.m[i] / (1 - 0.9 ** self.t)
            vh = self.v[i] / (1 - 0.999 ** self.t)
            p.data -= self.lr * mh / (np.sqrt(vh) + 1e-8)

    def zero(self):
        for p in self.p:
            p.grad = np.zeros_like(p.data)


# =============================================================================
#  PART 6 — Helpers: witness mass, surface-only ablation, evaluation.
# =============================================================================

def witness_mass(P, X, G):
    """Mean reconciled softmax mass carried by each prototype (how load-bearing
    each 'witness' is). Dead wood has ~0 mass."""
    return forward(P, X, G)["pf_m"].data.mean(0)


def surface_only_pred(P, X, G):
    """ABLATION: ignore the acrostic channel entirely and read the referent
    from the (corruptible) surface alone. This is the reader who trusts the
    body of the rite and is therefore fooled by a forged surface."""
    x = T(X); Gm = T(G)
    hs = (x.matmul(P.Ws1) + P.bs1).tanh()
    s = (hs.matmul(P.Ws2) + P.bs2).tanh()
    ps = softmax(s.matmul(P.Us) + P.cs).matmul(Gm)
    return ps.data.argmax(1)


def evaluate(P, rng, mu, kappa, D_s, D_a, G, n=3000, sfac=1.0, afac=1.0):
    X, y, acc, clean, kind = sample_batch(rng, n, mu, kappa, D_s, D_a, sfac, afac)
    out = forward(P, X, G)
    pred = out["pf_k"].data.argmax(1)
    v = out["v"].data[:, 0]
    so = surface_only_pred(P, X, G)
    R = {}
    for nm, kv in [("clean", 0), ("noise", 1), ("forgery", 2)]:
        m = kind == kv
        R["fused_" + nm] = float((pred[m] == y[m]).mean())
        R["surf_" + nm] = float((so[m] == y[m]).mean())
    ap = (v > 0.5).astype(int)
    R["validity_acc"] = float((ap == acc).mean())
    R["accept_innocent"] = float(ap[(kind == 0) | (kind == 1)].mean())
    R["reject_forgery"] = float(1 - ap[kind == 2].mean())
    return R


# =============================================================================
#  PART 7 — Guarded graft-not-delete.
#  Every so often: find a DEAD spare prototype (class -1, ~0 witness mass),
#  find the currently WEAKEST referent, and re-load the spare with a strong
#  witness of that referent (the oak's wood -> the chapel). KEEP the graft only
#  if held-out loss does not rise (Rome tempering Boniface's zeal); else revert.
# =============================================================================

def try_graft(P, proto_class, K, M, D_s, D_a, rng, mu, kappa, G):
    probe = sample_batch(rng, 600, mu, kappa, D_s, D_a)
    Xp, yp = probe[0], probe[1]
    wm = witness_mass(P, Xp, G)
    spares = [j for j in range(M) if proto_class[j] == -1 and wm[j] < 0.01]
    if not spares:
        return G, None
    pred = forward(P, Xp, G)["pf_k"].data.argmax(1)
    errs = [(1 - (pred[yp == c] == c).mean()) if (yp == c).any() else 0.0
            for c in range(K)]
    cstar = int(np.argmax(errs))                      # neediest referent
    live = [j for j in range(M) if proto_class[j] == cstar]
    if not live:
        return G, None
    j0 = live[int(np.argmax(wm[live]))]               # its strongest witness
    j = spares[0]                                     # the dead wood
    before, _ = loss_fn(P, *probe[:4], G)
    before = float(before.data)
    snap = (P.Us.data[:, j].copy(), P.cs.data[j].copy(),
            P.Ua.data[:, j].copy(), P.ca.data[j].copy(), proto_class[j])
    # re-load the spare from the strong witness (+ small novelty)
    P.Us.data[:, j] = P.Us.data[:, j0] + rng.standard_normal(P.Us.data.shape[0]) * 0.05
    P.cs.data[j] = P.cs.data[j0]
    P.Ua.data[:, j] = P.Ua.data[:, j0] + rng.standard_normal(P.Ua.data.shape[0]) * 0.05
    P.ca.data[j] = P.ca.data[j0]
    proto_class[j] = cstar
    Gtry = make_group_matrix(proto_class, K)
    after, _ = loss_fn(P, *probe[:4], Gtry)
    after = float(after.data)
    mass_after = float(witness_mass(P, Xp, Gtry)[j])
    if after <= before + 1e-4:                        # guard: do no harm
        return Gtry, ("KEEP", j, cstar, round(before, 4), round(after, 4),
                      round(float(wm[j]), 4), round(mass_after, 4))
    # revert
    (P.Us.data[:, j], P.cs.data[j], P.Ua.data[:, j], P.ca.data[j],
     proto_class[j]) = snap
    return G, ("REVERT", j, cstar, round(before, 4), round(after, 4),
               round(float(wm[j]), 4), 0.0)


# =============================================================================
#  PART 8 — The mandatory finite-difference gradient check.
# =============================================================================

def gradient_check():
    rng = np.random.default_rng(0)
    D_s, D_a, K, M = 8, 4, 4, 6
    P = Params(rng, D_s + D_a, K, M, h_s=10, d_s=6, h_a=8, d_a=4, h_v=5)
    proto_class = list(range(K)) + [-1] * (M - K)
    G = make_group_matrix(proto_class, K)
    X, y, acc, clean, _ = sample_batch(rng, 5, *build_referents(rng, K, D_s, D_a),
                                       D_s, D_a)
    for p in P.all():
        p.grad = np.zeros_like(p.data)
    L, _ = loss_fn(P, X, y, acc, clean, G)
    L.backward()
    h, max_rel = 1e-5, 0.0
    for p in P.all():
        flat = p.data.reshape(-1); n = flat.size
        idxs = range(n) if n <= 12 else rng.choice(n, 12, replace=False)
        for k in idxs:
            o = flat[k]
            flat[k] = o + h; Lp, _ = loss_fn(P, X, y, acc, clean, G)
            flat[k] = o - h; Lm, _ = loss_fn(P, X, y, acc, clean, G)
            flat[k] = o
            num = (Lp.data - Lm.data) / (2 * h)
            ana = p.grad.reshape(-1)[k]
            max_rel = max(max_rel, abs(num - ana) / max(1e-8, abs(num) + abs(ana)))
    return max_rel


# =============================================================================
#  PART 9 — Train, ablate, graft, and report.
# =============================================================================

def train(seed=3, steps=2200, do_graft=True, verbose=True):
    rng = np.random.default_rng(seed)
    D_s, D_a, K, M = 16, 8, 6, 10
    proto_class = list(range(K)) + [-1] * (M - K)     # 6 live + 4 dead spares
    G = make_group_matrix(proto_class, K)
    mu, kappa = build_referents(rng, K, D_s, D_a)
    P = Params(rng, D_s + D_a, K, M, h_s=32, d_s=16, h_a=24, d_a=8, h_v=8)
    opt = Adam(P.all(), lr=3e-3)
    graft_events = []
    for it in range(steps):
        X, y, acc, clean, kind = sample_batch(rng, 64, mu, kappa, D_s, D_a)
        opt.zero()
        L, _ = loss_fn(P, X, y, acc, clean, G)
        L.backward()
        opt.step()
        if do_graft and it >= 800 and it % 300 == 0:
            G, ev = try_graft(P, proto_class, K, M, D_s, D_a, rng, mu, kappa, G)
            if ev:
                graft_events.append((it,) + ev)
        if verbose and (it % 400 == 0 or it == steps - 1):
            r = evaluate(P, np.random.default_rng(99), mu, kappa, D_s, D_a, G, 800)
            print(f"  step {it:4d} | loss {float(L.data):5.3f} | "
                  f"fused c/n/f {r['fused_clean']:.2f}/{r['fused_noise']:.2f}/"
                  f"{r['fused_forgery']:.2f} | surf-forge {r['surf_forgery']:.2f} | "
                  f"validity {r['validity_acc']:.2f}")
    return P, proto_class, G, (mu, kappa, D_s, D_a, K, M), graft_events


def main():
    np.seterr(over="ignore")
    print("=" * 74)
    print(" Chapter 181 · St. Boniface — 'The Orthography of Grace'")
    print(" A dual-channel reference engine (surface + protected acrostic)")
    print("=" * 74)

    print("\n[1] GRADIENT CHECK (central finite differences vs. autodiff)")
    rel = gradient_check()
    print(f"    max relative error = {rel:.3e}  ->  "
          f"{'PASS' if rel < 1e-4 else 'FAIL'}")
    assert rel < 1e-4, "gradient check failed"

    print("\n[2] TRAINING (with guarded graft-not-delete)")
    P, proto_class, G, world, graft_events = train(seed=3, verbose=True)
    mu, kappa, D_s, D_a, K, M = world
    ev_rng = np.random.default_rng(2024)

    print("\n[3] FINAL METRICS (held-out)")
    R = evaluate(P, ev_rng, mu, kappa, D_s, D_a, G, 5000)
    print(f"    reference recovery  clean / innocent-noise / forged-surface :"
          f" {R['fused_clean']:.3f} / {R['fused_noise']:.3f} / {R['fused_forgery']:.3f}")
    print(f"    validity gate       accuracy / accept-innocent / reject-forgery:"
          f" {R['validity_acc']:.3f} / {R['accept_innocent']:.3f} / {R['reject_forgery']:.3f}")

    print("\n[4] ABLATION — surface-only vs. cross-channel on FORGED surfaces")
    print(f"    surface-only recovers the true reference : {R['surf_forgery']:.3f}"
          f"   (fooled: it follows the forged surface)")
    print(f"    cross-channel recovers the true reference: {R['fused_forgery']:.3f}"
          f"   (the protected acrostic pulls it back)")

    print("\n[5] STRESS TEST — corrupt the protected channel 2x")
    S = evaluate(P, ev_rng, mu, kappa, D_s, D_a, G, 4000, afac=2.0)
    print(f"    reference recovery clean/noise/forgery : "
          f"{S['fused_clean']:.3f}/{S['fused_noise']:.3f}/{S['fused_forgery']:.3f}"
          f"  (degrades gracefully — the acrostic is doing the work)")

    print("\n[6] GUARDED GRAFTS (oak's wood -> chapel, tempered by the guard)")
    kept = [e for e in graft_events if e[1] == "KEEP"]
    rev = [e for e in graft_events if e[1] == "REVERT"]
    for it, tag, j, c, b, a, mb, ma in graft_events:
        note = (f"reused dead proto #{j} as extra witness for referent {c}"
                if tag == "KEEP" else
                f"reverted proto #{j}->{c} (would raise loss)")
        print(f"    step {it:4d} [{tag:6}] loss {b}->{a} | {note}"
              + (f" | mass {mb}->{ma}" if tag == "KEEP" else ""))
    print(f"    prototype->class map after grafting: {proto_class}")
    print(f"    ({len(kept)} kept, {len(rev)} reverted — the guard admits reuse "
          f"only when it does not break what works)")

    print("\n[7] SELF-TESTS")
    checks = [
        ("gradient check < 1e-4", rel < 1e-4),
        ("clean recovery > 0.95", R["fused_clean"] > 0.95),
        ("innocent-noise recovery > 0.90", R["fused_noise"] > 0.90),
        ("cross-channel recovers forged reference > 0.85", R["fused_forgery"] > 0.85),
        ("surface-only is fooled by forgery < 0.25", R["surf_forgery"] < 0.25),
        ("cross-channel beats surface-only on forgery by > 0.6",
         R["fused_forgery"] - R["surf_forgery"] > 0.6),
        ("validity gate accuracy > 0.90", R["validity_acc"] > 0.90),
        ("rejects forgeries > 0.85", R["reject_forgery"] > 0.85),
        ("accepts innocent noise > 0.85", R["accept_innocent"] > 0.85),
        ("every kept graft did not raise held-out loss",
         all(a <= b + 1e-4 for _, tag, _, _, b, a, _, _ in kept)),
        ("every kept graft reactivated dead material (mass rose)",
         all(ma > mb for _, tag, _, _, _, _, mb, ma in kept)),
    ]
    ok = True
    for name, passed in checks:
        print(f"    [{'PASS' if passed else 'FAIL'}] {name}")
        ok = ok and passed
    print("\n" + "=" * 74)
    print(" ALL SELF-TESTS PASSED" if ok else " SOME SELF-TESTS FAILED")
    print(" The mind's thesis, made mechanical: reference survives a corrupted")
    print(" surface because it is carried redundantly; a forger is caught not by")
    print(" a malformed surface but by a confident surface that DISAGREES with")
    print(" the protected channel; and falsified material is re-loaded, never")
    print(" simply deleted — but only when the reuse does no harm.")
    print("=" * 74)
    assert ok, "self-tests failed"


if __name__ == "__main__":
    main()
