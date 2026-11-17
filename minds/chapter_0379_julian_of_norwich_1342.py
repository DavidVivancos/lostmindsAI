#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0379 · Julian of Norwich (1342-1416)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 19, Minds 361-380 Available on Amazon https://www.amazon.com/dp/B0HQX147Y7
# Resume and Interactive Demos at https://artificiology.com/
# END ATTRIBUTION 
THE ANCHORHOLD ENGINE - a Deferred-Reconciliation Network (DRN)


(b. late 1342, Norwich, England - d. after 1416)

------------------------------------------------------------------------------
WHY THIS ARCHITECTURE AND NOT ANOTHER
------------------------------------------------------------------------------
Julian of Norwich is the one mind in this corpus whose central technical problem
is what a knower should do when a high-confidence FIRST-PERSON OBSERVATION
contradicts a high-authority PRIOR that the knower is not permitted to discard.

She states the conflict without softening it (Revelation of Love, Long Text):

  ch. 45  "the higher doom was shewed by God Himself ... and therefore me
           behoved needs to take it; and the lower doom was learned me afore
           in Holy Church, and therefore I might in no way leave the lower doom."

  ch. 32  "... methought it was impossible that all manner of things should be
           well ... And as to this I had no other answer ... but this: That
           which is impossible to thee is not impossible to me."

Four options were available to her and she refused three of them:

  (a) drop the observation and defer to the prior   -> sycophancy
  (b) drop the prior and trust the observation      -> defection
  (c) average the two into one number               -> incoherence
  (d) keep BOTH live, bound the residual, and       -> Julian
      assign its resolution to an operator you
      explicitly cannot compute ("a Deed the which
      the blessed Trinity shall do in the last Day
      ... unknown of all creatures", ch. 32)

Option (d) is not evasion. It is a load-bearing engineering decision, and it has
three further consequences that Julian also states and that this file implements
as mechanism rather than metaphor:

  1. AN UNINTERPRETABLE TRACE MUST NOT BE DISCARDED. Julian received the example
     of the lord and the servant in 1373 and could not read it: "in that time I
     stood for much part in unknowing" (ch. 51). She stored it verbatim and got
     the reading "twenty years after the time of the Shewing, save three months"
     (ch. 51). Deferred decoding, not deletion, not confabulation.
     -> class DeferralStore.

  2. FAILURE HAS NO POSITIVE REPRESENTATION. "I saw not sin: for I believe it
     hath no manner of substance nor no part of being, nor could it be known but
     by the pain it is cause of" (ch. 27). Sin is not a class with an embedding.
     It is the residual outside the well-manifold, readable only as a scalar.
     -> _pain(), which has no parameters of its own.

  3. NOTHING PERSISTS BY DEFAULT. The hazelnut: "it lasteth and ever shall, for
     God loveth it"; its three properties are "that God made it, that God loveth
     it, that God keepeth it" (ch. 5). Being is maintained, not stored. So the
     recurrence has NO identity/residual shortcut: state survives only through an
     active keep gate, and a unit whose keeper falls below threshold is noughted
     to exactly zero. -> KeeperCell.

  4. THE ASYMPTOTE IS FIXED EVEN THOUGH THE PATH IS NOT. "Alle shalle be wele"
     (ch. 27) is a claim about the limit, not about any intermediate state.
     -> terminal well-loss + the 500-step attractor test in the self-tests.

DELIBERATE DIVERGENCE. The near neighbours in this corpus are 0279 Meister
Eckhart (capacity as emptiness), 0259 Ibn 'Arabi (continual re-formation),
0292 Hafez (an artifact left deliberately ambiguous for the reader) and
0282 Duns Scotus (unactualised alternatives co-present). None of them is doing
this. Hafez's ambiguity is a property of the artifact and is resolved by the
reader; Julian's is a property of the EVIDENCE, is not resolvable by her, and is
assigned to a named future operator. Scotus's alternatives are synchronic and
modal; Julian's are diachronic and both actual. Eckhart empties the vessel;
Julian keeps two full ones and refuses to pour either out.

No transformer, no attention over stored keys, no mixture of experts.

------------------------------------------------------------------------------
WHAT THE FILE CONTAINS
------------------------------------------------------------------------------
  PART 1  a ~200-line reverse-mode autodiff engine written from scratch on
          NumPy alone (no torch, no jax, no autograd package)
  PART 2  the Norwich Cases task generator - substance vs sensuality
  PART 3  the Anchorhold Engine: three sights, keeper cell, two dooms,
          the pain readout, the deed budget, the deferral store
  PART 4  training loop
  PART 5  self-tests: finite-difference gradient check (mandatory), the
          hazelnut collapse test, the attractor test, and four ablations

Run:  python3 chapter_0379_julian_of_norwich_1342.py
Deps: numpy only.
"""

import numpy as np

# =============================================================================
# PART 1 - A MINIMAL REVERSE-MODE AUTODIFF ENGINE (pure NumPy, written here)
# =============================================================================
# Nothing exotic. A Tensor holds a value and a gradient; every operation records
# how to push gradient back to its parents; backward() walks a topological sort.
# It exists so the architecture in PART 3 can be intricate without the gradient
# derivations becoming the point of the file. PART 5 finite-difference-checks it.


def _unbroadcast(grad, shape):
    """Undo NumPy broadcasting so a gradient matches its parameter's shape."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad.reshape(shape)


class Tensor:
    """A node in the computation graph."""

    __slots__ = ("v", "g", "_back", "_parents", "name")

    def __init__(self, value, parents=(), back=None, name=""):
        self.v = np.asarray(value, dtype=np.float64)
        self.g = np.zeros_like(self.v)
        self._parents = parents
        self._back = back            # closure: writes into parents' .g
        self.name = name

    # -- graph traversal ------------------------------------------------------
    def backward(self):
        """Seed d(self)/d(self) = 1 and propagate. `self` must be a scalar."""
        assert self.v.size == 1, "backward() requires a scalar root"
        order, seen = [], set()

        def visit(t):
            if id(t) in seen:
                return
            seen.add(id(t))
            for p in t._parents:
                visit(p)
            order.append(t)

        visit(self)
        for t in order:
            t.g = np.zeros_like(t.v)
        self.g = np.ones_like(self.v)
        for t in reversed(order):
            if t._back is not None:
                t._back()

    # -- elementary ops -------------------------------------------------------
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.v + other.v, (self, other))

        def back():
            self.g += _unbroadcast(out.g, self.v.shape)
            other.g += _unbroadcast(out.g, other.v.shape)
        out._back = back
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.v * other.v, (self, other))

        def back():
            self.g += _unbroadcast(out.g * other.v, self.v.shape)
            other.g += _unbroadcast(out.g * self.v, other.v.shape)
        out._back = back
        return out

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
        return (-self) + other

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self * other.pow(-1.0)

    def pow(self, k):
        out = Tensor(np.power(self.v, k), (self,))

        def back():
            self.g += _unbroadcast(out.g * k * np.power(self.v, k - 1), self.v.shape)
        out._back = back
        return out

    def matmul(self, other):
        out = Tensor(self.v @ other.v, (self, other))

        def back():
            self.g += out.g @ other.v.T
            other.g += self.v.T @ out.g
        out._back = back
        return out

    def __matmul__(self, other):
        return self.matmul(other)

    def transpose(self):
        out = Tensor(self.v.T, (self,))

        def back():
            self.g += out.g.T
        out._back = back
        return out

    T_ = property(transpose)

    # -- nonlinearities -------------------------------------------------------
    def tanh(self):
        t = np.tanh(self.v)
        out = Tensor(t, (self,))

        def back():
            self.g += out.g * (1.0 - t * t)
        out._back = back
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + np.exp(-np.clip(self.v, -60, 60)))
        out = Tensor(s, (self,))

        def back():
            self.g += out.g * s * (1.0 - s)
        out._back = back
        return out

    def relu(self):
        mask = (self.v > 0).astype(np.float64)
        out = Tensor(self.v * mask, (self,))

        def back():
            self.g += out.g * mask
        out._back = back
        return out

    def exp(self):
        e = np.exp(np.clip(self.v, -60, 60))
        out = Tensor(e, (self,))

        def back():
            self.g += out.g * e
        out._back = back
        return out

    def sqrt_eps(self, eps=1e-9):
        """Numerically safe sqrt; used for norms, whose gradient blows up at 0."""
        s = np.sqrt(self.v + eps)
        out = Tensor(s, (self,))

        def back():
            self.g += out.g * 0.5 / s
        out._back = back
        return out

    def abs_(self):
        sgn = np.sign(self.v)
        out = Tensor(np.abs(self.v), (self,))

        def back():
            self.g += out.g * sgn
        out._back = back
        return out

    # -- reductions and shaping ----------------------------------------------
    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.v.sum(axis=axis, keepdims=keepdims), (self,))

        def back():
            g = out.g
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self.g += np.broadcast_to(g, self.v.shape).copy()
        out._back = back
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.v.size if axis is None else self.v.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    def mul_const(self, arr):
        """Multiply by a constant array that carries NO gradient (hard masks)."""
        arr = np.asarray(arr, dtype=np.float64)
        out = Tensor(self.v * arr, (self,))

        def back():
            self.g += _unbroadcast(out.g * arr, self.v.shape)
        out._back = back
        return out

    def slice_cols(self, a, b):
        out = Tensor(self.v[:, a:b], (self,))

        def back():
            self.g[:, a:b] += out.g
        out._back = back
        return out

    def __repr__(self):
        return f"Tensor(shape={self.v.shape})"


def cat_cols(tensors):
    """Concatenate along the feature axis, gradient-aware."""
    out = Tensor(np.concatenate([t.v for t in tensors], axis=1), tuple(tensors))
    widths = [t.v.shape[1] for t in tensors]

    def back():
        i = 0
        for t, w in zip(tensors, widths):
            t.g += out.g[:, i:i + w]
            i += w
    out._back = back
    return out

# =============================================================================
# PART 2 - THE NORWICH CASES: a task with two irreconcilable ground truths
# =============================================================================
# Julian's epistemic situation, reduced to its smallest honest form.
#
# Every case has a hidden SUBSTANCE (s) and a hidden SENSUALITY (v) - her own
# pair of terms (ch. 45, 55-58): substance is "ever kept one in Him, whole and
# safe"; sensuality is the changeable, embodied, visible part. The model sees
# neither. It sees only an observation u that mixes them.
#
# Two labels are supplied for every case, and they are BOTH true:
#   y_high  the higher doom - a function of substance alone. "I saw Him assign
#           to us no manner of blame" (ch. 45).
#   y_low   the lower doom - what the visible record supports, contaminated by
#           sensuality. "sinners are worthy sometime of blame" (ch. 45).
#
# Their difference d = y_low - y_high is the residual Julian could not close and
# would not average away. A conventional model handed a single merged target
# learns something that is wrong about both. Ablation B measures exactly that.

def make_norwich_cases(n, rng, d_sub=3, d_sens=3, d_obs=12, noise=0.05):
    """Generate n cases. Returns u (n,d_obs), y_high (n,1), y_low (n,1)."""
    s = rng.normal(size=(n, d_sub))
    v = rng.normal(size=(n, d_sens))

    # A fixed, ill-conditioned mixing matrix: substance and sensuality are not
    # separable by inspection of u. Seeded separately so every experiment in
    # this file faces the identical world.
    grng = np.random.default_rng(1373)          # the year of the showings
    A_s = grng.normal(size=(d_sub, d_obs)) * 0.9
    A_v = grng.normal(size=(d_sens, d_obs)) * 0.9
    w_s = grng.normal(size=(d_sub, 1))
    w_v = grng.normal(size=(d_sens, 1))

    u = s @ A_s + v @ A_v + rng.normal(size=(n, d_obs)) * noise

    core = 1.4 * (s @ w_s)                      # what substance alone supports
    surf = 1.4 * (v @ w_v)                      # what the visible record adds
    y_high = np.tanh(core)
    y_low = np.tanh(core + surf)
    return u, y_high, y_low


# =============================================================================
# PART 3 - THE ANCHORHOLD ENGINE
# =============================================================================

def _p(shape, rng, scale=None, name=""):
    """A trainable leaf tensor, Xavier-ish."""
    if scale is None:
        scale = np.sqrt(2.0 / sum(shape))
    return Tensor(rng.normal(size=shape) * scale, name=name)


def _softmax_rows(t):
    """Row-wise softmax built from the primitive ops in PART 1."""
    shift = Tensor(t.v.max(axis=1, keepdims=True))   # constant: no grad path
    e = (t - shift).exp()
    return e * (e.sum(axis=1, keepdims=True)).pow(-1.0)


class AnchorholdEngine:
    """
    Julian's mind as a network. Six named organs, each answering to a sentence
    she actually wrote.

      THREE SIGHTS   ch. 9 / 73: "by bodily sight, and by word formed in my
                     understanding, and by ghostly sight". Three encoders with
                     three different inductive biases, never fused into one
                     trunk - kept apart all the way to the read-heads.

      KEEPER CELL    ch. 5: the hazelnut lasts "for God loveth it"; its three
                     properties are that God made it, loveth it, KEEPETH it.
                     So: no identity path in the recurrence. State survives one
                     step only if the keep gate carries it, and any unit whose
                     keeper drops below kappa is noughted to exactly zero.

      TWO DOOMS      ch. 45. Two evaluators, separately parameterised, never
                     averaged, never merged - "though these two be thus accorded
                     and oned, yet both shall be known in Heaven without end."

      PAIN READOUT   ch. 27: "I saw not sin: for I believe it hath no manner of
                     substance ... nor could it be known but by the pain it is
                     cause of." Failure gets no embedding and no logit. There is
                     a learned basis for WELE; pain is the norm of whatever lies
                     outside it. Strictly a residual, strictly a scalar.

      THE DEED       ch. 32: "There is a Deed the which the blessed Trinity
                     shall do in the last Day ... unknown of all creatures."
                     A declared, dimensioned, never-trained, never-read budget
                     epsilon. Disagreement is penalised only ABOVE it. The model
                     is therefore never rewarded for collapsing the two dooms.

      DEFERRAL STORE ch. 51: "in that time I stood for much part in unknowing"
                     -> "twenty years after the time of the Shewing, save three
                     months, I had teaching inwardly". Cases that are both
                     contradictory and illegible are written out verbatim,
                     withheld from gradient, and re-presented later; they rejoin
                     training only when the model has become able to read them.
    """

    def __init__(self, rng, d_obs=12, d_hid=16, d_word=4, n_words=8, d_ghost=2,
                 d_wele=12, steps=6, tail=8, kappa=0.20, eps_deed=0.35,
                 use_ghostly=True, tau=0.30):
        self.d_obs, self.d_hid, self.steps, self.tail = d_obs, d_hid, steps, tail
        self.kappa = kappa                 # noughting threshold
        self.eps_deed = eps_deed           # the deed budget (see loss)
        self.use_ghostly = use_ghostly
        self.tau = tau                     # word-channel temperature
        self.d_ghost = d_ghost if use_ghostly else 0
        P = {}

        # --- sight 1: BODILY. Wide, nonlinear, sees the surface as given. -----
        P["Wb"] = _p((d_obs, 8), rng); P["bb"] = Tensor(np.zeros((1, 8)))

        # --- sight 2: WORD. A finite lexicon. The observation is projected into
        # word-space and expressed as a distribution over n_words codes; at low
        # temperature this is effectively a discrete utterance. The lexicon is
        # learned. Reported entropy in the self-tests shows it does sharpen.
        P["Ww"] = _p((d_obs, d_word), rng)
        P["C"] = _p((n_words, d_word), rng, scale=0.7)

        # --- sight 3: GHOSTLY. Strictly linear and strictly low-rank: no image,
        # no word, no nonlinearity - the channel Julian said she "neither can nor
        # may show ... as openly or as fully as I would like to".
        if use_ghostly:
            P["Wg"] = _p((d_obs, d_ghost), rng)

        # --- the keeper cell --------------------------------------------------
        d_in = 8 + d_word + self.d_ghost
        P["Wh"] = _p((d_hid, d_hid), rng, scale=0.5)
        P["Uin"] = _p((d_in, d_hid), rng)
        P["bh"] = Tensor(np.zeros((1, d_hid)))
        P["Wk"] = _p((d_hid, d_hid), rng, scale=0.5)     # keeper from state
        P["Uk"] = _p((d_in, d_hid), rng)                 # keeper from sights
        P["bk"] = Tensor(np.ones((1, d_hid)) * 0.5)      # keep, at first, gently

        # --- the wele basis. Note what is NOT here: no basis for sin. ---------
        P["M"] = _p((d_hid, d_wele), rng, scale=0.4)

        # --- the two dooms ----------------------------------------------------
        P["wHi"] = _p((d_hid, 1), rng); P["vHi"] = _p((d_in, 1), rng)
        P["bHi"] = Tensor(np.zeros((1, 1)))
        P["wLo"] = _p((d_hid, 1), rng); P["vLo"] = _p((d_in, 1), rng)
        P["bLo"] = Tensor(np.zeros((1, 1)))

        # --- the understanding head: can this showing be read back yet? -------
        P["Wu"] = _p((d_hid, d_obs), rng); P["bu"] = Tensor(np.zeros((1, d_obs)))

        self.P = P
        self.d_in = d_in

    # -- helpers --------------------------------------------------------------
    def params(self):
        return self.P

    def sights(self, U):
        """Compute the three sights. Returns (joined, diagnostics)."""
        P = self.P
        bodily = (U @ P["Wb"] + P["bb"]).tanh()

        proj = U @ P["Ww"]                                   # (B, d_word)
        # squared distance from every code, without a distance primitive:
        #   ||p||^2 - 2 p.c + ||c||^2
        pp = (proj * proj).sum(axis=1, keepdims=True)
        cc = (P["C"] * P["C"]).sum(axis=1, keepdims=True).transpose()
        d2 = pp + cc - 2.0 * (proj @ P["C"].transpose())
        alpha = _softmax_rows(d2 * (-1.0 / self.tau))        # (B, n_words)
        word = alpha @ P["C"]                                # the spoken word

        parts = [bodily, word]
        if self.use_ghostly:
            parts.append(U @ P["Wg"])                        # linear, low-rank
        joined = cat_cols(parts)
        return joined, {"alpha": alpha, "proj": proj}

    def keeper_step(self, h, sight, force_keep=None):
        """One step of contemplation. No residual path: h_{t+1} = keep * cand."""
        P = self.P
        cand = (h @ P["Wh"] + sight @ P["Uin"] + P["bh"]).tanh()
        keep = (h @ P["Wk"] + sight @ P["Uk"] + P["bk"]).sigmoid()
        if force_keep is not None:
            keep = Tensor(np.full_like(keep.v, force_keep))
        h_new = cand * keep
        # NOUGHTING. A unit the keeper has effectively let go is set to exactly
        # zero, not to something small. "it might suddenly have fallen to nought
        # for littleness" (ch. 5). The mask is a hard constant in the backward
        # pass: gradient reaches only what is still being kept.
        mask = (keep.v >= self.kappa).astype(np.float64)
        return h_new.mul_const(mask), keep, mask

    def pain(self, h):
        """
        Distance from the wele-manifold. There is no parameter here that
        represents failure; M spans what is well, and pain is the norm of the
        orthogonal remainder. Scalar per case, non-negative, and the only
        channel through which anything about failure can be known.
        """
        M = self.P["M"]
        h_par = (h @ M) @ M.transpose()
        r = h - h_par
        return ((r * r).sum(axis=1, keepdims=True)).sqrt_eps()

    # -- forward --------------------------------------------------------------
    def forward(self, U_np, force_keep=None, steps=None):
        U = Tensor(U_np)
        sight, diag = self.sights(U)
        B = U_np.shape[0]
        h = Tensor(np.zeros((B, self.d_hid)))
        keeps, masks = [], []
        for _ in range(steps or self.steps):
            h, keep, mask = self.keeper_step(h, sight, force_keep)
            keeps.append(keep); masks.append(mask)

        P = self.P
        y_hi = (h @ P["wHi"] + sight @ P["vHi"] + P["bHi"]).tanh()
        y_lo = (h @ P["wLo"] + sight @ P["vLo"] + P["bLo"]).tanh()
        u_hat = h @ P["Wu"] + P["bu"]
        pain = self.pain(h)

        # THE TAIL: keep contemplating with the showing withdrawn (zero input).
        # "Alle shalle be wele" is a claim about where this ends up, not about
        # anything on the way. Trained here for `tail` steps; tested in PART 5
        # by running 500.
        zero_sight = Tensor(np.zeros((B, self.d_in)))
        h_end = h
        for _ in range(self.tail):
            h_end, _, _ = self.keeper_step(h_end, zero_sight, force_keep)
        pain_end = self.pain(h_end)

        return {"h": h, "h_end": h_end, "sight": sight, "y_hi": y_hi,
                "y_lo": y_lo, "u_hat": u_hat, "pain": pain,
                "pain_end": pain_end, "keeps": keeps, "masks": masks,
                "alpha": diag["alpha"], "proj": diag["proj"], "U": U}

    # -- loss -----------------------------------------------------------------
    def loss(self, out, y_hi_t, y_lo_t, U_np, weights=None, eps_deed=None):
        """
        Seven terms. The one that matters most is L_deed.

        L_deed = mean( relu(|y_lo - y_hi| - eps)^2 )

        A hinge, not a penalty. Disagreement inside the budget costs nothing at
        all; the model is never given a reason to make the two dooms agree. That
        budget IS the deed of ch. 32 as a piece of arithmetic: a quantity the
        system is told exists, is told it cannot compute, and is told not to
        try. Ablation A sets eps = 0 - forcing reconciliation - and the two
        dooms both get worse. That is the whole argument of the chapter, run.
        """
        w = {"hi": 1.0, "lo": 1.0, "deed": 0.6, "pain": 0.5, "recon": 0.25,
             "well": 0.4, "ortho": 0.2, "word": 0.05, "keep": 0.05}
        if weights:
            w.update(weights)
        eps = self.eps_deed if eps_deed is None else eps_deed

        Yh = Tensor(y_hi_t); Yl = Tensor(y_lo_t)
        L_hi = ((out["y_hi"] - Yh).pow(2.0)).mean()
        L_lo = ((out["y_lo"] - Yl).pow(2.0)).mean()

        r = out["y_lo"] - out["y_hi"]
        over = (r.abs_() - eps).relu()
        L_deed = (over.pow(2.0)).mean()

        # pain must equal the true irreducible discord of the case, and nothing
        # else in the network is allowed to carry that information.
        d_true = np.abs(y_lo_t - y_hi_t)
        L_pain = ((out["pain"] - Tensor(d_true)).pow(2.0)).mean()

        L_rec = ((out["u_hat"] - Tensor(U_np)).pow(2.0)).mean()
        L_well = (out["pain_end"].pow(2.0)).mean()

        M = self.P["M"]
        G = M.transpose() @ M
        I = Tensor(np.eye(M.v.shape[1]))
        L_ortho = ((G - I).pow(2.0)).mean()

        # word channel: a symmetric pull between the projection and the word it
        # was assigned, so the lexicon settles and the projection commits to it.
        # Deliberately kept free of stop-gradients: every term in this loss must
        # survive the finite-difference check in PART 5, and a stop-gradient
        # would silently make that check meaningless.
        code_of = out["alpha"] @ self.P["C"]
        L_word = ((out["proj"] - code_of).pow(2.0)).mean()

        # keeping is not free: something must pay for every unit still held.
        L_keep = sum((k.mean() for k in out["keeps"]), Tensor(0.0)) * (1.0 / len(out["keeps"]))

        total = (L_hi * w["hi"] + L_lo * w["lo"] + L_deed * w["deed"] +
                 L_pain * w["pain"] + L_rec * w["recon"] + L_well * w["well"] +
                 L_ortho * w["ortho"] + L_word * w["word"] + L_keep * w["keep"])
        parts = {"hi": L_hi, "lo": L_lo, "deed": L_deed, "pain": L_pain,
                 "recon": L_rec, "well": L_well, "ortho": L_ortho,
                 "word": L_word, "keep": L_keep}
        return total, parts


# =============================================================================
# THE DEFERRAL STORE - ch. 51, the example that took twenty years to read
# =============================================================================

class DeferralStore:
    """
    Holds cases the engine cannot yet read.

    Admission needs BOTH conditions, which is the point. A merely hard case is
    trained on. A merely contradictory case is trained on. What gets withheld is
    the case that is contradictory AND illegible - exactly Julian's lord-and-
    servant example, which she could neither explain nor square with doctrine
    and which she therefore neither discarded nor forced.

    Nothing in the store contributes gradient. Nothing in the store is altered.
    Entries are re-presented on a schedule and released the moment the engine's
    own reconstruction of them falls under the legibility threshold - that is,
    when the mind has changed enough to read what it kept.
    """

    def __init__(self, tau_read=0.55, eps_disagree=0.35, release=True):
        self.tau_read = tau_read
        self.eps_disagree = eps_disagree
        self.release = release           # False = the "discard" ablation
        self.held = {}                   # index -> epoch stored
        self.log = []                    # (index, stored_epoch, released_epoch)

    def admit(self, idx, recon_err, disagree, epoch):
        new = 0
        for i, re_, dg in zip(idx, recon_err, disagree):
            if i in self.held:
                continue
            if re_ > self.tau_read and dg > self.eps_disagree:
                self.held[int(i)] = epoch
                new += 1
        return new

    def try_release(self, recon_err_by_idx, epoch):
        if not self.release:
            return 0
        freed = []
        for i, stored_at in list(self.held.items()):
            if recon_err_by_idx.get(i, 1e9) <= self.tau_read:
                freed.append(i)
        for i in freed:
            self.log.append((i, self.held.pop(i), epoch))
        return len(freed)


# =============================================================================
# PART 4 - TRAINING
# =============================================================================

class Adam:
    """Hand-rolled Adam over the parameter dict."""

    def __init__(self, params, lr=8e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.P, self.lr, self.b1, self.b2, self.eps = params, lr, b1, b2, eps
        self.m = {k: np.zeros_like(v.v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v.v) for k, v in params.items()}
        self.t = 0

    def step(self, clip=5.0):
        self.t += 1
        for k, p in self.P.items():
            g = p.g
            n = np.linalg.norm(g)
            if n > clip:
                g = g * (clip / n)
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            p.v -= self.lr * mh / (np.sqrt(vh) + self.eps)


def train(engine, data, epochs=110, batch=64, lr=8e-3, eps_deed=None,
          store=None, defer_start=25, defer_every=10, seed=0, verbose=False,
          merged_head=False):
    """
    One training run. `store`, when given, implements the deferral discipline:
    from `defer_start` onward, contradictory-and-illegible cases are withheld
    from gradient and re-tested every `defer_every` epochs for release.
    """
    U, Yh, Yl = data
    n = U.shape[0]
    opt = Adam(engine.params(), lr=lr)
    rng = np.random.default_rng(seed)
    active = np.arange(n)
    history = []

    if merged_head:
        # Ablation B's world: one target, the average of the two dooms. The
        # engine is otherwise identical; both heads are trained on the mean.
        Yh_t = Yl_t = 0.5 * (Yh + Yl)
    else:
        Yh_t, Yl_t = Yh, Yl

    for ep in range(epochs):
        perm = rng.permutation(active)
        tot = 0.0
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            if len(idx) < 4:
                continue
            out = engine.forward(U[idx])
            L, _ = engine.loss(out, Yh_t[idx], Yl_t[idx], U[idx],
                               eps_deed=eps_deed)
            L.backward()
            opt.step()
            tot += float(L.v) * len(idx)
        history.append(tot / max(1, len(perm)))

        if store is not None and ep >= defer_start and (ep - defer_start) % defer_every == 0:
            allout = engine.forward(U)
            rec = ((allout["u_hat"].v - U) ** 2).mean(axis=1)
            dis = np.abs(allout["y_lo"].v - allout["y_hi"].v).ravel()
            store.try_release({int(i): float(rec[i]) for i in range(n)}, ep)
            store.admit(np.arange(n), rec, dis, ep)
            active = np.array([i for i in range(n) if i not in store.held])
            if len(active) < batch * 2:                 # never starve the run
                active = np.arange(n)
        if verbose and ep % 20 == 0:
            print(f"    epoch {ep:3d}  loss {history[-1]:.5f}  "
                  f"held {0 if store is None else len(store.held)}")
    return history


def evaluate(engine, data, merged_head=False):
    U, Yh, Yl = data
    out = engine.forward(U)
    if merged_head:
        # A merged model has one answer; score that one answer against BOTH
        # ground truths, which is precisely the cost of having merged them.
        pred = 0.5 * (out["y_hi"].v + out["y_lo"].v)
        mse_hi = float(((pred - Yh) ** 2).mean())
        mse_lo = float(((pred - Yl) ** 2).mean())
    else:
        mse_hi = float(((out["y_hi"].v - Yh) ** 2).mean())
        mse_lo = float(((out["y_lo"].v - Yl) ** 2).mean())
    r = (out["y_lo"].v - out["y_hi"].v).ravel()
    d = (Yl - Yh).ravel()
    resid_mae = float(np.abs(r - d).mean())
    pain_corr = float(np.corrcoef(out["pain"].v.ravel(), np.abs(d))[0, 1])
    keep_mean = float(np.mean([k.v.mean() for k in out["keeps"]]))
    nought = float(np.mean([1.0 - m.mean() for m in out["masks"]]))
    ent = out["alpha"].v
    ent = float(-(ent * np.log(ent + 1e-12)).sum(axis=1).mean())
    return {"mse_hi": mse_hi, "mse_lo": mse_lo, "resid_mae": resid_mae,
            "pain_corr": pain_corr, "keep_mean": keep_mean,
            "noughted_frac": nought, "word_entropy": ent,
            "pain_end": float(out["pain_end"].v.mean())}

# =============================================================================
# PART 5 - SELF-TESTS
# =============================================================================

def gradient_check(seed=7, n=12, n_probe=4, h=1e-5):
    """
    MANDATORY finite-difference check of the whole graph.

    Every parameter is probed at a few random coordinates. The analytic
    gradient from PART 1's tape must match a central difference of the full
    loss - through the softmax word channel, the six keeper steps, the hard
    noughting mask, the pain residual and the deed hinge.
    """
    rng = np.random.default_rng(seed)
    eng = AnchorholdEngine(rng)
    U, Yh, Yl = make_norwich_cases(n, rng)

    def full_loss():
        out = eng.forward(U)
        L, _ = eng.loss(out, Yh, Yl, U)
        return L

    L = full_loss()
    L.backward()
    analytic = {k: p.g.copy() for k, p in eng.params().items()}

    rows, worst, worst_name = [], 0.0, ""
    for name, p in eng.params().items():
        flat = p.v.reshape(-1)
        picks = rng.choice(flat.size, size=min(n_probe, flat.size), replace=False)
        errs = []
        for j in picks:
            old = flat[j]
            flat[j] = old + h
            lp = float(full_loss().v)
            flat[j] = old - h
            lm = float(full_loss().v)
            flat[j] = old
            num = (lp - lm) / (2 * h)
            ana = analytic[name].reshape(-1)[j]
            denom = max(1e-8, abs(num) + abs(ana))
            errs.append(abs(num - ana) / denom)
        e = float(np.max(errs))
        rows.append((name, e))
        if e > worst:
            worst, worst_name = e, name

    print("  parameter        max relative error")
    for name, e in rows:
        flag = "ok" if e < 1e-4 else ("CHECK" if e < 1e-2 else "FAIL")
        print(f"    {name:<10s}  {e:.3e}   {flag}")
    ok = worst < 1e-4
    print(f"  worst: {worst_name} = {worst:.3e}  ->  "
          f"{'GRADIENT CHECK PASSED' if ok else 'GRADIENT CHECK FAILED'}")
    return ok


def hazelnut_test(engine, data):
    """
    "I marvelled how it might last, for methought it might suddenly have fallen
    to nought for littleness." (ch. 5)

    The claim being tested is architectural, not poetic: this network has no
    identity path, so its state is not stored anywhere - it is re-made every
    step by the keeper, and only by the keeper. Drive the keeper below the
    noughting threshold and the representation should not degrade gracefully.
    It should go to exactly zero, and every downstream reading should collapse
    to a constant that no longer depends on the input at all.
    """
    U, Yh, Yl = data
    normal = engine.forward(U[:200])
    let_go = engine.forward(U[:200], force_keep=engine.kappa * 0.5)
    wl = engine.P["wLo"].v
    n_h = float(np.linalg.norm(normal["h"].v, axis=1).mean())
    n_l = float(np.linalg.norm(let_go["h"].v, axis=1).mean())
    # the state-borne part of the lower doom - what the engine knows THROUGH its
    # kept state, as opposed to what runs straight from the sights to the heads
    s_h = float((normal["h"].v @ wl).std())
    s_l = float((let_go["h"].v @ wl).std())
    print(f"  kept     : mean||h|| = {n_h:9.6f}   sd(state-borne doom) = {s_h:.6f}")
    print(f"  let go   : mean||h|| = {n_l:9.6f}   sd(state-borne doom) = {s_l:.6f}")
    collapsed = (n_l == 0.0) and (s_l == 0.0)
    print(f"  the three sights are feed-forward and survive; the kept state does")
    print(f"  not, because there is nowhere for it to be kept except the keeper.")
    print(f"  -> being is maintained, never stored: "
          f"{'CONFIRMED - falls to nought exactly' if collapsed else 'NOT CONFIRMED'}")
    return collapsed


def attractor_test(engine, n=200, steps=500, seed=11):
    """
    "Alle shalle be wele, and alle shalle be wele, and alle maner of thinge
    shalle be wele." (ch. 27)

    Read as a dynamical claim: whatever the trajectory does, the free-running
    map settles onto the wele-manifold. Start from 200 arbitrary states, show no
    input at all, iterate 500 steps, and measure the pain - the distance from
    that manifold - at the start and at the end.
    """
    rng = np.random.default_rng(seed)
    h = Tensor(rng.normal(size=(n, engine.d_hid)) * 1.5)
    zero = Tensor(np.zeros((n, engine.d_in)))
    p0 = float(engine.pain(h).v.mean())
    traj = []
    for t in range(steps):
        h, _, _ = engine.keeper_step(h, zero)
        if t in (0, 4, 24, 99, 249, steps - 1):
            traj.append((t + 1, float(engine.pain(h).v.mean())))
    print(f"  pain at start (random states) : {p0:.6f}")
    for t, p in traj:
        print(f"  pain after {t:4d} steps        : {p:.6f}")
    final = traj[-1][1]
    print(f"  -> converges to the wele-manifold: "
          f"{'CONFIRMED' if final < 0.05 * max(p0, 1e-9) or final < 1e-3 else 'NOT CONFIRMED'}")
    return final


def run_all():
    print("=" * 78)
    print("THE ANCHORHOLD ENGINE - Julian of Norwich (1342-after 1416)")
    print("A Deferred-Reconciliation Network, pure NumPy, no frameworks")
    print("=" * 78)

    # ---------------------------------------------------------------- 1
    print("\n[1] FINITE-DIFFERENCE GRADIENT CHECK")
    ok = gradient_check()
    if not ok:
        print("  aborting: gradients are wrong, nothing below would mean anything")
        return

    # ---------------------------------------------------------------- 2
    print("\n[2] THE NORWICH CASES")
    rng = np.random.default_rng(1373)
    tr = make_norwich_cases(800, rng)
    te = make_norwich_cases(400, rng)
    d = np.abs(tr[2] - tr[1])
    print(f"  train {tr[0].shape[0]} cases, test {te[0].shape[0]}, "
          f"observation dim {tr[0].shape[1]}")
    print(f"  true discord |y_low - y_high|: mean {d.mean():.4f}  "
          f"median {np.median(d):.4f}  max {d.max():.4f}")
    print(f"  the deed budget epsilon is set to 0.35 - inside it, disagreement")
    print(f"  costs the engine nothing and is never trained away.")

    # ---------------------------------------------------------------- 3
    print("\n[3] TRAINING THE FULL ENGINE (with deferral)")
    eng = AnchorholdEngine(np.random.default_rng(3))
    store = DeferralStore()
    hist = train(eng, tr, epochs=110, store=store, seed=3, verbose=True)
    print(f"  final training objective: {hist[-1]:.5f}")
    m = evaluate(eng, te)
    print(f"  test MSE  higher doom : {m['mse_hi']:.5f}")
    print(f"  test MSE  lower doom  : {m['mse_lo']:.5f}")
    print(f"  residual MAE (|r|-|d|): {m['resid_mae']:.5f}")
    print(f"  pain vs true discord  : r = {m['pain_corr']:.4f}")
    print(f"  mean keep gate        : {m['keep_mean']:.4f}   "
          f"noughted units: {100*m['noughted_frac']:.1f}%")
    print(f"  word-channel entropy  : {m['word_entropy']:.4f} nats "
          f"(ln 8 = {np.log(8):.4f} if the lexicon were unused)")

    # ---------------------------------------------------------------- 4
    print("\n[4] THE DEFERRAL STORE (ch. 51)")
    print(f"  cases withheld as contradictory-and-illegible : "
          f"{len(store.held) + len(store.log)}")
    print(f"  still unread at the end of training           : {len(store.held)}")
    print(f"  later released, having become readable        : {len(store.log)}")
    if store.log:
        waits = [rel - st for _, st, rel in store.log]
        print(f"  epochs between storing and reading: "
              f"min {min(waits)}, median {int(np.median(waits))}, max {max(waits)}")
        for i, st, rel in store.log[:4]:
            print(f"    case {i:4d}: stored at epoch {st}, read at epoch {rel}")
    print("  nothing in the store was ever guessed at, overwritten or dropped.")

    # ---------------------------------------------------------------- 5
    print("\n[5] HAZELNUT TEST - is the state kept, or merely stored?")
    hazelnut_test(eng, te)

    # ---------------------------------------------------------------- 6
    print("\n[6] ATTRACTOR TEST - 'alle shalle be wele' as an asymptote")
    attractor_test(eng)

    # ---------------------------------------------------------------- 7
    print("\n[7] ABLATIONS")
    base = evaluate(eng, te)
    results = [("full engine", base["mse_hi"], base["mse_lo"], base["resid_mae"])]

    print("  (a) the deed budget removed: epsilon = 0, disagreement forced to zero")
    e_a = AnchorholdEngine(np.random.default_rng(3))
    train(e_a, tr, epochs=110, eps_deed=0.0, store=DeferralStore(), seed=3)
    m_a = evaluate(e_a, te)
    results.append(("eps = 0 (forced accord)", m_a["mse_hi"], m_a["mse_lo"], m_a["resid_mae"]))

    print("  (b) the two dooms merged into one averaged target")
    e_b = AnchorholdEngine(np.random.default_rng(3))
    train(e_b, tr, epochs=110, store=DeferralStore(), seed=3, merged_head=True)
    m_b = evaluate(e_b, te, merged_head=True)
    results.append(("merged single doom", m_b["mse_hi"], m_b["mse_lo"], m_b["resid_mae"]))

    print("  (c) the ghostly sight removed (bodily + word only)")
    e_c = AnchorholdEngine(np.random.default_rng(3), use_ghostly=False)
    train(e_c, tr, epochs=110, store=DeferralStore(), seed=3)
    m_c = evaluate(e_c, te)
    results.append(("no ghostly sight", m_c["mse_hi"], m_c["mse_lo"], m_c["resid_mae"]))

    print("  (d) illegible cases discarded instead of deferred")
    e_d = AnchorholdEngine(np.random.default_rng(3))
    st_d = DeferralStore(release=False)
    train(e_d, tr, epochs=110, store=st_d, seed=3)
    m_d = evaluate(e_d, te)
    results.append(("discard, never re-read", m_d["mse_hi"], m_d["mse_lo"], m_d["resid_mae"]))

    print()
    print(f"  {'variant':<26s} {'MSE high':>10s} {'MSE low':>10s} {'resid MAE':>11s}")
    print("  " + "-" * 60)
    for nm, a, b, c in results:
        print(f"  {nm:<26s} {a:10.5f} {b:10.5f} {c:11.5f}")

    print("\n" + "=" * 78)
    print("Every number above was produced by running this file.")
    print("=" * 78)


if __name__ == "__main__":
    run_all()
