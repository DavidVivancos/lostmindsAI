"""
================================================================================
 chapter_0006_enheduanna_-2285.py
 Enheduanna (c. 2285 BCE – c. 2250 BCE, Ur)

 Part Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
 How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
 Tome 1 Minds 1 - 20 Available on Amazon https://www.amazon.com/dp/B0H6F9L324
 Resume and Interactive Demos at https://artificiology.com/

================================================================================

 THE TEMPLE OF THE ME
 A conditioned generative language model with built-in provenance.

 --------------------------------------------------------------------------
 WHY THIS ARCHITECTURE, FOR THIS MIND
 --------------------------------------------------------------------------
 Enheduanna is the earliest author in history we can name. As EN (high
 priestess) of the moon-god Nanna at Ur, her job was not to *invent* truth
 but to *receive* the divine order and *re-transmit* it to a human community
 with authority. Three ideas from her world drive this code:

   1. THE ME (pronounced "may", Sumerian 𒈨): the discrete, transferable
      "decrees" of civilisation -- kingship, priesthood, truth, the crafts,
      even "the descent to the underworld". The Sumerians imagined culture
      as an *enumerable instruction set* that could be handed from one holder
      to another. We model that literally: generation is CONDITIONED on an
      explicit, discrete `me` code that modulates the network (FiLM-style
      top-down control), rather than being left implicit in the weights.

   2. AUTHORSHIP / THE SIGNATURE: Enheduanna did the unprecedented -- she
      signed her name inside the text ("I, Enheduanna"). For her, an utterance
      without a known source is not yet civilised speech. We bake that in with
      a PROVENANCE HEAD: from the same hidden state that drives generation, a
      linear probe must recover *which me authored the sequence*. Outputs are
      required to be attributable. Provenance is a trained objective, not an
      afterthought.

   3. WISDOM AS THE EAR (ĝeštug): in Sumerian, understanding is seated in the
      "ear" -- intelligence is reception before it is production. So the model
      is built around conditioning/listening: the `me` reaches DOWN into the
      hidden layer and reshapes it before any token is produced.

 This is NOT a thematic mock-up with frozen random weights. It is a small but
 genuine neural language model:

      embedding -> windowed context -> dense hidden layer
                -> FiLM conditioning by the active `me`  (top-down)
                -> softmax over the vocabulary           (generation head)
                -> softmax over the me-set               (provenance head)

 trained by real backpropagation + Adam on a synthetic liturgical corpus, with
 a numerical gradient check proving the analytic gradients are correct. Run the
 file and the test battery at the bottom executes end to end.

 --------------------------------------------------------------------------
 DEPENDENCIES: numpy only.  RUN:  python3 chapter_0006_enheduanna_-2285.py
 --------------------------------------------------------------------------
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np


# =============================================================================
# 0. SMALL NUMERICAL HELPERS
# =============================================================================

def softmax(z: np.ndarray) -> np.ndarray:
    """Row-wise numerically stable softmax for a [B, K] matrix of logits."""
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def cross_entropy(probs: np.ndarray, targets: np.ndarray) -> float:
    """Mean negative log-likelihood. `probs` is [B, K]; `targets` is [B] of ints."""
    n = probs.shape[0]
    p_correct = probs[np.arange(n), targets]
    return float(-np.mean(np.log(p_correct + 1e-12)))


# =============================================================================
# 1. THE LITURGICAL CORPUS GENERATOR  ("the me-grammar")
# =============================================================================
#
# We do not have Enheduanna's Sumerian on tap, so we build a small *formal
# language* that has the two properties her hymns actually have:
#
#   (a) a FIXED RITUAL SKELETON shared by every hymn -- invocation, praise,
#       a descent, a return, and a sealing colophon (the signature);
#   (b) CONTENT that depends on WHICH me (which divine office) is being sung.
#
# A model that wants low loss must therefore learn the shared structure AND
# specialise its content per me. The `me` is exactly the variable our FiLM
# conditioning is given -- so the architecture is matched to the task by design.

# Structural marker tokens (shared by all hymns, fixed positions / transitions)
MARKERS = ["<INVOKE>", "<PRAISE>", "<DESCENT>", "<RETURN>", "<SEAL>", "<END>"]

# Four "me" (divine offices). Each owns a private set of content words. These
# stand in for the enumerable decrees of civilisation that an EN-priestess
# would arrange: heaven-craft, war-craft, the scribal arts, the harvest.
ME_NAMES = ["AN_HEAVEN", "INANNA_WAR", "NISABA_SCRIBE", "EZINU_HARVEST"]

ME_CONTENT = {
    "AN_HEAVEN":     ["sky", "star", "crown", "throne", "light", "decree"],
    "INANNA_WAR":    ["storm", "lion", "battle", "fury", "standard", "victory"],
    "NISABA_SCRIBE": ["reed", "tablet", "number", "list", "measure", "name"],
    "EZINU_HARVEST": ["grain", "furrow", "rain", "ox", "granary", "bread"],
}


def build_vocab() -> Tuple[Dict[str, int], Dict[int, str]]:
    """Assemble the token<->id maps for markers + every me's content words."""
    toks: List[str] = list(MARKERS)
    for me in ME_NAMES:
        toks.extend(ME_CONTENT[me])
    # de-duplicate while preserving order
    seen, ordered = set(), []
    for t in toks:
        if t not in seen:
            seen.add(t)
            ordered.append(t)
    stoi = {t: i for i, t in enumerate(ordered)}
    itos = {i: t for t, i in stoi.items()}
    return stoi, itos


def make_hymn(me_id: int, stoi: Dict[str, int], rng: np.random.Generator) -> List[int]:
    """
    Produce one hymn as a list of token ids for a given me.

    Ritual skeleton (same every time):
        <INVOKE>  w w  <PRAISE>  w w w  <DESCENT>  w w  <RETURN>  w w w  <SEAL> <END>

    The 'w' content words are drawn ONLY from this me's private word set, so
    the identity of the me is fully encoded in the content -- which is what the
    provenance head must learn to read back out.
    """
    words = ME_CONTENT[ME_NAMES[me_id]]

    def draw(k: int) -> List[int]:
        return [stoi[words[rng.integers(len(words))]] for _ in range(k)]

    seq: List[int] = []
    seq += [stoi["<INVOKE>"]]  + draw(2)
    seq += [stoi["<PRAISE>"]]  + draw(3)
    seq += [stoi["<DESCENT>"]] + draw(2)
    seq += [stoi["<RETURN>"]]  + draw(3)
    seq += [stoi["<SEAL>"], stoi["<END>"]]
    return seq


def build_dataset(
    n_per_me: int,
    ctx: int,
    stoi: Dict[str, int],
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Turn many hymns into supervised next-token examples via a sliding window.

    Returns three aligned arrays:
        X   [N, ctx]  : the `ctx` previous token ids (the context window)
        y   [N]       : the next token id (generation target)
        m   [N]       : the me id that authored the sequence (provenance target)
    """
    rng = np.random.default_rng(seed)
    pad = stoi["<END>"]  # left-pad short contexts with <END>
    X, y, m = [], [], []
    for me_id in range(len(ME_NAMES)):
        for _ in range(n_per_me):
            seq = make_hymn(me_id, stoi, rng)
            padded = [pad] * ctx + seq
            for i in range(ctx, len(padded)):
                X.append(padded[i - ctx:i])
                y.append(padded[i])
                m.append(me_id)
    return np.array(X), np.array(y), np.array(m)


# =============================================================================
# 2. THE MODEL: TempleOfTheMe
# =============================================================================

@dataclass
class TempleConfig:
    vocab_size: int
    n_me: int
    ctx: int = 3          # how many previous tokens the priestess "hears"
    d_emb: int = 24       # token embedding width
    d_hidden: int = 48    # hidden ("reception chamber") width
    prov_weight: float = 0.5   # how much we insist outputs be attributable
    seed: int = 7


class TempleOfTheMe:
    """
    A windowed neural language model whose hidden state is modulated, top-down,
    by the active `me` (FiLM: feature-wise linear modulation), with a second
    head that must recover the me from that same hidden state (provenance).

    Parameter inventory
    -------------------
        E      [V, d_emb]            token embeddings ("the signs on the reed")
        W1,b1  [ctx*d_emb, h], [h]   context -> hidden ("the reception chamber")
        gamma  [n_me, h]             per-me multiplicative modulation  (the me
        beta   [n_me, h]             per-me additive modulation         reaching
                                     DOWN into the chamber -- top-down control)
        W2,b2  [h, V], [V]           hidden -> next-token logits   (generation)
        P,  pb [h, n_me], [n_me]     hidden -> me logits           (provenance)
    """

    def __init__(self, cfg: TempleConfig):
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        d, h, V, M, ctx = cfg.d_emb, cfg.d_hidden, cfg.vocab_size, cfg.n_me, cfg.ctx

        # He-ish / Xavier-ish small inits keep early gradients well behaved.
        self.E     = rng.normal(0, 0.1, (V, d))
        self.W1    = rng.normal(0, np.sqrt(2.0 / (ctx * d)), (ctx * d, h))
        self.b1    = np.zeros(h)
        self.gamma = np.ones((M, h))          # start as identity modulation...
        self.beta  = np.zeros((M, h))         # ...so the me begins "neutral"
        self.W2    = rng.normal(0, np.sqrt(1.0 / h), (h, V))
        self.b2    = np.zeros(V)
        self.P     = rng.normal(0, np.sqrt(1.0 / h), (h, M))
        self.pb    = np.zeros(M)

    # -- the eight learnable tensors, named, for the optimizer & grad-check ----
    def params(self) -> Dict[str, np.ndarray]:
        return {"E": self.E, "W1": self.W1, "b1": self.b1, "gamma": self.gamma,
                "beta": self.beta, "W2": self.W2, "b2": self.b2,
                "P": self.P, "pb": self.pb}

    # ------------------------------------------------------------------ forward
    def forward(self, X: np.ndarray, m: np.ndarray) -> Tuple[Dict, Dict]:
        """
        X [B, ctx] int, m [B] int.  Returns (outputs, cache).
        outputs['p_tok'] [B,V] next-token probs ; outputs['p_me'] [B,M] provenance.
        """
        B = X.shape[0]
        d = self.cfg.d_emb

        emb  = self.E[X]                       # [B, ctx, d]  -- look up the signs
        xcat = emb.reshape(B, -1)              # [B, ctx*d]   -- the heard context
        z1   = xcat @ self.W1 + self.b1        # [B, h]       -- raw reception

        g = self.gamma[m]                      # [B, h]  the me's multiplicative voice
        be = self.beta[m]                      # [B, h]  the me's additive voice
        z1m = g * z1 + be                      # FiLM: the decree reshapes reception
        a1  = np.tanh(z1m)                     # [B, h]  the consecrated hidden state

        tok_logits = a1 @ self.W2 + self.b2    # [B, V]
        me_logits  = a1 @ self.P  + self.pb    # [B, M]
        p_tok = softmax(tok_logits)
        p_me  = softmax(me_logits)

        cache = dict(X=X, m=m, emb=emb, xcat=xcat, z1=z1, g=g, be=be,
                     z1m=z1m, a1=a1)
        return {"p_tok": p_tok, "p_me": p_me}, cache

    # ------------------------------------------------------------------- loss
    def loss(self, out: Dict, y: np.ndarray, m: np.ndarray) -> Tuple[float, Dict]:
        """Combined loss = next-token CE + prov_weight * provenance CE."""
        lm   = cross_entropy(out["p_tok"], y)
        prov = cross_entropy(out["p_me"], m)
        total = lm + self.cfg.prov_weight * prov
        return total, {"lm": lm, "prov": prov, "total": total}

    # --------------------------------------------------------------- backward
    def backward(self, cache: Dict, out: Dict, y: np.ndarray,
                 m: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Analytic gradients for every parameter. Each step is the textbook
        derivative for the layer above it; the only fiddly parts are the
        scatter-adds back into E (because tokens repeat) and into gamma/beta
        (because each me appears in many rows).
        """
        cfg = self.cfg
        B = cache["X"].shape[0]
        a1, z1, g = cache["a1"], cache["z1"], cache["g"]
        xcat = cache["xcat"]

        # ---- generation head: dL/d(tok_logits) = (softmax - onehot)/B --------
        dtok = out["p_tok"].copy()
        dtok[np.arange(B), y] -= 1.0
        dtok /= B
        dW2 = a1.T @ dtok
        db2 = dtok.sum(axis=0)

        # ---- provenance head (scaled by prov_weight) -------------------------
        dme = out["p_me"].copy()
        dme[np.arange(B), m] -= 1.0
        dme /= B
        dme *= cfg.prov_weight
        dP  = a1.T @ dme
        dpb = dme.sum(axis=0)

        # ---- both heads feed back into a1 ------------------------------------
        da1 = dtok @ self.W2.T + dme @ self.P.T          # [B, h]

        # ---- through tanh: dz1m = da1 * (1 - a1^2) ---------------------------
        dz1m = da1 * (1.0 - a1 ** 2)                      # [B, h]

        # ---- through FiLM (z1m = g*z1 + beta) --------------------------------
        dz1   = dz1m * g                                  # to the linear layer
        dgamma_rows = dz1m * z1                            # per-row dgamma
        dbeta_rows  = dz1m                                 # per-row dbeta
        dgamma = np.zeros_like(self.gamma)
        dbeta  = np.zeros_like(self.beta)
        np.add.at(dgamma, m, dgamma_rows)                  # scatter to each me
        np.add.at(dbeta,  m, dbeta_rows)

        # ---- through the dense layer (z1 = xcat@W1 + b1) ---------------------
        dW1 = xcat.T @ dz1
        db1 = dz1.sum(axis=0)
        dxcat = dz1 @ self.W1.T                            # [B, ctx*d]

        # ---- through the embedding lookup ------------------------------------
        d = cfg.d_emb
        demb = dxcat.reshape(B, cfg.ctx, d)                # [B, ctx, d]
        dE = np.zeros_like(self.E)
        np.add.at(dE, cache["X"], demb)                    # tokens repeat -> add

        return {"E": dE, "W1": dW1, "b1": db1, "gamma": dgamma, "beta": dbeta,
                "W2": dW2, "b2": db2, "P": dP, "pb": dpb}

    # ----------------------------------------------------- inference: generate
    def generate(self, me_id: int, stoi: Dict[str, int], itos: Dict[int, str],
                 max_len: int = 24, temperature: float = 0.6,
                 seed: int = 0) -> List[str]:
        """
        Autoregressively sing a hymn for a given me. Greedy-ish sampling with a
        low temperature so the learned ritual skeleton shows through clearly.
        """
        rng = np.random.default_rng(seed)
        ctx = self.cfg.ctx
        pad = stoi["<END>"]
        window = [pad] * ctx
        out_tokens: List[str] = []
        for _ in range(max_len):
            X = np.array(window[-ctx:]).reshape(1, ctx)
            o, _ = self.forward(X, np.array([me_id]))
            logits = np.log(o["p_tok"][0] + 1e-12) / max(temperature, 1e-3)
            p = softmax(logits.reshape(1, -1))[0]
            nxt = int(rng.choice(len(p), p=p))
            tok = itos[nxt]
            out_tokens.append(tok)
            window.append(nxt)
            if tok == "<END>":
                break
        return out_tokens


# =============================================================================
# 3. ADAM OPTIMIZER (from scratch)
# =============================================================================

class Adam:
    """Plain Adam. Keeps one running mean/variance per named parameter tensor."""

    def __init__(self, params: Dict[str, np.ndarray], lr: float = 5e-3,
                 b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]):
        self.t += 1
        for k in params:
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


# =============================================================================
# 4. TRAINING LOOP
# =============================================================================

def train(model: TempleOfTheMe, X: np.ndarray, y: np.ndarray, m: np.ndarray,
          steps: int = 1500, batch: int = 128, lr: float = 5e-3,
          seed: int = 0, log_every: int = 250, verbose: bool = True
          ) -> List[Tuple[int, float, float]]:
    """Mini-batch Adam training. Returns a log of (step, lm_loss, prov_loss)."""
    rng = np.random.default_rng(seed)
    opt = Adam(model.params(), lr=lr)
    N = X.shape[0]
    history: List[Tuple[int, float, float]] = []
    params = model.params()  # live references -> Adam updates in place
    for s in range(1, steps + 1):
        idx = rng.integers(0, N, size=batch)
        out, cache = model.forward(X[idx], m[idx])
        _, parts = model.loss(out, y[idx], m[idx])
        grads = model.backward(cache, out, y[idx], m[idx])
        opt.step(params, grads)
        if verbose and (s % log_every == 0 or s == 1):
            history.append((s, parts["lm"], parts["prov"]))
            print(f"  step {s:5d} | lm-loss {parts['lm']:.4f} "
                  f"| prov-loss {parts['prov']:.4f}")
    return history


def evaluate(model: TempleOfTheMe, X: np.ndarray, y: np.ndarray,
             m: np.ndarray, stoi: Dict[str, int]) -> Dict[str, float]:
    """
    Four metrics, each measuring something the model *can* in principle learn:

      next_token_acc : raw accuracy (capped near ~0.48 because content words are
                       drawn uniformly at random -- irreducible entropy, by design)
      provenance_acc : can the me be read back out of the hidden state? (signature)
      structure_acc  : accuracy ON marker targets only -- did it learn the rite?
      me_vocab_mass  : on content targets, how much probability mass lands on the
                       *correct* me's decreed word-set -- did it 'install the me'?
    """
    out, _ = model.forward(X, m)
    p_tok = out["p_tok"]
    tok_pred = p_tok.argmax(axis=1)
    me_pred = out["p_me"].argmax(axis=1)

    marker_ids = np.array([stoi[t] for t in MARKERS])
    me_word_ids = [np.array([stoi[w] for w in ME_CONTENT[name]]) for name in ME_NAMES]

    is_marker_target = np.isin(y, marker_ids)
    structure_acc = float(np.mean(tok_pred[is_marker_target] == y[is_marker_target]))

    # for each content-target row, sum the prob assigned to that row's me's words
    content_rows = ~is_marker_target
    masses = []
    for i in np.where(content_rows)[0]:
        masses.append(float(p_tok[i, me_word_ids[m[i]]].sum()))
    me_vocab_mass = float(np.mean(masses)) if masses else 1.0

    return {
        "next_token_acc": float(np.mean(tok_pred == y)),
        "provenance_acc": float(np.mean(me_pred == m)),
        "structure_acc": structure_acc,
        "me_vocab_mass": me_vocab_mass,
    }


# =============================================================================
# 5. TEST BATTERY  (this is what makes it "tested, not a demo")
# =============================================================================

def test_gradient_check() -> None:
    """
    Verify analytic gradients against central finite differences on a tiny
    instance. If backprop is wrong, this fails loudly. This is the single most
    important guarantee that the model is a real, trainable network.
    """
    print("\n[TEST 1] numerical gradient check ...")
    stoi, _ = build_vocab()
    cfg = TempleConfig(vocab_size=len(stoi), n_me=len(ME_NAMES),
                       ctx=2, d_emb=6, d_hidden=8, prov_weight=0.5, seed=1)
    model = TempleOfTheMe(cfg)
    rng = np.random.default_rng(3)
    B = 5
    X = rng.integers(0, len(stoi), size=(B, cfg.ctx))
    y = rng.integers(0, len(stoi), size=B)
    m = rng.integers(0, cfg.n_me, size=B)

    out, cache = model.forward(X, m)
    grads = model.backward(cache, out, y, m)

    eps = 1e-5
    worst = 0.0
    params = model.params()
    for name, P in params.items():
        flat = P.ravel()
        # check a handful of random coordinates per tensor (keeps it fast)
        coords = rng.choice(flat.size, size=min(8, flat.size), replace=False)
        for c in coords:
            orig = flat[c]
            flat[c] = orig + eps
            o_plus, _ = model.forward(X, m)
            l_plus, _ = model.loss(o_plus, y, m)
            flat[c] = orig - eps
            o_minus, _ = model.forward(X, m)
            l_minus, _ = model.loss(o_minus, y, m)
            flat[c] = orig
            num = (l_plus - l_minus) / (2 * eps)
            ana = grads[name].ravel()[c]
            denom = max(1e-8, abs(num) + abs(ana))
            rel = abs(num - ana) / denom
            worst = max(worst, rel)
    print(f"         worst relative error = {worst:.2e}")
    assert worst < 1e-4, f"gradient check FAILED (worst rel err {worst:.2e})"
    print("         PASS -- analytic gradients match finite differences.")


def test_learning() -> TempleOfTheMe:
    """Train the full model and assert it actually learned the liturgy."""
    print("\n[TEST 2] training reduces loss & learns structure + provenance ...")
    stoi, itos = build_vocab()
    Xtr, ytr, mtr = build_dataset(n_per_me=300, ctx=5, stoi=stoi, seed=0)
    Xte, yte, mte = build_dataset(n_per_me=60,  ctx=5, stoi=stoi, seed=999)

    cfg = TempleConfig(vocab_size=len(stoi), n_me=len(ME_NAMES), ctx=5,
                       d_emb=24, d_hidden=48, prov_weight=0.5, seed=7)
    model = TempleOfTheMe(cfg)

    out0, _ = model.forward(Xtr, mtr)
    _, parts0 = model.loss(out0, ytr, mtr)
    init_lm = parts0["lm"]
    print(f"         initial lm-loss = {init_lm:.4f}")

    train(model, Xtr, ytr, mtr, steps=1500, batch=128, lr=5e-3, seed=1)

    metrics = evaluate(model, Xte, yte, mte, stoi)
    out1, _ = model.forward(Xtr, mtr)
    _, parts1 = model.loss(out1, ytr, mtr)
    final_lm = parts1["lm"]
    print(f"         final lm-loss   = {final_lm:.4f}")
    print(f"         held-out next-token accuracy = {metrics['next_token_acc']:.3f} "
          f"(content is random by design; ceiling ~0.48)")
    print(f"         held-out STRUCTURE accuracy  = {metrics['structure_acc']:.3f} "
          f"(markers)")
    print(f"         held-out me-vocab mass       = {metrics['me_vocab_mass']:.3f} "
          f"(content drawn from the right me)")
    print(f"         held-out provenance accuracy = {metrics['provenance_acc']:.3f}")

    assert final_lm < 0.5 * init_lm, "loss did not fall enough"
    assert metrics["structure_acc"] > 0.95, "did not learn the ritual skeleton"
    assert metrics["me_vocab_mass"] > 0.80, "did not install the active me's vocabulary"
    assert metrics["provenance_acc"] > 0.90, "outputs are not attributable"
    print("         PASS -- the temple learned the rite and can sign its work.")
    return model


def test_generation(model: TempleOfTheMe) -> None:
    """Sing one hymn per me and check the ritual skeleton is reproduced."""
    print("\n[TEST 3] generation reproduces the ritual skeleton ...")
    stoi, itos = build_vocab()
    required = ["<INVOKE>", "<PRAISE>", "<DESCENT>", "<RETURN>", "<SEAL>", "<END>"]
    for me_id, me_name in enumerate(ME_NAMES):
        toks = model.generate(me_id, stoi, itos, seed=me_id + 1)
        present = [r for r in required if r in toks]
        # check the markers that appear do so in the canonical order
        order_ok = present == [r for r in required if r in present]
        print(f"         {me_name:<14} -> {' '.join(toks)}")
        assert "<INVOKE>" in toks, "hymn never opened with an invocation"
        assert order_ok, "ritual markers came out of order"
    print("         PASS -- every me is sung in the correct liturgical order.")


def test_film_ablation() -> None:
    """
    Show the me-conditioning EARNS its place: a model with FiLM frozen to the
    identity (no top-down control of the hidden state) cannot specialise content
    per me and ends with a clearly higher loss. This is the empirical argument
    that the 'me reaching down into the chamber' is doing real work.
    """
    print("\n[TEST 4] ablation: FiLM conditioning lowers loss ...")
    stoi, _ = build_vocab()
    Xtr, ytr, mtr = build_dataset(n_per_me=300, ctx=5, stoi=stoi, seed=0)

    def run(condition: bool) -> float:
        cfg = TempleConfig(vocab_size=len(stoi), n_me=len(ME_NAMES), ctx=5,
                           d_emb=24, d_hidden=48, prov_weight=0.0, seed=7)
        model = TempleOfTheMe(cfg)
        rng = np.random.default_rng(1)
        opt = Adam(model.params(), lr=5e-3)
        params = model.params()
        for _ in range(1200):
            idx = rng.integers(0, Xtr.shape[0], size=128)
            out, cache = model.forward(Xtr[idx], mtr[idx])
            grads = model.backward(cache, out, ytr[idx], mtr[idx])
            if not condition:               # freeze the me's voice to identity
                grads["gamma"][:] = 0.0
                grads["beta"][:] = 0.0
            opt.step(params, grads)
        out, _ = model.forward(Xtr, mtr)
        return cross_entropy(out["p_tok"], ytr)

    with_film = run(True)
    without   = run(False)
    print(f"         lm-loss  with FiLM = {with_film:.4f}")
    print(f"         lm-loss  no  FiLM = {without:.4f}")
    assert with_film < without - 0.02, "FiLM did not help -- ablation inconclusive"
    print("         PASS -- top-down me-conditioning measurably improves the model.")


# =============================================================================
# 6. ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    print("=" * 78)
    print(" THE TEMPLE OF THE ME  ·  Enheduanna (c. 2285 BCE)  ·  Mind #6")
    print(" a conditioned generative language model with built-in provenance")
    print("=" * 78)

    test_gradient_check()
    trained = test_learning()
    test_generation(trained)
    test_film_ablation()

    print("\n" + "=" * 78)
    print(" ALL TESTS PASSED.")
    print(" The model: receives a context (the 'ear'), is reshaped top-down by")
    print(" the active me, generates the next sign, and can name its own author.")
    print(" Reception, conditioning, generation, signature -- the priestess in code.")
    print("=" * 78)
