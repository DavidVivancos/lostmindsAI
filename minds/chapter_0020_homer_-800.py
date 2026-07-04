#!/usr/bin/env python3
# =============================================================================
#  chapter_0020_homer_-800.py  —  THE METIS ENGINE
#  Chapter 20 · Homer (c. 800 BCE) · an oral-formulaic generator in pure NumPy
# Part Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 1 Minds 1 - 20 Available on Amazon https://www.amazon.com/dp/B0H6F9L324
# Resume and Interactive Demos at https://artificiology.com/
# =============================================================================
#
#  THE ONE IDEA THIS FILE EMBODIES
#  -------------------------------
#  Homer gives us two pictures of a capable mind that turn out to be a single
#  picture seen from two sides:
#
#    * METIS — the cunning of Odysseus, "the man of many turns" (polytropos),
#      who never wins by force (BIE) but by finding the one move that FITS a
#      situation he cannot change: Nobody / Outis in the Cyclops' cave, the
#      bow no suitor can string, the Wooden Horse. He wins *inside* the
#      constraint, never by smashing it.
#
#    * ORAL-FORMULAIC COMPOSITION — Milman Parry and Albert Lord showed that
#      the Iliad and Odyssey were not memorised texts but *composed in
#      performance*: the singer fills each slot of the dactylic-hexameter line
#      from a deep store of ready-made formulae ("swift-footed Achilles",
#      "wine-dark sea") whose metrical shapes fit the slot exactly. The metre
#      is an inviolable external constraint; intelligence is the art of the
#      legal move that also carries the story forward.
#
#  These are the SAME cognitive act: improvisation under a hard constraint by
#  recombining a finite store of units. So the architecture is built around a
#  HARD METRICAL GATE. At every step the foot-budget of the line masks out
#  every formula that does not fit the feet that remain. The network can choose
#  *which* legal move to make — but it physically cannot make an illegal one.
#  Metis is the gate; biE (brute force) is what the gate forbids.
#
#  WHY NOT A TRANSFORMER?  A transformer attends softly over stored keys and
#  can say anything; nothing is forbidden, only down-weighted. Homer's mind is
#  the opposite: the constraint is absolute and non-negotiable, and *that* is
#  where the intelligence lives. So this is a recurrent generator (theme =
#  narrative momentum carried in a hidden state) wrapped in a hard combinatorial
#  gate (the metre). Everything is from-scratch NumPy: a finite-difference
#  gradient check, a real training loop (Adam), composition-by-sampling, and a
#  head-to-head test of METIS (gated) vs BIE (ungated) generation.
#
#  RUN:  python3 chapter_0020_homer_-800.py
# Author: David Vivancos · Chapter 0020 · Homer
# =============================================================================

import numpy as np

# -----------------------------------------------------------------------------
# 1.  METRICAL PRIMITIVES  — the constraint that cannot be argued with.
# -----------------------------------------------------------------------------
# A Homeric line is dactylic hexameter: exactly SIX feet. Each foot is a dactyl
# (— u u) or a spondee (— —). We abstract a "foot" to one metrical unit of
# budget. Every formula occupies a fixed integer number of feet (its COST).
# A line is complete when its formulae sum to exactly six feet — no more, no
# less. This budget *is* the metre as far as the network is concerned.
LINE_FEET = 6

# Each entry: (English gloss of the formula, foot-cost, semantic role tag).
# The gloss stands in for a Greek noun-epithet formula; the foot-cost is its
# fixed metrical shape. Several formulae share a cost and a role — these are
# the thrifty substitution sets Parry described (one slot, interchangeable
# fillers), and they are what gives the network a real choice to learn.
LEXICON = [
    ("[and then]",                  1, "FILL"),     # 0
    ("swift-footed Achilles",       2, "HERO"),     # 1
    ("resourceful Odysseus",        2, "HERO"),     # 2
    ("Hector, tamer-of-horses",     3, "HERO"),     # 3
    ("rosy-fingered Dawn",          2, "DAWN"),     # 4
    ("the wine-dark sea",           2, "SEA"),      # 5
    ("the hollow ships",            2, "SHIP"),     # 6
    ("grey-eyed Athena",            2, "GOD"),      # 7
    ("cloud-gathering Zeus",        2, "GOD"),      # 8
    ("spoke winged words",          3, "SPEECH"),   # 9
    ("raged in his heart",          2, "PASSION"),  # 10
    ("devised a cunning plan",      3, "METIS"),    # 11
    ("Sing, O Muse",                2, "INVOKE"),   # 12
    ("of the anger",                1, "THEME"),    # 13
    ("of the homecoming",           2, "THEME"),    # 14
    ("over the loud-roaring deep",  3, "SEA"),      # 15
    ("when Dawn appeared",          2, "DAWN"),     # 16
    ("bright-helmed and shining",   2, "EPITHET"),  # 17
    ("godlike and tireless",        2, "EPITHET"),  # 18
    ("then answered",               1, "SPEECH"),   # 19
]
GLOSS = [x[0] for x in LEXICON]
COST  = np.array([x[1] for x in LEXICON], dtype=np.int64)
ROLE  = [x[2] for x in LEXICON]
V     = len(LEXICON)

# -----------------------------------------------------------------------------
# 2.  TYPE-SCENES  — the macro-structure the singer recombines.
# -----------------------------------------------------------------------------
# Lord's "themes" / "type-scenes" are recurrent narrative blocks (an arming, a
# voyage, a speech-and-answer). Each template below is a short scene whose every
# line sums to exactly six feet. The training corpus is built by sampling these
# scenes; the network learns the transition structure (what tends to follow
# what) while the gate guarantees the metre.
TEMPLATES = [
    [[12, 13, 1, 0], [2, 10, 0, 0]],   # PROEM   : Sing O Muse / of anger / Achilles / ... ; Odysseus / raged / ...
    [[16, 5, 6],     [4, 6, 0, 0]],    # VOYAGE  : when Dawn / sea / ships ; Dawn / ships / ...
    [[2, 9, 0],      [19, 9, 0, 0]],   # SPEECH  : Odysseus / winged words / ; then answered / winged words / ...
    [[2, 11, 0],     [1, 11, 0]],      # METIS   : Odysseus / cunning plan / ; Achilles / cunning plan / ...
    [[6, 15, 0],     [5, 6, 0, 0]],    # SEA     : ships / loud-roaring deep / ; sea / ships / ...
    [[7, 17, 0, 0],  [8, 18, 6]],      # GODS    : Athena / bright-helmed / ; Zeus / godlike / ships
]

def _check_templates():
    for ti, scene in enumerate(TEMPLATES):
        for li, line in enumerate(scene):
            s = int(sum(COST[i] for i in line))
            assert s == LINE_FEET, f"template {ti} line {li} = {s} feet, not 6"
_check_templates()

def build_corpus(n_songs=240, seed=0):
    """Each 'song' is a flat sequence of formula indices forming valid 6-foot
    lines. Returns a list of 1-D int arrays."""
    rng = np.random.default_rng(seed)
    songs = []
    for _ in range(n_songs):
        scene = TEMPLATES[rng.integers(len(TEMPLATES))]
        seq = [i for line in scene for i in line]
        songs.append(np.array(seq, dtype=np.int64))
    return songs

# -----------------------------------------------------------------------------
# 3.  THE MODEL  — a recurrent generator behind a hard metrical gate.
# -----------------------------------------------------------------------------
def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()

class MetisEngine:
    """
    State h_t  : the 'theme' — narrative momentum carried across the line/song.
    At step t:
        q       = Wq h_t                       (state -> a query in formula space)
        logits  = E q + b                       (preference for each formula)
        logits += mask(remaining_feet)          (the METIS GATE: illegal -> -inf)
        p       = softmax(logits)               (choose among LEGAL moves only)
    Then we spend the foot-cost of the chosen formula, reset the budget at the
    end of a line, and advance the state with the emitted formula's embedding.
    """
    def __init__(self, d=16, H=24, seed=0):
        rng = np.random.default_rng(seed); s = 0.3
        self.d, self.H = d, H
        self.E   = rng.normal(0, s, (V, d))   # formula embeddings
        self.Wq  = rng.normal(0, s, (d, H))   # state -> query (d <- H)
        self.b   = np.zeros(V)                # formula bias
        self.Whh = rng.normal(0, s, (H, H))   # recurrence (theme momentum)
        self.Whx = rng.normal(0, s, (H, d))   # emitted formula -> state
        self.bh  = np.zeros(H)
        self.h0  = np.zeros(H)                # learned initial theme

    def params(self):
        return {'E': self.E, 'Wq': self.Wq, 'b': self.b,
                'Whh': self.Whh, 'Whx': self.Whx, 'bh': self.bh, 'h0': self.h0}

    @staticmethod
    def mask(r):
        """The metis gate: any formula whose foot-cost exceeds the feet that
        remain in the current line is forbidden (logit -> -inf)."""
        m = np.zeros(V)
        m[COST > r] = -np.inf
        return m

    # ---- forward: teacher-forced next-formula loss with budget masking -------
    def forward(self, seq, cache=False):
        h = self.h0.copy(); r = LINE_FEET; loss = 0.0; T = len(seq)
        hs = [h.copy()]; qs = []; ps = []
        for t in range(T):
            q = self.Wq @ h
            logits = self.E @ q + self.b + self.mask(r)
            p = softmax(logits); y = int(seq[t])
            loss += -np.log(p[y] + 1e-12)
            qs.append(q); ps.append(p)
            r -= int(COST[y])
            if r == 0: r = LINE_FEET
            h = np.tanh(self.Whh @ h + self.Whx @ self.E[y] + self.bh)
            hs.append(h.copy())
        return loss / T, (hs, qs, ps)

    # ---- backward: hand-written BPTT -----------------------------------------
    def backward(self, seq, cache):
        hs, qs, ps = cache
        T = len(seq)
        g = {k: np.zeros_like(v) for k, v in self.params().items()}
        dh_next = np.zeros(self.H)
        for t in reversed(range(T)):
            h_t, q, p, y = hs[t], qs[t], ps[t], int(seq[t])
            dlogits = p.copy(); dlogits[y] -= 1.0; dlogits /= T   # masked -> 0
            g['b'] += dlogits
            g['E'] += np.outer(dlogits, q)
            dq = self.E.T @ dlogits
            g['Wq'] += np.outer(dq, h_t)
            dh = self.Wq.T @ dq + dh_next
            if t >= 1:
                dpre = dh * (1.0 - h_t * h_t)              # tanh'(pre_{t-1})
                g['Whh'] += np.outer(dpre, hs[t-1])
                g['Whx'] += np.outer(dpre, self.E[int(seq[t-1])])
                g['bh']  += dpre
                g['E'][int(seq[t-1])] += self.Whx.T @ dpre
                dh_next = self.Whh.T @ dpre
            else:
                g['h0'] += dh
                dh_next = np.zeros(self.H)
        return g

    # ---- composition-in-performance: sample a song ---------------------------
    def compose(self, n_lines=4, seed=0, gated=True, temp=0.7):
        """Generate a song. gated=True -> METIS (legal moves only, metre is
        guaranteed). gated=False -> BIE (ignore the gate; brute force)."""
        rng = np.random.default_rng(seed)
        h = self.h0.copy(); r = LINE_FEET
        lines, cur, cur_feet = [], [], 0
        guard = 0
        while len(lines) < n_lines and guard < 400:
            guard += 1
            q = self.Wq @ h
            logits = (self.E @ q + self.b) / temp
            if gated:
                logits = logits + self.mask(r)
            p = softmax(logits)
            y = int(rng.choice(V, p=p))
            cur.append(y); cur_feet += int(COST[y])
            h = np.tanh(self.Whh @ h + self.Whx @ self.E[y] + self.bh)
            if gated:
                r -= int(COST[y])
                if r == 0:
                    lines.append(cur); cur, cur_feet = [], 0; r = LINE_FEET
            else:
                # ungated: a line "closes" once feet >= 6; valid only if == 6
                if cur_feet >= LINE_FEET:
                    lines.append(cur); cur, cur_feet = [], 0
        return lines

# -----------------------------------------------------------------------------
# 4.  TRAINING  — Adam over the corpus.
# -----------------------------------------------------------------------------
class Adam:
    def __init__(self, params, lr=0.02, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0
    def step(self, params, grads):
        self.t += 1
        for k in params:
            self.m[k] = self.b1*self.m[k] + (1-self.b1)*grads[k]
            self.v[k] = self.b2*self.v[k] + (1-self.b2)*(grads[k]**2)
            mhat = self.m[k]/(1-self.b1**self.t)
            vhat = self.v[k]/(1-self.b2**self.t)
            params[k] -= self.lr*mhat/(np.sqrt(vhat)+self.eps)

def train(model, corpus, epochs=60, lr=0.02, log_every=10, seed=0):
    opt = Adam(model.params(), lr=lr)
    rng = np.random.default_rng(seed)
    history = []
    for ep in range(epochs):
        order = rng.permutation(len(corpus))
        ep_loss = 0.0
        for i in order:
            seq = corpus[i]
            loss, cache = model.forward(seq, cache=True)
            grads = model.backward(seq, cache)
            opt.step(model.params(), grads)
            ep_loss += loss
        ep_loss /= len(corpus)
        history.append(ep_loss)
        if ep % log_every == 0 or ep == epochs-1:
            print(f"  epoch {ep:3d}   mean loss {ep_loss:.4f}   "
                  f"perplexity {np.exp(ep_loss):.2f}")
    return history

# -----------------------------------------------------------------------------
# 5.  TESTS & DEMO
# -----------------------------------------------------------------------------
def gradient_check():
    """Finite-difference vs analytic gradient (mandatory)."""
    m = MetisEngine(d=8, H=10, seed=1)
    seq = build_corpus(1, seed=3)[0]
    _, cache = m.forward(seq, cache=True)
    g = m.backward(seq, cache)
    eps = 1e-5; maxrel = 0.0; rng = np.random.default_rng(7)
    for name, P in m.params().items():
        flat = P.ravel()
        for i in rng.choice(flat.size, size=min(8, flat.size), replace=False):
            old = flat[i]
            flat[i] = old + eps; Lp, _ = m.forward(seq)
            flat[i] = old - eps; Lm, _ = m.forward(seq)
            flat[i] = old
            num = (Lp - Lm) / (2*eps)
            ana = g[name].ravel()[i]
            maxrel = max(maxrel, abs(num-ana)/max(1e-9, abs(num)+abs(ana)))
    return maxrel

def feet_of_line(line):
    return int(sum(COST[i] for i in line))

def fraction_metrical(lines):
    if not lines: return 0.0
    return np.mean([feet_of_line(l) == LINE_FEET for l in lines])

def render(lines):
    out = []
    for l in lines:
        out.append("    " + " | ".join(GLOSS[i] for i in l) +
                    f"    ({feet_of_line(l)} feet)")
    return "\n".join(out)

def main():
    np.set_printoptions(precision=3, suppress=True)
    print("="*70)
    print("THE METIS ENGINE  —  composition under an inviolable metre")
    print("="*70)

    print("\n[1] GRADIENT CHECK (analytic BPTT vs finite differences)")
    rel = gradient_check()
    print(f"    max relative error = {rel:.2e}   "
          f"-> {'PASS' if rel < 1e-4 else 'FAIL'}")
    assert rel < 1e-4, "gradient check failed"

    print("\n[2] TRAINING the singer on the type-scenes")
    corpus = build_corpus(n_songs=240, seed=0)
    model = MetisEngine(d=16, H=24, seed=0)
    hist = train(model, corpus, epochs=60, lr=0.02, log_every=15)
    best = float(np.min(hist))
    print(f"    loss fell {hist[0]:.3f} -> {best:.3f} (best)  "
          f"(perplexity {np.exp(hist[0]):.2f} -> {np.exp(best):.2f}); the gate"
          f"\n    already removes most uncertainty, so the headroom is small by design")
    assert best < hist[0] - 0.15, "training did not reduce loss"

    print("\n[3] METIS vs BIE  —  metre kept inside vs broken by force")
    metis = [model.compose(n_lines=8, seed=s, gated=True)  for s in range(40)]
    bie   = [model.compose(n_lines=8, seed=s, gated=False) for s in range(40)]
    metis_lines = [l for song in metis for l in song]
    bie_lines   = [l for song in bie   for l in song]
    fm = fraction_metrical(metis_lines)
    fb = fraction_metrical(bie_lines)
    print(f"    METIS (gated)  : {100*fm:5.1f}% of lines are valid hexameter "
          f"(guaranteed by the gate)")
    print(f"    BIE   (ungated): {100*fb:5.1f}% of lines are valid hexameter "
          f"(force ignores the metre)")
    assert fm == 1.0, "gated generation must be 100% metrical"
    assert fb < fm, "ungated 'brute force' should be worse"

    print("\n[4] A SONG COMPOSED IN PERFORMANCE (gated, never seen in training)")
    print(render(model.compose(n_lines=5, seed=123, gated=True, temp=0.6)))

    print("\n" + "="*70)
    print("All tests passed. Metre is the gate; intelligence is the legal move.")
    print("="*70)

if __name__ == "__main__":
    main()
