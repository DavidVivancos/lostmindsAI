"""
================================================================================
Chapter 136 - Nagarjuna (c. 150-250 CE)
The Sunyata Relational Engine: a from-scratch neural architecture in which
NOTHING has an inherent representation of its own.
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 7 Minds 121 - 140 Available on Amazon https://www.amazon.com/dp/B0HFN6GXMH
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 136: Nagarjuna (c. 150-250 CE)
================================================================================   

WHY THIS ARCHITECTURE IS *NAGARJUNA* AND NOT A GENERIC GRAPH NET
----------------------------------------------------------------
Nagarjuna's single most radical claim is *nihsvabhava*: no phenomenon carries
"svabhava" - own-being, an intrinsic essence it would possess independently of
everything else. A thing is only what its conditions make it; strip the
relations away and there is no residue left over. Emptiness (sunyata) is just
another name for this: the absence of self-standing essence. And, famously,
emptiness = dependent origination (pratityasamutpada). A mind that took this
seriously would refuse, at the level of mechanism, to store a fixed vector
"for" any entity.

Almost every modern network violates this on line one. A Transformer keeps an
embedding table: each token owns a private vector - a literal svabhava - looked
up before any context is seen. This architecture deliberately removes that
table. Every node in a graph begins life with the *identical, non-learnable
seed* (content-free, the same for all nodes and all graphs). A node acquires an
identity ONLY through message passing over its relations. Two nodes that sit in
different relational positions end up different; two nodes in identical
positions remain identical. Identity is 100% dependently originated. That is
the thesis, made into code.

On top of that svabhava-free substrate sit two further mind-specific parts:

  * A CATUSKOTI HEAD. Nagarjuna reasons with the tetralemma (four corners):
    a proposition may be affirmed, denied, both, or neither. So the readout is
    not a binary/true-false gate but a genuine 4-way head over
    {IS, IS-NOT, BOTH, NEITHER}. The task itself is defined so that the correct
    corner is fixed *purely by relational structure* (see below), never by any
    intrinsic property of the node - because there are no intrinsic properties.

  * A TWO-TRUTHS LOSS. Conventional truth (samvrti) = task accuracy: the net
    must actually work in the world. Ultimate truth (paramartha) = an
    "emptiness regularizer" that forbids representations from hardening into
    large, self-standing attractors (reified essences). The trade-off weight
    lambda_e IS the Middle Way: too little and reps reify (eternalism); too
    much and they collapse to nothing (nihilism). Learning is walking that edge.

WHAT THE TASK PROVES
--------------------
For a queried node we must name its koti from (has_incoming, has_outgoing):
    (in & out) -> BOTH     : arises from conditions AND conditions others
    (in only)  -> IS        : dependently arisen
    (out only) -> IS-NOT    : posits itself as an uncaused source -> the very
                              svabhava Nagarjuna denies -> its "existence" is
                              rejected
    (neither)  -> NEITHER   : isolated / unconditioned limit case
This label is *unknowable* from a node in isolation. A "svabhava baseline"
(zero message-passing rounds) can therefore do no better than guessing the
majority corner; the relational engine, seeing only relations, solves it. The
gap between them is emptiness paying rent.

ENGINEERING CONVENTIONS
-----------------------
Pure NumPy, hand-written forward and backward passes, a finite-difference
gradient check that must pass, a real training loop, and self-tests. No deep
learning framework is used or needed.
================================================================================
"""

import numpy as np

RNG = np.random.default_rng(137)  # Nagarjuna is figure 137

# Four corners of the tetralemma (catuskoti). Order is fixed and referenced by
# the labelling rule and by every readout in the model.
KOTI = ["IS", "IS_NOT", "BOTH", "NEITHER"]
IS, IS_NOT, BOTH, NEITHER = 0, 1, 2, 3


# ------------------------------------------------------------------------------
# 1. THE WORLD: random dependency graphs whose queried-node label is purely
#    relational. This is our stand-in for "phenomena arising in dependence".
# ------------------------------------------------------------------------------
def make_graph(n_min=4, n_max=9, p_edge=0.28):
    """
    Build one random directed graph and pick one node to query.

    Returns
    -------
    A : (N, N) float adjacency, A[i, j] = 1 means edge i -> j ("i conditions j")
    q : int, index of the queried node
    y : int, the correct catuskoti corner for node q (see module docstring)
    """
    n = int(RNG.integers(n_min, n_max + 1))
    A = (RNG.random((n, n)) < p_edge).astype(np.float64)
    np.fill_diagonal(A, 0.0)  # no self-loops: nothing conditions itself alone
    q = int(RNG.integers(0, n))

    has_in = A[:, q].sum() > 0    # does anything point INTO q? (q has a cause)
    has_out = A[q, :].sum() > 0   # does q point OUT? (q conditions something)
    if has_in and has_out:
        y = BOTH
    elif has_in and not has_out:
        y = IS
    elif has_out and not has_in:
        y = IS_NOT
    else:
        y = NEITHER
    return A, q, y


def normalized_adjacencies(A):
    """
    Row-normalized incoming/outgoing neighbour operators.

    Nout @ H  = mean over the nodes that q points to  (q's effects)
    Nin  @ H  = mean over the nodes that point to q   (q's conditions)

    Row-normalization keeps message magnitudes bounded regardless of degree,
    so that "how many relations" never smuggles in a hidden intrinsic scalar.
    """
    out_deg = A.sum(axis=1, keepdims=True)
    in_deg = A.sum(axis=0, keepdims=True).T
    Nout = A / np.where(out_deg > 0, out_deg, 1.0)
    Nin = A.T / np.where(in_deg > 0, in_deg, 1.0)
    return Nin, Nout


# ------------------------------------------------------------------------------
# 2. THE MODEL: SunyataRelationalEngine
# ------------------------------------------------------------------------------
class SunyataRelationalEngine:
    """
    A message-passing engine with NO per-node embedding table. Every node starts
    from one shared, fixed, non-learnable seed; identity is produced only by
    T rounds of relational update. A catuskoti (4-way) head reads the queried
    node. Weights are shared across rounds (the same "law of dependent
    origination" applies at every step).
    """

    def __init__(self, dim=16, rounds=3, seed=137):
        self.dim = dim
        self.rounds = rounds
        r = np.random.default_rng(seed)
        s = 1.0 / np.sqrt(dim)

        # The seed is FIXED, not a parameter: no node owns an inherent vector.
        # A small constant pattern breaks symmetry between feature channels
        # without encoding anything about any particular node.
        self.seed = np.linspace(-0.1, 0.1, dim)

        # Learnable parameters - all of them relational transforms, none of them
        # node identities.
        self.P = {
            "Wself": r.standard_normal((dim, dim)) * s,   # carry own state
            "Win":  r.standard_normal((dim, dim)) * s,    # transform conditions
            "Wout": r.standard_normal((dim, dim)) * s,    # transform effects
            "b":    np.zeros(dim),
            "Whead": r.standard_normal((4, dim)) * s,     # catuskoti readout
            "bhead": np.zeros(4),
        }

    # ---- forward ----------------------------------------------------------
    def forward(self, A, q, cache=False):
        """Return catuskoti logits (4,) for node q; optionally cache for backprop."""
        P = self.P
        Nin, Nout = normalized_adjacencies(A)
        N = A.shape[0]
        H = np.tile(self.seed, (N, 1))  # every node: identical empty seed

        Hs, Zs = [H], []
        for _ in range(self.rounds):
            agg_in = Nin @ H
            agg_out = Nout @ H
            Z = H @ P["Wself"].T + agg_in @ P["Win"].T + agg_out @ P["Wout"].T + P["b"]
            H = np.tanh(Z)
            Zs.append(Z)
            Hs.append(H)

        hq = H[q]
        logits = P["Whead"] @ hq + P["bhead"]
        if cache:
            self._cache = dict(A=A, q=q, Nin=Nin, Nout=Nout, Hs=Hs, Zs=Zs, hq=hq)
        return logits

    # ---- loss (two truths) ------------------------------------------------
    def loss(self, A, q, y, lambda_e=0.02):
        """
        Total = conventional (cross-entropy on the correct corner)
              + lambda_e * ultimate (emptiness reg: mean squared magnitude of
                the final relational states, penalizing self-standing essence).
        """
        logits = self.forward(A, q, cache=True)
        z = logits - logits.max()
        p = np.exp(z) / np.exp(z).sum()
        conv = -np.log(p[y] + 1e-12)

        Hfin = self._cache["Hs"][-1]
        empt = np.mean(Hfin ** 2)               # ultimate-truth term
        self._cache["p"] = p
        self._cache["y"] = y
        self._cache["lambda_e"] = lambda_e
        return conv + lambda_e * empt, conv, empt, p

    # ---- backward ---------------------------------------------------------
    def backward(self):
        """Hand-written reverse-mode gradients for every learnable parameter."""
        c = self._cache
        P = self.P
        Nin, Nout, Hs, Zs = c["Nin"], c["Nout"], c["Hs"], c["Zs"]
        q, y, p, lam = c["q"], c["y"], c["p"], c["lambda_e"]
        N = Hs[0].shape[0]
        T = self.rounds

        g = {k: np.zeros_like(v) for k, v in P.items()}

        # dL/dlogits from softmax cross-entropy
        dlogits = p.copy()
        dlogits[y] -= 1.0

        hq = c["hq"]
        g["Whead"] += np.outer(dlogits, hq)
        g["bhead"] += dlogits

        # gradient into the final H, from the head (only node q) ...
        dH = np.zeros_like(Hs[-1])
        dH[q] += P["Whead"].T @ dlogits
        # ... and from the emptiness regularizer (all nodes, all channels)
        dH += lam * (2.0 / Hs[-1].size) * Hs[-1]

        # walk the shared-weight recurrence backwards
        for t in reversed(range(T)):
            H_prev = Hs[t]
            Z = Zs[t]
            dZ = dH * (1.0 - np.tanh(Z) ** 2)   # tanh'

            g["b"] += dZ.sum(axis=0)
            g["Wself"] += dZ.T @ H_prev
            agg_in = Nin @ H_prev
            agg_out = Nout @ H_prev
            g["Win"] += dZ.T @ agg_in
            g["Wout"] += dZ.T @ agg_out

            # propagate to H_prev through the three paths (self, in, out)
            dH_prev = dZ @ P["Wself"]
            dH_prev += Nin.T @ (dZ @ P["Win"])
            dH_prev += Nout.T @ (dZ @ P["Wout"])
            dH = dH_prev

        return g

    # ---- utilities --------------------------------------------------------
    def predict(self, A, q):
        return int(np.argmax(self.forward(A, q)))

    def get_flat(self):
        return np.concatenate([v.ravel() for v in self.P.values()])

    def set_flat(self, vec):
        i = 0
        for k, v in self.P.items():
            n = v.size
            self.P[k] = vec[i:i + n].reshape(v.shape).copy()
            i += n

    def grad_flat(self, g):
        return np.concatenate([g[k].ravel() for k in self.P])


# ------------------------------------------------------------------------------
# 3. GRADIENT CHECK (mandatory) - central finite differences vs analytic grads.
# ------------------------------------------------------------------------------
def gradient_check(eps=1e-6, tol=1e-5):
    model = SunyataRelationalEngine(dim=8, rounds=2, seed=7)
    A, q, y = make_graph()
    # ensure a non-trivial graph
    while A.sum() == 0:
        A, q, y = make_graph()

    model.loss(A, q, y)
    g = model.backward()
    ana = model.grad_flat(g)

    theta0 = model.get_flat()
    num = np.zeros_like(theta0)
    for i in range(theta0.size):
        tp = theta0.copy(); tp[i] += eps
        model.set_flat(tp); lp, *_ = model.loss(A, q, y)
        tm = theta0.copy(); tm[i] -= eps
        model.set_flat(tm); lm, *_ = model.loss(A, q, y)
        num[i] = (lp - lm) / (2 * eps)
    model.set_flat(theta0)

    rel = np.linalg.norm(ana - num) / (np.linalg.norm(ana) + np.linalg.norm(num) + 1e-12)
    print(f"[grad-check] params={theta0.size}  relative-error={rel:.3e}  "
          f"-> {'PASS' if rel < tol else 'FAIL'}")
    return rel < tol


# ------------------------------------------------------------------------------
# 4. TRAINING LOOP - plain SGD with momentum over freshly sampled graphs.
# ------------------------------------------------------------------------------
def train(model, steps=4000, lr=0.05, mom=0.9, lambda_e=0.02, log_every=500):
    vel = {k: np.zeros_like(v) for k, v in model.P.items()}
    run_acc, run_loss = [], []
    for step in range(1, steps + 1):
        A, q, y = make_graph()
        total, conv, empt, p = model.loss(A, q, y, lambda_e=lambda_e)
        g = model.backward()
        for k in model.P:
            vel[k] = mom * vel[k] - lr * g[k]
            model.P[k] += vel[k]
        run_loss.append(conv)
        run_acc.append(int(np.argmax(p) == y))
        if step % log_every == 0:
            print(f"  step {step:5d} | conv-loss {np.mean(run_loss[-log_every:]):.3f} "
                  f"| emptiness {empt:.3f} | train-acc {np.mean(run_acc[-log_every:]):.3f}")
    return model


def evaluate(model, n=2000):
    labels = np.zeros(4)
    correct = np.zeros(4)
    tot_ok = 0
    for _ in range(n):
        A, q, y = make_graph()
        pred = model.predict(A, q)
        labels[y] += 1
        if pred == y:
            correct[y] += 1
            tot_ok += 1
    print(f"[eval] overall accuracy = {tot_ok / n:.3f}  (n={n})")
    for k in range(4):
        if labels[k] > 0:
            print(f"       {KOTI[k]:8s}: {correct[k]/labels[k]:.3f}  "
                  f"(support {int(labels[k])})")
    return tot_ok / n


def svabhava_baseline(n=2000):
    """
    The control: ZERO message-passing rounds. With no relations consulted,
    every node collapses to the same seed-derived vector, so the model can only
    ever output one corner. Its ceiling is the majority-class frequency -
    empirically the price of pretending phenomena have inherent existence.
    """
    model = SunyataRelationalEngine(dim=16, rounds=0, seed=1)
    train(model, steps=1500, lr=0.05, lambda_e=0.0, log_every=1500)
    return evaluate(model, n=n)


# ------------------------------------------------------------------------------
# 5. SELF-TESTS
# ------------------------------------------------------------------------------
def self_tests():
    print("\n[self-tests]")
    # (a) invariance: a node with no relations must never read as IS/IS_NOT/BOTH
    #     more than chance from structure - the label rule is well-formed.
    counts = {k: 0 for k in KOTI}
    for _ in range(3000):
        _, _, y = make_graph()
        counts[KOTI[y]] += 1
    print("  label distribution:", counts)
    assert all(c > 0 for c in counts.values()), "every corner should occur"

    # (b) determinism of forward pass
    m = SunyataRelationalEngine(dim=8, rounds=2, seed=3)
    A, q, _ = make_graph()
    l1 = m.forward(A, q); l2 = m.forward(A, q)
    assert np.allclose(l1, l2), "forward pass must be deterministic"

    # (c) relational sensitivity: adding an incoming edge to an isolated node
    #     must change its representation (identity is dependently originated)
    n = 6
    A = np.zeros((n, n)); q = 0
    h_iso = _final_state(m, A, q)
    A[1, 0] = 1.0  # give q a condition
    h_dep = _final_state(m, A, q)
    assert not np.allclose(h_iso, h_dep), "representation must respond to relations"
    print("  forward determinism: OK")
    print("  relational sensitivity (no svabhava): OK")
    print("  all self-tests passed.")


def _final_state(model, A, q):
    Nin, Nout = normalized_adjacencies(A)
    H = np.tile(model.seed, (A.shape[0], 1))
    for _ in range(model.rounds):
        Z = (H @ model.P["Wself"].T + (Nin @ H) @ model.P["Win"].T
             + (Nout @ H) @ model.P["Wout"].T + model.P["b"])
        H = np.tanh(Z)
    return H[q]


# ------------------------------------------------------------------------------
# 6. MAIN
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 72)
    print("Chapter 137 - Nagarjuna : The Sunyata Relational Engine")
    print("=" * 72)

    print("\n[1] Gradient check")
    ok = gradient_check()
    assert ok, "gradient check failed"

    self_tests()

    print("\n[2] Training the relational engine (identity from relations only)")
    model = SunyataRelationalEngine(dim=16, rounds=3, seed=137)
    train(model, steps=4000, lr=0.05, lambda_e=0.02)

    print("\n[3] Evaluation (emptiness paying rent)")
    acc = evaluate(model)

    print("\n[4] Svabhava baseline (0 rounds - phenomena treated as self-existent)")
    base = svabhava_baseline()

    print("\n" + "=" * 72)
    print(f"Relational engine : {acc:.3f}")
    print(f"Svabhava baseline : {base:.3f}")
    print(f"Emptiness dividend: {acc - base:+.3f}")
    print("The corners cannot be named from a thing in itself - only from its")
    print("conditions. That gap is Nagarjuna's thesis, measured.")
    print("=" * 72)
