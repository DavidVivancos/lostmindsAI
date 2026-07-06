"""
================================================================================
chapter_0047_pindar_-518.py  --  "THE EAGLE AND THE CROW"
A Pindaric Neural Architecture (pure NumPy, from scratch, gradient-checked)
Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
Tome 3 Minds 41 - 60 Available on Amazon https://www.amazon.com/dp/B0H6TVX69S
Resume and Interactive Demos at https://artificiology.com/
Author: David Vivancos · Chapter 0047 · Pindar of Cynoscephalae
================================================================================

FIGURE 47 of the 1000-Minds terabook:  PINDAR of Cynoscephalae (c.518 - c.438 BCE),
greatest of the nine canonical Greek lyric poets, master of the epinikion
(victory ode).

------------------------------------------------------------------------------
WHY THIS ARCHITECTURE IS *PINDAR'S* AND NO ONE ELSE'S
------------------------------------------------------------------------------
Most "poet" architectures default to rhythm/meter or to a transformer wearing a
laurel wreath.  We refuse that.  We build instead from the one cognitive claim
that is uniquely Pindar's, stated flat-out in Olympian 2.86:

        sophos ho polla eidos PHYAi          (Greek: phya = inborn nature)
        "Wise is he who knows many things BY NATURE,
         not those who have only LEARNED the lore --
         turbulent, intemperate of tongue, a pair of CROWS
         that chatter in vain against the god-like bird of Zeus."   (the EAGLE)

Pindar splits knowing into two irreconcilable kinds:

   * PHYA   (the EAGLE): innate, low-dimensional, single-leap insight. It does
            not accumulate; it *recognises*. It reaches the target in one bound.
   * MATHOS (the CROW):  learned, high-capacity, accumulative chatter. It piles
            up detail over time but smears, and "chatters in vain."

To that he welds two more signatures that recur across the 45 surviving odes:

   * KAIROS / METRON -- due measure and the critical moment. Pindar's poetics
     are governed not by saying *everything* but by striking the ONE right note
     at the ONE right time, then STOPPING ("the limit must be observed").
   * EPHEMERALITY + IMMORTAL SONG -- "creatures of a day... man is the dream of
     a shadow" (Pythian 8.95-96). Mortal states FADE. The poet's word is the one
     immortality available to mortals: it FIXES the worthy deed in lasting song.
   * THE TRIAD -- every Pindaric ode is built in triads: strophe (turn),
     antistrophe (counter-turn), epode (synthesis).

------------------------------------------------------------------------------
THE MECHANISM (one screen)
------------------------------------------------------------------------------
For a sequence of "deeds" x_1..x_T:

   EAGLE  e_t = tanh( (x_t A) B )          A:(D_in,R) B:(R,D),  R tiny  -> innate,
                                           cannot memorise, only recognises.
   CROW   strophe   af_t = rho*af_{t-1} + x_t Wc       (forward accumulation)
          antistrophe ab_t = rho*ab_{t+1} + x_t Wc     (reverse accumulation)
          epode      m_t  = tanh( 0.5*(af_t + ab_t) )  (synthesis)
          rho in (0,1) is the "dream of a shadow" DECAY: mortal memory fades.

   TRUST GATE  mix_t = sigma(z_t w_mix + b_mix)   z_t = [e_t ; m_t]
               blend_t = mix_t*e_t + (1-mix_t)*m_t   (eagle vs crow, per deed)

   KAIROS WRITE GATE  wr_t = sigma(z_t w_wr + b_wr)   (the critical moment)
   SONG MEMORY (immortal, NON-decaying weighted sum -- the poet fixing the deed):
               S = sum_t wr_t*blend_t / (sum_t wr_t + eps)
   READOUT     logits = S Wout + b_out          (we read from the SONG, not the
                                                  fading state -- immortality)

   LOSS = CrossEntropy(logits, y)  +  beta * mean_t(wr_t)     <-- METRON penalty:
          the model is punished for immortalising too much; it must observe
          due measure and fix only the deed that is worthy.

This is NOT attention-over-stored-keys: the song weights come from a temporal
write-gate (the poet choosing the deed), not from query-key dot products.
It is NOT a transformer, NOT MoE. The eagle is a hard low-rank bottleneck; the
crow is a bidirectional leaky integrator; the song memory is a gated, decay-
free accumulator. Each piece encodes a specific Pindaric idea, and the whole
trains end-to-end with hand-derived gradients verified by finite differences.

------------------------------------------------------------------------------
THE TASK IT IS TRAINED ON ("The Deed Worth Immortalising")
------------------------------------------------------------------------------
Each example is a stream of T deeds. Almost all are random "chatter." Exactly
one deed -- at a RANDOM position, often early -- carries a god-sent GLEAM marker
and a class label. The model must output that label at the end. Because mortal
memory DECAYS (rho<1), a system that only accumulates (the crow alone) forgets
an early gleam by the end. Survival requires the EAGLE to recognise the gleam in
one leap and the KAIROS gate to FIX it in the immortal song. The task is the
thesis made executable.

Run:  python3 chapter_0047_pindar_-518.py
================================================================================
"""

import numpy as np

# ------------------------------------------------------------------------------
# 0. Small numerical helpers
# ------------------------------------------------------------------------------

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))

def softmax(v):
    v = v - np.max(v)
    e = np.exp(v)
    return e / (np.sum(e) + 1e-12)


# ------------------------------------------------------------------------------
# 1. THE PINDARIC MIND
#    A single coherent module holding every parameter, with an exact forward
#    pass, a hand-derived backward pass, and a self-contained loss.
# ------------------------------------------------------------------------------

class PindaricMind:
    """The Eagle-and-Crow network. One ode-unit = one forward pass over a
    sequence of deeds, producing one immortalised verdict (class logits)."""

    def __init__(self, D_in, D, R, C, beta=0.04, seed=0):
        """
        D_in : dimension of each incoming 'deed' vector
        D    : model (song) dimension
        R    : EAGLE rank -- the number of innate modes (kept TINY on purpose:
               the eagle knows by nature, so it must not be allowed to memorise)
        C    : number of classes (verdicts)
        beta : weight of the METRON penalty (due-measure on the kairos gate)
        """
        rng = np.random.default_rng(seed)
        self.D_in, self.D, self.R, self.C, self.beta = D_in, D, R, C, beta

        s = 1.0 / np.sqrt(D_in)
        # EAGLE (phya): a hard low-rank bottleneck D_in -> R -> D
        self.A = rng.normal(0, s, (D_in, R))
        self.B = rng.normal(0, 1.0 / np.sqrt(R), (R, D))
        # CROW (mathos): shared strophe/antistrophe accumulation weights
        self.Wc = rng.normal(0, s, (D_in, D))
        # rho ('dream of a shadow' decay) via sigmoid(rho_param); init ~0.6
        self.rho_param = np.array(0.4)
        # trust gate (eagle vs crow) and kairos write gate operate on [e;m]
        self.w_mix = rng.normal(0, 1.0 / np.sqrt(2 * D), (2 * D,))
        self.b_mix = np.array(0.0)
        self.w_wr = rng.normal(0, 1.0 / np.sqrt(2 * D), (2 * D,))
        self.b_wr = np.array(0.0)
        # readout from the immortal song memory
        self.Wout = rng.normal(0, 1.0 / np.sqrt(D), (D, C))
        self.b_out = np.zeros(C)

    # --- parameter plumbing (used by the optimiser and the gradient check) ---
    def params(self):
        return {
            "A": self.A, "B": self.B, "Wc": self.Wc, "rho_param": self.rho_param,
            "w_mix": self.w_mix, "b_mix": self.b_mix,
            "w_wr": self.w_wr, "b_wr": self.b_wr,
            "Wout": self.Wout, "b_out": self.b_out,
        }

    def set_params(self, P):
        for k, v in P.items():
            setattr(self, k, v)

    # -------------------------------------------------------------------------
    # FORWARD  (returns loss, and a cache for the backward pass)
    # -------------------------------------------------------------------------
    def forward(self, X, y):
        """
        X : (T, D_in) sequence of deeds
        y : int class label (the deed worth immortalising)
        """
        T, D, R, eps = X.shape[0], self.D, self.R, 1e-8
        rho = float(sigmoid(self.rho_param))

        # EAGLE: innate single-leap recognition, order-independent
        PE = X @ self.A                      # (T,R)  projection onto innate modes
        U = PE @ self.B                      # (T,D)
        E = np.tanh(U)                       # (T,D)  eagle candidates

        # CROW strophe (forward leaky accumulation) -- "dream of a shadow" decay
        XWc = X @ self.Wc                    # (T,D)
        AF = np.zeros((T, D)); prev = np.zeros(D)
        for t in range(T):
            prev = rho * prev + XWc[t]; AF[t] = prev
        # CROW antistrophe (reverse leaky accumulation)
        AB = np.zeros((T, D)); nxt = np.zeros(D)
        for t in range(T - 1, -1, -1):
            nxt = rho * nxt + XWc[t]; AB[t] = nxt
        S_pre = 0.5 * (AF + AB)              # epode synthesis (pre-activation)
        M = np.tanh(S_pre)                   # (T,D) crow candidates

        # gates
        Z = np.concatenate([E, M], axis=1)   # (T,2D)
        a_mix = Z @ self.w_mix + self.b_mix  # (T,)
        MIX = sigmoid(a_mix)                  # (T,)  eagle-vs-crow trust
        a_wr = Z @ self.w_wr + self.b_wr      # (T,)
        WR = sigmoid(a_wr)                    # (T,)  kairos write gate
        BL = MIX[:, None] * E + (1.0 - MIX)[:, None] * M   # (T,D) blended deed

        # IMMORTAL SONG MEMORY: gated, decay-free weighted average
        Wsum = np.sum(WR) + eps
        G = np.sum(WR[:, None] * BL, axis=0)               # (D,)
        Ssong = G / Wsum                                   # (D,) the song

        logits = Ssong @ self.Wout + self.b_out            # (C,)
        p = softmax(logits)
        ce = -np.log(p[y] + 1e-12)
        metron = self.beta * np.mean(WR)                   # due-measure penalty
        loss = ce + metron

        cache = dict(X=X, y=y, rho=rho, PE=PE, E=E, XWc=XWc, AF=AF, AB=AB,
                     M=M, Z=Z, MIX=MIX, WR=WR, BL=BL, Wsum=Wsum, G=G,
                     Ssong=Ssong, p=p, T=T)
        return loss, cache

    # -------------------------------------------------------------------------
    # BACKWARD  (exact hand-derived gradients; verified by finite differences)
    # -------------------------------------------------------------------------
    def backward(self, cache):
        X, y, rho = cache["X"], cache["y"], cache["rho"]
        E, M, Z, MIX, WR, BL = cache["E"], cache["M"], cache["Z"], cache["MIX"], cache["WR"], cache["BL"]
        AF, AB, XWc, PE = cache["AF"], cache["AB"], cache["XWc"], cache["PE"]
        Wsum, G, Ssong, p, T = cache["Wsum"], cache["G"], cache["Ssong"], cache["p"], cache["T"]
        D, R = self.D, self.R

        g = {k: np.zeros_like(v) for k, v in self.params().items()}

        # output layer
        dlogits = p.copy(); dlogits[y] -= 1.0                 # dL/dlogits
        g["Wout"] += np.outer(Ssong, dlogits)
        g["b_out"] += dlogits
        dSsong = self.Wout @ dlogits                          # (D,)

        # song memory  S = G/Wsum
        dG = dSsong / Wsum                                    # (D,)
        dWsum = -np.dot(dSsong, Ssong) / Wsum                 # scalar
        dWR = BL @ dG + dWsum                                 # (T,) via G and Wsum
        dWR += self.beta / T                                  # metron penalty
        dBL = WR[:, None] * dG[None, :]                       # (T,D)

        # kairos write gate
        da_wr = dWR * WR * (1.0 - WR)                         # (T,)
        g["w_wr"] += Z.T @ da_wr
        g["b_wr"] += np.sum(da_wr)
        dZ = np.outer(da_wr, self.w_wr)                       # (T,2D)

        # blend -> mix, eagle, crow
        dMIX = np.sum(dBL * (E - M), axis=1)                  # (T,)
        dE = dBL * MIX[:, None]                               # (T,D) via blend
        dM = dBL * (1.0 - MIX)[:, None]                       # (T,D) via blend
        da_mix = dMIX * MIX * (1.0 - MIX)
        g["w_mix"] += Z.T @ da_mix
        g["b_mix"] += np.sum(da_mix)
        dZ += np.outer(da_mix, self.w_mix)
        dE += dZ[:, :D]                                       # gate path into E
        dM += dZ[:, D:]                                       # gate path into M

        # EAGLE backward:  E = tanh(U), U = PE@B, PE = X@A
        dU = dE * (1.0 - E ** 2)                              # (T,D)
        g["B"] += PE.T @ dU                                   # (R,D)
        dPE = dU @ self.B.T                                   # (T,R)
        g["A"] += X.T @ dPE                                   # (D_in,R)

        # CROW backward:  M = tanh(S_pre), S_pre = 0.5(AF+AB)
        dS_pre = dM * (1.0 - M ** 2)                          # (T,D)
        dAF = 0.5 * dS_pre
        dAB = 0.5 * dS_pre
        drho = 0.0

        # BPTT through forward accumulation AF_t = rho*AF_{t-1} + XWc_t
        carry = np.zeros(D)
        for t in range(T - 1, -1, -1):
            Gaf = dAF[t] + rho * carry
            g["Wc"] += np.outer(X[t], Gaf)
            prevAF = AF[t - 1] if t > 0 else np.zeros(D)
            drho += np.dot(Gaf, prevAF)
            carry = Gaf
        # BPTT through reverse accumulation AB_t = rho*AB_{t+1} + XWc_t
        carry = np.zeros(D)
        for t in range(T):
            Gab = dAB[t] + rho * carry
            g["Wc"] += np.outer(X[t], Gab)
            nextAB = AB[t + 1] if t < T - 1 else np.zeros(D)
            drho += np.dot(Gab, nextAB)
            carry = Gab

        # rho = sigmoid(rho_param)
        g["rho_param"] += np.array(drho * rho * (1.0 - rho))
        return g


# ------------------------------------------------------------------------------
# 2. GRADIENT CHECK (mandatory)  --  central finite differences vs. backprop
# ------------------------------------------------------------------------------

def gradient_check(seed=1, eps=1e-6):
    """Compare analytic gradients to central finite differences on a tiny model
    averaged over a small batch. Returns the worst relative error."""
    rng = np.random.default_rng(seed)
    D_in, D, R, C, T = 5, 4, 2, 3, 6
    mind = PindaricMind(D_in, D, R, C, beta=0.05, seed=seed)
    batch = [(rng.normal(size=(T, D_in)), int(rng.integers(C))) for _ in range(4)]

    def batch_loss_and_grad():
        gtot = {k: np.zeros_like(v) for k, v in mind.params().items()}
        L = 0.0
        for X, y in batch:
            loss, cache = mind.forward(X, y)
            L += loss
            gb = mind.backward(cache)
            for k in gtot:
                gtot[k] += gb[k]
        n = len(batch)
        return L / n, {k: v / n for k, v in gtot.items()}

    _, ganalytic = batch_loss_and_grad()

    worst = 0.0
    for name, P in mind.params().items():
        flat = P.ravel()
        ga = ganalytic[name].ravel()
        idxs = range(flat.size) if flat.size <= 12 else rng.choice(flat.size, 12, replace=False)
        for i in idxs:
            orig = flat[i]
            flat[i] = orig + eps
            Lp, _ = batch_loss_and_grad()
            flat[i] = orig - eps
            Lm, _ = batch_loss_and_grad()
            flat[i] = orig
            num = (Lp - Lm) / (2 * eps)
            ana = ga[i]
            denom = max(1e-9, abs(num) + abs(ana))
            rel = abs(num - ana) / denom
            worst = max(worst, rel)
    return worst


# ------------------------------------------------------------------------------
# 3. THE TASK: "The Deed Worth Immortalising"
# ------------------------------------------------------------------------------

def make_example(rng, T, C, D_in, noise=0.7, gleam_early_bias=True):
    """One stream of deeds. EVERY deed makes a CLAIM (a label one-hot in dims
    1..C) -- the chatter deeds make FALSE claims, like crows. dim 0 is the
    god-gleam marker, =1 only on the ONE worthy deed; the true label is that
    deed's claim. Because false claims fill every position, a model that
    immortalises everything (uniform song) blends true+false and fails: only
    the kairos gate, fixing the single gleam deed, recovers the truth.
    The gleam is biased EARLY so the decaying crow accumulator forgets it."""
    X = noise * rng.normal(size=(T, D_in))
    X[:, 0] = 0.0
    X[:, 1:1 + C] = 0.0
    # every deed advances a claim (a one-hot in the label block)
    claims = rng.integers(0, C, size=T)
    for t in range(T):
        X[t, 1 + int(claims[t])] = 1.0
    if gleam_early_bias:
        pos = int(rng.integers(0, max(1, T // 3) + 1))   # worthy deed: early
    else:
        pos = int(rng.integers(0, T))
    label = int(claims[pos])           # the truth is the worthy deed's claim
    X[pos, 0] = 1.0                    # the god-sent gleam marks the worthy deed
    return X, label, pos


def make_batch(rng, n, T, C, D_in, **kw):
    return [make_example(rng, T, C, D_in, **kw)[:2] for _ in range(n)]


# ------------------------------------------------------------------------------
# 4. TRAINING  (plain momentum SGD over the hand-derived gradients)
# ------------------------------------------------------------------------------

def accuracy(mind, data):
    ok = 0
    for X, y in data:
        _, cache = mind.forward(X, y)
        ok += int(np.argmax(cache["p"]) == y)
    return ok / len(data)


def train(mind, rng, steps=600, batch=16, lr=0.08, mom=0.9, T=12, C=4, log=True):
    D_in = mind.D_in
    vel = {k: np.zeros_like(v) for k, v in mind.params().items()}
    val = make_batch(rng, 200, T, C, D_in)
    hist = []
    for step in range(1, steps + 1):
        data = make_batch(rng, batch, T, C, D_in)
        gtot = {k: np.zeros_like(v) for k, v in mind.params().items()}
        L = 0.0
        for X, y in data:
            loss, cache = mind.forward(X, y)
            L += loss
            gb = mind.backward(cache)
            for k in gtot:
                gtot[k] += gb[k]
        for k in gtot:
            gtot[k] /= batch
            vel[k] = mom * vel[k] - lr * gtot[k]
            getattr(mind, k)
            P = mind.params()[k] + vel[k]
            setattr(mind, k, P)
        if log and (step % 100 == 0 or step == 1):
            acc = accuracy(mind, val)
            hist.append((step, L / batch, acc))
            print(f"  step {step:4d} | loss {L / batch:6.4f} | val_acc {acc:5.3f}")
    return val, hist


# ------------------------------------------------------------------------------
# 5. ABLATION: does the KAIROS song memory actually matter?
#    Replace the learned write gate with a UNIFORM one (immortalise everything
#    equally -- "no due measure"). The decay then smears the early gleam away.
# ------------------------------------------------------------------------------

def accuracy_uniform_song(mind, data):
    """Forward but with WR forced uniform (kairos disabled). Crude surgery on
    the forward pass to isolate the contribution of the kairos write gate."""
    ok = 0
    for X, y in data:
        T = X.shape[0]
        rho = float(sigmoid(mind.rho_param))
        PE = X @ mind.A; E = np.tanh(PE @ mind.B)
        XWc = X @ mind.Wc
        AF = np.zeros_like(E); prev = np.zeros(mind.D)
        for t in range(T):
            prev = rho * prev + XWc[t]; AF[t] = prev
        AB = np.zeros_like(E); nxt = np.zeros(mind.D)
        for t in range(T - 1, -1, -1):
            nxt = rho * nxt + XWc[t]; AB[t] = nxt
        M = np.tanh(0.5 * (AF + AB))
        Z = np.concatenate([E, M], axis=1)
        MIX = sigmoid(Z @ mind.w_mix + mind.b_mix)
        BL = MIX[:, None] * E + (1 - MIX)[:, None] * M
        Ssong = np.mean(BL, axis=0)               # UNIFORM song: no kairos
        p = softmax(Ssong @ mind.Wout + mind.b_out)
        ok += int(np.argmax(p) == y)
    return ok / len(data)


# ------------------------------------------------------------------------------
# 6. MAIN: gradient check -> train -> self-tests
# ------------------------------------------------------------------------------

def main():
    np.set_printoptions(precision=4, suppress=True)
    print("=" * 78)
    print(" PINDAR  --  'The Eagle and the Crow'  (phya vs mathos; kairos; song)")
    print("=" * 78)

    print("\n[1] FINITE-DIFFERENCE GRADIENT CHECK (mandatory)")
    worst = gradient_check(seed=3)
    print(f"    worst relative error across all parameter groups: {worst:.3e}")
    assert worst < 1e-5, "gradient check FAILED"
    print("    PASS  (analytic backprop matches numerical gradients)")

    print("\n[2] TRAINING on 'The Deed Worth Immortalising'")
    print("    (gleam biased EARLY; decay rho<1 makes pure accumulation forget)")
    rng = np.random.default_rng(7)
    mind = PindaricMind(D_in=16, D=24, R=3, C=4, beta=0.04, seed=11)
    val, hist = train(mind, rng, steps=600, batch=16, lr=0.08, T=12, C=4)
    final_acc = accuracy(mind, val)
    print(f"    final validation accuracy: {final_acc:5.3f}")

    print("\n[3] SELF-TESTS")
    # (a) learning happened
    assert final_acc > 0.85, "model failed to learn the task"
    print(f"    (a) learned task (acc {final_acc:.3f} > 0.85)            PASS")

    # (b) kairos matters: ablating the write gate (uniform song) should hurt
    abl = accuracy_uniform_song(mind, val)
    print(f"    (b) kairos ablation accuracy (uniform song): {abl:.3f}")
    assert final_acc - abl > 0.15, "kairos write gate not doing real work"
    print(f"        kairos gate lifts accuracy by {final_acc - abl:.3f}     PASS")

    # (c) due measure (metron): the write gate is SPARSE -- it fixes few deeds.
    #     Measure how concentrated WR is on the true gleam position.
    rng2 = np.random.default_rng(99)
    on_gleam, mean_wr = [], []
    for _ in range(300):
        X, y, pos = make_example(rng2, 12, 4, 16)
        _, cache = mind.forward(X, y)
        wr = cache["WR"]
        on_gleam.append(wr[pos] / (np.sum(wr) + 1e-8))
        mean_wr.append(np.mean(wr))
    frac_on_gleam = float(np.mean(on_gleam))
    print(f"    (c) fraction of song-weight on the true gleam deed: {frac_on_gleam:.3f}")
    print(f"        mean write-gate activation (lower = more measured): {np.mean(mean_wr):.3f}")
    assert frac_on_gleam > 0.5, "kairos gate failed to find the worthy deed"
    print(f"        kairos concentrates on the worthy deed             PASS")

    # (d) eagle bottleneck is genuinely low-rank (innate, cannot memorise)
    assert mind.R < mind.D, "eagle is not a bottleneck"
    print(f"    (d) eagle rank R={mind.R} << song dim D={mind.D} (phya bottleneck) PASS")

    # (e) the 'dream of a shadow' decay learned to be < 1 (memory truly fades)
    rho = float(sigmoid(mind.rho_param))
    print(f"    (e) learned decay rho = {rho:.3f}  (mortal memory fades)   "
          f"{'PASS' if rho < 0.999 else 'note'}")

    print("\n" + "=" * 78)
    print(" VERIFIED SUMMARY (paste into chapter)")
    print("=" * 78)
    print(f"  gradient_check_worst_rel_err : {worst:.3e}")
    print(f"  final_val_accuracy           : {final_acc:.3f}")
    print(f"  kairos_ablation_accuracy     : {abl:.3f}")
    print(f"  kairos_accuracy_lift         : {final_acc - abl:.3f}")
    print(f"  song_weight_on_true_gleam    : {frac_on_gleam:.3f}")
    print(f"  learned_decay_rho            : {rho:.3f}")
    print("=" * 78)

    return dict(worst=worst, final_acc=final_acc, abl=abl,
                frac_on_gleam=frac_on_gleam, rho=rho)


if __name__ == "__main__":
    main()
