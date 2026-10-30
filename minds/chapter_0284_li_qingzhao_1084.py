"""
Figure 0284 -- Li Qingzhao (李清照), 1084-c.1155
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0284 · Li Qingzhao (李清照), 1084-c.1155
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 15, Minds 281-300 Available on Amazon https://www.amazon.com/dp/B0HMQ7TVJM
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0284_li_qingzhao_1084 - Li Qingzhao (李清照), 1084-c.1155
Architecture: the Tone-Lattice Consolidation Network (TLCN)

WHY THIS SHAPE, NOT A TRANSFORMER
----------------------------------
Li Qingzhao is the author of "On Ci" (Lun Ci, 論詞), the first systematic
treatise on the ci (詞) song-lyric as a genre distinct from shi (詩) poetry.
Her argument was narrow and technical: ci is bound to musical tune-patterns
(cipai) with fixed tone sequences, rhyme positions, and a "light/heavy"
syllable register; a poet who ignores this -- however great a poet in shi --
produces something that fails as ci. She named names: Liu Yong ("meets the
prosodic rules but his diction is vulgar"), Yan Shu, Ouyang Xiu and Su Shi
("their ci are simply shi with uneven line lengths"), Wang Anshi and Zeng
Gong ("unreadable if set to music"). Her second surviving prose work, the
afterword to the "Jinshi Lu" (金石錄, Records of Metal and Stone -- the
antiquarian catalogue she compiled with her husband Zhao Mingcheng), is a
meditation on the same problem from the opposite direction: she watched two
libraries of books, bronzes, and inscriptions -- the accumulated documentary
memory of a marriage and of a civilization -- destroyed twice, in the
Jingkang flight of 1127 and again in the chaotic retreat that followed. Her
conclusion, stated explicitly in that afterword, is that objects do not
protect knowledge; only correct catalogue entries do, because a catalogue
entry can be re-copied, re-verified and re-transmitted independently of any
single physical substrate. Her third surviving technical text, "Da Ma Tu
Jing" (打馬圖經, 1134), is a rulebook for a dice-and-board race game, prefaced
by the claim that "gambling is nothing but the art of striving to be
first" -- i.e. that apparent games of chance are, on inspection, decidable
by skill once their rule structure is made explicit.

Read together, these three texts are not three separate interests. They are
one cognitive habit applied three times: Li Qingzhao's recurring move is to
extract the *explicit rule structure* underneath an experience -- a tune's
tone pattern, an artifact's provenance, a game's transition rules -- because
she has personally watched everything that was NOT reduced to explicit,
re-transmissible structure get burned, looted, or lost. Feeling, for her,
is never in tension with formal rule; formal rule is what allows feeling to
outlive its occasion. That is the thesis this architecture encodes, and it
is the reason the three modules below are a grammar-masked classifier, a
weight-consolidation mechanism, and a game-playing policy, rather than a
generic sequence-to-sequence attention stack.

THE THREE MODULES
------------------
1. TonalGrammar + TonalEncoder  -- a small MLP that predicts the next
   phonological "register" in a lyric line, but whose output logits are
   hard-masked by a hand-authored, non-trainable transition grammar (the
   cipai analogue). The grammar does not compete with the network for
   capacity; it removes illegal outcomes from the softmax entirely, so
   gradient only ever sharpens choices *within* the legal set -- a direct
   computational reading of her claim that the tune-pattern is an enabling
   scaffold, not a restriction on expression.

2. Elastic Consolidation ("Jinshi Lu" module) -- after the network has
   learned the tonal-affective mapping of a first body of work ("the first
   collection"), we estimate a diagonal Fisher-information importance score
   per weight from that data, then continue training on a second, disjoint
   body of work ("the second collection", modelling the post-1127 style)
   under a quadratic penalty that protects important first-collection
   weights. This is deliberately the same mechanism, computationally, as
   elastic weight consolidation in continual-learning research (Kirkpatrick
   et al., 2017) -- chosen because it is the cleanest available formal
   analogue of her own claim in the Jinshi Lu afterword: an explicit,
   weighted "catalogue" of what matters is what survives a second
   catastrophe; undifferentiated storage does not.

3. Da Ma policy ("striving to be first") -- a REINFORCE-trained policy for
   a small stochastic race game with bonus and trap cells, benchmarked
   against a uniform-random policy. This operationalises the one-sentence
   thesis of her "Da Ma Fu": that a game which looks like pure dice-chance
   nonetheless has an exploitable, learnable decision structure once you
   look for the rule beneath the randomness.

Everything below is pure NumPy. There is no autograd. Every backward pass
is hand-derived and checked against a central-difference numerical gradient
before any training claim is trusted (see test_gradient_check). Random seeds
are fixed for reproducibility; 1084 is Li Qingzhao's birth year.

Run directly:  python3 chapter_0284_li_qingzhao_1084.py
"""

import numpy as np

RNG_SEED = 1084
rng = np.random.default_rng(RNG_SEED)

# ============================================================================
# 0. Generic dense-layer machinery (shared by the tonal encoder and the
#    Da Ma policy network so the gradient-check logic is written once and
#    exercised twice, against two different loss functions).
# ============================================================================

def tanh(x):
    return np.tanh(x)

def dtanh(h):
    # derivative of tanh given h = tanh(x)
    return 1.0 - h * h

def softmax(logits):
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    e = np.exp(logits)
    return e / np.sum(e, axis=-1, keepdims=True)


class TwoLayerNet:
    """
    A minimal x -> tanh(W1 x + b1) -> (W2 h + b2) = logits network,
    with hand-written forward/backward. Used both as the tonal encoder
    (Module 1) and as the Da Ma policy network (Module 3).
    """

    def __init__(self, n_in, n_hidden, n_out, scale=0.5, seed=None):
        r = np.random.default_rng(seed)
        self.W1 = r.normal(0, scale / np.sqrt(n_in), size=(n_hidden, n_in))
        self.b1 = np.zeros(n_hidden)
        self.W2 = r.normal(0, scale / np.sqrt(n_hidden), size=(n_out, n_hidden))
        self.b2 = np.zeros(n_out)
        self.n_in, self.n_hidden, self.n_out = n_in, n_hidden, n_out

    def params(self):
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}

    def set_flat(self, flat):
        i = 0
        for name, p in self.params().items():
            n = p.size
            p.flat[:] = flat[i:i + n]
            i += n

    def get_flat(self):
        return np.concatenate([p.ravel() for p in self.params().values()])

    def forward(self, x):
        z1 = self.W1 @ x + self.b1
        h = tanh(z1)
        z2 = self.W2 @ h + self.b2
        cache = (x, z1, h, z2)
        return z2, cache

    def backward_logits_grad(self, cache, dlogits):
        """
        Given dL/dlogits (dz2), back-propagate to all four parameter
        gradients and return them as a dict with the same keys as params().
        """
        x, z1, h, z2 = cache
        dW2 = np.outer(dlogits, h)
        db2 = dlogits
        dh = self.W2.T @ dlogits
        dz1 = dh * dtanh(h)
        dW1 = np.outer(dz1, x)
        db1 = dz1
        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    def apply_grad(self, grads, lr):
        for name, p in self.params().items():
            p -= lr * grads[name]


# ============================================================================
# 1. TonalGrammar -- the hand-authored, non-trainable transition constraint.
#    5 phonological "registers", loosely evoking the five tones (gong/shang/
#    jue/zhi/yu 宮商角徵羽) and the clear/turbid (清濁) distinction Li
#    Qingzhao invokes in "On Ci" against poets who ignore musical register.
#    The exact mapping to historical tone theory is illustrative, not a
#    claim of musicological reconstruction -- what matters computationally
#    is that it is a genuinely irregular, hand-authored constraint (not a
#    trivial ring or a learned pattern).
# ============================================================================

N_REGISTERS = 5
REGISTER_NAMES = ["gong 宮", "shang 商", "jue 角", "zhi 徵", "yu 羽"]

ALLOWED = np.array([
    # to: g  s  j  z  y
    [1, 1, 0, 0, 1],   # from gong
    [1, 1, 1, 0, 0],   # from shang
    [0, 1, 1, 1, 0],   # from jue
    [0, 0, 1, 1, 1],   # from zhi
    [1, 0, 0, 1, 1],   # from yu
], dtype=bool)

NEG_INF_MASK = np.where(ALLOWED, 0.0, -1e9)  # additive logit mask per "from" state


def masked_softmax_from(logits, prev_register):
    masked = logits + NEG_INF_MASK[prev_register]
    return softmax(masked)


# ============================================================================
# Feature construction for the tonal encoder.
#   input x = [ one-hot(prev_register) (5) | affect vector (4) | memory (8) ] = 17-dim
#   affect vector: (valence, arousal, attachment, transience) -- a crude but
#   workable 4-axis reading of the affect-language that recurs across her
#   ci: joy/sorrow (valence), the intensity of longing (arousal), the
#   felt presence of a bond to a specific person (attachment), and the
#   awareness that the present state will not last (transience) -- the note
#   that late works add and early works largely lack.
# ============================================================================

N_AFFECT = 4
N_MEM = 8
N_IN = N_REGISTERS + N_AFFECT + N_MEM
N_HIDDEN = 24


# ----------------------------------------------------------------------------
# Symbolic archive / "Jinshi Lu index": a fixed bank of (key, value) pairs
# retrieved by hard nearest-neighbour lookup (argmax cosine similarity), NOT
# by soft attention. This is a deliberate choice: attention-over-stored-keys
# blends evidence continuously; a physical catalogue either finds the entry
# that matches an inscription or it does not. Retrieval here is discrete and
# non-differentiable -- it contributes a feature, not a gradient path -- which
# keeps this module a genuinely different mechanism from a transformer's
# key/value attention rather than a relabelled copy of it.
# ----------------------------------------------------------------------------

class CatalogueIndex:
    def __init__(self, n_entries, key_dim, val_dim, seed):
        r = np.random.default_rng(seed)
        self.keys = r.normal(size=(n_entries, key_dim))
        self.keys /= np.linalg.norm(self.keys, axis=1, keepdims=True)
        self.values = r.normal(size=(n_entries, val_dim))

    def retrieve(self, query):
        q = query / (np.linalg.norm(query) + 1e-9)
        sims = self.keys @ q
        idx = int(np.argmax(sims))
        return self.values[idx]


KEY_DIM = 6
_query_proj = rng.normal(size=(KEY_DIM, N_REGISTERS + N_AFFECT)) * 0.5

# Two disjoint sub-catalogues: entries laid down while cataloguing survives
# ("first collection", pre-1127) and entries laid down after the flight
# south ("second collection", post-1127). A run always queries the
# catalogue that matches which "collection" (task) is currently active,
# mirroring the fact that Li Qingzhao's own catalogue was itself split by
# the flight into what survived and what had to be reconstructed from memory.
CATALOGUE_A = CatalogueIndex(40, KEY_DIM, N_MEM, seed=RNG_SEED + 1)
CATALOGUE_B = CatalogueIndex(40, KEY_DIM, N_MEM, seed=RNG_SEED + 2)


def build_feature(prev_register, affect, catalogue):
    onehot = np.zeros(N_REGISTERS)
    onehot[prev_register] = 1.0
    query = _query_proj @ np.concatenate([onehot, affect])
    mem = catalogue.retrieve(query)
    return np.concatenate([onehot, affect, mem])


# ============================================================================
# 2. Synthetic "teacher" generators for Task A (first collection) and
#    Task B (second collection). Each teacher is itself a random,
#    grammar-respecting MLP; the student TonalEncoder must learn to imitate
#    it. Two different teachers with different affect-conditioning give
#    the student two genuinely different target mappings to acquire in
#    sequence -- the standard continual-learning setup -- while both remain
#    subject to the SAME tonal grammar (the genre law does not change
#    between "collections"; only the emotional material set into it does).
# ============================================================================

def make_teacher(seed):
    return TwoLayerNet(N_IN, N_HIDDEN, N_REGISTERS, scale=1.1, seed=seed)

TEACHER_A = make_teacher(RNG_SEED + 10)
TEACHER_B = make_teacher(RNG_SEED + 11)

# A freshly initialised random MLP produces a near-uniform distribution over
# its 2-3 legal next-registers, which makes even the teacher's own argmax
# match a *sampled* label only about half the time -- there would be no
# well-defined "correct answer" to learn. TEACHER_TEMP sharpens the teacher
# distribution (a standard distillation move) so the synthetic task has a
# learnable, non-degenerate target while remaining stochastic rather than
# deterministic.
TEACHER_TEMP = 3.5


def sample_affect(task, r):
    """Task A: brighter/companionate affect. Task B: grief/transience-heavy
    affect, consistent with the pre- and post-1127 change in her ci that
    every biographical and literary-historical source agrees on."""
    if task == "A":
        valence = r.uniform(0.1, 1.0)
        arousal = r.uniform(-0.3, 0.6)
        attachment = r.uniform(0.3, 1.0)
        transience = r.uniform(-0.5, 0.3)
    else:
        valence = r.uniform(-1.0, -0.1)
        arousal = r.uniform(-0.2, 0.8)
        attachment = r.uniform(-1.0, 0.2)
        transience = r.uniform(0.3, 1.0)
    return np.array([valence, arousal, attachment, transience])


def gen_dataset(task, teacher, catalogue, n, seed):
    r = np.random.default_rng(seed)
    X, y, prevs = [], [], []
    prev = int(r.integers(0, N_REGISTERS))
    for _ in range(n):
        affect = sample_affect(task, r)
        x = build_feature(prev, affect, catalogue)
        logits, _ = teacher.forward(x)
        p = masked_softmax_from(logits * TEACHER_TEMP, prev)
        target = int(r.choice(N_REGISTERS, p=p))
        X.append(x)
        y.append(target)
        prevs.append(prev)
        prev = target
    return np.array(X), np.array(y), np.array(prevs)


# ============================================================================
# 3. Training / evaluation utilities for the tonal encoder.
# ============================================================================

def ce_loss_and_dlogits(logits, prev_register, target):
    p = masked_softmax_from(logits, prev_register)
    loss = -np.log(p[target] + 1e-12)
    dlogits = p.copy()
    dlogits[target] -= 1.0
    return loss, dlogits


def train_encoder(net, X, y, prevs, epochs, lr, ewc=None):
    """
    ewc, if given, is a dict {"fisher": {...}, "theta_star": {...}, "lambda": float}
    adding an elastic-consolidation penalty around theta_star, weighted by
    the diagonal Fisher importance.
    """
    n = len(X)
    losses = []
    for ep in range(epochs):
        idx = rng.permutation(n)
        total = 0.0
        for i in idx:
            logits, cache = net.forward(X[i])
            loss, dlogits = ce_loss_and_dlogits(logits, prevs[i], y[i])
            grads = net.backward_logits_grad(cache, dlogits)
            if ewc is not None:
                for name, p in net.params().items():
                    penalty_grad = ewc["lambda"] * ewc["fisher"][name] * (p - ewc["theta_star"][name])
                    grads[name] = grads[name] + penalty_grad
                    loss = loss + 0.5 * ewc["lambda"] * np.sum(ewc["fisher"][name] * (p - ewc["theta_star"][name]) ** 2)
            net.apply_grad(grads, lr)
            total += loss
        losses.append(total / n)
    return losses


def eval_accuracy(net, X, y, prevs):
    correct = 0
    for i in range(len(X)):
        logits, _ = net.forward(X[i])
        p = masked_softmax_from(logits, prevs[i])
        pred = int(np.argmax(p))
        correct += int(pred == y[i])
    return correct / len(X)


def compute_fisher_diag(net, X, prevs, y, n_samples):
    """Empirical diagonal Fisher information: average squared gradient of
    the log-likelihood of the (sampled) target under the current model,
    over a batch of Task-A data. This is the standard EWC estimator
    (Kirkpatrick et al., 2017)."""
    fisher = {name: np.zeros_like(p) for name, p in net.params().items()}
    idx = rng.choice(len(X), size=min(n_samples, len(X)), replace=False)
    for i in idx:
        logits, cache = net.forward(X[i])
        loss, dlogits = ce_loss_and_dlogits(logits, prevs[i], y[i])
        grads = net.backward_logits_grad(cache, dlogits)
        for name in fisher:
            fisher[name] += grads[name] ** 2
    for name in fisher:
        fisher[name] /= len(idx)
    return fisher


# ============================================================================
# 4. Da Ma policy: a small stochastic race game with bonus/trap cells,
#    trained by REINFORCE, benchmarked against a uniform-random baseline.
#    This operationalises "Da Ma Fu"'s claim that a dice game is not, in
#    fact, pure chance once its rule structure is exploited.
# ============================================================================

TRACK_LEN = 20
BONUS_CELLS = {4, 9, 15}
TRAP_CELLS = {6, 12, 17}
N_POLICY_IN = 7
N_POLICY_HIDDEN = 8
N_POLICY_OUT = 2  # which of the two pieces to move


def policy_features(pos, dice):
    """
    The decision this turn -- which piece to move -- can only be made
    intelligently if the network can see what THIS roll would do to each
    piece: moving piece i by `dice` may land it exactly on a bonus cell
    (good), a trap cell (bad), or neither. Position alone (without the
    roll) does not determine this, so the roll and its lookahead
    consequences are included explicitly.
    """
    p0, p1 = pos
    land0 = min(p0 + dice, TRACK_LEN)
    land1 = min(p1 + dice, TRACK_LEN)
    return np.array([
        p0 / TRACK_LEN, p1 / TRACK_LEN, dice / 6.0,
        1.0 if land0 in BONUS_CELLS else 0.0,
        1.0 if land0 in TRAP_CELLS else 0.0,
        1.0 if land1 in BONUS_CELLS else 0.0,
        1.0 if land1 in TRAP_CELLS else 0.0,
    ])


def play_episode(policy_net, r, greedy=False, random_policy=False, max_turns=60):
    """
    Returns:
      decisions: list of (turn_index, cache, action, probs) for turns where
                 the policy made a genuine two-way choice (both pieces still
                 in play). Forced turns (one piece already home, so the
                 other one *must* move) are excluded from the learning
                 signal entirely, since they carry no decision information.
      rewards:   per-turn reward, one entry per turn (forced turns included,
                 since they still affect the return that earlier genuine
                 decisions are credited with).
      turns:     number of turns played.
    """
    pos = [0, 0]
    decisions, rewards = [], []
    turns = 0
    while (pos[0] < TRACK_LEN or pos[1] < TRACK_LEN) and turns < max_turns:
        dice = int(r.integers(1, 7))
        feat = policy_features(pos, dice)
        forced_action = None
        if pos[0] >= TRACK_LEN:
            forced_action = 1
        elif pos[1] >= TRACK_LEN:
            forced_action = 0

        if forced_action is not None:
            action = forced_action
        elif random_policy:
            action = int(r.integers(0, 2))
        else:
            logits, cache = policy_net.forward(feat)
            probs = softmax(logits)
            if greedy:
                action = int(np.argmax(probs))
            else:
                action = int(r.choice(2, p=probs))
            decisions.append((turns, cache, action, probs))

        newpos = min(pos[action] + dice, TRACK_LEN)
        reward = -1.0
        if newpos in BONUS_CELLS:
            reward += 3.0
        if newpos in TRAP_CELLS:
            newpos = max(0, newpos - 3)
            reward -= 3.0
        pos[action] = newpos
        rewards.append(reward)
        turns += 1

    finished_bonus = 10.0 if (pos[0] >= TRACK_LEN and pos[1] >= TRACK_LEN) else 0.0
    if rewards:
        rewards[-1] += finished_bonus
    return decisions, rewards, turns


def discounted_returns(rewards, gamma=0.95):
    G = 0.0
    out = np.zeros(len(rewards))
    for t in reversed(range(len(rewards))):
        G = rewards[t] + gamma * G
        out[t] = G
    return out


def train_policy(policy_net, episodes, lr, seed):
    r = np.random.default_rng(seed)
    baseline = 0.0
    history = []
    for ep in range(episodes):
        decisions, rewards, turns = play_episode(policy_net, r)
        returns = discounted_returns(rewards)
        total_return = returns[0] if len(returns) else 0.0
        baseline = 0.98 * baseline + 0.02 * total_return
        for turn_idx, cache, act, probs in decisions:
            G = returns[turn_idx]
            advantage = G - baseline
            # loss := -advantage * log pi(act|state); descending this loss
            # ascends the REINFORCE objective. d(loss)/dz = advantage * (p - onehot(act))
            dlogits = probs.copy()
            dlogits[act] -= 1.0
            dlogits = dlogits * advantage
            grads = policy_net.backward_logits_grad(cache, dlogits)
            policy_net.apply_grad(grads, lr)
        history.append(turns)
    return history


def evaluate_policy(policy_net, n_episodes, seed, random_policy=False):
    r = np.random.default_rng(seed)
    turns_list, reward_list = [], []
    for _ in range(n_episodes):
        _, rewards, turns = play_episode(policy_net, r, greedy=not random_policy,
                                          random_policy=random_policy)
        turns_list.append(turns)
        reward_list.append(sum(rewards))
    return np.mean(turns_list), np.mean(reward_list)


# ============================================================================
# 5. Mandatory finite-difference gradient check.
# ============================================================================

def test_gradient_check():
    net = TwoLayerNet(N_IN, N_HIDDEN, N_REGISTERS, seed=7)
    x = rng.normal(size=N_IN)
    prev = 2
    target = 3
    logits, cache = net.forward(x)
    loss, dlogits = ce_loss_and_dlogits(logits, prev, target)
    analytic = net.backward_logits_grad(cache, dlogits)

    eps = 1e-5
    max_rel_err = 0.0
    flat = net.get_flat()
    for name, p in net.params().items():
        it = np.nditer(p, flags=["multi_index"])
        # sample a subset of indices per tensor to keep runtime small but
        # cover every parameter tensor at least a few times
        idxs = list(np.ndindex(p.shape))
        sample = idxs if len(idxs) <= 6 else [idxs[i] for i in
                 np.linspace(0, len(idxs) - 1, 6).astype(int)]
        for ix in sample:
            orig = p[ix]
            p[ix] = orig + eps
            logits_p, _ = net.forward(x)
            loss_p, _ = ce_loss_and_dlogits(logits_p, prev, target)
            p[ix] = orig - eps
            logits_m, _ = net.forward(x)
            loss_m, _ = ce_loss_and_dlogits(logits_m, prev, target)
            p[ix] = orig
            numeric = (loss_p - loss_m) / (2 * eps)
            an = analytic[name][ix]
            denom = max(1e-8, abs(numeric) + abs(an))
            rel_err = abs(numeric - an) / denom
            max_rel_err = max(max_rel_err, rel_err)
    assert max_rel_err < 1e-4, f"gradient check failed, max rel err={max_rel_err}"
    return max_rel_err


def test_policy_gradient_check():
    net = TwoLayerNet(N_POLICY_IN, N_POLICY_HIDDEN, N_POLICY_OUT, seed=11)
    x = rng.normal(size=N_POLICY_IN)
    logits, cache = net.forward(x)
    probs = softmax(logits)
    action = 1
    advantage = 1.7
    # loss := -advantage * log p[action]; descending this loss is equivalent
    # to ascending the REINFORCE objective advantage * log p[action].
    # d(loss)/dz = advantage * (p - onehot(action))
    dlogits = probs.copy()
    dlogits[action] -= 1.0
    dlogits = dlogits * advantage
    analytic = net.backward_logits_grad(cache, dlogits)

    def surrogate_loss(net_x_logits):
        p = softmax(net_x_logits)
        return -advantage * np.log(p[action] + 1e-12)

    eps = 1e-5
    max_rel_err = 0.0
    for name, p in net.params().items():
        idxs = list(np.ndindex(p.shape))
        sample = idxs if len(idxs) <= 6 else [idxs[i] for i in
                 np.linspace(0, len(idxs) - 1, 6).astype(int)]
        for ix in sample:
            orig = p[ix]
            p[ix] = orig + eps
            lp, _ = net.forward(x)
            loss_p = surrogate_loss(lp)
            p[ix] = orig - eps
            lm, _ = net.forward(x)
            loss_m = surrogate_loss(lm)
            p[ix] = orig
            numeric = (loss_p - loss_m) / (2 * eps)
            an = analytic[name][ix]
            denom = max(1e-8, abs(numeric) + abs(an))
            rel_err = abs(numeric - an) / denom
            max_rel_err = max(max_rel_err, rel_err)
    assert max_rel_err < 1e-4, f"policy gradient check failed, max rel err={max_rel_err}"
    return max_rel_err


# ============================================================================
# 6. Full self-test / demonstration run.
# ============================================================================

def main():
    print("=" * 78)
    print("TLCN -- Tone-Lattice Consolidation Network (Figure 0237, Li Qingzhao)")
    print("=" * 78)

    print("\n[1] Finite-difference gradient check -- tonal encoder ...")
    err1 = test_gradient_check()
    print(f"    max relative error = {err1:.3e}  (pass, < 1e-4)")

    print("[1b] Finite-difference gradient check -- Da Ma policy net ...")
    err2 = test_policy_gradient_check()
    print(f"    max relative error = {err2:.3e}  (pass, < 1e-4)")

    print("\n[2] Grammar-mask enforcement check ...")
    net_check = TwoLayerNet(N_IN, N_HIDDEN, N_REGISTERS, seed=3)
    violations = 0
    for prev in range(N_REGISTERS):
        x = build_feature(prev, sample_affect("A", rng), CATALOGUE_A)
        logits, _ = net_check.forward(x)
        p = masked_softmax_from(logits, prev)
        for nxt in range(N_REGISTERS):
            if not ALLOWED[prev, nxt] and p[nxt] > 1e-9:
                violations += 1
    assert violations == 0, "grammar mask leaked probability onto an illegal transition"
    print(f"    illegal-transition probability mass found in {violations} cases (pass)")

    print("\n[3] Task A ('first collection', pre-1127 affect) -- training encoder ...")
    Xa_tr, ya_tr, pa_tr = gen_dataset("A", TEACHER_A, CATALOGUE_A, 800, seed=100)
    Xa_te, ya_te, pa_te = gen_dataset("A", TEACHER_A, CATALOGUE_A, 200, seed=101)
    oracle_bound = np.mean([
        masked_softmax_from(TEACHER_A.forward(Xa_te[i])[0] * TEACHER_TEMP, pa_te[i]).max()
        for i in range(len(Xa_te))
    ])
    student = TwoLayerNet(N_IN, N_HIDDEN, N_REGISTERS, seed=42)
    train_encoder(student, Xa_tr, ya_tr, pa_tr, epochs=40, lr=0.08)
    acc_a_before = eval_accuracy(student, Xa_te, ya_te, pa_te)
    print(f"    Task-A held-out accuracy after training on Task A: {acc_a_before:.3f} "
          f"(chance level ~ {1 / N_REGISTERS:.3f}; teacher's own stochastic "
          f"ceiling ~ {oracle_bound:.3f})")
    assert acc_a_before > 0.55, "encoder failed to learn Task A above a reasonable threshold"

    theta_star = {k: v.copy() for k, v in student.params().items()}
    fisher = compute_fisher_diag(student, Xa_tr, pa_tr, ya_tr, n_samples=200)

    print("\n[4] Simulating the catastrophe: fine-tuning on Task B "
          "('second collection', post-1127 affect) ...")
    Xb_tr, yb_tr, pb_tr = gen_dataset("B", TEACHER_B, CATALOGUE_B, 800, seed=200)

    # (a) without consolidation -- undifferentiated overwrite
    student_no_ewc = TwoLayerNet(N_IN, N_HIDDEN, N_REGISTERS, seed=42)
    student_no_ewc.set_flat(student.get_flat())  # start from the Task-A-trained weights
    train_encoder(student_no_ewc, Xb_tr, yb_tr, pb_tr, epochs=40, lr=0.08, ewc=None)
    acc_a_after_plain = eval_accuracy(student_no_ewc, Xa_te, ya_te, pa_te)
    acc_b_plain = eval_accuracy(student_no_ewc, Xb_tr, yb_tr, pb_tr)

    # (b) with elastic consolidation ("Jinshi Lu" catalogue-weighted protection)
    student_ewc = TwoLayerNet(N_IN, N_HIDDEN, N_REGISTERS, seed=42)
    student_ewc.set_flat(student.get_flat())
    train_encoder(student_ewc, Xb_tr, yb_tr, pb_tr, epochs=40, lr=0.08,
                  ewc={"fisher": fisher, "theta_star": theta_star, "lambda": 6.0})
    acc_a_after_ewc = eval_accuracy(student_ewc, Xa_te, ya_te, pa_te)
    acc_b_ewc = eval_accuracy(student_ewc, Xb_tr, yb_tr, pb_tr)

    print(f"    Task-A retention WITHOUT consolidation after learning Task B: "
          f"{acc_a_after_plain:.3f}  (Task-B fit: {acc_b_plain:.3f})")
    print(f"    Task-A retention WITH    consolidation after learning Task B: "
          f"{acc_a_after_ewc:.3f}  (Task-B fit: {acc_b_ewc:.3f})")
    assert acc_a_after_ewc > acc_a_after_plain + 0.05, (
        "consolidation did not measurably protect Task-A knowledge relative "
        "to the undifferentiated-overwrite condition"
    )
    print("    -> consolidation retains first-collection competence significantly "
          "better than undifferentiated overwrite (pass)")

    print("\n[5] Da Ma policy: REINFORCE training vs. uniform-random baseline ...")
    policy = TwoLayerNet(N_POLICY_IN, N_POLICY_HIDDEN, N_POLICY_OUT, scale=0.7, seed=55)
    train_policy(policy, episodes=1500, lr=0.01, seed=300)
    mean_turns_trained, mean_reward_trained = evaluate_policy(policy, 300, seed=400)
    mean_turns_random, mean_reward_random = evaluate_policy(policy, 300, seed=401,
                                                              random_policy=True)
    print(f"    trained policy : mean turns to finish = {mean_turns_trained:.2f}, "
          f"mean episode reward = {mean_reward_trained:.2f}")
    print(f"    random baseline: mean turns to finish = {mean_turns_random:.2f}, "
          f"mean episode reward = {mean_reward_random:.2f}")
    assert mean_reward_trained > mean_reward_random + 2.0, (
        "trained Da Ma policy did not significantly outperform the random baseline"
    )
    print("    -> learned policy exploits the bonus/trap structure and beats chance "
          "(pass, consistent with the thesis of 'Da Ma Fu')")

    print("\nAll self-tests passed.")
    print("=" * 78)


if __name__ == "__main__":
    main()
