"""
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0283 · Anna Comnena
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 15, Minds 281-300 Available on Amazon https://www.amazon.com/dp/B0HMQ7TVJM
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0283_anna_comnena_1083 - Anna Comnena (c.1083-c.1153)

THE INTERESTED WITNESS
A Provenance-Weighted Testimony Reconciliation Network (PWTRN)
modelled on the documented cognitive method of Anna Komnene (1083 - c.1153)

--------------------------------------------------------------------------
WHY THIS ARCHITECTURE, AND NOT A GENERIC TRANSFORMER
--------------------------------------------------------------------------
Anna Komnene's own surviving practice -- not a modern gloss on it -- gives
two distinctive, documented cognitive operations:

  (1) BIAS-DECLARED RECONCILIATION. In the Alexiad's own prologue and in
      book XIV.7, Anna states outright that she is an "interested witness"
      -- Alexios's daughter, writing in grief, thirty years after most of
      the events -- and that this makes impartiality difficult even though
      she is trying for it. She names her sources by kind: her own
      childhood memory, her own adult memory, the oral testimony of her
      father's veteran soldiers gathered around her at the Kecharitomene
      monastery (XIV.vii), written court records, and -- pointedly -- a
      RIVAL written account, her own late husband Nikephoros Bryennios's
      unfinished "Hyle Historias" ("Material for History"), which favours
      HIS grandfather over HER father. She does not pretend these sources
      are equally reliable, and she does not pretend her own testimony is
      unbiased. She tags reliability by provenance and by interest.

  (2) GAP-DIRECTED COMMISSIONING. After 1118 Anna organised and funded a
      circle of scholars (chiefly Michael of Ephesus, probably also
      Eustratios of Nicaea) whose explicit brief was to write commentaries
      on exactly the works of Aristotle that NO ONE HAD YET COMMENTED ON --
      principally the biological corpus (Robert Browning, "An Unpublished
      Funeral Oration on Anna Comnena", Proc. Camb. Philol. Soc. n.s. 8
      (1962); Katerina Ierodiakonou, "Byzantine Philosophy", Stanford
      Encyclopedia of Philosophy). The operative move is not synthesis of
      what is already known -- it is mapping the shape of what is NOT yet
      covered, and directing scarce expert effort there.

Neither operation is "intelligence imposes order on chaos" (the corpus's
overused default), nor Galen's pneumatic-hydraulic routing (id 136), nor
Thucydides's myth-free compact world-model (id 65), nor Sima Qian's
distributed-truth multi-biography solution (id 101). It is narrower and
more specific than any of those: TAG EVERY CLAIM WITH WHO MADE IT AND HOW
MUCH THEY STAND TO GAIN FROM SHADING IT, DISCOUNT ACCORDINGLY, REFUSE TO
INVENT A CHRONOLOGY THE EVIDENCE DOES NOT SUPPORT, AND SPEND NEW EFFORT ON
THE PARTS OF THE RECORD THAT ARE STILL BLANK RATHER THAN THE PARTS ALREADY
WELL ATTESTED.

This file implements exactly that, as a small multi-task network, trained
on a synthetic corpus of "witnessed events" standing in for the Alexiad's
source situation:

  Stage 0  GAP MAP        (non-parametric): from a topic x time coverage
                            grid, compute which cells are under-attested
                            and should receive commissioned effort.
  Stage 1  ENCODE          claim content -> claim embedding (Linear+tanh)
  Stage 2  CREDIT          provenance-type embedding + a learned, MONOTONE
                            self-interest discount -> per-claim credibility
                            in (0,1). This is the "declared bias" term.
  Stage 3  POOL            per-event weighted pooling of claim embeddings,
                            weight = credibility x gap-priority^temperature,
                            softmax-normalised within each event -> event
                            embedding. (Gap-priority raises the vote of
                            claims from under-attested cells, mirroring
                            commissioning effort at the lacuna.)
  Stage 4  RECONCILE       event embedding -> predicted "what actually
                            happened" (regression head).
  Stage 5  AUDIT           (claim embedding, credibility) -> predicted
                            distortion magnitude of that specific claim
                            (a supervised probe standing in for Anna's own
                            explicit "discount this, it is my father"
                            asides).
  Stage 6  SEQUENCE        pairs of event embeddings + coarse time hints
                            -> {before, after, UNKNOWN}. The UNKNOWN class
                            is not an error class, it is a first-class
                            output: Anna explicitly refuses a chronology
                            the witnesses cannot support (Alexiad XIV.7,
                            on events that "did not come to me in the
                            order in which I have set them down").

Everything below is pure NumPy. There is no autograd library: Stage 1-6
are built on a ~250-line reverse-mode autodiff engine written for this
file (class Tensor), so that the whole pipeline is a single differentiable
graph trained end to end by real backpropagation, gradient-checked against
finite differences before a single training step is trusted.

Run directly:  python3 chapter_0283_anna_comnena_1083.py
"""

import numpy as np

RNG = np.random.default_rng(1083)

# ==========================================================================
# PART 1 -- MINIMAL REVERSE-MODE AUTODIFF ENGINE (pure NumPy, from scratch)
# ==========================================================================


def _unbroadcast(grad, shape):
    """Sum a gradient back down to `shape` after NumPy broadcasting."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad.reshape(shape)


class Tensor:
    """A NumPy array with a backward() that runs real backpropagation."""

    __slots__ = ("data", "grad", "_backward", "_prev", "op", "requires_grad")

    def __init__(self, data, children=(), op="", requires_grad=True):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(children)
        self.op = op
        self.requires_grad = requires_grad

    # ---- shape helpers ----
    @property
    def shape(self):
        return self.data.shape

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, op={self.op})"

    # ---- arithmetic ----
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += _unbroadcast(out.grad, self.data.shape)
            other.grad += _unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    __radd__ = __add__

    def __neg__(self):
        out = Tensor(-self.data, (self,), "neg")

        def _backward():
            self.grad += -out.grad

        out._backward = _backward
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        return self + (-other)

    def __rsub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        return other + (-self)

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += _unbroadcast(out.grad * other.data, self.data.shape)
            other.grad += _unbroadcast(out.grad * self.data, other.data.shape)

        out._backward = _backward
        return out

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        return self * (other ** -1.0)

    def __pow__(self, p):
        assert isinstance(p, (int, float))
        out = Tensor(self.data ** p, (self,), f"**{p}")

        def _backward():
            self.grad += _unbroadcast((p * (self.data ** (p - 1))) * out.grad, self.data.shape)

        out._backward = _backward
        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data @ other.data, (self, other), "@")

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    # ---- indexing (row gather, used to build per-event / per-pair groups) ----
    def __getitem__(self, idx):
        out = Tensor(self.data[idx], (self,), "getitem")

        def _backward():
            np.add.at(self.grad, idx, out.grad)

        out._backward = _backward
        return out

    # ---- elementwise nonlinearities ----
    def exp(self):
        e = np.exp(self.data)  # inputs to exp() are bounded in this model, no shift needed
        out = Tensor(e, (self,), "exp")

        def _backward():
            self.grad += e * out.grad

        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(np.clip(self.data, 1e-12, None)), (self,), "log")

        def _backward():
            self.grad += out.grad / np.clip(self.data, 1e-12, None)

        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t * t) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + np.exp(-self.data))
        out = Tensor(s, (self,), "sigmoid")

        def _backward():
            self.grad += s * (1 - s) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(0.0, self.data), (self,), "relu")

        def _backward():
            self.grad += (self.data > 0) * out.grad

        out._backward = _backward
        return out

    def softplus(self):
        # log(1+exp(x)), numerically stable
        out = Tensor(np.logaddexp(0.0, self.data), (self,), "softplus")

        def _backward():
            self.grad += (1.0 / (1.0 + np.exp(-self.data))) * out.grad

        out._backward = _backward
        return out

    # ---- reductions ----
    def sum(self):
        out = Tensor(self.data.sum(), (self,), "sum")

        def _backward():
            self.grad += np.ones_like(self.data) * out.grad

        out._backward = _backward
        return out

    def mean(self):
        n = self.data.size
        out = Tensor(self.data.mean(), (self,), "mean")

        def _backward():
            self.grad += (np.ones_like(self.data) / n) * out.grad

        out._backward = _backward
        return out

    # ---- reshape (view), needed to softmax a single event's claim-weights ----
    def reshape(self, *shape):
        orig_shape = self.data.shape
        out = Tensor(self.data.reshape(*shape), (self,), "reshape")

        def _backward():
            self.grad += out.grad.reshape(orig_shape)

        out._backward = _backward
        return out

    # ---- sum over axis 0, keeping dims (used for weighted pooling) ----
    def sum0(self):
        out = Tensor(self.data.sum(axis=0, keepdims=True), (self,), "sum0")

        def _backward():
            self.grad += np.ones_like(self.data) * out.grad

        out._backward = _backward
        return out

    # ---- row-wise softmax over the last axis ----
    def softmax_rows(self):
        x = self.data
        x = x - x.max(axis=-1, keepdims=True)
        e = np.exp(x)
        s = e / e.sum(axis=-1, keepdims=True)
        out = Tensor(s, (self,), "softmax_rows")

        def _backward():
            # standard softmax-jacobian-vector product, per row
            g = out.grad
            dot = np.sum(g * s, axis=-1, keepdims=True)
            self.grad += s * (g - dot)

        out._backward = _backward
        return out

    # ---- backward pass over the whole graph ----
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


def cat(tensors, axis=1):
    """Concatenate a list of Tensors along `axis`, differentiably."""
    data = np.concatenate([t.data for t in tensors], axis=axis)
    out = Tensor(data, tuple(tensors), "cat")
    sizes = [t.data.shape[axis] for t in tensors]

    def _backward():
        start = 0
        for t, sz in zip(tensors, sizes):
            sl = [slice(None)] * data.ndim
            sl[axis] = slice(start, start + sz)
            t.grad += out.grad[tuple(sl)]
            start += sz

    out._backward = _backward
    return out


def cross_entropy(logits, labels):
    """logits: Tensor [N,C]; labels: int array [N]. Returns mean CE loss Tensor."""
    x = logits.data
    x = x - x.max(axis=-1, keepdims=True)
    logp = x - np.log(np.exp(x).sum(axis=-1, keepdims=True))
    n = x.shape[0]
    loss_val = -logp[np.arange(n), labels].mean()
    out = Tensor(loss_val, (logits,), "cross_entropy")

    def _backward():
        p = np.exp(logp)
        onehot = np.zeros_like(p)
        onehot[np.arange(n), labels] = 1.0
        logits.grad += (p - onehot) / n * out.grad

    out._backward = _backward
    return out


def mse(pred, target):
    target = target if isinstance(target, Tensor) else Tensor(target, requires_grad=False)
    diff = pred - target
    return (diff * diff).mean()


# ==========================================================================
# PART 2 -- SYNTHETIC ALEXIAD-STYLE TESTIMONY CORPUS
# ==========================================================================
#
# Provenance types, matching what Anna herself names as her source kinds:
#   0 SELF_CHILD    her own memory as a child/young woman at court
#   1 SELF_ADULT    her own memory as an adult (campaigns, her father's death)
#   2 VETERAN       oral testimony of Alexios's retired soldiers (Alex. XIV.vii)
#   3 RIVAL_WRITTEN her husband Nikephoros Bryennios's "Hyle Historias",
#                   which favours HIS grandfather over HER father
#   4 COURT_DOC     chrysobulls, treaties, administrative record
#   5 HEARSAY       secondhand report from a non-participant
#
PROVENANCE_NAMES = ["SELF_CHILD", "SELF_ADULT", "VETERAN", "RIVAL_WRITTEN", "COURT_DOC", "HEARSAY"]
N_PROV = len(PROVENANCE_NAMES)

TOPIC_NAMES = ["norman_wars", "pecheneg_wars", "first_crusade", "court_politics", "fathers_illness"]
N_TOPICS = len(TOPIC_NAMES)
N_TIMEBINS = 8  # coarse decade-ish bins across 1081-1118

# Baseline affective interest (how much a source-kind stands to gain from
# shading its report) and distortion direction (+1 = glorify Alexios,
# -1 = diminish him) -- this is the numeric form of Anna's own declared
# "this witness is not neutral" bookkeeping.
PROV_BASE_INTEREST = {
    0: 0.55,   # child memory: fond, but young and less invested in the record
    1: 0.85,   # adult memory: written decades later, in grief, of a beloved father
    2: 0.40,   # veteran soldiers: loyal to their old commander, moderately so
    3: 0.70,   # rival written account: strongly interested, but AGAINST Alexios
    4: 0.15,   # court document: low personal stake (some institutional gloss)
    5: 0.30,   # hearsay: low stake, but high variance
}
PROV_DISTORTION_DIRECTION = {0: +1, 1: +1, 2: +1, 3: -1, 4: +1, 5: 0}
PROV_NOISE_SCALE = {0: 0.22, 1: 0.15, 2: 0.18, 3: 0.18, 4: 0.10, 5: 0.35}

CONTENT_DIM = 6  # [reported_value, 5 narrative-texture features]


def make_corpus(n_events=36, seed=1083):
    """Build a synthetic corpus of events, each attested by 1-5 claims of
    varying provenance, deliberately leaving some (topic,time_bin) cells
    empty (a genuine gap) and others richly covered."""
    rng = np.random.default_rng(seed)
    events = []
    claims = []  # each: dict(event_id, provenance, affective_interest, content[6], reported_value)

    # Deliberately-planted coverage pattern: fathers_illness is covered almost
    # only by SELF_ADULT + COURT_DOC (a handful of witnesses, near her at the
    # end); court_politics around the succession is covered richly by rival
    # and self accounts (contested); one whole (topic,time_bin) cell is left
    # at ZERO witnesses to stand in for a genuine, undocumented lacuna.
    planted_gap_topic, planted_gap_time = 1, 6  # "pecheneg_wars", late bin: no witnesses at all

    eid = 0
    for topic in range(N_TOPICS):
        for tbin in range(N_TIMEBINS):
            if topic == planted_gap_topic and tbin == planted_gap_time:
                continue  # genuine lacuna: no event generated here at all
            # not every (topic,timebin) cell has an event; keep it sparse/uneven
            if rng.random() < 0.22:
                continue
            true_time = tbin + rng.uniform(-0.4, 0.4)
            true_value = rng.uniform(-1.0, 1.0)  # "what actually happened", hidden from the model
            narrative_texture = rng.normal(0, 1, size=5)

            # how many / which kinds of witnesses saw this event
            if topic == 4:  # fathers_illness: only she and the record were present
                prov_pool = [1, 1, 4]
            elif topic == 3:  # court_politics: contested, both sides write about it
                prov_pool = [1, 3, 4, 5]
            elif topic == 2:  # first_crusade: veterans + her own presence
                prov_pool = [1, 2, 2, 5]
            else:  # norman/pecheneg campaigns: veterans dominate, some rival, some hearsay
                prov_pool = [0, 2, 2, 3, 5]

            n_witness = rng.integers(1, min(len(prov_pool), 4) + 1)
            chosen = rng.choice(prov_pool, size=n_witness, replace=False)

            for prov in chosen:
                prov = int(prov)
                interest = float(np.clip(PROV_BASE_INTEREST[prov] + rng.normal(0, 0.08), 0.02, 0.98))
                direction = PROV_DISTORTION_DIRECTION[prov]
                noise = rng.normal(0, PROV_NOISE_SCALE[prov])
                distortion = direction * interest * 0.9 + noise
                reported_value = float(np.clip(true_value + distortion, -2.5, 2.5))
                content = np.concatenate([[reported_value], narrative_texture + rng.normal(0, 0.25, 5)])
                claims.append(dict(
                    event_id=eid, provenance=prov, affective_interest=interest,
                    content=content, reported_value=reported_value,
                    true_distortion=abs(distortion),
                ))

            events.append(dict(
                event_id=eid, topic=topic, time_bin=tbin, true_time=true_time,
                true_value=true_value, n_witness=n_witness,
            ))
            eid += 1

    return events, claims, (planted_gap_topic, planted_gap_time)


# ==========================================================================
# PART 3 -- GAP MAP (non-parametric: "where is the record thin or blank?")
# ==========================================================================

def gap_priority_map(events, n_topics=N_TOPICS, n_timebins=N_TIMEBINS):
    """Coverage density per (topic,time_bin) cell -> priority for commissioned
    attention, inversely related to density. This is deliberately NOT a
    learned component: identifying a hole in the record is an observation
    about the record, not a fitted judgement -- exactly how Anna and Michael
    of Ephesus could SEE which of Aristotle's works had no commentary
    without needing to be "trained" to see it."""
    density = np.zeros((n_topics, n_timebins))
    for ev in events:
        density[ev["topic"], ev["time_bin"]] += ev["n_witness"]
    priority = 1.0 / (1.0 + density)
    priority = priority / priority.max()
    return density, priority


def build_pairs(events, margin=0.6, n_pairs=250, seed=99):
    """Sample event pairs and label them {0:before, 1:after, 2:UNKNOWN} from
    the HIDDEN true_time. The model never sees true_time -- only a coarse
    time_bin and an uncertainty feature derived from witness count -- so it
    must learn to output class 2 (refuse to order) exactly when the margin
    check below would also refuse. This is the "chronology I cannot fully
    reconstruct" admission from Alexiad XIV.7, made into a supervised class."""
    rng = np.random.default_rng(seed)
    ids = [e["event_id"] for e in events]
    events_by_id = {e["event_id"]: e for e in events}
    n = len(ids)
    n_pairs = min(n_pairs, n * (n - 1) // 2)
    seen, pairs = set(), []
    while len(pairs) < n_pairs:
        a, b = rng.choice(n, size=2, replace=False)
        ida, idb = ids[a], ids[b]
        key = (min(ida, idb), max(ida, idb))
        if key in seen:
            continue
        seen.add(key)
        ta, tb = events_by_id[ida]["true_time"], events_by_id[idb]["true_time"]
        if ta < tb - margin:
            label = 0
        elif ta > tb + margin:
            label = 1
        else:
            label = 2
        pairs.append((ida, idb, label))
    return pairs


# ==========================================================================
# PART 4 -- MODEL: PARAMETERS AND FORWARD PASS
# ==========================================================================

D_PROV = 4    # provenance-type embedding dimension
D_HID = 8     # claim / event embedding dimension


def init_params(seed=7):
    rng = np.random.default_rng(seed)

    def rnd(*shape, scale=0.3):
        return rng.normal(0, scale, size=shape)

    return {
        "E_prov": rnd(N_PROV, D_PROV),
        "W_cred": rnd(D_PROV + 1, 1),
        "b_cred": np.zeros((1, 1)),
        "W_enc": rnd(CONTENT_DIM, D_HID),
        "b_enc": np.zeros((1, D_HID)),
        "t_raw": np.array([[0.5]]),
        "W_out": rnd(D_HID, 1),
        "b_out": np.zeros((1, 1)),
        "W_dist1": rnd(D_HID + 1, 8),
        "b_dist1": np.zeros((1, 8)),
        "W_dist2": rnd(8, 1),
        "b_dist2": np.zeros((1, 1)),
        "W_pair1": rnd(2 * (D_HID + 2), 16),
        "b_pair1": np.zeros((1, 16)),
        "W_pair2": rnd(16, 3),
        "b_pair2": np.zeros((1, 3)),
    }


def build_tensors(params_np):
    return {k: Tensor(v, requires_grad=True) for k, v in params_np.items()}


def forward(pt, claims, events, gap_priority, pairs):
    """One full forward pass: Stage1 encode -> Stage2 credit -> Stage3 pool
    -> Stage4 reconcile -> Stage5 audit -> Stage6 sequence. Returns a dict
    with the total loss Tensor and everything needed for inspection/tests."""
    events_by_id = {e["event_id"]: e for e in events}

    content_mat = np.stack([c["content"] for c in claims]).astype(np.float64)
    prov_idx = np.array([c["provenance"] for c in claims])
    interest = np.array([[c["affective_interest"]] for c in claims])
    true_distortion = np.array([[c["true_distortion"]] for c in claims])
    claim_event_id = np.array([c["event_id"] for c in claims])
    gap_of_claim = np.array([
        [gap_priority[events_by_id[eid]["topic"], events_by_id[eid]["time_bin"]]]
        for eid in claim_event_id
    ])

    # ---- Stage 1: ENCODE claim content -> claim embedding ----
    content_t = Tensor(content_mat, requires_grad=False)
    h = (content_t @ pt["W_enc"] + pt["b_enc"]).tanh()          # (Nc, D_HID)

    # ---- Stage 2: CREDIT -- provenance embedding + declared self-interest ----
    prov_emb = pt["E_prov"][prov_idx]                            # (Nc, D_PROV)
    interest_t = Tensor(interest, requires_grad=False)           # (Nc, 1)
    cred_in = cat([prov_emb, interest_t], axis=1)                # (Nc, D_PROV+1)
    credibility = (cred_in @ pt["W_cred"] + pt["b_cred"]).sigmoid()  # (Nc,1)

    # ---- gap-priority raises the vote of under-attested claims ----
    gap_t = Tensor(gap_of_claim, requires_grad=False)             # (Nc,1), in (0,1]
    temperature = pt["t_raw"].softplus()                          # (1,1), > 0
    gap_weight = (gap_t.log() * temperature).exp()                # gap ** temperature
    raw_weight = credibility * gap_weight                         # (Nc,1)

    # ---- Stage 3: POOL claims into one embedding per event ----
    event_ids_sorted = sorted(events_by_id.keys())
    id_to_row = {eid: i for i, eid in enumerate(event_ids_sorted)}
    pooled_rows = []
    for eid in event_ids_sorted:
        idx = np.where(claim_event_id == eid)[0]
        w_grp = raw_weight[idx].reshape(1, len(idx))
        w_norm = w_grp.softmax_rows().reshape(len(idx), 1)   # normalised within THIS event only
        pooled_rows.append((h[idx] * w_norm).sum0())          # (1, D_HID)
    event_emb = cat(pooled_rows, axis=0)                       # (n_events, D_HID)

    # ---- Stage 4: RECONCILE -- predict "what actually happened" ----
    event_val_pred = (event_emb @ pt["W_out"] + pt["b_out"]).tanh()
    true_value_target = np.array([[events_by_id[eid]["true_value"]] for eid in event_ids_sorted])
    loss_value = mse(event_val_pred, true_value_target)

    # ---- Stage 5: AUDIT -- how much did THIS claim likely distort? ----
    dist_in = cat([h, credibility], axis=1)                    # (Nc, D_HID+1)
    d1 = (dist_in @ pt["W_dist1"] + pt["b_dist1"]).relu()
    dist_pred = (d1 @ pt["W_dist2"] + pt["b_dist2"]).softplus()
    loss_dist = mse(dist_pred, true_distortion)

    # ---- Stage 6: SEQUENCE -- before / after / UNKNOWN ----
    n_witness_arr = np.array([events_by_id[eid]["n_witness"] for eid in event_ids_sorted]).reshape(-1, 1)
    time_bin_arr = (np.array([events_by_id[eid]["time_bin"] for eid in event_ids_sorted])
                     .reshape(-1, 1).astype(np.float64) / max(1, N_TIMEBINS - 1))
    uncertainty_arr = 1.0 / (1.0 + n_witness_arr)
    extra_feat = Tensor(np.concatenate([time_bin_arr, uncertainty_arr], axis=1), requires_grad=False)
    event_feat = cat([event_emb, extra_feat], axis=1)          # (n_events, D_HID+2)

    i_idx = np.array([id_to_row[p[0]] for p in pairs])
    j_idx = np.array([id_to_row[p[1]] for p in pairs])
    labels = np.array([p[2] for p in pairs])
    pair_in = cat([event_feat[i_idx], event_feat[j_idx]], axis=1)
    ph1 = (pair_in @ pt["W_pair1"] + pt["b_pair1"]).relu()
    plogits = ph1 @ pt["W_pair2"] + pt["b_pair2"]               # (P, 3)
    loss_pair = cross_entropy(plogits, labels)

    # L2 regularisation on the weight matrices (not biases/embeddings' raw
    # scale) so that CREDIT learns a graded trust score instead of driving
    # the sigmoid to a saturated 0/1 step -- a discount, not a veto.
    reg_terms = [pt["W_cred"], pt["W_out"], pt["W_pair1"], pt["W_pair2"], pt["W_dist1"], pt["W_dist2"]]
    reg_sum = (reg_terms[0] * reg_terms[0]).sum()
    for w in reg_terms[1:]:
        reg_sum = reg_sum + (w * w).sum()

    total_loss = loss_value * 1.0 + loss_pair * 0.5 + loss_dist * 0.3 + reg_sum * 4e-4

    return dict(
        total_loss=total_loss, loss_value=loss_value, loss_pair=loss_pair, loss_dist=loss_dist,
        credibility=credibility, event_emb=event_emb, event_val_pred=event_val_pred,
        plogits=plogits, event_ids_sorted=event_ids_sorted, prov_idx=prov_idx,
        interest=interest, claim_event_id=claim_event_id, pair_labels=labels,
    )


# ==========================================================================
# PART 5 -- TRAINING LOOP (manual Adam optimiser, no ML framework)
# ==========================================================================

def adam_step(params_np, grads, state, t, lr=0.04, b1=0.9, b2=0.999, eps=1e-8):
    for k in params_np:
        g = grads[k]
        state["m"][k] = b1 * state["m"][k] + (1 - b1) * g
        state["v"][k] = b2 * state["v"][k] + (1 - b2) * (g * g)
        mhat = state["m"][k] / (1 - b1 ** t)
        vhat = state["v"][k] / (1 - b2 ** t)
        params_np[k] -= lr * mhat / (np.sqrt(vhat) + eps)


def train(params_np, claims, events, gap_priority, pairs, iters=400, lr=0.04, verbose=True):
    state = {"m": {k: np.zeros_like(v) for k, v in params_np.items()},
              "v": {k: np.zeros_like(v) for k, v in params_np.items()}}
    history = []
    for t in range(1, iters + 1):
        pt = build_tensors(params_np)
        out = forward(pt, claims, events, gap_priority, pairs)
        out["total_loss"].backward()
        grads = {k: pt[k].grad for k in params_np}
        adam_step(params_np, grads, state, t, lr=lr)
        history.append(float(out["total_loss"].data))
        if verbose and (t == 1 or t % 50 == 0 or t == iters):
            print(f"  iter {t:4d}   total={out['total_loss'].data:.4f}   "
                  f"value={out['loss_value'].data:.4f}   "
                  f"pair={out['loss_pair'].data:.4f}   "
                  f"audit={out['loss_dist'].data:.4f}")
    return history


# ==========================================================================
# PART 6 -- MANDATORY FINITE-DIFFERENCE GRADIENT CHECK
# ==========================================================================

def gradient_check(params_np, claims, events, gap_priority, pairs, eps=1e-5, n_checks=8, seed=0):
    rng = np.random.default_rng(seed)
    keys = list(params_np.keys())
    results = []
    for _ in range(n_checks):
        k = keys[rng.integers(len(keys))]
        arr = params_np[k]
        idx = tuple(rng.integers(0, s) for s in arr.shape)
        orig = arr[idx]

        arr[idx] = orig + eps
        loss_plus = float(forward(build_tensors(params_np), claims, events, gap_priority, pairs)["total_loss"].data)

        arr[idx] = orig - eps
        loss_minus = float(forward(build_tensors(params_np), claims, events, gap_priority, pairs)["total_loss"].data)

        arr[idx] = orig  # restore exactly
        numeric_grad = (loss_plus - loss_minus) / (2 * eps)

        pt = build_tensors(params_np)
        out = forward(pt, claims, events, gap_priority, pairs)
        out["total_loss"].backward()
        analytic_grad = float(pt[k].grad[idx])

        rel_err = abs(numeric_grad - analytic_grad) / max(1e-8, abs(numeric_grad) + abs(analytic_grad))
        results.append((k, idx, numeric_grad, analytic_grad, rel_err))
    return results


# ==========================================================================
# PART 7 -- MAIN (drives training + all self-tests; see PART 6 above for the
#            finite-difference gradient check, run first and asserted on)
# ==========================================================================

def main():
    print("=" * 78)
    print("THE INTERESTED WITNESS")
    print("Provenance-Weighted Testimony Reconciliation Network -- Anna Komnene")
    print("=" * 78)

    events, claims, gap_cell = make_corpus()
    density, priority = gap_priority_map(events)
    pairs = build_pairs(events)
    print(f"\nCorpus: {len(events)} events, {len(claims)} claims, {len(pairs)} ordered/unordered pairs sampled")
    print(f"Planted lacuna at (topic={TOPIC_NAMES[gap_cell[0]]}, time_bin={gap_cell[1]}): "
          f"density={density[gap_cell]:.0f}, gap_priority={priority[gap_cell]:.3f} "
          f"(max possible = 1.000)")

    params = init_params()

    print("\n--- Gradient check (before any training; must pass to trust backprop) ---")
    gc = gradient_check(params, claims, events, priority, pairs, n_checks=8)
    max_rel_err = max(r[4] for r in gc)
    for k, idx, ng, ag, err in gc:
        print(f"  param={k:10s} idx={idx}  numeric={ng: .6f}  analytic={ag: .6f}  rel_err={err:.2e}")
    assert max_rel_err < 2e-3, f"GRADIENT CHECK FAILED: max relative error {max_rel_err:.2e}"
    print(f"  -> PASS  (max relative error {max_rel_err:.2e})")

    print("\n--- Training (pure-NumPy Adam, hand-derived backprop) ---")
    history = train(params, claims, events, priority, pairs, iters=400, lr=0.04)
    assert history[-1] < history[0] * 0.35, "training did not substantially reduce loss"
    print(f"  -> loss fell from {history[0]:.4f} to {history[-1]:.4f} "
          f"({100*(1-history[-1]/history[0]):.1f}% reduction)")

    print("\n--- Test: gap map ranks the planted lacuna above the best-covered cell ---")
    well_covered = np.unravel_index(np.argmax(density), density.shape)
    print(f"  planted lacuna  ({TOPIC_NAMES[gap_cell[0]]}, bin {gap_cell[1]}): "
          f"density={density[gap_cell]:.0f}  priority={priority[gap_cell]:.3f}")
    print(f"  best-covered    ({TOPIC_NAMES[well_covered[0]]}, bin {well_covered[1]}): "
          f"density={density[well_covered]:.0f}  priority={priority[well_covered]:.3f}")
    assert density[gap_cell] == 0 and priority[gap_cell] > priority[well_covered]
    print("  -> PASS  (a genuinely uncovered cell outranks the richest one for commissioned effort)")

    print("\n--- Test: learned credibility discounts self-interested testimony ---")
    pt = build_tensors(params)
    out = forward(pt, claims, events, priority, pairs)
    cred = out["credibility"].data.reshape(-1)
    interest = out["interest"].reshape(-1)
    prov_idx = out["prov_idx"]
    # compare mean credibility of the two HIGH-interest, opposite-direction
    # provenance kinds (SELF_ADULT, RIVAL_WRITTEN) against COURT_DOC (low
    # interest, low distortion in the generative model)
    mean_self_adult = cred[prov_idx == 1].mean()
    mean_rival = cred[prov_idx == 3].mean()
    mean_court = cred[prov_idx == 4].mean()
    print("  learned credibility by provenance (declared baseline self-interest in parens):")
    for p in range(N_PROV):
        mask = prov_idx == p
        if mask.any():
            print(f"    {PROVENANCE_NAMES[p]:14s} (interest~{PROV_BASE_INTEREST[p]:.2f})  "
                  f"mean credibility = {cred[mask].mean():.3f}   n={int(mask.sum())}")
    assert mean_court > mean_self_adult and mean_court > mean_rival, (
        "the model failed to learn that the two high-interest source kinds "
        "should be discounted relative to the low-interest court record")
    print("  -> PASS  (both self-interested source kinds are discounted below the disinterested record,")
    print("           even though they are biased in OPPOSITE directions -- the model penalises")
    print("           declared interest itself, not merely one political side)")

    print("\n--- Test: the model uses UNKNOWN, not confident invention, on genuinely close pairs ---")
    plogits = out["plogits"].data
    pair_labels = out["pair_labels"]
    pred_class = plogits.argmax(axis=1)
    unknown_mask = pair_labels == 2
    acc_unknown = (pred_class[unknown_mask] == 2).mean() if unknown_mask.any() else float("nan")
    acc_ordered = (pred_class[~unknown_mask] == pair_labels[~unknown_mask]).mean()
    print(f"  accuracy on truly-ordered pairs   : {acc_ordered:.3f}  (n={int((~unknown_mask).sum())})")
    print(f"  accuracy recognising UNKNOWN pairs: {acc_unknown:.3f}  (n={int(unknown_mask.sum())})")
    assert acc_ordered > 0.60, "sequence head failed to learn clearly-ordered pairs"
    assert acc_unknown > 0.35, "sequence head never learns to withhold an order it cannot support"
    print("  -> PASS  (the network orders what the evidence supports and withholds an order otherwise)")

    print("\n" + "=" * 78)
    print("ALL SELF-TESTS PASSED")
    print("=" * 78)


if __name__ == "__main__":
    main()
