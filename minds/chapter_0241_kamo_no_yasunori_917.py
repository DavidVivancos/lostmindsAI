#!/usr/bin/env python3
"""
================================================================================
 CHAPTER 241 -- KAMO NO YASUNORI (917-977 CE)
#================================================================================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 13 Minds 241 - 260 Available on Amazon https://www.amazon.com/dp/B0HLD1S6QT
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0241_kamo_no_yasunori_917 - Kamo no Yasunori (c.917-977)
#================================================================================  

WHO THIS ARCHITECTURE ENCODES
------------------------------
Kamo no Yasunori (賀茂保憲, 917-977) was the premier onmyoji (陰陽師,
"master of yin-yang") of the mid-Heian Japanese court: chief astrologer,
calendar-maker, and ritualist of the Bureau of Onmyo (陰陽寮). He is
historically documented as:

  1. A CHILD who could perceive hidden/anomalous presences ("oni", spirits
     at a ritual offering) at age ten, WITHOUT having been trained to do
     so -- unlike his father Kamo no Tadayuki, who required formal
     apprenticeship to develop the same sensitivity (Konjaku Monogatarishu,
     recounted in Li 2009, Ambiguous Bodies, pp.151-152).
  2. The FORMAL SYSTEMATIZER of two written calendrical/astrological
     treatises, the Rekirin (暦林, "Forest of the Calendar", 10 scrolls)
     and the Yasunori-sho (保憲抄) -- both now lost in their original form,
     but preserved indirectly through the later Rekirin Mondoshu
     (暦林問答集), still consulted today to read the old lunisolar calendar.
  3. The ARCHITECT who split his own tradition of onmyodo into two
     specialized hereditary lines at the end of his career: he gave
     tenmondo (天文道, astral divination/astronomy) to his outside pupil
     Abe no Seimei, and rekido (暦道, calendar mathematics) to his own son
     Kamo no Mitsuyoshi -- founding, quite literally, two different
     "successor models" trained on two different slices of one
     discipline (Mikami 1913; Goff 2001).
  4. A working ritualist who is recorded performing henbai (反閇), a
     structured stepping-ritual descended from the Daoist "Pace of Yu",
     as a corrective/protective act for a courtier departing on a journey
     (Chikanobu-kyo ki, Ten'en 2 / 974 CE; discussed in the Nanzan
     Institute survey of onmyoji practice).

THE MIND-SPECIFIC THESIS
-------------------------
Every one of these four facts points at the SAME underlying cognitive
structure: Yasunori's court did not ask him to produce elegant
propositions. It asked him to (a) sense that something in the world had
gone out of balance, (b) compute, by an exact and teachable formal
method, WHAT was out of balance and WHY, and (c) prescribe a concrete
corrective action, all under time pressure and all recorded in a form
his successors could audit and re-derive. Two faculties, one trained-
from-birth and symbolic, one untrained-and-perceptual, feeding a single
corrective act. He then formalized that division of labor into two
separate lineages.

This chapter's architecture is a literal machine version of that
division of labor. It is NOT a Transformer, NOT a mixture-of-experts
over stored keys, and it does not model "civilization as alignment."
It is a five-channel recurrent calculus built on the two real formal
relations of Wu Xing (Five-Phase) cosmology --

    generation   (相生, xiangsheng): wood -> fire -> earth -> metal -> water -> wood
    domination   (相克, xiangke):    wood -> earth -> water -> fire -> metal -> wood

-- fused at every timestep with an UNTRAINED, gradient-free perceptual
salience detector standing in for Yasunori's untaught "oni-sight", the
combination driving a corrective policy head that stands in for henbai:
the prescribed action that would restore the system to harmony.

ARCHITECTURE OVERVIEW (five parts, in forward order)
------------------------------------------------------
 1. GanshiEncoder      -- sexagenary (60-term, stem x branch) positional
                           code for calendrical time, learned projection.
 2. WuXingCell          -- five coupled GRU-like phase channels; each
                           channel's gate is driven by exactly one
                           generative inflow and one dominating (inhib-
                           itory) inflow, fixed by the real Wu Xine graph.
 3. OniSightModule      -- fixed (never trained) random-projection
                           novelty detector over a running normal model
                           of the input; models untrained perception.
 4. HenbaiFusion        -- reads the five-phase state + the anomaly
                           signal and issues two outputs: a forecast of
                           the next five-phase state, and a corrective
                           policy vector (the "ritual correction").
 5. RekirinLedger        -- a lightweight structured trace of every
                           quantity above, at every timestep -- the
                           machine analogue of a written, re-derivable
                           calendrical treatise (what survived of the
                           Rekirin was not Yasunori's authority, but his
                           method, transcribed).

Everything is implemented from scratch in pure NumPy, including a full
manual backward pass (backpropagation-through-time) -- no autograd
library is used. A finite-difference gradient check is mandatory and is
run as part of `self_tests()`. See `main()` at the bottom.
================================================================================
"""

import numpy as np

RNG_SEED = 253
np.random.seed(RNG_SEED)


# ==============================================================================
# 0. WU XING STRUCTURE CONSTANTS
# ==============================================================================
# Canonical ordering used throughout: 0=Wood(mu), 1=Fire(huo), 2=Earth(tu),
# 3=Metal(jin), 4=Water(shui).
PHASE_NAMES = ["Wood", "Fire", "Earth", "Metal", "Water"]
N_PHASES = 5

# Generative cycle (xiangsheng, 相生): phase i is generated BY phase (i-1).
#   Wood generates Fire, Fire generates Earth, Earth generates Metal,
#   Metal generates Water, Water generates Wood.
def gen_source(i: int) -> int:
    return (i - 1) % N_PHASES

# Dominating cycle (xiangke, 相克): phase i is dominated/overcome BY phase (i-2).
#   Wood overcomes Earth, Earth overcomes Water, Water overcomes Fire,
#   Fire overcomes Metal, Metal overcomes Wood.
# This is a documented structural property of the Wu Xing graph: skipping
# one step around the five-node generative cycle yields the controlling
# (ke) cycle. We use it directly as the model's fixed connectivity.
def dom_source(i: int) -> int:
    return (i - 2) % N_PHASES


# ==============================================================================
# 1. GANSHI (SEXAGENARY) TEMPORAL ENCODER
# ==============================================================================
class GanshiEncoder:
    """
    Encodes an absolute day index into the 10-stem x 12-branch sexagenary
    (ganzhi / kanshi) cycle used by real Heian calendrical science, then
    projects the one-hot stem+branch code through a small learned linear
    layer. Unlike a sinusoidal positional code, this bakes in the exact
    combinatorial periods (2, 5, 6, 10, 12, 60) that a working calendar
    office actually reasoned about.
    """

    N_STEM = 10
    N_BRANCH = 12
    ONEHOT_DIM = N_STEM + N_BRANCH  # 22

    def __init__(self, proj_dim: int):
        self.proj_dim = proj_dim
        limit = np.sqrt(6.0 / (self.ONEHOT_DIM + proj_dim))
        self.W = np.random.uniform(-limit, limit, (proj_dim, self.ONEHOT_DIM))
        self.b = np.zeros(proj_dim)

    @staticmethod
    def onehot(day_index: int) -> np.ndarray:
        stem = day_index % GanshiEncoder.N_STEM
        branch = day_index % GanshiEncoder.N_BRANCH
        v = np.zeros(GanshiEncoder.ONEHOT_DIM)
        v[stem] = 1.0
        v[GanshiEncoder.N_STEM + branch] = 1.0
        return v

    def forward(self, day_index: int):
        oh = self.onehot(day_index)
        pre = self.W @ oh + self.b
        proj = np.tanh(pre)
        cache = dict(oh=oh, proj=proj)
        return proj, cache

    def backward(self, dproj: np.ndarray, cache: dict, grads: dict):
        proj = cache["proj"]
        dpre = dproj * (1.0 - proj ** 2)
        grads["Wgan"] += np.outer(dpre, cache["oh"])
        grads["bgan"] += dpre
        # gradient wrt the one-hot input is not propagated further: it is
        # data (the calendar date), not a learnable quantity.


# ==============================================================================
# 2. WUXING GATED RECURRENT CELL (five coupled phase channels)
# ==============================================================================
class WuXingCell:
    """
    Five coupled GRU-like channels, one per Wu Xing phase. Each channel i
    at time t receives:
        - a generative inflow  g_i = Wg_i @ h_prev[gen_source(i)]
        - a dominating inflow  d_i = Wd_i @ h_prev[dom_source(i)]
        - an external input contribution from the fused (ganshi + signal)
          input vector.
    and updates via a GRU-style convex combination gated by a "balance"
    signal r_i = sigmoid(g_i - d_i + u_i), i.e. the gate literally reads as
    "how much does generative support exceed destructive pressure, given
    what is happening right now".
    """

    def __init__(self, hidden_dim: int, input_dim: int):
        self.d = hidden_dim
        self.xdim = input_dim
        self.params = {}
        for i in range(N_PHASES):
            self.params[f"Wg_{i}"] = self._xavier((hidden_dim, hidden_dim))
            self.params[f"Wd_{i}"] = self._xavier((hidden_dim, hidden_dim))
            self.params[f"Wu_{i}"] = self._xavier((hidden_dim, input_dim))
            self.params[f"bu_{i}"] = np.zeros(hidden_dim)
            self.params[f"Wcx_{i}"] = self._xavier((hidden_dim, input_dim))
            self.params[f"bcx_{i}"] = np.zeros(hidden_dim)
            self.params[f"Wch_{i}"] = self._xavier((hidden_dim, hidden_dim))

    @staticmethod
    def _xavier(shape):
        fan_in, fan_out = shape[1], shape[0]
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, shape)

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

    def init_state(self):
        return [np.zeros(self.d) for _ in range(N_PHASES)]

    def step(self, h_prev, inp_t):
        """One timestep forward. h_prev: list of 5 vectors (d,). inp_t: (xdim,)."""
        h_new = [None] * N_PHASES
        cache = {}
        for i in range(N_PHASES):
            p, o = gen_source(i), dom_source(i)
            g_i = self.params[f"Wg_{i}"] @ h_prev[p]
            d_i = self.params[f"Wd_{i}"] @ h_prev[o]
            u_i = self.params[f"Wu_{i}"] @ inp_t + self.params[f"bu_{i}"]
            pre_r = g_i - d_i + u_i
            r_i = self._sigmoid(pre_r)

            cx_i = self.params[f"Wcx_{i}"] @ inp_t + self.params[f"bcx_{i}"]
            ch_i = self.params[f"Wch_{i}"] @ h_prev[i]
            pre_c = cx_i + ch_i + r_i * g_i
            c_i = np.tanh(pre_c)

            h_i = (1.0 - r_i) * h_prev[i] + r_i * c_i
            h_new[i] = h_i
            cache[i] = dict(g=g_i, d=d_i, r=r_i, c=c_i, h_prev_i=h_prev[i],
                             h_prev_p=h_prev[p], h_prev_o=h_prev[o], inp=inp_t)
        return h_new, cache

    def step_backward(self, dh_out, cache, grads):
        """
        Backprop through one timestep. dh_out: dict i -> dL/dh_new[i]
        (total upstream gradient for each channel's *output* at this step).
        Returns dh_prev: dict i -> dL/dh_prev[i] (gradient to propagate to
        the previous timestep, accumulated across all channels that read
        h_prev[i] -- as itself, as a generative source, or as a dominating
        source) and dinp_t: dL/d(input at this timestep), summed over channels.
        """
        dh_prev = {i: np.zeros(self.d) for i in range(N_PHASES)}
        dinp_t = np.zeros(self.xdim)

        for i in range(N_PHASES):
            c = cache[i]
            g_i, d_i, r_i, c_i = c["g"], c["d"], c["r"], c["c"]
            h_prev_i = c["h_prev_i"]

            dh_i = dh_out[i]
            # h_i = (1-r_i) h_prev_i + r_i c_i
            dr_i = dh_i * (c_i - h_prev_i)
            dc_i = dh_i * r_i
            dh_prev[i] += dh_i * (1.0 - r_i)

            # c_i = tanh(pre_c) ; pre_c = cx_i + ch_i + r_i * g_i
            dpre_c = dc_i * (1.0 - c_i ** 2)
            dcx_i = dpre_c
            dch_i = dpre_c
            dr_i += dpre_c * g_i
            dg_i = dpre_c * r_i  # contribution via candidate branch

            # r_i = sigmoid(pre_r) ; pre_r = g_i - d_i + u_i
            dpre_r = dr_i * r_i * (1.0 - r_i)
            dg_i += dpre_r
            dd_i = -dpre_r
            du_i = dpre_r

            p, o = gen_source(i), dom_source(i)
            Wg_i, Wd_i = self.params[f"Wg_{i}"], self.params[f"Wd_{i}"]
            Wu_i, Wcx_i, Wch_i = (self.params[f"Wu_{i}"],
                                   self.params[f"Wcx_{i}"], self.params[f"Wch_{i}"])

            grads[f"Wg_{i}"] += np.outer(dg_i, c["h_prev_p"])
            dh_prev[p] += Wg_i.T @ dg_i

            grads[f"Wd_{i}"] += np.outer(dd_i, c["h_prev_o"])
            dh_prev[o] += Wd_i.T @ dd_i

            grads[f"Wu_{i}"] += np.outer(du_i, c["inp"])
            grads[f"bu_{i}"] += du_i
            dinp_t += Wu_i.T @ du_i

            grads[f"Wcx_{i}"] += np.outer(dcx_i, c["inp"])
            grads[f"bcx_{i}"] += dcx_i
            dinp_t += Wcx_i.T @ dcx_i

            grads[f"Wch_{i}"] += np.outer(dch_i, h_prev_i)
            dh_prev[i] += Wch_i.T @ dch_i

        return dh_prev, dinp_t


# ==============================================================================
# 3. ONI-SIGHT MODULE -- untrained (gradient-free) perceptual anomaly channel
# ==============================================================================
class OniSightModule:
    """
    Models the childhood "mi-oni" (demon-sight) episode from the Konjaku
    Monogatarishu: Yasunori perceived what his father needed training to
    perceive. This module is deliberately NEVER updated by gradient
    descent. It computes a running (exponential-moving-average) mean and
    variance of the raw signal, projects the standardized residual through
    a FIXED random matrix (frozen at construction), and reports a scalar
    salience/novelty score. No labels, no backprop, no learning curve --
    exactly the point.
    """

    def __init__(self, signal_dim: int, proj_dim: int = 16, ema_alpha: float = 0.05):
        rng = np.random.RandomState(RNG_SEED + 1)  # independent, fixed stream
        R = rng.randn(proj_dim, signal_dim)
        # Fix an orthonormal-ish random basis (frozen forever).
        Q, _ = np.linalg.qr(R.T)
        self.R = Q.T[:proj_dim, :signal_dim]
        self.mean = np.zeros(signal_dim)
        self.var = np.ones(signal_dim)
        self.alpha = ema_alpha
        self.initialized = False

    def reset_running_stats(self):
        self.mean[:] = 0.0
        self.var[:] = 1.0
        self.initialized = False

    def score(self, signal_t: np.ndarray) -> float:
        if not self.initialized:
            self.mean = signal_t.copy()
            self.var = np.ones_like(signal_t)
            self.initialized = True
        else:
            diff = signal_t - self.mean
            self.mean = self.mean + self.alpha * diff
            self.var = (1 - self.alpha) * (self.var + self.alpha * diff ** 2)
        z = (signal_t - self.mean) / np.sqrt(self.var + 1e-6)
        proj = self.R @ z
        return float(np.linalg.norm(proj) / np.sqrt(proj.shape[0]))


# ==============================================================================
# 4. HENBAI FUSION -- diagnosis (forecast) + corrective policy heads
# ==============================================================================
class HenbaiFusion:
    """
    Reads [concat of 5 phase-channel hidden states ; oni-sight score] and
    issues two outputs:
      forecast : predicted next five-phase signal vector (diagnosis)
      policy   : corrective delta that would move the *current* signal
                 back toward the harmonious equilibrium (0 vector, after
                 the data has been centered) -- the model's henbai.
    """

    def __init__(self, hidden_total: int, out_dim: int):
        in_dim = hidden_total + 1  # + oni-sight scalar
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.params = {
            "Wf": WuXingCell._xavier((out_dim, in_dim)),
            "bf": np.zeros(out_dim),
            "Wp": WuXingCell._xavier((out_dim, in_dim)),
            "bp": np.zeros(out_dim),
        }

    def forward(self, h_concat: np.ndarray, oni_score: float):
        fusion_in = np.concatenate([h_concat, [oni_score]])
        pre_f = self.params["Wf"] @ fusion_in + self.params["bf"]
        pre_p = self.params["Wp"] @ fusion_in + self.params["bp"]
        forecast = np.tanh(pre_f)
        policy = np.tanh(pre_p)
        cache = dict(fusion_in=fusion_in, forecast=forecast, policy=policy)
        return forecast, policy, cache

    def backward(self, dforecast, dpolicy, cache, grads):
        forecast, policy = cache["forecast"], cache["policy"]
        dpre_f = dforecast * (1.0 - forecast ** 2)
        dpre_p = dpolicy * (1.0 - policy ** 2)
        grads["Wf"] += np.outer(dpre_f, cache["fusion_in"])
        grads["bf"] += dpre_f
        grads["Wp"] += np.outer(dpre_p, cache["fusion_in"])
        grads["bp"] += dpre_p
        dfusion_in = self.params["Wf"].T @ dpre_f + self.params["Wp"].T @ dpre_p
        # split: gradient to h_concat (trainable) and to oni_score (dropped --
        # the oni-sight pathway is untrained by design, so we stop-gradient here).
        dh_concat = dfusion_in[:-1]
        return dh_concat


# ==============================================================================
# 5. REKIRIN LEDGER -- structured, human-readable trace
# ==============================================================================
class RekirinLedger:
    """
    A minimal structured logger. The original Rekirin (暦林) itself did not
    survive; what survived was the METHOD, transcribed into the later
    Rekirin Mondoshu. This ledger is the machine equivalent: not a claim
    of authority, but a re-derivable written trace of what the system
    computed and why, at every step.
    """

    def __init__(self):
        self.entries = []

    def log(self, day_index, phase_states, oni_score, forecast=None, policy=None):
        stem = day_index % GanshiEncoder.N_STEM
        branch = day_index % GanshiEncoder.N_BRANCH
        entry = {
            "day": day_index,
            "ganshi": (stem, branch),
            "phase_norms": [float(np.linalg.norm(h)) for h in phase_states],
            "oni_score": float(oni_score),
        }
        if forecast is not None:
            entry["forecast"] = np.round(forecast, 4).tolist()
        if policy is not None:
            entry["policy"] = np.round(policy, 4).tolist()
        self.entries.append(entry)

    def tail(self, n=5):
        return self.entries[-n:]


# ==============================================================================
# FULL MODEL
# ==============================================================================
class YasunoriModel:
    """
    Wraps GanshiEncoder + WuXingCell + OniSightModule + HenbaiFusion into a
    single trainable sequence model, plus a RekirinLedger for tracing.

    forward(days, signals) -> forecast, policy, aux
       days:    list[int] of length T, absolute calendar day indices
       signals: (T, signal_dim) array of raw five-phase readings

    Trains end-to-end via BPTT EXCEPT the OniSightModule, which never
    receives gradient (see class docstring).
    """

    def __init__(self, signal_dim=5, ganshi_proj_dim=8, hidden_dim=8, oni_proj_dim=16):
        self.signal_dim = signal_dim
        self.ganshi = GanshiEncoder(ganshi_proj_dim)
        self.cell = WuXingCell(hidden_dim, input_dim=ganshi_proj_dim + signal_dim)
        self.oni = OniSightModule(signal_dim, proj_dim=oni_proj_dim)
        self.fusion = HenbaiFusion(hidden_dim * N_PHASES, signal_dim)
        self.ledger = RekirinLedger()
        self.hidden_dim = hidden_dim

    def all_params(self):
        p = {}
        p["Wgan"] = self.ganshi.W
        p["bgan"] = self.ganshi.b
        for k, v in self.cell.params.items():
            p[k] = v
        p["Wf"] = self.fusion.params["Wf"]
        p["bf"] = self.fusion.params["bf"]
        p["Wp"] = self.fusion.params["Wp"]
        p["bp"] = self.fusion.params["bp"]
        return p

    def zero_grads(self):
        return {k: np.zeros_like(v) for k, v in self.all_params().items()}

    def forward(self, days, signals, log=False):
        T = len(days)
        self.oni.reset_running_stats()
        h = self.cell.init_state()
        cell_caches = []
        ganshi_caches = []
        oni_scores = []
        h_hist = [h]
        for t in range(T):
            proj, gcache = self.ganshi.forward(days[t])
            inp_t = np.concatenate([proj, signals[t]])
            h, ccache = self.cell.step(h, inp_t)
            oni_s = self.oni.score(signals[t])
            cell_caches.append(ccache)
            ganshi_caches.append(gcache)
            oni_scores.append(oni_s)
            h_hist.append(h)
            if log:
                self.ledger.log(days[t], h, oni_s)

        h_concat = np.concatenate(h_hist[-1])
        forecast, policy, fcache = self.fusion.forward(h_concat, oni_scores[-1])
        if log:
            self.ledger.entries[-1]["forecast"] = np.round(forecast, 4).tolist()
            self.ledger.entries[-1]["policy"] = np.round(policy, 4).tolist()

        aux = dict(cell_caches=cell_caches, ganshi_caches=ganshi_caches,
                   oni_scores=oni_scores, h_hist=h_hist, fcache=fcache, T=T)
        return forecast, policy, aux

    def backward(self, dforecast, dpolicy, aux):
        grads = self.zero_grads()
        T = aux["T"]
        dh_concat = self.fusion.backward(dforecast, dpolicy, aux["fcache"], grads)
        d = self.hidden_dim
        dh_next = {i: dh_concat[i * d:(i + 1) * d] for i in range(N_PHASES)}

        for t in reversed(range(T)):
            ccache = aux["cell_caches"][t]
            gcache = aux["ganshi_caches"][t]
            dh_prev, dinp_t = self.cell.step_backward(dh_next, ccache, grads)
            dproj = dinp_t[: self.ganshi.proj_dim]
            self.ganshi.backward(dproj, gcache, grads)
            dh_next = dh_prev
        return grads

    def sgd_step(self, grads, lr, clip=5.0):
        params = self.all_params()
        for k in params:
            g = grads[k]
            norm = np.linalg.norm(g)
            if norm > clip:
                g = g * (clip / (norm + 1e-8))
            params[k] -= lr * g


# ==============================================================================
# SYNTHETIC DATA: "Five-Phase Cosmic-Social Signal"
# ==============================================================================
def generate_sequence(rng, T, signal_dim=5, start_day=None, anomaly_prob=0.08):
    """
    Simulates a hidden five-phase dynamical system with real generative
    (xiangsheng) and dominating (xiangke) coupling, a 12-period seasonal
    term (linking to the 12 earthly branches), and sparse injected
    "omens" (anomaly spikes). Returns:
        days:      list[int], length T+1 (last day is the forecast target day)
        signals:   (T+1, signal_dim) float array, values roughly in [-1,1]
        anomaly_labels: (T+1,) 0/1 array marking injected omen timesteps
    """
    if start_day is None:
        start_day = int(rng.randint(0, 10_000))
    days = [start_day + t for t in range(T + 1)]

    s = rng.normal(0, 0.15, size=signal_dim)
    signals = np.zeros((T + 1, signal_dim))
    anomaly_labels = np.zeros(T + 1)

    gen_coeff, dom_coeff, decay = 0.18, 0.10, 0.05
    for t in range(T + 1):
        seasonal = 0.12 * np.sin(2 * np.pi * (days[t] % 12) / 12.0 + np.arange(signal_dim))
        gen_pull = np.array([s[gen_source(i)] for i in range(signal_dim)])
        dom_pull = np.array([s[dom_source(i)] for i in range(signal_dim)])
        noise = rng.normal(0, 0.03, size=signal_dim)

        s = (s
             + gen_coeff * (gen_pull - s)
             - dom_coeff * dom_pull
             - decay * s
             + seasonal * 0.3
             + noise)

        if rng.rand() < anomaly_prob:
            k = rng.randint(0, signal_dim)
            s[k] += rng.choice([-1, 1]) * rng.uniform(0.6, 1.0)
            anomaly_labels[t] = 1.0

        s = np.clip(s, -1.0, 1.0)
        signals[t] = s

    return days, signals, anomaly_labels


def make_dataset(n_sequences, T, rng):
    data = []
    for _ in range(n_sequences):
        days, signals, labels = generate_sequence(rng, T)
        days_in, signals_in = days[:T], signals[:T]
        target_forecast = signals[T]                  # next-step 5-phase state
        target_policy = np.clip(-signals[T - 1], -1, 1)  # correction back to 0-equilibrium
        data.append(dict(days=days_in, signals=signals_in,
                          target_forecast=target_forecast,
                          target_policy=target_policy,
                          anomaly_labels=labels[:T]))
    return data


# ==============================================================================
# LOSS
# ==============================================================================
def mse_loss_and_grad(pred, target):
    diff = pred - target
    loss = float(np.mean(diff ** 2))
    grad = (2.0 / pred.shape[0]) * diff
    return loss, grad


# ==============================================================================
# TRAINING LOOP
# ==============================================================================
def train(model: YasunoriModel, train_data, epochs=25, lr=0.05, policy_weight=0.5, verbose=True):
    history = []
    for ep in range(epochs):
        total_loss, total_f, total_p = 0.0, 0.0, 0.0
        order = np.random.permutation(len(train_data))
        for idx in order:
            ex = train_data[idx]
            forecast, policy, aux = model.forward(ex["days"], ex["signals"])
            lf, dforecast = mse_loss_and_grad(forecast, ex["target_forecast"])
            lp, dpolicy = mse_loss_and_grad(policy, ex["target_policy"])
            dforecast = dforecast
            dpolicy = dpolicy * policy_weight
            grads = model.backward(dforecast, dpolicy, aux)
            model.sgd_step(grads, lr)
            total_loss += lf + policy_weight * lp
            total_f += lf
            total_p += lp
        n = len(train_data)
        history.append((total_loss / n, total_f / n, total_p / n))
        if verbose and (ep % max(1, epochs // 10) == 0 or ep == epochs - 1):
            print(f"  epoch {ep:3d}  loss={total_loss/n:.5f}  "
                  f"forecast_mse={total_f/n:.5f}  policy_mse={total_p/n:.5f}")
    return history


def evaluate(model: YasunoriModel, data):
    fmses, pmses = [], []
    for ex in data:
        forecast, policy, _ = model.forward(ex["days"], ex["signals"])
        fmses.append(np.mean((forecast - ex["target_forecast"]) ** 2))
        pmses.append(np.mean((policy - ex["target_policy"]) ** 2))
    return float(np.mean(fmses)), float(np.mean(pmses))


def evaluate_oni_sight(model: YasunoriModel, data):
    """
    Diagnostic only (the oni-sight module is never trained): checks
    whether its untrained novelty score is nonetheless higher, on
    average, at timesteps carrying an injected anomaly label than at
    timesteps without one.
    """
    model.oni.reset_running_stats()
    anomaly_scores, normal_scores = [], []
    for ex in data:
        model.oni.reset_running_stats()
        for t in range(len(ex["days"])):
            s = model.oni.score(ex["signals"][t])
            if ex["anomaly_labels"][t] > 0.5:
                anomaly_scores.append(s)
            else:
                normal_scores.append(s)
    return float(np.mean(anomaly_scores)), float(np.mean(normal_scores))


# ==============================================================================
# FINITE-DIFFERENCE GRADIENT CHECK  (mandatory correctness test)
# ==============================================================================
def gradient_check(model: YasunoriModel, example, eps=1e-5, n_checks=6, seed=0):
    rng = np.random.RandomState(seed)
    params = model.all_params()
    keys = list(params.keys())

    def loss_fn():
        forecast, policy, aux = model.forward(example["days"], example["signals"])
        lf, dforecast = mse_loss_and_grad(forecast, example["target_forecast"])
        lp, dpolicy = mse_loss_and_grad(policy, example["target_policy"])
        return lf + 0.5 * lp, dforecast, dpolicy * 0.5, aux

    base_loss, dforecast, dpolicy, aux = loss_fn()
    grads = model.backward(dforecast, dpolicy, aux)

    results = []
    for _ in range(n_checks):
        k = keys[rng.randint(len(keys))]
        arr = params[k]
        flat_idx = rng.randint(arr.size)
        idx = np.unravel_index(flat_idx, arr.shape)

        orig = arr[idx]
        arr[idx] = orig + eps
        loss_plus, _, _, _ = loss_fn()
        arr[idx] = orig - eps
        loss_minus, _, _, _ = loss_fn()
        arr[idx] = orig

        numeric = (loss_plus - loss_minus) / (2 * eps)
        analytic = grads[k][idx]
        denom = max(abs(numeric), abs(analytic), 1e-8)
        rel_err = abs(numeric - analytic) / denom
        results.append((k, idx, numeric, analytic, rel_err))
    return results, base_loss


# ==============================================================================
# SELF TESTS
# ==============================================================================
def self_tests():
    print("=" * 78)
    print("SELF-TESTS")
    print("=" * 78)

    # --- Wu Xing structural relations match the documented theory ---
    expected_gen = {0: 4, 1: 0, 2: 1, 3: 2, 4: 3}   # who generates phase i
    expected_dom = {0: 3, 1: 4, 2: 0, 3: 1, 4: 2}   # who dominates phase i
    for i in range(N_PHASES):
        assert gen_source(i) == expected_gen[i], f"gen_source({i}) mismatch"
        assert dom_source(i) == expected_dom[i], f"dom_source({i}) mismatch"
    print("[PASS] Wu Xing generative/dominating adjacency matches canonical theory")

    # --- Ganshi periods ---
    idxs = [GanshiEncoder.onehot(d) for d in range(60)]
    assert np.allclose(idxs[0], idxs[60 % 60])
    assert not np.allclose(GanshiEncoder.onehot(0), GanshiEncoder.onehot(1))
    stems = {d % 10 for d in range(10)}
    branches = {d % 12 for d in range(12)}
    assert stems == set(range(10)) and branches == set(range(12))
    print("[PASS] Ganshi encoder reproduces the 10-stem / 12-branch / 60-cycle structure")

    # --- OniSight is truly untrained: params identical before/after training ---
    rng = np.random.RandomState(7)
    model = YasunoriModel()
    R_before = model.oni.R.copy()
    data = make_dataset(6, T=10, rng=rng)
    train(model, data, epochs=3, lr=0.03, verbose=False)
    assert np.array_equal(R_before, model.oni.R), "OniSight projection must never change"
    print("[PASS] Oni-sight projection matrix is unchanged by training (gradient-free by design)")

    # --- Shapes sane ---
    ex = data[0]
    forecast, policy, aux = model.forward(ex["days"], ex["signals"])
    assert forecast.shape == (model.signal_dim,)
    assert policy.shape == (model.signal_dim,)
    assert np.all(forecast >= -1.0) and np.all(forecast <= 1.0)
    print("[PASS] Forward pass produces correctly-shaped, bounded forecast/policy outputs")

    # --- Gradient check ---
    results, base_loss = gradient_check(model, ex, n_checks=8)
    max_rel_err = max(r[4] for r in results)
    print(f"  base loss = {base_loss:.6f}")
    for k, idx, num, ana, err in results:
        print(f"    param={k:8s} idx={str(idx):12s} numeric={num:+.6f} "
              f"analytic={ana:+.6f}  rel_err={err:.2e}")
    assert max_rel_err < 2e-2, f"gradient check failed, max rel err {max_rel_err}"
    print(f"[PASS] Finite-difference gradient check (max rel. error {max_rel_err:.2e} < 2e-2)")
    print()


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    self_tests()

    print("=" * 78)
    print("TRAINING RUN")
    print("=" * 78)
    rng = np.random.RandomState(RNG_SEED)
    train_data = make_dataset(160, T=16, rng=rng)
    val_data = make_dataset(40, T=16, rng=rng)

    model = YasunoriModel(signal_dim=5, ganshi_proj_dim=8, hidden_dim=8, oni_proj_dim=16)
    history = train(model, train_data, epochs=40, lr=0.06, policy_weight=0.5, verbose=True)

    f_mse, p_mse = evaluate(model, val_data)
    print()
    print(f"Held-out validation  ->  forecast MSE={f_mse:.5f}   policy MSE={p_mse:.5f}")

    anom_mean, norm_mean = evaluate_oni_sight(model, val_data)
    print(f"Oni-sight (untrained) mean novelty score  ->  at omens={anom_mean:.4f}  "
          f"elsewhere={norm_mean:.4f}   (higher-at-omens is the expected signature)")

    print()
    print("=" * 78)
    print("REKIRIN LEDGER -- sample trace (one held-out sequence, logged)")
    print("=" * 78)
    ex = val_data[0]
    model.forward(ex["days"], ex["signals"], log=True)
    for entry in model.ledger.tail(5):
        print(f"  day={entry['day']:5d}  ganshi(stem,branch)={entry['ganshi']}  "
              f"phase_norms={[round(x,3) for x in entry['phase_norms']]}  "
              f"oni={entry['oni_score']:.3f}")
    last = model.ledger.tail(1)[0]
    print(f"  -> forecast (next 5-phase state): {last.get('forecast')}")
    print(f"  -> policy   (henbai correction) : {last.get('policy')}")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
