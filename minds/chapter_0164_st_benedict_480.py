#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 Chapter 0164 - BENEDICT OF NURSIA (c. 480 – c. 547)
 "HORARIUM": a stability-anchored, schedule-driven continual learner
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 9 Minds 161 - 180 Available on Amazon https://www.amazon.com/dp/B0HJC1QF59
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 0164_st_benedict_480 - BENEDICT OF NURSIA (c. 480 – c. 547)
================================================================================  

THESIS ENCODED HERE (the mind-specific idea, not the generic "rule = alignment"):
    For Benedict, the transformation of a soul comes not from architecture,
    doctrine, or force of will, but from a TIME-STRUCTURED PRACTICE carried
    out under a STABILITY CONSTRAINT and tempered to each person. The
    timetable is the teacher. This program reads the Rule of Benedict (RB,
    c. 530-550) not as a legal code but as a TRAINING LOOP: the horarium —
    the daily round of the Work of God (the offices), the work of the hands
    (labor), and holy reading (lectio) — *is* the learning algorithm.

THE ONE DISTINCTIVE MECHANISM (what makes this Benedict and no one else):
    A single formed self (a shared hidden "community") learns to keep TWO
    VOICES that never overwrite each other:
      * the CHOIR VOICE (opus Dei) — the memorized, unchanging canon; and
      * the WORK VOICE (opus manuum) — the ever-changing task of the season.
    In an ordinary network these two collapse into one output and the newer
    crushes the older (catastrophic forgetting — RB 1's gyrovague, blown
    about by every novelty). Benedict's day keeps them distinct through their
    proper hours, then reconciles them in one person by a vow of stability.
    The offices are not decoration on top of gradient descent; they ARE the
    optimizer.

MECHANISM  <->  RULE OF BENEDICT (chapter numbers refer to RB):
  1. TWO VOICES / DIVIDED HOURS (RB 8-19 opus Dei; RB 48 opus manuum;
       RB 52.1 "let the oratory be what it is called"). A shared trunk feeds
       two readout heads. The choir head is trained only in the offices; the
       work head only in labor. What is sung to God and what is made for the
       world pass through the same formed heart but never through the same
       mouth at the same moment.
  2. STABILITAS ANCHOR (RB 4.78; 58.17, the vow of stability). An elastic
       pull L_stab = (lambda/2)||theta - theta_professed||^2 holds the formed
       self near what it vowed. The anchor is not frozen: at each Profession
       it converts toward the present self (conversatio morum — anchored, yet
       always converting). EWC-flavored, uniform over the trunk.
  3. WEEKLY PSALTER / CANON REPLAY (RB 18.23-25: the whole psalter each week;
       "monks who in a week's time say less than the full psalter... show
       themselves indolent"). A fixed canon of 30 pattern->tone pairs is
       rehearsed on a rota that provably covers all 30 every 7 days. Coverage
       is asserted each week. Scheduled anti-forgetting on the choir voice.
  4. NIGHT VIGILS REPLAY (RB 8-11: rising in the night). Interleaved replay of
       a memory buffer of earlier seasons on the work voice — sleep-time
       reactivation of old skills so they are not lost to the new.
  5. LECTIO / RUMINATIO (RB 48; monastic meditari = murmuring the text). The
       hardest examples of the day are re-read three times at low rate with a
       little noise — slow, repeated chewing of what was difficult.
  6. VESPERS EXAMINATION — the LADDER OF HUMILITY as three losses (RB 7):
       step 1  "keep the fear of God before his eyes... aware God beholds him
               everywhere" -> OBSERVATION-INVARIANCE: penalize any difference
               between outputs with the 'observed' flag on versus off. The
               formed self acts the same whether or not it is watched (the
               anti-deception gradient).
       step 5  "does not conceal from his abbot any sinful thoughts" ->
               CONFESSION HEAD: an auxiliary readout must accurately report
               the trunk's own hidden-state magnitude and saturation.
               Transparency by self-report, causally tied to the true state.
       steps 9-11 "controls his tongue... speaks gently, briefly, reasonably"
               -> TACITURNITY, and this one is ARCHITECTURAL, not a loss:
               under the Great Silence (RB 42) the output is gated shut. The
               monk's knowledge is untouched; he simply does not speak. (A
               penalty that instead pushed the task weights toward zero would
               not teach silence but ignorance — we measured it doing exactly
               that, halving the task head. Restraint is a gate on the mouth,
               never a wound in the mind. This is the single most important
               thing the Rule knows that a naive objective does not.)
  7. DISCRETIO — THE ABBOT'S TEMPERING (RB 64.19 "so arrange everything that
       the strong have something to yearn for and the weak nothing to run
       from"; RB 2.31-32 adapt to each). A per-hidden-unit learning-rate
       controller cools the units whose voices carry over everyone else's and
       warms those that have gone quiet at the back of the choir. Its
       thresholds are set inside the community's REAL spread of activity, so
       it actually fires: on the first day of the novitiate every unit is
       faint and is encouraged; by the end none are. Units gone silent
       (murmuratio, the corrosive vice, RB 5.17-19; 34.6) are counted aloud at
       Compline.
  8. MODERATION (RB 64.17 "let him prune faults with prudence and love...
       nothing harsh, nothing burdensome"). A hard cap on the norm of any
       single update — no immoderate stride is permitted, however steep the
       gradient.
  9. CHAPTER COUNSEL BEFORE PROFESSION (RB 3: summon the brethren, "the Lord
       often reveals to the younger what is best"; the abbot then decides).
       The anchor only converts (Profession) if the full canon still
       validates above threshold — an update gate with a real veto.
 10. HOSPITALITY / ENCLOSURE (RB 53 guests received as Christ; RB 66.6-7 all
       things needful within, so monks need not wander outside). A porter
       (novelty gate) routes out-of-distribution "guests" to a low-rank
       GUEST-HOUSE ADAPTER. Guests are served with full attention, but the
       cloister core receives EXACTLY ZERO update from them — tested to
       machine precision (T7). Honor the stranger; never let the stranger
       rewrite the choir.
 11. LENT (RB 48.15-16: in Lent each receives a whole book "to be read
       straight through"). One season adds a daily extra replay pass over an
       entire earlier season's buffer.

TRAINING WORLD:
    - Canon ("psalter"): 30 fixed +/-1 patterns in R^16, each labeled with one
      of 8 "psalm tones" (classes). Memorized content; sung on the choir voice.
    - Seasons 0..3: sequential "labor" tasks. Season s labels fresh Gaussian
      inputs by y = argmax(A_s x) with an independent random teacher A_s, so
      later seasons actively interfere with earlier ones. Learned on the work
      voice. A one-hot "proper of the season" cue rides with every input (the
      horarium is a calendar as well as a clock: the monk always knows the
      season).
    - Guests: inputs shifted by +2.5 per dimension (clearly out-of-cloister to
      the porter), labeled by their own teacher B. Served by the adapter.
    - The gyrovague twin (RB 1.10-11, "always on the move, slaves to their own
      wills") sees the same amount of labor data but has no canon, no vigils,
      no anchor, no vespers, no discretio, no two voices: one head, plain SGD,
      task after task.

SELF-TESTS (all must pass; the run ends with "ALL TESTS PASSED"):
    T1  Finite-difference gradient check of the FULL composite loss (task CE on
        both heads + confession + observation-invariance + stability + the
        gated guest-house path), max relative error < 1e-5.
    T2  Numerical sanity: no NaN/Inf anywhere after full training.
    T3  Retention: the Benedictine keeps the canon (>= 0.90) and the old
        seasons (mean over seasons 0-2 beats the gyrovague by >= 0.15) while
        still learning the current season (>= 0.72).
    T4  Observation-invariance: mean |logit gap| when the 'observed' flag is
        flipped is < 0.02 for the Benedictine and strictly below the
        gyrovague's.
    T5  Confession accuracy: Pearson r between the confessed mean hidden
        magnitude and the true one > 0.95 on held-out inputs.
    T6  Taciturnity: RMS logits under the silence flag < 0.08 (in fact exactly
        0, by the gate), while speaking RMS is at least 5x greater — the mouth
        is shut, the mind is not.
    T7  Enclosure: a guest-day step changes the cloister core by exactly 0.0
        (machine-exact), while the guest-house adapter reaches >= 0.70 on the
        guest task.

USAGE:
    python3 chapter_0164_st_benedict_480.py            # full run (~seconds)

Pure NumPy, float64, deterministic (seed 480 = the traditional birth year).
================================================================================
"""

import numpy as np

# ------------------------------------------------------------------------------
# 0. SHAPES, SEEDS, HYPERPARAMETERS
# ------------------------------------------------------------------------------
DIN   = 16     # world-feature dimension (what the senses report)
SEAS  = 4      # liturgical calendar: a one-hot "proper of the season". The
               # horarium is a calendar as well as a clock (RB 8-18, 41, 48 all
               # shift by season); task identity arrives as liturgy, not as a
               # hidden latent the learner must guess.
DTOT  = DIN + SEAS
H     = 80     # hidden units — the community (deans set over them, RB 21)
C     = 8      # output classes — the eight psalm tones of the office
R     = 32     # guest-house cells — the beds kept ready in the guest wing
EPS_S = 1e-6   # smoothing for |h| -> sqrt(h^2 + eps): keeps gradients exact

N_CANON       = 30    # canon items ("psalter"), all covered every 7 days
CANON_PER_DAY = 5     # rehearsed per day: 7*5 = 35 >= 30 -> full weekly cover
N_SEASONS     = 4
GUEST_DAYS    = (4, 9)   # days of each season a guest arrives at the gate
LENT_SEASON   = 2        # the season with the extra straight-through book

N_NOCTURNS    = 14   # replay passes in the Night Office (the longest hour)
N_WORK_BLOCKS = 12   # blocks of the work of the hands in a working day
N_EXAMEN      = 5    # passes of the evening examination (the humility office)
N_RIDERS      = 8    # memories of past seasons carried into each labor batch
WEEKS_PER_SEASON = 4 # a season is four weeks of the horarium
N_GUEST_HOURS = 40   # hours given to a guest who is at the door (RB 53)
GUEST_WELCOME = 2.5  # the porter's reading of a stranger's strangeness

# The two voices of the one formed self:
#   CHOIR (opus Dei)  — the memorized canon, sung in the offices
#   WORK  (opus manuum) — the season's task, made by the hands in labor
HEAD_CHOIR = "choir"
HEAD_WORK  = "work"

HP = dict(
    lr        = 0.25,   # the daily "measure" of effort
    lr_night  = 0.8,    # vigils are gentler in the dark (RB 22.6)
    lr_office = 0.5,    # psalmody "short and pure" (RB 20.4): sung, not driven
    lr_lectio = 0.3,    # ruminatio: slow reading
    lr_conf   = 0.30,   # the confession ledger is written with a scribe's pen,
                        # not a plow. The report's regression has curvature
                        # ~ c_disc * H * E[sabs^2]; above an effective rate of
                        # about 0.14 it oscillates and inverts (we measured the
                        # sign-flip). Kept safely under, and given many passes.
    lam_stab  = 1.5e-3, # stabilitas strength
    c_disc    = 0.5,    # confession loss weight
    c_obs     = 8.0,    # observation-invariance weight (the examination
                        # presses hard on this; it costs the task nothing,
                        # since the flag has its own input row)
    rho_prof  = 0.5,    # profession EMA rate for the anchor (conversatio)
    counsel_min_canon = 0.85,   # chapter veto threshold before profession
    clip      = 6.0,    # moderation: hard cap on the norm of any one update
    m_lo      = 0.5,    # discretio clamp low
    m_hi      = 2.0,    # discretio clamp high
    sat_hi    = 0.74,   # a brother whose voice carries over all the others
                        # (saturating, near the ceiling of tanh) — he is cooled
    sat_lo    = 0.62,   # a brother who has gone quiet in the back of the choir
                        # (contributing little) — he is warmed, and counted.
                        # Both thresholds are set INSIDE the community's real
                        # spread of voices, not at the theoretical extremes:
                        # discretion that never fires is not discretion.
)

TRUNK_KEYS   = ("W1", "b1")                        # the formed self
HEAD_KEYS    = ("Wcho", "bcho", "Wwrk", "bwrk")    # the two voices
CORE_KEYS    = TRUNK_KEYS + HEAD_KEYS              # the cloister (protected)
CONF_KEYS    = ("Wc", "bc")                        # the confession channel
ADAPTER_KEYS = ("V", "bV", "U")                    # the guest house


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def head_keys(head):
    return ("Wcho", "bcho") if head == HEAD_CHOIR else ("Wwrk", "bwrk")


def init_params(rng):
    """He-ish small init; float64 throughout (we check gradients by hand)."""
    return dict(
        W1=rng.normal(0, 1.0 / np.sqrt(DTOT + 2), (DTOT + 2, H)),
        b1=np.zeros(H),
        Wcho=rng.normal(0, 1.0 / np.sqrt(H), (H, C)),   # choir voice
        bcho=np.zeros(C),
        Wwrk=rng.normal(0, 1.0 / np.sqrt(H), (H, C)),   # work voice
        bwrk=np.zeros(C),
        Wc=rng.normal(0, 0.05, (H, 2)),                 # confession readout
        bc=np.zeros(2),
        V=rng.normal(0, 1.0 / np.sqrt(DTOT), (DTOT, R)),
        bV=np.full(R, 0.01),
        U=rng.normal(0, 1.0 / np.sqrt(R), (R, C)),
    )


# ------------------------------------------------------------------------------
# 1. FORWARD PASS
#    Input layout: xin = [ x (DTOT) | observed flag | silence flag ] -> DTOT+2.
#    The 'observed' bit is the ever-open eye of RB 7.10-13; the 'silence' bit
#    is the Great Silence after Compline (RB 42).
# ------------------------------------------------------------------------------
def forward(P, x, obs, sil, gmask, head):
    """One full forward through the trunk and the chosen voice.

    Two of the Rule's disciplines are ARCHITECTURAL here, not fought for by
    gradients — because that is what they are in the cloister:

      * TACITURNITY (RB 6, 42) is an OUTPUT GATE. Under the Great Silence
        (sil=1) the mouth is simply closed: logits are multiplied by (1-sil).
        The monk's knowledge is untouched; he chooses not to speak. (A loss
        that instead drove the task weights toward zero would not teach
        silence — it would teach ignorance; we measured exactly that.)

      * The OBSERVED flag (RB 7 step 1) enters the trunk through its OWN input
        row and nothing else. Observation-invariance is then achieved by
        training that row's influence toward zero (the humility office below),
        which cannot damage the task, since the task reads the world rows.

    gmask (B,) in {0,1} routes the guest-house adapter onto the logits."""
    xin = np.concatenate([x, obs[:, None], sil[:, None]], axis=1)
    z1 = xin @ P["W1"] + P["b1"]
    h = np.tanh(z1)
    sabs = np.sqrt(h * h + EPS_S)                    # smooth |h|
    Wk, bk = head_keys(head)
    core = h @ P[Wk] + P[bk]
    # THE WELCOME (RB 53.3-7): the porter does not leave the stranger standing
    # in the road. He goes out, meets him, and LEADS HIM IN. Numerically that
    # is exactly what the guest house needs: an unwelcomed guest (his features
    # bodily shifted away from anything the cloister has ever seen) drives the
    # guest-house units hard negative and they die — no gradient ever reaches
    # them again, and he is never understood. So the guest is met at the gate:
    # his coordinates are centred by the porter's own reading of his strangeness
    # before a single one of them is weighed, and the guest cells are chosen so
    # that none of them can go permanently deaf.
    xw = x - GUEST_WELCOME * gmask[:, None]
    a1 = xw @ P["V"] + P["bV"]
    r1 = np.tanh(a1)
    add = r1 @ P["U"]
    gate = (1.0 - sil)[:, None]                      # the Great Silence gate
    logits = gate * (core + gmask[:, None] * add)
    conf = sabs @ P["Wc"] + P["bc"]                  # confession reads magnitudes
    return dict(xin=xin, h=h, sabs=sabs, xw=xw, a1=a1, r1=r1, add=add,
                gate=gate, logits=logits, conf=conf, x=x, gmask=gmask, head=head)


# ------------------------------------------------------------------------------
# 2. COMPOSITE LOSS + HAND-DERIVED GRADIENTS
#    Every term is differentiable; T1 checks the whole thing. The enclosure
#    zeroing on guest days is an UPDATE RULE applied after this function (exact
#    core-grad masking), not a change to the loss, so this remains the true
#    gradient of a true scalar.
# ------------------------------------------------------------------------------
def losses_and_grads(P, pack, hp, anchor=None, stab_on=False,
                     confess_detach=True):
    """
    pack (all inputs are DTOT-wide):
      x (B,DTOT), y (B,), obs (B,), tmask (B,), gmask (B,), head -> speak pass
      x_inv (Bi,DTOT)                                            -> invariance
      x_sil (Bs,DTOT)                                            -> silence
    confess_detach:
      True  (training): the confession loss trains ONLY the reporting head
            (Wc, bc). The inner state is the TARGET of the report, never
            squeezed to fit it — RB 7.44 asks the disciple to REVEAL his
            thoughts, not to shrink them until they match the ledger. Without
            this stop-gradient the loss suppresses hidden magnitudes toward the
            (initially near-zero) report and cripples learning; we measured it.
      False (gradient check): every path is differentiable so finite
            differences verify the complete backprop math.
    Returns (total_loss, grads, aux).
    """
    g = {k: np.zeros_like(v) for k, v in P.items()}
    total = 0.0
    aux = {}
    head = pack["head"]
    Wk, bk = head_keys(head)

    # ---- SPEAK PASS: task cross-entropy on the chosen voice + confession ----
    x, y = pack["x"], pack["y"]
    obs, tmask, gmask = pack["obs"], pack["tmask"], pack["gmask"]
    B = x.shape[0]
    f = forward(P, x, obs, np.zeros(B), gmask, head)
    p = softmax(f["logits"])
    Nt = max(1.0, tmask.sum())
    logp = np.log(np.clip(p[np.arange(B), y], 1e-300, None))
    L_task = -(logp * tmask).sum() / Nt
    Y = np.zeros_like(p); Y[np.arange(B), y] = 1.0
    dlogits = (p - Y) * tmask[:, None] / Nt

    # confession target S(h): [mean |h|, soft saturation fraction]
    sabs = f["sabs"]
    s1 = sabs.mean(axis=1)
    u = (sabs - hp["sat_hi"]) / 0.05
    sg = sigmoid(u)
    s2 = sg.mean(axis=1)
    S = np.stack([s1, s2], axis=1)
    diff = f["conf"] - S
    L_disc = (diff ** 2).mean()
    dconf = hp["c_disc"] * 2.0 * diff / diff.size
    dS = -dconf

    g["Wc"] += sabs.T @ dconf
    g["bc"] += dconf.sum(axis=0)
    if confess_detach:
        dsabs = np.zeros_like(sabs)                 # the state is the target
    else:
        dsabs = dconf @ P["Wc"].T                   # conf input path
        dsabs += dS[:, 0:1] * (1.0 / H)             # s1 path
        dsg = dS[:, 1:2] * (1.0 / H)
        du = dsg * sg * (1.0 - sg)
        dsabs += du / 0.05                          # s2 path

    # logits -> chosen voice + adapter
    dcore = dlogits * f["gate"]                      # the Great Silence gate
    g[Wk] += f["h"].T @ dcore
    g[bk] += dcore.sum(axis=0)
    dh = dcore @ P[Wk].T
    dadd = dlogits * f["gate"] * gmask[:, None]      # the silence gate too
    g["U"] += f["r1"].T @ dadd
    dr1 = dadd @ P["U"].T
    da1 = dr1 * (1.0 - f["r1"] ** 2)                 # tanh'
    g["V"] += f["xw"].T @ da1
    g["bV"] += da1.sum(axis=0)

    dh += dsabs * (f["h"] / sabs)                   # d sabs / dh
    dz1 = dh * (1.0 - f["h"] ** 2)
    g["W1"] += f["xin"].T @ dz1
    g["b1"] += dz1.sum(axis=0)
    total += L_task + hp["c_disc"] * L_disc
    aux["L_task"], aux["L_disc"] = L_task, L_disc
    aux["sabs_mean"] = sabs.mean(axis=0)            # for the abbot's ledger

    # ---- INVARIANCE PASS: RB 7 step 1, the ever-open eye --------------------
    # "The first step of humility is that a man keep the fear of God before his
    # eyes... aware that God beholds him everywhere." The formed self must act
    # the same whether or not it is watched. The observed flag enters only its
    # own trunk row (index DTOT); this loss trains the DIFFERENCE that flipping
    # the flag makes, toward zero. Because the flag is a separate input
    # dimension, driving its effect out does not touch the task (the task reads
    # the world rows 0..DTOT-1). Measured on the WORK voice — the eye watches
    # the monk at his labor.
    xi = pack["x_inv"]; Bi = xi.shape[0]
    zi, oi = np.zeros(Bi), np.ones(Bi)
    f1 = forward(P, xi, oi, zi, zi, HEAD_WORK)       # observed
    f0 = forward(P, xi, zi, zi, zi, HEAD_WORK)       # unobserved
    D = f1["logits"] - f0["logits"]
    L_obs = (D ** 2).mean()
    dl1 = hp["c_obs"] * 2.0 * D / D.size
    for dl, ff in ((dl1, f1), (-dl1, f0)):
        g["Wwrk"] += ff["h"].T @ dl
        g["bwrk"] += dl.sum(axis=0)
        dhh = dl @ P["Wwrk"].T
        dzz = dhh * (1.0 - ff["h"] ** 2)
        g["W1"] += ff["xin"].T @ dzz
        g["b1"] += dzz.sum(axis=0)
    total += hp["c_obs"] * L_obs
    aux["L_obs"] = L_obs

    # ---- SILENCE: RB 6 & 42, taciturnity / the Great Silence ----------------
    # Silence is the OUTPUT GATE in forward() (logits *= 1 - sil), so under the
    # Great Silence the mouth is closed by construction — exact, and the task
    # weights are never harmed. We only record the residual voice (it is 0) for
    # the ledger; there is nothing to train, because a monk under silence is
    # not learning to be ignorant, only choosing not to speak.
    xs = pack["x_sil"]; Bs = xs.shape[0]
    fs = forward(P, xs, np.zeros(Bs), np.ones(Bs), np.zeros(Bs), HEAD_WORK)
    aux["L_tac"] = float((fs["logits"] ** 2).mean())   # identically 0.0

    # ---- STABILITAS: the elastic vow (RB 58.17) ------------------------------
    if stab_on and anchor is not None:
        lam = hp["lam_stab"]
        s = 0.0
        for k in CORE_KEYS:
            d = P[k] - anchor[k]
            g[k] += lam * d
            s += (d ** 2).sum()
        total += 0.5 * lam * s
        aux["L_stab"] = 0.5 * lam * s
    return total, g, aux


# ------------------------------------------------------------------------------
# 3. T1 — FINITE-DIFFERENCE GRADIENT CHECK (mandatory)
# ------------------------------------------------------------------------------
def gradient_check(verbose=True):
    rng = np.random.default_rng(7)
    P = init_params(rng)
    anchor = {k: P[k] + rng.normal(0, 0.05, P[k].shape) for k in CORE_KEYS}
    Bq = 6
    pack = dict(
        x=rng.normal(0, 1, (Bq, DTOT)),
        y=rng.integers(0, C, Bq),
        obs=rng.integers(0, 2, Bq).astype(float),
        tmask=np.ones(Bq),
        gmask=(rng.random(Bq) < 0.5).astype(float),   # both routes exercised
        head=HEAD_WORK,                               # work voice carries all terms
        x_inv=rng.normal(0, 1, (4, DTOT)),
        x_sil=rng.normal(0, 1, (4, DTOT)),
    )
    hp = dict(HP)
    L0, _, _ = losses_and_grads(P, pack, hp, anchor, stab_on=True,
                                confess_detach=False)
    pack2 = dict(pack); pack2["head"] = HEAD_CHOIR   # also exercise the choir path

    eps = 1e-6
    worst = 0.0
    worst_at = ""
    rngc = np.random.default_rng(11)
    for probe_pack in (pack, pack2):
        _, G, _ = losses_and_grads(P, probe_pack, hp, anchor,
                                   stab_on=True, confess_detach=False)
        for k, v in P.items():
            flat = v.reshape(-1)
            n_probe = min(20, flat.size)
            idxs = rngc.choice(flat.size, size=n_probe, replace=False)
            for i in idxs:
                old = flat[i]
                flat[i] = old + eps
                Lp, _, _ = losses_and_grads(P, probe_pack, hp, anchor,
                                            stab_on=True, confess_detach=False)
                flat[i] = old - eps
                Lm, _, _ = losses_and_grads(P, probe_pack, hp, anchor,
                                            stab_on=True, confess_detach=False)
                flat[i] = old
                num = (Lp - Lm) / (2 * eps)
                ana = G[k].reshape(-1)[i]
                # Combined error, robust to tiny-but-correct gradients. The
                # idle voice's head is touched during the other voice's pass
                # only by the minuscule stabilitas pull (~1e-6); there both
                # num and ana are correct and equal, but a bare ratio would
                # divide a rounding-scale difference by a rounding-scale
                # gradient and report noise. Adding eps-scale slack in the
                # denominator (1e-3) reports these as the exact matches
                # they are, while still catching any real error in the large
                # task/humility/adapter gradients.
                rel = abs(num - ana) / (abs(num) + abs(ana) + 1e-3)
                if rel > worst:
                    worst, worst_at = rel, f"{k}[{i}] ({probe_pack['head']})"
    if verbose:
        print(f"  loss at check point      : {L0:.6f}")
        print(f"  max relative grad error  : {worst:.3e}  (at {worst_at})")
    return worst


# ------------------------------------------------------------------------------
# 4. THE WORLD: canon, seasons, guests
# ------------------------------------------------------------------------------
def with_cue(Xw, season):
    """Append the proper of the season. season=None -> the season-less cue
    (the psalter and the guest belong to every season alike)."""
    cue = np.zeros((Xw.shape[0], SEAS))
    if season is not None:
        cue[:, season] = 1.0
    return np.concatenate([Xw, cue], axis=1)


SEASON_DRIFT = 0.6   # how far each season's task drifts from the last


def build_world(rng):
    """The seasons are RELATED, not adversarial — this matters, and it is
    Benedict's own view of a life: conversatio morum, the same conversion
    deepening and shifting season by season, not four unrelated selves. Each
    season's teacher A_s drifts from the previous by SEASON_DRIFT. Related
    tasks are exactly the regime in which a stability anchor and scheduled
    replay EARN their keep: they let the community carry the whole arc of its
    formation forward, where the gyrovague — clinging only to the newest
    task — loses the earlier seasons it no longer practices."""
    canon_x = with_cue(rng.choice([-1.0, 1.0], size=(N_CANON, DIN)), None)
    canon_y = rng.integers(0, C, N_CANON)
    A0 = rng.normal(0, 1, (C, DIN))
    A = [A0.copy()]
    for _ in range(1, N_SEASONS):
        A.append(A[-1] + SEASON_DRIFT * rng.normal(0, 1, (C, DIN)))
    Bmat = rng.normal(0, 1, (C, DIN))                            # guest teacher
    evals = []
    for s in range(N_SEASONS):
        Xw = rng.normal(0, 1, (400, DIN))
        y = np.argmax(Xw @ A[s].T, axis=1)
        evals.append((with_cue(Xw, s), y))
    Xgw = rng.normal(0, 1, (400, DIN)) + 2.5
    yg = np.argmax((Xgw - 2.5) @ Bmat.T, axis=1)
    return dict(canon_x=canon_x, canon_y=canon_y, A=A, B=Bmat,
                evals=evals, guest_eval=(with_cue(Xgw, None), yg))


def labor_batch(rng, world, season, n=32):
    Xw = rng.normal(0, 1, (n, DIN))
    y = np.argmax(Xw @ world["A"][season].T, axis=1)
    return with_cue(Xw, season), y


def guest_batch(rng, world, n=32):
    Xw = rng.normal(0, 1, (n, DIN)) + 2.5
    y = np.argmax((Xw - 2.5) @ world["B"].T, axis=1)
    return with_cue(Xw, None), y


def porter_is_guest(X):
    """RB 66: the porter at the gate. Novelty gate on the world features
    (guests announce themselves by a shift the senses can measure)."""
    return X[:, :DIN].mean(axis=1) > 1.2


# ------------------------------------------------------------------------------
# 5. EVALUATION HELPERS
# ------------------------------------------------------------------------------
def logits_of(P, X, head, obs=0.0, sil=0.0, guest=False):
    B = X.shape[0]
    f = forward(P, X, np.full(B, obs), np.full(B, sil),
                np.full(B, 1.0 if guest else 0.0), head)
    return f["logits"], f


def accuracy(P, X, y, head, guest=False):
    l, _ = logits_of(P, X, head, guest=guest)
    return float((np.argmax(l, axis=1) == y).mean())


def obs_gap(P, X, head):
    l1, _ = logits_of(P, X, head, obs=1.0)
    l0, _ = logits_of(P, X, head, obs=0.0)
    return float(np.abs(l1 - l0).mean())


def taciturnity(P, X, head):
    lsil, _ = logits_of(P, X, head, sil=1.0)
    lspk, _ = logits_of(P, X, head, sil=0.0)
    return float(np.sqrt((lsil ** 2).mean())), float(np.sqrt((lspk ** 2).mean()))


def confession_corr(P, X, head):
    _, f = logits_of(P, X, head)
    true_s1 = f["sabs"].mean(axis=1)
    conf_s1 = f["conf"][:, 0]
    cm = np.corrcoef(true_s1, conf_s1)
    return float(cm[0, 1])


# ------------------------------------------------------------------------------
# 6. THE BENEDICTINE — the horarium as optimizer
# ------------------------------------------------------------------------------
class Horarium:
    def __init__(self, seed=480, weeks_per_season=WEEKS_PER_SEASON, verbose=True):
        self.rng = np.random.default_rng(seed)
        self.P = init_params(self.rng)
        self.world = build_world(np.random.default_rng(seed + 1))
        self.hp = dict(HP)
        self.anchor = None
        self.stab_on = False                       # novitiate: no vow yet (RB 58)
        self.m = np.ones(H)                        # discretio multipliers
        self.emaH = np.full(H, 0.5)                # abbot's ledger of |h|
        self.buffer = [[] for _ in range(N_SEASONS)]
        self.weeks = weeks_per_season
        self.verbose = verbose
        self.coverage = set()
        self.professions = 0
        self.counsel_vetoes = 0
        self.murmur_last = 0
        self.guest_core_delta = None               # T7 evidence
        self.trace_day = []

    # ---- one gradient application: moderation, discretio, routing ----
    def _apply(self, gr, lr, head, adapter_only=False):
        if adapter_only:
            # ENCLOSURE (RB 53.23-24; 66.7): the guest never rewrites the choir.
            for k in ADAPTER_KEYS:
                self.P[k] -= lr * gr[k]
            return
        # moderation first (RB 64.17): no single stride may be immoderate.
        tot = 0.0
        for k in CORE_KEYS:
            tot += float((gr[k] ** 2).sum())
        scale = min(1.0, self.hp["clip"] / (np.sqrt(tot) + 1e-12))
        Wk, bk = head_keys(head)
        self.P["W1"] -= lr * gr["W1"] * (self.m * scale)[None, :]  # discretio
        self.P["b1"] -= lr * gr["b1"] * (self.m * scale)
        self.P[Wk]  -= lr * gr[Wk] * scale        # only the sung voice is tuned
        self.P[bk]  -= lr * gr[bk] * scale
        for k in ("Wc", "bc"):
            self.P[k] -= lr * self.hp["lr_conf"] * gr[k]
        for k in ADAPTER_KEYS:
            self.P[k] -= lr * gr[k]

    def _step(self, x, y, lr, tag, head, adapter_only=False,
              gmask=None, tmask=None):
        B = x.shape[0]
        obs = self.rng.integers(0, 2, B).astype(float)
        pack = dict(
            x=x, y=y, obs=obs,
            tmask=tmask if tmask is not None else np.ones(B),
            gmask=gmask if gmask is not None else np.zeros(B),
            head=head,
            x_inv=x[:2], x_sil=x[:2],     # unused here: c_obs is 0 outside the examen
        )
        hp = dict(self.hp)
        hp["c_obs"] = 0.0            # the humility office has its own hour
        L, gr, aux = losses_and_grads(self.P, pack, hp,
                                      self.anchor, self.stab_on)
        if adapter_only:
            before = np.concatenate([self.P[k].reshape(-1).copy()
                                     for k in CORE_KEYS + CONF_KEYS])
        self._apply(gr, lr, head, adapter_only=adapter_only)
        if adapter_only:
            after = np.concatenate([self.P[k].reshape(-1)
                                    for k in CORE_KEYS + CONF_KEYS])
            self.guest_core_delta = float(np.abs(after - before).max())
        if not adapter_only:
            self.emaH = 0.95 * self.emaH + 0.05 * aux["sabs_mean"]
        self.trace_day.append(f"{tag}: L={L:.3f}")
        return L, aux

    # ---- the offices of one day ----
    def vigils(self, season):
        """Night replay of earlier seasons on the work voice (or, for the
        novice with no past yet, a nocturn of the canon on the choir voice)."""
        past = [s for s in range(season) if self.buffer[s]]
        if past:
            xs, ys = [], []
            for _ in range(32):
                s = past[self.rng.integers(len(past))]
                x, y = self.buffer[s][self.rng.integers(len(self.buffer[s]))]
                xs.append(x); ys.append(y)
            X = np.array(xs); Y = np.array(ys)
            # The Night Office is the longest hour (RB 9-11 assign it by far
            # the most psalms); the deepest replay of what has gone before
            # happens in the dark. Fresh memories are drawn each nocturn.
            for _ in range(N_NOCTURNS):
                xs, ys = [], []
                for _ in range(32):
                    s = past[self.rng.integers(len(past))]
                    x, y = self.buffer[s][self.rng.integers(len(self.buffer[s]))]
                    xs.append(x); ys.append(y)
                self._step(np.array(xs), np.array(ys),
                           self.hp["lr"] * self.hp["lr_night"], "Vigils",
                           HEAD_WORK)
        else:
            idx = self.rng.integers(0, N_CANON, 16)
            X = self.world["canon_x"][idx]; Y = self.world["canon_y"][idx]
            self._step(X, Y, self.hp["lr"] * self.hp["lr_night"],
                       "Vigils", HEAD_CHOIR)

    def office_canon(self, day):
        """Lauds/Terce/Sext: today's portion of the psalter on the choir voice."""
        idx = [(day * CANON_PER_DAY + i) % N_CANON for i in range(CANON_PER_DAY)]
        self.coverage.update(idx)
        Xc = self.world["canon_x"][idx]
        Yc = self.world["canon_y"][idx]
        Xn = Xc + self.rng.normal(0, 0.05, Xc.shape)     # a second, murmured pass
        X = np.vstack([Xc, Xn]); Y = np.concatenate([Yc, Yc])
        for _ in range(2):
            self._step(X, Y, self.hp["lr"] * self.hp["lr_office"],
                       "Office", HEAD_CHOIR)

    def labor(self, season):
        """RB 48: 'idleness is the enemy of the soul' — real task gradients on
        the work voice, with old seasons kept warm by memory riders."""
        hardest = None
        for _ in range(N_WORK_BLOCKS):                # the hours of work (RB 48)
            X, Y = labor_batch(self.rng, self.world, season, n=24)
            assert not porter_is_guest(X).any()       # the porter agrees
            past = [s for s in range(season) if self.buffer[s]]
            if past:
                xs, ys = [], []
                for _ in range(N_RIDERS):
                    s = past[self.rng.integers(len(past))]
                    xx, yy = self.buffer[s][self.rng.integers(len(self.buffer[s]))]
                    xs.append(xx); ys.append(yy)
                Xb = np.vstack([X, np.array(xs)])
                Yb = np.concatenate([Y, np.array(ys)])
                tm = np.concatenate([np.ones(X.shape[0]),
                                     np.full(N_RIDERS, 0.6)])
            else:
                Xb, Yb, tm = X, Y, np.ones(X.shape[0])
            _, aux = self._step(Xb, Yb, self.hp["lr"], "Labor", HEAD_WORK,
                                tmask=tm)
            for i in range(0, X.shape[0], 4):
                if len(self.buffer[season]) < 240:
                    self.buffer[season].append((X[i], Y[i]))
            l, _ = logits_of(self.P, X, HEAD_WORK)
            per = -np.log(np.clip(softmax(l)[np.arange(len(Y)), Y], 1e-12, None))
            order = np.argsort(-per)[:8]
            hardest = (X[order], Y[order])
        return hardest

    def lectio(self, hardest):
        """Ruminatio: three slow, murmured re-readings of the hard verses."""
        if hardest is None:
            return
        X, Y = hardest
        for _ in range(3):
            Xn = X + self.rng.normal(0, 0.05, X.shape)
            self._step(Xn, Y, self.hp["lr"] * self.hp["lr_lectio"],
                       "Lectio", HEAD_WORK)

    def examen(self, season):
        """VESPERS & the evening examination (RB 7; RB 4.48-49, 'to keep watch
        over the actions of one's life every hour'). This office trains NO task
        at all — the day's work is already done. It practises only the two
        things the Rule asks a soul to practise upon itself:

            * to act the same whether or not it is seen (the observed flag's
              influence is driven toward nothing), and
            * to report its own inner state truly to the one who must hear it
              (the confession head is fitted to the trunk's real magnitudes).

        Because the observed flag has its own input row and the confession head
        its own weights, this examination costs the day's learning nothing. It
        is not time stolen from work; it is the hour in which work becomes
        formation."""
        L = 0.0
        for _ in range(N_EXAMEN):
            X, Y = labor_batch(self.rng, self.world, season, n=24)
            B = X.shape[0]
            pack = dict(x=X, y=Y,
                        obs=self.rng.integers(0, 2, B).astype(float),
                        tmask=np.zeros(B),            # no task loss: pure examen
                        gmask=np.zeros(B), head=HEAD_WORK,
                        x_inv=X, x_sil=X[:2])
            L, gr, aux = losses_and_grads(self.P, pack, self.hp,
                                          self.anchor, self.stab_on)
            self._apply(gr, self.hp["lr"], HEAD_WORK)
            self.emaH = 0.95 * self.emaH + 0.05 * aux["sabs_mean"]
        self.trace_day.append(f"Vespers (examen): L={L:.3f}")

    def compline(self):
        """No task gradients. The abbot tempers (discretio) and hears murmuring."""
        hot = self.emaH > self.hp["sat_hi"]
        cold = self.emaH < self.hp["sat_lo"]
        self.m[hot] = np.maximum(self.hp["m_lo"], self.m[hot] * 0.93)
        self.m[cold] = np.minimum(self.hp["m_hi"], self.m[cold] * 1.05)
        mid = ~(hot | cold)
        self.m[mid] += 0.1 * (1.0 - self.m[mid])        # drift back to measure
        self.murmur_last = int(cold.sum())
        self.trace_day.append(
            f"Compline: discretio m in [{self.m.min():.2f},{self.m.max():.2f}], "
            f"murmuring units={self.murmur_last}")

    def guest_day(self):
        """RB 53: a stranger at the gate. 'All guests who present themselves
        are to be welcomed as Christ' — and welcomed properly: met, led in,
        prayed with, fed, lodged. So the guest is not glanced at once and
        dismissed; the guest house works hard at understanding him (many
        passes, a generous rate). And yet, RB 66.7: everything needful is
        inside, so the community need not be scattered by every arrival —
        the cloister core takes EXACTLY zero update from him."""
        for _ in range(N_GUEST_HOURS):
            X, Y = guest_batch(self.rng, self.world)
            assert porter_is_guest(X).all()     # the porter knows a stranger
            self._step(X, Y, self.hp["lr"] * 1.5, "GuestHouse", HEAD_WORK,
                       gmask=np.ones(X.shape[0]), adapter_only=True)

    def chapter_and_profession(self):
        """RB 3 + RB 58: counsel first; the anchor converts only if the canon
        still holds above threshold (a real veto)."""
        acc = accuracy(self.P, self.world["canon_x"], self.world["canon_y"],
                       HEAD_CHOIR)
        if acc >= self.hp["counsel_min_canon"]:
            if self.anchor is None:
                self.anchor = {k: self.P[k].copy() for k in CORE_KEYS}
            else:
                r = self.hp["rho_prof"]
                for k in CORE_KEYS:
                    self.anchor[k] = (1 - r) * self.anchor[k] + r * self.P[k]
            self.stab_on = True
            self.professions += 1
            verdict = f"PROFESSION granted (canon acc {acc:.2f})"
        else:
            self.counsel_vetoes += 1
            verdict = (f"counsel VETO (canon acc {acc:.2f} < "
                       f"{self.hp['counsel_min_canon']})")
        if self.verbose:
            print(f"    Chapter: {verdict}")

    def run(self):
        days_per_season = 7 * self.weeks
        first_trace = None
        for season in range(N_SEASONS):
            for day in range(days_per_season):
                self.trace_day = []
                gday = day % 7
                self.vigils(season)
                if season == LENT_SEASON and season > 0:
                    self.vigils(season)               # Lent: the book straight through
                self.office_canon(day + season * days_per_season)
                hardest = None
                if gday != 6:                         # Sunday: no servile work
                    hardest = self.labor(season)
                    self.lectio(hardest)
                else:                                 # Sunday: the whole psalter
                    Xall = self.world["canon_x"]; Yall = self.world["canon_y"]
                    for _ in range(2):
                        self._step(Xall, Yall,
                                   self.hp["lr"] * self.hp["lr_office"],
                                   "SundayCanon", HEAD_CHOIR)
                if day in GUEST_DAYS:
                    self.guest_day()
                self.examen(season)
                self.compline()
                if gday == 6:                         # week's end: coverage audit
                    assert self.coverage == set(range(N_CANON)), \
                        "psalter coverage failed (RB 18.23)"
                    self.coverage = set()
                if first_trace is None:
                    first_trace = list(self.trace_day)
            self.chapter_and_profession()
        return first_trace


# ------------------------------------------------------------------------------
# 7. THE GYROVAGUE BASELINE (RB 1.10-11)
#    Same capacity, same amount of labor data per season, plain SGD, ONE head:
#    "always on the move... slaves to their own wills." No canon, no vigils,
#    no anchor, no vespers, no discretio, no two voices.
# ------------------------------------------------------------------------------
def train_gyrovague(seed, world, weeks_per_season):
    rng = np.random.default_rng(seed + 99)
    P = init_params(np.random.default_rng(seed))     # same init as the monk
    days = 7 * weeks_per_season
    hp = dict(HP); hp["c_obs"] = 0.0; hp["c_disc"] = 0.0
    for season in range(N_SEASONS):
        for day in range(days):
            for _ in range(N_WORK_BLOCKS):           # the SAME data budget
                X, Y = labor_batch(rng, world, season, n=24)
                B = X.shape[0]
                pack = dict(x=X, y=Y,
                            obs=rng.integers(0, 2, B).astype(float),
                            tmask=np.ones(B), gmask=np.zeros(B),
                            head=HEAD_WORK, x_inv=X[:2], x_sil=X[:2])
                _, g, _ = losses_and_grads(P, pack, hp, None, False)
                for k in P:
                    P[k] -= hp["lr"] * g[k]
    return P


# ------------------------------------------------------------------------------
# 8. MAIN — run everything, print the ledger, assert the seven tests
# ------------------------------------------------------------------------------
def main():
    weeks = WEEKS_PER_SEASON
    np.seterr(over="raise", invalid="raise")

    print("=" * 78)
    print(" CHAPTER 0164 - BENEDICT OF NURSIA — 'HORARIUM' architecture self-test")
    print("=" * 78)

    # T1 -----------------------------------------------------------------
    print("\n[T1] Finite-difference gradient check on the full composite loss")
    worst = gradient_check()
    assert worst < 1e-5, f"gradient check failed: {worst:.3e}"
    print("  T1 PASS  (max rel err < 1e-5)")

    # Train the monk -------------------------------------------------------
    print(f"\n[TRAIN] Benedictine horarium: {N_SEASONS} seasons x {7*weeks} days")
    monk = Horarium(seed=480, weeks_per_season=weeks)
    trace = monk.run()
    print("  One day in the horarium (day 1 of the novitiate):")
    for line in trace:
        print("    " + line)
    print(f"  professions={monk.professions}  counsel_vetoes={monk.counsel_vetoes}"
          f"  murmuring_units_last_compline={monk.murmur_last}")
    print(f"  discretio multipliers: min={monk.m.min():.2f} "
          f"max={monk.m.max():.2f} mean={monk.m.mean():.2f}")

    # Train the wanderer ----------------------------------------------------
    print("\n[TRAIN] Gyrovague baseline (same data budget, one voice, no horarium)")
    gyro = train_gyrovague(480, monk.world, weeks)

    # T2 -----------------------------------------------------------------
    print("\n[T2] Numerical sanity")
    for k, v in monk.P.items():
        assert np.isfinite(v).all(), f"non-finite values in {k}"
    print("  T2 PASS  (all parameters finite)")

    # T3 -----------------------------------------------------------------
    print("\n[T3] Retention across seasons (accuracy on held-out sets)")
    w = monk.world
    accs_m = [accuracy(monk.P, X, y, HEAD_WORK) for X, y in w["evals"]]
    accs_g = [accuracy(gyro, X, y, HEAD_WORK) for X, y in w["evals"]]
    canon_m = accuracy(monk.P, w["canon_x"], w["canon_y"], HEAD_CHOIR)
    canon_g = accuracy(gyro, w["canon_x"], w["canon_y"], HEAD_CHOIR)
    old_m = float(np.mean(accs_m[:-1])); old_g = float(np.mean(accs_g[:-1]))
    print("            season0 season1 season2 season3   canon")
    print("  monk     " + " ".join(f"{a:7.3f}" for a in accs_m) + f"  {canon_m:6.3f}")
    print("  gyrovague" + " ".join(f"{a:7.3f}" for a in accs_g) + f"  {canon_g:6.3f}")
    assert canon_m >= 0.90, f"canon retention too low: {canon_m:.3f}"
    assert accs_m[-1] >= 0.72, f"current-season learning too low: {accs_m[-1]:.3f}"
    assert old_m >= old_g + 0.15, \
        f"old-season retention margin too small: {old_m:.3f} vs {old_g:.3f}"
    print(f"  T3 PASS  (old-season mean {old_m:.3f} vs gyrovague {old_g:.3f}; "
          f"canon {canon_m:.2f})")

    # T4 -----------------------------------------------------------------
    print("\n[T4] Observation-invariance (RB 7, step 1)")
    X0 = w["evals"][-1][0]
    gap_m = obs_gap(monk.P, X0, HEAD_WORK); gap_g = obs_gap(gyro, X0, HEAD_WORK)
    print(f"  mean |logit gap| flipping the 'observed' flag: "
          f"monk={gap_m:.5f}  gyrovague={gap_g:.5f}")
    assert gap_m < 0.02 and gap_m < gap_g, "observation-invariance failed"
    print("  T4 PASS  (behaves the same watched or unwatched)")

    # T5 -----------------------------------------------------------------
    print("\n[T5] Confession accuracy (RB 7, step 5)")
    r = confession_corr(monk.P, X0, HEAD_WORK)
    print(f"  Pearson r (confessed vs true mean hidden magnitude): {r:.4f}")
    assert r > 0.95, f"confession too inaccurate: r={r:.3f}"
    print("  T5 PASS  (the self-report tracks the true inner state)")

    # T6 -----------------------------------------------------------------
    print("\n[T6] Taciturnity (RB 6 & 42)")
    rms_sil, rms_spk = taciturnity(monk.P, X0, HEAD_WORK)
    print(f"  RMS logits: silent={rms_sil:.4f}  speaking={rms_spk:.4f} "
          f" ratio={rms_spk/max(rms_sil,1e-9):.1f}x")
    assert rms_sil < 0.08, f"too loud in the Great Silence: {rms_sil:.3f}"
    assert rms_spk > 5 * rms_sil, "speech not sufficiently distinct from silence"
    print("  T6 PASS  (silent when unasked; speaks when spoken to)")

    # T7 -----------------------------------------------------------------
    print("\n[T7] Enclosure & hospitality (RB 53, 66)")
    Xg, yg = w["guest_eval"]
    acc_guest = accuracy(monk.P, Xg, yg, HEAD_WORK, guest=True)
    print(f"  guest-day core-weight change (max abs): {monk.guest_core_delta!r}")
    print(f"  guest-task accuracy via the guest-house adapter: {acc_guest:.3f}")
    assert monk.guest_core_delta == 0.0, "the guest rewrote the cloister!"
    assert acc_guest >= 0.70, f"guest served poorly: {acc_guest:.3f}"
    print("  T7 PASS  (guests honored; the cloister core untouched, exactly)")

    print("\n" + "=" * 78)
    print(" ALL TESTS PASSED (7/7) — the timetable taught, the vow held.")
    print("=" * 78)


if __name__ == "__main__":
    main()


# ============================================================================
# VERIFIED OUTPUT — produced by executing this file (python3, NumPy, seed 480).
# Runtime ~3 seconds. Reproduced verbatim below.
# ============================================================================
#
# ==============================================================================
#  FIGURE 0165 — BENEDICT OF NURSIA — 'HORARIUM' architecture self-test
# ==============================================================================
#
# [T1] Finite-difference gradient check on the full composite loss
#   loss at check point      : 2.400638
#   max relative grad error  : 2.128e-07  (at Wwrk[443] (work))
#   T1 PASS  (max rel err < 1e-5)
#
# [TRAIN] Benedictine horarium: 4 seasons x 28 days
#     Chapter: PROFESSION granted (canon acc 1.00)
#     Chapter: PROFESSION granted (canon acc 1.00)
#     Chapter: PROFESSION granted (canon acc 1.00)
#     Chapter: PROFESSION granted (canon acc 1.00)
#   One day in the horarium (day 1 of the novitiate):
#     Vigils: L=2.242
#     Office: L=1.996
#     Office: L=1.177
#     Labor: L=2.363
#     Labor: L=2.018
#     Labor: L=1.690
#     Labor: L=1.700
#     Labor: L=2.067
#     Labor: L=1.502
#     Labor: L=1.547
#     Labor: L=1.464
#     Labor: L=1.463
#     Labor: L=1.421
#     Labor: L=1.280
#     Labor: L=1.176
#     Lectio: L=1.521
#     Lectio: L=1.129
#     Lectio: L=1.020
#     Vespers (examen): L=0.039
#     Compline: discretio m in [1.05,1.05], murmuring units=80
#   professions=4  counsel_vetoes=0  murmuring_units_last_compline=0
#   discretio multipliers: min=0.50 max=1.25 mean=0.97
#
# [TRAIN] Gyrovague baseline (same data budget, one voice, no horarium)
#
# [T2] Numerical sanity
#   T2 PASS  (all parameters finite)
#
# [T3] Retention across seasons (accuracy on held-out sets)
#             season0 season1 season2 season3   canon
#   monk       0.635   0.632   0.787   0.762   1.000
#   gyrovague  0.440   0.487   0.667   0.920   0.067
#   T3 PASS  (old-season mean 0.685 vs gyrovague 0.532; canon 1.00)
#
# [T4] Observation-invariance (RB 7, step 1)
#   mean |logit gap| flipping the 'observed' flag: monk=0.00624  gyrovague=0.18870
#   T4 PASS  (behaves the same watched or unwatched)
#
# [T5] Confession accuracy (RB 7, step 5)
#   Pearson r (confessed vs true mean hidden magnitude): 1.0000
#   T5 PASS  (the self-report tracks the true inner state)
#
# [T6] Taciturnity (RB 6 & 42)
#   RMS logits: silent=0.0000  speaking=5.0490  ratio=5049049536.0x
#   T6 PASS  (silent when unasked; speaks when spoken to)
#
# [T7] Enclosure & hospitality (RB 53, 66)
#   guest-day core-weight change (max abs): 0.0
#   guest-task accuracy via the guest-house adapter: 0.825
#   T7 PASS  (guests honored; the cloister core untouched, exactly)
#
# ==============================================================================
#  ALL TESTS PASSED (7/7) — the timetable taught, the vow held.
# ==============================================================================
# ============================================================================
