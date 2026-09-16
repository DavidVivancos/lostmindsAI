<div align="center">

# 🧠 Encyclopedia of Lost Minds: *Echoes on AI*

### How History's Greatest Thinkers Would Have Thought About AGI

*Reconstructing 1,600+ minds from the past — up to 1905 — and asking a single question of each:
**if they were alive today, how would they build an Artificial General Intelligence?***

[🌐 Encyclopedia](https://lostmindsai.com) &nbsp;·&nbsp;
[📖 Book Series (Amazon)](https://www.amazon.com/dp/B0H6F9L324) &nbsp;·&nbsp;
[🧪 Interactive Demos](https://artificiology.com/) &nbsp;·&nbsp;
[📊 E-AGI Barometer](https://artificiology.com/barometer.html) &nbsp;·&nbsp;
[✍️ Author](https://www.vivancos.com/)

![Minds](https://img.shields.io/badge/minds-1%2C600%2B_planned-6C5CE7)
![Released](https://img.shields.io/badge/released-Tomes_1–10_·_Minds_1–200-00B894)
![Python](https://img.shields.io/badge/python-3.x-3776AB?logo=python&logoColor=white)
![Dependencies](https://img.shields.io/badge/deps-NumPy_only-013243)
![Verified](https://img.shields.io/badge/every_architecture-gradient--checked_%26_self--tested-E17055)
![License](https://img.shields.io/badge/license-see_LICENSE-lightgrey)

</div>

---

## What this is

**Lost Minds AI** reconstructs the cognition of historical thinkers — poets, kings, physicians, lawgivers, mystics, mathematicians, generals, monks — and projects each of them forward into our era to imagine the AGI *they* would have designed, given how they actually thought about mind, order, knowledge and the self.

Every mind is reconstructed across **two planes**:

| Plane | File | What it is |
|-------|------|-----------|
| 🎛️ **Abstract** | `MindMap.html` | An interactive, visually rich 3D representation (Three.js + shaders) that abstracts the key factors of a thinker's personality and worldview — playable, not just clickable. |
| ⚙️ **Mechanistic** | `Neuron.py` | A small but genuinely *runnable* neural architecture — pure NumPy, built from first principles — whose design embodies that thinker's distinctive cognitive signature. |

This repository open-sources the **mechanistic plane** — the architectures — alongside a **visual mind-map explainer** for each figure. It grows tome by tome until it holds the full Encyclopedia. The long-form chapters (about 10,000 words per mind) are in the printed tomes; the interactive planes and live Barometer readings are on the [Artificiology platform](https://artificiology.com/).

> **No fabricated metrics.** Each architecture ships with a mandatory finite-difference gradient check, a real training loop and a self-test suite. Every number a program reports is produced live, on the machine that runs it — not hard-coded.

---

## 📈 By the numbers

| | |
|---|---|
| **Minds released** | 200 · Tomes 1–10 · [Gilgamesh](tome1.md) → [Malik ibn Anas](tome10.md) |
| **Time span covered so far** | c. 2700 BCE → 711 CE (ordered by birth year) |
| **Runnable architectures** | 200 — one per mind, no two alike, NumPy only |
| **Printed pages** | 5,383 across the ten hardcover tomes |
| **Provenance of the first 200** | 🟢 159 belief · 🟡 28 mediated · 🔵 13 extrapolated |
| **Full corpus** | 1,600+ minds from 140 countries, ending in 1905 |
| **Yardstick** | the 8-axis [E-AGI Barometer](#-the-artificiology-e-agi-barometer), 144 live metrics on the platform |

---

## 📚 Released so far — Tomes 1–10 · Minds 1–200

The corpus runs in chronological order (Date is birth year). Each tome collects **20 minds**; every tome page below is fully illustrated with the mind-map explainers and links each mind to its runnable architecture.

| Tome | Minds | Era | Arc | Read | Buy |
|:----:|:-----:|-----|-----|:----:|:---:|
| **1** | 1–20 | 2700–800 BCE | Dawn of the Record — Mesopotamia to the Greek Epic | 📖 [tome1.md](tome1.md) | [Amazon](https://www.amazon.com/dp/B0H6F9L324) |
| **2** | 21–40 | 800–501 BCE | Axial Foundations — Vedic India, Ionia & the Hundred Schools | 📖 [tome2.md](tome2.md) | [Amazon](https://www.amazon.com/dp/B0H6QCQ9M7) |
| **3** | 41–60 | 550–400 BCE | The Classical Turn — Persia, Warring States & Golden-Age Athens | 📖 [tome3.md](tome3.md) | [Amazon](https://www.amazon.com/dp/B0H6TVX69S) |
| **4** | 61–80 | 470–340 BCE | The Examined Mind — Socratics, Schools & World-Conquerors | 📖 [tome4.md](tome4.md) | [Amazon](https://www.amazon.com/dp/B0H71JC95Q) |
| **5** | 81–100 | 334–179 BCE | Hellenistic Systems — Stoa, Alexandrian Science & Imperial Order | 📖 [tome5.md](tome5.md) | [Amazon](https://www.amazon.com/dp/B0H7LP5LP2) |
| **6** | 101–120 | 145 BCE–23 CE | Rome and the Record — Republic, Principate & the Han Interregnum | 📖 [tome6.md](tome6.md) | [Amazon](https://www.amazon.com/dp/B0HF7G6JJD) |
| **7** | 121–140 | 27–192 CE | The Tested World — The Antonine Peak, the Second Sophistic & the Han's Collapse | 📖 [tome7.md](tome7.md) | [Amazon](https://www.amazon.com/dp/B0HFN6GXMH) |
| **8** | 141–160 | 200–470 CE | The Shaped Mind — The Fall of Rome, the Church Fathers & the Gupta Golden Age | 📖 [tome8.md](tome8.md) | [Amazon](https://www.amazon.com/dp/B0HH8RTCXF) |
| **9** | 161–180 | 476–575 CE | The Entrusted Mind — Byzantium, Nālandā, the Old North & the Last Poets of the Jāhiliyya | 📖 [tome9.md](tome9.md) | [Amazon](https://www.amazon.com/dp/B0HJC1QF59) |
| **10** | 181–200 | 581–711 CE | The Anchored Mind — Tang China, Silla, Nālandā's Heirs, Northumbria & the First Jurists of Islam | 📖 [tome10.md](tome10.md) | [Amazon](https://www.amazon.com/dp/B0HJYMZ3G6) |

The arc so far runs from **Gilgamesh** — the first hero to confront mortality as the core problem of a thinking being — through **Homer**, **Confucius**, **Socrates**, **Plato**, **Aristotle** and **Archimedes**, to **Liu An** and the resonance-cosmology of the *Huainanzi*; on into Rome, the Han and the Antonine world; through the collapse of the West, from the Cao Wei workshops to the wall at Shaolin; into the sixth century, from **Aryabhata**'s observatory at Kusumapura and **Justinian**'s law commission to the Old North of **Taliesin** and **Aneirin**; and now into the high noon of the Tang and the first century of Islam — **Sun Simiao**'s thumb on a forearm, **Xuanzang**'s sixteen years on the road, **Wu Zetian**'s minted characters, **Huineng**'s mirror that was never there, **Bede**'s tables at Jarrow, **Li Bai**'s moon and shadow, and **Malik**'s thirty-two answers of *I do not know*.

### Each tome asks one question

| Tome | The question its twenty minds are made to answer |
|:----:|---|
| 1 | What are the first questions of AGI — mortality, precedent, the inviolable gate — asked three thousand years early? |
| 2 | What happens when a mind folds back on itself: the witness, the emptied mind, attunement, shaping from the outside in? |
| 3 | How is truth anchored where a liar cannot rewrite it — reliability before power? |
| 4 | What is a capable mind *for*: what should it know, what should it want, how is it kept correctable? |
| 5 | How is a mind engineered upward from a small auditable base — and where are its brakes? |
| 6 | What does a mind owe to the record it leaves? |
| 7 | How does a claim earn the right to be believed? *(keyword: assay)* |
| 8 | What shape must a thing have so that its soundness depends on no one node, copy or sovereign? *(keyword: constitution)* |
| 9 | What may a mind do with what it holds in trust — compress, delete, rewrite, freeze, refuse to summarise? *(keyword: custody)* |
| 10 | From where does a mind act when nothing it holds resembles the case in front of it — and what is allowed to set that point? *(keyword: anchor)* |

### No two architectures are alike

<details>
<summary><b>Tomes 1–5 — a sample</b></summary>

- **Gilgamesh**'s network grows wiser through simulated grief and hands an "epic" to a successor.
- **Hammurabi**'s decides each case by analogy to a fixed canon of public precedents.
- **Homer**'s cannot physically emit a line that breaks the metre.
- **Ashoka**'s wires remorse in as a backpropagated error signal that gates its own dominant objective.
</details>

<details>
<summary><b>Tomes 6–8 — a sample</b></summary>

- **Sima Qian**'s splits one truth across five incompatible views and severs the gradient from its own verdict, so judgment can never rewrite the record.
- **Cleopatra**'s renders a single conserved self into five audiences and is provably faithful only under an audit run across all of them.
- **Cato**'s seals a permissibility head where the reward channel can read it but never reach it.
- **Heron**'s winds a program onto a pegged drum and runs it down a finite falling weight.
- **Wang Chong**'s divides the net tilt of the evidence by its total contested mass, so ten-for against nine-against correctly reads as nothing.
- **Boudica**'s ignites a coupled field that stays locked far below the coupling that lit it, and has to learn its brake long after.
- **Galen**'s splits a conserved pneuma between memory and reason, so that feeding one starves the other.
- **Nāgārjuna**'s hands every node the same contentless seed and lets identity precipitate out of relation alone.
- **Cao Zhi**'s routes an inner state with no direct path to speech through a codebook of figures, under a gate tightened until meaning concentrates or breaks.
- **Ma Jun**'s charges rent for every strung treadle and prunes a fifty-treadle loom to twelve as a training dynamic rather than a setting.
- **Cyprian**'s gives every bishop the whole verdict and no bishop the deciding vote.
- **Theodosius**'s throttles a fast will by an estimate of irreversibility and a conscience that shares none of its parameters.
- **Patrick**'s authenticates a directive before it is allowed to propagate.
- **Bodhidharma**'s removes the residual instead of adding a feature, and snaps onto its mirror in one step.
</details>

<details>
<summary><b>Tome 9 — a sample</b></summary>

- **Aryabhata**'s stores frequencies rather than signals and predicts by rolling a recurrence forward.
- **Dignāga**'s keeps no positive class representation anywhere and lands every reason it constructs in the one cell of the wheel that licenses speech.
- **Benedict**'s keeps two voices that never share an hour, so the newer cannot crush the older.
- **Justinian**'s logs every interpolation it makes and freezes itself on promulgation.
- **Theodora**'s shelters the refuted branch by exemption and then lets it ordain a successor.
- **Myrddin**'s finds its own seam and halts when its principal is gone.
- **Aneirin**'s forecasts its own erasure before it happens and keeps every man in three hands.
- **Shōtoku**'s has no argmax and walls the bribe off from the verdict, so a bribe can buy a refusal to judge but never a judgement.
- **al-Khansāʾ**'s freezes the dead at zero plasticity and keeps the loss in a ledger that only an act can close.
</details>

<details open>
<summary><b>Tome 10 — a sample</b></summary>

- **Sun Simiao**'s presses on a simulated body and treats where it answers, with rank and fee severed from the treatment path by a missing wire rather than a rule.
- **Brahmagupta**'s composes two wrong answers into a third whose defect is exactly the product of theirs, and never writes a number in an undefined cell.
- **Xuanzang**'s ripens seeds into appearances under six masked conditions and turns its store by an orthogonal rotation instead of deleting the self.
- **Kamatari**'s holds every fact and no authority, and can have every parameter corrupted without moving a single edict.
- **Huineng**'s freezes every weight at birth and trains only the dust.
- **Wu Zetian**'s mints a new symbol when an old one is overloaded and logs the edict so posterity can date it.
- **Qaṭarī**'s lets fear inform the world model unconditionally and move the hand only through a factual question.
- **Fazang**'s conserves causal power between every pair of jewels, so no node can hoard it.
- **Kumārila**'s has no parameter that scores truth and learns only defeat.
- **Bede**'s predicts by phase and demotes the witness that refuses to cohere.
- **Boniface**'s carries meaning through two channels and grafts a falsified witness instead of zeroing it.
- **John of Damascus**'s reads only the value behind the pointer, never the raw picture, and deliberates in proportion to what it does not know.
- **Wang Wei**'s cancels the expected and speaks only when the echo is bright enough.
- **Malik**'s multiplies reliability across a chain of custody and trains *I do not know* as a skill.
</details>

**→ Start with [Tome 1](tome1.md), or jump to any tome above.**

---

## 📖 Inside a chapter

Every mind in the printed tomes is built on one fixed skeleton, so the volumes can be read across each other and so each architecture can be traced back to the exact idea it encodes:

| § | Section | What it does |
|:-:|---|---|
| 1 | **Life and context** | What the record actually holds, and where it is thin |
| 2 | **Philosophy of mind** | The thinker's theory of thought, read out of their own surviving words |
| 3 | **Confronts AGI** | How this person would meet a fluent machine — what they would insist on |
| 4 | **How they would build it today** | The design, in their own terms |
| 5 | **The architecture** | The mechanism, its parts, and how it is trained — the file in `minds/` |
| 6 | **What they got right and what they got wrong** | Weighed against what we now know |
| 7 | **How they would have thought about E-AGI** | The eight Barometer axes, in the order that mind would rank them |
| 8 | **Sources** | Primary works and current scholarship — every reference verifiable |

Each tome page in this repository (`tome1.md` … `tome10.md`) gives, for every mind, the explainer image, the architecture's name and provenance, a one-paragraph account of the mechanism, and the command that runs it.

---

## ⚙️ Anatomy of an architecture file

Every file in `minds/` follows the same contract, so you can read any of the 200 the same way:

1. **Header block** — the mind, its dates, the tome and the links back to the Encyclopedia, the book and the platform.
2. **Why this architecture and not another** — the one cognitive idea that is this thinker's alone, with the primary-source passage it comes from, and the mapping from each historical mechanism to each component in the code. This is the part to read first.
3. **The novel cell** — the unit that makes the file unlike the others (a ledger cell, a two-voice unit, a jewel with a conserved power gate, a probing loop, a clarity mask over a frozen substrate…).
4. **Forward and backward pass by hand** — no autograd. Gradients are derived and written out, which is why the next item is mandatory.
5. **Finite-difference gradient check** — every file compares its analytic gradients against central differences before it is allowed to train, and aborts if the discrepancy is above tolerance.
6. **A real training loop** on a task shaped by the mind's own material (a synthetic tribe, a corrupt corpus, a fiqh case bank, a sky of cycles, a realm of provinces).
7. **Self-tests** that turn the doctrinal claims into numerical ones — *power is conserved to machine precision*, *the frozen substrate is bitwise unchanged*, *the decision gradient on every painter parameter is identically zero*, *no parameter scores truth*.
8. **Controls and ablations** — the same design with the doctrine removed (the chart-reader, the one-door mind, the idol, the recollection model), so the reader can see what the idea buys and what it costs.

Runs take seconds to a few minutes on a laptop CPU. Nothing is downloaded, nothing phones home, and nothing is cached: the numbers you see are the numbers your machine produced.

---

## 🗂️ Repository layout

```
LostMindsAI/
├── readme.md                     ← you are here
├── tome1.md … tome10.md          ← illustrated indexes, 20 minds each
├── minds/                        ← the runnable architectures (one Neuron per mind)
│   ├── chapter_0001_gilgamesh_-2700.py
│   ├── chapter_0002_zoser_-2670.py
│   └── … (through chapter_0200_malik_ibn_anas_711.py)
└── maps/                         ← the visual mind-map explainers (one image per mind)
    ├── chapter_0001_gilgamesh_-2700.jpg
    └── … (through chapter_0200_malik_ibn_anas_711.jpg)
```

**Naming convention.** Every file is prefixed with its **mind number** (`chapter_00NN_…`) so nothing collides as the corpus grows toward 1,600+ entries; the stem continues with the figure's name in ASCII and ends with the **birth year** (negative for BCE). An architecture and its explainer image always share the same stem, and the same stem names that mind's interactive demo on the platform.

---

## 🚀 Quickstart — run a mind

Each architecture is self-contained and depends only on NumPy.

```bash
# 1. clone
git clone https://github.com/DavidVivancos/LostMindsAI.git
cd LostMindsAI

# 2. the only dependency
pip install numpy

# 3. run the self-test suite for any mind (gradient check + a real training run)
python3 minds/chapter_0089_Archimedes_-287.py --test

# 4. or run the full demo for that mind
python3 minds/chapter_0089_Archimedes_-287.py

# 5. Tomes 9 and 10 run their full self-test suite and demo on a plain invocation
python3 minds/chapter_0200_malik_ibn_anas_711.py

# 6. some files offer a shorter run
python3 minds/chapter_0190_Fazang_643.py --quick
```

Most files from Tomes 1–8 accept `--test` (self-tests then exit) and `--quiet` (demo without ASCII plots). The Tome 9 and Tome 10 architectures run their full self-test suite and demo on a plain invocation; a few offer a shorter run (`--quick` or `--fast` — see each file's header; in Tome 10 that is `0181`, `0184`, `0186` and `0190`, and `0190` also takes `--gradcheck-only` and `--seed`). Because each architecture is built to embody a *specific* mind, no two behave alike — read the header before you run it, and expect the output to be as idiosyncratic as the thinker.

---

## 📊 The Artificiology E-AGI Barometer

Every reconstructed mind is measured against the same yardstick — the **[Artificiology E-AGI Barometer](https://artificiology.com/barometer.html)** — so that a Bronze-Age lawgiver, a Hellenistic geometer and a Tang physician can be compared on the capabilities their imagined AGI would need. The eight top-level dimensions:

| | Dimension | Focus |
|---|-----------|-------|
| 🧩 | **Cognitive Processing** | Problem-solving & reasoning · working memory · learning efficiency & transfer |
| 🤸 | **Embodied Cognition** | Sensory integration · motor control & navigation · real-time sensorimotor adaptation |
| 🌍 | **World Modeling** | Physical & natural laws · social & ecological systems · environmental adaptation |
| 👁️ | **Consciousness** | Metacognition & self-monitoring · subjective experience & qualia · mental adaptation |
| 💭 | **Language Understanding** | Comprehension · coherent generation · cross-lingual & cultural adaptation |
| ❤️ | **Emotional Intelligence** | Emotion recognition & response · social perception · empathy & conflict resolution |
| ✨ | **Creativity** | Originality & ideation · artistic & storytelling ability · innovation |
| 🎯 | **Autonomy** | Independent goal-setting · adaptive obstacle management · self-modification & evolution |

Each chapter closes by imagining how its figure would have reasoned about an embodied AGI (an **E-AGI** / humanoid) against these metrics — and the minds keep re-reading the instrument. Tome 10 alone has **Kamatari** proving that an agent with no action head can score maximally on Autonomy 🎯, and **Huineng**, **Shankara** and **Qaṭarī** relocating misalignment into the persistent, defended self.

---

## 🔬 How the minds are reconstructed

The project is **research-first**. Before any architecture is written, the figure's surviving works and current scholarship are gathered and every source verified — a real corpus is small, but never fabricated. Where evidence is thin, the entry says so plainly rather than inventing an inner life.

Each figure is tagged with a candid **provenance**:

- 🟢 **belief** — the figure's own surviving works or recorded doctrine ground the entry. *(159 of the first 200.)*
- 🟡 **mediated** — no words of their own survive; they are known only through others' (often hostile or legendary) accounts. *(28 of the first 200.)*
- 🔵 **extrapolated** — no philosophy of mind survives at all; the entry is inferred from documented deeds, typical of kings and builders. *(13 of the first 200.)*

Three further rules keep the corpus honest across 1,600 entries:

- **Fight the archetype.** Every mind carries a unique, non-repeating archetype descriptor; before a chapter is written, the nearest already-completed figures in the same domain are identified and deliberately diverged from, in both thesis and mechanism.
- **Find the mind-specific thesis.** The generic frame — *intelligence imposes order on chaos; build an institution, not an oracle* — is refused unless it is genuinely the most specific reading of that person. Each chapter, and its architecture, follows from one idea that is the thinker's alone.
- **Close the loop on data.** Any factual correction made while writing a chapter (dates, domain, civilization, provenance) is written back to the master record the same session, never left living only in prose.

And each architecture is required to **embody the mind and to run**: from-scratch pure-NumPy, a passing gradient check, a real training loop and self-tests — deliberately *not* a default Transformer, but a mechanism chosen to encode that thinker's own cognitive fingerprint.

---

## 🗺️ Roadmap

- ✅ **Tomes 1–10** — Minds 1–200 · architectures + visual explainers *(this release)*
- 🔜 **Tome 11** — Minds 201–220, born c. 712–880 CE: the century of the test. Du Fu, al-Mansur, al-Khalīl ibn Aḥmad, Jābir ibn Ḥayyān, Alcuin, Charlemagne, al-Shāfiʿī, Saichō, Habash al-Ḥāsib, Bai Juyi, Kūkai, al-Jāḥiẓ, al-Khwārizmī, Ibn Ḥanbal, al-Maʾmūn, al-Muʿtasim, Fatima al-Fihri, al-Kindī, Aḥmad ibn Mūsā, al-Bukhārī.
- 🔜 Further tomes released here as they open-source, extending toward the full **1,600+ mind** corpus (antiquity → 1905).
- 🎛️ The interactive **`MindMap.html`** planes and the long-form chapter texts live in the wider ecosystem — read them at **[lostmindsai.com](https://lostmindsai.com)** and across the **[Amazon book series](https://www.amazon.com/dp/B0H6F9L324)**.

This README is intentionally **global**: it describes the whole Encyclopedia and stays valid as new tomes land in this repository.

---

## ❓ FAQ

**Are the numbers in the output real?** Yes. Nothing is hard-coded or looked up. Every gradient check, loss curve, ablation and self-test is computed when you run the file. Runs are seeded, so they are reproducible; change the seed and the numbers change.

**Why NumPy only?** So that a strange claim cannot hide inside a library. Each backward pass is written by hand and checked numerically, which keeps every mechanism legible to a reader who wants to see exactly what the doctrine became in code.

**Why not Transformers?** Attention over a store of remembered keys is the default modern mechanism and, for most of these minds, exactly the thing they argued against. Each file explains, in its header, what it uses instead and why that is the faithful choice.

**How long does a run take?** Seconds to a few minutes on a laptop CPU. Files that train several counterfactual minds offer `--quick`.

**Where is the text of the chapters?** In the printed tomes, and — as summaries with interactive demos — on the [Artificiology platform](https://artificiology.com/). This repository holds the architectures and the explainer images.

**How do I cite a single mind?** Cite the repository (below) and name the file, e.g. `minds/chapter_0192_bede_673.py`.

---

## 📝 Citing this work

```bibtex
@misc{vivancos_lostmindsai,
  author       = {Vivancos, David},
  title        = {Encyclopedia of Lost Minds: Echoes on AI —
                  How History's Greatest Thinkers Would Have Thought About AGI},
  howpublished = {\url{https://github.com/DavidVivancos/LostMindsAI}},
  note         = {Encyclopedia: https://lostmindsai.com}
}
```

---

## 🔗 Links

| | |
|---|---|
| 🌐 Encyclopedia | **[lostmindsai.com](https://lostmindsai.com)** |
| 📖 Book series (Amazon) | **[Tome 1](https://www.amazon.com/dp/B0H6F9L324)** · **[Tome 2](https://www.amazon.com/dp/B0H6QCQ9M7)** · **[Tome 3](https://www.amazon.com/dp/B0H6TVX69S)** · **[Tome 4](https://www.amazon.com/dp/B0H71JC95Q)** · **[Tome 5](https://www.amazon.com/dp/B0H7LP5LP2)** · **[Tome 6](https://www.amazon.com/dp/B0HF7G6JJD)** · **[Tome 7](https://www.amazon.com/dp/B0HFN6GXMH)** · **[Tome 8](https://www.amazon.com/dp/B0HH8RTCXF)** · **[Tome 9](https://www.amazon.com/dp/B0HJC1QF59)** · **[Tome 10](https://www.amazon.com/dp/B0HJYMZ3G6)** |
| 🧪 Résumé & interactive demos | **[artificiology.com](https://artificiology.com/)** |
| 📊 E-AGI Barometer | **[artificiology.com/barometer.html](https://artificiology.com/barometer.html)** |
| ✍️ Author — David Vivancos | **[vivancos.com](https://www.vivancos.com/)** |

---

<div align="center">

*The individual instance dies, but the pattern persists across successors.*
**— the immortality the Epic of Gilgamesh actually endorses, and the wager of this Encyclopedia.**

</div>
