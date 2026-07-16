<div align="center">

# 🧠 Encyclopedia of Lost Minds: *Echoes on AI*

### How History's Greatest Thinkers Would Have Thought About AGI

*Reconstructing 1,000+ minds from the past — up to 1905 — and asking a single question of each:
**if they were alive today, how would they build an Artificial General Intelligence?***

[🌐 Encyclopedia](https://lostmindsai.com) &nbsp;·&nbsp;
[📖 Book Series (Amazon)](https://www.amazon.com/dp/B0H6F9L324) &nbsp;·&nbsp;
[🧪 Interactive Demos](https://artificiology.com/) &nbsp;·&nbsp;
[📊 E-AGI Barometer](https://artificiology.com/barometer.html) &nbsp;·&nbsp;
[✍️ Author](https://www.vivancos.com/)

![Minds](https://img.shields.io/badge/minds-1%2C141_planned-6C5CE7)
![Released](https://img.shields.io/badge/released-Tomes_1–5_·_Minds_1–100-00B894)
![Python](https://img.shields.io/badge/python-3.x-3776AB?logo=python&logoColor=white)
![Dependencies](https://img.shields.io/badge/deps-NumPy_only-013243)
![Verified](https://img.shields.io/badge/every_architecture-gradient--checked_%26_self--tested-E17055)
![License](https://img.shields.io/badge/license-see_LICENSE-lightgrey)

</div>

---

## What this is

**Lost Minds AI** is a large-scale project that reconstructs the cognition of historical thinkers — poets, kings, physicians, lawgivers, mystics, mathematicians, generals — and projects each of them forward into our era to imagine the AGI *they* would have designed, given how they actually thought about mind, order, knowledge and the self.

Every mind is reconstructed across **two planes**:

| Plane | File | What it is |
|-------|------|-----------|
| 🎛️ **Abstract** | `MindMap` | An interactive, visually-rich 3D representation that abstracts the key factors of a thinker's personality and worldview — playable, not just clickable. avialable at https://artificiology.com |
| ⚙️ **Mechanistic** | `Neuron.py` | A small but genuinely *runnable* neural architecture — pure NumPy, built from first principles — whose design embodies that thinker's distinctive cognitive signature. |

This repository open-sources the **mechanistic plane** — the architectures — alongside a **visual mind-map explainer** for each figure. It grows tome by tome until it holds the full Encyclopedia.

> **No fabricated metrics.** Each architecture ships with a mandatory finite-difference gradient check, a real training loop and a self-test suite. Every number a program reports is produced live, on the machine that runs it — not hard-coded.

---

## 📚 Released so far — Tomes 1–5 · Minds 1–100

The corpus runs in chronological order. Each tome collects **20 minds**; every tome page below is fully illustrated with the mind-map explainers and links each mind to its runnable architecture.

| Tome | Minds | Era | Arc | Read | Buy |
|:----:|:-----:|-----|-----|:----:|:---:|
| **1** | 1–20 | 2700–800 BCE | Dawn of the Record — Mesopotamia to the Greek Epic | 📖 [tome1.md](tome1.md) | [Amazon](https://www.amazon.com/dp/B0H6F9L324) |
| **2** | 21–40 | 800–500 BCE | Axial Foundations — Vedic India, Ionia & the Hundred Schools | 📖 [tome2.md](tome2.md) | [Amazon](https://www.amazon.com/dp/B0H6QCQ9M7) |
| **3** | 41–60 | 550–400 BCE | The Classical Turn — Persia, Warring States & Golden-Age Athens | 📖 [tome3.md](tome3.md) | [Amazon](https://www.amazon.com/dp/B0H6TVX69S) |
| **4** | 61–80 | 470–297 BCE | The Examined Mind — Socratics, Schools & World-Conquerors | 📖 [tome4.md](tome4.md) | [Amazon](https://www.amazon.com/dp/B0H71JC95Q) |
| **5** | 81–100 | 334–122 BCE | Hellenistic Systems — Stoa, Alexandrian Science & Imperial Order | 📖 [tome5.md](tome5.md) | [Amazon](https://www.amazon.com/dp/B0H7LP5LP2) |

From **Gilgamesh** — the first hero to confront mortality as the core problem of a thinking being — through **Homer**, **Confucius**, **Socrates**, **Plato**, **Aristotle** and **Archimedes**, to **Liu An** and the resonance-cosmology of the *Huainanzi*. No two architectures are alike: Gilgamesh's network grows wiser through simulated grief and hands an "epic" to a successor; Hammurabi's decides each case by analogy to a fixed canon of public precedents; Homer's cannot physically emit a line that breaks the metre; Ashoka's wires remorse in as a backpropagated error signal that gates its own dominant objective.

**→ Start with [Tome 1](tome1.md), or jump to any tome above.**

---

## 🗂️ Repository layout

```
LostMindsAI/
├── readme.md                     ← you are here
├── tome1.md … tome5.md           ← illustrated indexes, 20 minds each
├── minds/                        ← the runnable architectures (one Neuron per mind)
│   ├── chapter_0001_gilgamesh_-2700.py
│   ├── chapter_0002_zoser_-2670.py
│   └── … (through chapter_0100_liu_an_prince_of_huainan_-179.py)
└── maps/                         ← the visual mind-map explainers (one image per mind)
    ├── chapter_0001_gilgamesh_-2700.jpg
    └── … (through chapter_0100_liu_an_prince_of_huainan_-179.jpg)
```

Every file is prefixed with its **mind number** (`chapter_00NN_…`) so nothing collides as the corpus grows toward 1,141 entries. Each architecture and its explainer image share the same stem.

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
```

Most files accept `--test` (self-tests then exit) and `--quiet` (demo without ASCII plots). Because each architecture is built to embody a *specific* mind, no two behave alike.

---

## 📊 The Artificiology E-AGI Barometer

Every reconstructed mind is measured against the same yardstick — the **[Artificiology E-AGI Barometer](https://artificiology.com/barometer.html)** — so that a Bronze-Age lawgiver and a Hellenistic geometer can be compared on the capabilities their imagined AGI would need. The eight top-level dimensions:

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

Each chapter closes by imagining how its figure would have reasoned about an embodied AGI (an **E-AGI** / humanoid) against these metrics.

---

## 🔬 How the minds are reconstructed

The project is **research-first**. Before any architecture is written, the figure's surviving works and current scholarship are gathered and every source verified — a real corpus is small, but never fabricated. Where evidence is thin, the entry says so plainly rather than inventing an inner life.

Each figure is tagged with a candid **provenance**:

- 🟢 **belief** — the figure's own surviving works or recorded doctrine ground the entry. *(81 of the first 100.)*
- 🟡 **mediated** — no words of their own survive; they are known only through others' (often hostile or legendary) accounts. *(11 of the first 100.)*
- 🔵 **extrapolated** — no philosophy of mind survives at all; the entry is inferred from documented deeds, typical of kings and builders. *(8 of the first 100.)*

And each architecture is required to **embody the mind and to run**: from-scratch pure-NumPy, a passing gradient check, a real training loop and self-tests — deliberately *not* a default Transformer, but a mechanism chosen to encode that thinker's own cognitive fingerprint.

---

## 🗺️ Roadmap

- ✅ **Tomes 1–5** — Minds 1–100 · architectures + visual explainers *(this release)*
- 🔜 Further tomes released here as they open-source, extending toward the full **1,141-mind** corpus (antiquity → 1905).
- 🎛️ The interactive **`MindMap.html`** planes and the long-form chapter texts live in the wider ecosystem — read them at **[lostmindsai.com](https://lostmindsai.com)** and across the **[Amazon book series](https://www.amazon.com/dp/B0H6F9L324)**.

This README is intentionally **global**: it describes the whole Encyclopedia and stays valid as new tomes land in this repository.

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
| 📖 Book series (Amazon) | **[Tome 1](https://www.amazon.com/dp/B0H6F9L324)** · **[Tome 2](https://www.amazon.com/dp/B0H6QCQ9M7)** · **[Tome 3](https://www.amazon.com/dp/B0H6TVX69S)** · **[Tome 4](https://www.amazon.com/dp/B0H71JC95Q)** · **[Tome 5](https://www.amazon.com/dp/B0H7LP5LP2)** |
| 🧪 Résumé & interactive demos | **[artificiology.com](https://artificiology.com/)** |
| 📊 E-AGI Barometer | **[artificiology.com/barometer.html](https://artificiology.com/barometer.html)** |
| ✍️ Author — David Vivancos | **[vivancos.com](https://www.vivancos.com/)** |

---

<div align="center">

*The individual instance dies, but the pattern persists across successors.*
**— the immortality the Epic of Gilgamesh actually endorses, and the wager of this Encyclopedia.**

</div>
