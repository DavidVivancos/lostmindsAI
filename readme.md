<div align="center">

# 🧠 Encyclopedia of Lost Minds: *Echoes on AI* Open Source Code

### How History's Greatest Thinkers Would Have Thought About AGI

*Reconstructing 1,000+ minds from the past — up to 1905 — and asking a single question of each:
**if they were alive today, how would they build an Artificial General Intelligence?***

[🌐 Encyclopedia](https://lostmindsai.com) &nbsp;·&nbsp;
[📖 Book Series (Amazon)](https://www.amazon.com/dp/B0H6F9L324) &nbsp;·&nbsp;
[🧪 Interactive Demos](https://artificiology.com/) &nbsp;·&nbsp;
[📊 E-AGI Barometer](https://artificiology.com/barometer.html) &nbsp;·&nbsp;
[✍️ Author](https://www.vivancos.com/)

![Minds](https://img.shields.io/badge/minds-1%2C141_planned-6C5CE7)
![Tome 1](https://img.shields.io/badge/Tome_1-Minds_1–20_released-00B894)
![Python](https://img.shields.io/badge/python-3.x-3776AB?logo=python&logoColor=white)
![Dependencies](https://img.shields.io/badge/deps-NumPy_only-013243)
![License](https://img.shields.io/badge/license-see_LICENSE-lightgrey)

</div>

---

## What this is

**Lost Minds AI** is a large-scale project that reconstructs the cognition of historical thinkers — poets, kings, physicians, lawgivers, mystics, mathematicians — and projects each of them forward into our era to imagine the AGI *they* would have designed, given how they actually thought about mind, order, knowledge and the self.

Psychohistory turned inward. Where Asimov's psychohistorians forecast the future of the many, this series reconstructs the inner architecture of the singular.

This repository open-sources the **mechanistic plane** — the architectures — alongside a **visual mind-map explainer** for each figure. It will grow, tome by tome, until it holds the full Encyclopedia.

> **As long as possible no fabricated metrics.** Each architecture ships with a finite-difference gradient check, a real training loop and a self-test suite. Every number a program reports is produced live, on the machine that runs it — not hard-coded.

## Why I created this

As an explorer of the human mind and a builder of artificial ones for decades, I have always wondered whether it might be possible to travel back in time and look directly into the great human figures who shaped our present.

For me, going back — doing what I call techno-anthropology — is fundamental to connecting the dots. The past is there for a reason. Some would say it is better to forget it and simply build the future; I believe the opposite, because the past lets us see the paths that clever humans used to conquer their environment and their times — some for good, some arguably not, but here without judging. And this holds across every domain, from the arts to the sciences and everything in between.

As we build AI, AGI, or even E-AGI in its physical form, we sometimes struggle to understand why some things work and others do not. Over almost 30,000 hours of my own life researching the field, I have drawn a great deal of inspiration from the pioneers — often from areas of human knowledge entirely unrelated to computing, because it is there that you can pick up a thread and follow it to a discovery somewhere else.

Today's AIs have the remarkable capacity to synthesize most of the written knowledge humanity has produced since the dawn of the printing press, and earlier still — something I explored in my 2023 book The End of Knowledge. Since then we have begun to acquire tools for looking at the past as never before in human history. This led me to ask whether it might be possible to reconstruct these vanished minds and put to them one simple question:
How would they have built AGI and E-AGI with their own mind and their own model of the world?

That question is the essence of this encyclopedia (here as the open source code of each mind): to discover, in a deliberate mix of science and fiction, parallel universes of thought — grounded as far as possible in what is actually happening in AI research today, so that it may spark new research in the field while also serving the general reader as a way to learn and discover. 

In a sense it is a first, technically feasible approximation to what Isaac Asimov called Psychohistory in his seminal Foundation - but in reverse, only here focused entirely on AI. Connections will emerge, and time will tell how far they lead us.

The work is built in batches of twenty minds per volume, running from the deep past of 2700 BCE to 1905 — a horizon that may yet be extended — and using each figure's birth year as the marker. Ending at the start of the twentieth century keeps the project to minds of the past. At the time of writing the corpus holds 1000+ figures, so roughly fifty-seven tomes are planned; watch for the follow-ups.
---

## 📚 Released so far (in Open Source Code)

### ▶️ [Tome 1 — Minds 1–20 (2700–800 BCE)](tome1.md)

The first twenty minds, from **Gilgamesh** — the first hero to confront mortality as the core problem of a thinking being — to **Homer**, who composed the *Iliad* live under the inviolable gate of the hexameter. Each comes with a runnable architecture and a full-page visual mind-map.

**[→ Open the illustrated Tome 1 index](tome1.md)** for all twenty explainer images of each Neuron Code, one-line theses and run commands.

<div align="center">

| # | Mind | Era | Architecture |
|---|------|-----|--------------|
| 01 | Gilgamesh | c. 2700 BCE | The Composite Self |
| 02 | Zoser (Djoser) | c. 2670 BCE | The Step-Pyramid Network |
| 03 | Imhotep | c. 2650 BCE | The Imhotep Diagnostic Architecture |
| 04 | Ptahhotep | c. 2450 BCE | The Ptahhotep Architecture |
| 05 | Sargon of Akkad | c. 2334 BCE | The Provincial Mixture-of-Experts |
| 06 | Enheduanna | c. 2285 BCE | Temple of the Me |
| 07 | Ur-Nammu | c. 2100 BCE | The Codified Equity Network |
| 08 | Khety II | c. 2000 BCE | The Scribe-Vizier Architecture |
| 09 | Sin-Muballit | c. 1850 BCE | The Sexagesimal–Omen Mind |
| 10 | Hammurabi | r. 1792–1750 BCE | The Stele Network |
| 11 | Hatshepsut | r. 1479–1458 BCE | The Maat-Field Network |
| 12 | Thutmose III | r. 1479–1425 BCE | The Aruna Engine |
| 13 | Akhenaten | r. 1353–1336 BCE | AtenNet |
| 14 | Nefertiti | c. 1370 BCE | The Aten Broadcast Network |
| 15 | Ramses II | r. 1279–1213 BCE | The Ramesside Replication Network |
| 16 | Nefertari | c. 1300 BCE | The Parity Coupler |
| 17 | Amenemope | c. 1300–1100 BCE | The Ger-Maa Cell |
| 18 | Zoroaster | c. 1500–600 BCE | The Dual-Spirit Choice Network |
| 19 | Shalmaneser III | r. 859–824 BCE | The Annalist |
| 20 | Homer | c. 800 BCE | The Metis Engine |

</div>

---

## 🗂️ Repository layout

```
LostMindsAI/
├── readme.md                     ← you are here
├── tome1.md                      ← illustrated index of Minds 1–20
├── minds/                        ← the runnable architectures (Neuron.py per mind)
│   ├── chapter_0001_gilgamesh_-2700.py
│   ├── chapter_0002_zoser_-2670.py
│   └── … (through chapter_0020_homer_-800.py)
└── maps/                         ← the visual mind-map explainers (one image per mind)
    ├── chapter_0001_gilgamesh_-2700.jpg
    └── … (through chapter_0020_homer_-800.jpg)
```

Every file is prefixed with its **mind number** (`chapter_00NN_…`) so nothing collides as the corpus grows toward 1,000+ entries. Code and its explainer image share the same stem.

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
python3 minds/chapter_0020_homer_-800.py --test

# 4. or run the full demo for that mind
python3 minds/chapter_0020_homer_-800.py
```

Most files accept `--test` (self-tests then exit) and `--quiet` (demo without ASCII plots). Because each architecture is built to embody a *specific* mind, no two behave alike: Gilgamesh's network grows wiser through simulated grief and passes an "epic" to a successor instance; Hammurabi's decides each case by analogy to a fixed canon of public precedents; Homer's cannot physically emit a line that breaks the metre.

---

## 📊 The Artificiology E-AGI Barometer

Every reconstructed mind is measured against the same yardstick — the **[Artificiology E-AGI Barometer](https://artificiology.com/barometer.html)** — so that a Bronze-Age lawgiver and a Greek poet can be compared on the capabilities their imagined AGI would need. The eight top-level dimensions:

| | Dimension | Focus |
|---|-----------|-------|
| 🧩 | **Cognitive Processing** | Problem-solving & reasoning · working memory · learning transfer |
| 🤸 | **Embodied Cognition** | Sensory integration · motor control & navigation · real-time sensorimotor adaptation |
| 🌍 | **World Modeling** | Physical & natural laws · social & ecological systems · environmental adaptation |
| 👁️ | **Consciousness** | Metacognition & self-monitoring · subjective experience · mental adaptation |
| 💭 | **Language Understanding** | Comprehension · coherent generation · cross-lingual & cultural adaptation |
| ❤️ | **Emotional Intelligence** | Emotion recognition & response · social perception · empathy & conflict resolution |
| ✨ | **Creativity** | Originality & ideation · artistic & storytelling ability · innovation |
| 🎯 | **Autonomy** | Independent goal-setting · adaptive obstacle management · self-modification |

---

## 🔬 How the minds are reconstructed

The project is **research-first**. Before any architecture is written, the figure's surviving works and current scholarship are gathered and every source verified — a real corpus is small, but never fabricated. Where evidence is thin, the chapter says so plainly rather than inventing an inner life.

Each figure is tagged with a **provenance**:

- 🟢 **belief** — the figure's own surviving works or recorded doctrine ground the entry.
- 🟡 **mediated** — no words of their own survive; they are known only through others' (often hostile or legendary) accounts.
- 🔵 **extrapolated** — no philosophy of mind survives at all; the entry is inferred from documented deeds (typical of kings and builders).

And the architecture is required to **embody the mind and to run**: from-scratch pure-NumPy, a passing gradient check, a real training loop and self-tests — deliberately *not* a default Transformer, but a mechanism chosen to encode that thinker's own cognitive fingerprint.

---

## 🗺️ Roadmap

- ✅ **Tome 1** — Minds 1–20 · architectures + visual explainers *(this release)*
- 🔜 Further tomes released here as they open-source, extending toward the full **1,000+ minds** corpus (antiquity → 1905).
- 🎛️ The interactive **MindMaps** planes and the long-form chapter texts live in the wider ecosystem — read them at **[lostmindsai.com](https://lostmindsai.com)** and in the **[Amazon book series](https://www.amazon.com/dp/B0H6F9L324)**.

This README describes the whole Encyclopedia and will stay valid as new tomes land in this repository.

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
| 📖 Book series (Amazon) | **[amazon.com/dp/B0H6F9L324](https://www.amazon.com/dp/B0H6F9L324)** |
| 🧪 Résumé & interactive demos | **[artificiology.com](https://artificiology.com/)** |
| 📊 E-AGI Barometer | **[artificiology.com/barometer.html](https://artificiology.com/barometer.html)** |
| ✍️ Author — David Vivancos | **[vivancos.com](https://www.vivancos.com/)** |

---

<div align="center">

*The individual instance dies, but the pattern persists across successors.*
**— the immortality the Epic of Gilgamesh actually endorses, and the wager of this Encyclopedia.**

</div>
