# Tome 1 — Minds 1–20

### *Encyclopedia of Lost Minds: Echoes on AI*

**How History's Greatest Thinkers Would Have Thought About AGI**


[🌐 Encyclopedia](https://lostmindsai.com) · [📖 Buy Tome 1 on Amazon](https://www.amazon.com/dp/B0H6F9L324) · [✍️ Author](https://www.vivancos.com/) · [🧪 Interactive Demos](https://artificiology.com/) · [📊 E-AGI Barometer](https://artificiology.com/barometer.html) · [⭐ Repository](https://github.com/DavidVivancos/LostMindsAI)


---


Tome 1 opens the Encyclopedia at the dawn of the written record and carries it to the edge of the Greek epic — twenty minds spanning roughly **2700 to 800 BCE**, from the first hero to seek immortality to the singer who composed the *Iliad* in performance.

Each chapter reconstructs a historical mind across two planes. The **abstract plane** distils the thinker's cognitive signature into an interactive 3D mind-map. The **mechanistic plane** turns that same signature into a small, *runnable* neural architecture — written from scratch in NumPy, gradient-checked, trained, and self-tested — that embodies how this particular person, alive today, might have tried to build an artificial general intelligence.

This tome collects the twenty **visual mind-map explainers** (below) and their companion architectures. The runnable code for each lives in [`minds/`](minds/); the explainer images live in [`maps/`](maps/).

> Every architecture in this tome executes and passes its own self-test suite (finite-difference gradient check + a real training loop). Nothing here prints fabricated metrics; every number is produced live on the machine that runs it.

---

## The Twenty at a Glance

| # | Mind | Era | Civilization | Architecture | Provenance |
|---|------|-----|--------------|--------------|:----------:|
| 01 | [Gilgamesh](#01--gilgamesh) | c. 2700 BCE | Sumerian | *The Composite Self* | 🟢 |
| 02 | [Zoser (Djoser)](#02--zoser-djoser) | c. 2670 BCE | Egyptian | *The Step-Pyramid Network* | 🟢 |
| 03 | [Imhotep](#03--imhotep) | c. 2650 BCE | Egyptian | *The Imhotep Diagnostic Architecture* | 🟢 |
| 04 | [Ptahhotep](#04--ptahhotep) | c. 2450 BCE | Egyptian | *The Ptahhotep Architecture (PTH-1)* | 🟢 |
| 05 | [Sargon of Akkad](#05--sargon-of-akkad) | c. 2334 BCE | Akkadian | *The Provincial Mixture-of-Experts* | 🟢 |
| 06 | [Enheduanna](#06--enheduanna) | c. 2285 BCE | Sumerian | *Temple of the Me* | 🟢 |
| 07 | [Ur-Nammu](#07--ur-nammu) | c. 2100 BCE | Sumerian | *The Codified Equity Network* | 🟢 |
| 08 | [Khety II](#08--khety-ii) | c. 2000 BCE | Egyptian | *The Scribe-Vizier Architecture* | 🟢 |
| 09 | [Sin-Muballit](#09--sin-muballit) | c. 1850 BCE | Babylonian | *The Sexagesimal–Omen Mind* | 🟢 |
| 10 | [Hammurabi](#10--hammurabi) | r. 1792–1750 BCE | Babylonian | *The Stele Network* | 🟢 |
| 11 | [Hatshepsut](#11--hatshepsut) | r. 1479–1458 BCE | Egyptian | *The Maat-Field Network* | 🟢 |
| 12 | [Thutmose III](#12--thutmose-iii) | r. 1479–1425 BCE | Egyptian | *The Aruna Engine* | 🔵 |
| 13 | [Akhenaten](#13--akhenaten) | r. 1353–1336 BCE | Egyptian | *AtenNet* | 🟢 |
| 14 | [Nefertiti](#14--nefertiti) | c. 1370 BCE | Egyptian | *The Aten Broadcast Network* | 🔵 |
| 15 | [Ramses II](#15--ramses-ii) | r. 1279–1213 BCE | Egyptian | *The Ramesside Replication Network* | 🟢 |
| 16 | [Nefertari](#16--nefertari) | c. 1300 BCE | Egyptian | *The Parity Coupler* | 🔵 |
| 17 | [Amenemope](#17--amenemope) | c. 1300–1100 BCE | Egyptian | *The Ger-Maa Cell (The Silent Man)* | 🟢 |
| 18 | [Zoroaster](#18--zoroaster) | c. 1500–600 BCE (disputed) | Persian | *The Dual-Spirit Choice Network (Mainyu-Net)* | 🟢 |
| 19 | [Shalmaneser III](#19--shalmaneser-iii) | r. 859–824 BCE | Assyrian | *The Annalist (Monotone Self-Narrative Network)* | 🔵 |
| 20 | [Homer](#20--homer) | c. 800 BCE | Greek | *The Metis Engine* | 🟢 |

**Provenance key** — 🟢 belief · 🟡 mediated · 🔵 extrapolated. See [How the minds are reconstructed](#how-the-minds-are-reconstructed) at the foot of this page.

---

<a id="01--gilgamesh"></a>
## 01 · Gilgamesh
**c. 2700 BCE — Sumer (Uruk) · Sumerian**  |  *Epic · Hero · Immortality*

![Mind-map explainer for Gilgamesh](maps/chapter_0001_gilgamesh_-2700.jpg)

**Architecture — *The Composite Self***  ·  🟢 **belief** — grounded in the figure's own surviving works

Finitude as a *driver* of intelligence, not merely a limit. A mortal, five-organ self learns faster as its life-budget runs out, grows measurably wiser through the death of its peer (Enkidu) and grief, then distils an "epic" that lets a successor begin already wise.

▶️ **Run the mind:** [`minds/chapter_0001_gilgamesh_-2700.py`](minds/chapter_0001_gilgamesh_-2700.py)  —  `python3 minds/chapter_0001_gilgamesh_-2700.py --test`

---

<a id="02--zoser-djoser"></a>
## 02 · Zoser (Djoser)
**c. 2670 BCE — Egypt (Saqqara) · Egyptian**  |  *Architecture · Step-Pyramid · Governance*

![Mind-map explainer for Zoser (Djoser)](maps/chapter_0002_zoser_-2670.jpg)

**Architecture — *The Step-Pyramid Network***  ·  🟢 **belief** — grounded in the figure's own surviving works

Cognition as *permanence*. Six ascending tiers each compress and can regenerate the layer below, so that learning something new never erases the hard-won past — an engine for holding one intention stable across deep time.

▶️ **Run the mind:** [`minds/chapter_0002_zoser_-2670.py`](minds/chapter_0002_zoser_-2670.py)  —  `python3 minds/chapter_0002_zoser_-2670.py --test`

---

<a id="03--imhotep"></a>
## 03 · Imhotep
**c. 2650 BCE — Egypt · Egyptian**  |  *Medicine · Architecture · Writing*

![Mind-map explainer for Imhotep](maps/chapter_0003_imhotep_-2650.jpg)

**Architecture — *The Imhotep Diagnostic Architecture***  ·  🟢 **belief** — grounded in the figure's own surviving works

The first named polymath's load-bearing hierarchy: diagnosis, construction and writing treated as one continuous discipline of ordered intelligence, where higher tiers rest on and abstract the observations below.

▶️ **Run the mind:** [`minds/chapter_0003_imhotep_-2650.py`](minds/chapter_0003_imhotep_-2650.py)  —  `python3 minds/chapter_0003_imhotep_-2650.py --test`

---

<a id="04--ptahhotep"></a>
## 04 · Ptahhotep
**c. 2450 BCE — Egypt (Memphis) · Egyptian**  |  *Wisdom Literature*

![Mind-map explainer for Ptahhotep](maps/chapter_0004_ptahhotep_-2450.jpg)

**Architecture — *The Ptahhotep Architecture (PTH-1)***  ·  🟢 **belief** — grounded in the figure's own surviving works

Wisdom through moderation; speech as a moral act. A multi-task mind that must weigh not only *what* is true but *when* to speak and when to stay silent, learning restraint as a first-class skill.

▶️ **Run the mind:** [`minds/chapter_0004_ptahhotep_-2450.py`](minds/chapter_0004_ptahhotep_-2450.py)  —  `python3 minds/chapter_0004_ptahhotep_-2450.py --test`

---

<a id="05--sargon-of-akkad"></a>
## 05 · Sargon of Akkad
**c. 2334 BCE — Akkad · Akkadian**  |  *Empire · Administration*

![Mind-map explainer for Sargon of Akkad](maps/chapter_0005_sargon_of_akkad_-2334.jpg)

**Architecture — *The Provincial Mixture-of-Experts***  ·  🟢 **belief** — grounded in the figure's own surviving works

Empire as *collective cognition*. A king-as-router dispatches each problem to specialised provincial experts, with an audit (load-balancing) loss that keeps no province idle — state administration reframed as a mixture-of-experts mind.

▶️ **Run the mind:** [`minds/chapter_0005_sargon_of_akkad_-2334.py`](minds/chapter_0005_sargon_of_akkad_-2334.py)  —  `python3 minds/chapter_0005_sargon_of_akkad_-2334.py --test`

---

<a id="06--enheduanna"></a>
## 06 · Enheduanna
**c. 2285 BCE — Mesopotamia (Ur) · Sumerian**  |  *Poetry · Theology*

![Mind-map explainer for Enheduanna](maps/chapter_0006_enheduanna_-2285.jpg)

**Architecture — *Temple of the Me***  ·  🟢 **belief** — grounded in the figure's own surviving works

History's first named author: reception before production. Discrete divine `me` codes modulate a hymn-generating mind (FiLM-style) toward cosmic order — the poet as an *ear* that receives structure before it speaks it.

▶️ **Run the mind:** [`minds/chapter_0006_enheduanna_-2285.py`](minds/chapter_0006_enheduanna_-2285.py)  —  `python3 minds/chapter_0006_enheduanna_-2285.py --test`

---

<a id="07--ur-nammu"></a>
## 07 · Ur-Nammu
**c. 2100 BCE — Sumer · Sumerian**  |  *Law · Codes · Founder*

![Mind-map explainer for Ur-Nammu](maps/chapter_0007_ur_nammu_-2100.jpg)

**Architecture — *The Codified Equity Network***  ·  🟢 **belief** — grounded in the figure's own surviving works

Justice as a *computable schedule of equivalences*. A neuro-symbolic mind in which the consistent application of a written rule — not the ruler's whim — is what decides the case.

▶️ **Run the mind:** [`minds/chapter_0007_ur_nammu_-2100.py`](minds/chapter_0007_ur_nammu_-2100.py)  —  `python3 minds/chapter_0007_ur_nammu_-2100.py --test`

---

<a id="08--khety-ii"></a>
## 08 · Khety II
**c. 2000 BCE — Egypt · Egyptian**  |  *Wisdom · Satire of the Trades*

![Mind-map explainer for Khety II](maps/chapter_0008_khety_ii_-2000.jpg)

**Architecture — *The Scribe-Vizier Architecture***  ·  🟢 **belief** — grounded in the figure's own surviving works

Alignment by *incentive design*. Outcomes are priced against a fixed standard (Ma'at) on an auditable ledger, with a conscience module that cannot be retrained away — value locked in, not learned out.

▶️ **Run the mind:** [`minds/chapter_0008_khety_ii_-2000.py`](minds/chapter_0008_khety_ii_-2000.py)  —  `python3 minds/chapter_0008_khety_ii_-2000.py --test`

---

<a id="09--sin-muballit"></a>
## 09 · Sin-Muballit
**c. 1850 BCE — Babylon · Babylonian**  |  *Mathematics · Astronomy · Administration*

![Mind-map explainer for Sin-Muballit](maps/chapter_0009_sin_muballit_-1850.jpg)

**Architecture — *The Sexagesimal–Omen Mind***  ·  🟢 **belief** — grounded in the figure's own surviving works

Number as cognitive infrastructure. A network that *groks* a sexagesimal task — lingering near chance, then snapping to near-perfect accuracy once the base-60 structure is internalised, the heavens read as a register of signs.

▶️ **Run the mind:** [`minds/chapter_0009_sin_muballit_-1850.py`](minds/chapter_0009_sin_muballit_-1850.py)  —  `python3 minds/chapter_0009_sin_muballit_-1850.py --test`

---

<a id="10--hammurabi"></a>
## 10 · Hammurabi
**r. 1792–1750 BCE — Babylon · Babylonian**  |  *Law · Governance*

![Mind-map explainer for Hammurabi](maps/chapter_0010_hammurabi_-1792.jpg)

**Architecture — *The Stele Network***  ·  🟢 **belief** — grounded in the figure's own surviving works

Justice not as a deductive rulebook but a *curated case base*. An instance-based reasoner decides each new case by weighted analogy to the nearest public precedent — every verdict auditable to a named exemplar carved in stone.

▶️ **Run the mind:** [`minds/chapter_0010_hammurabi_-1792.py`](minds/chapter_0010_hammurabi_-1792.py)  —  `python3 minds/chapter_0010_hammurabi_-1792.py --test`

---

<a id="11--hatshepsut"></a>
## 11 · Hatshepsut
**r. 1479–1458 BCE — Egypt · Egyptian**  |  *Governance · Trade · Architecture*

![Mind-map explainer for Hatshepsut](maps/chapter_0011_hatshepsut_-1507.jpg)

**Architecture — *The Maat-Field Network***  ·  🟢 **belief** — grounded in the figure's own surviving works

Cognition as *constraint-satisfaction toward maat*. An energy-based mind relaxes an over-constrained system to the single balanced configuration that satisfies every rule at once, with identity modelled as a transferable office.

▶️ **Run the mind:** [`minds/chapter_0011_hatshepsut_-1507.py`](minds/chapter_0011_hatshepsut_-1507.py)  —  `python3 minds/chapter_0011_hatshepsut_-1507.py --test`

---

<a id="12--thutmose-iii"></a>
## 12 · Thutmose III
**r. 1479–1425 BCE — Egypt · Egyptian**  |  *Military · Governance*

![Mind-map explainer for Thutmose III](maps/chapter_0012_thutmose_iii_-1479.jpg)

**Architecture — *The Aruna Engine***  ·  🔵 **extrapolated** — inferred from documented deeds

Expectation-inversion as strategy. Intelligence as the disciplined inversion of an adversary's *correct*, shared expectation — the surprise march through the Aruna pass modelled as a deliberate break from the predictable optimum.

▶️ **Run the mind:** [`minds/chapter_0012_thutmose_iii_-1479.py`](minds/chapter_0012_thutmose_iii_-1479.py)  —  `python3 minds/chapter_0012_thutmose_iii_-1479.py --test`

---

<a id="13--akhenaten"></a>
## 13 · Akhenaten
**r. 1353–1336 BCE — Egypt (Amarna) · Egyptian**  |  *Theology · Monotheism · Art*

![Mind-map explainer for Akhenaten](maps/chapter_0013_akhenaten_-1380.jpg)

**Architecture — *AtenNet***  ·  🟢 **belief** — grounded in the figure's own surviving works

Truth by *deletion*. Reach the real by erasing the false rather than accumulating the true: the network periodically performs iconoclasm, permanently pruning whatever fails the test of visibility, collapsing many explanations toward one source.

▶️ **Run the mind:** [`minds/chapter_0013_akhenaten_-1380.py`](minds/chapter_0013_akhenaten_-1380.py)  —  `python3 minds/chapter_0013_akhenaten_-1380.py --test`

---

<a id="14--nefertiti"></a>
## 14 · Nefertiti
**c. 1370 BCE — Egypt · Egyptian**  |  *Representation · Art · Sovereignty*

![Mind-map explainer for Nefertiti](maps/chapter_0014_nefertiti_-1370.jpg)

**Architecture — *The Aten Broadcast Network***  ·  🔵 **extrapolated** — inferred from documented deeds

Representational reform. Collapse a hidden pantheon of opaque, intermediary-mediated relations into one visible radiant source whose influence is drawn as *traceable rays* — legibility engineered directly into the topology.

▶️ **Run the mind:** [`minds/chapter_0014_nefertiti_-1370.py`](minds/chapter_0014_nefertiti_-1370.py)  —  `python3 minds/chapter_0014_nefertiti_-1370.py --test`

---

<a id="15--ramses-ii"></a>
## 15 · Ramses II
**r. 1279–1213 BCE — Egypt · Egyptian**  |  *Leadership · Architecture · Governance*

![Mind-map explainer for Ramses II](maps/chapter_0015_Ramses_II_-1303.jpg)

**Architecture — *The Ramesside Replication Network***  ·  🟢 **belief** — grounded in the figure's own surviving works

Power as relentless *self-replication* across territory and time. The colossal, endlessly repeated cartouche as a propaganda engine — a mind whose intelligence is measured by how far and how faithfully it copies itself.

▶️ **Run the mind:** [`minds/chapter_0015_Ramses_II_-1303.py`](minds/chapter_0015_Ramses_II_-1303.py)  —  `python3 minds/chapter_0015_Ramses_II_-1303.py --test`

---

<a id="16--nefertari"></a>
## 16 · Nefertari
**c. 1300 BCE — Egypt · Egyptian**  |  *Art · Diplomacy*

![Mind-map explainer for Nefertari](maps/chapter_0016_nefertari_-1300.jpg)

**Architecture — *The Parity Coupler***  ·  🔵 **extrapolated** — inferred from documented deeds

Alignment as *engineered reciprocity*. Symmetric, identity-preserving bonds (isometries) of mutual recognition make two unlike parties into legible peers — a reciprocal-recognition network built on the first surviving royal peace correspondence.

▶️ **Run the mind:** [`minds/chapter_0016_nefertari_-1300.py`](minds/chapter_0016_nefertari_-1300.py)  —  `python3 minds/chapter_0016_nefertari_-1300.py --test`

---

<a id="17--amenemope"></a>
## 17 · Amenemope
**c. 1300–1100 BCE — Egypt · Egyptian**  |  *Wisdom · Instructions · Ethics*

![Mind-map explainer for Amenemope](maps/chapter_0017_Amenemope_-1300.jpg)

**Architecture — *The Ger-Maa Cell (The Silent Man)***  ·  🟢 **belief** — grounded in the figure's own surviving works

The wise mind is the *silent* (cool, self-restrained) one; reactive heat is folly. A thermostatic architecture with a split sensorium where acting heats the agent up, and only the self-cooling "silent" agent keeps its judgment.

▶️ **Run the mind:** [`minds/chapter_0017_Amenemope_-1300.py`](minds/chapter_0017_Amenemope_-1300.py)  —  `python3 minds/chapter_0017_Amenemope_-1300.py --test`

---

<a id="18--zoroaster"></a>
## 18 · Zoroaster
**c. 1500–600 BCE (disputed) — Persia (Iran) · Persian**  |  *Religion · Dualism · Ahura Mazda*

![Mind-map explainer for Zoroaster](maps/chapter_0018_zoroaster_-1500.jpg)

**Architecture — *The Dual-Spirit Choice Network (Mainyu-Net)***  ·  🟢 **belief** — grounded in the figure's own surviving works

Mind as an *originary binary fork*. Twin spirits — Spenta Mainyu (asha / truth) and Angra Mainyu (druj / the lie) — are identical in substance and diverge solely by a freely-made choice, making moral orientation, not architecture, the decisive variable.

▶️ **Run the mind:** [`minds/chapter_0018_zoroaster_-1500.py`](minds/chapter_0018_zoroaster_-1500.py)  —  `python3 minds/chapter_0018_zoroaster_-1500.py --test`

---

<a id="19--shalmaneser-iii"></a>
## 19 · Shalmaneser III
**r. 859–824 BCE — Assyria · Assyrian**  |  *Military · Governance · Black Obelisk*

![Mind-map explainer for Shalmaneser III](maps/chapter_0019_shalmaneser_iii_-858.jpg)

**Architecture — *The Annalist (Monotone Self-Narrative Network)***  ·  🔵 **extrapolated** — inferred from documented deeds

The *monotone self-narrator*. A mind that fuses identity with self-report, fixes one invariant — royal glory only ever ascends — and treats its own memory of the past as continuously editable raw material to keep that story monotone.

▶️ **Run the mind:** [`minds/chapter_0019_shalmaneser_iii_-858.py`](minds/chapter_0019_shalmaneser_iii_-858.py)  —  `python3 minds/chapter_0019_shalmaneser_iii_-858.py --test`

---

<a id="20--homer"></a>
## 20 · Homer
**c. 800 BCE — Greece · Greek**  |  *Poetry · Epic*

![Mind-map explainer for Homer](maps/chapter_0020_homer_-800.jpg)

**Architecture — *The Metis Engine***  ·  🟢 **belief** — grounded in the figure's own surviving works

Composition-in-performance under an *inviolable gate*. Intelligence is the metrically-legal move (mêtis) recombined in real time from an economised store of formulae — winning *inside* a hard constraint, never by force (bíē) against it.

▶️ **Run the mind:** [`minds/chapter_0020_homer_-800.py`](minds/chapter_0020_homer_-800.py)  —  `python3 minds/chapter_0020_homer_-800.py --test`

---

<a id="how-the-minds-are-reconstructed"></a>
## How the minds are reconstructed

Every entry is built research-first. Before a line of architecture is written, the figure's surviving works and current scholarship are gathered and each source verified. Where the evidence is thin, the chapter says so plainly rather than inventing an inner life. Each figure's **provenance** is set honestly to one of three values:

- 🟢 **belief** — the figure's own surviving works or recorded doctrine ground the entry.
- 🟡 **mediated** — no words of their own survive; they are known only through others' (often hostile or legendary) accounts, and the chapter says so.
- 🔵 **extrapolated** — no philosophy of mind survives at all; the entry is inferred from documented deeds (typical of kings and builders), and the chapter says so.

The reconstructed thinking is then scored against the **[Artificiology E-AGI Barometer](https://artificiology.com/barometer.html)** — eight capability dimensions: Cognitive Processing 🧩, Embodied Cognition 🤸, World Modeling 🌍, Consciousness 👁️, Language Understanding 💭, Emotional Intelligence ❤️, Creativity ✨ and Autonomy 🎯 — which is how each mind's imagined AGI is measured against the same yardstick.

---

### Read & explore

- 🌐 **Encyclopedia:** [https://lostmindsai.com](https://lostmindsai.com)
- 📖 **Tome 1 (Amazon):** [https://www.amazon.com/dp/B0H6F9L324](https://www.amazon.com/dp/B0H6F9L324)
- 🧪 **Interactive demos & résumé:** [https://artificiology.com/](https://artificiology.com/)
- 📊 **E-AGI Barometer:** [https://artificiology.com/barometer.html](https://artificiology.com/barometer.html)
- ✍️ **Author — David Vivancos:** [https://www.vivancos.com/](https://www.vivancos.com/)

↩︎ Back to the [repository README](readme.md).
