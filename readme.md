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
![Released](https://img.shields.io/badge/released-Tomes_1–14_·_Minds_1–280-00B894)
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
| **Minds released** | 280 · Tomes 1–14 · [Gilgamesh](tome1.md) → [Peter Abelard](tome14.md) |
| **Time span covered so far** | c. 2700 BCE → 1079 CE (ordered by birth year) |
| **Runnable architectures** | 280 — one per mind, no two alike, NumPy only |
| **Printed pages** | 7,499 across the fourteen hardcover tomes |
| **Provenance of the first 280** | 🟢 223 belief · 🟡 35 mediated · 🔵 22 extrapolated |
| **Full corpus** | 1,600+ minds from 140 countries, ending in 1905 |
| **Yardstick** | the 8-axis [E-AGI Barometer](#-the-artificiology-e-agi-barometer), 144 live metrics on the platform |

---

## 📚 Released so far — Tomes 1–14 · Minds 1–280

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
| **11** | 201–220 | 712–810 CE | The Warranted Mind — Tang Poets, Abbasid Baghdad, the Carolingian Correctors, Heian Japan & the Science of Transmitters | 📖 [tome11.md](tome11.md) | [Amazon](https://www.amazon.com/dp/B0HKF7TRFF) |
| **12** | 221–240 | 810–915 CE | The Carried Mind — Tang Chan, the Slavonic Mission, the Samanid Court, al-Andalus, Heian Japan & the Norse Sagas | 📖 [tome12.md](tome12.md) | [Amazon](https://www.amazon.com/dp/B0HKVBKNTT) |
| **13** | 241–260 | 917–1004 CE | The Standing Mind — Heian Onmyōdō, Ottonian Saxony, Umayyad Córdoba, Ghaznavid Khurasan, Fatimid Cairo, Song Printing, Buyid Baghdad, Árpád Hungary & Kyivan Rus' | 📖 [tome13.md](tome13.md) | [Amazon](https://www.amazon.com/dp/B0HLD1S6QT) |
| **14** | 261–280 | 1009–1079 CE | The Discerning Mind — Ghaznavid Lahore, Karakhanid Transoxiana, Northern Song China, Chola South India, Reform Rome, Taifa al-Andalus, Norman England, Seljuk Persia, Commonwealth Iceland & Capetian Paris | 📖 [tome14.md](tome14.md) | [Amazon](https://www.amazon.com/dp/B0HLYSVY5D) |

The arc so far runs from **Gilgamesh** — the first hero to confront mortality as the core problem of a thinking being — through **Homer**, **Confucius**, **Socrates**, **Plato**, **Aristotle** and **Archimedes**, to **Liu An** and the resonance-cosmology of the *Huainanzi*; on into Rome, the Han and the Antonine world; through the collapse of the West, from the Cao Wei workshops to the wall at Shaolin; into the sixth century, from **Aryabhata**'s observatory at Kusumapura and **Justinian**'s law commission to the Old North of **Taliesin** and **Aneirin**; into the high noon of the Tang and the first century of Islam — **Sun Simiao**'s thumb on a forearm, **Xuanzang**'s sixteen years on the road, **Wu Zetian**'s minted characters, **Huineng**'s mirror that was never there, **Bede**'s tables at Jarrow, **Li Bai**'s moon and shadow, and **Malik**'s thirty-two answers of *I do not know*; and now into the Abbasid century of the checked claim — **Du Fu**'s couplets written through the rebellion, **al-Manṣūr**'s round city, **Charlemagne**'s inspectors riding in pairs, **Kūkai**'s world that is already speaking, **al-Khwārizmī**'s six forms, **al-Maʾmūn**'s two observatories, and **al-Bukhārī**'s one report admitted in two hundred and thirty; and on into the century of passage, where every claim has to be carried across a border — **Linji**'s shout above the ford, **Cyril**'s distinction in the sounds, **Clement**'s graft at Ohrid, **Michizane**'s return marks, **al-Rāzī**'s second column beside Galen, **Rudaki**'s eleven syllables that brought an amir home, and **al-Mutanabbī**'s measure that meets the magnitude; and on into the century of standing, where the one who acts must be licensed by something it cannot write — **Ferdowsi**'s king who keeps every craft and loses the glory, **al-Zahrāwī**'s ink before the iron, **Ibn al-Haytham**'s grid of errors, **Murasaki**'s poppy seed that would not wash out, **al-Bīrūnī**'s thousand copies counted as one liar, **Atiśa**'s ladder that a broken vow withdraws, and **Nāṣir Khusraw**'s cup put down at forty; and on into the century of discernment, where a mind must learn what kind of thing is in front of it before it acts — **al-Hujwiri**'s polish that tells a clouded mirror from a stone, **al-Sarakhsi**'s cause admitted only by its attested effect, **Sima Guang**'s worst arrow, **Shen Kuo**'s two hundred sightings of one star, **Anselm**'s two appetites, **Omar Khayyam**'s claim that must survive a change of notation, **al-Ghazālī**'s habit that is not necessity, and **Peter Abelard**'s meaning made in the act of attending.

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
| 11 | By what route did a claim reach the mind — and what, outside the claim, can strike it out? *(keyword: warrant)* |
| 12 | What does the passage do to what it carries — and who, other than the carrier, can read the change? *(keyword: passage)* |
| 13 | What entitles the one who acts to act — and what, other than the actor, can read that standing? *(keyword: standing)* |
| 14 | What kind of thing is in front of the one who must act — and what, other than how it looks, can tell? *(keyword: discernment)* |

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

<details>
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

<details>
<summary><b>Tome 11 — a sample</b></summary>

- **Du Fu**'s answers every line with a counterpart operator that must learn to be its own inverse, and keeps a witness ledger it can append to but never edit.
- **al-Manṣūr**'s routes every report through a star with no lateral roads and acts only when two independent channels agree.
- **al-Khalīl**'s scores a root's every anagram at once and keeps a register of which arrangements are empty.
- **Jābir**'s holds every state as a conserved proportion of four natures and computes the hidden face as the inverse of the manifest one.
- **Alcuin**'s pulls every damaged copy toward a small canon through a correcting loop that is a contraction.
- **Charlemagne**'s descends toward learned exemplars along a metric steep for meaning and shallow for ornament.
- **al-Shāfiʿī**'s never throws an attested text away and revises its ruling on a new report without retraining.
- **Saichō**'s freezes its archetypes at birth, trains only the veil, and fails its own tests if any class is left unreachable.
- **Ḥabash**'s folds each problem by an exact rotation and re-enters its parallax table until the correction vanishes.
- **Bai Juyi**'s hands its certificate to a listener too weak to be flattered and scatters its archive across independent sites.
- **Kūkai**'s quantises the world onto an alphabet fixed before training and must speak its reading back out.
- **al-Jāḥiẓ**'s gives its fifth channel no input of its own — only the residue the other four cannot explain.
- **al-Khwārizmī**'s caps its router at six slots and stays dark when a second derivation disagrees.
- **Ibn Ḥanbal**'s stores its transmitters rather than their content and condemns a witness by the direction of its drift.
- **al-Maʾmūn**'s opens its referent gate only when independent instruments agree, and may answer that a dispute cannot be settled.
- **Fatima al-Fihri**'s cuts every principal from its own ground and serves no new ring until its fast is over.
- **al-Bukhārī**'s grades a chain by its weakest joint, counts twenty routes through one man as one, and has no generative head.
</details>

<details>
<summary><b>Tome 12 — a sample</b></summary>

- **Linji**'s falls silent when a removal pulls out the support its answer needed, and has no coordinate for how high it stands.
- **Ibn Firnās**'s starts without a tail and grows one from an audit of its falls.
- **Eriugena**'s feeds its self-model only on returned effects, and answers from its outputs what it cannot answer about itself.
- **Methodius**'s prunes the canons its host cannot bear and commutes every sanction into the host's own currency, keeping the order of offences.
- **Muslim**'s gives each rank of witness a permission, so no volume of corroboration can originate warrant.
- **Cyril**'s counts the distinctions before it assigns meaning, and shows that a distinction the script drops cannot be recovered downstream.
- **Boris**'s puts one inventory of conduct to rival authorities and compares their reasons, not their verdicts.
- **Naum**'s is a choir of coupled oscillators that must hold a band and never collapse into unison.
- **Ibn Duraid**'s keeps its lexicon as a census with a budget and a derivation ledger that is forbidden to collapse.
- **Clement**'s takes content only from the scion and vigour only from the rootstock, joined at a union whose conductance can be read.
- **Michizane**'s reads through a near-permutation it can show you, and abstains on the graph it cannot place.
- **Harald**'s carries its goal in an uncombed public accumulator and lets the districts that cannot bear one law secede.
- **al-Rāzī**'s freezes Galen beside a signed register of failures and forbids its responder to route by its own confidence.
- **Rudaki**'s is read by resonance, clocked in morae, and re-anchored by a radīf at every line-end.
- **al-Fārābī**'s collapses when severed from the intellect outside it, and may not let an image invert the order of what it imitates.
- **ʿAbd al-Raḥmān III**'s broadcasts one vector to every faction and measures its own competence apart from its confidence.
- **al-Hamdānī**'s has no bias term: no vessel can hold what the ore did not contain.
- **al-Masʿūdī**'s indexes every belief by where it was acquired and keeps a ledger of its own amendments.
- **Rābiʿa**'s ties tighten with every pull away and loosen only on slack.
- **al-Mutanabbī**'s never sees a task's absolute size, only its ratio to a measure it must learn from verdicts on its deeds.
</details>

<details>
<summary><b>Tome 13 — a sample</b></summary>

- **Yasunori**'s runs the calendar, the sky and the rite in one accountable head, and keeps one eye that training never touches.
- **Hrotsvitha**'s holds the form fixed and inverts the payload with an exact reflection, and reports what it holds apart from what it says.
- **al-Zahrāwī**'s marks every correction before training and caps its reach short of the nearest protected anchor, exactly zero beyond.
- **Ferdowsi**'s keeps warrant in a witness that remembers longer than the actor, and gives the actor no path to its own licence.
- **Maslama**'s divides in the measure where equality is true, then carries the division onto the plate, the canon's errors and all.
- **Narekatsi**'s confesses a shortcut before it fails, in a lament sealed from the voice and sent to a physician.
- **Ibn al-Haytham**'s measures the conditions of seeing by a path that never sees the answer, and pays for scrutiny only when a faculty is about to fail.
- **Sei Shōnagon**'s stores a category as its members, never their centre, and audits what it left out.
- **Bi Sheng**'s composes each job in a medium that locks for the run and leaves every shared part bit-identical after it.
- **al-Māwardī**'s makes authority the quantity that flows, narrowing at every delegation and conserved down the tree.
- **Murasaki**'s has no name tokens, and trusts the residue an act leaves in the world over the agent's own report.
- **al-Bīrūnī**'s counts a thousand copies of one lie as one liar, gates by physics before counting, and ships its discards.
- **al-Maʿarrī**'s learns harm as voice times felt, and plans on the felt term alone.
- **Stephen**'s hears each foreign-schooled tongue alone, pays each a stipend, and bounds every vote under a native crown.
- **al-Khalīlī**'s corrects each city's partisanship before its word travels, and counts same-city agreement once.
- **Yaroslav**'s closes the search at two hops and fires a payment when the list comes up empty.
- **Ibn Sīnā**'s estimates first, leaps to the middle term, and reaches understanding only through a frozen intellect it did not write.
- **Atiśa**'s admits its swift rungs only while a slowly recovering licence holds, and grows power only from licensed acts.
- **Ibn Ḥazm**'s seals word meaning from every verdict and extends a ruling only by entailment, never by resemblance.
- **Nāṣir Khusraw**'s values each hour by the learnability it adds, read on a sealed judge, and finds the cup worth nothing.
</details>

<details open>
<summary><b>Tome 14 — a sample</b></summary>

- **al-Hujwiri**'s re-standardises every channel and reads the kind of a dimmed input from how its reflection answers the polish, never from confidence.
- **al-Sarakhsi**'s opens a cause's gate only on contrast pairs, keeps it closed by default, and never lets fit to the rulings reach it.
- **Shao Yong**'s is given the calendar and learns one shape of waxing and waning, read at every scale, with no trend term.
- **Ramanuja**'s keeps six private units on one inner controller, and lets only a scheduled grace carry karma past the floor training can reach.
- **Zhou Dunyi**'s reads only small changes that begin near rest, from a still core with no standing lean, and prices commitment by form.
- **Sima Guang**'s reads reach from the best arrows and character only from the worst, and appoints by the highest lower bound.
- **Su Song**'s runs free and corrects against the sky on a fixed beat, with a head that predicts its own next correction.
- **Gregory VII**'s freezes one root judge, opens an appeal that skips every rank, and measures what the antipope loses.
- **Zhang Zai**'s carries the dispersed qi it never observes beside the condensed qi it sees, and conserves the total exactly.
- **Ibn Gabirol**'s peels a thing form by form, a Will choosing each removal, down to one shared matter.
- **William**'s reconciles every report with a kingdom-wide sum, binds consequential nodes directly, and commits through a ratchet.
- **Ibn al-Zarqālluh**'s keeps every vintage of a constant and sets it turning on a small circle, so each is true at its date.
- **Shen Kuo**'s weighs channels by their disagreement and keeps an append-only ledger of every revision outside the trained weights.
- **Anselm**'s gives the will two appetites and a solver of necessary reasons, then severs the justice head to watch what remains.
- **Su Shi**'s fixes one plan before the first step, stops where the content ends, and is trained on the rule, not the likeness.
- **Omar Khayyam**'s freezes its trunk and rewrites the numbers in new bases, to see which facts were in the thing.
- **al-Ghazālī**'s re-tests its own experience, consolidates what never fails, keeps habit plastic, and suspends judgement by a rule it cannot reach.
- **Ari**'s solves every date at once from relative statements and a few anchors, storing statements and never dates.
- **Halevi**'s clones a practice whole and amends a component only where its own model re-derives how the measure bends.
- **Abelard**'s varies understanding by the act of attending, leaves live contradictions open, and holds consent alone responsible.
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

Each tome page in this repository (`tome1.md` … `tome14.md`) gives, for every mind, the explainer image, the architecture's name and provenance, a one-paragraph account of the mechanism, and the command that runs it.

---

## ⚙️ Anatomy of an architecture file

Every file in `minds/` follows the same contract, so you can read any of the 280 the same way:

1. **Header block** — the mind, its dates, the tome and the links back to the Encyclopedia, the book and the platform.
2. **Why this architecture and not another** — the one cognitive idea that is this thinker's alone, with the primary-source passage it comes from, and the mapping from each historical mechanism to each component in the code. This is the part to read first.
3. **The novel cell** — the unit that makes the file unlike the others (a ledger cell, a two-voice unit, a jewel with a conserved power gate, a probing loop, a clarity mask over a frozen substrate…).
4. **Forward and backward pass by hand** — no autograd. Gradients are derived and written out, which is why the next item is mandatory.
5. **Finite-difference gradient check** — every file compares its analytic gradients against central differences before it is allowed to train, and aborts if the discrepancy is above tolerance.
6. **A real training loop** on a task shaped by the mind's own material (a synthetic tribe, a corrupt corpus, a fiqh case bank, a sky of cycles, a realm of provinces).
7. **Self-tests** that turn the doctrinal claims into numerical ones — *power is conserved to machine precision*, *the frozen substrate is bitwise unchanged*, *the decision gradient on every painter parameter is identically zero*, *no parameter scores truth*.
8. **Controls and ablations** — the same design with the doctrine removed (the chart-reader, the one-door mind, the idol, the recollection model), so the reader can see what the idea buys and what it costs.

Runs take seconds to a few minutes on a laptop CPU; the heaviest, such as Tome 12's `0228` (Naum's three trained choirs) and `0230` (Clement's graft experiments), and Tome 13's `0241` (Yasunori's integrator and its untrained eye), `0242` (Hrotsvitha's contrafactum engine) and `0247` (Ibn al-Haytham's battery of conditions), and Tome 14's `0267` (Su Song's three correction regimes), `0269` (Zhang Zai's conserving field and its rivals) and `0275` (Su Shi's seeded runs), take longer. Nothing is downloaded, nothing phones home, and nothing is cached: the numbers you see are the numbers your machine produced.

---

## 🗂️ Repository layout

```
LostMindsAI/
├── readme.md                     ← you are here
├── tome1.md … tome14.md          ← illustrated indexes, 20 minds each
├── minds/                        ← the runnable architectures (one Neuron per mind)
│   ├── chapter_0001_gilgamesh_-2700.py
│   ├── chapter_0002_zoser_-2670.py
│   └── … (through chapter_0280_peter_abelard_1079.py)
└── maps/                         ← the visual mind-map explainers (one image per mind)
    ├── chapter_0001_gilgamesh_-2700.jpg
    └── … (through chapter_0280_peter_abelard_1079.jpg)
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

# 5. Tomes 9–14 run their full self-test suite and demo on a plain invocation
python3 minds/chapter_0280_peter_abelard_1079.py

# 6. some files offer a shorter run
python3 minds/chapter_0190_Fazang_643.py --quick
```

Most files from Tomes 1–8 accept `--test` (self-tests then exit) and `--quiet` (demo without ASCII plots). The Tome 9–14 architectures run their full self-test suite and demo on a plain invocation; a few offer a shorter run (`--quick` or `--fast` — see each file's header; in Tome 10 that is `0181`, `0184`, `0186` and `0190`, and `0190` also takes `--gradcheck-only` and `--seed`; in Tome 11 it is `0209` and `0213`, and `0213` also takes `--epochs`, `--seed`, `--save` and `--resume` — its `--quick` run is a smoke test in which only the gradient check is binding, so run it in full for the eight self-tests; in Tome 12 it is `0224`, `0229`, `0237`, `0239` and `0240` with `--quick` and `0232` with `--fast`, while `0224` also takes `--seed`, `--steps` and `--json`, `0237` takes `--seed`, and `0240` takes `--seed`, `--seeds`, `--json`, `--card`, `--mutant` and `--data`, its `--quick` being one seed with reduced updates and every correctness test; in Tome 13, `0243`, `0245`, `0246`, `0248`, `0249`, `0253`, `0254`, `0258`, `0259` and `0260` share one protocol — `--quick` (one seed and every correctness test), `--seed`, `--seeds`, `--json`, `--card` (print the mind card and exit), `--mutant` and `--data` — and `0250` takes `--quick` and `--seed`, its `--quick` being a smoke test in which the behavioural check that hard cases rise to the court is not binding, so run it in full for that result; in Tome 14, `0261`, `0262`, `0263`, `0265`, `0266`, `0269`, `0270`, `0272`, `0278` and `0279` share the same protocol, and their `--quick` run also checks that it finishes within 20 seconds, a check a slow or busy machine can miss while every correctness test passes). Because each architecture is built to embody a *specific* mind, no two behave alike — read the header before you run it, and expect the output to be as idiosyncratic as the thinker.

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

Each chapter closes by imagining how its figure would have reasoned about an embodied AGI (an **E-AGI** / humanoid) against these metrics — and the minds keep re-reading the instrument. Tome 10 alone has **Kamatari** proving that an agent with no action head can score maximally on Autonomy 🎯, and **Huineng**, **Shankara** and **Qaṭarī** relocating misalignment into the persistent, defended self. Tome 11 bends the eighth axis further still: **al-Kindī** caps Autonomy 🎯 on principle, **al-Muʿtaṣim** measures it as capability over the principal's power to withdraw, **al-Bukhārī** requires self-modification to be a dated event, **Ibn Ḥanbal** scores at its ceiling by defining it as the capacity to hold a verdict against a decree, and **Fatima al-Fihri** adds a ninth dial: absence. Tome 12 turns to the fourth axis: on Consciousness 👁️ **Eriugena**, **al-Rāzī** and **al-Mutanabbī** each hold that a mind cannot read its own measure and must be read through its effects or by witnesses, **Rābiʿa** makes experience necessary for knowing another's worth, **Linji** makes doing nothing a valid act of Autonomy 🎯, and **al-Mutanabbī** alone ranks Autonomy first. Tome 13 reads the instrument for what it leaves out: **Ferdowsi** asks for a ninth column — who holds a system's warrant, and how fast it can be withdrawn — **Atiśa** and **al-Māwardī** note that the axes measure what a thing can do and none asks under what standing it does it, **Ibn al-Haytham** would replace every score with a surface over the conditions it was measured under, **Murasaki** would score each axis twice, as capability and as legibility from outside, and on Consciousness 👁️ **Murasaki**, **Ibn al-Haytham** and **Ibn Sīnā** agree that a system's report on itself is not evidence. Tome 14 takes no axis as given: **Judah Halevi** adds a ninth question to every axis — does the machine know which of its abilities it understands, and does it leave the others alone? — **Zhang Zai** pairs the axes and calls a high score on one member of a pair a mind with an outside, **al-Ghazālī** asks of each whether it is necessary-grade or habit speaking in necessity's voice, **Peter Abelard** refuses to treat them as eight faculties at all, and only **Zhou Dunyi** and **Ari Þorgilsson** print scores, both putting Autonomy 🎯 near the bottom.

---

## 🔬 How the minds are reconstructed

The project is **research-first**. Before any architecture is written, the figure's surviving works and current scholarship are gathered and every source verified — a real corpus is small, but never fabricated. Where evidence is thin, the entry says so plainly rather than inventing an inner life.

Each figure is tagged with a candid **provenance**:

- 🟢 **belief** — the figure's own surviving works or recorded doctrine ground the entry. *(223 of the first 280.)*
- 🟡 **mediated** — no words of their own survive; they are known only through others' (often hostile or legendary) accounts. *(35 of the first 280.)*
- 🔵 **extrapolated** — no philosophy of mind survives at all; the entry is inferred from documented deeds, typical of kings and builders. *(22 of the first 280.)*

Three further rules keep the corpus honest across 1,600 entries:

- **Fight the archetype.** Every mind carries a unique, non-repeating archetype descriptor; before a chapter is written, the nearest already-completed figures in the same domain are identified and deliberately diverged from, in both thesis and mechanism.
- **Find the mind-specific thesis.** The generic frame — *intelligence imposes order on chaos; build an institution, not an oracle* — is refused unless it is genuinely the most specific reading of that person. Each chapter, and its architecture, follows from one idea that is the thinker's alone.
- **Close the loop on data.** Any factual correction made while writing a chapter (dates, domain, civilization, provenance) is written back to the master record the same session, never left living only in prose.

And each architecture is required to **embody the mind and to run**: from-scratch pure-NumPy, a passing gradient check, a real training loop and self-tests — deliberately *not* a default Transformer, but a mechanism chosen to encode that thinker's own cognitive fingerprint.

---

## 🗺️ Roadmap

- ✅ **Tomes 1–14** — Minds 1–280 · architectures + visual explainers *(this release)*
- 🔜 **Tome 15** — Minds 281–300, born c. 1081–1130 CE, seven of them women: what a mind keeps, leases and must take back. Urraca of León, Sanai, Anna Komnene, Li Qingzhao, Avempace, Ahmed Yasawi, Ibn Zuhr, Hildegard of Bingen, Peter Lombard, al-Idrisi, Yennenga, Ibn Tufail, Euphrosyne of Polotsk, Nashwan al-Himyari, Bhaskara II, Khaqani, Eleanor of Aquitaine, Parakramabahu I, Ibn Rushd, Zhu Xi.
- 🔜 Further tomes released here as they open-source, extending toward the full **1,600+ mind** corpus (antiquity → 1905).
- 🎛️ The interactive **`MindMap.html`** planes and the long-form chapter texts live in the wider ecosystem — read them at **[lostmindsai.com](https://lostmindsai.com)** and across the **[Amazon book series](https://www.amazon.com/dp/B0H6F9L324)**.

This README is intentionally **global**: it describes the whole Encyclopedia and stays valid as new tomes land in this repository.

---

## ❓ FAQ

**Are the numbers in the output real?** Yes. Nothing is hard-coded or looked up. Every gradient check, loss curve, ablation and self-test is computed when you run the file. Runs are seeded, so they are reproducible; change the seed and the numbers change.

**Why NumPy only?** So that a strange claim cannot hide inside a library. Each backward pass is written by hand and checked numerically, which keeps every mechanism legible to a reader who wants to see exactly what the doctrine became in code.

**Why not Transformers?** Attention over a store of remembered keys is the default modern mechanism and, for most of these minds, exactly the thing they argued against. Each file explains, in its header, what it uses instead and why that is the faithful choice.

**How long does a run take?** Seconds to a few minutes on a laptop CPU for most files; a few that train several counterfactual minds run longer, and many of those offer `--quick` or `--fast`.

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
| 📖 Book series (Amazon) | **[Tome 1](https://www.amazon.com/dp/B0H6F9L324)** · **[Tome 2](https://www.amazon.com/dp/B0H6QCQ9M7)** · **[Tome 3](https://www.amazon.com/dp/B0H6TVX69S)** · **[Tome 4](https://www.amazon.com/dp/B0H71JC95Q)** · **[Tome 5](https://www.amazon.com/dp/B0H7LP5LP2)** · **[Tome 6](https://www.amazon.com/dp/B0HF7G6JJD)** · **[Tome 7](https://www.amazon.com/dp/B0HFN6GXMH)** · **[Tome 8](https://www.amazon.com/dp/B0HH8RTCXF)** · **[Tome 9](https://www.amazon.com/dp/B0HJC1QF59)** · **[Tome 10](https://www.amazon.com/dp/B0HJYMZ3G6)** · **[Tome 11](https://www.amazon.com/dp/B0HKF7TRFF)** · **[Tome 12](https://www.amazon.com/dp/B0HKVBKNTT)** · **[Tome 13](https://www.amazon.com/dp/B0HLD1S6QT)** · **[Tome 14](https://www.amazon.com/dp/B0HLYSVY5D)** |
| 🧪 Résumé & interactive demos | **[artificiology.com](https://artificiology.com/)** |
| 📊 E-AGI Barometer | **[artificiology.com/barometer.html](https://artificiology.com/barometer.html)** |
| ✍️ Author — David Vivancos | **[vivancos.com](https://www.vivancos.com/)** |

---

<div align="center">

*The individual instance dies, but the pattern persists across successors.*
**— the immortality the Epic of Gilgamesh actually endorses, and the wager of this Encyclopedia.**

</div>
