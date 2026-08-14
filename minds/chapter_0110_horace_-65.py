#!/usr/bin/env python3
"""
Chapter 110: Horace (65-8 BCE) - Poetic and Lyric Cognition Architecture
Figure ID: 110 | Domain: poetry, lyric | Region: Rome
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 6 Minds 101 - 120 Available on Amazon https://www.amazon.com/dp/B0HF7G6JJD
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 110: Horace (-65 to -8 BCE)
================================================================================
Horace's Ars Poetica and Odes established frameworks for understanding poetic
composition, audience response, and the craft of verse. His doctrine of
"decorium" (appropriateness) and the " utile et dulce" (useful and sweet)
principle shaped Western literary theory for two millennia.

This architecture models:
- The creative process of lyric composition
- Meter and prosodic cognition
- Audience reception and aesthetic judgment
- The "summer pond" theory of poetic unity
- Self-correction and revision in craft
"""

from __future__ import annotations

import random
import re
import string
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    TypeVar,
)


# =============================================================================
# ENUMS: Poetic Forms, Meters, and Aesthetic States
# =============================================================================


class MeterType(Enum):
    """Ancient Greek/Roman metrical systems Horace would have mastered."""
    ALCAIC = auto()      # Alcaic stanza (4 lines: ~-v-|~-v-|~-v-|-v--)
    SAPPHIC = auto()     # Sapphic stanza (3 long + 1 short Adonic)
    HEXAMETER = auto()   # Dactylic hexameter (epic verse)
    IAMBIC = auto()      # Iambic trimeter (satire, dialogue)
    GLYCONIC = auto()    # Lyric meter for personal ode
    ASCELEPIADEAN = auto() # Choral lyric meter


class PoeticMode(Enum):
    """Modes of poetic expression in Horace's system."""
    LYRIC = auto()       # Personal emotion, song, ode
    EPISTOLARY = auto()  # Letter, conversational verse
    SATIRIC = auto()     # Mocking social commentary
    EPIC = auto()        # Grand narrative, elevated diction
    HYMNIC = auto()      # Religious praise, gods and heroes


class AestheticQuality(Enum):
    """Qualities Horace evaluates in poetry."""
    APTUM = auto()       # Appropriateness, fitness
    VERISIMILITUDO = auto() # Verisimilitude, believability
    DIVINUS = auto()     # Divine inspiration, sublimity
    DULCE = auto()       # Sweetness, pleasure
    UTILE = auto()       # Usefulness, moral instruction
    MEDIOCRITAS = auto() # The golden mean, moderation


class EmotionalTone(Enum):
    """Emotional registers available to the lyric poet."""
    JOY = auto()
    MELANCHOLY = auto()
    PHILOSOPHICAL = auto()
    CELEBRATORY = auto()
    MOURNFUL = auto()
    IRONIC = auto()
    CONTEMPLATIVE = auto()
    URGENT = auto()


class RevisionPhase(Enum):
    """Stages of the Horatian revision process."""
    INCIPIT = auto()     # First draft, raw emotion
    CRAFTSMANSHIP = auto() # Metrical refinement
    DECORUM_CHECK = auto() # Fitness and propriety
    POLISH = auto()      # Final polish, word choice


class AudienceClass(Enum):
    """Horace's conceived audience types."""
    PATRICIAN = auto()   # Roman aristocrats, educated
    EQUITES = auto()     # Business class, practical
    PLEBIAN = auto()      # Common people
    POETIC_CIRCLE = auto() # fellow poets, critics


# =============================================================================
# PROTOCOLS: Behavioral Contracts
# =============================================================================


class PoeticInspirationHandler(Protocol):
    """Protocol for objects that can generate poetic content."""
    def generate_line(self, meter: MeterType, tone: EmotionalTone) -> str: ...
    def evaluate_line(self, line: str, meter: MeterType) -> float: ...


class AudienceResponseModel(Protocol):
    """Protocol for modeling audience reactions."""
    def predict_delight(self, text: str, audience: AudienceClass) -> float: ...
    def predict_instruction(self, text: str, audience: AudienceClass) -> float: ...


# =============================================================================
# GENERIC TYPE VARIABLES
# =============================================================================

T = TypeVar('T')
U = TypeVar('U')


# =============================================================================
# DATACLASSES: Core Poetic Entities
# =============================================================================


@dataclass(frozen=True)
class VerseLine:
    """A single line of verse with metrical and semantic properties."""
    text: str
    meter: MeterType
    syllable_count: int
    ictus_positions: Tuple[int, ...]  # Stress positions
    semantic_field: str  # Dominant meaning cluster
    emotional_charge: float  # -1.0 to 1.0
    is_aptum_satisfying: bool = True

    def __post_init__(self):
        assert self.syllable_count > 0, "Line must have syllables"
        assert -1.0 <= self.emotional_charge <= 1.0


@dataclass
class Stanza:
    """A stanza of verse, typically 4 lines in Horatian forms."""
    lines: List[VerseLine] = field(default_factory=list)
    stanza_form: str = "unknown"
    unity_score: float = 0.0  # "Summer pond" coherence

    def add_line(self, line: VerseLine) -> None:
        self.lines.append(line)

    def compute_unity(self) -> float:
        """Compute thematic unity - summer pond effect."""
        if len(self.lines) < 2:
            return 1.0
        fields = [l.semantic_field for l in self.lines]
        # Simple cohesion: do semantic fields overlap?
        base = fields[0]
        matches = sum(1 for f in fields[1:] if f == base)
        self.unity_score = matches / (len(fields) - 1)
        return self.unity_score


@dataclass
class Ode:
    """A complete Horatian ode."""
    title: str
    stanzas: List[Stanza] = field(default_factory=list)
    meter: MeterType = MeterType.ALCAIC
    mode: PoeticMode = PoeticMode.LYRIC
    dominant_tone: EmotionalTone = EmotionalTone.CONTEMPLATIVE
    dulce_score: float = 0.0  # Aesthetic pleasure
    utile_score: float = 0.0  # Moral usefulness
    aptum_score: float = 0.0  # Appropriateness

    def total_lines(self) -> int:
        return sum(len(s.lines) for s in self.stanzas)

    def compute_balance(self) -> float:
        """Compute dulce/utile balance - Horatian ideal."""
        if self.dulce_score + self.utile_score == 0:
            return 0.0
        ratio = self.dulce_score / (self.dulce_score + self.utile_score)
        # Ideal is around 0.5-0.6 (slightly more dulce than utile)
        return 1.0 - abs(ratio - 0.55) * 2


@dataclass
class PoeticImage:
    """A vivid image within the poetic tradition."""
    description: str
    classical_precedent: str  # Homer, Sappho, etc.
    emotional_resonance: float
    novelty: float  # 0 = conventional, 1 = original
    aptum_fitness: float  # Appropriateness to context


@dataclass
class AudienceProfile:
    """Profile of Horace's intended audience."""
    audience_type: AudienceClass
    education_level: float  # 0.0 to 1.0
    emotional_sophistication: float
    values: Set[str] = field(default_factory=set)
    preferred_meters: Set[MeterType] = field(default_factory=set)

    def will_appreciate(self, ode: Ode) -> float:
        """Predict appreciation score for an ode."""
        base = self.education_level * 0.4 + self.emotional_sophistication * 0.4
        meter_match = 0.2 if ode.meter in self.preferred_meters else 0.05
        return min(1.0, base + meter_match)


# =============================================================================
# COMPONENT CLASSES: Architecture Components
# =============================================================================


class SyllableCounter:
    """Count syllables in Latin/Roman-style verse."""

    VOWELS = set('aeiouAEIOU')
    DIPHTHONGS = {'ae', 'au', 'ei', 'eu', 'oi', 'ou'}

    @classmethod
    def count(cls, word: str) -> int:
        """Count syllables in a word using Latin prosodic rules."""
        count = 0
        i = 0
        while i < len(word):
            # Check for diphthongs
            if i < len(word) - 1:
                digram = word[i:i+2].lower()
                if digram in cls.DIPHTHONGS:
                    count += 1
                    i += 2
                    continue
            if word[i] in cls.VOWELS:
                count += 1
            i += 1
        return max(1, count)


class MetricalPatternMatcher:
    """Match verse lines to metrical patterns."""

    # Simplified metrical templates (syllable counts per foot)
    TEMPLATES: Dict[MeterType, List[int]] = {
        MeterType.ALCAIC: [6, 6, 5, 5],  # Rough approximation
        MeterType.SAPPHIC: [5, 5, 5, 5],  # Adonic line
        MeterType.HEXAMETER: [6, 6, 6, 6, 6, 6],
        MeterType.IAMBIC: [6, 6, 6],
        MeterType.GLYCONIC: [8, 4],
        MeterType.ASCELEPIADEAN: [8, 8, 8],
    }

    @classmethod
    def analyze_fit(cls, line: VerseLine, meter: MeterType) -> float:
        """How well does a line fit the meter? Returns 0.0-1.0."""
        template = cls.TEMPLATES.get(meter, [])
        if not template:
            return 0.0
        # Very simplified: just check syllable count proximity
        total_expected = sum(template)
        ratio = min(line.syllable_count, total_expected) / max(line.syllable_count, total_expected)
        return ratio

    @classmethod
    def generate_pattern(cls, meter: MeterType) -> str:
        """Generate a metrical pattern description."""
        template = cls.TEMPLATES.get(meter, [])
        symbols = {6: '−−−−−−', 5: '−−−−−', 8: '−−−−−−−−', 4: '−−−−'}
        return ' | '.join(symbols.get(s, '?') for s in template)


class ImagePool:
    """Pool of classical images available to the poet."""

    def __init__(self):
        self.images: List[PoeticImage] = []
        self._initialize_classical_pool()

    def _initialize_classical_pool(self) -> None:
        """Initialize with classical precedents from Homer, Sappho, etc."""
        classical_images = [
            PoeticImage("Dawn's rosied fingers", "Homer", 0.8, 0.3, 0.9),
            PoeticImage("The fox's cunning", "Aesop", 0.4, 0.4, 0.7),
            PoeticImage("Wine-dark sea", "Homer", 0.7, 0.2, 0.9),
            PoeticImage("Ship tossed in storm", "Greek lyric", 0.6, 0.3, 0.8),
            PoeticImage("Wax wings melting", "Icarus myth", 0.7, 0.6, 0.8),
            PoeticImage("Golden mean arrow", "Horace's own", 0.5, 0.5, 0.95),
            PoeticImage("Sweet honey", "Epicurean", 0.9, 0.2, 0.9),
            PoeticImage("Fickle Fortune", "Hellenistic", 0.5, 0.5, 0.9),
            PoeticImage("Autumn fruit", "Greek lyric", 0.8, 0.4, 0.85),
            PoeticImage("Pastoral sheep", "Theocritus", 0.7, 0.3, 0.8),
            PoeticImage("Arrow of Cupid", "Sappho", 0.9, 0.3, 0.85),
            PoeticImage("The just man", "Stoic tradition", 0.6, 0.5, 0.9),
        ]
        self.images.extend(classical_images)

    def select_image(
        self,
        tone: EmotionalTone,
        novelty_desired: float = 0.5,
        classical_preference: Optional[str] = None
    ) -> Optional[PoeticImage]:
        """Select an image appropriate to the emotional tone."""
        candidates = self.images
        if classical_preference:
            candidates = [i for i in candidates if i.classical_precedent == classical_preference]
        if not candidates:
            return None
        # Score based on emotional resonance and novelty
        scored = [(img, img.emotional_resonance * 0.6 + img.novelty * novelty_desired * 0.4)
                  for img in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0] if scored else None


class DecorumEnforcer:
    """Enforce the Horatian principle of aptum (appropriateness)."""

    TONE_METER_MAP: Dict[EmotionalTone, Set[MeterType]] = {
        EmotionalTone.JOY: {MeterType.SAPPHIC, MeterType.ALCAIC},
        EmotionalTone.MELANCHOLY: {MeterType.GLYCONIC, MeterType.ALCAIC},
        EmotionalTone.PHILOSOPHICAL: {MeterType.IAMBIC, MeterType.HEXAMETER},
        EmotionalTone.CELEBRATORY: {MeterType.SAPPHIC, MeterType.ASCELEPIADEAN},
        EmotionalTone.MOURNFUL: {MeterType.GLYCONIC},
        EmotionalTone.IRONIC: {MeterType.IAMBIC},
        EmotionalTone.CONTEMPLATIVE: {MeterType.ALCAIC, MeterType.HEXAMETER},
        EmotionalTone.URGENT: {MeterType.IAMBIC},
    }

    TONE_DICTION_MAP: Dict[EmotionalTone, Set[str]] = {
        EmotionalTone.JOY: {"laetus", "gaudens", "dulcis", " aureus"},
        EmotionalTone.MELANCHOLY: {"singultus", "flere", "lacrimae", "nox"},
        EmotionalTone.PHILOSOPHICAL: {"virtus", "ratio", "sapientia", "parcere"},
        EmotionalTone.CELEBRATORY: {"cantare", "pocula", "lecti", "coronae"},
        EmotionalTone.MOURNFUL: {"funere", "umbra", "styx", "plangere"},
        EmotionalTone.IRONIC: {"ridere", "stultus", "fucum", "fallere"},
        EmotionalTone.CONTEMPLATIVE: {"aspice", "quid", "quoque", "sic"},
        EmotionalTone.URGENT: {"iam", "nunc", "mox", "festina"},
    }

    @classmethod
    def check_aptum(cls, ode: Ode) -> float:
        """Return aptum score from 0.0 to 1.0."""
        appropriate_meters = cls.TONE_METER_MAP.get(ode.dominant_tone, set())
        meter_fit = 1.0 if ode.meter in appropriate_meters else 0.4
        # Check diction consistency
        diction_fit = 0.8  # simplified
        return (meter_fit * 0.6 + diction_fit * 0.4)


class RevisionEngine:
    """Horace's self-revision process - the 'se ipsum mordendo'."""

    def __init__(self):
        self.phase = RevisionPhase.INCIPIT
        self.iteration = 0

    def revise_line(self, line: VerseLine, phase: RevisionPhase) -> VerseLine:
        """Apply revision based on phase."""
        self.iteration += 1
        self.phase = phase

        if phase == RevisionPhase.INCIPIT:
            # Raw first draft - return as-is
            return line

        elif phase == RevisionPhase.CRAFTSMANSHIP:
            # Metrical refinement - ensure syllables count
            base_syllables = line.syllable_count
            adjustment = random.choice([-1, 0, 0, 1])  # Slight variations
            return VerseLine(
                text=line.text,
                meter=line.meter,
                syllable_count=base_syllables + adjustment,
                ictus_positions=line.ictus_positions,
                semantic_field=line.semantic_field,
                emotional_charge=line.emotional_charge,
                is_aptum_satisfying=True,
            )

        elif phase == RevisionPhase.DECORUM_CHECK:
            # Check aptum
            aptum = random.random() > 0.2  # 80% pass rate
            return VerseLine(
                text=line.text,
                meter=line.meter,
                syllable_count=line.syllable_count,
                ictus_positions=line.ictus_positions,
                semantic_field=line.semantic_field,
                emotional_charge=line.emotional_charge,
                is_aptum_satisfying=aptum,
            )

        else:  # POLISH
            # Final word-level polish
            words = line.text.split()
            if len(words) > 3:
                # Swap word order occasionally
                if random.random() > 0.7:
                    words[1], words[2] = words[2], words[1]
            return VerseLine(
                text=' '.join(words),
                meter=line.meter,
                syllable_count=line.syllable_count,
                ictus_positions=line.ictus_positions,
                semantic_field=line.semantic_field,
                emotional_charge=line.emotional_charge,
                is_aptum_satisfying=True,
            )

    def full_revision_pass(self, ode: Ode) -> Ode:
        """Perform a full revision cycle on an ode."""
        phases = [
            RevisionPhase.CRAFTSMANSHIP,
            RevisionPhase.DECORUM_CHECK,
            RevisionPhase.POLISH,
        ]
        for phase in phases:
            for stanza in ode.stanzas:
                stanza.lines = [
                    self.revise_line(line, phase) for line in stanza.lines
                ]
        return ode


class AudienceResponsePredictor:
    """Predict how different audience classes respond to poetry."""

    def __init__(self):
        self.audiences: Dict[AudienceClass, AudienceProfile] = {
            AudienceClass.PATRICIAN: AudienceProfile(
                audience_type=AudienceClass.PATRICIAN,
                education_level=0.95,
                emotional_sophistication=0.9,
                values={'virtus', 'dignitas', 'auctoritas'},
                preferred_meters={MeterType.ALCAIC, MeterType.HEXAMETER},
            ),
            AudienceClass.EQUITES: AudienceProfile(
                audience_type=AudienceClass.EQUITES,
                education_level=0.6,
                emotional_sophistication=0.7,
                values={'utilitas', 'fides', 'prosperitas'},
                preferred_meters={MeterType.IAMBIC, MeterType.GLYCONIC},
            ),
            AudienceClass.PLEBIAN: AudienceProfile(
                audience_type=AudienceClass.PLEBIAN,
                education_level=0.3,
                emotional_sophistication=0.5,
                values={'panem', 'circenses', 'libertas'},
                preferred_meters={MeterType.IAMBIC},
            ),
            AudienceClass.POETIC_CIRCLE: AudienceProfile(
                audience_type=AudienceClass.POETIC_CIRCLE,
                education_level=1.0,
                emotional_sophistication=1.0,
                values={'ars', 'ingenium', 'novitas'},
                preferred_meters={MeterType.SAPPHIC, MeterType.ALCAIC, MeterType.ASCELEPIADEAN},
            ),
        }

    def predict_delight(self, ode: Ode, audience: AudienceClass) -> float:
        """How much will this audience enjoy the ode?"""
        profile = self.audiences.get(audience)
        if not profile:
            return 0.5
        base = profile.will_appreciate(ode)
        dulce_bonus = ode.dulce_score * 0.3
        aptum_bonus = ode.aptum_score * 0.2
        return min(1.0, base + dulce_bonus + aptum_bonus)

    def predict_instruction(self, ode: Ode, audience: AudienceClass) -> float:
        """How much moral instruction does this audience get?"""
        profile = self.audiences.get(audience)
        if not profile:
            return 0.3
        utile_weight = ode.utile_score * 0.4
        education_factor = profile.education_level * 0.3
        return min(1.0, utile_weight + education_factor)

    def predict_total_response(self, ode: Ode) -> Dict[AudienceClass, float]:
        """Predict total response across all audiences."""
        results = {}
        for audience in AudienceClass:
            delight = self.predict_delight(ode, audience)
            instruction = self.predict_instruction(ode, audience)
            # Horatian ideal: utile et dulce combined
            results[audience] = (delight * 0.6 + instruction * 0.4)
        return results


class ArsPoeticaPrinciples:
    """Encoding of Horace's Ars Poetica rules for poetic composition."""

    RULES = [
        ("Nature first", "Follow nature in all things"),
        ("Decorum", "Adapt style to subject matter"),
        ("Unity", "Let the whole cohere like a summer pond"),
        ("Fitness", "Let each part be appropriate to its function"),
        ("Clarity", "Avoid obscurity - the audience must understand"),
        ("Delight and instruct", "Combine pleasure with moral benefit"),
        ("Ideal types", "Prefer universal truths to mere particulars"),
        ("Beginning in medias res", "Start the story where the action begins"),
        ("Avoid mixing styles", "Keep tragic and comic modes separate"),
        ("The golden mean", "Avoid excess in any direction"),
    ]

    @classmethod
    def get_rule(cls, index: int) -> Tuple[str, str]:
        """Get a specific rule by index."""
        if 0 <= index < len(cls.RULES):
            return cls.RULES[index]
        return ("Unknown rule", "")

    @classmethod
    def check_ode_against_rules(cls, ode: Ode) -> Dict[str, float]:
        """Check an ode against all Ars Poetica rules."""
        scores = {}
        for i, (name, _) in enumerate(cls.RULES):
            # Simplified scoring - in reality would use NLP
            scores[name] = random.uniform(0.6, 1.0)
        # Apply specific known issues
        if ode.total_lines() < 8:
            scores["Unity"] *= 0.8
        if ode.aptum_score < 0.5:
            scores["Decorum"] *= 0.7
        return scores


# =============================================================================
# MAIN COGNITIVE ARCHITECTURE
# =============================================================================


class HoratianPoeticCognition:
    """
    Complete cognitive architecture for Horatian poetic composition.

    Models:
    1. Inspiration capture (raw emotion → poetic image)
    2. Metrical framing (choosing the right meter for the occasion)
    3. Composition (building stanza by stanza)
    4. Revision (self-correction: se ipsum mordendo)
    5. Audience response (predicting how Maecenas et al. will react)
    6. Decorum enforcement (aptum across all levels)
    """

    def __init__(self):
        self.image_pool = ImagePool()
        self.revision_engine = RevisionEngine()
        self.response_predictor = AudienceResponsePredictor()
        self.decorum_enforcer = DecorumEnforcer()
        self.current_ode: Optional[Ode] = None
        self.composition_log: List[str] = []

    def receive_inspiration(
        self,
        emotion: EmotionalTone,
        occasion: str,
        addressee: str
    ) -> str:
        """Phase 1: Receive raw inspiration and frame it poetically."""
        img = self.image_pool.select_image(emotion)
        image_desc = img.description if img else "generic poetic image"

        inspiration = (
            f"When {occasion}, my heart turns to {addressee} "
            f"like {image_desc.lower()}. "
            f"The {emotion.name.lower()} swells within."
        )
        self.composition_log.append(f"[INSPIRATION] {inspiration}")
        return inspiration

    def compose_ode(
        self,
        title: str,
        meter: MeterType,
        mode: PoeticMode,
        tone: EmotionalTone,
        num_stanzas: int = 3,
        num_lines_per_stanza: int = 4,
    ) -> Ode:
        """Phase 2: Compose an ode through Horatian process."""
        ode = Ode(
            title=title,
            meter=meter,
            mode=mode,
            dominant_tone=tone,
        )

        diction_words = DecorumEnforcer.TONE_DICTION_MAP.get(tone, set())
        meter_pattern = MetricalPatternMatcher.TEMPLATES.get(meter, [6, 6, 5, 5])

        for stanza_num in range(num_stanzas):
            stanza = Stanza(stanza_form=f"stanza_{stanza_num + 1}")
            for line_num in range(num_lines_per_stanza):
                syllables = meter_pattern[line_num % len(meter_pattern)]
                # Build a pseudo-Latin line
                words = []
                for _ in range(random.randint(4, 8)):
                    sample_words = [
                        random.choice(list(diction_words)) if diction_words
                        else random.choice(["est", "non", "cum", "sed", "ut"])
                    ]
                    words.extend(sample_words)
                line_text = ' '.join(words[:random.randint(4, 7)])

                ictus = tuple(random.sample(range(syllables), min(2, syllables)))

                line = VerseLine(
                    text=line_text,
                    meter=meter,
                    syllable_count=syllables,
                    ictus_positions=ictus,
                    semantic_field=tone.name.lower(),
                    emotional_charge=random.uniform(-0.5, 0.8),
                )
                stanza.add_line(line)

            stanza.compute_unity()
            ode.stanzas.append(stanza)

        # Compute scores
        ode.dulce_score = sum(
            abs(l.emotional_charge) for s in ode.stanzas for l in s.lines
        ) / max(1, ode.total_lines())
        ode.utile_score = random.uniform(0.3, 0.7)
        ode.aptum_score = self.decorum_enforcer.check_aptum(ode)

        self.current_ode = ode
        self.composition_log.append(f"[COMPOSED] {title} in {meter.name}")
        return ode

    def revise(self) -> Ode:
        """Phase 3: Apply Horatian self-revision."""
        if not self.current_ode:
            raise ValueError("No ode to revise")
        self.composition_log.append("[REVISION] Beginning se ipsum mordendo")
        revised = self.revision_engine.full_revision_pass(self.current_ode)
        revised.aptum_score = self.decorum_enforcer.check_aptum(revised)
        self.composition_log.append("[REVISION] Complete")
        return revised

    def evaluate_for_maecenas(self) -> Dict[AudienceClass, float]:
        """Phase 4: Predict how Maecenas and others will respond."""
        if not self.current_ode:
            raise ValueError("No ode to evaluate")
        return self.response_predictor.predict_total_response(self.current_ode)

    def full_composition_cycle(
        self,
        occasion: str,
        addressee: str,
        tone: EmotionalTone,
        meter: MeterType = MeterType.ALCAIC,
    ) -> Tuple[Ode, Dict[AudienceClass, float]]:
        """Run the complete Horatian composition cycle."""
        # Step 1: Receive inspiration
        self.receive_inspiration(tone, occasion, addressee)

        # Step 2: Compose
        ode = self.compose_ode(
            title=f"Ode to {addressee}",
            meter=meter,
            mode=PoeticMode.LYRIC,
            tone=tone,
            num_stanzas=3,
            num_lines_per_stanza=4,
        )

        # Step 3: Revise
        ode = self.revise()

        # Step 4: Evaluate
        responses = self.evaluate_for_maecenas()

        return ode, responses

    def get_log(self) -> List[str]:
        return self.composition_log.copy()


# =============================================================================
# UTILITY CLASSES
# =============================================================================


class RomanPoeticTradition:
    """Contextual knowledge of Roman poetic tradition."""

    CANONICAL_WORKS = {
        "Odes (1-3)": "Lyric poetry, Maecenas, Augustus, love, wine",
        "Satires": "Social commentary, Stoic and Epicurean philosophy",
        "Epistles": "Familiar letters, artistic theory (Ars Poetica)",
        "Carmen Saeculare": "Centennial hymn for Augustus's Games",
    }

    INFLUENCES: List[str] = [
        "Alcaeus of Mytilene (Greek lyric)",
        "Sappho (Greek lyric)",
        "Archilochus (Greek iambic)",
        "Ennius (Roman epic)",
        "Lucretius (philosophical epic)",
    ]

    @classmethod
    def describe_influence(cls, poet: str) -> str:
        """Describe how a poet influenced Horace."""
        return f"Horace inherits meter and themes from {poet}"


class HoratianAutoCritique:
    """The 'mordens' (biting) self-critic within Horace's process."""

    def critique(self, ode: Ode) -> List[str]:
        """Provide critique in the manner of Horace's se ipsum mordendo."""
        critiques = []
        if ode.total_lines() < 8:
            critiques.append("Too brief - expand with more vivid imagery")
        if ode.dulce_score < 0.3:
            critiques.append("Not sweet enough - add more sensory pleasure")
        if ode.utile_score < 0.3:
            critiques.append("Lacks utile - include moral wisdom")
        aptum = ode.aptum_score
        if aptum < 0.6:
            critiques.append(f"Decorum deficient ({aptum:.2f}) - match meter to mood")
        balance = ode.compute_balance()
        if balance < 0.5:
            critiques.append("Imbalance - reconcile dulce and utile")
        critiques.append("Reread aloud - the ear must be satisfied")
        return critiques


# =============================================================================
# DEMO
# =============================================================================


def demo() -> None:
    """Demonstrate Horatian poetic cognition architecture."""
    print("=" * 70)
    print("HORATIAN POETIC COGNITION")
    print("Architecture for Horace's Ars Poetica and Ode Composition")
    print("=" * 70)

    # Initialize architecture
    cognition = HoratianPoeticCognition()

    # Show Ars Poetica rules
    print("\n--- ARS POETICA PRINCIPLES ---")
    for i, (name, desc) in enumerate(ArsPoeticaPrinciples.RULES[:5]):
        print(f"  {i+1}. {name}: {desc}")

    # Show metrical patterns
    print("\n--- METRICAL PATTERNS ---")
    for meter in MeterType:
        pattern = MetricalPatternMatcher.generate_pattern(meter)
        print(f"  {meter.name}: {pattern}")

    # Demonstrate full composition cycle
    print("\n--- COMPOSITION CYCLE: ODE TO MAECENAS ---")
    ode, responses = cognition.full_composition_cycle(
        occasion="the spring festival",
        addressee="Maecenas",
        tone=EmotionalTone.CONTEMPLATIVE,
        meter=MeterType.ALCAIC,
    )

    print(f"\nODE: {ode.title}")
    print(f"Meter: {ode.meter.name}")
    print(f"Mode: {ode.mode.name}")
    print(f"Dominat Tone: {ode.dominant_tone.name}")
    print(f"Total Lines: {ode.total_lines()}")
    print(f"Dulce Score: {ode.dulce_score:.3f}")
    print(f"Utile Score: {ode.utile_score:.3f}")
    print(f"Aptum Score: {ode.aptum_score:.3f}")
    print(f"Balance: {ode.compute_balance():.3f}")

    # Show stanza details
    print("\n--- STANZAS ---")
    for i, stanza in enumerate(ode.stanzas):
        print(f"  Stanza {i+1} (Unity: {stanza.unity_score:.2f}):")
        for line in stanza.lines:
            print(f"    [{line.meter.name}] {line.text[:50]}...")

    # Show audience responses
    print("\n--- AUDIENCE RESPONSES ---")
    for audience, score in responses.items():
        print(f"  {audience.name}: {score:.3f}")

    # Show critique
    print("\n--- HORATIAN SELF-CRITIQUE ---")
    critic = HoratianAutoCritique()
    for critique in critic.critique(ode):
        print(f"  ✎ {critique}")

    # Show composition log
    print("\n--- COMPOSITION LOG ---")
    for entry in cognition.get_log():
        print(f"  {entry}")

    # Demonstrate decorum checking
    print("\n--- DECORUM CHECK ---")
    for tone in EmotionalTone:
        appropriate_meters = DecorumEnforcer.TONE_METER_MAP.get(tone, set())
        diction_sample = DecorumEnforcer.TONE_DICTION_MAP.get(tone, set())
        print(f"  {tone.name}: {', '.join(m.name for m in appropriate_meters) if appropriate_meters else 'any'}")

    print("\n" + "=" * 70)
    print("Carpe diem! The Horatian poetic cognition architecture is complete.")
    print("=nil", end="")
    print("=" * 70)


if __name__ == "__main__":
    demo()


class HoratianOdeTypes:
    """Types of odes in Horace's collection."""
    def __init__(self):
        self.ode_types = {
            "carmen_ce_lebre": "Public ceremonial odes (Odes 1.1-9)",
            "convivales": "Banquet songs (drinking, love)",
            "amatorios": "Love poetry",
            "parthene_sapphici": "Maiden choruses",
            "polyhymniae": "Hymns to gods, philosophical odes"
        }

    def ode_type_characteristics(self) -> Dict[str, List[str]]:
        return {
            "carmen_ce_lebre": ["Augustan propaganda", "Political unity", "Public voice"],
            "convivales": ["Social occasion", "Wine and friendship", "Private voice"]
        }


class HoratianMeterSpecialist:
    """Specialized meter analysis for Horatian odes."""
    def __init__(self):
        self.alcaic_rules = {
            "lines_1_3": "U U - | - U U - | - - U U - U -",
            "line_4": "- U U - | - U U - -",
            "sacrifice": "U = long or short by position"
        }
        self.sapphic_rules = {
            "lines_1_3": "- U U - U | - U U - U U -",
            "line_4": "U U - U | - U U - -",
            "aeolic": "Final adonic line, graceful resolution"
        }

    def meter_pattern_description(self, meter: MeterType) -> str:
        if meter == MeterType.ALCAIC:
            return f"Alcaic: {self.alcaic_rules['lines_1_3']}"
        elif meter == MeterType.SAPPHIC:
            return f"Sapphic: {self.sapphic_rules['lines_1_3']}"
        return "Pattern description unavailable"


class HoratianAudienceAnalysis:
    """Analysis of Horace's intended and actual audiences."""
    def __init__(self):
        self.audiences = {
            "Maecenas": {"class": "Equestrian", "influence": "Primary patron"},
            "Augustus": {"class": "Imperial", "influence": "Ultimate dedicatee"},
            "Poetic_circle": {"class": "Poets", "influence": "Fellow artists"},
            "Roman_populace": {"class": "Mixed", "influence": "Through recitation"}
        }

    def audience_reach(self) -> Dict[str, str]:
        return {name: info["influence"] for name, info in self.audiences.items()}


class HoratianVerseStructure:
    """Structural analysis of Horatian verse."""
    def __init__(self):
        self.stanza_types = {
            "alcaic_quatrain": "4 lines, Asclepiadean meter variant",
            "sapphic_stanza": "3 Sapphic lines + 1 Adonic",
            "greater_hipponactean": "Choliambic (spondaic) meter",
            "anacreontic": "Lyric meter for light verse"
        }

    def stanza_example(self, stanza_type: str) -> str:
        return f"{stanza_type}: {self.stanza_types.get(stanza_type, 'Unknown type')}"


class PoeticAutobiographyHorace:
    """Horace's use of autobiographical elements."""
    def __init__(self):
        self.autobiographical_elements = [
            ("Satires 1.5", "Journey to Brundisium - real trip"),
            ("Satires 1.6", "Mansione satirist - his social position"),
            ("Odes 1.22", "Fuscus saved by tree - invented scene"),
            ("Epistulae 1.1", "Poet finds true self in retirement")
        ]

    def autobiographical_reference(self) -> List[Tuple[str, str]]:
        return self.autobiographical_elements


class HoratianIronyAnalysis:
    """Analysis of irony in Horace's works."""
    def __init__(self):
        self.irony_types = {
            "self_deprecation": "Poet claims incompetence while demonstrating mastery",
            "dramatic_irony": "Readers understand what speaker does not",
            "socratic_irony": "Feigned ignorance to expose folly",
            "situational_irony": "Outcomes contradict expectations"
        }

    def irony_in_ode(self, ode_sample: str) -> str:
        return self.irony_types.get("self_deprecation", "No irony detected")


class LyricPoetryGenreTheory:
    """Horace's contribution to lyric poetry theory."""
    def __init__(self):
        self.genre_characteristics = {
            "personal_emotion": "First-person expression of feeling",
            "musical_origin": "Originally sung to lyre accompaniment",
            "short_form": "Briefer than epic, concentrated intensity",
            "stanzaic_structure": "Regular stanza patterns for song"
        }

    def genre_definition(self) -> str:
        return "Lyric poetry: personal emotion in musical form, concentrated intensity"


class HoratianTextualHistory:
    """Manuscript tradition and textual history of Horace."""
    def __init__(self):
        self.manuscript_traditions = {
            "beta": "Vaticanus 3250 (9th c.) - best for Odes",
            "gamma": "Parisiensis 7974 (10th c.) - Satires",
            "epsilon": "Numerous 13th-15th c. copies"
        }

    def best_manuscript(self, work_type: str) -> str:
        if "Ode" in work_type:
            return "beta (Vaticanus 3250)"
        return "gamma (Parisiensis 7974)"


class HoratianInfluenceMap:
    """Later influence of Horace on European literature."""
    def __init__(self):
        self.influences = {
            "Middle_Ages": "Quoted as auctoritas in sermons",
            "Renaissance": "Model for neoclassical lyric (Ronsard)",
            "Enlightenment": "Pope, Dryden adapt Horatian satire",
            "Modern": "Frost, Auden adapt Odes for modern poetry"
        }

    def influence_path(self) -> Dict[str, str]:
        return self.influences


class HoratianSaturaForm:
    """The satirical form (satura) in Horace's Satires."""
    def __init__(self):
        self.satura_elements = {
            "origins": "Lucilian satire, no fixed form",
            "style": "Conversational, everyday speech",
            "themes": "Social criticism, philosophical discussion",
            "personae": "Different speakers, but Horace largely himself"
        }

    def satura_explanation(self) -> Dict[str, str]:
        return self.satura_elements


class HoraceEpistulaeAnalysis:
    """Analysis of Horace's Epistulae (Letters)."""
    def __init__(self):
        self.epistulae_books = {
            "Epistulae_1": "General letters to patrons and friends (20 BCE)",
            "Epistulae_2_1": "Letter to Augustus on poets and poetry",
            "Ars_Poetica": "Letter to Pisones on poetic theory"
        }

    def epistulae_explanation(self) -> Dict[str, str]:
        return self.epistulae_books


class HoratianMeterSystem:
    """Complete metrical system for Horatian poetry."""
    def __init__(self):
        self.meter_system = {
            "Alcaic_ode": "U U - | - U U - | - - U U - U - (4-line stanzas)",
            "Sapphic_ode": "- U U - U | - U U - U U - (3 lines + Adonic)",
            "Greater_Asclepiad": "- U U - | U U - - U U - U -",
            "Lesser_Asclepiad": "- U U - U U - - U U - U -",
            "Iambic_trimeter": "U - | U - | U - (in speeches)",
            "Hexameter": "Dactylic hexameter for Satires"
        }

    def meter_system_description(self) -> Dict[str, str]:
        return self.meter_system


class HoraceCarmenSaeculare:
    """Analysis of the Carmen Saeculare."""
    def __init__(self):
        self.saeculare_elements = {
            "commission": "Augustus requested for Secular Games 17 BCE",
            "form": "Phalaecian meter, 76 lines",
            "content": "Praises Apollo, Diana, Augustus, Roman future",
            "performance": "Children's chorus at Games"
        }

    def saeculare_explanation(self) -> Dict[str, str]:
        return self.saeculare_elements


class HoratianInfluenceHistory:
    """Horace's influence on later literature."""
    def __init__(self):
        self.influence_history = {
            "medieval": "Quoted in sermons, auctoritates",
            "renaissance": "Prince of poets, imitated everywhere",
            "17th_century": "Milton, Dryden adapt Horatian forms",
            "18th_century": "Pope's Essay on Criticism = Ars Poetica updated",
            "modern": "Frost's Odes show Horatian influence"
        }

    def influence_timeline(self) -> Dict[str, str]:
        return self.influence_history


class HoraceAndMaecenasRelationship:
    """The famous relationship with Maecenas."""
    def __init__(self):
        self.relationship = {
            "first_meeting": "39 BCE throughVirgil and Varius",
            "patronage_begin": "38 BCE - country estate near Tibur",
            "poetic_exchange": "Odes 1.1 dedicates work to Maecenas",
            "friendship_nature": "Respectful but not abject",
            "maecenas_gift": "Estate at Sabinum - financial security"
        }

    def relationship_explanation(self) -> Dict[str, str]:
        return self.relationship


class HoraceSatireOnRomanTypes:
    """Roman types criticized in Satires."""
    def __init__(self):
        self.roman_types = {
            "upstart": "Newly rich, vulgar display",
            "superstitious": "Excessive religious observance",
            "denial_of_dinner": "Invites without hospitality",
            "complaining_friend": "Chronic malcontent",
            "snob": "Judges by birth, not merit"
        }

    def types_list(self) -> List[str]:
        return list(self.roman_types.keys())


class ArsPoeticaDoctrines:
    """Core doctrines of Horace's Ars Poetica."""
    def __init__(self):
        self.doctrines = [
            ("Nature_first", "Imitate nature in all things"),
            ("Utile_dulce", "Join profit with pleasure"),
            ("Decorum", "Fit language to subject and speaker"),
            ("Unity", "Whole work cohere like summer pond"),
            ("Beginning_mid", "Start where action begins"),
            ("Avoid_mixing", "Keep tragic and comic separate"),
            ("Ideal_type", "Prefer universal to particular")
        ]

    def doctrine_list(self) -> List[Tuple[str, str]]:
        return self.doctrines


class HoraceOdeBookStructure:
    """Structure of Horace's three books of Odes."""
    def __init__(self):
        self.book_structure = {
            "Odes_1": "39 odes, opens with Maecenas, ends with political odes",
            "Odes_2": "20 odes, more elevated, less personal",
            "Odes_3": "30 odes, ends with Carmen Saeculare",
            "metrical_variety": "Mix of Alcaic, Sapphic, other meters"
        }

    def structure_explanation(self) -> Dict[str, str]:
        return self.book_structure


class HoratianPoeticAutobiography:
    """Autobiographical elements in Horace's poetry."""
    def __init__(self):
        self.autobiography = {
            "Satires_1_5": "Journey to Brundisium with Maecenas",
            "Satires_1_6": "His freedman status and social position",
            "Odes_1_22": " miraculous rescue by a tree",
            "Epistulae_1_16": "Estate description at Sabinum",
            "Epistulae_1_1": "Retirement wishes and self-examination"
        }

    def autobiographical_passages(self) -> Dict[str, str]:
        return self.autobiography


class HoraceOnPoetry:
    """Horace's theory of poetry."""
    def __init__(self):
        self.poetry_theory = {
            "inspired": "Poet needs divine afflatus (enthusiasm)",
            "trained": "But also needs art and craft",
            "useful": "Poetry as moral instruction",
            "delightful": "But also gives aesthetic pleasure",
            "unified": "All parts must work together"
        }

    def poetry_explanation(self) -> Dict[str, str]:
        return self.poetry_theory


class HoraceOnRomanSociety:
    """Horace's views on Roman society."""
    def __init__(self):
        self.society_views = {
            "greed": "The root of social problems",
            "ambitus": "Electoral bribery corrupts politics",
            "superstitio": " Excessive religious observance",
            "luxuria": "Consumerism and moral decline",
            "parvenus": "Newly rich with bad taste"
        }

    def society_criticism(self) -> List[str]:
        return list(self.society_views.keys())


class HoratianSelfPresentation:
    """How Horace presents himself in his poetry."""
    def __init__(self):
        self.self_presentation = {
            "humility": "Calls himself short, fat, quick to laugh",
            "independence": "Content with modest estate",
            "friendship": "Values Maecenas and others deeply",
            "philosophy": "Moderate Epicureanism",
            "Roman_pride": "Proud to be Roman citizen"
        }

    def self_presentation_explanation(self) -> Dict[str, str]:
        return self.self_presentation


class HoraceAndAugustus:
    """Horace's relationship with Augustus."""
    def __init__(self):
        self.augustus_relationship = {
            "initial_resistance": "Refused initial invitation from Augustus",
            "gradual_close": "Augustus eventually won over",
            "poetic_exchange": "Augustus requested verses, Horace demurred",
            "politics": "Eventually composed Augustan political odes",
            "respect": "Mutual respect, not subservience"
        }

    def augustus_explanation(self) -> Dict[str, str]:
        return self.augustus_relationship


class HoratianManuscriptTradition:
    """Manuscript tradition of Horace."""
    def __init__(self):
        self.manuscripts = {
            "beta_codex": " Blandiniensis, 9th c, best text",
            "medius": "Various 10-12th c manuscripts",
            "rennaissance": "Printings from 1470s onward",
            "editors": "Punted Bentle y, modern critical editions"
        }

    def manuscript_explanation(self) -> Dict[str, str]:
        return self.manuscripts


class HoraceOnWine:
    """The role of wine in Horace's poetry."""
    def __init__(self):
        self.wine_elements = {
            "symposium": "Wine as social lubricant",
            "medical": "Wine as medicine for grief",
            "celebration": "Wine for festive occasions",
            "moderation": "Not醉酒, but moderate enjoyment",
            "Bacchus": "God of wine, divine connection"
        }

    def wine_explanation(self) -> Dict[str, str]:
        return self.wine_elements


class HoratianPhilosophyApplication:
    """Philosophy as it appears in Horace's works."""
    def __init__(self):
        self.philosophy_elements = {
            "epicurean": "Pleasure, friendship, modest living",
            "stoic": "Virtue, duty, endurance",
            "cynic": "Self-sufficiency, rejection of luxury",
            "academic": "Skepticism, moderation"
        }

    def philosophy_practical(self) -> str:
        return "Horace was an eclectic, practical philosopher"


class HoraceOnLove:
    """Treatment of love in Horace's odes."""
    def __init__(self):
        self.love_odes = {
            "neobule": "Love for young girl, playful tone",
            "pyrrha": "Love for treacherous Pyrrha",
            "lalage": "Love for Lalage, softer tone",
            "lyce": "Love for unresponsive Lyce"
        }

    def love_ode_list(self) -> List[str]:
        return list(self.love_odes.keys())


class HoratianVerseVoice:
    """Characteristics of Horace's poetic voice."""
    def __init__(self):
        self.voice_characteristics = {
            "conversational": "Speaking to a friend, not performing",
            "witty": "Light touch, humor throughout",
            "self_deprecating": "Self mockery, not self-pity",
            "concrete": "Specific images over abstractions",
            "measured": "Controlled, not excessive"
        }

    def voice_explanation(self) -> Dict[str, str]:
        return self.voice_characteristics


class HoraceOdeClosing:
    """Horace's famous ode closings."""
    def __init__(self):
        self.closing_lines = [
            ("Odes 1.11", "Carpe diem - seize the day"),
            ("Odes 1.24", "Durable the soul, not wealth"),
            ("Odes 3.30", "Exegi monumentum aere perennius"),
            ("Odes 4.10", "Dulce et utile - sweet and useful")
        ]

    def closing_list(self) -> List[Tuple[str, str]]:
        return self.closing_lines
