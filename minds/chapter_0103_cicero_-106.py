"""
Chapter 103: Cicero
====================
Figure 103: Cicero (106-43 BCE)
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 6 Minds 101 - 120 Available on Amazon https://www.amazon.com/dp/B0HF7G6JJD
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 103: Cicero (-106 to -43 BCE)
================================================================================
Domain: Rhetoric, Philosophy, Politics

Selection Rationale:
    Roman statesman, orator, lawyer, and writer; considered one of
    Rome's greatest orators and prose stylists; developed the
    philosophical concept of "自然会" (natural law) based on Stoic
    principles; compiled and transmitted Greek philosophy to Roman
    world; wrote prolifically on rhetoric, philosophy, politics,
    and law; executed by Antony for his political writings.

Key Belief About Mind:
    The orator must be a philosopher to be truly effective; the ideal
    orator combines wisdom with eloquence; philosophical training
    develops moral character; the properly educated mind can discern
    truth and speak it persuasively.

Agitation Relevance:
    Cicero = ideal orator as philosophical AI; natural law as
    universal ethical reasoning; rhetoric as persuasion technology;
    philosophical synthesis as knowledge compression; moral education
    as alignment training.

Sources:
    - Cicero, Complete Works
    - Kennedy (1972), 'Cicero'
    - Polybius, The Histories
    - Cochrane (1929), 'Thucydides and the Science of History'
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Dict, List, Optional, Tuple, Any, Callable,
    Generator, Iterator, TypeVar, Generic, Protocol,
    NamedTuple, Union, Set
)
from datetime import datetime
import json
import copy


# =============================================================================
# ENUMS
# =============================================================================

class RhetoricalGenre(Enum):
    """Genres of rhetorical discourse."""
    DELIBERATIVE = auto()  # political/advice
    JUDICIAL = auto()      # legal/forensic
    EPIDEICTIC = auto()    # ceremonial/panegyric


class PhilosophicalSchool(Enum):
    """Schools influencing Cicero."""
    STOIC = auto()
    ACADEMIC = auto()      # New Academy
    PERIPATETIC = auto()   # Aristotelian
    EPICUREAN = auto()


class Virtue(Enum):
    """Ciceronian virtues."""
    WISDOM = auto()
    JUSTICE = auto()
    FORTITUDE = auto()
    TEMPERANCE = auto()


class OratoricalDevice(Enum):
    """Rhetorical devices for persuasion."""
    RHETORICAL_QUESTION = auto()
    PARALLELISM = auto()
    ANTITHESIS = auto()
    TRIAD = auto()
    CLIMAX = auto()


class LegalProcedure(Enum):
    """Roman legal procedures."""
    COGNITIO = auto()
    PROCEDENDO = auto()
    LIBERUM = auto()


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass(frozen=True)
class Oration:
    """A complete oration by Cicero."""
    title: str
    date_delivered: Optional[int]
    genre: RhetoricalGenre
    main_thesis: str
    arguments: Tuple[str, ...]
    stylistic_devices: Tuple[OratoricalDevice, ...]
    historical_context: str


@dataclass
class PhilosophicalTreatise:
    """A philosophical work by Cicero."""
    title: str
    school: PhilosophicalSchool
    key_claims: Tuple[str, ...]
    greek_sources: Tuple[str, ...]
    roman_application: str


@dataclass
class Orator:
    """Model of the ideal orator."""
    name: str
    rhetorical_skill: float  # 0-1
    philosophical_knowledge: float
    moral_character: float
    political_experience: float

    def is_ideal_orator(self) -> bool:
        return all([
            self.rhetorical_skill >= 0.8,
            self.philosophical_knowledge >= 0.7,
            self.moral_character >= 0.8
        ])


@dataclass
class NaturalLawPrinciple:
    """A principle of natural law theory."""
    principle: str
    derivation: str
    applications: Tuple[str, ...]


@dataclass
class RhetoricalArgument:
    """An argument structure in rhetoric."""
    premise: str
    evidence: Tuple[str, ...]
    inference: str
    conclusion: str

    def strength_score(self) -> float:
        return min(1.0, len(self.evidence) * 0.2 + 0.3)


@dataclass
class PhilosophicalSynthesis:
    """Synthesis of Greek philosophy for Roman use."""
    greek_doctrine: str
    roman_context: str
    ciceronian_reformulation: str
    practical_application: str


@dataclass
class PoliticalSpeech:
    """A speech on political matters."""
    occasion: str
    audience: str
    main_claim: str
    supporting_reasons: Tuple[str, ...]
    emotional_appeals: Tuple[str, ...]


@dataclass
class LegalCase:
    """A legal case structure."""
    charges: Tuple[str, ...]
    defense_arguments: Tuple[str, ...]
    evidence_presented: Tuple[str, ...]
    verdict_likelihood: float


@dataclass
class RhetoricalTraining:
    """Cicero's method of rhetorical training."""
    stages: Tuple[str, ...]
    exercises: Tuple[str, ...]
    models_studied: Tuple[str, ...]
    duration_years: int


# =============================================================================
# TYPING CONSTRUCTS
# =============================================================================

T = TypeVar('T')


class OrationBuilder:
    """Build orations in Ciceronian style."""
    def __init__(self):
        self.thesis = ""
        self.arguments: List[str] = []
        self.devices: List[OratoricalDevice] = []

    def set_thesis(self, thesis: str) -> None:
        self.thesis = thesis

    def add_argument(self, arg: str) -> None:
        self.arguments.append(arg)

    def add_device(self, device: OratoricalDevice) -> None:
        self.devices.append(device)

    def build(self) -> str:
        parts = [f"Thesis: {self.thesis}", ""]
        parts.append("Arguments:")
        for i, arg in enumerate(self.arguments, 1):
            parts.append(f"  {i}. {arg}")
        parts.append("")
        parts.append(f"Stylistic devices: {', '.join(d.name for d in self.devices)}")
        return "\n".join(parts)


class IdealOratorChecker:
    """Check whether someone meets the ideal orator standard."""
    def __init__(self):
        self.min_rhetorical = 0.8
        self.min_philosophical = 0.7
        self.min_moral = 0.8

    def check(self, orator: Orator) -> Tuple[bool, List[str]]:
        deficiencies = []
        if orator.rhetorical_skill < self.min_rhetorical:
            deficiencies.append("Insufficient rhetorical skill")
        if orator.philosophical_knowledge < self.min_philosophical:
            deficiencies.append("Lacks philosophical training")
        if orator.moral_character < self.min_moral:
            deficiencies.append("Moral character needs development")
        is_ideal = len(deficiencies) == 0
        return is_ideal, deficiencies


class NaturalLawReasoner:
    """Reason about natural law principles."""
    def __init__(self):
        self.principles: List[NaturalLawPrinciple] = []

    def add_principle(self, principle: str, derivation: str,
                     applications: Tuple[str, ...]) -> None:
        self.principles.append(NaturalLawPrinciple(principle, derivation, applications))

    def derive_from_reason(self, premise: str) -> List[str]:
        results = []
        for p in self.principles:
            if premise.lower() in p.principle.lower():
                results.append(p.principle)
        return results

    def apply_principle(self, principle_name: str, case: str) -> str:
        for p in self.principles:
            if p.principle == principle_name:
                return f"Applying {principle_name} to {case}: {p.applications[0] if p.applications else 'no application'}"
        return "Principle not found"


class PhilosophicalSynthesizer:
    """Synthesize Greek philosophy for Roman context."""
    def __init__(self):
        self.syntheses: List[PhilosophicalSynthesis] = []

    def synthesize(self, greek: str, roman_context: str,
                  reformulation: str, application: str) -> PhilosophicalSynthesis:
        syn = PhilosophicalSynthesis(greek, roman_context, reformulation, application)
        self.syntheses.append(syn)
        return syn

    def get_syntheses_by_school(self, school: PhilosophicalSchool) -> List[PhilosophicalSynthesis]:
        return [s for s in self.syntheses if self._school_matches(s, school)]

    def _school_matches(self, s: PhilosophicalSynthesis, school: PhilosophicalSchool) -> bool:
        if school == PhilosophicalSchool.STOIC:
            return "stoic" in s.greek_doctrine.lower() or "logos" in s.greek_doctrine.lower()
        if school == PhilosophicalSchool.ACADEMIC:
            return "academy" in s.greek_doctrine.lower()
        return False


class RhetoricalDeviceApplicator:
    """Apply rhetorical devices in composition."""
    def __init__(self):
        self.history: List[str] = []

    def apply_triplet(self, items: List[str]) -> str:
        result = " - ".join(items)
        self.history.append(f"Triad: {result}")
        return f"{items[0]} - {items[1]} - {items[2]}"

    def apply_antithesis(self, thing1: str, thing2: str) -> str:
        result = f"{thing1} ... {thing2}"
        self.history.append(f"Antithesis: {result}")
        return result

    def apply_climax(self, items: List[str]) -> str:
        result = " ... ".join(items)
        self.history.append(f"Climax: {result}")
        return result


class PoliticalReasoner:
    """Reason about political situations."""
    def __init__(self):
        self.speeches: List[PoliticalSpeech] = []

    def analyze_situation(self, context: str) -> Dict[str, Any]:
        return {
            "context": context,
            "likely_claims": ["Unity is strength", "Danger requires action"],
            "audience_concerns": ["Security", "Honor", "Prosperity"]
        }

    def prepare_speech(self, occasion: str, audience: str,
                      claim: str, reasons: Tuple[str, ...]) -> PoliticalSpeech:
        speech = PoliticalSpeech(occasion, audience, claim, reasons,
                               ("Appeal to ancestral wisdom", "Appeal to mutual interest"))
        self.speeches.append(speech)
        return speech


class LegalAnalyzer:
    """Analyze legal cases in Roman style."""
    def __init__(self):
        self.cases: List[LegalCase] = []

    def analyze_case(self, charges: Tuple[str, ...],
                    defense: Tuple[str, ...],
                    evidence: Tuple[str, ...]) -> LegalCase:
        strength = min(1.0, len(defense) * 0.2 + len(evidence) * 0.1)
        case = LegalCase(charges, defense, evidence, strength)
        self.cases.append(case)
        return case

    def estimate_outcome(self, case: LegalCase) -> str:
        if case.verdict_likelihood >= 0.7:
            return "Favorable for defense"
        elif case.verdict_likelihood >= 0.4:
            return "Uncertain outcome"
        else:
            return "Likely unfavorable for defense"


class MoralCharacterEvaluator:
    """Evaluate moral character of historical figures."""
    def __init__(self):
        self.evaluations: Dict[str, Dict[Virtue, int]] = {}

    def evaluate(self, name: str, virtues: Dict[Virtue, int]) -> None:
        self.evaluations[name] = virtues

    def compare_virtues(self, name1: str, name2: str, virtue: Virtue) -> str:
        if name1 not in self.evaluations or name2 not in self.evaluations:
            return "Evaluation not available"
        v1 = self.evaluations[name1].get(virtue, 0)
        v2 = self.evaluations[name2].get(virtue, 0)
        if v1 > v2:
            return f"{name1} exceeded {name2} in {virtue.name}"
        elif v2 > v1:
            return f"{name2} exceeded {name1} in {virtue.name}"
        return f"Equal in {virtue.name}"


class RhetoricalTrainingProgram:
    """Cicero's rhetorical training method."""
    def __init__(self):
        self.program = RhetoricalTraining(
            stages=("Memory", "Delivery", " Invention", "Arrangement", "Style"),
            exercises=("Imitation", "Composition", "Debate", " declamation"),
            models_studied=("Demosthenes", "Lysias", "Pericles", "Greek philosophers"),
            duration_years=2
        )

    def get_training_stages(self) -> Tuple[str, ...]:
        return self.program.stages

    def get_exercises(self) -> Tuple[str, ...]:
        return self.program.exercises


# =============================================================================
# MAIN CLASS
# =============================================================================

class CiceroSystem:
    """
    Ciceronian rhetorical and philosophical system.

    Implements:
    - Oration construction in Ciceronian style
    - Ideal orator evaluation
    - Natural law reasoning
    - Greek philosophy synthesis
    - Political speech preparation
    - Legal case analysis
    - Moral character evaluation
    - Rhetorical training methodology
    """

    def __init__(self):
        self.orations: List[Oration] = []
        self.treatises: List[PhilosophicalTreatise] = []
        self.natural_law = NaturalLawReasoner()
        self.synthesizer = PhilosophicalSynthesizer()
        self.oration_builder = OrationBuilder()
        self.ideal_orator_checker = IdealOratorChecker()
        self.device_applicator = RhetoricalDeviceApplicator()
        self.political_reasoner = PoliticalReasoner()
        self.legal_analyzer = LegalAnalyzer()
        self.moral_evaluator = MoralCharacterEvaluator()
        self.training_program = RhetoricalTrainingProgram()

        self._initialize_orations()
        self._initialize_natural_law()
        self._initialize_syntheses()

    def _initialize_orations(self) -> None:
        self.orations = [
            Oration(
                title="In Catilinam",
                date_delivered=-63,
                genre=RhetoricalGenre.DELIBERATIVE,
                main_thesis="Catiline must be expelled from Rome",
                arguments=(
                    "He conspires against the state",
                    "He has raised arms against Rome",
                    "The people must defend themselves"
                ),
                stylistic_devices=(OratoricalDevice.TRIAD, OratoricalDevice.ANTITHESIS),
                historical_context="Catilinarian conspiracy during consulship"
            ),
            Oration(
                title="Pro Milone",
                date_delivered=-52,
                genre=RhetoricalGenre.JUDICIAL,
                main_thesis="Milo should be acquitted of murder",
                arguments=(
                    "Clodius was the aggressor",
                    "Self-defense is legitimate",
                    "The people support Milo"
                ),
                stylistic_devices=(OratoricalDevice.PARALLELISM, OratoricalDevice.RHETORICAL_QUESTION),
                historical_context="Trial for killing Clodius"
            ),
            Oration(
                title="Pro Archia",
                date_delivered=-62,
                genre=RhetoricalGenre.DELIBERATIVE,
                main_thesis="Archias should retain citizenship",
                arguments=(
                    "Talent benefits the state",
                    "Education elevates citizens",
                    "Rome has always welcomed scholars"
                ),
                stylistic_devices=(OratoricalDevice.CLIMAX,),
                historical_context="Defense of poet Archias's citizenship"
            ),
        ]

    def _initialize_natural_law(self) -> None:
        self.natural_law.add_principle(
            "True law is right reason",
            "According to nature, applicable to all peoples",
            ("Natural rights exist", "Law applies universally")
        )
        self.natural_law.add_principle(
            "Justice is the crowning glory of virtue",
            "Without justice, no virtue has worth",
            ("Fair treatment of citizens", "Protection of innocents")
        )
        self.natural_law.add_principle(
            "The state exists for the citizen",
            "Government serves those governed",
            ("Rule for common good", "Citizens have natural rights")
        )

    def _initialize_syntheses(self) -> None:
        self.synthesizer.synthesize(
            "Stoic Logos doctrine",
            "Roman political context",
            "Natural Law as universal rational principle",
            "Apply to politics and law"
        )
        self.synthesizer.synthesize(
            "Academic skepticism",
            "Roman rhetorical practice",
            "Probabilistic knowledge with practical action",
            "Use in oratory and debate"
        )
        self.synthesizer.synthesize(
            "Peripatetic ethics",
            "Roman aristocratic values",
            "Virtue in action within society",
            "Guide for statesmen"
        )

    def add_oration(self, oration: Oration) -> None:
        self.orations.append(oration)

    def find_oration(self, title: str) -> Optional[Oration]:
        for o in self.orations:
            if o.title == title:
                return o
        return None

    def check_orator(self, orator: Orator) -> Tuple[bool, List[str]]:
        return self.ideal_orator_checker.check(orator)

    def apply_rhetorical_device(self, device_type: str, content: Any) -> str:
        if device_type == "triad":
            return self.device_applicator.apply_triplet(list(content)[:3])
        elif device_type == "antithesis":
            items = list(content)
            return self.device_applicator.apply_antithesis(items[0], items[1])
        elif device_type == "climax":
            return self.device_applicator.apply_climax(list(content))
        return "Device not recognized"

    def analyze_political_situation(self, context: str) -> Dict[str, Any]:
        return self.political_reasoner.analyze_situation(context)

    def prepare_political_speech(self, occasion: str, audience: str,
                                claim: str, reasons: Tuple[str, ...]) -> PoliticalSpeech:
        return self.political_reasoner.prepare_speech(occasion, audience, claim, reasons)

    def analyze_legal_case(self, charges: Tuple[str, ...],
                          defense: Tuple[str, ...],
                          evidence: Tuple[str, ...]) -> LegalCase:
        return self.legal_analyzer.analyze_case(charges, defense, evidence)

    def evaluate_moral_character(self, name: str, virtues: Dict[Virtue, int]) -> None:
        self.moral_evaluator.evaluate(name, virtues)

    def compare_persons(self, name1: str, name2: str, virtue: Virtue) -> str:
        return self.moral_evaluator.compare_virtues(name1, name2, virtue)

    def get_training_program(self) -> RhetoricalTraining:
        return self.training_program.program


# =============================================================================
# DEMO
# =============================================================================

def demo() -> None:
    print("=" * 70)
    print("CICERO: IDEAL ORATOR AND NATURAL LAW PHILOSOPHY")
    print("106-43 BCE | Roman Statesman | Orator | Philosopher")
    print("=" * 70)

    system = CiceroSystem()

    print("\n1. CICERONIAN ORATIONS")
    print("-" * 40)
    for oration in system.orations:
        print(f"  {oration.title} (-{oration.date_delivered})")
        print(f"    Genre: {oration.genre.name}")
        print(f"    Thesis: {oration.main_thesis}")
        print(f"    Arguments: {len(oration.arguments)}")
        devices = [d.name for d in oration.stylistic_devices]
        print(f"    Devices: {', '.join(devices)}")
        print()

    print("\n2. IDEAL ORATOR EVALUATION")
    print("-" * 40)
    cicero = Orator("Cicero", 0.95, 0.85, 0.90, 0.80)
    antony = Orator("Antony", 0.70, 0.50, 0.50, 0.75)
    demosthenes = Orator("Demosthenes", 0.90, 0.75, 0.85, 0.65)
    for person in [cicero, antony, demosthenes]:
        is_ideal, deficiencies = system.check_orator(person)
        status = "IDEAL ORATOR" if is_ideal else "Below standard"
        print(f"  {person.name}: {status}")
        if deficiencies:
            print(f"    Deficiencies: {', '.join(deficiencies)}")

    print("\n3. NATURAL LAW PRINCIPLES")
    print("-" * 40)
    principle = system.natural_law.principles[0]
    print(f"  Principle: {principle.principle}")
    print(f"  Derivation: {principle.derivation}")
    print(f"  Applications: {', '.join(principle.applications)}")
    derivations = system.natural_law.derive_from_reason("law")
    print(f"  Derived: {derivations}")

    print("\n4. RHETORICAL DEVICE APPLICATION")
    print("-" * 40)
    triad = system.apply_rhetorical_device("triad", ["unity", "courage", "wisdom"])
    print(f"  Triplet: {triad}")
    antithesis = system.apply_rhetorical_device("antithesis", ["peace", "war"])
    print(f"  Antithesis: {antithesis}")
    climax = system.apply_rhetorical_device("climax", ["first", "second", "third"])
    print(f"  Climax: {climax}")

    print("\n5. GREEK PHILOSOPHY SYNTHESES")
    print("-" * 40)
    stoic_syn = system.synthesizer.get_syntheses_by_school(PhilosophicalSchool.STOIC)
    print(f"  Stoic syntheses: {len(stoic_syn)}")
    for syn in stoic_syn:
        print(f"    Greek: {syn.greek_doctrine}")
        print(f"    Roman: {syn.roman_context}")
        print(f"    Application: {syn.practical_application}")

    print("\n6. POLITICAL SPEECH PREPARATION")
    print("-" * 40)
    analysis = system.analyze_political_situation("Senate debate on war")
    print(f"  Context: {analysis['context']}")
    print(f"  Likely claims: {', '.join(analysis['likely_claims'])}")
    speech = system.prepare_political_speech(
        "Senate session",
        "Roman Senators",
        "War is necessary for peace",
        ("Security requires it", "Enemy threatens allies", "Honor demands response")
    )
    print(f"  Speech prepared: {speech.main_claim}")

    print("\n7. LEGAL CASE ANALYSIS")
    print("-" * 40)
    case = system.analyze_legal_case(
        ("Murder", "Conspiracy"),
        ("Self-defense", "Provocation"),
        ("Witness testimony", "Physical evidence")
    )
    print(f"  Charges: {', '.join(case.charges)}")
    print(f"  Defense arguments: {', '.join(case.defense_arguments)}")
    print(f"  Evidence: {', '.join(case.evidence_presented)}")
    outcome = system.legal_analyzer.estimate_outcome(case)
    print(f"  Estimated outcome: {outcome}")

    print("\n8. MORAL CHARACTER EVALUATION")
    print("-" * 40)
    system.evaluate_moral_character("Cicero", {
        Virtue.WISDOM: 4, Virtue.JUSTICE: 4,
        Virtue.FORTITUDE: 3, Virtue.TEMPERANCE: 3
    })
    system.evaluate_moral_character("Catiline", {
        Virtue.WISDOM: 2, Virtue.JUSTICE: 1,
        Virtue.FORTITUDE: 3, Virtue.TEMPERANCE: 1
    })
    comparison = system.compare_persons("Cicero", "Catiline", Virtue.JUSTICE)
    print(f"  Comparison: {comparison}")

    print("\n9. RHETORICAL TRAINING PROGRAM")
    print("-" * 40)
    program = system.get_training_program()
    print(f"  Duration: {program.duration_years} years")
    print(f"  Stages: {', '.join(program.stages)}")
    print(f"  Exercises: {', '.join(program.exercises)}")
    print(f"  Models: {', '.join(program.models_studied)}")

    print("\n10. ORATION BUILDER")
    print("-" * 40)
    builder = OrationBuilder()
    builder.set_thesis("Rome must defend its allies")
    builder.add_argument("Allies provide strategic support")
    builder.add_argument("Abandoning them dishonors Rome")
    builder.add_argument("Future allies will not trust us")
    builder.add_device(OratoricalDevice.TRIAD)
    builder.add_device(OratoricalDevice.ANTITHESIS)
    built = builder.build()
    print(built[:200])

    print("\n" + "=" * 70)
    print("CICERO SYSTEM DEMONSTRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    demo()

class OratoryStyleAnalyzer:
    """Analyze Cicero's oratory styles."""
    def __init__(self):
        self.styles = {
            "deliberative": "Political advice, encouraging or discouraging action",
            "judicial": "Legal argumentation, accusation or defense",
            "epideictic": "Ceremonial, praise or blame"
        }

    def classify_oratory(self, text: str) -> List[str]:
        matches = []
        text_lower = text.lower()
        if any(word in text_lower for word in ["should", "must", "ought", "propose"]):
            matches.append("deliberative")
        if any(word in text_lower for word in ["crime", "guilt", "innocent", "justice"]):
            matches.append("judicial")
        if any(word in text_lower for word in ["honor", "praise", "glory", "famous"]):
            matches.append("epideictic")
        return matches if matches else ["unknown"]

    def style_description(self, style: str) -> str:
        return self.styles.get(style, "Unknown style")


class PoliticalAllianceTracker:
    """Track political alliances in the Late Republic."""
    def __init__(self):
        self.alliances: Dict[str, Set[str]] = {}

    def form_alliance(self, person1: str, person2: str) -> None:
        if person1 not in self.alliances:
            self.alliances[person1] = set()
        if person2 not in self.alliances:
            self.alliances[person2] = set()
        self.alliances[person1].add(person2)
        self.alliances[person2].add(person1)

    def break_alliance(self, person1: str, person2: str) -> None:
        if person1 in self.alliances:
            self.alliances[person1].discard(person2)
        if person2 in self.alliances:
            self.alliances[person2].discard(person1)

    def allies_of(self, person: str) -> Set[str]:
        return self.alliances.get(person, set())

    def common_allies(self, person1: str, person2: str) -> Set[str]:
        return self.allies_of(person1) & self.allies_of(person2)


class LegalArgumentBuilder:
    """Build legal arguments in Roman style."""
    def __init__(self):
        self.arguments: List[Dict[str, Any]] = []

    def build_argument(self, charge: str, evidence: List[str],
                     witnesses: List[str], laws: List[str],
                     precedent: Optional[str] = None) -> Dict[str, Any]:
        argument = {
            "charge": charge,
            "evidence": evidence,
            "witnesses": witnesses,
            "laws": laws,
            "precedent": precedent
        }
        self.arguments.append(argument)
        return argument

    def strongest_argument(self) -> Optional[Dict[str, Any]]:
        if not self.arguments:
            return None
        return max(self.arguments, key=lambda x: len(x["evidence"]))


class RhetoricalDeviceAnalyzer:
    """Analyze rhetorical devices in speeches."""
    def __init__(self):
        self.devices = {
            "anaphora": ["repeated_first", "repetition_at_start"],
            "epistrophe": ["repeated_end", "repetition_at_end"],
            "antithesis": ["contrast", "opposition"],
            "rhetorical_question": ["question_not_answered"],
            "tricolon": ["three_parts", "triple"]
        }

    def detect_device(self, text: str) -> List[str]:
        found = []
        for device, patterns in self.devices.items():
            if any(p.replace("_", " ") in text.lower() for p in patterns):
                found.append(device)
        return found

    def device_count(self, text: str) -> Dict[str, int]:
        counts = {}
        for device in self.devices:
            if device in self.detect_device(text):
                counts[device] = counts.get(device, 0) + 1
        return counts


class PhilosophicalInfluenceMapper:
    """Map philosophical influences on Cicero."""
    def __init__(self):
        self.influences: Dict[str, List[str]] = {
            "Plato": ["Academy", "dialogue_form", "theory_of_forms"],
            "Aristotle": ["Peripatetics", "rhetoric_treatises", "ethics"],
            "Stoics": ["duty_concept", "natural_law", "cosmopolitanism"],
            "Epicureans": ["pleasure_ethics", "withdrawal_from_politics"]
        }

    def get_influences(self, philosopher: str) -> List[str]:
        return self.influences.get(philosopher, [])

    def all_influences(self) -> Dict[str, List[str]]:
        return self.influences


class CorrespondenceNetworkAnalyzer:
    """Analyze Cicero's correspondence network."""
    def __init__(self):
        self.letters: List[Dict[str, Any]] = []

    def add_letter(self, sender: str, recipient: str, date: str,
                  subject: str, tone: str) -> None:
        self.letters.append({
            "sender": sender,
            "recipient": recipient,
            "date": date,
            "subject": subject,
            "tone": tone
        })

    def correspondence_between(self, person1: str, person2: str) -> List[Dict[str, Any]]:
        return [l for l in self.letters
                if {l["sender"], l["recipient"]} == {person1, person2}]

    def most_frequent_correspondent(self, person: str) -> Optional[str]:
        correspondent_counts: Dict[str, int] = {}
        for letter in self.letters:
            if letter["sender"] == person:
                other = letter["recipient"]
            elif letter["recipient"] == person:
                other = letter["sender"]
            else:
                continue
            correspondent_counts[other] = correspondent_counts.get(other, 0) + 1
        if not correspondent_counts:
            return None
        return max(correspondent_counts.items(), key=lambda x: x[1])[0]


class PoliticalCareerReconstructor:
    """Reconstruct Cicero's political career."""
    def __init__(self):
        self.positions: List[Dict[str, Any]] = []

    def add_position(self, office: str, year: int,
                    achievements: List[str], challenges: List[str]) -> None:
        self.positions.append({
            "office": office,
            "year": year,
            "achievements": achievements,
            "challenges": challenges
        })

    def career_timeline(self) -> List[Dict[str, Any]]:
        return sorted(self.positions, key=lambda x: x["year"])

    def positions_in_year_range(self, start_year: int, end_year: int) -> List[Dict[str, Any]]:
        return [p for p in self.positions if start_year <= p["year"] <= end_year]


class LiteraryWorkClassifier:
    """Classify Cicero's literary works."""
    def __init__(self):
        self.works: Dict[str, str] = {
            "De Oratore": "philosophical_dialogue",
            "Orator": "rhetorical_treatise",
            "Brutus": "historical_essay",
            "De Re Publica": "philosophical_dialogue",
            "De Legibus": "philosophical_dialogue",
            "Letters to Atticus": "personal_correspondence",
            "Philippics": "political_oratory"
        }

    def classify_work(self, title: str) -> Optional[str]:
        return self.works.get(title)

    def works_by_type(self, work_type: str) -> List[str]:
        return [title for title, wt in self.works.items() if wt == work_type]


if __name__ == "__main__":
    demo()


class LegalCaseAnalyzer:
    """Analyze specific legal cases from Cicero's career."""
    def __init__(self):
        self.cases: List[Dict[str, Any]] = []

    def add_case(self, case_name: str, year: int, client: str,
                charges: List[str], verdict: str,
                cicero_role: str) -> None:
        self.cases.append({
            "case_name": case_name,
            "year": year,
            "client": client,
            "charges": charges,
            "verdict": verdict,
            "role": cicero_role
        })

    def cases_in_year(self, year: int) -> List[Dict[str, Any]]:
        return [c for c in self.cases if c["year"] == year]

    def won_cases(self) -> List[Dict[str, Any]]:
        return [c for c in self.cases if "guilty" not in c["verdict"].lower()]


class OratoricalTechniqueLibrary:
    """Library of Cicero's oratorical techniques."""
    def __init__(self):
        self.techniques = {
            "exordium": "Opening that gains audience attention",
            "narratio": "Presentation of facts",
            "argumentatio": "Proof and refutation",
            "peroratio": "Emotional conclusion"
        }

    def get_technique(self, name: str) -> Optional[str]:
        return self.techniques.get(name)

    def all_techniques(self) -> List[str]:
        return list(self.techniques.keys())


class ConstitutionalPrincipleExtractor:
    """Extract constitutional principles from Cicero's works."""
    def __init__(self):
        self.principles: Dict[str, List[str]] = {
            "Separation of Powers": ["consul", "senate", "people"],
            "Rule of Law": ["law", "legal", "justice"],
            "Popular Sovereignty": ["people", "populus", "assembly"]
        }

    def find_principles(self, text: str) -> List[str]:
        text_lower = text.lower()
        found = []
        for principle, keywords in self.principles.items():
            if any(kw in text_lower for kw in keywords):
                found.append(principle)
        return found


class HistoricalPrecedentFinder:
    """Find historical precedents cited by Cicero."""
    def __init__(self):
        self.precedents: List[Dict[str, str]] = []

    def add_precedent(self, era: str, figure: str, event: str,
                     cicero_citation: str) -> None:
        self.precedents.append({
            "era": era,
            "figure": figure,
            "event": event,
            "citation": cicero_citation
        })

    def precedents_from_era(self, era: str) -> List[Dict[str, str]]:
        return [p for p in self.precedents if p["era"] == era]

    def precedents_about_figure(self, figure: str) -> List[Dict[str, str]]:
        return [p for p in self.precedents if p["figure"] == figure]


class RhetoricalSituationClassifier:
    """Classify rhetorical situations Cicero faced."""
    def __init__(self):
        self.situations: List[Dict[str, Any]] = []

    def add_situation(self, context: str, audience: str,
                     purpose: str, constraints: List[str],
                     appropriate_style: str) -> None:
        self.situations.append({
            "context": context,
            "audience": audience,
            "purpose": purpose,
            "constraints": constraints,
            "style": appropriate_style
        })

    def situations_in_context(self, context: str) -> List[Dict[str, Any]]:
        return [s for s in self.situations if context in s["context"]]


class LatinPhraseCollector:
    """Collect important Latin phrases from Cicero."""
    def __init__(self):
        self.phrases: Dict[str, str] = {
            "Carthago delenda est": "The city of Carthage must be destroyed",
            "O tempora! O mores!": "O the times! O the customs!",
            "Veni, vidi, vici": "I came, I saw, I conquered (Caesar, not Cicero)",
            "Dulce et decorum est": "It is sweet and honorable (Horace)",
            "Salus populi suprema lex": "The welfare of the people is the supreme law"
        }

    def translate_phrase(self, phrase: str) -> Optional[str]:
        return self.phrases.get(phrase)

    def all_phrases(self) -> Dict[str, str]:
        return self.phrases


class OratoricalSuccessMetrics:
    """Measure success of oratorical efforts."""
    def __init__(self):
        self.metrics_weights = {
            "persuasion": 0.4,
            "eloquence": 0.3,
            "logic": 0.2,
            "emotional_appeal": 0.1
        }

    def calculate_success(self, persuasion_score: float,
                        eloquence_score: float,
                        logic_score: float,
                        emotion_score: float) -> float:
        return (persuasion_score * 0.4 +
                eloquence_score * 0.3 +
                logic_score * 0.2 +
                emotion_score * 0.1)


if __name__ == "__main__":
    demo()
