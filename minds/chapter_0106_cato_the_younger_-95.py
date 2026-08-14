"""
Chapter 106: Cato the Younger
==============================
Figure 106: Cato the Younger (95-46 BCE)
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 6 Minds 101 - 120 Available on Amazon https://www.amazon.com/dp/B0HF7G6JJD
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 106: Cato the Younger (-95 to -46 BCE)
================================================================================
Domain: Politics, Ethics, Stoicism

Selection Rationale:
    Roman senator, statesman, and Stoic philosopher; great-grandson of
    Cato the Elder; renowned for his uncompromising integrity and
    opposition to Julius Caesar; served as tribune, quaestor, and praetor;
    committed suicide at Utica after Caesar's victory at Thapsus rather
    than submit to the dictator; symbol of republican virtue.

Key Belief About Mind:
    Virtue is the only true good; the wise man is self-sufficient and
    independent of external circumstances; moral integrity admits no
    compromise; death is preferable to submission to tyranny; Stoic
    principles must guide political action.

Agitation Relevance:
    Cato = moral integrity as terminal value; no-compromise ethics
    as alignment constraint; suicide as integrity-preserving action;
    Stoic virtue as internal state priority; republic as political
    structure for virtue.

Sources:
    - Plutarch, Life of Cato the Younger
    - Cicero, Cato Maior
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

class StoicPrinciple(Enum):
    """Core Stoic principles."""
    VIRTUE_AS_GOOD = auto()
    EXTERNAL_INDIFFERENT = auto()
    LIVING_ACCORDING_TO_NATURE = auto()
    LOGOS = auto()
    OKEIOSIS = auto()


class PoliticalPosition(Enum):
    """Political positions held by Cato."""
    REPUBLICAN = auto()
    SENATORIAL = auto()
    ANTI_CAESAR = auto()
    PRO_SULCIAN = auto()


class VirtueType(Enum):
    """Types of virtue."""
    PRUDENCE = auto()
    JUSTICE = auto()
    FORTITUDE = auto()
    TEMPERANCE = auto()


class DecisionType(Enum):
    """Types of decisions Cato faced."""
    POLITICAL = auto()
    MILITARY = auto()
    PERSONAL = auto()


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass(frozen=True)
class StoicBelief:
    """A Stoic belief held by Cato."""
    principle: StoicPrinciple
    application: str
    practical_expression: str


@dataclass
class PoliticalAction:
    """A political action taken by Cato."""
    year: int
    description: str
    position: PoliticalPosition
    outcome: str
    moral_justification: str


@dataclass
class Decision:
    """A key decision Cato faced."""
    decision_type: DecisionType
    description: str
    options: Tuple[str, ...]
    choice: str
    reasoning: str
    outcome: str

    def is_virtuous(self) -> bool:
        return "virtue" in self.reasoning.lower() or "principle" in self.reasoning.lower()


@dataclass
class SpeechRecord:
    """Record of one of Cato's speeches."""
    occasion: str
    year: int
    main_arguments: Tuple[str, ...]
    audience: str
    effect: str


@dataclass
class MoralPrinciple:
    """A moral principle guiding Cato."""
    principle: str
    source: str
    application: str
    non_negotiable: bool


@dataclass
class PoliticalContext:
    """Context for political situation."""
    year: int
    situation: str
    key_players: Tuple[str, ...]
    stakes: str


@dataclass
class IntegrityTest:
    """A test of Cato's integrity."""
    description: str
    pressure_applied: str
    cato_response: str
    principle_invoked: str


# =============================================================================
# TYPING CONSTRUCTS
# =============================================================================

T = TypeVar('T')


class StoicPhilosophy:
    """Model of Stoic philosophical system."""
    def __init__(self):
        self.beliefs: List[StoicBelief] = []

    def add_belief(self, principle: StoicPrinciple,
                  application: str, expression: str) -> None:
        self.beliefs.append(StoicBelief(principle, application, expression))

    def get_beliefs_by_principle(self, principle: StoicPrinciple) -> List[StoicBelief]:
        return [b for b in self.beliefs if b.principle == principle]

    def evaluate_action(self, action: str) -> bool:
        return any("virtue" in b.practical_expression.lower() or
                   "principle" in b.practical_expression.lower()
                   for b in self.beliefs)


class IntegrityChecker:
    """Check if actions maintain integrity."""
    def __init__(self):
        self.tests: List[IntegrityTest] = []

    def add_test(self, test: IntegrityTest) -> None:
        self.tests.append(test)

    def pass_test(self, description: str, response: str) -> bool:
        for test in self.tests:
            if description == test.description:
                return response == test.cato_response
        return False

    def calculate_integrity_score(self, responses: List[str]) -> float:
        if not responses:
            return 0.0
        passed = sum(1 for r in responses if any(r == t.cato_response for t in self.tests))
        return passed / max(len(responses), 1)


class DecisionAnalyzer:
    """Analyze decisions using Stoic framework."""
    def __init__(self):
        self.decisions: List[Decision] = []

    def add_decision(self, decision: Decision) -> None:
        self.decisions.append(decision)

    def get_virtuous_decisions(self) -> List[Decision]:
        return [d for d in self.decisions if d.is_virtuous()]

    def evaluate_reasoning(self, reasoning: str) -> Dict[str, Any]:
        return {
            "uses_virtue_language": "virtue" in reasoning.lower(),
            "references_principle": "principle" in reasoning.lower(),
            "practical": "practical" in reasoning.lower(),
            "stoic_aligned": "nature" in reasoning.lower() or "logos" in reasoning.lower()
        }


class PoliticalTracker:
    """Track political positions and actions."""
    def __init__(self):
        self.actions: List[PoliticalAction] = []

    def add_action(self, action: PoliticalAction) -> None:
        self.actions.append(action)

    def actions_by_year(self, year: int) -> List[PoliticalAction]:
        return [a for a in self.actions if a.year == year]

    def actions_by_position(self, position: PoliticalPosition) -> List[PoliticalAction]:
        return [a for a in self.actions if a.position == position]

    def consistent_positions(self) -> bool:
        republican_actions = [a for a in self.actions
                            if a.position == PoliticalPosition.REPUBLICAN]
        return len(republican_actions) >= len(self.actions) * 0.7


class MoralPrincipleManager:
    """Manage moral principles."""
    def __init__(self):
        self.principles: List[MoralPrinciple] = []

    def add_principle(self, principle: str, source: str,
                    application: str, non_negotiable: bool) -> None:
        self.principles.append(MoralPrinciple(principle, source, application, non_negotiable))

    def get_non_negotiable(self) -> List[MoralPrinciple]:
        return [p for p in self.principles if p.non_negotiable]

    def check_violation(self, action: str) -> List[MoralPrinciple]:
        violated = []
        for p in self.principles:
            if p.non_negotiable and p.principle.lower() not in action.lower():
                violated.append(p)
        return violated


class SpeechAnalyzer:
    """Analyze Cato's speeches."""
    def __init__(self):
        self.speeches: List[SpeechRecord] = []

    def add_speech(self, occasion: str, year: int,
                  arguments: Tuple[str, ...],
                  audience: str, effect: str) -> None:
        self.speeches.append(SpeechRecord(occasion, year, arguments, audience, effect))

    def find_speeches_by_year(self, year: int) -> List[SpeechRecord]:
        return [s for s in self.speeches if s.year == year]

    def speeches_about(self, topic: str) -> List[SpeechRecord]:
        return [s for s in self.speeches
                if topic.lower() in s.main_arguments[0].lower()]


class ContextAnalyzer:
    """Analyze political context."""
    def __init__(self):
        self.contexts: List[PoliticalContext] = []

    def add_context(self, context: PoliticalContext) -> None:
        self.contexts.append(context)

    def get_context_for_year(self, year: int) -> Optional[PoliticalContext]:
        for c in self.contexts:
            if c.year == year:
                return c
        return None

    def key_player_involved(self, player: str) -> List[PoliticalContext]:
        return [c for c in self.contexts if player in c.key_players]


# =============================================================================
# MAIN CLASS
# =============================================================================

class CatoSystem:
    """
    Cato the Younger's Stoic and political system.

    Implements:
    - Stoic belief management
    - Integrity checking
    - Decision analysis
    - Political action tracking
    - Moral principle management
    - Speech analysis
    - Political context analysis
    """

    def __init__(self):
        self.stoic_philosophy = StoicPhilosophy()
        self.integrity_checker = IntegrityChecker()
        self.decision_analyzer = DecisionAnalyzer()
        self.political_tracker = PoliticalTracker()
        self.moral_manager = MoralPrincipleManager()
        self.speech_analyzer = SpeechAnalyzer()
        self.context_analyzer = ContextAnalyzer()

        self._initialize_beliefs()
        self._initialize_principles()
        self._initialize_actions()
        self._initialize_decisions()
        self._initialize_speeches()
        self._initialize_contexts()
        self._initialize_integrity_tests()

    def _initialize_beliefs(self) -> None:
        beliefs = [
            (StoicPrinciple.VIRTUE_AS_GOOD,
             "Only virtue constitutes good",
             "Chose virtue over political advantage"),
            (StoicPrinciple.EXTERNAL_INDIFFERENT,
             "Wealth, power, life are indifferent",
             "Rejected honors from corrupt sources"),
            (StoicPrinciple.LIVING_ACCORDING_TO_NATURE,
             "Live according to nature and reason",
             "Maintained principles under pressure"),
        ]
        for principle, application, expression in beliefs:
            self.stoic_philosophy.add_belief(principle, application, expression)

    def _initialize_principles(self) -> None:
        principles = [
            ("Virtue is the only good", "Stoic teaching", "Never compromise virtue", True),
            ("Honor must be preserved", "Roman tradition", "Rejectdishonor", True),
            ("Republic is ideal form", "Political belief", "Oppose tyranny", True),
            ("Death is preferable to shame", "Personal conviction", "Choose death over submission", True),
            ("Truth must be spoken", "Philosophical commitment", "Speak truth to power", False),
        ]
        for principle, source, application, non_neg in principles:
            self.moral_manager.add_principle(principle, source, application, non_neg)

    def _initialize_actions(self) -> None:
        actions = [
            PoliticalAction(-63, "Opposed Catiline conspiracy",
                          PoliticalPosition.REPUBLICAN,
                          "Served Cicero's suppression",
                          "Duty to preserve republic"),
            PoliticalAction(-58, "Opposed Clodius",
                          PoliticalPosition.REPUBLICAN,
                          "Exile from Rome",
                          "Would not compromise with criminal"),
            PoliticalAction(-49, "Opposed Caesar's advance",
                          PoliticalPosition.ANTI_CAESAR,
                          "Failed to stop civil war",
                          "Republic must be defended"),
            PoliticalAction(-48, "Fought at Pharsalus",
                          PoliticalPosition.REPUBLICAN,
                          "Defeat at Caesar's hands",
                          "Chose duty over safety"),
            PoliticalAction(-46, "Committed suicide at Utica",
                          PoliticalPosition.ANTI_CAESAR,
                          "Preferredexile to submission",
                          "Integrity preserved unto death"),
        ]
        for action in actions:
            self.political_tracker.add_action(action)

    def _initialize_decisions(self) -> None:
        decisions = [
            Decision(DecisionType.POLITICAL,
                    "Oppose Catiline",
                    ("Support conspiracy", "Oppose conspiracy", "Remain neutral"),
                    "Oppose conspiracy",
                    "Virtue requires opposing corruption",
                    "Served republic"),
            Decision(DecisionType.POLITICAL,
                    "Refuse Caesar's offer of clemency",
                    ("Accept clemency", "Refuse and fight", "Negotiate"),
                    "Refuse and continue resistance",
                    "Virtue admits no compromise with tyrant",
                    "Maintained integrity"),
            Decision(DecisionType.PERSONAL,
                    "Suicide at Utica",
                    ("Captured", "Flee", "Suicide"),
                    "Suicide",
                    "Death preferable to submitting to tyranny",
                    "Preserved honor"),
        ]
        for decision in decisions:
            self.decision_analyzer.add_decision(decision)

    def _initialize_speeches(self) -> None:
        speeches = [
            SpeechRecord("Against Catiline", -63,
                        ("Conspiracy must be suppressed", "Senators must act"),
                        "Senate",
                        "Convinced senators to act"),
            SpeechRecord("Against Caesar's agrarian law", -59,
                        ("Principle over expedience", "Corrupt legislation"),
                        "People's Assembly",
                        "Lost vote but maintained principle"),
            SpeechRecord("Defense of Metellus", -57,
                        ("Principled defense", "Refused to abandon friend"),
                        "Forum",
                        "Maintained integrity at cost"),
        ]
        for speech in speeches:
            self.speech_analyzer.add_speech(
                speech.occasion, speech.year, speech.main_arguments,
                speech.audience, speech.effect
            )

    def _initialize_contexts(self) -> None:
        contexts = [
            PoliticalContext(-63, "Catiline conspiracy",
                           ("Catiline", "Cicero", "Caesar"),
                           "Republic's survival"),
            PoliticalContext(-49, "Caesar crosses Rubicon",
                           ("Caesar", "Pompey", "Senate"),
                           "Freedom vs tyranny"),
            PoliticalContext(-46, "Thapsus and aftermath",
                           ("Caesar", "Metellus", "Senators"),
                           "Integrity vs survival"),
        ]
        for context in contexts:
            self.context_analyzer.add_context(context)

    def _initialize_integrity_tests(self) -> None:
        tests = [
            IntegrityTest("Caesar offers clemency",
                         "Pardon and high office",
                         "Refused all offers",
                         "Virtue cannot compromise"),
            IntegrityTest("Political advantage through compromise",
                         "Wealth and position for softening",
                         "Rejected completely",
                         "Principle non-negotiable"),
            IntegrityTest("Death or submission",
                         "Submit and live, or die",
                         "Chose death",
                         "Honor above life"),
        ]
        for test in tests:
            self.integrity_checker.add_test(test)

    def get_stoic_beliefs(self) -> List[StoicBelief]:
        return self.stoic_philosophy.beliefs

    def evaluate_action(self, action: str) -> bool:
        return self.stoic_philosophy.evaluate_action(action)

    def get_decision(self, decision_type: DecisionType) -> Optional[Decision]:
        for d in self.decision_analyzer.decisions:
            if d.decision_type == decision_type:
                return d
        return None

    def get_political_actions(self, year: int) -> List[PoliticalAction]:
        return self.political_tracker.actions_by_year(year)

    def get_moral_principles(self) -> List[MoralPrinciple]:
        return self.moral_manager.principles

    def get_non_negotiable_principles(self) -> List[MoralPrinciple]:
        return self.moral_manager.get_non_negotiable()


# =============================================================================
# DEMO
# =============================================================================

def demo() -> None:
    print("=" * 70)
    print("CATO THE YOUNGER: STOIC VIRTUE AND POLITICAL INTEGRITY")
    print("95-46 BCE | Roman Senator | Stoic Philosopher")
    print("=" * 70)

    system = CatoSystem()

    print("\n1. STOIC BELIEFS")
    print("-" * 40)
    beliefs = system.get_stoic_beliefs()
    for belief in beliefs:
        print(f"  [{belief.principle.name}]")
        print(f"    Application: {belief.application}")
        print(f"    Expression: {belief.practical_expression}")
        print()

    print("\n2. MORAL PRINCIPLES")
    print("-" * 40)
    principles = system.get_moral_principles()
    for p in principles:
        status = "NON-NEGOTIABLE" if p.non_negotiable else "Flexible"
        print(f"  {p.principle} ({status})")
        print(f"    Application: {p.application}")
    non_neg = system.get_non_negotiable_principles()
    print(f"\n  Non-negotiable principles: {len(non_neg)}")

    print("\n3. POLITICAL ACTIONS")
    print("-" * 40)
    for action in system.political_tracker.actions:
        print(f"  {action.year}: {action.description}")
        print(f"    Position: {action.position.name}")
        print(f"    Outcome: {action.outcome}")
        print(f"    Justification: {action.moral_justification}")
    consistent = system.political_tracker.consistent_positions()
    print(f"\n  Consistent positions: {consistent}")

    print("\n4. KEY DECISIONS")
    print("-" * 40)
    decisions = system.decision_analyzer.decisions
    for decision in decisions:
        print(f"  [{decision.decision_type.name}] {decision.description}")
        print(f"    Choice: {decision.choice}")
        print(f"    Reasoning: {decision.reasoning}")
        print(f"    Virtuous: {decision.is_virtuous()}")
        print()
    virtuous = system.decision_analyzer.get_virtuous_decisions()
    print(f"  Virtuous decisions: {len(virtuous)}/{len(decisions)}")

    print("\n5. SPEECHES")
    print("-" * 40)
    speeches = system.speech_analyzer.speeches
    for speech in speeches:
        print(f"  {speech.occasion} (-{abs(speech.year)})")
        print(f"    Arguments: {', '.join(speech.main_arguments)}")
        print(f"    Audience: {speech.audience}")
        print(f"    Effect: {speech.effect}")

    print("\n6. INTEGRITY TESTS")
    print("-" * 40)
    tests = system.integrity_checker.tests
    for test in tests:
        print(f"  Test: {test.description}")
        print(f"    Pressure: {test.pressure_applied}")
        print(f"    Cato's response: {test.cato_response}")
        print(f"    Principle: {test.principle_invoked}")

    print("\n7. POLITICAL CONTEXT")
    print("-" * 40)
    contexts = system.context_analyzer.contexts
    for ctx in contexts:
        print(f"  Year {ctx.year}: {ctx.situation}")
        print(f"    Key players: {', '.join(ctx.key_players)}")
        print(f"    Stakes: {ctx.stakes}")

    print("\n8. EVALUATING ACTION")
    print("-" * 40)
    test_actions = [
        "Opposed Caesar to preserve republic",
        "Compromised with Caesar for peace",
        "Committed suicide to preserve honor",
    ]
    for action in test_actions:
        virtuous = system.evaluate_action(action)
        print(f"  '{action[:40]}...' -> Stoically virtuous: {virtuous}")

    print("\n9. DECISION ANALYSIS")
    print("-" * 40)
    for decision in system.decision_analyzer.decisions:
        eval_result = system.decision_analyzer.evaluate_reasoning(decision.reasoning)
        print(f"  {decision.description[:30]}:")
        for k, v in eval_result.items():
            if v:
                print(f"    {k}: {v}")

    print("\n10. PRINCIPLES IN ACTION")
    print("-" * 40)
    for p in system.get_non_negotiable_principles():
        print(f"  {p.principle}")
        print(f"    -> {p.application}")

    print("\n" + "=" * 70)
    print("CATO THE YOUNGER SYSTEM COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    demo()

class RepublicanSenateFaction:
    """Analyze senatorial factions supporting Republic."""
    def __init__(self):
        self.members: Dict[str, Dict[str, Any]] = {}

    def add_senator(self, name: str, wealth: int,
                   military_service: bool, speeches: int,
                   faction_role: str) -> None:
        self.members[name] = {
            "wealth": wealth,
            "military": military_service,
            "speeches": speeches,
            "role": faction_role
        }

    def most_influential(self) -> Optional[str]:
        if not self.members:
            return None
        return max(self.members.items(), key=lambda x: x[1]["speeches"])[0]

    def senators_by_wealth(self, min_wealth: int) -> List[str]:
        return [name for name, info in self.members.items()
                if info["wealth"] >= min_wealth]


class StoicPhilosophyApplicator:
    """Apply Stoic philosophy to political situations."""
    def __init__(self):
        self.principles = {
            "virtue": "Wisdom, courage, justice, temperance",
            "apatheia": "Freedom from destructive emotions",
            "cosmopolitanism": "Citizenship in the cosmos",
            "duty": "Living in accordance with nature"
        }

    def principle_explanation(self, principle: str) -> Optional[str]:
        return self.principles.get(principle)

    def apply_to_situation(self, situation: str, principle: str) -> str:
        return f"Applying {principle} to {situation}"


class LateRepublicCrisisAnalyzer:
    """Analyze crises of the Late Republic."""
    def __init__(self):
        self.crises = {
            "land_distribution": "Giant landowners displacing small farmers",
            "military_reforms": "Marian professional army replacing citizen forces",
            "provincial_exploitation": "Governors enriching themselves",
            "popular_unrest": "Urban poor demanding bread and games",
            "senatorial_gridlock": "Optimates vs Populares deadlock"
        }

    def crisis_description(self, crisis_name: str) -> Optional[str]:
        return self.crises.get(crisis_name)


class CatoSpeechThemes:
    """Analyze recurring themes in Cato's speeches."""
    def __init__(self):
        self.themes = {
            "moral_decline": ["corruption", "degeneration", "ancestors"],
            "roman_tradition": ["mos maiorum", "discipline", "simplicity"],
            "civic_duty": ["service", "republic", "freedom"],
            "virtue": ["honesty", "frugality", "steadfastness"]
        }

    def themes_in_speech(self, speech_text: str) -> List[str]:
        found = []
        text_lower = speech_text.lower()
        for theme, keywords in self.themes.items():
            if any(kw in text_lower for kw in keywords):
                found.append(theme)
        return found


class CaesarCatoRelationshipAnalyzer:
    """Analyze the political rivalry between Caesar and Cato."""
    def __init__(self):
        self.conflicts = [
            ("Catiline Conspiracy", "Cato accused Caesar of involvement"),
            ("First Triumvirate", "Secret alliance vs public opposition"),
            ("Gallia Conquest", "Cato criticized war methods"),
            ("Civil War", "Cato opposed Caesar's dictatorship")
        ]

    def all_conflicts(self) -> List[Tuple[str, str]]:
        return self.conflicts

    def nature_of_rivalry(self) -> str:
        return "Ideological: Stoic virtue vs Populares reform"


class OptimatesFactionAnalyzer:
    """Analyze the optimates senatorial faction."""
    def __init__(self):
        self.leaders = ["Metellus Scipio", "Domitius Ahenobarbus", "Cato"]
        self.ideology = {
            "senate_supremacy": "Power concentrated in Senate",
            "traditional_values": "Preserve mos maiorum",
            "aristocratic_rule": "Rule by nobiles",
            "opposition_to_reform": "Resist populares measures"
        }

    def faction_ideology(self) -> Dict[str, str]:
        return self.ideology

    def leaders_list(self) -> List[str]:
        return self.leaders


class CatoCharacterTraits:
    """Document Cato's character traits."""
    def __init__(self):
        self.traits = {
            "steadfastness": "Unwavering commitment to principles",
            "austerity": "Simple living, rejection of luxury",
            "eloquence": "Powerful orator despite voice issues",
            "integrity": "Cannot be bribed or intimidated",
            "severity": "Strict enforcement of laws",
            "independence": "Ally of neither Caesar nor Pompey"
        }

    def trait_description(self, trait: str) -> Optional[str]:
        return self.traits.get(trait)


class RomanMoralityCodeAnalyzer:
    """Analyze the Roman code of morality."""
    def __init__(self):
        self.moral_codes = {
            "pietas": "Duty to gods, family, and state",
            "fides": "Faithfulness to promises",
            "gravitas": "Dignity and seriousness",
            "constantia": "Steadfastness in adversity",
            "verecundia": "Respect for others"
        }

    def code_description(self, code: str) -> Optional[str]:
        return self.moral_codes.get(code)


class RepublicanInstitutionalAnalyzer:
    """Analyze how institutions sustained Republic."""
    def __init__(self):
        self.institutions = {
            "senate": "Deliberative body advising magistrates",
            "magistracies": "Elected offices with term limits",
            "tribunes": "Representatives of plebeians",
            "comitia": "Popular assemblies for voting",
            "auspices": "Religious sanction for actions"
        }

    def institution_purpose(self, name: str) -> Optional[str]:
        return self.institutions.get(name)


class CatoHistoricalLegacy:
    """Document Cato's historical legacy."""
    def __init__(self):
        self.legacy_aspects = {
            "republican_symbol": "Martyr for republican ideals",
            "stoic_exemplar": "Model of philosophical virtue",
            "moral_critic": "Conscience opposing corruption",
            "suicide_as_protest": "Final rejection of tyranny"
        }

    def legacy_description(self, aspect: str) -> Optional[str]:
        return self.legacy_aspects.get(aspect)


class RomanSenateProcedure:
    """Document Roman Senate procedures."""
    def __init__(self):
        self.procedures = {
            "senatus_habitus": "Formal session with presiding consul",
            "consultum": "Formal decree after deliberation",
            "fidelis": "Oath-binding decision",
            "patrum_auctoritas": "Patrician approval for laws"
        }

    def procedure_description(self, procedure: str) -> Optional[str]:
        return self.procedures.get(procedure)


class PoliticalViolenceAnalyzer:
    """Analyze political violence in Late Republic."""
    def __init__(self):
        self.incidents = [
            ("Sulla's March on Rome", -88, "First military march on capital"),
            ("Murder of Saturninus", -100, "Stoning by mob"),
            ("Clodius/Cato violence", -50, "Gangs controlling streets"),
            ("Caesar's Assassination", -44, "Ides of March plot")
        ]

    def incidents_list(self) -> List[Tuple[str, int, str]]:
        return self.incidents


class mosMaiorumAnalyzer:
    """Analyze the ancestral customs (mos maiorum)."""
    def __init__(self):
        self.customs = {
            "religious_duty": "Proper worship of gods",
            "ancestor_veneration": "Honor past generations",
            "agricultural_tradition": "Farmer-soldier ideal",
            "simplicity": "Rejection of foreign luxuries",
            "discipline": "Military and civic obedience"
        }

    def custom_description(self, custom: str) -> Optional[str]:
        return self.customs.get(custom)


if __name__ == "__main__":
    demo()


class CatoWritingsAnalyzer:
    """Analyze writings attributed to or about Cato."""
    def __init__(self):
        self.writings = {
            "Origines": "Historical works on Roman origins",
            "De Re Militari": "Military treatise",
            "Orationes": "Collection of speeches",
            "Epistulae": "Letters to various recipients"
        }

    def writing_description(self, title: str) -> Optional[str]:
        return self.writings.get(title)


class RomanElectionsAnalyzer:
    """Analyze Roman electoral procedures."""
    def __init__(self):
        self.offices = {
            "Quaestor": {"min_age": 27, "wealth": 38000},
            "Aedile": {"min_age": 36, "wealth": 115000},
            "Praetor": {"min_age": 39, "wealth": 230000},
            "Consul": {"min_age": 42, "wealth": 380000}
        }

    def office_requirements(self, office_name: str) -> Optional[Dict[str, int]]:
        return self.offices.get(office_name)


class SenatorialCareerAnalyzer:
    """Analyze senatorial career paths."""
    def __init__(self):
        self.career_stages = [
            "Military service (tribune or centurion)",
            "Quaestor (financial officer)",
            "Aedile (public works and games)",
            "Praetor (judicial and military)",
            "Consul (highest office)",
            "Censor (moral supervision)"
        ]

    def career_path(self) -> List[str]:
        return self.career_stages


class RepublicanValueSystem:
    """Document the value system of Roman Republic."""
    def __init__(self):
        self.values = {
            "res_publica": "The public thing - republic",
            "senatus_populusque_romanus": "Senate and people of Rome",
            "virtus": "Manly virtue and courage",
            "honor": "Public honor and reputation",
            "gloria": "Glory earned through service"
        }

    def value_meaning(self, value: str) -> Optional[str]:
        return self.values.get(value)


class PoliticalAllianceNegotiator:
    """Analyze political alliance formation."""
    def __init__(self):
        self.alliances: Dict[str, Set[str]] = {}

    def propose_alliance(self, person1: str, person2: str) -> bool:
        if person1 not in self.alliances:
            self.alliances[person1] = set()
        if person2 not in self.alliances:
            self.alliances[person2] = set()
        self.alliances[person1].add(person2)
        self.alliances[person2].add(person1)
        return True

    def alliance_exists(self, person1: str, person2: str) -> bool:
        return person2 in self.alliances.get(person1, set())


class RomanLawPrincipleLibrary:
    """Library of Roman legal principles."""
    def __init__(self):
        self.principles = {
            "dura_lex_sed_lex": "The law is harsh but it is the law",
            "volenti_non_fit_injuria": "No injury to one who consents",
            "pacta_sunt_servanda": "Agreements must be kept",
            "qui_scribis_bene_scribis": "He who writes, writes well"
        }

    def principle_text(self, principle: str) -> Optional[str]:
        return self.principles.get(principle)


class CatoDeathSignificance:
    """Analyze significance of Cato's suicide."""
    def __init__(self):
        self.meanings = {
            "political_statement": "Rejected Caesar's victory",
            "stoic_purpose": "True philosopher dies for principles",
            "republican_martyrdom": "Last stand of republican virtue",
            "personal_integrity": "Could not survive under tyranny"
        }

    def meaning_description(self, meaning: str) -> Optional[str]:
        return self.meanings.get(meaning)


class RomanPatricianAnalyzer:
    """Analyze patrician families in Late Republic."""
    def __init__(self):
        self.families = {
            "Cornelii": ["Sulla", "Cinna", "Scipio"],
            "Aemilii": ["Paullus", "Scaurus"],
            "Julii": ["Caesar", "Caesar (Octavian's adoptive)"],
            "Claudii": ["Nero", "Claudius"],
            "Fabii": ["Various conservative senators"]
        }

    def family_members(self, family_name: str) -> List[str]:
        return self.families.get(family_name, [])


class SenatorialOratoryStyle:
    """Analyze senatorial oratory styles."""
    def __init__(self):
        self.styles = {
            "cato_style": "Austere, moralistic, direct",
            "caesar_style": "Elegant, persuasive, subtle",
            "cicero_style": "Rhetorical, elaborate, philosophical"
        }

    def style_description(self, style_name: str) -> Optional[str]:
        return self.styles.get(style_name)


class RomanPublicAssemblySystem:
    """Analyze Roman public assemblies."""
    def __init__(self):
        self.assemblies = {
            "comitia_centuriata": "Voted by wealth-based centuries",
            "comitia_tributa": "Voted by tribal divisions",
            "concilium_plebis": "Plebeian assembly only"
        }

    def assembly_purpose(self, assembly_name: str) -> Optional[str]:
        return self.assemblies.get(assembly_name)


class LateRepublicTimelineBuilder:
    """Build timeline of Late Republic events."""
    def __init__(self):
        self.events: List[Tuple[int, str]] = [
            (-133, "Tiberius Gracchus killed"),
            (-121, "Gaius Gracchus killed"),
            (-107, "Marian reforms begin"),
            (-88, "Sulla marches on Rome"),
            (-82, "Sulla becomes dictator"),
            (-73, "Spartacus revolt"),
            (-63, "Catiline conspiracy"),
            (-60, "First Triumvirate"),
            (-49, "Caesar crosses Rubicon"),
            (-44, "Caesar assassinated"),
            (-43, "Second Triumvirate"),
            (-42, "Battle of Philippi"),
        ]

    def events_list(self) -> List[Tuple[int, str]]:
        return sorted(self.events, key=lambda x: x[0])


class RepublicanConsulAnalyzer:
    """Analyze consular activities."""
    def __init__(self):
        self.consuls: Dict[str, Dict[str, Any]] = {}

    def add_consul(self, name: str, year: int,
                  colleague: str, major_actions: List[str]) -> None:
        self.consuls[name] = {
            "year": year,
            "colleague": colleague,
            "actions": major_actions
        }

    def consul_info(self, name: str) -> Optional[Dict[str, Any]]:
        return self.consuls.get(name)


class RomanTriumphProcedure:
    """Document Roman triumph procedure."""
    def __init__(self):
        self.requirements = {
            "victory_type": "Decisive military victory",
            "senate_approval": "Senators must vote approval",
            "religious_sacrifice": "Proper animal sacrifice",
            "procession_route": "From Campus Martius to Capitol"
        }

    def requirement_description(self, req: str) -> Optional[str]:
        return self.requirements.get(req)


if __name__ == "__main__":
    demo()


class CatoPhilosophicalInfluence:
    """Analyze Cato's philosophical influence on later thinkers."""
    def __init__(self):
        self.influences: Dict[str, List[str]] = {
            "Seneca": ["Stoic virtue ethics", "Moral integrity"],
            "Epictetus": ["Virtue as sufficient for happiness", "Externals are indifferent"],
            "Marcus_Aurelius": ["Duty over personal safety", "Stoic ruler philosophy"],
            "Cicero": ["Republican virtue", "Philosophical dialogue"],
            "Plutarch": ["Moral biography", "Comparative lives"],
        }

    def get_influence_on(self, thinker: str) -> List[str]:
        return self.influences.get(thinker, [])


class RomanVotingSystems:
    """Analyze Roman voting assemblies and procedures."""
    def __init__(self):
        self.assemblies = {
            "comitia_centuriata": {"type": "Wealth-based voting", "leader": "Consul"},
            "comitia_tributa": {"type": "Tribe-based voting", "leader": "Consul"},
            "concilium_plebis": {"type": "Plebeian-only assembly", "leader": "Tribune"},
        }

    def voting_procedure(self, assembly: str) -> str:
        info = self.assemblies.get(assembly, {})
        return f"{assembly}: {info.get('type', 'Unknown')}"


class SenatorialDebateAnalyzer:
    """Analyze senatorial debate procedures."""
    def __init__(self):
        self.debate_phases = [
            "Relation of issue by magistrate",
            "Senators speak in order of seniority",
            "Pedarius consulted if needed",
            "Consul summarizes (perduellio)",
            "Division (discessio) called",
            "Minority may protest (senatus consulta ultimum)"
        ]

    def analyze_debate(self) -> List[Tuple[int, str]]:
        return list(enumerate(self.debate_phases, 1))


class CatoOratoryStyle:
    """Analyze Cato's distinctive oratory style."""
    def __init__(self):
        self.characteristics = {
            "austerity": "Plain language, no rhetorical embellishment",
            "moral_urgency": "Constant reference to ancestral virtue",
            "directness": "Short, punchy sentences",
            "stoic_framework": "Arguments framed in philosophical terms",
            "republican_patriotism": "Passionate defense of liberty"
        }

    def style_description(self) -> Dict[str, str]:
        return self.characteristics


class RomanAmbitusAnalyzer:
    """Analyze electoral bribery (ambitus) as social problem."""
    def __init__(self):
        self.laws = [
            ("Lex Gabinia", -67, "Bribery at elections"),
            ("Lex Cassia", -61, "Electoral corruption"),
            ("Lex Pompeia", -52, "Strengthened penalties"),
        ]

    def get_laws(self) -> List[Tuple[str, int, str]]:
        return self.laws
