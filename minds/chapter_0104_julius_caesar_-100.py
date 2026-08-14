"""
Chapter 104: Julius Caesar
===========================
Figure 104: Julius Caesar (-100 to -44 BCE)
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 6 Minds 101 - 120 Available on Amazon https://www.amazon.com/dp/B0HF7G6JJD
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 104: Julius Caesar (-100 to -44 BCE)
================================================================================
Domain: Military, Leadership, Writing

Selection Rationale:
    Roman statesman, general, and author; transformed the Roman Republic
    into the Roman Empire; conquered Gaul; crossed the Rubicon; reformed
    the calendar; wrote Commentarii de Bello Gallico; established the
    principate; assassinated on the Ides of March.

Key Belief About Mind:
    Leadership requires clear strategic vision; action is superior to
    hesitation; the general must inspire and direct; written records
    serve both glory and strategy; decisive action shapes history.

Agitation Relevance:
    Caesar = military AI as strategic optimization; decisive action
    vs deliberation; the general as decision-making system; commentarii
    as structured knowledge; conquest as expansion of cognitive reach.

Sources:
    - Caesar, Commentarii de Bello Gallico
    - Suetonius, Life of Julius Caesar
    - Plutarch, Life of Caesar
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

class CampaignType(Enum):
    """Types of military campaigns."""
    CONQUEST = auto()
    DEFENSE = auto()
    PACIFICATION = auto()
    EXPANSION = auto()


class MilitaryFormation(Enum):
    """Roman military formations."""
    MANIPLE = auto()
    COHORT = auto()
    LEGION = auto()
    TESTUDO = auto()


class StrategicDecision(Enum):
    """Types of strategic decisions."""
    BATTLE = auto()
    SIEGE = auto()
    NEGOTIATION = auto()
    RETREAT = auto()


class WritingGenre(Enum):
    """Genres of Caesar's writings."""
    MILITARYCommentary = auto()
    LETTERS = auto()
    SPEECHES = auto()


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass(frozen=True)
class MilitaryCampaign:
    """A military campaign led by Caesar."""
    name: str
    year_start: int
    year_end: int
    campaign_type: CampaignType
    battles: Tuple[str, ...]
    outcome: str
    strategic_assessment: str


@dataclass
class BattleRecord:
    """Record of a single battle."""
    name: str
    date: int
    location: str
    enemy: str
    roman_strength: int
    enemy_strength: int
    outcome: str
    casualties_ratio: str
    strategic_significance: float  # 0-1

    def is_victory(self) -> bool:
        return "victory" in self.outcome.lower() or "won" in self.outcome.lower()


@dataclass
class StrategicPlan:
    """A strategic plan for campaign."""
    objective: str
    phases: Tuple[str, ...]
    resource_requirements: Tuple[str, ...]
    timeline_months: int
    contingency_plans: Tuple[str, ...]

    def feasibility_score(self) -> float:
        base = min(1.0, len(self.phases) * 0.15 + 0.3)
        return min(base, 1.0)


@dataclass
class Commentarius:
    """A commentary written in Caesar's style."""
    title: str
    subject: str
    entries: Tuple[str, ...]
    factual_claims: Tuple[str, ...]
    stylistic_features: Tuple[str, ...]


@dataclass
class LegionData:
    """Data about a Roman legion."""
    number: int
    name: str
    soldiers_count: int
    campaigns: Tuple[str, ...]
    key_battles: Tuple[str, ...]


@dataclass
class GeographicAssessment:
    """Assessment of a geographic region."""
    region: str
    strategic_value: float
    resources: Tuple[str, ...]
    population: str
    key_features: Tuple[str, ...]


@dataclass
class PoliticalReform:
    """A political or social reform."""
    name: str
    year_enacted: int
    key_provisions: Tuple[str, ...]
    opposition_faced: Tuple[str, ...]
    lasting_impact: str


@dataclass
class CalendarReform:
    """The Julian calendar reform."""
    year_enacted: int
    months_redistributed: Tuple[str, ...]
    leap_year_rule: str
    accuracy_improvement: float


# =============================================================================
# TYPING CONSTRUCTS
# =============================================================================

T = TypeVar('T')


class CampaignPlanner:
    """Plan military campaigns in Caesarian style."""
    def __init__(self):
        self.plans: List[StrategicPlan] = []

    def create_plan(self, objective: str, phases: Tuple[str, ...],
                   resources: Tuple[str, ...], timeline: int) -> StrategicPlan:
        plan = StrategicPlan(
            objective=objective,
            phases=phases,
            resource_requirements=resources,
            timeline_months=timeline,
            contingency_plans=("Alternate approach", "Negotiation fallback")
        )
        self.plans.append(plan)
        return plan

    def evaluate_plan(self, plan: StrategicPlan) -> Dict[str, Any]:
        feasibility = plan.feasibility_score()
        return {
            "objective": plan.objective,
            "feasibility": feasibility,
            "phases": len(plan.phases),
            "timeline": plan.timeline_months,
            "recommendation": "Execute" if feasibility >= 0.7 else "Modify"
        }


class BattleAnalyzer:
    """Analyze individual battles."""
    def __init__(self):
        self.battles: List[BattleRecord] = []

    def add_battle(self, battle: BattleRecord) -> None:
        self.battles.append(battle)

    def get_victories(self) -> List[BattleRecord]:
        return [b for b in self.battles if b.is_victory()]

    def casualties_ratio(self, battle: BattleRecord) -> float:
        if "3:1" in battle.casualties_ratio:
            return 3.0
        elif "2:1" in battle.casualties_ratio:
            return 2.0
        return 1.0

    def most_significant(self) -> Optional[BattleRecord]:
        if not self.battles:
            return None
        return max(self.battles, key=lambda b: b.strategic_significance)


class CommentaryWriter:
    """Write commentaries in Caesar's style."""
    def __init__(self):
        self.commentaries: List[Commentarius] = []

    def write_entry(self, subject: str, events: Tuple[str, ...],
                   facts: Tuple[str, ...], style: Tuple[str, ...]) -> Commentarius:
        commentary = Commentarius(
            title=f"Commentary on {subject}",
            subject=subject,
            entries=events,
            factual_claims=facts,
            stylistic_features=style
        )
        self.commentaries.append(commentary)
        return commentary

    def get_entries_summary(self, commentary: Commentarius) -> str:
        return f"{len(commentary.entries)} entries covering {commentary.subject}"


class LegionManager:
    """Manage and track legion data."""
    def __init__(self):
        self.legions: Dict[int, LegionData] = {}

    def add_legion(self, number: int, name: str, count: int,
                  campaigns: Tuple[str, ...], battles: Tuple[str, ...]) -> None:
        self.legions[number] = LegionData(number, name, count, campaigns, battles)

    def get_legion(self, number: int) -> Optional[LegionData]:
        return self.legions.get(number)

    def legions_in_campaign(self, campaign_name: str) -> List[LegionData]:
        return [l for l in self.legions.values() if campaign_name in l.campaigns]


class GeographicStrategist:
    """Assess geographic strategic value."""
    def __init__(self):
        self.assessments: Dict[str, GeographicAssessment] = {}

    def assess(self, region: str, value: float,
              resources: Tuple[str, ...], population: str,
              features: Tuple[str, ...]) -> GeographicAssessment:
        assessment = GeographicAssessment(region, value, resources, population, features)
        self.assessments[region] = assessment
        return assessment

    def compare_regions(self, region1: str, region2: str) -> str:
        r1 = self.assessments.get(region1)
        r2 = self.assessments.get(region2)
        if not r1 or not r2:
            return "Region not assessed"
        if r1.strategic_value > r2.strategic_value:
            return f"{region1} has higher strategic value"
        elif r2.strategic_value > r1.strategic_value:
            return f"{region2} has higher strategic value"
        return "Equal strategic value"


class ReformAnalyzer:
    """Analyze political reforms."""
    def __init__(self):
        self.reforms: List[PoliticalReform] = []

    def add_reform(self, name: str, year: int,
                  provisions: Tuple[str, ...],
                  opposition: Tuple[str, ...],
                  impact: str) -> None:
        self.reforms.append(PoliticalReform(name, year, provisions, opposition, impact))

    def reforms_by_year(self, start_year: int, end_year: int) -> List[PoliticalReform]:
        return [r for r in self.reforms if start_year <= r.year_enacted <= end_year]

    def most_impactful(self) -> Optional[PoliticalReform]:
        if not self.reforms:
            return None
        return max(self.reforms, key=lambda r: len(r.key_provisions))


class CalendarSystem:
    """Julian calendar implementation."""
    def __init__(self):
        self.reform = CalendarReform(
            year_enacted=-45,
            months_redistributed=("January", "February", "March", "April",
                                  "May", "June", "July", "August",
                                  "September", "October", "November", "December"),
            leap_year_rule="Every 4 years",
            accuracy_improvement=0.0075
        )

    def days_in_month(self, month: int) -> int:
        if month in (1, 3, 5, 7, 8, 10, 12):
            return 31
        elif month in (4, 6, 9, 11):
            return 30
        return 29 if month != 2 else 28

    def is_leap_year(self, year: int) -> bool:
        return year % 4 == 0


# =============================================================================
# MAIN CLASS
# =============================================================================

class CaesarSystem:
    """
    Julius Caesar's military and political system.

    Implements:
    - Military campaign planning and analysis
    - Battle record management
    - Commentary writing in Caesarian style
    - Legion tracking and management
    - Geographic strategic assessment
    - Political reform analysis
    - Calendar reform implementation
    """

    def __init__(self):
        self.campaigns: List[MilitaryCampaign] = []
        self.battle_analyzer = BattleAnalyzer()
        self.commentary_writer = CommentaryWriter()
        self.legion_manager = LegionManager()
        self.geo_strategist = GeographicStrategist()
        self.reform_analyzer = ReformAnalyzer()
        self.calendar = CalendarSystem()
        self.planner = CampaignPlanner()

        self._initialize_campaigns()
        self._initialize_legions()
        self._initialize_reforms()
        self._initialize_geography()

    def _initialize_campaigns(self) -> None:
        self.campaigns = [
            MilitaryCampaign("Gallic Wars", -58, -50, CampaignType.CONQUEST,
                           ("Bibracte", "Vercingetorix", "Alesia"),
                           "Complete conquest of Gaul",
                           "Conquest expanded Roman territory significantly"),
            MilitaryCampaign("Civil War", -49, -45, CampaignType.DEFENSE,
                           ("Pharsalus", "Thapsus", "Munda"),
                           "Victory over Pompey and opponents",
                           "Established sole rule"),
            MilitaryCampaign("Egyptian Campaign", -48, -47, CampaignType.EXPANSION,
                           ("Nile Delta",),
                           "Alliance with Cleopatra",
                           "Secured eastern trade routes"),
        ]

        battles = [
            BattleRecord("Battle of Alesia", -52, "Alesia", "Gauls",
                        60000, 80000, "Decisive Roman victory",
                        "3:1 casualties in Roman favor", 0.95),
            BattleRecord("Battle of Pharsalus", -48, "Pharsalus", "Pompey's forces",
                        40000, 70000, "Decisive Caesarian victory",
                        "5:1 casualties in Roman favor", 0.98),
            BattleRecord("Battle of Zela", -47, "Zela", "Pontic forces",
                        20000, 50000, "Quick Roman victory",
                        "Massive enemy casualties", 0.75),
            BattleRecord("Battle of Alesia", -52, "Alesia", "Gauls",
                        60000, 80000, "Roman victory through siege",
                        "Enemy casualties higher", 0.90),
        ]
        for battle in battles:
            self.battle_analyzer.add_battle(battle)

    def _initialize_legions(self) -> None:
        self.legion_manager.add_legion(10, "Equestris", 6000,
                                       ("Gallic Wars",), ("Alesia", "Pharsalus"))
        self.legion_manager.add_legion(13, "Gemina", 6000,
                                       ("Gallic Wars", "Civil War"), ("Alesia", "Pharsalus"))
        self.legion_manager.add_legion(7, "Claudia", 6000,
                                       ("Gallic Wars",), ("Alesia",))

    def _initialize_reforms(self) -> None:
        self.reform_analyzer.add_reform(
            "Lex Julia",
            -59,
            ("Land redistribution", "Corn dole"),
            ("Senatorial opposition",),
            "Benefited many citizens"
        )
        self.reform_analyzer.add_reform(
            "Calendar Reform",
            -45,
            ("365.25 day year", "Leap year system"),
            ("Conservative resistance",),
            "Lasted 1500 years with minor changes"
        )

    def _initialize_geography(self) -> None:
        self.geo_strategist.assess("Gaul", 0.9,
                                  ("Population", "Rich land", "Iron"),
                                  "Millions",
                                  ("多个部落", "River systems", "Forests"))
        self.geo_strategist.assess("Britannia", 0.6,
                                  ("Tin", "Pearls"),
                                  "Hundreds of thousands",
                                  ("Island", "Uncharted"))

    def plan_campaign(self, objective: str, phases: Tuple[str, ...],
                     resources: Tuple[str, ...], timeline: int) -> StrategicPlan:
        return self.planner.create_plan(objective, phases, resources, timeline)

    def get_campaign(self, name: str) -> Optional[MilitaryCampaign]:
        for c in self.campaigns:
            if c.name.lower() in name.lower():
                return c
        return None

    def analyze_battle(self, battle_name: str) -> Optional[BattleRecord]:
        for b in self.battle_analyzer.battles:
            if b.name == battle_name:
                return b
        return None

    def write_commentary(self, subject: str, events: Tuple[str, ...],
                        facts: Tuple[str, ...]) -> Commentarius:
        style = ("Third person", "Objective tone", "Strategic focus", "No personal vanity")
        return self.commentary_writer.write_entry(subject, events, facts, style)

    def get_legion(self, number: int) -> Optional[LegionData]:
        return self.legion_manager.get_legion(number)

    def assess_geographic_value(self, region: str) -> Optional[GeographicAssessment]:
        return self.geo_strategist.assessments.get(region)

    def get_reforms(self, start_year: int, end_year: int) -> List[PoliticalReform]:
        return self.reform_analyzer.reforms_by_year(start_year, end_year)


# =============================================================================
# DEMO
# =============================================================================

def demo() -> None:
    print("=" * 70)
    print("JULIUS CAESAR: MILITARY GENIUS AND POLITICAL REFORMER")
    print("-100 to -44 BCE | Roman General | Dictator | Reformer")
    print("=" * 70)

    system = CaesarSystem()

    print("\n1. MILITARY CAMPAIGNS")
    print("-" * 40)
    for campaign in system.campaigns:
        print(f"  {campaign.name} ({campaign.year_start} to {campaign.year_end})")
        print(f"    Type: {campaign.campaign_type.name}")
        print(f"    Battles: {', '.join(campaign.battles)}")
        print(f"    Outcome: {campaign.outcome}")
        print()

    print("\n2. BATTLE ANALYSIS")
    print("-" * 40)
    print(f"  Total battles recorded: {len(system.battle_analyzer.battles)}")
    victories = system.battle_analyzer.get_victories()
    print(f"  Victories: {len(victories)}")
    most_sig = system.battle_analyzer.most_significant()
    if most_sig:
        print(f"  Most significant: {most_sig.name} ({most_sig.strategic_significance:.2f})")

    print("\n3. LEGION MANAGEMENT")
    print("-" * 40)
    for num in [10, 13, 7]:
        legion = system.get_legion(num)
        if legion:
            print(f"  Legion {legion.number} {legion.name}:")
            print(f"    Soldiers: {legion.soldiers_count}")
            print(f"    Campaigns: {', '.join(legion.campaigns)}")
            print(f"    Key battles: {', '.join(legion.key_battles)}")

    print("\n4. STRATEGIC PLANNING")
    print("-" * 40)
    plan = system.plan_campaign(
        "Conquer Britannia",
        ("Reconnaissance", "Landing", "Consolidation", "Expansion"),
        ("100 ships", "2 legions", "Supplies for 6 months"),
        12
    )
    print(f"  Plan: {plan.objective}")
    print(f"  Phases: {', '.join(plan.phases)}")
    print(f"  Timeline: {plan.timeline_months} months")
    evaluation = system.planner.evaluate_plan(plan)
    print(f"  Feasibility: {evaluation['feasibility']:.2f}")
    print(f"  Recommendation: {evaluation['recommendation']}")

    print("\n5. COMMENTARY WRITING")
    print("-" * 40)
    commentary = system.write_commentary(
        "Gallic Wars",
        ("第一年: 高卢征服开始", "第二年: 维钦托利克斯起义", "第三年: 围困阿莱西亚"),
        ("Caesar captured Alesia", "Vercingetorix surrendered", "Gaul fully conquered")
    )
    print(f"  Title: {commentary.title}")
    print(f"  Entries: {len(commentary.entries)}")
    print(f"  Style: {', '.join(commentary.stylistic_features)}")
    summary = system.commentary_writer.get_entries_summary(commentary)
    print(f"  Summary: {summary}")

    print("\n6. GEOGRAPHIC STRATEGIC ASSESSMENT")
    print("-" * 40)
    for region in ["Gaul", "Britannia"]:
        assess = system.assess_geographic_value(region)
        if assess:
            print(f"  {region}:")
            print(f"    Strategic value: {assess.strategic_value:.2f}")
            print(f"    Resources: {', '.join(assess.resources)}")
            print(f"    Population: {assess.population}")
    comparison = system.geo_strategist.compare_regions("Gaul", "Britannia")
    print(f"  Comparison: {comparison}")

    print("\n7. POLITICAL REFORMS")
    print("-" * 40)
    reforms = system.get_reforms(-60, -40)
    print(f"  Reforms in period: {len(reforms)}")
    for reform in reforms:
        print(f"  - {reform.name} ({reform.year_enacted}): {reform.key_provisions[0]}")
    impactful = system.reform_analyzer.most_impactful()
    if impactful:
        print(f"  Most impactful: {impactful.name}")

    print("\n8. CALENDAR REFORM")
    print("-" * 40)
    cal = system.calendar
    print(f"  Year enacted: {cal.reform.year_enacted}")
    print(f"  Leap year rule: {cal.reform.leap_year_rule}")
    print(f"  Accuracy improvement: {cal.reform.accuracy_improvement:.4f}")
    print(f"  45 BCE was leap year: {cal.is_leap_year(-45)}")
    print(f"  44 BCE was leap year: {cal.is_leap_year(-44)}")
    print(f"  Days in July: {cal.days_in_month(7)}")
    print(f"  Days in February (non-leap): {cal.days_in_month(2)}")

    print("\n9. DECISIVE ACTION ANALYSIS")
    print("-" * 40)
    decisions = [
        ("Crossing the Rubicon", -49, "Defiance of Senate", 0.95),
        ("Appointment as dictator", -49, "Emergency powers", 0.70),
        ("Battle of Pharsalus", -48, "Engage Pompey in Greece", 0.90),
        ("Pursuit of Ptolemy", -47, "Continue after Egypt", 0.60),
        ("Crossing to Britain", -55, "First invasion attempt", 0.75),
    ]
    for name, year, context, impact in decisions:
        print(f"  {name} (-{abs(year)}): {context}")
        print(f"    Historical impact: {impact:.2f}")

    print("\n10. CAESAR'S WRITINGS")
    print("-" * 40)
    writings = [
        ("Commentarii de Bello Gallico", "Gallic Wars", "Military history", -50),
        ("Commentarii de Bello Civili", "Civil War", "Conflict with Pompey", -45),
        ("Anti-Cato", "Critique of Cato", "Political pamphlet", -45),
    ]
    for title, subject, genre, year in writings:
        print(f"  {title}:")
        print(f"    Subject: {subject}")
        print(f"    Genre: {genre}")
        print(f"    Written: c. {-year} BCE")

    print("\n" + "=" * 70)
    print("JULIUS CAESAR SYSTEM COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    demo()

class CampaignRouteCalculator:
    """Calculate routes and distances for military campaigns."""
    def __init__(self):
        self.locations: Dict[str, Tuple[int, int]] = {}

    def add_location(self, name: str, lat: int, lon: int) -> None:
        self.locations[name] = (lat, lon)

    def distance(self, loc1: str, loc2: str) -> float:
        if loc1 not in self.locations or loc2 not in self.locations:
            return 0.0
        lat1, lon1 = self.locations[loc1]
        lat2, lon2 = self.locations[loc2]
        return ((lat2 - lat1)**2 + (lon2 - lon1)**2)**0.5

    def route_distance(self, stops: List[str]) -> float:
        total = 0.0
        for i in range(len(stops) - 1):
            total += self.distance(stops[i], stops[i + 1])
        return total


class BattleOutcomePredictor:
    """Predict battle outcomes based on troop factors."""
    def __init__(self):
        self.factors = {
            "numerical_superiority": 0.25,
            "experience": 0.30,
            "morale": 0.25,
            "terrain": 0.10,
            "leadership": 0.10
        }

    def predict_outcome(self, attacker_factors: Dict[str, float],
                       defender_factors: Dict[str, float]) -> Dict[str, Any]:
        attacker_score = sum(
            attacker_factors.get(k, 0.5) * v
            for k, v in self.factors.items()
        )
        defender_score = sum(
            defender_factors.get(k, 0.5) * v
            for k, v in self.factors.items()
        )
        return {
            "attacker_score": attacker_score,
            "defender_score": defender_score,
            "predicted_winner": "attacker" if attacker_score > defender_score else "defender",
            "confidence": abs(attacker_score - defender_score)
        }


class PoliticalReformAnalyzer:
    """Analyze Caesar's political reforms."""
    def __init__(self):
        self.reforms: List[Dict[str, Any]] = []

    def add_reform(self, name: str, year: int, description: str,
                  opposition: List[str], outcome: str) -> None:
        self.reforms.append({
            "name": name,
            "year": year,
            "description": description,
            "opposition": opposition,
            "outcome": outcome
        })

    def reforms_by_year(self, year: int) -> List[Dict[str, Any]]:
        return [r for r in self.reforms if r["year"] == year]

    def successful_reforms(self) -> List[Dict[str, Any]]:
        return [r for r in self.reforms if r["outcome"] == "enacted"]


class ConspiracyMemberTracker:
    """Track members of conspiracies against Caesar."""
    def __init__(self):
        self.members: Dict[str, Dict[str, Any]] = {}

    def add_member(self, name: str, role: str, motivation: str,
                  fate: str, prior_relationship: str) -> None:
        self.members[name] = {
            "role": role,
            "motivation": motivation,
            "fate": fate,
            "prior_relationship": prior_relationship
        }

    def conspirators_by_role(self, role: str) -> List[str]:
        return [name for name, info in self.members.items()
                if info["role"] == role]

    def survived_assassination(self) -> List[str]:
        return [name for name, info in self.members.items()
                if "survived" in info["fate"].lower()]


class GallicTribeDatabase:
    """Database of Gallic tribes encountered by Caesar."""
    def __init__(self):
        self.tribes: Dict[str, Dict[str, Any]] = {}

    def add_tribe(self, name: str, location: str, population: int,
                 military_strength: int, allies: List[str],
                 resistance_level: str) -> None:
        self.tribes[name] = {
            "location": location,
            "population": population,
            "military": military_strength,
            "allies": allies,
            "resistance": resistance_level
        }

    def tribe_info(self, name: str) -> Optional[Dict[str, Any]]:
        return self.tribes.get(name)

    def most_resistant(self) -> List[str]:
        return [name for name, info in self.tribes.items()
                if info["resistance"] == "high"]


class RomanCalendarAdjuster:
    """Work with the Roman calendar reforms."""
    def __init__(self):
        self.months = {
            "January": 31, "February": 28, "March": 31,
            "April": 30, "May": 31, "June": 30,
            "July": 31, "August": 31, "September": 30,
            "October": 31, "November": 30, "December": 31
        }

    def days_in_year(self) -> int:
        return sum(self.months.values())

    def convert_to_julian_day(self, day: int, month: str, year: int) -> int:
        if month not in self.months:
            return 0
        month_idx = list(self.months.keys()).index(month)
        total_days = sum(list(self.months.values())[:month_idx]) + day
        total_days += (year - 1) * 365
        total_days += (year - 1) // 4
        return total_days


class CommentariiAnalyzer:
    """Analyze Caesar's Commentarii."""
    def __init__(self):
        self.books = {
            "De Bello Gallico": 8,
            "De Bello Civili": 3,
            "De Bello Africo": 1,
            "De Bello Alexandrino": 1,
            "De Bello Hispaniensi": 1
        }

    def total_books(self) -> int:
        return sum(self.books.values())

    def campaigns_in_work(self, work: str) -> int:
        return self.books.get(work, 0)


class MilitaryRankStructure:
    """Model Roman military rank structure."""
    def __init__(self):
        self.ranks = [
            "Legatus Augusti Pro Praetore",
            "Legatus Legionis",
            "Tribunus Militum",
            "Praefectus Equitum",
            "Centurio",
            "Optio",
            "Miles"
        ]

    def rank_level(self, rank: str) -> int:
        try:
            return self.ranks.index(rank)
        except ValueError:
            return -1

    def rank_above(self, rank: str) -> Optional[str]:
        level = self.rank_level(rank)
        if 0 < level < len(self.ranks):
            return self.ranks[level - 1]
        return None


class SiegeWeaponCalculator:
    """Calculate siege weapon specifications."""
    def __init__(self):
        self.weapons: Dict[str, Dict[str, Any]] = {}

    def add_weapon(self, name: str, range_meters: int,
                  damage: int, reload_time_seconds: int,
                  crew_size: int) -> None:
        self.weapons[name] = {
            "range": range_meters,
            "damage": damage,
            "reload": reload_time_seconds,
            "crew": crew_size
        }

    def effectiveness(self, weapon_name: str) -> float:
        if weapon_name not in self.weapons:
            return 0.0
        w = self.weapons[weapon_name]
        return (w["range"] / 100 * 0.3 +
                w["damage"] / 50 * 0.4 +
                (30 / w["reload"]) * 0.3)


class ConquestTimelineBuilder:
    """Build timeline of Gallic conquest."""
    def __init__(self):
        self.events: List[Dict[str, Any]] = []

    def add_event(self, year: int, campaign: str,
                 battles: List[str], outcome: str,
                 territory_gained: str) -> None:
        self.events.append({
            "year": year,
            "campaign": campaign,
            "battles": battles,
            "outcome": outcome,
            "territory": territory_gained
        })

    def events_by_year(self, year: int) -> List[Dict[str, Any]]:
        return [e for e in self.events if e["year"] == year]

    def total_conquests(self) -> int:
        return len([e for e in self.events if "victory" in e["outcome"].lower()])


if __name__ == "__main__":
    demo()


class DictatorshipPowerAnalyzer:
    """Analyze powers exercised during dictatorship."""
    def __init__(self):
        self.powers = {
            "military_command": "Supreme command of armies",
            "legislative_initiative": "Ability to propose laws",
            "senate_control": "Control over Senate proceedings",
            "judicial_authority": "Final judicial authority",
            "provincial_governance": "Governor of all provinces",
            "tribunician_power": "Protection of plebeians"
        }

    def power_scope(self, power_name: str) -> Optional[str]:
        return self.powers.get(power_name)

    def all_powers(self) -> Dict[str, str]:
        return self.powers


class triumvirateAnalyzer:
    """Analyze the Second Triumvirate."""
    def __init__(self):
        self.members = {
            "Octavian": {"age": 23, "strength": "Political legitimacy"},
            "Mark Antony": {"age": 41, "strength": "Military command"},
            "Lepidus": {"age": 46, "strength": "Infantry loyalty"}
        }

    def member_info(self, name: str) -> Optional[Dict[str, Any]]:
        return self.members.get(name)

    def combined_strengths(self) -> List[str]:
        return [m["strength"] for m in self.members.values()]


class SenateProceedingsRecorder:
    """Record proceedings of Senate under Caesar."""
    def __init__(self):
        self.proceedings: List[Dict[str, Any]] = []

    def add_proceeding(self, date: str, topic: str,
                      speakers: List[str], outcome: str) -> None:
        self.proceedings.append({
            "date": date,
            "topic": topic,
            "speakers": speakers,
            "outcome": outcome
        })

    def proceedings_about(self, topic: str) -> List[Dict[str, Any]]:
        return [p for p in self.proceedings if topic.lower() in p["topic"].lower()]


class RomanCitizenshipGranter:
    """Track citizenship grants by Caesar."""
    def __init__(self):
        self.grants: List[Dict[str, str]] = []

    def add_grant(self, recipient: str, original_city: str,
                 year: int, reason: str) -> None:
        self.grants.append({
            "recipient": recipient,
            "origin": original_city,
            "year": year,
            "reason": reason
        })

    def grants_by_origin(self, city: str) -> List[Dict[str, str]]:
        return [g for g in self.grants if g["origin"] == city]

    def total_grants(self) -> int:
        return len(self.grants)


class BreadCircusCalculator:
    """Calculate bread distribution metrics."""
    def __init__(self):
        self.population_estimate = 400000
        self.bread_allocation_per_person = 0.5

    def total_bread_needed(self) -> float:
        return self.population_estimate * self.bread_allocation_per_person

    def daily_consumption(self, grain_modifier: float = 1.0) -> float:
        return self.total_bread_needed() * grain_modifier


class BuildingProjectTracker:
    """Track Caesar's building projects."""
    def __init__(self):
        self.projects: List[Dict[str, Any]] = []

    def add_project(self, name: str, project_type: str,
                   completion_year: int, cost_sestertii: int,
                   significance: str) -> None:
        self.projects.append({
            "name": name,
            "type": project_type,
            "year": completion_year,
            "cost": cost_sestertii,
            "significance": significance
        })

    def projects_by_type(self, ptype: str) -> List[Dict[str, Any]]:
        return [p for p in self.projects if p["type"] == ptype]


class RomanNameAnalyzer:
    """Analyze Roman naming conventions."""
    def __init__(self):
        self.praenomen_list = ["Gaius", "Lucius", "Marcus", "Quintus", "Publius"]
        self.nomen_list = ["Julius", "Claudii", "Cornelius", "Aemilius"]
        self.cognomen_examples = ["Caesar", "Sulla", "Cato", "Africanus"]

    def full_name_parts(self, praenomen: str, nomen: str, cognomen: str) -> List[str]:
        return [praenomen, nomen, cognomen]

    def is_patrician_name(self, nomen: str) -> bool:
        return nomen in self.nomen_list


class ForumUsageAnalyzer:
    """Analyze Roman Forum usage patterns."""
    def __init__(self):
        self.activities = {
            "political": ["elections", "speeches", "voting"],
            "legal": ["trials", "lawyers", "courts"],
            "commercial": ["trade", "shops", "banking"],
            "religious": ["sacrifices", "temples", "priests"],
            "social": ["meetings", "greetings", "news"]
        }

    def activities_by_area(self, area: str) -> List[str]:
        return self.activities.get(area, [])


class MilitaryFormationAnalyzer:
    """Analyze Roman military formations."""
    def __init__(self):
        self.formations = {
            "testudo": {"purpose": "Siege defense", "units": 50},
            "triplex_acies": {"purpose": "Battle formation", "units": 3000},
            "cuneus": {"purpose": "Breaking enemy lines", "units": 500},
            "orbis": {"purpose": "Circular defense", "units": 200}
        }

    def formation_info(self, name: str) -> Optional[Dict[str, Any]]:
        return self.formations.get(name)


class CaesarQuoteAnalyzer:
    """Analyze famous quotes attributed to Caesar."""
    def __init__(self):
        self.quotes = {
            "Veni, vidi, vici": {"context": "Zela victory", "year": -47},
            "Alea iacta est": {"context": "Crossing Rubicon", "year": -49},
            "Et tu, Brute?": {"context": "Assassination", "year": -44},
            "Tu quoque, Brute?": {"context": "Assassination", "year": -44}
        }

    def quote_info(self, quote: str) -> Optional[Dict[str, Any]]:
        return self.quotes.get(quote)

    def all_quotes(self) -> List[str]:
        return list(self.quotes.keys())


if __name__ == "__main__":
    demo()
