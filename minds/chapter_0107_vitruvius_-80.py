"""
Chapter 107: Vitruvius
=======================
Figure 107: Vitruvius (c. 80–15 BCE)
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 6 Minds 101 - 120 Available on Amazon https://www.amazon.com/dp/B0HF7G6JJD
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 107: Vitruvius (-80 to -15 BCE)
================================================================================
Domain: Architecture, Engineering

Selection Rationale:
    Roman architect and engineer; wrote De Architectura — the only
    surviving treatise on ancient architecture; established the triad
    firmitas, utilitas, venustas (strength, utility, beauty) as the
    principles of architecture; described the design of ballistae
    and siege engines; influenced Renaissance architecture through
    his concept of perfect proportion and the Vitruvian Man.

Key Belief About Mind:
    Architecture must be strong (firmitas), useful (utilitas), and
    beautiful (venustas); proportion and harmony in design mirror
    the proportions of the human body; engineering and aesthetics
    are unified in the trained architect's mind; practical military
    knowledge is inseparable from architectural understanding.

Agitation Relevance:
    Vitruvius = design constraints as optimization criteria; the
    Vitruvian triad as multi-objective optimization; proportion
    as aesthetic constraint; Vitruvian Man = anthropomorphic
    design principle; interdisciplinarity of engineering and
    aesthetics; architectural knowledge representation.

Sources:
    - Vitruvius, De Architectura (Ten Books on Architecture)
    - Rykwert (1996), 'The Unability of Vitruvius'
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
from fractions import Fraction
import math
import json
import copy


# =============================================================================
# ENUMS
# =============================================================================

class ArchitecturalOrder(Enum):
    """The three classical orders."""
    DORIC = auto()
    IONIC = auto()
    CORINTHIAN = auto()


class BuildingType(Enum):
    """Types of buildings."""
    TEMPLE = auto()
    HOUSE = auto()
    BATHS = auto()
    THEATER = auto()
    FORUM = auto()
    AQUEDUCT = auto()


class DesignPrinciple(Enum):
    """Core design principles from Vitruvius."""
    FIRMITY = auto()      # strength/durability
    UTILITY = auto()      # functionality
    VENUSTY = auto()      # beauty/elegance
    PROPORTION = auto()   # harmony of parts
    SYMMETRY = auto()     # balanced composition


class EngineeringDevice(Enum):
    """Types of military/engineering devices."""
    BALLISTA = auto()
    CATAPULT = auto()
    RAM = auto()
    SCRIPICULA = auto()
    HOIST = auto()


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass(frozen=True)
class ProportionalSystem:
    """A system of proportions for design."""
    name: str
    ratio: Fraction
    application: str
    aesthetic_principle: str


@dataclass
class BuildingDesign:
    """A building design specification."""
    building_type: BuildingType
    dimensions: Tuple[int, int, int]  # length, width, height
    order: Optional[ArchitecturalOrder]
    principles: Tuple[DesignPrinciple, ...]
    materials: Tuple[str, ...]

    def calculate_volume(self) -> int:
        return self.dimensions[0] * self.dimensions[1] * self.dimensions[2]

    def strength_score(self) -> float:
        base = 0.5
        if DesignPrinciple.FIRMITY in self.principles:
            base += 0.3
        if DesignPrinciple.PROPORTION in self.principles:
            base += 0.1
        return min(base, 1.0)


@dataclass
class ColumnSpecification:
    """Specification for a column."""
    order: ArchitecturalOrder
    height_to_diameter_ratio: Fraction
    base_present: bool
    capital_style: str
    fluting: bool

    def validate_proportions(self) -> bool:
        if self.order == ArchitecturalOrder.DORIC:
            return 6 <= float(self.height_to_diameter_ratio) <= 8
        elif self.order == ArchitecturalOrder.IONIC:
            return 8 <= float(self.height_to_diameter_ratio) <= 10
        elif self.order == ArchitecturalOrder.CORINTHIAN:
            return 10 <= float(self.height_to_diameter_ratio) <= 12
        return False


@dataclass
class TempleDesign:
    """Design for a temple."""
    name: str
    columns_per_side: int
    order: ArchitecturalOrder
    stylobate_height: int
    entablature_height: int
    pediment_angle: int
    cella_proportions: Tuple[int, int]

    def calculate_column_height(self) -> int:
        total_exposed = self.stylobate_height + self.entablature_height + 10
        return total_exposed // 2

    def validate_symmetry(self) -> bool:
        return True  # Temples must be symmetric


@dataclass
class MilitaryEngine:
    """A military engine design."""
    engine_type: EngineeringDevice
    power_output: int  # in talents
    range: int  # in feet
    accuracy: float
    reload_time: int  # in minutes

    def effectiveness_score(self) -> float:
        return min(1.0, (self.power_output / 100) * 0.3 +
                   (self.range / 500) * 0.3 +
                   (self.accuracy) * 0.4)


@dataclass
class AqueductSpecification:
    """Specification for an aqueduct."""
    length: int  # feet
    drop: int  # feet over length
    flow_capacity: int  # cubic feet per day
    arch_spacing: int  # feet between arches
    construction_type: str

    def validate_gradient(self) -> bool:
        gradient = Fraction(self.drop, self.length)
        return float(gradient) < Fraction(1, 100)

    def calculate_arch_count(self) -> int:
        return self.length // self.arch_spacing


@dataclass
class TheaterDesign:
    """Design for a theater."""
    cavea_radius: int
    orchestra_radius: int
    stage_width: int
    number_of_seats: int
    acoustic_score: float

    def validate_acoustics(self) -> bool:
        return self.acoustic_score >= 0.75


@dataclass
class HousePlan:
    """Plan for a Roman house."""
    atriol_width: int
    tablinum_position: str
    peristyle_present: bool
    peristyle_columns: int
    total_rooms: int
    elegant_score: float

    def is_elegant(self) -> bool:
        return self.elegant_score >= 0.7


@dataclass
class UrbanPlan:
    """Plan for a city/urban area."""
    main_axis_length: int
    secondary_axis_length: int
    forum_position: str
    temple_location: str
    bath_locations: Tuple[str, ...]
    aqueduct_present: bool


# =============================================================================
# TYPING CONSTRUCTS
# =============================================================================

T = TypeVar('T')


class ProportionCalculator:
    """Calculate proportions for architectural design."""
    def __init__(self):
        self.systems: List[ProportionalSystem] = []

    def add_system(self, name: str, ratio: Fraction,
                  application: str, principle: str) -> None:
        self.systems.append(ProportionalSystem(name, ratio, application, principle))

    def calculate_for_module(self, module: int, system_name: str) -> Dict[str, int]:
        for sys in self.systems:
            if sys.name == system_name:
                return {
                    "column_diameter": module,
                    "column_height": module * float(sys.ratio),
                    "entablature_height": module * 2,
                    "stylobate_height": module * 3,
                }
        return {}

    def golden_section_apply(self, total_length: int) -> Dict[str, int]:
        phi = (1 + math.sqrt(5)) / 2
        major = int(total_length / phi)
        minor = total_length - major
        return {"major": major, "minor": minor, "total": total_length}


class ArchitecturalValidator:
    """Validate architectural designs."""
    def __init__(self):
        self.violations: List[str] = []

    def validate_building(self, design: BuildingDesign) -> bool:
        self.violations = []
        volume = design.calculate_volume()
        if volume > 1000000:
            self.violations.append("Unreasonably large")
        if DesignPrinciple.FIRMITY not in design.principles:
            self.violations.append("Missing strength principle")
        return len(self.violations) == 0

    def validate_temple(self, temple: TempleDesign) -> bool:
        if temple.columns_per_side < 4:
            self.violations.append("Too few columns")
        if not temple.validate_symmetry():
            self.violations.append("Asymmetric design")
        return len(self.violations) == 0

    def get_violations(self) -> List[str]:
        return self.violations


class ColumnDesigner:
    """Design columns according to Vitruvian principles."""
    def __init__(self):
        self.specifications: List[ColumnSpecification] = []

    def design_doric(self, module: int) -> ColumnSpecification:
        return ColumnSpecification(
            order=ArchitecturalOrder.DORIC,
            height_to_diameter_ratio=Fraction(6, 1),
            base_present=False,
            capital_style="Simple capital with echinus and abacus",
            fluting=True
        )

    def design_ionic(self, module: int) -> ColumnSpecification:
        return ColumnSpecification(
            order=ArchitecturalOrder.IONIC,
            height_to_diameter_ratio=Fraction(9, 1),
            base_present=True,
            capital_style="Volutes on four sides",
            fluting=True
        )

    def design_corinthian(self, module: int) -> ColumnSpecification:
        return ColumnSpecification(
            order=ArchitecturalOrder.CORINTHIAN,
            height_to_diameter_ratio=Fraction(11, 1),
            base_present=True,
            capital_style="Acanthus leaves with volutes",
            fluting=True
        )

    def add_column(self, spec: ColumnSpecification) -> None:
        self.specifications.append(spec)

    def validate_all(self) -> List[bool]:
        return [s.validate_proportions() for s in self.specifications]


class MilitaryEngineer:
    """Design military engines."""
    def __init__(self):
        self.engines: List[MilitaryEngine] = []

    def design_ballista(self, power: int, range_limit: int) -> MilitaryEngine:
        return MilitaryEngine(
            engine_type=EngineeringDevice.BALLISTA,
            power_output=power,
            range=min(range_limit, 600),
            accuracy=0.85,
            reload_time=10
        )

    def design_catapult(self, projectile_weight: int) -> MilitaryEngine:
        power = projectile_weight * 10
        return MilitaryEngine(
            engine_type=EngineeringDevice.CATAPULT,
            power_output=power,
            range=400,
            accuracy=0.75,
            reload_time=15
        )

    def add_engine(self, engine: MilitaryEngine) -> None:
        self.engines.append(engine)

    def most_effective(self) -> Optional[MilitaryEngine]:
        if not self.engines:
            return None
        return max(self.engines, key=lambda e: e.effectiveness_score())


class AqueductBuilder:
    """Design aqueduct systems."""
    def __init__(self):
        self.designs: List[AqueductSpecification] = []

    def design_simple(self, length: int, drop: int) -> AqueductSpecification:
        return AqueductSpecification(
            length=length,
            drop=drop,
            flow_capacity=length * drop // 10,
            arch_spacing=20,
            construction_type="Arched stone construction"
        )

    def add_design(self, design: AqueductSpecification) -> None:
        self.designs.append(design)

    def validate_gradient(self, design: AqueductSpecification) -> bool:
        return design.validate_gradient()


class TheaterAcoustics:
    """Design theaters for optimal acoustics."""
    def __init__(self):
        self.designs: List[TheaterDesign] = []

    def design_optimal(self, capacity: int) -> TheaterDesign:
        cavea_radius = int(math.sqrt(capacity / 3) * 3)
        orchestra_radius = cavea_radius // 3
        return TheaterDesign(
            cavea_radius=cavea_radius,
            orchestra_radius=orchestra_radius,
            stage_width=orchestra_radius * 2,
            number_of_seats=capacity,
            acoustic_score=0.95
        )

    def add_design(self, design: TheaterDesign) -> None:
        self.designs.append(design)

    def best_acoustics(self) -> Optional[TheaterDesign]:
        if not self.designs:
            return None
        return max(self.designs, key=lambda d: d.acoustic_score)


class HouseDesigner:
    """Design Roman houses."""
    def __init__(self):
        self.plans: List[HousePlan] = []

    def design_atrium_house(self, size: str) -> HousePlan:
        if size == "large":
            return HousePlan(
                atriol_width=30,
                tablinum_position="Facing entrance",
                peristyle_present=True,
                peristyle_columns=20,
                total_rooms=25,
                elegant_score=0.9
            )
        elif size == "medium":
            return HousePlan(
                atriol_width=20,
                tablinum_position="Right of entrance",
                peristyle_present=True,
                peristyle_columns=12,
                total_rooms=15,
                elegant_score=0.75
            )
        else:
            return HousePlan(
                atriol_width=15,
                tablinum_position="Opposite entrance",
                peristyle_present=False,
                peristyle_columns=0,
                total_rooms=8,
                elegant_score=0.5
            )

    def add_plan(self, plan: HousePlan) -> None:
        self.plans.append(plan)


class UrbanPlanner:
    """Plan urban spaces."""
    def __init__(self):
        self.plans: List[UrbanPlan] = []

    def design_forum_centered(self, size: int) -> UrbanPlan:
        return UrbanPlan(
            main_axis_length=size * 3,
            secondary_axis_length=size,
            forum_position="Center",
            temple_location="North side of forum",
            bath_locations=("East quarter", "West quarter"),
            aqueduct_present=True
        )

    def add_plan(self, plan: UrbanPlan) -> None:
        self.plans.append(plan)


# =============================================================================
# MAIN CLASS
# =============================================================================

class VitruviusSystem:
    """
    Vitruvian architectural and engineering system.

    Implements:
    - Building design management
    - Proportional calculations
    - Column specifications
    - Temple design
    - Military engine design
    - Aqueduct planning
    - Theater acoustics
    - House planning
    - Urban planning
    """

    def __init__(self):
        self.buildings: List[BuildingDesign] = []
        self.proportion_calc = ProportionCalculator()
        self.validator = ArchitecturalValidator()
        self.column_designer = ColumnDesigner()
        self.military_engineer = MilitaryEngineer()
        self.aqueduct_builder = AqueductBuilder()
        self.theater_acoustics = TheaterAcoustics()
        self.house_designer = HouseDesigner()
        self.urban_planner = UrbanPlanner()

        self._initialize_proportional_systems()
        self._initialize_columns()
        self._initialize_buildings()
        self._initialize_temples()

    def _initialize_proportional_systems(self) -> None:
        systems = [
            ("Modular", Fraction(6, 1), "Columns and entablature", "Module-based proportion"),
            ("Golden", Fraction(1618, 1000), "Facades and principal dimensions", "Harmony through irrational ratio"),
            ("Doric", Fraction(6, 1), "Doric temples", "Strength and masculine beauty"),
            ("Ionic", Fraction(9, 1), "Ionic temples", "Grace and feminine elegance"),
            ("Corinthian", Fraction(11, 1), "Corinthian temples", "Luxury and elaboration"),
        ]
        for name, ratio, application, principle in systems:
            self.proportion_calc.add_system(name, ratio, application, principle)

    def _initialize_columns(self) -> None:
        self.column_designer.design_doric(1)
        self.column_designer.design_ionic(1)
        self.column_designer.design_corinthian(1)

    def _initialize_buildings(self) -> None:
        buildings = [
            BuildingDesign(BuildingType.TEMPLE, (60, 30, 40),
                          ArchitecturalOrder.IONIC,
                          (DesignPrinciple.FIRMITY, DesignPrinciple.UTILITY, DesignPrinciple.VENUSTY),
                          ("Marble", "Stone", "Timber")),
            BuildingDesign(BuildingType.THEATER, (100, 80, 50),
                          None,
                          (DesignPrinciple.VENUSTY, DesignPrinciple.UTILITY, DesignPrinciple.PROPORTION),
                          ("Stone", "Marble")),
            BuildingDesign(BuildingType.AQUEDUCT, (1000, 10, 50),
                          None,
                          (DesignPrinciple.FIRMITY, DesignPrinciple.UTILITY),
                          ("Stone", "Concrete")),
        ]
        for building in buildings:
            self.buildings.append(building)

    def _initialize_temples(self) -> None:
        self.temples: List[TempleDesign] = [
            TempleDesign("Parthenon-style", 8, ArchitecturalOrder.DORIC, 5, 12, 15, (50, 30)),
            TempleDesign("Ionic peripteral", 6, ArchitecturalOrder.IONIC, 4, 10, 17, (40, 25)),
        ]

    def design_temple(self, name: str, columns: int, order: ArchitecturalOrder) -> TempleDesign:
        temple = TempleDesign(name, columns, order, 5, 12, 15, (50, 30))
        self.temples.append(temple)
        return temple

    def design_building(self, building_type: BuildingType, dimensions: Tuple[int, int, int],
                      order: Optional[ArchitecturalOrder] = None) -> BuildingDesign:
        design = BuildingDesign(building_type, dimensions, order,
                              (DesignPrinciple.FIRMITY, DesignPrinciple.UTILITY, DesignPrinciple.VENUSTY),
                              ("Stone",))
        self.buildings.append(design)
        return design

    def design_military_engine(self, engine_type: EngineeringDevice,
                              power: int, range_limit: int) -> MilitaryEngine:
        if engine_type == EngineeringDevice.BALLISTA:
            return self.military_engineer.design_ballista(power, range_limit)
        elif engine_type == EngineeringDevice.CATAPULT:
            return self.military_engineer.design_catapult(power)
        else:
            return MilitaryEngine(engine_type, power, range_limit, 0.7, 15)

    def design_aqueduct(self, length: int, drop: int) -> AqueductSpecification:
        return self.aqueduct_builder.design_simple(length, drop)

    def design_theater(self, capacity: int) -> TheaterDesign:
        return self.theater_acoustics.design_optimal(capacity)

    def design_house(self, size: str) -> HousePlan:
        return self.house_designer.design_atrium_house(size)

    def plan_city(self, size: int) -> UrbanPlan:
        return self.urban_planner.design_forum_centered(size)

    def calculate_proportions(self, module: int, system_name: str) -> Dict[str, int]:
        return self.proportion_calc.calculate_for_module(module, system_name)

    def validate_design(self, design: BuildingDesign) -> Tuple[bool, List[str]]:
        valid = self.validator.validate_building(design)
        return valid, self.validator.get_violations()


# =============================================================================
# DEMO
# =============================================================================

def demo() -> None:
    print("=" * 70)
    print("VITRUVIUS: ROMAN ARCHITECTURE AND ENGINEERING")
    print("c. 80-15 BCE | De Architectura | The Vitruvian Triad")
    print("=" * 70)

    system = VitruviusSystem()

    print("\n1. THE VITRUVIAN TRIAD")
    print("-" * 40)
    principles = [
        ("FIRMITY", "Strength/Durability", "The building must stand firmly"),
        ("UTILITY", "Functionality", "The building must serve its purpose"),
        ("VENUSTY", "Beauty/Elegance", "The building must please the eye"),
    ]
    for code, name, desc in principles:
        print(f"  {code}: {name}")
        print(f"    {desc}")

    print("\n2. PROPORTIONAL SYSTEMS")
    print("-" * 40)
    systems = system.proportion_calc.systems
    for sys in systems:
        print(f"  {sys.name} (ratio: {float(sys.ratio):.3f})")
        print(f"    Application: {sys.application}")
        print(f"    Principle: {sys.aesthetic_principle}")

    print("\n3. GOLDEN SECTION CALCULATION")
    print("-" * 40)
    for length in [100, 200, 500]:
        result = system.proportion_calc.golden_section_apply(length)
        print(f"  Length {length}: major={result['major']}, minor={result['minor']}")

    print("\n4. ARCHITECTURAL ORDERS")
    print("-" * 40)
    orders = [
        ("Doric", 6, "Masculine, simple, strong"),
        ("Ionic", 9, "Graceful, feminine, scroll capitals"),
        ("Corinthian", 11, "Luxurious, elaborate, acanthus"),
    ]
    for name, ratio, desc in orders:
        col = system.column_designer.design_doric(1) if name == "Doric" else \
              system.column_designer.design_ionic(1) if name == "Ionic" else \
              system.column_designer.design_corinthian(1)
        print(f"  {name}: height/diameter ratio = {float(col.height_to_diameter_ratio)}:1")
        print(f"    {desc}")
        print(f"    Capital: {col.capital_style}")
        print(f"    Fluting: {col.fluting}")
        print()

    print("\n5. BUILDING DESIGNS")
    print("-" * 40)
    for building in system.buildings:
        print(f"  {building.building_type.name}")
        print(f"    Dimensions: {building.dimensions}")
        print(f"    Order: {building.order.name if building.order else 'None'}")
        print(f"    Volume: {building.calculate_volume()} cubic feet")
        print(f"    Strength score: {building.strength_score():.2f}")

    print("\n6. TEMPLE DESIGNS")
    print("-" * 40)
    for temple in system.temples:
        print(f"  {temple.name}")
        print(f"    Columns: {temple.columns_per_side} per side")
        print(f"    Order: {temple.order.name}")
        print(f"    Column height: {temple.calculate_column_height()} feet")
        print(f"    Symmetric: {temple.validate_symmetry()}")

    print("\n7. MILITARY ENGINES")
    print("-" * 40)
    ballista = system.design_military_engine(EngineeringDevice.BALLISTA, 80, 500)
    catapult = system.design_military_engine(EngineeringDevice.CATAPULT, 50, 400)
    engines = [ballista, catapult]
    for engine in engines:
        print(f"  {engine.engine_type.name}")
        print(f"    Power: {engine.power_output} talents")
        print(f"    Range: {engine.range} feet")
        print(f"    Accuracy: {engine.accuracy:.2f}")
        print(f"    Reload time: {engine.reload_time} minutes")
        print(f"    Effectiveness: {engine.effectiveness_score():.2f}")
        print()

    print("\n8. AQUEDUCT DESIGN")
    print("-" * 40)
    aqueduct = system.design_aqueduct(5000, 50)
    print(f"  Length: {aqueduct.length} feet")
    print(f"  Drop: {aqueduct.drop} feet")
    print(f"  Gradient valid: {aqueduct.validate_gradient()}")
    print(f"  Arch count: {aqueduct.calculate_arch_count()}")
    print(f"  Flow capacity: {aqueduct.flow_capacity} cubic feet/day")
    print(f"  Construction: {aqueduct.construction_type}")

    print("\n9. THEATER DESIGN WITH ACOUSTICS")
    print("-" * 40)
    theater = system.design_theater(5000)
    print(f"  Capacity: {theater.number_of_seats} seats")
    print(f"  Cavea radius: {theater.cavea_radius} feet")
    print(f"  Orchestra radius: {theater.orchestra_radius} feet")
    print(f"  Stage width: {theater.stage_width} feet")
    print(f"  Acoustic score: {theater.acoustic_score:.2f}")
    print(f"  Acoustically valid: {theater.validate_acoustics()}")

    print("\n10. HOUSE DESIGNS")
    print("-" * 40)
    for size in ["small", "medium", "large"]:
        house = system.design_house(size)
        print(f"  {size.capitalize()} house:")
        print(f"    Atrium width: {house.atriol_width} feet")
        print(f"    Peristyle present: {house.peristyle_present}")
        print(f"    Total rooms: {house.total_rooms}")
        print(f"    Elegant score: {house.elegant_score:.2f}")
        print(f"    Is elegant: {house.is_elegant()}")

    print("\n11. URBAN PLANNING")
    print("-" * 40)
    city = system.plan_city(500)
    print(f"  Main axis: {city.main_axis_length} feet")
    print(f"  Secondary axis: {city.secondary_axis_length} feet")
    print(f"  Forum position: {city.forum_position}")
    print(f"  Temple location: {city.temple_location}")
    print(f"  Bath locations: {', '.join(city.bath_locations)}")
    print(f"  Aqueduct present: {city.aqueduct_present}")

    print("\n12. DESIGN VALIDATION")
    print("-" * 40)
    test_design = system.design_building(BuildingType.FORUM, (200, 150, 40),
                                        ArchitecturalOrder.IONIC)
    valid, violations = system.validate_design(test_design)
    print(f"  Test forum design valid: {valid}")
    if violations:
        print(f"  Violations: {', '.join(violations)}")

    print("\n" + "=" * 70)
    print("VITRUVIUS ARCHITECTURAL SYSTEM COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    demo()

class ArchitecturalOrderDetails:
    """Detailed specifications for each classical order."""
    def __init__(self):
        self.orders = {
            "Doric": {
                "origin": "Peloponnese",
                "column_height_ratio": "6:1",
                "capital": "Simple echinus and abacus",
                "base": "None",
                "entasis": "Slight convex curve",
                "character": "Masculine, severe, simple"
            },
            "Ionic": {
                "origin": "Ionia (Asia Minor)",
                "column_height_ratio": "9:1",
                "capital": "Volutes (scrolls)",
                "base": "Two tori with plinth",
                "entasis": "Gradual taper",
                "character": "Feminine, graceful, ornate"
            },
            "Corinthian": {
                "origin": "Corinth",
                "column_height_ratio": "11:1",
                "capital": "Acanthus leaves and volutes",
                "base": "Three tiers of moldings",
                "entasis": "Subtle curve",
                "character": "Luxurious, elaborate, decorative"
            }
        }

    def get_order_details(self, order_name: str) -> Dict[str, str]:
        return self.orders.get(order_name, {})


class BuildingMaterialProperties:
    """Properties of building materials per Vitruvius."""
    def __init__(self):
        self.materials = {
            "tufa": {"weight": "light", "durability": "moderate", "use": "Domestic walls"},
            "travertine": {"weight": "heavy", "durability": "high", "use": "Public buildings"},
            "marble": {"weight": "medium", "durability": "very high", "use": "Temples, decoration"},
            "concrete": {"weight": "light", "durability": "high", "use": "Vaults, foundations"},
            "brick": {"weight": "light", "durability": "moderate", "use": "Walls, facing"},
            "timber": {"weight": "light", "durability": "low", "use": "Roofs, centering"}
        }

    def material_properties(self, material: str) -> Dict[str, str]:
        return self.materials.get(material, {})


class SiteOrientationPrinciples:
    """Vitruvian principles for building orientation."""
    def __init__(self):
        self.orientations = {
            "temple": {"primary": "East-West axis", "reason": "Sun light for cult statue"},
            "theater": {"primary": "South-facing cavea", "reason": "Acoustic advantage"},
            "house": {"primary": "South exposure", "reason": "Winter warmth"},
            "bath": {"primary": "West-facing", "reason": "Evening bathing"},
            "forum": {"primary": "North-South axis", "reason": "Market not in glare"}
        }

    def recommended_orientation(self, building_type: str) -> str:
        info = self.orientations.get(building_type, {})
        return f"{info.get('primary', 'No recommendation')} — {info.get('reason', '')}"


class VitruvianModuleSystem:
    """The modular system for proportions."""
    def __init__(self):
        self.module_values = {
            "Doric": 1/6,  # Lower diameter module
            "Ionic": 1/8,  # Lower diameter module  
            "Corinthian": 1/10
        }

    def calculate_dimensions(self, order: str, column_diameter: float) -> Dict[str, float]:
        module = self.module_values.get(order, 1/8)
        return {
            "diameter": column_diameter,
            "height": column_diameter / module,
            "capital_height": column_diameter,
            "entablature_height": column_diameter * 2
        }


class WindEffectsAnalyzer:
    """Analyze wind effects on building design."""
    def __init__(self):
        self.wind_regimes = {
            "north": {"name": "Boreas", "effect": "Cold, piercing"},
            "south": {"name": "Notus", "effect": "Warm, moist"},
            "east": {"name": "Eurus", "effect": "Mild, variable"},
            "west": {"name": "Zephyrus", "effect": "Gentle, beneficial"}
        }

    def avoid_winds(self, winds: List[str]) -> str:
        avoid = []
        for wind in winds:
            if wind.lower() in self.wind_regimes:
                avoid.append(wind)
        return f"Avoid winds from: {', '.join(avoid)}"


class DefensiveArchitecturePrinciples:
    """Vitruvian principles for military/defensive architecture."""
    def __init__(self):
        self.principles = {
            "walls": "Thickness equals height/20 or greater",
            "towers": "Project beyond wall face for flanking",
            "gates": "Never face directly the strongest wall",
            "ditches": "Filled with water or sharpened stakes"
        }

    def fortification_design(self) -> Dict[str, str]:
        return self.principles


class WaterSupplyEngineering:
    """Principles for water supply systems."""
    def __init__(self):
        self.water_types = {
            "rain": {"collection": "Roofs and cisterns", "quality": "Moderate"},
            "spring": {"collection": "Aqueduct from highlands", "quality": "Excellent"},
            "river": {"collection": "Intake with settling basins", "quality": "Fair"},
            "well": {"collection": "Subterranean collection", "quality": "Variable"}
        }

    def recommended_source(self) -> str:
        return "Spring water via aqueduct is most healthy"


class SundialAndTimekeeping:
    """Vitruvius on sundials and time measurement."""
    def __init__(self):
        self.time_instruments = {
            "horologium": "Water clock",
            "sciotherium": "Shadow clock",
            "meridian": "Meridian line",
            "nocturnium": "Night time device"
        }

    def time_instrument_description(self, instrument: str) -> str:
        return self.time_instruments.get(instrument, "Unknown time instrument")


class VitruvianAcousticTheory:
    """Vitruvius on theater acoustics."""
    def __init__(self):
        self.acoustic_principles = {
            "vessel_resonance": "Bronze vessels placed under seats to amplify",
            "orchestra_circle": "Sound central distribution point",
            "cavea_shape": "Curved seating for sound distribution",
            "stage_height": "Proper elevation for actor projection"
        }

    def acoustic_design_requirements(self, capacity: int) -> Dict[str, Any]:
        vessels_needed = capacity // 50
        return {
            "bronze_vessels": vessels_needed,
            "orchestra_radius": capacity // 30,
            "cavea_incline": "Steep for sound transmission"
        }


class CityPlanningPrinciples:
    """Urban planning principles from Vitruvius."""
    def __init__(self):
        self.planning_rules = {
            "climate_orientation": " Streets set E-W for climate moderation",
            "healthy_site": "Elevated ground away from marshes",
            "water_access": "Pure water supply essential",
            "military_considerations": "Defensible position preferred",
            "forum_center": "Central civic space"
        }

    def plan_city(self, size: str) -> Dict[str, Any]:
        if size == "large":
            return {"forum_size": 400, "street_width": 40, "blocks": 20}
        elif size == "medium":
            return {"forum_size": 300, "street_width": 30, "blocks": 12}
        else:
            return {"forum_size": 200, "street_width": 20, "blocks": 6}


class DecorativePaintingPrinciples:
    """Vitruvian principles for architectural painting."""
    def __init__(self):
        self.styles = {
            "architectural": "Fanciful columns, perspective frames",
            "landscape": "Gardens, harbors, sacred landscapes",
            "mythological": "Divine scenes, heroic narratives",
            "civic_painting": "Votive offerings, processions"
        }

    def appropriate_style(self, building_type: str) -> str:
        return self.styles.get(building_type, "architectural")


class MechanicalPrinciples:
    """Vitruvius's mechanical and engineering principles."""
    def __init__(self):
        self.mechanical_powers = {
            "lever": "Multiply force applied",
            "windlass": "Hoisting heavy loads",
            "crane": "Vertical lifting",
            "water_lift": "Archimedean screw, water raising",
            "siege_engine": "Ballista, catapult, ram mechanics"
        }

    def mechanical_advantage(self, device: str) -> float:
        advantages = {"lever": 3.0, "windlass": 2.0, "crane": 2.5}
        return advantages.get(device, 1.0)


class VitruvianMathematics:
    """Mathematical principles in Vitruvian architecture."""
    def __init__(self):
        self.mathematical_systems = {
            "arithmetic": "Whole number ratios for proportions",
            "geometry": "Euclidean geometry for planning",
            "harmonic_proportions": "Musical intervals in spatial ratios",
            "astronomy": "Solar alignments for orientation"
        }

    def calculate_proportional_system(self, system_name: str) -> Dict[str, Any]:
        systems = {
            "modular": {"base": "column diameter", "ratio": 6},
            "golden": {"base": "total height", "ratio": 1.618}
        }
        return systems.get(system_name, {})


class BuildingInspectionProtocol:
    """Vitruvian protocol for inspecting buildings."""
    def __init__(self):
        self.inspection_points = [
            "Foundation solidity",
            "Wall construction quality",
            "Roof structure integrity",
            "Waterproofing systems",
            "Ventilation adequacy",
            "Lighting conditions"
        ]

    def inspection_checklist(self) -> List[str]:
        return self.inspection_points


class VitruvianWorkshopOrganization:
    """Organization of the architect's workshop."""
    def __init__(self):
        self.workshop_elements = {
            "drawing_walls": "Large surfaces for scaled drawings",
            "model_storage": "Storage for architectural models",
            "instrument_case": "Measurement and drawing tools",
            "reference_library": "Books, treatises, precedents",
            "sample_materials": "Material samples for client presentation"
        }

    def workshop_requirements(self) -> Dict[str, str]:
        return self.workshop_elements


class VitruvianClientRelationship:
    """Managing the architect-client relationship."""
    def __init__(self):
        self.client_types = {
            "imperial": {"budget": "unlimited", "expectation": "monumental scale"},
            "senatorial": {"budget": "substantial", "expectation": "high quality"},
            "equestrian": {"budget": "moderate", "expectation": "practical elegance"},
            "common": {"budget": "limited", "expectation": "functional design"}
        }

    def client_advice(self, client_type: str) -> str:
        info = self.client_types.get(client_type, {"budget": "unknown", "expectation": "reasonable"})
        return f"For {client_type}: budget={info['budget']}, expectation={info['expectation']}"
