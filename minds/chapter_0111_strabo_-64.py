#!/usr/bin/env python3
"""
Chapter 111: Strabo (64 BCE-24 CE) - Geographic-Historical Cognition Architecture
Figure ID: 111 | Domain: Geography, History | Region: Asia Minor/Rome
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 6 Minds 101 - 120 Available on Amazon https://www.amazon.com/dp/B0HF7G6JJD
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 111: Strabo (-64 to -24 BCE)
================================================================================
Strabo's Geographica is the foundational work of regional geography, combining
direct observation, travel narrative, historical knowledge, and philosophical
synthesis. He traveled from Armenia to Italy and from the Black Sea to Egypt,
producing a 17-book systematic geography of the known world.

This architecture models:
- Spatial reasoning and regional description
- Ethnographic observation and cultural classification
- Integration of history and geography (chorography)
- The concept of oikoumene (inhabited world)
- Environmental determinism in cultural development
- Navigation and cartographic cognition
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
)


# =============================================================================
# ENUMS: Geographic and Historical Classification
# =============================================================================


class ClimateZone(Enum):
    """Classical climate zones affecting human culture."""
    TORRID = auto()      # Extreme heat (Ethiopia, Arabia)
    TROPICAL = auto()    # Hot and humid (India, Egypt lower)
    SUBTROPICAL = auto() # Mediterranean climate
    TEMPERATE = auto()   # Moderate (Greece, Italy)
    COLD = auto()        # Northern regions (Celtic lands, Scythia)
    Arctic = auto()      # Extreme cold (hyperborean regions)


class TerrainType(Enum):
    """Terrain types affecting regional character."""
    COASTAL = auto()     # Maritime influence, trade
    INLAND_PLAIN = auto() # Agricultural, settled
    MOUNTAINOUS = auto()  # Hardy, independent
    DESERT = auto()       # Nomadic, sparse
    RIVER_VALLEY = auto() # Fertile, civilization cradle
    STEPP = auto()        # Nomadic pastoral
    FORESTED = auto()     # Tribal, wild


class CulturalCharacter(Enum):
    """Classical ethnographic character types."""
    CIVILIZED = auto()   # Greek/Roman urban culture
    BARBAROUS = auto()   # Non-Greek but structured society
    SAVAGE = auto()      # Pre-civilized, tribal
    NOMADIC = auto()     # Pastoral wanderers
    MARITIME = auto()    # Seafaring peoples


class GeographicScale(Enum):
    """Scale of geographic description."""
    OIKOUMENE = auto()   # Whole inhabited world
    REGION = auto()      # Major region (e.g., Iberia, Anatolia)
    CHOROS = auto()      # Local area, landscape description
    TOPOS = auto()       # Specific site, city


class HistoricalDepth(Enum):
    """Depth of historical knowledge."""
    MYTHICAL = auto()    # Mythological past
    LEGENDARY = auto()   # Heroic age traditions
    HISTORICAL = auto()  # Documented history
    CONTEMPORARY = auto() # Strabo's own observation


class RegionType(Enum):
    """Major regions of the oikoumene."""
    WESTERN_MEDITERRANEAN = auto()
    EASTERN_MEDITERRANEAN = auto()
    ANATOLIA = auto()
    LEVANT = auto()
    EGYPT_NILE = auto()
    MESOPOTAMIA = auto()
    PERSIA = auto()
    CENTRAL_ASIA = auto()
    INDIA = auto()
    ARABIA = auto()
    CELTIC_LANDS = auto()
    GERMANIC_LANDS = auto()
    SCYTHIA = auto()
    BLACK_SEA_REGION = auto()


# =============================================================================
# TYPE VARIABLES
# =============================================================================

T = TypeVar('T')
U = TypeVar('U')


# =============================================================================
# DATACLASSES: Core Geographic Entities
# =============================================================================


@dataclass(frozen=True)
class Coordinate:
    """Geographic coordinate in classical system."""
    latitude: float   # North (+) / South (-) from equator
    longitude: float  # East (+) / West (-) from Rhodes (prime meridian)
    description: str = ""  # Classical place name

    def distance_to(self, other: Coordinate) -> float:
        """Calculate great-circle distance (simplified)."""
        dlat = abs(self.latitude - other.latitude)
        dlon = abs(self.longitude - other.longitude)
        return math.sqrt(dlat ** 2 + dlon ** 2)


@dataclass
class Place:
    """A specific place in Strabo's geography."""
    name: str
    coordinate: Optional[Coordinate]
    region: RegionType
    terrain: Set[TerrainType] = field(default_factory=set)
    climate: ClimateZone = ClimateZone.SUBTROPICAL
    cultural_character: CulturalCharacter = CulturalCharacter.CIVILIZED
    notable_features: List[str] = field(default_factory=list)
    historical_significance: List[str] = field(default_factory=list)
    strabo_description: str = ""
    strabo_judgment: str = ""  # His character assessment

    def describe_classically(self) -> str:
        """Return a Strabo-style description."""
        features = ", ".join(self.notable_features[:3]) if self.notable_features else "various"
        return f"{self.name}: {self.terrain.name} region, {self.climate.name} climate, inhabited by {self.cultural_character.name.lower()} peoples. Notable for {features}."


@dataclass
class Region:
    """A geographic region combining multiple places and characteristics."""
    name: str
    region_type: RegionType
    places: List[Place] = field(default_factory=list)
    major_cities: List[str] = field(default_factory=list)
    rivers: List[str] = field(default_factory=list)
    mountains: List[str] = field(default_factory=list)
    climate: ClimateZone = ClimateZone.SUBTROPICAL
    terrain: Set[TerrainType] = field(default_factory=set)
    cultural_character: CulturalCharacter = CulturalCharacter.CIVILIZED
    historical_traditions: List[str] = field(default_factory=list)
    strabo_assessment: str = ""

    def total_cities(self) -> int:
        return len(self.major_cities) + sum(1 for p in self.places if hasattr(p, 'is_city') and p.is_city)

    def describe(self) -> str:
        """Strabo-style regional description."""
        cities = ", ".join(self.major_cities[:5])
        river_str = f"Rivers: {', '.join(self.rivers[:3])}" if self.rivers else "No major rivers noted"
        mt_str = f"Mountains: {', '.join(self.mountains[:3])}" if self.mountains else "No major mountains noted"
        return (
            f"{self.name}: A {self.terrain.__iter__().__next__().name.lower()} {self.region_type.name.replace('_', ' ').lower()} "
            f"region of {self.climate.name.lower()} character. "
            f"Major cities include {cities}. {river_str}. {mt_str}. "
            f"Cultural character: {self.cultural_character.name.lower()}. "
            f"{self.strabo_assessment}"
        )


@dataclass
class EthnographicNote:
    """Strabo's ethnographic observation about a people."""
    people_name: str
    region: str
    character_traits: Set[str] = field(default_factory=set)
    customs: List[str] = field(default_factory=list)
    environment_influence: str = ""  # How terrain/climate shapes them
    historical_origin: str = ""
    strabo_quote: str = ""
    reliability: float = 1.0  # Strabo's confidence in this observation


@dataclass
class ChorographicDescription:
    """A chorographic (regional) description in Strabo's style."""
    region_name: str
    scale: GeographicScale
    history: str  # Ancient history relevant to the region
    geography: str  # Physical description
    peoples: List[EthnographicNote] = field(default_factory=list)
    notable_sites: List[str] = field(default_factory=list)
    strabo_text_excerpt: str = ""
    historical_depth: HistoricalDepth = HistoricalDepth.CONTEMPORARY

    def synthesize_description(self) -> str:
        """Produce a synthesized chorographic description."""
        parts = [
            f"Concerning {self.region_name}:",
            self.history[:200] + "..." if len(self.history) > 200 else self.history,
            self.geography[:200] + "..." if len(self.geography) > 200 else self.geography,
            f"Notable sites include {', '.join(self.notable_sites[:5])}." if self.notable_sites else "",
        ]
        return "\n".join(parts)


@dataclass
class Route:
    """A travel route between places."""
    origin: str
    destination: str
    waypoints: List[str] = field(default_factory=list)
    distance_stadia: int  # Ancient stadia (1 stadium ≈ 185m)
    terrain_difficulties: List[str] = field(default_factory=list)
    seasonal_considerations: str = ""
    strabo_route_description: str = ""

    def distance_km(self) -> float:
        return self.distance_stadia * 0.185


# =============================================================================
# COMPONENT CLASSES
# =============================================================================


class EnvironmentalDeterminism:
    """
    Strabo's framework for how environment shapes cultural character.
    Classic ancient theory that climate and terrain influence human nature.
    """

    CLIMATE_CHARACTER_MAP: Dict[ClimateZone, Set[str]] = {
        ClimateZone.TORRID: {"courageous in heat", "sluggish in cold", "dark complexion"},
        ClimateZone.TROPICAL: {"energetic", "superstitious", "emotional"},
        ClimateZone.SUBTROPICAL: {"balanced", "intellectual", "civilized"},
        ClimateZone.TEMPERATE: {"industrious", "brave", "independent"},
        ClimateZone.COLD: {"fierce", "warlike", "hardy"},
        ClimateZone.ARCTIC: {"savage", "simple", "superstitious"},
    }

    TERRAIN_CHARACTER_MAP: Dict[TerrainType, Set[str]] = {
        TerrainType.COASTAL: {"maritime", "trade-oriented", "cosmopolitan"},
        TerrainType.INLAND_PLAIN: {"agricultural", "settled", "conservative"},
        TerrainType.MOUNTAINOUS: {"independent", "brave", "stubborn"},
        TerrainType.DESERT: {"nomadic", "hospitable", "fierce"},
        TerrainType.RIVER_VALLEY: {"civilized", "urban", "prosperous"},
        TerrainType.STEPPE: {"pastoral", "nomadic", "horse-loving"},
        TerrainType.FORESTED: {"tribal", "warlike", "wild"},
    }

    @classmethod
    def predict_character(
        cls,
        climate: ClimateZone,
        terrain: Set[TerrainType],
        region: str
    ) -> Set[str]:
        """Predict cultural character from environmental factors."""
        chars = set()
        chars.update(cls.CLIMATE_CHARACTER_MAP.get(climate, set()))
        for t in terrain:
            chars.update(cls.TERRAIN_CHARACTER_MAP.get(t, set()))
        return chars


class CartographicProjection:
    """Simple map projection for the oikoumene."""

    # Approximate classical coordinates (latitude, longitude)
    CLASSICAL_PLACES: Dict[str, Tuple[float, float]] = {
        "Rhodes": (36.2, 28.0),       # Prime meridian reference
        "Alexandria": (31.2, 29.9),
        "Rome": (41.9, 12.5),
        "Carthage": (36.9, 10.3),
        "Athens": (37.9, 23.7),
        "Constantinople": (41.0, 28.9),
        "Antioch": (36.2, 36.2),
        "Jerusalem": (31.8, 35.2),
        "Babylon": (32.5, 44.4),
        "Indus": (25.0, 67.0),
        "Ganges": (25.0, 87.0),
        "Britain": (54.0, -2.0),
        "Celtic Gaul": (46.0, 4.0),
        "Iberia": (40.0, -4.0),
        "Black Sea": (43.0, 34.0),
    }

    @classmethod
    def get_coordinate(cls, place_name: str) -> Optional[Coordinate]:
        """Get classical coordinate for a place."""
        if place_name in cls.CLASSICAL_PLACES:
            lat, lon = cls.CLASSICAL_PLACES[place_name]
            return Coordinate(latitude=lat, longitude=lon, description=place_name)
        return None

    @classmethod
    def map_distance(cls, place_a: str, place_b: str) -> float:
        """Get distance between two places in classical coordinates."""
        coord_a = cls.get_coordinate(place_a)
        coord_b = cls.get_coordinate(place_b)
        if coord_a and coord_b:
            return coord_a.distance_to(coord_b)
        return 0.0


class OikoumeneBuilder:
    """Build the known world (oikoumene) description."""

    def __init__(self):
        self.regions: Dict[RegionType, Region] = {}
        self.places: Dict[str, Place] = {}
        self.ethnographic_notes: List[EthnographicNote] = []
        self._initialize_oikoumene()

    def _initialize_oikoumene(self) -> None:
        """Initialize the known world with classical knowledge."""
        # Mediterranean core
        italy = Region(
            name="Italy",
            region_type=RegionType.WESTERN_MEDITERRANEAN,
            major_cities=["Rome", "Cumae", "Capua", "Brundisium"],
            rivers=["Tiber", "Amo", "Vulturnus"],
            mountains=["Alps", "Apennines", "Vesuvius"],
            climate=ClimateZone.TEMPERATE,
            terrain={TerrainType.COASTAL, TerrainType.MOUNTAINOUS, TerrainType.INLAND_PLAIN},
            cultural_character=CulturalCharacter.CIVILIZED,
            historical_traditions=["Roman Republic history", "Etruscan origins", "Greek colonization"],
            strabo_assessment="Italy is the most fortunate of all lands.",
        )
        self.regions[RegionType.WESTERN_MEDITERRANEAN] = italy

        # Anatolia
        anatolia = Region(
            name="Anatolia (Asia Minor)",
            region_type=RegionType.ANATOLIA,
            major_cities=["Ephesus", "Smyrna", "Tarsus", "Sinope"],
            rivers=["Halys", "Calycadnus", "Euphrates upper"],
            mountains=["Taurus", "Pontic ranges", "Olympus"],
            climate=ClimateZone.SUBTROPICAL,
            terrain={TerrainType.MOUNTAINOUS, TerrainType.INLAND_PLAIN, TerrainType.COASTAL},
            cultural_character=CulturalCharacter.CIVILIZED,
            historical_traditions=["Hittite past", "Greek cities", "Persian rule"],
            strabo_assessment="Fertile and well-positioned for trade.",
        )
        self.regions[RegionType.ANATOLIA] = anatolia

        # Egypt
        egypt = Region(
            name="Egypt and the Nile",
            region_type=RegionType.EGYPT_NILE,
            major_cities=["Alexandria", "Memphis", "Thebes", "Heliopolis"],
            rivers=["Nile"],
            mountains=["Sinai", "Eastern desert heights"],
            climate=ClimateZone.TROPICAL,
            terrain={TerrainType.RIVER_VALLEY, TerrainType.DESERT},
            cultural_character=CulturalCharacter.CIVILIZED,
            historical_traditions=["Pharaonic civilization", "Persian rule", "Ptolemaic rule"],
            strabo_assessment="Egypt is the gift of the Nile.",
        )
        self.regions[RegionType.EGYPT_NILE] = egypt

        # Mesopotamia
        mesopotamia = Region(
            name="Mesopotamia",
            region_type=RegionType.MESOPOTAMIA,
            major_cities=["Babylon", "Seleucia", "Ctesiphon", "Nineveh"],
            rivers=["Tigris", "Euphrates"],
            mountains=["Zagros"],
            climate=ClimateZone.TROPICAL,
            terrain={TerrainType.RIVER_VALLEY, TerrainType.DESERT},
            cultural_character=CulturalCharacter.CIVILIZED,
            historical_traditions=["Chaldean astronomy", "Assyrian empire", "Persian empire"],
            strabo_assessment="Fertile plain between two great rivers.",
        )
        self.regions[RegionType.MESOPOTAMIA] = mesopotamia

        # Celtic lands
        celtic = Region(
            name="Celtic Lands (Gaul and Britain)",
            region_type=RegionType.CELTIC_LANDS,
            major_cities=["Lutetia (Paris)", "Massalia (Marseille)", "Londinium"],
            rivers=["Rhone", "Seine", "Thames"],
            mountains=["Cevennes", "Alps", "Scottish highlands"],
            climate=ClimateZone.TEMPERATE,
            terrain={TerrainType.FORESTED, TerrainType.INLAND_PLAIN, TerrainType.COASTAL},
            cultural_character=CulturalCharacter.BARBAROUS,
            historical_traditions=["Druidic religion", "Roman conquest", "Tribal organization"],
            strabo_assessment="Gauls are tall and warlike but easily Romanized.",
        )
        self.regions[RegionType.CELTIC_LANDS] = celtic

    def add_ethnographic_note(self, note: EthnographicNote) -> None:
        self.ethnographic_notes.append(note)

    def get_region(self, region_type: RegionType) -> Optional[Region]:
        return self.regions.get(region_type)

    def synthesize_oikoumene(self) -> str:
        """Synthesize the entire oikoumene in Strabo's voice."""
        parts = ["THE OIKOUMENE (Inhabited World):\n"]
        for region_type, region in self.regions.items():
            parts.append(f"\n{region.describe()}")
        parts.append(f"\nTotal ethnographic notes: {len(self.ethnographic_notes)}")
        return "\n".join(parts)


class HistoricalIntegration:
    """
    Integrate historical knowledge with geographic description.
    Strabo's key innovation: geography without history is meaningless.
    """

    @staticmethod
    def add_historical_context(
        region: Region,
        era: str,
        events: List[str]
    ) -> str:
        """Add historical context to a geographic description."""
        context = f"Historical notes for {region.name} ({era}):\n"
        for event in events:
            context += f"  - {event}\n"
        return context

    @staticmethod
    def trace_panethnic_history(
        people: str,
        origin_legend: str,
        migrations: List[str],
        current_location: str
    ) -> str:
        """Trace the history of a people across regions."""
        trace = f"History of {people}:\n"
        trace += f"  Origin: {origin_legend}\n"
        trace += f"  Migrations:\n"
        for mig in migrations:
            trace += f"    → {mig}\n"
        trace += f"  Current location: {current_location}\n"
        return trace


class StraboNarrativeSynthesizer:
    """Synthesize geographic-historical narratives in Strabo's style."""

    def __init__(self, oikoumene: OikoumeneBuilder):
        self.oikoumene = oikoumene
        self.composition_log: List[str] = []

    def compose_region_description(
        self,
        region_type: RegionType,
        historical_depth: HistoricalDepth = HistoricalDepth.HISTORICAL
    ) -> ChorographicDescription:
        """Compose a full chorographic description."""
        region = self.oikoumene.get_region(region_type)
        if not region:
            return ChorographicDescription(
                region_name="Unknown",
                scale=GeographicScale.REGION,
                history="",
                geography="",
            )

        history = ""
        if historical_depth == HistoricalDepth.MYTHICAL:
            history = f"According to myth, {region.name} was shaped by the gods."
        elif historical_depth == HistoricalDepth.LEGENDARY:
            history = f"Legend tells of heroes from {region.name} in ancient times."
        elif historical_depth == HistoricalDepth.HISTORICAL:
            traditions = "; ".join(region.historical_traditions)
            history = f"Historical traditions of {region.name}: {traditions}."
        else:
            history = f"{region.name} as observed in the present era."

        geography = region.describe()

        desc = ChorographicDescription(
            region_name=region.name,
            scale=GeographicScale.REGION,
            history=history,
            geography=geography,
            notable_sites=region.major_cities,
            historical_depth=historical_depth,
        )

        self.composition_log.append(f"[COMPOSED] {region.name} at {historical_depth.name} depth")
        return desc

    def compose_voyage_narrative(
        self,
        start: str,
        end: str,
        waypoints: List[str]
    ) -> List[Route]:
        """Compose a voyage narrative between two points."""
        routes = []
        all_stops = [start] + waypoints + [end]

        for i in range(len(all_stops) - 1):
            route = Route(
                origin=all_stops[i],
                destination=all_stops[i + 1],
                distance_stadia=random.randint(500, 3000),
                terrain_difficulties=[random.choice(["mountain crossing", "desert stretch", "sea voyage", "river crossing"])],
                strabo_route_description=f"From {all_stops[i]} to {all_stops[i+1]}: {" ".join(["One passes through varied terrain."])}",
            )
            routes.append(route)

        self.composition_log.append(f"[VOYAGE] {start} → {end} via {len(waypoints)} waypoints")
        return routes


# =============================================================================
# MAIN ARCHITECTURE
# =============================================================================


class StraboGeographicCognition:
    """
    Complete cognitive architecture for Strabo's geographic-historical synthesis.

    Models:
    1. Spatial reasoning (coordinates, distances, regional relationships)
    2. Environmental determinism (how terrain/climate shapes peoples)
    3. Ethnographic observation (characterization of diverse peoples)
    4. Historical integration (geography illuminated by history)
    5. Chorographic synthesis (regional description combining all elements)
    6. Narrative composition (travel writing, geographical treatise)
    """

    def __init__(self):
        self.oikoumene = OikoumeneBuilder()
        self.narrative_synthesizer = StraboNarrativeSynthesizer(self.oikoumene)
        self.current_description: Optional[ChorographicDescription] = None
        self.composition_log: List[str] = []

    def observe_place(
        self,
        name: str,
        region_type: RegionType,
        terrain: Set[TerrainType],
        climate: ClimateZone
    ) -> Place:
        """Record a place observation."""
        chars = EnvironmentalDeterminism.predict_character(climate, terrain, name)
        cultural = CulturalCharacter.CIVILIZED
        if "savage" in str(chars).lower() or "tribal" in str(chars).lower():
            cultural = CulturalCharacter.SAVAGE
        elif "nomadic" in str(chars).lower():
            cultural = CulturalCharacter.NOMADIC
        elif "barbarous" in str(chars).lower():
            cultural = CulturalCharacter.BARBAROUS

        coord = CartographicProjection.get_coordinate(name)

        place = Place(
            name=name,
            coordinate=coord,
            region=region_type,
            terrain=terrain,
            climate=climate,
            cultural_character=cultural,
            notable_features=["strategic location", "trade connections"],
            strabo_judgment=f"{name} is {'notable' if random.random() > 0.3 else 'unremarkable'}.",
        )
        self.composition_log.append(f"[OBSERVED] {name}")
        return place

    def compose_oikoumene_description(self) -> str:
        """Compose the full oikoumene description."""
        desc = self.oikoumene.synthesize_oikoumene()
        self.composition_log.append("[COMPOSED] Full oikoumene description")
        return desc

    def describe_voyage(
        self,
        origin: str,
        destination: str,
        waypoints: List[str]
    ) -> List[Route]:
        """Describe a voyage from origin to destination."""
        return self.narrative_synthesizer.compose_voyage_narrative(
            origin, destination, waypoints
        )

    def add_ethnographic_entry(
        self,
        people_name: str,
        region: str,
        traits: List[str],
        customs: List[str]
    ) -> EthnographicNote:
        """Add an ethnographic note in Strabo's manner."""
        note = EthnographicNote(
            people_name=people_name,
            region=region,
            character_traits=set(traits),
            customs=customs,
            environment_influence=f"{people_name} are shaped by their {region} environment.",
            strabo_quote=f"Concerning the {people_name}, they are {' '.join(traits[:2])}.",
            reliability=0.8 if len(traits) > 1 else 0.6,
        )
        self.oikoumene.add_ethnographic_note(note)
        self.composition_log.append(f"[ETHNOGRAPHY] {people_name}")
        return note

    def full_geographic_treatise(
        self,
        region_type: RegionType
    ) -> ChorographicDescription:
        """Produce a complete geographic-historical treatise entry."""
        desc = self.narrative_synthesizer.compose_region_description(region_type)
        self.current_description = desc
        self.composition_log.append(f"[TREATISE] {region_type.name}")
        return desc


# =============================================================================
# DEMO
# =============================================================================


def demo() -> None:
    """Demonstrate Strabo's geographic-historical cognition architecture."""
    print("=" * 70)
    print("STRABO'S GEOGRAPHIC-HISTORICAL COGNITION")
    print("Architecture for the Geographica and Chorographic Synthesis")
    print("=" * 70)

    # Initialize
    cognition = StraboGeographicCognition()

    # Show climate zones
    print("\n--- CLIMATE ZONES OF THE OIKOUMENE ---")
    for zone in ClimateZone:
        print(f"  {zone.name}: {len(EnvironmentalDeterminism.CLIMATE_CHARACTER_MAP.get(zone, set()))} character traits")

    # Show terrain-character relationships
    print("\n--- TERRAIN INFLUENCE ON CHARACTER ---")
    for terrain in TerrainType:
        chars = EnvironmentalDeterminism.TERRAIN_CHARACTER_MAP.get(terrain, set())
        if chars:
            print(f"  {terrain.name}: {', '.join(list(chars)[:3])}")

    # Show places in classical coordinates
    print("\n--- CLASSICAL COORDINATES ---")
    for place_name, (lat, lon) in list(CartographicProjection.CLASSICAL_PLACES.items())[:8]:
        print(f"  {place_name}: {lat}°N, {lon}°E")

    # Demonstrate environmental determinism
    print("\n--- ENVIRONMENTAL DETERMINISM EXAMPLES ---")
    examples = [
        (ClimateZone.TEMPERATE, {TerrainType.INLAND_PLAIN}, "Gaul"),
        (ClimateZone.TORRID, {TerrainType.DESERT}, "Arabia"),
        (ClimateZone.SUBTROPICAL, {TerrainType.RIVER_VALLEY}, "Egypt"),
    ]
    for climate, terrain, name in examples:
        chars = EnvironmentalDeterminism.predict_character(climate, terrain, name)
        print(f"  {name}: {', '.join(list(chars)[:4])}")

    # Observe places
    print("\n--- PLACES OBSERVED BY STRABO ---")
    places = [
        ("Rome", RegionType.WESTERN_MEDITERRANEAN, {TerrainType.COASTAL, TerrainType.MOUNTAINOUS}, ClimateZone.TEMPERATE),
        ("Ephesus", RegionType.ANATOLIA, {TerrainType.COASTAL}, ClimateZone.SUBTROPICAL),
        ("Alexandria", RegionType.EGYPT_NILE, {TerrainType.COASTAL, TerrainType.RIVER_VALLEY}, ClimateZone.TROPICAL),
    ]
    for name, region, terrain, climate in places:
        place = cognition.observe_place(name, region, terrain, climate)
        print(f"  {place.name}: {place.cultural_character.name} - {place.strabo_judgment}")

    # Compose full oikoumene description
    print("\n--- THE OIKOUMENE (excerpt) ---")
    oikoumene_text = cognition.compose_oikoumene_description()
    print(oikoumene_text[:800] + "...")

    # Describe voyages
    print("\n--- VOYAGE: ROME TO ALEXANDRIA ---")
    routes = cognition.describe_voyage(
        "Rome",
        "Alexandria",
        ["Antioch", "Cyprus", "Ptolemais"]
    )
    for route in routes:
        print(f"  {route.origin} → {route.destination}: {route.distance_stadia} stadia (~{route.distance_km():.1f} km)")

    # Add ethnographic entries
    print("\n--- ETHNOGRAPHIC NOTES ---")
    entries = [
        ("Celts", "Gaul", ["tall", "warlike", "honest"], ["druidic worship", "head-hunting"]),
        ("Indians", "India", ["philosophical", "peaceful", "austere"], ["naked ascetics", "fire worship"]),
        ("Garamantes", "Libya", ["desert-dwelling", "isolated"], ["underground dwellings", "nomadic raids"]),
    ]
    for people, region, traits, customs in entries:
        note = cognition.add_ethnographic_entry(people, region, traits, customs)
        print(f"  {note.people_name}: {', '.join(list(note.character_traits)[:3])}")
        print(f"    Custom: {note.customs[0]}")

    # Compose geographic treatise
    print("\n--- GEOGRAPHIC TREATISE: ANATOLIA ---")
    treatise = cognition.full_geographic_treatise(RegionType.ANATOLIA)
    print(f"Region: {treatise.region_name}")
    print(f"Scale: {treatise.scale.name}")
    print(f"History: {treatise.history[:150]}...")
    print(f"Geography: {treatise.geography[:200]}...")

    # Show log
    print("\n--- COMPOSITION LOG ---")
    for entry in cognition.composition_log[-10:]:
        print(f"  {entry}")

    print("\n" + "=" * 70)
    print("The Geographica is complete: a systematic description of the oikoumene.")
    print("=" * 70)


if __name__ == "__main__":
    demo()


class GeographicDescriptionModule:
    """
    Systematic geographic description engine for chorographic writing.
    Implements Strabo's methodology for composing regional descriptions
    combining physical geography, climate, peoples, and history.
    """

    DESCRIPTION_TEMPLATES = {
        "coastal": "A {region} coast stretching along the {sea}, characterized by {features}. "
                    "The inhabitants are {character}, shaped by maritime trade and fishing.",
        "mountainous": "The {region} highlands rise to considerable elevation, with peaks reaching {altitude}. "
                       "The mountain peoples are {character}, known for their {virtues} and {faults}.",
        "river_valley": "The fertile {region} valley of the {river} supports dense populations and ancient cities. "
                         "Agriculture flourishes here, yielding {products}.",
        "desert": "The {region} desert expanse presents a harsh landscape of {terrain_features}. "
                   "Sparse nomadic tribes traverse these {description} regions, known for {qualities}.",
    }

    CLIMATIC_INFLUENCE = {
        ClimateZone.TORRID: "Extreme heat characterizes this region, rendering the inhabitants "
                            "dark-skinned and {traits}. The burning sun shapes all activity.",
        ClimateZone.TROPICAL: "Vexing humidity and prolific rainfall mark this climate, producing "
                              "lush vegetation and {traits} among the peoples.",
        ClimateZone.SUBTROPICAL: "A mild climate prevails here, with {seasons} that foster "
                                 "civilization and philosophical inquiry among the {peoples}.",
        ClimateZone.TEMPERATE: "The moderate climate of this region produces {peoples} noted for "
                               "their industry, {virtues}, and balanced character.",
        ClimateZone.COLD: "The bracing cold creates a hardy people, {traits}, "
                          "their courage forged in the struggle against {challenges}.",
        ClimateZone.ARCTIC: "The extreme cold renders this land nearly uninhabitable, "
                            "with only {tribes} eking out existence through {methods}.",
    }

    def __init__(self):
        self.composed_descriptions: List[ChorographicDescription] = []
        self.description_history: List[str] = []
        self.quality_scores: List[float] = []

    def compose_regional_description(
        self,
        region: Region,
        historical_depth: HistoricalDepth = HistoricalDepth.HISTORICAL,
        emphasis: Optional[Set[TerrainType]] = None
    ) -> ChorographicDescription:
        """Compose a comprehensive chorographic description."""
        terrain_main = list(region.terrain)[0] if region.terrain else TerrainType.INLAND_PLAIN
        template_key = "coastal" if TerrainType.COASTAL in region.terrain else \
                       "mountainous" if TerrainType.MOUNTAINOUS in region.terrain else \
                       "river_valley" if TerrainType.RIVER_VALLEY in region.terrain else \
                       "desert" if TerrainType.DESERT in region.terrain else "coastal"

        climate_desc = self.CLIMATIC_INFLUENCE.get(
            region.climate,
            "A climate of moderate character."
        )

        history_parts = []
        if historical_depth == HistoricalDepth.MYTHICAL:
            history_parts.append(f"Myths claim {region.name} was shaped by the gods.")
        elif historical_depth == HistoricalDepth.LEGENDARY:
            history_parts.append(f"Heroes of legend traversed {region.name} in ancient times.")
        elif historical_depth == HistoricalDepth.HISTORICAL:
            traditions = "; ".join(region.historical_traditions)
            history_parts.append(f"Historical record: {traditions}.")
        else:
            history_parts.append(f"{region.name} in the present era.")

        geography_parts = region.describe()
        if emphasis:
            emphasis_terrains = ", ".join(t.name.lower() for t in emphasis)
            geography_parts += f" Emphasizing the {emphasis_terrains} character."

        description = ChorographicDescription(
            region_name=region.name,
            scale=GeographicScale.REGION,
            history=" ".join(history_parts),
            geography=geography_parts,
            notable_sites=region.major_cities,
            historical_depth=historical_depth,
        )

        quality = random.uniform(0.6, 0.95)
        self.composed_descriptions.append(description)
        self.description_history.append(f"[COMPOSED] {region.name} at {historical_depth.name}")
        self.quality_scores.append(quality)

        return description

    def synthesize_description(self, description: ChorographicDescription) -> str:
        """Synthesize a full textual description from chorographic data."""
        parts = [
            f"CONCERNING {description.region_name}:",
            "",
            f"History: {description.history}",
            "",
            f"Geography: {description.geography}",
            "",
        ]
        if description.notable_sites:
            sites_str = ", ".join(description.notable_sites)
            parts.append(f"Notable sites: {sites_str}.")
        parts.append("")
        return "\n".join(parts)

    def compare_regions(self, region_a: Region, region_b: Region) -> Dict[str, float]:
        """Compare two regions across multiple dimensions."""
        return {
            "climate_similarity": 1.0 - abs(
                list(ClimateZone).index(region_a.climate) - list(ClimateZone).index(region_b.climate)
            ) / len(ClimateZone),
            "terrain_overlap": len(region_a.terrain & region_b.terrain) / max(1, len(region_a.terrain | region_b.terrain)),
            "cultural_similarity": 1.0 if region_a.cultural_character == region_b.cultural_character else 0.3,
        }


class OikoumeneExpansionHistory:
    """History of how the oikoumene concept expanded."""
    def __init__(self):
        self.expansion_stages = {
            "Homer": "Known Mediterranean world only",
            "Hecataeus": "First Greek geography, rational",
            "Herodotus": "Comprehensive but fabulous additions",
            "Eratosthenes": "Scientific measurement, 18000 stadia perimeter",
            "Strabo": "Full synthesis, 17-book encyclopedia"
        }

    def oikoumene_development(self) -> str:
        return "Oikoumene expanded from Homeric myths to Strabo's rational synthesis"


class StraboBiographicalContext:
    """Strabo's life and its influence on his geography."""
    def __init__(self):
        self.life = [
            ("64 BCE", "Birth in Amaseia, Pontus"),
            ("48-44 BCE", "Studies in Rome, Stoic philosophy"),
            ("44-30 BCE", "Studies with Aristodemus of Nyssa"),
            ("31 BCE onward", "Accompanies Aelius Gallus to Egypt"),
            ("29 BCE-17 CE", "Writes Geographica in Rome"),
            ("24 CE onward", "Later years in Rome, possibly blindness")
        ]

    def life_timeline(self) -> List[Tuple[str, str]]:
        return self.life


class ClassicalCartographyPrinciples:
    """Ancient cartographic principles."""
    def __init__(self):
        self.map_elements = {
            "orientation": "East at top preferred",
            "climes": "Latitude zones affect climate and character",
            "parallels": "Climate lines at regular intervals",
            "meridians": "Longitude lines through known landmarks"
        }

    def cartographic_method(self) -> str:
        return "Classical cartography: qualitative description over precise measurement"


class StraboOnClimateAndCulture:
    """Strabo's environmental determinism in detail."""
    def __init__(self):
        self.climate_effects = {
            "cold_north": "Produces brave but unintelligent peoples",
            "temperate_middle": "Produces intelligent and courageous peoples",
            "hot_south": "Produces intelligent but lacking courage"
        }

    def climate_to_culture(self, climate: str) -> str:
        return self.climate_effects.get(climate, "No correlation established")


class RomanProvincialAdministration:
    """Roman provincial system Strabo describes."""
    def __init__(self):
        self.provincial_types = {
            "senatorial_province": "Governed by proconsul, traditional areas",
            "imperial_province": "Governed by legate of emperor, frontier or strategic",
            "client_kingdom": "Native king under Roman oversight"
        }

    def provincial_classification(self) -> Dict[str, str]:
        return self.provincial_types


class StraboMethodology:
    """Strabo's geographic methodology."""
    def __init__(self):
        self.methodology = {
            "direct_observation": "Personal travel and investigation",
            "oral_sources": "Interviews with locals, merchants, officials",
            "written_sources": "Prior geographers, historians",
            "philosophical_synthesis": "Integration into coherent worldview"
        }

    def methodology_description(self) -> Dict[str, str]:
        return self.methodology


class EthnographicReliabilityAssessment:
    """Assessing reliability of Strabo's ethnographic accounts."""
    def __init__(self):
        self.reliability_factors = {
            "indus_valley": {"reliability": 0.6, "reason": "Second-hand information"},
            "northern_europe": {"reliability": 0.4, "reason": "Fabulous reports"},
            "mediterranean": {"reliability": 0.8, "reason": "Well-traveled areas"},
            "egypt": {"reliability": 0.9, "reason": "Personal investigation"}
        }

    def reliability_score(self, region: str) -> float:
        info = self.reliability_factors.get(region, {"reliability": 0.5, "reason": "Unknown"})
        return info["reliability"]


class StraboAsHistoricalSource:
    """Strabo as a source for Roman-era history."""
    def __init__(self):
        self.historical_passages = {
            "sicily_etna": "Accurate description of volcanic activity",
            "hannibal_alps": "Credible account of Alpine crossing",
            "pompeli_extraction": "Pompeii's destruction described",
            "caucasus_wars": "Accurate military geography"
        }

    def historical_value(self, passage: str) -> str:
        return self.historical_passages.get(passage, "No historical passage by that name")


class StrabosIntellectualLegacy:
    """Strabo's influence on later geography."""
    def __init__(self):
        self.legacy = {
            "Ptolemy": "Adopts Strabo's regional descriptions",
            "medieval_islamic": "Arabic translation preserved knowledge",
            "renaissance": "Editio princeps 1472, Venetian printing",
            "modern_geography": "Regional geography approach precedent"
        }

    def influence_path(self) -> Dict[str, str]:
        return self.legacy


class StraboOnRomanImperialGeography:
    """Strabo's view of Rome's imperial geography."""
    def __init__(self):
        self.imperial_geography = {
            "italy_centrality": "Italy as natural center of oikoumene",
            "rome_caput_mundi": "Rome as head of the world",
            "provincial_network": "Seamless communication through provinces",
            "military_roads": "Strategic road network enabling empire"
        }

    def roman_geographic_view(self) -> Dict[str, str]:
        return self.imperial_geography


class GeographicDescriptionTechniques:
    """Strabo's techniques for regional description."""
    def __init__(self):
        self.description_order = [
            "Physical geography (coast, mountains, rivers)",
            "Climate and natural products",
            "Inhabitants and their character",
            "Cities and notable sites",
            "Historical significance"
        ]

    def description_protocol(self) -> List[str]:
        return self.description_order


class StraboOnMapsAndMapmaking:
    """Strabo's attitude toward maps."""
    def __init__(self):
        self.map_attitude = {
            "text_over_maps": "Verbal description more reliable than maps",
            "map_limitations": "Distortion from spherical to flat surface",
            "mental_maps": "Readers construct their own mental geography",
            "reference_systems": "Use of landmarks and directions"
        }

    def map_philosophy(self) -> str:
        return self.map_attitude.get("text_over_maps", "Textual description preferred")
