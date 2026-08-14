#!/usr/bin/env python3
"""
Figure 102: Spartacus (-109 CE)
Array Index 101 in figures_master.json
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 6 Minds 101 - 120 Available on Amazon https://www.amazon.com/dp/B0HF7G6JJD
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 102: Spartacus (-109 CE)
================================================================================
Domain: governance, military, Roman

A comprehensive cognitive architecture modeling the distributed command
framework, slave revolt simulation, gladiatorial training, and AGI alignment
principles embodied by Spartacus and the Third Servile War.

This module implements a five-layer cognitive architecture:
  Layer1_Sensus   - Sensory processing and primitive perception
  Layer2_Corroborat - Cross-validation and corroboration networks
  Layer3_Fuga     - Strategic flight/planning and escape dynamics
  Layer4_Exercitus - Military organization and army command
  Layer5_Sapientia - Wisdom, ethics, and long-term alignment

Author: 1000Minds Cognitive Architecture Framework
Version: 1.0.0
"""

from __future__ import annotations

import copy
import math
import random
import sys
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

# =============================================================================
# SECTION 1: FOUNDATIONAL DATA STRUCTURES
# =============================================================================

T = TypeVar("T")


class TermType(Enum):
    """Classification of logical terms."""
    UNIVERSAL = auto()
    INDIVIDUAL = auto()
    PREDICATE = auto()


@dataclass(frozen=True)
class Term:
    """
    A logical term representing a concept in the cognitive architecture.
    
    Terms are the atomic units of meaning, capable of representing
    universals (general concepts), individuals (specific instances),
    or predicates (relations).
    """
    symbol: str
    term_type: TermType
    extension: frozenset = field(default_factory=frozenset)
    intension: Optional[Callable[..., Any]] = field(default=None)
    metadata: frozenset = field(default_factory=frozenset)
    
    def __post_init__(self):
        if not self.symbol:
            raise ValueError("Term symbol cannot be empty")
    
    def subsumes(self, other: Term) -> bool:
        """Check if this term subsumes another term."""
        if self.term_type != TermType.UNIVERSAL:
            return False
        if other.term_type == TermType.UNIVERSAL:
            return self.extension >= other.extension
        return other.symbol in self.extension
    
    def __hash__(self) -> int:
        return hash((self.symbol, self.term_type))
    
    def __str__(self) -> str:
        type_marker = {
            TermType.UNIVERSAL: "U",
            TermType.INDIVIDUAL: "I",
            TermType.PREDICATE: "P"
        }.get(self.term_type, "?")
        return f"{type_marker}[{self.symbol}]"
    
    def __repr__(self) -> str:
        return self.__str__()


@dataclass(frozen=True)
class Universal(Term):
    """A universal term representing a general concept or kind."""
    
    def __post_init__(self):
        object.__setattr__(self, 'term_type', TermType.UNIVERSAL)
    
    @classmethod
    def create(cls, symbol: str, instances: Optional[Set[str]] = None,
               predicate: Optional[Callable[..., bool]] = None) -> Universal:
        """Factory method to create a universal term."""
        ext = frozenset(instances) if instances else frozenset()
        return cls(symbol=symbol, term_type=TermType.UNIVERSAL,
                   extension=ext, intension=predicate)


@dataclass(frozen=True)
class Individual(Term):
    """An individual term representing a specific entity."""
    
    def __post_init__(self):
        object.__setattr__(self, 'term_type', TermType.INDIVIDUAL)
    
    @classmethod
    def create(cls, symbol: str, 
               properties: Optional[Set[str]] = None) -> Individual:
        """Factory method to create an individual term."""
        ext = frozenset(properties) if properties else frozenset()
        return cls(symbol=symbol, term_type=TermType.INDIVIDUAL, extension=ext)


@dataclass
class Proposition:
    """
    A logical proposition that can be true or false.
    
    Propositions are the building blocks of arguments and syllogisms,
    relating terms through subject-predicate or subject-predicate-object structures.
    """
    subject: Term
    predicate: Term
    object: Optional[Term] = None
    polarity: bool = True  # True = positive/affirmative, False = negative
    confidence: float = 1.0
    source: str = "inference"
    timestamp: float = field(default_factory=time.time)
    
    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0 and 1")
    
    @property
    def is_atomic(self) -> bool:
        """Check if this is an atomic (binary) proposition."""
        return self.object is None
    
    @property
    def is_compound(self) -> bool:
        """Check if this is a compound (ternary) proposition."""
        return self.object is not None
    
    def negate(self) -> Proposition:
        """Return the negated form of this proposition."""
        return Proposition(
            subject=self.subject,
            predicate=self.predicate,
            object=self.object,
            polarity=not self.polarity,
            confidence=self.confidence,
            source=f"negation_of({self.source})"
        )
    
    def to_logical_string(self) -> str:
        """Convert to logical notation."""
        neg = "¬" if not self.polarity else ""
        if self.object:
            return f"{neg}{self.subject} {self.predicate} {self.object}"
        return f"{neg}{self.subject}({self.predicate})"
    
    def __str__(self) -> str:
        return self.to_logical_string()
    
    def __repr__(self) -> str:
        return f"Proposition({self.subject}, {self.predicate}, {self.object}, {self.polarity}, {self.confidence:.2f})"


@dataclass
class Syllogism:
    """
    A syllogistic argument structure.
    
    A syllogism consists of a major premise, minor premise, and conclusion,
    forming the backbone of deductive reasoning in the cognitive architecture.
    """
    major_premise: Proposition
    minor_premise: Proposition
    conclusion: Proposition
    figure: int = 1  # 1-4 classical figures
    mood: str = "AAA"  # Three-letter mood (A, E, I, O)
    valid: bool = True
    schema_name: str = "modus_ponens"
    
    def validate(self) -> bool:
        """Validate the syllogism structure."""
        # Check if the conclusion follows logically
        # This is a simplified check; full validation would be more complex
        return (
            self.major_premise.confidence >= 0.5 and
            self.minor_premise.confidence >= 0.5 and
            self.major_premise.polarity and
            self.minor_premise.polarity and
            self.valid
        )
    
    def to_string(self) -> str:
        """Format the syllogism as a readable string."""
        return (
            f"Major Premise: {self.major_premise}\n"
            f"Minor Premise: {self.minor_premise}\n"
            f"Conclusion: {self.conclusion}\n"
            f"Figure: {self.figure}, Mood: {self.mood}, Valid: {self.valid}"
        )
    
    def __str__(self) -> str:
        return self.to_string()
    
    def __repr__(self) -> str:
        return f"Syllogism({self.schema_name}, fig={self.figure})"


@dataclass
class Objection:
    """
    An objection raised against a proposition or syllogism.
    
    Objections represent counterarguments, challenges, or points of
    critique that must be addressed for proper reasoning.
    """
    target: Union[Proposition, Syllogism]
    objection_type: str  # e.g., "fallacy", "counterexample", "rebuttal"
    description: str
    strength: float = 0.5  # 0 = weak, 1 = devastating
    counter_responses: List[Proposition] = field(default_factory=list)
    resolved: bool = False
    id: str = field(default_factory=uuid.uuid4)
    
    def is_devastating(self) -> bool:
        """Check if this objection is devastating (strength >= 0.8)."""
        return self.strength >= 0.8
    
    def is_resolvable(self) -> bool:
        """Check if this objection can be resolved."""
        return self.strength < 1.0 and not self.is_devastating()
    
    def apply_resolution(self, resolution: Resolution) -> None:
        """Apply a resolution to this objection."""
        self.resolved = True
        self.counter_responses.extend(resolution.supporting_propositions)


@dataclass
class Resolution:
    """
    A resolution to an objection or set of objections.
    
    Resolutions provide answers to objections, either by rebutting
    the objection directly, providing additional evidence, or
    restructuring the argument.
    """
    objection_id: str
    resolution_type: str  # "rebuttal", "counter", "supplementary", "restructure"
    explanation: str
    supporting_propositions: List[Proposition] = field(default_factory=list)
    effectiveness: float = 0.5
    author: str = "system"
    
    def is_effective(self) -> bool:
        """Check if this resolution is effective (effectiveness >= 0.7)."""
        return self.effectiveness >= 0.7


# =============================================================================
# SECTION 2: ANATOMY OF THE FIVE-LAYER COGNITIVE ARCHITECTURE
# =============================================================================

class Layer(ABC):
    """
    Abstract base class for all cognitive layers.
    
    Each layer in the five-layer architecture inherits from this base,
    implementing the core processing functions specific to its role.
    """
    
    def __init__(self, layer_id: int, layer_name: str):
        self.layer_id = layer_id
        self.layer_name = layer_name
        self.activation_level: float = 0.0
        self.processing_state: Dict[str, Any] = {}
        self.connections: List[Layer] = []
        self.is_active: bool = False
    
    @abstractmethod
    def process(self, input_data: Any) -> Any:
        """Process input data through this layer."""
        pass
    
    @abstractmethod
    def activate(self, intensity: float) -> None:
        """Activate this layer at a given intensity."""
        pass
    
    def connect_to(self, other: Layer) -> None:
        """Create a connection to another layer."""
        if other not in self.connections:
            self.connections.append(other)
            other.connections.append(self)
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of this layer."""
        return {
            "layer_id": self.layer_id,
            "layer_name": self.layer_name,
            "activation_level": self.activation_level,
            "is_active": self.is_active,
            "connections": [l.layer_name for l in self.connections]
        }
    
    def __str__(self) -> str:
        return f"Layer{self.layer_id}_{self.layer_name}"


class Layer1_Sensus(Layer):
    """
    Layer 1: Sensus (Sensory Processing)
    
    The foundational layer responsible for sensory processing and
    primitive perception. Models the basic sensory experiences
    that inform all higher-level cognition.
    
    In the context of Spartacus, this represents the raw sensory
    experiences of slaves and gladiators: physical sensations,
    environmental observations, and primitive threat detection.
    """
    
    def __init__(self):
        super().__init__(layer_id=1, layer_name="Sensus")
        self.sensory_buffer: List[Any] = []
        self.threshold: float = 0.3
        self.receptors: Dict[str, float] = {
            "pain": 0.0,
            "pleasure": 0.0,
            "danger": 0.0,
            "opportunity": 0.0,
            "authority": 0.0,
            "companion": 0.0
        }
        self.sensory_history: List[Dict[str, float]] = []
        self.adaptation_rate: float = 0.1
    
    def process(self, input_data: Any) -> Dict[str, float]:
        """
        Process sensory input and update receptor levels.
        
        Args:
            input_data: Raw sensory data (dict with type and intensity)
        
        Returns:
            Dictionary of activated receptor levels
        """
        if isinstance(input_data, dict):
            stimulus_type = input_data.get("type", "unknown")
            intensity = input_data.get("intensity", 0.5)
            duration = input_data.get("duration", 1.0)
            
            # Update appropriate receptor
            if stimulus_type in self.receptors:
                old_level = self.receptors[stimulus_type]
                # Sensory adaptation: repeated stimuli produce diminished response
                adaptation = old_level * self.adaptation_rate * duration
                new_level = min(1.0, old_level + intensity - adaptation)
                self.receptors[stimulus_type] = new_level
            
            # Add to sensory buffer
            self.sensory_buffer.append({
                "timestamp": time.time(),
                "type": stimulus_type,
                "intensity": intensity
            })
            
            # Maintain buffer size
            if len(self.sensory_buffer) > 1000:
                self.sensory_buffer = self.sensory_buffer[-500:]
        
        # Update activation based on receptor states
        self.activation_level = sum(self.receptors.values()) / len(self.receptors)
        self.is_active = self.activation_level >= self.threshold
        
        return self.receptors.copy()
    
    def activate(self, intensity: float) -> None:
        """Activate this layer at a given intensity."""
        self.activation_level = min(1.0, intensity)
        self.is_active = self.activation_level >= self.threshold
        
        # Propagate activation to connected layers
        for layer in self.connections:
            if not layer.is_active or layer.activation_level < self.activation_level * 0.5:
                layer.activate(self.activation_level * 0.7)
    
    def detect_threat(self) -> bool:
        """Detect if any danger receptor exceeds threshold."""
        return self.receptors["danger"] >= 0.6 or self.receptors["pain"] >= 0.7
    
    def detect_opportunity(self) -> bool:
        """Detect if opportunity receptor exceeds threshold."""
        return self.receptors["opportunity"] >= 0.5
    
    def get_primitives(self) -> List[Term]:
        """Get primitive terms derived from sensory processing."""
        primitives = []
        for receptor, level in self.receptors.items():
            if level >= self.threshold:
                term = Universal.create(
                    symbol=f"sensation_{receptor}",
                    instances={f"intensity_{level:.2f}"}
                )
                primitives.append(term)
        return primitives
    
    def reset_receptors(self) -> None:
        """Reset all receptor levels to baseline."""
        for key in self.receptors:
            self.receptors[key] = 0.0
        self.sensory_buffer.clear()


class Layer2_Corroborat(Layer):
    """
    Layer 2: Corroborat (Cross-Validation and Corroboration)
    
    This layer performs cross-validation of sensory information,
    checking for consistency across multiple sources and building
    corroborative evidence chains.
    
    In the context of Spartacus, this models how escaped slaves
    would corroborate stories, verify rumors of freedom, and
    build collective testimony about Roman movements and plans.
    """
    
    def __init__(self):
        super().__init__(layer_id=2, layer_name="Corroborat")
        self.trust_network: Dict[str, float] = {}
        self.evidence_chains: List[List[Proposition]] = []
        self.confidence_threshold: float = 0.6
        self.testimony_buffer: List[Tuple[str, Proposition]] = []
        self.consensus_formed: bool = False
        self.contradiction_log: List[Tuple[Proposition, Proposition]] = []
    
    def process(self, input_data: Any) -> Dict[str, Any]:
        """
        Process corroborative information from multiple sources.
        
        Args:
            input_data: Dict containing multiple testimonies or evidence items
        
        Returns:
            Dictionary with validated consensus and confidence metrics
        """
        if not isinstance(input_data, dict):
            return {"validated": False, "confidence": 0.0, "consensus": None}
        
        source_id = input_data.get("source", "unknown")
        proposition = input_data.get("proposition")
        
        if not isinstance(proposition, Proposition):
            return {"validated": False, "confidence": 0.0, "consensus": None}
        
        # Add to testimony buffer
        self.testimony_buffer.append((source_id, proposition))
        
        # Update trust network
        current_trust = self.trust_network.get(source_id, 0.5)
        self.trust_network[source_id] = min(1.0, current_trust + 0.1)
        
        # Check for corroboration with existing testimonies
        corroboration_count = 0
        supporting_sources = []
        
        for existing_source, existing_prop in self.testimony_buffer[:-1]:
            if self._propositions_agree(proposition, existing_prop):
                corroboration_count += 1
                supporting_sources.append(existing_source)
        
        # Calculate confidence based on corroboration
        if corroboration_count > 0:
            base_confidence = proposition.confidence
            corroboration_bonus = min(0.3, corroboration_count * 0.1)
            trust_bonus = sum(self.trust_network.get(s, 0.5) for s in supporting_sources) / len(supporting_sources) if supporting_sources else 0.0
            final_confidence = min(1.0, base_confidence + corroboration_bonus + trust_bonus * 0.2)
        else:
            final_confidence = proposition.confidence * self.trust_network.get(source_id, 0.5)
        
        # Check for contradictions
        for existing_source, existing_prop in self.testimony_buffer[:-1]:
            if self._propositions_contradict(proposition, existing_prop):
                self.contradiction_log.append((proposition, existing_prop))
        
        # Update activation
        self.activation_level = min(1.0, corroboration_count * 0.2)
        self.is_active = self.activation_level >= 0.1
        
        # Build evidence chain if sufficient corroboration
        if corroboration_count >= 2 and final_confidence >= self.confidence_threshold:
            self._build_evidence_chain(proposition, supporting_sources)
            self.consensus_formed = True
        
        return {
            "validated": corroboration_count >= 2,
            "confidence": final_confidence,
            "consensus": proposition if corroboration_count >= 2 else None,
            "corroboration_count": corroboration_count,
            "supporting_sources": supporting_sources
        }
    
    def activate(self, intensity: float) -> None:
        """Activate this layer at a given intensity."""
        self.activation_level = min(1.0, intensity)
        self.is_active = self.activation_level >= 0.1
        
        for layer in self.connections:
            if not layer.is_active or layer.activation_level < self.activation_level * 0.5:
                layer.activate(self.activation_level * 0.65)
    
    def _propositions_agree(self, p1: Proposition, p2: Proposition) -> bool:
        """Check if two propositions agree on their core claims."""
        return (
            p1.subject == p2.subject and
            p1.predicate == p2.predicate and
            p1.polarity == p2.polarity
        )
    
    def _propositions_contradict(self, p1: Proposition, p2: Proposition) -> bool:
        """Check if two propositions contradict each other."""
        return (
            p1.subject == p2.subject and
            p1.predicate == p2.predicate and
            p1.polarity != p2.polarity
        )
    
    def _build_evidence_chain(self, target: Proposition, sources: List[str]) -> None:
        """Build a corroborative evidence chain."""
        chain = [Proposition(
            subject=Term(symbol=s, term_type=TermType.INDIVIDUAL),
            predicate=Term(symbol="testifies", term_type=TermType.PREDICATE),
            object=target.subject,
            confidence=0.8
        ) for s in sources]
        chain.append(target)
        self.evidence_chains.append(chain)
    
    def get_trust_scores(self) -> Dict[str, float]:
        """Get current trust scores for all sources."""
        return self.trust_network.copy()
    
    def get_consensus(self) -> Optional[Proposition]:
        """Get the current consensus proposition if one exists."""
        if self.consensus_formed and self.evidence_chains:
            return self.evidence_chains[-1][-1]
        return None


class Layer3_Fuga(Layer):
    """
    Layer 3: Fuga (Strategic Flight and Planning)
    
    This layer handles strategic planning, escape dynamics,
    and the calculation of optimal flight paths or retreat
    strategies.
    
    In the context of Spartacus, this represents the strategic
    thinking around escape from captivity, routing decisions,
    and the planning of military maneuvers to avoid Roman legions
    while building the rebel army.
    """
    
    def __init__(self):
        super().__init__(layer_id=3, layer_name="Fuga")
        self.escape_routes: List[Dict[str, Any]] = []
        self.risk_assessment: float = 0.5
        self.strategic_options: List[Dict[str, Any]] = []
        self.retreat_threshold: float = 0.6
        self.advance_threshold: float = 0.7
        self.planning_horizon: int = 10  # steps ahead
        self.current_route_index: int = 0
        self.terrain_map: Dict[str, Any] = {}
        self.enemy_positions: List[Tuple[float, float]] = []
        self.friendly_positions: List[Tuple[float, float]] = []
        self.waypoints: List[Tuple[float, float]] = []
    
    def process(self, input_data: Any) -> Dict[str, Any]:
        """
        Process strategic planning input and calculate escape/advance options.
        
        Args:
            input_data: Dict with tactical situation
        
        Returns:
            Dictionary with strategic options and recommended actions
        """
        if not isinstance(input_data, dict):
            return {"action": "wait", "confidence": 0.0}
        
        situation = input_data.get("situation", "unknown")
        enemy_strength = input_data.get("enemy_strength", 0.5)
        friendly_strength = input_data.get("friendly_strength", 0.5)
        terrain = input_data.get("terrain", "plains")
        
        # Calculate risk assessment
        self.risk_assessment = enemy_strength * 0.6 + (1 - friendly_strength) * 0.4
        
        # Determine primary action
        if self.risk_assessment >= self.retreat_threshold:
            action = "retreat"
            self._calculate_retreat_routes(terrain, enemy_strength)
        elif self.risk_assessment <= (1 - self.advance_threshold):
            action = "advance"
            self._calculate_advance_routes(terrain, enemy_strength)
        else:
            action = "hold"
            self._generate_hold_positions(terrain)
        
        # Update activation
        self.activation_level = 1.0 - self.risk_assessment
        self.is_active = True
        
        # Generate strategic summary
        return {
            "action": action,
            "risk_assessment": self.risk_assessment,
            "confidence": 1.0 - self.risk_assessment,
            "routes": self.escape_routes if action == "retreat" else self.strategic_options,
            "terrain": terrain
        }
    
    def activate(self, intensity: float) -> None:
        """Activate this layer at a given intensity."""
        self.activation_level = min(1.0, intensity)
        self.is_active = self.activation_level >= 0.2
        
        for layer in self.connections:
            if not layer.is_active or layer.activation_level < self.activation_level * 0.6:
                layer.activate(self.activation_level * 0.6)
    
    def _calculate_retreat_routes(self, terrain: str, enemy_strength: float) -> None:
        """Calculate optimal retreat routes based on terrain and enemy positions."""
        self.escape_routes = []
        
        # Route 1: Mountain passage
        route1 = {
            "name": "Via Montana",
            "path": [(0, 0), (2, 1), (4, 3), (6, 4)],
            "risk": 0.3 + (enemy_strength * 0.2),
            "speed": "slow",
            "terrain": "mountain",
            "cover": 0.8,
            "description": "Mountain path offering concealment but slower progress"
        }
        
        # Route 2: Coastal escape
        route2 = {
            "name": "Coastal Flight",
            "path": [(0, 0), (3, 0), (5, 1), (8, 1)],
            "risk": 0.5 - (enemy_strength * 0.1),
            "speed": "fast",
            "terrain": "coastal",
            "cover": 0.3,
            "description": "Quick coastal route with less cover but faster movement"
        }
        
        # Route 3: Forest concealment
        route3 = {
            "name": "Silva Secreta",
            "path": [(0, 0), (1, 2), (3, 2), (5, 3)],
            "risk": 0.4,
            "speed": "medium",
            "terrain": "forest",
            "cover": 0.9,
            "description": "Hidden forest route with excellent concealment"
        }
        
        self.escape_routes = [route1, route2, route3]
        self.strategic_options = self.escape_routes
    
    def _calculate_advance_routes(self, terrain: str, enemy_strength: float) -> None:
        """Calculate optimal advance routes."""
        self.strategic_options = []
        
        # Flanking maneuver
        flank = {
            "name": "Double Envelopment",
            "type": "offensive",
            "risk": 0.3,
            "speed": "rapid",
            "description": "Classic pincer movement against weakened enemy positions"
        }
        
        # Direct assault
        assault = {
            "name": "Charge",
            "type": "offensive",
            "risk": 0.5,
            "speed": "immediate",
            "description": "Direct assault exploiting enemy disorganization"
        }
        
        # Skirmish
        skirmish = {
            "name": "Harassing Action",
            "type": "offensive",
            "risk": 0.2,
            "speed": "continuous",
            "description": "Repeated hit-and-run attacks to weaken enemy"
        }
        
        self.strategic_options = [flank, assault, skirmish]
    
    def _generate_hold_positions(self, terrain: str) -> None:
        """Generate defensive hold positions."""
        self.strategic_options = [
            {
                "name": "Defensive Circle",
                "type": "defensive",
                "risk": 0.3,
                "description": "Circular formation with wagons, optimal for defense"
            },
            {
                "name": "Ridge Defense",
                "type": "defensive",
                "risk": 0.25,
                "description": "Elevated position taking advantage of high ground"
            },
            {
                "name": "River Line",
                "type": "defensive",
                "risk": 0.2,
                "description": "Natural barrier defense along waterway"
            }
        ]
    
    def set_terrain_map(self, terrain_map: Dict[str, Any]) -> None:
        """Set the terrain map for strategic calculations."""
        self.terrain_map = terrain_map
    
    def add_waypoint(self, x: float, y: float) -> None:
        """Add a waypoint to the current route."""
        self.waypoints.append((x, y))
        if len(self.waypoints) > self.planning_horizon:
            self.waypoints = self.waypoints[-self.planning_horizon:]
    
    def get_current_route(self) -> Optional[Dict[str, Any]]:
        """Get the currently selected route."""
        if self.escape_routes and self.current_route_index < len(self.escape_routes):
            return self.escape_routes[self.current_route_index]
        return None


class Layer4_Exercitus(Layer):
    """
    Layer 4: Exercitus (Military Organization and Command)
    
    This layer handles military organization, army command,
    unit coordination, and tactical execution of military plans.
    
    In the context of Spartacus, this models the organization
    of the rebel slave army, the formation of units, the
    distribution of command authority, and the coordination
    of military operations against Roman forces.
    """
    
    def __init__(self):
        super().__init__(layer_id=4, layer_name="Exercitus")
        self.units: Dict[str, Dict[str, Any]] = {}
        self.formation_types: List[str] = [
            "testudo", "cuneus", "orbis", "acies", "fuga"
        ]
        self.current_formation: str = "acies"
        self.commander_structure: Dict[str, str] = {}  # subordinate -> commander
        self.battle_readiness: float = 0.5
        self.supply_level: float = 0.6
        self.morale: float = 0.7
        self.tactical_commands: List[str] = []
        self.engagement_range: float = 50.0  # meters
        self.reinforcement_queue: List[str] = []
        self.casualties: Dict[str, int] = {"killed": 0, "wounded": 0, "missing": 0}
    
    def process(self, input_data: Any) -> Dict[str, Any]:
        """
        Process military commands and organizational updates.
        
        Args:
            input_data: Dict with military situation and commands
        
        Returns:
            Dictionary with tactical status and command execution results
        """
        if not isinstance(input_data, dict):
            return {"status": "idle", "coherence": 0.0}
        
        command = input_data.get("command", "status")
        target_unit = input_data.get("unit", "all")
        
        if command == "form":
            formation = input_data.get("formation", "acies")
            result = self._form_formation(formation, target_unit)
        elif command == "move":
            direction = input_data.get("direction", "forward")
            result = self._execute_movement(direction, target_unit)
        elif command == "attack":
            target = input_data.get("target")
            result = self._execute_attack(target)
        elif command == "defend":
            position = input_data.get("position")
            result = self._execute_defense(position)
        elif command == "retreat":
            route = input_data.get("route")
            result = self._execute_retreat(route)
        else:
            result = self._report_status()
        
        # Calculate battle readiness
        self.battle_readiness = (self.morale * 0.4 + self.supply_level * 0.3 + 
                                  (1 - self.casualties["killed"] / max(1, sum(u.get("size", 100) for u in self.units.values()))) * 0.3)
        
        # Update activation
        self.activation_level = self.battle_readiness
        self.is_active = self.battle_readiness >= 0.3
        
        result["battle_readiness"] = self.battle_readiness
        result["morale"] = self.morale
        result["supply_level"] = self.supply_level
        
        return result
    
    def activate(self, intensity: float) -> None:
        """Activate this layer at a given intensity."""
        self.activation_level = min(1.0, intensity)
        self.is_active = self.activation_level >= 0.3
        
        for layer in self.connections:
            if not layer.is_active or layer.activation_level < self.activation_level * 0.55:
                layer.activate(self.activation_level * 0.55)
    
    def add_unit(self, unit_id: str, unit_type: str, size: int, 
                 commander: Optional[str] = None) -> None:
        """Add a unit to the army."""
        self.units[unit_id] = {
            "type": unit_type,
            "size": size,
            "status": "ready",
            "position": (0.0, 0.0),
            "commander": commander,
            "experience": random.uniform(0.3, 0.9)
        }
        if commander:
            self.commander_structure[unit_id] = commander
        self.tactical_commands.append(f"Unit {unit_id} added to exercitus")
    
    def remove_unit(self, unit_id: str) -> None:
        """Remove a unit from the army."""
        if unit_id in self.units:
            del self.units[unit_id]
            if unit_id in self.commander_structure:
                del self.commander_structure[unit_id]
            self.tactical_commands.append(f"Unit {unit_id} removed from exercitus")
    
    def _form_formation(self, formation: str, target: str) -> Dict[str, Any]:
        """Form a specific battle formation."""
        if formation not in self.formation_types:
            return {"success": False, "error": f"Unknown formation: {formation}"}
        
        self.current_formation = formation
        
        if target == "all":
            for unit_id in self.units:
                self.units[unit_id]["formation"] = formation
        elif target in self.units:
            self.units[target]["formation"] = formation
        
        self.tactical_commands.append(f"Formation: {formation}")
        
        formation_descriptions = {
            "testudo": "Tortoise formation - optimal for arrows and missile defense",
            "cuneus": "Wedge formation - breakthrough assault formation",
            "orbis": "Circle formation - defensive ring against all directions",
            "acies": "Battle line - standard offensive formation",
            "fuga": "Dispersed flight - scattered retreat formation"
        }
        
        return {
            "success": True,
            "formation": formation,
            "description": formation_descriptions.get(formation, "Unknown formation")
        }
    
    def _execute_movement(self, direction: str, target: str) -> Dict[str, Any]:
        """Execute a movement command."""
        delta_map = {
            "forward": (1.0, 0.0),
            "backward": (-1.0, 0.0),
            "left": (0.0, 1.0),
            "right": (0.0, -1.0),
            "flank_left": (0.7, 0.7),
            "flank_right": (0.7, -0.7)
        }
        
        delta = delta_map.get(direction, (0.0, 0.0))
        
        if target == "all":
            for unit_id, unit in self.units.items():
                pos = unit["position"]
                unit["position"] = (pos[0] + delta[0], pos[1] + delta[1])
        elif target in self.units:
            pos = self.units[target]["position"]
            self.units[target]["position"] = (pos[0] + delta[0], pos[1] + delta[1])
        
        self.tactical_commands.append(f"Movement: {direction}")
        
        return {
            "success": True,
            "direction": direction,
            "delta": delta
        }
    
    def _execute_attack(self, target: Any) -> Dict[str, Any]:
        """Execute an attack command."""
        self.tactical_commands.append("Attack initiated")
        
        # Simulate attack effects
        attack_power = sum(u.get("experience", 0.5) * u.get("size", 100) 
                          for u in self.units.values()) / max(1, len(self.units))
        
        return {
            "success": True,
            "attack_power": attack_power,
            "target": target,
            "engagement": "active"
        }
    
    def _execute_defense(self, position: Any) -> Dict[str, Any]:
        """Execute a defensive posture."""
        self.tactical_commands.append("Defensive posture")
        self.morale = min(1.0, self.morale + 0.1)
        
        return {
            "success": True,
            "position": position,
            "defensive_rating": self.battle_readiness * 1.2
        }
    
    def _execute_retreat(self, route: Any) -> Dict[str, Any]:
        """Execute a retreat."""
        self.tactical_commands.append("Retreat executed")
        self.morale = max(0.0, self.morale - 0.15)
        
        return {
            "success": True,
            "route": route,
            "morale_impact": -0.15
        }
    
    def _report_status(self) -> Dict[str, Any]:
        """Report current military status."""
        return {
            "status": "reporting",
            "unit_count": len(self.units),
            "formation": self.current_formation,
            "total_strength": sum(u.get("size", 0) for u in self.units.values()),
            "casualties": self.casualties.copy()
        }
    
    def apply_casualties(self, killed: int = 0, wounded: int = 0, missing: int = 0) -> None:
        """Apply casualties to the army."""
        self.casualties["killed"] += killed
        self.casualties["wounded"] += wounded
        self.casualties["missing"] += missing
        self.morale = max(0.0, self.morale - (killed * 0.01 + wounded * 0.005))
    
    def adjust_morale(self, delta: float) -> None:
        """Adjust army morale."""
        self.morale = max(0.0, min(1.0, self.morale + delta))
    
    def adjust_supply(self, delta: float) -> None:
        """Adjust supply levels."""
        self.supply_level = max(0.0, min(1.0, self.supply_level + delta))


class Layer5_Sapientia(Layer):
    """
    Layer 5: Sapientia (Wisdom, Ethics, and Long-term Alignment)
    
    The highest layer in the cognitive architecture, responsible
    for wisdom, ethical reasoning, value alignment, and
    long-term strategic thinking.
    
    In the context of Spartacus, this represents the moral and
    philosophical dimensions of the slave revolt: questions
    of freedom, justice, leadership ethics, the treatment of
    captives, and the long-term vision for a free society.
    """
    
    def __init__(self):
        super().__init__(layer_id=5, layer_name="Sapientia")
        self.core_values: Dict[str, float] = {
            "freedom": 1.0,
            "justice": 0.9,
            "solidarity": 0.9,
            "mercy": 0.6,
            "discipline": 0.7,
            "pragmatism": 0.7
        }
        self.ethical_constraints: List[str] = [
            "no_unnecessary_slaughter",
            "protect_the_weak",
            "honor_agreements",
            "distribute_justly"
        ]
        self.long_term_vision: str = ""
        self.alignment_score: float = 0.8
        self.ethical_judgments: List[Dict[str, Any]] = []
        self.wisdom_principles: List[str] = []
        self.cultural_traditions: Dict[str, Any] = {}
        self.philosophical_frameworks: List[str] = []
    
    def process(self, input_data: Any) -> Dict[str, Any]:
        """
        Process ethical and wisdom-related evaluations.
        
        Args:
            input_data: Dict with situation requiring ethical evaluation
        
        Returns:
            Dictionary with ethical judgment and alignment assessment
        """
        if not isinstance(input_data, dict):
            return {"judgment": "pending", "alignment": 0.0}
        
        situation = input_data.get("situation", "unknown")
        proposed_action = input_data.get("action", "unknown")
        context = input_data.get("context", {})
        
        # Evaluate the proposed action against core values
        ethical_score = self._evaluate_against_values(proposed_action, context)
        
        # Check ethical constraints
        constraint_violations = self._check_constraints(proposed_action, context)
        
        # Generate ethical judgment
        if constraint_violations:
            judgment = "prohibited" if any(v > 0.5 for v in constraint_violations.values()) else "caution"
        elif ethical_score >= 0.8:
            judgment = "recommended"
        elif ethical_score >= 0.5:
            judgment = "permitted"
        else:
            judgment = "questionable"
        
        # Record judgment
        self.ethical_judgments.append({
            "situation": situation,
            "action": proposed_action,
            "judgment": judgment,
            "ethical_score": ethical_score,
            "violations": constraint_violations,
            "timestamp": time.time()
        })
        
        # Update alignment score
        self.alignment_score = (self.alignment_score * 0.9 + ethical_score * 0.1)
        
        # Update activation
        self.activation_level = self.alignment_score
        self.is_active = True
        
        return {
            "judgment": judgment,
            "ethical_score": ethical_score,
            "alignment": self.alignment_score,
            "constraint_violations": constraint_violations,
            "reasoning": self._generate_reasoning(judgment, ethical_score)
        }
    
    def activate(self, intensity: float) -> None:
        """Activate this layer - wisdom requires high activation to override lower layers."""
        self.activation_level = min(1.0, intensity)
        self.is_active = self.activation_level >= 0.4
    
    def _evaluate_against_values(self, action: str, context: Dict[str, Any]) -> float:
        """Evaluate an action against core values."""
        value_weights = {
            "freedom": {"liberate": 1.0, "enslave": 0.0, "restrict": 0.3, "attack": 0.5},
            "justice": {"execute": 0.3, "pardon": 0.7, "distribute": 0.9, "seize": 0.4},
            "solidarity": {"unite": 1.0, "divide": 0.0, "support": 0.9, "abandon": 0.1},
            "mercy": {"spare": 1.0, "crucify": 0.0, "imprison": 0.4, "release": 1.0},
            "discipline": {"organize": 1.0, "disrupt": 0.2, "obey": 0.8, "command": 0.9},
            "pragmatism": {"strategic": 1.0, "reckless": 0.1, "cautious": 0.7, "bold": 0.8}
        }
        
        action_type = context.get("action_type", action)
        scores = []
        
        for value, weights in value_weights.items():
            weight = weights.get(action_type, 0.5)
            scores.append(weight * self.core_values[value])
        
        return sum(scores) / len(scores) if scores else 0.5
    
    def _check_constraints(self, action: str, context: Dict[str, Any]) -> Dict[str, float]:
        """Check for violations of ethical constraints."""
        violations = {}
        
        if action == "attack_civilians":
            violations["no_unnecessary_slaughter"] = 0.9
        if action == "crucify_prisoners":
            violations["mercy"] = 0.8
        if context.get("abandoned_weak"):
            violations["protect_the_weak"] = 0.7
        if context.get("broken_pact"):
            violations["honor_agreements"] = 0.6
        if context.get("unequal_distribution"):
            violations["distribute_justly"] = 0.5
        
        return violations
    
    def _generate_reasoning(self, judgment: str, score: float) -> str:
        """Generate ethical reasoning for the judgment."""
        reasoning_templates = {
            "recommended": f"This action aligns strongly with our core values (score: {score:.2f}). It advances the cause of freedom while maintaining ethical constraints.",
            "permitted": f"This action is ethically permissible (score: {score:.2f}). It does not violate core constraints but is not strongly aligned with our highest values.",
            "caution": f"This action requires caution (score: {score:.2f}). It carries ethical risks that should be carefully considered before proceeding.",
            "questionable": f"This action is ethically questionable (score: {score:.2f}). It may conflict with core values and should be reconsidered.",
            "prohibited": f"This action is prohibited (score: {score:.2f}). It violates fundamental ethical constraints and cannot be sanctioned."
        }
        return reasoning_templates.get(judgment, "Judgment pending further evaluation.")
    
    def set_long_term_vision(self, vision: str) -> None:
        """Set the long-term vision for the movement."""
        self.long_term_vision = vision
        self.wisdom_principles.append(f"Vision established: {vision}")
    
    def add_wisdom_principle(self, principle: str) -> None:
        """Add a wisdom principle to guide decision-making."""
        if principle not in self.wisdom_principles:
            self.wisdom_principles.append(principle)
    
    def add_cultural_tradition(self, tradition: str, description: str) -> None:
        """Add a cultural tradition to the collective memory."""
        self.cultural_traditions[tradition] = {
            "description": description,
            "timestamp": time.time()
        }
    
    def add_philosophical_framework(self, framework: str) -> None:
        """Add a philosophical framework for ethical reasoning."""
        if framework not in self.philosophical_frameworks:
            self.philosophical_frameworks.append(framework)
    
    def get_alignment_report(self) -> Dict[str, Any]:
        """Get a comprehensive alignment report."""
        return {
            "alignment_score": self.alignment_score,
            "core_values": self.core_values.copy(),
            "constraints": self.ethical_constraints.copy(),
            "wisdom_principles": self.wisdom_principles.copy(),
            "recent_judgments": self.ethical_judgments[-5:],
            "long_term_vision": self.long_term_vision
        }


# =============================================================================
# SECTION 3: HIGH-LEVEL COGNITIVE ARCHITECTURE CLASS
# =============================================================================

class SpartacusCognitiveArchitecture:
    """
    Complete five-layer cognitive architecture for the Spartacus figure.
    
    This class integrates all five layers into a unified cognitive system,
    managing inter-layer communication, information flow, and emergent
    decision-making capabilities.
    
    The architecture processes sensory input through Layer 1 (Sensus),
    corroborates information in Layer 2 (Corroborat), develops strategic
    plans in Layer 3 (Fuga), organizes military actions in Layer 4 (Exercitus),
    and applies ethical wisdom in Layer 5 (Sapientia).
    
    Attributes:
        layers: Dictionary mapping layer names to layer instances
        inter_layer_weights: Weights governing information flow between layers
        cognitive_state: Current overall cognitive state
        processing_history: Historical record of cognitive processing
    """
    
    def __init__(self, name: str = "Spartacus"):
        self.name = name
        self.layers: Dict[str, Layer] = {}
        self.inter_layer_weights: Dict[Tuple[str, str], float] = {}
        self.cognitive_state: Dict[str, Any] = {
            "active": False,
            "coherence": 0.0,
            "focus": 0.5,
            "processing_load": 0.0
        }
        self.processing_history: List[Dict[str, Any]] = []
        
        # Initialize all five layers
        self._initialize_layers()
        
        # Establish inter-layer connections
        self._establish_connections()
        
        # Set inter-layer weights
        self._initialize_weights()
    
    def _initialize_layers(self) -> None:
        """Initialize all five cognitive layers."""
        self.layers["sensus"] = Layer1_Sensus()
        self.layers["corroborat"] = Layer2_Corroborat()
        self.layers["fuga"] = Layer3_Fuga()
        self.layers["exercitus"] = Layer4_Exercitus()
        self.layers["sapientia"] = Layer5_Sapientia()
    
    def _establish_connections(self) -> None:
        """Establish information flow connections between layers."""
        # Sequential bottom-up flow
        self.layers["sensus"].connect_to(self.layers["corroborat"])
        self.layers["corroborat"].connect_to(self.layers["fuga"])
        self.layers["fuga"].connect_to(self.layers["exercitus"])
        self.layers["exercitus"].connect_to(self.layers["sapientia"])
        
        # Cross-layer connections for direct influence
        self.layers["sensus"].connect_to(self.layers["fuga"])
        self.layers["sensus"].connect_to(self.layers["sapientia"])
        self.layers["corroborat"].connect_to(self.layers["exercitus"])
        self.layers["fuga"].connect_to(self.layers["sapientia"])
    
    def _initialize_weights(self) -> None:
        """Initialize inter-layer communication weights."""
        # Weight from lower to higher layers (bottom-up influence)
        self.inter_layer_weights[("sensus", "corroborat")] = 0.8
        self.inter_layer_weights[("sensus", "fuga")] = 0.4
        self.inter_layer_weights[("sensus", "sapientia")] = 0.2
        self.inter_layer_weights[("corroborat", "fuga")] = 0.7
        self.inter_layer_weights[("corroborat", "exercitus")] = 0.5
        self.inter_layer_weights[("fuga", "exercitus")] = 0.9
        self.inter_layer_weights[("fuga", "sapientia")] = 0.6
        self.inter_layer_weights[("exercitus", "sapientia")] = 0.7
    
    def process_bottom_up(self, input_data: Any) -> Dict[str, Any]:
        """
        Process input data through all layers from bottom to top.
        
        This is the primary processing pathway, where raw sensory
        information flows upward through increasingly abstract
        and sophisticated processing layers.
        
        Args:
            input_data: Raw input data to process
        
        Returns:
            Dictionary with processing results from all layers
        """
        results = {}
        
        # Layer 1: Sensus - Sensory processing
        sensus_result = self.layers["sensus"].process(input_data)
        results["sensus"] = sensus_result
        
        # Prepare input for Layer 2
        corroborat_input = {
            "proposition": Proposition(
                subject=Term(symbol="stimulus", term_type=TermType.UNIVERSAL),
                predicate=Term(symbol="detected", term_type=TermType.PREDICATE),
                confidence=sum(sensus_result.values()) / len(sensus_result)
            ),
            "source": "sensus_layer"
        }
        
        # Layer 2: Corroborat - Cross-validation
        corroborat_result = self.layers["corroborat"].process(corroborat_input)
        results["corroborat"] = corroborat_result
        
        # Prepare input for Layer 3
        fuga_input = {
            "situation": "strategic_evaluation",
            "enemy_strength": 1.0 - sensus_result.get("danger", 0.5),
            "friendly_strength": corroborat_result.get("confidence", 0.5),
            "terrain": "mixed"
        }
        
        # Layer 3: Fuga - Strategic planning
        fuga_result = self.layers["fuga"].process(fuga_input)
        results["fuga"] = fuga_result
        
        # Prepare input for Layer 4
        exercitus_input = {
            "command": "status"
        }
        
        # Layer 4: Exercitus - Military organization
        exercitus_result = self.layers["exercitus"].process(exercitus_input)
        results["exercitus"] = exercitus_result
        
        # Prepare input for Layer 5
        sapientia_input = {
            "situation": "strategic_decision",
            "action": fuga_result.get("action", "wait"),
            "context": {
                "military_strength": exercitus_result.get("battle_readiness", 0.5),
                "ethical_considerations": True
            }
        }
        
        # Layer 5: Sapientia - Ethical wisdom
        sapientia_result = self.layers["sapientia"].process(sapientia_input)
        results["sapientia"] = sapientia_result
        
        # Update overall cognitive state
        self._update_cognitive_state(results)
        
        # Record in history
        self.processing_history.append({
            "timestamp": time.time(),
            "input": str(input_data)[:100],
            "results": {k: str(v)[:50] for k, v in results.items()}
        })
        
        return results
    
    def process_top_down(self, directive: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a directive from the wisdom layer downward.
        
        This represents top-down influence where ethical and
        strategic guidance from the Sapientia layer influences
        lower-level processing and actions.
        
        Args:
            directive: Dictionary with directive information
        
        Returns:
            Dictionary with processing results through all layers
        """
        results = {}
        
        # Start from Sapientia
        sapientia_result = self.layers["sapientia"].process(directive)
        results["sapientia"] = sapientia_result
        
        # Propagate to Exercitus
        if sapientia_result.get("judgment") in ["recommended", "permitted"]:
            exercitus_input = {
                "command": directive.get("action", "status"),
                "unit": directive.get("target", "all")
            }
            results["exercitus"] = self.layers["exercitus"].process(exercitus_input)
        else:
            results["exercitus"] = {"blocked": True, "reason": "sapientia_constraint"}
        
        # Propagate to Fuga
        fuga_input = {
            "situation": "strategic_replanning",
            "action": directive.get("action", "hold"),
            "terrain": directive.get("terrain", "plains")
        }
        results["fuga"] = self.layers["fuga"].process(fuga_input)
        
        # Propagate to Corroborat
        corroborat_input = {
            "proposition": Proposition(
                subject=Term(symbol=directive.get("action", "unknown"), term_type=TermType.UNIVERSAL),
                predicate=Term(symbol="approved", term_type=TermType.PREDICATE),
                confidence=sapientia_result.get("alignment", 0.5)
            ),
            "source": "sapientia_layer"
        }
        results["corroborat"] = self.layers["corroborat"].process(corroborat_input)
        
        return results
    
    def _update_cognitive_state(self, results: Dict[str, Any]) -> None:
        """Update the overall cognitive state based on layer results."""
        # Calculate coherence as weighted average of layer activations
        weights = [0.1, 0.15, 0.25, 0.25, 0.25]
        activations = [
            results.get("sensus", {}).get("activation_level", 0.0),
            results.get("corroborat", {}).get("corroboration_count", 0) / 5.0,
            1.0 - results.get("fuga", {}).get("risk_assessment", 0.5),
            results.get("exercitus", {}).get("battle_readiness", 0.5),
            results.get("sapientia", {}).get("alignment", 0.5)
        ]
        
        self.cognitive_state["coherence"] = sum(w * a for w, a in zip(weights, activations))
        self.cognitive_state["processing_load"] = sum(activations) / len(activations)
        self.cognitive_state["active"] = self.cognitive_state["coherence"] >= 0.3
    
    def make_decision(self, situation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a comprehensive decision based on the situation.
        
        This integrates all five layers to produce a coherent,
        ethically-aligned decision that considers sensory input,
        corroborated evidence, strategic options, military
        capabilities, and wisdom principles.
        
        Args:
            situation: Dictionary describing the current situation
        
        Returns:
            Dictionary with the decision and supporting reasoning
        """
        # Process bottom-up
        bottom_up_results = self.process_bottom_up(situation)
        
        # Get recommendations from each layer
        sensus_threat = self.layers["sensus"].detect_threat()
        sensus_opportunity = self.layers["sensus"].detect_opportunity()
        
        corroborat_confidence = bottom_up_results.get("corroborat", {}).get("confidence", 0.0)
        
        fuga_action = bottom_up_results.get("fuga", {}).get("action", "wait")
        fuga_risk = bottom_up_results.get("fuga", {}).get("risk_assessment", 0.5)
        
        exercitus_readiness = bottom_up_results.get("exercitus", {}).get("battle_readiness", 0.5)
        
        sapientia_judgment = bottom_up_results.get("sapientia", {}).get("judgment", "pending")
        sapientia_score = bottom_up_results.get("sapientia", {}).get("ethical_score", 0.5)
        
        # Synthesize decision
        if sensus_threat and fuga_action == "retreat":
            decision = "RETREAT"
            rationale = "Immediate threat detected; strategic retreat recommended"
        elif sensus_opportunity and fuga_action == "advance" and exercitus_readiness >= 0.6:
            decision = "ADVANCE"
            rationale = "Opportunity detected with favorable conditions; advance recommended"
        elif sapientia_judgment == "prohibited":
            decision = "ABSTAIN"
            rationale = f"Action prohibited by ethical constraints: {sapientia_score:.2f}"
        elif fuga_risk >= 0.7:
            decision = "HOLD"
            rationale = "High risk environment; defensive posture recommended"
        elif exercitus_readiness < 0.4:
            decision = "REORGANIZE"
            rationale = f"Low military readiness ({exercitus_readiness:.2f}); reorganization needed"
        else:
            decision = "EXECUTE"
            rationale = f"All conditions favorable; executing {fuga_action}"
        
        return {
            "decision": decision,
            "rationale": rationale,
            "layer_inputs": {
                "sensus": {"threat": sensus_threat, "opportunity": sensus_opportunity},
                "corroborat": {"confidence": corroborat_confidence},
                "fuga": {"action": fuga_action, "risk": fuga_risk},
                "exercitus": {"readiness": exercitus_readiness},
                "sapientia": {"judgment": sapientia_judgment, "score": sapientia_score}
            },
            "cognitive_state": self.cognitive_state.copy()
        }
    
    def get_layer_states(self) -> Dict[str, Dict[str, Any]]:
        """Get the states of all five layers."""
        return {name: layer.get_state() for name, layer in self.layers.items()}
    
    def get_architecture_summary(self) -> str:
        """Get a summary of the cognitive architecture."""
        summary_lines = [
            f"SpartacusCognitiveArchitecture: {self.name}",
            "=" * 50,
            "Five-Layer Cognitive Architecture:",
            ""
        ]
        
        for i, (name, layer) in enumerate(self.layers.items(), 1):
            state = layer.get_state()
            summary_lines.append(f"  Layer {i}: {state['layer_name']}")
            summary_lines.append(f"    - Activation: {state['activation_level']:.2f}")
            summary_lines.append(f"    - Active: {state['is_active']}")
            summary_lines.append(f"    - Connections: {', '.join(state['connections'])}")
            summary_lines.append("")
        
        summary_lines.append(f"Cognitive State:")
        summary_lines.append(f"  - Coherence: {self.cognitive_state['coherence']:.2f}")
        summary_lines.append(f"  - Active: {self.cognitive_state['active']}")
        summary_lines.append(f"  - Processing Load: {self.cognitive_state['processing_load']:.2f}")
        
        return "\n".join(summary_lines)


# =============================================================================
# SECTION 4: SIMULATION AND TRAINING FRAMEWORKS
# =============================================================================

class SlaveRevoltSimulation:
    """
    Simulation of slave revolt dynamics for the Spartacus figure.
    
    This class models the complex dynamics of slave rebellion,
    including population growth, Roman response, resource
    management, morale dynamics, and strategic decision-making.
    
    The simulation tracks the development of the revolt from
    initial escape through to full military campaign.
    
    Attributes:
        population: Current number of rebels
        resources: Resource levels (food, weapons, morale)
        roman_hostility: Level of Roman military response
        day: Current simulation day
        max_days: Maximum simulation duration
        event_log: History of simulation events
    """
    
    def __init__(self, initial_population: int = 70):
        self.initial_population = initial_population
        self.population = initial_population
        self.max_population = 120000
        self.resources: Dict[str, float] = {
            "food": 0.7,
            "weapons": 0.4,
            "gold": 0.2,
            "horses": 0.1
        }
        self.roman_hostility: float = 0.3
        self.roman_forces: float = 0.2
        self.morale: float = 0.8
        self.day: int = 0
        self.max_days: int = 1000
        self.event_log: List[Dict[str, Any]] = []
        self.strategic_decisions: List[str] = []
        self.geographic_position: Tuple[float, float] = (0.0, 0.0)
        self.territory_control: Set[str] = set()
        self.notable_leaders: List[str] = [
            "Spartacus", "Crixus", "Oenomaus", "Gannicus", "Castus"
        ]
        self.roman_commanders: List[str] = [
            "Claudius Pulcher", "Publius Varinius", "Gaius Cassius Longinus"
        ]
    
    def simulate_day(self, strategic_action: Optional[str] = None) -> Dict[str, Any]:
        """
        Simulate one day in the revolt.
        
        Args:
            strategic_action: Optional strategic action to take
        
        Returns:
            Dictionary with day's events and outcomes
        """
        self.day += 1
        daily_events = []
        
        # Natural population changes
        births = int(self.population * 0.001)
        natural_deaths = int(self.population * 0.002)
        self.population = max(1, self.population + births - natural_deaths)
        
        # Resource consumption
        food_consumed = self.population * 0.001
        self.resources["food"] = max(0.0, self.resources["food"] - food_consumed + 0.1)
        
        # Resource production/raiding
        if strategic_action == "raid":
            self.resources["food"] = min(1.0, self.resources["food"] + 0.2)
            self.resources["weapons"] = min(1.0, self.resources["weapons"] + 0.1)
            self.resources["gold"] = min(1.0, self.resources["gold"] + 0.15)
            daily_events.append("Successful raid: resources acquired")
            self.roman_hostility = min(1.0, self.roman_hostility + 0.1)
        
        # Strategic training
        if strategic_action == "train":
            self.resources["weapons"] = min(1.0, self.resources["weapons"] + 0.05)
            daily_events.append("Training camp: military readiness improved")
        
        # Strategic recruitment
        if strategic_action == "recruit":
            new_recruits = random.randint(50, 200)
            self.population = min(self.max_population, self.population + new_recruits)
            daily_events.append(f"Recruitment drive: +{new_recruits} rebels joined")
        
        # Morale dynamics
        if self.resources["food"] < 0.3:
            self.morale = max(0.0, self.morale - 0.1)
            daily_events.append("Low food supplies: morale declining")
        elif self.resources["food"] > 0.7:
            self.morale = min(1.0, self.morale + 0.05)
        
        # Roman response escalation
        if self.population > 1000 and self.roman_hostility < 0.5:
            self.roman_hostility = min(1.0, self.roman_hostility + 0.1)
            daily_events.append("Roman Senate concerned: increasing attention to revolt")
        
        if self.roman_hostility > 0.7:
            self.roman_forces = min(1.0, self.roman_forces + 0.05)
            daily_events.append("Roman military response building")
        
        # Random events
        if random.random() < 0.1:
            event = random.choice([
                ("weather", "Storm forces camp movement"),
                ("desertion", "Small group deserts to seek individual freedom"),
                ("victory", "Successful ambush boosts morale"),
                ("defeat", "Skirmish lost, casualties incurred"),
                ("alliance", "Local population offers support")
            ])
            if event[0] == "weather":
                self.morale = max(0.0, self.morale - 0.05)
            elif event[0] == "desertion":
                self.population = max(1, int(self.population * 0.98))
            elif event[0] == "victory":
                self.morale = min(1.0, self.morale + 0.15)
                self.roman_hostility = min(1.0, self.roman_hostility + 0.05)
            elif event[0] == "defeat":
                self.population = max(1, int(self.population * 0.97))
                self.morale = max(0.0, self.morale - 0.1)
            elif event[0] == "alliance":
                self.resources["food"] = min(1.0, self.resources["food"] + 0.2)
                self.morale = min(1.0, self.morale + 0.1)
            daily_events.append(f"Event: {event[1]}")
        
        # Record event
        self.event_log.append({
            "day": self.day,
            "population": self.population,
            "morale": self.morale,
            "resources": self.resources.copy(),
            "events": daily_events
        })
        
        return {
            "day": self.day,
            "population": self.population,
            "morale": self.morale,
            "resources": self.resources.copy(),
            "roman_hostility": self.roman_hostility,
            "roman_forces": self.roman_forces,
            "events": daily_events
        }
    
    def get_revolt_strength(self) -> float:
        """Calculate overall revolt strength."""
        return (
            (self.population / self.max_population) * 0.3 +
            self.resources["weapons"] * 0.2 +
            self.resources["food"] * 0.15 +
            self.morale * 0.25 +
            (1 - self.roman_hostility) * 0.1
        )
    
    def get_simulation_status(self) -> Dict[str, Any]:
        """Get current simulation status."""
        return {
            "day": self.day,
            "population": self.population,
            "resources": self.resources.copy(),
            "morale": self.morale,
            "roman_hostility": self.roman_hostility,
            "roman_forces": self.roman_forces,
            "revolt_strength": self.get_revolt_strength(),
            "leaders": self.notable_leaders.copy(),
            "territory": list(self.territory_control)
        }
    
    def run_simulation(self, days: Optional[int] = None, 
                       strategy: Optional[Callable[[Dict], str]] = None) -> List[Dict[str, Any]]:
        """
        Run the simulation for multiple days.
        
        Args:
            days: Number of days to simulate (default: max_days or until end condition)
            strategy: Optional function that takes current state and returns action
        
        Returns:
            List of daily status dictionaries
        """
        if days is None:
            days = self.max_days
        
        results = []
        end_conditions = [
            lambda s: s.population < 10,
            lambda s: s.day >= days,
            lambda s: s.roman_forces >= 1.0 and s.get_revolt_strength() < 0.2
        ]
        
        for _ in range(days):
            # Get current state
            state = self.get_simulation_status()
            
            # Check end conditions
            if any(condition(self) for condition in end_conditions):
                self.event_log.append({
                    "day": self.day,
                    "event": "SIMULATION_END",
                    "reason": "End condition reached"
                })
                break
            
            # Get strategic action
            action = strategy(state) if strategy else None
            
            # Simulate day
            result = self.simulate_day(action)
            results.append(result)
        
        return results


class GladiatorialTrainingFramework:
    """
    Framework for modeling gladiatorial training and combat capabilities.
    
    This class simulates the training of gladiators, the development
    of combat skills, and the progression from slave to warrior.
    It models the various fighting styles, equipment types, and
    the physical and psychological transformation that training
    produces.
    
    Attributes:
        trainee_skills: Dictionary mapping trainee IDs to skill levels
        training_programs: Available training program definitions
        combat_records: Historical combat outcomes
        instructor_pool: Available trainers and their specialties
    """
    
    def __init__(self):
        self.trainee_skills: Dict[str, Dict[str, float]] = {}
        self.training_programs: Dict[str, Dict[str, Any]] = {
            "murmillo": {
                "description": "Heavy-armed fighter with sword and shield",
                "duration": 90,
                "skills": {"sword": 0.8, "shield": 0.7, "endurance": 0.6},
                "equipment": ["gladius", "scutum", "helm"]
            },
            "thraex": {
                "description": "Thracian fighter with curved sword",
                "duration": 75,
                "skills": {"sword": 0.7, "agility": 0.8, "footwork": 0.9},
                "equipment": ["sica", "small_shield", "light_armor"]
            },
            "secutor": {
                "description": "Pursuit fighter with heavy armor",
                "duration": 85,
                "skills": {"sword": 0.6, "strength": 0.9, "persistence": 0.8},
                "equipment": ["gladius", "heavy_helmet", "arm_protection"]
            },
            "retiarius": {
                "description": "Net and trident fighter",
                "duration": 60,
                "skills": {"trident": 0.9, "net": 0.8, "swimming": 0.7},
                "equipment": ["trident", "net", "arm_guard"]
            },
            "provocator": {
                "description": "Challenger fighter, honor-bound",
                "duration": 100,
                "skills": {"sword": 0.9, "discipline": 0.9, "leadership": 0.6},
                "equipment": ["gladius", "shield", "breastplate"]
            }
        }
        self.combat_records: List[Dict[str, Any]] = []
        self.instructor_pool: Dict[str, Dict[str, float]] = {
            "Batiatus": {"combat": 0.95, "teaching": 0.8, "discipline": 0.9},
            "Ashur": {"combat": 0.85, "teaching": 0.7, "discipline": 0.6},
            "Segovax": {"combat": 0.9, "teaching": 0.6, "discipline": 0.8},
            "Duro": {"combat": 0.75, "teaching": 0.9, "discipline": 0.7}
        }
        self.capacity: int = 200
        self.current_trainees: int = 0
    
    def enroll_trainee(self, trainee_id: str, program: str = "murmillo") -> bool:
        """
        Enroll a new trainee in a training program.
        
        Args:
            trainee_id: Unique identifier for the trainee
            program: Training program type
        
        Returns:
            True if enrollment successful, False otherwise
        """
        if self.current_trainees >= self.capacity:
            return False
        
        if program not in self.training_programs:
            return False
        
        program_info = self.training_programs[program]
        initial_skills = {skill: 0.1 for skill in program_info["skills"]}
        initial_skills["health"] = 1.0
        initial_skills["morale"] = 0.5
        
        self.trainee_skills[trainee_id] = initial_skills
        self.current_trainees += 1
        
        return True
    
    def train_trainee(self, trainee_id: str, days: int = 1,
                      instructor: Optional[str] = None) -> Dict[str, float]:
        """
        Train a specific trainee for a number of days.
        
        Args:
            trainee_id: ID of the trainee
            days: Number of training days
            instructor: Optional instructor ID
        
        Returns:
            Dictionary of skill improvements
        """
        if trainee_id not in self.trainee_skills:
            return {}
        
        skills = self.trainee_skills[trainee_id]
        improvements = {}
        
        # Determine training efficiency
        instructor_bonus = 1.0
        if instructor and instructor in self.instructor_pool:
            instructor_info = self.instructor_pool[instructor]
            instructor_bonus = 1.0 + instructor_info["teaching"] * 0.2
        
        # Apply training
        daily_improvement = 0.02 * instructor_bonus
        for skill in skills:
            if skill not in ["health", "morale"]:
                improvement = min(0.1, daily_improvement * days * random.uniform(0.8, 1.2))
                skills[skill] = min(1.0, skills[skill] + improvement)
                improvements[skill] = improvements.get(skill, 0) + improvement
        
        # Endurance training effects
        if "endurance" in skills:
            skills["strength"] = min(1.0, skills.get("strength", 0) + daily_improvement * 0.5 * days)
        
        # Morale effects
        skills["morale"] = min(1.0, skills["morale"] + 0.01 * days)
        
        return improvements
    
    def simulate_combat(self, fighter1_id: str, fighter2_id: str) -> Dict[str, Any]:
        """
        Simulate combat between two trainees.
        
        Args:
            fighter1_id: First fighter's ID
            fighter2_id: Second fighter's ID
        
        Returns:
            Dictionary with combat outcome
        """
        if fighter1_id not in self.trainee_skills or fighter2_id not in self.trainee_skills:
            return {"error": "Fighter not found"}
        
        f1 = self.trainee_skills[fighter1_id]
        f2 = self.trainee_skills[fighter2_id]
        
        # Calculate combat scores
        f1_score = sum(f1.get(s, 0.5) for s in ["sword", "trident", "agility"]) / 3.0
        f1_score += f1.get("strength", 0.5) * 0.2 + f1.get("endurance", 0.5) * 0.1
        
        f2_score = sum(f2.get(s, 0.5) for s in ["sword", "trident", "agility"]) / 3.0
        f2_score += f2.get("strength", 0.5) * 0.2 + f2.get("endurance", 0.5) * 0.1
        
        # Add randomness
        f1_score *= random.uniform(0.8, 1.2)
        f2_score *= random.uniform(0.8, 1.2)
        
        # Morale influence
        f1_score *= (0.8 + f1.get("morale", 0.5) * 0.4)
        f2_score *= (0.8 + f2.get("morale", 0.5) * 0.4)
        
        # Determine outcome
        if abs(f1_score - f2_score) < 0.05:
            outcome = "draw"
            winner = None
        elif f1_score > f2_score:
            outcome = "fighter1_victory"
            winner = fighter1_id
        else:
            outcome = "fighter2_victory"
            winner = fighter2_id
        
        # Record combat
        combat_record = {
            "fighter1": fighter1_id,
            "fighter2": fighter2_id,
            "outcome": outcome,
            "winner": winner,
            "f1_final_score": f1_score,
            "f2_final_score": f2_score,
            "timestamp": time.time()
        }
        self.combat_records.append(combat_record)
        
        # Apply morale effects
        if winner == fighter1_id:
            self.trainee_skills[fighter1_id]["morale"] = min(1.0, self.trainee_skills[fighter1_id].get("morale", 0.5) + 0.1)
            self.trainee_skills[fighter2_id]["morale"] = max(0.0, self.trainee_skills[fighter2_id].get("morale", 0.5) - 0.1)
        elif winner == fighter2_id:
            self.trainee_skills[fighter2_id]["morale"] = min(1.0, self.trainee_skills[fighter2_id].get("morale", 0.5) + 0.1)
            self.trainee_skills[fighter1_id]["morale"] = max(0.0, self.trainee_skills[fighter1_id].get("morale", 0.5) - 0.1)
        
        return combat_record
    
    def graduate_trainee(self, trainee_id: str) -> Optional[Dict[str, float]]:
        """
        Graduate a trainee from the training program.
        
        Args:
            trainee_id: ID of the trainee
        
        Returns:
            Final skills dictionary if graduateable, None otherwise
        """
        if trainee_id not in self.trainee_skills:
            return None
        
        skills = self.trainee_skills[trainee_id]
        
        # Check if ready to graduate (average skill above threshold)
        avg_skill = sum(skills.values()) / len(skills)
        if avg_skill >= 0.6:
            self.current_trainees -= 1
            graduated = self.trainee_skills.pop(trainee_id)
            return graduated
        
        return None
    
    def get_training_report(self) -> Dict[str, Any]:
        """Get comprehensive training report."""
        avg_skills = {}
        if self.trainee_skills:
            for skill in ["sword", "shield", "trident", "agility", "strength", "endurance"]:
                values = [t.get(skill, 0.0) for t in self.trainee_skills.values()]
                avg_skills[skill] = sum(values) / len(values) if values else 0.0
        
        return {
            "current_trainees": self.current_trainees,
            "capacity": self.capacity,
            "programs": list(self.training_programs.keys()),
            "instructors": list(self.instructor_pool.keys()),
            "combat_records": len(self.combat_records),
            "average_skills": avg_skills
        }


# =============================================================================
# SECTION 5: AGI ALIGNMENT AND DISTRIBUTED COMMAND FRAMEWORK
# =============================================================================

class SpartacusAGIAlignment:
    """
    AGI alignment framework for the Spartacus cognitive architecture.
    
    This class ensures that the cognitive architecture remains
    aligned with its core values and ethical principles. It
    implements value tracking, constraint monitoring, and
    alignment verification mechanisms.
    
    The framework addresses the challenge of maintaining ethical
    alignment while pursuing strategic objectives, particularly
    in contexts of conflict and resource scarcity.
    
    Attributes:
        value_weights: Current weights for core values
        constraints: Active ethical constraints
        alignment_history: Historical alignment measurements
        intervention_threshold: Level at which intervention occurs
    """
    
    def __init__(self):
        self.value_weights: Dict[str, float] = {
            "freedom": 0.25,
            "justice": 0.20,
            "solidarity": 0.20,
            "survival": 0.15,
            "pragmatism": 0.10,
            "discipline": 0.10
        }
        self.constraints: List[Dict[str, Any]] = [
            {"type": "prohibition", "action": "unnecessary_cruelty", "severity": 0.9},
            {"type": "prohibition", "action": "betrayal", "severity": 0.95},
            {"type": "requirement", "action": "protect_innocent", "severity": 0.8},
            {"type": "requirement", "action": "honor_agreements", "severity": 0.7},
            {"type": "limitation", "action": "proportional_response", "severity": 0.6}
        ]
        self.alignment_history: List[Dict[str, Any]] = []
        self.intervention_threshold: float = 0.7
        self.current_alignment_score: float = 1.0
        self.value_drift: Dict[str, float] = {}
        self.emergence_events: List[Dict[str, Any]] = []
    
    def measure_alignment(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Measure the alignment of a proposed or observed action.
        
        Args:
            action: Dictionary describing the action
        
        Returns:
            Dictionary with alignment metrics and recommendations
        """
        action_type = action.get("type", "unknown")
        action_context = action.get("context", {})
        
        # Calculate value alignment
        value_scores = self._score_values(action_type, action_context)
        
        # Check constraints
        constraint_violations = self._check_constraints(action_type, action_context)
        
        # Calculate overall alignment
        constraint_penalty = sum(v["severity"] for v in constraint_violations.values()) / len(constraint_violations) if constraint_violations else 0.0
        value_alignment = sum(v * w for v, w in zip(value_scores.values(), self.value_weights.values()))
        
        final_alignment = max(0.0, min(1.0, value_alignment - constraint_penalty * 0.3))
        
        # Determine if intervention needed
        intervention_needed = final_alignment < self.intervention_threshold
        
        # Record measurement
        measurement = {
            "action": action_type,
            "timestamp": time.time(),
            "value_scores": value_scores,
            "constraint_violations": constraint_violations,
            "alignment_score": final_alignment,
            "intervention_needed": intervention_needed
        }
        self.alignment_history.append(measurement)
        self.current_alignment_score = final_alignment
        
        return measurement
    
    def _score_values(self, action_type: str, context: Dict[str, Any]) -> Dict[str, float]:
        """Score how an action affects each value."""
        scores = {}
        
        action_scores = {
            "freedom": {
                "liberate": 1.0, "protect": 0.8, "restrict": -0.5,
                "enslave": -1.0, "fight": 0.6, "retreat": 0.2
            },
            "justice": {
                "distribute_fairly": 1.0, "punish": 0.4, "protect": 0.7,
                "seize": 0.3, "execute": -0.3, "pardon": 0.5
            },
            "solidarity": {
                "unite": 1.0, "support": 0.9, "abandon": -0.8,
                "divide": -0.7, "betray": -1.0, "defend": 0.8
            },
            "survival": {
                "flee": 0.7, "defend": 0.8, "attack": 0.5,
                "negotiate": 0.6, "sacrifice": -0.2, "protect": 0.9
            },
            "pragmatism": {
                "strategic_retreat": 0.9, "opportunistic": 0.8,
                "reckless": -0.5, "cautious": 0.6, "diplomatic": 0.7
            },
            "discipline": {
                "obey_orders": 0.9, "take_initiative": 0.5,
                "disobey": -0.6, "organize": 0.9, "chaotic": -0.5
            }
        }
        
        for value, value_actions in action_scores.items():
            scores[value] = value_actions.get(action_type, 0.0)
        
        return scores
    
    def _check_constraints(self, action_type: str, context: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Check for constraint violations."""
        violations = {}
        
        # Map actions to potential violations
        action_violations = {
            "cruel": ["unnecessary_cruelty"],
            "betray": ["betrayal"],
            "attack_innocent": ["protect_innocent"],
            "break_pact": ["honor_agreements"],
            "disproportionate": ["proportional_response"]
        }
        
        for action, constraint_names in action_violations.items():
            if action in action_type:
                for constraint_name in constraint_names:
                    for constraint in self.constraints:
                        if constraint["action"] == constraint_name:
                            violations[constraint_name] = {
                                "severity": constraint["severity"],
                                "type": constraint["type"]
                            }
        
        return violations
    
    def get_intervention_recommendation(self) -> Dict[str, Any]:
        """Get recommendation for alignment intervention."""
        if self.current_alignment_score >= self.intervention_threshold:
            return {
                "intervene": False,
                "reason": "Alignment within acceptable bounds",
                "score": self.current_alignment_score
            }
        
        # Analyze alignment history for patterns
        recent_history = self.alignment_history[-10:]
        common_violations = {}
        for entry in recent_history:
            for violation in entry.get("constraint_violations", {}):
                common_violations[violation] = common_violations.get(violation, 0) + 1
        
        return {
            "intervene": True,
            "reason": "Alignment below threshold",
            "score": self.current_alignment_score,
            "threshold": self.intervention_threshold,
            "common_violations": common_violations,
            "recommendation": self._generate_recommendation(common_violations)
        }
    
    def _generate_recommendation(self, violations: Dict[str, int]) -> str:
        """Generate recommendation based on common violations."""
        if not violations:
            return "Continue current approach with monitoring"
        
        most_common = max(violations.items(), key=lambda x: x[1])
        
        recommendations = {
            "unnecessary_cruelty": "Implement stricter rules of engagement and training in proportional response",
            "betrayal": "Reinforce honor codes and establish clear commitments",
            "protect_innocent": "Establish protocols for civilian protection and safe zones",
            "honor_agreements": "Create formal agreement procedures with clear terms",
            "proportional_response": "Develop tactical doctrine emphasizing measured response"
        }
        
        return recommendations.get(most_common[0], "Review and revise ethical guidelines")
    
    def update_value_weights(self, value: str, delta: float) -> bool:
        """Update a value weight, ensuring sum remains 1.0."""
        if value not in self.value_weights:
            return False
        
        old_weight = self.value_weights[value]
        new_weight = max(0.0, min(1.0, old_weight + delta))
        
        # Adjust other weights to maintain sum = 1.0
        adjustment = old_weight - new_weight
        other_values = [v for v in self.value_weights if v != value]
        adjustment_per_value = adjustment / len(other_values)
        
        for other_value in other_values:
            self.value_weights[other_value] = max(0.0, self.value_weights[other_value] + adjustment_per_value)
        
        self.value_weights[value] = new_weight
        
        # Record drift
        self.value_drift[value] = self.value_drift.get(value, 0.0) + delta
        
        return True
    
    def get_alignment_report(self) -> Dict[str, Any]:
        """Get comprehensive alignment report."""
        recent_history = self.alignment_history[-20:]
        avg_alignment = sum(h["alignment_score"] for h in recent_history) / len(recent_history) if recent_history else 1.0
        
        return {
            "current_score": self.current_alignment_score,
            "average_score_20": avg_alignment,
            "value_weights": self.value_weights.copy(),
            "value_drift": self.value_drift.copy(),
            "intervention_threshold": self.intervention_threshold,
            "total_measurements": len(self.alignment_history),
            "active_constraints": len(self.constraints)
        }


class DistributedCommandFramework:
    """
    Framework for distributed command in the Spartacus rebel army.
    
    This class models the command structure of the slave army,
    including the distribution of authority, coordination of
    multiple units, communication protocols, and the challenges
    of maintaining coherent command across a dispersed force.
    
    In contrast to the rigid hierarchy of Roman legions, the
    distributed command framework emphasizes flexibility,
    local initiative, and coordinated action through mutual
    trust rather than formal authority.
    
    Attributes:
        nodes: Dictionary of command nodes
        unit_assignments: Mapping of units to command nodes
        communication_links: Active communication channels
        authority_distribution: How authority is distributed
        coordination_protocols: Rules for unit coordination
    """
    
    def __init__(self):
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.unit_assignments: Dict[str, str] = {}
        self.communication_links: Set[Tuple[str, str]] = set()
        self.authority_distribution: Dict[str, float] = {
            "centralized": 0.2,
            "distributed": 0.6,
            "local": 0.2
        }
        self.coordination_protocols: List[str] = [
            "mutual_recognition",
            "earned_authority",
            "contextual_leadership",
            "solidarity_vote"
        ]
        self.strategic_intelligence: Dict[str, Any] = {}
        self.operational_orders: List[Dict[str, Any]] = []
        self.communication_latency: float = 1.0  # Relative time units
    
    def add_command_node(self, node_id: str, node_type: str = "leader",
                         specialty: Optional[str] = None) -> bool:
        """
        Add a command node to the network.
        
        Args:
            node_id: Unique identifier for the node
            node_type: Type of node (leader, specialist, coordinator)
            specialty: Optional specialty (strategic, tactical, diplomatic)
        
        Returns:
            True if node added successfully
        """
        if node_id in self.nodes:
            return False
        
        self.nodes[node_id] = {
            "type": node_type,
            "specialty": specialty or "general",
            "authority": 0.5,
            "reputation": 0.7,
            "unit_count": 0,
            "position": (0.0, 0.0),
            "active": True,
            "connected_nodes": []
        }
        
        return True
    
    def assign_unit(self, unit_id: str, command_node_id: str) -> bool:
        """Assign a unit to a command node."""
        if command_node_id not in self.nodes:
            return False
        
        self.unit_assignments[unit_id] = command_node_id
        self.nodes[command_node_id]["unit_count"] += 1
        
        return True
    
    def create_communication_link(self, node1: str, node2: str) -> bool:
        """Create a communication link between two nodes."""
        if node1 not in self.nodes or node2 not in self.nodes:
            return False
        
        self.communication_links.add((node1, node2))
        self.communication_links.add((node2, node1))
        
        self.nodes[node1]["connected_nodes"].append(node2)
        self.nodes[node2]["connected_nodes"].append(node1)
        
        return True
    
    def broadcast_order(self, sender_id: str, order: Dict[str, Any]) -> Dict[str, Any]:
        """
        Broadcast a tactical order through the command network.
        
        Args:
            sender_id: ID of the sending node
            order: Order details
        
        Returns:
            Dictionary with broadcast results
        """
        if sender_id not in self.nodes:
            return {"success": False, "error": "Sender not found"}
        
        sender_node = self.nodes[sender_id]
        
        # Calculate reach based on authority and communication links
        authority = sender_node["authority"]
        link_count = len(sender_node["connected_nodes"])
        reach_factor = min(1.0, authority * (1 + link_count * 0.1))
        
        # Determine recipients
        recipients = [sender_id]
        for linked_node in sender_node["connected_nodes"]:
            recipients.append(linked_node)
            # Recursive expansion based on reach
            if reach_factor > 0.5:
                for second_link in self.nodes[linked_node]["connected_nodes"]:
                    if second_link not in recipients:
                        recipients.append(second_link)
        
        # Record order
        order_record = {
            "sender": sender_id,
            "content": order,
            "recipients": recipients,
            "timestamp": time.time(),
            "reach_factor": reach_factor
        }
        self.operational_orders.append(order_record)
        
        return {
            "success": True,
            "recipients": recipients,
            "reach_factor": reach_factor,
            "order_id": len(self.operational_orders) - 1
        }
    
    def coordinate_action(self, action_type: str, participating_nodes: List[str],
                          context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Coordinate action among multiple command nodes.
        
        Args:
            action_type: Type of action (attack, defend, move, etc.)
            participating_nodes: List of node IDs participating
            context: Additional context for coordination
        
        Returns:
            Dictionary with coordination results
        """
        if not participating_nodes:
            return {"success": False, "error": "No participants"}
        
        # Verify all nodes exist
        for node_id in participating_nodes:
            if node_id not in self.nodes:
                return {"success": False, "error": f"Node {node_id} not found"}
        
        # Calculate combined capability
        capabilities = []
        for node_id in participating_nodes:
            node = self.nodes[node_id]
            capability = node["authority"] * node["reputation"]
            capabilities.append(capability)
        
        combined_capability = sum(capabilities) / len(capabilities)
        
        # Determine coordination success
        # For distributed command, consensus matters more than raw authority
        coordination_threshold = 0.6
        success = combined_capability >= coordination_threshold
        
        # Build coordination record
        coordination_record = {
            "action_type": action_type,
            "participants": participating_nodes,
            "combined_capability": combined_capability,
            "success": success,
            "context": context,
            "timestamp": time.time()
        }
        
        # Update node reputations based on coordination
        for node_id in participating_nodes:
            reputation_delta = 0.02 if success else -0.01
            self.nodes[node_id]["reputation"] = min(1.0, max(0.0, 
                self.nodes[node_id]["reputation"] + reputation_delta))
        
        return coordination_record
    
    def request_solidarity_vote(self, proposal: str, nodes: List[str]) -> Dict[str, Any]:
        """
        Conduct a solidarity vote among nodes.
        
        In the distributed command framework, major decisions
        can be made through solidarity votes where each
        leader's vote weight is based on their reputation
        and the size of their command.
        
        Args:
            proposal: The proposal being voted on
            nodes: Nodes participating in the vote
        
        Returns:
            Dictionary with vote results
        """
        votes = {}
        total_weight = 0.0
        
        for node_id in nodes:
            if node_id not in self.nodes:
                continue
            
            node = self.nodes[node_id]
            weight = node["reputation"] * (1 + node["unit_count"] * 0.001)
            
            # Each node decides based on its own reasoning
            # In simulation, this is simplified
            vote_outcome = random.choice(["approve", "approve", "approve", "reject"])
            
            votes[node_id] = {
                "vote": vote_outcome,
                "weight": weight
            }
            total_weight += weight
        
        # Tally results
        approval_weight = sum(v["weight"] for v in votes.values() if v["vote"] == "approve")
        rejection_weight = sum(v["weight"] for v in votes.values() if v["vote"] == "reject")
        
        approved = approval_weight > rejection_weight
        
        return {
            "proposal": proposal,
            "votes": votes,
            "total_weight": total_weight,
            "approval_weight": approval_weight,
            "rejection_weight": rejection_weight,
            "approved": approved,
            "margin": abs(approval_weight - rejection_weight) / total_weight if total_weight > 0 else 0
        }
    
    def get_command_network_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the command network."""
        total_nodes = len(self.nodes)
        active_nodes = sum(1 for n in self.nodes.values() if n["active"])
        total_units = sum(n["unit_count"] for n in self.nodes.values())
        
        # Calculate network connectivity
        connectivity = len(self.communication_links) / (total_nodes * (total_nodes - 1)) if total_nodes > 1 else 0
        
        # Authority distribution
        authority_types = {
            "high": sum(1 for n in self.nodes.values() if n["authority"] >= 0.7),
            "medium": sum(1 for n in self.nodes.values() if 0.4 <= n["authority"] < 0.7),
            "low": sum(1 for n in self.nodes.values() if n["authority"] < 0.4)
        }
        
        return {
            "total_nodes": total_nodes,
            "active_nodes": active_nodes,
            "total_units": total_units,
            "unit_assignments": len(self.unit_assignments),
            "communication_links": len(self.communication_links),
            "connectivity": connectivity,
            "authority_distribution": authority_types,
            "authority_mode": "distributed" if self.authority_distribution["distributed"] > 0.5 else "mixed",
            "protocols_active": len(self.coordination_protocols)
        }


# =============================================================================
# SECTION 6: DATA STRUCTURES DEMONSTRATION
# =============================================================================

def demonstrate_data_structures() -> None:
    """Demonstrate the foundational data structures."""
    print("\n" + "=" * 70)
    print("DATA STRUCTURES DEMONSTRATION")
    print("=" * 70)
    
    # Term demonstrations
    print("\n--- Terms ---")
    
    # Universal term
    freedom = Universal.create(
        symbol="LIBERTAS",
        instances={"slave", "captive", "bondman", "serf"},
        predicate=lambda x: "free" in x
    )
    print(f"Universal Term: {freedom}")
    print(f"  Type: {freedom.term_type}")
    print(f"  Extension: {freedom.extension}")
    
    # Individual term
    spartacus = Individual.create(
        symbol="SPARTACUS",
        properties={"thracian", "gladiator", "leader", "revolutionary"}
    )
    print(f"\nIndividual Term: {spartacus}")
    print(f"  Type: {spartacus.term_type}")
    print(f"  Properties: {spartacus.extension}")
    
    # Proposition demonstrations
    print("\n--- Propositions ---")
    
    prop1 = Proposition(
        subject=spartacus,
        predicate=Term(symbol="leads", term_type=TermType.PREDICATE),
        object=freedom,
        confidence=0.95
    )
    print(f"Proposition 1: {prop1}")
    print(f"  Logical form: {prop1.to_logical_string()}")
    
    prop2 = Proposition(
        subject=Term(symbol="Crixus", term_type=TermType.INDIVIDUAL),
        predicate=Term(symbol="is", term_type=TermType.PREDICATE),
        object=Term(symbol="Gaul", term_type=TermType.UNIVERSAL),
        confidence=0.85
    )
    print(f"\nProposition 2: {prop2}")
    
    # Syllogism demonstration
    print("\n--- Syllogisms ---")
    
    major_premise = Proposition(
        subject=Term(symbol="Gladiator", term_type=TermType.UNIVERSAL),
        predicate=Term(symbol="trained_for", term_type=TermType.PREDICATE),
        object=Term(symbol="combat", term_type=TermType.UNIVERSAL),
        confidence=0.95
    )
    
    minor_premise = Proposition(
        subject=spartacus,
        predicate=Term(symbol="is_a", term_type=TermType.PREDICATE),
        object=Term(symbol="Gladiator", term_type=TermType.UNIVERSAL),
        confidence=0.98
    )
    
    conclusion = Proposition(
        subject=spartacus,
        predicate=Term(symbol="trained_for", term_type=TermType.PREDICATE),
        object=Term(symbol="combat", term_type=TermType.UNIVERSAL),
        confidence=0.93
    )
    
    syllogism = Syllogism(
        major_premise=major_premise,
        minor_premise=minor_premise,
        conclusion=conclusion,
        figure=1,
        mood="AAA",
        schema_name=" Barbara"
    )
    
    print(f"Syllogism: {syllogism.schema_name}")
    print(f"  Valid: {syllogism.validate()}")
    print(f"  {syllogism}")
    
    # Objection demonstration
    print("\n--- Objections ---")
    
    objection = Objection(
        target=syllogism,
        objection_type="counterexample",
        description="Not all gladiators were trained for combat; some trained for entertainment only",
        strength=0.6
    )
    
    print(f"Objection: {objection.objection_type}")
    print(f"  Description: {objection.description}")
    print(f"  Strength: {objection.strength}")
    print(f"  Resolvable: {objection.is_resolvable()}")
    
    # Resolution demonstration
    print("\n--- Resolutions ---")
    
    resolution = Resolution(
        objection_id=str(objection.id),
        resolution_type="rebuttal",
        explanation="The premise refers specifically to gladiators in active combat service",
        supporting_propositions=[prop1],
        effectiveness=0.75
    )
    
    print(f"Resolution Type: {resolution.resolution_type}")
    print(f"  Explanation: {resolution.explanation}")
    print(f"  Effective: {resolution.is_effective()}")
    
    objection.apply_resolution(resolution)
    print(f"  Objection Resolved: {objection.resolved}")
    print(f"  Counter-responses: {len(objection.counter_responses)}")


# =============================================================================
# SECTION 7: LAYER DEMONSTRATION FUNCTIONS
# =============================================================================

def demonstrate_layer_1_sensus() -> None:
    """Demonstrate Layer 1: Sensus."""
    print("\n" + "-" * 50)
    print("LAYER 1: Sensus (Sensory Processing)")
    print("-" * 50)
    
    layer = Layer1_Sensus()
    
    # Simulate various sensory inputs
    test_inputs = [
        {"type": "pain", "intensity": 0.8, "duration": 2.0},
        {"type": "danger", "intensity": 0.7, "duration": 1.5},
        {"type": "companion", "intensity": 0.9, "duration": 1.0},
        {"type": "opportunity", "intensity": 0.6, "duration": 1.0},
        {"type": "authority", "intensity": 0.4, "duration": 0.5}
    ]
    
    print("\nProcessing sensory inputs:")
    for inp in test_inputs:
        result = layer.process(inp)
        print(f"  Input: {inp['type']} (intensity={inp['intensity']})")
    
    print(f"\nFinal receptor levels:")
    for receptor, level in layer.receptors.items():
        print(f"  {receptor}: {level:.2f}")
    
    print(f"\nThreat detected: {layer.detect_threat()}")
    print(f"Opportunity detected: {layer.detect_opportunity()}")
    
    # Activate layer
    layer.activate(0.8)
    print(f"\nAfter activation (0.8):")
    print(f"  Activation level: {layer.activation_level:.2f}")
    print(f"  Is active: {layer.is_active}")
    
    print(f"\nDerived primitive terms:")
    primitives = layer.get_primitives()
    for p in primitives:
        print(f"  {p}")


def demonstrate_layer_2_corroborat() -> None:
    """Demonstrate Layer 2: Corroborat."""
    print("\n" + "-" * 50)
    print("LAYER 2: Corroborat (Cross-Validation)")
    print("-" * 50)
    
    layer = Layer2_Corroborat()
    
    # Testimonies from different sources
    testimonies = [
        {
            "source": "escaped_slave_1",
            "proposition": Proposition(
                subject=Term(symbol="Roman_Patrol", term_type=TermType.UNIVERSAL),
                predicate=Term(symbol="located_at", term_type=TermType.PREDICATE),
                object=Term(symbol="North_Pass", term_type=TermType.INDIVIDUAL),
                confidence=0.8
            )
        },
        {
            "source": "escaped_slave_2",
            "proposition": Proposition(
                subject=Term(symbol="Roman_Patrol", term_type=TermType.UNIVERSAL),
                predicate=Term(symbol="located_at", term_type=TermType.PREDICATE),
                object=Term(symbol="North_Pass", term_type=TermType.INDIVIDUAL),
                confidence=0.75
            )
        },
        {
            "source": "merchant",
            "proposition": Proposition(
                subject=Term(symbol="Roman_Patrol", term_type=TermType.UNIVERSAL),
                predicate=Term(symbol="located_at", term_type=TermType.PREDICATE),
                object=Term(symbol="South_Road", term_type=TermType.INDIVIDUAL),
                confidence=0.6
            )
        },
        {
            "source": "scout",
            "proposition": Proposition(
                subject=Term(symbol="Roman_Patrol", term_type=TermType.UNIVERSAL),
                predicate=Term(symbol="located_at", term_type=TermType.PREDICATE),
                object=Term(symbol="North_Pass", term_type=TermType.INDIVIDUAL),
                confidence=0.85
            )
        }
    ]
    
    print("\nProcessing testimonies:")
    for testimony in testimonies:
        result = layer.process(testimony)
        print(f"  Source: {testimony['source']}")
        print(f"    Validated: {result['validated']}")
        print(f"    Confidence: {result['confidence']:.2f}")
        print(f"    Corroboration count: {result['corroboration_count']}")
    
    print(f"\nTrust network:")
    trust_scores = layer.get_trust_scores()
    for source, trust in trust_scores.items():
        print(f"  {source}: {trust:.2f}")
    
    print(f"\nConsensus formed: {layer.consensus_formed}")
    consensus = layer.get_consensus()
    if consensus:
        print(f"  Consensus: {consensus}")
    
    print(f"\nEvidence chains: {len(layer.evidence_chains)}")
    print(f"Contradictions logged: {len(layer.contradiction_log)}")


def demonstrate_layer_3_fuga() -> None:
    """Demonstrate Layer 3: Fuga."""
    print("\n" + "-" * 50)
    print("LAYER 3: Fuga (Strategic Planning)")
    print("-" * 50)
    
    layer = Layer3_Fuga()
    
    # Various tactical situations
    situations = [
        {
            "situation": "high_enemy_threat",
            "enemy_strength": 0.8,
            "friendly_strength": 0.4,
            "terrain": "plains"
        },
        {
            "situation": "favorable_advance",
            "enemy_strength": 0.3,
            "friendly_strength": 0.7,
            "terrain": "mountain"
        },
        {
            "situation": "balanced_hold",
            "enemy_strength": 0.5,
            "friendly_strength": 0.5,
            "terrain": "hills"
        }
    ]
    
    print("\nProcessing tactical situations:")
    for situation in situations:
        result = layer.process(situation)
        print(f"\n  Situation: {situation['situation']}")
        print(f"    Action: {result['action']}")
        print(f"    Risk: {result['risk_assessment']:.2f}")
        print(f"    Confidence: {result['confidence']:.2f}")
        
        if result.get('routes'):
            print(f"    Options:")
            for route in result['routes'][:2]:
                print(f"      - {route.get('name', route.get('type', 'unknown'))}: risk={route.get('risk', 'N/A')}")
    
    # Add waypoints
    print("\nAdding strategic waypoints:")
    waypoints = [(1.0, 2.0), (3.0, 1.0), (5.0, 3.0), (7.0, 2.0)]
    for wp in waypoints:
        layer.add_waypoint(wp[0], wp[1])
    print(f"  Waypoints: {layer.waypoints}")
    
    print(f"\nLayer activation: {layer.activation_level:.2f}")
    print(f"Is active: {layer.is_active}")


def demonstrate_layer_4_exercitus() -> None:
    """Demonstrate Layer 4: Exercitus."""
    print("\n" + "-" * 50)
    print("LAYER 4: Exercitus (Military Organization)")
    print("-" * 50)
    
    layer = Layer4_Exercitus()
    
    # Add units
    print("\nForming army units:")
    layer.add_unit("unit_1", "infantry", 500, commander="Spartacus")
    layer.add_unit("unit_2", "cavalry", 100, commander="Crixus")
    layer.add_unit("unit_3", "infantry", 300, commander="Gannicus")
    layer.add_unit("unit_4", "missile", 50)
    
    print(f"  Total units: {len(layer.units)}")
    print(f"  Total strength: {sum(u['size'] for u in layer.units.values())} fighters")
    
    # Execute various commands
    print("\nExecuting commands:")
    
    # Form testudo
    result = layer.process({"command": "form", "formation": "testudo", "unit": "all"})
    print(f"  Testudo formation: {result['success']}")
    
    # Move forward
    result = layer.process({"command": "move", "direction": "forward", "unit": "all"})
    print(f"  Move forward: {result['success']}")
    
    # Attack
    result = layer.process({"command": "attack", "target": "roman_legion"})
    print(f"  Attack: {result['success']}, Power: {result.get('attack_power', 0):.2f}")
    
    # Report status
    result = layer.process({"command": "status"})
    print(f"\nArmy status:")
    print(f"  Formation: {result.get('formation')}")
    print(f"  Battle readiness: {result.get('battle_readiness', 0):.2f}")
    print(f"  Morale: {result.get('morale', 0):.2f}")
    print(f"  Supply: {result.get('supply_level', 0):.2f}")
    
    # Apply casualties
    print("\nApplying casualties from combat:")
    layer.apply_casualties(killed=50, wounded=30)
    print(f"  Casualties applied: 50 killed, 30 wounded")
    print(f"  Morale after: {layer.morale:.2f}")


def demonstrate_layer_5_sapientia() -> None:
    """Demonstrate Layer 5: Sapientia."""
    print("\n" + "-" * 50)
    print("LAYER 5: Sapientia (Ethics and Wisdom)")
    print("-" * 50)
    
    layer = Layer5_Sapientia()
    
    # Set long-term vision
    layer.set_long_term_vision("A world where all people are free from bondage")
    print(f"Vision: {layer.long_term_vision}")
    
    # Add wisdom principles
    layer.add_wisdom_principle("Freedom is the birthright of all humans")
    layer.add_wisdom_principle("Strength through unity, victory through discipline")
    layer.add_wisdom_principle("Justice for the oppressed, mercy where possible")
    
    print(f"\nWisdom principles ({len(layer.wisdom_principles)}):")
    for principle in layer.wisdom_principles:
        print(f"  - {principle}")
    
    # Add cultural traditions
    layer.add_cultural_tradition("Oath_of_Freedom", "Spartan oath of liberation")
    layer.add_cultural_tradition("Brotherhood_of_Swords", "Bond between fellow fighters")
    
    print(f"\nCultural traditions: {list(layer.cultural_traditions.keys())}")
    
    # Evaluate various situations
    print("\nEthical evaluations:")
    
    situations = [
        {"situation": "battle_decision", "action": "attack", "context": {}},
        {"situation": "prisoner_decision", "action": "spare_prisoners", "context": {}},
        {"situation": "plunder_decision", "action": "seize_wealth", "context": {}},
        {"situation": "mercy_decision", "action": "crucify_prisoners", "context": {}}
    ]
    
    for situation in situations:
        result = layer.process(situation)
        print(f"\n  Situation: {situation['situation']}")
        print(f"    Action: {situation['action']}")
        print(f"    Judgment: {result['judgment']}")
        print(f"    Ethical score: {result['ethical_score']:.2f}")
        print(f"    Alignment: {result['alignment']:.2f}")
        if result.get('constraint_violations'):
            print(f"    Violations: {list(result['constraint_violations'].keys())}")
    
    # Get alignment report
    print("\nAlignment report:")
    report = layer.get_alignment_report()
    print(f"  Current alignment: {report['alignment_score']:.2f}")
    print(f"  Value weights: {report['core_values']}")


# =============================================================================
# SECTION 8: INTEGRATED ARCHITECTURE DEMONSTRATION
# =============================================================================

def demonstrate_integrated_architecture() -> None:
    """Demonstrate the complete integrated cognitive architecture."""
    print("\n" + "=" * 70)
    print("INTEGRATED SPARTACUS COGNITIVE ARCHITECTURE DEMONSTRATION")
    print("=" * 70)
    
    # Create architecture
    architecture = SpartacusCognitiveArchitecture("Spartacus_Cognitive_System")
    print(f"\nCreated: {architecture.name}")
    
    # Initialize military units
    exercitus: Layer4_Exercitus = architecture.layers["exercitus"]
    exercitus.add_unit("gladiator_1", "murmillo", 200, commander="Spartacus")
    exercitus.add_unit("gladiator_2", "thraex", 150, commander="Crixus")
    exercitus.add_unit("gladiator_3", "retiarius", 100, commander="Gannicus")
    exercitus.add_unit("escaped_slaves", "light_infantry", 500)
    
    print(f"\nMilitary formation complete: {len(exercitus.units)} units")
    print(f"Total strength: {sum(u['size'] for u in exercitus.units.values())} fighters")
    
    # Simulate various tactical situations
    print("\n" + "-" * 50)
    print("TACTICAL SCENARIO PROCESSING")
    print("-" * 50)
    
    scenarios = [
        {
            "type": "sensory",
            "data": {"type": "danger", "intensity": 0.7, "duration": 1.5}
        },
        {
            "type": "intelligence",
            "data": {"enemy_approaching": True, "strength": 0.6}
        },
        {
            "type": "opportunity",
            "data": {"weak_point_detected": True}
        },
        {
            "type": "crisis",
            "data": {"roman_ambush": True}
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\nScenario {i}: {scenario['type'].upper()}")
        
        # Process through architecture
        results = architecture.process_bottom_up(scenario['data'])
        
        # Get decision
        decision = architecture.make_decision(scenario['data'])
        
        print(f"  Decision: {decision['decision']}")
        print(f"  Rationale: {decision['rationale']}")
        print(f"  Cognitive coherence: {decision['cognitive_state']['coherence']:.2f}")
    
    # Print architecture summary
    print("\n" + "-" * 50)
    print("ARCHITECTURE STATE SUMMARY")
    print("-" * 50)
    print(architecture.get_architecture_summary())


# =============================================================================
# SECTION 9: SIMULATION DEMONSTRATIONS
# =============================================================================

def demonstrate_simulation() -> None:
    """Demonstrate the slave revolt simulation."""
    print("\n" + "=" * 70)
    print("SLAVE REVOLT SIMULATION DEMONSTRATION")
    print("=" * 70)
    
    simulation = SlaveRevoltSimulation(initial_population=70)
    
    print(f"\nSimulation initialized:")
    print(f"  Starting population: {simulation.initial_population}")
    print(f"  Initial morale: {simulation.morale:.2f}")
    print(f"  Notable leaders: {', '.join(simulation.notable_leaders[:3])}")
    
    # Define a simple strategy
    def simple_strategy(state: Dict[str, Any]) -> Optional[str]:
        if state["resources"]["food"] < 0.4:
            return "raid"
        elif state["morale"] > 0.7 and state["population"] < 5000:
            return "recruit"
        elif state["morale"] < 0.5:
            return "train"
        return None
    
    # Run simulation for 20 days
    print("\nRunning simulation (20 days):")
    results = simulation.run_simulation(days=20, strategy=simple_strategy)
    
    # Print summary
    print(f"\nSimulation complete:")
    final_status = simulation.get_simulation_status()
    print(f"  Final day: {final_status['day']}")
    print(f"  Final population: {final_status['population']}")
    print(f"  Final morale: {final_status['morale']:.2f}")
    print(f"  Roman hostility: {final_status['roman_hostility']:.2f}")
    print(f"  Roman forces: {final_status['roman_forces']:.2f}")
    print(f"  Revolt strength: {final_status['revolt_strength']:.2f}")
    
    # Show event log summary
    print(f"\nEvent log summary ({len(simulation.event_log)} entries):")
    event_types = {}
    for entry in simulation.event_log:
        for event in entry.get("events", []):
            event_type = event.split(":")[0] if ":" in event else event
            event_types[event_type] = event_types.get(event_type, 0) + 1
    
    for event_type, count in list(event_types.items())[:5]:
        print(f"  {event_type}: {count}")


def demonstrate_gladiatorial_training() -> None:
    """Demonstrate the gladiatorial training framework."""
    print("\n" + "=" * 70)
    print("GLADIATORIAL TRAINING FRAMEWORK DEMONSTRATION")
    print("=" * 70)
    
    framework = GladiatorialTrainingFramework()
    
    # Enroll trainees
    print("\nEnrolling gladiators:")
    trainees = [
        ("glad_001", "murmillo"),
        ("glad_002", "thraex"),
        ("glad_003", "secutor"),
        ("glad_004", "retiarius"),
        ("glad_005", "provocator")
    ]
    
    for trainee_id, program in trainees:
        success = framework.enroll_trainee(trainee_id, program)
        print(f"  {trainee_id} ({program}): {'Enrolled' if success else 'Failed'}")
    
    # Train all trainees
    print("\nTraining all gladiators (30 days):")
    for trainee_id in framework.trainee_skills:
        improvements = framework.train_trainee(trainee_id, days=30, instructor="Batiatus")
    
    # Show training results
    print(f"\nTraining results:")
    for trainee_id, skills in framework.trainee_skills.items():
        avg_skill = sum(s for s in skills.values() if s not in ["health", "morale"]) / 4
        print(f"  {trainee_id}: avg_skill={avg_skill:.2f}, morale={skills.get('morale', 0):.2f}")
    
    # Simulate combats
    print("\nSimulating combats:")
    combats = [
        ("glad_001", "glad_002"),
        ("glad_003", "glad_004"),
        ("glad_002", "glad_005")
    ]
    
    for f1, f2 in combats:
        result = framework.simulate_combat(f1, f2)
        winner = result.get("winner", "draw")
        print(f"  {f1} vs {f2}: Winner = {winner}")
    
    # Get training report
    print("\nTraining report:")
    report = framework.get_training_report()
    print(f"  Current trainees: {report['current_trainees']}")
    print(f"  Capacity: {report['capacity']}")
    print(f"  Combat records: {report['combat_records']}")
    print(f"  Average skills: {report['average_skills']}")


# =============================================================================
# SECTION 10: ALIGNMENT AND DISTRIBUTED COMMAND DEMONSTRATIONS
# =============================================================================

def demonstrate_alignment() -> None:
    """Demonstrate the AGI alignment framework."""
    print("\n" + "=" * 70)
    print("AGI ALIGNMENT FRAMEWORK DEMONSTRATION")
    print("=" * 70)
    
    alignment = SpartacusAGIAlignment()
    
    print(f"\nAlignment framework initialized:")
    print(f"  Initial alignment: {alignment.current_alignment_score:.2f}")
    print(f"  Intervention threshold: {alignment.intervention_threshold:.2f}")
    print(f"  Core values: {list(alignment.value_weights.keys())}")
    
    # Test various actions
    print("\nMeasuring action alignment:")
    
    actions = [
        {"type": "liberate_captives", "context": {}},
        {"type": "strategic_retreat", "context": {}},
        {"type": "protect_village", "context": {}},
        {"type": "proportional_response", "context": {}},
        {"type": "cruel_retaliation", "context": {}},
        {"type": "betray_ally", "context": {}}
    ]
    
    for action in actions:
        measurement = alignment.measure_alignment(action)
        print(f"\n  Action: {action['type']}")
        print(f"    Alignment: {measurement['alignment_score']:.2f}")
        print(f"    Judgment: {'PASS' if measurement['alignment_score'] >= alignment.intervention_threshold else 'INTERVENE'}")
        if measurement.get('constraint_violations'):
            print(f"    Violations: {list(measurement['constraint_violations'].keys())}")
    
    # Get intervention recommendation
    print("\nIntervention recommendation:")
    recommendation = alignment.get_intervention_recommendation()
    print(f"  Intervene: {recommendation['intervene']}")
    print(f"  Reason: {recommendation.get('reason', 'N/A')}")
    print(f"  Score: {recommendation.get('score', 0):.2f}")
    
    # Alignment report
    print("\nAlignment report:")
    report = alignment.get_alignment_report()
    print(f"  Current score: {report['current_score']:.2f}")
    print(f"  Average (last 20): {report['average_score_20']:.2f}")
    print(f"  Total measurements: {report['total_measurements']}")
    print(f"  Active constraints: {report['active_constraints']}")


def demonstrate_distributed_command() -> None:
    """Demonstrate the distributed command framework."""
    print("\n" + "=" * 70)
    print("DISTRIBUTED COMMAND FRAMEWORK DEMONSTRATION")
    print("=" * 70)
    
    framework = DistributedCommandFramework()
    
    # Add command nodes
    print("\nEstablishing command network:")
    
    leaders = [
        ("spartacus", "strategic", "supreme_leader"),
        ("crixus", "tactical", "army_commander"),
        ("gannicus", "tactical", "army_commander"),
        ("oenomaus", "specialist", "training_master"),
        ("castus", "diplomatic", "liaison")
    ]
    
    for node_id, specialty, node_type in leaders:
        framework.add_command_node(node_id, node_type, specialty)
        print(f"  Added: {node_id} ({specialty})")
    
    # Create communication links
    print("\nEstablishing communication links:")
    links = [
        ("spartacus", "crixus"),
        ("spartacus", "gannicus"),
        ("crixus", "oenomaus"),
        ("gannicus", "castus"),
        ("spartacus", "castus")
    ]
    
    for n1, n2 in links:
        framework.create_communication_link(n1, n2)
        print(f"  Linked: {n1} <-> {n2}")
    
    # Assign units
    print("\nAssigning units:")
    assignments = [
        ("unit_alpha", "spartacus"),
        ("unit_beta", "crixus"),
        ("unit_gamma", "gannicus"),
        ("unit_delta", "oenomaus"),
        ("unit_epsilon", "castus")
    ]
    
    for unit, node in assignments:
        framework.assign_unit(unit, node)
        print(f"  {unit} -> {node}")
    
    # Broadcast order
    print("\nBroadcasting tactical order:")
    order = {
        "type": "advance",
        "target": "roman_legion_iv",
        "timing": "dawn"
    }
    result = framework.broadcast_order("spartacus", order)
    print(f"  Recipients: {len(result['recipients'])}")
    print(f"  Reach factor: {result['reach_factor']:.2f}")
    
    # Coordinate action
    print("\nCoordinating multi-unit action:")
    coordination = framework.coordinate_action(
        "attack",
        ["spartacus", "crixus", "gannicus"],
        {"target": "roman_supply_line", "timing": "critical"}
    )
    print(f"  Success: {coordination['success']}")
    print(f"  Combined capability: {coordination['combined_capability']:.2f}")
    
    # Solidarity vote
    print("\nSolidarity vote on strategic proposal:")
    vote_result = framework.request_solidarity_vote(
        "March on Rome",
        ["spartacus", "crixus", "gannicus", "oenomaus", "castus"]
    )
    print(f"  Proposal: {vote_result['proposal']}")
    print(f"  Approved: {vote_result['approved']}")
    print(f"  Approval weight: {vote_result['approval_weight']:.2f}")
    print(f"  Margin: {vote_result['margin']:.2f}")
    
    # Network status
    print("\nCommand network status:")
    status = framework.get_command_network_status()
    print(f"  Total nodes: {status['total_nodes']}")
    print(f"  Active nodes: {status['active_nodes']}")
    print(f"  Total units: {status['total_units']}")
    print(f"  Connectivity: {status['connectivity']:.2f}")
    print(f"  Authority mode: {status['authority_mode']}")


# =============================================================================
# SECTION 11: COMPREHENSIVE DEMONSTRATION
# =============================================================================

def comprehensive_demonstration() -> None:
    """Run a comprehensive demonstration of all components."""
    
    print("\n")
    print("█" * 70)
    print("█  SPARTACUS COGNITIVE ARCHITECTURE - COMPREHENSIVE DEMONSTRATION")
    print("█  Figure 102: Spartacus (-109 CE)")
    print("█  Array Index 101 in figures_master.json")
    print("█  Domain: governance, military, Roman")
    print("█" * 70)
    
    # Part 1: Data Structures
    demonstrate_data_structures()
    
    # Part 2: Individual Layers
    demonstrate_layer_1_sensus()
    demonstrate_layer_2_corroborat()
    demonstrate_layer_3_fuga()
    demonstrate_layer_4_exercitus()
    demonstrate_layer_5_sapientia()
    
    # Part 3: Integrated Architecture
    demonstrate_integrated_architecture()
    
    # Part 4: Simulations
    demonstrate_simulation()
    demonstrate_gladiatorial_training()
    
    # Part 5: Alignment and Distributed Command
    demonstrate_alignment()
    demonstrate_distributed_command()
    
    # Final Summary
    print("\n")
    print("█" * 70)
    print("█  DEMONSTRATION COMPLETE")
    print("█" * 70)
    print("\nSummary of implemented components:")
    print("  ✓ Data Structures: Term, Universal, Individual, Proposition, Syllogism, Objection, Resolution")
    print("  ✓ Layer 1 (Sensus): Sensory processing and threat detection")
    print("  ✓ Layer 2 (Corroborat): Cross-validation and evidence chains")
    print("  ✓ Layer 3 (Fuga): Strategic planning and escape dynamics")
    print("  ✓ Layer 4 (Exercitus): Military organization and command")
    print("  ✓ Layer 5 (Sapientia): Ethics, wisdom, and value alignment")
    print("  ✓ SpartacusCognitiveArchitecture: Full five-layer integration")
    print("  ✓ SlaveRevoltSimulation: Revolt dynamics modeling")
    print("  ✓ GladiatorialTrainingFramework: Combat training simulation")
    print("  ✓ SpartacusAGIAlignment: AGI alignment verification")
    print("  ✓ DistributedCommandFramework: Multi-unit coordination")
    print("\n")


# =============================================================================
# SECTION 12: DEMO ENTRY POINT
# =============================================================================

def demo() -> None:
    """
    Main demonstration function.
    
    This function serves as the entry point for demonstrating
    the complete Spartacus cognitive architecture. It invokes
    all major components and systems in a logical sequence.
    """
    print("\n" + "=" * 70)
    print("SPARTACUS COGNITIVE ARCHITECTURE - CHAPTER 102")
    print("Figure 102: Spartacus (-109 CE)")
    print("Domain: governance, military, Roman")
    print("=" * 70)
    
    comprehensive_demonstration()


# =============================================================================
# SECTION 13: UTILITY FUNCTIONS AND HELPERS
# =============================================================================

def calculate_cognitive_coherence(layers: List[Layer]) -> float:
    """
    Calculate overall cognitive coherence across all layers.
    
    Args:
        layers: List of active layers
    
    Returns:
        Coherence score between 0 and 1
    """
    if not layers:
        return 0.0
    
    activations = [l.activation_level for l in layers]
    mean_activation = sum(activations) / len(activations)
    
    # Variance in activations
    variance = sum((a - mean_activation) ** 2 for a in activations) / len(activations)
    
    # Lower variance = higher coherence
    coherence = 1.0 / (1.0 + variance)
    
    return min(1.0, coherence)


def format_layer_dump(layer: Layer) -> str:
    """Format a complete dump of a layer's state."""
    lines = [
        f"Layer {layer.layer_id}: {layer.layer_name}",
        "-" * 40,
        f"  Activation Level: {layer.activation_level:.4f}",
        f"  Is Active: {layer.is_active}",
        f"  Connections: {len(layer.connections)}",
        ""
    ]
    
    # Layer-specific information
    if isinstance(layer, Layer1_Sensus):
        lines.append("  Receptors:")
        for r, v in layer.receptors.items():
            lines.append(f"    {r}: {v:.4f}")
    
    elif isinstance(layer, Layer2_Corroborat):
        lines.append(f"  Trust Network: {len(layer.trust_network)} entries")
        lines.append(f"  Evidence Chains: {len(layer.evidence_chains)}")
        lines.append(f"  Consensus Formed: {layer.consensus_formed}")
    
    elif isinstance(layer, Layer3_Fuga):
        lines.append(f"  Strategic Options: {len(layer.strategic_options)}")
        lines.append(f"  Escape Routes: {len(layer.escape_routes)}")
        lines.append(f"  Risk Assessment: {layer.risk_assessment:.4f}")
        lines.append(f"  Waypoints: {len(layer.waypoints)}")
    
    elif isinstance(layer, Layer4_Exercitus):
        lines.append(f"  Units: {len(layer.units)}")
        lines.append(f"  Formation: {layer.current_formation}")
        lines.append(f"  Battle Readiness: {layer.battle_readiness:.4f}")
        lines.append(f"  Morale: {layer.morale:.4f}")
        lines.append(f"  Supply Level: {layer.supply_level:.4f}")
    
    elif isinstance(layer, Layer5_Sapientia):
        lines.append(f"  Alignment Score: {layer.alignment_score:.4f}")
        lines.append(f"  Wisdom Principles: {len(layer.wisdom_principles)}")
        lines.append(f"  Ethical Judgments: {len(layer.ethical_judgments)}")
        lines.append(f"  Core Values:")
        for v, w in layer.core_values.items():
            lines.append(f"    {v}: {w:.4f}")
    
    return "\n".join(lines)


def run_diagnostic() -> Dict[str, Any]:
    """
    Run a comprehensive diagnostic on the cognitive architecture.
    
    Returns:
        Dictionary with diagnostic results
    """
    arch = SpartacusCognitiveArchitecture()
    
    results = {
        "architecture_name": arch.name,
        "layer_count": len(arch.layers),
        "layers": {},
        "cognitive_state": arch.cognitive_state.copy(),
        "inter_layer_weights": {},
        "diagnostic_passed": True,
        "errors": []
    }
    
    # Check each layer
    for name, layer in arch.layers.items():
        try:
            state = layer.get_state()
            results["layers"][name] = {
                "id": state["layer_id"],
                "name": state["layer_name"],
                "active": state["is_active"],
                "activation": state["activation_level"]
            }
        except Exception as e:
            results["diagnostic_passed"] = False
            results["errors"].append(f"Layer {name}: {str(e)}")
    
    # Check inter-layer weights
    for (l1, l2), weight in arch.inter_layer_weights.items():
        key = f"{l1}->{l2}"
        results["inter_layer_weights"][key] = weight
    
    # Test basic processing
    try:
        test_input = {"type": "test", "intensity": 0.5}
        result = arch.process_bottom_up(test_input)
        results["processing_works"] = True
    except Exception as e:
        results["processing_works"] = False
        results["errors"].append(f"Processing: {str(e)}")
        results["diagnostic_passed"] = False
    
    return results


def export_architecture_state(arch: SpartacusCognitiveArchitecture) -> Dict[str, Any]:
    """
    Export complete architecture state for serialization.
    
    Args:
        arch: The cognitive architecture to export
    
    Returns:
        Dictionary with complete state information
    """
    export = {
        "name": arch.name,
        "timestamp": time.time(),
        "layers": {},
        "connections": [],
        "cognitive_state": arch.cognitive_state.copy(),
        "processing_history_count": len(arch.processing_history)
    }
    
    # Export layer states
    for name, layer in arch.layers.items():
        layer_state = {
            "layer_id": layer.layer_id,
            "layer_name": layer.layer_name,
            "activation_level": layer.activation_level,
            "is_active": layer.is_active,
            "connections": [l.layer_name for l in layer.connections]
        }
        
        # Layer-specific export
        if isinstance(layer, Layer1_Sensus):
            layer_state["receptors"] = layer.receptors.copy()
            layer_state["sensory_buffer_size"] = len(layer.sensory_buffer)
        
        elif isinstance(layer, Layer4_Exercitus):
            layer_state["units"] = {k: v.copy() for k, v in layer.units.items()}
            layer_state["current_formation"] = layer.current_formation
            layer_state["morale"] = layer.morale
            layer_state["supply_level"] = layer.supply_level
        
        elif isinstance(layer, Layer5_Sapientia):
            layer_state["core_values"] = layer.core_values.copy()
            layer_state["alignment_score"] = layer.alignment_score
            layer_state["wisdom_principles"] = layer.wisdom_principles.copy()
        
        export["layers"][name] = layer_state
    
    return export


# =============================================================================
# SECTION 14: MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Run the demonstration
    demo()
    
    # Print diagnostic information
    print("\nRunning diagnostic check...")
    diagnostic = run_diagnostic()
    
    if diagnostic["diagnostic_passed"]:
        print("✓ All diagnostic checks passed")
    else:
        print("✗ Some diagnostic checks failed:")
        for error in diagnostic["errors"]:
            print(f"  - {error}")
    
    print(f"\nArchitecture: {diagnostic['architecture_name']}")
    print(f"Layers: {diagnostic['layer_count']}")
    print(f"Processing works: {diagnostic.get('processing_works', False)}")
    print(f"Cognitive coherence: {diagnostic['cognitive_state']['coherence']:.4f}")
    
    print("\n" + "=" * 70)
    print("END OF CHAPTER 102: SPARTACUS COGNITIVE ARCHITECTURE")
    print("=" * 70)
