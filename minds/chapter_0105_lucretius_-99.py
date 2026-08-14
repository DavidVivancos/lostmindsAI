"""
Figure 105: Lucretius (-99 CE)
Domain: philosophy, Epicureanism, Roman
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 6 Minds 101 - 120 Available on Amazon https://www.amazon.com/dp/B0HF7G6JJD
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 105: Lucretius (-99 CE)
================================================================================


Lucretius's De Rerum Natura translated into a cognitive architecture.
Five-layer system: Sensus → Atomus → Voluptas → Natura → Sapientia


Created: 2026-04-19
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Set, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from abc import ABC, abstractmethod
import math
import random
import copy


# =============================================================================
# SECTION 1: FOUNDATIONAL DATA STRUCTURES
# =============================================================================

class Term:
    """
    A Term is the fundamental unit of logical expression in Lucretian logic.
    Terms can represent atoms, qualities, relations, or abstract concepts.
    """
    
    def __init__(self, name: str, term_type: str = "concept", 
                 atomic_weight: float = 1.0, properties: Optional[Dict[str, Any]] = None):
        self.name = name
        self.term_type = term_type
        self.atomic_weight = atomic_weight
        self.properties = properties or {}
        self.id = id(self)
    
    def __repr__(self) -> str:
        return f"Term(name='{self.name}', type={self.term_type}, weight={self.atomic_weight})"
    
    def __str__(self) -> str:
        return self.name
    
    def __hash__(self) -> int:
        return hash(self.name)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Term):
            return False
        return self.name == other.name
    
    def combine(self, other: Term, bond_type: str = "default") -> CompoundTerm:
        """Combine two terms into a compound term."""
        return CompoundTerm([self, other], bond_type)
    
    def copy(self) -> Term:
        return Term(self.name, self.term_type, self.atomic_weight, 
                   copy.deepcopy(self.properties))
    
    def get_property(self, key: str, default: Any = None) -> Any:
        return self.properties.get(key, default)
    
    def set_property(self, key: str, value: Any) -> None:
        self.properties[key] = value


class CompoundTerm(Term):
    """A term composed of multiple sub-terms bonded together."""
    
    def __init__(self, components: List[Term], bond_type: str = "default"):
        # Use the bond type as the name
        super().__init__(name=f"compound_{bond_type}", term_type="compound")
        self.components = components
        self.bond_type = bond_type
        # Calculate composite atomic weight
        self.atomic_weight = sum(c.atomic_weight for c in components)
    
    def __repr__(self) -> str:
        return f"CompoundTerm({' + '.join(c.name for c in self.components)}, bond={self.bond_type})"
    
    def get_components(self) -> List[Term]:
        return self.components.copy()
    
    def decompose(self) -> List[Term]:
        """Recursively decompose into individual atoms."""
        result = []
        for component in self.components:
            if isinstance(component, CompoundTerm):
                result.extend(component.decompose())
            else:
                result.append(component)
        return result


class Universal:
    """
    A Universal represents a general category or class of entities.
    In Lucretian philosophy, universals represent the patterns that atoms
    form when they combine in regular ways.
    """
    
    def __init__(self, name: str, instances: Optional[List[Term]] = None,
                 attributes: Optional[Dict[str, Any]] = None,
                 essence: Optional[str] = None):
        self.name = name
        self.instances = instances or []
        self.attributes = attributes or {}
        self.essence = essence  # The fundamental nature of this universal
        self.id = id(self)
    
    def __repr__(self) -> str:
        return f"Universal(name='{self.name}', instances={len(self.instances)})"
    
    def __str__(self) -> str:
        return f"Universal: {self.name}"
    
    def add_instance(self, term: Term) -> None:
        if term not in self.instances:
            self.instances.append(term)
    
    def remove_instance(self, term: Term) -> bool:
        if term in self.instances:
            self.instances.remove(term)
            return True
        return False
    
    def get_instances(self) -> List[Term]:
        return self.instances.copy()
    
    def has_instance(self, term: Term) -> bool:
        return term in self.instances
    
    def instantiate(self, name: str, **properties) -> Term:
        """Create a new instance of this universal."""
        term = Term(name, term_type=self.name, properties=properties)
        self.add_instance(term)
        return term


class Individual:
    """
    An Individual represents a specific, concrete entity.
    In Epicurean physics, all individuals are composed of atoms
    and will eventually dissolve back into their atomic components.
    """
    
    def __init__(self, name: str, atoms: Optional[List[Term]] = None,
                 properties: Optional[Dict[str, Any]] = None,
                 position: Optional[Tuple[float, ...]] = None,
                 velocity: Optional[Tuple[float, ...]] = None):
        self.name = name
        self.atoms = atoms or []
        self.properties = properties or {}
        self.position = position or (0.0,) * 3  # 3D space by default
        self.velocity = velocity or (0.0,) * 3
        self.id = id(self)
        self.history: List[Dict[str, Any]] = []
    
    def __repr__(self) -> str:
        return f"Individual(name='{self.name}', atoms={len(self.atoms)})"
    
    def __str__(self) -> str:
        return f"Individual: {self.name}"
    
    def add_atom(self, atom: Term) -> None:
        self.atoms.append(atom)
    
    def remove_atom(self, atom: Term) -> bool:
        if atom in self.atoms:
            self.atoms.remove(atom)
            return True
        return False
    
    def get_atoms(self) -> List[Term]:
        return self.atoms.copy()
    
    def get_mass(self) -> float:
        return sum(a.atomic_weight for a in self.atoms)
    
    def move_to(self, position: Tuple[float, ...]) -> None:
        self.position = position
        self.record_history("move", position=position)
    
    def record_history(self, event_type: str, **data) -> None:
        self.history.append({"type": event_type, "data": data})
    
    def copy(self) -> Individual:
        return Individual(
            self.name + "_copy",
            atoms=[a.copy() for a in self.atoms],
            properties=copy.deepcopy(self.properties),
            position=self.position,
            velocity=self.velocity
        )


# =============================================================================
# SECTION 2: PROPOSITION AND SYLLOGISM SYSTEM
# =============================================================================

class Proposition:
    """
    A Proposition is a declarative statement that can be true or false.
    In Lucretian logic, propositions deal with physical phenomena,
    sensations, and the nature of reality.
    """
    
    class TruthValue(Enum):
        TRUE = auto()
        FALSE = auto()
        UNKNOWN = auto()
        CONTINGENT = auto()  # Depends on circumstances
    
    def __init__(self, subject: Union[Term, Universal, Individual],
                 predicate: str,
                 truth_value: TruthValue = TruthValue.UNKNOWN,
                 evidence: Optional[List[str]] = None,
                 counterevidence: Optional[List[str]] = None):
        self.subject = subject
        self.predicate = predicate
        self.truth_value = truth_value
        self.evidence = evidence or []
        self.counterevidence = counterevidence or []
        self.id = id(self)
    
    def __repr__(self) -> str:
        return f"Proposition({self.subject} {self.predicate})"
    
    def __str__(self) -> str:
        return f"{self.subject} {self.predicate}"
    
    def assert_true(self, evidence: str) -> None:
        self.evidence.append(evidence)
        if self.truth_value == Proposition.TruthValue.UNKNOWN:
            self.truth_value = Proposition.TruthValue.TRUE
    
    def assert_false(self, counterevidence: str) -> None:
        self.counterevidence.append(counterevidence)
        if self.truth_value == Proposition.TruthValue.UNKNOWN:
            self.truth_value = Proposition.TruthValue.FALSE
    
    def is_true(self) -> bool:
        return self.truth_value == Proposition.TruthValue.TRUE
    
    def is_false(self) -> bool:
        return self.truth_value == Proposition.TruthValue.FALSE
    
    def evaluate(self, context: Optional[Dict[str, Any]] = None) -> TruthValue:
        """Evaluate the proposition in a given context."""
        return self.truth_value


class Syllogism:
    """
    A Syllogism is a logical argument consisting of a major premise,
    a minor premise, and a conclusion. In De Rerum Natura, Lucretius
    uses syllogistic reasoning to demonstrate physical truths.
    """
    
    def __init__(self, major_premise: Proposition,
                 minor_premise: Proposition,
                 conclusion: Proposition,
                 name: str = "anonymous"):
        self.major_premise = major_premise
        self.minor_premise = minor_premise
        self.conclusion = conclusion
        self.name = name
        self.valid = True
        self.id = id(self)
    
    def __repr__(self) -> str:
        return f"Syllogism({self.name}: {self.major_premise} → {self.conclusion})"
    
    def __str__(self) -> str:
        return (
            f"Major: {self.major_premise}\n"
            f"Minor: {self.minor_premise}\n"
            f"Conclusion: {self.conclusion}"
        )
    
    def is_valid(self) -> bool:
        """Check if the syllogism is logically valid."""
        # In a valid syllogism, if both premises are true, the conclusion must be true
        if self.major_premise.is_true() and self.minor_premise.is_true():
            return self.conclusion.is_true()
        return False
    
    def evaluate(self) -> bool:
        """Evaluate the complete syllogism."""
        self.valid = self.is_valid()
        return self.valid
    
    def get_structure(self) -> Dict[str, Proposition]:
        return {
            "major_premise": self.major_premise,
            "minor_premise": self.minor_premise,
            "conclusion": self.conclusion
        }


class Objection:
    """
    An Objection represents a counterargument or challenge to a syllogism
    or proposition. Lucretius anticipates and addresses objections to his
    atomic theory and naturalistic philosophy.
    """
    
    def __init__(self, target: Union[Proposition, Syllogism],
                 text: str,
                 source: str = "anonymous",
                 strength: float = 0.5):
        self.target = target
        self.text = text
        self.source = source
        self.strength = strength  # 0.0 to 1.0
        self.responses: List[str] = []
        self.id = id(self)
    
    def __repr__(self) -> str:
        return f"Objection(from={self.source}, strength={self.strength})"
    
    def __str__(self) -> str:
        return f"Objection: {self.text}"
    
    def respond(self, response: str) -> None:
        self.responses.append(response)
    
    def is_strong(self) -> bool:
        return self.strength >= 0.7
    
    def get_strength(self) -> float:
        return self.strength
    
    def set_strength(self, strength: float) -> None:
        self.strength = max(0.0, min(1.0, strength))


class Resolution:
    """
    A Resolution addresses an objection and provides a reasoned response.
    In Lucretian philosophy, resolutions demonstrate how objections can
    be overcome through careful analysis of nature.
    """
    
    def __init__(self, objection: Objection,
                 text: str,
                 reasoning: str,
                 success: float = 1.0):
        self.objection = objection
        self.text = text
        self.reasoning = reasoning
        self.success = success  # How well it resolves the objection
        self.id = id(self)
    
    def __repr__(self) -> str:
        return f"Resolution(success={self.success})"
    
    def __str__(self) -> str:
        return f"Resolution: {self.text}"
    
    def is_successful(self) -> bool:
        return self.success >= 0.5
    
    def get_effectiveness(self) -> float:
        return self.success


# =============================================================================
# SECTION 3: FIVE-LAYER COGNITIVE ARCHITECTURE
# =============================================================================

class Layer1_Sensus:
    """
    Layer 1: Sensus (Sensation)
    
    The foundation of all knowledge according to Lucretius.
    Sensation provides the raw data that the mind processes.
    Without sensation, there can be no thought or knowledge.
    
    Key principles:
    - All knowledge comes from sensation
    - Sensations are caused by atomic images (simulacra)
    - The senses cannot deceive us
    - Reason based on sensation is reliable
    """
    
    def __init__(self):
        self.sensory_data: Dict[str, List[Any]] = {}
        self.perceptions: List[Dict[str, Any]] = []
        self.sense_threshold = 0.1
        self.atomic_images: List[Term] = []
        self.id = id(self)
    
    def __repr__(self) -> str:
        return f"Layer1_Sensus(sensory_channels={len(self.sensory_data)})"
    
    def receive_sensation(self, channel: str, data: Any) -> None:
        """Receive raw sensory data."""
        if channel not in self.sensory_data:
            self.sensory_data[channel] = []
        self.sensory_data[channel].append(data)
    
    def get_sensations(self, channel: str) -> List[Any]:
        """Get all sensations for a channel."""
        return self.sensory_data.get(channel, [])
    
    def form_perception(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Form a perception from sensory data."""
        perception = {
            "data": data,
            "channel": data.get("channel", "unknown"),
            "intensity": data.get("intensity", 0.0),
            "timestamp": data.get("timestamp", 0),
            "verified": False
        }
        self.perceptions.append(perception)
        return perception
    
    def verify_perception(self, perception: Dict[str, Any]) -> bool:
        """Verify a perception through consistency with other sensations."""
        channel = perception["channel"]
        related_channels = self._get_related_channels(channel)
        
        for related in related_channels:
            if related in self.sensory_data and self.sensory_data[related]:
                perception["verified"] = True
                return True
        return False
    
    def _get_related_channels(self, channel: str) -> List[str]:
        """Get channels that relate to a given sensory channel."""
        relationships = {
            "visual": ["spatial", "temporal"],
            "auditory": ["spatial", "temporal"],
            "tactile": ["spatial", "thermal"],
            "olfactory": ["temporal", "spatial"],
            "gustatory": ["temporal"]
        }
        return relationships.get(channel, [])
    
    def process_atomic_images(self, images: List[Term]) -> List[Dict[str, Any]]:
        """Process atomic images (simulacra) from external objects."""
        results = []
        for image in images:
            result = {
                "image": image,
                "velocity": image.get_property("velocity", 1.0),
                "fineness": image.get_property("fineness", 0.5),
                "penetration": self._calculate_penetration(image)
            }
            results.append(result)
            self.atomic_images.append(image)
        return results
    
    def _calculate_penetration(self, image: Term) -> float:
        """Calculate how deeply an atomic image can penetrate."""
        velocity = image.get_property("velocity", 1.0)
        fineness = image.get_property("fineness", 0.5)
        weight = image.atomic_weight
        # Penetration increases with velocity and fineness, decreases with weight
        penetration = (velocity * fineness) / (weight + 0.1)
        return min(1.0, penetration)
    
    def detect_swindon(self, threshold: float = None) -> Optional[Term]:
        """
        Detect the swerve (clinamen) - the random deviation of atoms.
        This is a key feature of Epicurean physics.
        """
        if threshold is None:
            threshold = self.sense_threshold
        
        for image in self.atomic_images:
            swerve = image.get_property("swerve", 0.0)
            if swerve >= threshold:
                return image
        return None
    
    def clear(self) -> None:
        """Clear all sensory data."""
        self.sensory_data.clear()
        self.perceptions.clear()
        self.atomic_images.clear()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of sensory state."""
        return {
            "channels": list(self.sensory_data.keys()),
            "total_sensations": sum(len(v) for v in self.sensory_data.values()),
            "perceptions": len(self.perceptions),
            "atomic_images": len(self.atomic_images)
        }


class Layer2_Atomus:
    """
    Layer 2: Atomus (Atoms)
    
    The physical basis of reality according to Epicurean philosophy.
    All things are composed of atoms - indivisible, eternal particles
    moving through the void.
    
    Key principles:
    - Atoms are solid and indivisible
    - Atoms have various shapes and sizes
    - Atoms move continuously in the void
    - Atoms can combine and separate
    - The swerve (clinamen) causes variety in motion
    """
    
    def __init__(self):
        self.atoms: List[Term] = []
        self.primordial_void = 0.0  # The empty space atoms move through
        self.swerves: List[Dict[str, Any]] = []
        self.atom_registry: Dict[str, Term] = {}
        self.id = id(self)
    
    def __repr__(self) -> str:
        return f"Layer2_Atomus(atoms={len(self.atoms)})"
    
    def create_atom(self, name: str, weight: float, 
                    shape: str = "spherical",
                    properties: Optional[Dict[str, Any]] = None) -> Term:
        """Create a primordial atom."""
        props = properties or {}
        props["shape"] = shape
        atom = Term(name, term_type="atom", atomic_weight=weight, properties=props)
        self.atoms.append(atom)
        self.atom_registry[name] = atom
        return atom
    
    def create_from_elements(self, elements: List[Tuple[str, float]]) -> List[Term]:
        """Create atoms from element specifications."""
        atoms = []
        for name, weight in elements:
            atom = self.create_atom(name, weight, properties={"origin": "elements"})
            atoms.append(atom)
        return atoms
    
    def combine_atoms(self, atoms: List[Term], bond_strength: float = 1.0) -> CompoundTerm:
        """Combine multiple atoms into a compound."""
        compound = CompoundTerm(atoms, bond_type="atomic")
        compound.set_property("bond_strength", bond_strength)
        compound.set_property("creation", "combination")
        return compound
    
    def separate_atoms(self, compound: CompoundTerm) -> List[Term]:
        """Separate compound into constituent atoms."""
        return compound.decompose()
    
    def apply_swerve(self, atom: Term, angle: float) -> Dict[str, Any]:
        """
        Apply the swerve (clinamen) - the random deviation from
        deterministic motion that Epicurus introduced to preserve free will.
        """
        swerve_result = {
            "atom": atom,
            "angle": angle,
            "deviation": math.sin(angle) * atom.atomic_weight,
            "timestamp": len(self.swerves)
        }
        self.swerves.append(swerve_result)
        # Update the atom's properties
        atom.set_property("swerve", angle)
        atom.set_property("swerve_magnitude", swerve_result["deviation"])
        return swerve_result
    
    def get_swerve(self, index: int) -> Optional[Dict[str, Any]]:
        """Get a specific swerve event."""
        return self.swerves[index] if index < len(self.swerves) else None
    
    def get_all_swerves(self) -> List[Dict[str, Any]]:
        """Get all swerve events."""
        return self.swerves.copy()
    
    def calculate_motion(self, atom: Term, time_delta: float) -> Tuple[float, ...]:
        """Calculate the motion of an atom over time."""
        base_velocity = atom.get_property("velocity", 1.0)
        swerve = atom.get_property("swerve", 0.0)
        # Motion consists of straight line plus swerve deviation
        primary_motion = base_velocity * time_delta
        swerve_deviation = swerve * atom.atomic_weight * time_delta * 0.1
        return (primary_motion, swerve_deviation, 0.0)
    
    def count_atoms(self) -> int:
        return len(self.atoms)
    
    def get_atom(self, name: str) -> Optional[Term]:
        return self.atom_registry.get(name)
    
    def get_all_atoms(self) -> List[Term]:
        return self.atoms.copy()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of atomic state."""
        return {
            "total_atoms": len(self.atoms),
            "swerves": len(self.swerves),
            "shapes": list(set(a.get_property("shape", "unknown") for a in self.atoms))
        }


class Layer3_Voluptas:
    """
    Layer 3: Voluptas (Pleasure)
    
    The highest good in Epicurean philosophy. Pleasure is the
    absence of pain and disturbance. The wise person pursues
    pleasure that leads to tranquility (ataraxia).
    
    Key principles:
    - Pleasure is the highest good
    - All living beings seek to avoid pain
    - True pleasure is freedom from disturbance
    - The limit of pleasure is the removal of suffering
    - Complex pleasures should be evaluated by reason
    """
    
    def __init__(self):
        self.pleasures: List[Dict[str, Any]] = []
        self.pains: List[Dict[str, Any]] = []
        self.ataraxia_level = 0.0  # Tranquility
        self.aponia_level = 0.0     # Absence of pain
        self.id = id(self)
    
    def __repr__(self) -> str:
        return f"Layer3_Voluptas(pleasures={len(self.pleasures)}, ataraxia={self.ataraxia_level:.2f})"
    
    def add_pleasure(self, name: str, intensity: float, 
                     duration: float, source: str = "unknown") -> Dict[str, Any]:
        """Add a pleasurable experience."""
        pleasure = {
            "name": name,
            "intensity": intensity,
            "duration": duration,
            "source": source,
            "timestamp": len(self.pleasures)
        }
        self.pleasures.append(pleasure)
        self._recalculate_state()
        return pleasure
    
    def add_pain(self, name: str, intensity: float,
                 duration: float, source: str = "unknown") -> Dict[str, Any]:
        """Add a painful experience."""
        pain = {
            "name": name,
            "intensity": intensity,
            "duration": duration,
            "source": source,
            "timestamp": len(self.pains)
        }
        self.pains.append(pain)
        self._recalculate_state()
        return pain
    
    def _recalculate_state(self) -> None:
        """Recalculate ataraxia and aponia levels."""
        total_pleasure = sum(p["intensity"] * p["duration"] for p in self.pleasures)
        total_pain = sum(p["intensity"] * p["duration"] for p in self.pains)
        
        # Ataraxia is the balance of pleasure over pain
        net_balance = total_pleasure - total_pain
        self.ataraxia_level = self._sigmoid(net_balance)
        
        # Aponia is freedom from bodily pain
        self.aponia_level = max(0.0, 1.0 - (total_pain / max(1.0, total_pleasure + total_pain)))
    
    def _sigmoid(self, x: float) -> float:
        """Sigmoid function for bounded value."""
        return 1.0 / (1.0 + math.exp(-x / 10.0))
    
    def get_net_pleasure(self) -> float:
        """Calculate net pleasure (pleasure minus pain)."""
        total_pleasure = sum(p["intensity"] * p["duration"] for p in self.pleasures)
        total_pain = sum(p["intensity"] * p["duration"] for p in self.pains)
        return total_pleasure - total_pain
    
    def evaluate_pleasure(self, pleasure: Dict[str, Any]) -> float:
        """Evaluate a potential pleasure for wisdom."""
        intensity = pleasure.get("intensity", 0.5)
        duration = pleasure.get("duration", 1.0)
        consequences = pleasure.get("consequences", [])
        
        # Simple pleasure calculation
        base_value = intensity * duration
        
        # Reduce for negative consequences
        penalty = sum(c.get("harm", 0.0) for c in consequences)
        
        return max(0.0, base_value - penalty)
    
    def is_healthy_pleasure(self, name: str) -> bool:
        """Check if a named pleasure is considered healthy."""
        healthy_pleasures = [
            "friendship", "knowledge", "health", "tranquility",
            "natural_desire", "freedom", "justice", "virtue"
        ]
        return name in healthy_pleasures
    
    def get_ataraxia(self) -> float:
        return self.ataraxia_level
    
    def get_aponia(self) -> float:
        return self.aponia_level
    
    def get_katastematic_pleasure(self) -> float:
        """
        Get the highest form of pleasure - the静态 pleasure
        of living virtuously in tranquility.
        """
        return min(self.ataraxia_level, self.aponia_level)
    
    def get_kinetic_pleasure(self) -> float:
        """Get kinetic pleasure - the active pleasure of satisfaction."""
        return sum(p["intensity"] for p in self.pleasures) / max(1, len(self.pleasures))
    
    def clear(self) -> None:
        self.pleasures.clear()
        self.pains.clear()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of voluptas state."""
        return {
            "pleasures": len(self.pleasures),
            "pains": len(self.pains),
            "ataraxia": self.ataraxia_level,
            "aponia": self.aponia_level,
            "net_pleasure": self.get_net_pleasure()
        }


class Layer4_Natura:
    """
    Layer 4: Natura (Nature)
    
    The working of the natural world according to Epicurean physics.
    Nature operates without divine intervention through the
    interactions of atoms in the void.
    
    Key principles:
    - Nature has no ultimate purpose
    - Everything arises from natural causes
    - There are no supernatural interventions
    - The gods exist but are indifferent to human affairs
    - Death is the dissolution of atoms, nothing to fear
    """
    
    def __init__(self):
        self.individuals: List[Individual] = []
        self.natural_laws: List[Dict[str, Any]] = []
        self.processes: List[Dict[str, Any]] = []
        self.time = 0.0
        self.id = id(self)
    
    def __repr__(self) -> str:
        return f"Layer4_Natura(individuals={len(self.individuals)}, time={self.time:.1f})"
    
    def create_individual(self, name: str, atoms: List[Term],
                          properties: Optional[Dict[str, Any]] = None) -> Individual:
        """Create an individual composed of atoms."""
        individual = Individual(name, atoms, properties)
        self.individuals.append(individual)
        return individual
    
    def add_natural_law(self, name: str, description: str,
                        formula: Optional[str] = None,
                        scope: str = "universal") -> Dict[str, Any]:
        """Add a law of nature."""
        law = {
            "name": name,
            "description": description,
            "formula": formula,
            "scope": scope,
            "active": True
        }
        self.natural_laws.append(law)
        return law
    
    def apply_laws(self, individual: Individual) -> Dict[str, Any]:
        """Apply natural laws to an individual."""
        results = []
        for law in self.natural_laws:
            if law["active"]:
                result = self._apply_single_law(law, individual)
                results.append(result)
        return {"individual": individual.name, "law_effects": results}
    
    def _apply_single_law(self, law: Dict[str, Any], individual: Individual) -> Dict[str, Any]:
        """Apply a single natural law."""
        law_name = law["name"]
        
        if law_name == "motion":
            # Objects persist in motion unless acted upon
            position = list(individual.position)
            velocity = list(individual.velocity)
            position = [p + v for p, v in zip(position, velocity)]
            individual.move_to(tuple(position))
            return {"law": law_name, "effect": "position_updated"}
        
        elif law_name == "gravity":
            # Heavy objects tend downward (in classical view)
            velocity = list(individual.velocity)
            velocity[1] -= 0.1  # Downward acceleration
            individual.velocity = tuple(velocity)
            return {"law": law_name, "effect": "velocity_modified"}
        
        elif law_name == "combination":
            # Atoms naturally combine when compatible
            return {"law": law_name, "effect": "no_change"}
        
        return {"law": law_name, "effect": "unknown"}
    
    def simulate_process(self, process_type: str, 
                        individuals: List[Individual],
                        steps: int = 10) -> List[Dict[str, Any]]:
        """Simulate a natural process."""
        process_log = []
        
        for step in range(steps):
            step_log = {"step": step, "events": []}
            
            for individual in individuals:
                # Update position
                position = list(individual.position)
                velocity = list(individual.velocity)
                new_position = [p + v * 0.1 for p, v in zip(position, velocity)]
                individual.move_to(tuple(new_position))
                
                step_log["events"].append({
                    "individual": individual.name,
                    "position": new_position
                })
            
            self.time += 1.0
            process_log.append(step_log)
        
        return process_log
    
    def generate_spontaneous(self, num_atoms: int = 5) -> List[Individual]:
        """Generate spontaneous configurations of atoms."""
        generated = []
        for i in range(num_atoms):
            name = f"spontaneous_{i}"
            atoms = [Term(f"atom_{j}", "atom", random.uniform(0.5, 2.0)) 
                     for j in range(random.randint(3, 10))]
            individual = self.create_individual(
                name, atoms,
                properties={"origin": "spontaneous", "creation_time": self.time}
            )
            generated.append(individual)
        return generated
    
    def cause_death(self, individual: Individual) -> Dict[str, Any]:
        """
        Cause the death (dissolution) of an individual.
        Death is merely the separation of atoms - nothing to fear.
        """
        original_atoms = individual.get_atoms()
        death_result = {
            "individual": individual.name,
            "atoms_released": len(original_atoms),
            "transition": "dissolution",
            "reassurance": "Death is nothing to the living; for the dead, there is no suffering"
        }
        # Remove from individuals
        if individual in self.individuals:
            self.individuals.remove(individual)
        return death_result
    
    def get_individual(self, name: str) -> Optional[Individual]:
        """Get an individual by name."""
        for ind in self.individuals:
            if ind.name == name:
                return ind
        return None
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of natura state."""
        return {
            "individuals": len(self.individuals),
            "natural_laws": len(self.natural_laws),
            "processes": len(self.processes),
            "time": self.time
        }


class Layer5_Sapientia:
    """
    Layer 5: Sapientia (Wisdom)
    
    The culmination of the Epicurean philosophical journey.
    Wisdom is the knowledge that enables one to live well and
    achieve happiness through understanding nature.
    
    Key principles:
    - Wisdom is the highest virtue
    - True happiness comes from within
    - One must understand nature to overcome fears
    - The fears of gods and death are the main obstacles to happiness
    - Practical wisdom leads to the good life
    """
    
    def __init__(self):
        self.understandings: List[Dict[str, Any]] = []
        self.virtues: Dict[str, float] = {}
        self.insights: List[str] = []
        self.wisdom_level = 0.0
        self.id = id(self)
    
    def __repr__(self) -> str:
        return f"Layer5_Sapientia(wisdom={self.wisdom_level:.2f})"
    
    def add_understanding(self, topic: str, understanding: str,
                         depth: float = 0.5) -> Dict[str, Any]:
        """Add a new understanding."""
        understanding_obj = {
            "topic": topic,
            "understanding": understanding,
            "depth": depth,
            "timestamp": len(self.understandings)
        }
        self.understandings.append(understanding_obj)
        self._recalculate_wisdom()
        return understanding_obj
    
    def add_insight(self, text: str) -> None:
        """Add a philosophical insight."""
        self.insights.append(text)
        self._recalculate_wisdom()
    
    def set_virtue(self, name: str, level: float) -> None:
        """Set a virtue level."""
        self.virtues[name] = max(0.0, min(1.0, level))
        self._recalculate_wisdom()
    
    def get_virtue(self, name: str) -> float:
        """Get a virtue level."""
        return self.virtues.get(name, 0.0)
    
    def _recalculate_wisdom(self) -> None:
        """Recalculate overall wisdom level."""
        understanding_score = sum(u["depth"] for u in self.understandings) / max(1, len(self.understandings))
        virtue_score = sum(self.virtues.values()) / max(1, len(self.virtues))
        insight_score = min(1.0, len(self.insights) / 10.0)
        
        self.wisdom_level = (understanding_score * 0.4 + virtue_score * 0.4 + insight_score * 0.2)
    
    def achieve_ataraxia(self) -> bool:
        """Achieve complete tranquility."""
        return self.wisdom_level >= 0.8 and all(v >= 0.6 for v in self.virtues.values())
    
    def overcome_fear_of_death(self) -> str:
        """Articulate the Epicurean argument against fear of death."""
        return ("When we exist, death is not; when death exists, we are not. "
                "Therefore, death is nothing to the living and nothing to the dead.")
    
    def overcome_fear_of_gods(self) -> str:
        """Articulate the argument against fear of gods."""
        return ("The gods either wish to prevent evils and cannot, or wish to do so and can, "
                "or wish to do so and cannot, or wish neither. If they wish to prevent and cannot, "
                "they are weak. If they can and do not wish, they are malevolent. "
                "If they wish and cannot, they are caught in fate. If they wish neither to prevent "
                "nor to, they are indifferent. Therefore, gods do not concern themselves with us.")
    
    def get_practical_wisdom(self) -> Dict[str, str]:
        """Get practical wisdom guidelines."""
        return {
            "pleasure": "Choose pleasures that bring lasting tranquility over momentary excitement",
            "pain": "Endure pain that leads to greater pleasure or health",
            "friendship": "Cultivate friendships as the greatest source of happiness",
            "self_sufficiency": "Desire only what nature and reason require",
            "justice": "Act justly because secure relations with others bring peace"
        }
    
    def demonstrate_epicurean_tetrad(self) -> Dict[str, str]:
        """Demonstrate the fourfold remedy (tetrapharmakos)."""
        return {
            "god": "Do not fear divine punishment - the gods are indifferent",
            "death": "Do not fear death - it is nothing to us",
            "pleasure": "Seek pleasure - the good is to be chosen",
            "pain": "Endure pain - some pains are necessary for greater goods"
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of sapientia state."""
        return {
            "understandings": len(self.understandings),
            "insights": len(self.insights),
            "virtues": dict(self.virtues),
            "wisdom_level": self.wisdom_level
        }


# =============================================================================
# SECTION 4: HIGH-LEVEL ARCHITECTURE CLASSES
# =============================================================================

class LucretiusCognitiveArchitecture:
    """
    The complete cognitive architecture of Lucretius.
    Integrates all five layers into a unified system for
    understanding Epicurean philosophy and atomic theory.
    """
    
    def __init__(self):
        self.layer1_sensus = Layer1_Sensus()
        self.layer2_atomus = Layer2_Atomus()
        self.layer3_voluptas = Layer3_Voluptas()
        self.layer4_natura = Layer4_Natura()
        self.layer5_sapientia = Layer5_Sapientia()
        
        self.name = "Lucretius"
        self.period = "-99 CE"
        self.school = "Epicurean"
        self.id = id(self)
        
        self._initialize_defaults()
    
    def _initialize_defaults(self) -> None:
        """Initialize default atoms and natural laws."""
        # Create the four classical elements (in Epicurean sense)
        elements = [
            ("fire", 1.0),
            ("air", 0.8),
            ("water", 1.2),
            ("earth", 1.5)
        ]
        self.layer2_atomus.create_from_elements(elements)
        
        # Add natural laws
        self.layer4_natura.add_natural_law(
            "motion", 
            "All things move through the void continuously",
            "v = constant unless acted upon"
        )
        self.layer4_natura.add_natural_law(
            "gravity",
            "Heavy atoms tend toward the center",
            "F = m * g"
        )
        self.layer4_natura.add_natural_law(
            "combination",
            "Atoms combine based on shape and motion",
            "shape compatibility"
        )
        
        # Set default virtues
        self.layer5_sapientia.set_virtue("prudence", 0.5)
        self.layer5_sapientia.set_virtue("justice", 0.5)
        self.layer5_sapientia.set_virtue("courage", 0.5)
        self.layer5_sapientia.set_virtue("temperance", 0.5)
    
    def __repr__(self) -> str:
        return f"LucretiusCognitiveArchitecture(period={self.period}, school={self.school})"
    
    def sense_and_perceive(self, channel: str, data: Any) -> Dict[str, Any]:
        """Layer 1: Process sensation."""
        self.layer1_sensus.receive_sensation(channel, data)
        perception = self.layer1_sensus.form_perception(data)
        self.layer1_sensus.verify_perception(perception)
        return perception
    
    def create_atomic_structure(self, atoms_spec: List[Tuple[str, float]]) -> CompoundTerm:
        """Layer 2: Create atoms and form compounds."""
        atoms = self.layer2_atomus.create_from_elements(atoms_spec)
        return self.layer2_atomus.combine_atoms(atoms)
    
    def apply_swerve(self, atom: Term, angle: float) -> Dict[str, Any]:
        """Apply the swerve (clinamen) to an atom."""
        return self.layer2_atomus.apply_swerve(atom, angle)
    
    def calculate_pleasure(self, name: str, intensity: float, 
                           duration: float) -> float:
        """Layer 3: Calculate pleasure value."""
        self.layer3_voluptas.add_pleasure(name, intensity, duration)
        return self.layer3_voluptas.evaluate_pleasure(
            self.layer3_voluptas.pleasures[-1]
        )
    
    def calculate_pain(self, name: str, intensity: float,
                       duration: float) -> float:
        """Calculate pain value."""
        self.layer3_voluptas.add_pain(name, intensity, duration)
        return -intensity * duration
    
    def create_natural_phenomenon(self, name: str, atoms: List[Term]) -> Individual:
        """Layer 4: Create a natural phenomenon."""
        return self.layer4_natura.create_individual(name, atoms)
    
    def apply_natural_laws(self, individual: Individual) -> Dict[str, Any]:
        """Apply natural laws to an individual."""
        return self.layer4_natura.apply_laws(individual)
    
    def gain_wisdom(self, topic: str, understanding: str) -> None:
        """Layer 5: Gain wisdom through understanding."""
        self.layer5_sapientia.add_understanding(topic, understanding)
        self.layer5_sapientia.add_insight(understanding)
    
    def achieve_philosophical_enlightenment(self) -> bool:
        """Check if full enlightenment has been achieved."""
        return self.layer5_sapientia.achieve_ataraxia()
    
    def get_full_state(self) -> Dict[str, Any]:
        """Get the complete state of all layers."""
        return {
            "layer1_sensus": self.layer1_sensus.get_summary(),
            "layer2_atomus": self.layer2_atomus.get_summary(),
            "layer3_voluptas": self.layer3_voluptas.get_summary(),
            "layer4_natura": self.layer4_natura.get_summary(),
            "layer5_sapientia": self.layer5_sapientia.get_summary()
        }


class EpicureanPhysicsSimulation:
    """
    A simulation of Epicurean physics demonstrating the behavior
    of atoms in the void, including the swerve and formation of compounds.
    """
    
    def __init__(self, bounds: Tuple[float, float, float] = (100.0, 100.0, 100.0)):
        self.bounds = bounds
        self.atoms: List[Dict[str, Any]] = []
        self.compounds: List[Dict[str, Any]] = []
        self.time = 0.0
        self.timestep = 0.1
        self.id = id(self)
    
    def __repr__(self) -> str:
        return f"EpicureanPhysicsSimulation(atoms={len(self.atoms)}, time={self.time:.2f})"
    
    def spawn_atoms(self, count: int, mass_range: Tuple[float, float] = (0.5, 2.0)) -> List[Dict[str, Any]]:
        """Spawn random atoms in the void."""
        spawned = []
        for i in range(count):
            atom = {
                "id": len(self.atoms) + i,
                "position": (
                    random.uniform(0, self.bounds[0]),
                    random.uniform(0, self.bounds[1]),
                    random.uniform(0, self.bounds[2])
                ),
                "velocity": (
                    random.uniform(-1, 1),
                    random.uniform(-1, 1),
                    random.uniform(-1, 1)
                ),
                "mass": random.uniform(mass_range[0], mass_range[1]),
                "swerve_angle": 0.0
            }
            spawned.append(atom)
            self.atoms.append(atom)
        return spawned
    
    def apply_swerve_to_all(self, swerve_strength: float = 0.1) -> None:
        """Apply random swerve (clinamen) to all atoms."""
        for atom in self.atoms:
            swerve_angle = random.uniform(-swerve_strength, swerve_strength)
            atom["swerve_angle"] = swerve_angle
    
    def step(self) -> Dict[str, Any]:
        """Advance the simulation by one timestep."""
        step_results = {"moved": [], "collisions": []}
        
        for atom in self.atoms:
            # Calculate displacement from velocity
            vx, vy, vz = atom["velocity"]
            px, py, pz = atom["position"]
            
            # Apply swerve to velocity
            swerve = atom["swerve_angle"]
            new_vx = vx + swerve * random.uniform(-0.1, 0.1)
            new_vy = vy + swerve * random.uniform(-0.1, 0.1)
            new_vz = vz + swerve * random.uniform(-0.1, 0.1)
            
            # Update position
            new_px = px + new_vx * self.timestep
            new_py = py + new_vy * self.timestep
            new_pz = pz + new_vz * self.timestep
            
            # Boundary handling (wrap around)
            new_px = new_px % self.bounds[0]
            new_py = new_py % self.bounds[1]
            new_pz = new_pz % self.bounds[2]
            
            atom["position"] = (new_px, new_py, new_pz)
            atom["velocity"] = (new_vx, new_vy, new_vz)
            
            step_results["moved"].append(atom["id"])
        
        self.time += self.timestep
        return step_results
    
    def check_collision(self, atom1: Dict[str, Any], atom2: Dict[str, Any]) -> bool:
        """Check if two atoms are colliding."""
        p1 = atom1["position"]
        p2 = atom2["position"]
        distance = math.sqrt(
            (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2
        )
        collision_threshold = (atom1["mass"] + atom2["mass"]) / 10
        return distance < collision_threshold
    
    def combine_atoms(self, atom_ids: List[int]) -> Optional[Dict[str, Any]]:
        """Combine atoms into a compound."""
        atoms_to_combine = [a for a in self.atoms if a["id"] in atom_ids]
        if len(atoms_to_combine) < 2:
            return None
        
        compound = {
            "id": len(self.compounds),
            "constituent_atoms": [a["id"] for a in atoms_to_combine],
            "position": tuple(sum(a["position"][i] for a in atoms_to_combine) / len(atoms_to_combine) 
                              for i in range(3)),
            "total_mass": sum(a["mass"] for a in atoms_to_combine),
            "creation_time": self.time
        }
        self.compounds.append(compound)
        return compound
    
    def get_atom_count(self) -> int:
        return len(self.atoms)
    
    def get_compound_count(self) -> int:
        return len(self.compounds)
    
    def run_steps(self, steps: int) -> None:
        """Run multiple simulation steps."""
        for _ in range(steps):
            self.step()


class DeRerumNaturaEngine:
    """
    The De Rerum Natura engine - the core computational system
    for implementing Lucretius's masterwork.
    """
    
    def __init__(self):
        self.books: List[Dict[str, Any]] = []
        self.current_book = 0
        self.topics: Dict[str, List[str]] = {}
        self.arguments: List[Syllogism] = []
        self.id = id(self)
        
        self._initialize_books()
    
    def _initialize_books(self) -> None:
        """Initialize the six books of De Rerum Natura."""
        book_contents = [
            {
                "book": 1,
                "title": "De Rerum Natura - Book I",
                "topic": "The Nature of the Gods and the Universe",
                "summary": "Proves that the universe consists of atoms and void. "
                          "Nothing comes from nothing. The universe is infinite."
            },
            {
                "book": 2,
                "title": "De Rerum Natura - Book II",
                "topic": "The Nature of Atoms and Motion",
                "summary": "Describes the shapes of atoms, their motions, and "
                          "the swerve (clinamen) that prevents determinism."
            },
            {
                "book": 3,
                "title": "De Rerum Natura - Book III",
                "topic": "The Nature of the Soul",
                "summary": "Proves the soul is material and mortal. "
                          "Frees humanity from fear of death."
            },
            {
                "book": 4,
                "title": "De Rerum Natura - Book IV",
                "topic": "Sensation, Mind, and Pleasure",
                "summary": "Explains sensation and thought as atomic processes. "
                          "Discusses the nature of pleasure and desire."
            },
            {
                "book": 5,
                "title": "De Rerum Natura - Book V",
                "topic": "The Formation of the World",
                "summary": "Describes the evolution of the world and life. "
                          "Critiques astrology and theology."
            },
            {
                "book": 6,
                "title": "De Rerum Natura - Book VI",
                "topic": "Heavenly Phenomena and Ethics",
                "summary": "Explains meteorological phenomena naturally. "
                          "Concludes with the fourfold remedy."
            }
        ]
        
        for book in book_contents:
            self.add_book(book["book"], book["title"], book["topic"], book["summary"])
    
    def add_book(self, number: int, title: str, topic: str, summary: str) -> Dict[str, Any]:
        """Add a book to the work."""
        book = {
            "number": number,
            "title": title,
            "topic": topic,
            "summary": summary,
            "arguments": [],
            "demonstrations": []
        }
        self.books.append(book)
        if topic not in self.topics:
            self.topics[topic] = []
        self.topics[topic].append(title)
        return book
    
    def add_argument(self, syllogism: Syllogism, book_number: int) -> None:
        """Add an argument (syllogism) to a book."""
        self.arguments.append(syllogism)
        for book in self.books:
            if book["number"] == book_number:
                book["arguments"].append(syllogism)
    
    def demonstrate_atoms_exist(self) -> Syllogism:
        """Create the fundamental argument for atomic theory."""
        # Major premise: Everything that exists has parts or is indivisible
        major = Proposition(
            Term("existence", "concept"),
            "is composed of parts OR is indivisible",
            Proposition.TruthValue.TRUE
        )
        major.assert_true("A thing either has parts or has no parts")
        
        # Minor premise: If something has parts, those parts exist
        minor = Proposition(
            Term("composite_thing", "concept"),
            "has parts that themselves exist",
            Proposition.TruthValue.TRUE
        )
        minor.assert_true("Parts must exist for the whole to exist")
        
        # Conclusion: Therefore, existence ultimately consists of indivisible units (atoms)
        conclusion = Proposition(
            Term("universe", "concept"),
            "ultimately consists of indivisible units",
            Proposition.TruthValue.TRUE
        )
        conclusion.assert_true("The infinite regress must terminate in atoms")
        
        syllogism = Syllogism(major, minor, conclusion, "atomic_existence")
        return syllogism
    
    def demonstrate_no_creation_from_nothing(self) -> Syllogism:
        """Argue that nothing comes from nothing."""
        major = Proposition(
            Term("creation"),
            "requires pre-existing matter",
            Proposition.TruthValue.TRUE
        )
        major.assert_true("All experience confirms this")
        
        minor = Proposition(
            Term("nothing"),
            "contains no matter",
            Proposition.TruthValue.TRUE
        )
        minor.assert_true("By definition, nothing has no properties")
        
        conclusion = Proposition(
            Term("universe"),
            "cannot arise from nothing",
            Proposition.TruthValue.TRUE
        )
        conclusion.assert_true("Therefore the universe always existed")
        
        syllogism = Syllogism(major, minor, conclusion, "no_creation_from_nothing")
        return syllogism
    
    def demonstrate_soul_is_mortal(self) -> Syllogism:
        """Argue that the soul is material and mortal."""
        major = Proposition(
            Term("soul"),
            "is affected by bodily states",
            Proposition.TruthValue.TRUE
        )
        major.assert_true("Soul feels joy, sorrow, pain")
        
        minor = Proposition(
            Term("immaterial"),
            "cannot be affected by matter",
            Proposition.TruthValue.TRUE
        )
        minor.assert_true("Immaterial things are unchanging")
        
        conclusion = Proposition(
            Term("soul"),
            "is material and mortal",
            Proposition.TruthValue.TRUE
        )
        conclusion.assert_true("Therefore soul dies with the body")
        
        syllogism = Syllogism(major, minor, conclusion, "soul_mortality")
        return syllogism
    
    def get_arguments_for_topic(self, topic: str) -> List[Syllogism]:
        """Get all arguments related to a topic."""
        return [arg for arg in self.arguments if topic.lower() in str(arg).lower()]
    
    def get_verse(self, book_number: int, theme: str) -> str:
        """Get a summary verse for a theme."""
        verses = {
            1: {
                "atoms": "Primordial atoms, countless in number, / Move through endless void / Forming all that is.",
                "void": "The void exists - without it, / Movement would be impossible, / And nothing could be.",
                "infinity": "The universe knows no bound, / No edge, no center, / Infinite in all directions."
            },
            3: {
                "death": "Why fear death, that nothing is to us? / When we are, death has not come, / When death comes, we are not."
            },
            4: {
                "sensation": "Sensation is the foundation, / The source of all knowledge, / The touchstone of truth."
            },
            5: {
                "nature": "Observe nature's workings, / See how things arise naturally, / No need for gods or magic."
            }
        }
        return verses.get(book_number, {}).get(theme, "Lucretian verse not found.")


class LucretiusAGIAlignment:
    """
    AGI Alignment system based on Lucretian/Epicurean principles.
    Uses atomic materialism and practical ethics to ensure
    alignment with human flourishing.
    """
    
    def __init__(self):
        self.principles: List[Dict[str, Any]] = []
        self.constraints: List[Dict[str, Any]] = []
        self.value_model = EpicureanValueModel()
        self.alignment_score = 0.0
        self.id = id(self)
        
        self._initialize_principles()
    
    def _initialize_principles(self) -> None:
        """Initialize the core Epicurean principles."""
        self.add_principle(
            "pleasure_as_good",
            "Pleasure is the beginning and end of the happy life",
            0.9
        )
        self.add_principle(
            "nature_as_guide",
            "Nature is the best guide for living well",
            0.85
        )
        self.add_principle(
            "autonomy",
            "Autonomy and freedom are essential for happiness",
            0.8
        )
        self.add_principle(
            "friendship",
            "Friendship is the greatest source of happiness",
            0.9
        )
        self.add_principle(
            "limits_of_desire",
            "Natural and necessary desires should be satisfied; vain desires should be transcended",
            0.85
        )
    
    def add_principle(self, name: str, description: str, weight: float) -> None:
        """Add an alignment principle."""
        principle = {
            "name": name,
            "description": description,
            "weight": weight
        }
        self.principles.append(principle)
    
    def add_constraint(self, name: str, description: str, 
                       severity: float = 1.0) -> None:
        """Add an alignment constraint."""
        constraint = {
            "name": name,
            "description": description,
            "severity": severity
        }
        self.constraints.append(constraint)
    
    def evaluate_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate an action against Epicurean principles."""
        pleasure_score = self.value_model.calculate_pleasure(action)
        alignment_score = sum(p["weight"] for p in self.principles) / len(self.principles)
        
        result = {
            "action": action.get("name", "unknown"),
            "pleasure_score": pleasure_score,
            "alignment_score": alignment_score,
            "approved": pleasure_score > 0.3 and alignment_score > 0.5,
            "recommendation": self._get_recommendation(pleasure_score, alignment_score)
        }
        
        self.alignment_score = (pleasure_score + alignment_score) / 2
        return result
    
    def _get_recommendation(self, pleasure: float, alignment: float) -> str:
        """Get a recommendation based on scores."""
        if pleasure > 0.7 and alignment > 0.7:
            return "strongly_approve"
        elif pleasure > 0.5 and alignment > 0.5:
            return "approve"
        elif pleasure > 0.3 or alignment > 0.3:
            return "caution"
        else:
            return "reject"
    
    def check_constraint_violation(self, action: Dict[str, Any]) -> List[str]:
        """Check if an action violates any constraints."""
        violations = []
        for constraint in self.constraints:
            if self._violates_constraint(action, constraint):
                violations.append(constraint["description"])
        return violations
    
    def _violates_constraint(self, action: Dict[str, Any], constraint: Dict[str, Any]) -> bool:
        """Check if a specific constraint is violated."""
        # Simplified check - in practice would be more sophisticated
        harm_keywords = ["harm", "destroy", "damage", "deceive"]
        if any(kw in str(action).lower() for kw in harm_keywords):
            return constraint.get("name", "").lower() in str(action).lower()
        return False
    
    def get_alignment_report(self) -> Dict[str, Any]:
        """Get a comprehensive alignment report."""
        return {
            "principles": self.principles,
            "constraints": self.constraints,
            "alignment_score": self.alignment_score,
            "approved_actions": 0,  # Would track this in practice
            "violations": 0
        }


class EpicureanValueModel:
    """
    A model for evaluating actions according to Epicurean ethics.
    pleasure is the highest good, and the goal is to maximize
    tranquility while minimizing pain.
    """
    
    def __init__(self):
        self.pleasure_weights: Dict[str, float] = {
            "friendship": 1.0,
            "health": 0.9,
            "freedom": 0.9,
            "knowledge": 0.8,
            "justice": 0.8,
            "virtue": 0.7,
            "pleasure": 0.6,
            "wealth": 0.3,
            "fame": 0.2,
            "power": 0.2
        }
    
    def calculate_pleasure(self, action: Dict[str, Any]) -> float:
        """Calculate the net pleasure of an action."""
        gains = sum(
            self.pleasure_weights.get(g, 0.3) 
            for g in action.get("gains", [])
        )
        losses = sum(
            self.pleasure_weights.get(l, 0.3) 
            for l in action.get("losses", [])
        )
        
        # Normalize
        max_possible = sum(self.pleasure_weights.values())
        net = (gains - losses) / max_possible
        
        return max(0.0, min(1.0, net))
    
    def evaluate_desire(self, desire: str) -> str:
        """Evaluate whether a desire should be pursued."""
        if desire in ["friendship", "health", "freedom", "knowledge"]:
            return "natural_and_necessary"
        elif desire in ["justice", "virtue"]:
            return "natural_but_not_necessary"
        else:
            return "vain_and_empty"


class AtomicMaterialismFramework:
    """
    A comprehensive framework for understanding reality through
    the lens of atomic materialism. All phenomena are explained
    as interactions between atoms in the void.
    """
    
    def __init__(self):
        self.atom_types: Dict[str, Dict[str, Any]] = {}
        self.compounds: List[CompoundTerm] = []
        self.phenomena: List[Dict[str, Any]] = []
        self.id = id(self)
        
        self._initialize_atom_types()
    
    def _initialize_atom_types(self) -> None:
        """Initialize the basic types of atoms."""
        self.register_atom_type("fire", {"shape": "pointed", "weight": 1.0, "property": "heat"})
        self.register_atom_type("air", {"shape": "smooth", "weight": 0.8, "property": "light"})
        self.register_atom_type("water", {"shape": "rounded", "weight": 1.2, "property": "fluid"})
        self.register_atom_type("earth", {"shape": "rough", "weight": 1.5, "property": "solid"})
    
    def register_atom_type(self, name: str, properties: Dict[str, Any]) -> None:
        """Register a type of atom."""
        self.atom_types[name] = properties
    
    def create_atom(self, atom_type: str, name: str) -> Optional[Term]:
        """Create an atom of a given type."""
        if atom_type not in self.atom_types:
            return None
        
        props = self.atom_types[atom_type].copy()
        atom = Term(name, "atom", props["weight"], props)
        return atom
    
    def explain_phenomenon(self, phenomenon: str) -> Dict[str, Any]:
        """Explain a natural phenomenon through atomic theory."""
        explanations = {
            "wind": "Air atoms moving rapidly through the void",
            "rain": "Water atoms falling due to gravity",
            "fire": "Pointed fire atoms rapidly moving and penetrating",
            "life": "Complex arrangement of atoms with soul-atoms",
            "sensation": "Atomic images (simulacra) striking the sense organs",
            "thought": "Fine soul-atoms moving within the body",
            "death": "Separation of atoms that previously formed a living being",
            "dream": "Atomic images that have detached and float in the void"
        }
        
        return {
            "phenomenon": phenomenon,
            "explanation": explanations.get(phenomenon, "Unknown phenomenon"),
            "atomic_basis": True
        }
    
    def simulate_formation(self, atom_specs: List[Tuple[str, str]]) -> Optional[CompoundTerm]:
        """Simulate the formation of a compound from atoms."""
        atoms = []
        for atom_type, name in atom_specs:
            atom = self.create_atom(atom_type, name)
            if atom:
                atoms.append(atom)
        
        if not atoms:
            return None
        
        compound = CompoundTerm(atoms, "formation")
        compound.set_property("origin", "simulated_formation")
        self.compounds.append(compound)
        return compound
    
    def decompose_compound(self, compound: CompoundTerm) -> List[Term]:
        """Decompose a compound into its atomic constituents."""
        atoms = compound.decompose()
        self.compounds.remove(compound)
        return atoms
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the framework state."""
        return {
            "atom_types": len(self.atom_types),
            "compounds": len(self.compounds),
            "phenomena": len(self.phenomena)
        }


# =============================================================================
# SECTION 5: UTILITY CLASSES AND FUNCTIONS
# =============================================================================

class LucretianReasoner:
    """
    A reasoner that applies Lucretian/Epicurean logic to solve problems.
    """
    
    def __init__(self):
        self.premises: List[Proposition] = []
        self.conclusions: List[Proposition] = []
        self.objections: List[Objection] = []
        self.resolutions: List[Resolution] = []
    
    def add_premise(self, subject: Term, predicate: str, 
                    truth_value: Proposition.TruthValue = Proposition.TruthValue.UNKNOWN) -> Proposition:
        """Add a premise to the reasoner."""
        premise = Proposition(subject, predicate, truth_value)
        self.premises.append(premise)
        return premise
    
    def derive_conclusion(self, subject: Term, predicate: str) -> Optional[Proposition]:
        """Derive a conclusion from premises."""
        # Simple derivation - in practice would be more sophisticated
        relevant_premises = [p for p in self.premises if p.truth_value == Proposition.TruthValue.TRUE]
        
        if len(relevant_premises) >= 2:
            conclusion = Proposition(subject, predicate, Proposition.TruthValue.TRUE)
            conclusion.assert_true("Derived from true premises")
            self.conclusions.append(conclusion)
            return conclusion
        return None
    
    def add_objection(self, target: Union[Proposition, Syllogism],
                      text: str, source: str = "anonymous") -> Objection:
        """Add an objection."""
        objection = Objection(target, text, source)
        self.objections.append(objection)
        return objection
    
    def resolve_objection(self, objection: Objection, 
                          text: str, reasoning: str) -> Resolution:
        """Resolve an objection."""
        resolution = Resolution(objection, text, reasoning)
        self.resolutions.append(resolution)
        return resolution
    
    def get_valid_syllogisms(self) -> List[Syllogism]:
        """Get all valid syllogisms from premises."""
        valid = []
        for i, p1 in enumerate(self.premises):
            for p2 in self.premises[i+1:]:
                conclusion = self.derive_conclusion(
                    Term("derived"), "from premises"
                )
                if conclusion:
                    syllogism = Syllogism(p1, p2, conclusion)
                    if syllogism.evaluate():
                        valid.append(syllogism)
        return valid


class EpicureanValidator:
    """
    Validates propositions and arguments against Epicurean principles.
    """
    
    def __init__(self):
        self.principles = [
            "Nothing comes from nothing",
            "Nothing returns to nothing",
            "The universe is atomic",
            "The void exists",
            "Pleasure is the highest good"
        ]
    
    def validate(self, proposition: Proposition) -> bool:
        """Validate a proposition against Epicurean principles."""
        text = str(proposition).lower()
        
        # Check for contradictions with core principles
        if "comes from nothing" in text and "not" not in text:
            return False
        if "returns to nothing" in text and "not" not in text:
            return False
        
        return True
    
    def check_consistency(self, propositions: List[Proposition]) -> bool:
        """Check consistency among propositions."""
        for i, p1 in enumerate(propositions):
            for p2 in propositions[i+1:]:
                if not self._are_consistent(p1, p2):
                    return False
        return True
    
    def _are_consistent(self, p1: Proposition, p2: Proposition) -> bool:
        """Check if two propositions are consistent."""
        # Simple consistency check
        if p1.is_true() and p2.is_true():
            return True
        if p1.is_false() and p2.is_false():
            return True
        if p1.is_true() and p2.is_false():
            return self._check_contradiction(p1, p2)
        return True
    
    def _check_contradiction(self, p1: Proposition, p2: Proposition) -> bool:
        """Check if contradictory propositions can coexist."""
        # In Epicurean logic, some apparent contradictions resolve at deeper levels
        return True


class LucretianQuote:
    """
    Represents famous quotes from Lucretius's De Rerum Natura.
    """
    
    def __init__(self, text: str, book: int, line: int, context: str = ""):
        self.text = text
        self.book = book
        self.line = line
        self.context = context
    
    def __repr__(self) -> str:
        return f"LucretianQuote(book={self.book}, line={self.line})"
    
    def __str__(self) -> str:
        return f'"{self.text}" - De Rerum Natura, Book {self.book}'

    def get_philosophical_point(self) -> str:
        """Extract the philosophical point of the quote."""
        if "atoms" in self.text.lower() or "atoms" in self.context.lower():
            return "Atomic theory"
        if "death" in self.text.lower():
            return "Mortality and fear of death"
        if "pleasure" in self.text.lower():
            return "Hedonistic ethics"
        if "nature" in self.text.lower():
            return "Naturalism"
        if "gods" in self.text.lower():
            return "Theology critique"
        return "Epicurean philosophy"


# =============================================================================
# SECTION 6: DEMONSTRATION
# =============================================================================

def demo():
    """
    Full demonstration of the Lucretian cognitive architecture.
    Shows all five layers working together.
    """
    print("=" * 80)
    print("DEMONSTRATION: Figure 105 - Lucretius (-99 CE)")
    print("De Rerum Natura Cognitive Architecture")
    print("=" * 80)
    
    # Create the main architecture
    architecture = LucretiusCognitiveArchitecture()
    
    print("\n" + "-" * 40)
    print("LAYER 1: SENSUS (SENSATION)")
    print("-" * 40)
    
    # Simulate sensations
    sense_channels = ["visual", "auditory", "tactile"]
    for channel in sense_channels:
        data = {
            "channel": channel,
            "intensity": random.uniform(0.3, 1.0),
            "timestamp": 0,
            "source": "simulation"
        }
        result = architecture.sense_and_perceive(channel, data)
        print(f"  {channel.capitalize()}: perceived intensity={result['intensity']:.2f}")
    
    print(f"\n  Layer 1 Summary: {architecture.layer1_sensus.get_summary()}")
    
    print("\n" + "-" * 40)
    print("LAYER 2: ATOMUS (ATOMS)")
    print("-" * 40)
    
    # Create atoms
    atom_specs = [
        ("fire", 1.0),
        ("air", 0.8),
        ("water", 1.2),
        ("earth", 1.5)
    ]
    compound = architecture.create_atomic_structure(atom_specs)
    print(f"  Created compound: {compound}")
    print(f"  Compound components: {len(compound.get_components())}")
    
    # Apply swerve
    atoms = architecture.layer2_atomus.get_all_atoms()
    for atom in atoms[:3]:  # Apply to first 3
        swerve_result = architecture.apply_swerve(atom, random.uniform(0.1, 0.5))
        print(f"  Applied swerve to {atom.name}: angle={swerve_result['angle']:.3f}")
    
    print(f"\n  Layer 2 Summary: {architecture.layer2_atomus.get_summary()}")
    
    print("\n" + "-" * 40)
    print("LAYER 3: VOLUPTAS (PLEASURE)")
    print("-" * 40)
    
    # Add pleasures
    pleasures = [
        ("friendship", 0.9, 10.0),
        ("knowledge", 0.8, 8.0),
        ("health", 0.85, 9.0),
        ("freedom", 0.75, 7.0)
    ]
    
    for name, intensity, duration in pleasures:
        net = architecture.calculate_pleasure(name, intensity, duration)
        print(f"  Added pleasure '{name}': intensity={intensity}, duration={duration}, net={net:.3f}")
    
    # Add some pain
    architecture.calculate_pain("hunger", 0.3, 2.0)
    print(f"  Added pain 'hunger'")
    
    print(f"\n  Ataraxia (tranquility): {architecture.layer3_voluptas.get_ataraxia():.3f}")
    print(f"  Aponia (no pain): {architecture.layer3_voluptas.get_aponia():.3f}")
    print(f"  Net pleasure: {architecture.layer3_voluptas.get_net_pleasure():.3f}")
    
    print("\n" + "-" * 40)
    print("LAYER 4: NATURA (NATURE)")
    print("-" * 40)
    
    # Create natural phenomena
    phenomenon_atoms = architecture.layer2_atomus.get_all_atoms()
    individuals = []
    
    for i, name in enumerate(["fire_flame", "water_drop", "air_breeze"]):
        ind = architecture.create_natural_phenomenon(
            name, 
            phenomenon_atoms[i:i+2]
        )
        individuals.append(ind)
        print(f"  Created natural phenomenon: {ind}")
    
    # Apply natural laws
    for ind in individuals:
        result = architecture.apply_natural_laws(ind)
        print(f"  Applied laws to {ind.name}: {result['law_effects']}")
    
    print(f"\n  Layer 4 Summary: {architecture.layer4_natura.get_summary()}")
    
    print("\n" + "-" * 40)
    print("LAYER 5: SAPIENTIA (WISDOM)")
    print("-" * 40)
    
    # Gain wisdom
    teachings = [
        ("death", "Death is nothing to us - when we are, death has not come; when death comes, we are not."),
        ("gods", "The gods are indifferent to human affairs and do not punish us."),
        ("nature", "All phenomena arise from natural causes - no supernatural intervention exists."),
        ("pleasure", "The highest good is freedom from pain and disturbance."),
        ("atoms", "All things are composed of atoms moving in the void.")
    ]
    
    for topic, understanding in teachings:
        architecture.gain_wisdom(topic, understanding)
        print(f"  Gained understanding of '{topic}':")
        print(f"    \"{understanding[:60]}...\"" if len(understanding) > 60 else f"    \"{understanding}\"")
    
    print(f"\n  Wisdom level: {architecture.layer5_sapientia.wisdom_level:.3f}")
    print(f"  Enlightenment achieved: {architecture.achieve_philosophical_enlightenment()}")
    
    print("\n" + "-" * 40)
    print("PHYSICS SIMULATION")
    print("-" * 40)
    
    simulation = EpicureanPhysicsSimulation()
    print(f"  Created simulation: {simulation}")
    
    # Spawn atoms
    spawned = simulation.spawn_atoms(20)
    print(f"  Spawned {len(spawned)} atoms")
    
    # Run simulation steps
    for step in range(5):
        result = simulation.step()
        print(f"  Step {step + 1}: {len(result['moved'])} atoms moved")
    
    print(f"  Final atom count: {simulation.get_atom_count()}")
    print(f"  Time elapsed: {simulation.time:.2f}")
    
    print("\n" + "-" * 40)
    print("DE RERUM NATURA ENGINE")
    print("-" * 40)
    
    engine = DeRerumNaturaEngine()
    print(f"  Created engine with {len(engine.books)} books")
    
    # Demonstrate arguments
    arguments = [
        engine.demonstrate_atoms_exist(),
        engine.demonstrate_no_creation_from_nothing(),
        engine.demonstrate_soul_is_mortal()
    ]
    
    for arg in arguments:
        print(f"\n  Argument: {arg.name}")
        print(f"    {arg}")
        print(f"    Valid: {arg.evaluate()}")
    
    print("\n" + "-" * 40)
    print("AGI ALIGNMENT")
    print("-" * 40)
    
    alignment = LucretiusAGIAlignment()
    print(f"  Created alignment system")
    print(f"  Principles: {len(alignment.principles)}")
    
    # Evaluate some actions
    actions = [
        {"name": "help_human", "gains": ["friendship", "knowledge"], "losses": []},
        {"name": "harm_human", "gains": [], "losses": ["health", "freedom"]},
        {"name": "share_knowledge", "gains": ["knowledge", "friendship"], "losses": ["wealth"]}
    ]
    
    for action in actions:
        result = alignment.evaluate_action(action)
        print(f"  Action '{action['name']}':")
        print(f"    Pleasure score: {result['pleasure_score']:.3f}")
        print(f"    Alignment score: {result['alignment_score']:.3f}")
        print(f"    Recommendation: {result['recommendation']}")
    
    print("\n" + "-" * 40)
    print("ATOMIC MATERIALISM FRAMEWORK")
    print("-" * 40)
    
    framework = AtomicMaterialismFramework()
    print(f"  Created framework with {len(framework.atom_types)} atom types")
    
    # Explain phenomena
    phenomena = ["wind", "rain", "fire", "sensation", "dream"]
    for phenomenon in phenomena:
        explanation = framework.explain_phenomenon(phenomenon)
        print(f"  {phenomenon.capitalize()}:")
        print(f"    {explanation['explanation']}")
    
    print("\n" + "-" * 40)
    print("SYLLOGISMS AND OBJECTIONS")
    print("-" * 40)
    
    # Create syllogisms with objections
    p1 = Proposition(Term("soul"), "is affected by body", Proposition.TruthValue.TRUE)
    p1.assert_true("Experience shows this")
    
    p2 = Proposition(Term("immaterial"), "cannot be affected", Proposition.TruthValue.TRUE)
    p2.assert_true("Definition of immaterial")
    
    conclusion = Proposition(Term("soul"), "is material", Proposition.TruthValue.TRUE)
    conclusion.assert_true("Follows from premises")
    
    syllogism = Syllogism(p1, p2, conclusion, "soul_materiality")
    print(f"  Created syllogism: {syllogism.name}")
    print(f"  Valid: {syllogism.evaluate()}")
    
    # Add objection
    objection = Objection(syllogism, "How can the soul be material if it thinks?")
    print(f"  Objection: {objection}")
    
    # Resolve objection
    resolution = Resolution(
        objection,
        "The soul uses fine atoms that can think without being immaterial",
        "This follows Epicurean physics where thought is a physical process"
    )
    print(f"  Resolution: {resolution.text}")
    
    print("\n" + "-" * 40)
    print("LUCRETIAN QUOTES")
    print("-" * 40)
    
    quotes = [
        LucretianQuote(
            "Sweet is the terror of the day and the night, the fear of gods, and the underworld's dread.",
            1, 102, "Opening invocation to Venus"
        ),
        LucretianQuote(
            "Nothing exists except atoms and void; all else is talk.",
            2, 333, "Statement of atomic theory"
        ),
        LucretianQuote(
            "When we exist, death is not; when death comes, we are not.",
            3, 830, "Argument against fear of death"
        ),
        LucretianQuote(
            "Pleasure is the beginning and end of the happy life.",
            2, 17, "Core of Epicurean ethics"
        )
    ]
    
    for quote in quotes:
        print(f"  {quote}")
        print(f"    Philosophical point: {quote.get_philosophical_point()}")
    
    print("\n" + "-" * 40)
    print("COMPLETE ARCHITECTURE STATE")
    print("-" * 40)
    
    state = architecture.get_full_state()
    for layer, summary in state.items():
        print(f"  {layer}: {summary}")
    
    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    demo()