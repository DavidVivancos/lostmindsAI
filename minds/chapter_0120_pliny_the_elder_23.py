#!/usr/bin/env python3
"""
Chapter 120: Pliny the Elder — The Naturalist's Encyclopedia and the Empirical Mind
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 6 Minds 101 - 120 Available on Amazon https://www.amazon.com/dp/B0HF7G6JJD
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 120: Pliny the Elder (23 to 79 CE)
================================================================================
Plinian Encyclopedia Architecture implementing Pliny's principles:
- Empirical observation as foundation of knowledge
- Encyclopedia structure with hierarchical classification
- Multi-source data integration and quality assessment
- Scientific reasoning (pattern recognition, causal reasoning, model building)
- Uncertainty propagation through knowledge system
- Cross-domain integration and synthesis
- Curiosity-driven learning and knowledge discovery

This architecture demonstrates how Pliny's encyclopedic approach
translates into modern AI frameworks for knowledge management.
"""

import math
import random
from typing import Dict, List, Any, Optional, Tuple, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque


# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================

class Kingdom(Enum):
    """Broad natural kingdoms."""
    LIVING = "living"
    NON_LIVING = "non_living"
    ABSTRACT = "abstract"


class SourceType(Enum):
    """Types of information sources."""
    DIRECT_OBSERVATION = "direct_observation"
    WRITTEN = "written"
    ORAL = "oral"
    EXPERIMENTAL = "experimental"
    DERIVED = "derived"


class Certainty(Enum):
    """Certainty levels for knowledge."""
    CERTAIN = "certain"
    PROBABLE = "probable"
    POSSIBLE = "possible"
    DOUBTFUL = "doubtful"


@dataclass
class Entity:
    """A natural entity."""
    id: str
    name: str
    kingdom: Kingdom
    properties: Dict[str, Any] = field(default_factory=dict)
    relationships: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class Observation:
    """An observation record."""
    entity_id: str
    property_name: str
    value: Any
    source_type: SourceType
    source_name: str
    certainty: Certainty = Certainty.PROBABLE
    timestamp: float = 0.0


@dataclass
class KnowledgeState:
    """State of knowledge about an entity."""
    entity_id: str
    observations: List[Observation] = field(default_factory=list)
    inferred_properties: Dict[str, float] = field(default_factory=dict)
    uncertainty: float = 0.5


# ============================================================================
# TAXONOMIC HIERARCHY
# ============================================================================

class TaxonomicHierarchy:
    """
    Hierarchical classification system.
    
    Implements Pliny's approach to organizing nature into
    hierarchical categories from broadest to most specific.
    """
    
    def __init__(self):
        self.levels = {
            'domain': [],
            'kingdom': [],
            'phylum': [],
            'class_level': [],
            'order': [],
            'family': [],
            'genus': [],
            'species': [],
            'instance': []
        }
        self.entity_to_level = {}
        
    def add_entity(self, entity_id: str, classification: Dict[str, str]):
        """Add an entity with its classification."""
        self.entity_to_level[entity_id] = classification
        
        for level, value in classification.items():
            if value not in self.levels.get(level, []):
                self.levels[level].append(value)
                
    def get_parent(self, entity_id: str, level: str) -> Optional[str]:
        """Get parent of entity at specified level."""
        if entity_id not in self.entity_to_level:
            return None
            
        classification = self.entity_to_level[entity_id]
        
        # Define parent levels
        level_order = ['instance', 'species', 'genus', 'family', 'order', 
                      'class_level', 'phylum', 'kingdom', 'domain']
        
        if level not in level_order:
            return None
            
        level_idx = level_order.index(level)
        if level_idx >= len(level_order) - 1:
            return None
            
        parent_level = level_order[level_idx + 1]
        
        return classification.get(parent_level)
        
    def get_children(self, level: str, value: str) -> List[str]:
        """Get all entities below a level."""
        children = []
        
        for entity_id, classification in self.entity_to_level.items():
            if classification.get(level) == value:
                children.append(entity_id)
                
        return children
        
    def get_lineage(self, entity_id: str) -> Dict[str, str]:
        """Get full lineage from domain to instance."""
        if entity_id not in self.entity_to_level:
            return {}
            
        return self.entity_to_level[entity_id].copy()


# ============================================================================
# KNOWLEDGE BASE
# ============================================================================

class KnowledgeBase:
    """
    Structured knowledge base.
    
    Implements Pliny's encyclopedia — a structured repository
    of natural knowledge organized for efficient retrieval.
    """
    
    def __init__(self):
        self.entities = {}
        self.observations = []
        self.relationships = defaultdict(list)
        self.uncertainty_model = {}
        
    def add_entity(self, entity: Entity):
        """Add an entity to the knowledge base."""
        self.entities[entity.id] = entity
        
    def add_observation(self, observation: Observation):
        """Add an observation."""
        self.observations.append(observation)
        
    def add_relationship(self, entity1_id: str, relationship: str, entity2_id: str):
        """Add a relationship between entities."""
        self.relationships[entity1_id].append({
            'type': relationship,
            'target': entity2_id
        })
        
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID."""
        return self.entities.get(entity_id)
        
    def get_observations(self, entity_id: str, property_name: str = None) -> List[Observation]:
        """Get observations for an entity."""
        obs = [o for o in self.observations if o.entity_id == entity_id]
        
        if property_name:
            obs = [o for o in obs if o.property_name == property_name]
            
        return obs
        
    def query(self, criteria: Dict) -> List[Entity]:
        """Query entities by criteria."""
        results = []
        
        for entity in self.entities.values():
            match = True
            
            for key, value in criteria.items():
                if key == 'kingdom':
                    if entity.kingdom != value:
                        match = False
                        break
                elif key == 'property':
                    if value not in entity.properties:
                        match = False
                        break
                elif key not in entity.properties and key != 'kingdom':
                    match = False
                    break
                    
            if match:
                results.append(entity)
                
        return results


# ============================================================================
# SOURCE EVALUATION
# ============================================================================

class SourceEvaluator:
    """
    Source quality evaluation.
    
    Implements Pliny's careful evaluation of source reliability.
    """
    
    def __init__(self):
        self.source_records = {}
        self.evaluation_history = []
        
    def register_source(self, source_id: str, source_info: Dict):
        """Register a source with its characteristics."""
        self.source_records[source_id] = {
            'name': source_info.get('name', 'Unknown'),
            'type': source_info.get('type', SourceType.WRITTEN),
            'domain_expertise': source_info.get('domain_expertise', {}),
            'historical_accuracy': source_info.get('historical_accuracy', 0.5),
            'consistency': 0.5,
            'total_reports': 0,
            'agreement_count': 0
        }
        
    def evaluate_report(self, source_id: str, report: Dict) -> Dict:
        """Evaluate a specific report from a source."""
        if source_id not in self.source_records:
            return {'reliability': 0.3, 'confidence': 0.0}
            
        source = self.source_records[source_id]
        
        # Calculate reliability
        base_reliability = source['historical_accuracy']
        expertise_bonus = source['domain_expertise'].get(
            report.get('domain', 'general'), 0.0
        )
        consistency_bonus = source['consistency'] * 0.2
        
        reliability = min(1.0, base_reliability + expertise_bonus * 0.3 + consistency_bonus)
        
        # Update source record
        source['total_reports'] += 1
        if report.get('consistent_with_other_sources', True):
            source['agreement_count'] += 1
            source['consistency'] = source['agreement_count'] / source['total_reports']
            
        return {
            'reliability': reliability,
            'confidence': reliability * source['consistency'],
            'source_type': source['type'].value
        }
        
    def get_source_quality(self, source_id: str) -> float:
        """Get overall quality score for a source."""
        if source_id not in self.source_records:
            return 0.3
            
        source = self.source_records[source_id]
        return (source['historical_accuracy'] * 0.5 + 
                source['consistency'] * 0.3 +
                sum(source['domain_expertise'].values()) / max(1, len(source['domain_expertise'])) * 0.2)


# ============================================================================
# UNCERTAINTY PROPAGATION
# ============================================================================

class UncertaintyPropagation:
    """
    Uncertainty propagation through knowledge system.
    
    Implements Pliny's acknowledgment of uncertainty
    and its propagation through reasoning.
    """
    
    def __init__(self):
        self.uncertainty_sources = {}
        self.propagation_rules = {}
        
    def observe_uncertainty(self, observation_id: str, uncertainty: float, source: str):
        """Record uncertainty in an observation."""
        self.uncertainty_sources[observation_id] = {
            'uncertainty': uncertainty,
            'source': source
        }
        
    def propagate_through_reasoning(self, input_uncertainties: List[float],
                                   reasoning_type: str) -> float:
        """
        Propagate uncertainty through a reasoning process.
        
        Different reasoning types propagate uncertainty differently.
        """
        if reasoning_type == 'deduction':
            # Deduction: uncertainty bounded by premises
            return min(input_uncertainties) if input_uncertainties else 1.0
            
        elif reasoning_type == 'induction':
            # Induction: uncertainty increases with generalization
            base_uncertainty = max(input_uncertainties) if input_uncertainties else 0.0
            generalization_factor = 1.1  # Induction adds uncertainty
            return min(1.0, base_uncertainty * generalization_factor)
            
        elif reasoning_type == 'abduction':
            # Abduction: uncertainty based on best explanation
            return min(input_uncertainties) * 0.9 if input_uncertainties else 1.0
            
        elif reasoning_type == 'analogy':
            # Analogy: uncertainty based on similarity
            avg_uncertainty = sum(input_uncertainties) / len(input_uncertainties) if input_uncertainties else 0.5
            return min(1.0, avg_uncertainty * 1.2)  # Analogy is less certain
            
        return 0.5  # Default
        
    def calculate_conclusion_uncertainty(self, premises: List[Dict]) -> float:
        """Calculate overall conclusion uncertainty from premises."""
        if not premises:
            return 1.0
            
        uncertainties = [p.get('uncertainty', 0.5) for p in premises]
        
        # Combine uncertainties using evidence accumulation
        combined = 1.0
        for u in uncertainties:
            combined *= (1.0 - u)
            
        return 1.0 - combined


# ============================================================================
# PATTERN RECOGNITION
# ============================================================================

class PatternRecognition:
    """
    Pattern recognition from observations.
    
    Implements Pliny's observation of regularities in nature.
    """
    
    def __init__(self):
        self.patterns = {}
        self.observation_history = []
        
    def detect_patterns(self, observations: List[Observation]) -> List[Dict]:
        """Detect patterns in observations."""
        detected = []
        
        # Group by entity and property
        grouped = defaultdict(list)
        for obs in observations:
            key = (obs.entity_id, obs.property_name)
            grouped[key].append(obs)
            
        # Look for patterns in groups
        for (entity_id, prop), obs_list in grouped.items():
            if len(obs_list) >= 3:
                pattern = self._analyze_sequence(obs_list)
                if pattern:
                    detected.append(pattern)
                    
        return detected
        
    def _analyze_sequence(self, obs_list: List[Observation]) -> Optional[Dict]:
        """Analyze a sequence of observations for patterns."""
        values = [o.value for o in obs_list]
        
        if not all(isinstance(v, (int, float)) for v in values):
            return None
            
        # Check for trend
        increasing = all(values[i] <= values[i+1] for i in range(len(values)-1))
        decreasing = all(values[i] >= values[i+1] for i in range(len(values)-1))
        
        if increasing:
            return {
                'type': 'increasing_trend',
                'entity': obs_list[0].entity_id,
                'property': obs_list[0].property_name,
                'values': values
            }
        elif decreasing:
            return {
                'type': 'decreasing_trend',
                'entity': obs_list[0].entity_id,
                'property': obs_list[0].property_name,
                'values': values
            }
            
        return None
        
    def match_pattern(self, pattern: Dict, new_data: List[float]) -> float:
        """Match new data against a known pattern."""
        if pattern['type'] == 'increasing_trend':
            return self._match_trend(new_data, increasing=True)
        elif pattern['type'] == 'decreasing_trend':
            return self._match_trend(new_data, increasing=False)
        return 0.0
        
    def _match_trend(self, data: List[float], increasing: bool) -> float:
        """Match data to a trend."""
        if increasing:
            matches = sum(1 for i in range(len(data)-1) if data[i] <= data[i+1])
        else:
            matches = sum(1 for i in range(len(data)-1) if data[i] >= data[i+1])
            
        return matches / max(1, len(data) - 1)


# ============================================================================
# CAUSAL REASONING
# ============================================================================

class CausalReasoning:
    """
    Causal reasoning engine.
    
    Implements Pliny's interest in causes and effects in nature.
    """
    
    def __init__(self):
        self.causal_graph = defaultdict(list)
        self.causal_strength = {}
        
    def add_causal_relation(self, cause: str, effect: str, 
                           strength: float = 0.5,
                           mechanism: str = "unknown"):
        """Add a causal relationship."""
        self.causal_graph[cause].append({
            'effect': effect,
            'strength': strength,
            'mechanism': mechanism
        })
        self.causal_strength[(cause, effect)] = strength
        
    def infer_causes(self, effect: str) -> List[Tuple[str, float]]:
        """Infer possible causes of an effect."""
        causes = []
        
        for cause, effects in self.causal_graph.items():
            for e in effects:
                if e['effect'] == effect:
                    causes.append((cause, e['strength']))
                    
        causes.sort(key=lambda x: x[1], reverse=True)
        return causes
        
    def infer_effects(self, cause: str) -> List[Tuple[str, float]]:
        """Infer possible effects of a cause."""
        effects = []
        
        for e in self.causal_graph.get(cause, []):
            effects.append((e['effect'], e['strength']))
            
        effects.sort(key=lambda x: x[1], reverse=True)
        return effects
        
    def predict(self, cause: str, chain_length: int = 2) -> Dict[str, float]:
        """Predict effects through causal chain."""
        predictions = {}
        current_causes = [(cause, 1.0)]
        visited = set()
        
        for _ in range(chain_length):
            next_causes = []
            
            for c, prob in current_causes:
                if c in visited:
                    continue
                visited.add(c)
                
                effects = self.infer_effects(c)
                for effect, strength in effects:
                    if effect not in predictions:
                        predictions[effect] = 0.0
                    predictions[effect] += prob * strength
                    next_causes.append((effect, prob * strength))
                    
            current_causes = next_causes
            
        return predictions


# ============================================================================
# ENCYCLOPEDIA INTERFACE
# ============================================================================

class EncyclopediaInterface:
    """
    Encyclopedia query and navigation interface.
    
    Implements Pliny's encyclopedia as a usable knowledge system.
    """
    
    def __init__(self, knowledge_base: KnowledgeBase, 
                 taxonomy: TaxonomicHierarchy):
        self.kb = knowledge_base
        self.taxonomy = taxonomy
        self.query_log = []
        
    def query_entity(self, query: str) -> List[Entity]:
        """Query for entities by name or property."""
        self.query_log.append({'type': 'entity_query', 'query': query})
        
        results = []
        query_lower = query.lower()
        
        for entity in self.kb.entities.values():
            if query_lower in entity.name.lower():
                results.append(entity)
            elif any(query_lower in str(v).lower() for v in entity.properties.values()):
                results.append(entity)
                
        return results
        
    def query_by_classification(self, level: str, value: str) -> List[Entity]:
        """Query entities by taxonomic classification."""
        self.query_log.append({
            'type': 'classification_query',
            'level': level,
            'value': value
        })
        
        results = []
        for entity_id, classification in self.taxonomy.entity_to_level.items():
            if classification.get(level) == value:
                entity = self.kb.get_entity(entity_id)
                if entity:
                    results.append(entity)
                    
        return results
        
    def query_by_property(self, property_name: str, 
                         value: Any = None) -> List[Entity]:
        """Query entities by property."""
        self.query_log.append({
            'type': 'property_query',
            'property': property_name,
            'value': value
        })
        
        results = []
        for entity in self.kb.entities.values():
            if property_name in entity.properties:
                if value is None or entity.properties[property_name] == value:
                    results.append(entity)
                    
        return results
        
    def get_related_entities(self, entity_id: str, 
                           relationship_type: str = None) -> List[str]:
        """Get entities related to specified entity."""
        related = []
        
        for rel in self.kb.relationships.get(entity_id, []):
            if relationship_type is None or rel['type'] == relationship_type:
                related.append(rel['target'])
                
        return related
        
    def navigate_hierarchy(self, entity_id: str, direction: str = 'up') -> List[str]:
        """Navigate up or down the taxonomic hierarchy."""
        if direction == 'up':
            return self._navigate_up(entity_id)
        else:
            return self._navigate_down(entity_id)
            
    def _navigate_up(self, entity_id: str) -> List[str]:
        """Navigate up to higher taxonomic levels."""
        lineage = self.taxonomy.get_lineage(entity_id)
        result = []
        
        level_order = ['instance', 'species', 'genus', 'family', 'order',
                      'class_level', 'phylum', 'kingdom', 'domain']
                      
        for level in level_order[1:]:  # Skip instance
            if level in lineage:
                result.append(lineage[level])
                
        return result
        
    def _navigate_down(self, entity_id: str) -> List[str]:
        """Navigate down to lower taxonomic levels."""
        lineage = self.taxonomy.get_lineage(entity_id)
        entity_level = lineage.get('level', 'species')
        
        level_order = ['instance', 'species', 'genus', 'family', 'order',
                      'class_level', 'phylum', 'kingdom', 'domain']
                      
        if entity_level not in level_order:
            return []
            
        level_idx = level_order.index(entity_level)
        if level_idx >= len(level_order) - 1:
            return []
            
        next_level = level_order[level_idx + 1]
        next_value = lineage.get(entity_level)
        
        return self.taxonomy.get_children(next_level, next_value)


# ============================================================================
# CURIOSITY MODULE
# ============================================================================

class CuriosityModule:
    """
    Curiosity-driven learning module.
    
    Implements Pliny's insatiable curiosity as a驱动因子
    for active knowledge acquisition.
    """
    
    def __init__(self):
        self.knowledge_gaps = []
        self.exploration_history = []
        self.question_queue = deque()
        
    def identify_gaps(self, knowledge_base: KnowledgeBase) -> List[Dict]:
        """Identify gaps in current knowledge."""
        gaps = []
        
        for entity_id, entity in knowledge_base.entities.items():
            # Check for missing properties
            if len(entity.properties) < 3:
                gaps.append({
                    'entity_id': entity_id,
                    'gap_type': 'sparse_properties',
                    'priority': 1.0 - len(entity.properties) / 10
                })
                
            # Check for unobserved relationships
            if entity_id not in knowledge_base.relationships:
                gaps.append({
                    'entity_id': entity_id,
                    'gap_type': 'no_relationships',
                    'priority': 0.7
                })
                
        self.knowledge_gaps = gaps
        return gaps
        
    def generate_question(self, gap: Dict) -> str:
        """Generate a question to fill a knowledge gap."""
        entity = gap.get('entity_id', 'unknown')
        gap_type = gap.get('gap_type', 'unknown')
        
        if gap_type == 'sparse_properties':
            return f"What properties does {entity} have?"
        elif gap_type == 'no_relationships':
            return f"What is related to {entity}?"
        else:
            return f"Tell me more about {entity}."
            
    def prioritize_exploration(self, gaps: List[Dict]) -> List[Dict]:
        """Prioritize exploration targets by expected knowledge value."""
        return sorted(gaps, key=lambda g: g.get('priority', 0.5), reverse=True)


# ============================================================================
# MODEL BUILDING
# ============================================================================

class ModelBuilder:
    """
    Scientific model building.
    
    Implements Pliny's construction of mental models of natural systems.
    """
    
    def __init__(self):
        self.models = {}
        self.model_parameters = {}
        
    def build_static_model(self, entity: Entity) -> Dict:
        """Build a static model of an entity."""
        model_id = f"static_{entity.id}"
        
        model = {
            'type': 'static',
            'entity_id': entity.id,
            'properties': entity.properties.copy(),
            'relationships': entity.relationships.copy(),
            'structure_score': self._assess_structure(entity)
        }
        
        self.models[model_id] = model
        return model
        
    def build_dynamic_model(self, entity_id: str, 
                           time_series: List[Dict]) -> Dict:
        """Build a dynamic model from time series data."""
        model_id = f"dynamic_{entity_id}"
        
        # Analyze time series for patterns
        values = [t.get('value', 0) for t in time_series]
        
        model = {
            'type': 'dynamic',
            'entity_id': entity_id,
            'time_series': time_series,
            'mean': sum(values) / len(values) if values else 0,
            'variance': self._calculate_variance(values),
            'trend': self._detect_trend(values),
            'periodicity': self._detect_periodicity(values)
        }
        
        self.models[model_id] = model
        return model
        
    def build_causal_model(self, causal: CausalReasoning, 
                         root_cause: str) -> Dict:
        """Build a causal model starting from a root cause."""
        model_id = f"causal_{root_cause}"
        
        effects = causal.predict(root_cause, chain_length=3)
        
        model = {
            'type': 'causal',
            'root_cause': root_cause,
            'predictions': effects,
            'complexity': len(effects)
        }
        
        self.models[model_id] = model
        return model
        
    def _assess_structure(self, entity: Entity) -> float:
        """Assess structural richness of entity model."""
        property_score = min(1.0, len(entity.properties) / 10)
        relationship_score = min(1.0, sum(len(r) for r in entity.relationships.values()) / 10)
        return (property_score + relationship_score) / 2
        
    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of values."""
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((v - mean)**2 for v in values) / len(values)
        return variance
        
    def _detect_trend(self, values: List[float]) -> str:
        """Detect trend in values."""
        if len(values) < 2:
            return 'insufficient_data'
            
        increasing = sum(1 for i in range(len(values)-1) if values[i] <= values[i+1])
        decreasing = sum(1 for i in range(len(values)-1) if values[i] >= values[i+1])
        
        if increasing > len(values) * 0.7:
            return 'increasing'
        elif decreasing > len(values) * 0.7:
            return 'decreasing'
        else:
            return 'stable'
            
    def _detect_periodicity(self, values: List[float]) -> Optional[float]:
        """Detect periodicity (simplified)."""
        # Very simplified periodicity detection
        if len(values) < 4:
            return None
            
        # Check for repeating patterns
        for period in range(2, len(values) // 2):
            matches = 0
            for i in range(len(values) - period):
                if abs(values[i] - values[i + period]) < 0.1 * abs(values[i]):
                    matches += 1
                    
            if matches > len(values) * 0.5:
                return float(period)
                
        return None


# ============================================================================
# MAIN PLINIAN AGENT
# ============================================================================

class PlinianAgent:
    """
    Complete Plinian Encyclopedia Agent.
    
    Integrates all Plinian architecture components
    for encyclopedic knowledge management.
    """
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        
        # Core components
        self.kb = KnowledgeBase()
        self.taxonomy = TaxonomicHierarchy()
        self.encyclopedia = EncyclopediaInterface(self.kb, self.taxonomy)
        self.sources = SourceEvaluator()
        self.uncertainty = UncertaintyPropagation()
        self.patterns = PatternRecognition()
        self.causal = CausalReasoning()
        self.curiosity = CuriosityModule()
        self.models = ModelBuilder()
        
        self.state_history = []
        
    def observe(self, entity_id: str, property_name: str, value: Any,
               source_type: SourceType, source_name: str) -> Observation:
        """Record an observation."""
        obs = Observation(
            entity_id=entity_id,
            property_name=property_name,
            value=value,
            source_type=source_type,
            source_name=source_name
        )
        
        self.kb.add_observation(obs)
        
        # Update entity properties
        entity = self.kb.get_entity(entity_id)
        if entity:
            entity.properties[property_name] = value
            
        return obs
        
    def query(self, query: str) -> List[Entity]:
        """Query the encyclopedia."""
        return self.encyclopedia.query_entity(query)
        
    def build_model(self, entity_id: str, model_type: str = 'static') -> Dict:
        """Build a model of an entity."""
        entity = self.kb.get_entity(entity_id)
        
        if not entity:
            return {'error': 'Entity not found'}
            
        if model_type == 'static':
            return self.models.build_static_model(entity)
        elif model_type == 'dynamic':
            # Get time series for entity
            obs = self.kb.get_observations(entity_id)
            time_series = [{'value': o.value, 'time': o.timestamp} for o in obs]
            return self.models.build_dynamic_model(entity_id, time_series)
        elif model_type == 'causal':
            return self.models.build_causal_model(self.causal, entity_id)
            
        return {'error': 'Unknown model type'}
        
    def discover_knowledge_gaps(self) -> List[Dict]:
        """Identify gaps in knowledge and generate questions."""
        gaps = self.curiosity.identify_gaps(self.kb)
        
        questions = []
        for gap in gaps:
            questions.append(self.curiosity.generate_question(gap))
            
        return questions
        
    def assess_source_quality(self, source_id: str) -> float:
        """Assess the quality of a source."""
        return self.sources.get_source_quality(source_id)
        
    def get_encyclopedia_stats(self) -> Dict:
        """Get statistics about the encyclopedia."""
        return {
            'total_entities': len(self.kb.entities),
            'total_observations': len(self.kb.observations),
            'total_relationships': sum(len(v) for v in self.kb.relationships.values()),
            'taxonomic_levels': {k: len(v) for k, v in self.taxonomy.levels.items()},
            'patterns_detected': len(self.patterns.patterns),
            'causal_relations': len(self.causal.causal_graph)
        }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("Plinian Encyclopedia Architecture - Empirical Knowledge System")
    
    agent = PlinianAgent("pliny_test")
    
    # Add sample entities
    agent.kb.add_entity(Entity(
        id="lion_001",
        name="Lion",
        kingdom=Kingdom.LIVING,
        properties={'is_carnivore': True, 'has_mane': True, 'num_legs': 4}
    ))
    
    agent.kb.add_entity(Entity(
        id="oak_001",
        name="Oak Tree",
        kingdom=Kingdom.LIVING,
        properties={'is_deciduous': True, 'has_rough_bark': True, 'num_legs': 0}
    ))
    
    # Add taxonomic classification
    agent.taxonomy.add_entity("lion_001", {
        'domain': 'eukarya', 'kingdom': 'animalia', 'phylum': 'chordata',
        'class_level': 'mammalia', 'order': 'carnivora', 'family': 'felidae',
        'genus': 'panthera', 'species': 'leo', 'instance': 'lion_001'
    })
    
    # Record observations
    agent.observe("lion_001", "weight", 180.0, SourceType.DIRECT_OBSERVATION, "Pliny")
    agent.observe("lion_001", "speed", 80.0, SourceType.WRITTEN, "Aristotle")
    
    # Add causal relationships
    agent.causal.add_causal_relation("hunger", "hunting", 0.8, "physiological_need")
    agent.causal.add_causal_relation("hunting", "energy_gain", 0.9, "food_consumption")
    
    # Query encyclopedia
    results = agent.query("lion")
    print(f"Query results: {len(results)} entities found")
    
    # Build model
    model = agent.build_model("lion_001", "static")
    print(f"Model type: {model.get('type', 'unknown')}")
    
    # Discover knowledge gaps
    gaps = agent.discover_knowledge_gaps()
    print(f"Knowledge gaps identified: {len(gaps)}")
    
    # Source evaluation
    agent.sources.register_source("Pliny", {
        'name': 'Pliny the Elder',
        'type': SourceType.DIRECT_OBSERVATION,
        'domain_expertise': {'zoology': 0.8, 'botany': 0.7},
        'historical_accuracy': 0.75
    })
    
    quality = agent.assess_source_quality("Pliny")
    print(f"Pliny's source quality: {quality:.3f}")
    
    # Get encyclopedia statistics
    stats = agent.get_encyclopedia_stats()
    print(f"Total entities: {stats['total_entities']}")
    print(f"Total observations: {stats['total_observations']}")
    
    print("\nPlinian Encyclopedia Agent initialized successfully.")
    print(f"Total lines: {len(open(__file__, encoding='utf-8').read().splitlines())}")


class NaturalCalendarSystem:
    """Pliny's calendar of natural phenomena."""
    def __init__(self):
        self.seasonal_phenomena = {
            "spring": ["Birdsong peaks", "Trees bud", "Insects emerge"],
            "summer": ["Harvest begins", "Solstice period", "Thunderstorms"],
            "autumn": ["Leaf fall", "Bird migrations", "Harvest moon"],
            "winter": ["Solstice period", "Dormancy begins", "Northern Lights"]
        }

    def seasonal_phenomena_list(self, season: str) -> List[str]:
        return self.seasonal_phenomena.get(season, [])


class RomanMiningTechniques:
    """Roman mining methods Pliny describes."""
    def __init__(self):
        self.mining_methods = {
            "hydraulic": "Hushing - water floods to reveal ore",
            "open_pit": "Quarrying for surface deposits",
            "shaft": "Vertical shafts to reach deep veins",
            "adit": "Horizontal tunnels for drainage and access"
        }

    def mining_method_description(self, method: str) -> str:
        return self.mining_methods.get(method, "Unknown mining method")


class PharmaceuticalKnowledgeBase:
    """Pliny's pharmaceutical knowledge."""
    def __init__(self):
        self.remedies = {
            "veratrum": "White hellebore - emetic",
            "hellebore": "Black hellebore - purges",
            "opium": "Poppy juice - pain relief",
            "garlic": "Widespread remedy, 27 uses"
        }

    def remedy_uses(self, substance: str) -> str:
        return self.remedies.get(substance, "No recorded uses")


class ZoologicalBehaviorStudy:
    """Animal behavior observations from Pliny."""
    def __init__(self):
        self.behaviors = {
            "elephants": ["Mourning dead", "Teaching young", "Obeying authority"],
            "dolphins": ["Play with humans", "Protect drowning", "Musical sense"],
            "storks": ["Filial piety", "Migration patterns", "Nesting habits"],
            "ants": ["Food storage", "Communication", "Colony organization"]
        }

    def animal_behavior(self, species: str) -> List[str]:
        return self.behaviors.get(species, ["No observations recorded"])
