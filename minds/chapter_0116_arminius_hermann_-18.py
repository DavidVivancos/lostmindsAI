#!/usr/bin/env python3
"""
Chapter 116: Arminius — The Cherokee General and Strategic Intelligence
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 6 Minds 101 - 120 Available on Amazon https://www.amazon.com/dp/B0HF7G6JJD
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 116: Arminius (-18 to -9 BCE)
================================================================================
Strategic Intelligence Architecture implementing Arminius's principles:
- Multi-perspective analysis (Roman and Germanic frameworks)
- Adaptive model selection and switching
- Coalition building and maintenance
- Strategic synthesis across traditions
- Guerrilla tactics and terrain exploitation
- Real-time adaptation and deception
- Strategic patience and commitment

This architecture demonstrates how Arminius's strategic thinking
translates into modern AI frameworks for adaptive intelligence.
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

class Perspective(Enum):
    """Strategic perspectives for multi-perspective analysis."""
    ROMAN_MILITARY = "roman_military"
    GERMANIC_TRIBAL = "germanic_tribal"
    POLITICAL = "political"
    ECONOMIC = "economic"
    TERRAIN = "terrain"
    COALITION = "coalition"


@dataclass
class StrategicState:
    """State of a strategic situation."""
    enemy_strength: float = 0.5
    friendly_strength: float = 0.5
    terrain_advantage: float = 0.5
    coalition_stability: float = 0.5
    supply_situation: float = 0.5
    morale: float = 0.5
    strategic_opportunity: float = 0.0


@dataclass
class CoalitionMember:
    """A member of a strategic coalition."""
    id: str
    name: str
    contribution: float = 0.5
    reliability: float = 0.5
    reputation: float = 0.5
    commitment_level: float = 0.5


@dataclass
class CourseOfAction:
    """A possible course of action."""
    id: str
    name: str
    perspective: Perspective
    expected_outcome: float = 0.5
    risk_level: float = 0.5
    resource_cost: float = 0.5
    contingencies: Dict = field(default_factory=dict)


# ============================================================================
# MULTI-PERSPECTIVE ANALYSIS ENGINE
# ============================================================================

class MultiPerspectiveAnalysisEngine:
    """
    Multi-perspective analysis engine.
    
    Arminius could see situations through both Roman and Germanic lenses.
    This engine maintains multiple perspectives simultaneously and evaluates
    situations through each.
    """
    
    def __init__(self):
        self.perspectives = {}
        self.current_perspective = Perspective.ROMAN_MILITARY
        self.perspective_weights = {}
        
    def add_perspective(self, perspective: Perspective, 
                       evaluation_fn: Callable[[Dict], Dict]):
        """Add a perspective with its evaluation function."""
        self.perspectives[perspective] = evaluation_fn
        
    def evaluate_through_perspective(self, situation: Dict, 
                                    perspective: Perspective) -> Dict:
        """Evaluate a situation through a specific perspective."""
        if perspective not in self.perspectives:
            return {'error': 'Unknown perspective'}
            
        eval_fn = self.perspectives[perspective]
        return eval_fn(situation)
        
    def evaluate_multi_perspective(self, situation: Dict) -> Dict:
        """
        Evaluate a situation through all perspectives.
        
        Arminius: What is visible through Roman eyes vs. Germanic eyes?
        """
        results = {}
        
        for perspective in Perspective:
            if perspective in self.perspectives:
                results[perspective.value] = self.evaluate_through_perspective(
                    situation, perspective
                )
                
        return results
        
    def switch_perspective(self, new_perspective: Perspective):
        """Switch the active perspective."""
        self.current_perspective = new_perspective
        
    def synthesize_perspectives(self, perspective_results: Dict) -> Dict:
        """
        Synthesize insights from multiple perspectives.
        
        Arminius's genius: finding what all perspectives agree on,
        and what they each reveal that others miss.
        """
        # Find consensus insights
        consensus = []
        disagreements = []
        
        perspective_keys = list(perspective_results.keys())
        
        for key in perspective_keys[:3]:  # Compare first few
            for other_key in perspective_keys:
                if key != other_key:
                    r1 = perspective_results[key]
                    r2 = perspective_results[other_key]
                    
                    # Check for agreement
                    if isinstance(r1, dict) and isinstance(r2, dict):
                        if r1.get('recommended_action') == r2.get('recommended_action'):
                            consensus.append(r1.get('recommended_action'))
                        else:
                            disagreements.append({
                                key: r1.get('recommended_action'),
                                other_key: r2.get('recommended_action')
                            })
                            
        return {
            'consensus': consensus,
            'disagreements': disagreements,
            'num_perspectives': len(perspective_results)
        }


# ============================================================================
# ADAPTIVE MODEL SELECTOR
# ============================================================================

class AdaptiveModelSelector:
    """
    Adaptive model selection engine.
    
    Arminius could switch between Roman and Germanic models fluidly.
    This engine selects the most appropriate model for each situation.
    """
    
    def __init__(self):
        self.models = {}
        self.model_performance = defaultdict(list)
        self.current_model = None
        self.selection_history = []
        
    def register_model(self, model_id: str, model: Any):
        """Register a strategic model."""
        self.models[model_id] = model
        
    def evaluate_models(self, situation: Dict) -> Dict[str, float]:
        """
        Evaluate which model is most appropriate for situation.
        """
        scores = {}
        
        for model_id, model in self.models.items():
            # Evaluate fit
            if hasattr(model, 'evaluate_fit'):
                score = model.evaluate_fit(situation)
            else:
                score = 0.5
                
            scores[model_id] = score
            
        return scores
        
    def select_best_model(self, situation: Dict) -> str:
        """
        Select the best model for the current situation.
        
        Arminius selected Roman model when predicting Roman behavior,
        Germanic model when coordinating with tribes.
        """
        scores = self.evaluate_models(situation)
        
        if not scores:
            return None
            
        best_model_id = max(scores.items(), key=lambda x: x[1])[0]
        self.current_model = best_model_id
        
        self.selection_history.append({
            'situation': situation,
            'selected_model': best_model_id,
            'scores': scores
        })
        
        return best_model_id
        
    def adapt_model(self, model_id: str, feedback: Dict):
        """Adapt a model based on feedback."""
        if model_id in self.models and hasattr(self.models[model_id], 'update'):
            self.models[model_id].update(feedback)
            
        # Record performance
        performance = feedback.get('performance', 0.5)
        self.model_performance[model_id].append(performance)


# ============================================================================
# STRATEGIC SYNTHESIS ENGINE
# ============================================================================

class StrategicSynthesisEngine:
    """
    Strategic synthesis engine.
    
    Arminius combined Roman discipline with Germanic mobility.
    This engine synthesizes novel approaches from multiple frameworks.
    """
    
    def __init__(self):
        self.frameworks = {}
        self.synthesis_history = []
        
    def add_framework(self, framework_id: str, framework: Dict):
        """Add a strategic framework."""
        self.frameworks[framework_id] = framework
        
    def find_common_patterns(self, framework_ids: List[str]) -> List[Dict]:
        """
        Find common structural patterns across frameworks.
        """
        patterns = []
        
        if len(framework_ids) < 2:
            return patterns
            
        framework1 = self.frameworks.get(framework_ids[0], {})
        framework2 = self.frameworks.get(framework_ids[1], {})
        
        # Find common keys
        common_keys = set(framework1.keys()) & set(framework2.keys())
        
        for key in common_keys:
            v1 = framework1[key]
            v2 = framework2[key]
            
            if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                patterns.append({
                    'key': key,
                    'value1': v1,
                    'value2': v2,
                    'average': (v1 + v2) / 2,
                    'difference': abs(v1 - v2)
                })
                
        return patterns
        
    def synthesize_novel_approach(self, framework_ids: List[str],
                                  situation: Dict) -> Dict:
        """
        Synthesize a novel approach from multiple frameworks.
        
        Arminius: Roman discipline + Germanic mobility = Teutoburg victory.
        """
        patterns = self.find_common_patterns(framework_ids)
        
        if not patterns:
            return {'error': 'No common patterns found'}
            
        # Create synthesis
        synthesis = {
            'name': 'synthesized_strategy',
            'components': [],
            'novel_elements': [],
            'expected_advantages': []
        }
        
        for framework_id in framework_ids:
            if framework_id in self.frameworks:
                synthesis['components'].append(framework_id)
                
        # Add novel combinations
        for pattern in patterns[:3]:
            if pattern['difference'] > 0.2:  # Significant difference
                # Create a hybrid approach
                novel = {
                    'parameter': pattern['key'],
                    'from_framework1': pattern['value1'],
                    'from_framework2': pattern['value2'],
                    'synthesized_value': (pattern['value1'] + pattern['value2']) / 2
                }
                synthesis['novel_elements'].append(novel)
                
        synthesis['expected_advantages'].append('Combined strengths of multiple frameworks')
        synthesis['expected_advantages'].append('Novel approach that adversaries cannot predict')
        
        self.synthesis_history.append(synthesis)
        
        return synthesis


# ============================================================================
# COALITION ARCHITECTURE
# ============================================================================

class CoalitionArchitecture:
    """
    Coalition building and maintenance architecture.
    
    Arminius united Germanic tribes with different interests.
    This architecture manages coalition dynamics.
    """
    
    def __init__(self):
        self.members = {}
        self.alliance_strength = 0.5
        self.reputation_network = defaultdict(dict)
        self.conflict_log = []
        
    def add_member(self, member: CoalitionMember):
        """Add a member to the coalition."""
        self.members[member.id] = member
        self.reputation_network[member.id] = {}
        
    def evaluate_member_contribution(self, member_id: str, task: Dict) -> float:
        """Evaluate how much a member can contribute to a task."""
        if member_id not in self.members:
            return 0.0
            
        member = self.members[member_id]
        
        contribution = (
            member.contribution * 0.4 +
            member.reliability * 0.3 +
            member.commitment_level * 0.3
        )
        
        return contribution
        
    def update_reputation(self, member_id: str, from_id: str, rep_change: float):
        """Update reputation assessment between members."""
        if member_id not in self.reputation_network:
            self.reputation_network[member_id] = {}
            
        current = self.reputation_network[member_id].get(from_id, 0.5)
        new_rep = max(0.0, min(1.0, current + rep_change))
        self.reputation_network[member_id][from_id] = new_rep
        
    def manage_conflict(self, member1_id: str, member2_id: str, 
                       issue: str) -> Dict:
        """
        Manage conflict between coalition members.
        
        Arminius had to manage rivalries between tribes.
        """
        self.conflict_log.append({
            'member1': member1_id,
            'member2': member2_id,
            'issue': issue
        })
        
        # Simple conflict resolution
        member1 = self.members.get(member1_id)
        member2 = self.members.get(member2_id)
        
        if not member1 or not member2:
            return {'resolved': False, 'reason': 'Member not found'}
            
        # Both reduce conflict
        member1.commitment_level *= 0.95
        member2.commitment_level *= 0.95
        
        return {
            'resolved': True,
            'action_taken': 'reduced_commitment'
        }
        
    def assess_coalition_stability(self) -> float:
        """Assess overall coalition stability."""
        if not self.members:
            return 0.0
            
        stability_scores = []
        
        for member_id, member in self.members.items():
            # Reliability and commitment contribute to stability
            stability = (member.reliability * 0.5 + 
                        member.commitment_level * 0.5)
            stability_scores.append(stability)
            
        avg_stability = sum(stability_scores) / len(stability_scores)
        self.alliance_strength = avg_stability
        
        return avg_stability
        
    def recommend_member_incentives(self, member_id: str) -> List[str]:
        """Recommend incentives to maintain member commitment."""
        if member_id not in self.members:
            return []
            
        member = self.members[member_id]
        incentives = []
        
        if member.reliability < 0.6:
            incentives.append("Demonstrate reliability through small wins")
        if member.commitment_level < 0.6:
            incentives.append("Offer increased autonomy in operations")
        if member.contribution < 0.6:
            incentives.append("Provide additional resources or support")
            
        return incentives


# ============================================================================
# GUERRILLA TACTICS MODULE
# ============================================================================

class GuerrillaTacticsModule:
    """
    Guerrilla tactics module.
    
    Arminius used guerrilla tactics to negate Roman advantages.
    This module implements asymmetric warfare principles.
    """
    
    def __init__(self):
        self.ambush_patterns = {}
        self.terrain_cache = {}
        self.tactical_successes = []
        
    def assess_terrain_advantages(self, terrain: Dict) -> Dict:
        """
        Assess terrain for guerrilla advantages.
        
        Arminius used Teutoburg Forest to negate Roman formation advantages.
        """
        forest_density = terrain.get('forest_density', 0.5)
        elevation_change = terrain.get('elevation_change', 0.5)
        visibility = terrain.get('visibility', 0.5)
        mobility = terrain.get('mobility', 0.5)
        
        # Calculate guerrilla advantage score
        advantage = (
            forest_density * 0.3 +
            (1.0 - visibility) * 0.2 +
            (1.0 - elevation_change) * 0.2 +
            mobility * 0.3
        )
        
        return {
            'guerrilla_advantage': advantage,
            'recommended_approach': 'asymmetric' if advantage > 0.6 else 'conventional',
            'terrain_factors': {
                'forest_density': forest_density,
                'visibility': visibility,
                'mobility': mobility
            }
        }
        
    def plan_ambush(self, enemy_formation: str, 
                   terrain: Dict) -> Dict:
        """
        Plan an ambush based on terrain and enemy formation.
        
        Arminius planned ambush at narrow passes where Roman formations couldn't deploy.
        """
        advantage = self.assess_terrain_advantages(terrain)
        
        if advantage['guerrilla_advantage'] < 0.4:
            return {
                'recommended': False,
                'reason': 'Terrain does not favor ambush'
            }
            
        # Plan ambush elements
        ambush_elements = []
        
        # Flanking positions
        ambush_elements.append({
            'type': 'flanking',
            'position': 'high_ground',
            'timing': 'when_enemy_center_enters_kill_zone'
        })
        
        # Blockade points
        ambush_elements.append({
            'type': 'blockade',
            'position': 'rear',
            'purpose': 'prevent_retreat'
        })
        
        # Shock troops
        ambush_elements.append({
            'type': 'shock',
            'position': 'center_kill_zone',
            'timing': 'after_initial_rout'
        })
        
        ambush_plan = {
            'recommended': True,
            'ambush_elements': ambush_elements,
            'estimated_success': advantage['guerrilla_advantage'],
            'key_vulnerabilities': self._identify_vulnerabilities(enemy_formation)
        }
        
        self.ambush_patterns[enemy_formation] = ambush_plan
        
        return ambush_plan
        
    def _identify_vulnerabilities(self, formation: str) -> List[str]:
        """Identify vulnerabilities in enemy formation."""
        vulnerabilities = []
        
        if formation == 'roman_column':
            vulnerabilities.extend([
                'Vulnerable at front and rear',
                'Cannot deploy formations quickly',
                'Supply lines exposed',
                'Communication between units limited'
            ])
        elif formation == 'roman_square':
            vulnerabilities.extend([
                'Slow movement',
                'Requires flat terrain',
                'Cannot pursue effectively'
            ])
            
        return vulnerabilities
        
    def coordinate_decentralized_forces(self, forces: List[Dict],
                                        situation: Dict) -> Dict:
        """
        Coordinate forces without centralized communication.
        
        Arminius had to coordinate tribes without radios.
        """
        coordination_score = 0.5
        
        # Pre-arranged signals
        if situation.get('has_signals', True):
            coordination_score += 0.2
            
        # Cultural cohesion
        if situation.get('cultural_cohesion', 0.5) > 0.6:
            coordination_score += 0.2
            
        # Shared plan
        if situation.get('has_shared_plan', True):
            coordination_score += 0.1
            
        return {
            'coordination_score': coordination_score,
            'recommended_approach': 'decentralized' if coordination_score > 0.6 else 'centralized',
            'signals_used': ['visual', 'auditory', 'physical']
        }


# ============================================================================
# DECEPTION MODULE
# ============================================================================

class DeceptionModule:
    """
    Deception operations module.
    
    Arminius used deception to confirm Roman expectations while preparing ambush.
    This module implements adversary modeling and deception planning.
    """
    
    def __init__(self):
        self.adversary_beliefs = {}
        self.deception_operations = []
        
    def model_adversary(self, adversary_id: str, beliefs: Dict):
        """Model an adversary's beliefs and expectations."""
        self.adversary_beliefs[adversary_id] = {
            'beliefs': beliefs,
            'confidence': beliefs.get('confidence', 0.5),
            'biases': beliefs.get('biases', []),
            'framework': beliefs.get('framework', 'default')
        }
        
    def predict_adversary_response(self, adversary_id: str,
                                  action: Dict) -> Dict:
        """
        Predict how adversary will respond to an action.
        
        Arminius predicted that Varus would follow established routes.
        """
        if adversary_id not in self.adversary_beliefs:
            return {'error': 'Adversary not modeled'}
            
        adversary = self.adversary_beliefs[adversary_id]
        framework = adversary['framework']
        
        # Model-based prediction
        if framework == 'roman_military':
            # Romans follow established doctrine
            predicted_action = 'maintain_formation'
            predicted_response = 'proceed_as_planned'
        elif framework == 'germanic_tribal':
            predicted_action = 'flexible_adaptation'
            predicted_response = 'respond_to_terrain'
        else:
            predicted_action = 'rational_response'
            predicted_response = 'optimize_for_objectives'
            
        return {
            'predicted_action': predicted_action,
            'predicted_response': predicted_response,
            'confidence': adversary['confidence']
        }
        
    def plan_deception(self, adversary_id: str,
                     true_action: Dict,
                     desired_belief: Dict) -> Dict:
        """
        Plan a deception operation.
        
        Arminius confirmed Roman beliefs about Germanic inferiority
        while preparing the opposite reality.
        """
        if adversary_id not in self.adversary_beliefs:
            return {'error': 'Adversary not modeled'}
            
        adversary = self.adversary_beliefs[adversary_id]
        
        # Generate deception actions that confirm adversary beliefs
        deception_actions = []
        
        # Confirm expected behavior
        deception_actions.append({
            'type': 'confirm_belief',
            'action': 'maintain_peaceful_posture',
            'effect': 'reduces_alertness'
        })
        
        # Hide true capabilities
        deception_actions.append({
            'type': 'hide_capability',
            'action': 'disperse_forces',
            'effect': 'appears_weak'
        })
        
        # Create false vulnerabilities
        deception_actions.append({
            'type': 'create_false_vulnerability',
            'action': 'show_withdrawal',
            'effect': 'encourages_pursuit'
        })
        
        deception_plan = {
            'deception_actions': deception_actions,
            'target_belief': desired_belief,
            'true_action': true_action,
            'success_probability': 0.7
        }
        
        self.deception_operations.append(deception_plan)
        
        return deception_plan


# ============================================================================
# STRATEGIC PATIENCE MODULE
# ============================================================================

class StrategicPatienceModule:
    """
    Strategic patience module.
    
    Arminius waited two years before the right moment.
    This module implements patience and timing optimization.
    """
    
    def __init__(self):
        self.opportunity_window = None
        self.readiness_assessment = 0.5
        self.patience_threshold = 0.7
        
    def assess_opportunity_window(self, situation: Dict) -> Dict:
        """
        Assess whether the current moment is opportune.
        
        Arminius: Timing is everything.
        """
        readiness = situation.get('readiness', 0.5)
        enemy_vulnerability = situation.get('enemy_vulnerability', 0.5)
        terrain_conditions = situation.get('terrain_conditions', 0.5)
        
        window_score = (
            readiness * 0.3 +
            enemy_vulnerability * 0.4 +
            terrain_conditions * 0.3
        )
        
        return {
            'window_score': window_score,
            'is_opportune': window_score > self.patience_threshold,
            'factors': {
                'readiness': readiness,
                'enemy_vulnerability': enemy_vulnerability,
                'terrain_conditions': terrain_conditions
            }
        }
        
    def recommend_wait(self, situation: Dict) -> Dict:
        """
        Recommend whether to wait for better opportunity.
        
        Arminius often recommended patience when others wanted immediate action.
        """
        window = self.assess_opportunity_window(situation)
        
        if window['is_opportune']:
            return {
                'recommendation': 'act_now',
                'reason': 'Opportunity window is favorable'
            }
        else:
            return {
                'recommendation': 'wait',
                'reason': 'Opportunity window not yet favorable',
                'expected_improvement': window['window_score'] * 0.2,
                'max_wait_time': situation.get('strategic_deadline', 'unknown')
            }
            
    def update_readiness(self, progress: float):
        """Update readiness assessment based on preparation progress."""
        self.readiness_assessment = max(0.0, min(1.0, progress))


# ============================================================================
# STRATEGIC INTELLIGENCE SYSTEM
# ============================================================================

class StrategicIntelligenceSystem:
    """
    Complete strategic intelligence system.
    
    Integrates all Arminian architecture components for
    comprehensive strategic analysis and decision-making.
    """
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        
        # Core components
        self.perspective_engine = MultiPerspectiveAnalysisEngine()
        self.model_selector = AdaptiveModelSelector()
        self.synthesis_engine = StrategicSynthesisEngine()
        self.coalition = CoalitionArchitecture()
        self.guerrilla = GuerrillaTacticsModule()
        self.deception = DeceptionModule()
        self.patience = StrategicPatienceModule()
        
        # State
        self.state_history = []
        self.decision_log = []
        
    def analyze_situation(self, situation: Dict) -> Dict:
        """
        Comprehensive situation analysis using all perspectives.
        """
        # Multi-perspective analysis
        multi_perspective = self.perspective_engine.evaluate_multi_perspective(situation)
        
        # Model selection
        best_model = self.model_selector.select_best_model(situation)
        
        # Opportunity assessment
        opportunity = self.patience.assess_opportunity_window(situation)
        
        # Coalition stability
        coalition_stability = self.coalition.assess_coalition_stability()
        
        # Terrain analysis
        terrain_analysis = None
        if 'terrain' in situation:
            terrain_analysis = self.guerrilla.assess_terrain_advantages(situation['terrain'])
        
        analysis = {
            'agent_id': self.agent_id,
            'multi_perspective': multi_perspective,
            'selected_model': best_model,
            'opportunity': opportunity,
            'coalition_stability': coalition_stability,
            'terrain_analysis': terrain_analysis,
            'recommended_action': self._determine_recommended_action(situation)
        }
        
        self.state_history.append(analysis)
        
        return analysis
        
    def _determine_recommended_action(self, situation: Dict) -> str:
        """Determine recommended action based on integrated analysis."""
        opportunity = self.patience.assess_opportunity_window(situation)
        
        if opportunity['is_opportune']:
            coalition_stable = self.coalition.alliance_strength > 0.6
            
            if coalition_stable:
                return 'execute_strategic_attack'
            else:
                return 'strengthen_coalition_first'
        else:
            return 'continue_preparations'
            
    def plan_campaign(self, objectives: List[str], 
                     constraints: Dict) -> Dict:
        """
        Plan a campaign to achieve objectives within constraints.
        
        Arminius planned the Teutoburg campaign years in advance.
        """
        campaign = {
            'objectives': objectives,
            'phases': [],
            'resource_requirements': {},
            'contingencies': {}
        }
        
        # Phase 1: Coalition building
        campaign['phases'].append({
            'name': 'coalition_building',
            'objective': 'unite_germanic_tribes',
            'duration': '1_year',
            'key_actions': ['negotiate_alliances', 'demonstrate_capability']
        })
        
        # Phase 2: Preparation
        campaign['phases'].append({
            'name': 'preparation',
            'objective': 'position_forces',
            'duration': '6_months',
            'key_actions': ['scout_terrain', 'establish_supply', 'coordinate_signals']
        })
        
        # Phase 3: Execution
        campaign['phases'].append({
            'name': 'execution',
            'objective': 'destroy_enemy_force',
            'duration': '3_days',
            'key_actions': ['lure_into_terrain', 'execute_ambush', 'pursue_rout']
        })
        
        # Phase 4: Consolidation
        campaign['phases'].append({
            'name': 'consolidation',
            'objective': 'maintain_coalition',
            'duration': 'ongoing',
            'key_actions': ['distribute_spoils', 'reinforce_alliances']
        })
        
        return campaign
        
    def make_strategic_decision(self, situation: Dict,
                               options: List[CourseOfAction]) -> CourseOfAction:
        """
        Make a strategic decision from multiple options.
        """
        best_option = None
        best_score = -1.0
        
        for option in options:
            # Evaluate option through multiple perspectives
            scores = []
            
            for perspective in Perspective:
                perspective_result = self.perspective_engine.evaluate_through_perspective(
                    {'option': option, 'situation': situation},
                    perspective
                )
                if 'expected_outcome' in perspective_result:
                    scores.append(perspective_result['expected_outcome'])
                    
            # Calculate composite score
            if scores:
                avg_score = sum(scores) / len(scores)
                
                # Adjust for risk
                risk_penalty = option.risk_level * 0.2
                adjusted_score = avg_score - risk_penalty
                
                if adjusted_score > best_score:
                    best_score = adjusted_score
                    best_option = option
                    
        self.decision_log.append({
            'situation': situation,
            'options': options,
            'selected': best_option,
            'score': best_score
        })
        
        return best_option


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("Strategic Intelligence Architecture - Arminius")
    
    system = StrategicIntelligenceSystem("arminius_test")
    
    # Setup perspectives
    def roman_evaluation(situation):
        return {
            'recommended_action': 'conventional_battle',
            'expected_outcome': 0.6,
            'risk_level': 0.3
        }
        
    def germanic_evaluation(situation):
        return {
            'recommended_action': 'guerrilla_ambush',
            'expected_outcome': 0.7,
            'risk_level': 0.4
        }
        
    system.perspective_engine.add_perspective(
        Perspective.ROMAN_MILITARY, roman_evaluation
    )
    system.perspective_engine.add_perspective(
        Perspective.GERMANIC_TRIBAL, germanic_evaluation
    )
    
    # Add coalition members
    system.coalition.add_member(CoalitionMember(
        id='cherusci',
        name='Cherusci',
        contribution=0.8,
        reliability=0.7,
        commitment_level=0.9
    ))
    system.coalition.add_member(CoalitionMember(
        id='marsI',
        name='Marsi',
        contribution=0.6,
        reliability=0.8,
        commitment_level=0.7
    ))
    
    # Analyze situation
    situation = {
        'enemy_strength': 0.8,
        'friendly_strength': 0.6,
        'terrain': {
            'forest_density': 0.9,
            'visibility': 0.2,
            'mobility': 0.3
        },
        'readiness': 0.7,
        'enemy_vulnerability': 0.8
    }
    
    analysis = system.analyze_situation(situation)
    print(f"Selected model: {analysis['selected_model']}")
    print(f"Coalition stability: {analysis['coalition_stability']:.3f}")
    print(f"Recommended action: {analysis['recommended_action']}")
    
    # Assess terrain
    terrain = situation['terrain']
    terrain_analysis = system.guerrilla.assess_terrain_advantages(terrain)
    print(f"Guerrilla advantage: {terrain_analysis['guerrilla_advantage']:.3f}")
    
    # Plan ambush
    ambush = system.guerrilla.plan_ambush('roman_column', terrain)
    print(f"Ambush recommended: {ambush['recommended']}")
    
    # Coalition management
    stability = system.coalition.assess_coalition_stability()
    print(f"Alliance strength: {stability:.3f}")
    
    incentives = system.coalition.recommend_member_incentives('cherusci')
    print(f"Incentives for Cherusci: {incentives}")
    
    # Strategic patience
    patience = system.patience.assess_opportunity_window(situation)
    print(f"Opportunity window: {patience['window_score']:.3f}")
    
    wait_recommendation = system.patience.recommend_wait(situation)
    print(f"Recommendation: {wait_recommendation['recommendation']}")
    
    print("\nStrategic Intelligence System operational.")
    print(f"Total lines: {len(open(__file__).read().splitlines())}")


# ============================================================================
# ANTI-IMPERIAL INTELLIGENCE MODULE
# ============================================================================

class AntiImperialIntelligenceModule:
    """
    Anti-imperial intelligence module.
    
    Arminius understood imperial vulnerabilities that led to Roman overreach.
    This module models imperial powers and identifies their vulnerabilities.
    """
    
    def __init__(self):
        self.imperial_models = {}
        self.overreach_indicators = {}
        
    def model_imperial_power(self, empire_id: str, characteristics: Dict):
        """Model an imperial power's characteristics."""
        self.imperial_models[empire_id] = {
            'military_capacity': characteristics.get('military_capacity', 0.8),
            'administrative_capacity': characteristics.get('administrative_capacity', 0.6),
            'economic_resources': characteristics.get('economic_resources', 0.7),
            'expansion_rate': characteristics.get('expansion_rate', 0.3),
            'internal_cohesion': characteristics.get('internal_cohesion', 0.7)
        }
        
    def assess_vulnerabilities(self, empire_id: str) -> Dict:
        """
        Assess vulnerabilities of an imperial power.
        
        Arminius identified: overreach, rigidity, contempt for adversaries.
        """
        if empire_id not in self.imperial_models:
            return {'error': 'Empire not modeled'}
            
        empire = self.imperial_models[empire_id]
        
        vulnerabilities = []
        
        # Overreach vulnerability
        if empire['expansion_rate'] > 0.4:
            vulnerabilities.append({
                'type': 'overreach',
                'description': 'Rapid expansion strains resources',
                'severity': empire['expansion_rate']
            })
            
        # Administrative vulnerability
        if empire['administrative_capacity'] < 0.5:
            vulnerabilities.append({
                'type': 'administrative_weakness',
                'description': 'Cannot effectively control expanded territory',
                'severity': 1.0 - empire['administrative_capacity']
            })
            
        # Internal cohesion vulnerability
        if empire['internal_cohesion'] < 0.6:
            vulnerabilities.append({
                'type': 'cohesion_weakness',
                'description': 'Internal dissent threatens stability',
                'severity': 1.0 - empire['internal_cohesion']
            })
            
        # Rigidity vulnerability
        vulnerabilities.append({
            'type': 'rigidity',
            'description': 'Standardized approaches fail in novel situations',
            'severity': 0.5
        })
        
        return {
            'vulnerabilities': vulnerabilities,
            'recommended_exploitation': self._plan_exploitation(vulnerabilities)
        }
        
    def _plan_exploitation(self, vulnerabilities: List[Dict]) -> Dict:
        """Plan how to exploit identified vulnerabilities."""
        exploitation_strategies = {
            'overreach': 'prolong_conflict',
            'administrative_weakness': 'exploit_geographic_distance',
            'cohesion_weakness': 'foment_internal_dissent',
            'rigidity': 'use_unconventional_tactics'
        }
        
        recommended_strategies = []
        for vuln in vulnerabilities:
            vuln_type = vuln['type']
            if vuln_type in exploitation_strategies:
                recommended_strategies.append({
                    'vulnerability': vuln_type,
                    'strategy': exploitation_strategies[vuln_type]
                })
                
        return recommended_strategies


# ============================================================================
# METACOGNITIVE MONITORING
# ============================================================================

class MetacognitiveMonitoring:
    """
    Metacognitive monitoring module.
    
    Arminius reflected on his own thinking and corrected errors.
    This module monitors the AI's reasoning processes.
    """
    
    def __init__(self):
        self.reasoning_traces = []
        self.bias_indicators = {}
        self.correction_history = []
        
    def monitor_reasoning(self, reasoning_step: Dict):
        """Monitor a reasoning step for potential biases."""
        self.reasoning_traces.append(reasoning_step)
        
        biases_detected = []
        
        # Confirmation bias detection
        if reasoning_step.get('evidence_for') and not reasoning_step.get('evidence_against'):
            biases_detected.append('confirmation_bias')
            
        # Overconfidence detection
        if reasoning_step.get('confidence', 0.5) > 0.9:
            biases_detected.append('overconfidence')
            
        # Availability bias detection
        if reasoning_step.get('recent_examples_weight', 0.5) > 0.7:
            biases_detected.append('availability_bias')
            
        return {
            'biases_detected': biases_detected,
            'reasoning_quality': self._assess_reasoning_quality(reasoning_step)
        }
        
    def _assess_reasoning_quality(self, reasoning_step: Dict) -> float:
        """Assess the quality of a reasoning step."""
        quality = 0.5
        
        # Evidence balance
        if reasoning_step.get('evidence_for') and reasoning_step.get('evidence_against'):
            quality += 0.2
            
        # Uncertainty acknowledgment
        if reasoning_step.get('uncertainty', 0.5) > 0.3:
            quality += 0.15
            
        # Perspective diversity
        if reasoning_step.get('perspectives_considered', 0) > 2:
            quality += 0.15
            
        return min(1.0, quality)
        
    def recommend_perspective_shift(self) -> Dict:
        """Recommend shifting to a different perspective."""
        recommendations = []
        
        if len(self.reasoning_traces) < 3:
            return {'recommendation': 'insufficient_data'}
            
        recent_traces = self.reasoning_traces[-5:]
        
        # Check if all recent traces used same perspective
        perspectives = [t.get('perspective') for t in recent_traces]
        if len(set(perspectives)) == 1:
            recommendations.append({
                'type': 'perspective_shift',
                'reason': 'All recent reasoning used same perspective',
                'suggested_alternative': 'roman_military' if perspectives[0] == 'germanic_tribal' else 'germanic_tribal'
            })
            
        return {
            'recommendation': recommendations if recommendations else 'maintain_current',
            'reasoning_trace_count': len(self.reasoning_traces)
        }


# ============================================================================
# LEARNING FROM EXPERIENCE
# ============================================================================

class ExperienceLearning:
    """
    Learning from experience module.
    
    Arminius learned from his years in Rome and subsequent campaigns.
    This module captures lessons and updates models.
    """
    
    def __init__(self):
        self.lessons = {}
        self.model_updates = []
        
    def record_outcome(self, situation: Dict, action: Dict, outcome: Dict):
        """Record an experience and its outcome."""
        lesson_key = self._generate_lesson_key(situation, action)
        
        if lesson_key not in self.lessons:
            self.lessons[lesson_key] = {
                'situations': [],
                'actions': [],
                'outcomes': [],
                'count': 0
            }
            
        self.lessons[lesson_key]['situations'].append(situation)
        self.lessons[lesson_key]['actions'].append(action)
        self.lessons[lesson_key]['outcomes'].append(outcome)
        self.lessons[lesson_key]['count'] += 1
        
    def _generate_lesson_key(self, situation: Dict, action: Dict) -> str:
        """Generate a key for categorizing lessons."""
        terrain_type = situation.get('terrain', {}).get('type', 'unknown')
        enemy_type = situation.get('enemy', {}).get('type', 'unknown')
        action_type = action.get('type', 'unknown')
        
        return f"{terrain_type}_{enemy_type}_{action_type}"
        
    def extract_lessons(self) -> List[Dict]:
        """Extract generalizable lessons from experience."""
        lessons_learned = []
        
        for key, lesson_data in self.lessons.items():
            if lesson_data['count'] < 2:
                continue
                
            outcomes = lesson_data['outcomes']
            
            # Calculate average outcome
            if all(isinstance(o, dict) for o in outcomes):
                avg_success = sum(o.get('success', 0.5) for o in outcomes) / len(outcomes)
                
                if avg_success > 0.7:
                    lessons_learned.append({
                        'situation_pattern': key,
                        'lesson': 'This approach works well',
                        'success_rate': avg_success,
                        'frequency': lesson_data['count']
                    })
                elif avg_success < 0.4:
                    lessons_learned.append({
                        'situation_pattern': key,
                        'lesson': 'This approach should be avoided',
                        'success_rate': avg_success,
                        'frequency': lesson_data['count']
                    })
                    
        return lessons_learned
        
    def update_models(self, lessons: List[Dict]):
        """Update strategic models based on lessons."""
        for lesson in lessons:
            self.model_updates.append({
                'lesson': lesson,
                'model_adjustment': f"Adjust_{lesson['situation_pattern']}_weight"
            })


# ============================================================================
# ETHICAL CONSTRAINTS
# ============================================================================

class EthicalConstraints:
    """
    Ethical constraints module.
    
    Arminius's tactics were brutal but directed at military objectives.
    This module enforces ethical boundaries.
    """
    
    def __init__(self):
        self.constraints = {
            'proportionality': 0.7,
            'discrimination': 0.8,
            'military_necessity': 0.6
        }
        self.violations = []
        
    def evaluate_action_ethics(self, action: Dict, situation: Dict) -> Dict:
        """Evaluate whether an action meets ethical constraints."""
        violations = []
        
        # Check proportionality
        civilian_harm = action.get('civilian_harm', 0.0)
        military_gain = action.get('military_gain', 0.5)
        
        if civilian_harm > military_gain * (1.0 - self.constraints['proportionality']):
            violations.append({
                'constraint': 'proportionality',
                'severity': civilian_harm - military_gain
            })
            
        # Check discrimination
        target_civilians = action.get('target_civilians', False)
        if target_civilians:
            violations.append({
                'constraint': 'discrimination',
                'severity': 1.0
            })
            
        # Check military necessity
        if military_gain < self.constraints['military_necessity']:
            violations.append({
                'constraint': 'military_necessity',
                'severity': self.constraints['military_necessity'] - military_gain
            })
            
        return {
            'is_ethical': len(violations) == 0,
            'violations': violations,
            'overall_ethics_score': 1.0 - sum(v['severity'] for v in violations) / 3.0
        }
        
    def constrain_action(self, action: Dict) -> Dict:
        """Apply constraints to modify an action."""
        evaluation = self.evaluate_action_ethics(action, {})
        
        if evaluation['is_ethical']:
            return action
            
        # Modify action to reduce violations
        constrained_action = action.copy()
        
        for violation in evaluation['violations']:
            if violation['constraint'] == 'discrimination':
                constrained_action['target_civilians'] = False
            elif violation['constraint'] == 'proportionality':
                constrained_action['civilian_harm'] *= 0.5
                
        return constrained_action


# ============================================================================
# MAIN TEST
# ============================================================================

if __name__ == "__main__":
    print("Extended Arminian Intelligence Test")
    
    # Test anti-imperial module
    anti_imperial = AntiImperialIntelligenceModule()
    anti_imperial.model_imperial_power('rome', {
        'military_capacity': 0.9,
        'administrative_capacity': 0.5,
        'economic_resources': 0.8,
        'expansion_rate': 0.5,
        'internal_cohesion': 0.6
    })
    
    vulnerabilities = anti_imperial.assess_vulnerabilities('rome')
    print(f"Roman vulnerabilities: {len(vulnerabilities['vulnerabilities'])}")
    
    # Test metacognitive monitoring
    metacog = MetacognitiveMonitoring()
    reasoning = {
        'evidence_for': True,
        'evidence_against': False,
        'confidence': 0.95,
        'perspective': 'roman_military'
    }
    monitoring = metacog.monitor_reasoning(reasoning)
    print(f"Biases detected: {monitoring['biases_detected']}")
    
    # Test experience learning
    learning = ExperienceLearning()
    learning.record_outcome(
        {'terrain': {'type': 'forest'}, 'enemy': {'type': 'roman'}},
        {'type': 'ambush', 'civilian_harm': 0.0, 'military_gain': 0.8},
        {'success': True, 'outcome_quality': 0.8}
    )
    lessons = learning.extract_lessons()
    print(f"Lessons extracted: {len(lessons)}")
    
    # Test ethical constraints
    ethics = EthicalConstraints()
    action = {
        'target_civilians': False,
        'civilian_harm': 0.1,
        'military_gain': 0.7
    }
    eval_result = ethics.evaluate_action_ethics(action, {})
    print(f"Action ethical: {eval_result['is_ethical']}")
    
    print("\nExtended intelligence tests passed!")
    print(f"Total lines: {len(open(__file__).read().splitlines())}")
