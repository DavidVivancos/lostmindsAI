#!/usr/bin/env python3
"""
1000Minds Book — Chapter 118: Heron of Alexandria
==================================================
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 6 Minds 101 - 120 Available on Amazon https://www.amazon.com/dp/B0HF7G6JJD
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 118: Heron of Alexandria (10 to 70 CE)
================================================================================
Heron of Alexandria (c. 10–70 CE): "The greatest experimentalist of antiquity."
Key inventions: Aeolipile (first steam turbine), vending machine, automata, hydraulic organs.

Philosophy of Mind:
- Mind as mechanism: cognition is a mechanical process
- Automata as cognitive models: complex behavior from simple mechanical parts
- Feedback as regulation: closed-loop control maintains stable behavior
- Sequencing: complex behavior = sequence of simpler operations
- Programmable behavior: stored programs determine behavior
- Embodiment: cognition requires physical body

This module implements the Heron Automaton Network (HAN) architecture:

1. AEOLIPILE DYNAMICS MODULE (AeolipileDynamics)
   - Rotational attractor dynamics — continuous rotation generates stable states
   - Thermal-inspired momentum: continuous rotation from steady energy input
   - Nozzle configuration → rotation direction/speed (attention steering)

2. PNEUMATIC CONTROL LAYER (PneumaticControlLayer)
   - Global broadcast of regulatory signals
   - Pressure-modulated processing thresholds
   - Fluid-like information propagation throughout network

3. GEAR-BASED PROCESSING HIERARCHY (GearBasedHierarchy)
   - Staged transformations: input → gear stages → output
   - Each gear stage: rotation+scaling+translation of representations
   - Gear engagement: learned attention over transformation stages

4. AUTOMATA SEQUENCER (AutomataSequencer)
   - Cam-drum program storage: sequence of control patterns
   - Multi-timescale: short/medium/long behavioral sequences
   - Re-camming: learning new sequences by modifying cam profiles

5. FEEDBACK REGULATION NETWORK (FeedbackRegulationNetwork)
   - Multi-scale error signals: local, regional, global
   - Prediction-error-driven weight updates
   - Homeostatic regulation maintaining stability

6. HYDRAULIC MEMORY SYSTEM (HydraulicMemory)
   - Pressure-state associative memory
   - Query by pressure pattern → retrieval at output sites
   - Hierarchical vessels: primary + secondary + tertiary storage

7. GEOMETRY AND SPACE PROCESSOR (GeometryProcessor)
   - Heron's geometric algorithms: area, volume, distance computation
   - Spatial reasoning with dioptra-inspired angular computation
   - Shape representation and transformation

8. EMBODIMENT INTERFACE (EmbodimentInterface)
   - Connects HAN to simulated or physical body
   - Sensorimotor coupling: perception-action loops
   - Environmental interaction feedback

9. PROGRAM SYNTHESIS MODULE (ProgramSynthesis)
   - Generates new sequences for Automata Sequencer
   - Combines existing subsequences into novel programs
   - Selection by performance: evolutionary pressure

Demonstration: HAN processes geometric patterns, stores sequences,
regulates its own behavior through feedback, and generates novel programs.

Author: 1000Minds AI Scholar
Topic: Heron of Alexandria, Automata, Mechanical Philosophy of Mind, Neural Architecture
"""

from __future__ import annotations

import math
import copy
import json
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Callable, List, Dict, Tuple, Optional, Any, Set
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from enum import Enum, auto


# =============================================================================
# SECTION 1: FOUNDATIONAL DATA STRUCTURES AND UTILITIES
# =============================================================================

class PRNG:
    """
    Deterministic pseudo-random number generator for reproducible experiments.
    Heron would have appreciated the precision of this: just as his automata
    produced consistent outputs from consistent inputs, our PRNG produces
    consistent random sequences from consistent seeds.
    """
    
    def __init__(self, seed: int = 120):
        self._state = seed
        self._original_seed = seed
    
    def random(self) -> float:
        """Returns a float in [0, 1)."""
        self._state = (self._state * 1103515245 + 12345) & 0x7fffffff
        return self._state / 0x7fffffff
    
    def uniform(self, low: float, high: float) -> float:
        """Returns a float in [low, high)."""
        return low + (high - low) * self.random()
    
    def randint(self, low: int, high: int) -> int:
        """Returns an int in [low, high] inclusive."""
        return int(low + (high - low + 1) * self.random())
    
    def choice(self, seq: List[Any]) -> Any:
        """Returns a random element from seq."""
        return seq[self.randint(0, len(seq) - 1)]
    
    def shuffle(self, seq: List[Any]) -> List[Any]:
        """Returns a shuffled copy of seq (Fisher-Yates)."""
        result = list(seq)
        for i in range(len(result) - 1, 0, -1):
            j = self.randint(0, i)
            result[i], result[j] = result[j], result[i]
        return result
    
    def gauss(self, mu: float = 0.0, sigma: float = 1.0) -> float:
        """Box-Muller transform for normally distributed random numbers."""
        u1 = self.random()
        while u1 == 0:
            u1 = self.random()
        u2 = self.random()
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        return mu + sigma * z
    
    def sample(self, population: List[Any], k: int) -> List[Any]:
        """Sample k unique elements from population (without replacement)."""
        pop = list(population)
        if k > len(pop):
            k = len(pop)
        result = []
        for _ in range(k):
            idx = self.randint(0, len(pop) - 1)
            result.append(pop[idx])
            pop.pop(idx)
        return result
    
    def reset(self) -> None:
        """Reset to original seed for reproducible experiments."""
        self._state = self._original_seed


_global_prng = PRNG(seed=120)


def set_global_seed(seed: int) -> PRNG:
    """Set the global PRNG seed and return the new PRNG."""
    global _global_prng
    _global_prng = PRNG(seed=seed)
    return _global_prng


def get_global_prng() -> PRNG:
    """Get the current global PRNG."""
    return _global_prng


def sigmoid(x: float) -> float:
    """Standard sigmoid activation function."""
    if x < -500:
        return 0.0
    if x > 500:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


def sigmoid_derivative(s: float) -> float:
    """Derivative of sigmoid given sigmoid output s."""
    return s * (1.0 - s)


def relu(x: float) -> float:
    """ReLU activation."""
    return max(0.0, x)


def relu_derivative(x: float) -> float:
    """Derivative of ReLU."""
    return 1.0 if x > 0.0 else 0.0


def tanh_activation(x: float) -> float:
    """Hyperbolic tangent activation."""
    if x < -20:
        return -1.0
    if x > 20:
        return 1.0
    e2x = math.exp(2.0 * x)
    return (e2x - 1.0) / (e2x + 1.0)


def softmax(inputs: List[float]) -> List[float]:
    """Softmax activation over a list of inputs."""
    if not inputs:
        return []
    max_inp = max(inputs)
    exps = [math.exp(x - max_inp) for x in inputs]
    sum_exps = sum(exps)
    return [e / sum_exps for e in exps]


def clip(x: float, low: float, high: float) -> float:
    """Clip a value to a range."""
    return max(low, min(high, x))


def euclidean_distance(a: List[float], b: List[float]) -> float:
    """Euclidean distance between two vectors."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x ** 2 for x in a))
    norm_b = math.sqrt(sum(y ** 2 for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def matmul(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """Matrix multiplication of two 2D lists."""
    if not A or not B:
        return []
    n = len(A)
    m = len(B[0]) if B else 0
    k = len(B)
    result = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            for p in range(k):
                result[i][j] += A[i][p] * B[p][j]
    return result


def vecadd(a: List[float], b: List[float]) -> List[float]:
    """Add two vectors."""
    return [x + y for x, y in zip(a, b)]


def vecsub(a: List[float], b: List[float]) -> List[float]:
    """Subtract two vectors."""
    return [x - y for x, y in zip(a, b)]


def vecscale(a: List[float], s: float) -> List[float]:
    """Scale a vector by a scalar."""
    return [x * s for x in a]


def vecnormalize(a: List[float]) -> List[float]:
    """Normalize a vector to unit length."""
    norm = math.sqrt(sum(x ** 2 for x in a))
    if norm == 0.0:
        return [0.0] * len(a)
    return [x / norm for x in a]


def hadamard(a: List[float], b: List[float]) -> List[float]:
    """Hadamard (element-wise) product of two vectors."""
    return [x * y for x, y in zip(a, b)]


def outer_product(a: List[float], b: List[float]) -> List[List[float]]:
    """Outer product of two vectors, resulting in a matrix."""
    return [[ai * bj for bj in b] for ai in a]


# =============================================================================
# SECTION 2: AEOLIPILE DYNAMICS MODULE
# =============================================================================

class AeolipileNode:
    """
    A single rotational dynamics node inspired by Heron's aeolipile.
    
    The aeolipile was a bronze sphere mounted on a central axis with two
    nozzles through which steam escaped, causing the sphere to rotate.
    The key insight: continuous energy input (steam) generates continuous
    rotational motion (attractor dynamics).
    
    In this implementation, each AeolipileNode maintains a continuous
    rotational state (phase angle) that evolves over time. The rotation
    is driven by an input signal (equivalent to steam pressure), and the
    rotation generates an output (equivalent to the rotational force).
    
    Multiple AeolipileNodes can be coupled to form rotational networks
    that produce complex attractor dynamics.
    """
    
    def __init__(
        self,
        node_id: int,
        input_dim: int = 8,
        output_dim: int = 8,
        friction: float = 0.1,
        inertia: float = 1.0,
        seed: int = 120
    ):
        self.node_id = node_id
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.friction = friction
        self.inertia = inertia
        
        self.prng = PRNG(seed=seed + node_id * 31)
        
        # Rotational state
        self.phase: float = self.prng.uniform(0.0, 2.0 * math.pi)
        self.angular_velocity: float = 0.0
        
        # Input weights (maps input to angular acceleration)
        self.W_input = [
            [self.prng.gauss(0.0, 0.1) for _ in range(input_dim)]
            for _ in range(1)
        ]
        
        # Output weights (maps angular position to output vector)
        self.W_output = [
            [self.prng.gauss(0.0, 0.1) for _ in range(output_dim)]
            for _ in range(8)
        ]
        
        # Nozzle configuration (determines direction and speed of rotation)
        self.nozzle_weights = [
            self.prng.gauss(0.0, 0.05) for _ in range(input_dim)
        ]
        
        self.nozzle_bias = self.prng.gauss(0.0, 0.01)
        
        self._cache: Optional[List[float]] = None
    
    def reset(self) -> None:
        """Reset the rotational state to initial conditions."""
        self.phase = self.prng.uniform(0.0, 2.0 * math.pi)
        self.angular_velocity = 0.0
    
    def set_phase(self, phase: float) -> None:
        """Set the phase angle directly."""
        self.phase = phase
    
    def step(self, input_vector: List[float], dt: float = 0.1) -> List[float]:
        """
        Advance the aeolipile by one timestep.
        
        Args:
            input_vector: Input signal (like steam pressure)
            dt: Timestep duration
            
        Returns:
            Output vector generated from current rotational state
        """
        if len(input_vector) != self.input_dim:
            raise ValueError(
                f"Expected input dim {self.input_dim}, got {len(input_vector)}"
            )
        
        # Compute effective nozzle pressure from input
        nozzle_pressure = sum(
            w * x for w, x in zip(self.nozzle_weights, input_vector)
        ) + self.nozzle_bias
        nozzle_pressure = tanh_activation(nozzle_pressure)
        
        # Compute angular acceleration (input drives rotation)
        angular_accel = nozzle_pressure * 2.0
        
        # Apply friction
        angular_accel -= self.friction * self.angular_velocity
        
        # Update angular velocity and position (Euler integration)
        self.angular_velocity += angular_accel * dt
        self.phase += self.angular_velocity * dt
        
        # Wrap phase to [0, 2π)
        self.phase = self.phase % (2.0 * math.pi)
        
        # Generate output from rotational state using sinusoidal basis functions
        # This is like the nozzles on the aeolipile creating directional outputs
        output = []
        for i in range(self.output_dim):
            freq = (i % 4) + 1
            phase_offset = (i * math.pi) / self.output_dim
            val = math.sin(freq * self.phase + phase_offset)
            output.append(val)
        
        # Apply output weight transformation
        weighted_output = []
        for j in range(self.output_dim):
            w_row = self.W_output[j % len(self.W_output)]
            val = sum(w * out for w, out in zip(w_row, output)) / len(w_row)
            weighted_output.append(val)
        
        self._cache = weighted_output
        return weighted_output
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current internal state of the node."""
        return {
            'node_id': self.node_id,
            'phase': self.phase,
            'angular_velocity': self.angular_velocity,
        }
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Set the internal state of the node."""
        if 'phase' in state:
            self.phase = state['phase']
        if 'angular_velocity' in state:
            self.angular_velocity = state['angular_velocity']


class AeolipileDynamicsLayer:
    """
    A layer of AeolipileNodes forming a rotational dynamics network.
    
    This layer implements continuous rotational dynamics across a population
    of nodes. Each node maintains its own rotational state, and nodes can
    be coupled to each other through learned coupling weights.
    
    The layer acts as an attractor network: inputs push the system into
    different attractor states, and the system maintains those states
    (like the aeolipile continuing to rotate after the fire is lit).
    
    Heron's insight: the aeolipile converts steady input (fire → steam)
    into continuous rotational motion. This layer converts steady input
    signals into continuous attractor dynamics.
    """
    
    def __init__(
        self,
        num_nodes: int = 16,
        input_dim: int = 8,
        output_dim: int = 8,
        coupling_strength: float = 0.05,
        seed: int = 120
    ):
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.coupling_strength = coupling_strength
        
        self.prng = PRNG(seed=seed)
        
        # Create nodes
        self.nodes: List[AeolipileNode] = []
        for i in range(num_nodes):
            node = AeolipileNode(
                node_id=i,
                input_dim=input_dim,
                output_dim=output_dim,
                seed=seed + i * 17
            )
            self.nodes.append(node)
        
        # Coupling weights between nodes
        self.coupling_weights = [
            [self.prng.gauss(0.0, coupling_strength)
             for _ in range(num_nodes)]
            for _ in range(num_nodes)
        ]
        
        self._cache: Optional[List[List[float]]] = None
    
    def reset(self) -> None:
        """Reset all nodes to initial states."""
        for node in self.nodes:
            node.reset()
    
    def step(self, input_vector: List[float]) -> List[List[float]]:
        """
        Advance all nodes in the layer by one timestep.
        
        Args:
            input_vector: Global input to the layer
            
        Returns:
            List of output vectors, one per node
        """
        # Compute coupling influences
        all_phases = [node.phase for node in self.nodes]
        all_velocities = [node.angular_velocity for node in self.nodes]
        
        outputs = []
        for i, node in enumerate(self.nodes):
            # Compute coupling from other nodes (via phase differences)
            coupling = 0.0
            for j, other_node in enumerate(self.nodes):
                if i != j:
                    phase_diff = math.sin(all_phases[j] - all_phases[i])
                    coupling += self.coupling_weights[i][j] * phase_diff
            
            # Combine direct input with coupling
            augmented_input = list(input_vector)
            if augmented_input:
                coupling_signal = [coupling] * min(len(augmented_input), 1)
                augmented_input[0] += coupling_signal[0] if augmented_input else coupling
            
            output = node.step(input_vector if augmented_input == [coupling] else augmented_input)
            outputs.append(output)
        
        self._cache = outputs
        return outputs
    
    def get_attractor_state(self) -> List[float]:
        """Get the aggregate attractor state (mean phase and velocity)."""
        if not self.nodes:
            return []
        mean_phase = sum(n.phase for n in self.nodes) / len(self.nodes)
        mean_velocity = sum(n.angular_velocity for n in self.nodes) / len(self.nodes)
        return [math.sin(mean_phase), math.cos(mean_phase), mean_velocity]
    
    def apply_coupling_matrix(
        self,
        coupling_matrix: List[List[float]]
    ) -> None:
        """Update the coupling weight matrix."""
        if len(coupling_matrix) == len(self.coupling_weights):
            for i in range(len(coupling_matrix)):
                if len(coupling_matrix[i]) == len(self.coupling_weights[0]):
                    self.coupling_weights[i] = list(coupling_matrix[i])


# =============================================================================
# SECTION 3: PNEUMATIC CONTROL LAYER
# =============================================================================

class PneumaticControlLayer:
    """
    Global broadcast layer using pneumatic (fluid pressure) principles.
    
    Heron's pneumatic devices used pressurized air to transmit signals
    throughout a machine — the pressure applied at one point was felt
    everywhere in the system. This layer implements a similar principle:
    a global regulatory signal that modulates processing throughout the
    entire network.
    
    Key properties:
    - Broadcast: regulatory signal reaches all processing units
    - Pressure semantics: signal strength determines modulation intensity
    - Propagation: gradual transmission through the network (not instant)
    - Compression: signals can be compressed or amplified
    
    In neural network terms, this layer implements something like
    a global attention or modulation mechanism, similar to how
    acetylcholine or norepinephrine act as neuromodulators in the brain.
    """
    
    def __init__(
        self,
        num_sources: int = 8,
        num_targets: int = 64,
        pressure_decay: float = 0.1,
        seed: int = 120
    ):
        self.num_sources = num_sources
        self.num_targets = num_targets
        self.pressure_decay = pressure_decay
        
        self.prng = PRNG(seed=seed)
        
        # Source nodes (generate regulatory signals)
        self.sources = [
            RegulatorySource(source_id=i, seed=seed + i * 13)
            for i in range(num_sources)
        ]
        
        # Propagation weights: how signals spread from sources to targets
        self.propagation_weights = [
            [self.prng.gauss(0.0, 0.1) for _ in range(num_targets)]
            for _ in range(num_sources)
        ]
        
        # Target thresholds (modulated by incoming pressure)
        self.target_thresholds = [
            0.5 for _ in range(num_targets)
        ]
        
        # Current pressure state at each target
        self.target_pressures = [0.0] * num_targets
        
        # History of pressure states
        self.pressure_history: List[List[float]] = []
    
    def generate_regulatory_signals(
        self,
        global_signal: List[float],
        target_activity: List[float]
    ) -> List[float]:
        """
        Generate regulatory signals from global inputs and target states.
        
        Args:
            global_signal: Global input signal (e.g., from environment)
            target_activity: Current activity levels at targets
            
        Returns:
            Regulatory signal strengths for each target
        """
        if len(global_signal) < self.num_sources:
            padded = list(global_signal) + [0.0] * (self.num_sources - len(global_signal))
        else:
            padded = list(global_signal[:self.num_sources])
        
        # Update source nodes
        for i, source in enumerate(self.sources):
            source.update(padded[i])
        
        # Compute pressure at each target
        regulatory_signals = []
        for t in range(self.num_targets):
            pressure = 0.0
            for s, source in enumerate(self.sources):
                source_signal = source.get_signal()
                weight = self.propagation_weights[s][t]
                pressure += source_signal * weight
            
            # Apply threshold modulation
            threshold = self.target_thresholds[t]
            if pressure > threshold:
                # Above threshold: amplify
                modulation = 1.0 + (pressure - threshold)
            else:
                # Below threshold: attenuate
                modulation = pressure / (threshold + 1e-6)
            
            regulatory_signals.append(clip(modulation, 0.0, 5.0))
            self.target_pressures[t] = regulatory_signals[t]
        
        self.pressure_history.append(list(self.target_pressures))
        return regulatory_signals
    
    def modulate(
        self,
        target_values: List[float],
        regulatory_signals: List[float]
    ) -> List[float]:
        """
        Apply regulatory modulation to target values.
        
        Args:
            target_values: Values to be modulated
            regulatory_signals: Regulatory signal strengths
            
        Returns:
            Modulated values
        """
        if len(target_values) != len(regulatory_signals):
            raise ValueError(
                f"Dimension mismatch: {len(target_values)} vs {len(regulatory_signals)}"
            )
        
        modulated = []
        for val, pressure in zip(target_values, regulatory_signals):
            # Higher pressure amplifies and expands range
            modulated_val = val * pressure
            modulated.append(modulated_val)
        
        return modulated
    
    def step(
        self,
        global_signal: List[float],
        target_activity: List[float],
        target_values: List[float]
    ) -> List[float]:
        """
        Full pneumatic control step: generate signals and apply modulation.
        
        Args:
            global_signal: Global environmental signal
            target_activity: Current activity at targets
            target_values: Values to modulate
            
        Returns:
            Modulated values
        """
        signals = self.generate_regulatory_signals(global_signal, target_activity)
        return self.modulate(target_values, signals)


class RegulatorySource:
    """
    A single source of regulatory (pressure) signals.
    
    Each source generates a continuous pressure signal that varies
    over time based on input. Multiple sources can be active
    simultaneously, generating complex regulatory patterns.
    """
    
    def __init__(self, source_id: int, seed: int = 120):
        self.source_id = source_id
        self.prng = PRNG(seed=seed)
        self.current_signal: float = 0.0
        self.signal_momentum: float = 0.0
        self.base_level: float = self.prng.uniform(0.3, 0.7)
        self.response_rate: float = self.prng.uniform(0.1, 0.3)
    
    def update(self, input_val: float) -> None:
        """Update the signal based on input."""
        target = self.base_level + self.response_rate * input_val
        self.signal_momentum = 0.7 * self.signal_momentum + 0.3 * (target - self.current_signal)
        self.current_signal += self.signal_momentum
        self.current_signal = clip(self.current_signal, 0.0, 2.0)
    
    def get_signal(self) -> float:
        """Get the current signal value."""
        return self.current_signal


# =============================================================================
# SECTION 4: GEAR-BASED PROCESSING HIERARCHY
# =============================================================================

class GearStage:
    """
    A single gear stage in a gear-based processing hierarchy.
    
    Heron's gear systems transformed motion through a series of stages:
    a gear wheel engaging with a pinion, which engages with another gear,
    which engages with a rack, and so on. Each stage performs a specific
    transformation: rotation direction can be reversed, speed can be
    increased or decreased, and force can be amplified or reduced.
    
    In this implementation, a GearStage performs a parameterized
    affine transformation: output = rotation * weight * input + bias,
    where rotation can be +1 or -1 (direction reversal), weight
    controls speed/scaling, and bias controls offset.
    
    The gear ratio (weight) and direction (sign) are learnable parameters.
    """
    
    def __init__(
        self,
        stage_id: int,
        input_dim: int,
        output_dim: int,
        gear_ratio: Optional[float] = None,
        direction: Optional[int] = None,
        seed: int = 120
    ):
        self.stage_id = stage_id
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.prng = PRNG(seed=seed + stage_id * 23)
        
        # Gear parameters
        if gear_ratio is None:
            self.gear_ratio = self.prng.uniform(0.5, 2.0)
        else:
            self.gear_ratio = gear_ratio
        
        if direction is None:
            self.direction = 1 if self.prng.random() > 0.5 else -1
        else:
            self.direction = direction
        
        # Transformation weights
        self.weights = [
            [self.prng.gauss(0.0, 0.1) for _ in range(input_dim)]
            for _ in range(output_dim)
        ]
        
        # Bias (offset)
        self.bias = [self.prng.gauss(0.0, 0.01) for _ in range(output_dim)]
        
        # Accumulated statistics for learning
        self.input_sum = [0.0] * input_dim
        self.output_sum = [0.0] * output_dim
        self.step_count = 0
    
    def transform(self, input_vector: List[float]) -> List[float]:
        """
        Apply the gear transformation to an input vector.
        
        Args:
            input_vector: Input data to transform
            
        Returns:
            Transformed output vector
        """
        if len(input_vector) != self.input_dim:
            raise ValueError(
                f"Expected input dim {self.input_dim}, got {len(input_vector)}"
            )
        
        # Compute weighted sum
        raw_output = []
        for i in range(self.output_dim):
            total = sum(
                self.weights[i][j] * input_vector[j]
                for j in range(self.input_dim)
            )
            # Apply gear ratio scaling and direction reversal
            scaled = self.gear_ratio * self.direction * total
            # Apply bias
            biased = scaled + self.bias[i]
            # Activation function
            activated = tanh_activation(biased)
            raw_output.append(activated)
        
        # Update statistics
        self.input_sum = vecadd(self.input_sum, input_vector)
        self.output_sum = vecadd(self.output_sum, raw_output)
        self.step_count += 1
        
        return raw_output
    
    def get_average_input(self) -> List[float]:
        """Get the average input seen by this stage."""
        if self.step_count == 0:
            return [0.0] * self.input_dim
        return [s / self.step_count for s in self.input_sum]
    
    def get_average_output(self) -> List[float]:
        """Get the average output produced by this stage."""
        if self.step_count == 0:
            return [0.0] * self.output_dim
        return [s / self.step_count for s in self.output_sum]
    
    def update_weights(
        self,
        delta_weights: List[List[float]],
        delta_bias: List[float],
        learning_rate: float = 0.01
    ) -> None:
        """Update weights based on computed deltas."""
        for i in range(self.output_dim):
            for j in range(self.input_dim):
                self.weights[i][j] += learning_rate * delta_weights[i][j]
            self.bias[i] += learning_rate * delta_bias[i]


class GearBasedHierarchy:
    """
    A deep hierarchical network organized as a series of gear stages.
    
    Heron's gear trains processed motion through a sequence of stages,
    with each stage performing a specific transformation. This hierarchy
    implements a similar architecture: data flows through a sequence of
    GearStages, each performing a different transformation on the
    representation.
    
    The key innovation of this architecture is the "gear engagement"
    mechanism: instead of passing through every stage, data can be
    routed through selected stages, allowing the network to attend to
    different levels of the hierarchy.
    
    Attributes:
        num_stages: Number of gear stages in the hierarchy
        stage_dims: Dimensions of each stage's input/output
        stages: The actual GearStage instances
        engagement_weights: How much each stage is engaged
    """
    
    def __init__(
        self,
        input_dim: int = 8,
        hidden_dims: List[int] = None,
        output_dim: int = 8,
        num_stages: int = 4,
        seed: int = 120
    ):
        if hidden_dims is None:
            hidden_dims = [16, 16, 16]
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.num_stages = num_stages
        
        self.prng = PRNG(seed=seed)
        
        # Build stages
        self.stages: List[GearStage] = []
        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            stage = GearStage(
                stage_id=i,
                input_dim=prev_dim,
                output_dim=hidden_dim,
                seed=seed + i * 37
            )
            self.stages.append(stage)
            prev_dim = hidden_dim
        
        # Final output stage
        final_stage = GearStage(
            stage_id=len(hidden_dims),
            input_dim=prev_dim,
            output_dim=output_dim,
            seed=seed + len(hidden_dims) * 37
        )
        self.stages.append(final_stage)
        
        # Engagement weights (attention over stages)
        self.engagement_weights = [
            self.prng.uniform(0.0, 1.0) for _ in range(num_stages + 1)
        ]
        self._normalize_engagement()
    
    def _normalize_engagement(self) -> None:
        """Normalize engagement weights to sum to 1."""
        total = sum(self.engagement_weights)
        if total > 0:
            self.engagement_weights = [w / total for w in self.engagement_weights]
    
    def forward(
        self,
        input_vector: List[float],
        engagement: Optional[List[float]] = None
    ) -> List[float]:
        """
        Forward pass through the gear hierarchy.
        
        Args:
            input_vector: Input data
            engagement: Optional engagement weights (default: use stored)
            
        Returns:
            Transformed output vector
        """
        current = list(input_vector)
        
        if engagement is not None:
            eng = list(engagement)
        else:
            eng = list(self.engagement_weights)
        
        # Ensure engagement matches number of stages
        while len(eng) < len(self.stages):
            eng.append(0.0)
        eng = eng[:len(self.stages)]
        
        outputs = []
        for i, stage in enumerate(self.stages):
            transformed = stage.transform(current)
            outputs.append(transformed)
        
        # Weighted combination of stage outputs (engagement = attention)
        self._normalize_engagement()
        combined = outputs[0]
        for i in range(1, len(outputs)):
            combined = vecadd(
                combined,
                [c * self.engagement_weights[i] for c in outputs[i]]
            )
        
        return combined
    
    def full_forward(self, input_vector: List[float]) -> Tuple[List[float], List[List[float]]]:
        """
        Full forward pass returning all intermediate outputs.
        
        Returns:
            Tuple of (final_output, list_of_intermediate_outputs)
        """
        current = list(input_vector)
        all_outputs = [current]
        
        for stage in self.stages:
            current = stage.transform(current)
            all_outputs.append(current)
        
        return current, all_outputs[1:]
    
    def update_engagement(
        self,
        relevance_scores: List[float]
    ) -> None:
        """Update engagement weights based on relevance scores."""
        if len(relevance_scores) != len(self.stages):
            raise ValueError(
                f"Expected {len(self.stages)} scores, got {len(relevance_scores)}"
            )
        
        for i in range(len(self.stages)):
            self.engagement_weights[i] = max(0.0, self.engagement_weights[i] + 0.1 * relevance_scores[i])
        
        self._normalize_engagement()
    
    def get_stage_states(self) -> List[Dict[str, Any]]:
        """Get the state of each stage for inspection."""
        return [
            {
                'stage_id': s.stage_id,
                'gear_ratio': s.gear_ratio,
                'direction': s.direction,
                'avg_input': s.get_average_input()[:3],
                'avg_output': s.get_average_output()[:3],
            }
            for s in self.stages
        ]


# =============================================================================
# SECTION 5: AUTOMATA SEQUENCER (CAM-DRUM PROGRAM STORAGE)
# =============================================================================

class CamProfile:
    """
    A single cam profile on a cam-drum, determining one step in a sequence.
    
    Heron's automata used cam drums: rotating cylinders with pegs or
    shoulders that engaged with levers to produce complex sequences
    of motions. Each cam profile on the drum determined one step
    in the sequence.
    
    A CamProfile stores the control pattern for one step: which
    modules should be active, what their parameters should be,
    and how long this step should last.
    """
    
    def __init__(
        self,
        cam_id: int,
        control_pattern: Optional[Dict[int, float]] = None,
        duration: float = 1.0,
        seed: int = 120
    ):
        self.cam_id = cam_id
        self.duration = duration
        
        self.prng = PRNG(seed=seed + cam_id * 41)
        
        # Control pattern: module_id -> activation level
        if control_pattern is None:
            self.control_pattern: Dict[int, float] = {}
        else:
            self.control_pattern = dict(control_pattern)
    
    def set_control(self, module_id: int, level: float) -> None:
        """Set the control level for a specific module."""
        self.control_pattern[module_id] = clip(level, 0.0, 1.0)
    
    def get_control(self, module_id: int) -> float:
        """Get the control level for a module, default 0.0."""
        return self.control_pattern.get(module_id, 0.0)
    
    def mutate(
        self,
        mutation_rate: float = 0.1,
        noise_scale: float = 0.1
    ) -> 'CamProfile':
        """Create a mutated copy of this cam profile."""
        new_pattern = {
            k: clip(v + self.prng.gauss(0.0, noise_scale), 0.0, 1.0)
            for k, v in self.control_pattern.items()
        }
        # Possibly add new module controls
        if self.prng.random() < mutation_rate:
            new_module = self.prng.randint(0, 16)
            new_pattern[new_module] = self.prng.uniform(0.0, 1.0)
        
        new_duration = self.duration + self.prng.gauss(0.0, 0.1)
        new_duration = max(0.1, new_duration)
        
        return CamProfile(
            cam_id=self.cam_id + 1000,
            control_pattern=new_pattern,
            duration=new_duration,
            seed=self.prng.randint(0, 10000)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'cam_id': self.cam_id,
            'duration': self.duration,
            'control_pattern': self.control_pattern,
        }


class CamDrum:
    """
    A cam drum storing a sequence of control programs.
    
    Heron's automata used rotating drums with multiple cam profiles
    arranged around their circumference. As the drum rotated, each
    cam engaged with its corresponding lever in sequence, producing
    a choreographed sequence of actions.
    
    This CamDrum stores an ordered sequence of CamProfiles, and
    supports operations for reading the current profile, advancing
    the drum, and modifying the sequence.
    """
    
    def __init__(
        self,
        drum_id: int,
        capacity: int = 16,
        seed: int = 120
    ):
        self.drum_id = drum_id
        self.capacity = capacity
        
        self.prng = PRNG(seed=seed)
        
        # Cam profiles in sequence
        self.cams: List[CamProfile] = []
        
        # Current position on the drum (which cam is active)
        self.position: int = 0
        
        # Current phase within the active cam (0.0 to 1.0)
        self.phase: float = 0.0
    
    def load_sequence(self, cams: List[CamProfile]) -> None:
        """Load a sequence of cam profiles onto the drum."""
        self.cams = list(cams[:self.capacity])
        self.position = 0
        self.phase = 0.0
    
    def generate_random_sequence(
        self,
        num_cams: int,
        num_modules: int = 8
    ) -> None:
        """Generate a random cam sequence."""
        self.cams = []
        for i in range(min(num_cams, self.capacity)):
            pattern = {
                m: self.prng.uniform(0.0, 1.0)
                for m in range(num_modules)
                if self.prng.random() > 0.3
            }
            duration = self.prng.uniform(0.5, 2.0)
            cam = CamProfile(cam_id=i, control_pattern=pattern, duration=duration)
            self.cams.append(cam)
        self.position = 0
        self.phase = 0.0
    
    def get_current_cam(self) -> Optional[CamProfile]:
        """Get the currently active cam profile."""
        if not self.cams:
            return None
        return self.cams[self.position % len(self.cams)]
    
    def advance(self, dt: float) -> None:
        """
        Advance the drum by dt time units.
        
        Args:
            dt: Time step
        """
        if not self.cams:
            return
        
        current_cam = self.get_current_cam()
        if current_cam is None:
            return
        
        self.phase += dt / current_cam.duration
        
        if self.phase >= 1.0:
            self.phase = 0.0
            self.position = (self.position + 1) % len(self.cams)
    
    def get_current_controls(self) -> Dict[int, float]:
        """Get the control values for all modules from the current cam."""
        cam = self.get_current_cam()
        if cam is None:
            return {}
        
        # Apply phase to modulate control values
        controls = {}
        for module_id, level in cam.control_pattern.items():
            # Smooth modulation using sin at transition points
            if self.phase < 0.1:
                fade = math.sin(self.phase / 0.1 * math.pi / 2)
            elif self.phase > 0.9:
                fade = math.sin((1.0 - self.phase) / 0.1 * math.pi / 2)
            else:
                fade = 1.0
            controls[module_id] = level * fade
        
        return controls
    
    def re_cam(
        self,
        cam_index: int,
        new_pattern: Dict[int, float]
    ) -> bool:
        """
        Re-cam the drum: modify a cam profile to implement new behavior.
        
        Args:
            cam_index: Which cam to modify
            new_pattern: New control pattern for this cam
            
        Returns:
            True if successful, False if index out of range
        """
        if 0 <= cam_index < len(self.cams):
            old_cam = self.cams[cam_index]
            self.cams[cam_index] = CamProfile(
                cam_id=old_cam.cam_id,
                control_pattern=new_pattern,
                duration=old_cam.duration,
                seed=old_cam.cam_id * 777
            )
            return True
        return False
    
    def crossover(self, other: 'CamDrum', crossover_point: int) -> 'CamDrum':
        """
        Create a new drum by crossing over with another drum at a point.
        
        Args:
            other: Another CamDrum to crossover with
            crossover_point: Index at which to swap
            
        Returns:
            New CamDrum with combined sequence
        """
        new_drum = CamDrum(
            drum_id=self.drum_id * 100 + other.drum_id,
            capacity=max(self.capacity, other.capacity)
        )
        
        combined = list(self.cams)
        for i, cam in enumerate(other.cams):
            if i >= len(combined):
                combined.append(cam)
            elif i >= crossover_point:
                combined[i] = cam
        
        new_drum.load_sequence(combined[:new_drum.capacity])
        return new_drum
    
    def __len__(self) -> int:
        """Number of cams currently loaded."""
        return len(self.cams)


class AutomataSequencer:
    """
    Program storage and execution system based on Heron's automata.
    
    This sequencer stores behavioral programs as sequences of cam
    profiles on drum, and executes them by advancing through the
    sequence over time. It supports:
    - Multiple timescales: short, medium, and long sequences
    - Program modification: re-camming to learn new behaviors
    - Program combination: crossover to create novel programs
    - Hierarchical sequencing: sequences that call sub-sequences
    
    This is the Heronian equivalent of a stored-program computer.
    """
    
    def __init__(
        self,
        num_modules: int = 8,
        short_capacity: int = 8,
        medium_capacity: int = 16,
        long_capacity: int = 32,
        seed: int = 120
    ):
        self.num_modules = num_modules
        self.short_capacity = short_capacity
        self.medium_capacity = medium_capacity
        self.long_capacity = long_capacity
        
        self.prng = PRNG(seed=seed)
        
        # Three levels of drum (short/medium/long sequences)
        self.short_drum = CamDrum(drum_id=1, capacity=short_capacity, seed=seed)
        self.medium_drum = CamDrum(drum_id=2, capacity=medium_capacity, seed=seed + 100)
        self.long_drum = CamDrum(drum_id=3, capacity=long_capacity, seed=seed + 200)
        
        # Current timescale being executed
        self.current_timescale: str = 'short'
        
        # Execution state
        self.current_time: float = 0.0
        self.tick_count: int = 0
    
    def initialize(self) -> None:
        """Initialize with random sequences at all timescales."""
        self.short_drum.generate_random_sequence(self.short_capacity, self.num_modules)
        self.medium_drum.generate_random_sequence(self.medium_capacity, self.num_modules)
        self.long_drum.generate_random_sequence(self.long_capacity, self.num_modules)
    
    def step(self, dt: float = 1.0) -> Dict[int, float]:
        """
        Advance the sequencer by one timestep.
        
        Args:
            dt: Time step
            
        Returns:
            Current control signals for all modules
        """
        self.current_time += dt
        self.tick_count += 1
        
        # Advance drums
        self.short_drum.advance(dt)
        
        if self.tick_count % 4 == 0:
            self.medium_drum.advance(dt)
        
        if self.tick_count % 16 == 0:
            self.long_drum.advance(dt)
        
        # Get current controls from the current timescale
        controls = self._get_current_controls()
        
        return controls
    
    def _get_current_controls(self) -> Dict[int, float]:
        """Get the current control signals from all active drums."""
        controls = {}
        
        # Add short drum controls (always active)
        short_controls = self.short_drum.get_current_controls()
        for module_id, level in short_controls.items():
            controls[module_id] = controls.get(module_id, 0.0) + 0.6 * level
        
        # Add medium drum controls
        medium_controls = self.medium_drum.get_current_controls()
        for module_id, level in medium_controls.items():
            controls[module_id] = controls.get(module_id, 0.0) + 0.3 * level
        
        # Add long drum controls
        long_controls = self.long_drum.get_current_controls()
        for module_id, level in long_controls.items():
            controls[module_id] = controls.get(module_id, 0.0) + 0.1 * level
        
        # Clip to valid range
        return {k: clip(v, 0.0, 1.0) for k, v in controls.items()}
    
    def get_control_for_module(self, module_id: int) -> float:
        """Get the current control signal for a specific module."""
        controls = self._get_current_controls()
        return controls.get(module_id, 0.0)
    
    def learn_sequence(
        self,
        sequence_id: str,
        cam_sequence: List[CamProfile]
    ) -> bool:
        """
        Learn a new sequence by loading it onto the appropriate drum.
        
        Args:
            sequence_id: 'short', 'medium', or 'long'
            cam_sequence: Sequence of cam profiles to learn
            
        Returns:
            True if successful
        """
        if sequence_id == 'short':
            drum = self.short_drum
        elif sequence_id == 'medium':
            drum = self.medium_drum
        elif sequence_id == 'long':
            drum = self.long_drum
        else:
            return False
        
        drum.load_sequence(cam_sequence)
        return True
    
    def mutate_current_sequence(
        self,
        sequence_id: str,
        mutation_rate: float = 0.1
    ) -> bool:
        """Mutate the current sequence on a drum."""
        if sequence_id == 'short':
            drum = self.short_drum
        elif sequence_id == 'medium':
            drum = self.medium_drum
        elif sequence_id == 'long':
            drum = self.long_drum
        else:
            return False
        
        if not drum.cams:
            return False
        
        # Mutate a random cam
        cam_index = self.prng.randint(0, len(drum.cams) - 1)
        mutated = drum.cams[cam_index].mutate(mutation_rate)
        new_sequence = list(drum.cams)
        new_sequence[cam_index] = mutated
        drum.load_sequence(new_sequence)
        return True
    
    def evolve_sequence(
        self,
        other: 'AutomataSequencer',
        sequence_id: str,
        crossover_point: int
    ) -> bool:
        """Evolve by crossing over with another sequencer's sequence."""
        if sequence_id not in ('short', 'medium', 'long'):
            return False
        
        my_drum = getattr(self, f'{sequence_id}_drum')
        other_drum = getattr(other, f'{sequence_id}_drum')
        
        new_drum = my_drum.crossover(other_drum, crossover_point)
        setattr(self, f'{sequence_id}_drum', new_drum)
        return True
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the sequencer."""
        return {
            'timescale': self.current_timescale,
            'time': self.current_time,
            'tick': self.tick_count,
            'short_pos': self.short_drum.position,
            'medium_pos': self.medium_drum.position,
            'long_pos': self.long_drum.position,
            'short_len': len(self.short_drum),
            'medium_len': len(self.medium_drum),
            'long_len': len(self.long_drum),
        }


# =============================================================================
# SECTION 6: FEEDBACK REGULATION NETWORK
# =============================================================================

class FeedbackLoop:
    """
    A single feedback loop comparing predictions with actual outputs.
    
    Heron's devices used feedback constantly: the vending machine
    compared the coin's weight with the expected weight, the
    fountain compared water flow with desired flow. This loop
    implements a generic prediction-error feedback mechanism.
    """
    
    def __init__(
        self,
        loop_id: int,
        target_dim: int = 8,
        learning_rate: float = 0.01,
        adaptation_rate: float = 0.1,
        seed: int = 120
    ):
        self.loop_id = loop_id
        self.target_dim = target_dim
        self.learning_rate = learning_rate
        self.adaptation_rate = adaptation_rate
        
        self.prng = PRNG(seed=seed)
        
        # Predictive model: predicts next state from current state
        self.predictor_weights = [
            [self.prng.gauss(0.0, 0.01) for _ in range(target_dim)]
            for _ in range(target_dim)
        ]
        self.predictor_bias = [0.0] * target_dim
        
        # Current predicted state
        self.predicted_state: List[float] = [0.0] * target_dim
    
    def predict(self, current_state: List[float]) -> List[float]:
        """
        Generate prediction of next state from current state.
        
        Args:
            current_state: Current observation
            
        Returns:
            Predicted next state
        """
        if len(current_state) != self.target_dim:
            raise ValueError(f"Expected dim {self.target_dim}, got {len(current_state)}")
        
        predicted = []
        for i in range(self.target_dim):
            pred = self.predictor_bias[i]
            for j in range(self.target_dim):
                pred += self.predictor_weights[i][j] * current_state[j]
            predicted.append(tanh_activation(pred))
        
        self.predicted_state = list(predicted)
        return predicted
    
    def compute_error(
        self,
        current_state: List[float],
        next_state: List[float]
    ) -> Tuple[List[float], float]:
        """
        Compute prediction error and update predictor.
        
        Args:
            current_state: State at time t
            next_state: State at time t+1 (actual)
            
        Returns:
            Tuple of (error_vector, scalar_error)
        """
        prediction = self.predict(current_state)
        
        # Prediction error
        error = vecsub(next_state, prediction)
        scalar_error = sum(e ** 2 for e in error) / len(error)
        
        # Update predictor weights (Hebbian + error-driven)
        for i in range(self.target_dim):
            for j in range(self.target_dim):
                # Hebbian update based on correlation
                hebbian = self.learning_rate * error[i] * current_state[j]
                self.predictor_weights[i][j] += hebbian
        
        # Update bias
        for i in range(self.target_dim):
            self.predictor_bias[i] += self.learning_rate * error[i]
        
        return error, scalar_error
    
    def get_prediction(self) -> List[float]:
        """Get the current prediction without computing error."""
        return list(self.predicted_state)


class FeedbackRegulationNetwork:
    """
    Multi-scale feedback regulation network inspired by Heron's control systems.
    
    Heron's machines used feedback at multiple scales: low-level feedback
    regulated individual mechanisms, mid-level feedback coordinated groups
    of mechanisms, and high-level feedback compared overall behavior with
    goals. This network implements a similar multi-scale architecture.
    
    The network has three levels:
    - Local feedback: individual units regulate themselves
    - Regional feedback: groups of units regulate each other
    - Global feedback: whole system regulated by comparison with goals
    
    The feedback signals drive learning throughout the system, enabling
    the network to adapt and improve its predictions and behaviors.
    """
    
    def __init__(
        self,
        num_local_units: int = 16,
        num_regions: int = 4,
        goal_dim: int = 8,
        learning_rate: float = 0.01,
        seed: int = 120
    ):
        self.num_local_units = num_local_units
        self.num_regions = num_regions
        self.goal_dim = goal_dim
        self.learning_rate = learning_rate
        
        self.prng = PRNG(seed=seed)
        
        # Local feedback loops (one per unit group)
        units_per_region = max(1, num_local_units // num_regions)
        self.local_loops: List[FeedbackLoop] = []
        for i in range(num_regions):
            loop = FeedbackLoop(
                loop_id=i,
                target_dim=units_per_region,
                learning_rate=learning_rate,
                seed=seed + i * 19
            )
            self.local_loops.append(loop)
        
        # Regional feedback loops (one per region)
        self.regional_loops: List[FeedbackLoop] = []
        for i in range(num_regions):
            loop = FeedbackLoop(
                loop_id=100 + i,
                target_dim=goal_dim,
                learning_rate=learning_rate * 0.5,
                seed=seed + 100 + i * 19
            )
            self.regional_loops.append(loop)
        
        # Global feedback loop
        self.global_loop = FeedbackLoop(
            loop_id=999,
            target_dim=goal_dim,
            learning_rate=learning_rate * 0.2,
            seed=seed + 999
        )
        
        # Goal state (what the system is trying to achieve)
        self.goal_state: List[float] = [
            self.prng.uniform(0.3, 0.7) for _ in range(goal_dim)
        ]
        
        # Error history
        self.local_errors: List[float] = []
        self.regional_errors: List[float] = []
        self.global_errors: List[float] = []
        
        self.step_count = 0
    
    def step(
        self,
        local_states: List[List[float]],
        regional_states: List[List[float]],
        global_state: List[float],
        next_global_state: List[float]
    ) -> Dict[str, Any]:
        """
        Advance the feedback network by one timestep.
        
        Args:
            local_states: States of local units (grouped by region)
            regional_states: States of regions
            global_state: Global state at time t
            next_global_state: Global state at time t+1 (actual)
            
        Returns:
            Dictionary of error signals at all scales
        """
        self.step_count += 1
        
        # Local feedback
        local_error_total = 0.0
        for r, loop in enumerate(self.local_loops):
            if r < len(local_states):
                current = local_states[r]
                if r + 1 < len(local_states):
                    next_state = local_states[r + 1]
                else:
                    next_state = current
                
                if len(current) == loop.target_dim:
                    _, err = loop.compute_error(current, next_state)
                    local_error_total += err
        
        self.local_errors.append(local_error_total / max(1, len(self.local_loops)))
        
        # Regional feedback
        regional_error_total = 0.0
        for r, loop in enumerate(self.regional_loops):
            if r < len(regional_states):
                current = regional_states[r]
                if len(current) == loop.target_dim:
                    _, err = loop.compute_error(current, self.goal_state)
                    regional_error_total += err
        
        self.regional_errors.append(regional_error_total / max(1, len(self.regional_loops)))
        
        # Global feedback
        _, global_err = self.global_loop.compute_error(global_state, next_global_state)
        self.global_errors.append(global_err)
        
        return {
            'local_error': local_error_total / max(1, len(self.local_loops)),
            'regional_error': regional_error_total / max(1, len(self.regional_loops)),
            'global_error': global_err,
            'step': self.step_count,
        }
    
    def set_goal(self, goal: List[float]) -> None:
        """Set the goal state for global regulation."""
        self.goal_state = list(goal)
    
    def get_goal(self) -> List[float]:
        """Get the current goal state."""
        return list(self.goal_state)
    
    def get_error_history(self) -> Dict[str, List[float]]:
        """Get the error history at all scales."""
        return {
            'local': list(self.local_errors[-100:]),
            'regional': list(self.regional_errors[-100:]),
            'global': list(self.global_errors[-100:]),
        }


# =============================================================================
# SECTION 7: HYDRAULIC MEMORY SYSTEM
# =============================================================================

class PressureVessel:
    """
    A single pressure vessel for storing information in a hydraulic memory.
    
    Heron's hydraulic devices stored energy and information in the
    pressure of water columns. A taller column meant higher pressure,
    which could be used to drive mechanisms. This vessel implements
    a similar principle: the "pressure" in the vessel stores a value,
    and the vessel can be connected to other vessels to form
    associative memory networks.
    """
    
    def __init__(
        self,
        vessel_id: int,
        capacity: float = 1.0,
        leak_rate: float = 0.01,
        seed: int = 120
    ):
        self.vessel_id = vessel_id
        self.capacity = capacity
        self.leak_rate = leak_rate
        
        self.prng = PRNG(seed=seed)
        
        self.pressure: float = 0.0
        self.connections: Dict[int, float] = {}  # vessel_id -> conductance
        self.input_history: List[float] = []
        self.output_history: List[float] = []
    
    def fill(self, amount: float) -> None:
        """Add pressure to the vessel."""
        self.pressure = clip(self.pressure + amount, 0.0, self.capacity)
        self.input_history.append(amount)
    
    def drain(self, amount: float) -> float:
        """Remove pressure from the vessel."""
        drained = min(amount, self.pressure)
        self.pressure -= drained
        self.output_history.append(drained)
        return drained
    
    def leak(self) -> None:
        """Apply leak rate to pressure."""
        self.pressure *= (1.0 - self.leak_rate)
    
    def connect(self, vessel_id: int, conductance: float) -> None:
        """Connect this vessel to another with given conductance."""
        self.connections[vessel_id] = clip(conductance, 0.0, 1.0)
    
    def propagate(self, other_vessels: Dict[int, 'PressureVessel']) -> None:
        """Propagate pressure to connected vessels."""
        for vessel_id, conductance in self.connections.items():
            if vessel_id in other_vessels:
                transfer = self.pressure * conductance * 0.1
                other_vessels[vessel_id].fill(transfer)
                self.drain(transfer)
    
    def get_pressure(self) -> float:
        """Get the current pressure level."""
        return self.pressure
    
    def set_pressure(self, pressure: float) -> None:
        """Set the pressure directly."""
        self.pressure = clip(pressure, 0.0, self.capacity)


class HydraulicMemory:
    """
    Associative memory system using hydraulic principles.
    
    Heron's hydraulic systems stored and transmitted energy through
    networks of connected vessels and tubes. This memory system
    implements a similar architecture: information is stored as
    pressure levels in vessels, retrieval is achieved by applying
    a query pressure and reading the resulting pattern.
    
    The system has three levels of storage:
    - Primary vessels: fast-access, low-capacity
    - Secondary vessels: medium-access, medium-capacity
    - Tertiary vessels: slow-access, high-capacity
    
    Associations between patterns are stored as connections
    between vessels with specific conductances.
    """
    
    def __init__(
        self,
        primary_count: int = 8,
        secondary_count: int = 16,
        tertiary_count: int = 32,
        pattern_dim: int = 8,
        seed: int = 120
    ):
        self.primary_count = primary_count
        self.secondary_count = secondary_count
        self.tertiary_count = tertiary_count
        self.pattern_dim = pattern_dim
        
        self.prng = PRNG(seed=seed)
        
        # Create vessels
        self.primary: List[PressureVessel] = []
        for i in range(primary_count):
            v = PressureVessel(
                vessel_id=i,
                capacity=1.0,
                leak_rate=0.05,
                seed=seed + i * 11
            )
            self.primary.append(v)
        
        self.secondary: List[PressureVessel] = []
        for i in range(secondary_count):
            v = PressureVessel(
                vessel_id=1000 + i,
                capacity=2.0,
                leak_rate=0.02,
                seed=seed + 1000 + i * 11
            )
            self.secondary.append(v)
        
        self.tertiary: List[PressureVessel] = []
        for i in range(tertiary_count):
            v = PressureVessel(
                vessel_id=10000 + i,
                capacity=5.0,
                leak_rate=0.005,
                seed=seed + 10000 + i * 11
            )
            self.tertiary.append(v)
        
        # Association weights between patterns and vessels
        self.all_vessels = self.primary + self.secondary + self.tertiary
        self.vessel_map = {v.vessel_id: v for v in self.all_vessels}
        
        # Pattern storage
        self.stored_patterns: List[Dict[str, Any]] = []
        
        # Build random associations
        self._build_random_associations()
    
    def _build_random_associations(self) -> None:
        """Build random associations between vessels."""
        for i, v in enumerate(self.primary):
            # Connect to a few secondary vessels
            targets = self.prng.sample(
                [s.vessel_id for s in self.secondary],
                min(3, len(self.secondary))
            )
            for t in targets:
                v.connect(t, self.prng.uniform(0.1, 0.5))
        
        for i, s in enumerate(self.secondary):
            # Connect to tertiary vessels
            targets = self.prng.sample(
                [t.vessel_id for t in self.tertiary],
                min(4, len(self.tertiary))
            )
            for t in targets:
                s.connect(t, self.prng.uniform(0.1, 0.3))
    
    def store(self, pattern: List[float]) -> int:
        """
        Store a pattern in the hydraulic memory.
        
        Args:
            pattern: Pattern to store (will be encoded as pressures)
            
        Returns:
            Index of stored pattern
        """
        if len(pattern) > len(self.primary):
            pattern = pattern[:len(self.primary)]
        elif len(pattern) < len(self.primary):
            pattern = list(pattern) + [0.0] * (len(self.primary) - len(pattern))
        
        # Fill primary vessels with pattern
        for i, p in enumerate(pattern):
            self.primary[i].fill(clip(p, 0.0, 1.0))
        
        # Propagate to secondary and tertiary
        for _ in range(3):
            for v in self.primary:
                v.propagate(self.vessel_map)
            for s in self.secondary:
                s.propagate(self.vessel_map)
        
        # Record stored pattern
        pattern_record = {
            'index': len(self.stored_patterns),
            'primary_pressures': [v.get_pressure() for v in self.primary],
            'secondary_pressures': [v.get_pressure() for v in self.secondary],
            'tertiary_pressures': [v.get_pressure() for v in self.tertiary],
        }
        self.stored_patterns.append(pattern_record)
        
        return pattern_record['index']
    
    def retrieve(self, query: List[float], top_k: int = 1) -> Tuple[List[float], List[float]]:
        """
        Retrieve the closest matching pattern to a query.
        
        Args:
            query: Query pattern
            top_k: Number of top matches to return
            
        Returns:
            Tuple of (best_match_pattern, match_scores)
        """
        if len(query) > len(self.primary):
            query = query[:len(self.primary)]
        elif len(query) < len(self.primary):
            query = list(query) + [0.0] * (len(self.primary) - len(query))
        
        # Create temporary query vessels
        query_vessels = [PressureVessel(vessel_id=-i, capacity=1.0, seed=i * 777)
                        for i in range(len(query))]
        for i, q in enumerate(query_vessels):
            q.set_pressure(clip(query[i], 0.0, 1.0))
        
        # Compute similarity to each stored pattern
        scores = []
        for record in self.stored_patterns:
            stored = record['primary_pressures']
            sim = cosine_similarity(query, stored[:len(query)])
            scores.append(sim)
        
        if not scores:
            return [0.0] * self.pattern_dim, [0.0]
        
        # Get top-k matches
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, score in indexed_scores[:top_k]]
        top_scores = [scores[idx] for idx in top_indices]
        
        # Return the best matching pattern
        if top_indices:
            best_idx = top_indices[0]
            best_record = self.stored_patterns[best_idx]
            result = best_record['primary_pressures'][:self.pattern_dim]
            return result, top_scores
        
        return [0.0] * self.pattern_dim, [0.0]
    
    def step(self) -> None:
        """Advance the memory system by applying leaks and propagation."""
        for v in self.all_vessels:
            v.leak()
        
        for _ in range(2):
            for v in self.primary:
                v.propagate(self.vessel_map)
            for s in self.secondary:
                s.propagate(self.vessel_map)


# =============================================================================
# SECTION 8: GEOMETRY AND SPACE PROCESSOR
# =============================================================================

class GeometryProcessor:
    """
    Geometric reasoning processor based on Heron's mathematical algorithms.
    
    Heron's Metrica contained algorithms for computing areas and volumes
    of various geometric shapes — triangles, circles, cylinders, spheres,
    and more complex solids. These algorithms represent a remarkable
    achievement of ancient mathematics. This processor implements
    geometric reasoning capabilities inspired by Heron's work.
    
    The processor can:
    - Compute distances and areas from given measurements
    - Transform geometric representations
    - Reason about spatial relationships
    - Perform surveying-like calculations (like Heron's dioptra)
    """
    
    def __init__(self, seed: int = 120):
        self.prng = PRNG(seed=seed)
        
        # Geometric primitives cache
        self.point_cache: List[Tuple[float, float]] = []
        self.line_cache: List[Dict[str, Any]] = []
        self.shape_cache: List[Dict[str, Any]] = []
    
    def add_point(self, x: float, y: float) -> int:
        """Add a point to the workspace."""
        self.point_cache.append((x, y))
        return len(self.point_cache) - 1
    
    def distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Compute Euclidean distance between two points."""
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    
    def triangle_area(self, p1: Tuple[float, float], p2: Tuple[float, float],
                      p3: Tuple[float, float]) -> float:
        """
        Compute the area of a triangle using Heron's formula.
        
        Heron's formula: area = sqrt(s * (s-a) * (s-b) * (s-c))
        where a, b, c are the side lengths and s = (a+b+c)/2
        """
        a = self.distance(p1, p2)
        b = self.distance(p2, p3)
        c = self.distance(p3, p1)
        
        s = (a + b + c) / 2.0
        
        # Heron's formula
        area_sq = s * (s - a) * (s - b) * (s - c)
        
        if area_sq < 0:
            return 0.0
        
        return math.sqrt(area_sq)
    
    def circle_area(self, radius: float) -> float:
        """Compute area of a circle."""
        return math.pi * radius ** 2
    
    def sphere_volume(self, radius: float) -> float:
        """Compute volume of a sphere."""
        return (4.0 / 3.0) * math.pi * radius ** 3
    
    def cylinder_volume(self, radius: float, height: float) -> float:
        """Compute volume of a cylinder."""
        return math.pi * radius ** 2 * height
    
    def cone_volume(self, radius: float, height: float) -> float:
        """Compute volume of a cone."""
        return (1.0 / 3.0) * math.pi * radius ** 2 * height
    
    def dioptra_angle(self, p1: Tuple[float, float], p2: Tuple[float, float],
                      reference: Tuple[float, float] = (1.0, 0.0)) -> float:
        """
        Compute angle using dioptra-inspired method.
        
        The dioptra was a surveying instrument that measured angles
        with high precision. This method computes the angle between
        two vectors from a common origin.
        """
        v1 = (p1[0] - reference[0], p1[1] - reference[1])
        v2 = (p2[0] - reference[0], p2[1] - reference[1])
        
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
        mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
        
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        cos_angle = clip(dot / (mag1 * mag2), -1.0, 1.0)
        return math.acos(cos_angle)
    
    def project_point_to_line(
        self,
        point: Tuple[float, float],
        line_start: Tuple[float, float],
        line_end: Tuple[float, float]
    ) -> Tuple[float, float]:
        """
        Project a point onto a line segment (closest point).
        
        Returns the point on the line segment closest to the given point.
        """
        dx = line_end[0] - line_start[0]
        dy = line_end[1] - line_start[1]
        
        if dx == 0 and dy == 0:
            return line_start
        
        t = ((point[0] - line_start[0]) * dx + (point[1] - line_start[1]) * dy) / (dx ** 2 + dy ** 2)
        t = clip(t, 0.0, 1.0)
        
        return (line_start[0] + t * dx, line_start[1] + t * dy)
    
    def point_in_polygon(self, point: Tuple[float, float],
                         polygon: List[Tuple[float, float]]) -> bool:
        """
        Test if a point is inside a polygon using ray casting.
        
        This is a standard algorithm but Heron's work on polygon
        areas implicitly involved similar spatial reasoning.
        """
        n = len(polygon)
        inside = False
        
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            
            if ((yi > point[1]) != (yj > point[1])) and \
               (point[0] < (xj - xi) * (point[1] - yi) / (yj - yi) + xi):
                inside = not inside
            
            j = i
        
        return inside
    
    def polygon_area(self, polygon: List[Tuple[float, float]]) -> float:
        """
        Compute area of a polygon using the shoelace formula.
        
        The shoelace formula: area = 0.5 * |sum(x_i * y_{i+1} - x_{i+1} * y_i)|
        """
        n = len(polygon)
        if n < 3:
            return 0.0
        
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += polygon[i][0] * polygon[j][1]
            area -= polygon[j][0] * polygon[i][1]
        
        return abs(area) / 2.0
    
    def transform_vector(self, vector: List[float],
                        scale: float = 1.0,
                        rotation: float = 0.0,
                        translation: Tuple[float, float] = (0.0, 0.0)) -> List[float]:
        """
        Apply geometric transformation to a 2D vector.
        
        Args:
            vector: [x, y] input
            scale: Scaling factor
            rotation: Rotation angle in radians
            translation: (dx, dy) translation
            
        Returns:
            Transformed [x', y'] vector
        """
        x, y = vector[0], vector[1]
        
        # Scale
        x *= scale
        y *= scale
        
        # Rotate
        xr = x * math.cos(rotation) - y * math.sin(rotation)
        yr = x * math.sin(rotation) + y * math.cos(rotation)
        
        # Translate
        xr += translation[0]
        yr += translation[1]
        
        return [xr, yr]


# =============================================================================
# SECTION 9: EMBODIMENT INTERFACE
# =============================================================================

class SimulatedBody:
    """
    A simple simulated body for embodied cognition experiments.
    
    Heron's automata were embodied: they had physical bodies that
    moved through space, interacting with objects and obstacles.
    This simulated body provides a simple 2D embodiment for the HAN.
    """
    
    def __init__(
        self,
        x: float = 0.0,
        y: float = 0.0,
        heading: float = 0.0,
        size: float = 1.0,
        seed: int = 120
    ):
        self.x = x
        self.y = y
        self.heading = heading
        self.size = size
        
        self.prng = PRNG(seed=seed)
        
        self.velocity: float = 0.0
        self.angular_velocity: float = 0.0
        
        self.sensors: Dict[str, float] = {
            'proximity_front': 10.0,
            'proximity_back': 10.0,
            'proximity_left': 10.0,
            'proximity_right': 10.0,
        }
        
        self.objects: List[Dict[str, Any]] = []
        self._generate_objects(10)
    
    def _generate_objects(self, count: int) -> None:
        """Generate random objects in the environment."""
        for _ in range(count):
            obj = {
                'x': self.prng.uniform(-20.0, 20.0),
                'y': self.prng.uniform(-20.0, 20.0),
                'radius': self.prng.uniform(0.5, 2.0),
                'type': self.prng.choice(['obstacle', 'target', 'neutral']),
            }
            self.objects.append(obj)
    
    def step(self, forward: float, turn: float) -> None:
        """
        Advance the body by one timestep.
        
        Args:
            forward: Forward velocity command (-1 to 1)
            turn: Turn command (-1 to 1)
        """
        self.velocity = clip(forward * 2.0, -2.0, 2.0)
        self.angular_velocity = clip(turn * 0.5, -0.5, 0.5)
        
        self.heading += self.angular_velocity
        self.heading = self.heading % (2.0 * math.pi)
        
        self.x += self.velocity * math.cos(self.heading)
        self.y += self.velocity * math.sin(self.heading)
        
        self._update_sensors()
    
    def _update_sensors(self) -> None:
        """Update proximity sensors based on current position and objects."""
        directions = {
            'front': self.heading,
            'back': (self.heading + math.pi) % (2.0 * math.pi),
            'left': (self.heading - math.pi / 2) % (2.0 * math.pi),
            'right': (self.heading + math.pi / 2) % (2.0 * math.pi),
        }
        
        for direction_name, direction_angle in directions.items():
            min_dist = 100.0
            
            for obj in self.objects:
                # Distance to object center
                dx = obj['x'] - self.x
                dy = obj['y'] - self.y
                dist = math.sqrt(dx ** 2 + dy ** 2)
                
                # Direction to object
                angle_to_obj = math.atan2(dy, dx)
                angle_diff = abs(angle_to_obj - direction_angle)
                angle_diff = min(angle_diff, 2.0 * math.pi - angle_diff)
                
                if angle_diff < math.pi / 4:
                    effective_dist = dist - obj['radius']
                    min_dist = min(min_dist, effective_dist)
            
            self.sensors[f'proximity_{direction_name}'] = max(0.0, min_dist)
    
    def get_sensor_vector(self) -> List[float]:
        """Get sensor readings as a vector."""
        return [
            self.sensors['proximity_front'],
            self.sensors['proximity_back'],
            self.sensors['proximity_left'],
            self.sensors['proximity_right'],
            math.cos(self.heading),
            math.sin(self.heading),
            self.velocity,
            self.angular_velocity,
        ]
    
    def get_position(self) -> Tuple[float, float]:
        """Get current position."""
        return (self.x, self.y)


class EmbodimentInterface:
    """
    Interface connecting HAN to a simulated or physical body.
    
    Heron understood that cognition is embodied — that intelligent
    behavior emerges from the interaction of a cognitive system with
    a physical body moving through a physical environment. This
    interface implements that connection.
    """
    
    def __init__(
        self,
        input_dim: int = 8,
        output_dim: int = 2,
        body: Optional[SimulatedBody] = None,
        seed: int = 120
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.prng = PRNG(seed=seed)
        
        self.body = body if body is not None else SimulatedBody(seed=seed)
        
        # Sensor processing
        self.sensor_weights = [
            [self.prng.gauss(0.0, 0.1) for _ in range(input_dim)]
            for _ in range(input_dim)
        ]
        
        # Motor mapping
        self.motor_weights = [
            [self.prng.gauss(0.0, 0.1) for _ in range(output_dim)]
            for _ in range(input_dim)
        ]
        
        self.step_count = 0
    
    def get_sensor_input(self) -> List[float]:
        """Get current sensor readings as input to the network."""
        raw = self.body.get_sensor_vector()
        
        # Pad or truncate to input_dim
        if len(raw) < self.input_dim:
            raw = raw + [0.0] * (self.input_dim - len(raw))
        elif len(raw) > self.input_dim:
            raw = raw[:self.input_dim]
        
        # Apply sensor processing
        processed = []
        for i in range(self.input_dim):
            val = sum(self.sensor_weights[i][j] * raw[j] for j in range(self.input_dim))
            processed.append(tanh_activation(val))
        
        return processed
    
    def apply_motor_command(self, motor_output: List[float]) -> None:
        """Convert network output to motor commands and apply to body."""
        if len(motor_output) < self.output_dim:
            motor_output = list(motor_output) + [0.0] * (self.output_dim - len(motor_output))
        elif len(motor_output) > self.output_dim:
            motor_output = motor_output[:self.output_dim]
        
        # Apply motor mapping weights
        forward = sum(self.motor_weights[0][j] * motor_output[j] for j in range(self.output_dim))
        turn = sum(self.motor_weights[1][j] * motor_output[j] for j in range(self.output_dim))
        
        self.body.step(forward, turn)
        self.step_count += 1
    
    def get_body_state(self) -> Dict[str, Any]:
        """Get the current state of the body."""
        pos = self.body.get_position()
        return {
            'x': pos[0],
            'y': pos[1],
            'heading': self.body.heading,
            'velocity': self.body.velocity,
            'step': self.step_count,
            'sensors': dict(self.body.sensors),
        }


# =============================================================================
# SECTION 10: PROGRAM SYNTHESIS MODULE
# =============================================================================

class ProgramSynthesis:
    """
    Module for synthesizing new behavioral programs.
    
    Heron understood that the power of automata lay not just in their
    mechanical construction but in the programs that drove them — the
    arrangement of cams on the drum. This module generates new programs
    by combining and modifying existing sequences.
    
    The synthesis uses an evolutionary approach:
    1. Generate candidate programs through mutation and crossover
    2. Evaluate candidates using the AutomataSequencer
    3. Select the best candidates for retention
    
    This is Heron's insight made computational: the program (cam
    arrangement) can be modified independently of the hardware,
    enabling learning and adaptation.
    """
    
    def __init__(
        self,
        num_modules: int = 8,
        population_size: int = 16,
        mutation_rate: float = 0.1,
        seed: int = 120
    ):
        self.num_modules = num_modules
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        
        self.prng = PRNG(seed=seed)
        
        # Population of programs (each is a list of CamProfiles)
        self.population: List[List[CamProfile]] = []
        self.fitness_scores: List[float] = []
        
        # Initialize with random programs
        self._initialize_population()
    
    def _initialize_population(self) -> None:
        """Initialize population with random programs."""
        for _ in range(self.population_size):
            num_cams = self.prng.randint(4, 12)
            cams = []
            for i in range(num_cams):
                pattern = {
                    m: self.prng.uniform(0.0, 1.0)
                    for m in range(self.num_modules)
                    if self.prng.random() > 0.2
                }
                duration = self.prng.uniform(0.5, 2.0)
                cam = CamProfile(cam_id=i, control_pattern=pattern, duration=duration)
                cams.append(cam)
            self.population.append(cams)
        
        self.fitness_scores = [0.0] * len(self.population)
    
    def evaluate(
        self,
        program: List[CamProfile],
        evaluator_fn: Callable[[List[CamProfile]], float]
    ) -> float:
        """
        Evaluate a program using an evaluator function.
        
        Args:
            program: List of CamProfiles (the program)
            evaluator_fn: Function that takes a program and returns a fitness score
            
        Returns:
            Fitness score
        """
        return evaluator_fn(program)
    
    def mutate(self, program: List[CamProfile]) -> List[CamProfile]:
        """Create a mutated copy of a program."""
        new_program = []
        for cam in program:
            if self.prng.random() < self.mutation_rate:
                new_program.append(cam.mutate(mutation_rate=self.mutation_rate))
            else:
                new_program.append(cam)
        
        # Possibly add or remove a cam
        if self.prng.random() < 0.1 and len(new_program) > 2:
            idx = self.prng.randint(0, len(new_program) - 1)
            new_program.pop(idx)
        elif self.prng.random() < 0.1 and len(new_program) < 16:
            # Add a new random cam
            pattern = {
                m: self.prng.uniform(0.0, 1.0)
                for m in range(self.num_modules)
            }
            new_cam = CamProfile(
                cam_id=self.prng.randint(10000, 99999),
                control_pattern=pattern,
                duration=self.prng.uniform(0.5, 2.0)
            )
            new_program.insert(self.prng.randint(0, len(new_program)), new_cam)
        
        return new_program
    
    def crossover(
        self,
        program1: List[CamProfile],
        program2: List[CamProfile]
    ) -> List[CamProfile]:
        """Create a new program by crossing over two programs."""
        min_len = min(len(program1), len(program2))
        if min_len < 2:
            return list(program1)
        
        point = self.prng.randint(1, min_len - 1)
        
        new_program = list(program1[:point]) + list(program2[point:])
        return new_program
    
    def select_parent(self) -> Tuple[List[CamProfile], int]:
        """Select a parent program using fitness-proportionate selection."""
        if not self.fitness_scores:
            return self.prng.choice(self.population), 0
        
        total_fitness = sum(max(0.0, f) for f in self.fitness_scores)
        if total_fitness == 0:
            return self.prng.choice(self.population), self.population.index(self.prng.choice(self.population))
        
        threshold = self.prng.uniform(0.0, total_fitness)
        cumulative = 0.0
        for i, fitness in enumerate(self.fitness_scores):
            cumulative += max(0.0, fitness)
            if cumulative >= threshold:
                return self.population[i], i
        
        return self.population[-1], len(self.population) - 1
    
    def evolve_one_generation(
        self,
        evaluator_fn: Callable[[List[CamProfile]], float]
    ) -> float:
        """
        Evolve one generation of programs.
        
        Args:
            evaluator_fn: Function to evaluate program fitness
            
        Returns:
            Best fitness in the new generation
        """
        # Evaluate all programs
        for i, program in enumerate(self.population):
            self.fitness_scores[i] = self.evaluate(program, evaluator_fn)
        
        # Create new population
        new_population = []
        new_fitness = []
        
        # Elitism: keep the best program unchanged
        best_idx = max(range(len(self.fitness_scores)), key=lambda i: self.fitness_scores[i])
        new_population.append(list(self.population[best_idx]))
        new_fitness.append(self.fitness_scores[best_idx])
        
        # Generate rest through mutation and crossover
        while len(new_population) < self.population_size:
            p1, _ = self.select_parent()
            p2, _ = self.select_parent()
            
            if self.prng.random() < 0.7:
                child = self.crossover(p1, p2)
            else:
                child = list(p1)
            
            child = self.mutate(child)
            new_population.append(child)
            new_fitness.append(0.0)
        
        self.population = new_population
        self.fitness_scores = new_fitness
        
        return max(self.fitness_scores)
    
    def get_best_program(self) -> Tuple[List[CamProfile], float]:
        """Get the best program and its fitness score."""
        if not self.fitness_scores:
            return self.population[0], 0.0
        
        best_idx = max(range(len(self.fitness_scores)), key=lambda i: self.fitness_scores[i])
        return self.population[best_idx], self.fitness_scores[best_idx]


# =============================================================================
# SECTION 11: HERON AUTOMATON NETWORK (MAIN ARCHITECTURE)
# =============================================================================

class HeronAutomatonNetwork:
    """
    The Heron Automaton Network (HAN): A complete neural architecture
    inspired by Heron of Alexandria's philosophy of mind.
    
    This architecture combines all six of Heron's key principles:
    1. Aeolipile Dynamics: Continuous rotational attractor dynamics
    2. Pneumatic Control: Global broadcast regulatory signals
    3. Gear-Based Hierarchy: Deep staged transformation network
    4. Automata Sequencer: Stored-program behavior control
    5. Feedback Regulation: Multi-scale error-driven learning
    6. Hydraulic Memory: Associative memory with pressure semantics
    
    Plus supporting systems:
    7. Geometry Processor: Spatial/geometric reasoning
    8. Embodiment Interface: Body-environment coupling
    9. Program Synthesis: Evolutionary program generation
    
    The HAN is designed to learn patterns, store sequences, regulate
    its own behavior through feedback, and generate novel programs.
    """
    
    def __init__(
        self,
        input_dim: int = 8,
        hidden_dims: List[int] = None,
        output_dim: int = 8,
        num_aeolipile_nodes: int = 16,
        num_pneumatic_sources: int = 8,
        num_regions: int = 4,
        memory_pattern_dim: int = 8,
        seed: int = 120
    ):
        if hidden_dims is None:
            hidden_dims = [16, 16, 16]
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.num_aeolipile_nodes = num_aeolipile_nodes
        self.num_pneumatic_sources = num_pneumatic_sources
        self.num_regions = num_regions
        self.memory_pattern_dim = memory_pattern_dim
        
        self.prng = PRNG(seed=seed)
        
        # Core processing components
        self.aeolipile_layer = AeolipileDynamicsLayer(
            num_nodes=num_aeolipile_nodes,
            input_dim=input_dim,
            output_dim=hidden_dims[0] if hidden_dims else output_dim,
            coupling_strength=0.05,
            seed=seed
        )
        
        self.pneumatic_layer = PneumaticControlLayer(
            num_sources=num_pneumatic_sources,
            num_targets=num_aeolipile_nodes,
            pressure_decay=0.1,
            seed=seed
        )
        
        self.gear_hierarchy = GearBasedHierarchy(
            input_dim=hidden_dims[0] if hidden_dims else input_dim,
            hidden_dims=hidden_dims[1:] if len(hidden_dims) > 1 else [16, 16],
            output_dim=output_dim,
            num_stages=len(hidden_dims),
            seed=seed
        )
        
        self.sequencer = AutomataSequencer(
            num_modules=num_aeolipile_nodes,
            short_capacity=8,
            medium_capacity=16,
            long_capacity=32,
            seed=seed
        )
        
        self.feedback_network = FeedbackRegulationNetwork(
            num_local_units=num_aeolipile_nodes,
            num_regions=num_regions,
            goal_dim=output_dim,
            learning_rate=0.01,
            seed=seed
        )
        
        self.hydraulic_memory = HydraulicMemory(
            primary_count=input_dim,
            secondary_count=16,
            tertiary_count=32,
            pattern_dim=memory_pattern_dim,
            seed=seed
        )
        
        self.geometry_processor = GeometryProcessor(seed=seed)
        
        self.embodiment = EmbodimentInterface(
            input_dim=input_dim,
            output_dim=2,
            body=SimulatedBody(seed=seed),
            seed=seed
        )
        
        self.program_synthesis = ProgramSynthesis(
            num_modules=num_aeolipile_nodes,
            population_size=16,
            mutation_rate=0.1,
            seed=seed
        )
        
        # Internal state
        self.step_count = 0
        self.internal_state: Dict[str, Any] = {}
        self.history: List[Dict[str, Any]] = []
    
    def initialize(self) -> None:
        """Initialize all components with default configurations."""
        self.aeolipile_layer.reset()
        self.sequencer.initialize()
        self.step_count = 0
        self.history = []
    
    def step(
        self,
        input_vector: Optional[List[float]] = None,
        use_sequencer: bool = True,
        use_feedback: bool = True,
        use_memory: bool = False
    ) -> Dict[str, Any]:
        """
        Advance the HAN by one timestep.
        
        Args:
            input_vector: External input (optional)
            use_sequencer: Whether to use the automata sequencer
            use_feedback: Whether to use feedback regulation
            use_memory: Whether to query/store in hydraulic memory
            
        Returns:
            Dictionary of outputs and internal states
        """
        self.step_count += 1
        
        if input_vector is None:
            input_vector = self.embodiment.get_sensor_input()
        
        # Pad/truncate input to expected dimension
        if len(input_vector) < self.input_dim:
            input_vector = list(input_vector) + [0.0] * (self.input_dim - len(input_vector))
        elif len(input_vector) > self.input_dim:
            input_vector = list(input_vector[:self.input_dim])
        
        # Step 1: Get sequencer controls (automata program)
        sequencer_controls: Dict[int, float] = {}
        if use_sequencer:
            sequencer_controls = self.sequencer.step(dt=1.0)
        
        # Step 2: Pneumatic modulation
        aeolipile_outputs = self.aeolipile_layer.step(input_vector)
        
        # Get global signal for pneumatic layer
        global_signal = [
            sum(aeolipile_outputs[i][j] for i in range(len(aeolipile_outputs))) / max(1, len(aeolipile_outputs))
            for j in range(min(len(aeolipile_outputs[0]), self.num_pneumatic_sources))
        ]
        
        target_activity = [sum(v) / len(v) for v in aeolipile_outputs]
        
        pneumatic_signals = self.pneumatic_layer.generate_regulatory_signals(
            global_signal, target_activity
        )
        
        # Apply pneumatic modulation to aeolipile outputs
        modulated_outputs = self.pneumatic_layer.modulate(
            [aeolipile_outputs[0][j] if aeolipile_outputs else 0.0
             for j in range(len(global_signal))],
            pneumatic_signals[:len(global_signal)]
        )
        
        # Step 3: Gear hierarchy processing
        gear_input = modulated_outputs if modulated_outputs else input_vector[:len(modulated_outputs)]
        if len(gear_input) < self.hidden_dims[0] if self.hidden_dims else self.output_dim:
            gear_input = gear_input + [0.0] * ((self.hidden_dims[0] if self.hidden_dims else self.output_dim) - len(gear_input))
        
        gear_output, all_stage_outputs = self.gear_hierarchy.full_forward(gear_input[:self.hidden_dims[0] if self.hidden_dims else self.output_dim])
        
        # Step 4: Feedback (if enabled)
        feedback_result = {}
        if use_feedback:
            local_states = [list(aeolipile_outputs[i]) for i in range(min(len(aeolipile_outputs), self.num_regions))]
            regional_states = all_stage_outputs[1:-1] if len(all_stage_outputs) > 2 else all_stage_outputs
            global_state = gear_output
            
            # Simulate next global state (in real system, would be actual next state)
            next_global_state = [g + self.prng.gauss(0.0, 0.01) for g in global_state]
            
            feedback_result = self.feedback_network.step(
                local_states, regional_states, global_state, next_global_state
            )
        
        # Step 5: Memory operations (if enabled)
        memory_result = {}
        if use_memory:
            if self.step_count % 10 == 0:
                self.hydraulic_memory.store(gear_output)
            
            query = gear_output[:self.memory_pattern_dim]
            retrieved, scores = self.hydraulic_memory.retrieve(query)
            memory_result = {'retrieved': retrieved, 'scores': scores}
            
            self.hydraulic_memory.step()
        
        # Step 6: Apply to embodiment
        motor_output = gear_output[:2]  # First 2 dims -> motor command
        self.embodiment.apply_motor_command(motor_output)
        
        # Collect all outputs
        result = {
            'step': self.step_count,
            'input': input_vector,
            'aeolipile_outputs': aeolipile_outputs[:3],
            'gear_output': gear_output,
            'modulated': modulated_outputs,
            'sequencer_controls': sequencer_controls,
            'feedback': feedback_result,
            'memory': memory_result,
            'embodiment_state': self.embodiment.get_body_state(),
        }
        
        self.history.append(result)
        return result
    
    def learn_sequence(
        self,
        sequence_id: str,
        evaluator_fn: Callable[[List[CamProfile]], float],
        generations: int = 10
    ) -> float:
        """
        Learn a behavioral sequence using evolutionary program synthesis.
        
        Args:
            sequence_id: Which sequencer drum to update ('short', 'medium', 'long')
            evaluator_fn: Function evaluating program fitness
            generations: Number of evolution generations
            
        Returns:
            Best fitness achieved
        """
        best_fitness = 0.0
        
        for gen in range(generations):
            best_fitness = self.program_synthesis.evolve_one_generation(evaluator_fn)
        
        best_program, best_score = self.program_synthesis.get_best_program()
        self.sequencer.learn_sequence(sequence_id, best_program)
        
        return best_score
    
    def get_attractor_state(self) -> List[float]:
        """Get the current attractor state of the aeolipile layer."""
        return self.aeolipile_layer.get_attractor_state()
    
    def get_full_state(self) -> Dict[str, Any]:
        """Get a comprehensive snapshot of the network's current state."""
        return {
            'step': self.step_count,
            'aeolipile_attractor': self.get_attractor_state(),
            'sequencer_state': self.sequencer.get_state(),
            'feedback_errors': self.feedback_network.get_error_history(),
            'embodiment_state': self.embodiment.get_body_state(),
            'history_len': len(self.history),
        }
    
    def run_episode(
        self,
        num_steps: int,
        use_sequencer: bool = True,
        use_feedback: bool = True,
        use_memory: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Run a complete episode of multiple timesteps.
        
        Args:
            num_steps: Number of steps to run
            use_sequencer: Whether to use the automata sequencer
            use_feedback: Whether to use feedback regulation
            use_memory: Whether to use hydraulic memory
            
        Returns:
            List of step results
        """
        results = []
        for _ in range(num_steps):
            result = self.step(
                use_sequencer=use_sequencer,
                use_feedback=use_feedback,
                use_memory=use_memory
            )
            results.append(result)
        
        return results


# =============================================================================
# SECTION 12: DEMONSTRATION
# =============================================================================

def demo() -> Dict[str, Any]:
    """
    Demonstrate the Heron Automaton Network working on a pattern
    learning and behavioral sequence task.
    
    This demo shows:
    1. Pattern storage and retrieval in hydraulic memory
    2. Behavioral sequencing via automata sequencer
    3. Feedback-driven error regulation
    4. Geometric reasoning with the geometry processor
    5. Embodied interaction with a simulated body
    6. Program synthesis via evolutionary algorithms
    """
    print("=" * 70)
    print("Heron Automaton Network (HAN) — Demonstration")
    print("=" * 70)
    print()
    
    print("[1] Creating the Heron Automaton Network...")
    han = HeronAutomatonNetwork(
        input_dim=8,
        hidden_dims=[16, 16, 16],
        output_dim=8,
        num_aeolipile_nodes=16,
        num_pneumatic_sources=8,
        num_regions=4,
        memory_pattern_dim=8,
        seed=120
    )
    han.initialize()
    print(f"    HAN created with {han.step_count} initial steps")
    print()
    
    print("[2] Running 20 timesteps with sequencer, feedback, and memory...")
    for i in range(20):
        result = han.step(use_sequencer=True, use_feedback=True, use_memory=(i % 5 == 0))
        
        if i % 5 == 0:
            body = result['embodiment_state']
            print(f"    Step {result['step']:3d}: pos=({body['x']:.2f}, {body['y']:.2f}) "
                  f"heading={body['heading']:.2f} "
                  f"feedback_error={result['feedback'].get('global_error', 0.0):.4f}")
    print()
    
    print("[3] Testing Hydraulic Memory: storing and retrieving patterns...")
    test_patterns = [
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        [0.2, 0.4, 0.6, 0.8, 1.0, 0.8, 0.6, 0.4],
    ]
    
    for i, pattern in enumerate(test_patterns):
        idx = han.hydraulic_memory.store(pattern)
        print(f"    Stored pattern {i}: area={han.geometry_processor.polygon_area([(0, 0), (pattern[0]*10, 0), (pattern[0]*10, pattern[1]*10)]) if i == 0 else 0:.2f}")
    
    # Retrieve
    query = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]
    retrieved, scores = han.hydraulic_memory.retrieve(query)
    print(f"    Retrieved from query {query[:3]}... : {retrieved[:3]}... score={scores[0]:.4f}")
    print()
    
    print("[4] Testing Geometry Processor (Heron's algorithms)...")
    gp = han.geometry_processor
    
    # Triangle area (Heron's formula)
    p1 = (0.0, 0.0)
    p2 = (3.0, 0.0)
    p3 = (0.0, 4.0)
    tri_area = gp.triangle_area(p1, p2, p3)
    print(f"    Triangle (0,0), (3,0), (0,4): area = {tri_area:.2f} (expected 6.0)")
    
    # Circle area
    circ_area = gp.circle_area(radius=2.0)
    print(f"    Circle radius=2: area = {circ_area:.4f} (expected {4*math.pi:.4f})")
    
    # Sphere volume
    sph_vol = gp.sphere_volume(radius=2.0)
    print(f"    Sphere radius=2: volume = {sph_vol:.4f} (expected {(4/3)*math.pi*8:.4f})")
    
    # Dioptra angle
    angle = gp.dioptra_angle((1.0, 0.0), (0.0, 1.0))
    print(f"    Dioptra angle between (1,0) and (0,1): {angle:.4f} rad (expected {math.pi/2:.4f})")
    
    # Polygon area (shoelace formula)
    square = [(0, 0), (1, 0), (1, 1), (0, 1)]
    poly_area = gp.polygon_area(square)
    print(f"    Unit square polygon: area = {poly_area:.2f} (expected 1.0)")
    print()
    
    print("[5] Testing Automata Sequencer (cam-drum program storage)...")
    for timescale in ['short', 'medium', 'long']:
        drum = getattr(han.sequencer, f'{timescale}_drum')
        print(f"    {timescale.capitalize()} drum: {len(drum)} cams, position={drum.position}")
    print()
    
    print("[6] Running embodied episode (50 steps with body navigation)...")
    for i in range(50):
        han.step(use_sequencer=True, use_feedback=True, use_memory=False)
    
    final_state = han.embodiment.get_body_state()
    print(f"    Final body position: ({final_state['x']:.2f}, {final_state['y']:.2f})")
    print(f"    Total steps: {final_state['step']}")
    print()
    
    print("[7] Testing Program Synthesis (evolutionary learning)...")
    def simple_evaluator(program: List[CamProfile]) -> float:
        """Simple fitness: reward longer, diverse programs."""
        if not program:
            return 0.0
        # Diversity bonus
        all_controls = set()
        for cam in program:
            for m, v in cam.control_pattern.items():
                all_controls.add((m, round(v, 1)))
        diversity = len(all_controls) / max(1, len(program) * 3)
        # Length bonus
        length_bonus = min(len(program) / 8.0, 1.0)
        return diversity * 0.5 + length_bonus * 0.5
    
    best_fitness = han.learn_sequence(
        'short',
        simple_evaluator,
        generations=5
    )
    print(f"    Best evolved fitness: {best_fitness:.4f}")
    print()
    
    print("[8] Full network state summary...")
    state = han.get_full_state()
    print(f"    Total steps run: {state['step']}")
    print(f"    Aeolipile attractor: {state['aeolipile_attractor'][:3]}")
    print(f"    Feedback global errors (last 5): {state['feedback_errors']['global'][-5:]}")
    print()
    
    print("=" * 70)
    print("Demonstration complete. HAN is functional.")
    print("=" * 70)
    
    return {
        'han': han,
        'final_state': state,
        'test_results': {
            'geometry': {
                'triangle_area': tri_area,
                'circle_area': circ_area,
                'sphere_volume': sph_vol,
                'dioptra_angle': angle,
                'polygon_area': poly_area,
            },
            'memory_retrieval_score': scores[0] if scores else 0.0,
            'evolved_fitness': best_fitness,
        }
    }


if __name__ == '__main__':
    results = demo()
