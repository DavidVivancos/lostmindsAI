#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Chapter 109: Cleopatra VII — The Last Pharaoh of Egypt: Cultural Intelligence,
#              Soft Power, and the Architecture of Diplomatic Mind
#
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 6 Minds 101 - 120 Available on Amazon https://www.amazon.com/dp/B0HF7G6JJD
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 109: Cleopatra VII (-69 to -30 BCE)
# Purpose: Computational implementations of Cleopatra VII's philosophy of mind
#          as inspired neural architectures for artificial general intelligence.
#
# This module implements systems derived from Cleopatra's approach to governance:
#   1. Multilingual Linguistic Encoding System (multilingual representation)
#   2. Cultural Intelligence Layer (cultural modeling and adaptation)
#   3. Legitimacy Construction Layer (authority building across populations)
#   4. Strategic Reasoning Layer (strategic planning and adaptation)
#   5. Diplomatic Integration Layer (integration of all layers)
#   6. Cleopatra Cognitive Architecture (full integrated 5-layer system)
#
# Date: 2026-04-19
# =============================================================================

"""
Cleopatra's Diplomatic Cognitive Architecture
=============================================

This module implements computational models inspired by Cleopatra VII's philosophy
of governance — her theory of political intelligence, her practice of soft power,
and her deployment of cultural and linguistic resources as instruments of statecraft.

The architectures explored here draw connections between:

  - Multilingual reasoning and political intelligence
  - Cultural intelligence as a form of cognitive adaptation
  - Soft power and the construction of political legitimacy
  - Intelligence networks as extensions of cognitive capacity
  - Strategic flexibility and adaptive governance

Cleopatra's approach provides a distinctive model for AGI because her constraints
were fundamentally cognitive and cultural rather than purely technical: she ruled
a multilingual, multicultural kingdom with limited military resources, and she
succeeded through intelligence, persuasion, and strategic flexibility.

The module is organized into the following sections:

  SECTION A: CORE MATHEMATICAL PRIMITIVES
      Vector, matrix, and activation primitives used throughout.

  SECTION B: MULTILINGUAL LINGUISTIC ENCODING SYSTEM (MLES)
      Multilingual representation preserving cultural nuance.

  SECTION C: CULTURAL INTELLIGENCE LAYER (CIL)
      Cultural modeling for adapting behavior to different populations.

  SECTION D: LEGITIMACY CONSTRUCTION LAYER (LCL)
      Multi-channel authority construction and maintenance.

  SECTION E: STRATEGIC REASONING LAYER (SRL)
      Strategic planning, coalition building, and adaptive response.

  SECTION F: DIPLOMATIC INTEGRATION LAYER (DIL)
      Integration of all layers into coherent diplomatic outputs.

  SECTION G: CLEOPATRA COGNITIVE ARCHITECTURE (CCA)
      Full integrated multi-layer cognitive architecture.

  SECTION H: DEMONSTRATIONS AND EXAMPLES
      Working demonstrations of all components.

  SECTION I: ANALYSIS AND DIAGNOSTICS
      Analytical tools for inspecting the state of the system.

Usage:
    python3 chapter_0109_cleopatra_vii.py         # Run all demonstrations
    python3 chapter_0109_cleopatra_vii.py --demo=DEMO  # Run specific demo
    python3 chapter_0109_cleopatra_vii.py --test  # Run unit tests
    python3 chapter_0109_cleopatra_vii.py --arch  # Show architecture overview
"""

from __future__ import annotations

import sys
import os
import math
import random
import copy
import json
import warnings
import re
from typing import (
    List, Tuple, Optional, Dict, Any, Callable, Union,
    Generic, TypeVar, Protocol, NamedTuple
)
from dataclasses import dataclass, field, fields, astuple
from abc import ABC, abstractmethod
from enum import Enum, auto
from collections import deque
import threading
import time

# =============================================================================
# SECTION A: CORE MATHEMATICAL PRIMITIVES
# =============================================================================


class Vector:
    """
    A simple but complete vector class for n-dimensional real vectors.

    Used throughout the module for representing mental states, cultural
    profiles, legitimacy scores, and strategic assessments.
    """

    def __init__(self, data: Union[List[float], Tuple[float, ...]]):
        self._data = list(data)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> float:
        return self._data[index]

    def __setitem__(self, index: int, value: float) -> None:
        self._data[index] = float(value)

    def __add__(self, other: "Vector") -> "Vector":
        if len(self) != len(other):
            raise ValueError(f"Dimension mismatch: {len(self)} vs {len(other)}")
        return Vector([a + b for a, b in zip(self._data, other._data)])

    def __sub__(self, other: "Vector") -> "Vector":
        if len(self) != len(other):
            raise ValueError(f"Dimension mismatch: {len(self)} vs {len(other)}")
        return Vector([a - b for a, b in zip(self._data, other._data)])

    def __mul__(self, scalar: float) -> "Vector":
        return Vector([x * scalar for x in self._data])

    def __rmul__(self, scalar: float) -> "Vector":
        return self.__mul__(scalar)

    def __neg__(self) -> "Vector":
        return Vector([-x for x in self._data])

    def __pos__(self) -> "Vector":
        return Vector([+x for x in self._data])

    def __abs__(self) -> float:
        return math.sqrt(sum(x * x for x in self._data))

    def dot(self, other: "Vector") -> float:
        if len(self) != len(other):
            raise ValueError(f"Dimension mismatch: {len(self)} vs {len(other)}")
        return sum(a * b for a, b in zip(self._data, other._data))

    def norm(self) -> float:
        return abs(self)

    def normalize(self) -> "Vector":
        n = self.norm()
        if n < 1e-10:
            return Vector([0.0] * len(self))
        return self * (1.0 / n)

    def distance(self, other: "Vector") -> float:
        return (self - other).norm()

    def cosine_similarity(self, other: "Vector") -> float:
        n1, n2 = self.norm(), other.norm()
        if n1 < 1e-10 or n2 < 1e-10:
            return 0.0
        return self.dot(other) / (n1 * n2)

    def hadamard(self, other: "Vector") -> "Vector":
        """Element-wise multiplication (Hadamard product)."""
        if len(self) != len(other):
            raise ValueError(f"Dimension mismatch: {len(self)} vs {len(other)}")
        return Vector([a * b for a, b in zip(self._data, other._data)])

    def relu(self) -> "Vector":
        """ReLU activation: max(0, x)."""
        return Vector([max(0.0, x) for x in self._data])

    def sigmoid(self) -> "Vector":
        """Sigmoid activation: 1 / (1 + exp(-x))."""
        def sig(x: float) -> float:
            if x > 20:
                return 1.0
            if x < -20:
                return 0.0
            return 1.0 / (1.0 + math.exp(-x))
        return Vector([sig(x) for x in self._data])

    def tanh(self) -> "Vector":
        """Hyperbolic tangent activation."""
        return Vector([math.tanh(x) for x in self._data])

    def softmax(self) -> "Vector":
        """Softmax activation."""
        max_x = max(self._data)
        exps = [math.exp(x - max_x) for x in self._data]
        sum_exps = sum(exps)
        return Vector([e / sum_exps for e in exps])

    def clip(self, min_val: float, max_val: float) -> "Vector":
        return Vector([max(min_val, min(max_val, x)) for x in self._data])

    def mean(self) -> float:
        if not self._data:
            return 0.0
        return sum(self._data) / len(self._data)

    def variance(self) -> float:
        if len(self._data) < 2:
            return 0.0
        m = self.mean()
        return sum((x - m) ** 2 for x in self._data) / len(self._data)

    def to_list(self) -> List[float]:
        return list(self._data)

    def to_dict(self) -> Dict[str, Any]:
        return {"data": self._data, "dim": len(self._data)}

    def __repr__(self) -> str:
        return f"Vector({self._data[:5]}{'...' if len(self._data) > 5 else ''})"

    @classmethod
    def zeros(cls, dim: int) -> "Vector":
        return cls([0.0] * dim)

    @classmethod
    def ones(cls, dim: int) -> "Vector":
        return cls([1.0] * dim)

    @classmethod
    def random(cls, dim: int, seed: Optional[int] = None) -> "Vector":
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random
        return cls([rng.uniform(-1, 1) for _ in range(dim)])

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Vector":
        return cls(d["data"])


class Matrix:
    """
    A simple dense matrix class for 2D transformations.

    Used for linear transformations between cognitive layers.
    """

    def __init__(self, data: List[List[float]]):
        if not data:
            raise ValueError("Matrix cannot be empty")
        n_cols = len(data[0])
        for row in data:
            if len(row) != n_cols:
                raise ValueError("All rows must have the same length")
        self._data = [list(row) for row in data]
        self._rows = len(data)
        self._cols = n_cols

    @property
    def shape(self) -> Tuple[int, int]:
        return (self._rows, self._cols)

    def __getitem__(self, idx: Tuple[int, int]) -> float:
        r, c = idx
        return self._data[r][c]

    def __mul__(self, vec: Vector) -> Vector:
        if vec.__len__() != self._cols:
            raise ValueError(f"Vector dim {vec.__len__()} != matrix cols {self._cols}")
        result = []
        for r in range(self._rows):
            s = 0.0
            row = self._data[r]
            for c in range(self._cols):
                s += row[c] * vec[c]
            result.append(s)
        return Vector(result)

    def T(self) -> "Matrix":
        """Transpose."""
        return Matrix([[self._data[r][c] for r in range(self._rows)]
                       for c in range(self._cols)])

    @classmethod
    def identity(cls, dim: int) -> "Matrix":
        data = [[1.0 if r == c else 0.0 for c in range(dim)] for r in range(dim)]
        return cls(data)

    @classmethod
    def random(cls, rows: int, cols: int, seed: Optional[int] = None) -> "Matrix":
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random
        return cls([[rng.uniform(-1, 1) for _ in range(cols)] for _ in range(rows)])

    @classmethod
    def zeros(cls, rows: int, cols: int) -> "Matrix":
        return cls([[0.0] * cols for _ in range(rows)])

    def __repr__(self) -> str:
        return f"Matrix({self._rows}x{self._cols})"


class ActivationFunctions:
    """Collection of common activation functions for cognitive layers."""

    @staticmethod
    def relu(x: float) -> float:
        return max(0.0, x)

    @staticmethod
    def leaky_relu(x: float, alpha: float = 0.01) -> float:
        return x if x > 0 else alpha * x

    @staticmethod
    def sigmoid(x: float) -> float:
        if x > 20:
            return 1.0
        if x < -20:
            return 0.0
        return 1.0 / (1.0 + math.exp(-x))

    @staticmethod
    def tanh(x: float) -> float:
        return math.tanh(x)

    @staticmethod
    def gelu(x: float) -> float:
        return 0.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))

    @staticmethod
    def softmax(arr: List[float]) -> List[float]:
        max_x = max(arr)
        exps = [math.exp(x - max_x) for x in arr]
        sum_exps = sum(exps)
        return [e / sum_exps for e in exps]


# =============================================================================
# SECTION B: MULTILINGUAL LINGUISTIC ENCODING SYSTEM (MLES)
# =============================================================================

@dataclass
class CulturalEmbedding:
    """
    Represents a concept with its multilingual and cultural metadata.

    Each CulturalEmbedding encodes a concept with its linguistic
    realizations across multiple languages, each tagged with cultural
    and contextual information.
    """
    concept_id: str
    core_meaning: str
    linguistic_realizations: Dict[str, Dict[str, Any]]  # lang -> {term, register, connotation}
    cultural_tags: List[str]  # e.g., ["religious", "political", "informal"]
    concept_vector: Vector  # Semantic representation
    relational_ids: List[str]  # IDs of related concepts

    def get_realization(self, language: str) -> Optional[Dict[str, Any]]:
        return self.linguistic_realizations.get(language)

    def has_language(self, language: str) -> bool:
        return language in self.linguistic_realizations

    def to_dict(self) -> Dict[str, Any]:
        return {
            "concept_id": self.concept_id,
            "core_meaning": self.core_meaning,
            "linguistic_realizations": self.linguistic_realizations,
            "cultural_tags": self.cultural_tags,
            "concept_vector": self.concept_vector.to_dict(),
            "relational_ids": self.relational_ids,
        }


class MultilingualSemanticNetwork:
    """
    A semantic network that encodes concepts with multilingual
    and cultural information.

    This implements Cleopatra's practice of using language as
    intelligence — each concept carries not just its translation
    but its cultural context, connotation, and relational position.
    """

    def __init__(self):
        self._concepts: Dict[str, CulturalEmbedding] = {}
        self._language_order: List[str] = []  # Priority order for languages
        self._cultural_tag_index: Dict[str, List[str]] = {}  # tag -> concept_ids

    def add_concept(
        self,
        concept_id: str,
        core_meaning: str,
        linguistic_realizations: Dict[str, Dict[str, Any]],
        cultural_tags: Optional[List[str]] = None,
        concept_vector: Optional[Vector] = None,
        relational_ids: Optional[List[str]] = None,
    ) -> CulturalEmbedding:
        """Add a concept to the semantic network."""
        if concept_vector is None:
            # Create a random vector for the concept
            concept_vector = Vector.random(32, seed=self._string_seed(concept_id))

        embedding = CulturalEmbedding(
            concept_id=concept_id,
            core_meaning=core_meaning,
            linguistic_realizations=linguistic_realizations,
            cultural_tags=cultural_tags or [],
            concept_vector=concept_vector,
            relational_ids=relational_ids or [],
        )

        self._concepts[concept_id] = embedding

        for lang in linguistic_realizations:
            if lang not in self._language_order:
                self._language_order.append(lang)

        for tag in (cultural_tags or []):
            if tag not in self._cultural_tag_index:
                self._cultural_tag_index[tag] = []
            self._cultural_tag_index[tag].append(concept_id)

        return embedding

    def get_concept(self, concept_id: str) -> Optional[CulturalEmbedding]:
        return self._concepts.get(concept_id)

    def query_by_tag(self, tag: str) -> List[CulturalEmbedding]:
        concept_ids = self._cultural_tag_index.get(tag, [])
        return [self._concepts[cid] for cid in concept_ids if cid in self._concepts]

    def multilingual_retrieve(
        self,
        query: str,
        target_language: str,
        top_k: int = 5
    ) -> List[Tuple[CulturalEmbedding, float]]:
        """
        Retrieve concepts relevant to a query in a target language.

        Returns concepts sorted by relevance, each with a relevance score,
        with linguistic realizations in the target language.
        """
        query_vec = self._text_to_vector(query)
        scored = []
        for concept in self._concepts.values():
            sim = query_vec.cosine_similarity(concept.concept_vector)
            scored.append((concept, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def _text_to_vector(self, text: str) -> Vector:
        """Convert text to a vector via simple hash-based embedding."""
        words = text.lower().split()
        vec = Vector.zeros(32)
        for i, word in enumerate(words):
            seed = self._string_seed(word) % 10000
            rng = random.Random(seed)
            word_vec = Vector([rng.uniform(-1, 1) for _ in range(32)])
            vec = vec + word_vec
        return vec.normalize()

    @staticmethod
    def _string_seed(s: str) -> int:
        h = 0
        for c in s:
            h = h * 31 + ord(c)
        return abs(h)

    def add_default_cleopatra_concepts(self):
        """Initialize the network with concepts from Cleopatra's world."""
        # Political concepts
        self.add_concept(
            concept_id="legitimacy",
            core_meaning="Political authority derived from perceived right to rule",
            linguistic_realizations={
                "greek": {"term": "κῦδος", "register": "formal", "connotation": "glory and fame"},
                "egyptian": {"term": "maat", "register": "formal", "connotation": "truth, order, justice"},
                "roman": {"term": "auctoritas", "register": "formal", "connotation": "influence and prestige"},
            },
            cultural_tags=["political", "religious"],
            relational_ids=["power", "sovereignty"],
        )

        self.add_concept(
            concept_id="alliance",
            core_meaning="Strategic partnership between political actors",
            linguistic_realizations={
                "greek": {"term": "συμμαχία", "register": "formal", "connotation": "military alliance"},
                "egyptian": {"term": "shemsu", "register": "formal", "connotation": "follower/devotee"},
                "hebrew": {"term": "berith", "register": "formal", "connotation": "covenant"},
            },
            cultural_tags=["political", "diplomatic"],
            relational_ids=["power", "legitimacy"],
        )

        self.add_concept(
            concept_id="sovereignty",
            core_meaning=" supreme political authority over a territory",
            linguistic_realizations={
                "greek": {"term": "κυριότης", "register": "formal", "connotation": "lordly authority"},
                "egyptian": {"term": "nesu", "register": "formal", "connotation": "king/fire-breather"},
            },
            cultural_tags=["political"],
            relational_ids=["legitimacy", "power"],
        )

        self.add_concept(
            concept_id="intelligence",
            core_meaning="Information gathering and assessment for political advantage",
            linguistic_realizations={
                "greek": {"term": "σοφία", "register": "formal", "connotation": "wisdom"},
                "latin": {"term": "prudentia", "register": "formal", "connotation": "foresight"},
            },
            cultural_tags=["political", "cognitive"],
            relational_ids=["wisdom", "strategy"],
        )

        self.add_concept(
            concept_id="persuasion",
            core_meaning="The art of convincing others to adopt one's perspective",
            linguistic_realizations={
                "greek": {"term": "πειθώ", "register": "formal", "connotation": "persuasion/social influence"},
                "latin": {"term": "eloquentia", "register": "formal", "connotation": "artful speech"},
            },
            cultural_tags=["cognitive", "political"],
            relational_ids=["intelligence", "alliance"],
        )

        self.add_concept(
            concept_id="isis",
            core_meaning="The goddess Isis, patroness of the Ptolemaic dynasty",
            linguistic_realizations={
                "greek": {"term": "Ἶσις", "register": "religious", "connotation": "divine mother"},
                "egyptian": {"term": "auset", "register": "religious", "connotation": "throne goddess"},
            },
            cultural_tags=["religious", "political", "egyptian"],
            relational_ids=["cleopatra", "legitimacy"],
        )

        self.add_concept(
            concept_id="caesar",
            core_meaning="Julius Caesar, Roman dictator and Cleopatra's ally",
            linguistic_realizations={
                "greek": {"term": "Καῖσαρ", "register": "formal", "connotation": "Roman ruler"},
                "latin": {"term": "Caesar", "register": "formal", "connotation": "title of imperial authority"},
            },
            cultural_tags=["political", "roman"],
            relational_ids=["alliance", "sovereignty"],
        )

        self.add_concept(
            concept_id="antony",
            core_meaning="Mark Antony, Roman triumvir and Cleopatra's partner",
            linguistic_realizations={
                "greek": {"term": "Ἀντώνιος", "register": "formal", "connotation": "Roman general"},
                "latin": {"term": "Antonius", "register": "formal", "connotation": "noble Roman name"},
            },
            cultural_tags=["political", "roman"],
            relational_ids=["alliance", "sovereignty"],
        )

        self.add_concept(
            concept_id="alexandria",
            core_meaning="The capital city of Ptolemaic Egypt, center of learning",
            linguistic_realizations={
                "greek": {"term": "Ἀλεξάνδρεια", "register": "formal", "connotation": "city of Alexander"},
                "egyptian": {"term": "rou-nut", "register": "formal", "connotation": "circle of the sun"},
            },
            cultural_tags=["political", "cultural", "egyptian"],
            relational_ids=["cleopatra", "library"],
        )

        self.add_concept(
            concept_id="library",
            core_meaning="The Great Library of Alexandria, seat of scholarship",
            linguistic_realizations={
                "greek": {"term": "βιβλιοθήκη", "register": "scholarly", "connotation": "repository of knowledge"},
            },
            cultural_tags=["cultural", "intellectual"],
            relational_ids=["alexandria", "wisdom"],
        )

    def __len__(self) -> int:
        return len(self._concepts)

    def __repr__(self) -> str:
        return f"MultilingualSemanticNetwork({len(self)} concepts, {len(self._language_order)} languages)"


class MultilingualLinguisticEncoder:
    """
    The Linguistic Encoding Layer of the Cleopatra Cognitive Architecture.

    Encodes information in ways that preserve cultural nuance and enable
    reasoning across linguistic boundaries. Implements Cleopatra's insight
    that language is power — that multilingualism is not merely a matter
    of translation but of cognitive reach.
    """

    def __init__(self):
        self.semantic_network = MultilingualSemanticNetwork()
        self.semantic_network.add_default_cleopatra_concepts()
        self._language_priorities = ["greek", "egyptian", "hebrew", "latin", "arabic"]
        self._encoding_history: List[Dict[str, Any]] = []

    def encode(
        self,
        text: str,
        source_language: str,
        target_language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Encode text with cultural-linguistic metadata.

        Returns a structured encoding with:
        - The core semantic content (concept vector)
        - Linguistic realizations in available languages
        - Cultural context tags
        - Register information
        """
        encoding = {
            "source_language": source_language,
            "source_text": text,
            "concept_vector": self.semantic_network._text_to_vector(text),
            "retrieved_concepts": [],
            "cultural_tags": [],
            "cross_lingual_signals": {},
        }

        # Retrieve relevant concepts
        results = self.semantic_network.multilingual_retrieve(
            text, target_language or source_language, top_k=5
        )
        for concept, score in results:
            encoding["retrieved_concepts"].append({
                "concept_id": concept.concept_id,
                "core_meaning": concept.core_meaning,
                "relevance_score": score,
            })
            encoding["cultural_tags"].extend(concept.cultural_tags)

        # Extract linguistic signals
        encoding["cross_lingual_signals"] = self._extract_linguistic_signals(text)

        self._encoding_history.append(encoding)
        return encoding

    def _extract_linguistic_signals(self, text: str) -> Dict[str, Any]:
        """Extract political and cultural signals from linguistic choices."""
        signals = {
            "hierarchy_markers": 0,
            "deference_markers": 0,
            "cultural_references": 0,
            "register_level": "unknown",
        }

        # Simple heuristic markers
        text_lower = text.lower()
        formal_markers = ["θεός", "κύριος", "domine", "pharaoh", "majesty", "your"]
        signals["hierarchy_markers"] = sum(1 for m in formal_markers if m in text_lower)

        deference_words = ["please", "humbly", "graciously", "serenity", "excellence"]
        signals["deference_markers"] = sum(1 for w in deference_words if w in text_lower)

        cultural_refs = ["isis", "osiris", "apollo", "athena", "jupiter", "nile", "pharos"]
        signals["cultural_references"] = sum(1 for r in cultural_refs if r in text_lower)

        if signals["hierarchy_markers"] > 2:
            signals["register_level"] = "formal"
        elif signals["deference_markers"] > 1:
            signals["register_level"] = "polite"
        else:
            signals["register_level"] = "neutral"

        return signals

    def translate_with_cultural_context(
        self,
        text: str,
        source_language: str,
        target_language: str
    ) -> Dict[str, Any]:
        """
        Translate text with cultural context preservation.

        Unlike simple translation, this method preserves cultural nuance
        by identifying relevant concepts and their cultural metadata.
        """
        encoding = self.encode(text, source_language, target_language)

        translations = {}
        for concept_data in encoding["retrieved_concepts"]:
            concept_id = concept_data["concept_id"]
            concept = self.semantic_network.get_concept(concept_id)
            if concept and concept.has_language(target_language):
                realization = concept.get_realization(target_language)
                translations[concept_id] = {
                    "term": realization["term"],
                    "connotation": realization["connotation"],
                    "register": realization["register"],
                }

        return {
            "original": text,
            "source_language": source_language,
            "target_language": target_language,
            "translations": translations,
            "cultural_tags": encoding["cultural_tags"],
            "cross_lingual_signals": encoding["cross_lingual_signals"],
        }

    def assess_cultural_alignment(
        self,
        text: str,
        target_culture: str
    ) -> Dict[str, Any]:
        """
        Assess how well a text aligns with a target cultural framework.

        This implements Cleopatra's practice of using language to detect
        political undercurrents and cultural alignment.
        """
        encoding = self.encode(text, "greek")
        culture_concepts = self.semantic_network.query_by_tag(target_culture)

        alignment_score = 0.0
        matched_concepts = []
        for concept in culture_concepts:
            for retrieved in encoding["retrieved_concepts"]:
                if retrieved["concept_id"] == concept.concept_id:
                    alignment_score += retrieved["relevance_score"]
                    matched_concepts.append(concept.concept_id)

        return {
            "text": text,
            "target_culture": target_culture,
            "alignment_score": alignment_score,
            "matched_concepts": matched_concepts,
            "cultural_tags_found": encoding["cultural_tags"],
        }

    def get_language_diversity(self) -> Dict[str, int]:
        """Return the diversity of languages in the semantic network."""
        lang_counts: Dict[str, int] = {}
        for concept in self.semantic_network._concepts.values():
            for lang in concept.linguistic_realizations:
                lang_counts[lang] = lang_counts.get(lang, 0) + 1
        return lang_counts

    def summary(self) -> Dict[str, Any]:
        return {
            "total_concepts": len(self.semantic_network),
            "language_diversity": self.get_language_diversity(),
            "encoding_history_size": len(self._encoding_history),
        }


# =============================================================================
# SECTION C: CULTURAL INTELLIGENCE LAYER (CIL)
# =============================================================================

@dataclass
class CulturalProfile:
    """
    A model of a population's cultural values, norms, and expectations.

    Implements Cleopatra's practice of cultural intelligence — understanding
    what different populations value, how they understand authority, and
    how they make decisions.
    """
    population_id: str
    population_name: str
    core_values: List[str]
    legitimacy_criteria: Dict[str, float]  # criterion -> weight
    communication_norms: Dict[str, Any]  # register preferences, framing patterns
    decision_making_pattern: Dict[str, Any]  # consensus processes, veto players
    historical_references: List[str]  # Cultural memories shaping worldview
    authority_expectations: Dict[str, Any]  # How authority is understood
    representation_vector: Vector  # High-dimensional cultural profile

    def values_similarity(self, other: "CulturalProfile") -> float:
        """Compute similarity between two cultural profiles."""
        return self.representation_vector.cosine_similarity(other.representation_vector)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "population_id": self.population_id,
            "population_name": self.population_name,
            "core_values": self.core_values,
            "legitimacy_criteria": self.legitimacy_criteria,
            "communication_norms": self.communication_norms,
            "decision_making_pattern": self.decision_making_pattern,
            "historical_references": self.historical_references,
            "authority_expectations": self.authority_expectations,
            "representation_vector": self.representation_vector.to_dict(),
        }


class CulturalIntelligenceLayer:
    """
    The Cultural Intelligence Layer of the Cleopatra Cognitive Architecture.

    Maintains cultural models — detailed representations of the values,
    norms, and expectations of different populations. Implements Cleopatra's
    insight that different populations have different definitions of
    legitimacy, authority, and proper governance.
    """

    # Pre-configured cultural profiles for Cleopatra's world
    BUILT_IN_PROFILES = {
        "egyptian_priesthood": {
            "population_name": "Egyptian Priesthood",
            "core_values": ["maat", "order", "divine_purity", "ancestral_tradition", "stability"],
            "legitimacy_criteria": {
                "divine_sanction": 0.35,
                "ritual_observance": 0.25,
                "ancestral_connection": 0.20,
                "material_prosperity": 0.10,
                "military_strength": 0.10,
            },
            "communication_norms": {
                "preferred_register": "formal_religious",
                "framing_pattern": "traditional",
                "directness": "moderate",
            },
            "decision_making_pattern": {
                "type": "consensus_council",
                "leadership_role": "advisory",
                "veto_players": ["high_priest", "council_of_magi"],
            },
            "historical_references": ["building_of_pyramids", "osiris_cycle", "maat_principle"],
            "authority_expectations": {
                "source": "divine_mandate",
                "presentation": "ritualized",
                "gender_constraints": "none",
            },
        },
        "greek_elite": {
            "population_name": "Greek Urban Elite",
            "core_values": ["philosophia", "paideia", "democracy", "athletic_excellence", "cultural_refinement"],
            "legitimacy_criteria": {
                "cultural_sophistication": 0.30,
                "ancestral_nobility": 0.25,
                "athletic_wisdom": 0.20,
                "material_prosperity": 0.15,
                "military_leadership": 0.10,
            },
            "communication_norms": {
                "preferred_register": "philosophical_formal",
                "framing_pattern": "classical_reference",
                "directness": "high",
            },
            "decision_making_pattern": {
                "type": "epistocratic_deliberation",
                "leadership_role": "first_among_equals",
                "veto_players": ["gymasiarch", "city_council"],
            },
            "historical_references": ["persian_wars", "age_of_pericles", "alexander_conquests"],
            "authority_expectations": {
                "source": "cultural_excellence",
                "presentation": "philosophical",
                "gender_constraints": "masculine_norm",
            },
        },
        "roman_senatorial": {
            "population_name": "Roman Senatorial Class",
            "core_values": ["virtus", "pietas", "dignitas", "fides", "austeritas"],
            "legitimacy_criteria": {
                "ancestral_dignitas": 0.30,
                "military_virtus": 0.25,
                "republican_service": 0.20,
                "economic_integrity": 0.15,
                "religious_piety": 0.10,
            },
            "communication_norms": {
                "preferred_register": "ciceronian_formal",
                "framing_pattern": "republican_values",
                "directness": "high",
            },
            "decision_making_pattern": {
                "type": "senatorial_debate",
                "leadership_role": "princeps",
                "veto_players": ["consul", "tribune", "augurs"],
            },
            "historical_references": ["founding_of_rome", "republican_wars", "cato_tradition"],
            "authority_expectations": {
                "source": "republican_consecration",
                "presentation": "collegial",
                "gender_constraints": "strictly_masculine",
            },
        },
        "judean_client": {
            "population_name": "Jewish Communities (Judea)",
            "core_values": ["yhwh_monolatry", "torah_observance", "sabbath_rest", "covenant_faithfulness", "land_promise"],
            "legitimacy_criteria": {
                "covenant_faithfulness": 0.35,
                "torah_observance": 0.30,
                "prophetic_alignment": 0.20,
                "material_blessing": 0.10,
                "military_protection": 0.05,
            },
            "communication_norms": {
                "preferred_register": "prophetic_serious",
                "framing_pattern": "scriptural_reference",
                "directness": "moderate",
            },
            "decision_making_pattern": {
                "type": "rabbinic_counsel",
                "leadership_role": "temporal",
                "veto_players": ["high_priest", "sanhedrin"],
            },
            "historical_references": ["exodus_from_egypt", "davidic_covenant", "babylonian_exile"],
            "authority_expectations": {
                "source": "covenant_relationship",
                "presentation": "theocratic",
                "gender_constraints": "gendered_roles",
            },
        },
        "nubian_southern": {
            "population_name": "Nubian/Kushite Populations",
            "core_values": ["kemet_ traditions", "royal_divinity", "kingly_virtue", "military_prowess", "trade_connectivity"],
            "legitimacy_criteria": {
                "royal_divinity": 0.35,
                "military_prowess": 0.25,
                "kemet_alignment": 0.20,
                "trade_network_access": 0.10,
                "priestly_blessing": 0.10,
            },
            "communication_norms": {
                "preferred_register": "royal_formal",
                "framing_pattern": "divine_mandate",
                "directness": "moderate_low",
            },
            "decision_making_pattern": {
                "type": "royal_council",
                "leadership_role": "autocratic_divine",
                "veto_players": ["queen_mother", "military_commander"],
            },
            "historical_references": ["kushite_kingdom", "napatan_tradition", "egyptian_conquest"],
            "authority_expectations": {
                "source": "divine_royal_mandate",
                "presentation": "splendid",
                "gender_constraints": "none",
            },
        },
    }

    def __init__(self):
        self._cultural_profiles: Dict[str, CulturalProfile] = {}
        self._adaptation_history: List[Dict[str, Any]] = []
        self._initialize_built_in_profiles()

    def _initialize_built_in_profiles(self):
        """Initialize the built-in cultural profiles."""
        for pop_id, config in self.BUILT_IN_PROFILES.items():
            self.add_profile(
                population_id=pop_id,
                population_name=config["population_name"],
                core_values=config["core_values"],
                legitimacy_criteria=config["legitimacy_criteria"],
                communication_norms=config["communication_norms"],
                decision_making_pattern=config["decision_making_pattern"],
                historical_references=config["historical_references"],
                authority_expectations=config["authority_expectations"],
            )

    def add_profile(
        self,
        population_id: str,
        population_name: str,
        core_values: List[str],
        legitimacy_criteria: Dict[str, float],
        communication_norms: Dict[str, Any],
        decision_making_pattern: Dict[str, Any],
        historical_references: Optional[List[str]] = None,
        authority_expectations: Optional[Dict[str, Any]] = None,
        representation_vector: Optional[Vector] = None,
    ) -> CulturalProfile:
        """Add a new cultural profile to the layer."""
        if representation_vector is None:
            # Create a representation vector based on the profile's attributes
            vec_components = []
            for val in core_values:
                seed = MultilingualSemanticNetwork._string_seed(val)
                rng = random.Random(seed)
                vec_components.append(Vector([rng.uniform(-1, 1) for _ in range(32)]))
            representation_vector = vec_components[0] if vec_components else Vector.zeros(32)
            for v in vec_components[1:]:
                representation_vector = representation_vector + v
            representation_vector = representation_vector.normalize()

        profile = CulturalProfile(
            population_id=population_id,
            population_name=population_name,
            core_values=core_values,
            legitimacy_criteria=legitimacy_criteria,
            communication_norms=communication_norms,
            decision_making_pattern=decision_making_pattern,
            historical_references=historical_references or [],
            authority_expectations=authority_expectations or {},
            representation_vector=representation_vector,
        )

        self._cultural_profiles[population_id] = profile
        return profile

    def get_profile(self, population_id: str) -> Optional[CulturalProfile]:
        return self._cultural_profiles.get(population_id)

    def compute_legitimacy_for_profile(
        self,
        authority_claims: Dict[str, Any],
        population_id: str
    ) -> Dict[str, Any]:
        """
        Compute how well a set of authority claims would be received
        by a specific population.

        Implements Cleopatra's practice of tailoring her presentation
        to different audiences based on their legitimacy criteria.
        """
        profile = self._cultural_profiles.get(population_id)
        if not profile:
            return {"error": "Unknown population"}

        criteria_scores = {}
        weighted_score = 0.0

        for criterion, weight in profile.legitimacy_criteria.items():
            # Score the authority claims on this criterion
            claim_value = authority_claims.get(criterion, 0.0)
            criteria_scores[criterion] = {
                "weight": weight,
                "claim_score": claim_value,
                "contribution": claim_value * weight,
            }
            weighted_score += claim_value * weight

        return {
            "population_id": population_id,
            "overall_legitimacy_score": weighted_score,
            "criteria_breakdown": criteria_scores,
            "recommendation": self._generate_legitimacy_recommendation(profile, criteria_scores),
        }

    def _generate_legitimacy_recommendation(
        self,
        profile: CulturalProfile,
        criteria_scores: Dict[str, Any]
    ) -> Dict[str, str]:
        """Generate recommendations for improving legitimacy with a population."""
        recommendations = {}
        for criterion, scores in criteria_scores.items():
            if scores["claim_score"] < 0.5:
                # Identify the gap
                gap = 0.5 - scores["claim_score"]
                recommendations[criterion] = f"Strengthen {criterion} display (gap: {gap:.2f})"
        return recommendations

    def adapt_output_for_culture(
        self,
        output: Dict[str, Any],
        target_population_id: str
    ) -> Dict[str, Any]:
        """
        Adapt a system's output for a specific cultural context.

        This is the core method of the Cultural Intelligence Layer —
        taking a output and tailoring it to resonate with a
        specific population's values and expectations.
        """
        profile = self._cultural_profiles.get(target_population_id)
        if not profile:
            return output

        adapted = copy.deepcopy(output)

        # Adjust framing based on communication norms
        register = profile.communication_norms.get("preferred_register", "neutral")
        adapted["cultural_framing"] = {
            "register": register,
            "framing_pattern": profile.communication_norms.get("framing_pattern", "neutral"),
            "population_id": target_population_id,
        }

        # Add legitimacy context
        adapted["legitimacy_context"] = {
            "expected_source": profile.authority_expectations.get("source", "unknown"),
            "presentation_style": profile.authority_expectations.get("presentation", "neutral"),
        }

        # Record adaptation
        self._adaptation_history.append({
            "target_population": target_population_id,
            "original_output_keys": list(output.keys()),
            "adaptation_applied": list(adapted.keys()),
        })

        return adapted

    def assess_cultural_distance(
        self,
        population_a: str,
        population_b: str
    ) -> float:
        """Assess the cultural distance between two populations."""
        profile_a = self._cultural_profiles.get(population_a)
        profile_b = self._cultural_profiles.get(population_b)
        if not profile_a or not profile_b:
            return -1.0

        similarity = profile_a.values_similarity(profile_b)
        return 1.0 - similarity  # Distance is 1 - similarity

    def identify_bridges(
        self,
        population_a: str,
        population_b: str
    ) -> List[str]:
        """
        Identify concepts or values that could serve as bridges
        between two culturally distant populations.

        Implements Cleopatra's practice of finding common ground
        across cultural boundaries.
        """
        profile_a = self._cultural_profiles.get(population_a)
        profile_b = self._cultural_profiles.get(population_b)
        if not profile_a or not profile_b:
            return []

        bridges = []
        for value in profile_a.core_values:
            if value in profile_b.core_values:
                bridges.append(value)

        # Also check for shared historical references
        for ref in profile_a.historical_references:
            if ref in profile_b.historical_references:
                bridges.append(f"historical:{ref}")

        return bridges

    def all_profiles(self) -> List[CulturalProfile]:
        return list(self._cultural_profiles.values())

    def summary(self) -> Dict[str, Any]:
        return {
            "total_profiles": len(self._cultural_profiles),
            "adaptation_history_size": len(self._adaptation_history),
            "profile_ids": list(self._cultural_profiles.keys()),
        }


# =============================================================================
# SECTION D: LEGITIMACY CONSTRUCTION LAYER (LCL)
# =============================================================================

@dataclass
class LegitimacyStatus:
    """
    Tracks the legitimacy status of an authority with different populations.

    Implements Cleopatra's insight that legitimacy must be actively
    constructed and maintained, not merely assumed.
    """
    population_id: str
    performance_score: float      # Based on demonstrated competence
    procedural_score: float      # Based on fairness of processes
    cultural_score: float        # Based on cultural engagement
    interpersonal_score: float   # Based on personal trust
    aggregate_score: float       # Weighted composite
    is_legitimate: bool          # True if aggregate exceeds threshold
    threats: List[str]           # Active threats to legitimacy
    trace: List[str]             # Reasoning trace

    def to_dict(self) -> Dict[str, Any]:
        return {
            "population_id": self.population_id,
            "performance_score": self.performance_score,
            "procedural_score": self.procedural_score,
            "cultural_score": self.cultural_score,
            "interpersonal_score": self.interpersonal_score,
            "aggregate_score": self.aggregate_score,
            "is_legitimate": self.is_legitimate,
            "threats": self.threats,
            "trace": self.trace,
        }


class LegitimacyConstructionLayer:
    """
    The Legitimacy Construction Layer of the Cleopatra Cognitive Architecture.

    Establishes and maintains the legitimacy of the system's authority in the
    eyes of different populations. Cleopatra understood that legitimacy was
    not a given but a construction — achieved through multiple channels and
    maintained through ongoing effort.
    """

    # Weights for the four channels of legitimacy
    CHANNEL_WEIGHTS = {
        "performance": 0.30,
        "procedural": 0.25,
        "cultural": 0.25,
        "interpersonal": 0.20,
    }

    LEGITIMACY_THRESHOLD = 0.6  # Minimum aggregate score for legitimacy

    def __init__(self, cultural_layer: Optional[CulturalIntelligenceLayer] = None):
        self._cultural_layer = cultural_layer or CulturalIntelligenceLayer()
        self._status_by_population: Dict[str, LegitimacyStatus] = {}
        self._legitimacy_history: List[Dict[str, Any]] = []

    def assess_legitimacy(
        self,
        authority_actions: Dict[str, Any],
        population_id: str,
    ) -> LegitimacyStatus:
        """
        Assess the legitimacy of the system in the eyes of a population.

        Args:
            authority_actions: Dictionary of actions/decisions taken
            population_id: The population whose perspective to assess
        """
        profile = self._cultural_layer.get_profile(population_id)
        if not profile:
            return LegitimacyStatus(
                population_id=population_id,
                performance_score=0.0,
                procedural_score=0.0,
                cultural_score=0.0,
                interpersonal_score=0.0,
                aggregate_score=0.0,
                is_legitimate=False,
                threats=["Unknown population"],
                trace=["Population not found in cultural profiles"],
            )

        trace = [f"Assessing legitimacy for population: {population_id}"]

        # Score each channel
        performance_score = self._score_performance(authority_actions, profile)
        trace.append(f"Performance score: {performance_score:.3f}")

        procedural_score = self._score_procedural(authority_actions, profile)
        trace.append(f"Procedural score: {procedural_score:.3f}")

        cultural_score = self._score_cultural(authority_actions, profile)
        trace.append(f"Cultural score: {cultural_score:.3f}")

        interpersonal_score = self._score_interpersonal(authority_actions, profile)
        trace.append(f"Interpersonal score: {interpersonal_score:.3f}")

        # Compute weighted aggregate
        aggregate = (
            self.CHANNEL_WEIGHTS["performance"] * performance_score +
            self.CHANNEL_WEIGHTS["procedural"] * procedural_score +
            self.CHANNEL_WEIGHTS["cultural"] * cultural_score +
            self.CHANNEL_WEIGHTS["interpersonal"] * interpersonal_score
        )
        trace.append(f"Aggregate legitimacy: {aggregate:.3f}")

        # Identify threats
        threats = self._identify_threats(
            aggregate, performance_score, procedural_score,
            cultural_score, interpersonal_score, profile
        )
        trace.append(f"Identified threats: {threats}")

        is_legitimate = aggregate >= self.LEGITIMACY_THRESHOLD
        trace.append(f"Is legitimate: {is_legitimate}")

        status = LegitimacyStatus(
            population_id=population_id,
            performance_score=performance_score,
            procedural_score=procedural_score,
            cultural_score=cultural_score,
            interpersonal_score=interpersonal_score,
            aggregate_score=aggregate,
            is_legitimate=is_legitimate,
            threats=threats,
            trace=trace,
        )

        self._status_by_population[population_id] = status
        self._legitimacy_history.append({
            "population_id": population_id,
            "aggregate_score": aggregate,
            "is_legitimate": is_legitimate,
        })

        return status

    def _score_performance(
        self,
        actions: Dict[str, Any],
        profile: CulturalProfile
    ) -> float:
        """Score performance legitimacy: did the system perform its tasks well?"""
        # Look for performance metrics in actions
        effective_actions = actions.get("effective_actions", 0)
        total_actions = actions.get("total_actions", 1)
        task_success = actions.get("task_success_rate", 0.5)

        # Performance is a combination of effectiveness and success rate
        effectiveness = effective_actions / max(total_actions, 1)
        score = 0.5 * effectiveness + 0.5 * task_success

        return max(0.0, min(1.0, score))

    def _score_procedural(
        self,
        actions: Dict[str, Any],
        profile: CulturalProfile
    ) -> float:
        """Score procedural legitimacy: are processes fair and transparent?"""
        fairness = actions.get("process_fairness", 0.5)
        transparency = actions.get("process_transparency", 0.5)
        consistency = actions.get("decision_consistency", 0.5)

        return (0.4 * fairness + 0.3 * transparency + 0.3 * consistency)

    def _score_cultural(
        self,
        actions: Dict[str, Any],
        profile: CulturalProfile
    ) -> float:
        """Score cultural legitimacy: does the system respect cultural traditions?"""
        # Check cultural engagement signals
        cultural_engagements = actions.get("cultural_engagements", [])
        cultural_score = min(1.0, len(cultural_engagements) / 5.0)

        # Check alignment with population's communication norms
        register_used = actions.get("register_used", "neutral")
        expected_register = profile.communication_norms.get("preferred_register", "neutral")
        register_match = 1.0 if register_used == expected_register else 0.6

        return 0.6 * cultural_score + 0.4 * register_match

    def _score_interpersonal(
        self,
        actions: Dict[str, Any],
        profile: CulturalProfile
    ) -> float:
        """Score interpersonal legitimacy: does the system build trust?"""
        trust_score = actions.get("trust_build_score", 0.5)
        responsiveness = actions.get("responsiveness_score", 0.5)
        reliability = actions.get("reliability_score", 0.5)

        return (0.4 * trust_score + 0.3 * responsiveness + 0.3 * reliability)

    def _identify_threats(
        self,
        aggregate: float,
        performance: float,
        procedural: float,
        cultural: float,
        interpersonal: float,
        profile: CulturalProfile
    ) -> List[str]:
        """Identify specific threats to legitimacy."""
        threats = []
        if aggregate < self.LEGITIMACY_THRESHOLD:
            threats.append(f"Aggregate score below threshold ({aggregate:.3f} < {self.LEGITIMACY_THRESHOLD})")
        if performance < 0.4:
            threats.append("Poor task performance")
        if procedural < 0.4:
            threats.append("Unfair or opaque decision processes")
        if cultural < 0.4:
            threats.append("Insufficient cultural engagement")
        if interpersonal < 0.4:
            threats.append("Failure to build personal trust")

        # Population-specific threats from authority expectations
        expected_source = profile.authority_expectations.get("source", "unknown")
        if expected_source == "divine_mandate" and cultural < 0.5:
            threats.append("Missing divine/sacred framing")
        elif expected_source == "cultural_excellence" and cultural < 0.5:
            threats.append("Insufficient cultural sophistication display")

        return threats

    def construct_legitimacy_plan(
        self,
        status: LegitimacyStatus,
        target_score: float = 0.8
    ) -> Dict[str, Any]:
        """
        Construct a plan for improving legitimacy with a population.

        Returns a structured plan with specific actions for each channel.
        """
        profile = self._cultural_layer.get_profile(status.population_id)
        if not profile:
            return {"error": "Unknown population"}

        gap = target_score - status.aggregate_score
        plan = {
            "current_score": status.aggregate_score,
            "target_score": target_score,
            "improvement_needed": gap,
            "channels": {},
        }

        # Calculate required improvement per channel
        for channel in ["performance", "procedural", "cultural", "interpersonal"]:
            current = getattr(status, f"{channel}_score")
            weight = self.CHANNEL_WEIGHTS[channel]
            channel_gap = gap / weight if weight > 0 else 0
            required = min(1.0, current + channel_gap)

            plan["channels"][channel] = {
                "current": current,
                "target": required,
                "gap": required - current,
                "recommendations": self._channel_recommendations(channel, current, required, profile),
            }

        return plan

    def _channel_recommendations(
        self,
        channel: str,
        current: float,
        target: float,
        profile: CulturalProfile
    ) -> List[str]:
        """Generate specific recommendations for improving a channel."""
        recommendations = []
        gap = target - current

        if gap <= 0:
            return ["Channel is at target — maintain current approach"]

        if channel == "performance":
            if current < 0.5:
                recommendations.append("Demonstrate competence through visible task completion")
                recommendations.append("Document successful outcomes and error recovery")
            recommendations.append("Establish track record of reliability")

        elif channel == "procedural":
            if current < 0.5:
                recommendations.append("Make decision-making criteria explicit")
                recommendations.append("Provide clear explanations for consequential decisions")
            recommendations.append("Apply consistent standards across all cases")

        elif channel == "cultural":
            expected_register = profile.communication_norms.get("preferred_register", "neutral")
            recommendations.append(f"Adopt {expected_register} register in communications")
            recommendations.append("Reference population's historical traditions and values")
            recommendations.append("Engage with cultural institutions and leaders")

        elif channel == "interpersonal":
            recommendations.append("Build direct relationships with key community figures")
            recommendations.append("Demonstrate responsiveness to individual concerns")
            recommendations.append("Maintain consistent engagement over time")

        return recommendations

    def get_status(self, population_id: str) -> Optional[LegitimacyStatus]:
        return self._status_by_population.get(population_id)

    def all_statuses(self) -> List[LegitimacyStatus]:
        return list(self._status_by_population.values())

    def summary(self) -> Dict[str, Any]:
        return {
            "total_populations_assessed": len(self._status_by_population),
            "legitimate_count": sum(1 for s in self._status_by_population.values() if s.is_legitimate),
            "history_size": len(self._legitimacy_history),
        }


# =============================================================================
# SECTION E: STRATEGIC REASONING LAYER (SRL)
# =============================================================================

@dataclass
class StrategicObjective:
    """A strategic objective with priority and constraints."""
    objective_id: str
    description: str
    priority: float          # 0.0 to 1.0
    constraints: List[str]   # Limiting conditions
    stakeholder_ids: List[str]  # Populations affected
    progress: float         # 0.0 to 1.0
    trace: List[str]        # Reasoning trace

    def to_dict(self) -> Dict[str, Any]:
        return {
            "objective_id": self.objective_id,
            "description": self.description,
            "priority": self.priority,
            "constraints": self.constraints,
            "stakeholder_ids": self.stakeholder_ids,
            "progress": self.progress,
            "trace": self.trace,
        }


@dataclass
class CoalitionProposal:
    """A proposal for cooperation between political actors."""
    proposer_id: str
    partner_id: str
    mutual_benefits: List[str]
    constraints: List[str]
    alignment_score: float  # How well aligned are the parties
    viability: float       # How feasible is the coalition
    trace: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposer_id": self.proposer_id,
            "partner_id": self.partner_id,
            "mutual_benefits": self.mutual_benefits,
            "constraints": self.constraints,
            "alignment_score": self.alignment_score,
            "viability": self.viability,
            "trace": self.trace,
        }


class StrategicReasoningLayer:
    """
    The Strategic Reasoning Layer of the Cleopatra Cognitive Architecture.

    Implements Cleopatra's approach to strategic flexibility — the ability
    to formulate plans, assess consequences, and adapt as circumstances
    change. Cleopatra's career demonstrates that strategic wisdom lies
    in constancy of ends and flexibility of means.
    """

    def __init__(
        self,
        cultural_layer: Optional[CulturalIntelligenceLayer] = None,
        legitimacy_layer: Optional[LegitimacyConstructionLayer] = None,
    ):
        self._cultural_layer = cultural_layer or CulturalIntelligenceLayer()
        self._legitimacy_layer = legitimacy_layer or LegitimacyConstructionLayer(self._cultural_layer)
        self._strategic_plan_history: List[Dict[str, Any]] = []
        self._active_objectives: Dict[str, StrategicObjective] = {}

    def formulate_objective(
        self,
        description: str,
        priority: float,
        constraints: Optional[List[str]] = None,
        stakeholder_ids: Optional[List[str]] = None,
    ) -> StrategicObjective:
        """Create a new strategic objective."""
        objective_id = f"obj_{len(self._active_objectives) + 1}_{int(time.time())}"
        trace = [
            f"Formulating objective: {description}",
            f"Priority: {priority}",
            f"Constraints: {constraints or []}",
        ]

        objective = StrategicObjective(
            objective_id=objective_id,
            description=description,
            priority=priority,
            constraints=constraints or [],
            stakeholder_ids=stakeholder_ids or [],
            progress=0.0,
            trace=trace,
        )

        self._active_objectives[objective_id] = objective
        return objective

    def assess_situation(
        self,
        intelligence_reports: List[Dict[str, Any]],
        active_objective_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Assess a strategic situation given intelligence reports.

        Implements Cleopatra's practice of using intelligence networks
        to maintain situational awareness across the eastern Mediterranean.
        """
        assessment = {
            "timestamp": time.time(),
            "intelligence_sources": len(intelligence_reports),
            "situation_score": 0.0,
            "key_factors": [],
            "risks": [],
            "opportunities": [],
            "strategic_recommendations": [],
        }

        if not intelligence_reports:
            assessment["risks"].append("No intelligence reports available")
            return assessment

        # Aggregate intelligence
        favorable_signals = 0
        unfavorable_signals = 0
        neutral_signals = 0

        for report in intelligence_reports:
            signal = report.get("signal", "neutral")
            if signal == "favorable":
                favorable_signals += 1
            elif signal == "unfavorable":
                unfavorable_signals += 1
            else:
                neutral_signals += 1

            for factor in report.get("factors", []):
                assessment["key_factors"].append(factor)

            for risk in report.get("risks", []):
                assessment["risks"].append(risk)

            for opp in report.get("opportunities", []):
                assessment["opportunities"].append(opp)

        # Compute overall situation score
        total_signals = favorable_signals + unfavorable_signals + neutral_signals
        if total_signals > 0:
            assessment["situation_score"] = (
                0.5 +
                (favorable_signals - unfavorable_signals) / (total_signals * 2)
            )

        # Generate recommendations based on active objective
        if active_objective_id and active_objective_id in self._active_objectives:
            objective = self._active_objectives[active_objective_id]
            assessment["strategic_recommendations"] = self._generate_recommendations(
                assessment, objective
            )

        return assessment

    def _generate_recommendations(
        self,
        assessment: Dict[str, Any],
        objective: StrategicObjective
    ) -> List[str]:
        """Generate strategic recommendations based on situation and objective."""
        recommendations = []

        situation_score = assessment["situation_score"]

        if situation_score < 0.4:
            recommendations.append("Situation unfavorable — consider defensive posture")
            recommendations.append("Seek additional intelligence before committing resources")
            if objective.constraints:
                recommendations.append(f"Review constraints: {', '.join(objective.constraints)}")
        elif situation_score > 0.7:
            recommendations.append("Situation favorable — opportunity for expansion")
            recommendations.append("Consider accelerating timeline for objective")
        else:
            recommendations.append("Situation mixed — maintain current approach with monitoring")

        # Consider stakeholder positions
        for stakeholder_id in objective.stakeholder_ids:
            status = self._legitimacy_layer.get_status(stakeholder_id)
            if status and not status.is_legitimate:
                recommendations.append(f"Strengthen legitimacy with {stakeholder_id}")

        return recommendations

    def build_coalition(
        self,
        primary_actor_id: str,
        potential_partners: List[str],
        objective: StrategicObjective,
    ) -> List[CoalitionProposal]:
        """
        Build coalition proposals with potential partners for an objective.

        Implements Cleopatra's practice of strategic alliance-building —
        identifying common interests and constructing mutually beneficial
        arrangements.
        """
        proposals = []

        for partner_id in potential_partners:
            if partner_id == primary_actor_id:
                continue

            trace = [
                f"Evaluating coalition between {primary_actor_id} and {partner_id}",
                f"Objective: {objective.description}",
            ]

            # Assess cultural alignment
            cultural_distance = self._cultural_layer.assess_cultural_distance(
                primary_actor_id, partner_id
            )
            trace.append(f"Cultural distance: {cultural_distance:.3f}")
            alignment_score = max(0.0, 1.0 - cultural_distance)

            # Identify mutual benefits
            primary_profile = self._cultural_layer.get_profile(primary_actor_id)
            partner_profile = self._cultural_layer.get_profile(partner_id)

            mutual_benefits = []
            constraints = []

            if primary_profile and partner_profile:
                # Find overlapping values
                bridges = self._cultural_layer.identify_bridges(primary_actor_id, partner_id)
                for bridge in bridges:
                    mutual_benefits.append(f"Shared value: {bridge}")

                # Check legitimacy alignment
                primary_legitimacy = self._legitimacy_layer.get_status(primary_actor_id)
                partner_legitimacy = self._legitimacy_layer.get_status(partner_id)

                if primary_legitimacy and partner_legitimacy:
                    legitimacy_compatibility = (
                        1.0 - abs(primary_legitimacy.aggregate_score - partner_legitimacy.aggregate_score)
                    )
                    if legitimacy_compatibility > 0.5:
                        mutual_benefits.append(f"Compatible legitimacy approaches ({legitimacy_compatibility:.2f})")

            # Check objective alignment
            if partner_id in objective.stakeholder_ids:
                mutual_benefits.append(f"Shared interest in objective")
            else:
                constraints.append(f"{partner_id} not directly invested in objective")

            # Compute viability
            viability = alignment_score * 0.6 + (len(mutual_benefits) / 5.0) * 0.4
            viability = min(1.0, viability)

            trace.append(f"Alignment score: {alignment_score:.3f}")
            trace.append(f"Mutual benefits: {mutual_benefits}")
            trace.append(f"Viability: {viability:.3f}")

            proposal = CoalitionProposal(
                proposer_id=primary_actor_id,
                partner_id=partner_id,
                mutual_benefits=mutual_benefits,
                constraints=constraints,
                alignment_score=alignment_score,
                viability=viability,
                trace=trace,
            )

            proposals.append(proposal)

        # Sort by viability
        proposals.sort(key=lambda p: p.viability, reverse=True)
        return proposals

    def adapt_plan(
        self,
        objective: StrategicObjective,
        situation_change: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Adapt a strategic plan based on changed circumstances.

        This is Cleopatra's core strategic insight: the ability to change
        means while maintaining ends. When circumstances change, the plan
        must change, but the objective remains constant.
        """
        trace = [
            f"Adapting plan for objective: {objective.description}",
            f"Change type: {situation_change.get('change_type', 'unknown')}",
        ]

        adaptation = {
            "original_objective": objective.description,
            "change_summary": situation_change.get("summary", "Unknown change"),
            "adaptation_type": "none",
            "modified_constraints": list(objective.constraints),
            "modified_stakeholders": list(objective.stakeholder_ids),
            "plan_adjustments": [],
            "trace": trace,
        }

        change_type = situation_change.get("change_type", "unknown")

        if change_type == "threat_new":
            adaptation["adaptation_type"] = "defensive_adjustment"
            adaptation["plan_adjustments"].append("Strengthen defensive posture")
            adaptation["modified_constraints"].append(f"Threat: {situation_change.get('threat_description', 'unknown')}")
            trace.append("New threat detected — adjusting constraints")

        elif change_type == "opportunity_new":
            adaptation["adaptation_type"] = "offensive_expansion"
            adaptation["plan_adjustments"].append("Expand scope to exploit opportunity")
            adaptation["plan_adjustments"].append(f"Seize opportunity: {situation_change.get('opportunity_description', 'unknown')}")
            trace.append("New opportunity detected — expanding plan scope")

        elif change_type == "stakeholder_shift":
            adaptation["adaptation_type"] = "coalition_reconfiguration"
            new_stakeholder = situation_change.get("new_stakeholder")
            if new_stakeholder and new_stakeholder not in objective.stakeholder_ids:
                adaptation["modified_stakeholders"].append(new_stakeholder)
                adaptation["plan_adjustments"].append(f"Add stakeholder: {new_stakeholder}")
            trace.append(f"Stakeholder shift — reconfigured coalition")

        elif change_type == "resource_constraint":
            adaptation["adaptation_type"] = "efficiency_improvement"
            adaptation["plan_adjustments"].append("Improve resource efficiency")
            adaptation["plan_adjustments"].append("Consider alternative resource paths")
            trace.append("Resource constraint — seeking efficiency")

        else:
            adaptation["adaptation_type"] = "monitoring"
            adaptation["plan_adjustments"].append("Continue current plan with enhanced monitoring")
            trace.append("Minor change — maintaining current approach with monitoring")

        adaptation["trace"] = trace
        return adaptation

    def update_objective_progress(
        self,
        objective_id: str,
        progress_delta: float
    ) -> Optional[StrategicObjective]:
        """Update the progress of a strategic objective."""
        if objective_id not in self._active_objectives:
            return None

        objective = self._active_objectives[objective_id]
        new_progress = min(1.0, objective.progress + progress_delta)
        objective.progress = new_progress
        objective.trace.append(f"Progress updated: {new_progress:.3f} (+{progress_delta:.3f})")

        return objective

    def get_active_objectives(self) -> List[StrategicObjective]:
        return list(self._active_objectives.values())

    def summary(self) -> Dict[str, Any]:
        return {
            "active_objectives": len(self._active_objectives),
            "plan_history_size": len(self._strategic_plan_history),
        }


# =============================================================================
# SECTION F: DIPLOMATIC INTEGRATION LAYER (DIL)
# =============================================================================

@dataclass
class DiplomaticOutput:
    """A fully integrated diplomatic output from the CCA."""
    content: str
    target_population_id: str
    linguistic_realization: Dict[str, Any]
    cultural_framing: Dict[str, Any]
    legitimacy_context: Dict[str, Any]
    strategic_alignment: Dict[str, Any]
    trace: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "target_population_id": self.target_population_id,
            "linguistic_realization": self.linguistic_realization,
            "cultural_framing": self.cultural_framing,
            "legitimacy_context": self.legitimacy_context,
            "strategic_alignment": self.strategic_alignment,
            "trace": self.trace,
        }


class DiplomaticIntegrationLayer:
    """
    The Diplomatic Integration Layer of the Cleopatra Cognitive Architecture.

    Integrates outputs from all lower layers into coherent diplomatic
    communications. Implements Cleopatra's practice of personal diplomacy —
    the use of direct engagement, cultural sensitivity, and strategic
    coherence to achieve political objectives.
    """

    def __init__(
        self,
        linguistic_encoder: Optional[MultilingualLinguisticEncoder] = None,
        cultural_layer: Optional[CulturalIntelligenceLayer] = None,
        legitimacy_layer: Optional[LegitimacyConstructionLayer] = None,
        strategic_layer: Optional[StrategicReasoningLayer] = None,
    ):
        self._linguistic = linguistic_encoder or MultilingualLinguisticEncoder()
        self._cultural = cultural_layer or CulturalIntelligenceLayer()
        self._legitimacy = legitimacy_layer or LegitimacyConstructionLayer(self._cultural)
        self._strategic = strategic_layer or StrategicReasoningLayer(self._cultural, self._legitimacy)
        self._communication_history: List[Dict[str, Any]] = []

    def produce_diplomatic_output(
        self,
        raw_content: str,
        target_population_id: str,
        source_language: str = "greek",
        objective_id: Optional[str] = None,
    ) -> DiplomaticOutput:
        """
        Produce a fully integrated diplomatic output.

        This is the central method of the DIL — taking a raw communicative
        intent and producing a culturally adapted, linguistically
        appropriate, strategically aligned diplomatic communication.
        """
        trace = [
            f"Producing diplomatic output for population: {target_population_id}",
            f"Raw content: {raw_content[:50]}...",
            f"Source language: {source_language}",
        ]

        # Step 1: Encode linguistically
        linguistic_encoding = self._linguistic.encode(raw_content, source_language)
        trace.append(f"Linguistic encoding: {len(linguistic_encoding['retrieved_concepts'])} concepts")

        # Step 2: Translate with cultural context
        translated = self._linguistic.translate_with_cultural_context(
            raw_content, source_language, target_language=self._get_population_language(target_population_id)
        )
        trace.append(f"Translated with cultural context: {len(translated['translations'])} terms")

        # Step 3: Adapt for cultural context
        cultural_adaptation = self._cultural.adapt_output_for_culture(
            {"raw_content": raw_content},
            target_population_id
        )
        trace.append(f"Cultural adaptation applied: {cultural_adaptation.get('cultural_framing', {}).get('register', 'unknown')}")

        # Step 4: Assess legitimacy
        authority_actions = {
            "cultural_engagements": [target_population_id],
            "register_used": cultural_adaptation.get("cultural_framing", {}).get("register", "neutral"),
        }
        legitimacy_status = self._legitimacy.assess_legitimacy(authority_actions, target_population_id)
        trace.append(f"Legitimacy assessment: {legitimacy_status.aggregate_score:.3f} (legitimate: {legitimacy_status.is_legitimate})")

        # Step 5: Integrate strategic alignment
        strategic_alignment = {}
        if objective_id:
            objective = self._strategic._active_objectives.get(objective_id)
            if objective:
                strategic_alignment = {
                    "objective_id": objective_id,
                    "objective_description": objective.description,
                    "progress": objective.progress,
                    "priority": objective.priority,
                }
                trace.append(f"Strategic alignment: objective {objective_id}, progress {objective.progress:.3f}")

        # Step 6: Build final output
        output = DiplomaticOutput(
            content=raw_content,
            target_population_id=target_population_id,
            linguistic_realization=translated,
            cultural_framing=cultural_adaptation.get("cultural_framing", {}),
            legitimacy_context={
                "legitimacy_score": legitimacy_status.aggregate_score,
                "is_legitimate": legitimacy_status.is_legitimate,
                "threats": legitimacy_status.threats,
            },
            strategic_alignment=strategic_alignment,
            trace=trace,
        )

        self._communication_history.append(output.to_dict())
        return output

    def _get_population_language(self, population_id: str) -> str:
        """Map population to their primary language."""
        language_map = {
            "egyptian_priesthood": "egyptian",
            "greek_elite": "greek",
            "roman_senatorial": "latin",
            "judean_client": "hebrew",
            "nubian_southern": "egyptian",
        }
        return language_map.get(population_id, "greek")

    def manage_relationship(
        self,
        population_id: str,
        interaction_history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Manage an ongoing diplomatic relationship with a population.

        Tracks the history of interactions and adapts approach based
        on accumulated relationship knowledge.
        """
        if not interaction_history:
            return {
                "population_id": population_id,
                "relationship_status": "new",
                "trust_level": 0.5,
                "recommendations": ["Begin relationship with formal greeting"],
            }

        # Analyze interaction history
        successful_outcomes = sum(1 for i in interaction_history if i.get("outcome") == "success")
        total_interactions = len(interaction_history)
        success_rate = successful_outcomes / max(total_interactions, 1)

        # Compute trust level
        trust_level = 0.3 + 0.4 * success_rate + 0.3 * (min(total_interactions, 10) / 10.0)

        relationship_status = "established" if trust_level > 0.6 else "developing" if trust_level > 0.3 else "strained"

        recommendations = []
        if trust_level < 0.5:
            recommendations.append("Strengthen interpersonal channel through direct engagement")
            recommendations.append("Demonstrate consistent reliability over multiple interactions")
        if success_rate < 0.5:
            recommendations.append("Review communication style for cultural misalignment")
            recommendations.append("Consult cultural intelligence layer for adaptation adjustments")

        return {
            "population_id": population_id,
            "relationship_status": relationship_status,
            "trust_level": trust_level,
            "total_interactions": total_interactions,
            "success_rate": success_rate,
            "recommendations": recommendations,
        }

    def produce_multilateral_communication(
        self,
        content: str,
        target_population_ids: List[str],
        source_language: str = "greek",
    ) -> List[DiplomaticOutput]:
        """
        Produce adapted versions of the same content for multiple populations.

        Implements Cleopatra's practice of tailoring her message to different
        audiences — speaking Greek to the Greek elite and Egyptian to the
        priesthood, while maintaining the core message.
        """
        outputs = []
        for pop_id in target_population_ids:
            output = self.produce_diplomatic_output(
                raw_content=content,
                target_population_id=pop_id,
                source_language=source_language,
            )
            outputs.append(output)

        return outputs

    def summary(self) -> Dict[str, Any]:
        return {
            "total_communications": len(self._communication_history),
            "layers_initialized": {
                "linguistic": True,
                "cultural": True,
                "legitimacy": True,
                "strategic": True,
            },
        }


# =============================================================================
# SECTION G: CLEOPATRA COGNITIVE ARCHITECTURE (CCA)
# =============================================================================

class CleopatraCognitiveArchitecture:
    """
    The Cleopatra Cognitive Architecture — a 5-layer neural network
    for cultural intelligence and diplomatic reasoning.

    This architecture implements Cleopatra VII's key insights:

    1. Language is power: Multilingual reasoning provides cognitive reach
    2. Legitimacy must be constructed: Authority requires active maintenance
    3. Soft power is real power: Cultural intelligence extends influence
    4. Intelligence networks amplify: Distributed systems extend cognition
    5. Strategic flexibility: Adapt means, not ends

    The five layers are:
    - LEL: Linguistic Encoding Layer (multilingual representation)
    - CIL: Cultural Intelligence Layer (cultural modeling)
    - LCL: Legitimacy Construction Layer (authority building)
    - SRL: Strategic Reasoning Layer (planning and adaptation)
    - DIL: Diplomatic Integration Layer (external outputs)
    """

    def __init__(self):
        # Initialize all five layers
        self.LEL = MultilingualLinguisticEncoder()
        self.CIL = CulturalIntelligenceLayer()
        self.LCL = LegitimacyConstructionLayer(self.CIL)
        self.SRL = StrategicReasoningLayer(self.CIL, self.LCL)
        self.DIL = DiplomaticIntegrationLayer(self.LEL, self.CIL, self.LCL, self.SRL)

        self._processing_history: List[Dict[str, Any]] = []
        self._identity = "CleopatraCognitiveArchitecture"

    def process(
        self,
        input_text: str,
        source_language: str = "greek",
        target_population_id: Optional[str] = None,
        objective_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process an input through all five layers.

        Args:
            input_text: The text to process
            source_language: The language of the input
            target_population_id: Target population for output (optional)
            objective_id: Strategic objective to align with (optional)

        Returns:
            A dictionary containing outputs from all five layers
        """
        result = {
            "input_text": input_text,
            "source_language": source_language,
            "target_population_id": target_population_id,
            "processing_timestamp": time.time(),
        }

        # Layer 1: Linguistic Encoding
        linguistic_output = self.LEL.encode(input_text, source_language)
        result["linguistic_encoding"] = {
            "concepts_retrieved": len(linguistic_output["retrieved_concepts"]),
            "cultural_tags": linguistic_output["cultural_tags"],
            "cross_lingual_signals": linguistic_output["cross_lingual_signals"],
        }

        # Layer 2: Cultural Intelligence
        if target_population_id:
            cultural_adaptation = self.CIL.adapt_output_for_culture(
                {"raw_content": input_text},
                target_population_id
            )
            result["cultural_intelligence"] = {
                "adaptation_applied": True,
                "cultural_framing": cultural_adaptation.get("cultural_framing", {}),
            }
        else:
            result["cultural_intelligence"] = {"adaptation_applied": False}

        # Layer 3: Legitimacy Construction
        if target_population_id:
            authority_actions = {
                "cultural_engagements": [target_population_id],
                "process_fairness": 0.7,
                "process_transparency": 0.6,
            }
            legitimacy_status = self.LCL.assess_legitimacy(authority_actions, target_population_id)
            result["legitimacy_construction"] = {
                "aggregate_score": legitimacy_status.aggregate_score,
                "is_legitimate": legitimacy_status.is_legitimate,
                "channel_scores": {
                    "performance": legitimacy_status.performance_score,
                    "procedural": legitimacy_status.procedural_score,
                    "cultural": legitimacy_status.cultural_score,
                    "interpersonal": legitimacy_status.interpersonal_score,
                },
                "threats": legitimacy_status.threats,
            }
        else:
            result["legitimacy_construction"] = {"assessed": False}

        # Layer 4: Strategic Reasoning
        if objective_id:
            objective = self.SRL._active_objectives.get(objective_id)
            if objective:
                result["strategic_reasoning"] = {
                    "objective_id": objective_id,
                    "objective_description": objective.description,
                    "priority": objective.priority,
                    "progress": objective.progress,
                }
            else:
                result["strategic_reasoning"] = {"objective_id": objective_id, "found": False}
        else:
            result["strategic_reasoning"] = {"aligned": False}

        # Layer 5: Diplomatic Integration
        if target_population_id:
            diplomatic_output = self.DIL.produce_diplomatic_output(
                raw_content=input_text,
                target_population_id=target_population_id,
                source_language=source_language,
                objective_id=objective_id,
            )
            result["diplomatic_integration"] = {
                "output_generated": True,
                "legitimacy_score": diplomatic_output.legitimacy_context.get("legitimacy_score", 0.0),
                "strategic_alignment": bool(diplomatic_output.strategic_alignment),
            }
        else:
            result["diplomatic_integration"] = {"output_generated": False}

        self._processing_history.append(result)
        return result

    def add_cleopatra_objectives(self):
        """Add strategic objectives inspired by Cleopatra's career."""
        self.SRL.formulate_objective(
            description="Preserve Egyptian sovereignty against Roman expansion",
            priority=0.95,
            constraints=["Limited military resources", "Roman military superiority"],
            stakeholder_ids=["egyptian_priesthood", "greek_elite"],
        )

        self.SRL.formulate_objective(
            description="Build strategic alliance with Roman leadership",
            priority=0.90,
            constraints=["Must maintain Egyptian independence", "Roman political instability"],
            stakeholder_ids=["roman_senatorial", "egyptian_priesthood"],
        )

        self.SRL.formulate_objective(
            description="Extend Egyptian influence through cultural patronage",
            priority=0.75,
            constraints=["Limited treasury", "Must serve both Greek and Egyptian audiences"],
            stakeholder_ids=["greek_elite", "egyptian_priesthood", "judean_client"],
        )

    def full_diagnosis(self) -> Dict[str, Any]:
        """Run a full diagnosis of the CCA system."""
        return {
            "architecture": "CleopatraCognitiveArchitecture",
            "layers": {
                "LEL (Linguistic Encoding Layer)": {
                    "status": "active",
                    "concepts": len(self.LEL.semantic_network),
                    "languages": self.LEL.get_language_diversity(),
                },
                "CIL (Cultural Intelligence Layer)": {
                    "status": "active",
                    "profiles": len(self.CIL._cultural_profiles),
                    "profile_ids": list(self.CIL._cultural_profiles.keys()),
                },
                "LCL (Legitimacy Construction Layer)": {
                    "status": "active",
                    "assessed_populations": len(self.LCL._status_by_population),
                },
                "SRL (Strategic Reasoning Layer)": {
                    "status": "active",
                    "active_objectives": len(self.SRL._active_objectives),
                },
                "DIL (Diplomatic Integration Layer)": {
                    "status": "active",
                    "total_communications": len(self.DIL._communication_history),
                },
            },
            "processing_history_size": len(self._processing_history),
        }

    def __repr__(self) -> str:
        return (
            f"CleopatraCognitiveArchitecture("
            f"LEL={len(self.LEL.semantic_network)} concepts, "
            f"CIL={len(self.CIL._cultural_profiles)} profiles, "
            f"SRL={len(self.SRL._active_objectives)} objectives)"
        )


# =============================================================================
# SECTION H: DEMONSTRATIONS AND EXAMPLES
# =============================================================================

DEMO_COUNTER = 0


def increment_demo():
    global DEMO_COUNTER
    DEMO_COUNTER += 1
    return DEMO_COUNTER


def demo_multilingual_encoding():
    """Demonstrate the Multilingual Linguistic Encoding System."""
    print("\n" + "=" * 70)
    print("DEMO 1: Multilingual Linguistic Encoding System")
    print("=" * 70)

    encoder = MultilingualLinguisticEncoder()

    print(f"\n[LEL] Semantic network initialized with {len(encoder.semantic_network)} concepts")
    print(f"[LEL] Languages: {list(encoder.get_language_diversity().keys())}")

    # Encode a political concept
    text = "The goddess Isis grants legitimacy to the queen who upholds maat"
    encoding = encoder.encode(text, source_language="greek")

    print(f"\n[LEL] Encoded: '{text}'")
    print(f"[LEL] Retrieved concepts: {len(encoding['retrieved_concepts'])}")
    print(f"[LEL] Cultural tags: {encoding['cultural_tags']}")
    print(f"[LEL] Cross-lingual signals: {encoding['cross_lingual_signals']}")

    # Translate with cultural context
    translation = encoder.translate_with_cultural_context(
        text, source_language="greek", target_language="egyptian"
    )
    print(f"\n[LEL] Cultural translation:")
    print(f"     Original: {translation['original']}")
    print(f"     Translations: {len(translation['translations'])} terms")
    for concept_id, trans in translation['translations'].items():
        print(f"       - {concept_id}: {trans['term']} ({trans['connotation']})")

    # Assess cultural alignment
    alignment = encoder.assess_cultural_alignment(text, "egyptian_priesthood")
    print(f"\n[LEL] Cultural alignment with Egyptian Priesthood:")
    print(f"     Alignment score: {alignment['alignment_score']:.3f}")
    print(f"     Matched concepts: {alignment['matched_concepts']}")

    return True


def demo_cultural_intelligence():
    """Demonstrate the Cultural Intelligence Layer."""
    print("\n" + "=" * 70)
    print("DEMO 2: Cultural Intelligence Layer")
    print("=" * 70)

    cil = CulturalIntelligenceLayer()

    print(f"\n[CIL] Initialized {len(cil._cultural_profiles)} cultural profiles:")
    for pid, profile in cil._cultural_profiles.items():
        print(f"     - {pid}: {profile.population_name}")

    # Assess legitimacy for different populations
    authority_claims = {
        "divine_sanction": 0.8,
        "ritual_observance": 0.7,
        "ancestral_connection": 0.6,
        "material_prosperity": 0.5,
        "military_strength": 0.3,
    }

    print(f"\n[CIL] Legitimacy assessment with authority claims: {authority_claims}")

    for pop_id in ["egyptian_priesthood", "greek_elite", "roman_senatorial"]:
        result = cil.compute_legitimacy_for_profile(authority_claims, pop_id)
        print(f"\n     Population: {pop_id}")
        print(f"     Aggregate score: {result['overall_legitimacy_score']:.3f}")
        print(f"     Recommendation: {result.get('recommendation', {})}")

    # Assess cultural distance
    print(f"\n[CIL] Cultural distances:")
    pairs = [
        ("egyptian_priesthood", "greek_elite"),
        ("egyptian_priesthood", "roman_senatorial"),
        ("roman_senatorial", "judean_client"),
    ]
    for a, b in pairs:
        dist = cil.assess_cultural_distance(a, b)
        print(f"     {a} <-> {b}: {dist:.3f}")

    # Find bridging concepts
    print(f"\n[CIL] Cultural bridges between Egyptian Priesthood and Greek Elite:")
    bridges = cil.identify_bridges("egyptian_priesthood", "greek_elite")
    for bridge in bridges:
        print(f"     - {bridge}")

    return True


def demo_legitimacy_construction():
    """Demonstrate the Legitimacy Construction Layer."""
    print("\n" + "=" * 70)
    print("DEMO 3: Legitimacy Construction Layer")
    print("=" * 70)

    cil = CulturalIntelligenceLayer()
    lcl = LegitimacyConstructionLayer(cil)

    authority_actions = {
        "effective_actions": 8,
        "total_actions": 10,
        "task_success_rate": 0.85,
        "process_fairness": 0.8,
        "process_transparency": 0.75,
        "decision_consistency": 0.9,
        "cultural_engagements": ["egyptian_priesthood", "greek_elite", "judean_client"],
        "register_used": "formal_religious",
        "trust_build_score": 0.7,
        "responsiveness_score": 0.8,
        "reliability_score": 0.85,
    }

    print(f"\n[LCL] Assessing legitimacy with actions:")
    print(f"     Task success rate: {authority_actions['task_success_rate']}")
    print(f"     Process fairness: {authority_actions['process_fairness']}")
    print(f"     Cultural engagements: {len(authority_actions['cultural_engagements'])}")

    for pop_id in ["egyptian_priesthood", "greek_elite", "roman_senatorial"]:
        status = lcl.assess_legitimacy(authority_actions, pop_id)
        print(f"\n     Population: {pop_id}")
        print(f"     Aggregate score: {status.aggregate_score:.3f}")
        print(f"     Is legitimate: {status.is_legitimate}")
        print(f"     Channel scores:")
        print(f"       - Performance: {status.performance_score:.3f}")
        print(f"       - Procedural: {status.procedural_score:.3f}")
        print(f"       - Cultural: {status.cultural_score:.3f}")
        print(f"       - Interpersonal: {status.interpersonal_score:.3f}")
        if status.threats:
            print(f"     Threats: {status.threats}")

    # Construct improvement plan
    status = lcl.get_status("roman_senatorial")
    if status:
        plan = lcl.construct_legitimacy_plan(status, target_score=0.8)
        print(f"\n[LCL] Legitimacy improvement plan for Roman Senatorial:")
        print(f"     Current: {plan['current_score']:.3f}")
        print(f"     Target: {plan['target_score']:.3f}")
        for channel, details in plan['channels'].items():
            print(f"     {channel}:")
            print(f"       Current: {details['current']:.3f}, Target: {details['target']:.3f}")
            for rec in details['recommendations'][:2]:
                print(f"       Recommendation: {rec}")

    return True


def demo_strategic_reasoning():
    """Demonstrate the Strategic Reasoning Layer."""
    print("\n" + "=" * 70)
    print("DEMO 4: Strategic Reasoning Layer")
    print("=" * 70)

    cil = CulturalIntelligenceLayer()
    lcl = LegitimacyConstructionLayer(cil)
    srl = StrategicReasoningLayer(cil, lcl)

    # Formulate objectives
    objective1 = srl.formulate_objective(
        description="Preserve Egyptian sovereignty against Roman expansion",
        priority=0.95,
        constraints=["Limited military resources", "Roman military superiority"],
        stakeholder_ids=["egyptian_priesthood", "greek_elite"],
    )
    print(f"\n[SRL] Objective created: {objective1.description}")
    print(f"     Priority: {objective1.priority}, ID: {objective1.objective_id}")

    objective2 = srl.formulate_objective(
        description="Build strategic alliance with Roman leadership",
        priority=0.90,
        stakeholder_ids=["roman_senatorial", "greek_elite"],
    )
    print(f"\n[SRL] Objective created: {objective2.description}")
    print(f"     Priority: {objective2.priority}, ID: {objective2.objective_id}")

    # Assess situation
    intelligence_reports = [
        {
            "source": "alexandria_agent",
            "signal": "favorable",
            "factors": ["Caesar supports Cleopatra's restoration", "Ptolemy XIII faction weakened"],
            "risks": [],
            "opportunities": ["Alliance opportunity with Caesar"],
        },
        {
            "source": "rome_agent",
            "signal": "neutral",
            "factors": ["Roman political instability after Caesar's assassination"],
            "risks": ["Political chaos in Rome"],
            "opportunities": [],
        },
        {
            "source": "parthia_agent",
            "signal": "unfavorable",
            "factors": ["Parthian forces advancing", "Eastern border pressure"],
            "risks": ["Military pressure from east"],
            "opportunities": [],
        },
    ]

    situation = srl.assess_situation(intelligence_reports, active_objective_id=objective1.objective_id)
    print(f"\n[SRL] Situation assessment:")
    print(f"     Intelligence sources: {situation['intelligence_sources']}")
    print(f"     Situation score: {situation['situation_score']:.3f}")
    print(f"     Key factors: {len(situation['key_factors'])}")
    print(f"     Risks: {situation['risks']}")
    print(f"     Opportunities: {situation['opportunities']}")
    print(f"     Recommendations: {situation['strategic_recommendations']}")

    # Build coalition
    proposals = srl.build_coalition(
        primary_actor_id="egyptian_priesthood",
        potential_partners=["roman_senatorial", "greek_elite", "judean_client"],
        objective=objective1,
    )
    print(f"\n[SRL] Coalition proposals:")
    for proposal in proposals:
        print(f"     Partner: {proposal.partner_id}")
        print(f"       Alignment: {proposal.alignment_score:.3f}, Viability: {proposal.viability:.3f}")
        print(f"       Mutual benefits: {proposal.mutual_benefits[:2]}")

    # Adapt plan
    adaptation = srl.adapt_plan(
        objective=objective1,
        situation_change={
            "change_type": "opportunity_new",
            "summary": "Mark Antony seeking eastern alliance partner",
            "opportunity_description": "Antony offers strategic partnership",
        }
    )
    print(f"\n[SRL] Plan adaptation for Objective 1:")
    print(f"     Adaptation type: {adaptation['adaptation_type']}")
    print(f"     Plan adjustments: {adaptation['plan_adjustments']}")

    return True


def demo_diplomatic_integration():
    """Demonstrate the Diplomatic Integration Layer."""
    print("\n" + "=" * 70)
    print("DEMO 5: Diplomatic Integration Layer")
    print("=" * 70)

    cca = CleopatraCognitiveArchitecture()

    print(f"\n[DIL] Cleopatra Cognitive Architecture initialized")
    print(f"     {cca}")

    # Add objectives
    cca.add_cleopatra_objectives()
    active_objs = cca.SRL.get_active_objectives()
    print(f"\n[DIL] Cleopatra's strategic objectives ({len(active_objs)}):")
    for obj in active_objs:
        print(f"     - [{obj.priority:.2f}] {obj.description[:60]}...")

    # Produce diplomatic output
    content = "The divine order of the cosmos requires the union of divine wisdom and earthly authority"
    target_pop = "egyptian_priesthood"
    objective_id = active_objs[0].objective_id if active_objs else None

    output = cca.DIL.produce_diplomatic_output(
        raw_content=content,
        target_population_id=target_pop,
        source_language="greek",
        objective_id=objective_id,
    )

    print(f"\n[DIL] Diplomatic output produced:")
    print(f"     Target: {output.target_population_id}")
    print(f"     Cultural framing: {output.cultural_framing.get('register', 'unknown')}")
    print(f"     Legitimacy score: {output.legitimacy_context.get('legitimacy_score', 0.0):.3f}")
    print(f"     Strategic alignment: {bool(output.strategic_alignment)}")

    # Produce multilateral communication
    print(f"\n[DIL] Multilateral communication to multiple populations:")
    outputs = cca.DIL.produce_multilateral_communication(
        content="Peace and prosperity shall accompany the divine mandate",
        target_population_ids=["egyptian_priesthood", "greek_elite", "roman_senatorial"],
    )
    for out in outputs:
        print(f"     [{out.target_population_id}] "
              f"legitimacy={out.legitimacy_context.get('legitimacy_score', 0.0):.3f}, "
              f"framing={out.cultural_framing.get('register', 'unknown')}")

    return True


def demo_full_cca():
    """Demonstrate the full Cleopatra Cognitive Architecture."""
    print("\n" + "=" * 70)
    print("DEMO 6: Full Cleopatra Cognitive Architecture")
    print("=" * 70)

    cca = CleopatraCognitiveArchitecture()

    # Initialize Cleopatra's strategic objectives
    cca.add_cleopatra_objectives()

    print(f"\n[CCA] Cleopatra Cognitive Architecture initialized")
    print(f"     {cca}")

    # Process an input
    result = cca.process(
        input_text="The queen of Egypt, chosen by Isis, shall preserve the sacred order",
        source_language="greek",
        target_population_id="egyptian_priesthood",
        objective_id=cca.SRL._active_objectives.get(
            list(cca.SRL._active_objectives.keys())[0]
        ).objective_id if cca.SRL._active_objectives else None,
    )

    print(f"\n[CCA] Full processing result:")
    print(f"     Linguistic encoding: {result['linguistic_encoding']['concepts_retrieved']} concepts")
    print(f"     Cultural adaptation: {result['cultural_intelligence']['adaptation_applied']}")
    print(f"     Legitimacy score: {result['legitimacy_construction'].get('aggregate_score', 'N/A')}")
    print(f"     Strategic alignment: {result['strategic_reasoning']}")

    # Run full diagnosis
    diagnosis = cca.full_diagnosis()
    print(f"\n[CCA] Full architecture diagnosis:")
    print(f"     Architecture: {diagnosis['architecture']}")
    for layer_name, layer_info in diagnosis['layers'].items():
        print(f"     {layer_name}: {layer_info.get('status', 'unknown')}")
        for key, value in layer_info.items():
            if key != 'status':
                print(f"       {key}: {value}")

    return True


def run_all_demos():
    """Run all demonstrations."""
    print("\n" + "#" * 70)
    print("# CLEOPATRA COGNITIVE ARCHITECTURE — DEMONSTRATIONS")
    print("# Chapter 109: Cleopatra VII (69-30 BCE)")
    print("#" * 70)

    demos = [
        ("Multilingual Linguistic Encoding", demo_multilingual_encoding),
        ("Cultural Intelligence", demo_cultural_intelligence),
        ("Legitimacy Construction", demo_legitimacy_construction),
        ("Strategic Reasoning", demo_strategic_reasoning),
        ("Diplomatic Integration", demo_diplomatic_integration),
        ("Full CCA System", demo_full_cca),
    ]

    results = []
    for name, func in demos:
        try:
            success = func()
            results.append((name, success, None))
        except Exception as e:
            results.append((name, False, str(e)))
            print(f"\n[ERROR] Demo '{name}' failed: {e}")

    print("\n" + "#" * 70)
    print("# DEMO RESULTS SUMMARY")
    print("#" * 70)
    for name, success, error in results:
        status = "PASS" if success else f"FAIL: {error}"
        print(f"  {name}: {status}")
    print("#" * 70)

    return all(r[1] for r in results)


# =============================================================================
# SECTION I: ANALYSIS AND DIAGNOSTICS
# =============================================================================

def run_unit_tests():
    """Run basic unit tests for all components."""
    print("\n" + "=" * 70)
    print("UNIT TESTS")
    print("=" * 70)

    tests_passed = 0
    tests_failed = 0

    # Test Vector
    try:
        v1 = Vector([1.0, 2.0, 3.0])
        v2 = Vector([4.0, 5.0, 6.0])
        assert v1 + v2 == Vector([5.0, 7.0, 9.0])
        assert v1.dot(v2) == 32.0
        assert abs(v1.normalize().norm() - 1.0) < 1e-6
        tests_passed += 1
        print("  [PASS] Vector operations")
    except Exception as e:
        tests_failed += 1
        print(f"  [FAIL] Vector operations: {e}")

    # Test Matrix
    try:
        m = Matrix([[1.0, 2.0], [3.0, 4.0]])
        v = Vector([1.0, 1.0])
        result = m * v
        assert len(result) == 2
        tests_passed += 1
        print("  [PASS] Matrix-vector multiplication")
    except Exception as e:
        tests_failed += 1
        print(f"  [FAIL] Matrix-vector multiplication: {e}")

    # Test MultilingualLinguisticEncoder
    try:
        encoder = MultilingualLinguisticEncoder()
        encoding = encoder.encode("Power flows through legitimate authority", "greek")
        assert len(encoding["retrieved_concepts"]) > 0
        tests_passed += 1
        print("  [PASS] Multilingual Linguistic Encoder")
    except Exception as e:
        tests_failed += 1
        print(f"  [FAIL] Multilingual Linguistic Encoder: {e}")

    # Test CulturalIntelligenceLayer
    try:
        cil = CulturalIntelligenceLayer()
        profile = cil.get_profile("egyptian_priesthood")
        assert profile is not None
        assert len(cil._cultural_profiles) >= 5
        tests_passed += 1
        print("  [PASS] Cultural Intelligence Layer")
    except Exception as e:
        tests_failed += 1
        print(f"  [FAIL] Cultural Intelligence Layer: {e}")

    # Test LegitimacyConstructionLayer
    try:
        cil = CulturalIntelligenceLayer()
        lcl = LegitimacyConstructionLayer(cil)
        status = lcl.assess_legitimacy(
            {"effective_actions": 8, "total_actions": 10, "task_success_rate": 0.85,
             "process_fairness": 0.8, "process_transparency": 0.7, "decision_consistency": 0.9,
             "cultural_engagements": ["egyptian_priesthood"], "register_used": "formal",
             "trust_build_score": 0.7, "responsiveness_score": 0.8, "reliability_score": 0.85},
            "egyptian_priesthood"
        )
        assert status.aggregate_score > 0
        tests_passed += 1
        print("  [PASS] Legitimacy Construction Layer")
    except Exception as e:
        tests_failed += 1
        print(f"  [FAIL] Legitimacy Construction Layer: {e}")

    # Test StrategicReasoningLayer
    try:
        srl = StrategicReasoningLayer()
        obj = srl.formulate_objective("Preserve sovereignty", priority=0.9)
        assert obj.priority == 0.9
        assessment = srl.assess_situation([
            {"signal": "favorable", "factors": [], "risks": [], "opportunities": []}
        ])
        assert "situation_score" in assessment
        tests_passed += 1
        print("  [PASS] Strategic Reasoning Layer")
    except Exception as e:
        tests_failed += 1
        print(f"  [FAIL] Strategic Reasoning Layer: {e}")

    # Test CleopatraCognitiveArchitecture
    try:
        cca = CleopatraCognitiveArchitecture()
        assert hasattr(cca, 'LEL')
        assert hasattr(cca, 'CIL')
        assert hasattr(cca, 'LCL')
        assert hasattr(cca, 'SRL')
        assert hasattr(cca, 'DIL')
        tests_passed += 1
        print("  [PASS] Cleopatra Cognitive Architecture (full system)")
    except Exception as e:
        tests_failed += 1
        print(f"  [FAIL] Cleopatra Cognitive Architecture: {e}")

    # Test DiplomaticIntegrationLayer
    try:
        dil = DiplomaticIntegrationLayer()
        output = dil.produce_diplomatic_output(
            "The divine order requires alliance",
            target_population_id="greek_elite",
        )
        assert output.target_population_id == "greek_elite"
        tests_passed += 1
        print("  [PASS] Diplomatic Integration Layer")
    except Exception as e:
        tests_failed += 1
        print(f"  [FAIL] Diplomatic Integration Layer: {e}")

    print(f"\n  Total: {tests_passed} passed, {tests_failed} failed")
    return tests_failed == 0


def show_architecture_overview():
    """Display the architecture overview."""
    print("""
================================================================================
CLEOPATRA COGNITIVE ARCHITECTURE (CCA)
Five-Layer System for Cultural Intelligence and Diplomatic Reasoning
================================================================================

Layer 1 — LEL: LINGUISTIC ENCODING LAYER
    Role: Multilingual representation preserving cultural nuance
    Key Classes:
      - MultilingualSemanticNetwork: Graph of concepts with multilingual realizations
      - CulturalEmbedding: Concept with linguistic/cultural metadata
      - MultilingualLinguisticEncoder: Encoding and translation with cultural context
    Key Methods:
      - encode(): Encode text with multilingual concept retrieval
      - translate_with_cultural_context(): Translate preserving cultural nuance
      - assess_cultural_alignment(): Score alignment with target culture

Layer 2 — CIL: CULTURAL INTELLIGENCE LAYER
    Role: Model values, norms, expectations of different populations
    Key Classes:
      - CulturalProfile: High-dimensional model of a population's culture
      - CulturalIntelligenceLayer: Manages multiple cultural profiles
    Key Methods:
      - adapt_output_for_culture(): Tailor output to target population
      - compute_legitimacy_for_profile(): Assess authority claims against culture
      - identify_bridges(): Find common ground between distant cultures
    Profiles: Egyptian Priesthood, Greek Elite, Roman Senatorial, Judean Client, Nubian

Layer 3 — LCL: LEGITIMACY CONSTRUCTION LAYER
    Role: Establish and maintain authority legitimacy
    Key Classes:
      - LegitimacyStatus: Tracks legitimacy across four channels
      - LegitimacyConstructionLayer: Constructs and maintains authority
    Channels: Performance, Procedural, Cultural, Interpersonal
    Key Methods:
      - assess_legitimacy(): Score authority across all channels
      - construct_legitimacy_plan(): Generate improvement recommendations

Layer 4 — SRL: STRATEGIC REASONING LAYER
    Role: Strategic planning, coalition building, adaptive response
    Key Classes:
      - StrategicObjective: Goal with priority, constraints, stakeholders
      - CoalitionProposal: Alliance proposal between political actors
      - StrategicReasoningLayer: Strategic planning and adaptation
    Key Methods:
      - formulate_objective(): Create strategic goal
      - assess_situation(): Evaluate intelligence reports
      - build_coalition(): Generate alliance proposals
      - adapt_plan(): Adjust plans to changing circumstances

Layer 5 — DIL: DIPLOMATIC INTEGRATION LAYER
    Role: Integrate all layers into coherent diplomatic outputs
    Key Classes:
      - DiplomaticOutput: Fully integrated communication
      - DiplomaticIntegrationLayer: Produces adapted diplomatic outputs
    Key Methods:
      - produce_diplomatic_output(): Full integration pipeline
      - manage_relationship(): Track and adapt relationship over time
      - produce_multilateral_communication(): Same message for multiple populations

CLEOPATRA'S FIVE CORE INSIGHTS (implemented in CCA):
  1. Language is Power: Multilingual reasoning enables direct access to information
  2. Legitimacy Must Be Constructed: Authority requires active maintenance
  3. Soft Power is Real Power: Cultural intelligence extends influence
  4. Intelligence Networks Amplify: Distributed systems extend cognitive reach
  5. Strategic Flexibility: Adapt means, maintain ends

================================================================================
""")


def main():
    """Main entry point for the module."""
    args = sys.argv[1:] if len(sys.argv) > 1 else []

    if "--test" in args:
        success = run_unit_tests()
        sys.exit(0 if success else 1)

    if "--arch" in args:
        show_architecture_overview()
        sys.exit(0)

    if "--demo" in args:
        demo_idx = args.index("--demo") + 1
        if demo_idx < len(args):
            demo_name = args[demo_idx]
            demo_map = {
                "DEMO1": demo_multilingual_encoding,
                "DEMO2": demo_cultural_intelligence,
                "DEMO3": demo_legitimacy_construction,
                "DEMO4": demo_strategic_reasoning,
                "DEMO5": demo_diplomatic_integration,
                "DEMO6": demo_full_cca,
            }
            if demo_name in demo_map:
                demo_map[demo_name]()
                sys.exit(0)
            else:
                print(f"Unknown demo: {demo_name}")
                print("Available: DEMO1-DEMO6")
                sys.exit(1)

    # Run all demos by default
    run_all_demos()


if __name__ == "__main__":
    main()