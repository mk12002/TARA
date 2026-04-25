"""
Query Router
===============
Routes queries to the appropriate processing path based on intent and complexity.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from loguru import logger

from src.query.intent_classifier import IntentClassifier, QueryIntent
from src.query.entity_extractor import EntityExtractor, ExtractedEntities
from src.query.query_enricher import QueryEnricher
from src.query.complexity_scorer import ComplexityScorer


class ProcessingPath(str, Enum):
    FAST = "FAST"          # Direct RAG, <2s
    REASONING = "REASONING"  # Agentic reasoning path


@dataclass
class RoutedQuery:
    """A query with routing metadata attached."""
    original_query: str
    enriched_query: str
    intent: QueryIntent
    entities: ExtractedEntities
    complexity: float
    path: ProcessingPath


class QueryRouter:
    """Routes queries through understanding → routing pipeline."""

    def __init__(self):
        self.intent_classifier = IntentClassifier()
        self.entity_extractor = EntityExtractor()
        self.query_enricher = QueryEnricher()
        self.complexity_scorer = ComplexityScorer()

    def route(self, query: str) -> RoutedQuery:
        """
        Full query understanding and routing.
        
        Steps:
        1. Classify intent
        2. Extract entities
        3. Enrich query
        4. Score complexity
        5. Determine processing path
        """
        # Step 1: Intent
        intent = self.intent_classifier.classify(query)

        # Step 2: Entity extraction
        entities = self.entity_extractor.extract(query)
        entity_count = (
            len(entities.spec_refs) + len(entities.cell_ids)
            + len(entities.kpis) + len(entities.procedures)
        )

        # Step 3: Enrich
        enriched = self.query_enricher.enrich(query)

        # Step 4: Complexity
        complexity = self.complexity_scorer.score(query, intent, entity_count)

        # Step 5: Route
        path = (
            ProcessingPath.REASONING
            if self.complexity_scorer.needs_reasoning(complexity)
            else ProcessingPath.FAST
        )

        routed = RoutedQuery(
            original_query=query,
            enriched_query=enriched,
            intent=intent,
            entities=entities,
            complexity=complexity,
            path=path,
        )

        logger.info(f"Routed → {path.value} | intent={intent.value} | complexity={complexity:.2f}")
        return routed
