"""
Complexity Scorer
===================
Scores query complexity to decide fast path vs. reasoning path.
"""

from loguru import logger
from src.query.intent_classifier import QueryIntent


class ComplexityScorer:
    """Scores query complexity (0-1) to determine routing."""

    # Weights for each complexity signal
    WEIGHTS = {
        "word_count": 0.2,
        "intent_complexity": 0.4,
        "multi_entity": 0.2,
        "comparison": 0.2,
    }

    COMPLEXITY_THRESHOLD = 0.5  # Above this → reasoning path

    def score(
        self,
        query: str,
        intent: QueryIntent,
        entity_count: int = 0,
    ) -> float:
        """
        Score query complexity.
        
        Returns:
            Float 0-1 where higher = more complex
        """
        scores = {}

        # Word count signal (longer = more complex)
        word_count = len(query.split())
        scores["word_count"] = min(word_count / 30, 1.0)

        # Intent complexity
        scores["intent_complexity"] = {
            QueryIntent.QNA: 0.2,
            QueryIntent.COMPARISON: 0.5,
            QueryIntent.ANOMALY: 0.7,
            QueryIntent.RCA: 0.9,
        }.get(intent, 0.3)

        # Multi-entity complexity
        scores["multi_entity"] = min(entity_count / 3, 1.0)

        # Weighted sum
        total = sum(
            scores[k] * self.WEIGHTS[k]
            for k in self.WEIGHTS
            if k in scores
        )

        logger.debug(f"Complexity: {total:.2f} (threshold: {self.COMPLEXITY_THRESHOLD})")
        return total

    def needs_reasoning(self, score: float) -> bool:
        """Whether this query needs the agentic reasoning path."""
        return score >= self.COMPLEXITY_THRESHOLD
