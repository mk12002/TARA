"""
Intent Classifier
===================
Classifies user queries into intent categories to determine routing.
"""

import re
from enum import Enum
from typing import Optional

from loguru import logger


class QueryIntent(str, Enum):
    QNA = "QNA"                # Factual question
    RCA = "RCA"                # Root cause analysis
    ANOMALY = "ANOMALY"        # Anomaly detection
    COMPARISON = "COMPARISON"  # Technology comparison


# Keyword-based classification rules (fast, no LLM needed)
INTENT_RULES = {
    QueryIntent.RCA: [
        "root cause", "rca", "failure", "fault", "alarm", "degradation",
        "why is", "why does", "troubleshoot", "diagnose", "debug",
        "not working", "dropped", "high packet loss",
    ],
    QueryIntent.ANOMALY: [
        "anomaly", "anomalous", "unusual", "spike", "degraded",
        "performance issue", "threshold", "kpi violation",
    ],
    QueryIntent.COMPARISON: [
        "compare", "comparison", "difference between", "vs",
        "versus", "which is better", "trade-off", "pros and cons",
    ],
}


class IntentClassifier:
    """
    Classifies query intent using keyword rules (fast path)
    with optional LLM fallback for ambiguous queries.
    """

    def classify(self, query: str) -> QueryIntent:
        """
        Classify a query into an intent category.
        
        Uses keyword matching for speed. Falls back to QNA for
        queries that don't match any specific pattern.
        """
        query_lower = query.lower()

        for intent, keywords in INTENT_RULES.items():
            for keyword in keywords:
                if keyword in query_lower:
                    logger.debug(f"Intent: {intent.value} (matched: '{keyword}')")
                    return intent

        # Default to QNA for general questions
        logger.debug(f"Intent: QNA (default)")
        return QueryIntent.QNA
