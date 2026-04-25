"""
Uncertainty Handler
=====================
Handles low-confidence scenarios with graceful "I don't know" responses.
"""

from dataclasses import dataclass
from typing import Optional

from loguru import logger


@dataclass
class UncertaintyDecision:
    """Decision on how to handle uncertain responses."""
    action: str  # "answer", "qualify", "abstain"
    message: Optional[str] = None
    original_answer: str = ""


class UncertaintyHandler:
    """
    Decides what to do when confidence is low.
    
    Thresholds:
    - HIGH (>= 0.75): Answer normally
    - MEDIUM (0.4 - 0.75): Answer with qualification  
    - LOW (< 0.4): Abstain — "I don't have sufficient information"
    """

    HIGH_THRESHOLD = 0.75
    LOW_THRESHOLD = 0.4

    def handle(self, answer: str, confidence: float) -> UncertaintyDecision:
        """
        Decide how to handle a response based on confidence.
        """
        if confidence >= self.HIGH_THRESHOLD:
            return UncertaintyDecision(
                action="answer",
                original_answer=answer,
            )

        elif confidence >= self.LOW_THRESHOLD:
            qualifier = (
                "⚠️ **Note**: This answer has moderate confidence. "
                "Some information may be incomplete or require verification "
                "against the relevant 3GPP specifications.\n\n"
            )
            return UncertaintyDecision(
                action="qualify",
                message=qualifier + answer,
                original_answer=answer,
            )

        else:
            abstain_msg = (
                "I don't have sufficient information in my knowledge base to "
                "answer this question confidently. I recommend:\n\n"
                "1. Checking the relevant 3GPP specification directly\n"
                "2. Rephrasing your question with more specific details\n"
                "3. Consulting a domain expert for this particular topic\n\n"
                f"*Confidence: {confidence:.0%} — below threshold for reliable answers.*"
            )
            logger.warning(f"Abstaining due to low confidence ({confidence:.2f})")
            return UncertaintyDecision(
                action="abstain",
                message=abstain_msg,
                original_answer=answer,
            )
