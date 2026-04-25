"""
Hypothesis Generator
======================
Generates RCA hypotheses from failure pattern KB and retrieval context.
"""

from typing import Dict, List, Optional

from loguru import logger
from src.reasoning.rca_agent import Hypothesis, FAILURE_PATTERNS


class HypothesisGenerator:
    """
    Generates ranked hypotheses for root cause analysis.
    
    Strategy: Pattern-match first, retrieve-augmented LLM fallback.
    """

    def __init__(self, retrieval_pipeline=None, llm_client=None):
        self.retrieval = retrieval_pipeline
        self.llm = llm_client
        self.patterns = FAILURE_PATTERNS

    def generate(self, problem_info: Dict) -> List[Hypothesis]:
        """
        Generate hypotheses for a given problem.
        
        Args:
            problem_info: Dict with 'symptom', 'network_element', 'procedure'
            
        Returns:
            Ranked list of hypotheses (max 5)
        """
        symptom = problem_info.get("symptom", "").lower()
        hypotheses = []

        # Phase 1: Pattern matching (fast, deterministic)
        for pattern_name, pattern in self.patterns.items():
            if any(s.lower() in symptom or symptom in s.lower() for s in pattern["symptoms"]):
                for i, cause in enumerate(pattern["common_causes"]):
                    hypotheses.append(Hypothesis(
                        id=f"{pattern_name}_{i}",
                        description=cause["cause"],
                        probability=round(1.0 / (i + 1), 2),
                    ))

        if hypotheses:
            logger.info(f"Pattern-matched {len(hypotheses)} hypotheses")
            return hypotheses[:5]

        # Phase 2: Retrieval-augmented LLM fallback
        if self.retrieval and self.llm:
            hypotheses = self._llm_fallback(problem_info)

        if not hypotheses:
            hypotheses = [Hypothesis(
                id="unknown_0",
                description="Unable to determine hypothesis — insufficient pattern data",
                probability=0.1,
            )]

        return hypotheses[:5]

    def _llm_fallback(self, problem_info: Dict) -> List[Hypothesis]:
        """Use retrieval + LLM to generate hypotheses when no pattern matches."""
        query = f"Root causes for {problem_info.get('symptom', '')} in telecom networks"

        try:
            docs = self.retrieval.retrieve(query, top_k=5)
            context = "\n".join(d.text[:300] for d in docs)

            response = self.llm.generate(
                f"Based on this context, list 3 possible root causes:\n{context}\n\nProblem: {problem_info.get('symptom', '')}"
            )

            # Parse numbered list
            hypotheses = []
            for i, line in enumerate(response.text.split("\n")):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith("-")):
                    hypotheses.append(Hypothesis(
                        id=f"llm_{i}",
                        description=line.lstrip("0123456789.-) "),
                        probability=0.3,
                    ))

            return hypotheses
        except Exception as e:
            logger.warning(f"LLM hypothesis fallback failed: {e}")
            return []
