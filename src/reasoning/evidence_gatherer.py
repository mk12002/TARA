"""
Evidence Gatherer
====================
Gathers supporting and contradicting evidence for RCA hypotheses.
"""

from typing import Dict, List

from loguru import logger
from src.reasoning.rca_agent import Hypothesis


class EvidenceGatherer:
    """
    Gathers evidence for/against hypotheses from retrieval context.
    Updates hypothesis posteriors based on evidence strength.
    """

    def __init__(self, retrieval_pipeline=None):
        self.retrieval = retrieval_pipeline

    def gather(
        self,
        hypotheses: List[Hypothesis],
        problem_info: Dict,
    ) -> List[Hypothesis]:
        """
        For each hypothesis, search for supporting/contradicting evidence.
        
        Updates hypothesis posteriors using simple Bayesian update.
        """
        for hypothesis in hypotheses:
            if self.retrieval:
                self._search_evidence(hypothesis, problem_info)
            self._update_posterior(hypothesis)

        # Sort by posterior
        hypotheses.sort(key=lambda h: h.posterior, reverse=True)

        logger.info(
            f"Evidence gathered for {len(hypotheses)} hypotheses. "
            f"Top: {hypotheses[0].description[:50]}... (p={hypotheses[0].posterior:.2f})"
            if hypotheses else "No hypotheses"
        )
        return hypotheses

    def _search_evidence(self, hypothesis: Hypothesis, problem_info: Dict):
        """Search retrieval for evidence related to a hypothesis."""
        query = f"{hypothesis.description} {problem_info.get('symptom', '')}"

        try:
            docs = self.retrieval.retrieve(query, top_k=3)
            for doc in docs:
                # Simple heuristic: if doc mentions the cause, it's supporting
                if any(
                    word in doc.text.lower()
                    for word in hypothesis.description.lower().split()[:3]
                ):
                    hypothesis.supporting_evidence.append(doc.text[:200])
                else:
                    hypothesis.contradicting_evidence.append(doc.text[:200])
        except Exception as e:
            logger.warning(f"Evidence search failed: {e}")

    def _update_posterior(self, hypothesis: Hypothesis):
        """
        Simple Bayesian-style posterior update.
        
        posterior = prior * (1 + support_weight - contradict_weight)
        """
        support_weight = len(hypothesis.supporting_evidence) * 0.2
        contradict_weight = len(hypothesis.contradicting_evidence) * 0.15

        hypothesis.posterior = hypothesis.probability * (1 + support_weight - contradict_weight)
        hypothesis.posterior = max(0.01, min(0.99, hypothesis.posterior))
