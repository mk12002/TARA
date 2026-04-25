"""
RAGAS Evaluator
==================
Wraps the RAGAS framework for RAG evaluation.
"""

from typing import Dict, List, Optional

from loguru import logger


class RAGASEvaluator:
    """
    Evaluates RAG pipeline using RAGAS metrics:
    - Faithfulness
    - Answer Relevancy
    - Context Precision
    - Context Recall
    """

    def __init__(self):
        self._ragas_available = False
        try:
            import ragas
            self._ragas_available = True
        except ImportError:
            logger.warning("RAGAS not installed. Install with: pip install ragas")

    def evaluate(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: List[str],
    ) -> Dict[str, float]:
        """
        Run RAGAS evaluation.
        
        Args:
            questions: List of user queries
            answers: List of generated answers
            contexts: List of retrieved context lists (per query)
            ground_truths: List of reference answers
            
        Returns:
            Dict of RAGAS metric scores
        """
        if not self._ragas_available:
            logger.error("RAGAS not available. Install with: pip install ragas")
            return {}

        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            )

            # Build RAGAS dataset
            data = {
                "question": questions,
                "answer": answers,
                "contexts": contexts,
                "ground_truth": ground_truths,
            }
            dataset = Dataset.from_dict(data)

            # Run evaluation
            result = evaluate(
                dataset,
                metrics=[
                    faithfulness,
                    answer_relevancy,
                    context_precision,
                    context_recall,
                ],
            )

            scores = {k: float(v) for k, v in result.items() if isinstance(v, (int, float))}
            logger.info(f"RAGAS scores: {scores}")
            return scores

        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}")
            return {"error": str(e)}
