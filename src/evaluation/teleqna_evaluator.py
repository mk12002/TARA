"""
TeleQnA Evaluator
====================
Evaluates TARA against the TeleQnA benchmark dataset.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger


class TeleQnAEvaluator:
    """Evaluates TARA on TeleQnA multiple-choice questions."""

    def __init__(self, dataset_path: str = "data/raw/teleqna/TeleQnA.json"):
        self.dataset_path = Path(dataset_path)
        self.questions = []

    def load_dataset(self) -> int:
        """Load TeleQnA dataset. Returns number of questions loaded."""
        if not self.dataset_path.exists():
            logger.error(f"TeleQnA dataset not found at: {self.dataset_path}")
            logger.info("Download from: https://github.com/netop-team/TeleQnA")
            return 0

        with open(self.dataset_path) as f:
            data = json.load(f)

        # TeleQnA format: list of questions with options and answer
        if isinstance(data, dict):
            self.questions = list(data.values())
        elif isinstance(data, list):
            self.questions = data

        logger.info(f"Loaded {len(self.questions)} TeleQnA questions")
        return len(self.questions)

    def evaluate(
        self,
        answer_fn,
        max_questions: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Run evaluation.
        
        Args:
            answer_fn: Callable(question_text, options) -> predicted_answer_key
            max_questions: Limit evaluation to N questions
            
        Returns:
            Dict with accuracy and per-category scores
        """
        if not self.questions:
            self.load_dataset()

        questions = self.questions[:max_questions] if max_questions else self.questions

        correct = 0
        total = 0
        results = []

        for q in questions:
            question_text = q.get("question", "")
            options = {
                k: v for k, v in q.items()
                if k.startswith("option") or k in ("A", "B", "C", "D")
            }
            ground_truth = q.get("answer", "")

            try:
                prediction = answer_fn(question_text, options)
                is_correct = prediction.strip() == ground_truth.strip()
                if is_correct:
                    correct += 1
            except Exception as e:
                logger.warning(f"Error on question: {e}")
                is_correct = False
                prediction = ""

            total += 1
            results.append({
                "question": question_text[:100],
                "predicted": prediction,
                "ground_truth": ground_truth,
                "correct": is_correct,
            })

        accuracy = correct / max(total, 1)
        logger.info(f"TeleQnA Accuracy: {accuracy:.4f} ({correct}/{total})")

        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "results": results,
        }
