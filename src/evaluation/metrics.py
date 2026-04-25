"""
Evaluation Metrics
=====================
Calculates MRR, Top-k Accuracy, Accuracy, Recall, and Faithfulness.
"""

from typing import Dict, List

from loguru import logger


def mean_reciprocal_rank(results: List[List[int]], ground_truths: List[int]) -> float:
    """
    Calculate Mean Reciprocal Rank.
    
    Args:
        results: For each query, ranked list of document IDs
        ground_truths: For each query, the correct document ID
        
    Returns:
        MRR score (0-1)
    """
    mrr_sum = 0.0

    for ranked_ids, truth in zip(results, ground_truths):
        for rank, doc_id in enumerate(ranked_ids, 1):
            if doc_id == truth:
                mrr_sum += 1.0 / rank
                break

    mrr = mrr_sum / max(len(results), 1)
    logger.info(f"MRR: {mrr:.4f}")
    return mrr


def top_k_accuracy(results: List[List[int]], ground_truths: List[int], k: int = 10) -> float:
    """
    Calculate Top-k Accuracy.
    
    Args:
        results: For each query, ranked list of document IDs
        ground_truths: For each query, the correct document ID
        k: Number of top results to consider
        
    Returns:
        Top-k accuracy (0-1)
    """
    hits = sum(
        1 for ranked, truth in zip(results, ground_truths)
        if truth in ranked[:k]
    )

    acc = hits / max(len(results), 1)
    logger.info(f"Top-{k} Accuracy: {acc:.4f}")
    return acc


def recall_at_k(results: List[List[int]], ground_truths: List[List[int]], k: int = 10) -> float:
    """
    Calculate Recall@k.
    
    Args:
        results: For each query, ranked list of document IDs
        ground_truths: For each query, list of relevant document IDs
        
    Returns:
        Recall@k (0-1)
    """
    recall_sum = 0.0

    for ranked, truths in zip(results, ground_truths):
        retrieved_set = set(ranked[:k])
        truth_set = set(truths)
        if truth_set:
            recall_sum += len(retrieved_set & truth_set) / len(truth_set)

    recall = recall_sum / max(len(results), 1)
    logger.info(f"Recall@{k}: {recall:.4f}")
    return recall


def accuracy(predictions: List[str], ground_truths: List[str]) -> float:
    """Calculate answer accuracy (exact match)."""
    correct = sum(
        1 for pred, truth in zip(predictions, ground_truths)
        if pred.strip().lower() == truth.strip().lower()
    )
    acc = correct / max(len(predictions), 1)
    logger.info(f"Accuracy: {acc:.4f}")
    return acc


def faithfulness(citations_verified: List[bool]) -> float:
    """Calculate faithfulness score."""
    if not citations_verified:
        return 0.0
    score = sum(citations_verified) / len(citations_verified)
    logger.info(f"Faithfulness: {score:.4f}")
    return score


def compute_all_metrics(
    retrieval_results: List[List[int]],
    retrieval_truths: List[int],
    answer_predictions: List[str],
    answer_truths: List[str],
    citation_verified: List[bool],
    k: int = 10,
) -> Dict[str, float]:
    """Compute all target KPIs."""
    return {
        "mrr": mean_reciprocal_rank(retrieval_results, retrieval_truths),
        "top_k_accuracy": top_k_accuracy(retrieval_results, retrieval_truths, k),
        "accuracy": accuracy(answer_predictions, answer_truths),
        "faithfulness": faithfulness(citation_verified),
    }
