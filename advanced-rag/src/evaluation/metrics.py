"""
Standard information-retrieval metrics for evaluating RAG retrievers.

All functions take:
    ranked_ids:   list[str]  -- chunk/doc ids returned by a retriever,
                                 best match first.
    relevant_ids: set[str]   -- ground-truth relevant ids for that query.

"Accuracy" for a RAG retriever is usually reported as Hit Rate@k
(a.k.a. Success@k): the % of queries where at least one correct
document appears in the top-k results. That is the headline number
for the bar chart. MRR and Recall@k are reported alongside it because
Hit Rate alone doesn't reward putting the right answer at rank 1
instead of rank 10.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Set


def hit_at_k(ranked_ids: Sequence[str], relevant_ids: Set[str], k: int) -> float:
    """1.0 if any relevant id appears in the top-k, else 0.0."""
    if not relevant_ids:
        return 0.0
    return 1.0 if any(doc_id in relevant_ids for doc_id in ranked_ids[:k]) else 0.0


def recall_at_k(ranked_ids: Sequence[str], relevant_ids: Set[str], k: int) -> float:
    """Fraction of all relevant ids that were found in the top-k."""
    if not relevant_ids:
        return 0.0
    hits = len(set(ranked_ids[:k]) & relevant_ids)
    return hits / len(relevant_ids)


def precision_at_k(ranked_ids: Sequence[str], relevant_ids: Set[str], k: int) -> float:
    """Fraction of the top-k results that are relevant."""
    if k == 0:
        return 0.0
    hits = len(set(ranked_ids[:k]) & relevant_ids)
    return hits / k


def reciprocal_rank(ranked_ids: Sequence[str], relevant_ids: Set[str]) -> float:
    """1 / rank of the first relevant result (0 if none found)."""
    for rank, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_ids: Sequence[str], relevant_ids: Set[str], k: int) -> float:
    """Binary-relevance NDCG@k (standard for BEIR-style eval)."""
    import math

    dcg = 0.0
    for rank, doc_id in enumerate(ranked_ids[:k], start=1):
        if doc_id in relevant_ids:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))

    return dcg / idcg if idcg > 0 else 0.0


def evaluate_single_query(
    ranked_ids: Sequence[str],
    relevant_ids: Set[str],
    k_values: Iterable[int] = (1, 3, 5, 10),
) -> Dict[str, float]:
    """Compute every metric, at every k, for one query."""
    result: Dict[str, float] = {}
    for k in k_values:
        result[f"hit@{k}"] = hit_at_k(ranked_ids, relevant_ids, k)
        result[f"recall@{k}"] = recall_at_k(ranked_ids, relevant_ids, k)
        result[f"precision@{k}"] = precision_at_k(ranked_ids, relevant_ids, k)
        result[f"ndcg@{k}"] = ndcg_at_k(ranked_ids, relevant_ids, k)
    result["mrr"] = reciprocal_rank(ranked_ids, relevant_ids)
    return result


def aggregate(per_query_results: List[Dict[str, float]]) -> Dict[str, float]:
    """Average every metric across all queries."""
    if not per_query_results:
        return {}
    keys = per_query_results[0].keys()
    n = len(per_query_results)
    return {key: sum(r[key] for r in per_query_results) / n for key in keys}
