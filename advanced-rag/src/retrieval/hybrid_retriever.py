"""
Hybrid retrieval using:

    BM25  -> sparse keyword retrieval
    FAISS -> dense semantic retrieval
    RRF   -> rank-based result fusion

The retriever does not combine raw BM25 and FAISS scores because
the two scoring systems operate on different scales.

Instead, Reciprocal Rank Fusion (RRF) combines their rankings:

    RRF(d) = sum(
        weight_r / (rrf_k + rank_r(d))
    )

A deterministic chunk_id is used to identify the same chunk across
the BM25 and FAISS stores.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..ingestion.chunker import Chunk
from ..utils.logger import logger
from .bm25_retriever import BM25Store
from .vector_store import VectorStore


@dataclass(slots=True)
class HybridResult:
    """
    Result produced by hybrid BM25 + FAISS retrieval.

    The individual ranks and scores are retained for transparency
    and later research/evaluation.
    """

    chunk: Chunk
    score: float

    dense_rank: Optional[int] = None
    dense_score: Optional[float] = None

    sparse_rank: Optional[int] = None
    sparse_score: Optional[float] = None

    def as_tuple(self) -> tuple[Chunk, float]:
        """
        Return the same basic shape used by the individual
        retrievers.
        """
        return self.chunk, self.score


class HybridRetriever:
    """
    Combines dense FAISS retrieval and sparse BM25 retrieval
    using Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        bm25_store: Optional[BM25Store] = None,
        rrf_k: int = 60,
    ) -> None:

        if rrf_k <= 0:
            raise ValueError(
                "rrf_k must be greater than zero."
            )

        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.rrf_k = rrf_k

    # ------------------------------------------------------------------
    # PROPERTIES
    # ------------------------------------------------------------------

    @property
    def has_dense(self) -> bool:
        """Return True when the FAISS store is ready."""
        return (
            self.vector_store is not None
            and self.vector_store.is_ready
        )

    @property
    def has_sparse(self) -> bool:
        """Return True when the BM25 store is ready."""
        return (
            self.bm25_store is not None
            and self.bm25_store.is_ready
        )

    @property
    def is_ready(self) -> bool:
        """
        Return True when at least one retrieval source is ready.
        """
        return self.has_dense or self.has_sparse

    def __repr__(self) -> str:
        return (
            "HybridRetriever("
            f"dense={'ready' if self.has_dense else 'off'}, "
            f"sparse={'ready' if self.has_sparse else 'off'}, "
            f"rrf_k={self.rrf_k}"
            ")"
        )

    # ------------------------------------------------------------------
    # RETRIEVAL
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        query_embedding: Optional[np.ndarray] = None,
        top_k: int = 10,
        candidate_pool: int = 5,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
    ) -> list[HybridResult]:
        """
        Retrieve and fuse results from FAISS and BM25.

        Parameters
        ----------
        query:
            Original user query. Used by BM25.

        query_embedding:
            Precomputed BGE embedding for the query.
            Required for dense retrieval.

        top_k:
            Number of final hybrid results.

        candidate_pool:
            Number of candidates retrieved from each source.

        dense_weight:
            Weight applied to dense RRF contribution.

        sparse_weight:
            Weight applied to sparse RRF contribution.

        Returns
        -------
        list[HybridResult]
            Results sorted by descending RRF score.
        """

        if not query or not query.strip():
            return []

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        if candidate_pool <= 0:
            raise ValueError(
                "candidate_pool must be greater than zero."
            )

        if dense_weight < 0:
            raise ValueError(
                "dense_weight must be non-negative."
            )

        if sparse_weight < 0:
            raise ValueError(
                "sparse_weight must be non-negative."
            )

        if dense_weight == 0 and sparse_weight == 0:
            raise ValueError(
                "At least one retrieval weight must be greater than zero."
            )

        if not self.is_ready:
            logger.warning(
                "Hybrid retrieval skipped | "
                "no ready retriever"
            )
            return []

        start_time = time.perf_counter()

        # --------------------------------------------------------------
        # Dense retrieval
        # --------------------------------------------------------------

        dense_results: list[tuple[Chunk, float]] = []

        dense_start = time.perf_counter()

        if self.has_dense:

            if query_embedding is not None:

                pool = min(
                    candidate_pool,
                    self.vector_store.size,
                )

                dense_results = self.vector_store.search(
                    query_embedding,
                    top_k=pool,
                )

            else:

                logger.warning(
                    "Dense retrieval skipped | "
                    "query embedding not provided"
                )

        dense_time = time.perf_counter() - dense_start

        # --------------------------------------------------------------
        # Sparse retrieval
        # --------------------------------------------------------------

        sparse_results: list[tuple[Chunk, float]] = []

        sparse_start = time.perf_counter()

        if self.has_sparse:

            pool = min(
                candidate_pool,
                self.bm25_store.size,
            )

            sparse_results = self.bm25_store.search(
                query,
                top_k=pool,
            )

        sparse_time = time.perf_counter() - sparse_start

        # --------------------------------------------------------------
        # No candidates
        # --------------------------------------------------------------

        if not dense_results and not sparse_results:

            total_time = time.perf_counter() - start_time

            logger.info(
                "Hybrid retrieval | "
                "dense=0 | "
                "sparse=0 | "
                "results=0 | "
                "total=%.4fs",
                total_time,
            )

            return []

        # --------------------------------------------------------------
        # RRF fusion
        # --------------------------------------------------------------

        fusion_start = time.perf_counter()

        fused_results = self._fuse(
            dense_results=dense_results,
            sparse_results=sparse_results,
            top_k=top_k,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
        )

        fusion_time = time.perf_counter() - fusion_start

        total_time = time.perf_counter() - start_time

        # --------------------------------------------------------------
        # Logging
        # --------------------------------------------------------------

        logger.info(
            "Hybrid retrieval | "
            "dense_candidates=%d | "
            "sparse_candidates=%d | "
            "fused_results=%d | "
            "dense_time=%.4fs | "
            "sparse_time=%.4fs | "
            "fusion_time=%.4fs | "
            "total=%.4fs",
            len(dense_results),
            len(sparse_results),
            len(fused_results),
            dense_time,
            sparse_time,
            fusion_time,
            total_time,
        )

        return fused_results

    # ------------------------------------------------------------------
    # RRF FUSION
    # ------------------------------------------------------------------

    def _fuse(
        self,
        dense_results: list[tuple[Chunk, float]],
        sparse_results: list[tuple[Chunk, float]],
        top_k: int,
        dense_weight: float,
        sparse_weight: float,
    ) -> list[HybridResult]:
        """
        Fuse dense and sparse rankings using weighted RRF.
        """

        # --------------------------------------------------------------
        # Store chunk information by deterministic chunk ID
        # --------------------------------------------------------------

        chunk_lookup: dict[str, Chunk] = {}

        # Dense rankings and scores
        dense_rank: dict[str, int] = {}
        dense_score: dict[str, float] = {}

        for rank, (chunk, score) in enumerate(
            dense_results,
            start=1,
        ):

            chunk_id = chunk.chunk_id

            dense_rank[chunk_id] = rank
            dense_score[chunk_id] = float(score)

            chunk_lookup[chunk_id] = chunk

        # Sparse rankings and scores
        sparse_rank: dict[str, int] = {}
        sparse_score: dict[str, float] = {}

        for rank, (chunk, score) in enumerate(
            sparse_results,
            start=1,
        ):

            chunk_id = chunk.chunk_id

            sparse_rank[chunk_id] = rank
            sparse_score[chunk_id] = float(score)

            # Preserve dense chunk object when available.
            chunk_lookup.setdefault(
                chunk_id,
                chunk,
            )

        # --------------------------------------------------------------
        # Calculate RRF scores
        # --------------------------------------------------------------

        all_chunk_ids = (
            set(dense_rank)
            | set(sparse_rank)
        )

        fused_scores: dict[str, float] = {}

        for chunk_id in all_chunk_ids:

            rrf_score = 0.0

            # Dense contribution
            if chunk_id in dense_rank:

                rrf_score += (
                    dense_weight
                    / (
                        self.rrf_k
                        + dense_rank[chunk_id]
                    )
                )

            # Sparse contribution
            if chunk_id in sparse_rank:

                rrf_score += (
                    sparse_weight
                    / (
                        self.rrf_k
                        + sparse_rank[chunk_id]
                    )
                )

            fused_scores[chunk_id] = rrf_score

        # --------------------------------------------------------------
        # Sort by RRF score
        # --------------------------------------------------------------

        ranked_ids = sorted(
            fused_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        ranked_ids = ranked_ids[:top_k]

        # --------------------------------------------------------------
        # Create final result objects
        # --------------------------------------------------------------

        results = []

        for chunk_id, score in ranked_ids:

            results.append(
                HybridResult(
                    chunk=chunk_lookup[chunk_id],
                    score=float(score),

                    dense_rank=dense_rank.get(
                        chunk_id
                    ),

                    dense_score=dense_score.get(
                        chunk_id
                    ),

                    sparse_rank=sparse_rank.get(
                        chunk_id
                    ),

                    sparse_score=sparse_score.get(
                        chunk_id
                    ),
                )
            )

        return results

    # ------------------------------------------------------------------
    # BATCH RETRIEVAL
    # ------------------------------------------------------------------

    def retrieve_batch(
        self,
        queries: list[str],
        query_embeddings: Optional[list[np.ndarray]] = None,
        top_k: int = 10,
        candidate_pool: int = 5,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
    ) -> list[list[HybridResult]]:
        """
        Perform hybrid retrieval for multiple queries.

        If query_embeddings is provided, it must have the same
        number of elements as queries.
        """

        if query_embeddings is not None:

            if len(query_embeddings) != len(queries):
                raise ValueError(
                    "query_embeddings must have the same "
                    "length as queries."
                )

        results = []

        for index, query in enumerate(queries):

            embedding = None

            if query_embeddings is not None:
                embedding = query_embeddings[index]

            query_results = self.retrieve(
                query=query,
                query_embedding=embedding,
                top_k=top_k,
                candidate_pool=candidate_pool,
                dense_weight=dense_weight,
                sparse_weight=sparse_weight,
            )

            results.append(query_results)

        return results