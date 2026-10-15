from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, Union

from src.ingestion.chunker import Chunk
from src.retrieval.hybrid_retriever import HybridResult
from src.utils.logger import logger

if TYPE_CHECKING:
    # Only seen by the type checker, never executed at runtime. This gives
    # Pylance a name that is *always* the real flashrank.Ranker class, so
    # it can be used in annotations below -- unlike the runtime `Ranker`
    # name a few lines down, which is sometimes `None` (when flashrank
    # isn't installed) and therefore isn't valid as a type on its own.
    from flashrank import Ranker as RankerType

try:
    from flashrank import Ranker, RerankRequest
except ImportError as exc:
    Ranker = None
    RerankRequest = None
    _IMPORT_ERROR: Optional[ImportError] = exc
else:
    _IMPORT_ERROR = None

# ms-marco-TinyBERT-L-2-v2 (the flashrank library default) is a 2-layer
# cross-encoder: very fast and very small, but weak -- it consistently
# underperforms the larger MS MARCO cross-encoders on accuracy. Since
# this project's evaluation showed reranking sometimes *hurting* Hit
# Rate versus hybrid retrieval alone, the default here is upgraded to
# the 12-layer MiniLM cross-encoder, which trades a bit of latency for
# meaningfully better relevance judgments. Override model_name at call
# sites if you need the tiny model back for latency-sensitive paths.
DEFAULT_MODEL_NAME = "ms-marco-MiniLM-L-12-v2"
DEFAULT_MAX_LENGTH = 512

RerankCandidate = Union[Chunk, HybridResult, Tuple[Chunk, float]]


@dataclass(slots=True)
class RerankedResult:
    chunk: Chunk
    rerank_score: float
    rerank_rank: int
    original_score: Optional[float] = None
    original_rank: Optional[int] = None

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    @property
    def document_name(self) -> str:
        return self.chunk.document_name

    @property
    def page_numbers(self) -> List[int]:
        return self.chunk.page_numbers

    @property
    def text(self) -> str:
        return self.chunk.text

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_name": self.document_name,
            "page_numbers": self.page_numbers,
            "text": self.text,
            "rerank_score": self.rerank_score,
            "rerank_rank": self.rerank_rank,
            "original_score": self.original_score,
            "original_rank": self.original_rank,
        }


@dataclass(slots=True)
class RerankResponse:
    results: List[RerankedResult]
    query: str
    model_name: str
    num_candidates: int
    latency_seconds: float

    def __iter__(self):
        return iter(self.results)

    def __len__(self) -> int:
        return len(self.results)

    def as_tuples(self) -> List[Tuple[Chunk, float]]:
        return [(result.chunk, result.rerank_score) for result in self.results]


def _normalize_candidate(candidate: RerankCandidate) -> Tuple[Chunk, Optional[float]]:
    if isinstance(candidate, HybridResult):
        return candidate.chunk, float(candidate.score)
    if isinstance(candidate, Chunk):
        return candidate, None
    if isinstance(candidate, tuple) and len(candidate) == 2:
        chunk, score = candidate
        if not isinstance(chunk, Chunk):
            raise TypeError(
                "Expected a (Chunk, score) tuple, got "
                f"({type(chunk).__name__}, {type(score).__name__})."
            )
        return chunk, float(score)
    raise TypeError(
        "Each candidate must be a Chunk, a (Chunk, score) tuple, "
        "or a HybridResult. "
        f"Received: {type(candidate).__name__}"
    )


class FlashRankReranker:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        cache_dir: Optional[str] = None,
        max_length: int = DEFAULT_MAX_LENGTH,
        metrics_log_path: Optional[str] = None,
    ) -> None:
        if Ranker is None:
            raise ImportError(
                "flashrank is required for reranking but is not installed. "
                "Install it with: pip install flashrank"
            ) from _IMPORT_ERROR
        if not model_name or not model_name.strip():
            raise ValueError("model_name cannot be empty.")
        if max_length <= 0:
            raise ValueError("max_length must be greater than zero.")

        self.model_name = model_name
        self.cache_dir = cache_dir
        self.max_length = max_length
        self.metrics_log_path = Path(metrics_log_path) if metrics_log_path else None
        self._ranker: Optional[RankerType] = None

        logger.info(
            "FlashRank reranker configured | model=%s | max_length=%d",
            self.model_name, self.max_length,
        )

    def _get_ranker(self) -> RankerType:
        if self._ranker is None:
            start_time = time.perf_counter()
            logger.info("Loading FlashRank model: %s", self.model_name)
            ranker_kwargs: Dict[str, Any] = {
                "model_name": self.model_name,
                "max_length": self.max_length,
            }
            if self.cache_dir:
                ranker_kwargs["cache_dir"] = self.cache_dir
            self._ranker = Ranker(**ranker_kwargs)
            elapsed = time.perf_counter() - start_time
            logger.info("FlashRank model loaded | model=%s | time=%.4fs", self.model_name, elapsed)
        return self._ranker

    @property
    def is_ready(self) -> bool:
        return self._ranker is not None

    def rerank(self, query: str, candidates: Sequence[RerankCandidate], top_k: int = 5) -> RerankResponse:
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string.")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if not candidates:
            logger.warning("rerank() called with no candidates.")
            return RerankResponse(results=[], query=query, model_name=self.model_name, num_candidates=0, latency_seconds=0.0)

        seen: set[str] = set()
        chunk_lookup: Dict[str, Tuple[Chunk, Optional[float], int]] = {}
        passages: List[Dict[str, Any]] = []

        for original_rank, candidate in enumerate(candidates, start=1):
            chunk, original_score = _normalize_candidate(candidate)
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            chunk_lookup[chunk.chunk_id] = (chunk, original_score, original_rank)
            passages.append({"id": chunk.chunk_id, "text": chunk.text})

        if not passages:
            return RerankResponse(results=[], query=query, model_name=self.model_name, num_candidates=0, latency_seconds=0.0)

        ranker = self._get_ranker()

        start_time = time.perf_counter()
        rerank_request = RerankRequest(query=query, passages=passages)
        raw_results = ranker.rerank(rerank_request)
        elapsed = time.perf_counter() - start_time

        raw_results = sorted(raw_results, key=lambda item: float(item["score"]), reverse=True)

        # Walk the full ranked list (not just the first top_k slots) and
        # backfill from further down whenever an id doesn't match a known
        # candidate. Without this, one bad/unknown id from FlashRank would
        # silently shrink the result set below top_k instead of just
        # skipping that one entry and moving on.
        results: List[RerankedResult] = []

        for item in raw_results:
            if len(results) >= top_k:
                break

            chunk_id = item["id"]
            if chunk_id not in chunk_lookup:
                logger.warning("FlashRank returned unknown chunk_id: %s", chunk_id)
                continue

            chunk, original_score, original_rank = chunk_lookup[chunk_id]
            results.append(RerankedResult(
                chunk=chunk, rerank_score=float(item["score"]), rerank_rank=len(results) + 1,
                original_score=original_score, original_rank=original_rank,
            ))

        logger.info(
            "FlashRank rerank | model=%s | candidates=%d | kept=%d | latency=%.4fs",
            self.model_name, len(passages), len(results), elapsed,
        )

        response = RerankResponse(
            results=results, query=query, model_name=self.model_name,
            num_candidates=len(passages), latency_seconds=elapsed,
        )

        if self.metrics_log_path is not None:
            self._log_metrics(response)

        return response

    def rerank_batch(self, queries, candidate_lists, top_k: int = 5) -> List[RerankResponse]:
        if len(queries) != len(candidate_lists):
            raise ValueError("queries and candidate_lists must have the same length.")
        return [self.rerank(query=q, candidates=c, top_k=top_k) for q, c in zip(queries, candidate_lists)]

    def _log_metrics(self, response: RerankResponse) -> None:
        if self.metrics_log_path is None:
            return
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_name": response.model_name,
            "query": response.query,
            "num_candidates": response.num_candidates,
            "num_kept": len(response.results),
            "latency_seconds": response.latency_seconds,
            "top_result_chunk_id": response.results[0].chunk_id if response.results else None,
        }
        self.metrics_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.metrics_log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def __repr__(self) -> str:
        return f"FlashRankReranker(model={self.model_name!r}, max_length={self.max_length}, ready={self.is_ready})"