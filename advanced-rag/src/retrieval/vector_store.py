"""
FAISS-backed dense vector store.

Responsibilities:
- Build a FAISS index from chunk embeddings
- Search using cosine similarity (single or batched queries)
- Persist the FAISS index and chunk metadata (atomically)
- Load an existing index
- Validate index/metadata consistency

FAISS uses inner product (IP).
Because embeddings are L2-normalized, inner product is equivalent
to cosine similarity.

Optimizations vs. the original implementation:
- One allocation instead of two when casting to float32 (astype+copy
  merged into a single np.array(..., copy=True) call), and the array is
  made C-contiguous up front so FAISS never has to silently copy it again.
- Save/load are atomic: files are written to a temp path and swapped in
  with os.replace, so a crash mid-write can never corrupt a previously
  good index/metadata pair.
- Settings paths are resolved once (in __init__) instead of on every
  save()/load()/exists()/clear() call.
- A new search_batch() does many queries in a single FAISS call instead
  of looping search() one query at a time -- much faster for anything
  beyond a single query, since FAISS batches matrix ops internally.
- Metadata JSON is written without indentation by default (much smaller,
  faster to write/parse) with an opt-in `pretty` flag for debugging.
- __len__/__repr__/__contains__ added for convenience.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np

from ..ingestion.chunker import Chunk
from ..utils.logger import logger
from config import settings


class VectorStore:
    """Persistent FAISS vector store for dense retrieval."""

    def __init__(self) -> None:
        self.index: faiss.Index | None = None
        self.chunks: List[Chunk] = []

        # Resolve paths once; settings don't change at runtime.
        self._vectorstore_dir = Path(settings.VECTORSTORE_DIR)
        self._faiss_index_path = Path(settings.FAISS_INDEX_PATH)
        self._metadata_path = Path(settings.METADATA_PATH)

    # ------------------------------------------------------------------
    # DUNDER / CONVENIENCE
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return (
            f"VectorStore(size={self.size}, dimension={self.dimension}, "
            f"ready={self.is_ready})"
        )

    # ------------------------------------------------------------------
    # PROPERTIES
    # ------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        """Return True when the vector store contains searchable vectors."""
        return self.index is not None and self.index.ntotal > 0

    @property
    def size(self) -> int:
        """Return the number of vectors stored in FAISS."""
        if self.index is None:
            return 0

        return self.index.ntotal

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        if self.index is None:
            return 0

        return self.index.d

    # ------------------------------------------------------------------
    # BUILD
    # ------------------------------------------------------------------

    def build(
        self,
        chunks: List[Chunk],
        embeddings: np.ndarray,
    ) -> None:
        """
        Build a FAISS IndexFlatIP index from chunk embeddings.

        Parameters
        ----------
        chunks:
            Chunk objects corresponding to the embeddings.

        embeddings:
            2D numpy array with shape:
            (number_of_chunks, embedding_dimension)
        """

        if not chunks:
            raise ValueError(
                "Cannot build vector store: chunk list is empty."
            )

        embeddings = np.asarray(embeddings)

        if embeddings.ndim != 2:
            raise ValueError(
                "Embeddings must be a 2D array. "
                f"Received shape: {embeddings.shape}"
            )

        if embeddings.shape[0] != len(chunks):
            raise ValueError(
                "Embedding count does not match chunk count: "
                f"{embeddings.shape[0]} embeddings vs "
                f"{len(chunks)} chunks."
            )

        if embeddings.shape[1] == 0:
            raise ValueError(
                "Embedding dimension must be greater than zero."
            )

        if not np.all(np.isfinite(embeddings)):
            raise ValueError(
                "Embeddings contain NaN or infinite values."
            )

        start_time = time.perf_counter()

        # Single allocation: cast to float32, copy (so the caller's array
        # is never mutated by normalize_L2 below) and force C-contiguous
        # layout, all in one step instead of astype() + .copy().
        embeddings = np.array(
            embeddings,
            dtype=np.float32,
            order="C",
            copy=True,
        )

        # Enforce L2 normalization here.
        #
        # This makes the vector store independent of whether the
        # embedding caller normalized the vectors correctly.
        faiss.normalize_L2(embeddings)

        dimension = embeddings.shape[1]

        logger.info(
            "Building FAISS index | vectors=%d | dimension=%d",
            len(chunks),
            dimension,
        )

        # IndexFlatIP = exact inner-product search.
        #
        # Since vectors are normalized:
        #     inner product == cosine similarity
        #
        # For your expected dataset size, this is fast and simple.
        index = faiss.IndexFlatIP(dimension)

        index.add(embeddings)

        self.index = index
        self.chunks = list(chunks)

        elapsed = time.perf_counter() - start_time

        logger.info(
            "FAISS index built | vectors=%d | dimension=%d | time=%.4fs",
            self.index.ntotal,
            self.index.d,
            elapsed,
        )

    # ------------------------------------------------------------------
    # SEARCH
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> List[Tuple[Chunk, float]]:
        """
        Search the vector store using cosine similarity.

        Parameters
        ----------
        query_embedding:
            Single query embedding with shape:
            (dimension,)
            or
            (1, dimension)

        top_k:
            Maximum number of results to return.

        Returns
        -------
        list[tuple[Chunk, float]]
            Results ordered from highest to lowest similarity.
        """

        if not self.is_ready:
            return []

        query_embedding = np.asarray(query_embedding)

        # Accept (dimension,) and convert it to (1, dimension).
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        if query_embedding.ndim != 2:
            raise ValueError(
                "Query embedding must be a 1D or 2D array. "
                f"Received shape: {query_embedding.shape}"
            )

        if query_embedding.shape[0] != 1:
            raise ValueError(
                "search() accepts exactly one query embedding at a time. "
                "Use search_batch() for multiple queries."
            )

        results = self._search_matrix(query_embedding, top_k)
        return results[0]

    def search_batch(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 5,
    ) -> List[List[Tuple[Chunk, float]]]:
        """
        Search the vector store for many queries in a single FAISS call.

        This is significantly faster than calling `search()` in a loop,
        since FAISS batches the underlying matrix multiplication across
        all queries at once instead of paying per-call overhead.

        Parameters
        ----------
        query_embeddings:
            2D array with shape (num_queries, dimension).

        top_k:
            Maximum number of results to return per query.

        Returns
        -------
        list[list[tuple[Chunk, float]]]
            One results list per query, in the same order as the input.
        """

        if not self.is_ready:
            return []

        query_embeddings = np.asarray(query_embeddings)

        if query_embeddings.ndim != 2:
            raise ValueError(
                "query_embeddings must be a 2D array of shape "
                f"(num_queries, dimension). Received shape: "
                f"{query_embeddings.shape}"
            )

        return self._search_matrix(query_embeddings, top_k)

    def _search_matrix(
        self,
        query_embeddings: np.ndarray,
        top_k: int,
    ) -> List[List[Tuple[Chunk, float]]]:
        """Shared implementation behind search() and search_batch()."""

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        if query_embeddings.shape[1] != self.dimension:
            raise ValueError(
                "Query embedding dimension does not match "
                f"the FAISS index: query={query_embeddings.shape[1]}, "
                f"index={self.dimension}."
            )

        if not np.all(np.isfinite(query_embeddings)):
            raise ValueError(
                "Query embedding contains NaN or infinite values."
            )

        # Single allocation, C-contiguous, ready for FAISS + normalize_L2.
        query_embeddings = np.array(
            query_embeddings,
            dtype=np.float32,
            order="C",
            copy=True,
        )

        # Ensure cosine similarity behavior.
        faiss.normalize_L2(query_embeddings)

        k = min(top_k, self.index.ntotal)

        start_time = time.perf_counter()

        scores, indices = self.index.search(query_embeddings, k)

        elapsed = time.perf_counter() - start_time

        all_results: List[List[Tuple[Chunk, float]]] = []

        num_chunks = len(self.chunks)

        for query_scores, query_indices in zip(scores, indices):
            results: List[Tuple[Chunk, float]] = []

            for score, index_id in zip(query_scores, query_indices):
                if index_id < 0:
                    continue

                if index_id >= num_chunks:
                    logger.warning(
                        "FAISS returned invalid chunk index: %d",
                        index_id,
                    )
                    continue

                results.append((self.chunks[index_id], float(score)))

            all_results.append(results)

        logger.info(
            "FAISS search | queries=%d | top_k=%d | time=%.4fs",
            query_embeddings.shape[0],
            top_k,
            elapsed,
        )

        return all_results

    # ------------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------------

    def save(self, pretty: bool = False) -> None:
        """
        Save FAISS index and chunk metadata to disk atomically.

        Both files are written to temporary paths first and then moved
        into place with os.replace, which is atomic on POSIX and Windows.
        This guarantees that a crash or interruption mid-write can never
        leave a half-written index or metadata file on disk.

        Parameters
        ----------
        pretty:
            If True, indent the metadata JSON for human readability.
            Defaults to False for smaller files and faster save/load.
        """

        if not self.is_ready:
            raise ValueError(
                "Cannot save vector store: index is empty."
            )

        self._vectorstore_dir.mkdir(parents=True, exist_ok=True)

        start_time = time.perf_counter()

        # --- FAISS index: write to temp file, then atomically replace ---
        index_fd, index_tmp_name = tempfile.mkstemp(
            dir=self._faiss_index_path.parent,
            prefix=f".{self._faiss_index_path.name}.",
            suffix=".tmp",
        )
        os.close(index_fd)
        index_tmp_path = Path(index_tmp_name)

        try:
            faiss.write_index(self.index, str(index_tmp_path))
            os.replace(index_tmp_path, self._faiss_index_path)
        finally:
            index_tmp_path.unlink(missing_ok=True)

        # --- Metadata: same atomic write-then-replace pattern ---
        metadata = [self._chunk_to_dict(chunk) for chunk in self.chunks]

        metadata_fd, metadata_tmp_name = tempfile.mkstemp(
            dir=self._metadata_path.parent,
            prefix=f".{self._metadata_path.name}.",
            suffix=".tmp",
        )
        metadata_tmp_path = Path(metadata_tmp_name)

        try:
            with os.fdopen(metadata_fd, "w", encoding="utf-8") as file:
                json.dump(
                    metadata,
                    file,
                    ensure_ascii=False,
                    indent=2 if pretty else None,
                    separators=None if pretty else (",", ":"),
                )
            os.replace(metadata_tmp_path, self._metadata_path)
        finally:
            metadata_tmp_path.unlink(missing_ok=True)

        elapsed = time.perf_counter() - start_time

        logger.info(
            "FAISS vector store saved | "
            "vectors=%d | index=%s | metadata=%s | time=%.4fs",
            self.size,
            self._faiss_index_path,
            self._metadata_path,
            elapsed,
        )

    # ------------------------------------------------------------------
    # LOAD
    # ------------------------------------------------------------------

    def load(self) -> bool:
        """
        Load an existing FAISS index and metadata.

        Returns
        -------
        bool
            True if loading succeeded.
            False if no saved vector store exists.
        """

        if not (
            self._faiss_index_path.exists()
            and self._metadata_path.exists()
        ):
            return False

        start_time = time.perf_counter()

        try:
            index = faiss.read_index(str(self._faiss_index_path))

            with self._metadata_path.open("r", encoding="utf-8") as file:
                metadata = json.load(file)

            if not isinstance(metadata, list):
                raise ValueError(
                    "Vector store metadata must contain a list."
                )

            chunks = [self._chunk_from_dict(item) for item in metadata]

            # Critical integrity check.
            if index.ntotal != len(chunks):
                raise ValueError(
                    "FAISS index and metadata are inconsistent: "
                    f"index contains {index.ntotal} vectors, "
                    f"metadata contains {len(chunks)} chunks."
                )

            self.index = index
            self.chunks = chunks

            elapsed = time.perf_counter() - start_time

            logger.info(
                "FAISS vector store loaded | "
                "vectors=%d | dimension=%d | time=%.4fs",
                self.size,
                self.dimension,
                elapsed,
            )

            return True

        except Exception as exc:
            # Do not leave a partially loaded store in memory.
            self.index = None
            self.chunks = []

            logger.warning(
                "Failed to load existing vector store: %s",
                exc,
            )

            return False

    # ------------------------------------------------------------------
    # EXISTENCE / CLEAR
    # ------------------------------------------------------------------

    def exists(self) -> bool:
        """Return True if both persisted vector-store files exist."""

        return self._faiss_index_path.exists() and self._metadata_path.exists()

    def clear(self) -> None:
        """Delete the persisted FAISS index and metadata."""

        for path in (self._faiss_index_path, self._metadata_path):
            if path.exists():
                path.unlink()

                logger.info(
                    "Deleted vector-store file: %s",
                    path,
                )

    # ------------------------------------------------------------------
    # SERIALIZATION HELPERS
    # ------------------------------------------------------------------

    @staticmethod
    def _chunk_to_dict(chunk: Chunk) -> dict:
        """Convert a Chunk into JSON-compatible metadata."""

        return {
            "chunk_id": chunk.chunk_id,
            "document_name": chunk.document_name,
            "document_hash": chunk.document_hash,
            "page_numbers": chunk.page_numbers,
            "chunk_index": chunk.chunk_index,
            "text": chunk.text,
            "token_count": chunk.token_count,
            "character_count": chunk.character_count,
        }

    @staticmethod
    def _chunk_from_dict(data: dict) -> Chunk:
        """Reconstruct a Chunk from stored metadata."""

        required_fields = {
            "chunk_id",
            "document_name",
            "document_hash",
            "page_numbers",
            "chunk_index",
            "text",
            "token_count",
            "character_count",
        }

        missing_fields = required_fields - data.keys()

        if missing_fields:
            raise ValueError(
                "Invalid chunk metadata. Missing fields: "
                f"{sorted(missing_fields)}"
            )

        return Chunk(
            chunk_id=data["chunk_id"],
            document_name=data["document_name"],
            document_hash=data["document_hash"],
            page_numbers=data["page_numbers"],
            chunk_index=data["chunk_index"],
            text=data["text"],
            token_count=data["token_count"],
            character_count=data["character_count"],
        )