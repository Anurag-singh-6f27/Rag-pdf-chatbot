from __future__ import annotations

import json
import math
import os
import re
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from ..ingestion.chunker import Chunk
from ..utils.logger import logger
from config import settings


# ----------------------------------------------------------------------
# BM25 TOKENIZER
# ----------------------------------------------------------------------

# BM25 works on complete terms/words.
# The BGE tokenizer used for embeddings should NOT be reused here.
_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> List[str]:
    """
    Tokenize text for BM25.

    Converts text to lowercase and keeps alphanumeric terms.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string.")

    return _TOKEN_PATTERN.findall(text.lower())


# ----------------------------------------------------------------------
# BM25 STORE
# ----------------------------------------------------------------------

class BM25Store:
    """
    Persistent Okapi BM25 sparse keyword index.

    BM25 operates directly on chunk text and is independent
    of the FAISS dense vector index.

    Responsibilities:
        - Tokenize chunk text
        - Build inverted index
        - Calculate IDF
        - Calculate BM25 relevance scores
        - Rank chunks
        - Persist index to disk
        - Load persisted index
        - Validate index consistency
    """

    def __init__(
        self,
        k1: float | None = None,
        b: float | None = None,
    ) -> None:

        # --------------------------------------------------------------
        # BM25 parameters
        # --------------------------------------------------------------

        self.k1 = (
            k1
            if k1 is not None
            else getattr(settings, "BM25_K1", 1.5)
        )

        self.b = (
            b
            if b is not None
            else getattr(settings, "BM25_B", 0.75)
        )

        if self.k1 <= 0:
            raise ValueError("BM25 k1 must be greater than 0.")

        if not 0 <= self.b <= 1:
            raise ValueError("BM25 b must be between 0 and 1.")

        # --------------------------------------------------------------
        # Indexed chunks
        # --------------------------------------------------------------

        self.chunks: List[Chunk] = []

        # Number of tokens in every chunk
        self.doc_lengths: List[int] = []

        # Average chunk length
        self.avgdl: float = 0.0

        # --------------------------------------------------------------
        # Inverted index
        #
        # term -> {
        #     chunk_index: term_frequency
        # }
        # --------------------------------------------------------------

        self._inverted_index: Dict[str, Dict[int, int]] = {}

        # --------------------------------------------------------------
        # Precomputed IDF
        #
        # term -> idf score
        # --------------------------------------------------------------

        self._idf: Dict[str, float] = {}

        # --------------------------------------------------------------
        # Persistent index path
        # --------------------------------------------------------------

        self._index_path = Path(settings.BM25_INDEX_PATH)

    # ==================================================================
    # PROPERTIES
    # ==================================================================

    @property
    def is_ready(self) -> bool:
        """
        Return True if the BM25 index contains searchable documents.
        """
        return bool(self.chunks) and bool(self._inverted_index)

    @property
    def size(self) -> int:
        """
        Number of chunks indexed.
        """
        return len(self.chunks)

    @property
    def vocabulary_size(self) -> int:
        """
        Number of unique terms in the vocabulary.
        """
        return len(self._idf)

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return (
            f"BM25Store("
            f"size={self.size}, "
            f"vocabulary={self.vocabulary_size}, "
            f"ready={self.is_ready})"
        )

    # ==================================================================
    # BUILD
    # ==================================================================

    def build(self, chunks: List[Chunk]) -> None:
        """
        Build the BM25 inverted index from document chunks.

        BM25 uses only chunk text. No embeddings are required.
        """

        if not chunks:
            raise ValueError(
                "Cannot build BM25 index: chunk list is empty."
            )

        start_time = time.perf_counter()

        inverted_index: Dict[str, Dict[int, int]] = defaultdict(dict)
        doc_lengths: List[int] = []

        # --------------------------------------------------------------
        # Build term-frequency postings
        # --------------------------------------------------------------

        for doc_index, chunk in enumerate(chunks):

            if not isinstance(chunk.text, str):
                raise TypeError(
                    f"Chunk {doc_index} text must be a string."
                )

            terms = tokenize(chunk.text)

            if not terms:
                raise ValueError(
                    f"Chunk {doc_index} contains no BM25 tokens."
                )

            doc_lengths.append(len(terms))

            term_frequencies = Counter(terms)

            for term, term_frequency in term_frequencies.items():
                inverted_index[term][doc_index] = term_frequency

        # --------------------------------------------------------------
        # Corpus statistics
        # --------------------------------------------------------------

        num_docs = len(chunks)

        avgdl = (
            sum(doc_lengths) / num_docs
            if num_docs > 0
            else 0.0
        )

        if avgdl <= 0:
            raise ValueError(
                "Average document length must be greater than zero."
            )

        # --------------------------------------------------------------
        # Calculate IDF
        #
        # Lucene-style BM25 IDF:
        #
        # log(
        #     1 + (N - df + 0.5) / (df + 0.5)
        # )
        # --------------------------------------------------------------

        idf: Dict[str, float] = {}

        for term, postings in inverted_index.items():

            document_frequency = len(postings)

            idf[term] = math.log(
                1.0
                + (
                    num_docs
                    - document_frequency
                    + 0.5
                )
                / (
                    document_frequency
                    + 0.5
                )
            )

        # --------------------------------------------------------------
        # Commit index
        # --------------------------------------------------------------

        self._inverted_index = dict(inverted_index)
        self._idf = idf
        self.doc_lengths = doc_lengths
        self.avgdl = avgdl
        self.chunks = list(chunks)

        elapsed = time.perf_counter() - start_time

        logger.info(
            "BM25 index built | "
            "documents=%d | "
            "unique_terms=%d | "
            "avgdl=%.1f | "
            "time=%.4fs",
            num_docs,
            len(idf),
            avgdl,
            elapsed,
        )

    # ==================================================================
    # SEARCH
    # ==================================================================

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Tuple[Chunk, float]]:
        """
        Search the BM25 index.

        Returns:
            List of (Chunk, BM25 score), sorted by descending score.

        Chunks with no matching query terms are not returned.
        """

        if not self.is_ready:
            return []

        if not isinstance(query, str) or not query.strip():
            raise ValueError(
                "Query must be a non-empty string."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        query_terms = Counter(tokenize(query))

        if not query_terms:
            return []

        start_time = time.perf_counter()

        scores: Dict[int, float] = defaultdict(float)

        # --------------------------------------------------------------
        # Calculate BM25 score
        # --------------------------------------------------------------

        for term, query_term_frequency in query_terms.items():

            idf = self._idf.get(term)

            # Term does not exist in corpus.
            if idf is None:
                continue

            postings = self._inverted_index.get(term, {})

            for doc_index, term_frequency in postings.items():

                doc_length = self.doc_lengths[doc_index]

                # BM25 document-length normalization
                length_normalization = (
                    1.0
                    - self.b
                    + self.b
                    * (
                        doc_length
                        / self.avgdl
                    )
                )

                denominator = (
                    term_frequency
                    + self.k1
                    * length_normalization
                )

                term_score = (
                    idf
                    * (
                        term_frequency
                        * (self.k1 + 1.0)
                    )
                    / denominator
                )

                # Query term frequency contribution
                scores[doc_index] += (
                    term_score
                    * query_term_frequency
                )

        # --------------------------------------------------------------
        # Rank only documents that matched at least one term
        # --------------------------------------------------------------

        ranked = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        ranked = ranked[:top_k]

        results = [
            (
                self.chunks[doc_index],
                float(score),
            )
            for doc_index, score in ranked
        ]

        elapsed = time.perf_counter() - start_time

        matched_terms = sum(
            1
            for term in query_terms
            if term in self._idf
        )

        logger.info(
            "BM25 search | "
            "top_k=%d | "
            "matched_terms=%d | "
            "results=%d | "
            "time=%.4fs",
            top_k,
            matched_terms,
            len(results),
            elapsed,
        )

        return results

    # ==================================================================
    # BATCH SEARCH
    # ==================================================================

    def search_batch(
        self,
        queries: List[str],
        top_k: int = 5,
    ) -> List[List[Tuple[Chunk, float]]]:
        """
        Search multiple queries.

        This is mainly useful for evaluation and benchmarking.
        """

        if not queries:
            return []

        if any(
            not isinstance(query, str) or not query.strip()
            for query in queries
        ):
            raise ValueError(
                "All queries must be non-empty strings."
            )

        return [
            self.search(
                query,
                top_k=top_k,
            )
            for query in queries
        ]

    # ==================================================================
    # SAVE
    # ==================================================================

    def save(self, pretty: bool = False) -> None:
        """
        Persist the BM25 index to JSON.

        The temporary file is atomically replaced into the
        final index path.
        """

        if not self.is_ready:
            raise ValueError(
                "Cannot save BM25 index: index is empty."
            )

        self._index_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        start_time = time.perf_counter()

        # --------------------------------------------------------------
        # Prepare serializable payload
        # --------------------------------------------------------------

        payload = {
            "version": 1,
            "k1": self.k1,
            "b": self.b,
            "avgdl": self.avgdl,
            "doc_lengths": self.doc_lengths,

            "inverted_index": {
                term: {
                    str(doc_index): term_frequency
                    for doc_index, term_frequency
                    in postings.items()
                }
                for term, postings
                in self._inverted_index.items()
            },

            "idf": self._idf,

            "chunks": [
                self._chunk_to_dict(chunk)
                for chunk in self.chunks
            ],
        }

        # --------------------------------------------------------------
        # Atomic write
        # --------------------------------------------------------------

        fd, temp_name = tempfile.mkstemp(
            dir=self._index_path.parent,
            prefix=f".{self._index_path.name}.",
            suffix=".tmp",
        )

        temp_path = Path(temp_name)

        try:

            with os.fdopen(
                fd,
                "w",
                encoding="utf-8",
            ) as file:

                json.dump(
                    payload,
                    file,
                    ensure_ascii=False,
                    indent=2 if pretty else None,
                    separators=(
                        None
                        if pretty
                        else (",", ":")
                    ),
                )

                file.flush()
                os.fsync(file.fileno())

            os.replace(
                temp_path,
                self._index_path,
            )

        finally:
            temp_path.unlink(
                missing_ok=True
            )

        elapsed = time.perf_counter() - start_time

        logger.info(
            "BM25 index saved | "
            "documents=%d | "
            "vocabulary=%d | "
            "path=%s | "
            "time=%.4fs",
            self.size,
            self.vocabulary_size,
            self._index_path,
            elapsed,
        )

    # ==================================================================
    # LOAD
    # ==================================================================

    def load(self) -> bool:
        """
        Load a previously saved BM25 index.

        Returns:
            True if loading succeeds.
            False if the index does not exist or is invalid.
        """

        if not self._index_path.exists():
            logger.info(
                "BM25 index not found: %s",
                self._index_path,
            )
            return False

        start_time = time.perf_counter()

        try:

            with self._index_path.open(
                "r",
                encoding="utf-8",
            ) as file:

                payload = json.load(file)

            # ----------------------------------------------------------
            # Validate required fields
            # ----------------------------------------------------------

            required_fields = {
                "version",
                "k1",
                "b",
                "avgdl",
                "doc_lengths",
                "inverted_index",
                "idf",
                "chunks",
            }

            missing_fields = (
                required_fields
                - payload.keys()
            )

            if missing_fields:
                raise ValueError(
                    "Invalid BM25 index. "
                    f"Missing fields: "
                    f"{sorted(missing_fields)}"
                )

            # ----------------------------------------------------------
            # Validate BM25 parameters
            # ----------------------------------------------------------

            loaded_k1 = float(payload["k1"])
            loaded_b = float(payload["b"])

            if loaded_k1 <= 0:
                raise ValueError(
                    "Invalid BM25 k1 value."
                )

            if not 0 <= loaded_b <= 1:
                raise ValueError(
                    "Invalid BM25 b value."
                )

            # ----------------------------------------------------------
            # Restore chunks
            # ----------------------------------------------------------

            chunks = [
                self._chunk_from_dict(item)
                for item in payload["chunks"]
            ]

            # ----------------------------------------------------------
            # Restore inverted index
            # ----------------------------------------------------------

            inverted_index = {
                term: {
                    int(doc_index): int(term_frequency)
                    for doc_index, term_frequency
                    in postings.items()
                }
                for term, postings
                in payload["inverted_index"].items()
            }

            # ----------------------------------------------------------
            # Restore document statistics
            # ----------------------------------------------------------

            doc_lengths = [
                int(length)
                for length in payload["doc_lengths"]
            ]

            avgdl = float(payload["avgdl"])

            idf = {
                str(term): float(score)
                for term, score
                in payload["idf"].items()
            }

            # ----------------------------------------------------------
            # Integrity validation
            # ----------------------------------------------------------

            if len(doc_lengths) != len(chunks):
                raise ValueError(
                    "BM25 index and metadata are inconsistent: "
                    f"index contains {len(doc_lengths)} "
                    f"document lengths, but metadata contains "
                    f"{len(chunks)} chunks."
                )

            if len(chunks) == 0:
                raise ValueError(
                    "Loaded BM25 index contains no chunks."
                )

            if avgdl <= 0:
                raise ValueError(
                    "Loaded BM25 average document length is invalid."
                )

            # Validate posting references
            num_chunks = len(chunks)

            for term, postings in inverted_index.items():

                for doc_index, term_frequency in postings.items():

                    if not 0 <= doc_index < num_chunks:
                        raise ValueError(
                            "BM25 inverted index contains "
                            f"invalid document index: {doc_index}"
                        )

                    if term_frequency <= 0:
                        raise ValueError(
                            "BM25 term frequency must be positive."
                        )

            # ----------------------------------------------------------
            # Commit loaded state
            # ----------------------------------------------------------

            self.k1 = loaded_k1
            self.b = loaded_b
            self.avgdl = avgdl
            self.doc_lengths = doc_lengths
            self._inverted_index = inverted_index
            self._idf = idf
            self.chunks = chunks

            elapsed = time.perf_counter() - start_time

            logger.info(
                "BM25 index loaded | "
                "documents=%d | "
                "vocabulary=%d | "
                "time=%.4fs",
                self.size,
                self.vocabulary_size,
                elapsed,
            )

            return True

        except Exception as exc:

            # Do not leave partially loaded state.
            self._reset()

            logger.warning(
                "Failed to load BM25 index: %s",
                exc,
            )

            return False

    # ==================================================================
    # EXISTS
    # ==================================================================

    def exists(self) -> bool:
        """
        Return True if a persisted BM25 index exists.
        """
        return self._index_path.exists()

    # ==================================================================
    # CLEAR
    # ==================================================================

    def clear(self) -> None:
        """
        Delete the persisted BM25 index and reset memory.
        """

        if self._index_path.exists():
            self._index_path.unlink()

            logger.info(
                "Deleted BM25 index: %s",
                self._index_path,
            )

        self._reset()

    # ==================================================================
    # RESET
    # ==================================================================

    def _reset(self) -> None:
        """
        Reset the in-memory BM25 state.
        """

        self.chunks = []
        self.doc_lengths = []
        self.avgdl = 0.0
        self._inverted_index = {}
        self._idf = {}

    # ==================================================================
    # SERIALIZATION
    # ==================================================================

    @staticmethod
    def _chunk_to_dict(chunk: Chunk) -> dict:
        """
        Convert Chunk to JSON-compatible dictionary.
        """

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
        """
        Reconstruct Chunk from stored metadata.
        """

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

        missing_fields = (
            required_fields
            - data.keys()
        )

        if missing_fields:
            raise ValueError(
                "Invalid chunk metadata. "
                f"Missing fields: "
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