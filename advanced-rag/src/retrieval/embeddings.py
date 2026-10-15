"""
Embedding generation using Sentence Transformers.

The embedding model is loaded once and reused for the lifetime
of the application.
"""

import threading
import time

import numpy as np
from sentence_transformers import SentenceTransformer

from ..ingestion.chunker import Chunk
from ..utils.logger import logger
from config.settings import EMBEDDING_MODEL


EMBEDDING_MODEL_NAME = EMBEDDING_MODEL

_model = None
_model_lock = threading.Lock()


def get_embedding_model() -> SentenceTransformer:
    """
    Load and cache the embedding model.

    The model is loaded only once per process.
    """

    global _model

    if _model is None:
        with _model_lock:
            if _model is None:

                start_time = time.perf_counter()

                logger.info(
                    "Loading embedding model: %s",
                    EMBEDDING_MODEL_NAME,
                )

                _model = SentenceTransformer(
                    EMBEDDING_MODEL_NAME
                )

                elapsed = time.perf_counter() - start_time

                logger.info(
                    "Embedding model loaded in %.4fs",
                    elapsed,
                )

    return _model


def embed_texts(
    texts: list[str],
    batch_size: int = 32,
) -> np.ndarray:
    """
    Generate normalized embeddings for a list of texts.

    Returns:
        np.ndarray with shape:
        (number_of_texts, embedding_dimension)
    """

    if not texts:
        return np.empty(
            (0, 0),
            dtype=np.float32,
        )

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be greater than zero."
        )

    if any(
        not isinstance(text, str) or not text.strip()
        for text in texts
    ):
        raise ValueError(
            "All texts must be non-empty strings."
        )

    start_time = time.perf_counter()

    model = get_embedding_model()

    logger.info(
        "Generating embeddings | texts=%d | batch_size=%d",
        len(texts),
        batch_size,
    )

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    if embeddings.ndim != 2:
        raise ValueError(
            "Embedding model returned an unexpected shape: "
            f"{embeddings.shape}"
        )

    if embeddings.shape[0] != len(texts):
        raise ValueError(
            "Embedding count does not match text count."
        )

    if not np.all(np.isfinite(embeddings)):
        raise ValueError(
            "Generated embeddings contain NaN or infinite values."
        )

    elapsed = time.perf_counter() - start_time

    logger.info(
        "Embeddings generated | vectors=%d | dimension=%d | time=%.4fs",
        embeddings.shape[0],
        embeddings.shape[1],
        elapsed,
    )

    return embeddings


def embed_chunks(
    chunks: list[Chunk],
    batch_size: int = 32,
) -> np.ndarray:
    """
    Generate embeddings for document chunks.
    """

    if not chunks:
        return np.empty(
            (0, 0),
            dtype=np.float32,
        )

    texts = [
        chunk.text
        for chunk in chunks
    ]

    return embed_texts(
        texts=texts,
        batch_size=batch_size,
    )


def embedding_dimension() -> int:
    """
    Return the embedding model's vector dimension.
    """

    model = get_embedding_model()

    dimension = model.get_sentence_embedding_dimension()

    if dimension is None:
        raise RuntimeError(
            "Unable to determine embedding dimension."
        )

    return int(dimension)

def embed_query(query: str) -> np.ndarray:
    """
    Embed a single query string.

    Returns a normalized 1D vector with shape (dimension,),
    matching what VectorStore.search() and HybridRetriever expect.
    """

    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string.")

    return embed_texts([query.strip()])[0]