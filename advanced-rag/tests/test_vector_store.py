"""
Integration test for the FAISS vector store.

Pipeline tested:

PDF
 ↓
PDF extraction
 ↓
Chunking
 ↓
BGE embeddings
 ↓
FAISS build
 ↓
FAISS save
 ↓
FAISS load
 ↓
Dense retrieval
 ↓
Batch retrieval
"""

import time
from pathlib import Path

import numpy as np

from src.ingestion.pdf_extractor import process_pdf
from src.ingestion.chunker import chunk_document
from src.retrieval.embeddings import embed_chunks, embed_texts
from src.retrieval.vector_store import VectorStore


# ============================================================
# CONFIGURATION
# ============================================================

PDF_PATH = Path(r"C:\Users\anura\OneDrive\Documents\study plan.pdf")

TOP_K = 5
BATCH_QUERY_COUNT = 3


# ============================================================
# HELPERS
# ============================================================

def print_separator() -> None:
    print("-" * 70)


def validate_results(
    results,
    expected_max_results: int,
) -> None:
    """Validate FAISS search results."""

    assert isinstance(results, list)

    assert len(results) <= expected_max_results

    for chunk, score in results:
        assert chunk is not None

        assert isinstance(score, float)

        assert np.isfinite(score)

        assert chunk.text.strip()

        assert chunk.document_name

        assert chunk.chunk_id

        assert chunk.page_numbers

    print(
        f"Validation passed: {len(results)} search results are valid."
    )


# ============================================================
# MAIN TEST
# ============================================================

def main() -> None:

    total_start = time.perf_counter()

    print()
    print("=" * 70)
    print("ADVANCED RAG — VECTOR STORE TEST")
    print("=" * 70)
    print()

    # --------------------------------------------------------
    # 1. CHECK PDF
    # --------------------------------------------------------

    print("[1] Checking PDF...")

    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"PDF not found: {PDF_PATH}\n"
            "Place a PDF inside data/documents/ or update PDF_PATH."
        )

    print(f"PDF: {PDF_PATH}")
    print(f"Size: {PDF_PATH.stat().st_size:,} bytes")

    # --------------------------------------------------------
    # 2. PROCESS PDF
    # --------------------------------------------------------

    print()
    print("[2] Processing PDF...")
    print_separator()

    start = time.perf_counter()

    with PDF_PATH.open("rb") as file:
        file_bytes = file.read()

    document = process_pdf(
        PDF_PATH.name,
        file_bytes,
    )

    extraction_time = time.perf_counter() - start

    print(f"Document: {document.name}")
    print(f"Pages: {len(document.pages)}")
    print(f"Characters: {sum(len(page.text) for page in document.pages):,}")
    print(f"Extraction time: {extraction_time:.4f}s")

    assert document is not None
    assert len(document.pages) > 0
    assert sum(len(page.text) for page in document.pages) > 0

    # --------------------------------------------------------
    # 3. CHUNK DOCUMENT
    # --------------------------------------------------------

    print()
    print("[3] Chunking document...")
    print_separator()

    start = time.perf_counter()

    chunks, _ = chunk_document(document)

    chunking_time = time.perf_counter() - start

    print(f"Chunks generated: {len(chunks)}")
    print(f"Chunking time: {chunking_time:.4f}s")

    assert chunks
    assert len(chunks) > 0

    for chunk in chunks:

        assert chunk.text.strip()

        assert chunk.token_count > 0

        assert chunk.document_name == document.name

        assert chunk.page_numbers

    print("Chunk validation passed.")

    # --------------------------------------------------------
    # 4. GENERATE EMBEDDINGS
    # --------------------------------------------------------

    print()
    print("[4] Generating embeddings...")
    print_separator()

    start = time.perf_counter()

    embeddings = embed_chunks(
        chunks,
        batch_size=32,
    )

    embedding_time = time.perf_counter() - start

    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embedding dtype: {embeddings.dtype}")
    print(f"Embedding time: {embedding_time:.4f}s")

    # Basic embedding validation.

    assert embeddings.ndim == 2

    assert embeddings.shape[0] == len(chunks)

    assert embeddings.shape[1] > 0

    assert embeddings.dtype == np.float32

    assert np.all(np.isfinite(embeddings))

    # Check normalization.
    norms = np.linalg.norm(
        embeddings,
        axis=1,
    )

    max_norm_error = np.max(
        np.abs(norms - 1.0)
    )

    print(
        f"Maximum embedding norm error: "
        f"{max_norm_error:.8f}"
    )

    assert max_norm_error < 1e-5

    print("Embedding validation passed.")

    # --------------------------------------------------------
    # 5. BUILD VECTOR STORE
    # --------------------------------------------------------

    print()
    print("[5] Building FAISS vector store...")
    print_separator()

    store = VectorStore()

    start = time.perf_counter()

    store.build(
        chunks=chunks,
        embeddings=embeddings,
    )

    build_time = time.perf_counter() - start

    print(f"FAISS vectors: {store.size}")
    print(f"FAISS dimension: {store.dimension}")
    print(f"Build time: {build_time:.4f}s")

    assert store.is_ready

    assert store.size == len(chunks)

    assert store.dimension == embeddings.shape[1]

    print("FAISS build validation passed.")

    # --------------------------------------------------------
    # 6. SAVE VECTOR STORE
    # --------------------------------------------------------

    print()
    print("[6] Saving vector store...")
    print_separator()

    start = time.perf_counter()

    store.save()

    save_time = time.perf_counter() - start

    print(f"Save time: {save_time:.4f}s")

    assert store.exists()

    print("Vector store files exist.")

    # --------------------------------------------------------
    # 7. LOAD INTO A NEW VECTOR STORE
    # --------------------------------------------------------

    print()
    print("[7] Loading vector store...")
    print_separator()

    loaded_store = VectorStore()

    start = time.perf_counter()

    loaded = loaded_store.load()

    load_time = time.perf_counter() - start

    print(f"Load successful: {loaded}")
    print(f"Load time: {load_time:.4f}s")

    assert loaded is True

    assert loaded_store.is_ready

    assert loaded_store.size == len(chunks)

    assert loaded_store.dimension == embeddings.shape[1]

    assert len(loaded_store.chunks) == len(chunks)

    # Verify metadata survived the save/load cycle.

    for original, loaded_chunk in zip(
        chunks,
        loaded_store.chunks,
    ):
        assert original.chunk_id == loaded_chunk.chunk_id

        assert original.document_name == loaded_chunk.document_name

        assert original.document_hash == loaded_chunk.document_hash

        assert original.page_numbers == loaded_chunk.page_numbers

        assert original.text == loaded_chunk.text

    print("Save/load metadata validation passed.")

    # --------------------------------------------------------
    # 8. SINGLE QUERY SEARCH
    # --------------------------------------------------------

    print()
    print("[8] Testing single-query FAISS search...")
    print_separator()

    query_text = (
        "What are the main topics covered in the study plan?"
    )

    print(f"Query: {query_text}")

    start = time.perf_counter()

    query_embedding = embed_texts(
        [query_text],
        batch_size=1,
    )

    embedding_query_time = time.perf_counter() - start

    start = time.perf_counter()

    results = loaded_store.search(
        query_embedding[0],
        top_k=TOP_K,
    )

    search_time = time.perf_counter() - start

    print()
    print(f"Query embedding time: {embedding_query_time:.4f}s")
    print(f"FAISS search time: {search_time:.4f}s")
    print(f"Results returned: {len(results)}")

    validate_results(
        results,
        TOP_K,
    )

    print()
    print("TOP RESULTS")
    print_separator()

    for rank, (chunk, score) in enumerate(
        results,
        start=1,
    ):
        preview = (
            chunk.text.replace("\n", " ")[:180]
        )

        print(
            f"{rank}. "
            f"score={score:.4f} | "
            f"pages={chunk.page_numbers} | "
            f"chunk={chunk.chunk_index}"
        )

        print(f"   {preview}...")
        print()

    # --------------------------------------------------------
    # 9. TEST BATCH SEARCH
    # --------------------------------------------------------

    print()
    print("[9] Testing batch FAISS search...")
    print_separator()

    batch_queries = [
        "What programming languages are mentioned?",
        "What machine learning topics are included?",
        "What cloud technologies are discussed?",
    ]

    batch_queries = batch_queries[:BATCH_QUERY_COUNT]

    query_embeddings = embed_texts(
        batch_queries,
        batch_size=32,
    )

    assert query_embeddings.shape[0] == len(batch_queries)

    start = time.perf_counter()

    batch_results = loaded_store.search_batch(
        query_embeddings,
        top_k=TOP_K,
    )

    batch_search_time = time.perf_counter() - start

    print(
        f"Queries: {len(batch_queries)}"
    )

    print(
        f"Batch search time: "
        f"{batch_search_time:.4f}s"
    )

    assert len(batch_results) == len(batch_queries)

    for query, results_for_query in zip(
        batch_queries,
        batch_results,
    ):
        print()
        print(f"Query: {query}")
        print(
            f"Results: "
            f"{len(results_for_query)}"
        )

        validate_results(
            results_for_query,
            TOP_K,
        )

    print()
    print("Batch search validation passed.")

    # --------------------------------------------------------
    # 10. CHECK RESULT ORDER
    # --------------------------------------------------------

    print()
    print("[10] Checking similarity-score ordering...")
    print_separator()

    for results_for_query in batch_results:

        scores = [
            score
            for _, score in results_for_query
        ]

        assert scores == sorted(
            scores,
            reverse=True,
        )

    print(
        "All FAISS results are ordered "
        "from highest to lowest similarity."
    )

    # --------------------------------------------------------
    # 11. FINAL SUMMARY
    # --------------------------------------------------------

    total_time = time.perf_counter() - total_start

    print()
    print("=" * 70)
    print("VECTOR STORE TEST COMPLETED")
    print("=" * 70)

    print()
    print("SUMMARY")
    print_separator()

    print(f"Document:          {document.name}")
    print(f"Pages:             {len(document.pages)}")
    print(f"Chunks:            {len(chunks)}")
    print(f"Embedding shape:   {embeddings.shape}")
    print(f"FAISS vectors:     {store.size}")
    print(f"FAISS dimension:   {store.dimension}")
    print(f"Build time:        {build_time:.4f}s")
    print(f"Save time:         {save_time:.4f}s")
    print(f"Load time:         {load_time:.4f}s")
    print(f"Single search:     {search_time:.4f}s")
    print(f"Batch search:      {batch_search_time:.4f}s")
    print(f"Total test time:   {total_time:.4f}s")

    print()
    print("=" * 70)
    print("ALL VECTOR STORE TESTS PASSED")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()