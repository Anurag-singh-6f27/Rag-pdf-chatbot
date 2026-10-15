from pathlib import Path
import time

from src.ingestion.pdf_extractor import process_pdf
from src.ingestion.chunker import chunk_document

from src.retrieval.bm25_retriever import BM25Store
from src.retrieval.vector_store import VectorStore
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.embeddings import get_embedding_model


PDF_PATH = Path(
    r"C:\Users\anura\OneDrive\Documents\study plan.pdf"
)


def main():

    total_start = time.perf_counter()

    print("=" * 70)
    print("ADVANCED RAG — HYBRID RETRIEVER TEST")
    print("=" * 70)

    # ===============================================================
    # 1. CHECK PDF
    # ===============================================================

    print("\n[1] Checking PDF...")
    print("-" * 70)

    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"PDF not found: {PDF_PATH}"
        )

    print(f"PDF: {PDF_PATH}")
    print(
        f"Size: "
        f"{PDF_PATH.stat().st_size:,} bytes"
    )

    # ===============================================================
    # 2. PROCESS PDF
    # ===============================================================

    print("\n[2] Processing PDF...")
    print("-" * 70)

    start = time.perf_counter()

    with PDF_PATH.open("rb") as file:
        file_bytes = file.read()

    document = process_pdf(
        PDF_PATH.name,
        file_bytes,
    )

    extraction_time = (
        time.perf_counter() - start
    )

    character_count = sum(
        len(page.text)
        for page in document.pages
    )

    print(
        f"Document: {PDF_PATH.name}"
    )

    print(
        f"Pages: {len(document.pages)}"
    )

    print(
        f"Characters: {character_count:,}"
    )

    print(
        f"Extraction time: "
        f"{extraction_time:.4f}s"
    )

    assert document is not None
    assert len(document.pages) > 0
    assert character_count > 0

    print("PDF validation passed.")

    # ===============================================================
    # 3. CHUNK DOCUMENT
    # ===============================================================

    print("\n[3] Chunking document...")
    print("-" * 70)

    start = time.perf_counter()

    chunks, duplicate_count = (
        chunk_document(document)
    )

    chunking_time = (
        time.perf_counter() - start
    )

    print(
        f"Chunks generated: "
        f"{len(chunks)}"
    )

    print(
        f"Chunking time: "
        f"{chunking_time:.4f}s"
    )

    print(
        f"Duplicates removed: "
        f"{duplicate_count}"
    )

    assert len(chunks) > 0

    for chunk in chunks:

        assert chunk.text.strip()
        assert chunk.token_count > 0
        assert chunk.chunk_id

    print("Chunk validation passed.")

    # ===============================================================
    # 4. BUILD BM25
    # ===============================================================

    print("\n[4] Building BM25...")
    print("-" * 70)

    bm25 = BM25Store()

    start = time.perf_counter()

    bm25.build(chunks)

    bm25_build_time = (
        time.perf_counter() - start
    )

    print(
        f"BM25 chunks: "
        f"{len(bm25)}"
    )

    print(
        f"BM25 vocabulary: "
        f"{bm25.vocabulary_size}"
    )

    print(
        f"BM25 average document length: "
        f"{bm25.avgdl:.2f}"
    )

    print(
        f"BM25 k1: "
        f"{bm25.k1}"
    )

    print(
        f"BM25 b: "
        f"{bm25.b}"
    )

    print(
        f"BM25 build time: "
        f"{bm25_build_time:.4f}s"
    )

    assert bm25.is_ready
    assert len(bm25) == len(chunks)
    assert bm25.vocabulary_size > 0
    assert bm25.avgdl > 0

    print("BM25 validation passed.")

    # ===============================================================
    # 5. LOAD EMBEDDING MODEL
    # ===============================================================

    print("\n[5] Loading embedding model...")
    print("-" * 70)

    start = time.perf_counter()

    embedding_model = get_embedding_model()

    embedding_model_load_time = (
        time.perf_counter() - start
    )

    print(
        f"Embedding model load time: "
        f"{embedding_model_load_time:.4f}s"
    )

    assert embedding_model is not None

    print("Embedding model validation passed.")

    # ===============================================================
    # 6. GENERATE CHUNK EMBEDDINGS
    # ===============================================================

    print("\n[6] Generating chunk embeddings...")
    print("-" * 70)

    chunk_texts = [
        chunk.text
        for chunk in chunks
    ]

    start = time.perf_counter()

    embeddings = embedding_model.encode(
        chunk_texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    embedding_time = (
        time.perf_counter() - start
    )

    print(
        f"Embeddings generated: "
        f"{len(embeddings)}"
    )

    print(
        f"Embedding shape: "
        f"{embeddings.shape}"
    )

    print(
        f"Embedding generation time: "
        f"{embedding_time:.4f}s"
    )

    assert embeddings is not None
    assert len(embeddings) == len(chunks)
    assert embeddings.shape[0] == len(chunks)
    assert embeddings.shape[1] > 0

    print("Embedding validation passed.")

    # ===============================================================
    # 7. BUILD FAISS
    # ===============================================================

    print("\n[7] Building FAISS...")
    print("-" * 70)

    vector_store = VectorStore()

    start = time.perf_counter()

    vector_store.build(
        chunks,
        embeddings,
    )

    faiss_build_time = (
        time.perf_counter() - start
    )

    print(
        f"FAISS vectors: "
        f"{len(vector_store)}"
    )

    print(
        f"FAISS build time: "
        f"{faiss_build_time:.4f}s"
    )

    assert vector_store.is_ready
    assert len(vector_store) == len(chunks)

    print("FAISS validation passed.")

    # ===============================================================
    # 8. CREATE HYBRID RETRIEVER
    # ===============================================================

    print("\n[8] Creating Hybrid Retriever...")
    print("-" * 70)

    hybrid = HybridRetriever(
        vector_store=vector_store,
        bm25_store=bm25,
        rrf_k=60,
    )

    print(
        f"Retriever: {hybrid}"
    )

    print(
        f"RRF k: {hybrid.rrf_k}"
    )

    print(
        f"Dense ready: "
        f"{hybrid.has_dense}"
    )

    print(
        f"Sparse ready: "
        f"{hybrid.has_sparse}"
    )

    assert hybrid.is_ready
    assert hybrid.has_dense
    assert hybrid.has_sparse

    print(
        "Hybrid Retriever validation passed."
    )

    # ===============================================================
    # 9. GENERATE QUERY EMBEDDING
    # ===============================================================

    print("\n[9] Generating query embedding...")
    print("-" * 70)

    query = (
        "What skills are required "
        "for an internship?"
    )

    print(
        f"Query: {query}"
    )

    start = time.perf_counter()

    query_embedding = embedding_model.encode(
        query,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    query_embedding_time = (
        time.perf_counter() - start
    )

    print(
        f"Query embedding time: "
        f"{query_embedding_time:.4f}s"
    )

    print(
        f"Embedding dimension: "
        f"{len(query_embedding)}"
    )

    assert query_embedding is not None
    assert len(query_embedding) == embeddings.shape[1]

    print(
        "Query embedding validation passed."
    )

    # ===============================================================
    # 10. HYBRID RETRIEVAL
    # ===============================================================

    print("\n[10] Testing hybrid retrieval...")
    print("-" * 70)

    start = time.perf_counter()

    results = hybrid.retrieve(
        query=query,
        query_embedding=query_embedding,
        top_k=10,
        candidate_pool=5,
        dense_weight=1.0,
        sparse_weight=1.0,
    )

    hybrid_search_time = (
        time.perf_counter() - start
    )

    print(
        f"Hybrid search time: "
        f"{hybrid_search_time:.4f}s"
    )

    print(
        f"Results returned: "
        f"{len(results)}"
    )

    assert len(results) > 0
    assert len(results) <= 10

    print(
        "Hybrid retrieval validation passed."
    )

    # ===============================================================
    # 11. DISPLAY RESULTS
    # ===============================================================

    print("\nTOP HYBRID RESULTS")
    print("-" * 70)

    for rank, result in enumerate(
        results,
        start=1,
    ):

        preview = (
            result.chunk.text
            .replace("\n", " ")
        )

        if len(preview) > 250:
            preview = (
                preview[:250]
                + "..."
            )

        print(
            f"{rank}. "
            f"RRF={result.score:.6f} | "
            f"dense_rank={result.dense_rank} | "
            f"sparse_rank={result.sparse_rank}"
        )

        print(
            f"   dense_score="
            f"{result.dense_score} | "
            f"sparse_score="
            f"{result.sparse_score}"
        )

        print(
            f"   pages="
            f"{result.chunk.page_numbers} | "
            f"chunk="
            f"{result.chunk.chunk_index}"
        )

        print(
            f"   {preview}"
        )

        print()

    # ===============================================================
    # 12. CHECK RRF ORDERING
    # ===============================================================

    print("\n[12] Checking RRF score ordering...")
    print("-" * 70)

    scores = [
        result.score
        for result in results
    ]

    assert scores == sorted(
        scores,
        reverse=True,
    )

    print(
        "RRF results are correctly ordered "
        "from highest to lowest score."
    )

    # ===============================================================
    # 13. CHECK DEDUPLICATION
    # ===============================================================

    print("\n[13] Checking deduplication...")
    print("-" * 70)

    chunk_ids = [
        result.chunk.chunk_id
        for result in results
    ]

    assert len(chunk_ids) == len(
        set(chunk_ids)
    )

    print(
        f"Results: {len(chunk_ids)}"
    )

    print(
        f"Unique chunks: "
        f"{len(set(chunk_ids))}"
    )

    print(
        "Deduplication validation passed."
    )

    # ===============================================================
    # 14. CHECK SOURCE PARTICIPATION
    # ===============================================================

    print(
        "\n[14] Checking retrieval "
        "source participation..."
    )

    print("-" * 70)

    dense_hits = sum(
        result.dense_rank is not None
        for result in results
    )

    sparse_hits = sum(
        result.sparse_rank is not None
        for result in results
    )

    both_hits = sum(
        result.dense_rank is not None
        and result.sparse_rank is not None
        for result in results
    )

    print(
        f"Dense results represented: "
        f"{dense_hits}"
    )

    print(
        f"Sparse results represented: "
        f"{sparse_hits}"
    )

    print(
        f"Results found by both: "
        f"{both_hits}"
    )

    assert dense_hits > 0
    assert sparse_hits > 0

    print(
        "Dense and sparse retrieval "
        "participation validated."
    )

    # ===============================================================
    # 15. CHECK RRF CALCULATION
    # ===============================================================

    print("\n[15] Checking RRF calculation...")
    print("-" * 70)

    for result in results:

        expected_score = 0.0

        if result.dense_rank is not None:

            expected_score += (
                1.0
                / (
                    60
                    + result.dense_rank
                )
            )

        if result.sparse_rank is not None:

            expected_score += (
                1.0
                / (
                    60
                    + result.sparse_rank
                )
            )

        assert abs(
            result.score
            - expected_score
        ) < 1e-12

    print(
        "RRF score calculation validated."
    )

    # ===============================================================
    # 16. BATCH RETRIEVAL
    # ===============================================================

    print("\n[16] Testing batch hybrid retrieval...")
    print("-" * 70)

    batch_queries = [
        "Java",
        "machine learning",
        "internship skills",
    ]

    print(
        f"Queries: {batch_queries}"
    )

    # Generate query embeddings.
    start = time.perf_counter()

    batch_embeddings = (
        embedding_model.encode(
            batch_queries,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    )

    batch_embedding_time = (
        time.perf_counter() - start
    )

    # Convert numpy matrix into a list
    # of individual query embeddings.
    batch_embeddings = list(
        batch_embeddings
    )

    start = time.perf_counter()

    batch_results = (
        hybrid.retrieve_batch(
            queries=batch_queries,
            query_embeddings=batch_embeddings,
            top_k=5,
            candidate_pool=5,
            dense_weight=1.0,
            sparse_weight=1.0,
        )
    )

    batch_search_time = (
        time.perf_counter() - start
    )

    print(
        f"Batch embedding time: "
        f"{batch_embedding_time:.4f}s"
    )

    print(
        f"Batch hybrid search time: "
        f"{batch_search_time:.4f}s"
    )

    assert len(batch_results) == len(
        batch_queries
    )

    for query_text, query_results in zip(
        batch_queries,
        batch_results,
    ):

        print(
            f"Query: {query_text} | "
            f"Results: {len(query_results)}"
        )

        assert len(query_results) > 0
        assert len(query_results) <= 5

        # Check ordering for every query.
        query_scores = [
            result.score
            for result in query_results
        ]

        assert query_scores == sorted(
            query_scores,
            reverse=True,
        )

    print(
        "Batch hybrid retrieval "
        "validation passed."
    )

    # ===============================================================
    # 17. EMPTY QUERY HANDLING
    # ===============================================================

    print("\n[17] Testing empty query handling...")
    print("-" * 70)

    empty_results = hybrid.retrieve(
        query="",
        query_embedding=query_embedding,
        top_k=10,
        candidate_pool=5,
    )

    assert empty_results == []

    print(
        "Empty query handling validated."
    )

    # ===============================================================
    # 18. INVALID CONFIGURATION
    # ===============================================================

    print(
        "\n[18] Testing invalid configuration "
        "handling..."
    )

    print("-" * 70)

    # Invalid top_k
    try:

        hybrid.retrieve(
            query=query,
            query_embedding=query_embedding,
            top_k=0,
            candidate_pool=5,
        )

        raise AssertionError(
            "Expected ValueError for top_k=0."
        )

    except ValueError:
        print(
            "top_k validation passed."
        )

    # Invalid candidate pool
    try:

        hybrid.retrieve(
            query=query,
            query_embedding=query_embedding,
            top_k=5,
            candidate_pool=0,
        )

        raise AssertionError(
            "Expected ValueError for "
            "candidate_pool=0."
        )

    except ValueError:
        print(
            "candidate_pool validation passed."
        )

    # Invalid dense weight
    try:

        hybrid.retrieve(
            query=query,
            query_embedding=query_embedding,
            top_k=5,
            candidate_pool=5,
            dense_weight=-1.0,
        )

        raise AssertionError(
            "Expected ValueError for "
            "negative dense_weight."
        )

    except ValueError:
        print(
            "dense_weight validation passed."
        )

    # Both weights zero
    try:

        hybrid.retrieve(
            query=query,
            query_embedding=query_embedding,
            top_k=5,
            candidate_pool=5,
            dense_weight=0.0,
            sparse_weight=0.0,
        )

        raise AssertionError(
            "Expected ValueError when "
            "both weights are zero."
        )

    except ValueError:
        print(
            "zero-weight validation passed."
        )

    print(
        "Invalid configuration handling passed."
    )

    # ===============================================================
    # SUMMARY
    # ===============================================================

    total_time = (
        time.perf_counter()
        - total_start
    )

    print("\n" + "=" * 70)
    print(
        "HYBRID RETRIEVER TEST COMPLETED"
    )
    print("=" * 70)

    print("\nSUMMARY")
    print("-" * 70)

    print(
        f"Document:             "
        f"{PDF_PATH.name}"
    )

    print(
        f"Pages:                "
        f"{len(document.pages)}"
    )

    print(
        f"Chunks:               "
        f"{len(chunks)}"
    )

    print(
        f"BM25 chunks:          "
        f"{len(bm25)}"
    )

    print(
        f"FAISS vectors:        "
        f"{len(vector_store)}"
    )

    print(
        f"BM25 build:           "
        f"{bm25_build_time:.4f}s"
    )

    print(
        f"Embedding model load: "
        f"{embedding_model_load_time:.4f}s"
    )

    print(
        f"Chunk embedding:      "
        f"{embedding_time:.4f}s"
    )

    print(
        f"FAISS build:          "
        f"{faiss_build_time:.4f}s"
    )

    print(
        f"Query embedding:      "
        f"{query_embedding_time:.4f}s"
    )

    print(
        f"Hybrid search:        "
        f"{hybrid_search_time:.4f}s"
    )

    print(
        f"Batch embedding:      "
        f"{batch_embedding_time:.4f}s"
    )

    print(
        f"Batch hybrid search:  "
        f"{batch_search_time:.4f}s"
    )

    print(
        f"Total test time:      "
        f"{total_time:.4f}s"
    )

    print("\n" + "=" * 70)
    print(
        "ALL HYBRID RETRIEVER TESTS PASSED"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()