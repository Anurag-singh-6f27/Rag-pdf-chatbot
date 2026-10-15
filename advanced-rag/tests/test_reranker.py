import time

from src.ingestion.pdf_extractor import process_pdf
from src.ingestion.chunker import chunk_document
from src.retrieval.embeddings import get_embedding_model
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_retriever import BM25Store
from src.retrieval.hybrid_retriever import HybridRetriever
from src.reranking.reranker import FlashRankReranker


PDF_PATH =r"C:\Users\anura\OneDrive\Documents\study plan.pdf"



def main():
    total_start = time.perf_counter()

    print("=" * 70)
    print("FLASHRANK RERANKER INTEGRATION TEST")
    print("=" * 70)

    # ==============================================================
    # 1. LOAD PDF
    # ==============================================================

    start = time.perf_counter()

    with open(PDF_PATH, "rb") as file:
        file_bytes = file.read()

    document = process_pdf(
        file_name="study plan.pdf",
        file_bytes=file_bytes,
    )

    print(
        f"\nPDF loaded | "
        f"pages={len(document.pages)} | "
        f"time={time.perf_counter() - start:.4f}s"
    )

    # ==============================================================
    # 2. CHUNK DOCUMENT
    # ==============================================================

    chunks, _ = chunk_document(document)

    print(
        f"Chunks created | "
        f"chunks={len(chunks)}"
    )

    assert len(chunks) > 0

    # ==============================================================
    # 3. BUILD BM25
    # ==============================================================

    start = time.perf_counter()

    bm25 = BM25Store()
    bm25.build(chunks)

    bm25_time = time.perf_counter() - start

    print(
        f"BM25 built | "
        f"documents={len(chunks)} | "
        f"vocabulary={len(bm25._idf)} | "
        f"time={bm25_time:.4f}s"
    )

    # ==============================================================
    # 4. LOAD EMBEDDING MODEL
    # ==============================================================

    start = time.perf_counter()

    model = get_embedding_model()

    model_load_time = time.perf_counter() - start

    print(
        f"Embedding model ready | "
        f"time={model_load_time:.4f}s"
    )

    # ==============================================================
    # 5. GENERATE CHUNK EMBEDDINGS
    # ==============================================================

    start = time.perf_counter()

    texts = [chunk.text for chunk in chunks]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    embedding_time = time.perf_counter() - start

    print(
        f"Chunk embeddings generated | "
        f"shape={embeddings.shape} | "
        f"time={embedding_time:.4f}s"
    )

    # ==============================================================
    # 6. BUILD FAISS
    # ==============================================================

    start = time.perf_counter()

    vector_store = VectorStore()

    vector_store.build(
        chunks,
        embeddings,
    )

    faiss_time = time.perf_counter() - start

    print(
        f"FAISS built | "
        f"vectors={len(vector_store)} | "
        f"dimension={vector_store.dimension} | "
        f"time={faiss_time:.4f}s"
    )

    # ==============================================================
    # 7. CREATE HYBRID RETRIEVER
    # ==============================================================

    hybrid = HybridRetriever(
        vector_store=vector_store,
        bm25_store=bm25,
        rrf_k=60,
    )

    print(
        f"Hybrid retriever ready | "
        f"dense={hybrid.has_dense} | "
        f"sparse={hybrid.has_sparse}"
    )

    assert hybrid.is_ready

    # ==============================================================
    # 8. QUERY
    # ==============================================================

    query = "What skills are required for an internship?"

    print()
    print("-" * 70)
    print(f"QUERY: {query}")
    print("-" * 70)

    # ==============================================================
    # 9. QUERY EMBEDDING
    # ==============================================================

    start = time.perf_counter()

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )[0]

    query_embedding_time = time.perf_counter() - start

    print(
        f"Query embedding generated | "
        f"time={query_embedding_time:.4f}s"
    )

    # ==============================================================
    # 10. HYBRID RETRIEVAL
    # ==============================================================

    start = time.perf_counter()

    hybrid_results = hybrid.retrieve(
        query=query,
        query_embedding=query_embedding,
        top_k=10,
        candidate_pool=5,
    )

    hybrid_time = time.perf_counter() - start

    print(
        f"Hybrid retrieval complete | "
        f"candidates={len(hybrid_results)} | "
        f"time={hybrid_time:.4f}s"
    )

    assert len(hybrid_results) > 0
    assert len(hybrid_results) <= 10

    # ==============================================================
    # 11. DISPLAY HYBRID RESULTS
    # ==============================================================

    print()
    print("-" * 70)
    print("BEFORE FLASHRANK — HYBRID RESULTS")
    print("-" * 70)

    for result in hybrid_results:

        preview = result.chunk.text.replace(
            "\n",
            " ",
        )

        if len(preview) > 150:
            preview = preview[:150] + "..."

        print(
            f"\nHybrid Rank: "
            f"{result.dense_rank or result.sparse_rank}"
        )

        print(
            f"Chunk ID: {result.chunk.chunk_id}"
        )

        print(
            f"RRF Score: {result.score:.6f}"
        )

        print(
            f"Dense Rank: {result.dense_rank}"
        )

        print(
            f"Sparse Rank: {result.sparse_rank}"
        )

        print(
            f"Pages: {result.chunk.page_numbers}"
        )

        print(
            f"Text: {preview}"
        )

    # ==============================================================
    # 12. CREATE FLASHRANK RERANKER
    # ==============================================================

    reranker = FlashRankReranker(
        model_name="ms-marco-TinyBERT-L-2-v2",
        max_length=512,
    )

    print()
    print(
        f"Reranker created | "
        f"ready={reranker.is_ready}"
    )

    # Model should NOT be loaded yet because of lazy loading.
    assert reranker.is_ready is False

    # ==============================================================
    # 13. FLASHRANK RERANKING
    # ==============================================================

    start = time.perf_counter()

    response = reranker.rerank(
        query=query,
        candidates=hybrid_results,
        top_k=5,
    )

    total_rerank_time = time.perf_counter() - start

    # ==============================================================
    # 14. VALIDATE RESPONSE
    # ==============================================================

    print()
    print("-" * 70)
    print("FLASHRANK RESULTS")
    print("-" * 70)

    print(
        f"Model: {response.model_name}"
    )

    print(
        f"Candidates: {response.num_candidates}"
    )

    print(
        f"Returned: {len(response.results)}"
    )

    print(
        f"Reranking latency: "
        f"{response.latency_seconds:.4f}s"
    )

    print(
        f"Measured total rerank call: "
        f"{total_rerank_time:.4f}s"
    )

    # Basic validation
    assert response.num_candidates == len(
        hybrid_results
    )

    assert len(response.results) == min(
        5,
        len(hybrid_results),
    )

    assert reranker.is_ready is True

    # ==============================================================
    # 15. VALIDATE RANK ORDER
    # ==============================================================

    ranks = [
        result.rerank_rank
        for result in response.results
    ]

    expected_ranks = list(
        range(
            1,
            len(response.results) + 1,
        )
    )

    assert ranks == expected_ranks

    # ==============================================================
    # 16. VALIDATE SCORE ORDER
    # ==============================================================

    scores = [
        result.rerank_score
        for result in response.results
    ]

    assert scores == sorted(
        scores,
        reverse=True,
    )

    # ==============================================================
    # 17. VALIDATE NO DUPLICATES
    # ==============================================================

    chunk_ids = [
        result.chunk_id
        for result in response.results
    ]

    assert len(chunk_ids) == len(
        set(chunk_ids)
    )

    # ==============================================================
    # 18. VALIDATE METADATA PRESERVATION
    # ==============================================================

    for result in response.results:

        assert result.chunk_id
        assert result.document_name
        assert result.page_numbers
        assert result.text

        assert result.original_rank is not None
        assert result.original_score is not None

    # ==============================================================
    # 19. PRINT RERANKED RESULTS
    # ==============================================================

    for result in response.results:

        preview = result.text.replace(
            "\n",
            " ",
        )

        if len(preview) > 180:
            preview = preview[:180] + "..."

        print()
        print(
            f"Rerank Rank: {result.rerank_rank}"
        )

        print(
            f"Rerank Score: "
            f"{result.rerank_score:.6f}"
        )

        print(
            f"Original Rank: "
            f"{result.original_rank}"
        )

        print(
            f"Original RRF: "
            f"{result.original_score:.6f}"
        )

        print(
            f"Chunk ID: "
            f"{result.chunk_id}"
        )

        print(
            f"Document: "
            f"{result.document_name}"
        )

        print(
            f"Pages: "
            f"{result.page_numbers}"
        )

        print(
            f"Text: {preview}"
        )

    # ==============================================================
    # 20. TEST EMPTY QUERY
    # ==============================================================

    try:

        reranker.rerank(
            query="",
            candidates=hybrid_results,
            top_k=5,
        )

        raise AssertionError(
            "Empty query should have raised ValueError."
        )

    except ValueError:

        print(
            "\nEmpty query validation: PASSED"
        )

    # ==============================================================
    # 21. TEST INVALID TOP-K
    # ==============================================================

    try:

        reranker.rerank(
            query=query,
            candidates=hybrid_results,
            top_k=0,
        )

        raise AssertionError(
            "top_k=0 should have raised ValueError."
        )

    except ValueError:

        print(
            "Invalid top_k validation: PASSED"
        )

    # ==============================================================
    # 22. SECOND CALL — TEST MODEL REUSE
    # ==============================================================

    start = time.perf_counter()

    response_2 = reranker.rerank(
        query=query,
        candidates=hybrid_results,
        top_k=5,
    )

    second_call_time = time.perf_counter() - start

    assert reranker.is_ready is True

    assert len(response_2.results) == len(
        response.results
    )

    print(
        f"Second reranking call: "
        f"{second_call_time:.4f}s"
    )

    # ==============================================================
    # 23. FINAL SUMMARY
    # ==============================================================

    total_time = (
        time.perf_counter()
        - total_start
    )

    print()
    print("=" * 70)
    print("FLASHRANK TEST SUMMARY")
    print("=" * 70)

    print(
        f"PDF pages:              {len(document.pages)}"
    )

    print(
        f"Chunks:                 {len(chunks)}"
    )

    print(
        f"Hybrid candidates:      {len(hybrid_results)}"
    )

    print(
        f"FlashRank results:      {len(response.results)}"
    )

    print(
        f"FlashRank latency:      "
        f"{response.latency_seconds:.4f}s"
    )

    print(
        f"Second call latency:    "
        f"{second_call_time:.4f}s"
    )

    print(
        f"Total test time:        "
        f"{total_time:.4f}s"
    )

    print()
    print("Validation:")
    print("  Candidate count       PASSED")
    print("  Top-K selection       PASSED")
    print("  Rank ordering         PASSED")
    print("  Score ordering        PASSED")
    print("  Duplicate detection   PASSED")
    print("  Metadata preservation PASSED")
    print("  Empty query handling  PASSED")
    print("  Invalid top-K         PASSED")
    print("  Model reuse           PASSED")

    print()
    print("=" * 70)
    print("ALL FLASHRANK TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()