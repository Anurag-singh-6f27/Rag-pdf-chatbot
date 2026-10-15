from pathlib import Path
import time

from src.ingestion.pdf_extractor import process_pdf
from src.ingestion.chunker import chunk_document
from src.retrieval.bm25_retriever import BM25Store, tokenize
from config.settings import BM25_INDEX_PATH


PDF_PATH = Path(
    r"C:\Users\anura\OneDrive\Documents\study plan.pdf"
)


def main():

    total_start = time.perf_counter()

    print("=" * 70)
    print("ADVANCED RAG — BM25 RETRIEVER TEST")
    print("=" * 70)

    # ---------------------------------------------------------------
    # 1. Check PDF
    # ---------------------------------------------------------------

    print("\n[1] Checking PDF...")
    print("-" * 70)

    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"PDF not found: {PDF_PATH}"
        )

    print(f"PDF: {PDF_PATH}")
    print(f"Size: {PDF_PATH.stat().st_size:,} bytes")

    # ---------------------------------------------------------------
    # 2. Process PDF
    # ---------------------------------------------------------------

    print("\n[2] Processing PDF...")
    print("-" * 70)

    start = time.perf_counter()

    with PDF_PATH.open("rb") as file:
        file_bytes = file.read()

    document = process_pdf(
        PDF_PATH.name,
        file_bytes,
    )

    extraction_time = time.perf_counter() - start

    character_count = sum(
        len(page.text)
        for page in document.pages
    )

    print(f"Document: {PDF_PATH.name}")
    print(f"Pages: {len(document.pages)}")
    print(f"Characters: {character_count:,}")
    print(f"Extraction time: {extraction_time:.4f}s")

    assert document is not None
    assert len(document.pages) > 0
    assert character_count > 0

    # ---------------------------------------------------------------
    # 3. Chunk document
    # ---------------------------------------------------------------

    print("\n[3] Chunking document...")
    print("-" * 70)

    start = time.perf_counter()

    chunks, duplicate_count = chunk_document(document)

    chunking_time = time.perf_counter() - start

    print(f"Chunks generated: {len(chunks)}")
    print(f"Chunking time: {chunking_time:.4f}s")
    print(f"Duplicates removed: {duplicate_count}")

    assert len(chunks) > 0

    for chunk in chunks:
        assert chunk.text.strip()
        assert chunk.token_count > 0

    print("Chunk validation passed.")

    # ---------------------------------------------------------------
    # 4. Build BM25
    # ---------------------------------------------------------------

    print("\n[4] Building BM25 index...")
    print("-" * 70)

    bm25 = BM25Store()

    start = time.perf_counter()

    bm25.build(chunks)

    build_time = time.perf_counter() - start

    print(f"BM25 chunks: {len(bm25)}")
    print(f"BM25 vocabulary: {bm25.vocabulary_size}")
    print(f"BM25 average document length: {bm25.avgdl:.2f}")
    print(f"BM25 k1: {bm25.k1}")
    print(f"BM25 b: {bm25.b}")
    print(f"Build time: {build_time:.4f}s")

    assert bm25.is_ready
    assert len(bm25) == len(chunks)
    assert bm25.vocabulary_size > 0
    assert bm25.avgdl > 0

    print("BM25 build validation passed.")

    # ---------------------------------------------------------------
    # 5. Save
    # ---------------------------------------------------------------

    print("\n[5] Saving BM25 index...")
    print("-" * 70)

    start = time.perf_counter()

    bm25.save()

    save_time = time.perf_counter() - start

    print(f"Save time: {save_time:.4f}s")
    print(f"Index: {BM25_INDEX_PATH}")

    assert bm25.exists()
    assert BM25_INDEX_PATH.exists()
    assert BM25_INDEX_PATH.stat().st_size > 0

    print("BM25 save validation passed.")

    # ---------------------------------------------------------------
    # 6. Load
    # ---------------------------------------------------------------

    print("\n[6] Loading BM25 index...")
    print("-" * 70)

    loaded_bm25 = BM25Store()

    start = time.perf_counter()

    success = loaded_bm25.load()

    load_time = time.perf_counter() - start

    print(f"Load successful: {success}")
    print(f"Load time: {load_time:.4f}s")

    assert success
    assert loaded_bm25.is_ready
    assert len(loaded_bm25) == len(chunks)
    assert loaded_bm25.vocabulary_size == bm25.vocabulary_size

    print("BM25 load validation passed.")

    # ---------------------------------------------------------------
    # 7. Single-query search
    # ---------------------------------------------------------------

    print("\n[7] Testing BM25 keyword search...")
    print("-" * 70)

    # Get a guaranteed-existing keyword from the indexed chunks.
    test_tokens = tokenize(chunks[0].text)

    assert test_tokens, (
        "First chunk contains no searchable tokens."
    )

    query = test_tokens[0]

    print(f"Query: {query}")

    start = time.perf_counter()

    results = loaded_bm25.search(
        query,
        top_k=5,
    )

    search_time = time.perf_counter() - start

    print(f"Search time: {search_time:.4f}s")
    print(f"Results returned: {len(results)}")

    assert len(results) > 0
    assert len(results) <= 5

    for chunk, score in results:
        assert chunk is not None
        assert isinstance(score, float)
        assert score >= 0

    print("Search validation passed.")

    # ---------------------------------------------------------------
    # 8. Display results
    # ---------------------------------------------------------------

    print("\nTOP BM25 RESULTS")
    print("-" * 70)

    for rank, (chunk, score) in enumerate(
        results,
        start=1,
    ):

        preview = chunk.text.replace("\n", " ")

        if len(preview) > 250:
            preview = preview[:250] + "..."

        print(
            f"{rank}. "
            f"score={score:.4f} | "
            f"pages={chunk.page_numbers} | "
            f"chunk={chunk.chunk_index}"
        )

        print(f"   {preview}")
        print()

    # ---------------------------------------------------------------
    # 9. Batch search
    # ---------------------------------------------------------------

    print("\n[8] Testing batch BM25 search...")
    print("-" * 70)

    # Select guaranteed-existing terms from the document.
    batch_queries = []

    for chunk in chunks:
        tokens = tokenize(chunk.text)

        for token in tokens:
            if token not in batch_queries:
                batch_queries.append(token)

            if len(batch_queries) == 3:
                break

        if len(batch_queries) == 3:
            break

    assert len(batch_queries) == 3

    print(f"Queries: {batch_queries}")

    start = time.perf_counter()

    batch_results = loaded_bm25.search_batch(
        batch_queries,
        top_k=5,
    )

    batch_time = time.perf_counter() - start

    print(f"Batch search time: {batch_time:.4f}s")

    assert len(batch_results) == len(batch_queries)

    for query, query_results in zip(
        batch_queries,
        batch_results,
    ):

        print(f"\nQuery: {query}")
        print(f"Results: {len(query_results)}")

        assert len(query_results) > 0
        assert len(query_results) <= 5

        for chunk, score in query_results:
            assert chunk is not None
            assert isinstance(score, float)
            assert score >= 0

    print("\nBatch search validation passed.")

    # ---------------------------------------------------------------
    # 10. Score ordering
    # ---------------------------------------------------------------

    print("\n[9] Checking BM25 score ordering...")
    print("-" * 70)

    scores = [
        score
        for _, score in results
    ]

    assert scores == sorted(
        scores,
        reverse=True,
    )

    print(
        "All BM25 results are ordered "
        "from highest to lowest score."
    )

    # ---------------------------------------------------------------
    # 11. Keyword behavior test
    # ---------------------------------------------------------------

    print("\n[10] Checking keyword matching behavior...")
    print("-" * 70)

    keyword_query = test_tokens[0]

    keyword_results = loaded_bm25.search(
        keyword_query,
        top_k=5,
    )

    print(f"Query: {keyword_query}")
    print(
        f"Matching results: "
        f"{len(keyword_results)}"
    )

    assert len(keyword_results) > 0

    print(
        "Keyword matching validation passed."
    )

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------

    total_time = time.perf_counter() - total_start

    print("\n" + "=" * 70)
    print("BM25 RETRIEVER TEST COMPLETED")
    print("=" * 70)

    print("\nSUMMARY")
    print("-" * 70)

    print(f"Document:          {PDF_PATH.name}")
    print(f"Pages:             {len(document.pages)}")
    print(f"Chunks:            {len(chunks)}")
    print(f"BM25 chunks:       {len(loaded_bm25)}")
    print(
        f"Vocabulary:        "
        f"{loaded_bm25.vocabulary_size}"
    )
    print(
        f"Average doc length: "
        f"{loaded_bm25.avgdl:.2f}"
    )
    print(f"Build time:        {build_time:.4f}s")
    print(f"Save time:         {save_time:.4f}s")
    print(f"Load time:         {load_time:.4f}s")
    print(f"Single search:     {search_time:.4f}s")
    print(f"Batch search:      {batch_time:.4f}s")
    print(f"Total test time:   {total_time:.4f}s")

    print("\n" + "=" * 70)
    print("ALL BM25 RETRIEVER TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()