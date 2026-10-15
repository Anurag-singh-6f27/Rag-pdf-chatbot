import time

from src.ingestion.pdf_extractor import process_pdf
from src.ingestion.chunker import chunk_document


PDF_PATH = r"C:\Users\anura\OneDrive\Documents\study plan.pdf"


def main():
    print("=" * 70)
    print("ADVANCED RAG — CHUNKER TEST")
    print("=" * 70)

    total_start = time.perf_counter()

    # ---------------------------------------------------------
    # 1. Read PDF
    # ---------------------------------------------------------

    print("\n[1] Reading PDF...")

    read_start = time.perf_counter()

    with open(PDF_PATH, "rb") as file:
        pdf_bytes = file.read()

    read_time = time.perf_counter() - read_start

    print(f"PDF size: {len(pdf_bytes):,} bytes")
    print(f"Read time: {read_time:.4f}s")

    # ---------------------------------------------------------
    # 2. Process PDF
    # ---------------------------------------------------------

    print("\n[2] Processing PDF...")

    document = process_pdf(
        file_name="study plan.pdf",
        file_bytes=pdf_bytes,
    )

    print("\nPDF INFORMATION")
    print("-" * 70)

    print("Name:", document.name)
    print("Pages:", document.num_pages)
    print("Extractable pages:", document.extractable_pages)
    print("Characters:", document.total_characters)
    print("Hash:", document.content_hash[:16] + "...")
    print("Processing time:", f"{document.processing_time:.4f}s")

    # ---------------------------------------------------------
    # 3. Chunk document
    # ---------------------------------------------------------

    print("\n[3] Chunking document...")

    chunk_start = time.perf_counter()

    chunks, duplicate_count = chunk_document(
        document=document,
    )

    chunk_time = time.perf_counter() - chunk_start

    # ---------------------------------------------------------
    # 4. Chunk summary
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("CHUNKING SUMMARY")
    print("=" * 70)

    print("Total chunks:", len(chunks))
    print("Duplicate chunks removed:", duplicate_count)
    print("Chunking time:", f"{chunk_time:.4f}s")

    if not chunks:
        print("\nERROR: No chunks were generated.")
        return

    # ---------------------------------------------------------
    # 5. Token statistics
    # ---------------------------------------------------------

    token_counts = [
        chunk.token_count
        for chunk in chunks
    ]

    print("\nTOKEN STATISTICS")
    print("-" * 70)

    print("Minimum tokens:", min(token_counts))
    print("Maximum tokens:", max(token_counts))
    print(
        "Average tokens:",
        f"{sum(token_counts) / len(token_counts):.2f}",
    )

    print("Total tokens:", sum(token_counts))

    # ---------------------------------------------------------
    # 6. Page statistics
    # ---------------------------------------------------------

    cross_page_chunks = [
        chunk
        for chunk in chunks
        if len(chunk.page_numbers) > 1
    ]

    print("\nPAGE STATISTICS")
    print("-" * 70)

    print(
        "Chunks spanning multiple pages:",
        len(cross_page_chunks),
    )

    # ---------------------------------------------------------
    # 7. Validate chunk limits
    # ---------------------------------------------------------

    print("\nVALIDATION")
    print("-" * 70)

    max_allowed = 220

    oversized_chunks = [
        chunk
        for chunk in chunks
        if chunk.token_count > max_allowed
    ]

    print(
        "Chunks exceeding max token limit:",
        len(oversized_chunks),
    )

    if oversized_chunks:
        print("\nWARNING: Oversized chunks detected:")

        for chunk in oversized_chunks:
            print(
                f"  Chunk {chunk.chunk_index}: "
                f"{chunk.token_count} tokens | "
                f"pages={chunk.page_numbers}"
            )

    else:
        print("✓ All chunks are within the configured limit.")

    # ---------------------------------------------------------
    # 8. Display chunks
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("GENERATED CHUNKS")
    print("=" * 70)

    for chunk in chunks:

        print("\n" + "-" * 70)

        print("Chunk index:", chunk.chunk_index)
        print(
            "Chunk ID:",
            chunk.chunk_id[:16] + "...",
        )
        print("Pages:", chunk.page_numbers)
        print("Tokens:", chunk.token_count)
        print("Characters:", chunk.character_count)

        print("\nText:")
        print(chunk.text)

    # ---------------------------------------------------------
    # 9. Final timing
    # ---------------------------------------------------------

    total_time = time.perf_counter() - total_start

    print("\n" + "=" * 70)
    print("FINAL TIMING")
    print("=" * 70)

    print("PDF reading:", f"{read_time:.4f}s")
    print(
        "PDF processing:",
        f"{document.processing_time:.4f}s",
    )
    print("Chunking:", f"{chunk_time:.4f}s")
    print("Total:", f"{total_time:.4f}s")

    print("\n" + "=" * 70)
    print("TEST COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()