import hashlib
import time
from dataclasses import dataclass, field

import pymupdf
from ..utils.logger import logger
from .text_cleaner import clean_text


class PDFProcessingError(Exception):
    """Raised when a PDF cannot be processed."""


@dataclass(slots=True)
class PageText:
    """Cleaned text extracted from one PDF page."""
    page_number: int
    text: str
    character_count: int = 0

@dataclass(slots=True)
class ProcessedDocument:
    """Result of PDF ingestion."""
    name: str
    content_hash: str
    num_pages: int
    pages: list[PageText] = field(default_factory=list)
    extractable_pages: int = 0
    total_characters: int = 0
    processing_time: float = 0.0

    @property
    def has_text(self) -> bool:
        return self.extractable_pages > 0

def file_hash(file_bytes: bytes) -> str:
    """Return SHA-256 hash of the PDF bytes."""
    return hashlib.sha256(file_bytes).hexdigest()

def process_pdf(
    file_name: str,
    file_bytes: bytes,
) -> ProcessedDocument:
    """
    Extract and clean PDF text page-by-page.
    Preserves page numbers and collects basic ingestion statistics.
    """

    start_time = time.perf_counter()

    if not file_bytes:
        raise PDFProcessingError(
            f"'{file_name}' appears to be empty."
        )

    hash_start = time.perf_counter()
    content_hash = file_hash(file_bytes)
    hash_time = time.perf_counter() - hash_start

    try:
        with pymupdf.open(
            stream=file_bytes,
            filetype="pdf",
        ) as document:
            num_pages = len(document)
            if num_pages == 0:
                raise PDFProcessingError(
                    f"'{file_name}' contains no pages."
                )
            pages: list[PageText] = []
            extractable_pages = 0
            total_characters = 0
            extraction_start = time.perf_counter()

            for page_number, page in enumerate(
                document,
                start=1,
            ):
                try:
                    raw_text = page.get_text("text")

                    cleaned_text = clean_text(raw_text)

                except Exception as exc:
                    logger.warning(
                        "Failed to extract page %d from '%s': %s",
                        page_number,
                        file_name,
                        exc,
                    )
                    cleaned_text = ""

                character_count = len(cleaned_text)
                
                if character_count > 0:
                    extractable_pages += 1

                total_characters += character_count

                pages.append(
                    PageText(
                        page_number=page_number,
                        text=cleaned_text,
                        character_count=character_count,
                    )
                )
            extraction_time = time.perf_counter() - extraction_start
    except PDFProcessingError:
        raise

    except Exception as exc:
        raise PDFProcessingError(
            f"Could not open '{file_name}' as a PDF."
        ) from exc
    
    processing_time = time.perf_counter() - start_time

    processed_document = ProcessedDocument(
        name=file_name,
        content_hash=content_hash,
        num_pages=num_pages,
        pages=pages,
        extractable_pages=extractable_pages,
        total_characters=total_characters,
        processing_time=processing_time,
    )
    if not processed_document.has_text:
        raise PDFProcessingError(
            f"'{file_name}' contains no extractable text. "
            "It may be a scanned or image-only PDF."
        )

    logger.info(
        "Processed '%s': %d pages, %d characters, %.3f seconds",
        file_name,
        num_pages,
        total_characters,
        processing_time,
    )
    logger.info(
        "Processed '%s' | pages=%d | characters=%d | "
        "hash=%.4fs | extraction=%.4fs | total=%.4fs",
        file_name,
        num_pages,
        total_characters,
        hash_time,
        extraction_time,
        processing_time,
    )

    return processed_document