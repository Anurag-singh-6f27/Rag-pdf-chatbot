"""
Sentence-aware, token-based document chunker for the Advanced RAG system.

Features:
- Token-based chunk sizing
- Sentence-boundary preservation (abbreviation- and decimal-aware)
- Sentence-level overlap
- Cross-page chunks and cross-page tokenizer batching
- Page metadata preservation
- Deterministic chunk IDs
- Same-document duplicate removal
- No silent content loss: undersized chunks are merged rather than dropped,
  and merging never pushes a chunk past the configured max_tokens ceiling
- Oversized single sentences are split on word boundaries (not raw token
  positions) so pieces stay readable, with clause-boundary preference
- Detailed performance logging
"""

import hashlib
import re
import threading
import time
from dataclasses import dataclass

from transformers import AutoTokenizer

from ..utils.logger import logger
from .pdf_extractor import ProcessedDocument

from config.settings import (
    TOKENIZER_NAME,
    MAX_CHUNK_TOKENS,
    CHUNK_OVERLAP_TOKENS,
    MIN_CHUNK_TOKENS,
)


SENTENCE_TOKENIZE_BATCH_SIZE = 256




@dataclass(slots=True)
class Chunk:
    """A retrieval chunk with document and page metadata."""

    chunk_id: str
    document_name: str
    document_hash: str
    page_numbers: list[int]
    chunk_index: int
    text: str
    token_count: int
    character_count: int


@dataclass(slots=True)
class SentenceUnit:
    """A logical text unit with source page and token count."""

    text: str
    page_number: int
    token_count: int
    unit_type: str = "text"


_tokenizer = None
_tokenizer_lock = threading.Lock()


#_context_free_join = None


def get_tokenizer():
    """
    Load and cache the tokenizer.

    The tokenizer is loaded only once during the lifetime of the
    process. Double-checked locking avoids loading it twice if two
    threads race on first use.
    """

    global _tokenizer

    if _tokenizer is None:
        with _tokenizer_lock:
            if _tokenizer is None:
                start_time = time.perf_counter()

                logger.info("Loading tokenizer: %s", TOKENIZER_NAME)

                _tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

                elapsed = time.perf_counter() - start_time
                logger.info("Tokenizer loaded in %.4fs", elapsed)

    return _tokenizer


def count_tokens(text: str) -> int:
    """
    Return the tokenizer's exact token count for a single text.
    """
    tokenizer = get_tokenizer()
    return len(tokenizer.encode(text, add_special_tokens=False))


def count_tokens_batch(texts: list[str]) -> list[int]:
    """
    Return token counts for many texts in a single tokenizer call.
    """

    if not texts:
        return []

    tokenizer = get_tokenizer()

    encoded = tokenizer(
        texts,
        add_special_tokens=False,
    )

    return [len(ids) for ids in encoded["input_ids"]]


# def _tokenizer_supports_context_free_join() -> bool:
#     """
#     Determine (once, and cache) whether this tokenizer's token count for
#     "a b" equals count(a) + count(b) for whitespace-separated text.
#     """

#     global _context_free_join

#     if _context_free_join is None:
#         probe_a = "testing edge-case boundary check,"
#         probe_b = "here is another sample fragment!"

#         joined_count = count_tokens(f"{probe_a} {probe_b}")
#         summed_count = count_tokens(probe_a) + count_tokens(probe_b)

#         _context_free_join = joined_count == summed_count

#         logger.info(
#             "Tokenizer whitespace-join is context-free: %s "
#             "(chunk token counts will use the %s path)",
#             _context_free_join,
#             "additive fast" if _context_free_join else "exact re-tokenization",
#         )

#     return _context_free_join


# ---------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------
# ---------------------------------------------------------
# Structural / sentence splitting
# ---------------------------------------------------------

_PROTECTED_ABBREVIATIONS = [
    r"e\.g\.",
    r"i\.e\.",
    r"etc\.",
    r"et al\.",
    r"cf\.",
    r"viz\.",
    r"vol\.",
    r"ed\.",
    r"eds\.",
    r"Fig\.",
    r"Figs\.",
    r"Eq\.",
    r"Eqs\.",
    r"Sec\.",
    r"Secs\.",
    r"No\.",
    r"vs\.",
    r"approx\.",
    r"Dr\.",
    r"Mr\.",
    r"Mrs\.",
    r"Ms\.",
    r"Prof\.",
    r"Rev\.",
    r"Jr\.",
    r"Sr\.",
    r"St\.",
    r"Ave\.",
    r"Corp\.",
    r"Inc\.",
    r"Ltd\.",
    r"Co\.",
    r"U\.S\.",
    r"U\.K\.",
    r"U\.N\.",
    r"Ph\.D\.",
    r"M\.D\.",
    r"B\.A\.",
    r"M\.A\.",
    r"B\.S\.",
    r"M\.S\.",
    r"a\.m\.",
    r"p\.m\.",
]

_ABBREVIATION_PATTERN = re.compile(
    r"(?<!\w)(?:" + "|".join(_PROTECTED_ABBREVIATIONS) + r")",
    flags=re.IGNORECASE,
)

_PLACEHOLDER_PREFIX = "\x00P"
_PLACEHOLDER_SUFFIX = "\x00"
_DECIMAL_PLACEHOLDER = "\x00DOT\x00"
_INITIAL_PLACEHOLDER = "\x00INIT\x00"


# Sentence boundary:
#   "This is a sentence. Next sentence"
#
# We deliberately keep this simple because structural splitting
# is handled separately below.
_SENTENCE_BOUNDARY_PATTERN = re.compile(
    r'(?<=[.!?])[ \t]+(?=[A-Z(\["\'])'
)


# Common structural markers found in PDFs.
_BULLET_PATTERN = re.compile(
    r"^\s*(?:[•●▪◦‣⁃*-]|\d+[.)]|[a-zA-Z][.)])\s+"
)

_HEADING_PATTERN = re.compile(
    r"^\s*(?:"
    r"PHASE\s+\d+"
    r"|"
    r"(?:[A-Z][A-Za-z&/\- ]{2,50})\s+Roadmap"
    r"|"
    r"Final\s+Summary"
    r"|"
    r"Monthly\s+Focus"
    r"|"
    r"Project\s+Ideas"
    r"|"
    r"Resources"
    r")\b",
    flags=re.IGNORECASE,
)


def _protect_special_patterns(text: str) -> tuple[str, list[str]]:
    """
    Protect abbreviations, decimals and initials before sentence splitting.
    """

    protected_spans: list[str] = []

    def _protect(match: re.Match) -> str:
        protected_spans.append(match.group(0))
        return (
            f"{_PLACEHOLDER_PREFIX}"
            f"{len(protected_spans) - 1}"
            f"{_PLACEHOLDER_SUFFIX}"
        )

    text = _ABBREVIATION_PATTERN.sub(_protect, text)

    # Protect decimal points: 3.14
    text = re.sub(
        r"(?<=\d)\.(?=\d)",
        _DECIMAL_PLACEHOLDER,
        text,
    )

    # Protect initials: A. Smith
    text = re.sub(
        r"\b([A-Z])\.\s+(?=[A-Z][a-z])",
        rf"\1{_INITIAL_PLACEHOLDER} ",
        text,
    )

    return text, protected_spans


def _restore_special_patterns(
    text: str,
    protected_spans: list[str],
) -> str:
    """Restore protected patterns after splitting."""

    for index, original in enumerate(protected_spans):
        text = text.replace(
            f"{_PLACEHOLDER_PREFIX}{index}{_PLACEHOLDER_SUFFIX}",
            original,
        )

    text = text.replace(_DECIMAL_PLACEHOLDER, ".")
    text = text.replace(_INITIAL_PLACEHOLDER, ".")

    return text.strip()


def _split_line_into_sentences(line: str) -> list[str]:
    """
    Split one structural line into sentences where appropriate.

    Keeping this separate from line/structure detection prevents PDF
    formatting from being destroyed before chunking.
    """

    line = line.strip()

    if not line:
        return []

    protected_line, protected_spans = _protect_special_patterns(line)

    pieces = _SENTENCE_BOUNDARY_PATTERN.split(protected_line)

    result = []

    for piece in pieces:
        piece = _restore_special_patterns(
            piece.strip(),
            protected_spans,
        )

        if piece:
            result.append(piece)

    return result


def split_text_units(text: str) -> list[tuple[str, str]]:
    """
    Convert PDF text into logical structural units.

    Returns:
        [(text, unit_type), ...]

    unit_type can be:
        heading
        bullet
        numbered
        text
        line
    """

    if not text or not text.strip():
        return []

    # Normalize line endings but IMPORTANTLY preserve line structure.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove excessive blank lines, but don't flatten normal lines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    raw_lines = text.split("\n")

    units: list[tuple[str, str]] = []

    for raw_line in raw_lines:

        line = raw_line.strip()

        if not line:
            continue

        # Normalize repeated spaces inside the line.
        line = re.sub(r"[ \t]+", " ", line)

        # ---------------------------------------------
        # Bullet / numbered items
        # ---------------------------------------------
        if _BULLET_PATTERN.match(line):

            if re.match(r"^\s*\d+[.)]\s+", line):
                unit_type = "numbered"
            else:
                unit_type = "bullet"

            units.append((line, unit_type))
            continue

        # ---------------------------------------------
        # Heading
        # ---------------------------------------------
        if _HEADING_PATTERN.match(line):
            units.append((line, "heading"))
            continue

        # ---------------------------------------------
        # Normal line
        # ---------------------------------------------
        sentences = _split_line_into_sentences(line)

        for sentence in sentences:
            units.append((sentence, "text"))

    return units

# def split_sentences(text: str) -> list[str]:
#     """
#     Split academic/professional text into sentences while protecting
#     abbreviations, decimal numbers, and initials from being mistaken for
#     sentence boundaries.
#     """

#     text = text.strip()

#     if not text:
#         return []

#     text = re.sub(r"[ \t]+", " ", text)
#     text = re.sub(r"\n{2,}", "\n\n", text)
#     text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

#     protected_spans: list[str] = []

#     def _protect(match: re.Match) -> str:
#         protected_spans.append(match.group(0))
#         return f"{_PLACEHOLDER_PREFIX}{len(protected_spans) - 1}{_PLACEHOLDER_SUFFIX}"

#     text = _ABBREVIATION_PATTERN.sub(_protect, text)

    
#     text = re.sub(r"(?<=\d)\.(?=\d)", _DECIMAL_PLACEHOLDER, text)

   
#     text = re.sub(
#         r"\b([A-Z])\.\s+(?=[A-Z][a-z])",
#         rf"\1{_INITIAL_PLACEHOLDER} ",
#         text,
#     )

#     sentences = _SENTENCE_SPLIT_PATTERN.split(text)

#     result = []

#     for sentence in sentences:

        
#         for index, original in enumerate(protected_spans):
#             sentence = sentence.replace(
#                 f"{_PLACEHOLDER_PREFIX}{index}{_PLACEHOLDER_SUFFIX}",
#                 original,
#             )

#         sentence = sentence.replace(_DECIMAL_PLACEHOLDER, ".")
#         sentence = sentence.replace(_INITIAL_PLACEHOLDER, ".")

#         sentence = sentence.strip()

#         if sentence:
#             result.append(sentence)

#     return result




def build_sentence_units(
    document: ProcessedDocument,
    tokenize_batch_size: int = SENTENCE_TOKENIZE_BATCH_SIZE,
) -> list[SentenceUnit]:
    """
    Convert PDF pages into structural text units.

    PDF line structure is preserved so headings, bullets, numbered
    items and table-like lines are not accidentally merged into one
    giant sentence.
    """

    start_time = time.perf_counter()

    sentence_units: list[SentenceUnit] = []

    pending_texts: list[str] = []
    pending_pages: list[int] = []
    pending_types: list[str] = []

    def flush_pending() -> None:

        if not pending_texts:
            return

        token_counts = count_tokens_batch(pending_texts)

        for text, page, token_count, unit_type in zip(
            pending_texts,
            pending_pages,
            token_counts,
            pending_types,
        ):
            sentence_units.append(
                SentenceUnit(
                    text=text,
                    page_number=page,
                    token_count=token_count,
                    unit_type=unit_type,
                )
            )

        pending_texts.clear()
        pending_pages.clear()
        pending_types.clear()

    for page in document.pages:

        if not page.text:
            logger.debug(
                "Skipping empty page %d",
                page.page_number,
            )
            continue

        units = split_text_units(page.text)

        logger.info(
            "Page %d: %d text units detected",
            page.page_number,
            len(units),
        )

        for text, unit_type in units:

            pending_texts.append(text)
            pending_pages.append(page.page_number)
            pending_types.append(unit_type)

            if len(pending_texts) >= tokenize_batch_size:
                flush_pending()

    flush_pending()

    elapsed = time.perf_counter() - start_time

    logger.info(
        "Text-unit preparation completed | units=%d | time=%.4fs",
        len(sentence_units),
        elapsed,
    )

    return sentence_units


# ---------------------------------------------------------
# Oversized-sentence handling
# ---------------------------------------------------------

_SOFT_BREAK_SUFFIXES = (",", ";", ":", ")", "]", "\u2014")  # \u2014 = em dash

_WORD_PATTERN = re.compile(r"\S+")


def _hard_split_text_by_tokens(
    text: str,
    max_tokens: int,
    page_number: int,
) -> list[SentenceUnit]:
    """
    Last-resort raw token-level slicing for a single run of text that
    has no whitespace to break on and alone exceeds max_tokens (e.g. a
    very long URL, DOI, or hash produced by messy PDF extraction).

    Used only as a fallback inside split_oversized_sentence -- normal
    long sentences are split on word boundaries instead, which reads
    far better than an arbitrary token cut.
    """

    tokenizer = get_tokenizer()

    token_ids = tokenizer.encode(text, add_special_tokens=False)

    pieces: list[SentenceUnit] = []

    for start in range(0, len(token_ids), max_tokens):
        window = token_ids[start:start + max_tokens]

        piece_text = tokenizer.decode(window).strip()

        if piece_text:
            pieces.append(
                SentenceUnit(
                    text=piece_text,
                    page_number=page_number,
                    token_count=len(window),
                )
            )

    return pieces


def split_oversized_sentence(
    sentence: SentenceUnit,
    max_tokens: int,
) -> list[SentenceUnit]:
    """
    Split a single sentence that exceeds max_tokens on its own into
    pieces that each fit within max_tokens.

    Falls back to raw token slicing only for the pathological case of a
    single "word" (e.g. a long URL/DOI) that alone exceeds max_tokens.
    """

    words = _WORD_PATTERN.findall(sentence.text)

    if not words:
        return []

    word_token_counts = count_tokens_batch(words)

    pieces: list[SentenceUnit] = []

    
    current: list[tuple[str, int]] = []
    current_tokens = 0
    last_soft_break = -1  

    def flush(chunk_words: list[tuple[str, int]]) -> None:
        if not chunk_words:
            return

        piece_text = " ".join(w for w, _ in chunk_words)

        pieces.append(
            SentenceUnit(
                text=piece_text,
                page_number=sentence.page_number,
                token_count=count_tokens(piece_text),
            )
        )

    for word, word_tokens in zip(words, word_token_counts):

        if word_tokens > max_tokens:
            flush(current)
            current, current_tokens, last_soft_break = [], 0, -1

            pieces.extend(
                _hard_split_text_by_tokens(word, max_tokens, sentence.page_number)
            )
            continue

        if current and current_tokens + word_tokens > max_tokens:

            half_point = len(current) // 2

            if last_soft_break >= half_point:
                emit = current[: last_soft_break + 1]
                carry = current[last_soft_break + 1:]
            else:
                emit = current
                carry = []

            flush(emit)

            current = carry
            current_tokens = sum(tok for _, tok in current)
            last_soft_break = -1

            for idx, (carried_word, _) in enumerate(current):
                if carried_word.rstrip().endswith(_SOFT_BREAK_SUFFIXES):
                    last_soft_break = idx

        current.append((word, word_tokens))
        current_tokens += word_tokens

        if word.rstrip().endswith(_SOFT_BREAK_SUFFIXES):
            last_soft_break = len(current) - 1

    flush(current)

    return pieces


def expand_oversized_sentences(
    sentence_units: list[SentenceUnit],
    max_tokens: int,
) -> list[SentenceUnit]:
    """
    Replace any sentence that exceeds max_tokens with split pieces that
    each fit within max_tokens. Sentences within the limit pass through
    unchanged.
    """

    expanded: list[SentenceUnit] = []

    for sentence in sentence_units:
        sentence_tokens = sentence.token_count

        
        if sentence.token_count > max_tokens:

            pieces = split_oversized_sentence(sentence, max_tokens)

            logger.warning(
                "Sentence exceeded chunk limit and was split on word "
                "boundaries | page=%d | original_tokens=%d | limit=%d | "
                "pieces=%d",
                sentence.page_number,
                sentence.token_count,
                max_tokens,
                len(pieces),
            )

            expanded.extend(pieces)

        else:
            expanded.append(sentence)

    return expanded


# ---------------------------------------------------------
# Chunk ID
# ---------------------------------------------------------

def create_chunk_id(
    document_hash: str,
    page_numbers: list[int],
    text: str,
) -> str:
    """
    Create a deterministic chunk ID.
    Same document + same pages + same text always produces the same ID.
    """

    content = (
        f"{document_hash}|"
        f"{','.join(map(str, page_numbers))}|"
        f"{text}"
    )

    return hashlib.sha256(content.encode("utf-8")).hexdigest()


# ---------------------------------------------------------
# Chunk creation
# ---------------------------------------------------------

def create_chunk(
    document: ProcessedDocument,
    sentences: list[SentenceUnit],
    chunk_index: int,
) -> Chunk:

    text = " ".join(sentence.text for sentence in sentences).strip()

    page_numbers = sorted({sentence.page_number for sentence in sentences})

    token_count = count_tokens(text)

    chunk_id = create_chunk_id(
        document_hash=document.content_hash,
        page_numbers=page_numbers,
        text=text,
    )

    return Chunk(
        chunk_id=chunk_id,
        document_name=document.name,
        document_hash=document.content_hash,
        page_numbers=page_numbers,
        chunk_index=chunk_index,
        text=text,
        token_count=token_count,
        character_count=len(text),
    )


def _merge_into_chunk(
    document: ProcessedDocument,
    base: Chunk,
    extra_sentences: list[SentenceUnit],
) -> Chunk:
    """
    Merge extra sentences into an already-created chunk.
    """

    extra_text = " ".join(sentence.text for sentence in extra_sentences).strip()

    combined_text = f"{base.text} {extra_text}".strip()

    page_numbers = sorted(
        set(base.page_numbers) | {sentence.page_number for sentence in extra_sentences}
    )
    token_count = count_tokens(combined_text)

    chunk_id = create_chunk_id(
        document_hash=document.content_hash,
        page_numbers=page_numbers,
        text=combined_text,
    )

    return Chunk(
        chunk_id=chunk_id,
        document_name=base.document_name,
        document_hash=base.document_hash,
        page_numbers=page_numbers,
        chunk_index=base.chunk_index,
        text=combined_text,
        token_count=token_count,
        character_count=len(combined_text),
    )


def _build_overlap(
    sentences: list[SentenceUnit],
    overlap_tokens: int,
) -> tuple[list[SentenceUnit], int]:
    """
    Select trailing sentences from `sentences` to carry into the next
    chunk as overlap, staying within overlap_tokens.
    """

    overlap_sentences: list[SentenceUnit] = []
    overlap_count = 0

    for previous in reversed(sentences):

        if overlap_count + previous.token_count > overlap_tokens:
            break

        overlap_sentences.insert(0, previous)
        overlap_count += previous.token_count

    if not overlap_sentences and overlap_tokens > 0:
        logger.debug(
            "No overlap carried forward: the most recent sentence "
            "alone (%d tokens) exceeds overlap_tokens (%d)",
            sentences[-1].token_count if sentences else 0,
            overlap_tokens,
        )

    return overlap_sentences, overlap_count


# ---------------------------------------------------------
# Main chunking function
# ---------------------------------------------------------

def chunk_document(
    document: ProcessedDocument,
    max_tokens: int = MAX_CHUNK_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
    min_tokens: int = MIN_CHUNK_TOKENS,
) -> tuple[list[Chunk], int]:

    start_time = time.perf_counter()

    logger.info(
        "Starting chunking | document='%s' | "
        "max_tokens=%d | overlap_tokens=%d | min_tokens=%d",
        document.name,
        max_tokens,
        overlap_tokens,
        min_tokens,
    )

    if max_tokens <= 0:
        raise ValueError("max_tokens must be greater than 0.")

    if overlap_tokens < 0:
        raise ValueError("overlap_tokens cannot be negative.")

    if overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be smaller than max_tokens.")

    if min_tokens < 0:
        raise ValueError("min_tokens cannot be negative.")

    if min_tokens >= max_tokens:
        raise ValueError("min_tokens must be smaller than max_tokens.")

    sentence_units = build_sentence_units(document)

    if not sentence_units:
        logger.warning("No sentences found in '%s'", document.name)
        return [], 0

    sentence_units = expand_oversized_sentences(sentence_units, max_tokens)

    chunks: list[Chunk] = []

    current_sentences: list[SentenceUnit] = []
    current_tokens = 0

    chunk_index = 0
    merged_small_chunk_count = 0

    for sentence in sentence_units:

        sentence_tokens = sentence.token_count

        # Check whether adding this sentence would exceed the limit.
        if (
            current_sentences
            and current_tokens + sentence_tokens > max_tokens
        ):
            chunk = create_chunk(
                document=document,
                sentences=current_sentences,
                chunk_index=chunk_index,
            )

            # -------------------------------------------------
            # Normal chunk
            # -------------------------------------------------
            if chunk.token_count >= min_tokens:

                chunks.append(chunk)

                logger.info(
                    "Created chunk %d | pages=%s | tokens=%d | chars=%d",
                    chunk.chunk_index,
                    chunk.page_numbers,
                    chunk.token_count,
                    chunk.character_count,
                )

                chunk_index += 1

                # Prepare overlap for the next chunk.
                overlap_sentences, overlap_count = _build_overlap(
                    current_sentences,
                    overlap_tokens,
                )

                current_sentences = overlap_sentences
                current_tokens = overlap_count

                # If overlap + this sentence would exceed max_tokens,
                # don't carry the overlap forward.
                if current_tokens + sentence_tokens > max_tokens:
                    current_sentences = []
                    current_tokens = 0

            # -------------------------------------------------
            # Small chunk
            # -------------------------------------------------
            else:

                # Keep it rather than dropping information.
                chunks.append(chunk)

                logger.warning(
                    "Chunk below min_tokens (%d < %d); "
                    "kept to avoid data loss | pages=%s",
                    chunk.token_count,
                    min_tokens,
                    chunk.page_numbers,
                )

                chunk_index += 1

                # Do NOT carry overlap from an undersized chunk.
                current_sentences = []
                current_tokens = 0

        # Add the current sentence.
        current_sentences.append(sentence)
        current_tokens += sentence_tokens

    # -----------------------------------------------------
    # Step 4: Final chunk
    # -----------------------------------------------------

    if current_sentences:

        chunk = create_chunk(
            document=document,
            sentences=current_sentences,
            chunk_index=chunk_index,
        )

        if chunk.token_count >= min_tokens or not chunks:
            # Also keep it if it's the *only* chunk in the document --
            # better to return one undersized chunk than none at all.
            chunks.append(chunk)

            logger.info(
                "Created final chunk %d | pages=%s | tokens=%d | chars=%d",
                chunk.chunk_index,
                chunk.page_numbers,
                chunk.token_count,
                chunk.character_count,
            )

        else:
            merge_candidate = _merge_into_chunk(document, chunks[-1], current_sentences)

            if merge_candidate.token_count <= max_tokens:
                logger.warning(
                    "Final chunk below min_tokens (%d < %d); merged into "
                    "previous chunk (combined tokens=%d) | pages=%s",
                    chunk.token_count,
                    min_tokens,
                    merge_candidate.token_count,
                    chunk.page_numbers,
                )
                chunks[-1] = merge_candidate
                merged_small_chunk_count += 1

            else:
                logger.warning(
                    "Final chunk below min_tokens (%d < %d) but merging "
                    "would exceed max_tokens (%d > %d); kept as its own "
                    "undersized chunk instead | pages=%s",
                    chunk.token_count,
                    min_tokens,
                    merge_candidate.token_count,
                    max_tokens,
                    chunk.page_numbers,
                )
                chunks.append(chunk)

    before_deduplication = len(chunks)

    unique_chunks: list[Chunk] = []
    seen_text_hashes: set[str] = set()

    for chunk in chunks:

        duplicate_key = hashlib.sha256(
            f"{document.content_hash}|{chunk.text}".encode("utf-8")
        ).hexdigest()

        if duplicate_key in seen_text_hashes:
            logger.info("Duplicate chunk removed | chunk_index=%d", chunk.chunk_index)
            continue

        seen_text_hashes.add(duplicate_key)
        unique_chunks.append(chunk)

    duplicate_count = before_deduplication - len(unique_chunks)

    for index, chunk in enumerate(unique_chunks):
        chunk.chunk_index = index

    elapsed = time.perf_counter() - start_time

    total_tokens = sum(chunk.token_count for chunk in unique_chunks)

    average_tokens = (
        total_tokens / len(unique_chunks) if unique_chunks else 0
    )

    logger.info(
        "Chunking completed | document='%s' | chunks=%d | "
        "duplicates=%d | merged_small_chunks=%d | total_tokens=%d | "
        "avg_tokens=%.1f | time=%.4fs",
        document.name,
        len(unique_chunks),
        duplicate_count,
        merged_small_chunk_count,
        total_tokens,
        average_tokens,
        elapsed,
    )

    return unique_chunks, duplicate_count