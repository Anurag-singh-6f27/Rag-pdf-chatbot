"""
Answer generation (the "G" in RAG) using Groq.

Groq's API is OpenAI-compatible, so this uses the `openai` package (already in
requirements.txt) pointed at https://api.groq.com/openai/v1.

Contract expected by app.py:
    MAX_CONTEXT_CHUNKS : int
    is_available()     -> bool
    generate_answer(query, chunks) -> GeneratedAnswer   (has .text)
        raises RuntimeError on any generation failure.

Configuration (environment variables or the project's .env file):
    GROQ_API_KEY  -- required to enable generation (starts with "gsk_")
    GROQ_MODEL    -- optional, defaults to "openai/gpt-oss-120b"
                     (see https://console.groq.com/docs/models)
"""

from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from dotenv import load_dotenv

from config import settings
from src.ingestion.chunker import Chunk
from src.utils.logger import logger

load_dotenv(settings.BASE_DIR / ".env")

MAX_CONTEXT_CHUNKS: int = getattr(settings, "MAX_CONTEXT_CHUNKS", 5)
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "openai/gpt-oss-120b"
TEMPERATURE = 0.2
MAX_OUTPUT_TOKENS = 2048
REQUEST_TIMEOUT_SECONDS = 120

SYSTEM_PROMPT = (
    "You are a careful assistant that answers questions strictly using the "
    "numbered context passages provided. Rules:\n"
    "1. Use ONLY the provided context. Do not use outside knowledge.\n"
    "2. If the context does not contain the answer, say so plainly instead of guessing.\n"
    "3. Cite supporting passages inline with their numbers, e.g. [1] or [2][3].\n"
    "4. Be concise and accurate."
)


@dataclass
class GeneratedAnswer:
    text: str
    model: str
    cited_indices: List[int] = field(default_factory=list)


_client = None
_client_lock = threading.Lock()


def _api_key() -> Optional[str]:
    return os.getenv("GROQ_API_KEY") or None


def _model_name() -> str:
    return os.getenv("GROQ_MODEL", DEFAULT_MODEL)


def is_available() -> bool:
    """True when an API key is set and the openai SDK is installed."""
    if not _api_key():
        return False
    try:
        import openai  # noqa: F401
    except ImportError:
        return False
    return True


def _get_client():
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                from openai import OpenAI

                _client = OpenAI(
                    api_key=_api_key(),
                    base_url=GROQ_BASE_URL,
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )
    return _client


def build_context(chunks: Sequence[Chunk]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        page = chunk.page_numbers[0] if chunk.page_numbers else "?"
        parts.append(f"[{i}] (Source: {chunk.document_name}, Page {page})\n{chunk.text}")
    return "\n\n".join(parts)


def build_prompt(query: str, chunks: Sequence[Chunk]) -> str:
    return (
        f"Context passages:\n\n{build_context(chunks)}\n\n"
        f"Question: {query}\n\n"
        "Answer using only the context above, with [n] citations:"
    )


def _extract_citations(text: str, max_index: int) -> List[int]:
    found = {int(m) for m in re.findall(r"\[(\d+)\]", text)}
    return sorted(i for i in found if 1 <= i <= max_index)


def generate_answer(query: str, chunks: Sequence[Chunk]) -> GeneratedAnswer:
    if not is_available():
        raise RuntimeError("Generation is not configured (set GROQ_API_KEY in .env).")
    if not chunks:
        return GeneratedAnswer(
            text="No relevant passages were found for this question.",
            model=_model_name(),
        )

    chunks = list(chunks)[:MAX_CONTEXT_CHUNKS]

    try:
        response = _get_client().chat.completions.create(
            model=_model_name(),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(query, chunks)},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_OUTPUT_TOKENS,
        )
    except Exception as exc:  # network, auth, quota, bad model name, etc.
        logger.error("Groq generation failed: %s", exc)
        raise RuntimeError(str(exc)) from exc

    text = ""
    if response.choices:
        text = (response.choices[0].message.content or "").strip()
    if not text:
        raise RuntimeError("The model returned an empty response.")

    return GeneratedAnswer(
        text=text,
        model=_model_name(),
        cited_indices=_extract_citations(text, len(chunks)),
    )