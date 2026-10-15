"""
Backend for the provided React frontend (Document / KnowledgeBaseStatus /
ChatMessage / RetrievalDetails / BuildStats -- see src/api/types.ts in the
frontend project). Implements the same contract as src/api/mockApi.ts, for
real, using the actual pipeline in src/.

Two-phase flow, matching the frontend's own UX (upload, then separately
"Build Knowledge Base"):

    upload   -> extract the PDF (so real page count is known), save it,
                register it as an unselected-or-selected candidate.
                NOT indexed yet.
    build    -> chunk + embed + index whichever documents are currently
                selected. This is the only step that makes anything
                searchable, exactly like mockApi.buildKnowledgeBase()
                only operates on `documents.filter(d => d.selected)`.
    chat     -> retrieve from the built index, optionally narrowed to
                whichever documents are selected *at chat time* (which
                may differ from what was selected at build time).

Run from the project root:

    pip install fastapi "uvicorn[standard]" python-multipart
    uvicorn app:app --reload

Then run the React frontend against it (see react-frontend/README additions).
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import settings
from src.generation import answer_generator
from src.ingestion.chunker import Chunk, chunk_document
from src.ingestion.pdf_extractor import PDFProcessingError, process_pdf
from src.reranking.reranker import FlashRankReranker
from src.retrieval.bm25_retriever import BM25Store
from src.retrieval.embeddings import embed_chunks, embed_query
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.vector_store import VectorStore
from src.utils.logger import logger

app = FastAPI(title="RAG Chat Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOADS_DIR = Path(getattr(settings, "DOCUMENTS_DIR", Path("data/documents")))
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

RERANK_METHOD_NAME = "FlashRank"
CANDIDATE_POOL = getattr(settings, "RETRIEVAL_CANDIDATE_POOL", 10)
PREVIEW_CHARS = 180


# ----------------------------------------------------------------------
# State
# ----------------------------------------------------------------------

@dataclass
class UploadedDoc:
    id: str                 # == content_hash
    name: str
    file_path: Path
    size_bytes: int
    pages: int
    uploaded_at: str
    selected: bool = True


@dataclass
class KBState:
    chunks: List[Chunk] = field(default_factory=list)
    bm25: BM25Store = field(default_factory=BM25Store)
    vector_store: VectorStore = field(default_factory=VectorStore)
    hybrid: Optional[HybridRetriever] = None
    built_doc_ids: set = field(default_factory=set)
    num_pages: int = 0
    duplicate_chunks_removed: int = 0

    def __post_init__(self):
        self.hybrid = HybridRetriever(vector_store=self.vector_store, bm25_store=self.bm25)

    @property
    def is_ready(self) -> bool:
        return bool(self.chunks)

    def clear(self) -> None:
        self.chunks = []
        self.bm25 = BM25Store()
        self.vector_store = VectorStore()
        self.hybrid = HybridRetriever(vector_store=self.vector_store, bm25_store=self.bm25)
        self.built_doc_ids = set()
        self.num_pages = 0
        self.duplicate_chunks_removed = 0


UPLOADS: Dict[str, UploadedDoc] = {}
KB = KBState()
RERANKER: Optional[FlashRankReranker] = None
RERANKER_ERROR: Optional[str] = None


@app.on_event("startup")
def startup() -> None:
    global RERANKER, RERANKER_ERROR
    try:
        RERANKER = FlashRankReranker()
    except ImportError as exc:
        RERANKER_ERROR = str(exc)
        logger.warning("Reranker unavailable: %s", exc)


# ----------------------------------------------------------------------
# Response models (mirroring src/api/types.ts exactly)
# ----------------------------------------------------------------------

class DocumentOut(BaseModel):
    id: str
    name: str
    size: str
    pages: int
    uploadedAt: str
    status: str
    contentHash: Optional[str] = None
    selected: bool = True


class KnowledgeBaseStatusOut(BaseModel):
    isReady: bool
    documentsCount: int
    pagesCount: int
    chunksCount: int
    faissStatus: str
    bm25Status: str
    flashRankStatus: str
    geminiStatus: str
    embeddingModel: str


class BuildStatsOut(BaseModel):
    num_documents: int
    num_chunks: int
    duplicate_chunks_removed: int
    embedding_model: str


class SourceOut(BaseModel):
    index: int
    doc_name: str
    page: int
    chunk_id: Optional[str] = None
    preview: Optional[str] = None


class BM25HitOut(BaseModel):
    doc_name: str
    page: int
    score: float
    preview: str


class SemanticHitOut(BaseModel):
    doc_name: str
    page: int
    similarity: float
    preview: str


class RerankedHitOut(BaseModel):
    rank: int
    doc_name: str
    page: int
    score: float
    method: str
    preview: str


class MergeStatsOut(BaseModel):
    bm25_count: int
    semantic_count: int
    unique_after_merge: int
    duplicates_removed: int


class ContextStatsOut(BaseModel):
    retrieved_chunks: int
    after_reranking: int
    after_deduplication: int
    final_context: int


class RetrievalDetailsOut(BaseModel):
    query: str
    bm25_hits: List[BM25HitOut]
    semantic_hits: List[SemanticHitOut]
    merge_stats: MergeStatsOut
    reranked: List[RerankedHitOut]
    context_stats: ContextStatsOut
    final_context_string: str


class ChatMessageOut(BaseModel):
    id: str
    sender: str
    text: str
    timestamp: str
    sources: List[SourceOut] = []
    retrievalDetails: Optional[RetrievalDetailsOut] = None
    groundedness: Optional[float] = None
    isError: bool = False


class ToggleAllRequest(BaseModel):
    select: bool


class ChatRequest(BaseModel):
    query: str
    selected_doc_ids: List[str] = []


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")


def _hhmm() -> str:
    return datetime.now().strftime("%I:%M %p").lstrip("0")


def _size_str(num_bytes: int) -> str:
    return f"{num_bytes / (1024 * 1024):.1f} MB"


def _preview(text: str, n: int = PREVIEW_CHARS) -> str:
    text = " ".join(text.split())
    return f"...{text[:n]}..." if len(text) > n else text


def _doc_out(doc: UploadedDoc, status: str = "ready") -> DocumentOut:
    return DocumentOut(
        id=doc.id, name=doc.name, size=_size_str(doc.size_bytes), pages=doc.pages,
        uploadedAt=doc.uploaded_at, status=status, contentHash=doc.id, selected=doc.selected,
    )


def _documents_list() -> List[DocumentOut]:
    # Most-recently-uploaded first, matching the mock's `[...newDocs, ...this.documents]`.
    return [_doc_out(d) for d in sorted(UPLOADS.values(), key=lambda d: d.uploaded_at, reverse=True)]


# ----------------------------------------------------------------------
# Documents
# ----------------------------------------------------------------------

@app.get("/api/documents", response_model=List[DocumentOut])
def get_documents() -> List[DocumentOut]:
    return _documents_list()


@app.post("/api/documents/upload", response_model=List[DocumentOut])
async def upload_documents(files: List[UploadFile] = File(...)) -> List[DocumentOut]:
    for upload in files:
        if not upload.filename.lower().endswith(".pdf"):
            logger.warning("Skipping non-PDF upload: %s", upload.filename)
            continue

        file_bytes = await upload.read()
        try:
            document = process_pdf(upload.filename, file_bytes)
        except PDFProcessingError as exc:
            # One bad PDF in a multi-file drop shouldn't fail the others --
            # the frontend has no per-file error slot to show this in anyway,
            # so we skip it and log, matching the mock's silent-success UX.
            logger.warning("Skipping '%s': %s", upload.filename, exc)
            continue

        doc_id = document.content_hash
        file_path = UPLOADS_DIR / f"{doc_id}.pdf"
        file_path.write_bytes(file_bytes)

        UPLOADS[doc_id] = UploadedDoc(
            id=doc_id,
            name=upload.filename,
            file_path=file_path,
            size_bytes=len(file_bytes),
            pages=document.num_pages,
            uploaded_at=_now_str(),
            selected=True,
        )

    return _documents_list()


@app.delete("/api/documents/{doc_id}", response_model=List[DocumentOut])
def remove_document(doc_id: str) -> List[DocumentOut]:
    doc = UPLOADS.pop(doc_id, None)
    if doc and doc.file_path.exists():
        doc.file_path.unlink()

    if not UPLOADS:
        KB.clear()

    return _documents_list()


@app.post("/api/documents/{doc_id}/toggle", response_model=List[DocumentOut])
def toggle_document(doc_id: str) -> List[DocumentOut]:
    doc = UPLOADS.get(doc_id)
    if doc:
        doc.selected = not doc.selected
    return _documents_list()


@app.post("/api/documents/toggle-all", response_model=List[DocumentOut])
def toggle_all_documents(request: ToggleAllRequest) -> List[DocumentOut]:
    for doc in UPLOADS.values():
        doc.selected = request.select
    return _documents_list()


# ----------------------------------------------------------------------
# Knowledge base status + build
# ----------------------------------------------------------------------

@app.get("/api/kb/status", response_model=KnowledgeBaseStatusOut)
def get_kb_status() -> KnowledgeBaseStatusOut:
    ready = KB.is_ready
    return KnowledgeBaseStatusOut(
        isReady=ready,
        documentsCount=len(KB.built_doc_ids),
        pagesCount=KB.num_pages,
        chunksCount=len(KB.chunks),
        faissStatus="ready" if ready else "empty",
        bm25Status="ready" if ready else "empty",
        flashRankStatus="ready" if RERANKER is not None else "empty",
        geminiStatus="ready" if answer_generator.is_available() else "unconfigured",
        embeddingModel=settings.EMBEDDING_MODEL,
    )


@app.post("/api/kb/build", response_model=BuildStatsOut)
def build_knowledge_base() -> BuildStatsOut:
    selected = [doc for doc in UPLOADS.values() if doc.selected]
    if not selected:
        raise HTTPException(status_code=400, detail="No documents are selected to build.")

    all_chunks: List[Chunk] = []
    raw_chunk_count = 0

    for doc in selected:
        try:
            document = process_pdf(doc.name, doc.file_path.read_bytes())
            chunks, _dupes_within_doc = chunk_document(document)
        except Exception as exc:
            logger.error("Failed to chunk '%s' during build: %s", doc.name, exc)
            raise HTTPException(status_code=500, detail=f"Failed processing '{doc.name}': {exc}") from exc

        raw_chunk_count += len(chunks)
        all_chunks.extend(chunks)

    # De-duplicate across documents by chunk_id (identical text appearing in
    # more than one uploaded file).
    seen: set = set()
    unique_chunks: List[Chunk] = []
    for chunk in all_chunks:
        if chunk.chunk_id in seen:
            continue
        seen.add(chunk.chunk_id)
        unique_chunks.append(chunk)

    usable_chunks = [c for c in unique_chunks if re.search(r"[A-Za-z0-9]", c.text)]

    if not usable_chunks:
        raise HTTPException(status_code=400, detail="Selected documents produced no usable text.")

    try:
        embeddings = embed_chunks(usable_chunks)
        KB.bm25.build(usable_chunks)
        KB.vector_store.build(usable_chunks, embeddings)
    except Exception as exc:
        logger.error("Indexing failed during build: %s", exc)
        raise HTTPException(status_code=500, detail=f"Indexing failed: {exc}") from exc

    KB.chunks = usable_chunks
    KB.hybrid = HybridRetriever(vector_store=KB.vector_store, bm25_store=KB.bm25)
    KB.built_doc_ids = {doc.id for doc in selected}
    KB.num_pages = sum(doc.pages for doc in selected)
    KB.duplicate_chunks_removed = raw_chunk_count - len(usable_chunks)

    return BuildStatsOut(
        num_documents=len(selected),
        num_chunks=len(usable_chunks),
        duplicate_chunks_removed=KB.duplicate_chunks_removed,
        embedding_model=settings.EMBEDDING_MODEL,
    )


# ----------------------------------------------------------------------
# Chat
# ----------------------------------------------------------------------

def _filter_by_doc(pairs, allowed_ids: Optional[set]):
    if not allowed_ids:
        return pairs
    return [(chunk, score) for chunk, score in pairs if chunk.document_hash in allowed_ids]


def _fallback_answer(chunks: List[Chunk]) -> str:
    if not chunks:
        return "No relevant passages were found for this question."
    lines = ["Generation isn't configured (set GROQ_API_KEY in .env to enable it), "
             "so here are the most relevant passages retrieved for your question:\n"]
    for i, chunk in enumerate(chunks, start=1):
        lines.append(f"[{i}] ({chunk.document_name}, Page {chunk.page_numbers[0]}): {_preview(chunk.text, 220)}")
    return "\n\n".join(lines)


@app.post("/api/chat/ask", response_model=ChatMessageOut)
def chat_ask(request: ChatRequest) -> ChatMessageOut:
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    if not KB.is_ready:
        raise HTTPException(
            status_code=400,
            detail="The knowledge base hasn't been built yet. Select documents and build it first.",
        )

    # Only restrict to a subset if the caller actually selected a strict
    # subset of what's indexed -- an empty or full selection means "search
    # everything that's indexed", matching how a user would expect "select
    # all" or "deselect all mid-chat" to behave.
    allowed_ids = set(request.selected_doc_ids) & KB.built_doc_ids
    scope = allowed_ids if (allowed_ids and allowed_ids != KB.built_doc_ids) else None

    t0 = time.perf_counter()

    bm25_pairs_full = _filter_by_doc(KB.bm25.search(query, top_k=CANDIDATE_POOL), scope)
    query_embedding = embed_query(query)
    dense_pairs_full = _filter_by_doc(KB.vector_store.search(query_embedding, top_k=CANDIDATE_POOL), scope)

    hybrid_results = KB.hybrid.retrieve(
        query, query_embedding=query_embedding, top_k=CANDIDATE_POOL, candidate_pool=CANDIDATE_POOL,
    )
    if scope:
        hybrid_results = [r for r in hybrid_results if r.chunk.document_hash in scope]

    reranked_results = []
    if RERANKER is not None and hybrid_results:
        try:
            response = RERANKER.rerank(query, hybrid_results, top_k=min(5, len(hybrid_results)))
            reranked_results = response.results
        except Exception as exc:
            logger.warning("Reranking failed, falling back to hybrid order: %s", exc)

    ranked_for_answer = reranked_results or [
        type("FallbackRank", (), {"chunk": r.chunk, "rerank_score": r.score})() for r in hybrid_results
    ]
    top_for_answer = ranked_for_answer[: answer_generator.MAX_CONTEXT_CHUNKS]
    answer_chunks = [r.chunk for r in top_for_answer]

    if answer_generator.is_available():
        try:
            generated = answer_generator.generate_answer(query, answer_chunks)
            answer_text = generated.text
        except RuntimeError as exc:
            answer_text = f"(Generation failed: {exc})\n\n" + _fallback_answer(answer_chunks)
    else:
        answer_text = _fallback_answer(answer_chunks)

    sources = [
        SourceOut(
            index=i, doc_name=chunk.document_name, page=chunk.page_numbers[0],
            chunk_id=chunk.chunk_id, preview=_preview(chunk.text),
        )
        for i, chunk in enumerate(answer_chunks, start=1)
    ]

    unique_pool_ids = {c.chunk_id for c, _ in bm25_pairs_full} | {c.chunk_id for c, _ in dense_pairs_full}
    merge_stats = MergeStatsOut(
        bm25_count=len(bm25_pairs_full),
        semantic_count=len(dense_pairs_full),
        unique_after_merge=len(unique_pool_ids),
        duplicates_removed=len(bm25_pairs_full) + len(dense_pairs_full) - len(unique_pool_ids),
    )

    context_stats = ContextStatsOut(
        retrieved_chunks=len(hybrid_results),
        after_reranking=len(reranked_results) or len(hybrid_results),
        after_deduplication=len(ranked_for_answer),
        final_context=len(answer_chunks),
    )

    final_context_string = "\n\n".join(
        f"[{i}] (Source: {chunk.document_name}, Page {chunk.page_numbers[0]})\n{chunk.text}"
        for i, chunk in enumerate(answer_chunks, start=1)
    )

    retrieval_details = RetrievalDetailsOut(
        query=query,
        bm25_hits=[
            BM25HitOut(doc_name=c.document_name, page=c.page_numbers[0], score=round(s, 3), preview=_preview(c.text))
            for c, s in bm25_pairs_full[:5]
        ],
        semantic_hits=[
            SemanticHitOut(doc_name=c.document_name, page=c.page_numbers[0], similarity=round(s, 4), preview=_preview(c.text))
            for c, s in dense_pairs_full[:5]
        ],
        merge_stats=merge_stats,
        reranked=[
            RerankedHitOut(
                rank=i, doc_name=r.chunk.document_name, page=r.chunk.page_numbers[0],
                score=round(r.rerank_score, 3),
                method=RERANK_METHOD_NAME if reranked_results else "Hybrid (no reranker)",
                preview=_preview(r.chunk.text),
            )
            for i, r in enumerate(ranked_for_answer[:5], start=1)
        ],
        context_stats=context_stats,
        final_context_string=final_context_string,
    )

    # Heuristic groundedness: FlashRank's cross-encoder scores already sit
    # roughly in 0..1, so the top result's score is used directly, clamped.
    # This is NOT a faithfulness/hallucination check against the generated
    # text -- it's a proxy for "how confident was retrieval", shown as-is.
    groundedness = None
    if ranked_for_answer:
        top_score = getattr(ranked_for_answer[0], "rerank_score", None)
        if top_score is not None:
            groundedness = round(min(1.0, max(0.0, float(top_score))), 3)

    logger.info("Chat query answered in %.3fs | query=%r", time.perf_counter() - t0, query)

    return ChatMessageOut(
        id=f"msg-{int(time.time() * 1000)}",
        sender="assistant",
        text=answer_text,
        timestamp=_hhmm(),
        sources=sources,
        retrievalDetails=retrieval_details,
        groundedness=groundedness,
    )


# ----------------------------------------------------------------------
# Static frontend (the built React app lands in static/ via `npm run build`)
#
# Mounted last and at "/" so it acts as a catch-all: the specific /api/...
# routes above are matched first (Starlette checks routes in the order
# they were registered), and anything else falls through to the static
# files -- including a SPA refresh on a client-side route, which html=True
# resolves to index.html instead of a 404.
# ----------------------------------------------------------------------

app.mount("/", StaticFiles(directory="static", html=True), name="static")
