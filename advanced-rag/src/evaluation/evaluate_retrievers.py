"""
Evaluate BM25 / semantic (FAISS) / hybrid (RRF) / hybrid+reranked retrieval
on a BEIR benchmark dataset, using YOUR project's actual retriever classes.

Drop this file (and beir_loader.py, metrics.py) into src/evaluation/,
then run from the project root:

    python -m src.evaluation.evaluate_retrievers --dataset scifact --top-k 10

Requirements (on top of what your project already uses):
    pip install requests matplotlib

What it does
------------
1. Downloads a BEIR dataset (corpus + test queries + relevance judgments).
2. Wraps every BEIR document in your project's `Chunk` dataclass and builds
   a fresh BM25Store and VectorStore from them (in-memory, not saved over
   your real indices).
3. Runs every test query through:
       - BM25 alone
       - Semantic (FAISS) alone
       - Hybrid (BM25 + FAISS via RRF)
       - Hybrid candidates re-ordered by FlashRank reranker
4. Scores each method's ranked list against the ground-truth relevant
   document ids (qrels) with Hit Rate@k, Recall@k, MRR, NDCG@k.
5. Prints a results table, saves it as CSV, and saves two chart PNGs:
       - retrieval_accuracy_bar.png   (Hit Rate@k side-by-side per method)
       - retrieval_accuracy_curve.png (Hit Rate vs k, one line per method)
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from src.ingestion.chunker import Chunk
from src.reranking.reranker import FlashRankReranker
from src.retrieval.bm25_retriever import BM25Store
from src.retrieval.embeddings import embed_texts
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.vector_store import VectorStore

from .beir_loader import load_beir_dataset
from .diagonostics import fused_order, run_diagnostics
from .metrics import aggregate, evaluate_single_query

K_VALUES = (1, 3, 5, 10)
METHOD_NAMES = ["BM25", "Semantic (FAISS)", "Hybrid (RRF)", "Hybrid + Reranked", "Hybrid + Reranked (fused)"]


# ----------------------------------------------------------------------
# BUILD INDICES FROM A BEIR CORPUS
# ----------------------------------------------------------------------

def corpus_to_chunks(corpus: Dict[str, dict]) -> List[Chunk]:
    """Wrap each BEIR document as one Chunk (title + text)."""
    chunks = []
    for doc_id, doc in corpus.items():
        text = (doc["title"] + ". " + doc["text"]).strip() if doc["title"] else doc["text"]
        if not text.strip():
            continue
        chunks.append(
            Chunk(
                chunk_id=doc_id,
                document_name=doc_id,
                document_hash=doc_id,
                page_numbers=[1],
                chunk_index=0,
                text=text,
                token_count=len(text.split()),
                character_count=len(text),
            )
        )
    return chunks


def build_indices(chunks: List[Chunk], embed_batch_size: int = 64,
                   cache_dir: Path | None = None, dataset_name: str = "default"):
    bm25 = BM25Store()
    bm25.build(chunks)
    print(f"BM25 index built ({len(chunks)} docs).")

    embeddings = None
    cache_path = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{dataset_name}_embeddings.npz"
        if cache_path.exists():
            print(f"Found cached embeddings at {cache_path} -- checking it matches this corpus ...")
            cached = np.load(cache_path, allow_pickle=True)
            cached_ids = list(cached["chunk_ids"])
            current_ids = [c.chunk_id for c in chunks]
            if cached_ids == current_ids:
                embeddings = cached["embeddings"]
                print(f"Cache matches ({len(chunks)} docs) -- skipping embedding, saved ~{len(chunks)*0.18:.0f}s.")
            else:
                print("Cache doesn't match this corpus (different dataset/order) -- re-embedding.")

    if embeddings is None:
        print(f"Embedding {len(chunks)} documents (this step is silent internally -- "
              f"showing chunk-level progress here instead) ...")

        embed_chunk_size = 500
        embeddings_parts = []
        start = time.perf_counter()
        for i in range(0, len(chunks), embed_chunk_size):
            batch = chunks[i : i + embed_chunk_size]
            embeddings_parts.append(
                embed_texts([c.text for c in batch], batch_size=embed_batch_size)
            )
            done = min(i + embed_chunk_size, len(chunks))
            elapsed = time.perf_counter() - start
            rate = done / elapsed if elapsed > 0 else 0
            remaining = (len(chunks) - done) / rate if rate > 0 else float("inf")
            print(f"  embedded {done}/{len(chunks)} docs "
                  f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")
        embeddings = np.concatenate(embeddings_parts, axis=0)

        if cache_path is not None:
            np.savez(cache_path, embeddings=embeddings, chunk_ids=[c.chunk_id for c in chunks])
            print(f"Cached embeddings to {cache_path} -- future runs on this dataset will skip embedding entirely.")

    vector_store = VectorStore()
    vector_store.build(chunks, embeddings)
    print("FAISS index built.")

    hybrid = HybridRetriever(vector_store=vector_store, bm25_store=bm25)

    return bm25, vector_store, hybrid


# ----------------------------------------------------------------------
# RUN ALL FOUR RETRIEVAL METHODS FOR ONE QUERY
# ----------------------------------------------------------------------

def rank_all_methods(
    query: str,
    bm25: BM25Store,
    vector_store: VectorStore,
    hybrid: HybridRetriever,
    reranker: FlashRankReranker | None,
    top_k: int,
    candidate_pool: int,
    fusion_alpha: float = 0.5,
):
    """Returns (rankings_by_method, trace). The trace feeds diagnostics.py."""
    latency: Dict[str, float] = {}

    t = time.perf_counter()
    bm25_ranked = [c.chunk_id for c, _ in bm25.search(query, top_k=top_k)]
    latency["bm25"] = time.perf_counter() - t

    t = time.perf_counter()
    query_embedding = embed_texts([query])[0]
    dense_ranked = [c.chunk_id for c, _ in vector_store.search(query_embedding, top_k=top_k)]
    latency["dense"] = time.perf_counter() - t

    t = time.perf_counter()
    hybrid_results = hybrid.retrieve(
        query,
        query_embedding=query_embedding,
        top_k=candidate_pool,
        candidate_pool=candidate_pool,
    )
    latency["hybrid"] = time.perf_counter() - t

    pool_ids = [r.chunk.chunk_id for r in hybrid_results]
    hybrid_scores = {r.chunk.chunk_id: float(r.score) for r in hybrid_results}
    hybrid_ranked = pool_ids[:top_k]

    rerank_scores: Dict[str, float] = {}
    rerank_order: List[str] = []
    if reranker is not None and hybrid_results:
        # Score EVERY candidate so we can both (a) use the reranker's order
        # alone and (b) blend its score with the hybrid retriever's score.
        t = time.perf_counter()
        rerank_response = reranker.rerank(query, hybrid_results, top_k=len(hybrid_results))
        latency["rerank"] = time.perf_counter() - t

        rerank_scores = {r.chunk_id: r.rerank_score for r in rerank_response.results}
        rerank_order = [r.chunk_id for r in rerank_response.results]
        reranked_ranked = rerank_order[:top_k]
        fused_ranked = fused_order(pool_ids, hybrid_scores, rerank_scores, fusion_alpha)[:top_k]
    else:
        reranked_ranked = hybrid_ranked  # fall back if flashrank isn't installed
        fused_ranked = hybrid_ranked

    rankings = {
        "BM25": bm25_ranked,
        "Semantic (FAISS)": dense_ranked,
        "Hybrid (RRF)": hybrid_ranked,
        "Hybrid + Reranked": reranked_ranked,
        "Hybrid + Reranked (fused)": fused_ranked,
    }
    trace = {
        "pool_ids": pool_ids,
        "hybrid_scores": hybrid_scores,
        "rerank_scores": rerank_scores,
        "rerank_order": rerank_order,
        "bm25_ids": bm25_ranked,
        "dense_ids": dense_ranked,
        "latency": latency,
    }
    return rankings, trace


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="scifact", help="BEIR dataset name")
    parser.add_argument("--split", default="test")
    parser.add_argument("--top-k", type=int, default=10, help="Max results per method")
    parser.add_argument("--candidate-pool", type=int, default=30, help="Pool size fed into RRF/reranker")
    parser.add_argument("--max-queries", type=int, default=None, help="Limit queries for a quick run")
    parser.add_argument("--headline-k", type=int, default=5, help="k used in the bar chart")
    parser.add_argument(
        "--reranker-model",
        default=None,
        help=(
            "FlashRank model to use. Leave unset to use FlashRankReranker's own "
            "default (defined once, in reranker.py -- not duplicated here). Pass "
            "e.g. 'rank-T5-flan' (most accurate, slowest) or 'ms-marco-TinyBERT-L-2-v2' "
            "(tiny/fast, weakest) to override it for a single run."
        ),
    )
    parser.add_argument(
        "--fusion-alpha", type=float, default=0.5,
        help="Weight of the reranker score when blending with the hybrid score "
             "(0 = hybrid only, 1 = reranker only).",
    )
    parser.add_argument("--out-dir", default="./eval_results")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus, queries, qrels = load_beir_dataset(args.dataset, split=args.split)
    chunks = corpus_to_chunks(corpus)
    embed_cache_dir = out_dir / "embed_cache"
    bm25, vector_store, hybrid = build_indices(chunks, cache_dir=embed_cache_dir, dataset_name=args.dataset)

    try:
        reranker_kwargs = {"model_name": args.reranker_model} if args.reranker_model else {}
        reranker = FlashRankReranker(**reranker_kwargs)
        print(f"Reranker model: {reranker.model_name}")
        print("Loading/downloading reranker model now (first run only) ...")
        load_start = time.perf_counter()
        reranker._get_ranker()  # force eager load so any delay is clearly attributed here
        print(f"Reranker model ready in {time.perf_counter() - load_start:.1f}s.")
    except ImportError:
        print("flashrank not installed (pip install flashrank) -- skipping reranker stage.")
        reranker = None

    query_items = list(queries.items())
    if args.max_queries:
        query_items = query_items[: args.max_queries]

    per_method_scores: Dict[str, list] = {name: [] for name in METHOD_NAMES}
    records: List[dict] = []

    for i, (query_id, query_text) in enumerate(query_items, start=1):
        relevant_ids = {doc_id for doc_id, score in qrels.get(query_id, {}).items() if score > 0}
        if not relevant_ids:
            continue

        ranked_by_method, trace = rank_all_methods(
            query_text, bm25, vector_store, hybrid, reranker,
            top_k=args.top_k, candidate_pool=args.candidate_pool,
            fusion_alpha=args.fusion_alpha,
        )

        records.append({"query_id": query_id, "query": query_text,
                        "relevant": relevant_ids, **trace})

        for method_name, ranked_ids in ranked_by_method.items():
            per_method_scores[method_name].append(
                evaluate_single_query(ranked_ids, relevant_ids, k_values=K_VALUES)
            )

        if i % 10 == 0:
            print(f"  evaluated {i}/{len(query_items)} queries")

    results = {name: aggregate(scores) for name, scores in per_method_scores.items()}

    print_and_save_table(results, out_dir)
    plot_bar_chart(results, args.headline_k, out_dir)
    plot_curve_chart(results, out_dir)

    run_diagnostics(
        records,
        out_dir=out_dir,
        text_by_id={c.chunk_id: c.text for c in chunks},
        top_k=args.top_k,
        candidate_pool=args.candidate_pool,
        fusion_alpha=args.fusion_alpha,
        reranker_name=reranker.model_name if reranker else "none",
        reranker_max_length=reranker.max_length if reranker else 512,
    )


def print_and_save_table(results: Dict[str, Dict[str, float]], out_dir: Path):
    metric_names = list(next(iter(results.values())).keys())

    print("\n=== Retrieval accuracy by method ===")
    header = "Method".ljust(20) + "".join(m.ljust(12) for m in metric_names)
    print(header)
    for method, scores in results.items():
        row = method.ljust(20) + "".join(f"{scores[m]:.3f}".ljust(12) for m in metric_names)
        print(row)

    csv_path = out_dir / "retrieval_accuracy.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["method"] + metric_names)
        for method, scores in results.items():
            writer.writerow([method] + [scores[m] for m in metric_names])
    print(f"\nSaved table -> {csv_path}")


def plot_bar_chart(results: Dict[str, Dict[str, float]], k: int, out_dir: Path):
    methods = list(results.keys())
    values = [results[m][f"hit@{k}"] * 100 for m in methods]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(methods, values, color=["#8891A8", "#5B7FBF", "#3E9A6E", "#D98E3B", "#B5533C"])
    ax.set_ylabel(f"Hit Rate @ {k} (%)")
    ax.set_title(f"Retrieval Accuracy by Method (Hit Rate @ {k})")
    ax.set_ylim(0, 100)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.5, f"{value:.1f}%", ha="center")
    plt.xticks(rotation=15)
    plt.tight_layout()

    path = out_dir / "retrieval_accuracy_bar.png"
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved bar chart -> {path}")


def plot_curve_chart(results: Dict[str, Dict[str, float]], out_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for method, scores in results.items():
        ys = [scores[f"hit@{k}"] * 100 for k in K_VALUES]
        ax.plot(K_VALUES, ys, marker="o", label=method)

    ax.set_xlabel("k")
    ax.set_ylabel("Hit Rate @ k (%)")
    ax.set_title("Retrieval Accuracy vs. k")
    ax.set_xticks(list(K_VALUES))
    ax.legend()
    plt.tight_layout()

    path = out_dir / "retrieval_accuracy_curve.png"
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved curve chart -> {path}")


if __name__ == "__main__":
    main()