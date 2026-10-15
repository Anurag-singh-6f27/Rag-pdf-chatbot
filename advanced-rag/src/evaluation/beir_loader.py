"""
Downloads and loads a BEIR benchmark dataset for retrieval evaluation.

BEIR datasets are self-contained: a corpus of documents, a set of test
queries, and human-annotated relevance judgments (qrels) saying which
documents are actually relevant to which query. That combination is
exactly the ground truth you need to score BM25 / semantic / hybrid /
reranked retrieval.

Recommended datasets for a project like this (small enough to embed
and index on a laptop CPU in a few minutes):
    "scifact"   -  5,183 docs |   300 test queries  (science abstracts)
    "nfcorpus"  -  3,633 docs |   323 test queries  (medical)
    "fiqa"      - 57,638 docs |   648 test queries  (finance, larger)

Usage:
    from evaluation.beir_loader import load_beir_dataset
    corpus, queries, qrels = load_beir_dataset("scifact", split="test")
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Dict, Tuple

import requests

BEIR_URL_TEMPLATE = (
    "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
)

DEFAULT_CACHE_DIR = Path("./data/beir")


def download_beir_dataset(dataset_name: str, cache_dir: Path = DEFAULT_CACHE_DIR) -> Path:
    """
    Download and unzip a BEIR dataset if it isn't already cached locally.
    Returns the path to the extracted dataset directory.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = cache_dir / dataset_name

    if dataset_dir.exists():
        return dataset_dir

    zip_path = cache_dir / f"{dataset_name}.zip"
    url = BEIR_URL_TEMPLATE.format(name=dataset_name)

    print(f"Downloading {dataset_name} from {url} ...")
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()

    with zip_path.open("wb") as file:
        for chunk in response.iter_content(chunk_size=1 << 20):
            file.write(chunk)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(cache_dir)

    zip_path.unlink()
    print(f"Saved to {dataset_dir}")
    return dataset_dir


def load_beir_dataset(
    dataset_name: str,
    split: str = "test",
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> Tuple[Dict[str, dict], Dict[str, str], Dict[str, Dict[str, int]]]:
    """
    Returns:
        corpus: {doc_id: {"title": str, "text": str}}
        queries: {query_id: query_text}
        qrels: {query_id: {doc_id: relevance_score}}   (score > 0 means relevant)
    """
    dataset_dir = download_beir_dataset(dataset_name, cache_dir)

    # --- corpus ---
    corpus: Dict[str, dict] = {}
    with (dataset_dir / "corpus.jsonl").open("r", encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            corpus[row["_id"]] = {
                "title": row.get("title", ""),
                "text": row.get("text", ""),
            }

    # --- queries ---
    all_queries: Dict[str, str] = {}
    with (dataset_dir / "queries.jsonl").open("r", encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            all_queries[row["_id"]] = row["text"]

    # --- qrels (tab-separated: query-id, corpus-id, score) ---
    qrels_path = dataset_dir / "qrels" / f"{split}.tsv"
    qrels: Dict[str, Dict[str, int]] = {}
    with qrels_path.open("r", encoding="utf-8") as file:
        next(file)  # header row
        for line in file:
            query_id, doc_id, score = line.strip().split("\t")
            qrels.setdefault(query_id, {})[doc_id] = int(score)

    # Only keep queries that actually have qrels for this split.
    queries = {qid: text for qid, text in all_queries.items() if qid in qrels}

    print(
        f"{dataset_name} [{split}]: "
        f"{len(corpus)} docs | {len(queries)} queries | "
        f"{sum(len(v) for v in qrels.values())} relevance judgments"
    )

    return corpus, queries, qrels
