from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================
BM25_K1 = 1.5
BM25_B = 0.75
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
VECTORSTORE_DIR = DATA_DIR / "indexes"

BM25_INDEX_PATH = VECTORSTORE_DIR / "bm25.json"
# ============================================================
# VECTOR STORE
# ============================================================

FAISS_INDEX_PATH = VECTORSTORE_DIR / "faiss.index"
METADATA_PATH = VECTORSTORE_DIR / "chunks.json"


# ============================================================
# EMBEDDING / TOKENIZER
# ============================================================

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
TOKENIZER_NAME = "BAAI/bge-small-en-v1.5"


# ============================================================
# CHUNKING
# ============================================================

MAX_CHUNK_TOKENS = 220
CHUNK_OVERLAP_TOKENS = 40
MIN_CHUNK_TOKENS = 20