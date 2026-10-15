export interface Document {
  id: string;
  name: string;
  size: string;
  pages: number;
  uploadedAt: string;
  status: 'ready' | 'processing' | 'error';
  contentHash?: string;
  selected?: boolean;
}

export interface Source {
  index: number;
  doc_name: string;
  page: number;
  chunk_id?: string;
  preview?: string;
}

export interface BM25Hit {
  doc_name: string;
  page: number;
  score: number;
  preview: string;
}

export interface SemanticHit {
  doc_name: string;
  page: number;
  similarity: number;
  preview: string;
}

export interface RerankedHit {
  rank: number;
  doc_name: string;
  page: number;
  score: number;
  method: string;
  preview: string;
}

export interface MergeStats {
  bm25_count: number;
  semantic_count: number;
  unique_after_merge: number;
  duplicates_removed: number;
}

export interface ContextStats {
  retrieved_chunks: number;
  after_reranking: number;
  after_deduplication: number;
  final_context: number;
}

export interface RetrievalDetails {
  query: string;
  bm25_hits: BM25Hit[];
  semantic_hits: SemanticHit[];
  merge_stats: MergeStats;
  reranked: RerankedHit[];
  context_stats: ContextStats;
  final_context_string?: string;
}

export interface ChatMessage {
  id: string;
  sender: 'user' | 'assistant';
  text: string;
  timestamp: string;
  sources?: Source[];
  retrievalDetails?: RetrievalDetails;
  groundedness?: number;
  isLoading?: boolean;
  isError?: boolean;
}

export interface KnowledgeBaseStatus {
  isReady: boolean;
  documentsCount: number;
  pagesCount: number;
  chunksCount: number;
  faissStatus: 'ready' | 'building' | 'empty';
  bm25Status: 'ready' | 'building' | 'empty';
  flashRankStatus: 'ready' | 'building' | 'empty';
  geminiStatus: 'ready' | 'unconfigured' | 'error';
  embeddingModel: string;
}

export interface BuildStats {
  num_documents: number;
  num_chunks: number;
  duplicate_chunks_removed: number;
  embedding_model: string;
}
