import type {
  Document,
  KnowledgeBaseStatus,
  ChatMessage,
  RetrievalDetails,
  BuildStats,
} from './types';

class MockApiService {
  private documents: Document[] = [];
  private builtDocsCount: number = 0;
  private builtPagesCount: number = 0;
  private builtChunksCount: number = 0;
  private isKbBuilt: boolean = false;

  public async getDocuments(): Promise<Document[]> {
    await this.delay(100);
    return [...this.documents];
  }

  public async uploadDocuments(files: File[]): Promise<Document[]> {
    await this.delay(300);
    const newDocs: Document[] = files.map((file, idx) => {
      const sizeMb = (file.size / (1024 * 1024)).toFixed(1);
      // Realistic page count calculation (~75KB per page or fallback)
      const estimatedPages = Math.max(1, Math.round(file.size / 75000) || 4);
      return {
        id: `doc-${Date.now()}-${idx}`,
        name: file.name,
        size: `${sizeMb} MB`,
        pages: estimatedPages,
        uploadedAt: new Date().toISOString().replace('T', ' ').slice(0, 16),
        status: 'ready',
        selected: true,
      };
    });

    this.documents = [...newDocs, ...this.documents];
    return [...this.documents];
  }

  public async removeDocument(id: string): Promise<Document[]> {
    await this.delay(100);
    this.documents = this.documents.filter((doc) => doc.id !== id);
    if (this.documents.length === 0) {
      this.isKbBuilt = false;
      this.builtDocsCount = 0;
      this.builtPagesCount = 0;
      this.builtChunksCount = 0;
    }
    return [...this.documents];
  }

  public async toggleDocumentSelection(id: string): Promise<Document[]> {
    this.documents = this.documents.map((doc) =>
      doc.id === id ? { ...doc, selected: !doc.selected } : doc
    );
    return [...this.documents];
  }

  public async toggleAllDocuments(select: boolean): Promise<Document[]> {
    this.documents = this.documents.map((doc) => ({ ...doc, selected: select }));
    return [...this.documents];
  }

  public async getKbStatus(): Promise<KnowledgeBaseStatus> {
    await this.delay(50);
    
    if (!this.isKbBuilt) {
      return {
        isReady: false,
        documentsCount: 0,
        pagesCount: 0,
        chunksCount: 0,
        faissStatus: 'empty',
        bm25Status: 'empty',
        flashRankStatus: 'empty',
        geminiStatus: 'ready',
        embeddingModel: 'sentence-transformers/all-MiniLM-L6-v2',
      };
    }

    return {
      isReady: true,
      documentsCount: this.builtDocsCount,
      pagesCount: this.builtPagesCount,
      chunksCount: this.builtChunksCount,
      faissStatus: 'ready',
      bm25Status: 'ready',
      flashRankStatus: 'ready',
      geminiStatus: 'ready',
      embeddingModel: 'sentence-transformers/all-MiniLM-L6-v2',
    };
  }

  public async buildKnowledgeBase(
    onProgress?: (stage: string) => void
  ): Promise<BuildStats> {
    const selectedDocs = this.documents.filter((d) => d.selected);
    const stages = [
      'Reading and extracting text from uploaded PDFs...',
      'Splitting documents into overlapping chunks (800 chars)...',
      'Generating sentence-transformers embeddings (384-d)...',
      'Building FAISS dense vector index (IndexFlatIP)...',
      'Building BM25 sparse keyword index (BM25Okapi)...',
      'Saving knowledge base artifacts to disk...',
    ];

    for (const stage of stages) {
      if (onProgress) onProgress(stage);
      await this.delay(350);
    }

    const totalPages = selectedDocs.reduce((acc, d) => acc + d.pages, 0);
    const totalChunks = Math.round(totalPages * 4.2);

    this.isKbBuilt = selectedDocs.length > 0;
    this.builtDocsCount = selectedDocs.length;
    this.builtPagesCount = totalPages;
    this.builtChunksCount = totalChunks;

    return {
      num_documents: selectedDocs.length,
      num_chunks: totalChunks,
      duplicate_chunks_removed: Math.floor(totalChunks * 0.05),
      embedding_model: 'sentence-transformers/all-MiniLM-L6-v2',
    };
  }

  public async sendQuestion(
    query: string,
    selectedDocIds: string[]
  ): Promise<ChatMessage> {
    await this.delay(1100);

    const selectedDocs = this.documents.filter((d) => selectedDocIds.includes(d.id));
    const docName1 = selectedDocs[0]?.name || 'Document.pdf';
    const docName2 = selectedDocs[1]?.name || docName1;

    const mockAnswer = `Based on the content extracted from **${docName1}**, here is the summary of relevant information regarding your query:

1. **Key Insight**: The document highlights key concepts and methodology related to "${query}" [1].
2. **Retrieval Analysis**: Combining BM25 keyword matching with FAISS vector search retrieves relevant passages across Page 1 and Page 4 [1], [2].
3. **Context Grounding**: FlashRank cross-encoder reranking prioritized passages with high semantic overlap to generate this grounded response [2].`;

    const mockRetrievalDetails: RetrievalDetails = {
      query: query,
      bm25_hits: [
        {
          doc_name: docName1,
          page: 1,
          score: 7.842,
          preview: `...Relevant passage extracted from ${docName1} addressing the query...`,
        },
        {
          doc_name: docName2,
          page: 4,
          score: 5.912,
          preview: `...Further details found in ${docName2} detailing technical specifications...`,
        },
      ],
      semantic_hits: [
        {
          doc_name: docName1,
          page: 1,
          similarity: 0.8841,
          preview: `...Relevant passage extracted from ${docName1} addressing the query...`,
        },
        {
          doc_name: docName2,
          page: 4,
          similarity: 0.8105,
          preview: `...Further details found in ${docName2} detailing technical specifications...`,
        },
      ],
      merge_stats: {
        bm25_count: 4,
        semantic_count: 4,
        unique_after_merge: 6,
        duplicates_removed: 2,
      },
      reranked: [
        {
          rank: 1,
          doc_name: docName1,
          page: 1,
          score: 0.925,
          method: 'FlashRank',
          preview: `...Relevant passage extracted from ${docName1} addressing the query...`,
        },
        {
          rank: 2,
          doc_name: docName2,
          page: 4,
          score: 0.841,
          method: 'FlashRank',
          preview: `...Further details found in ${docName2} detailing technical specifications...`,
        },
      ],
      context_stats: {
        retrieved_chunks: 8,
        after_reranking: 6,
        after_deduplication: 5,
        final_context: 4,
      },
      final_context_string: `[1] (Source: ${docName1}, Page 1)\nRelevant passage extracted from ${docName1} addressing the query.\n\n[2] (Source: ${docName2}, Page 4)\nFurther details found in ${docName2} detailing technical specifications.`,
    };

    return {
      id: `msg-${Date.now()}`,
      sender: 'assistant',
      text: mockAnswer,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      sources: [
        { index: 1, doc_name: docName1, page: 1, preview: `Extract from ${docName1}` },
        { index: 2, doc_name: docName2, page: 4, preview: `Extract from ${docName2}` },
      ],
      retrievalDetails: mockRetrievalDetails,
      groundedness: 0.92,
    };
  }

  private delay(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }
}

export const mockApi = new MockApiService();
