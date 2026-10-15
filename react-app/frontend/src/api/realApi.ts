import type {
  Document,
  KnowledgeBaseStatus,
  ChatMessage,
  BuildStats,
} from './types';

const BASE = '/api';

async function asJson(res: Response) {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch {
      /* body wasn't JSON -- keep statusText */
    }
    throw new Error(detail);
  }
  return res.json();
}

/**
 * Same public interface as MockApiService (api/mockApi.ts), so App.tsx only
 * needs its single import line changed to switch from the mock to this.
 *
 * One behavioral difference from the mock, by necessity: uploading a file
 * only extracts and registers it (so its real page count is known) -- it
 * does NOT index it. Nothing is searchable until buildKnowledgeBase() runs,
 * which chunks + embeds + indexes whichever documents are currently
 * selected. This matches what buildKnowledgeBase() already implies in the
 * mock (`documents.filter(d => d.selected)`), just made real.
 */
class RealApiService {
  public async getDocuments(): Promise<Document[]> {
    const res = await fetch(`${BASE}/documents`);
    return asJson(res);
  }

  public async uploadDocuments(files: File[]): Promise<Document[]> {
    const form = new FormData();
    for (const file of files) form.append('files', file);
    const res = await fetch(`${BASE}/documents/upload`, { method: 'POST', body: form });
    return asJson(res);
  }

  public async removeDocument(id: string): Promise<Document[]> {
    const res = await fetch(`${BASE}/documents/${encodeURIComponent(id)}`, { method: 'DELETE' });
    return asJson(res);
  }

  public async toggleDocumentSelection(id: string): Promise<Document[]> {
    const res = await fetch(`${BASE}/documents/${encodeURIComponent(id)}/toggle`, { method: 'POST' });
    return asJson(res);
  }

  public async toggleAllDocuments(select: boolean): Promise<Document[]> {
    const res = await fetch(`${BASE}/documents/toggle-all`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ select }),
    });
    return asJson(res);
  }

  public async getKbStatus(): Promise<KnowledgeBaseStatus> {
    const res = await fetch(`${BASE}/kb/status`);
    return asJson(res);
  }

  public async buildKnowledgeBase(onProgress?: (stage: string) => void): Promise<BuildStats> {
    // The real build runs as one blocking request (no per-stage server
    // events), so we show one honest stage message instead of the mock's
    // six simulated steps -- it would be misleading to fake progress we
    // can't actually observe without adding a streaming endpoint.
    if (onProgress) onProgress('Chunking, embedding, and indexing selected documents...');
    const res = await fetch(`${BASE}/kb/build`, { method: 'POST' });
    return asJson(res);
  }

  public async sendQuestion(query: string, selectedDocIds: string[]): Promise<ChatMessage> {
    const res = await fetch(`${BASE}/chat/ask`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, selected_doc_ids: selectedDocIds }),
    });
    return asJson(res);
  }
}

export const realApi = new RealApiService();
