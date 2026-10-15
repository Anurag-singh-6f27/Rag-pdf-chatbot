import React from 'react';
import type { ChatMessage } from '../../api/types';
import { ChatMessageItem } from './ChatMessageItem';
import { MessageSquare, FileSearch, Sparkles, Loader2, Upload } from 'lucide-react';

interface ChatMessageListProps {
  messages: ChatMessage[];
  selectedCount: number;
  totalDocsCount: number;
  isLoading: boolean;
}

export const ChatMessageList: React.FC<ChatMessageListProps> = ({
  messages,
  selectedCount,
  totalDocsCount,
  isLoading,
}) => {
  // Empty state when no documents exist at all in the library
  if (totalDocsCount === 0) {
    return (
      <div className="h-full flex flex-col items-center justify-center p-8 text-center text-zinc-500 space-y-4 select-none">
        <div className="w-16 h-16 rounded-2xl bg-zinc-900 border border-zinc-800 flex items-center justify-center text-zinc-400 shadow-inner">
          <Upload className="w-7 h-7 text-zinc-500" />
        </div>
        <div className="max-w-xs space-y-1">
          <h3 className="text-sm font-semibold text-zinc-200">
            No documents yet
          </h3>
          <p className="text-xs text-zinc-400 leading-relaxed">
            Upload PDF files to build your knowledge base and start chatting.
          </p>
        </div>
      </div>
    );
  }

  // Empty state when documents exist in library but 0 are selected
  if (selectedCount === 0) {
    return (
      <div className="h-full flex flex-col items-center justify-center p-8 text-center text-zinc-500 space-y-4 select-none">
        <div className="w-16 h-16 rounded-2xl bg-zinc-900 border border-zinc-800 flex items-center justify-center text-zinc-400 shadow-inner">
          <FileSearch className="w-7 h-7 text-zinc-500" />
        </div>
        <div className="max-w-xs space-y-1">
          <h3 className="text-sm font-semibold text-zinc-200">
            Select a document first
          </h3>
          <p className="text-xs text-zinc-400 leading-relaxed">
            Choose one or more documents from the library to start asking questions.
          </p>
        </div>
      </div>
    );
  }

  // Empty state when documents are selected but no conversation started
  if (messages.length === 0 && !isLoading) {
    return (
      <div className="h-full flex flex-col items-center justify-center p-8 text-center text-zinc-500 space-y-4 select-none">
        <div className="w-16 h-16 rounded-2xl bg-indigo-950/40 border border-indigo-800/40 flex items-center justify-center text-indigo-400 shadow-inner">
          <MessageSquare className="w-8 h-8" />
        </div>
        <div className="max-w-xs space-y-1">
          <h3 className="text-sm font-semibold text-zinc-200">
            Start a conversation
          </h3>
          <p className="text-xs text-zinc-400 leading-relaxed">
            Ask questions about your selected documents and get grounded answers with exact source citations.
          </p>
          <span className="inline-block mt-2 text-[11px] bg-indigo-950 text-indigo-300 border border-indigo-800/50 px-2.5 py-0.5 rounded-full font-mono">
            {selectedCount} document{selectedCount === 1 ? '' : 's'} selected
          </span>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto p-6 space-y-6">
      {messages.map((msg) => (
        <ChatMessageItem key={msg.id} message={msg} />
      ))}

      {/* RAG Loading Animation */}
      {isLoading && (
        <div className="flex flex-col items-start space-y-2 max-w-lg mr-auto">
          <div className="flex items-center gap-2 text-[11px] font-semibold text-indigo-400 uppercase tracking-wider">
            <Sparkles className="w-3.5 h-3.5" />
            <span>ASSISTANT</span>
          </div>

          <div className="bg-zinc-900/90 border border-zinc-800/90 p-4 rounded-2xl rounded-tl-sm text-xs space-y-2 text-zinc-300 w-full shadow-md">
            <div className="flex items-center gap-2.5 text-indigo-300 font-medium">
              <Loader2 className="w-4 h-4 animate-spin text-indigo-400" />
              <span>Generating grounded answer...</span>
            </div>
            <div className="space-y-1 text-[11px] text-zinc-400 font-mono pl-6 border-l border-zinc-800">
              <p className="animate-pulse">Searching documents (BM25 + FAISS)...</p>
              <p className="animate-pulse delay-150">Retrieving relevant passages & reranking...</p>
              <p className="animate-pulse delay-300">Formulating Gemini answer with citations...</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
