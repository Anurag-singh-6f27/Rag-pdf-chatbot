import React from 'react';
import {
  FileText,
  MessageSquare,
  Upload,
  Cpu,
  Sparkles,
} from 'lucide-react';

interface LeftNavProps {
  activeTab: 'documents' | 'chat';
  setActiveTab: (tab: 'documents' | 'chat') => void;
  onOpenUploadModal: () => void;
  selectedCount: number;
}

export const LeftNav: React.FC<LeftNavProps> = ({
  activeTab,
  setActiveTab,
  onOpenUploadModal,
  selectedCount,
}) => {
  return (
    <aside className="w-[240px] bg-zinc-900 border-r border-zinc-800/80 flex flex-col justify-between select-none shrink-0 h-full">
      {/* Top Header & Branding */}
      <div className="p-4 space-y-6">
        <div className="flex items-center gap-3 px-2 py-1">
          <div className="w-8 h-8 rounded-lg bg-indigo-600/20 border border-indigo-500/30 flex items-center justify-center text-indigo-400">
            <Sparkles className="w-5 h-5" />
          </div>
          <div>
            <h1 className="font-semibold text-sm tracking-wide text-zinc-100 flex items-center gap-1.5">
              RAG AI
            </h1>
            <p className="text-[11px] text-zinc-400 font-mono tracking-tight">
              Advanced Document QA
            </p>
          </div>
        </div>

        {/* Primary Action Button */}
        <button
          onClick={onOpenUploadModal}
          className="w-full bg-indigo-600 hover:bg-indigo-500 active:bg-indigo-700 text-white font-medium text-xs py-2.5 px-3 rounded-lg transition-all flex items-center justify-center gap-2 shadow-sm shadow-indigo-950 cursor-pointer"
        >
          <Upload className="w-4 h-4" />
          <span>Upload Documents</span>
        </button>

        {/* Navigation Items */}
        <nav className="space-y-1 pt-2">
          <button
            onClick={() => setActiveTab('documents')}
            className={`w-full flex items-center justify-between px-3 py-2 rounded-lg text-xs font-medium transition-colors cursor-pointer ${
              activeTab === 'documents'
                ? 'bg-zinc-800 text-indigo-400 border-l-2 border-indigo-500 font-semibold'
                : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/50'
            }`}
          >
            <div className="flex items-center gap-2.5">
              <FileText className="w-4 h-4" />
              <span>My Documents</span>
            </div>
            {selectedCount > 0 && (
              <span className="text-[10px] bg-indigo-950 text-indigo-300 border border-indigo-800/50 px-1.5 py-0.5 rounded-full font-mono">
                {selectedCount}
              </span>
            )}
          </button>

          <button
            onClick={() => setActiveTab('chat')}
            className={`w-full flex items-center justify-between px-3 py-2 rounded-lg text-xs font-medium transition-colors cursor-pointer ${
              activeTab === 'chat'
                ? 'bg-zinc-800 text-indigo-400 border-l-2 border-indigo-500 font-semibold'
                : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/50'
            }`}
          >
            <div className="flex items-center gap-2.5">
              <MessageSquare className="w-4 h-4" />
              <span>Chat</span>
            </div>
          </button>
        </nav>
      </div>

      {/* System Status Section */}
      <div className="p-4 border-t border-zinc-800/80 bg-zinc-900/50 space-y-3">
        <div className="flex items-center justify-between">
          <span className="text-[11px] font-semibold text-zinc-400 uppercase tracking-wider flex items-center gap-1.5">
            <Cpu className="w-3.5 h-3.5 text-zinc-500" />
            System Status
          </span>
          <span className="text-[10px] text-emerald-400 bg-emerald-950/60 border border-emerald-800/50 px-2 py-0.5 rounded-full flex items-center gap-1.5 font-medium">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
            Frontend Demo
          </span>
        </div>

        <div className="text-[11px] text-zinc-400 space-y-1.5 pt-1 font-mono">
          <div className="flex justify-between items-center text-zinc-400">
            <span>Retrieval</span>
            <span className="text-zinc-300">BM25 + FAISS</span>
          </div>
          <div className="flex justify-between items-center text-zinc-400">
            <span>Reranker</span>
            <span className="text-zinc-300">FlashRank</span>
          </div>
          <div className="flex justify-between items-center text-zinc-400">
            <span>Generator</span>
            <span className="text-zinc-300">Gemini 2.5</span>
          </div>
        </div>
      </div>
    </aside>
  );
};
