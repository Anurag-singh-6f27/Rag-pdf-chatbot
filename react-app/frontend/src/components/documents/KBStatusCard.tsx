import React from 'react';
import type { KnowledgeBaseStatus } from '../../api/types';
import { Database, CheckCircle2, Loader2, Play, AlertCircle } from 'lucide-react';

interface KBStatusCardProps {
  status: KnowledgeBaseStatus;
  uploadedCount: number;
  isBuilding: boolean;
  buildingStage: string;
  onBuildKb: () => void;
}

export const KBStatusCard: React.FC<KBStatusCardProps> = ({
  status,
  uploadedCount,
  isBuilding,
  buildingStage,
  onBuildKb,
}) => {
  return (
    <div className="bg-zinc-900/80 border border-zinc-800/80 rounded-xl p-3.5 space-y-3 shrink-0">
      {/* Header */}
      <div className="flex items-center justify-between text-xs font-semibold text-zinc-300">
        <span className="flex items-center gap-1.5">
          <Database className="w-3.5 h-3.5 text-indigo-400" />
          Knowledge Base Status
        </span>
        {status.isReady ? (
          <span className="text-[10px] text-emerald-400 bg-emerald-950/60 border border-emerald-800/50 px-2 py-0.5 rounded-full font-mono flex items-center gap-1">
            <CheckCircle2 className="w-3 h-3" /> Ready
          </span>
        ) : (
          <span className="text-[10px] text-zinc-400 bg-zinc-800/80 border border-zinc-700/60 px-2 py-0.5 rounded-full font-mono flex items-center gap-1">
            <AlertCircle className="w-3 h-3 text-zinc-500" /> Not Built
          </span>
        )}
      </div>

      {/* Metrics Grid */}
      <div className="grid grid-cols-3 gap-2 text-center bg-zinc-950/60 border border-zinc-800/60 rounded-lg p-2 font-mono">
        <div>
          <div className="text-xs font-bold text-zinc-100">{status.documentsCount}</div>
          <div className="text-[10px] text-zinc-400 font-sans">Indexed</div>
        </div>
        <div>
          <div className="text-xs font-bold text-zinc-100">{status.pagesCount}</div>
          <div className="text-[10px] text-zinc-400 font-sans">Pages</div>
        </div>
        <div>
          <div className="text-xs font-bold text-zinc-100">{status.chunksCount}</div>
          <div className="text-[10px] text-zinc-400 font-sans">Chunks</div>
        </div>
      </div>

      {/* Component Status Grid */}
      <div className="grid grid-cols-2 gap-1.5 text-[11px] font-mono pt-0.5">
        <div className="flex items-center justify-between px-2 py-1 bg-zinc-800/40 rounded border border-zinc-800/60">
          <span className="text-zinc-400">FAISS</span>
          <span className={status.isReady ? "text-emerald-400" : "text-zinc-500"}>
            {status.isReady ? "● Ready" : "○ Idle"}
          </span>
        </div>
        <div className="flex items-center justify-between px-2 py-1 bg-zinc-800/40 rounded border border-zinc-800/60">
          <span className="text-zinc-400">BM25</span>
          <span className={status.isReady ? "text-emerald-400" : "text-zinc-500"}>
            {status.isReady ? "● Ready" : "○ Idle"}
          </span>
        </div>
        <div className="flex items-center justify-between px-2 py-1 bg-zinc-800/40 rounded border border-zinc-800/60">
          <span className="text-zinc-400">FlashRank</span>
          <span className={status.isReady ? "text-emerald-400" : "text-zinc-500"}>
            {status.isReady ? "● Ready" : "○ Idle"}
          </span>
        </div>
        <div className="flex items-center justify-between px-2 py-1 bg-zinc-800/40 rounded border border-zinc-800/60">
          <span className="text-zinc-400">Gemini</span>
          <span className="text-emerald-400">● Ready</span>
        </div>
      </div>

      {/* Build Action Button */}
      {isBuilding ? (
        <div className="bg-indigo-950/40 border border-indigo-800/50 rounded-lg p-2.5 text-center space-y-1.5">
          <div className="flex items-center justify-center gap-2 text-xs text-indigo-300 font-medium">
            <Loader2 className="w-4 h-4 animate-spin text-indigo-400" />
            <span>Building Knowledge Base...</span>
          </div>
          <p className="text-[11px] text-indigo-300/80 font-mono animate-pulse">
            {buildingStage}
          </p>
        </div>
      ) : (
        <button
          onClick={onBuildKb}
          disabled={uploadedCount === 0}
          className="w-full bg-indigo-600 hover:bg-indigo-500 active:bg-indigo-700 text-white font-medium text-xs py-2 px-3 rounded-lg transition-all flex items-center justify-center gap-2 shadow-sm disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer"
        >
          <Play className="w-3.5 h-3.5 fill-current" />
          <span>Build Knowledge Base</span>
        </button>
      )}
    </div>
  );
};
