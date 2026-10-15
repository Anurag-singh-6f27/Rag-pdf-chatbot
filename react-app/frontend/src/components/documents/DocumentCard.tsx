import React from 'react';
import type { Document } from '../../api/types';
import { FileText, Trash2, CheckSquare, Square } from 'lucide-react';

interface DocumentCardProps {
  document: Document;
  onToggleSelect: (id: string) => void;
  onRemove: (id: string) => void;
}

export const DocumentCard: React.FC<DocumentCardProps> = ({
  document,
  onToggleSelect,
  onRemove,
}) => {
  return (
    <div
      onClick={() => onToggleSelect(document.id)}
      className={`group relative p-3 rounded-xl border transition-all cursor-pointer select-none ${
        document.selected
          ? 'bg-zinc-800/90 border-indigo-500/60 shadow-md shadow-indigo-950/20'
          : 'bg-zinc-900/60 border-zinc-800/80 hover:bg-zinc-800/50 hover:border-zinc-700/80'
      }`}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-start gap-3 min-w-0">
          {/* Checkbox */}
          <button
            onClick={(e) => {
              e.stopPropagation();
              onToggleSelect(document.id);
            }}
            className="mt-0.5 text-zinc-400 hover:text-indigo-400 focus:outline-none transition-colors"
          >
            {document.selected ? (
              <CheckSquare className="w-4 h-4 text-indigo-400" />
            ) : (
              <Square className="w-4 h-4 text-zinc-600 group-hover:text-zinc-400" />
            )}
          </button>

          {/* PDF File Icon */}
          <div className="w-8 h-8 rounded-lg bg-zinc-800 border border-zinc-700/60 flex items-center justify-center text-indigo-400 shrink-0">
            <FileText className="w-4 h-4" />
          </div>

          {/* Details */}
          <div className="min-w-0 space-y-0.5">
            <h4 className="text-xs font-semibold text-zinc-200 truncate group-hover:text-white transition-colors">
              {document.name}
            </h4>
            <div className="flex items-center gap-2 text-[11px] text-zinc-400 font-mono">
              <span>{document.pages} pages</span>
              <span>•</span>
              <span>{document.size}</span>
            </div>
          </div>
        </div>

        {/* Status Indicator & Remove Action */}
        <div className="flex items-center gap-2 shrink-0">
          <span className="inline-flex items-center gap-1 text-[10px] text-emerald-400 bg-emerald-950/60 border border-emerald-800/50 px-2 py-0.5 rounded-full font-medium">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
            Ready
          </span>

          <button
            onClick={(e) => {
              e.stopPropagation();
              onRemove(document.id);
            }}
            className="opacity-0 group-hover:opacity-100 text-zinc-500 hover:text-red-400 p-1 rounded hover:bg-zinc-800 transition-all"
            title="Remove document"
          >
            <Trash2 className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>
    </div>
  );
};
