import React from 'react';
import type { Source } from '../../api/types';
import { FileText, ExternalLink } from 'lucide-react';

interface SourcesListProps {
  sources: Source[];
}

export const SourcesList: React.FC<SourcesListProps> = ({ sources }) => {
  if (!sources || sources.length === 0) return null;

  return (
    <div className="space-y-2 pt-2">
      <div className="text-[11px] font-semibold text-zinc-400 uppercase tracking-wider flex items-center gap-1.5">
        <FileText className="w-3.5 h-3.5 text-indigo-400" />
        <span>Sources ({sources.length})</span>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
        {sources.map((src, idx) => (
          <div
            key={idx}
            className="group flex items-center justify-between p-2.5 bg-zinc-900/80 hover:bg-zinc-800/80 border border-zinc-800/80 hover:border-zinc-700/80 rounded-lg text-xs transition-all cursor-pointer select-none shadow-sm"
          >
            <div className="flex items-center gap-2.5 min-w-0">
              <span className="w-5 h-5 rounded-md bg-indigo-950/80 border border-indigo-800/60 text-indigo-300 font-mono text-[10px] font-bold flex items-center justify-center shrink-0">
                [{src.index}]
              </span>
              <div className="min-w-0">
                <p className="text-zinc-200 font-medium truncate group-hover:text-white transition-colors">
                  {src.doc_name}
                </p>
                <p className="text-[10px] text-zinc-400 font-mono">
                  Page {src.page}
                </p>
              </div>
            </div>
            <ExternalLink className="w-3.5 h-3.5 text-zinc-600 group-hover:text-indigo-400 transition-colors shrink-0 ml-2" />
          </div>
        ))}
      </div>
    </div>
  );
};
