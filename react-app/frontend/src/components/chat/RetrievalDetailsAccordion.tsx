import React, { useState } from 'react';
import type { RetrievalDetails } from '../../api/types';
import { ChevronDown, ChevronRight, Search, Zap, Layers, Sparkles, Filter, Code } from 'lucide-react';

interface RetrievalDetailsAccordionProps {
  details: RetrievalDetails;
}

export const RetrievalDetailsAccordion: React.FC<RetrievalDetailsAccordionProps> = ({ details }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [showFinalContext, setShowFinalContext] = useState(false);

  if (!details) return null;

  return (
    <div className="border border-zinc-800/80 rounded-xl bg-zinc-900/40 overflow-hidden text-xs space-y-0">
      {/* Accordion Trigger */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full px-4 py-2.5 flex items-center justify-between text-zinc-300 hover:text-white bg-zinc-900/80 hover:bg-zinc-800/80 transition-colors cursor-pointer"
      >
        <span className="flex items-center gap-2 font-medium">
          <Search className="w-3.5 h-3.5 text-indigo-400" />
          <span>View Retrieval Details</span>
        </span>
        {isOpen ? (
          <ChevronDown className="w-4 h-4 text-zinc-400" />
        ) : (
          <ChevronRight className="w-4 h-4 text-zinc-400" />
        )}
      </button>

      {/* Accordion Body */}
      {isOpen && (
        <div className="p-4 space-y-4 border-t border-zinc-800/80 bg-zinc-950/60 font-sans">
          {/* Query */}
          <div className="space-y-1">
            <div className="text-[11px] font-semibold text-zinc-400 uppercase tracking-wider">
              Query
            </div>
            <div className="bg-zinc-900 border border-zinc-800 p-2.5 rounded-lg text-zinc-200 font-mono text-[11px]">
              {details.query}
            </div>
          </div>

          {/* BM25 Retrieval */}
          <div className="space-y-1.5">
            <div className="flex items-center justify-between text-[11px] font-semibold text-zinc-400 uppercase tracking-wider">
              <span className="flex items-center gap-1.5">
                <Filter className="w-3 h-3 text-amber-400" />
                BM25 Keyword Retrieval
              </span>
              <span className="font-mono text-zinc-500">{details.bm25_hits.length} hits</span>
            </div>
            <div className="space-y-1">
              {details.bm25_hits.map((hit, idx) => (
                <div key={idx} className="bg-zinc-900/80 border border-zinc-800/60 p-2 rounded-lg flex items-center justify-between text-[11px]">
                  <span className="text-zinc-300 font-medium truncate max-w-[240px]">
                    {hit.doc_name} · Page {hit.page}
                  </span>
                  <span className="font-mono text-amber-400">Score: {hit.score}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Semantic Retrieval */}
          <div className="space-y-1.5">
            <div className="flex items-center justify-between text-[11px] font-semibold text-zinc-400 uppercase tracking-wider">
              <span className="flex items-center gap-1.5">
                <Zap className="w-3 h-3 text-indigo-400" />
                Semantic Vector Retrieval
              </span>
              <span className="font-mono text-zinc-500">{details.semantic_hits.length} hits</span>
            </div>
            <div className="space-y-1">
              {details.semantic_hits.map((hit, idx) => (
                <div key={idx} className="bg-zinc-900/80 border border-zinc-800/60 p-2 rounded-lg flex items-center justify-between text-[11px]">
                  <span className="text-zinc-300 font-medium truncate max-w-[240px]">
                    {hit.doc_name} · Page {hit.page}
                  </span>
                  <span className="font-mono text-indigo-400">Similarity: {hit.similarity}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Hybrid Merging Stats */}
          <div className="space-y-1.5">
            <div className="text-[11px] font-semibold text-zinc-400 uppercase tracking-wider flex items-center gap-1.5">
              <Layers className="w-3 h-3 text-cyan-400" />
              Hybrid Retrieval Synthesis
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-center font-mono">
              <div className="bg-zinc-900 border border-zinc-800 p-2 rounded-lg">
                <div className="text-zinc-200 font-bold">{details.merge_stats.bm25_count}</div>
                <div className="text-[10px] text-zinc-500 font-sans">BM25</div>
              </div>
              <div className="bg-zinc-900 border border-zinc-800 p-2 rounded-lg">
                <div className="text-zinc-200 font-bold">{details.merge_stats.semantic_count}</div>
                <div className="text-[10px] text-zinc-500 font-sans">Semantic</div>
              </div>
              <div className="bg-zinc-900 border border-zinc-800 p-2 rounded-lg">
                <div className="text-zinc-200 font-bold">{details.merge_stats.unique_after_merge}</div>
                <div className="text-[10px] text-zinc-500 font-sans">Merged</div>
              </div>
              <div className="bg-zinc-900 border border-zinc-800 p-2 rounded-lg">
                <div className="text-zinc-200 font-bold">{details.merge_stats.duplicates_removed}</div>
                <div className="text-[10px] text-zinc-500 font-sans">Duplicates</div>
              </div>
            </div>
          </div>

          {/* Reranking */}
          <div className="space-y-1.5">
            <div className="text-[11px] font-semibold text-zinc-400 uppercase tracking-wider flex items-center gap-1.5">
              <Sparkles className="w-3 h-3 text-emerald-400" />
              FlashRank Cross-Encoder Reranking
            </div>
            <div className="space-y-1">
              {details.reranked.map((hit, idx) => (
                <div key={idx} className="bg-zinc-900/80 border border-zinc-800/60 p-2 rounded-lg flex items-center justify-between text-[11px]">
                  <span className="text-zinc-300 font-medium">
                    #{hit.rank} {hit.doc_name} · Page {hit.page}
                  </span>
                  <span className="font-mono text-emerald-400">Score: {hit.score}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Context Optimization */}
          <div className="space-y-1.5 pt-1 border-t border-zinc-800/60">
            <div className="flex items-center justify-between text-[11px] font-semibold text-zinc-400 uppercase tracking-wider">
              <span>Context Optimization Window</span>
              <button
                onClick={() => setShowFinalContext(!showFinalContext)}
                className="text-[10px] text-indigo-400 hover:underline flex items-center gap-1 cursor-pointer"
              >
                <Code className="w-3 h-3" />
                <span>{showFinalContext ? 'Hide Context' : 'View Context String'}</span>
              </button>
            </div>
            <div className="bg-zinc-900 border border-zinc-800 p-2.5 rounded-lg text-zinc-300 font-mono text-[11px] flex justify-between">
              <span>Retrieved: {details.context_stats.retrieved_chunks}</span>
              <span>Reranked: {details.context_stats.after_reranking}</span>
              <span className="text-indigo-400 font-bold">Final Context: {details.context_stats.final_context} chunks</span>
            </div>

            {showFinalContext && details.final_context_string && (
              <pre className="p-3 bg-zinc-950 border border-zinc-800 rounded-lg text-[10px] text-zinc-400 font-mono overflow-x-auto whitespace-pre-wrap">
                {details.final_context_string}
              </pre>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
