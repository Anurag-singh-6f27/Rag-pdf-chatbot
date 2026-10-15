import React, { useState } from 'react';
import type { Document, KnowledgeBaseStatus } from '../../api/types';
import { DocumentCard } from './DocumentCard';
import { KBStatusCard } from './KBStatusCard';
import { Search, FileText, CheckSquare, Square, Upload } from 'lucide-react';

interface DocumentLibraryProps {
  documents: Document[];
  kbStatus: KnowledgeBaseStatus;
  isBuildingKb: boolean;
  buildingStage: string;
  onToggleSelect: (id: string) => void;
  onToggleAll: (select: boolean) => void;
  onRemoveDocument: (id: string) => void;
  onBuildKb: () => void;
  onOpenUploadModal: () => void;
}

export const DocumentLibrary: React.FC<DocumentLibraryProps> = ({
  documents,
  kbStatus,
  isBuildingKb,
  buildingStage,
  onToggleSelect,
  onToggleAll,
  onRemoveDocument,
  onBuildKb,
  onOpenUploadModal,
}) => {
  const [searchQuery, setSearchQuery] = useState('');

  const filteredDocs = documents.filter((doc) =>
    doc.name.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const selectedCount = documents.filter((d) => d.selected).length;
  const allSelected = documents.length > 0 && selectedCount === documents.length;

  return (
    <div className="w-[340px] bg-zinc-900/60 border-r border-zinc-800/80 flex flex-col justify-between shrink-0 h-full overflow-hidden select-none">
      {/* Header & Search */}
      <div className="p-4 space-y-3 border-b border-zinc-800/80 shrink-0">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <h2 className="text-sm font-semibold text-zinc-100">Documents</h2>
            <span className="text-[11px] bg-zinc-800 text-zinc-400 font-mono px-2 py-0.5 rounded-full">
              {documents.length} uploaded
            </span>
          </div>

          {documents.length > 0 && (
            <button
              onClick={() => onToggleAll(!allSelected)}
              className="text-[11px] text-zinc-400 hover:text-indigo-400 flex items-center gap-1 transition-colors cursor-pointer"
            >
              {allSelected ? (
                <>
                  <CheckSquare className="w-3.5 h-3.5 text-indigo-400" />
                  <span>Deselect All</span>
                </>
              ) : (
                <>
                  <Square className="w-3.5 h-3.5" />
                  <span>Select All</span>
                </>
              )}
            </button>
          )}
        </div>

        {/* Search Input Bar */}
        <div className="relative">
          <Search className="w-3.5 h-3.5 text-zinc-500 absolute left-3 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            placeholder="Search documents..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full bg-zinc-950/80 border border-zinc-800 rounded-lg pl-8 pr-3 py-1.5 text-xs text-zinc-200 placeholder-zinc-500 focus:outline-none focus:border-indigo-500/60 transition-all"
          />
        </div>
      </div>

      {/* Scrollable Document List or Empty State */}
      <div className="flex-1 overflow-y-auto p-3 space-y-2">
        {documents.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center p-6 text-center text-zinc-500 space-y-3">
            <div className="w-12 h-12 rounded-full bg-zinc-800/60 border border-zinc-700/50 flex items-center justify-center text-zinc-400">
              <FileText className="w-6 h-6" />
            </div>
            <div>
              <h3 className="text-xs font-semibold text-zinc-300">
                No documents yet
              </h3>
              <p className="text-[11px] text-zinc-400 mt-1 max-w-[200px] leading-relaxed">
                Upload PDF files to build your knowledge base.
              </p>
            </div>
            <button
              onClick={onOpenUploadModal}
              className="mt-2 text-xs bg-indigo-600/20 text-indigo-300 border border-indigo-500/30 px-3 py-1.5 rounded-lg hover:bg-indigo-600/30 transition-colors font-medium flex items-center gap-1.5 cursor-pointer"
            >
              <Upload className="w-3.5 h-3.5" />
              <span>Upload PDF</span>
            </button>
          </div>
        ) : filteredDocs.length === 0 ? (
          <div className="p-4 text-center text-xs text-zinc-500">
            No documents matching "{searchQuery}"
          </div>
        ) : (
          filteredDocs.map((doc) => (
            <DocumentCard
              key={doc.id}
              document={doc}
              onToggleSelect={onToggleSelect}
              onRemove={onRemoveDocument}
            />
          ))
        )}
      </div>

      {/* Knowledge Base Status Card Footer */}
      <div className="p-3 border-t border-zinc-800/80 bg-zinc-900/40 shrink-0">
        <KBStatusCard
          status={kbStatus}
          uploadedCount={documents.length}
          isBuilding={isBuildingKb}
          buildingStage={buildingStage}
          onBuildKb={onBuildKb}
        />
      </div>
    </div>
  );
};
