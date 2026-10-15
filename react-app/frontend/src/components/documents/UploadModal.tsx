import React, { useState, useRef } from 'react';
import { Upload, X, FileText, CheckCircle2, AlertCircle } from 'lucide-react';

interface UploadModalProps {
  isOpen: boolean;
  onClose: () => void;
  onUpload: (files: File[]) => void;
}

export const UploadModal: React.FC<UploadModalProps> = ({
  isOpen,
  onClose,
  onUpload,
}) => {
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [dragActive, setDragActive] = useState<boolean>(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  if (!isOpen) return null;

  const handleFiles = (filesList: FileList | null) => {
    if (!filesList) return;
    setErrorMessage(null);
    const pdfs: File[] = [];

    for (let i = 0; i < filesList.length; i++) {
      const file = filesList[i];
      if (file.type === 'application/pdf' || file.name.endsWith('.pdf')) {
        pdfs.push(file);
      } else {
        setErrorMessage('Only PDF files are supported.');
      }
    }

    if (pdfs.length > 0) {
      setSelectedFiles((prev) => [...prev, ...pdfs]);
    }
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    handleFiles(e.dataTransfer.files);
  };

  const removeFile = (index: number) => {
    setSelectedFiles((prev) => prev.filter((_, i) => i !== index));
  };

  const handleSubmit = () => {
    if (selectedFiles.length === 0) return;
    onUpload(selectedFiles);
    setSelectedFiles([]);
    onClose();
  };

  return (
    <div className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4">
      <div className="bg-zinc-900 border border-zinc-800 rounded-xl w-full max-w-lg overflow-hidden shadow-2xl space-y-0">
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-zinc-800">
          <div className="flex items-center gap-2 text-zinc-100 font-semibold text-sm">
            <Upload className="w-4 h-4 text-indigo-400" />
            <span>Upload Documents</span>
          </div>
          <button
            onClick={onClose}
            className="text-zinc-400 hover:text-zinc-200 p-1 rounded-lg hover:bg-zinc-800 transition-colors cursor-pointer"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Drop Area */}
        <div className="p-5 space-y-4">
          <div
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className={`border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-all ${
              dragActive
                ? 'border-indigo-500 bg-indigo-500/10 scale-[0.99]'
                : 'border-zinc-700/80 hover:border-zinc-600 bg-zinc-950/40'
            }`}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf"
              multiple
              className="hidden"
              onChange={(e) => handleFiles(e.target.files)}
            />
            <div className="w-10 h-10 rounded-full bg-zinc-800/80 border border-zinc-700/60 flex items-center justify-center mx-auto mb-3 text-zinc-300">
              <Upload className="w-5 h-5" />
            </div>
            <p className="text-xs font-medium text-zinc-200">
              Drop PDF files here or <span className="text-indigo-400 hover:underline">browse</span>
            </p>
            <p className="text-[11px] text-zinc-400 mt-1">PDF documents only</p>
          </div>

          {errorMessage && (
            <div className="flex items-center gap-2 text-xs text-amber-400 bg-amber-950/40 border border-amber-800/50 p-2.5 rounded-lg">
              <AlertCircle className="w-4 h-4 shrink-0" />
              <span>{errorMessage}</span>
            </div>
          )}

          {/* Selected File List */}
          {selectedFiles.length > 0 && (
            <div className="space-y-2 max-h-48 overflow-y-auto pr-1">
              <span className="text-[11px] font-semibold text-zinc-400 uppercase tracking-wider">
                Selected Files ({selectedFiles.length})
              </span>
              {selectedFiles.map((file, idx) => (
                <div
                  key={idx}
                  className="flex items-center justify-between p-2.5 bg-zinc-800/60 border border-zinc-700/50 rounded-lg text-xs"
                >
                  <div className="flex items-center gap-2.5 min-w-0 pr-2">
                    <FileText className="w-4 h-4 text-indigo-400 shrink-0" />
                    <span className="truncate text-zinc-200 font-medium">
                      {file.name}
                    </span>
                    <span className="text-[10px] text-zinc-400 shrink-0 font-mono">
                      ({(file.size / (1024 * 1024)).toFixed(2)} MB)
                    </span>
                  </div>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      removeFile(idx);
                    }}
                    className="text-zinc-400 hover:text-red-400 p-1 hover:bg-zinc-700/50 rounded transition-colors cursor-pointer"
                  >
                    <X className="w-3.5 h-3.5" />
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-2 px-5 py-3.5 bg-zinc-950/60 border-t border-zinc-800">
          <button
            onClick={onClose}
            className="px-3.5 py-1.5 rounded-lg text-xs text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800 transition-colors cursor-pointer"
          >
            Cancel
          </button>
          <button
            onClick={handleSubmit}
            disabled={selectedFiles.length === 0}
            className="px-4 py-1.5 rounded-lg text-xs font-medium bg-indigo-600 hover:bg-indigo-500 text-white disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center gap-1.5 shadow-sm cursor-pointer"
          >
            <CheckCircle2 className="w-3.5 h-3.5" />
            <span>Add {selectedFiles.length} File(s)</span>
          </button>
        </div>
      </div>
    </div>
  );
};
