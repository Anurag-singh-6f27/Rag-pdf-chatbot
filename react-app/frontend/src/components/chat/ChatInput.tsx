import React, { useState, useRef, useEffect } from 'react';
import { ArrowUp, CornerDownLeft } from 'lucide-react';

interface ChatInputProps {
  onSendMessage: (query: string) => void;
  disabled: boolean;
  isLoading: boolean;
  selectedCount: number;
}

export const ChatInput: React.FC<ChatInputProps> = ({
  onSendMessage,
  disabled,
  isLoading,
  selectedCount,
}) => {
  const [query, setQuery] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const MAX_CHARS = 500;

  // Auto-resize textarea height
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 140)}px`;
    }
  }, [query]);

  const handleSubmit = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    const trimmed = query.trim();
    if (!trimmed || disabled || isLoading || selectedCount === 0) return;
    onSendMessage(trimmed);
    setQuery('');
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const isOverLimit = query.length > MAX_CHARS;

  return (
    <div className="p-4 bg-zinc-900/80 border-t border-zinc-800/80 shrink-0">
      <form onSubmit={handleSubmit} className="relative space-y-2">
        <div className="relative bg-zinc-950 border border-zinc-800 focus-within:border-indigo-500/60 rounded-xl p-2 transition-all shadow-inner">
          <textarea
            ref={textareaRef}
            rows={1}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={
              selectedCount === 0
                ? 'Select a document to ask a question...'
                : 'Ask a question about your documents... (Shift+Enter for newline)'
            }
            disabled={disabled || isLoading || selectedCount === 0}
            maxLength={MAX_CHARS}
            className="w-full bg-transparent text-xs text-zinc-100 placeholder-zinc-500 resize-none focus:outline-none pr-10 pl-2 pt-1.5 pb-1 max-h-32 disabled:opacity-50 disabled:cursor-not-allowed font-sans"
          />

          <button
            type="submit"
            disabled={!query.trim() || disabled || isLoading || selectedCount === 0 || isOverLimit}
            className="absolute right-2.5 bottom-2.5 w-7 h-7 rounded-lg bg-indigo-600 hover:bg-indigo-500 active:bg-indigo-700 text-white disabled:opacity-40 disabled:cursor-not-allowed transition-all flex items-center justify-center shadow-sm cursor-pointer"
          >
            <ArrowUp className="w-4 h-4 stroke-[2.5]" />
          </button>
        </div>

        {/* Footer info: Character Counter & Shift+Enter tip */}
        <div className="flex items-center justify-between px-1 text-[11px] text-zinc-500 font-mono">
          <span className="flex items-center gap-1">
            <CornerDownLeft className="w-3 h-3 text-zinc-600" />
            <span>Enter to send</span>
          </span>
          <span className={isOverLimit ? 'text-red-400 font-bold' : 'text-zinc-500'}>
            {query.length} / {MAX_CHARS} characters
          </span>
        </div>
      </form>
    </div>
  );
};
