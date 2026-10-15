import React from 'react';
import { MessageSquare, Trash2, PlusCircle, FileCheck } from 'lucide-react';

interface ChatHeaderProps {
  selectedCount: number;
  totalDocsCount: number;
  onNewChat: () => void;
  onClearChat: () => void;
  hasMessages: boolean;
}

export const ChatHeader: React.FC<ChatHeaderProps> = ({
  selectedCount,
  onNewChat,
  onClearChat,
  hasMessages,
}) => {
  return (
    <div className="px-6 py-3.5 bg-zinc-900/80 border-b border-zinc-800/80 flex items-center justify-between shrink-0 select-none">
      <div className="flex items-center gap-3">
        <div className="w-8 h-8 rounded-lg bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center text-indigo-400">
          <MessageSquare className="w-4 h-4" />
        </div>
        <div>
          <h2 className="text-xs font-semibold text-zinc-100 flex items-center gap-2">
            <span>Chat</span>
          </h2>
          <div className="flex items-center gap-1.5 text-[11px] text-zinc-400 font-mono">
            <FileCheck className="w-3 h-3 text-indigo-400" />
            <span>
              {selectedCount} document{selectedCount === 1 ? '' : 's'} selected
            </span>
          </div>
        </div>
      </div>

      <div className="flex items-center gap-2">
        <button
          onClick={onNewChat}
          className="px-2.5 py-1.5 rounded-lg text-xs font-medium bg-zinc-800 hover:bg-zinc-700 text-zinc-200 transition-colors flex items-center gap-1.5 cursor-pointer"
        >
          <PlusCircle className="w-3.5 h-3.5" />
          <span>New Chat</span>
        </button>

        {hasMessages && (
          <button
            onClick={onClearChat}
            className="px-2 py-1.5 rounded-lg text-xs text-zinc-400 hover:text-red-400 hover:bg-zinc-800 transition-colors flex items-center gap-1 cursor-pointer"
            title="Clear current conversation"
          >
            <Trash2 className="w-3.5 h-3.5" />
            <span>Clear</span>
          </button>
        )}
      </div>
    </div>
  );
};
