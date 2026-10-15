import React from 'react';
import type { ChatMessage } from '../../api/types';
import { SourcesList } from './SourcesList';
import { RetrievalDetailsAccordion } from './RetrievalDetailsAccordion';
import { User, Sparkles, AlertTriangle, ShieldCheck } from 'lucide-react';

interface ChatMessageItemProps {
  message: ChatMessage;
}

export const ChatMessageItem: React.FC<ChatMessageItemProps> = ({ message }) => {
  const isUser = message.sender === 'user';

  if (isUser) {
    return (
      <div className="flex flex-col items-end space-y-1.5 max-w-2xl ml-auto">
        <div className="flex items-center gap-2 text-[11px] font-semibold text-zinc-400 uppercase tracking-wider pr-1">
          <span>YOU</span>
          <User className="w-3.5 h-3.5 text-zinc-400" />
        </div>
        <div className="bg-indigo-600 text-white p-3.5 rounded-2xl rounded-tr-sm text-xs leading-relaxed shadow-sm font-sans">
          {message.text}
        </div>
        <span className="text-[10px] text-zinc-500 font-mono pr-1">
          {message.timestamp}
        </span>
      </div>
    );
  }

  return (
    <div className="flex flex-col items-start space-y-2.5 max-w-3xl mr-auto">
      <div className="flex items-center justify-between w-full pr-2">
        <div className="flex items-center gap-2 text-[11px] font-semibold text-indigo-400 uppercase tracking-wider">
          <Sparkles className="w-3.5 h-3.5" />
          <span>ASSISTANT</span>
        </div>
        {message.groundedness !== undefined && (
          <span className="text-[10px] text-emerald-400 bg-emerald-950/60 border border-emerald-800/50 px-2 py-0.5 rounded-full font-mono flex items-center gap-1">
            <ShieldCheck className="w-3 h-3" />
            {Math.round(message.groundedness * 100)}% Grounded
          </span>
        )}
      </div>

      {/* Message Content */}
      <div className="bg-zinc-900/90 border border-zinc-800/90 text-zinc-100 p-4 rounded-2xl rounded-tl-sm text-xs leading-relaxed shadow-md space-y-3 font-sans w-full">
        {message.isError ? (
          <div className="flex items-start gap-2.5 text-red-400 bg-red-950/40 border border-red-900/50 p-3 rounded-xl">
            <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
            <div>
              <p className="font-semibold text-xs">Error Generating Answer</p>
              <p className="text-[11px] text-red-300/80 mt-0.5">{message.text}</p>
            </div>
          </div>
        ) : (
          <div className="whitespace-pre-wrap font-sans text-zinc-200 leading-relaxed">
            {message.text}
          </div>
        )}

        {/* Sources Section */}
        {message.sources && message.sources.length > 0 && (
          <SourcesList sources={message.sources} />
        )}

        {/* Retrieval Transparency Details */}
        {message.retrievalDetails && (
          <div className="pt-2">
            <RetrievalDetailsAccordion details={message.retrievalDetails} />
          </div>
        )}
      </div>

      <span className="text-[10px] text-zinc-500 font-mono pl-1">
        {message.timestamp}
      </span>
    </div>
  );
};
