import React from 'react';
import type { ChatMessage } from '../../api/types';
import { ChatHeader } from './ChatHeader';
import { ChatMessageList } from './ChatMessageList';
import { ChatInput } from './ChatInput';

interface ChatPaneProps {
  messages: ChatMessage[];
  selectedCount: number;
  totalDocsCount: number;
  isLoading: boolean;
  onSendMessage: (query: string) => void;
  onNewChat: () => void;
  onClearChat: () => void;
}

export const ChatPane: React.FC<ChatPaneProps> = ({
  messages,
  selectedCount,
  totalDocsCount,
  isLoading,
  onSendMessage,
  onNewChat,
  onClearChat,
}) => {
  return (
    <div className="flex-1 bg-zinc-950 flex flex-col justify-between h-full overflow-hidden min-w-0">
      <ChatHeader
        selectedCount={selectedCount}
        totalDocsCount={totalDocsCount}
        onNewChat={onNewChat}
        onClearChat={onClearChat}
        hasMessages={messages.length > 0}
      />

      <ChatMessageList
        messages={messages}
        selectedCount={selectedCount}
        totalDocsCount={totalDocsCount}
        isLoading={isLoading}
      />

      <ChatInput
        onSendMessage={onSendMessage}
        disabled={selectedCount === 0}
        isLoading={isLoading}
        selectedCount={selectedCount}
      />
    </div>
  );
};
