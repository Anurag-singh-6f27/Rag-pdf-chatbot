import { useState, useEffect } from 'react';
import { LeftNav } from './components/navigation/LeftNav';
import { DocumentLibrary } from './components/documents/DocumentLibrary';
import { ChatPane } from './components/chat/ChatPane';
import { UploadModal } from './components/documents/UploadModal';
import type { Document, KnowledgeBaseStatus, ChatMessage } from './api/types';
import { realApi } from './api/realApi';

export function App() {
  const [activeTab, setActiveTab] = useState<'documents' | 'chat'>('documents');
  const [documents, setDocuments] = useState<Document[]>([]);
  const [kbStatus, setKbStatus] = useState<KnowledgeBaseStatus>({
    isReady: false,
    documentsCount: 0,
    pagesCount: 0,
    chunksCount: 0,
    faissStatus: 'empty',
    bm25Status: 'empty',
    flashRankStatus: 'empty',
    geminiStatus: 'ready',
    embeddingModel: 'sentence-transformers/all-MiniLM-L6-v2',
  });
  const [isBuildingKb, setIsBuildingKb] = useState<boolean>(false);
  const [buildingStage, setBuildingStage] = useState<string>('');
  const [isUploadModalOpen, setIsUploadModalOpen] = useState<boolean>(false);

  // Chat State
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isChatLoading, setIsChatLoading] = useState<boolean>(false);

  // Load initial state from service
  useEffect(() => {
    const initData = async () => {
      const docs = await realApi.getDocuments();
      setDocuments(docs);
      const status = await realApi.getKbStatus();
      setKbStatus(status);
    };
    initData();
  }, []);

  // Document actions
  const handleToggleSelectDocument = async (id: string) => {
    const updatedDocs = await realApi.toggleDocumentSelection(id);
    setDocuments(updatedDocs);
    const updatedStatus = await realApi.getKbStatus();
    setKbStatus(updatedStatus);
  };

  const handleToggleAllDocuments = async (select: boolean) => {
    const updatedDocs = await realApi.toggleAllDocuments(select);
    setDocuments(updatedDocs);
    const updatedStatus = await realApi.getKbStatus();
    setKbStatus(updatedStatus);
  };

  const handleRemoveDocument = async (id: string) => {
    const updatedDocs = await realApi.removeDocument(id);
    setDocuments(updatedDocs);
    const updatedStatus = await realApi.getKbStatus();
    setKbStatus(updatedStatus);
  };

  const handleUploadDocuments = async (files: File[]) => {
    const updatedDocs = await realApi.uploadDocuments(files);
    setDocuments(updatedDocs);
    const updatedStatus = await realApi.getKbStatus();
    setKbStatus(updatedStatus);
  };

  // Build KB Action
  const handleBuildKnowledgeBase = async () => {
    setIsBuildingKb(true);
    await realApi.buildKnowledgeBase((stage) => {
      setBuildingStage(stage);
    });
    setIsBuildingKb(false);
    setBuildingStage('');
    const updatedStatus = await realApi.getKbStatus();
    setKbStatus(updatedStatus);
  };

  // Chat Actions
  const handleSendMessage = async (query: string) => {
    const selectedDocIds = documents.filter((d) => d.selected).map((d) => d.id);
    if (selectedDocIds.length === 0) return;

    const userMessage: ChatMessage = {
      id: `user-${Date.now()}`,
      sender: 'user',
      text: query,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    setMessages((prev) => [...prev, userMessage]);
    setIsChatLoading(true);

    try {
      const assistantResponse = await realApi.sendQuestion(query, selectedDocIds);
      setMessages((prev) => [...prev, assistantResponse]);
    } catch {
      const errorMessage: ChatMessage = {
        id: `err-${Date.now()}`,
        sender: 'assistant',
        text: 'Failed to process retrieval query. Please try again.',
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        isError: true,
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsChatLoading(false);
    }
  };

  const handleNewChat = () => {
    setMessages([]);
  };

  const handleClearChat = () => {
    setMessages([]);
  };

  const selectedDocsCount = documents.filter((d) => d.selected).length;

  return (
    <div className="flex h-screen w-screen bg-zinc-950 text-zinc-100 overflow-hidden font-sans">
      {/* Persistent Left Navigation Sidebar */}
      <LeftNav
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        onOpenUploadModal={() => setIsUploadModalOpen(true)}
        selectedCount={selectedDocsCount}
      />

      {/* Main Content Area */}
      <main className="flex-1 h-full overflow-hidden flex">
        {activeTab === 'documents' ? (
          /* EXACT SCREEN SHOWN IN THE USER'S SCREENSHOT (3-PANE LAYOUT) */
          <>
            <DocumentLibrary
              documents={documents}
              kbStatus={kbStatus}
              isBuildingKb={isBuildingKb}
              buildingStage={buildingStage}
              onToggleSelect={handleToggleSelectDocument}
              onToggleAll={handleToggleAllDocuments}
              onRemoveDocument={handleRemoveDocument}
              onBuildKb={handleBuildKnowledgeBase}
              onOpenUploadModal={() => setIsUploadModalOpen(true)}
            />
            <ChatPane
              messages={messages}
              selectedCount={selectedDocsCount}
              totalDocsCount={documents.length}
              isLoading={isChatLoading}
              onSendMessage={handleSendMessage}
              onNewChat={handleNewChat}
              onClearChat={handleClearChat}
            />
          </>
        ) : (
          /* CHAT ONLY VIEW (EXPANDED CHAT PANE) */
          <ChatPane
            messages={messages}
            selectedCount={selectedDocsCount}
            totalDocsCount={documents.length}
            isLoading={isChatLoading}
            onSendMessage={handleSendMessage}
            onNewChat={handleNewChat}
            onClearChat={handleClearChat}
          />
        )}
      </main>

      {/* Document Upload Modal Overlay */}
      <UploadModal
        isOpen={isUploadModalOpen}
        onClose={() => setIsUploadModalOpen(false)}
        onUpload={handleUploadDocuments}
      />
    </div>
  );
}

export default App;
