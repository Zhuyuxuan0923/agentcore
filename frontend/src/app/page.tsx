"use client";

import { useState, useCallback } from "react";
import Sidebar from "@/components/chat/sidebar";
import ChatArea from "@/components/chat/chat-area";
import InputArea from "@/components/chat/input-area";
import { chat, getHistory, type Citation } from "@/lib/api";

interface Message {
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
}

export default function Home() {
  const [activeKB, setActiveKB] = useState("默认知识库");
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [sidebarRefreshKey, setSidebarRefreshKey] = useState(0);

  const handleSend = useCallback(
    async (question: string) => {
      const userMsg: Message = { role: "user", content: question };
      setMessages((prev) => [...prev, userMsg]);
      setIsLoading(true);

      try {
        const result = await chat(question, activeKB, conversationId);
        const assistantMsg: Message = {
          role: "assistant",
          content: result.answer,
          citations: result.citations,
        };
        setMessages((prev) => [...prev, assistantMsg]);

        // 新对话时更新 conversationId 并刷新侧边栏
        if (result.conversation_id !== conversationId) {
          setConversationId(result.conversation_id);
          setSidebarRefreshKey((k) => k + 1);
        }
      } catch (err: unknown) {
        const errorMsg = err instanceof Error ? err.message : "请求失败";
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: `[错误] ${errorMsg}。请确认后端服务已启动。`,
          },
        ]);
      } finally {
        setIsLoading(false);
      }
    },
    [activeKB, conversationId]
  );

  const handleSwitchKB = useCallback((name: string) => {
    setActiveKB(name);
    setMessages([]);
    setConversationId(null);
  }, []);

  const handleNewConversation = useCallback(() => {
    setMessages([]);
    setConversationId(null);
  }, []);

  const handleDeleteKB = useCallback(
    (name: string) => {
      if (name === activeKB) {
        setActiveKB("默认知识库");
        setMessages([]);
        setConversationId(null);
      }
      setSidebarRefreshKey((k) => k + 1);
    },
    [activeKB]
  );

  const handleSelectConversation = useCallback(async (convId: string) => {
    setConversationId(convId);
    setIsLoading(true);
    try {
      const history = await getHistory(convId);
      if (history.length > 0) {
        setMessages(history);
      }
    } catch {
      // 加载失败则保持空消息
    } finally {
      setIsLoading(false);
    }
  }, []);

  const handleUploadSuccess = useCallback(() => {
    // 上传后刷新知识库列表
    setSidebarRefreshKey((k) => k + 1);
  }, []);

  return (
    <div className="flex h-screen overflow-hidden">
      {/* Sidebar */}
      <div className="w-64 shrink-0">
        <Sidebar
          activeKB={activeKB}
          onSwitchKB={handleSwitchKB}
          onNewConversation={handleNewConversation}
          onSelectConversation={handleSelectConversation}
          onDeleteKB={handleDeleteKB}
          activeConversationId={conversationId}
          refreshKey={sidebarRefreshKey}
        />
      </div>

      {/* Main */}
      <div className="flex flex-1 flex-col min-w-0">
        <ChatArea messages={messages} isLoading={isLoading} />
        <InputArea
          activeKB={activeKB}
          onSend={handleSend}
          onUploadSuccess={handleUploadSuccess}
          disabled={isLoading}
        />
      </div>
    </div>
  );
}
