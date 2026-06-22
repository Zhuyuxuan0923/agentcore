"use client";

import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import {
  listKBs,
  createKB,
  deleteKB,
  listConversations,
  deleteConversation,
  type Conversation,
} from "@/lib/api";

interface SidebarProps {
  activeKB: string;
  onSwitchKB: (name: string) => void;
  onNewConversation: () => void;
  onSelectConversation: (id: string) => void;
  onDeleteKB?: (name: string) => void;
  activeConversationId: string | null;
  refreshKey: number;
}

export default function Sidebar({
  activeKB,
  onSwitchKB,
  onNewConversation,
  onSelectConversation,
  onDeleteKB,
  activeConversationId,
  refreshKey,
}: SidebarProps) {
  const [kbs, setKbs] = useState<string[]>(["默认知识库"]);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [newKBName, setNewKBName] = useState("");
  const [showNewKB, setShowNewKB] = useState(false);

  useEffect(() => {
    listKBs().then(setKbs).catch(() => {});
    listConversations().then(setConversations).catch(() => {});
  }, [refreshKey]);

  const handleCreateKB = async () => {
    if (!newKBName.trim()) return;
    await createKB(newKBName.trim());
    setKbs((prev) => [...prev, newKBName.trim()]);
    setNewKBName("");
    setShowNewKB(false);
  };

  const handleDeleteKB = async (name: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (name === "默认知识库") return;
    const ok = window.confirm(`确定要删除知识库 "${name}" 及其所有文档吗？`);
    if (!ok) return;
    try {
      await deleteKB(name);
      setKbs((prev) => prev.filter((k) => k !== name));
      onDeleteKB?.(name); // 通知父组件
    } catch {
      // ignore
    }
  };

  const handleDeleteConversation = async (convId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    const ok = window.confirm("确定要删除这个对话吗？");
    if (!ok) return;
    try {
      await deleteConversation(convId);
      setConversations((prev) => prev.filter((c) => c.id !== convId));
    } catch {
      // ignore
    }
  };

  return (
    <aside className="flex h-full flex-col border-r border-border/40 bg-background/80 backdrop-blur">
      {/* Logo */}
      <div className="px-5 py-5">
        <h1 className="text-lg font-serif font-bold tracking-tight text-foreground">
          PersonalQA
        </h1>
        <p className="mt-0.5 text-xs text-muted-foreground">知识库问答助手</p>
      </div>

      <Separator className="mx-4 w-auto" />

      <ScrollArea className="flex-1 px-3 py-3">
        {/* Knowledge Base Switcher */}
        <div className="mb-4">
          <div className="flex items-center justify-between px-2 mb-2">
            <span className="text-[10px] font-semibold uppercase tracking-[0.12em] text-muted-foreground">
              知识库
            </span>
            <button
              onClick={() => setShowNewKB(!showNewKB)}
              className="text-xs text-muted-foreground hover:text-accent transition-colors"
              title="创建新知识库"
            >
              +
            </button>
          </div>

          {showNewKB && (
            <div className="flex gap-1 px-2 mb-2">
              <input
                type="text"
                value={newKBName}
                onChange={(e) => setNewKBName(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleCreateKB()}
                placeholder="知识库名称"
                className="flex-1 rounded-md border border-border/60 bg-background px-2 py-1 text-xs outline-none focus:border-accent/50"
              />
              <button
                onClick={handleCreateKB}
                className="rounded-md bg-accent px-2 py-1 text-xs text-accent-foreground hover:bg-accent/80 transition-colors"
              >
                创建
              </button>
            </div>
          )}

          <div className="space-y-0.5">
            {kbs.map((kb) => (
              <div key={kb} className="group/kb relative">
                <button
                  onClick={() => onSwitchKB(kb)}
                  className={`w-full rounded-md px-3 py-1.5 text-left text-sm transition-colors ${
                    kb === activeKB
                      ? "bg-accent/10 text-accent font-medium"
                      : "text-muted-foreground hover:bg-accent/5 hover:text-foreground"
                  }`}
                >
                  {kb}
                </button>
                {kb !== "默认知识库" && (
                  <button
                    onClick={(e) => handleDeleteKB(kb, e)}
                    className="absolute right-1.5 top-1/2 -translate-y-1/2 p-1 rounded opacity-0 group-hover/kb:opacity-100 hover:bg-destructive/10 text-muted-foreground hover:text-destructive transition-all"
                    title="删除知识库"
                  >
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M3 6h18" />
                      <path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6" />
                      <path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2" />
                    </svg>
                  </button>
                )}
              </div>
            ))}
          </div>
        </div>

        <Separator className="my-3" />

        {/* Conversation History */}
        <div>
          <div className="px-2 mb-2">
            <span className="text-[10px] font-semibold uppercase tracking-[0.12em] text-muted-foreground">
              对话历史
            </span>
          </div>

          <div className="space-y-0.5">
            {conversations.length === 0 && (
              <p className="px-3 py-4 text-xs text-muted-foreground/60 text-center">
                暂无对话
              </p>
            )}
            {conversations.map((conv) => (
              <div
                key={conv.id}
                className="group/conv relative"
              >
                <div
                  onClick={() => onSelectConversation(conv.id)}
                  className={`rounded-md px-3 py-1.5 pr-7 text-xs cursor-pointer transition-colors truncate ${
                    conv.id === activeConversationId
                      ? "bg-accent/10 text-accent font-medium"
                      : "text-muted-foreground hover:bg-accent/5"
                  }`}
                  title={conv.title}
                >
                  {conv.title}
                </div>
                <button
                  onClick={(e) => handleDeleteConversation(conv.id, e)}
                  className="absolute right-1.5 top-1/2 -translate-y-1/2 p-1 rounded opacity-0 group-hover/conv:opacity-100 hover:bg-destructive/10 text-muted-foreground hover:text-destructive transition-all"
                  title="删除对话"
                >
                  <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M3 6h18" />
                    <path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6" />
                    <path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2" />
                  </svg>
                </button>
              </div>
            ))}
          </div>
        </div>
      </ScrollArea>

      {/* New Conversation */}
      <div className="px-4 py-3 border-t border-border/40">
        <Button
          variant="outline"
          className="w-full justify-start text-sm font-normal text-muted-foreground hover:text-foreground"
          onClick={onNewConversation}
        >
          新建对话
        </Button>
      </div>
    </aside>
  );
}
