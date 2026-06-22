"use client";

import { useRef, useEffect } from "react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Card } from "@/components/ui/card";
import MarkdownRenderer from "./markdown-renderer";
import type { Citation } from "@/lib/api";

interface Message {
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
}

interface ChatAreaProps {
  messages: Message[];
  isLoading: boolean;
}

function CitationBadge({ number }: { number: number }) {
  return (
    <sup className="inline-flex items-center justify-center w-4 h-4 ml-0.5 rounded-full bg-accent/15 text-[10px] font-semibold text-accent cursor-pointer hover:bg-accent/25 transition-colors align-top">
      {number}
    </sup>
  );
}

function MessageBubble({ msg }: { msg: Message }) {
  const isUser = msg.role === "user";
  const hasCitations = msg.citations && msg.citations.length > 0;

  /**
   * 渲染消息内容。
   *
   * 三种情况：
   *   1. 用户消息 -> 纯文本（用户不写 Markdown）
   *   2. AI 消息无引用 -> 直接渲染 Markdown
   *   3. AI 消息有引用 -> split [N] -> 文本段 Markdown 渲染，引用段 CitationBadge
   *
   * 为什么 split 和分段渲染？
   *   引用编号 [1] 在 Markdown 中会被误解析为链接语法的一部分。
   *   先把引用编号剥离出来，分别渲染文字和 Badge，就能避免冲突。
   *   代价：横跨引用编号的 Markdown 格式会断裂（如 **粗[1]体**），
   *   但实际场景中引用编号几乎都在句尾，此问题极少发生。
   */
  const renderContent = () => {
    // 情况 1：用户消息 -> 纯文本
    if (isUser) {
      return <p className="whitespace-pre-wrap">{msg.content}</p>;
    }

    // 情况 2：无引用 -> 完整 Markdown 渲染
    if (!hasCitations) {
      return <MarkdownRenderer>{msg.content}</MarkdownRenderer>;
    }

    // 情况 3：有引用 -> split + 分段渲染
    const parts = msg.content.split(/(\[\d+\])/g);
    return parts.map((part, i) => {
      const match = part.match(/^\[(\d+)\]$/);
      if (match) {
        const num = parseInt(match[1], 10);
        const citation = msg.citations!.find((c) => c.number === num);
        if (citation) {
          return (
            <span
              key={i}
              className="inline cursor-pointer group relative"
              title={citation.text.slice(0, 100)}
            >
              <CitationBadge number={num} />
              <span className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 hidden group-hover:block w-64 rounded-lg border border-border/60 bg-background p-3 text-xs text-muted-foreground shadow-lg z-50 leading-relaxed">
                {citation.text}
              </span>
            </span>
          );
        }
        // 引用编号在 citations 中找不到（不应发生，但容错）
        return <span key={i}>{part}</span>;
      }
      // 文本片段 -> Markdown 渲染
      return <MarkdownRenderer key={i}>{part}</MarkdownRenderer>;
    });
  };

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"} mb-6`}>
      <div className={`max-w-[75%] ${isUser ? "order-1" : "order-1"}`}>
        <div className="flex items-center gap-2 mb-1.5">
          <span className="text-[10px] font-semibold uppercase tracking-[0.1em] text-muted-foreground/60">
            {isUser ? "You" : "PersonalQA"}
          </span>
        </div>
        {isUser ? (
          <Card className="px-4 py-3 bg-accent/5 border-accent/10 text-sm leading-relaxed">
            {renderContent()}
          </Card>
        ) : (
          <div className="space-y-3">
            <div className="text-sm leading-relaxed text-foreground">{renderContent()}</div>
            {hasCitations && (
              <div className="pt-3 border-t border-border/30">
                <p className="text-[10px] font-semibold uppercase tracking-[0.1em] text-muted-foreground mb-2">
                  引用来源
                </p>
                <div className="space-y-1.5">
                  {msg.citations!.map((c) => (
                    <div
                      key={c.number}
                      className="flex gap-3 rounded-md border-l-2 border-accent/40 bg-accent/3 px-3 py-2 text-xs text-muted-foreground"
                    >
                      <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-accent/10 text-[10px] font-semibold text-accent">
                        {c.number}
                      </span>
                      <span className="leading-relaxed line-clamp-2">{c.text}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default function ChatArea({ messages, isLoading }: ChatAreaProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  return (
    <ScrollArea className="flex-1">
      <div className="mx-auto max-w-2xl px-6 py-8">
        {messages.length === 0 && !isLoading && (
          <div className="flex flex-col items-center justify-center py-32 text-center">
            <div className="mb-4 text-4xl font-serif font-bold text-accent/40">&ldquo;&rdquo;</div>
            <h2 className="mb-2 text-lg font-serif font-semibold text-foreground">
              上传文档，开始提问
            </h2>
            <p className="max-w-sm text-sm text-muted-foreground/70">
              支持 PDF、DOCX、Markdown 格式。知识库中的内容会成为回答的依据，并标注引用来源。
            </p>
          </div>
        )}

        {messages.map((msg, i) => (
          <MessageBubble key={i} msg={msg} />
        ))}

        {isLoading && (
          <div className="flex items-center gap-2 mb-6">
            <div className="flex gap-1">
              <span className="h-2 w-2 animate-bounce rounded-full bg-accent/40 [animation-delay:0ms]" />
              <span className="h-2 w-2 animate-bounce rounded-full bg-accent/40 [animation-delay:150ms]" />
              <span className="h-2 w-2 animate-bounce rounded-full bg-accent/40 [animation-delay:300ms]" />
            </div>
            <span className="text-xs text-muted-foreground/60">检索中...</span>
          </div>
        )}

        <div ref={bottomRef} />
      </div>
    </ScrollArea>
  );
}
