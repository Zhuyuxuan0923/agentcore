"use client";

import { useState, useRef } from "react";
import { Button } from "@/components/ui/button";
import { uploadFile, listKBs } from "@/lib/api";

interface InputAreaProps {
  activeKB: string;
  onSend: (question: string) => void;
  onUploadSuccess: () => void;
  disabled: boolean;
}

export default function InputArea({ activeKB, onSend, onUploadSuccess, disabled }: InputAreaProps) {
  const [input, setInput] = useState("");
  const [uploading, setUploading] = useState(false);
  const [uploadMsg, setUploadMsg] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);

  const handleSend = () => {
    if (!input.trim() || disabled) return;
    onSend(input.trim());
    setInput("");
  };

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploading(true);
    setUploadMsg("");
    try {
      const result = await uploadFile(file, activeKB);
      setUploadMsg(`已索引 "${result.file_name}"（${result.chunk_count} 个文档块）`);
      onUploadSuccess();
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "上传失败";
      setUploadMsg(`上传失败: ${msg}`);
    } finally {
      setUploading(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  };

  return (
    <div className="border-t border-border/40 bg-background/80 backdrop-blur px-5 py-4">
      <div className="mx-auto max-w-2xl">
        {/* Upload + status */}
        <div className="flex items-center gap-2 mb-3">
          <input
            ref={fileRef}
            type="file"
            accept=".pdf,.docx,.md,.txt"
            onChange={handleUpload}
            className="hidden"
            id="file-upload"
          />
          <label
            htmlFor="file-upload"
            className="inline-flex cursor-pointer items-center gap-1.5 rounded-md px-3 py-1 text-xs text-muted-foreground hover:text-accent hover:bg-accent/5 transition-colors border border-border/30"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="17 8 12 3 7 8" />
              <line x1="12" y1="3" x2="12" y2="15" />
            </svg>
            {uploading ? "上传中..." : "上传文件"}
          </label>
          <span className="text-[10px] text-muted-foreground/50">|</span>
          <span className="text-[10px] text-muted-foreground/50">
            当前知识库: {activeKB}
          </span>
          {uploadMsg && (
            <span className="text-[10px] text-accent truncate max-w-[200px]">{uploadMsg}</span>
          )}
        </div>

        {/* Input row */}
        <div className="flex gap-2">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                handleSend();
              }
            }}
            placeholder="输入问题... (Enter 发送, Shift+Enter 换行)"
            rows={1}
            disabled={disabled}
            className="flex-1 resize-none rounded-xl border border-border/60 bg-background px-4 py-2.5 text-sm outline-none placeholder:text-muted-foreground/40 focus:border-accent/40 focus:ring-1 focus:ring-accent/20 transition-all disabled:opacity-50"
          />
          <Button
            onClick={handleSend}
            disabled={disabled || !input.trim()}
            size="sm"
            className="rounded-xl px-4 shrink-0 bg-accent text-accent-foreground hover:bg-accent/85 transition-colors"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <line x1="22" y1="2" x2="11" y2="13" />
              <polygon points="22 2 15 22 11 13 2 9 22 2" />
            </svg>
          </Button>
        </div>
      </div>
    </div>
  );
}
