/** API 客户端 — 连接 FastAPI 后端 (localhost:8000) */

const BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface Citation {
  number: number;
  text: string;
}

export interface ChatResponse {
  question: string;
  answer: string;
  sources: { doc_id: string; text: string; score: number }[];
  citations: Citation[];
  conversation_id: string;
}

export interface UploadResponse {
  status: string;
  message: string;
  kb_name: string;
  file_name: string;
  chunk_count: number;
}

export interface Conversation {
  id: string;
  title: string;
  kb_name: string;
  created_at: string;
}

export interface Message {
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
}

export async function uploadFile(file: File, kbName: string): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", file);
  form.append("kb_name", kbName);

  const res = await fetch(`${BASE_URL}/api/upload`, { method: "POST", body: form });
  if (!res.ok) throw new Error((await res.json()).detail || "Upload failed");
  return res.json();
}

export async function chat(
  question: string,
  kbName: string = "默认知识库",
  conversationId?: string | null
): Promise<ChatResponse> {
  const res = await fetch(`${BASE_URL}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, kb_name: kbName, conversation_id: conversationId }),
  });
  if (!res.ok) throw new Error((await res.json()).detail || "Chat failed");
  return res.json();
}

export async function listKBs(): Promise<string[]> {
  const res = await fetch(`${BASE_URL}/api/kb`);
  if (!res.ok) return [];
  const data = await res.json();
  return data.knowledge_bases || [];
}

export async function createKB(name: string): Promise<void> {
  await fetch(`${BASE_URL}/api/kb`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
}

export async function deleteKB(name: string): Promise<void> {
  await fetch(`${BASE_URL}/api/kb/${encodeURIComponent(name)}`, { method: "DELETE" });
}

export async function listConversations(): Promise<Conversation[]> {
  const res = await fetch(`${BASE_URL}/api/conversations`);
  if (!res.ok) return [];
  const data = await res.json();
  return data.conversations || [];
}

export async function getHistory(convId: string): Promise<Message[]> {
  const res = await fetch(`${BASE_URL}/api/history/${convId}`);
  if (!res.ok) return [];
  const data = await res.json();
  return data.messages || [];
}

export async function deleteConversation(convId: string): Promise<void> {
  const res = await fetch(`${BASE_URL}/api/conversations/${encodeURIComponent(convId)}`, {
    method: "DELETE",
  });
  if (!res.ok) throw new Error((await res.json()).detail || "Delete failed");
}

export async function newConversation(): Promise<string> {
  const res = await fetch(`${BASE_URL}/api/conversations/new`, { method: "POST" });
  if (!res.ok) return "";
  const data = await res.json();
  return data.conversation_id || "";
}
