import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "PersonalQA - 个人知识库问答 Agent",
  description: "上传文档，提问，获得带引用的答案。",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="zh-CN" className="h-full antialiased">
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
