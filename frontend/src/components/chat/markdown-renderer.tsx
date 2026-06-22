"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import type { Components } from "react-markdown";

/**
 * 代码块的 props 类型。
 * react-markdown 传给 code 组件的 className 格式是 "language-xxx"
 */
interface CodeBlockProps {
  className?: string;
  children?: React.ReactNode;
  [key: string]: unknown;
}

/**
 * 我们的暖色调代码高亮主题。
 * 基于 Warm Scholar 配色体系定义每种 token 的颜色。
 */
const warmScholarTheme = {
  'code[class*="language-"]': {
    color: "oklch(0.25 0.02 80)",
    background: "oklch(0.96 0.01 85)",
    fontFamily: '"JetBrains Mono", "Consolas", monospace',
    fontSize: "0.8125rem",
    lineHeight: "1.7",
    tabSize: 4,
    hyphens: "none",
  },
  'pre[class*="language-"]': {
    color: "oklch(0.25 0.02 80)",
    background: "oklch(0.96 0.01 85)",
    fontFamily: '"JetBrains Mono", "Consolas", monospace',
    fontSize: "0.8125rem",
    lineHeight: "1.7",
    tabSize: 4,
    hyphens: "none",
    margin: "0",
    overflow: "auto",
    borderRadius: "0 0 0.5rem 0.5rem",
  },
  comment: { color: "oklch(0.55 0.02 80)", fontStyle: "italic" },
  prolog: { color: "oklch(0.55 0.02 80)" },
  doctype: { color: "oklch(0.55 0.02 80)" },
  cdata: { color: "oklch(0.55 0.02 80)" },
  punctuation: { color: "oklch(0.45 0.03 80)" },
  property: { color: "oklch(0.45 0.08 55)" },
  keyword: { color: "oklch(0.45 0.10 45)", fontWeight: "500" },
  tag: { color: "oklch(0.45 0.10 45)" },
  'class-name': { color: "oklch(0.40 0.08 55)", fontWeight: "500" },
  boolean: { color: "oklch(0.48 0.12 65)" },
  constant: { color: "oklch(0.48 0.12 65)" },
  symbol: { color: "oklch(0.48 0.12 65)" },
  deleted: { color: "oklch(0.48 0.12 65)" },
  number: { color: "oklch(0.48 0.12 65)" },
  selector: { color: "oklch(0.42 0.09 140)" },
  'attr-name': { color: "oklch(0.42 0.09 140)" },
  string: { color: "oklch(0.40 0.08 140)" },
  char: { color: "oklch(0.40 0.08 140)" },
  builtin: { color: "oklch(0.45 0.10 45)" },
  inserted: { color: "oklch(0.40 0.08 140)" },
  operator: { color: "oklch(0.45 0.03 80)" },
  entity: { color: "oklch(0.45 0.10 45)", cursor: "help" },
  url: { color: "oklch(0.42 0.09 140)" },
  variable: { color: "oklch(0.30 0.03 80)" },
  atrule: { color: "oklch(0.48 0.12 65)" },
  'attr-value': { color: "oklch(0.40 0.08 140)" },
  function: { color: "oklch(0.35 0.06 55)", fontWeight: "500" },
  regex: { color: "oklch(0.48 0.12 65)" },
  important: { color: "oklch(0.48 0.12 65)", fontWeight: "bold" },
  bold: { fontWeight: "bold" },
  italic: { fontStyle: "italic" },
} as const;

/**
 * 自定义代码块渲染组件。
 * 带语言标签头部 + 语法高亮代码区。
 */
function CodeBlock({ className, children, ...props }: CodeBlockProps) {
  const match = /language-(\w+)/.exec(className || "");
  const language = match ? match[1] : "";

  // 行内代码（无语言标注，不在 pre 内）
  const isInline = !match && !className;

  if (isInline) {
    return (
      <code
        className="px-1 py-0.5 rounded-md bg-accent/8 text-[0.8125rem] font-mono text-accent/90"
        {...props}
      >
        {children}
      </code>
    );
  }

  return (
    <div className="my-4 rounded-xl border border-border/60 overflow-hidden shadow-sm">
      {/* 头部栏：语言标签 */}
      <div className="flex items-center justify-between px-4 py-2 bg-accent/8 border-b border-border/30">
        <span className="text-[10px] font-semibold uppercase tracking-[0.12em] text-muted-foreground">
          {language || "code"}
        </span>
        <span className="flex gap-1.5">
          <span className="h-2.5 w-2.5 rounded-full bg-accent/25" />
          <span className="h-2.5 w-2.5 rounded-full bg-accent/25" />
          <span className="h-2.5 w-2.5 rounded-full bg-accent/25" />
        </span>
      </div>
      {/* 代码区 */}
      <SyntaxHighlighter
        style={warmScholarTheme}
        language={language || "text"}
        PreTag="pre"
        CodeTag="code"
        customStyle={{
          margin: 0,
          padding: "1rem 1.25rem",
          background: "oklch(0.97 0.005 85)",
          borderRadius: 0,
        }}
      >
        {String(children).replace(/\n$/, "")}
      </SyntaxHighlighter>
    </div>
  );
}

/**
 * react-markdown 的组件映射。
 * 把标准 Markdown 元素替换为自定义组件。
 */
const markdownComponents: Partial<Components> = {
  code: CodeBlock as Components["code"],
  // 标题使用衬线体，与 Warm Scholar 主题一致
  h1: ({ children, ...props }) => (
    <h1 className="mt-6 mb-3 text-xl font-serif font-bold text-foreground" {...props}>
      {children}
    </h1>
  ),
  h2: ({ children, ...props }) => (
    <h2 className="mt-5 mb-2 text-lg font-serif font-semibold text-foreground" {...props}>
      {children}
    </h2>
  ),
  h3: ({ children, ...props }) => (
    <h3 className="mt-4 mb-2 text-base font-serif font-semibold text-foreground" {...props}>
      {children}
    </h3>
  ),
  // 段落
  p: ({ children, ...props }) => (
    <p className="mb-3 leading-relaxed" {...props}>
      {children}
    </p>
  ),
  // 无序列表
  ul: ({ children, ...props }) => (
    <ul className="mb-3 pl-5 space-y-1 list-disc" {...props}>
      {children}
    </ul>
  ),
  // 有序列表
  ol: ({ children, ...props }) => (
    <ol className="mb-3 pl-5 space-y-1 list-decimal" {...props}>
      {children}
    </ol>
  ),
  // 列表项
  li: ({ children, ...props }) => (
    <li className="text-sm leading-relaxed" {...props}>
      {children}
    </li>
  ),
  // 粗体
  strong: ({ children, ...props }) => (
    <strong className="font-semibold text-foreground/90" {...props}>
      {children}
    </strong>
  ),
  // 链接
  a: ({ children, href, ...props }) => (
    <a
      href={href}
      className="text-accent/80 underline underline-offset-2 hover:text-accent transition-colors"
      target="_blank"
      rel="noopener noreferrer"
      {...props}
    >
      {children}
    </a>
  ),
  // 引用块
  blockquote: ({ children, ...props }) => (
    <blockquote
      className="my-3 border-l-3 border-accent/30 pl-4 text-sm text-muted-foreground italic"
      {...props}
    >
      {children}
    </blockquote>
  ),
  // 水平分割线
  hr: (props) => <hr className="my-6 border-border/40" {...props} />,
  // 表格
  table: ({ children, ...props }) => (
    <div className="my-4 overflow-x-auto">
      <table className="min-w-full border-collapse text-sm" {...props}>
        {children}
      </table>
    </div>
  ),
  thead: ({ children, ...props }) => (
    <thead className="border-b border-border/60 bg-accent/5" {...props}>
      {children}
    </thead>
  ),
  th: ({ children, ...props }) => (
    <th className="px-3 py-2 text-left text-xs font-semibold text-muted-foreground" {...props}>
      {children}
    </th>
  ),
  td: ({ children, ...props }) => (
    <td className="px-3 py-2 border-b border-border/20" {...props}>
      {children}
    </td>
  ),
};

interface MarkdownRendererProps {
  children: string;
}

/**
 * Markdown 渲染器。
 * 把 Markdown 文本渲染成带样式的 React 元素。
 *
 * 支持：
 *   - GitHub Flavored Markdown（表格、删除线、任务列表）
 *   - 代码块语法高亮（react-syntax-highlighter + Prism）
 *   - Warm Scholar 暖色调主题
 */
export default function MarkdownRenderer({ children }: MarkdownRendererProps) {
  return (
    <div className="text-sm leading-relaxed text-foreground">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={markdownComponents}
      >
        {children}
      </ReactMarkdown>
    </div>
  );
}
