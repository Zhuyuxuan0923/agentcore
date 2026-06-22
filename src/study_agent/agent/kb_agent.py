"""PersonalQA Agent -- 个人知识库问答 Agent 主逻辑。

这是整个 Week 4 的核心类，把前三周的技术栈整合到一个 Agent 里：
  - LLMClient (Week 1)     -> 调用 LLM
  - Prompt 模板 (Week 2)    -> system prompt 设计
  - RAG 管道 (Week 3)       -> 检索 + 生成 + 引用
  - ChromaDB (Week 3)       -> 向量数据库
  - 对话历史 (Week 4 新增)  -> SQLite 存储多轮对话

PersonalQAAgent 对外暴露三个核心方法：
  - upload(file_path, kb_name)  -> 上传文档并索引
  - chat(question, conv_id)     -> 提问并返回答案+引用
  - list_knowledge_bases()       -> 列出所有知识库
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from study_agent.agent.conversation import ConversationManager, Message
from study_agent.agent.knowledge_base import KnowledgeBaseManager
from study_agent.llm.client import LLMClient
from study_agent.rag.chunking import ChunkStrategy, chunk_document
from study_agent.rag.embedding import Embedder
from study_agent.rag.generator import RAGGenerator
from study_agent.rag.pipeline import RAGResult
from study_agent.rag.retriever import HybridRetriever


class PersonalQAAgent:
    """个人知识库问答 Agent。

    使用方式：
        agent = PersonalQAAgent()
        agent.upload("docs/公司制度.pdf", kb_name="公司制度")
        result = agent.chat("年假怎么申请？")
        print(result.answer)       # 带 [1] [2] 引用的答案
        print(result.citations)     # 引用详情
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        embedder: Embedder | None = None,
        chroma_dir: str = "data/chromadb",
        db_path: str = "data/conversations.db",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        top_k: int = 5,
        sim_threshold: float = 0.0,  # RRF 分数尺度与余弦相似度不同，0.6 会误杀
    ):
        self.llm = llm_client or LLMClient.from_env()
        self.embedder = embedder or Embedder()
        self.kb_manager = KnowledgeBaseManager(chroma_dir)
        self.conv_manager = ConversationManager(db_path)
        self.generator = RAGGenerator(self.llm)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.sim_threshold = sim_threshold
        self._current_kb: str | None = None
        self._current_conv_id: str | None = None

    # ── 文件上传 ──────────────────────────────────────────

    def upload(self, file_path: str, kb_name: str | None = None) -> dict[str, Any]:
        """上传文档到知识库。

        内部流程：读取文件 -> 解析为纯文本 -> 分块 -> Embedding向量化 -> 存入ChromaDB

        Args:
            file_path: 文件路径（支持 .pdf / .docx / .md / .txt）
            kb_name: 知识库名称，不指定则使用"默认知识库"

        Returns:
            {"kb_name": ..., "file_name": ..., "chunk_count": ...}
        """
        kb_name = kb_name or "默认知识库"
        filepath = Path(file_path)
        if not filepath.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        raw_text = self._parse_file(filepath)
        chunks = chunk_document(
            raw_text,
            strategy=ChunkStrategy.RECURSIVE,
            chunk_size=self.chunk_size,
            overlap=self.chunk_overlap,
        )

        chunk_texts = [c.text for c in chunks]
        embeddings = self.embedder.embed(chunk_texts)

        metadatas = [
            {
                "source_file": filepath.name,
                "kb_name": kb_name,
                "chunk_index": i,
                "char_start": c.start,
                "char_end": c.end,
            }
            for i, c in enumerate(chunks)
        ]

        self.kb_manager.add_documents(
            kb_name=kb_name,
            documents=chunk_texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        self._current_kb = kb_name
        return {
            "kb_name": kb_name,
            "file_name": filepath.name,
            "chunk_count": len(chunks),
        }

    # ── 问答 ──────────────────────────────────────────────

    def chat(self, question: str, conversation_id: str | None = None) -> RAGResult:
        """向当前知识库提问。

        内部流程：问题向量化 -> ChromaDB 检索 -> RRF 混合检索 -> Prompt 拼接 -> LLM 生成 -> 引用解析

        Args:
            question: 用户问题
            conversation_id: 可选，继续已有对话；不传则自动创建新对话

        Returns:
            RAGResult: 包含 answer / sources / citations
        """
        kb_name = self._current_kb or "默认知识库"
        if conversation_id:
            self._current_conv_id = conversation_id
        collection = self.kb_manager.get(kb_name)
        if collection is None:
            return RAGResult(
                question=question,
                answer="请先上传文档后再提问。",
            )

        # 获取知识库中的所有文档文本（用于关键词检索）
        kb_data = collection.get()
        corpus: list[str] = kb_data.get("documents") or []

        # 构建混合检索器
        retriever = HybridRetriever(
            corpus=corpus,
            collection=collection,
            embedder=self.embedder,
        )

        # Step 1: 检索
        hits = retriever.search(question, top_k=self.top_k)

        # Step 2: 相似度过滤
        hits = [h for h in hits if h["score"] >= self.sim_threshold]

        if not hits:
            result = RAGResult(
                question=question,
                answer="根据现有资料无法回答您的问题。",
            )
        else:
            chunk_texts = [h["text"] for h in hits]
            source_map = {i + 1: text for i, text in enumerate(chunk_texts)}

            # Step 3: 渲染 Prompt + 调用 LLM
            prompt = self.generator.prompt_manager.render(
                self.generator.template_name,
                role="知识库问答助手",
                chunks=chunk_texts,
                question=question,
            )
            raw_answer = self.llm.chat(prompt)

            from study_agent.rag.generator import parse_citations

            citation_result = parse_citations(raw_answer, source_map)

            result = RAGResult(
                question=question,
                answer=citation_result.answer,
                sources=[
                    {
                        "doc_id": h.get("doc_id", ""),
                        "text": h["text"],
                        "score": h["score"],
                        "metadata": h.get("metadata", {}),
                    }
                    for h in hits
                ],
                citations=citation_result.citations,
            )

        # 保存对话历史
        if self._current_conv_id is None:
            self._current_conv_id = self.conv_manager.create_conversation(
                knowledge_base_name=kb_name
            )

        self.conv_manager.add_message(
            self._current_conv_id,
            Message(role="user", content=question),
        )
        self.conv_manager.add_message(
            self._current_conv_id,
            Message(
                role="assistant",
                content=result.answer,
                citations=result.citations,
            ),
        )

        return result

    # ── 知识库管理 ────────────────────────────────────────

    def list_knowledge_bases(self) -> list[str]:
        return self.kb_manager.list_kbs()

    def switch_kb(self, kb_name: str) -> bool:
        if self.kb_manager.get(kb_name) is None:
            return False
        self._current_kb = kb_name
        self._current_conv_id = None  # 切换知识库后开始新对话
        return True

    def delete_kb(self, kb_name: str) -> bool:
        return self.kb_manager.delete(kb_name)

    def new_conversation(self) -> str:
        kb_name = self._current_kb or "默认知识库"
        self._current_conv_id = self.conv_manager.create_conversation(knowledge_base_name=kb_name)
        return self._current_conv_id

    def get_history(self, conv_id: str | None = None) -> list[Message]:
        cid = conv_id or self._current_conv_id
        if cid is None:
            return []
        return self.conv_manager.get_history(cid)

    def list_conversations(self) -> list[dict[str, Any]]:
        convs = self.conv_manager.list_conversations()
        return [
            {
                "id": c.id,
                "title": c.title,
                "kb_name": c.knowledge_base_name,
                "created_at": c.created_at,
            }
            for c in convs
        ]

    # ── 文件解析 ──────────────────────────────────────────

    @staticmethod
    def _parse_file(filepath: Path) -> str:
        """根据文件后缀选择对应的解析器。"""
        suffix = filepath.suffix.lower()

        if suffix == ".txt" or suffix == ".md":
            return filepath.read_text(encoding="utf-8")

        if suffix == ".docx":
            try:
                from docx import Document

                doc = Document(str(filepath))
                return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
            except ImportError:
                raise ImportError("请先安装 python-docx: poetry add python-docx")

        if suffix == ".pdf":
            try:
                from PyPDF2 import PdfReader

                reader = PdfReader(str(filepath))
                return "\n".join(page.extract_text() or "" for page in reader.pages)
            except ImportError:
                raise ImportError("请先安装 PyPDF2: poetry add PyPDF2")

        raise ValueError(f"不支持的文件格式: {suffix}，支持 .txt / .md / .docx / .pdf")
