"""端到端 RAG 管道 -- 从 w3d5_generation_demo.py 的 rag_pipeline() 重构。

检索 -> 增强(prompt拼装) -> 生成 -> 引用解析，一个函数搞定全流程。
Week 4 的 Agent 直接调用这个管道。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from study_agent.llm.client import LLMClient
from study_agent.rag.embedding import Embedder
from study_agent.rag.generator import RAGGenerator
from study_agent.rag.retriever import HybridRetriever


@dataclass
class RAGResult:
    """RAG 管道的一次完整结果。"""

    question: str
    answer: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)


class RAGPipeline:
    """端到端 RAG 管道。

    封装了 检索 -> 相似度过滤 -> Prompt 渲染 -> LLM 生成 -> 引用解析 全流程。

    使用方式:
        pipeline = RAGPipeline(llm, embedder, retriever)
        result = pipeline.query("年假怎么申请？")
        print(result.answer)       # 带引用的答案
        print(result.citations)     # 引用详情
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        embedder: Embedder | None = None,
        retriever: HybridRetriever | None = None,
        generator: RAGGenerator | None = None,
        top_k: int = 5,
        sim_threshold: float = 0.0,  # RRF 模式下用低阈值或不设阈值
    ):
        self.llm = llm_client or LLMClient.from_env()
        self.embedder = embedder or Embedder()
        self.retriever = retriever
        self.generator = generator or RAGGenerator(self.llm)
        self.top_k = top_k
        self.sim_threshold = sim_threshold

    def query(self, question: str) -> RAGResult:
        """执行一次完整的 RAG 查询。

        Args:
            question: 用户问题

        Returns:
            RAGResult: 包含答案、来源、引用详情的完整结果
        """
        # Step 1: 检索
        if self.retriever is None:
            return RAGResult(
                question=question,
                answer="[系统错误] 未配置检索器，请先上传文档。",
            )

        all_hits = self.retriever.search(question, top_k=self.top_k)

        # Step 2: 相似度过滤
        hits = [h for h in all_hits if h["score"] >= self.sim_threshold]

        if not hits:
            return RAGResult(
                question=question,
                answer="根据现有资料无法回答您的问题。",
                sources=[],
                citations=[],
            )

        # Step 3: 构建 source_map（编号 -> 原文）
        chunk_texts = [h["text"] for h in hits]
        source_map = {i + 1: text for i, text in enumerate(chunk_texts)}

        # Step 4: 用 Jinja2 模板渲染 Prompt
        prompt = self.generator.prompt_manager.render(
            self.generator.template_name,
            role="知识库问答助手",
            chunks=chunk_texts,
            question=question,
        )

        # Step 5: LLM 生成
        raw_answer = self.llm.chat(prompt)

        # Step 6: 解析引用
        from study_agent.rag.generator import parse_citations

        citation_result = parse_citations(raw_answer, source_map)

        return RAGResult(
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
