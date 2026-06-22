"""RAG 生成与引用 -- 从 w3d5_generation_demo.py 重构。

核心职责：
  1. 用 Jinja2 模板渲染 RAG Prompt
  2. 调用 LLM 生成带引用的答案
  3. 解析引用编号 [N] 映射回原文档
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from study_agent.llm.client import LLMClient
from study_agent.prompt.templates import PromptManager


@dataclass
class CitationResult:
    """带引用的生成结果。"""

    answer: str
    citations: list[dict[str, Any]] = field(default_factory=list)
    # citations = [{"number": 1, "text": "原文...", "metadata": {...}}, ...]


def parse_citations(answer: str, source_map: dict[int, str]) -> CitationResult:
    """从 LLM 回复中提取引用编号 [N]，映射回原文档。

    Args:
        answer: LLM 生成的文本（含 [1] [2] 等标注）
        source_map: 编号 -> 原文档内容的映射

    Returns:
        CitationResult: 包含原答案和引用详情列表
    """
    cited_numbers = set()
    for match in re.finditer(r"\[(\d+)\]", answer):
        cited_numbers.add(int(match.group(1)))

    citations = []
    for num in sorted(cited_numbers):
        if num in source_map:
            citations.append(
                {
                    "number": num,
                    "text": source_map[num],
                }
            )

    return CitationResult(answer=answer, citations=citations)


class RAGGenerator:
    """RAG 生成器：将检索到的文档块 + 用户问题拼成 Prompt，调用 LLM 生成答案。"""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        template_dir: str = "src/study_agent/prompt/templates",
        template_name: str = "rag_generation",
    ):
        self.llm = llm_client or LLMClient.from_env()
        self.prompt_manager = PromptManager(template_dir)
        self.template_name = template_name

    def generate(
        self,
        question: str,
        chunks: list[str],
        role: str = "知识库问答助手",
    ) -> str:
        """根据检索到的文档块生成答案。

        Args:
            question: 用户问题
            chunks: 检索到的文档块列表
            role: Agent 角色描述

        Returns:
            LLM 生成的答案（含 [N] 引用标注）
        """
        prompt = self.prompt_manager.render(
            self.template_name,
            role=role,
            chunks=chunks,
            question=question,
        )
        return self.llm.chat(prompt)

    def generate_with_citations(
        self,
        question: str,
        chunks: list[str],
        role: str = "知识库问答助手",
    ) -> CitationResult:
        """生成答案并解析引用。

        Returns:
            CitationResult: 包含答案和引用详情
        """
        answer = self.generate(question, chunks, role)
        # source_map: 给每个 chunk 分配一个编号（从 1 开始），
        # 编号对应 LLM 在回答中使用的 [1] [2] 标注
        source_map = {i + 1: chunk for i, chunk in enumerate(chunks)}
        return parse_citations(answer, source_map)
