"""RAG 模块 -- 文档分块、向量化、检索、生成、引用。

从 Week 3 的 demo 脚本重构而来。
"""

from study_agent.rag.chunking import ChunkStrategy, chunk_document
from study_agent.rag.embedding import Embedder
from study_agent.rag.generator import CitationResult, RAGGenerator, parse_citations
from study_agent.rag.pipeline import RAGPipeline, RAGResult
from study_agent.rag.retriever import HybridRetriever, KeywordRetriever, VectorRetriever

__all__ = [
    "chunk_document",
    "ChunkStrategy",
    "Embedder",
    "VectorRetriever",
    "KeywordRetriever",
    "HybridRetriever",
    "RAGGenerator",
    "CitationResult",
    "parse_citations",
    "RAGPipeline",
    "RAGResult",
]
