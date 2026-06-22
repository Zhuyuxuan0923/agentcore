"""检索策略 -- 从 w3d4_retrieval_demo.py 重构。

三种检索器：
  - VectorRetriever: ChromaDB 向量语义检索
  - KeywordRetriever: TF-IDF 关键词检索
  - HybridRetriever: RRF 混合检索（向量 + 关键词融合）

RRF (Reciprocal Rank Fusion): 用排名替代分数，绕过"向量相似度和TF-IDF分数不可比"的问题。
"""

from __future__ import annotations

import math
from typing import Any

import chromadb

from study_agent.rag.embedding import Embedder


class VectorRetriever:
    """基于 ChromaDB 的语义向量检索。

    用 Embedding 模型将查询转为向量，在 ChromaDB 中找最相似的文档块。
    擅长：语义相近的表达（"工资" -> "薪酬"），同义词替换，跨语言
    弱项：专有名词精确匹配（"Wi-Fi 密码" 可能被语义相近的文档淹没）
    """

    def __init__(
        self,
        collection: chromadb.Collection,
        embedder: Embedder | None = None,
    ):
        self.collection = collection
        self.embedder = embedder or Embedder()

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """向量检索，返回 Top-K 文档块。"""
        query_vec = self.embedder.embed_query(query)
        query_vec = self.embedder.embed_query(query)
        raw_results = self.collection.query(
            query_embeddings=[query_vec],  # type: ignore[arg-type]
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        hits: list[dict[str, Any]] = []
        ids_list: list[list[str]] = raw_results.get("ids", [[]]) or [[]]
        docs_list: list[list[str]] = raw_results.get("documents", [[]]) or [[]]
        metas_list: list[list[dict[str, Any]]] = raw_results.get("metadatas", [[]]) or [[]]
        dists_list: list[list[float]] = raw_results.get("distances", [[]]) or [[]]
        if not ids_list or not ids_list[0]:
            return []

        for i, doc_id in enumerate(ids_list[0]):
            doc_text = docs_list[0][i] if docs_list[0] else ""
            metadata = metas_list[0][i] if metas_list[0] else {}
            dist = dists_list[0][i] if dists_list[0] else 0.0
            hits.append(
                {
                    "doc_id": doc_id,
                    "text": doc_text,
                    "score": round(1.0 - dist, 4),
                    "metadata": metadata,
                    "source": "vector",
                }
            )
        return hits


class KeywordRetriever:
    """基于 TF-IDF 的关键词检索。

    不依赖 Embedding 模型，只统计"词在文档中出现的频率"。
    TF (Term Frequency): 词在这篇文档里出现了几次？
    IDF (Inverse Document Frequency): 词在整个语料库中多罕见？

    擅长：精确匹配专有名词、数字、代码
    弱项：同义词替换、语义改写
    """

    def __init__(self, corpus: list[str] | None = None):
        self.corpus = corpus or []
        self.doc_tokens = [self._tokenize(doc) for doc in self.corpus]
        self.idf = self._build_idf()

    def add_documents(self, texts: list[str]):
        """动态添加文档到关键词索引。"""
        self.corpus.extend(texts)
        new_tokens = [self._tokenize(doc) for doc in texts]
        self.doc_tokens.extend(new_tokens)
        self.idf = self._build_idf()

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """简单中文分词：1-gram + 2-gram 字符级切分。

        这不是最好的分词方式，但零依赖。生产环境可换 jieba。
        """
        cleaned = ""
        for ch in text:
            if ch.isalnum() or "一" <= ch <= "鿿":
                cleaned += ch
        tokens = []
        for i in range(len(cleaned)):
            tokens.append(cleaned[i])
            if i < len(cleaned) - 1:
                tokens.append(cleaned[i : i + 2])
        return tokens

    def _build_idf(self) -> dict[str, float]:
        """构建 IDF 词典。"""
        idf = {}
        n_docs = len(self.doc_tokens) if self.doc_tokens else 1
        vocab = set(t for tokens in self.doc_tokens for t in tokens)
        for word in sorted(vocab):
            doc_count = sum(1 for tokens in self.doc_tokens if word in tokens)
            idf[word] = math.log((n_docs + 1) / (doc_count + 1)) + 1.0
        return idf

    def _tfidf_vector(self, tokens: list[str]) -> dict[str, float]:
        """计算 token 序列的 TF-IDF 向量。"""
        total = len(tokens) if tokens else 1
        tf: dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        vec = {}
        for word, count in tf.items():
            if word in self.idf:
                vec[word] = (count / total) * self.idf[word]
        return vec

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """关键词检索，返回 Top-K 文档块。"""
        if not self.corpus:
            return []

        q_tokens = self._tokenize(query)
        q_vec = self._tfidf_vector(q_tokens)

        scores = []
        for idx, doc_tokens in enumerate(self.doc_tokens):
            d_vec = self._tfidf_vector(doc_tokens)
            dot = sum(q_vec.get(w, 0) * d_vec.get(w, 0) for w in set(q_vec) | set(d_vec))
            norm_q = math.sqrt(sum(v**2 for v in q_vec.values()))
            norm_d = math.sqrt(sum(v**2 for v in d_vec.values()))
            sim = dot / (norm_q * norm_d) if norm_q > 0 and norm_d > 0 else 0.0
            scores.append((idx, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        hits = []
        for idx, score in scores[:top_k]:
            if score > 0:
                hits.append(
                    {
                        "doc_id": f"kw-{idx}",
                        "text": self.corpus[idx],
                        "score": round(score, 4),
                        "metadata": {},
                        "source": "keyword",
                    }
                )
        return hits


def reciprocal_rank_fusion(
    vector_hits: list[dict[str, Any]],
    keyword_hits: list[dict[str, Any]],
    k: int = 60,
    weight_vector: float = 0.5,
    weight_keyword: float = 0.5,
) -> list[dict[str, Any]]:
    """RRF 融合两个检索器结果。

    每个文档在两个检索器中各有一个排名(rank)，
    RRF 分数 = weight / (k + rank)，k 是平滑参数防止 rank=1 权重过高。

    为什么用 RRF 而不是简单加权分数？
    向量分数 [0,1] 和 TF-IDF 分数 [0,1] 不可比——0.85 vs 0.85 含义不同。
    RRF 用排名替代分数，绕过尺度不同的问题。
    """
    merged: dict[str, dict[str, Any]] = {}

    for rank, hit in enumerate(vector_hits, start=1):
        doc_id = hit["doc_id"]
        merged[doc_id] = {
            **hit,
            "score_vector": hit["score"],
            "rank_vector": rank,
            "score_keyword": 0.0,
            "rank_keyword": 999,
            "rrf": 0.0,
        }

    for rank, hit in enumerate(keyword_hits, start=1):
        doc_id = hit["doc_id"]
        if doc_id in merged:
            merged[doc_id]["score_keyword"] = hit["score"]
            merged[doc_id]["rank_keyword"] = rank
        else:
            merged[doc_id] = {
                **hit,
                "score_vector": 0.0,
                "rank_vector": 999,
                "score_keyword": hit["score"],
                "rank_keyword": rank,
                "rrf": 0.0,
            }

    for info in merged.values():
        info["rrf"] = weight_vector / (k + info["rank_vector"]) + weight_keyword / (
            k + info["rank_keyword"]
        )

    result = sorted(merged.values(), key=lambda x: x["rrf"], reverse=True)
    for hit in result:
        hit["score"] = round(hit["rrf"], 6)
        hit["source"] = "hybrid"
    return result


class HybridRetriever:
    """混合检索器：向量检索 + 关键词检索 -> RRF 融合。

    融合后兼顾：
    - 向量检索的语义理解（同义词、改写）
    - 关键词检索的精确匹配（专有名词、数字）
    """

    def __init__(
        self,
        corpus: list[str],
        collection: chromadb.Collection,
        embedder: Embedder | None = None,
    ):
        self.vector_retriever = VectorRetriever(collection, embedder)
        self.keyword_retriever = KeywordRetriever(corpus)

    def search(
        self,
        query: str,
        top_k: int = 5,
        weight_vector: float = 0.5,
        weight_keyword: float = 0.5,
    ) -> list[dict[str, Any]]:
        vec_hits = self.vector_retriever.search(query, top_k=top_k * 2)
        kw_hits = self.keyword_retriever.search(query, top_k=top_k * 2)

        fused = reciprocal_rank_fusion(
            vec_hits,
            kw_hits,
            weight_vector=weight_vector,
            weight_keyword=weight_keyword,
        )
        return fused[:top_k]

    def add_documents(self, texts: list[str]):
        """同步添加文档到关键词索引（向量索引由外部 ChromaDB 管理）。"""
        self.keyword_retriever.add_documents(texts)
