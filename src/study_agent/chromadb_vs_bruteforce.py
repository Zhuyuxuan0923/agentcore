"""
暴力检索 vs ChromaDB — 对比实验
目的：直观感受"有向量数据库"和"没有"的差距

运行: poetry run python src/study_agent/chromadb_vs_bruteforce.py
"""

import math
import random
import time

import chromadb


def make_documents(n: int) -> tuple[list[str], list[list[float]]]:
    """生成 n 条模拟文档及其向量"""
    random.seed(42)
    docs = []
    vectors = []
    topics = ["技术", "美食", "旅游", "健身", "理财", "教育", "医疗", "法律", "设计", "管理"]
    for i in range(n):
        vec = [random.random() for _ in range(4)]
        docs.append(f"这是第 {i+1} 篇文档，内容是关于主题 {topics[i % 10]} 的讨论。")
        vectors.append(vec)
    return docs, vectors


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x**2 for x in a))
    norm_b = math.sqrt(sum(x**2 for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def brute_force_search(
    query: list[float],
    all_vectors: list[list[float]],
    all_docs: list[str],
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """暴力检索：逐个计算余弦相似度"""
    scores = []
    for i, vec in enumerate(all_vectors):
        scores.append((i, cosine_similarity(query, vec)))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [(all_docs[i], score) for i, score in scores[:top_k]]


def main():
    docs, vectors = make_documents(200)
    query_vec = [0.5, 0.5, 0.5, 0.5]

    # ── 方案 A：暴力检索 ──
    start = time.perf_counter()
    bf_results = brute_force_search(query_vec, vectors, docs, top_k=5)
    bf_time = time.perf_counter() - start

    # ── 方案 B：ChromaDB ──
    client = chromadb.EphemeralClient()
    # 指定 hnsw:space=cosine 使用余弦距离，和暴力检索保持一致
    collection = client.get_or_create_collection(
        name="vs_demo",
        metadata={"hnsw:space": "cosine"},
    )
    collection.add(
        documents=docs,
        embeddings=vectors,
        ids=[f"doc-{i}" for i in range(len(docs))],
    )

    start = time.perf_counter()
    chroma_results = collection.query(
        query_embeddings=[query_vec],
        n_results=5,
    )
    chroma_time = time.perf_counter() - start

    # ── 输出对比 ──
    print("=" * 55)
    print("  暴力检索 vs ChromaDB — 200 条文档检索对比")
    print("=" * 55)
    print()

    print("【方案 A：暴力检索】")
    print(f"  耗时: {bf_time * 1000:.2f} ms")
    print("  Top-5 结果:")
    for i, (doc, score) in enumerate(bf_results):
        print(f"    {i+1}. [相似度={score:.6f}] {doc[:45]}...")

    print()
    print("【方案 B：ChromaDB】")
    print(f"  耗时: {chroma_time * 1000:.2f} ms")
    print("  Top-5 结果:")
    for i, (doc_id, doc_text, dist) in enumerate(
        zip(
            chroma_results["ids"][0],
            chroma_results["documents"][0],
            chroma_results["distances"][0],
        )
    ):
        print(f"    {i+1}. [距离={dist:.6f}] {doc_text[:45]}...")

    print()
    print("【核心差异总结】")
    print("  代码量：   暴力检索 ~25 行，ChromaDB ~8 行")
    print("  元数据过滤：暴力检索 [无]，ChromaDB [有] (where 条件)")
    print("  数据持久化：暴力检索 [无]，ChromaDB [有] (PersistentClient)")
    print("  大规模加速：暴力检索 O(n)，ChromaDB O(log n) 索引")


if __name__ == "__main__":
    main()
