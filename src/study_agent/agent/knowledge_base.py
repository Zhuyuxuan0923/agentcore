"""多知识库管理 -- ChromaDB Collection 层封装。

一个知识库 = 一个 ChromaDB Collection。
不同知识库互相隔离——在"课程讲义"知识库里提问不会搜到"公司制度"的内容。

Collection 是什么？你可以把它想象成 SQL 里的一张"表"，
每个知识库是一张独立的表，检索只在当前表里搜。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chromadb


@dataclass
class KnowledgeBase:
    name: str
    doc_count: int = 0
    chunk_count: int = 0
    created_at: str = ""


class KnowledgeBaseManager:
    """多知识库管理器。

    管理 ChromaDB PersistentClient 下的多个 Collection，
    每个 Collection 对应一个知识库。

    使用方式：
        mgr = KnowledgeBaseManager()
        mgr.create("我的课程讲义")
        kb = mgr.get_or_create("公司制度")
        print(mgr.list_kbs())  # -> ["我的课程讲义", "公司制度"]
    """

    def __init__(self, persist_dir: str = "data/chromadb"):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))

    def create(self, name: str) -> chromadb.Collection:
        """创建新知识库（如果已存在则返回现有）。"""
        safe_name = self._sanitize_name(name)
        return self.client.get_or_create_collection(
            name=safe_name,
            metadata={
                "hnsw:space": "cosine",
                "display_name": name,  # 保存原始中文名称
            },
        )

    def get(self, name: str) -> chromadb.Collection | None:
        """获取知识库的 Collection，不存在返回 None。"""
        try:
            return self.client.get_collection(self._sanitize_name(name))
        except Exception:
            return None

    def get_or_create(self, name: str) -> chromadb.Collection:
        return self.create(name)

    def delete(self, name: str) -> bool:
        try:
            self.client.delete_collection(self._sanitize_name(name))
            return True
        except Exception:
            return False

    def list_kbs(self) -> list[str]:
        """列出所有知识库（返回原始中文名称）。"""
        collections = self.client.list_collections()
        names = []
        for col_name in collections:
            try:
                # list_collections() 返回的是 Collection 对象 name 属性（字符串），
                # 但某些 ChromaDB 版本返回 Collection 对象本身
                actual_name = col_name if isinstance(col_name, str) else col_name.name
                col = self.client.get_collection(actual_name)
                meta = col.metadata or {}
                names.append(meta.get("display_name", actual_name))
            except Exception:
                names.append(col_name if isinstance(col_name, str) else str(col_name))
        return sorted(names, key=str)

    def add_documents(
        self,
        kb_name: str,
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
    ) -> int:
        """向知识库添加文档块。返回添加后的总块数。"""
        collection = self.get_or_create(kb_name)
        if ids is None:
            existing_count = collection.count()
            ids = [
                f"{self._sanitize_name(kb_name)}-doc-{existing_count + i}"
                for i in range(len(documents))
            ]
        if metadatas is None:
            metadatas = [{} for _ in documents]

        collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )
        return collection.count()

    def get_stats(self, name: str) -> KnowledgeBase:
        collection = self.get(name)
        if collection is None:
            return KnowledgeBase(name=name, doc_count=0, chunk_count=0)
        return KnowledgeBase(
            name=name,
            chunk_count=collection.count(),
            doc_count=len(set(m.get("source_file", "") for m in collection.get()["metadatas"])),
        )

    @staticmethod
    def _sanitize_name(name: str) -> str:
        """清理知识库名称，确保 ChromaDB 兼容。

        ChromaDB 要求 Collection 名称只含 [a-zA-Z0-9._-]，3-512 字符。
        中文名称会被 hash 成 "kb-" + 8 位十六进制（碰撞概率极低）。
        """
        name = name.strip().replace(" ", "_")
        # 检查是否全是 ASCII 合法字符
        valid = all(c.isascii() and (c.isalnum() or c in "._-") for c in name)
        if valid and 3 <= len(name) <= 512:
            return name
        # 含中文或其他特殊字符 -> 用 hash
        import hashlib

        hash_suffix = hashlib.md5(name.encode()).hexdigest()[:8]
        return f"kb-{hash_suffix}"
