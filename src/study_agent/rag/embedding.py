"""Embedding 向量化封装 -- 三级自动降级策略。

策略 1（远程 API）：OpenAI text-embedding-3-small
  - 需要 OPENAI_API_KEY，1536 维高质量语义向量

策略 2（本地模型）：sentence-transformers all-MiniLM-L6-v2
  - 需要能从 HuggingFace 下载模型（约 80MB），384 维

策略 3（统计向量）：纯本地字符级特征向量
  - 零依赖，零下载，离线可用
  - 用字符 n-gram 频率构建 256 维向量，相似度计算退化为表面形式匹配
  - 效果不如前两种，但能让整个系统在没有网络时也能跑通

自动选择逻辑：
  1. 有 OPENAI_API_KEY -> 策略 1
  2. 能下载 HuggingFace 模型 -> 策略 2
  3. 否则 -> 策略 3
"""

from __future__ import annotations

import hashlib
import os

from study_agent.config.settings import get_config


class Embedder:
    """文本向量化器 -- 三级降级，保证在任何环境下都能工作。"""

    def __init__(
        self,
        model: str | None = None,
        use_local: bool | None = None,
    ):
        cfg = get_config()
        self.model = model or cfg.embedding_model
        self._strategy: str | None = None  # "api" | "huggingface" | "fallback"
        self._hf_model = None
        self._api_client = None

        # 判断策略
        has_openai = bool(os.getenv("OPENAI_API_KEY"))
        if use_local is False or has_openai:
            self._strategy = "api"
        elif use_local is True:
            self._strategy = "huggingface"
        else:
            # auto: 先试 huggingface，失败则 fallback
            self._strategy = "huggingface"

    def embed(self, texts: str | list[str]) -> list[list[float]]:
        """将文本转为向量。"""
        if isinstance(texts, str):
            texts = [texts]

        if self._strategy == "api":
            return self._embed_api(texts)
        elif self._strategy == "huggingface":
            return self._embed_huggingface(texts)
        else:
            return self._embed_fallback(texts)

    def embed_query(self, text: str) -> list[float]:
        """将单个查询文本转为向量（便捷方法）。"""
        return self.embed(text)[0]

    # ── 策略 1：远程 API ─────────────────────────────────

    def _embed_api(self, texts: list[str]) -> list[list[float]]:
        client = self._get_api_client()
        response = client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]

    def _get_api_client(self):
        if self._api_client is None:
            from openai import OpenAI

            self._api_client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url="https://api.openai.com/v1",
            )
        return self._api_client

    # ── 策略 2：本地 HuggingFace 模型 ─────────────────────

    def _embed_huggingface(self, texts: list[str]) -> list[list[float]]:
        if self._hf_model is None:
            try:
                from sentence_transformers import SentenceTransformer

                local_name = "all-MiniLM-L6-v2"
                print(f"[Embedder] 加载本地模型 {local_name}" f"（首次下载约 80MB，后续秒开）...")
                self._hf_model = SentenceTransformer(local_name)
            except Exception as e:
                print(f"[Embedder] 本地模型加载失败: {e}" f"\n[Embedder] 降级到纯本地统计向量")
                self._strategy = "fallback"
                return self._embed_fallback(texts)

        return self._hf_model.encode(texts, normalize_embeddings=True).tolist()

    # ── 策略 3：纯本地统计向量（零依赖，零下载）────────────

    def _embed_fallback(self, texts: list[str]) -> list[list[float]]:
        """用字符 n-gram 频率构建 256 维向量。

        原理：
          把文本切成连续的字符片段（1-gram + 2-gram + 3-gram），
          对每个片段做 hash 映射到 256 个桶中的一个，
          统计每个桶的出现频率，得到 256 维向量。

        这不是真正的语义向量——它只看"用了哪些字符组合"。
        但在文档集合内部，主题相近的文本确实会共享更多字符组合，
        所以有一定实用性。

        优点：零依赖、零下载、永远可用。
        缺点：无法跨语言泛化、对同义词无效。
        """
        dim = 256
        result = []
        for text in texts:
            vec = [0.0] * dim
            # 1-gram, 2-gram, 3-gram
            for n in (1, 2, 3):
                for i in range(len(text) - n + 1):
                    gram = text[i : i + n]
                    bucket = int(hashlib.md5(gram.encode()).hexdigest(), 16) % dim
                    vec[bucket] += 1.0
            # 归一化
            total = sum(v * v for v in vec) ** 0.5
            if total > 0:
                vec = [v / total for v in vec]
            result.append(vec)
        return result
