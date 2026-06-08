"""统一 LLM 客户端 —— 一行配置切换 OpenAI / Anthropic / DeepSeek / GLM-4。

这是 agent-core 包的核心模块。对外暴露两个接口：
  client.chat("你好")              → 非流式对话，返回完整回复
  client.chat_stream("你好")       → 流式对话，逐字返回（打字机效果）

设计目标：
  换模型 = 只改 provider 字符串，业务代码完全不动。

支持的 Provider：
  - openai       → OpenAI SDK（gpt-4o, gpt-4o-mini 等）
  - anthropic    → Anthropic SDK（claude-sonnet-4-6 等）
  - deepseek     → OpenAI 兼容格式（deepseek-chat 等）
  - zhipu        → OpenAI 兼容格式（glm-4-flash, glm-4-plus 等）
  - moonshot     → OpenAI 兼容格式（moonshot-v1-8k 等）

用法示例：
  from study_agent.llm import LLMClient

  # 一行切换
  client = LLMClient(provider="anthropic")
  reply = client.chat("什么是递归？")
  for chunk in client.chat_stream("讲个故事"):
      print(chunk, end="")
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from typing import Any

from anthropic import Anthropic
from openai import OpenAI

from study_agent.config.settings import DEFAULT_MODELS, PROVIDER_CONFIGS
from study_agent.llm.retry import create_retry_decorator

logger = logging.getLogger(__name__)


class LLMClient:
    """切换模型只需改一行配置的 LLM 客户端。

    设计思路：
    - 对外暴露统一接口：chat() 和 chat_stream()
    - 对内根据 sdk_type 分派到 OpenAI SDK 或 Anthropic SDK
    - 每个 provider 的差异（base_url、model 名称）在初始化时消化掉
    - 所有 API 调用自动带重试（临时性错误）和日志
    """

    def __init__(
        self,
        provider: str,
        model: str | None = None,
        api_key: str | None = None,
        max_retries: int = 3,
    ):
        """创建 LLM 客户端。

        参数：
          provider    → 选哪家厂商（"openai" / "anthropic" / "deepseek" / "zhipu" / "moonshot"）
          model       → 具体模型名，不传则自动选该厂商的默认模型
          api_key     → API Key，不传则从环境变量读
          max_retries → API 调用失败时最多重试几次（默认 3）

        Raises:
          ValueError: provider 不支持 或 API Key 找不到
        """
        cfg = PROVIDER_CONFIGS.get(provider)
        if cfg is None:
            raise ValueError(
                f"不支持的 provider: {provider}。可选: {list(PROVIDER_CONFIGS.keys())}"
            )

        self.provider = provider
        self.sdk_type: str = cfg["sdk_type"]
        self.max_retries = max_retries

        # API Key —— 优先用传入的，否则读环境变量
        self.api_key = api_key or os.getenv(cfg["env_key"])
        if not self.api_key:
            raise ValueError(
                f"找不到 {provider} 的 API Key！\n"
                f"-> 请在 .env 中设置 {cfg['env_key']}=\"your-key\"\n"
                f'-> 或者创建 LLMClient 时传入 api_key="your-key"'
            )

        # 模型 —— 优先用传入的，否则选该厂商的默认款
        self.model = model or DEFAULT_MODELS.get(provider, "unknown")

        # 根据 SDK 类型创建底层客户端
        # 类型标注为 Any 是因为：两种 SDK 的接口不同，MyPy 无法静态判断
        # 运行时由 self.sdk_type 控制走哪个分支，不会出错
        if self.sdk_type == "anthropic":
            self._client: Any = Anthropic(api_key=self.api_key)
        else:
            self._client = OpenAI(api_key=self.api_key, base_url=cfg["base_url"])

        # 构建重试装饰器（从 llm.retry 模块导入）
        self._retry_decorator = create_retry_decorator(max_retries=max_retries)

    # ── 工厂方法：从环境变量创建 ──────────────────────
    @classmethod
    def from_env(cls) -> LLMClient:
        """从 LLM_PROVIDER 环境变量读取配置，一键创建客户端。

        用法：
          $env:LLM_PROVIDER="deepseek"   → 自动选 DeepSeek
          $env:LLM_PROVIDER="anthropic"  → 自动选 Claude
        """
        provider = os.getenv("LLM_PROVIDER", "deepseek")
        return cls(provider=provider)

    # ═══════════════════════════════════════════════════════
    # 公开接口
    # ═══════════════════════════════════════════════════════

    def chat(self, user_message: str, system: str | None = None) -> str:
        """发送一条消息，获取完整回复（非流式）。

        参数：
          user_message → 用户说的话
          system       → system prompt（AI 的行为准则），不传则不设

        返回：
          模型的完整回复文本
        """
        if self.sdk_type == "anthropic":
            return self._chat_anthropic(user_message, system)
        else:
            return self._chat_openai(user_message, system)

    def chat_stream(self, user_message: str, system: str | None = None) -> Iterator[str]:
        """发送一条消息，逐块获取回复（流式，打字机效果）。

        Iterator[str] 的意思是：调用方用 for 循环逐个接收文本块。
        每个块就是几个字，全部拼起来才是完整回复。
        """
        if self.sdk_type == "anthropic":
            yield from self._chat_stream_anthropic(user_message, system)
        else:
            yield from self._chat_stream_openai(user_message, system)

    # ═══════════════════════════════════════════════════════
    # 私有方法 —— OpenAI SDK 调用
    # ═══════════════════════════════════════════════════════

    def _chat_openai(self, user_message: str, system: str | None) -> str:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user_message})

        @self._retry_decorator
        def _call() -> Any:
            return self._client.chat.completions.create(
                model=self.model,
                messages=messages,
            )

        response = _call()
        return response.choices[0].message.content or ""

    def _chat_stream_openai(self, user_message: str, system: str | None) -> Iterator[str]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user_message})

        @self._retry_decorator
        def _call() -> Any:
            return self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
            )

        stream = _call()
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    # ═══════════════════════════════════════════════════════
    # 私有方法 —— Anthropic SDK 调用
    # ═══════════════════════════════════════════════════════

    def _chat_anthropic(self, user_message: str, system: str | None) -> str:
        kwargs: dict[str, object] = {
            "model": self.model,
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": user_message}],
        }
        if system:
            kwargs["system"] = system  # Anthropic 的 system 是独立参数，不是 message

        @self._retry_decorator
        def _call() -> Any:
            return self._client.messages.create(**kwargs)

        response = _call()
        return response.content[0].text if response.content else ""

    def _chat_stream_anthropic(self, user_message: str, system: str | None) -> Iterator[str]:
        kwargs = {
            "model": self.model,
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": user_message}],
        }
        if system:
            kwargs["system"] = system

        @self._retry_decorator
        def _call() -> Any:
            return self._client.messages.stream(**kwargs)

        with _call() as stream:
            yield from stream.text_stream
