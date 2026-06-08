"""统一 LLM 客户端 — 一行配置切换 OpenAI / Anthropic / DeepSeek / GLM-4。

🤖 AI 生成骨架，✍️ 你重点理解"抽象层怎么设计的"。

设计目标：
  换模型 = 只改 provider 和 model 两个字符串，业务代码完全不动。

为什么需要这个？
  1. 每家 SDK 的 API 不一样（尤其 Anthropic vs 其他）
  2. 项目中不想到处写 if-else 判断"这是哪家模型"
  3. 面试必问："你怎么设计的 LLM 抽象层？"

支持的 Provider：
  - openai       → OpenAI SDK（gpt-4o, gpt-4o-mini 等）
  - anthropic    → Anthropic SDK（claude-sonnet-4-6 等）
  - deepseek     → OpenAI 兼容格式（deepseek-chat 等）
  - zhipu        → OpenAI 兼容格式（glm-4-flash, glm-4-plus 等）
  - moonshot     → OpenAI 兼容格式（moonshot-v1-8k 等）
  - custom       → 任意 OpenAI 兼容的 API

用法示例：
  from study_agent.llm_client import LLMClient

  # 一行切换
  client = LLMClient.from_env()             # 从环境变量读配置
  client = LLMClient(provider="anthropic")  # 或直接指定

  # 统一接口
  reply = client.chat("什么是递归？")                      # 非流式
  reply = client.chat("你好", system="你是助手")           # 带 system prompt
  for chunk in client.chat_stream("讲个故事"):              # 流式
      print(chunk, end="")
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator

from anthropic import Anthropic
from anthropic import (
    APIConnectionError as AnthropicAPIConnectionError,
)
from anthropic import (
    APIStatusError as AnthropicAPIStatusError,
)
from anthropic import (
    AuthenticationError as AnthropicAuthenticationError,
)
from anthropic import (
    BadRequestError as AnthropicBadRequestError,
)
from anthropic import (
    RateLimitError as AnthropicRateLimitError,
)
from openai import (
    APIConnectionError as OpenaiAPIConnectionError,
)
from openai import (
    APIStatusError as OpenaiAPIStatusError,
)
from openai import (
    AuthenticationError as OpenaiAuthenticationError,
)
from openai import (
    BadRequestError as OpenaiBadRequestError,
)
from openai import (
    OpenAI,
)
from openai import (
    RateLimitError as OpenaiRateLimitError,
)
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════
# ① 配置字典 —— 每家 provider 的默认参数
# ═══════════════════════════════════════════════════════════
#
# 这个字典就是一个"通讯录"，记录每家模型公司的地址和兼容方式。
# 新增一家 provider 只需在这里加一行。
#
# 字段说明：
#   base_url   → API 的网址，发请求的目的地
#   sdk_type   → "openai"（用 OpenAI SDK 调）还是 "anthropic"（用 Anthropic SDK 调）
#   env_key    → 读哪个环境变量获取 API Key

PROVIDER_CONFIGS = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "sdk_type": "openai",
        "env_key": "OPENAI_API_KEY",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "sdk_type": "openai",
        "env_key": "DEEPSEEK_API_KEY",
    },
    "zhipu": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "sdk_type": "openai",
        "env_key": "ZHIPU_API_KEY",
    },
    "moonshot": {
        "base_url": "https://api.moonshot.cn/v1",
        "sdk_type": "openai",
        "env_key": "MOONSHOT_API_KEY",
    },
    "anthropic": {
        "base_url": None,  # Anthropic SDK 不需要 base_url（官方 SDK 自己知道地址）
        "sdk_type": "anthropic",
        "env_key": "ANTHROPIC_API_KEY",
    },
}


# ═══════════════════════════════════════════════════════════
# ② LLMClient 类 —— 统一入口
# ═══════════════════════════════════════════════════════════
class LLMClient:
    """切换模型只需改一行配置的 LLM 客户端。

    设计思路：
    - 对外暴露统一接口：chat() 和 chat_stream()
    - 对内根据 sdk_type 分派到两个 SDK 的具体调用
    - 每个 provider 的差异（base_url、model 名称）在初始化时消化掉
    """

    def __init__(
        self,
        provider: str,
        model: str | None = None,
        api_key: str | None = None,
        max_retries: int = 3,
    ):
        """
        参数说明：
          provider    → 选哪家厂商（"openai" / "anthropic" / "deepseek" / "zhipu" / "moonshot"）
          model       → 具体模型名，不传则自动选该厂商的默认模型
          api_key     → API Key，不传则从环境变量读
          max_retries → API 调用失败时最多重试几次（默认 3）
                         Key 错误不会重试（重试也没用）
                         被限流或网络抖动时会自动重试
        """
        cfg = PROVIDER_CONFIGS.get(provider)
        if cfg is None:
            raise ValueError(
                f"不支持的 provider: {provider}。可选: {list(PROVIDER_CONFIGS.keys())}"
            )

        self.provider = provider
        self.sdk_type = cfg["sdk_type"]
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
        if model:
            self.model = model
        else:
            self.model = DEFAULT_MODELS.get(provider, "unknown")

        # 根据 SDK 类型创建底层客户端
        if self.sdk_type == "anthropic":
            self._client = Anthropic(api_key=self.api_key)
        else:
            # 所有 OpenAI 兼容的 provider 都走这里
            self._client = OpenAI(api_key=self.api_key, base_url=cfg["base_url"])

        # ── 构建重试装饰器 ──
        # 为什么在此创建？因为 max_retries 是实例变量，不同 client 可以不同。
        #
        # 重试判断逻辑（retry_if_exception）：
        #   用一个自定义函数决定"该不该重试"，而不是简单按类型匹配。
        #   原因：AuthenticationError 是 APIStatusError 的子类，
        #   如果用 retry_if_exception_type(APIStatusError)，401 错误也会被重试——这是浪费。
        #
        #   所以这里写一个判断函数：只对"临时性"错误重试，永久性错误直接放弃。
        def _should_retry(exc: BaseException) -> bool:
            # 永久性错误 —— 重试也治不好
            if isinstance(exc, (OpenaiAuthenticationError, AnthropicAuthenticationError)):
                return False
            if isinstance(exc, (OpenaiBadRequestError, AnthropicBadRequestError)):
                return False
            # 临时性错误 —— 重试可能治好
            if isinstance(
                exc,
                (
                    OpenaiAPIConnectionError,
                    AnthropicAPIConnectionError,
                    OpenaiRateLimitError,
                    AnthropicRateLimitError,
                    OpenaiAPIStatusError,
                    AnthropicAPIStatusError,
                ),
            ):
                return True
            # 不认识的异常 —— 保守起见不重试
            return False

        self._retry_decorator = retry(
            wait=wait_exponential(multiplier=1, min=1, max=30),
            stop=stop_after_attempt(max_retries + 1),  # +1 因为第一次不算 retry
            retry=retry_if_exception(_should_retry),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )

    # ── 静态工厂方法：从环境变量读 provider ──────────
    @staticmethod
    def from_env() -> LLMClient:
        """从 LLM_PROVIDER 环境变量读取当前要用的厂商。

        用法：
          $env:LLM_PROVIDER="deepseek"   → 自动选 DeepSeek
          $env:LLM_PROVIDER="anthropic"  → 自动选 Claude
        """
        provider = os.getenv("LLM_PROVIDER", "deepseek")
        return LLMClient(provider=provider)

    # ── 核心接口 1：非流式对话 ────────────────────────
    def chat(self, user_message: str, system: str | None = None) -> str:
        """发送一条消息，获取完整回复。

        参数：
          user_message → 用户说的话
          system       → system prompt（AI 的行为准则），不传就不设

        返回：
          模型的完整回复文本
        """
        if self.sdk_type == "anthropic":
            return self._chat_anthropic(user_message, system)
        else:
            return self._chat_openai(user_message, system)

    # ── 核心接口 2：流式对话 ──────────────────────────
    def chat_stream(self, user_message: str, system: str | None = None) -> Iterator[str]:
        """发送一条消息，逐块获取回复（打字机效果）。

        Iterator[str] 的意思是：调用方用 for 循环逐个接收文本块。
        每个块就是几个字，全部拼起来才是完整回复。
        """
        if self.sdk_type == "anthropic":
            yield from self._chat_stream_anthropic(user_message, system)
        else:
            yield from self._chat_stream_openai(user_message, system)

    # ── 私有方法：OpenAI SDK 调用（带自动重试）───────
    def _chat_openai(self, user_message: str, system: str | None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user_message})

        @self._retry_decorator
        def _call():
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
        def _call():
            return self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
            )

        stream = _call()
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    # ── 私有方法：Anthropic SDK 调用（带自动重试）─────
    def _chat_anthropic(self, user_message: str, system: str | None) -> str:
        kwargs = {
            "model": self.model,
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": user_message},
            ],
        }
        if system:
            kwargs["system"] = system  # ← 注意！Anthropic 的 system 是独立参数

        @self._retry_decorator
        def _call():
            return self._client.messages.create(**kwargs)

        response = _call()
        # Anthropic 的 content 是一个 content block 列表
        return response.content[0].text if response.content else ""

    def _chat_stream_anthropic(self, user_message: str, system: str | None) -> Iterator[str]:
        kwargs = {
            "model": self.model,
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": user_message},
            ],
        }
        if system:
            kwargs["system"] = system

        @self._retry_decorator
        def _call():
            return self._client.messages.stream(**kwargs)

        with _call() as stream:
            yield from stream.text_stream


# ═══════════════════════════════════════════════════════════
# ③ 各 Provider 的默认模型
# ═══════════════════════════════════════════════════════════
DEFAULT_MODELS = {
    "openai": "gpt-4o",
    "anthropic": "claude-sonnet-4-6",
    "deepseek": "deepseek-chat",
    "zhipu": "glm-4-flash",
    "moonshot": "moonshot-v1-8k",
}


# ═══════════════════════════════════════════════════════════
# ④ 演示代码 —— 对比三家模型的回复
# ═══════════════════════════════════════════════════════════
def demo_multi_provider():
    """用同一个问题问三家不同的模型，对比回复。

    这是 LLMClient 最核心的价值——同样的代码，只改 provider 字符串。
    """
    import sys

    sys.stdout.reconfigure(encoding="utf-8")

    question = "用一句话解释什么是 API"
    providers_to_try = []

    # 检查哪些 provider 有 Key
    for name in ["deepseek", "zhipu", "anthropic", "openai"]:
        cfg = PROVIDER_CONFIGS[name]
        if os.getenv(cfg["env_key"]):
            providers_to_try.append(name)

    if not providers_to_try:
        print("[WARNING] 没找到任何 API Key，演示无法运行")
        print("-> 请在 .env 文件中至少设置一个 provider 的 API Key")
        print('-> 例如 DEEPSEEK_API_KEY="sk-your-key"')
        return

    print(f"发现 {len(providers_to_try)} 家有 Key，开始对比：{providers_to_try}\n")

    for provider in providers_to_try:
        print(f"--- {provider} ---")
        try:
            client = LLMClient(provider=provider)
            reply = client.chat(question)
            print(f"  回复: {reply}")
        except Exception as e:
            print(f"  [ERROR] {type(e).__name__}: {e}")
        print()


# ═══════════════════════════════════════════════════════════
# ✍️ 动手区：创建一个你自己的 client 并测试
# ═══════════════════════════════════════════════════════════
def your_turn():
    """✍️ 学生动手区 —— 创建你的第一个 LLMClient。

    任务：
    1. 从环境变量创建 LLMClient（假设你已有 DEEPSEEK_API_KEY）
    2. 用 chat() 问一个问题
    3. 用 chat_stream() 流式输出
    4. 尝试带 system prompt 的对话
    """
    import sys

    sys.stdout.reconfigure(encoding="utf-8")

    print("\n" + "=" * 60)
    print("✍️ 动手区：创建你的 LLMClient")
    print("=" * 60)

    # 提示：用 from_env() 会自动读取 LLM_PROVIDER 环境变量
    # 如果没有设置 LLM_PROVIDER，默认用 deepseek

    try:
        client = LLMClient.from_env()
        print(f"\n使用中的 Provider: {client.provider}")
        print(f"使用中的 Model: {client.model}")
    except ValueError as e:
        print(f"\n创建失败: {e}")
        print("\n请先确保至少有一家 provider 的 API Key 在 .env 中。")
        return

    # 任务 1：基础对话
    print("\n--- 任务 1: 基础对话 ---")
    reply = client.chat("用一句话解释什么是机器学习")
    print(f"回复: {reply}")

    # 任务 2：带 system prompt
    print("\n--- 任务 2: 带 system prompt ---")
    reply = client.chat(
        user_message="解释什么是递归",
        system="你是一个小学老师，用最通俗的语言解释概念，不超过3句话。",
    )
    print(f"回复: {reply}")

    # 任务 3：流式输出
    print("\n--- 任务 3: 流式输出 ---")
    print("流式: ", end="", flush=True)
    for chunk in client.chat_stream("用三句话介绍 Python"):
        print(chunk, end="", flush=True)
    print()


# ==================== 主程序 ====================
if __name__ == "__main__":
    your_turn()
    # 如果你有多个 provider 的 Key，取消下面这行注释做对比：
    # demo_multi_provider()
