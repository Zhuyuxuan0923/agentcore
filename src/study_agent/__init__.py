"""Study Agent — AI Agent 工程师学习项目。

这个包是 16 周学习计划的代码产出，包含三个核心子模块：

  study_agent.config   → 配置管理（provider 信息、默认模型、环境变量）
  study_agent.llm      → LLM 调用层（统一客户端、重试机制）
  study_agent.tools    → 工具抽象层（工具定义、基类）

快速开始：
  from study_agent import LLMClient

  client = LLMClient(provider="deepseek")
  reply = client.chat("什么是机器学习？")
  print(reply)
"""

__version__ = "0.1.0"

# 从子模块导出最常用的类，让用户不需要记住它们在哪
from study_agent.config import DEFAULT_MODELS, PROVIDER_CONFIGS, LLMConfig
from study_agent.llm import LLMClient, create_retry_decorator, should_retry
from study_agent.tools import BaseTool, ToolDefinition, ToolParameter

__all__ = [
    # 版本
    "__version__",
    # config
    "PROVIDER_CONFIGS",
    "DEFAULT_MODELS",
    "LLMConfig",
    # llm
    "LLMClient",
    "create_retry_decorator",
    "should_retry",
    # tools
    "BaseTool",
    "ToolDefinition",
    "ToolParameter",
]
