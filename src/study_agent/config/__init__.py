"""config 模块 —— agent-core 的配置中心。

从这里可以拿到：
- PROVIDER_CONFIGS → 所有支持的模型厂商信息
- DEFAULT_MODELS   → 每家厂商的默认模型
- LLMConfig         → 一次 LLM 调用的完整配置（推荐使用）
- get_api_key()     → 读取某家厂商的 API Key
- list_available_providers() → 列出已配 Key 的厂商
"""

from study_agent.config.settings import (
    DEFAULT_MODELS,
    PROVIDER_CONFIGS,
    LLMConfig,
    get_api_key,
    get_default_provider,
    list_available_providers,
)

__all__ = [
    "PROVIDER_CONFIGS",
    "DEFAULT_MODELS",
    "LLMConfig",
    "get_api_key",
    "get_default_provider",
    "list_available_providers",
]
