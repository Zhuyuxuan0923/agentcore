"""tools 模块 —— agent-core 的工具抽象层。

从这里可以拿到：
- BaseTool       → 所有工具的基类（继承它来创建新工具）
- ToolDefinition → 工具的定义/元数据（告诉 LLM "我能做什么"）
- ToolParameter  → 工具参数的定义
"""

from study_agent.tools.base import BaseTool, ToolDefinition, ToolParameter

__all__ = [
    "BaseTool",
    "ToolDefinition",
    "ToolParameter",
]
