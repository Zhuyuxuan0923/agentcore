"""工具基类 —— agent-core 的工具抽象层。

这个模块定义了"工具"是什么，以及 LLM 如何描述和使用工具。

什么是"工具"（Tool）？
  工具 = LLM 可以调用的外部功能。
  比如：搜索网页、读文件、发邮件、查数据库...
  LLM 本身只会"说话"，工具让它能"做事"。

  类比：LLM 是大脑，工具是手。大脑决定"我要查一下天气"，
  手（工具）去执行查天气的操作，把结果拿回来给大脑。

为什么需要一个基类？
  1. 所有工具都有共同的特征：名字、描述、参数定义、执行逻辑
  2. 基类强制每个工具实现这些——不会漏
  3. 后面做 Tool Calling 循环时，可以统一遍历"所有工具"

这个模块目前是地基。Week 2 Day 4 会在这个基础上实现完整的 tool calling 循环。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolParameter:
    """工具参数的定义。

    比如一个"搜索"工具的参数可能是：
      ToolParameter(name="query", type="string", description="搜索关键词", required=True)
    """

    name: str
    type: str  # "string" | "number" | "boolean" | "array" | "object"
    description: str
    required: bool = True
    default: str | None = None


@dataclass
class ToolDefinition:
    """工具的定义（元数据），用于告诉 LLM "我能做什么"。

    LLM 不认识你的代码，它只认识 JSON。
    所以每个工具需要把自己的能力"翻译"成 LLM 能理解的 JSON 描述。

    这个类就是那个"翻译官"。

    字段说明：
      name        → 工具的唯一标识名（如 "search_web"）
      description → 告诉 LLM 这个工具是干什么的（要写清楚，LLM 据此决定用不用它）
      parameters  → 工具需要什么参数，每个参数的类型和含义
    """

    name: str
    description: str
    parameters: list[ToolParameter] = field(default_factory=list)

    def to_openai_function(self) -> dict[str, object]:
        """把工具定义转成 OpenAI Function Calling 的 JSON 格式。

        OpenAI 要求的格式：
        {
          "type": "function",
          "function": {
            "name": "search_web",
            "description": "搜索互联网获取信息",
            "parameters": { ... JSON Schema ... }
          }
        }
        """
        properties = {}
        required = []

        for param in self.parameters:
            properties[param.name] = {
                "type": param.type,
                "description": param.description,
            }
            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }


class BaseTool(ABC):
    """所有工具的基类。每个具体的工具都必须继承它并实现 execute()。

    如何创建你自己的工具：
      class SearchWebTool(BaseTool):
          @property
          def definition(self) -> ToolDefinition:
              return ToolDefinition(
                  name="search_web",
                  description="搜索互联网获取实时信息",
                  parameters=[
                      ToolParameter(name="query", type="string",
                                    description="搜索关键词"),
                  ],
              )

          def execute(self, query: str) -> str:
              # 实际执行搜索的逻辑
              return f"关于 '{query}' 的搜索结果..."

    关键点：
      - definition 告诉 LLM "我能做什么"（元数据）
      - execute() 是实际干活的地方（业务逻辑）
      - LLM 决定调用哪个工具，但执行是在你的代码里
    """

    @property
    @abstractmethod
    def definition(self) -> ToolDefinition:
        """返回工具的定义（名字、描述、参数）。

        @property 的意思是：调用时不需要加括号，像属性一样访问：
          tool.definition  ← 不需要写 tool.definition()

        @abstractmethod 的意思是：子类必须实现这个方法，否则不能实例化。
          这是 Python 的"强制合同"——你想当工具，就必须填这张表。
        """
        ...

    @abstractmethod
    def execute(self, **kwargs: Any) -> str:
        """执行工具的逻辑。

        **kwargs 的意思是：接受任意数量的关键字参数。
        比如 execute(query="天气", limit=5) → kwargs = {"query": "天气", "limit": 5}

        每个子类根据自己的需要来解析参数。
        """
        ...
