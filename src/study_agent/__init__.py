"""Study Agent -- AI Agent 工程师学习项目.

这个包是 16 周学习计划的代码产出，包含以下核心模块:

  study_agent.config   -> 配置管理 (provider 信息、默认模型、环境变量)
  study_agent.llm      -> LLM 调用层 (统一客户端、重试机制)
  study_agent.prompt   -> Prompt 工程 (Jinja2 模板、Few-Shot、评测)
  study_agent.tools    -> 工具抽象层 (Tool Calling 基础设施)
  study_agent.rag      -> RAG 全链路 (分块、向量化、检索、生成、引用)
  study_agent.agent    -> PersonalQA Agent (整合上述模块的完整应用)
  study_agent.api       -> FastAPI 接口层 (/upload、/chat、/kb)

快速开始:
  from study_agent import PersonalQAAgent

  agent = PersonalQAAgent()
  agent.upload("docs/公司制度.pdf", kb_name="公司制度")
  result = agent.chat("年假怎么申请?")
  print(result.answer)  # 带 [1] [2] 引用的答案
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
