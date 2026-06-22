"""FastAPI 应用工厂 -- PersonalQA Agent 的 HTTP 入口。

启动方式:
    poetry run uvicorn study_agent.api.app:create_app --reload --host 0.0.0.0 --port 8000

然后访问:
    http://localhost:8000/docs  -> 自动生成的 API 文档（Swagger UI）
    http://localhost:8000/      -> 健康检查
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from study_agent.agent.kb_agent import PersonalQAAgent

_agent: PersonalQAAgent | None = None


def get_agent() -> PersonalQAAgent:
    """获取全局 Agent 单例（延迟初始化）。"""
    global _agent
    if _agent is None:
        _agent = PersonalQAAgent()
    return _agent


def create_app() -> FastAPI:
    """创建并配置 FastAPI 应用。"""
    app = FastAPI(
        title="PersonalQA -- 个人知识库问答 Agent",
        description="上传文档，提问，获得带引用的答案。Week 4 作品。",
        version="0.1.0",
    )

    # CORS 中间件：允许前端 Next.js 跨域访问
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 注册路由
    from study_agent.api.chat import router as chat_router
    from study_agent.api.kb import router as kb_router
    from study_agent.api.upload import router as upload_router

    app.include_router(upload_router)
    app.include_router(chat_router)
    app.include_router(kb_router)

    # 初始化 Agent
    get_agent()

    @app.get("/")
    async def health():
        agent = get_agent()
        return {
            "name": "PersonalQA",
            "status": "running",
            "knowledge_bases": agent.list_knowledge_bases(),
        }

    return app
