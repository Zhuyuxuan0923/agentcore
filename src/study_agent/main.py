"""PersonalQA -- 个人知识库问答 Agent 启动入口。

启动方式:
    poetry run uvicorn study_agent.main:app --reload --host 0.0.0.0 --port 8000

然后访问:
    http://localhost:8000/docs  -> API 文档（可在线测试所有接口）
    http://localhost:8000/      -> 健康检查
"""

from study_agent.api.app import create_app

app = create_app()
