"""知识库管理 API。

GET  /api/kb          -> 列出所有知识库
POST /api/kb          -> 创建新知识库
DELETE /api/kb/{name} -> 删除知识库
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api", tags=["knowledge-base"])


class CreateKBRequest(BaseModel):
    name: str


def get_agent():
    from study_agent.api.app import get_agent as ga

    return ga()


@router.get("/kb")
async def list_kbs():
    """列出所有知识库。"""
    agent = get_agent()
    kbs = agent.list_knowledge_bases()
    return {"knowledge_bases": kbs}


@router.post("/kb")
async def create_kb(request: CreateKBRequest):
    """创建新知识库。"""
    if not request.name.strip():
        raise HTTPException(status_code=400, detail="知识库名称不能为空")
    agent = get_agent()
    agent.kb_manager.create(request.name)
    return {"status": "ok", "name": request.name}


@router.delete("/kb/{name}")
async def delete_kb(name: str):
    """删除知识库。"""
    agent = get_agent()
    success = agent.delete_kb(name)
    if not success:
        raise HTTPException(status_code=404, detail=f"知识库 '{name}' 不存在")
    return {"status": "ok", "deleted": name}
