"""问答 API。

POST /api/chat
  - 接收问题文本
  - 执行 RAG 全链路：检索 -> 增强 -> 生成 -> 引用解析
  - 返回答案 + 引用来源

GET /api/history/{conv_id}
  - 获取对话历史

GET /api/conversations
  - 列出所有对话
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api", tags=["chat"])


class ChatReq(BaseModel):
    question: str
    kb_name: str = "默认知识库"
    conversation_id: str | None = None


def get_agent():
    from study_agent.api.app import get_agent as ga

    return ga()


@router.post("/chat")
async def chat(request: ChatReq):
    """向知识库提问。

    Request body:
        {"question": "年假怎么申请？", "kb_name": "公司制度"}

    Response:
        {
            "question": "年假怎么申请？",
            "answer": "年假需提前一周向直属上级申请 [1]...",
            "sources": [...],
            "citations": [...]
        }
    """
    from study_agent.rag.security import detect_injection

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")

    if len(request.question) > 2000:
        raise HTTPException(status_code=400, detail="问题过长，请控制在 2000 字以内")

    if detect_injection(request.question):
        raise HTTPException(
            status_code=400,
            detail="检测到潜在的 Prompt Injection 攻击，请求已被拒绝",
        )

    try:
        agent = get_agent()
        # 切换到指定知识库
        if request.kb_name != "默认知识库":
            agent.switch_kb(request.kb_name)

        result = agent.chat(request.question, conversation_id=request.conversation_id)
        return {
            "question": result.question,
            "answer": result.answer,
            "sources": result.sources,
            "citations": result.citations,
            "conversation_id": agent._current_conv_id,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"问答处理失败: {e}")


@router.get("/history/{conv_id}")
async def get_history(conv_id: str):
    """获取指定对话的完整历史。"""
    agent = get_agent()
    messages = agent.get_history(conv_id)
    return {
        "conversation_id": conv_id,
        "messages": [
            {"role": m.role, "content": m.content, "citations": m.citations} for m in messages
        ],
    }


@router.get("/conversations")
async def list_conversations():
    """列出所有对话。"""
    agent = get_agent()
    return {"conversations": agent.list_conversations()}


@router.post("/conversations/new")
async def new_conversation(kb_name: str = "默认知识库"):
    """创建新对话。"""
    agent = get_agent()
    conv_id = agent.new_conversation()
    return {"conversation_id": conv_id}


@router.delete("/conversations/{conv_id}")
async def delete_conversation(conv_id: str):
    """删除指定对话及其所有消息。"""
    agent = get_agent()
    success = agent.conv_manager.delete_conversation(conv_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"对话 '{conv_id}' 不存在")
    # 如果删除的是当前对话，清除当前对话 ID
    if agent._current_conv_id == conv_id:
        agent._current_conv_id = None
    return {"status": "ok", "deleted": conv_id}
