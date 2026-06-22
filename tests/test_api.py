"""API 集成测试 -- FastAPI TestClient。

覆盖：
  - 健康检查端点
  - 文件上传端点（格式校验、空文件）
  - 问答端点（空问题、正常提问）
  - 知识库 CRUD
  - 对话列表 / 历史
"""

import os
import tempfile

import pytest
from fastapi.testclient import TestClient

from study_agent.api.app import create_app


@pytest.fixture
def client():
    """创建 TestClient -- 不启动真实服务器，直接在内存中测。"""
    app = create_app()
    return TestClient(app)


@pytest.fixture
def temp_txt_file():
    """创建一个临时 .txt 文件用于上传测试。"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("这是一段测试文本。" * 50)  # ~400 字
        f.flush()
        yield f.name
    os.unlink(f.name)


# ── 健康检查 ─────────────────────────────────────────────


def test_health_check(client):
    """GET / 返回健康状态。"""
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "PersonalQA"
    assert data["status"] == "running"


# ── 文件上传 ────────────────────────────────────────────


def test_upload_txt(client, temp_txt_file):
    """上传 .txt 文件返回成功。"""
    with open(temp_txt_file, "rb") as f:
        resp = client.post(
            "/api/upload",
            files={"file": ("test.txt", f, "text/plain")},
            data={"kb_name": "测试库"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["chunk_count"] >= 1


def test_upload_unsupported_format(client):
    """上传 .exe 文件应返回 400。"""
    resp = client.post(
        "/api/upload",
        files={"file": ("malware.exe", b"fake", "application/octet-stream")},
        data={"kb_name": "test"},
    )
    assert resp.status_code == 400
    assert "不支持" in resp.json()["detail"]


def test_upload_empty_kb_name(client, temp_txt_file):
    """空知识库名称使用默认值。"""
    with open(temp_txt_file, "rb") as f:
        resp = client.post(
            "/api/upload",
            files={"file": ("test.txt", f, "text/plain")},
        )
    assert resp.status_code == 200
    # 默认知识库名称
    assert resp.json()["kb_name"] == "默认知识库"


# ── 问答 ─────────────────────────────────────────────────


def test_chat_missing_question(client):
    """空问题返回 422（FastAPI 自动校验）。"""
    resp = client.post("/api/chat", json={"question": "", "kb_name": "test"})
    # FastAPI 可能返回 400 或 422
    assert resp.status_code in [400, 422]


def test_chat_normal_question(client):
    """正常提问返回 200（即使无文档）。"""
    resp = client.post("/api/chat", json={"question": "年假怎么申请？"})
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert "question" in data


def test_chat_missing_field(client):
    """缺少必填字段返回 422。"""
    resp = client.post("/api/chat", json={})
    assert resp.status_code == 422


# ── 知识库管理 ──────────────────────────────────────────


def test_list_kbs(client):
    """列出知识库返回 200。"""
    resp = client.get("/api/kb")
    assert resp.status_code == 200
    assert "knowledge_bases" in resp.json()


def test_create_and_delete_kb(client):
    """创建然后删除知识库。"""
    # 创建
    resp = client.post("/api/kb", json={"name": "临时测试库"})
    assert resp.status_code == 200
    assert resp.json()["name"] == "临时测试库"

    # 删除（URL 编码中文名称）
    from urllib.parse import quote

    resp = client.delete(f"/api/kb/{quote('临时测试库')}")
    assert resp.status_code == 200


def test_delete_nonexistent_kb(client):
    """删除不存在的知识库返回 404。"""
    resp = client.delete("/api/kb/不存在的库")
    assert resp.status_code == 404


# ── 对话管理 ────────────────────────────────────────────


def test_list_conversations(client):
    """列出对话返回 200。"""
    resp = client.get("/api/conversations")
    assert resp.status_code == 200
    assert "conversations" in resp.json()


def test_new_conversation(client):
    """创建新对话返回 ID。"""
    resp = client.post("/api/conversations/new")
    assert resp.status_code == 200
    assert "conversation_id" in resp.json()


def test_get_history_nonexistent(client):
    """查询不存在的对话历史返回空。"""
    resp = client.get("/api/history/fake-id-123")
    assert resp.status_code == 200
    assert resp.json()["messages"] == []


def test_delete_nonexistent_conversation(client):
    """删除不存在的对话返回 404。"""
    resp = client.delete("/api/conversations/fake-id-123")
    assert resp.status_code == 404
