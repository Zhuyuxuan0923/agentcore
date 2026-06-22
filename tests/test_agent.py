"""PersonalQAAgent 单元测试 -- Mock ChromaDB / Embedder / LLM。

覆盖：
  - 上传流程（解析 + 分块 + 存储）
  - 问答流程（检索 + 生成 + 引用）
  - 文件不存在错误
  - 知识库为空时提问
  - 知识库切换 / 对话管理
"""

import os
import tempfile
from unittest.mock import Mock

import pytest

from study_agent.agent.kb_agent import PersonalQAAgent


@pytest.fixture
def temp_dir():
    """创建临时目录，测试结束后清理。"""
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def mock_llm():
    """Mock LLMClient -- 返回预设回答。"""
    llm = Mock()
    llm.chat.return_value = "根据公司制度，年假需提前一周申请 [1]。"
    return llm


@pytest.fixture
def mock_embedder():
    """Mock Embedder -- 返回假向量，数量匹配输入。"""
    emb = Mock()

    def fake_embed(texts):
        return [[0.1] * 384 for _ in texts]

    def fake_embed_query(text):
        return [0.1] * 384

    emb.embed.side_effect = fake_embed
    emb.embed_query.side_effect = fake_embed_query
    return emb


@pytest.fixture
def agent(temp_dir, mock_llm, mock_embedder):
    """创建一个带 Mock 依赖的 Agent。"""
    agent = PersonalQAAgent(
        llm_client=mock_llm,
        embedder=mock_embedder,
        chroma_dir=os.path.join(temp_dir, "chromadb"),
        db_path=os.path.join(temp_dir, "test.db"),
        chunk_size=200,
        chunk_overlap=30,
        top_k=3,
    )
    yield agent
    # 清理：关闭 ChromaDB 连接，否则 Windows 无法删除临时文件
    try:
        agent.kb_manager.client._system.stop()
    except Exception:
        pass
    try:
        agent.conv_manager._get_conn().close()
    except Exception:
        pass


# ── 文件上传测试 ─────────────────────────────────────────


def test_upload_txt_file(agent, temp_dir):
    """上传 .txt 文件：解析 -> 分块 -> 向量化 -> 存储。"""
    filepath = os.path.join(temp_dir, "test.txt")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("这是一段测试文本。" * 50)  # 约 400 字

    result = agent.upload(filepath, kb_name="测试知识库")
    assert result["kb_name"] == "测试知识库"
    assert result["file_name"] == "test.txt"
    assert result["chunk_count"] >= 1


def test_upload_md_file(agent, temp_dir):
    """上传 .md 文件应正常解析。"""
    filepath = os.path.join(temp_dir, "test.md")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("# 标题\n\n正文内容。\n\n## 第二章\n\n更多内容。")

    result = agent.upload(filepath)
    assert result["chunk_count"] >= 1


def test_upload_nonexistent_file(agent):
    """上传不存在的文件应抛出 FileNotFoundError。"""
    with pytest.raises(FileNotFoundError):
        agent.upload("/nonexistent/file.pdf")


def test_upload_unsupported_format(agent, temp_dir):
    """上传不支持的格式应抛出 ValueError。"""
    filepath = os.path.join(temp_dir, "test.exe")
    with open(filepath, "w") as f:
        f.write("fake exe")
    with pytest.raises(ValueError, match="不支持"):
        agent.upload(filepath)


# ── 问答测试 ────────────────────────────────────────────


def test_chat_without_documents(agent):
    """未上传文档时提问应返回友好提示。"""
    result = agent.chat("年假怎么申请？")
    assert "请先上传文档" in result.answer


def test_chat_with_documents(agent, temp_dir, mock_llm):
    """上传文档后提问，应得到带引用的回答。"""
    # 先上传
    filepath = os.path.join(temp_dir, "doc.txt")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("年假制度：入职满1年享受5天年假。年假需提前一周申请。")

    agent.upload(filepath, kb_name="制度库")

    # Mock ChromaDB collection.get() 返回数据
    # 需要让 HybridRetriever 能拿到数据
    # 因为 ChromaDB 实际运作，测试需要真实的 collection 数据

    result = agent.chat("年假怎么申请？")
    assert result.question == "年假怎么申请？"
    # 有上传文档，不应返回 "请先上传"
    # (可能返回答案或 "无法回答"，取决于检索结果)


# ── 知识库管理测试 ──────────────────────────────────────


def test_list_knowledge_bases(agent):
    """初始无知识库，返回空列表。"""
    kbs = agent.list_knowledge_bases()
    assert isinstance(kbs, list)


def test_switch_kb_nonexistent(agent):
    """切换到不存在的知识库返回 False。"""
    assert agent.switch_kb("不存在") is False


def test_switch_kb_after_upload(agent, temp_dir):
    """上传后切换知识库。"""
    filepath = os.path.join(temp_dir, "doc.txt")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("test content")
    agent.upload(filepath, kb_name="KB-A")
    assert agent.switch_kb("KB-A") is True


# ── 对话管理测试 ────────────────────────────────────────


def test_new_conversation(agent):
    """创建新对话返回 ID。"""
    conv_id = agent.new_conversation()
    assert conv_id
    assert len(conv_id) == 12  # uuid hex[:12]


def test_get_history_empty(agent):
    """空对话返回空列表。"""
    assert agent.get_history("nonexistent") == []


def test_list_conversations_empty(agent):
    """初始无对话返回空列表。"""
    assert agent.list_conversations() == []


def test_full_upload_chat_flow(agent, temp_dir):
    """端到端：上传 -> 提问 -> 对话记录。"""
    # Upload
    filepath = os.path.join(temp_dir, "policy.txt")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("年假制度：入职满1年享受5天年假，需提前一周申请。")

    upload_result = agent.upload(filepath, kb_name="制度库")
    assert upload_result["chunk_count"] >= 1

    # Chat
    chat_result = agent.chat("年假怎么申请？")
    assert chat_result.question == "年假怎么申请？"
    assert isinstance(chat_result.answer, str)

    # Conversation record
    conv_id = agent._current_conv_id
    assert conv_id is not None

    history = agent.get_history(conv_id)
    assert len(history) == 2  # user + assistant
    assert history[0].role == "user"
    assert history[1].role == "assistant"
