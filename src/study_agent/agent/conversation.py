"""对话历史管理 -- SQLite 存储。

每一轮对话记录为一条 row，包含：
  - conversation_id: 对话 ID
  - turn: 第几轮
  - role: user / assistant
  - content: 消息内容
  - citations: 引用信息（JSON 字符串）

SQLite 是嵌入式数据库——不需要安装任何服务，一个文件就是整个数据库。
适合单用户场景，零配置，零维护。
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass
class Message:
    role: str  # "user" | "assistant"
    content: str
    citations: list[dict] = field(default_factory=list)


@dataclass
class Conversation:
    id: str
    title: str
    knowledge_base_name: str
    messages: list[Message] = field(default_factory=list)
    created_at: str = ""


class ConversationManager:
    """对话历史管理器 -- SQLite 作为存储后端。

    什么是 SQLite？一个文件就是一个完整的数据库，不需要安装 MySQL 之类的东西。
    你的对话数据存在 data/conversations.db 这一个文件里。

    使用方式：
        mgr = ConversationManager()
        conv_id = mgr.create_conversation("我的课程讲义")
        mgr.add_message(conv_id, Message(role="user", content="年假怎么申请？"))
        history = mgr.get_history(conv_id)  # -> list[Message]
    """

    def __init__(self, db_path: str = "data/conversations.db"):
        self.db_path = db_path
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """创建两张表：conversations（对话列表）、messages（消息记录）。"""
        conn = self._get_conn()
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL DEFAULT '新对话',
                knowledge_base_name TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                turn INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                citations TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            );
            """
        )
        conn.commit()
        conn.close()

    def create_conversation(self, knowledge_base_name: str = "", title: str = "新对话") -> str:
        conv_id = uuid.uuid4().hex[:12]
        now = datetime.now(UTC).isoformat()
        conn = self._get_conn()
        conn.execute(
            "INSERT INTO conversations "
            "(id, title, knowledge_base_name, created_at) VALUES (?, ?, ?, ?)",
            (conv_id, title, knowledge_base_name, now),
        )
        conn.commit()
        conn.close()
        return conv_id

    def add_message(self, conversation_id: str, message: Message) -> int:
        conn = self._get_conn()
        # 找到当前对话的最大 turn 编号
        row = conn.execute(
            "SELECT COALESCE(MAX(turn), 0) AS max_turn FROM messages WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
        next_turn = row["max_turn"] + 1
        now = datetime.now(UTC).isoformat()
        conn.execute(
            "INSERT INTO messages (conversation_id, turn, role, content, citations, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                conversation_id,
                next_turn,
                message.role,
                message.content,
                json.dumps(message.citations, ensure_ascii=False),
                now,
            ),
        )
        # 自动更新对话标题（用第一条用户消息的前 20 字）
        if next_turn == 1 and message.role == "user":
            title = message.content[:20]
            conn.execute(
                "UPDATE conversations SET title = ? WHERE id = ?", (title, conversation_id)
            )
        conn.commit()
        conn.close()
        return next_turn

    def get_history(self, conversation_id: str) -> list[Message]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT role, content, citations FROM messages WHERE conversation_id = ? ORDER BY turn",
            (conversation_id,),
        ).fetchall()
        conn.close()
        return [
            Message(
                role=row["role"],
                content=row["content"],
                citations=json.loads(row["citations"]),
            )
            for row in rows
        ]

    def list_conversations(self, knowledge_base_name: str | None = None) -> list[Conversation]:
        conn = self._get_conn()
        if knowledge_base_name:
            rows = conn.execute(
                "SELECT * FROM conversations "
                "WHERE knowledge_base_name = ? ORDER BY created_at DESC",
                (knowledge_base_name,),
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM conversations ORDER BY created_at DESC").fetchall()
        conn.close()
        return [
            Conversation(
                id=row["id"],
                title=row["title"],
                knowledge_base_name=row["knowledge_base_name"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def delete_conversation(self, conversation_id: str) -> bool:
        conn = self._get_conn()
        # 先检查是否存在
        row = conn.execute(
            "SELECT id FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()
        if row is None:
            conn.close()
            return False
        conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        conn.commit()
        conn.close()
        return True
