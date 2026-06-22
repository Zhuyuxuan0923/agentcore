# PersonalQA — 个人知识库问答 Agent

上传你的文档（PDF / DOCX / Markdown / TXT），用自然语言提问，获得带来源引用的答案。

**这是「16 周 AI Agent 工程师养成计划」的作品 1**，技术栈覆盖：Python · FastAPI · RAG · ChromaDB · Prompt Engineering · Next.js。

## Demo 预览

```
用户: 年假怎么申请？

Agent: 年假需提前一周向直属上级提交申请 [1]。
       审批通过后，HR 系统会自动扣除年假余额 [2]。

       引用来源:
       [1] 《员工手册》第3章第2节：年假申请流程...
       [2] 《考勤制度》第5条：年假审批与扣减规则...
```

## 核心特性

- **多格式文档上传** — 支持 PDF、DOCX、Markdown、TXT，拖拽即可索引
- **RAG 全链路** — 检索（Hybrid Search）→ 增强（Prompt 模板）→ 生成（LLM）→ 引用解析
- **带引用的答案** — 每个关键事实标注来源编号 [1] [2]，点击可查看原文
- **多知识库切换** — 按主题创建独立知识库（如「公司制度」「技术文档」「个人笔记」）
- **对话历史持久化** — SQLite 存储，支持多轮对话，随时回溯
- **前端聊天界面** — Next.js + shadcn/ui，Markdown 渲染 + 代码语法高亮
- **安全防护** — Prompt Injection 检测、文件大小限制、输入长度截断
- **测试覆盖** — 单元测试 + 集成测试 + 安全测试，共 25 个用例

## 系统架构

```
+--------------------+       +---------------------+
|   Next.js 前端      |       |   FastAPI 后端       |
|   (localhost:3000)  |<----->|   (localhost:8000)   |
|                    |  REST |                     |
|  +--------------+  |  API  |  +---------------+  |
|  | ChatArea     |  |       |  | /api/upload   |  |
|  | Sidebar      |  |       |  | /api/chat     |  |
|  | InputArea    |  |       |  | /api/kb       |  |
|  | MarkdownRenderer|     |  | /api/conversations |  |
|  +--------------+  |       |  +---------------+  |
+--------------------+       +-------+---+---------+
                                      |   |
                          +-----------+   +-----------+
                          |                           |
                    +-----v------+             +------v------+
                    | PersonalQA |             | Conversation|
                    |   Agent    |             |   Manager   |
                    +-----+------+             +------+------+
                          |                           |
              +-----------+-----------+       +-------v--------+
              |           |           |       |    SQLite      |
        +-----v---+ +----v----+ +---v------+ | (conversations)|
        |ChromaDB | | LLM     | | Prompt   | +----------------+
        |向量数据库| | Client  | | Templates|
        +---------+ +---------+ +----------+

数据流:
  上传: 文件 -> 解析 -> 分块 -> Embedding -> ChromaDB
  问答: 问题 -> 向量检索 + 关键词检索 -> RRF混合排序 -> Prompt渲染 -> LLM生成 -> 引用解析 -> 前端渲染
```

## 快速开始

### 前置条件

- Python 3.11+
- Node.js 20+
- Poetry（Python 包管理器）
- 至少一个 LLM 提供商的 API Key（推荐 DeepSeek，便宜够用）

### 1. 克隆并安装

```bash
git clone https://github.com/<your-username>/study-agent.git
cd study-agent
poetry install
```

### 2. 配置环境变量

在项目根目录创建 `.env` 文件：

```ini
# 至少配一家
DEEPSEEK_API_KEY="sk-your-key"
# 可选
OPENAI_API_KEY="sk-your-key"
ANTHROPIC_API_KEY="sk-your-key"
ZHIPU_API_KEY="your-key"
MOONSHOT_API_KEY="sk-your-key"

# 默认使用的 LLM 厂商
LLM_PROVIDER="deepseek"

# Embedding 模型（可选，默认 text-embedding-3-small）
EMBEDDING_MODEL="text-embedding-3-small"
```

### 3. 启动后端

```bash
poetry run uvicorn study_agent.main:app --reload --host 0.0.0.0 --port 8000
```

访问 http://localhost:8000/docs 查看自动生成的 API 文档（Swagger UI）。

### 4. 启动前端

```bash
cd frontend
npm install
npm run dev
```

访问 http://localhost:3000 打开聊天界面。

### 5. 开始使用

1. 在左侧边栏创建知识库（如「公司制度」）
2. 上传 PDF / DOCX / Markdown 文件
3. 在聊天框输入问题，Agent 将从你的文档中检索并回答

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/` | 健康检查 |
| `POST` | `/api/upload` | 上传文档到知识库 |
| `POST` | `/api/chat` | 向知识库提问 |
| `POST` | `/api/kb` | 创建知识库 |
| `GET` | `/api/kb` | 列出所有知识库 |
| `DELETE` | `/api/kb/{name}` | 删除知识库 |
| `GET` | `/api/conversations` | 列出所有对话 |
| `GET` | `/api/history/{conv_id}` | 获取对话历史 |
| `POST` | `/api/conversations/new` | 创建新对话 |
| `DELETE` | `/api/conversations/{conv_id}` | 删除对话 |

`POST /api/chat` 请求示例：

```json
{
  "question": "年假怎么申请？",
  "kb_name": "公司制度",
  "conversation_id": null
}
```

响应示例：

```json
{
  "question": "年假怎么申请？",
  "answer": "年假需提前一周向直属上级提交申请 [1]...",
  "sources": [
    {"doc_id": "员工手册.pdf_chunk_0", "text": "第3章第2节...", "score": 0.95}
  ],
  "citations": [
    {"number": 1, "text": "第3章第2节..."}
  ],
  "conversation_id": "abc123"
}
```

## 项目结构

```
study-agent/
+-- src/study_agent/           # 后端源代码
|   +-- agent/                 # PersonalQA Agent 核心
|   |   +-- kb_agent.py        #   Agent 主逻辑（整合 LLM + RAG + 对话）
|   |   +-- knowledge_base.py  #   知识库管理（ChromaDB CRUD）
|   |   \-- conversation.py    #   对话管理（SQLite 持久化）
|   +-- api/                   # FastAPI 接口层
|   |   +-- app.py             #   应用工厂 + CORS + 路由注册
|   |   +-- upload.py          #   文件上传端点（格式校验 + 大小限制）
|   |   +-- chat.py            #   问答端点（安全检测 + RAG 调用）
|   |   \-- kb.py              #   知识库管理端点
|   +-- rag/                   # RAG 管道
|   |   +-- chunking.py        #   文档分块（递归/固定/语义 3 种策略）
|   |   +-- embedding.py       #   Embedding 向量化
|   |   +-- retriever.py       #   混合检索（向量 + 关键词 + RRF 融合）
|   |   +-- generator.py       #   生成 + 引用解析
|   |   +-- pipeline.py        #   端到端 RAG 管道
|   |   \-- security.py        #   Prompt Injection 检测 + 输入净化
|   +-- llm/                   # LLM 调用层
|   |   +-- client.py          #   统一 LLMClient（5 家 provider）
|   |   +-- retry.py           #   智能重试机制（指数退避）
|   |   \-- structured.py      #   结构化输出提取
|   +-- prompt/                # Prompt 工程
|   |   +-- templates.py       #   Jinja2 模板引擎
|   |   +-- examples.py        #   Few-Shot 示例管理
|   |   +-- evaluator.py       #   Prompt 评测框架
|   |   \-- templates/         #   .j2 模板文件
|   +-- tools/                 # Tool Calling 工具系统
|   +-- config/                # 配置中心
|   |   \-- settings.py        #   Provider 注册 / LLMConfig
|   \-- main.py                # 应用入口
+-- frontend/                  # Next.js 前端
|   +-- src/
|   |   +-- app/
|   |   |   +-- layout.tsx     #    根布局
|   |   |   \-- page.tsx       #    主页面（聊天界面）
|   |   +-- components/
|   |   |   +-- chat/          #    聊天组件
|   |   |   |   +-- chat-area.tsx         # 消息列表
|   |   |   |   +-- input-area.tsx        # 输入框 + 文件上传
|   |   |   |   +-- sidebar.tsx           # 知识库/对话列表
|   |   |   |   \-- markdown-renderer.tsx  # Markdown + 代码高亮
|   |   |   \-- ui/            #    UI 基础组件（shadcn/ui）
|   |   \-- lib/
|   |       \-- api.ts         #    API 客户端
|   \-- package.json
+-- tests/                     # 测试
|   +-- test_chunking.py       #   分块单元测试
|   +-- test_agent.py          #   Agent 单元测试
|   +-- test_api.py            #   API 集成测试
|   \-- test_security.py       #   安全测试
+-- data/                      # 运行时数据（gitignore）
|   +-- chromadb/              #   向量数据库持久化
|   +-- conversations.db       #   对话历史 SQLite
|   \-- uploads/               #   上传文件暂存
+-- docs/                      # 学习日志
+-- pyproject.toml             # 项目配置
\-- .env                       # 环境变量（gitignore）
```

## 技术栈

| 层级 | 技术 |
|------|------|
| 后端框架 | FastAPI + Uvicorn |
| 前端框架 | Next.js 16 + React 19 + TypeScript |
| UI 组件 | shadcn/ui + Tailwind CSS v4 |
| LLM 调用 | OpenAI SDK + Anthropic SDK（5 家 provider 统一封装） |
| 向量数据库 | ChromaDB（本地嵌入，零配置） |
| Embedding | text-embedding-3-small（通过 OpenAI API） |
| Prompt 引擎 | Jinja2 模板 + Few-Shot 管理 |
| 对话存储 | SQLite（本地持久化） |
| 重试机制 | tenacity（指数退避 + 智能错误分类） |
| Markdown 渲染 | react-markdown + remark-gfm + react-syntax-highlighter |
| 测试 | pytest |
| 代码质量 | Black + Ruff + MyPy + Pre-commit |

## 运行测试

```bash
# 运行全部测试
poetry run pytest tests/ -v

# 运行单个测试文件
poetry run pytest tests/test_chunking.py -v

# 运行单个测试函数
poetry run pytest tests/test_security.py::test_detect_injection_ignore_instruction -v
```

## 设计决策

**为什么用 ChromaDB 而不是 Milvus/Qdrant？**
ChromaDB 是嵌入式向量数据库，pip install 即可用，无需 Docker 或外部服务。对个人知识库（几千到几万条文档块）完全够用。生产环境推荐升级到 Milvus 或 Qdrant。

**为什么用 Hybrid Search（向量 + 关键词）？**
纯向量检索擅长语义匹配，但对精确关键词（如「第3条」「2024年」）容易漏。BM25 关键词检索恰好补上了这个短板。RRF（Reciprocal Rank Fusion）将两种排序结果融合，实际测试中比纯向量检索召回率高 15-20%。

**为什么 SQLite 而不是 PostgreSQL？**
作品 1 的目标是「clone 下来就能跑」，SQLite 零配置、单文件、不需要额外安装。对话历史数据量不大（每人每天几十条），SQLite 完全够用。多用户场景再升级到 PostgreSQL。

**Prompt Injection 防护够吗？**
当前的防护是「基础防线」——关键词检测 + System Prompt 加固 + 输入长度截断。能挡住常见的越狱尝试，但不是 100% 安全。生产环境需要三层防护：输入 Guardrails（Nemo/Guardrails-AI）+ System Prompt 硬化 + 输出内容审核。

## License

MIT
