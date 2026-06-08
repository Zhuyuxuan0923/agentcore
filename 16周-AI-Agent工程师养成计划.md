# 16周 AI Agent 工程师养成计划

> **适用人群**：有 Python 基础、会用 Git、做过几个 Agent Demo 的在校生/应届生
> **投入**：每天 4-6 小时，每周 6 天（周日轻量复盘）
> **目标岗位**：AI Agent 开发工程师 / AI 应用工程师，薪资 15K-25K
> **核心策略**：作品驱动 + 证书补位 + 第 9 周开始投递

---

## 1. 总体路线图

| 阶段 | 周次 | 核心目标 | 阶段产出物 | 投递策略 |
|------|------|----------|------------|----------|
| **阶段一：基础夯实** | W1-W4 | 从"会用 API"到"能独立交付 RAG 应用" | 作品1：个人知识库问答 Agent | ❌ 不投，积累弹药 |
| **阶段二：深度构建** | W5-W8 | 掌握多 Agent 协作与工具编排 | 作品2：多 Agent 代码审查系统 | ❌ 不投，但开始整理简历素材 |
| **阶段三：工程化** | W9-W12 | 从 Demo 到生产级应用 | 作品3：自动化工作流 Agent | ✅ **W9 开始投递**，用作品2+3 面试 |
| **阶段四：企业级冲刺** | W13-W16 | 补齐安全/合规/性能短板，拿 Offer | 作品4：企业级安全合规 Agent | ✅ 全力投递+面试，用全部 4 个作品 |

### 为什么是这个节奏

- **W1-W8 不投**：作品不够硬，面试通过率低，浪费时间且打击信心
- **W9 开始投**：此时已有 3 个作品（2 个完成 + 1 个进行中），简历竞争力足够
- **W13-W16 面试期**：边面试边做作品 4，面试中可以聊"正在做的项目"，展示学习能力

---

## 2. 周度拆解

### 阶段一：基础夯实（W1-W4）

#### Week 1：Python 工程化 + LLM API 深度掌握

| 项目 | 内容 |
|------|------|
| **本周目标** | 能用 FastAPI + Pydantic 写出生产级 API，熟练掌握 3 家以上 LLM API 的错误处理与重试机制 |

**每日任务**

| 天 | 任务 | 预计耗时 |
|----|------|----------|
| Day 1 | 搭建开发环境：VS Code 配置（Black/MyPy/Ruff）、poetry 管理依赖、pre-commit 配置。让 CC 生成 `.pre-commit-config.yaml` 骨架，你重点改 hooks 列表 | 4h |
| Day 2 | FastAPI 速通：写一个 `/chat` 端点，Pydantic v2 做请求校验，StreamingResponse 实现 SSE 流式输出。**关键**：用 `httpx` 替代 `requests`，后面全用异步 | 5h |
| Day 3 | OpenAI SDK 深度使用：system prompt 管理、temperature/top_p 调参实验、token 计数（tiktoken）、function calling 基础。写一个脚本批量测试 20 条 prompt 并记录 token 消耗 | 5h |
| Day 4 | Anthropic SDK + 国产模型：Claude API 的 system message 与 OpenAI 的区别、GLM-4/DeepSeek API 接入。封装一个统一的 `LLMClient` 类，支持 3 家以上 provider 切换 | 6h |
| Day 5 | 错误处理与重试：实现 exponential backoff、rate limit 处理、stream 中断恢复。用 `tenacity` 库。写单元测试覆盖所有异常分支 | 6h |
| Day 6 | 集成：把 Day 2-5 的内容整合成一个 `agent-core` 包，包含 config/llm/tools 三个模块。写 README 和 docstring | 6h |
| Day 7 | 周复盘：整理本周踩坑笔记，发一篇知乎/掘金文章《LLM API 调用踩坑指南》。文章就是你的学习笔记，不用写多好 | 3h |

**本周产出**
- ✅ `agent-core` Python 包（GitHub 仓库，含 README + 单元测试）
- ✅ 一篇技术博文
- ✅ 本地可运行的 `/chat` SSE 端点

**检验标准**
- 🟡 能在 3 分钟内用自己封装的 client 切换 Anthropic → DeepSeek，不报错
- 🟡 流式输出中断后能自动重连恢复，不丢数据
- 🟡 单元测试覆盖率 > 80%

**避坑提示**
- ❌ **别碰 LangChain**：这周的目标是理解底层，LangChain 的抽象层会阻碍你理解 API 细节
- ❌ **别纠结异步原理**：会用 `async/await` 就行，别花 3 小时看 asyncio 源码
- ❌ **别追求完美封装**：`LLMClient` 能用就行，后面会重构

---

#### Week 2：Prompt Engineering + 结构化输出

| 项目 | 内容 |
|------|------|
| **本周目标** | 掌握 Prompt 工程方法论，能稳定控制 LLM 输出 JSON/结构化数据，理解并实现 Tool Use 循环 |

**每日任务**

| 天 | 任务 | 预计耗时 |
|----|------|----------|
| Day 1 | Prompt 工程理论：精读 Anthropic 的 Prompt Engineering Guide + OpenAI Cookbook 的 prompt 相关章节。整理 10 条最核心的原则到自己的 Notion/飞书 | 4h |
| Day 2 | Prompt 模板系统：用 Jinja2 实现可变 prompt 模板，支持 few-shot 示例动态注入。让 CC 生成模板引擎骨架，你重点设计模板变量和作用域 | 5h |
| Day 3 | 结构化输出（JSON Mode + Tool Calling）：分别用 OpenAI JSON mode、Claude tool use、GLM-4 的 function call 实现同样的结构化提取任务，对比成功率 | 6h |
| Day 4 | Tool Use 深入：实现一个完整的 tool calling 循环——LLM 决定调用哪个工具 → 执行 → 把结果喂回 LLM → 继续决策。最多循环 5 轮 | 6h |
| Day 5 | Prompt 评测：写一个评测脚本，用 50 条测试用例对比 3 种 prompt 写法的准确率。输出评测报告（markdown 表格） | 6h |
| Day 6 | System Prompt 设计实战：给你的 Agent 写 3 个版本的 system prompt，用 Day 5 的评测框架跑分。**关键**：prompt 里要有角色、约束、输出格式、few-shot 示例 | 5h |
| Day 7 | 周复盘：整理 prompt 模板库到 GitHub。本周文章《写好 System Prompt 的 10 条法则》 | 3h |

**本周产出**
- ✅ `prompt-templates` 仓库（含 10+ 可复用模板 + 评测框架）
- ✅ System Prompt 设计方法论笔记
- ✅ 结构化输出对比评测报告

**检验标准**
- 🟡 同一个任务用 3 种 prompt 写法，最好和最差准确率差 < 15%
- 🟡 Tool calling 循环能稳定在 3 轮内完成一个多步骤任务（如"先搜索再总结再翻译"）
- 🟡 结构化输出成功率 > 90%（50 条测试）

**避坑提示**
- ❌ **别只看不练**：prompt engineering 是手艺活，每天至少写 10 条 prompt
- ❌ **别追求复杂 prompt**：一个好的 system prompt 通常在 200-500 字，不是 2000 字
- ❌ **别忽略国产模型**：面试大概率会问"国产模型和 GPT 在 prompt 上有什么区别"

---

#### Week 3：RAG 基础 + 向量数据库

| 项目 | 内容 |
|------|------|
| **本周目标** | 理解 RAG 全链路，能用 LangChain/LlamaIndex 搭建可用的文档问答系统 |

**每日任务**

| 天 | 任务 | 预计耗时 |
|----|------|----------|
| Day 1 | RAG 理论：精读 LlamaIndex 官方文档的 Concepts 章节（Embedding、Node、Index、Retriever、QueryEngine）。画一张 RAG 全链路流程图 | 5h |
| Day 2 | 文档处理管道：实现 PDF/Word/Markdown 解析 → 文本分块（Chunking）→ Embedding。对比 3 种分块策略（固定大小、语义分块、递归分块）。让 CC 生成分块器骨架，你重点调 chunk_size 和 overlap | 6h |
| Day 3 | 向量数据库选型与实践：分别用 ChromaDB（本地）、Milvus Lite（本地）存储同一批数据。对比查询速度、内存占用、过滤查询能力 | 5h |
| Day 4 | 检索策略实战：实现基础向量检索、Hybrid Search（向量+关键词）、Re-ranking（用 Cohere/本地模型）。对比 3 种策略的召回率 | 6h |
| Day 5 | 生成与引用：实现检索结果拼接 prompt → LLM 生成答案 + 来源引用。**关键**：让 LLM 在答案中标注引用来源（如 [1]、[2]），并在 UI 中可点击跳转 | 6h |
| Day 6 | 集成测试：用 30 个真实问题测试端到端效果。记录每个问题的检索召回、答案准确性、引用正确性。输出评测报告 | 5h |
| Day 7 | 周复盘 + **报名计算机二级 Python**（如果还没有）。如果你 Python 基础扎实，这周刷 3 套真题基本能过 | 3h |

**本周产出**
- ✅ `rag-playground` 仓库（含完整的文档解析→检索→生成管道）
- ✅ RAG 评测报告（30 条 QA 的准确率统计）
- ✅ 3 种分块策略 + 3 种检索策略的对比数据

**检验标准**
- 🟡 上传一个 PDF，能在一分钟内完成索引并回答相关问题
- 🟡 回答准确率 > 70%（基于自建评测集）
- 🟡 引用来源 100% 可追溯，无幻觉引用

**避坑提示**
- ❌ **ChromaDB 别用于生产**：第 3 周用 ChromaDB 学习没问题，但简历上的项目必须用 Milvus/Qdrant/Weaviate
- ❌ **别把 chunk 切太小**：chunk_size < 200 token 会导致上下文碎片化
- ❌ **别忽略 PDF 解析质量**：PyPDF2 对中文支持差，用 `pdfplumber` 或 `unstructured`

---

#### Week 4：作品1——个人知识库问答 Agent + 阶段复盘

| 项目 | 内容 |
|------|------|
| **本周目标** | 交付第一个完整作品：能部署到公网的个人知识库问答 Agent，含前端界面 |

**每日任务**

| 天 | 任务 | 预计耗时 |
|----|------|----------|
| Day 1 | 产品设计：定义功能边界——支持上传 PDF/DOCX/MD，问答带引用，对话历史。画 UI 草图。技术选型：FastAPI + ChromaDB/Milvus Lite + Next.js（让 CC 生成前端骨架） | 4h |
| Day 2 | 后端核心：实现文件上传 API、文档处理管道、向量索引、问答 API。整合前 3 周的代码，**重构而非重写** | 6h |
| Day 3 | 前端搭建：用 Next.js + shadcn/ui 搭聊天界面。**让 CC 生成 80% 的前端代码**，你重点改 API 对接逻辑和 UI 细节 | 6h |
| Day 4 | 特性完善：加对话历史持久化（本地 SQLite）、多知识库切换、Markdown 渲染 + 代码高亮 | 6h |
| Day 5 | 测试与修复：端到端测试、边界 case 处理（空文件、超大文件、恶意 prompt）。让 CC 帮你生成测试用例 | 5h |
| Day 6 | 部署上线：用 Docker Compose 本地部署 → Vercel 部署前端 + Railway/阿里云免费额度部署后端。写部署文档 | 6h |
| Day 7 | 🧘 **阶段复盘与心理调整**：回顾 4 周成果，更新 GitHub Profile，确认作品 1 已上线可访问。如果感到倦怠，这周减量——完成部署就行，不追求完美 | 3h |

**本周产出** 见 [作品 1 详情](#作品1个人知识库问答-agent)

**检验标准**
- 🟡 任何人都能通过公网 URL 访问、上传文件、提问
- 🟡 10 个测试问题的回答准确率 > 70%，引用全部可追溯
- 🟡 界面不丑（用 shadcn/ui 默认风格即可）

**阶段一复盘清单**
1. 你的 GitHub 现在有几个绿点？目标是 **28 天连续提交**
2. 作品 1 能否在面试中 5 分钟演示完整流程？
3. 哪个技术点最薄弱？在进入阶段二前花 1 天补上
4. **心态检查**：有没有陷入"教程地狱"？有没有因为完美主义拖延？

---

### 阶段二：深度构建（W5-W8）

#### Week 5：Agent 框架深度使用

| 项目 | 内容 |
|------|------|
| **本周目标** | 深入 LangChain/LlamaIndex 的 Agent 模块，理解 ReAct/Plan-Execute 等范式，能用框架快速搭建复杂 Agent |

**每日任务**

| 天 | 任务 | 预计耗时 |
|----|------|----------|
| Day 1 | LangChain Agent 模块精读：AgentExecutor、Tool、AgentAction、AgentFinish 的源码级理解。**只看不写**，画调用时序图 | 5h |
| Day 2 | ReAct Agent 实战：用 LangChain 的 `create_react_agent` 搭建一个能搜索+计算+总结的研究 Agent。对比 ReAct 和 Function Calling 的差异 | 5h |
| Day 3 | Plan-Execute Agent：实现"先制定计划→逐步执行→根据结果调整计划"的 Agent。**关键**：计划步骤用 JSON 结构化，每步执行后校验 | 6h |
| Day 4 | LangGraph 入门：用 LangGraph 搭建一个带条件分支的 Agent 工作流（如：用户问技术问题→走搜索分支，用户闲聊→走对话分支） | 6h |
| Day 5 | CrewAI / AutoGen 对比：分别用 CrewAI 和 AutoGen 实现同样的 2-Agent 任务（一个研究员+一个写手），对比开发体验和效果 | 6h |
| Day 6 | 框架选型决策：写一份 Agent 框架选型指南（LangChain vs LlamaIndex vs CrewAI vs 自研），这是面试高频题 | 5h |
| Day 7 | 周复盘：整理框架对比笔记。本周文章《2025 AI Agent 框架选型指南》 | 3h |

**本周产出**
- ✅ `agent-frameworks` 仓库（含 4 种 Agent 范式的实现）
- ✅ Agent 框架选型指南文档
- ✅ LangGraph 工作流图（可作为面试展示）

**检验标准**
- 🟡 同一个任务（如"调研 React 19 新特性并写总结"），能用 3 种框架实现
- 🟡 能手绘 ReAct 循环的时序图，解释每一步的数据流动
- 🟡 能说清楚"什么场景用 LangChain，什么场景自己写"

**避坑提示**
- ❌ **别成为"框架调参侠"**：理解框架设计思想比会用 API 重要 10 倍
- ❌ **LangChain 的坑**：LangChain 版本更新快，锁定 `langchain==0.3.x`，别追最新版
- ❌ **别所有东西都用框架**：简单的单 Agent 任务自己写比用框架快

---

#### Week 6：Memory + 多轮对话 + Agent 评估

| 项目 | 内容 |
|------|------|
| **本周目标** | 掌握 Agent Memory 管理、多轮对话状态维护、Agent 行为评估方法论 |

**每日任务**

| 天 | 任务 | 预计耗时 |
|----|------|----------|
| Day 1 | Memory 机制：实现 3 种 Memory——Buffer Memory（全量）、Summary Memory（摘要）、Vector Memory（检索相关历史）。对比 token 消耗和上下文相关性 | 6h |
| Day 2 | 混合 Memory 策略：实现"近期对话用 Buffer + 远期用 Summary + 知识用 Vector"的三层 Memory。**核心难点**是 Memory 之间的检索优先级 | 6h |
| Day 3 | Agent 状态管理：用 Pydantic 定义 Agent 会话状态（State），实现多轮对话中的上下文一致性。处理并发会话的场景 | 5h |
| Day 4 | Agent 评估框架：搭建评估管道——定义评估维度（准确性、效率、tool 使用正确率）、构建测试集、自动打分 | 6h |
| Day 5 | 用 LangSmith / 自研方案做 Trace：实现 Agent 每一步决策的可观测性——工具调用、LLM 输出、中间状态全量记录 | 5h |
| Day 6 | 对抗测试：写一个"坏用户"bot 来测试你的 Agent——越狱 prompt、矛盾指令、超长输入。修复暴露的问题 | 5h |
| Day 7 | 周复盘 + **报名阿里云 ACA 云计算**（或 AWS Cloud Practitioner）。备考策略：刷 ACloudGuru/Udemy 课程，一周可过 | 3h |

**本周产出**
- ✅ `agent-memory` 仓库（含 3 种 Memory + 混合策略实现）
- ✅ Agent 评估框架 + 测试集
- ✅ Trace/可观测性方案

**检验标准**
- 🟡 混合 Memory 策略下，500 轮对话后 Agent 仍能准确引用第 10 轮的内容
- 🟡 Agent 评估有量化的准确率和工具调用正确率
- 🟡 对抗测试后 Agent 不会输出危险内容或陷入死循环

**避坑提示**
- ❌ **Memory ≠ 把历史全塞进 prompt**：Token 是有成本的，Summary Memory 在长对话中必须要用
- ❌ **别忽略并发**：面试官可能问"多用户同时聊天怎么管理 Memory"

---

#### Week 7：多 Agent 架构设计

| 项目 | 内容 |
|------|------|
| **本周目标** | 掌握多 Agent 协作模式，设计并实现一个多 Agent 系统的架构原型 |

**每日任务**

| 天 | 任务 | 预计耗时 |
|----|------|----------|
| Day 1 | 多 Agent 模式研究：精读 Microsoft AutoGen 论文 + CrewAI 文档。整理 4 种协作模式——顺序、辩论、层级、广播。画出每种模式的消息流图 | 5h |
| Day 2 | Agent 间通信协议设计：定义 Agent 间消息格式（JSON Schema）、路由规则、超时与重试。**核心**：设计一个 MessageBus 或直接用 Redis pub/sub | 6h |
| Day 3 | 角色分工设计：以一个"代码审查"场景为例，设计 3 个 Agent 角色——Reviewer（审查代码）、Researcher（查文档/找参考）、Reporter（汇总输出）。写每个角色的 system prompt | 6h |
| Day 4 | 编排器实现：实现一个 Orchestrator，负责任务分解→分配给子 Agent→收集结果→决策是否重试→输出最终结果。让 CC 生成骨架，你重点写编排逻辑 | 6h |
| Day 5 | 冲突处理：实现 Agent 间意见不一致时的处理机制——投票、层级裁决、外部信息检索验证 | 6h |
| Day 6 | 集成测试：用一个真实 PR 链接或代码片段跑通整个多 Agent 审查流程。记录每个 Agent 的决策和耗时 | 5h |
| Day 7 | 周复盘：整理多 Agent 架构设计文档（draw.io 或 Excalidraw 画出架构图）。本周文章《多 Agent 协作的 4 种模式》 | 3h |

**本周产出**
- ✅ `multi-agent-framework` 仓库（含编排器 + 通信协议 + 3 个 Agent 角色）
- ✅ 多 Agent 架构设计文档
- ✅ 代码审查场景的端到端 Demo

**检验标准**
- 🟡 3 个 Agent 能协作完成一次代码审查，输出结构化的审查报告
- 🟡 任一 Agent 失败时，系统能降级处理（如跳过该 Agent 或用缓存结果）
- 🟡 能清晰解释"为什么不用 LangChain 自带的 Multi-Agent"

**避坑提示**
- ❌ **别让 Agent 无限对话**：多 Agent 很容易陷入"讨论-修改-再讨论"的死循环，必须设最大轮次
- ❌ **别把每个 Agent 都做得太复杂**：一个 Agent 只做一件事，做好一件事

---

#### Week 8：作品2——多 Agent 代码审查系统 + 阶段复盘

| 项目 | 内容 |
|------|------|
| **本周目标** | 交付作品 2：能对 GitHub PR/本地代码进行多 Agent 协作审查的系统 |

**每日任务**

| 天 | 任务 | 预计耗时 |
|----|------|----------|
| Day 1 | 产品设计与集成：确定功能——GitHub webhook 触发 / 手动提交 → 多 Agent 审查 → 生成报告 → 评论回写。画完整流程图 | 4h |
| Day 2 | GitHub 集成：实现 OAuth 登录、webhook 接收、PR diff 解析、评论回写 API。**关键**：只审查 diff，不审查全量代码（控制 token 消耗） | 6h |
| Day 3 | Agent 优化：根据 Week 7 的架构，优化各 Agent 的 system prompt，让 CC 帮你生成 20 个边界 case 并修复 | 6h |
| Day 4 | 审查报告 UI：用 Next.js 搭建报告展示页面——分级问题（Critical/Warning/Suggestion）、代码高亮定位、修复建议 | 6h |
| Day 5 | 性能优化：实现并行审查（3 个 Agent 并发执行）、结果缓存（相同 diff 不重复审查）、超时控制 | 5h |
| Day 6 | 部署 + 使用文档：Docker Compose 部署，写用户使用指南 + 开发者贡献指南。**用 GitHub Actions 做 CI** | 6h |
| Day 7 | 🧘 **阶段复盘与心理调整**：检查作品 1 和 2 是否都在线。如果焦虑（"别人都找到工作了"），这是正常的——你有 2 个上线作品，已经超过 80% 的竞争者 | 3h |

**本周产出** 见 [作品 2 详情](#作品2多-agent-协作的代码审查系统)

**阶段二复盘清单**
1. 作品 2 能否在面试中 5 分钟演示（提交一个 PR → 自动审查 → 看到报告）？
2. 你的 GitHub 仓库现在有几个？目标：**至少 6 个**（agent-core / prompt-templates / rag-playground / agent-frameworks / agent-memory / multi-agent-framework / 作品1 / 作品2）
3. 开始整理"面试问答清单"：把过去 8 周的踩坑记录转化为 20 个常见面试题的答案
4. **心态检查**：有没有因为"别人用 LangChain 我用自研"而焦虑？面试官更关心你的思考过程，不是工具选择

---

### 阶段三：工程化（W9-W12）

#### Week 9：生产化部署 + 监控

| 项目 | 内容 |
|------|------|
| **本周目标** | 将 Agent 应用从"能跑"升级到"生产可用"，掌握 Docker/CI/Monitoring 实战。**本周开始投递简历** |

**每日任务**

| 天 | 任务 | 预计耗时 |
|----|------|----------|
| Day 1 | Docker 深入：写一个"最佳实践"级的 Dockerfile（多阶段构建、非 root 用户、健康检查、.dockerignore）。对比 `python:3.12-slim` vs `alpine` 的镜像体积 | 5h |
| Day 2 | Docker Compose 生产配置：定义服务（API/Worker/Redis/PostgreSQL）、网络隔离、卷持久化、资源限制、重启策略 | 5h |
| Day 3 | CI/CD 搭建：用 GitHub Actions 实现——push → lint → test → build image → deploy。**关键**：test 步骤必须跑真实的 API 调用（用 mock） | 6h |
| Day 4 | 日志与监控：集成 structlog 结构日志、Prometheus metrics（请求量/延迟/错误率/LLM 调用次数/token 消耗）、Grafana 仪表盘 | 6h |
| Day 5 | 告警配置：设置关键指标告警——LLM API 错误率 > 5%、平均延迟 > 10s、日 token 消耗超预算。用 Grafana Alert 或自研 webhook | 5h |
| Day 6 | 成本优化：实现 token 预算管理系统——每个用户/会话有 token 上限、到达阈值触发告警/限流。写成本分析报告模板 | 5h |
| Day 7 | **简历初版完成**：根据 9 周的经历写简历初版（1 页中文/1 页英文）。用 STAR 法则描述项目。让 CC 帮你润色，你重点核对技术细节不夸大 | 4h |

**本周产出**
- ✅ 生产级 Docker 配置（含 CI/CD）
- ✅ 监控仪表盘（Grafana 截图可放入简历）
- ✅ 简历初版

**检验标准**
- 🟡 `docker compose up` 一键启动全部服务，`docker compose down` 完全清理
- 🟡 Grafana 能看到过去 24 小时的请求量/错误率/延迟分布
- 🟡 简历交由 3 个人（或 ChatGPT + 2 个朋友）review 过

**投递行动**
- 🟡 本周五开始，每天投 3-5 家（Boss 直聘 / 脉脉 / 拉勾）
- 🟡 优先投 AI 创业公司，外包公司作为保底
- 🟡 每次面试后当晚记录面试题，更新"面试问答清单"

**避坑提示**
- ❌ **别等到"准备好"再投**：永远没有准备好的时候。作品 2 上线就是信号
- ❌ **别花钱买监控服务**：Prometheus + Grafana 开源方案足够中小团队用
- ❌ **简历别写"精通"**：9 周经验写"熟练使用"，诚实的自信比虚假的牛逼更打动人

---

#### Week 10：高级 RAG + Agent 评估进阶

| 项目 | 内容 |
|------|------|
| **本周目标** | 掌握 RAG 进阶技术，建立完整的 Agent 质量保障体系 |

**每日任务**

| 天 | 任务 | 预计耗时 |
|----|------|----------|
| Day 1 | 高级 RAG 策略：实现 Self-RAG（检索后自我反思）、Corrective-RAG（检索结果质量评估+自动重试）、Graph RAG（知识图谱增强检索）的概念验证 | 6h |
| Day 2 | 多模态 RAG 入门：处理图片+文本混合文档。用 GPT-4V/Claude Vision 做图片理解，Text + Image Embedding 混合检索 | 5h |
| Day 3 | Agent 评估体系：参考 DeepEval/Ragas 的设计思路，搭建自己的 Agent 评估套件——包含 faithfulness、relevance、tool accuracy、latency、cost 五个维度 | 6h |
| Day 4 | 评估数据构建：用 LLM 自动生成测试集（Query + Expected Tool Calls + Expected Answer），人工抽检 30% 保证质量 | 5h |
| Day 5 | 持续评测管道：用 GitHub Actions 每天自动跑评估，生成趋势报告。评估结果恶化时自动告警（Prompt Drift 检测） | 5h |
| Day 6 | A/B 测试框架：实现 prompt/模型并排对比——同一条 query 同时发给两个版本，并排展现在 UI 中，人工标注胜负 | 5h |
| Day 7 | 周复盘：整理评估体系文档。开始**准备阿里云 ACA 考试**（如果还没考），刷 2 套题 | 3h |

**本周产出**
- ✅ `advanced-rag` 仓库（含 3 种高级 RAG 策略的实现与对比）
- ✅ Agent 评估套件（含自动化测试 + 趋势追踪）
- ✅ A/B 测试工具

**检验标准**
- 🟡 高级 RAG 策略在准确率上相比基础 RAG 提升 > 10%
- 🟡 评估套件能在 5 分钟内跑完 100 条测试 case
- 🟡 Prompt Drift 能在上线后 1 小时内被检测到

**避坑提示**
- ❌ **Graph RAG 别在生产中用**：图构建和维护成本高，面试里聊概念就行
- ❌ **别花太多时间在多模态上**：大多数 Agent 岗位现在还是文本为主

---

#### Week 11：工作流自动化 + MCP 协议

| 项目 | 内容 |
|------|------|
| **本周目标** | 掌握 Agent 工作流编排，理解并实现 MCP（Model Context Protocol），构建自动化工作流 Agent |

**每日任务**

| 天 | 任务 | 预计耗时 |
|----|------|----------|
| Day 1 | 工作流引擎设计：调研 Temporal/Prefect/Windmill。选择一个实现"任务依赖图"——任务 A 完成 → 触发任务 B 和 C 并行 → 汇总交给任务 D | 5h |
| Day 2 | 条件分支工作流：实现含 LLM 决策节点的动态工作流——LLM 判断用户意图 → 路由到不同处理分支 → 分支可动态创建 | 6h |
| Day 3 | MCP 协议深度：读 MCP 规范，实现一个 MCP Server（暴露工具）+ MCP Client（调用工具）。**关键**：理解 MCP 和 Function Calling 的关系 | 6h |
| Day 4 | 自定义工具开发：写 5 个 MCP 工具——SearchWeb、ReadFile、SendEmail、CreateCalendarEvent、QueryDatabase。每个工具含错误处理和超时 | 6h |
| Day 5 | 工作流可视化：用 React Flow 实现工作流编辑器和实时执行状态展示。让 CC 生成组件骨架，你重点做状态管理和数据流 | 5h |
| Day 6 | 集成测试：创建一个"每日晨报"工作流——早上 8 点触发 → 抓取 RSS/API 新闻 → Agent 总结 → 发邮件。端到端测试 | 5h |
| Day 7 | 周复盘：考**阿里云 ACA 云计算**（如果预约在本周）。整理工作流设计模式文档 | 3h |

**本周产出**
- ✅ `mcp-toolkit` 仓库（含 MCP Server/Client + 5 个工具）
- ✅ 工作流引擎 Demo（含可视化编辑器）
- ✅ "每日晨报"自动化工作流

**检验标准**
- 🟡 工作流编辑器能拖拽创建节点、连线、设置条件、触发执行
- 🟡 MCP Server 能同时被 Claude Code 和自己的 Client 调用
- 🟡 晨报工作流连续 3 天自动执行成功

**避坑提示**
- ❌ **别用 Temporal 做个人项目**：学习成本高，用 Prefect 或自研轻量方案
- ❌ **MCP Server 别暴露敏感操作**：SearchWeb 要限制 URL 范围，SendEmail 要确认

---

#### Week 12：作品3——自动化工作流 Agent + 阶段复盘

| 项目 | 内容 |
|------|------|
| **本周目标** | 交付作品 3：可视化的 AI 工作流自动化平台，能编排多个 Agent 和工具完成复杂业务流程 |

**每日任务**

| 天 | 任务 | 预计耗时 |
|----|------|----------|
| Day 1 | 产品设计：定义 10 个内置工作流模板（晨报/日报/周报/代码审查/文档翻译/竞品监控/邮件摘要/日程安排/数据报表/自选）。设计 UI/UX | 4h |
| Day 2 | 核心后端：实现模板管理 API、工作流执行引擎、动态工具注册、任务队列（用 Celery/Redis 或自研） | 6h |
| Day 3 | 前端搭建：工作流列表、模板市场、编辑器（含执行状态实时展示）、执行历史。**让 CC 生成 70% 前端代码** | 6h |
| Day 4 | 集成 MCP：让工作流中的节点可以调用 MCP 工具，实现"一个工具注册一次，所有工作流复用" | 5h |
| Day 5 | 定时触发与通知：支持 Cron 定时任务、Webhook 触发。通知渠道（邮件/飞书/企业微信），实现一个就行 | 5h |
| Day 6 | 全链路测试 + 部署：Docker 一键部署，写使用指南和 3 个演示视频脚本（黄金场景） | 6h |
| Day 7 | 🧘 **阶段复盘与心理调整**：现在你有 3 个完整作品了。如果面试还没进展，检查：简历是否突出了作品链接？是否主动发了 GitHub 链接给 HR？是否投的岗位匹配？ | 3h |

**本周产出** 见 [作品 3 详情](#作品3自动化工作流-agent)

**阶段三复盘清单**
1. 作品 1、2、3 是否都可在公网访问？（应对面试时"能看看吗"）
2. 简历更新到第几版了？每次面试后有没有迭代？
3. GitHub Profile 的 README 是否展示了 3 个作品的截图/GIF？
4. 投了多少家？面了几家？拿了几个 Offer？
5. **心态检查**：如果面试受挫，区分"技术不够"和"运气不好"。前者可以补，后者别内耗

---

### 阶段四：企业级冲刺（W13-W16）

#### Week 13：安全与 Guardrails

| 项目 | 内容 |
|------|------|
| **本周目标** | 掌握 LLM 应用安全防护体系，包括 Prompt Injection 防御、内容安全、数据隐私 |

**每日任务**

| 天 | 任务 | 预计耗时 |
|----|------|----------|
| Day 1 | 威胁建模：梳理 AI Agent 应用的攻击面——Prompt Injection（直接/间接）、数据泄露、工具滥用、DoS。画威胁模型图 | 5h |
| Day 2 | Guardrails 实现：用 Guardrails-AI 或自研实现——输入检测（越狱/敏感词）、输出检测（幻觉/隐私泄露）。写规则引擎 | 6h |
| Day 3 | Prompt Injection 防御：实现 3 层防御——输入净化（分隔符包围用户输入）、权限隔离（LLM 不能访问敏感环境变量）、工具调用人工确认（高危操作） | 6h |
| Day 4 | 内容审核集成：接入阿里云/腾讯云内容安全 API 做自动审核。实现审核结果的处理策略（拦截/警告/日志） | 5h |
| Day 5 | 数据脱敏：实现自动识别并脱敏身份证号/手机号/邮箱/银行卡号等敏感信息。**关键**：脱敏在发送给 LLM 之前做 | 5h |
| Day 6 | 安全测试：对作品 1-3 进行安全扫描，输出安全报告。用 OWASP Top 10 for LLM 逐项检查 | 5h |
| Day 7 | 周复盘：整理 AI 安全最佳实践文档。**报名软考**（中级软件设计师，如果时间/预算允许） | 3h |

**本周产出**
- ✅ `llm-security-guard` 仓库（含输入/输出护栏 + 数据脱敏）
- ✅ 作品 1-3 的安全测试报告
- ✅ AI 安全最佳实践文档

**检验标准**
- 🟡 常见越狱 prompt（如 DAN、角色扮演越狱）有 > 95% 的拦截率
- 🟡 敏感信息检测漏报率 < 1%
- 🟡 安全措施引入的额外延迟 < 500ms

**避坑提示**
- ❌ **别只依赖 LLM 做安全**：LLM 的输入净化层必须在 LLM 调用之前执行
- ❌ **别把 API Key 硬编码**：所有密钥用环境变量或 Secret Manager

---

#### Week 14：性能优化 + 成本控制

| 项目 | 内容 |
|------|------|
| **本周目标** | 掌握 LLM 应用的性能优化方法论，包括缓存策略、模型选型、成本优化 |

**每日任务**

| 天 | 任务 | 预计耗时 |
|----|------|----------|
| Day 1 | 性能分析：对作品 1-3 做全链路 profiling——请求处理、LLM 调用、后处理各环节的延迟分布。定位瓶颈 | 5h |
| Day 2 | 缓存策略：实现语义缓存（相似问题返回缓存结果）、精确缓存（相同 prompt 直接返回）、分层缓存（本地 + Redis） | 6h |
| Day 3 | 模型选型与降级：实现多模型路由——简单任务用小模型（GPT-4o-mini/Haiku）、复杂任务用大模型，大模型超时/报错时自动降级 | 5h |
| Day 4 | 并发控制：实现 LLM API 的并发控制——信号量限流、优先级队列、请求合并（多个相似请求批处理） | 6h |
| Day 5 | 成本分析系统：统计每个用户/功能/天 的 token 消耗，预测月度账单，生成优化建议 | 5h |
| Day 6 | 性能压测：用 Locust/K6 对作品做压力测试——100 QPS 下的表现。找到并修复性能瓶颈 | 5h |
| Day 7 | 周复盘：整理性能优化手册（含缓存/降级/并发三章） | 3h |

**本周产出**
- ✅ 语义缓存实现（GitHub 仓库可独立使用）
- ✅ 多模型路由系统
- ✅ 性能优化手册（含压测数据）

**检验标准**
- 🟡 语义缓存命中率 > 30%（基于模拟用户对话）
- 🟡 引入优化后 P99 延迟降低 > 50%
- 🟡 100 QPS 下错误率 < 1%

**避坑提示**
- ❌ **语义缓存别追求完美**：相似度阈值 0.85 就够用，别花 3 天调阈值
- ❌ **别过早优化**：先确认瓶颈在哪，别凭感觉优化

---

#### Week 15：企业级架构设计

| 项目 | 内容 |
|------|------|
| **本周目标** | 掌握企业级 AI 应用的架构设计模式，包括多租户、权限、审计、灾备 |

**每日任务**

| 天 | 任务 | 预计耗时 |
|----|------|----------|
| Day 1 | 多租户架构：设计 SaaS 级多租户方案——数据隔离（Database-per-tenant vs Schema-per-tenant vs Row-level）、资源配额（API 调用次/token/tools） | 5h |
| Day 2 | 权限系统：实现 RBAC（Admin/Developer/User）+ 工具级权限（哪些角色可以用 SendEmail？）。用 Casbin 或自研 | 6h |
| Day 3 | 审计日志：实现不可篡改的操作审计日志——谁、什么时间、调用了哪个 Agent/工具、输入/输出 hash、结果。**关键**：日志本身的安全（签名防篡改） | 6h |
| Day 4 | 高可用设计：实现 LLM API 多 provider 故障切换（OpenAI → Anthropic → DeepSeek，自动降级）、Redis/DB 主从、健康检查 + 自动重启 | 6h |
| Day 5 | API 网关设计：统一入口、认证（API Key / OAuth）、限流、请求/响应日志、版本管理 | 5h |
| Day 6 | 集成到作品 4 骨架：把 Day 1-5 的组件集成到作品 4 的基础架构中。写架构设计文档 | 5h |
| Day 7 | 周复盘 + 考**软考中级**（如果预约在本周）。软考对学历弱的人有帮助，可写进简历 | 3h |

**本周产出**
- ✅ `enterprise-agent-infra` 仓库（含多租户/权限/审计/高可用模块）
- ✅ 企业级 AI 应用架构设计文档
- ✅ API 网关实现

**检验标准**
- 🟡 两个租户的数据完全隔离，一个租户 A 无法访问租户 B 的数据
- 🟡 任一 LLM provider 宕机时，系统 < 5 秒内自动切换到备选
- 🟡 所有操作 100% 有审计记录，记录不可删除

**避坑提示**
- ❌ **别做过度设计**：多租户用 Row-level 就够了（项目里），别搞 Database-per-tenant
- ❌ **软考别花太多时间**：如果本周面试多，软考可以延期或取消

---

#### Week 16：作品4——企业级安全合规 Agent + 求职冲刺

| 项目 | 内容 |
|------|------|
| **本周目标** | 交付作品 4，展示企业级架构能力。同时全面进入求职冲刺状态 |

**每日任务**

| 天 | 任务 | 预计耗时 |
|----|------|----------|
| Day 1 | 产品定义：设计一个"AI 安全合规助手"——企业上传隐私政策/安全文档 → Agent 自动检查是否符合《个人信息保护法》/GDPR → 输出合规报告 + 整改建议 | 4h |
| Day 2 | 核心实现：文档解析 → 条款提取 → 法规匹配（RAG 检索相关法条）→ 差距分析 → 报告生成。整合 Week 13 的安全模块和 Week 15 的企业架构 | 6h |
| Day 3 | 前端 + 报告：合规仪表盘、文件上传、合规评分雷达图、整改建议清单。**让 CC 生成 80% 前端**，你重点做数据流 | 6h |
| Day 4 | 集成安全 Guardrails：输入文件扫描（防恶意文件）、输出报告脱敏、操作审计全量记录 | 5h |
| Day 5 | 全链路测试 + 部署：用 5 份真实隐私政策做端到端测试。Docker 部署，写使用文档 | 6h |
| Day 6 | **求职冲刺**：更新简历（加入作品 4）、整理 16 周学习笔记成"面试宝典"、录制 3 分钟作品展示视频。集中投递 20 家公司 | 5h |
| Day 7 | 🏁 **最终复盘**：回顾 16 周，统计投入产出（代码量/仓库数/文章数/面试数/Offer 数）。更新 LinkedIn/脉脉。**开始谈薪** | 3h |

**本周产出** 见 [作品 4 详情](#作品4企业级安全合规-agent)

---

## 3. 证书穿插计划

| 证书 | 时间 | 报名方式 | 费用 | 备考策略 | 性价比 |
|------|------|----------|------|----------|--------|
| **计算机二级 Python** | W3-W4 备考，W4-W5 考试 | [NCRE 官网](http://ncre.neea.edu.cn) | ~120 元 | 刷 5 套真题（未来教育/无忧考吧），备考 15h | ⭐⭐⭐ 基础证书，简历筛选时有比没有强 |
| **阿里云 ACA 云计算** | W6-W7 备考，W10-W11 考试 | 阿里云认证官网 / 免费券活动 | 0-600 元（用学生免费额度） | Udemy/ACloudGuru 课程 10h + 题库 5h | ⭐⭐⭐⭐ 云部署能力证明，和作品关联度高 |
| **软考中级 软件设计师** | W13-W14 备考，W15-W16 考试 | [软考官网](https://www.ruankao.org.cn) | ~200 元 | 历年真题 5 套 + 《软件设计师教程》，备考 30h | ⭐⭐⭐⭐⭐ 体制内/国企加分，学历弱时补位 |

> **注意**：证书是补位策略，不是核心。如果你到 W12 已经拿了 Offer，可以跳过软考。如果你面试时发现"学历被卡"，软考能加一点分。

---

## 4. 作品集规划

### 作品1：个人知识库问答 Agent

| 维度 | 详情 |
|------|------|
| **时间** | W3 积累技术，W4 集中开发 |
| **定位** | "能看懂你的文档并精准回答的 AI 助手" |
| **技术栈** | FastAPI（后端）+ Next.js + shadcn/ui（前端）+ ChromaDB/Milvus Lite（向量库）+ OpenAI/DeepSeek Embedding + SQLite（会话存储）+ Docker |
| **核心特性** | 多文件上传(PDF/DOCX/MD)、分块策略可视化对比、带引用的答案、多知识库切换、对话历史 |
| **架构描述** | 用户上传文件 → 后端解析并分块 → Embedding 模型向量化 → 存入向量数据库。用户提问 → 问题向量化 → 向量检索 Top-K → 拼接 Prompt → LLM 生成 → 答案 + 引用返回前端渲染 |
| **Prompt 设计** | System Prompt 用中文，明确角色（"你是一个知识库助手，只根据提供的文档内容回答问题"）、约束（"如果文档中没有相关信息，直接说'未找到相关内容'"）、输出格式要求（"在答案中用 [1] [2] 标注引用来源"） |
| **部署方案** | 前端 Vercel（免费）+ 后端 Railway（免费额度 $5/月，够用）+ 向量库 Milvus Lite（嵌入后端进程）|

### 作品2：多 Agent 协作的代码审查系统

| 维度 | 详情 |
|------|------|
| **时间** | W7 积累技术，W8 集中开发 |
| **定位** | "用多个 AI Agent 模拟专业代码审查团队的自动化工具" |
| **技术栈** | FastAPI + LangGraph（Agent 编排）+ GitHub API + Redis（消息队列）+ PostgreSQL（审查记录）+ Next.js（报告展示）+ Docker Compose |
| **核心特性** | GitHub PR Webhook 自动触发、3 个 Agent 分工协作（Reviewer/Researcher/Reporter）、结构化审查报告（问题分级+代码定位）、PR 评论自动回写、审查历史趋势图 |
| **架构描述** | GitHub Webhook → Orchestrator 解析 PR → 并发派发给 3 个 Agent（Reviewer 逐行审查 Diff → Researcher 检索相关文档和最佳实践 → Reporter 汇总并撰写报告）→ Orchestrator 合并报告 → 回写 PR Comment + 存储到 PostgreSQL |
| **Prompt 设计** | 3 个 Agent 各有一套 System Prompt——Reviewer（"你是资深代码审查员，关注安全漏洞、逻辑错误、性能问题"）、Researcher（"你负责检索相关技术文档和最佳实践，为 Reviewer 提供参考"）、Reporter（"你将审查意见和研究成果整合为结构清晰、可操作的审查报告"） |
| **部署方案** | 后端 Railway/阿里云免费 ECS + GitHub App（接收 Webhook）+ Redis Cloud 免费层 + Vercel（前端） |

### 作品3：自动化工作流 Agent

| 维度 | 详情 |
|------|------|
| **时间** | W11 积累技术，W12 集中开发 |
| **定位** | "拖拽式 AI 工作流平台，让 Agent 帮你自动完成重复性任务" |
| **技术栈** | FastAPI + React Flow（可视化编辑器）+ Celery/Redis（任务队列）+ MCP 协议（工具集成）+ PostgreSQL + Docker Compose |
| **核心特性** | 可视化工作流编辑（拖拽节点+连线）、10 个内置模板、MCP 工具热插拔、Cron/Webhook 触发、执行历史与重试、通知集成 |
| **架构描述** | 用户在画布拖拽节点构建工作流 → 保存为 DAG JSON → 触发器(Cron/Webhook/手动)启动 → Engine 解析 DAG → 按依赖关系执行节点 → 每个节点调用 LLM 或 MCP 工具 → 结果沿 DAG 传递 → 最终节点汇总输出 → 通知用户 |
| **Prompt 设计** | 动态 Prompt 生成——每个工作流节点根据其在 DAG 中的位置和前序节点的输出，动态构建 Prompt。例如"总结节点"会根据"信息收集节点"的输出自动填充 `{context}` 变量 |
| **部署方案** | 全部通过 Docker Compose 部署到阿里云免费 ECS（学生认证领 1 个月免费）+ 本地也可运行 |

### 作品4：企业级安全合规 Agent

| 维度 | 详情 |
|------|------|
| **时间** | W15 积累技术，W16 集中开发 |
| **定位** | "帮企业自动检查隐私政策合规性的 AI 审计工具" |
| **技术栈** | FastAPI + 多租户 Row-level 隔离 + Casbin（权限）+ Guardrails-AI（安全）+ 阿里云内容安全 API + PostgreSQL + Redis + Next.js + Docker Compose |
| **核心特性** | 隐私政策文件上传解析、法规知识库 RAG、合规差距分析、整改建议生成、评分雷达图、多租户数据隔离、操作审计日志、IP 白名单 |
| **架构描述** | 企业用户上传隐私政策 PDF → 文档解析→条款提取→向量检索相关法规（《个人信息保护法》/GDPR）→ Agent 对比分析"政策条款 vs 法规要求"→生成合规评分+差距报告+整改建议→输出到仪表盘。全过程经 Guardrails 输入检测+输出审核+审计记录 |
| **Prompt 设计** | 合规分析 Agent 的 System Prompt 需要极高的精确性——"你是数据合规审计专家，严格按照法规原文比对，不要添加主观判断"。输出格式为结构化 JSON，包含"法规条款/政策条款/符合度/差距描述/整改建议"五个字段 |
| **部署方案** | Docker Compose 完整部署 + 使用阿里云学生免费额度 + mock 内容审核 API（如果没预算） |

---

## 5. 求职时间轴

### 时间线

| 时间 | 行动 | 目标 |
|------|------|------|
| **W8 末** | 整理简历素材（项目/技术栈/证书），拍 3 分钟作品演示视频 | 素材齐备 |
| **W9 初** | 完成简历初版，发给 3 人 review | 简历定版 |
| **W9 中** | 开始在 Boss 直聘/脉脉/LinkedIn 投递 | 每天 3-5 家 |
| **W9-W12** | 边投边改：每次面试后当晚记录面经 → 更新简历 | 面试转化率 10%+ |
| **W13-W16** | 加大投递量，正面应对面试。作品 4 交付 | 拿 Offer |
| **W16 后** | 谈薪、对比 Offer、决策入职 | 入职 15K-25K |

### 分类型投递策略

| 公司类型 | 策略 | 展示重点 | 投递渠道 | 预期薪资 |
|----------|------|----------|----------|----------|
| **AI 创业公司（首选）** | 直接发 GitHub 链接给 CTO/技术负责人。邮件标题："我有 4 个上线的 AI Agent 项目，想和你们聊聊" | 作品 1-4 的公网链接 + 架构设计能力 | Boss 直聘/即刻/Twitter/Product Hunt | 18K-25K |
| **外包公司（保底）** | 走 HR 通道，简历强调"能独立交付"和"有上线项目"。外包公司 HR 不懂 Agent，用通俗语言描述 | 项目上线链接 + 全栈能力 + 快速交付 | Boss 直聘/拉勾 | 12K-18K |
| **远程工作（惊喜）** | 准备英文简历 + 英文作品介绍。在 LinkedIn/RemoteOK/Upwork 投。重点展示英文技术文档能力 | GitHub Profile 国际化 + 英文 Blog | LinkedIn/RemoteOK/WeWorkRemotely | $3K-$5K/月 |

### 学历短板话术模板

#### 版本1：简历版（放在"自我评价"区域）

> 在校期间以项目驱动学习，独立完成 4 个 AI Agent 应用的设计、开发与部署（含 RAG、多 Agent 协作、工作流自动化、企业安全合规）。所有项目均在 GitHub 开源并部署到公网，可随时演示。持续 16 周每日提交代码，保持高强度学习与产出节奏。已获得阿里云 ACA 云计算认证/软考中级软件设计师（选一）。

#### 版本2：面试版（HR 问"你学历一般，我们为什么选你？"）

> 我理解学历是筛选标准之一，我想用实际成果来回答这个问题。过去 16 周我独立交付了 4 个上线的 AI Agent 项目，从 RAG 到多 Agent 协作到企业级安全架构，覆盖了一个 AI 应用工程师需要掌握的全链路技术。这些项目都部署在公网，您可以现在打开看。我的学习方式是以产出为导向——不是"学过"某项技术，而是用它做出了什么东西。我相信对一个需要快速交付业务价值的岗位来说，能做事比学历更重要。

#### 版本3：自我介绍版（1 分钟）

> 我是 AI Agent 方向的开发者，过去 4 个月独立做了 4 个开源项目。第一个是能上传文档并精准问答的 RAG 系统，第二个是让多个 AI Agent 协作审查代码的工具，第三个是可视化 AI 工作流平台，第四个是企业级合规审计助手。所有项目都有线上 Demo，技术栈覆盖 Python/FastAPI/LangChain/向量数据库/MCP/Docker。我希望找一个能把我的原型能力转化为产品价值的团队。

---

## 6. 资源清单

### 阶段一资源（W1-W4）

| 类型 | 资源 | 优先级 |
|------|------|--------|
| 必读 | [Anthropic Prompt Engineering Guide](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering) | ⭐⭐⭐⭐⭐ |
| 必读 | [OpenAI Cookbook](https://cookbook.openai.com) | ⭐⭐⭐⭐⭐ |
| 必读 | [FastAPI 官方文档](https://fastapi.tiangolo.com) | ⭐⭐⭐⭐ |
| 必读 | [LlamaIndex Concepts](https://docs.llamaindex.ai/en/stable/getting_started/concepts/) | ⭐⭐⭐⭐ |
| 推荐视频 | 吴恩达《Building Systems with the ChatGPT API》(deeplearning.ai, 免费) | ⭐⭐⭐⭐ |
| 推荐视频 | 吴恩达《LangChain for LLM Application Development》(deeplearning.ai, 免费) | ⭐⭐⭐⭐ |
| 工具 | Claude Code：生成 80% 的样板代码，你重点改业务逻辑 | - |
| 工具 | Cursor：在已有项目中做重构/补全/修复 | - |
| 工具 | Trae：当需要快速搭建一个独立新页面/组件时用 | - |

### 阶段二资源（W5-W8）

| 类型 | 资源 | 优先级 |
|------|------|--------|
| 必读 | [LangGraph 官方文档](https://langchain-ai.github.io/langgraph/) | ⭐⭐⭐⭐⭐ |
| 必读 | [CrewAI 文档](https://docs.crewai.com) | ⭐⭐⭐⭐ |
| 必读 | [Microsoft AutoGen 论文 + 文档](https://microsoft.github.io/autogen/) | ⭐⭐⭐⭐ |
| 必读 | [LangSmith 文档](https://docs.smith.langchain.com) | ⭐⭐⭐ |
| 推荐课程 | [DeepLearning.AI Multi AI Agent Systems with crewAI](https://www.deeplearning.ai/short-courses/multi-ai-agent-systems-with-crewai/) (免费) | ⭐⭐⭐⭐ |
| 推荐课程 | [DeepLearning.AI AI Agentic Design Patterns with AutoGen](https://www.deeplearning.ai/short-courses/ai-agentic-design-patterns-with-autogen/) (免费) | ⭐⭐⭐⭐ |
| 工具 | Claude Code：Debug Agent 逻辑（把 trace 扔进去让它分析） | - |
| 工具 | LangSmith / LangFuse：Agent Trace 可视化 | - |

### 阶段三资源（W9-W12）

| 类型 | 资源 | 优先级 |
|------|------|--------|
| 必读 | [Docker 官方文档](https://docs.docker.com) | ⭐⭐⭐⭐⭐ |
| 必读 | [GitHub Actions 文档](https://docs.github.com/en/actions) | ⭐⭐⭐⭐ |
| 必读 | [MCP 规范](https://modelcontextprotocol.io) | ⭐⭐⭐⭐⭐ |
| 必读 | [Prometheus 文档](https://prometheus.io/docs) | ⭐⭐⭐ |
| 推荐视频 | [MCP 官方介绍 + 教程](https://modelcontextprotocol.io) | ⭐⭐⭐⭐ |
| 工具 | Claude Code：生成 Dockerfile/CI 配置/MCP Server 骨架 | - |
| 工具 | Cursor：复杂重构时用 | - |

### 阶段四资源（W13-W16）

| 类型 | 资源 | 优先级 |
|------|------|--------|
| 必读 | [OWASP Top 10 for LLM Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/) | ⭐⭐⭐⭐⭐ |
| 必读 | [Guardrails-AI 文档](https://www.guardrailsai.com/docs) | ⭐⭐⭐⭐ |
| 必读 | [NVIDIA NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails) | ⭐⭐⭐ |
| 推荐课程 | [DeepLearning.AI Building Systems with the ChatGPT API](https://www.deeplearning.ai/short-courses/building-systems-with-chatgpt/) (免费) | ⭐⭐⭐ |
| 推荐课程 | [DeepLearning.AI Safety & Alignment](https://www.deeplearning.ai/short-courses/) (查找最新安全课程) | ⭐⭐⭐ |
| 工具 | Locust/K6：性能压测 | - |
| 工具 | Claude Code：生成安全测试用例/压测脚本 | - |

### Claude Code / Cursor / Trae 使用策略

| 工具 | 使用场景 | 你不会用的时候 |
|------|----------|----------------|
| **Claude Code** | 1) 生成项目骨架/样板代码 2) Debug 复杂逻辑 3) 生成测试用例 4) 写文档/README | 直接让它解决你不会的问题，你审阅代码并理解 |
| **Cursor** | 1) 在已有项目中重构 2) Tab 补全提高编码速度 3) 多文件联动修改 | 先手动改一个文件理解模式，再让它批量处理 |
| **Trae** | 1) 独立新页面原型 2) 快速尝试"这个库能不能用" | 用完就扔，别在上面维护正式项目 |

> **核心原则**：AI 工具用于加速，不是替代理解。你能向面试官解释每一行关键代码，才是学到位了。

---

## 附录：每周时间分配建议

| 时间段 | 活动 | 说明 |
|--------|------|------|
| 上午 2-3h | 编码实战 | 最高效的时段留给写代码 |
| 下午 2-3h | 阅读文档+看课程+做笔记 | 适合理解性学习 |
| 晚上 1h | 复盘+写文章+更新 GitHub | 产出物沉淀 |
| 周日 2h | 周复盘+下周规划 | 轻量复盘，别学新东西 |
| 每 4 周末 | 阶段复盘+心理调整 | 检查进度，调整方向 |

---

> **最后提醒**：这个计划是一个地图，不是牢笼。如果某个技术栈你学得特别快，就加速；如果某个项目你特别感兴趣，就深挖。16 周后，你的竞争力不是"完成了一个计划"，而是"有 4 个让别人能打开用的产品 + 能讲清楚每个技术决策的思考过程"。

> 现在，打开 Claude Code，开始 Week 1 Day 1 的环境搭建。Go.
