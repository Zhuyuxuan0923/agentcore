"""
Week 3 Day 3 — 文档分块与索引流水线 完整 Demo

运行: poetry run python src/study_agent/w3d3_chunking_demo.py

本 Demo 展示
  1. 三种分块策略的对比固定大小 / 句子边界 / 递归分块
  2. chunk_overlap 的可视化效果
  3. 完整索引流水线文件  分块  Embedding  ChromaDB
"""

import math
from pathlib import Path

import chromadb
from openai import OpenAI

# ══════════════════════════════════════════════════════════════════
# 第 1 部分三种分块策略
# ══════════════════════════════════════════════════════════════════

# ── 示例文档模拟一份真实的公司内部文档──
SAMPLE_DOC = """
# 员工手册2026 版

## 第一章入职指引

欢迎加入公司入职第一天请携带身份证学历证书原件到前台办理入职手续
HR 会为你分配工位电脑设备和企业邮箱账号入职培训时长为 2 天涵盖公司文化
规章制度信息安全等模块培训结束后需要在系统中完成在线考试80 分及格

## 第二章考勤与假期

公司实行弹性工作制核心工作时间为 10:00-16:00其余时间可自由安排
考勤通过企业微信打卡忘记打卡需在 24 小时内提交补卡申请

年假政策入职满 1 年的员工享有 5 天带薪年假满 3 年 10 天满 5 年 15 天
年假需提前一周向直属上级申请病假需提供医院证明事假每年累计不超过 10 天
紧急情况可事后补请假申请但需在 3 个工作日内完成审批

## 第三章报销制度

员工报销需在系统中填写报销申请单附上发票照片提交给直属上级审批
审批通过后财务部在 5 个工作日内打款到工资卡出差报销标准一线城市住宿标准
500 元/晚二线城市 350 元/晚交通费实报实销餐补每天 100 元
所有报销需提供正规发票发票抬头为公司全称

## 第四章IT 与安全

办公室 Wi-Fi 密码为 Study@20265G 频段如遇网络故障请拨打 IT 服务热线 8888
工作日 9:00-18:00 有人值班公司配备的电脑不得安装盗版软件不得将公司代码
上传到公共代码托管平台所有内部文档不得通过个人微信/QQ 外传

## 第五章代码规范

所有代码需通过 Black 格式化line-length=100使用 Ruff 做 lint 检查
提交前必须运行 pre-commit hooks单元测试覆盖率需大于 80%
PR 提交后需至少一位团队成员 Code Review 通过所有 CI 检查必须通过后方可合并到 main 分支
"""


def fixed_size_chunk(text: str, chunk_size: int = 120) -> list[dict]:
    """策略 A固定大小分块——每 chunk_size 个字符切一刀

    这是最原始的策略就像切豆腐——不管你豆腐里有什么一刀下去
    """
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk_text = text[i : i + chunk_size]
        if chunk_text.strip():
            chunks.append(
                {
                    "text": chunk_text,
                    "start": i,
                    "end": min(i + chunk_size, len(text)),
                    "strategy": "fixed_size",
                }
            )
    return chunks


def sentence_chunk(text: str, max_chunk_size: int = 200) -> list[dict]:
    """策略 B句子边界分块——在句号问号感叹号换行处切

    这就像裁缝剪布——沿着缝线句子边界剪不会把图案剪烂
    """
    # 找所有句子边界
    boundaries = []
    for i, ch in enumerate(text):
        if ch in "\n" and i > 0:
            boundaries.append(i + 1)  # +1 表示包含这个标点

    if not boundaries:
        return [{"text": text, "start": 0, "end": len(text), "strategy": "sentence"}]

    chunks = []
    start = 0
    for boundary in boundaries:
        # 如果当前积累的文本加上这句还没超限就继续积累
        if boundary - start <= max_chunk_size:
            continue
        # 超限了把积累的文本切出来
        chunk_text = text[start:boundary].strip()
        if chunk_text:
            chunks.append(
                {
                    "text": chunk_text,
                    "start": start,
                    "end": boundary,
                    "strategy": "sentence",
                }
            )
        start = boundary

    # 收尾最后一段
    if start < len(text):
        chunk_text = text[start:].strip()
        if chunk_text:
            chunks.append(
                {
                    "text": chunk_text,
                    "start": start,
                    "end": len(text),
                    "strategy": "sentence",
                }
            )
    return chunks


def recursive_chunk(text: str, chunk_size: int = 150) -> list[dict]:
    """策略 C递归分块——按分隔符优先级递归切分

    优先级双换行 > 单换行 > 句号 > 固定大小
    这就像整理文件柜——先按年份分再按月份分最后按日期分
    """
    separators = ["\n\n", "\n", "", "", "", "", " "]

    def _split(t: str, sep_idx: int = 0) -> list[str]:
        if sep_idx >= len(separators):
            # 最后手段固定大小硬切
            return [t[i : i + chunk_size] for i in range(0, len(t), chunk_size)]

        sep = separators[sep_idx]
        if sep not in t:
            return _split(t, sep_idx + 1)

        parts = t.split(sep)
        result = []
        for part in parts:
            if len(part) <= chunk_size:
                result.append(part)
            else:
                # 这段还是太大用下一级分隔符继续切
                result.extend(_split(part, sep_idx + 1))
        return result

    pieces = _split(text)
    chunks = []
    pos = 0
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        # 在原文中找到这段的位置近似
        idx = text.find(piece, pos)
        if idx == -1:
            idx = pos
        chunks.append(
            {
                "text": piece,
                "start": idx,
                "end": idx + len(piece),
                "strategy": "recursive",
            }
        )
        pos = idx + len(piece)
    return chunks


# ══════════════════════════════════════════════════════════════════
# 第 2 部分可视化对比
# ══════════════════════════════════════════════════════════════════


def print_chunks(chunks: list[dict], strategy_name: str):
    """把分块结果友好地打印出来"""
    print(f"\n{'='*60}")
    print(f"  {strategy_name}")
    print(f"  共 {len(chunks)} 个 chunk")
    print(f"{'='*60}")

    for i, chunk in enumerate(chunks):
        text_preview = chunk["text"].replace("\n", "")[:80]
        print(
            f"\n  ┌─ Chunk #{i+1} (字符 {chunk['start']}-{chunk['end']}, 长度 {len(chunk['text'])})"
        )
        print(f"  │  {text_preview}...")


# ══════════════════════════════════════════════════════════════════
# 第 3 部分chunk_overlap 效果演示
# ══════════════════════════════════════════════════════════════════


def chunk_with_overlap(text: str, chunk_size: int = 80, overlap: int = 20) -> list[dict]:
    """固定大小分块 + overlap

    overlap 就像拍照时的"重叠区"——两张相邻照片有一部分重叠
    这样拼全景图时不会漏掉画面
    """
    chunks = []
    step = chunk_size - overlap  # 每次前进的步长
    if step <= 0:
        raise ValueError(f"overlap ({overlap}) 必须小于 chunk_size ({chunk_size})")

    i = 0
    while i < len(text):
        chunk_text = text[i : i + chunk_size]
        if chunk_text.strip():
            chunks.append(
                {
                    "text": chunk_text,
                    "start": i,
                    "end": min(i + chunk_size, len(text)),
                    "strategy": f"fixed+overlap({overlap})",
                }
            )
        i += step
        # 如果已经是最后一段退出
        if i + chunk_size > len(text) and i < len(text):
            # 保证最后一段也被包含
            last_chunk = text[-chunk_size:]
            if chunks and chunks[-1]["text"] != last_chunk:
                chunks.append(
                    {
                        "text": last_chunk,
                        "start": len(text) - chunk_size,
                        "end": len(text),
                        "strategy": f"fixed+overlap({overlap})",
                    }
                )
            break
    return chunks


def demonstrate_overlap():
    """用一个短例子展示 overlap 的价值"""
    text = (
        "第一章Python 异步编程简介"
        "Python 的 asyncio 库提供了事件循环机制允许在单线程中并发执行多个协程"
        "使用 async def 定义协程函数await 关键字用于等待协程执行完成"
        "第二章实战案例本案例展示了一个异步 Web 爬虫的实现"
    )

    print("\n\n" + "=" * 60)
    print("  chunk_overlap 效果演示")
    print("=" * 60)
    print(f"\n  原文{len(text)} 字:")
    print(f"  {text}")

    # 无 overlap
    no_overlap = chunk_with_overlap(text, chunk_size=50, overlap=0)
    print(f"\n  ── chunk_size=50, overlap=0  {len(no_overlap)} 个 chunk ──")
    for i, c in enumerate(no_overlap):
        print(f"  Chunk #{i+1}: {c['text']}")

    # 有 overlap
    with_overlap = chunk_with_overlap(text, chunk_size=50, overlap=15)
    print(f"\n  ── chunk_size=50, overlap=15  {len(with_overlap)} 个 chunk ──")
    for i, c in enumerate(with_overlap):
        # 标出重叠部分
        if i > 0:
            prev_end = with_overlap[i - 1]["end"]
            overlap_start = c["start"]
            overlap_len = prev_end - overlap_start
            if overlap_len > 0:
                print(f"  Chunk #{i+1}: {c['text']}  前 {overlap_len} 字与上一块重叠")
            else:
                print(f"  Chunk #{i+1}: {c['text']}")
        else:
            print(f"  Chunk #{i+1}: {c['text']}")


# ══════════════════════════════════════════════════════════════════
# 第 4 部分完整索引流水线
# ══════════════════════════════════════════════════════════════════


def build_index_pipeline(file_path: str):
    """完整流水线读取文件  分块  Embedding  存入 ChromaDB  语义查询

    这就是 RAG 系统中"入库Indexing"阶段的实际实现
    """
    # ── Step 1: 读取文件 ──
    filepath = Path(file_path)
    if not filepath.exists():
        print(f"文件不存在: {file_path}")
        return

    raw_text = filepath.read_text(encoding="utf-8")
    print(f" 读取文件: {filepath.name} ({len(raw_text)} 字符)")

    # ── Step 2: 分块 ──
    # 用递归分块策略每个 chunk 约 200 字符
    chunks = recursive_chunk(raw_text, chunk_size=200)
    print(f"  分块完成: {len(chunks)} 个 chunk递归分块chunk_size200")

    # ── Step 3: 用 OpenAI 生成 Embedding ──
    print(" 正在生成 Embedding...")
    openai_client = OpenAI()
    chunk_texts = [c["text"] for c in chunks]

    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=chunk_texts,
    )
    embeddings = [item.embedding for item in response.data]
    print(f"   已为 {len(embeddings)} 个 chunk 生成向量每向量 {len(embeddings[0])} 维")

    # ── Step 4: 存入 ChromaDB ──
    chroma_client = chromadb.EphemeralClient()
    collection = chroma_client.get_or_create_collection(name="w3d3_index_demo")

    collection.add(
        documents=chunk_texts,
        embeddings=embeddings,
        metadatas=[
            {
                "source": filepath.name,
                "chunk_index": i,
                "char_start": c["start"],
                "char_end": c["end"],
            }
            for i, c in enumerate(chunks)
        ],
        ids=[f"{filepath.stem}-chunk-{i}" for i in range(len(chunks))],
    )
    print(f" 已存入 ChromaDB: Collection 共 {collection.count()} 条数据")

    # ── Step 5: 测试语义搜索 ──
    print("\n" + "=" * 60)
    print("  测试语义搜索")
    print("=" * 60)

    test_queries = [
        "年假怎么申请",
        "出差补助标准是多少",
        "代码审查要做什么",
        "Wi-Fi 密码是什么",
    ]

    for query in test_queries:
        # 把查询问题也 Embedding
        q_embedding = (
            openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=[query],
            )
            .data[0]
            .embedding
        )

        results = collection.query(
            query_embeddings=[q_embedding],
            n_results=2,
        )

        print(f"\n   问题: {query}")
        for rank, (doc_id, text, dist, meta) in enumerate(
            zip(
                results["ids"][0],
                results["documents"][0],
                results["distances"][0],
                results["metadatas"][0],
            )
        ):
            similarity = 1 - dist
            preview = text.replace("\n", " ")[:70]
            print(
                f"    #{rank+1} [相似度={similarity:.4f}] (chunk #{meta['chunk_index']}) {preview}..."
            )

    return collection


# ══════════════════════════════════════════════════════════════════
# 第 5 部分提取文档章节结构
# ══════════════════════════════════════════════════════════════════


def extract_chapter_structure(text: str):
    """从 Markdown 文档中提取 ## 标题结构辅助理解文档骨架

    RAG 系统中章节结构可以存为 metadata 的 'section' 字段
    增强检索的可追溯性
    """
    chapters = []
    for line in text.split("\n"):
        if line.startswith("## "):
            chapters.append(line.strip("# ").strip())
    return chapters


# ══════════════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════════════


def main():
    print("╔══════════════════════════════════════════════════════╗")
    print("║  Week 3 Day 3 — 文档分块与索引流水线                    ║")
    print("╚══════════════════════════════════════════════════════╝")

    # ── Part 1: 三种分块策略对比 ──
    print("\n\n  === 第一部分三种分块策略对比 ===")

    print_chunks(
        fixed_size_chunk(SAMPLE_DOC, chunk_size=120), "策略 A固定大小分块 (chunk_size=120)"
    )
    print_chunks(
        sentence_chunk(SAMPLE_DOC, max_chunk_size=200), "策略 B句子边界分块 (max_chunk_size=200)"
    )
    print_chunks(recursive_chunk(SAMPLE_DOC, chunk_size=150), "策略 C递归分块 (chunk_size=150)")

    # ── Part 2: Overlap 演示 ──
    print("\n\n  === 第二部分chunk_overlap 效果演示 ===")
    demonstrate_overlap()

    # ── Part 3: 文档结构提取 ──
    print("\n\n  === 第三部分文档章节结构提取 ===")
    chapters = extract_chapter_structure(SAMPLE_DOC)
    print(f"  检测到 {len(chapters)} 个章节:")
    for ch in chapters:
        print(f"     {ch}")

    # ── Part 4: 模拟完整流水线不用 API用示意向量 ──
    print("\n\n  === 第四部分完整索引流水线示意向量版 ===")
    simulate_index_pipeline(SAMPLE_DOC)


def simulate_index_pipeline(text: str):
    """用示意向量模拟完整流水线不调用 OpenAI API

    这样你可以随时运行随时实验不消耗 API 额度
    真实流水线换成 OpenAI Embedding 即可——概念和流程完全一样
    """
    # Step 1 & 2: 分块
    chunks = recursive_chunk(text, chunk_size=200)
    print(f"Step 1-2: 读取并分块  {len(chunks)} 个 chunk")

    # Step 3: 模拟 Embedding用简单的统计向量替代真实 Embedding
    def simulate_embedding(text: str, dim: int = 8) -> list[float]:
        """用一个简单的统计特征做示意 Embedding

        这不是真实的语义向量只是让你看到流水线跑通的样子
        真实代码中这里换成 OpenAI embeddings.create()
        """
        vec = [0.0] * dim
        keywords_map = {
            0: ["报销", "发票", "财务", "打款", "出差", "住宿", "餐补", "交通"],
            1: ["请假", "年假", "病假", "事假", "假期", "考勤", "打卡"],
            2: ["代码", "PR", "CI", "lint", "test", "Black", "Ruff", "pre-commit"],
            3: ["Wi-Fi", "密码", "网络", "IT", "电脑", "安全"],
            4: ["入职", "培训", "HR", "手续", "分配"],
            5: ["规范", "制度", "政策", "标准", "规定"],
        }
        for dim_idx, keywords in keywords_map.items():
            for kw in keywords:
                if kw in text:
                    vec[dim_idx] += 0.3
        # 归一化让所有向量在同一尺度上可比
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    embeddings = [simulate_embedding(c["text"]) for c in chunks]
    print(f"Step 3: 生成 Embedding  {len(embeddings)} 个 {len(embeddings[0])} 维向量模拟")

    # Step 4: 存入 ChromaDB
    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection(
        name="w3d3_pipeline_demo",
        metadata={"hnsw:space": "cosine"},
    )

    collection.add(
        documents=[c["text"] for c in chunks],
        embeddings=embeddings,
        metadatas=[
            {"chunk_id": i, "char_range": f"{c['start']}-{c['end']}"} for i, c in enumerate(chunks)
        ],
        ids=[f"chunk-{i}" for i in range(len(chunks))],
    )
    print(f"Step 4: 存入 ChromaDB  Collection 有 {collection.count()} 条数据")

    # Step 5: 语义搜索测试
    print(f"\n{'='*55}")
    print("  Step 5: 语义搜索验证")
    print(f"{'='*55}")

    queries = [
        ("年假怎么申请", [0, 1]),  # 期望命中"请假"相关
        ("代码提交流程", [0, 2]),  # 期望命中"代码规范"相关
        ("出差住酒店报销标准", [0, 0]),  # 期望命中"报销"相关
        ("入职要带什么材料", [0, 4]),  # 期望命中"入职"相关
    ]

    for query_text, _ in queries:
        q_vec = simulate_embedding(query_text)
        results = collection.query(query_embeddings=[q_vec], n_results=2)

        print(f"\n   {query_text}")
        for rank, (doc_id, text, dist) in enumerate(
            zip(results["ids"][0], results["documents"][0], results["distances"][0])
        ):
            similarity = 1 - dist
            preview = text.replace("\n", " ")[:80]
            bar = "" * int(similarity * 20)
            print(f"    #{rank+1} [{bar}] sim={similarity:.4f}")
            print(f"        {preview}...")

    print(f"\n{'='*55}")
    print("  流水线总结:")
    print(f"    输入: 1 个文档 ({len(text)} 字符)")
    print(f"    分块: {len(chunks)} 个 chunk")
    print("    索引: 存入 ChromaDB (Collection: w3d3_pipeline_demo)")
    print("    查询: 4 个问题全部完成语义搜索")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
