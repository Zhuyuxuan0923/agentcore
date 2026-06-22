"""
ChromaDB 中文语义搜索 — 完整 RAG 检索演示
Week 3 Day 2 核心 Demo

本 Demo 使用手动构造的向量（免下载、免 API Key），展示：
  1. ChromaDB + 手动向量 的写入与查询
  2. 语义搜索 vs 关键词搜索的本质区别
  3. 元数据过滤（语义 + 精确条件组合）
  4. 相似度阈值的作用

运行: poetry run python src/study_agent/chromadb_openai_demo.py

注：真实 RAG 项目中，手动向量替换为 OpenAI text-embedding-3-small 即可。
    概念和 API 调用方式完全一样，只是向量的来源不同。
"""

import chromadb


# ── 生成模拟的"语义向量"（8维示意，真实是1536维）──
# 原理：同类文档的向量在空间中"靠近"。
#   请假/报销类 → 第1维高（HR政策相关）
#   IT/技术类    → 第2维高
#   招聘类       → 第3维高
#   编程规范类   → 第4维高
def make_vector(topic: str) -> list[float]:
    """根据主题生成一个8维示意向量"""
    base = [0.02] * 8
    if topic == "请假":
        base[0] = 0.92
        base[1] = 0.04
    elif topic == "报销":
        base[0] = 0.85
        base[1] = 0.05
    elif topic == "IT":
        base[1] = 0.95
        base[0] = 0.03
    elif topic == "招聘":
        base[2] = 0.90
        base[3] = 0.05
    elif topic == "编程规范":
        base[3] = 0.88
        base[4] = 0.08
    elif topic == "请假+病假":
        base[0] = 0.94
    elif topic == "报销+住宿":
        base[0] = 0.88
    elif topic == "IT+网络":
        base[1] = 0.90
        base[5] = 0.08
    elif topic == "编程+审查":
        base[3] = 0.92
    return base


def main():
    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection(name="chinese_docs")

    # ══════════════════════════════════════════════════════════
    # 第 1 步：准备中文文档并存入 ChromaDB
    # ══════════════════════════════════════════════════════════
    docs = [
        "员工报销流程：员工需在系统中填写《报销申请单》，附上发票照片，"
        "提交给直属上级审批。审批通过后财务部在 5 个工作日内打款。",
        "年假政策：入职满 1 年的员工享有 5 天带薪年假，满 3 年 10 天，"
        "满 5 年 15 天。年假需提前一周向直属上级申请。",
        "办公室 Wi-Fi 密码为 Study@2026，5G 频段。如遇网络故障请拨打"
        "IT 服务热线 8888，工作日 9:00-18:00 有人值班。",
        "招聘流程：各部门在系统中提交 HC 申请，HR 发布职位并筛选简历，"
        "面试分为技术面、HR 面和终面三轮。整个流程约需 2-3 周。",
        "Python 编程规范：所有代码需通过 Black 格式化，line-length=100，"
        "使用 Ruff 做 lint 检查，提交前必须运行 pre-commit hooks。",
        "请假规定：病假需提供医院证明，事假每年累计不超过 10 天。"
        "紧急情况可事后补请假申请，但需在 3 个工作日内完成。",
        "出差报销标准：一线城市住宿标准 500 元/晚，二线城市 350 元/晚。"
        "交通费实报实销，餐补每天 100 元。所有报销需提供正规发票。",
        "代码审查流程：PR 提交后需至少一位团队成员 Code Review 通过，"
        "所有 CI 检查（lint / test / typecheck）必须通过后方可合并到 main 分支。",
    ]

    topics = ["报销", "请假", "IT", "招聘", "编程规范", "请假+病假", "报销+住宿", "编程+审查"]

    collection.add(
        documents=docs,
        embeddings=[make_vector(t) for t in topics],
        metadatas=[
            {"category": "人事", "source": "员工手册"},
            {"category": "人事", "source": "员工手册"},
            {"category": "IT", "source": "员工手册"},
            {"category": "人事", "source": "员工手册"},
            {"category": "IT", "source": "编程规范"},
            {"category": "人事", "source": "员工手册"},
            {"category": "财务", "source": "员工手册"},
            {"category": "IT", "source": "编程规范"},
        ],
        ids=[f"doc-{i}" for i in range(len(docs))],
    )
    print(f"已存入 {collection.count()} 篇中文文档\n")

    # ══════════════════════════════════════════════════════════
    # 第 2 步：语义搜索 — 四个真实查询
    # ══════════════════════════════════════════════════════════
    queries = [
        ("我生病了怎么请假？", "请假+病假"),
        ("出差住酒店能报销多少钱？", "报销+住宿"),
        ("代码提交流程是什么？", "编程+审查"),
        ("Wi-Fi 连不上了怎么办？", "IT+网络"),
    ]

    SIMILARITY_THRESHOLD = 0.7  # 余弦相似度阈值

    for query_text, query_topic in queries:
        results = collection.query(
            query_embeddings=[make_vector(query_topic)],
            n_results=4,
        )

        print(f"{'='*55}")
        print(f"Query: {query_text}")
        print(f"       (查询向量主题: {query_topic})")
        print(f"{'-'*55}")

        for rank, (doc_id, text, dist, meta) in enumerate(
            zip(
                results["ids"][0],
                results["documents"][0],
                results["distances"][0],
                results["metadatas"][0],
            )
        ):
            similarity = 1 - dist
            if similarity >= SIMILARITY_THRESHOLD:
                status = "PASS"
            elif similarity >= 0.5:
                status = "WEAK"
            else:
                status = "FAIL"

            print(f"  #{rank+1} [{status}] sim={similarity:.4f} dist={dist:.4f}")
            print(f"       {text[:65]}...")
            print(f"       分类={meta['category']}, 来源={meta['source']}")
        print()


if __name__ == "__main__":
    main()
