"""
ChromaDB 入门演示 — 第一个向量数据库程序
Week 3 Day 2 核心代码

注意：这里用"手动向量"代替 ChromaDB 内置的 Embedding 模型，
避免首次运行时下载 80MB 的 all-MiniLM-L6-v2 模型。
实际项目中你会用 OpenAI text-embedding-3-small 生成向量（见 chromadb_openai_demo.py）。

运行: poetry run python src/study_agent/chromadb_demo.py
"""

import chromadb


# ── 辅助：生成简单的"语义向量"（4维示意，真实是1536维）──
# 这里手动构造向量让"同类"文档的向量接近：
#   编程类 → 向量第1维高
#   美食类 → 向量第2维高
#   其他类 → 向量第3维高
def make_vector(topic: str) -> list[float]:
    """根据主题生成一个4维示意向量"""
    if topic == "编程":
        return [0.9, 0.05, 0.03, 0.02]
    elif topic == "美食":
        return [0.05, 0.9, 0.03, 0.02]
    elif topic == "IT":
        return [0.8, 0.05, 0.1, 0.05]
    elif topic == "行政":
        return [0.03, 0.03, 0.9, 0.04]
    elif topic == "运维":
        return [0.7, 0.05, 0.2, 0.05]
    else:
        return [0.1, 0.1, 0.1, 0.7]


def demo_first_collection():
    """演示：创建 Collection → 写入（带向量）→ 语义查询"""
    print("=" * 50)
    print("1. 第一个 ChromaDB 程序")
    print("=" * 50)

    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection(name="my_first_collection")

    # add 时同时传入 documents（原文）和 embeddings（向量）
    # 如果只传 documents 不传 embeddings，ChromaDB 会自动下载内置模型来生成向量
    collection.add(
        documents=[
            "Python 是一种解释型编程语言，广泛用于数据科学和 AI 开发。",
            "红烧肉的经典做法：五花肉焯水后加冰糖炒糖色，再慢炖两小时。",
            "Git 是分布式版本控制系统，用于跟踪代码变更和团队协作。",
            "制作提拉米苏需要马斯卡彭奶酪、手指饼干和浓缩咖啡。",
        ],
        embeddings=[
            make_vector("编程"),  # Python → 编程类向量
            make_vector("美食"),  # 红烧肉 → 美食类向量
            make_vector("编程"),  # Git → 编程类向量
            make_vector("美食"),  # 提拉米苏 → 美食类向量
        ],
        ids=["doc-1", "doc-2", "doc-3", "doc-4"],
    )

    # 查询时也手动传入查询向量（"美食"话题的向量）
    results = collection.query(
        query_embeddings=[make_vector("美食")],  # 代替 query_texts
        n_results=2,
    )

    print("查询向量 → 美食类（第2维高）")
    print("期望结果 → 红烧肉、提拉米苏 排在前面")
    print("-" * 40)
    for i, (doc_id, text, distance) in enumerate(
        zip(results["ids"][0], results["documents"][0], results["distances"][0])
    ):
        print(f"  #{i+1} | ID: {doc_id} | 距离: {distance:.4f}")
        print(f"       | {text}")
    print()


def demo_crud():
    """演示：ChromaDB 的增删改查完整操作"""
    print("=" * 50)
    print("2. CRUD 完整操作")
    print("=" * 50)

    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection(name="crud_demo")

    # Create — 手动传 embeddings 避免下载内置模型
    collection.add(
        documents=["Python 入门教程", "Java 入门教程", "C++ 入门教程"],
        embeddings=[[0.9, 0.05, 0.05], [0.85, 0.1, 0.05], [0.8, 0.15, 0.05]],
        ids=["py", "java", "cpp"],
    )
    print(f"  add 后 count = {collection.count()}")  # 3

    # Read — 按 ID 精确获取（不需要向量）
    item = collection.get(ids=["py"])
    print(f"  get('py') = {item['documents']}")

    # Update — 更新文档时也要传 embeddings，否则触发生成模型下载
    collection.update(
        ids=["py"],
        documents=["Python 从入门到精通（已更新版）"],
        embeddings=[[0.9, 0.05, 0.05]],
    )
    updated = collection.get(ids=["py"])
    print(f"  update 后 = {updated['documents']}")

    # Delete
    collection.delete(ids=["cpp"])
    print(f"  delete 后 count = {collection.count()}")  # 2

    all_ids = collection.get()["ids"]
    print(f"  剩余 ids = {all_ids}")
    print()


def demo_metadata_filter():
    """演示：元数据过滤 — 语义搜索 + 精确条件组合"""
    print("=" * 50)
    print("3. 元数据过滤")
    print("=" * 50)

    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection(name="metadata_demo")

    collection.add(
        documents=[
            "Wi-Fi 密码及网络故障报修流程",
            "办公用品申领流程",
            "数据库备份操作指南",
        ],
        embeddings=[
            make_vector("IT"),
            make_vector("行政"),
            make_vector("运维"),
        ],
        metadatas=[
            {"category": "IT", "source": "员工手册 v2.0"},
            {"category": "行政", "source": "员工手册 v2.0"},
            {"category": "IT", "source": "运维手册 v1.0"},
        ],
        ids=["it-wifi", "admin-office", "it-backup"],
    )

    # 用 IT 类向量查询 + 只搜 IT 分类
    results = collection.query(
        query_embeddings=[make_vector("IT")],
        n_results=3,
        where={"category": "IT"},
    )

    print("查询: IT类向量 + 仅IT分类")
    print("-" * 40)
    for i, (doc_id, text, meta, dist) in enumerate(
        zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ):
        print(f"  #{i+1} | ID: {doc_id} | 分类: {meta['category']} | 距离: {dist:.4f}")
        print(f"       | {text}")
    print()


def demo_persistent():
    """演示：PersistentClient 持久化存储"""
    print("=" * 50)
    print("4. 持久化存储 (PersistentClient)")
    print("=" * 50)

    import os
    import shutil
    import tempfile

    tmpdir = tempfile.mkdtemp()
    try:
        path = os.path.join(tmpdir, "chroma_data")

        # 第一次运行：创建并写入
        client1 = chromadb.PersistentClient(path=path)
        col1 = client1.get_or_create_collection(name="persistent_demo")
        col1.add(
            documents=["持久化测试文档"],
            embeddings=[[0.5, 0.5, 0.5, 0.5]],
            ids=["test-1"],
        )
        print(f"  写入完成，count = {col1.count()}，数据在: {path}")
        del client1  # 释放文件句柄

        # 第二次运行（新的 client 实例）：自动加载之前的数据
        client2 = chromadb.PersistentClient(path=path)
        col2 = client2.get_or_create_collection(name="persistent_demo")
        print(f"  重新加载，count = {col2.count()}")  # 还是 1
        print(f"  加载的数据: {col2.get(ids=['test-1'])['documents']}")
        del client2  # 释放文件句柄
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    print()


if __name__ == "__main__":
    demo_first_collection()
    demo_crud()
    demo_metadata_filter()
    demo_persistent()
    print("全部演示完成！")
