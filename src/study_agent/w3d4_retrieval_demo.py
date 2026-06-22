"""
Week 3 Day 4 — 检索策略实战 Demo

运行: poetry run python src/study_agent/w3d4_retrieval_demo.py

三种检索策略的完整对比：
  1. 向量检索（Vector Search）—— 语义相似度
  2. 关键词检索（Keyword Search）—— TF-IDF 词频匹配
  3. 混合检索（Hybrid Search）—— RRF 融合两者结果

对比维度：召回率、精确率、互补性
"""

import math

import chromadb

# ══════════════════════════════════════════════════════════════════
# 第 0 部分：准备测试数据 —— 一份模拟的"公司知识库"
# ══════════════════════════════════════════════════════════════════

CORPUS = [
    # 0-3: 报销相关
    "员工报销流程：在系统中填写报销申请单，附上发票照片，提交给直属上级审批。审批通过后财务部在 5 个工作日内打款到工资卡。",
    "出差报销标准：一线城市住宿标准 500 元每晚，二线城市 350 元每晚。交通费实报实销，餐补每天 100 元。所有报销需提供正规发票。",
    "招待费报销规定：招待客户需提前申请，人均标准不超过 200 元。报销时需注明招待对象、事由和参与人员名单。",
    "报销单填写规范：发票抬头必须为公司全称，发票日期需在 3 个月以内。电子发票需打印后贴在 A4 纸上，附在报销单后面。",
    # 4-7: 请假/考勤
    "年假政策：入职满 1 年享有 5 天带薪年假，满 3 年 10 天，满 5 年 15 天。年假需提前一周向直属上级申请。",
    "病假规定：请病假需提供二级以上医院开具的病假证明。紧急情况可在 24 小时内补交证明。病假期间工资按国家规定执行。",
    "事假管理：事假每年累计不超过 10 天。超过 3 天的事假需提前 3 个工作日申请，并经部门负责人审批。",
    "考勤打卡制度：公司实行弹性工作制，核心工作时间 10:00-16:00。忘记打卡需在 24 小时内通过企业微信提交补卡申请。",
    # 8-11: IT/技术
    "办公网络配置：公司 Wi-Fi 密码为 Study@2026，支持 5G 频段。访客网络密码请向前台索取，有效期 24 小时。",
    "VPN 使用指南：远程办公需使用公司 VPN 连接内网。VPN 客户端下载地址见 IT 知识库首页，使用企业邮箱账号登录。",
    "信息安全规范：公司配发的电脑不得安装未经授权的软件。所有内部文档禁止通过个人微信、QQ 或私人邮箱外传。",
    "代码提交规范：所有代码提交前必须通过 Ruff lint 检查和 Black 格式化。PR 需至少一位同事 Code Review 通过后方可合并。",
    # 12-15: 招聘/入职
    "招聘面试流程：技术岗位面试分为技术面、算法面、HR 面三轮。非技术岗位为 HR 面和部门负责人面两轮。",
    "入职手续办理：入职第一天需携带身份证原件、学历学位证书复印件、离职证明、体检报告到 HR 部门办理入职手续。",
    "试用期管理规定：试用期为 3 个月，表现优秀者可提前转正。试用期工资为转正工资的 80%。试用期内双方均可提前 3 天通知解除劳动关系。",
    "培训与发展：新员工入职培训为期 2 天，涵盖企业文化、规章制度、信息安全等内容。每月举办一次技术分享会。",
    # 16-19: 薪酬/福利
    "薪资发放说明：每月 10 号发放上月工资，如遇节假日顺延至下一个工作日。工资条通过企业微信推送，请注意查收。",
    "五险一金缴纳：公司按照国家规定为员工缴纳五险一金，缴纳基数每年 7 月调整一次。详情可咨询 HR 薪酬专员。",
    "加班与调休：工作日加班可安排调休或领取加班费。周末加班优先安排调休。调休需在加班后一个月内使用完毕。",
    "节日福利政策：春节、中秋、端午三大节日发放节日礼品或购物卡。员工生日当月发放生日蛋糕券。",
]


# 为每条文档构造"主题向量"（8维示意，真实环境换 OpenAI Embedding）
def make_doc_vector(idx: int) -> list[float]:
    """根据文档索引生成示意向量。

    用 8 维向量编码语义方向：
      dim 0-1: 财务/报销
      dim 2-3: 人事/考勤请假
      dim 4-5: IT/技术
      dim 6-7: 其他
    """
    vec = [0.01] * 8
    if idx <= 3:  # 报销类
        vec[0] = 0.85 + idx * 0.03
        vec[1] = 0.15
    elif idx <= 7:  # 请假考勤类
        vec[2] = 0.85 + (idx - 4) * 0.03
        vec[3] = 0.12
    elif idx <= 11:  # IT/技术类
        vec[4] = 0.85 + (idx - 8) * 0.03
        vec[5] = 0.12
    elif idx <= 15:  # 招聘入职类
        vec[6] = 0.80 + (idx - 12) * 0.04
        vec[7] = 0.15
    else:  # 薪酬福利类
        vec[6] = 0.60 + (idx - 16) * 0.05
        vec[7] = 0.50 + (idx - 16) * 0.05
    # 归一化
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec]


TEST_QUERIES = [
    # (查询文字, [相关文档索引], 查询类型)
    ("出差住酒店一晚能报销多少钱？", [1], "向量+关键词都能"),
    ("我发烧了需要请病假，要什么证明？", [5], "语义为主"),
    ("公司 Wi-Fi 密码是什么？", [8], "关键词为主"),
    ("提交代码前要做什么检查？", [11], "向量+关键词都能"),
    ("入职第一天要带什么材料？", [13], "向量+关键词都能"),
    ("每个月几号发工资？", [16], "关键词为主"),
    ("年假每年有多少天？", [4], "向量+关键词都能"),
    ("试用期多久？工资怎么算？", [14], "向量+关键词都能"),
]

# ══════════════════════════════════════════════════════════════════
# 第 1 部分：策略 A — 纯向量检索（Vector Search）
# ══════════════════════════════════════════════════════════════════


class VectorRetriever:
    """基于 ChromaDB 的纯语义向量检索。

    这是 Day 2-3 你已经在用的方式——问题向量 vs 文档向量，算余弦距离。
    """

    def __init__(self, corpus: list[str]):
        self.corpus = corpus
        self.client = chromadb.EphemeralClient()
        self.collection = self.client.get_or_create_collection(
            name="w3d4_vector",
            metadata={"hnsw:space": "cosine"},
        )
        # 入库
        embeddings = [make_doc_vector(i) for i in range(len(corpus))]
        self.collection.add(
            documents=corpus,
            embeddings=embeddings,
            ids=[f"doc-{i}" for i in range(len(corpus))],
        )

    def search(self, query: str, query_vector: list[float], top_k: int = 5) -> list[dict]:
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
        )
        hits = []
        for doc_id, text, dist in zip(
            results["ids"][0], results["documents"][0], results["distances"][0]
        ):
            doc_idx = int(doc_id.split("-")[1])
            similarity = 1.0 - dist  # 余弦距离 → 余弦相似度
            hits.append(
                {
                    "doc_idx": doc_idx,
                    "text": text,
                    "score": round(similarity, 4),
                    "source": "vector",
                }
            )
        return hits


# ══════════════════════════════════════════════════════════════════
# 第 2 部分：策略 B — 关键词检索（Keyword Search）
# ══════════════════════════════════════════════════════════════════


class KeywordRetriever:
    """基于 TF-IDF 的关键词检索。

    不依赖 Embedding 模型，只看"词在文档中出现的频率"。
    这是传统搜索引擎（如 Elasticsearch）的基础算法。

    TF (Term Frequency): 词在这篇文档里出现了几次？
    IDF (Inverse Document Frequency): 这个词在整个语料库中多罕见？
      罕见词（如"Wi-Fi"）的 IDF 高 → 权重高 → 更能区分文档
      常见词（如"的"、"公司"）的 IDF 低 → 权重低 → 没啥区分度
    """

    def __init__(self, corpus: list[str]):
        self.corpus = corpus
        # 分词（简单版：按字切 2-gram，中文不需要专业分词器也能干活）
        self.doc_tokens = [self._tokenize(doc) for doc in corpus]
        self.vocab = sorted(set(t for tokens in self.doc_tokens for t in tokens))
        self.N = len(corpus)
        # 预计算 IDF
        self.idf = {}
        for word in self.vocab:
            doc_count = sum(1 for tokens in self.doc_tokens if word in tokens)
            # IDF = log(总文档数 / 包含该词的文档数) + 1，避免除零
            self.idf[word] = math.log((self.N + 1) / (doc_count + 1)) + 1.0

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """简单中文分词：2-gram 字符级切分。

        举例: "报销流程" → ["报销", "销流", "流程"]
        这不是最好的分词方式，但零依赖、直接跑、概念清楚。
        生产环境可替换为 jieba 分词或 BM25。
        """
        # 去除标点和空白
        cleaned = ""
        for ch in text:
            if ch.isalnum() or "一" <= ch <= "鿿":
                cleaned += ch
        # 1-gram + 2-gram
        tokens = []
        for i in range(len(cleaned)):
            tokens.append(cleaned[i])  # 单字
            if i < len(cleaned) - 1:
                tokens.append(cleaned[i : i + 2])  # 双字
        return tokens

    def _tfidf_vector(self, tokens: list[str]) -> dict[str, float]:
        """计算一个 token 序列的 TF-IDF 向量。"""
        total = len(tokens) if tokens else 1
        tf = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        # TF 归一化 + 乘 IDF
        vec = {}
        for word, count in tf.items():
            if word in self.idf:
                vec[word] = (count / total) * self.idf[word]
        return vec

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """用余弦相似度比较查询 TF-IDF 向量和文档 TF-IDF 向量。"""
        q_tokens = self._tokenize(query)
        q_vec = self._tfidf_vector(q_tokens)

        scores = []
        for idx, doc_tokens in enumerate(self.doc_tokens):
            d_vec = self._tfidf_vector(doc_tokens)
            # 余弦相似度
            dot = sum(q_vec.get(w, 0) * d_vec.get(w, 0) for w in set(q_vec) | set(d_vec))
            norm_q = math.sqrt(sum(v**2 for v in q_vec.values()))
            norm_d = math.sqrt(sum(v**2 for v in d_vec.values()))
            sim = dot / (norm_q * norm_d) if norm_q > 0 and norm_d > 0 else 0.0
            scores.append((idx, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        hits = []
        for idx, score in scores[:top_k]:
            if score > 0:
                hits.append(
                    {
                        "doc_idx": idx,
                        "text": self.corpus[idx],
                        "score": round(score, 4),
                        "source": "keyword",
                    }
                )
        return hits


# ══════════════════════════════════════════════════════════════════
# 第 3 部分：策略 C — 混合检索（Hybrid Search）
# ══════════════════════════════════════════════════════════════════


def reciprocal_rank_fusion(
    vector_hits: list[dict],
    keyword_hits: list[dict],
    k: int = 60,
    weight_vector: float = 0.5,
    weight_keyword: float = 0.5,
) -> list[dict]:
    """RRF (Reciprocal Rank Fusion) —— 融合两个检索器结果的最简单算法。

    原理：
      每个文档在两个检索器中各有一个排名（rank）。
      RRF 分数 = weight / (k + rank)
      k 是平滑参数（通常 60），防止 1/rank 在 rank=1 时权重过高。

    举例：
      文档 A: 向量排名 #1，关键词排名 #5
        RRF = 0.5/(60+1) + 0.5/(60+5) = 0.0082 + 0.0077 = 0.0159
      文档 B: 向量排名 #3，关键词排名 #2
        RRF = 0.5/(60+3) + 0.5/(60+2) = 0.0079 + 0.0081 = 0.0160
      → 文档 B 胜出，因为它在两个检索器中都表现稳定。

    为什么用 RRF 而不是简单加权分数？
      向量分数的范围是 [0,1]（余弦相似度）
      TF-IDF 分数的范围也是 [0,1]
      但两者不可比——向量 0.85 ≠ TF-IDF 0.85
      RRF 用排名替代分数，绕过"不同尺度不可比"的问题。
    """
    # 将两个列表统一为 {doc_idx: merged_info}
    merged: dict[int, dict] = {}

    for rank, hit in enumerate(vector_hits, start=1):
        idx = hit["doc_idx"]
        merged[idx] = {
            "doc_idx": idx,
            "text": hit["text"],
            "score_vector": hit["score"],
            "rank_vector": rank,
            "score_keyword": 0.0,
            "rank_keyword": 999,
            "rrf": 0.0,
        }

    for rank, hit in enumerate(keyword_hits, start=1):
        idx = hit["doc_idx"]
        if idx in merged:
            merged[idx]["score_keyword"] = hit["score"]
            merged[idx]["rank_keyword"] = rank
        else:
            merged[idx] = {
                "doc_idx": idx,
                "text": hit["text"],
                "score_vector": 0.0,
                "rank_vector": 999,
                "score_keyword": hit["score"],
                "rank_keyword": rank,
                "rrf": 0.0,
            }

    # 计算 RRF
    for info in merged.values():
        info["rrf"] = weight_vector / (k + info["rank_vector"]) + weight_keyword / (
            k + info["rank_keyword"]
        )

    # 按 RRF 降序
    result = sorted(merged.values(), key=lambda x: x["rrf"], reverse=True)
    for hit in result:
        hit["score"] = round(hit["rrf"], 6)
        hit["source"] = "hybrid"
    return result


class HybridRetriever:
    """混合检索器：向量检索 + 关键词检索 → RRF 融合。"""

    def __init__(self, corpus: list[str]):
        self.vector_retriever = VectorRetriever(corpus)
        self.keyword_retriever = KeywordRetriever(corpus)

    def search(
        self,
        query: str,
        query_vector: list[float],
        top_k: int = 5,
        weight_vector: float = 0.5,
        weight_keyword: float = 0.5,
    ) -> list[dict]:
        # 从两个检索器各拿 top_k * 2（多拿一些，给融合留空间）
        vec_hits = self.vector_retriever.search(query, query_vector, top_k=top_k * 2)
        kw_hits = self.keyword_retriever.search(query, top_k=top_k * 2)

        fused = reciprocal_rank_fusion(
            vec_hits,
            kw_hits,
            weight_vector=weight_vector,
            weight_keyword=weight_keyword,
        )
        return fused[:top_k]


# ══════════════════════════════════════════════════════════════════
# 第 4 部分：对比评测框架
# ══════════════════════════════════════════════════════════════════


def evaluate_retrievers():
    """对三种检索器在 8 条测试查询上做对比评测。"""
    print("初始化检索器...")
    vector_ret = VectorRetriever(CORPUS)
    keyword_ret = KeywordRetriever(CORPUS)
    hybrid_ret = HybridRetriever(CORPUS)

    # 为每条查询生成向量
    def query_vector_for(idx: int) -> list[float]:
        """简单映射：根据查询大致主题生成向量。"""
        topic_vecs = [
            make_doc_vector(1),  # Q0: 报销类
            make_doc_vector(5),  # Q1: 病假类
            make_doc_vector(8),  # Q2: Wi-Fi/IT类
            make_doc_vector(11),  # Q3: 代码规范类
            make_doc_vector(13),  # Q4: 入职类
            make_doc_vector(16),  # Q5: 薪资类
            make_doc_vector(4),  # Q6: 年假类
            make_doc_vector(14),  # Q7: 试用期类
        ]
        return topic_vecs[idx]

    # ── 评测指标 ──
    # Recall@K: 在 Top-K 结果中，相关文档被找到了几个？
    # Precision@K: Top-K 结果中，有几个是真正相关的？
    def recall_at_k(hits: list[dict], relevant: list[int], k: int) -> float:
        if not relevant:
            return 1.0
        found = set(h["doc_idx"] for h in hits[:k]) & set(relevant)
        return len(found) / len(relevant)

    def precision_at_k(hits: list[dict], relevant: list[int], k: int) -> float:
        if not hits[:k]:
            return 0.0
        found = sum(1 for h in hits[:k] if h["doc_idx"] in relevant)
        return found / min(k, len(hits))

    print(f"\n{'='*70}")
    print("  三种检索策略对比评测")
    print(f"{'='*70}")

    # 汇总统计
    stats = {
        "vector": {"recall@3": [], "recall@5": [], "prec@3": [], "prec@5": []},
        "keyword": {"recall@3": [], "recall@5": [], "prec@3": [], "prec@5": []},
        "hybrid": {"recall@3": [], "recall@5": [], "prec@5": [], "prec@3": []},
    }

    for qi, (query, relevant, qtype) in enumerate(TEST_QUERIES):
        q_vec = query_vector_for(qi)
        v_hits = vector_ret.search(query, q_vec, top_k=5)
        k_hits = keyword_ret.search(query, top_k=5)
        h_hits = hybrid_ret.search(query, q_vec, top_k=5)

        print(f"\n  ┌─ Q{qi+1}: {query}")
        print(f"  │  期望命中: doc-{relevant}")

        # 打印每种策略的 Top-3
        for name, hits, emoji in [
            ("向量检索", v_hits, "V"),
            ("关键词", k_hits, "K"),
            ("混合检索", h_hits, "H"),
        ]:
            top_ids = [f"doc-{h['doc_idx']}({h['score']:.3f})" for h in hits[:3]]
            hit_rel = "[HIT]" if any(h["doc_idx"] in relevant for h in hits[:1]) else "[MISS]"
            print(f"  │ [{emoji}] {hit_rel} Top-3: {', '.join(top_ids)}")

            # 统计
            r3 = recall_at_k(hits, relevant, 3)
            r5 = recall_at_k(hits, relevant, 5)
            p3 = precision_at_k(hits, relevant, 3)
            p5 = precision_at_k(hits, relevant, 5)
            stats[{"向量检索": "vector", "关键词": "keyword", "混合检索": "hybrid"}[name]][
                "recall@3"
            ].append(r3)
            stats[{"向量检索": "vector", "关键词": "keyword", "混合检索": "hybrid"}[name]][
                "recall@5"
            ].append(r5)
            stats[{"向量检索": "vector", "关键词": "keyword", "混合检索": "hybrid"}[name]][
                "prec@3"
            ].append(p3)
            stats[{"向量检索": "vector", "关键词": "keyword", "混合检索": "hybrid"}[name]][
                "prec@5"
            ].append(p5)

    # ── 汇总表 ──
    print(f"\n\n{'='*70}")
    print("  汇总对比")
    print(f"{'='*70}")

    def avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    print(f"\n  {'策略':<12} {'Recall@3':>10} {'Recall@5':>10} {'Prec@3':>10} {'Prec@5':>10}")
    print(f"  {'-'*52}")
    for name, key in [("向量检索", "vector"), ("关键词检索", "keyword"), ("混合检索", "hybrid")]:
        s = stats[key]
        print(
            f"  {name:<12} {avg(s['recall@3']):>10.1%} {avg(s['recall@5']):>10.1%} {avg(s['prec@3']):>10.1%} {avg(s['prec@5']):>10.1%}"
        )

    print("\n  ┌─ 指标解释 ─────────────────────────────────┐")
    print("  │ Recall@K = 相关文档被找到的比例              │")
    print("  │ Precision@K = Top-K 中真正相关的比例         │")
    print("  │ 召回率高 = 不漏。精确率高 = 不杂。          │")
    print("  └────────────────────────────────────────────┘")


# ══════════════════════════════════════════════════════════════════
# 第 5 部分：深入理解 —— 探究向量和关键词的"互补性"
# ══════════════════════════════════════════════════════════════════


def demonstrate_complementarity():
    """用一个具体例子展示向量检索和关键词检索各有什么擅长。

    核心洞察：
      向量检索擅长"语义相近"——同义词、改写、跨语言
      关键词检索擅长"精确匹配"——专有名词、数字、代码
    """
    print(f"\n\n{'='*70}")
    print("  深入理解：向量 vs 关键词 的互补关系")
    print(f"{'='*70}")

    vector_ret = VectorRetriever(CORPUS)
    keyword_ret = KeywordRetriever(CORPUS)

    # 案例 1：语义改写 → 向量胜
    print("\n  【案例 1：语义改写 —— 向量擅长的】")
    query = "我需要去医院看病，公司有什么规定？"
    q_vec = make_doc_vector(5)  # 病假类
    print(f"  查询: {query}")
    print("  说明: 查询中没有'病假'这个词，但语义指向病假规定")

    v_hits = vector_ret.search(query, q_vec, top_k=3)
    k_hits = keyword_ret.search(query, top_k=3)

    print("  向量检索 Top-3:")
    for h in v_hits:
        print(f"    doc-{h['doc_idx']} [{h['score']:.4f}] {h['text'][:60]}...")
    print("  关键词检索 Top-3:")
    for h in k_hits:
        print(f"    doc-{h['doc_idx']} [{h['score']:.4f}] {h['text'][:60]}...")

    # 案例 2：精确匹配 → 关键词胜
    print("\n  【案例 2：精确匹配 —— 关键词擅长的】")
    query = "Wi-Fi 密码是 Study@2026 吗？"
    q_vec = make_doc_vector(8)  # IT类
    print(f"  查询: {query}")
    print("  说明: 查询中包含专有名词'Wi-Fi'和'Study@2026'")

    v_hits = vector_ret.search(query, q_vec, top_k=3)
    k_hits = keyword_ret.search(query, top_k=3)

    print("  向量检索 Top-3:")
    for h in v_hits:
        print(f"    doc-{h['doc_idx']} [{h['score']:.4f}] {h['text'][:60]}...")
    print("  关键词检索 Top-3:")
    for h in k_hits:
        print(f"    doc-{h['doc_idx']} [{h['score']:.4f}] {h['text'][:60]}...")


# ══════════════════════════════════════════════════════════════════
# 第 6 部分：RRF 权重调参实验
# ══════════════════════════════════════════════════════════════════


def rrf_weight_experiment():
    """探究 weight_vector / weight_keyword 的不同配比对结果的影响。"""
    print(f"\n\n{'='*70}")
    print("  RRF 权重调参实验")
    print(f"{'='*70}")

    hybrid_ret = HybridRetriever(CORPUS)

    # 用一条"半语义半关键词"的查询
    query = "提交代码前要做什么检查？"
    q_vec = make_doc_vector(11)  # 代码规范类
    relevant = [11]

    print(f"\n  查询: {query}")
    print(f"  期望命中: doc-{relevant}")
    print(f"\n  {'权重配比':<20} {'Top-3 结果':<50} {'命中?':<8}")
    print(f"  {'-'*78}")

    weight_pairs = [
        ("纯向量 (1.0/0.0)", 1.0, 0.0),
        ("偏向量 (0.7/0.3)", 0.7, 0.3),
        ("均衡 (0.5/0.5)", 0.5, 0.5),
        ("偏关键词 (0.3/0.7)", 0.3, 0.7),
        ("纯关键词 (0.0/1.0)", 0.0, 1.0),
    ]

    for label, wv, wk in weight_pairs:
        hits = hybrid_ret.search(query, q_vec, top_k=3, weight_vector=wv, weight_keyword=wk)
        ids = [f"doc-{h['doc_idx']}" for h in hits]
        hit_rel = "[HIT]" if any(h["doc_idx"] in relevant for h in hits[:1]) else "[MISS]"
        print(f"  {label:<20} {', '.join(ids):<50} {hit_rel:<8}")

    print("\n  结论: RRF 权重决定了你更相信语义还是更相信关键词。")
    print("  默认均衡(0.5/0.5)在大多数场景够用，特殊领域可偏置。")


# ══════════════════════════════════════════════════════════════════
# 第 7 部分：策略选型决策树
# ══════════════════════════════════════════════════════════════════

RETRIEVAL_GUIDE = r"""
  检索策略选型决策树：

  你的文档类型？
  ├─ 以自然语言为主（文章、报告、手册）
  │   ├─ 知识库不大 (<1万条) → 纯向量检索即可
  │   └─ 知识库较大 + 需要精确过滤 → 混合检索
  │
  ├─ 包含大量专有名词/编号/代码（API文档、法律条文）
  │   └─ 混合检索（偏关键词 weight=0.7）
  │
  ├─ 用户查询以"搜索式"为主而非"对话式"
  │   └─ 关键词检索为主 + 向量辅助
  │
  └─ FAQ / 客服场景（短问答对）
      └─ 混合检索，两个都要
"""


# ══════════════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════════════


def main():
    print("╔══════════════════════════════════════════════════════╗")
    print("║  Week 3 Day 4 — 检索策略实战                          ║")
    print("╚══════════════════════════════════════════════════════╝")

    # Part 1: 三种策略的完整对比评测
    evaluate_retrievers()

    # Part 2: 向量 vs 关键词的深层对比
    demonstrate_complementarity()

    # Part 3: RRF 权重调参
    rrf_weight_experiment()

    # Part 4: 策略选型指南
    print(RETRIEVAL_GUIDE)

    print("=" * 70)
    print("  Day 4 核心收获:")
    print("    1. 向量检索 = 语义相似，适合同义词/改写/跨语言")
    print("    2. 关键词检索 = 精确匹配，适合专有名词/数字/代码")
    print("    3. 混合检索 = RRF 融合两者，互补短板")
    print("    4. RRF 权重(0.5/0.5)是通用起点，可根据场景偏置")
    print("=" * 70)


if __name__ == "__main__":
    main()
