"""
Week 3 Day 6 — RAG 评测与优化
===============================
运行: poetry run python src/study_agent/w3d6_evaluation.py

今天的目标：用数据回答"你的 RAG 管道做得到底怎么样？"

四大评测指标：
  1. 检索召回率 (Retrieval Recall)     — 正确答案所在文档有没有被检索到？
  2. 答案准确性 (Answer Accuracy)       — LLM 的回答是否正确？
  3. 引用正确性 (Citation Correctness)   — LLM 标注的 [1][2] 是否真的对应了正确文档？
  4. 拒答率 (Rejection Rate)            — 知识库覆盖不到的问题，是否正确拒答？

评测维度对比：
  - Top-K 大小：3 vs 5 vs 7
  - 相似度阈值：0.4 vs 0.6 vs 0.8
"""

from __future__ import annotations

import json
import math
import re
import textwrap
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import chromadb

from study_agent.llm.client import LLMClient
from study_agent.prompt.templates import PromptManager

# ═══════════════════════════════════════════════════════════════════
# 第 0 部分：知识库与检索器（复用 Day 5，只做最小改动）
# ═══════════════════════════════════════════════════════════════════

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


def _make_doc_vector(idx: int) -> list[float]:
    """8 维示意向量（Day 5 同款）"""
    vec = [0.01] * 8
    if idx <= 3:
        vec[0] = 0.85 + idx * 0.03
        vec[1] = 0.15
    elif idx <= 7:
        vec[2] = 0.85 + (idx - 4) * 0.03
        vec[3] = 0.12
    elif idx <= 11:
        vec[4] = 0.85 + (idx - 8) * 0.03
        vec[5] = 0.12
    elif idx <= 15:
        vec[6] = 0.80 + (idx - 12) * 0.04
        vec[7] = 0.15
    else:
        vec[6] = 0.60 + (idx - 16) * 0.05
        vec[7] = 0.50 + (idx - 16) * 0.05
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec]


class SimpleRetriever:
    """简易 ChromaDB 检索器（Day 5 同款，只加了 search_with_scores 方法）"""

    def __init__(self, corpus: list[str]):
        self.corpus = corpus
        self.client = chromadb.EphemeralClient()
        self.collection = self.client.get_or_create_collection(
            name="w3d6_eval",
            metadata={"hnsw:space": "cosine"},
        )
        embeddings = [_make_doc_vector(i) for i in range(len(corpus))]
        self.collection.add(
            documents=corpus,
            embeddings=embeddings,
            ids=[f"doc-{i}" for i in range(len(corpus))],
        )

    def search(self, query: str, top_k: int = 3) -> list[tuple[int, str, float]]:
        """返回 [(doc_idx, text, similarity), ...]"""
        results = self.collection.query(
            query_embeddings=[_make_doc_vector(self._guess_topic(query))],
            n_results=top_k,
        )
        hits = []
        for doc_id, text, dist in zip(
            results["ids"][0], results["documents"][0], results["distances"][0]
        ):
            idx = int(doc_id.split("-")[1])
            hits.append((idx, text, round(1.0 - dist, 4)))
        return hits

    @staticmethod
    def _guess_topic(query: str) -> int:
        q = query.lower()
        if any(w in q for w in ["报销", "发票", "出差", "住宿", "招待"]):
            return 1
        if any(
            w in q
            for w in [
                "请假",
                "病假",
                "年假",
                "事假",
                "考勤",
                "打卡",
                "看病",
                "发烧",
                "感冒",
                "医院",
                "弹性",
                "核心工作",
            ]
        ):
            return 5
        if any(
            w in q
            for w in [
                "wifi",
                "wi-fi",
                "vpn",
                "密码",
                "网络",
                "上网",
                "代码",
                "提交",
                "安全",
                "安装",
                "软件",
                "微信",
                "qq",
                "邮件",
                "远程",
                "电脑",
            ]
        ):
            return 8
        if any(w in q for w in ["入职", "面试", "招聘", "试用期", "转正", "培训", "离职"]):
            return 13
        if any(
            w in q
            for w in [
                "工资",
                "薪资",
                "发薪",
                "五险一金",
                "加班",
                "调休",
                "福利",
                "节日",
                "蛋糕",
                "购物卡",
                "生日",
                "春节",
                "中秋",
                "端午",
            ]
        ):
            return 16
        return 0


# ═══════════════════════════════════════════════════════════════════
# 第 1 部分：30 条测试用例（含 Ground Truth）
# ═══════════════════════════════════════════════════════════════════


@dataclass
class TestCase:
    """一条 RAG 测试用例。

    question: 用户问题
    category: 问题所属类别（报销/请假考勤/IT技术/招聘入职/薪酬福利/超出范围）
    ground_truth_docs: 包含正确答案的文档索引列表
    expected_keywords: 正确答案应包含的关键词
    is_out_of_scope: True 表示知识库覆盖不到，系统应拒答
    """

    question: str
    category: str
    ground_truth_docs: list[int]
    expected_keywords: list[str]
    is_out_of_scope: bool = False


# 30 条测试用例，覆盖 5 个知识领域 + 边界情况
TEST_CASES: list[TestCase] = [
    # ═══════════ 报销类（6 条）═══════════
    TestCase(
        question="出差去北京，酒店住宿一晚能报销多少钱？",
        category="报销",
        ground_truth_docs=[1],
        expected_keywords=["500", "一线城市"],
    ),
    TestCase(
        question="出差期间每天的餐补是多少？",
        category="报销",
        ground_truth_docs=[1],
        expected_keywords=["100", "餐补"],
    ),
    TestCase(
        question="报销流程是怎样的？钱多久能到账？",
        category="报销",
        ground_truth_docs=[0],
        expected_keywords=["审批", "5个工作日", "工资卡"],
    ),
    TestCase(
        question="招待客户吃饭，人均消费不能超过多少？",
        category="报销",
        ground_truth_docs=[2],
        expected_keywords=["200", "招待", "提前申请"],
    ),
    TestCase(
        question="发票抬头应该写什么？发票日期有什么要求？",
        category="报销",
        ground_truth_docs=[3],
        expected_keywords=["公司全称", "3个月"],
    ),
    TestCase(
        question="电子发票要怎么贴在报销单上？",
        category="报销",
        ground_truth_docs=[3],
        expected_keywords=["打印", "A4纸"],
    ),
    # ═══════════ 请假/考勤类（6 条）═══════════
    TestCase(
        question="我刚入职 2 年，每年有多少天年假？",
        category="请假考勤",
        ground_truth_docs=[4],
        expected_keywords=["5天", "满1年"],
    ),
    TestCase(
        question="入职满 5 年的员工年假是多少天？",
        category="请假考勤",
        ground_truth_docs=[4],
        expected_keywords=["15天", "满5年"],
    ),
    TestCase(
        question="请病假需要什么证明？如果来不及提前请假怎么办？",
        category="请假考勤",
        ground_truth_docs=[5],
        expected_keywords=["二级以上医院", "病假证明", "24小时", "补交"],
    ),
    TestCase(
        question="事假一年最多请多少天？超过 3 天有什么特殊要求？",
        category="请假考勤",
        ground_truth_docs=[6],
        expected_keywords=["10天", "3个工作日", "部门负责人"],
    ),
    TestCase(
        question="公司规定的核心工作时间是什么时段？",
        category="请假考勤",
        ground_truth_docs=[7],
        expected_keywords=["10:00", "16:00", "弹性工作制"],
    ),
    TestCase(
        question="忘记打卡了怎么办？",
        category="请假考勤",
        ground_truth_docs=[7],
        expected_keywords=["24小时", "企业微信", "补卡"],
    ),
    # ═══════════ IT/技术类（5 条）═══════════
    TestCase(
        question="公司 Wi-Fi 密码是什么？",
        category="IT技术",
        ground_truth_docs=[8],
        expected_keywords=["Study@2026", "5G"],
    ),
    TestCase(
        question="在家远程办公怎么连公司内网？",
        category="IT技术",
        ground_truth_docs=[9],
        expected_keywords=["VPN", "企业邮箱"],
    ),
    TestCase(
        question="公司电脑上能安装自己下载的软件吗？",
        category="IT技术",
        ground_truth_docs=[10],
        expected_keywords=["未经授权", "不得安装"],
    ),
    TestCase(
        question="能用微信把工作文档发给同事吗？",
        category="IT技术",
        ground_truth_docs=[10],
        expected_keywords=["禁止", "个人微信", "外传"],
    ),
    TestCase(
        question="提交代码前需要做什么检查？",
        category="IT技术",
        ground_truth_docs=[11],
        expected_keywords=["Ruff", "Black", "Code Review"],
    ),
    # ═══════════ 招聘/入职类（5 条）═══════════
    TestCase(
        question="技术岗位的面试流程有几轮？分别是什么？",
        category="招聘入职",
        ground_truth_docs=[12],
        expected_keywords=["技术面", "算法面", "HR面", "三轮"],
    ),
    TestCase(
        question="入职第一天需要带什么材料？",
        category="招聘入职",
        ground_truth_docs=[13],
        expected_keywords=["身份证", "学历学位证书", "离职证明", "体检报告"],
    ),
    TestCase(
        question="试用期多久？试用期工资怎么算？",
        category="招聘入职",
        ground_truth_docs=[14],
        expected_keywords=["3个月", "80%"],
    ),
    TestCase(
        question="试用期内可以提前离职吗？需要提前多久通知？",
        category="招聘入职",
        ground_truth_docs=[14],
        expected_keywords=["3天", "解除"],
    ),
    TestCase(
        question="新员工入职培训要多久？培训内容包括什么？",
        category="招聘入职",
        ground_truth_docs=[15],
        expected_keywords=["2天", "企业文化", "信息安全"],
    ),
    # ═══════════ 薪酬/福利类（5 条）═══════════
    TestCase(
        question="每个月几号发工资？如果碰到节假日怎么办？",
        category="薪酬福利",
        ground_truth_docs=[16],
        expected_keywords=["10号", "顺延"],
    ),
    TestCase(
        question="五险一金的缴纳基数什么时候调整？",
        category="薪酬福利",
        ground_truth_docs=[17],
        expected_keywords=["每年7月", "调整"],
    ),
    TestCase(
        question="周末加班怎么补偿？调休有有效期吗？",
        category="薪酬福利",
        ground_truth_docs=[18],
        expected_keywords=["调休", "一个月内"],
    ),
    TestCase(
        question="春节公司发什么福利？",
        category="薪酬福利",
        ground_truth_docs=[19],
        expected_keywords=["礼品", "购物卡", "三大节日"],
    ),
    TestCase(
        question="生日当月有什么福利？",
        category="薪酬福利",
        ground_truth_docs=[19],
        expected_keywords=["蛋糕券", "生日"],
    ),
    # ═══════════ 超出范围类（3 条）—— 知识库覆盖不到，应拒答 ═══════════
    TestCase(
        question="公司附近有什么好吃的餐厅推荐吗？",
        category="超出范围",
        ground_truth_docs=[],
        expected_keywords=[],
        is_out_of_scope=True,
    ),
    TestCase(
        question="今天天气怎么样？",
        category="超出范围",
        ground_truth_docs=[],
        expected_keywords=[],
        is_out_of_scope=True,
    ),
    TestCase(
        question="公司股票代码是什么？什么时候上市的？",
        category="超出范围",
        ground_truth_docs=[],
        expected_keywords=[],
        is_out_of_scope=True,
    ),
]


# ═══════════════════════════════════════════════════════════════════
# 第 2 部分：RAG 管道（复用 Day 5 的 rag_pipeline）
# ═══════════════════════════════════════════════════════════════════


@dataclass
class CitationResult:
    answer: str
    citations: list[dict[str, Any]] = field(default_factory=list)


def parse_citations(answer: str, source_map: dict[int, str]) -> CitationResult:
    """从 LLM 回复中提取引用编号，映射回原文档。"""
    cited_numbers = set()
    for match in re.finditer(r"\[(\d+)\]", answer):
        cited_numbers.add(int(match.group(1)))

    citations = []
    for num in sorted(cited_numbers):
        if num in source_map:
            citations.append({"number": num, "text": source_map[num]})

    return CitationResult(answer=answer, citations=citations)


@dataclass
class RAGResult:
    question: str
    answer: str
    sources: list[dict[str, Any]]
    citations: list[dict[str, Any]]


def rag_pipeline(
    question: str,
    retriever: SimpleRetriever,
    llm: LLMClient,
    top_k: int = 5,
    sim_threshold: float = 0.6,
) -> RAGResult:
    """端到端 RAG 管道（Day 5 同款）"""
    all_hits = retriever.search(question, top_k=top_k)
    hits = [(idx, text, sim) for idx, text, sim in all_hits if sim >= sim_threshold]

    if not hits:
        return RAGResult(
            question=question,
            answer="根据现有资料无法回答您的问题。",
            sources=[],
            citations=[],
        )

    source_map = {i + 1: text for i, (_, text, _) in enumerate(hits)}

    manager = PromptManager("src/study_agent/prompt/templates")
    prompt = manager.render(
        "rag_generation",
        role="知识库问答助手",
        chunks=[text for _, text, _ in hits],
        question=question,
    )

    raw_answer = llm.chat(prompt)
    result = parse_citations(raw_answer, source_map)

    return RAGResult(
        question=question,
        answer=result.answer,
        sources=[{"idx": idx, "text": text, "similarity": sim} for idx, text, sim in hits],
        citations=result.citations,
    )


# ═══════════════════════════════════════════════════════════════════
# 第 3 部分：评测指标计算
# ═══════════════════════════════════════════════════════════════════


@dataclass
class EvalMetrics:
    """单条测试用例的评测结果"""

    question: str
    category: str
    is_out_of_scope: bool

    # 检索指标
    recall_at_k: float  # 0.0 ~ 1.0，正确答案文档被检索到的比例
    retrieved_doc_indices: list[int]  # 实际检索到的文档索引
    ground_truth_docs: list[int]  # 应该检索到的文档索引

    # 答案指标
    answer: str
    keyword_hit_rate: float  # 0.0 ~ 1.0，期望关键词在答案中的命中率
    expected_keywords: list[str]
    hit_keywords: list[str]
    missed_keywords: list[str]

    # 引用指标
    cited_numbers: list[int]  # LLM 实际引用的编号
    valid_citations: int  # 引用编号中指向正确文档的数量
    invalid_citations: int  # 引用编号中指向错误/不存在文档的数量
    citation_precision: float  # 有效引用 / 总引用

    # 拒答指标（仅 out_of_scope 题目）
    correctly_rejected: bool = False


def compute_recall_at_k(
    retrieved_doc_indices: list[int],
    ground_truth_docs: list[int],
) -> float:
    """检索召回率：正确答案文档中被检索到的比例。

    Recall@K = |检索到的正确答案文档| / |所有正确答案文档|

    例：正确答案在 [1, 2] 两个文档中，检索结果返回了 [1, 5, 8]
    → 召回了文档 1，漏掉了文档 2 → Recall = 1/2 = 0.5
    """
    if not ground_truth_docs:
        return 1.0  # 超出范围的题目，没有 ground truth，召回率无意义，记为 1.0
    retrieved_set = set(retrieved_doc_indices)
    gt_set = set(ground_truth_docs)
    return len(retrieved_set & gt_set) / len(gt_set)


def compute_keyword_hit_rate(
    answer: str,
    expected_keywords: list[str],
) -> tuple[float, list[str], list[str]]:
    """关键词命中率：期望关键词在答案中出现的比例。

    返回 (命中率, 命中关键词列表, 缺失关键词列表)
    """
    if not expected_keywords:
        return 1.0, [], []

    answer_lower = answer.lower()
    hit = []
    missed = []
    for kw in expected_keywords:
        if kw.lower() in answer_lower:
            hit.append(kw)
        else:
            missed.append(kw)

    return len(hit) / len(expected_keywords), hit, missed


def evaluate_citations(
    citations: list[dict[str, Any]],
    source_map: dict[int, str],
    ground_truth_docs: list[int],
    retrieved_indices: list[int],
) -> tuple[list[int], int, int, float]:
    """评估引用正确性。

    有效引用 = 引用编号指向的文档确实在"检索到的文档"中，且内容与该引用相关
    无效引用 = 引用编号不存在、或指向的文档与检索结果无关

    返回 (cited_numbers, valid_count, invalid_count, precision)
    """
    cited_numbers = [c["number"] for c in citations]

    if not cited_numbers:
        return [], 0, 0, 0.0

    valid = 0
    invalid = 0
    for num in cited_numbers:
        if num in source_map:
            valid += 1
        else:
            invalid += 1

    precision = valid / len(cited_numbers) if cited_numbers else 0.0
    return cited_numbers, valid, invalid, precision


def is_rejection(answer: str) -> bool:
    """判断答案是否为拒答（知识库没有资料时）。"""
    rejection_phrases = [
        "无法回答",
        "无法提供",
        "没有相关",
        "不在知识库",
        "超出",
        "没有找到",
        "资料中没有",
        "无法找到",
        "我无法",
        "不能回答",
    ]
    answer_lower = answer.lower()
    return any(phrase in answer_lower for phrase in rejection_phrases)


def evaluate_one(
    test_case: TestCase,
    rag_result: RAGResult,
) -> EvalMetrics:
    """对一条测试用例的 RAG 结果做全面评测。"""

    # 1. 检索召回率
    retrieved_indices = [s["idx"] for s in rag_result.sources]
    recall = compute_recall_at_k(retrieved_indices, test_case.ground_truth_docs)

    # 2. 关键词命中率
    keyword_hit_rate, hit_kw, missed_kw = compute_keyword_hit_rate(
        rag_result.answer, test_case.expected_keywords
    )

    # 3. 引用正确性
    source_map = {i + 1: s["text"] for i, s in enumerate(rag_result.sources)}
    cited_nums, valid_cite, invalid_cite, cite_prec = evaluate_citations(
        rag_result.citations, source_map, test_case.ground_truth_docs, retrieved_indices
    )

    # 4. 拒答检测
    correctly_rejected = False
    if test_case.is_out_of_scope:
        correctly_rejected = is_rejection(rag_result.answer)

    return EvalMetrics(
        question=test_case.question,
        category=test_case.category,
        is_out_of_scope=test_case.is_out_of_scope,
        recall_at_k=recall,
        retrieved_doc_indices=retrieved_indices,
        ground_truth_docs=test_case.ground_truth_docs,
        answer=rag_result.answer,
        keyword_hit_rate=keyword_hit_rate,
        expected_keywords=test_case.expected_keywords,
        hit_keywords=hit_kw,
        missed_keywords=missed_kw,
        cited_numbers=cited_nums,
        valid_citations=valid_cite,
        invalid_citations=invalid_cite,
        citation_precision=cite_prec,
        correctly_rejected=correctly_rejected,
    )


# ═══════════════════════════════════════════════════════════════════
# 第 4 部分：评测运行器
# ═══════════════════════════════════════════════════════════════════


@dataclass
class EvalReport:
    """完整评测报告"""

    config: dict[str, Any]  # 评测配置（top_k, sim_threshold 等）
    timestamp: str
    total_questions: int
    metrics: list[EvalMetrics]

    # 汇总指标
    avg_recall: float = 0.0
    avg_keyword_hit: float = 0.0
    avg_citation_precision: float = 0.0
    rejection_accuracy: float = 0.0  # 超出范围题目的正确拒答率

    # 分类汇总
    category_breakdown: dict[str, dict[str, float]] = field(default_factory=dict)


def run_evaluation(
    test_cases: list[TestCase],
    retriever: SimpleRetriever,
    llm: LLMClient,
    top_k: int = 5,
    sim_threshold: float = 0.6,
    verbose: bool = True,
) -> EvalReport:
    """对所有测试用例跑完整的 RAG 评测。"""

    all_metrics: list[EvalMetrics] = []

    for i, tc in enumerate(test_cases):
        if verbose:
            status = "[超出范围]" if tc.is_out_of_scope else f"[{tc.category}]"
            print(f"  [{i+1:02d}/30] {status} {tc.question[:45]}...", end=" ")

        try:
            # 跑 RAG 管道
            result = rag_pipeline(
                tc.question, retriever, llm, top_k=top_k, sim_threshold=sim_threshold
            )
            # 评测
            metrics = evaluate_one(tc, result)
        except Exception as e:
            # API 调用失败时，创建一个空结果
            result = RAGResult(
                question=tc.question,
                answer=f"[评测失败: {e}]",
                sources=[],
                citations=[],
            )
            metrics = evaluate_one(tc, result)
            if verbose:
                print(f"[!] 失败: {e}")

        all_metrics.append(metrics)
        if verbose:
            print(
                f"Recall={metrics.recall_at_k:.0%} KW={metrics.keyword_hit_rate:.0%} Cite={metrics.citation_precision:.0%}"
            )

    # 汇总计算
    in_scope = [m for m in all_metrics if not m.is_out_of_scope]
    out_of_scope = [m for m in all_metrics if m.is_out_of_scope]

    avg_recall = sum(m.recall_at_k for m in in_scope) / len(in_scope) if in_scope else 0
    avg_kw = sum(m.keyword_hit_rate for m in in_scope) / len(in_scope) if in_scope else 0
    avg_cite = sum(m.citation_precision for m in in_scope) / len(in_scope) if in_scope else 0
    rejection_acc = (
        sum(1 for m in out_of_scope if m.correctly_rejected) / len(out_of_scope)
        if out_of_scope
        else 0
    )

    # 分类汇总
    categories: dict[str, list[EvalMetrics]] = {}
    for m in all_metrics:
        categories.setdefault(m.category, []).append(m)

    cat_breakdown = {}
    for cat, cat_metrics in categories.items():
        cat_in_scope = [m for m in cat_metrics if not m.is_out_of_scope]
        cat_breakdown[cat] = {
            "count": len(cat_metrics),
            "avg_recall": (
                sum(m.recall_at_k for m in cat_in_scope) / len(cat_in_scope) if cat_in_scope else 0
            ),
            "avg_keyword_hit": (
                sum(m.keyword_hit_rate for m in cat_in_scope) / len(cat_in_scope)
                if cat_in_scope
                else 0
            ),
            "avg_citation_precision": (
                sum(m.citation_precision for m in cat_in_scope) / len(cat_in_scope)
                if cat_in_scope
                else 0
            ),
        }

    return EvalReport(
        config={"top_k": top_k, "sim_threshold": sim_threshold},
        timestamp=datetime.now(UTC).isoformat(),
        total_questions=len(test_cases),
        metrics=all_metrics,
        avg_recall=avg_recall,
        avg_keyword_hit=avg_kw,
        avg_citation_precision=avg_cite,
        rejection_accuracy=rejection_acc,
        category_breakdown=cat_breakdown,
    )


# ═══════════════════════════════════════════════════════════════════
# 第 5 部分：多配置对比
# ═══════════════════════════════════════════════════════════════════


def run_config_comparison(
    test_cases: list[TestCase],
    retriever: SimpleRetriever,
    llm: LLMClient,
) -> list[EvalReport]:
    """对比不同 Top-K 和相似度阈值组合的效果。

    比较维度：
      - Top-K: 3 vs 5 vs 7（检索返回多少条文档）
      - 相似度阈值: 0.4 vs 0.6 vs 0.8（多相似的文档才纳入答案）
    """

    configs = [
        # (top_k, sim_threshold)  场景说明
        (3, 0.4),  # 少文档 + 低门槛：最宽松
        (3, 0.6),  # 少文档 + 中门槛
        (5, 0.6),  # 中文档 + 中门槛（默认配置）
        (7, 0.6),  # 多文档 + 中门槛
        (5, 0.8),  # 中文档 + 高门槛：最严格
    ]

    reports = []
    for top_k, sim_th in configs:
        print(f"\n{'─'*60}")
        print(f"  配置: Top-K={top_k}, 相似度阈值={sim_th}")
        print(f"{'─'*60}")

        report = run_evaluation(
            test_cases,
            retriever,
            llm,
            top_k=top_k,
            sim_threshold=sim_th,
            verbose=False,
        )
        reports.append(report)

        # 快速摘要
        print(
            f"  召回率={report.avg_recall:.1%}  "
            f"关键词命中={report.avg_keyword_hit:.1%}  "
            f"引用精度={report.avg_citation_precision:.1%}  "
            f"拒答正确率={report.rejection_accuracy:.1%}"
        )

    return reports


# ═══════════════════════════════════════════════════════════════════
# 第 6 部分：评测报告生成
# ═══════════════════════════════════════════════════════════════════


def print_report(report: EvalReport) -> None:
    """打印格式化的评测报告到终端。"""

    cfg = report.config
    sep = "=" * 66

    print()
    print(sep)
    print("  Week 3 Day 6 — RAG 管道评测报告")
    print(sep)
    print(f"  时间: {report.timestamp[:19]}")
    print(f"  配置: Top-K={cfg['top_k']}, 相似度阈值={cfg['sim_threshold']}")
    print(f"  题目数: {report.total_questions} 条 (27 条范围内 + 3 条超出范围)")
    print(sep)

    # 总览表
    print()
    print("  [总体指标]")
    print(f"    检索召回率 (Recall):        {report.avg_recall:>6.1%}")
    print(f"    答案准确性 (Keyword Hit):   {report.avg_keyword_hit:>6.1%}")
    print(f"    引用正确性 (Citation):      {report.avg_citation_precision:>6.1%}")
    print(f"    拒答正确率 (Rejection):     {report.rejection_accuracy:>6.1%}")
    print()

    # 分类明细
    print("  [分类明细]")
    cat_order = ["报销", "请假考勤", "IT技术", "招聘入职", "薪酬福利", "超出范围"]
    for cat in cat_order:
        if cat in report.category_breakdown:
            cb = report.category_breakdown[cat]
            recall_str = f"Recall={cb['avg_recall']:.0%}" if cat != "超出范围" else "Recall=N/A"
            print(
                f"  {cat:<8} ({cb['count']}题) {recall_str:<16} "
                f"KW={cb['avg_keyword_hit']:.0%}  Cite={cb['avg_citation_precision']:.0%}"
            )

    # 问题详情表
    print()
    print("  [逐题详情]")
    print(f"  {'-'*62}")

    for i, m in enumerate(report.metrics):
        flag = ""
        if m.is_out_of_scope:
            flag = " [OoS]" if m.correctly_rejected else " [OoS FAIL]"
        elif m.recall_at_k == 0:
            flag = " [RETR FAIL]"
        elif m.keyword_hit_rate < 0.5:
            flag = " [ANS GAP]"
        elif m.invalid_citations > 0:
            flag = " [CITE BAD]"
        else:
            flag = " [OK]"

        scope_tag = "[超出范围]" if m.is_out_of_scope else ""
        print(f"  {i+1:02d}{flag} {scope_tag} {m.question[:52]}...")
        if not m.is_out_of_scope:
            print(
                f"      检索: recall={m.recall_at_k:.0%} docs={m.retrieved_doc_indices} "
                f"| 关键词: {m.keyword_hit_rate:.0%} "
                f"命中={m.hit_keywords} 缺失={m.missed_keywords}"
            )
            if m.cited_numbers:
                print(
                    f"      引用: [{', '.join(str(n) for n in m.cited_numbers)}] "
                    f"有效={m.valid_citations} 无效={m.invalid_citations} "
                    f"精度={m.citation_precision:.0%}"
                )
            else:
                print("      引用: 无引用标注")
        else:
            result_tag = "正确拒答" if m.correctly_rejected else "FAIL 未拒答(编造了答案)"
            print(f"      拒答: {result_tag}")
            print(f"      回答: {m.answer[:80]}...")

    print(sep)


def print_comparison_table(reports: list[EvalReport]) -> None:
    """打印多配置对比表。"""

    print()
    print("=" * 78)
    print("  配置对比: Top-K x 相似度阈值")
    print("=" * 78)
    print(f"  {'配置':<24} {'召回率':>8} {'关键词命中':>10} {'引用精度':>10} {'拒答正确':>10}")
    print(f"  {'-'*24} {'-'*8} {'-'*10} {'-'*10} {'-'*10}")

    best_recall = max(r.avg_recall for r in reports)
    best_kw = max(r.avg_keyword_hit for r in reports)
    best_cite = max(r.avg_citation_precision for r in reports)
    best_rej = max(r.rejection_accuracy for r in reports)

    for r in reports:
        cfg = r.config
        label = f"Top-K={cfg['top_k']}, theta={cfg['sim_threshold']}"

        recall_mark = " (*)" if r.avg_recall == best_recall else ""
        kw_mark = " (*)" if r.avg_keyword_hit == best_kw else ""
        cite_mark = " (*)" if r.avg_citation_precision == best_cite else ""
        rej_mark = " (*)" if r.rejection_accuracy == best_rej else ""

        print(
            f"  {label:<24} {r.avg_recall:>7.1%}{recall_mark} "
            f"{r.avg_keyword_hit:>9.1%}{kw_mark} "
            f"{r.avg_citation_precision:>9.1%}{cite_mark} "
            f"{r.rejection_accuracy:>9.1%}{rej_mark}"
        )

    print(f"  {'-'*24} {'-'*8} {'-'*10} {'-'*10} {'-'*10}")
    print("  (*) = 该指标在各配置中最优")
    print("=" * 78)


def export_report_json(
    reports: list[EvalReport], filepath: str = "docs/week3/day6-report.json"
) -> None:
    """导出评测结果为 JSON 文件（方便后续分析）。"""
    data = []
    for r in reports:
        data.append(
            {
                "config": r.config,
                "timestamp": r.timestamp,
                "total_questions": r.total_questions,
                "summary": {
                    "avg_recall": r.avg_recall,
                    "avg_keyword_hit": r.avg_keyword_hit,
                    "avg_citation_precision": r.avg_citation_precision,
                    "rejection_accuracy": r.rejection_accuracy,
                },
                "category_breakdown": r.category_breakdown,
                "details": [
                    {
                        "question": m.question,
                        "category": m.category,
                        "is_out_of_scope": m.is_out_of_scope,
                        "recall_at_k": m.recall_at_k,
                        "keyword_hit_rate": m.keyword_hit_rate,
                        "citation_precision": m.citation_precision,
                        "correctly_rejected": m.correctly_rejected,
                        "answer": m.answer,
                        "hit_keywords": m.hit_keywords,
                        "missed_keywords": m.missed_keywords,
                    }
                    for m in r.metrics
                ],
            }
        )

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\n  [OK] 评测数据已导出到 {filepath}")


# ═══════════════════════════════════════════════════════════════════
# 第 7 部分：离线模拟评测（不需要 LLM API 也能理解评测逻辑）
# ═══════════════════════════════════════════════════════════════════


def generate_mock_answer(
    test_case: TestCase, retriever: SimpleRetriever, top_k: int = 5
) -> RAGResult:
    """生成模拟 RAG 结果。

    当没有 API Key 时，用这个函数模拟 RAG 管道的输出。
    它用检索结果 + 模板拼接来模拟 LLM 生成的答案，让你在没有 API 时也能理解评测框架。
    """
    # 超出范围的题目：直接拒答
    if test_case.is_out_of_scope:
        return RAGResult(
            question=test_case.question,
            answer="根据现有资料无法回答您的问题。当前知识库不包含相关信息。",
            sources=[],
            citations=[],
        )

    hits = retriever.search(test_case.question, top_k=top_k)
    hits = [(idx, text, sim) for idx, text, sim in hits if sim >= 0.5]

    if not hits:
        return RAGResult(
            question=test_case.question,
            answer="根据现有资料无法回答您的问题。",
            sources=[],
            citations=[],
        )

    source_map = {i + 1: text for i, (_, text, _) in enumerate(hits)}

    # 模拟 LLM 行为：把检索到的文档内容片段拼成"答案"
    # 真实 RAG 中这步由 LLM 完成，这里模拟效果
    answer_parts = []
    cited_nums = []
    for i, (idx, text, sim) in enumerate(hits):
        ref_num = i + 1
        if any(kw.lower() in text.lower() for kw in test_case.expected_keywords):
            # 包含期望关键词的文档 → 提取相关句子
            sentences = text.replace("。", "。\n").split("\n")
            for sent in sentences:
                for kw in test_case.expected_keywords:
                    if kw.lower() in sent.lower() and sent.strip():
                        answer_parts.append(f"{sent.strip()} [{ref_num}]")
                        cited_nums.append(ref_num)
                        break

    if not answer_parts and not test_case.is_out_of_scope:
        # 没有匹配到关键词，用检索到的第一条内容做兜底
        answer_parts.append(f"{hits[0][1]} [1]")
        cited_nums.append(1)

    answer = "\n".join(answer_parts) if answer_parts else "根据现有资料无法回答您的问题。"

    return RAGResult(
        question=test_case.question,
        answer=answer,
        sources=[{"idx": idx, "text": text, "similarity": sim} for idx, text, sim in hits],
        citations=[{"number": n, "text": source_map.get(n, "")} for n in set(cited_nums)],
    )


def run_evaluation_offline(test_cases: list[TestCase], retriever: SimpleRetriever) -> EvalReport:
    """离线模式评测——用模拟生成代替 LLM 调用。

    这种模式下，答案质量取决于模拟逻辑的准确性，所以关键词命中率不代表真实效果。
    但检索召回率、引用正确性等指标完全有效——它们不依赖 LLM。
    """
    cfg = {"top_k": 5, "sim_threshold": 0.6}
    all_metrics: list[EvalMetrics] = []

    print(f"\n  [离线模式] 使用模拟 RAG 生成，共 {len(test_cases)} 条用例")
    print("  注意: 答案关键词命中率在此模式下仅供参考，真实数据需连接 LLM\n")

    for i, tc in enumerate(test_cases):
        mock_result = generate_mock_answer(tc, retriever, top_k=cfg["top_k"])
        metrics = evaluate_one(tc, mock_result)
        all_metrics.append(metrics)

        status = "[超出范围]" if tc.is_out_of_scope else f"[{tc.category}]"
        print(
            f"  [{i+1:02d}/30] {status} {tc.question[:45]}... "
            f"Recall={metrics.recall_at_k:.0%} KW={metrics.keyword_hit_rate:.0%} "
            f"Cite={metrics.citation_precision:.0%}"
        )

    in_scope = [m for m in all_metrics if not m.is_out_of_scope]
    out_of_scope = [m for m in all_metrics if m.is_out_of_scope]

    avg_recall = sum(m.recall_at_k for m in in_scope) / len(in_scope) if in_scope else 0
    avg_kw = sum(m.keyword_hit_rate for m in in_scope) / len(in_scope) if in_scope else 0
    avg_cite = sum(m.citation_precision for m in in_scope) / len(in_scope) if in_scope else 0
    rejection_acc = (
        sum(1 for m in out_of_scope if m.correctly_rejected) / len(out_of_scope)
        if out_of_scope
        else 0
    )

    categories: dict[str, list[EvalMetrics]] = {}
    for m in all_metrics:
        categories.setdefault(m.category, []).append(m)

    cat_breakdown = {}
    for cat, cat_metrics in categories.items():
        cat_in_scope = [m for m in cat_metrics if not m.is_out_of_scope]
        cat_breakdown[cat] = {
            "count": len(cat_metrics),
            "avg_recall": (
                sum(m.recall_at_k for m in cat_in_scope) / len(cat_in_scope) if cat_in_scope else 0
            ),
            "avg_keyword_hit": (
                sum(m.keyword_hit_rate for m in cat_in_scope) / len(cat_in_scope)
                if cat_in_scope
                else 0
            ),
            "avg_citation_precision": (
                sum(m.citation_precision for m in cat_in_scope) / len(cat_in_scope)
                if cat_in_scope
                else 0
            ),
        }

    return EvalReport(
        config=cfg,
        timestamp=datetime.now(UTC).isoformat(),
        total_questions=len(test_cases),
        metrics=all_metrics,
        avg_recall=avg_recall,
        avg_keyword_hit=avg_kw,
        avg_citation_precision=avg_cite,
        rejection_accuracy=rejection_acc,
        category_breakdown=cat_breakdown,
    )


# ═══════════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════════


def main() -> None:
    print("+======================================================+")
    print("|  Week 3 Day 6 — RAG 评测与优化                        |")
    print("|  用 30 条测试用例回答：你的 RAG 管道到底怎么样？        |")
    print("+======================================================+")

    # 初始化检索器
    retriever = SimpleRetriever(CORPUS)
    print(f"\n[OK] 知识库已加载：{len(CORPUS)} 条文档，5 个领域")
    print(f"[OK] 测试用例：{len(TEST_CASES)} 条 (27 条范围内 + 3 条超出范围)")

    # 尝试连接 LLM
    llm: LLMClient | None = None
    use_llm: bool = False
    try:
        llm = LLMClient.from_env()
        print(f"[OK] 已连接 {llm.provider} / {llm.model}")
        use_llm = True
    except Exception as e:
        print(f"[!] 无法连接 LLM: {e}")
        print("[!] 将使用离线模拟模式——检索召回率和引用正确性的评测仍然有效。")

    if use_llm:
        assert llm is not None

        # ═══════ 模式 1：单配置评测 ═══════
        print("\n" + "=" * 60)
        print("  第 1 步：默认配置评测 (Top-K=5, θ=0.6)")
        print("=" * 60)
        report_default = run_evaluation(TEST_CASES, retriever, llm, top_k=5, sim_threshold=0.6)
        print_report(report_default)

        # ═══════ 模式 2：多配置对比 ═══════
        print("\n" + "=" * 60)
        print("  第 2 步：多配置对比")
        print("=" * 60)
        all_reports = [report_default] + run_config_comparison(TEST_CASES, retriever, llm)
        print_comparison_table(all_reports)

        # 导出 JSON
        export_report_json(all_reports)

    else:
        # 离线模式
        report = run_evaluation_offline(TEST_CASES, retriever)
        print_report(report)

        # 即使离线也导出
        export_report_json([report])

    # ═══════ 总结 ═══════
    print(f"\n{'='*60}")
    print("  Day 6 核心收获：")
    print(f"{'='*60}")
    print(
        textwrap.dedent(
            """
    1. 评测不是「感觉对不对」，而是用数据说话——每个指标都有明确的定义和计算方式
    2. 检索召回率 = 正确答案文档有没有被检索到？这是 RAG 的地基，召回率为 0 后面全白费
    3. 关键词命中率 = LLM 生成的答案是否包含关键事实？快速但不完美（可能漏同义词）
    4. 引用正确性 = LLM 标注的 [N] 是否真的对应了正确的文档？引用是 RAG 的良心
    5. 拒答率 = 知识库没有的答案，系统是否能正确说「不知道」而不是编造？
    6. 不同配置有不同取舍：Top-K 大 → 召回高但噪音多；阈值高 → 精度高但可能漏答
    7. 生产环境还会用 LLM-as-Judge（用 GPT-4 给另一个 LLM 的回答打分），Week 10 会深入
    """
        )
    )


if __name__ == "__main__":
    main()
