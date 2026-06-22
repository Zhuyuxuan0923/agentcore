"""W2D5 Demo — Prompt 评测：3 种 Prompt 写法对比 50 条测试用例

这个脚本做的事：
  1. 定义 3 种 prompt 风格（最简 / 结构化 / Few-Shot）
  2. 加载 50 条客服消息分类测试用例
  3. 每种风格跑一遍全部用例
  4. 输出 Markdown 对比报告

如何运行：
  # 用默认 provider（环境变量 LLM_PROVIDER，默认 deepseek）
  python src/study_agent/w2d5_demo.py

  # 指定 provider
  $env:LLM_PROVIDER="anthropic"
  python src/study_agent/w2d5_demo.py

  # 只跑前 10 条（快速验证，约 1 分钟）
  python src/study_agent/w2d5_demo.py --quick

三种 Prompt 风格的设计理念：

  ┌──────────────┬────────────────────────┬───────────────────────┐
  │ 风格          │ 特点                    │ 类比                  │
  ├──────────────┼────────────────────────┼───────────────────────┤
  │ minimal      │ 只给任务，其他什么都不说 │ 对实习生说"去写报告"   │
  │ structured   │ 角色+规则+输出格式       │ 给模板+格式+要求       │
  │ fewshot      │ 角色+规则+输出格式+范例   │ 给模板+3份参考样例     │
  └──────────────┴────────────────────────┴───────────────────────┘

预期结果：
  fewshot > structured >> minimal
  （规则 + 范例让 LLM 能更准确地理解你的分类标准）
"""

from __future__ import annotations

import argparse
import logging

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("w2d5")

# ═══════════════════════════════════════════════════════════════
# ① 三种 Prompt 风格定义
# ═══════════════════════════════════════════════════════════════
#
# 每种风格是一个函数，签名是：
#   def xxx_prompt(text: str) -> tuple[str | None, str]:
#       返回 (system_prompt, user_prompt)
#
# system_prompt = 给 LLM 的"行为准则"（角色、规则、约束）
# user_prompt   = 给 LLM 的"具体任务"（要分类的文本）
#
# 你可以修改这三个函数来实验不同的 prompt 写法！
# 改完后重新跑 python src/study_agent/w2d5_demo.py 看效果变化。


def build_minimal_prompt(text: str) -> tuple[str | None, str]:
    """风格 A：最简 Prompt —— 只告诉 LLM "做什么"，不给任何指导。

    这是大多数人的"第一版 prompt"——觉得 LLM 什么都能理解，不需要多说。

    v2 改进：之前连 category 有哪些选项都不说，LLM 只能凭空猜。
    现在至少列出五个选项名——这是"最简"的底线。
    """
    system = (
        "请将用户消息分类。category 从以下五项中选择："
        "billing、technical、account、product、complaint。"
        "priority 为 high/medium/low。"
        "sentiment 为 negative/neutral/positive。"
        "只输出 JSON，不要其他文字。"
    )
    user = f"消息：{text}"
    return system, user


def build_structured_prompt(text: str) -> tuple[str | None, str]:
    """风格 B：结构化 Prompt —— 定义角色 + 决策树 + 边界规则 + 输出格式。

    v2 改进（针对第一轮 category 准确率只有 70% 的问题）：
      1. 新增"分类决策树"——先判断是不是钱的问题，再看其他
      2. 每个 category 增加"什么是"和"什么不是"的正反例
      3. 边界 case 明确规则：支付报错=billing 不是 technical；
         抱怨账单=billing 不是 complaint；问套餐价格=billing 不是 product
    """
    system = (
        "你是一位经验丰富的客服分类专家，负责将用户消息精确分类。\n"
        "\n"
        "## 分类决策树（按优先级从上到下，匹配第一个就停止）\n"
        "\n"
        "1. 涉及钱/付款/扣款/退款/发票/订阅/套餐价格 → **billing**\n"
        "   重要：即使语气像投诉或问题涉及技术，只要根因是钱，就是 billing！\n"
        "   例如：「支付失败」→ billing（不是 technical）\n"
        "   例如：「乱扣费太坑了」→ billing（不是 complaint）\n"
        "   例如：「年付有优惠吗」→ billing（不是 product）\n"
        "\n"
        "2. 涉及对服务/客服/产品的不满、要求赔偿、情绪宣泄 → **complaint**\n"
        "   前提：不涉及金钱（涉及钱已经归类为 billing 了）\n"
        "   例如：「客服三天不回」→ complaint\n"
        "   例如：「功能越砍越少」→ complaint\n"
        "\n"
        "3. 涉及 Bug/报错/闪退/兼容性/加载慢/API 故障 → **technical**\n"
        "   前提：不涉及支付（支付报错归 billing）\n"
        "   例如：「App 一直闪退」→ technical\n"
        "   例如：「CSV 导出乱码」→ technical\n"
        "\n"
        "4. 涉及登录/密码/注册/注销/安全设置/账号冻结 → **account**\n"
        "   例如：「密码忘了」→ account\n"
        "   例如：「账号被冻结」→ account\n"
        "\n"
        "5. 其他功能咨询、使用方法、规格对比、集成 → **product**\n"
        "   例如：「支持批量导入吗」→ product\n"
        "\n"
        "## priority 判断标准\n"
        "- high：涉及金钱损失/账户安全/数据丢失/完全无法使用\n"
        "- medium：影响使用但不完全阻塞\n"
        "- low：一般咨询、赞扬、无紧急性\n"
        "\n"
        "## sentiment 判断标准\n"
        "- negative：生气、沮丧、失望、焦虑（看语气，不是看分类）\n"
        "- neutral：客观描述、询问，无明显情绪\n"
        "- positive：满意、感谢、喜欢\n"
        "\n"
        "## 输出要求\n"
        "- 只输出一个 JSON 对象，字段名小写：category, priority, sentiment\n"
        "- category 值必须恰好是 billing/technical/account/product/complaint 之一\n"
        "- priority 值必须恰好是 high/medium/low 之一\n"
        "- sentiment 值必须恰好是 negative/neutral/positive 之一\n"
        "- 不输出任何 JSON 以外的文字，不用 ``` 包裹"
    )
    user = f"消息：{text}"
    return system, user


def build_fewshot_prompt(text: str) -> tuple[str | None, str]:
    """风格 C：Few-Shot Prompt —— 结构化 prompt + 4 个精心挑选的范例。

    v2 改进（针对第一轮 category 80% 仍不够好的问题）：
      范例选择策略改变——不再展示"简单题"，而是展示"容易搞混的边界 case"。
      四个例子分别教会 LLM 四条关键边界：
        例 1 → 付款遇到技术问题 ≠ technical，是 billing
        例 2 → 抱怨账单扣费 ≠ complaint，是 billing
        例 3 → 问订阅/续费 ≠ account，是 billing
        例 4 → 真正的 complaint 长什么样（无关金钱的服务投诉）
    """
    system = (
        "你是一位经验丰富的客服分类专家。\n"
        "\n"
        "## 分类决策树（按优先级，匹配即停）\n"
        "\n"
        "1. 涉及钱/付款/扣款/退款/发票/订阅/套餐 → **billing**\n"
        "   （支付报错=billing，抱怨扣费=billing，问套餐价格=billing）\n"
        "2. 不涉及钱的不满/投诉/要求赔偿 → **complaint**\n"
        "3. 技术故障/Bug/兼容性（不涉及支付） → **technical**\n"
        "4. 登录/密码/注册/安全 → **account**\n"
        "5. 功能咨询/使用方法 → **product**\n"
        "\n"
        "## priority\n"
        "- high：金钱损失/安全风险/数据丢失/完全不可用\n"
        "- medium：影响使用但不阻塞\n"
        "- low：咨询/赞扬\n"
        "\n"
        "## sentiment\n"
        "- negative：生气/沮丧/失望/焦虑\n"
        "- neutral：客观/无情绪\n"
        "- positive：满意/感谢\n"
        "\n"
        "## 范例\n"
        "\n"
        "例 1 — 付款时遇到技术问题 → billing（不是 technical）\n"
        "输入: 支付宝付款一直转圈圈，钱扣了但订单没生成\n"
        '输出: {"category": "billing", "priority": "high", "sentiment": "negative"}\n'
        "\n"
        "例 2 — 抱怨扣费问题 → billing（不是 complaint）\n"
        "输入: 上个月就取消订阅了，怎么这月又扣我钱？给我退回来！\n"
        '输出: {"category": "billing", "priority": "high", "sentiment": "negative"}\n'
        "\n"
        "例 3 — 查询续费/套餐 → billing（不是 account 也不是 product）\n"
        "输入: 我的会员还有多久到期？续费的话有没有折扣？\n"
        '输出: {"category": "billing", "priority": "low", "sentiment": "neutral"}\n'
        "\n"
        "例 4 — 真正的 complaint（不涉及钱的服务投诉）\n"
        "输入: 你们的客服态度也太差了，问个问题爱答不理的\n"
        '输出: {"category": "complaint", "priority": "medium", "sentiment": "negative"}\n'
        "\n"
        "## 输出要求\n"
        "- 只输出 JSON，字段 category/priority/sentiment 三个\n"
        "- 值必须来自限定列表，不要自创\n"
        "- 不要输出任何其他文字"
    )
    user = f"消息：{text}"
    return system, user


# ═══════════════════════════════════════════════════════════════
# ② 主流程
# ═══════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="Prompt 评测——对比 3 种 prompt 写法")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="快速模式：只跑前 10 条用例（约 1 分钟）",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        help="LLM provider（覆盖环境变量 LLM_PROVIDER）",
    )
    parser.add_argument(
        "--export-csv",
        type=str,
        default=None,
        help="导出详细结果到 CSV 文件（如: results.csv）",
    )
    args = parser.parse_args()

    # ── 导入 ──
    from study_agent.llm import LLMClient, StructuredExtractor
    from study_agent.prompt.evaluator import CLASSIFY_SCHEMA_DEF, PromptEvaluator
    from study_agent.prompt.test_cases import CLASSIFICATION_CASES

    # ── 创建 LLM 客户端 ──
    if args.provider:
        client = LLMClient(provider=args.provider)
    else:
        client = LLMClient.from_env()
    logger.info("Provider: %s, Model: %s", client.provider, client.model)

    # ── 准备测试用例 ──
    cases = CLASSIFICATION_CASES[:10] if args.quick else CLASSIFICATION_CASES
    logger.info("测试用例数: %d", len(cases))

    # ── 定义要对比的 prompt 风格 ──
    # 这个字典的 key 是风格名称，value 是 prompt 构建函数
    # 你可以添加更多风格来实验！
    styles = {
        "minimal": build_minimal_prompt,
        "structured": build_structured_prompt,
        "fewshot": build_fewshot_prompt,
    }

    # ── 创建评测器 ──
    extractor = StructuredExtractor(client)
    evaluator = PromptEvaluator(client, extractor, CLASSIFY_SCHEMA_DEF)

    # ── 跑评测 ──
    total_calls = len(cases) * len(styles)
    print(f"\n{'='*60}")
    print(f"开始评测: {len(cases)} 条用例 × {len(styles)} 种风格 = {total_calls} 次调用")
    print(f"{'='*60}\n")

    results = evaluator.run_batch(cases, styles)

    # ── 输出报告 ──
    report = evaluator.report(results)
    print(report)

    # ── 可选：导出 CSV ──
    if args.export_csv:
        csv_content = evaluator.export_csv(results)
        with open(args.export_csv, "w", encoding="utf-8") as f:
            f.write(csv_content)
        logger.info("详细结果已导出到: %s", args.export_csv)

    # ── 快速验证提示 ──
    if args.quick:
        print("\n---")
        print("这是快速模式（10 条用例）。完整评测请运行:")
        print("  python src/study_agent/w2d5_demo.py")
        print("（完整评测约 50 条 × 3 种风格 = 150 次调用，约 2-5 分钟）")


if __name__ == "__main__":
    main()
