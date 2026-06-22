"""W2D6 Demo — System Prompt 设计实战：3 种设计哲学 × 50 条用例对比

本脚本是 Week 2 Day 6 的核心内容。W2D5 构建了评测工具，
W2D6 则聚焦于 **如何使用评测工具来改进 prompt 设计**。

═══════════════════════════════════════════════════════════════
核心学习目标
═══════════════════════════════════════════════════════════════

学完本日内容后，你应该能回答：

  1. 为什么"说清楚为什么"比"只说做什么"更有效？
  2. 正面例子和反面例子，哪个对边界判断帮助更大？
  3. 让 LLM "先想再说"（推理模板）真的能提高准确率吗？
  4. 如何通过评测报告找到 prompt 的具体弱点？

═══════════════════════════════════════════════════════════════
三种设计哲学对比
═══════════════════════════════════════════════════════════════

  ┌──────────────────┬──────────────────────┬──────────────────────┐
  │ 设计哲学          │ 核心思路              │ 要验证的假设          │
  ├──────────────────┼──────────────────────┼──────────────────────┤
  │ explain_why      │ 解释分类的"为什么"    │ 理解业务目的 ->        │
  │                  │ 给 LLM 建立业务上下文  │ 更准确的判断          │
  ├──────────────────┼──────────────────────┼──────────────────────┤
  │ negative_examples│ 给出常见错误 + 为什么错│ 知道"什么不是" ->      │
  │                  │ 用反例划定边界          │ 更少踩坑              │
  ├──────────────────┼──────────────────────┼──────────────────────┤
  │ reasoning_template│ 给出思考步骤模板      │ 结构化推理 ->          │
  │                  │ 先分析信号 -> 再判断    │ 更稳定准确的结果      │
  └──────────────────┴──────────────────────┴──────────────────────┘

═══════════════════════════════════════════════════════════════
如何使用
═══════════════════════════════════════════════════════════════

  # 完整评测（50 条用例 × 3 种风格 = 150 次调用，约 2-5 分钟）
  python src/study_agent/w2d6_demo.py

  # 快速模式（只跑前 10 条，约 30 秒）
  python src/study_agent/w2d6_demo.py --quick

  # 同时对比 W2D5 的最佳风格（4 种风格 = 200 次调用）
  python src/study_agent/w2d6_demo.py --compare-with-w2d5

  # 导出 CSV 详细结果
  python src/study_agent/w2d6_demo.py --export-csv w2d6_results.csv

═══════════════════════════════════════════════════════════════
✍️ 动手实验区
═══════════════════════════════════════════════════════════════

本脚本中有三处标记了 ✍️ 的区域，你可以修改其中的 prompt 内容，
重新运行看效果变化。建议的实验顺序：

  实验 1：修改 v1 中某个分类的"为什么"，看准确率怎么变
  实验 2：给 v2 增加一条你发现的新的"误判边界"
  实验 3：修改 v3 的推理步骤数量（3 步 vs 5 步），看影响

每次改完运行 `python src/study_agent/w2d6_demo.py --quick` 快速验证。
"""

from __future__ import annotations

import argparse
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("w2d6")


# ═══════════════════════════════════════════════════════════════
# ① 三种 Prompt 设计哲学
# ═══════════════════════════════════════════════════════════════
#
# 每种设计哲学对应一个函数，签名统一为：
#   def build_xxx_prompt(text: str) -> tuple[str | None, str]:
#       返回 (system_prompt, user_prompt)
#
# system_prompt -> 给 LLM 的"角色说明书"，定义它的行为准则
# user_prompt   -> 给 LLM 的"具体任务"，即要分类的消息文本


# ── 哲学 A：说清楚为什么（Explain Why） ──────────────────────
#
# 核心假设：LLM 理解了分类的"业务目的"后，会在边界 case 上做出
# 更符合人类期望的判断。
#
# 对比 W2D5 的 structured 版本（纯规则），这个版本多了：
#   - 每个分类对应的处理团队（让 LLM 知道"分错了会怎样"）
#   - 每个分类的核心用户诉求（让 LLM 理解用户背后的需求）
#   - 误分类的代价说明（让 LLM 重视准确性）


def build_explain_why_prompt(text: str) -> tuple[str | None, str]:
    """设计哲学 A：说清楚为什么 —— 解释每个分类背后的业务逻辑。

    与 W2D5 structured 版本的核心区别：
      structured 说"涉及钱->billing"（规则）
      explain_why 说"涉及钱->billing，因为需要财务团队处理，延误可能造成实际损失"（规则+原因）

    假设：理解 WHY 能帮助 LLM 在边界 case 上做出更好的判断。
    """

    # ✍️ 动手实验 1：尝试修改某个分类的"为什么"部分，
    # 比如把 billing 的"对应用户诉求"改成你认为更准确的描述，
    # 然后用 --quick 模式重新跑，看效果变化。
    system = (
        "你是一位资深客服分类专家，你的工作是将用户消息路由到正确的处理团队。\n"
        "\n"
        "## 为什么分类准确性很重要？\n"
        "\n"
        "每一次分类错误意味着：用户被转给错误的团队 -> 重新排队 -> 重复解释问题。\n"
        "这会让用户满意度下降 40%（行业数据），并增加 15 分钟平均处理时间。\n"
        "你的分类直接影响用户体验和公司运营效率。\n"
        "\n"
        "## 五个分类及其业务含义\n"
        "\n"
        "### 1. billing — 账单与付款\n"
        "- 对应用户诉求：「我的钱怎么了？」或「这个多少钱？」\n"
        "- 路由到：财务/支付团队（响应时限：high 类 30 分钟内）\n"
        "- 常见场景：扣款疑问、退款请求、发票申请、套餐变更、支付失败\n"
        "- **关键判断标准**：消息中是否涉及「钱」的流动——\n"
        "  付款、扣费、退款、发票、订阅、套餐价格。只要涉及钱，就是 billing。\n"
        "- **常见误判**：支付报错看起来像 technical，但根因是钱，归 billing。\n"
        "  抱怨乱扣费听起来像 complaint，但涉及具体金钱，归 billing。\n"
        "\n"
        "### 2. complaint — 投诉与不满\n"
        "- 对应用户诉求：「我不满意，给我一个说法」\n"
        "- 路由到：客户关系团队（需特别关注，有流失风险）\n"
        "- 常见场景：服务态度差、功能缩水、响应太慢、对产品方向不满\n"
        "- **前提条件**：不直接涉及金钱（涉及钱的按 billing 处理）\n"
        "- **关键判断标准**：用户在表达不满、抱怨、谴责，但核心诉求是\n"
        "  「被重视/被补偿/被道歉」，而不是「处理一笔具体的钱」。\n"
        "\n"
        "### 3. technical — 技术问题\n"
        "- 对应用户诉求：「这东西坏了，帮我修好」\n"
        "- 路由到：技术支撑团队\n"
        "- 常见场景：闪退、报错、加载慢、不兼容、功能异常\n"
        "- **前提条件**：不涉及支付/金钱（支付报错归 billing）\n"
        "- **关键判断标准**：用户描述的是一个「功能异常」——某功能不按预期工作。\n"
        "\n"
        "### 4. account — 账户管理\n"
        "- 对应用户诉求：「我进不去了」或「我的账户设置需要改」\n"
        "- 路由到：账户安全/运营团队\n"
        "- 常见场景：登录问题、密码重置、安全设置、注销、权限管理\n"
        "\n"
        "### 5. product — 产品咨询\n"
        "- 对应用户诉求：「这个功能怎么用？」或「你们和竞品比怎么样？」\n"
        "- 路由到：产品顾问团队\n"
        "- 常见场景：功能使用方法、规格对比、集成方案、价格方案咨询\n"
        "- **注意**：问价格方案 ≠ billing。问「年付有优惠吗」是 billing，\n"
        "  问「免费版和付费版有什么区别」是 product。\n"
        "  区别在于：前者涉及具体的付费行为，后者只是了解产品功能边界。\n"
        "\n"
        "## priority 判断标准\n"
        "\n"
        "- **high**：用户面临实际损失——钱被多扣、账号被盗、完全无法使用\n"
        "- **medium**：影响使用但不阻塞——某功能不好用、等待回复超过预期\n"
        "- **low**：一般咨询、赞扬、不紧急的询问\n"
        "\n"
        "## sentiment 判断标准\n"
        "\n"
        "- **negative**：包含明确的负面情绪词（生气、沮丧、失望、焦虑）\n"
        "  或整体语气是抱怨/指责/不满的\n"
        "- **neutral**：客观描述事实或提问，无明显情绪色彩\n"
        "- **positive**：包含感谢、赞扬、满意、推荐意愿\n"
        "\n"
        "## 输出要求\n"
        "只输出一个 JSON 对象，字段为 category、priority、sentiment。\n"
        "不输出任何 JSON 以外的文字，不用代码块包裹。"
    )

    user = f"请分类以下客服消息：\n\n{text}"
    return system, user


# ── 哲学 B：举反例（Negative Examples） ──────────────────────
#
# 核心假设：告诉 LLM "什么不是 X"，比告诉它"什么是 X"更能防止误判。
#
# 为什么反例可能更有效？
#   正面例子告诉 LLM"正确的路在哪里"，
#   反面例子告诉 LLM"哪条路看起来像但其实是死胡同"。
#   对于分类任务来说，边界 case 的误判是主要错误来源，
#   而反例直接针对边界做标记。
#
# 与 W2D5 fewshot（4 个正面例子）的区别：
#   fewshot 展示的是"对的答案长什么样"，
#   本版本展示的是"容易错的陷阱长什么样"。


def build_negative_examples_prompt(text: str) -> tuple[str | None, str]:
    """设计哲学 B：举反例 —— 用常见误判案例帮助 LLM 避开陷阱。

    与 W2D5 fewshot 版本的核心区别：
      fewshot 给出 4 个「正确答案」-> 告诉 LLM 什么是好的
      negative_examples 给出 6 个「常见错误」-> 告诉 LLM 什么容易搞混

    假设：在边界模糊的分类任务中，反例比正例更有信息量。
    """

    # ✍️ 动手实验 2：仔细观察 --quick 运行的评测结果，
    # 找到一条 LLM 经常分错的 case，在这里加一个新的反例条目，
    # 重新运行看准确率是否提升。
    system = (
        "你是一位客服消息分类专家。请将每条用户消息分类为以下五类之一：\n"
        "billing、technical、account、product、complaint。\n"
        "\n"
        "## 分类定义\n"
        "\n"
        "- **billing**：涉及付款、扣款、退款、发票、订阅、套餐价格\n"
        "- **technical**：涉及 Bug、报错、闪退、兼容性、功能异常\n"
        "- **account**：涉及登录、密码、安全、注册、注销、权限\n"
        "- **product**：涉及功能咨询、使用方法、规格对比、方案了解\n"
        "- **complaint**：涉及不满、投诉、要求赔偿、情绪宣泄\n"
        "\n"
        "## priority 判断\n"
        "- **high**：涉及金钱损失/安全风险/完全无法使用\n"
        "- **medium**：影响使用但不完全阻塞\n"
        "- **low**：一般咨询或赞扬\n"
        "\n"
        "## sentiment 判断\n"
        "- **negative**：生气/沮丧/失望/焦虑\n"
        "- **neutral**：客观描述，无明显情绪\n"
        "- **positive**：满意/感谢/喜欢\n"
        "\n"
        "## [!] 常见误判陷阱 —— 以下情况最容易分错，请特别留意\n"
        "\n"
        "### 陷阱 1：支付失败 ≠ technical\n"
        '[X] 错误："支付宝付款一直转圈圈，钱扣了但订单没生成" -> technical\n'
        "[OK] 正确：-> billing\n"
        "**原因**：虽然有技术现象，但根因是支付流程。涉及钱的优先归 billing。\n"
        "\n"
        "### 陷阱 2：语气愤怒 ≠ complaint\n"
        '[X] 错误："上个月就取消订阅了，怎么这月又扣我钱？给我退回来！" -> complaint\n'
        "[OK] 正确：-> billing\n"
        "**原因**：虽然情绪强烈，但核心诉求是「处理扣款」——涉及具体金钱，归 billing。\n"
        "complaint 是「不涉及金钱的服务不满」。\n"
        "\n"
        "### 陷阱 3：问订阅/续费 ≠ account\n"
        '[X] 错误："我的会员还有多久到期？续费有没有折扣？" -> account\n'
        "[OK] 正确：-> billing\n"
        "**原因**：订阅到期时间和续费都涉及付费，归 billing。\n"
        "account 是登录/密码/安全设置，不涉及付费行为。\n"
        "\n"
        "### 陷阱 4：问套餐/价格方案 ≠ billing\n"
        '[X] 错误："免费版和付费版具体差在哪里？有对比表吗？" -> billing\n'
        "[OK] 正确：-> product\n"
        "**原因**：用户只是想了解产品功能差异，没有进行或讨论具体付费行为。\n"
        '问"年付有优惠吗"是 billing（涉及具体付费决策），问"有什么区别"是 product。\n'
        "\n"
        "### 陷阱 5：描述功能不好用 ≠ complaint\n"
        '[X] 错误："升级后界面变得好难用，能不能退回旧版？" -> complaint\n'
        "[OK] 正确：-> product（如果只描述产品体验）或 technical（如果是功能异常）\n"
        "**原因**：用户在对产品设计提意见，基调是反馈而非投诉。\n"
        "complaint 需要「明确的不满情绪 + 要求对方负责」，而不仅是提意见。\n"
        "\n"
        "### 陷阱 6：数据丢失/系统故障 ≠ technical\n"
        '[X] 错误："系统故障导致我丢了三个月的项目数据，你们怎么赔偿？" -> technical\n'
        "[OK] 正确：-> complaint\n"
        "**原因**：虽然有技术故障，但用户的核心诉求是「要求赔偿」——\n"
        "这是投诉的核心特征。而且这里不涉及支付，所以归 complaint。\n"
        "\n"
        "## 输出要求\n"
        '只输出 JSON：{"category": "...", "priority": "...", "sentiment": "..."}\n'
        "不要输出其他文字。"
    )

    user = f"消息：{text}"
    return system, user


# ── 哲学 C：推理模板（Reasoning Template） ──────────────────
#
# 核心假设：让 LLM 先"说出思考过程"再做判断，比让它"直接给答案"
# 能产生更准确的结果。这就是著名的 Chain-of-Thought（CoT）技术。
#
# 为什么"先想再说"有效？
#   LLM 生成文本是逐 token 的——前面的 token 会影响后面的 token。
#   如果第一步就跳到结论（"这是 billing"），后面的 token 会围绕
#   这个结论展开，忽略与结论不一致的信号。
#   如果第一步是"先列出所有信号"，则所有信息都有机会被关注。
#
#   类比：做数学题时，在草稿纸上列步骤的人比心算的人准确率高。
#         不是因为他们更聪明，而是因为"外化思考过程"减少了遗漏。


def build_reasoning_template_prompt(text: str) -> tuple[str | None, str]:
    """设计哲学 C：推理模板 —— 给 LLM 一个"思考草稿纸"。

    与 W2D5 fewshot 版本的核心区别：
      fewshot 让 LLM 模仿正确答案的「结果」
      reasoning_template 让 LLM 模仿正确的「思考方式」

    假设：控制推理过程比控制输出格式更能提升准确率。
    """

    # ✍️ 动手实验 3：尝试修改推理步骤。比如把 3 步改成 5 步：
    #   步骤 1：提取消息中的关键词
    #   步骤 2：判断是否涉及金钱
    #   步骤 3：判断是否有负面情绪
    #   步骤 4：根据决策树确定 category
    #   步骤 5：确定 priority 和 sentiment
    # 然后运行 --quick 看效果。步骤多了更好还是更差？

    system = (
        "你是一位客服消息分类专家。\n"
        "\n"
        "## 你的任务\n"
        "将用户消息分类为以下五类之一：billing、technical、account、product、complaint。\n"
        "同时判断 priority（high/medium/low）和 sentiment（negative/neutral/positive）。\n"
        "\n"
        "## 推理流程 —— 请严格按以下步骤思考\n"
        "\n"
        "在给出最终分类之前，请先在心中完成以下三个步骤。\n"
        "你的最终输出中，先写分析过程，再写 JSON 结果。\n"
        "\n"
        "### 步骤 1：信号检测\n"
        "扫描消息，找出以下信号：\n"
        "- 是否涉及金钱？（付款、扣费、退款、订阅、价格） -> 有/无\n"
        "- 是否描述技术异常？（报错、闪退、不兼容、加载慢） -> 有/无\n"
        "- 是否涉及账户操作？（登录、密码、安全、绑定、注销） -> 有/无\n"
        "- 情绪级别？（生气/中性/满意） -> 判断\n"
        "\n"
        "### 步骤 2：按决策树分类\n"
        "按以下优先级判断 category，匹配第一个就停止：\n"
        "\n"
        "1. 涉及金钱？ -> billing（即使同时有技术现象或不满情绪）\n"
        "   理由：涉及金钱的问题需要财务团队优先处理\n"
        "2. 明确不满+要求对方负责，且不涉及金钱？ -> complaint\n"
        "   理由：投诉需要客户关系团队介入防止流失\n"
        "3. 描述技术异常/Bug/兼容性？ -> technical（前提：不涉及金钱）\n"
        "4. 涉及账户/登录/安全？ -> account\n"
        "5. 以上都不是？ -> product\n"
        "\n"
        "### 步骤 3：确定 priority 和 sentiment\n"
        "- priority：有实际损失（钱/数据/安全）-> high；影响使用但不阻塞 -> medium；其他 -> low\n"
        "- sentiment：看用户的用词和语气，而非看 category 是什么\n"
        '  例如："你们涨价了，不过我理解" — 这可以是 neutral，虽然涉及不愉快的话题\n'
        "\n"
        "## 输出格式\n"
        "\n"
        "先输出简短的分析（2-3 句话），然后输出 JSON。\n"
        'JSON 格式：{"category": "...", "priority": "...", "sentiment": "..."}\n'
        "\n"
        "示例输出：\n"
        "分析：消息涉及扣款问题，金钱信号=有。虽然用户情绪愤怒，但根因是金钱，按决策树第1条归 billing。涉及金钱损失 -> priority=high。语气愤怒 -> sentiment=negative。\n"
        '结果：{"category": "billing", "priority": "high", "sentiment": "negative"}\n'
        "\n"
        "现在开始分析用户消息。"
    )

    user = f"消息：{text}"
    return system, user


# ═══════════════════════════════════════════════════════════════
# ② 可选：W2D5 对照组
# ═══════════════════════════════════════════════════════════════
#
# 当你用 --compare-with-w2d5 运行时，会额外加载 W2D5 的最佳风格
# 作为对照。这样你就能看到：新设计是否超越了旧方案？


def build_w2d5_fewshot_prompt(text: str) -> tuple[str | None, str]:
    """W2D5 的 fewshot 版本（复用，作为对照组）。

    这是 W2D5 三轮迭代后的最佳版本：规则 + 4 个正面例子。
    将它作为 baseline，看 W2D6 的三种新设计能否超越它。
    """
    system = (
        "你是一位经验丰富的客服分类专家。\n"
        "\n"
        "## 分类决策树（按优先级，匹配即停）\n"
        "\n"
        "1. 涉及钱/付款/扣款/退款/发票/订阅/套餐 -> **billing**\n"
        "   （支付报错=billing，抱怨扣费=billing，问套餐价格=billing）\n"
        "2. 不涉及钱的不满/投诉/要求赔偿 -> **complaint**\n"
        "3. 技术故障/Bug/兼容性（不涉及支付） -> **technical**\n"
        "4. 登录/密码/注册/安全 -> **account**\n"
        "5. 功能咨询/使用方法 -> **product**\n"
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
        "例 1 — 付款时遇到技术问题 -> billing（不是 technical）\n"
        "输入: 支付宝付款一直转圈圈，钱扣了但订单没生成\n"
        '输出: {"category": "billing", "priority": "high", "sentiment": "negative"}\n'
        "\n"
        "例 2 — 抱怨扣费问题 -> billing（不是 complaint）\n"
        "输入: 上个月就取消订阅了，怎么这月又扣我钱？给我退回来！\n"
        '输出: {"category": "billing", "priority": "high", "sentiment": "negative"}\n'
        "\n"
        "例 3 — 查询续费/套餐 -> billing（不是 account 也不是 product）\n"
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
# ③ 增强版报告 —— 比 W2D5 多了"设计哲学分析"
# ═══════════════════════════════════════════════════════════════


def print_design_analysis(summary: dict) -> None:
    """在评测报告后打印设计哲学分析。

    这不是自动生成的——它基于我们对三种设计哲学的理解，
    结合评测结果给出解读。目的是教会你"如何阅读评测数据"。
    """
    print("\n" + "=" * 60)
    print("  设计哲学分析：从数字看设计优劣")
    print("=" * 60)

    # ── 找到各风格的指标 ──
    styles = {}
    for name, stats in summary.items():
        styles[name] = stats

    print("\n[核心指标] 先看核心指标：\n")

    # 按通过率排序
    ranked = sorted(summary.items(), key=lambda x: x[1]["overall_rate"], reverse=True)
    for i, (name, stats) in enumerate(ranked):
        medal = ["1st", "2nd", "3rd", "4."][i] if i < 4 else f"{i+1}."
        print(
            f"  {medal} {name:<25s}  "
            f"通过率 {stats['overall_rate']:.0%}  "
            f"JSON合法 {stats['json_valid_rate']:.0%}  "
            f"字段完整 {stats['fields_complete_rate']:.0%}"
        )

    # ── 分析 1：解释"为什么"有多大的作用？ ──
    print("\n[分析] 分析 1：'说清楚为什么' 的效果如何？\n")

    if "explain_why" in styles and "reasoning_template" in styles:
        ew = styles["explain_why"]
        rt = styles["reasoning_template"]

        print("  explain_why 的设计假设是：LLM 理解业务背景后会在边界 case 上")
        print("  做出更好的判断。")
        print()
        print(
            f"  实际结果：通过率 {ew['overall_rate']:.0%}，"
            f"JSON 合法率 {ew['json_valid_rate']:.0%}"
        )
        print()

        if ew["overall_rate"] < 0.75:
            print("  [!] 通过率偏低。可能原因：")
            print("    1. '说清楚为什么'增加了很多文字，但核心的决策逻辑不够清晰")
            print("    2. 业务背景信息可能让 LLM '想太多'——")
            print("       本来简单的判断，因为考虑了过多因素而犹豫")
            print("    3. 解决方法：在保留'为什么'的同时，增加明确的决策规则")
        else:
            print("  [OK] 通过率较好。'说清楚为什么'确实有帮助，尤其是在：")
            print("    - 边界 case 上（LLM 理解了每个分类的'精神'）")
            print("    - 长文本上（丰富的上下文帮助 LLM 定位关键信息）")

    # ── 分析 2：反例 vs 正例 ──
    print("\n[分析] 分析 2：'举反例' 相比纯正例有什么优势？\n")

    if "negative_examples" in styles:
        ne = styles["negative_examples"]
        print(f"  negative_examples 的通过率：{ne['overall_rate']:.0%}")
        print()

        # 检查字段准确率
        fa = ne.get("field_accuracy", {})
        weak_field = min(fa.items(), key=lambda x: x[1]) if fa else ("unknown", 0)

        print(f"  最弱字段：{weak_field[0]}（{weak_field[1]:.0%}）")
        print()
        print("  反例策略的理论优势：")
        print("    - 直接标记'容易搞混的边界'，相当于给 LLM 画了警戒线")
        print("    - 6 个陷阱覆盖了最常见的误判模式")
        print("    - 每个反例都有'为什么错'的解释，帮助 LLM 理解原则")
        print()
        print("  反例策略的潜在问题：")
        print("    - 反例太多（6 个）可能让 LLM'过度警觉'")
        print("    - 某些反例可能与其他规则产生冲突")
        print("    - 如果 LLM 的记忆窗口有限，反例可能挤占规则的空间")

    # ── 分析 3：推理模板的有效性 ──
    print("\n[分析] 分析 3：'先想再说' 真的更准确吗？\n")

    if "reasoning_template" in styles:
        rt = styles["reasoning_template"]

        if "explain_why" in styles:
            ew = styles["explain_why"]
            diff = rt["overall_rate"] - ew["overall_rate"]
            print(
                f"  reasoning_template vs explain_why："
                f"{rt['overall_rate']:.0%} vs {ew['overall_rate']:.0%}"
                f"（差距 {diff:+.0%}）"
            )

        print()
        print("  Chain-of-Thought（思维链）的原理：")
        print("    1. LLM 逐 token 生成文本，前面的 token 影响后面")
        print("    2. 如果第一步就是结论，后续 token 会围绕结论展开")
        print("       -> 忽略与结论不一致的信号（确认偏误）")
        print("    3. 如果第一步是'列出所有信号'，所有信息都有机会被关注")
        print("       -> 更全面的判断")
        print()
        print("  推理模板适用场景：")
        print("    [OK] 需要多步骤判断的任务（如本任务：信号->分类->优先级->情感）")
        print("    [OK] 有明确判断标准的任务")
        print("    [X] 简单的单步任务（增加推理步骤反而浪费 token）")

    # ── 分析 4：总结 —— 哪种设计哲学最适合这个任务？ ──
    print("\n[分析] 总结：三种设计哲学的适用场景\n")

    print("  ┌──────────────────┬──────────────────┬──────────────────┐")
    print("  │ 设计哲学          │ 最适合的场景      │ 最不适合的场景    │")
    print("  ├──────────────────┼──────────────────┼──────────────────┤")
    print("  │ explain_why      │ 需要主观判断的任务 │ 纯规则性任务      │")
    print("  │                  │ 边界模糊的分类    │ 简单的是/否判断   │")
    print("  ├──────────────────┼──────────────────┼──────────────────┤")
    print("  │ negative_examples│ 类别容易混淆的任务 │ 类别定义清晰的    │")
    print("  │                  │ 常见错误模式固定  │ 错误模式多变      │")
    print("  ├──────────────────┼──────────────────┼──────────────────┤")
    print("  │ reasoning_template│ 多步骤推理任务   │ 单步简单任务      │")
    print("  │                  │ 综合判断类任务    │ 已有很多规则的    │")
    print("  └──────────────────┴──────────────────┴──────────────────┘")

    print()
    print("  [TIP] 实战建议：最佳 prompt 通常不是单一哲学，而是组合拳。")
    print("     比如：explain_why（让 LLM 理解意图）+ negative_examples（防误判）")
    print("          + reasoning_template（保证推理质量）三者结合。")
    print()
    print("     但组合时要注意 token 预算——prompt 太长会挤占 LLM 的注意力。")
    print("     一般来说，system prompt 控制在 500-1500 字之间效果最好。")


# ═══════════════════════════════════════════════════════════════
# ④ 主流程
# ═══════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="W2D6：System Prompt 设计实战 — 3 种设计哲学对比评测",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python src/study_agent/w2d6_demo.py                     # 完整评测
  python src/study_agent/w2d6_demo.py --quick             # 快速模式
  python src/study_agent/w2d6_demo.py --compare-with-w2d5 # 与W2D5对比
  python src/study_agent/w2d6_demo.py --export-csv r.csv  # 导出CSV
        """,
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="快速模式：只跑前 10 条用例（约 30 秒）",
    )
    parser.add_argument(
        "--compare-with-w2d5",
        action="store_true",
        help="加入 W2D5 的 fewshot 版本作为对照组",
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
        help="导出详细结果到 CSV 文件",
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

    # ── W2D6 的三种新设计 ──
    styles: dict = {
        "explain_why": build_explain_why_prompt,
        "negative_examples": build_negative_examples_prompt,
        "reasoning_template": build_reasoning_template_prompt,
    }

    # ── 可选：加入 W2D5 对照组 ──
    if args.compare_with_w2d5:
        styles["w2d5_fewshot (baseline)"] = build_w2d5_fewshot_prompt
        logger.info("已加入 W2D5 fewshot 作为对照组")

    # ── 创建评测器 ──
    extractor = StructuredExtractor(client)
    evaluator = PromptEvaluator(client, extractor, CLASSIFY_SCHEMA_DEF)

    # ── 跑评测 ──
    total_calls = len(cases) * len(styles)
    print(f"\n{'='*60}")
    print("W2D6 System Prompt 设计实战评测")
    print(f"{'='*60}")
    print(f"测试用例: {len(cases)} 条")
    print(f"Prompt 风格: {len(styles)} 种")
    print(f"总调用次数: {total_calls}")
    print(f"预计耗时: ~{total_calls * 1.5 // 60} 分钟")
    print(f"{'='*60}\n")

    t_start = time.time()
    results = evaluator.run_batch(cases, styles)
    elapsed = time.time() - t_start
    print(f"\n评测完成，耗时 {elapsed:.0f} 秒\n")

    # ── 输出报告 ──
    report = evaluator.report(results)
    print(report)

    # ── 输出设计哲学分析 ──
    summary = evaluator.summarize(results)
    print_design_analysis(summary)

    # ── 详细失败分析 ──
    print("\n" + "=" * 60)
    print("  失败案例详解（前 10 个）")
    print("=" * 60)
    print()
    print("以下列出具体的失败案例，帮助你理解'哪种 prompt 在哪种 case 上栽了'：\n")

    failures = [r for r in results if not r.overall_pass]
    # 构建 case_id -> text 的映射
    case_map = {case.id: case.text for case in cases}
    for f in failures[:10]:
        print(f"  [{f.case_id}] style={f.style}")
        print(f"    输入: {case_map.get(f.case_id, '?')[:80]}...")
        print(f"    错误: {'; '.join(f.errors) if f.errors else '字段不匹配'}")
        if f.parsed:
            print(f"    实际输出: {f.parsed}")
        print()

    if len(failures) > 10:
        print(f"  ... 还有 {len(failures) - 10} 条失败，使用 --export-csv 查看完整列表\n")

    # ── 导出 CSV ──
    if args.export_csv:
        csv_content = evaluator.export_csv(results)
        with open(args.export_csv, "w", encoding="utf-8") as f:
            f.write(csv_content)
        logger.info("详细结果已导出到: %s", args.export_csv)

    # ── 快速模式提示 ──
    if args.quick:
        print("\n---")
        print("这是快速模式（10 条用例）。完整评测请运行:")
        print("  python src/study_agent/w2d6_demo.py")
        print()
        print("[TIP] 建议：先用 --quick 快速验证你的 prompt 修改，")
        print("   确认方向对了再跑完整的 50 条评测。")


if __name__ == "__main__":
    main()
