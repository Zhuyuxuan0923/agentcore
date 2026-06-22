"""Week 2 Day 3 演示 —— 结构化输出三种方案对比

演示流程：
  1. 定义"产品信息提取"Schema（6 个字段）
  2. 准备 10 条真实产品描述文本
  3. 对每条文本用三种方案（+ 基线）分别提取
  4. 检查每个结果：JSON 合法？必填字段齐全？类型正确？
  5. 输出对比报告

运行方式：
  python src/study_agent/w2d3_demo.py

  可以指定 provider：
  $env:LLM_PROVIDER="deepseek"
  python src/study_agent/w2d3_demo.py

  也可以只测某一个方案：
  python src/study_agent/w2d3_demo.py --method json_mode
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Any

from study_agent.llm.client import LLMClient
from study_agent.llm.structured import ExtractionSchema, StructuredExtractor

# ═══════════════════════════════════════════════════════════════
# ① 定义提取任务 —— 从产品描述中提取结构化信息
# ═══════════════════════════════════════════════════════════════

PRODUCT_SCHEMA = ExtractionSchema(
    name="extract_product_info",
    description="从产品描述文本中提取产品的结构化信息",
    properties={
        "product_name": {
            "type": "string",
            "description": "产品名称",
        },
        "category": {
            "type": "string",
            "description": "产品类别（如：笔记工具、代码编辑器、健身应用等）",
        },
        "features": {
            "type": "array",
            "items": {"type": "string"},
            "description": "核心功能列表",
        },
        "target_users": {
            "type": "array",
            "items": {"type": "string"},
            "description": "目标用户群体",
        },
        "pricing": {
            "type": "object",
            "properties": {
                "monthly": {"type": "number", "description": "月费（元）"},
                "annual": {"type": "number", "description": "年费（元）"},
            },
            "description": "价格信息",
        },
        "competitors": {
            "type": "array",
            "items": {"type": "string"},
            "description": "主要竞品名称",
        },
    },
    required=["product_name", "category", "features", "target_users"],
)


# ═══════════════════════════════════════════════════════════════
# ② 测试数据集 —— 10 条真实产品描述
# ═══════════════════════════════════════════════════════════════

TEST_CASES = [
    # Case 1：笔记工具
    "小记灵是一款面向个人创作者的AI笔记助手，支持语音转文字、智能摘要、自动标签分类三大核心功能。"
    "产品定价为月费29元、年费199元，主要竞品包括Notion AI和印象笔记。"
    "目标用户是自媒体作者、学生和知识工作者。",
    # Case 2：健身应用
    "FitTrack Pro是一个专注于力量训练的AI健身教练应用。它能根据用户的体能自动生成训练计划、"
    "实时纠正动作姿势、追踪力量增长曲线。月费39元，年费299元。同类产品有Keep和训记。"
    "面向健身入门者和希望突破瓶颈的中级训练者。",
    # Case 3：代码工具
    "CodeBuddy是面向全栈开发者的AI编程助手，基于GPT-4o深度定制。核心功能包括：智能代码补全、"
    "跨文件上下文感知、一键生成单元测试、PR代码审查。采用免费增值模式，基础版免费，"
    "专业版月费15美元。主要竞争对手是GitHub Copilot和Cursor。",
    # Case 4：阅读应用
    "ReadWise是一款通过AI技术帮助用户高效阅读和记忆的阅读助手。它具备智能摘要生成、"
    "间隔重复复习提醒、跨平台高亮同步、PDF和EPUB双格式支持等功能。价格方案为月费19元、"
    "年费149元。面向学生、研究人员和需要大量阅读的职场人士。竞品包括MarginNote和LiquidText。",
    # Case 5：设计工具
    "Canva AI助手集成在Canva平台中，提供AI文案生成、智能抠图、设计建议、品牌色彩推荐和"
    "批量模板生成功能。作为Canva的增值功能，需要Canva Pro订阅（月费12美元）。"
    "目标用户是社交媒体运营、小企业主和营销人员。竞争对手包括Adobe Express和稿定设计。",
    # Case 6：语言学习
    "LinguaBot是一款基于GPT-4的AI外语口语陪练。核心功能包括实时语音对话、语法错误即时纠正、"
    "情景对话模拟（餐厅、机场、面试等）、个性化学习路径推荐。月费49元，年费349元。"
    "主要服务想提高口语能力的英语学习者和留学准备者。竞品有Speak和多邻国。",
    # Case 7：项目管理
    "飞书项目助手是基于飞书平台的AI项目管理工具。支持自动生成会议纪要、任务智能分配、"
    "风险预警、进度日报自动推送和甘特图一键生成。作为飞书企业版的增值模块，"
    "按团队人数计费，10人团队月费299元。面向中小企业的项目经理和团队Leader。"
    "竞品包括钉钉宜搭和Teambition。",
    # Case 8：炒股工具
    "StockMind是面向个人投资者的AI投资研究平台。支持A股和港股的财报智能解读、"
    "舆情监控、技术面自动分析、选股策略回测和持仓风险评估。月费99元，年费799元。"
    "目标用户是中高级个人投资者和理财顾问。竞品包括同花顺AI和富途牛牛的智能分析。",
    # Case 9：写作工具
    "笔灵AI是为中文内容创作者打造的全场景AI写作工具。功能覆盖公众号文章生成、小红书文案、"
    "知乎问答、短视频脚本、广告文案和SEO文章优化。内置30+写作模板。月费39元，年费299元。"
    "面向自媒体作者、电商运营和中小企业市场人员。竞品有Jasper AI和秘塔写作猫。",
    # Case 10：医疗健康
    "HealthGuard AI是面向慢性病患者的健康管理应用。核心功能包括用药智能提醒、"
    "饮食营养分析（拍照识别）、健康指标趋势追踪、在线问诊预约和个性化健康报告生成。"
    "月费59元，年费399元，提供7天免费试用。目标用户是糖尿病和高血压等慢性病患者。"
    "竞品包括丁香医生和微医。",
]


# ═══════════════════════════════════════════════════════════════
# ③ 结果评估
# ═══════════════════════════════════════════════════════════════


@dataclass
class EvalResult:
    """单次提取的评估结果。"""

    json_valid: bool  # JSON 是否合法
    required_fields_ok: bool  # 必填字段是否齐全
    extra_fields: list[str]  # 多出的字段
    raw_result: dict | None  # 提取结果（成功时）


def evaluate_result(result: dict[str, Any] | None, schema: ExtractionSchema) -> EvalResult:
    """评估一次提取结果的質量。"""
    if result is None:
        return EvalResult(
            json_valid=False,
            required_fields_ok=False,
            extra_fields=[],
            raw_result=None,
        )

    # 检查必填字段
    missing = [f for f in schema.required if f not in result or result[f] is None]
    required_fields_ok = len(missing) == 0

    # 检查多余字段（Schema 里没定义的字段）
    all_defined = set(schema.properties.keys())
    extra_fields = [f for f in result if f not in all_defined]

    return EvalResult(
        json_valid=True,
        required_fields_ok=required_fields_ok,
        extra_fields=extra_fields,
        raw_result=result,
    )


# ═══════════════════════════════════════════════════════════════
# ④ 主演示逻辑
# ═══════════════════════════════════════════════════════════════


def print_header(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print("=" * 70)


def run_benchmark(provider: str, methods: list[str]) -> dict[str, list[EvalResult]]:
    """对指定 provider 用所有方法跑全部测试用例，返回各方法的评估结果列表。"""
    print(f"\n📡 Provider: {provider}")
    try:
        client = LLMClient(provider=provider)
    except ValueError as e:
        print(f"  ⚠️ 跳过：{e}")
        return {}

    extractor = StructuredExtractor(client)
    available = extractor.available_methods()
    methods_to_test = [m for m in methods if m in available]

    if not methods_to_test:
        print("  无可用的提取方法")
        return {}

    print(f"  可用方法: {', '.join(available)}")
    print(f"  本次测试: {', '.join(methods_to_test)}")
    print(f"  模型: {client.model}")

    results: dict[str, list[EvalResult]] = {}

    for method in methods_to_test:
        print(f"\n  🔧 运行 {method}...")
        method_results: list[EvalResult] = []

        for i, test_text in enumerate(TEST_CASES, 1):
            try:
                extracted = extractor.extract(test_text, PRODUCT_SCHEMA, method=method)
                ev = evaluate_result(extracted, PRODUCT_SCHEMA)
                method_results.append(ev)

                status = "✅" if ev.json_valid and ev.required_fields_ok else "❌"
                missing_info = ""
                if not ev.required_fields_ok and ev.raw_result:
                    missing = [f for f in PRODUCT_SCHEMA.required if f not in ev.raw_result]
                    missing_info = f" 缺字段: {missing}"
                print(f"    Case {i:2d} {status}{missing_info}")
            except Exception as e:
                method_results.append(
                    EvalResult(
                        json_valid=False, required_fields_ok=False, extra_fields=[], raw_result=None
                    )
                )
                print(f"    Case {i:2d} ❌ 异常: {e}")

        results[method] = method_results

    return results


def print_summary(all_results: dict[str, dict[str, list[EvalResult]]]) -> None:
    """打印所有 provider + 方法的汇总对比表。"""
    print_header("📊 汇总对比")

    for provider, methods in all_results.items():
        if not methods:
            continue
        print(f"\n  Provider: {provider}")
        print(f"  {'方法':<16} {'JSON合法':<10} {'必填齐全':<10} {'成功率':<10}")
        print(f"  {'-'*16} {'-'*10} {'-'*10} {'-'*10}")

        for method, results in methods.items():
            total = len(results)
            json_ok = sum(1 for r in results if r.json_valid)
            required_ok = sum(1 for r in results if r.required_fields_ok)
            rate = f"{required_ok}/{total} ({required_ok/total*100:.0f}%)" if total > 0 else "N/A"

            print(f"  {method:<16} {json_ok}/{total:<9} {required_ok}/{total:<9} {rate:<10}")


def main() -> None:
    # 解析命令行参数
    methods = ["prompt_only", "json_mode", "function_call", "tool_use"]
    if "--method" in sys.argv:
        idx = sys.argv.index("--method")
        if idx + 1 < len(sys.argv):
            methods = [sys.argv[idx + 1]]

    print_header("🧪 Week 2 Day 3 — 结构化输出方案对比")
    print("\n任务：从产品描述中提取 6 个字段的结构化信息")
    print(f"测试用例数：{len(TEST_CASES)}")
    print("\nSchema 字段：")
    for name, info in PRODUCT_SCHEMA.properties.items():
        req = "（必填）" if name in PRODUCT_SCHEMA.required else "（选填）"
        print(f"  - {name}: {info['type']} {req} — {info.get('description', '')}")

    # 对每个可用的 provider 跑测试
    from study_agent.config.settings import list_available_providers

    available_providers = list_available_providers()
    if not available_providers:
        print("\n⚠️ 没有检测到任何可用的 Provider！")
        print("请先在 .env 文件中设置至少一个 API Key")
        print("例如: DEEPSEEK_API_KEY=your-key")
        return

    print(f"\n可用的 Provider: {', '.join(available_providers)}")

    all_results: dict[str, dict[str, list[EvalResult]]] = {}

    for provider in available_providers:
        results = run_benchmark(provider, methods)
        if results:
            all_results[provider] = results

    # 打印汇总
    if all_results:
        print_summary(all_results)

        # 展示一个成功例子
        print_header("📝 成功提取示例")
        for provider, methods in all_results.items():
            for method, results in methods.items():
                for r in results:
                    if r.json_valid and r.required_fields_ok and r.raw_result:
                        print(f"\n  Provider: {provider} | 方法: {method}")
                        print(f"  {json.dumps(r.raw_result, ensure_ascii=False, indent=2)}")
                        return  # 只展示第一个成功的
    else:
        print("\n❌ 所有 provider 都失败了，请检查 API Key 和网络连接")


if __name__ == "__main__":
    main()
