"""Week 2 Day 4 演示 —— Tool Calling 循环实战

演示场景：
  1. 简单工具调用（1轮）：查今天日期
  2. 多步推理（2-3轮）：先查日期，再算3天后，再算相隔天数
  3. 数学计算（1轮）：安全计算器
  4. 文本处理（1-2轮）：统计 + 转换
  5. 跨工具协作（2-3轮）：查日期 + 用日期数字做计算

每个场景展示完整的"LLM决策→工具执行→结果回传"循环。

运行方式：
  python src/study_agent/w2d4_demo.py

  只跑指定场景：
  python src/study_agent/w2d4_demo.py --scene 1
"""

from __future__ import annotations

import logging
import sys
import time

from study_agent.llm.client import LLMClient
from study_agent.tools.builtin_tools import CalculatorTool, DateTimeTool, TextStatsTool
from study_agent.tools.tool_loop import ToolCallLoop

# 显示 INFO 级别日志，让你看到每轮 LLM 调了什么工具
logging.basicConfig(
    level=logging.INFO,
    format="  [%(levelname)s] %(message)s",
)

# ═══════════════════════════════════════════════════════════
# ① 场景定义
# ═══════════════════════════════════════════════════════════

SCENES = [
    {
        "title": "单工具调用 —— 查今天日期",
        "task": "今天是几号？星期几？",
        "system": "你是一个有帮助的助手。当需要日期信息时，请调用 datetime 工具。",
        "expected_tools": ["datetime"],
        "expected_rounds": 1,
    },
    {
        "title": "多步推理 —— 日期计算",
        "task": "今天是几号？3天后是几号？这两个日期相差几天？",
        "system": "你是一个有帮助的助手。需要日期信息时请调用 datetime 工具，需要数学计算时请调用 calculator 工具。",
        "expected_tools": ["datetime", "calculator"],
        "expected_rounds": 2,
    },
    {
        "title": "安全计算器 —— 多步数学",
        "task": "计算 (15*8 + 27)/3 的结果，然后计算这个结果的平方。",
        "system": "你是一个数学助手。需要进行数学计算时请调用 calculator 工具。一步一步算。",
        "expected_tools": ["calculator"],
        "expected_rounds": 2,
    },
    {
        "title": "文本处理 —— 统计+转换",
        "task": '分析 "Hello World, AI Agent!" 这句话：统计它的字数和词数，然后把它反转，再把反转结果转成大写。',
        "system": "你是一个文本处理助手。需要分析或转换文本时请调用 text_stats 工具。一次只调用一个工具，根据结果决定下一步。",
        "expected_tools": ["text_stats"],
        "expected_rounds": 3,
    },
    {
        "title": "跨工具协作 —— 日期 + 计算",
        "task": "查一下今天的日期，然后把今天的日期数字（几号）乘以100，告诉我结果。",
        "system": "你是一个有帮助的助手。查日期用 datetime，做计算用 calculator。先查日期，拿到日期数字后再计算。",
        "expected_tools": ["datetime", "calculator"],
        "expected_rounds": 2,
    },
]


# ═══════════════════════════════════════════════════════════
# ② 主演示
# ═══════════════════════════════════════════════════════════


def print_sep(title: str) -> None:
    print(f"\n{'='*65}")
    print(f"  {title}")
    print("=" * 65)


def run_scene(loop: ToolCallLoop, scene: dict, idx: int) -> None:
    """跑一个场景并打印结果。"""
    print_sep(f"场景 {idx}: {scene['title']}")
    print(f"\n  用户任务：{scene['task']}")
    print(f"  期望工具：{', '.join(scene['expected_tools'])}")
    print(f"  期望轮次：~{scene['expected_rounds']} 轮")
    print("\n  --- 循环开始 ---")

    start = time.time()
    try:
        answer = loop.run(scene["task"], system=scene.get("system"))
        elapsed = time.time() - start
        print(f"\n  --- 循环结束（耗时 {elapsed:.1f}s）---")
        print(f"\n  [OK] 最终回答：\n{answer}")
    except Exception as e:
        elapsed = time.time() - start
        print(f"\n  --- 异常中断（耗时 {elapsed:.1f}s）---")
        print(f"\n  [ERROR] 错误：{type(e).__name__}: {e}")


def main() -> None:
    # 解析参数
    scene_filter: int | None = None
    if "--scene" in sys.argv:
        idx = sys.argv.index("--scene")
        if idx + 1 < len(sys.argv):
            scene_filter = int(sys.argv[idx + 1])

    print_sep("Week 2 Day 4 — Tool Calling 循环演示")
    print("\n可用工具：calculator（安全计算器）、datetime（日期时间）、text_stats（文本统计）")
    print("每个场景展示 LLM 如何自主决定：用哪个工具 → 看结果 → 是否需要再调工具\n")

    # 创建 client 和 loop
    client = LLMClient.from_env()
    tools = [CalculatorTool(), DateTimeTool(), TextStatsTool()]
    loop = ToolCallLoop(client, tools=tools, max_rounds=5)

    # 先打印工具定义（让用户看到 LLM "看"到了什么）
    print("LLM 看到的工具定义：")
    for tool in tools:
        d = tool.definition
        print(f"  [tool] {d.name}: {d.description.split('。')[0]}")
    print()

    # 跑场景
    for i, scene in enumerate(SCENES, 1):
        if scene_filter is not None and i != scene_filter:
            continue
        run_scene(loop, scene, i)

    print_sep("演示结束")


if __name__ == "__main__":
    main()
