"""Day 5 自我评估自查脚本 —— 逐条验证你是否真的掌握了。

每一条对应日志里的一个自评 checkbox。
运行方式：poetry run python src/study_agent/day5_self_check.py
"""

from __future__ import annotations

import sys

sys.stdout.reconfigure(encoding="utf-8")


def check_1_exponential_backoff():
    """第1条：我能解释什么是指数退避，以及为什么不用固定间隔重试。

    问题：100 个请求同时失败，大家都固定等 2 秒→同时重试→又同时失败→死循环。
    这叫做"惊群效应"（Thundering Herd）。

    指数退避让每次等待时间翻倍：1, 2, 4, 8, 16...
    不同请求开始时间不同，重试自然散开，不给服务器雪上加霜。

    自问：multiplier=1 和 multiplier=3 有什么区别？
    """
    print("=" * 50)
    print("第1条：指数退避的等待序列")
    print("=" * 50)

    for multiplier in [1, 3]:
        print(f"\nmultiplier = {multiplier}:")
        for attempt in range(5):
            wait = multiplier * (2**attempt)
            print(f"  第 {attempt+1} 次重试前等 {wait} 秒")


def check_2_http_codes():
    """第2条：我能说出 HTTP 401/429/500/503 各自代表什么以及哪些该重试。

    自问：以下每个错误码是什么意思？该重试吗？为什么？
    """
    print("\n" + "=" * 50)
    print("第2条：HTTP 错误码速查")
    print("=" * 50)

    codes = [
        (401, "Unauthorized", "Key 错了/没传 Key", "NO", "重试 100 次 Key 也不会变对"),
        (429, "Too Many Requests", "请求太快被限流", "YES", "等一会儿配额恢复就好了"),
        (500, "Internal Server Error", "服务器内部 bug", "YES", "服务器可能马上恢复"),
        (503, "Service Unavailable", "服务器过载/维护中", "YES", "临时状态，等下就好"),
        (400, "Bad Request", "请求格式错误", "NO", "你的请求有问题，重试还是错"),
    ]
    for code, name, meaning, retry, reason in codes:
        print(f"  HTTP {code} ({name})")
        print(f"    含义: {meaning}")
        print(f"    重试: {retry}")
        print(f"    理由: {reason}")
        print()


def check_3_tenacity_three_params():
    """第3条：我能用 tenacity 写出包含 wait、stop、retry 三个配置的 @retry 装饰器。

    自问：下面这个装饰器是什么意思？每一行在干什么？
    """
    print("=" * 50)
    print("第3条：tenacity 三要素")
    print("=" * 50)

    import logging

    from openai import APIConnectionError
    from tenacity import (
        before_sleep_log,
        retry,
        retry_if_exception_type,
        stop_after_attempt,
        wait_exponential,
    )

    test_logger = logging.getLogger("self_check")

    @retry(
        # ── wait：两次重试之间等多久 ──
        wait=wait_exponential(multiplier=1, min=1, max=30),
        # multiplier=1 → 基础单位 1 秒
        # min=1 → 最少等 1 秒
        # max=30 → 最多等 30 秒（不会无限增长）
        # ── stop：什么时候停止重试 ──
        stop=stop_after_attempt(3),
        # 总共尝试 3 次（包含第一次），即最多重试 2 次
        # ── retry：什么情况下才重试 ──
        retry=retry_if_exception_type(APIConnectionError),
        # 只有网络连接错误才重试。Key 错了不重试。
        before_sleep=before_sleep_log(test_logger, logging.WARNING),
    )
    def demo():
        pass  # 这里只是展示装饰器语法

    print(
        """
    装饰器结构：
      @retry(
          wait=...,    ← 等多久？（1s, 2s, 4s... 最长30s）
          stop=...,    ← 试几次？（最多3次）
          retry=...,   ← 什么错才试？（只试网络错误）
      )

    三个参数缺一不可：
      - 没有 wait → 立刻重试，打爆服务器
      - 没有 stop → 无限重试，程序卡死
      - 没有 retry → 什么错都重试，Key 错了也重试 3 次（浪费）
    """
    )


def check_4_retry_if_exception_vs_type():
    """第4条：retry_if_exception_type 和 retry_if_exception 的区别。

    自问：为什么第二个比第一个更安全？
    """
    print("=" * 50)
    print("第4条：两种重试判断方式对比")
    print("=" * 50)

    print(
        """
    ┌─────────────────────────────────────────────────────────┐
    │ retry_if_exception_type(APIStatusError)                 │
    │                                                         │
    │   按"类型"匹配——APIStatusError 及其所有子类都重试。     │
    │                                                         │
    │   问题：AuthenticationError 是 APIStatusError 的子类！  │
    │         HTTP 401 也会被重试——纯浪费。                   │
    │                                                         │
    │   APIStatusError                                        │
    │   ├── AuthenticationError (401) ← 不该重试但被误伤了！  │
    │   ├── RateLimitError (429)      ← 该重试                │
    │   └── InternalServerError (500) ← 该重试                │
    └─────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────┐
    │ retry_if_exception(_should_retry)                       │
    │                                                         │
    │   按"自定义函数"匹配——每个异常都经过你写的判断逻辑。   │
    │                                                         │
    │   def _should_retry(exc):                               │
    │       if isinstance(exc, AuthenticationError):          │
    │           return False  ← 明确排除！                    │
    │       if isinstance(exc, (RateLimitError, ...)):        │
    │           return True   ← 只对临时性错误重试            │
    │                                                         │
    │   优势：你完全控制"什么该重试"，不会被继承关系坑。     │
    └─────────────────────────────────────────────────────────┘

    一句话总结：
      retry_if_exception_type = "这类异常统统重试"（粗粒度）
      retry_if_exception =        "我来逐个判断"（精细粒度）
      后者更安全，因为它不会因为继承关系而误伤。
    """
    )


def check_5_mock():
    """第5条：Mock 测试的原理。

    自问：为什么要用假错误代替真错误？
    """
    print("=" * 50)
    print("第5条：Mock 测试原理")
    print("=" * 50)

    from unittest.mock import Mock

    # ── 不用 Mock 的测试（不可行）──
    print(
        """
    ❌ 不加 Mock 的测试：
       def test_network_error():
           # 真的要拔网线？真的要等 30 秒超时？
           # 每次跑测试都不一样（有时候网络好，有时候不好）
           client.chat.completions.create(...)

    ✅ 加 Mock 的测试：
    """
    )

    # ── 演示 Mock ──
    fake_client = Mock()
    # 告诉 Mock："当你调用 create 时，抛出连接错误"
    import httpx
    from openai import APIConnectionError

    fake_request = httpx.Request("POST", "https://fake.api/v1/chat")
    fake_client.chat.completions.create.side_effect = APIConnectionError(
        message="Network error",
        request=fake_request,
    )

    print("  fake_client = Mock()")
    print("  fake_client.chat.completions.create.side_effect = APIConnectionError(...)")
    print()
    print("  现在每次调用 fake_client.chat.completions.create()")
    print("  都会稳定地抛出 APIConnectionError——可重复、可预测。")

    # 验证
    try:
        fake_client.chat.completions.create(model="test", messages=[])
        print("  [FAIL] 没有抛异常")
    except APIConnectionError as e:
        print(f"  [OK] 捕获到: {e.message}")

    print(
        """
    Mock 的核心思想：
      真环境是不可控的（网速、服务器状态、时间）。
      Mock 造一个"假的但可预测的"环境。
      跑 100 次结果都一样 → 测试是可靠的。
    """
    )


def check_6_stream_recovery():
    """第6条：Stream 中断恢复的核心思路。

    自问：收到一半断了，怎么不从零开始？
    """
    print("=" * 50)
    print("第6条：Stream 中断恢复")
    print("=" * 50)

    print(
        """
    假设你在用流式（打字机效果）接收 LLM 的回复：

      收到: "机器学习"
      收到: "是人工智能"
      收到: "的一个分支"
      ── 网络断了！──

    这时 accumulated = "机器学习是人工智能的一个分支"

    ── 重连后 ──

    把已收内容附在新请求里：
      user_message = "...你之前回复的内容是：
        机器学习是人工智能的一个分支
        请从断点处继续，不要重复。"

    LLM 继续：
      收到: "，它让计算机从数据中学习。"

    最终结果: "机器学习是人工智能的一个分支，它让计算机从数据中学习。"

    ═══════════════════════════════════════════
    核心三步：
      1. 每收到一个 chunk → 立刻存进 accumulated
      2. 中断了 → 记住 accumulated 的值
      3. 重连时 → 把 accumulated 附在新请求里，让 LLM 续写
    ═══════════════════════════════════════════

    关键：如果不存 accumulated，断了就全丢了，只能从头开始。
    这和"文档编辑器自动保存"是同一个道理。
    """
    )


def check_7_auth_error_subclass_trap():
    """第7条：AuthenticationError 继承自 APIStatusError。

    自问：为什么这会坑人？
    """
    print("=" * 50)
    print("第7条：AuthenticationError 的继承陷阱")
    print("=" * 50)

    from openai import APIStatusError, AuthenticationError

    # 用代码证明
    print()
    print("  验证：AuthenticationError 是 APIStatusError 的子类吗？")
    result = issubclass(AuthenticationError, APIStatusError)
    print(f"  → {result}")

    print(
        """
    ═══════════════════════════════════════════════════════
    这就是陷阱：

    APIStatusError                        ← "所有 HTTP 错误"
    ├── AuthenticationError (401)         ← "你没权限"
    ├── RateLimitError (429)             ← "你太快了"
    └── InternalServerError (500)        ← "服务器挂了"

    如果你写：
      retry=retry_if_exception_type(APIStatusError)

    那么 401、429、500 全都会重试！
    但 401（Key 错了）重试 100 次也不会突然变对。
    ═══════════════════════════════════════════════════════

    解决方案（这就是我们 Day 5 的做法）：
      不用 retry_if_exception_type
      改用 retry_if_exception + 自定义判断函数
      在函数里显式地写：
        isinstance(exc, AuthenticationError) → return False
    """
    )


# ==================== 主程序 ====================
if __name__ == "__main__":
    print("Day 5 自我评估——逐条自查")
    print()
    print("每看完一条，问自己：我能用自己的话讲出来吗？")
    print("如果可以 → 那条可以打勾 ✅")
    print("如果不行 → 把那一节的代码跑一遍，或者来问我")
    print()

    check_1_exponential_backoff()
    check_2_http_codes()
    check_3_tenacity_three_params()
    check_4_retry_if_exception_vs_type()
    check_5_mock()
    check_6_stream_recovery()
    check_7_auth_error_subclass_trap()

    print("=" * 50)
    print("自查完成！")
    print("=" * 50)
    print()
    print("现在打开 docs/week1/day5-error-handling.txt")
    print("逐条检查自我评估——能脱口而出的打勾 ✅")
    print("还需要再看看的，把这条来问我。")
