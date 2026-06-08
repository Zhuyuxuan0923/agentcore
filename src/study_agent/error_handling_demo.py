"""Week 1 Day 5 — 异常处理与重试机制。

🤖 AI 生成骨架与演示代码，✍️ 你需要动手跑每个示例并回答末尾的思考题。

═══════════════════════════════════════════════════════
📖 在写代码之前，先用生活场景理解三个核心概念：
═══════════════════════════════════════════════════════

1️⃣ 异常（Exception）= 程序运行中发生的"意外"
   生活类比：你开车去上班，路被封了 → 这是"路被封异常"
   程序类比：你调用 API，网络断了 → 这是"网络异常"
   关键认知：异常不是你的代码写错了，而是"外部世界不配合"

2️⃣ 重试（Retry）= 失败了再试一次
   生活类比：电话打不通 → 过一会儿再拨
   程序类比：API 返回错误 → 等几秒再调一次
   关键认知：不是所有失败都值得重试（密码错误重试没用，网络抖动重试有用）

3️⃣ 指数退避（Exponential Backoff）= 每次重试等更久
   生活类比：敲门 → 等 1 秒 → 敲重一点 → 等 2 秒 → 再敲 → 等 4 秒...
   程序类比：请求失败 → 等 1s 重试 → 又失败 → 等 2s 重试 → 等 4s → 等 8s...
   关键认知：如果服务器已经在过载，你越重试它越炸。慢慢来，给服务器喘息时间。

═══════════════════════════════════════════════════════
API 调用最常见的 5 种失败类型：
═══════════════════════════════════════════════════════

| 错误类型              | HTTP 码 | 原因                     | 该重试吗？     |
|-----------------------|---------|--------------------------|----------------|
| AuthenticationError   | 401     | API Key 错误             | ❌ 重试没用    |
| RateLimitError        | 429     | 请求太频繁，被限流       | ✅ 等一会儿    |
| APIConnectionError    | 无      | 网络断了                 | ✅ 等网络恢复  |
| APITimeoutError       | 无      | 请求超时（模型思考太久） | ✅ 可以试      |
| InternalServerError   | 500+    | API 服务本身挂了         | ✅ 短暂等待    |

关键规律：只有"临时性"的失败才值得重试。永久性的失败重试是浪费时间。
"""

from __future__ import annotations

import logging
import os
import sys

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_fixed,
)

sys.stdout.reconfigure(encoding="utf-8")

# 为演示文件创建一个专用 logger
_demo_logger = logging.getLogger("error_handling_demo")
logging.basicConfig(level=logging.WARNING, format="[%(levelname)s] %(message)s")


# ═══════════════════════════════════════════════════════════════
# 第一部分：没有错误处理 → 程序直接炸
# ═══════════════════════════════════════════════════════════════
def demo_pain_no_handling():
    """❌ 反例：调用 API 但不做任何错误处理。

    这就是 Day 4 里 llm_client.py 的现状——
    API Key 错了？炸。网络断了？炸。被限流了？炸。
    用户看到的是冰冷的 traceback，不是友好的提示。
    """
    print("\n" + "=" * 60)
    print("第一部分：没有错误处理的世界")
    print("=" * 60)

    from openai import OpenAI

    # 故意用假 Key —— 模拟"配置错误"场景
    bad_client = OpenAI(
        api_key="sk-this-is-a-fake-key-12345",
        base_url="https://api.deepseek.com",
    )

    print("\n[场景] 用错误的 API Key 调用模型...")
    print("[预期] 程序直接崩溃，甩给你一屏幕红字\n")

    # 这里没有任何 try/except，异常会直接穿透，程序终止
    bad_client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": "hello"}],
    )

    # 这行永远执行不到，因为上一行已经炸了
    print("如果看到这行，说明没炸——但这不可能。")


# ═══════════════════════════════════════════════════════════════
# 第二部分：try/except — 给程序装个安全气囊
# ═══════════════════════════════════════════════════════════════
#
# try/except 的思维模型：
#   try 块 = "你试试这段代码，可能出问题"
#   except 块 = "如果出了某种问题，按这个方案处理"
#
# 语法结构：
#   try:
#       可能出错的代码
#   except 具体错误类型 as e:
#       出这种错时怎么办
#   except 另一种错误类型 as e:
#       出那种错时怎么办


def demo_basic_try_except():
    """✅ 基础方案：用 try/except 捕获最常见的两类错误。

    这里演示捕获两大类异常：
    1. openai 自己定义的异常（API 返回的错误）
    2. 网络层面的异常（根本连不上服务器）
    """
    print("\n" + "=" * 60)
    print("第二部分：try/except — 基础错误处理")
    print("=" * 60)

    from openai import (
        APIConnectionError,  # 连不上服务器
        AuthenticationError,  # API Key 不对
        OpenAI,  # 客户端
    )

    # 用假 Key 创建客户端（肯定失败）
    client = OpenAI(
        api_key="sk-fake-key",
        base_url="https://api.deepseek.com",
    )

    # ── 场景1：捕获认证错误 ──
    print("\n── 场景1：捕获 AuthenticationError ──")
    try:
        client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": "hello"}],
        )
    except AuthenticationError as e:
        # e 是异常对象，里面装着错误的详细信息
        # status_code=401 表示"你没权限"
        print(f"  [CAUGHT] 认证失败！HTTP 状态码: {e.status_code}")
        print(f"  [CAUGHT] 错误体: {e.body}")
        print("  [行动] 检查你的 API Key 是否正确/过期")

    # ── 场景2：捕获连接错误 ──
    print("\n── 场景2：捕获 APIConnectionError ──")
    try:
        # 连到一个不存在的地址（模拟网络不通）
        bad_network_client = OpenAI(
            api_key="sk-whatever",
            base_url="https://this-host-does-not-exist-12345.com",
        )
        bad_network_client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": "hello"}],
        )
    except APIConnectionError as e:
        print("  [CAUGHT] 网络连接失败！")
        print(f"  [CAUGHT] 原因: {e.message}")
        print("  [行动] 检查网络 / base_url 是否正确 / 目标服务是否在线")
    except AuthenticationError:
        # 注意：上面的场景中，如果网络请求根本没发出去，
        # 就不会走到 AuthenticationError。这里只是演示 catch 的顺序。
        #
        # ⭐ 关键规则：把更具体的异常放在前面！
        #   ✅ 正确：except AuthenticationError → except APIError
        #   ❌ 错误：except APIError → except AuthenticationError
        #            ↑ 因为 AuthenticationError 是 APIError 的子类，
        #              如果先 catch APIError，AuthenticationError 会被它吞掉，
        #              永远不会进入 AuthenticationError 的分支。
        print("  [CAUGHT] 认证失败（但这种情况网络都连不上，认证检查不会触发）")

    # ── 场景3：兜底捕获 ──
    print("\n── 场景3：用 APIError 兜底 ──")
    print("  APIError 是 OpenAI 所有异常的父亲。")
    print("  上面没 catch 到的异常，最终会被它兜住。")
    print('  就像：AuthenticationError 是"钥匙不对"，APIError 是"所有门锁问题"。')

    # 总结信息层次
    print(
        """
    ╔══════════════════════════════════════════════════╗
    ║  异常层次结构（从具体到通用）                    ║
    ║                                                  ║
    ║  AuthenticationError ← 最具体：Key 错了          ║
    ║  RateLimitError      ← 很具体：请求太快了        ║
    ║  APIConnectionError  ← 具体：网络问题            ║
    ║  APIError            ← 通用：所有 OpenAI 异常     ║
    ║  Exception           ← 最通用：所有 Python 异常   ║
    ║                                                  ║
    ║  Catch 顺序：越具体越靠前。                      ║
    ║  如果先 catch Exception，后面全白写了。          ║
    ╚══════════════════════════════════════════════════╝
    """
    )


# ═══════════════════════════════════════════════════════════════
# 第三部分：重试机制 — 失败了再试试
# ═══════════════════════════════════════════════════════════════
#
# "重试"本身很简单——把代码包在一个 while 循环里。
# 但裸重试有 3 个问题：
#   1. 如果服务器宕机了，你一秒钟重试 1000 次也没用 → 浪费资源
#   2. 如果是因为请求太快被限流，你立刻重试又被打回来 → 进入死循环
#   3. 重试的日志、次数限制、等待时间 → 每次都要写一遍，容易出错
#
# 所以就有了 tenacity 这个库——帮你把"怎么重试"标准化。
#   名字来源于 tenacity（坚韧），表示"坚持不懈"。
#
# tenacity 提供三个核心配置：
#   wait   → 两次重试之间等多久
#   stop   → 最多重试多少次（或多久）
#   retry  → 什么情况下才重试


def demo_bare_retry():
    """❌ 裸重试：不推荐的方式——演示它为什么不好。"""
    print("\n" + "=" * 60)
    print("第三部分-A：裸重试（反例）")
    print("=" * 60)

    print(
        """
    # 你可能会想这样写：
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(...)
            break  # 成功了，退出循环
        except Exception:
            if attempt == max_retries - 1:
                raise  # 最后一次了，不重试了
            time.sleep(2)  # 固定等 2 秒

    # 问题1：固定等待 2 秒——如果服务器说要等 30 秒呢？
    # 问题2：不分错误类型——Key 错了也重试 5 次？纯浪费。
    # 问题3：没有指数退避——高并发时大家一起等 2 秒，然后一起重试，服务器又炸了。
    # 问题4：日志、统计、回调——这些都要自己写，散落在各处。

    -> 所以真正项目里我们直接用 tenacity。它把这些脏活都封装好了。
    """
    )


def demo_tenacity_basic():
    """✅ 用 tenacity 库——简洁、专业、不容易出错。

    ═══════════════════════════════════════════════
    核心配置参数（每个都要理解为什么是这个值）：
    ═══════════════════════════════════════════════

    wait=wait_exponential(multiplier=1, min=1, max=60)
      → 第1次重试等 1*2^0 = 1秒
      → 第2次重试等 1*2^1 = 2秒
      → 第3次重试等 1*2^2 = 4秒
      → ...最长不超过 60 秒
      → 这就是"指数退避"——越等越久，给服务器喘息时间

    stop=stop_after_attempt(3)
      → 最多重试 3 次
      → 为什么是 3？经验值。1-2 次可能不够，5 次太多用户等不及。

    retry=retry_if_exception_type(APIConnectionError)
      → 只有这种异常类型才重试
      → Key 错了（AuthenticationError）重试也白费

    额外考虑：jitter（抖动）
      → 在等待时间上加一个随机小偏移（如 ±0.5秒）
      → 防止"惊群效应"：100 个请求同时失败、同时等 1 秒、同时重试、同时打爆服务器
      → tenacity 默认不带 jitter，但生产环境建议加。
    """
    print("\n" + "=" * 60)
    print("第三部分-B：tenacity — 标准重试方案")
    print("=" * 60)

    from openai import APIConnectionError, OpenAI

    # ── 定义一个带了重试装饰器的函数 ──
    # @retry 是 Python 的"装饰器"（Decorator）语法。
    #
    # 装饰器是什么？
    #   类比：在门外加了一个门卫。
    #   你调用 call_api() → 实际先经过门卫 → 门卫决定是否放行/重试 → 才到你的函数。
    #   @retry(...) 就是在函数外面包了一层"自动重试"的逻辑。
    #
    # 你现在不需要完全理解装饰器原理，知道它"给函数加了个自动重试能力"就够了。
    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=10),
        #    ↑ 指数退避：第1次等1秒，第2次等2秒，第3次等4秒...最多10秒
        stop=stop_after_attempt(3),
        #    ↑ 最多重试3次，第4次还失败就放弃
        retry=retry_if_exception_type(APIConnectionError),
        #    ↑ 只有连接错误才重试。Key错了/没钱了重试没用。
        reraise=True,
        #    ↑ 重试全部失败后，把最后一次的异常抛出去（让上层知道"我尽力了"）
    )
    def call_with_retry(client, model, messages):
        """带自动重试的 API 调用。"""
        print("      [调用中...]")
        return client.chat.completions.create(
            model=model,
            messages=messages,
        )

    # ── 演示：用假地址触发 APIConnectionError，看十重试过程 ──
    print("\n── 演示：tenacity 自动重试过程 ──")
    client = OpenAI(
        api_key="sk-test",
        base_url="https://this-host-does-not-exist-98765.com",
    )

    try:
        call_with_retry(client, "deepseek-chat", [{"role": "user", "content": "hi"}])
    except Exception as e:
        print(f"  [结果] 3 次重试后依然失败：{type(e).__name__}: {e}")
        print("  [结论] 重试不是万能药。服务器真挂了，重试也没用。")


# ═══════════════════════════════════════════════════════════════
# 第四部分：Rate Limit — 你太快了，慢一点
# ═══════════════════════════════════════════════════════════════
#
# 什么是 Rate Limit（限流）？
#   每个 API 都有"每分钟最多调用多少次"的限制。
#   比如 DeepSeek 免费版可能限制每分钟 60 次请求。
#   超过这个数，API 返回 HTTP 429 "Too Many Requests"。
#
# 生活类比：
#   奶茶店一次只能做 5 杯。你点了第 6 杯，店员说"等 3 分钟"。
#   你过 1 分钟又去问，店员说"不是说了 3 分钟吗"。
#   你过 3 分钟再去，好了。


def demo_rate_limit_handling():
    """✅ 正确处理限流：看 Retry-After 头，等够了再重试。

    HTTP 429 响应里通常带一个 Retry-After 头：
      Retry-After: 30  → 等 30 秒再试
      Retry-After: 60  → 等 60 秒再试

    如果你不等够就重试，只会再吃一个 429。
    """
    print("\n" + "=" * 60)
    print("第四部分：Rate Limit 处理")
    print("=" * 60)

    from openai import OpenAI, RateLimitError

    # ── before_sleep：重试前的日志回调 ──
    # 必须定义在 @retry 装饰器之前！（Python 要求装饰器引用的东西要先存在）
    def before_sleep(retry_state):
        """每次重试等待前，打印一下当前状态。

        retry_state 有三种获取"下次"时间的方式：
          1. outcome.exception() → 从异常中提取 retry-after 头
          2. retry_state.next_action.sleep → RetryAction.sleep 属性
          3. retry_state.upcoming  → wait 耗时（最通用）

        tenacity 内部优先使用 HTTP 429 返回的 Retry-After 头作为等待时长，
        所以我们直接看 retry_state.next_action.sleep。
        """
        sleep_time = retry_state.next_action.sleep
        attempt = retry_state.attempt_number
        exc = retry_state.outcome.exception()
        print(
            f"    [RateLimit 第{attempt}次重试] "
            f"收到错误: {type(exc).__name__}, "
            f"等待 {sleep_time} 秒后重试..."
        )

    # ── 带限流感知的重试 ──
    @retry(
        wait=wait_exponential(multiplier=2, min=2, max=30),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(RateLimitError),
        before_sleep=before_sleep,
    )
    def call_with_rate_limit_awareness(client, model, messages):
        return client.chat.completions.create(model=model, messages=messages)

    print("\n── 模拟：被限流后的重试行为 ──")
    print("  (用错误 URL 模拟，实际 429 场景行为完全一样)")
    client = OpenAI(
        api_key="sk-test",
        base_url="https://this-host-does-not-exist-22222.com",
    )

    try:
        call_with_rate_limit_awareness(client, "deepseek-chat", [{"role": "user", "content": "hi"}])
    except Exception as e:
        print(f"  [最终结果] 重试耗尽，放弃: {type(e).__name__}")

    print(
        """
    ╔══════════════════════════════════════════════════════╗
    ║  Rate Limit 处理的最佳实践：                        ║
    ║                                                     ║
    ║  1. 读 Retry-After 响应头，优先用它给的等待时间     ║
    ║  2. 如果没给 Retry-After，用指数退避兜底           ║
    ║  3. 在 before_sleep 里打日志，方便排查"谁在重试"     ║
    ║  4. 不要在 retry 里包 AuthenticationError——         ║
    ║     Key 错了重试 10 次也没用！                      ║
    ╚══════════════════════════════════════════════════════╝
    """
    )


# ═══════════════════════════════════════════════════════════════
# 第五部分：Stream 中断恢复 — 打字到一半断了怎么办
# ═══════════════════════════════════════════════════════════════
#
# 流式输出（Streaming）的问题：
#   非流式：发请求 → 等 5 秒 → 拿到完整回复 → 成功 ✓
#   流式：  发请求 → 逐字收到 → 收到一半 → 网络断了！→ 前面收到的字也没了 ✗
#
# 为什么流式中断比普通中断更棘手？
#   因为每次流式请求返回的内容是"增量"的（每个 chunk 只有几个字），
#   中断后需要知道"已经收到了哪些字"，才能从断点继续。
#
# 策略：累积已收到的 token → 重连时把已收到的内容附在新请求里


def demo_stream_handling():
    """流式请求中断时，如何优雅地重连。

    思路很简单：
    1. 流式收数据时，每收到一个 chunk 就存到"已收内容"变量里
    2. 如果中途断了，把已收内容拼到重试请求里（告诉 LLM "从这里继续"）
    3. 重连后继续收新的 chunk，追加到已收内容后面

    注意：这是概念演示。实际产品中可能会用更复杂的方案（如 buffer 分段确认）。
    """
    print("\n" + "=" * 60)
    print("第五部分：Stream 中断恢复")
    print("=" * 60)

    from openai import APIError, OpenAI

    # ── 模拟一个"收到一半就断"的流 ──
    print("\n── 场景演示：流式输出中断后重连 ──")

    # 用一个变量累积已收到的完整文本
    received_text = ""

    @retry(
        wait=wait_fixed(1),  # 固定等 1 秒（流式场景用固定等待更合理，用户不想等太久）
        stop=stop_after_attempt(2),
        retry=retry_if_exception_type(APIError),
        reraise=True,
    )
    def stream_with_recovery(client, model, messages, previous_text=""):
        """流式调用，支持断点续传。

        previous_text = 之前已经收到的内容。
        如果中断了重试，把 previous_text 附在 user message 后面，
        告诉 LLM "你之前说到这了，请继续"。
        """
        nonlocal received_text  # 让内部函数能修改外部变量

        # 构建"续写"prompt
        if previous_text:
            # 把之前收到的一半内容拼接进去
            continuation_prompt = (
                f"{messages[-1]['content']}"
                f"\n\n[你之前的回复被中断了，这是你已写的内容：]\n{previous_text}"
                f"\n[请从断点处继续，不要重复已写的内容。]"
            )
            messages = list(messages)  # 复制一份，避免修改原列表
            messages[-1] = {"role": "user", "content": continuation_prompt}

        received_text = previous_text  # 从断点开始累积

        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                received_text += token

        return received_text

    # ── 用真实 API 演示（需要有效的 Key）──
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY", "sk-no-key"),
        base_url="https://api.deepseek.com",
    )

    prompt = "用五句话介绍什么是机器学习"
    print(f"  提问: {prompt}")

    # 用一个可控的方式演示：正常流式接收
    received_text = ""
    try:
        result = stream_with_recovery(
            client, "deepseek-chat", [{"role": "user", "content": prompt}]
        )
        print(f"  完整回复 ({len(result)} 字): {result}")
    except Exception as e:
        print(f"  [失败] {type(e).__name__}: {e}")
        print("  (如果你看到认证错误，说明 API Key 没设置——这不影响理解机制)")

    print(
        """
    ╔══════════════════════════════════════════════════════╗
    ║  Stream 中断恢复的核心思路：                        ║
    ║                                                     ║
    ║  1. 每收到一个 chunk 就存起来（别只在内存里记）     ║
    ║  2. 中断时记录"已收到多少 token"                   ║
    ║  3. 重连时把已收内容附在新请求里                    ║
    ║  4. LLM 从断点继续，而不是从头开始                  ║
    ║                                                     ║
    ║  局限性：                                           ║
    ║  - 两次请求的回复可能不完全接续（模型不同次可能    ║
    ║    措辞略有不同）                                   ║
    ║  - 会消耗额外的 token（重复发送已收内容）           ║
    ║  - 对于特别长的回复（几万字）累积的成本不低        ║
    ╚══════════════════════════════════════════════════════╝
    """
    )


# ═══════════════════════════════════════════════════════════════
# 第六部分：完整的 LLMClient 重试集成示例
# ═══════════════════════════════════════════════════════════════
#
# 把前面学的东西整合到 Day 4 的 LLMClient 里。
# 核心思路：
#   在 LLMClient 内部加一层"带重试的调用"。
#   对外接口不变，但所有 API 调用都自动具备重试能力。
#
# 这是装饰器的进阶用法——把一个 @retry 装饰器做成"可配置的"。
# 你不需要完全理解这个函数怎么写的，知道怎么用就行。


def create_retry_decorator(max_retries: int = 3):
    """创建一个可配置的重试装饰器。

    参数：
      max_retries → 最多重试几次（默认3次）

    这个函数的返回值是一个 @retry 装饰器，可以直接用在任何函数上。

    为什么要把装饰器"包在函数里"？
      因为不同的场景可能需要不同的重试次数：
      - 聊天：3 次就够了，用户不想等太久
      - 文档处理：可以多试几次（5-10 次），因为耗时操作失败代价高
    """
    from openai import (
        APIConnectionError,
        APIStatusError,
        AuthenticationError,
        BadRequestError,
        RateLimitError,
    )

    # ⭐ 用自定义函数判断"该不该重试"，而不是简单按异常类型匹配。
    # 原因：AuthenticationError 继承自 APIStatusError（HTTP 401），
    # 如果只写 retry_if_exception_type(APIStatusError)，401 也会被重试——浪费。
    def _should_retry(exc: BaseException) -> bool:
        # 永久性错误 —— 重试也治不好
        if isinstance(exc, (AuthenticationError, BadRequestError)):
            return False
        # 临时性错误 —— 重试可能治好
        if isinstance(exc, (APIConnectionError, RateLimitError, APIStatusError)):
            return True
        # 不认识的异常 —— 保守起见不重试
        return False

    return retry(
        wait=wait_exponential(multiplier=1, min=1, max=30),
        stop=stop_after_attempt(max_retries),
        retry=retry_if_exception(_should_retry),
        before_sleep=before_sleep_log(_demo_logger, logging.WARNING),
        reraise=True,
    )


def demo_integrated_retry():
    """演示：把重试装饰器用在 LLMClient 风格的调用上。"""
    print("\n" + "=" * 60)
    print("第六部分：集成到 LLMClient 的完整方案")
    print("=" * 60)

    from openai import OpenAI

    def call_api_with_retry(client, model, messages, max_retries=3):
        """一个玩具示例：在函数内部用 tenacity 的 retry 装饰器。"""

        # 在函数内部动态创建装饰器——因为 max_retries 可能是变量
        retry_decorator = create_retry_decorator(max_retries=max_retries)

        @retry_decorator
        def _do_call():
            return client.chat.completions.create(model=model, messages=messages)

        return _do_call()

    print("\n── 用假地址测试（会触发重试 + 最终失败）──")
    client = OpenAI(
        api_key="sk-test",
        base_url="https://this-host-does-not-exist-33333.com",
    )

    try:
        call_api_with_retry(
            client,
            "deepseek-chat",
            [{"role": "user", "content": "hi"}],
            max_retries=2,
        )
    except Exception as e:
        print(f"  [最终] {type(e).__name__}: 重试 2 次后仍失败")

    print(
        """
    ╔══════════════════════════════════════════════════════╗
    ║  整合要点回顾：                                    ║
    ║                                                     ║
    ║  1. @retry 装饰器直接加在 API 调用函数上            ║
    ║  2. retry 参数只选"临时性异常"                      ║
    ║  3. wait_exponential 防止压垮服务器                  ║
    ║  4. before_sleep 日志方便排查                        ║
    ║  5. reraise=True 确保上层知道"已经努力过了"         ║
    ╚══════════════════════════════════════════════════════╝
    """
    )


# ═══════════════════════════════════════════════════════════════
# ✍️ 动手区
# ═══════════════════════════════════════════════════════════════
def your_turn():
    """✍️ 学生动手区 —— 完成下方的 3 个任务。"""
    print("\n" + "=" * 60)
    print("✍️ 动手区")
    print("=" * 60)

    # ── 任务1：写出正确的异常捕获层次 ──
    print("\n── 任务1：异常捕获顺序练习 ──")
    print("  下面这段代码的 except 顺序有问题，你能发现吗？\n")

    print("  ```python")
    print("  try:")
    print("      client.chat.completions.create(...)")
    print("  except Exception as e:          # ← 第一个 catch")
    print("      print(f'出错了: {e}')")
    print("  except AuthenticationError as e: # ← 第二个 catch")
    print("      print(f'认证失败！')")
    print("  ```")
    print()
    print("  ❓ 问题：第二个 except 永远不会执行。为什么？")
    print("  ✍️ 写出正确的顺序，然后运行后面的任务 2 验证。")

    # ── 任务2：动手写一个带重试的函数 ──
    print("\n── 任务2：用 tenacity 写一个带重试的函数 ──")
    print("  ✍️ 目标：写一个函数 call_with_retry()，满足以下要求：")
    print("     1. 网络连接错误时自动重试，最多 3 次")
    print("     2. 每次重试间隔递增：1秒 → 2秒 → 4秒")
    print("     3. 认证错误时不重试（直接抛异常）")
    print("     4. 每次重试前打印日志")
    print()
    print("  提示：参考第三部分-B 的代码。你需要修改的有：")
    print("    - retry 参数里指定正确的异常类型")
    print("    - wait 和 stop 参数")
    print("    - 加一个 before_sleep 日志函数")
    print()
    print("  写完后用下面的测试框架验证。")

    # ── 任务3：理解"哪些异常该重试" ──
    print("\n── 任务3：判断哪些错误该重试 ──")
    print("  ✍️ 给每种错误标注 ✅（该重试）或 ❌（不该重试），并写一句理由：")
    print("     1. APIConnectionError — 网络不通")
    print("     2. AuthenticationError — API Key 错误")
    print("     3. RateLimitError — 被限流（HTTP 429）")
    print("     4. APIStatusError (HTTP 503) — 服务器内部错误")
    print("     5. BadRequestError (HTTP 400) — 请求参数格式错误")
    print()
    print('  提示：问自己"再试一次能解决问题吗？"')


# ==================== 主程序 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("Week 1 Day 5 — 异常处理与重试机制")
    print("=" * 60)
    print()
    print("今天我们要解决的问题：")
    print("  API 调用会失败——Key 错了、网络断了、太频繁被限流...")
    print("  如果不处理，程序直接崩溃。")
    print("  如果处理得当，程序能自动重试、优雅降级。")
    print()
    print("运行顺序：")
    print("  1. demo_pain_no_handling()    ← 先看反面：不处理有多痛")
    print("  2. demo_basic_try_except()   ← 基础武器：try/except")
    print("  3. demo_tenacity_basic()     ← 进阶武器：tenacity 自动重试")
    print("  4. demo_rate_limit_handling()← 限流处理：尊重 429")
    print("  5. demo_stream_handling()    ← 流中断恢复")
    print("  6. demo_integrated_retry()   ← 整合到 LLMClient")
    print("  7. your_turn()               ← ✍️ 你动手")

    # 默认只运行安全的演示。
    # ✍️ 逐个取消注释来学习每个部分：
    #    先看反例 → 再学解决方案 → 最后动手

    print("\n>>> 运行 反例演示（会故意崩溃，让你看看「不处理异常」的后果）...")
    try:
        demo_pain_no_handling()
    except Exception as e:
        print("\n[看到了吗？程序直接崩溃了。]")
        print(f"  错误类型: {type(e).__name__}")
        print(f"  错误信息: {e}")
        print("  如果这是在真实应用里，用户看到的就是这个红字 traceback。")
        print("  → 接下来学 try/except 就是为了避免这种情况。\n")

    print("\n>>> 运行 基础 try/except 演示...")
    demo_basic_try_except()

    print("\n>>> 运行 裸重试反例...")
    demo_bare_retry()

    print("\n>>> 运行 tenacity 标准重试演示...")
    demo_tenacity_basic()

    print("\n>>> 运行 Rate Limit 处理演示...")
    demo_rate_limit_handling()

    print("\n>>> 运行 Stream 中断恢复演示...")
    demo_stream_handling()

    print("\n>>> 运行 集成方案演示...")
    demo_integrated_retry()

    your_turn()
