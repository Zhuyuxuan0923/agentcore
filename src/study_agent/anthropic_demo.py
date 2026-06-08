"""Anthropic SDK 入门 — Claude API 与 OpenAI 的关键区别。

🤖 这节课 AI 生成骨架，你重点理解"两个 SDK 哪里不一样"。

Day 3 用 OpenAI SDK 调了 DeepSeek，今天用 Anthropic SDK 调 Claude。
把两套 SDK 对比着学，你就能看出设计哲学的差异。

核心差异预览：
1. System Prompt 位置 — OpenAI 塞在 messages 里，Anthropic 是独立参数
2. max_tokens 必填 — Anthropic 必须你设上限，OpenAI 有默认值
3. 模型名称体系 — 完全不同的命名规则
4. content 结构 — Anthropic 支持多 content block（文字+图片+工具调用混排）
"""

import os
import sys

from anthropic import Anthropic

sys.stdout.reconfigure(encoding="utf-8")

API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not API_KEY:
    print("[WARNING] 找不到 ANTHROPIC_API_KEY 环境变量！")
    print("-> 如果你还没有 Anthropic 账号，这节课可以先看代码理解差异")
    print("-> 后面的 LLMClient 封装会支持用 DeepSeek 代替 Claude 来跑")
    print("-> 获取 Key：https://console.anthropic.com → API Keys")
    print()

# 即使没有 Key 也先把客户端创建出来（不实际调用就行），让你看创建方式
client = Anthropic(api_key=API_KEY or "sk-placeholder")

# Anthropic 的模型名称规则：
#   claude-{系列名}-{主版本}-{次版本}-{日期可选}
#   例如 claude-sonnet-4-6  → Sonnet 4.6（性价比首选）
#        claude-opus-4-7   → Opus 4.7（最强，最贵，最慢）
#        claude-haiku-4-5  → Haiku 4.5（最快，最便宜）
MODEL = "claude-sonnet-4-6"


# ═══════════════════════════════════════════════════════════
# 区别 1：System Prompt 放哪里？
# ═══════════════════════════════════════════════════════════
def demo_system_prompt_difference():
    """展示 Anthropic 和 OpenAI 在 system prompt 设计上的根本差异。

    OpenAI 的 messages：
        messages = [
            {"role": "system", "content": "你是助手"},    # ← system 是一条消息
            {"role": "user", "content": "你好"},
        ]

    Anthropic 的参数：
        system="你是助手"    # ← system 是独立参数，不在 messages 里！
        messages=[{"role": "user", "content": "你好"}]

    为什么不一样？
    OpenAI 的设计哲学：system 也是对话的一部分，只是说话的人不同。
    Anthropic 的设计哲学：system 是"规则布告栏"，贴在对话外面，AI 始终看得见。
    这影响了行为——Anthropic 的 system 比 OpenAI 的更难被用户 prompt 覆盖。
    """
    print("\n" + "=" * 60)
    print("区别 1：System Prompt 的位置")
    print("=" * 60)

    if not API_KEY or API_KEY == "sk-placeholder":
        print("[SKIP] 没有有效的 API Key，跳过实际调用")
        print("代码对比（概念层面）：")
        print("  OpenAI:  messages里塞一条 {role:'system', ...}")
        print("  Anthropic: system='...' 独立参数，不在 messages 里")
        return

    response = client.messages.create(
        model=MODEL,
        max_tokens=200,  # ← Anthropic 强制要求这个参数！
        system="你是一个极简主义者，所有回答不超过15个字。",
        messages=[
            {"role": "user", "content": "什么是函数式编程？"},
        ],
    )

    # Anthropic 的返回结构和 OpenAI 也不同
    # 对比：
    #   OpenAI:    response.choices[0].message.content
    #   Anthropic: response.content[0].text
    print(f"Claude 回复: {response.content[0].text}")


# ═══════════════════════════════════════════════════════════
# 区别 2：max_tokens 是必填参数
# ═══════════════════════════════════════════════════════════
def demo_max_tokens_required():
    """Anthropic 要求你每次调用都设 max_tokens，不设就报错。

    为什么？
    OpenAI 有默认值（通常很大），你可以不传。
    Anthropic 的设计哲学：逼你思考"这个回复最多要多少 token"，
    目的是控制成本——防止系统 prompt 太长导致模型自动超长输出。
    """
    print("\n" + "=" * 60)
    print("区别 2：max_tokens 是必填的")
    print("=" * 60)

    if not API_KEY or API_KEY == "sk-placeholder":
        print("[SKIP] 没有有效的 API Key")
        return

    # ✅ 正确：明确设 max_tokens
    response = client.messages.create(
        model=MODEL,
        max_tokens=100,  # ← 表示"回复最多 100 个 token"
        messages=[{"role": "user", "content": "说'hello'"}],
    )
    print(f"✅ 带了 max_tokens=100 → 正常返回: {response.content[0].text}")

    # ❌ 错误：不设 max_tokens
    print("\n下面是反例——不传 max_tokens 会怎样？")
    try:
        client.messages.create(
            model=MODEL,
            # max_tokens 没设！
            messages=[{"role": "user", "content": "说'hello'"}],
        )
    except Exception as e:
        print(f"[EXPECTED CRASH] 错误类型: {type(e).__name__}")
        print(f"   信息: {e}")


# ═══════════════════════════════════════════════════════════
# 区别 3：Streaming 的返回结构不一样
# ═══════════════════════════════════════════════════════════
def demo_streaming():
    """对比两个 SDK 的流式返回结构。

    OpenAI streaming：
        for chunk in stream:
            if chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content)

    Anthropic streaming：
        with client.messages.stream(...) as stream:
            for text in stream.text_stream:
                print(text)

    Anthropic 封装得更人性化——直接给你 .text_stream 迭代器，
    不用自己去判断 delta.content 是不是 None。
    """
    print("\n" + "=" * 60)
    print("区别 3：Streaming 的返回结构")
    print("=" * 60)

    if not API_KEY or API_KEY == "sk-placeholder":
        print("[SKIP] 没有有效的 API Key")
        return

    print("Claude 流式输出: ", end="", flush=True)
    with client.messages.stream(
        model=MODEL,
        max_tokens=200,
        messages=[{"role": "user", "content": "用三句话介绍 Python 语言"}],
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
    print()


# ═══════════════════════════════════════════════════════════
# 区别 4：Anthropic 独有——Thinking（推理过程可见）
# ═══════════════════════════════════════════════════════════
def demo_thinking():
    """Claude 的"思考过程可见"功能——模型把推理步骤也返回给你。

    这很实用：你可以看到模型是怎么分析的，对调试 prompt 帮助很大。
    相当于能看 AI 的"草稿纸"。

    thinking_budget_tokens：给思考过程分配多少 token 预算（不算在回复里）。
    设为 0 或省略 → 不启用思考。
    设为 > 1024 → 至少 1024（API 最低要求）。
    """
    print("\n" + "=" * 60)
    print("区别 4：Thinking — 看 Claude 的推理过程")
    print("=" * 60)

    if not API_KEY or API_KEY == "sk-placeholder":
        print("[SKIP] 没有有效的 API Key")
        return

    response = client.messages.create(
        model=MODEL,
        max_tokens=500,
        messages=[
            {
                "role": "user",
                "content": (
                    "如果所有的猫都能飞，且 Tom 是一只猫，" "那么 Tom 能飞吗？解释你的推理过程。"
                ),
            }
        ],
        thinking={"type": "enabled", "budget_tokens": 1024},
    )

    # 思考内容存在单独的 content block 里，type 是 "thinking"
    for block in response.content:
        if block.type == "thinking":
            print(f"[Claude 的推理过程]:\n{block.thinking}")
        elif block.type == "text":
            print(f"\n[Claude 的最终回答]:\n{block.text}")


# ═══════════════════════════════════════════════════════════
# 区别 5：Content Block 结构
# ═══════════════════════════════════════════════════════════
def demo_content_blocks():
    """Anthropic 的消息内容可以包含多种 block 类型混合排列。

    OpenAI messages 中的 content 通常就是纯字符串。
    Anthropic 的 content 是一个 list，每个元素是一个 content block，
    可以混合：
    - text block：{"type": "text", "text": "..."}
    - image block：{"type": "image", "source": {...}}
    - tool_use block：{"type": "tool_use", ...}

    这个设计让多模态和工具调用更自然——图片和文字穿插着放。
    """
    print("\n" + "=" * 60)
    print("区别 5：Content Block — 不只存文字")
    print("=" * 60)

    if not API_KEY or API_KEY == "sk-placeholder":
        print("[SKIP] 没有有效的 API Key")
        return

    # 看看纯文字的 content 结构——注意 user 的 content 也可以写成 block 格式
    response = client.messages.create(
        model=MODEL,
        max_tokens=100,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "用一句话解释什么是 API"},
                ],
            }
        ],
    )
    # 返回的 content 也是一个 list
    print(f"response.content 类型: {type(response.content)}")
    print(f"block 数量: {len(response.content)}")
    for i, block in enumerate(response.content):
        print(f"  block[{i}].type = {block.type}")
        if hasattr(block, "text"):
            print(f"  block[{i}].text = {block.text}")


# ═══════════════════════════════════════════════════════════
# 对比总结
# ═══════════════════════════════════════════════════════════
def print_summary():
    """打印 OpenAI SDK 和 Anthropic SDK 的对比总结。"""
    print("\n" + "=" * 60)
    print("OpenAI SDK vs Anthropic SDK — 速查表")
    print("=" * 60)

    print(
        """
┌────────────────────┬─────────────────────────┬──────────────────────────┐
│       维度         │      OpenAI SDK         │     Anthropic SDK        │
├────────────────────┼─────────────────────────┼──────────────────────────┤
│ 安装               │ pip install openai      │ pip install anthropic    │
│ 客户端创建         │ OpenAI(api_key=,        │ Anthropic(api_key=)      │
│                    │        base_url=)        │                          │
│ 调用方法           │ client.chat.            │ client.messages.         │
│                    │   completions.create()   │   create()               │
│ system prompt      │ messages 数组里一条      │ 独立参数 system=""       │
│                    │ {role:"system", ...}     │                          │
│ max_tokens         │ 可选（有默认值）         │ 必填！                   │
│ 模型名称           │ gpt-4o, gpt-4o-mini     │ claude-sonnet-4-6 等     │
│ 回复取值           │ response.choices[0]     │ response.content[0]      │
│                    │   .message.content       │   .text                  │
│ 流式               │ for chunk in stream:    │ with client.messages.    │
│                    │   chunk.choices[0]       │   stream(...) as stream: │
│                    │   .delta.content         │   for t in stream.       │
│                    │                          │   text_stream            │
│ 思考过程           │ o1 系列有，不开放细节    │ thinking: {enabled,      │
│                    │                          │   budget_tokens}         │
│ content 结构       │ 通常是纯字符串           │ 是 list[ContentBlock]    │
│ temperature 默认值 │ 1.0                      │ 1.0                      │
│ timeout            │ 可在客户端设全局 timeout │ 每次调用可设 timeout     │
└────────────────────┴─────────────────────────┴──────────────────────────┘

关键结论：
  1. 90% 的功能两个 SDK 都能做，区别在 API 设计
  2. Anthropic 更"严谨"——max_tokens 必填、system 独立参数
  3. OpenAI 更"宽松"——更多默认值、更灵活的输入格式
  4. 国产模型几乎全用 OpenAI 兼容格式（DeepSeek/Moonshot/智谱都这样）
  5. 封装 LLMClient 的目的：抹平这些差异，让你换模型只改一行配置
"""
    )


# ==================== 主程序 ====================
if __name__ == "__main__":
    print("=== Anthropic SDK 学习脚本启动 ===\n")
    print("今天的核心问题：Anthropic SDK 和 OpenAI SDK 到底哪里不一样？\n")

    demo_system_prompt_difference()
    demo_max_tokens_required()
    demo_streaming()
    demo_thinking()
    demo_content_blocks()
    print_summary()

    print("\n👉 下一步：打开 llm_client.py 学习如何封装统一客户端")
