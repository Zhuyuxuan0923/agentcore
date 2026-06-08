"""LLM SDK 入门 — 用 Python 调用大模型。

这节课演示用 OpenAI SDK（兼容格式）调用大模型的四个核心概念：
1. Chat Completions — 发送对话，获取回复
2. System Prompt — 给 AI "定人设"
3. Temperature — 控制创造力的旋钮
4. Streaming — 逐字输出，不用等完整回复

📌 重要知识点：DeepSeek、Moonshot、智谱等国产模型的 API 都采用
   "OpenAI 兼容格式"，同一套代码换 URL + Key 就能调用不同厂商的模型。
"""

import os
import sys

from openai import OpenAI

# Windows 命令行默认用 GBK 编码，但 API 返回的字符可能超出 GBK 范围
# 这一行强制用 UTF-8 输出，解决中文乱码问题
sys.stdout.reconfigure(encoding="utf-8")

# ① 读取配置
#   通过环境变量选择用哪家模型，默认用 DeepSeek
API_KEY = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
# deepseek-chat = DeepSeek V3（性价比之选，等效 GPT-4o 级别）

if not API_KEY:
    print("[ERROR] 找不到 API Key！")
    print('-> 请在 .env 文件中填入：DEEPSEEK_API_KEY = "sk-your-key-here"')
    exit(1)

print(f"[INFO] 连接目标：{BASE_URL}")
print(f"[INFO] 使用模型：{MODEL}")

# ② 创建客户端 —— 比 OpenAI 多了一个 base_url 参数！
#   base_url 告诉 SDK "往哪个网址发请求"
#   换成 DeepSeek：https://api.deepseek.com
#   换成月之暗面：https://api.moonshot.cn/v1
#   换成智谱：    https://open.bigmodel.cn/api/paas/v4
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


# ═══════════════════════════════════════════════════════════
# 示例 1：最简单的对话 — 问一句，答一句
# ═══════════════════════════════════════════════════════════
def demo_basic_chat():
    """发送一条消息给大模型，获取完整回复。"""
    print("\n" + "=" * 60)
    print("示例 1：基础对话")
    print("=" * 60)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "用一句话解释什么是递归"}],
    )

    answer = response.choices[0].message.content
    print(f"模型回复：{answer}")


# ═══════════════════════════════════════════════════════════
# 示例 2：System Prompt — 给 AI 定人设
# ═══════════════════════════════════════════════════════════
def demo_system_prompt():
    """用 system prompt 让模型扮演特定角色。"""
    print("\n" + "=" * 60)
    print("示例 2：System Prompt — 让模型扮演小学老师")
    print("=" * 60)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是一个耐心的小学数学老师，用最简单的话解释概念，每次回答不超过 3 句话。"
                ),
            },
            {"role": "user", "content": "什么是分数？"},
        ],
    )

    print(f"模型（小学数学老师）回复：{response.choices[0].message.content}")


# ═══════════════════════════════════════════════════════════
# 示例 3：Temperature — 控制创造力的旋钮
# ═══════════════════════════════════════════════════════════
def demo_temperature():
    """对比 temperature=0（死板）和 temperature=1.5（天马行空）的区别。"""
    print("\n" + "=" * 60)
    print("示例 3：Temperature — 创造力的旋钮")
    print("=" * 60)

    prompt = "给我取 3 个宠物猫的名字"

    for temp in [0.0, 1.5]:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=temp,
        )
        print(f"temperature={temp}：{response.choices[0].message.content}")
        print()


# ═══════════════════════════════════════════════════════════
# 示例 4：Streaming — 逐字输出，像打字机一样
# ═══════════════════════════════════════════════════════════
def demo_streaming():
    """用 streaming 流式输出，不等完整回复。"""
    print("\n" + "=" * 60)
    print("示例 4：Streaming — 逐字输出")
    print("=" * 60)

    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "用三句话介绍 Python 语言"}],
        stream=True,
    )

    print("模型回复：", end="", flush=True)
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()


# ═══════════════════════════════════════════════════════════
# 示例 5：多轮对话 — 模型没有记忆，每次要重新发历史
# ═══════════════════════════════════════════════════════════
def demo_multi_turn():
    """演示多轮对话：把整段历史每次都发回去。"""
    print("\n" + "=" * 60)
    print("示例 5：多轮对话 — 把聊天记录打包发回去")
    print("=" * 60)

    messages = [
        {"role": "system", "content": "你是一个 Python 编程助手，回答要简洁。"},
    ]

    # 第一轮
    messages.append({"role": "user", "content": "Python 里列表和元组有什么区别？"})
    response = client.chat.completions.create(model=MODEL, messages=messages)
    reply_1 = response.choices[0].message.content
    print("第1轮 — 用户：列表和元组有什么区别？")
    print(f"       助手：{reply_1}")

    # 把助手的回复也加入 messages，这样下一轮 AI 就知道刚才说了什么
    messages.append({"role": "assistant", "content": reply_1})

    # 第二轮（AI 知道上下文）
    messages.append({"role": "user", "content": "那什么时候该用元组？"})
    response = client.chat.completions.create(model=MODEL, messages=messages)
    reply_2 = response.choices[0].message.content
    print("\n第2轮 — 用户：那什么时候该用元组？")
    print(f"       助手：{reply_2}")


# ═══════════════════════════════════════════════════════════
# ❌ 反例：不带错误处理——网络断了直接炸
# ═══════════════════════════════════════════════════════════
def demo_bad_no_error_handling():
    """反例：直接调用 API 但不做任何错误处理。"""
    print("\n" + "=" * 60)
    print("反例：不处理异常 — 用假的 Key，程序直接炸")
    print("=" * 60)

    bad_client = OpenAI(
        api_key="sk-this-is-fake-key",
        base_url=BASE_URL,
    )
    bad_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "hello"}],
    )


# ==================== 主程序 ====================
if __name__ == "__main__":
    print("=== LLM SDK 学习脚本启动 ===\n")

    demo_basic_chat()
    demo_system_prompt()
    demo_temperature()
    demo_streaming()
    demo_multi_turn()

    print("\n" + "=" * 60)
    print("接下来是反例演示...")
    print("=" * 60)

    try:
        demo_bad_no_error_handling()
    except Exception as e:
        print("[CRASH] 程序崩溃！")
        print(f"   错误类型：{type(e).__name__}")
        print(f"   错误信息：{e}")
        print("-> 这就是不处理异常的结果——程序直接死掉，什么都没留下")
