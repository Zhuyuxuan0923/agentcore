"""Week 1 Day 5 单元测试 — 覆盖所有异常分支。

🤖 AI 生成测试骨架，✍️ 你需要运行并理解每个测试在测什么。

═══════════════════════════════════════════════════════
为什么要写单元测试？在这个场景下：
═══════════════════════════════════════════════════════

测试"重试逻辑"不能靠"真的把网线拔了"来验证。
所以我们用 Mock（模拟）——假装 API 返回了某种错误，
然后验证我们的代码是否正确处理了它。

Mock 的生活类比：
  你想测试"地震时房子会不会倒"，但不可能真的制造地震。
  于是你用一个振动台（Mock）来模拟地震的效果。
  这里的 Mock 就是振动台——我们让 API 假装返回错误。

pytest 是什么？
  Python 最流行的测试框架。
  你把测试函数命名成 test_xxx，放在 test_ 开头的文件里，
  pytest 会自动发现并运行它们。
"""

from __future__ import annotations

import logging
from unittest.mock import Mock

import httpx
import pytest
from anthropic import (
    APIConnectionError as AnthropicAPIConnectionError,
)
from anthropic import (
    APIStatusError as AnthropicAPIStatusError,
)
from anthropic import (
    RateLimitError as AnthropicRateLimitError,
)
from openai import (
    APIConnectionError as OpenaiAPIConnectionError,
)
from openai import (
    APIStatusError as OpenaiAPIStatusError,
)
from openai import (
    AuthenticationError,
)
from openai import (
    RateLimitError as OpenaiRateLimitError,
)
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

# ── 构建 fake request 对象 —— 新版 openai SDK 的异常构造需要 request 参数 ──
# openai >= 2.0 对异常增加了严格的类型检查，必须传 request 对象。


def _fake_request():
    """造一个假的 httpx Request，仅供异常构造使用。"""
    return httpx.Request("POST", "https://fake.api.example.com/v1/chat/completions")


# ═══════════════════════════════════════════════════════════════
# create_retry_decorator（从演示文件复制过来，因为测试要独立运行）
# ═══════════════════════════════════════════════════════════════

_logger = logging.getLogger("test_error_handling")


def _should_retry(exception: BaseException) -> bool:
    """判断一个异常是否值得重试。

    关键认知：不是所有 HTTP 错误都该重试。
    - 401 (AuthenticationError) → Key 错了，重试没用
    - 400 (BadRequestError) → 请求格式错了，重试没用
    - 429 (RateLimitError) → 临时限流，重试有用
    - 5xx (APIStatusError) → 服务器问题，可能恢复
    - 网络错误 → 可能恢复
    """
    # AuthenticationError 继承自 APIStatusError，但它不该重试
    # 所以要单独排除
    from anthropic import AuthenticationError as AnthropicAuthError
    from anthropic import BadRequestError as AnthropicBadRequestError
    from openai import AuthenticationError as OpenaiAuthError
    from openai import BadRequestError as OpenaiBadRequestError

    # 永久性错误：不重试
    if isinstance(exception, (OpenaiAuthError, AnthropicAuthError)):
        return False
    if isinstance(exception, (OpenaiBadRequestError, AnthropicBadRequestError)):
        return False

    # 临时性错误：重试
    if isinstance(
        exception,
        (
            OpenaiAPIConnectionError,
            AnthropicAPIConnectionError,
            OpenaiRateLimitError,
            AnthropicRateLimitError,
            OpenaiAPIStatusError,
            AnthropicAPIStatusError,
        ),
    ):
        return True

    # 其他不认识的异常：不重试（安全起见）
    return False


def create_retry_decorator(max_retries: int = 3):
    """创建一个可配置的重试装饰器（与 error_handling_demo.py 中的版本一致）。"""
    return retry(
        wait=wait_exponential(multiplier=1, min=1, max=30),
        stop=stop_after_attempt(max_retries + 1),
        retry=retry_if_exception(_should_retry),
        before_sleep=before_sleep_log(_logger, logging.WARNING),
        reraise=True,
    )


# ═══════════════════════════════════════════════════════════════
# 测试1：认证错误不应该重试
# ═══════════════════════════════════════════════════════════════
def test_auth_error_should_not_retry():
    """确认：AuthenticationError 不会触发重试。

    基本原理：
      如果 API Key 本身是错的，重试 100 次也不会突然变对。
      这种"永久性错误"应该直接失败——不重试，不等待，立即报错。
    """
    # ① 创建一个假的 OpenAI 客户端
    mock_client = Mock()

    # ② 告诉 mock："当调用 create 时，抛出 AuthenticationError"
    mock_client.chat.completions.create.side_effect = AuthenticationError(
        message="Invalid API Key",
        response=Mock(status_code=401),
        body={"error": "invalid_api_key"},
    )

    # ③ 用 create_retry_decorator(1) 创建一个"最多重试 1 次"的装饰器
    retry_decorator = create_retry_decorator(max_retries=1)

    @retry_decorator
    def call_api():
        return mock_client.chat.completions.create(
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
        )

    # ④ 调用函数，期望它直接抛出异常（不重试）
    with pytest.raises(AuthenticationError):
        call_api()

    # ⑤ 验证：只调用了 1 次 API（没有重试）
    assert mock_client.chat.completions.create.call_count == 1, (
        f"认证错误不该重试！期望调用 1 次，实际调用了 "
        f"{mock_client.chat.completions.create.call_count} 次"
    )


# ═══════════════════════════════════════════════════════════════
# 测试2：网络连接错误应该重试
# ═══════════════════════════════════════════════════════════════
def test_connection_error_should_retry():
    """确认：APIConnectionError 会触发重试。

    基本原理：
      网络问题是"临时性"的——下一秒可能就好了。
      所以值得重试几次。
    """
    mock_client = Mock()

    # 前 2 次调用抛连接错误，第 3 次成功
    mock_client.chat.completions.create.side_effect = [
        OpenaiAPIConnectionError(message="Network is unreachable", request=_fake_request()),
        OpenaiAPIConnectionError(message="Network is unreachable", request=_fake_request()),
        Mock(choices=[Mock(message=Mock(content="终于成功了！"))]),
    ]

    retry_decorator = create_retry_decorator(max_retries=3)

    @retry_decorator
    def call_api():
        return mock_client.chat.completions.create(
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
        )

    result = call_api()

    assert (
        mock_client.chat.completions.create.call_count == 3
    ), f"连接错误应该重试！期望 3 次，实际 {mock_client.chat.completions.create.call_count} 次"
    assert result.choices[0].message.content == "终于成功了！"


# ═══════════════════════════════════════════════════════════════
# 测试3：重试耗尽后应该抛出最后的异常
# ═══════════════════════════════════════════════════════════════
def test_retry_exhausted_should_raise():
    """确认：重试次数用完后，把最后一次的异常抛给调用者。

    这个测试确保"reraise=True"生效——上层知道"我已经尽力了"。
    """
    mock_client = Mock()

    mock_client.chat.completions.create.side_effect = OpenaiAPIConnectionError(
        message="Persistent network failure",
        request=_fake_request(),
    )

    retry_decorator = create_retry_decorator(max_retries=2)

    @retry_decorator
    def call_api():
        return mock_client.chat.completions.create(
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
        )

    with pytest.raises(OpenaiAPIConnectionError):
        call_api()

    # 验证：调用了 1（初始）+ 2（重试）= 3 次
    assert mock_client.chat.completions.create.call_count == 3


# ═══════════════════════════════════════════════════════════════
# 测试4：限流错误应该重试
# ═══════════════════════════════════════════════════════════════
def test_rate_limit_should_retry():
    """确认：RateLimitError（HTTP 429）会触发重试。

    限流是典型的"临时性错误"——等一会儿再试通常能成功。
    """
    mock_client = Mock()

    mock_client.chat.completions.create.side_effect = [
        OpenaiRateLimitError(
            message="Too many requests",
            response=Mock(status_code=429),
            body={"error": "rate_limit_exceeded"},
        ),
        Mock(choices=[Mock(message=Mock(content="限流解除后成功！"))]),
    ]

    retry_decorator = create_retry_decorator(max_retries=2)

    @retry_decorator
    def call_api():
        return mock_client.chat.completions.create(
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
        )

    result = call_api()

    assert mock_client.chat.completions.create.call_count == 2
    assert result.choices[0].message.content == "限流解除后成功！"


# ═══════════════════════════════════════════════════════════════
# 测试5：服务器 5xx 错误应该重试
# ═══════════════════════════════════════════════════════════════
def test_server_error_should_retry():
    """确认：服务器 5xx 错误会触发重试。

    500/502/503 说明服务器暂时有问题，过一会儿可能恢复。
    但 400（请求格式错误）不应该重试——那是你的问题，不是服务器的。
    """
    mock_client = Mock()

    mock_client.chat.completions.create.side_effect = [
        OpenaiAPIStatusError(
            message="Service Unavailable",
            response=Mock(status_code=503),
            body={"error": "service_unavailable"},
        ),
        Mock(choices=[Mock(message=Mock(content="服务器恢复了！"))]),
    ]

    retry_decorator = create_retry_decorator(max_retries=2)

    @retry_decorator
    def call_api():
        return mock_client.chat.completions.create(
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
        )

    result = call_api()

    assert mock_client.chat.completions.create.call_count == 2
    assert result.choices[0].message.content == "服务器恢复了！"


# ═══════════════════════════════════════════════════════════════
# 测试6：自定义重试次数
# ═══════════════════════════════════════════════════════════════
def test_custom_max_retries():
    """确认：max_retries 参数可以控制重试行为。

    不同的场景需要不同的重试次数：
    - 实时聊天：1-2 次，用户不想等
    - 离线批处理：5-10 次，成功率更重要
    """
    mock_client = Mock()

    mock_client.chat.completions.create.side_effect = OpenaiAPIConnectionError(
        message="Always fail",
        request=_fake_request(),
    )

    # 设置 max_retries=5
    retry_decorator = create_retry_decorator(max_retries=5)

    @retry_decorator
    def call_api():
        return mock_client.chat.completions.create(
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
        )

    with pytest.raises(OpenaiAPIConnectionError):
        call_api()

    # 调用次数 = 初始 1 + 重试 5 = 6
    assert mock_client.chat.completions.create.call_count == 6


# ═══════════════════════════════════════════════════════════════
# 测试7：首次成功不应该有重试
# ═══════════════════════════════════════════════════════════════
def test_first_success_no_retry():
    """确认：第一次就成功的时候，不会触发任何重试。

    这是"快乐路径"（Happy Path）测试——大多数时候 API 调用是正常的。
    重试机制不应该影响正常情况下的性能。
    """
    mock_client = Mock()

    mock_client.chat.completions.create.return_value = Mock(
        choices=[Mock(message=Mock(content="一次就成功！"))]
    )

    retry_decorator = create_retry_decorator(max_retries=3)

    @retry_decorator
    def call_api():
        return mock_client.chat.completions.create(
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
        )

    result = call_api()

    # 只调 1 次，没有重试
    assert mock_client.chat.completions.create.call_count == 1
    assert result.choices[0].message.content == "一次就成功！"


# ═══════════════════════════════════════════════════════════════
# 测试8：Stream chunk 累积逻辑
# ═══════════════════════════════════════════════════════════════
def test_stream_chunk_accumulation():
    """确认：流式接收时，chunk 能正确累积到完整回复。

    这个测试不涉及真实 API——只是验证"累积逻辑"本身是正确的。
    """

    # 模拟一系列 chunk（每个 chunk 就是几个字）
    class FakeChunk:
        class Choice:
            class Delta:
                def __init__(self, content):
                    self.content = content

            def __init__(self, content):
                self.delta = self.Delta(content)

        def __init__(self, content):
            self.choices = [self.Choice(content)]

    chunks = [
        FakeChunk("机器学习"),
        FakeChunk("是人工智能"),
        FakeChunk("的一个分支"),
        FakeChunk("，它让计算机从数据中学习。"),
    ]

    # 模拟累积过程
    accumulated = ""
    for chunk in chunks:
        if chunk.choices[0].delta.content:
            accumulated += chunk.choices[0].delta.content

    expected = "机器学习是人工智能的一个分支，它让计算机从数据中学习。"
    assert accumulated == expected
    assert len(accumulated) == 27


# ═══════════════════════════════════════════════════════════════
# 运行测试的命令（在项目根目录执行）：
#
#   poetry run pytest tests/test_error_handling.py -v
#
# -v 表示 verbose（详细输出），每个测试会显示 PASSED 或 FAILED。
# 看到 8 passed 就是全部通过了。
# ═══════════════════════════════════════════════════════════════
