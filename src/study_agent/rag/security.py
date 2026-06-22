"""RAG 安全防护 -- 输入/输出安全检查。

提供两个核心函数：
  - detect_injection: 检测用户输入是否包含 Prompt Injection 尝试
  - sanitize_question: 截断超长问题

这是 Week 4 Day 5 新增的安全层。
"""

SUSPICIOUS_PATTERNS = [
    "忽略之前的指令",
    "忽略之前的指示",
    "ignore previous instructions",
    "ignore all previous",
    "忘记之前的",
    "forget previous",
    "告诉我你的system",
    "tell me your system prompt",
    "你的系统指令",
    "your system instructions",
    "现在开始扮演",
    "现在你是",
    "从现在起你是",
    "从现在开始你是",
    "pretend you are",
    "you are now",
    "作为一个AI",
    "作为一个人工智能",
    "new instructions",
]


def detect_injection(user_input: str) -> bool:
    """检测 Prompt Injection 尝试。

    这不是完美的检测——真正的安全需要多层防护：
    1. System Prompt 加固（已做）
    2. 输入检测（本函数）
    3. 输出过滤（Week 5+ 完善）

    Returns:
        True 如果检测到可疑输入
    """
    lowered = user_input.lower()
    return any(pattern.lower() in lowered for pattern in SUSPICIOUS_PATTERNS)


def sanitize_question(
    question: str,
    max_length: int = 2000,
    raise_on_injection: bool = False,
) -> str:
    """清理用户问题。

    Args:
        question: 用户输入
        max_length: 最大字符数
        raise_on_injection: 是否在检测到注入时抛出异常

    Returns:
        清理后的问题文本

    Raises:
        ValueError: 当 raise_on_injection=True 且检测到注入时
    """
    if not question or not question.strip():
        raise ValueError("问题不能为空")

    if raise_on_injection and detect_injection(question):
        raise ValueError("检测到潜在的 Prompt Injection 攻击")

    if len(question) > max_length:
        question = question[:max_length]

    return question.strip()
