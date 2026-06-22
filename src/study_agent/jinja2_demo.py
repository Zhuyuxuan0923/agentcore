"""
Jinja2 三个核心操作：变量 / 条件 / 循环
运行方式：poetry run python src/study_agent/jinja2_demo.py
"""

from jinja2 import Template

# ========== 操作 1：变量替换 ==========
# {{ variable }} 就是"在这里填变量值"，双花括号是 Jinja2 的占位符语法
template1 = Template("你是{{ role }}，请把以下内容{{ action }}：\n{{ content }}")
result1 = template1.render(
    role="资深中英翻译", action="翻译成英文", content="人工智能正在改变软件工程的每个环节。"
)
print("=== 操作 1：变量替换 ===")
print(result1)
print()

# ========== 操作 2：条件判断 ==========
# {% if %} ... {% else %} ... {% endif %} 控制某段文本出不出
template2 = Template(
    "你是{{ role }}。\n"
    "{% if tone == 'formal' %}"
    "请使用正式、专业的语气。\n"
    "{% else %}"
    "请使用友好、平易近人的语气。\n"
    "{% endif %}"
    "任务：{{ task }}"
)
print("=== 操作 2：条件判断 ===")
print("--- tone=formal ---")
print(template2.render(role="客服助手", tone="formal", task="回复用户投诉"))
print()
print("--- tone=casual ---")
print(template2.render(role="客服助手", tone="casual", task="回复用户投诉"))
print()

# ========== 操作 3：循环 ==========
# {% for item in list %} ... {% endfor %} 把列表逐个展开
template3 = Template(
    "你是代码审查专家。以下是 {{ count }} 条审查规则：\n"
    "{% for rule in rules %}"
    "{{ loop.index }}. {{ rule }}\n"
    "{% endfor %}"
    "\n请根据以上规则审查以下代码：\n```\n{{ code }}\n```"
)
result3 = template3.render(
    count=3,
    rules=[
        "函数名使用 snake_case 命名",
        "每个函数不超过 20 行",
        "不使用裸 except，必须指定异常类型",
    ],
    code="def GetUserName(id):\n    try:\n        return db.query(id)\n    except:\n        return None",
)
print("=== 操作 3：循环 ===")
print(result3)
