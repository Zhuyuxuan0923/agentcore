"""
Week 2 Day 2 — Prompt 模板系统完整演示

运行方式：poetry run python src/study_agent/prompt/demo.py

演示内容：
  1. 不用模板的痛 → 用模板解决
  2. 同一个模板 + 不同变量 → 不同输出
  3. Few-Shot 示例注入（代码审查场景）
  4. 不同模板应对不同场景
"""

from study_agent.prompt import FewShotManager, PromptManager

TEMPLATE_DIR = "src/study_agent/prompt/templates"

print("=" * 64)
print("Week 2 Day 2 — Prompt 模板系统演示")
print("=" * 64)

# ═══════════════════════════════════════════════════════════
# 第一步：看看有哪些模板可用
# ═══════════════════════════════════════════════════════════
manager = PromptManager(TEMPLATE_DIR)
print("\n[Templates] 已加载模板:", manager.list_templates())

# ═══════════════════════════════════════════════════════════
# 第二步：渲染一个不带 Few-Shot 的基础 prompt
# ═══════════════════════════════════════════════════════════
print("\n" + "─" * 64)
print("场景 1：代码审查（无 Few-Shot 示例）")
print("─" * 64)

prompt1 = manager.render(
    "code_review",
    role="资深 Python 后端工程师",
    context="正在审查一个用户登录模块的代码，关注安全性和错误处理。",
    rules=[
        "SQL 查询必须使用参数化，严禁字符串拼接",
        "密码不得明文存储，必须使用 bcrypt 哈希",
        "异常处理必须指定具体异常类型，禁止裸 except",
        "敏感操作必须记录审计日志",
    ],
    output_format="按严重程度（致命/严重/建议）分类列出问题，每条包含行号和修复建议。",
    few_shot_text="",  # ← 没有示例
    code='def login(username, pw):\n    query = "SELECT * FROM users WHERE name=\'" + username + "\'"\n    result = db.execute(query)\n    if result and result[3] == pw:\n        return True\n    return False',
)
print(prompt1)

# ═══════════════════════════════════════════════════════════
# 第三步：准备 Few-Shot 示例仓库
# ═══════════════════════════════════════════════════════════
print("\n" + "─" * 64)
print("场景 2：Few-Shot 管理器 — 添加、筛选、格式化示例")
print("─" * 64)

few_shot = FewShotManager()

# 添加 5 个示例，打好标签
few_shot.add(
    input_text='def get_user(id):\n    q = "SELECT * FROM users WHERE id=" + str(id)\n    return db.execute(q)',
    output_text='[致命] 第2行 SQL 注入漏洞：使用字符串拼接构造查询。修复：改为参数化查询 db.execute("SELECT * FROM users WHERE id=?", [id])',
    tags=["代码审查", "安全", "SQL注入"],
)
few_shot.add(
    input_text='try:\n    file = open("/etc/config")\nexcept:\n    print("error")',
    output_text="[严重] 第2行 裸 except：捕获了所有异常包括 KeyboardInterrupt。修复：改为 except IOError as e: 并记录日志",
    tags=["代码审查", "错误处理"],
)
few_shot.add(
    input_text="def check_pw(user, pw):\n    stored = get_stored_pw(user)\n    if stored == pw:\n        return True",
    output_text="[致命] 第3行 密码明文比较：用户密码被直接比较。修复：使用 bcrypt.checkpw(pw, stored) 进行哈希后比较",
    tags=["代码审查", "安全", "密码"],
)
few_shot.add(
    input_text='def transfer(from_acc, to_acc, amount):\n    db.execute(f"UPDATE accounts SET balance=balance-{amount} WHERE id={from_acc}")\n    db.execute(f"UPDATE accounts SET balance=balance+{amount} WHERE id={to_acc}")',
    output_text="[致命] 第2-3行 SQL注入 + 缺少事务：应使用参数化查询并包裹在事务中。第2行失败时第3行不会回滚，导致金额凭空消失。",
    tags=["代码审查", "安全", "事务"],
)
few_shot.add(
    input_text='def search(keyword):\n    # 搜索产品\n    results = Product.objects.filter(name__icontains=keyword)\n    return [{"id": r.id, "name": r.name, "price": r.price} for r in results]',
    output_text="[建议] 第3行 未限制返回数量：搜索结果可能返回上万条，造成内存溢出和响应超时。修复：添加 .limit(100)",
    tags=["代码审查", "性能"],
)

print(f"已添加 {few_shot.count} 个示例\n")

# 按"安全"标签筛选 3 个示例
selected = few_shot.pick(tags=["安全"], max_count=3)
print(f"按 [安全] 筛选出 {len(selected)} 个示例：")
for ex in selected:
    print(f"  {ex}")

formatted = few_shot.format(selected)
print(f"\n格式化后的文本（{len(formatted)} 字符）:\n{formatted[:200]}...")

# ═══════════════════════════════════════════════════════════
# 第四步：把 Few-Shot 示例注入模板
# ═══════════════════════════════════════════════════════════
print("─" * 64)
print("场景 3：代码审查（注入 3 个 Few-Shot 安全示例）")
print("─" * 64)

prompt2 = manager.render(
    "code_review",
    role="资深 Python 后端工程师",
    context="正在审查一个用户登录模块的代码，关注安全性和错误处理。",
    rules=[
        "SQL 查询必须使用参数化，严禁字符串拼接",
        "密码不得明文存储，必须使用 bcrypt 哈希",
        "异常处理必须指定具体异常类型，禁止裸 except",
        "敏感操作必须记录审计日志",
    ],
    output_format="按严重程度（致命/严重/建议）分类列出问题，每条包含行号和修复建议。",
    few_shot_text=formatted,  # ← 这次注入示例
    code='def login(username, pw):\n    query = "SELECT * FROM users WHERE name=\'" + username + "\'"\n    result = db.execute(query)\n    if result and result[3] == pw:\n        return True\n    return False',
)
print(prompt2)

# ═══════════════════════════════════════════════════════════
# 第五步：对比——同一个模板、不同变量
# ═══════════════════════════════════════════════════════════
print("─" * 64)
print("场景 4：同一个 template、不同角色 + 不同规则 → 完全不同用途")
print("─" * 64)

prompt3 = manager.render(
    "code_review",
    role="前端性能优化专家",
    context="正在审查一段 React 组件代码，关注渲染性能和内存泄漏。",
    rules=[
        "不要在 render 中创建新函数或对象（每次渲染都会重新创建）",
        "useEffect 必须有清理函数（防止内存泄漏）",
        "大列表必须使用虚拟滚动（react-window 或 react-virtuoso）",
    ],
    output_format="按性能影响（高/中/低）分级，每项给出优化前后的代码对比。",
    few_shot_text="",
    code="""function UserList({ users }) {
  const [count, setCount] = useState(0);
  useEffect(() => {
    const timer = setInterval(() => setCount(c => c + 1), 1000);
  }, []);
  return users.map(u => <UserCard key={u.id} user={u} onClick={() => alert(u.name)} />);
}""",
)
print(prompt3)

print("\n" + "=" * 64)
print("演示完成！总结今天学到的：")
print("  1. Jinja2 把 prompt 从 Python 字符串变成了可管理的模板文件")
print("  2. {{ var }} 替换变量，{%% if %%} 控制条件，{%% for %%} 循环展开")
print("  3. FewShotManager 独立管理示例→按标签筛选→注入模板")
print("  4. 同一个 .j2 模板 + 不同变量 = 完全不同的 prompt")
print("=" * 64)
