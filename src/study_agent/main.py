"""第一个 FastAPI 应用 — 图书查询 API。

这个文件演示了 FastAPI 最核心的三个概念：
1. 创建一个"服务员"（FastAPI app）
2. 给服务员挂上"菜单项"（路由 / 端点）
3. 让服务员开始接客（uvicorn 启动）
"""

from fastapi import FastAPI

# ① 创建服务员 —— app 就是我们整个 API 的入口
#   FastAPI() 是框架的"大脑"，所有请求都先到这里
app = FastAPI(
    title="图书查询 API",
    description="我的第一个 FastAPI 应用",
    version="0.1.0",
)


# ② 路由装饰器 —— 告诉服务员"这个网址归这个函数管"
#   @app.get("/") 意思是：有人访问 http://127.0.0.1:8000/ 时，调用 root()
@app.get("/")
def root() -> dict:
    """根路径，返回欢迎信息。"""
    return {"message": "欢迎来到图书查询 API！去 /docs 看文档"}


# ③ 路径参数 —— URL 里的动态部分
#   {book_id} 不是字面量，是占位符
#   访问 /books/3 → book_id = 3
#   访问 /books/42 → book_id = 42
@app.get("/books/{book_id}")
def get_book(book_id: int) -> dict:
    """根据 ID 查询一本书。

    FastAPI 看到 book_id: int 后会自动：
    1. 把 URL 里的字符串 "3" 转成整数 3
    2. 如果有人传 /books/abc，自动返回 422 错误（类型不对）
    """
    # ✍️ 假装这是一个数据库，实际是硬编码的书库
    books = {
        1: {"title": "三体", "author": "刘慈欣", "year": 2008},
        2: {"title": "红楼梦", "author": "曹雪芹", "year": 1791},
        3: {"title": "百年孤独", "author": "马尔克斯", "year": 1967},
    }

    if book_id not in books:
        return {"error": f"没有 ID 为 {book_id} 的书"}

    return books[book_id]


# ═══════════════════════════════════════════════════════════
# 下面三个端点是有意写错的"教学反例"——逐个展示 FastAPI 能抓出什么问题
# ═══════════════════════════════════════════════════════════


# ❌ 反例 1：没有类型注解 —— FastAPI 无法校验
#   对比 /books/abc → 自动 422
#   这里 /bad-book/abc → book_id 是字符串 "abc"，静默通过
@app.get("/bad-book/{book_id}")
def get_book_bad(book_id):  # ← 注意！没有 :int
    """反例 1：路径参数没写类型，FastAPI 不会自动校验。"""
    # 这里 book_id 始终是字符串，即使你写的 URL 是 /bad-book/3
    # book_id 也是 "3"（字符串），不是 3（整数）
    books = {
        "1": {"title": "三体", "author": "刘慈欣", "year": 2008},
    }
    return books[book_id]


# ❌ 反例 2：不检查数据是否存在 —— 程序直接炸
#   请求一个不存在的 ID，FastAPI 服务端会报 500 Internal Server Error
@app.get("/bad-book2/{book_id}")
def get_book_crash(book_id: int) -> dict:
    """反例 2：没有检查 book_id 是否存在，直接返回。

    访问 /bad-book2/999 → KeyError，服务端 500 错误。
    你什么都没做错，但看到了 500，这就是"未处理异常"的后果。
    """
    books = {1: {"title": "三体"}}
    return books[book_id]  # ← 如果 book_id 不在 books 里，直接 KeyError！


# ❌ 反例 3：返回类型和声明的不一致 —— MyPy 能抓
#   声明了 -> dict，但实际返回了 str
@app.get("/bad-book3")
def get_book_wrong_type() -> dict:
    """反例 3：返回值类型和声明不匹配。

    声明 -> dict，实际返回字符串。FastAPI 运行时不报错（JSON 序列化什么都接受），
    但 MyPy 静态检查时会抓到这个矛盾。
    """
    return "这应该是个字典但我偏要返回字符串"  # ← 声明说返回 dict，实际是 str
