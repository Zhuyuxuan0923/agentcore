"""文件上传 API。

POST /api/upload
  - 接收 PDF/DOCX/MD/TXT 文件
  - 解析 -> 分块 -> Embedding -> 存入 ChromaDB
  - 返回索引结果（文件名、分块数、所属知识库）
"""

from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

router = APIRouter(prefix="/api", tags=["upload"])

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".md", ".txt", ".markdown"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB


def get_agent():
    """延迟获取 Agent 实例（避免循环导入）。"""
    from study_agent.api.app import get_agent as ga

    return ga()


@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    kb_name: str = Form(default="默认知识库"),
):
    """上传文档并索引到知识库。

    文件格式支持: PDF, DOCX, Markdown, TXT
    最大文件大小: 10MB
    """
    # 校验文件后缀
    original_name = file.filename or "unknown"
    suffix = Path(original_name).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式 '{suffix}'。支持的格式: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    # 保存到临时文件，同时检查大小
    temp_path = UPLOAD_DIR / original_name
    try:
        total_size = 0
        with open(temp_path, "wb") as f:
            while True:
                chunk = file.file.read(8192)
                if not chunk:
                    break
                total_size += len(chunk)
                if total_size > MAX_FILE_SIZE:
                    f.close()
                    temp_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=400,
                        detail=f"文件过大 (>{MAX_FILE_SIZE // 1024 // 1024}MB)，请压缩后重试",
                    )
                f.write(chunk)

        if total_size == 0:
            temp_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=400,
                detail="文件为空，请上传包含内容的文件",
            )

        # 调用 Agent 上传
        agent = get_agent()
        result = agent.upload(str(temp_path), kb_name=kb_name)
        return {
            "status": "ok",
            "message": (
                f"文件 '{result['file_name']}' 已索引到知识库 "
                f"'{result['kb_name']}'，共 {result['chunk_count']} 个文档块"
            ),
            **result,
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="文件不存在")
    except ImportError as e:
        raise HTTPException(
            status_code=500,
            detail=f"缺少依赖: {e}。请用 poetry add 安装相应库。",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件处理失败: {e}")
