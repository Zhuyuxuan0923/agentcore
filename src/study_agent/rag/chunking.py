"""文档分块策略 -- 从 w3d3_chunking_demo.py 重构。

三种策略：
  - FIXED_SIZE: 按字符数等距切分，最快但可能在句子中间截断
  - RECURSIVE: 按分隔符优先级(段落 -> 句子 -> 词)逐级切分，推荐默认
  - SENTENCE: 保留完整句子，在换行处切分

chunk_overlap: 相邻块之间共享的字符数。
  例: overlap=50 -> 每块末尾 50 字 = 下一块开头 50 字。
  作用: 防止关键信息刚好落在分块边界上，导致两个块都搜不到。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class ChunkStrategy(StrEnum):
    FIXED_SIZE = "fixed_size"
    RECURSIVE = "recursive"
    SENTENCE = "sentence"


@dataclass
class Chunk:
    text: str
    start: int
    end: int
    strategy: str
    metadata: dict = field(default_factory=dict)


def _merge_small_pieces(pieces: list[str], max_size: int) -> list[str]:
    """将相邻的小片段合并，尽量填满 max_size。

    原则：在不超出 max_size 的前提下，相邻片段尽可能合并成一个块。
    这样每个块的信息密度更高，不会被切成零碎的短句。

    Args:
        pieces: 已按自然边界切分的文本片段列表
        max_size: 合并后的块不超过此大小

    Returns:
        合并后的片段列表
    """
    if not pieces:
        return []

    merged: list[str] = []
    buffer = ""
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue

        # 如果当前片段本身已超过 max_size，先清空 buffer，再单独处理
        if len(piece) > max_size:
            if buffer:
                merged.append(buffer)
                buffer = ""
            # 长片段按 max_size 切分
            for i in range(0, len(piece), max_size):
                sub = piece[i : i + max_size].strip()
                if sub:
                    merged.append(sub)
            continue

        # 尝试合并到 buffer
        sep = "" if not buffer else " "
        if len(buffer) + len(sep) + len(piece) <= max_size:
            buffer = buffer + sep + piece if buffer else piece
        else:
            merged.append(buffer)
            buffer = piece

    if buffer:
        merged.append(buffer)
    return merged


def _add_overlap_to_chunks(chunks: list[Chunk], text: str, overlap: int) -> list[Chunk]:
    """在已有分块基础上添加重叠区域。

    每个块从其前一个块的"结尾处往前 overlap 字符"开始，
    确保边界处的信息同时出现在相邻两个块中。

    注意：重叠只影响 chunk.text 的内容，不影响 start/end 位置标记。
    这样引用时仍然可以定位到段落在原文中的准确位置。
    """
    if not chunks or overlap <= 0:
        return chunks

    result: list[Chunk] = []
    for i, chunk in enumerate(chunks):
        if i == 0:
            result.append(chunk)
            continue

        prev = result[-1]
        # 前一块末尾 overlap 字符作为当前块的前缀
        prev_end = prev.start + (prev.end - prev.start)
        overlap_start = max(prev.start, prev_end - overlap)
        if overlap_start < prev_end:
            prefix = text[overlap_start:prev_end]
            # 在自然边界处截断（避免从词中间开始）
            if len(prefix) > 10:
                cut = max(prefix.find(" "), prefix.find("\n"), prefix.find("。"), prefix.find("."))
                if cut > len(prefix) // 2:
                    prefix = prefix[cut + 1 :]

            new_text = prefix + chunk.text if prefix.strip() else chunk.text
            chunk = Chunk(
                text=new_text,
                start=chunk.start,
                end=chunk.end,
                strategy=chunk.strategy,
            )

        result.append(chunk)

    return result


def fixed_size_chunk(text: str, chunk_size: int = 500) -> list[Chunk]:
    """策略 A：按固定字符数等距切分。"""
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk_text = text[i : i + chunk_size]
        if chunk_text.strip():
            chunks.append(
                Chunk(
                    text=chunk_text,
                    start=i,
                    end=min(i + chunk_size, len(text)),
                    strategy=ChunkStrategy.FIXED_SIZE.value,
                )
            )
    return chunks


def recursive_chunk(text: str, chunk_size: int = 500) -> list[Chunk]:
    """策略 B：按分隔符优先级递归切分 + 合并小片段。

    优先级: 双换行 > 单换行 > 句号 > 空格 > 固定大小
    切分后：合并相邻小片段，尽量让每个块信息密度更高。

    与旧版的区别：
      - 旧版：按分隔符切完后，每个句子独立成一个块（10 字也一块）
      - 新版：相邻片段合并填满 chunk_size（10 字 + 50 字 + 30 字 -> 一块 90 字）
    """
    separators = ["\n\n", "\n", "。", ".", " ", ""]

    def _split(t: str, sep_idx: int = 0) -> list[str]:
        if sep_idx >= len(separators):
            return [t[i : i + chunk_size] for i in range(0, len(t), chunk_size)]

        sep = separators[sep_idx]
        if sep and sep not in t:
            return _split(t, sep_idx + 1)

        if not sep:
            return [t[i : i + chunk_size] for i in range(0, len(t), chunk_size)]

        parts = t.split(sep)
        result = []
        for part in parts:
            if len(part) <= chunk_size:
                result.append(part)
            else:
                result.extend(_split(part, sep_idx + 1))
        return result

    pieces = _split(text)
    # 合并小片段
    merged = _merge_small_pieces(pieces, chunk_size)

    chunks = []
    pos = 0
    for piece in merged:
        idx = text.find(piece, pos)
        if idx == -1:
            idx = pos
        chunks.append(
            Chunk(
                text=piece,
                start=idx,
                end=idx + len(piece),
                strategy=ChunkStrategy.RECURSIVE.value,
            )
        )
        pos = idx + len(piece)
    return chunks


def sentence_chunk(text: str, max_chunk_size: int = 500) -> list[Chunk]:
    """策略 C：在换行处切分，尽量保持语义段落完整。"""
    boundaries = []
    for i, ch in enumerate(text):
        if ch in "\n" and i > 0:
            boundaries.append(i + 1)

    if not boundaries:
        return [Chunk(text=text, start=0, end=len(text), strategy=ChunkStrategy.SENTENCE.value)]

    chunks = []
    start = 0
    for boundary in boundaries:
        if boundary - start > max_chunk_size:
            chunk_text = text[start:boundary].strip()
            if chunk_text:
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        start=start,
                        end=boundary,
                        strategy=ChunkStrategy.SENTENCE.value,
                    )
                )
            start = boundary

    if start < len(text):
        chunk_text = text[start:].strip()
        if chunk_text:
            chunks.append(
                Chunk(
                    text=chunk_text,
                    start=start,
                    end=len(text),
                    strategy=ChunkStrategy.SENTENCE.value,
                )
            )
    return chunks


def chunk_document(
    text: str,
    strategy: ChunkStrategy = ChunkStrategy.RECURSIVE,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[Chunk]:
    """统一的文档分块入口。

    流程：先按策略切分 -> 重叠 -> 完成

    [修复] 旧版在 overlap > 0 时会用 chunk_with_overlap (纯字符级滑动窗口)
    替换掉策略切分的结果，导致句子被拦腰截断。新版改为在策略切分结果上
    追加重叠，保持自然边界的同时实现信息冗余。

    Args:
        text: 原始文档文本
        strategy: 分块策略（默认 RECURSIVE，按段落/句子切分）
        chunk_size: 每块最大字符数（默认 500，旧版 200 偏小）
        overlap: 相邻块重叠字符数（默认 50，约 chunk_size 的 10%）

    Returns:
        list[Chunk]: 分块列表，每块的 text 可能包含与前一块重叠的内容
    """
    if strategy == ChunkStrategy.FIXED_SIZE:
        chunks = fixed_size_chunk(text, chunk_size)
    elif strategy == ChunkStrategy.RECURSIVE:
        chunks = recursive_chunk(text, chunk_size)
    elif strategy == ChunkStrategy.SENTENCE:
        chunks = sentence_chunk(text, chunk_size)
    else:
        chunks = recursive_chunk(text, chunk_size)

    if overlap > 0:
        chunks = _add_overlap_to_chunks(chunks, text, overlap)

    return chunks
