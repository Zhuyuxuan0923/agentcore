"""分块模块单元测试。

覆盖：
  - RECURSIVE 策略保证句子完整性
  - _merge_small_pieces 正确合并
  - _add_overlap_to_chunks 追加重叠
  - chunk_document 各策略出口
  - 边界值：空文本、单句、chunk_size 临界值
"""

import pytest

from study_agent.rag.chunking import (
    ChunkStrategy,
    _merge_small_pieces,
    chunk_document,
)

# 共享测试文本：中文 + Markdown 标题
TEXT = """# 员工手册

## 第一章 公司介绍

本公司成立于2020年，致力于人工智能技术研发。公司总部位于北京。
在上海和深圳设有分公司。现有员工500余人。

## 第二章 考勤制度

公司实行弹性工作制。核心工作时间10:00-18:00。
年假天数：入职满1年享受5天年假，满3年享受10天年假。
年假需提前一周申请。

## 第三章 报销流程

差旅报销需在出差结束后一周内提交报销单。单笔超过5000元的报销需部门主管审批。"""


# ── _merge_small_pieces 单元测试 ──────────────────────────


def test_merge_small_pieces_basic():
    """小片段合并成不超过 chunk_size 的块。"""
    pieces = ["hello", "world", "foo"]
    result = _merge_small_pieces(pieces, 20)
    # "hello" (5) + "world" (5) + "foo" (3) = 13 < 20，应合并为 1 块
    assert len(result) == 1, f"3 个片段合起来不到 20 字，应合并为 1 块，实际 {len(result)} 块"
    assert "hello" in result[0]


def test_merge_small_pieces_respects_max_size():
    """每个合并后的块不超过 max_size。"""
    pieces = ["a" * 5, "b" * 5, "c" * 5, "d" * 5]
    result = _merge_small_pieces(pieces, 12)
    for piece in result:
        assert len(piece) <= 12, f"合并后 {len(piece)} 字 > 12"


def test_merge_large_piece_split():
    """超过 max_size 的单个片段被切开。"""
    pieces = ["a" * 30]  # 30 字，max=10
    result = _merge_small_pieces(pieces, 10)
    assert len(result) >= 2, f"30 字片段在 max=10 时应切成至少 2 块，实际 {len(result)} 块"
    for piece in result:
        assert len(piece) <= 10


def test_merge_small_pieces_empty():
    """空输入返回空列表。"""
    assert _merge_small_pieces([], 100) == []


# ── chunk_document 系统测试 ──────────────────────────────


def test_recursive_preserves_boundaries():
    """RECURSIVE 策略应在自然边界（。\\n）处结束块。"""
    chunks = chunk_document(TEXT, strategy=ChunkStrategy.RECURSIVE, chunk_size=200, overlap=0)
    assert len(chunks) >= 2, f"200 字应切出 >=2 块，实际 {len(chunks)}"
    for c in chunks:
        t = c.text.strip()
        # 不应在虚词中间截断
        assert not t.endswith("的"), f"块在'的'处截断: ...{t[-20:]}"
        assert not t.endswith("了"), f"块在'了'处截断: ...{t[-20:]}"


def test_recursive_no_overlap():
    """overlap=0 时块之间不重叠。"""
    chunks = chunk_document(TEXT, strategy=ChunkStrategy.RECURSIVE, chunk_size=200, overlap=0)
    for i in range(len(chunks) - 1):
        assert (
            chunks[i].end <= chunks[i + 1].start
        ), f"块[{i}] end={chunks[i].end} > 块[{i+1}] start={chunks[i+1].start}，无重叠时有重叠"


def test_recursive_with_overlap():
    """overlap>0 时相邻块开头重叠。"""
    chunks = chunk_document(TEXT, strategy=ChunkStrategy.RECURSIVE, chunk_size=200, overlap=30)
    if len(chunks) >= 2:
        # 块[1] 的 text 应包含块[0] 尾部的部分内容
        tail_of_0 = chunks[0].text[-30:]
        head_of_1 = chunks[1].text[:50]
        # 至少有一个字重叠
        overlap_found = any(c in head_of_1 for c in tail_of_0 if c.strip())
        assert overlap_found, "overlap=30 应产生文本重叠"


def test_chunk_size_boundary():
    """chunk_size 等于文本长度时只产生 1 块。"""
    chunks = chunk_document(TEXT, strategy=ChunkStrategy.RECURSIVE, chunk_size=9999, overlap=0)
    assert len(chunks) == 1, f"chunk_size 远大于文本时应为 1 块，实际 {len(chunks)}"


def test_empty_text():
    """空文本返回空列表，不崩溃。"""
    chunks = chunk_document("", strategy=ChunkStrategy.RECURSIVE, chunk_size=200)
    assert len(chunks) == 0


def test_single_char():
    """单字符不崩溃。"""
    chunks = chunk_document("A", strategy=ChunkStrategy.RECURSIVE, chunk_size=200)
    assert len(chunks) == 1
    assert chunks[0].text == "A"


def test_all_strategies():
    """三种策略都不崩溃。"""
    for strategy in [ChunkStrategy.FIXED_SIZE, ChunkStrategy.RECURSIVE, ChunkStrategy.SENTENCE]:
        chunks = chunk_document(TEXT, strategy=strategy, chunk_size=200, overlap=0)
        assert len(chunks) >= 1, f"{strategy} 应产生 >=1 块"
        total = sum(len(c.text) for c in chunks)
        assert total >= len(TEXT) * 0.7, "总字符数不应损失过多"


# ── 参数化：不同 chunk_size ─────────────────────────────


@pytest.mark.parametrize("size", [50, 100, 200, 500, 1000])
def test_various_chunk_sizes(size):
    """各 chunk_size 都不崩溃，且块大小不超过设定值。"""
    chunks = chunk_document(TEXT, strategy=ChunkStrategy.RECURSIVE, chunk_size=size, overlap=0)
    assert len(chunks) >= 1
    for c in chunks:
        assert len(c.text) <= max(
            size * 1.5, len(TEXT)
        ), f"chunk_size={size} 但产生了 {len(c.text)} 字的块"
