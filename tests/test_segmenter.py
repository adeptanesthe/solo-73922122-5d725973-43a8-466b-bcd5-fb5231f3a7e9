# -*- coding: utf-8 -*-
"""断句模块测试：语料学习、间隙合法性、原文不变量、长度约束、复核语义与确定性。"""
from __future__ import annotations

import unicodedata

import pytest

from app.articles_data import ARTICLES
from app.segmenter import (
    BoundaryModel,
    SegmentationError,
    get_boundary_model,
    segment_text,
)

TERMINALS = "。.．！!？?；;"


def _strip_terminal(s: str) -> str:
    return "".join(ch for ch in s if ch not in TERMINALS)


def _content_len(s: str) -> int:
    return sum(1 for ch in s if not unicodedata.category(ch).startswith(("Z", "P")))


def _make_manuscript(start: int = 0, count: int = 2) -> str:
    """拼若干篇正文并去掉句末标点，得到 120~350 字的无句末标点稿件。"""
    text = "".join(_strip_terminal(ARTICLES[(start + i) % len(ARTICLES)]["body"])
                   for i in range(count))
    assert 120 <= _content_len(text) <= 350
    return text


def _remove_inserted(original: str, result) -> str:
    """按建议插入位置从建议稿中删掉句号，应无损还原原文。"""
    out = result.suggested_text
    sug = sorted((i for i in result.insertions if i.is_suggested),
                 key=lambda x: x.source_index)
    for order, ins in reversed(list(enumerate(sug))):
        pos = ins.source_index + sum(1 for q in sug[:order] if q.is_suggested)
        assert out[pos] == "。"
        out = out[:pos] + out[pos + 1:]
    return out


def test_model_learns_from_all_30_articles():
    model = get_boundary_model()
    assert model.article_count == 30
    # 30 篇 × 每篇 5 句 = 150 个句末边界
    assert model.n_boundary == 150
    assert 0 < model.prior < 0.1  # 边界是稀有间隙
    assert 25 <= model.mean_len <= 35
    assert model.calibration_sample == 150
    # 阈值是有限实数
    assert model.threshold == model.threshold


def test_threshold_calibrated_only_from_articles():
    # 阈值在处理任何用户稿件之前就已由语料确定，且处理稿件后不改变
    model = BoundaryModel()
    before = model.threshold
    model.segment(_make_manuscript())
    model.segment(_make_manuscript(start=4))
    assert model.threshold == before


def test_suggested_text_round_trips_to_original():
    text = _make_manuscript()
    result = segment_text(text)
    assert _remove_inserted(text, result) == text


def test_insertions_only_at_content_gaps():
    text = _make_manuscript()
    result = segment_text(text)
    for ins in result.insertions:
        # 间隙两侧必须是原文字符且都是内容字（非标点、非空白）
        assert text[ins.source_index - 1] == ins.left_char
        assert text[ins.source_index] == ins.right_char
        assert unicodedata.category(ins.left_char).startswith("L")
        assert unicodedata.category(ins.right_char).startswith("L")


def test_no_insertion_at_existing_punctuation():
    # 含残留句末标点与句内标点的稿件：已有标点间隙绝不产生插入
    text = "先交代背景与基本概念。" + _strip_terminal(ARTICLES[1]["body"])
    result = segment_text(text)
    punct_positions = {i for i, ch in enumerate(text) if unicodedata.category(ch).startswith("P")}
    for ins in result.insertions:
        assert ins.source_index not in punct_positions
    # 残留句号仍在建议稿中，且数量不减少、不重复
    assert result.suggested_text.count("。") == text.count("。") + result.suggested_count
    assert _remove_inserted(text, result) == text


def test_new_segments_respect_15_to_90_chars():
    text = _make_manuscript()
    result = segment_text(text)
    # 系统新切出的句段（插入项记录的段长）都在 15~90
    for ins in result.insertions:
        assert 15 <= ins.segment_length <= 90
    assert result.segment_lengths and all(15 <= L <= 90 for L in result.segment_lengths)


def test_review_positions_not_forced_with_period():
    text = _make_manuscript()
    result = segment_text(text)
    # 待复核位置不出现在建议稿的新增句号中
    suggested_positions = {i.source_index for i in result.insertions if i.is_suggested}
    review_positions = {i.source_index for i in result.insertions if not i.is_suggested}
    assert not (suggested_positions & review_positions)
    # 建议稿句号增量恰好等于建议数（复核位不插标点）
    assert result.suggested_text.count("。") - text.count("。") == result.suggested_count
    # 每个插入项判定取值合法
    assert all(i.confidence in ("suggested", "review") for i in result.insertions)


def test_evidence_below_threshold_marks_review():
    model = get_boundary_model()
    text = _make_manuscript()
    result = model.segment(text)
    for ins in result.insertions:
        if ins.confidence == "suggested":
            assert ins.evidence >= result.threshold
        else:
            assert ins.evidence < result.threshold


def test_deterministic_output():
    text = _make_manuscript(start=7)
    r1 = segment_text(text)
    r2 = segment_text(text)
    assert r1.suggested_text == r2.suggested_text
    assert [i.gap_id for i in r1.insertions] == [i.gap_id for i in r2.insertions]
    assert r1.segment_lengths == r2.segment_lengths


def test_original_characters_and_order_preserved():
    text = _make_manuscript()
    result = segment_text(text)
    # 删掉建议稿里所有新增句号后，与原文逐字相等（更强的字符/顺序校验）
    stripped = result.suggested_text.replace("。", "")
    # 原文本身无句号，故全部句号都是新增的
    assert "。" not in text
    assert stripped == text


def test_input_length_validation():
    with pytest.raises(SegmentationError):
        segment_text("短" * 100)
    with pytest.raises(SegmentationError):
        segment_text("长" * 400)
    with pytest.raises(SegmentationError):
        segment_text("   ")
    with pytest.raises(SegmentationError):
        segment_text("")


def test_boundary_lengths_accepted():
    # 恰好 120 与 350 内容字（无标点人造稿）应可处理，且不报错、可还原
    for n in (120, 350):
        text = ("甲乙丙丁" * (n // 4))[:n]
        result = segment_text(text)
        assert _remove_inserted(text, result) == text
        # 人造稿无字面证据，新增句号应保守（全部或多数进入复核，不强行插入）
        assert result.suggested_count <= result.review_count + 1


def test_terminal_punctuation_in_corpus_is_learned():
    model = get_boundary_model()
    # 语料每篇恰好 5 句，模型应学到接近 5 的每篇句数
    assert round(model.mean_sentences) == 5


def test_inner_punctuation_gaps_never_eligible():
    model = get_boundary_model()
    text = _make_manuscript()
    result = model.segment(text)
    for ins in result.insertions:
        # 左字不可能是句内标点（该间隙不可插入）
        assert ins.left_char not in "，、,：:；;"
        assert ins.right_char not in "，、,：:；;"
