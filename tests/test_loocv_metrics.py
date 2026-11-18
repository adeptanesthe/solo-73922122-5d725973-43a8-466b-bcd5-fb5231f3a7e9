# -*- coding: utf-8 -*-
"""留一（LOOCV）泛化评测：较慢，默认跳过，用 `pytest --run-loocv` 显式执行。

复核 README“小语料下的能力边界”一节引用的数字：
  * 间隙证据分 AUC 约 0.65；
  * 端到端 DP 边界精确 F1 约 0.36、±2 字近命中率约 0.69；
  * 实际建议插入（过可信阈值）位置的精度约 0.67。
"""
from __future__ import annotations

import pytest

from app.articles_data import ARTICLES
from app.segmenter import BoundaryModel, _parse

TERMINALS = "。.．！!？?；;"


def _strip(s: str) -> str:
    return "".join(c for c in s if c not in TERMINALS)


def _evidence_auc_loocv() -> float:
    pos_scores, neg_scores = [], []
    for i in range(len(ARTICLES)):
        model = BoundaryModel(ARTICLES[:i] + ARTICLES[i + 1:])
        parsed = _parse(ARTICLES[i]["body"])
        seq, kg = parsed.content, parsed.kind
        terms = {g for g, k in kg.items() if k == "terminal"}
        if parsed.trailing_terminal:
            terms.add(len(seq) - 1)
        for g in range(1, len(seq)):
            if kg.get(g) == "inner":
                continue
            score = model.evidence(seq[g - 1], seq[g])
            (pos_scores if g in terms else neg_scores).append(score)
    gt = sum(1 for p in pos_scores for q in neg_scores if p > q)
    eq = sum(1 for p in pos_scores for q in neg_scores if p == q)
    return (gt + 0.5 * eq) / (len(pos_scores) * len(neg_scores))


def _e2e_loocv() -> tuple[float, float, float]:
    tp = fp = fn = 0
    near = total_true = 0
    sug_tp = sug_fp = 0
    for i in range(len(ARTICLES)):
        model = BoundaryModel(ARTICLES[:i] + ARTICLES[i + 1:])
        body = ARTICLES[i]["body"]
        parsed = _parse(body)
        true_gaps = {g for g, k in parsed.kind.items() if k == "terminal"}
        if parsed.trailing_terminal:
            true_gaps.add(len(parsed.content))
        result = model.segment(_strip(body))
        chosen = {ins.gap_id for ins in result.insertions}
        true_inner = {g for g in true_gaps if g < len(parsed.content)}
        tp += len(chosen & true_inner)
        fp += len(chosen - true_inner)
        fn += len(true_inner - chosen)
        for g in true_inner:
            total_true += 1
            if any(abs(g - c) <= 2 for c in chosen):
                near += 1
        sug = {ins.gap_id for ins in result.insertions if ins.is_suggested}
        sug_tp += len(sug & true_inner)
        sug_fp += len(sug - true_inner)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    suggested_precision = sug_tp / (sug_tp + sug_fp) if sug_tp + sug_fp else 0.0
    return f1, near / total_true, suggested_precision


@pytest.mark.loocv
def test_loocv_generalization_metrics():
    auc = _evidence_auc_loocv()
    f1, near, suggested_precision = _e2e_loocv()
    # 宽松断言：守住“有判别力但保守”的量级，防止回归时静默劣化
    assert auc >= 0.60
    assert f1 >= 0.30
    assert near >= 0.60
    assert suggested_precision >= 0.55
