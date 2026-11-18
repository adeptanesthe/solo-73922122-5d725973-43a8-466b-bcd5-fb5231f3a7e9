# -*- coding: utf-8 -*-
"""断句 API 测试：长度校验、原文不变量、不重复标点、复核语义、校准信息。"""
from __future__ import annotations

import unicodedata

import pytest
from fastapi.testclient import TestClient

from app.articles_data import ARTICLES
from app import main

TERMINALS = "。.．！!？?；;"


def _strip(s: str) -> str:
    return "".join(ch for ch in s if ch not in TERMINALS)


def _content_len(s: str) -> int:
    return sum(1 for ch in s if not unicodedata.category(ch).startswith(("Z", "P")))


@pytest.fixture()
def client():
    with TestClient(main.app) as c:
        yield c


def _manuscript(count: int = 2) -> str:
    text = "".join(_strip(ARTICLES[i]["body"]) for i in range(count))
    assert 120 <= _content_len(text) <= 350
    return text


def test_segment_info_reports_calibration(client):
    r = client.get("/api/segment/info")
    assert r.status_code == 200
    data = r.json()
    assert data["training_articles"] == 30
    assert data["training_boundaries"] == 150
    assert data["min_input_chars"] == 120
    assert data["max_input_chars"] == 350
    assert data["min_segment_len"] == 15
    assert data["max_segment_len"] == 90


def test_segment_endpoint_round_trip(client):
    text = _manuscript()
    r = client.post("/api/segment", json={"text": text})
    assert r.status_code == 200
    data = r.json()
    # 建议稿删掉新增句号后逐字等于原文
    out = data["suggested_text"]
    sug = sorted((i for i in data["insertions"] if i["confidence"] == "suggested"),
                 key=lambda x: x["source_index"])
    for order, ins in reversed(list(enumerate(sug))):
        pos = ins["source_index"] + sum(1 for q in sug[:order] if q["confidence"] == "suggested")
        assert out[pos] == "。"
        out = out[:pos] + out[pos + 1:]
    assert out == text


def test_segment_insertions_only_content_gaps(client):
    text = _manuscript()
    data = client.post("/api/segment", json={"text": text}).json()
    for ins in data["insertions"]:
        assert text[ins["source_index"] - 1] == ins["left_char"]
        assert text[ins["source_index"]] == ins["right_char"]
        assert unicodedata.category(ins["left_char"]).startswith("L")
        assert unicodedata.category(ins["right_char"]).startswith("L")
        assert ins["confidence"] in ("suggested", "review")


def test_segment_respects_length_bounds(client):
    data = client.post("/api/segment", json={"text": _manuscript()}).json()
    assert all(15 <= L <= 90 for L in data["segment_lengths"])


def test_existing_punctuation_not_duplicated(client):
    text = "先交代背景与基本概念。" + _strip(ARTICLES[1]["body"])
    data = client.post("/api/segment", json={"text": text}).json()
    punct = {i for i, ch in enumerate(text) if unicodedata.category(ch).startswith("P")}
    for ins in data["insertions"]:
        assert ins["source_index"] not in punct
    # 建议稿句号 = 原文句号 + 新增建议句号
    assert data["suggested_text"].count("。") == text.count("。") + data["suggested_count"]


def test_review_positions_have_no_period(client):
    text = _manuscript()
    data = client.post("/api/segment", json={"text": text}).json()
    review = {i["source_index"] for i in data["insertions"] if i["confidence"] == "review"}
    suggested = {i["source_index"] for i in data["insertions"] if i["confidence"] == "suggested"}
    assert not (review & suggested)
    assert data["suggested_count"] + data["review_count"] == len(data["insertions"])


def test_segment_validation_errors(client):
    r1 = client.post("/api/segment", json={"text": "短" * 100})
    assert r1.status_code == 422
    r2 = client.post("/api/segment", json={"text": "长" * 400})
    assert r2.status_code == 422
    r3 = client.post("/api/segment", json={"text": "   "})
    assert r3.status_code == 422


def test_segment_does_not_affect_recommendation_state(client):
    # 断句是无状态推断，不应改动推荐系统的任何状态
    before = client.get("/api/status").json()
    client.post("/api/segment", json={"text": _manuscript()})
    after = client.get("/api/status").json()
    assert before["state"] == after["state"]
    assert before["job"] == after["job"]
