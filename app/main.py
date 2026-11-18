# -*- coding: utf-8 -*-
"""FastAPI 应用：推荐队列的条件管理、异步重查、分组揭示与演示页面。"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import database
from .articles_data import ARTICLES, TOPICS
from .segmenter import SegmentationError, get_boundary_model
from .tfidf_model import get_model
from .worker import get_worker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("recommender")

DEFAULT_INTEREST = "我想了解人工智能模型训练与数据驱动的分析方法"
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


class ConditionIn(BaseModel):
    interest: str | None = Field(default=None, description="兴趣短文")
    quotas: dict[str, int] | None = Field(default=None, description="各主题精确篇数")
    compute_delay_ms: int = Field(default=0, ge=0, le=30000)


class ExclusionIn(BaseModel):
    article_id: int = Field(ge=1)
    compute_delay_ms: int = Field(default=0, ge=0, le=30000)


class SegmentIn(BaseModel):
    text: str = Field(description="缺少句末标点的中文稿件，120~350 字（不含空白）")

def _validate_quotas(quotas: dict[str, int]) -> None:
    unknown = set(quotas) - set(TOPICS)
    if unknown:
        raise HTTPException(422, f"未知主题: {sorted(unknown)}")
    missing = set(TOPICS) - set(quotas)
    if missing:
        raise HTTPException(422, f"缺少主题配额: {sorted(missing)}")
    if any(v < 0 for v in quotas.values()):
        raise HTTPException(422, "配额不能为负数")
    if sum(quotas.values()) != 6:
        raise HTTPException(422, f"六个主题配额之和必须为 6，当前为 {sum(quotas.values())}")


def _validate_interest(interest: str) -> None:
    if not interest or not interest.strip():
        raise HTTPException(422, "兴趣短文不能为空")
    if len(interest) > 2000:
        raise HTTPException(422, "兴趣短文过长（上限 2000 字）")


def _article_brief(a: dict) -> dict:
    return {"id": a["id"], "title": a["title"], "topic": a["topic"], "author": a["author"]}


def _status_payload() -> dict:
    state = database.get_state()
    job = database.get_latest_job()
    plan = None
    if job and job["status"] == "done" and job["generation"] == state["generation"]:
        plan = job["result"]
    return {
        "state": state,
        "job": job,
        "plan": plan,
        "topics": TOPICS,
        "articles": [_article_brief(a) for a in ARTICLES],
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    database.init_db()
    # 预热 TF-IDF（仅文章拟合），失败不影响启动，worker 内会再试
    try:
        get_model()
    except Exception:  # pragma: no cover
        log.exception("TF-IDF 预热失败")
    # 预热断句证据模型（仅在 30 篇文章正文上统计，纯标准库，失败不阻断启动）
    try:
        get_boundary_model()
    except Exception:  # pragma: no cover
        log.exception("断句模型预热失败")
    get_worker().start()
    try:
        yield
    finally:
        get_worker().stop()


app = FastAPI(title="技术编辑选题推荐系统", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/api/status")
def status() -> dict:
    return _status_payload()


@app.get("/api/articles/{article_id}")
def article_detail(article_id: int) -> dict:
    for a in ARTICLES:
        if a["id"] == article_id:
            return a
    raise HTTPException(404, "文章不存在")


@app.put("/api/condition")
def update_condition(body: ConditionIn) -> dict:
    """修改兴趣和/或配额，开启新世代并触发异步重算。"""
    current = database.get_state()
    new_quotas = body.quotas if body.quotas is not None else current["quotas"]
    _validate_quotas(new_quotas)
    new_interest = body.interest if body.interest is not None else current["interest"]
    _validate_interest(new_interest)

    # 与已揭示前缀的兼容性前置检查（详细不可行原因由重算任务给出）
    from .optimizer import solve_queue  # 局部导入避免循环
    model = get_model()
    check = solve_queue(
        relevance=model.relevance_vector(new_interest),
        quotas=new_quotas,
        prefix=current["revealed"],
        excluded_ids=set(current["excluded_ids"]),
    )
    if not check.feasible:
        raise HTTPException(409, f"新条件与已揭示前缀不兼容: {check.reason}")

    database.begin_new_generation(
        interest=new_interest,
        quotas=new_quotas,
        compute_delay_ms=body.compute_delay_ms,
    )
    return _status_payload()


@app.post("/api/exclusions")
def add_exclusion(body: ExclusionIn) -> dict:
    if not any(a["id"] == body.article_id for a in ARTICLES):
        raise HTTPException(404, "文章不存在")
    try:
        database.add_exclusion(body.article_id, body.compute_delay_ms)
    except ValueError as exc:
        raise HTTPException(409, str(exc))
    return _status_payload()


@app.delete("/api/exclusions/{article_id}")
def remove_exclusion(article_id: int, compute_delay_ms: int = 0) -> dict:
    try:
        database.remove_exclusion(article_id, compute_delay_ms)
    except ValueError as exc:
        raise HTTPException(409, str(exc))
    return _status_payload()


@app.post("/api/reveal")
def reveal() -> dict:
    """揭示下一组两篇；同条件同进度下并发只推进一次。"""
    try:
        outcome = database.reveal_next_two()
    except RuntimeError as exc:
        raise HTTPException(409, str(exc))
    payload = _status_payload()
    payload["reveal_outcome"] = outcome
    return payload


@app.post("/api/reset")
def reset_demo() -> dict:
    """重置演示：清空揭示、排除与任务，世代回到 1。"""
    database.reset_all(DEFAULT_INTEREST, {t: 1 for t in TOPICS})
    return _status_payload()


# ---------------- 断句建议（句末标点恢复） ----------------


@app.get("/api/segment/info")
def segment_info() -> dict:
    """返回断句模型的校准信息（阈值来源、语料统计、约束常量），供页面展示。"""
    model = get_boundary_model()
    return {
        "training_articles": model.article_count,
        "training_boundaries": model.n_boundary,
        "calibration_sample": model.calibration_sample,
        "threshold": round(float(model.threshold), 4),
        "review_quantile": 0.25,
        "mean_sentence_len": round(float(model.mean_len), 2),
        "mean_sentences_per_article": round(float(model.mean_sentences), 2),
        "min_segment_len": 15,
        "max_segment_len": 90,
        "min_input_chars": 120,
        "max_input_chars": 350,
    }


@app.post("/api/segment")
def segment_manuscript(body: SegmentIn) -> dict:
    """对缺少句末标点的稿件给出整篇断句建议。

    只在两个内容字直接相邻、且没有任何标点的字符间隙插入句号；原文已有标点
    （含残留句末标点）的位置绝不重复插入。证据不足的边界标记为待复核，
    建议稿中不插入句号。
    """
    try:
        result = get_boundary_model().segment(body.text)
    except SegmentationError as exc:
        raise HTTPException(422, str(exc))
    return {
        "original_text": result.original_text,
        "suggested_text": result.suggested_text,
        "threshold": result.threshold,
        "segment_lengths": result.segment_lengths,
        "stats": result.stats,
        "insertions": [
            {
                "gap_id": ins.gap_id,
                "source_index": ins.source_index,
                "left_char": ins.left_char,
                "right_char": ins.right_char,
                "evidence": ins.evidence,
                "confidence": ins.confidence,
                "segment_length": ins.segment_length,
            }
            for ins in result.insertions
        ],
        "suggested_count": result.suggested_count,
        "review_count": result.review_count,
    }
