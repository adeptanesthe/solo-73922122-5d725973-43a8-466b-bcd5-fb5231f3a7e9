# -*- coding: utf-8 -*-
"""中文句末标点恢复（断句建议）核心模块，仅依赖标准库。

设计目标
========
技术编辑收到句末标点丢失的中文稿件，本模块从内置 30 篇文章正文的句末标点
学习"边界证据"，再结合 15~90 字的句长结构先验，对整篇稿件做全局断句：

* **只在缺少标点的原文字符间隙插入句号**：两个内容字直接相邻（中间无任何
  标点/空白）的间隙才是可插入间隙；逗号、顿号、冒号等句内标点所在间隙、
  引号括号间隙以及原文已有的句末标点位置一律不插入，已有标点绝不重复。
* **原有字符与顺序保持不变**：建议稿等于原文加若干个"。"，删除插入的句号后
  与原文逐字相等。
* **证据不足不强行加标点**：整篇动态规划选出边界后，证据分低于"由 30 篇
  语料校准出的可信阈值"的位置只标记为待复核（review），建议稿中不插入句号。
* **阈值完全由内置文章校准**：输入稿件只做推断，不参与任何拟合或调阈，
  与 TF-IDF "仅文章拟合" 的哲学一致。

证据模型（在 30 篇正文上学习）
------------------------------
对每个内容间隙 g（左字 x=content[g-1]、右字 y=content[g]）统计：

* 左单字 x、右单字 y、邻接二元组 x+y 在"句末边界间隙"与"全部间隙"上的计数；
* 以 add-k 平滑的条件对数几率（相对全局先验的增量）加权求和，得到证据分。

小语料下字面证据跨文章迁移有限（留一评测 AUC≈0.69），因此证据分只作为
结构 DP 的辅助项与最终的可信分级依据，断句位置主要由句长/句数结构决定。

结构先验与整篇 DP
-----------------
* 从语料学习句长均值 mu 与每篇句数；每段内容长度硬约束在 15~90 字；
* 原文残留的句末标点作为固定边界，把全文切成若干子区间；
* 每个子区间枚举候选句数，做分层动态规划，目标为
  ``证据分加权和 + 长度高斯奖励``，选出整篇全局最优的边界集合。
"""
from __future__ import annotations

import math
import threading
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable

from .articles_data import ARTICLES

# ---- 标点分类 ----
# 句末标点：边界本身，已存在则保留、不重复插入
TERMINAL_PUNCT = frozenset("。.．！!？?；;")
# 句内标点：所在间隙不可插入句号
INNER_PUNCT = frozenset("，、,：:")
# 引号、括号、省略号、破折号等：所在间隙同样不可插入
NEUTRAL_PUNCT = frozenset(
    "“”‘’「」『』（）()《》〈〉【】[]…—·~～\"'"
)

# ---- 业务约束（与需求/接口校验保持一致）----
MIN_SEGMENT_LEN = 15
MAX_SEGMENT_LEN = 90
MIN_INPUT_CHARS = 120
MAX_INPUT_CHARS = 350

# ---- 模型超参（由离线留一评测确定，详见 README）----
_EV_WEIGHTS = (0.35, 0.45, 0.20)      # 左单字 / 右单字 / 二元组
_EV_SMOOTH = (3.0, 3.0, 12.0)         # 对应 add-k 平滑强度
_EV_LAMBDA = 0.3                      # DP 中证据分相对长度奖励的权重
_LENGTH_SD = 6.0                      # 句长高斯先验标准差（字）
_REVIEW_QUANTILE = 0.25               # 可信阈值取语料真实边界证据分的 25 分位
_FALLBACK_MIN_LEN = 8                 # 严格下限不可行时的兜底放宽下限


class SegmentationError(ValueError):
    """输入稿件不满足断句前置条件。"""


@dataclass
class Insertion:
    """一个候选断句位置（对应原文中两个字符之间的间隙）。"""

    gap_id: int                 # 内容间隙编号 g，位于 content[g-1] 与 content[g] 之间
    source_index: int           # 原文插入点（左内容字在原文中的下标 + 1）
    left_char: str
    right_char: str
    evidence: float             # 边界证据分（对数几率尺度）
    confidence: str             # "suggested"（建议插入）或 "review"（待复核）
    segment_length: int         # 该边界结束句段的内容字数

    @property
    def is_suggested(self) -> bool:
        return self.confidence == "suggested"


@dataclass
class SegmentationResult:
    original_text: str
    suggested_text: str                     # 仅在 suggested 间隙插入"。"后的建议稿
    insertions: list[Insertion]
    segment_lengths: list[int]
    threshold: float
    stats: dict = field(default_factory=dict)

    @property
    def suggested_count(self) -> int:
        return sum(1 for i in self.insertions if i.is_suggested)

    @property
    def review_count(self) -> int:
        return sum(1 for i in self.insertions if not i.is_suggested)


def _is_punct(ch: str) -> bool:
    return unicodedata.category(ch).startswith("P")


def _is_space(ch: str) -> bool:
    return unicodedata.category(ch).startswith(("Z", "C"))


def _logit(p: float) -> float:
    p = min(1.0 - 1e-9, max(1e-9, p))
    return math.log(p / (1.0 - p))


def validate_input(text: str) -> str:
    """校验稿件长度（按非空白字符计数），返回去除首尾空白后的文本。"""
    if text is None:
        raise SegmentationError("稿件内容不能为空")
    cleaned = text.strip()
    if not cleaned:
        raise SegmentationError("稿件内容不能为空")
    n = sum(1 for ch in cleaned if not _is_space(ch))
    if n < MIN_INPUT_CHARS:
        raise SegmentationError(
            f"稿件需 {MIN_INPUT_CHARS}~{MAX_INPUT_CHARS} 字，当前仅 {n} 字（不含空白）"
        )
    if n > MAX_INPUT_CHARS:
        raise SegmentationError(
            f"稿件需 {MIN_INPUT_CHARS}~{MAX_INPUT_CHARS} 字，当前 {n} 字（不含空白），超出上限"
        )
    return cleaned


@dataclass
class _Parsed:
    content: list[str]                 # 内容字序列（剥离所有标点/空白）
    source: list[int]                  # 每个内容字在原文中的下标
    # 内容间隙 g(1..n-1) 的属性：terminal=原文已有句末标点；inner=句内标点；
    # neutral=引号等；eligible=两内容字直接相邻、可插入
    kind: dict[int, str]
    trailing_terminal: bool            # 原文以句末标点结尾（边界在全文最后一个内容字之后）


def _parse(text: str) -> _Parsed:
    content: list[str] = []
    source: list[int] = []
    kind: dict[int, str] = {}
    prev_content_idx = -1              # 上一个内容字在 content 中的下标
    pending = None                     # 自上个内容字以来见到的标点优先级
    trailing_terminal = False
    rank = {"neutral": 1, "inner": 2, "terminal": 3}

    for pos, ch in enumerate(text):
        if _is_space(ch):
            continue
        if ch in TERMINAL_PUNCT:
            cat = "terminal"
        elif ch in INNER_PUNCT:
            cat = "inner"
        elif ch in NEUTRAL_PUNCT or _is_punct(ch):
            cat = "neutral"
        else:
            # 内容字：结算它与上一个内容字之间的间隙
            cur = len(content)
            if prev_content_idx >= 0:
                gap = cur  # 位于 content[cur-1] 与 content[cur] 之间
                kind[gap] = pending if pending is not None else "eligible"
            content.append(ch)
            source.append(pos)
            prev_content_idx = cur
            pending = None
            trailing_terminal = False
            continue
        if pending is None or rank[cat] > rank[pending]:
            pending = cat
        if cat == "terminal":
            trailing_terminal = True
        elif cat in ("inner", "neutral"):
            trailing_terminal = False

    return _Parsed(
        content=content,
        source=source,
        kind=kind,
        trailing_terminal=trailing_terminal,
    )


class BoundaryModel:
    """从 30 篇文章正文学习的边界证据 + 句长结构模型（单例、线程安全初始化）。"""

    def __init__(self, articles: Iterable[dict] | None = None) -> None:
        articles = list(ARTICLES if articles is None else articles)
        self.article_count = len(articles)
        self.left_b: dict[str, int] = {}
        self.left_t: dict[str, int] = {}
        self.right_b: dict[str, int] = {}
        self.right_t: dict[str, int] = {}
        self.bi_b: dict[str, int] = {}
        self.bi_t: dict[str, int] = {}
        self.n_gap = 0
        self.n_boundary = 0
        seg_lengths: list[int] = []
        sents_per_article: list[int] = []
        boundary_scores: list[float] = []

        for art in articles:
            parsed = _parse(art["body"])
            seq, kg = parsed.content, parsed.kind
            terminals = {g for g, k in kg.items() if k == "terminal"}
            if parsed.trailing_terminal:
                terminals.add(len(seq))  # 文末句号：边界在最后一个内容字之后
            # 文末边界不对应“两内容字之间”的可插入间隙，单独计入边界总数
            self.n_boundary += 1 if parsed.trailing_terminal else 0
            for g in range(1, len(seq)):
                if kg.get(g) == "inner":
                    continue  # 句内标点间隙不纳入证据统计
                x, y = seq[g - 1], seq[g]
                lab = 1 if g in terminals else 0
                self.n_gap += 1
                self.n_boundary += lab
                self.left_t[x] = self.left_t.get(x, 0) + 1
                self.left_b[x] = self.left_b.get(x, 0) + lab
                self.right_t[y] = self.right_t.get(y, 0) + 1
                self.right_b[y] = self.right_b.get(y, 0) + lab
                bigram = x + y
                self.bi_t[bigram] = self.bi_t.get(bigram, 0) + 1
                self.bi_b[bigram] = self.bi_b.get(bigram, 0) + lab
            # 句长（内容字数）与每篇句数
            prev = 0
            for g in sorted(terminals):
                seg_lengths.append(g - prev)
                prev = g
            sents_per_article.append(len(terminals))

        self.prior = self.n_boundary / self.n_gap
        self.mean_len = sum(seg_lengths) / len(seg_lengths)
        self.mean_sentences = sum(sents_per_article) / len(sents_per_article)
        # 校准可信阈值：语料真实边界证据分的低分位（输入稿件不参与）
        for art in articles:
            parsed = _parse(art["body"])
            seq, kg = parsed.content, parsed.kind
            term_gaps = {g for g, k in kg.items() if k == "terminal"}
            # 文末句号绑定到最后一个内容间隙（n-1），其证据分取该间隙
            if parsed.trailing_terminal and len(seq) >= 2:
                term_gaps.add(len(seq) - 1)
            for g in term_gaps:
                if 0 < g < len(seq):
                    boundary_scores.append(self.evidence(seq[g - 1], seq[g]))
        boundary_scores.sort()
        q_idx = int(len(boundary_scores) * _REVIEW_QUANTILE)
        self.threshold = boundary_scores[min(q_idx, len(boundary_scores) - 1)]
        self.calibration_sample = len(boundary_scores)

    def _cond_logodds(self, cb: int, ct: int, smooth: float) -> float:
        p = (cb + smooth * self.prior) / (ct + smooth)
        return _logit(p) - _logit(self.prior)

    def evidence(self, x: str, y: str) -> float:
        """间隙 (x,y) 的边界证据分：全局先验对数几率 + 三项加权条件增量。"""
        wl, wr, wb = _EV_WEIGHTS
        al, ar, ab = _EV_SMOOTH
        return (
            _logit(self.prior)
            + wl * self._cond_logodds(self.left_b.get(x, 0), self.left_t.get(x, 0), al)
            + wr * self._cond_logodds(self.right_b.get(y, 0), self.right_t.get(y, 0), ar)
            + wb * self._cond_logodds(
                self.bi_b.get(x + y, 0), self.bi_t.get(x + y, 0), ab
            )
        )

    # ---------------- 整篇动态规划 ----------------

    def _length_reward(self, length: int) -> float:
        return -0.5 * ((length - self.mean_len) / _LENGTH_SD) ** 2

    def _dp_fixed_k(self, ev: list[float | None], eligible: set[int],
                    lo: int, hi: int, k: int, min_len: int
                    ) -> tuple[list[int], float] | None:
        """在内容区间 [lo, hi) 内恰好选 k 个 eligible 边界，最大化目标。

        边界 g 满足 lo < g < hi；每段内容长度在 min_len..MAX_SEGMENT_LEN 之间。
        dp[t][i] 表示：内容区间 [lo, i) 已用 t 个边界完整覆盖时的最优分，
        其中第 t 个边界就落在 i（i 同时是下一段的起点）。
        返回 (边界列表, 总分)；不可行返回 None。
        """
        neg = float("-inf")
        dp = [[neg] * (hi + 1) for _ in range(k + 1)]
        prev = [[-1] * (hi + 1) for _ in range(k + 1)]
        dp[0][lo] = 0.0
        for t in range(1, k + 1):
            # i = 第 t 个边界的位置（也是已覆盖前缀的终点）
            for i in range(lo + 1, hi + 1):
                if i != hi and i not in eligible:
                    continue  # 内部边界必须落在可插入间隙；hi 是区间终点而非边界
                best, best_j = neg, -1
                # 上一个边界 j（或区间起点 lo），末段为 [j, i)，长度 i-j
                low_j = max(lo, i - MAX_SEGMENT_LEN)
                high_j = i - min_len
                for j in range(low_j, high_j + 1):
                    if j != lo and j not in eligible:
                        continue
                    if dp[t - 1][j] == neg:
                        continue
                    seg_len = i - j
                    edge = _EV_LAMBDA * ev[i] if i != hi else 0.0
                    val = dp[t - 1][j] + edge + self._length_reward(seg_len)
                    if val > best:
                        best, best_j = val, j
                dp[t][i], prev[t][i] = best, best_j
        if dp[k][hi] == neg:
            return None
        chosen: list[int] = []
        t, i = k, hi
        while t > 0:
            j = prev[t][i]
            if j < 0:
                return None
            chosen.append(i)  # 第 t 个边界落在 i（hi 作为终点时 t 已先递减）
            i, t = j, t - 1
        chosen.reverse()
        # 回溯得到的最后一个元素可能是区间终点 hi（非真实边界），剔除
        if chosen and chosen[-1] == hi:
            chosen = chosen[:-1]
        return chosen, dp[k][hi]

    def _solve_range(self, ev: list[float | None], eligible: set[int],
                     lo: int, hi: int, min_len: int) -> list[int]:
        """子区间 [lo, hi) 内的最优边界（句数围绕语料均值估计，软择优）。"""
        span = hi - lo
        est = max(1, round(span / self.mean_len))
        best: tuple[list[int], float] | None = None
        for k in range(max(1, est - 2), est + 3):
            # k 个边界最多需要 (k+1)*min_len 字、最少 k*min_len+1
            if (k + 1) * min_len > span:
                break
            res = self._dp_fixed_k(ev, eligible, lo, hi, k, min_len)
            if res is None:
                continue
            chosen, score = res
            score -= 0.15 * abs(k - est)  # 句数软约束
            if best is None or score > best[1]:
                best = (chosen, score)
        if best is not None:
            return best[0]
        # 兜底：放宽下限再试
        if min_len > _FALLBACK_MIN_LEN:
            return self._solve_range(ev, eligible, lo, hi, _FALLBACK_MIN_LEN)
        # 最终兜底：等距切分（全部会因证据偏低而进入复核）
        step = max(min_len, int(self.mean_len))
        return [g for g in range(lo + step, hi, step) if g in eligible]

    def segment(self, text: str) -> SegmentationResult:
        cleaned = validate_input(text)
        parsed = _parse(cleaned)
        seq, kg, src = parsed.content, parsed.kind, parsed.source
        n = len(seq)

        eligible = {g for g in range(1, n) if kg.get(g) == "eligible"}
        fixed = sorted(g for g in range(1, n) if kg.get(g) == "terminal")
        if parsed.trailing_terminal:
            fixed.append(n)  # 文末句号作为最后一个固定边界
        fixed = sorted(set(fixed))

        ev: list[float | None] = [None] * (n + 1)
        for g in eligible:
            ev[g] = self.evidence(seq[g - 1], seq[g])

        # 原文残留句末标点作为固定边界，分段求解后合并
        bounds = sorted(set(fixed) | set(self._solve_ranges(ev, eligible, fixed, n)))

        insertions: list[Insertion] = []
        segment_lengths: list[int] = []
        suggested_gaps: set[int] = set()
        prev = 0
        for g in bounds:
            segment_lengths.append(g - prev)
            if g in eligible:  # 固定边界（原文已有标点）不产生插入项
                conf = "suggested" if ev[g] >= self.threshold else "review"
                if conf == "suggested":
                    suggested_gaps.add(g)
                insertions.append(
                    Insertion(
                        gap_id=g,
                        source_index=src[g - 1] + 1,
                        left_char=seq[g - 1],
                        right_char=seq[g],
                        evidence=round(float(ev[g]), 4),
                        confidence=conf,
                        segment_length=g - prev,
                    )
                )
            prev = g
        if prev < n:
            segment_lengths.append(n - prev)

        suggested_text = self._build_suggested_text(cleaned, insertions, suggested_gaps)
        stats = {
            "content_chars": n,
            "boundary_total": len(bounds),
            "fixed_terminal_gaps": len(fixed),
            "eligible_gaps": len(eligible),
            "mean_train_sentence_len": round(self.mean_len, 2),
            "mean_train_sentences_per_article": round(self.mean_sentences, 2),
            "review_quantile": _REVIEW_QUANTILE,
            "training_articles": self.article_count,
            "training_boundaries": self.n_boundary,
            "calibration_sample": self.calibration_sample,
        }
        return SegmentationResult(
            original_text=cleaned,
            suggested_text=suggested_text,
            insertions=insertions,
            segment_lengths=segment_lengths,
            threshold=round(float(self.threshold), 4),
            stats=stats,
        )

    def _solve_ranges(self, ev, eligible, fixed, n):
        """按原文残留句末标点（固定边界）切成子区间分别规划。

        最左侧前缀段（文章开头到第一个固定边界）由原文标点决定，放宽长度下限；
        其余区间内系统新切出的句段严格遵守 15~90 字。
        """
        chosen: list[int] = []
        start = 0
        bounds = fixed + [n]
        for idx, f in enumerate(bounds):
            min_len = MIN_SEGMENT_LEN if idx > 0 or f == n else _FALLBACK_MIN_LEN
            if f - start > 0:
                chosen.extend(self._solve_range(ev, eligible, start, f, min_len))
            start = f
        return chosen

    @staticmethod
    def _build_suggested_text(text: str, insertions: list[Insertion],
                              suggested_gaps: set[int]) -> str:
        # 从后往前在原文间隙插入句号，避免位移影响
        out = text
        for ins in sorted(insertions, key=lambda x: x.source_index, reverse=True):
            if ins.gap_id in suggested_gaps:
                pos = ins.source_index
                out = out[:pos] + "。" + out[pos:]
        return out


_model: BoundaryModel | None = None
_model_lock = threading.Lock()


def get_boundary_model() -> BoundaryModel:
    """进程内单例，避免重复统计语料。"""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = BoundaryModel()
    return _model


def segment_text(text: str) -> SegmentationResult:
    """便捷入口：用单例模型对稿件做断句。"""
    return get_boundary_model().segment(text)
