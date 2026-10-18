#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""decide.py —— issue-014 的 **决策层**（2026-10-10）。

主理人 2026-10-10 定：「决策不是以最高分数优先而是以最高排名优先。」

两个目标
--------
`score`（期望分最大化）
    四项在模型下互相独立 ⇒ 逐项取 argmax：胜方取 p 大者；选图取「集合命中期望分」最大者；
    MVP 取 P(队内第一) 最大者；比分取命中概率最大者。这是**最高分**版本。

`rank`（期望排名最大化）
    比赛结果是随机的，最终名次取决于**总分分布的形状**，不只是期望。
    本模块把它做成可计算的形式：

      * 每场每个候选预测 → 该场得分的一个分布（四项独立 ⇒ 卷积）；
      * 我方总分 T = Σ X_m；对手（场）总分 T_j（各按一种「人群策略」行事，与我们的选择无关）；
      * **E[rank] = 1 + Σ_j P(T_j > T)**（名次越小越好）。

    总分是几十~几百个独立小项之和 ⇒ 用**正态近似**表示分布（CLT；`--mc` 可做蒙特卡洛校验）：

      * 我方 T ~ N(μ, σ²)，对手 T_j ~ N(μ_j, σ_j²)；
      * P(T_j > T) = Φ( (μ_j − μ) / √(σ_j² + σ²) )。

    于是「抢方差」变成可优化项：在强场（μ_j 高）里略增方差反而抬高名次概率；弱场里降方差。

`run` 在 test 上同时跑两种目标，报告**期望总分、模拟名次、以及两种目标选法不同的场次**。

用法
----
  python -X utf8 decide.py selftest
  python -X utf8 decide.py run [--opponents 20] [--noise 0.25] [--mc]
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")  # # 仓库布局：数据与代码分目录（Alpha 1.1.1）
sys.path.insert(0, HERE)
sys.path.insert(0, HERE)  # scorer.py 与本模块同目录（仓库布局）

import veto_model as VM  # noqa: E402

MAX_MAPSETS = 3
MAX_MVP = 3

# MVP 项的**实测标定值**，**按 bo 分档**（2026-12）
#
# 候选池 = **胜方的 5 名选手**，故随机基线 = 1/5 = 0.20（不是 1/11）。
# ★ 但 bo1 与 bo3/bo5 的机制不同：**bo1 只有 1 张图**，MVP 更随机；
#   而 adj 是长周期衰减 rating，反映的是整体表现。
#
# ★ 关键教训一：语料上的「argmax 31.2%（n=308）」是**自证**，不可采信。
#   model_v2.mvp_label()（Q-008）=「取胜方队内 rating 最高者」，
#   旧 mvp_probs 也按 rating 排 ⇒ 只需复现该构造。
#   且 csapi_matches_all.jsonl 无 mvp 字段 ⇒ 语料里 **0 条真实 MVP 标签**。
#
# ★ 关键教训二：bo3/bo5 上选出的规则**不能**直接用到 bo1 赛事。
#   首次按 112 场（bo3/bo5 为主）选出 z(adj)，直接用到 Logitech G
#   （35 场，30 场是 bo1）⇒ 380 → 370（**−10 分**）。
#
# 真实标签实测（主理人提供，含留出验证）：
#   打分                  bo3/bo5(112)      bo1(30)
#   rating                  33.9%            16.7%
#   adj                     33.0%            13.3%
#   adj + toprate(旧)       33.0%            13.3%
#   toprate                 30.4%            20.0%
#   随机基线                20.0%            20.0%
#
# ⇒ bo3/bo5：期望 **0.31**（旧值 0.96 虚高约 3 倍）
# ⇒ bo1    ：**没有任何规则超过随机** ⇒ 期望按随机 **0.20**（选谁都一样）
#
# 注：scorer.py:193 要求 MVP 须出自**预测胜方**，否则整项 0，
#   故实际收益还要乘以胜方正确率（实测 62.9%）。
# 丰度门槛：选手历史出场场次中位数 < 此值时 MVP 打分不可靠（见 mvp_probs 注释）
MVP_RICH_MIN = 18
MVP_CAL_RICH = 0.31       # 丰度 >= MVP_RICH_MIN：真实命中 39~41%
MVP_CAL_POOR = 0.20       # 丰度 <  MVP_RICH_MIN：真实命中 12~15% ≈ 随机
MVP_CAL_PROB = MVP_CAL_RICH          # 兼容旧调用点（默认按高丰度）
DEFAULT_TAU = 1.0          # MVP softmax 温度


def phi(x):
    """标准正态 CDF。"""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def dist_moments(d):
    """{值: 概率} → (均值, 方差)。"""
    mu = sum(v * p for v, p in d.items())
    var = sum(p * (v - mu) ** 2 for v, p in d.items())
    return mu, max(0.0, var)


def convolve_moments(a, b):
    """两个独立分布的一二阶矩直接相加。"""
    return (a[0] + b[0], a[1] + b[1])


# ------------------------------------------------------------------ 比分模型
class ScoreModel:
    """按「预测胜率」给条件比分分布：强队更可能 2:0 / 3:0。"""

    def __init__(self, bins=None):
        # bins[bo] = [(p_upper, P(loser 拿 0 图)), ...]（按 p 分箱的经验频率）
        self.bins = bins or {}

    @staticmethod
    def fit(samples, edge=0.05):
        """用 train 段拟合：把 p_win（对最终胜方）分箱，统计「败方一图未得」的频率。"""
        raw = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
        for s in samples:
            t = s.get("truth")
            if not t or t.get("bo") not in (3, 5):
                continue
            sa, sb = t["score"]
            if sa == sb:
                continue
            w = s["teamA"] if s["y"] == 1 else s["teamB"]
            p = s.get("pA")
            if p is None:
                continue
            q = p if w == s["teamA"] else 1 - p
            loser_maps = min(sa, sb)
            bucket = min(9, int(q * 10))
            cell = raw[t["bo"]][bucket]
            cell[0] += 1
            cell[1] += 1 if loser_maps == 0 else 0
        bins = {}
        for bo, cells in raw.items():
            rows = []
            for b in range(10):
                n, k = cells.get(b, [0, 0])
                pr = (k + 1.0) / (n + 2.0)          # Laplace 平滑
                rows.append(((b + 1) / 10.0, pr))
            bins[bo] = rows
        return ScoreModel(bins)

    def p_loser_zero(self, bo, q):
        rows = self.bins.get(bo) or []
        for upper, pr in rows:
            if q <= upper:
                return pr
        return rows[-1][1] if rows else 0.5

    def cond_dist(self, bo, q):
        """胜方视角的比分分布：{(k,0): p, (k,1): 1-p}（bo3 k=2，bo5 k=3）。"""
        if bo == 3:
            p0 = self.p_loser_zero(3, q)
            return {(2, 0): p0, (2, 1): 1 - p0}
        if bo == 5:
            p0 = self.p_loser_zero(5, q)
            # 把「败方 0 图」的余量对半分给 3:1 / 3:2（bo5 样本少，不细调）
            return {(3, 0): p0, (3, 1): (1 - p0) * 0.5, (3, 2): (1 - p0) * 0.5}
        return {(1, 0): 1.0} if bo == 1 else {}


# ------------------------------------------------------------------ MVP 分布
def mvp_probs(rows, tau=DEFAULT_TAU, bo=3):
    """{player: P(队内第一)}。

    ★ 打分函数**按 rating 数据丰度分档**（2026-12，142 场**真实** MVP 标签）。

    起因是一个被主理人纠正的错误结论。我曾写「bo1 上没有任何规则超过随机，
    因为 1 张图的 MVP 本身接近随机」。主理人指出：
      「这只是现在的样本不够多、集中于冷门赛事而已。你要是遇到 spirit、vitality
        打 bo1，rating 会越来越有效的。」

    实测证实了主理人的判断（按 lineup 选手**历史出场场次中位数**分层）：

        丰度        n      z(adj) 命中
        <10 场     13       15.4%
        10~18 场   42       11.9%
        18~26 场   84       39.3%     ← 跃升
        ≥26 场      3       33.3%
        ────────────────────────────────
        <18 场     55       12.7%
        ≥18 场     87       39.1%

    效应量 26pp，**远大于 bo 的影响**（bo1 13.3% vs bo3 31.5%）。
    而 `bo1 n=30 的平均丰度 15.1` vs `bo3 n=108 的 19.8` ——
    我那 30 场 bo1 样本恰好集中在 rating 稀疏的队伍上，两个因素被绑在一起，
    我却把结论推广到了整个 bo1。**主理人的纠正是对的。**

    ⇒ 真正的机制是：**`adj` 需要足够的历史场次才稳定**。
      Spirit/Vitality 的选手有几十上百场 ⇒ adj 可用；
      冷门队选手十几场 ⇒ adj 基本是噪声。
    ⇒ 故分档依据是**丰度**（`state.p_w` 中位数），不是 bo。

    另：`toprate` 在高丰度段同样是净负担（40.5% → 39.3%）。
    """
    if not rows:
        return {}
    def zs(key):
        vals = [r[key] for r in rows]
        n = len(vals)
        mu = sum(vals) / n
        sd = (sum((v - mu) ** 2 for v in vals) / n) ** 0.5 or 1.0
        return [(r[key] - mu) / sd for r in rows]
    za = zs("adj")
    # 丰度门槛 18 场：<18 时 adj 与 toprate 都不可靠，加权反而稀释噪声
    use_tp = _mvp_richness(rows) < MVP_RICH_MIN
    if use_tp:
        zt = zs("toprate")
        logits = [tau * (za[i] + zt[i]) for i in range(len(rows))]
    else:
        logits = [tau * za[i] for i in range(len(rows))]
    mx = max(logits)
    exps = [math.exp(x - mx) for x in logits]
    s = sum(exps)
    return {r["player"]: exps[i] / s for i, r in enumerate(rows)}


def _mvp_richness(rows):
    """选手历史场次中位数（来自 `player_row` 注入的 `exp` 字段）。"""
    v = sorted(float(r.get("exp") or 0) for r in rows)
    return v[len(v) // 2] if v else 0.0


# ------------------------------------------------------------------ 每场分布
class MatchDist:
    """一场比赛中：真值分布（胜负/图集/MVP/比分）+ 任意候选计划的得分分布。"""

    def __init__(self, sample, mmodel, score_model):
        self.sample = sample
        self.A, self.B = sample["teamA"], sample["teamB"]
        self.bo = int(sample["bo"])
        self.k = max(1, min(self.bo, 5))
        p = sample["pA"]
        self.p_win = {self.A: p, self.B: 1 - p}
        # 选图真值分布（Plackett–Luce）
        msample = {"bo": self.bo, "rows": sample["map_rows"]}
        self.picked, map_probs = VM.predict_set(mmodel, msample, self.k)
        self.map_true = mapset_distribution(map_probs, self.k)
        # 比分真值分布（条件于胜方）
        self.score_true = {}
        for w in (self.A, self.B):
            q = self.p_win[w]
            d = score_model.cond_dist(self.bo, q)
            self.score_true[w] = {self._orient(sl, w): pr for sl, pr in d.items()}
        # MVP 真值分布（条件于胜方）
        self.mvp_true = {w: mvp_probs(sample["player_rows"].get(w) or [],
                                      bo=self.bo)
                         for w in (self.A, self.B)}
        self.plans = self._plans()

    def _orient(self, sl, winner):
        """把 (k,l) 换成 (teamA 回合, teamB 回合)。"""
        a, b = sl
        return (a, b) if winner == self.A else (b, a)

    def _plans(self):
        s = self.sample
        sets = [tuple(sorted(x)) for x, _ in sorted(self.map_true.items(), key=lambda kv: -kv[1])][:MAX_MAPSETS]
        if tuple(sorted(self.picked)) not in sets:
            sets = [tuple(sorted(self.picked))] + sets
        plans = []
        for w in (self.A, self.B):
            sls = sorted(self.score_true[w], key=lambda sl: -self.score_true[w][sl])
            if self.bo == 1:
                sls = [(1, 0) if w == self.A else (0, 1)]
            rows = s["player_rows"].get(w) or []
            mvps = [r["player"] for r in sorted(rows, key=lambda r: -(r["adj"] + r["toprate"]))[:MAX_MVP]]
            for sl in sls[:2]:
                for ms in sets:
                    for mv in (mvps or [""]):
                        plans.append({"winner": w, "score": sl, "maps": ms, "mvp": mv})
        # 去重
        seen = set()
        out = []
        for pl in plans:
            key = (pl["winner"], pl["score"], pl["maps"], pl["mvp"])
            if key not in seen:
                seen.add(key)
                out.append(pl)
        return out

    def points_dist(self, plan):
        """该计划的得分分布 {分值: 概率}。"""
        p = self.p_win
        d = collections.defaultdict(float)
        # 胜方项
        for val, pr in ((10, p[plan["winner"]]), (0, 1 - p[plan["winner"]])):
            d[val] += pr
        d = {k: v for k, v in d.items()}
        # 比分项
        if self.bo in (3, 5):
            hit = self.score_true[plan["winner"]].get(tuple(plan["score"]), 0.0)
            q = p[plan["winner"]]
            d = _mix(d, {10: q * hit, 0: 1 - q * hit})
        # 选图项（与胜负独立）
        md = collections.defaultdict(float)
        for tset, pr in self.map_true.items():
            md[5 * sum(1 for m in plan["maps"] if m in tset)] += pr
        d = _mix(d, dict(md))
        # MVP 项 —— **按 bo 分档标定**（2026-12，详见 MVP_CAL_PROB_* 的注释）。
        #
        #   bo3/bo5 → 0.31   （真实标签 33.9%，留出 30.3%；随机 20%）
        #   bo1     → 0.20   （真实标签 30 场**无任何规则超过随机**）
        #
        # 这两个值都是**实测**替换掉旧的自证值 0.96（虚高约 3 倍）。
        # 选谁仍按 mvp_probs 的 argmax（bo3/bo5 已改为纯 z(adj)），
        # 但**期望值**不再随 argmax 的自信度虚高。
        # ★ 按**丰度**分档（2026-12，见 mvp_probs）：选手历史出场场次中位数
        #   ≥ MVP_RICH_MIN ⇒ 真实命中 ~39~41%；< 门槛 ⇒ ~12~15% ≈ 随机。
        #   ⚠ 分档依据是 rating 数据丰度，**不是 bo**——我曾按 bo 分档并写下
        #     「bo1 的 MVP 不可预测」，主理人纠正：那是我 30 场 bo1 样本恰好
        #     集中在 rating 稀疏的冷门队伍上，两个因素被绑在一起。
        win_rows = self.sample["player_rows"].get(plan["winner"]) or []
        mh = (MVP_CAL_POOR if _mvp_richness(win_rows) < MVP_RICH_MIN
              else MVP_CAL_RICH)
        q = p[plan["winner"]]
        d = _mix(d, {10: q * mh, 0: 1 - q * mh})
        return d

    def moments(self, plan):
        return dist_moments(self.points_dist(plan))

    def max_points(self):
        """单场满分——**一律问官方计分器**（`scorer.max_total`），与 `predict.py` / `model_v2` 同一口径。

        早先这里是现算的 `10 + (10 if bo in (3,5) else 0) + 5*k + 10`，其中 `k` 是
        **我们预测的图数**（`k = max(1, min(bo, 5))`）。当真值是 `maps_mode="played"`
        （读法 A 降级：bo3 只打了 2 张）时，官方满分是 **40** 而这里给 **45** ⇒
        test 的 `realized_max` 比 `predict.py backtest` 多 5 分（13,545 vs 13,540）。
        满分是**真值的属性**，不该由我们的预测决定。
        """
        import model_v2 as M
        from scorer import max_total
        t = self.sample.get("truth")
        if t:
            return max_total(M.to_truth(t, self.sample.get("mvp_truth", "")))
        # 无真值（纯模拟路径）时退回按 bo 估算
        return 10 + (10 if self.bo in (3, 5) else 0) + 5 * self.k + 10


def _mix(a, b):
    out = collections.defaultdict(float)
    for va, pa in a.items():
        if pa <= 0:
            continue
        for vb, pb in b.items():
            if pb > 0:
                out[va + vb] += pa * pb
    return dict(out)


def _score_real(md, plan):
    """把计划拿到真实结果上计分（用官方计分器）。"""
    import model_v2 as M
    from scorer import Prediction, score_match
    t = md.sample["truth"]
    if not t:
        return 0
    truth = M.to_truth(t, md.sample.get("mvp_truth", ""))
    pred = Prediction(winner=plan["winner"], score=tuple(plan["score"]),
                      maps=list(plan["maps"]), mvp=plan["mvp"])
    return score_match(pred, truth).total


def mapset_distribution(map_probs_by_map, k):
    """逐图进入图集的概率 → 图集分布（Plackett–Luce：不放回加权抽样）。"""
    maps = list(map_probs_by_map)
    if k >= len(maps):
        return {frozenset(maps): 1.0}
    if k <= 0:
        return {frozenset(): 1.0}
    out = collections.defaultdict(float)

    def rec(remaining, chosen, prob, need):
        if need == 0:
            out[frozenset(chosen)] += prob
            return
        tot = sum(map_probs_by_map[m] for m in remaining) or 1.0
        for m in remaining:
            rec([x for x in remaining if x != m], chosen + [m],
                prob * map_probs_by_map[m] / tot, need - 1)

    rec(maps, [], 1.0, k)
    s = sum(out.values())
    return {kk: v / s for kk, v in out.items()}


# ------------------------------------------------------------------ 场（对手）
def crowd_plan(md, strategy, rng, noise):
    """人群策略（对手的选法）。`noise` = 偏离该策略的概率。"""
    s = md.sample
    A, B = md.A, md.B
    fav = A if md.p_win[A] >= 0.5 else B
    rows = s["map_rows"]
    plan = {"winner": fav}
    # ⚠ `maps` 保持**字母规范序**：它是计划的**身份**（_plans 去重靠它，
    # scorer 默认 maps_order_matters=False 按集合判分，顺序不影响得分）。
    # 但**给人看的顺序是另一回事** —— 模型明明排出了 Mirage 74% > Nuke 65% >
    # Dust2 57%，显示层却按字母排成 "Dust2 / Mirage / Nuke"，让人以为小镇是第一顺位。
    # 2026-10-19 主理人当场抓到这个问题。所以另开 `maps_ranked`（概率降序）专供显示，
    # `maps` 一个字节都不改 ⇒ 计分与历史结果逐位不变。
    if strategy == "popular":          # 只看全局流行度
        _rank = [f["map"] for f in sorted(rows, key=lambda f: (-f["g_app"], f["map"]))[:md.k]]
        plan["maps_ranked"] = tuple(_rank)
    elif strategy == "affinity":       # 只看两队亲和
        _rank = [f["map"] for f in sorted(rows, key=lambda f: (-(f["a_app"] + f["b_app"]), f["map"]))[:md.k]]
        plan["maps_ranked"] = tuple(_rank)
    elif strategy == "model_maps":     # 用了 veto 模型（强对手）
        plan["maps_ranked"] = tuple(md.picked)
    else:
        plan["maps_ranked"] = tuple(md.picked)
    plan["maps"] = tuple(sorted(plan["maps_ranked"]))
    mrows = s["player_rows"].get(fav) or []
    if strategy == "raw_star" and mrows:
        plan["mvp"] = max(mrows, key=lambda r: r["raw"])["player"]
    elif mrows:
        plan["mvp"] = max(mrows, key=lambda r: r["adj"])["player"]
    else:
        plan["mvp"] = ""
    # 比分：众数派永远 2:1 / 3:1；条件派按模型
    if strategy == "modal_score":
        plan["score"] = (2, 1) if fav == A else (1, 2)
        if md.bo == 5:
            plan["score"] = (3, 1) if fav == A else (1, 3)
        if md.bo == 1:
            plan["score"] = (1, 0) if fav == A else (0, 1)
    else:
        best = max(md.score_true[fav], key=lambda sl: md.score_true[fav][sl])
        plan["score"] = best
    # 噪声
    if noise > 0 and rng.random() < noise:
        w = rng.choice([A, B])
        if w != plan["winner"]:
            plan["winner"] = w
            mrows = s["player_rows"].get(w) or []
            plan["mvp"] = (max(mrows, key=lambda r: r["adj"])["player"] if mrows else "")
            plan["score"] = md._orient((1, 0) if md.bo == 1 else ((2, 1) if md.bo == 3 else (3, 1)), w)
    return plan


_PLAN_DIST_CACHE = {}


def _cache_plan_dist(pl):
    return _PLAN_DIST_CACHE[id(pl)]


def clone_field(match_dists, our_plans, opponents, noise, seed=99):
    """**同水平对手**：每个对手照抄我们的计划，但以 `noise` 概率随机换一个候选。

    这刻画「场里全是和我水平相当的人」——此时名次几乎完全由运气（方差）决定，
    正是「最高分」与「最高排名」分歧最大的情形。
    """
    rng = random.Random(seed)
    moments = []
    for j in range(opponents):
        mu = var = 0.0
        for i, md in enumerate(match_dists):
            for pl in md.plans:
                _PLAN_DIST_CACHE[id(pl)] = md.points_dist(pl)
            d = collections.defaultdict(float)
            for pl in md.plans:
                pr = (1.0 - noise) / 1.0 if pl is our_plans[i] else 0.0
                pr += noise / len(md.plans)
                for val, pv in _cache_plan_dist(pl).items():
                    d[val] += pr * pv
            m, v = dist_moments(dict(d))
            mu += m
            var += v
        # 给每个对手一点个体差异（否则所有人完全相同 ⇒ 名次退化为 1 / N+1）
        jitter = 1.0 + 0.02 * (j - opponents / 2.0) / max(1, opponents / 2.0)
        moments.append((mu * jitter, var * (1.0 + 0.5 * abs(jitter - 1.0))))
    return moments


class Field:
    """对手模型：每个对手一个「策略」，其总分分布 = N(μ, σ²)（逐场矩相加）。"""

    def __init__(self, opponents, noise=0.25, seed=20261010, moments=None):
        self.opponents = opponents
        if moments is not None:
            self.moments = moments
            self.noise = noise
            self.assign = ["<injected>"] * opponents
            self.seeds = [0] * opponents
            return
        self.strats = ["popular", "affinity", "model_maps", "raw_star", "modal_score"]
        rng = random.Random(seed)
        self.assign = [self.strats[i % len(self.strats)] for i in range(opponents)]
        self.seeds = [rng.randrange(1 << 30) for _ in range(opponents)]
        self.noise = noise
        self.moments = None

    def fit(self, match_dists):
        """给每个对手算**这批场次合计**的总分矩（赛场级语义：一场竞赛包含 `match_dists` 全部对局）。

        注意：`fit([md])` 就是**逐场语义**（对手也只提交这一场）——单场预测 `--objective rank` 必须用它，
        否则我方（一场 ~20 分）会与对手（整批 ~7000 分）相比，E[rank] 恒等于 N+1，选择退化成无脑追方差。
        """
        out = []
        for j, strat in enumerate(self.assign):
            rng = random.Random(self.seeds[j])
            mu = var = 0.0
            for md in match_dists:
                plan = crowd_plan(md, strat, rng, self.noise)
                m, v = md.moments(plan)
                mu += m
                var += v
            out.append((mu, var))
        self.moments = out
        return self

    def expected_rank(self, our_mu, our_var):
        """E[rank] = 1 + Σ_j Φ( (μ_j − μ) / √(σ_j² + σ²) )（正态近似，并列由 Φ 连续处理）。"""
        moments = self.moments or []
        tot = 0.0
        for (mj, vj) in moments:
            sd = math.sqrt(max(1e-9, vj + our_var))
            tot += phi((mj - our_mu) / sd)
        return 1.0 + tot          # 名次 = 1 + 期望被超越人数（并列按 0.5 由 Φ 连续近似处理）


# ------------------------------------------------------------------ 目标
def choose_score(md):
    """「最高分」目标的选法 = **逐项取众数（联合众数）**，与离线回测同一条口径：

      胜方取 P 大者；比分取该胜方下最可能的比分；选图取 veto 模型的组内 top-k；
      MVP 取「z(rating)+z(担当率)」最大者。放弃与期望分最大化的微小差别，
      换取「预测的是最可能的赛果」这一可解释性（两者在 test 上只差 8/317 场、约 40 分）。
    """
    A, B = md.A, md.B
    w = A if md.p_win[A] >= md.p_win[B] else B
    if md.bo == 1:
        sl = (1, 0) if w == A else (0, 1)
    else:
        sl = max(md.score_true[w], key=lambda k: md.score_true[w][k])
    mvp_probs = md.mvp_true.get(w) or {}
    mvp = max(mvp_probs, key=lambda p: mvp_probs[p]) if mvp_probs else ""
    return {"winner": w, "score": sl,
            "maps": tuple(sorted(md.picked)),      # 规范序＝计划身份（计分用）
            "maps_ranked": tuple(md.picked),        # 概率降序＝给人看
            "mvp": mvp}


def retarget(md, plan, winner):
    """把计划的胜方改成 `winner`，并把**比分与 MVP 一并重算到新胜方名下**
    （选图与胜方无关，保持原值）。

    为什么必须有这个函数：predict.py 的 rank-first 裁决（2026-10-12/13）会用
    rank 热门覆盖模型胜方。第一版只翻转了比分、**没动 MVP** ⇒ 第 4 项拿到的是
    「模型那一方」的选手，而 scorer 要求 MVP 出自**预测胜方**
    ⇒ 只要发生分歧，MVP 项（10 分）就**必然 0 分**（2026-10-13 实测：
    Vitality vs Falcons 胜方被改成 Falcons、MVP 却仍是 zywoo）。
    比分同理不能只靠翻转：两方的比分分布不同，翻过来的未必是新胜方下的众数。
    """
    if plan["winner"] == winner:
        return plan
    if md.bo == 1:
        sl = (1, 0) if winner == md.A else (0, 1)
    else:
        sc = md.score_true.get(winner) or {}
        sl = (max(sc, key=lambda k: sc[k]) if sc
              else (plan["score"][1], plan["score"][0]))
    mp = md.mvp_true.get(winner) or {}
    mvp = max(mp, key=lambda z: mp[z]) if mp else ""
    return dict(plan, winner=winner, score=sl, mvp=mvp)


def choose_score_maxexp(md):
    """（备选口径）纯期望分最大化：会偶尔为了别的项去赌更低概率的胜方。"""
    return max(md.plans, key=lambda pl: md.moments(pl)[0])


def running_moments(plans_moments):
    mu = sum(m for m, _v in plans_moments)
    var = sum(v for _m, v in plans_moments)
    return mu, var


def choose_rank(md, i, all_moments, field):
    """坐标下降的一步：固定其余场次，选使 E[rank] 最小的计划。"""
    others = [all_moments[j] for j in range(len(all_moments)) if j != i]
    base_mu, base_var = running_moments(others)
    best = None
    for pl in md.plans:
        m, v = md.moments(pl)
        er = field.expected_rank(base_mu + m, base_var + v)
        if best is None or er < best[0] - 1e-12:
            best = (er, pl)
    return best[1]


# ------------------------------------------------------------------ 校验/分析
def monte_carlo_rank(match_dists, our_plans, field, trials=2000, seed=7):
    """用蒙特卡洛校验正态近似的名次（返回 (平均名次, 前 25% 分位名次)）。"""
    rng = random.Random(seed)
    ranks = []
    for _ in range(trials):
        ours = 0.0
        for md, pl in zip(match_dists, our_plans):
            d = md.points_dist(pl)
            r = rng.random()
            acc = 0.0
            for val, pr in d.items():
                acc += pr
                if r <= acc:
                    ours += val
                    break
        better = 0
        for (mj, vj) in field.moments:
            theirs = 0.0
            for md in match_dists:
                plan = crowd_plan(md, "affinity", random.Random(rng.randrange(1 << 30)), field.noise)
                d = md.points_dist(plan)
                r = rng.random()
                acc = 0.0
                for val, pr in d.items():
                    acc += pr
                    if r <= acc:
                        theirs += val
                        break
            if theirs > ours:
                better += 1
        ranks.append(1 + better)
    ranks.sort()
    return sum(ranks) / len(ranks), ranks[len(ranks) // 4]


# ------------------------------------------------------------------ selftest
def _selftest():
    fails = []

    n_chk = [0]

    def chk(name, got, want, tol=1e-9):
        n_chk[0] += 1
        if isinstance(want, float) or isinstance(got, float):
            ok = abs(got - want) <= tol
        else:
            ok = got == want
        if not ok:
            fails.append("%s: got=%r want=%r" % (name, got, want))

    chk("phi 0", phi(0.0), 0.5)
    chk("phi 1.96 > 0.975", phi(1.96) > 0.975, True)
    m, v = dist_moments({0: 0.5, 10: 0.5})
    chk("moments mean", m, 5.0)
    chk("moments var", v, 25.0)

    md = mapset_distribution({"A": 0.8, "B": 0.2}, 1)
    chk("PL single A", round(md[frozenset(["A"])], 6), 0.8)
    md2 = mapset_distribution({"A": 1.0, "B": 1.0, "C": 1.0}, 2)
    chk("PL total prob", round(sum(md2.values()), 9), 1.0)
    chk("PL set count", len(md2), 3)

    # 比分模型：q 高 ⇒ 败方 0 图概率高
    sm = ScoreModel({3: [(0.3, 0.2), (0.6, 0.4), (0.8, 0.6), (1.0, 0.8)]})
    chk("p_loser_zero low q", sm.p_loser_zero(3, 0.2), 0.2)
    chk("p_loser_zero high q", sm.p_loser_zero(3, 0.9), 0.8)
    d = sm.cond_dist(3, 0.9)
    chk("cond dist sums", round(sum(d.values()), 9), 1.0)

    # MVP softmax
    rows = [{"player": "s", "adj": 1.4, "toprate": 0.7},
            {"player": "t", "adj": 1.0, "toprate": 0.2}]
    pr = mvp_probs(rows, tau=2.0)
    chk("mvp prob sums", round(sum(pr.values()), 9), 1.0)
    chk("mvp favors star", pr["s"] > pr["t"], True)
    # MVP 打分按**丰度**分档（2026-12，详见 mvp_probs）：丰度低时 adj 不可靠，
    # 加权 toprate 反而稀释噪声；丰度高时 toprate 是净负担，去掉。
    # exp = 该选手历史出场场次（player_row 注入）。
    poor = [{"player": "s", "adj": 1.00, "toprate": 0.9, "exp": 5.0},
            {"player": "t", "adj": 1.40, "toprate": 0.1, "exp": 6.0}]
    rich = [{"player": "s", "adj": 1.00, "toprate": 0.9, "exp": 60.0},
            {"player": "t", "adj": 1.40, "toprate": 0.1, "exp": 70.0}]
    chk("低丰度：adj 低的 toprate 高的可能反超", _mvp_richness(poor) < MVP_RICH_MIN, True)
    chk("高丰度：adj 高的必胜（纯 adj）",
        max(mvp_probs(rich), key=lambda k: mvp_probs(rich)[k]), "t")
    chk("高丰度：toprate 不再参与（纯 adj 的 argmax）",
        max(mvp_probs(rich), key=lambda k: mvp_probs(rich)[k]),
        max([{"player": "s", "adj": 1.00, "toprate": 0.0, "exp": 60.0},
              {"player": "t", "adj": 1.40, "toprate": 0.0, "exp": 70.0}],
             key=lambda r: r["adj"])["player"])
    chk("高丰度+高adj 不会被 toprate 反超",
        max(mvp_probs(rich), key=lambda k: mvp_probs(rich)[k]), "t")

    # 期望排名：对手恒 100 分（方差 0），我方恒 200 ⇒ 名次 1；我方恒 50 ⇒ 名次 2
    f = ("ignore",)
    class _F:
        moments = [(100.0, 0.0)]
        def expected_rank(self, mu, var, **kw):
            return 1.0 + phi((100.0 - mu) / math.sqrt(1e-9 + var))
    chk("rank win", round(_F().expected_rank(200, 0), 6), 1.0)
    chk("rank lose", round(_F().expected_rank(50, 0), 6), 2.0)
    chk("rank tie", round(_F().expected_rank(100, 1e-9), 4), 1.5)

    # ---- choose_rank 必须能为了方差而放弃均值（机制校验）----
    # 造一场：低方差高均值 (L) vs 高方差低均值 (H)；场越强越该选 H
    class _MD:
        plans = ["L", "H"]
        def moments(self, pl):
            return (40.0, 4.0) if pl == "L" else (36.0, 400.0)
    md = _MD()
    allm = [md.moments("L")]
    weak = Field(1, 0.0, moments=[(30.0, 4.0)])       # 我们大幅领先 ⇒ 要低方差
    strong = Field(1, 0.0, moments=[(60.0, 4.0)])     # 我们大幅落后 ⇒ 要高方差
    chk("choose_rank leads → low var", choose_rank(md, 0, allm, weak), "L")
    chk("choose_rank behind → high var", choose_rank(md, 0, allm, strong), "H")

    # ---- 满分口径必须与官方计分器逐场一致（2026-10-11 修，见 max_points 注释）----
    import model_v2 as _M
    from scorer import max_total as _official

    class _MDp:
        bo, k, sample = 3, 3, {}

    _veto_t = {"teamA": "A", "teamB": "B", "winner": "A", "score": [2, 0], "bo": 3,
               "maps": ["Dust2", "Mirage", "Nuke"], "maps_played": ["Dust2", "Mirage"],
               "maps_mode": "veto"}
    _played_t = dict(_veto_t, maps=["Dust2", "Mirage"], maps_mode="played")
    for _t, _want in ((_veto_t, 45), (_played_t, 40)):
        _o = _MDp()
        _o.sample = {"truth": _t, "mvp_truth": ""}
        chk("max_points %s" % _t["maps_mode"], MatchDist.max_points(_o), _want)
        chk("max_points %s == official" % _t["maps_mode"],
            MatchDist.max_points(_o), _official(_M.to_truth(_t, "")))

    if fails:
        print("DECIDE_SELFTEST_FAIL")
        for f in fails:
            print("  -", f)
        return 1
    print("DECIDE_SELFTEST_OK (%d checks)" % n_chk[0])
    return 0


# ------------------------------------------------------------------ run（test 上对比两目标）
def analyse(opponents=20, noise=0.25, do_mc=False, verbose=True):
    import split as SPLIT
    import pool_experiment as PE
    import model_v2 as M

    # **单一建模路径**：与 `model_v2.run` / `predict.py` 完全同一套模型
    prep = M.prepare(verbose=False)
    rows, meta = prep["rows"], prep["meta"]
    data = prep["data"]
    wmodel, mmodel = prep["wmodel"], prep["mmodel"]
    wsel = prep["wsel"]

    # train 段的 pA 用 **train-only** 模型（供 ScoreModel 拟合时不带样本内乐观偏差）；
    # valid / test 用最终模型（样本外）。
    for sp in data["train"]:
        sp["pA"] = M.predict_proba_win(wsel, sp)
    for k in ("valid", "test"):
        for sp in data[k]:
            sp["pA"] = M.predict_proba_win(wmodel, sp)
    smodel = ScoreModel.fit(data["train"])

    def build_mds(bucket):
        return [MatchDist(s, mmodel, smodel) for s in data[bucket]]

    out = {"opponents": opponents, "noise": noise}
    for bucket in ("valid", "test"):
        mds = build_mds(bucket)
        field = Field(opponents, noise).fit(mds)

        # score 目标
        plans_score = [choose_score(md) for md in mds]
        mom_score = [md.moments(pl) for md, pl in zip(mds, plans_score)]
        mu_s, var_s = running_moments(mom_score)
        er_s = field.expected_rank(mu_s, var_s)

        # rank 目标：从 score 解出发做坐标下降
        plans_rank = list(plans_score)
        mom_rank = list(mom_score)
        for _ in range(3):
            changed = 0
            for i, md in enumerate(mds):
                pl = choose_rank(md, i, mom_rank, field)
                if pl is not plans_rank[i] and \
                        (pl["winner"], pl["score"], pl["maps"], pl["mvp"]) != \
                        (plans_rank[i]["winner"], plans_rank[i]["score"], plans_rank[i]["maps"], plans_rank[i]["mvp"]):
                    plans_rank[i] = pl
                    mom_rank[i] = md.moments(pl)
                    changed += 1
            if not changed:
                break
        mu_r, var_r = running_moments(mom_rank)
        er_r = field.expected_rank(mu_r, var_r)

        diff = sum(1 for a, b in zip(plans_score, plans_rank)
                   if (a["winner"], a["score"], a["maps"], a["mvp"]) !=
                      (b["winner"], b["score"], b["maps"], b["mvp"]))
        # 现实对照：把 score 计划拿到真值上计分（检验模型自估是否乐观）
        realized = 0
        maxpts = 0
        for md, pl in zip(mds, plans_score):
            t = md.sample["truth"]
            realized += _score_real(md, pl)
            maxpts += md.max_points()
        # 方差杠杆诊断：∂E[rank]/∂σ 相对 ∂E[rank]/∂μ 的量级 ≈ D / (2σ)
        # D < 0（我们领先）或 |D|/(2σ) 远小于 1 时，方差几乎换不来名次 ⇒ 两目标重合。
        _fmu = sum(m for m, _ in field.moments) / max(1, len(field.moments))
        _lever = (_fmu - mu_s) / (2.0 * math.sqrt(var_s)) if var_s > 0 else 0.0
        out[bucket] = {
            "n": len(mds),
            "variance_lever_D_over_2sigma": _lever,
            "realized_points_score_plan": realized,
            "realized_max": maxpts,
            "score_objective": {"exp_points": mu_s, "sd": math.sqrt(var_s), "exp_rank": er_s},
            "rank_objective": {"exp_points": mu_r, "sd": math.sqrt(var_r), "exp_rank": er_r},
            "field": {"n_opponents": opponents, "mean": sum(m for m, _ in field.moments) / opponents,
                      "best": max(m for m, _ in field.moments)},
            "differing_matches": diff,
        }
        if do_mc:
            mc_s = monte_carlo_rank(mds, plans_score, field)
            out[bucket]["score_objective"]["mc_rank"] = mc_s
        if verbose:
            print("")
            print("== %s（%d 场，对手 %d 人 × 噪声 %.2f）==" % (bucket, len(mds), opponents, noise))
            print("  score 目标: 期望总分 %.0f (sd %.0f) → 模拟名次 %.2f" % (mu_s, math.sqrt(var_s), er_s))
            print("  rank  目标: 期望总分 %.0f (sd %.0f) → 模拟名次 %.2f" % (mu_r, math.sqrt(var_r), er_r))
            print("  对手最好者期望总分 %.0f；两目标选法不同的场次: %d / %d" % (
                out[bucket]["field"]["best"], diff, len(mds)))
            print("  方差杠杆 D/(2σ) = %.3f（D<0 = 我们领先场 ⇒ 方差只会伤名次；"
                  "且备选都是「用均值换方差」，单场幅度过小 ⇒ 两目标重合）" % _lever)
            print("  实盘对照: score 计划真实得分 %d / %d = %.1f%%（模型自估期望 %.0f，偏乐观 %.1f%%）" % (
                realized, maxpts, 100.0 * realized / maxpts, mu_s, 100.0 * (mu_s / realized - 1)))
            # 同水平对手场
            clone_mom = clone_field(mds, plans_score, opponents, noise)
            f2 = Field(opponents, noise, moments=clone_mom)
            er2_s = f2.expected_rank(mu_s, var_s)
            plans_r2 = list(plans_score)
            mom_r2 = list(mom_score)
            for _ in range(3):
                ch = 0
                for i, md in enumerate(mds):
                    pl = choose_rank(md, i, mom_r2, f2)
                    if (pl["winner"], pl["score"], pl["maps"], pl["mvp"]) !=                        (plans_r2[i]["winner"], plans_r2[i]["score"], plans_r2[i]["maps"], plans_r2[i]["mvp"]):
                        plans_r2[i] = pl
                        mom_r2[i] = md.moments(pl)
                        ch += 1
                if not ch:
                    break
            mu_r2, var_r2 = running_moments(mom_r2)
            er2_r = f2.expected_rank(mu_r2, var_r2)
            diff2 = sum(1 for a, b in zip(plans_score, plans_r2)
                        if (a["winner"], a["score"], a["maps"], a["mvp"]) !=
                           (b["winner"], b["score"], b["maps"], b["mvp"]))
            out[bucket]["clone_field"] = {
                "score_objective": {"exp_points": mu_s, "sd": math.sqrt(var_s), "exp_rank": er2_s},
                "rank_objective": {"exp_points": mu_r2, "sd": math.sqrt(var_r2), "exp_rank": er2_r},
                "differing_matches": diff2,
            }
            print("  [同水平场 %d 人] score 目标名次 %.2f → rank 目标名次 %.2f（改 %d 场，sd %.0f→%.0f，期望分 %.0f→%.0f）" % (
                opponents, er2_s, er2_r, diff2, math.sqrt(var_s), math.sqrt(var_r2), mu_s, mu_r2))
            if do_mc:
                print("  蒙特卡洛校验 score 目标名次: %.2f（前 25%% 分位 %.2f）" % mc_s)
    with io.open(os.path.join(DATA, "decide_results.json"), "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    return out


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        raise SystemExit(_selftest())
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        ap = argparse.ArgumentParser()
        ap.add_argument("--opponents", type=int, default=20)
        ap.add_argument("--noise", type=float, default=0.25)
        ap.add_argument("--mc", action="store_true")
        args = ap.parse_args(sys.argv[2:])
        analyse(args.opponents, args.noise, args.mc)
        raise SystemExit(0)
    print(__doc__)
