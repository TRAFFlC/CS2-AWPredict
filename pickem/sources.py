#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""胜率来源（U1）—— issue-015 的**公开推理端**。

本模块只做「给定强度 → 胜率」这一步，**不读任何数据文件**。
语料拟合、bootstrap 置信区间、时间安全回测与「014 Predictor 泄漏对照」留在私有区
（`data/Deepseek/Deepseek-V4.1 Flash/2026-10-07/015-scratch/sources.py`、`elo.py`），
公开区只带走**推理公式 + 已拟合的默认斜率**。

三条来源：

  * `EloSource`        : p = sigmoid(k * (r_a - r_b) / 400)
  * `RankPointsSource` : p = sigmoid(k * (log1p(pts_a) - log1p(pts_b)))   —— HLTV 周排名 points
  * `MatrixSource`     : **直接给两两胜率**（bo3），bo1 用 α=0.75 在 logit 上折算
                         —— 这是把 **014 已训模型 / 任何外部模型**的意见带进来的桥：
                         公共工具不读语料、不 import 私有代码，只吃 roster 里的矩阵。

为什么要分两条：014 语料实测 **log1p(points) 比线性 points 的 loglik 更好**
（bo3 −1614.8 → −1587.6）；而 Elo 由赛前逐场累计、信息更密，但需要按自己的量纲重新定斜率。

bo1 / bo3 各用一条斜率。默认值 = 在 014 语料上拟合的结果（见 `FITTED`），
可被 roster 的 `slopes` 字段或构造参数覆盖。

复现自检：python -X utf8 sources.py selftest
"""
from __future__ import annotations

import math

# ---------------------------------------------------------------- 默认斜率
# 来源：014 语料 `csapi_matches_all.jsonl` 的逐场拟合 @2026-10-13。
#   rank_points（log1p 特征）：全语料 bo1 n=287 / bo3 n=2,407
#   elo（分差/400 特征）      ：全语料 bo1 n=308 / bo3 n=2,492
# 记住两条已知限制（015 issue §5.1）：
#   ① 两条来源的 bo1 与 bo3 斜率置信区间**重叠** ⇒ bo 效应测不出，只能沿用拟合值；
#   ② 语料起点 2025-10-14、只有约一年跨度 ⇒ 早阶段的赛前历史可能极短（Elo 会失真）。
FITTED = {
    "elo": {"bo1": 2.2469, "bo3": 2.2394, "n1": 308, "n3": 2492},
    "rank_points": {"bo1": 0.4045, "bo3": 0.5584, "n1": 287, "n3": 2407},
}

# matrix 源的 bo1 折算系数：014 只给系列（bo3）概率，bo1 用先验 α 在 logit 上压缩。
# 依据 issue-015 §3（“014 Predictor 只有系列概率，bo1 用先验 α=0.75 在 logit 上折算”）。
ALPHA_BO1_DEFAULT = 0.75


def sigmoid(x: float) -> float:
    """数值稳定的 logistic。"""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


# ---------------------------------------------------------------- 来源
class EloSource:
    """Elo 差 / 400 的 logistic。键为**队名原文**（公开区不做别名归一）。"""

    def __init__(self, ratings, k1=None, k3=None, default=1500.0, scale=400.0):
        d = FITTED["elo"]
        self.ratings = {str(k): float(v) for k, v in ratings.items()}
        self.k1 = float(d["bo1"] if k1 is None else k1)
        self.k3 = float(d["bo3"] if k3 is None else k3)
        self.default = float(default)
        self.scale = float(scale)

    def rating(self, team):
        return self.ratings.get(str(team), self.default)

    def pwin(self, a, b, best_of):
        k = self.k1 if int(best_of) == 1 else self.k3
        return sigmoid(k * (self.rating(a) - self.rating(b)) / self.scale)


class MatrixSource:
    """两两胜率矩阵来源（bo3 概率直接给定；bo1 按 logit 折算）。

    用途：把**外部模型**（014 `predict.py`、任何第三方预测器）接进本工具，
    而不破坏「公共工具不抓数据、不读语料、不依赖私有路径」这条约定。

    * `p3(a, b)` 由矩阵给出；`p1(a, b) = sigmoid(α · logit(p3(a, b)))`，
      α 默认 **0.75**（issue-015 §3 的先验折算：014 只给系列概率）。
    * 只给一向也可以：缺的那一向取 `1 − 已有值`（矩阵必须自洽，见 `roster` 校验）。
    * 概率必须**严格落在 (0, 1)**：0/1 会让 logit 发散，属非法输入。
    """

    def __init__(self, matrix, alpha_bo1=0.75):
        self.matrix = {str(a): {str(b): float(v) for b, v in row.items()}
                       for a, row in matrix.items()}
        self.alpha = float(alpha_bo1)

    def p3(self, a, b):
        a, b = str(a), str(b)
        if a == b:
            return 0.5
        if b in self.matrix.get(a, {}):
            return self.matrix[a][b]
        if a in self.matrix.get(b, {}):
            return 1.0 - self.matrix[b][a]
        raise KeyError("矩阵缺少 %s → %s 的胜率" % (a, b))

    def pwin(self, a, b, best_of):
        p = self.p3(a, b)
        if int(best_of) == 1:
            lg = math.log(p / (1.0 - p))
            return sigmoid(self.alpha * lg)
        return p

    def strength(self, team):
        """展示用的「对场胜率」：对全部对手的 bo3 胜率均值（0..1）。

        矩阵源没有标量强度，这是给报表一个可读的替代列（不参与任何计算）。
        """
        t = str(team)
        # 对手集合 = 出现过的行名 ∪ 出现过的列名（矩阵可以只给上三角）
        opp = {x for x in self.matrix if x != t}
        for row in self.matrix.values():
            opp.update(x for x in row if x != t)
        opp = sorted(opp)
        if not opp:
            return 0.5
        return sum(self.p3(t, o) for o in opp) / len(opp)


class RankPointsSource:
    """log1p(points) 差的 logistic。未知队按 `default_points` 处理。"""

    def __init__(self, points, k1=None, k3=None, default_points=0.0):
        d = FITTED["rank_points"]
        self.points = {str(k): float(v) for k, v in points.items() if v is not None}
        self.k1 = float(d["bo1"] if k1 is None else k1)
        self.k3 = float(d["bo3"] if k3 is None else k3)
        self.default_points = float(default_points)

    def pts(self, team):
        return self.points.get(str(team), self.default_points)

    def pwin(self, a, b, best_of):
        k = self.k1 if int(best_of) == 1 else self.k3
        return sigmoid(k * (math.log1p(self.pts(a)) - math.log1p(self.pts(b))))


# ---------------------------------------------------------------- 预计算
def fast_pwin(src, teams):
    """把所有有序对 × {bo1, bo3} 的胜率预计算成查表函数。

    `simulate_scenes` 会调用 pwin 数十万次；直接调用非查表实现会让模拟慢到不可用。
    返回 `f(a, b, best_of) -> float`。
    """
    table = {}
    for a in teams:
        for b in teams:
            if a == b:
                continue
            table[(a, b, 1)] = float(src.pwin(a, b, 1))
            table[(a, b, 3)] = float(src.pwin(a, b, 3))

    def f(a, b, bo):
        return table.get((a, b, int(bo)), 0.5)

    return f


# ---------------------------------------------------------------- roster → 来源
def check_matrix(matrix, names):
    """校验两两胜率矩阵；返回 problems 列表（空 = 合法）。

    必须查的东西（每一条都对应一种**静默算错**的方式）：
      1. 队名覆盖：每对有序 (a,b) 至少要有一向给出（否则模拟会拿 0.5 兜底）；
      2. 取值域：严格 (0,1)（0/1 会让 bo1 的 logit 折算发散）；
      3. 自洽：两向都给时必须 `m_ab + m_ba ≈ 1`（否则等于矩阵里装了两套意见）。
    """
    probs = []
    if not isinstance(matrix, dict) or not matrix:
        return ["source=matrix 时必须给 matrix（{队名: {队名: 胜率}}）"]
    names = [str(n) for n in names]
    known = set(names)
    for a, row in matrix.items():                     # 0) 未知队名（拼错会让整行静默失效）
        if str(a) not in known:
            return ["矩阵含未知队名 %s（不在 roster 的 teams 里）" % a]
        if not isinstance(row, dict):
            return ["矩阵行 %s 应为 {队名: 胜率}" % a]
        for b in row:
            if str(b) not in known:
                return ["矩阵 %s 行含未知队名 %s" % (a, b)]
    for a in names:
        row = matrix.get(a) or {}
        for b in names:
            if a == b:
                continue
            v = row.get(b)
            back = (matrix.get(b) or {}).get(a)
            if v is None and back is None:            # 1) 每一对有序都要有落点
                return ["矩阵既无 %s→%s 也无反向（缺项会被模拟拿 0.5 兜底）" % (a, b)]
            for who, x in ((a + "→" + b, v), (b + "→" + a, back)):
                if x is None:
                    continue
                if isinstance(x, bool) or not isinstance(x, (int, float)):   # 2) 取值域
                    return ["矩阵 %s 不是数字（%r）" % (who, x)]
                if not (0.0 < float(x) < 1.0):
                    return ["矩阵 %s = %r 必须在开区间 (0,1) 内" % (who, x)]
            if v is not None and back is not None and abs(float(v) + float(back) - 1.0) > 0.02:
                return ["矩阵不自洽：%s→%s = %.4f 而反向 = %.4f（两者之和应 ≈ 1）"
                        % (a, b, float(v), float(back))]     # 3) 自洽
    return probs


def source_from_roster(roster):
    """由 roster dict 构造胜率来源。返回 (src, label, coefs)。

    roster 约定（见 README「roster 格式」）：
        source == "elo"          → 每队取 `rating`
        source == "rank_points"  → 每队取 `points`
        source == "matrix"       → 取 `matrix`（两两 bo3 胜率）+ 可选 `alpha_bo1`
        slopes（可选）           → {"bo1": x, "bo3": y} 覆盖默认斜率（matrix 源不用）
    """
    teams = roster["teams"]
    kind = roster.get("source", "elo")
    slopes = roster.get("slopes") or {}
    k1, k3 = slopes.get("bo1"), slopes.get("bo3")
    if kind == "elo":
        ratings = {t["name"]: t["rating"] for t in teams}
        src = EloSource(ratings, k1=k1, k3=k3)
        d = FITTED["elo"]
    elif kind == "rank_points":
        pts = {t["name"]: t["points"] for t in teams}
        src = RankPointsSource(pts, k1=k1, k3=k3)
        d = FITTED["rank_points"]
    elif kind == "matrix":
        names = [t["name"] for t in teams]
        probs = check_matrix(roster.get("matrix"), names)
        if probs:
            raise ValueError("; ".join(probs))
        src = MatrixSource(roster["matrix"], alpha_bo1=roster.get("alpha_bo1", 0.75))
        d = {"bo1": ALPHA_BO1_DEFAULT, "bo3": None}
    else:
        raise ValueError("未知 source %r（可选 elo / rank_points / matrix）" % kind)
    coefs = {"bo1": src.k1 if hasattr(src, "k1") else src.alpha,
             "bo3": src.k3 if hasattr(src, "k3") else None,
             "default_bo1": d["bo1"], "default_bo3": d["bo3"]}
    return src, kind, coefs


# ---------------------------------------------------------------- 自证
def _selftest():
    ok = 0

    def chk(cond, msg):
        nonlocal ok
        if not cond:
            raise AssertionError("FAIL: " + msg)
        ok += 1

    src = EloSource({"A": 1800.0, "B": 1400.0}, k1=0.8, k3=1.2)
    chk(abs(src.pwin("A", "B", 3) + src.pwin("B", "A", 3) - 1.0) < 1e-12, "Elo 应对称")
    chk(src.pwin("A", "B", 3) > src.pwin("A", "B", 1) > 0.5, "bo3 应比 bo1 更极端")
    chk(abs(src.rating("ZZZ") - 1500.0) < 1e-12, "未知队用默认 1500")
    chk(abs(src.pwin("A", "A", 3) - 0.5) < 1e-12, "同队应 0.5")

    rp = RankPointsSource({"A": 1000.0, "B": 100.0}, k1=0.4, k3=0.6)
    chk(abs(rp.pwin("A", "B", 1) + rp.pwin("B", "A", 1) - 1.0) < 1e-12, "排名源应对称")
    chk(rp.pwin("A", "B", 3) > rp.pwin("A", "B", 1) > 0.5, "bo3 应更极端")
    chk(abs(rp.pwin("A", "ZZZ", 3)
            - sigmoid(0.6 * (math.log1p(1000.0) - math.log1p(0.0)))) < 1e-12, "未知队用 0 分")

    # 单调性：分差越大越极端
    chk(EloSource({"A": 1600.0, "B": 1400.0}, 2.0, 2.0).pwin("A", "B", 3)
        > EloSource({"A": 1510.0, "B": 1400.0}, 2.0, 2.0).pwin("A", "B", 3), "分差大应更悬殊")

    # 默认斜率来自 FITTED（未被覆盖）
    d = EloSource({"A": 1.0})
    chk(abs(d.k1 - FITTED["elo"]["bo1"]) < 1e-12 and abs(d.k3 - FITTED["elo"]["bo3"]) < 1e-12,
        "默认斜率应取 FITTED")

    # fast_pwin 与直接调用一致，且覆盖 bo1/bo3
    teams = ["A", "B", "C"]
    f = fast_pwin(src, teams)
    chk(abs(f("A", "B", 3) - src.pwin("A", "B", 3)) < 1e-12, "fast_pwin bo3 一致")
    chk(abs(f("A", "B", 1) - src.pwin("A", "B", 1)) < 1e-12, "fast_pwin bo1 一致")
    chk(abs(f("A", "A", 3) - 0.5) < 1e-12, "fast_pwin 同队回退 0.5")

    # ---- matrix 源（外部模型的桥：014 Predictor 等） ----
    m = MatrixSource({"A": {"B": 0.8, "C": 0.6}, "B": {"C": 0.5}}, alpha_bo1=0.75)
    chk(abs(m.p3("A", "B") - 0.8) < 1e-12, "matrix 正向取值")
    chk(abs(m.p3("B", "A") - 0.2) < 1e-12, "matrix 缺反向时取 1−p")
    chk(abs(m.pwin("A", "A", 3) - 0.5) < 1e-12, "matrix 同队 0.5")
    chk(abs(m.pwin("A", "B", 3) - 0.8) < 1e-12, "matrix bo3 直给")
    p1 = m.pwin("A", "B", 1)
    chk(0.5 < p1 < 0.8, "bo1 应被 α 向 0.5 压缩（α=0.75）")
    chk(abs(p1 - sigmoid(0.75 * math.log(0.8 / 0.2))) < 1e-12, "bo1 折算公式应为 sigmoid(α·logit(p3))")
    chk(abs(MatrixSource({"A": {"B": 0.8}}, alpha_bo1=1.0).pwin("A", "B", 1) - 0.8) < 1e-12,
        "α=1 时 bo1 应退化为 bo3")
    chk(abs(m.strength("A") - 0.7) < 1e-12, "strength 应为对场均值（(0.8+0.6)/2）")
    chk(abs(m.p3("C", "B") - 0.5) < 1e-12 and abs(m.p3("C", "A") - 0.4) < 1e-12,
        "三队矩阵的补向应正确")
    try:
        MatrixSource({"A": {"B": 0.8}, "B": {}}).p3("B", "A") + \
            MatrixSource({"A": {}, "B": {}}).p3("A", "B")
        chk(False, "矩阵缺项应抛 KeyError")
    except KeyError:
        chk(True, "矩阵缺项应抛 KeyError")

    GOOD = {"A": {"B": 0.8}, "B": {"A": 0.2}}
    chk(check_matrix(GOOD, ["A", "B"]) == [], "自洽矩阵应通过")
    for bad, why in (({"A": {"B": 0.8}}, "缺反向且只有单向时也算合法（取补）"),
                     ({"A": {"B": 0.8, "C": 0.5}, "B": {"A": 0.2}}, "缺 C 反向应报错"),
                     ({"A": {"B": 1.0}, "B": {}}, "概率 =1 应报错"),
                     ({"A": {"B": 0.0}, "B": {}}, "概率 =0 应报错"),
                     ({"A": {"B": 0.8}, "B": {"A": 0.5}}, "两向之和 ≠1 应报错"),
                     ({"A": {"B": "0.8"}, "B": {}}, "非数字应报错"),
                     ({"B": {}}, "缺行应报错")):
        got = check_matrix(bad, ["A", "B"])
        want_bad = "也算合法" not in why
        chk((got != []) == want_bad, why)

    mroster = {"source": "matrix", "alpha_bo1": 0.6,
               "matrix": {"A": {"B": 0.7}, "B": {"A": 0.3}},
               "teams": [{"name": "A", "seed": 1}, {"name": "B", "seed": 2}]}
    msrc, mlabel, mcoefs = source_from_roster(mroster)
    chk(mlabel == "matrix" and abs(msrc.alpha - 0.6) < 1e-12, "matrix 源与 α 覆盖")
    chk(abs(mcoefs["bo1"] - 0.6) < 1e-12 and mcoefs["bo3"] is None, "matrix 的 coefs 应报 α")
    try:
        source_from_roster({"source": "matrix", "teams": mroster["teams"]})
        chk(False, "matrix 源缺 matrix 字段应抛错")
    except ValueError:
        chk(True, "matrix 源缺 matrix 字段应抛错")

    # roster → 来源
    roster = {"source": "elo", "slopes": {"bo3": 2.5},
              "teams": [{"name": "A", "seed": 1, "rating": 1600.0},
                        {"name": "B", "seed": 2, "rating": 1500.0}]}
    s2, label, coefs = source_from_roster(roster)
    chk(label == "elo" and abs(coefs["bo3"] - 2.5) < 1e-12, "roster 覆盖 bo3 斜率")
    chk(abs(coefs["bo1"] - FITTED["elo"]["bo1"]) < 1e-12, "未覆盖的 bo1 应回退默认")
    try:
        source_from_roster({"source": "bogus", "teams": []})
        chk(False, "未知 source 应抛错")
    except ValueError:
        chk(True, "未知 source 应抛错")

    print("SOURCES_SELFTEST_OK (%d checks)" % ok)
    return 0


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        raise SystemExit(_selftest())
    print(__doc__)
