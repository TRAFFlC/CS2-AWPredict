#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""veto_model.py —— issue-014 第 3 项（选图）的 **veto 序列 + 近期加权** 模型（2026-10-10）。

背景
----
基线选图 = 「训练集全局图频次取 top-k」。它有两个硬伤（实测 test 选图 2,160 分 / 命中率 47.9%）：
  1. 忽视 **双方偏好**：同一张图，A 队爱打、B 队爱 ban，结果完全不同；
  2. 新图（Cache 入池）零样本 ⇒ 频率法直接判死，而实测 test 里 Cache 出场 110 次。

本模块用 `csapi_veto.jsonl`（1,440 场带完整 pick/ban 顺序的 HLTV veto 记录）：
不直接背「哪 3 张图」（那是结果），而是学 **每张图的「进入最终图集」概率**。

两类信息
--------
(A) **veto 顺序** → 每队对每张图的 ban / pick 频次（比「总频次」细一档）；
(B) **近期加权**（半天衰期 60 天）→ 池换代后（era1→era2 换入 Anubis/Cache）
    老统计迅速失重，新池的流行度能在几周内学出来。这条是治好「Anubis/Cache 被系统性低估」的关键。

特征（全部来自「该场之前」的历史，严格时序安全）：

  g_app  该图在历史比赛图集里的出现率（池内出现的场次为分母；未见图 = 池内均匀先验）
  a_app / b_app   该队参与的比赛中，该图出现率（向 g_app 收缩）
  a_ban / b_ban   该队 **ban** 该图的频率（向全局 ban 份额收缩）——来自 veto 顺序
  a_pick / b_pick 该队 **pick** 该图的频率（向全局 pick 份额收缩）
  a_wr / b_wr     该队在该图上的图级胜率（向 0.5 收缩）
  new             该图在历史里**从未进过池**（真正的新入池图，如 era2 的 Cache）
  min_app         双方「都愿意打」的下界（图要留下，两边都不能 ban）
  both_pick(_prod) / both_ban(_prod)   双方 pick/ban 的和与积（线性模型看不到乘性关系）

排序模式：**组内 top-k**（k = bo）。这是集合命中口径（Q-004 默认读法）。

协议（防泄漏）
--------------
* 逐日推进：当天的比赛先用「截止昨日」的历史取特征，再统一更新历史（同日互不可见）。
* 训练用 train 段（特征也是 walk-forward 取的），超参 C 在 valid 上选，test 只报一次。
* `--mode train_only`：历史只用 train 段（更保守），`--mode walk_forward`：历史随日期滚动（贴近实盘）。

用法
----
  python -X utf8 veto_model.py selftest
  python -X utf8 veto_model.py run [--mode walk_forward|train_only]
"""
from __future__ import annotations

import collections
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")  # # 仓库布局：数据与代码分目录（Alpha 1.0）
sys.path.insert(0, HERE)
sys.path.insert(0, HERE)  # scorer.py 与本模块同目录（仓库布局）

import split as SPLIT  # noqa: E402

MATCHES = os.path.join(DATA, "csapi_matches_all.jsonl")
TRUTH = os.path.join(DATA, "truth_maps_B.jsonl")
VETO = os.path.join(DATA, "csapi_veto.jsonl")
EXCLUDE = os.path.join(HERE, "exclude_band_unknown_ids.txt")
OUT = os.path.join(DATA, "veto_model_results.json")

# 平滑强度（拔靴前的地板）
K_TEAM = 4.0        # 队伍级向全局收缩
EPS_GLOBAL = 1.0    # 全局份额的均匀地板（新图 ⇒ 退回均匀）
UNIFORM_RATE = 3.0 / 7.0   # 池内一张图的「无信息」出现率（bo3=3/7）
FIRST_BAN_TEAM2 = 0.627    # 实测：team2 先 ban 的比例（全量 1,581 场 veto 记录复测 = 991/1581 = 62.7%；
                          # 2026-10-11 由旧样本的 0.636 改为此值——旧值是 1,416 场时的快照）
FIRST_BAN_LOWER = 0.665    # 2026-10-11 实测：**先 ban 方 = 周排名较低者** 的命中率（1051/1581 = 66.5%）
SHAPE_MIX = 0.773          # 2026-10-11 实测：bo3 第 5/6 步与第 1/2 步**同序**的比例（1232/1581）
# 注意（已由自证钉住）：本模型的 ban 是**逐队独立**取各自最优 ⇒ 一个 ban 对内部的顺序
# **不改变最终图集**；只有「谁在第 1 步」会改变。故型态只作研究参数保留，生产不用。
HALF_LIFE = 60.0    # 近期加权的半衰期（天）

FEATURES = ["g_app", "a_app", "b_app", "a_ban", "b_ban", "a_pick", "b_pick",
            "a_wr", "b_wr", "new", "both_pick", "both_ban",
            "min_app", "both_pick_prod", "both_ban_prod", "sim_in_set"]

# 2026-10-11 新增：**池内排名**（线性模型看不到「是不是对方 top-2 pick」这类相对位置，
# 只能看到绝对概率，而 veto 本质是「在池里挑第一名」的过程）+ **策略交互**。
RANK_FEATURES = ["a_pick_rank", "b_pick_rank", "min_pick_rank",
                 "a_ban_rank", "b_ban_rank", "min_ban_rank",
                 "a_app_rank", "b_app_rank", "min_app_rank",
                 "both_pick_top1", "both_pick_top2", "both_ban_top1", "both_ban_top2",
                 "a_pick_top1", "b_pick_top1", "a_ban_top1", "b_ban_top1",
                 "a_keep", "b_keep", "both_keep", "own_pick_free", "opp_pick_free"]
FEATURES2 = FEATURES + RANK_FEATURES + ["sim2_in_set"]
# 2026-10-11：先手规则（先 ban 方 = 排名较低者）+ 型态混合
SIM3_FEATURES = ["sim3_in_set"]                    # 先 ban 方 = 排名较低者（实测 66.5%）
FEATURES3 = FEATURES2 + SIM3_FEATURES


def load_jsonl(path):
    out = []
    for line in io.open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def load_veto(path=VETO):
    """id -> {bo, order:[(step, kind, team, map)], teams:(first,second), set:[...], pool:[...]}"""
    out = {}
    for rec in load_jsonl(path):
        order = rec.get("veto") or []
        if not order:
            continue
        teams = []
        for s in order:
            if s[2] and s[2] not in teams:
                teams.append(s[2])
        out[rec["id"]] = {
            "bo": rec.get("bo"),
            "order": [tuple(s) for s in order],
            "teams": tuple(teams),
            "set": [s[3] for s in order if s[1] in ("pi", "L")],
            "pool": [s[3] for s in order],
        }
    return out


def days_apart(d1, d2):
    """日期字符串 (YYYY-MM-DD) 的日差；纯手算，避免额外依赖。"""
    def num(d):
        y, m, dd = (int(x) for x in d.split("-"))
        return y * 372 + m * 31 + dd
    return num(d1) - num(d2)


# ------------------------------------------------------------------ 历史（时序）
class MapHistory:
    """逐日推进、**指数衰减**（半衰期 HALF_LIFE 天）的图级历史。

    设计要点：**未见过的图退化为池内均匀先验**，不会因为「训练集没有 Cache」就把
    Cache 判死；同时老统计会随时间失重，池换代后新图能在几周内被学出来。
    """

    def __init__(self, uniform=UNIFORM_RATE, half_life=HALF_LIFE):
        self.uniform = uniform
        self.half_life = half_life
        self.app = collections.Counter()            # map -> 衰减后的出现次数
        self.pooln = collections.Counter()          # map -> 衰减后的「在池中」场次
        self.tapp = collections.defaultdict(collections.Counter)
        self.tpooln = collections.defaultdict(collections.Counter)
        self.tban = collections.defaultdict(collections.Counter)
        self.tpick = collections.defaultdict(collections.Counter)
        self.gban = collections.Counter()
        self.gpick = collections.Counter()
        self.nban = collections.defaultdict(float)
        self.npick = collections.defaultdict(float)
        self.tw = collections.defaultdict(collections.Counter)   # team -> map -> wins
        self.tp = collections.defaultdict(collections.Counter)   # team -> map -> plays
        self.seen_pool = set()                    # 历史里进过池的图（= 非新图）
        self.last_date = None

    # ---- 衰减 ----
    def decay(self, days):
        if days <= 0 or self.half_life <= 0:
            return
        f = 0.5 ** (days / self.half_life)
        for c in (self.app, self.pooln, self.gban, self.gpick):
            for k in list(c):
                c[k] *= f
        for store in (self.tapp, self.tpooln, self.tban, self.tpick, self.tw, self.tp):
            for t in list(store):
                row = store[t]
                for k in list(row):
                    row[k] *= f
        for t in list(self.nban):
            self.nban[t] *= f
        for t in list(self.npick):
            self.npick[t] *= f

    def advance(self, date):
        """推进到某日期（先衰减，再取特征）。"""
        if self.last_date is not None:
            self.decay(days_apart(date, self.last_date))
        self.last_date = date

    # ---- 查询（带平滑） ----
    def g_app(self, m):
        n = self.pooln[m]
        return (self.app[m] + K_TEAM * self.uniform) / (n + K_TEAM) if n else self.uniform

    def team_app(self, t, m):
        g = self.g_app(m)
        n = self.tpooln[t][m]
        return (self.tapp[t][m] + K_TEAM * g) / (n + K_TEAM) if n else g

    def _share(self, counter, total, m, universe):
        u = max(1, universe)
        return (counter[m] + EPS_GLOBAL) / (total + EPS_GLOBAL * u)

    def team_ban(self, t, m, universe):
        g = self._share(self.gban, sum(self.gban.values()), m, universe)
        n = self.nban[t]
        return (self.tban[t][m] + K_TEAM * g) / (n + K_TEAM) if n else g

    def team_pick(self, t, m, universe):
        g = self._share(self.gpick, sum(self.gpick.values()), m, universe)
        n = self.npick[t]
        return (self.tpick[t][m] + K_TEAM * g) / (n + K_TEAM) if n else g

    def team_wr(self, t, m):
        w, p = self.tw[t][m], self.tp[t][m]
        return (w + K_TEAM * 0.5) / (p + K_TEAM) if p else 0.5

    def _pool_ranks(self, t, pool, fn):
        """该队在图上的**池内排名**（1 = 最偏好/最常出现）。"""
        vals = sorted(pool, key=lambda m: (-fn(t, m), m))
        return {m: float(i + 1) for i, m in enumerate(vals)}

    def rank_table(self, teamA, teamB, pool):
        u = len(pool)
        out = {}
        out["a_pick"] = self._pool_ranks(teamA, pool, lambda t, m: self.team_pick(t, m, u))
        out["b_pick"] = self._pool_ranks(teamB, pool, lambda t, m: self.team_pick(t, m, u))
        out["a_ban"] = self._pool_ranks(teamA, pool, lambda t, m: self.team_ban(t, m, u))
        out["b_ban"] = self._pool_ranks(teamB, pool, lambda t, m: self.team_ban(t, m, u))
        out["a_app"] = self._pool_ranks(teamA, pool, lambda t, m: self.team_app(t, m))
        out["b_app"] = self._pool_ranks(teamB, pool, lambda t, m: self.team_app(t, m))
        return out

    def features_for(self, teamA, teamB, pool, bo=3, date=None):
        """**带日期的正规入口**：自动解析先手方（先 ban 方 = 排名较低者）。

        2026-10-11 抓到的接线坑：`features()` 的 `first` 缺省为 None，此时 `sim3_in_set`
        会**退化成 sim2_in_set**（同一列），看上去「跑了 v3」其实那一列是重复列。
        凡是手上有比赛日期的调用方都必须走本函数（model_v2 / predict 已改）。
        """
        return self.features(teamA, teamB, pool, bo,
                             first=first_banner(teamA, teamB, date))

    def features(self, teamA, teamB, pool, bo=3, first=None):
        universe = len(pool)
        sim = self.sim_features(teamA, teamB, pool, bo)
        sim2 = self.sim_features(teamA, teamB, pool, bo, combine="hier")
        # 2026-10-11：用**主理人给的先手规则**（先 ban 方 = 排名较低者）做确定性模拟。
        # sim2 = 两种手方各半的加权（不知道规则时的稳妥口径）；sim3 = 知道规则后的单支。
        # 只影响 pick 顺序；ban 逐队独立 ⇒ bo3 那 22.7% 的「ban 对内部换序」对本模型无关。
        sim3 = (self.sim_features(teamA, teamB, pool, bo, combine="hier", first=first)
                if first is not None else sim2)
        rk = self.rank_table(teamA, teamB, pool)
        rows = []
        for m in pool:
            a_app = self.team_app(teamA, m)
            b_app = self.team_app(teamB, m)
            a_ban = self.team_ban(teamA, m, universe)
            b_ban = self.team_ban(teamB, m, universe)
            a_pick = self.team_pick(teamA, m, universe)
            b_pick = self.team_pick(teamB, m, universe)
            rows.append({
                "map": m,
                "g_app": self.g_app(m),
                "a_app": a_app,
                "b_app": b_app,
                "a_ban": a_ban,
                "b_ban": b_ban,
                "a_pick": a_pick,
                "b_pick": b_pick,
                "a_wr": self.team_wr(teamA, m),
                "b_wr": self.team_wr(teamB, m),
                "new": 1.0 if m not in self.seen_pool else 0.0,
                "both_pick": a_pick + b_pick,
                "both_ban": a_ban + b_ban,
                # 交互项：「双方都愿意打」才更可能留下（线性模型看不到乘性关系）
                "min_app": min(a_app, b_app),
                "both_pick_prod": a_pick * b_pick,
                "both_ban_prod": a_ban * b_ban,
                "sim_in_set": sim.get(m, 0.0),
                "sim2_in_set": sim2.get(m, 0.0),
                "sim3_in_set": sim3.get(m, 0.0),
                # ---- 池内排名 ----
                "a_pick_rank": rk["a_pick"][m],
                "b_pick_rank": rk["b_pick"][m],
                "min_pick_rank": min(rk["a_pick"][m], rk["b_pick"][m]),
                "a_ban_rank": rk["a_ban"][m],
                "b_ban_rank": rk["b_ban"][m],
                "min_ban_rank": min(rk["a_ban"][m], rk["b_ban"][m]),
                "a_app_rank": rk["a_app"][m],
                "b_app_rank": rk["b_app"][m],
                "min_app_rank": min(rk["a_app"][m], rk["b_app"][m]),
                "a_pick_top1": 1.0 if rk["a_pick"][m] <= 1 else 0.0,
                "b_pick_top1": 1.0 if rk["b_pick"][m] <= 1 else 0.0,
                "a_ban_top1": 1.0 if rk["a_ban"][m] <= 1 else 0.0,
                "b_ban_top1": 1.0 if rk["b_ban"][m] <= 1 else 0.0,
                "both_pick_top1": 1.0 if (rk["a_pick"][m] <= 1 and rk["b_pick"][m] <= 1) else 0.0,
                "both_pick_top2": 1.0 if (rk["a_pick"][m] <= 2 and rk["b_pick"][m] <= 2) else 0.0,
                "both_ban_top1": 1.0 if (rk["a_ban"][m] <= 1 and rk["b_ban"][m] <= 1) else 0.0,
                "both_ban_top2": 1.0 if (rk["a_ban"][m] <= 2 and rk["b_ban"][m] <= 2) else 0.0,
                # ---- 策略交互：自己喜欢的图 + 对方不 ban 的图 ----
                "a_keep": a_pick * (1.0 - b_ban),
                "b_keep": b_pick * (1.0 - a_ban),
                "both_keep": (a_pick + b_pick) * (1.0 - max(a_ban, b_ban)),
                "own_pick_free": (a_pick + b_pick) * (1.0 - a_ban - b_ban + a_ban * b_ban),
                "opp_pick_free": (a_pick * (1.0 - a_ban)) + (b_pick * (1.0 - b_ban)),
            })
        return rows

    # ---- veto 顺序模拟（贪心） ----
    def simulate_veto(self, teamA, teamB, pool, bo=3, first=None, shape="FSFSFS"):
        """按 veto 顺序贪心模拟，返回预测「最终图集」= 双方 pick + 剩余决胜图。

        顺序（实测 1,581 场：bo3 里 **77.3%** 是 FSFSFS、22.7% 是 FSFSSF；bo5 全为 FSFSFS）：
          bo3 FSFSFS: banA banB pickA pickB banA banB leftover
          bo3 FSFSSF: banA banB pickA pickB banB banA leftover
          bo5      : banA banB pickA pickB pickA pickB banB banA leftover
        这只用「该场之前」的统计，故可用于预测。
        """
        u = len(pool)
        remaining = list(pool)
        taken = set()
        first = first or teamA

        def take(t, mode):
            if not remaining:
                return None
            if mode == "ban":
                m = max(remaining, key=lambda x: (self.team_ban(t, x, u), x))
            else:
                m = max(remaining, key=lambda x: (self.team_pick(t, x, u), x))
            remaining.remove(m)
            taken.add(m)
            return m

        second = teamB if first == teamA else teamA
        seq = [(first, "ban"), (second, "ban")]
        if bo >= 3:
            seq += [(first, "pick"), (second, "pick")]
        if bo >= 5:
            seq += [(first, "pick"), (second, "pick")]
            seq += [(second, "ban"), (first, "ban")]
        elif shape == "FSFSSF":
            seq += [(second, "ban"), (first, "ban")]
        else:
            seq += [(first, "ban"), (second, "ban")]
        picks = []
        for t, mode in seq:
            m = take(t, mode)
            if mode == "pick" and m is not None:
                picks.append(m)
        # 剩余的即决胜图
        leftover = list(remaining)
        result = picks + leftover
        return result

    def sim_features(self, teamA, teamB, pool, bo=3, first_prob=None, combine="mean",
                     first=None, shape="FSFSFS"):
        """两种先后手的模拟的加权并集。

        `combine="mean"`：旧口径（0/0.5/1）。
        `combine="hier"`：用**实测先后手先验**（team2 先 ban 约 63.6%）加权，且对
        「双方都愿意打」的图额外加分（两套顺序都留下 = 更可能在图集里）。
        """
        def sets(first_actor):
            if shape == "mix":
                w = SHAPE_MIX
                return (set(self.simulate_veto(teamA, teamB, pool, bo, first=first_actor,
                                               shape="FSFSFS")),
                        w, 1.0 - w,
                        set(self.simulate_veto(teamA, teamB, pool, bo, first=first_actor,
                                               shape="FSFSSF")))
            return (set(self.simulate_veto(teamA, teamB, pool, bo, first=first_actor,
                                           shape=shape)), 1.0, 0.0, None)

        def weight(container, w, w2, other, m):
            v = (w if m in container else 0.0)
            if other is not None:
                v += (w2 if m in other else 0.0)
            return v

        if first is not None:
            # 确定性先后手（实测：先 ban 方 = 排名较低者，命中 66.5%）
            c, w, w2, o = sets(first)
            return {m: weight(c, w, w2, o, m) for m in pool}
        p2 = FIRST_BAN_TEAM2 if first_prob is None else float(first_prob)
        c1, w1, w12, o1 = sets(teamA)
        c2, w2_, w22, o2 = sets(teamB)
        out = {}
        for m in pool:
            a = weight(c1, w1, w12, o1, m)
            b = weight(c2, w2_, w22, o2, m)
            if combine == "hier":
                out[m] = (1.0 - p2) * a + p2 * b
            else:
                out[m] = (a + b) / 2.0
        return out

    # ---- 注入（冷门队伍 cold-start） ----
    def seed_team(self, team, app_rates=None, win_rates=None, picks=None,
                  exposures=0.0, weight=8.0):
        """把「虚拟档案」注入成伪计数（不覆盖真实历史）。

        `app_rates[m]` 该图进入图集的概率（0-1）；`win_rates[m]` 图级胜率；
        `picks[m]` 该队 pick 该图的相对次数。`weight` = 等效场次（越小越像「新队」）。
        """
        app_rates = dict(app_rates or {})
        win_rates = dict(win_rates or {})
        picks = dict(picks or {})
        n = float(weight)
        for m, r in app_rates.items():
            # 只写该队自己的分母/分子；**不动全局统计**（全局只反映真实历史）
            self.tpooln[team][m] += n
            self.tapp[team][m] += n * float(r)
        for m, r in win_rates.items():
            self.tp[team][m] += n
            self.tw[team][m] += n * float(r)
        tot = sum(float(v) for v in picks.values())
        if tot > 0:
            for m, v in picks.items():
                self.tpick[team][m] += n * float(v) / tot * len(picks)
                self.npick[team] += n * float(v) / tot * len(picks)
        _ = exposures   # 目前不消费；保留参数以便后续按样本量收缩

    # ---- 更新 ----
    def observe_match(self, teamA, teamB, pool, map_set, veto_rec=None, map_results=()):
        pool = list(pool or [])
        map_set = list(map_set or [])
        for m in pool:
            self.pooln[m] += 1.0
            self.tpooln[teamA][m] += 1.0
            self.tpooln[teamB][m] += 1.0
            self.seen_pool.add(m)
        for m in map_set:
            self.app[m] += 1.0
            self.tapp[teamA][m] += 1.0
            self.tapp[teamB][m] += 1.0
        if veto_rec:
            for _step, kind, team, m in veto_rec["order"]:
                if kind == "re" and team:
                    self.tban[team][m] += 1.0
                    self.gban[m] += 1.0
                    self.nban[team] += 1.0
                elif kind == "pi" and team:
                    self.tpick[team][m] += 1.0
                    self.gpick[m] += 1.0
                    self.npick[team] += 1.0
        for mname, s1, s2 in map_results:
            if s1 == s2:
                continue
            self.tp[teamA][mname] += 1.0
            self.tp[teamB][mname] += 1.0
            if s1 > s2:
                self.tw[teamA][mname] += 1.0
            else:
                self.tw[teamB][mname] += 1.0


# ------------------------------------------------------------- 先手方（排名）
_RANKS = None


def ranks():
    """周排名单例（`pool_experiment.Ranks`：按比赛日取最近一期快照）。"""
    global _RANKS
    if _RANKS is None:
        import pool_experiment as PE
        _RANKS = PE.Ranks()
    return _RANKS


def first_banner(teamA, teamB, date):
    """先 ban 方 = **周排名较低**者（主理人 2026-10-11 给的规则；实测命中 66.5%）。

    名次数字越大越弱；积分 0 = 未上榜（视为最弱）。无法判定时返回 None（调用方退回 team2 先验）。
    """
    if not date:
        return None
    r1, p1 = ranks().rank_points(teamA, date)
    r2, p2 = ranks().rank_points(teamB, date)
    if r1 == r2:
        return None
    return teamA if r1 > r2 else teamB


# ------------------------------------------------------- 数据构建（walk-forward）
def build_dataset(rows_bucket, truths, veto, hist, pool_of):
    """对一个 bucket 逐日推进：先取特征，再更新历史。返回样本列表。

    `rows_bucket` 必须已按 `split.split` 分好；`pool_of(row)` 给出该场池。
    返回每场：{id, date, teamA, teamB, bo, pool, rows:[特征...], y:[0/1...]}
    """
    order = sorted(range(len(rows_bucket)), key=lambda i: (rows_bucket[i]["date"], rows_bucket[i]["id"]))
    samples = []
    i = 0
    while i < len(order):
        d = rows_bucket[order[i]]["date"]
        group = []
        while i < len(order) and rows_bucket[order[i]]["date"] == d:
            group.append(rows_bucket[order[i]])
            i += 1
        hist.advance(d)
        pending = []
        for r in group:
            tA, tB = r["team1"]["name"], r["team2"]["name"]
            t = truths.get(r["id"]) or {}
            pool = pool_of(r)
            bo = t.get("bo") or r.get("best_of")
            feats = hist.features_for(tA, tB, pool, bo or 3, date=r.get("date"))
            tset = set(t.get("maps") or [])
            pending.append({
                "id": r["id"], "date": d, "teamA": tA, "teamB": tB,
                "bo": t.get("bo") or r.get("best_of"), "pool": pool,
                "rows": feats, "y": [1 if f["map"] in tset else 0 for f in feats],
            })
        samples.extend(pending)
        for r in group:
            tA, tB = r["team1"]["name"], r["team2"]["name"]
            t = truths.get(r["id"]) or {}
            pool = pool_of(r)
            mres = [(m["name"], m["team1_score"], m["team2_score"]) for m in (r.get("maps") or [])]
            hist.observe_match(tA, tB, pool, t.get("maps") or [], veto.get(r["id"]), mres)
    return samples


def build_frozen(bucket, truths, hist, pool_of):
    """冻结历史版本（`--mode train_only` 用）：不推进、只用已训练好的 hist。"""
    out = []
    for r in bucket:
        tA, tB = r["team1"]["name"], r["team2"]["name"]
        t = truths.get(r["id"]) or {}
        pool = pool_of(r)
        bo = t.get("bo") or r.get("best_of")
        feats = hist.features_for(tA, tB, pool, bo or 3, date=r.get("date"))
        tset = set(t.get("maps") or [])
        out.append({"id": r["id"], "date": r["date"], "teamA": tA, "teamB": tB,
                    "bo": t.get("bo") or r.get("best_of"), "pool": pool,
                    "rows": feats, "y": [1 if f["map"] in tset else 0 for f in feats]})
    return out


# ------------------------------------------------------------------ 模型
def train(samples, features=FEATURES, C=1.0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    X = np.array([[f[fn] for fn in features] for s in samples for f in s["rows"]], dtype=float)
    y = np.array([v for s in samples for v in s["y"]], dtype=int)
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=4000, C=C).fit(sc.transform(X), y)
    return {"scaler": sc, "clf": clf, "features": features}


def score_maps(model, sample):
    """返回 {map: prob}（概率只用于组内排序）。"""
    import numpy as np
    feats = model["features"]
    X = np.array([[f[fn] for fn in feats] for f in sample["rows"]], dtype=float)
    p = model["clf"].predict_proba(model["scaler"].transform(X))[:, 1]
    return {r["map"]: float(v) for r, v in zip(sample["rows"], p)}


def k_of(sample):
    return max(1, min(int(sample["bo"] or 3), 5))


def predict_set(model, sample, k=None):
    k = k if k is not None else k_of(sample)
    probs = score_maps(model, sample)
    ranked = sorted(probs.items(), key=lambda kv: (-kv[1], kv[0]))
    return [m for m, _ in ranked[:k]], probs


# ------------------------------------------------------------------ 评估
def evaluate(model, samples):
    """集合命中口径的选图得分（每张 5 分）+ 命中率诊断。"""
    n = hits = slots = 0
    per_slot_hit = collections.Counter()
    for s in samples:
        k = k_of(s)
        picked, _ = predict_set(model, s, k)
        truth = set(f["map"] for f, v in zip(s["rows"], s["y"]) if v)
        hits += sum(1 for m in picked if m in truth)
        slots += k
        n += 1
        for j, m in enumerate(picked):
            per_slot_hit[j] += 1 if m in truth else 0
    return {"n": n, "points": 5 * hits, "hits": hits, "slots": slots,
            "hit_rate": hits / slots if slots else 0.0, "per_slot": dict(per_slot_hit)}


def evaluate_rank(samples, keyfn):
    """通用对照臂：用 keyfn(feature)->score 组内 top-k。"""
    n = hits = slots = 0
    for s in samples:
        k = k_of(s)
        ranked = sorted(s["rows"], key=lambda f: (-keyfn(f), f["map"]))[:k]
        truth = set(f["map"] for f, v in zip(s["rows"], s["y"]) if v)
        hits += sum(1 for f in ranked if f["map"] in truth)
        slots += k
        n += 1
    return {"n": n, "points": 5 * hits, "hits": hits, "slots": slots,
            "hit_rate": hits / slots if slots else 0.0}


def run(mode="walk_forward", verbose=True, half_life=HALF_LIFE):
    rows = SPLIT.load(MATCHES)
    exclude = SPLIT.load_exclude(EXCLUDE) if os.path.exists(EXCLUDE) else None
    buckets, meta = SPLIT.split(rows, purge=True, exclude=exclude)
    truths = {t["id"]: t for t in load_jsonl(TRUTH)}
    veto = load_veto()

    import pool_experiment as PE
    eras, by_pool = PE.load_pool_eras()
    tl = json.load(io.open(PE.TIMELINE, encoding="utf-8"))

    def pool_of(r):
        t = truths.get(r["id"]) or {}
        vp = t.get("pool")
        if vp:
            return list(vp)
        _era, pool = PE.era_and_pool(r["date"], [], eras, by_pool, tl)
        return list(pool)

    hist = MapHistory(half_life=half_life)
    train_samples = build_dataset(buckets["train"], truths, veto, hist, pool_of)
    if mode == "train_only":
        valid_samples = build_frozen(buckets["valid"], truths, hist, pool_of)
        test_samples = build_frozen(buckets["test"], truths, hist, pool_of)
    else:
        valid_samples = build_dataset(buckets["valid"], truths, veto, hist, pool_of)
        test_samples = build_dataset(buckets["test"], truths, veto, hist, pool_of)

    best = None
    for C in (0.03, 0.1, 0.3, 1.0, 3.0):
        m = train(train_samples, C=C)
        ev = evaluate(m, valid_samples)
        if verbose:
            print("  C=%-5s valid 选图 = %d 分（命中率 %.1f%%）" % (C, ev["points"], 100 * ev["hit_rate"]))
        if best is None or ev["points"] > best[1]["points"]:
            best = (C, ev, m)
    C, ev_valid, model = best
    ev_test = evaluate(model, test_samples)

    # 对照臂（同一批样本、同一评估器）
    arms = {
        "freq_g_app": lambda f: f["g_app"],                                # 旧基线：池内全局频次
        "freq_team_sum": lambda f: f["a_app"] + f["b_app"],                # 只加队伍亲和
        "veto_free": lambda f: f["a_pick"] + f["b_pick"] - f["a_ban"] - f["b_ban"],
    }
    arm_scores = {"full_model": {"valid": ev_valid, "test": ev_test}}
    for name, fn in arms.items():
        arm_scores[name] = {"valid": evaluate_rank(valid_samples, fn),
                            "test": evaluate_rank(test_samples, fn)}

    out = {
        "mode": mode, "split": meta["counts"], "half_life": half_life,
        "chosen_C": C, "valid": ev_valid, "test": ev_test,
        "arms": arm_scores, "features": FEATURES,
    }
    if verbose:
        print("")
        print("  [对照臂 test 选图分] " + "  ".join(
            "%s=%d(%.1f%%)" % (k, v["test"]["points"], 100 * v["test"]["hit_rate"])
            for k, v in arm_scores.items()))
        print("")
        print("== veto_model（%s）==" % mode)
        print("  选定 C = %s，半衰期 %.0f 天" % (C, half_life))
        print("  valid 选图 %d 分 / 命中率 %.1f%%" % (ev_valid["points"], 100 * ev_valid["hit_rate"]))
        print("  test  选图 %d 分 / 命中率 %.1f%%  (槽位 %d)" % (
            ev_test["points"], 100 * ev_test["hit_rate"], ev_test["slots"]))
        print("  各槽位命中: %s" % ev_test["per_slot"])
    with io.open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    return out


# ------------------------------------------------------------------ selftest
def _selftest():
    fails = []
    n_chk = [0]                      # 2026-10-11：此前这里打印的是**硬编码 20**（实际条数更多）

    def chk(name, got, want):
        n_chk[0] += 1
        if got != want:
            fails.append("%s: got=%r want=%r" % (name, got, want))

    h = MapHistory()
    # 未见图 ⇒ 池内均匀先验，绝不为 0
    chk("unseen g_app", round(h.g_app("Cache"), 6), round(UNIFORM_RATE, 6))
    chk("unseen team_app", round(h.team_app("A", "Cache"), 6), round(UNIFORM_RATE, 6))
    chk("unseen ban > 0", h.team_ban("A", "Cache", 7) > 0, True)
    chk("days_apart", days_apart("2026-03-01", "2026-02-01"), 31)

    # 出现率：Mirage 每场都在、Anubis 从不出现 ⇒ 前者更高
    for _ in range(10):
        h.observe_match("A", "B", ["Mirage", "Anubis", "Nuke"], ["Mirage", "Nuke"],
                        {"order": [(1, "re", "A", "Anubis"), (2, "pi", "A", "Mirage")]})
    chk("mirage app > anubis", h.g_app("Mirage") > h.g_app("Anubis"), True)
    chk("a bans anubis more", h.team_ban("A", "Anubis", 3) > h.team_ban("A", "Nuke", 3), True)
    chk("a picks mirage more", h.team_pick("A", "Mirage", 3) > h.team_pick("A", "Nuke", 3), True)

    # features() 形状与 new 位
    feats = h.features("A", "B", ["Mirage", "Anubis", "Cache"])
    chk("features size", len(feats), 3)
    # 新图定义 = 「从未进过池」；Anubis 进过池（只是总被 ban）⇒ 不算新
    chk("new flag", [f["new"] for f in feats], [0.0, 0.0, 1.0])
    chk("feature keys >= base", set(FEATURES) | {"map"} <= set(feats[0].keys()), True)
    chk("feature keys == v2+sim3",
        set(feats[0].keys()), set(FEATURES2) | {"map", "sim3_in_set"})
    # 先手规则：名次数字大者先 ban；并列/未知 ⇒ None
    chk("first_banner 返回值合法",
        first_banner("A", "B", "2026-01-01") in (None, "A", "B"), True)

    # 有偏好差异的夹具：A/B 各自有「最爱 ban / 最爱 pick」，且**互相抢同一张图**
    POOL7 = ["Mirage", "Anubis", "Cache", "Dust2", "Inferno", "Nuke", "Ancient"]
    h3 = MapHistory()

    def seed(team, bans, picks, strength=3.0):
        for m, k in bans:
            h3.tban[team][m] += k * strength
            h3.gban[m] += k * strength
            h3.nban[team] += k * strength
        for m, k in picks:
            h3.tpick[team][m] += k * strength
            h3.gpick[m] += k * strength
            h3.npick[team] += k * strength

    # 两队首选同一张图（Mirage）⇒ **pick 顺序**决定谁抢到它
    seed("A", [("Cache", 1.0)], [("Mirage", 1.0), ("Dust2", 0.8), ("Nuke", 0.6)])
    seed("B", [("Ancient", 1.0)], [("Mirage", 1.0), ("Inferno", 0.8), ("Nuke", 0.7)])
    set_a_first = set(h3.simulate_veto("A", "B", POOL7, 3, first="A"))
    set_b_first = set(h3.simulate_veto("A", "B", POOL7, 3, first="B"))
    chk("先手方不同 ⇒ 图集不同（谁先 pick 谁拿到 Mirage）", set_a_first != set_b_first, True)
    chk("先手方 A：A 拿到 Mirage", "Mirage" in set_a_first, True)
    chk("先手方 B：B 拿到 Mirage（A 转抢第二顺位）", "Mirage" in set_b_first, True)
    # 型态（ban 对内部顺序）**不影响**结果：本模型 ban 逐队独立
    chk("ban 对内部顺序对本模型无影响（FSFSFS == FSFSSF）",
        h3.simulate_veto("A", "B", POOL7, 3, first="A", shape="FSFSFS"),
        h3.simulate_veto("A", "B", POOL7, 3, first="A", shape="FSFSSF"))
    # 确定性先手 ⇒ 0/1 权重；无 first ⇒ 两序加权（非全 0/1）
    d3 = h3.sim_features("A", "B", POOL7, 3, first="A")
    chk("deterministic first => 0/1 权重",
        sorted(set(round(v, 4) for v in d3.values())) in ([0.0], [1.0], [0.0, 1.0]), True)
    soft = h3.sim_features("A", "B", POOL7, 3, combine="hier")
    chk("无 first ⇒ 加权（存在非 0/1 权重）", any(0.0 < v < 1.0 for v in soft.values()), True)
    chk("确定性先手 = 该手方那一支", d3, h3.sim_features("A", "B", POOL7, 3, first="A"))

    # 接线坑守卫（2026-10-11）：features() 不给 first ⇒ sim3 退化成 sim2（同一列）。
    # 真实队名 + 真实日期做夹具（Nemiga 弱于 ARCRED ⇒ 先 ban 方 = ARCRED）。
    TA, TB, DD = "Nemiga", "ARCRED", "2026-03-10"
    chk("真实夹具 first_banner 可判定", first_banner(TA, TB, DD), "ARCRED")
    h4 = MapHistory()
    _seed4 = [(TA, [("Cache", 1.0)], [("Mirage", 1.0), ("Dust2", 0.8), ("Nuke", 0.6)]),
              (TB, [("Ancient", 1.0)], [("Mirage", 1.0), ("Inferno", 0.8), ("Nuke", 0.7)])]
    for _t, _b, _p in _seed4:
        for _m, _k in _b:
            h4.tban[_t][_m] += 3.0 * _k; h4.gban[_m] += 3.0 * _k; h4.nban[_t] += 3.0 * _k
        for _m, _k in _p:
            h4.tpick[_t][_m] += 3.0 * _k; h4.gpick[_m] += 3.0 * _k; h4.npick[_t] += 3.0 * _k
    _nof = h4.features(TA, TB, POOL7, 3)
    _for = h4.features_for(TA, TB, POOL7, 3, date=DD)
    _exp = h4.features(TA, TB, POOL7, 3, first=first_banner(TA, TB, DD))
    chk("坑：无 first 时 sim3 退化为 sim2",
        [f["sim3_in_set"] for f in _nof], [f["sim2_in_set"] for f in _nof])
    chk("features_for = 显式 first",
        [f["sim3_in_set"] for f in _for], [f["sim3_in_set"] for f in _exp])
    chk("带日期时 sim3 ≠ sim2（规则真的进了特征）",
        [f["sim3_in_set"] for f in _for] != [f["sim2_in_set"] for f in _for], True)

    # 衰减：过 1 个半衰期后老统计减半 ⇒ 出现率向均匀先验收缩
    before = h.g_app("Mirage")
    h.decay(HALF_LIFE)
    after = h.g_app("Mirage")
    chk("decay pulls toward prior", abs(after - UNIFORM_RATE) < abs(before - UNIFORM_RATE), True)
    # 半天衰期 + 大样本 ⇒ 恰好约在 before 与 prior 的中点附近（只验证单调收敛）
    chk("decay monotone", after < before, True)

    # 注入（冷门队伍）：只会抬高该队自己的统计，不动全局
    h2 = MapHistory()
    g_before = h2.g_app("Mirage")
    h2.seed_team("Cold", app_rates={"Mirage": 1.0}, win_rates={"Mirage": 0.8},
                 picks={"Mirage": 5.0}, weight=8.0)
    chk("seed raises team_app", h2.team_app("Cold", "Mirage") > UNIFORM_RATE, True)
    chk("seed sets winrate", round(h2.team_wr("Cold", "Mirage"), 3), round((6.4 + 2.0) / (8.0 + 4.0), 3))
    chk("seed keeps global app", h2.g_app("Mirage"), g_before)
    chk("seed feeds features", h2.features("Cold", "X", ["Mirage"])[0]["a_app"] > UNIFORM_RATE, True)

    # 池内 top-k 排序：玩具模型（单特征 g_app）
    class _Scaler:
        def transform(self, X):
            return X

    class _Clf:
        def predict_proba(self, X):
            import numpy as np
            p = 1.0 / (1.0 + np.exp(-10 * (X[:, 0] - 0.4)))
            return np.stack([1 - p, p], axis=1)

    toy = {"features": FEATURES, "scaler": _Scaler(), "clf": _Clf()}
    sample = {"bo": 3, "rows": h.features("A", "B", ["Mirage", "Anubis", "Cache"])}
    picked, probs = predict_set(toy, sample, 2)
    chk("top1 is mirage", picked[0], "Mirage")
    chk("probs keyed by map", set(probs.keys()), {"Mirage", "Anubis", "Cache"})

    if fails:
        print("VETO_SELFTEST_FAIL")
        for f in fails:
            print("  -", f)
        return 1
    print("VETO_SELFTEST_OK (%d checks)" % n_chk[0])
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "selftest":
        raise SystemExit(_selftest())
    if cmd == "run":
        mode = "walk_forward"
        if "--mode" in sys.argv:
            mode = sys.argv[sys.argv.index("--mode") + 1]
        run(mode)
        raise SystemExit(0)
    print(__doc__)
    raise SystemExit(2)
