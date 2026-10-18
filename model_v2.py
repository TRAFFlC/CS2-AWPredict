#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""model_v2.py —— issue-014 的 **时序强度 + 选手级 rating** 预测核心（2026-10-10）。

四项的建模分工：
  1. 获胜方 —— **Elo 时序强度**（二值 + 比分两种口径）→ Logistic（特征只有 2 个，抗过拟合）
  2. 获胜比分 —— 按预测胜率的**条件比分**（强队更可能 2:0，接近局 2:1），阈值在 valid 上选
  3. BP 选图 —— 由 `veto_model.py` 负责（本模块 import 复用）
  4. MVP —— **选手级判别模型**：对「预测胜方」的 5 人阵容逐个估 P(该选手是队内系列第一)

相对 `baseline_predictor.py`（四项基线，test 5,770 / 13,540 = 42.6%）的实测增益见 `model_v2_results.json`。

时间安全
--------
单次按日期推进：当天比赛先取特征/预测，再统一更新状态（Elo / 衰减历史 / 选手 rating）。
系数用 **train** 段拟合，超参在 **valid** 上选，**test 只报一次**。同日多场互不可见。

冷门队伍（cold-start）
----------------------
`State.seed_team()` 可把映射器（`coldstart.py`）推出的虚拟档案注入状态；
`State.alias()` 可把新队名直接指向已有队伍。注入只填补空缺，不覆盖真实历史。

用法
----
  python -X utf8 model_v2.py selftest
  python -X utf8 model_v2.py run
"""
from __future__ import annotations

import collections
import io
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")  # # 仓库布局：数据与代码分目录（Alpha 1.1.1）
sys.path.insert(0, HERE)
sys.path.insert(0, HERE)  # scorer.py 与本模块同目录（仓库布局）

import split as SPLIT  # noqa: E402
import pool_experiment as PE  # noqa: E402
import veto_model as VM  # noqa: E402
from scorer import Prediction, Truth, score_match, max_total  # noqa: E402

MATCHES = os.path.join(DATA, "csapi_matches_all.jsonl")
TRUTH = os.path.join(DATA, "truth_maps_B.jsonl")
STATS = os.path.join(DATA, "csapi_stats_all.jsonl")
EXCLUDE = os.path.join(HERE, "exclude_band_unknown_ids.txt")
OUT = os.path.join(DATA, "model_v2_results.json")

HALF_LIFE_HIST = 120.0     # 历史胜率/交手的半衰期（天）
HALF_LIFE_FORM = 45.0      # 「近期状态」胜率的半衰期（天）——比历史胜率快得多
HALF_LIFE_PLAYER = 90.0    # 选手 rating 的半衰期（天）
LINEUP_WINDOW = 240        # 阵容候选：近 N 天内为该队出场过
ELO0 = 1500.0
K_NEW = 40.0               # 比赛数少时的 K（新队/新阵容适应快）
K_OLD = 30.0               # 比赛数多时的 K
K_SAT = 30.0               # K 的饱和尺度
OPP_ADJ = 0.35             # 选手 rating 的对手强度校正系数（每 400 Elo）
SEED_W = 6.0               # 冷门队伍注入选手时的「等效场次」
# 阵容变动感知（2026-10-11 新增）
CHURN_WINDOW = 180.0       # 换人影响窗口（天）
CHURN_HALF = 90.0          # 换人权重的半衰期（天）
RHO = 0.35                 # 每次换 1/5 人时 Elo 向均值回归的比例（valid 上选）
RHO_G = 0.0                # 同一事件对「有效场次」的缩放（0 = 不动 K）
# 选手级「图偏好」（2026-10-11 新增）：由 csapi `by_map` 的逐图选手 rating 学「某队在某图有多强」
K_MAP = 3.0                # (team,map) 性能的收缩尺度（等效观测数；0 观测 ⇒ 退回该队总体水平）

# 第 1 项特征：只用 Elo 两口径（valid 上选出的最简且最优组合）
WIN_FEATURES = ["d_elo", "d_elom"]
WIN_C = 0.05

# 候选特征组（valid 上选，test 只报一次）
_E2 = ["d_elo", "d_elom"]
WIN_FEATURE_SETS = [
    ("elo2", _E2),
    ("rank", ["d_rank"]),
    ("rank+logp", ["d_rank", "d_logp"]),
    ("elo2+rank", _E2 + ["d_rank"]),
    ("elo2+logp", _E2 + ["d_logp"]),
    ("elo2+rank+logp", _E2 + ["d_rank", "d_logp"]),
    ("elo2+rank+h2h", _E2 + ["d_rank", "d_h2h"]),
    ("elo2+rank+wr", _E2 + ["d_rank", "d_wr"]),
    ("elo2+rank+lineup", _E2 + ["d_rank", "d_lineup"]),
    ("elo2+rank+poolwr", _E2 + ["d_rank", "d_poolwr"]),
    ("elo2+rank+churn", _E2 + ["d_rank", "d_churn"]),
    ("elo2+rank+churn2", _E2 + ["d_rank", "d_churn", "sum_churn"]),
    ("elo2+rank+h2h+lineup", _E2 + ["d_rank", "d_h2h", "d_lineup"]),
    ("...+form", _E2 + ["d_rank", "d_h2h", "d_lineup", "d_form"]),
    ("...+rest", _E2 + ["d_rank", "d_h2h", "d_lineup", "d_rest"]),
    ("...+games", _E2 + ["d_rank", "d_h2h", "d_lineup", "d_games"]),
    ("...+form+rest", _E2 + ["d_rank", "d_h2h", "d_lineup", "d_form", "d_rest"]),
    ("...+form+rest+games", _E2 + ["d_rank", "d_h2h", "d_lineup", "d_form", "d_rest", "d_games"]),
    ("...+form+games", _E2 + ["d_rank", "d_h2h", "d_lineup", "d_form", "d_games"]),
    ("core+form+rank", _E2 + ["d_rank", "d_h2h", "d_lineup", "d_form", "d_rank"]),
    ("el2+rank+form", _E2 + ["d_rank", "d_form"]),
    ("all13", _E2 + ["d_rank", "d_logp", "d_wr", "d_h2h", "d_poolwr", "d_lineup",
                     "d_churn", "sum_churn", "d_form", "d_rest", "d_games"]),
    ("elo2+churn", _E2 + ["d_churn"]),
    ("elo2+churn2", _E2 + ["d_churn", "sum_churn"]),
    ("elo2+h2h", _E2 + ["d_h2h"]),
    ("elo2+lineup", _E2 + ["d_lineup"]),
    ("all8", ["d_elo", "d_elom", "d_rank", "d_logp", "d_wr", "d_h2h", "d_poolwr", "d_lineup"]),
    ("all10", ["d_elo", "d_elom", "d_rank", "d_logp", "d_wr", "d_h2h", "d_poolwr",
               "d_lineup", "d_churn", "sum_churn"]),
    # 2026-10-11：选手级图偏好（`by_map` 逐图选手 rating；此前被整块丢弃）
    ("...+poolperf", _E2 + ["d_rank", "d_h2h", "d_lineup", "d_poolperf"]),
    ("...+poolperf+best", _E2 + ["d_rank", "d_h2h", "d_lineup", "d_poolperf", "d_bestperf"]),
    ("all13+perf", _E2 + ["d_rank", "d_logp", "d_wr", "d_h2h", "d_poolwr", "d_lineup",
                          "d_churn", "sum_churn", "d_form", "d_rest", "d_games",
                          "d_poolperf", "d_bestperf", "d_perfspread"]),
]

# 第 4 项特征
MVP_FEATURES = ["adj", "raw", "toprate", "d_team", "rank", "exp", "d_opp", "days"]


def load_jsonl(path):
    out = []
    for line in io.open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def load_stats(path=STATS):
    """matchid -> {team_name: [(player, rating), ...]}"""
    out = {}
    for rec in load_jsonl(path):
        block = None
        for b in (rec.get("overall") or []):
            if b.get("name") == "All":
                block = b
                break
        if block is None:
            continue
        per = {}
        for key in ("team1", "team2"):
            t = block.get(key) or {}
            name = t.get("name")
            if name:
                per[name] = [(p.get("name"), p.get("rating")) for p in (t.get("players") or [])]
        if per:
            out[rec.get("matchid")] = per
    return out


_MAP_STATS_CACHE = {}


def load_stats_map(path=STATS):
    """matchid -> {map_name: {team_name: [(player, rating), ...]}}（csapi `by_map`）。

    这是**之前被丢掉的信息**：`load_stats` 只读 `overall`，而每场每个打过的图都有
    逐选手 rating ⇒ 可以学「该队在该图上的表现水平」（选手级图偏好）。
    """
    key = (path, os.path.getmtime(path) if os.path.exists(path) else 0)
    if key in _MAP_STATS_CACHE:
        return _MAP_STATS_CACHE[key]
    out = {}
    for rec in load_jsonl(path):
        per_map = {}
        for blk in (rec.get("by_map") or []):
            mname = blk.get("name")
            if not mname or mname == "All":
                continue
            per_team = {}
            for k in ("team1", "team2"):
                t = blk.get(k) or {}
                name = t.get("name")
                if name:
                    per_team[name] = [(p.get("name"), p.get("rating"))
                                      for p in (t.get("players") or [])]
            if per_team:
                per_map[mname] = per_team
        out[rec.get("matchid")] = per_map
    _MAP_STATS_CACHE[key] = out
    return out


def mvp_label(match, stats):
    """Q-008 口径真值：胜方队内系列 rating 最高者。"""
    per = stats.get(match["id"])
    if not per:
        return ""
    best = None
    for name, rating in (per.get(match["winner"]["name"]) or []):
        if name is None or rating is None:
            continue
        if best is None or float(rating) > best[1]:
            best = (name, float(rating))
    return best[0] if best else ""


def _decay_f(days, half_life):
    return 0.5 ** (max(0.0, days) / half_life) if half_life > 0 else 1.0


# ------------------------------------------------------------------ 状态
class State:
    """逐日推进的强度状态：Elo（两种口径）+ 衰减历史 + 选手 rating + 阵容。"""

    def __init__(self, half_life_hist=HALF_LIFE_HIST, half_life_player=HALF_LIFE_PLAYER,
                 k_new=K_NEW, k_old=K_OLD, k_sat=K_SAT, elo0=ELO0, opp_adj=OPP_ADJ,
                 rho=RHO, rho_g=RHO_G, churn_window=CHURN_WINDOW, churn_half=CHURN_HALF,
                 half_life_form=HALF_LIFE_FORM):
        self.hl_hist = half_life_hist
        self.hl_player = half_life_player
        self.hl_form = half_life_form
        self.k_new = k_new
        self.k_old = k_old
        self.k_sat = k_sat
        self.elo0 = elo0
        self.opp_adj = opp_adj
        self.rho = rho
        self.rho_g = rho_g
        self.churn_window = churn_window
        self.churn_half = churn_half
        self.last_date = None
        # Elo
        self.elo = collections.defaultdict(lambda: elo0)
        self.elo_m = collections.defaultdict(lambda: elo0)
        self.games = collections.defaultdict(int)
        # 衰减历史
        self.mw = collections.defaultdict(lambda: [0.0, 0.0])     # 系列 胜/负
        self.mf = collections.defaultdict(lambda: [0.0, 0.0])     # 近期状态 胜/负（半衰期更短）
        self.last_match = {}                                      # team -> 最近一场日期
        self.mapw = collections.defaultdict(lambda: [0.0, 0.0])   # (team,map)
        self.h2h_hist = collections.defaultdict(lambda: [0.0, 0.0])
        # 选手
        self.p_sum = collections.defaultdict(float)      # 衰减加权 rating 和
        self.p_w = collections.defaultdict(float)        # 权重和
        self.p_adj = collections.defaultdict(float)      # 对手校正后的加权和
        self.p_play = collections.defaultdict(float)     # 出场次数（衰减）
        self.p_top = collections.defaultdict(float)      # 队内第一次数（衰减）
        self.p_team = {}
        self.p_date = {}
        self.team_players = collections.defaultdict(dict)   # team -> player -> last_date
        # 阵容变动（roster churn）
        self.roster_sig = {}                       # team -> 上一次出场 5 人集合
        self.churn = collections.defaultdict(list)  # team -> [(date, 换人比例)]
        # 选手级图偏好（2026-10-11）：(team,map) 的对手校正 rating 均值 + 队伍总体水平（收缩先验）
        self.pm_sum = collections.defaultdict(float)
        self.pm_w = collections.defaultdict(float)
        self.t_level_sum = collections.defaultdict(float)
        self.t_level_w = collections.defaultdict(float)
        # 注入（cold-start）
        self.aliases = {}
        self.injected = {}

    # ---- 注入 API（冷门队伍） ----
    def alias(self, team, donor):
        self.aliases[team] = donor

    def seed_team(self, team, elo=None, elo_m=None, games=0, players=None,
                  winrate=None, map_rates=None, date=None, weight=SEED_W, toprate=0.4,
                  exp_games=None):
        """把映射器推出的虚拟档案注入状态（不覆盖真实历史）。

        `players` = [(name, rating), ...]；会同时写入选手统计，使第 4 项可用。
        """
        self.injected[team] = {
            "elo": self.elo0 if elo is None else float(elo),
            "elo_m": (self.elo0 if elo_m is None else float(elo_m)),
            "games": int(games),
            # 「经验」项与 K 的饱和场次解耦：冷门队伍的 games 故意给小（K 大、纠正快），
            # 但 d_games 特征应当用**donor 的经验水平**（我们假定它像 donor）。
            "exp_games": int(games if exp_games is None else exp_games),
            "players": list(players or []),
            "winrate": winrate,
            "map_rates": dict(map_rates or {}),
            "date": date,
        }
        for name, rating in (players or []):
            r = float(rating)
            self.p_sum[name] += r * weight
            self.p_w[name] += weight
            self.p_adj[name] += r * weight
            self.p_play[name] += weight
            self.p_top[name] += toprate * weight
            self.p_team[name] = team
            if date:
                self.p_date[name] = date
                self.team_players[team][name] = date

    def _resolve(self, t):
        return self.aliases.get(t, t)

    # ---- 衰减 ----
    def _decay(self, days):
        if days <= 0:
            return
        fh = _decay_f(days, self.hl_hist)
        fp = _decay_f(days, self.hl_player)
        ff = _decay_f(days, self.hl_form)
        for store, f in ((self.mw, fh), (self.mapw, fh), (self.h2h_hist, fh), (self.mf, ff)):
            for k in list(store):
                v = store[k]
                store[k] = [v[0] * f, v[1] * f]
        for store in (self.p_sum, self.p_w, self.p_adj, self.p_play, self.p_top,
                      self.pm_sum, self.t_level_sum):
            for k in list(store):
                store[k] *= fp
        for store in (self.pm_w, self.t_level_w):
            for k in list(store):
                store[k] *= fp

    def advance(self, date):
        if self.last_date is not None:
            self._decay(VM.days_apart(date, self.last_date))
        self.last_date = date

    # ---- 查询 ----
    @staticmethod
    def _rate(w, l, prior=0.5, k=2.0):
        n = w + l
        return prior if n <= 0 else (w + k * prior) / (n + k)

    def winrate(self, t):
        t = self._resolve(t)
        inj = self.injected.get(t)
        if inj and inj["winrate"] is not None and self.mw[t][0] + self.mw[t][1] == 0:
            return inj["winrate"]
        w, l = self.mw[t]
        return self._rate(w, l, 0.5, 2.0)

    def pool_wr(self, t, pool):
        t = self._resolve(t)
        w = sum(self.mapw[(t, m)][0] for m in pool)
        l = sum(self.mapw[(t, m)][1] for m in pool)
        if w + l == 0:
            inj = self.injected.get(t)
            if inj and inj["map_rates"]:
                vals = [inj["map_rates"][m] for m in pool if m in inj["map_rates"]]
                if vals:
                    return sum(vals) / len(vals)
            return self.winrate(t)
        return self._rate(w, l, 0.5, 2.0)

    def h2h_rate(self, a, b):
        a, b = self._resolve(a), self._resolve(b)
        key = (a, b) if a <= b else (b, a)
        wa, wb = self.h2h_hist[key]
        mine, theirs = (wa, wb) if a <= b else (wb, wa)
        return self._rate(mine, theirs, 0.5, 1.0)

    def elo_of(self, t):
        t = self._resolve(t)
        inj = self.injected.get(t)
        if inj and self.games[t] == 0:
            return inj["elo"], inj["games"]
        return self.elo[t], self.games[t]

    def elo_m_of(self, t):
        t = self._resolve(t)
        inj = self.injected.get(t)
        if inj and self.games[t] == 0:
            return inj["elo_m"], inj["games"]
        return self.elo_m[t], self.games[t]

    # ---- 选手 ----
    def player_mean(self, p):
        return self.p_sum[p] / self.p_w[p] if self.p_w[p] > 0 else None

    def player_adj(self, p):
        return self.p_adj[p] / self.p_w[p] if self.p_w[p] > 0 else None

    def player_toprate(self, p):
        n = self.p_play[p]
        return self.p_top[p] / n if n > 0 else 0.0

    def lineup(self, t, date, window=LINEUP_WINDOW, n=5):
        """该队「最可能出场」的阵容：近窗口出场者中按（近度→rating）取前 n。"""
        t = self._resolve(t)
        cands = []
        for p, d in self.team_players.get(t, {}).items():
            if VM.days_apart(date, d) > window:
                continue
            cands.append((d, self.player_mean(p) or 0.0, p))
        if not cands:
            inj = self.injected.get(t)
            if inj and inj["players"]:
                return [p for p, _ in inj["players"][:n]]
            return []
        cands.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return [p for _d, _r, p in cands[:n]]

    def lineup_strength(self, t, date):
        vals = [self.player_adj(p) for p in self.lineup(t, date)]
        vals = [v for v in vals if v is not None]
        if not vals:
            inj = self.injected.get(self._resolve(t))
            if inj and inj["players"]:
                ratings = [r for _n, r in inj["players"][:5]]
                return sum(ratings) / len(ratings)
            return None
        return sum(vals) / len(vals)

    # ---- 选手级图偏好（2026-10-11） ----
    def team_level(self, t):
        """该队「选手总体水平」（衰减加权的对手校正 rating 均值）；无观测返回 None。"""
        t = self._resolve(t)
        w = self.t_level_w[t]
        return self.t_level_sum[t] / w if w > 0 else None

    def map_perf(self, t, mapname, k0=K_MAP):
        """该队在该图上的表现水平（对手校正 rating 均值），向**该队总体水平**收缩。

        0 观测 ⇒ 正好等于总体水平（差分时给出中性的 0）；无历史 ⇒ 1.0（rating 的全局中位）。
        """
        t = self._resolve(t)
        prior = self.team_level(t)
        if prior is None:
            prior = 1.0
        w = self.pm_w[(t, mapname)]
        return (self.pm_sum[(t, mapname)] + k0 * prior) / (w + k0)

    def pool_perf(self, t, pool):
        """池内各图表现水平的均值（未知图用总体水平填充）。"""
        vals = [self.map_perf(t, m) for m in pool] if pool else []
        if not vals:
            lv = self.team_level(t)
            return 1.0 if lv is None else lv
        return sum(vals) / len(vals)

    def player_row(self, team, p, date, opp_elo, lineup):
        """选手级特征（全部来自该场之前）。"""
        adj = self.player_adj(p)
        raw = self.player_mean(p)
        vals = [self.player_adj(x) for x in lineup]
        vals = [v for v in vals if v is not None]
        team_mean = sum(vals) / len(vals) if vals else 0.0
        order = sorted(lineup, key=lambda x: -(self.player_adj(x) if self.player_adj(x) is not None else -9))
        rank = (order.index(p) + 1) if p in order else len(lineup)
        own_elo = self.elo_of(team)[0]
        return {
            "player": p,
            "adj": adj if adj is not None else 0.0,
            "raw": raw if raw is not None else 0.0,
            "toprate": self.player_toprate(p),
            "d_team": (adj if adj is not None else 0.0) - team_mean,
            "rank": float(rank),
            "exp": self.p_w[p],
            "d_opp": (opp_elo - own_elo) / 400.0,
            "days": float(VM.days_apart(date, self.p_date[p])) if p in self.p_date else 999.0,
        }

    # ---- 近期状态 / 休息天数 / 经验 ----
    def form(self, t):
        t = self._resolve(t)
        w, l = self.mf[t]
        return self._rate(w, l, 0.5, 2.0)

    def rest_days(self, t, date):
        t = self._resolve(t)
        d = self.last_match.get(t)
        return 999.0 if d is None else float(abs(VM.days_apart(date, d)))

    def exp_log(self, t):
        """log(1+有效场次)。冷门注入队伍用映射器给的等效场次（否则会被当成「零经验」而系统性看低）。"""
        t = self._resolve(t)
        n = self.games[t]
        if n == 0:
            inj = self.injected.get(t)
            if inj:
                n = inj.get("exp_games", inj.get("games", 0))
        return math.log1p(n)

    # ---- 阵容变动 ----
    @staticmethod
    def snapshot(players):
        return set(n for n, r in (players or []) if n and r is not None)

    def churn_rate(self, t, date):
        """近 CHURN_WINDOW 天内该队的换人强度（指数加权，0~1）。**只用过去记录**，时间安全。"""
        t = self._resolve(t)
        tot = 0.0
        for d, frac in self.churn.get(t, []):
            gap = abs(VM.days_apart(date, d))
            if gap <= self.churn_window:
                tot += frac * _decay_f(gap, self.churn_half)
        return min(1.0, tot)

    # ---- 更新 ----
    def observe(self, date, teamA, teamB, winner, map_results, per_team_players,
                map_stats=None):
        # ---- 阵容变动感知：检测到换人就把 Elo 拉回均值（新阵容的旧 Elo 不可信）----
        # 关键：**在本场 Elo 更新之前**做，因为本场是「新阵容」打的第一场。
        for _t, _players in (per_team_players or {}).items():
            snap = self.snapshot(_players)
            if not snap:
                continue
            prev = self.roster_sig.get(_t)
            self.roster_sig[_t] = snap
            if prev is None:
                continue
            frac = len(snap - prev) / 5.0
            if frac <= 0:
                continue
            self.churn[_t].append((date, frac))
            if self.rho > 0:
                r = self.rho * frac
                self.elo[_t] = self.elo0 + (self.elo[_t] - self.elo0) * (1.0 - r)
                self.elo_m[_t] = self.elo0 + (self.elo_m[_t] - self.elo0) * (1.0 - r)
            if self.rho_g > 0:
                self.games[_t] = max(0, int(round(self.games[_t] * (1.0 - self.rho_g * frac))))
        a_win = (winner == teamA)
        sA = 1.0 if a_win else 0.0
        wa = sum(1 for _m, s1, s2 in map_results if s1 > s2)
        wb = sum(1 for _m, s1, s2 in map_results if s2 > s1)
        tot = wa + wb
        sA_margin = (wa / tot) if tot else sA
        for t in (teamA, teamB):
            self.games[t] += 1
        eA = 1.0 / (1.0 + 10 ** (-(self.elo[teamA] - self.elo[teamB]) / 400.0))
        eAm = 1.0 / (1.0 + 10 ** (-(self.elo_m[teamA] - self.elo_m[teamB]) / 400.0))
        kA = self.k_new + (self.k_old - self.k_new) * (self.games[teamA] / (self.games[teamA] + self.k_sat))
        kB = self.k_new + (self.k_old - self.k_new) * (self.games[teamB] / (self.games[teamB] + self.k_sat))
        self.elo[teamA] += kA * (sA - eA)
        self.elo[teamB] += kB * ((1 - sA) - (1 - eA))
        self.elo_m[teamA] += kA * (sA_margin - eAm)
        self.elo_m[teamB] += kB * ((1 - sA_margin) - (1 - eAm))
        # 历史
        self.mw[teamA][0 if a_win else 1] += 1.0
        self.mw[teamB][0 if not a_win else 1] += 1.0
        self.mf[teamA][0 if a_win else 1] += 1.0
        self.mf[teamB][0 if not a_win else 1] += 1.0
        self.last_match[teamA] = date
        self.last_match[teamB] = date
        key = (teamA, teamB) if teamA <= teamB else (teamB, teamA)
        first_is_a = (teamA <= teamB)
        if a_win:
            self.h2h_hist[key][0 if first_is_a else 1] += 1.0
        else:
            self.h2h_hist[key][1 if first_is_a else 0] += 1.0
        for mname, s1, s2 in map_results:
            if s1 == s2:
                continue
            tw, tl = (teamA, teamB) if s1 > s2 else (teamB, teamA)
            self.mapw[(tw, mname)][0] += 1.0
            self.mapw[(tl, mname)][1] += 1.0
        # 选手
        opp_elo = {teamA: self.elo[teamB], teamB: self.elo[teamA]}
        for team, players in (per_team_players or {}).items():
            rated = [(n, float(r)) for n, r in players if n and r is not None]
            if not rated:
                continue
            top = max(rated, key=lambda x: x[1])[0]
            for name, rating in rated:
                self.p_sum[name] += rating
                self.p_w[name] += 1.0
                self.p_adj[name] += rating - self.opp_adj * ((opp_elo.get(team, self.elo0) - self.elo0) / 400.0)
                self.p_play[name] += 1.0
                if name == top:
                    self.p_top[name] += 1.0
                self.p_team[name] = team
                self.p_date[name] = date
                self.team_players[team][name] = date
            # 队伍总体水平（= map_perf 的收缩先验）
            mean = sum(r for _n, r in rated) / len(rated)
            self.t_level_sum[team] += mean - self.opp_adj * ((opp_elo.get(team, self.elo0) - self.elo0) / 400.0)
            self.t_level_w[team] += 1.0
        # 选手级图偏好：把 by_map 的逐图选手 rating 累加到 (team, map)
        for mname, per_map in (map_stats or {}).items():
            for team, players in per_map.items():
                if team not in (teamA, teamB):
                    continue
                rated_m = [float(r) for _n, r in players if r is not None]
                if not rated_m:
                    continue
                mean_m = sum(rated_m) / len(rated_m)
                self.pm_sum[(team, mname)] += mean_m - self.opp_adj * (
                    (opp_elo.get(team, self.elo0) - self.elo0) / 400.0)
                self.pm_w[(team, mname)] += 1.0


# ------------------------------------------------------------------ 样本
def build(buckets, truths, veto, stats, ranks, eras, by_pool, tl, state, mh,
          map_stats=None):
    """按 train→valid→test 顺序推进，返回 {bucket: [sample...]}。

    每场存：
      feats    第 1 项特征
      map_rows / y_maps   第 3 项（交给 veto_model）
      player_rows[team]   第 4 项：该队阵容选手的赛前特征 + label（该选手是否队内第一）
      mvp_truth / truth   真值
    """
    if map_stats is None:
        map_stats = load_stats_map()

    def pool_of(r):
        t = truths.get(r["id"]) or {}
        if t.get("pool"):
            return list(t["pool"])
        return list(PE.era_and_pool(r["date"], [], eras, by_pool, tl)[1])

    out = {}
    for bucket in ("train", "valid", "test"):
        # 与基线口径一致：只保留 bo ∈ {1,3,5}（排除 4 场 `best_of=25` 的 EWC 计分异常场）
        rows = [r for r in buckets[bucket]
                if isinstance(r.get("best_of"), int) and r["best_of"] in (1, 3, 5)]
        order = sorted(range(len(rows)), key=lambda i: (rows[i]["date"], rows[i]["id"]))
        samples = []
        i = 0
        while i < len(order):
            d = rows[order[i]]["date"]
            state.advance(d)
            mh.advance(d)
            group = []
            while i < len(order) and rows[order[i]]["date"] == d:
                group.append(rows[order[i]])
                i += 1
            pending = []
            for r in group:
                tA, tB = r["team1"]["name"], r["team2"]["name"]
                t = truths.get(r["id"]) or {}
                pool = pool_of(r)
                bo = t.get("bo") or r.get("best_of") or 3
                ea, ga = state.elo_of(tA)
                eb, gb = state.elo_of(tB)
                ma, _ = state.elo_m_of(tA)
                mb, _ = state.elo_m_of(tB)
                ca, cb = state.churn_rate(tA, d), state.churn_rate(tB, d)
                form_a, form_b = state.form(tA), state.form(tB)
                rest_a, rest_b = state.rest_days(tA, d), state.rest_days(tB, d)
                ra, pa = ranks.rank_points(tA, d)
                rb, pb = ranks.rank_points(tB, d)
                la, lb = state.lineup_strength(tA, d), state.lineup_strength(tB, d)
                feats = {
                    "d_elo": (ea - eb) / 400.0,
                    "d_elom": (ma - mb) / 400.0,
                    "d_rank": float(rb - ra),
                    "d_logp": math.log1p(pa) - math.log1p(pb),
                    "d_wr": state.winrate(tA) - state.winrate(tB),
                    "d_h2h": state.h2h_rate(tA, tB) - state.h2h_rate(tB, tA),
                    "d_poolwr": state.pool_wr(tA, pool) - state.pool_wr(tB, pool),
                    "d_lineup": (la - lb) if (la is not None and lb is not None) else 0.0,
                    "a_churn": ca, "b_churn": cb,
                    "d_form": form_a - form_b,
                    "d_rest": math.log1p(min(rest_a, 180.0)) - math.log1p(min(rest_b, 180.0)),
                    "d_games": state.exp_log(tA) - state.exp_log(tB),
                    "d_churn": ca - cb, "sum_churn": ca + cb,
                }
                # 选手级「图偏好」（2026-10-11 新增；来自 csapi `by_map` 的逐图选手 rating，
                # 此前 `load_stats` 只读 overall 而把它整块丢弃）：池内逐图表现水平差。
                pa = [state.map_perf(tA, m) for m in pool]
                pb = [state.map_perf(tB, m) for m in pool]
                pdd = [x - y for x, y in zip(pa, pb)]
                feats["d_poolperf"] = sum(pdd) / len(pdd) if pdd else 0.0
                feats["d_bestperf"] = max(pdd) if pdd else 0.0
                feats["d_perfspread"] = ((max(pa) - min(pa)) - (max(pb) - min(pb))) if pa else 0.0
                perf_rows = dict((m, (pa[i], pb[i])) for i, m in enumerate(pool))
                # 必须走 features_for（带日期）⇒ sim3_in_set 才是真·先手模拟；
                # 直接调 features() 会让 sim3 退化成 sim2（2026-10-11 抓到的接线坑）
                mfeats = mh.features_for(tA, tB, pool, bo, date=d)
                tset = set(t.get("maps") or [])
                per = stats.get(r["id"]) or {}
                player_rows = {}
                for team, opp in ((tA, tB), (tB, tA)):
                    lineup = state.lineup(team, d)
                    opp_elo = state.elo_of(opp)[0]
                    rat = dict((p, r) for p, r in (per.get(team) or []) if r is not None)
                    top = max(rat, key=lambda x: rat[x]) if rat else None
                    player_rows[team] = [
                        dict(state.player_row(team, p, d, opp_elo, lineup),
                             label=(1 if p == top else 0))
                        for p in lineup
                    ]
                pending.append({
                    "id": r["id"], "date": d, "teamA": tA, "teamB": tB, "bo": bo,
                    "pool": pool, "truth": t, "winner": r["winner"]["name"],
                    "y": 1 if r["winner"]["name"] == tA else 0,
                    "feats": feats,
                    "elo": {tA: ea, tB: eb}, "elom": {tA: ma, tB: mb},
                    "map_rows": mfeats,
                    "perf": perf_rows,
                    "y_maps": [1 if f["map"] in tset else 0 for f in mfeats],
                    "mvp_truth": mvp_label(r, stats),
                    "player_rows": player_rows,
                })
            samples.extend(pending)
            for r in group:
                tA, tB = r["team1"]["name"], r["team2"]["name"]
                mres = [(m["name"], m["team1_score"], m["team2_score"]) for m in (r.get("maps") or [])]
                t = truths.get(r["id"]) or {}
                state.observe(d, tA, tB, r["winner"]["name"], mres, stats.get(r["id"], {}),
                              (map_stats or {}).get(r["id"]))
                mh.observe_match(tA, tB, pool_of(r), t.get("maps") or [], veto.get(r["id"]), mres)
        out[bucket] = samples
    return out


# ------------------------------------------------------------------ 模型
def _winner_matrix(samples, features):
    import numpy as np
    return np.array([[s["feats"][f] for f in features] for s in samples], dtype=float), \
        np.array([s["y"] for s in samples], dtype=int)


CV_FOLDS = (0.40, 0.55, 0.70, 0.85, 1.00)


def cv_acc(samples, features, C):
    """**只用给定样本**的扩展窗口交叉验证准确率（用于选特征组 / C / 超参）。

    比「单看 242 条 valid」稳健得多，且完全不接触 test。
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    n = len(samples)
    accs, cnts, prev = [], [], 0
    for cut in CV_FOLDS:
        end = int(n * cut)
        if end - prev < 20 or prev < 100:
            prev = end
            continue
        tr = samples[:prev]
        va = samples[prev:end]
        X = np.array([[t["feats"][f] for f in features] for t in tr], dtype=float)
        y = np.array([t["y"] for t in tr], dtype=int)
        sc = StandardScaler().fit(X)
        clf = LogisticRegression(max_iter=4000, C=C).fit(sc.transform(X), y)
        Xv = np.array([[t["feats"][f] for f in features] for t in va], dtype=float)
        yv = np.array([t["y"] for t in va], dtype=int)
        p = clf.predict_proba(sc.transform(Xv))[:, 1]
        accs.append(float(np.mean((p >= 0.5) == (yv == 1))))
        cnts.append(len(va))
        prev = end
    return sum(a * c for a, c in zip(accs, cnts)) / sum(cnts) if cnts else 0.0


def oof_proba(samples, features, C):
    """**扩展窗口样本外概率**：每折用更早的数据训练，预测本折；首段（无训练集）返回 None。

    与 `cv_acc` 同一折点，故「准确率」与「用它做的规则选型」口径一致。
    """
    out = [None] * len(samples)
    n = len(samples)
    prev = 0
    for cut in CV_FOLDS:
        end = int(n * cut)
        if end - prev < 20 or prev < 100:
            prev = end
            continue
        model = _train_winner(samples[:prev], C=C, features=features)
        for i in range(prev, end):
            out[i] = predict_proba_win(model, samples[i])
        prev = end
    return out


def score_item_points(s, pA, score_fn):
    """`014-scorer` 第 2 项得分的**逐字复刻**（bo1 不计；方向须与预测胜方一致）。"""
    t = s["truth"]
    if not t or s["bo"] == 1:
        return 0
    winA = pA >= 0.5
    sc = score_fn(s, winA, pA if winA else 1.0 - pA)
    a, b = sc
    if a == b:
        return 0
    pwin = s["teamA"] if a > b else s["teamB"]
    if pwin != s["winner"]:
        return 0
    return 10 if tuple(sc) == tuple(t["score"]) else 0


def oof_drop_acc(samples, oof, gmodel):
    """诊断：gap_logit 对「真实获胜方丢 >=1 张图」的样本外方向准确率。"""
    hit = tot = 0
    for smp, pA in zip(samples, oof):
        t = smp["truth"]
        if t is None or smp["bo"] == 1 or pA is None:
            continue
        a, b = t["score"]
        lab = 1 if min(a, b) >= 1 else 0
        pred = 1 if gap_p(gmodel, smp) >= 0.5 else 0
        tot += 1
        hit += 1 if pred == lab else 0
    return hit / tot if tot else 0.0


def predict_proba_win(model, sample):
    import numpy as np
    X = np.array([[sample["feats"][f] for f in model["features"]]], dtype=float)
    return float(model["clf"].predict_proba(model["scaler"].transform(X))[0, 1])


def mvp_pick(rows):
    """MVP 规则（**无拟合参数，故无过拟合风险**）：

        分数 = z(对手校正 rating) + z(队内担当率)

    都在「预测胜方」的 5 人阵容内做 z 标准化，等价于问两个问题：
    「谁比队友明显更强」「谁一贯是这支队的第一」。valid 上选出（550 分，test 700）。

    `rows` 为空（冷门队伍拿不到选手数据）时返回 ""。
    """
    if not rows:
        return "", {}
    def zs(key):
        vals = [r[key] for r in rows]
        n = len(vals)
        mu = sum(vals) / n
        sd = (sum((v - mu) ** 2 for v in vals) / n) ** 0.5 or 1.0
        return [(r[key] - mu) / sd for r in rows]
    za, zt = zs("adj"), zs("toprate")
    scored = []
    for i, r in enumerate(rows):
        scored.append((za[i] + zt[i], r["raw"], -r["days"], r["player"]))
    scored.sort(reverse=True)
    return scored[0][3], {r["player"]: (za[i] + zt[i]) for i, r in enumerate(rows)}


def predict_mvp(sample, team):
    pick, _ = mvp_pick(sample["player_rows"].get(team) or [])
    return pick


# ------------------------------------------------------------------ 评估
def to_truth(t, mvp=""):
    return Truth(teamA=t["teamA"], teamB=t["teamB"], winner=t["winner"],
                 score=tuple(t["score"]), maps=list(t.get("maps") or []),
                 maps_played=list(t.get("maps_played") or []),
                 maps_mode=t.get("maps_mode") or "veto", best_of=t.get("bo"), mvp=mvp)


def scoreline_for(bo, win_is_A, p_win, thresh):
    if bo == 1:
        return (1, 0) if win_is_A else (0, 1)
    if bo == 5:
        l = 0 if p_win >= thresh[1] else (1 if p_win >= thresh[0] else 2)
        a, b = (3, l)
    else:
        l = 0 if p_win >= thresh[0] else 1
        a, b = (2, l)
    return (a, b) if win_is_A else (b, a)


# ---------------------------------------------------------------- 第 2 项（比分）
# 规格（`014-scorer/scorer.py`）：比分项的**方向必须与第 1 项一致**，数值逐位相同才得 10 分。
# 因此第 2 项真正的信息量只有「获胜方会不会丢张」。
#
# 2026-10-11 复核发现（上一轮的真实缺陷）：原先用「预测胜率阈值」决定丢不丢图
# （`p_win >= thresh[0]` 才 2:0），而 Elo 逻辑回归（C=0.05）的概率被强烈压向 0.5，
# 绝大多数场次落在阈值以下 ⇒ 系统性押 2:1，可语料里 2:0 才是多数（train 56.1%）。
# 现改为**在 valid 上从一族候选规则里选**（含「获胜方通吃」这个退化但正确的选项），
# test 仍只报一次。
GAP_KEYS = ["d_elo", "d_elom", "d_poolwr"]   # 只用对称的强度差特征（无泄漏）


def _win_maps(bo):
    return 1 if bo == 1 else (3 if bo == 5 else 2)


def _gap_vector(s):
    return [abs(s["feats"][k]) for k in GAP_KEYS] + [(1.0 if s["bo"] == 5 else 0.0)]


def _gap_labels(samples):
    """训练标签：**真实获胜方**是否丢掉至少 1 张图（bo1 不计此项）。

    特征取「强弱差距的绝对值」⇒ 与「从预测胜方视角看」同分布，故训练/预测两侧口径一致。
    """
    import numpy as np
    X, y = [], []
    for s in samples:
        t = s["truth"]
        if not t or s["bo"] == 1:
            continue
        a, b = t["score"]
        X.append(_gap_vector(s))
        y.append(1 if min(a, b) >= 1 else 0)
    return np.array(X, dtype=float), np.array(y, dtype=int)


def fit_gap_model(samples, C=0.5):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    X, y = _gap_labels(samples)
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=4000, C=C).fit(sc.transform(X), y)
    return {"scaler": sc, "clf": clf, "n": int(len(y)), "pos": int(y.sum()),
            "coef": [float(v) for v in clf.coef_[0]], "intercept": float(clf.intercept_[0])}


def gap_p(model, s):
    """P(获胜方丢 >=1 张图)。"""
    import numpy as np
    x = np.array([_gap_vector(s)], dtype=float)
    return float(model["clf"].predict_proba(model["scaler"].transform(x))[0, 1])


def rule_modal(modal):
    def f(s, winA, pwin_prob):
        w, l = (modal or {}).get(s["bo"], (2, 1))
        return (w, l) if winA else (l, w)
    return f


def rule_always_win0(s, winA, pwin_prob):
    """获胜方通吃（bo3 2:0 / bo5 3:0）——退化但合法的候选，必须参与 valid 选型。"""
    w = _win_maps(s["bo"])
    return (w, 0) if winA else (0, w)


def rule_thresh(thresh):
    def f(s, winA, pwin_prob):
        return scoreline_for(s["bo"], winA, pwin_prob, thresh)
    return f


def rule_gap_thresh(T, key="d_elo"):
    def f(s, winA, pwin_prob):
        w = _win_maps(s["bo"])
        l = 0 if abs(s["feats"][key]) >= T else min(1, w - 1)
        return (w, l) if winA else (l, w)
    return f


def rule_gap_logit(model, T=0.5):
    def f(s, winA, pwin_prob):
        w = _win_maps(s["bo"])
        l = min(1 if gap_p(model, s) >= T else 0, w - 1)
        return (w, l) if winA else (l, w)
    return f


def evaluate_bucket(samples, wmodel, mmodel, use_mvp=True, oracle_winner=False,
                    thresh=(0.70, 0.85), score_mode="cond", modal=None,
                    want_details=False, score_fn=None):
    agg = collections.Counter()
    details = []
    for s in samples:
        t = s["truth"]
        if not t:
            continue
        p = predict_proba_win(wmodel, s)            # P(teamA 胜)
        winA = (s["winner"] == s["teamA"]) if oracle_winner else (p >= 0.5)
        pwin = s["teamA"] if winA else s["teamB"]
        pwin_prob = p if winA else 1 - p
        k = max(1, min(int(s["bo"]), 5))
        pmaps, _ = VM.predict_set(mmodel, {"bo": s["bo"], "rows": s["map_rows"]}, k)
        if score_fn is not None:
            sc = score_fn(s, winA, pwin_prob)
        elif score_mode == "modal":
            w, l = (modal or {}).get(s["bo"], (2, 1))
            sc = (w, l) if winA else (l, w)
        else:
            sc = scoreline_for(s["bo"], winA, pwin_prob, thresh)
        mvp = predict_mvp(s, pwin) if use_mvp else ""
        truth = to_truth(t, s["mvp_truth"])
        pred = Prediction(winner=pwin, score=sc, maps=pmaps, mvp=mvp)
        b = score_match(pred, truth)
        agg["n"] += 1
        agg["max"] += max_total(truth)
        agg["winner"] += b.winner
        agg["score"] += b.score
        agg["maps"] += b.maps
        agg["mvp"] += b.mvp
        agg["total"] += b.total
        agg["winner_max"] += 10
        sc_max = 10 if s["bo"] in (3, 5) else 0
        agg["score_max"] += sc_max
        agg["maps_max"] += max_total(truth) - 20 - sc_max   # 与计分器满分口径一致
        agg["mvp_max"] += 10
        agg["winner_hit"] += 1 if pwin == s["winner"] else 0
        agg["mvp_hit"] += 1 if b.mvp else 0
        if want_details:
            details.append({"id": s["id"], "pA": p, "pred_winner": pwin, "truth_winner": s["winner"],
                            "pred_score": sc, "truth_score": t["score"], "pred_maps": pmaps,
                            "truth_maps": t.get("maps"), "pred_mvp": mvp, "truth_mvp": s["mvp_truth"],
                            "points": b.total, "mvp_points": b.mvp, "winner_points": b.winner,
                            "maps_points": b.maps, "score_points": b.score})
    return (dict(agg), details) if want_details else dict(agg)


def render(res, title):
    def line(name, got, mx):
        return "  %-7s %6d / %6d = %5.1f%%   丢 %d" % (
            name, got, mx, 100.0 * got / mx if mx else 0.0, mx - got)
    L = ["== %s ==" % title, "  n=%d  满分=%d" % (res["n"], res["max"])]
    L.append(line("winner", res["winner"], res["winner_max"]))
    L.append(line("score", res["score"], res["score_max"]))
    L.append(line("maps", res["maps"], res["maps_max"]))
    L.append(line("mvp", res["mvp"], res["mvp_max"]))
    L.append("  TOTAL   %6d / %6d = %5.1f%%   （胜方命中 %d/%d = %.1f%%，MVP 命中 %d）" % (
        res["total"], res["max"], 100.0 * res["total"] / res["max"] if res["max"] else 0,
        res["winner_hit"], res["n"], 100.0 * res["winner_hit"] / res["n"] if res["n"] else 0,
        res["mvp_hit"]))
    return "\n".join(L)


def _train_winner(samples, C=WIN_C, features=None):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    features = list(features or WIN_FEATURES)
    X, y = _winner_matrix(samples, features)
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=4000, C=C).fit(sc.transform(X), y)
    return {"scaler": sc, "clf": clf, "features": features}


def prepare(verbose=False):
    """**唯一的建模入口**：选型 + 最终模型拟合，返回全部产物。

    `model_v2.run()`、`predict.py`、`decide.py` 都走这里 ⇒ 工具不会有「第二套逻辑」。
    选型只用 train（扩展窗口 CV / OOF）与 valid；test 只被评估，从不参与选择。
    """
    rows = SPLIT.load(MATCHES)
    exclude = SPLIT.load_exclude(EXCLUDE) if os.path.exists(EXCLUDE) else None
    buckets, meta = SPLIT.split(rows, purge=True, exclude=exclude)
    truths = {t["id"]: t for t in load_jsonl(TRUTH)}
    veto = VM.load_veto()
    stats = load_stats()
    ranks = PE.Ranks()
    eras, by_pool = PE.load_pool_eras()
    tl = json.load(io.open(PE.TIMELINE, encoding="utf-8"))

    state = State()
    mh = VM.MapHistory()
    data = build(buckets, truths, veto, stats, ranks, eras, by_pool, tl, state, mh)

    # 第 1 项：**特征组 + C 只用 train 上的扩展窗口 CV 选**（valid 只复核，test 只报一次）
    # 242 条 valid 上的 1~2 个样本差异是噪声；CV 用 ~1,600 条验证更稳。
    cand = []
    for fname, feats in WIN_FEATURE_SETS:
        for C in (0.02, 0.05, 0.1, 0.3, 1.0):
            cand.append({"cv": cv_acc(data["train"], feats, C), "name": fname,
                         "feats": list(feats), "C": C})
    cand.sort(key=lambda r: (-r["cv"], len(r["feats"]), r["C"]))
    n_val = int(len(data["train"]) * (CV_FOLDS[-1] - CV_FOLDS[0]))
    p0 = cand[0]["cv"]
    se = math.sqrt(max(p0 * (1.0 - p0), 1e-9) / max(1, n_val))
    pick = cand[0]
    if verbose:
        print("")
        print("  [第 1 项选型] train 上扩展窗口 CV（1SE=%.4f）：" % se)
        for r in cand[:6]:
            print("    %-22s C=%-5s cv=%.4f%s" % (r["name"], r["C"], r["cv"],
                                                  "  <-选定" if r is pick else ""))
        print("    -> %s（%d 特征，cv=%.4f，valid 复核见下）"
              % (pick["name"], len(pick["feats"]), pick["cv"]))
    C = pick["C"]
    winfeats = pick["feats"]
    # 选型阶段一律用 **train-only** 模型（valid 保持样本外，供比分规则选型用）
    wsel = _train_winner(data["train"], C=C, features=winfeats)
    win_name = pick["name"]
    if verbose:
        tr = sum(1 for s in data["train"] if (predict_proba_win(wsel, s) >= 0.5) == (s["y"] == 1))
        vl = sum(1 for s in data["valid"] if (predict_proba_win(wsel, s) >= 0.5) == (s["y"] == 1))
        print("    -> 选定 %s C=%s ；train acc=%.2f%% valid acc=%.2f%%"
              % (win_name, C, 100.0 * tr / max(1, len(data["train"])),
                 100.0 * vl / max(1, len(data["valid"]))))
    # 第 3 项（选图）：特征组 {base, v2, v3} × C 在 **valid** 上选（与 veto_model 同口径）
    # v3 = v2 + sim3_in_set（先 ban 方 = 排名较低者；主理人 2026-10-11 确认的规则）
    def _mpair(bucket):
        return [{"bo": s["bo"], "rows": s["map_rows"], "y": s["y_maps"]} for s in data[bucket]]

    map_sel = None
    _mcands = [("base", VM.FEATURES), ("v2", VM.FEATURES2)]
    if hasattr(VM, "FEATURES3"):
        _mcands.append(("v3", VM.FEATURES3))
    for mname, mfeats in _mcands:
        for mC in (0.03, 0.1, 0.3, 1.0):
            m = VM.train(_mpair("train"), features=mfeats, C=mC)
            ev = VM.evaluate(m, _mpair("valid"))
            if map_sel is None or ev["points"] > map_sel[2]["points"]:
                map_sel = (mname, mfeats, ev, mC)
            if verbose:
                print("    选图 %-4s C=%-5s valid=%d 分（%.1f%%）"
                      % (mname, mC, ev["points"], 100 * ev["hit_rate"]))
    map_name, map_feats, map_ev, map_C = map_sel
    if verbose:
        print("    -> 选图：%s 特征（%d 个），C=%s，valid %d 分（%.1f%%）"
              % (map_name, len(map_feats), map_C, map_ev["points"], 100 * map_ev["hit_rate"]))
    mh_sel = VM.train(_mpair("train"), features=map_feats, C=map_C)

    # 第 2 项：比分阈值在 valid 上选
    tbest = None
    for t0 in (0.55, 0.60, 0.65, 0.70, 0.75, 0.80):
        for t1 in (0.80, 0.90):
            if t1 < t0:
                continue
            res = evaluate_bucket(data["valid"], wsel, mh_sel, True, False, (t0, t1), "cond", None)
            if tbest is None or res["score"] > tbest[1]:
                tbest = ((t0, t1), res["score"])
    thresh = tbest[0]

    modal = {}
    mt = [s["truth"] for s in data["train"] if s["truth"]]
    for bo in (3, 5):
        c = collections.Counter()
        for t in mt:
            if t.get("bo") != bo:
                continue
            sa, sb = t["score"]
            if sa == sb:
                continue
            c[(max(sa, sb), min(sa, sb))] += 1
        modal[bo] = c.most_common(1)[0][0] if c else (2, 1)

    # 第 2 项：**用 train 的样本外（OOF）得分**在候选规则族里选。
    # 上一轮用 242 条 valid 选，一次命中 = 10 分、候选间只差 1~2 次命中 ⇒ 选择噪声极大
    # （这正是上一轮比分规则不稳的根因）。改成 train 的扩展窗口 OOF（~1,600 条、方向也样本外），
    # 规则族也刻意收窄，降低「在噪声里挑最大」的偏差。
    gmodel = fit_gap_model(data["train"])
    gmodelC1 = fit_gap_model(data["train"], C=1.0)
    cands = collections.OrderedDict()
    cands["modal"] = rule_modal(modal)
    cands["always_win0"] = rule_always_win0
    cands["thresh%s" % (tuple(thresh),)] = rule_thresh(thresh)
    for T in (0.05, 0.10, 0.15, 0.20):
        cands["gap_T%.2f" % T] = rule_gap_thresh(T)
    for T in (0.35, 0.40, 0.45, 0.50, 0.55, 0.60):
        cands["gap_logit_T%.2f" % T] = rule_gap_logit(gmodel, T)
    cands["gap_logitC1_T0.45"] = rule_gap_logit(gmodelC1, 0.45)
    oof = oof_proba(data["train"], winfeats, C)
    n_oof = sum(1 for v in oof if v is not None)
    if verbose:
        print("")
        print("  [第 2 项选型] train 上 **样本外** 比分项得分（OOF n=%d；1 次命中 = 10 分）：" % n_oof)
        print("    [诊断] gap_logit 对「获胜方丢图」的样本外方向准确率 = %.1f%%"
              % (100.0 * oof_drop_acc(data["train"], oof, gmodel)))
    got = collections.OrderedDict()
    for name, fn in cands.items():
        pts = 0
        for smp, pA in zip(data["train"], oof):
            if pA is None:
                continue
            pts += score_item_points(smp, pA, fn)
        got[name] = pts
        if verbose:
            print("    %-22s OOF score = %5d" % (name, pts))
    # 1 次命中 = 10 分；OOF 的 1SE ≈ 168 分 ⇒ 「最高分 ±1SE」内一律视为并列，
    # 此时取**最简单**的规则（奥卡姆偏好）。否则会在纯噪声里挑出复杂模型。
    top = max(got.values())
    tol = 10.0 * math.sqrt(0.25 / max(1, n_oof)) * n_oof
    order = ["always_win0", "modal", "thresh", "gap_T", "gap_logitC1", "gap_logit"]
    within = [n for n, v in got.items() if v >= top - tol]

    def rank_of(n):
        for i, k in enumerate(order):
            if n.startswith(k):
                return i
        return len(order)

    within.sort(key=lambda n: (rank_of(n), -got[n], n))
    score_name = within[0]
    score_fn = cands[score_name]
    _sv = got[score_name]
    gap_T = 0.45
    if score_name.startswith("gap_logit"):
        gap_T = float(score_name.rsplit("_T", 1)[1])
    elif score_name.startswith("gap_T"):
        gap_T = float(score_name.rsplit("_T", 1)[1])
    if verbose:
        print("    -> 1SE 容差 = %.0f 分；并列者 %s" % (tol, within))
        print("    -> 选定 %s（OOF %d 分 / %d 场，最高 %d）"
              % (score_name, _sv, n_oof, top))

    # ---- 最终模型：超参已定，用 **train + valid** 重拟合（不碰 test）----
    fit_data = data["train"] + data["valid"]
    wmodel = _train_winner(fit_data, C=C, features=winfeats)
    mmodel = VM.train([{"rows": s["map_rows"], "y": s["y_maps"]} for s in fit_data],
                      features=map_feats, C=map_C)
    gmodel = fit_gap_model(fit_data)
    score_fn = rule_gap_logit(gmodel, gap_T)
    if verbose:
        print("    [最终模型] 用 train+valid（%d 场）重拟合；比分规则 %s T=%s"
              % (len(fit_data), score_name, gap_T))

    return {
        "rows": rows, "meta": meta, "data": data, "state": state, "mh": mh,
        "ranks": ranks, "truths": truths, "veto": veto, "stats": stats,
        "wmodel": wmodel, "mmodel": mmodel, "gmodel": gmodel,
        "wsel": wsel, "mh_sel": mh_sel,
        "winfeats": winfeats, "C": C, "win_name": win_name,
        "cv_cand": cand, "cv_se": se, "thresh": thresh, "modal": modal,
        "score_name": score_name, "gap_T": gap_T, "n_oof": n_oof, "oof": oof,
        "map_name": map_name, "map_feats": map_feats, "map_C": map_C,
        "map_valid_points": map_ev["points"],
    }


def run(verbose=True, prep=None):
    prep = prep or prepare(verbose=verbose)
    data, meta, state, mh = prep["data"], prep["meta"], prep["state"], prep["mh"]
    wmodel, mmodel, gmodel = prep["wmodel"], prep["mmodel"], prep["gmodel"]
    winfeats, C, win_name = prep["winfeats"], prep["C"], prep["win_name"]
    thresh, modal = prep["thresh"], prep["modal"]
    score_name, gap_T, gmodel = prep["score_name"], prep["gap_T"], prep["gmodel"]
    score_fn = (rule_gap_logit(prep["gmodel"], gap_T) if score_name.startswith("gap_logit")
                else rule_always_win0 if score_name == "always_win0"
                else rule_modal(modal) if score_name == "modal"
                else rule_thresh(thresh))
    fit_data = data["train"] + data["valid"]
    res = {}
    details = {}
    variants = {
        "full": {},
        "score_modal": {"score_mode": "modal"},
        "score_always_win0": {"score_fn": rule_always_win0},
        "score_thresh": {"score_fn": rule_thresh(thresh)},
        "score_gap_logit": {"score_fn": rule_gap_logit(gmodel, gap_T)},
        "no_mvp": {"use_mvp": False},
        "oracle": {"oracle_winner": True},
    }
    for k in ("train", "valid", "test"):
        res[k] = {}
        for vn, kw in variants.items():
            kw2 = dict(kw)
            fn = kw2.pop("score_fn", None)
            res[k][vn] = evaluate_bucket(data[k], wmodel, mmodel, kw2.get("use_mvp", True),
                                         kw2.get("oracle_winner", False), thresh,
                                         kw2.get("score_mode", "cond"), modal,
                                         score_fn=fn)
        res[k]["picked"] = evaluate_bucket(data[k], wmodel, mmodel, True, False, thresh,
                                           "cond", modal, score_fn=score_fn)
        _r, details[k] = evaluate_bucket(data[k], wmodel, mmodel, True, False,
                                         thresh, "cond", modal, want_details=True,
                                         score_fn=score_fn)
    out = {"split": meta["counts"], "winner_C": C, "winner_features": winfeats,
           "fit_on": "train+valid",
           "score_rule_final": score_name,
           "maps_features": prep["map_name"], "maps_C": prep["map_C"],
           "maps_valid_points": prep["map_valid_points"],
           "winner_cv_top": [{k: r[k] for k in ("name", "C", "cv")} for r in prep["cv_cand"][:8]],
           "winner_cv_se": prep["cv_se"],
           "thresh": list(thresh),
           "score_rule": score_name,
           "gap_model": {"n": gmodel["n"], "pos": gmodel["pos"],
                         "keys": GAP_KEYS, "coef": gmodel["coef"],
                         "intercept": gmodel["intercept"]},
           "modal": {str(k): list(v) for k, v in modal.items()}, "results": res}
    if verbose:
        for k in ("train", "valid", "test"):
            print("")
            print(render(res[k]["picked"], k))
            print("  [变体] 比分=众数 total=%d ／ 比分=通吃 total=%d ／ 比分=旧阈值 total=%d ／ 比分=gap_logit total=%d" % (
                res[k]["score_modal"]["total"], res[k]["score_always_win0"]["total"],
                res[k]["score_thresh"]["total"], res[k]["score_gap_logit"]["total"]))
            print("  [变体] 去掉 MVP total=%d" % res[k]["no_mvp"]["total"])
            print("  [边际] 胜方 oracle ⇒ total=%d（+%d）" % (
                res[k]["oracle"]["total"], res[k]["oracle"]["total"] - res[k]["picked"]["total"]))
    with io.open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    with io.open(os.path.join(DATA, "model_v2_test_details.json"), "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(details["test"], ensure_ascii=False, indent=1) + "\n")
    return out


# ------------------------------------------------------------------ selftest
class _SelfScaler:
    """恒等 scaler（自证里用来构造固定的 gap_logit）。"""

    def transform(self, X):
        return X


class _ConstClf:
    """常数概率分类器（自证用）。"""

    def __init__(self, p):
        self.p = float(p)

    def predict_proba(self, X):
        import numpy as np
        n = len(X)
        return np.array([[1.0 - self.p, self.p]] * n)


def _selftest():
    fails = []

    n_chk = [0]

    def chk(name, got, want):
        n_chk[0] += 1
        if got != want:
            fails.append("%s: got=%r want=%r" % (name, got, want))

    st = State()
    chk("fresh elo", st.elo_of("X")[0], ELO0)
    chk("fresh winrate", st.winrate("X"), 0.5)
    chk("empty lineup", st.lineup("X", "2026-01-01"), [])
    chk("lineup strength None", st.lineup_strength("X", "2026-01-01"), None)

    st.advance("2026-01-01")
    for _ in range(10):
        st.observe("2026-01-01", "A", "B", "A", [("m", 13, 5)], {"A": [("a1", 1.2)], "B": [("b1", 0.8)]})
    chk("A elo up", st.elo_of("A")[0] > ELO0, True)
    chk("B elo down", st.elo_of("B")[0] < ELO0, True)
    chk("A games", st.elo_of("A")[1], 10)
    chk("h2h favors A", st.h2h_rate("A", "B") > 0.5, True)
    chk("a1 top rate", st.player_toprate("a1"), 1.0)
    chk("lineup of A", st.lineup("A", "2026-01-02"), ["a1"])
    chk("lineup strength", round(st.lineup_strength("A", "2026-01-02"), 3), round(st.player_adj("a1"), 3))
    chk("stale lineup empty", st.lineup("A", "2028-01-01"), [])
    chk("elo_m differs", st.elo_m_of("A")[0] > ELO0, True)

    # 对手校正：对手更强 ⇒ 同样 rating 的 adj 更低
    st2 = State()
    st2.advance("2026-01-01")
    st2.elo["SB"] = 1900.0
    st2.observe("2026-01-01", "SA", "SB", "SA", [("m", 13, 5)], {"SA": [("p", 1.2)], "SB": [("q", 1.2)]})
    strong_adj = st2.player_adj("p")
    st3 = State()
    st3.advance("2026-01-01")
    st3.elo["WB"] = 1100.0
    st3.observe("2026-01-01", "SA", "WB", "SA", [("m", 13, 5)], {"SA": [("p", 1.2)], "WB": [("q", 1.2)]})
    weak_adj = st3.player_adj("p")
    chk("opponent adjust larger vs weak", weak_adj > strong_adj, True)

    # 注入：冷门队伍
    st4 = State()
    st4.seed_team("Cold", elo=1420.0, games=0, players=[("x", 1.1), ("y", 1.0)],
                  winrate=0.4, map_rates={"Mirage": 0.6, "Nuke": 0.3})
    chk("cold elo", st4.elo_of("Cold")[0], 1420.0)
    chk("cold winrate", st4.winrate("Cold"), 0.4)
    chk("cold lineup", st4.lineup("Cold", "2026-01-01"), ["x", "y"])
    chk("cold pool_wr", round(st4.pool_wr("Cold", ["Mirage", "Nuke"]), 3), round((0.6 + 0.3) / 2, 3))
    chk("cold player seeded", round(st4.player_mean("x"), 3), 1.1)
    st4.alias("Cold2", "Cold")
    chk("alias elo", st4.elo_of("Cold2")[0], 1420.0)

    # 选手行
    row = st4.player_row("Cold", "x", "2026-01-02", 1500.0, ["x", "y"])
    chk("player_row keys", set(row.keys()), set(MVP_FEATURES) | {"player"})
    chk("rank of x", row["rank"], 1.0)

    # MVP 规则：adj/toprate 都占优者胜出
    rows = [{"player": "s", "adj": 1.30, "raw": 1.30, "toprate": 0.6, "d_team": 0.2,
             "rank": 1.0, "exp": 20.0, "d_opp": 0.0, "days": 1.0},
            {"player": "t", "adj": 1.10, "raw": 1.10, "toprate": 0.2, "d_team": 0.0,
             "rank": 2.0, "exp": 20.0, "d_opp": 0.0, "days": 1.0}]
    chk("mvp_pick star", mvp_pick(rows)[0], "s")
    chk("mvp_pick empty", mvp_pick([]), ("", {}))

    # 比分
    chk("bo3 strong 2:0", scoreline_for(3, True, 0.9, (0.70, 0.85)), (2, 0))
    chk("bo3 close 2:1", scoreline_for(3, True, 0.55, (0.70, 0.85)), (2, 1))
    chk("bo3 B win", scoreline_for(3, False, 0.9, (0.70, 0.85)), (0, 2))
    chk("bo1", scoreline_for(1, True, 0.5, (0.70, 0.85)), (1, 0))
    chk("bo5 strong 3:0", scoreline_for(5, True, 0.9, (0.70, 0.85)), (3, 0))

    # mvp_label
    mt = {"id": 9, "winner": {"name": "W"}}
    mstat = {9: {"W": [("a", 1.1), ("b", 1.4), ("c", 0.9)], "L": [("z", 2.0)]}}
    chk("mvp_label", mvp_label(mt, mstat), "b")
    chk("mvp_label missing", mvp_label(mt, {}), "")

    # ---- 2026-10-11 新增：阵容变动感知 ----
    st5 = State(rho=0.5, rho_g=0.0)
    st5.advance("2026-01-01")
    five = [(p, 1.0) for p in ("p1", "p2", "p3", "p4", "p5")]
    for _ in range(8):
        st5.observe("2026-01-01", "A", "B", "A", [("m", 13, 5)], {"A": five, "B": five})
    elo_before = st5.elo["A"]
    churn_before = st5.churn_rate("A", "2026-01-01")
    chk("churn zero before change", churn_before, 0.0)
    # 换 2 人 ⇒ 首次被检出，且 Elo 向均值回归
    newfive = [(p, 1.0) for p in ("p1", "p2", "p3", "p6", "p7")]
    st5.observe("2026-01-02", "A", "B", "A", [("m", 13, 5)], {"A": newfive, "B": five})
    chk("churn detected", st5.churn_rate("A", "2026-01-02") > 0.0, True)
    chk("churn frac 2 of 5", round(st5.churn["A"][0][1], 3), 0.4)
    chk("elo regressed toward mean after change",
        abs(st5.elo["A"] - ELO0) < abs(elo_before - ELO0), True)
    chk("churn decays with time", st5.churn_rate("A", "2031-01-02"), 0.0)
    # rho=0 时不回归（消融对照）
    st6 = State(rho=0.0)
    st6.advance("2026-01-01")
    for _ in range(8):
        st6.observe("2026-01-01", "A", "B", "A", [("m", 13, 5)], {"A": five, "B": five})
    e0 = st6.elo["A"]
    st6.observe("2026-01-02", "A", "B", "A", [("m", 13, 5)], {"A": newfive, "B": five})
    chk("rho=0 keeps churn feature but no regression", st6.churn["A"][0][1], 0.4)
    chk("rho=0 elo keeps rising (no pull)", st6.elo["A"] > e0, True)

    # ---- 2026-10-11 新增：冷门注入的等效场次（exp_games）----
    st7 = State()
    st7.seed_team("Cold", elo=1500.0, games=8, exp_games=180, players=[("x", 1.0)])
    chk("injected exp_log uses exp_games", round(st7.exp_log("Cold"), 4),
        round(math.log1p(180), 4))
    st8 = State()
    st8.seed_team("Cold", elo=1500.0, games=8, players=[("x", 1.0)])
    chk("injected exp_log falls back to games", round(st8.exp_log("Cold"), 4),
        round(math.log1p(8), 4))

    # ---- 2026-10-11 新增：比分规则（第 2 项）----
    s3 = {"bo": 3, "feats": {"d_elo": 0.5, "d_elom": 0.5, "d_poolwr": 0.2}}
    s5 = {"bo": 5, "feats": {"d_elo": 0.1, "d_elom": 0.1, "d_poolwr": 0.0}}
    chk("always_win0 bo3 A", rule_always_win0(s3, True, 0.9), (2, 0))
    chk("always_win0 bo3 B", rule_always_win0(s3, False, 0.9), (0, 2))
    chk("always_win0 bo5 A", rule_always_win0(s5, True, 0.6), (3, 0))
    chk("gap_thresh big gap -> no drop", rule_gap_thresh(0.20)(s3, True, 0.9), (2, 0))
    chk("gap_thresh small gap -> drop", rule_gap_thresh(0.90)(s3, True, 0.9), (2, 1))
    chk("gap_thresh bo5 caps at 1 drop", rule_gap_thresh(0.90)(s5, True, 0.9), (3, 1))
    chk("rule_modal bo3", rule_modal({3: (2, 1)})(s3, True, 0.9), (2, 1))

    gm = {"scaler": _SelfScaler(), "clf": _ConstClf(0.9)}
    chk("gap_logit T=0.95 -> no drop", rule_gap_logit(gm, 0.95)(s3, True, 0.9), (2, 0))
    chk("gap_logit T=0.05 -> drop", rule_gap_logit(gm, 0.05)(s3, True, 0.9), (2, 1))
    chk("gap_p in [0,1]", 0.0 <= gap_p(gm, s3) <= 1.0, True)
    # score_item_points 与计分器口径一致
    smp = {"bo": 3, "teamA": "A", "teamB": "B", "winner": "A",
           "truth": {"score": [2, 0]}, "feats": s3["feats"]}
    chk("score_item_points hit", score_item_points(smp, 0.7, rule_always_win0), 10)
    smp2 = dict(smp, truth={"score": [2, 1]})
    chk("score_item_points miss", score_item_points(smp2, 0.7, rule_always_win0), 0)
    smp3 = dict(smp, winner="B")
    chk("score_item_points wrong direction", score_item_points(smp3, 0.7, rule_always_win0), 0)
    chk("score_item_points bo1 not scored",
        score_item_points(dict(smp, bo=1), 0.7, rule_always_win0), 0)

    # ---- 2026-10-11 新增：选手级图偏好（by_map 逐图选手 rating）----
    # (a) 收缩数学：与 Elo 机制解耦，直接设定「总体水平」与 (team,map) 观测
    st9 = State()
    chk("map_perf unseen team -> 1.0", st9.map_perf("X", "Mirage"), 1.0)
    chk("pool_perf unseen team -> 1.0", st9.pool_perf("X", ["Mirage", "Nuke"]), 1.0)
    st9.t_level_sum["A"], st9.t_level_w["A"] = 1.25, 1.0
    chk("map_perf 0 观测 == 该队总体水平", round(st9.map_perf("A", "Ancient"), 6), 1.25)
    st9.pm_sum[("A", "Mirage")], st9.pm_w[("A", "Mirage")] = 1.5, 1.0
    chk("收缩 k0=3 向先验", round(st9.map_perf("A", "Mirage"), 6), round((1.5 + 3 * 1.25) / 4.0, 6))
    chk("pool_perf = 池内均值", round(st9.pool_perf("A", ["Mirage", "Ancient"]), 6),
        round(((1.5 + 3 * 1.25) / 4.0 + 1.25) / 2.0, 6))
    st9.advance("2026-01-01")
    st9.advance("2026-01-31")                      # 30 天 ⇒ 权重按选手半衰期衰减
    f30 = 0.5 ** (30.0 / HALF_LIFE_PLAYER)
    chk("map_perf 按选手半衰期衰减", round(st9.map_perf("A", "Mirage"), 6),
        round((1.5 * f30 + 3 * 1.25) / (f30 + 3), 6))
    chk("总体水平同样衰减（先验同步）", round(st9.team_level("A"), 6), round((1.25 * f30) / f30, 6))

    # (b) observe() 端到端：写入 (team,map) 与队伍总体水平，且强弱图排序正确
    st10 = State()
    st10.advance("2026-01-01")
    st10.observe("2026-01-01", "A", "B", "A", [("Mirage", 13, 5)],
                 {"A": [("a1", 1.30), ("a2", 1.20)], "B": [("b1", 0.90), ("b2", 0.80)]},
                 {"Mirage": {"A": [("a1", 1.60), ("a2", 1.40)], "B": [("b1", 0.70), ("b2", 0.60)]},
                  "Nuke": {"A": [("a1", 0.90), ("a2", 0.80)], "B": [("b1", 0.95), ("b2", 0.85)]}})
    chk("observe 写入 (team,map) 计数", st10.pm_w[("A", "Mirage")], 1.0)
    chk("observe 写入队伍总体水平计数", st10.t_level_w["A"], 1.0)
    chk("未观测图 == 总体水平（差分中性）",
        round(st10.map_perf("A", "Ancient"), 6), round(st10.team_level("A"), 6))
    chk("强图 > 总体水平", st10.map_perf("A", "Mirage") > st10.team_level("A"), True)
    chk("弱图 < 总体水平", st10.map_perf("A", "Nuke") < st10.team_level("A"), True)
    chk("两队图偏好差为正（A 强于 B on Mirage）",
        st10.map_perf("A", "Mirage") - st10.map_perf("B", "Mirage") > 0, True)
    chk("对手校正：胜方(A) 总体水平略高于其原始均值 1.25",
        st10.team_level("A") > 1.25, True)

    # 接线守卫（2026-10-11 抓到的坑）：本模块构造 map 特征必须走 features_for（带日期），
    # 否则 sim3_in_set 会退化成 sim2_in_set（重复列），「跑了 v3」其实是空转。
    _src = io.open(os.path.abspath(__file__), encoding="utf-8").read()
    _needle = "." + "features("       # 拼接以免守卫命中它自己的检查名
    chk("model_v2 不裸调 features()（须走 features_for）",
        _needle not in _src, True)

    if fails:
        print("MODEL_V2_SELFTEST_FAIL")
        for f in fails:
            print("  -", f)
        return 1
    print("MODEL_V2_SELFTEST_OK (%d checks)" % n_chk[0])
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "selftest":
        raise SystemExit(_selftest())
    if cmd == "run":
        run()
        raise SystemExit(0)
    print(__doc__)
    raise SystemExit(2)
