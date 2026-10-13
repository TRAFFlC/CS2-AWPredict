#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
issue-014 计分器（按用户规格实现）+ 自证夹具。

规格来源：issues/014-cs2-major-match-prediction.md §3（2026-10-07 用户口头规格）。

四项计分：
  1. 获胜方            10 分
  2. 获胜比分          10 分（比分的获胜方必须与第 1 项一致，否则该项 0；
                       **bo1 不记此项**——2026-10-08 用户裁决 Q-007：
                       bo1 只记胜方 10 + 选图 5 + MVP 10 = 25 分）
  3. BP 选图           每命中一张 5 分（默认集合命中；可切顺序命中）
  4. 比赛 MVP          10 分（必须出自第 1 项预测的获胜方队伍）

单场满分：bo3 = 45，bo5 = 55，**bo1 = 25**（bo1 不记比分项，见下）。

第 3 项真值口径（2026-10-08 用户裁决 Q-010，**读法 B**）：**真值图集 = veto 后全部图，
含实际没打的决胜图**（bo3 恒 3 张、bo5 恒 5 张）。因此 `Truth.maps` 存的是「应有图集」，
`Truth.maps_played` 只存「实际打过比分的图」供诊断。缺 veto 数据时可退化为
`maps_mode="played"`（读法 A：只算实际打过的图），此时满分随实际图数下降并打降级提示。
纯标准库，秒级，无第三方依赖。

运行自证：python -X utf8 scorer.py --selftest
"""
from __future__ import annotations

from dataclasses import dataclass, field

WINNER_POINTS = 10
SCORE_POINTS = 10
MAP_POINTS = 5
MVP_POINTS = 10


@dataclass
class Prediction:
    winner: str                 # 预测获胜方的队名
    score: tuple[int, int]      # 比分，按 (teamA 回合, teamB 回合) 排列
    maps: list[str] = field(default_factory=list)   # 预测会打到的图（顺序：可选）
    mvp: str = ""               # 预测 MVP 选手名


@dataclass
class Truth:
    teamA: str
    teamB: str
    winner: str                 # 实际获胜方队名（必须等于 teamA 或 teamB）
    score: tuple[int, int]      # 实际比分，按 (teamA 回合, teamB 回合)
    maps: list[str] = field(default_factory=list)   # 计分真值图集：读法 B = veto 后全部图（含未打的决胜图）
    mvp: str = ""               # 实际 MVP 选手名
    best_of: int | None = None  # 可选；None 时由图数推断（1 张图 ⇒ bo1）
    maps_played: list[str] = field(default_factory=list)  # 实际打过比分的图（诊断用；读法 A 降级时即真值）
    maps_mode: str = "veto"     # "veto"（读法 B，默认）| "played"（读法 A 降级）


@dataclass
class Breakdown:
    winner: int = 0
    score: int = 0
    maps: int = 0
    mvp: int = 0
    maps_hit: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.winner + self.score + self.maps + self.mvp


def scoring_maps(truth: "Truth") -> list[str]:
    """本次计分使用的真值图集。

    读法 B（默认，2026-10-08 用户裁决 Q-010）：veto 后全部图，**含未打的决胜图** = `truth.maps`。
    读法 A（`maps_mode="played"` 降级）：只算实际打过的图。
    """
    if truth.maps_mode == "played":
        return list(truth.maps_played) if truth.maps_played else list(truth.maps)
    return list(truth.maps)


def best_of_of(truth: "Truth") -> int | None:
    """bo 判定的唯一入口：显式 best_of 优先，否则由图数推断。

    读法 B 下 `maps` 是应有图集（bo1→1 / bo3→3 / bo5→5），推断唯一；
    读法 A 下 `maps` 是实际打过的图（bo3 可能只有 2 张），也不会误判成 bo1。
    """
    if truth.best_of is not None:
        return truth.best_of
    return len(scoring_maps(truth)) or None


BO_FULL_MAPS = {1: 1, 3: 3, 5: 5}


def max_total(truth: "Truth") -> int:
    """该场可达满分（与用户规格逐字一致）：bo1=25，bo3=45，bo5=55。

    读法 B 下满分由 **bo** 决定，与实际打了几张无关（这正是裁决 B 的直接效果：
    bo3 打成 2:0 时满分仍是 45）。
    读法 A（降级）下满分随实际图数下降，因为真值本身就只有那些图。
    """
    bo = best_of_of(truth)
    score_points = 0 if bo == 1 else SCORE_POINTS
    if truth.maps_mode == "veto":
        n_maps = BO_FULL_MAPS.get(bo, len(scoring_maps(truth)))
    else:
        n_maps = len(scoring_maps(truth))
    return WINNER_POINTS + score_points + MAP_POINTS * n_maps + MVP_POINTS


def _scoreline_winner(score: tuple[int, int], teamA: str, teamB: str) -> str | None:
    """由比分推断获胜方；平局返回 None。"""
    a, b = score
    if a > b:
        return teamA
    if b > a:
        return teamB
    return None


def score_match(pred: Prediction, truth: Truth, maps_order_matters: bool = False) -> Breakdown:
    """按规格计分。maps_order_matters=False 时第 3 项按集合命中。"""
    b = Breakdown()

    # --- 第 1 项：获胜方 ---
    if pred.winner == truth.winner:
        b.winner = WINNER_POINTS

    # --- 第 2 项：比分（方向必须与第 1 项一致；bo1 整项不计）---
    bo = best_of_of(truth)
    pred_winner_by_score = _scoreline_winner(pred.score, truth.teamA, truth.teamB)
    if bo == 1:
        # 2026-10-08 裁决 Q-007：bo1 不记比分，满分 25。
        b.notes.append("bo1 不记比分项（满分 25）")
        # 一致性仍检查，但只作提示，不影响得分。
        if pred_winner_by_score is not None and pred_winner_by_score != pred.winner:
            b.notes.append(
                f"（提示）比分方向与胜方选择不一致（比分为 {pred_winner_by_score} 胜，"
                f"但预测胜方为 {pred.winner}）；bo1 下不影响得分"
            )
    elif pred_winner_by_score is None:
        b.notes.append("比分出现平局，按 0 计")
    elif pred_winner_by_score != pred.winner:
        b.notes.append(
            f"比分方向与胜方选择不一致（比分为 {pred_winner_by_score} 胜，但预测胜方为 {pred.winner}）→ 比分项 0"
        )
    elif tuple(pred.score) == tuple(truth.score):
        b.score = SCORE_POINTS
    else:
        b.notes.append("比分方向一致但数值不符 → 0")

    # --- 第 3 项：BP 选图（真值图集口径见 scoring_maps / 模块说明）---
    tmaps = scoring_maps(truth)
    if truth.maps_mode == "played":
        b.notes.append(
            "选图按读法 A（只算实际打过的图）计分 —— 缺 veto 数据时的降级模式，"
            "与读法 B 的分数不可直接比较"
        )
    else:
        n_played = len(truth.maps_played) if truth.maps_played else None
        if n_played is not None and n_played < len(tmaps):
            b.notes.append(
                f"真值图集含 {len(tmaps) - n_played} 张未打的决胜图（读法 B）"
            )
        expected = BO_FULL_MAPS.get(bo)
        if expected is not None and len(tmaps) != expected:
            b.notes.append(
                f"（提示）读法 B 下 bo{bo} 应有 {expected} 张图，真值只给了 {len(tmaps)} 张"
            )
    if maps_order_matters:
        hits = [p for p, m in zip(pred.maps, tmaps) if p == m]
    else:
        remaining = list(tmaps)
        hits = []
        for m in pred.maps:
            if m in remaining:
                remaining.remove(m)
                hits.append(m)
    b.maps_hit = hits
    b.maps = MAP_POINTS * len(hits)

    # --- 第 4 项：MVP（必须出自第 1 项预测的获胜方队伍）---
    if pred.mvp == "":
        b.notes.append("未预测 MVP → 0")
    elif pred.mvp != truth.mvp:
        b.notes.append("MVP 不是实际 MVP → 0")
    else:
        # 实际 MVP 一定属于实际获胜方；若第 1 项错，则预测 MVP 必不属于预测胜方
        if pred.winner == truth.winner:
            b.mvp = MVP_POINTS
        else:
            b.notes.append("MVP 命中但胜方错，违反『MVP 须出自预测胜方』→ 0")

    return b


# 说明：MVP 是否“出自预测获胜方队伍”需要 roster 数据（选手→队伍）。
# 计分器在与真实数据对接时，由调用方传入 roster 硬校验；此处用
# “pred.mvp == truth.mvp 且 pred.winner == truth.winner”等价判定
# （因实际 MVP 必属实际获胜方）。见 issues/014 §3 与 §8 风险 2。


# ------------------------------------------------------------------ selftest

def _selftest() -> int:
    fails: list[str] = []

    n_checks = 0

    def check(name: str, got, want):
        nonlocal n_checks
        n_checks += 1
        if got != want:
            fails.append(f"{name}: got={got!r} want={want!r}")

    A, B = "TeamA", "TeamB"

    # 1) 全对 bo3 → 45
    t = Truth(A, B, A, (2, 1), ["Mirage", "Inferno", "Nuke"], "s1mpleX")
    p = Prediction(A, (2, 1), ["Mirage", "Inferno", "Nuke"], "s1mpleX")
    r = score_match(p, t)
    check("perfect_bo3.total", r.total, 45)
    check("perfect_bo3.breakdown", (r.winner, r.score, r.maps, r.mvp), (10, 10, 15, 10))

    # 2) 全错 bo3 → 0
    p = Prediction(B, (0, 2), ["Ancient", "Anubis", "Dust2"], "otherPlayer")
    r = score_match(p, t)
    check("all_wrong.total", r.total, 0)

    # 3) 胜方对、比分错 → 10 + 0 + 15 + 10 = 35
    p = Prediction(A, (2, 0), ["Mirage", "Inferno", "Nuke"], "s1mpleX")
    r = score_match(p, t)
    check("winner_right_score_wrong.total", r.total, 35)

    # 4) 比分方向与胜方不一致 → 比分项 0
    p = Prediction(A, (1, 2), ["Mirage", "Inferno", "Nuke"], "s1mpleX")
    r = score_match(p, t)
    check("direction_mismatch.score", r.score, 0)
    check("direction_mismatch.total", r.total, 35)  # 10 + 0 + 15 + 10
    check("direction_mismatch.has_note", any("不一致" in n for n in r.notes), True)

    # 5) MVP 选错人（实际 MVP 属 A，预测胜方 A 但 MVP 填了别的名字）→ mvp 0
    p = Prediction(A, (2, 1), ["Mirage", "Inferno", "Nuke"], "Bplayer")
    r = score_match(p, t)
    check("mvp_wrong_person.mvp", r.mvp, 0)
    check("mvp_wrong_person.total", r.total, 35)  # 10+10+15+0

    # 6) bo5 满分 55
    t5 = Truth(A, B, B, (2, 3), ["M1", "M2", "M3", "M4", "M5"], "Bstar")
    p5 = Prediction(B, (2, 3), ["M1", "M2", "M3", "M4", "M5"], "Bstar")
    r = score_match(p5, t5)
    check("perfect_bo5.total", r.total, 55)

    # 7) 部分命中选图：命中 2/3 → 10 分
    p = Prediction(A, (2, 1), ["Mirage", "Nuke", "Ancient"], "s1mpleX")
    r = score_match(p, t)
    check("partial_maps.maps", r.maps, 10)
    check("partial_maps.maps_hit", sorted(r.maps_hit), sorted(["Mirage", "Nuke"]))
    check("partial_maps.total", r.total, 40)  # 10+10+10+10

    # 8) bo1 满分 25（不记比分项；2026-10-08 裁决 Q-007）
    t1 = Truth(A, B, A, (1, 0), ["Dust2"], "Ace")
    p1 = Prediction(A, (1, 0), ["Dust2"], "Ace")
    r = score_match(p1, t1)
    check("perfect_bo1.total", r.total, 25)
    check("perfect_bo1.score", r.score, 0)
    check("perfect_bo1.maps", r.maps, 5)
    check("perfect_bo1.notes", any("不记比分" in n for n in r.notes), True)

    # 8b) bo1 的比分写错、甚至方向写反，都不得扣分（该整项不计）
    r = score_match(Prediction(A, (0, 1), ["Dust2"], "Ace"), t1)
    check("bo1_wrong_scoreline.total", r.total, 25)
    check("bo1_wrong_scoreline.winner", r.winner, 10)

    # 8c) 显式 best_of 优先于图数推断
    t1b = Truth(A, B, A, (1, 0), ["Dust2"], "Ace", best_of=1)
    check("explicit_best_of.total", score_match(p1, t1b).total, 25)

    # 8d) 可达满分随 bo 变化：25 / 45 / 55
    check("max_total.bo1", max_total(Truth(A, B, A, (1, 0), ["Dust2"], "x")), 25)
    check("max_total.bo3", max_total(t), 45)
    check("max_total.bo5",
          max_total(Truth(A, B, A, (3, 1), ["a", "b", "c", "d", "e"], "x")), 55)
    check("max_total.bo1_explicit_bo5_maps",
          max_total(Truth(A, B, A, (3, 1), ["a", "b", "c", "d", "e"], "x",
                          best_of=None)), 55)

    # 9) 集合命中不因预测图序打乱而失真：顺序打乱仍全中
    p = Prediction(A, (2, 1), ["Nuke", "Mirage", "Inferno"], "s1mpleX")
    r = score_match(p, t)
    check("set_maps_order_agnostic.maps", r.maps, 15)

    # 10) 顺序命中模式：按位置比较，只有第 0 位相同 → 5 分
    r = score_match(Prediction(A, (2, 1), ["Mirage", "Nuke", "Inferno"], "s1mpleX"), t, maps_order_matters=True)
    check("ordered_maps.maps", r.maps, 5)

    # 11) 顺序命中模式负样本：整体轮转（位置全不同）→ 0 分
    r = score_match(Prediction(A, (2, 1), ["Nuke", "Mirage", "Inferno"], "s1mpleX"), t, maps_order_matters=True)
    check("ordered_maps_rotation.maps", r.maps, 0)

    # 12) 读法 B（Q-010，2026-10-08 用户裁决）：真值 = veto 后全部图，含未打的决胜图
    #     bo3 打成 2:0 → 满分仍是 45
    tB = Truth(A, B, A, (2, 0), ["Mirage", "Inferno", "Nuke"], "s1mpleX",
               best_of=3, maps_played=["Mirage", "Inferno"])
    pB = Prediction(A, (2, 0), ["Mirage", "Inferno", "Nuke"], "s1mpleX")
    r = score_match(pB, tB)
    check("readingB_full.total", r.total, 45)
    check("readingB_full.maps", r.maps, 15)
    check("readingB_full.max_total", max_total(tB), 45)
    check("readingB_full.note_unplayed",
          any("未打的决胜图" in n for n in r.notes), True)

    # 12b) 读法 B 的核心增益：**命中那张根本没打的决胜图也给分**
    r = score_match(Prediction(A, (2, 0), ["Mirage", "Nuke"], "s1mpleX"), tB)
    check("readingB_decider_hit.maps", r.maps, 10)
    check("readingB_decider_hit.total", r.total, 40)

    # 12c) 读法 A 降级：同一场 2:0，满分只有 40，且必须自带降级提示
    tA = Truth(A, B, A, (2, 0), ["Mirage", "Inferno"], "s1mpleX",
               best_of=3, maps_played=["Mirage", "Inferno"], maps_mode="played")
    check("readingA_degraded.max_total", max_total(tA), 40)
    r = score_match(Prediction(A, (2, 0), ["Mirage", "Inferno", "Nuke"], "s1mpleX"), tA)
    check("readingA_degraded.maps", r.maps, 10)
    check("readingA_degraded.total", r.total, 40)
    check("readingA_degraded.note", any("降级" in n for n in r.notes), True)

    # 12d) 读法 B 下满分只由 bo 决定，与实际图数无关
    check("readingB_max.bo5_played3",
          max_total(Truth(A, B, A, (3, 0), ["a", "b", "c", "d", "e"], "x",
                          best_of=5, maps_played=["a", "b", "c"])), 55)
    check("readingB_max.bo1",
          max_total(Truth(A, B, A, (1, 0), ["a"], "x",
                          best_of=1, maps_played=["a"])), 25)

    # 12e) 读法 B 下真值给少了（漏了决胜图）必须留提示
    tShort = Truth(A, B, A, (2, 0), ["Mirage", "Inferno"], "s1mpleX", best_of=3)
    check("readingB_short.note",
          any("应有 3 张" in n for n in score_match(pB, tShort).notes), True)

    if fails:
        print("SELFTEST_FAIL")
        for f in fails:
            print("  -", f)
        return 1
    print(f"SCORER_SELFTEST_OK ({n_checks} assertions; bo1=25; map-truth=B)")
    return 0


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    print(__doc__)
