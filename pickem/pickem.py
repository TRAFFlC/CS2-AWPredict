#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Pick'Em 十格选择优化器（U3）—— issue-015。

输入：瑞士轮模拟出来的**联合场景**（每个场景 = 16 队最终战绩）。
输出：2×3:0 / 6×晋级(3:1 或 3:2) / 2×0:3 的十格选择，以及 **P(命中 ≥k)**。

为什么不能用「各格取最优」：正确事件由同一批瑞士轮结果驱动、**强相关**，
目标是**阈值概率** —— 这是组合选择（投资组合）问题，不是十个独立分类问题。
本模块同时给出「独立假设闭式解」作对照，量化忽略相关性的乐观偏差。

搜索策略：先把候选限制在各类别**边际概率靠前**的队伍（P(3:0) 近 0 的队永远不值得选），
再在候选内做随机重启 + 首次改进爬山。评估次数有预算上限。
"""
from __future__ import annotations

import random

import numpy as np

N30, NADV, N03 = 2, 6, 2

CAT30, CATADV, CAT03, CATOTHER = 0, 1, 2, 3


def categorize(wl):
    """(wins, losses) → 类别编码。"""
    w, l = wl
    if w == 3 and l == 0:
        return CAT30
    if w == 3:
        return CATADV
    if w == 0 and l == 3:
        return CAT03
    return CATOTHER


def build_cat_matrix(scenes, teams):
    """(T, N) int8：每队每个场景的类别。teams 顺序即行顺序。"""
    idx = {t: i for i, t in enumerate(teams)}
    T, N = len(teams), len(scenes)
    C = np.full((T, N), CATOTHER, dtype=np.int8)
    for s, sc in enumerate(scenes):
        for t in teams:
            C[idx[t], s] = categorize(sc[t])
    return C


def _masks(C):
    return (C == CAT30), (C == CATADV), (C == CAT03)


def p_at_least(p30, padv, p03, masks, k, idx):
    """给定三组**队名**选择，计算 P(命中 ≥k)。idx = {队名: 行号}。"""
    m30, madv, m03 = masks
    total = (m30[[idx[t] for t in p30]].sum(axis=0)
             + madv[[idx[t] for t in padv]].sum(axis=0)
             + m03[[idx[t] for t in p03]].sum(axis=0))
    return float((total >= k).mean())


def marginals(C, teams):
    """逐队 P(3:0) / P(晋级) / P(0:3)。"""
    m30, madv, m03 = _masks(C)
    return {t: {"p30": float(m30[i].mean()), "padv": float(madv[i].mean()),
                "p03": float(m03[i].mean())} for i, t in enumerate(teams)}


def _candidates(marg, teams, top30, topadv, top03):
    return {
        0: sorted(teams, key=lambda t: -marg[t]["p30"])[:top30],
        1: sorted(teams, key=lambda t: -marg[t]["padv"])[:topadv],
        2: sorted(teams, key=lambda t: -marg[t]["p03"])[:top03],
    }


# ---------------------------------------------------------------- 优化
def optimize(C, teams, k=5, seed=20261012, restarts=8, max_evals=20000,
             top30=8, topadv=12, top03=6):
    """随机重启 + 首次改进爬山。返回 (best_dict, P, evals)。"""
    masks = _masks(C)
    idx = {t: i for i, t in enumerate(teams)}
    marg = marginals(C, teams)
    cand = _candidates(marg, teams, top30, topadv, top03)
    rng = random.Random(seed)
    evals = 0
    best, best_p = None, -1.0
    m30, madv, m03 = masks

    def score(groups):
        nonlocal evals
        evals += 1
        total = (m30[[idx[t] for t in groups[0]]].sum(0)
                 + madv[[idx[t] for t in groups[1]]].sum(0)
                 + m03[[idx[t] for t in groups[2]]].sum(0))
        return float((total >= k).mean())

    sizes = (N30, NADV, N03)
    for _ in range(restarts):
        if evals >= max_evals:
            break
        used = set()
        groups = [[], [], []]
        ok_init = True
        for gi in range(3):
            avail = [t for t in cand[gi] if t not in used]
            if len(avail) < sizes[gi]:
                ok_init = False
                break
            rng.shuffle(avail)
            groups[gi] = avail[:sizes[gi]]
            used.update(groups[gi])
        if not ok_init:
            continue
        cur = score(groups)
        improved = True
        while improved and evals < max_evals:
            improved = False
            for gi in range(3):
                for pi in range(len(groups[gi])):
                    used_now = set(groups[0]) | set(groups[1]) | set(groups[2])
                    for t in cand[gi]:                      # (a) 换成未选中的候选
                        if t in used_now:
                            continue
                        trial = [list(x) for x in groups]
                        trial[gi][pi] = t
                        p = score(trial)
                        if p > cur + 1e-12:
                            groups, cur, improved = trial, p, True
                    for gj in range(3):                     # (b) 与其它组对调
                        if gj == gi:
                            continue
                        for pj in range(len(groups[gj])):
                            trial = [list(x) for x in groups]
                            trial[gi][pi], trial[gj][pj] = trial[gj][pj], trial[gi][pi]
                            p = score(trial)
                            if p > cur + 1e-12:
                                groups, cur, improved = trial, p, True
                    if evals >= max_evals:
                        break
                if evals >= max_evals:
                    break
        if cur > best_p:
            best_p, best = cur, {"p30": list(groups[0]), "padv": list(groups[1]),
                                 "p03": list(groups[2])}
    if best is None:
        best = heuristic_picks(marg)
        best_p = score([best["p30"], best["padv"], best["p03"]])
    return best, best_p, evals


def heuristic_picks(marg):
    """朴素「热门」基线：3:0 取 p30 前 2；晋级取剩余里 padv 前 6；0:3 取剩余里 p03 前 2。"""
    teams = list(marg.keys())
    a = sorted(teams, key=lambda t: -marg[t]["p30"])[:N30]
    rest = [t for t in teams if t not in a]
    b = sorted(rest, key=lambda t: -marg[t]["padv"])[:NADV]
    rest2 = [t for t in rest if t not in b]
    c = sorted(rest2, key=lambda t: -marg[t]["p03"])[:N03]
    return {"p30": a, "padv": b, "p03": c}


def independent_p_at_least(probs, k):
    """泊松二项 DP：假设十格独立时 P(总命中 ≥k)。"""
    dp = [1.0]
    for p in probs:
        nd = [0.0] * (len(dp) + 1)
        for i, v in enumerate(dp):
            nd[i] += v * (1 - p)
            nd[i + 1] += v * p
        dp = nd
    return float(sum(dp[k:]))


def pick_probs(picks, marg):
    probs = [marg[t]["p30"] for t in picks["p30"]]
    probs += [marg[t]["padv"] for t in picks["padv"]]
    probs += [marg[t]["p03"] for t in picks["p03"]]
    return probs


def brute_force(C, teams, k=5, sizes=None):
    """小规模精确枚举（自证用）。sizes 可覆盖 (2,6,2)。"""
    import itertools
    s30, sadv, s03 = sizes or (N30, NADV, N03)
    masks = _masks(C)
    idx = {t: i for i, t in enumerate(teams)}
    best_p, best = -1.0, None
    for a in itertools.combinations(teams, s30):
        rest = [t for t in teams if t not in a]
        for b in itertools.combinations(rest, sadv):
            rest2 = [t for t in rest if t not in b]
            for c in itertools.combinations(rest2, s03):
                p = p_at_least(a, b, c, masks, k, idx)
                if p > best_p:
                    best_p, best = p, (a, b, c)
    return best, best_p


def synth_scenes(n_teams, n_scenes, seed=0, n30=1, nadv=2, n03=1, teams=None):
    """合成场景（自证用）：每场景恰好 n30 队 3-0、nadv 队晋级、n03 队 0-3。"""
    rng = random.Random(seed)
    teams = teams or ["S%02d" % i for i in range(n_teams)]
    scenes = []
    for _ in range(n_scenes):
        perm = list(range(n_teams))
        rng.shuffle(perm)
        sc = {}
        for j, i in enumerate(perm):
            if j < n30:
                sc[teams[i]] = (3, 0)
            elif j < n30 + nadv:
                sc[teams[i]] = (3, 1)
            elif j < n30 + nadv + n03:
                sc[teams[i]] = (0, 3)
            else:
                sc[teams[i]] = (1, 3)
        scenes.append(sc)
    return scenes


# ---------------------------------------------------------------- 自证
def _selftest():
    ok = 0

    def chk(cond, msg):
        nonlocal ok
        if not cond:
            raise AssertionError("FAIL: " + msg)
        ok += 1

    teams = ["T%02d" % i for i in range(1, 17)]
    idx = {t: i for i, t in enumerate(teams)}
    seeds = {t: i + 1 for i, t in enumerate(teams)}

    # 1) 类别判定
    chk(categorize((3, 0)) == CAT30, "3-0 应为 CAT30")
    chk(categorize((3, 1)) == CATADV and categorize((3, 2)) == CATADV, "3-1/3-2 应为 CATADV")
    chk(categorize((0, 3)) == CAT03, "0-3 应为 CAT03")
    chk(categorize((1, 3)) == CATOTHER, "1-3 应为 CATOTHER")

    import swiss as SW

    def pw(a, b, bo):
        return 0.7 if seeds[a] < seeds[b] else 0.3

    scenes, _ = SW.simulate_scenes(teams, seeds, pw, 3000, seed=5)
    C = build_cat_matrix(scenes, teams)
    chk(C.shape == (16, 3000), "类别矩阵形状应为 (16,3000)")
    marg = marginals(C, teams)
    chk(marg["T01"]["p30"] > marg["T16"]["p30"], "强队 3:0 概率应更高")

    # 2) P(≥1) 应等于并集频率
    masks = _masks(C)
    p_any = p_at_least(["T01"], [], ["T16"], masks, 1, idx)
    manual = float(((C[0] == CAT30) | (C[15] == CAT03)).mean())
    chk(abs(p_any - manual) < 1e-12, "P(≥1) 应等于并集频率")

    # 3) 独立假设 DP
    chk(abs(independent_p_at_least([0.5, 0.5], 1) - 0.75) < 1e-12, "2×0.5 ⇒ P(≥1)=0.75")
    chk(abs(independent_p_at_least([0.5, 0.5], 2) - 0.25) < 1e-12, "2×0.5 ⇒ P(≥2)=0.25")
    chk(abs(independent_p_at_least([1.0, 1.0], 2) - 1.0) < 1e-12, "确定性 ⇒ P(≥2)=1")

    # 4) 优化器 vs 精确枚举（合成缩小规模：8 队 / 1-2-1）
    small = ["S%02d" % i for i in range(8)]
    ssc = synth_scenes(8, 400, seed=8, n30=1, nadv=2, n03=1, teams=small)
    C2 = build_cat_matrix(ssc, small)
    sizes = (1, 2, 1)
    global N30, NADV, N03
    old = (N30, NADV, N03)
    N30, NADV, N03 = sizes
    try:
        _bf, bf_p = brute_force(C2, small, k=2, sizes=sizes)
        _got, got_p, _ev = optimize(C2, small, k=2, seed=3, restarts=6, max_evals=8000,
                                    top30=8, topadv=8, top03=8)
        chk(abs(got_p - bf_p) < 1e-12, "优化器应达枚举最优 %.6f，实得 %.6f" % (bf_p, got_p))
    finally:
        N30, NADV, N03 = old

    # 5) 十格数量与互斥
    best, best_p, probes = optimize(C, teams, k=5, seed=1, restarts=4, max_evals=6000)
    chk(probes > 0, "应有评估发生")
    chk(len(best["p30"]) == 2 and len(best["padv"]) == 6 and len(best["p03"]) == 2, "十格数量正确")
    chk(len(set(best["p30"] + best["padv"] + best["p03"])) == 10, "十格应互不重复")

    # 6) 优化器不劣于热门基线
    h = heuristic_picks(marg)
    p_heur = p_at_least(h["p30"], h["padv"], h["p03"], masks, 5, idx)
    chk(best_p >= p_heur - 1e-12, "优化器不应劣于热门基线（%.4f vs %.4f）" % (best_p, p_heur))

    # 7) 确定性
    _a, pa, _ = optimize(C, teams, k=5, seed=42, restarts=3, max_evals=3000)
    _b, pb, _ = optimize(C, teams, k=5, seed=42, restarts=3, max_evals=3000)
    chk(pa == pb, "同 seed 优化结果应一致")

    print("PICKEM_SELFTEST_OK (%d checks)" % ok)
    return 0


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        raise SystemExit(_selftest())
    print(__doc__)
