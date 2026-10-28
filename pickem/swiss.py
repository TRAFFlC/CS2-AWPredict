#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""瑞士轮模拟器（Major Stage 1/2/3 通用）—— issue-015 的 U2。

规则（Valve Major Supplemental Rulebook）：
  * 16 队，三胜晋级、三败淘汰，最多 5 轮。
  * 配对：R1 固定种子 1v9 / 2v10 / … / 8v16；其后**同战绩配对**
    （同 Buchholz 的平局键默认按种子升序；`tie_break="random"` 改均匀抽签，见 `pair_group`）、
    **最高种子对最低可用种子**、**同阶段不重赛（if possible）**。
  * bo：**所有直接决定晋级或淘汰的比赛 = bo3，其余 bo1**。
    ⇒ 判定式：本场存在某队已 2 胜（本场赢者将到 3 胜）或已 2 负（本场输者将到 3 负）
    时，该场为 bo3。每阶段应恰好 **20 场 bo1 + 13 场 bo3**。

本模块只负责「给定胜率函数 → 阶段最终战绩的联合分布」，不含任何强度模型。
强度来源见 `sources.py`；选择优化见 `pickem.py`。

接口：
    simulate_stage(teams, seeds, pwin, rng) -> ({team: (wins, losses)}, {"bo1":n,"bo3":m})
    simulate_scenes(teams, seeds, pwin, n, seed) -> [ {team: (w,l)}, ... ]

`pwin(a, b, best_of)` 返回 a 在该场战胜 b 的概率（0..1）。
"""
from __future__ import annotations

import random
from collections import defaultdict

STAGE_ROUNDS = 5
TARGET_WINS = 3
TARGET_LOSSES = 3


# ---------------------------------------------------------------- 配对
def pair_round1(seeds):
    """R1 固定种子配对：1v9, 2v10, …, 8v16（种子为 1..16 的整数）。"""
    order = sorted(seeds, key=lambda t: seeds[t])
    if len(order) != 16:
        raise ValueError("瑞士轮阶段必须是 16 支队伍，当前 %d" % len(order))
    return [(order[i], order[i + 8]) for i in range(8)]


def _played_has(played, a, b):
    """a 与 b 本阶段是否已交手（played 允许 dict-of-set 或 defaultdict）。"""
    return b in played.get(a, ())


def buchholz(team, played, wins):
    """巴克霍兹（Buchholz）分：本阶段**已交手对手的当前胜场之和**。

    组内排序的第一关键字。为什么不是种子：见 `pair_group` 的取证说明。
    """
    return sum(wins.get(o, 0) for o in played.get(team, ()))


def _match(order, played, allow_rematch):
    """在组内求完美匹配：每次给**最靠前**的未配对者挑**最靠后**的可行对手。

    候选顺序按 `order` 中的**位置**（不是种子）——`order` 已经是 Buchholz 序，
    「最靠后」= Buchholz 最低，正是「最好对最差」。
    allow_rematch=False 时**只允许未交手配对**，找不到完美匹配返回 None。
    """
    index = {t: i for i, t in enumerate(order)}
    used = set()
    pairs = []

    def rec(i):
        while i < len(order) and order[i] in used:
            i += 1
        if i >= len(order):
            return True
        a = order[i]
        used.add(a)
        if allow_rematch:
            cands = [t for t in order[i + 1:] if t not in used]
        else:
            cands = [t for t in order[i + 1:] if t not in used and not _played_has(played, a, t)]
        for b in sorted(cands, key=lambda t: -index[t]):   # 组内最靠后（最差）优先
            used.add(b)
            pairs.append((a, b))
            if rec(i + 1):
                return True
            pairs.pop()
            used.discard(b)
        used.discard(a)
        return False

    return pairs if rec(0) else None


def pair_group(group, seeds, played, wins, tie_break="seed", rng=None):
    """同战绩组内配对（Valve Major 瑞士轮规则）。

    组内按 **Buchholz 分降序**分簇；簇内平局键由 `tie_break` 选：
    `"seed"`（默认）用**种子升序**，与历史实现**逐字节等价**（已公布数字不作废）；
    `"random"` 用 `rng` **均匀抽签**（issue §5.4 取证：Stage 2 上「R2 完美」与「R5 完美」的
    种子序**结构性互斥、交集 0** ⇒ 真实平局打破倾向抽签，确定性种子序只是近似）。
    随后再「最好对最差」；
    最后**先求「无重赛」完美匹配，只有不存在时才允许重赛**（规则书 "if possible"）。

    为什么是 Buchholz 而不是种子（2026-10-13 主理人提醒后取证，见 `pairing_probe.py`）：
    `StarLadder Budapest Major 2025 Stage 1` 的 33 场真实配对里，
    Buchholz 口径能**逐轮全部复现（33/33）**；而纯种子口径即使把 8! 种种子排列搜遍，
    上界也只有 **26/33** ⇒ 原来的「按种子排序」是错的，不是种子重建的锅。

    用回溯而非贪心：贪心会漏掉可行配对、产生本可避免的重赛（实测 200 个场景
    里会多出 65 次强制重赛）。
    """
    if tie_break not in ("seed", "random"):
        raise ValueError("tie_break 只能是 'seed' 或 'random'，实得 %r" % (tie_break,))
    if tie_break == "random" and rng is None:
        raise ValueError("tie_break='random' 需要传入 rng（均匀抽签的随机源）")
    clusters = {}
    for t in group:
        clusters.setdefault(buchholz(t, played, wins), []).append(t)
    order = []
    for bh in sorted(clusters, reverse=True):
        cluster = clusters[bh]
        if tie_break == "random" and len(cluster) > 1:
            cluster = list(cluster)
            rng.shuffle(cluster)                      # 同 Buchholz 簇内均匀抽签
        else:
            cluster = sorted(cluster, key=lambda t: seeds[t])
        order.extend(cluster)
    res = _match(order, played, allow_rematch=False)
    if res is None:
        res = _match(order, played, allow_rematch=True)
    if res is None:
        raise RuntimeError("组内无法配对：%r" % (order,))
    return res


def is_decisive(wins, losses, a, b):
    """该场是否直接决定晋级或淘汰 ⇒ bo3。"""
    return (wins[a] == 2 or wins[b] == 2 or losses[a] == 2 or losses[b] == 2)


# ---------------------------------------------------------------- 单阶段
def simulate_stage(teams, seeds, pwin, rng, played=None, tie_break="seed"):
    """模拟一个 16 队瑞士轮阶段，返回 ({team: (wins, losses)}, 赛程统计)。"""
    wins = {t: 0 for t in teams}
    losses = {t: 0 for t in teams}
    if played is None:
        played = defaultdict(set)
    n_bo1 = n_bo3 = 0
    for rnd in range(1, STAGE_ROUNDS + 1):
        active = [t for t in teams if wins[t] < TARGET_WINS and losses[t] < TARGET_LOSSES]
        groups = defaultdict(list)
        for t in active:
            groups[(wins[t], losses[t])].append(t)
        for _rec, grp in sorted(groups.items()):
            pairs = (pair_round1(seeds) if rnd == 1 else
                     pair_group(grp, seeds, played, wins, tie_break=tie_break, rng=rng))
            for a, b in pairs:
                bo = 3 if is_decisive(wins, losses, a, b) else 1
                if bo == 3:
                    n_bo3 += 1
                else:
                    n_bo1 += 1
                p = pwin(a, b, bo)
                w, l = (a, b) if rng.random() < p else (b, a)
                wins[w] += 1
                losses[l] += 1
                played[a].add(b)
                played[b].add(a)
    return {t: (wins[t], losses[t]) for t in teams}, {"bo1": n_bo1, "bo3": n_bo3}


def simulate_scenes(teams, seeds, pwin, n, seed=20261012, tie_break="seed"):
    """模拟 n 次，返回 (场景列表, 最后一次赛程统计)。同 seed 可复现。"""
    rng = random.Random(seed)
    scenes = []
    stages = None
    for _ in range(n):
        sc, stages = simulate_stage(teams, seeds, pwin, rng, tie_break=tie_break)
        scenes.append(sc)
    return scenes, stages


def record_str(wl):
    return "%d-%d" % (wl[0], wl[1])


def summarize(scenes, teams):
    """逐队最终战绩分布（概率）。"""
    from collections import Counter
    out = {}
    for t in teams:
        c = Counter(record_str(s[t]) for s in scenes)
        out[t] = {k: v / len(scenes) for k, v in sorted(c.items(), key=lambda kv: -kv[1])}
    return out


# ---------------------------------------------------------------- 自证
def _selftest():
    ok = 0

    def chk(cond, msg):
        nonlocal ok
        if not cond:
            raise AssertionError("FAIL: " + msg)
        ok += 1

    teams = ["T%02d" % i for i in range(1, 17)]
    seeds = {t: i + 1 for i, t in enumerate(teams)}

    # 1) R1 配对 = 1v9 … 8v16
    r1 = pair_round1(seeds)
    chk(r1[0] == ("T01", "T09"), "R1 第一对应为 seed1 vs seed9")
    chk(len(r1) == 8 and all(seeds[a] < seeds[b] for a, b in r1), "R1 共 8 对且高种子在前")

    # 2) bo3 判定：恰好 13 场（恒 0.5 胜率）
    sc, stages = simulate_stage(teams, seeds, lambda a, b, bo: 0.5, random.Random(1))
    chk(stages["bo3"] == 13, "每阶段 bo3 应恰好 13 场，实得 %d" % stages["bo3"])
    chk(stages["bo1"] == 20, "每阶段 bo1 应恰好 20 场，实得 %d" % stages["bo1"])
    chk(stages["bo1"] + stages["bo3"] == 33, "总场次应恰好 33")

    # 3) 终局必为 3 胜或 3 负
    for t in teams:
        w, l = sc[t]
        chk(max(w, l) == 3, "%s 终局应为 3 胜或 3 负，实得 %d-%d" % (t, w, l))

    # 4) 晋级/淘汰人数守恒
    adv = sum(1 for t in teams if sc[t][0] == 3)
    elim = sum(1 for t in teams if sc[t][1] == 3)
    chk(adv == 8 and elim == 8, "应 8 晋级 8 淘汰，实得 %d/%d" % (adv, elim))

    # 5) 3-0 恰好 2 队、0-3 恰好 2 队
    chk(sum(1 for t in teams if sc[t] == (3, 0)) == 2, "3-0 应恰好 2 队")
    chk(sum(1 for t in teams if sc[t] == (0, 3)) == 2, "0-3 应恰好 2 队")

    # 6) 同阶段不重赛：300 个随机场景里重赛数应为 0（本结构下重赛永远可避免）
    replay = 0
    for s in range(300):
        rng = random.Random(100 + s)
        played = defaultdict(set)
        w = {t: 0 for t in teams}
        l = {t: 0 for t in teams}
        for rnd in range(1, 6):
            act = [t for t in teams if w[t] < 3 and l[t] < 3]
            grp = defaultdict(list)
            for t in act:
                grp[(w[t], l[t])].append(t)
            for _r, gg in sorted(grp.items()):
                prs = pair_round1(seeds) if rnd == 1 else pair_group(gg, seeds, played, w)
                for a, b in prs:
                    if _played_has(played, a, b):
                        replay += 1
                    played[a].add(b)
                    played[b].add(a)
                    ww, ll = (a, b) if rng.random() < 0.5 else (b, a)
                    w[ww] += 1
                    l[ll] += 1
    chk(replay == 0, "同阶段重赛次数应为 0，实得 %d" % replay)

    # 6b) Buchholz：组内排序键必须是「对手当前胜场和」而不是种子
    #     构造一个两者给出不同配对的最小场景：组内 4 人，种子与 Buchholz 顺序恰好相反
    _seeds = {"A": 1, "B": 2, "C": 3, "D": 4}
    _played = defaultdict(set)
    _wins = {"A": 0, "B": 0, "C": 0, "D": 0}
    # A、B 赢过强对手（各 2 胜），C、D 赢过弱对手（各 0 胜）
    _opp = {"P": 2, "Q": 2, "R": 0, "S": 0}
    _wins.update(_opp)
    _played["A"] = {"P"}; _played["P"] = {"A"}
    _played["B"] = {"Q"}; _played["Q"] = {"B"}
    _played["C"] = {"R"}; _played["R"] = {"C"}
    _played["D"] = {"S"}; _played["S"] = {"D"}
    chk(buchholz("A", _played, _wins) == 2 and buchholz("C", _played, _wins) == 0,
        "Buchholz 应为对手当前胜场之和")
    by_bh = pair_group(["A", "B", "C", "D"], _seeds, _played, _wins)
    by_seed_pairs = [(("A", "D")), (("B", "C"))]
    chk(set(map(frozenset, by_bh)) == set(map(frozenset, by_seed_pairs)),
        "Buchholz 序（A,B,C,D）应给出 A-D / B-C，实得 %r" % (by_bh,))
    # 同分时用种子打破平局：Buchholz 全相等 ⇒ 退化回种子序
    _flat = {t: 0 for t in _seeds}
    eq = pair_group(["A", "B", "C", "D"], _seeds, defaultdict(set), _flat)
    chk(set(map(frozenset, eq)) == {frozenset(("A", "D")), frozenset(("B", "C"))},
        "Buchholz 全平时应按种子排序（A-D / B-C），实得 %r" % (eq,))

    # 7) 确定性
    a1, _ = simulate_scenes(teams, seeds, lambda a, b, bo: 0.5, 50, seed=7)
    a2, _ = simulate_scenes(teams, seeds, lambda a, b, bo: 0.5, 50, seed=7)
    chk(a1 == a2, "同 seed 两次连跑应逐场相同")

    # 8) 强队应更多 3-0
    def pw(a, b, bo):
        return 0.75 if seeds[a] <= 4 else 0.25
    scenes, _ = simulate_scenes(teams, seeds, pw, 2000, seed=11)
    p30 = {t: sum(1 for s in scenes if s[t] == (3, 0)) / len(scenes) for t in teams}
    chk(p30["T01"] > p30["T16"], "强队 3-0 概率应高于弱队")

    # 9) 同 Buchholz 并列的平局键：seed（默认，逐字节不变）vs random（issue §5.4 证据倾向）
    tied = ["A", "B", "C", "D"]
    tseeds = {"A": 1, "B": 2, "C": 3, "D": 4}
    flat = {t: 0 for t in tied}
    none_played = defaultdict(set)
    d_def = pair_group(tied, tseeds, none_played, flat)
    chk(d_def == pair_group(tied, tseeds, none_played, flat, tie_break="seed"),
        "tie_break='seed' 应与默认逐字节相同")
    chk(set(map(frozenset, d_def)) == {frozenset(("A", "D")), frozenset(("B", "C"))},
        "seed 口径仍应种子升序（A-D / B-C）")
    rng7a = pair_group(tied, tseeds, none_played, flat, tie_break="random", rng=random.Random(7))
    rng7b = pair_group(tied, tseeds, none_played, flat, tie_break="random", rng=random.Random(7))
    chk(rng7a == rng7b, "random 口径同 rng 种子应可复现")
    chk(set(map(frozenset, rng7a)) in (
        {frozenset(("A", "B")), frozenset(("C", "D"))},
        {frozenset(("A", "C")), frozenset(("B", "D"))},
        {frozenset(("A", "D")), frozenset(("B", "C"))}),
        "random 口径应给出三种完美匹配之一，实得 %r" % (rng7a,))
    import collections as _c
    cnt = _c.Counter()
    for s in range(600):
        pr = pair_group(tied, tseeds, none_played, flat, tie_break="random",
                        rng=random.Random(s))
        cnt[frozenset(map(frozenset, pr))] += 1
    chk(len(cnt) == 3, "600 次抽签应三种匹配都出现，实得 %d 种" % len(cnt))
    chk(600 * 0.24 <= min(cnt.values()) and max(cnt.values()) <= 600 * 0.43,
        "三种匹配应近似均匀（各 ≈1/3），实得 %r" % (sorted(cnt.values()),))

    # 10) random 口径的规则不变量与可复现性
    sc_r, st_r = simulate_stage(teams, seeds, lambda a, b, bo: 0.5, random.Random(3),
                                tie_break="random")
    chk(st_r["bo1"] == 20 and st_r["bo3"] == 13, "random 口径 bo 计数仍应为 20/13")
    chk(sum(1 for t in teams if sc_r[t][0] == 3) == 8
        and sum(1 for t in teams if sc_r[t][1] == 3) == 8, "random 口径仍应 8 晋级 8 淘汰")
    chk(sum(1 for t in teams if sc_r[t] == (3, 0)) == 2
        and sum(1 for t in teams if sc_r[t] == (0, 3)) == 2, "random 口径仍应恰 2×3-0 / 2×0-3")
    r1, _ = simulate_scenes(teams, seeds, lambda a, b, bo: 0.5, 30, seed=99, tie_break="random")
    r2, _ = simulate_scenes(teams, seeds, lambda a, b, bo: 0.5, 30, seed=99, tie_break="random")
    chk(r1 == r2, "random 口径同 seed 两次连跑应逐场相同")
    s1, _ = simulate_scenes(teams, seeds, lambda a, b, bo: 0.5, 30, seed=99)
    s2, _ = simulate_scenes(teams, seeds, lambda a, b, bo: 0.5, 30, seed=99, tie_break="seed")
    chk(s1 == s2, "tie_break='seed' 应与默认逐字节相同")
    chk(r1 != s1, "random 与 seed 口径应产生不同赛程")
    try:
        pair_group(tied, tseeds, none_played, flat, tie_break="bogus")
        chk(False, "非法 tie_break 应抛 ValueError")
    except ValueError:
        chk(True, "非法 tie_break 应抛 ValueError")
    try:
        pair_group(tied, tseeds, none_played, flat, tie_break="random")
        chk(False, "tie_break='random' 缺 rng 应抛 ValueError")
    except ValueError:
        chk(True, "tie_break='random' 缺 rng 应抛 ValueError")

    replay_r = 0
    for s in range(200):
        rrng = random.Random(500 + s)
        rplayed = defaultdict(set)
        rw = {t: 0 for t in teams}
        rl = {t: 0 for t in teams}
        for rnd in range(1, 6):
            act = [t for t in teams if rw[t] < 3 and rl[t] < 3]
            gg2 = defaultdict(list)
            for t in act:
                gg2[(rw[t], rl[t])].append(t)
            for _r, grp2 in sorted(gg2.items()):
                prs = (pair_round1(seeds) if rnd == 1
                       else pair_group(grp2, seeds, rplayed, rw, tie_break="random", rng=rrng))
                for a, b in prs:
                    if _played_has(rplayed, a, b):
                        replay_r += 1
                    rplayed[a].add(b)
                    rplayed[b].add(a)
                    ww, ll = (a, b) if rrng.random() < 0.5 else (b, a)
                    rw[ww] += 1
                    rl[ll] += 1
    chk(replay_r == 0, "random 口径下同阶段重赛次数应为 0，实得 %d" % replay_r)

    print("SWISS_SELFTEST_OK (%d checks)" % ok)
    return 0


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        raise SystemExit(_selftest())
    print(__doc__)
