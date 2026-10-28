#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""σ 不确定性传播（U1 的强度后验端）—— issue-015 §5.5 的公开落地。

立场（`data/public/findings/008`）：零一线交手的陌生队**不该靠采集去认识**，
而应把「不知道」显式建模成**强度后验的方差**，让它在瑞士轮模拟里传播到
最终战绩分布，而不是假装知道（点估计）。

机制（与私有区 `data/Qwen/Qwen-3.8 Flash/2026-10-13/uncertain_sim.py` 同式）：

  * σE 分桶（先验量级，非拟合）：E=0 → 120、E1-2 → 80、E≥3 → 40（Elo 单位）
  * 逐队合成   σ_i = sqrt( σE(E_i)² + (150/√(1+n_i))² )
  * 低暴露队（E < BLEND_E=5）的均值 μ 回归到 **rank 先验**（队伍内拟合
    rating ~ a + b·ln(1+rank)），修正「弱旅在弱对手身上刷分」的高估；
    E ≥ 5 的队 μ = 自身 rating
  * 每场景抽 θ_i ~ N(μ_i, σ_i) 后跑**完整瑞士轮**（`swiss.simulate_stage`）

roster 输入（见 README「roster 格式」）：每队可选 `exposure` / `games` / `rank`，
或直接给 `sigma` 跳过 σ 推导。**仅 `source=elo`**——σ 模型只在 Elo 单位上验证过。

RNG 布局（与私有区一致，保证可复现）：θ 抽样用 `numpy.random.default_rng(seed)`，
比赛用 `random.Random(seed + 1)`。σ≡0 时 θ=μ 逐位相等 ⇒ 本臂**逐字节退化为点估计臂**
（自证第 7 条，seed 偏移 +1 的关系即由此而来）。

诚实边界（引用时须保留）：σE 是先验量级（6 个 E=0 样本定不了小数点）；
队间独立噪声**低估**系统性平移 ⇒ 结果偏保守；已验证靶只有 Budapest S1 单阶段，
Cologne S1/S2 对照臂**未跑**（见 issue §5.5 待办）。

复现自检：python -X utf8 uncertain.py selftest
"""
from __future__ import annotations

import math
import random

import numpy as np

import sources as S
import swiss as SW

# 低暴露队均值回归的门槛（私有区 uncertain_sim.py 的 BLEND_E=5）
BLEND_E = 5
# σE 分桶（Elo 单位；findings/008 实测先验量级）
SIGMA_E_LOW = 120.0    # E = 0
SIGMA_E_MID = 80.0     # E = 1..2
SIGMA_E_HIGH = 40.0    # E >= 3
# 采样项：150/√(1+n)
SAMPLE_SIGMA = 150.0
# rank 先验拟合所需最少点数（不足则不回归，μ=自身 rating）
MIN_PRIOR_POINTS = 6


def sigma_e(e: int) -> float:
    """暴露度分桶 σE（Elo 单位）。"""
    if e == 0:
        return SIGMA_E_LOW
    if e <= 2:
        return SIGMA_E_MID
    return SIGMA_E_HIGH


def team_sigma(e: int, games: int) -> float:
    """逐队合成 σ = sqrt(σE(E)² + (150/√(1+n))²)。"""
    return math.sqrt(sigma_e(e) ** 2 + (SAMPLE_SIGMA / math.sqrt(1.0 + games)) ** 2)


def fit_rank_prior(ratings, ranks):
    """队伍内最小二乘拟合 rating ~ a + b·ln(1+rank)。

    只用同时有 rating 与 rank 的队；有效点 < MIN_PRIOR_POINTS 或 rank 无散布
    （分母为 0）返回 None。返回 (a, b) 或 None。
    """
    xs, ys = [], []
    for t, r in ratings.items():
        rk = ranks.get(t)
        if rk is None:
            continue
        xs.append(math.log1p(rk))
        ys.append(float(r))
    if len(xs) < MIN_PRIOR_POINTS:
        return None
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0.0:
        return None
    b = num / den
    return (my - b * mx, b)


def build_posterior(roster):
    """roster dict → 不确定性后验（校验 + σ + μ + blend）。

    返回 {"blend_e": BLEND_E, "fit": {a,b}|None, "teams": {name: {...}}}。
    非法输入抛 ValueError（由 CLI 转 rc=1）：
      * source != "elo"
      * 无 `sigma` 的队缺 / 持有非法的 `exposure` 或 `games`
    """
    kind = roster.get("source", "elo")
    if kind != "elo":
        raise ValueError("--uncertain 仅支持 source=elo（σ 后验只在 Elo 单位上验证过），"
                         "实得 %r" % kind)
    teams = roster["teams"]

    def _num(v, lo, what, name):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError("队 %s：%s 必须是数值，实得 %r" % (name, what, v))
        if v < lo:
            raise ValueError("队 %s：%s 必须 ≥%s，实得 %r" % (name, what, lo, v))
        return float(v)

    vals = {}
    for t in teams:
        name = t["name"]
        rating = float(t["rating"])
        rk = t.get("rank")
        if rk is not None:
            rk = int(_num(rk, 1, "rank", name))
        sig = t.get("sigma")
        if sig is not None:
            sig = _num(sig, 0.0, "sigma", name)
            vals[name] = {"mu": rating, "sigma": sig, "own": rating,
                          "exposure": None, "games": None, "rank": rk,
                          "blended": False, "sigma_from": "sigma"}
            continue
        if t.get("exposure") is None:
            raise ValueError("队 %s：--uncertain 需要 exposure（或直接给 sigma）" % name)
        if t.get("games") is None:
            raise ValueError("队 %s：--uncertain 需要 games（或直接给 sigma）" % name)
        e = int(_num(t["exposure"], 0, "exposure", name))
        g = _num(t["games"], 0, "games", name)
        vals[name] = {"mu": rating, "sigma": team_sigma(e, g), "own": rating,
                      "exposure": e, "games": g, "rank": rk,
                      "blended": False, "sigma_from": "derived"}

    ratings = {t["name"]: float(t["rating"]) for t in teams}
    ranks = {t["name"]: int(t["rank"]) for t in teams if t.get("rank") is not None}
    fit = fit_rank_prior(ratings, ranks)
    if fit is not None:
        a, b = fit
        for name, v in vals.items():
            rk = v["rank"]
            if rk is not None and v["exposure"] is not None and v["exposure"] < BLEND_E:
                v["mu"] = a + b * math.log1p(rk)   # 弱旅刷分修正：回归 rank 先验
                v["blended"] = True
    return {"blend_e": BLEND_E,
            "fit": ({"a": fit[0], "b": fit[1]} if fit is not None else None),
            "teams": vals}


def simulate_uncertain(teams, seeds, mus, sigmas, k1, k3, n, seed, tie_break="seed"):
    """n 个场景：抽 θ ~ N(μ, σ) → 完整瑞士轮。返回 (场景列表, 赛程统计)。

    θ 抽样 `default_rng(seed)`；比赛流 `Random(seed + 1)`（与私有区一致）。
    `tie_break` 原样透传给 `swiss.simulate_stage`（`"seed"` 默认 / `"random"` 均匀抽签）。
    sigmas 全 0 且 μ=自身 rating 时，输出与 `swiss.simulate_scenes(seed+1)` 逐字节相同
    （θ=μ+0·z 逐位相等，pwin 走同一式 `sources.sigmoid`）。
    """
    rng_np = np.random.default_rng(seed)
    rng = random.Random(seed + 1)
    elo_arr = np.asarray(list(mus), dtype=float)
    sig_arr = np.asarray(list(sigmas), dtype=float)
    stages = None
    scenes = []
    for _ in range(n):
        theta = elo_arr + sig_arr * rng_np.standard_normal(len(teams))
        th = {t: float(theta[i]) for i, t in enumerate(teams)}

        def pwin(a, b, bo, th=th):
            k = k1 if int(bo) == 1 else k3
            return S.sigmoid(k * (th[a] - th[b]) / 400.0)

        sc, stages = SW.simulate_stage(teams, seeds, pwin, rng, tie_break=tie_break)
        scenes.append(sc)
    return scenes, stages


# ---------------------------------------------------------------- 自证
def _selftest():
    ok = 0

    def chk(cond, msg):
        nonlocal ok
        if not cond:
            raise AssertionError("FAIL: " + msg)
        ok += 1

    # 1) σE 分桶
    chk(sigma_e(0) == 120.0, "E=0 ⇒ σE=120")
    chk(sigma_e(1) == 80.0 and sigma_e(2) == 80.0, "E1-2 ⇒ σE=80")
    chk(sigma_e(3) == 40.0 and sigma_e(99) == 40.0, "E≥3 ⇒ σE=40")

    # 2) σ 合成：精确值 + 两个单调方向
    chk(abs(team_sigma(0, 0) - math.sqrt(120.0 ** 2 + 150.0 ** 2)) < 1e-9,
        "σ(0,0) 应为 sqrt(120²+150²)")
    chk(all(team_sigma(0, g) > team_sigma(0, g + 1) for g in range(20)),
        "σ 应随样本数 n 单调递减")
    chk(team_sigma(0, 5) > team_sigma(3, 5), "同 n 下 E 越大 σ 越小")

    # 3) rank 先验拟合：完美线性数据应还原系数
    teams = ["T%02d" % i for i in range(1, 17)]
    seeds = {t: i + 1 for i, t in enumerate(teams)}
    ranks = {t: 10 * (i + 1) for i, t in enumerate(teams)}          # 10..160
    ratings = {t: 1700.0 - 30.0 * math.log1p(ranks[t]) for t in teams}
    fit = fit_rank_prior(ratings, ranks)
    chk(fit is not None, "16 点应可拟合")
    chk(abs(fit[0] - 1700.0) < 1e-6 and abs(fit[1] + 30.0) < 1e-6,
        "线性数据应还原 a=1700 / b=-30，实得 %r" % (fit,))
    short = {k: ranks[k] for k in teams[:5]}
    chk(fit_rank_prior(ratings, short) is None, "rank 点 <6 应不拟合")

    # 4) build_posterior：校验负样本
    def mk_roster(e_over=None, rating_over=None, drop_exposure=(), rank_points=False):
        e_map = {t: 5 for t in teams}
        if e_over:
            e_map.update(e_over)
        tlist = []
        for i, t in enumerate(teams):
            d = {"name": t, "seed": i + 1,
                 "points": 1000.0 - 10.0 * i if rank_points else None}
            if not rank_points:
                d["rating"] = ratings[t] + (rating_over or {}).get(t, 0.0)
                d["exposure"] = e_map[t]
                d["games"] = 20 if t != "T09" else 2
                d["rank"] = ranks[t]
                if t in drop_exposure:
                    del d["exposure"]
            tlist.append({k: v for k, v in d.items() if v is not None})
        return {"source": "rank_points" if rank_points else "elo", "teams": tlist}

    try:
        build_posterior(mk_roster(rank_points=True))
        chk(False, "rank_points 源应抛 ValueError")
    except ValueError:
        chk(True, "rank_points 源应抛 ValueError")
    try:
        build_posterior(mk_roster(drop_exposure=("T01",)))
        chk(False, "缺 exposure 应抛 ValueError")
    except ValueError:
        chk(True, "缺 exposure 应抛 ValueError")
    bad = mk_roster(e_over={"T01": -1})
    try:
        build_posterior(bad)
        chk(False, "负 exposure 应抛 ValueError")
    except ValueError:
        chk(True, "负 exposure 应抛 ValueError")

    # 5) build_posterior：blend 规则与 σ 公式
    #    T09 低暴露(E=0)、T05 高暴露(E=5)，两者 rating 各虚高 +120
    inflated = {"T09": 120.0, "T05": 120.0}
    own = {t: ratings[t] + inflated.get(t, 0.0) for t in teams}
    roster = mk_roster(e_over={"T09": 0, "T05": 5}, rating_over=inflated)
    post = build_posterior(roster)
    chk(post["fit"] is not None, "16 队有 rank ⇒ 应有先验拟合")
    p9 = post["teams"]["T09"]
    chk(p9["blended"] and p9["mu"] < own["T09"] - 40.0,
        "低暴露虚高队应向 rank 先验回归（mu=%.1f own=%.1f）" % (p9["mu"], own["T09"]))
    p5 = post["teams"]["T05"]
    chk(not p5["blended"] and p5["mu"] == own["T05"],
        "E≥5 应保持自身 rating")
    chk(abs(p9["sigma"] - team_sigma(0, 2)) < 1e-12, "σ 应按公式合成")

    # 6) sigma 直接覆盖：跳过推导、不需求 exposure
    ov = {"source": "elo", "teams": [{"name": t, "seed": i + 1,
                                      "rating": ratings[t], "sigma": 55.5}
                                     for i, t in enumerate(teams)]}
    post2 = build_posterior(ov)
    chk(all(v["sigma"] == 55.5 for v in post2["teams"].values()), "sigma 覆盖应生效")
    chk(all(not v["blended"] for v in post2["teams"].values()),
        "无 exposure ⇒ 不做均值回归")

    # 7) σ=0 ⇒ 与点估计臂逐字节一致（RNG 布局的等价性）
    k1, k3 = 2.0, 2.0
    rr = {t: 1375.0 + 25.0 * (i + 1) for i, t in enumerate(teams)}   # 1400..1775
    mus = [rr[t] for t in teams]
    zero = [0.0] * 16
    su, stg = simulate_uncertain(teams, seeds, mus, zero, k1, k3, 300, 41)
    sp, _ = SW.simulate_scenes(teams, seeds,
                               S.fast_pwin(S.EloSource(rr, k1=k1, k3=k3), teams),
                               300, seed=42)
    chk(su == sp, "σ=0 臂应与点估计臂逐字节相同")
    chk(stg["bo3"] == 13 and stg["bo1"] == 20, "σ 臂 bo 计数应同规则 20/13")
    su2, _ = simulate_uncertain(teams, seeds, mus, zero, k1, k3, 300, 41)
    chk(su == su2, "同 seed 两次连跑应逐字节一致")
    su3, _ = simulate_uncertain(teams, seeds, mus, zero, k1, k3, 300, 42)
    chk(su != su3, "不同 seed 应给出不同场景")

    # 8) 机制在场：大 σ 应压平尾部（强队 P(3:0) 降、弱队 P(3:0) 升）
    def p30(scenes, t):
        return sum(1 for s in scenes if s[t] == (3, 0)) / len(scenes)

    spt, _ = SW.simulate_scenes(teams, seeds,
                                S.fast_pwin(S.EloSource(rr, k1=k1, k3=k3), teams),
                                2000, seed=8)
    sb, _ = simulate_uncertain(teams, seeds, mus, [400.0] * 16, k1, k3, 2000, 7)
    chk(p30(sb, "T16") < p30(spt, "T16") - 0.02,
        "大 σ 应压低强队 P(3:0)（%.3f → %.3f）" % (p30(spt, "T16"), p30(sb, "T16")))
    chk(p30(sb, "T01") > p30(spt, "T01") + 0.02,
        "大 σ 应抬高弱队 P(3:0)（%.3f → %.3f）" % (p30(spt, "T01"), p30(sb, "T01")))

    # 9) tie_break 透传给 swiss：random 口径可复现且与默认不同
    sig80 = [80.0] * 16
    ur1, _ = simulate_uncertain(teams, seeds, mus, sig80, k1, k3, 200, 5, tie_break="random")
    ur2, _ = simulate_uncertain(teams, seeds, mus, sig80, k1, k3, 200, 5, tie_break="random")
    chk(ur1 == ur2, "tie_break='random' 同 seed 两次应逐字节一致")
    us1, _ = simulate_uncertain(teams, seeds, mus, sig80, k1, k3, 200, 5)
    chk(us1 != ur1, "random 与默认（seed）口径应产生不同场景")

    print("UNCERTAIN_SELFTEST_OK (%d checks)" % ok)
    return 0


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        raise SystemExit(_selftest())
    print(__doc__)
