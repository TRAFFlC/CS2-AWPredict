"""争第一决策层：给定榜首真实累计分，最大化 P(我的最终总分 > 榜首最终总分)。

与 `predict.py match --objective rank` 的区别（这是新增的部分）：
  · rank 目标旧版只优化 **E[名次]**。E[名次] 是人群密度的函数，与"能不能拿第一"无关。
  · 这里优化的是 **P(总分超过榜首)**，阈值 = 榜首的**真实累计分**（由台账给出），
    这是一个**有明确阈值的一次事件**，不是排名期望。
  ⇒ 行为差异是**涌现的**，不是手写规则：
    落后 40 分时阈值远高于我的均值 → 加方差有正收益 → λ 变高（赌）；
    领先 10 分时阈值贴近我的均值 → 加方差是负收益 → λ 归零（稳）。

三件必须说清的事：
  1. **蒙特卡洛采样的是真值，不是我的提交。** 每场只采一次真值，所有候选方案
     共享同一批真值 ⇒ 方案之间可比，差值不是采样噪声。
  2. **榜首用一个行为模型表示**：`--p-opt` 概率按 argmax-E 提交，其余概率从候选里随机挑。
     这不是"模拟榜首"，只是"榜首 = 一个比我更接近最优的对手"的最小假设。
     榜首历史预测拿不到 ⇒ 没有更好的做法，只能把 p-opt 当敏感参数扫。
  3. **δ 不进这里。** 台账实测 δ=+0.044（0.4σ）⇒ 与 0 无异，不做修正。
     若将来 δ 显著为正/负，用 `--delta` 平移胜方概率。

用法：
  python decide_rank.py --A FURIA --B MOUZ --bo 3 --A2 BIG --B2 Spirit --bo2 3 \
      --my-score 855 --leader-score 900 --trials 2000
"""
import argparse, collections, json, math, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "../data")  # # 仓库布局：数据与代码分目录（Alpha 1.1.1）
sys.path.insert(0, HERE)
sys.path.insert(0, HERE)  # scorer.py 与本模块同目录（仓库布局）

import decide
import predict as PR
import scorer as SC


def load_ledger_gap():
    p = os.path.join(DATA, "groundtruth", "leaderboard_ledger.jsonl")
    if not os.path.exists(p):
        return None
    rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    if not rows:
        return None
    ls = rows[-1].get("leader_score")
    return None if ls is None else ls


def build_md(A, B, bo, pred, delta=0.0):
    s = pred.live_sample(A, B, bo, None)
    if not s:
        raise SystemExit("no sample for %s vs %s" % (A, B))
    if delta:
        p = min(max(s["pA"] + delta * (1 if True else 0), 0.001), 0.999)
        s = dict(s, pA=p)
    return decide.MatchDist(s, pred.mmodel, pred.smodel), s


def sample_truth(md, rng, n):
    """采 n 个真值。返回 list[SC.Truth]（读法 B：maps = veto 后全部图）。"""
    wsel = list(md.p_win.items())
    ws = [w for w, _ in wsel]
    pw = np.array([p for _, p in wsel])
    out = []
    mapsets = list(md.map_true.items())
    mset = [k for k, _ in mapsets]
    msp = np.array([v for _, v in mapsets])
    for _ in range(n):
        w = ws[rng.choice(len(ws), p=pw)]
        sd = list(md.score_true[w].items())
        sl = [k for k, _ in sd]
        sp = np.array([v for _, v in sd])
        score = sl[rng.choice(len(sl), p=sp)]
        ms = mset[rng.choice(len(mset), p=msp)]
        mv = rng.choice(list(md.mvp_true[w].keys()),
                        p=np.array(list(md.mvp_true[w].values())))
        out.append(SC.Truth(teamA=md.A, teamB=md.B, winner=w, score=tuple(score),
                            maps=list(ms), mvp=str(mv), best_of=md.bo,
                            maps_mode="veto"))
    return out


def payoff(plans, truths):
    """P[i][t] = 候选 i 在真值 t 上的得分。"""
    P = np.zeros((len(plans), len(truths)))
    for i, pl in enumerate(plans):
        pred = SC.Prediction(winner=pl["winner"], score=tuple(pl["score"]),
                             maps=list(pl["maps"]), mvp=pl["mvp"])
        for t, tr in enumerate(truths):
            P[i, t] = SC.score_match(pred, tr).total
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--A", action="append", required=True)
    ap.add_argument("--B", action="append", required=True)
    ap.add_argument("--bo", type=int, action="append", required=True)
    ap.add_argument("--my-score", type=int, default=None)
    ap.add_argument("--leader-score", type=int, default=None)
    ap.add_argument("--p-opt", type=float, default=0.85,
                    help="榜首按 argmax-E 提交的概率（敏感参数）")
    ap.add_argument("--leader-observed", type=float, default=None,
                    help="榜首在某赛事已拿到的**真实**分数（用于标定其剩余表现）")
    ap.add_argument("--leader-max", type=float, default=None,
                    help="上述赛事的满分。给了这两个就**不再用 p_opt 猜**榜首，"
                         "而是按 榜首已实现得分率 × 剩余满分 推其剩余期望分")
    ap.add_argument("--trials", type=int, default=2000)
    ap.add_argument("--top-plans", type=int, default=10)
    ap.add_argument("--delta", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=20261012)
    a = ap.parse_args()

    assert len(a.A) == len(a.B) == len(a.bo), "A/B/bo 数量必须一致"
    n_match = len(a.A)
    my = a.my_score
    ld = a.leader_score if a.leader_score is not None else load_ledger_gap()
    if my is None or ld is None:
        raise SystemExit("需要 --my-score 和 --leader-score（或台账里已有榜首分）")
    gap = ld - my
    print("== 争第一决策层 ==")
    print("我 %d / 榜首 %d  ⇒  缺口 %+d 分  （%d 场待预测）"
          % (my, ld, gap, n_match))
    # 榜首剩余期望分：优先用**真实榜首得分率**，没有才退回 p_opt 行为模型。
    # 真实锚点（2026-10-12 主理人提供）：
    #   EPL S24      #1 1070/1855 = 57.68%
    #   裂变天地 S3   #1  650/1315 = 49.43%   ← 同一批人，两个赛事差 8.25pp
    # ⇒ **榜单难度按赛事变化，不能用固定 N 跨赛事比较**。
    rem_max = sum({1: 25, 3: 45, 5: 55}.get(b, 45) for b in a.bo)
    if a.leader_observed is not None and a.leader_max:
        rate = a.leader_observed / a.leader_max
        rem_mu = rate * rem_max
        print("榜首标定：实测 %g/%g = %.2f%% ⇒ 剩余 %d 分的期望 %.1f"
              % (a.leader_observed, a.leader_max, 100 * rate, rem_max, rem_mu))
    else:
        rate = rem_mu = None
        print("榜首标定：**未提供真实基准**，退回行为模型 p_opt=%.2f（拍脑袋值）"
              % a.p_opt)

    pr = PR.Predictor(quiet=True)
    rng = np.random.default_rng(a.seed)

    MDS, PLAYS, PMEANS, PSDS, DIAG = [], [], [], [], []
    for i in range(n_match):
        md, _ = build_md(a.A[i], a.B[i], a.bo[i], pr, a.delta)
        # ---- 候选集必须**风险剖面有差异**，否则 λ 扫描是平的（v1 的坑）----
        # 关键：胜方项方差 100·p(1−p) 对 p/1−p 对称 ⇒ 选冷门**不增方差**，只降均值。
        # 真正独立于胜负的方差源是**选图项** 25·Var(重叠数)。故候选按「风险档」取，
        # 而不是按 E+λσ 排序后截断（那样会留下近乎重复的图集变体）。
        cand = {}
        mom = {}
        for pl in md.plans:
            k = (pl["winner"], tuple(pl["score"]), tuple(pl["maps"]), pl["mvp"])
            if k not in mom:
                mom[k] = md.moments(pl)
        keys = list(mom)
        E = np.array([mom[k][0] for k in keys])
        S = np.array([math.sqrt(max(0.0, mom[k][1])) for k in keys])
        for lam0 in (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 20.0):
            cand[int(np.argmax(E + lam0 * S))] = None
        cand[int(np.argmax(S))] = None          # 纯最大方差
        cand[int(np.argmin(S))] = None          # 纯最小方差（保守）
        cand[int(np.argmax(E))] = None          # 纯最大期望
        # 图集维度单独取样：重叠数方差最大的图集
        ov = {}
        for k in keys:
            ov.setdefault(k[2], []).append(mom[k])
        ovk = list(ov)
        ovv = np.array([max(v[1] for v in ov[k]) for k in ovk])
        cand[int(np.argmax([i for i, k in enumerate(keys) if k[2] == ovk[int(np.argmax(ovv))]][0]))] = None
        cand[int(np.argmax([i for i, k in enumerate(keys) if k[2] == ovk[int(np.argmin(ovv))]][0]))] = None

        idx = sorted(cand)
        idx = idx[:max(a.top_plans, 8)]
        plans = []
        for j in idx:
            k = keys[j]
            plans.append({"winner": k[0], "score": list(k[1]),
                          "maps": list(k[2]), "mvp": k[3]})
        MDS.append(md)
        PLAYS.append(plans)
        PMEANS.append([mom[keys[j]][0] for j in idx])
        PSDS.append([math.sqrt(max(0.0, mom[keys[j]][1])) for j in idx])
        print("\n-- %s vs %s (bo%d)  %d/%d 候选（按风险档选取）--"
              % (a.A[i], a.B[i], a.bo[i], len(plans), len(md.plans)))
        for j in np.argsort(-np.array(PMEANS[-1])):
            pl = plans[j]
            print("   %-7s %s  maps=%-32s mvp=%-12s E=%5.1f sd=%4.1f"
                  % (pl["winner"], "%d:%d" % tuple(pl["score"]),
                     ",".join(pl["maps"][:3]), pl["mvp"],
                     PMEANS[-1][j], PSDS[-1][j]))
        print("   ⇒ sd 范围 %.1f ~ %.1f（方差确实是活杠杆）"
              % (min(PSDS[-1]), max(PSDS[-1])))
        # ---- 诊断：这个提交空间里到底有没有「高方差低均值」的赌选项？----
        allE = np.array([mom[k][0] for k in keys])
        allS = np.array([math.sqrt(max(0.0, mom[k][1])) for k in keys])
        cc = float(np.corrcoef(allE, allS)[0, 1]) if len(allE) > 2 else float("nan")
        maxE = allE.max()
        gamble = [(allE[i], allS[i], keys[i]) for i in range(len(allE))
                  if allS[i] > allS[int(np.argmax(allE))] - 1e-9
                  and allE[i] < maxE - 1e-9]
        DIAG.append((cc, len(gamble), len(allE)))
        print("   ⇒ 全 %d 个方案：corr(E, sd) = %+.3f；"
              "「sd 不低于最高 E 方案、但 E 更低」的赌选项 = %d 个"
              % (len(allE), cc, len(gamble)))

    T = a.trials
    print("\n== 蒙特卡洛 %d 组真值（全场共享，方案间可比）==" % T)
    PAY = []
    for i in range(n_match):
        truths = sample_truth(MDS[i], rng, T)
        P = payoff(PLAYS[i], truths)
        PAY.append(P)
        print("   %s vs %s  期望 %.1f  最好单场上限 %.1f  实际得分 sd %.1f"
              % (a.A[i], a.B[i], P[0].mean(), P.max(axis=0).mean(), P[0].std()))

    # 榜首：p_opt 按 argmax-E，否则随机候选；若有真实基准则整体缩放到实测得分率
    ld_tot = np.zeros(T)
    for i in range(n_match):
        P = PAY[i]
        best = P[0]
        pick = rng.random(T) < a.p_opt
        rnd_idx = rng.integers(0, P.shape[0], T)
        ld_tot += np.where(pick, best, P[rnd_idx, np.arange(T)])
    if rem_mu is not None:
        # 行为模型的形状保留（它刻画「怎么错」），尺度按真实榜首得分率重标。
        cur = ld_tot.mean()
        if cur > 0:
            ld_tot *= rem_mu / cur
            print("榜首剩余期望：行为模型 %.1f → 按实测基准缩放到 %.1f"
                  % (cur, rem_mu))

    print("\n== λ 扫描（λ 越大越重方差）==")
    print("   λ     E[我的分]   P(超榜首)   最差5%%分位   解读")
    rows = []
    for lam in (0.0, 0.15, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
        mine = np.zeros(T)
        for i in range(n_match):
            k = int(np.argmax(np.array(PMEANS[i]) + lam * np.array(PSDS[i])))
            mine += PAY[i][k]
        # 阈值 = 榜首**累计**分 − 我的累计分。注意要拿"剩余分之差 > gap"，
        # 不能只比剩余分——否则榜首累计分完全不起作用（v1 的 bug，恒为同一值）。
        win = float(np.mean(mine - ld_tot > gap))
        rows.append((lam, mine.mean(), win, np.percentile(mine, 5)))
        print("  %4.2f   %8.1f   %8.1f%%   %9.1f"
              % (lam, mine.mean(), 100 * win, np.percentile(mine, 5)))
    best = max(rows, key=lambda r: (r[2], r[1]))
    lam = best[0]
    print("\n推荐 λ = %.2f   ⇒ P(超榜首) = %.1f%%   E[我的分] = %.1f"
          % (best[0], 100 * best[2], best[1]))
    flat = len({r[1] for r in rows}) == 1
    if flat:
        mcc = np.mean([d[0] for d in DIAG])
        ng = sum(d[1] for d in DIAG)
        nt = sum(d[2] for d in DIAG)
        print("  ⚠ **λ 扫描完全平坦**：每个 λ 的 E[我的分] 相同 ⇒ 提交空间里不存在"
              "「高方差低均值」的方案。")
        print("    证据：%d 个方案上 corr(E, sd) = %+.3f（正相关），"
              "符合条件的赌选项 %d 个。" % (nt, mcc, ng))
        print("    含义：**「落后就加注」在本模型下没有可执行形式**——不是搜索失败，"
              "是提交空间的性质。")
        print("    ⇒ 决策层退化为「最大化 E[分]」，P(超榜首) 只作为诚实的胜率上报。")
    else:
        print("  ⇒ 缺口 %+d：阈值%s ⇒ 加方差%s"
              % (gap, "远高于我的期望" if gap > best[1] else "贴近/低于我的期望",
                 "有上行收益，值得赌" if lam > 0 else "只有下行风险，按 E[分] 走"))

    print("\n== 最终提交（λ=%.2f）==" % lam)
    out = []
    for i in range(n_match):
        k = int(np.argmax(np.array(PMEANS[i]) + lam * np.array(PSDS[i])))
        pl = PLAYS[i][k]
        mu, sd = PMEANS[i][k], PSDS[i][k]
        # ⚠ 显示用概率降序的 maps_ranked，别直接 join(pl["maps"])（那是字母序，
        # 会把 Mirage 74% 排在 Dust2 57% 后面 —— 2026-10-19 主理人抓过）。
        # out[] 里的 "maps" 保持字母规范序：它是计分/去重用的身份。
        _rk = list(pl.get("maps_ranked") or pl["maps"])
        print("  %s vs %s (bo%d)：%s  %d:%d  [%s]  MVP %s   E=%.1f sd=%.1f"
              % (a.A[i], a.B[i], a.bo[i], pl["winner"], pl["score"][0],
                 pl["score"][1], ",".join(_rk), pl["mvp"], mu, sd))
        out.append({"A": a.A[i], "B": a.B[i], "bo": a.bo[i], "lambda": lam,
                   "winner": pl["winner"], "score": list(pl["score"]),
                   "maps": list(pl["maps"]), "maps_ranked": _rk, "mvp": pl["mvp"],
                    "E": round(mu, 2), "sd": round(sd, 2),
                    "p_win": {a.A[i]: MDS[i].p_win[a.A[i]],
                              a.B[i]: MDS[i].p_win[a.B[i]]}})
    json.dump({"my_score": my, "leader_score": ld, "gap": gap, "lambda": lam,
               "p_beat": round(best[2], 4), "p_opt_leader": a.p_opt,
               "delta": a.delta, "trials": T, "matches": out},
              open(os.path.join(HERE, "decide_rank_out.json"), "w",
                   encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n已写 decide_rank_out.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
