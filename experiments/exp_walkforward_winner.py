# -*- coding: utf-8 -*-
"""exp_walkforward_winner.py —— 胜方项裁决：**更新后的模型** vs **rank 热门** vs **置信度门控**。

背景（2026-10-13，主理人要一个明确裁决）
  · 语料只到 2026-09-05；主理人另录 112 场 09-08~10-11 的赛果（new_records_2026-09.jsonl）。
  · 现行胜方规则 = rank 热门（predict.py 2026-10-12 裁决），依据是 43 场对照
    「rank 69.8% vs 模型 62.8%」——**但那批预测用的是停在 09-05 的模型状态**，
    新数据当时还没摄入（或在预测之后才摄入）。⇒ 「模型不如 rank」是在
    「模型落后 5 周」的条件下得出的，结论未在更新后的模型上重测。

本实验（严格时序前进式，任何一场比赛的状态里都**只有更早日期**的比赛）：
  for 按日期排序的每一天 D:
      先用「截至 D-1 日末」的状态预测 D 日全部比赛
      再把 D 日比赛摄入账本（state + map history）
  ⇒ 112 场全部是 out-of-sample，无前视、无同日泄漏（与 state_snapshot.ingest
     的「逐日两趟」约束一致：同日比赛互不可见）。

四条臂：
  模型·更新   逐步摄入（每场只见过更早的赛果）  ← 主理人想知道的「新数据有没有被用上」
  模型·冻结   永不摄入，停在 09-05              ← 对照：更新到底带来多少
  rank 热门   csapi 09-05 快照（现行规则）
  门控(t)     模型置信度 ≥ t 跟模型，否则跟 rank（含 t→0.5 的「全跟模型」）

统计：McNemar 精确检验（scipy binomtest，双侧），按 1SE 粗判；分 9 月 / 10 月子集
（rank 快照到 10 月已陈旧 25~36 天，分层看陈旧度是否吃掉 rank 的优势）。

用法：  python -X utf8 exp_walkforward_winner.py
"""
from __future__ import annotations

import io
import json
import os
import sys

import numpy as np
from scipy.stats import binomtest

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "../data")  # # 仓库布局：数据与代码分目录（Alpha 1.1.0）
sys.path.insert(0, HERE)
sys.path.insert(0, HERE)  # scorer.py 与本模块同目录（仓库布局）

# 两个 Predictor 都从 **09-05 语料基线**出发；更新臂靠自己逐步摄入推进。
os.environ["ISSUE014_NO_SNAPSHOT"] = "1"

import model_v2 as M            # noqa: E402
import pool_experiment as PE    # noqa: E402
import predict as PR            # noqa: E402
import record_schema as RS      # noqa: E402
import state_snapshot as SS     # noqa: E402

NEW = os.path.join(DATA, "groundtruth", "new_records_2026-09.jsonl")


def load():
    recs = [json.loads(l) for l in io.open(NEW, encoding="utf-8") if l.strip()]
    recs = [r for r in recs if r.get("winner") and r.get("best_of") in (1, 3, 5)]
    recs.sort(key=lambda r: (r["date"], str(r["id"])))
    return recs


def make_ctx(prep):
    return SS.make_ctx(prep["truths"], prep["veto"], M.load_stats(),
                       M.load_stats_map(), *PE.load_pool_eras(),
                       json.load(io.open(PE.TIMELINE, encoding="utf-8")))


def acc(hit, n):
    return hit / n if n else 0.0


def mcnemar(a, b):
    """a/b = 两个 0/1 命中序列 → (b_only, a_only, p 值)。"""
    a_only = int(np.sum((np.array(a) == 1) & (np.array(b) == 0)))
    b_only = int(np.sum((np.array(a) == 0) & (np.array(b) == 1)))
    n = a_only + b_only
    p = float(binomtest(a_only, n, 0.5).pvalue) if n else 1.0
    return a_only, b_only, p


def main():
    recs = load()
    print("赛果 %d 场  %s .. %s" % (len(recs), recs[0]["date"], recs[-1]["date"]))

    P_upd = PR.Predictor(quiet=True)     # 更新臂（逐步摄入）
    P_old = PR.Predictor(quiet=True)     # 冻结臂（永不摄入）
    ctx = make_ctx(P_upd.prep)
    print("[state] 更新臂基线 last_date=%s（快照未载入，符合预期）"
          % getattr(P_upd.state, "last_date", "?"))

    # ---- 按日推进 ----
    days = []
    for r in recs:
        if not days or days[-1][0] != r["date"]:
            days.append((r["date"], []))
        days[-1][1].append(r)

    rows = []
    for date, group in days:
        for r in group:
            A, B, bo = r["team1"]["name"], r["team2"]["name"], r["best_of"]
            truth = r["winner"]

            s_new = P_upd.live_sample(A, B, bo, date=date)
            s_old = P_old.live_sample(A, B, bo, date=date)
            p_new, p_old = float(s_new["pA"]), float(s_old["pA"])

            mw_new = A if p_new >= 0.5 else B
            mw_old = A if p_old >= 0.5 else B
            rw, rinfo = P_upd.rank_pick(s_new, mw_new)   # rank 快照与状态无关

            rows.append({
                "date": date, "A": A, "B": B, "bo": bo, "truth": truth,
                "p_new": p_new, "p_old": p_old,
                "conf_new": max(p_new, 1 - p_new),
                "conf_old": max(p_old, 1 - p_old),
                "m_new": mw_new, "m_old": mw_old,
                "r": rw, "rank_ok": bool(rinfo.get("ok")),
                "rankA": rinfo.get("rankA"), "rankB": rinfo.get("rankB"),
            })
        # 预测完 D 日之后才摄入 D 日（同日互不可见）
        SS.ingest(P_upd.state, P_upd.mh, [RS.to_ingest_row(r) for r in group], ctx)

    print("[state] 摄入后 last_date=%s（应为 %s）"
          % (getattr(P_upd.state, "last_date", "?"), recs[-1]["date"]))

    # ---- 计分 ----
    n = len(rows)
    y = [x["truth"] for x in rows]
    hit = {
        "模型·更新": [x["m_new"] == x["truth"] for x in rows],
        "模型·冻结": [x["m_old"] == x["truth"] for x in rows],
        # rank 缺失时回退到更新后的模型（生产同口径）
        "rank 热门": [(x["r"] or x["m_new"]) == x["truth"] for x in rows],
    }
    n_rank = sum(1 for x in rows if x["rank_ok"])
    print("\n=== 时间前进式留出 %d 场（09-08 ~ 10-11，全部无泄漏）===" % n)
    print("  其中两队都有 rank 快照的：%d 场（缺失 %d 场 → 回退模型）" % (n_rank, n - n_rank))
    print("\n  %-12s %8s %8s" % ("规则", "命中", "准确率"))
    print("  " + "-" * 34)
    for k, v in hit.items():
        h = sum(v)
        print("  %-12s %6d/%-3d %7.1f%%" % (k, h, n, 100 * acc(h, n)))

    # ---- 门控 ----
    print("\n=== 置信度门控（模型置信 ≥ t 跟模型，否则跟 rank）===")
    print("  %-10s %8s %8s %10s" % ("阈值 t", "命中", "准确率", "vs rank p"))
    print("  " + "-" * 42)
    gate_rows = []
    for t in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90):
        picks = []
        for x in rows:
            r = x["r"] or x["m_new"]
            if x["conf_new"] >= t:
                picks.append(x["m_new"])
            else:
                picks.append(r)
        k = sum(1 for p_, t_ in zip(picks, y) if p_ == t_)
        _, _, p = mcnemar([p_ == t_ for p_, t_ in zip(picks, y)], hit["rank 热门"])
        gate_rows.append((t, k, acc(k, n), p))
        print("  %-10.2f %5d/%-3d %7.1f%% %10.3f"
              % (t, k, n, 100 * acc(k, n), p))

    # ---- McNemar ----
    print("\n=== 配对检验（McNemar 精确，双侧）===")
    pairs = [("模型·更新", "rank 热门"), ("模型·冻结", "rank 热门"),
             ("模型·更新", "模型·冻结")]
    for a, b in pairs:
        ao, bo_, p = mcnemar(hit[a], hit[b])
        d = acc(sum(hit[a]), n) - acc(sum(hit[b]), n)
        print("  %-8s vs %-8s  仅A对 %2d / 仅B对 %2d   Δ=%+.1fpp  p=%.3f  ⇒ %s"
              % (a, b, ao, bo_, 100 * d, p,
                 "A 显著更好" if p < 0.05 and ao > bo_ else
                 ("B 显著更好" if p < 0.05 else "差异不显著")))

    # ---- 分层：9 月 vs 10 月（rank 快照陈旧度）----
    print("\n=== 分层（rank 快照 = csapi 09-05；10 月场次已陈旧 25~36 天）===")
    print("  %-10s %6s | %-9s %-9s %-9s" % ("月份", "场次", "模型·更新", "模型·冻结", "rank"))
    for tag, pref in (("2026-09", "2026-09"), ("2026-10", "2026-10")):
        idx = [i for i, x in enumerate(rows) if x["date"].startswith(pref)]
        if not idx:
            continue
        line = "  %-10s %6d |" % (tag, len(idx))
        for k in ("模型·更新", "模型·冻结", "rank 热门"):
            h = sum(1 for i in idx if hit[k][i])
            line += " %-9.1f%%" % (100 * acc(h, len(idx)))
        print(line)

    # ---- 置信分档 + 分歧诊断 ----
    print("\n=== 模型（更新臂）置信分档的真实命中率 ===")
    bins = [(0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.01)]
    for lo, hi in bins:
        idx = [i for i, x in enumerate(rows) if lo <= x["conf_new"] < hi]
        if not idx:
            continue
        h = sum(1 for i in idx if hit["模型·更新"][i])
        print("  [%.2f,%.2f)  %3d 场  模型 %5.1f%%" % (lo, hi, len(idx), 100 * acc(h, len(idx))))

    dis = [x for x in rows if x["r"] and x["r"] != x["m_new"]]
    print("\n=== 分歧场次（rank ≠ 模型·更新）%d/%d ===" % (len(dis), n))
    dw = sum(1 for x in dis if x["r"] == x["truth"])
    dm = sum(1 for x in dis if x["m_new"] == x["truth"])
    print("  rank 对 %d/%d = %.1f%%   模型对 %d/%d = %.1f%%"
          % (dw, len(dis), 100 * acc(dw, len(dis)), dm, len(dis), 100 * acc(dm, len(dis))))
    if dis:
        ao, bo_, p = mcnemar([x["m_new"] == x["truth"] for x in dis],
                             [x["r"] == x["truth"] for x in dis])
        print("  McNemar：仅模型对 %d / 仅 rank 对 %d  p=%.3f" % (ao, bo_, p))
    print("\n  逐条分歧（日期 / 对阵 / 实际 / 模型置信 / 模型选 / rank 选）：")
    for x in dis:
        print("    %s  %-14s vs %-14s 实际 %-14s  conf=%.2f  模型 %-14s rank %-14s %s"
              % (x["date"], x["A"], x["B"], x["truth"], x["conf_new"],
                 x["m_new"], x["r"],
                 "✔rank" if x["r"] == x["truth"] else ("✔模型" if x["m_new"] == x["truth"] else "✘")))

    # ---- 落盘 ----
    out = os.path.join(HERE, "walkforward_winner_results.json")
    json.dump({"n": n, "rows": rows,
               "acc": {k: acc(sum(v), n) for k, v in hit.items()},
               "gates": [{"t": t, "hit": k, "acc": a, "p_vs_rank": p}
                         for t, k, a, p in gate_rows]},
              io.open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n落盘：%s" % os.path.basename(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
