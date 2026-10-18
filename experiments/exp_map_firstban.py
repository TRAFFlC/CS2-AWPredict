"""exp_map_firstban.py —— 选图模型「先手方 / 厌恶度」三处改动的对照实验

## 要验证的三件事（主理人 2026-10-19 指出的反例）

2026-07-19 的 bo5 决赛（BetBoom vs FURIA）上，模型把 **Inferno** 排进 5 张预测
（概率 0.268，第 5），实际 BetBoom 首 ban 了 Inferno ⇒ 白丢 5 分。主理人问：
「BetBoom 几乎从来不打小镇，模型为什么还预测打 Inferno？」

查出来的根因不是「模型没看见」，而是**ban 目标选错了**：

    simulate_veto() 的 ban 步取的是 team_ban()（**历史 ban 频次**）
      BetBoom 的 ban 率：Anubis 0.178 > Inferno 0.156 > Nuke 0.144 …
    但 BetBoom 的 pick 率：Inferno 0.042 << Anubis 0.169   ← 恰好相反

⇒ 真实 veto 是「ban 自己**不想打**的图」，而模型用「历史上 ban 过几次」。
图不在自己常用池时，两者会系统性地背离。

三条改法（**默认全部关闭**，生产路径逐位不变）：

  A. 把先手方归属做成特征：`first_banner()` 早就算出来了（实测弱队先 ban
     命中率 66.5%），但**从来没进过任何特征集**。而「谁先 ban」决定了
     「谁的意见算数」——`a_ban`/`b_ban` 是无主的。
  B. 厌恶度特征：`team_averse = max(0, g_app − team_app)`，
     以及 pick 侧的 `team_pick_averse`。
  C. `simulate_veto(ban_rule=...)`：ban 步改按厌恶度（averse）/ 折中（mixed）。

## 为什么必须做 A/B 而不能直接改

「首 ban 了 Inferno」是**单场证据**。它能说明模型在这一场错了，不能说明
「按厌恶度选 ban」总体更好。所以这里跑**同一条 walk-forward**、同一份 split、
同一个 C 扫法，逐变体比 valid/test 的选图分。基线（改动前实测）：

    valid 选图 2130 分（命中 64.4%）
    test  选图 2765 分（命中 61.4%，槽位 901）  各槽位 {0:208,1:179,2:163,3:2,4:1}

⇒ **只有 test 上的提升才算数**，valid 只用来选 C（避免用 test 选型）。

## 读数时特别注意

`各槽位命中 {3:2, 4:1}` —— **槽位 3/4 基本是 0**。bo5 只有少数几场，
后两槽位的样本量极小（个位数）。任何"槽位 3/4 变好了"的说法都要看分母。

用法：
    python -X utf8 exp_map_firstban.py            # 全部变体
    python -X utf8 exp_map_firstban.py --quick    # 只跑基线 + 三个特征变体
"""
import argparse
import collections
import io
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "../data")  # # 仓库布局：数据与代码分目录（Alpha 1.1.1）
sys.path.insert(0, HERE)
import model_v2 as M
import pool_experiment as PE
import split as SPLIT
import veto_model as VM

TRUTH = os.path.join(DATA, "truth_maps_B.jsonl")
MATCHES = os.path.join(DATA, "csapi_matches_all.jsonl")
EXCLUDE = os.path.join(HERE, "split_manifest__purge_exclude.txt")
CS = (0.03, 0.1, 0.3, 1.0, 3.0)

BASE = VM.FEATURES
FIRST_F = ["first_is_a", "first_is_b", "a_first_ban", "b_first_ban",
           "first_averse", "first_pick_averse"]
AVERSE_F = ["a_averse", "b_averse", "a_pick_averse", "b_pick_averse"]

VARIANTS = {
    "V0 基线":            BASE,
    "V1 +先手":          BASE + FIRST_F,
    "V2 +厌恶度":        BASE + AVERSE_F,
    "V3 +先手+厌恶度":   BASE + FIRST_F + AVERSE_F,
    "V3r 生产集+全部新":  VM.FEATURES3 + VM.EXTRA_FEATURES,
}


def load():
    rows = SPLIT.load(MATCHES)
    exclude = SPLIT.load_exclude(EXCLUDE) if os.path.exists(EXCLUDE) else None
    buckets, _meta = SPLIT.split(rows, purge=True, exclude=exclude)
    truths = {t["id"]: t for t in
              (json.loads(l) for l in io.open(TRUTH, encoding="utf-8") if l.strip())}
    veto = VM.load_veto()
    eras, by_pool = PE.load_pool_eras()
    tl = json.load(io.open(PE.TIMELINE, encoding="utf-8"))

    def pool_of(r):
        t = truths.get(r["id"]) or {}
        if t.get("pool"):
            return list(t["pool"])
        return list(PE.era_and_pool(r["date"], [], eras, by_pool, tl)[1])
    return buckets, truths, veto, pool_of


def build_ext(rows_bucket, truths, veto, hist, pool_of):
    """= `veto_model.build_dataset`，但每场额外记下三种 ban 规则下的模拟图集。

    ⚠ 时序与原函数**逐行一致**：同日先全部取特征，再全部 observe
      ⇒ 同日比赛互不可见，训练/评测口径不变。
    """
    order = sorted(range(len(rows_bucket)),
                   key=lambda i: (rows_bucket[i]["date"], rows_bucket[i]["id"]))
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
            bo = (t.get("bo") or r.get("best_of") or 3)
            feats = hist.features_for(tA, tB, pool, bo, date=r.get("date"))
            tset = set(t.get("maps") or [])
            first = VM.first_banner(tA, tB, r.get("date"))
            k = max(1, min(int(bo or 3), 5))
            sims = {}
            for rule in ("hist", "averse", "mixed"):
                sims[rule] = hist.simulate_veto(tA, tB, pool, bo, first=first,
                                                ban_rule=rule)[:k]
            pending.append({"id": r["id"], "date": d, "teamA": tA, "teamB": tB,
                            "bo": bo, "pool": pool, "rows": feats,
                            "y": [1 if f["map"] in tset else 0 for f in feats],
                            "sims": sims, "first": first,
                            "y_set": tset})
        samples.extend(pending)
        for r in group:
            tA, tB = r["team1"]["name"], r["team2"]["name"]
            t = truths.get(r["id"]) or {}
            pool = pool_of(r)
            mres = [(m["name"], m["team1_score"], m["team2_score"])
                    for m in (r.get("maps") or [])]
            hist.observe_match(tA, tB, pool, t.get("maps") or [],
                               veto.get(r["id"]), mres)
    return samples


def score_set(samples, key=None):
    """直接按「预测图集」计分（不给分类器用）。`key` = sample['sims'][rule]。"""
    pts = hits = slots = 0
    per = collections.Counter()
    for s in samples:
        k = max(1, min(int(s["bo"] or 3), 5))
        if key:
            pred = set(s["sims"][key][:k])
        else:
            pred = set(s.get("_pred") or [])[:k]
        h = sum(1 for m in pred if m in s["y_set"])
        pts += 5 * h
        hits += h
        slots += len(pred)
        per[min(k - 1, 4)] += 0
    return {"points": pts, "hits": hits, "slots": slots,
            "hit_rate": hits / slots if slots else 0.0}


def eval_model(samples, feats, C):
    import numpy as np
    m = VM.train(samples, features=feats, C=C)
    pts = hits = slots = 0
    per_slot = collections.Counter()
    per_slot_n = collections.Counter()
    per_match = []
    by_bo = collections.defaultdict(lambda: [0, 0])
    for s in samples:
        k = max(1, min(int(s["bo"] or 3), 5))
        X = np.array([[r[f] for f in m["features"]] for r in s["rows"]],
                     dtype=float)
        p = m["clf"].predict_proba(m["scaler"].transform(X))[:, 1]
        ranked = [r["map"] for _pp, r in
                  sorted(zip(p, s["rows"]), key=lambda kv: (-kv[0], kv[1]["map"]))]
        pred = ranked[:k]
        h0 = 0
        for j, mname in enumerate(pred):
            ok = 1 if mname in s["y_set"] else 0
            hits += ok
            pts += 5 * ok
            slots += 1
            h0 += ok
            per_slot[j] += ok
            per_slot_n[j] += 1
        per_match.append(h0)
        b = by_bo[s["bo"]]
        b[0] += h0
        b[1] += len(pred)
    return {"points": pts, "hits": hits, "slots": slots,
            "hit_rate": hits / slots if slots else 0.0,
            "per_match": per_match,          # ← 供配对自助法用
            "per_slot": {j: "%d/%d" % (per_slot[j], per_slot_n[j])
                         for j in sorted(per_slot_n)},
            "by_bo": by_bo}


def paired_bootstrap(diff, trials=20000, seed=20260719):
    """配对自助法：逐场命中数之差。两组跑的是**同一批场次**，
    场次难度被消掉，所以只比「改动的边际效果」。"""
    import random
    n = len(diff)
    if not n:
        return {}
    rng = random.Random(seed)
    mean = sum(diff) / n
    boots = sorted(sum(diff[rng.randrange(n)] for _ in range(n)) / n
                   for _ in range(trials))
    lo, hi = boots[int(0.025 * trials)], boots[int(0.975 * trials)]
    p = 2.0 * min(sum(1 for x in boots if x <= 0),
                  sum(1 for x in boots if x >= 0)) / trials
    return {"n": n, "mean_hits": mean, "ci95_hits": [lo, hi], "p": p,
            "wins": sum(1 for x in diff if x > 0),
            "losses": sum(1 for x in diff if x < 0),
            "changed": sum(1 for x in diff if x != 0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="exp_map_firstban_results.json")
    a = ap.parse_args()

    buckets, truths, veto, pool_of = load()
    # ⚠ 与 `veto_model.run()` **同一口径**：一个 hist 依次推进 train → valid → test，
    #   valid/test 必须看到 train 期间积累的历史。分成两个 hist 就是另一个实验了。
    hist = VM.MapHistory(half_life=VM.HALF_LIFE)
    tr = build_ext(buckets["train"], truths, veto, hist, pool_of)
    va = build_ext(buckets["valid"], truths, veto, hist, pool_of)
    te = build_ext(buckets["test"], truths, veto, hist, pool_of)
    print("样本：train %d / valid %d / test %d\n" % (len(tr), len(va), len(te)))

    res = {}
    print("=" * 78)
    print("%-22s %10s %10s %10s" % ("变体", "valid", "test", "test 命中率"))
    print("=" * 78)
    names = list(VARIANTS)[:4] if a.quick else list(VARIANTS)
    for name in names:
        feats = VARIANTS[name]
        missing = set(feats) - set(tr[0]["rows"][0].keys())
        if missing:
            print("%-22s 跳过（缺特征 %s）" % (name, sorted(missing)[:3]))
            continue
        bv = bvC = None
        for C in CS:                      # C **只**用 valid 选（与 veto_model.run 同口径）
            ev = eval_model(va, feats, C)
            if bv is None or ev["points"] > bv["points"]:
                bv, bvC = ev, C
        bt = eval_model(te, feats, bvC)   # test 只跑一遍，不在这里选
        res[name] = {"valid": bv, "valid_C": bvC, "test": bt}
        print("%-22s %10d %10d %9.1f%%   (C=%s)"
              % (name, bv["points"], bt["points"], 100 * bt["hit_rate"], bvC))
    b0 = res.get("V0 基线", {}).get("test", {}).get("points")
    ref = res.get("V0 基线", {}).get("test", {}).get("per_match")
    if b0:
        print("-" * 78)
        for name, r in res.items():
            if not name.startswith("V"):
                continue
            d = r["test"]["points"] - b0
            line = "  %-22s Δtest = %+d 分（%+.2f%%）" % (name, d, 100.0 * d / b0)
            if ref and r["test"].get("per_match"):
                bs = paired_bootstrap([x - y for x, y in
                                      zip(r["test"]["per_match"], ref)])
                if bs:
                    line += ("   配对自助：每场 %+.3f 命中，95%%CI [%+.2f, %+.2f]，"
                             "p=%.3f（%d 场变好 / %d 场变差 / %d 场有变化）"
                             % (bs["mean_hits"], bs["ci95_hits"][0],
                                bs["ci95_hits"][1], bs["p"], bs["wins"],
                                bs["losses"], bs["changed"]))
                    r["bootstrap_vs_V0"] = bs
            print(line)

    print("\n--- 分 bo 的命中率（V0 vs 生产集+全部新）---")
    best = max((n for n in res if n.startswith("V")),
               key=lambda n: res[n]["test"]["points"])
    for bo in sorted(res["V0 基线"]["test"]["by_bo"],
                     key=lambda x: str(x)):
        h0, s0 = res["V0 基线"]["test"]["by_bo"][bo]
        h1, s1 = res[best]["test"]["by_bo"].get(bo, [0, 0])
        print("  bo%-4s V0 %3d/%3d = %5.1f%%   %s %3d/%3d = %5.1f%%"
              % (bo, h0, s0, 100.0 * h0 / max(1, s0), best,
                 h1, s1, 100.0 * h1 / max(1, s1)))

    print("\n--- 直接用 simulate_veto 的图集（不经分类器）---")
    print("%-22s %10s %10s %10s" % ("ban_rule", "valid", "test", "test 命中率"))
    for rule in ("hist", "averse", "mixed"):
        ev = score_set(va, rule)
        et = score_set(te, rule)
        res["sim:" + rule] = {"valid": ev, "test": et}
        print("%-22s %10d %10d %9.1f%%"
              % ("simulate_veto " + rule, ev["points"], et["points"],
                 100 * et["hit_rate"]))

    with io.open(os.path.join(HERE, a.out), "w", encoding="utf-8",
                 newline="\n") as f:
        f.write(json.dumps(res, ensure_ascii=False, indent=2) + "\n")
    print("\n已写 %s" % a.out)


if __name__ == "__main__":
    sys.exit(main())