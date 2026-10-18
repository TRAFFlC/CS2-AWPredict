"""exp_map_shrink.py —— 收缩口径 × ban 规则的网格对照

## 起因（主理人 2026-07-19 指出的实质错误）

Thunderpick WC 2026 决赛（BetBoom vs FURIA, bo5）模型把 Inferno 排进 5 张预测
（0.268，第 5），真实 BetBoom 首 ban 了它 ⇒ 白丢 5 分。主理人：「所有人都知道
BetBoom 不打小镇，只有模型认为会，这不该修复？」

查出的**根因不是 ban 规则，是收缩把实测压烂了**：

    BetBoom 对 Inferno
      实测 tapp=0.026 / tpooln=15.9  ⇒ 真实比率 0.0016（0.16%）
      收缩 (tapp + K·g_app)/(n + K)，K_TEAM=4、g_app(Inferno)=0.366
           = (0.026 + 1.465)/19.9 = 0.0749          ← 先验项是实测的 56 倍
      对照 Mirage：tapp=10.50（实测 400 倍）→ 收缩后 0.627
    ⇒ 原始证据差 400 倍，收缩后只差 8.4 倍。小样本 + 强先验 ⇒ 证据被淹没。

统计上这是正确的贝叶斯收缩，**实质上错**：先验断言「BetBoom 打这张图的概率
是全局的 36.6%」，而证据说 0.16%。

## 三种收缩

  hist     —— 原实现：先验 = 这张图的全局比率（默认，**生产行为不变**）
  teamavg  —— 先验 = **本队自己在全部图上的平均比率**（对任一张图都中性）
  raw      —— 不收缩，直接用实测比率

## 网格

每个 (收缩, ban规则) 都**重建样本**（特征随收缩变、模拟图集随两者变），
同一条 walk-forward、同一个 C 扫法、**C 只用 valid 选**。
test 按 **bo 分开报**：bo3 有 834 槽位（有功效），bo5 只有 10（**无功效**）。

基线（改动前 `veto_model.run()` 实测）：valid 2130 / test 2765（命中 61.4%）
"""
import argparse
import collections
import io
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "../data")  # # 仓库布局：数据与代码分目录（Alpha 1.1.0）
sys.path.insert(0, HERE)
import model_v2 as M
import numpy as np
import pool_experiment as PE
import split as SPLIT
import veto_model as VM

TRUTH = os.path.join(DATA, "truth_maps_B.jsonl")
MATCHES = os.path.join(DATA, "csapi_matches_all.jsonl")
EXCLUDE = os.path.join(HERE, "split_manifest__purge_exclude.txt")
CS = (0.03, 0.1, 0.3, 1.0, 3.0)
FEATS = VM.FEATURES3 + VM.EXTRA_FEATURES      # 生产特征集 + 新增候选


def load():
    rows = SPLIT.load(MATCHES)
    exclude = SPLIT.load_exclude(EXCLUDE) if os.path.exists(EXCLUDE) else None
    buckets, _m = SPLIT.split(rows, purge=True, exclude=exclude)
    truths = {t["id"]: t for t in
              (json.loads(l) for l in io.open(TRUTH, encoding="utf-8") if l.strip())}
    veto = VM.load_veto()
    eras, by_pool = PE.load_pool_eras()
    tl = json.load(io.open(PE.TIMELINE, encoding="utf-8"))

    def pool_of(r):
        t = truths.get(r["id"]) or {}
        return list(t["pool"]) if t.get("pool") else \
            list(PE.era_and_pool(r["date"], [], eras, by_pool, tl)[1])
    return buckets, truths, veto, pool_of


def build(rows_bucket, truths, veto, hist, pool_of, shrink, ban_rule="hist"):
    order = sorted(range(len(rows_bucket)),
                   key=lambda i: (rows_bucket[i]["date"], rows_bucket[i]["id"]))
    samples, i = [], 0
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
            pool, bo = pool_of(r), (t.get("bo") or r.get("best_of") or 3)
            feats = hist.features_for(tA, tB, pool, bo, date=r["date"],
                                      shrink=shrink, ban_rule=ban_rule)
            tset = set(t.get("maps") or [])
            first = VM.first_banner(tA, tB, r["date"])
            k = max(1, min(int(bo or 3), 5))
            sims = {rule: hist.simulate_veto(tA, tB, pool, bo, first=first,
                                             ban_rule=rule, shrink=shrink)[:k]
                    for rule in ("hist", "averse", "mixed")}
            pending.append({"id": r["id"], "bo": bo, "pool": pool, "rows": feats,
                            "y": [1 if f["map"] in tset else 0 for f in feats],
                            "sims": sims, "y_set": tset})
        samples.extend(pending)
        for r in group:
            tA, tB = r["team1"]["name"], r["team2"]["name"]
            t = truths.get(r["id"]) or {}
            mres = [(m["name"], m["team1_score"], m["team2_score"])
                    for m in (r.get("maps") or [])]
            hist.observe_match(tA, tB, pool_of(r), t.get("maps") or [],
                               veto.get(r["id"]), mres)
    return samples


def score_model(samples, C):
    samples = [s for s in samples if s["y_set"]]   # 与生产同口径（veto_model.is_scorable）
    m = VM.train(samples, features=FEATS, C=C)
    per, by_bo = [], collections.defaultdict(lambda: [0, 0])
    for s in samples:
        k = max(1, min(int(s["bo"] or 3), 5))
        X = np.array([[r[f] for f in m["features"]] for r in s["rows"]], dtype=float)
        p = m["clf"].predict_proba(m["scaler"].transform(X))[:, 1]
        ranked = [r["map"] for _pp, r in
                  sorted(zip(p, s["rows"]), key=lambda kv: (-kv[0], kv[1]["map"]))]
        h = sum(1 for x in ranked[:k] if x in s["y_set"])
        per.append(h)
        b = by_bo[s["bo"]]
        b[0] += h
        b[1] += len(ranked[:k])
    return per, by_bo


def score_sim(samples, rule):
    samples = [s for s in samples if s["y_set"]]   # 同上
    per, by_bo = [], collections.defaultdict(lambda: [0, 0])
    for s in samples:
        k = max(1, min(int(s["bo"] or 3), 5))
        pred = s["sims"][rule][:k]
        h = sum(1 for x in pred if x in s["y_set"])
        per.append(h)
        b = by_bo[s["bo"]]
        b[0] += h
        b[1] += len(pred)
    return per, by_bo


def boot(diff, trials=20000, seed=20260719):
    n = len(diff)
    if not n:
        return None
    rng = random.Random(seed)
    mean = sum(diff) / n
    bs = sorted(sum(diff[rng.randrange(n)] for _ in range(n)) / n
                for _ in range(trials))
    p = 2.0 * min(sum(1 for x in bs if x <= 0),
                  sum(1 for x in bs if x >= 0)) / trials
    return {"mean": mean, "ci": [bs[int(.025 * trials)], bs[int(.975 * trials)]],
            "p": p, "wins": sum(1 for x in diff if x > 0),
            "losses": sum(1 for x in diff if x < 0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="exp_map_shrink_results.json")
    a = ap.parse_args()
    buckets, truths, veto, pool_of = load()
    res = {}

    def run(shrink, kind, rule=None):
        br = rule if (kind == "model" and rule) else "hist"
        hist = VM.MapHistory(half_life=VM.HALF_LIFE)
        tr = build(buckets["train"], truths, veto, hist, pool_of, shrink, br)
        va = build(buckets["valid"], truths, veto, hist, pool_of, shrink, br)
        te = build(buckets["test"], truths, veto, hist, pool_of, shrink, br)
        if kind == "model":
            bv, bc = -1, None
            for C in CS:
                per_v, _ = score_model(va, C)
                tot = sum(per_v)
                if tot > bv:
                    bv, bc = tot, C
            per_te, by_bo = score_model(te, bc)
            return sum(per_te) * 5, per_te, by_bo, bc, bv * 5
        per_va, _ = score_sim(va, rule)
        per_te, by_bo = score_sim(te, rule)
        return sum(per_te) * 5, per_te, by_bo, rule, sum(per_va) * 5

    print("=" * 96)
    print("分类器（特征集 = 生产集 + 新增候选）")
    print("%-22s %8s %8s   %-34s %s"
          % ("收缩口径", "valid", "test", "配对自助 vs hist（test 逐场命中）", "bo3 / bo5"))
    print("=" * 96)
    base = None
    base_per = None
    for shrink in ("hist", "teamavg", "raw"):
        for rule in ("hist", "averse"):
            pts, per, by_bo, C, vpts = run(shrink, "model", rule)
            if base_per is None:                 # ⚠ 基准只认第一次，别每轮重置
                base_per = per
            b = boot([x - y for x, y in zip(per, base_per)])
            res["model:%s:%s" % (shrink, rule)] = {
                "test": pts, "valid": vpts, "C": C, "by_bo": dict(by_bo), "boot": b}
            print("%-12s %-8s %8d %8d   %-42s %s"
                  % (shrink, rule, vpts, pts,
                     ("每场 %+.3f 95%%CI[%+.2f,%+.2f] p=%.3f %d↑/%d↓"
                      % (b["mean"], b["ci"][0], b["ci"][1], b["p"], b["wins"],
                         b["losses"])) if b else "—",
                     " / ".join("%s %d/%d" % (bo, by_bo[bo][0], by_bo[bo][1])
                                for bo in sorted(by_bo))))

    print("\n" + "=" * 96)
    print("纯 simulate_veto（不经分类器；无 C 选型，最干净）")
    print("%-22s %-10s %8s   %s" % ("收缩口径", "ban规则", "test", "bo3 / bo5"))
    print("=" * 96)
    simbase = None
    for shrink in ("hist", "teamavg", "raw"):
        for rule in ("hist", "averse"):
            pts, per, by_bo, _, vpts = run(shrink, "sim", rule)
            if simbase is None and shrink == "hist" and rule == "hist":
                simbase = per
            b = boot([x - y for x, y in zip(per, simbase)]) if simbase else None
            res["sim:%s:%s" % (shrink, rule)] = {
                "test": pts, "valid": vpts, "by_bo": dict(by_bo), "boot": b}
            print("%-22s %-10s %8d   %-28s %s"
                  % (shrink, rule, pts,
                     ("每场 %+.3f p=%.3f" % (b["mean"], b["p"])) if b else "—",
                     " / ".join("%s %d/%d" % (bo, by_bo[bo][0], by_bo[bo][1])
                                for bo in sorted(by_bo))))

    with io.open(os.path.join(HERE, a.out), "w", encoding="utf-8",
                 newline="\n") as f:
        f.write(json.dumps(res, ensure_ascii=False, indent=2) + "\n")
    print("\n已写 %s" % a.out)


if __name__ == "__main__":
    sys.exit(main())