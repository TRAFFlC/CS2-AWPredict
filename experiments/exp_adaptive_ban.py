"""exp_adaptive_ban.py —— adaptive ban 步数：配对检验「结构修复是否值得」。

## 背景

`simulate_veto` 对 bo>=5 一律排 8 个动作（2 ban + 4 pick + 2 ban），
但小图池赛事喂不满 8 步。实测（池 = 7，bo5）：**只出 4 张，少 1 张**。
而选图按集合判分、集合大小须等于 bo ⇒ 那一格白丢。

`adaptive=True` 把 ban 数压到 `clamp(len(pool) - bo, 0, 标准上限)`，
保证 `len(picks) == bo` 恒成立。**结构上这是修对了**。

## 但代价不小

它同时改掉 bo3 + 大图池的 sim_features（原来返回「2 pick + 全部剩余」=
远超 bo 张，现在恰好 bo 张）。粗跑 valid +10 / test −15（净值 −5），
不足以证明收益。

## 本脚本

同一模型、同一批场次、同一评估器，只切 `adaptive`，**配对**比每场命中数，
再做自助法给 p 值。valid/test 分开报，也分开报 bo3/bo5（结构修复主要影响 bo5）。

用法：python -X utf8 exp_adaptive_ban.py
"""
import collections
import io
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pool_experiment as PE
import split as SPLIT
import veto_model as VM


def build():
    rows = SPLIT.load(VM.MATCHES)
    ex = SPLIT.load_exclude(VM.EXCLUDE) if os.path.exists(VM.EXCLUDE) else None
    buckets, meta = SPLIT.split(rows, purge=True, exclude=ex)
    truths = {t["id"]: t for t in
              (json.loads(l) for l in io.open(VM.TRUTH, encoding="utf-8") if l.strip())}
    eras, by_pool = PE.load_pool_eras()
    tl = json.load(io.open(PE.TIMELINE, encoding="utf-8"))

    def pool_of(r):
        t = truths.get(r["id"]) or {}
        if t.get("pool"):
            return list(t["pool"])
        return list(PE.era_and_pool(r["date"], [], eras, by_pool, tl)[1])

    return buckets, meta, truths, pool_of


def per_match(model, samples):
    """每场的命中数（用于配对比较）。"""
    out = []
    for s in samples:
        k = VM.k_of(s)
        picked, _ = VM.predict_set(model, s, k)
        truth = set(f["map"] for f, v in zip(s["rows"], s["y"]) if v)
        out.append(sum(1 for m in picked if m in truth))
    return out


def boot(diff, trials=20000, seed=20260719):
    n = len(diff)
    if not n:
        return 1.0
    rng = random.Random(seed)
    mean = sum(diff) / n
    bs = sorted(sum(diff[rng.randrange(n)] for _ in range(n)) / n for _ in range(trials))
    p = 2.0 * min(sum(1 for x in bs if x <= 0), sum(1 for x in bs if x >= 0)) / trials
    return p


def main():
    buckets, meta, truths, pool_of = build()
    veto = VM.load_veto()
    res = {}
    for ad in (False, True):
        # 每个臂都要**从零重建** hist：特征含 sim_*，adaptive 会改变它们，
        # 复用同一个 hist 会让两臂看到不同的历史推进，破坏配对。
        hist = VM.MapHistory(half_life=VM.HALF_LIFE)
        orig = VM.MapHistory.features_for

        def patched(self, *a, _o=orig, _ad=ad, **kw):
            kw.setdefault("adaptive", _ad)
            return _o(self, *a, **kw)

        VM.MapHistory.features_for = patched
        try:
            tr = VM.build_dataset(buckets["train"], truths, veto, hist, pool_of)
            va = VM.build_dataset(buckets["valid"], truths, veto, hist, pool_of)
            te = VM.build_dataset(buckets["test"], truths, veto, hist, pool_of)
        finally:
            VM.MapHistory.features_for = orig
        best = None
        for C in (0.03, 0.1, 0.3, 1.0, 3.0):
            m = VM.train(tr, C=C)
            ev = VM.evaluate(m, va)
            if best is None or ev["points"] > best[1]["points"]:
                best = (C, ev, m)
        res[ad] = {"C": best[0], "valid": va, "test": te, "model": best[2],
                   "ev_valid": best[1]}
        print("adaptive=%-5s 选定 C=%-5s valid=%d 分 (%.2f%%)"
              % (ad, best[0], best[1]["points"], 100 * best[1]["hit_rate"]), flush=True)

    print()
    print("=" * 76)
    print("配对比较：adaptive=True 减 adaptive=False（每场命中数之差）")
    print("=" * 76)
    print("  %-6s %-10s %-10s %-10s %-9s %s"
          % ("桶", "旧", "新", "每场Δ", "p", "注"))
    for b in ("valid", "test"):
        old = per_match(res[False]["model"], res[False][b])
        new = per_match(res[True]["model"], res[True][b])
        d = [x - y for x, y in zip(new, old)]
        print("  %-6s %-10d %-10d %+10.4f %-9.3f %s"
              % (b, sum(old), sum(new), sum(d) / len(d), boot(d),
                 "配对 n=%d" % len(d)))

    print()
    print("  按 bo 拆开（结构修复主要动 bo5 / 小图池）：")
    for b in ("valid", "test"):
        sam = res[True][b]
        by = collections.defaultdict(list)
        old = per_match(res[False]["model"], res[False][b])
        new = per_match(res[True]["model"], res[True][b])
        for i, s in enumerate(sam):
            by[int(s.get("bo") or 3)].append(new[i] - old[i])
        for bo in sorted(by):
            v = by[bo]
            print("    %s bo%-3d n=%-4d Δ=%+.4f" % (b, bo, len(v), sum(v) / len(v)))
    print()
    print("  读法：Δ>0 = adaptive 更好。**p 大 ⇒ 打平**，")
    print("  则应按「结构正确性」而非「实测收益」决策：")
    print("  旧实现在 7 图 bo5 下只出 4 张，是**确定的错误**（集合大小≠bo）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
