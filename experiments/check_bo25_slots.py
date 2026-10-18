"""check_bo25_slots.py —— bo25 真值为空却在吃 test 分母，量化影响。

## 事实

truth_maps_B.jsonl 里 bo=25 的 4 场，**真值自己就标了可疑**：

    "bo": 25, "bo_suspect": true, "suspicious": true,
    "maps": [],            ← 真值图集是空的
    "pool": null,
    "notes": ["未知 bo=25，无法判定应有图数"]

而 veto_model.evaluate()（veto_model.py:683-697）对 suspicious / 空真值
**没有任何过滤**：

    truth = set(f["map"] for f, v in zip(s["rows"], s["y"]) if v)   # => set()
    hits  += sum(1 for m in picked if m in truth)                    # => 恒 0
    slots += k                                                       # k=5，照算

⇒ 每场 bo25 白送 5 个**必然 miss** 的槽位进分母。

这不是「模型不行」的证据，是**数据缺失被当成了反例**。不可评分的场次必须出分母，
否则 test 命中率被系统性压低。

## 本脚本只做两件事

1. 列出所有「真值为空」的场次（不止 bo25），确认范围
2. 给出剔除前后的 test 命中率对比

用法：python -X utf8 check_bo25_slots.py
"""
import collections
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "../data")  # # 仓库布局：数据与代码分目录（Alpha 1.1.1）
sys.path.insert(0, HERE)
import split as SPLIT
import veto_model as VM

TRUTH = os.path.join(DATA, "truth_maps_B.jsonl")


def main():
    rows = SPLIT.load(VM.MATCHES)
    # ⚠ 必须和 veto_model.run() 完全一致：它用的是 VM.EXCLUDE，不是 purge manifest
    exclude = SPLIT.load_exclude(VM.EXCLUDE) if os.path.exists(VM.EXCLUDE) else None
    buckets, meta = SPLIT.split(rows, purge=True, exclude=exclude)
    truths = {t["id"]: t for t in
              (json.loads(l) for l in io.open(TRUTH, encoding="utf-8") if l.strip())}

    print("=" * 68)
    print("① 真值为空 / 被标可疑的场次（全集，不分桶）")
    print("=" * 68)
    bad = [t for t in truths.values() if not (t.get("maps") or [])]
    by_bo = collections.Counter(int(t.get("bo") or 0) for t in bad)
    print("真值为空的场次：%d / %d" % (len(bad), len(truths)))
    print("  按 bo 分布：%s" % dict(sorted(by_bo.items())))
    print("  按 suspicious：%s" % dict(collections.Counter(
        bool(t.get("suspicious")) for t in bad)))
    for t in sorted(bad, key=lambda x: (x.get("date") or "", x["id"])):
        print("   %-9s %s  bo=%-3s %-22s  %s" % (
            t["id"], t.get("date"), t.get("bo"),
            (t.get("teamA") or "?") + " vs " + (t.get("teamB") or "?"),
            "; ".join(t.get("notes") or [])[:40]))
    print()

    # ---- 完全复用 veto_model.run() 的样本构建路径 ----
    import pool_experiment as PE
    eras, by_pool = PE.load_pool_eras()
    tl = json.load(io.open(PE.TIMELINE, encoding="utf-8"))
    veto = VM.load_veto()

    def pool_of(r):
        t = truths.get(r["id"]) or {}
        vp = t.get("pool")
        if vp:
            return list(vp)
        _era, pool = PE.era_and_pool(r["date"], [], eras, by_pool, tl)
        return list(pool)

    hist = VM.MapHistory(half_life=VM.HALF_LIFE)
    train_samples = VM.build_dataset(buckets["train"], truths, veto, hist, pool_of)
    valid_samples = VM.build_dataset(buckets["valid"], truths, veto, hist, pool_of)
    test_samples = VM.build_dataset(buckets["test"], truths, veto, hist, pool_of)

    # C 选取照抄 run()（valid 上按 points 选最大）
    best = None
    for C in (0.03, 0.1, 0.3, 1.0, 3.0):
        m = VM.train(train_samples, C=C)
        ev = VM.evaluate(m, valid_samples)
        if best is None or ev["points"] > best[1]["points"]:
            best = (C, ev, m)
    C, ev_valid, model = best
    print("  （C 选取照抄生产：valid 上按 points 取最大 ⇒ C=%s）" % C)
    print()
    print("  %-6s %-8s %-10s %-8s %-10s %-10s %s" % (
        "桶", "场次", "槽位", "命中", "命中率", "剔除后", "差"))
    for b, samples in (("valid", valid_samples), ("test", test_samples)):
        full = VM.evaluate(model, samples)
        keep = [s for s in samples if any(v for v in s["y"])]   # 真值非空才可评分
        cut = VM.evaluate(model, keep)
        print("  %-6s %-8d %-10d %-8d %-10s %-10s %+.2fpp" % (
            b, full["n"], full["slots"], full["hits"],
            "%.2f%%" % (100 * full["hit_rate"]),
            "%.2f%%" % (100 * cut["hit_rate"]),
            100 * (cut["hit_rate"] - full["hit_rate"])))
    print()
    print("  被剔除的场次里**没有一张能命中**（真值为空），所以剔除只抬高命中率、")
    print("  不改变任何预测，也不改变 C 的选取（valid 里没有空真值场次）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
