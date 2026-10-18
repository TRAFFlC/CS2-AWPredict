"""check_bo5_power.py —— bo5 改动到底有没有统计功效？

## 为什么需要这个（2026-07-19 的直接起因）

主理人报了一 bo5 决赛（BetBoom vs FURIA）选图丢 5 分，我改了三处（先手特征 /
厌恶度 ban / 显式 ban 模拟）并做了对照实验。结论卡在这里：

    分类器侧：hist/averse +30 分 p=0.214 ；raw/hist +50 分 p=0.069
    纯规则侧：hist/averse +150 分 **p=0.017**

分类器侧不显著的原因不是「改动没用」，而是 **bo5 的样本量根本不够**：

    各槽位命中 {0: 208, 1: 179, 2: 163, 3: **2**, 4: **1**}

槽位 3/4 只有个位数命中 —— 而 bo5 恰好落在后两个槽位。
**任何针对 bo5 的改动，在当前语料上都无法证伪**（证不实、也证不伪）。

## 判据

bo5 槽位数 `n5`。粗略地，配对自助要能分辨「每场 +0.05 命中」需要
`n5 ≳ 100`（本项目实测的效应量）。低于此线 ⇒ 只报数、不下结论。

用法：
    python -X utf8 check_bo5_power.py            # 报当前功效
    python -X utf8 check_bo5_power.py --target 100
"""
import argparse
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

MATCHES = os.path.join(DATA, "csapi_matches_all.jsonl")
TRUTH = os.path.join(DATA, "truth_maps_B.jsonl")
EXCLUDE = os.path.join(HERE, "split_manifest__purge_exclude.txt")
DEFAULT_TARGET = 100


def _slots5(rows, truths):
    """该桶里 bo==5 的槽位数（严格只算 bo=5，不含 bo25）。"""
    n = 0
    for r in rows:
        bo = (truths.get(r["id"]) or {}).get("bo") or r.get("best_of")
        if int(bo or 3) == 5:
            n += 5
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=DEFAULT_TARGET)
    a = ap.parse_args()
    rows = SPLIT.load(MATCHES)
    exclude = SPLIT.load_exclude(EXCLUDE) if os.path.exists(EXCLUDE) else None
    buckets, meta = SPLIT.split(rows, purge=True, exclude=exclude)
    truths = {t["id"]: t for t in
              (json.loads(l) for l in io.open(TRUTH, encoding="utf-8") if l.strip())}

    print("切分：%s" % {k: len(v) for k, v in buckets.items()})
    print()
    tot = collections.Counter()
    for b in ("valid", "test"):
        c = collections.Counter()
        for r in buckets[b]:
            bo = (truths.get(r["id"]) or {}).get("bo") or r.get("best_of")
            c["场次_bo" + str(bo)] += 1
            c["槽位_bo" + str(bo)] += max(1, min(int(bo or 3), 5))
        print("[%s] %s" % (b, {k: v for k, v in sorted(c.items(), key=str)}))
        tot.update(c)
    print()
    v5, t5 = _slots5(buckets["valid"], truths), _slots5(buckets["test"], truths)
    print("=" * 62)
    print("bo5 槽位数（valid + test）= %d    目标 = %d" % (v5 + t5, a.target))
    print("  valid %d + test %d（严格只算 bo==5）" % (v5, t5))
    print("=" * 62)
    # ⚠ bo25 是**另一码事**，绝不能混进 bo5 计数
    if tot.get("场次_bo25"):
        print("⚠ 另有 bo25 %d 场（占 test 槽位 %d）——major 长赛制，"
              % (tot["场次_bo25"], tot["槽位_bo25"]))
        print("   **不计入 bo5**。且它们真值 maps=[] 属不可评分，")
        print("   白吃 test 分母 —— 见 check_bo25_slots.py")
    n5 = v5 + t5
    if n5 >= a.target:
        print("✅ **有功效**：bo5 上的改动现在可以被证伪/证实，可以重跑")
        print("   exp_map_firstban.py / exp_map_shrink.py")
        return 0
    need = a.target - n5
    print("⚠ **无功效**：bo5 改动无法被证伪，**不要据此转正或否决**")
    print("   还差 %d 个槽位。bo5 每场贡献 5 个 ⇒ 大约还需 %d 场 bo5。"
          % (need, -(-need // 5)))
    print("   在到线之前：① 只在 bo3（n=%d）上测；② 对 bo5 的改动只报「机制是否修好"
          "了主理人报的那场」，不报总体收益。" % tot.get("槽位_bo3", 0))
    return 0


if __name__ == "__main__":
    sys.exit(main())