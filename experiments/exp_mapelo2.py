"""实验（补上设计缺陷）：图级 Elo **加进**现有胜方特征组，能否抬 CV？

上一版只测了 mapelo 单独用（CV 0.591 < 主模型 0.6233），那不回答问题。
本版把 mapelo 的四个值**注入现有样本的 feats 字典**，然后用 `model_v2.cv_acc`
（与 v3 完全同口径：train 扩展窗口、28 组 × 5 个 C、test 从不参与）做选型。

对齐保证：`build()` 与本脚本的 collect() 都按 (date, id) 排序、都过滤 bo∈{1,3,5}、
都逐日两趟（当日互不可见），所以第 i 个样本严格对应第 i 场。
"""
import io, json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import model_v2 as M
import veto_model as VM
import split as SPLIT
import pool_experiment as PE
import exp_mapelo as EX          # 复用 MapElo 与 collect 的驱动循环

NEW = ["d_mapelo_mean", "d_mapelo_max", "d_mapelo_min", "d_mapelo_spread"]


def main():
    print("== 1) prepare() 拿到与 v3 完全同口径的样本 ==")
    prep = M.prepare(verbose=False)
    data = prep["data"]
    print("   train=%d valid=%d test=%d  选中的胜方组=%s (C=%s)"
          % (len(data["train"]), len(data["valid"]), len(data["test"]),
             prep["win_name"], prep["C"]))

    print("\n== 2) 用同一驱动循环补算图级 Elo 特征 ==")
    rows = SPLIT.load(M.MATCHES)
    exc = SPLIT.load_exclude(M.EXCLUDE) if os.path.exists(M.EXCLUDE) else None
    buckets, _ = SPLIT.split(rows, purge=True, exclude=exc)
    truths = {t["id"]: t for t in M.load_jsonl(M.TRUTH)}
    veto = VM.load_veto()
    stats = M.load_stats()
    eras, by_pool = PE.load_pool_eras()
    tl = json.load(io.open(PE.TIMELINE, encoding="utf-8"))
    s, h = M.State(), VM.MapHistory()
    tr = EX.collect(s, h, buckets["train"], truths, veto, eras, by_pool, tl, stats)
    va = EX.collect(s, h, buckets["valid"], truths, veto, eras, by_pool, tl, stats)
    print("   补到 train=%d valid=%d（与样本数一致=%s/%s）"
          % (len(tr), len(va), len(tr) == len(data["train"]), len(va) == len(data["valid"])))
    assert len(tr) == len(data["train"]) and len(va) == len(data["valid"]), "对齐失败"

    for bucket, extra in (("train", tr), ("valid", va)):
        for smp, ex in zip(data[bucket], extra):
            for k in NEW:
                smp["feats"][k] = ex[k]

    print("\n== 3) 单变量方向命中率（valid，与 d_elo 同口径对照）==")
    for k in ["d_elo", "d_elom", "d_rank", "d_h2h"] + NEW:
        if k not in data["valid"][0]["feats"]:
            continue
        pos = sum(1 for t in data["valid"]
                  if (t["feats"][k] > 0) == (t["y"] == 1))
        print("   %-20s %.2f%%" % (k, 100 * max(pos, len(data["valid"]) - pos) / len(data["valid"])))

    print("\n== 4) 选型：现有 28 组 vs 追加图级 Elo 的变体 ==")
    base_sets = list(M.WIN_FEATURE_SETS)
    cands = [("现生产 %s" % prep["win_name"], list(prep["winfeats"]))]
    sel = list(prep["winfeats"])
    for k in NEW:
        cands.append(("现生产 + %s" % k, sel + [k]))
    cands.append(("现生产 + min&max", sel + ["d_mapelo_min", "d_mapelo_max"]))
    cands.append(("现生产 + min&spread", sel + ["d_mapelo_min", "d_mapelo_spread"]))
    cands.append(("现生产 + 全部四列", sel + NEW))
    # 也试「从零开始」的图级-only 组，防止只在现有组上叠
    cands.append(("仅 d_elo + d_mapelo_min", ["d_elo", "d_mapelo_min"]))
    cands.append(("仅 d_elo + 全部四列", ["d_elo"] + NEW))

    n_val = int(len(data["train"]) * (M.CV_FOLDS[-1] - M.CV_FOLDS[0]))
    results = []
    print("   %-38s %s" % ("特征组", "  ".join("C=%-5s" % c for c in (0.02, 0.05, 0.1, 0.3, 1.0))))
    for nm, f in cands:
        row = []
        for C in (0.02, 0.05, 0.1, 0.3, 1.0):
            row.append(M.cv_acc(data["train"], f, C))
        best = max(row)
        results.append((nm, f, row, best))
        print("   %-38s %s   best=%.4f" % (nm, "  ".join("%.4f" % x for x in row), best))

    print("\n== 5) 判据：与 v3 同口径 —— train 扩展窗口 CV，1SE 容差内取最简 ==")
    n_val = int(len(data["train"]) * (M.CV_FOLDS[-1] - M.CV_FOLDS[0]))
    base_best = results[0][3]
    # 累计验证条数（与 v3 报的 n≈1,253 同量级）
    se = math.sqrt(max(base_best * (1 - base_best), 1e-9) / max(1, n_val))
    print("   参照（现生产）CV=%.4f   1SE=%.4f（n_val≈%d）" % (base_best, se, n_val))
    win = [r for r in results[1:] if r[3] > base_best + se]
    if win:
        print("   ★ 超过 1SE 容差的组：")
        for nm, f, row, b in win:
            print("     %-38s CV=%.4f（+%+.4f）特征=%s" % (nm, b, b - base_best, f))
    else:
        print("   ⇒ **没有任何变体超过 1SE 容差** ⇒ 按协议不采纳，test 不报。")
        top = max(results[1:], key=lambda r: r[3])
        print("      最接近的是 %s：CV=%.4f（%+.4f，%s）"
              % (top[0], top[3], top[3] - base_best,
                 "仍在 1SE 内" if top[3] <= base_best + se else "超了"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
