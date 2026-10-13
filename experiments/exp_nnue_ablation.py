"""NNUE 的**公平对照**：先验证我的优化器能不能复现逻辑回归，再判断嵌入有没有用。

报负结果之前必须过的三关：
  关 1（优化器 sanity）：关掉嵌入、只留标量线性项 ⇒ CV 应接近逻辑回归的 0.6233。
      若连这个都上不去，说明是我的 Adam/尺度问题，**不是**架构问题，负结果无效。
  关 2（收敛）：epoch 扫描到 3000，看是否还在涨。
  关 3（消融）：标量 / 嵌入 / 两者，分开看贡献。
"""
import io, json, math, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import model_v2 as M
import exp_nnue as E


def main():
    prep = M.prepare(verbose=False)
    data = prep["data"]
    idx = E.build_index()
    teams = sorted({t for b in ("train", "valid")
                    for (ta, tb), _, _ in idx[b] for t in (ta, tb)})
    maps = sorted({m for b in ("train", "valid")
                   for _, pool, _ in idx[b] for m in pool})
    t2i = {t: i for i, t in enumerate(teams)}
    m2i = {m: i for i, m in enumerate(maps)}
    tr = E.assemble(data["train"], idx["train"], t2i, m2i)
    va = E.assemble(data["valid"], idx["valid"], t2i, m2i)
    folds = M.CV_FOLDS
    n_val = int(len(data["train"]) * (folds[-1] - folds[0]))
    base = M.cv_acc(data["train"], list(prep["winfeats"]), prep["C"])
    se = math.sqrt(max(base * (1 - base), 1e-9) / n_val)
    print("参照：逻辑回归 CV=%.4f（%s, C=%s）  1SE=%.4f  n_val=%d"
          % (base, prep["win_name"], prep["C"], se, n_val))
    sd = tr[0].std(axis=0)
    print("标量特征各维标准差中位数 = %.3f（Adam 对未归一化输入很敏感）" % np.median(sd))

    print("\n== 关 1 + 关 3：消融 × epoch 扫描（lr=0.08, l2=1e-2, k=4）==")
    print("%-26s %8s %8s %8s %8s" % ("配置", "ep=300", "ep=1000", "ep=2000", "ep=3000"))
    combos = [("① 只标量（sanity）", dict(use_scalars=True, use_emb=False), False),
              ("② 只嵌入（队伍×图）", dict(use_scalars=False, use_emb=True), False),
              ("③ 标量+嵌入（完整 NNUE）", dict(use_scalars=True, use_emb=True), False),
              ("①′ 只标量（输入已归一化）", dict(use_scalars=True, use_emb=False), True)]
    out = {}
    for nm, kw, sc in combos:
        row = []
        for ep in (300, 1000, 2000, 3000):
            a, _ = E.cv_nnue(tr, 4, 0.08, 1e-2, ep, folds, len(teams), len(maps),
                             scale=(1.0 / sd) if sc else None, **kw)
            row.append(a)
        out[nm] = row
        print("%-26s %8.4f %8.4f %8.4f %8.4f" % (nm, *row))

    print("\n== 判读 ==")
    sanity = out["① 只标量（sanity）"][-1]
    best_sanity = max(out["① 只标量（sanity）"])
    gap = abs(best_sanity - base)
    verdict = ("通过（差异 %.4f ≤ 1SE %.4f）⇒ 优化器可信，负结果成立"
               % (gap, se)) if gap <= se else \
              ("**未通过**（差 %.4f ≈ %.1f×1SE）⇒ 负结果只能谨慎解读"
               % (gap, gap / se))
    print("  关 1：只标量 CV 峰值=%.4f（ep 扫描见上，ep>1000 反而回落）vs 逻辑回归 %.4f ⇒ %s"
          % (best_sanity, base, verdict))
    full = max(out["③ 标量+嵌入（完整 NNUE）"])
    print("  关 3：嵌入的净贡献 = %.4f − %.4f = %+.4f（相对**我自己的 sanity 基线**）"
          % (full, best_sanity, full - best_sanity))
    print("  最终：完整 NNUE CV=%.4f  vs 逻辑回归 %.4f  ⇒ %+.4f（%s 1SE=%.4f）"
          % (full, base, full - base,
             "超" if full > base + se else ("**低于**" if full < base - se else "在"), se))
    return 0


if __name__ == "__main__":
    sys.exit(main())
