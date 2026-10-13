"""MVP 天花板：v3 笔记写「条件命中上限 ~30~35%，已 32.9%」——本脚本**实测**这个上界，
不引用笔记，并说清它到底是怎么来的。

需要区分的两个不同的量（v3 把它们混在一个「天花板」里）：
  (1) **候选集上界**：真实 MVP 落在模型预测的候选阵容里的比例。
      若这个接近 95~97%，那"候选不够"就不是瓶颈。
  (2) **恰好命中上界**：即使候选集完美，也只能靠「猜中该队的第一名」。
      这受两个因素夹击：阵容时点错配（模型用赛前 240 天的阵容，真实 MVP 是当场的那支）
      与 rating 差距极小（中位 margin 仅 0.14，38.7% 的场次 ≤0.10）。
"""
import collections, io, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import model_v2 as M
import split as SPLIT


def main():
    prep = M.prepare(verbose=False)
    data = prep["data"]
    for bucket in ("train", "valid", "test"):
        ss = data[bucket]
        cand_total = 0
        truth_is_cand = 0
        truth_is_first = 0
        ncand = []
        for s in ss:
            pr = s.get("player_rows") or {}
            mt = s.get("mvp_truth") or ""
            w = s.get("winner")
            rows = pr.get(w) or []
            if not rows or not mt:
                continue
            cand_total += 1
            ncand.append(len(rows))
            names = {r["player"] for r in rows}
            truth_is_cand += int(mt in names)
            truth_is_first += int(any(r["player"] == mt and r.get("label") for r in rows))
        if not cand_total:
            continue
        print("== %s (n=%d) ==" % (bucket, cand_total))
        print("   平均候选人数 = %.1f" % (sum(ncand) / len(ncand)))
        print("   真实 MVP 落在**胜方候选集**里 = %d/%d = %.2f%%   ← (1) 候选集上界"
              % (truth_is_cand, cand_total, 100 * truth_is_cand / cand_total))
        print("   真实 MVP 同时是候选里 label=1 的那个 = %d/%d = %.2f%%   ← (2) 恰好命中上界"
              % (truth_is_first, cand_total, 100 * truth_is_first / cand_total))

    print("\n== rating margin（胜方候选里第一 − 第二）分布 ==")
    mm = []
    for s in data["test"]:
        pr = s.get("player_rows") or {}
        rows = pr.get(s.get("winner")) or []
        cand = [(r["player"], r.get("raw")) for r in rows if r.get("raw") is not None]
        if len(cand) < 2:
            continue
        cand.sort(key=lambda x: -x[1])
        mm.append(cand[0][1] - cand[1][1])
    if mm:
        mm.sort()
        n = len(mm)
        print("   n=%d  中位=%.3f  均值=%.3f  ≤0.10 占 %.1f%%  ≤0.20 占 %.1f%%"
              % (n, mm[n // 2], sum(mm) / n,
                 100 * sum(1 for x in mm if x <= 0.10) / n,
                 100 * sum(1 for x in mm if x <= 0.20) / n))
    return 0


if __name__ == "__main__":
    sys.exit(main())
