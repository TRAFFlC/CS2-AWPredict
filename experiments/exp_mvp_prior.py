"""MVP 先验检验（扩到 109 场真实标签 + 严格时序 + 留出验证）。

主理人 2026-12 的判断：
> 「我一般倾向于认为是 rating 最高。我说的是**个人 rating**，不是本场 rating。
>   mvp 是由 rating 决定的，所以个人平均 rating 更高的一定在一场比赛中成为
>   mvp 的概率更高。」

★ 与既有模型的区别（这是本脚本存在的理由）：
  `model_v2.mvp_label()` 的标签就是「胜方队内 rating 最高者」，而
  `decide.mvp_probs()` 也按 rating 排 ⇒ **自证**，语料 31.2% 不是预测能力。
  本脚本只用**主理人提供的真实 MVP**，且 rating 只取**赛前**可见值。

⚠ 时序正确性：语料 roster 只到 2026-09-05，而赛事跨到 09-27。
   所以「赛前 rating」只能用**截至 09-05 的最后已知 roster** 作代理。
   这是近似，对所有规则**同等**，横向可比；但不能说它是严格的 point-in-time。

★ 留出验证：前 70% 选规则，后 30% 只报一次（避免在 109 场上调规则再自证）。
"""
import collections
import io
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "../data")  # # 仓库布局：数据与代码分目录（Alpha 1.1.0）
sys.path.insert(0, HERE)

EPL = (r"E:\world\another\data\Mimo\MiMo-V2.6-Flash\2026-10-12"
       r"\live_tournament_results.jsonl")
FILES = ["fissure_2026-09-08_results.json", "fissure_2026-09-09_results.json",
         "fissure_2026-09-10_results.json", "fissure_2026-09-11_results.json",
         "fissure_2026-09-12_results.json", "fissure_2026-09-13_results.json",
         "fissure_2026-09-20-bo5_results.json",
         "fissure_2026-09-17_results.json", "fissure_2026-09-18_results.json",
         "fissure_2026-09-19_results.json",
         "fissure_cold1_results.json", "fissure_cold2_results.json"]

DATES = {"EPL": {}, "fissure_2026-09-08_results.json": "2026-09-08",
         "fissure_2026-09-09_results.json": "2026-09-09",
         "fissure_2026-09-10_results.json": "2026-09-10",
         "fissure_2026-09-11_results.json": "2026-09-11",
         "fissure_2026-09-12_results.json": "2026-09-12",
         "fissure_2026-09-13_results.json": "2026-09-13",
         "fissure_2026-09-20-bo5_results.json": "2026-09-20",
         "fissure_2026-09-17_results.json": "2026-09-17",
         "fissure_2026-09-18_results.json": "2026-09-18",
         "fissure_2026-09-19_results.json": "2026-09-19",
         "fissure_cold1_results.json": "2026-09-21",
         "fissure_cold2_results.json": "2026-09-22"}


def load_live():
    out = []
    for l in io.open(EPL, encoding="utf-8"):
        if not l.strip():
            continue
        r = json.loads(l)
        if r.get("mvp") and r.get("winner"):
            out.append((r["date"], r["winner"], r["mvp"]))
    for f in FILES:
        p = os.path.join(HERE, f)
        if not os.path.exists(p):
            continue
        d = DATES.get(f)
        for i, x in enumerate(json.load(io.open(p, encoding="utf-8"))):
            if x.get("mvp"):
                out.append(("%s#%02d" % (d, i), x["winner"], x["mvp"]))
    return out


def main():
    live = load_live()
    recs = [json.loads(l) for l in
            io.open(os.path.join(HERE, "records_v1.jsonl"), encoding="utf-8")
            if l.strip()]
    recs.sort(key=lambda r: r["date"])
    last = {}
    for r in recs:
        for team, pl in (r.get("rosters") or {}).items():
            if pl:
                last[team] = pl

    rows = []
    prev = {}
    for key, w, mvp in live:
        pool = [(p["name"], float(p.get("rating") or 0))
                for p in (last.get(w) or []) if p.get("name")]
        if len(pool) < 2:
            continue
        n = len(pool)
        r_top = max(pool, key=lambda x: x[1])[0]
        r_rand = pool[hash(key) % n][0]
        pm = prev.get(w)
        r_prev = pm if pm in [p[0] for p in pool] else None
        prev[w] = mvp
        rows.append({"k": key, "mvp": mvp, "top": r_top, "rand": r_rand,
                     "prev": r_prev, "n": n})

    k = int(len(rows) * 0.70)
    tr, te = rows[:k], rows[k:]
    print("可评估 %d 场（有 roster）   训练段 %d / 留出段 %d\n"
          % (len(rows), len(tr), len(te)))

    def ev(rs, tag, key):
        n = sum(1 for r in rs if key(r))
        if not n:
            return
        acc = sum(1 for r in rs if key(r) == r["mvp"]) / len(rs)
        se = math.sqrt(acc * (1 - acc) / len(rs))
        print("  %-26s %3d/%-3d = %5.1f%%  (1SE %.1fpp, 随机 20%% ⇒ %+.1fσ)"
              % (tag, sum(1 for r in rs if key(r) == r["mvp"]), len(rs),
                 100 * acc, 100 * se, (acc - 0.2) / se if se else 0))

    print("== 训练段（用于比较规则）==")
    ev(tr, "R0 随机（固定）", lambda r: r["rand"])
    ev(tr, "R1 队内 rating 最高", lambda r: r["top"])
    rn = [r for r in tr if r["prev"]]
    if rn:
        acc = sum(1 for r in rn if r["prev"] == r["mvp"]) / len(rn)
        print("  %-26s %3d/%-3d = %5.1f%%  （仅 %d 场有上一场记录）"
              % ("R3 上一场 MVP", sum(1 for r in rn if r["prev"] == r["mvp"]),
                 len(rn), 100 * acc, len(rn)))

    print("\n== 留出段（只报一次）==")
    ev(te, "R0 随机（固定）", lambda r: r["rand"])
    ev(te, "R1 队内 rating 最高", lambda r: r["top"])
    rn = [r for r in te if r["prev"]]
    if rn:
        acc = sum(1 for r in rn if r["prev"] == r["mvp"]) / len(rn)
        print("  %-26s %3d/%-3d = %5.1f%%  （仅 %d 场）"
              % ("R3 上一场 MVP", sum(1 for r in rn if r["prev"] == r["mvp"]),
                 len(rn), 100 * acc, len(rn)))

    print("\n== 全体（参考）==")
    ev(rows, "R0 随机（固定）", lambda r: r["rand"])
    ev(rows, "R1 队内 rating 最高", lambda r: r["top"])
    json.dump([r for r in rows], open(os.path.join(HERE, "mvp_prior_rows.json"), "w",
                                      encoding="utf-8"), ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
