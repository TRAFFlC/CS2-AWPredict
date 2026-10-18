"""选图模型的"层级 vs 丰度"混淆检验 —— 与 MVP 那次同类的问题。

背景（2026-12）：我一直写「选图模型在 T4 赛事上失效 15pp」
（一线队 60.2% vs 冷门赛事 45.7%）。但 MVP 那次已经暴露过同类错误：
我曾写「bo1 的 MVP 不可预测」，主理人纠正为「那只是样本集中在冷门赛事、
rating 数据稀疏」。**「T4」与「数据稀疏」在冷门赛事里同样被绑在一起。**

本脚本用真实赛果（142 场带真实图集的场次）分层检验：
  · 按赛事层级（csapi rank 判定，含前8队？T1/T2/T3/T4）
  · 按队伍丰度（选手历史出场场次中位数的队伍级中位）
  两者交叉，看哪个才是主导因素。

判据：逐图命中率（读法 B：报 k 张与真值图集比）。
"""
import collections
import io
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "../data")  # # 仓库布局：数据与代码分目录（Alpha 1.1.1）
sys.path.insert(0, HERE)
import model_v2 as M                                  # noqa: E402
import veto_model as VM                               # noqa: E402

POOL = ["Ancient", "Anubis", "Cache", "Dust2", "Inferno", "Mirage", "Nuke"]
FILES = {
    "fissure_2026-09-08_results.json": "2026-09-08",
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
    "fissure_cold2_results.json": "2026-09-22",
}


def load_rank():
    rk = {}
    ev = collections.defaultdict(set)
    for l in io.open(os.path.join(DATA, "csapi_matches_all.jsonl"),
                     encoding="utf-8"):
        if not l.strip():
            continue
        r = json.loads(l)
        for t in (r["team1"], r["team2"]):
            if t.get("rank"):
                rk[t["name"]] = int(t["rank"])
                ev[r.get("event") or "?"].add(int(t["rank"]))
    tier = {}
    for e, s in ev.items():
        n8 = sum(1 for x in s if x <= 8)
        n16 = sum(1 for x in s if x <= 16)
        tier[e] = ("T1" if (n8 >= 3 or ("major" in e.lower() and n8 >= 1))
                   else "T2" if n8 >= 1 else "T3" if n16 >= 4 else "T4")
    return rk, tier


def event_tier(A, B, rk, tier):
    """该**对阵**的层级：两队 rank 都 ≤16 记为 elite，否则 general。"""
    ra, rb = rk.get(A), rk.get(B)
    if ra is None or rb is None:
        return "unknown"
    return "elite" if max(ra, rb) <= 16 else "general"


def main():
    prep = M.prepare(verbose=False)
    st, mh, mm = prep["state"], prep["mh"], prep["mmodel"]
    rk, _evt = load_rank()
    out = []
    for fn, d in FILES.items():
        p = os.path.join(HERE, fn)
        if not os.path.exists(p):
            continue
        for x in json.load(io.open(p, encoding="utf-8")):
            bo = 5 if len(x["maps"]) == 5 else 3
            mh.advance(d)
            rows = mh.features_for(x["A"], x["B"], POOL, bo, date=d)
            if not rows:
                continue
            picked, _ = VM.predict_set(mm, {"bo": bo, "rows": rows}, bo)
            truth = set(x["maps"])
            hit = sum(1 for m in picked if m in truth)
            # 丰度：两队 lineup 选手历史场次的中位
            rich = []
            for t in (x["A"], x["B"]):
                lu = st.lineup(t, d)
                if lu:
                    g = sorted(float(st.p_w.get(q) or 0) for q in lu)
                    rich.append(g[len(g) // 2])
            out.append({"hit": hit, "n": len(picked),
                        "tier": event_tier(x["A"], x["B"], rk, _evt),
                        "rich": sum(rich) / len(rich) if rich else 0.0})
    print("可评估 %d 场\n" % len(out))

    def rep(sub, tag):
        if not sub:
            return
        h = sum(r["hit"] for r in sub)
        n = sum(r["n"] for r in sub)
        acc = h / max(1, n)
        se = math.sqrt(acc * (1 - acc) / max(1, n))
        print("  %-26s n=%3d 张=%3d  命中率 %5.1f%%  (1SE %.1fpp)"
              % (tag, len(sub), n, 100 * acc, 100 * se))
        return acc

    print("== A. 按对阵层级 ==")
    for t in ("elite", "general", "unknown"):
        rep([r for r in out if r["tier"] == t], t)
    print("\n== B. 按队伍丰度（两队中位场次）==")
    for lo, hi in ((0, 15), (15, 20), (20, 30), (30, 1e9)):
        rep([r for r in out if lo <= r["rich"] < hi], "[%g,%g)" % (lo, 999 if hi > 1e8 else hi))
    print("\n== C. 交叉：层级 × 丰度 ==")
    for t in ("elite", "general"):
        for lo, hi in ((0, 20), (20, 1e9)):
            sub = [r for r in out if r["tier"] == t and lo <= r["rich"] < hi]
            rep(sub, "%s × 丰度[%g,%s)" % (t, lo, "∞" if hi > 1e8 else hi))
    return 0


if __name__ == "__main__":
    sys.exit(main())
