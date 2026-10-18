"""选图模型的丰度依赖：能否用「借用全局先验」补上稀疏队伍？

现状（2026-12 实测，68 场真实赛果）：
  general × 丰度[0,20)  n=43  49.6%   ← 低数据区
  general × 丰度[20,∞)  n=8   66.7%   ← 高数据区

`MapHistory` 的设计文档写明：
  「**未见过的图退化为池内均匀先验**，不会因为『训练集没有 Cache』就把 Cache 判死」
  ⇒ 图级（per-map）已有兜底。但**队伍级**（per-team pick/ban）呢？

要检验的：
  A. 低丰度队伍是否**根本没有**队伍级统计（tpick/tban 全空）？
  B. 若是，能否用「同层级队伍的图级先验」兜底而不损失 elite 表现？
  C. 或者：低丰度时干脆只用图级（忽略队伍级），看命中率变化。

C 是最有操作价值的——它不需要新数据，只改打分。
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
RICH_MIN = 18


def load_rank():
    rk = {}
    for l in io.open(os.path.join(DATA, "csapi_matches_all.jsonl"),
                     encoding="utf-8"):
        if not l.strip():
            continue
        r = json.loads(l)
        for t in (r["team1"], r["team2"]):
            if t.get("rank"):
                rk[t["name"]] = int(t["rank"])
    return rk


def rich_of(st, team, d):
    lu = st.lineup(team, d)
    if not lu:
        return 0.0
    g = sorted(float(st.p_w.get(q) or 0) for q in lu)
    return g[len(g) // 2]


def main():
    prep = M.prepare(verbose=False)
    st, mm = prep["state"], prep["mmodel"]
    rk = load_rank()
    rows = []
    for fn, d in FILES.items():
        p = os.path.join(HERE, fn)
        if not os.path.exists(p):
            continue
        for x in json.load(io.open(p, encoding="utf-8")):
            bo = 5 if len(x["maps"]) == 5 else 3
            rows.append((d, x["A"], x["B"], bo, set(x["maps"])))
    print("可评估 %d 场\n" % len(rows))

    print("== A. 队伍级统计是否存在 ==")
    mh = prep["mh"]
    sample_teams = sorted({t for _d, a, b, _bo, _m in rows for t in (a, b)})
    have_pick = sum(1 for t in sample_teams if mh.npick.get(t, 0) > 0)
    have_ban = sum(1 for t in sample_teams if mh.nban.get(t, 0) > 0)
    print("  参赛队伍 %d 支：有 tpick 的 %d，有 tban 的 %d"
          % (len(sample_teams), have_pick, have_ban))

    # 队伍级特征的**真实前缀**是 a_ / b_ / both_（见 VM.FEATURES 与
    # MapHistory.features_for 的实测输出），**不是 t_**——第一版按 `t_*` 清零
    # 全部落空，两版数字完全相同才发现。
    TEAM_LEVEL = ("a_app", "b_app", "a_ban", "b_ban", "a_pick", "b_pick",
                  "a_wr", "b_wr", "both_pick", "both_ban", "min_app",
                  "both_pick_prod", "both_ban_prod")

    def run(drop_team_level):
        import copy
        h = copy.deepcopy(mh)
        res = collections.defaultdict(lambda: [0, 0])
        for d, A, B, bo, truth in rows:
            h.advance(d)
            fr = h.features_for(A, B, POOL, bo, date=d)
            if not fr:
                continue
            if drop_team_level:
                for r in fr:
                    if isinstance(r, dict):
                        for kk in TEAM_LEVEL:
                            r[kk] = 0.0
            picked, _ = VM.predict_set(mm, {"bo": bo, "rows": fr}, bo)
            k = sum(1 for m in picked if m in truth)
            rr = (rich_of(st, A, d) + rich_of(st, B, d)) / 2
            band = "低丰度" if rr < RICH_MIN else "高丰度"
            res[band][0] += k
            res[band][1] += bo
        return res

    print("\n== B/C. 队伍级特征的开/关（低丰度段是关键）==")
    for tag, drop in (("全特征(现状)", False),
                      ("去掉队伍级(a_/b_/both_)", True)):
        r = run(drop)
        print("  %s" % tag)
        for band in ("低丰度", "高丰度"):
            h, n = r[band]
            if n:
                print("    %-6s %3d 张  命中率 %5.1f%%"
                      % (band, n, 100 * h / n))
    print("\n  ⇒ 若「去掉队伍级」在低丰度段更好，说明队伍级统计在稀疏时是**噪声源**；")
    print("     若在高丰度段也更好，说明队伍级特征整体无效（更严重的结论）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
