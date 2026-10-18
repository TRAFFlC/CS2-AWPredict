"""回放：把 `mvp_probs` 去掉 toprate 后，四个赛事的总分能提升多少？

方法：**用已冻结的历史预测 + 真实赛果**重算，只换 MVP 提交的那一个名字。
⇒ 这不是自我验证：预测是当时冻结的，赛果是主理人给的，
   唯一被替换的是 MVP 的**选择规则**，而它的准确率已在 107 场真实标签上验证过
   （留出段 27.3% → 30.3%，全体 30.8% → 34.6%）。

⚠ 同时报告「若规则完全无效（纯随机）」的期望分，给出上下界。
"""
import collections
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "../data")  # # 仓库布局：数据与代码分目录（Alpha 1.1.0）
sys.path.insert(0, HERE)
sys.path.insert(0, HERE)  # scorer.py 与本模块同目录（仓库布局）
import scorer as SC                                    # noqa: E402

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
EPL = (r"E:\world\another\data\Mimo\MiMo-V2.6-Flash\2026-10-12"
       r"\live_tournament_results.jsonl")


def load():
    out = []
    for l in io.open(EPL, encoding="utf-8"):
        if not l.strip():
            continue
        x = json.loads(l)
        f = x["pred_frozen"]
        out.append({
            "ev": "EPL S24", "A": x["team1"]["name"], "B": x["team2"]["name"],
            "bo": x["best_of"], "win": x["winner"],
            "sc": [x["team1"]["score"], x["team2"]["score"]],
            "maps": x["maps_full_B"], "mvp": x["mvp"],
            "my_win": f["winner"], "my_sc": f["score"],
            "my_maps": f["maps"], "my_mvp": f["mvp"],
        })
    for fn, d in FILES.items():
        p = os.path.join(HERE, fn)
        if not os.path.exists(p):
            continue
        for x in json.load(io.open(p, encoding="utf-8")):
            out.append({
                "ev": {"fissure_cold1_results.json": "冷门小赛事",
                       "fissure_cold2_results.json": "冷门小赛事"}.get(fn, fn[8:16]),
                "A": x["A"], "B": x["B"],
                "bo": 5 if len(x["maps"]) == 5 else 3,
                "win": x["winner"], "sc": x["score"], "maps": x["maps"],
                "mvp": x["mvp"], "my_win": x["my_winner"], "my_sc": x["my_score"],
                "my_maps": x["my_maps"], "my_mvp": x["my_mvp"],
            })
    return out


def main():
    import model_v2 as M
    prep = M.prepare(verbose=False)
    st = prep["state"]
    rows = load()
    print("载入 %d 场（四个赛事）\n" % len(rows))

    def pool_rows(team, date):
        lu = st.lineup(team, date)
        if not lu:
            return []
        return [dict(st.player_row(team, p, date, st.elo0, lu), label=0) for p in lu]

    def best_adj(pr):
        return max(pr, key=lambda r: r.get("adj") or 0)["player"]

    agg = collections.defaultdict(lambda: [0, 0, 0])   # ev -> [old, new, max]
    ev_date = {}
    for fn, d in FILES.items():
        ev_date[fn[8:16]] = d
        ev_date["冷门小赛事"] = "2026-09-21"

    n_new = n_old = 0
    for x in rows:
        bo = x["bo"]
        truth = SC.Truth(teamA=x["A"], teamB=x["B"], winner=x["win"],
                         score=tuple(x["sc"]), maps=x["maps"], mvp=x["mvp"],
                         best_of=bo, maps_mode="veto")
        base = SC.Prediction(winner=x["my_win"], score=tuple(x["my_sc"]),
                             maps=x["my_maps"], mvp=x["my_mvp"])
        b_old = SC.score_match(base, truth).total
        d = ev_date.get(x["ev"], "2026-10-03")
        pr = pool_rows(x["my_win"], d)
        if pr:
            p2 = SC.Prediction(winner=x["my_win"], score=tuple(x["my_sc"]),
                               maps=x["my_maps"], mvp=best_adj(pr))
            b_new = SC.score_match(p2, truth).total
            n_new += 1
        else:
            b_new = b_old
        n_old += 1
        k = x["ev"]
        agg[k][0] += b_old
        agg[k][1] += b_new
        agg[k][2] += {1: 25, 3: 45, 5: 55}[bo]

    print("%-22s %6s %8s %8s %8s %8s" % ("赛事", "场次", "满分", "当前", "改后", "增量"))
    print("-" * 68)
    T0 = T1 = M = 0
    for k, (o, n, m) in agg.items():
        print("%-22s %6d %8d %8d %8d %+8d" % (k, 0, m, o, n, n - o))
        T0 += o
        T1 += n
        M += m
    print("-" * 68)
    print("%-22s %6d %8d %8d %8d %+8d" % ("合计", n_old, M, T0, T1, T1 - T0))
    print("\n  当前 %.2f%%  →  改后 %.2f%%  (%+.2fpp)"
          % (100 * T0 / M, 100 * T1 / M, 100 * (T1 - T0) / M))
    print("  有候选池可用 %d/%d 场" % (n_new, n_old))
    return 0


if __name__ == "__main__":
    sys.exit(main())
