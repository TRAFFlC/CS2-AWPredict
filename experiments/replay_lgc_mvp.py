"""Logitech G Connect 的 MVP 规则回放：旧规则 z(adj)+z(toprate) vs 新规则 z(adj)。

⚠ 为什么单独写：`replay_mvp_rule.py` 的 `load()` 只读四个赛事的 results jsonl，
**不包含 Logitech G 的 35 场**（它当时还没跑）。于是「+50 分」的结论里
没有这 35 场，而 `score_lgc.py` 又显示 380 → 370（−10）。必须补齐才能定论。

同时检验：把新规则**按 bo 分档**（bo1 用 0.20 期望）是否有帮助。
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
import decide as D                                     # noqa: E402
import model_v2 as M                                   # noqa: E402
import score_lgc as S                                  # noqa: E402
import scorer as SC                                   # noqa: E402

LGC_DATE = "2026-09-20"


def main():
    st = M.prepare(verbose=False)["state"]

    def rows_for(team):
        lu = st.lineup(team, LGC_DATE)
        if not lu:
            return []
        return [dict(st.player_row(team, p, LGC_DATE, st.elo0, lu), label=0)
                for p in lu]

    # 冻结的预测（胜负/选图不变，只换 MVP）
    pred = {(p["A"], p["B"]): p for p in
            json.load(io.open(os.path.join(DATA, "groundtruth", "lgc_predictions.json"),
                              encoding="utf-8"))}

    print("== Logitech G Connect 35 场 · MVP 规则回放 ==\n")
    print("%-30s %6s %7s %7s %6s" % ("对阵", "满分", "旧规则", "z(adj)", "差"))
    print("-" * 60)
    agg = collections.defaultdict(lambda: [0, 0, 0])
    for A, B, W, zh, mvp in S.BO1:
        x = pred[(A, B)]
        pr = rows_for(x["winner"])
        if not pr:
            print("%-30s %6d %7s %7s %6s  （无候选池）"
                  % (A + " vs " + B, 25, "-", "-", "-"))
            continue
        # 旧规则：z(adj)+z(toprate)（tau 同默认值）
        def zs(key):
            v = [r[key] for r in pr]
            mu = sum(v) / len(v)
            sd = (sum((t - mu) ** 2 for t in v) / len(v)) ** 0.5 or 1.0
            return [(t - mu) / sd for t in v]
        za, zt = zs("adj"), zs("toprate")
        old = max(range(len(pr)),
                  key=lambda i: D.DEFAULT_TAU * (za[i] + zt[i]))
        new = max(range(len(pr)), key=lambda i: D.DEFAULT_TAU * za[i])
        m_old = pr[old]["player"]
        m_new = pr[new]["player"]

        truth = SC.Truth(A, B, W, (1, 0), [S.ZH2EN[zh]], mvp,
                         best_of=1, maps_mode="veto")

        def sc(m):
            return SC.score_match(SC.Prediction(x["winner"], (1, 0),
                                                x["maps"].split(" / "), m),
                                  truth).total
        a, c = sc(m_old), sc(m_new)
        k = agg[1]
        k[0] += a
        k[1] += c
        k[2] += 25
        flag = " ★" if m_old != m_new else ""
        print("%-30s %6d %7d %7d %+6d%s%s"
              % (A + " vs " + B, 25, a, c, c - a, flag,
                 "  (MVP %s→%s)" % (m_old, m_new) if m_old != m_new else ""))

    for A, B, W, sa, sb, zh, mvp, bo in S.REST:
        x = pred[(A, B)]
        pr = rows_for(x["winner"])
        maps = [S.ZH2EN[m] for m in zh]
        truth = SC.Truth(A, B, W, (sa, sb), maps, mvp, best_of=bo, maps_mode="veto")
        mx = {3: 45, 5: 55}[bo]
        if not pr:
            print("%-30s %6d %7s %7s %6s  （bo%d 无候选池）"
                  % (A + " vs " + B, mx, "-", "-", "-", bo))
            continue

        def zs(key):
            v = [r[key] for r in pr]
            mu = sum(v) / len(v)
            sd = (sum((t - mu) ** 2 for t in v) / len(v)) ** 0.5 or 1.0
            return [(t - mu) / sd for t in v]
        za, zt = zs("adj"), zs("toprate")
        m_old = pr[max(range(len(pr)),
                      key=lambda i: D.DEFAULT_TAU * (za[i] + zt[i]))]["player"]
        m_new = pr[max(range(len(pr)), key=lambda i: D.DEFAULT_TAU * za[i])]["player"]

        def sc(m):
            return SC.score_match(SC.Prediction(
                x["winner"], tuple(int(v) for v in x["score"].split(":")),
                x["maps"].split(" / "), m), truth).total
        a, c = sc(m_old), sc(m_new)
        k = agg[bo]
        k[0] += a
        k[1] += c
        k[2] += mx
        print("%-30s %6d %7d %7d %+6d   bo%d%s"
              % (A + " vs " + B, mx, a, c, c - a, bo,
                 "  ★" if m_old != m_new else ""))

    print("-" * 60)
    O = sum(v[0] for v in agg.values())
    N = sum(v[1] for v in agg.values())
    F = sum(v[2] for v in agg.values())
    for bo in sorted(agg):
        o, n, f = agg[bo]
        print("  bo%d  满分 %4d   旧 %4d   新 %4d   %+d" % (bo, f, o, n, n - o))
    print("  ───────────────────────────────────────")
    print("  合计 满分 %d   旧规则 %d   z(adj) %d   %+d" % (F, O, N, N - O))
    print("  得分率 %.2f%% → %.2f%% (%+.2fpp)"
          % (100 * O / F, 100 * N / F, 100 * (N - O) / F))
    return 0


if __name__ == "__main__":
    sys.exit(main())
