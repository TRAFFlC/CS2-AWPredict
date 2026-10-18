"""比分项诊断：为什么实测 27.9% 远低于语料基准 55.2%？

已知：
  · 语料里「已知胜方后报 2:0」的命中率 = **55.2%**（n=2439）
  · `decide.choose_score` 取 `score_true[w]` 的 argmax（众数），方向没错
  · 但四个赛事实测比分项 **12/43 = 27.9%**（且该项条件于胜方正确）

三条可能，必须分离：
  H1 **模型的条件分布有偏**：`score_model.cond_dist(bo, q)` 在真实赛事上
     系统性高估「横扫」概率。查法：把模型给的 2:0 概率与真实命中率对齐。
  H2 **条件于「模型也选对了胜方」的子集**，横扫率本身就低
     —— 即「容易横扫的场次，模型的胜方预测反而容易错」。
     这两个是**耦合**的（都取决于 P(强弱悬殊)），查法：按胜方置信度分层。
  H3 **veto 比分存在反 相关**：模型选图越准 ⇒ 越可能是精心设计的 BP ⇒ 越打满。

本脚本对 43 场真实赛果做诊断，并按**模型置信度**分层。
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
sys.path.insert(0, HERE)  # scorer.py 与本模块同目录（仓库布局）
import decide as D                                    # noqa: E402
import model_v2 as M                                  # noqa: E402

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


def main():
    prep = M.prepare(verbose=False)
    import predict as PR
    pr = PR.Predictor(quiet=True)
    st, sm = pr.state, pr.smodel
    rows = []
    for fn, d in FILES.items():
        p = os.path.join(HERE, fn)
        if not os.path.exists(p):
            continue
        for x in json.load(io.open(p, encoding="utf-8")):
            bo = 5 if len(x["maps"]) == 5 else 3
            rows.append((d, x, bo))
    b3 = [r for r in rows if r[2] == 3]
    print("可诊断 bo3 %d 场（bo5 仅 %d 场，单列）\n"
          % (len(b3), len(rows) - len(b3)))

    # 语料基准：已知胜方后的横扫率
    corpus = collections.Counter()
    for l in io.open(os.path.join(DATA, "csapi_matches_all.jsonl"),
                     encoding="utf-8"):
        if not l.strip():
            continue
        r = json.loads(l)
        if r["best_of"] != 3:
            continue
        s1, s2 = r["team1"]["score"], r["team2"]["score"]
        if s1 == s2:
            continue
        corpus[(max(s1, s2), min(s1, s2))] += 1
    tot = sum(corpus.values())
    print("语料基准（已知胜方，bo3）：2:0 = %.1f%%   2:1 = %.1f%%  (n=%d)"
          % (100 * corpus[(2, 0)] / tot, 100 * corpus[(2, 1)] / tot, tot))

    diag = []
    for d, x, bo in rows:
        if bo != 3:
            continue
        A, B, W = x["A"], x["B"], x["winner"]
        sa, sb = x["score"]
        real = (sa, sb) if sa > sb else (sb, sa)
        myw = x["my_winner"]
        q = 1.0
        lu_a, lu_b = st.lineup(A, d), st.lineup(B, d)
        if lu_a and lu_b:
            ea = st.elo_of(A)[0]
            eb = st.elo_of(B)[0]
            q = 1.0 / (1.0 + 10 ** (-(ea - eb) / 400.0))
        else:
            q = None
        # 模型当时预测的是 2:0（相对 myw 朝向）
        diag.append({"A": A, "B": B, "W": W, "real": real, "myw": myw,
                     "win_ok": myw == W, "q": q,
                     "sweep": real == (2, 0)})
    n = len(diag)
    ok = [d for d in diag if d["win_ok"]]
    print("\n== 1. 实测（bo3，%d 场）==" % n)
    print("  胜方命中 %d/%d = %.1f%%" % (len(ok), n, 100 * len(ok) / n))
    sw = sum(d["sweep"] for d in ok)
    print("  在胜方正确的 %d 场里，真实横扫率 = %d/%d = %.1f%%   （语料 %.1f%%）"
          % (len(ok), sw, len(ok), 100 * sw / len(ok), 100 * corpus[(2, 0)] / tot))
    print("  ⇒ 比分项实测 %d/%d = %.1f%%" % (sw, n, 100 * sw / n))

    print("\n== 2. 按 Elo 差分层（强弱悬殊度）==")
    print("  %-16s %5s %10s %14s" % ("|Elo差|/400", "n", "横扫率", "语料基准"))
    print("  " + "-" * 48)
    bands = [(0.0, 0.5), (0.5, 1.0), (1.0, 1e9)]
    for lo, hi in bands:
        sub = [d for d in diag if d["q"] is not None and lo <= abs(d["q"] - 0.5) * 2 < hi]
        if sub:
            s2 = sum(d["sweep"] for d in sub)
            print("  %-16s %5d %9.1f%% %13.1f%%"
                  % ("[%.1f,%s)" % (lo, "∞" if hi > 1e8 else hi), len(sub),
                     100 * s2 / len(sub), 100 * corpus[(2, 0)] / tot))

    print("\n== 3. 模型的条件分布校准（score_true 的 argmax 是否合理）==")
    print("  抽查 8 场：模型给 2:0 的概率 vs 实际是否横扫")
    shown = 0
    for d, x, bo in rows:
        if bo != 3 or shown >= 8:
            continue
        A, B = x["A"], x["B"]
        s = st.lineup(A, "2026-09-05")
        if not s:
            continue
        shown += 1
        ea, eb = st.elo_of(A)[0], st.elo_of(B)[0]
        q = 1.0 / (1.0 + 10 ** (-(ea - eb) / 400.0))
        cds = sm.cond_dist(3, q if x["my_winner"] == A else 1 - q)
        a = max(cds, key=lambda k: cds[k])
        sa, sb = x["score"]
        real = (sa, sb) if sa > sb else (sb, sa)
        print("    %-11s vs %-11s  q=%.2f  模型 argmax=%s(%.2f)  实际=%s  %s"
              % (A, B, q, "%d:%d" % a, cds[a], "%d:%d" % real,
                 "✓" if a == real else "✗"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
