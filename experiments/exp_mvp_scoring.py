"""MVP 打分函数选型：三种打分在**真实 MVP 标签**上的对比 + 留出验证。

背景（2026-12 查明 + 主理人 2026-12 判断）：
  * `model_v2.mvp_label()` 的标签 =「胜方队内 rating 最高者」，
    而 `decide.mvp_probs()` 也按 rating 类特征排序 ⇒ **自证**，
    语料 test 上 31.2% 不是预测能力。
  * 真实赛事 120 场里**模型 argmax 只有 16.7%**（随机 20%），与随机不可区分。
  * 主理人判断：「MVP 由 rating 决定，个人平均 rating 更高的更可能成为 MVP」。

候选打分（全部只用赛前可见信息）：
  S0  当前实现  z(adj) + z(toprate)     ← decide.mvp_probs 现用
  S1  纯 rating                        ← 主理人的直觉
  S2  纯 rating + 近期加权
  S3  rating 与 adj/toprate 的组合

`rating` 口径要分清（主理人强调「个人 rating，不是本场 rating」）：
  * `r["rating"]`：该场 roster 里的**赛前累计** rating（≈ 个人平均）  ⇒ 符合直觉
  * `r["adj"]`  ：由本场逐图数据算出的**调整后** rating（含本场表现）⇒ 含未来信息
  * `r["toprate"]`：本场 top rate                                        ⇒ 含未来信息
  ⚠ 若 S1（纯 rating）显著优于 S0，那说明**含本场信息反而有害**。

★ 留出：前 70% 选配置，后 30% 只报一次。
"""
import collections
import io
import json
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "../data")  # # 仓库布局：数据与代码分目录（Alpha 1.1.0）
sys.path.insert(0, HERE)
import decide as D                                    # noqa: E402

EPL = (r"E:\world\another\data\Mimo\MiMo-V2.6-Flash\2026-10-12"
       r"\live_tournament_results.jsonl")
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

# ★ Logitech G Connect 的 30 场 bo1 —— **bo1 子集的唯一来源**（EPL/裂变/StarLadder
#   全是 bo3/bo5）。漏掉它 ⇒ 「按 bo 分档标定」无从谈起：2026-12 首次跑本脚本时
#   bo1 子集为 0 场，直接导致把 bo3/bo5 选出的规则误用到 bo1 赛事上（380→370）。
LGC_DATE = "2026-09-20"


def load_lgc_truth():
    """从 score_lgc.py 的真值表取 Logitech G 的 35 场（bo1 30 + bo3 4 + bo5 1）。"""
    import score_lgc as S
    out = []
    for A, B, W, zh, mvp in S.BO1:
        out.append((A, B, W, (1, 0), [S.ZH2EN[zh]], mvp, 1))
    for A, B, W, sa, sb, zh, mvp, bo in S.REST:
        out.append((A, B, W, (sa, sb), [S.ZH2EN[m] for m in zh], mvp, bo))
    return out


def z(vals):
    n = len(vals)
    mu = sum(vals) / n
    sd = (sum((v - mu) ** 2 for v in vals) / n) ** 0.5 or 1.0
    return [(v - mu) / sd for v in vals]


def load_rows():
    """[(date_key, team, player_rows, mvp)]。

    ⚠ `player_rows` 用 `State.player_row()`（model_v2.py:428），其注释明写
    「**选手级特征（全部来自该场之前）**」⇒ `adj`/`toprate` 是**赛前**的衰减累计值，
    不含本场信息，盲预测时**可用**。
    （`records_v1.rosters` 只有 name+rating，是另一条更粗的口径。）
    """
    import model_v2 as M
    prep = M.prepare(verbose=False)
    st = prep["state"]
    recs = [json.loads(l) for l in
            io.open(os.path.join(HERE, "records_v1.jsonl"), encoding="utf-8")
            if l.strip()]
    recs.sort(key=lambda r: r["date"])

    # 复现 live_sample：state.lineup + player_row（用快照 state，时点正确）
    out = []
    def rows_for(team, date):
        lu = st.lineup(team, date)
        if not lu:
            return []
        opp = st.elo0
        return [dict(st.player_row(team, p, date, opp, lu), label=0) for p in lu]

    for l in io.open(EPL, encoding="utf-8"):
        if not l.strip():
            continue
        x = json.loads(l)
        if not (x.get("mvp") and x.get("winner")):
            continue
        pr = rows_for(x["winner"], x["date"])
        if len(pr) >= 2:
            out.append((x["date"], x["winner"], pr, x["mvp"], x["best_of"]))
    for f, d in FILES.items():
        p = os.path.join(HERE, f)
        if not os.path.exists(p):
            continue
        for i, x in enumerate(json.load(io.open(p, encoding="utf-8"))):
            if not x.get("mvp"):
                continue
            pr = rows_for(x["winner"], d)
            if len(pr) >= 2:
                bo = 5 if len(x["maps"]) == 5 else 3
                out.append(("%s#%02d" % (d, i), x["winner"], pr, x["mvp"], bo))
    # Logitech G Connect：bo1 的唯一来源
    for j, (A, B, W, sc, maps, mvp, bo) in enumerate(load_lgc_truth()):
        pr = rows_for(W, LGC_DATE)
        if len(pr) >= 2:
            out.append(("LGC#%02d" % j, W, pr, mvp, bo))
    return out


def pick(pl, key):
    return max(pl, key=lambda r: r.get(key) or 0)["player"]


def score_argmax(pl, mode, tau=1.0):
    names = [r["player"] for r in pl]
    if mode == "rating":
        return max(pl, key=lambda r: r.get("rating") or 0)["player"]
    if mode == "adj":
        return max(pl, key=lambda r: r.get("adj") or 0)["player"]
    if mode == "toprate":
        return max(pl, key=lambda r: r.get("toprate") or 0)["player"]
    if mode == "adj+toprate":
        pr = D.mvp_probs(pl)          # {player: prob}
        return max(pr, key=lambda p: pr[p])
    if mode == "rating+adj":
        zr, za = z([r.get("rating") or 0 for r in pl]), z([r.get("adj") or 0 for r in pl])
        return names[max(range(len(pl)), key=lambda i: zr[i] + za[i])]
    if mode == "rating+adj+toprate":
        zr = z([r.get("rating") or 0 for r in pl])
        za = z([r.get("adj") or 0 for r in pl])
        zt = z([r.get("toprate") or 0 for r in pl])
        return names[max(range(len(pl)), key=lambda i: zr[i] + za[i] + zt[i])]
    raise ValueError(mode)


MODES = ["adj+toprate", "rating", "adj", "toprate",
         "rating+adj", "rating+adj+toprate"]


def report(rows, modes, title):
    print("\n== %s（n=%d）==" % (title, len(rows)))
    print("  %-22s %3s %8s %9s" % ("打分", "命中", "准确率", "距随机"))
    print("  " + "-" * 46)
    for m in modes:
        k = sum(1 for r in rows if score_argmax(r[2], m) == r[3])
        acc = k / len(rows)
        se = math.sqrt(acc * (1 - acc) / len(rows))
        star = "  ←" if m == "rating" else ""
        print("  %-22s %3d %7.1f%% %+8.1fσ%s"
              % (m, k, 100 * acc, (acc - 0.2) / se if se else 0, star))
    print("  %-22s %3s %7.1f%%" % ("（随机基线）", "-", 20.0))


def main():
    rows = load_rows()
    # 按 bo 分档：**bo1 只有 1 张图**，MVP 更接近随机；bo3/bo5 才是
    # 「整体表现决定 MVP」的场合。
    # 2026-12 实测：在 bo3/bo5 为主的 107 场选出 z(adj) 最优，直接用到
    # bo1 为主的 Logitech G（35 场）反而 380 → 370（−10 分）⇒ 必须分档标定。
    b1 = [r for r in rows if r[4] == 1]
    b35 = [r for r in rows if r[4] != 1]
    print("真实 MVP 标签可评估：%d 场   bo1 %d / bo3+ %d"
          % (len(rows), len(b1), len(b35)))
    k = int(len(rows) * 0.70)
    tr, te = rows[:k], rows[k:]
    print("训练段 %d / 留出段 %d\n" % (len(tr), len(te)))
    report(tr, MODES, "训练段（选配置）")
    report(te, MODES, "留出段（只报一次）")
    report(rows, MODES, "全体（参考）")
    print("\n★ 分赛制（决定 MVP_CAL_PROB 是否该按 bo 分档）==")
    report(b35, MODES, "bo3/bo5 子集")
    report(b1, MODES, "bo1 子集")
    return 0


if __name__ == "__main__":
    sys.exit(main())
