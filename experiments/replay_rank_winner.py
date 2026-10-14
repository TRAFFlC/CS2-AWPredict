"""胜方项改用 rank 热门的回放：能挽回多少分？

背景（2026-10-12，43 场真实盲预测）：
  模型胜方 27/43 (62.8%)   rank 热门 30/43 (69.8%)   差 −7.0pp
且「模型 vs rank 热门」在两个赛事上高度一致（−6.9 / −7.1pp），
语料 test 上也是负的（−3.2pp）⇒ 模型在胜方项上**没有超过 rank**。

本脚本用**同一份冻结件**（选图 / MVP / 比分 = 2:0 全部不变），
只把胜方项换成 rank 更高的队，重算总分 ⇒ 直接回答「改用 rank 能不能进前 100」。

注意比分项会**跟着变**：预测 2:0 是相对「我选的那一方」的朝向，
换胜方后朝向必须同步翻转，否则会把 2:0 变成 0:2 而白丢 10 分。
这是本脚本最容易写错的地方（第一版就这么错过一次）。
"""
import io, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "../data")  # # 仓库布局：数据与代码分目录（Alpha 1.0）
sys.path.insert(0, HERE)
sys.path.insert(0, HERE)  # scorer.py 与本模块同目录（仓库布局）
import scorer as SC

DAYS = ("2026-09-08", "2026-09-09", "2026-09-10", "2026-09-11",
        "2026-09-12", "2026-09-13", "2026-09-17", "2026-09-18",
        "2026-09-19", "2026-09-20", "2026-09-20-bo5")
# 09-20 决赛的真实 bo 未标注，首轮按 bo3 预测。bo5 重测那份**取代**bo3 那份，
# 但只取代决赛**那一场**——整个日期跳过会连带丢掉当天另外两场。
BO5_KEY = ("Vitality", "Aurora")


def load():
    res = []
    for d in DAYS:
        if d == "2026-09-20-bo5":
            continue
        p = os.path.join(HERE, "fissure_%s_results.json" % d)
        if not os.path.exists(p):
            continue
        for x in json.load(io.open(p, encoding="utf-8")):
            if (x["A"], x["B"]) == BO5_KEY:
                continue          # 决赛交给 bo5 重测那份
            res.append(x)
    p = os.path.join(DATA, "groundtruth", "fissure_2026-09-20-bo5_results.json")
    if os.path.exists(p):
        res += json.load(io.open(p, encoding="utf-8"))
    return res


def total(res, use_rank):
    """use_rank=True 时胜方换成 rank 更高的队；其余项全部沿用冻结件。"""
    tot = {"winner": 0, "score": 0, "maps": 0, "mvp": 0}
    detail = []
    for x in res:
        bo = 5 if len(x["maps"]) == 5 else 3
        truth = SC.Truth(teamA=x["A"], teamB=x["B"], winner=x["winner"],
                         score=tuple(x["score"]), maps=x["maps"], mvp=x["mvp"],
                         best_of=bo, maps_mode="veto")
        if use_rank:
            w = x["A"] if x["my_rank"]["A"] < x["my_rank"]["B"] else x["B"]
        else:
            w = x["my_winner"]
        # 比分项的隐含胜方必须与 pred.winner 一致（scorer.py:144），否则整项 0。
        # 冻结件的 my_score 是 **(A,B) 朝向**；换胜方到对面时必须把比分线**翻转**，
        # 否则「我方 2:0」会变成「对方 2:0」，凭空丢 10 分。
        # （第一版硬编码 (2,0) 错；第二版原样用 my_score 也错——它同样不翻转。）
        ms = tuple(x["my_score"])
        if w != x["my_winner"]:
            ms = (ms[1], ms[0])
        pred = SC.Prediction(winner=w, score=ms,
                             maps=x["my_maps"], mvp=x["my_mvp"])
        bd = SC.score_match(pred, truth)
        for k in tot:
            tot[k] += getattr(bd, k)
        detail.append((x["A"], x["B"], x["winner"], x["my_winner"], w,
                       bd.total, x["total"]))
    return tot, detail


def main():
    res = load()
    mx = sum(55 if len(x["maps"]) == 5 else 45 for x in res)
    n = len(res)
    print("=== %d 场 / 满分 %d ===\n" % (n, mx))
    print("%-10s %8s %8s %8s %8s %8s %9s" %
          ("胜方来源", "胜方", "比分", "选图", "MVP", "总分", "得分率"))
    print("-" * 66)
    out = {}
    for tag, ur in (("模型", False), ("rank热门", True)):
        tot, det = total(res, ur)
        T = sum(tot.values())
        out[tag] = (tot, T)
        print("%-10s %8d %8d %8d %8d %8d %8.2f%%" %
              (tag, tot["winner"], tot["score"], tot["maps"], tot["mvp"], T,
               100 * T / mx))
    d = out["rank热门"][1] - out["模型"][1]
    print("-" * 66)
    print("rank 热门 vs 模型：%+d 分（%+.2fpp）" % (d, 100 * d / mx))
    print()
    print("逐场差异（仅列出有变化的）：")
    _, det_m = total(res, False)
    _, det_r = total(res, True)
    ch = [(a, b, w, mw, rw, tm, tr) for a, b, w, mw, rw, tm, tr in det_r
          if tm != tr]
    for a, b, w, mw, rw, tm, tr in ch:
        print("  %-11s %-13s 实际 %-11s │ 模型选 %-11s %+3d   rank选 %-11s %+3d   %+d"
              % (a, b, w, mw, tm, rw, tr, tr - tm))
    print("\n  变化 %d 场，合计 %+d 分" % (len(ch), d))
    print("\n  对照：StarLadder 榜单 #100 = 335，模型实得 %d" % out["模型"][1])
    print("       若改用 rank，%d 分 ⇒ %s 前 100"
          % (out["rank热门"][1],
             "仍差 %d 分，**进不了**" % (335 - out["rank热门"][1])
             if out["rank热门"][1] < 335 else "**进得了**"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
