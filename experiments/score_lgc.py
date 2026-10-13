"""Logitech G Connect 2026 计分（35 场 / 满分 985）。

计分口径（官方计分器 `014-scorer/scorer.py`，读法 B）：
  bo1 满分 **25** —— 比分项**整项不计**（裁决 Q-007）
  bo3 满分 45、bo5 满分 55 —— 含比分项 10 分

**bo1 的结构差异（本赛事的主体，30/35 场）**：
  第 1 项 胜方   10 分
  第 2 项 比分    0 分  ← bo1 不计
  第 3 项 选图    5 分  ← **7 张图里蒙 1 张**，基准 ≈14%
  第 4 项 MVP    10 分  ← 5 选 1，基准 =20%，实测模型 ≈14~31%
  ⇒ 期望得分率天然远低于 bo3/bo5，这是**计分结构**决定的。

⚠ 主理人提到的两个坑在这里都会被实际触发：
  * bo5 决赛报的是 **3:0**，而语料 bo5 横扫率只有 36.7%（bo3 的 2:0 是 55.2%）
    ⇒ 若实际是 3:1/3:2，主理人「无脑 3:0 不是最优」的直觉会**当场兑现**
  * bo1 选图只有 5 分，且模型对 Luminosity（语料 2 场）/BBL（1 场）几乎没数据
"""
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, HERE)  # scorer.py 与本模块同目录（仓库布局）
import scorer as SC

ZH2EN = {"遗迹": "Dust2", "迷城": "Mirage", "沙二": "Anubis", "阿努比斯": "Anubis",
         "死城": "Cache", "核子": "Nuke", "小镇": "Inferno", "远古": "Ancient"}

# bo1: (A, B, 胜方, 图, MVP) —— bo1 无比分
BO1 = [
    ("B8", "Metizport", "B8", "死城", "kensizor"),
    ("M80", "Ninjas in Pyjamas", "M80", "迷城", "slaxz-"),
    ("GamerLegion", "Luminosity", "GamerLegion", "核子", "fl4mus"),
    ("Luminosity", "Metizport", "Luminosity", "迷城", "afro"),
    ("B8", "GamerLegion", "B8", "遗迹", "kensizor"),
    ("M80", "GamerLegion", "M80", "迷城", "jba"),
    ("B8", "Ninjas in Pyjamas", "B8", "遗迹", "s1zzi"),
    ("B8", "Luminosity", "B8", "沙二", "npl"),
    ("Ninjas in Pyjamas", "Metizport", "Ninjas in Pyjamas", "核子", "sjuush"),
    ("GamerLegion", "Ninjas in Pyjamas", "GamerLegion", "沙二", "tauson"),
    ("M80", "Metizport", "M80", "阿努比斯", "slaxz-"),
    ("Luminosity", "M80", "Luminosity", "遗迹", "rainwaker"),
    ("Metizport", "GamerLegion", "Metizport", "核子", "mail09"),
    ("B8", "M80", "B8", "迷城", "kensizor"),
    ("Luminosity", "M80", "Luminosity", "遗迹", "azuwu"),
    ("Inner Circle", "BBL", "Inner Circle", "死城", "headtr1ck"),
    ("Liquid", "INFINITE", "Liquid", "迷城", "naf"),
    ("3DMAX", "EYEBALLERS", "3DMAX", "遗迹", "graviti"),
    ("BBL", "INFINITE", "BBL", "核子", "leakz"),
    ("Inner Circle", "Liquid", "Inner Circle", "死城", "dawy"),
    ("Inner Circle", "3DMAX", "Inner Circle", "核子", "cptkurtka023"),
    ("EYEBALLERS", "Liquid", "EYEBALLERS", "迷城", "dex"),
    ("Inner Circle", "INFINITE", "Inner Circle", "迷城", "headtr1ck"),
    ("3DMAX", "BBL", "3DMAX", "遗迹", "kursy"),
    ("Liquid", "3DMAX", "Liquid", "核子", "elige"),
    ("BBL", "EYEBALLERS", "BBL", "遗迹", "boye"),
    ("BBL", "Liquid", "BBL", "核子", "nickyb"),
    ("EYEBALLERS", "INFINITE", "EYEBALLERS", "沙二", "jw"),
    ("EYEBALLERS", "Inner Circle", "EYEBALLERS", "阿努比斯", "ro1f"),
    ("3DMAX", "INFINITE", "3DMAX", "沙二", "lucky"),
]

# bo3/bo5: (A, B, 胜方, sa, sb, [图], MVP)
REST = [
    ("Luminosity", "BBL", "Luminosity", 2, 0, ["核子", "迷城", "遗迹"], "afro", 3),
    ("3DMAX", "M80", "3DMAX", 2, 1, ["小镇", "遗迹", "沙二"], "lucky", 3),
    ("Luminosity", "B8", "Luminosity", 2, 0, ["小镇", "遗迹", "沙二"], "afro", 3),
    ("3DMAX", "Inner Circle", "3DMAX", 2, 1, ["遗迹", "沙二", "死城"], "graviti", 3),
    ("3DMAX", "Luminosity", "3DMAX", 3, 2, ["死城", "沙二", "遗迹", "小镇", "核子"], "maka", 5),
]


def main():
    pred = {(p["A"], p["B"]): p for p in
            json.load(io.open(os.path.join(HERE, "lgc_predictions.json"),
                              encoding="utf-8"))}
    tot = {"winner": 0, "score": 0, "maps": 0, "mvp": 0}
    per_bo = {}

    print("== bo1（30 场，满分 25，比分项不计）==\n")
    for A, B, W, zh, mvp in BO1:
        m = ZH2EN[zh]
        truth = SC.Truth(teamA=A, teamB=B, winner=W, score=(1, 0), maps=[m],
                         mvp=mvp, best_of=1, maps_mode="veto")
        p = pred.get((A, B)) or pred.get((B, A))
        if p is None:
            print("  %-20s vs %-20s  **缺预测**" % (A, B))
            continue
        pr = SC.Prediction(winner=p["winner"], score=(1, 0),
                           maps=p["maps"].split(" / "), mvp=p["mvp"])
        bd = SC.score_match(pr, truth)
        for k in tot:
            tot[k] += getattr(bd, k)
        e = per_bo.setdefault(1, [0, 0])
        e[0] += bd.total
        e[1] += 25
        print("  %-20s %-20s 实际 %-20s │ 我判 %-20s 胜%s 图%s MVP%s │ +%2d/25"
              % (A, B, W, p["winner"], "✓" if bd.winner else "✗",
                 "✓" if bd.maps else "✗", "✓" if bd.mvp else "✗", bd.total))

    print("\n== bo3/bo5（5 场）==\n")
    for A, B, W, sa, sb, zh, mvp, bo in REST:
        maps = [ZH2EN[m] for m in zh]
        truth = SC.Truth(teamA=A, teamB=B, winner=W, score=(sa, sb), maps=maps,
                         mvp=mvp, best_of=bo, maps_mode="veto")
        p = pred.get((A, B)) or pred.get((B, A))
        pr = SC.Prediction(winner=p["winner"],
                           score=tuple(int(x) for x in p["score"].split(":")),
                           maps=p["maps"].split(" / "), mvp=p["mvp"])
        bd = SC.score_match(pr, truth)
        for k in tot:
            tot[k] += getattr(bd, k)
        e = per_bo.setdefault(bo, [0, 0])
        e[0] += bd.total
        e[1] += {3: 45, 5: 55}[bo]
        print("  bo%d %-14s %d:%d %-14s │ 我判 %-14s %s │ 胜%s 比分%s 图%d/%d MVP%s │ +%d/%d"
              % (bo, A, sa, sb, B, p["winner"], p["score"],
                 "✓" if bd.winner else "✗", "✓" if bd.score else "✗",
                 len(bd.maps_hit), len(maps), "✓" if bd.mvp else "✗",
                 bd.total, {3: 45, 5: 55}[bo]))

    n1, n3, n5 = len(BO1), 4, 1
    full = n1 * 25 + n3 * 45 + n5 * 55
    T = sum(tot.values())
    print("\n  胜方 %d/%d = %d 分" % (tot["winner"] // 10, n1 + n3 + n5, tot["winner"]))
    print("  比分 %d/%d = %d 分   （bo1 %d 场本项不计）"
          % (tot["score"] // 10, n3 + n5, tot["score"], n1))
    print("  选图 %d/%d = %d 分"
          % (tot["maps"] // 5, n1 + 3 * n3 + 5 * n5, tot["maps"]))
    print("  MVP  %d/%d = %d 分" % (tot["mvp"] // 10, n1 + n3 + n5, tot["mvp"]))
    # 自估 = 各场 exp_points 之和（**实算**）。
    # 原先这里硬编码 `est = 362.8`，那是某个时刻手抄的数字：之后 MVP 标定改成
    # 按丰度分档、胜方改成 rank 优先，自估早就变了，却仍在打印旧值 ⇒ 偏差栏失真。
    est = 0.0
    miss_est = 0
    for key, p in pred.items():
        # ⚠ 预测文件里字段名是 `exp`（不是 `exp_points`）；早先写死 `est=362.8`
        #   就是因为没去实算，改用实算后本赛事自估为 341.5。
        e = p.get("exp", p.get("exp_points"))
        if isinstance(e, dict):                 # 容错：若某版存成分项字典
            e = sum(v for v in e.values() if isinstance(v, (int, float)))
        if isinstance(e, (int, float)):
            est += float(e)
        else:
            miss_est += 1
    print("  ───────────────────────────────")
    print("  合计 %d / %d = %.2f%%" % (T, full, 100 * T / full))
    if miss_est:
        print("  自估 **不可用**：%d 场预测缺 exp_points 字段"
              "（预测文件由旧版 predict 产出，需重跑 predict_lgc.py）" % miss_est)
    else:
        print("  自估 %.1f (%.2f%%)  偏差 %+.1f 分 (%+.2fpp)"
              % (est, 100 * est / full, est - T, 100 * (est - T) / full))
    print("\n  按赛制：")
    for bo in (1, 3, 5):
        if bo in per_bo:
            g, f2 = per_bo[bo]
            print("    bo%d  实得 %3d / %3d = %5.1f%%" % (bo, g, f2, 100 * g / f2))
    json.dump({"total": T, "max": full, "breakdown": tot,
               "per_bo": per_bo}, open(os.path.join(HERE, "lgc_score.json"), "w",
                                       encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n已写 lgc_score.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
