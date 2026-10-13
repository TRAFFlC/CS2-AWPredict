"""裂变天地 S3 两日赛果计分（通用：按 --date 选当日真值）。

官方计分器计分，读法 B（bo3 恒 3 张含未打决胜图），满分 45。

中文图名 → 英文：遗迹=Dust2 迷城=Mirage 沙二=Anubis 阿努比斯=Anubis
                 死城=Cache 核子=Nuke 小镇=Inferno 远古=Ancient
（「沙二」与「阿努比斯」同图）

09-08 主理人特别标注：FaZe vs Alliance 的 MVP 是 **Alliance 的 twist**，
不是 FaZe 的 twistzz（两队都有 twist 系选手）。
"""
import io, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, HERE)  # scorer.py 与本模块同目录（仓库布局）
import scorer as SC

ZH2EN = {"遗迹": "Dust2", "迷城": "Mirage", "沙二": "Anubis", "阿努比斯": "Anubis",
         "死城": "Cache", "核子": "Nuke", "小镇": "Inferno", "远古": "Ancient"}

TRUTHS = {
    "2026-09-08": [
        ("Legacy", "magic", "Legacy", 2, 1, ["遗迹", "沙二", "迷城"], "dumau"),
        ("9z", "5star", "9z", 2, 0, ["死城", "核子", "小镇"], "dgt"),
        ("The MongolZ", "MIBR", "MIBR", 1, 2, ["遗迹", "迷城", "小镇"], "venomzera"),
        ("FaZe", "Alliance", "Alliance", 0, 2, ["遗迹", "核子", "阿努比斯"], "twist"),
        ("BetBoom", "BIG", "BetBoom", 2, 0, ["死城", "核子", "遗迹"], "d1ledez"),
        ("PARIVISION", "TYLOO", "PARIVISION", 2, 0, ["小镇", "遗迹", "阿努比斯"], "fl1t"),
        ("FURIA", "GamerLegion", "FURIA", 2, 1, ["死城", "迷城", "沙二"], "molodoy"),
        ("G2", "Astralis", "Astralis", 1, 2, ["遗迹", "迷城", "小镇"], "jabbi"),
    ],
    "2026-09-09": [
        ("magic", "FaZe", "magic", 2, 1, ["遗迹", "沙二", "死城"], "masvai"),
        ("5star", "The MongolZ", "5star", 2, 0, ["小镇", "核子", "遗迹"], "rate"),
        ("9z", "MIBR", "MIBR", 0, 2, ["迷城", "核子", "小镇"], "nqz"),
        ("Legacy", "Alliance", "Legacy", 2, 0, ["小镇", "沙二", "遗迹"], "latto"),
        ("BIG", "G2", "G2", 1, 2, ["死城", "迷城", "遗迹"], "nertz"),
        ("TYLOO", "GamerLegion", "TYLOO", 2, 0, ["遗迹", "迷城", "死城"], "moseyuh"),
        ("BetBoom", "Astralis", "BetBoom", 2, 0, ["遗迹", "阿努比斯", "核子"], "boombl4"),
        ("PARIVISION", "FURIA", "FURIA", 0, 2, ["遗迹", "小镇", "迷城"], "kscerato"),
    ],
    "2026-09-10": [
        ("Alliance", "TYLOO", "Alliance", 2, 0, ["小镇", "遗迹", "阿努比斯"], "bobeksde"),
        ("9z", "G2", "G2", 1, 2, ["沙二", "遗迹", "小镇"], "heavygod"),
        ("magic", "PARIVISION", "magic", 2, 0, ["遗迹", "小镇", "迷城"], "tenzy"),
        ("Astralis", "5star", "5star", 1, 2, ["小镇", "遗迹", "死城"], "clouden"),
        ("Legacy", "FURIA", "Legacy", 2, 0, ["迷城", "遗迹", "沙二"], "latto"),
        ("MIBR", "BetBoom", "BetBoom", 0, 2, ["迷城", "阿努比斯", "核子"], "d1ledez"),
        ("Alliance", "magic", "Alliance", 2, 0, ["遗迹", "小镇", "沙二"], "twist"),
        ("G2", "5star", "G2", 2, 0, ["小镇", "遗迹", "阿努比斯"], "matys"),
    ],
    "2026-09-11": [
        ("FURIA", "G2", "G2", 0, 2, ["遗迹", "沙二", "迷城"], "heavygod"),
        ("MIBR", "Alliance", "MIBR", 2, 0, ["小镇", "死城", "遗迹"], "nqz"),
    ],
    "2026-09-12": [
        ("BetBoom", "G2", "G2", 1, 2, ["迷城", "阿努比斯", "遗迹"], "nertz"),
        ("Legacy", "MIBR", "Legacy", 2, 1, ["核子", "遗迹", "小镇"], "n1ssim"),
    ],
    "2026-09-13": [
        ("Legacy", "G2", "Legacy", 3, 0, ["遗迹", "迷城", "沙二", "小镇", "死城"], "latto"),
    ],
    # ---- StarLadder 秋季群星系列赛 ----
    "2026-09-17": [
        ("MOUZ", "NRG", "NRG", 1, 2, ["死城", "小镇", "沙二"], "nitr0"),
        ("Vitality", "magic", "Vitality", 2, 0, ["迷城", "沙二", "小镇"], "apex"),
        ("Natus Vincere", "Aurora", "Aurora", 0, 2, ["核子", "迷城", "死城"], "jimpphat"),
    ],
    "2026-09-18": [
        ("FURIA", "MIBR", "FURIA", 2, 1, ["小镇", "核子", "死城"], "kscerato"),
        ("MOUZ", "Natus Vincere", "MOUZ", 2, 1, ["死城", "小镇", "迷城"], "xertion"),
        ("magic", "MIBR", "MIBR", 1, 2, ["遗迹", "迷城", "小镇"], "insani"),
    ],
    "2026-09-19": [
        ("NRG", "Aurora", "Aurora", 0, 2, ["死城", "迷城", "核子"], "xantares"),
        ("Vitality", "FURIA", "Vitality", 2, 0, ["迷城", "沙二", "死城"], "zywoo"),
        ("MIBR", "NRG", "MIBR", 2, 0, ["死城", "小镇", "核子"], "lnz"),
        ("MOUZ", "FURIA", "FURIA", 1, 2, ["死城", "迷城", "核子"], "kscerato"),
        ("Aurora", "Vitality", "Aurora", 2, 1, ["阿努比斯", "小镇", "核子"], "jimpphat"),
    ],
    "2026-09-20": [
        ("MIBR", "FURIA", "FURIA", 0, 2, ["迷城", "核子", "小镇"], "yekindar"),
        ("Vitality", "FURIA", "Vitality", 2, 0, ["死城", "阿努比斯", "核子"], "zywoo"),
        ("Vitality", "Aurora", "Vitality", 3, 1, ["阿努比斯", "小镇", "迷城", "死城", "核子"], "zywoo"),
    ],
    # 09-20 决赛的真实 bo 未标注，首轮按 bo3 预测 ⇒ 选图项只报了 3 张。
    # 这里用 bo=5 重跑一遍，单独检验「bo 假设错误」的代价。
    "2026-09-20-bo5": [
        ("Vitality", "Aurora", "Vitality", 3, 1, ["阿努比斯", "小镇", "迷城", "死城", "核子"], "zywoo"),
    ],
    # ---- 冷门小赛事（T4：9 支队 rank #24~#190，无前16队）----
    "cold1": [
        ("Astralis", "Sangal", "Astralis", 2, 1, ["核子", "遗迹", "死城"], "ryu"),
        ("Sashi", "FOKUS", "FOKUS", 0, 2, ["沙二", "核子", "遗迹"], "jocab"),
        ("FOKUS", "Astralis", "Astralis", 0, 2, ["死城", "遗迹", "迷城"], "staehr"),
        ("Sashi", "Sangal", "Sangal", 0, 2, ["核子", "死城", "沙二"], "r4dyx"),
        ("Sangal", "FOKUS", "Sangal", 2, 1, ["沙二", "核子", "迷城"], "joey"),
        ("HEROIC", "Johnny Speeds", "HEROIC", 2, 0, ["死城", "沙二", "核子"], "brollan"),
        ("Virtus.pro", "BBL", "BBL", 0, 2, ["核子", "迷城", "遗迹"], "leakz"),
        ("HEROIC", "BBL", "HEROIC", 2, 0, ["核子", "小镇", "遗迹"], "martinezsa"),
    ],
    # ---- 冷门小赛事 第二天 ----
    # ⚠ 两场比分缺失（`phantom fnatic` / `100thieves astralis`）：
    #   胜方由 **MVP 反推**（MVP 必属胜方；jambo=fnatic，rain=100 Thieves，
    #   两者均在同日其他场次被证实），比分用 (1,1) 平局占位 ⇒ 计分器按 0 计（正确）。
    "cold2": [
        ("Johnny Speeds", "Virtus.pro", "Virtus.pro", 0, 2, ["死城", "沙二", "迷城"], "draken"),
        ("Johnny Speeds", "BBL", "BBL", 0, 2, ["迷城", "核子", "死城"], "salazar"),
        ("fnatic", "Phantom", "Phantom", 1, 2, ["小镇", "沙二", "迷城"], "nicoodoz"),
        ("100 Thieves", "EAC", "100 Thieves", 2, 0, ["迷城", "小镇", "核子"], "rain"),
        ("EAC", "fnatic", "fnatic", 1, 2, ["迷城", "阿努比斯", "遗迹"], "jambo"),
        ("100 Thieves", "Phantom", "100 Thieves", 2, 1, ["核子", "迷城", "小镇"], "sirah"),
        ("Phantom", "fnatic", "fnatic", 1, 1, ["遗迹", "阿努比斯", "小镇"], "jambo"),
        ("JiJieHao", "Eternal Fire", "JiJieHao", 2, 0, ["阿努比斯", "迷城", "遗迹"], "m1n1"),
        ("betclic", "ASTRAL", "betclic", 2, 1, ["核子", "遗迹", "迷城"], "prism"),
        ("JiJieHao", "betclic", "JiJieHao", 2, 0, ["迷城", "阿努比斯", "遗迹"], "0samas"),
        ("Eternal Fire", "ASTRAL", "Eternal Fire", 2, 0, ["小镇", "沙二", "死城"], "jottaaa"),
        ("Eternal Fire", "betclic", "betclic", 1, 2, ["死城", "阿努比斯", "遗迹"], "prism"),
        ("HEROIC", "fnatic", "HEROIC", 2, 1, ["死城", "迷城", "小镇"], "nilo"),
        ("100 Thieves", "BBL", "100 Thieves", 2, 0, ["死城", "迷城", "阿努比斯"], "rain"),
        ("Astralis", "betclic", "Astralis", 2, 1, ["核子", "遗迹", "死城"], "phzy"),
        ("JiJieHao", "Sangal", "Sangal", 0, 2, ["死城", "遗迹", "迷城"], "bnox"),
        ("100 Thieves", "Astralis", "100 Thieves", 1, 1, ["阿努比斯", "遗迹", "小镇"], "rain"),
        ("HEROIC", "Sangal", "HEROIC", 2, 1, ["死城", "沙二", "迷城"], "brollan"),
        ("100 Thieves", "HEROIC", "HEROIC", 1, 2, ["沙二", "遗迹", "小镇"], "brollan"),
    ],
}
# 我的预测文件用的队名顺序（A,B），与真值表的 (A,B) 对齐
PRED_ALIAS = {("The MongolZ", "5star"): ("5star", "The MongolZ")}


def main():
    date = sys.argv[1] if len(sys.argv) > 1 else "2026-09-08"
    TRUTH = TRUTHS[date]
    # 09-08 的预测文件是第一版命名 fissure_s3_predictions.json
    cands = ["fissure_%s_predictions.json" % date,
             "fissure_s3_predictions.json" if date == "2026-09-08" else None]
    ppath = next((os.path.join(HERE, c) for c in cands
                  if c and os.path.exists(os.path.join(HERE, c))), None)
    if not ppath:
        raise SystemExit("找不到 %s 的预测文件" % date)
    pred_doc = json.load(io.open(ppath, encoding="utf-8"))
    pm = {(p["A"], p["B"]): p for p in pred_doc}
    # 预测文件里 A/B 顺序可能与真值表不同，先建反查
    rev = {}
    for (A, B, W, sa, sb, zh, mvp) in TRUTH:
        if (A, B) in pm:
            rev[(A, B)] = pm[(A, B)]
        elif (B, A) in pm:
            rev[(A, B)] = pm[(B, A)]
        else:
            hit = [p for p in pred_doc if {p["A"], p["B"]} == {A, B}]
            if len(hit) != 1:
                raise SystemExit("无法匹配 %s vs %s" % (A, B))
            rev[(A, B)] = hit[0]

    print("== 裂变天地 S3 · %s 官方计分（读法 B，满分 45）==\n" % date)
    tot = {"winner": 0, "score": 0, "maps": 0, "mvp": 0}
    rows, out = [], []
    for A, B, W, sa, sb, zh, mvp in TRUTH:
        maps = [ZH2EN[m] for m in zh]
        bo = 5 if len(maps) == 5 else 3      # 决赛 bo5：满分 55、图集 5 张
        truth = SC.Truth(teamA=A, teamB=B, winner=W, score=(sa, sb), maps=maps,
                         mvp=mvp, best_of=bo, maps_mode="veto")
        p = rev[(A, B)]
        ps = tuple(int(x) for x in p["score"].split(":"))
        # 比分项要按**真值的 (A,B) 朝向**对齐，否则 2:0/0:2 会误判
        if p["A"] != A:
            ps = (ps[1], ps[0])
        pred = SC.Prediction(winner=p["winner"], score=ps,
                             maps=p["maps"].split(" / "), mvp=p["mvp"])
        bd = SC.score_match(pred, truth)
        for k in tot:
            tot[k] += getattr(bd, k)
        rows.append((A, B, W, p, bd))
        out.append({"A": A, "B": B, "winner": W, "score": [sa, sb],
                    "maps": maps, "mvp": mvp, "my_winner": p["winner"],
                    "my_score": list(ps), "my_maps": pred.maps, "my_mvp": pred.mvp,
                    "my_p": p["p_win"], "my_rank": {"A": p["rankA"], "B": p["rankB"]},
                    "pts": {"winner": bd.winner, "score": bd.score,
                            "maps": bd.maps, "mvp": bd.mvp}, "total": bd.total})
        print("  %-11s %d:%d %-11s │ 我判 %-11s %s │ 胜%s 比分%s 图%d/3 MVP%s │ +%d"
              % (A, sa, sb, B, p["winner"], "%d:%d" % ps, "✓" if bd.winner else "✗",
                 "✓" if bd.score else "✗", len(bd.maps_hit),
                 "✓" if bd.mvp else "✗", bd.total))
    n = len(TRUTH)
    mx = sum(55 if len(z) == 5 else 45 for _a, _b, _c, _sa, _sb, z, _m in TRUTH)
    nmap = sum(len(z) for _a, _b, _c, _sa, _sb, z, _m in TRUTH)
    T = sum(tot.values())
    est = sum(p["exp"] for p in (r[3] for r in rows))
    print("\n  胜方 %d/%d = %d    比分 %d/%d = %d    选图 %d/%d = %d    MVP %d/%d = %d"
          % (tot["winner"] // 10, n, tot["winner"], tot["score"] // 10, n,
             tot["score"], tot["maps"] // 5, nmap, tot["maps"],
             tot["mvp"] // 10, n, tot["mvp"]))
    print("  合计 %d / %d = %.2f%%    自估 %.1f ⇒ 偏差 %+.1f 分 (%+.1fpp)"
          % (T, mx, 100 * T / mx, est, est - T, 100 * (est - T) / mx))

    base_ok = 0
    for A, B, W, p, bd in rows:
        fav = A if p["rankA"] < p["rankB"] else B
        ok = (fav == W)
        base_ok += ok
        if p["winner"] != fav:
            print("    分歧 %s(#%d) vs %s(#%d)：我判 %s，热门 %s，实际 %s ⇒ %s"
                  % (A, p["rankA"], B, p["rankB"], p["winner"], fav, W,
                     "热门对，我错" if ok else "我对，热门错"))
    mw = tot["winner"] // 10
    print("    胜方：模型 %d/%d (%.1f%%)  vs  rank 热门 %d/%d (%.1f%%)  ⇒ %+.1fpp"
          % (mw, n, 100 * mw / n, base_ok, n, 100 * base_ok / n,
             100 * (mw - base_ok) / n))
    json.dump(out, open(os.path.join(HERE, "fissure_%s_results.json" % date), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n已写 fissure_%s_results.json" % date)
    return 0


if __name__ == "__main__":
    sys.exit(main())
