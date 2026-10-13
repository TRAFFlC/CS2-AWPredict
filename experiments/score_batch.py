"""两轮累计计分（官方计分器 014-scorer/scorer.py），并做置信度校准分析。
(teamA, teamB) 顺序 = 冻结预测中的 A/B 顺序；truth 分数按同序录入。
"""
import io, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # scorer.py 与本模块同目录（仓库布局）
import scorer as SC

MAPS = {"dust2": "Dust2", "dustii": "Dust2", "dust ii": "Dust2",
        "ancient": "Ancient", "anubis": "Anubis", "cache": "Cache",
        "inferno": "Inferno", "mirage": "Mirage", "nuke": "Nuke",
        "overpass": "Overpass", "train": "Train",
        "沙二": "Dust2", "沙2": "Dust2", "沙": "Dust2",
        "遗迹": "Ancient", "小镇": "Mirage", "迷城": "Inferno",
        "死城": "Cache", "核子": "Nuke", "阿努比斯": "Anubis",
        "死服": "Overpass", "死亡游乐园": "Overpass", "火车": "Train"}

def nm(x):
    k = x.strip().lower().replace("_", " ")
    if k in MAPS:
        return MAPS[k]
    c = k.replace(" ", "")
    return MAPS.get(c, x.strip().title())

TRUTH = {
"r1": [
    ("Falcons", "TYLOO",         "Falcons",       (2, 0), ["Mirage", "Anubis", "Nuke"],    "niko"),
    ("Legacy",  "PARIVISION",    "PARIVISION",    (1, 2), ["Dust2", "Inferno", "Ancient"],  "jame"),
    ("Vitality","1win",          "Vitality",      (2, 0), ["Dust2", "Cache", "Inferno"],   "zywoo"),
    ("Spirit",  "ShindeN",       "Spirit",        (2, 1), ["Dust2", "Nuke", "Mirage"],     "donk"),
    ("Aurora",  "9z",            "Aurora",        (2, 0), ["Dust2", "Mirage", "Nuke"],     "wicadia"),
    ("MOUZ",    "M80",           "MOUZ",          (2, 0), ["Cache", "Mirage", "Inferno"],  "torzsi"),
    ("G2",      "Natus Vincere", "Natus Vincere", (0, 2), ["Cache", "Inferno", "Mirage"],   "w0nderful"),
    ("FURIA",   "BetBoom",       "FURIA",         (2, 0), ["Mirage", "Cache", "Nuke"],     "yekindar"),
],
"r2": [
    ("ShindeN", "Legacy",        "Legacy",        (0, 2), ["Nuke", "Mirage", "Dust2"],     "latto"),
    ("Spirit",  "PARIVISION",    "Spirit",        (2, 1), ["Ancient", "Dust2", "Anubis"],  "donk"),
    ("9z",      "TYLOO",         "9z",            (2, 0), ["Nuke", "Mirage", "Inferno"],   "luchov"),
    ("Falcons", "Aurora",        "Falcons",       (2, 0), ["Mirage", "Dust2", "Anubis"],   "m0nesy"),
    ("G2",      "1win",          "1win",          (2, 1), ["Dust2", "Ancient", "Mirage"],  "belchonokk"),
    ("FURIA",   "MOUZ",          "MOUZ",          (0, 2), ["Cache", "Mirage", "Dust2"],    "xertion"),
    ("Vitality","Natus Vincere", "Vitality",      (2, 1), ["Anubis", "Nuke", "Mirage"],    "mezii"),
    ("BetBoom", "M80",           "BetBoom",       (2, 0), ["Cache", "Anubis", "Mirage"],   "s1ren"),
],
"r3": [
    ("ShindeN", "G2",            "G2",            (0, 2), ["沙二", "遗迹", "小镇"],          "matys"),
    ("PARIVISION", "FURIA",      "FURIA",         (1, 2), ["遗迹", "迷城", "小镇"],          "yekindar"),
    ("Legacy",  "1win",          "1win",          (0, 2), ["沙二", "遗迹", "迷城"],          "artfrost"),
    ("9z",      "Natus Vincere", "Natus Vincere", (0, 2), ["核子", "死城", "小镇"],          "b1t"),
    ("M80",     "TYLOO",         "M80",           (2, 1), ["迷城", "死城", "anubis"],        "slaxz-"),
    ("Spirit",  "MOUZ",          "MOUZ",          (0, 2), ["沙二", "迷城", "核子"],          "xelex"),
    ("Vitality","Falcons",       "Vitality",      (2, 1), ["阿努比斯", "沙二", "迷城"],      "mezii"),
    ("Aurora",  "BetBoom",       "Aurora",        (2, 1), ["沙二", "迷城", "核子"],          "wicadia"),
],
"r4": [
    ("Spirit",  "1win",          "1win",          (0, 2), ["死城", "阿努比斯", "沙二"],      "ax1le"),
    ("G2",      "PARIVISION",    "PARIVISION",    (1, 2), ["遗迹", "阿努比斯", "小镇"],      "jame"),
    ("9z",      "BetBoom",       "9z",            (2, 0), ["沙二", "迷城", "核子"],          "dgt"),
    ("FURIA",   "Aurora",        "Aurora",        (1, 2), ["沙二", "迷城", "核子"],          "wicadia"),
    ("Legacy",  "M80",           "M80",           (1, 2), ["死城", "迷城", "沙二"],          "s1n"),
    ("Falcons", "Natus Vincere", "Falcons",       (2, 1), ["小镇", "阿努比斯", "迷城"],      "Niko"),
],
"r5": [
    ("FURIA",   "9z",            "FURIA",         (2, 0), ["核子", "小镇", "迷城"],          "molodoy"),
    ("PARIVISION", "Natus Vincere", "PARIVISION", (2, 1), ["小镇", "死城", "迷城"],          "xielo"),
    ("Spirit",  "M80",           "Spirit",        (2, 0), ["沙二", "死城", "迷城"],          "sh1ro"),
],
"r6": [
    ("1win",    "Aurora",        "Aurora",        (0, 2), ["沙二", "阿努比斯", "迷城"],      "jimpphat"),
    ("Vitality","PARIVISION",    "Vitality",      (2, 0), ["小镇", "沙二", "阿努比斯"],      "zywoo"),
    ("FURIA",   "MOUZ",          "MOUZ",          (0, 2), ["核子", "小镇", "遗迹"],          "pr"),
    ("Falcons", "Spirit",        "Spirit",        (0, 2), ["迷城", "阿努比斯", "沙二"],      "donk"),
],
"r7": [
    ("Aurora",  "Vitality",      "Vitality",      (2, 1), ["阿努比斯", "死城", "迷城"],      "zywoo"),
    ("MOUZ",    "Spirit",        "MOUZ",          (2, 1), ["沙二", "迷城", "核子"],          "spinx"),
    ("Aurora",  "Spirit",        "Aurora",        (2, 0), ["迷城", "阿努比斯", "核子"],      "wicadia"),
],
"r8": [
    ("MOUZ",    "Vitality",      "Vitality",      (0, 3), ["死城", "小镇", "沙二", "核子", "迷城"], "zywoo"),
],
}

FILES = {"r1": "predictions_2026-10-12.json",
         "r2": "predictions_2026-10-12_r2.json",
         "r3": "predictions_2026-10-12_r3.json",
         "r4": "predictions_2026-10-12_r4.json",
         "r5": "predictions_2026-10-12_r5.json",
         "r6": "predictions_2026-10-12_r6.json",
         "r7": "predictions_2026-10-12_r7.json",
         "r8": "predictions_2026-10-12_r8.json"}
cum = dict(winner=0, score=0, maps=0, mvp=0)
cum_mx = 0
hist = []
self_est = 0.0

for rnd in ("r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8"):
    doc = json.load(io.open(FILES[rnd], encoding="utf-8"))
    PRED = {(m["A"], m["B"]): m for m in doc["matches"]}
    self_est += sum(m["exp_points"] for m in doc["matches"])
    rt = dict(winner=0, score=0, maps=0, mvp=0)
    rmx = 0
    print("=== %s ===" % rnd.upper())
    print("%-12s %-14s %-4s %-4s %-5s %-4s %4s %4s  %s" %
          ("A", "B", "胜方", "比分", "选图", "MVP", "得分", "/45", "命中图"))
    for a, b, winner, score, maps, mvp in TRUTH[rnd]:
        p = PRED[(a, b)]
        maps = [nm(m) for m in maps]
        truth = SC.Truth(teamA=a, teamB=b, winner=winner, score=score, maps=maps,
                         mvp=mvp, best_of=(5 if len(maps) >= 5 else 3), maps_mode="veto")
        pred = SC.Prediction(winner=p["winner"], score=tuple(p["score"]),
                             maps=list(p["maps"]), mvp=p["mvp"])
        br = SC.score_match(pred, truth)
        m = SC.max_total(truth)
        pts = br.winner + br.score + br.maps + br.mvp
        rmx += m
        for k in rt:
            rt[k] += getattr(br, k)
        for k in cum:
            cum[k] += getattr(br, k)
        cum_mx += m
        conf = 100 * max(p["win_prob"].values())
        hist.append((conf, br.winner > 0))
        print("%-12s %-14s %-4s %-4s %-5s %-4s %4d %4d  %-22s conf=%.1f%%" %
              (a, b, "+" if br.winner else "x", "+" if br.score else "x",
               ("%d张" % (br.maps // 5)) if br.maps else "x", "+" if br.mvp else "x",
               pts, m, "/".join(br.maps_hit) or "-", conf))
    print("  轮合计 %d / %d = %.1f%%   [胜方 %d 比分 %d 选图 %d MVP %d]\n" %
          (sum(rt.values()), rmx, 100.0 * sum(rt.values()) / rmx,
           rt["winner"], rt["score"], rt["maps"], rt["mvp"]))

print("=" * 84)
nm_ = len(hist)
maps_max = cum_mx - 30 * nm_
print("场次 %d（其中 bo5 %d）" % (nm_, sum(1 for t in TRUTH["r8"] for _ in [0] if len(t[4]) >= 5)))
print("累计：胜方 %d/%d  比分 %d/%d  选图 %d/%d  MVP %d/%d"
      % (cum["winner"], 10 * nm_, cum["score"], 10 * nm_,
         cum["maps"], maps_max, cum["mvp"], 10 * nm_))
for k, d in (("winner", 10 * nm_), ("score", 10 * nm_), ("maps", maps_max), ("mvp", 10 * nm_)):
    print("       %-6s %6.2f%%" % (k, 100.0 * cum[k] / d))
s = sum(cum.values())
print("累计总分 %d / %d = %.2f%%   （主理人 890 / 1855 = %.2f%%，差 %+d 分）"
      % (s, cum_mx, 100.0 * s / cum_mx, 100.0 * 890 / 1855, s - 890))
print("模型五轮自估合计 %.1f / %d = %.1f%%  → 实际 %+.1f 分（%+.1f%%）"
      % (self_est, cum_mx, 100.0 * self_est / cum_mx, s - self_est,
         100.0 * (s - self_est) / self_est))

ok = [c for c, w in hist if w]
bad = [c for c, w in hist if not w]
print("校准：猜对的 %d 场平均置信 %.1f%%；猜错的 %d 场平均置信 %.1f%%"
      % (len(ok), sum(ok) / len(ok), len(bad), sum(bad) / len(bad)))
print("      错的五场置信度 " + " / ".join("%.1f%%" % c for c in sorted(bad)))
