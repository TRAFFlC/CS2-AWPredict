"""比分项事实核查：bo3 的 2:0 / 2:1 真实占比 + 按「强弱」分层。

修正主理人 Q-020 的前提：bo3、胜方赢单图概率 p 时
    P(2:0) = p²,  P(2:1) = 2p(1−p),  比值 = 2(1−p)/p
⇒ 比值在 p=0.5 处**取最大值 2.0**，达不到 3。且真实 p > 0.5 ⇒ 2:0 反而更常见。
"""
import io, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import model_v2 as M
import veto_model as VM
import split as SPLIT
import pool_experiment as PE

rows = SPLIT.load(M.MATCHES)
exc = SPLIT.load_exclude(M.EXCLUDE) if os.path.exists(M.EXCLUDE) else None
buckets, _ = SPLIT.split(rows, purge=True, exclude=exc)
split_of = {}
for b in ("train", "valid", "test"):
    for r in buckets[b]:
        split_of[r["id"]] = b
truths = {t["id"]: t for t in M.load_jsonl(M.TRUTH)}
veto = VM.load_veto()
stats = M.load_stats()
ms = M.load_stats_map()
eras, by_pool = PE.load_pool_eras()
tl = json.load(io.open(PE.TIMELINE, encoding="utf-8"))

usable = [r for r in rows
          if isinstance(r.get("best_of"), int) and r["best_of"] in (1, 3, 5)
          and r["id"] in split_of]
usable.sort(key=lambda r: (r["date"], r["id"]))


def pool_of(r):
    t = truths.get(r["id"]) or {}
    return list(t["pool"]) if t.get("pool") else \
        list(PE.era_and_pool(r["date"], [], eras, by_pool, tl)[1])


def walk():
    s, h = M.State(), VM.MapHistory()
    recs = []
    i = 0
    while i < len(usable):
        d = usable[i]["date"]
        s.advance(d)
        h.advance(d)
        grp = []
        while i < len(usable) and usable[i]["date"] == d:
            grp.append(usable[i])
            i += 1
        for r in grp:
            t = truths.get(r["id"]) or {}
            bo = t.get("bo") or r.get("best_of") or 3
            tA, tB = r["team1"]["name"], r["team2"]["name"]
            w = r["winner"]["name"]
            ea, _ = s.elo_of(tA)
            eb, _ = s.elo_of(tB)
            sc = (r["team1"]["score"], r["team2"]["score"])
            recs.append({
                "split": split_of[r["id"]], "bo": bo, "w_fav": (ea >= eb) == (w == tA),
                "sweep": 1 if (max(sc) == 2 and min(sc) == 0) else 0,
                "gap": abs(ea - eb),
                "maps_played": sum(sc),
            })
        for r in grp:
            tA, tB = r["team1"]["name"], r["team2"]["name"]
            mres = [(m["name"], m["team1_score"], m["team2_score"])
                    for m in (r.get("maps") or [])]
            t = truths.get(r["id"]) or {}
            s.observe(d, tA, tB, r["winner"]["name"], mres,
                      stats.get(r["id"], {}), ms.get(r["id"]))
            h.observe_match(tA, tB, pool_of(r), t.get("maps") or [],
                             veto.get(r["id"]), mres)
    return recs


R = walk()


def pct(x):
    return 100.0 * x


print("== 1) bo3 的比分分布（全体）==")
for b in ("train", "valid", "test"):
    sel = [x for x in R if x["bo"] == 3 and x["split"] == b]
    n = len(sel)
    sw = sum(x["sweep"] for x in sel)
    print("  %-6s bo3 %4d 场   通吃 2:0 = %5.2f%%   2:1 = %5.2f%%   比值 P(2:1)/P(2:0) = %.3f"
          % (b, n, pct(sw / n), pct((n - sw) / n), (n - sw) / sw))
allb3 = [x for x in R if x["bo"] == 3]
n = len(allb3)
sw = sum(x["sweep"] for x in allb3)
print("  %-6s bo3 %4d 场   通吃 2:0 = %5.2f%%   2:1 = %5.2f%%   比值 = %.3f"
      % ("合计", n, pct(sw / n), pct((n - sw) / n), (n - sw) / sw))

print("\n== 2) 按「胜方是不是赛前 Elo 更高的队」分层（bo3 全体）==")
for tag, sel in (("胜方 = 强队", [x for x in allb3 if x["w_fav"]]),
                 ("胜方 = 弱队", [x for x in allb3 if not x["w_fav"]])):
    m = len(sel)
    k = sum(x["sweep"] for x in sel)
    print("  %-12s n=%4d   通吃率 %5.2f%%   2:1 率 %5.2f%%"
          % (tag, m, pct(k / m), pct((m - k) / m)))

print("\n== 3) 再按 Elo 差距分层（强队赢的场次里，差距越大越容易通吃？）==")
wins = [x for x in allb3 if x["w_fav"]]
wins.sort(key=lambda x: x["gap"])
q = len(wins) // 4
for i in range(4):
    seg = wins[i * q:(i + 1) * q] if i < 3 else wins[3 * q:]
    k = sum(x["sweep"] for x in seg)
    print("  Elo 差四分位 Q%d  [%.0f, %.0f)  n=%4d   通吃率 %5.2f%%"
          % (i + 1, seg[0]["gap"], seg[-1]["gap"], len(seg), pct(k / len(seg))))

print("\n== 4) 理论对照：P(2:1)/P(2:0) = 2(1−p)/p ==")
for p in (0.55, 0.60, 0.65, 0.70):
    print("  单图胜率 p=%.2f ⇒ 通吃 %.1f%% / 2:1 %.1f%%   比值 %.2f"
          % (p, 100 * p * p, 100 * 2 * p * (1 - p), 2 * (1 - p) / p))
print("  ⇒ 比值随 p 增大而**下降**；p=0.5 时已是最大值 2.0，主理人说的「三倍」达不到。")
