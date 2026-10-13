"""H2 MVP × 地图交互：按「已打图/预测图集」取最高 rating，与 Q-008 真值口径的差距"""
import io, json, collections

# 真值口径（Q-008）：胜方系列 overall rating 最高者
def load_overall():
    d = {}
    for l in io.open("csapi_stats_all.jsonl", encoding="utf-8"):
        r = json.loads(l)
        mid = r["matchid"]
        blk = (r.get("overall") or [{}])[0]
        t1, t2 = blk.get("team1") or {}, blk.get("team2") or {}
        d[mid] = {
            t1.get("name"): {p["name"]: p.get("rating") for p in (t1.get("players") or [])},
            t2.get("name"): {p["name"]: p.get("rating") for p in (t2.get("players") or [])},
        }
    return d

def load_by_map():
    d = {}
    for l in io.open("csapi_stats_all.jsonl", encoding="utf-8"):
        r = json.loads(l)
        mid = r["matchid"]
        pm = {}
        for blk in r.get("by_map") or []:
            mn = blk.get("name")
            if not mn or mn == "All":
                continue
            pm[mn] = {}
            for side in ("team1", "team2"):
                tb = blk.get(side) or {}
                pm[mn][tb.get("name")] = {p["name"]: p.get("rating")
                                          for p in (tb.get("players") or [])}
        d[mid] = pm
    return d

matches = {json.loads(l)["id"]: json.loads(l) for l in io.open("csapi_matches_all.jsonl", encoding="utf-8")}
truth = {json.loads(l)["id"]: json.loads(l) for l in io.open("truth_maps_B.jsonl", encoding="utf-8")}
overall = load_overall()
bymap = load_by_map()

def argmax(d):
    return max(d, key=lambda k: (d[k] if d[k] is not None else -9)) if d else None

n = agree_played = agree_fullB = 0
n_played_maps = 0
mvcs = collections.Counter()
for mid, m in matches.items():
    t = truth.get(mid)
    if not t or t.get("maps_mode") != "veto":
        continue
    w = t["winner"]
    ov = overall.get(mid, {}).get(w)
    if not ov:
        continue
    lab = argmax(ov)
    if lab is None:
        continue
    n += 1
    mvcs[lab] += 1
    pm = bymap.get(mid) or {}
    # (a) 只在「实际打过」的图上
    played = t.get("maps_played") or []
    rates_played = [pm[x][w] for x in played if x in pm and w in pm[x]]
    n_played_maps += len(rates_played)
    if rates_played and argmax({k: v for x in played if x in pm and w in pm[x]
                                for k, v in pm[x][w].items()}) == lab:
        agree_played += 1
    # (b) 只在「读法B 全图集」上（= 预测时手上的信息）
    fullB = t.get("maps") or []
    rates_b = [pm[x][w] for x in fullB if x in pm and w in pm[x]]
    if rates_b and argmax({k: v for x in fullB if x in pm and w in pm[x]
                           for k, v in pm[x][w].items()}) == lab:
        agree_fullB += 1

print("=== H2 MVP 地图条件化（n=%d 场，读法B 可用）===" % n)
print("  Q-008 真值 = 胜方 overall rating 最高者")
print("  (a) 改成「已打图上的最高 rating」  一致率 %.2f%%   ⇒ 偏离 %.2f%%"
      % (100.0 * agree_played / n, 100 - 100.0 * agree_played / n))
print("  (b) 改成「读法B 图集上的最高 rating」（预测时可得）一致率 %.2f%%  ⇒ 偏离 %.2f%%"
      % (100.0 * agree_fullB / n, 100 - 100.0 * agree_fullB / n))
print("  平均每场已打图 %d 张的逐图 rating" % (n_played_maps / n))
print()
print("  最常成为真值 MVP 的选手 Top8：")
for p, c in mvcs.most_common(8):
    print("    %-14s %4d 场 (%.1f%%)" % (p, c, 100.0 * c / n))
