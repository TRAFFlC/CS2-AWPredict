"""H1 田忌赛马式选图检验（时序安全）"""
import io, json, collections, math

matches = [json.loads(l) for l in io.open("csapi_matches_all.jsonl", encoding="utf-8")]
veto = {}
for l in io.open("csapi_veto.jsonl", encoding="utf-8"):
    r = json.loads(l)
    if r.get("veto_ok"):
        veto[r["id"]] = r
tl = json.load(io.open("map_pool_timeline.json", encoding="utf-8"))
ERAS = sorted(tl["eras"], key=lambda e: e["start"])

def pool_at(d):
    for e in ERAS:
        if e["start"] <= d <= e["end"]:
            return e["pool"]
    return ERAS[-1]["pool"]

matches.sort(key=lambda r: r["date"])
pick_freq = collections.defaultdict(lambda: collections.defaultdict(float))
map_wr = collections.defaultdict(lambda: [0.0, 0.0])
N0 = 8.0
obs = {"pi": [], "re": []}

def pct_rank(vals, x):
    """x 在 vals 中的百分位（0=最小=最弱）"""
    lo = sum(1 for v in vals if v < x)
    eq = sum(1 for v in vals if v == x)
    return (lo + (eq - 1) / 2.0) / max(1, len(vals) - 1)

for r in matches:
    pool = pool_at(r["date"])
    t1, t2 = r["team1"]["name"], r["team2"]["name"]
    for mp in r["maps"]:
        m = mp["name"]
        if m not in pool:
            continue
        w1 = 1 if mp["team1_score"] > mp["team2_score"] else 0
        for t, w in ((t1, w1), (t2, 1 - w1)):
            c = map_wr[(t, m)]
            c[0] += w
            c[1] += 1
    v = veto.get(r["id"])
    if not v:
        continue
    for step, kind, actor, m in v["veto"]:
        if not actor or m not in pool:
            continue
        opp = t2 if actor == t1 else t1
        wrs = []
        for x in pool:
            w, n = map_wr[(opp, x)]
            wrs.append((w + 0.5 * N0) / (n + N0) if (w + n) else 0.5)
        opp_pct = pct_rank(wrs, (map_wr[(opp, m)][0] + 0.5 * N0) /
                           (map_wr[(opp, m)][1] + N0) if sum(map_wr[(opp, m)]) else 0.5)
        fr = sorted(pick_freq[actor].get(x, 0.0) for x in pool)
        own_pct = pct_rank(fr, pick_freq[actor].get(m, 0.0))
        obs[kind].append((own_pct, opp_pct))
        pick_freq[actor][m] += 1.0

def stat(name, rows, idx):
    xs = [r[idx] for r in rows]
    n = len(xs)
    mu = sum(xs) / n
    sd = math.sqrt(sum((x - mu) ** 2 for x in xs) / (n - 1))
    se = sd / math.sqrt(n)
    z = (mu - 0.5) / se
    return name, n, mu, se, z

print("=== H1 选图行为检验（百分位 0.5 = 无信号）===")
print("%-14s %6s %10s %10s %12s %12s" % ("动作", "n", "自身频率", "SE", "对手胜率", "z vs 0.5"))
for kind, nm in (("pi", "PICK 选图"), ("re", "BAN 禁图")):
    _, n, a, sea, za = stat(nm, obs[kind], 0)
    _, _, b, seb, zb = stat(nm, obs[kind], 1)
    print("%-14s %6d %10.4f %10.4f %12.4f %12.1f" % (nm, n, a, sea, b, zb))
print()
print("读法：")
print("  · PICK 的「对手胜率」显著 < 0.5 ⇒ 存在「专挑对手弱图」的田忌赛马信号")
print("  · BAN 的「对手胜率」显著 > 0.5 ⇒ 禁图针对对手强图（经典 ban 逻辑）")
print("  · PICK 的「自身频率」显著 > 0.5 ⇒ 选图仍主要由自身习惯驱动")
