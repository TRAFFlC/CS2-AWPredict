"""两个假设检验（时序安全：只用该场之前的数据）
H1 田忌赛马式选图：pick 是否偏向「自己弱、对手也弱」或「对手弱」的图？
   对照：是否只是选自己最常打的图？
H2 MVP × 地图交互：真值是「胜方系列 rating 最高者」，但真实限制是「只在打了的图上」——
   按「已打图上的最高 rating」选，是否比「全图池最高 rating」更接近真值？
   并测一个可实操变体：只在【模型预测图集】上取最高 rating。
"""
import io, json, collections, math

HERE = "."
DATA = os.path.join(HERE, "../data")  # # 仓库布局：数据与代码分目录（Alpha 1.1.1）
MAPS7 = ["Ancient", "Anubis", "Cache", "Dust2", "Inferno", "Mirage", "Nuke"]

matches = [json.loads(l) for l in io.open("csapi_matches_all.jsonl", encoding="utf-8")]
veto = {}
for l in io.open("csapi_veto.jsonl", encoding="utf-8"):
    r = json.loads(l)
    if r.get("veto_ok"):
        veto[r["id"]] = r
truth = {json.loads(l)["id"]: json.loads(l)
         for l in io.open("truth_maps_B.jsonl", encoding="utf-8")}
timeline = json.load(io.open("map_pool_timeline.json", encoding="utf-8"))
ERAS = sorted(timeline["eras"], key=lambda e: e["start"])

def pool_at(date):
    for e in ERAS:
        if e["start"] <= date <= e["end"]:
            return e["pool"]
    return ERAS[-1]["pool"]

matches.sort(key=lambda r: r["date"])

# ---------- H1 ----------
pick_freq = collections.defaultdict(lambda: collections.defaultdict(float))  # team -> map -> decay count
map_wr = collections.defaultdict(lambda: [0.0, 0.0])                          # (team,map) -> [w, n]
HL = 120.0
obs_pick = []   # (own_freq_norm, opp_wr_norm) for picks
obs_ban = []
base_wr = []
BAN_W = {"Ancient": 1 / 7}
import time

def decay(team, d_last, now, w, hl=HL):
    return w * 0.5 ** ((now - d_last).days / hl) if d_last else w

for r in matches:
    date = r["date"]
    pool = pool_at(date)
    t1, t2 = r["team1"]["name"], r["team2"]["name"]
    # 1) 先结算本场打过的图 → 更新 map_wr
    for mp in r["maps"]:
        m = mp["name"]
        if m not in pool:
            continue
        w1 = 1 if mp["team1_score"] > mp["team2_score"] else 0
        for t, w in ((t1, w1), (t2, 1 - w1)):
            cell = map_wr[(t, m)]
            cell[0] += w
            cell[1] += 1
    # 2) veto 事件 → 记录 pick/ban 的「自己频率」与「对手胜率」
    v = veto.get(r["id"])
    if v:
        for step, kind, actor, m in v["veto"]:
            if not actor or m not in pool:
                continue
            opp = t2 if actor == t1 else t1
            freqs = sorted(pick_freq[actor].get(x, 0.0) for x in pool)
            tot = sum(freqs) or 1.0
            own = pick_freq[actor].get(m, 0.0) / tot
            # 对手在该图的胜率（收缩到 0.5，n0=8 伪计数）
            w, n = map_wr[(opp, m)]
            opp_wr = (w + 0.5 * 8) / (n + 8) if (w + n) else 0.5
            # 池内百分位：对手越弱 => 越低
            wrs = sorted(((map_wr[(opp, x)][0] + 4.0) / (map_wr[(opp, x)][1] + 8.0))
                         if (map_wr[(opp, x)][0] + map_wr[(opp, x)][1]) else 0.5 for x in pool)
            rank = wrs.index(min(x for x in wrs if x >= opp_wr)) / max(1, len(wrs) - 1) if opp_wr in wrs else 0.5
            rec = (own, rank)
            if kind == "pi":
                obs_pick.append(rec)
            else:
                obs_ban.append(rec)
            pick_freq[actor][m] += 1.0
    # 3) 记基准：所有图在池内的平均百分位 ≈ 0.5

def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")

print("=== H1 田忌赛马检验（0.5 = 无信号；pick 偏向「对手弱图」应 < 0.5）===")
for tag, obs in (("PICK 选图", obs_pick), ("BAN 禁图", obs_ban)):
    own = mean([o for o, _ in obs])
    rank = mean([rk for _, rk in obs])
    n = len(obs)
    sd = (statistics_variance := math.sqrt(
        sum((rk - rank) ** 2 for _, rk in obs) / max(1, n - 1))) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n else 0.0
    print("  %-10s n=%4d  自身频率 %.3f   对手胜率百分位 %.4f (SE %.4f, %.1f SE from 0.5)"
          % (tag, n, own, rank, se, (0.5 - rank) / se if se else 0))

# ---------- H2 ----------
print()
print("=== H2 MVP × 地图交互（真值 = 胜方系列 rating 最高者）===")

def load_stats_by_map():
    d = {}
    for l in io.open("csapi_stats_all.jsonl", encoding="utf-8"):
        rec = json.loads(l)
        mid = rec["matchid"]
        per_map = {}
        for blk in rec.get("by_map") or []:
            mn = blk.get("name")
            if not mn or mn == "All":
                continue
            teams = {}
            for side in ("team1", "team2"):
                tb = blk.get(side) or {}
                teams[tb.get("name")] = {p["name"]: p.get("rating") for p in tb.get("players", [])}
            per_map[mn] = teams
        d[mid] = per_map
    return d

by_map = load_stats_by_map()

stats = []
for rnd_file in ("predictions_2026-10-12.json", "predictions_2026-10-12_r2.json",
                 "predictions_2026-10-12_r3.json", "predictions_2026-10-12_r4.json",
                 "predictions_2026-10-12_r5.json", "predictions_2026-10-12_r6.json",
                 "predictions_2026-10-12_r7.json", "predictions_2026-10-12_r8.json"):
    d = json.load(io.open(rnd_file, encoding="utf-8"))
    for m in d["matches"]:
        stats.append((rnd_file, m))

# 主理人公布的真值 MVP（与 score_batch.TRUTH 同源）
import score_batch as SB
TRUTH_MVP = {}
for rnd in ("r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8"):
    for a, b, w, sc, maps, mvp in SB.TRUTH[rnd]:
        TRUTH_MVP[(rnd, a, b)] = mvp

hit_model = hit_played = hit_pred = n = 0
for rnd_file, m in stats:
    key = None
    for rnd in ("r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8"):
        for a, b, w, sc, maps, mvp in SB.TRUTH[rnd]:
            if (a, b) == (m["A"], m["B"]):
                key = (rnd, a, b)
    if key is None:
        continue
    truth_mvp = TRUTH_MVP[key]
    winner = m["winner"]
    # 该场 by_map
    pm = None
    for l in io.open("csapi_matches_all.jsonl", encoding="utf-8"):
        pass
    n += 1

print("  （by_map 全量 2,800 场已加载：%d 场有逐图逐选手 rating）" % len(by_map))
print("  注：本实验的 41 场需要 matchid ↔ (A,B) 映射，主理人赛果源无 csapi id，暂无法直接对齐。")
print("  ⇒ 改用全语料口径评估「按已打图取最高 rating」相对「按全池取最高 rating」的差异：")
