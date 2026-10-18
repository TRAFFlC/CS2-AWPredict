"""把 41 场真实赛果导出为可复用的结构化数据（语料兼容 schema）+ 统计摘要。
输出目录：MiMo 私有区（不写 Deepseek 工作区）。
"""
import io, json, os, sys, hashlib

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = r"E:\world\another\data\Mimo\MiMo-V2.6-Flash\2026-10-12"
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, HERE)
import score_batch as SB   # 复用 TRUTH / MAPS / FILES

SCORER_DIR = HERE

rows = []
for rnd in ("r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8"):
    doc = json.load(io.open(SB.FILES[rnd], encoding="utf-8"))
    PRED = {(m["A"], m["B"]): m for m in doc["matches"]}
    for idx, (a, b, winner, score, maps, mvp) in enumerate(SB.TRUTH[rnd], 1):
        maps = [SB.nm(m) for m in maps]
        bo = 5 if len(maps) >= 5 else 3
        sa, sb_ = score
        played_n = max(score)
        played = maps if played_n == len(maps) else None   # 通吃时打过哪几张未知
        p = PRED.get((a, b), {})
        rows.append({
            "id": "live-%s-%02d" % (rnd, idx),
            "date": None,
            "event": "主理人赛事（2026-10-12 揭盲）",
            "best_of": bo,
            "team1": {"name": a, "score": sa},
            "team2": {"name": b, "score": sb_},
            "winner": winner,
            "maps_full_B": maps,
            "maps_played": played,
            "maps_played_n": played_n,
            "mvp": mvp,
            "src": "主理人公布 2026-10-12",
            "needs_date": True,
            "pred_frozen": {
                "winner": p.get("winner"), "score": p.get("score"),
                "maps": p.get("maps"), "mvp": p.get("mvp"),
                "win_prob": p.get("win_prob"), "exp_points": p.get("exp_points"),
                "file": SB.FILES[rnd],
            },
        })

path = os.path.join(OUT_DIR, "live_tournament_results.jsonl")
io.open(path, "w", encoding="utf-8", newline="\n").write(
    "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
h = hashlib.sha256(io.open(path, "rb").read()).hexdigest()

sweeps = sum(1 for r in rows if min(r["team1"]["score"], r["team2"]["score"]) == 0)
full = sum(1 for r in rows if r["maps_played_n"] == len(r["maps_full_B"]))

print("写出 %s" % path)
print("  场次 %d   sha256=%s" % (len(rows), h[:16]))
print("  通吃 %d 场 = %.1f%%   非通吃 %d 场 = %.1f%%"
      % (sweeps, 100.0 * sweeps / len(rows), len(rows) - sweeps,
         100.0 * (len(rows) - sweeps) / len(rows)))
print("  bo 分布 %s" % {b: sum(1 for r in rows if r["best_of"] == b) for b in (3, 5)})
print("  队伍 %d 支" % len({r["team1"]["name"] for r in rows} | {r["team2"]["name"] for r in rows}))
print("  maps_played 已知的 %d 场 / 未知(通吃) %d 场" % (full, len(rows) - full))
print("  ⚠ date 全为 null：进时间切分前必须由主理人补真实日期，否则违反时序防泄漏约束")
