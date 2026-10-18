"""014 盲预测批次：一次建模，逐场预测并冻结。
用法: python -X utf8 blind_batch.py <out.json> [<matchups.json>]
matchups.json 形如 [["G2","Natus Vincere",3], ...]；不给则用内置第一轮名单。
"""
import io, json, os, sys, hashlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import predict as P

DEFAULT = [
    ["G2", "Natus Vincere", 3], ["FURIA", "BetBoom", 3],
    ["Spirit", "ShindeN", 3], ["Aurora", "9z", 3],
    ["MOUZ", "M80", 3], ["Falcons", "TYLOO", 3],
    ["Legacy", "PARIVISION", 3], ["Vitality", "1win", 3],
]

out_path = sys.argv[1] if len(sys.argv) > 1 else "predictions_2026-10-12.json"
MATCHES = json.load(io.open(sys.argv[2], encoding="utf-8")) if len(sys.argv) > 2 else DEFAULT

P_ = P.Predictor()
print("语料末日 =", P_.last_date, " 硬编码图池 =",
      getattr(P_, "pool", None) or P_.prep.get("pool"))
print("-" * 84)

out = {"generated": "2026-10-12", "corpus_last_date": P_.last_date, "matches": []}
for a, b, bo in MATCHES:
    p = P_.predict(a, b, bo, None, "score", quiet=True)
    pr = P_.predict(a, b, bo, None, "rank", quiet=True, opponents=20, noise=0.25)
    row = {"A": a, "B": b, "bo": bo,
           "winner": p["winner"],
           "win_prob": {k if isinstance(k, str) else str(k): round(v, 4)
                        for k, v in p["win_prob"].items()},
           "score": list(p["score"]), "maps": list(p["maps"]), "mvp": p["mvp"],
           "exp_points": round(p["exp_points"], 2), "sd": round(p["sd"], 2),
           "exp_rank_if_20": (round(pr["exp_rank"], 2) if pr and pr.get("exp_rank") else None),
           "map_probs": {k: round(v, 4) for k, v in p.get("map_probs", {}).items()},
           "mvp_probs": {k: round(v, 4) for k, v in p.get("mvp_probs", {}).items()},
           "score_probs": {k if isinstance(k, str) else str(k): round(v, 4)
                           for k, v in p.get("score_probs", {}).items()},
           "cold": p.get("cold", {})}
    out["matches"].append(row)
    print("== %s vs %s (bo%d) ==" % (a, b, bo))
    print("  胜方 %-14s P=%.1f%%   比分 %s   期望 %.1f (sd %.1f)%s"
          % (p["winner"], 100 * max(p["pA"], 1 - p["pA"]), "%d:%d" % tuple(p["score"]),
             p["exp_points"], p["sd"],
             ("   冷门注入:%s" % ",".join(k for k, v in p.get("cold", {}).items() if v))
             if p.get("cold") else ""))
    # ⚠ 按概率降序 + 每张图带概率。直接 join(p["maps"]) 是字母序，
    # 会把 Mirage 74% 排在 Dust2 57% 后面（2026-10-19 主理人抓过）。
    _pm = p.get("map_probs") or {}
    _pk = [m for m in (p.get("maps_ranked") or p["maps"])]
    _pk.sort(key=lambda m: -_pm.get(m, 0.0))
    print("  选图（按概率降序） " + "  ·  ".join(
        "%s %.0f%%" % (m, 100 * _pm[m]) if m in _pm else m for m in _pk))
    print("  MVP  %s" % (p["mvp"] or "(无选手数据)"))
    _rest = sorted(((m, v) for m, v in _pm.items() if m not in _pk),
                   key=lambda kv: -kv[1])
    if _rest:
        print("  池内未选 " + "  ".join("%s %.0f%%" % (m, 100 * v) for m, v in _rest))
    print("  图集概率 " + "  ".join("%s %.2f" % (k, v) for k, v in
                                    sorted(_pm.items(), key=lambda x: -x[1])[:5]))
    print("  MVP 概率 " + "  ".join("%s %.2f" % (k, v) for k, v in
                                    sorted(p.get("mvp_probs", {}).items(), key=lambda x: -x[1])[:3]))

io.open(out_path, "w", encoding="utf-8").write(json.dumps(out, ensure_ascii=False, indent=2))
h = hashlib.sha256(io.open(out_path, "rb").read()).hexdigest()
tot = sum(m["exp_points"] for m in out["matches"])
mx = 45 * len(out["matches"])
print("-" * 84)
print("FROZEN -> %s  sha256=%s  n=%d  自估 %.1f/%d = %.1f%%"
      % (out_path, h[:16], len(out["matches"]), tot, mx, 100.0 * tot / mx))
