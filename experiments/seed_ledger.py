"""用 41 场真实结果回填台账 R1–R8（append-only，一次性）。

为什么现在回填：台账的 δ 自愈需要「冻结概率 vs 实际得分」的历史。
R1–R8 已经有 41 场冻结件 + 41 场官方结果 ⇒ 立刻就能算出真实赛场的 δ，
不用等下一轮。榜首分数历史拿不到 ⇒ leader_score 记 null（不是 0，避免伪造）。

一次性脚本：跑完即弃，重跑会因「轮次已存在」被拒。
"""
import io, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # scorer.py 与本模块同目录（仓库布局）
sys.path.insert(0, HERE)

import scorer as SC
import ledger

LIVE = r"E:\world\another\data\Mimo\MiMo-V2.6-Flash\2026-10-12\live_tournament_results.jsonl"


def main():
    rows = ledger.load()
    if rows:
        print("台账非空（%d 行），拒绝回填" % len(rows))
        return 1

    per_round = {}
    for line in io.open(LIVE, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        rnd = int(re.match(r"live-r(\d+)-", r["id"]).group(1))
        f = r["pred_frozen"]
        A, B = r["team1"]["name"], r["team2"]["name"]
        truth = SC.Truth(
            teamA=A, teamB=B, winner=r["winner"],
            score=(r["team1"]["score"], r["team2"]["score"]),
            maps=list(r["maps_full_B"]), mvp=r["mvp"],
            best_of=r["best_of"], maps_mode="veto")
        pred = SC.Prediction(winner=f["winner"], score=tuple(f["score"]),
                             maps=list(f["maps"]), mvp=f["mvp"])
        bd = SC.score_match(pred, truth)
        per_round.setdefault(rnd, []).append({
            "A": A, "B": B, "bo": r["best_of"],
            "winner": f["winner"], "win_prob": f["win_prob"],
            "score": f["score"], "maps": f["maps"], "mvp": f["mvp"],
            "exp_points": f["exp_points"],
            "_got": bd.total, "_max": SC.max_total(truth),
            "_b": {"winner": bd.winner, "score": bd.score,
                   "maps": bd.maps, "mvp": bd.mvp},
        })

    cum = 0
    for rnd in sorted(per_round):
        ms = per_round[rnd]
        got = sum(m["_got"] for m in ms)
        mx = sum(m["_max"] for m in ms)
        cum += got
        rows.append({
            "round": rnd,
            "ts": "2026-10-12T%02d:00:00" % rnd,
            "pred_file": ms[0]["A"] and ("predictions_2026-10-12%s.json"
                                         % ("" if rnd == 1 else "_r%d" % rnd)),
            "pred_sha256": "(backfilled)",
            "n_matches": len(ms),
            "max_points": mx,
            "self_est_total": round(sum(m["exp_points"] for m in ms), 2),
            "my_score": cum,
            "leader_score": None,
            "field_n": None,
            "objective": "score",
            "frozen_predictions": [{k: v for k, v in m.items()
                                    if not k.startswith("_")} for m in ms],
            "round_points": got,
            "breakdown": {
                "winner": sum(m["_b"]["winner"] for m in ms),
                "score": sum(m["_b"]["score"] for m in ms),
                "maps": sum(m["_b"]["maps"] for m in ms),
                "mvp": sum(m["_b"]["mvp"] for m in ms),
            },
        })
        print("R%-2d  %2d 场  得 %3d / 满分 %3d  累计 %4d / %4d   %s"
              % (rnd, len(ms), got, mx, cum, sum(x["max_points"] for x in rows),
                 json.dumps(rows[-1]["breakdown"], ensure_ascii=False)))
    ledger.save(rows)
    print("\n已回填 %d 轮，累计 %d / %d (%.2f%%)"
          % (len(rows), cum, sum(x["max_points"] for x in rows),
             100 * cum / sum(x["max_points"] for x in rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
