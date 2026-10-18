"""把主理人 2026-10-19 提供的 Thunderpick World Championship 2026 赛果
导出成 match_record（13 场已完赛）+ 1 场进行中的 bo5 决赛（单独挂 pending）。

赛事核验（读仓库自有语料，不凭记忆）：
  `tier1.jsonl` / `stier_events.jsonl` 里
    tournament="Thunderpick World Championship 2026", tier="S-Tier",
    date="Oct 14-18, 2026", location="Attard", prize="$1,000,000",
    participants="8", winner="TBD"
  8 支队伍名在 `csapi_teams.json` 里逐个精确命中，无需 alias。
  用到的 7 张图 = `map_pool_timeline` era 2 的图池全集。

赛制（双败分组，主理人给的 stage 名 + 日期自洽，14 场无重无漏）：
  A 组 Falcons / NiP / 9z / Aurora      B 组 Legacy / PARIVISION / FURIA / BETBOOM
    10-14 胜者组首轮 ×2                   10-14 胜者组首轮 ×2
    10-15 胜者组 + 败者组首轮             10-15 胜者组 + 败者组首轮
    10-16 出线战（9z vs Aurora）          10-16 出线战（Legacy vs BETBOOM）
    ⇒ 胜者组冠军直接晋级；出线战败者淘汰
  10-17 半决赛 FURIA 2-0 Aurora / BETBOOM 2-1 Falcons
  10-18 季军赛 Aurora 2-1 Falcons ／ **决赛 bo5 BETBOOM vs FURIA 进行中**

⚠ 三处**不猜**（宁可缺字段，不可编造）：
  1. 逐图比分（map_results）主理人未给 ⇒ fuel ① 缺，`elo_m` 会退回胜负指示。
  2. 阵容 / 逐图逐选手 rating（rosters / by_map）未给 ⇒ fuel ②③ 缺。
  3. veto 的 ban 步未给 ⇒ `veto_order` 只含主理人明确说过的 2 pick + 1 decider，
     **不补 ban**（记 `re` 等于编造；`veto_rec_build` 的 ban 分支本来就空转）。
     ⇒ 后果：这批只进 `tpick/gpick/npick`，不进 `tban/gban/nban`。
  4. 图池 era 2 的 `end` 停在 2026-09-05，**10 月无权威图池记录**。
     这里假设 era 2 延续，标 `pool_src: "INFERRED"`，不静默处理。

用法：python -X utf8 export_tpwc2026_records.py
"""
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import record_schema as RS

EVENT = "Thunderpick World Championship 2026"
SRC = "主理人赛果 Thunderpick WC 2026"
# era 2 的 7 张图池（= 主理人这批用到的全部 7 张图）
POOL = ["Ancient", "Anubis", "Cache", "Dust2", "Inferno", "Mirage", "Nuke"]
OUT = os.path.join(HERE, "new_records_2026-10.jsonl")
OUT_PEND = os.path.join(HERE, "pending_tpwc2026_bo5_final.jsonl")

DEGRADED = ("缺 map_results/rosters/by_map 三组燃料（主理人只给了逐图选图 + MVP）；"
            "veto ban 步未给，故只有 pick 归因、没有 ban 归因")

# (日期, stage, 队A[胜], 比分, 队B, [(图, pick方|"L"), ...], decider是否已打,
#  MVP, rating)
# ⚠ veto_order 的 step 用 **1 起**（与权威语料 records_v1.jsonl 同基数；
#   build_veto_rec 的自动路径是 0 起，两种都过不了语义，只作流水号，故取语料口径）。
ROWS = [
    ("2026-10-14", "小组赛首轮 · A 组胜者组首轮", "Falcons", 2, "Ninjas in Pyjamas", 0,
     [("Ancient", "Ninjas in Pyjamas"), ("Dust2", "Falcons"), ("Mirage", None)],
     "m0NESY", 1.62),
    ("2026-10-14", "小组赛首轮 · A 组胜者组首轮", "9z", 2, "Aurora", 0,
     [("Inferno", "9z"), ("Nuke", "Aurora"), ("Dust2", None)],
     "luchov", 1.61),
    ("2026-10-14", "小组赛首轮 · B 组胜者组首轮", "Legacy", 2, "PARIVISION", 0,
     [("Dust2", "PARIVISION"), ("Mirage", "Legacy"), ("Ancient", None)],
     "dumau", 1.21),
    ("2026-10-14", "小组赛首轮 · B 组胜者组首轮", "FURIA", 2, "BETBOOM", 0,
     [("Ancient", "FURIA"), ("Mirage", "FURIA"), ("Nuke", None)],
     "KSCERATO", 1.72),

    ("2026-10-15", "A 组胜者组", "Falcons", 2, "9z", 1,
     [("Nuke", "9z"), ("Mirage", "9z"), ("Dust2", None)], "TeSeS", 1.26),
    ("2026-10-15", "A 组败者组首轮", "Aurora", 2, "Ninjas in Pyjamas", 0,
     [("Cache", "Aurora"), ("Dust2", "Aurora"), ("Mirage", None)], "Wicadia", 1.65),
    ("2026-10-15", "B 组胜者组", "FURIA", 2, "Legacy", 1,
     [("Cache", "Legacy"), ("Inferno", "FURIA"), ("Mirage", None)], "KSCERATO", 1.29),
    ("2026-10-15", "B 组败者组首轮", "BETBOOM", 2, "PARIVISION", 1,
     [("Ancient", "PARIVISION"), ("Anubis", "BETBOOM"), ("Mirage", None)],
     "s1ren", 1.41),

    ("2026-10-16", "A 组出线战", "Aurora", 2, "9z", 0,
     [("Cache", "Aurora"), ("Inferno", "9z"), ("Mirage", None)], "kyxsan", 1.45),
    ("2026-10-16", "B 组出线战", "BETBOOM", 2, "Legacy", 1,
     [("Nuke", "BETBOOM"), ("Dust2", "Legacy"), ("Mirage", None)], "Magnojez", 1.17),

    ("2026-10-17", "半决赛", "FURIA", 2, "Aurora", 0,
     [("Dust2", "Aurora"), ("Inferno", "FURIA"), ("Nuke", None)], "KSCERATO", 1.42),
    ("2026-10-17", "半决赛", "BETBOOM", 2, "Falcons", 1,
     [("Anubis", "Falcons"), ("Dust2", "BETBOOM"), ("Mirage", None)],
     "Magnojez", 1.39),

    ("2026-10-18", "季军赛", "Aurora", 2, "Falcons", 1,
     [("Anubis", "Falcons"), ("Dust2", "Aurora"), ("Nuke", None)], "Jimpphat", 1.37),
]

# ⚠ 进行中的 bo5 决赛：只登记**已知**部分（选图），比分/胜方/MVP 一律不填。
PENDING = {
    "schema": "match_record_pending/1",
    "id": "tpwc2026-10-18|BETBOOM|FURIA",
    "date": "2026-10-18",
    "date_src": "read",
    "event": EVENT,
    "stage": "决赛（bo5）",
    "best_of": 5,
    "team1": {"name": "BETBOOM", "score": None},
    "team2": {"name": "FURIA", "score": None},
    "winner": None,
    "status": "in_progress",
    "maps_full_B": ["Anubis", "Dust2", "Nuke", "Mirage", "Cache"],
    "maps_played": None,
    "veto_order": [[1, "pi", "FURIA", "Anubis"], [2, "pi", "BETBOOM", "Dust2"],
                   [3, "pi", "FURIA", "Nuke"], [4, "pi", "BETBOOM", "Mirage"],
                   [5, "L", None, "Cache"]],
    "pool": POOL,
    "pool_src": "INFERRED",
    "src": SRC,
    "notes": [
        "主理人 2026-10-19 录入时该场仍在进行，比分/胜方/MVP 均未产生。",
        "选图为 bo5 真实结构：4 pick + 1 decider（Cache），ban 步未给故不记。",
        "「黑豹」= FURIA 系由「决赛只有 BETBOOM 与 FURIA 两队、bb = BETBOOM」"
        "反推，非主理人原字；已请主理人确认。",
        "⇒ **不得**进入 new_records_*.jsonl：schema 要求 winner 必填，"
        "填占位值会造出 ingest_new.py 文档里点名的「幽灵记录」。",
        "结果出来后补 team1/team2.score、winner、maps_played、mvp、mvp_rating，"
        "再并入 new_records_2026-10.jsonl。",
    ],
}


def _mk(date, stage, a, sa, b, sb, ms, mvp, rating):
    maps = [m for m, _p in ms]
    played = ms[:2] + ([ms[2]] if sb == 1 else [])
    veto = [[i + 1, "pi", p, m] if p else [i + 1, "L", None, m]
            for i, (m, p) in enumerate(ms)]
    return {
        "schema": RS.SCHEMA_VERSION,
        "id": "tpwc2026-%s|%s|%s" % (date, a, b),
        "date": date, "date_src": "read",
        "event": EVENT, "stage": stage,
        "best_of": 3,
        "team1": {"name": a, "score": sa},
        "team2": {"name": b, "score": sb},
        "winner": a,
        "maps_full_B": maps,
        "maps_played": [m for m, _ in played],
        "veto_order": veto,
        "pool": POOL, "pool_src": "INFERRED",
        "mvp": mvp, "mvp_rating": rating,
        "maps_verified": True,
        "map_results_known": False, "maps_evidence": "user_bp",
        "src": SRC,
        "degraded": True, "degraded_reason": DEGRADED,
    }


def main():
    recs = [_mk(d, s, a, sa, b, sb, ms, mvp, rt)
            for d, s, a, sa, b, sb, ms, mvp, rt in ROWS]

    ok, bad, fuel_n = 0, [], dict.fromkeys(RS.FUEL, 0)
    for r in recs:
        good, probs, fuel = RS.validate(r, strict=False)
        if good:
            ok += 1
        else:
            bad.append((r["id"], probs))
        for k, v in fuel.items():
            fuel_n[k] += bool(v)
    print("Thunderpick WC 2026：导出 %d 场（%s .. %s）"
          % (len(recs), recs[0]["date"], recs[-1]["date"]))
    print("  schema 校验通过 %d 场，不通过 %d 场" % (ok, len(bad)))
    for pid, pr in bad:
        print("    ✗ %s → %s" % (pid, pr))

    # ---- 赛制自洽 ----
    # id 必须唯一（ingest 的游标按 (last_date, last_id) 断点续传，重复 id 会漏摄入）
    ids = [r["id"] for r in recs] + [PENDING["id"]]
    assert len(set(ids)) == len(ids), "id 重复"
    # 同一对队伍二次交手是双败赛制的正常现象（重赛），**不是**重复记录：
    # 只统计、报出来，不当错误。
    seen = {}
    for r in recs:
        seen.setdefault(frozenset((r["team1"]["name"], r["team2"]["name"])),
                        []).append((r["date"], r["stage"], r["winner"]))
    rematch = {k: v for k, v in seen.items() if len(v) > 1}
    print("  id 唯一（13 场 + 1 场 pending）✓")
    if rematch:
        print("  重赛（双败赛制正常，共 %d 对）：" % len(rematch))
        for k, v in rematch.items():
            pair = " vs ".join(sorted(k))
            for d, st, w in v:
                print("    %s  %-22s  胜 %s" % (d, st, w))

    print("\n  四组燃料到位情况：")
    for k, v in fuel_n.items():
        print("    %-13s %3d/%d 场" % (k, v, len(recs)))

    # 读法 B 张数 + 逐图已打/未打一致性（decider 未打 ⇔ 2:0）
    for r in recs:
        n = len(r["maps_played"])
        assert n == r["team1"]["score"] + r["team2"]["score"], \
            "%s 逐图 %d 张 ≠ 系列 %d:%d" % (r["id"], n, r["team1"]["score"],
                                           r["team2"]["score"])
        assert all(m in POOL for m in r["maps_full_B"]), r["id"]
    print("  逐图张数 == 系列图数、图名全在图池内 ✓")

    with io.open(OUT, "w", encoding="utf-8", newline="\n") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with io.open(OUT_PEND, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(PENDING, ensure_ascii=False) + "\n")

    print("\n已写：")
    for p in (OUT, OUT_PEND):
        print("  %-38s %2d 行" % (os.path.basename(p),
                                 sum(1 for _ in io.open(p, encoding="utf-8"))))
    print("\n⚠ 干跑结束，**未入库**。缺 %s ⇒ elo_m 退回胜负指示、阵容类特征不更新。"
          % "/".join(("map_results", "rosters", "by_map")))
    print("  veto ban 步未给 ⇒ 只有 pick 归因（tpick/gpick/npick），没有 ban 归因。")
    print("  bo5 决赛在 pending 文件里，**不在** records 里（winner 未知）。")
    print("  pool_src=INFERRED：era 2 图池 end 停在 2026-09-05，10 月图池无权威记录。")
    return 0


if __name__ == "__main__":
    sys.exit(main())