"""把 EPL S24 的 41 场赛果转成 match_record（match_record/1），供增量摄入。

⚠ 如实标注：这批数据**只有系列胜负 + 读法 B 图集**，没有逐图比分 / 完整阵容+rating /
veto 顺序 ⇒ `map_results`、`rosters`、`by_map`、`veto_order` 四组燃料为空，
增量只会更新 games / elo / mw / mf / last_match / h2h_hist（+ app/tapp/pooln/tpooln）。
这正是 [P0]-1 第 A 步要解决的 schema 缺口：**今后的采集必须产出这四组**。
"""
import io, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "../data")  # # 仓库布局：数据与代码分目录（Alpha 1.1.1）
sys.path.insert(0, HERE)
import record_schema as RS

LIVE = r"E:\world\another\data\Mimo\MiMo-V2.6-Flash\2026-10-12\live_tournament_results.jsonl"
OUT = os.path.join(DATA, "groundtruth", "records_epl_s24.jsonl")
POOL = ["Ancient", "Anubis", "Cache", "Dust2", "Inferno", "Mirage", "Nuke"]


def main():
    rows = []
    n_deg = 0
    for i, l in enumerate(io.open(LIVE, encoding="utf-8").read().split("\n")):
        if not l.strip():
            continue
        r = json.loads(l)
        rec = {
            "schema": RS.SCHEMA_VERSION,
            "id": 900000 + i,
            "date": r["date"], "time": r.get("time"),
            "event": r["event"], "best_of": r["best_of"],
            "team1": {"name": r["team1"]["name"], "score": r["team1"]["score"]},
            "team2": {"name": r["team2"]["name"], "score": r["team2"]["score"]},
            "winner": r["winner"],
            "maps_full_B": r["maps_full_B"],
            "maps_played": r.get("maps_played"),
            "pool": POOL,
            "map_results": [],            # ⚠ 缺
            "rosters": {},                # ⚠ 缺
            "by_map": {},                 # ⚠ 缺
            "veto_order": [],             # ⚠ 缺
            "src": r.get("src", "主理人公布"),
            "date_src": r.get("date_src", "read"),
            "fuel_note": "仅系列胜负 + 读法B 图集；四组燃料皆空 ⇒ 只能部分增量更新",
        }
        ok, pr, fu = RS.validate(rec, strict=False)
        if not ok:
            only_b = all("读法 B" in x for x in pr)
            if not only_b:
                print("  丢弃 id=%d: %s" % (rec["id"], pr))
                continue
            rec["degraded"] = True
            rec["degraded_reason"] = pr
            rec["maps_mode"] = "played"
            n_deg += 1
        else:
            rec["degraded"] = False
        rows.append(rec)

    with io.open(OUT, "w", encoding="utf-8", newline="\n") as f:
        for rec in rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    full = sum(1 for r in rows if all(
        r.get(k) for k in ("map_results", "rosters", "by_map", "veto_order")))
    print("写出 %s：%d 条（降级 %d，四组齐全 %d）"
          % (os.path.basename(OUT), len(rows), n_deg, full))
    print("⇒ 这批数据只能做**部分**增量；完整增量需要采集侧补齐四组燃料")
    return 0


if __name__ == "__main__":
    sys.exit(main())
