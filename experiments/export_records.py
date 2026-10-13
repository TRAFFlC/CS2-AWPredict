"""把现有语料导出成 match_record（[P0]-1 第 A 步的「燃料」）。

输入（4 个现有文件，全部已在私有区）：
  csapi_matches_all.jsonl  → 逐图比分 + 队伍分数
  truth_maps_B.jsonl       → 读法 B 图集 + pool + 实际打过的
  csapi_stats_all.jsonl    → 系列阵容 rating（overall）+ 逐图逐选手 rating（by_map）
  csapi_veto.jsonl         → veto 顺序
输出：records_v1.jsonl（只含时间切分实际摄入的场次，与 ingest 口径一致）
"""
import collections, io, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import model_v2 as M
import split as SPLIT
import veto_model as VM
import pool_experiment as PE
import record_schema as RS

OUT = os.path.join(HERE, "records_v1.jsonl")


def main():
    rows = {r["id"]: r for r in M.load_jsonl(M.MATCHES)}
    truths = {t["id"]: t for t in M.load_jsonl(M.TRUTH)}
    veto = {}
    for rec in M.load_jsonl(VM.VETO):
        order = rec.get("veto") or []
        if order:
            veto[rec["id"]] = [list(s) for s in order]

    stats_overall, stats_bymap = {}, {}
    for rec in M.load_jsonl(M.STATS):
        mid = rec["matchid"]
        blk = (rec.get("overall") or [{}])[0]
        ro = {}
        for side in ("team1", "team2"):
            tb = blk.get(side) or {}
            if tb.get("name"):
                ro[tb["name"]] = [{"name": p["name"], "rating": p.get("rating")}
                                  for p in (tb.get("players") or [])]
        stats_overall[mid] = ro
        bm = {}
        for b in (rec.get("by_map") or []):
            mn = b.get("name")
            if not mn or mn == "All":
                continue
            d = {}
            for side in ("team1", "team2"):
                tb = b.get(side) or {}
                if tb.get("name"):
                    d[tb["name"]] = [{"name": p["name"], "rating": p.get("rating")}
                                     for p in (tb.get("players") or [])]
            bm[mn] = d
        stats_bymap[mid] = bm
    print("读入：matches=%d truths=%d veto=%d stats=%d" %
          (len(rows), len(truths), len(veto), len(stats_overall)))

    exc = SPLIT.load_exclude(M.EXCLUDE) if os.path.exists(M.EXCLUDE) else set()
    buckets, _meta = SPLIT.split(SPLIT.load(M.MATCHES), purge=True, exclude=exc)
    ids = [r["id"] for b in ("train", "valid", "test")
           for r in buckets[b]
           if isinstance(r.get("best_of"), int) and r["best_of"] in (1, 3, 5)]
    print("切分实际摄入 %d 场" % len(ids))

    fuel_tally = collections.Counter()
    n_bad = 0
    n_degraded = 0
    out = []
    for mid in ids:
        r = rows.get(mid)
        t = truths.get(mid)
        if not r or not t:
            continue
        bo = t.get("bo") or r.get("best_of") or 3
        s1, s2 = r["team1"]["score"], r["team2"]["score"]
        raw = None
        if bo == 1:
            # ⚠ bo1 陷阱（见 014-veto-probe 笔记）：csapi **与** truth_maps_B 里
            # bo1 的 score 都是**回合分**（13:11 / 19:16），不是系列分。
            # 按 Q-007「bo1 不记录比分」⇒ 归一化成 1:0，原始回合分另存。
            raw = [s1, s2]
            s1, s2 = (1, 0) if r["winner"]["name"] == r["team1"]["name"] else (0, 1)
        rec = {
            "schema": RS.SCHEMA_VERSION,
            "id": mid, "date": r["date"], "event": r.get("event"), "best_of": bo,
            "team1": {"name": r["team1"]["name"], "score": s1},
            "team2": {"name": r["team2"]["name"], "score": s2},
            "winner": r["winner"]["name"],
            "raw_round_score": raw,
            "maps_full_B": t.get("maps") or [],
            "maps_played": t.get("maps_played_true") or t.get("maps_played") or [],
            "pool": t.get("pool") or [],
            "map_results": [{"map": m["name"], "s1": m["team1_score"],
                             "s2": m["team2_score"]} for m in (r.get("maps") or [])],
            "rosters": stats_overall.get(mid) or {},
            "by_map": stats_bymap.get(mid) or {},
            "veto_order": veto.get(mid) or [],
            "src": "csapi",
        }
        ok, pr, fu = RS.validate(rec, strict=False)
        if not ok:
            # 读法 B 张数不足是**已知合法降级态**（`veto_labels.py` 的 maps_mode="played"，
            # 全语料仅 id 2397397 一例：bo5 但只取到 4 张图）。这类记录 build() 是照常摄入的
            # ⇒ 这里也必须保留，只打降级标记，**不能丢**（丢了会少一次 last_match / tapp 贡献）。
            only_b_len = all("读法 B" in x for x in pr)
            if not only_b_len:
                n_bad += 1
                if n_bad <= 3:
                    print("  校验失败(丢弃) id=%d: %s" % (mid, pr))
                continue
            rec["degraded"] = True
            rec["degraded_reason"] = pr
            rec["maps_mode"] = "played"
            n_degraded += 1
        else:
            rec["degraded"] = False
        for k, v in fu.items():
            fuel_tally[k] += 1 if v else 0
        out.append(rec)

    with io.open(OUT, "w", encoding="utf-8", newline="\n") as f:
        for rec in out:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("\n写出 %s：%d 条（丢弃 %d，降级保留 %d）"
          % (os.path.basename(OUT), len(out), n_bad, n_degraded))
    print("四组燃料覆盖率：")
    for k in RS.FUEL:
        c = fuel_tally[k]
        print("  %-14s %5d / %d = %6.2f%%" % (k, c, len(out), 100.0 * c / max(1, len(out))))
    full = sum(1 for rec in out
               if rec["map_results"] and rec["rosters"] and rec["by_map"] and rec["veto_order"])
    print("  四组齐全（可完整增量）      %5d / %d = %6.2f%%"
          % (full, len(out), 100.0 * full / max(1, len(out))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
