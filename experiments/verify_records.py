"""证明 match_record 携带的信息**足以**做逐比特等价的增量重建。

用法：records_v1.jsonl → 构建 ctx → SS.ingest() → 与 prepare() 的账本逐字段比对。
若通过 ⇒ 今后任何新赛事只要产出 match_record，就能完整增量更新全部账本。
"""
import io, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import model_v2 as M
import veto_model as VM
import state_snapshot as SS
import record_schema as RS
import pool_experiment as PE

RECS = os.path.join(HERE, "records_v1.jsonl")


def main():
    print("== 读 records ==")
    recs = [json.loads(l) for l in io.open(RECS, encoding="utf-8").read().split("\n") if l.strip()]
    print("  %d 条" % len(recs))

    truths, veto, stats, map_stats = {}, {}, {}, {}
    rows = []
    for rec in recs:
        row = RS.to_ingest_row(rec)
        rows.append(row)
        truths[row["id"]] = {"maps": rec["maps_full_B"], "pool": rec.get("pool") or []}
        if rec.get("veto_order"):
            veto[row["id"]] = {"bo": rec["best_of"],
                               "order": [tuple(s) for s in rec["veto_order"]]}
        if rec.get("rosters"):
            stats[row["id"]] = row["_rosters"]
        if rec.get("by_map"):
            map_stats[row["id"]] = row["_by_map"]
    print("  ctx：truths=%d veto=%d rosters=%d by_map=%d"
          % (len(truths), len(veto), len(stats), len(map_stats)))

    ctx = SS.make_ctx(truths, veto, stats, map_stats, *PE.load_pool_eras(),
                      json.load(io.open(PE.TIMELINE, encoding="utf-8")))
    npool = sum(1 for r in rows if not (truths[r["id"]].get("pool")))
    print("  ctx：pool 缺失 %d 场（回落到 map_pool_timeline）" % npool)
    rows.sort(key=lambda r: (r["date"], r["id"]))

    print("\n== 参考：prepare() 全量重放 ==")
    prep = M.prepare(verbose=False)
    ref_s, ref_h = prep["state"], prep["mh"]

    print("== 对照：records 增量摄入 ==")
    s, h = M.State(), VM.MapHistory()
    n, cur = SS.ingest(s, h, rows, ctx)
    d1, w1 = SS.diff_ledgers(ref_s, s, SS.STATE_KEYS)
    d2, w2 = SS.diff_ledgers(ref_h, h, SS.MH_KEYS)
    print("  摄入 %d 场  末日 %s" % (n, cur["last_date"]))
    print("  State   最大差 = %.3e  %s" % (d1, w1))
    print("  MapHist 最大差 = %.3e  %s" % (d2, w2))
    ok = d1 < 1e-12 and d2 < 1e-12
    print("\n%s" % ("RECORDS_FULLFUEL_EQUIV_OK  ⇒ match_record 足以完整重建账本"
                    if ok else "RECORDS_FULLFUEL_EQUIV_FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
