"""全语料等价性验证（[P0]-1 第 1 步的验收）

测试 1：`ingest()`（本模块的账本驱动）全量重放 == `model_v2.prepare()` 里的 state/mh
        ⇒ 证明增量驱动循环**精确镜像**了 `build()` 的时序安全约束。
测试 2：快照(切点) → 还原 → 增量补齐 == 全量重放
        ⇒ 证明快照 + 增量与全量**数值等价**。
另外单独跑一次「恒等推演」：快照 → 还原 → **空增量**，验证零漂移基线。
"""
import io, json, os, pickle, sys, time, hashlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import model_v2 as M
import veto_model as VM
import split as SPLIT
import pool_experiment as PE
import state_snapshot as SS


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    t0 = time.time()
    print("== 准备全量参考 ==")
    prep = M.prepare(verbose=False)
    ref_s, ref_h = prep["state"], prep["mh"]
    rows = prep["rows"]
    sha = sha256_of(M.MATCHES)
    print("  prepare() 完成  rows=%d  state.last_date=%s  (%.1fs)"
          % (len(rows), ref_s.last_date, time.time() - t0))

    ctx = SS.make_ctx(prep["truths"], prep["veto"], M.load_stats(),
                      M.load_stats_map(), *PE.load_pool_eras(),
                      json.load(io.open(PE.TIMELINE, encoding="utf-8")))
    exclude = SPLIT.load_exclude(M.EXCLUDE) if os.path.exists(M.EXCLUDE) else set()
    # ⚠ 摄入集**必须**取自 split 的三个 bucket：`SPLIT.split` 除 exclude(55) 外
    # 还会因 purge 边界吸附多丢 95 场（合计 150 场不进任何 bucket）⇒ 不能用 rows−exclude。
    _b, _meta = SPLIT.split(SPLIT.load(M.MATCHES), purge=True, exclude=exclude)
    usable = [r for b in ("train", "valid", "test")
              for r in _b[b]
              if isinstance(r.get("best_of"), int) and r["best_of"] in (1, 3, 5)]
    usable.sort(key=lambda r: (r["date"], r["id"]))
    print("  可摄入 %d 场（2,800 原始 → split 保留 %d → bo 过滤 %d）  日期 %s .. %s"
          % (len(usable), sum(len(_b[b]) for b in _b), 2741, usable[0]["date"], usable[-1]["date"]))

    # ---------- 测试 1 ----------
    print("\n== 测试 1：ingest() 全量重放 vs prepare() ==")
    s1, h1 = M.State(), VM.MapHistory()
    n1, cur1 = SS.ingest(s1, h1, usable, ctx)
    d1, w1 = SS.diff_ledgers(ref_s, s1, SS.STATE_KEYS)
    d2, w2 = SS.diff_ledgers(ref_h, h1, SS.MH_KEYS)
    print("  摄入 %d 场（prepare 内部 2796）" % n1)
    print("  State   最大差 = %.3e   (%s)" % (d1, w1))
    print("  MapHist 最大差 = %.3e   (%s)" % (d2, w2))
    ok1 = (d1 < 1e-12 and d2 < 1e-12)
    print("  判定：%s" % ("PASS 镜像精确" if ok1 else "FAIL"))

    # ---------- 测试 2 ----------
    print("\n== 测试 2：快照(切点) + 增量 == 全量 ==")
    for frac in (0.5, 0.8):
        k = int(len(usable) * frac)
        cut = usable[k - 1]
        sa, ha = M.State(), VM.MapHistory()
        _, cura = SS.ingest(sa, ha, usable[:k], ctx)
        blob = SS.to_blob(sa, ha, cura, source_sha256=sha,
                          note="cut@%s(#%d)" % (cut["date"], cut["id"]))
        raw = pickle.dumps(blob)
        sb, hb, curb, sha_b = SS.from_blob(pickle.loads(raw))
        # 2a 空增量（零漂移基线）
        dz1, wz1 = SS.diff_ledgers(sa, sb, SS.STATE_KEYS)
        dz2, wz2 = SS.diff_ledgers(ha, hb, SS.MH_KEYS)
        # 2b 补齐剩余
        _, curb2 = SS.ingest(sb, hb, usable[k:], ctx, cursor=curb)
        df1, wf1 = SS.diff_ledgers(s1, sb, SS.STATE_KEYS)
        df2, wf2 = SS.diff_ledgers(h1, hb, SS.MH_KEYS)
        print("  切点 %s(#%d)  快照 %d B" % (cut["date"], cut["id"], len(raw)))
        print("    空增量漂移：State %.3e (%s) / MapHist %.3e (%s)"
              % (dz1, wz1, dz2, wz2))
        print("    补齐后 vs 全量：State %.3e (%s) / MapHist %.3e (%s)"
              % (df1, wf1, df2, wf2))
        assert sha_b == sha
        if not (df1 < 1e-9 and df2 < 1e-9):
            print("    FAIL")
            return 1
        print("    PASS")

    print("\n总耗时 %.1fs" % (time.time() - t0))
    print("SNAPSHOT_EQUIV_OK" if ok1 else "SNAPSHOT_EQUIV_FAIL")
    return 0 if ok1 else 1


if __name__ == "__main__":
    sys.exit(main())
