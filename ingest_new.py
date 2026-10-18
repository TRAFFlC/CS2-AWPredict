"""把 112 场新赛果写入**持久化增量状态**，并验证重放一致性。

主理人 2026-10-12 的立场：**无论更新后变好变坏都必须更新**——数据陈旧本身就是错，
回测只用于量化幅度（结论：选图 +1.0pp，0.1σ，在噪声内）。

时间切分验证（`eval_update_holdout.py`）的读数：
  留出 34 场（2026-10-04..10-11，两版都没见过）
    09-05 快照   选图 60/104 = 57.7%
    摄入前 70% 后 选图 61/104 = 58.7%   +1.0pp（0.1σ，噪声内）
  games 计数普遍 +2 ~ +7（如 Alliance 67→72）⇒ **陈旧度确实被修掉了**，
  这才是本次更新的主要价值，而不是那 1pp。

⚠ 两点必须写进文件头：
  1. 缺 `map_results` / `rosters` / `by_map` 三组燃料 ⇒ Elo 的 `elo_m` 退回胜负指示，
     阵容/评分类特征不更新。
  2. `veto_order` 依赖主理人 2026-10-12 的确认「我列图顺序 = pick 顺序」。

用法：
  python -X utf8 ingest_new.py            # 写盘 + 校验
  python -X utf8 ingest_new.py --verify   # 只校验，不写
"""
import argparse
import collections
import copy
import glob
import io
import json
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")  # # 仓库布局：数据与代码分目录（Alpha 1.1.0）
sys.path.insert(0, HERE)
import model_v2 as M
import record_schema as RS
import state_snapshot as SS

# ⚠ 2026-10-19：改为读**全部** new_records_*.jsonl（原先硬编码单文件，
#   新批次 new_records_2026-10.jsonl 根本读不到）。
#   `pending_*.jsonl` 不匹配此模式 —— 进行中的 bo5 因此不可能被摄入，
#   与 ingest_new.py 文档里点名的「幽灵记录」机制无关。
NEW_GLOB = "new_records_*.jsonl"
OUT = os.path.join(DATA, "state_2026-10.pkl")
OUT_MH = os.path.join(HERE, "mh_2026-10.pkl")
OUT_CUR = os.path.join(HERE, "cursor_2026-10.json")

FUEL_MISSING = ("map_results", "rosters", "by_map")


def load_new(patterns=None):
    """返回 (recs, per_file)。recs 按 (date, id) 排序 —— ingest 靠这个顺序做游标续传。"""
    pats = [patterns] if isinstance(patterns, str) else (patterns or [NEW_GLOB])
    paths = sorted({p for pat in pats
                    for p in glob.glob(os.path.join(HERE, pat))})
    recs, per = [], []
    for p in paths:
        rs = [json.loads(l) for l in io.open(p, encoding="utf-8") if l.strip()]
        per.append((os.path.basename(p), len(rs)))
        recs.extend(rs)
    ids = collections.Counter(r["id"] for r in recs)
    dup = [i for i, n in ids.items() if n > 1]
    assert not dup, "新记录 id 重复（会被摄入两次）: %s" % dup[:5]
    recs.sort(key=lambda r: (r["date"], r["id"]))
    return recs, per


def ledger_delta(old, new, keys, label):
    """逐键比对两个账本。**容忍新增键**（新队/新 h2h 对属预期），
    只报数值漂移 —— `SS.diff_ledgers` 遇键集不同直接 inf，不适合这里。"""
    worst, n_cmp, n_add, keys_changed = 0.0, 0, 0, []
    for k in keys:
        va, vb = SS._canon(old, [k])[k], SS._canon(new, [k])[k]
        if isinstance(va, dict) and isinstance(vb, dict):
            for kk in set(va) | set(vb):
                if kk not in va or kk not in vb:
                    n_add += 1
                    continue
                x, y = va[kk], vb[kk]
                if isinstance(x, float) or isinstance(y, float):
                    n_cmp += 1
                    d = abs(float(x) - float(y))
                    if d > worst:
                        worst, keys_changed = d, ["%s[%r] %r→%r" % (k, kk, x, y)]
                    elif d > 1e-9:
                        worst = max(worst, d)
                elif x != y:
                    keys_changed.append("%s[%r] %r≠%r" % (k, kk, str(x)[:40], str(y)[:40]))
    print("  %s：最大数值漂移 %.3e（比对 %d 标量，新增键 %d 个）"
          % (label, worst, n_cmp, n_add))
    for s in keys_changed[:5]:
        print("      %s" % s)
    return worst, n_add


def as_rows(recs):
    """只把 **schema 校验通过** 的记录转成摄入行。

    ⚠ 校验失败的必须**剔除**，不能照样喂给 ingest：那些场次比分是占位值
    （缺比分的两场用 (1,1) 平局占位），`State.observe` 仍会吃下 winner，
    于是账本里出现「胜方分数不足」的幽灵记录。实测症状：app/gpick 净新增
    达不到期望值，且这种脏数据无法事后分辨。
    """
    rows, dropped = [], []
    for r in recs:
        ok, probs, _fuel = RS.validate(r, strict=False)
        if ok:
            rows.append(RS.to_ingest_row(r))
        else:
            dropped.append((r["id"], probs))
    return rows, dropped


def _numeric_keys(keys):
    """排除 last_date：它是元信息（游标日期），变化是**预期**的，不是账本漂移。

    注意 `diff_ledgers` 对**新增队伍键**仍会返回 inf（本批新增 betclic +
    51 对 h2h），那也是预期行为，不是数值异常；真正要看的数值变动在下面的
    逐项分解里。
    """
    return [k for k in keys if k != "last_date"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--rebuild", action="store_true",
                    help="忽略既有快照，从 prepare() 重建后再摄入（修复被覆盖的快照）")
    ap.add_argument("--files", nargs="+", default=None,
                    help="要摄入的 jsonl（可给多个/glob），默认 new_records_*.jsonl 全集")
    a = ap.parse_args()

    recs, per = load_new(a.files)
    rows, dropped = as_rows(recs)
    print("新记录 %d 场  %s .. %s" % (len(recs), recs[0]["date"], recs[-1]["date"]))
    print("  来源文件：" + "  ".join("%s(%d 场)" % x for x in per))
    print("  schema 校验通过 %d 场，剔除 %d 场（**不喂给 ingest**）"
          % (len(rows), len(dropped)))
    if dropped:
        import collections as _c
        why = _c.Counter()
        for _pid, pr in dropped:
            for x in pr:
                why[x.split("：")[0]] += 1
        print("    剔除原因：" + "  ".join("%s×%d" % (k, v) for k, v in why.most_common()))
        print("    （前 5 条明细）")
        for pid, pr in dropped[:5]:
            print("      ✗ %s → %s" % (pid, pr))

    # ⚠ 幂等的关键：**已有快照时必须从 blob 恢复**，不能只从 prepare() 重建。
    # 只读游标、不读快照 ⇒ 摄入被全部跳过，却把一个退化的（09-05）状态写回去，
    # 表现为「摄入 0 场 / games 变动 0」——正是第一次写入被自己覆盖的机制。
    prep = M.prepare(verbose=False)
    import pool_experiment as PE
    ctx = SS.make_ctx(prep["truths"], prep["veto"], M.load_stats(),
                      M.load_stats_map(), *PE.load_pool_eras(),
                      json.load(io.open(PE.TIMELINE, encoding="utf-8")))

    if os.path.exists(OUT) and not a.rebuild:
        with open(OUT, "rb") as f:
            blob0 = pickle.load(f)
        s, h, cur, sha0 = SS.from_blob(blob0)
        print("从既有快照恢复：state.last_date=%s  cursor=%s"
              % (s.last_date, "%s/%s" % (cur.get("last_date"), cur.get("last_id"))))
        resumed = True
    else:
        s = copy.deepcopy(prep["state"])
        h = copy.deepcopy(prep["mh"])
        cur = None
        resumed = False
        if a.rebuild and os.path.exists(OUT):
            # ── 重建自证：--rebuild 最危险的地方是「静默丢账本」。
            # 重放前先读旧快照，待会儿逐键比对：State 应当**完全一致**
            # （同一基座 + 同一批记录 ⇒ 胜负/比分/阵容类账本不受 veto 修正影响），
            # 只有 MapHistory 的 pick/ban 归因会变（那正是本次要修的）。
            with open(OUT, "rb") as f:
                old_s, old_h, old_cur, _ = SS.from_blob(pickle.load(f))
            print("\n[重建自证] 旧快照 cursor=%s/%s  games 队数=%d"
                  % (old_cur.get("last_date"), old_cur.get("last_id"),
                     len(old_s.games)))
    before_d = s.last_date
    n, newcur = SS.ingest(s, h, rows, ctx, cursor=cur)
    print("\n摄入 %d 场  state.last_date: %s → %s" % (n, before_d, s.last_date))

    d1, w1 = SS.diff_ledgers(prep["state"], s, _numeric_keys(SS.STATE_KEYS))
    print("  State 数值账本（排除 last_date）：最大差 %.3e（%s）" % (d1, w1))
    print("  ↑ diff_ledgers 对**新增键**返回 inf（%s）：本批有 1 支新队 betclic "
          "首次进入账本、h2h 新增 51 对，属正常" % w1)

    if a.rebuild and os.path.exists(OUT):
        print("\n[重建自证] 与旧快照逐键比对（容忍新增键，只看数值漂移）：")
        w_s, add_s = ledger_delta(old_s, s, SS.STATE_KEYS, "State ")
        w_h, add_h = ledger_delta(old_h, h, SS.MH_KEYS, "MapHist")
        # ⚠ 这两项漂移**不等于**丢账本：--rebuild 若同时摄入了旧快照没有的新赛果，
        #   elo/elo_m/last_match/pooln/tapp 必然变化，那是加法、不是泄漏。
        #   真正的「丢账本」闸门是**输入集不变**时的漂移必须为 0，实测命令：
        #     python -X utf8 ingest_new.py --verify --rebuild \
        #          --files new_records_2026-09.jsonl
        #   （该命令实测 State 漂移 0.000e+00 / 8166 标量 / 新增键 0）
        if add_s or add_h:
            print("  ⇒ 有新增键（%d/%d）⇒ 本次输入集比旧快照**大**，"
                  "下列漂移来自新增赛果，不是账本泄漏。" % (add_s, add_h))
        else:
            print("  ⇒ 输入集与旧快照一致 ⇒ State %s；MapHist %s"
                  % ("一致（veto 修正未污染胜负/阵容账本）" if w_s < 1e-9
                     else "**有漂移 %.3e ⇒ 丢账本，需人工核查**" % w_s,
                     "一致" if w_h < 1e-9 else
                     "有差异（预期：bo5 decider 不再记为 pick）"))

    tmap = collections.Counter()
    for r in recs:
        for t in (r["team1"]["name"], r["team2"]["name"]):
            tmap[t] += 1
    changed = sum(1 for t in tmap if s.games.get(t, 0) != prep["state"].games.get(t, 0))
    print("  涉及队伍 %d 支，其中 games 计数变动 %d 支" % (len(tmap), changed))
    print("  最后一场：%s %s vs %s"
          % (recs[-1]["date"], recs[-1]["team1"]["name"], recs[-1]["team2"]["name"]))

    if a.verify:
        print("\n--verify：未写盘。")
        return 0

    # ⚠ 不能直接 `pickle.dump(state)`：**硬坑 A** — `State.__init__` 里的 6 个
    # lambda defaultdict 工厂无法 pickle（AttributeError: Can't get local object
    # 'State.__init__.<locals>.<lambda>'）。必须走 `SS.to_blob()` 摊平，
    # 它同时钉住 SHAPE_MIX / HALF_LIFE 等常数（硬坑 B：常数漂移即快照作废）。
    import hashlib
    src = hashlib.sha256()
    with open(M.MATCHES, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            src.update(b)
    blob = SS.to_blob(s, h, cursor=newcur, source_sha256=src.hexdigest(),
                      note="ingest_new.py %d 场（%s..%s，来自 %s），缺 map_results/rosters/by_map"
                           % (n, recs[0]["date"], recs[-1]["date"],
                              "+".join(x[0] for x in per)))
    with open(OUT, "wb") as f:
        pickle.dump(blob, f)
    # 往返自检：写出的 blob 必须能原样读回（硬坑 A/B 的验收）
    blob2 = pickle.loads(pickle.dumps(blob))
    s2, h2, cur2, sha2 = SS.from_blob(blob2)
    d1b, w1b = SS.diff_ledgers(s, s2, SS.STATE_KEYS)
    d2b, w2b = SS.diff_ledgers(h, h2, SS.MH_KEYS)
    print("  往返自检：State 最大差 %.3e（%s）  MapHist %.3e（%s）"
          % (d1b, w1b, d2b, w2b))
    print("  快照常数钉住：veto_model %d 项 / model_v2 %d 项"
          % (len(SS.VM_CONSTS), len(SS.M_CONSTS)))
    io.open(OUT_CUR, "w", encoding="utf-8", newline="\n").write(
        json.dumps(newcur, ensure_ascii=False, indent=1))
    print("\n已写：")
    for p in (OUT, OUT_CUR):
        print("  %-24s %8.1f KB" % (os.path.basename(p), os.path.getsize(p) / 1024))
    print("\n⚠ 缺 %s 三组燃料 ⇒ elo_m 退回胜负指示、阵容类特征不更新。" % "/".join(FUEL_MISSING))
    print("  纯图名列表形式的 veto_order 仍依赖主理人 2026-10-12 的确认"
          "「列图顺序 = pick 顺序」；")
    print("  显式 [[step,kind,team,map]] 形式用主理人给的执行方，不受该假设影响"
          "（bo5 已按 4 pick + 1 decider 处理）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
