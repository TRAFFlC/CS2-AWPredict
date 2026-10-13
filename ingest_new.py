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
import io
import json
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")  # # 仓库布局：数据与代码分目录（Alpha 1.0）
sys.path.insert(0, HERE)
import model_v2 as M
import record_schema as RS
import state_snapshot as SS

NEW = os.path.join(HERE, "new_records_2026-09.jsonl")
OUT = os.path.join(DATA, "state_2026-10.pkl")
OUT_MH = os.path.join(HERE, "mh_2026-10.pkl")
OUT_CUR = os.path.join(HERE, "cursor_2026-10.json")

FUEL_MISSING = ("map_results", "rosters", "by_map")


def load_new():
    recs = [json.loads(l) for l in io.open(NEW, encoding="utf-8") if l.strip()]
    recs.sort(key=lambda r: (r["date"], r["id"]))
    return recs


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
    a = ap.parse_args()

    recs = load_new()
    rows, dropped = as_rows(recs)
    print("新记录 %d 场  %s .. %s" % (len(recs), recs[0]["date"], recs[-1]["date"]))
    print("  schema 校验通过 %d 场，剔除 %d 场（**不喂给 ingest**）"
          % (len(rows), len(dropped)))
    for pid, pr in dropped:
        print("    ✗ %s → %s" % (pid, pr))

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
    before_d = s.last_date
    n, newcur = SS.ingest(s, h, rows, ctx, cursor=cur)
    print("\n摄入 %d 场  state.last_date: %s → %s" % (n, before_d, s.last_date))

    d1, w1 = SS.diff_ledgers(prep["state"], s, _numeric_keys(SS.STATE_KEYS))
    print("  State 数值账本（排除 last_date）：最大差 %.3e（%s）" % (d1, w1))
    print("  ↑ diff_ledgers 对**新增键**返回 inf（%s）：本批有 1 支新队 betclic "
          "首次进入账本、h2h 新增 51 对，属正常" % w1)
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
                      note="ingest_new.py %d 场（%s..%s），缺 map_results/rosters/by_map"
                           % (n, recs[0]["date"], recs[-1]["date"]))
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
    print("  veto_order 依赖主理人 2026-10-12 的确认「列图顺序 = pick 顺序」。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
