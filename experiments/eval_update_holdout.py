"""时间切分验证：09-05 快照 vs 摄入新数据后的模型。

主理人 2026-10-12 的立场：**无论更新后变好变坏都必须更新**（数据陈旧本身就是错）。
本脚本不参与「要不要更新」的决策，只**量化幅度**，因此设计成严格的时序切分：

  112 场新数据按日期排序 → 前 70% 摄入 → 后 30% 当作**从未见过**的 holdout
  在**同一批 holdout** 上比较：09-05 快照 vs 摄入后

⚠ 不做「同一批数据入库前后对比」——那是 in-sample，会给出偏乐观的数字
（本会话已因自我验证出过多次错：MVP 标定、比分朝向、bo5 满分）。

评估两个计分项：
  选图：唯一仍有信号、且新数据（veto 顺序）真正能更新的项
  胜方：新数据会更新 team Elo，但**胜方项现已改用 rank** ⇒ 只作对照
"""
import io
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "../data")  # # 仓库布局：数据与代码分目录（Alpha 1.1.0）
sys.path.insert(0, HERE)
import model_v2 as M
import record_schema as RS
import state_snapshot as SS
import veto_model as VM

NEW = os.path.join(DATA, "groundtruth", "new_records_2026-09.jsonl")
HOLD_FRAC = 0.30


def load_new():
    """载入新记录并转成 state_snapshot.ingest() 认识的行。

    ingest() 要的是 csapi 的行格式（winner 是 {"name": ...}），故用
    record_schema.to_ingest_row 转换。map_results 为空（无逐图比分），
    State.observe 会优雅降级（sA_margin 退回胜负指示）。
    """
    recs = [json.loads(l) for l in io.open(NEW, encoding="utf-8") if l.strip()]
    recs.sort(key=lambda r: (r["date"], r["id"]))
    return recs


def as_row(r):
    return RS.to_ingest_row(r)


def build_state(rows, ctx, cursor=None):
    s, h = M.State(), VM.MapHistory()
    n, cur = SS.ingest(s, h, rows, ctx, cursor=cursor)
    return s, h, n, cur


def maps_hit(h, mmodel, rec, pool):
    """该场选图命中张数。

    走**实时预测的同一条路径**：`MapHistory.features_for()` 出特征，
    `VM.predict_set()` 用 mmodel 权重挑图。mmodel 本身离线冻结、不随摄入重训，
    变的是喂给它的 h（逐日推进 + 指数衰减）⇒ 这才是真实受新数据影响的部分。
    """
    A, B = rec["team1"]["name"], rec["team2"]["name"]
    bo = rec["best_of"]
    date = rec["date"]
    h.advance(date)
    rows = h.features_for(A, B, pool, bo, date=date)
    picked, _ = VM.predict_set(mmodel, {"bo": bo, "rows": rows}, bo)
    truth = set(rec["maps_full_B"])
    return sum(1 for m in picked if m in truth), len(picked)


def rank_pick(rec, model_winner):
    import predict as PR
    p = PR.Predictor.__new__(PR.Predictor)
    p._rank_idx, p._rank_snap = None, ""
    win, info = p.rank_pick({"teamA": rec["team1"]["name"],
                             "teamB": rec["team2"]["name"]}, model_winner)
    return win


def main():
    recs = load_new()
    k = int(len(recs) * (1 - HOLD_FRAC))
    tr, ho = recs[:k], recs[k:]
    print("新数据 %d 场  %s .. %s" % (len(recs), recs[0]["date"], recs[-1]["date"]))
    print("  摄入段 %d 场（%s .. %s）" % (len(tr), tr[0]["date"], tr[-1]["date"]))
    print("  留出段 %d 场（%s .. %s）  ← 两版模型都**没见过**\n"
          % (len(ho), ho[0]["date"], ho[-1]["date"]))

    prep = M.prepare(verbose=False)
    ctx = SS.make_ctx(prep["truths"], prep["veto"], M.load_stats(),
                      M.load_stats_map(), *_pool_eras(),
                      json.load(io.open(_timeline(), encoding="utf-8")))
    pool = ["Ancient", "Anubis", "Cache", "Dust2", "Inferno", "Mirage", "Nuke"]

    # ---- 版本 A：09-05 快照（现状）----
    hA = prep["mh"]
    mm = prep["mmodel"]          # 权重离线冻结，两版共用；差异来自 h / state

    # ---- 版本 B：只摄入**前 70%**（留出段绝不能进，否则自我验证）----
    import copy
    hB, sB = copy.deepcopy(prep["mh"]), copy.deepcopy(prep["state"])
    rows_tr = [as_row(r) for r in tr]
    nB, _cur = SS.ingest(sB, hB, rows_tr, ctx)
    print("版本 B 摄入 %d 场（**不含留出段**）  h.last_date=%s"
          % (nB, hB.last_date))

    def evaluate(tag, h):
        hit = tot = 0
        for r in ho:
            a, b = maps_hit(h, mm, r, pool)
            hit += a
            tot += b
        acc = hit / max(1, tot)
        se = (acc * (1 - acc) / max(1, tot)) ** 0.5
        print("  %-14s 选图 %3d/%-3d = %.1f%%  (1SE %.1fpp)"
              % (tag, hit, tot, 100 * acc, 100 * se))
        return acc, se

    print("\n== 留出段（%d 场，%s .. %s）上的选图准确率 =="
          % (len(ho), ho[0]["date"], ho[-1]["date"]))
    aA, seA = evaluate("09-05 快照", copy.deepcopy(hA))
    aB, seB = evaluate("摄入前70%后", hB)
    d = aB - aA
    sed = (seA ** 2 + seB ** 2) ** 0.5
    print("\n  差 %+.1fpp  (合并 1SE %.1fpp ⇒ %.1fσ)"
          % (100 * d, 100 * sed, d / sed if sed else 0))
    print("  判定：%s" % ("**改善**" if d > sed else
                        ("**恶化**" if d < -sed else "在噪声内")))
    print("\n  ⇒ 按主理人 2026-10-12 立场，无论方向都更新；本结果只量化幅度。")

    print("\n== 摄入段队伍在两版 state 下的 games 计数（陈旧度）==")
    ts = sorted({t for r in tr for t in (r["team1"]["name"], r["team2"]["name"])})
    sA = prep["state"]
    print("  %-22s %8s %8s" % ("队伍", "快照", "更新后"))
    for t in ts[:8]:
        print("  %-22s %8d %8d" % (t, sA.games.get(t, 0), sB.games.get(t, 0)))
    return 0


def _pool_eras():
    import pool_experiment as PE
    return PE.load_pool_eras()


def _timeline():
    import pool_experiment as PE
    return PE.TIMELINE


if __name__ == "__main__":
    sys.exit(main())
