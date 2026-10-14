# -*- coding: utf-8 -*-
"""exp_veto_order.py —— 2026-10-11（会话三·续二）：**先手方规则**的实测核查。

主理人 2026-10-11 给出规则：
  * 先 ban 方 = **排名较低**的一方；
  * bo3 顺序：A ban, B ban, A pick, B pick, **B ban, A ban**, 剩余 = decide（A = 排名较高者）。

本脚本只用**语料本身**（csapi veto 顺序 + 当日周排名）核查这两条，输出：
  1. 各 bo 的**行动人型态**（每步由谁执行）；
  2. 第一步行动人 vs「排名较低者」的一致率（对比旧的 team2 口径 63.6%）；
  3. 第 5/6 步是同序（A,B）还是反序（B,A）。

加 `--sim` 再跑第二阶段：把该规则灌进 `veto_model` 的特征（`sim3_in_set`）后，
手方可判定率、两口径选图差异率、以及**单特征命中**（选型前的定量核查）。
"""
import collections
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "../data")  # # 仓库布局：数据与代码分目录（Alpha 1.0）
sys.path.insert(0, HERE)
sys.path.insert(0, HERE)  # scorer.py 与本模块同目录（仓库布局）
os.chdir(HERE)

import pool_experiment as PE  # noqa: E402


def load_jsonl(p):
    return [json.loads(l) for l in io.open(p, encoding="utf-8") if l.strip()]


def main():
    veto = load_jsonl("csapi_veto.jsonl")
    matches = {m["id"]: m for m in load_jsonl("csapi_matches_all.jsonl")}
    ranks = PE.Ranks()

    print("veto 记录 %d（veto_ok=%d）" % (len(veto), sum(1 for r in veto if r.get("veto_ok"))))

    by_bo = collections.Counter()
    patt = collections.defaultdict(collections.Counter)
    first_lower = first_team2 = n_rank = 0
    both_ranked = [0, 0]
    one_unrank = [0, 0]
    pair56 = collections.Counter()
    step_team = collections.defaultdict(collections.Counter)
    step5_lower = [0, 0]

    for r in veto:
        v = r.get("veto") or []
        if not r.get("veto_ok") or not v:
            continue
        m = matches.get(r["id"]) or {}
        d = m.get("date")
        t1 = (m.get("team1") or {}).get("name")
        t2 = (m.get("team2") or {}).get("name")
        bo = r.get("bo") or m.get("best_of")
        by_bo[bo] += 1

        acts = [s[2] for s in v if s[2]]
        if not acts:
            continue
        first_actor = acts[0]
        pattern = "".join("F" if a == first_actor else "S" for a in acts)
        patt[bo][pattern] += 1
        for i, a in enumerate(acts, 1):
            step_team[i]["F" if a == first_actor else "S"] += 1

        lower = None
        if d and t1 and t2:
            r1, p1 = ranks.rank_points(t1, d)      # (名次, 积分)；名次数字越小越强；0 分 = 未上榜
            r2, p2 = ranks.rank_points(t2, d)
            if r1 != r2:
                lower = t1 if r1 > r2 else t2      # 名次数字大 = 排名低
                n_rank += 1
                first_lower += (first_actor == lower)
                first_team2 += (first_actor == t2)
                if p1 > 0 and p2 > 0:
                    both_ranked[0] += 1
                    both_ranked[1] += (first_actor == lower)
                else:
                    one_unrank[0] += 1
                    one_unrank[1] += (first_actor == lower)
                if len(acts) >= 5:
                    step5_lower[0] += 1
                    step5_lower[1] += (acts[4] == lower)

        if len(acts) >= 6:
            if acts[4] == acts[0] and acts[5] == acts[1]:
                pair56["同序 (A,B)"] += 1
            elif acts[4] == acts[1] and acts[5] == acts[0]:
                pair56["反序 (B,A)"] += 1
            else:
                pair56["其它"] += 1

    print("\n各 bo 记录数:", dict(by_bo))
    for bo in sorted(patt, key=lambda x: (x is None, x)):
        tot = sum(patt[bo].values())
        print("\n== bo%s（%d 场）== 行动人型态（F=第 1 步执行者，S=另一方）" % (bo, tot))
        for p, c in patt[bo].most_common(6):
            print("   %-12s %5d  (%4.1f%%)" % (p, c, 100.0 * c / tot))

    print("\n各步由「第 1 步执行者」执行的比例：")
    for i in sorted(step_team):
        c = step_team[i]
        tot = c["F"] + c["S"]
        if tot:
            print("   第 %d 步: F=%4d  S=%4d  → F 占 %4.1f%%" % (i, c["F"], c["S"], 100.0 * c["F"] / tot))

    if n_rank:
        pct = lambda a, b: 100.0 * a / b if b else float("nan")
        print("\n第一步行动人 vs **排名较低者**：%d/%d = %.1f%%（主理人规则）"
              % (first_lower, n_rank, pct(first_lower, n_rank)))
        print("第一步行动人 vs **team2（HLTV 记录序）**：%d/%d = %.1f%%（旧口径 63.6%%）"
              % (first_team2, n_rank, pct(first_team2, n_rank)))
        print("  其中双方都上榜：%d/%d = %.1f%%"
              % (both_ranked[1], both_ranked[0], pct(both_ranked[1], both_ranked[0])))
        print("  其中一方未上榜：%d/%d = %.1f%%"
              % (one_unrank[1], one_unrank[0], pct(one_unrank[1], one_unrank[0])))
        print("第 5 步行动人 = 排名较低者：%d/%d = %.1f%%"
              % (step5_lower[1], step5_lower[0], pct(step5_lower[1], step5_lower[0])))
    print("\n第 5/6 步：", dict(pair56))


def sim_check():
    """规则**进入特征**后的覆盖率与单特征命中（选型前的定量核查）。"""
    import exp_maps as EM
    import veto_model as VM
    data, meta = EM.build_all()
    print()
    print("[--sim] 分桶:", meta["counts"])
    for bucket in ("train", "valid", "test"):
        samples = data[bucket]
        det = 0
        slots = hit2 = hit3 = 0
        diff = 0
        for s in samples:
            first = VM.first_banner(s["teamA"], s["teamB"], s["date"])
            if first is not None:
                det += 1
            k = max(1, min(int(s["bo"] or 3), 5))
            truth = set(f["map"] for f, v in zip(s["rows"], s["y"]) if v)
            top2 = sorted(s["rows"], key=lambda f: (-f["sim2_in_set"], f["map"]))[:k]
            top3 = sorted(s["rows"], key=lambda f: (-f["sim3_in_set"], f["map"]))[:k]
            slots += k
            hit2 += sum(1 for f in top2 if f["map"] in truth)
            hit3 += sum(1 for f in top3 if f["map"] in truth)
            if [f["map"] for f in top2] != [f["map"] for f in top3]:
                diff += 1
        n = len(samples)
        print("  %-5s 场=%d  手方可判定=%d(%.1f%%)  两口径选图不同=%d(%.1f%%)  槽位=%d  "
              "sim2命中=%d(%.1f%%)  sim3命中=%d(%.1f%%)"
              % (bucket, n, det, 100.0 * det / max(1, n), diff, 100.0 * diff / max(1, n),
                 slots, hit2, 100.0 * hit2 / max(1, slots), hit3, 100.0 * hit3 / max(1, slots)))


if __name__ == "__main__":
    main()
    if "--sim" in sys.argv:
        sim_check()
